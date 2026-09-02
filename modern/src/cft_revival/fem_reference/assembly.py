"""P2 Galerkin assembly for the axisymmetric flux-function weak form."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from time import perf_counter

import numpy as np

from .models import CSRMatrix, FEMProblem, FEMValidationError, P2Mesh

# Dunavant degree-five rule. Weights sum to one and are multiplied by physical area.
_QUADRATURE = (
    ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.225),
    ((0.059715871789770, 0.470142064105115, 0.470142064105115), 0.132394152788506),
    ((0.470142064105115, 0.059715871789770, 0.470142064105115), 0.132394152788506),
    ((0.470142064105115, 0.470142064105115, 0.059715871789770), 0.132394152788506),
    ((0.797426985353087, 0.101286507323456, 0.101286507323456), 0.125939180544827),
    ((0.101286507323456, 0.797426985353087, 0.101286507323456), 0.125939180544827),
    ((0.101286507323456, 0.101286507323456, 0.797426985353087), 0.125939180544827),
)
_LINE_POINTS, _LINE_WEIGHTS = np.polynomial.legendre.leggauss(3)


@dataclass(frozen=True, slots=True)
class AssembledSystem:
    matrix: CSRMatrix
    rhs: np.ndarray
    free_dofs: np.ndarray
    prescribed_dofs: np.ndarray
    prescribed_values: np.ndarray
    physical_load: np.ndarray
    assembly_seconds: float


def p2_shape(
    barycentric: np.ndarray, grad_lambda: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    l0, l1, l2 = barycentric
    values = np.asarray(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ),
        dtype=np.float64,
    )
    gradients = np.vstack(
        (
            (4.0 * l0 - 1.0) * grad_lambda[0],
            (4.0 * l1 - 1.0) * grad_lambda[1],
            (4.0 * l2 - 1.0) * grad_lambda[2],
            4.0 * (l0 * grad_lambda[1] + l1 * grad_lambda[0]),
            4.0 * (l1 * grad_lambda[2] + l2 * grad_lambda[1]),
            4.0 * (l2 * grad_lambda[0] + l0 * grad_lambda[2]),
        )
    )
    return values, gradients


def _fixed_dofs(problem: FEMProblem, mesh: P2Mesh) -> tuple[np.ndarray, np.ndarray]:
    fixed: set[int] = set()
    if problem.domain.r_min_m == 0.0:
        edge_indices = mesh.boundary_edges["axis"]
    else:
        edge_indices = mesh.boundary_edges["inner_radial"]
    for edge_index in edge_indices:
        fixed.update(int(value) for value in mesh.edges[edge_index])
        fixed.add(int(mesh.edge_midpoint_dofs[edge_index]))
    if problem.outer_boundary == "dirichlet":
        for name in ("outer_radial", "z_min", "z_max"):
            for edge_index in mesh.boundary_edges[name]:
                fixed.update(int(value) for value in mesh.edges[edge_index])
                fixed.add(int(mesh.edge_midpoint_dofs[edge_index]))
    prescribed = np.asarray(sorted(fixed), dtype=np.int64)
    values = np.zeros(len(prescribed), dtype=np.float64)
    if problem.dirichlet_a_phi is not None:
        for index, dof in enumerate(prescribed):
            r_m, z_m = mesh.p2_nodes_rz_m[dof]
            values[index] = (
                0.0
                if r_m == 0.0
                else problem.dirichlet_a_phi(float(r_m), float(z_m))
            )
    return prescribed, values


def _triangle_geometry(points: np.ndarray) -> tuple[float, np.ndarray]:
    jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0.0:
        raise FEMValidationError("triangle orientation must be positive")
    grad_lambda = np.empty((3, 2), dtype=np.float64)
    inverse_transpose = np.linalg.inv(jacobian).T
    grad_lambda[1] = inverse_transpose[:, 0]
    grad_lambda[2] = inverse_transpose[:, 1]
    grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
    return 0.5 * determinant, grad_lambda


def _dipole_alpha(boundary: str, r_m: float, z_m: float, center_z_m: float) -> float:
    rho2 = r_m * r_m + (z_m - center_z_m) ** 2
    if boundary == "outer_radial":
        return 3.0 * r_m / rho2 - 2.0 / r_m
    return 3.0 * abs(z_m - center_z_m) / rho2


def _edge_line_basis(parameter: float) -> np.ndarray:
    left = 0.5 * (1.0 - parameter)
    right = 0.5 * (1.0 + parameter)
    return np.asarray(
        (left * (2.0 * left - 1.0), right * (2.0 * right - 1.0), 4.0 * left * right)
    )


def _edge_dofs(mesh: P2Mesh, edge_index: int) -> np.ndarray:
    return np.asarray(
        (
            mesh.edges[edge_index, 0],
            mesh.edges[edge_index, 1],
            mesh.edge_midpoint_dofs[edge_index],
        ),
        dtype=np.int64,
    )


def _matches_sheet(points: np.ndarray, sheet) -> bool:
    tolerance = 5.0e-12 * max(
        abs(sheet.coordinate_m), abs(sheet.span_min_m), abs(sheet.span_max_m), 1.0
    )
    if sheet.orientation == "constant_r":
        fixed = points[:, 0]
        span = points[:, 1]
    else:
        fixed = points[:, 1]
        span = points[:, 0]
    return bool(
        np.all(np.abs(fixed - sheet.coordinate_m) <= tolerance)
        and np.min(span) >= sheet.span_min_m - tolerance
        and np.max(span) <= sheet.span_max_m + tolerance
    )


def assemble(
    problem: FEMProblem,
    mesh: P2Mesh,
    *,
    required_available_ram_bytes: int = 0,
) -> AssembledSystem:
    """Assemble the sparse symmetric weak system in a regular P2 ``A_phi`` space.

    The represented flux test functions are ``v_psi = r N_i``. This is exactly
    the weak form in ``psi=r A_phi`` while enforcing ``psi=O(r^2)`` on the axis.
    """

    if (
        isinstance(required_available_ram_bytes, bool)
        or not isinstance(required_available_ram_bytes, int)
        or required_available_ram_bytes < 0
    ):
        raise FEMValidationError("required available RAM must be a non-negative integer")
    from .resource_policy import available_ram_bytes, guard_allocation

    guard_allocation(
        "assembly",
        p2_dofs=len(mesh.p2_nodes_rz_m),
        triangles=len(mesh.triangles),
        robin_edges=sum(
            len(mesh.boundary_edges[name])
            for name in ("outer_radial", "z_min", "z_max")
        ),
        third_level=False,
    )
    if required_available_ram_bytes:
        available = available_ram_bytes()
        if available < required_available_ram_bytes:
            from .resource_policy import ResourceBlockedError

            raise ResourceBlockedError(
                "NOT_EVALUATED: assembly RAM recheck failed: requires "
                f"{required_available_ram_bytes} free bytes, found {available}"
            )
    started = perf_counter()
    regions = problem.regions_by_id
    prescribed, prescribed_values = _fixed_dofs(problem, mesh)
    node_count = len(mesh.p2_nodes_rz_m)
    is_prescribed = np.zeros(node_count, dtype=bool)
    is_prescribed[prescribed] = True
    free = np.flatnonzero(~is_prescribed).astype(np.int64)
    free_map = np.full(node_count, -1, dtype=np.int64)
    free_map[free] = np.arange(len(free), dtype=np.int64)
    prescribed_by_dof = np.zeros(node_count, dtype=np.float64)
    prescribed_by_dof[prescribed] = prescribed_values
    rhs = np.zeros(len(free), dtype=np.float64)
    physical_load = np.zeros(node_count, dtype=np.float64)

    robin_edges: list[tuple[str, int]] = []
    if problem.outer_boundary == "dipole_robin":
        robin_edges = [
            (boundary, int(edge_index))
            for boundary in ("outer_radial", "z_min", "z_max")
            for edge_index in mesh.boundary_edges[boundary]
        ]

    # Pass one builds a preallocated topology-only COO key stream, then
    # compresses it to the final CSR sparsity pattern. No Python row
    # dictionaries or floating-point COO copies are retained.
    maximum_contributions = 36 * len(mesh.triangles) + 9 * len(robin_edges)
    coo_keys = np.empty(maximum_contributions, dtype=np.int64)
    cursor = 0
    for dofs in mesh.element_dofs:
        mapped = free_map[dofs]
        free_rows = mapped[mapped >= 0]
        count = len(free_rows) ** 2
        coo_keys[cursor : cursor + count] = (
            np.repeat(free_rows, len(free_rows)) * len(free)
            + np.tile(free_rows, len(free_rows))
        )
        cursor += count
    for _, edge_index in robin_edges:
        dofs = _edge_dofs(mesh, edge_index)
        mapped = free_map[dofs]
        free_rows = mapped[mapped >= 0]
        count = len(free_rows) ** 2
        coo_keys[cursor : cursor + count] = (
            np.repeat(free_rows, len(free_rows)) * len(free)
            + np.tile(free_rows, len(free_rows))
        )
        cursor += count
    sorted_keys = coo_keys[:cursor]
    sorted_keys.sort(kind="stable")
    unique_mask = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    unique_keys = sorted_keys[unique_mask]
    unique_rows = unique_keys // len(free)
    unique_columns = unique_keys % len(free)
    row_counts = np.bincount(unique_rows, minlength=len(free))
    indptr = np.empty(len(free) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    matrix = CSRMatrix(
        indptr,
        unique_columns.astype(np.int64, copy=False),
        np.zeros(len(unique_columns), dtype=np.float64),
        (len(free), len(free)),
    )
    del coo_keys, sorted_keys, unique_mask, unique_keys, unique_rows, row_counts

    def add_matrix(dofs: np.ndarray, local_matrix: np.ndarray) -> None:
        mapped = free_map[dofs]
        free_local = np.flatnonzero(mapped >= 0)
        fixed_local = np.flatnonzero(mapped < 0)
        if len(free_local):
            free_rows = mapped[free_local]
            if len(fixed_local):
                rhs[free_rows] -= (
                    local_matrix[np.ix_(free_local, fixed_local)]
                    @ prescribed_by_dof[dofs[fixed_local]]
                )
            for local_row, row in zip(free_local, free_rows):
                start, stop = matrix.indptr[row : row + 2]
                columns = matrix.indices[start:stop]
                positions = start + np.searchsorted(columns, free_rows)
                matrix.data[positions] += local_matrix[local_row, free_local]

    # Pass two evaluates element and boundary actions and fills the preallocated
    # COO arrays in element/local-row order.
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        area, grad_lambda = _triangle_geometry(points)
        dofs = mesh.element_dofs[element]
        region = regions[mesh.triangle_region_ids[element]]
        local_matrix = np.zeros((6, 6), dtype=np.float64)
        local_load = np.zeros(6, dtype=np.float64)
        for barycentric_tuple, weight in _QUADRATURE:
            barycentric = np.asarray(barycentric_tuple)
            r_m, z_m = barycentric @ points
            values, gradients = p2_shape(barycentric, grad_lambda)
            psi_gradients = np.column_stack(
                (
                    values + r_m * gradients[:, 0],
                    r_m * gradients[:, 1],
                )
            )
            measure = area * weight
            local_matrix += (
                region.reluctivity_per_m_h
                * measure
                / r_m
                * (psi_gradients @ psi_gradients.T)
            )
            current = float(problem.free_current_phi(float(r_m), float(z_m)))
            if not np.isfinite(current):
                raise FEMValidationError("free-current source returned a nonfinite value")
            local_load += measure * (
                current * r_m * values
                + region.reluctivity_per_m_h
                * (
                    region.remanence_z_t * psi_gradients[:, 0]
                    - region.remanence_r_t * psi_gradients[:, 1]
                )
            )
        physical_load[dofs] += local_load
        mapped = free_map[dofs]
        free_local = np.flatnonzero(mapped >= 0)
        rhs[mapped[free_local]] += local_load[free_local]
        add_matrix(dofs, local_matrix)

    edge_to_elements: dict[tuple[int, int], list[int]] = {}
    for element, triangle in enumerate(mesh.triangles):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_to_elements.setdefault(tuple(sorted((int(first), int(second)))), []).append(
                element
            )

    for boundary, edge_index in robin_edges:
        edge = mesh.edges[edge_index]
        points = mesh.vertices_rz_m[edge]
        length = hypot(*(points[1] - points[0]))
        adjacent = edge_to_elements[tuple(edge)][0]
        nu = regions[mesh.triangle_region_ids[adjacent]].reluctivity_per_m_h
        local = np.zeros((3, 3), dtype=np.float64)
        for parameter, weight in zip(_LINE_POINTS, _LINE_WEIGHTS):
            shape = _edge_line_basis(float(parameter))
            point = 0.5 * (
                (1.0 - parameter) * points[0] + (1.0 + parameter) * points[1]
            )
            alpha = _dipole_alpha(
                boundary, float(point[0]), float(point[1]), problem.source_center_z_m
            )
            local += (
                0.5
                * length
                * weight
                * nu
                * alpha
                * point[0]
                * np.outer(shape, shape)
            )
        add_matrix(_edge_dofs(mesh, edge_index), local)

    for sheet in problem.sheets:
        matched_length = 0.0
        for edge_index, edge in enumerate(mesh.edges):
            points = mesh.vertices_rz_m[edge]
            if not _matches_sheet(points, sheet):
                continue
            length = hypot(*(points[1] - points[0]))
            matched_length += length
            dofs = _edge_dofs(mesh, edge_index)
            local = np.zeros(3, dtype=np.float64)
            for parameter, weight in zip(_LINE_POINTS, _LINE_WEIGHTS):
                shape = _edge_line_basis(float(parameter))
                point = 0.5 * (
                    (1.0 - parameter) * points[0] + (1.0 + parameter) * points[1]
                )
                local += (
                    0.5
                    * length
                    * weight
                    * sheet.k_phi_a_per_m
                    * point[0]
                    * shape
                )
            physical_load[dofs] += local
            mapped = free_map[dofs]
            free_local = np.flatnonzero(mapped >= 0)
            rhs[mapped[free_local]] += local[free_local]
        expected = sheet.span_max_m - sheet.span_min_m
        if abs(matched_length - expected) > 2.0e-10 * max(expected, 1.0):
            raise FEMValidationError(
                f"sheet {sheet.source_id!r} is not represented by conforming mesh edges"
            )

    nonzero = matrix.data != 0.0
    if not np.all(nonzero):
        rows = np.repeat(np.arange(len(free), dtype=np.int64), np.diff(matrix.indptr))
        rows = rows[nonzero]
        columns = matrix.indices[nonzero]
        data = matrix.data[nonzero]
        row_counts = np.bincount(rows, minlength=len(free))
        indptr = np.empty(len(free) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        matrix = CSRMatrix(indptr, columns, data, matrix.shape)
    rows = np.repeat(np.arange(len(free), dtype=np.int64), np.diff(matrix.indptr))
    diagonal_rows = rows[rows == matrix.indices]
    if not np.array_equal(diagonal_rows, np.arange(len(free), dtype=np.int64)):
        raise FEMValidationError("assembled sparse row has no diagonal")
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise FEMValidationError("assembled system contains nonfinite values")
    return AssembledSystem(
        matrix,
        rhs,
        free,
        prescribed,
        prescribed_values,
        physical_load,
        perf_counter() - started,
    )
