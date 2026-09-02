"""Deterministic native-CSR solve and P2 field evaluation."""

from __future__ import annotations

from hashlib import sha256
from math import isfinite, pi, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np

from .assembly import AssembledSystem, assemble, p2_shape
from .models import (
    FEMConvergenceError,
    FEMProblem,
    FEMResult,
    FEMValidationError,
    P2Mesh,
    SolverDiagnostics,
    canonical_bytes,
)


def _matrix_rows(system: AssembledSystem) -> list[dict[int, float]]:
    matrix = system.matrix
    return [
        {
            int(column): float(value)
            for column, value in zip(
                matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]],
                matrix.data[matrix.indptr[row] : matrix.indptr[row + 1]],
            )
        }
        for row in range(matrix.shape[0])
    ]


class _IC0:
    def __init__(self, system: AssembledSystem) -> None:
        matrix_rows = _matrix_rows(system)
        self.lower: list[dict[int, float]] = []
        self.diagonal = np.empty(len(matrix_rows), dtype=np.float64)
        self.columns: list[list[tuple[int, float]]] = [[] for _ in matrix_rows]
        for row_index, matrix_row in enumerate(matrix_rows):
            lower_row: dict[int, float] = {}
            for column in sorted(key for key in matrix_row if key < row_index):
                shared = set(lower_row).intersection(self.lower[column])
                correction = sum(
                    lower_row[key] * self.lower[column][key]
                    for key in shared
                    if key < column
                )
                lower_row[column] = (
                    matrix_row[column] - correction
                ) / self.diagonal[column]
            raw_diagonal = matrix_row[row_index] - sum(
                value * value for value in lower_row.values()
            )
            floor = max(abs(matrix_row[row_index]) * 1.0e-14, 1.0e-300)
            self.diagonal[row_index] = sqrt(max(raw_diagonal, floor))
            self.lower.append(lower_row)
            for column, value in lower_row.items():
                self.columns[column].append((row_index, value))

    def apply(self, residual: np.ndarray) -> np.ndarray:
        forward = np.empty_like(residual)
        for row, entries in enumerate(self.lower):
            forward[row] = (
                residual[row]
                - sum(value * forward[column] for column, value in entries.items())
            ) / self.diagonal[row]
        output = np.empty_like(residual)
        for row in range(len(residual) - 1, -1, -1):
            output[row] = (
                forward[row]
                - sum(value * output[target] for target, value in self.columns[row])
            ) / self.diagonal[row]
        return output


class _DiagonalPreconditioner:
    def __init__(self, system: AssembledSystem) -> None:
        matrix = system.matrix
        self.diagonal = np.empty(matrix.shape[0], dtype=np.float64)
        for row in range(matrix.shape[0]):
            start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
            columns = matrix.indices[start:stop]
            positions = np.flatnonzero(columns == row)
            if len(positions) != 1:
                raise FEMValidationError("sparse matrix row must contain one diagonal")
            self.diagonal[row] = matrix.data[start + int(positions[0])]
        if np.any(self.diagonal <= 0.0) or not np.isfinite(self.diagonal).all():
            raise FEMValidationError("sparse matrix diagonal must be finite and positive")

    def apply(self, residual: np.ndarray) -> np.ndarray:
        return residual / self.diagonal


def _pcg(
    system: AssembledSystem,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    max_iterations: int,
    initial_solution: np.ndarray | None = None,
) -> tuple[np.ndarray, int, tuple[float, ...], float, float]:
    matrix, rhs = system.matrix, system.rhs
    solution = (
        np.zeros_like(rhs)
        if initial_solution is None
        else np.asarray(initial_solution, dtype=np.float64).copy()
    )
    if solution.shape != rhs.shape or not np.isfinite(solution).all():
        raise FEMValidationError("PCG initial solution must be finite and match free DOFs")
    residual = rhs - matrix.matvec(solution)
    initial = float(np.linalg.norm(residual))
    if initial == 0.0:
        return solution, 0, (0.0,), 0.0, 0.0
    reference = float(np.linalg.norm(rhs))
    threshold = max(absolute_tolerance, relative_tolerance * reference)
    preconditioner = (
        _IC0(system)
        if matrix.shape[0] <= 15000
        else _DiagonalPreconditioner(system)
    )
    z = preconditioner.apply(residual)
    direction = z.copy()
    rho = float(np.dot(residual, z))
    history = [initial]
    for iteration in range(1, max_iterations + 1):
        applied = matrix.matvec(direction)
        denominator = float(np.dot(direction, applied))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise FEMConvergenceError("PCG detected a non-positive search curvature")
        alpha = rho / denominator
        solution += alpha * direction
        residual -= alpha * applied
        if iteration % 20 == 0:
            residual = rhs - matrix.matvec(solution)
        norm = float(np.linalg.norm(residual))
        if iteration % 10 == 0:
            history.append(norm)
        if not np.isfinite(norm) or not np.isfinite(solution).all():
            raise FEMConvergenceError("PCG state became nonfinite")
        if norm <= threshold:
            true_residual = rhs - matrix.matvec(solution)
            true_norm = float(np.linalg.norm(true_residual))
            history.append(true_norm)
            if true_norm <= threshold:
                return solution, iteration, tuple(history), initial, true_norm
            residual = true_residual
        z = preconditioner.apply(residual)
        rho_new = float(np.dot(residual, z))
        direction = z + (rho_new / rho) * direction
        rho = rho_new
    true_norm = float(np.linalg.norm(rhs - matrix.matvec(solution)))
    raise FEMConvergenceError(
        f"PCG failed true-residual acceptance: {true_norm:.6e} > {threshold:.6e}"
    )


def _implementation_sha256() -> str:
    digest = sha256()
    root = Path(__file__).parent
    for filename in ("assembly.py", "mesh.py", "models.py", "solver.py"):
        digest.update(filename.encode())
        digest.update((root / filename).read_bytes())
    return digest.hexdigest()


def solve(
    problem: FEMProblem,
    mesh: P2Mesh,
    *,
    relative_tolerance: float = 2.0e-10,
    absolute_tolerance: float = 1.0e-12,
    max_iterations: int = 8000,
    initial_a_phi_dofs_t_m: np.ndarray | None = None,
    required_available_ram_bytes: int = 0,
) -> FEMResult:
    for name, value in (
        ("relative_tolerance", relative_tolerance),
        ("absolute_tolerance", absolute_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise FEMValidationError(f"{name} must be a finite real number")
    if not 0.0 < relative_tolerance < 1.0:
        raise FEMValidationError("relative_tolerance must lie strictly in (0,1)")
    if absolute_tolerance < 0.0:
        raise FEMValidationError("absolute_tolerance must be non-negative")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise FEMValidationError("max_iterations must be a positive integer")
    if not isinstance(mesh, P2Mesh):
        raise FEMValidationError("mesh must be a validated P2Mesh")
    if initial_a_phi_dofs_t_m is not None:
        initial_a_phi_dofs_t_m = np.asarray(
            initial_a_phi_dofs_t_m, dtype=np.float64
        )
        if (
            initial_a_phi_dofs_t_m.shape != (len(mesh.p2_nodes_rz_m),)
            or not np.isfinite(initial_a_phi_dofs_t_m).all()
        ):
            raise FEMValidationError(
                "initial A_phi solution must be finite and match mesh P2 DOFs"
            )
    system = assemble(
        problem,
        mesh,
        required_available_ram_bytes=required_available_ram_bytes,
    )
    started = perf_counter()
    free_solution, iterations, history, initial, final = _pcg(
        system,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        max_iterations=max_iterations,
        initial_solution=(
            None
            if initial_a_phi_dofs_t_m is None
            else initial_a_phi_dofs_t_m[system.free_dofs]
        ),
    )
    solve_seconds = perf_counter() - started
    solution = np.zeros(len(mesh.p2_nodes_rz_m), dtype=np.float64)
    solution[system.free_dofs] = free_solution
    solution[system.prescribed_dofs] = system.prescribed_values
    applied = system.matrix.matvec(free_solution)
    magnetic = pi * float(np.dot(free_solution, applied))
    source = pi * float(np.dot(solution, system.physical_load))
    energy_relative = abs(magnetic - source) / max(abs(magnetic), abs(source), 1.0e-300)
    working_set = int(
        mesh.vertices_rz_m.nbytes
        + mesh.triangles.nbytes
        + mesh.p2_nodes_rz_m.nbytes
        + mesh.element_dofs.nbytes
        + system.matrix.indptr.nbytes
        + system.matrix.indices.nbytes
        + system.matrix.data.nbytes
        + 10 * free_solution.nbytes
    )
    diagnostics = SolverDiagnostics(
        True,
        iterations,
        initial,
        final,
        final / max(float(np.linalg.norm(system.rhs)), 1.0e-300),
        history,
        magnetic,
        source,
        energy_relative,
        system.assembly_seconds,
        solve_seconds,
        working_set,
        (
            "numpy-csr-ic0-pcg"
            if len(system.free_dofs) <= 15000
            else "numpy-csr-jacobi-pcg"
        ),
    )
    implementation_sha256 = _implementation_sha256()
    initial_solution_sha256 = (
        "0" * 64
        if initial_a_phi_dofs_t_m is None
        else sha256(initial_a_phi_dofs_t_m.tobytes()).hexdigest()
    )
    run_payload = {
        "problem_id": problem.problem_id,
        "mesh_sha256": mesh.sha256,
        "geometry_sha256": problem.geometry_sha256,
        "magnetics_sha256": problem.magnetics_sha256,
        "implementation_sha256": implementation_sha256,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "max_iterations": max_iterations,
        "required_available_ram_bytes": required_available_ram_bytes,
        "initial_solution_sha256": initial_solution_sha256,
        "solution_sha256": sha256(solution.tobytes()).hexdigest(),
    }
    return FEMResult(
        problem,
        mesh,
        solution,
        diagnostics,
        sha256(canonical_bytes(run_payload)).hexdigest(),
        (
            ("relative_tolerance", relative_tolerance),
            ("absolute_tolerance", absolute_tolerance),
            ("max_iterations", max_iterations),
            ("required_available_ram_bytes", required_available_ram_bytes),
        ),
        implementation_sha256,
        initial_solution_sha256,
    )


def _barycentric(point: np.ndarray, triangle_points: np.ndarray) -> np.ndarray:
    matrix = np.column_stack(
        (triangle_points[1] - triangle_points[0], triangle_points[2] - triangle_points[0])
    )
    local = np.linalg.solve(matrix, point - triangle_points[0])
    return np.asarray((1.0 - local[0] - local[1], local[0], local[1]))


def locate_element(mesh: P2Mesh, r_m: float, z_m: float) -> tuple[int, np.ndarray]:
    point = np.asarray((r_m, z_m), dtype=np.float64)
    tolerance = 2.0e-11
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        if (
            r_m < np.min(points[:, 0]) - tolerance
            or r_m > np.max(points[:, 0]) + tolerance
            or z_m < np.min(points[:, 1]) - tolerance
            or z_m > np.max(points[:, 1]) + tolerance
        ):
            continue
        barycentric = _barycentric(point, points)
        if np.min(barycentric) >= -tolerance:
            return element, barycentric
    raise FEMValidationError(f"point ({r_m}, {z_m}) lies outside the FEM mesh")


def prolong_p2_solution(coarse: FEMResult, fine_mesh: P2Mesh) -> np.ndarray:
    """Interpolate a direct-parent P2 solution in linear element work."""

    if fine_mesh.parent_mesh_sha256 != coarse.mesh.sha256:
        raise FEMValidationError("fine mesh is not a direct child of the coarse result")
    parents = fine_mesh.element_parent_ids
    if np.any(parents >= len(coarse.mesh.triangles)):
        raise FEMValidationError("fine element parent lies outside coarse mesh")
    prolonged = np.full(len(fine_mesh.p2_nodes_rz_m), np.nan, dtype=np.float64)
    for fine_element, parent in enumerate(parents):
        parent = int(parent)
        triangle_points = coarse.mesh.vertices_rz_m[coarse.mesh.triangles[parent]]
        jacobian = np.column_stack(
            (
                triangle_points[1] - triangle_points[0],
                triangle_points[2] - triangle_points[0],
            )
        )
        fine_dofs = fine_mesh.element_dofs[fine_element]
        local = np.linalg.solve(
            jacobian,
            (
                fine_mesh.p2_nodes_rz_m[fine_dofs] - triangle_points[0]
            ).T,
        ).T
        barycentric = np.column_stack(
            (1.0 - local[:, 0] - local[:, 1], local[:, 0], local[:, 1])
        )
        l0, l1, l2 = barycentric.T
        values = np.column_stack(
            (
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            )
        )
        interpolated = values @ coarse.a_phi_dofs_t_m[
            coarse.mesh.element_dofs[parent]
        ]
        existing = prolonged[fine_dofs]
        assigned = np.isfinite(existing)
        if np.any(assigned) and not np.allclose(
            existing[assigned],
            interpolated[assigned],
            rtol=2.0e-13,
            atol=2.0e-15,
        ):
            raise FEMValidationError("parent P2 traces disagree on a fine DOF")
        prolonged[fine_dofs] = interpolated
    if not np.isfinite(prolonged).all():
        raise FEMValidationError("parent topology did not cover every fine P2 DOF")
    return prolonged


def field_at(result: FEMResult, r_m: float, z_m: float) -> tuple[float, float, float]:
    """Return ``(psi, B_r, B_z)`` from the quadratic ``A_phi`` solution."""

    if abs(r_m) <= 1.0e-14:
        traces: list[float] = []
        tolerance = 2.0e-12
        for candidate, triangle in enumerate(result.mesh.triangles):
            points = result.mesh.vertices_rz_m[triangle]
            axis_vertices = points[np.abs(points[:, 0]) <= tolerance]
            if len(axis_vertices) != 2:
                continue
            if not (
                np.min(axis_vertices[:, 1]) - tolerance
                <= z_m
                <= np.max(axis_vertices[:, 1]) + tolerance
            ):
                continue
            barycentric = _barycentric(np.asarray((0.0, z_m)), points)
            jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
            inverse_transpose = np.linalg.inv(jacobian).T
            grad_lambda = np.empty((3, 2))
            grad_lambda[1] = inverse_transpose[:, 0]
            grad_lambda[2] = inverse_transpose[:, 1]
            grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
            _, gradients = p2_shape(barycentric, grad_lambda)
            coefficients = result.a_phi_dofs_t_m[result.mesh.element_dofs[candidate]]
            traces.append(2.0 * float((coefficients @ gradients)[0]))
        if not traces:
            raise FEMValidationError(f"axis point (0, {z_m}) lies outside the FEM mesh")
        return 0.0, 0.0, float(sum(traces) / len(traces))

    element, barycentric = locate_element(result.mesh, r_m, z_m)
    triangle = result.mesh.triangles[element]
    points = result.mesh.vertices_rz_m[triangle]
    jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
    inverse_transpose = np.linalg.inv(jacobian).T
    grad_lambda = np.empty((3, 2))
    grad_lambda[1] = inverse_transpose[:, 0]
    grad_lambda[2] = inverse_transpose[:, 1]
    grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
    values, gradients = p2_shape(barycentric, grad_lambda)
    coefficients = result.a_phi_dofs_t_m[result.mesh.element_dofs[element]]
    a_phi = float(np.dot(values, coefficients))
    gradient = coefficients @ gradients
    psi = r_m * a_phi
    return psi, -float(gradient[1]), a_phi / r_m + float(gradient[0])


def bore_volume_average(
    result: FEMResult, radius_m: float, z_min_m: float, z_max_m: float
) -> float:
    """Exact axisymmetric volume average via ``integral Bz*r dr = psi(R,z)``."""

    if (
        radius_m <= 0.0
        or z_max_m <= z_min_m
        or radius_m > result.problem.domain.r_max_m
    ):
        raise FEMValidationError("bore volume-average window is invalid")
    points, weights = np.polynomial.legendre.leggauss(5)
    tolerance = 2.0e-11 * max(radius_m, z_max_m - z_min_m, 1.0)
    intervals: list[tuple[float, float, int]] = []
    for element, triangle in enumerate(result.mesh.triangles):
        triangle_points = result.mesh.vertices_rz_m[triangle]
        if (
            radius_m < float(np.min(triangle_points[:, 0])) - tolerance
            or radius_m > float(np.max(triangle_points[:, 0])) + tolerance
        ):
            continue
        crossings: list[float] = []
        for first, second in zip(
            triangle_points, np.roll(triangle_points, -1, axis=0)
        ):
            if abs(float(second[0] - first[0])) <= tolerance:
                if abs(float(first[0] - radius_m)) <= tolerance:
                    crossings.extend((float(first[1]), float(second[1])))
                continue
            fraction = (radius_m - first[0]) / (second[0] - first[0])
            if -tolerance <= fraction <= 1.0 + tolerance:
                crossings.append(
                    float(first[1] + fraction * (second[1] - first[1]))
                )
        if len(crossings) >= 2:
            lower = max(min(crossings), z_min_m)
            upper = min(max(crossings), z_max_m)
            if upper - lower > tolerance:
                intervals.append((lower, upper, element))
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    integral = 0.0
    covered = 0.0
    end_of_coverage = z_min_m
    for lower, upper, element in intervals:
        # Coincident traces on a shared vertical edge are owned by the first
        # deterministic element only.
        lower = max(lower, end_of_coverage)
        if upper - lower <= tolerance:
            continue
        triangle_points = result.mesh.vertices_rz_m[result.mesh.triangles[element]]
        jacobian = np.column_stack(
            (
                triangle_points[1] - triangle_points[0],
                triangle_points[2] - triangle_points[0],
            )
        )
        inverse_transpose = np.linalg.inv(jacobian).T
        grad_lambda = np.empty((3, 2))
        grad_lambda[1] = inverse_transpose[:, 0]
        grad_lambda[2] = inverse_transpose[:, 1]
        grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
        coefficients = result.a_phi_dofs_t_m[result.mesh.element_dofs[element]]
        for point, weight in zip(points, weights):
            z_m = 0.5 * ((upper - lower) * point + upper + lower)
            barycentric = _barycentric(
                np.asarray((radius_m, z_m)), triangle_points
            )
            values, _ = p2_shape(barycentric, grad_lambda)
            psi_value = radius_m * float(np.dot(values, coefficients))
            integral += (
                0.5
                * (upper - lower)
                * weight
                * psi_value
            )
        covered += upper - lower
        end_of_coverage = upper
    expected = z_max_m - z_min_m
    if abs(covered - expected) > 2.0e-10 * max(expected, 1.0):
        raise FEMValidationError("bore trace quadrature did not cover its complete window")
    return 2.0 * integral / (radius_m * radius_m * expected)


def bore_wall_line_average(
    result: FEMResult, radius_m: float, z_min_m: float, z_max_m: float
) -> float:
    """Legacy wall-line metric retained under an unambiguous name."""

    if radius_m <= 0.0 or z_max_m <= z_min_m:
        raise FEMValidationError("bore wall-line window is invalid")
    points, weights = np.polynomial.legendre.leggauss(12)
    center = 0.5 * (z_min_m + z_max_m)
    average = 0.0
    for point, weight in zip(points, weights):
        z_m = center + 0.5 * (z_max_m - z_min_m) * point
        average += 0.5 * weight * field_at(result, radius_m, float(z_m))[2]
    return float(average)


def patch_recovered_axis_bz(result: FEMResult, z_m: float, patch_edges: int = 6) -> float:
    """Weighted quadratic recovery from all adjacent axis-edge midpoint traces."""

    samples: list[tuple[float, float, float]] = []
    tolerance = 2.0e-12
    for element, triangle in enumerate(result.mesh.triangles):
        points = result.mesh.vertices_rz_m[triangle]
        axis_local = np.flatnonzero(np.abs(points[:, 0]) <= tolerance)
        if len(axis_local) != 2:
            continue
        sample_z = float(np.mean(points[axis_local, 1]))
        barycentric = _barycentric(np.asarray((0.0, sample_z)), points)
        jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
        inverse_transpose = np.linalg.inv(jacobian).T
        grad_lambda = np.empty((3, 2))
        grad_lambda[1] = inverse_transpose[:, 0]
        grad_lambda[2] = inverse_transpose[:, 1]
        grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
        _, gradients = p2_shape(barycentric, grad_lambda)
        coefficients = result.a_phi_dofs_t_m[result.mesh.element_dofs[element]]
        value = 2.0 * float((coefficients @ gradients)[0])
        length = abs(float(points[axis_local[1], 1] - points[axis_local[0], 1]))
        samples.append((sample_z, value, length))
    samples.sort(key=lambda item: (abs(item[0] - z_m), item[0]))
    selected = samples[: max(3, patch_edges)]
    if len(selected) < 3:
        raise FEMValidationError("axis patch has insufficient traces for recovery")
    scale = max(max(abs(item[0] - z_m) for item in selected), 1.0e-15)
    design = np.asarray(
        [
            (1.0, (item[0] - z_m) / scale, ((item[0] - z_m) / scale) ** 2)
            for item in selected
        ]
    )
    values = np.asarray([item[1] for item in selected])
    weights = np.sqrt(
        np.asarray([item[2] / (abs(item[0] - z_m) + 0.1 * scale) for item in selected])
    )
    coefficients, *_ = np.linalg.lstsq(
        design * weights[:, None], values * weights, rcond=None
    )
    return float(coefficients[0])


def qois(result: FEMResult, stage_windows: tuple[tuple[str, float, float, float], ...]):
    output: dict[str, float] = {}
    for name, radius, z_min, z_max in stage_windows:
        center = 0.5 * (z_min + z_max)
        output[f"{name}-axis-patch"] = patch_recovered_axis_bz(result, center)
        output[f"{name}-bore-average"] = bore_volume_average(
            result, radius, z_min, z_max
        )
        output[f"{name}-bore-wall-line-average"] = bore_wall_line_average(
            result, radius, z_min, z_max
        )
    return output
