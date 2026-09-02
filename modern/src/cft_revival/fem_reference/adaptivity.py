"""Residual, flux-jump, and QoI-focused adaptive marking."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .assembly import _LINE_POINTS, _LINE_WEIGHTS, p2_shape
from .mesh import element_diameters
from .models import FEMProblem, FEMResult, FEMValidationError


@dataclass(frozen=True, slots=True)
class IndicatorReport:
    residual_squared: np.ndarray
    flux_jump_squared: np.ndarray
    qoi_proxy_squared: np.ndarray
    total_squared: np.ndarray


def edge_flux_jump_term(edge_length: float, jump_values: np.ndarray) -> float:
    """Return ``h_e * integral_e jump^2 ds`` for three Gauss traces."""

    jumps = np.asarray(jump_values, dtype=np.float64)
    if (
        not isfinite(edge_length)
        or edge_length <= 0.0
        or jumps.shape != _LINE_WEIGHTS.shape
        or not np.isfinite(jumps).all()
    ):
        raise FEMValidationError("edge jump quadrature inputs are invalid")
    integral = (
        0.5
        * edge_length
        * float(np.dot(_LINE_WEIGHTS, jumps * jumps))
    )
    return edge_length * integral


def _geometry(points: np.ndarray) -> tuple[float, np.ndarray]:
    jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
    determinant = float(np.linalg.det(jacobian))
    inverse_transpose = np.linalg.inv(jacobian).T
    gradients = np.empty((3, 2))
    gradients[1] = inverse_transpose[:, 0]
    gradients[2] = inverse_transpose[:, 1]
    gradients[0] = -gradients[1] - gradients[2]
    return 0.5 * determinant, gradients


def _shape_hessians(grad_lambda: np.ndarray) -> np.ndarray:
    g0, g1, g2 = grad_lambda
    return np.asarray(
        (
            4.0 * np.outer(g0, g0),
            4.0 * np.outer(g1, g1),
            4.0 * np.outer(g2, g2),
            4.0 * (np.outer(g0, g1) + np.outer(g1, g0)),
            4.0 * (np.outer(g1, g2) + np.outer(g2, g1)),
            4.0 * (np.outer(g2, g0) + np.outer(g0, g2)),
        )
    )


def _flux(result: FEMResult, element: int, barycentric: np.ndarray) -> np.ndarray:
    mesh = result.mesh
    points = mesh.vertices_rz_m[mesh.triangles[element]]
    _, grad_lambda = _geometry(points)
    values, gradients = p2_shape(barycentric, grad_lambda)
    coefficients = result.a_phi_dofs_t_m[mesh.element_dofs[element]]
    r_m = float(barycentric @ points[:, 0])
    psi_gradient = np.asarray(
        (
            float(np.dot(coefficients, values + r_m * gradients[:, 0])),
            float(np.dot(coefficients, r_m * gradients[:, 1])),
        )
    )
    region = result.problem.regions_by_id[mesh.triangle_region_ids[element]]
    remanence_flux = np.asarray(
        (region.reluctivity_per_m_h * region.remanence_z_t,
         -region.reluctivity_per_m_h * region.remanence_r_t)
    )
    return region.reluctivity_per_m_h * psi_gradient / r_m - remanence_flux


def estimate_indicators(
    result: FEMResult,
    qoi_windows: tuple[tuple[str, float, float, float], ...] = (),
) -> IndicatorReport:
    """Compute a standard residual/jump estimator plus a local QoI proxy."""

    mesh = result.mesh
    problem = result.problem
    diameters = element_diameters(mesh)
    residual = np.zeros(len(mesh.triangles))
    jump = np.zeros(len(mesh.triangles))
    areas = np.zeros(len(mesh.triangles))
    centroids = np.zeros((len(mesh.triangles), 2))
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        area, grad_lambda = _geometry(points)
        areas[element] = area
        centroid = np.mean(points, axis=0)
        centroids[element] = centroid
        barycentric = np.full(3, 1.0 / 3.0)
        values, gradients = p2_shape(barycentric, grad_lambda)
        hessians = _shape_hessians(grad_lambda)
        coefficients = result.a_phi_dofs_t_m[mesh.element_dofs[element]]
        r_m, z_m = map(float, centroid)
        n_value = float(np.dot(coefficients, values))
        n_gradient = coefficients @ gradients
        n_hessian = np.tensordot(coefficients, hessians, axes=(0, 0))
        psi_r = n_value + r_m * n_gradient[0]
        psi_laplacian = (
            2.0 * n_gradient[0]
            + r_m * n_hessian[0, 0]
            + r_m * n_hessian[1, 1]
        )
        region = problem.regions_by_id[mesh.triangle_region_ids[element]]
        divergence = region.reluctivity_per_m_h * (
            psi_laplacian / r_m - psi_r / (r_m * r_m)
        )
        strong = float(problem.free_current_phi(r_m, z_m)) + divergence
        residual[element] = diameters[element] ** 2 * area * strong * strong

    edge_owners: dict[tuple[int, int], list[int]] = {}
    for element, triangle in enumerate(mesh.triangles):
        v0, v1, v2 = map(int, triangle)
        for edge in ((v0, v1), (v1, v2), (v2, v0)):
            edge_owners.setdefault(tuple(sorted(edge)), []).append(element)
    for edge, owners in edge_owners.items():
        if len(owners) != 2:
            continue
        points = mesh.vertices_rz_m[np.asarray(edge)]
        tangent = points[1] - points[0]
        length = float(np.linalg.norm(tangent))
        normal = np.asarray((tangent[1], -tangent[0])) / length
        jump_values = np.empty(len(_LINE_POINTS), dtype=np.float64)
        for point_index, parameter in enumerate(_LINE_POINTS):
            point = 0.5 * (
                (1.0 - parameter) * points[0] + (1.0 + parameter) * points[1]
            )
            traces = []
            for element in owners:
                element_points = mesh.vertices_rz_m[mesh.triangles[element]]
                matrix = np.column_stack(
                    (
                        element_points[1] - element_points[0],
                        element_points[2] - element_points[0],
                    )
                )
                local = np.linalg.solve(matrix, point - element_points[0])
                barycentric = np.asarray((1.0 - local.sum(), local[0], local[1]))
                traces.append(
                    float(np.dot(_flux(result, element, barycentric), normal))
                )
            jump_values[point_index] = traces[0] - traces[1]
        # Standard edge term h_e * integral_e [[q.n]]^2 ds. With h_e equal
        # to this straight edge's length, a constant jump scales as length^2.
        contribution = edge_flux_jump_term(length, jump_values)
        for element in owners:
            jump[element] += 0.5 * contribution

    base = residual + jump
    focus_values = np.zeros_like(base)
    for element, (r_m, z_m) in enumerate(centroids):
        for _, radius, z_min, z_max in qoi_windows:
            radial_distance = max(0.0, r_m - radius)
            axial_distance = max(z_min - z_m, 0.0, z_m - z_max)
            distance = (radial_distance**2 + axial_distance**2) ** 0.5
            local_h = max(diameters[element], 1.0e-300)
            focus = np.exp(-distance / (2.0 * local_h))
            if r_m <= radius and z_min <= z_m <= z_max:
                focus += 1.0
            focus_values[element] += focus
    qoi = 2.0 * focus_values * base
    focus_sum = float(np.sum(focus_values))
    if focus_sum > 0.0:
        # Give the QoI support a finite share even when interface jumps dominate
        # by several orders of magnitude.
        qoi += float(np.sum(base)) * focus_values / focus_sum
    total = base + qoi
    if not np.isfinite(total).all() or np.any(total < 0.0):
        raise FEMValidationError("adaptive estimator produced invalid indicators")
    return IndicatorReport(residual, jump, qoi, total)


def dorfler_mark(indicators_squared: np.ndarray, theta: float = 0.5) -> np.ndarray:
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not isfinite(theta):
        raise FEMValidationError("Dorfler theta must be finite")
    if not 0.0 < theta <= 1.0:
        raise FEMValidationError("Dorfler theta must lie in (0,1]")
    values = np.asarray(indicators_squared, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise FEMValidationError("Dorfler indicators must be finite and non-negative")
    total = float(np.sum(values))
    if total == 0.0:
        return np.empty(0, dtype=np.int64)
    ordering = sorted(range(len(values)), key=lambda index: (-values[index], index))
    marked = []
    accumulated = 0.0
    for index in ordering:
        marked.append(index)
        accumulated += float(values[index])
        if accumulated >= theta * total:
            break
    return np.asarray(sorted(marked), dtype=np.int64)


def component_dorfler_mark(
    report: IndicatorReport, theta: float = 0.5
) -> np.ndarray:
    """Take the deterministic union of residual, jump, and QoI bulk sets."""

    component_sets = (
        dorfler_mark(report.residual_squared, theta),
        dorfler_mark(report.flux_jump_squared, theta),
        dorfler_mark(report.qoi_proxy_squared, theta),
    )
    nonempty = [values for values in component_sets if len(values)]
    if not nonempty:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(nonempty))
