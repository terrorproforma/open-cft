"""Compact verification problems with analytic or representation-equivalent answers."""

from __future__ import annotations

from dataclasses import replace
from math import cos, log, pi, sqrt

import numpy as np

from cft_revival.geometry import (
    PermanentMagnetAuthority,
    PermanentMagnetRepresentationPlan,
    compact_high_gradient_stack,
)
from cft_revival.fields import (
    MU0_H_PER_M,
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    SolverConfig,
    solve_problem_cpu,
)

from .adapters import adapt_geometry
from .adaptivity import dorfler_mark, estimate_indicators
from .assembly import _QUADRATURE, p2_shape
from .mesh import build_body_fitted_mesh, element_diameters, refine_mesh
from .models import Domain, FEMProblem, FEMValidationError, Region
from .solver import field_at, solve


def _sample_relative_error(result, exact_a_phi, *, samples: int = 15) -> float:
    domain = result.problem.domain
    errors = []
    scales = []
    for r_m in np.linspace(domain.r_min_m, domain.r_max_m, samples):
        for z_m in np.linspace(domain.z_min_m, domain.z_max_m, samples):
            if r_m == 0.0:
                continue
            psi = field_at(result, float(r_m), float(z_m))[0]
            actual = psi / r_m
            exact = exact_a_phi(float(r_m), float(z_m))
            errors.append((actual - exact) ** 2)
            scales.append(exact**2)
    return sqrt(sum(errors) / max(sum(scales), 1.0e-300))


def smooth_manufactured_convergence(
    refinements: tuple[int, ...] = (4, 8, 16),
) -> dict[str, object]:
    """Return sampled L2-like P2 rates for a smooth, axis-regular solution."""

    domain = Domain(0.0, 1.0, -1.0, 1.0)
    wave_number = 0.5 * pi

    def exact_a(radial: float, axial: float) -> float:
        return radial * (1.0 - radial) * cos(wave_number * axial)

    def source(radial: float, axial: float) -> float:
        return (
            3.0 + wave_number**2 * radial * (1.0 - radial)
        ) * cos(wave_number * axial)

    l2_errors = []
    energy_errors = []
    h_values = []
    dofs = []
    estimator_norms = []
    estimator_effectivities = []
    qoi_localization_ratios = []
    mesh = None
    previous_refinement = None
    for refinement in refinements:
        problem = FEMProblem(
            "smooth-axis-manufactured",
            domain,
            (Region("uniform", "unit-reluctivity", 1.0),),
            lambda _r, _z: "uniform",
            free_current_phi=source,
            outer_boundary="dirichlet",
            dirichlet_a_phi=exact_a,
        )
        if mesh is None:
            mesh = build_body_fitted_mesh(
                domain,
                (),
                problem.region_at,
                radial_divisions=refinement,
                axial_divisions=2 * refinement,
            )
        else:
            ratio = refinement // int(previous_refinement)
            if refinement != int(previous_refinement) * ratio or ratio < 2:
                raise FEMValidationError("manufactured refinements must be nested integer multiples")
            for _ in range(round(log(ratio, 2.0))):
                mesh = refine_mesh(mesh, domain, reject_below_angle_deg=10.0)
        result = solve(problem, mesh, relative_tolerance=2.0e-11)
        l2_numerator = 0.0
        l2_denominator = 0.0
        energy_numerator = 0.0
        energy_denominator = 0.0
        for element, triangle in enumerate(mesh.triangles):
            element_points = mesh.vertices_rz_m[triangle]
            jacobian = np.column_stack(
                (
                    element_points[1] - element_points[0],
                    element_points[2] - element_points[0],
                )
            )
            area = 0.5 * float(np.linalg.det(jacobian))
            inverse_transpose = np.linalg.inv(jacobian).T
            grad_lambda = np.empty((3, 2))
            grad_lambda[1] = inverse_transpose[:, 0]
            grad_lambda[2] = inverse_transpose[:, 1]
            grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
            coefficients = result.a_phi_dofs_t_m[mesh.element_dofs[element]]
            for barycentric_tuple, weight in _QUADRATURE:
                barycentric = np.asarray(barycentric_tuple)
                radial, axial = barycentric @ element_points
                values, gradients = p2_shape(barycentric, grad_lambda)
                actual_a = float(np.dot(coefficients, values))
                actual_grad_a = coefficients @ gradients
                exact = exact_a(float(radial), float(axial))
                exact_grad_a = np.asarray(
                    (
                        (1.0 - 2.0 * radial) * cos(wave_number * axial),
                        -wave_number
                        * radial
                        * (1.0 - radial)
                        * np.sin(wave_number * axial),
                    )
                )
                actual_grad_psi = np.asarray(
                    (
                        actual_a + radial * actual_grad_a[0],
                        radial * actual_grad_a[1],
                    )
                )
                exact_grad_psi = np.asarray(
                    (
                        exact + radial * exact_grad_a[0],
                        radial * exact_grad_a[1],
                    )
                )
                measure = area * weight
                l2_numerator += measure * radial * (actual_a - exact) ** 2
                l2_denominator += measure * radial * exact**2
                energy_numerator += (
                    measure
                    / radial
                    * float(np.dot(actual_grad_psi - exact_grad_psi, actual_grad_psi - exact_grad_psi))
                )
                energy_denominator += (
                    measure
                    / radial
                    * float(np.dot(exact_grad_psi, exact_grad_psi))
                )
        l2_errors.append(sqrt(l2_numerator / l2_denominator))
        energy_errors.append(sqrt(energy_numerator / energy_denominator))
        indicators = estimate_indicators(
            result, (("manufactured-center", 0.5, -0.25, 0.25),)
        )
        estimator_norm = sqrt(
            float(
                np.sum(
                    indicators.residual_squared
                    + indicators.flux_jump_squared
                )
            )
        )
        estimator_norms.append(estimator_norm)
        estimator_effectivities.append(
            estimator_norm / max(sqrt(energy_numerator), 1.0e-300)
        )
        qoi_marked = dorfler_mark(indicators.qoi_proxy_squared, 0.5)
        centroids = np.asarray(
            [
                np.mean(mesh.vertices_rz_m[triangle], axis=0)
                for triangle in mesh.triangles
            ]
        )
        radial_distance = np.maximum(centroids[:, 0] - 0.5, 0.0)
        axial_distance = np.maximum.reduce(
            (-0.25 - centroids[:, 1], np.zeros(len(centroids)), centroids[:, 1] - 0.25)
        )
        distances = np.hypot(radial_distance, axial_distance)
        qoi_localization_ratios.append(
            float(np.mean(distances[qoi_marked]))
            / max(float(np.mean(distances)), 1.0e-300)
        )
        h_values.append(float(np.max(element_diameters(mesh))))
        dofs.append(len(mesh.p2_nodes_rz_m))
        previous_refinement = refinement
    l2_orders = [
        log(coarse / fine) / log(h_coarse / h_fine)
        for coarse, fine, h_coarse, h_fine in zip(
            l2_errors, l2_errors[1:], h_values, h_values[1:]
        )
    ]
    energy_orders = [
        log(coarse / fine) / log(h_coarse / h_fine)
        for coarse, fine, h_coarse, h_fine in zip(
            energy_errors, energy_errors[1:], h_values, h_values[1:]
        )
    ]
    return {
        "refinements": list(refinements),
        "p2_dofs": dofs,
        "h_qoi_m": h_values,
        "relative_errors": l2_errors,
        "integrated_l2_relative_errors": l2_errors,
        "integrated_energy_relative_errors": energy_errors,
        "observed_orders": l2_orders,
        "observed_l2_orders": l2_orders,
        "observed_energy_orders": energy_orders,
        "expected_l2_order": 3.0,
        "expected_energy_order": 2.0,
        "estimator_norms": estimator_norms,
        "estimator_effectivities": estimator_effectivities,
        "qoi_localization_ratios": qoi_localization_ratios,
    }


def piecewise_interface_case(*, oblique: bool) -> dict[str, float | str]:
    """Solve an exactly representable discontinuous-gradient interface problem."""

    domain = Domain(0.5, 1.5, -0.5, 0.5)
    nu_left, nu_right = 1.0, 11.0
    if oblique:
        slope = 2.5
        offset = -2.5

        def interface_r(axial: float) -> float:
            return (axial - offset) / slope

        def signed(radial: float, axial: float) -> float:
            return axial - slope * radial - offset

        def source(radial: float, axial: float) -> float:
            return (axial - offset) / radial**2

        interface_name = "oblique"
    else:
        slope = 0.0
        offset = 1.0

        def interface_r(_axial: float) -> float:
            return offset

        def signed(radial: float, _axial: float) -> float:
            return radial - offset

        def source(radial: float, _axial: float) -> float:
            return -offset / radial**2

        interface_name = "aligned"
    left_polygon = (
        (domain.r_min_m, domain.z_min_m),
        (interface_r(domain.z_min_m), domain.z_min_m),
        (interface_r(domain.z_max_m), domain.z_max_m),
        (domain.r_min_m, domain.z_max_m),
    )
    right_polygon = (
        (interface_r(domain.z_min_m), domain.z_min_m),
        (domain.r_max_m, domain.z_min_m),
        (domain.r_max_m, domain.z_max_m),
        (interface_r(domain.z_max_m), domain.z_max_m),
    )

    def region_at(radial: float, axial: float) -> str:
        return "left" if radial < interface_r(axial) else "right"

    def exact_a(radial: float, axial: float) -> float:
        nu = nu_left if region_at(radial, axial) == "left" else nu_right
        return signed(radial, axial) / nu

    problem = FEMProblem(
        f"piecewise-{interface_name}",
        domain,
        (
            Region("left", "mu-left", nu_left),
            Region("right", "mu-right", nu_right),
        ),
        region_at,
        free_current_phi=source,
        outer_boundary="dirichlet",
        dirichlet_a_phi=exact_a,
    )
    polygons = (("left", left_polygon), ("right", right_polygon))
    mesh = build_body_fitted_mesh(
        domain, polygons, region_at, radial_divisions=12, axial_divisions=12
    )
    result = solve(problem, mesh, relative_tolerance=2.0e-11)
    return {
        "interface": interface_name,
        "relative_solution_error": _sample_relative_error(result, exact_a, samples=13),
        "relative_true_residual": result.diagnostics.relative_true_residual_l2,
        "energy_action_relative": result.diagnostics.energy_action_relative,
    }


def dipole_robin_case(refinement: int = 18) -> dict[str, float]:
    """Verify the corrected Robin coefficient against an analytic vacuum dipole."""

    domain = Domain(0.4, 2.0, -1.0, 1.0)

    def exact_a(radial: float, axial: float) -> float:
        return radial / (radial * radial + axial * axial) ** 1.5

    problem = FEMProblem(
        "analytic-dipole-robin",
        domain,
        (Region("vacuum", "vacuum", 1.0),),
        lambda _r, _z: "vacuum",
        source_center_z_m=0.0,
        outer_boundary="dipole_robin",
        dirichlet_a_phi=exact_a,
    )
    mesh = build_body_fitted_mesh(
        domain,
        (),
        problem.region_at,
        radial_divisions=refinement,
        axial_divisions=refinement,
    )
    result = solve(problem, mesh, relative_tolerance=2.0e-11)
    return {
        "relative_solution_error": _sample_relative_error(result, exact_a),
        "relative_true_residual": result.diagnostics.relative_true_residual_l2,
    }


def pm_representation_equivalence(
    *, radial_divisions: int = 8, axial_divisions: int = 16
) -> dict[str, float]:
    geometry = compact_high_gradient_stack()
    recoil_problem, polygons = adapt_geometry(geometry, padding_factor=0.75)
    mesh = build_body_fitted_mesh(
        recoil_problem.domain,
        polygons,
        recoil_problem.region_at,
        radial_divisions=radial_divisions,
        axial_divisions=axial_divisions,
    )
    recoil = solve(recoil_problem, mesh, relative_tolerance=2.0e-10)
    authority = PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT
    equivalent_geometry = replace(
        geometry,
        permanent_magnet_plan=PermanentMagnetRepresentationPlan(
            f"{geometry.config_id}-{authority.value}-v1", authority
        ),
    )
    equivalent_problem, _ = adapt_geometry(equivalent_geometry, padding_factor=0.75)
    equivalent = solve(equivalent_problem, mesh, relative_tolerance=2.0e-10)
    scale = max(float(np.max(np.abs(recoil.a_phi_dofs_t_m))), 1.0e-300)
    difference = float(
        np.max(np.abs(recoil.a_phi_dofs_t_m - equivalent.a_phi_dofs_t_m)) / scale
    )
    flipped_geometry = replace(
        geometry,
        regions=tuple(
            replace(region, polarity=-region.polarity)
            if region.polarity is not None
            else region
            for region in geometry.regions
        ),
        stages=tuple(
            replace(
                stage,
                magnetization=(
                    "axial_negative"
                    if stage.magnetization.value == "axial_positive"
                    else "axial_positive"
                ),
            )
            for stage in geometry.stages
        ),
    )
    flipped_problem, _ = adapt_geometry(flipped_geometry, padding_factor=0.75)
    flipped = solve(flipped_problem, mesh, relative_tolerance=2.0e-10)
    polarity_error = float(
        np.max(np.abs(recoil.a_phi_dofs_t_m + flipped.a_phi_dofs_t_m)) / scale
    )
    return {
        "recoil_equivalent_relative_max_difference": difference,
        "polarity_reversal_relative_max_error": polarity_error,
        "recoil_energy_action_relative": recoil.diagnostics.energy_action_relative,
    }


def uniform_medium_l1a_crosscheck() -> dict[str, object]:
    """Compare the independent weak form with L1a for one resolved current band."""

    band = AzimuthalCurrentBand("uniform-check-band", 0.4, 0.6, -0.2, 0.2, 1.0)
    l1a_domain = AxisymmetricDomain(1.0, -1.0, 1.0, 64, 128)
    l1a = solve_problem_cpu(
        AxisymmetricProblem("uniform-check", l1a_domain, (band,)),
        SolverConfig(relative_tolerance=1.0e-10),
    )
    domain = Domain(0.0, 1.0, -1.0, 1.0)
    density = band.current_density_a_per_m2
    problem = FEMProblem(
        "uniform-check",
        domain,
        (Region("vacuum", "vacuum", 1.0 / MU0_H_PER_M),),
        lambda _r, _z: "vacuum",
        free_current_phi=lambda radial, axial: (
            density if 0.4 < radial < 0.6 and -0.2 < axial < 0.2 else 0.0
        ),
        outer_boundary="dirichlet",
        dirichlet_a_phi=lambda _r, _z: 0.0,
    )
    source_polygon = (("source-support", ((0.4, -0.2), (0.6, -0.2), (0.6, 0.2), (0.4, 0.2))),)
    mesh = build_body_fitted_mesh(
        domain,
        source_polygon,
        problem.region_at,
        radial_divisions=24,
        axial_divisions=48,
    )
    fem = solve(problem, mesh, relative_tolerance=1.0e-10)
    comparisons = []
    for axial in (-0.5, 0.0, 0.5):
        fem_value = field_at(fem, 0.0, axial)[2]
        axial_index = round((axial - l1a_domain.z_min_m) / l1a_domain.dz_m)
        l1a_value = l1a.b_z_t[0][axial_index]
        comparisons.append(
            {
                "z_m": axial,
                "fem_bz_t": fem_value,
                "l1a_bz_t": l1a_value,
                "relative_difference": abs(fem_value - l1a_value)
                / max(abs(fem_value), abs(l1a_value), 1.0e-300),
            }
        )
    return {
        "samples": comparisons,
        "maximum_relative_difference": max(
            item["relative_difference"] for item in comparisons
        ),
    }
