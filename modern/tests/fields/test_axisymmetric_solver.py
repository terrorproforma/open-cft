from __future__ import annotations

from math import isfinite

import pytest

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    FieldConvergenceError,
    FieldValidationError,
    SolverConfig,
    current_density_grid,
    run_manufactured_convergence,
    solve_problem_cpu,
)
from cft_revival.fields.numerics import solve_current_density_cpu
from cft_revival.fields.verification import manufactured_values


def problem(*sources: AzimuthalCurrentBand) -> AxisymmetricProblem:
    return AxisymmetricProblem(
        name="test",
        domain=AxisymmetricDomain(0.12, -0.15, 0.15, 24, 48),
        sources=tuple(sources),
    )


def coil(name: str = "coil", *, z_min: float = -0.02, z_max: float = 0.02, polarity: int = 1):
    return AzimuthalCurrentBand(name, 0.04, 0.06, z_min, z_max, 2_000.0, polarity)


def flatten(rows):
    return tuple(value for row in rows for value in row)


def test_manufactured_solution_shows_second_order_psi_and_field_convergence() -> None:
    report = run_manufactured_convergence()
    assert min(report.psi_orders) > 1.95
    assert min(report.field_orders) > 1.95
    assert report.cases[-1].source_operator_relative_l2 < 5.0e-6
    assert report.cases[-1].relative_solver_residual < 1.0e-10
    assert report.cases[-1].axis_bz_max_abs_t < 1.0e-8
    assert (
        report.cases[-1].max_flux_reconstruction_identity_t_per_m < 1.0e-12
    )


def test_manufactured_source_sign_matches_positive_operator() -> None:
    domain = AxisymmetricDomain(0.12, -0.15, 0.15, 16, 32)
    exact_psi, source, _, _ = manufactured_values(domain)
    centre = (domain.radial_intervals // 2) * domain.shape[1] + (
        domain.axial_intervals // 2
    )
    assert exact_psi[centre] > 0.0
    assert source[centre] > 0.0
    field = solve_current_density_cpu(
        domain,
        source,
        permeability_h_per_m=problem().permeability_h_per_m,
    )
    assert flatten(field.psi_wb)[centre] > 0.0


def test_zero_source_is_exact_and_immediately_converged() -> None:
    field = solve_problem_cpu(problem())
    assert field.diagnostics.converged
    assert field.diagnostics.iterations == 0
    assert field.diagnostics.final_residual_l2 == 0.0
    assert set(flatten(field.psi_wb)) == {0.0}
    assert set(flatten(field.b_r_t)) == {0.0}
    assert set(flatten(field.b_z_t)) == {0.0}


def test_linear_vacuum_limit_scales_with_ampere_turns() -> None:
    base = solve_problem_cpu(problem(coil()))
    doubled_source = AzimuthalCurrentBand(
        "double", 0.04, 0.06, -0.02, 0.02, 4_000.0, 1
    )
    doubled = solve_problem_cpu(problem(doubled_source))
    for actual, reference in zip(flatten(doubled.b_z_t), flatten(base.b_z_t)):
        assert actual == pytest.approx(2.0 * reference, rel=2.0e-12, abs=1.0e-15)


def test_opposing_bands_create_axis_cusp_at_symmetry_plane() -> None:
    field = solve_problem_cpu(
        problem(
            coil("left", z_min=-0.07, z_max=-0.03, polarity=1),
            coil("right", z_min=0.03, z_max=0.07, polarity=-1),
        )
    )
    centre = len(field.z_m) // 2
    peak = max(abs(value) for value in field.b_z_t[0])
    assert abs(field.b_z_t[0][centre]) < 1.0e-10 * peak
    assert field.b_z_t[0][centre - 2] * field.b_z_t[0][centre + 2] < 0.0
    assert field.diagnostics.max_flux_reconstruction_identity_t_per_m < 1.0e-10


def test_extreme_finite_source_remains_finite() -> None:
    source = AzimuthalCurrentBand("strong", 0.04, 0.06, -0.02, 0.02, 1.0e10)
    field = solve_problem_cpu(problem(source))
    assert all(isfinite(value) for value in flatten(field.psi_wb))
    assert all(isfinite(value) for value in flatten(field.b_r_t))
    assert all(isfinite(value) for value in flatten(field.b_z_t))
    assert max(abs(value) for value in flatten(field.b_z_t)) > 1.0e4


def test_nonconvergence_is_status_and_never_silent_acceptance() -> None:
    config = SolverConfig(relative_tolerance=1.0e-14, max_iterations=1)
    with pytest.raises(FieldConvergenceError, match="did not converge"):
        solve_problem_cpu(problem(coil()), config)
    field = solve_problem_cpu(problem(coil()), config, raise_on_nonconvergence=False)
    assert not field.diagnostics.converged
    assert field.diagnostics.relative_residual_l2 > config.relative_tolerance


def test_shape_nonfinite_and_geometry_errors_are_typed() -> None:
    domain = problem().domain
    with pytest.raises(FieldValidationError, match="expected"):
        solve_current_density_cpu(
            domain, [0.0], permeability_h_per_m=problem().permeability_h_per_m
        )
    invalid = [0.0] * (domain.shape[0] * domain.shape[1])
    invalid[10] = float("nan")
    with pytest.raises(FieldValidationError, match="finite"):
        solve_current_density_cpu(
            domain, invalid, permeability_h_per_m=problem().permeability_h_per_m
        )
    with pytest.raises(FieldValidationError, match="symmetry axis"):
        AzimuthalCurrentBand("axis", 0.0, 0.02, -0.01, 0.01, 1.0)
    with pytest.raises(FieldValidationError, match="strictly inside"):
        problem(AzimuthalCurrentBand("outside", 0.04, 0.13, -0.01, 0.01, 1.0))


def test_source_grid_uses_explicit_signed_equivalent_current_density() -> None:
    positive = coil("positive", polarity=1)
    negative = coil("negative", polarity=-1)
    assert positive.current_density_a_per_m2 == -negative.current_density_a_per_m2
    assert set(current_density_grid(problem(positive, negative))) == {0.0}
    sampled = current_density_grid(problem(positive))
    domain = problem().domain
    represented_ampere_turns = sum(sampled) * domain.dr_m * domain.dz_m
    assert represented_ampere_turns == pytest.approx(positive.ampere_turns_a, rel=2.0e-15)
