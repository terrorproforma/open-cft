from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pytest

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    MU0_H_PER_M,
    SolverConfig,
    current_density_grid,
    solve_problem_cpu,
)
from cft_revival.geometry import (
    PermanentMagnetAuthority,
    PermanentMagnetRepresentationPlan,
    divergent_exit_stack,
    historical_envelope_baseline,
)
from cft_revival.material_fields import (
    MaterialFieldValidationError,
    MaterialSolveConfig,
    RasterizedMaterialProblem,
    adapt_geometry,
    apply_material_operator,
    design_domain,
    minimum_operator_eigenvalue,
    raster_memory_preflight,
    solve_material_problem_cpu,
)
from cft_revival.material_fields.acceptance import _validate_qualification
from cft_revival.material_fields.numerics import _harmonic
from cft_revival.material_fields.adapters import (
    _dipole_robin_alpha_axial,
    _dipole_robin_alpha_radial,
    _polygon_rectangle_area,
    _series_face_reluctivity,
)

SCREENING_CONFIG = MaterialSolveConfig(allow_underresolved_screening=True)


def test_memory_limited_qualification_is_explicit_and_fail_closed() -> None:
    requested = AxisymmetricDomain(1.0, -1.0, 1.0, 3000, 3000)
    report = raster_memory_preflight(requested, enforce=False)
    assert report["fits"] is False
    with pytest.raises(MaterialFieldValidationError, match="memory preflight"):
        raster_memory_preflight(requested)

    qualification = {
        "schema_version": "cft_revival.material_fields.qualification/1.4.0",
        "study_scope": "MEMORY_LIMITED_REDUCED_SCREENING",
        "status": "NOT_EVALUATED",
        "reason_code": "HOST_MEMORY_LIMIT",
        "required_role_count": 10,
        "completed_role_count": 10,
        "not_evaluated_roles": [],
        "requested_base_grid": [3000, 3000],
        "executed_base_grid": [60, 120],
        "estimated_requested_raster_bytes": int(
            report["estimated_raster_bytes"]
        ),
        "safe_raster_bytes": int(report["safe_raster_bytes"]),
    }
    assert _validate_qualification(
        qualification, base_grid=[60, 120]
    )["status"] == "NOT_EVALUATED"
    promoted = {**qualification, "status": "EVALUATED"}
    with pytest.raises(MaterialFieldValidationError, match="fail closed"):
        _validate_qualification(promoted, base_grid=[60, 120])


def _uniform_problem(domain: AxisymmetricDomain, source: tuple[float, ...], mu: float):
    count = domain.shape[0] * domain.shape[1]
    return RasterizedMaterialProblem(
        "uniform",
        domain,
        "0" * 64,
        "1" * 64,
        "equivalent_bound_current",
        ("uniform",) * count,
        (None,) * count,
        (None,) * count,
        (1.0 / mu,) * count,
        (0.0,) * count,
        (0.0,) * count,
        source,
        (),
        (0.0, 0.0),
    )


def _flatten(rows):
    return tuple(value for row in rows for value in row)


def test_uniform_linear_medium_reduces_to_l1a() -> None:
    domain = AxisymmetricDomain(0.12, -0.15, 0.15, 24, 48)
    band = AzimuthalCurrentBand("coil", 0.04, 0.06, -0.03, 0.03, 1200.0)
    l1a_problem = AxisymmetricProblem("l1a", domain, (band,), MU0_H_PER_M)
    source = current_density_grid(l1a_problem)
    l1a = solve_problem_cpu(l1a_problem)
    l1b = solve_material_problem_cpu(
        _uniform_problem(domain, source, MU0_H_PER_M),
        MaterialSolveConfig(SolverConfig(relative_tolerance=1.0e-11)),
    )
    for actual, expected in zip(_flatten(l1b.field.psi_wb), _flatten(l1a.psi_wb)):
        assert actual == pytest.approx(expected, rel=3.0e-9, abs=1.0e-18)
    assert l1b.diagnostics.energy_balance_relative < 2.0e-9


def test_harmonic_material_face_and_piecewise_flux_continuity() -> None:
    domain = AxisymmetricDomain(1.0, -1.0, 1.0, 8, 8)
    count = domain.shape[0] * domain.shape[1]
    nu = [2.0] * count
    nz = domain.shape[1]
    for i in range(4, domain.shape[0]):
        for j in range(domain.shape[1]):
            nu[i * nz + j] = 8.0
    problem = replace(
        _uniform_problem(domain, (0.0,) * count, 0.5),
        reluctivity_per_m_h=tuple(nu),
        radial_face_reluctivity_per_m_h=(),
        axial_face_reluctivity_per_m_h=(),
    )
    vector = [0.0] * count
    vector[4 * nz + 4] = 1.0
    applied = apply_material_operator(problem, vector)
    assert all(isfinite(value) for value in applied)
    harmonic = 2.0 * 2.0 * 8.0 / (2.0 + 8.0)
    assert harmonic == 3.2
    expected_interface_coefficient = -harmonic / (
        (3.5 * domain.dr_m) * domain.dr_m**2
    )
    assert applied[3 * nz + 4] == pytest.approx(expected_interface_coefficient)


def test_geometry_adapter_preserves_hashes_ids_polarities_temperatures_and_tolerances() -> None:
    geometry = historical_envelope_baseline()
    domain = design_domain(geometry, radial_intervals=24, axial_intervals=48, padding_factor=0.5)
    raster = adapt_geometry(geometry, domain)
    assert raster.geometry_sha256 == geometry.canonical_sha256
    assert len(raster.magnetics_sha256) == 64
    pm_material_id = geometry.region_by_id(geometry.stages[0].magnet_region_id).material_id
    assert pm_material_id in raster.material_ids
    assert {-1, 1} <= {value for value in raster.polarities if value is not None}
    assert all(
        temperature == pytest.approx(293.15)
        for material, temperature in zip(raster.material_ids, raster.temperatures_k)
        if material == pm_material_id
    )
    assert raster.tolerances_m == (
        geometry.manufacturing.radial_tolerance_m,
        geometry.manufacturing.axial_tolerance_m,
    )
    assert max(
        abs(item.relative_volume_error)
        for item in raster.raster_diagnostics
        if item.item_id != "ambient-background"
    ) < 2.0e-14


def test_pm_polarity_reversal_is_antisymmetric() -> None:
    geometry = historical_envelope_baseline()
    domain = design_domain(geometry, radial_intervals=20, axial_intervals=40, padding_factor=0.5)
    problem = adapt_geometry(geometry, domain)
    positive = solve_material_problem_cpu(problem, SCREENING_CONFIG)
    reversed_problem = replace(
        problem,
        remanence_r_t=tuple(-value for value in problem.remanence_r_t),
        remanence_z_t=tuple(-value for value in problem.remanence_z_t),
        remanence_g_r_face_a_per_m=tuple(
            -value for value in problem.remanence_g_r_face_a_per_m
        ),
        remanence_g_z_face_a_per_m=tuple(
            -value for value in problem.remanence_g_z_face_a_per_m
        ),
        polarities=tuple(None if value is None else -value for value in problem.polarities),
    )
    negative = solve_material_problem_cpu(reversed_problem, SCREENING_CONFIG)
    for actual, expected in zip(_flatten(negative.field.b_z_t), _flatten(positive.field.b_z_t)):
        assert actual == pytest.approx(-expected, rel=2.0e-9, abs=1.0e-12)


def test_nonlinear_and_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(MaterialFieldValidationError, match="gated"):
        MaterialSolveConfig(nonlinear_enabled=True)
    domain = AxisymmetricDomain(1.0, -1.0, 1.0, 8, 8)
    count = domain.shape[0] * domain.shape[1]
    with pytest.raises(MaterialFieldValidationError, match="finite"):
        replace(
            _uniform_problem(domain, (0.0,) * count, MU0_H_PER_M),
            remanence_z_t=(float("nan"),) + (0.0,) * (count - 1),
        )


def test_pm_authority_is_structural_and_free_current_is_separate() -> None:
    geometry = historical_envelope_baseline()
    domain = design_domain(geometry, radial_intervals=16, axial_intervals=32, padding_factor=0.5)
    recoil = adapt_geometry(geometry, domain)
    with pytest.raises(MaterialFieldValidationError, match="forbids equivalent"):
        replace(
            recoil,
            pm_bound_current_phi_a_per_m2=(1.0,) + (0.0,) * (
                len(recoil.pm_bound_current_phi_a_per_m2) - 1
            ),
        )
    count = domain.shape[0] * domain.shape[1]
    equivalent = _uniform_problem(domain, (0.0,) * count, MU0_H_PER_M)
    bound = (1.0,) + (0.0,) * (count - 1)
    equivalent = replace(
        equivalent,
        pm_region_count=1,
        pm_bound_current_phi_a_per_m2=bound,
        free_current_phi_a_per_m2=(2.0,) + (0.0,) * (count - 1),
    )
    assert equivalent.free_current_phi_a_per_m2[0] == 2.0
    assert equivalent.pm_bound_current_phi_a_per_m2[0] == 1.0
    with pytest.raises(MaterialFieldValidationError, match="zero remanence"):
        replace(equivalent, remanence_z_t=(1.0,) + (0.0,) * (count - 1))


def test_equivalent_handoff_zeros_remanence_and_conserves_surface_action() -> None:
    geometry = historical_envelope_baseline()
    authority = PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT
    geometry = replace(
        geometry,
        permanent_magnet_plan=PermanentMagnetRepresentationPlan(
            f"{geometry.config_id}-{authority.value}-v1", authority
        ),
    )
    problem = adapt_geometry(
        geometry,
        design_domain(geometry, radial_intervals=32, axial_intervals=64, padding_factor=1.0),
    )
    assert set(problem.remanence_r_t) == {0.0}
    assert set(problem.remanence_z_t) == {0.0}
    assert any(value != 0.0 for value in problem.pm_bound_current_phi_a_per_m2)
    assert set(problem.free_current_phi_a_per_m2) == {0.0}
    constant_action = next(
        item for item in problem.weak_action_diagnostics if item.basis_id == "equivalent-value-one"
    )
    assert constant_action.absolute_bias_a < 1.0e-10


def test_recoil_and_mu_corrected_equivalent_sources_converge_together() -> None:
    recoil_geometry = historical_envelope_baseline()
    authority = PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT
    equivalent_geometry = replace(
        recoil_geometry,
        permanent_magnet_plan=PermanentMagnetRepresentationPlan(
            f"{recoil_geometry.config_id}-{authority.value}-v1", authority
        ),
    )
    gaps = []
    for radial_intervals in (16, 32, 64):
        domain = design_domain(
            recoil_geometry,
            radial_intervals=radial_intervals,
            axial_intervals=2 * radial_intervals,
            padding_factor=1.0,
        )
        recoil = solve_material_problem_cpu(
            adapt_geometry(recoil_geometry, domain), SCREENING_CONFIG
        )
        equivalent = solve_material_problem_cpu(
            adapt_geometry(equivalent_geometry, domain), SCREENING_CONFIG
        )
        stage_gaps = []
        for _, _, axial in recoil.problem.qoi_locations_rz_m:
            coordinate = (axial - domain.z_min_m) / domain.dz_m
            lower = int(coordinate)
            fraction = coordinate - lower
            left = (
                (1.0 - fraction) * recoil.field.b_z_t[0][lower]
                + fraction * recoil.field.b_z_t[0][lower + 1]
            )
            right = (
                (1.0 - fraction) * equivalent.field.b_z_t[0][lower]
                + fraction * equivalent.field.b_z_t[0][lower + 1]
            )
            stage_gaps.append(
                abs(left - right) / max(abs(left), abs(right), 1.0e-300)
            )
        gaps.append(max(stage_gaps))
    assert gaps[2] < gaps[1] < gaps[0]
    assert gaps[2] < 5.0e-3


def test_harmonic_reluctivity_is_stable_across_finite_range() -> None:
    assert _harmonic(1.0e308, 1.0e308) == 1.0e308
    assert _harmonic(5.0e-324, 5.0e-324) == 5.0e-324
    mixed = _harmonic(5.0e-324, 1.0e308)
    assert mixed == 1.0e-323


def test_off_face_series_resistance_matches_independent_oracle() -> None:
    high = 89.04097468
    fraction = (high - 67.11) / (high - 1.0)
    exact = 1.0 / (fraction / 1.0 + (1.0 - fraction) / high)
    implemented = _series_face_reluctivity(
        0.0,
        1.0,
        (0.0, fraction, 1.0),
        lambda coordinate: 1.0 if coordinate < fraction else high,
        radial=False,
    )
    arithmetic_wrong = fraction + (1.0 - fraction) * high
    assert arithmetic_wrong == pytest.approx(67.11, rel=1.0e-10)
    assert exact == pytest.approx(3.883, rel=2.0e-9)
    assert implemented == pytest.approx(exact, rel=2.0e-15)
    for interface in (0.13, 0.37, 0.81):
        for contrast in (2.0, 10.0, 1.0e3):
            expected = 1.0 / (
                interface / 1.0 + (1.0 - interface) / contrast
            )
            actual = _series_face_reluctivity(
                0.0,
                1.0,
                (0.0, interface, 1.0),
                lambda coordinate, cut=interface, value=contrast: (
                    1.0 if coordinate < cut else value
                ),
                radial=False,
            )
            assert actual == pytest.approx(expected, rel=2.0e-15)
    radial_expected = 1.5 / (
        0.5 * (1.4**2 - 1.0**2) / 2.0
        + 0.5 * (2.0**2 - 1.4**2) / 20.0
    )
    radial_actual = _series_face_reluctivity(
        1.0,
        2.0,
        (1.0, 1.4, 2.0),
        lambda coordinate: 2.0 if coordinate < 1.4 else 20.0,
        radial=True,
    )
    assert radial_actual == pytest.approx(radial_expected, rel=2.0e-15)


def test_tapered_geometry_regions_are_preserved_in_provenance() -> None:
    geometry = divergent_exit_stack()
    problem = adapt_geometry(
        geometry,
        design_domain(geometry, radial_intervals=20, axial_intervals=40, padding_factor=1.0),
    )
    tapered = [item for item in problem.geometry_region_provenance if item[1] == "linear_taper_annulus"]
    assert tapered
    assert all(item[2] or "preserved_geometry_only" in item[3] for item in tapered)
    assert problem.pm_region_count == len(geometry.stages)
    expected_active = {
        region.region_id
        for region in geometry.regions
        if geometry.material_by_id(region.material_id).relative_permeability != 1.0
    }
    assert {item[0] for item in problem.feature_effective_cells} == expected_active


def test_dipole_robin_boundary_allows_negative_radial_log_coefficient() -> None:
    geometry = historical_envelope_baseline()
    domain = design_domain(
        geometry, radial_intervals=32, axial_intervals=64, padding_factor=1.0
    )
    problem = adapt_geometry(geometry, domain)
    assert problem.outer_boundary_kind == "dipole_robin_psi"
    assert all(value > 0.0 for value in problem.robin_radial_q)
    _, z_min, z_max, _ = problem.source_envelope_m
    source_center = 0.5 * (z_min + z_max)
    j = min(
        range(domain.shape[1]),
        key=lambda index: abs(
            domain.z_min_m + index * domain.dz_m - source_center
        ),
    )
    axial = domain.z_min_m + j * domain.dz_m
    alpha = _dipole_robin_alpha_radial(
        domain.radius_m, axial, source_center
    )
    expected = 1.0 / (1.0 + alpha * domain.dr_m)
    assert problem.robin_radial_q[j] == pytest.approx(expected, rel=2.0e-15)
    assert min(
        _dipole_robin_alpha_radial(
            domain.radius_m, domain.z_min_m + index * domain.dz_m, source_center
        )
        for index in range(domain.shape[1])
    ) < 0.0


def test_dipole_robin_small_grid_operator_is_numerically_positive_definite() -> None:
    geometry = historical_envelope_baseline()
    problem = adapt_geometry(
        geometry,
        design_domain(
            geometry, radial_intervals=12, axial_intervals=20, padding_factor=1.0
        ),
    )
    assert minimum_operator_eigenvalue(problem) > 0.0


def test_dipole_robin_log_derivative_matches_every_outer_side_and_corner() -> None:
    radius = 0.8
    z_min, z_max, center = -1.1, 1.4, 0.15

    def psi(radial: float, axial: float) -> float:
        rho2 = radial * radial + (axial - center) ** 2
        return radial * radial / rho2**1.5

    for axial in (z_min, center, z_max):
        value = psi(radius, axial)
        derivative = value * (
            2.0 / radius
            - 3.0 * radius / (radius**2 + (axial - center) ** 2)
        )
        assert derivative + _dipole_robin_alpha_radial(
            radius, axial, center
        ) * value == pytest.approx(0.0, abs=2.0e-15)
    for axial, outward_sign in ((z_min, -1.0), (z_max, 1.0)):
        for radial in (0.2, 0.5, radius):
            value = psi(radial, axial)
            derivative_z = (
                -3.0
                * (axial - center)
                / (radial**2 + (axial - center) ** 2)
                * value
            )
            assert outward_sign * derivative_z + _dipole_robin_alpha_axial(
                radial, axial, center
            ) * value == pytest.approx(0.0, abs=2.0e-15)


def test_embedded_linear_polygon_area_is_exact_under_grid_shifts() -> None:
    polygon = ((0.21, -0.37), (0.72, -0.29), (0.64, 0.43), (0.16, 0.31))
    exact = 0.5 * abs(
        sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(polygon, polygon[1:] + polygon[:1])
        )
    )
    for intervals, shift in ((11, 0.0), (23, 0.013), (47, -0.009)):
        dr = 1.0 / intervals
        dz = 1.0 / intervals
        represented = 0.0
        for i in range(intervals):
            for j in range(intervals):
                represented += _polygon_rectangle_area(
                    polygon,
                    i * dr,
                    (i + 1) * dr,
                    -0.5 + shift + j * dz,
                    -0.5 + shift + (j + 1) * dz,
                )
        assert represented == pytest.approx(exact, rel=3.0e-14, abs=2.0e-15)


def test_embedded_polygon_full_small_cell_does_not_exceed_cell_area() -> None:
    r0, r1 = 0.0027016728624535313, 0.0028076208178438662
    z0, z1 = 0.021991181930693066, 0.022097308168316825
    polygon = ((0.002, 0.018), (0.003, 0.018), (0.004, 0.024), (0.003, 0.024))
    cell_area = (r1 - r0) * (z1 - z0)
    assert _polygon_rectangle_area(polygon, r0, r1, z0, z1) <= cell_area
    assert _polygon_rectangle_area(
        polygon, r0, r1, z0, z1
    ) == pytest.approx(cell_area, rel=2.0e-15)
