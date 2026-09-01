from math import isclose, sqrt
from random import Random

import pytest

from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    STANDARD_GRAVITY_M_PER_S2,
    XENON_ATOM_MASS_KG,
    PhysicsValidationError,
    PowerBoundaryInputs,
    XenonOperatingPoint,
    evaluate_batch,
    evaluate_performance,
)


def test_exact_single_charge_analytic_case(point_factory) -> None:
    mass = XENON_ATOM_MASS_KG
    voltage = mass / (2.0 * ELEMENTARY_CHARGE_C)
    point = point_factory(
        voltage_v=voltage,
        mass_flow_kg_per_s=mass,
        neutral=0.0,
        plus=1.0,
        double_plus=0.0,
        beam_factor=1.0,
        divergence_factor=1.0,
        cathode_power_w=0.0,
        ppu_margin_w=0.0,
    )

    result = evaluate_performance(point)

    assert result.total_xenon_particle_rate_per_s == 1.0
    assert result.xe_plus_particle_rate_per_s == 1.0
    assert result.xe_plus_speed_m_per_s == 1.0
    assert result.undiverged_ion_thrust_n == mass
    assert result.axial_thrust_n == mass
    assert isclose(
        result.specific_impulse_s,
        1.0 / STANDARD_GRAVITY_M_PER_S2,
        rel_tol=2e-16,
    )
    assert result.power_budget.beam_current_a == ELEMENTARY_CHARGE_C
    assert result.power_budget.beam_kinetic_power_w == 0.5 * mass
    assert result.power_budget.anode_to_beam_efficiency == 1.0


def test_mixed_xe_plus_and_double_plus_case(point_factory) -> None:
    point = point_factory(
        voltage_v=600.0,
        mass_flow_kg_per_s=2.0e-6,
        neutral=0.1,
        plus=0.6,
        double_plus=0.3,
        beam_factor=0.8,
        divergence_factor=0.75,
        cathode_power_w=12.0,
        ppu_margin_w=30.0,
    )
    result = evaluate_performance(point)
    rate = 2.0e-6 / XENON_ATOM_MASS_KG
    v_plus = sqrt(2.0 * ELEMENTARY_CHARGE_C * 600.0 / XENON_ATOM_MASS_KG)
    v_double = sqrt(2.0) * v_plus
    expected_undiverged = XENON_ATOM_MASS_KG * rate * (
        0.6 * v_plus + 0.3 * v_double
    )

    assert isclose(result.xe_double_plus_speed_m_per_s, v_double, rel_tol=1e-15)
    assert isclose(result.undiverged_ion_thrust_n, expected_undiverged, rel_tol=1e-15)
    assert result.axial_thrust_n == 0.75 * result.undiverged_ion_thrust_n
    assert isclose(
        result.power_budget.beam_current_a,
        ELEMENTARY_CHARGE_C * rate * (0.6 + 2.0 * 0.3),
        rel_tol=1e-15,
    )
    assert isclose(
        result.power_budget.anode_to_beam_efficiency, 0.8, rel_tol=2e-15
    )


def test_zero_flow_and_voltage_are_not_running_points(point_factory) -> None:
    with pytest.raises(PhysicsValidationError):
        point_factory(mass_flow_kg_per_s=0.0)
    with pytest.raises(PhysicsValidationError):
        point_factory(voltage_v=0.0)


def test_fully_neutral_endpoint_is_finite(point_factory) -> None:
    neutral = evaluate_performance(
        point_factory(
            neutral=1.0,
            plus=0.0,
            double_plus=0.0,
            cathode_power_w=0.0,
            ppu_margin_w=0.0,
        )
    )
    assert neutral.power_budget.beam_current_a == 0.0
    assert neutral.axial_thrust_n == 0.0


def test_zero_divergence_factor_removes_only_axial_momentum(point_factory) -> None:
    result = evaluate_performance(point_factory(divergence_factor=0.0))
    assert result.undiverged_ion_thrust_n > 0.0
    assert result.axial_thrust_n == 0.0
    assert result.specific_impulse_s == 0.0
    assert result.power_budget.beam_kinetic_power_w > 0.0


def test_power_budget_rejects_impossible_ppu_boundary(point_factory) -> None:
    point = point_factory(ppu_margin_w=0.0)
    invalid = XenonOperatingPoint(
        discharge_voltage_v=point.discharge_voltage_v,
        propellant_mass_flow=point.propellant_mass_flow,
        charge_state_fractions=point.charge_state_fractions,
        mass_utilization=point.mass_utilization,
        beam_divergence_factors=point.beam_divergence_factors,
        power_boundaries=PowerBoundaryInputs(
            point.power_boundaries.cathode_input_power_w,
            point.power_boundaries.ppu_input_power_w - 1.0,
        ),
    )
    with pytest.raises(PhysicsValidationError, match="PPU|ppu"):
        evaluate_performance(invalid)


def test_deterministic_random_batch_closes_conservation(point_factory) -> None:
    random = Random(20200901)
    points = []
    for _ in range(200):
        neutral = random.random()
        double_plus = (1.0 - neutral) * random.random()
        plus = 1.0 - neutral - double_plus
        points.append(
            point_factory(
                voltage_v=random.uniform(0.0, 2000.0),
                mass_flow_kg_per_s=random.uniform(0.0, 5.0e-6),
                neutral=neutral,
                plus=plus,
                double_plus=double_plus,
                beam_factor=random.uniform(0.2, 1.0),
                divergence_factor=random.random(),
                cathode_power_w=random.uniform(0.0, 50.0),
                ppu_margin_w=random.uniform(0.0, 100.0),
            )
        )

    first = evaluate_batch(points)
    second = evaluate_batch(points)
    assert first == second
    for point, result in zip(points, first, strict=True):
        diagnostics = result.diagnostics
        rate_scale = max(1.0, result.total_xenon_particle_rate_per_s)
        assert abs(diagnostics.particle_rate_residual_particles_per_s) <= 3e-16 * rate_scale
        assert abs(diagnostics.mass_flow_residual_kg_per_s) <= 5e-16 * max(
            1e-300, point.propellant_mass_flow.kg_per_s
        )
        assert abs(diagnostics.beam_current_residual_a) <= 1e-14
        assert abs(diagnostics.beam_power_residual_w) <= 5e-12 * max(
            1.0, result.power_budget.beam_kinetic_power_w
        )
        assert isclose(
            result.power_budget.beam_kinetic_power_w,
            result.power_budget.beam_current_a
            * point.discharge_voltage_v,
            rel_tol=5e-15,
            abs_tol=1e-20,
        )


def test_empty_reference_batch_is_rejected() -> None:
    with pytest.raises(PhysicsValidationError, match="empty"):
        evaluate_batch([])
