from dataclasses import fields, is_dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from math import copysign, fsum, isfinite, nextafter, ulp
from random import Random
from sys import float_info

import pytest

from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    STANDARD_GRAVITY_M_PER_S2,
    ChargeStateFractions,
    PhysicsValidationError,
    PowerBoundaryInputs,
    PropellantMassFlow,
    evaluate_batch,
    evaluate_performance,
)
from cft_revival.physics.numerics import prepare_operating_point
from cft_revival.physics.warp_backend import (
    device_available,
    evaluate_performance_warp,
)


def _with_ppu(point, ppu_input_w: float):
    return replace(
        point,
        power_boundaries=PowerBoundaryInputs(
            point.power_boundaries.cathode_input_power_w,
            ppu_input_w,
        ),
    )


def _all_float_values(value):
    if is_dataclass(value):
        for field in fields(value):
            yield from _all_float_values(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _all_float_values(item)
    elif isinstance(value, float):
        yield value


@pytest.mark.parametrize("cathode_power_w", [1.0e-200, 1.0, 1.0e200])
def test_ppu_boundary_uses_four_ulp_policy_across_scales(
    cathode_power_w: float, point_factory
) -> None:
    base = point_factory(
        neutral=1.0,
        plus=0.0,
        double_plus=0.0,
        cathode_power_w=cathode_power_w,
        ppu_margin_w=0.0,
    )
    required = prepare_operating_point(base).thruster_power
    tolerance = 4.0 * ulp(required)
    inside_below = required - tolerance
    outside_below = nextafter(inside_below, 0.0)
    inside_above = required + tolerance

    for reported in (inside_below, required, inside_above):
        result = evaluate_performance(_with_ppu(base, reported))
        assert result.power_budget.ppu_conversion_loss_w == 0.0
        assert copysign(1.0, result.power_budget.ppu_conversion_loss_w) == 1.0
        assert result.diagnostics.ppu_power_margin_w == 0.0
        assert result.power_budget.ppu_input_power_w == required
        assert result.power_budget.requested_ppu_input_power_w == reported
        assert result.power_budget.ppu_boundary_adjustment_w == required - reported

    with pytest.raises(PhysicsValidationError, match="four-ULP"):
        evaluate_performance(_with_ppu(base, outside_below))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("cathode_power_w", [1.0e-200, 13.0, 1.0e200])
def test_ppu_boundary_contract_matches_warp_and_canonicalizes_zero(
    device: str, cathode_power_w: float, point_factory
) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    base = point_factory(
        neutral=1.0,
        plus=0.0,
        double_plus=0.0,
        cathode_power_w=cathode_power_w,
        ppu_margin_w=0.0,
    )
    required = prepare_operating_point(base).thruster_power
    inside_below = required - 4.0 * ulp(required)
    accepted = _with_ppu(base, inside_below)
    rejected = _with_ppu(base, nextafter(inside_below, 0.0))

    cpu = evaluate_performance(accepted)
    warp = evaluate_performance_warp([accepted], device=device).results[0]
    for result in (cpu, warp):
        assert result.power_budget.ppu_conversion_loss_w == 0.0
        assert copysign(1.0, result.power_budget.ppu_conversion_loss_w) == 1.0
        assert result.diagnostics.ppu_power_margin_w == 0.0
        assert result.power_budget.ppu_input_power_w == required
        assert result.power_budget.ppu_input_to_beam_efficiency == (
            result.power_budget.beam_kinetic_power_w / required
        )

    with pytest.raises(PhysicsValidationError, match="four-ULP"):
        evaluate_performance(rejected)
    with pytest.raises(PhysicsValidationError, match="four-ULP"):
        evaluate_performance_warp([rejected], device=device)


def _reconstructed_boundary_points(point_factory):
    random = Random(1_024_5090)
    points = []
    for _ in range(1024):
        neutral = random.random()
        double_plus = (1.0 - neutral) * random.random()
        plus = 1.0 - neutral - double_plus
        base = point_factory(
            voltage_v=random.uniform(1.0e-6, 2.0e3),
            mass_flow_kg_per_s=10.0 ** random.uniform(-300.0, -3.0),
            neutral=neutral,
            plus=plus,
            double_plus=double_plus,
            beam_factor=random.uniform(0.1, 1.0),
            divergence_factor=random.random(),
            cathode_power_w=10.0 ** random.uniform(-200.0, 4.0),
        )
        fractions = base.charge_state_fractions
        caller_beam_current = (
            ELEMENTARY_CHARGE_C
            * base.propellant_mass_flow.kg_per_s
            / base.xenon_atom_mass_kg
            * fractions.charge_weighted_ion_fraction
        )
        caller_required = (
            base.discharge_voltage_v
            * caller_beam_current
            / base.beam_divergence_factors.beam_current_fraction_of_anode_current
            + base.power_boundaries.cathode_input_power_w
        )
        points.append(_with_ppu(base, caller_required))
    return points


@pytest.mark.parametrize("device", ["reference", "cpu", "cuda:0"])
def test_all_1024_regrouped_ppu_boundaries_are_canonical(
    device: str, point_factory
) -> None:
    if device != "reference" and not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    points = _reconstructed_boundary_points(point_factory)
    results = (
        evaluate_batch(points)
        if device == "reference"
        else evaluate_performance_warp(points, device=device).results
    )
    adjustments = 0
    for result in results:
        budget = result.power_budget
        assert budget.ppu_conversion_loss_w == 0.0
        assert copysign(1.0, budget.ppu_conversion_loss_w) == 1.0
        assert budget.ppu_input_power_w == budget.thruster_electrical_input_power_w
        if budget.ppu_input_power_w > 0.0:
            assert budget.ppu_input_to_beam_efficiency == (
                budget.beam_kinetic_power_w / budget.ppu_input_power_w
            )
            assert 0.0 <= budget.ppu_input_to_beam_efficiency <= 1.0
        adjustments += budget.ppu_boundary_adjustment_w != 0.0
    assert adjustments > 0


def test_fraction_sum_is_normalized_only_within_two_ulps() -> None:
    one_ulp_noise = ulp(1.0)
    accepted = ChargeStateFractions(1.0, one_ulp_noise, 0.0)
    assert fsum(
        (accepted.xe_neutral, accepted.xe_plus, accepted.xe_double_plus)
    ) == 1.0

    with pytest.raises(PhysicsValidationError, match="two exact binary64 ULPs"):
        ChargeStateFractions(1.0, 3.0 * one_ulp_noise, 0.0)
    with pytest.raises(PhysicsValidationError, match="sum to one"):
        ChargeStateFractions(0.2, 0.3, 0.5000000000005)
    rounded_inside_but_exactly_outside = (
        1.0,
        2.0 * one_ulp_noise,
        0.25 * one_ulp_noise,
    )
    assert abs(fsum(rounded_inside_but_exactly_outside) - 1.0) <= (
        2.0 * one_ulp_noise
    )
    exact_sum = sum(
        (Fraction.from_float(value) for value in rounded_inside_but_exactly_outside),
        start=Fraction(0),
    )
    assert exact_sum - 1 > Fraction.from_float(2.0 * one_ulp_noise)
    with pytest.raises(PhysicsValidationError, match="two exact binary64 ULPs"):
        ChargeStateFractions(*rounded_inside_but_exactly_outside)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_normalized_fraction_contract_is_shared_with_warp(
    device: str, point_factory
) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    point = point_factory(
        neutral=1.0,
        plus=ulp(1.0),
        double_plus=0.0,
    )
    reference = evaluate_performance(point)
    actual = evaluate_performance_warp([point], device=device).results[0]
    assert fsum(
        (
            point.charge_state_fractions.xe_neutral,
            point.charge_state_fractions.xe_plus,
            point.charge_state_fractions.xe_double_plus,
        )
    ) == 1.0
    assert actual.power_budget.beam_current_a > 0.0
    assert abs(
        actual.power_budget.beam_current_a
        / reference.power_budget.beam_current_a
        - 1.0
    ) <= 3.0e-15


def test_extreme_finite_inputs_are_rejected_before_nonfinite_publication(
    point_factory,
) -> None:
    base = point_factory()
    huge_voltage = replace(
        base,
        discharge_voltage_v=float_info.max,
        power_boundaries=PowerBoundaryInputs(0.0, float_info.max),
    )
    huge_flow = replace(
        base,
        propellant_mass_flow=PropellantMassFlow(float_info.max),
        power_boundaries=PowerBoundaryInputs(0.0, float_info.max),
    )
    for point in (huge_voltage, huge_flow):
        with pytest.raises(PhysicsValidationError, match="nonrelativistic|representable"):
            evaluate_performance(point)
        with pytest.raises(PhysicsValidationError, match="nonrelativistic|representable"):
            evaluate_performance_warp([point], device="cpu")


@pytest.mark.parametrize("device", ["reference", "cpu", "cuda:0"])
def test_max_mass_and_flow_preserve_representable_oracle_values(
    device: str, point_factory
) -> None:
    if device != "reference" and not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    base = point_factory(
        voltage_v=400.0,
        neutral=0.0,
        plus=1.0,
        double_plus=0.0,
        beam_factor=1.0,
        divergence_factor=1.0,
        cathode_power_w=0.0,
    )
    point = replace(
        base,
        propellant_mass_flow=PropellantMassFlow(float_info.max),
        xenon_atom_mass_kg=float_info.max,
        power_boundaries=PowerBoundaryInputs(0.0, 1.0),
    )
    result = (
        evaluate_performance(point)
        if device == "reference"
        else evaluate_performance_warp([point], device=device).results[0]
    )
    with localcontext() as context:
        context.prec = 100
        expected_speed = (
            Decimal(2)
            * Decimal(str(ELEMENTARY_CHARGE_C))
            * Decimal(400)
            / Decimal.from_float(float_info.max)
        ).sqrt()
        expected_thrust = Decimal.from_float(float_info.max) * expected_speed
    assert result.total_xenon_particle_rate_per_s == 1.0
    assert result.xe_plus_speed_m_per_s > 0.0
    assert result.power_budget.beam_current_a == ELEMENTARY_CHARGE_C
    assert abs(
        Decimal(str(result.xe_plus_speed_m_per_s)) / expected_speed - 1
    ) < Decimal("8e-16")
    assert abs(
        Decimal(str(result.axial_thrust_n)) / expected_thrust - 1
    ) < Decimal("8e-16")


def test_every_accepted_adversarial_point_publishes_only_finite_values(
    point_factory,
) -> None:
    points = [
        point_factory(voltage_v=1.0e-300, mass_flow_kg_per_s=1.0e-300),
        point_factory(voltage_v=1.0e-200, mass_flow_kg_per_s=1.0e200),
        point_factory(voltage_v=1.0e6, mass_flow_kg_per_s=1.0e-300),
        point_factory(neutral=1.0, plus=0.0, double_plus=0.0),
    ]
    for result in evaluate_batch(points):
        assert all(isfinite(value) for value in _all_float_values(result))


def test_tiny_specific_impulse_matches_high_precision_oracle(point_factory) -> None:
    point = point_factory(
        voltage_v=1.0e-300,
        mass_flow_kg_per_s=1.0e-300,
        neutral=0.0,
        plus=1.0,
        double_plus=0.0,
        beam_factor=1.0,
        divergence_factor=0.75,
        cathode_power_w=0.0,
        ppu_margin_w=0.0,
    )
    result = evaluate_performance(point)
    with localcontext() as context:
        context.prec = 100
        speed = (
            Decimal(2)
            * Decimal(str(ELEMENTARY_CHARGE_C))
            * Decimal("1e-300")
            / Decimal(str(point.xenon_atom_mass_kg))
        ).sqrt()
        expected = (
            Decimal("0.75")
            * speed
            / Decimal(str(STANDARD_GRAVITY_M_PER_S2))
        )
        relative_error = abs(
            Decimal(str(result.specific_impulse_s)) / expected - Decimal(1)
        )
    assert result.specific_impulse_s > 0.0
    assert result.power_budget.beam_current_a > 0.0
    assert relative_error < Decimal("5e-16")


@pytest.mark.parametrize("device", ["reference", "cpu", "cuda:0"])
def test_underflowed_zero_power_efficiency_is_explicitly_undefined(
    device: str, point_factory
) -> None:
    if device != "reference" and not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    point = point_factory(
        voltage_v=1.0e-300,
        mass_flow_kg_per_s=1.0e-300,
        neutral=0.0,
        plus=1.0,
        double_plus=0.0,
        beam_factor=1.0,
        divergence_factor=1.0,
        cathode_power_w=0.0,
        ppu_margin_w=0.0,
    )
    result = (
        evaluate_performance(point)
        if device == "reference"
        else evaluate_performance_warp([point], device=device).results[0]
    )
    budget = result.power_budget
    assert budget.beam_kinetic_power_w == 0.0
    assert budget.anode_input_power_w == 0.0
    assert budget.ppu_input_power_w == 0.0
    assert budget.ppu_conversion_loss_w == 0.0
    assert budget.anode_to_beam_efficiency is None
    assert budget.thruster_electrical_to_beam_efficiency is None
    assert budget.ppu_input_to_beam_efficiency is None


def test_positive_ppu_loss_and_efficiency_share_effective_budget(
    point_factory,
) -> None:
    base = point_factory(ppu_margin_w=0.0)
    required = prepare_operating_point(base).thruster_power
    point = _with_ppu(base, required + 100.0)
    budget = evaluate_performance(point).power_budget
    assert budget.requested_ppu_input_power_w == required + 100.0
    assert budget.ppu_input_power_w == required + 100.0
    assert budget.ppu_boundary_adjustment_w == 0.0
    assert budget.ppu_conversion_loss_w == budget.ppu_input_power_w - required
    assert budget.ppu_input_to_beam_efficiency == (
        budget.beam_kinetic_power_w / budget.ppu_input_power_w
    )


@pytest.mark.parametrize("device", ["reference", "cpu", "cuda:0"])
def test_representable_subnormal_beam_power_is_not_lost(
    device: str, point_factory
) -> None:
    if device != "reference" and not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    point = point_factory(
        voltage_v=1.0e-20,
        mass_flow_kg_per_s=1.0e-300,
        neutral=0.0,
        plus=1.0,
        double_plus=0.0,
        beam_factor=1.0,
        divergence_factor=1.0,
        cathode_power_w=0.0,
        ppu_margin_w=0.0,
    )
    result = (
        evaluate_performance(point)
        if device == "reference"
        else evaluate_performance_warp([point], device=device).results[0]
    )
    with localcontext() as context:
        context.prec = 100
        expected_decimal = (
            Decimal("1e-20")
            * Decimal("1e-300")
            * Decimal(str(ELEMENTARY_CHARGE_C))
            / Decimal(str(point.xenon_atom_mass_kg))
        )
    actual = result.power_budget.beam_kinetic_power_w
    assert 0.0 < actual < float_info.min
    assert abs(actual / float(expected_decimal) - 1.0) <= 2.0e-9


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_tiny_warp_observables_use_relative_not_absolute_parity(
    device: str, point_factory
) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    point = point_factory(
        voltage_v=1.0e-300,
        mass_flow_kg_per_s=1.0e-300,
        neutral=0.0,
        plus=1.0,
        double_plus=0.0,
        beam_factor=1.0,
        divergence_factor=0.75,
        cathode_power_w=0.0,
        ppu_margin_w=0.0,
    )
    reference = evaluate_performance(point)
    actual = evaluate_performance_warp([point], device=device).results[0]
    for actual_value, expected in (
        (actual.specific_impulse_s, reference.specific_impulse_s),
        (actual.xe_plus_speed_m_per_s, reference.xe_plus_speed_m_per_s),
        (actual.power_budget.beam_current_a, reference.power_budget.beam_current_a),
    ):
        assert actual_value > 0.0
        assert abs(actual_value / expected - 1.0) <= 3.0e-15


class _MismatchedSequence:
    def __len__(self):
        return 2

    def __iter__(self):
        return iter(())


@pytest.mark.parametrize("backend", ["reference", "warp"])
def test_batch_shape_errors_are_always_typed(
    backend: str, point_factory
) -> None:
    point = point_factory()

    def call(values):
        if backend == "reference":
            return evaluate_batch(values)
        return evaluate_performance_warp(values, device="cpu")

    invalid_values = (
        point,
        [[point], [point]],
        [[point], []],
        _MismatchedSequence(),
    )
    for values in invalid_values:
        with pytest.raises(PhysicsValidationError):
            call(values)

    numpy = pytest.importorskip("numpy")
    for values in (
        numpy.array(point, dtype=object),
        numpy.array([[point], [point]], dtype=object),
    ):
        with pytest.raises(PhysicsValidationError):
            call(values)
