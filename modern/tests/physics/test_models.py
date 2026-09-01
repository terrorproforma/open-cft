from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from cft_revival.physics import (
    BeamDivergenceFactors,
    ChargeStateFractions,
    MassUtilization,
    PhysicsValidationError,
    PowerBoundaryInputs,
    PropellantMassFlow,
    XenonOperatingPoint,
)


def test_models_are_immutable_and_si_explicit(point_factory) -> None:
    point = point_factory()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        point.discharge_voltage_v = 500.0
    assert point.propellant_mass_flow.kg_per_s == 1.0e-6
    assert point.power_boundaries.cathode_input_power_w == 5.0


@pytest.mark.parametrize(
    "fractions",
    [
        (-0.1, 1.1, 0.0),
        (0.2, 0.2, 0.2),
        (nan, 0.0, 1.0),
        (0.0, inf, 0.0),
    ],
)
def test_invalid_charge_fractions_fail(fractions: tuple[float, float, float]) -> None:
    with pytest.raises(PhysicsValidationError):
        ChargeStateFractions(*fractions)


@pytest.mark.parametrize(
    "beam,divergence",
    [(0.0, 1.0), (-0.1, 1.0), (1.1, 1.0), (1.0, -0.1), (1.0, 1.1), (nan, 1.0)],
)
def test_invalid_beam_and_divergence_factors_fail(
    beam: float, divergence: float
) -> None:
    with pytest.raises(PhysicsValidationError):
        BeamDivergenceFactors(beam, divergence)


@pytest.mark.parametrize("value", [-1.0, nan, inf])
def test_nonfinite_or_negative_scalar_inputs_fail(value: float) -> None:
    with pytest.raises(PhysicsValidationError):
        PropellantMassFlow(value)
    with pytest.raises(PhysicsValidationError):
        PowerBoundaryInputs(value, 1.0)
    with pytest.raises(PhysicsValidationError):
        PowerBoundaryInputs(1.0, value)


def test_zero_mass_flow_is_not_a_running_thruster_point() -> None:
    with pytest.raises(PhysicsValidationError, match="greater than zero"):
        PropellantMassFlow(0.0)


def test_mass_utilization_must_match_explicit_charge_states() -> None:
    fractions = ChargeStateFractions(0.2, 0.7, 0.1)
    with pytest.raises(PhysicsValidationError, match="mass utilization"):
        XenonOperatingPoint(
            discharge_voltage_v=300.0,
            propellant_mass_flow=PropellantMassFlow(1.0e-6),
            charge_state_fractions=fractions,
            mass_utilization=MassUtilization(0.9),
            beam_divergence_factors=BeamDivergenceFactors(1.0, 1.0),
            power_boundaries=PowerBoundaryInputs(0.0, 1000.0),
        )


@pytest.mark.parametrize("value", [0.0, -1.0, nan, inf])
def test_operating_point_rejects_invalid_voltage(value: float, point_factory) -> None:
    with pytest.raises(PhysicsValidationError):
        point_factory(voltage_v=value)
