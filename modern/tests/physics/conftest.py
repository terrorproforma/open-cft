from __future__ import annotations

from collections.abc import Callable
from math import fsum

import pytest

from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    XENON_ATOM_MASS_KG,
    BeamDivergenceFactors,
    ChargeStateFractions,
    MassUtilization,
    PowerBoundaryInputs,
    PropellantMassFlow,
    XenonOperatingPoint,
)


@pytest.fixture
def point_factory() -> Callable[..., XenonOperatingPoint]:
    def make_point(
        *,
        voltage_v: float = 400.0,
        mass_flow_kg_per_s: float = 1.0e-6,
        neutral: float = 0.2,
        plus: float = 0.7,
        double_plus: float = 0.1,
        beam_factor: float = 0.85,
        divergence_factor: float = 0.9,
        cathode_power_w: float = 5.0,
        ppu_margin_w: float = 20.0,
        ppu_input_power_w: float | None = None,
        xenon_mass_kg: float = XENON_ATOM_MASS_KG,
    ) -> XenonOperatingPoint:
        fractions = ChargeStateFractions(neutral, plus, double_plus)
        beam_current = mass_flow_kg_per_s * (
            ELEMENTARY_CHARGE_C / xenon_mass_kg
        ) * fractions.charge_weighted_ion_fraction
        anode_power = (
            voltage_v * beam_current / beam_factor
        )
        required_ppu_power = fsum((anode_power, cathode_power_w))
        return XenonOperatingPoint(
            discharge_voltage_v=voltage_v,
            propellant_mass_flow=PropellantMassFlow(mass_flow_kg_per_s),
            charge_state_fractions=fractions,
            mass_utilization=MassUtilization.from_charge_states(fractions),
            beam_divergence_factors=BeamDivergenceFactors(
                beam_factor, divergence_factor
            ),
            power_boundaries=PowerBoundaryInputs(
                cathode_input_power_w=cathode_power_w,
                ppu_input_power_w=(
                    required_ppu_power + ppu_margin_w
                    if ppu_input_power_w is None
                    else ppu_input_power_w
                ),
            ),
            xenon_atom_mass_kg=xenon_mass_kg,
        )

    return make_point
