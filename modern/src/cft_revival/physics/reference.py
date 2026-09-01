"""Dependency-free CPU reference for sourced L0 xenon performance relations."""

from __future__ import annotations

from math import fsum, isfinite
from typing import Sequence

from .models import (
    ELEMENTARY_CHARGE_C,
    ApplicabilityWarning,
    ApplicabilityWarningCode,
    ConservationDiagnostics,
    IdealPerformanceResult,
    PhysicsValidationError,
    ReportedPowerBudget,
    XenonOperatingPoint,
)
from .numerics import prepare_operating_point, validate_point_batch


def _warnings(point: XenonOperatingPoint) -> tuple[ApplicabilityWarning, ...]:
    warnings: list[ApplicabilityWarning] = [
        ApplicabilityWarning(
            ApplicabilityWarningCode.NO_INTERNAL_PLASMA_LOSSES,
            "L0 excludes ionization, excitation, wall, thermal, and cathode-plasma losses.",
        )
    ]
    fractions = point.charge_state_fractions
    factors = point.beam_divergence_factors
    if fractions.ionized_fraction == 0.0:
        warnings.append(
            ApplicabilityWarning(
                ApplicabilityWarningCode.FULLY_NEUTRAL_FLOW,
                "Fully neutral xenon produces no electrostatic beam in this model.",
            )
        )
    if fractions.xe_double_plus > 0.0:
        warnings.append(
            ApplicabilityWarning(
                ApplicabilityWarningCode.MULTIPLY_CHARGED_IONS_PRESENT,
                "Xe2+ uses the same xenon ion mass and twice the elementary charge.",
            )
        )
    if (
        factors.beam_current_fraction_of_anode_current != 1.0
        or factors.axial_momentum_fraction_of_ion_momentum != 1.0
    ):
        warnings.append(
            ApplicabilityWarning(
                ApplicabilityWarningCode.EMPIRICAL_FACTORS_REQUIRED,
                "Beam-current and divergence factors are external inputs, not L0 closures.",
            )
        )
    return tuple(warnings)


def evaluate_performance(point: XenonOperatingPoint) -> IdealPerformanceResult:
    """Evaluate conservation relations for one validated xenon operating point.

    Charge states accelerate independently through the supplied potential:
    ``v_z = sqrt(2 z e V / m_Xe)``. The momentum divergence factor is applied
    only to thrust. It does not alter the kinetic beam power.
    """

    prepared = prepare_operating_point(point)
    mass_flow = point.propellant_mass_flow.kg_per_s
    xenon_mass = point.xenon_atom_mass_kg
    cathode_power = point.power_boundaries.cathode_input_power_w
    ppu_input_power = point.power_boundaries.ppu_input_power_w
    reconstructed_total_rate = fsum(
        (prepared.neutral_rate, prepared.plus_rate, prepared.double_plus_rate)
    )
    reconstructed_mass_flow = xenon_mass * reconstructed_total_rate
    reconstructed_beam_current = fsum(
        (
            ELEMENTARY_CHARGE_C * prepared.plus_rate,
            2.0 * ELEMENTARY_CHARGE_C * prepared.double_plus_rate,
        )
    )
    diagnostics = ConservationDiagnostics(
        particle_rate_residual_particles_per_s=(
            reconstructed_total_rate - prepared.total_rate
        ),
        mass_flow_residual_kg_per_s=reconstructed_mass_flow - mass_flow,
        beam_current_residual_a=(
            reconstructed_beam_current - prepared.beam_current
        ),
        beam_power_residual_w=(
            prepared.particle_beam_power - prepared.beam_power
        ),
        ppu_power_margin_w=prepared.ppu_loss,
    )
    if any(
        not isfinite(value)
        for value in (
            diagnostics.particle_rate_residual_particles_per_s,
            diagnostics.mass_flow_residual_kg_per_s,
            diagnostics.beam_current_residual_a,
            diagnostics.beam_power_residual_w,
            diagnostics.ppu_power_margin_w,
        )
    ):
        raise PhysicsValidationError(
            "operating point produces non-representable conservation diagnostics"
        )
    budget = ReportedPowerBudget(
        beam_current_a=prepared.beam_current,
        anode_current_a=prepared.anode_current,
        beam_kinetic_power_w=prepared.beam_power,
        anode_input_power_w=prepared.anode_power,
        cathode_input_power_w=cathode_power,
        thruster_electrical_input_power_w=prepared.thruster_power,
        requested_ppu_input_power_w=ppu_input_power,
        ppu_input_power_w=prepared.effective_ppu_input,
        ppu_boundary_adjustment_w=prepared.ppu_boundary_adjustment,
        ppu_conversion_loss_w=prepared.ppu_loss,
        anode_to_beam_efficiency=prepared.anode_efficiency,
        thruster_electrical_to_beam_efficiency=prepared.thruster_efficiency,
        ppu_input_to_beam_efficiency=prepared.ppu_efficiency,
    )
    return IdealPerformanceResult(
        total_xenon_particle_rate_per_s=prepared.total_rate,
        neutral_particle_rate_per_s=prepared.neutral_rate,
        xe_plus_particle_rate_per_s=prepared.plus_rate,
        xe_double_plus_particle_rate_per_s=prepared.double_plus_rate,
        xe_plus_speed_m_per_s=prepared.plus_speed,
        xe_double_plus_speed_m_per_s=prepared.double_plus_speed,
        undiverged_ion_thrust_n=prepared.undiverged_thrust,
        axial_thrust_n=prepared.axial_thrust,
        specific_impulse_s=prepared.specific_impulse,
        power_budget=budget,
        diagnostics=diagnostics,
        applicability_warnings=_warnings(point),
    )


def evaluate_batch(
    points: Sequence[XenonOperatingPoint],
) -> tuple[IdealPerformanceResult, ...]:
    """Evaluate a non-empty batch deterministically in input order."""

    batch = validate_point_batch(points)
    return tuple(evaluate_performance(point) for point in batch)
