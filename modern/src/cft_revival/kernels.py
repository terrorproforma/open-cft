"""Validated numerical kernels with an optional native implementation."""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import fmean
from typing import Sequence

from .models import (
    CuspProbabilities,
    DesignPoint,
    FieldProfile,
    LegacyPhysicsConstants,
    PerformanceResult,
    PlasmaSolution,
    ValidationError,
)

try:
    from . import _native
except ImportError:  # Pure Python is the supported development fallback.
    _native = None


def cusp_arrival_probability(low_field_t: float, high_field_t: float) -> float:
    """Return the legacy isotropic loss-cone probability.

    This is the closed form of `cusp_prob.m:186-190`, evaluated through its
    rationalized form to avoid cancellation when `B_low/B_high` is tiny.
    """

    validate_cusp_fields(low_field_t, high_field_t)
    if low_field_t == 0.0:
        return 0.0
    if _native is not None:
        return float(_native.cusp_arrival_probability(low_field_t, high_field_t))
    return cusp_arrival_probability_python(low_field_t, high_field_t)


def cusp_arrival_probability_python(low_field_t: float, high_field_t: float) -> float:
    """Dependency-free analytic reference implementation."""

    validate_cusp_fields(low_field_t, high_field_t)
    if low_field_t == 0.0:
        return 0.0
    ratio = low_field_t / high_field_t
    return 0.5 * ratio / (1.0 + sqrt(1.0 - ratio))


def validate_cusp_fields(low_field_t: float, high_field_t: float) -> None:
    """Apply the canonical scalar loss-cone input contract."""

    if not isfinite(low_field_t) or not isfinite(high_field_t):
        raise ValidationError("magnetic fields must be finite")
    if low_field_t < 0.0 or high_field_t <= 0.0:
        raise ValidationError("fields require low >= 0 and high > 0")
    if low_field_t > high_field_t:
        raise ValidationError("low field cannot exceed high field")


def cusp_arrival_probabilities(
    low_field_t: Sequence[float], high_field_t: Sequence[float]
) -> CuspProbabilities:
    if len(low_field_t) != 4 or len(high_field_t) != 4:
        raise ValidationError("exactly four low/high cusp fields are required")
    raw = tuple(
        cusp_arrival_probability(low, high)
        for low, high in zip(low_field_t, high_field_t, strict=True)
    )
    # MATLAB computes axial order c4,c3,c2,c1, then reverses it to p1..p4.
    return CuspProbabilities(*reversed(raw))


def window_mean(profile: FieldProfile, start_mm: float, end_mm: float) -> float:
    if start_mm > end_mm:
        raise ValidationError("window start cannot exceed window end")
    samples = [
        abs(field)
        for position, field in zip(
            profile.positions_mm, profile.magnitudes_t, strict=True
        )
        if start_mm < position < end_mm
    ]
    if not samples:
        raise ValidationError(f"field profile has no samples in [{start_mm}, {end_mm}] mm")
    return fmean(samples)


def legacy_cusp_fields(
    centreline: FieldProfile, wall: FieldProfile
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Reproduce the hard-coded sampling windows and cusp-4 swap.

    These windows are compatibility behavior, not validated geometry.
    """

    wall_windows = ((0.0, 0.5), (4.4, 5.0), (15.0, 15.6), (20.0, 21.0))
    centre_windows = ((0.0, 0.5), (4.4, 5.0), (15.0, 15.6), (24.0, 25.0))
    high = [window_mean(wall, *window) for window in wall_windows]
    low = [window_mean(centreline, *window) for window in centre_windows]

    # `cusp_prob.m:180-181` applies this unexplained anode special case.
    high[0] = low[0]
    low[0] = low[1]
    return tuple(low), tuple(high)


def calculate_performance(
    design: DesignPoint,
    plasma: PlasmaSolution,
    constants: LegacyPhysicsConstants = LegacyPhysicsConstants(),
) -> PerformanceResult:
    """Translate the dimensional performance calculation in Performance_est.

    The function intentionally requires an externally validated plasma
    solution. It does not decide whether the legacy residual system is valid.
    """

    design.validate()
    constants.validate()
    if not plasma.converged:
        raise ValidationError("performance requires a converged plasma solution")

    values = plasma.values
    beam_power_w = values[0]
    phi_1_v = values[5]
    phi_2_v = values[6]
    anode_power_w = float(design.anode_voltage_v) * float(design.anode_current_a)
    mass_flow_kg_s = float(design.mass_flow_sccm) * constants.sccm_to_kg_per_s
    particles_per_second = mass_flow_kg_s / constants.xenon_atom_mass_kg
    mass_utilization = (
        float(design.anode_current_a) / constants.elementary_charge_c
    ) / particles_per_second
    beam_efficiency = beam_power_w / anode_power_w
    grid_efficiency = 1.0 - phi_1_v / phi_2_v
    total_efficiency = beam_efficiency * grid_efficiency * mass_utilization

    if not 0.0 <= total_efficiency <= 1.0:
        raise ValidationError(f"total efficiency outside [0, 1]: {total_efficiency}")
    thrust_n = sqrt(2.0 * mass_flow_kg_s * anode_power_w * total_efficiency)
    specific_impulse_s = (
        thrust_n / (mass_flow_kg_s * constants.standard_gravity_m_per_s2)
    )
    outputs = (
        thrust_n,
        total_efficiency,
        specific_impulse_s,
        anode_power_w,
        beam_efficiency,
        grid_efficiency,
        mass_utilization,
    )
    if any(not isfinite(value) for value in outputs):
        raise ValidationError("performance calculation produced a non-finite value")
    return PerformanceResult(*outputs)
