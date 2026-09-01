"""Shared binary64 validation and stable preparation for Python and Warp."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import frexp, fsum, isfinite, ldexp, sqrt, ulp
from typing import Sequence

from .models import (
    ELEMENTARY_CHARGE_C,
    STANDARD_GRAVITY_M_PER_S2,
    PhysicsValidationError,
    XenonOperatingPoint,
)

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
MAX_L0_SPEED_FRACTION_OF_LIGHT = 0.01
PPU_BOUNDARY_TOLERANCE_ULPS = 4


@dataclass(frozen=True, slots=True)
class PreparedOperatingPoint:
    total_rate: float
    neutral_rate: float
    plus_rate: float
    double_plus_rate: float
    plus_speed: float
    double_plus_speed: float
    momentum_velocity: float
    undiverged_thrust: float
    axial_thrust: float
    specific_impulse: float
    beam_current: float
    anode_current: float
    beam_power: float
    particle_beam_power: float
    anode_power: float
    thruster_power: float
    requested_ppu_input: float
    effective_ppu_input: float
    ppu_boundary_adjustment: float
    ppu_loss: float
    anode_efficiency: float | None
    thruster_efficiency: float | None
    ppu_efficiency: float | None


def validate_point_batch(
    points: Sequence[XenonOperatingPoint],
) -> tuple[XenonOperatingPoint, ...]:
    """Apply one shape/type contract to both reference and Warp batches."""

    if isinstance(points, XenonOperatingPoint):
        raise PhysicsValidationError("points must be a one-dimensional sequence")
    try:
        declared_count = len(points)
    except (TypeError, AttributeError) as error:
        raise PhysicsValidationError(
            "points must be a one-dimensional sequence"
        ) from error
    try:
        shape = getattr(points, "shape", None)
    except (TypeError, AttributeError) as error:
        raise PhysicsValidationError(
            "points must expose a valid one-dimensional shape"
        ) from error
    if shape is not None:
        try:
            dimensions = len(shape)
        except (TypeError, AttributeError) as error:
            raise PhysicsValidationError(
                "points must expose a valid one-dimensional shape"
            ) from error
        if dimensions != 1:
            raise PhysicsValidationError("points must be a one-dimensional sequence")
    if declared_count == 0:
        raise PhysicsValidationError("operating-point batch cannot be empty")
    try:
        validated = tuple(points)
    except (TypeError, AttributeError) as error:
        raise PhysicsValidationError(
            "points must be an iterable one-dimensional sequence"
        ) from error
    if len(validated) != declared_count:
        raise PhysicsValidationError(
            "operating-point batch length does not match its iterable contents"
        )
    if any(not isinstance(point, XenonOperatingPoint) for point in validated):
        raise PhysicsValidationError(
            "every batch entry must be a XenonOperatingPoint; nested/ragged inputs "
            "are not supported"
        )
    return validated


def canonical_ppu_budget(
    requested_w: float, required_w: float
) -> tuple[float, float, float]:
    """Return effective input, loss, and adjustment under a four-ULP contract."""

    scale_ulp = max(ulp(requested_w), ulp(required_w))
    tolerance = (
        Fraction.from_float(scale_ulp) * PPU_BOUNDARY_TOLERANCE_ULPS
    )
    exact_difference = (
        Fraction.from_float(requested_w) - Fraction.from_float(required_w)
    )
    if abs(exact_difference) <= tolerance:
        return required_w, 0.0, required_w - requested_w
    if exact_difference < 0:
        raise PhysicsValidationError(
            "ppu_input_power_w is below anode plus cathode power by more than "
            "the four-ULP IEEE-754 regrouping tolerance"
        )
    return requested_w, float(exact_difference), 0.0


def _scaled_product(name: str, *values: float) -> float:
    """Multiply finite binary64 values without avoidable intermediate collapse."""

    if any(value == 0.0 for value in values):
        return 0.0
    mantissa = 1.0
    exponent = 0
    for value in values:
        value_mantissa, value_exponent = frexp(value)
        mantissa *= value_mantissa
        exponent += value_exponent
        mantissa, adjustment = frexp(mantissa)
        exponent += adjustment
    try:
        result = ldexp(mantissa, exponent)
    except OverflowError as error:
        raise PhysicsValidationError(
            f"operating point produces non-representable {name}"
        ) from error
    return _require_finite_derived(name, result)


def _scaled_ratio(name: str, numerator: float, denominator: float) -> float:
    """Divide positive finite values using exponent-separated arithmetic."""

    numerator_mantissa, numerator_exponent = frexp(numerator)
    denominator_mantissa, denominator_exponent = frexp(denominator)
    try:
        result = ldexp(
            numerator_mantissa / denominator_mantissa,
            numerator_exponent - denominator_exponent,
        )
    except OverflowError as error:
        raise PhysicsValidationError(
            f"operating point produces non-representable {name}"
        ) from error
    return _require_finite_derived(name, result)


def _require_finite_derived(name: str, value: float) -> float:
    if not isfinite(value):
        raise PhysicsValidationError(
            f"operating point produces non-representable {name}"
        )
    return value


def prepare_operating_point(point: XenonOperatingPoint) -> PreparedOperatingPoint:
    """Validate and stably derive every scalar used by either backend."""

    if not isinstance(point, XenonOperatingPoint):
        raise PhysicsValidationError("point must be a XenonOperatingPoint")
    mass_flow = point.propellant_mass_flow.kg_per_s
    mass = point.xenon_atom_mass_kg
    voltage = point.discharge_voltage_v
    fractions = point.charge_state_fractions
    factors = point.beam_divergence_factors

    total_rate = _scaled_ratio("total xenon particle rate", mass_flow, mass)
    neutral_rate = _scaled_product(
        "neutral particle rate", fractions.xe_neutral, total_rate
    )
    plus_rate = _scaled_product(
        "Xe+ particle rate", fractions.xe_plus, total_rate
    )
    double_rate = _scaled_product(
        "Xe2+ particle rate", fractions.xe_double_plus, total_rate
    )

    speed_coefficient = _scaled_ratio(
        "acceleration coefficient",
        sqrt(2.0 * ELEMENTARY_CHARGE_C),
        sqrt(mass),
    )
    plus_speed = _require_finite_derived(
        "Xe+ speed",
        _scaled_product("Xe+ speed", sqrt(voltage), speed_coefficient),
    )
    double_speed = _scaled_product(
        "Xe2+ speed", sqrt(2.0), plus_speed
    )
    if double_speed > MAX_L0_SPEED_FRACTION_OF_LIGHT * SPEED_OF_LIGHT_M_PER_S:
        raise PhysicsValidationError(
            "discharge voltage is outside the L0 nonrelativistic domain: "
            "Xe2+ speed must not exceed 0.01 c"
        )

    momentum_velocity = _require_finite_derived(
        "charge-state-weighted exhaust speed",
        fsum(
            (
                fractions.xe_plus * plus_speed,
                fractions.xe_double_plus * double_speed,
            )
        ),
    )
    undiverged_thrust = _scaled_product(
        "undiverged thrust", mass_flow, momentum_velocity
    )
    axial_thrust = _scaled_product(
        "axial thrust",
        factors.axial_momentum_fraction_of_ion_momentum,
        undiverged_thrust,
    )
    specific_impulse = _scaled_ratio(
        "specific impulse",
        _scaled_product(
            "divergence-corrected exhaust speed",
            factors.axial_momentum_fraction_of_ion_momentum,
            momentum_velocity,
        ),
        STANDARD_GRAVITY_M_PER_S2,
    )

    charge_weight = fractions.charge_weighted_ion_fraction
    beam_current = _scaled_product(
        "beam current", total_rate, ELEMENTARY_CHARGE_C, charge_weight
    )
    anode_current = _scaled_ratio(
        "anode current", beam_current,
        factors.beam_current_fraction_of_anode_current,
    )
    beam_power = _scaled_product(
        "beam kinetic power", voltage, beam_current
    )
    plus_power = _scaled_product(
        "Xe+ kinetic power", plus_rate, ELEMENTARY_CHARGE_C, voltage
    )
    double_power = _scaled_product(
        "Xe2+ kinetic power",
        double_rate,
        2.0 * ELEMENTARY_CHARGE_C,
        voltage,
    )
    particle_beam_power = _require_finite_derived(
        "particle kinetic power",
        fsum((plus_power, double_power)),
    )
    anode_power = _scaled_product(
        "anode input power", voltage, anode_current
    )
    cathode_power = point.power_boundaries.cathode_input_power_w
    thruster_power = _require_finite_derived(
        "thruster electrical input power", fsum((anode_power, cathode_power))
    )
    requested_ppu_input = point.power_boundaries.ppu_input_power_w
    effective_ppu_input, ppu_loss, ppu_adjustment = canonical_ppu_budget(
        requested_ppu_input, thruster_power
    )

    anode_efficiency = (
        beam_power / anode_power if anode_power > 0.0 else None
    )
    thruster_efficiency = (
        beam_power / thruster_power if thruster_power > 0.0 else None
    )
    ppu_efficiency = (
        beam_power / effective_ppu_input
        if effective_ppu_input > 0.0
        else None
    )

    for name, value in (
        ("anode-to-beam efficiency", anode_efficiency),
        ("thruster-to-beam efficiency", thruster_efficiency),
        ("PPU-to-beam efficiency", ppu_efficiency),
    ):
        if value is not None:
            _require_finite_derived(name, value)

    return PreparedOperatingPoint(
        total_rate=total_rate,
        neutral_rate=neutral_rate,
        plus_rate=plus_rate,
        double_plus_rate=double_rate,
        plus_speed=plus_speed,
        double_plus_speed=double_speed,
        momentum_velocity=momentum_velocity,
        undiverged_thrust=undiverged_thrust,
        axial_thrust=axial_thrust,
        specific_impulse=specific_impulse,
        beam_current=beam_current,
        anode_current=anode_current,
        beam_power=beam_power,
        particle_beam_power=particle_beam_power,
        anode_power=anode_power,
        thruster_power=thruster_power,
        requested_ppu_input=requested_ppu_input,
        effective_ppu_input=effective_ppu_input,
        ppu_boundary_adjustment=ppu_adjustment,
        ppu_loss=ppu_loss,
        anode_efficiency=anode_efficiency,
        thruster_efficiency=thruster_efficiency,
        ppu_efficiency=ppu_efficiency,
    )
