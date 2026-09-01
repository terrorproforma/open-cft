"""Immutable, SI-explicit models for the verified L0 xenon performance layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import fsum, isfinite, nextafter, ulp

ELEMENTARY_CHARGE_C = 1.602176634e-19
STANDARD_GRAVITY_M_PER_S2 = 9.80665
XENON_ATOM_MASS_KG = 2.180171556711138e-25
FRACTION_SUM_TOLERANCE_ULPS = 2


class PhysicsError(Exception):
    """Base class for errors from the modern physics workstream."""


class PhysicsValidationError(PhysicsError, ValueError):
    """An input violates a documented physics-domain invariant."""


class OptionalDependencyError(PhysicsError, RuntimeError):
    """An explicitly requested optional implementation is unavailable."""


class PhysicsDeviceError(PhysicsError, RuntimeError):
    """An explicitly requested compute device is unavailable or invalid."""


def _finite(name: str, value: float) -> float:
    converted = float(value)
    if not isfinite(converted):
        raise PhysicsValidationError(f"{name} must be finite, got {value!r}")
    return converted


def _closed_unit_interval(name: str, value: float) -> float:
    converted = _finite(name, value)
    if not 0.0 <= converted <= 1.0:
        raise PhysicsValidationError(f"{name} must be in [0, 1], got {converted!r}")
    return converted


@dataclass(frozen=True, slots=True)
class PropellantMassFlow:
    """Xenon propellant mass flow at the thruster inlet, in kg/s."""

    kg_per_s: float

    def __post_init__(self) -> None:
        value = _finite("kg_per_s", self.kg_per_s)
        if value <= 0.0:
            raise PhysicsValidationError(
                "kg_per_s must be greater than zero for a running-thruster point"
            )
        object.__setattr__(self, "kg_per_s", value)


@dataclass(frozen=True, slots=True)
class ChargeStateFractions:
    """Number fractions of exhaust xenon atoms/ions; the three values sum to one."""

    xe_neutral: float
    xe_plus: float
    xe_double_plus: float

    def __post_init__(self) -> None:
        names = ("xe_neutral", "xe_plus", "xe_double_plus")
        values = list(
            _closed_unit_interval(name, value)
            for name, value in zip(
                names,
                (self.xe_neutral, self.xe_plus, self.xe_double_plus),
                strict=True,
            )
        )
        exact_values = [Fraction.from_float(value) for value in values]
        exact_total = sum(exact_values, start=Fraction(0))
        exact_tolerance = (
            Fraction.from_float(ulp(1.0)) * FRACTION_SUM_TOLERANCE_ULPS
        )
        if abs(exact_total - 1) > exact_tolerance:
            raise PhysicsValidationError(
                "xenon charge-state number fractions must sum to one within "
                "two exact binary64 ULPs"
            )
        largest = max(range(len(values)), key=values.__getitem__)
        exact_other_sum = sum(
            (
                exact_value
                for index, exact_value in enumerate(exact_values)
                if index != largest
            ),
            start=Fraction(0),
        )
        target = float(Fraction(1) - exact_other_sum)
        candidates = (
            target,
            nextafter(target, float("-inf")),
            nextafter(target, float("inf")),
        )
        values[largest] = min(
            (candidate for candidate in candidates if 0.0 <= candidate <= 1.0),
            key=lambda candidate: abs(
                exact_other_sum + Fraction.from_float(candidate) - 1
            ),
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise PhysicsValidationError(
                "fraction normalization would leave the closed unit interval"
            )
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)

    @property
    def ionized_fraction(self) -> float:
        return fsum((self.xe_plus, self.xe_double_plus))

    @property
    def charge_weighted_ion_fraction(self) -> float:
        return fsum((self.xe_plus, 2.0 * self.xe_double_plus))


@dataclass(frozen=True, slots=True)
class MassUtilization:
    """Fraction of inlet xenon mass represented by accelerated Xe+ and Xe2+."""

    fraction_of_inlet_mass: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fraction_of_inlet_mass",
            _closed_unit_interval(
                "fraction_of_inlet_mass", self.fraction_of_inlet_mass
            ),
        )

    @classmethod
    def from_charge_states(cls, fractions: ChargeStateFractions) -> MassUtilization:
        return cls(fractions.ionized_fraction)


@dataclass(frozen=True, slots=True)
class BeamDivergenceFactors:
    """Dimensionless reductions with explicitly named measurement boundaries."""

    beam_current_fraction_of_anode_current: float
    axial_momentum_fraction_of_ion_momentum: float

    def __post_init__(self) -> None:
        beam = _closed_unit_interval(
            "beam_current_fraction_of_anode_current",
            self.beam_current_fraction_of_anode_current,
        )
        axial = _closed_unit_interval(
            "axial_momentum_fraction_of_ion_momentum",
            self.axial_momentum_fraction_of_ion_momentum,
        )
        if beam == 0.0:
            raise PhysicsValidationError(
                "beam_current_fraction_of_anode_current must be greater than zero"
            )
        object.__setattr__(self, "beam_current_fraction_of_anode_current", beam)
        object.__setattr__(self, "axial_momentum_fraction_of_ion_momentum", axial)


@dataclass(frozen=True, slots=True)
class PowerBoundaryInputs:
    """Reported non-anode loads and PPU input, all in electrical watts."""

    cathode_input_power_w: float
    ppu_input_power_w: float

    def __post_init__(self) -> None:
        for name in ("cathode_input_power_w", "ppu_input_power_w"):
            value = _finite(name, getattr(self, name))
            if value < 0.0:
                raise PhysicsValidationError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class XenonOperatingPoint:
    """One L0 xenon operating point with all external quantities in SI units."""

    discharge_voltage_v: float
    propellant_mass_flow: PropellantMassFlow
    charge_state_fractions: ChargeStateFractions
    mass_utilization: MassUtilization
    beam_divergence_factors: BeamDivergenceFactors
    power_boundaries: PowerBoundaryInputs
    xenon_atom_mass_kg: float = XENON_ATOM_MASS_KG

    def __post_init__(self) -> None:
        voltage = _finite("discharge_voltage_v", self.discharge_voltage_v)
        mass = _finite("xenon_atom_mass_kg", self.xenon_atom_mass_kg)
        if voltage <= 0.0:
            raise PhysicsValidationError(
                "discharge_voltage_v must be greater than zero for a running-thruster point"
            )
        if mass <= 0.0:
            raise PhysicsValidationError("xenon_atom_mass_kg must be greater than zero")
        utilization = self.mass_utilization.fraction_of_inlet_mass
        ionized_fraction = self.charge_state_fractions.ionized_fraction
        utilization_tolerance = (
            Fraction.from_float(ulp(max(utilization, ionized_fraction)))
            * FRACTION_SUM_TOLERANCE_ULPS
        )
        utilization_difference = abs(
            Fraction.from_float(utilization)
            - Fraction.from_float(ionized_fraction)
        )
        if utilization_difference > utilization_tolerance:
            raise PhysicsValidationError(
                "mass utilization must equal the Xe+ plus Xe2+ mass fraction "
                "within two exact binary64 ULPs for this xenon representation"
            )
        object.__setattr__(self, "discharge_voltage_v", voltage)
        object.__setattr__(self, "xenon_atom_mass_kg", mass)


class ApplicabilityWarningCode(str, Enum):
    FULLY_NEUTRAL_FLOW = "fully_neutral_flow"
    MULTIPLY_CHARGED_IONS_PRESENT = "multiply_charged_ions_present"
    EMPIRICAL_FACTORS_REQUIRED = "empirical_factors_required"
    NO_INTERNAL_PLASMA_LOSSES = "no_internal_plasma_losses"


@dataclass(frozen=True, slots=True)
class ApplicabilityWarning:
    code: ApplicabilityWarningCode
    message: str


@dataclass(frozen=True, slots=True)
class ConservationDiagnostics:
    """Signed reconstruction residuals; exact solutions are zero up to roundoff."""

    particle_rate_residual_particles_per_s: float
    mass_flow_residual_kg_per_s: float
    beam_current_residual_a: float
    beam_power_residual_w: float
    ppu_power_margin_w: float


@dataclass(frozen=True, slots=True)
class ReportedPowerBudget:
    """Electrical/kinetic powers named by their physical accounting boundaries."""

    beam_current_a: float
    anode_current_a: float
    beam_kinetic_power_w: float
    anode_input_power_w: float
    cathode_input_power_w: float
    thruster_electrical_input_power_w: float
    requested_ppu_input_power_w: float
    ppu_input_power_w: float
    ppu_boundary_adjustment_w: float
    ppu_conversion_loss_w: float
    anode_to_beam_efficiency: float | None
    thruster_electrical_to_beam_efficiency: float | None
    ppu_input_to_beam_efficiency: float | None


@dataclass(frozen=True, slots=True)
class IdealPerformanceResult:
    """Conservation-based L0 result; it is not a calibrated thruster prediction."""

    total_xenon_particle_rate_per_s: float
    neutral_particle_rate_per_s: float
    xe_plus_particle_rate_per_s: float
    xe_double_plus_particle_rate_per_s: float
    xe_plus_speed_m_per_s: float
    xe_double_plus_speed_m_per_s: float
    undiverged_ion_thrust_n: float
    axial_thrust_n: float
    specific_impulse_s: float
    power_budget: ReportedPowerBudget
    diagnostics: ConservationDiagnostics
    applicability_warnings: tuple[ApplicabilityWarning, ...]
