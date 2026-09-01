"""Typed, SI-explicit contracts for the reduced Kornfeld global model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from sys import float_info
from typing import Sequence

CELL_COUNT = 4
STATE_SIZE = 25
RESIDUAL_SIZE = 28
MIN_NORMAL = float_info.min


class PlasmaError(Exception):
    """Base class for the isolated plasma workstream."""


class PlasmaValidationError(PlasmaError, ValueError):
    """An input or state violates a declared model-domain invariant."""


class PlasmaNumericsError(PlasmaError, ArithmeticError):
    """An intermediate or published value is not finite."""


class AnodeIonEnergySign(str, Enum):
    """Explicit hypotheses for a sign that is unclear in available equation images."""

    SOURCE_MINUS_SIGN = "source_minus_sign"
    OCR_PLUS_SIGN_ALTERNATIVE = "ocr_plus_sign_alternative"


def _finite(name: str, value: float) -> float:
    converted = float(value)
    if not isfinite(converted):
        raise PlasmaValidationError(f"{name} must be finite")
    return converted


def _positive(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise PlasmaValidationError(f"{name} must be greater than zero")
    return converted


def _tuple4(name: str, values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != CELL_COUNT:
        raise PlasmaValidationError(f"{name} must contain exactly four values")
    return tuple(  # type: ignore[return-value]
        _finite(f"{name}[{index}]", value) for index, value in enumerate(values)
    )


@dataclass(frozen=True, slots=True)
class XenonGlobalInputs:
    """Boundary conditions and closures for one four-cell discharge solve.

    Electron temperatures and ionization energy are expressed in electron-volts.
    Multiplication by a current therefore gives watts numerically because
    ``1 A * 1 eV/e = 1 W``.
    """

    anode_voltage_v: float
    anode_current_a: float
    cusp_arrival_probabilities: tuple[float, float, float, float]
    cathode_potential_v: float = 0.0
    cathode_electron_temperature_ev: float = 0.0
    cathode_perveance_a_per_v_3_2: float = 0.002
    xenon_ionization_energy_ev: float = 12.1
    excitation_fraction: float = 0.25
    ionization_fraction: float = 0.07
    thermalization_fraction: float = 0.68
    anode_ion_energy_sign: AnodeIonEnergySign = AnodeIonEnergySign.SOURCE_MINUS_SIGN

    def __post_init__(self) -> None:
        voltage = _positive("anode_voltage_v", self.anode_voltage_v)
        current = _positive("anode_current_a", self.anode_current_a)
        if voltage < MIN_NORMAL or current < MIN_NORMAL:
            raise PlasmaValidationError(
                "anode_voltage_v and anode_current_a residual scales must "
                "each be normal positive binary64 values"
            )
        input_power = voltage * current
        cathode_voltage_power = voltage * sqrt(voltage)
        derived_values = {
            "anode input power": input_power,
            "anode voltage upper bound": 1.5 * voltage,
            "electron-temperature upper bound": 2.0 * voltage,
            "current upper bound": 2.0 * current,
            "cathode voltage three-halves scale": cathode_voltage_power,
        }
        if any(not isfinite(value) for value in derived_values.values()):
            raise PlasmaValidationError(
                "inputs produce a non-representable derived power, scale, or bound"
            )
        if input_power < MIN_NORMAL:
            raise PlasmaValidationError(
                "anode_voltage_v * anode_current_a must be a normal positive "
                "binary64 power"
            )
        if cathode_voltage_power < MIN_NORMAL:
            raise PlasmaValidationError(
                "anode_voltage_v is too small for a normal cathode "
                "voltage^(3/2) scale"
            )
        object.__setattr__(
            self,
            "anode_voltage_v",
            voltage,
        )
        object.__setattr__(
            self,
            "anode_current_a",
            current,
        )
        object.__setattr__(
            self,
            "cusp_arrival_probabilities",
            _tuple4("cusp_arrival_probabilities", self.cusp_arrival_probabilities),
        )
        for index, probability in enumerate(self.cusp_arrival_probabilities):
            if not 0.0 <= probability < 1.0:
                raise PlasmaValidationError(
                    f"cusp_arrival_probabilities[{index}] must be in [0, 1)"
                )
        cathode_potential = _finite("cathode_potential_v", self.cathode_potential_v)
        cathode_temperature = _finite(
            "cathode_electron_temperature_ev", self.cathode_electron_temperature_ev
        )
        perveance = _positive(
            "cathode_perveance_a_per_v_3_2", self.cathode_perveance_a_per_v_3_2
        )
        ionization_energy = _positive(
            "xenon_ionization_energy_ev", self.xenon_ionization_energy_ev
        )
        fractions = (
            _finite("excitation_fraction", self.excitation_fraction),
            _finite("ionization_fraction", self.ionization_fraction),
            _finite("thermalization_fraction", self.thermalization_fraction),
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise PlasmaValidationError("energy-transfer fractions must each be in [0, 1]")
        if abs(sum(fractions) - 1.0) > 8.0e-15:
            raise PlasmaValidationError("energy-transfer fractions must sum to one")
        if cathode_temperature < 0.0:
            raise PlasmaValidationError("cathode_electron_temperature_ev must be non-negative")
        if cathode_potential >= self.anode_voltage_v:
            raise PlasmaValidationError("cathode_potential_v must be below anode_voltage_v")
        if not isinstance(self.anode_ion_energy_sign, AnodeIonEnergySign):
            raise PlasmaValidationError("anode_ion_energy_sign must be AnodeIonEnergySign")
        object.__setattr__(self, "cathode_potential_v", cathode_potential)
        object.__setattr__(self, "cathode_electron_temperature_ev", cathode_temperature)
        object.__setattr__(self, "cathode_perveance_a_per_v_3_2", perveance)
        object.__setattr__(self, "xenon_ionization_energy_ev", ionization_energy)
        object.__setattr__(self, "excitation_fraction", fractions[0])
        object.__setattr__(self, "ionization_fraction", fractions[1])
        object.__setattr__(self, "thermalization_fraction", fractions[2])


@dataclass(frozen=True, slots=True)
class PlasmaState:
    """Reduced 25-variable state after eliminating unidentifiable cusp potentials."""

    plasma_potential_v: tuple[float, float, float, float]
    electron_temperature_ev: tuple[float, float, float, float]
    ionization_source_current_a: tuple[float, float, float, float]
    electron_current_a: tuple[float, float, float, float, float]
    ion_current_a: tuple[float, float, float, float, float]
    cusp_ion_current_a: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plasma_potential_v",
            _tuple4("plasma_potential_v", self.plasma_potential_v),
        )
        object.__setattr__(
            self,
            "electron_temperature_ev",
            _tuple4("electron_temperature_ev", self.electron_temperature_ev),
        )
        object.__setattr__(
            self,
            "ionization_source_current_a",
            _tuple4("ionization_source_current_a", self.ionization_source_current_a),
        )
        for name, values, expected in (
            ("electron_current_a", self.electron_current_a, 5),
            ("ion_current_a", self.ion_current_a, 5),
            ("cusp_ion_current_a", self.cusp_ion_current_a, 3),
        ):
            if len(values) != expected:
                raise PlasmaValidationError(f"{name} must contain exactly {expected} values")
            converted = tuple(
                _finite(f"{name}[{index}]", value)
                for index, value in enumerate(values)
            )
            object.__setattr__(self, name, converted)

    def to_vector(self) -> tuple[float, ...]:
        return (
            *self.plasma_potential_v,
            *self.electron_temperature_ev,
            *self.ionization_source_current_a,
            *self.electron_current_a,
            *self.ion_current_a,
            *self.cusp_ion_current_a,
        )

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> PlasmaState:
        if len(values) != STATE_SIZE:
            raise PlasmaValidationError(f"state vector must contain exactly {STATE_SIZE} values")
        vector = tuple(_finite(f"state[{index}]", value) for index, value in enumerate(values))
        return cls(
            plasma_potential_v=vector[0:4],  # type: ignore[arg-type]
            electron_temperature_ev=vector[4:8],  # type: ignore[arg-type]
            ionization_source_current_a=vector[8:12],  # type: ignore[arg-type]
            electron_current_a=vector[12:17],  # type: ignore[arg-type]
            ion_current_a=vector[17:22],  # type: ignore[arg-type]
            cusp_ion_current_a=vector[22:25],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class StateBounds:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != STATE_SIZE or len(self.upper) != STATE_SIZE:
            raise PlasmaValidationError("state bounds must match the 25-value state layout")
        if any(not isfinite(value) for value in (*self.lower, *self.upper)):
            raise PlasmaValidationError("state bounds must be finite")
        if any(low > high for low, high in zip(self.lower, self.upper, strict=True)):
            raise PlasmaValidationError("every lower bound must be <= its upper bound")


@dataclass(frozen=True, slots=True)
class PowerBalance:
    beam_power_w: float
    ionization_loss_w: float
    excitation_loss_w: float
    cusp_loss_w: float
    anode_electron_loss_w: float
    anode_ion_energy_exchange_w: float
    anode_net_power_w: float
    input_power_w: float
    closure_w: float


@dataclass(frozen=True, slots=True)
class ConservationClosures:
    interface_current_residual_a: tuple[float, float, float, float, float]
    cusp_current_residual_a: tuple[float, float, float]
    cell_energy_residual_w: tuple[float, float, float, float]
    global_energy_residual_w: float


@dataclass(frozen=True, slots=True)
class ResidualEvaluation:
    raw: tuple[float, ...]
    normalized: tuple[float, ...]
    powers: PowerBalance
    closures: ConservationClosures


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    converged: bool
    reason: str
    iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    initial_cost: float
    final_cost: float
    residual_inf_norm: float
    gradient_inf_norm: float
    damping: float
    active_bound_count: int
    feasible: bool
    finite: bool
    normalized_residuals: tuple[float, ...]
    jacobian_rank: int
    jacobian_condition_estimate: float


@dataclass(frozen=True, slots=True)
class PlasmaSolveResult:
    state: PlasmaState | None
    evaluation: ResidualEvaluation | None
    diagnostics: SolverDiagnostics


@dataclass(frozen=True, slots=True)
class PlasmaMultiStartResult:
    """All deterministic attempts plus the best strict solution or residual floor."""

    best: PlasmaSolveResult
    attempts: tuple[PlasmaSolveResult, ...]
    selected_start_index: int
    residual_floor: float
