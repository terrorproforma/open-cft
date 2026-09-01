"""Typed SI contracts for topology-general plasma balance networks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from sys import float_info
from typing import TYPE_CHECKING, Sequence

from cft_revival.plasma import AnodeIonEnergySign, SolverDiagnostics, SolverOptions

if TYPE_CHECKING:
    from .topology import PlasmaChainTopology


class PlasmaNetworkError(Exception):
    """Base error for the isolated topology-general workstream."""


class NetworkValidationError(PlasmaNetworkError, ValueError):
    """A graph, input, state, or policy violates its declared contract."""


class NetworkNumericsError(PlasmaNetworkError, ArithmeticError):
    """A numerical operation cannot produce a finite auditable result."""


class PublicationPolicy(str, Enum):
    """Rank policy applied after strict numerical convergence."""

    REQUIRE_FULL_RANK = "require_full_rank"
    REPRESENT_NULLSPACE = "represent_nullspace"


def finite_value(name: str, value: float) -> float:
    converted = float(value)
    if not isfinite(converted):
        raise NetworkValidationError(f"{name} must be finite")
    return converted


def positive_value(name: str, value: float) -> float:
    converted = finite_value(name, value)
    if converted <= 0.0:
        raise NetworkValidationError(f"{name} must be greater than zero")
    return converted


def _sha256(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NetworkValidationError(f"{name} must be a lower-case SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class NetworkDimensions:
    """Dimensions derived only from the validated chain topology."""

    cell_count: int
    interior_cusp_count: int
    terminal_boundary_count: int
    state_size: int
    residual_size: int
    structural_rank: int
    structural_nullity: int

    @classmethod
    def for_cells(cls, cell_count: int) -> NetworkDimensions:
        if not isinstance(cell_count, int) or isinstance(cell_count, bool) or cell_count < 1:
            raise NetworkValidationError("cell_count must be an integer >= 1")
        return cls(
            cell_count=cell_count,
            interior_cusp_count=cell_count - 1,
            terminal_boundary_count=2,
            state_size=6 * cell_count + 1,
            residual_size=7 * cell_count,
            structural_rank=5 * cell_count + 2,
            structural_nullity=cell_count - 1,
        )


@dataclass(frozen=True, slots=True)
class NetworkInputs:
    """Boundary conditions and closures; volt/eV, ampere, and watt are explicit."""

    topology: PlasmaChainTopology
    anode_voltage_v: float
    anode_current_a: float
    anode_arrival_probability: float
    anode_arrival_standard_uncertainty: float
    anode_arrival_provenance_sha256: str
    cathode_potential_v: float = 0.0
    cathode_electron_temperature_ev: float = 0.0
    cathode_perveance_a_per_v_3_2: float = 0.002
    xenon_ionization_energy_ev: float = 12.1
    excitation_fraction: float = 0.25
    ionization_fraction: float = 0.07
    thermalization_fraction: float = 0.68
    anode_ion_energy_sign: AnodeIonEnergySign = AnodeIonEnergySign.SOURCE_MINUS_SIGN

    def __post_init__(self) -> None:
        # Avoid an import cycle while retaining a typed runtime boundary.
        from .topology import PlasmaChainTopology

        from .topology import validate_topology

        validate_topology(self.topology)
        voltage = positive_value("anode_voltage_v", self.anode_voltage_v)
        current = positive_value("anode_current_a", self.anode_current_a)
        if voltage < float_info.min or current < float_info.min:
            raise NetworkValidationError(
                "anode voltage and current scales must be normal positive binary64 values"
            )
        power = voltage * current
        voltage_power = voltage * sqrt(voltage)
        derived = (
            power,
            voltage_power,
            1.5 * voltage,
            2.0 * voltage,
            2.0 * current,
        )
        if any(not isfinite(value) for value in derived):
            raise NetworkValidationError("inputs produce a non-representable scale or bound")
        if power < float_info.min:
            raise NetworkValidationError("anode voltage times current must be a normal power")
        if voltage_power < float_info.min:
            raise NetworkValidationError("anode voltage is too small for voltage^(3/2)")
        probability = finite_value(
            "anode_arrival_probability", self.anode_arrival_probability
        )
        if not 0.0 <= probability < 1.0:
            raise NetworkValidationError("anode_arrival_probability must be in [0, 1)")
        uncertainty = finite_value(
            "anode_arrival_standard_uncertainty",
            self.anode_arrival_standard_uncertainty,
        )
        if uncertainty < 0.0:
            raise NetworkValidationError(
                "anode_arrival_standard_uncertainty must be non-negative"
            )
        cathode_potential = finite_value("cathode_potential_v", self.cathode_potential_v)
        cathode_temperature = finite_value(
            "cathode_electron_temperature_ev",
            self.cathode_electron_temperature_ev,
        )
        if cathode_potential >= voltage:
            raise NetworkValidationError("cathode_potential_v must be below anode_voltage_v")
        if cathode_temperature < 0.0:
            raise NetworkValidationError(
                "cathode_electron_temperature_ev must be non-negative"
            )
        fractions = (
            finite_value("excitation_fraction", self.excitation_fraction),
            finite_value("ionization_fraction", self.ionization_fraction),
            finite_value("thermalization_fraction", self.thermalization_fraction),
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise NetworkValidationError("energy fractions must each be in [0, 1]")
        if abs(sum(fractions) - 1.0) > 8.0e-15:
            raise NetworkValidationError("energy fractions must sum to one")
        if not isinstance(self.anode_ion_energy_sign, AnodeIonEnergySign):
            raise NetworkValidationError(
                "anode_ion_energy_sign must be AnodeIonEnergySign"
            )
        object.__setattr__(self, "anode_voltage_v", voltage)
        object.__setattr__(self, "anode_current_a", current)
        object.__setattr__(self, "anode_arrival_probability", probability)
        object.__setattr__(
            self, "anode_arrival_standard_uncertainty", uncertainty
        )
        object.__setattr__(
            self,
            "anode_arrival_provenance_sha256",
            _sha256(
                "anode_arrival_provenance_sha256",
                self.anode_arrival_provenance_sha256,
            ),
        )
        object.__setattr__(self, "cathode_potential_v", cathode_potential)
        object.__setattr__(self, "cathode_electron_temperature_ev", cathode_temperature)
        object.__setattr__(
            self,
            "cathode_perveance_a_per_v_3_2",
            positive_value(
                "cathode_perveance_a_per_v_3_2",
                self.cathode_perveance_a_per_v_3_2,
            ),
        )
        object.__setattr__(
            self,
            "xenon_ionization_energy_ev",
            positive_value(
                "xenon_ionization_energy_ev", self.xenon_ionization_energy_ev
            ),
        )
        object.__setattr__(self, "excitation_fraction", fractions[0])
        object.__setattr__(self, "ionization_fraction", fractions[1])
        object.__setattr__(self, "thermalization_fraction", fractions[2])

    @property
    def dimensions(self) -> NetworkDimensions:
        from .topology import PlasmaChainTopology

        topology = self.topology
        assert isinstance(topology, PlasmaChainTopology)
        return topology.dimensions

    @property
    def arrival_probabilities(self) -> tuple[float, ...]:
        from .topology import PlasmaChainTopology

        topology = self.topology
        assert isinstance(topology, PlasmaChainTopology)
        return tuple(cusp.loss_probability.value for cusp in topology.interior_cusps) + (
            self.anode_arrival_probability,
        )


@dataclass(frozen=True, slots=True)
class NetworkState:
    """Dynamic state layout: phi, Te, source, electron, ion, interior-cusp ion."""

    plasma_potential_v: tuple[float, ...]
    electron_temperature_ev: tuple[float, ...]
    ionization_source_current_a: tuple[float, ...]
    electron_current_a: tuple[float, ...]
    ion_current_a: tuple[float, ...]
    cusp_ion_current_a: tuple[float, ...]

    def __post_init__(self) -> None:
        cell_count = len(self.plasma_potential_v)
        dimensions = NetworkDimensions.for_cells(cell_count)
        expected = (
            ("plasma_potential_v", self.plasma_potential_v, cell_count),
            ("electron_temperature_ev", self.electron_temperature_ev, cell_count),
            (
                "ionization_source_current_a",
                self.ionization_source_current_a,
                cell_count,
            ),
            ("electron_current_a", self.electron_current_a, cell_count + 1),
            ("ion_current_a", self.ion_current_a, cell_count + 1),
            (
                "cusp_ion_current_a",
                self.cusp_ion_current_a,
                dimensions.interior_cusp_count,
            ),
        )
        for name, values, size in expected:
            if len(values) != size:
                raise NetworkValidationError(f"{name} must contain exactly {size} values")
            object.__setattr__(
                self,
                name,
                tuple(finite_value(f"{name}[{index}]", value) for index, value in enumerate(values)),
            )

    @property
    def dimensions(self) -> NetworkDimensions:
        return NetworkDimensions.for_cells(len(self.plasma_potential_v))

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
    def from_vector(cls, values: Sequence[float], cell_count: int) -> NetworkState:
        dimensions = NetworkDimensions.for_cells(cell_count)
        if len(values) != dimensions.state_size:
            raise NetworkValidationError(
                f"state vector must contain exactly {dimensions.state_size} values"
            )
        vector = tuple(finite_value(f"state[{index}]", value) for index, value in enumerate(values))
        n = cell_count
        return cls(
            plasma_potential_v=vector[0:n],
            electron_temperature_ev=vector[n : 2 * n],
            ionization_source_current_a=vector[2 * n : 3 * n],
            electron_current_a=vector[3 * n : 4 * n + 1],
            ion_current_a=vector[4 * n + 1 : 5 * n + 2],
            cusp_ion_current_a=vector[5 * n + 2 :],
        )


@dataclass(frozen=True, slots=True)
class DynamicBounds:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) == 0 or len(self.lower) != len(self.upper):
            raise NetworkValidationError("bounds must have equal nonzero length")
        if any(not isfinite(value) for value in (*self.lower, *self.upper)):
            raise NetworkValidationError("bounds must be finite")
        if any(low > high for low, high in zip(self.lower, self.upper, strict=True)):
            raise NetworkValidationError("every lower bound must be <= its upper bound")


@dataclass(frozen=True, slots=True)
class NetworkPowerBalance:
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
class NetworkClosures:
    electron_continuity_a: tuple[float, ...]
    ion_continuity_a: tuple[float, ...]
    interface_current_a: tuple[float, ...]
    cusp_current_a: tuple[float, ...]
    cell_energy_w: tuple[float, ...]
    global_energy_w: float


@dataclass(frozen=True, slots=True)
class NetworkResidualEvaluation:
    raw: tuple[float, ...]
    normalized: tuple[float, ...]
    equation_ids: tuple[str, ...]
    scales: tuple[float, ...]
    powers: NetworkPowerBalance
    closures: NetworkClosures


@dataclass(frozen=True, slots=True)
class IdentifiabilityDiagnostics:
    numerical_rank: int
    state_size: int
    nullity: int
    structural_rank: int
    structural_nullity: int
    condition_estimate: float
    rank_relative_tolerance: float
    nullspace_residual_tolerance: float
    max_nullspace_residual: float
    max_orthonormality_error: float
    variable_scales: tuple[float, ...]
    nullspace_basis: tuple[tuple[float, ...], ...]
    basis_valid: bool
    expected_rank: bool
    represented: bool

    @property
    def rank_deficient(self) -> bool:
        return self.numerical_rank < self.state_size


@dataclass(frozen=True, slots=True)
class NetworkSolveDiagnostics:
    numerical_converged: bool
    published: bool
    reason: str
    residual_inf_norm: float
    conservation_inf_norm: float
    feasible: bool
    deterministic_start_index: int
    equation_residuals: tuple[tuple[str, float], ...]
    identifiability: IdentifiabilityDiagnostics | None
    backend: SolverDiagnostics | None


@dataclass(frozen=True, slots=True)
class NetworkSolveResult:
    state: NetworkState | None
    evaluation: NetworkResidualEvaluation | None
    diagnostics: NetworkSolveDiagnostics


@dataclass(frozen=True, slots=True)
class NetworkMultiStartResult:
    best: NetworkSolveResult
    attempts: tuple[NetworkSolveResult, ...]
    selected_start_index: int
    residual_floor: float


@dataclass(frozen=True, slots=True)
class NetworkSolverOptions:
    least_squares: SolverOptions = SolverOptions()
    publication_policy: PublicationPolicy = PublicationPolicy.REQUIRE_FULL_RANK
    rank_relative_tolerance: float = 1.0e-11
    nullspace_residual_tolerance: float = 1.0e-10
    nullspace_orthonormality_tolerance: float = 1.0e-10
    conservation_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if not isinstance(self.least_squares, SolverOptions):
            raise NetworkValidationError("least_squares must be SolverOptions")
        if not isinstance(self.publication_policy, PublicationPolicy):
            raise NetworkValidationError("publication_policy must be PublicationPolicy")
        for name in (
            "rank_relative_tolerance",
            "nullspace_residual_tolerance",
            "nullspace_orthonormality_tolerance",
            "conservation_tolerance",
        ):
            value = finite_value(name, getattr(self, name))
            if value <= 0.0:
                raise NetworkValidationError(f"{name} must be positive")
            object.__setattr__(self, name, value)
