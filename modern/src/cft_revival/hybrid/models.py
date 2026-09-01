"""SI-explicit data contracts for the prescribed-field L2 hybrid first slice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Iterable

ELEMENTARY_CHARGE_C = 1.602176634e-19
XENON_ATOM_MASS_KG = 2.180171556711138e-25
BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23

Vec3 = tuple[float, float, float]


class HybridError(Exception):
    """Base error for the isolated hybrid workstream."""


class HybridValidationError(HybridError, ValueError):
    """An input violates a documented hybrid-slice invariant."""


class HybridOptionalDependencyError(HybridError, RuntimeError):
    """An explicitly requested optional backend is unavailable."""


class HybridDeviceError(HybridError, RuntimeError):
    """An explicitly requested backend device is invalid or unavailable."""


def finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HybridValidationError(f"{name} must be a real finite number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise HybridValidationError(
            f"{name} must be a real finite number"
        ) from error
    if not isfinite(converted):
        raise HybridValidationError(f"{name} must be finite")
    return converted


def finite_vec3(name: str, value: Vec3) -> Vec3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise HybridValidationError(f"{name} must be a three-tuple")
    return tuple(  # type: ignore[return-value]
        finite_scalar(f"{name}[{index}]", entry)
        for index, entry in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class XenonSpecies:
    """One represented heavy species; electron mass is outside this model."""

    symbol: str
    charge_state: int
    mass_kg: float = XENON_ATOM_MASS_KG
    identifier: str | None = None
    charge_c_override: float | None = None

    def __post_init__(self) -> None:
        if self.symbol not in {"Xe", "Xe+", "Xe2+"}:
            raise HybridValidationError("symbol must be Xe, Xe+, or Xe2+")
        if (
            type(self.charge_state) is not int
            or self.charge_state not in {0, 1, 2}
        ):
            raise HybridValidationError("charge_state must be 0, 1, or 2")
        expected = ("Xe", "Xe+", "Xe2+")[self.charge_state]
        if self.symbol != expected:
            raise HybridValidationError("symbol and charge_state are inconsistent")
        mass = finite_scalar("mass_kg", self.mass_kg)
        if mass <= 0.0:
            raise HybridValidationError("mass_kg must be positive")
        identifier = self.symbol if self.identifier is None else self.identifier
        if not isinstance(identifier, str) or not identifier.strip():
            raise HybridValidationError("species identifier must be non-empty")
        expected_charge = self.charge_state * ELEMENTARY_CHARGE_C
        if self.charge_c_override is not None:
            supplied_charge = finite_scalar(
                "charge_c_override", self.charge_c_override
            )
            if supplied_charge != expected_charge:
                raise HybridValidationError(
                    "xenon charge must equal charge_state * elementary charge"
                )
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "charge_c_override", expected_charge)

    @property
    def charge_c(self) -> float:
        assert self.charge_c_override is not None
        return self.charge_c_override


XE = XenonSpecies("Xe", 0)
XE_PLUS = XenonSpecies("Xe+", 1)
XE_DOUBLE_PLUS = XenonSpecies("Xe2+", 2)


class VelocityTimeLevel(str, Enum):
    """Time represented by velocity relative to position x^n."""

    SYNCHRONOUS_N = "synchronous_n"
    LEAPFROG_N_MINUS_HALF = "leapfrog_n_minus_one_half"


@dataclass(frozen=True, slots=True)
class Particle:
    """Weighted xenon state with an explicit position/velocity time level."""

    particle_id: int
    species: XenonSpecies
    position_m: Vec3
    velocity_m_per_s: Vec3
    weight: float = 1.0
    alive: bool = True
    velocity_time_level: VelocityTimeLevel = VelocityTimeLevel.LEAPFROG_N_MINUS_HALF

    def __post_init__(self) -> None:
        if (
            type(self.particle_id) is not int
            or not 0 <= self.particle_id < 1 << 64
        ):
            raise HybridValidationError(
                "particle_id must be an unsigned 64-bit integer"
            )
        if not isinstance(self.species, XenonSpecies):
            raise HybridValidationError("species must be a XenonSpecies")
        object.__setattr__(self, "position_m", finite_vec3("position_m", self.position_m))
        object.__setattr__(
            self,
            "velocity_m_per_s",
            finite_vec3("velocity_m_per_s", self.velocity_m_per_s),
        )
        weight = finite_scalar("weight", self.weight)
        if weight <= 0.0:
            raise HybridValidationError("weight must be positive")
        if not isinstance(self.alive, bool):
            raise HybridValidationError("alive must be a boolean")
        if not isinstance(self.velocity_time_level, VelocityTimeLevel):
            raise HybridValidationError(
                "velocity_time_level must be a VelocityTimeLevel"
            )
        object.__setattr__(self, "weight", weight)

    @property
    def represented_charge_c(self) -> float:
        return self.weight * self.species.charge_c

    @property
    def represented_mass_kg(self) -> float:
        return self.weight * self.species.mass_kg


def validated_particle_batch(
    particles: Iterable[Particle],
    *,
    canonical_order: bool = False,
) -> tuple[Particle, ...]:
    """Validate particle and RNG identity before a batch operation."""

    batch = tuple(particles)
    if any(not isinstance(particle, Particle) for particle in batch):
        raise HybridValidationError("all entries must be Particle instances")
    identifiers = [particle.particle_id for particle in batch]
    if len(set(identifiers)) != len(identifiers):
        raise HybridValidationError("particle_id values must be unique")
    species_identity: dict[str, XenonSpecies] = {}
    for particle in batch:
        identifier = particle.species.identifier
        assert identifier is not None
        previous = species_identity.setdefault(identifier, particle.species)
        if previous != particle.species:
            raise HybridValidationError(
                "one species identifier cannot describe different properties"
            )
    return (
        tuple(sorted(batch, key=lambda particle: particle.particle_id))
        if canonical_order
        else batch
    )


@dataclass(frozen=True, slots=True)
class BorisStepResult:
    """One leapfrog step and work-energy diagnostics at half velocity levels."""

    particle: Particle
    electric_work_j: float
    kinetic_energy_delta_j: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "electric_work_j",
            finite_scalar("electric_work_j", self.electric_work_j),
        )
        object.__setattr__(
            self,
            "kinetic_energy_delta_j",
            finite_scalar("kinetic_energy_delta_j", self.kinetic_energy_delta_j),
        )

    @property
    def work_energy_residual_j(self) -> float:
        return self.kinetic_energy_delta_j - self.electric_work_j


@dataclass(frozen=True, slots=True)
class UniformFields:
    """Externally prescribed, spatially uniform electromagnetic fields."""

    electric_v_per_m: Vec3 = (0.0, 0.0, 0.0)
    magnetic_t: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "electric_v_per_m", finite_vec3("electric_v_per_m", self.electric_v_per_m)
        )
        object.__setattr__(self, "magnetic_t", finite_vec3("magnetic_t", self.magnetic_t))


class BoundaryPolicy(str, Enum):
    PERIODIC = "periodic"
    REFLECTING = "reflecting"
    ABSORBING = "absorbing"


@dataclass(frozen=True, slots=True)
class AxisAlignedBox:
    lower_m: Vec3
    upper_m: Vec3
    policy: BoundaryPolicy

    def __post_init__(self) -> None:
        lower = finite_vec3("lower_m", self.lower_m)
        upper = finite_vec3("upper_m", self.upper_m)
        if any(high <= low for low, high in zip(lower, upper, strict=True)):
            raise HybridValidationError("each box upper bound must exceed its lower bound")
        if not isinstance(self.policy, BoundaryPolicy):
            raise HybridValidationError("policy must be a BoundaryPolicy")
        object.__setattr__(self, "lower_m", lower)
        object.__setattr__(self, "upper_m", upper)


@dataclass(frozen=True, slots=True)
class CartesianGrid1D:
    """Cell-centred 1-D mesh with a represented transverse area."""

    x_min_m: float
    x_max_m: float
    cell_count: int
    transverse_area_m2: float = 1.0

    def __post_init__(self) -> None:
        lower = finite_scalar("x_min_m", self.x_min_m)
        upper = finite_scalar("x_max_m", self.x_max_m)
        area = finite_scalar("transverse_area_m2", self.transverse_area_m2)
        if upper <= lower:
            raise HybridValidationError("x_max_m must exceed x_min_m")
        if not isinstance(self.cell_count, int) or self.cell_count < 2:
            raise HybridValidationError("cell_count must be an integer >= 2")
        if area <= 0.0:
            raise HybridValidationError("transverse_area_m2 must be positive")
        object.__setattr__(self, "x_min_m", lower)
        object.__setattr__(self, "x_max_m", upper)
        object.__setattr__(self, "transverse_area_m2", area)

    @property
    def spacing_m(self) -> float:
        return (self.x_max_m - self.x_min_m) / self.cell_count

    @property
    def cell_volume_m3(self) -> float:
        return self.spacing_m * self.transverse_area_m2


@dataclass(frozen=True, slots=True)
class DepositedMoments:
    """Cell densities whose volume integrals reconstruct particle totals."""

    number_per_m3: tuple[float, ...]
    charge_c_per_m3: tuple[float, ...]
    current_a_per_m2: tuple[Vec3, ...]
    momentum_kg_per_m2_s: tuple[Vec3, ...]
    kinetic_energy_j_per_m3: tuple[float, ...]
    position_time_level: str = "n"
    velocity_time_level: VelocityTimeLevel = (
        VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
    )

    def __post_init__(self) -> None:
        lengths = {
            len(self.number_per_m3),
            len(self.charge_c_per_m3),
            len(self.current_a_per_m2),
            len(self.momentum_kg_per_m2_s),
            len(self.kinetic_energy_j_per_m3),
        }
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise HybridValidationError(
                "deposited moment arrays must have one common length >= 2"
            )
        if self.position_time_level != "n":
            raise HybridValidationError("position_time_level must be n")
        if (
            self.velocity_time_level
            is not VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
        ):
            raise HybridValidationError(
                "deposition velocity_time_level must be leapfrog_n_minus_one_half"
            )
        number = tuple(
            finite_scalar(f"number_per_m3[{index}]", value)
            for index, value in enumerate(self.number_per_m3)
        )
        charge = tuple(
            finite_scalar(f"charge_c_per_m3[{index}]", value)
            for index, value in enumerate(self.charge_c_per_m3)
        )
        current = tuple(
            finite_vec3(f"current_a_per_m2[{index}]", value)
            for index, value in enumerate(self.current_a_per_m2)
        )
        momentum = tuple(
            finite_vec3(f"momentum_kg_per_m2_s[{index}]", value)
            for index, value in enumerate(self.momentum_kg_per_m2_s)
        )
        energy = tuple(
            finite_scalar(f"kinetic_energy_j_per_m3[{index}]", value)
            for index, value in enumerate(self.kinetic_energy_j_per_m3)
        )
        if any(value < 0.0 for value in number + charge + energy):
            raise HybridValidationError(
                "number, ion charge, and kinetic-energy densities must be non-negative"
            )
        object.__setattr__(self, "number_per_m3", number)
        object.__setattr__(self, "charge_c_per_m3", charge)
        object.__setattr__(self, "current_a_per_m2", current)
        object.__setattr__(self, "momentum_kg_per_m2_s", momentum)
        object.__setattr__(self, "kinetic_energy_j_per_m3", energy)


@dataclass(frozen=True, slots=True)
class SourceExchange:
    """Signed conservative exchange between ions and a named background."""

    ion_momentum_delta_kg_m_per_s: Vec3 = (0.0, 0.0, 0.0)
    background_momentum_delta_kg_m_per_s: Vec3 = (0.0, 0.0, 0.0)
    ion_energy_delta_j: float = 0.0
    background_energy_delta_j: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ion_momentum_delta_kg_m_per_s",
            finite_vec3("ion_momentum_delta", self.ion_momentum_delta_kg_m_per_s),
        )
        object.__setattr__(
            self,
            "background_momentum_delta_kg_m_per_s",
            finite_vec3(
                "background_momentum_delta", self.background_momentum_delta_kg_m_per_s
            ),
        )
        object.__setattr__(
            self,
            "ion_energy_delta_j",
            finite_scalar("ion_energy_delta_j", self.ion_energy_delta_j),
        )
        object.__setattr__(
            self,
            "background_energy_delta_j",
            finite_scalar("background_energy_delta_j", self.background_energy_delta_j),
        )

    @property
    def momentum_residual_kg_m_per_s(self) -> Vec3:
        return tuple(
            ion + background
            for ion, background in zip(
                self.ion_momentum_delta_kg_m_per_s,
                self.background_momentum_delta_kg_m_per_s,
                strict=True,
            )
        )  # type: ignore[return-value]

    @property
    def energy_residual_j(self) -> float:
        return self.ion_energy_delta_j + self.background_energy_delta_j


@dataclass(frozen=True, slots=True)
class ElectronFluidState:
    number_density_per_m3: tuple[float, ...]
    temperature_k: tuple[float, ...]
    pressure_pa: tuple[float, ...]
    anomalous_mobility_m2_per_v_s: None = None

    def __post_init__(self) -> None:
        lengths = {
            len(self.number_density_per_m3),
            len(self.temperature_k),
            len(self.pressure_pa),
        }
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise HybridValidationError(
                "electron state arrays must have one common length >= 2"
            )
        density = tuple(
            finite_scalar(f"number_density_per_m3[{index}]", value)
            for index, value in enumerate(self.number_density_per_m3)
        )
        temperature = tuple(
            finite_scalar(f"temperature_k[{index}]", value)
            for index, value in enumerate(self.temperature_k)
        )
        pressure = tuple(
            finite_scalar(f"pressure_pa[{index}]", value)
            for index, value in enumerate(self.pressure_pa)
        )
        if any(value < 0.0 for value in density + temperature + pressure):
            raise HybridValidationError(
                "electron density, temperature, and pressure must be non-negative"
            )
        if self.anomalous_mobility_m2_per_v_s is not None:
            raise HybridValidationError(
                "anomalous mobility is unresolved in this first slice"
            )
        object.__setattr__(self, "number_density_per_m3", density)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "pressure_pa", pressure)


@dataclass(frozen=True, slots=True)
class ElectronClosureResult:
    state: ElectronFluidState
    source_exchange: SourceExchange
    electric_field_v_per_m: tuple[float, ...] | None
    closure_name: str

    def __post_init__(self) -> None:
        if self.electric_field_v_per_m is not None:
            field = tuple(
                finite_scalar(f"electric_field_v_per_m[{index}]", value)
                for index, value in enumerate(self.electric_field_v_per_m)
            )
            if len(field) != len(self.state.number_density_per_m3):
                raise HybridValidationError(
                    "electron electric-field and state lengths must match"
                )
            object.__setattr__(self, "electric_field_v_per_m", field)
        if not isinstance(self.closure_name, str) or not self.closure_name.strip():
            raise HybridValidationError("closure_name must be non-empty")


@dataclass(frozen=True, slots=True)
class CollisionCrossSection:
    """Explicit constant test cross section; not a calibrated xenon dataset."""

    process: str
    sigma_m2: float
    provenance: str = "synthetic constant verification fixture"

    def __post_init__(self) -> None:
        if self.process not in {"charge_exchange", "elastic"}:
            raise HybridValidationError("process must be charge_exchange or elastic")
        sigma = finite_scalar("sigma_m2", self.sigma_m2)
        if sigma < 0.0:
            raise HybridValidationError("sigma_m2 must be non-negative")
        if not self.provenance.strip():
            raise HybridValidationError("cross-section provenance must be non-empty")
        object.__setattr__(self, "sigma_m2", sigma)


@dataclass(frozen=True, slots=True)
class CollisionBatchResult:
    particles: tuple[Particle, ...]
    collision_count: int
    expected_collision_count: float
    source_exchange: SourceExchange
    species_count_delta: tuple[tuple[str, int], ...] = ()
    represented_charge_delta_c: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.collision_count, int)
            or self.collision_count < 0
            or self.collision_count > len(self.particles)
        ):
            raise HybridValidationError("collision_count is inconsistent")
        expected = finite_scalar(
            "expected_collision_count", self.expected_collision_count
        )
        if not 0.0 <= expected <= len(self.particles):
            raise HybridValidationError("expected_collision_count is inconsistent")
        if any(
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(delta, int)
            or isinstance(delta, bool)
            for identifier, delta in self.species_count_delta
        ):
            raise HybridValidationError("species_count_delta is malformed")
        charge_delta = finite_scalar(
            "represented_charge_delta_c", self.represented_charge_delta_c
        )
        object.__setattr__(self, "expected_collision_count", expected)
        object.__setattr__(self, "represented_charge_delta_c", charge_delta)
