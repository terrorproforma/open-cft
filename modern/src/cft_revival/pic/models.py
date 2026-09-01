"""Typed SI contracts for the independent electrostatic PIC-MCC foundation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Literal

EPSILON_0_F_PER_M = 8.8541878128e-12
ELEMENTARY_CHARGE_C = 1.602176634e-19


class PICError(Exception):
    """Base error for this independent workstream."""


class PICValidationError(PICError, ValueError):
    """An input violates the documented numerical or physical contract."""


class PICConvergenceError(PICError, RuntimeError):
    """A numerical solve failed its explicit residual contract."""


class PICDeviceError(PICError, RuntimeError):
    """An optional execution device is unavailable."""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise PICValidationError(f"{name} must be numeric, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PICValidationError(f"{name} must be numeric") from error
    if not isfinite(converted):
        raise PICValidationError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class Grid1D:
    """Uniform periodic Cartesian mesh with an explicit transverse area."""

    x_min_m: float
    x_max_m: float
    cells: int
    transverse_area_m2: float = 1.0
    geometry: Literal["cartesian_1d"] = "cartesian_1d"

    def __post_init__(self) -> None:
        x_min = _finite("x_min_m", self.x_min_m)
        x_max = _finite("x_max_m", self.x_max_m)
        area = _finite("transverse_area_m2", self.transverse_area_m2)
        length = x_max - x_min
        if x_max <= x_min or not isfinite(length):
            raise PICValidationError("x_max_m must exceed x_min_m")
        if area <= 0.0:
            raise PICValidationError("transverse_area_m2 must be positive")
        if isinstance(self.cells, bool) or not isinstance(self.cells, int) or self.cells < 4:
            raise PICValidationError("cells must be an integer >= 4")
        if self.geometry != "cartesian_1d":
            raise PICValidationError("this verified foundation supports cartesian_1d only")
        object.__setattr__(self, "x_min_m", x_min)
        object.__setattr__(self, "x_max_m", x_max)
        object.__setattr__(self, "transverse_area_m2", area)
        if not isfinite(self.dx_m) or self.dx_m <= 0.0:
            raise PICValidationError("grid spacing must be finite and representable")

    @property
    def length_m(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def dx_m(self) -> float:
        return self.length_m / self.cells

    def wrap(self, x_m: float) -> float:
        return self.x_min_m + (x_m - self.x_min_m) % self.length_m


@dataclass(frozen=True, slots=True)
class Species:
    """One kinetic macro-particle species."""

    name: str
    charge_c: float
    mass_kg: float
    macro_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PICValidationError("species name must not be empty")
        charge = _finite("charge_c", self.charge_c)
        mass = _finite("mass_kg", self.mass_kg)
        weight = _finite("macro_weight", self.macro_weight)
        if charge == 0.0:
            raise PICValidationError("kinetic species charge_c must be nonzero")
        if mass <= 0.0 or weight <= 0.0:
            raise PICValidationError("mass_kg and macro_weight must be positive")
        object.__setattr__(self, "charge_c", charge)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "macro_weight", weight)


@dataclass(slots=True)
class ParticleState:
    """Mutable structure-of-arrays particle state.

    At stored position ``x^n``, velocity is ``v^(n-1/2)`` in ``PICStepper``.
    """

    x_m: list[float]
    vx_m_per_s: list[float]
    vy_m_per_s: list[float]
    vz_m_per_s: list[float]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate mutable arrays at every publication boundary."""

        arrays = (self.x_m, self.vx_m_per_s, self.vy_m_per_s, self.vz_m_per_s)
        if any(not isinstance(values, list) for values in arrays):
            raise PICValidationError("particle arrays must remain mutable lists")
        lengths = {len(values) for values in arrays}
        if len(lengths) != 1 or not self.x_m:
            raise PICValidationError("particle arrays must have one equal, nonzero length")
        try:
            finite = all(
                not isinstance(value, bool) and isfinite(float(value))
                for values in arrays
                for value in values
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise PICValidationError("particle state values must be numeric") from error
        if not finite:
            raise PICValidationError("particle state must contain only finite values")

    @property
    def count(self) -> int:
        return len(self.x_m)

    def copy(self) -> ParticleState:
        return ParticleState(
            self.x_m.copy(),
            self.vx_m_per_s.copy(),
            self.vy_m_per_s.copy(),
            self.vz_m_per_s.copy(),
        )


@dataclass(frozen=True, slots=True)
class PoissonConfig:
    relative_tolerance: float = 1.0e-11
    absolute_tolerance: float = 1.0e-12
    max_iterations: int = 10_000

    def __post_init__(self) -> None:
        relative = _finite("relative_tolerance", self.relative_tolerance)
        absolute = _finite("absolute_tolerance", self.absolute_tolerance)
        if relative <= 0.0 or absolute < 0.0:
            raise PICValidationError("Poisson tolerances must be positive/non-negative")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise PICValidationError("max_iterations must be an integer >= 1")
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "absolute_tolerance", absolute)


@dataclass(frozen=True, slots=True)
class PICConfig:
    dt_s: float
    background_charge_density_c_per_m3: float = 0.0
    seed: int = 0
    poisson: PoissonConfig = PoissonConfig()
    max_particle_courant: float = 1.0
    max_omega_p_dt: float = 0.2

    def __post_init__(self) -> None:
        dt = _finite("dt_s", self.dt_s)
        background = _finite(
            "background_charge_density_c_per_m3",
            self.background_charge_density_c_per_m3,
        )
        max_courant = _finite("max_particle_courant", self.max_particle_courant)
        max_omega = _finite("max_omega_p_dt", self.max_omega_p_dt)
        if dt <= 0.0 or max_courant <= 0.0 or max_omega <= 0.0:
            raise PICValidationError("dt and stability limits must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PICValidationError("seed must be a non-negative integer")
        object.__setattr__(self, "dt_s", dt)
        object.__setattr__(self, "background_charge_density_c_per_m3", background)
        object.__setattr__(self, "max_particle_courant", max_courant)
        object.__setattr__(self, "max_omega_p_dt", max_omega)


@dataclass(frozen=True, slots=True)
class StabilityReport:
    stable: bool
    particle_courant: float
    omega_p_dt: float
    violations: tuple[str, ...]


def stability_report(
    grid: Grid1D,
    species: Species,
    particles: ParticleState,
    config: PICConfig,
    physical_number_density_per_m3: float,
) -> StabilityReport:
    """Evaluate explicit cell-crossing and plasma-frequency gates."""

    particles.validate()
    for name, value in (
        ("grid.x_min_m", grid.x_min_m),
        ("grid.x_max_m", grid.x_max_m),
        ("grid.dx_m", grid.dx_m),
        ("grid.transverse_area_m2", grid.transverse_area_m2),
        ("species.charge_c", species.charge_c),
        ("species.mass_kg", species.mass_kg),
        ("species.macro_weight", species.macro_weight),
        ("config.dt_s", config.dt_s),
        ("config.max_particle_courant", config.max_particle_courant),
        ("config.max_omega_p_dt", config.max_omega_p_dt),
    ):
        _finite(name, value)
    if (
        grid.x_max_m <= grid.x_min_m
        or grid.dx_m <= 0.0
        or grid.transverse_area_m2 <= 0.0
        or species.charge_c == 0.0
        or species.mass_kg <= 0.0
        or species.macro_weight <= 0.0
        or config.dt_s <= 0.0
        or config.max_particle_courant <= 0.0
        or config.max_omega_p_dt <= 0.0
    ):
        raise PICValidationError("current grid/species/config state is invalid")
    if any(not grid.x_min_m <= position < grid.x_max_m for position in particles.x_m):
        raise PICValidationError("current particle positions lie outside the periodic grid")
    density = _finite("physical_number_density_per_m3", physical_number_density_per_m3)
    if density < 0.0:
        raise PICValidationError("physical_number_density_per_m3 must be non-negative")
    max_speed = max(abs(value) for value in particles.vx_m_per_s)
    courant = max_speed * config.dt_s / grid.dx_m
    if not isfinite(courant):
        raise PICValidationError("particle Courant metric is not finite")
    if density == 0.0:
        omega_p = 0.0
    else:
        log_omega = (
            log(abs(species.charge_c))
            + 0.5 * log(density)
            - 0.5 * log(EPSILON_0_F_PER_M)
            - 0.5 * log(species.mass_kg)
        )
        if not isfinite(log_omega) or log_omega >= 709.0:
            raise PICValidationError("plasma-frequency metric is not representable")
        omega_p = exp(log_omega)
    omega_dt = omega_p * config.dt_s
    if not isfinite(omega_dt):
        raise PICValidationError("plasma-frequency timestep metric is not finite")
    violations: list[str] = []
    if courant > config.max_particle_courant:
        violations.append("particle Courant limit exceeded")
    if omega_dt > config.max_omega_p_dt:
        violations.append("plasma-frequency timestep limit exceeded")
    return StabilityReport(not violations, courant, omega_dt, tuple(violations))
