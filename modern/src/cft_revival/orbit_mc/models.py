"""Typed SI contracts for deterministic electron full-orbit Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, pi
from numbers import Real
from typing import Mapping, Protocol

import numpy as np

ELECTRON_CHARGE_C = -1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
LIGHT_SPEED_M_PER_S = 299792458.0
EV_J = 1.602176634e-19


class OrbitMCError(Exception):
    """Base error for the isolated orbit workstream."""


class OrbitValidationError(OrbitMCError, ValueError):
    """An input or persistent contract is invalid."""


class OrbitNumericsError(OrbitMCError, ArithmeticError):
    """An orbit produced an invalid numerical state."""


class Termination(str, Enum):
    WALL_HIT = "wall_hit"
    REFLECTED = "reflected"
    DOMAIN_ESCAPE = "domain_escape"
    PATH_TIMEOUT = "path_timeout"
    TIME_TIMEOUT = "time_timeout"
    STEP_LIMIT = "step_limit"
    NONFINITE_STATE = "nonfinite_state"
    EXTREME_RELATIVITY = "extreme_relativity"
    FIELD_FAILURE = "field_failure"
    INITIAL_STATE_INVALID = "initial_state_invalid"


class EstimatorPolicy(str, Enum):
    UNWEIGHTED_BINOMIAL = "unweighted_binomial"


class AxisymmetricField(Protocol):
    max_b_t: float

    def magnetic_cartesian(self, position_m: np.ndarray) -> np.ndarray: ...

    def electric_cartesian(
        self, position_m: np.ndarray, time_s: float
    ) -> np.ndarray: ...


def _finite(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise OrbitValidationError(f"{name} must be a real scalar, not boolean")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise OrbitValidationError(f"{name} must be a real scalar") from error
    if not isfinite(converted):
        raise OrbitValidationError(f"{name} must be finite")
    return converted


def _positive(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise OrbitValidationError(f"{name} must be positive")
    return converted


@dataclass(frozen=True, slots=True)
class ElectronLaunch:
    """One immutable SI electron launch, including deterministic identity."""

    launch_id: str
    seed_id: int
    kinetic_energy_ev: float
    pitch_angle_rad: float
    position_m: tuple[float, float, float]
    parallel_direction: int
    gyrophase_rad: float
    flux_surface_id: str

    def __post_init__(self) -> None:
        if not self.launch_id or not self.flux_surface_id:
            raise OrbitValidationError("launch and flux-surface IDs must be non-empty")
        if (
            isinstance(self.seed_id, bool)
            or not isinstance(self.seed_id, int)
            or self.seed_id < 0
        ):
            raise OrbitValidationError("seed_id must be a nonnegative integer")
        energy = _positive("kinetic_energy_ev", self.kinetic_energy_ev)
        pitch = _finite("pitch_angle_rad", self.pitch_angle_rad)
        phase = _finite("gyrophase_rad", self.gyrophase_rad)
        if not 0.0 <= pitch <= 0.5 * pi:
            raise OrbitValidationError("pitch_angle_rad must lie in [0, pi/2]")
        if self.parallel_direction not in (-1, 1):
            raise OrbitValidationError("parallel_direction must be -1 or +1")
        if len(self.position_m) != 3:
            raise OrbitValidationError("position_m must have three SI coordinates")
        position = tuple(_finite(f"position_m[{i}]", value) for i, value in enumerate(self.position_m))
        object.__setattr__(self, "kinetic_energy_ev", energy)
        object.__setattr__(self, "pitch_angle_rad", pitch)
        object.__setattr__(self, "gyrophase_rad", phase % (2.0 * pi))
        object.__setattr__(self, "position_m", position)


@dataclass(frozen=True, slots=True)
class OrbitConfig:
    """Physical termination and conservative timestep policy."""

    wall_radius_m: float
    wall_z_min_m: float
    wall_z_max_m: float
    domain_radius_m: float
    domain_z_min_m: float
    domain_z_max_m: float
    max_time_s: float
    max_path_m: float
    max_steps: int = 2_000_000
    max_rotation_rad: float = 0.08
    event_tolerance_m: float = 1.0e-9
    maximum_gamma: float = 20.0
    fixed_dt_s: float | None = None

    def __post_init__(self) -> None:
        numeric = (
            "wall_radius_m", "wall_z_min_m", "wall_z_max_m",
            "domain_radius_m", "domain_z_min_m", "domain_z_max_m",
            "max_time_s", "max_path_m", "max_rotation_rad",
            "event_tolerance_m", "maximum_gamma",
        )
        values = {name: _finite(name, getattr(self, name)) for name in numeric}
        if (
            values["wall_radius_m"] <= 0.0
            or values["domain_radius_m"] < values["wall_radius_m"]
            or values["wall_z_max_m"] <= values["wall_z_min_m"]
            or values["domain_z_max_m"] <= values["domain_z_min_m"]
            or values["max_time_s"] <= 0.0
            or values["max_path_m"] <= 0.0
            or not 0.0 < values["max_rotation_rad"] <= 0.5
            or values["event_tolerance_m"] < 0.0
            or values["maximum_gamma"] <= 1.0
        ):
            raise OrbitValidationError("orbit geometry/time/rotation policy is invalid")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise OrbitValidationError("step controls must be positive integers")
        if self.fixed_dt_s is not None:
            object.__setattr__(self, "fixed_dt_s", _positive("fixed_dt_s", self.fixed_dt_s))


@dataclass(frozen=True, slots=True)
class GyroAverage:
    cycle_index: int
    phase_start_rad: float
    phase_end_rad: float
    mu_j_per_t: float


@dataclass(frozen=True, slots=True)
class OrbitResult:
    launch_id: str
    termination: Termination
    reason: str
    final_position_m: tuple[float, float, float]
    final_velocity_m_per_s: tuple[float, float, float]
    wall_endpoint_m: tuple[float, float, float] | None
    elapsed_time_s: float
    path_length_m: float
    steps: int
    accumulated_gyro_phase_rad: float
    complete_gyrocycles: int
    gyro_averages: tuple[GyroAverage, ...]
    initial_energy_j: float
    final_energy_j: float
    maximum_relative_energy_error: float
    maximum_instantaneous_mu_relative_variation: float | None
    transit_fraction: float
    maximum_b_t: float
    configured_max_time_s: float
    configured_max_path_m: float
    event_tolerance_m: float
    dt_s: float
    backend: str
    event_witness: Mapping[str, object]

    @property
    def wall_hit(self) -> bool:
        return self.termination is Termination.WALL_HIT


def kinetic_energy_j_from_velocity(velocity_m_per_s: np.ndarray) -> float:
    speed2 = float(np.dot(velocity_m_per_s, velocity_m_per_s))
    if not isfinite(speed2) or speed2 >= LIGHT_SPEED_M_PER_S**2:
        raise OrbitNumericsError("velocity is nonfinite or superluminal")
    gamma = 1.0 / np.sqrt(1.0 - speed2 / LIGHT_SPEED_M_PER_S**2)
    return float((gamma - 1.0) * ELECTRON_MASS_KG * LIGHT_SPEED_M_PER_S**2)


def velocity_from_energy_ev(energy_ev: float) -> float:
    gamma = 1.0 + _positive("kinetic_energy_ev", energy_ev) * EV_J / (
        ELECTRON_MASS_KG * LIGHT_SPEED_M_PER_S**2
    )
    return LIGHT_SPEED_M_PER_S * np.sqrt(1.0 - gamma**-2)
