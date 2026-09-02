"""Typed SI contracts for the axisymmetric (r,z) electrostatic PIC-MCC.

Every object here is immutable, validated on construction, and fails closed on
nonfinite or contradictory input.  Arrays use the codebase's radial-major
layout ``values[r_index][z_index]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from numbers import Real
from typing import Literal

import numpy as np

EPSILON_0_F_PER_M = 8.8541878128e-12
ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
XENON_MASS_KG = 2.1801714e-25  # 131.293 u (CIAAW standard atomic weight) x 1.66053906660e-27 kg
LIGHT_SPEED_M_PER_S = 299792458.0
BOLTZMANN_J_PER_K = 1.380649e-23
EV_J = ELEMENTARY_CHARGE_C


class PIC2DError(Exception):
    """Base error for the axisymmetric PIC workstream."""


class PIC2DValidationError(PIC2DError, ValueError):
    """An input violates the documented numerical or physical contract."""


class PIC2DConvergenceError(PIC2DError, RuntimeError):
    """A numerical solve failed its explicit residual contract."""


class PIC2DDeviceError(PIC2DError, RuntimeError):
    """An optional execution device is unavailable."""


class PIC2DStabilityError(PIC2DError, RuntimeError):
    """A fail-closed stability gate rejected the configured or observed state."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PIC2DValidationError(f"{name} must be a real scalar, not {type(value).__name__}")
    converted = float(value)
    if not isfinite(converted):
        raise PIC2DValidationError(f"{name} must be finite")
    return converted


def _positive(name: str, value: object) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise PIC2DValidationError(f"{name} must be positive")
    return converted


def _nonnegative(name: str, value: object) -> float:
    converted = _finite(name, value)
    if converted < 0.0:
        raise PIC2DValidationError(f"{name} must be non-negative")
    return converted


def _integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise PIC2DValidationError(f"{name} must be an integer")
    converted = int(value)
    if converted < minimum:
        raise PIC2DValidationError(f"{name} must be >= {minimum}")
    return converted


@dataclass(frozen=True, slots=True)
class ChannelGeometry:
    """Axisymmetric channel: straight bore, then a linear divergent exit cone.

    ``wall_radius_m(z)`` is ``bore_radius_m`` for ``z <= cone_start_z_m`` and
    grows linearly to ``exit_radius_m`` at ``z_max_m``.  The anode face is the
    plane ``z_min_m``; the exit (cathode reference) plane is ``z_max_m``.
    """

    bore_radius_m: float
    z_min_m: float
    z_max_m: float
    cone_start_z_m: float
    exit_radius_m: float

    def __post_init__(self) -> None:
        bore = _positive("bore_radius_m", self.bore_radius_m)
        z_min = _finite("z_min_m", self.z_min_m)
        z_max = _finite("z_max_m", self.z_max_m)
        cone = _finite("cone_start_z_m", self.cone_start_z_m)
        exit_radius = _positive("exit_radius_m", self.exit_radius_m)
        if not z_min < z_max:
            raise PIC2DValidationError("z_max_m must exceed z_min_m")
        if not z_min <= cone <= z_max:
            raise PIC2DValidationError("cone_start_z_m must lie within [z_min_m, z_max_m]")
        if exit_radius < bore:
            raise PIC2DValidationError("exit_radius_m must be >= bore_radius_m")
        if cone == z_max and exit_radius != bore:
            raise PIC2DValidationError("a zero-length cone must keep exit_radius_m == bore_radius_m")
        for name, value in (
            ("bore_radius_m", bore), ("z_min_m", z_min), ("z_max_m", z_max),
            ("cone_start_z_m", cone), ("exit_radius_m", exit_radius),
        ):
            object.__setattr__(self, name, value)

    @property
    def max_radius_m(self) -> float:
        return self.exit_radius_m

    @property
    def length_m(self) -> float:
        return self.z_max_m - self.z_min_m

    def wall_radius_m(self, z_m: np.ndarray | float) -> np.ndarray:
        z = np.asarray(z_m, dtype=np.float64)
        if self.cone_start_z_m >= self.z_max_m:
            return np.full_like(z, self.bore_radius_m)
        slope = (self.exit_radius_m - self.bore_radius_m) / (self.z_max_m - self.cone_start_z_m)
        return self.bore_radius_m + slope * np.clip(z - self.cone_start_z_m, 0.0, None)

    def to_dict(self) -> dict[str, float]:
        return {
            "bore_radius_m": self.bore_radius_m,
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
            "cone_start_z_m": self.cone_start_z_m,
            "exit_radius_m": self.exit_radius_m,
        }


@dataclass(frozen=True, slots=True)
class Grid2D:
    """Uniform node-centred (r,z) mesh over the geometry bounding box."""

    geometry: ChannelGeometry
    radial_cells: int
    axial_cells: int

    def __post_init__(self) -> None:
        nr = _integer("radial_cells", self.radial_cells, 2)
        nz = _integer("axial_cells", self.axial_cells, 2)
        object.__setattr__(self, "radial_cells", nr)
        object.__setattr__(self, "axial_cells", nz)
        if not isfinite(self.dr_m) or self.dr_m <= 0.0 or not isfinite(self.dz_m) or self.dz_m <= 0.0:
            raise PIC2DValidationError("grid spacing must be finite and positive")
        # The straight bore radius must lie on a radial grid line so the
        # straight dielectric wall is represented exactly (no staircase there).
        bore_index = self.geometry.bore_radius_m / self.dr_m
        if abs(bore_index - round(bore_index)) > 1.0e-9 or round(bore_index) < 1:
            raise PIC2DValidationError(
                "bore_radius_m must be an integer number (>=1) of radial cells"
            )

    @property
    def dr_m(self) -> float:
        return self.geometry.max_radius_m / self.radial_cells

    @property
    def dz_m(self) -> float:
        return self.geometry.length_m / self.axial_cells

    @property
    def node_shape(self) -> tuple[int, int]:
        return (self.radial_cells + 1, self.axial_cells + 1)

    @property
    def cell_shape(self) -> tuple[int, int]:
        return (self.radial_cells, self.axial_cells)

    @property
    def r_m(self) -> np.ndarray:
        return np.arange(self.radial_cells + 1, dtype=np.float64) * self.dr_m

    @property
    def z_m(self) -> np.ndarray:
        return self.geometry.z_min_m + np.arange(self.axial_cells + 1, dtype=np.float64) * self.dz_m

    def to_dict(self) -> dict[str, object]:
        return {
            "geometry": self.geometry.to_dict(),
            "radial_cells": self.radial_cells,
            "axial_cells": self.axial_cells,
            "dr_m": self.dr_m,
            "dz_m": self.dz_m,
            "layout": "radial-major; values[r_index][z_index]",
        }


@dataclass(frozen=True, slots=True)
class Species2D:
    """One kinetic macro-particle species with a uniform macro weight."""

    name: str
    charge_c: float
    mass_kg: float
    macro_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PIC2DValidationError("species name must be non-empty")
        charge = _finite("charge_c", self.charge_c)
        if charge == 0.0:
            raise PIC2DValidationError("kinetic species charge must be nonzero")
        object.__setattr__(self, "charge_c", charge)
        object.__setattr__(self, "mass_kg", _positive("mass_kg", self.mass_kg))
        object.__setattr__(self, "macro_weight", _positive("macro_weight", self.macro_weight))

    @property
    def charge_to_mass(self) -> float:
        return self.charge_c / self.mass_kg

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "charge_c": self.charge_c,
            "mass_kg": self.mass_kg,
            "macro_weight": self.macro_weight,
        }


def electron_species(macro_weight: float) -> Species2D:
    return Species2D("e-", -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, macro_weight)


def xenon_ion_species(macro_weight: float) -> Species2D:
    return Species2D("Xe+", ELEMENTARY_CHARGE_C, XENON_MASS_KG, macro_weight)


@dataclass(slots=True)
class ParticleArrays:
    """Structure-of-arrays particle state for one species (CPU reference).

    ``r_m``/``z_m`` are cylindrical positions.  ``vr``, ``vt``, ``vz`` are the
    radial, azimuthal and axial velocity components in the particle's own
    meridional frame.  In the leapfrog cycle the stored velocity is
    ``v^(n-1/2)`` at position ``x^n``.
    """

    r_m: np.ndarray
    z_m: np.ndarray
    vr_m_per_s: np.ndarray
    vt_m_per_s: np.ndarray
    vz_m_per_s: np.ndarray

    def __post_init__(self) -> None:
        arrays = [
            np.ascontiguousarray(np.asarray(values, dtype=np.float64))
            for values in (self.r_m, self.z_m, self.vr_m_per_s, self.vt_m_per_s, self.vz_m_per_s)
        ]
        (self.r_m, self.z_m, self.vr_m_per_s, self.vt_m_per_s, self.vz_m_per_s) = arrays
        self.validate()

    def validate(self) -> None:
        arrays = (self.r_m, self.z_m, self.vr_m_per_s, self.vt_m_per_s, self.vz_m_per_s)
        if any(values.ndim != 1 for values in arrays):
            raise PIC2DValidationError("particle arrays must be one-dimensional")
        if len({values.shape[0] for values in arrays}) != 1:
            raise PIC2DValidationError("particle arrays must have equal length")
        if any(not np.isfinite(values).all() for values in arrays):
            raise PIC2DValidationError("particle state must contain only finite values")
        if self.r_m.size and np.any(self.r_m < 0.0):
            raise PIC2DValidationError("particle radius must be non-negative")

    @property
    def count(self) -> int:
        return int(self.r_m.shape[0])

    @classmethod
    def empty(cls) -> "ParticleArrays":
        zeros = np.zeros(0, dtype=np.float64)
        return cls(zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())

    def copy(self) -> "ParticleArrays":
        return ParticleArrays(
            self.r_m.copy(), self.z_m.copy(), self.vr_m_per_s.copy(),
            self.vt_m_per_s.copy(), self.vz_m_per_s.copy(),
        )

    def select(self, mask: np.ndarray) -> "ParticleArrays":
        return ParticleArrays(
            self.r_m[mask], self.z_m[mask], self.vr_m_per_s[mask],
            self.vt_m_per_s[mask], self.vz_m_per_s[mask],
        )

    def append(self, other: "ParticleArrays") -> "ParticleArrays":
        return ParticleArrays(
            np.concatenate((self.r_m, other.r_m)),
            np.concatenate((self.z_m, other.z_m)),
            np.concatenate((self.vr_m_per_s, other.vr_m_per_s)),
            np.concatenate((self.vt_m_per_s, other.vt_m_per_s)),
            np.concatenate((self.vz_m_per_s, other.vz_m_per_s)),
        )

    def speed_squared(self) -> np.ndarray:
        return self.vr_m_per_s**2 + self.vt_m_per_s**2 + self.vz_m_per_s**2


@dataclass(frozen=True, slots=True)
class PoissonConfig2D:
    """Field-solve method; every path publishes only against a recomputed true residual.

    * ``direct``: exact block-Thomas factorisation on the host (columns blocked
      along z); identical on the CPU and Warp backends.
    * ``device-direct``: exact block-Thomas on the device (rows blocked along r,
      one CUDA graph, no host synchronisation); the CPU backend maps it to
      ``direct``.  Agrees with ``direct`` to roundoff, not bitwise.
    * ``pcg``: Jacobi preconditioned conjugate gradient (host or device).

    The contract is ``true residual <= max(absolute_tolerance, relative_tolerance * |rhs|)``.
    """

    method: Literal["direct", "device-direct", "pcg"] = "direct"
    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 0.0
    max_iterations: int = 20_000
    preconditioner: Literal["jacobi"] = "jacobi"

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_tolerance", _positive("relative_tolerance", self.relative_tolerance))
        object.__setattr__(self, "absolute_tolerance", _nonnegative("absolute_tolerance", self.absolute_tolerance))
        object.__setattr__(self, "max_iterations", _integer("max_iterations", self.max_iterations, 1))
        if self.preconditioner != "jacobi":
            raise PIC2DValidationError("only the Jacobi preconditioner is implemented")
        if self.method not in ("direct", "device-direct", "pcg"):
            raise PIC2DValidationError("Poisson method must be 'direct', 'device-direct' or 'pcg'")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "max_iterations": self.max_iterations,
            "preconditioner": self.preconditioner,
        }


@dataclass(frozen=True, slots=True)
class StabilityLimits:
    """Fail-closed explicit-PIC admission thresholds."""

    max_omega_pe_dt: float = 0.2
    max_omega_ce_dt: float = 0.2
    max_cell_debye_ratio: float = 1.0
    max_particle_courant: float = 1.0
    max_collision_probability: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "max_omega_pe_dt", "max_omega_ce_dt", "max_cell_debye_ratio",
            "max_particle_courant", "max_collision_probability",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if self.max_collision_probability >= 1.0:
            raise PIC2DValidationError("max_collision_probability must be < 1")

    def to_dict(self) -> dict[str, float]:
        return {
            "max_omega_pe_dt": self.max_omega_pe_dt,
            "max_omega_ce_dt": self.max_omega_ce_dt,
            "max_cell_debye_ratio": self.max_cell_debye_ratio,
            "max_particle_courant": self.max_particle_courant,
            "max_collision_probability": self.max_collision_probability,
        }


@dataclass(frozen=True, slots=True)
class StabilityReport2D:
    """Published explicit-PIC metrics; ``violations`` is empty when admitted."""

    dt_s: float
    reference_density_per_m3: float
    reference_electron_temperature_ev: float
    max_b_t: float
    omega_pe_rad_per_s: float
    omega_pe_dt: float
    omega_ce_rad_per_s: float
    omega_ce_dt: float
    debye_length_m: float
    cell_debye_ratio: float
    max_electron_speed_m_per_s: float
    particle_courant: float
    max_collision_probability: float
    violations: tuple[str, ...]

    @property
    def stable(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "dt_s": self.dt_s,
            "reference_density_per_m3": self.reference_density_per_m3,
            "reference_electron_temperature_ev": self.reference_electron_temperature_ev,
            "max_b_t": self.max_b_t,
            "omega_pe_rad_per_s": self.omega_pe_rad_per_s,
            "omega_pe_dt": self.omega_pe_dt,
            "omega_ce_rad_per_s": self.omega_ce_rad_per_s,
            "omega_ce_dt": self.omega_ce_dt,
            "debye_length_m": self.debye_length_m,
            "cell_debye_ratio": self.cell_debye_ratio,
            "max_electron_speed_m_per_s": self.max_electron_speed_m_per_s,
            "particle_courant": self.particle_courant,
            "max_collision_probability": self.max_collision_probability,
            "violations": list(self.violations),
            "stable": self.stable,
        }


def stability_report(
    grid: Grid2D,
    dt_s: float,
    *,
    reference_density_per_m3: float,
    reference_electron_temperature_ev: float,
    max_b_t: float,
    max_electron_energy_ev: float,
    max_collision_probability: float = 0.0,
    limits: StabilityLimits = StabilityLimits(),
) -> StabilityReport2D:
    """Evaluate explicit-PIC gates for a configured or observed state.

    Every metric must be finite; a nonfinite metric raises rather than
    publishing a partial report.  The report is a necessary admission gate,
    not proof that sheaths, gyro-orbits, or mean free paths are resolved.
    """

    dt = _positive("dt_s", dt_s)
    density = _nonnegative("reference_density_per_m3", reference_density_per_m3)
    temperature = _positive("reference_electron_temperature_ev", reference_electron_temperature_ev)
    b_max = _nonnegative("max_b_t", max_b_t)
    energy = _positive("max_electron_energy_ev", max_electron_energy_ev)
    probability = _nonnegative("max_collision_probability", max_collision_probability)
    omega_pe = sqrt(density * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ELECTRON_MASS_KG))
    omega_ce = ELEMENTARY_CHARGE_C * b_max / ELECTRON_MASS_KG
    if density > 0.0:
        debye = sqrt(EPSILON_0_F_PER_M * temperature * EV_J / (density * ELEMENTARY_CHARGE_C**2))
    else:
        debye = float("inf")
    cell = max(grid.dr_m, grid.dz_m)
    cell_ratio = 0.0 if not isfinite(debye) else cell / debye
    gamma = 1.0 + energy * EV_J / (ELECTRON_MASS_KG * LIGHT_SPEED_M_PER_S**2)
    speed = LIGHT_SPEED_M_PER_S * sqrt(1.0 - gamma**-2)
    courant = speed * dt / min(grid.dr_m, grid.dz_m)
    metrics = (omega_pe * dt, omega_ce * dt, cell_ratio, courant, probability)
    if any(not isfinite(value) for value in metrics):
        raise PIC2DValidationError("stability metrics are not finite")
    violations: list[str] = []
    if omega_pe * dt > limits.max_omega_pe_dt:
        violations.append("plasma-frequency timestep limit exceeded")
    if omega_ce * dt > limits.max_omega_ce_dt:
        violations.append("cyclotron timestep limit exceeded")
    if cell_ratio > limits.max_cell_debye_ratio:
        violations.append("cell size exceeds the Debye-length limit")
    if courant > limits.max_particle_courant:
        violations.append("particle Courant limit exceeded")
    if probability > limits.max_collision_probability:
        violations.append("null-collision probability limit exceeded")
    return StabilityReport2D(
        dt, density, temperature, b_max, omega_pe, omega_pe * dt, omega_ce, omega_ce * dt,
        debye if isfinite(debye) else float("inf"), cell_ratio, speed, courant, probability,
        tuple(violations),
    )


def require_stable(report: StabilityReport2D) -> StabilityReport2D:
    if not report.stable:
        raise PIC2DStabilityError("stability gate rejected the run: " + "; ".join(report.violations))
    return report


@dataclass(frozen=True, slots=True)
class BoundaryPotentials:
    anode_v: float
    exit_v: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "anode_v", _finite("anode_v", self.anode_v))
        object.__setattr__(self, "exit_v", _finite("exit_v", self.exit_v))

    def to_dict(self) -> dict[str, float]:
        return {"anode_v": self.anode_v, "exit_v": self.exit_v}


__all__ = [
    "BOLTZMANN_J_PER_K",
    "BoundaryPotentials",
    "ChannelGeometry",
    "ELECTRON_MASS_KG",
    "ELEMENTARY_CHARGE_C",
    "EPSILON_0_F_PER_M",
    "EV_J",
    "Grid2D",
    "LIGHT_SPEED_M_PER_S",
    "PIC2DConvergenceError",
    "PIC2DDeviceError",
    "PIC2DError",
    "PIC2DStabilityError",
    "PIC2DValidationError",
    "ParticleArrays",
    "PoissonConfig2D",
    "Species2D",
    "StabilityLimits",
    "StabilityReport2D",
    "XENON_MASS_KG",
    "electron_species",
    "require_stable",
    "stability_report",
    "xenon_ion_species",
]
