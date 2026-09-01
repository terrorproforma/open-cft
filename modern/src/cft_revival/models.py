"""Typed domain models.

Unit-bearing values use explicit suffixes because plain floats have no runtime
unit safety. This avoids an undeclared dependency while making every boundary
crossing auditable. A future units library can replace these aliases without
changing the public field names.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, NewType, Sequence

Volts = NewType("Volts", float)
Amperes = NewType("Amperes", float)
Sccm = NewType("Sccm", float)
Millimetres = NewType("Millimetres", float)
Tesla = NewType("Tesla", float)


class ValidationError(ValueError):
    """A domain value violates an explicit model invariant."""


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class DesignPoint:
    """The eight legacy `CFTOpt` decision variables, in original order."""

    anode_voltage_v: Volts
    anode_current_a: Amperes
    mass_flow_sccm: Sccm
    inner_magnet_radius_mm: Millimetres
    outer_magnet_radius_mm: Millimetres
    inner_shield_radius_mm: Millimetres
    outer_shield_radius_mm: Millimetres
    outer_enclosure_radius_mm: Millimetres

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> DesignPoint:
        if len(values) != 8:
            raise ValidationError(f"expected 8 design variables, got {len(values)}")
        point = cls(
            Volts(values[0]),
            Amperes(values[1]),
            Sccm(values[2]),
            Millimetres(values[3]),
            Millimetres(values[4]),
            Millimetres(values[5]),
            Millimetres(values[6]),
            Millimetres(values[7]),
        )
        point.validate()
        return point

    def validate(self) -> None:
        values = self.as_sequence()
        for name, value in zip(self.__dataclass_fields__, values, strict=True):
            _require_finite(name, value)

        ranges = (
            (1.0, 1000.0),
            (0.001, 10.0),
            (0.2, 50.0),
            (2.0, 50.0),
            (2.0, 50.0),
            (2.0, 50.0),
            (2.0, 50.0),
            (2.0, 50.0),
        )
        for name, value, (lower, upper) in zip(
            self.__dataclass_fields__, values, ranges, strict=True
        ):
            if not lower <= value <= upper:
                raise ValidationError(f"{name}={value} outside [{lower}, {upper}]")

        radii = values[3:]
        if radii[0] <= 2.5:
            raise ValidationError("inner_magnet_radius_mm must be > 2.5 (legacy constraint)")
        for inner, outer in zip(radii, radii[1:]):
            if outer - inner <= 0.01:
                raise ValidationError("successive radii must differ by more than 0.01 mm")
        if radii[-1] + 0.01 >= 50.0:
            raise ValidationError("outer_enclosure_radius_mm must be < 49.99 mm")

    def as_sequence(self) -> tuple[float, ...]:
        return (
            float(self.anode_voltage_v),
            float(self.anode_current_a),
            float(self.mass_flow_sccm),
            float(self.inner_magnet_radius_mm),
            float(self.outer_magnet_radius_mm),
            float(self.inner_shield_radius_mm),
            float(self.outer_shield_radius_mm),
            float(self.outer_enclosure_radius_mm),
        )


@dataclass(frozen=True)
class LegacyPhysicsConstants:
    xenon_ionization_energy_ev: float = 12.1
    excitation_fraction: float = 0.25
    ionization_fraction: float = 0.07
    thermalization_fraction: float = 0.68
    background_cusp_probability: float = 0.002
    background_potential_v: float = 0.0
    cathode_electron_temperature_ev: float = 0.0
    xenon_atom_mass_kg: float = 2.1801714e-25
    elementary_charge_c: float = 1.60217662e-19
    sccm_to_kg_per_s: float = 9.83009e-8
    standard_gravity_m_per_s2: float = 9.80655

    def validate(self) -> None:
        fractions = (
            self.excitation_fraction,
            self.ionization_fraction,
            self.thermalization_fraction,
        )
        if any(value < 0.0 for value in fractions):
            raise ValidationError("energy partition fractions cannot be negative")
        if abs(sum(fractions) - 1.0) > 1e-12:
            raise ValidationError("energy partition fractions must sum to one")
        if self.cathode_electron_temperature_ev < 0.0:
            raise ValidationError("electron temperature cannot be negative")


@dataclass(frozen=True)
class CuspProbabilities:
    p1: float
    p2: float
    p3: float
    p4: float

    def __post_init__(self) -> None:
        for name, value in zip(("p1", "p2", "p3", "p4"), self.as_tuple(), strict=True):
            if not isfinite(value) or not 0.0 <= value <= 0.5:
                raise ValidationError(f"{name} must be finite and in [0, 0.5]")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.p1, self.p2, self.p3, self.p4


@dataclass(frozen=True)
class FieldProfile:
    positions_mm: tuple[float, ...]
    magnitudes_t: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.positions_mm) != len(self.magnitudes_t) or not self.positions_mm:
            raise ValidationError("field profile arrays must be non-empty and equal length")
        if any(not isfinite(value) for value in self.positions_mm + self.magnitudes_t):
            raise ValidationError("field profile values must be finite")


@dataclass(frozen=True)
class MagneticFieldResult:
    centreline: FieldProfile
    wall: FieldProfile
    provenance: str


@dataclass(frozen=True)
class PlasmaSolution:
    values: tuple[float, ...]
    converged: bool
    residual_norm: float
    provenance: str

    def __post_init__(self) -> None:
        if len(self.values) != 30:
            raise ValidationError("legacy-compatible plasma solution requires 30 values")
        _require_finite("residual_norm", self.residual_norm)


@dataclass(frozen=True)
class PerformanceResult:
    thrust_n: float
    total_efficiency: float
    specific_impulse_s: float
    anode_power_w: float
    beam_efficiency: float
    grid_efficiency: float
    mass_utilization: float


@dataclass(frozen=True)
class AppConfig:
    output_directory: Path
    femm_export_directory: Path | None
    magnetic_backend: str
    plasma_backend: str
    serialize_femm: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], base: Path = Path(".")) -> AppConfig:
        output = base / str(raw.get("output_directory", "outputs"))
        femm_raw = raw.get("femm_export_directory")
        config = cls(
            output_directory=output.resolve(),
            femm_export_directory=(base / str(femm_raw)).resolve() if femm_raw else None,
            magnetic_backend=str(raw.get("magnetic_backend", "femm-export")),
            plasma_backend=str(raw.get("plasma_backend", "unimplemented")),
            serialize_femm=bool(raw.get("serialize_femm", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.magnetic_backend == "femm-export" and self.femm_export_directory is None:
            raise ValidationError("femm-export backend requires femm_export_directory")
        if self.magnetic_backend.startswith("femm") and not self.serialize_femm:
            raise ValidationError("FEMM must remain serialized until process isolation is proven")
