"""Strict SI models for hypothetical axisymmetric CFT/HEMP geometry.

The geometry borrows the mechanically useful periodic-permanent-magnet (PPM)
stack pattern found in travelling-wave tubes.  It does not model a TWT
slow-wave RF circuit, electron-beam amplification, or TWT operating physics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite, pi, ulp
from typing import Any

SCHEMA_VERSION = "cft_revival.geometry.axisymmetric_cft/1.1.0"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
SAFE_IDENTIFIER_PATTERN = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_SAFE_IDENTIFIER = re.compile(rf"^{SAFE_IDENTIFIER_PATTERN}$")


class GeometryValidationError(ValueError):
    """A geometry violates the closed CFT geometry contract."""


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise GeometryValidationError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryValidationError(f"{name} must be a finite real number") from error
    if not isfinite(result):
        raise GeometryValidationError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise GeometryValidationError(f"{name} must be positive")
    return 0.0 if result == 0.0 else result


def _identifier(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_IDENTIFIER.fullmatch(value)
        or ".." in value
    ):
        raise GeometryValidationError(
            f"{name} must match safe canonical pattern {SAFE_IDENTIFIER_PATTERN!r}"
        )
    return value


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometryValidationError(f"{name} must be non-empty text")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise GeometryValidationError(f"{name} contains forbidden control characters")
    return value


def _ulp_close(left: float, right: float, *, max_ulps: float = 2.0) -> bool:
    return abs(left - right) <= max_ulps * max(ulp(left), ulp(right))


def _strict_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise GeometryValidationError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


class RegionShape(str, Enum):
    RECTANGULAR_ANNULUS = "rectangular_annulus"
    LINEAR_TAPER_ANNULUS = "linear_taper_annulus"


class MagnetizationDirection(str, Enum):
    AXIAL_POSITIVE = "axial_positive"
    AXIAL_NEGATIVE = "axial_negative"

    @property
    def polarity(self) -> int:
        return 1 if self is MagnetizationDirection.AXIAL_POSITIVE else -1


class MaterialKind(str, Enum):
    VACUUM_PLASMA = "vacuum_or_plasma"
    DIELECTRIC = "dielectric"
    PERMANENT_MAGNET = "permanent_magnet"
    SOFT_MAGNETIC = "soft_magnetic"
    NONMAGNETIC_SHIELD = "nonmagnetic_shield"
    ELECTRODE = "electrode"


class PermanentMagnetAuthority(str, Enum):
    RECOIL_REMANENCE = "recoil_remanence_constitutive"
    EQUIVALENT_BOUND_CURRENT = "equivalent_bound_current"


@dataclass(frozen=True, slots=True)
class PermanentMagnetRepresentationPlan:
    plan_id: str
    authority: PermanentMagnetAuthority
    solver_authoritative: bool = True

    def __post_init__(self) -> None:
        _identifier("plan_id", self.plan_id)
        try:
            authority = PermanentMagnetAuthority(self.authority)
        except ValueError as error:
            raise GeometryValidationError(
                "unsupported permanent-magnet representation authority"
            ) from error
        if self.solver_authoritative is not True:
            raise GeometryValidationError(
                "geometry representation plans must be solver-authoritative"
            )
        object.__setattr__(self, "authority", authority)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "authority": self.authority.value,
            "solver_authoritative": self.solver_authoritative,
        }


@dataclass(frozen=True, slots=True)
class ManufacturingRules:
    minimum_thickness_m: float
    minimum_clearance_m: float
    radial_tolerance_m: float
    axial_tolerance_m: float
    thermal_clearance_m: float

    def __post_init__(self) -> None:
        for name in (
            "minimum_thickness_m",
            "minimum_clearance_m",
            "radial_tolerance_m",
            "axial_tolerance_m",
            "thermal_clearance_m",
        ):
            value = _finite(name, getattr(self, name), positive=True)
            object.__setattr__(self, name, value)
        if self.thermal_clearance_m < self.minimum_clearance_m:
            raise GeometryValidationError(
                "thermal_clearance_m must be at least minimum_clearance_m"
            )
        if 2.0 * self.radial_tolerance_m >= self.minimum_thickness_m:
            raise GeometryValidationError(
                "radial tolerance stack consumes the minimum manufacturable thickness"
            )
        if 2.0 * self.axial_tolerance_m >= self.minimum_thickness_m:
            raise GeometryValidationError(
                "axial tolerance stack consumes the minimum manufacturable thickness"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_thickness_m": self.minimum_thickness_m,
            "minimum_clearance_m": self.minimum_clearance_m,
            "radial_tolerance_m": self.radial_tolerance_m,
            "axial_tolerance_m": self.axial_tolerance_m,
            "thermal_clearance_m": self.thermal_clearance_m,
        }


@dataclass(frozen=True, slots=True)
class MaterialDefinition:
    material_id: str
    category: MaterialKind
    relative_permeability: float
    density_kg_per_m3: float | None
    provenance: str
    assumption: bool

    def __post_init__(self) -> None:
        _identifier("material_id", self.material_id)
        try:
            category = MaterialKind(self.category)
        except ValueError as error:
            raise GeometryValidationError("unsupported material category") from error
        object.__setattr__(self, "category", category)
        relative = _finite(
            "relative_permeability", self.relative_permeability, positive=True
        )
        object.__setattr__(self, "relative_permeability", relative)
        if self.density_kg_per_m3 is not None:
            density = _finite("density_kg_per_m3", self.density_kg_per_m3, positive=True)
            if not 1.0e-3 <= density <= 1.0e6:
                raise GeometryValidationError(
                    "density_kg_per_m3 must lie in the numerical/physical domain "
                    "[1e-3, 1e6]"
                )
            object.__setattr__(self, "density_kg_per_m3", density)
        _text("material provenance", self.provenance)
        if not isinstance(self.assumption, bool):
            raise GeometryValidationError("material assumption must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "material_id": self.material_id,
            "category": self.category.value,
            "relative_permeability": self.relative_permeability,
            "density_kg_per_m3": self.density_kg_per_m3,
            "provenance": self.provenance,
            "assumption": self.assumption,
        }


@dataclass(frozen=True, slots=True)
class MeridionalRegion:
    region_id: str
    owner_id: str
    role: str
    material_id: str
    shape: RegionShape
    r_inner_start_m: float
    r_inner_end_m: float
    r_outer_start_m: float
    r_outer_end_m: float
    z_min_m: float
    z_max_m: float
    polarity: int | None = None

    def __post_init__(self) -> None:
        _identifier("region_id", self.region_id)
        _identifier("owner_id", self.owner_id)
        _identifier("role", self.role)
        _identifier("material_id", self.material_id)
        try:
            shape = RegionShape(self.shape)
        except ValueError as error:
            raise GeometryValidationError("unsupported meridional region shape") from error
        object.__setattr__(self, "shape", shape)
        for name in (
            "r_inner_start_m",
            "r_inner_end_m",
            "r_outer_start_m",
            "r_outer_end_m",
            "z_min_m",
            "z_max_m",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if min(self.r_inner_start_m, self.r_inner_end_m) < 0.0:
            raise GeometryValidationError("region inner radii must be non-negative")
        if (
            self.r_outer_start_m <= self.r_inner_start_m
            or self.r_outer_end_m <= self.r_inner_end_m
        ):
            raise GeometryValidationError("region radial thickness must be positive")
        if self.z_max_m <= self.z_min_m:
            raise GeometryValidationError("region axial thickness must be positive")
        if shape is RegionShape.RECTANGULAR_ANNULUS and (
            self.r_inner_start_m != self.r_inner_end_m
            or self.r_outer_start_m != self.r_outer_end_m
        ):
            raise GeometryValidationError("rectangular regions require constant radii")
        if self.polarity is not None and (
            isinstance(self.polarity, bool) or self.polarity not in (-1, 1)
        ):
            raise GeometryValidationError("region polarity must be -1, +1, or null")

    @property
    def axial_thickness_m(self) -> float:
        return self.z_max_m - self.z_min_m

    @property
    def minimum_radial_thickness_m(self) -> float:
        return min(
            self.r_outer_start_m - self.r_inner_start_m,
            self.r_outer_end_m - self.r_inner_end_m,
        )

    @property
    def volume_m3(self) -> float:
        """Exact frustum-difference volume for a linearly varying annulus."""

        length = self.axial_thickness_m

        def frustum(radius_0: float, radius_1: float) -> float:
            return pi * length * (
                radius_0 * radius_0 + radius_0 * radius_1 + radius_1 * radius_1
            ) / 3.0

        volume = frustum(self.r_outer_start_m, self.r_outer_end_m) - frustum(
            self.r_inner_start_m, self.r_inner_end_m
        )
        if not isfinite(volume):
            raise GeometryValidationError(
                f"region {self.region_id} volume is not representable"
            )
        if volume <= 0.0:
            raise GeometryValidationError(
                f"region {self.region_id} positive volume underflowed"
            )
        return volume

    def radial_interval_at(self, z_m: float) -> tuple[float, float]:
        fraction = (z_m - self.z_min_m) / self.axial_thickness_m
        inner = self.r_inner_start_m + fraction * (
            self.r_inner_end_m - self.r_inner_start_m
        )
        outer = self.r_outer_start_m + fraction * (
            self.r_outer_end_m - self.r_outer_start_m
        )
        return inner, outer

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "owner_id": self.owner_id,
            "role": self.role,
            "material_id": self.material_id,
            "shape": self.shape.value,
            "r_inner_start_m": self.r_inner_start_m,
            "r_inner_end_m": self.r_inner_end_m,
            "r_outer_start_m": self.r_outer_start_m,
            "r_outer_end_m": self.r_outer_end_m,
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
            "polarity": self.polarity,
        }


@dataclass(frozen=True, slots=True)
class ChamberDefinition:
    inner_radius_m: float
    outer_radius_m: float
    length_m: float
    injector_length_m: float
    dielectric_thickness_m: float
    exit_length_m: float
    exit_outer_radius_m: float

    def __post_init__(self) -> None:
        for name in (
            "inner_radius_m",
            "outer_radius_m",
            "length_m",
            "injector_length_m",
            "dielectric_thickness_m",
            "exit_length_m",
            "exit_outer_radius_m",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.inner_radius_m < 0.0:
            raise GeometryValidationError("chamber inner radius must be non-negative")
        if self.outer_radius_m <= self.inner_radius_m:
            raise GeometryValidationError("chamber radial span must be positive")
        if self.length_m <= 0.0 or self.dielectric_thickness_m <= 0.0:
            raise GeometryValidationError(
                "chamber length and dielectric thickness must be positive"
            )
        if not 0.0 < self.injector_length_m < self.length_m:
            raise GeometryValidationError("injector length must lie inside the chamber")
        if not 0.0 <= self.exit_length_m < self.length_m - self.injector_length_m:
            raise GeometryValidationError("exit length must leave a positive straight channel")
        if self.exit_length_m == 0.0:
            if self.exit_outer_radius_m != self.outer_radius_m:
                raise GeometryValidationError(
                    "disabled divergent exit requires exit_outer_radius_m=outer_radius_m"
                )
        elif self.exit_outer_radius_m <= self.outer_radius_m:
            raise GeometryValidationError(
                "enabled divergent exit must increase the channel outer radius"
            )

    @property
    def exit_start_m(self) -> float:
        return self.length_m - self.exit_length_m

    def to_dict(self) -> dict[str, object]:
        return {
            "inner_radius_m": self.inner_radius_m,
            "outer_radius_m": self.outer_radius_m,
            "length_m": self.length_m,
            "injector_length_m": self.injector_length_m,
            "dielectric_thickness_m": self.dielectric_thickness_m,
            "exit_length_m": self.exit_length_m,
            "exit_outer_radius_m": self.exit_outer_radius_m,
        }


@dataclass(frozen=True, slots=True)
class ElectrodeDefinition:
    anode_region_id: str
    anode_thickness_m: float
    injector_region_id: str

    def __post_init__(self) -> None:
        _identifier("anode_region_id", self.anode_region_id)
        _identifier("injector_region_id", self.injector_region_id)
        object.__setattr__(
            self,
            "anode_thickness_m",
            _finite("anode_thickness_m", self.anode_thickness_m, positive=True),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "anode_region_id": self.anode_region_id,
            "anode_thickness_m": self.anode_thickness_m,
            "injector_region_id": self.injector_region_id,
        }


@dataclass(frozen=True, slots=True)
class PPMStage:
    stage_id: str
    index: int
    center_z_m: float
    pitch_m: float
    z_min_m: float
    z_max_m: float
    magnet_region_id: str
    pole_after_region_id: str | None
    magnetization: MagnetizationDirection

    def __post_init__(self) -> None:
        _identifier("stage_id", self.stage_id)
        _identifier("magnet_region_id", self.magnet_region_id)
        if self.pole_after_region_id is not None:
            _identifier("pole_after_region_id", self.pole_after_region_id)
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise GeometryValidationError("stage index must be an integer >= 0")
        object.__setattr__(self, "center_z_m", _finite("center_z_m", self.center_z_m))
        object.__setattr__(self, "pitch_m", _finite("pitch_m", self.pitch_m, positive=True))
        object.__setattr__(self, "z_min_m", _finite("z_min_m", self.z_min_m))
        object.__setattr__(self, "z_max_m", _finite("z_max_m", self.z_max_m))
        if self.z_max_m <= self.z_min_m:
            raise GeometryValidationError("stage axial envelope must be positive")
        if not self.z_min_m <= self.center_z_m <= self.z_max_m:
            raise GeometryValidationError("stage center must lie inside its envelope")
        try:
            direction = MagnetizationDirection(self.magnetization)
        except ValueError as error:
            raise GeometryValidationError("unsupported magnetization direction") from error
        object.__setattr__(self, "magnetization", direction)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "index": self.index,
            "center_z_m": self.center_z_m,
            "pitch_m": self.pitch_m,
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
            "magnet_region_id": self.magnet_region_id,
            "pole_after_region_id": self.pole_after_region_id,
            "magnetization": self.magnetization.value,
        }


@dataclass(frozen=True, slots=True)
class ExternalComponent:
    component_id: str
    kind: str
    axisymmetry: str
    location: str
    included_in_2d_model: bool

    def __post_init__(self) -> None:
        _identifier("component_id", self.component_id)
        _identifier("kind", self.kind)
        if self.axisymmetry != "external_non_axisymmetric_metadata":
            raise GeometryValidationError(
                "cathode/neutralizer must be explicit external non-axisymmetric metadata"
            )
        _text("external component location", self.location)
        if self.included_in_2d_model:
            raise GeometryValidationError(
                "external non-axisymmetric component cannot be included in the 2D model"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "kind": self.kind,
            "axisymmetry": self.axisymmetry,
            "location": self.location,
            "included_in_2d_model": self.included_in_2d_model,
        }


@dataclass(frozen=True, slots=True)
class EvidenceNote:
    note_id: str
    classification: str
    statement: str
    source: str

    def __post_init__(self) -> None:
        _identifier("note_id", self.note_id)
        if self.classification not in ("traceable", "assumption", "limitation"):
            raise GeometryValidationError("unsupported evidence classification")
        _text("evidence statement", self.statement)
        _text("evidence source", self.source)

    def to_dict(self) -> dict[str, object]:
        return {
            "note_id": self.note_id,
            "classification": self.classification,
            "statement": self.statement,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class AxisymmetricCFTGeometry:
    config_id: str
    title: str
    classification: str
    chamber: ChamberDefinition
    electrodes: ElectrodeDefinition
    manufacturing: ManufacturingRules
    permanent_magnet_plan: PermanentMagnetRepresentationPlan
    materials: tuple[MaterialDefinition, ...]
    regions: tuple[MeridionalRegion, ...]
    stages: tuple[PPMStage, ...]
    external_components: tuple[ExternalComponent, ...]
    evidence: tuple[EvidenceNote, ...]
    design_variable_order: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = "right_handed_cylindrical_r_z_axisymmetric"
    length_unit: str = "m"

    def __post_init__(self) -> None:
        _identifier("config_id", self.config_id)
        _text("title", self.title)
        if self.classification != "hypothetical_not_optimized_not_build_qualified":
            raise GeometryValidationError("geometry classification must preserve claim limits")
        if self.schema_version != SCHEMA_VERSION:
            raise GeometryValidationError("unsupported geometry schema version")
        if self.coordinate_system != "right_handed_cylindrical_r_z_axisymmetric":
            raise GeometryValidationError("unsupported coordinate system")
        if self.length_unit != "m":
            raise GeometryValidationError("all geometry lengths must use SI metres")
        tuple_names = (
            "materials",
            "regions",
            "stages",
            "external_components",
            "evidence",
            "design_variable_order",
        )
        if any(not isinstance(getattr(self, name), tuple) for name in tuple_names):
            raise GeometryValidationError("geometry collections must be immutable tuples")
        self._validate_identifiers_and_references()
        self._validate_stage_sequence()
        self._validate_chamber_coverage()
        self._validate_manufacturing()
        self._validate_overlap_and_ordering()
        self._validate_required_roles()
        self._validate_connected_region_graph()

    def _validate_identifiers_and_references(self) -> None:
        material_ids = [material.material_id for material in self.materials]
        region_ids = [region.region_id for region in self.regions]
        stage_ids = [stage.stage_id for stage in self.stages]
        external_ids = [component.component_id for component in self.external_components]
        evidence_ids = [note.note_id for note in self.evidence]
        for name, values in (
            ("material", material_ids),
            ("region", region_ids),
            ("stage", stage_ids),
            ("external component", external_ids),
            ("evidence", evidence_ids),
        ):
            if len(values) != len(set(values)):
                raise GeometryValidationError(f"{name} identifiers must be unique")
        if any(region.material_id not in set(material_ids) for region in self.regions):
            raise GeometryValidationError("every region must reference a supplied material")
        region_set = set(region_ids)
        if self.electrodes.anode_region_id not in region_set:
            raise GeometryValidationError("anode_region_id must reference a region")
        if self.electrodes.injector_region_id not in region_set:
            raise GeometryValidationError("injector_region_id must reference a region")
        for stage in self.stages:
            if stage.magnet_region_id not in region_set:
                raise GeometryValidationError("every stage magnet must reference a region")
            if (
                stage.pole_after_region_id is not None
                and stage.pole_after_region_id not in region_set
            ):
                raise GeometryValidationError("every stage pole must reference a region")
            magnet = self.region_by_id(stage.magnet_region_id)
            if (
                magnet.role != "permanent_magnet"
                or magnet.polarity != stage.magnetization.polarity
            ):
                raise GeometryValidationError("stage polarity must match its magnet region")
        if len(self.design_variable_order) != len(set(self.design_variable_order)):
            raise GeometryValidationError("design variable names must be unique")
        expected_plan_id = (
            f"{self.config_id}-{self.permanent_magnet_plan.authority.value}-v1"
        )
        if self.permanent_magnet_plan.plan_id != expected_plan_id:
            raise GeometryValidationError(
                "permanent-magnet plan ID must bind config ID and authority"
            )

    def _validate_stage_sequence(self) -> None:
        if len(self.stages) < 2:
            raise GeometryValidationError("a PPM stack requires at least two stages")
        ordered = sorted(self.stages, key=lambda stage: stage.index)
        if tuple(stage.index for stage in ordered) != tuple(range(len(ordered))):
            raise GeometryValidationError("stage indices must be contiguous from zero")
        if tuple(ordered) != self.stages:
            raise GeometryValidationError("stages must be ordered by index")
        pitch = ordered[0].pitch_m
        magnet_refs = tuple(stage.magnet_region_id for stage in ordered)
        pole_refs = tuple(
            stage.pole_after_region_id
            for stage in ordered
            if stage.pole_after_region_id is not None
        )
        if len(magnet_refs) != len(set(magnet_refs)):
            raise GeometryValidationError("stage magnet references must be unique")
        if len(pole_refs) != len(set(pole_refs)):
            raise GeometryValidationError("stage pole references must be unique")
        magnet_role_ids = {
            region.region_id
            for region in self.regions
            if region.role == "permanent_magnet"
        }
        pole_role_ids = {
            region.region_id for region in self.regions if region.role == "pole_piece"
        }
        if set(magnet_refs) != magnet_role_ids:
            raise GeometryValidationError(
                "stages must reference every permanent-magnet region exactly once"
            )
        if set(pole_refs) != pole_role_ids:
            raise GeometryValidationError(
                "stages must reference every pole-piece region exactly once"
            )
        chamber = self.chamber
        for stage_index, stage in enumerate(ordered):
            magnet = self.region_by_id(stage.magnet_region_id)
            magnet_center = (magnet.z_min_m + magnet.z_max_m) / 2.0
            if not _ulp_close(stage.center_z_m, magnet_center):
                raise GeometryValidationError(
                    f"stage {stage.stage_id} center does not match magnet region"
                )
            if stage.z_min_m < 0.0 or stage.z_max_m > chamber.length_m:
                raise GeometryValidationError(
                    f"stage {stage.stage_id} envelope lies outside chamber"
                )
            if (
                magnet.z_min_m < 0.0
                or magnet.z_max_m > chamber.length_m
                or magnet.z_min_m < stage.z_min_m
                or magnet.z_max_m > stage.z_max_m
                or magnet.z_min_m != stage.z_min_m
            ):
                raise GeometryValidationError(
                    f"stage {stage.stage_id} magnet lies outside stage/chamber envelope"
                )
            if stage_index < len(ordered) - 1:
                if stage.pole_after_region_id is None:
                    raise GeometryValidationError(
                        "every non-terminal stage requires pole_after"
                    )
                pole = self.region_by_id(stage.pole_after_region_id)
                if (
                    pole.shape is not RegionShape.RECTANGULAR_ANNULUS
                    or pole.z_min_m != magnet.z_max_m
                    or pole.z_max_m != stage.z_max_m
                    or pole.z_min_m < stage.z_min_m
                    or pole.z_max_m > chamber.length_m
                    or not _ulp_close(
                        pole.r_inner_start_m, magnet.r_inner_start_m
                    )
                    or not _ulp_close(
                        pole.r_outer_start_m, magnet.r_outer_start_m
                    )
                ):
                    raise GeometryValidationError(
                        f"stage {stage.stage_id} pole_after is not adjacent and "
                        "inside its envelope"
                    )
            elif stage.pole_after_region_id is not None:
                raise GeometryValidationError(
                    "terminal stage must not declare pole_after"
                )
            elif stage.z_max_m != magnet.z_max_m:
                raise GeometryValidationError(
                    "terminal stage envelope must end with its magnet"
                )
        for left, right in zip(ordered, ordered[1:]):
            if right.magnetization.polarity != -left.magnetization.polarity:
                raise GeometryValidationError("PPM cusp sequence must alternate polarity")
            if not _ulp_close(right.pitch_m, pitch):
                raise GeometryValidationError("all repeating PPM stages must share one pitch")
            if not _ulp_close(right.center_z_m - left.center_z_m, pitch):
                raise GeometryValidationError("stage centers must be separated by axial pitch")
            if left.z_max_m != right.z_min_m:
                raise GeometryValidationError(
                    "stage envelopes must form one ordered connected axial stack"
                )

    def _validate_chamber_coverage(self) -> None:
        chamber = self.chamber

        def validate_chain(
            regions: tuple[MeridionalRegion, ...],
            *,
            name: str,
        ) -> None:
            ordered = sorted(regions, key=lambda region: region.z_min_m)
            if not ordered or ordered[0].z_min_m != 0.0:
                raise GeometryValidationError(f"{name} must start exactly at z=0")
            if ordered[-1].z_max_m != chamber.length_m:
                raise GeometryValidationError(
                    f"{name} must end exactly at chamber length"
                )
            for left, right in zip(ordered, ordered[1:]):
                if left.z_max_m != right.z_min_m:
                    relation = "overlap" if left.z_max_m > right.z_min_m else "gap"
                    raise GeometryValidationError(
                        f"{name} has an axial {relation} between "
                        f"{left.region_id} and {right.region_id}"
                    )

        channel_regions = tuple(
            region
            for region in self.regions
            if region.role in ("injector_plasma", "channel_plasma")
        )
        dielectric_regions = tuple(
            region for region in self.regions if region.role == "dielectric_wall"
        )
        validate_chain(channel_regions, name="chamber fluid coverage")
        validate_chain(dielectric_regions, name="dielectric wall coverage")
        injector = self.region_by_id(self.electrodes.injector_region_id)
        if (
            injector.role != "injector_plasma"
            or injector.z_min_m != 0.0
            or injector.z_max_m != chamber.injector_length_m
        ):
            raise GeometryValidationError(
                "injector region must exactly cover the declared injector length"
            )
        if chamber.exit_length_m > 0.0:
            channel_exit = next(
                (
                    region
                    for region in channel_regions
                    if region.shape is RegionShape.LINEAR_TAPER_ANNULUS
                ),
                None,
            )
            wall_exit = next(
                (
                    region
                    for region in dielectric_regions
                    if region.shape is RegionShape.LINEAR_TAPER_ANNULUS
                ),
                None,
            )
            if channel_exit is None or wall_exit is None:
                raise GeometryValidationError(
                    "divergent exit requires tapered channel and wall regions"
                )
            if (
                channel_exit.z_min_m != chamber.exit_start_m
                or channel_exit.z_max_m != chamber.length_m
                or wall_exit.z_min_m != chamber.exit_start_m
                or wall_exit.z_max_m != chamber.length_m
            ):
                raise GeometryValidationError(
                    "divergent regions must exactly cover the declared exit"
                )
            endpoint_pairs = (
                (channel_exit.r_outer_start_m, chamber.outer_radius_m),
                (channel_exit.r_outer_end_m, chamber.exit_outer_radius_m),
                (wall_exit.r_inner_start_m, channel_exit.r_outer_start_m),
                (wall_exit.r_inner_end_m, channel_exit.r_outer_end_m),
                (
                    wall_exit.r_outer_start_m - wall_exit.r_inner_start_m,
                    chamber.dielectric_thickness_m,
                ),
                (
                    wall_exit.r_outer_end_m - wall_exit.r_inner_end_m,
                    chamber.dielectric_thickness_m,
                ),
            )
            if any(not _ulp_close(left, right) for left, right in endpoint_pairs):
                raise GeometryValidationError(
                    "divergent wall endpoints are not ULP-continuous"
                )
            channel_slope = (
                channel_exit.r_outer_end_m - channel_exit.r_outer_start_m
            ) / channel_exit.axial_thickness_m
            wall_inner_slope = (
                wall_exit.r_inner_end_m - wall_exit.r_inner_start_m
            ) / wall_exit.axial_thickness_m
            wall_outer_slope = (
                wall_exit.r_outer_end_m - wall_exit.r_outer_start_m
            ) / wall_exit.axial_thickness_m
            if not (
                _ulp_close(channel_slope, wall_inner_slope, max_ulps=4.0)
                and _ulp_close(channel_slope, wall_outer_slope, max_ulps=4.0)
            ):
                raise GeometryValidationError(
                    "divergent wall slopes are not continuous"
                )

    def _validate_manufacturing(self) -> None:
        rules = self.manufacturing
        if self.chamber.dielectric_thickness_m < rules.minimum_thickness_m:
            raise GeometryValidationError("dielectric wall is below minimum thickness")
        if self.electrodes.anode_thickness_m < rules.minimum_thickness_m:
            raise GeometryValidationError("anode is below minimum thickness")
        for region in self.regions:
            if (
                region.minimum_radial_thickness_m < rules.minimum_thickness_m
                or region.axial_thickness_m < rules.minimum_thickness_m
            ):
                raise GeometryValidationError(
                    f"region {region.region_id} is below minimum manufacturable thickness"
                )
        required_nominal_gap = (
            rules.thermal_clearance_m + 2.0 * rules.radial_tolerance_m
        )
        magnet_regions = tuple(
            self.region_by_id(stage.magnet_region_id) for stage in self.stages
        )
        dielectric_regions = tuple(
            region for region in self.regions if region.role == "dielectric_wall"
        )
        for magnet in magnet_regions:
            if magnet.shape is not RegionShape.RECTANGULAR_ANNULUS:
                raise GeometryValidationError(
                    "permanent-magnet regions must be rectangular until the "
                    "magnetics handoff supports tapered PM regions"
                )
            overlapping_wall_outer_radii: list[float] = []
            for wall in dielectric_regions:
                z_min = max(magnet.z_min_m, wall.z_min_m)
                z_max = min(magnet.z_max_m, wall.z_max_m)
                if z_max <= z_min:
                    continue
                overlapping_wall_outer_radii.extend(
                    (
                        wall.radial_interval_at(z_min)[1],
                        wall.radial_interval_at(z_max)[1],
                    )
                )
            if not overlapping_wall_outer_radii:
                raise GeometryValidationError(
                    f"magnet {magnet.region_id} has no axial dielectric coverage"
                )
            radial_gap = (
                magnet.r_inner_start_m - max(overlapping_wall_outer_radii)
            )
            if radial_gap <= required_nominal_gap:
                raise GeometryValidationError(
                    f"magnet {magnet.region_id} clearance must remain strictly "
                    "above thermal clearance over its complete axial span"
                )
        for left, right in zip(magnet_regions, magnet_regions[1:]):
            axial_gap = right.z_min_m - left.z_max_m
            required_axial_gap = (
                rules.minimum_clearance_m + 2.0 * rules.axial_tolerance_m
            )
            if axial_gap <= required_axial_gap:
                raise GeometryValidationError(
                    "inter-magnet axial gap must remain strictly above minimum "
                    "clearance after the axial tolerance stack"
                )

    def _validate_overlap_and_ordering(self) -> None:
        ordered = sorted(
            self.regions,
            key=lambda region: (
                region.z_min_m,
                region.r_inner_start_m,
                region.r_inner_end_m,
                region.region_id,
            ),
        )
        if tuple(ordered) != self.regions:
            raise GeometryValidationError(
                "regions must use deterministic (z,r,id) geometric ordering"
            )
        for index, left in enumerate(self.regions):
            for right in self.regions[index + 1 :]:
                z_low = max(left.z_min_m, right.z_min_m)
                z_high = min(left.z_max_m, right.z_max_m)
                if z_high <= z_low:
                    continue
                candidates = [z_low, z_high]

                def linear_value(
                    region: MeridionalRegion, z_m: float, boundary: str
                ) -> float:
                    return region.radial_interval_at(z_m)[
                        0 if boundary == "inner" else 1
                    ]

                # Radial overlap requires both outer-minus-opposite-inner
                # inequalities to be positive. Their roots partition the
                # common z-span into intervals with constant truth values.
                for first, first_boundary, second, second_boundary in (
                    (left, "outer", right, "inner"),
                    (right, "outer", left, "inner"),
                ):
                    low_value = linear_value(
                        first, z_low, first_boundary
                    ) - linear_value(second, z_low, second_boundary)
                    high_value = linear_value(
                        first, z_high, first_boundary
                    ) - linear_value(second, z_high, second_boundary)
                    if low_value != high_value:
                        root = z_low - low_value * (z_high - z_low) / (
                            high_value - low_value
                        )
                        if z_low < root < z_high:
                            candidates.append(root)
                partitions = sorted(set(candidates))
                samples = [
                    *partitions,
                    *(
                        (left_z + right_z) / 2.0
                        for left_z, right_z in zip(partitions, partitions[1:])
                    ),
                ]
                for z_m in samples:
                    z_left = min(max(z_m, left.z_min_m), left.z_max_m)
                    z_right = min(max(z_m, right.z_min_m), right.z_max_m)
                    left_inner, left_outer = left.radial_interval_at(z_left)
                    right_inner, right_outer = right.radial_interval_at(z_right)
                    if min(left_outer, right_outer) > max(left_inner, right_inner):
                        raise GeometryValidationError(
                            f"regions {left.region_id} and {right.region_id} overlap"
                        )

    def _validate_required_roles(self) -> None:
        roles = {region.role for region in self.regions}
        required = {
            "anode",
            "injector_plasma",
            "channel_plasma",
            "dielectric_wall",
            "permanent_magnet",
            "pole_piece",
            "shield",
            "yoke",
        }
        if not required <= roles:
            raise GeometryValidationError(
                f"missing required region roles {sorted(required - roles)}"
            )
        expected_material_kinds = {
            "anode": MaterialKind.ELECTRODE,
            "injector_plasma": MaterialKind.VACUUM_PLASMA,
            "channel_plasma": MaterialKind.VACUUM_PLASMA,
            "dielectric_wall": MaterialKind.DIELECTRIC,
            "permanent_magnet": MaterialKind.PERMANENT_MAGNET,
            "pole_piece": MaterialKind.SOFT_MAGNETIC,
            "shield": MaterialKind.NONMAGNETIC_SHIELD,
            "yoke": MaterialKind.SOFT_MAGNETIC,
        }
        for region in self.regions:
            expected = expected_material_kinds.get(region.role)
            material = self.material_by_id(region.material_id)
            if expected is None or material.category is not expected:
                raise GeometryValidationError(
                    f"region {region.region_id} role/material kind mismatch"
                )
            if region.role == "permanent_magnet" and region.polarity is None:
                raise GeometryValidationError(
                    "permanent-magnet regions require explicit polarity"
                )
            if region.role != "permanent_magnet" and region.polarity is not None:
                raise GeometryValidationError(
                    "only permanent-magnet regions may declare polarity"
                )
        anode = self.region_by_id(self.electrodes.anode_region_id)
        if anode.role != "anode":
            raise GeometryValidationError("anode_region_id must reference an anode")
        external_kinds = {component.kind for component in self.external_components}
        if not {"cathode", "neutralizer"} <= external_kinds:
            raise GeometryValidationError(
                "cathode and neutralizer must both be represented as external metadata"
            )
        if not any(
            note.classification == "limitation" and "TWT" in note.statement
            for note in self.evidence
        ):
            raise GeometryValidationError("TWT/CFT physics distinction must be explicit")

    def _validate_connected_region_graph(self) -> None:
        from .topology import interface_topology

        region_ids = {region.region_id for region in self.regions}
        graph: dict[str, set[str]] = {
            region_id: set() for region_id in region_ids
        }
        graph["ambient-background"] = set()
        graph["symmetry-axis"] = set()
        for descriptor in interface_topology(self):
            graph[descriptor.region_id].add(descriptor.adjacent_region_id)
            graph[descriptor.adjacent_region_id].add(descriptor.region_id)
        pending = ["ambient-background"]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(graph[current] - visited)
        if not region_ids <= visited:
            raise GeometryValidationError("material region graph is not connected")

    def region_by_id(self, region_id: str) -> MeridionalRegion:
        for region in self.regions:
            if region.region_id == region_id:
                return region
        raise GeometryValidationError(f"unknown region {region_id!r}")

    def material_by_id(self, material_id: str) -> MaterialDefinition:
        for material in self.materials:
            if material.material_id == material_id:
                return material
        raise GeometryValidationError(f"unknown material {material_id!r}")

    def to_payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "coordinate_system": self.coordinate_system,
            "length_unit": self.length_unit,
            "config_id": self.config_id,
            "title": self.title,
            "classification": self.classification,
            "chamber": self.chamber.to_dict(),
            "electrodes": self.electrodes.to_dict(),
            "manufacturing": self.manufacturing.to_dict(),
            "permanent_magnet_plan": self.permanent_magnet_plan.to_dict(),
            "materials": [material.to_dict() for material in self.materials],
            "regions": [region.to_dict() for region in self.regions],
            "stages": [stage.to_dict() for stage in self.stages],
            "external_components": [
                component.to_dict() for component in self.external_components
            ],
            "evidence": [note.to_dict() for note in self.evidence],
            "design_variable_order": list(self.design_variable_order),
        }

    @property
    def canonical_sha256(self) -> str:
        return sha256(canonical_json(self.to_payload_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = self.to_payload_dict()
        return {
            **payload,
            "integrity": {
                "algorithm": "sha256",
                "canonicalization": CANONICALIZATION,
                "payload_sha256": self.canonical_sha256,
            },
        }


def _json_value(value: object) -> object:
    if hasattr(value, "to_dict"):
        return _json_value(getattr(value, "to_dict")())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise GeometryValidationError("JSON keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return _finite("serialized float", value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    raise GeometryValidationError(f"unsupported JSON value {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GeometryValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_dataclass(cls: type[Any], value: object, keys: set[str], name: str) -> Any:
    if not isinstance(value, dict):
        raise GeometryValidationError(f"{name} must be an object")
    _strict_keys(value, keys, name)
    return cls(**value)


def geometry_from_dict(value: object) -> AxisymmetricCFTGeometry:
    if not isinstance(value, dict):
        raise GeometryValidationError("geometry must be an object")
    top_keys = {
        "schema_version",
        "coordinate_system",
        "length_unit",
        "config_id",
        "title",
        "classification",
        "chamber",
        "electrodes",
        "manufacturing",
        "permanent_magnet_plan",
        "materials",
        "regions",
        "stages",
        "external_components",
        "evidence",
        "design_variable_order",
        "integrity",
    }
    _strict_keys(value, top_keys, "geometry")
    integrity = value["integrity"]
    if not isinstance(integrity, dict):
        raise GeometryValidationError("integrity must be an object")
    _strict_keys(
        integrity,
        {"algorithm", "canonicalization", "payload_sha256"},
        "integrity",
    )
    if integrity["algorithm"] != "sha256" or integrity["canonicalization"] != CANONICALIZATION:
        raise GeometryValidationError("unsupported integrity declaration")
    sequence_names = (
        "materials",
        "regions",
        "stages",
        "external_components",
        "evidence",
        "design_variable_order",
    )
    if any(not isinstance(value[name], list) for name in sequence_names):
        raise GeometryValidationError("geometry collection fields must be arrays")
    chamber = _parse_dataclass(
        ChamberDefinition,
        value["chamber"],
        {
            "inner_radius_m",
            "outer_radius_m",
            "length_m",
            "injector_length_m",
            "dielectric_thickness_m",
            "exit_length_m",
            "exit_outer_radius_m",
        },
        "chamber",
    )
    electrodes = _parse_dataclass(
        ElectrodeDefinition,
        value["electrodes"],
        {"anode_region_id", "anode_thickness_m", "injector_region_id"},
        "electrodes",
    )
    manufacturing = _parse_dataclass(
        ManufacturingRules,
        value["manufacturing"],
        {
            "minimum_thickness_m",
            "minimum_clearance_m",
            "radial_tolerance_m",
            "axial_tolerance_m",
            "thermal_clearance_m",
        },
        "manufacturing",
    )
    permanent_magnet_plan = _parse_dataclass(
        PermanentMagnetRepresentationPlan,
        value["permanent_magnet_plan"],
        {"plan_id", "authority", "solver_authoritative"},
        "permanent_magnet_plan",
    )
    materials = tuple(
        _parse_dataclass(
            MaterialDefinition,
            item,
            {
                "material_id",
                "category",
                "relative_permeability",
                "density_kg_per_m3",
                "provenance",
                "assumption",
            },
            "material",
        )
        for item in value["materials"]
    )
    regions = tuple(
        _parse_dataclass(
            MeridionalRegion,
            item,
            {
                "region_id",
                "owner_id",
                "role",
                "material_id",
                "shape",
                "r_inner_start_m",
                "r_inner_end_m",
                "r_outer_start_m",
                "r_outer_end_m",
                "z_min_m",
                "z_max_m",
                "polarity",
            },
            "region",
        )
        for item in value["regions"]
    )
    stages = tuple(
        _parse_dataclass(
            PPMStage,
            item,
            {
                "stage_id",
                "index",
                "center_z_m",
                "pitch_m",
                "z_min_m",
                "z_max_m",
                "magnet_region_id",
                "pole_after_region_id",
                "magnetization",
            },
            "stage",
        )
        for item in value["stages"]
    )
    external = tuple(
        _parse_dataclass(
            ExternalComponent,
            item,
            {
                "component_id",
                "kind",
                "axisymmetry",
                "location",
                "included_in_2d_model",
            },
            "external component",
        )
        for item in value["external_components"]
    )
    evidence = tuple(
        _parse_dataclass(
            EvidenceNote,
            item,
            {"note_id", "classification", "statement", "source"},
            "evidence note",
        )
        for item in value["evidence"]
    )
    geometry = AxisymmetricCFTGeometry(
        config_id=value["config_id"],
        title=value["title"],
        classification=value["classification"],
        chamber=chamber,
        electrodes=electrodes,
        manufacturing=manufacturing,
        permanent_magnet_plan=permanent_magnet_plan,
        materials=materials,
        regions=regions,
        stages=stages,
        external_components=external,
        evidence=evidence,
        design_variable_order=tuple(value["design_variable_order"]),
        schema_version=value["schema_version"],
        coordinate_system=value["coordinate_system"],
        length_unit=value["length_unit"],
    )
    if integrity["payload_sha256"] != geometry.canonical_sha256:
        raise GeometryValidationError("geometry payload SHA-256 mismatch")
    return geometry


def deserialize_geometry(serialized: str) -> AxisymmetricCFTGeometry:
    if not isinstance(serialized, str):
        raise GeometryValidationError("serialized geometry must be text")
    try:
        value = json.loads(serialized, object_pairs_hook=_unique_object)
    except (TypeError, ValueError) as error:
        if isinstance(error, GeometryValidationError):
            raise
        raise GeometryValidationError("invalid geometry JSON") from error
    geometry = geometry_from_dict(value)
    if canonical_json(geometry.to_dict()) != serialized:
        raise GeometryValidationError("geometry JSON is not canonical")
    return geometry
