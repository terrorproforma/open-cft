"""Axisymmetric magnetization and equivalent bound-current source contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, pi
from sys import float_info

from .common import (
    MagneticsValidationError,
    VectorRZ,
    finite_float,
    nonempty_identifier,
)
from .materials import SmCoPermanentMagnet

MAGNETIZATION_RELATIVE_TOLERANCE = 32.0 * float_info.epsilon


@dataclass(frozen=True, slots=True)
class AxisymmetricBounds:
    """Closed rectangular meridional bounds in metres."""

    r_inner_m: float
    r_outer_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self) -> None:
        names = ("r_inner_m", "r_outer_m", "z_min_m", "z_max_m")
        values = {name: finite_float(name, getattr(self, name)) for name in names}
        if values["r_inner_m"] < 0.0:
            raise MagneticsValidationError("r_inner_m must be non-negative")
        if values["r_outer_m"] <= values["r_inner_m"]:
            raise MagneticsValidationError("radial thickness must be positive")
        if values["z_max_m"] <= values["z_min_m"]:
            raise MagneticsValidationError("axial thickness must be positive")
        finite_float(
            "radial thickness",
            values["r_outer_m"] - values["r_inner_m"],
        )
        finite_float("axial thickness", values["z_max_m"] - values["z_min_m"])
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def radial_thickness_m(self) -> float:
        return self.r_outer_m - self.r_inner_m

    @property
    def axial_thickness_m(self) -> float:
        return self.z_max_m - self.z_min_m

    def to_dict(self) -> dict[str, object]:
        return {
            "r_inner_m": self.r_inner_m,
            "r_outer_m": self.r_outer_m,
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
        }


class SheetOrientation(str, Enum):
    """Axisymmetric surface represented by a line in the meridional plane."""

    CONSTANT_R = "constant_r"
    CONSTANT_Z = "constant_z"


class PermanentMagnetRepresentation(str, Enum):
    """Exactly one authoritative permanent-magnet representation per region."""

    RECOIL_REMANENCE = "recoil_remanence_constitutive"
    EQUIVALENT_BOUND_CURRENT = "equivalent_bound_current"


@dataclass(frozen=True, slots=True)
class AxisymmetricBoundCurrentSheet:
    """Azimuthal bound surface-current sheet ``K_b=M×n`` in A/m."""

    source_id: str
    surface_name: str
    orientation: SheetOrientation
    coordinate_m: float
    span_min_m: float
    span_max_m: float
    outward_normal_rz: VectorRZ
    k_phi_a_per_m: float

    def __post_init__(self) -> None:
        nonempty_identifier("source_id", self.source_id)
        nonempty_identifier("surface_name", self.surface_name)
        try:
            orientation = SheetOrientation(self.orientation)
        except ValueError as error:
            raise MagneticsValidationError("unsupported sheet orientation") from error
        coordinate = finite_float("coordinate_m", self.coordinate_m)
        span_min = finite_float("span_min_m", self.span_min_m)
        span_max = finite_float("span_max_m", self.span_max_m)
        current = finite_float("k_phi_a_per_m", self.k_phi_a_per_m)
        if coordinate <= 0.0 and orientation is SheetOrientation.CONSTANT_R:
            raise MagneticsValidationError("constant-r sheet radius must be positive")
        if span_max <= span_min:
            raise MagneticsValidationError("sheet span must be positive")
        if orientation is SheetOrientation.CONSTANT_Z and span_min < 0.0:
            raise MagneticsValidationError(
                "constant-z sheet radial span must be non-negative"
            )
        normal = self.outward_normal_rz.normalized()
        if orientation is SheetOrientation.CONSTANT_R and normal.axial != 0.0:
            raise MagneticsValidationError("constant-r sheet normal must be radial")
        if orientation is SheetOrientation.CONSTANT_Z and normal.radial != 0.0:
            raise MagneticsValidationError("constant-z sheet normal must be axial")
        if orientation is SheetOrientation.CONSTANT_R:
            area = 2.0 * pi * coordinate * (span_max - span_min)
        else:
            area = pi * (span_max - span_min) * (span_max + span_min)
        finite_float("current sheet area_m2", area)
        if area <= 0.0:
            raise MagneticsValidationError("current sheet area must be positive")
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "coordinate_m", coordinate)
        object.__setattr__(self, "span_min_m", span_min)
        object.__setattr__(self, "span_max_m", span_max)
        object.__setattr__(self, "outward_normal_rz", normal)
        object.__setattr__(self, "k_phi_a_per_m", current)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "axisymmetric_bound_surface_current",
            "source_id": self.source_id,
            "surface_name": self.surface_name,
            "orientation": self.orientation.value,
            "coordinate_m": self.coordinate_m,
            "span_min_m": self.span_min_m,
            "span_max_m": self.span_max_m,
            "outward_normal_rz": self.outward_normal_rz.to_dict(),
            "k_phi_a_per_m": self.k_phi_a_per_m,
            "area_m2": self.area_m2,
        }

    @property
    def area_m2(self) -> float:
        if self.orientation is SheetOrientation.CONSTANT_R:
            area = (
                2.0
                * pi
                * self.coordinate_m
                * (self.span_max_m - self.span_min_m)
            )
        else:
            area = pi * (self.span_max_m - self.span_min_m) * (
                self.span_max_m + self.span_min_m
            )
        return finite_float("current sheet area_m2", area)


def bound_volume_current_density_phi_a_per_m2(
    d_m_radial_dz_a_per_m2: float,
    d_m_axial_dr_a_per_m2: float,
) -> float:
    """Return ``(curl M)_phi = dM_r/dz - dM_z/dr`` in A/m²."""

    radial_gradient = finite_float(
        "d_m_radial_dz_a_per_m2", d_m_radial_dz_a_per_m2
    )
    axial_gradient = finite_float("d_m_axial_dr_a_per_m2", d_m_axial_dr_a_per_m2)
    return finite_float(
        "bound volume current density",
        radial_gradient - axial_gradient,
    )


def bound_surface_current_density_phi_a_per_m(
    magnetization_a_per_m: VectorRZ,
    outward_normal_rz: VectorRZ,
) -> float:
    """Return ``(M×n)_phi = M_z n_r - M_r n_z`` in A/m."""

    normal = outward_normal_rz.normalized()
    return finite_float(
        "bound surface current density",
        magnetization_a_per_m.axial * normal.radial
        - magnetization_a_per_m.radial * normal.axial,
    )


@dataclass(frozen=True, slots=True)
class UniformAxisymmetricMagnetizationSource:
    """Uniform ``M_r e_r + M_z e_z`` over an axisymmetric rectangular region."""

    source_id: str
    region_id: str
    material: SmCoPermanentMagnet
    bounds: AxisymmetricBounds
    magnetization_direction_rz: VectorRZ
    temperature_k: float
    representation: PermanentMagnetRepresentation = (
        PermanentMagnetRepresentation.EQUIVALENT_BOUND_CURRENT
    )

    def __post_init__(self) -> None:
        nonempty_identifier("source_id", self.source_id)
        nonempty_identifier("region_id", self.region_id)
        if not isinstance(self.material, SmCoPermanentMagnet):
            raise MagneticsValidationError(
                "equivalent-current source material must be a permanent magnet"
            )
        direction = self.magnetization_direction_rz.normalized()
        if (
            self.bounds.r_inner_m == 0.0
            and direction.radial != 0.0
        ):
            raise MagneticsValidationError(
                "radial magnetization is not regular on a region touching r=0"
            )
        try:
            representation = PermanentMagnetRepresentation(self.representation)
        except ValueError as error:
            raise MagneticsValidationError(
                "unsupported permanent-magnet representation"
            ) from error
        if representation is not PermanentMagnetRepresentation.EQUIVALENT_BOUND_CURRENT:
            raise MagneticsValidationError(
                "a magnetization source must use equivalent_bound_current authority"
            )
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "magnetization_direction_rz", direction)
        temperature = finite_float("temperature_k", self.temperature_k)
        self.material.remanence_t(temperature)
        object.__setattr__(self, "temperature_k", temperature)

    @classmethod
    def from_permanent_magnet(
        cls,
        *,
        source_id: str,
        region_id: str,
        material: SmCoPermanentMagnet,
        bounds: AxisymmetricBounds,
        direction: VectorRZ,
        temperature_k: float,
    ) -> UniformAxisymmetricMagnetizationSource:
        """Construct a source using ``M=B_r/mu0`` at a validated temperature."""

        return cls(
            source_id=source_id,
            region_id=region_id,
            material=material,
            bounds=bounds,
            magnetization_direction_rz=direction,
            temperature_k=temperature_k,
        )

    @property
    def permanent_magnet_material_id(self) -> str:
        return self.material.material_id

    @property
    def magnetization_a_per_m(self) -> VectorRZ:
        """Derive ``M=Br(T)/mu0``; callers cannot inject an arbitrary magnitude."""

        return self.material.magnetization_a_per_m(
            self.temperature_k,
            self.magnetization_direction_rz,
        )

    def validate_against(self, material: SmCoPermanentMagnet) -> None:
        """Validate identity, temperature, direction, and ``M=Br(T)/mu0``."""

        if self.material != material:
            raise MagneticsValidationError(
                f"source {self.source_id} permanent-magnet parameters do not match "
                "the handoff material"
            )
        expected = material.magnetization_a_per_m(
            self.temperature_k,
            self.magnetization_direction_rz,
        )
        actual = self.magnetization_a_per_m
        for component_name in ("radial", "axial"):
            expected_component = getattr(expected, component_name)
            actual_component = getattr(actual, component_name)
            if expected_component == 0.0:
                matches = actual_component == 0.0
            else:
                matches = isclose(
                    actual_component,
                    expected_component,
                    rel_tol=MAGNETIZATION_RELATIVE_TOLERANCE,
                    abs_tol=0.0,
                )
            if not matches:
                raise MagneticsValidationError(
                    f"source {self.source_id} magnetization must equal Br(T)/mu0 "
                    f"within relative tolerance {MAGNETIZATION_RELATIVE_TOLERANCE}"
                )

    @property
    def bound_volume_current_density_phi_a_per_m2(self) -> float:
        """Uniform magnetization has zero bound volume current in its interior."""

        return 0.0

    def equivalent_bound_current_sheets(
        self,
    ) -> tuple[AxisymmetricBoundCurrentSheet, ...]:
        """Return all non-degenerate physical boundary sheets.

        A region touching ``r=0`` has no finite-area inner cylindrical surface,
        so no sheet is emitted there.
        """

        bounds = self.bounds
        surfaces: list[AxisymmetricBoundCurrentSheet] = []

        def append(
            name: str,
            orientation: SheetOrientation,
            coordinate: float,
            span_min: float,
            span_max: float,
            normal: VectorRZ,
        ) -> None:
            surfaces.append(
                AxisymmetricBoundCurrentSheet(
                    source_id=self.source_id,
                    surface_name=name,
                    orientation=orientation,
                    coordinate_m=coordinate,
                    span_min_m=span_min,
                    span_max_m=span_max,
                    outward_normal_rz=normal,
                    k_phi_a_per_m=bound_surface_current_density_phi_a_per_m(
                        self.magnetization_a_per_m, normal
                    ),
                )
            )

        if bounds.r_inner_m > 0.0:
            append(
                "r_inner",
                SheetOrientation.CONSTANT_R,
                bounds.r_inner_m,
                bounds.z_min_m,
                bounds.z_max_m,
                VectorRZ(-1.0, 0.0),
            )
        append(
            "r_outer",
            SheetOrientation.CONSTANT_R,
            bounds.r_outer_m,
            bounds.z_min_m,
            bounds.z_max_m,
            VectorRZ(1.0, 0.0),
        )
        append(
            "z_min",
            SheetOrientation.CONSTANT_Z,
            bounds.z_min_m,
            bounds.r_inner_m,
            bounds.r_outer_m,
            VectorRZ(0.0, -1.0),
        )
        append(
            "z_max",
            SheetOrientation.CONSTANT_Z,
            bounds.z_max_m,
            bounds.r_inner_m,
            bounds.r_outer_m,
            VectorRZ(0.0, 1.0),
        )
        return tuple(surfaces)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "uniform_axisymmetric_magnetization",
            "source_id": self.source_id,
            "permanent_magnet_material_id": self.permanent_magnet_material_id,
            "region_id": self.region_id,
            "representation": self.representation.value,
            "bounds": self.bounds.to_dict(),
            "magnetization_direction_rz": self.magnetization_direction_rz.to_dict(),
            "magnetization_a_per_m": self.magnetization_a_per_m.to_dict(),
            "temperature_k": self.temperature_k,
            "bound_volume_current_density_phi_a_per_m2": 0.0,
            "equivalent_bound_current_sheets": tuple(
                sheet.to_dict() for sheet in self.equivalent_bound_current_sheets()
            ),
        }
