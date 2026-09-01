"""Complete oriented boundary topology for meridional material regions."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from .model import (
    AxisymmetricCFTGeometry,
    GeometryValidationError,
    MeridionalRegion,
    _ulp_close,
)

SURFACE_ORDER = ("inner", "outer", "z_min", "z_max")


@dataclass(frozen=True, slots=True)
class InterfaceSurfaceDescriptor:
    interface_id: str
    region_id: str
    adjacent_region_id: str
    surface: str
    orientation: str
    start_rz_m: tuple[float, float]
    end_rz_m: tuple[float, float]
    unit_normal_rz: tuple[float, float]
    free_surface_current_phi_a_per_m: float = 0.0

    def __post_init__(self) -> None:
        if self.surface not in SURFACE_ORDER:
            raise GeometryValidationError("unsupported interface surface")
        if self.orientation not in (
            "outward_from_region_to_adjacent",
            "outward_from_region_to_ambient",
            "outward_to_symmetry_axis",
        ):
            raise GeometryValidationError("unsupported interface orientation")
        magnitude = hypot(*self.unit_normal_rz)
        if not isfinite(magnitude) or not _ulp_close(magnitude, 1.0, max_ulps=4.0):
            raise GeometryValidationError("interface normal must be a finite unit vector")
        if self.start_rz_m == self.end_rz_m:
            raise GeometryValidationError("interface segment must have positive length")

    def to_dict(self) -> dict[str, object]:
        return {
            "interface_id": self.interface_id,
            "region_id": self.region_id,
            "adjacent_region_id": self.adjacent_region_id,
            "surface": self.surface,
            "orientation": self.orientation,
            "start_rz_m": list(self.start_rz_m),
            "end_rz_m": list(self.end_rz_m),
            "unit_normal_rz": {
                "radial": self.unit_normal_rz[0],
                "axial": self.unit_normal_rz[1],
            },
            "free_surface_current_phi_a_per_m": (
                self.free_surface_current_phi_a_per_m
            ),
        }


def _radius(region: MeridionalRegion, surface: str, z_m: float) -> float:
    inner, outer = region.radial_interval_at(z_m)
    return inner if surface == "inner" else outer


def _slope(region: MeridionalRegion, surface: str) -> float:
    start = (
        region.r_inner_start_m if surface == "inner" else region.r_outer_start_m
    )
    end = region.r_inner_end_m if surface == "inner" else region.r_outer_end_m
    return (end - start) / region.axial_thickness_m


def _radial_normal(
    region: MeridionalRegion, surface: str
) -> tuple[float, float]:
    slope = _slope(region, surface)
    scale = hypot(1.0, slope)
    if surface == "outer":
        return (1.0 / scale, -slope / scale)
    return (-1.0 / scale, slope / scale)


def _radial_adjacent(
    region: MeridionalRegion,
    surface: str,
    z_m: float,
    regions: tuple[MeridionalRegion, ...],
) -> str:
    opposite = "outer" if surface == "inner" else "inner"
    matches: list[str] = []
    target_radius = _radius(region, surface, z_m)
    if surface == "inner" and target_radius == 0.0:
        return "symmetry-axis"
    target_slope = _slope(region, surface)
    for other in regions:
        if (
            other.region_id == region.region_id
            or not other.z_min_m < z_m < other.z_max_m
        ):
            continue
        if _ulp_close(_radius(other, opposite, z_m), target_radius) and _ulp_close(
            _slope(other, opposite), target_slope, max_ulps=4.0
        ):
            matches.append(other.region_id)
    if len(matches) > 1:
        raise GeometryValidationError(
            f"ambiguous adjacency on {region.region_id}:{surface}"
        )
    return matches[0] if matches else "ambient-background"


def _axial_adjacent(
    region: MeridionalRegion,
    surface: str,
    radius_m: float,
    regions: tuple[MeridionalRegion, ...],
) -> str:
    z_m = region.z_min_m if surface == "z_min" else region.z_max_m
    matches: list[str] = []
    for other in regions:
        if other.region_id == region.region_id:
            continue
        other_z = other.z_max_m if surface == "z_min" else other.z_min_m
        if other_z != z_m:
            continue
        inner, outer = other.radial_interval_at(other_z)
        if inner < radius_m < outer:
            matches.append(other.region_id)
    if len(matches) > 1:
        raise GeometryValidationError(
            f"ambiguous adjacency on {region.region_id}:{surface}"
        )
    return matches[0] if matches else "ambient-background"


def _radial_segments(
    region: MeridionalRegion,
    surface: str,
    regions: tuple[MeridionalRegion, ...],
) -> list[tuple[float, float, str]]:
    breaks = {region.z_min_m, region.z_max_m}
    for other in regions:
        if other.region_id == region.region_id:
            continue
        low = max(region.z_min_m, other.z_min_m)
        high = min(region.z_max_m, other.z_max_m)
        if high > low:
            midpoint = (low + high) / 2.0
            opposite = "outer" if surface == "inner" else "inner"
            if _ulp_close(
                _radius(region, surface, midpoint),
                _radius(other, opposite, midpoint),
            ) and _ulp_close(
                _slope(region, surface),
                _slope(other, opposite),
                max_ulps=4.0,
            ):
                breaks.update((low, high))
    ordered = sorted(breaks)
    return [
        (
            left,
            right,
            _radial_adjacent(
                region,
                surface,
                (left + right) / 2.0,
                regions,
            ),
        )
        for left, right in zip(ordered, ordered[1:])
        if right > left
    ]


def _axial_segments(
    region: MeridionalRegion,
    surface: str,
    regions: tuple[MeridionalRegion, ...],
) -> list[tuple[float, float, str]]:
    z_m = region.z_min_m if surface == "z_min" else region.z_max_m
    inner, outer = region.radial_interval_at(z_m)
    breaks = {inner, outer}
    for other in regions:
        if other.region_id == region.region_id:
            continue
        other_z = other.z_max_m if surface == "z_min" else other.z_min_m
        if other_z != z_m:
            continue
        other_inner, other_outer = other.radial_interval_at(other_z)
        low = max(inner, other_inner)
        high = min(outer, other_outer)
        if high > low:
            breaks.update((low, high))
    ordered = sorted(breaks)
    return [
        (
            left,
            right,
            _axial_adjacent(
                region,
                surface,
                (left + right) / 2.0,
                regions,
            ),
        )
        for left, right in zip(ordered, ordered[1:])
        if right > left
    ]


def interface_topology(
    geometry: AxisymmetricCFTGeometry,
) -> tuple[InterfaceSurfaceDescriptor, ...]:
    """Return every oriented inner/outer/z-min/z-max surface segment."""

    descriptors: list[InterfaceSurfaceDescriptor] = []
    for region in geometry.regions:
        for surface in SURFACE_ORDER:
            if surface in ("inner", "outer"):
                segments = _radial_segments(region, surface, geometry.regions)
                normal = _radial_normal(region, surface)
                for index, (z_min, z_max, adjacent) in enumerate(segments):
                    descriptors.append(
                        InterfaceSurfaceDescriptor(
                            interface_id=(
                                f"{region.region_id}-{surface}-{index:03d}"
                            ),
                            region_id=region.region_id,
                            adjacent_region_id=adjacent,
                            surface=surface,
                            orientation=(
                                "outward_to_symmetry_axis"
                                if adjacent == "symmetry-axis"
                                else (
                                    "outward_from_region_to_ambient"
                                    if adjacent == "ambient-background"
                                    else "outward_from_region_to_adjacent"
                                )
                            ),
                            start_rz_m=(_radius(region, surface, z_min), z_min),
                            end_rz_m=(_radius(region, surface, z_max), z_max),
                            unit_normal_rz=normal,
                        )
                    )
            else:
                segments = _axial_segments(region, surface, geometry.regions)
                normal = (0.0, -1.0 if surface == "z_min" else 1.0)
                z_m = region.z_min_m if surface == "z_min" else region.z_max_m
                for index, (r_inner, r_outer, adjacent) in enumerate(segments):
                    descriptors.append(
                        InterfaceSurfaceDescriptor(
                            interface_id=(
                                f"{region.region_id}-{surface}-{index:03d}"
                            ),
                            region_id=region.region_id,
                            adjacent_region_id=adjacent,
                            surface=surface,
                            orientation=(
                                "outward_from_region_to_ambient"
                                if adjacent == "ambient-background"
                                else "outward_from_region_to_adjacent"
                            ),
                            start_rz_m=(r_inner, z_m),
                            end_rz_m=(r_outer, z_m),
                            unit_normal_rz=normal,
                        )
                    )
    interface_ids = [descriptor.interface_id for descriptor in descriptors]
    if len(interface_ids) != len(set(interface_ids)):
        raise GeometryValidationError("interface identifiers must be unique")
    return tuple(descriptors)
