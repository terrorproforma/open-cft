"""Geometry-only descriptors for screening and optimization plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, pi

from .model import AxisymmetricCFTGeometry, GeometryValidationError, MaterialKind


@dataclass(frozen=True, slots=True)
class GeometryDescriptors:
    active_volume_m3: float
    channel_volume_m3: float
    channel_inlet_area_m2: float
    channel_exit_area_m2: float
    stage_pitch_m: float
    cusp_count: int
    minimum_radial_gap_m: float
    minimum_axial_gap_m: float
    magnet_mass_estimate_kg: float
    envelope_radius_m: float
    envelope_z_min_m: float
    envelope_z_max_m: float
    manufacturability_warnings: tuple[str, ...]
    design_variable_names: tuple[str, ...]
    design_variable_values_si: tuple[float, ...]

    def __post_init__(self) -> None:
        values = (
            self.active_volume_m3,
            self.channel_volume_m3,
            self.channel_inlet_area_m2,
            self.channel_exit_area_m2,
            self.stage_pitch_m,
            self.minimum_radial_gap_m,
            self.minimum_axial_gap_m,
            self.magnet_mass_estimate_kg,
            self.envelope_radius_m,
            self.envelope_z_min_m,
            self.envelope_z_max_m,
            *self.design_variable_values_si,
        )
        if any(not isfinite(value) for value in values):
            raise GeometryValidationError(
                "derived geometry descriptors must be finite and representable"
            )
        positive = (
            self.active_volume_m3,
            self.channel_volume_m3,
            self.channel_inlet_area_m2,
            self.channel_exit_area_m2,
            self.stage_pitch_m,
            self.minimum_radial_gap_m,
            self.minimum_axial_gap_m,
            self.magnet_mass_estimate_kg,
            self.envelope_radius_m,
        )
        if any(value <= 0.0 for value in positive):
            raise GeometryValidationError(
                "derived geometry descriptors must remain representably positive"
            )
        if (
            isinstance(self.cusp_count, bool)
            or not isinstance(self.cusp_count, int)
            or self.cusp_count < 1
        ):
            raise GeometryValidationError("cusp_count must be an integer >= 1")
        if len(self.design_variable_names) != len(self.design_variable_values_si):
            raise GeometryValidationError("descriptor design vector lengths differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "active_volume_m3": self.active_volume_m3,
            "channel_volume_m3": self.channel_volume_m3,
            "channel_inlet_area_m2": self.channel_inlet_area_m2,
            "channel_exit_area_m2": self.channel_exit_area_m2,
            "stage_pitch_m": self.stage_pitch_m,
            "cusp_count": self.cusp_count,
            "minimum_radial_gap_m": self.minimum_radial_gap_m,
            "minimum_axial_gap_m": self.minimum_axial_gap_m,
            "magnet_mass_estimate_kg": self.magnet_mass_estimate_kg,
            "magnet_density_basis": (
                "density_kg_per_m3 from the geometry material definition; assumed, "
                "not a selected or measured SmCo grade"
            ),
            "envelope": {
                "radius_m": self.envelope_radius_m,
                "z_min_m": self.envelope_z_min_m,
                "z_max_m": self.envelope_z_max_m,
            },
            "manufacturability_warnings": list(self.manufacturability_warnings),
            "design_variable_vector": {
                "names": list(self.design_variable_names),
                "values_si": list(self.design_variable_values_si),
            },
            "claim_limit": "geometric descriptors only; no propulsion performance prediction",
        }


def _design_variable_value(geometry: AxisymmetricCFTGeometry, name: str) -> float:
    chamber = geometry.chamber
    stage_magnet = geometry.region_by_id(geometry.stages[0].magnet_region_id)
    values = {
        "chamber_outer_radius_m": chamber.outer_radius_m,
        "chamber_length_m": chamber.length_m,
        "dielectric_thickness_m": chamber.dielectric_thickness_m,
        "magnet_inner_radius_m": stage_magnet.r_inner_start_m,
        "magnet_outer_radius_m": stage_magnet.r_outer_start_m,
        "stage_pitch_m": geometry.stages[0].pitch_m,
        "exit_length_m": chamber.exit_length_m,
        "exit_outer_radius_m": chamber.exit_outer_radius_m,
    }
    try:
        return values[name]
    except KeyError as error:
        raise GeometryValidationError(f"unsupported design variable {name!r}") from error


def _representable_mass_kg(volume_m3: float, density_kg_per_m3: float) -> float:
    mass = volume_m3 * density_kg_per_m3
    if not isfinite(mass):
        raise GeometryValidationError("magnet mass estimate is not representable")
    if volume_m3 > 0.0 and density_kg_per_m3 > 0.0 and mass == 0.0:
        raise GeometryValidationError(
            "nonzero-density magnet mass underflowed to zero"
        )
    return mass


def compute_descriptors(geometry: AxisymmetricCFTGeometry) -> GeometryDescriptors:
    channel_regions = tuple(
        region
        for region in geometry.regions
        if region.role in ("injector_plasma", "channel_plasma")
    )
    magnet_regions = tuple(
        geometry.region_by_id(stage.magnet_region_id) for stage in geometry.stages
    )
    try:
        channel_volume = fsum(region.volume_m3 for region in channel_regions)
        active_volume = fsum(
            region.volume_m3
            for region in geometry.regions
            if region.role not in ("anode",)
        )
    except (OverflowError, ValueError) as error:
        raise GeometryValidationError(
            "derived geometry volume is not representable"
        ) from error
    chamber = geometry.chamber
    try:
        inlet_area = pi * (
            chamber.outer_radius_m**2 - chamber.inner_radius_m**2
        )
        exit_area = pi * (
            chamber.exit_outer_radius_m**2 - chamber.inner_radius_m**2
        )
    except OverflowError as error:
        raise GeometryValidationError(
            "derived channel area is not representable"
        ) from error
    if not isfinite(inlet_area) or not isfinite(exit_area):
        raise GeometryValidationError("derived channel area is not representable")
    axial_gaps = tuple(
        right.z_min_m - left.z_max_m
        for left, right in zip(magnet_regions, magnet_regions[1:])
    )
    dielectric_regions = tuple(
        region for region in geometry.regions if region.role == "dielectric_wall"
    )
    radial_gaps: list[float] = []
    for magnet in magnet_regions:
        for wall in dielectric_regions:
            z_min = max(magnet.z_min_m, wall.z_min_m)
            z_max = min(magnet.z_max_m, wall.z_max_m)
            if z_max <= z_min:
                continue
            radial_gaps.extend(
                (
                    magnet.r_inner_start_m - wall.radial_interval_at(z_min)[1],
                    magnet.r_inner_end_m - wall.radial_interval_at(z_max)[1],
                )
            )
    if not radial_gaps:
        raise GeometryValidationError(
            "minimum radial PM clearance has no overlapping wall support"
        )
    radial_gap = min(radial_gaps)
    pm_materials = tuple(
        material
        for material in geometry.materials
        if material.category is MaterialKind.PERMANENT_MAGNET
    )
    if len(pm_materials) != 1:
        raise GeometryValidationError(
            "mass estimate requires exactly one permanent-magnet material"
        )
    smco = pm_materials[0]
    if smco.density_kg_per_m3 is None:
        raise GeometryValidationError("magnet density is required for mass estimate")
    try:
        magnet_volume = fsum(region.volume_m3 for region in magnet_regions)
        mass = _representable_mass_kg(
            magnet_volume, smco.density_kg_per_m3
        )
    except (OverflowError, ValueError) as error:
        raise GeometryValidationError("magnet mass estimate is not representable") from error
    warnings: list[str] = []
    rules = geometry.manufacturing
    if radial_gap - 2.0 * rules.radial_tolerance_m < rules.thermal_clearance_m:
        warnings.append(
            "Worst-case radial tolerance stack reduces magnet/dielectric gap below "
            "the nominal thermal-clearance requirement."
        )
    minimum_axial_gap = min(axial_gaps)
    if minimum_axial_gap - 2.0 * rules.axial_tolerance_m < rules.minimum_clearance_m:
        warnings.append(
            "Worst-case axial tolerance stack reduces an inter-magnet gap below "
            "the minimum clearance."
        )
    if geometry.chamber.exit_length_m > 0.0:
        warnings.append(
            "Divergent annulus requires tapered ceramic manufacturing and a solver "
            "representation that preserves the linear boundary."
        )
    if any(material.assumption for material in geometry.materials):
        warnings.append(
            "Material properties include screening assumptions and require grade-specific "
            "thermal, structural, and magnetic qualification."
        )
    values = tuple(
        _design_variable_value(geometry, name)
        for name in geometry.design_variable_order
    )
    return GeometryDescriptors(
        active_volume_m3=active_volume,
        channel_volume_m3=channel_volume,
        channel_inlet_area_m2=inlet_area,
        channel_exit_area_m2=exit_area,
        stage_pitch_m=geometry.stages[0].pitch_m,
        cusp_count=len(geometry.stages) - 1,
        minimum_radial_gap_m=radial_gap,
        minimum_axial_gap_m=minimum_axial_gap,
        magnet_mass_estimate_kg=mass,
        envelope_radius_m=max(
            max(region.r_outer_start_m, region.r_outer_end_m)
            for region in geometry.regions
        ),
        envelope_z_min_m=min(region.z_min_m for region in geometry.regions),
        envelope_z_max_m=max(region.z_max_m for region in geometry.regions),
        manufacturability_warnings=tuple(warnings),
        design_variable_names=geometry.design_variable_order,
        design_variable_values_si=values,
    )


def geometry_with_design_vector(
    geometry: AxisymmetricCFTGeometry, values_si: tuple[float, ...]
) -> dict[str, float]:
    """Return a checked name/value mapping for an external optimizer.

    Geometry regeneration is intentionally delegated to a variant-specific
    generator so invalid coupled radii cannot be silently patched in-place.
    """

    if not isinstance(values_si, tuple) or len(values_si) != len(
        geometry.design_variable_order
    ):
        raise GeometryValidationError("design vector length must match variable order")
    mapping: dict[str, float] = {}
    for name, value in zip(geometry.design_variable_order, values_si):
        if isinstance(value, bool):
            raise GeometryValidationError("geometry design variables must be finite reals")
        converted = float(value)
        if not isfinite(converted):
            raise GeometryValidationError("geometry design variables must be finite reals")
        if converted < 0.0:
            raise GeometryValidationError("geometry design variables must be non-negative")
        mapping[name] = converted
    return mapping
