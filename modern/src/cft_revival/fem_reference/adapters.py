"""Read-only adapters from accepted geometry/magnetics contracts."""

from __future__ import annotations

from math import ceil, isfinite

from cft_revival.geometry import (
    AxisymmetricCFTGeometry,
    PermanentMagnetAuthority,
    to_magnetics_handoff,
)
from cft_revival.magnetics import (
    MU0_H_PER_M,
    LinearPermeability,
    SmCoPermanentMagnet,
    checked_synthetic_smco_like_magnet,
    content_sha256,
)

from .mesh import Polygon, build_body_fitted_mesh
from .models import Domain, FEMProblem, FEMValidationError, Region, SheetSource


def geometry_polygons(
    geometry: AxisymmetricCFTGeometry,
) -> tuple[tuple[str, Polygon], ...]:
    return tuple(
        (
            region.region_id,
            (
                (region.r_inner_start_m, region.z_min_m),
                (region.r_outer_start_m, region.z_min_m),
                (region.r_outer_end_m, region.z_max_m),
                (region.r_inner_end_m, region.z_max_m),
            ),
        )
        for region in geometry.regions
    )


def _point_in_polygon(polygon: Polygon, radial: float, axial: float) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (previous[1] > axial) != (current[1] > axial):
            crossing = previous[0] + (axial - previous[1]) * (
                current[0] - previous[0]
            ) / (current[1] - previous[1])
            if radial < crossing:
                inside = not inside
        previous = current
    return inside


def design_domain(
    geometry: AxisymmetricCFTGeometry, *, padding_factor: float = 1.5
) -> Domain:
    if not isfinite(padding_factor) or padding_factor <= 0.0:
        raise FEMValidationError("padding factor must be finite and positive")
    outer = max(
        max(region.r_outer_start_m, region.r_outer_end_m) for region in geometry.regions
    )
    z_min = min(region.z_min_m for region in geometry.regions)
    z_max = max(region.z_max_m for region in geometry.regions)
    characteristic = max(outer, z_max - z_min)
    return Domain(
        0.0,
        outer + padding_factor * characteristic,
        z_min - padding_factor * characteristic,
        z_max + padding_factor * characteristic,
    )


def _registry(geometry: AxisymmetricCFTGeometry) -> dict[str, object]:
    permanent = checked_synthetic_smco_like_magnet()
    output: dict[str, object] = {}
    for definition in geometry.materials:
        if definition.material_id == permanent.material_id:
            output[definition.material_id] = permanent
        else:
            output[definition.material_id] = LinearPermeability(
                definition.material_id, definition.relative_permeability
            )
    return output


def adapt_geometry(
    geometry: AxisymmetricCFTGeometry,
    *,
    domain: Domain | None = None,
    padding_factor: float = 1.5,
) -> tuple[FEMProblem, tuple[tuple[str, Polygon], ...]]:
    """Build a FEM problem only after the accepted magnetics handoff validates."""

    registry = _registry(geometry)
    handoff = to_magnetics_handoff(geometry, material_registry=registry)
    selected_domain = domain or design_domain(geometry, padding_factor=padding_factor)
    polygons = geometry_polygons(geometry)
    regions = [
        Region("ambient-background", "ambient-vacuum", 1.0 / MU0_H_PER_M)
    ]
    authority = geometry.permanent_magnet_plan.authority
    permanent = next(
        material for material in registry.values() if isinstance(material, SmCoPermanentMagnet)
    )
    for geometric in geometry.regions:
        definition = geometry.material_by_id(geometric.material_id)
        remanence_z = 0.0
        if geometric.role == "permanent_magnet":
            relative = permanent.recoil_relative_permeability
            if authority is PermanentMagnetAuthority.RECOIL_REMANENCE:
                remanence_z = (
                    permanent.remanence_t(permanent.reference_temperature_k)
                    * float(geometric.polarity)
                )
        else:
            relative = definition.relative_permeability
        regions.append(
            Region(
                geometric.region_id,
                geometric.material_id,
                1.0 / (MU0_H_PER_M * relative),
                0.0,
                remanence_z,
            )
        )

    def region_at(radial: float, axial: float) -> str:
        matches = [
            region_id
            for region_id, polygon in polygons
            if _point_in_polygon(polygon, radial, axial)
        ]
        if len(matches) > 1:
            raise FEMValidationError("accepted geometry produced overlapping FEM regions")
        return matches[0] if matches else "ambient-background"

    sheets: list[SheetSource] = []
    for source in handoff.magnetization_sources:
        for sheet in source.equivalent_bound_current_sheets():
            sheets.append(
                SheetSource(
                    f"{sheet.source_id}-{sheet.surface_name}",
                    sheet.orientation.value,
                    sheet.coordinate_m,
                    sheet.span_min_m,
                    sheet.span_max_m,
                    sheet.k_phi_a_per_m / source.material.recoil_relative_permeability,
                )
            )
    active = [region for region in handoff.regions if region.region_id != "ambient-background"]
    source_z_min = min(region.bounds.z_min_m for region in active)
    source_z_max = max(region.bounds.z_max_m for region in active)
    problem = FEMProblem(
        problem_id=f"fem-reference-{handoff.problem_id}",
        domain=selected_domain,
        regions=tuple(regions),
        region_at=region_at,
        sheets=tuple(sheets),
        source_center_z_m=0.5 * (source_z_min + source_z_max),
        outer_boundary="dipole_robin",
        geometry_sha256=geometry.canonical_sha256,
        magnetics_sha256=content_sha256(handoff.to_dict()),
        metadata=(
            ("geometry_config_id", geometry.config_id),
            ("pm_authority", authority.value),
            ("mesh_geometry", "all accepted linear polygon boundaries"),
        ),
    )
    return problem, polygons


def mesh_geometry(
    geometry: AxisymmetricCFTGeometry,
    *,
    radial_divisions: int,
    axial_divisions: int,
    padding_factor: float = 1.5,
):
    problem, polygons = adapt_geometry(geometry, padding_factor=padding_factor)
    mesh = build_body_fitted_mesh(
        problem.domain,
        polygons,
        problem.region_at,
        radial_divisions=radial_divisions,
        axial_divisions=axial_divisions,
    )
    return problem, mesh


def _hardware_coordinates(
    mandatory: tuple[float, ...], target_h: float
) -> tuple[float, ...]:
    lengths = [
        right - left for left, right in zip(mandatory, mandatory[1:])
    ]
    counts = [max(1, ceil(length / target_h)) for length in lengths]
    for _ in range(10000):
        sizes = [length / count for length, count in zip(lengths, counts)]
        changed = False
        for index in range(len(sizes) - 1):
            ratio = max(sizes[index], sizes[index + 1]) / min(
                sizes[index], sizes[index + 1]
            )
            if ratio > 1.3:
                larger = index if sizes[index] > sizes[index + 1] else index + 1
                counts[larger] += 1
                changed = True
        if not changed:
            break
    coordinates = [mandatory[0]]
    for left, right, count in zip(mandatory, mandatory[1:], counts):
        coordinates.extend(
            left + (right - left) * index / count for index in range(1, count + 1)
        )
    return tuple(coordinates)


def graded_mesh_geometry(
    geometry: AxisymmetricCFTGeometry,
    *,
    bore_elements: int = 4,
    feature_elements: int = 3,
    padding_factor: float = 1.5,
):
    """Build a feature-graded initial mesh for subsequent nested refinement."""

    if (
        isinstance(bore_elements, bool)
        or not isinstance(bore_elements, int)
        or bore_elements < 2
        or isinstance(feature_elements, bool)
        or not isinstance(feature_elements, int)
        or feature_elements < 2
    ):
        raise FEMValidationError("graded feature element counts must be integers >=2")
    problem, polygons = adapt_geometry(geometry, padding_factor=padding_factor)
    hardware_r_max = max(point[0] for _, polygon in polygons for point in polygon)
    hardware_z_min = min(point[1] for _, polygon in polygons for point in polygon)
    hardware_z_max = max(point[1] for _, polygon in polygons for point in polygon)
    active = [
        region
        for region in geometry.regions
        if geometry.material_by_id(region.material_id).relative_permeability != 1.0
    ]
    feature_width = min(
        *(
            min(
                region.r_outer_start_m - region.r_inner_start_m,
                region.r_outer_end_m - region.r_inner_end_m,
                region.z_max_m - region.z_min_m,
            )
            for region in active
        ),
        geometry.chamber.dielectric_thickness_m,
    )
    bore_radius = geometry.chamber.outer_radius_m
    bore_h = bore_radius / bore_elements
    feature_h = feature_width / feature_elements
    local_h = min(bore_h, feature_h)
    mandatory_r = tuple(
        sorted(
            {
                0.0,
                hardware_r_max,
                problem.domain.r_max_m,
                bore_radius,
                *(
                    value
                    for region in geometry.regions
                    for value in (
                        (
                            region.r_inner_start_m,
                            region.r_outer_start_m,
                        )
                        if (
                            region.r_inner_start_m == region.r_inner_end_m
                            and region.r_outer_start_m == region.r_outer_end_m
                        )
                        else ()
                    )
                ),
            }
        )
    )
    mandatory_z = tuple(
        sorted(
            {
                problem.domain.z_min_m,
                hardware_z_min,
                hardware_z_max,
                problem.domain.z_max_m,
                *(point[1] for _, polygon in polygons for point in polygon),
                *(stage.z_min_m for stage in geometry.stages),
                *(stage.z_max_m for stage in geometry.stages),
            }
        )
    )
    axial_coordinates = _hardware_coordinates(mandatory_z, local_h)

    def local_size(radial: float, _axial: float) -> float:
        exterior_distance = max(radial - hardware_r_max, 0.0)
        return local_h * min(
            1.3,
            1.01 ** min(exterior_distance / local_h, 100.0),
        )

    mesh = build_body_fitted_mesh(
        problem.domain,
        polygons,
        problem.region_at,
        radial_divisions=2,
        axial_divisions=len(axial_coordinates) - 1,
        axial_coordinates=axial_coordinates,
        size_field=local_size,
        protected_radii_m=mandatory_r,
        protected_z_m=mandatory_z,
    )
    return problem, mesh
