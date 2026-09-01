"""Strict geometry/magnetics adapters and conservative rasterization."""

from __future__ import annotations

import json
import sys
from math import fsum, hypot, isfinite, pi
from typing import Mapping

from cft_revival.fields import AxisymmetricDomain, MU0_H_PER_M
from cft_revival.geometry import AxisymmetricCFTGeometry, to_magnetics_handoff
from cft_revival.magnetics import (
    AxisymmetricMaterialProblemContract,
    ConstitutiveLawKind,
    LinearPermeability,
    PermanentMagnetRepresentation,
    SmCoPermanentMagnet,
    TabulatedBHCurve,
    checked_synthetic_smco_like_magnet,
    content_sha256,
    serialize_handoff,
)

from .models import (
    MaterialFieldValidationError,
    RasterDiagnostic,
    RasterizedMaterialProblem,
    WeakActionDiagnostic,
)

_MAX_ESTIMATED_RASTER_BYTES = 8 * 1024**3


def raster_memory_preflight(
    domain: AxisymmetricDomain, *, enforce: bool = True
) -> dict[str, int | bool]:
    """Report and optionally enforce a conservative host-memory bound."""
    count = domain.shape[0] * domain.shape[1]
    # Python scalar/list ownership, conservative cut-cell work arrays, solver
    # copies and replay encoding peak well above the packed float64 footprint.
    estimated = count * 2048
    limit = _MAX_ESTIMATED_RASTER_BYTES
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            limit = min(
                limit,
                int(0.4 * status.available_physical),
                int(0.25 * status.available_page_file),
            )
    fits = estimated <= limit
    if enforce and estimated > 64 * 1024**2 and not fits:
        raise MaterialFieldValidationError(
            f"raster memory preflight requires about {estimated / 1024**3:.2f} GiB, "
            f"exceeding the safe {limit / 1024**3:.2f} GiB limit"
        )
    return {
        "estimated_raster_bytes": estimated,
        "safe_raster_bytes": limit,
        "fits": fits,
    }


def _preflight_raster_memory(domain: AxisymmetricDomain) -> int:
    """Compatibility projection for the enforced preflight."""
    report = raster_memory_preflight(domain)
    return int(report["estimated_raster_bytes"])


def design_domain(
    geometry: AxisymmetricCFTGeometry,
    *,
    radial_intervals: int = 48,
    axial_intervals: int = 96,
    padding_factor: float = 1.0,
) -> AxisymmetricDomain:
    """Return a finite box around the hardware; padding remains convergence evidence."""

    factor = float(padding_factor)
    if not isfinite(factor) or factor <= 0.0:
        raise MaterialFieldValidationError("padding_factor must be finite and positive")
    outer = max(max(region.r_outer_start_m, region.r_outer_end_m) for region in geometry.regions)
    z_min = min(region.z_min_m for region in geometry.regions)
    z_max = max(region.z_max_m for region in geometry.regions)
    characteristic = max(outer, z_max - z_min)
    return AxisymmetricDomain(
        radius_m=outer + factor * characteristic,
        z_min_m=z_min - factor * characteristic,
        z_max_m=z_max + factor * characteristic,
        radial_intervals=radial_intervals,
        axial_intervals=axial_intervals,
    )


def adapt_geometry(
    geometry: AxisymmetricCFTGeometry,
    domain: AxisymmetricDomain,
    *,
    authority: object | None = None,
    material_registry: Mapping[str, object] | None = None,
) -> RasterizedMaterialProblem:
    """Use accepted public handoffs, retaining geometry identity and tolerances."""

    plan = getattr(geometry, "permanent_magnet_plan", None)
    authority_value = getattr(authority, "value", authority)
    if authority is not None and plan is not None and plan.authority.value != authority_value:
        raise MaterialFieldValidationError(
            "requested PM authority conflicts with geometry representation plan"
        )
    registry = dict(material_registry or {})
    if not registry:
        permanent = checked_synthetic_smco_like_magnet()
        for definition in geometry.materials:
            category = getattr(definition.category, "value", definition.category)
            if category == "permanent_magnet":
                if definition.material_id != permanent.material_id:
                    raise MaterialFieldValidationError(
                        "geometry PM ID has no accepted default material law"
                    )
                registry[definition.material_id] = permanent
            else:
                registry[definition.material_id] = LinearPermeability(
                    definition.material_id, definition.relative_permeability
                )
    handoff = to_magnetics_handoff(geometry, material_registry=registry)
    return rasterize_handoff(handoff, domain, geometry=geometry)


def _intersection_volume(bounds, r0: float, r1: float, z0: float, z1: float) -> float:
    lower_r = max(bounds.r_inner_m, r0)
    upper_r = min(bounds.r_outer_m, r1)
    lower_z = max(bounds.z_min_m, z0)
    upper_z = min(bounds.z_max_m, z1)
    if upper_r <= lower_r or upper_z <= lower_z:
        return 0.0
    return pi * (upper_r * upper_r - lower_r * lower_r) * (upper_z - lower_z)


def _intersection_area(bounds, r0: float, r1: float, z0: float, z1: float) -> float:
    return max(0.0, min(bounds.r_outer_m, r1) - max(bounds.r_inner_m, r0)) * max(
        0.0, min(bounds.z_max_m, z1) - max(bounds.z_min_m, z0)
    )


def _cell_bounds(domain: AxisymmetricDomain, i: int, j: int) -> tuple[float, float, float, float]:
    r = i * domain.dr_m
    z = domain.z_min_m + j * domain.dz_m
    return (
        max(0.0, r - 0.5 * domain.dr_m),
        min(domain.radius_m, r + 0.5 * domain.dr_m),
        max(domain.z_min_m, z - 0.5 * domain.dz_m),
        min(domain.z_max_m, z + 0.5 * domain.dz_m),
    )


def _linear_node_weights(
    coordinates: tuple[float, ...], coordinate: float
) -> tuple[tuple[int, float], ...]:
    interior = coordinates[1:-1]
    if coordinate < interior[0] or coordinate > interior[-1]:
        raise MaterialFieldValidationError("PM sheet lies outside interior source support")
    for local, value in enumerate(interior, 1):
        if coordinate == value:
            return ((local, 1.0),)
        if value > coordinate:
            left = local - 1
            fraction = (coordinate - coordinates[left]) / (
                coordinates[local] - coordinates[left]
            )
            return ((left, 1.0 - fraction), (local, fraction))
    return ((len(coordinates) - 2, 1.0),)


def _stable_harmonic(left: float, right: float) -> float:
    smaller, larger = (left, right) if left <= right else (right, left)
    return smaller / (0.5 + 0.5 * smaller / larger)


def _polygon_line_breaks(
    polygon_rz: tuple[tuple[float, float], ...],
    *,
    axis: str,
    fixed: float,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    """Return exact line/polygon crossings for linear polygon edges."""
    crossings = [lower, upper]
    for first, second in zip(polygon_rz, polygon_rz[1:] + polygon_rz[:1]):
        varying_0, fixed_0 = (first[0], first[1]) if axis == "r" else (first[1], first[0])
        varying_1, fixed_1 = (second[0], second[1]) if axis == "r" else (second[1], second[0])
        if fixed_0 == fixed_1:
            if fixed == fixed_0:
                crossings.extend((max(lower, min(varying_0, varying_1)), min(upper, max(varying_0, varying_1))))
            continue
        fraction = (fixed - fixed_0) / (fixed_1 - fixed_0)
        if 0.0 <= fraction <= 1.0:
            value = varying_0 + fraction * (varying_1 - varying_0)
            if lower < value < upper:
                crossings.append(value)
    return tuple(sorted(set(crossings)))


def _point_in_polygon(
    polygon_rz: tuple[tuple[float, float], ...], radial: float, axial: float
) -> bool:
    inside = False
    previous = polygon_rz[-1]
    for current in polygon_rz:
        r0, z0 = previous
        r1, z1 = current
        if (z0 > axial) != (z1 > axial):
            crossing = r0 + (axial - z0) * (r1 - r0) / (z1 - z0)
            if radial < crossing:
                inside = not inside
        previous = current
    return inside


def _clip_polygon_rectangle(
    polygon_rz: tuple[tuple[float, float], ...],
    r0: float,
    r1: float,
    z0: float,
    z1: float,
) -> tuple[tuple[float, float], ...]:
    """Clip a linear polygon exactly to an axis-aligned control volume."""
    vertices = list(polygon_rz)
    boundaries = (
        (lambda point: point[0] >= r0, lambda a, b: (r0, a[1] + (b[1] - a[1]) * (r0 - a[0]) / (b[0] - a[0]))),
        (lambda point: point[0] <= r1, lambda a, b: (r1, a[1] + (b[1] - a[1]) * (r1 - a[0]) / (b[0] - a[0]))),
        (lambda point: point[1] >= z0, lambda a, b: (a[0] + (b[0] - a[0]) * (z0 - a[1]) / (b[1] - a[1]), z0)),
        (lambda point: point[1] <= z1, lambda a, b: (a[0] + (b[0] - a[0]) * (z1 - a[1]) / (b[1] - a[1]), z1)),
    )
    for inside, crossing in boundaries:
        if not vertices:
            break
        clipped: list[tuple[float, float]] = []
        previous = vertices[-1]
        previous_inside = inside(previous)
        for current in vertices:
            current_inside = inside(current)
            if current_inside != previous_inside:
                clipped.append(crossing(previous, current))
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        vertices = clipped
    return tuple(vertices)


def _polygon_rectangle_area(
    polygon_rz: tuple[tuple[float, float], ...],
    r0: float,
    r1: float,
    z0: float,
    z1: float,
) -> float:
    clipped = _clip_polygon_rectangle(polygon_rz, r0, r1, z0, z1)
    if len(clipped) < 3:
        return 0.0
    origin_r, origin_z = clipped[0]
    local = tuple(
        (radial - origin_r, axial - origin_z) for radial, axial in clipped
    )
    twice_area = fsum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(local, local[1:] + local[:1])
    )
    return 0.5 * abs(twice_area)


def _series_face_reluctivity(
    lower: float,
    upper: float,
    breakpoints: tuple[float, ...],
    reluctivity_at,
    *,
    radial: bool,
) -> float:
    """Exact 1-D series integration of ``a=nu/r`` along a node connection."""
    resistance = 0.0
    for first, second in zip(breakpoints, breakpoints[1:]):
        if second <= first:
            continue
        nu = reluctivity_at(0.5 * (first + second))
        resistance += (
            0.5 * (second * second - first * first) / nu
            if radial
            else (second - first) / nu
        )
    if resistance <= 0.0 or not isfinite(resistance):
        raise MaterialFieldValidationError("face series resistance is invalid")
    if radial:
        face_radius = 0.5 * (lower + upper)
        return face_radius * (upper - lower) / resistance
    return (upper - lower) / resistance


def _dipole_robin_alpha_radial(
    radius: float, axial: float, source_center_z: float
) -> float:
    """Return ``-partial_r(log(C*r^2/rho^3))`` at the outer radial side."""
    rho2 = radius * radius + (axial - source_center_z) ** 2
    return 3.0 * radius / rho2 - 2.0 / radius


def _dipole_robin_alpha_axial(
    radius: float, axial: float, source_center_z: float
) -> float:
    """Return ``-partial_n(log(C*r^2/rho^3))`` on either axial side."""
    offset = abs(axial - source_center_z)
    return 3.0 * offset / (radius * radius + offset * offset)


def _authority(contract: AxisymmetricMaterialProblemContract) -> str:
    recoil = any(
        region.permanent_magnet_representation
        is PermanentMagnetRepresentation.RECOIL_REMANENCE
        for region in contract.regions
    )
    equivalent = bool(contract.magnetization_sources)
    if recoil == equivalent:
        raise MaterialFieldValidationError(
            "handoff must select exactly one PM authority (recoil or equivalent current)"
        )
    return (
        PermanentMagnetRepresentation.RECOIL_REMANENCE.value
        if recoil
        else PermanentMagnetRepresentation.EQUIVALENT_BOUND_CURRENT.value
    )


def rasterize_handoff(
    contract: AxisymmetricMaterialProblemContract,
    domain: AxisymmetricDomain,
    *,
    geometry: AxisymmetricCFTGeometry | None = None,
) -> RasterizedMaterialProblem:
    """Rasterize a closed handoff with axisymmetric-volume/source conservation."""

    _preflight_raster_memory(domain)
    authority = _authority(contract)
    materials = {material.material_id: material for material in contract.materials}
    regions = tuple(sorted(contract.regions, key=lambda item: (-item.priority, item.region_id)))
    geometry_regions = {} if geometry is None else {
        region.region_id: region for region in geometry.regions
    }
    nr, nz = domain.shape
    material_ids: list[str] = []
    temperatures: list[float | None] = []
    polarities: list[int | None] = []
    reluctivity: list[float] = []
    br_r: list[float] = []
    br_z: list[float] = []
    free_current = [0.0] * (nr * nz)
    pm_bound_current = [0.0] * (nr * nz)
    minimum_priority = min(region.priority for region in regions)
    backgrounds = tuple(region for region in regions if region.priority == minimum_priority)
    if len(backgrounds) != 1:
        raise MaterialFieldValidationError("handoff requires one unambiguous background region")
    background = backgrounds[0]
    for left_index, left in enumerate(regions):
        if left is background:
            continue
        for right in regions[left_index + 1 :]:
            if right is background:
                continue
            radial_overlap = min(left.bounds.r_outer_m, right.bounds.r_outer_m) - max(
                left.bounds.r_inner_m, right.bounds.r_inner_m
            )
            axial_overlap = min(left.bounds.z_max_m, right.bounds.z_max_m) - max(
                left.bounds.z_min_m, right.bounds.z_min_m
            )
            if radial_overlap > 0.0 and axial_overlap > 0.0:
                raise MaterialFieldValidationError(
                    f"material regions {left.region_id} and {right.region_id} overlap"
                )

    def properties(region):
        material = materials[region.constitutive_law_id]
        if isinstance(material, TabulatedBHCurve):
            raise MaterialFieldValidationError(
                "nonlinear iron is gated until independent energy-residual validation"
            )
        if isinstance(material, LinearPermeability):
            return 1.0 / material.permeability_h_per_m, (0.0, 0.0), None
        if isinstance(material, SmCoPermanentMagnet):
            if region.constitutive_law_kind is not ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL:
                raise MaterialFieldValidationError("PM material/law-kind mismatch")
            temperature = material.reference_temperature_k
            magnitude = material.remanence_t(temperature)
            direction = region.magnetization_direction_rz
            assert direction is not None
            return (
                1.0 / (MU0_H_PER_M * material.recoil_relative_permeability),
                (magnitude * direction.radial, magnitude * direction.axial),
                temperature,
            )
        raise MaterialFieldValidationError("unsupported constitutive material")

    handoff_region_ids = {region.region_id for region in contract.regions}
    geometry_only_polygons: tuple[
        tuple[
            object,
            tuple[tuple[float, float], ...],
            float,
            tuple[float, float, float, float],
        ],
        ...,
    ] = ()
    if geometry is not None:
        geometry_only_polygons = tuple(
            (
                region,
                (
                    (region.r_inner_start_m, region.z_min_m),
                    (region.r_outer_start_m, region.z_min_m),
                    (region.r_outer_end_m, region.z_max_m),
                    (region.r_inner_end_m, region.z_max_m),
                ),
                1.0
                / (
                    MU0_H_PER_M
                    * geometry.material_by_id(region.material_id).relative_permeability
                ),
                (
                    min(region.r_inner_start_m, region.r_inner_end_m),
                    max(region.r_outer_start_m, region.r_outer_end_m),
                    region.z_min_m,
                    region.z_max_m,
                ),
            )
            for region in geometry.regions
            if region.region_id not in handoff_region_ids
        )

    def point_reluctivity(radial: float, axial: float) -> float:
        for region in regions:
            if region is background:
                continue
            bounds = region.bounds
            if (
                bounds.r_inner_m < radial < bounds.r_outer_m
                and bounds.z_min_m < axial < bounds.z_max_m
            ):
                return properties(region)[0]
        for _, polygon, value, _ in geometry_only_polygons:
            if _point_in_polygon(polygon, radial, axial):
                return value
        return properties(background)[0]

    def radial_breaks(r0: float, r1: float, axial: float) -> tuple[float, ...]:
        values = [r0, r1]
        for region in regions:
            if region is not background and region.bounds.z_min_m < axial < region.bounds.z_max_m:
                values.extend(
                    value
                    for value in (region.bounds.r_inner_m, region.bounds.r_outer_m)
                    if r0 < value < r1
                )
        for _, polygon, _, _ in geometry_only_polygons:
            values.extend(
                _polygon_line_breaks(
                    polygon, axis="r", fixed=axial, lower=r0, upper=r1
                )
            )
        return tuple(sorted(set(values)))

    def axial_breaks(z0: float, z1: float, radial: float) -> tuple[float, ...]:
        values = [z0, z1]
        for region in regions:
            if region is not background and region.bounds.r_inner_m < radial < region.bounds.r_outer_m:
                values.extend(
                    value
                    for value in (region.bounds.z_min_m, region.bounds.z_max_m)
                    if z0 < value < z1
                )
        for _, polygon, _, _ in geometry_only_polygons:
            values.extend(
                _polygon_line_breaks(
                    polygon, axis="z", fixed=radial, lower=z0, upper=z1
                )
            )
        return tuple(sorted(set(values)))

    def cut_cell_reluctivity(bounds: tuple[float, float, float, float]) -> float:
        area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
        if area <= 0.0:
            raise MaterialFieldValidationError("degenerate cut-cell face support")
        pieces = [
            (region, _intersection_area(region.bounds, *bounds))
            for region in regions
            if region is not background
        ]
        pieces = [(region, measure) for region, measure in pieces if measure > 0.0]
        occupied = fsum(measure for _, measure in pieces)
        if occupied > area * (1.0 + 1.0e-12):
            raise MaterialFieldValidationError("cut-cell face support overlaps")
        value = properties(background)[0] * max(0.0, area - occupied)
        value += fsum(properties(region)[0] * measure for region, measure in pieces)
        return value / area

    regions_by_radial_index = tuple(
        tuple(
            region
            for region in regions
            if region is not background
            and region.bounds.r_outer_m > _cell_bounds(domain, i, 0)[0]
            and region.bounds.r_inner_m < _cell_bounds(domain, i, 0)[1]
        )
        for i in range(nr)
    )
    polygons_by_radial_index = tuple(
        tuple(
            item
            for item in geometry_only_polygons
            if item[3][1] > _cell_bounds(domain, i, 0)[0]
            and item[3][0] < _cell_bounds(domain, i, 0)[1]
        )
        for i in range(nr)
    )
    for i in range(nr):
        for j in range(nz):
            bounds = _cell_bounds(domain, i, j)
            cell_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
            contributors = [
                (region, _intersection_area(region.bounds, *bounds))
                for region in regions_by_radial_index[i]
            ]
            contributors = [(region, volume) for region, volume in contributors if volume > 0.0]
            polygon_contributors = [
                (
                    original,
                    nu_value,
                    _polygon_rectangle_area(polygon, *bounds),
                )
                for original, polygon, nu_value, polygon_bounds in polygons_by_radial_index[i]
                if not (
                    bounds[1] <= polygon_bounds[0]
                    or bounds[0] >= polygon_bounds[1]
                    or bounds[3] <= polygon_bounds[2]
                    or bounds[2] >= polygon_bounds[3]
                )
            ]
            polygon_contributors = [
                item for item in polygon_contributors if item[2] > 0.0
            ]
            occupied = fsum(volume for _, volume in contributors) + fsum(
                volume for _, _, volume in polygon_contributors
            )
            if occupied > cell_area * (1.0 + 1.0e-12):
                raise MaterialFieldValidationError("material overlap exceeds control-volume size")
            background_volume = max(0.0, cell_area - occupied)
            weighted = [(background, background_volume), *contributors]
            nu_effective = 0.0
            g_r = 0.0
            g_z = 0.0
            for region, volume in weighted:
                nu_value, remanence, _ = properties(region)
                fraction = volume / cell_area
                nu_effective += fraction * nu_value
                g_r += fraction * nu_value * remanence[0]
                g_z += fraction * nu_value * remanence[1]
            nu_effective += fsum(
                nu_value * volume / cell_area
                for _, nu_value, volume in polygon_contributors
            )
            region = (
                max(contributors, key=lambda item: (item[0].priority, item[1]))[0]
                if contributors
                else background
            )
            _, _, temperature = properties(region)
            original = (
                max(polygon_contributors, key=lambda item: item[2])[0]
                if not contributors and polygon_contributors
                else geometry_regions.get(region.region_id)
            )
            material_ids.append(
                original.material_id if original is not None else region.constitutive_law_id
            )
            polarity = None if original is None else original.polarity
            temperatures.append(temperature)
            polarities.append(polarity)
            reluctivity.append(nu_effective)
            br_r.append(g_r / nu_effective)
            br_z.append(g_z / nu_effective)

    remanence_g_r_faces = [0.0] * ((nr - 1) * nz)
    remanence_g_z_faces = [0.0] * (nr * (nz - 1))
    if authority == PermanentMagnetRepresentation.RECOIL_REMANENCE.value:
        for region in contract.regions:
            material = materials[region.constitutive_law_id]
            if not isinstance(material, SmCoPermanentMagnet):
                continue
            direction = region.magnetization_direction_rz
            assert direction is not None
            nu_pm = 1.0 / (MU0_H_PER_M * material.recoil_relative_permeability)
            magnitude = material.remanence_t(material.reference_temperature_k)
            gr_value = nu_pm * magnitude * direction.axial
            gz_value = -nu_pm * magnitude * direction.radial
            for i in range(nr - 1):
                r0 = i * domain.dr_m
                r1 = r0 + domain.dr_m
                overlap_r = max(
                    0.0,
                    min(r1, region.bounds.r_outer_m)
                    - max(r0, region.bounds.r_inner_m),
                )
                if overlap_r == 0.0:
                    continue
                for j in range(nz):
                    _, _, z0, z1 = _cell_bounds(domain, i, j)
                    overlap_z = max(
                        0.0,
                        min(z1, region.bounds.z_max_m)
                        - max(z0, region.bounds.z_min_m),
                    )
                    if overlap_z > 0.0:
                        remanence_g_r_faces[i * nz + j] += (
                            gr_value
                            * overlap_r
                            * overlap_z
                            / (domain.dr_m * (z1 - z0))
                        )
            for i in range(nr):
                r0, r1, _, _ = _cell_bounds(domain, i, 0)
                overlap_r = max(
                    0.0,
                    min(r1, region.bounds.r_outer_m)
                    - max(r0, region.bounds.r_inner_m),
                )
                if overlap_r == 0.0:
                    continue
                for j in range(nz - 1):
                    z0 = domain.z_min_m + j * domain.dz_m
                    z1 = z0 + domain.dz_m
                    overlap_z = max(
                        0.0,
                        min(z1, region.bounds.z_max_m)
                        - max(z0, region.bounds.z_min_m),
                    )
                    remanence_g_z_faces[i * (nz - 1) + j] += (
                        gz_value
                        * overlap_z
                        * overlap_r
                        / (domain.dz_m * (r1 - r0))
                    )

    source_diagnostics: list[RasterDiagnostic] = []
    if authority == PermanentMagnetRepresentation.EQUIVALENT_BOUND_CURRENT.value:
        for source in contract.magnetization_sources:
            requested = 0.0
            represented = 0.0
            for sheet in source.equivalent_bound_current_sheets():
                sheet_k = (
                    sheet.k_phi_a_per_m
                    / source.material.recoil_relative_permeability
                )
                if sheet_k == 0.0:
                    continue
                span = sheet.span_max_m - sheet.span_min_m
                requested += sheet_k * span
                if sheet.orientation.value == "constant_r":
                    radial_coordinates = tuple(i * domain.dr_m for i in range(nr))
                    axial_coordinates = tuple(
                        domain.z_min_m + j * domain.dz_m for j in range(nz)
                    )
                    for i, radial_weight in _linear_node_weights(
                        radial_coordinates, sheet.coordinate_m
                    ):
                        for j in range(1, nz - 1):
                            r0, r1, z0, z1 = _cell_bounds(domain, i, j)
                            lower = max(z0, sheet.span_min_m)
                            upper = min(z1, sheet.span_max_m)
                            overlap = max(0.0, upper - lower)
                            if overlap == 0.0:
                                continue
                            centroid = 0.5 * (lower + upper)
                            contribution = sheet_k * overlap * radial_weight
                            for target_j, axial_weight in _linear_node_weights(
                                axial_coordinates, centroid
                            ):
                                tr0, tr1, tz0, tz1 = _cell_bounds(domain, i, target_j)
                                target_area = (tr1 - tr0) * (tz1 - tz0)
                                pm_bound_current[i * nz + target_j] += (
                                    contribution * axial_weight / target_area
                                )
                            represented += contribution
                else:
                    axial_coordinates = tuple(
                        domain.z_min_m + j * domain.dz_m for j in range(nz)
                    )
                    radial_coordinates = tuple(i * domain.dr_m for i in range(nr))
                    for j, axial_weight in _linear_node_weights(
                        axial_coordinates, sheet.coordinate_m
                    ):
                        for i in range(1, nr - 1):
                            r0, r1, z0, z1 = _cell_bounds(domain, i, j)
                            lower = max(r0, sheet.span_min_m)
                            upper = min(r1, sheet.span_max_m)
                            overlap = max(0.0, upper - lower)
                            if overlap == 0.0:
                                continue
                            centroid = 0.5 * (lower + upper)
                            contribution = sheet_k * overlap * axial_weight
                            for target_i, radial_weight in _linear_node_weights(
                                radial_coordinates, centroid
                            ):
                                tr0, tr1, tz0, tz1 = _cell_bounds(domain, target_i, j)
                                target_area = (tr1 - tr0) * (tz1 - tz0)
                                pm_bound_current[target_i * nz + j] += (
                                    contribution * radial_weight / target_area
                                )
                            represented += contribution
            scale = max(abs(requested), 1.0)
            source_diagnostics.append(
                RasterDiagnostic(
                    source.source_id,
                    0.0,
                    0.0,
                    0.0,
                    requested,
                    represented,
                    (represented - requested) / scale,
                )
            )
    else:
        for region in contract.regions:
            material = materials[region.constitutive_law_id]
            if not isinstance(material, SmCoPermanentMagnet):
                continue
            direction = region.magnetization_direction_rz
            assert direction is not None
            magnitude = material.remanence_t(material.reference_temperature_k)
            nu_pm = 1.0 / (MU0_H_PER_M * material.recoil_relative_permeability)
            area = (
                (region.bounds.r_outer_m - region.bounds.r_inner_m)
                * (region.bounds.z_max_m - region.bounds.z_min_m)
            )
            requested = nu_pm * magnitude * area
            source_diagnostics.append(
                RasterDiagnostic(
                    f"{region.region_id}-recoil-remanence",
                    0.0,
                    0.0,
                    0.0,
                    requested,
                    requested,
                    0.0,
                )
            )

    volume_diagnostics: list[RasterDiagnostic] = []
    for region in contract.regions:
        requested = pi * (
            region.bounds.r_outer_m**2 - region.bounds.r_inner_m**2
        ) * (region.bounds.z_max_m - region.bounds.z_min_m)
        represented = _intersection_volume(
            region.bounds,
            0.0,
            domain.radius_m,
            domain.z_min_m,
            domain.z_max_m,
        )
        clipped = (
            region.bounds.r_outer_m > domain.radius_m
            or region.bounds.z_min_m < domain.z_min_m
            or region.bounds.z_max_m > domain.z_max_m
        )
        if region.region_id != "ambient-background" and clipped:
            raise MaterialFieldValidationError(f"region {region.region_id} is clipped by domain")
        scale = max(requested, 1.0e-300)
        volume_diagnostics.append(
            RasterDiagnostic(
                region.region_id,
                requested,
                represented,
                (represented - requested) / scale,
            )
        )

    weak_actions: list[WeakActionDiagnostic] = []
    basis_ids = ("one", "r", "z", "r_z")
    analytical = {basis: 0.0 for basis in basis_ids}
    if authority == PermanentMagnetRepresentation.RECOIL_REMANENCE.value:
        for region in contract.regions:
            material = materials[region.constitutive_law_id]
            if not isinstance(material, SmCoPermanentMagnet):
                continue
            direction = region.magnetization_direction_rz
            assert direction is not None
            nu_pm = 1.0 / (MU0_H_PER_M * material.recoil_relative_permeability)
            magnitude = material.remanence_t(material.reference_temperature_k)
            gr = nu_pm * magnitude * direction.axial
            gz = -nu_pm * magnitude * direction.radial
            b = region.bounds
            area = (b.r_outer_m - b.r_inner_m) * (b.z_max_m - b.z_min_m)
            int_r = 0.5 * (b.r_outer_m**2 - b.r_inner_m**2) * (
                b.z_max_m - b.z_min_m
            )
            int_z = 0.5 * (b.z_max_m**2 - b.z_min_m**2) * (
                b.r_outer_m - b.r_inner_m
            )
            analytical["one"] += gr * area
            analytical["r"] += gr * int_r
            analytical["z"] += gz * area
            analytical["r_z"] += gr * int_z + gz * int_r
        represented_actions = {basis: 0.0 for basis in basis_ids}
        for i in range(nr - 1):
            r = (i + 0.5) * domain.dr_m
            for j in range(nz):
                z = domain.z_min_m + j * domain.dz_m
                _, _, z0, z1 = _cell_bounds(domain, i, j)
                measure = domain.dr_m * (z1 - z0)
                value = remanence_g_r_faces[i * nz + j] * measure
                represented_actions["one"] += value
                represented_actions["r"] += value * r
                represented_actions["r_z"] += value * z
        for i in range(nr):
            r = i * domain.dr_m
            r0, r1, _, _ = _cell_bounds(domain, i, 0)
            measure = domain.dz_m * (r1 - r0)
            for j in range(nz - 1):
                value = remanence_g_z_faces[i * (nz - 1) + j] * measure
                represented_actions["z"] += value
                represented_actions["r_z"] += value * r
        for basis in basis_ids:
            represented = represented_actions[basis]
            exact = analytical[basis]
            bias = represented - exact
            weak_actions.append(
                WeakActionDiagnostic(
                    f"recoil-gradient-{basis}",
                    exact,
                    represented,
                    abs(bias),
                    0.0 if exact == 0.0 and represented == 0.0 else abs(bias) / max(abs(exact), 1.0e-300),
                )
            )
    else:
        for source in contract.magnetization_sources:
            for sheet in source.equivalent_bound_current_sheets():
                k = (
                    sheet.k_phi_a_per_m
                    / source.material.recoil_relative_permeability
                )
                if sheet.orientation.value == "constant_r":
                    span = sheet.span_max_m - sheet.span_min_m
                    analytical["one"] += k * span
                    analytical["r"] += k * sheet.coordinate_m * span
                    analytical["z"] += k * 0.5 * (
                        sheet.span_max_m**2 - sheet.span_min_m**2
                    )
                    analytical["r_z"] += k * sheet.coordinate_m * 0.5 * (
                        sheet.span_max_m**2 - sheet.span_min_m**2
                    )
                else:
                    span = sheet.span_max_m - sheet.span_min_m
                    analytical["one"] += k * span
                    analytical["r"] += k * 0.5 * (
                        sheet.span_max_m**2 - sheet.span_min_m**2
                    )
                    analytical["z"] += k * sheet.coordinate_m * span
                    analytical["r_z"] += k * sheet.coordinate_m * 0.5 * (
                        sheet.span_max_m**2 - sheet.span_min_m**2
                    )
        represented_actions = {basis: 0.0 for basis in basis_ids}
        for i in range(nr):
            r = i * domain.dr_m
            for j in range(nz):
                source_value = pm_bound_current[i * nz + j]
                if source_value == 0.0:
                    continue
                z = domain.z_min_m + j * domain.dz_m
                bounds = _cell_bounds(domain, i, j)
                value = source_value * (
                    (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
                )
                represented_actions["one"] += value
                represented_actions["r"] += value * r
                represented_actions["z"] += value * z
                represented_actions["r_z"] += value * r * z
        for basis in basis_ids:
            represented = represented_actions[basis]
            exact = analytical[basis]
            bias = represented - exact
            weak_actions.append(
                WeakActionDiagnostic(
                    f"equivalent-value-{basis}",
                    exact,
                    represented,
                    abs(bias),
                    0.0 if exact == 0.0 and represented == 0.0 else abs(bias) / max(abs(exact), 1.0e-300),
                )
            )

    geometry_hash = "0" * 64
    tolerances = (0.0, 0.0)
    geometry_provenance: tuple[tuple[str, str, bool, str], ...] = ()
    geometry_pm_count = 0
    if geometry is not None:
        geometry_hash = geometry.canonical_sha256
        tolerances = (
            geometry.manufacturing.radial_tolerance_m,
            geometry.manufacturing.axial_tolerance_m,
        )
        handoff_ids = {region.region_id for region in contract.regions}
        geometry_pm_count = sum(
            region.role == "permanent_magnet" for region in geometry.regions
        )
        geometry_provenance = tuple(
            (
                region.region_id,
                region.shape.value,
                region.region_id in handoff_ids,
                (
                    "represented_by_magnetics_handoff"
                    if region.region_id in handoff_ids
                    else "preserved_geometry_only_nonrectangular_nonmagnetic_region"
                ),
            )
            for region in geometry.regions
        )
    source_bounds = tuple(
        region.bounds for region in contract.regions if region is not background
    )
    envelope_r_max = max(item.r_outer_m for item in source_bounds)
    envelope_z_min = min(item.z_min_m for item in source_bounds)
    envelope_z_max = max(item.z_max_m for item in source_bounds)
    characteristic = max(envelope_r_max, envelope_z_max - envelope_z_min)
    source_z_center = 0.5 * (envelope_z_min + envelope_z_max)
    robin_radial = tuple(
        1.0
        / (
            1.0
            + domain.dr_m
            * _dipole_robin_alpha_radial(
                domain.radius_m,
                domain.z_min_m + j * domain.dz_m,
                source_z_center,
            )
        )
        for j in range(nz)
    )
    robin_z_min = tuple(
        1.0
        / (
            1.0
            + domain.dz_m
            * _dipole_robin_alpha_axial(
                i * domain.dr_m,
                domain.z_min_m,
                source_z_center,
            )
        )
        for i in range(nr)
    )
    robin_z_max = tuple(
        1.0
        / (
            1.0
            + domain.dz_m
            * _dipole_robin_alpha_axial(
                i * domain.dr_m,
                domain.z_max_m,
                source_z_center,
            )
        )
        for i in range(nr)
    )
    feature_cells = (
        tuple(
            (
                region.region_id,
                min(
                    region.r_outer_start_m - region.r_inner_start_m,
                    region.r_outer_end_m - region.r_inner_end_m,
                )
                / domain.dr_m,
                (region.z_max_m - region.z_min_m) / domain.dz_m,
            )
            for region in geometry.regions
            if geometry.material_by_id(region.material_id).relative_permeability
            != 1.0
        )
        if geometry is not None
        else tuple(
            (
                region.region_id,
                (region.bounds.r_outer_m - region.bounds.r_inner_m) / domain.dr_m,
                (region.bounds.z_max_m - region.bounds.z_min_m) / domain.dz_m,
            )
            for region in contract.regions
            if region is not background
            and properties(region)[0] != properties(background)[0]
        )
    )
    qoi_locations = (
        tuple(
            (f"stage-{index + 1}-axis", 0.0, 0.5 * (stage.z_min_m + stage.z_max_m))
            for index, stage in enumerate(geometry.stages)
        )
        if geometry is not None
        else ()
    )
    qoi_bore_windows = (
        tuple(
            (
                f"stage-{index + 1}-bore-average",
                (
                    geometry.chamber.inner_radius_m
                    if geometry.chamber.inner_radius_m > 0.0
                    else geometry.chamber.outer_radius_m
                ),
                stage.z_min_m,
                stage.z_max_m,
            )
            for index, stage in enumerate(geometry.stages)
        )
        if geometry is not None
        else ()
    )
    radial_face_values: list[float] = []
    for i in range(nr - 1):
        r0 = i * domain.dr_m
        r1 = (i + 1) * domain.dr_m
        for j in range(nz):
            axial = domain.z_min_m + j * domain.dz_m
            radial_face_values.append(
                _series_face_reluctivity(
                    r0,
                    r1,
                    radial_breaks(r0, r1, axial),
                    lambda radial, z=axial: point_reluctivity(radial, z),
                    radial=True,
                )
            )
    axial_face_values: list[float] = []
    for i in range(nr):
        radial = i * domain.dr_m
        for j in range(nz - 1):
            z0 = domain.z_min_m + j * domain.dz_m
            z1 = z0 + domain.dz_m
            axial_face_values.append(
                _series_face_reluctivity(
                    z0,
                    z1,
                    axial_breaks(z0, z1, radial),
                    lambda axial, r=radial: point_reluctivity(r, axial),
                    radial=False,
                )
            )
    radial_faces = tuple(radial_face_values)
    axial_faces = tuple(axial_face_values)
    contract_pm_count = (
        sum(
            region.permanent_magnet_representation
            is PermanentMagnetRepresentation.RECOIL_REMANENCE
            for region in contract.regions
        )
        if authority == PermanentMagnetRepresentation.RECOIL_REMANENCE.value
        else len(contract.magnetization_sources)
    )
    if geometry is not None and contract_pm_count != geometry_pm_count:
        raise MaterialFieldValidationError("geometry/handoff permanent-magnet count mismatch")
    policy = contract.open_boundary_policy
    return RasterizedMaterialProblem(
        problem_id=contract.problem_id,
        domain=domain,
        geometry_sha256=geometry_hash,
        magnetics_sha256=content_sha256(contract.to_dict()),
        authority=authority,
        material_ids=tuple(material_ids),
        temperatures_k=tuple(temperatures),
        polarities=tuple(polarities),
        reluctivity_per_m_h=tuple(reluctivity),
        remanence_r_t=tuple(br_r),
        remanence_z_t=tuple(br_z),
        free_current_phi_a_per_m2=tuple(free_current),
        raster_diagnostics=tuple(volume_diagnostics + source_diagnostics),
        tolerances_m=tolerances,
        pm_bound_current_phi_a_per_m2=tuple(pm_bound_current),
        weak_action_diagnostics=tuple(weak_actions),
        geometry_region_provenance=geometry_provenance,
        pm_region_count=contract_pm_count,
        handoff_interface_count=len(contract.interfaces),
        open_boundary_policy=(
            ("minimum_padding_characteristic_lengths", policy.minimum_padding_characteristic_lengths),
            ("maximum_boundary_to_peak_field_ratio", policy.maximum_boundary_to_peak_field_ratio),
            ("domain_expansion_factor", policy.domain_expansion_factor),
            ("required_expansion_comparisons", policy.required_expansion_comparisons),
            ("maximum_qoi_relative_change", policy.maximum_qoi_relative_change),
        ),
        geometry_schema_version=(
            geometry.schema_version
            if geometry is not None
            else "cft_revival.geometry.axisymmetric_cft/1.1.0"
        ),
        source_envelope_m=(
            envelope_r_max,
            envelope_z_min,
            envelope_z_max,
            characteristic,
        ),
        feature_effective_cells=feature_cells,
        qoi_locations_rz_m=qoi_locations,
        qoi_bore_windows_m=qoi_bore_windows,
        radial_face_reluctivity_per_m_h=radial_faces,
        axial_face_reluctivity_per_m_h=axial_faces,
        remanence_g_r_face_a_per_m=tuple(remanence_g_r_faces),
        remanence_g_z_face_a_per_m=tuple(remanence_g_z_faces),
        authoritative_material_region_count=len(contract.regions),
        authoritative_free_current_source_count=0,
        outer_boundary_kind="dipole_robin_psi",
        robin_radial_q=robin_radial,
        robin_z_min_q=robin_z_min,
        robin_z_max_q=robin_z_max,
        geometry_bundle_json=(
            json.dumps(
                geometry.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            )
            if geometry is not None
            else ""
        ),
        magnetics_bundle_json=serialize_handoff(contract),
    )
