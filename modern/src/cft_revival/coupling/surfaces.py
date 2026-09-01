"""Dependency-free marching-squares tracing on axisymmetric ψ maps."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import hypot, isfinite, ulp

from .models import CouplingValidationError, TopologyResolutionError
from .profiles import stable_lerp
from .v3_models import (
    BoundaryNullDiagnostic,
    ContourFieldCertificate,
    FluxContour,
    FluxSurfacePolicy,
    ValidatedPsiMap,
)

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True, slots=True)
class _GraphChain:
    points: tuple[Point, ...]
    simple: bool
    reason: str
    unique_vertex_count: int
    edge_count: int


def _lerp_crossing(
    first: Point,
    second: Point,
    first_value: float,
    second_value: float,
    level: float,
) -> Point:
    scale = max(abs(first_value), abs(second_value), abs(level))
    if scale == 0.0:
        return (
            (first[0] + second[0]) * 0.5,
            (first[1] + second[1]) * 0.5,
        )
    normalized_first = first_value / scale
    normalized_second = second_value / scale
    normalized_level = level / scale
    denominator = normalized_second - normalized_first
    if denominator == 0.0:
        fraction = 0.5
    else:
        fraction = (normalized_level - normalized_first) / denominator
    fraction = min(1.0, max(0.0, fraction))
    return (
        first[0] + fraction * (second[0] - first[0]),
        first[1] + fraction * (second[1] - first[1]),
    )


def _same_point(first: Point, second: Point, tolerance_m: float) -> bool:
    return hypot(first[0] - second[0], first[1] - second[1]) <= tolerance_m


def _cell_segments(
    corners: tuple[Point, Point, Point, Point],
    values: tuple[float, float, float, float],
    level: float,
    psi_tolerance: float,
    connectivity_tolerance_m: float,
    saddle_tie_policy: str,
) -> tuple[Segment, ...]:
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    crossings: list[tuple[int, Point]] = []
    for edge_index, (left, right) in enumerate(edges):
        a = values[left] - level
        b = values[right] - level
        if abs(a) <= psi_tolerance and abs(b) <= psi_tolerance:
            continue
        if (a <= 0.0 <= b) or (b <= 0.0 <= a):
            point = _lerp_crossing(
                corners[left], corners[right], values[left], values[right], level
            )
            if not any(
                _same_point(point, existing, connectivity_tolerance_m)
                for _, existing in crossings
            ):
                crossings.append((edge_index, point))
    if len(crossings) == 2:
        return ((crossings[0][1], crossings[1][1]),)
    if len(crossings) != 4:
        return ()
    by_edge = {edge: point for edge, point in crossings}
    scale = max(abs(value) for value in values + (level,))
    normalized = tuple(
        (value / scale) - (level / scale) for value in values
    )
    determinant = normalized[0] * normalized[2] - normalized[1] * normalized[3]
    determinant_tolerance = 16.0 * 2.220446049250313e-16
    if abs(determinant) <= determinant_tolerance:
        if saddle_tie_policy == "reject":
            raise TopologyResolutionError(
                "exact marching-squares saddle requires an explicit tie policy"
            )
        if saddle_tie_policy == "pair_01_23":
            pairs = ((0, 1), (2, 3))
        elif saddle_tie_policy == "pair_03_12":
            pairs = ((0, 3), (1, 2))
        else:
            raise CouplingValidationError("invalid saddle_tie_policy")
    elif determinant > 0.0:
        pairs = ((0, 1), (2, 3))
    else:
        pairs = ((0, 3), (1, 2))
    return tuple((by_edge[first], by_edge[second]) for first, second in pairs)


def _orientation(first: Point, second: Point, third: Point) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_cross(
    first: Segment, second: Segment, tolerance_m: float
) -> bool:
    a, b = first
    c, d = second
    scale = max(
        hypot(b[0] - a[0], b[1] - a[1]),
        hypot(d[0] - c[0], d[1] - c[1]),
        tolerance_m,
    )
    tolerance = tolerance_m * scale
    values = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    first_straddles = (values[0] < -tolerance and values[1] > tolerance) or (
        values[1] < -tolerance and values[0] > tolerance
    )
    second_straddles = (values[2] < -tolerance and values[3] > tolerance) or (
        values[3] < -tolerance and values[2] > tolerance
    )
    if first_straddles and second_straddles:
        return True

    def on_segment(point: Point, segment: Segment) -> bool:
        left, right = segment
        return (
            abs(_orientation(left, right, point)) <= tolerance
            and min(left[0], right[0]) - tolerance_m
            <= point[0]
            <= max(left[0], right[0]) + tolerance_m
            and min(left[1], right[1]) - tolerance_m
            <= point[1]
            <= max(left[1], right[1]) + tolerance_m
        )

    return (
        on_segment(a, second)
        or on_segment(b, second)
        or on_segment(c, first)
        or on_segment(d, first)
    )


def _join_segments(
    segments: tuple[Segment, ...], tolerance_m: float
) -> tuple[_GraphChain, ...]:
    """Build deterministic edge graphs and retain malformed components."""

    vertices: list[Point] = []

    def vertex_id(point: Point) -> int:
        for index, existing in enumerate(vertices):
            if _same_point(point, existing, tolerance_m):
                return index
        vertices.append(point)
        return len(vertices) - 1

    edges: list[tuple[int, int]] = []
    duplicate_edges: set[tuple[int, int]] = set()
    for left, right in segments:
        first, second = vertex_id(left), vertex_id(right)
        key = (min(first, second), max(first, second))
        if first == second or key in {
            (min(a, b), max(a, b)) for a, b in edges
        }:
            duplicate_edges.add(key)
        edges.append((first, second))
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(vertices))}
    for edge_index, (first, second) in enumerate(edges):
        adjacency[first].append(edge_index)
        adjacency[second].append(edge_index)
    unused = set(range(len(edges)))
    result: list[_GraphChain] = []
    while unused:
        seed = min(unused)
        component_edges: set[int] = set()
        pending = [seed]
        while pending:
            edge_index = pending.pop()
            if edge_index in component_edges:
                continue
            component_edges.add(edge_index)
            for vertex in edges[edge_index]:
                pending.extend(
                    candidate
                    for candidate in adjacency[vertex]
                    if candidate not in component_edges
                )
        unused.difference_update(component_edges)
        component_vertices = {
            vertex for edge_index in component_edges for vertex in edges[edge_index]
        }
        degrees = {
            vertex: sum(
                edge_index in component_edges for edge_index in adjacency[vertex]
            )
            for vertex in component_vertices
        }
        endpoints = sorted(vertex for vertex, degree in degrees.items() if degree == 1)
        simple = True
        reason = "simple edge graph"
        if any(
            (min(edges[index]), max(edges[index])) in duplicate_edges
            for index in component_edges
        ):
            simple = False
            reason = "duplicate or retraced contour edge"
        elif any(degree > 2 or degree == 0 for degree in degrees.values()):
            simple = False
            reason = "branched or repeated contour vertex"
        elif len(endpoints) not in (0, 2):
            simple = False
            reason = "contour graph has invalid endpoint count"
        start = endpoints[0] if endpoints else min(component_vertices)
        chain_ids = [start]
        used_component: set[int] = set()
        current = start
        previous_edge: int | None = None
        while True:
            choices = sorted(
                edge_index
                for edge_index in adjacency[current]
                if edge_index in component_edges and edge_index not in used_component
            )
            if not choices:
                break
            edge_index = choices[0]
            if previous_edge is not None and len(choices) > 1:
                simple = False
                reason = "ambiguous traversal at repeated vertex"
            used_component.add(edge_index)
            first, second = edges[edge_index]
            current = second if first == current else first
            chain_ids.append(current)
            previous_edge = edge_index
            if current == start and len(used_component) == len(component_edges):
                break
        if len(used_component) != len(component_edges):
            simple = False
            reason = "not every component edge has one traversal"
        nonterminal = chain_ids[:-1] if chain_ids[-1] == chain_ids[0] else chain_ids
        if len(nonterminal) != len(set(nonterminal)):
            simple = False
            reason = "repeated nonterminal contour vertex"
        points = tuple(vertices[index] for index in chain_ids)
        chain_edges = tuple(zip(points, points[1:]))
        for first_index, first_edge in enumerate(chain_edges):
            for second_index in range(first_index + 1, len(chain_edges)):
                adjacent = second_index == first_index + 1 or (
                    first_index == 0
                    and second_index == len(chain_edges) - 1
                    and chain_ids[0] == chain_ids[-1]
                )
                if not adjacent and _segments_cross(
                    first_edge, chain_edges[second_index], tolerance_m
                ):
                    simple = False
                    reason = "self-intersecting contour"
        result.append(
            _GraphChain(
                points,
                simple,
                reason,
                len(component_vertices),
                len(component_edges),
            )
        )
    return tuple(result)


def validate_simple_contour(
    points: tuple[Point, ...],
    *,
    tolerance_m: float,
    domain_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[bool, str, int, int]:
    """Revalidate canonical simple-cycle invariants from retained contour points."""

    if len(points) < 4 or not _same_point(points[0], points[-1], tolerance_m):
        return False, "contour is not a closed cycle", len(set(points)), max(0, len(points) - 1)
    canonical: list[Point] = []
    ids: list[int] = []
    for point in points[:-1]:
        match = next(
            (
                index
                for index, existing in enumerate(canonical)
                if _same_point(point, existing, tolerance_m)
            ),
            None,
        )
        if match is None:
            canonical.append(point)
            match = len(canonical) - 1
        ids.append(match)
    ids.append(ids[0])
    if len(ids[:-1]) != len(set(ids[:-1])):
        return False, "repeated nonterminal contour vertex", len(canonical), len(ids) - 1
    edges = tuple(zip(ids, ids[1:]))
    undirected = tuple((min(a, b), max(a, b)) for a, b in edges)
    if any(a == b for a, b in edges) or len(undirected) != len(set(undirected)):
        return False, "duplicate or retraced contour edge", len(canonical), len(edges)
    geometric_edges = tuple(zip(points, points[1:]))
    for first_index, first in enumerate(geometric_edges):
        for second_index in range(first_index + 1, len(geometric_edges)):
            adjacent = second_index == first_index + 1 or (
                first_index == 0 and second_index == len(geometric_edges) - 1
            )
            if not adjacent and _segments_cross(
                first, geometric_edges[second_index], tolerance_m
            ):
                return False, "self-intersecting contour", len(canonical), len(edges)
    if domain_bounds is not None:
        r_min, r_max, z_min, z_max = domain_bounds
        if any(
            r <= r_min + tolerance_m
            or r >= r_max - tolerance_m
            or z <= z_min + tolerance_m
            or z >= z_max - tolerance_m
            for r, z in points
        ):
            return False, "contour contacts finite-domain boundary", len(canonical), len(edges)
    return True, "simple closed contour", len(canonical), len(edges)


def bilinear_sample(
    field: ValidatedPsiMap,
    values: tuple[tuple[float, ...], ...],
    point: Point,
) -> float:
    r, z = point
    if r < field.r_m[0] or r > field.r_m[-1] or z < field.z_m[0] or z > field.z_m[-1]:
        raise TopologyResolutionError("surface sample lies outside the verified map")
    i = min(len(field.r_m) - 2, max(0, bisect_right(field.r_m, r) - 1))
    j = min(len(field.z_m) - 2, max(0, bisect_right(field.z_m, z) - 1))
    r0, r1 = field.r_m[i], field.r_m[i + 1]
    z0, z1 = field.z_m[j], field.z_m[j + 1]
    fr = (r - r0) / (r1 - r0)
    fz = (z - z0) / (z1 - z0)
    low = stable_lerp(values[i][j], values[i + 1][j], fr)
    high = stable_lerp(values[i][j + 1], values[i + 1][j + 1], fr)
    result = stable_lerp(low, high, fz)
    if not isfinite(result):
        raise TopologyResolutionError("bilinear interpolation produced a nonfinite value")
    return result


def _segment_component_quadratic(
    field: ValidatedPsiMap,
    values: tuple[tuple[float, ...], ...],
    first: Point,
    second: Point,
) -> tuple[float, float, float]:
    midpoint = (
        stable_lerp(first[0], second[0], 0.5),
        stable_lerp(first[1], second[1], 0.5),
    )
    samples = (
        bilinear_sample(field, values, first),
        bilinear_sample(field, values, midpoint),
        bilinear_sample(field, values, second),
    )
    scale = max(abs(value) for value in samples)
    if scale == 0.0:
        return 0.0, 0.0, 0.0
    y0, ym, y1 = (value / scale for value in samples)
    c2 = 2.0 * (y0 + y1 - 2.0 * ym)
    c1 = y1 - y0 - c2
    return scale * y0, scale * c1, scale * c2


def _quadratic_value(coefficients: tuple[float, float, float], value: float) -> float:
    c0, c1, c2 = coefficients
    result = c0 + value * (c1 + value * c2)
    if not isfinite(result):
        raise TopologyResolutionError("segment field interpolation overflowed")
    return result


def certify_contour_field(
    field: ValidatedPsiMap,
    contour: FluxContour,
    *,
    null_floor_t: float,
    absolute_tolerance_t: float,
    relative_tolerance: float,
    maximum_depth: int,
) -> ContourFieldCertificate:
    """Certify field bounds on every contour edge using adaptive Lipschitz bounds."""

    if (
        not isfinite(null_floor_t)
        or null_floor_t < 0.0
        or not isfinite(absolute_tolerance_t)
        or absolute_tolerance_t < 0.0
        or not isfinite(relative_tolerance)
        or relative_tolerance < 0.0
        or maximum_depth < 1
    ):
        raise CouplingValidationError("invalid contour field certification policy")
    samples: dict[Point, float] = {}
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    maximum_gradient = 0.0
    subdivisions = 0
    deepest = 0
    regular = True
    reason = "all contour segments have certified positive field bounds"

    def sample(point: Point) -> float:
        if point not in samples:
            samples[point] = hypot(
                bilinear_sample(field, field.b_r_t, point),
                bilinear_sample(field, field.b_z_t, point),
            )
        return samples[point]

    for first, second in zip(contour.points_rz_m, contour.points_rz_m[1:]):
        length = hypot(second[0] - first[0], second[1] - first[1])
        if not isfinite(length) or length <= 0.0:
            regular = False
            reason = "contour contains a zero/nonfinite segment"
            continue
        try:
            br_coefficients = _segment_component_quadratic(
                field, field.b_r_t, first, second
            )
            bz_coefficients = _segment_component_quadratic(
                field, field.b_z_t, first, second
            )
        except TopologyResolutionError:
            regular = False
            reason = "segment interpolation is not representable"
            continue

        def recurse(left: float, right: float, depth: int) -> None:
            nonlocal deepest, maximum_gradient, regular, reason, subdivisions
            deepest = max(deepest, depth)
            midpoint_fraction = stable_lerp(left, right, 0.5)
            midpoint = (
                stable_lerp(first[0], second[0], midpoint_fraction),
                stable_lerp(first[1], second[1], midpoint_fraction),
            )
            midpoint_magnitude = sample(midpoint)
            derivative_values: list[float] = []
            for coefficients in (br_coefficients, bz_coefficients):
                _, c1, c2 = coefficients
                derivative_values.append(
                    max(abs(c1 + 2.0 * c2 * left), abs(c1 + 2.0 * c2 * right))
                )
            derivative_per_t = hypot(*derivative_values)
            if not isfinite(derivative_per_t):
                regular = False
                reason = "segment derivative bound overflowed"
                return
            maximum_gradient = max(maximum_gradient, derivative_per_t / length)
            radius = (right - left) * 0.5
            excursion = derivative_per_t * radius
            if not isfinite(excursion):
                regular = False
                reason = "segment field bound overflowed"
                return
            rounding_margin = 16.0 * ulp(max(midpoint_magnitude, excursion))
            lower = max(0.0, midpoint_magnitude - excursion - rounding_margin)
            upper = midpoint_magnitude + excursion + rounding_margin
            if not isfinite(upper):
                regular = False
                reason = "segment field upper bound is nonfinite"
                return
            tolerance = absolute_tolerance_t + relative_tolerance * max(
                midpoint_magnitude, upper
            )
            certified = lower > null_floor_t
            resolved = upper - lower <= tolerance
            if certified and resolved:
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                subdivisions += 1
                return
            if depth >= maximum_depth:
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                subdivisions += 1
                if not certified:
                    regular = False
                    reason = (
                        "segment may contain an exact/near magnetic null within "
                        "the certified refinement limit"
                    )
                else:
                    regular = False
                    reason = "segment extrema bounds did not converge"
                return
            recurse(left, midpoint_fraction, depth + 1)
            recurse(midpoint_fraction, right, depth + 1)

        sample(first)
        sample(second)
        recurse(0.0, 1.0, 0)
    ordered = tuple(sorted(samples.items(), key=lambda item: item[0]))
    sampled_values = tuple(value for _, value in ordered)
    if not sampled_values or not lower_bounds or not upper_bounds:
        return ContourFieldCertificate(
            0.0,
            0.0,
            0.0,
            0.0,
            maximum_gradient,
            tuple(point for point, _ in ordered),
            sampled_values,
            subdivisions,
            deepest,
            False,
            "contour has no certifiable field segments",
        )
    return ContourFieldCertificate(
        min(lower_bounds),
        min(sampled_values),
        max(sampled_values),
        max(upper_bounds),
        maximum_gradient,
        tuple(point for point, _ in ordered),
        sampled_values,
        subdivisions,
        deepest,
        regular,
        reason,
    )


def trace_flux_contours(
    field: ValidatedPsiMap,
    psi_wb: float,
    policy: FluxSurfacePolicy = FluxSurfacePolicy(),
) -> tuple[FluxContour, ...]:
    """Trace every connected component of ψ=constant using marching squares."""

    level = float(psi_wb)
    if not isfinite(level):
        raise CouplingValidationError("psi_wb must be finite")
    if policy.connectivity_tolerance_m <= 0.0:
        raise CouplingValidationError("connectivity_tolerance_m must be positive")
    psi_scale = max(abs(value) for row in field.psi_wb for value in row)
    psi_tolerance = max(
        policy.psi_absolute_tolerance_wb,
        policy.psi_relative_tolerance * max(abs(level), psi_scale),
    )
    if not isfinite(psi_tolerance) or psi_tolerance < 0.0:
        raise CouplingValidationError("psi contour tolerance is not representable")
    segments: list[Segment] = []
    for i in range(len(field.r_m) - 1):
        for j in range(len(field.z_m) - 1):
            corners = (
                (field.r_m[i], field.z_m[j]),
                (field.r_m[i + 1], field.z_m[j]),
                (field.r_m[i + 1], field.z_m[j + 1]),
                (field.r_m[i], field.z_m[j + 1]),
            )
            values = (
                field.psi_wb[i][j],
                field.psi_wb[i + 1][j],
                field.psi_wb[i + 1][j + 1],
                field.psi_wb[i][j + 1],
            )
            segments.extend(
                _cell_segments(
                    corners,
                    values,
                    level,
                    psi_tolerance,
                    policy.connectivity_tolerance_m,
                    policy.saddle_tie_policy,
                )
            )
    contours: list[FluxContour] = []
    r0, r1 = field.r_m[0], field.r_m[-1]
    z0, z1 = field.z_m[0], field.z_m[-1]
    for chain in _join_segments(tuple(segments), policy.connectivity_tolerance_m):
        if len(chain.points) < 2:
            continue
        gap = hypot(
            chain.points[0][0] - chain.points[-1][0],
            chain.points[0][1] - chain.points[-1][1],
        )
        closed = gap <= policy.connectivity_tolerance_m
        points = chain.points
        if closed and chain.points[-1] != chain.points[0]:
            points = chain.points + (chain.points[0],)
        touches_boundary = any(
            abs(r - r0) <= policy.connectivity_tolerance_m
            or abs(r - r1) <= policy.connectivity_tolerance_m
            or abs(z - z0) <= policy.connectivity_tolerance_m
            or abs(z - z1) <= policy.connectivity_tolerance_m
            for r, z in points
        )
        residual = max(
            abs(bilinear_sample(field, field.psi_wb, point) - level)
            for point in points
        )
        contours.append(
            FluxContour(
                level,
                points,
                closed,
                touches_boundary,
                residual,
                gap,
                chain.simple,
                chain.reason,
                chain.unique_vertex_count,
                chain.edge_count,
            )
        )
    return tuple(
        sorted(
            contours,
            key=lambda contour: (
                sum(point[1] for point in contour.points_rz_m)
                / len(contour.points_rz_m),
                len(contour.points_rz_m),
            ),
        )
    )


def require_same_flux_surface(
    field: ValidatedPsiMap,
    points_rz_m: tuple[Point, ...],
    policy: FluxSurfacePolicy = FluxSurfacePolicy(),
) -> float:
    """Validate that arbitrary samples share one ψ label within declared tolerance."""

    if len(points_rz_m) < 2:
        raise CouplingValidationError("at least two points are required")
    values = tuple(bilinear_sample(field, field.psi_wb, point) for point in points_rz_m)
    scale = max(abs(value) for value in values)
    tolerance = max(
        policy.psi_absolute_tolerance_wb,
        policy.psi_relative_tolerance * scale,
    )
    if max(values) - min(values) > tolerance:
        raise TopologyResolutionError(
            "samples lie on different psi surfaces; same-z is not a field-line binding"
        )
    return sum(values) / len(values)


def _opposite_signed_root_fraction(left: float, right: float) -> float:
    """Return the linear root fraction without forming an overflowing sum."""

    left_magnitude = abs(left)
    right_magnitude = abs(right)
    if left_magnitude == 0.0:
        return 0.0
    if right_magnitude == 0.0:
        return 1.0
    if left_magnitude <= right_magnitude:
        ratio = left_magnitude / right_magnitude
        return ratio / (1.0 + ratio)
    ratio = right_magnitude / left_magnitude
    return 1.0 / (1.0 + ratio)


def magnetic_null_geometry(
    field: ValidatedPsiMap,
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance_t: float = 1.0e-15,
    boundary_exclusion_cells: int = 1,
) -> tuple[tuple[tuple[float, float], ...], tuple[BoundaryNullDiagnostic, ...]]:
    """Return interior null geometry and separately typed finite-box boundary zeros."""

    scale = max(
        hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    tolerance = max(absolute_tolerance_t, relative_tolerance * scale)
    interior: list[Point] = []
    boundary: list[BoundaryNullDiagnostic] = []
    nr, nz = len(field.r_m), len(field.z_m)
    for i in range(nr):
        for j in range(nz):
            magnitude = hypot(field.b_r_t[i][j], field.b_z_t[i][j])
            # r=0 is a physical coordinate singularity, not a truncated-box edge.
            on_boundary = i == nr - 1 or j in (0, nz - 1)
            if magnitude <= tolerance and on_boundary:
                labels = []
                if i == nr - 1:
                    labels.append("radial_max")
                if j == 0:
                    labels.append("z_min")
                if j == nz - 1:
                    labels.append("z_max")
                boundary.append(
                    BoundaryNullDiagnostic(field.z_m[j], "+".join(labels), magnitude)
                )
            elif (
                magnitude <= tolerance
                and (i == 0 or boundary_exclusion_cells <= i)
                and i < nr - boundary_exclusion_cells
                and boundary_exclusion_cells <= j < nz - boundary_exclusion_cells
            ):
                interior.append((field.r_m[i], field.z_m[j]))
    # Axis sign-changing roots are interior geometry, but never endpoint cells.
    axis_bz = field.b_z_t[0]
    for j, (left, right) in enumerate(zip(axis_bz, axis_bz[1:])):
        if j < boundary_exclusion_cells or j + 1 >= nz - boundary_exclusion_cells:
            continue
        if (left < 0.0 < right) or (right < 0.0 < left):
            fraction = _opposite_signed_root_fraction(left, right)
            point = (
                field.r_m[0],
                stable_lerp(field.z_m[j], field.z_m[j + 1], fraction),
            )
            if not any(
                _same_point(point, existing, 1.0e-12) for existing in interior
            ):
                interior.append(point)
    # Locate off-grid vector nulls by Newton iteration on each bilinear cell.
    # Candidate cells must bracket zero independently in both components.
    for i in range(0, nr - boundary_exclusion_cells - 1):
        for j in range(
            boundary_exclusion_cells, nz - boundary_exclusion_cells - 1
        ):
            br_values = (
                field.b_r_t[i][j],
                field.b_r_t[i + 1][j],
                field.b_r_t[i + 1][j + 1],
                field.b_r_t[i][j + 1],
            )
            bz_values = (
                field.b_z_t[i][j],
                field.b_z_t[i + 1][j],
                field.b_z_t[i + 1][j + 1],
                field.b_z_t[i][j + 1],
            )
            if not (
                min(br_values) <= 0.0 <= max(br_values)
                and min(bz_values) <= 0.0 <= max(bz_values)
            ):
                continue
            component_scale = max(
                *(abs(value) for value in br_values + bz_values), 1.0e-300
            )
            brn = tuple(value / component_scale for value in br_values)
            bzn = tuple(value / component_scale for value in bz_values)
            u = v = 0.5
            converged = False
            for _ in range(16):
                def evaluate(
                    values: tuple[float, float, float, float]
                ) -> tuple[float, float, float]:
                    f00, f10, f11, f01 = values
                    value = (
                        (1.0 - u) * (1.0 - v) * f00
                        + u * (1.0 - v) * f10
                        + u * v * f11
                        + (1.0 - u) * v * f01
                    )
                    du = (1.0 - v) * (f10 - f00) + v * (f11 - f01)
                    dv = (1.0 - u) * (f01 - f00) + u * (f11 - f10)
                    return value, du, dv

                first, first_u, first_v = evaluate(brn)
                second, second_u, second_v = evaluate(bzn)
                if hypot(first, second) <= tolerance / component_scale:
                    converged = True
                    break
                determinant = first_u * second_v - first_v * second_u
                if abs(determinant) <= 1.0e-18:
                    break
                delta_u = (-first * second_v + first_v * second) / determinant
                delta_v = (-first_u * second + first * second_u) / determinant
                u += delta_u
                v += delta_v
                if not (-1.0e-12 <= u <= 1.0 + 1.0e-12 and -1.0e-12 <= v <= 1.0 + 1.0e-12):
                    break
            if converged:
                point = (
                    stable_lerp(field.r_m[i], field.r_m[i + 1], min(1.0, max(0.0, u))),
                    stable_lerp(field.z_m[j], field.z_m[j + 1], min(1.0, max(0.0, v))),
                )
                if not any(
                    _same_point(point, existing, 1.0e-9)
                    for existing in interior
                ):
                    interior.append(point)
    interior.sort(key=lambda point: (point[1], point[0]))
    return tuple(interior), tuple(boundary)
