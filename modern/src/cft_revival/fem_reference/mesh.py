"""Deterministic conforming triangular meshes with body-fitted linear interfaces."""

from __future__ import annotations

from collections import defaultdict
from math import acos, ceil, degrees, hypot, sqrt

import numpy as np

from .models import Domain, FEMValidationError, P2Mesh

Polygon = tuple[tuple[float, float], ...]


def _unique_sorted(values: list[float], scale: float) -> list[float]:
    tolerance = max(1.0e-14 * max(scale, 1.0), 1.0e-15)
    output: list[float] = []
    for value in sorted(values):
        if not output or abs(value - output[-1]) > tolerance:
            output.append(value)
    return output


def _side_value(polygon: Polygon, z_m: float, *, outer: bool) -> float:
    z_min, z_max = polygon[0][1], polygon[2][1]
    fraction = (z_m - z_min) / (z_max - z_min)
    start = polygon[1 if outer else 0][0]
    end = polygon[2 if outer else 3][0]
    return start + fraction * (end - start)


def _point_on_segment(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> bool:
    direction = second - first
    offset = point - first
    cross = direction[0] * offset[1] - direction[1] * offset[0]
    scale = max(float(np.linalg.norm(direction)), 1.0)
    if abs(float(cross)) > 2.0e-12 * scale:
        return False
    projection = float(np.dot(offset, direction))
    return -1.0e-14 <= projection <= float(np.dot(direction, direction)) + 1.0e-14


def _point_in_polygon(point: np.ndarray, polygon: Polygon) -> bool:
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        if _point_on_segment(point, np.asarray(first), np.asarray(second)):
            return True
    r_m, z_m = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (previous[1] > z_m) != (current[1] > z_m):
            crossing = previous[0] + (z_m - previous[1]) * (
                current[0] - previous[0]
            ) / (current[1] - previous[1])
            if r_m < crossing:
                inside = not inside
        previous = current
    return inside


def _triangulate_strip(
    bottom: list[int],
    top: list[int],
    vertices: list[tuple[float, float]],
) -> list[tuple[int, int, int]]:
    """Triangulate between two monotone chains while preserving both side edges."""

    if not bottom or not top or (len(bottom) == 1 and len(top) == 1):
        raise FEMValidationError("body-fitted strip has insufficient boundary nodes")
    if len(bottom) == 1:
        return [
            (bottom[0], top[index + 1], top[index])
            for index in range(len(top) - 1)
        ]
    if len(top) == 1:
        return [
            (bottom[index], bottom[index + 1], top[0])
            for index in range(len(bottom) - 1)
        ]
    result: list[tuple[int, int, int]] = []
    i = j = 0
    bottom_left = vertices[bottom[0]][0]
    bottom_width = vertices[bottom[-1]][0] - bottom_left
    top_left = vertices[top[0]][0]
    top_width = vertices[top[-1]][0] - top_left
    if bottom_width <= 0.0 or top_width <= 0.0:
        raise FEMValidationError("body-fitted strip has non-positive width")
    while i < len(bottom) - 1 or j < len(top) - 1:
        bottom_progress = (
            (vertices[bottom[i + 1]][0] - bottom_left) / bottom_width
            if i < len(bottom) - 1
            else float("inf")
        )
        top_progress = (
            (vertices[top[j + 1]][0] - top_left) / top_width
            if j < len(top) - 1
            else float("inf")
        )
        if bottom_progress <= top_progress:
            result.append((bottom[i], bottom[i + 1], top[j]))
            i += 1
        else:
            result.append((bottom[i], top[j + 1], top[j]))
            j += 1
    return result


def build_body_fitted_mesh(
    domain: Domain,
    polygons: tuple[tuple[str, Polygon], ...],
    region_at,
    *,
    radial_divisions: int,
    axial_divisions: int,
    radial_coordinates: tuple[float, ...] | None = None,
    axial_coordinates: tuple[float, ...] | None = None,
    target_size_m: float | None = None,
    size_field=None,
    constraint_tracks: tuple[tuple[float, float, float, float], ...] = (),
    protected_radii_m: tuple[float, ...] = (),
    protected_z_m: tuple[float, ...] = (),
    reject_below_angle_deg: float = 0.0,
) -> P2Mesh:
    """Mesh a rectangle while retaining every polygon side as an element edge."""

    if (
        isinstance(radial_divisions, bool)
        or not isinstance(radial_divisions, int)
        or isinstance(axial_divisions, bool)
        or not isinstance(axial_divisions, int)
        or radial_divisions < 2
        or axial_divisions < 2
    ):
        raise FEMValidationError("mesh divisions must both be at least two")
    if radial_coordinates is None and axial_coordinates is None:
        from .resource_policy import guard_allocation

        guard_allocation(
            "initial_mesh_request",
            p2_dofs=(2 * radial_divisions + 1) * (2 * axial_divisions + 1),
            triangles=2 * radial_divisions * axial_divisions,
            robin_edges=2 * radial_divisions + axial_divisions,
        )
    if target_size_m is not None and (
        not np.isfinite(target_size_m) or target_size_m <= 0.0
    ):
        raise FEMValidationError("target mesh size must be finite and positive")
    r_uniform = np.asarray(
        radial_coordinates
        if radial_coordinates is not None
        else (
            ()
            if target_size_m is not None or size_field is not None
            else np.linspace(domain.r_min_m, domain.r_max_m, radial_divisions + 1)
        ),
        dtype=np.float64,
    )
    z_values = list(
        axial_coordinates
        if axial_coordinates is not None
        else np.linspace(domain.z_min_m, domain.z_max_m, axial_divisions + 1)
    )
    for _, polygon in polygons:
        z_values.extend((polygon[0][1], polygon[2][1]))
    for _, _, z_min, z_max in constraint_tracks:
        z_values.extend((z_min, z_max))
    z_levels = _unique_sorted(
        [value for value in z_values if domain.z_min_m <= value <= domain.z_max_m],
        domain.z_max_m - domain.z_min_m,
    )

    radial_levels: list[list[float]] = []
    for z_m in z_levels:
        values = [domain.r_min_m, domain.r_max_m, *r_uniform]
        for _, polygon in polygons:
            if polygon[0][1] - 1.0e-15 <= z_m <= polygon[2][1] + 1.0e-15:
                values.extend(
                    (_side_value(polygon, z_m, outer=False), _side_value(polygon, z_m, outer=True))
                )
        for r_start, r_end, z_min, z_max in constraint_tracks:
            if z_min - 1.0e-15 <= z_m <= z_max + 1.0e-15:
                fraction = (z_m - z_min) / (z_max - z_min)
                values.append(r_start + fraction * (r_end - r_start))
        boundaries = _unique_sorted(
            [
                value
                for value in values
                if domain.r_min_m <= value <= domain.r_max_m
            ],
            domain.r_max_m - domain.r_min_m,
        )
        if target_size_m is not None or size_field is not None:
            subdivided = [boundaries[0]]
            for left, right in zip(boundaries, boundaries[1:]):
                target = target_size_m
                if size_field is not None:
                    local_target = float(size_field(0.5 * (left + right), z_m))
                    if not np.isfinite(local_target) or local_target <= 0.0:
                        raise FEMValidationError(
                            "mesh size field must return finite positive values"
                        )
                    target = (
                        local_target
                        if target is None
                        else min(target, local_target)
                    )
                assert target is not None
                count = max(1, ceil((right - left) / target))
                subdivided.extend(
                    left + (right - left) * index / count
                    for index in range(1, count + 1)
                )
            boundaries = subdivided
        radial_levels.append(boundaries)

    projected_vertices = sum(len(level) for level in radial_levels)
    projected_triangles = sum(
        len(left) + len(right) - 2
        for left, right in zip(radial_levels, radial_levels[1:])
    )
    projected_boundary_edges = (
        2 * (len(z_levels) - 1)
        + len(radial_levels[0])
        + len(radial_levels[-1])
        - 2
    )
    projected_edges = (
        3 * projected_triangles + projected_boundary_edges
    ) // 2
    from .resource_policy import guard_allocation

    guard_allocation(
        "initial_mesh_construction",
        p2_dofs=projected_vertices + projected_edges,
        triangles=projected_triangles,
        robin_edges=(
            len(radial_levels[-1]) - 1
            + 2 * (len(z_levels) - 1)
        ),
    )

    vertices: list[tuple[float, float]] = []
    level_indices: list[list[int]] = []
    for z_m, radial_values in zip(z_levels, radial_levels):
        indices = []
        for r_m in radial_values:
            indices.append(len(vertices))
            vertices.append((r_m, z_m))
        level_indices.append(indices)

    triangles: list[tuple[int, int, int]] = []
    for level in range(len(z_levels) - 1):
        z0, z1 = z_levels[level], z_levels[level + 1]
        midpoint = 0.5 * (z0 + z1)
        tracks = [(domain.r_min_m, domain.r_min_m), (domain.r_max_m, domain.r_max_m)]
        for _, polygon in polygons:
            if polygon[0][1] < midpoint < polygon[2][1]:
                tracks.extend(
                    (
                        (
                            _side_value(polygon, z0, outer=False),
                            _side_value(polygon, z1, outer=False),
                        ),
                        (
                            _side_value(polygon, z0, outer=True),
                            _side_value(polygon, z1, outer=True),
                        ),
                    )
                )
        for r_start, r_end, z_min, z_max in constraint_tracks:
            if z_min < midpoint < z_max:
                fraction_0 = (z0 - z_min) / (z_max - z_min)
                fraction_1 = (z1 - z_min) / (z_max - z_min)
                tracks.append(
                    (
                        r_start + fraction_0 * (r_end - r_start),
                        r_start + fraction_1 * (r_end - r_start),
                    )
                )
        tracks = sorted(tracks, key=lambda item: (0.5 * (item[0] + item[1]), item))
        filtered: list[tuple[float, float]] = []
        for track in tracks:
            if not filtered or max(
                abs(track[0] - filtered[-1][0]), abs(track[1] - filtered[-1][1])
            ) > 1.0e-14:
                filtered.append(track)
        for left, right in zip(filtered, filtered[1:]):
            if (
                right[0] < left[0]
                or right[1] < left[1]
                or (right[0] == left[0] and right[1] == left[1])
            ):
                raise FEMValidationError("crossing or collapsed body-fitted tracks")
            bottom = [
                index
                for r_m, index in zip(radial_levels[level], level_indices[level])
                if left[0] - 1.0e-14 <= r_m <= right[0] + 1.0e-14
            ]
            top = [
                index
                for r_m, index in zip(radial_levels[level + 1], level_indices[level + 1])
                if left[1] - 1.0e-14 <= r_m <= right[1] + 1.0e-14
            ]
            triangles.extend(_triangulate_strip(bottom, top, vertices))

    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    first_vectors = (
        vertex_array[triangle_array[:, 1]] - vertex_array[triangle_array[:, 0]]
    )
    second_vectors = (
        vertex_array[triangle_array[:, 2]] - vertex_array[triangle_array[:, 0]]
    )
    areas = (
        first_vectors[:, 0] * second_vectors[:, 1]
        - first_vectors[:, 1] * second_vectors[:, 0]
    )
    if np.any(areas <= 1.0e-24):
        raise FEMValidationError("mesh contains inverted or degenerate triangles")
    tags = tuple(
        region_at(*np.mean(vertex_array[triangle], axis=0)) for triangle in triangle_array
    )

    mesh = _construct_p2_mesh(
        vertex_array,
        triangle_array,
        tags,
        domain,
        protected_radii_m=protected_radii_m,
        protected_z_m=protected_z_m,
    )
    validate_region_conformity(mesh, polygons)
    quality = mesh_quality(mesh)
    if quality["minimum_angle_deg"] < reject_below_angle_deg:
        raise FEMValidationError(
            f"mesh minimum angle {quality['minimum_angle_deg']:.6g} deg is below "
            f"rejection threshold {reject_below_angle_deg:.6g} deg"
        )
    return mesh


def _construct_p2_mesh(
    vertex_array: np.ndarray,
    triangle_array: np.ndarray,
    tags: tuple[str, ...],
    domain: Domain,
    *,
    element_parent_ids: np.ndarray | None = None,
    refinement_level: int = 0,
    parent_mesh_sha256: str = "0" * 64,
    protected_radii_m: tuple[float, ...] = (),
    protected_z_m: tuple[float, ...] = (),
) -> P2Mesh:
    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element, triangle in enumerate(triangle_array):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_to_triangles[tuple(sorted((int(first), int(second))))].append(element)
    edges = np.asarray(sorted(edge_to_triangles), dtype=np.int64)
    midpoint_dofs = np.arange(
        len(vertex_array), len(vertex_array) + len(edges), dtype=np.int64
    )
    edge_to_dof = {
        tuple(edge): int(dof) for edge, dof in zip(edges.tolist(), midpoint_dofs)
    }
    midpoint_coordinates = 0.5 * (vertex_array[edges[:, 0]] + vertex_array[edges[:, 1]])
    p2_nodes = np.vstack((vertex_array, midpoint_coordinates))
    element_dofs = np.empty((len(triangle_array), 6), dtype=np.int64)
    for element, (v0, v1, v2) in enumerate(triangle_array):
        element_dofs[element] = (
            v0,
            v1,
            v2,
            edge_to_dof[tuple(sorted((int(v0), int(v1))))],
            edge_to_dof[tuple(sorted((int(v1), int(v2))))],
            edge_to_dof[tuple(sorted((int(v2), int(v0))))],
        )

    boundary: dict[str, list[int]] = {
        "axis" if domain.r_min_m == 0.0 else "inner_radial": [],
        "outer_radial": [],
        "z_min": [],
        "z_max": [],
    }
    tolerance = 2.0e-12 * max(
        domain.r_max_m - domain.r_min_m, domain.z_max_m - domain.z_min_m, 1.0
    )
    for edge_index, edge in enumerate(edges):
        if len(edge_to_triangles[tuple(edge)]) != 1:
            continue
        points = vertex_array[edge]
        if np.all(np.abs(points[:, 0] - domain.r_min_m) <= tolerance):
            boundary[next(iter(boundary))].append(edge_index)
        elif np.all(np.abs(points[:, 0] - domain.r_max_m) <= tolerance):
            boundary["outer_radial"].append(edge_index)
        elif np.all(np.abs(points[:, 1] - domain.z_min_m) <= tolerance):
            boundary["z_min"].append(edge_index)
        elif np.all(np.abs(points[:, 1] - domain.z_max_m) <= tolerance):
            boundary["z_max"].append(edge_index)
        else:
            raise FEMValidationError("unclassified exterior boundary edge")
    interface_indices: list[int] = []
    interface_pairs: list[tuple[str, str]] = []
    for edge_index, edge in enumerate(edges):
        owners = edge_to_triangles[tuple(edge)]
        if len(owners) == 2 and tags[owners[0]] != tags[owners[1]]:
            interface_indices.append(edge_index)
            interface_pairs.append(tuple(sorted((tags[owners[0]], tags[owners[1]]))))
    return P2Mesh(
        vertex_array,
        triangle_array,
        tags,
        p2_nodes,
        element_dofs,
        edges,
        midpoint_dofs,
        {name: np.asarray(indices, dtype=np.int64) for name, indices in boundary.items()},
        (
            np.arange(len(triangle_array), dtype=np.int64)
            if element_parent_ids is None
            else element_parent_ids
        ),
        np.asarray(interface_indices, dtype=np.int64),
        tuple(interface_pairs),
        refinement_level,
        parent_mesh_sha256,
        tuple(sorted(set(protected_radii_m))),
        tuple(sorted(set(protected_z_m))),
    )


def validate_region_conformity(
    mesh: P2Mesh, polygons: tuple[tuple[str, Polygon], ...]
) -> None:
    """Reject any triangle whose open interior is crossed by a material boundary."""

    bounded = [
        (
            polygon,
            min(point[0] for point in polygon),
            max(point[0] for point in polygon),
            min(point[1] for point in polygon),
            max(point[1] for point in polygon),
        )
        for _, polygon in polygons
    ]
    for triangle in mesh.triangles:
        points = mesh.vertices_rz_m[triangle]
        centroid = np.mean(points, axis=0)
        triangle_bounds = (
            float(np.min(points[:, 0])),
            float(np.max(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 1])),
        )
        for polygon, r_min, r_max, z_min, z_max in bounded:
            if (
                triangle_bounds[1] < r_min
                or triangle_bounds[0] > r_max
                or triangle_bounds[3] < z_min
                or triangle_bounds[2] > z_max
            ):
                continue
            statuses = [_point_in_polygon(point, polygon) for point in points]
            center = _point_in_polygon(centroid, polygon)
            if any(status != center for status in statuses):
                # Vertices on an interface count as inside; midpoint probes distinguish
                # a true crossing from a triangle that merely owns an interface edge.
                probes = [
                    0.98 * centroid + 0.02 * point
                    for point in points
                ]
                if len({_point_in_polygon(probe, polygon) for probe in probes}) > 1:
                    raise FEMValidationError("triangle crosses a material interface")


def mesh_quality(mesh: P2Mesh) -> dict[str, float | int]:
    minimum_angle = 180.0
    maximum_aspect = 0.0
    minimum_area = float("inf")
    for triangle in mesh.triangles:
        points = mesh.vertices_rz_m[triangle]
        lengths = [
            hypot(*(points[(index + 1) % 3] - points[index])) for index in range(3)
        ]
        first = points[1] - points[0]
        second = points[2] - points[0]
        area = 0.5 * abs(float(first[0] * second[1] - first[1] * second[0]))
        minimum_area = min(minimum_area, area)
        maximum_aspect = max(maximum_aspect, max(lengths) ** 2 / (4.0 * area))
        for index in range(3):
            adjacent_a = lengths[index - 1]
            adjacent_b = lengths[index]
            opposite = lengths[(index + 1) % 3]
            cosine = (adjacent_a**2 + adjacent_b**2 - opposite**2) / (
                2.0 * adjacent_a * adjacent_b
            )
            minimum_angle = min(minimum_angle, degrees(acos(max(-1.0, min(1.0, cosine)))))
    return {
        "vertices": len(mesh.vertices_rz_m),
        "triangles": len(mesh.triangles),
        "p2_dofs": len(mesh.p2_nodes_rz_m),
        "minimum_angle_deg": minimum_angle,
        "maximum_aspect_indicator": maximum_aspect,
        "minimum_area_m2": minimum_area,
    }


def element_minimum_angles(mesh: P2Mesh) -> np.ndarray:
    output = np.empty(len(mesh.triangles), dtype=np.float64)
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        lengths = [
            hypot(*(points[(index + 1) % 3] - points[index])) for index in range(3)
        ]
        angles = []
        for index in range(3):
            adjacent_a = lengths[index - 1]
            adjacent_b = lengths[index]
            opposite = lengths[(index + 1) % 3]
            cosine = (adjacent_a**2 + adjacent_b**2 - opposite**2) / (
                2.0 * adjacent_a * adjacent_b
            )
            angles.append(degrees(acos(max(-1.0, min(1.0, cosine)))))
        output[element] = min(angles)
    return output


def element_diameters(mesh: P2Mesh) -> np.ndarray:
    values = np.empty(len(mesh.triangles), dtype=np.float64)
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        values[element] = max(
            hypot(*(points[(index + 1) % 3] - points[index])) for index in range(3)
        )
    return values


def adjacent_size_growth(mesh: P2Mesh) -> float:
    sizes = np.empty(len(mesh.triangles), dtype=np.float64)
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        first = points[1] - points[0]
        second = points[2] - points[0]
        sizes[element] = sqrt(
            abs(float(first[0] * second[1] - first[1] * second[0]))
        )
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element, triangle in enumerate(mesh.triangles):
        v0, v1, v2 = map(int, triangle)
        for edge in ((v0, v1), (v1, v2), (v2, v0)):
            owners[tuple(sorted(edge))].append(element)
    ratios = []
    for elements in owners.values():
        if len(elements) == 2:
            left, right = elements
            ratios.append(
                max(sizes[left], sizes[right])
                / min(sizes[left], sizes[right])
            )
    return max(ratios, default=1.0)


def _refine_mesh_once(
    mesh: P2Mesh,
    domain: Domain,
    marked_elements: np.ndarray | tuple[int, ...] | None = None,
    *,
    reject_below_angle_deg: float = 10.0,
    red_marked: bool = False,
    red_elements: set[int] | None = None,
) -> P2Mesh:
    """Nested conforming longest-edge refinement with deterministic closure.

    Marked elements request their longest edge. If two edges of an element are
    requested, red closure requests its third edge. Elements therefore use only
    stable one-edge bisection or four-child red refinement.
    """

    uniform_red = marked_elements is None
    if uniform_red:
        marked = set(range(len(mesh.triangles)))
    else:
        array = np.asarray(marked_elements)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise FEMValidationError("marked elements must be a one-dimensional integer array")
        if np.any(array < 0) or np.any(array >= len(mesh.triangles)):
            raise FEMValidationError("marked element index is outside element range")
        marked = set(map(int, array))
    edge_lookup = {tuple(map(int, edge)): index for index, edge in enumerate(mesh.edges)}
    triangle_edges: list[tuple[int, int, int]] = []
    split_edges: set[int] = set()
    if uniform_red:
        split_edges.update(range(len(mesh.edges)))
    for element, triangle in enumerate(mesh.triangles):
        v0, v1, v2 = map(int, triangle)
        local = (
            edge_lookup[tuple(sorted((v0, v1)))],
            edge_lookup[tuple(sorted((v1, v2)))],
            edge_lookup[tuple(sorted((v2, v0)))],
        )
        triangle_edges.append(local)
        lengths = [
            float(
                np.linalg.norm(
                    mesh.vertices_rz_m[mesh.edges[edge_index, 1]]
                    - mesh.vertices_rz_m[mesh.edges[edge_index, 0]]
                )
            )
            for edge_index in local
        ]
        longest_edge = local[int(np.argmax(lengths))]
        if element in marked:
            if red_marked or (red_elements is not None and element in red_elements):
                split_edges.update(local)
            else:
                split_edges.add(longest_edge)
    changed = True
    while changed:
        changed = False
        for local in triangle_edges:
            requested = [edge for edge in local if edge in split_edges]
            if len(requested) >= 2:
                for edge in local:
                    if edge not in split_edges:
                        split_edges.add(edge)
                        changed = True

    child_triangles = 0
    for local in triangle_edges:
        split_count = sum(edge in split_edges for edge in local)
        child_triangles += (1, 2, 0, 4)[split_count]
    boundary_indices = {
        int(edge)
        for values in mesh.boundary_edges.values()
        for edge in values
    }
    boundary_edges = len(boundary_indices) + len(boundary_indices & split_edges)
    projected_vertices = len(mesh.vertices_rz_m) + len(split_edges)
    projected_edges = (3 * child_triangles + boundary_edges) // 2
    from .resource_policy import guard_allocation

    guard_allocation(
        "mesh_refinement",
        p2_dofs=projected_vertices + projected_edges,
        triangles=child_triangles,
        robin_edges=sum(
            len(mesh.boundary_edges[name])
            + sum(
                int(edge) in split_edges for edge in mesh.boundary_edges[name]
            )
            for name in ("outer_radial", "z_min", "z_max")
        ),
    )

    ordered_split_edges = sorted(split_edges)
    new_vertices = mesh.vertices_rz_m.tolist()
    edge_new_vertex: dict[int, int] = {}
    for edge_index in ordered_split_edges:
        edge_new_vertex[edge_index] = len(new_vertices)
        edge = mesh.edges[edge_index]
        new_vertices.append(
            (
                0.5
                * (
                    mesh.vertices_rz_m[edge[0]]
                    + mesh.vertices_rz_m[edge[1]]
                )
            ).tolist()
        )
    children: list[tuple[int, int, int]] = []
    tags: list[str] = []
    parents: list[int] = []

    def append(first: int, second: int, third: int, parent: int) -> None:
        triangle = (first, second, third)
        points = np.asarray([new_vertices[index] for index in triangle])
        cross = (points[1, 0] - points[0, 0]) * (
            points[2, 1] - points[0, 1]
        ) - (points[1, 1] - points[0, 1]) * (points[2, 0] - points[0, 0])
        children.append(triangle if cross > 0.0 else (first, third, second))
        tags.append(mesh.triangle_region_ids[parent])
        parents.append(parent)

    for element, triangle in enumerate(mesh.triangles):
        v0, v1, v2 = map(int, triangle)
        e01, e12, e20 = triangle_edges[element]
        requested = [edge for edge in (e01, e12, e20) if edge in split_edges]
        if not requested:
            append(v0, v1, v2, element)
        elif len(requested) == 1:
            edge = requested[0]
            midpoint = edge_new_vertex[edge]
            if edge == e01:
                append(v0, midpoint, v2, element)
                append(midpoint, v1, v2, element)
            elif edge == e12:
                append(v1, midpoint, v0, element)
                append(midpoint, v2, v0, element)
            else:
                append(v2, midpoint, v1, element)
                append(midpoint, v0, v1, element)
        elif len(requested) == 3:
            m01, m12, m20 = (
                edge_new_vertex[e01],
                edge_new_vertex[e12],
                edge_new_vertex[e20],
            )
            append(v0, m01, m20, element)
            append(m01, v1, m12, element)
            append(m20, m12, v2, element)
            append(m01, m12, m20, element)
        else:
            raise FEMValidationError("refinement closure left an unsupported two-edge split")
    refined = _construct_p2_mesh(
        np.asarray(new_vertices, dtype=np.float64),
        np.asarray(children, dtype=np.int64),
        tuple(tags),
        domain,
        element_parent_ids=np.asarray(parents, dtype=np.int64),
        refinement_level=mesh.refinement_level + 1,
        parent_mesh_sha256=mesh.sha256,
        protected_radii_m=mesh.protected_radii_m,
        protected_z_m=mesh.protected_z_m,
    )
    quality = mesh_quality(refined)
    if quality["minimum_angle_deg"] < reject_below_angle_deg:
        raise FEMValidationError(
            f"refined mesh minimum angle {quality['minimum_angle_deg']:.6g} deg is below "
            f"{reject_below_angle_deg:.6g} deg"
        )
    return refined


def _area_sizes(mesh: P2Mesh) -> np.ndarray:
    sizes = np.empty(len(mesh.triangles), dtype=np.float64)
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        first = points[1] - points[0]
        second = points[2] - points[0]
        sizes[element] = sqrt(
            abs(float(first[0] * second[1] - first[1] * second[0]))
        )
    return sizes


def _coarse_parents_across_growth_violations(
    mesh: P2Mesh, maximum_growth: float
) -> set[int]:
    sizes = _area_sizes(mesh)
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element, triangle in enumerate(mesh.triangles):
        v0, v1, v2 = map(int, triangle)
        for edge in ((v0, v1), (v1, v2), (v2, v0)):
            owners[tuple(sorted(edge))].append(element)
    coarse_parents: set[int] = set()
    for adjacent in owners.values():
        if len(adjacent) != 2:
            continue
        left, right = adjacent
        ratio = max(sizes[left], sizes[right]) / min(sizes[left], sizes[right])
        if ratio > maximum_growth * (1.0 + 2.0e-13):
            coarse = left if sizes[left] > sizes[right] else right
            coarse_parents.add(int(mesh.element_parent_ids[coarse]))
    return coarse_parents


def refine_mesh(
    mesh: P2Mesh,
    domain: Domain,
    marked_elements: np.ndarray | tuple[int, ...] | None = None,
    *,
    reject_below_angle_deg: float = 10.0,
    red_marked: bool = False,
    maximum_adjacent_size_growth: float | None = None,
) -> P2Mesh:
    """Refine conformingly and optionally propagate a strict gradation closure.

    The closure repeatedly marks the coarse parent across every violating
    child edge. A parent already selected for longest-edge bisection is
    promoted to red refinement. The returned mesh remains a single nested
    child level of ``mesh`` and retains direct parent IDs.
    """

    if maximum_adjacent_size_growth is not None and (
        isinstance(maximum_adjacent_size_growth, bool)
        or not isinstance(maximum_adjacent_size_growth, (int, float))
        or not np.isfinite(maximum_adjacent_size_growth)
        or maximum_adjacent_size_growth <= 1.0
    ):
        raise FEMValidationError(
            "maximum adjacent size growth must be finite and greater than one"
        )
    if maximum_adjacent_size_growth is None or marked_elements is None:
        return _refine_mesh_once(
            mesh,
            domain,
            marked_elements,
            reject_below_angle_deg=reject_below_angle_deg,
            red_marked=red_marked,
        )

    requested = np.asarray(marked_elements)
    if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
        raise FEMValidationError(
            "marked elements must be a one-dimensional integer array"
        )
    if np.any(requested < 0) or np.any(requested >= len(mesh.triangles)):
        raise FEMValidationError("marked element index is outside element range")
    marked = set(map(int, requested))
    red = set(marked) if red_marked else set()
    parent_sizes = _area_sizes(mesh)
    edge_owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for element, triangle in enumerate(mesh.triangles):
        v0, v1, v2 = map(int, triangle)
        for edge in ((v0, v1), (v1, v2), (v2, v0)):
            edge_owners[tuple(sorted(edge))].append(element)
    adjacent_parents = [
        tuple(owners)
        for owners in edge_owners.values()
        if len(owners) == 2
    ]

    def propagate() -> None:
        states = np.zeros(len(mesh.triangles), dtype=np.int8)
        if marked:
            states[np.asarray(sorted(marked), dtype=np.int64)] = 1
        if red:
            states[np.asarray(sorted(red), dtype=np.int64)] = 2
        factors = np.asarray((1.0, 1.0 / sqrt(2.0), 0.5))
        for _ in range(2 * len(mesh.triangles) + 1):
            changed = False
            effective = parent_sizes * factors[states]
            for left, right in adjacent_parents:
                ratio = max(effective[left], effective[right]) / min(
                    effective[left], effective[right]
                )
                if ratio <= maximum_adjacent_size_growth * (1.0 + 2.0e-13):
                    continue
                coarse = left if effective[left] > effective[right] else right
                if states[coarse] >= 2:
                    raise FEMValidationError(
                        "single-level bisection cannot satisfy the requested gradation"
                    )
                states[coarse] += 1
                effective[coarse] = parent_sizes[coarse] * factors[states[coarse]]
                changed = True
            if not changed:
                marked.update(map(int, np.flatnonzero(states >= 1)))
                red.update(map(int, np.flatnonzero(states >= 2)))
                return
        raise FEMValidationError("parent-level gradation closure did not terminate")

    for _ in range(4):
        propagate()
        candidate = _refine_mesh_once(
            mesh,
            domain,
            np.asarray(sorted(marked), dtype=np.int64),
            reject_below_angle_deg=0.0,
            red_elements=red,
        )
        coarse_parents = _coarse_parents_across_growth_violations(
            candidate, maximum_adjacent_size_growth
        )
        if not coarse_parents:
            quality = mesh_quality(candidate)
            if quality["minimum_angle_deg"] < reject_below_angle_deg:
                raise FEMValidationError(
                    f"refined mesh minimum angle {quality['minimum_angle_deg']:.6g} deg "
                    f"is below {reject_below_angle_deg:.6g} deg"
                )
            return candidate
        changed = False
        for parent in sorted(coarse_parents):
            if parent not in marked:
                marked.add(parent)
                changed = True
            elif parent not in red:
                red.add(parent)
                changed = True
        if not changed:
            raise FEMValidationError(
                "single-level bisection cannot satisfy the requested gradation"
            )
    raise FEMValidationError("gradation closure exceeded four topology checks")


def improve_mesh_quality(
    mesh: P2Mesh,
    domain: Domain,
    *,
    target_angle_deg: float = 20.0,
    reject_below_angle_deg: float = 10.0,
    max_passes: int = 8,
) -> P2Mesh:
    """Refine longest edges of poor elements until the quality target or limit."""

    current = mesh
    for _ in range(max_passes):
        angles = element_minimum_angles(current)
        marked = np.flatnonzero(angles < target_angle_deg)
        if not len(marked):
            break
        current = refine_mesh(
            current, domain, marked, reject_below_angle_deg=0.0
        )
    quality = mesh_quality(current)
    if quality["minimum_angle_deg"] < reject_below_angle_deg:
        raise FEMValidationError(
            f"quality recovery minimum angle {quality['minimum_angle_deg']:.6g} deg "
            f"is below {reject_below_angle_deg:.6g} deg"
        )
    return current
