"""Definition v3 machinery: axis nulls, Jacobian classification, separatrix tracing, wall cusps.

The frozen definition (protocol.json ``definition_v3``) follows the HEMP/DCFT literature
(Gildea 2012 MIT thesis; Kornfeld, Koch, Harmann IEPC-2007-108; Koch et al. IEPC-2011-236;
Lewerentz and Schneider 2023, DOI 10.3390/app13063491):

1. an *axis null* is a point (0, z_k) where B_z(0, z) changes sign (B_r == 0 on the axis);
   it is classified X/O/degenerate with the finite-difference Jacobian and winding index of
   the topology characterization v1 (imported with its frozen parameters) and, in addition,
   with the analytic Jacobian of the C1 bicubic interpolant;
2. the *separatrix* of an axis X-null is the field line leaving the null along the radial
   eigen-direction of its Jacobian; it is traced with an event-aware RK4 field-line
   integrator (the v4 coupling scheme: RK4 on the unit field direction, wall crossing
   resolved by step halving) on the C1 bicubic interpolant until it reaches the wall
   cylinder r = r_w or leaves the axial window;
3. a *wall cusp* is a separatrix-wall intersection z_c inside the straight dielectric
   section [z_min, exit_start]; a *cell* is the wall interval between consecutive cusps
   (plus the anode-side and exit-side partial cells); the mirror descriptor per cell is
   |B|(r_w, z_c) over the minimum of |B| along the wall between the cusps and over the
   on-axis |B_z| peak between the two axis nulls that generated the cusps (Lewerentz R_m
   with r_0 -> 0; the literal minimum of |B| on the axis between two nulls is zero by
   definition, which is why the peak is the reported axis descriptor).

Everything in this module is a pure function of (field grid, geometry, policy); no wall
clock, no randomness. Cross-checks: the wall intersection must coincide with the root of
psi(r_w, z) - psi_axis on the same interpolant (the separatrix is the g = 0 contour of the
axis-regular flux variable g = (psi - psi_axis) / r^2), and the trace is repeated with the
existing bilinear v4 integrator step as a reported comparison.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cft_revival.coupling import hash_psi_map
from cft_revival.coupling.models import CouplingValidationError, TopologyResolutionError
from cft_revival.coupling.v3_models import ValidatedPsiMap
from cft_revival.coupling.v4_records import _WallStageCrossing, _rk4_step
from cft_revival.orbit_mc import PsiBicubicField
from cft_revival.orbit_mc.models import OrbitMCError

# Reused characterization-v1 Jacobian / winding-index classification (frozen parameters).
from experiments.cft_topology_characterization_v1 import experiment as characterization_v1

TRACING_MATERIAL_ID = "v3-tracing-vacuum-bore"
Point = tuple[float, float]


@dataclass(frozen=True)
class ChannelGeometry:
    """Straight dielectric section and stack layout used to classify separatrix endpoints."""

    wall_radius_m: float
    straight_z_min_m: float
    straight_z_max_m: float
    chamber_length_m: float
    stage_pitch_m: float
    stage_centres_m: tuple[float, ...]
    injector_length_m: float

    def __post_init__(self) -> None:
        if not (self.wall_radius_m > 0.0 and self.straight_z_max_m > self.straight_z_min_m):
            raise ValueError("channel geometry must have a positive radius and straight span")
        if self.stage_pitch_m <= 0.0 or not self.stage_centres_m:
            raise ValueError("channel geometry requires a positive pitch and stage centres")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stage_centres_m"] = list(self.stage_centres_m)
        return value

    @property
    def stage_gap_centres_m(self) -> tuple[float, ...]:
        """Axial centres of the inter-magnet gaps plus the two outer magnet ends."""

        centres = self.stage_centres_m
        inner = tuple(0.5 * (left + right) for left, right in zip(centres[:-1], centres[1:]))
        return (centres[0] - 0.5 * self.stage_pitch_m,) + inner + (centres[-1] + 0.5 * self.stage_pitch_m,)


@dataclass(frozen=True)
class TopologyPolicy:
    """Frozen numerical parameters of definition v3 (mirrors protocol.json#definition_v3)."""

    axis_root_bracket_tolerance_m: float = 1.0e-12
    axis_root_max_bisections: int = 200
    axis_window_margin_mesh_factor: float = 2.5
    axis_window_pitch_factor: float = 1.0
    seed_radius_cell_fraction: float = 0.25
    trace_step_cell_fraction: float = 0.1
    trace_max_steps: int = 400000
    wall_tolerance_m: float = 1.0e-9
    wall_event_max_halvings: int = 64
    null_field_floor_relative: float = 1.0e-13
    flux_root_search_half_width_cells: float = 2.0
    trace_flux_root_tolerance_m: float = 5.0e-5
    boundary_ambiguity_tolerance_m: float = 2.5e-4
    wall_samples_per_cell: int = 400
    axis_samples_per_interval: int = 400
    jacobian_step_cell_fraction: float = 1.0e-3
    path_subsample_stride: int = 8
    path_max_points: int = 400

    @classmethod
    def from_protocol(cls, declaration: Mapping[str, Any]) -> "TopologyPolicy":
        return cls(**{key: declaration[key] for key in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# Tracing grid (r <= r_w, full axial range) and interpolants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TracingGrid:
    r_m: np.ndarray
    z_m: np.ndarray
    psi_wb: np.ndarray
    b_r_t: np.ndarray
    b_z_t: np.ndarray
    wall_radius_m: float

    @property
    def dr_m(self) -> float:
        return float(self.r_m[1] - self.r_m[0])

    @property
    def dz_m(self) -> float:
        return float(self.z_m[1] - self.z_m[0])

    @property
    def mesh_scale_m(self) -> float:
        return max(self.dr_m, self.dz_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_m": self.r_m.tolist(),
            "z_m": self.z_m.tolist(),
            "psi_wb": self.psi_wb.tolist(),
            "b_r_t": self.b_r_t.tolist(),
            "b_z_t": self.b_z_t.tolist(),
            "wall_radius_m": self.wall_radius_m,
        }

    def validated_psi_map(self) -> ValidatedPsiMap:
        """Bilinear map for the v1 Jacobian classification and the v4 integrator step."""

        r = tuple(float(v) for v in self.r_m)
        z = tuple(float(v) for v in self.z_m)
        psi = tuple(tuple(float(v) for v in row) for row in self.psi_wb)
        br = tuple(tuple(float(v) for v in row) for row in self.b_r_t)
        bz = tuple(tuple(float(v) for v in row) for row in self.b_z_t)
        provisional = ValidatedPsiMap(r, z, psi, br, bz, "0" * 64)
        return ValidatedPsiMap(r, z, psi, br, bz, hash_psi_map(provisional))


def tracing_grid(
    r_m: Sequence[float],
    z_m: Sequence[float],
    psi_wb: Sequence[Sequence[float]],
    b_r_t: Sequence[Sequence[float]],
    b_z_t: Sequence[Sequence[float]],
    wall_radius_m: float,
) -> TracingGrid:
    """Restrict a regular (r, z) map to the radial nodes 0..first node >= wall radius."""

    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if r.ndim != 1 or z.ndim != 1 or len(r) < 4 or len(z) < 4:
        raise ValueError("tracing grid requires at least 4x4 nodes")
    if r[0] != 0.0:
        raise ValueError("tracing grid must start on the axis (r_m[0] == 0)")
    i_max = int(np.searchsorted(r, wall_radius_m, side="left"))
    if i_max >= len(r):
        raise ValueError("wall radius lies outside the field grid")
    if i_max < 3:
        raise ValueError("fewer than three radial cells across the bore; refine the field")
    psi = np.asarray(psi_wb, dtype=np.float64)[: i_max + 1]
    br = np.asarray(b_r_t, dtype=np.float64)[: i_max + 1]
    bz = np.asarray(b_z_t, dtype=np.float64)[: i_max + 1]
    if psi.shape != (i_max + 1, len(z)) or br.shape != psi.shape or bz.shape != psi.shape:
        raise ValueError("tracing grid arrays have inconsistent shapes")
    if not (np.isfinite(psi).all() and np.isfinite(br).all() and np.isfinite(bz).all()):
        raise ValueError("tracing grid arrays must be finite")
    return TracingGrid(r[: i_max + 1].copy(), z.copy(), psi.copy(), br.copy(), bz.copy(), float(wall_radius_m))


def bicubic_field(
    grid: TracingGrid,
    *,
    source_identity_sha256: str,
    minimum_certificate_tightness_ratio: float,
    with_reference: bool = True,
) -> PsiBicubicField:
    material = np.full(grid.psi_wb.shape, TRACING_MATERIAL_ID, dtype=object)
    return PsiBicubicField(
        grid.r_m,
        grid.z_m,
        grid.psi_wb,
        material_id=material,
        plasma_material_id=TRACING_MATERIAL_ID,
        reference_br_t=grid.b_r_t if with_reference else None,
        reference_bz_t=grid.b_z_t if with_reference else None,
        minimum_certificate_tightness_ratio=minimum_certificate_tightness_ratio,
        source_identity_sha256=source_identity_sha256,
    )


# --------------------------------------------------------------------------
# Axis nulls
# --------------------------------------------------------------------------


def axis_bz(field: PsiBicubicField, z: float) -> float:
    return float(field.field_cylindrical(0.0, z)[1])


def _bisect(function: Callable[[float], float], low: float, high: float, policy: TopologyPolicy) -> tuple[float, int, float]:
    f_low = function(low)
    f_high = function(high)
    if f_low == 0.0:
        return low, 0, 0.0
    if f_high == 0.0:
        return high, 0, 0.0
    if f_low * f_high > 0.0:
        raise ValueError("bisection bracket does not change sign")
    iterations = 0
    while iterations < policy.axis_root_max_bisections:
        middle = 0.5 * (low + high)
        f_middle = function(middle)
        iterations += 1
        if f_middle == 0.0:
            return middle, iterations, 0.0
        if f_low * f_middle < 0.0:
            high, f_high = middle, f_middle
        else:
            low, f_low = middle, f_middle
        if high - low <= policy.axis_root_bracket_tolerance_m:
            return 0.5 * (low + high), iterations, high - low
    raise ValueError("bisection did not converge to the declared bracket tolerance")


def axis_window(
    grid: TracingGrid,
    geometry: ChannelGeometry,
    policy: TopologyPolicy,
    *,
    mesh_scale_m: float | None = None,
) -> tuple[float, float]:
    """Axial search window: channel extended by one pitch, kept inside the Jacobian stencil margin.

    ``mesh_scale_m`` lets the refined map reuse the accepted map's margin so both resolutions
    search exactly the same window (otherwise a null near the window edge could enter or
    leave the count for a purely geometric reason).
    """

    margin = policy.axis_window_margin_mesh_factor * (grid.mesh_scale_m if mesh_scale_m is None else mesh_scale_m)
    low = max(float(grid.z_m[0]) + margin, geometry.straight_z_min_m - policy.axis_window_pitch_factor * geometry.stage_pitch_m)
    high = min(float(grid.z_m[-1]) - margin, geometry.chamber_length_m + policy.axis_window_pitch_factor * geometry.stage_pitch_m)
    if not high > low:
        raise ValueError("axis search window is empty")
    return low, high


def analytic_axis_jacobian(field: PsiBicubicField, z_k: float, grid: TracingGrid, policy: TopologyPolicy) -> dict[str, Any]:
    """Jacobian of (B_r, B_z) at the axis null from the C1 interpolant and its eigen-directions."""

    h_r = policy.jacobian_step_cell_fraction * grid.dr_m
    h_z = policy.jacobian_step_cell_fraction * grid.dz_m
    br_plus, bz_plus = field.field_cylindrical(h_r, z_k)
    _, bz_zplus = field.field_cylindrical(0.0, z_k + h_z)
    _, bz_zminus = field.field_cylindrical(0.0, z_k - h_z)
    br_zplus, _ = field.field_cylindrical(h_r, z_k + h_z)
    br_zminus, _ = field.field_cylindrical(h_r, z_k - h_z)
    bz_axis = axis_bz(field, z_k)
    j_rr = br_plus / h_r
    j_rz = (br_zplus - br_zminus) / (2.0 * h_z)  # ~0 on the axis (B_r odd in r)
    j_zr = (bz_plus - bz_axis) / h_r  # ~0 on the axis (B_z even in r)
    j_zz = (bz_zplus - bz_zminus) / (2.0 * h_z)
    matrix = np.array([[j_rr, j_rz], [j_zr, j_zz]], dtype=np.float64)
    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    determinant = float(np.linalg.det(matrix))
    # Divergence identity on the axis: dBr/dr + Br/r + dBz/dz -> 2 J_rr + J_zz = 0.
    divergence_residual = abs(2.0 * j_rr + j_zz) / scale
    radial_index = int(np.argmax(np.abs(eigenvectors[0, :])))
    radial_vector = eigenvectors[:, radial_index]
    axial_vector = eigenvectors[:, 1 - radial_index]
    return {
        "matrix_t_per_m": matrix.tolist(),
        "finite_difference_steps_m": [h_r, h_z],
        "determinant_t2_per_m2": determinant,
        "classification": "X" if determinant < -1.0e-6 * scale * scale else ("O" if determinant > 1.0e-6 * scale * scale else "degenerate"),
        "eigenvalues_t_per_m": eigenvalues.tolist(),
        "radial_eigenvector": [float(radial_vector[0]), float(radial_vector[1])],
        "axial_eigenvector": [float(axial_vector[0]), float(axial_vector[1])],
        "radial_eigenvalue_t_per_m": float(eigenvalues[radial_index]),
        "separatrix_direction_is_radial": bool(abs(radial_vector[0]) >= 0.99),
        "divergence_identity_relative_residual": float(divergence_residual),
        "g_z_wb_per_m3": float(-j_rr),
    }


def _sign_change_brackets(samples: Sequence[float], values: Sequence[float]) -> list[tuple[float, float] | float]:
    """Brackets (or exact node roots) where the sampled axis B_z changes sign."""

    result: list[tuple[float, float] | float] = []
    count = len(samples)
    for index in range(count - 1):
        left, right = values[index], values[index + 1]
        if left == 0.0:
            # Exact node zero: a root only when the neighbouring nonzero values have opposite signs.
            previous = next((values[k] for k in range(index - 1, -1, -1) if values[k] != 0.0), None)
            following = next((values[k] for k in range(index + 1, count) if values[k] != 0.0), None)
            if previous is not None and following is not None and previous * following < 0.0:
                result.append(samples[index])
            continue
        if right != 0.0 and left * right < 0.0:
            result.append((samples[index], samples[index + 1]))
    return result


def find_axis_nulls(
    field: PsiBicubicField,
    grid: TracingGrid,
    geometry: ChannelGeometry,
    policy: TopologyPolicy,
    *,
    window_m: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Converged axis nulls in the search window with v1 and analytic classifications."""

    low, high = axis_window(grid, geometry, policy) if window_m is None else (float(window_m[0]), float(window_m[1]))
    nodes = [float(z) for z in grid.z_m if low < z < high]
    samples = sorted({low, *nodes, high})
    values = [axis_bz(field, z) for z in samples]
    bilinear = grid.validated_psi_map()
    nulls: list[dict[str, Any]] = []
    for bracket in _sign_change_brackets(samples, values):
        if isinstance(bracket, tuple):
            z_k, iterations, width = _bisect(lambda z: axis_bz(field, z), bracket[0], bracket[1], policy)
            bracket_record = [bracket[0], bracket[1]]
        else:
            z_k, iterations, width = float(bracket), 0, 0.0
            bracket_record = [z_k, z_k]
        if nulls and abs(nulls[-1]["z_m"] - z_k) <= policy.axis_root_bracket_tolerance_m:
            continue
        analytic = analytic_axis_jacobian(field, z_k, grid, policy)
        try:
            v1_local = characterization_v1.local_topology(bilinear, (0.0, z_k), grid.mesh_scale_m)
        except Exception as error:  # recorded, never hidden
            v1_local = {"classification": "degenerate", "jacobian_converged": False, "error": f"{type(error).__name__}: {error}"}
        if z_k < geometry.straight_z_min_m:
            zone = "anode_side"
        elif z_k <= geometry.straight_z_max_m:
            zone = "channel"
        elif z_k <= geometry.chamber_length_m:
            zone = "divergent_exit"
        else:
            zone = "downstream"
        nearest_gap = min(geometry.stage_gap_centres_m, key=lambda value: abs(value - z_k))
        nearest_centre = min(geometry.stage_centres_m, key=lambda value: abs(value - z_k))
        nulls.append(
            {
                "null_id": "",
                "z_m": z_k,
                "grid_bracket_m": bracket_record,
                "bisections": iterations,
                "final_bracket_width_m": width,
                "residual_bz_t": axis_bz(field, z_k),
                "converged": width <= policy.axis_root_bracket_tolerance_m,
                "zone": zone,
                "distance_to_nearest_stage_gap_m": abs(nearest_gap - z_k),
                "distance_to_nearest_stage_centre_m": abs(nearest_centre - z_k),
                "analytic_jacobian": analytic,
                "v1_local_topology": v1_local,
                "classification": v1_local.get("classification", "degenerate"),
                "classification_agrees": v1_local.get("classification") == analytic["classification"],
            }
        )
    for index, null in enumerate(nulls):
        null["null_id"] = f"axis-null-{index + 1:02d}"
    return {
        "window_m": [low, high],
        "sample_count": len(samples),
        "nulls": nulls,
        "count": len(nulls),
        "channel_count": sum(null["zone"] == "channel" for null in nulls),
        "all_converged": all(null["converged"] for null in nulls),
        "all_x_type": all(null["classification"] == "X" for null in nulls),
        "all_classifications_agree": all(null["classification_agrees"] for null in nulls),
    }


# --------------------------------------------------------------------------
# Separatrix tracing
# --------------------------------------------------------------------------


class _WallCrossing(Exception):
    pass


class _NullEncountered(Exception):
    pass


def _bicubic_unit(field: PsiBicubicField, wall_radius_m: float, floor_t: float, direction: int) -> Callable[[Point], Point]:
    def unit(point: Point) -> Point:
        r, z = point
        if r > wall_radius_m:
            raise _WallCrossing
        br, bz = field.field_cylindrical(abs(r), z)
        if r < 0.0:
            br = -br
        magnitude = math.hypot(br, bz)
        if not math.isfinite(magnitude) or magnitude <= floor_t:
            raise _NullEncountered
        return direction * br / magnitude, direction * bz / magnitude

    return unit


def _rk4(unit: Callable[[Point], Point], point: Point, step: float) -> Point:
    k1 = unit(point)
    k2 = unit((point[0] + 0.5 * step * k1[0], point[1] + 0.5 * step * k1[1]))
    k3 = unit((point[0] + 0.5 * step * k2[0], point[1] + 0.5 * step * k2[1]))
    k4 = unit((point[0] + step * k3[0], point[1] + step * k3[1]))
    return (
        point[0] + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        point[1] + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _drive_trace(
    stepper: Callable[[Point, float], Point],
    seed: Point,
    wall_radius_m: float,
    z_window: tuple[float, float],
    base_step: float,
    policy: TopologyPolicy,
) -> dict[str, Any]:
    """Event-aware driver shared by the bicubic and the v4 bilinear integrator steps."""

    points: list[Point] = [seed]
    termination = "step_limit"
    path_length = 0.0
    halvings_total = 0
    for _ in range(policy.trace_max_steps):
        current = points[-1]
        remaining = wall_radius_m - current[0]
        if 0.0 <= remaining <= policy.wall_tolerance_m:
            points.append((wall_radius_m, current[1]))
            termination = "wall"
            break
        step = base_step
        if remaining < 2.0 * step:
            step = max(0.5 * remaining, 0.25 * policy.wall_tolerance_m)
        candidate: Point | None = None
        for _attempt in range(policy.wall_event_max_halvings + 1):
            try:
                candidate = stepper(current, step)
                break
            except (_WallCrossing, _WallStageCrossing):
                step *= 0.5
                halvings_total += 1
            except (_NullEncountered, CouplingValidationError):
                termination = "null"
                break
            except (OrbitMCError, TopologyResolutionError, ValueError):
                termination = "domain_z"
                break
        if termination in ("null", "domain_z") and candidate is None:
            break
        if candidate is None:
            termination = "wall_event_unresolved"
            break
        if candidate[0] < 0.0:
            candidate = (-candidate[0], candidate[1])
        if candidate[0] >= wall_radius_m - policy.wall_tolerance_m:
            fraction = 1.0 if candidate[0] == current[0] else min(1.0, max(0.0, (wall_radius_m - current[0]) / (candidate[0] - current[0])))
            endpoint = (wall_radius_m, current[1] + fraction * (candidate[1] - current[1]))
            path_length += math.hypot(endpoint[0] - current[0], endpoint[1] - current[1])
            points.append(endpoint)
            termination = "wall"
            break
        if not (z_window[0] <= candidate[1] <= z_window[1]):
            clipped_z = min(max(candidate[1], z_window[0]), z_window[1])
            points.append((candidate[0], clipped_z))
            path_length += math.hypot(candidate[0] - current[0], clipped_z - current[1])
            termination = "domain_z"
            break
        path_length += math.hypot(candidate[0] - current[0], candidate[1] - current[1])
        points.append(candidate)
    return {
        "termination": termination,
        "endpoint_rz_m": [points[-1][0], points[-1][1]],
        "step_count": len(points) - 1,
        "path_length_m": path_length,
        "wall_event_halvings": halvings_total,
        "points": points,
    }


def _subsample(points: Sequence[Point], policy: TopologyPolicy) -> list[list[float]]:
    if len(points) <= policy.path_max_points:
        stride = 1
    else:
        stride = max(policy.path_subsample_stride, math.ceil(len(points) / policy.path_max_points))
    kept = list(points[::stride])
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return [[float(r), float(z)] for r, z in kept]


def trace_separatrix(
    field: PsiBicubicField,
    grid: TracingGrid,
    null: Mapping[str, Any],
    geometry: ChannelGeometry,
    policy: TopologyPolicy,
    *,
    z_window: tuple[float, float],
    keep_path: bool,
) -> dict[str, Any]:
    """Trace the separatrix of one axis null outward to the wall cylinder (bicubic interpolant)."""

    z_k = float(null["z_m"])
    seed_r = policy.seed_radius_cell_fraction * grid.dr_m
    floor = policy.null_field_floor_relative * field.max_b_t
    br_seed, bz_seed = field.field_cylindrical(seed_r, z_k)
    direction = 1 if br_seed > 0.0 else (-1 if br_seed < 0.0 else 0)
    psi_axis = float(field.psi_wb[0, 0])
    psi_seed = float(field.psi_gradient(seed_r, z_k)[0])
    if direction == 0:
        return {
            "null_id": null["null_id"],
            "seed_rz_m": [seed_r, z_k],
            "direction": 0,
            "termination": "degenerate_seed",
            "endpoint_rz_m": [seed_r, z_k],
            "step_count": 0,
            "path_length_m": 0.0,
            "wall_event_halvings": 0,
            "psi_seed_minus_axis_wb": psi_seed - psi_axis,
            "psi_drift_wb": 0.0,
            "reaches_wall": False,
            "path_rz_m": [[seed_r, z_k]] if keep_path else None,
        }
    unit = _bicubic_unit(field, geometry.wall_radius_m, floor, direction)
    driven = _drive_trace(
        lambda point, step: _rk4(unit, point, step),
        (seed_r, z_k),
        geometry.wall_radius_m,
        z_window,
        policy.trace_step_cell_fraction * min(grid.dr_m, grid.dz_m),
        policy,
    )
    end_r, end_z = driven["endpoint_rz_m"]
    reaches_wall = driven["termination"] == "wall"
    result: dict[str, Any] = {
        "null_id": null["null_id"],
        "seed_rz_m": [seed_r, z_k],
        "direction": direction,
        "termination": driven["termination"],
        "endpoint_rz_m": [end_r, end_z],
        "step_count": driven["step_count"],
        "path_length_m": driven["path_length_m"],
        "wall_event_halvings": driven["wall_event_halvings"],
        "psi_seed_minus_axis_wb": psi_seed - psi_axis,
        "reaches_wall": reaches_wall,
        "path_rz_m": _subsample(driven["points"], policy) if keep_path else None,
    }
    try:
        psi_end = float(field.psi_gradient(min(end_r, float(grid.r_m[-1])), end_z)[0])
        result["psi_drift_wb"] = psi_end - psi_seed
    except OrbitMCError:
        result["psi_drift_wb"] = None
    if reaches_wall:
        br, bz = field.field_cylindrical(geometry.wall_radius_m, end_z)
        magnitude = math.hypot(br, bz)
        result.update(
            {
                "z_c_m": end_z,
                "wall_b_t": magnitude,
                "wall_b_r_t": br,
                "wall_b_z_t": bz,
                "wall_normal_component_t": br,
                "angle_to_wall_normal_deg": math.degrees(math.atan2(abs(bz), abs(br))),
                "z_c_flux_root_m": wall_flux_root(field, grid, geometry.wall_radius_m, end_z, psi_axis, policy),
            }
        )
        root = result["z_c_flux_root_m"]
        result["flux_root_difference_m"] = None if root is None else abs(root - end_z)
        result["flux_root_consistent"] = root is not None and abs(root - end_z) <= policy.trace_flux_root_tolerance_m
    return result


def wall_flux_root(
    field: PsiBicubicField,
    grid: TracingGrid,
    wall_radius_m: float,
    z_guess: float,
    psi_axis: float,
    policy: TopologyPolicy,
) -> float | None:
    """Root of psi(r_w, z) - psi_axis nearest to z_guess (the separatrix is the g = 0 contour)."""

    half = policy.flux_root_search_half_width_cells * grid.dz_m
    low = max(float(grid.z_m[0]), z_guess - half)
    high = min(float(grid.z_m[-1]), z_guess + half)
    count = 64
    samples = [low + (high - low) * index / count for index in range(count + 1)]

    def value(z: float) -> float:
        return float(field.psi_gradient(wall_radius_m, z)[0]) - psi_axis

    values = [value(z) for z in samples]
    best: float | None = None
    for index in range(count):
        if values[index] == 0.0:
            candidate = samples[index]
        elif values[index] * values[index + 1] < 0.0:
            candidate, _, _ = _bisect(value, samples[index], samples[index + 1], policy)
        else:
            continue
        if best is None or abs(candidate - z_guess) < abs(best - z_guess):
            best = candidate
    return best


def trace_separatrix_bilinear_v4(
    grid: TracingGrid,
    bilinear: ValidatedPsiMap,
    null: Mapping[str, Any],
    geometry: ChannelGeometry,
    policy: TopologyPolicy,
    *,
    z_window: tuple[float, float],
) -> dict[str, Any]:
    """The same trace with the existing coupling-v4 RK4 step on the bilinear grid (comparison)."""

    class _Snapshot:
        field_map = bilinear

    z_k = float(null["z_m"])
    seed_r = policy.seed_radius_cell_fraction * grid.dr_m
    try:
        from cft_revival.coupling.surfaces import bilinear_sample

        br_seed = bilinear_sample(bilinear, bilinear.b_r_t, (seed_r, z_k))
    except (TopologyResolutionError, CouplingValidationError):
        br_seed = 0.0
    direction = 1 if br_seed > 0.0 else (-1 if br_seed < 0.0 else 0)
    if direction == 0:
        return {"null_id": null["null_id"], "termination": "degenerate_seed", "reaches_wall": False, "z_c_m": None}
    snapshot = _Snapshot()
    driven = _drive_trace(
        lambda point, step: _rk4_step(snapshot, point, direction, step, geometry.wall_radius_m),
        (seed_r, z_k),
        geometry.wall_radius_m,
        z_window,
        policy.trace_step_cell_fraction * min(grid.dr_m, grid.dz_m),
        policy,
    )
    reaches = driven["termination"] == "wall"
    return {
        "null_id": null["null_id"],
        "integrator": "cft_revival.coupling.v4_records._rk4_step (bilinear grid samples)",
        "termination": driven["termination"],
        "endpoint_rz_m": driven["endpoint_rz_m"],
        "step_count": driven["step_count"],
        "reaches_wall": reaches,
        "z_c_m": driven["endpoint_rz_m"][1] if reaches else None,
    }


# --------------------------------------------------------------------------
# Cusps, cells, mirror descriptors
# --------------------------------------------------------------------------


def _wall_profile(field: PsiBicubicField, wall_radius_m: float, z_low: float, z_high: float, count: int) -> tuple[float, float]:
    """Minimum |B| along the wall on [z_low, z_high] and its location."""

    best_value = math.inf
    best_z = z_low
    for index in range(count + 1):
        z = z_low + (z_high - z_low) * index / count
        magnitude = math.hypot(*field.field_cylindrical(wall_radius_m, z))
        if magnitude < best_value:
            best_value, best_z = magnitude, z
    return best_value, best_z


def _axis_peak(field: PsiBicubicField, z_low: float, z_high: float, count: int) -> tuple[float, float]:
    best_value = -math.inf
    best_z = z_low
    for index in range(count + 1):
        z = z_low + (z_high - z_low) * index / count
        magnitude = abs(axis_bz(field, z))
        if magnitude > best_value:
            best_value, best_z = magnitude, z
    return best_value, best_z


def build_cells(
    field: PsiBicubicField,
    grid: TracingGrid,
    geometry: ChannelGeometry,
    nulls: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    policy: TopologyPolicy,
    *,
    sweep_axis_bz_peaks_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Wall cusps (separatrix intersections inside the straight dielectric) and the cells they bound."""

    by_null = {null["null_id"]: null for null in nulls}
    wall_hits = sorted(
        (trace for trace in traces if trace["reaches_wall"]),
        key=lambda trace: trace["z_c_m"],
    )
    cusps: list[dict[str, Any]] = []
    other_intersections: list[dict[str, Any]] = []
    z_lo, z_hi = geometry.straight_z_min_m, geometry.straight_z_max_m
    for trace in wall_hits:
        z_c = float(trace["z_c_m"])
        inside = z_lo <= z_c <= z_hi
        if z_c < z_lo:
            zone = "anode_side"
        elif inside:
            zone = "straight_dielectric"
        elif z_c <= geometry.chamber_length_m:
            zone = "divergent_exit"
        else:
            zone = "downstream_stray_field"
        ambiguous = min(abs(z_c - z_lo), abs(z_c - z_hi)) <= policy.boundary_ambiguity_tolerance_m
        nearest_gap = min(geometry.stage_gap_centres_m, key=lambda value: abs(value - z_c))
        nearest_centre = min(geometry.stage_centres_m, key=lambda value: abs(value - z_c))
        row = {
            "cusp_id": "",
            "null_id": trace["null_id"],
            "axis_null_z_m": by_null[trace["null_id"]]["z_m"],
            "z_c_m": z_c,
            "z_c_over_length": z_c / geometry.chamber_length_m,
            "zone": zone,
            "inside_straight_dielectric": inside,
            "boundary_ambiguous": ambiguous,
            "wall_b_t": trace["wall_b_t"],
            "wall_b_r_t": trace["wall_b_r_t"],
            "wall_b_z_t": trace["wall_b_z_t"],
            "wall_normal_component_t": trace["wall_normal_component_t"],
            "angle_to_wall_normal_deg": trace["angle_to_wall_normal_deg"],
            "axis_to_wall_shift_m": z_c - by_null[trace["null_id"]]["z_m"],
            "distance_to_nearest_stage_gap_m": abs(nearest_gap - z_c),
            "distance_to_nearest_stage_centre_m": abs(nearest_centre - z_c),
            "z_c_flux_root_m": trace["z_c_flux_root_m"],
            "flux_root_consistent": trace["flux_root_consistent"],
        }
        (cusps if inside else other_intersections).append(row)
    for index, cusp in enumerate(cusps):
        cusp["cusp_id"] = f"wall-cusp-{index + 1:02d}"
    for index, row in enumerate(other_intersections):
        row["cusp_id"] = f"outside-intersection-{index + 1:02d}"

    # Cells along the straight wall.
    boundaries = [z_lo] + [cusp["z_c_m"] for cusp in cusps] + [z_hi]
    null_z_sorted = sorted(float(null["z_m"]) for null in nulls)
    cells: list[dict[str, Any]] = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        if not cusps:
            kind = "unbounded"
        elif index == 0:
            kind = "anode_partial"
        elif index == len(boundaries) - 2:
            kind = "exit_partial"
        else:
            kind = "interior"
        start_cusp = cusps[index - 1] if index >= 1 and cusps else None
        end_cusp = cusps[index] if index < len(cusps) else None
        wall_b_min, wall_b_min_z = _wall_profile(field, geometry.wall_radius_m, start, end, policy.wall_samples_per_cell)
        cusp_b = [item["wall_b_t"] for item in (start_cusp, end_cusp) if item is not None]
        # Axis interval between the generating nulls; a partial cell uses the neighbouring
        # axis null outside the straight section when one exists in the window, else the
        # straight-section end itself.
        if start_cusp is not None:
            axis_low = start_cusp["axis_null_z_m"]
        elif end_cusp is not None:
            axis_low = _previous_null(null_z_sorted, end_cusp["axis_null_z_m"], start)
        else:
            axis_low = start
        if end_cusp is not None:
            axis_high = end_cusp["axis_null_z_m"]
        elif start_cusp is not None:
            axis_high = _next_null(null_z_sorted, start_cusp["axis_null_z_m"], end)
        else:
            axis_high = end
        axis_low = max(axis_low, float(grid.z_m[0]))
        axis_high = min(axis_high, float(grid.z_m[-1]))
        if axis_high <= axis_low:
            axis_low, axis_high = min(start, end), max(start, end)
        axis_peak, axis_peak_z = _axis_peak(field, axis_low, axis_high, policy.axis_samples_per_interval)
        sweep_peaks_inside = (
            None
            if sweep_axis_bz_peaks_m is None
            else sum(axis_low <= value <= axis_high for value in sweep_axis_bz_peaks_m)
        )
        cells.append(
            {
                "cell_id": f"cell-{index + 1:02d}",
                "kind": kind,
                "z_start_m": start,
                "z_end_m": end,
                "length_m": end - start,
                "length_over_pitch": (end - start) / geometry.stage_pitch_m,
                "start_cusp_id": None if start_cusp is None else start_cusp["cusp_id"],
                "end_cusp_id": None if end_cusp is None else end_cusp["cusp_id"],
                "axis_interval_m": [axis_low, axis_high],
                "wall_b_min_t": wall_b_min,
                "wall_b_min_z_m": wall_b_min_z,
                "cusp_wall_b_min_t": min(cusp_b) if cusp_b else None,
                "cusp_wall_b_max_t": max(cusp_b) if cusp_b else None,
                "wall_mirror_ratio": (min(cusp_b) / wall_b_min) if cusp_b and wall_b_min > 0.0 else None,
                "wall_mirror_ratio_strong_end": (max(cusp_b) / wall_b_min) if cusp_b and wall_b_min > 0.0 else None,
                "axis_bz_peak_t": axis_peak,
                "axis_bz_peak_z_m": axis_peak_z,
                "axis_mirror_ratio": (min(cusp_b) / axis_peak) if cusp_b and axis_peak > 0.0 else None,
                "sweep_axis_bz_peaks_inside": sweep_peaks_inside,
                "stage_centres_inside": sum(start <= value <= end for value in geometry.stage_centres_m),
            }
        )
    return {
        "wall_cusps": cusps,
        "wall_cusp_count": len(cusps),
        "outside_intersections": other_intersections,
        "cells": cells,
        "cell_count": len(cells),
        "interior_cell_count": sum(cell["kind"] == "interior" for cell in cells),
        "four_wall_cusps": len(cusps) == 4,
        "four_cells": len(cells) == 4 and bool(cusps),
        "any_boundary_ambiguous": any(cusp["boundary_ambiguous"] for cusp in cusps) or any(row["boundary_ambiguous"] for row in other_intersections),
    }


def _previous_null(sorted_nulls: Sequence[float], reference: float, fallback: float) -> float:
    candidates = [value for value in sorted_nulls if value < reference - 1.0e-12]
    return max(candidates) if candidates else fallback


def _next_null(sorted_nulls: Sequence[float], reference: float, fallback: float) -> float:
    candidates = [value for value in sorted_nulls if value > reference + 1.0e-12]
    return min(candidates) if candidates else fallback


# --------------------------------------------------------------------------
# Full characterization of one map and cross-resolution stability
# --------------------------------------------------------------------------


def characterize_map(
    grid: TracingGrid,
    geometry: ChannelGeometry,
    policy: TopologyPolicy,
    *,
    source_identity_sha256: str,
    minimum_certificate_tightness_ratio: float,
    keep_paths: bool,
    sweep_axis_bz_peaks_m: Sequence[float] | None = None,
    axis_window_m: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Characterize one regular map under definition v3 (pure function of its inputs)."""

    field = bicubic_field(
        grid,
        source_identity_sha256=source_identity_sha256,
        minimum_certificate_tightness_ratio=minimum_certificate_tightness_ratio,
    )
    bilinear = grid.validated_psi_map()
    null_report = find_axis_nulls(field, grid, geometry, policy, window_m=axis_window_m)
    z_window = (float(grid.z_m[0]), float(grid.z_m[-1]))
    traces = [
        trace_separatrix(field, grid, null, geometry, policy, z_window=z_window, keep_path=keep_paths)
        for null in null_report["nulls"]
    ]
    bilinear_traces = [
        trace_separatrix_bilinear_v4(grid, bilinear, null, geometry, policy, z_window=z_window)
        for null in null_report["nulls"]
    ]
    for trace, comparison in zip(traces, bilinear_traces, strict=True):
        if trace["reaches_wall"] and comparison["reaches_wall"]:
            trace["v4_bilinear_z_c_m"] = comparison["z_c_m"]
            trace["v4_bilinear_difference_m"] = abs(comparison["z_c_m"] - trace["z_c_m"])
        else:
            trace["v4_bilinear_z_c_m"] = comparison.get("z_c_m")
            trace["v4_bilinear_difference_m"] = None
        trace["v4_bilinear_termination"] = comparison["termination"]
    cells = build_cells(field, grid, geometry, null_report["nulls"], traces, policy, sweep_axis_bz_peaks_m=sweep_axis_bz_peaks_m)
    terminations = {}
    for trace in traces:
        terminations[trace["termination"]] = terminations.get(trace["termination"], 0) + 1
    return {
        "grid": {
            "radial_samples": int(len(grid.r_m)),
            "axial_samples": int(len(grid.z_m)),
            "dr_m": grid.dr_m,
            "dz_m": grid.dz_m,
            "r_max_m": float(grid.r_m[-1]),
            "z_min_m": float(grid.z_m[0]),
            "z_max_m": float(grid.z_m[-1]),
            "radial_cells_across_bore": geometry.wall_radius_m / grid.dr_m,
            "full_map_sha256": bilinear.full_map_hash,
            "certificate": field.certificate_tightness.to_dict(),
            "max_b_t": field.max_b_t,
            "interpolation_error": field.reference_error().to_dict(),
        },
        "axis_nulls": null_report,
        "separatrix_traces": traces,
        "trace_terminations": terminations,
        "all_traces_terminate_cleanly": all(trace["termination"] in ("wall", "domain_z") for trace in traces),
        "all_wall_traces_flux_consistent": all(trace["flux_root_consistent"] for trace in traces if trace["reaches_wall"]),
        "topology": cells,
    }


def compare_resolutions(accepted: Mapping[str, Any], refined: Mapping[str, Any], tolerance_m: float) -> dict[str, Any]:
    """Refinement stability: same axis-null count, same wall-reaching separatrices, |dz_c| <= tolerance."""

    a_nulls = [null["z_m"] for null in accepted["axis_nulls"]["nulls"]]
    r_nulls = [null["z_m"] for null in refined["axis_nulls"]["nulls"]]
    a_hits = sorted(trace["z_c_m"] for trace in accepted["separatrix_traces"] if trace["reaches_wall"])
    r_hits = sorted(trace["z_c_m"] for trace in refined["separatrix_traces"] if trace["reaches_wall"])
    null_count_equal = len(a_nulls) == len(r_nulls)
    hits_equal = len(a_hits) == len(r_hits)
    null_shifts = [abs(a - r) for a, r in zip(sorted(a_nulls), sorted(r_nulls))] if null_count_equal else []
    hit_shifts = [abs(a - r) for a, r in zip(a_hits, r_hits)] if hits_equal else []
    max_null_shift = max(null_shifts) if null_shifts else None
    max_hit_shift = max(hit_shifts) if hit_shifts else None
    cusp_count_equal = accepted["topology"]["wall_cusp_count"] == refined["topology"]["wall_cusp_count"]
    stable = bool(
        null_count_equal
        and hits_equal
        and (max_null_shift is None or max_null_shift <= tolerance_m)
        and (max_hit_shift is None or max_hit_shift <= tolerance_m)
    )
    return {
        "tolerance_m": tolerance_m,
        "axis_null_count_equal": null_count_equal,
        "wall_reaching_count_equal": hits_equal,
        "wall_cusp_count_equal": cusp_count_equal,
        "wall_cusp_count_change_is_boundary_classification": bool(stable and not cusp_count_equal),
        "axis_null_shifts_m": null_shifts,
        "wall_intersection_shifts_m": hit_shifts,
        "max_axis_null_shift_m": max_null_shift,
        "max_wall_intersection_shift_m": max_hit_shift,
        "stable": stable,
    }
