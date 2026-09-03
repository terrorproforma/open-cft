"""Material-aware P2 FEM solves and regular-grid sampling for the L1b HEMP confirmation.

Per design the accepted geometry (soft-iron pole pieces mu_r = 4000 between the magnets,
soft-iron return yoke, SmCo-like magnets with recoil mu_r = 1.05 and remanence, BN and Al
at mu_r = 1) is meshed with the fem_reference body-fitted graded mesh and solved with the
independent quadratic FEM reference (``cft_revival.fem_reference``) through two nested
adaptive levels (level 0: graded initial mesh; level 1: Dorfler-marked red refinement with
the 1.3 gradation closure, initial guess prolonged from level 0). Every solve is CPU-only
(numpy CSR Jacobi-PCG, relative true residual 2e-10) and is preceded by the fail-closed
fem_reference allocation preflight against the campaign RAM budget.

The P2 solution is sampled on a regular (r, z) grid over the bore column r <= r_w (the
axis-regular flux psi = r A_phi, B_r = -dA_phi/dz, B_z = A_phi / r + dA_phi/dr; on the axis
B_z = 2 dA_phi/dr) so that the cusp topology search v3.1 definition can be applied verbatim
to the material-aware field (``topology.tracing_grid`` -> ``characterize_map``). Multi-hit
grid nodes on shared element edges take the mean of the element traces (the convention of
the hash-bound orbit v4 adapter). The sampled field is post-scaled by the design's L1a
``source_strength_scale`` so that the L1a and the P2 maps carry the same magnet strength
(the problem is linear; the scaling is exact).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np

from cft_revival.fem_reference import (
    FEMResult,
    ResourceBlockedError,
    ThirdLevelResourcePolicy,
    available_ram_bytes,
    component_dorfler_mark,
    current_process_rss_bytes,
    estimate_indicators,
    graded_mesh_geometry,
    mesh_quality,
    preflight_level_allocation,
    prolong_p2_solution,
    refine_mesh,
    solve,
)
from cft_revival.geometry import AxisymmetricCFTGeometry

from experiments.cusp_topology_search_v3_1.topology import TracingGrid, tracing_grid

ROBIN_BOUNDARIES = ("outer_radial", "z_min", "z_max")


# --------------------------------------------------------------------------
# Resource budget
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RamBudget:
    """Campaign RAM budget: a fixed fraction of the physical RAM free at campaign start."""

    free_at_start_bytes: int
    fraction: float
    maximum_p2_dofs: int

    @property
    def budget_bytes(self) -> int:
        return int(self.free_at_start_bytes * self.fraction)

    def policy(self) -> ThirdLevelResourcePolicy:
        return ThirdLevelResourcePolicy(maximum_p2_dofs=int(self.maximum_p2_dofs), one_design_at_a_time=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_at_start_bytes": int(self.free_at_start_bytes),
            "fraction": float(self.fraction),
            "budget_bytes": self.budget_bytes,
            "maximum_p2_dofs": int(self.maximum_p2_dofs),
        }


def ram_budget(value: Mapping[str, Any], *, free_bytes: int | None = None) -> RamBudget:
    resources = value["p2"]["resources"]
    free = available_ram_bytes() if free_bytes is None else int(free_bytes)
    return RamBudget(free, float(resources["ram_budget_fraction_of_free_at_start"]), int(resources["maximum_p2_dofs"]))


def _robin_edges(mesh: Any) -> int:
    return int(sum(len(mesh.boundary_edges[name]) for name in ROBIN_BOUNDARIES))


def allocation_preflight(mesh: Any, budget: RamBudget, *, phase: str) -> dict[str, Any]:
    """fem_reference fail-closed allocation preflight against the campaign budget (not the whole host)."""

    report = preflight_level_allocation(
        p2_dofs=len(mesh.p2_nodes_rz_m),
        triangles=len(mesh.triangles),
        robin_edges=_robin_edges(mesh),
        third_level=False,
        policy=budget.policy(),
        available_bytes=min(budget.budget_bytes, available_ram_bytes()),
        phase=phase,
    )
    return {key: (int(item) if isinstance(item, (int, np.integer)) and not isinstance(item, bool) else item) for key, item in report.items()}


# --------------------------------------------------------------------------
# Two-level adaptive P2 solve
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class P2Level:
    level: int
    result: FEMResult
    p2_dofs: int
    triangles: int
    robin_edges: int
    quality: dict[str, Any]
    allocation: dict[str, Any]
    solve_seconds: float
    rss_after_solve_bytes: int
    adaptivity: dict[str, Any] | None

    def evidence(self) -> dict[str, Any]:
        diagnostics = self.result.diagnostics
        return {
            "level": self.level,
            "p2_dofs": self.p2_dofs,
            "triangles": self.triangles,
            "robin_edges": self.robin_edges,
            "mesh_sha256": self.result.mesh.sha256,
            "parent_mesh_sha256": self.result.mesh.parent_mesh_sha256,
            "run_sha256": self.result.run_sha256,
            "implementation_sha256": self.result.implementation_sha256,
            "solver_controls": {key: value for key, value in self.result.solver_controls},
            "converged": bool(diagnostics.converged),
            "iterations": int(diagnostics.iterations),
            "relative_true_residual_l2": float(diagnostics.relative_true_residual_l2),
            "final_true_residual_l2": float(diagnostics.final_true_residual_l2),
            "energy_action_relative": float(diagnostics.energy_action_relative),
            "backend": diagnostics.backend,
            "assembly_seconds": float(diagnostics.assembly_seconds),
            "pcg_seconds": float(diagnostics.solve_seconds),
            "solve_wall_seconds": float(self.solve_seconds),
            "solver_working_set_bytes": int(diagnostics.peak_working_set_bytes),
            "rss_after_solve_bytes": int(self.rss_after_solve_bytes),
            "mesh_quality": dict(self.quality),
            "allocation_preflight": dict(self.allocation),
            "adaptivity": self.adaptivity,
        }


@dataclass(frozen=True)
class P2Solution:
    problem_id: str
    domain: dict[str, float]
    geometry_sha256: str
    magnetics_sha256: str
    regions: list[dict[str, Any]]
    levels: tuple[P2Level, ...]
    stage_windows: tuple[tuple[str, float, float, float], ...]
    peak_rss_bytes: int
    total_seconds: float

    @property
    def coarse(self) -> FEMResult:
        return self.levels[0].result

    @property
    def accepted(self) -> FEMResult:
        return self.levels[-1].result

    def evidence(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "domain": dict(self.domain),
            "geometry_sha256": self.geometry_sha256,
            "magnetics_sha256": self.magnetics_sha256,
            "regions": list(self.regions),
            "stage_windows": [list(window) for window in self.stage_windows],
            "levels": [level.evidence() for level in self.levels],
            "level_count": len(self.levels),
            "all_levels_converged": all(level.result.diagnostics.converged for level in self.levels),
            "peak_rss_bytes": int(self.peak_rss_bytes),
            "total_seconds": float(self.total_seconds),
        }


def sliver_report(mesh: Any, *, threshold_deg: float = 10.0) -> dict[str, Any]:
    """Count the elements below the qualification's 10 deg angle and name their regions (disclosure)."""

    from cft_revival.fem_reference.mesh import element_minimum_angles

    angles = element_minimum_angles(mesh)
    below = np.flatnonzero(angles < threshold_deg)
    regions: dict[str, int] = {}
    for element in below:
        region = str(mesh.triangle_region_ids[int(element)])
        regions[region] = regions.get(region, 0) + 1
    return {
        "threshold_deg": float(threshold_deg),
        "minimum_angle_deg": float(np.min(angles)) if len(angles) else None,
        "elements_below_threshold": int(len(below)),
        "element_count": int(len(angles)),
        "fraction_below_threshold": (float(len(below)) / float(len(angles))) if len(angles) else None,
        "regions_below_threshold": dict(sorted(regions.items())),
    }


def mesh_preflight_for_geometry(geometry: AxisymmetricCFTGeometry, value: Mapping[str, Any], budget: RamBudget) -> dict[str, Any]:
    """Level-0 mesh only (no solve): angle gate, sliver disclosure, DOF cap for level 1 (whole-set preflight)."""

    mesh_declaration = value["p2"]["mesh"]
    problem, mesh = graded_mesh_geometry(
        geometry,
        bore_elements=int(mesh_declaration["bore_elements"]),
        feature_elements=int(mesh_declaration["feature_elements"]),
        padding_factor=float(mesh_declaration["padding_factor"]),
    )
    quality = mesh_quality(mesh)
    gate = float(mesh_declaration["reject_below_angle_deg"])
    upper_bound = 4 * int(len(mesh.p2_nodes_rz_m))
    passes_angle = float(quality["minimum_angle_deg"]) >= gate
    fits_cap = upper_bound <= budget.maximum_p2_dofs
    try:
        allocation_passed = bool(allocation_preflight(mesh, budget, phase="preflight-level-0")["passed"])
    except ResourceBlockedError:
        allocation_passed = False
    return {
        "level0_p2_dofs": int(len(mesh.p2_nodes_rz_m)),
        "level0_triangles": int(len(mesh.triangles)),
        "minimum_angle_deg": float(quality["minimum_angle_deg"]),
        "reject_below_angle_deg": gate,
        "passes_angle_gate": bool(passes_angle),
        "level1_red_closure_p2_dof_upper_bound": upper_bound,
        "fits_dof_cap": bool(fits_cap),
        "level0_allocation_passed": allocation_passed,
        "sliver": sliver_report(mesh),
        "domain": problem.domain.to_dict(),
        "passed": bool(passes_angle and fits_cap and allocation_passed),
    }


def stage_windows_for(geometry: AxisymmetricCFTGeometry) -> tuple[tuple[str, float, float, float], ...]:
    return tuple(
        (f"stage-{index + 1}", float(geometry.chamber.outer_radius_m), float(stage.z_min_m), float(stage.z_max_m))
        for index, stage in enumerate(geometry.stages)
    )


def solve_two_level(geometry: AxisymmetricCFTGeometry, value: Mapping[str, Any], budget: RamBudget) -> P2Solution:
    """Level-0 graded solve + one Dorfler/red adaptive level (frozen fem_reference controls)."""

    p2 = value["p2"]
    mesh_declaration = p2["mesh"]
    solver_declaration = p2["solver"]
    adaptive = p2["adaptivity"]
    level_count = int(adaptive["levels"])
    if level_count < 2:
        raise ValueError("the confirmation requires at least two nested levels (coarse + accepted)")
    started = time.perf_counter()
    problem, mesh = graded_mesh_geometry(
        geometry,
        bore_elements=int(mesh_declaration["bore_elements"]),
        feature_elements=int(mesh_declaration["feature_elements"]),
        padding_factor=float(mesh_declaration["padding_factor"]),
    )
    windows = stage_windows_for(geometry)
    levels: list[P2Level] = []
    previous: FEMResult | None = None
    peak_rss = current_process_rss_bytes()
    for level in range(level_count):
        quality = mesh_quality(mesh)
        slivers = sliver_report(mesh)
        if float(quality["minimum_angle_deg"]) < float(mesh_declaration["reject_below_angle_deg"]):
            raise ValueError(f"level {level} mesh violates the minimum-angle rejection gate ({quality['minimum_angle_deg']:.3f} deg < {mesh_declaration['reject_below_angle_deg']} deg)")
        allocation = allocation_preflight(mesh, budget, phase=f"level-{level}")
        solve_started = time.perf_counter()
        result = solve(
            problem,
            mesh,
            relative_tolerance=float(solver_declaration["relative_tolerance"]),
            absolute_tolerance=float(solver_declaration["absolute_tolerance"]),
            max_iterations=int(solver_declaration["max_iterations"]),
            initial_a_phi_dofs_t_m=None if previous is None else prolong_p2_solution(previous, mesh),
            required_available_ram_bytes=int(allocation["effective_required_free_ram_bytes"]),
        )
        solve_seconds = time.perf_counter() - solve_started
        rss = current_process_rss_bytes()
        peak_rss = max(peak_rss, rss)
        adaptivity: dict[str, Any] | None = None
        next_mesh = None
        if level < level_count - 1:
            indicators = estimate_indicators(result, windows)
            marked = component_dorfler_mark(indicators, float(adaptive["dorfler_theta"]))
            upper_bound = 4 * len(mesh.p2_nodes_rz_m)
            if upper_bound > budget.maximum_p2_dofs:
                raise ResourceBlockedError(
                    f"NOT_EVALUATED: level {level + 1} red-closure upper bound {upper_bound} P2 DOFs exceeds the "
                    f"campaign cap {budget.maximum_p2_dofs}"
                )
            next_mesh = refine_mesh(
                mesh,
                problem.domain,
                marked,
                reject_below_angle_deg=float(mesh_declaration["reject_below_angle_deg"]),
                maximum_adjacent_size_growth=float(adaptive["maximum_adjacent_size_growth"]),
            )
            child_counts = np.bincount(next_mesh.element_parent_ids, minlength=len(mesh.triangles))
            total = float(np.sum(indicators.total_squared))
            adaptivity = {
                "theta": float(adaptive["dorfler_theta"]),
                "dorfler_marked_elements": len(marked),
                "element_count": len(mesh.triangles),
                "marked_indicator_fraction": float(np.sum(indicators.total_squared[marked]) / total) if total > 0.0 else None,
                "residual_indicator_sum": float(np.sum(indicators.residual_squared)),
                "flux_jump_indicator_sum": float(np.sum(indicators.flux_jump_squared)),
                "qoi_proxy_indicator_sum": float(np.sum(indicators.qoi_proxy_squared)),
                "next_level_red_closure_p2_dof_upper_bound": upper_bound,
                "refined_parents_after_gradation_closure": int(np.count_nonzero(child_counts > 1)),
                "next_level_p2_dofs": len(next_mesh.p2_nodes_rz_m),
                "next_level_triangles": len(next_mesh.triangles),
            }
            peak_rss = max(peak_rss, current_process_rss_bytes())
        levels.append(
            P2Level(
                level,
                result,
                len(mesh.p2_nodes_rz_m),
                len(mesh.triangles),
                _robin_edges(mesh),
                {**{key: (float(item) if isinstance(item, (float, np.floating)) else int(item)) for key, item in quality.items()}, "sliver": slivers},
                allocation,
                solve_seconds,
                rss,
                adaptivity,
            )
        )
        previous = result
        if next_mesh is not None:
            mesh = next_mesh
    regions = [
        {
            "region_id": region.region_id,
            "material_id": region.material_id,
            "reluctivity_per_m_h": float(region.reluctivity_per_m_h),
            "relative_permeability": float(1.0 / (region.reluctivity_per_m_h * 4.0e-7 * math.pi)),
            "remanence_z_t": float(region.remanence_z_t),
        }
        for region in problem.regions
    ]
    return P2Solution(
        problem.problem_id,
        problem.domain.to_dict(),
        problem.geometry_sha256,
        problem.magnetics_sha256,
        regions,
        tuple(levels),
        windows,
        peak_rss,
        time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# Regular-grid sampling of a P2 solution (vectorised per element)
# --------------------------------------------------------------------------


def _p2_values_and_gradients(l0: np.ndarray, l1: np.ndarray, l2: np.ndarray, grad_lambda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P2 shape values (n, 6) and gradients (n, 6, 2) for arrays of barycentric coordinates."""

    values = np.stack(
        (
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ),
        axis=1,
    )
    g0, g1, g2 = grad_lambda
    gradients = np.stack(
        (
            (4.0 * l0 - 1.0)[:, None] * g0,
            (4.0 * l1 - 1.0)[:, None] * g1,
            (4.0 * l2 - 1.0)[:, None] * g2,
            4.0 * (l0[:, None] * g1 + l1[:, None] * g0),
            4.0 * (l1[:, None] * g2 + l2[:, None] * g1),
            4.0 * (l2[:, None] * g0 + l0[:, None] * g2),
        ),
        axis=1,
    )
    return values, gradients


@dataclass(frozen=True)
class SampledField:
    r_m: np.ndarray
    z_m: np.ndarray
    psi_wb: np.ndarray
    b_r_t: np.ndarray
    b_z_t: np.ndarray
    hit_counts: np.ndarray
    scale: float

    def report(self) -> dict[str, Any]:
        return {
            "radial_samples": len(self.r_m),
            "axial_samples": len(self.z_m),
            "r_max_m": float(self.r_m[-1]),
            "z_min_m": float(self.z_m[0]),
            "z_max_m": float(self.z_m[-1]),
            "dr_m": float(self.r_m[1] - self.r_m[0]),
            "dz_m": float(self.z_m[1] - self.z_m[0]),
            "multi_hit_node_count": int(np.count_nonzero(self.hit_counts > 1)),
            "max_hits_per_node": int(np.max(self.hit_counts)),
            "source_strength_scale_applied": float(self.scale),
            "max_b_t": float(np.max(np.hypot(self.b_r_t, self.b_z_t))),
        }


def sample_regular_grid(result: FEMResult, r_nodes: Sequence[float], z_nodes: Sequence[float], *, scale: float = 1.0) -> SampledField:
    """Sample psi, B_r, B_z of a P2 solution on a regular grid (mean of element traces at shared edges)."""

    r = np.asarray(r_nodes, dtype=np.float64)
    z = np.asarray(z_nodes, dtype=np.float64)
    if r.ndim != 1 or z.ndim != 1 or len(r) < 2 or len(z) < 2 or r[0] != 0.0:
        raise ValueError("sampling grid must be 1-D, start on the axis and have at least two nodes per direction")
    if np.any(np.diff(r) <= 0.0) or np.any(np.diff(z) <= 0.0):
        raise ValueError("sampling grid nodes must be strictly increasing")
    mesh = result.mesh
    vertices = mesh.vertices_rz_m
    points = vertices[mesh.triangles]  # (n, 3, 2)
    r_min = np.min(points[:, :, 0], axis=1)
    r_max = np.max(points[:, :, 0], axis=1)
    z_min = np.min(points[:, :, 1], axis=1)
    z_max = np.max(points[:, :, 1], axis=1)
    tolerance = 2.0e-11
    selected = np.flatnonzero((r_max >= r[0] - tolerance) & (r_min <= r[-1] + tolerance) & (z_max >= z[0] - tolerance) & (z_min <= z[-1] + tolerance))
    a_sum = np.zeros((len(r), len(z)), dtype=np.float64)
    ar_sum = np.zeros_like(a_sum)
    az_sum = np.zeros_like(a_sum)
    counts = np.zeros(a_sum.shape, dtype=np.int64)
    coefficients_all = result.a_phi_dofs_t_m[mesh.element_dofs]  # (n, 6)
    for element in selected:
        i_lo = int(np.searchsorted(r, r_min[element] - tolerance, side="left"))
        i_hi = int(np.searchsorted(r, r_max[element] + tolerance, side="right"))
        j_lo = int(np.searchsorted(z, z_min[element] - tolerance, side="left"))
        j_hi = int(np.searchsorted(z, z_max[element] + tolerance, side="right"))
        if i_hi <= i_lo or j_hi <= j_lo:
            continue
        p0, p1, p2 = points[element]
        jacobian = np.column_stack((p1 - p0, p2 - p0))
        inverse = np.linalg.inv(jacobian)
        rr, zz = np.meshgrid(r[i_lo:i_hi], z[j_lo:j_hi], indexing="ij")
        delta = np.stack((rr.ravel() - p0[0], zz.ravel() - p0[1]), axis=0)
        local = inverse @ delta  # (2, m): lambda1, lambda2
        l1 = local[0]
        l2 = local[1]
        l0 = 1.0 - l1 - l2
        inside = (l0 >= -tolerance) & (l1 >= -tolerance) & (l2 >= -tolerance)
        if not np.any(inside):
            continue
        grad_lambda = np.empty((3, 2), dtype=np.float64)
        grad_lambda[1] = inverse[0]
        grad_lambda[2] = inverse[1]
        grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
        values, gradients = _p2_values_and_gradients(l0[inside], l1[inside], l2[inside], grad_lambda)
        coefficients = coefficients_all[element]
        a_phi = values @ coefficients
        gradient = np.einsum("k,nkd->nd", coefficients, gradients)
        ii = np.repeat(np.arange(i_lo, i_hi), j_hi - j_lo)[inside]
        jj = np.tile(np.arange(j_lo, j_hi), i_hi - i_lo)[inside]
        np.add.at(a_sum, (ii, jj), a_phi)
        np.add.at(ar_sum, (ii, jj), gradient[:, 0])
        np.add.at(az_sum, (ii, jj), gradient[:, 1])
        np.add.at(counts, (ii, jj), 1)
    if np.any(counts == 0):
        missing = np.argwhere(counts == 0)
        raise ValueError(f"{len(missing)} sampling nodes lie outside the FEM mesh (first: r={r[missing[0][0]]}, z={z[missing[0][1]]})")
    a_phi = a_sum / counts
    a_r = ar_sum / counts
    a_z = az_sum / counts
    radial = r[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        b_z = np.where(radial > 0.0, a_phi / np.where(radial > 0.0, radial, 1.0) + a_r, 2.0 * a_r)
    b_r = -a_z
    psi = radial * a_phi
    psi[0, :] = 0.0
    b_r[0, :] = 0.0
    return SampledField(r, z, scale * psi, scale * b_r, scale * b_z, counts, float(scale))


def sampling_nodes(value: Mapping[str, Any], wall_radius_m: float, domain: Mapping[str, float], *, refinement: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Regular sampling grid: r = 0..r_w in ``radial_intervals`` steps; z inset from the P2 domain by one step."""

    sampling = value["p2"]["sampling"]
    radial_intervals = int(sampling["radial_intervals"]) * int(refinement)
    h = float(wall_radius_m) / radial_intervals
    inset = float(sampling["axial_inset_steps"]) * (float(wall_radius_m) / int(sampling["radial_intervals"]))
    z_lo = float(domain["z_min_m"]) + inset
    z_hi = float(domain["z_max_m"]) - inset
    axial_intervals = int(math.floor((z_hi - z_lo) / h))
    if axial_intervals < 4:
        raise ValueError("sampling window is too short for a tracing grid")
    r_nodes = np.linspace(0.0, float(wall_radius_m), radial_intervals + 1)
    z_nodes = np.linspace(z_lo, z_hi, axial_intervals + 1)
    return r_nodes, z_nodes


def sampled_tracing_grid(sampled: SampledField, wall_radius_m: float) -> TracingGrid:
    return tracing_grid(sampled.r_m, sampled.z_m, sampled.psi_wb, sampled.b_r_t, sampled.b_z_t, wall_radius_m)


def field_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
