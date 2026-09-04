"""Per-design static B fields for the PIC design mini-sweep: production, hash binding and the PIC node map.

What the PIC model needs (``cft_revival.pic2d.fields``): a bilinear node field ``(B_r, B_z)`` on the PIC grid
over the plasma nodes of the channel (v1.x) or of the L-shaped channel + plume box (v2.0 / v2.1), evaluated
DIRECTLY from a hash-bound quadratic ``A_phi`` FEM solution whose mesh contains the whole PIC box (the v2.1
extension binds the ``domain-padding-1.5`` solve of the reference design through
``spec/pic2d/p2-field-plume-extension-v2.json``).

The reference design keeps its existing artifacts (authority level-1 checkpoint for the channel, padding-1.5
checkpoint for any plume box up to 60 x 48 mm).  Every other sweep design needs a NEW material-aware P2 solve:
the L1b v1.1 solutions exist only as bore-column samples (33 x ~230 nodes, r <= r_w) of a padding-0.5 domain
that ends 5-10 mm behind the exit - they cannot serve a plume box and were never kept as element data.

Production (``produce_field``, CPU, one design at a time, BLAS single-threaded by the caller):

1. rebuild the accepted geometry with identity proof (``designs.build_design``);
2. fem_reference graded body-fitted level-0 mesh (bore r_w / 8, features / 4 as the qualification's domain
   studies) at the smallest padding factor of a fixed ladder whose FEM box covers ``z <= L_channel + 24 mm`` and
   ``r <= 12 mm`` plus a 0.75 mm truncation margin (the same margin the v2.1 supported box keeps);
   whole-set mesh preflight (angle gate 5 deg with the L1b sliver disclosure) BEFORE any solve;
3. CPU Jacobi-PCG solve, relative true residual 2e-10 (the qualification controls), RAM-guarded;
4. checkpoint bundle written with ``fem_reference.write_checkpoint_bundle`` (the format ``BoundP2Evaluator``
   reads: canonical JSON + npz sidecar, payload / mesh / run / sidecar hashes);
5. gates on the bore column: the P2 solution is sampled on the L1b lattice (32 radial intervals), post-scaled
   by the design's L1a ``source_strength_scale`` (the catalogue's magnet strength; the FEM is linear), and the
   cusp topology search v3.1 definition is applied verbatim -> wall-cusp count equal to the catalogue, every
   cusp within ``max(r_w / 8, L1a dz)`` of its L1a position (the L1b v1.1 rule), Koch rho under iron reported;
   where an L1b v1.1 accepted (level-1) map exists the node-wise |dB| against it is recorded and gated at the
   authority's 0.02 T component bound; the reference design's checkpoint is compared with the qualified level-1
   channel bicubic exactly as ``p2_plume_field_map`` does;
6. ``fields/<design_id>/binding.json``: the hash-bound map declaration the PIC reads (checkpoint path + five
   hashes, bounding / supported box, region ids, scale, gates, identity of the design).

The PIC node map (``design_field_map``) evaluates the bound checkpoint at every plasma node of the PIC grid
(plasma-side limit on the front face, zero inside the body), multiplies by the scale, and records the binding's
hashes in the provenance so ``field_sha256`` is design- and checkpoint-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.fem_reference import (
    ResourceBlockedError,
    ThirdLevelResourcePolicy,
    artifact_from_result,
    available_ram_bytes,
    checkpoint_metadata_summary,
    current_process_rss_bytes,
    graded_mesh_geometry,
    mesh_quality,
    preflight_level_allocation,
    qois,
    solve,
    write_checkpoint_bundle,
)
from cft_revival.pic2d.fields import (
    DEFAULT_PLUME_EXTENSION_PATH,
    PLUME_EXTENSION_V2_PATH,
    MagneticFieldMap,
    build_p2_psi_field,
    p2_plume_field_map,
    sample_field_map,
)
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import PIC2DValidationError
from cft_revival.pic2d.p2_field import BoundP2Evaluator, file_sha256

from . import designs as design_module
from .designs import REPOSITORY, BuiltDesign, PicMapping

BINDING_SCHEMA = "cft.pic2d.design-mini-sweep.field-binding.v1"
CHECKPOINT_SCHEMA = "cft_revival.fem_reference.checkpoint/1.2.0"
CLASSIFICATION = "independent_numerical_reference_not_hardware_validation"
PADDING_LADDER = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
BORE_ELEMENTS = 8
FEATURE_ELEMENTS = 4
REJECT_BELOW_ANGLE_DEG = 5.0          # L1b v1.1 disclosed gate (the qualification's 10 deg lost 028 / 048 to mesher slivers)
SLIVER_DISCLOSURE_DEG = 10.0
RELATIVE_TOLERANCE = 2.0e-10
ABSOLUTE_TOLERANCE = 1.0e-12
MAX_ITERATIONS = 16000
MAXIMUM_P2_DOFS = 700_000
RAM_BUDGET_BYTES = 4 * 1024**3        # the task's host ceiling (peak RSS <= 4 GB); the fem_reference preflight enforces it
COVER_Z_BEHIND_EXIT_M = 0.024          # the v2.1 plume length
COVER_R_M = 0.012                      # the reference's plume radius (return yoke); every sweep yoke is <= 11.7 mm
TRUNCATION_MARGIN_M = 0.00075          # as the v2.1 supported box: keep the PIC boundary inside the FEM's outermost elements
CHANNEL_AGREEMENT_MAX_ABS_T = 0.02     # the authority's component bound (p2-field-authority-v1: maximum_b_component_absolute_error_t)
PLASMA_REGIONS = ("injector-zone", "channel-straight", "channel-divergent-exit", "dielectric-straight", "dielectric-divergent-exit", "ambient-background")
SOLID_REGION_PREFIXES = ("anode", "magnet-", "pole-", "shield-shell", "return-yoke")
SAMPLING_RADIAL_INTERVALS = 32         # the L1b v1.1 bore lattice
SAMPLING_AXIAL_INSET_STEPS = 1.0
TOPOLOGY_KEYS = ("wall_cusps", "cells", "axis_nulls", "rho", "min_rho_conservative", "min_rho_conservative_interior", "interior_rule",
                 "hemp_like_all_cusps", "all_traces_terminate_cleanly", "sampling")


def fields_dir(design_id: str) -> Path:
    return design_module.FIELDS_DIR / design_id


def binding_path(design_id: str) -> Path:
    return fields_dir(design_id) / "binding.json"


# --------------------------------------------------------------------------
# Padding and mesh preflight
# --------------------------------------------------------------------------


def _domain_for(built: BuiltDesign, padding_factor: float) -> dict[str, float]:
    from cft_revival.fem_reference import design_domain

    domain = design_domain(built.geometry, padding_factor=padding_factor)
    return domain.to_dict()


def coverage_requirement(built: BuiltDesign) -> dict[str, float]:
    length = float(built.geometry.chamber.length_m)
    return {"z_max_m": length + COVER_Z_BEHIND_EXIT_M + TRUNCATION_MARGIN_M, "r_max_m": COVER_R_M + TRUNCATION_MARGIN_M}


def padding_factor_for(built: BuiltDesign) -> tuple[float, dict[str, Any]]:
    """Smallest ladder padding whose FEM box covers the v2.1-sized plume box (z <= L + 24 mm, r <= 12 mm) with margin."""

    need = coverage_requirement(built)
    tried = []
    for factor in PADDING_LADDER:
        domain = _domain_for(built, factor)
        ok = domain["z_max_m"] >= need["z_max_m"] and domain["r_max_m"] >= need["r_max_m"]
        tried.append({"padding_factor": factor, "domain": domain, "covers": ok})
        if ok:
            return factor, {"required": need, "ladder": tried}
    raise ValueError(f"{built.design_id}: no ladder padding covers {need}")


def mesh_preflight(built: BuiltDesign, padding_factor: float) -> dict[str, Any]:
    """Level-0 mesh only (no solve): angle gate, sliver disclosure, DOF cap and RAM preflight (the whole-set preflight)."""

    from experiments.l1b_hemp_confirmation_v1_1.p2_fields import sliver_report

    started = time.perf_counter()
    problem, mesh = graded_mesh_geometry(built.geometry, bore_elements=BORE_ELEMENTS, feature_elements=FEATURE_ELEMENTS, padding_factor=padding_factor)
    quality = mesh_quality(mesh)
    robin = int(sum(len(mesh.boundary_edges[name]) for name in ("outer_radial", "z_min", "z_max")))
    p2_dofs = int(len(mesh.p2_nodes_rz_m))
    allocation: dict[str, Any]
    try:
        allocation = preflight_level_allocation(
            p2_dofs=p2_dofs, triangles=int(len(mesh.triangles)), robin_edges=robin, third_level=False,
            policy=ThirdLevelResourcePolicy(maximum_p2_dofs=MAXIMUM_P2_DOFS, one_design_at_a_time=True),
            available_bytes=min(RAM_BUDGET_BYTES, available_ram_bytes()), phase="design-mini-sweep-level-0",
        )
        allocation_passed = True
    except ResourceBlockedError as error:
        allocation = {"passed": False, "reason": str(error)}
        allocation_passed = False
    minimum_angle = float(quality["minimum_angle_deg"])
    report = {
        "design_id": built.design_id,
        "padding_factor": padding_factor,
        "domain": problem.domain.to_dict(),
        "bore_elements": BORE_ELEMENTS,
        "feature_elements": FEATURE_ELEMENTS,
        "level0_p2_dofs": p2_dofs,
        "level0_triangles": int(len(mesh.triangles)),
        "robin_edges": robin,
        "minimum_angle_deg": minimum_angle,
        "reject_below_angle_deg": REJECT_BELOW_ANGLE_DEG,
        "passes_angle_gate": bool(minimum_angle >= REJECT_BELOW_ANGLE_DEG),
        "sliver": sliver_report(mesh, threshold_deg=SLIVER_DISCLOSURE_DEG),
        "fits_dof_cap": bool(p2_dofs <= MAXIMUM_P2_DOFS),
        "allocation_preflight": {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else v) for k, v in allocation.items()},
        "mesh_seconds": time.perf_counter() - started,
    }
    report["passed"] = bool(report["passes_angle_gate"] and report["fits_dof_cap"] and allocation_passed)
    return report


# --------------------------------------------------------------------------
# Bore-column sampling and topology gates
# --------------------------------------------------------------------------


def _stage_windows(built: BuiltDesign) -> tuple[tuple[str, float, float, float], ...]:
    return tuple((f"stage-{i + 1}", float(built.geometry.chamber.outer_radius_m), float(s.z_min_m), float(s.z_max_m)) for i, s in enumerate(built.geometry.stages))


def topology_channel_geometry(built: BuiltDesign):
    from experiments.cusp_topology_search_v3_1.topology import ChannelGeometry as TopologyGeometry

    chamber = built.geometry.chamber
    return TopologyGeometry(
        wall_radius_m=float(chamber.outer_radius_m), straight_z_min_m=0.0, straight_z_max_m=float(chamber.exit_start_m),
        chamber_length_m=float(chamber.length_m), stage_pitch_m=float(built.geometry.stages[0].pitch_m),
        stage_centres_m=tuple(float(s.center_z_m) for s in built.geometry.stages), injector_length_m=float(chamber.injector_length_m),
    )


def sealed_l1a_record(design_id: str) -> dict[str, Any]:
    """The sealed catalogue record (sweep-v3 results for L1a designs; topology v3.1 results for the reference)."""

    entry = design_module.catalogue_entry(design_id)
    if design_id == design_module.REFERENCE_DESIGN_ID:
        root = design_module.V31_CATALOGUE_PATH.parents[1]
    else:
        root = design_module.V3_CATALOGUE_PATH.parents[1]
    return strict_json_file(root / entry["record_path"])


def _axis_window(record: Mapping[str, Any], built: BuiltDesign) -> tuple[float, float]:
    if record.get("axis_window_m") is not None:
        return (float(record["axis_window_m"][0]), float(record["axis_window_m"][1]))
    pitch = float(built.geometry.stages[0].pitch_m)
    return (-pitch, float(built.geometry.chamber.length_m) + pitch)


def _l1a_dz(record: Mapping[str, Any]) -> float | None:
    grid = (record.get("accepted") or {}).get("grid") or {}
    return float(grid["dz_m"]) if "dz_m" in grid else None


def characterize_bore(result, built: BuiltDesign, *, source_identity_sha256: str) -> dict[str, Any]:
    """v3.1 topology + sweep-v3 descriptors of the (scaled) P2 bore field, compared with the catalogue cusps."""

    from experiments.cusp_topology_search_v3_1 import topology as topology_module
    from experiments.l1a_geometry_sweep_v3 import descriptors as descriptor_module
    from experiments.l1b_hemp_confirmation_v1_1 import experiment as l1b_experiment
    from experiments.l1b_hemp_confirmation_v1_1 import p2_fields as l1b_fields

    l1b_protocol = l1b_experiment.protocol()
    policy = l1b_experiment.policy_from(l1b_protocol)
    tightness = float(l1b_protocol["definition_v3_import"]["minimum_certificate_dense_to_bound_ratio"])
    record = sealed_l1a_record(built.design_id)
    geometry = topology_channel_geometry(built)
    value = {"p2": {"sampling": {"radial_intervals": SAMPLING_RADIAL_INTERVALS, "axial_inset_steps": SAMPLING_AXIAL_INSET_STEPS}}}
    window = _axis_window(record, built)
    # sample only the axial span the v3.1 search needs (the padded domain is much longer than the channel)
    domain = result.problem.domain.to_dict()
    pitch = float(built.geometry.stages[0].pitch_m)
    span = {"z_min_m": max(domain["z_min_m"], window[0] - 3.0 * pitch), "z_max_m": min(domain["z_max_m"], window[1] + 3.0 * pitch)}
    r_nodes, z_nodes = l1b_fields.sampling_nodes(value, geometry.wall_radius_m, span, refinement=1)
    sampled = l1b_fields.sample_regular_grid(result, r_nodes, z_nodes, scale=built.source_strength_scale)
    grid = l1b_fields.sampled_tracing_grid(sampled, geometry.wall_radius_m)
    if not l1b_experiment.window_contained(grid, window, policy):
        raise ValueError(f"{built.design_id}: the sampled bore column does not contain the catalogue axis window with the v3.1 margin")
    characterization = topology_module.characterize_map(
        grid, geometry, policy, source_identity_sha256=source_identity_sha256, minimum_certificate_tightness_ratio=tightness,
        keep_paths=False, sweep_axis_bz_peaks_m=None, axis_window_m=window,
    )
    descriptors = descriptor_module.design_descriptors(
        grid, geometry, characterization, policy, source_identity_sha256=source_identity_sha256,
        minimum_certificate_tightness_ratio=tightness, stage_count=len(built.geometry.stages), with_profiles=False,
    )
    entry = design_module.catalogue_entry(built.design_id)
    catalogue_all = [float(c["z_c_m"]) for c in entry["wall_cusps"]]
    p2_all = [float(c["z_c_m"]) for c in characterization["topology"]["wall_cusps"]]
    # L1b v1.1 GATE (b) semantics: a cusp within the v3.1 boundary-ambiguity tolerance of either end of the straight
    # dielectric is a boundary classification (the iron moves the end nulls by 1-2 mm), not a cell boundary
    boundary_tolerance = float(l1b_protocol["definition_v3_import"]["numerical_parameters"]["boundary_ambiguity_tolerance_m"])
    lo, hi = geometry.straight_z_min_m + boundary_tolerance, geometry.straight_z_max_m - boundary_tolerance
    catalogue_cusps = [z for z in catalogue_all if lo <= z <= hi]
    p2_cusps = [z for z in p2_all if lo <= z <= hi]
    boundary_excluded = {"catalogue": [z for z in catalogue_all if z not in catalogue_cusps], "p2": [z for z in p2_all if z not in p2_cusps]}
    dz_l1a = _l1a_dz(record)
    tolerance = max(geometry.wall_radius_m / BORE_ELEMENTS, dz_l1a if dz_l1a is not None else 0.0)
    matches = []
    for z_ref in catalogue_cusps:
        nearest = min(p2_cusps, key=lambda z: abs(z - z_ref)) if p2_cusps else None
        matches.append({"catalogue_z_c_m": z_ref, "p2_z_c_m": nearest, "shift_m": None if nearest is None else abs(nearest - z_ref)})
    shifts = [m["shift_m"] for m in matches if m["shift_m"] is not None]
    comparison = {
        "catalogue_field": "P2 qualified level-1" if built.design_id == design_module.REFERENCE_DESIGN_ID else "L1a linear-vacuum equivalent-current",
        "catalogue_wall_cusp_count": len(catalogue_all), "p2_wall_cusp_count": len(p2_all),
        "count_equal_strict": len(catalogue_all) == len(p2_all),
        "boundary_ambiguity_tolerance_m": boundary_tolerance, "boundary_cusps_excluded_z_m": boundary_excluded,
        "catalogue_interior_cusp_count": len(catalogue_cusps), "p2_interior_cusp_count": len(p2_cusps),
        "count_equal": len(catalogue_cusps) == len(p2_cusps),
        "count_rule": "boundary-tolerant (L1b v1.1 GATE (b)): cusps within boundary_ambiguity_tolerance_m of the straight-section ends are not counted; the strict count is recorded",
        "matches": matches,
        "max_shift_m": max(shifts) if shifts else None, "tolerance_m": tolerance,
        "tolerance_rule": "max(r_w / 8, L1a dz) - the L1b v1.1 rule (one level-0 bore element, never below the L1a axial step)",
        "positions_within_tolerance": bool(shifts) and len(shifts) == len(catalogue_cusps) and max(shifts) <= tolerance,
        "catalogue_min_rho_conservative": design_module.design_summary(built.design_id)["min_rho_conservative"],
        "p2_min_rho_conservative": descriptors.get("min_rho_conservative"),
        "p2_hemp_like_all_cusps": bool(descriptors.get("hemp_like_all_cusps", False)),
        "axis_window_m": [window[0], window[1]],
    }
    l1b = design_module.l1b_record(built.design_id)
    if l1b is not None:
        l1b_cusps = [float(m["p2_z_c_m"]) for m in l1b["comparison"]["matched_cusps"]]
        comparison["l1b_v1_1"] = {
            "p2_level1_cusps_z_m": l1b_cusps,
            "max_shift_vs_l1b_m": max(min(abs(z - zc) for z in p2_cusps) for zc in l1b_cusps) if p2_cusps and l1b_cusps else None,
            "p2_level1_min_rho_conservative": l1b["comparison"]["p2_min_rho_conservative"],
        }
    comparison["passed"] = bool(comparison["count_equal"] and comparison["positions_within_tolerance"] and characterization["all_traces_terminate_cleanly"])
    interior_rho = [row["rho_conservative"] for row in descriptors["cusps"] if lo <= float(row["z_c_m"]) <= hi and row["rho_conservative"] is not None]
    comparison["p2_min_rho_conservative_interior"] = min(interior_rho) if interior_rho else None
    return {
        "min_rho_conservative_interior": min(interior_rho) if interior_rho else None,
        "interior_rule": f"cusps farther than {boundary_tolerance} m from the straight-section ends (boundary cusps of the moved end nulls are excluded)",
        "sampling": sampled.report(),
        "wall_cusps": [{k: c[k] for k in ("cusp_id", "z_c_m", "wall_b_t", "angle_to_wall_normal_deg")} for c in characterization["topology"]["wall_cusps"]],
        "cells": [{k: c[k] for k in ("cell_id", "kind", "z_start_m", "z_end_m", "axis_bz_peak_t", "wall_b_min_t")} for c in characterization["topology"]["cells"]],
        "axis_nulls": [{k: n[k] for k in ("null_id", "z_m", "zone", "classification")} for n in characterization["axis_nulls"]["nulls"]],
        "all_traces_terminate_cleanly": bool(characterization["all_traces_terminate_cleanly"]),
        "rho": [{k: row[k] for k in ("cusp_id", "z_c_m", "wall_b_t", "rho_conservative", "rho_wall", "hemp_like_conservative")} for row in descriptors["cusps"]],
        "min_rho_conservative": descriptors.get("min_rho_conservative"),
        "hemp_like_all_cusps": bool(descriptors.get("hemp_like_all_cusps", False)),
        "comparison": comparison,
    }


def l1b_accepted_grid_agreement(result, built: BuiltDesign) -> dict[str, Any] | None:
    """Node-wise |dB| between the new padded level-0 solution and the sealed L1b v1.1 accepted (level-1, padding-0.5) bore map."""

    import gzip

    from experiments.l1b_hemp_confirmation_v1_1 import p2_fields as l1b_fields

    l1b = design_module.l1b_record(built.design_id)
    if l1b is None:
        return None
    record = strict_json_file(design_module.L1B_RESULTS / l1b["record_path"])
    grid_path = design_module.L1B_RESULTS / record["accepted_grid_path"]
    raw = gzip.decompress(grid_path.read_bytes())
    payload = json.loads(raw.decode("utf-8"))
    payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    arrays = payload.get("arrays", payload)
    r_nodes = np.asarray(arrays["r_m"], dtype=np.float64)
    z_nodes = np.asarray(arrays["z_m"], dtype=np.float64)
    b_r_ref = np.asarray(arrays["b_r_t"], dtype=np.float64)
    b_z_ref = np.asarray(arrays["b_z_t"], dtype=np.float64)
    domain = result.problem.domain.to_dict()
    keep = (z_nodes >= domain["z_min_m"]) & (z_nodes <= domain["z_max_m"])
    sampled = l1b_fields.sample_regular_grid(result, r_nodes, z_nodes[keep], scale=built.source_strength_scale)
    diff = np.hypot(sampled.b_r_t - b_r_ref[:, keep], sampled.b_z_t - b_z_ref[:, keep])
    channel = (z_nodes[keep] >= 0.0) & (z_nodes[keep] <= float(built.geometry.chamber.length_m))
    return {
        "reference": "l1b_hemp_confirmation_v1_1 accepted (level-1, padding 0.5) bore map, scaled by source_strength_scale",
        "grid_path": record["accepted_grid_path"], "grid_payload_sha256_recorded": record["accepted_grid_payload_sha256"], "grid_payload_sha256": payload_sha,
        "nodes": int(diff.size), "max_abs_diff_t": float(diff.max()), "rms_diff_t": float(np.sqrt(np.mean(diff**2))),
        "max_abs_diff_channel_t": float(diff[:, channel].max()), "rms_diff_channel_t": float(np.sqrt(np.mean(diff[:, channel] ** 2))),
        "reference_max_b_t": float(np.max(np.hypot(b_r_ref[:, keep], b_z_ref[:, keep]))),
        "gate_max_abs_t": CHANNEL_AGREEMENT_MAX_ABS_T,
        "passed": bool(diff[:, channel].max() <= CHANNEL_AGREEMENT_MAX_ABS_T),
    }


# --------------------------------------------------------------------------
# Production
# --------------------------------------------------------------------------


def produce_field(design_id: str, *, output_root: Path | None = None, log=print) -> dict[str, Any]:
    """Padded level-0 material-aware P2 solve -> hash-bound checkpoint bundle + binding.json (one design, CPU)."""

    built = design_module.build_design(design_id)
    if design_id == design_module.REFERENCE_DESIGN_ID:
        return write_reference_binding(output_root=output_root)
    started = time.perf_counter()
    factor, coverage = padding_factor_for(built)
    preflight = mesh_preflight(built, factor)
    if not preflight["passed"]:
        raise ValueError(f"{design_id}: level-0 mesh preflight failed: angle {preflight['minimum_angle_deg']:.2f} deg, dofs {preflight['level0_p2_dofs']}")
    problem, mesh = graded_mesh_geometry(built.geometry, bore_elements=BORE_ELEMENTS, feature_elements=FEATURE_ELEMENTS, padding_factor=factor)
    log(f"[fields] {design_id}: padding {factor}, {len(mesh.p2_nodes_rz_m):,} P2 DOFs, {len(mesh.triangles):,} triangles, min angle {preflight['minimum_angle_deg']:.2f} deg")
    allocation = preflight["allocation_preflight"]
    solve_started = time.perf_counter()
    result = solve(problem, mesh, relative_tolerance=RELATIVE_TOLERANCE, absolute_tolerance=ABSOLUTE_TOLERANCE, max_iterations=MAX_ITERATIONS,
                   required_available_ram_bytes=int(allocation["effective_required_free_ram_bytes"]))
    solve_seconds = time.perf_counter() - solve_started
    rss_after_solve = current_process_rss_bytes()
    if not result.diagnostics.converged:
        raise ValueError(f"{design_id}: P2 solve did not converge ({result.diagnostics.relative_true_residual_l2:.3e})")
    log(f"[fields] {design_id}: solved in {solve_seconds:.0f} s, {result.diagnostics.iterations} iterations, residual {result.diagnostics.relative_true_residual_l2:.3e}, RSS {rss_after_solve/1e6:.0f} MB")
    windows = _stage_windows(built)
    values = qois(result, windows)
    bound = artifact_from_result(result, qoi_values=values, qoi_windows=windows)
    run_record = {
        "level": 0, "padding_factor": factor, "mesh_sha256": mesh.sha256, "p2_dofs": int(len(mesh.p2_nodes_rz_m)), "triangles": int(len(mesh.triangles)),
        "mesh_quality": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in mesh_quality(mesh).items()},
        "qois_bz_t": values, "iterations": int(result.diagnostics.iterations),
        "relative_true_residual_l2": float(result.diagnostics.relative_true_residual_l2),
        "assembly_seconds": float(result.diagnostics.assembly_seconds), "solve_seconds": float(result.diagnostics.solve_seconds),
        "peak_working_set_bytes": int(result.diagnostics.peak_working_set_bytes),
        "purpose": "pic2d design mini-sweep v1 static field (level-0 padded solve; NOT a qualification chain member)",
    }
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA, "classification": CLASSIFICATION, "config_id": built.geometry.config_id, "level": 0,
        "run_sha256": result.run_sha256, "mesh_sha256": mesh.sha256, "parent_mesh_sha256": mesh.parent_mesh_sha256,
        "previous_checkpoint_file_sha256": "0" * 64, "domain_study": {"padding_factor": factor}, "run": run_record, "bound_artifact": bound,
        "chain_authority": {"status": "standalone_design_mini_sweep_field_not_a_qualification_chain"}, "integrity": {},
    }
    root = design_module.FIELDS_DIR if output_root is None else Path(output_root)
    directory = root / design_id
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{design_id}.domain-padding-{factor:.2f}.level-0.json"
    file_hash = write_checkpoint_bundle(checkpoint_path, checkpoint)
    summary = checkpoint_metadata_summary(checkpoint_path)
    metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    sidecar = checkpoint_path.parent / metadata["array_sidecar"]["file"]
    log(f"[fields] {design_id}: checkpoint {checkpoint_path.name} ({checkpoint_path.stat().st_size/1e6:.1f} MB) + sidecar ({sidecar.stat().st_size/1e6:.1f} MB)")
    identity = design_module.canonical_sha256({"design_id": design_id, "checkpoint_file_sha256": file_hash, "scale": built.source_strength_scale})
    topology = characterize_bore(result, built, source_identity_sha256=identity)
    l1b_agreement = l1b_accepted_grid_agreement(result, built)
    domain = problem.domain.to_dict()
    solid_regions = sorted({r.region_id for r in problem.regions if r.region_id.startswith(SOLID_REGION_PREFIXES)})
    binding = {
        "schema": BINDING_SCHEMA,
        "status": "draft-field-artifact-for-a-not-yet-preregistered-sweep",
        "design_id": design_id,
        "design": built.design.to_dict(),
        "identity": built.identity,
        "geometry_config_id": built.geometry.config_id,
        "source_strength_scale": built.source_strength_scale,
        "scale_note": "the FEM is solved at the nominal SmCo-like remanence; the PIC map (and every gate here) multiplies B by the design's L1a source_strength_scale so the field carries the catalogue's magnet strength (linear problem, exact) - the same convention as L1b v1.1",
        "map": {
            "checkpoint_path": checkpoint_path.relative_to(REPOSITORY).as_posix(),
            "checkpoint_file_sha256": file_hash,
            "checkpoint_payload_sha256": summary["payload_sha256"],
            "mesh_sha256": mesh.sha256,
            "run_sha256": result.run_sha256,
            "sidecar_file_sha256": metadata["array_sidecar"]["file_sha256"],
            "fem_level": 0,
            "domain_study": {"padding_factor": factor},
            "classification": CLASSIFICATION,
            "materials": "linear soft-iron poles and return yoke mu_r 4000, SmCo-like recoil mu_r 1.05 + remanence, BN / Al / Cu at mu_r 1 (fem_reference.adapters.adapt_geometry; the L1b v1.1 materials)",
            "mesh": {"bore_elements": BORE_ELEMENTS, "feature_elements": FEATURE_ELEMENTS, "reject_below_angle_deg": REJECT_BELOW_ANGLE_DEG},
            "solver": {"relative_tolerance": RELATIVE_TOLERANCE, "absolute_tolerance": ABSOLUTE_TOLERANCE, "max_iterations": MAX_ITERATIONS, "backend": result.diagnostics.backend},
        },
        "bounding_box": domain,
        "supported_pic_box": {"r_max_m": domain["r_max_m"] - TRUNCATION_MARGIN_M, "z_max_m": domain["z_max_m"] - TRUNCATION_MARGIN_M},
        "coverage": coverage,
        "plasma_regions": list(PLASMA_REGIONS),
        "solid_regions_sampled": solid_regions,
        "front_face_note": "plasma nodes on the thruster front face (z = L_channel, r > exit radius) carry the plasma-side (ambient) limit of the P2 field; nodes inside solids are never evaluated",
        "field_convention": "B_r = -dA_phi/dz, B_z = A_phi/r + dA_phi/dr (2 dA_phi/dr on the axis)",
        "mesh_preflight": preflight,
        "solve": {**run_record, "solve_wall_seconds": solve_seconds, "rss_after_solve_bytes": int(rss_after_solve), "converged": bool(result.diagnostics.converged)},
        "gates": {
            "mesh_angle": {"passed": preflight["passes_angle_gate"], "minimum_angle_deg": preflight["minimum_angle_deg"], "gate_deg": REJECT_BELOW_ANGLE_DEG},
            "solver_converged": {"passed": bool(result.diagnostics.converged), "relative_true_residual_l2": float(result.diagnostics.relative_true_residual_l2), "gate": RELATIVE_TOLERANCE},
            "coverage": {"passed": True, **coverage["required"], "domain": domain},
            "topology_agreement": topology["comparison"],
            "l1b_accepted_grid_agreement": l1b_agreement,
        },
        "topology_under_iron": {k: topology[k] for k in TOPOLOGY_KEYS},
        "total_seconds": time.perf_counter() - started,
    }
    binding["gates"]["all_passed"] = bool(all(g["passed"] for g in binding["gates"].values() if isinstance(g, dict) and "passed" in g))
    _write_json(binding_path(design_id) if output_root is None else directory / "binding.json", binding)
    log(f"[fields] {design_id}: gates all_passed={binding['gates']['all_passed']}; cusps {len(topology['wall_cusps'])} (catalogue {topology['comparison']['catalogue_wall_cusp_count']}), "
        f"max shift {topology['comparison']['max_shift_m']} m (tol {topology['comparison']['tolerance_m']:.2e}), rho_iron {topology['min_rho_conservative']}")
    return binding


class _CheckpointMesh:
    def __init__(self, archive) -> None:
        self.vertices_rz_m = np.asarray(archive["mesh.vertices_rz_m"], dtype=np.float64)
        self.triangles = np.asarray(archive["mesh.triangles"], dtype=np.int64)
        self.element_dofs = np.asarray(archive["mesh.element_dofs"], dtype=np.int64)


class _CheckpointProblem:
    def __init__(self, domain: Mapping[str, float]) -> None:
        from cft_revival.fem_reference import Domain

        self.domain = Domain(float(domain["r_min_m"]), float(domain["r_max_m"]), float(domain["z_min_m"]), float(domain["z_max_m"]))


class CheckpointResult:
    """The slice of a FEMResult the samplers read (mesh arrays, A_phi DOFs, domain), loaded from a bound checkpoint."""

    def __init__(self, checkpoint_path: Path, declaration: Mapping[str, Any]) -> None:
        if file_sha256(checkpoint_path) != declaration["checkpoint_file_sha256"]:
            raise PIC2DValidationError("checkpoint file hash differs from the binding")
        metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        sidecar = checkpoint_path.parent / metadata["array_sidecar"]["file"]
        if file_sha256(sidecar) != declaration["sidecar_file_sha256"]:
            raise PIC2DValidationError("checkpoint sidecar hash differs from the binding")
        with np.load(sidecar, allow_pickle=False) as archive:
            self.mesh = _CheckpointMesh(archive)
            self.a_phi_dofs_t_m = np.asarray(archive["solution.a_phi_dofs_t_m"], dtype=np.float64)
        self.problem = _CheckpointProblem(metadata["bound_artifact"]["problem"]["domain"])


def regate_field(design_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Recompute the bore-column gates of an existing binding from its bound checkpoint (no solve) and rewrite the binding."""

    binding = load_binding(design_id, root=root)
    if design_id == design_module.REFERENCE_DESIGN_ID:
        return binding
    verify_binding(binding)
    built = design_module.build_design(design_id)
    result = CheckpointResult(REPOSITORY / binding["map"]["checkpoint_path"], binding["map"])
    identity = design_module.canonical_sha256({"design_id": design_id, "checkpoint_file_sha256": binding["map"]["checkpoint_file_sha256"], "scale": built.source_strength_scale})
    topology = characterize_bore(result, built, source_identity_sha256=identity)
    binding["gates"]["topology_agreement"] = topology["comparison"]
    binding["gates"]["l1b_accepted_grid_agreement"] = l1b_accepted_grid_agreement(result, built)
    binding["topology_under_iron"] = {k: topology[k] for k in TOPOLOGY_KEYS}
    binding["gates"]["all_passed"] = bool(all(g["passed"] for g in binding["gates"].values() if isinstance(g, dict) and "passed" in g))
    binding["regated"] = "gates recomputed from the bound checkpoint (fields.regate_field); solve untouched"
    _write_json(binding_path(design_id) if root is None else Path(root) / design_id / "binding.json", binding)
    return binding


def write_reference_binding(*, output_root: Path | None = None) -> dict[str, Any]:
    """The reference design binds its EXISTING artifacts (no new solve): authority level-1 for the channel, padding-1.5 for plume boxes."""

    from cft_revival.pic2d.fields import load_authority, load_plume_extension

    authority = load_authority()
    v1 = load_plume_extension(DEFAULT_PLUME_EXTENSION_PATH)
    v2 = load_plume_extension(PLUME_EXTENSION_V2_PATH)
    built = design_module.build_design(design_module.REFERENCE_DESIGN_ID)
    binding = {
        "schema": BINDING_SCHEMA,
        "status": "existing-artifacts-no-new-solve",
        "design_id": design_module.REFERENCE_DESIGN_ID,
        "design": built.design.to_dict(),
        "identity": built.identity,
        "source_strength_scale": 1.0,
        "existing_artifacts": {
            "channel": {"file": "modern/spec/pic2d/p2-field-authority-v1.json", "sha256": file_sha256(design_module.MODERN / "spec" / "pic2d" / "p2-field-authority-v1.json"),
                        "map": authority["maps"]["primary"], "note": "regular psi-grid bicubic of the qualified level-1 checkpoint (every v1.x steady-state run)"},
            "plume-12mm": {"file": "modern/spec/pic2d/p2-field-plume-extension-v1.json", "sha256": file_sha256(DEFAULT_PLUME_EXTENSION_PATH),
                           "bounding_box": v1["bounding_box"], "note": "direct node evaluation of the level-1 authority checkpoint (FEM box z <= 36.25 mm; v2.0 attempts 3-8)"},
            "plume-24mm": {"file": "modern/spec/pic2d/p2-field-plume-extension-v2.json", "sha256": file_sha256(PLUME_EXTENSION_V2_PATH),
                           "map": v2["map"], "bounding_box": v2["bounding_box"], "supported_pic_box": v2["supported_pic_box"],
                           "note": "domain-padding-1.5 level-0 solve of the qualification chain (FEM box z <= 60.75 mm; model v2.1, prepared not launched)"},
        },
        "gates": {"channel_cross_check": "performed at map build by p2_plume_field_map (0.02 T bound; measured 0.74 mT max for padding-1.5 vs level-1)", "all_passed": True},
    }
    root = design_module.FIELDS_DIR if output_root is None else Path(output_root)
    (root / design_module.REFERENCE_DESIGN_ID).mkdir(parents=True, exist_ok=True)
    _write_json(root / design_module.REFERENCE_DESIGN_ID / "binding.json", binding)
    return binding


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def load_binding(design_id: str, *, root: Path | None = None) -> dict[str, Any]:
    path = binding_path(design_id) if root is None else Path(root) / design_id / "binding.json"
    if not path.is_file():
        raise PIC2DValidationError(f"no field binding for {design_id} ({path}); run `fields` first")
    binding = json.loads(path.read_text(encoding="utf-8"))
    if binding.get("schema") != BINDING_SCHEMA or binding.get("design_id") != design_id:
        raise PIC2DValidationError(f"{path} is not a design-mini-sweep field binding for {design_id}")
    return binding


def verify_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash the bound checkpoint files (fails closed on any drift)."""

    if binding["design_id"] == design_module.REFERENCE_DESIGN_ID:
        checks = {}
        for name, block in binding["existing_artifacts"].items():
            checks[name] = file_sha256(REPOSITORY / block["file"]) == block["sha256"]
        if not all(checks.values()):
            raise PIC2DValidationError(f"reference field declarations changed: {checks}")
        return {"passed": True, "checks": checks}
    declaration = binding["map"]
    checkpoint_path = REPOSITORY / declaration["checkpoint_path"]
    if not checkpoint_path.is_file():
        raise PIC2DValidationError(f"bound checkpoint missing: {checkpoint_path}")
    file_ok = file_sha256(checkpoint_path) == declaration["checkpoint_file_sha256"]
    metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    sidecar = checkpoint_path.parent / metadata["array_sidecar"]["file"]
    checks = {
        "checkpoint_file_sha256": file_ok,
        "checkpoint_payload_sha256": metadata["integrity"]["payload_sha256"] == declaration["checkpoint_payload_sha256"],
        "mesh_sha256": metadata["mesh_sha256"] == declaration["mesh_sha256"],
        "run_sha256": metadata["run_sha256"] == declaration["run_sha256"],
        "sidecar_file_sha256": sidecar.is_file() and file_sha256(sidecar) == declaration["sidecar_file_sha256"],
    }
    if not all(checks.values()):
        raise PIC2DValidationError(f"{binding['design_id']}: field binding hashes differ: {checks}")
    return {"passed": True, "checks": checks}


# --------------------------------------------------------------------------
# PIC node map
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignFieldSource:
    design_id: str
    binding: Mapping[str, Any]


def design_field_map(mapping: PicMapping, binding: Mapping[str, Any], *, repository_root: Path = REPOSITORY) -> MagneticFieldMap:
    """The PIC node field for one design and domain option.

    Reference design: the existing pipeline verbatim (``build_p2_psi_field`` + ``sample_field_map`` for the channel box;
    ``p2_plume_field_map`` with the v1 / v2 extension for 12 / 24 mm plume boxes).  Other designs: direct evaluation of the
    bound checkpoint at every plasma node, scaled by ``source_strength_scale``; provenance carries the binding hashes.
    """

    grid = mapping.grid
    geometry = grid.geometry
    if binding["design_id"] != mapping.design_id:
        raise PIC2DValidationError("field binding / mapping design mismatch")
    if mapping.design_id == design_module.REFERENCE_DESIGN_ID:
        if not geometry.has_plume:
            psi_field, evidence = build_p2_psi_field(repository_root, role="primary")
            return sample_field_map(psi_field, grid, evidence)
        extension = DEFAULT_PLUME_EXTENSION_PATH if geometry.domain_z_max_m <= 0.03625 - 1e-12 else PLUME_EXTENSION_V2_PATH
        return p2_plume_field_map(repository_root, grid, role="primary", extension_path=extension)
    verify_binding(binding)
    declaration = binding["map"]
    bounds = binding["bounding_box"]
    supported = binding["supported_pic_box"]
    if geometry.max_radius_m > supported["r_max_m"] + 1e-12 or geometry.domain_z_max_m > supported["z_max_m"] + 1e-12 or geometry.z_min_m < bounds["z_min_m"] - 1e-12:
        raise PIC2DValidationError(f"{mapping.design_id}: the PIC box (r <= {geometry.max_radius_m}, z <= {geometry.domain_z_max_m}) exceeds the bound field's supported box {supported}")
    allowed = set(binding["plasma_regions"]) | set(binding["solid_regions_sampled"])
    evaluator = BoundP2Evaluator(repository_root / declaration["checkpoint_path"], declaration, allowed_regions=allowed, bounds=bounds)
    masks = build_mesh_masks(grid)
    plasma = masks.plasma_node
    scale = float(binding["source_strength_scale"])
    b_r = np.zeros(grid.node_shape, dtype=np.float64)
    b_z = np.zeros(grid.node_shape, dtype=np.float64)
    nudge = 1.0e-9
    regions_seen: set[str] = set()
    for i, radius in enumerate(grid.r_m):
        for j, axial in enumerate(grid.z_m):
            if not plasma[i, j]:
                continue
            query_z = float(axial)
            if geometry.has_plume and masks.body_face_node[i, j]:
                query_z = geometry.z_max_m + nudge
            (_, br, bz), regions = evaluator.evaluate_with_regions(float(radius), query_z)
            regions_seen |= regions
            b_r[i, j], b_z[i, j] = scale * br, scale * bz
    b_r[0, :] = 0.0
    provenance = {
        "kind": "p2-direct-node-sample-design-mini-sweep",
        "design_id": mapping.design_id,
        "domain_option": mapping.domain,
        "field_source": "design-mini-sweep-binding-v1",
        "binding_schema": binding["schema"],
        "checkpoint_path": declaration["checkpoint_path"],
        "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
        "checkpoint_payload_sha256": declaration["checkpoint_payload_sha256"],
        "mesh_sha256": declaration["mesh_sha256"],
        "run_sha256": declaration["run_sha256"],
        "sidecar_file_sha256": declaration["sidecar_file_sha256"],
        "fem_level": declaration["fem_level"],
        "padding_factor": declaration["domain_study"]["padding_factor"],
        "source_strength_scale": scale,
        "p2_classification": evaluator.classification,
        "bounding_box": dict(bounds),
        "supported_pic_box": dict(supported),
        "plasma_nodes_sampled": int(plasma.sum()),
        "regions_touched": sorted(regions_seen),
        "geometry_snaps": mapping.snaps,
        "node_sampling": "direct quadratic A_phi evaluation on the plasma nodes (plasma-side limit on the front face); zero on body nodes; scaled",
    }
    return MagneticFieldMap(grid, b_r, b_z, provenance)


__all__ = [
    "BINDING_SCHEMA", "CHANNEL_AGREEMENT_MAX_ABS_T", "COVER_R_M", "COVER_Z_BEHIND_EXIT_M", "PADDING_LADDER", "REJECT_BELOW_ANGLE_DEG",
    "CheckpointResult", "binding_path", "characterize_bore", "coverage_requirement", "design_field_map", "fields_dir",
    "l1b_accepted_grid_agreement", "load_binding", "mesh_preflight", "padding_factor_for", "produce_field", "regate_field", "verify_binding",
    "write_reference_binding",
]
