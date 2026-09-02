"""Run three nested adaptive levels for each accepted hypothetical CFT geometry."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import log, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np

from cft_revival.fem_reference import (
    adjacent_size_growth,
    artifact_from_result,
    component_dorfler_mark,
    checkpoint_metadata_summary,
    domain_study_evidence,
    evaluate_phase_matched_domain_expansion,
    FEMValidationError,
    estimate_indicators,
    graded_mesh_geometry,
    mesh_quality,
    load_checkpoint_bundle,
    preflight_third_level,
    preflight_level_allocation,
    prolong_p2_solution,
    qois,
    refine_mesh,
    replay_artifact,
    solve,
    viewer_contract,
    write_checkpoint_bundle,
    write_json,
)
from cft_revival.geometry import reference_variants

LEVELS = 3
PADDING_FACTOR = 0.5
DORFLER_THETA = 0.5
MAXIMUM_ADJACENT_SIZE_GROWTH = 1.3
MAXIMUM_P2_DOFS = 1_500_000
DEFAULT_EXECUTION_P2_DOFS = 400_000
MINIMUM_THIRD_LEVEL_FREE_RAM_BYTES = 8 * 1024**3


def _relative(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-300)


def _finalize_checkpoint_chain(
    output,
    anchors,
    authority_artifact,
    *,
    final_artifact=None,
    chain_kind="adaptive",
    require_third_level_ram=False,
):
    root = authority_artifact["acceptance_evidence"]["checkpoint_authority"]
    final_artifact = authority_artifact if final_artifact is None else final_artifact
    final_run = final_artifact["anchors"]["run_sha256"]
    final_mesh = final_artifact["anchors"]["mesh_sha256"]
    previous_file = "0" * 64
    finalized = []
    for old_anchor in anchors:
        if require_third_level_ram:
            preflight_third_level(1)
        path = output / old_anchor["file"]
        checkpoint, _verified = load_checkpoint_bundle(path)
        bound = checkpoint["bound_artifact"]
        mesh = bound["mesh"]
        checkpoint["previous_checkpoint_file_sha256"] = previous_file
        checkpoint["chain_authority"] = {
            "authority_root_sha256": root["authority_root_sha256"],
            "artifact_schema": root["artifact_schema"],
            "classification": root["classification"],
            "design_id": root["design_id"],
            "geometry_sha256": root["geometry_sha256"],
            "magnetics_sha256": root["magnetics_sha256"],
            "config_id": root["config_id"],
            "implementation_sha256": root["implementation_sha256"],
            "acceptance_code_sha256": root["acceptance_code_sha256"],
            "base_problem_sha256": root["problem_sha256"],
            "chain_kind": chain_kind,
            "final_checkpoint_run_sha256": final_run,
            "final_checkpoint_mesh_sha256": final_mesh,
        }
        payload = {key: value for key, value in checkpoint.items() if key != "integrity"}
        checkpoint["integrity"]["payload_sha256"] = sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        file_hash = write_checkpoint_bundle(path, checkpoint)
        checkpoint_metadata = checkpoint_metadata_summary(path)
        robin_edges = sum(
            len(mesh["boundary_edges"][name])
            for name in ("outer_radial", "z_min", "z_max")
        )
        finalized_anchor = {
                "level": old_anchor["level"],
                "file": old_anchor["file"],
                "file_sha256": file_hash,
                "payload_sha256": checkpoint_metadata["payload_sha256"],
                "mesh_sha256": checkpoint["mesh_sha256"],
                "parent_mesh_sha256": checkpoint["parent_mesh_sha256"],
                "previous_checkpoint_file_sha256": previous_file,
                "p2_dofs": len(mesh["p2_nodes_rz_m"]),
                "triangles": len(mesh["triangles"]),
                "robin_edges": robin_edges,
                "chain_final_run_sha256": final_run,
                "chain_final_mesh_sha256": final_mesh,
                "run_sha256": bound["anchors"]["run_sha256"],
                "problem_sha256": bound["acceptance_evidence"][
                    "checkpoint_authority"
                ]["problem_sha256"],
            }
        if "padding_factor" in old_anchor:
            finalized_anchor["padding_factor"] = old_anchor["padding_factor"]
        finalized.append(finalized_anchor)
        previous_file = file_hash
    return finalized


def _read_l1b_qois(path: Path) -> dict[str, float]:
    groups: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as source:
        iterator = iter(source)
        for line in iterator:
            if '"fixed_qois_bz_t"' not in line:
                continue
            values: dict[str, float] = {}
            for following in iterator:
                stripped = following.strip()
                if stripped.startswith("}"):
                    break
                if ":" in stripped:
                    key, value = stripped.rstrip(",").split(":", 1)
                    try:
                        values[json.loads(key)] = float(value)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
            if values:
                groups.append(values)
    if not groups:
        raise RuntimeError(f"no L1b summary QoIs found in {path}")
    return groups[-1]


def _element_area_sizes(mesh) -> np.ndarray:
    sizes = np.empty(len(mesh.triangles))
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        first, second = points[1] - points[0], points[2] - points[0]
        sizes[element] = sqrt(abs(float(first[0] * second[1] - first[1] * second[0])))
    return sizes


def _resolution(geometry, mesh, stage_windows) -> dict[str, object]:
    sizes = _element_area_sizes(mesh)
    centroids = np.asarray(
        [np.mean(mesh.vertices_rz_m[triangle], axis=0) for triangle in mesh.triangles]
    )
    bore_h: dict[str, float] = {}
    for name, radius, z_min, z_max in stage_windows:
        selected = (
            (centroids[:, 0] <= radius)
            & (centroids[:, 1] >= z_min)
            & (centroids[:, 1] <= z_max)
        )
        bore_h[f"{name}-bore-average"] = float(
            sqrt(float(np.mean(sizes[selected] ** 2)))
        )
    feature_h: dict[str, float] = {}
    feature_counts: dict[str, float] = {}
    for region in geometry.regions:
        relative = geometry.material_by_id(region.material_id).relative_permeability
        if relative == 1.0:
            continue
        selected = np.asarray(
            [tag == region.region_id for tag in mesh.triangle_region_ids]
        )
        if not np.any(selected):
            continue
        thickness = min(
            region.r_outer_start_m - region.r_inner_start_m,
            region.r_outer_end_m - region.r_inner_end_m,
            region.z_max_m - region.z_min_m,
        )
        local_h = float(sqrt(float(np.mean(sizes[selected] ** 2))))
        feature_h[region.region_id] = local_h
        feature_counts[region.region_id] = thickness / local_h
    return {
        "qoi_h_m": bore_h,
        "source_feature_h_m": feature_h,
        "bore_elements_across": {
            key: geometry.chamber.outer_radius_m / value for key, value in bore_h.items()
        },
        "feature_elements_across": feature_counts,
        "minimum_feature_elements_across": min(feature_counts.values()),
    }


def _observed_orders(runs, qoi_keys) -> dict[str, float | None]:
    if len(runs) < 3:
        return {key: None for key in qoi_keys}
    orders: dict[str, float | None] = {}
    for key in qoi_keys:
        first_delta = abs(runs[0]["qois_bz_t"][key] - runs[1]["qois_bz_t"][key])
        second_delta = abs(runs[1]["qois_bz_t"][key] - runs[2]["qois_bz_t"][key])
        h0 = runs[0]["resolution"]["qoi_h_m"][key]
        h2 = runs[2]["resolution"]["qoi_h_m"][key]
        denominator = log(sqrt(h0 / h2)) if h0 > h2 else 0.0
        orders[key] = (
            log(first_delta / second_delta) / denominator
            if first_delta > 0.0 and second_delta > 0.0 and denominator > 0.0
            else None
        )
    return orders


def _run_domain_studies(
    geometry,
    stage_windows,
    output,
    adaptive_authority_artifact,
    padding_half_result,
):
    name = geometry.config_id.removesuffix("-v1")
    domain_runs = []
    study_evidence = []
    provisional_anchors = []
    previous_file_hash = "0" * 64
    final_bound_artifact = None
    for study_index, padding_factor in enumerate((0.5, 1.0, 1.5)):
        preflight_third_level(1)
        if padding_factor == 0.5:
            result = padding_half_result
            problem = result.problem
            mesh = result.mesh
        else:
            problem, mesh = graded_mesh_geometry(
                geometry,
                bore_elements=8,
                feature_elements=4,
                padding_factor=padding_factor,
            )
            robin_edges = sum(
                len(mesh.boundary_edges[boundary])
                for boundary in ("outer_radial", "z_min", "z_max")
            )
            allocation = preflight_level_allocation(
                p2_dofs=len(mesh.p2_nodes_rz_m),
                triangles=len(mesh.triangles),
                robin_edges=robin_edges,
                third_level=True,
                phase=f"domain_padding_{padding_factor}",
            )
            result = solve(
                problem,
                mesh,
                relative_tolerance=2.0e-10,
                max_iterations=16000,
                required_available_ram_bytes=int(
                    allocation["effective_required_free_ram_bytes"]
                ),
            )
        preflight_third_level(1)
        values = qois(result, stage_windows)
        resolution = _resolution(geometry, mesh, stage_windows)
        bound = artifact_from_result(
            result,
            qoi_values=values,
            qoi_windows=stage_windows,
        )
        preflight_third_level(1)
        study = domain_study_evidence(bound, padding_factor)
        domain_run = {
            "padding_factor": padding_factor,
            "mesh_sha256": mesh.sha256,
            "p2_dofs": len(mesh.p2_nodes_rz_m),
            "triangles": len(mesh.triangles),
            "mesh_quality": mesh_quality(mesh),
            "adjacent_area_size_growth": adjacent_size_growth(mesh),
            "qois_bz_t": values,
            "resolution": resolution,
            "bound_local_h_m": study["local_h_m"],
            "iterations": result.diagnostics.iterations,
            "relative_true_residual_l2": (
                result.diagnostics.relative_true_residual_l2
            ),
            "assembly_seconds": result.diagnostics.assembly_seconds,
            "solve_seconds": result.diagnostics.solve_seconds,
            "peak_working_set_bytes": result.diagnostics.peak_working_set_bytes,
        }
        checkpoint = {
            "schema_version": "cft_revival.fem_reference.checkpoint/1.2.0",
            "classification": (
                "independent_numerical_reference_not_hardware_validation"
            ),
            "config_id": geometry.config_id,
            "level": 0,
            "run_sha256": result.run_sha256,
            "mesh_sha256": mesh.sha256,
            "parent_mesh_sha256": mesh.parent_mesh_sha256,
            "previous_checkpoint_file_sha256": previous_file_hash,
            "domain_study": {"padding_factor": padding_factor},
            "run": domain_run,
            "bound_artifact": bound,
            "chain_authority": {
                "status": "provisional_not_authoritative_until_finalized"
            },
            "integrity": {},
        }
        checkpoint_path = (
            output
            / "checkpoints"
            / f"{name}.domain-padding-{padding_factor:.1f}.json"
        )
        preflight_third_level(1)
        file_hash = write_checkpoint_bundle(checkpoint_path, checkpoint)
        summary = checkpoint_metadata_summary(checkpoint_path)
        provisional_anchors.append(
            {
                "level": 0,
                "padding_factor": padding_factor,
                "file": str(checkpoint_path.relative_to(output)),
                "file_sha256": file_hash,
                "payload_sha256": summary["payload_sha256"],
                "mesh_sha256": mesh.sha256,
                "parent_mesh_sha256": mesh.parent_mesh_sha256,
                "previous_checkpoint_file_sha256": previous_file_hash,
            }
        )
        previous_file_hash = file_hash
        domain_runs.append(domain_run)
        study_evidence.append(study)
        final_bound_artifact = bound
        print(
            f"{geometry.config_id} domain padding {padding_factor:.1f}: "
            f"{len(mesh.p2_nodes_rz_m):,} P2 DOFs; "
            f"true residual {result.diagnostics.relative_true_residual_l2:.3e}",
            flush=True,
        )
    assert final_bound_artifact is not None
    anchors = _finalize_checkpoint_chain(
        output,
        provisional_anchors,
        adaptive_authority_artifact,
        final_artifact=final_bound_artifact,
        chain_kind="domain",
        require_third_level_ram=True,
    )
    evaluation = evaluate_phase_matched_domain_expansion(
        tuple(
            {
                **study,
                "qois_bz_t": {
                    key: value
                    for key, value in study["qois_bz_t"].items()
                    if key.endswith("-bore-average")
                },
            }
            for study in study_evidence
        )
    )
    return anchors, domain_runs, evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design")
    parser.add_argument("--allow-third-level", action="store_true")
    arguments = parser.parse_args()
    geometries = tuple(
        geometry
        for geometry in reference_variants()
        if arguments.design is None
        or geometry.config_id.removesuffix("-v1") == arguments.design
    )
    if not geometries:
        raise RuntimeError(f"unknown FEM design {arguments.design!r}")
    resource_preflight = None
    if arguments.allow_third_level:
        resource_preflight = preflight_third_level(len(geometries))
    execution_limit = (
        MAXIMUM_P2_DOFS
        if arguments.allow_third_level
        else DEFAULT_EXECUTION_P2_DOFS
    )
    output = Path(__file__).resolve().parent / "artifacts"
    if arguments.allow_third_level:
        name = geometries[0].config_id.removesuffix("-v1")
        output = output / "third-level" / name
    l1b_root = Path(__file__).resolve().parents[1] / "material_fields" / "artifacts"
    campaign_started = perf_counter()
    entries: list[dict[str, object]] = []
    for geometry in geometries:
        name = geometry.config_id.removesuffix("-v1")
        stage_windows = tuple(
            (
                f"stage-{index + 1}",
                geometry.chamber.outer_radius_m,
                stage.z_min_m,
                stage.z_max_m,
            )
            for index, stage in enumerate(geometry.stages)
        )
        problem, mesh = graded_mesh_geometry(
            geometry,
            bore_elements=8,
            feature_elements=4,
            padding_factor=PADDING_FACTOR,
        )
        runs: list[dict[str, object]] = []
        level_results = []
        checkpoint_anchors: list[dict[str, object]] = []
        previous_checkpoint_file_sha256 = "0" * 64
        finest = None
        previous_result = None
        for level in range(LEVELS):
            stop_after_level = False
            quality = mesh_quality(mesh)
            if quality["minimum_angle_deg"] < 10.0:
                raise RuntimeError("adaptive mesh violated the ten-degree rejection gate")
            robin_edges = sum(
                len(mesh.boundary_edges[name])
                for name in ("outer_radial", "z_min", "z_max")
            )
            try:
                allocation_preflight = preflight_level_allocation(
                    p2_dofs=len(mesh.p2_nodes_rz_m),
                    triangles=len(mesh.triangles),
                    robin_edges=robin_edges,
                    third_level=arguments.allow_third_level or level >= 2,
                )
            except FEMValidationError as error:
                abort_payload = {
                    "schema_version": (
                        "cft_revival.fem_reference.resource_abort/1.0.0"
                    ),
                    "classification": (
                        "independent_numerical_reference_not_hardware_validation"
                    ),
                    "config_id": geometry.config_id,
                    "level": level,
                    "mesh_sha256": mesh.sha256,
                    "previous_checkpoint_file_sha256": (
                        previous_checkpoint_file_sha256
                    ),
                    "reason": str(error),
                }
                abort_checkpoint = {
                    **abort_payload,
                    "integrity": {
                        "algorithm": "sha256",
                        "canonicalization": "json-sort-keys-compact-utf8-v1",
                        "payload_sha256": sha256(
                            json.dumps(
                                abort_payload,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode()
                        ).hexdigest(),
                    },
                }
                write_json(
                    output / "checkpoints" / f"{name}.level-{level}.resource-abort.json",
                    abort_checkpoint,
                )
                raise
            result = solve(
                problem,
                mesh,
                relative_tolerance=2.0e-10,
                max_iterations=16000,
                initial_a_phi_dofs_t_m=(
                    None
                    if previous_result is None
                    else prolong_p2_solution(previous_result, mesh)
                ),
                required_available_ram_bytes=int(
                    allocation_preflight["effective_required_free_ram_bytes"]
                ),
            )
            values = qois(result, stage_windows)
            resolution = _resolution(geometry, mesh, stage_windows)
            run: dict[str, object] = {
                "level": level,
                "mesh_sha256": mesh.sha256,
                "parent_mesh_sha256": mesh.parent_mesh_sha256,
                "mesh_quality": quality,
                "adjacent_area_size_growth": adjacent_size_growth(mesh),
                "qois_bz_t": values,
                "resolution": resolution,
                "iterations": result.diagnostics.iterations,
                "relative_true_residual_l2": result.diagnostics.relative_true_residual_l2,
                "energy_action_relative_diagnostic": (
                    result.diagnostics.energy_action_relative
                ),
                "assembly_seconds": result.diagnostics.assembly_seconds,
                "solve_seconds": result.diagnostics.solve_seconds,
                "peak_working_set_bytes": result.diagnostics.peak_working_set_bytes,
                "allocation_preflight": allocation_preflight,
            }
            finest = result
            previous_result = result
            if level < LEVELS - 1:
                indicators = estimate_indicators(result, stage_windows)
                bulk_marked = component_dorfler_mark(
                    indicators, DORFLER_THETA
                )
                marked = bulk_marked
                run["adaptivity"] = {
                    "theta": DORFLER_THETA,
                    "dorfler_marked_elements": len(bulk_marked),
                    "marked_elements_after_conformity_closure": len(marked),
                    "element_count": len(mesh.triangles),
                    "marked_indicator_fraction": float(
                        np.sum(indicators.total_squared[marked])
                        / np.sum(indicators.total_squared)
                    ),
                    "residual_marked_fraction": float(
                        np.sum(indicators.residual_squared[marked])
                        / max(np.sum(indicators.residual_squared), 1.0e-300)
                    ),
                    "flux_jump_marked_fraction": float(
                        np.sum(indicators.flux_jump_squared[marked])
                        / max(np.sum(indicators.flux_jump_squared), 1.0e-300)
                    ),
                    "qoi_proxy_marked_fraction": float(
                        np.sum(indicators.qoi_proxy_squared[marked])
                        / max(np.sum(indicators.qoi_proxy_squared), 1.0e-300)
                    ),
                    "residual_indicator_sum": float(np.sum(indicators.residual_squared)),
                    "flux_jump_indicator_sum": float(np.sum(indicators.flux_jump_squared)),
                    "qoi_proxy_indicator_sum": float(np.sum(indicators.qoi_proxy_squared)),
                }
                red_closure_upper_bound = 4 * len(mesh.p2_nodes_rz_m)
                run["adaptivity"]["next_level_red_closure_p2_dof_upper_bound"] = (
                    red_closure_upper_bound
                )
                if red_closure_upper_bound > execution_limit:
                    run["adaptivity"]["refinement_skipped_reason"] = (
                        "strict_1.3_gradation_red_closure_upper_bound_exceeds_"
                        "active_resource_policy_limit"
                    )
                    stop_after_level = True
                else:
                    refined = refine_mesh(
                        mesh,
                        problem.domain,
                        marked,
                        reject_below_angle_deg=10.0,
                        maximum_adjacent_size_growth=MAXIMUM_ADJACENT_SIZE_GROWTH,
                    )
                    child_counts = np.bincount(
                        refined.element_parent_ids,
                        minlength=len(mesh.triangles),
                    )
                    run["adaptivity"]["refined_parents_after_gradation_closure"] = int(
                        np.count_nonzero(child_counts > 1)
                    )
                    mesh = refined
            runs.append(run)
            level_results.append(result)
            if arguments.allow_third_level:
                preflight_third_level(1)
            bound_artifact = artifact_from_result(
                result,
                qoi_values=values,
                qoi_windows=stage_windows,
            )
            checkpoint_payload = {
                "schema_version": "cft_revival.fem_reference.checkpoint/1.2.0",
                "classification": (
                    "independent_numerical_reference_not_hardware_validation"
                ),
                "config_id": geometry.config_id,
                "level": level,
                "run_sha256": result.run_sha256,
                "mesh_sha256": result.mesh.sha256,
                "parent_mesh_sha256": result.mesh.parent_mesh_sha256,
                "previous_checkpoint_file_sha256": (
                    previous_checkpoint_file_sha256
                ),
                "run": run,
                "bound_artifact": bound_artifact,
                "chain_authority": {
                    "status": "provisional_not_authoritative_until_finalized"
                },
            }
            checkpoint = {
                **checkpoint_payload,
                "integrity": {
                    "algorithm": "sha256",
                    "canonicalization": "json-sort-keys-compact-utf8-v1",
                    "payload_sha256": sha256(
                        json.dumps(
                            checkpoint_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode()
                    ).hexdigest(),
                },
            }
            checkpoint_path = (
                output / "checkpoints" / f"{name}.level-{level}.json"
            )
            if arguments.allow_third_level:
                preflight_third_level(1)
            checkpoint_file_sha256 = write_checkpoint_bundle(
                checkpoint_path, checkpoint
            )
            checkpoint_summary = checkpoint_metadata_summary(checkpoint_path)
            checkpoint_anchors.append(
                {
                    "level": level,
                    "file": str(checkpoint_path.relative_to(output)),
                    "file_sha256": checkpoint_file_sha256,
                    "payload_sha256": checkpoint_summary["payload_sha256"],
                    "mesh_sha256": result.mesh.sha256,
                    "parent_mesh_sha256": result.mesh.parent_mesh_sha256,
                    "previous_checkpoint_file_sha256": (
                        previous_checkpoint_file_sha256
                    ),
                }
            )
            previous_checkpoint_file_sha256 = checkpoint_file_sha256
            print(
                f"{geometry.config_id} level {level}: "
                f"{len(mesh.p2_nodes_rz_m):,} next-mesh P2 DOFs; "
                f"true residual {result.diagnostics.relative_true_residual_l2:.3e}",
                flush=True,
            )
            if stop_after_level:
                break
        assert finest is not None
        checkpoint_anchors = _finalize_checkpoint_chain(
            output,
            checkpoint_anchors,
            bound_artifact,
            require_third_level_ram=arguments.allow_third_level,
        )
        if arguments.allow_third_level:
            domain_anchors, domain_runs, domain_evaluation = _run_domain_studies(
                geometry,
                stage_windows,
                output,
                bound_artifact,
                level_results[0],
            )
        else:
            domain_anchors = []
            domain_runs = []
            domain_evaluation = {
                "phase_matched": False,
                "successive_qoi_relative_changes": [],
                "maximum_qoi_relative_change": 0.01,
                "passed": False,
            }
        qoi_keys = sorted(
            key for key in runs[0]["qois_bz_t"] if key.endswith("-bore-average")
        )
        changes = [
            {
                key: _relative(left["qois_bz_t"][key], right["qois_bz_t"][key])
                for key in qoi_keys
            }
            for left, right in zip(runs, runs[1:])
        ]
        orders = _observed_orders(runs, qoi_keys)
        two_successive = len(changes) >= 2 and all(
            change[key] < 0.01 for change in changes[-2:] for key in qoi_keys
        )
        stable_positive = all(
            orders[key] is not None and orders[key] > 0.0 for key in qoi_keys
        )
        growth_gate = all(
            run["adjacent_area_size_growth"] <= 1.3 + 1.0e-12 for run in runs
        )
        domain_expansion_gate = bool(domain_evaluation["passed"])
        convergence = {
            "adaptive_nested_levels": LEVELS,
            "completed_adaptive_levels": len(runs),
            "maximum_p2_dofs": MAXIMUM_P2_DOFS,
            "successive_volume_qoi_relative_changes": changes,
            "observed_orders_from_actual_qoi_h": orders,
            "two_successive_less_than_one_percent": two_successive,
            "stable_positive_order": stable_positive,
            "adjacent_size_growth_gate": growth_gate,
            "less_than_one_percent_reached": (
                two_successive
                and stable_positive
                and growth_gate
                and domain_expansion_gate
            ),
            "phase_matched_domain_expansion_gate": domain_expansion_gate,
            "domain_expansion": domain_evaluation,
            "acceptance_qois": qoi_keys,
            "diagnostic_only_qois": sorted(
                set(runs[-1]["qois_bz_t"]) - set(qoi_keys)
            ),
            "cell_interface_maxima_policy": "screening_only_not_used_for_acceptance",
            "energy_identity_policy": "diagnostic_only_not_an_acceptance_gate",
        }
        l1b_path = l1b_root / f"{name}.material-field.json"
        l1b_qois = _read_l1b_qois(l1b_path)
        l1b_comparison: dict[str, object] = {}
        for l1b_key, l1b_value in l1b_qois.items():
            fem_key = (
                l1b_key.replace("-axis", "-axis-patch")
                if l1b_key.endswith("-axis")
                else l1b_key
            )
            l1b_comparison[l1b_key] = {
                "fem_qoi_key": fem_key,
                "fem_reference_bz_t": runs[-1]["qois_bz_t"][fem_key],
                "l1b_structured_grid_bz_t": l1b_value,
                "relative_difference": _relative(
                    runs[-1]["qois_bz_t"][fem_key], l1b_value
                ),
                "identical_qoi_semantics": True,
                "fem_evaluation": (
                    "weighted_quadratic_axis_patch_recovery"
                    if l1b_key.endswith("-axis")
                    else "piecewise_P2_axisymmetric_volume_integral"
                ),
                "l1b_evaluation": (
                    "structured_grid_axis_interpolation"
                    if l1b_key.endswith("-axis")
                    else "structured_grid_axisymmetric_volume_quadrature"
                ),
            }
        comparisons = {
            "l1b_artifact": str(
                l1b_path.relative_to(Path(__file__).resolve().parents[2])
            ),
            "l1b_artifact_sha256": sha256(l1b_path.read_bytes()).hexdigest(),
            "l1b_fixed_and_volume_qois": l1b_comparison,
        }
        if arguments.allow_third_level:
            preflight_third_level(1)
        artifact = artifact_from_result(
            finest,
            qoi_values=runs[-1]["qois_bz_t"],
            qoi_windows=stage_windows,
            level_evidence=checkpoint_anchors,
            domain_studies=domain_anchors,
            evidence_base_path=str(
                output.relative_to(Path(__file__).resolve().parents[2])
            ),
            convergence=convergence,
            comparisons=comparisons,
        )
        artifact_path = output / f"{name}.fem-reference.json"
        viewer_path = output / f"{name}.fem-reference.viewer.json"
        artifact_file_hash = write_json(artifact_path, artifact)
        if arguments.allow_third_level:
            preflight_third_level(1)
        viewer_file_hash = write_json(viewer_path, viewer_contract(artifact))
        if arguments.allow_third_level:
            preflight_third_level(1)
        if not replay_artifact(artifact)["passed"]:
            raise RuntimeError("fresh FEM artifact failed deterministic replay")
        entries.append(
            {
                "config_id": geometry.config_id,
                "artifact": artifact_path.name,
                "viewer": viewer_path.name,
                "artifact_file_sha256": artifact_file_hash,
                "viewer_file_sha256": viewer_file_hash,
                "artifact_payload_sha256": artifact["integrity"]["payload_sha256"],
                "runs": runs,
                "checkpoints": checkpoint_anchors,
                "domain_checkpoints": domain_anchors,
                "domain_runs": domain_runs,
                "convergence": convergence,
                "qualification_status": (
                    "NUMERICAL_P2_QUALIFIED"
                    if convergence["less_than_one_percent_reached"]
                    else "SCREENING_ONLY"
                ),
                "l1b_comparison": l1b_comparison,
                "classification": artifact["classification"],
            }
        )
    payload = {
        "schema_version": "cft_revival.fem_reference.campaign/1.1.0",
        "classification": "independent_numerical_reference_not_hardware_validation",
        "artifact_authority": (
            "schema_v1.3_recomputed_acceptance_with_bound_checkpoint_chain"
        ),
        "dependencies": {
            "required": ["Python standard library", "NumPy"],
            "available_but_not_required": ["Warp"],
            "unavailable_and_not_installed": ["SciPy", "Gmsh", "meshio", "triangle"],
        },
        "adaptive_levels": LEVELS,
        "maximum_p2_dofs": MAXIMUM_P2_DOFS,
        "default_execution_p2_dofs": DEFAULT_EXECUTION_P2_DOFS,
        "resource_policy_revision": {
            "accuracy_gates_relaxed": False,
            "minimum_third_level_free_ram_bytes": (
                MINIMUM_THIRD_LEVEL_FREE_RAM_BYTES
            ),
            "one_design_at_a_time": True,
            "explicit_third_level_opt_in": True,
            "preflight": resource_preflight,
        },
        "dorfler_theta": DORFLER_THETA,
        "maximum_adjacent_size_growth": MAXIMUM_ADJACENT_SIZE_GROWTH,
        "padding_factor": PADDING_FACTOR,
        "domain_expansion_evidence": {
            "required": True,
            "phase_matched_fixed_local_h": True,
            "padding_factors": [0.5, 1.0, 1.5],
            "maximum_qoi_relative_change": 0.01,
            "status": (
                "completed"
                if arguments.allow_third_level
                else "not_run_screening_only"
            ),
        },
        "designs": entries,
        "wall_seconds": perf_counter() - campaign_started,
        "less_than_one_percent_all_designs": all(
            entry["convergence"]["less_than_one_percent_reached"] for entry in entries
        ),
        "diagnostic_policy": {
            "timing_and_memory": "DIAGNOSTIC_ONLY",
            "hardware_validation": False,
        },
        "limitations": [
            "Independent numerical reference, not hardware validation.",
            "Unmet convergence or grading gates retain screening status.",
            "Local cell/interface maxima remain screening-only.",
        ],
    }
    manifest = {
        **payload,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "payload_sha256": sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest(),
        },
    }
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
