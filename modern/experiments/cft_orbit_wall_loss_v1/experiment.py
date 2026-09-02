"""Preregistered full-orbit campaign mechanics and shared-runtime callbacks."""

from __future__ import annotations

import gzip
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file
from cft_revival.orbit_mc import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    EstimatorPolicy,
    OrbitConfig,
    Termination,
    analytic_magnetic_bottle,
    backend_parity,
    build_launch_ensemble,
    checkpoint,
    compare_maps,
    coupling_v42_handoff,
    frozen_batch_manifest,
    load_and_verify_artifact,
    merge_checkpoint_results,
    reduce_results,
    result_artifact,
    timestep_convergence,
    uniform_b_helix,
    varying_e_convergence,
    wall_event_accuracy,
    wilson_interval,
    write_artifact,
    write_checkpoint,
)
from cft_revival.orbit_mc.artifacts import content_hash
from cft_revival.orbit_mc.integrator import integrate_orbit

from .adapter import build_regular_field, file_sha256

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
LAUNCH_MANIFEST_PATH = EXPERIMENT / "launch-manifest.json"
BATCH_MANIFEST_PATH = EXPERIMENT / "batch-manifest.json"
SYNTHETIC_PREFLIGHT_PATH = EXPERIMENT / "synthetic-preflight.json"


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    return value


def build_launches(value: Mapping[str, Any]) -> tuple[Any, ...]:
    declaration = value["launches"]
    positions = [
        (item["flux_surface_id"], tuple(item["position_m"]))
        for item in declaration["position_seeds"]
    ]
    return build_launch_ensemble(
        ensemble_id=declaration["ensemble_id"],
        energies_ev=declaration["energies_ev"],
        pitch_angles_rad=[
            math.radians(item) for item in declaration["pitch_angles_deg"]
        ],
        positions=positions,
        directions=declaration["directions"],
        gyrophase_count=declaration["gyrophase_count"],
    )


def launch_records(launches: Sequence[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in sorted(launches, key=lambda item: item.launch_id)]


def runtime_launch_records(launches: Sequence[Any]) -> list[dict[str, Any]]:
    records = launch_records(launches)
    for record in records:
        record["seed_id"] = str(record["seed_id"])
    return records


def batch_records(value: Mapping[str, Any], launches: Sequence[Any]) -> list[dict[str, Any]]:
    return frozen_batch_manifest(
        launches,
        batch_size=int(value["launches"]["batch_size"]),
        estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
    )


def manufactured_gate_report(value: Mapping[str, Any]) -> dict[str, Any]:
    limits = value["gates"]
    helix = timestep_convergence()
    varying = varying_e_convergence()
    mirror = analytic_magnetic_bottle()
    energy = uniform_b_helix()
    wall = wall_event_accuracy()
    cpu = backend_parity(device="cpu")
    cuda = backend_parity(device="cuda:0")
    checks = {
        "uniform_b_energy": energy["relative_energy_error"]
        <= limits["maximum_relative_energy_error"],
        "helix_order": min(helix["observed_orders"])
        >= limits["minimum_helix_position_order"],
        "varying_e_order": min(varying["observed_orders"])
        >= limits["minimum_varying_e_position_order"],
        "mirror_smoke": mirror["relative_error"]
        <= limits["maximum_mirror_point_relative_error"],
        "wall_endpoint": wall["endpoint_error_m"]
        <= limits["maximum_wall_endpoint_error_m"],
        "cpu_parity": cpu["status"] == "evaluated"
        and cpu["maximum_relative_velocity_difference"]
        <= limits["maximum_cpu_cuda_relative_velocity_difference"],
        "cuda_parity": cuda["status"] == "evaluated"
        and cuda["maximum_relative_velocity_difference"]
        <= limits["maximum_cpu_cuda_relative_velocity_difference"],
    }
    return _plain({
        "checks": checks,
        "passed": all(checks.values()),
        "uniform_b": energy,
        "helix_convergence": helix,
        "varying_e_convergence": varying,
        "mirror": mirror,
        "wall_event": wall,
        "cpu_parity": cpu,
        "cuda_parity": cuda,
    })


def validate_frozen_input_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    authority = value["authority"]
    manifest_path = REPOSITORY / authority["manifest"]["path"]
    result_path = REPOSITORY / authority["result"]["path"]
    if (
        file_sha256(manifest_path) != authority["manifest"]["file_sha256"]
        or file_sha256(result_path) != authority["result"]["file_sha256"]
    ):
        raise ValueError("frozen P2 manifest/result bytes differ")
    manifest = strict_json_file(manifest_path)
    rows = [
        item
        for item in manifest["designs"]
        if item["config_id"] == "divergent-exit-stack-v1"
    ]
    if (
        len(rows) != 1
        or rows[0]["qualification_status"] != authority["required_qualification"]
        or set(authority["excluded_designs"])
        != {"historical-envelope-baseline", "compact-high-gradient-stack"}
    ):
        raise ValueError("P2 qualification/design authority differs")
    return {
        "manifest_file_sha256": authority["manifest"]["file_sha256"],
        "result_file_sha256": authority["result"]["file_sha256"],
        "qualification_status": rows[0]["qualification_status"],
        "design_id": authority["design_id"],
    }


def orbit_config(value: Mapping[str, Any], role: str, timestep: str) -> OrbitConfig:
    del role
    declaration = value["orbit"]
    wall = declaration["wall"]
    domain = declaration["domain"]
    return OrbitConfig(
        wall_radius_m=wall["radius_m"],
        wall_z_min_m=wall["z_min_m"],
        wall_z_max_m=wall["z_max_m"],
        domain_radius_m=domain["radius_m"],
        domain_z_min_m=domain["z_min_m"],
        domain_z_max_m=domain["z_max_m"],
        max_time_s=declaration["max_time_s"],
        max_path_m=declaration["max_path_m"],
        max_steps=declaration["max_steps"],
        max_rotation_rad=declaration["timestep_policies"][timestep][
            "max_rotation_rad"
        ],
        event_tolerance_m=declaration["event_tolerance_m"],
        maximum_gamma=declaration["maximum_gamma"],
    )


def _probability(successes: int, trials: int) -> dict[str, Any]:
    return asdict(wilson_interval(successes, trials))


def stratum_summaries(
    launches: Sequence[Any], results: Sequence[Any]
) -> list[dict[str, Any]]:
    authority = {item.launch_id: item for item in launches}
    groups: dict[tuple[Any, ...], list[Any]] = {}
    for result in results:
        launch = authority[result.launch_id]
        cell_id = launch.flux_surface_id.split("-r", 1)[0]
        key = (
            cell_id,
            launch.kinetic_energy_ev,
            round(math.degrees(launch.pitch_angle_rad), 12),
            launch.parallel_direction,
        )
        groups.setdefault(key, []).append(result)
    output = []
    for key in sorted(groups):
        rows = groups[key]
        counts = {
            termination.value: sum(item.termination is termination for item in rows)
            for termination in Termination
        }
        n = len(rows)
        timeout = n - counts["wall_hit"] - counts["reflected"] - counts["domain_escape"]
        repeat_ids = {
            authority[item.launch_id].flux_surface_id for item in rows
        }
        output.append(
            {
                "cell_id": key[0],
                "kinetic_energy_ev": key[1],
                "pitch_angle_deg": key[2],
                "parallel_direction": key[3],
                "trials": n,
                "physical_position_repeat_count": len(repeat_ids),
                "termination_counts": counts,
                "wall_hit": _probability(counts["wall_hit"], n),
                "reflected": _probability(counts["reflected"], n),
                "domain_escape": _probability(counts["domain_escape"], n),
                "timeout": _probability(timeout, n),
            }
        )
    return output


def _interval_overlap(left: Any, right: Any) -> bool:
    return max(left.wall_hit.lower, right.wall_hit.lower) <= min(
        left.wall_hit.upper, right.wall_hit.upper
    )


def _convergence(
    summaries: Mapping[tuple[str, str], Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    maximum = value["gates"]["maximum_successive_probability_change"]
    timestep_rows = []
    for role in ("primary", "refined", "enlarged"):
        ordered = [summaries[(role, step)] for step in ("N", "2N", "4N")]
        changes = [
            abs(right.wall_hit.probability - left.wall_hit.probability)
            for left, right in zip(ordered, ordered[1:], strict=True)
        ]
        overlaps = [
            _interval_overlap(left, right)
            for left, right in zip(ordered, ordered[1:], strict=True)
        ]
        timestep_rows.append(
            {
                "map_role": role,
                "probabilities": [item.wall_hit.probability for item in ordered],
                "successive_changes": changes,
                "adjacent_wilson_overlap": overlaps,
                "passed": max(changes) <= maximum and all(overlaps),
            }
        )
    map_rows = []
    for step in ("N", "2N", "4N"):
        ordered = [summaries[(role, step)] for role in ("primary", "refined", "enlarged")]
        changes = [
            abs(right.wall_hit.probability - left.wall_hit.probability)
            for left, right in zip(ordered, ordered[1:], strict=True)
        ]
        overlaps = [
            _interval_overlap(left, right)
            for left, right in zip(ordered, ordered[1:], strict=True)
        ]
        map_rows.append(
            {
                "timestep_policy": step,
                "probabilities": [item.wall_hit.probability for item in ordered],
                "successive_changes": changes,
                "adjacent_wilson_overlap": overlaps,
                "passed": max(changes) <= maximum and all(overlaps),
            }
        )
    return {
        "timestep": timestep_rows,
        "cross_map": map_rows,
        "timestep_passed": all(item["passed"] for item in timestep_rows),
        "cross_map_passed": all(item["passed"] for item in map_rows),
    }


def _result_gate_report(
    campaigns: Mapping[tuple[str, str], Mapping[str, Any]],
    convergence: Mapping[str, Any],
    manufactured: Mapping[str, Any],
    field_evidence: Mapping[str, Mapping[str, Any]],
    map_comparisons: Mapping[str, Mapping[str, Any]],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    all_results = [
        item for campaign in campaigns.values() for item in campaign["results"]
    ]
    incomplete = {
        termination.value: sum(item.termination is termination for item in all_results)
        for termination in (
            Termination.PATH_TIMEOUT,
            Termination.TIME_TIMEOUT,
            Termination.STEP_LIMIT,
            Termination.NONFINITE_STATE,
            Termination.EXTREME_RELATIVITY,
            Termination.FIELD_FAILURE,
            Termination.INITIAL_STATE_INVALID,
        )
    }
    wall_errors = [
        abs(math.hypot(*item.wall_endpoint_m[:2]) - value["orbit"]["wall"]["radius_m"])
        for item in all_results
        if item.wall_endpoint_m is not None
    ]
    witness_order = all(
        item.event_witness["event_fraction"]
        <= min(
            [
                candidate
                for candidate in item.event_witness["candidate_fractions"].values()
                if candidate is not None
            ]
            or [item.event_witness["event_fraction"]]
        )
        + 64.0 * np.finfo(float).eps
        for item in all_results
        if "candidate_fractions" in item.event_witness
    )
    maximum_energy = max(item.maximum_relative_energy_error for item in all_results)
    checks = {
        "manufactured": manufactured["passed"],
        "field_adapter": all(item["passed"] for item in field_evidence.values()),
        "field_map_convergence": all(
            item["b_relative_rms"]
            <= value["field_adapter"]["maximum_cross_map_b_relative_rms"]
            for item in map_comparisons.values()
        ),
        "timestep_probability_convergence": convergence["timestep_passed"],
        "cross_map_probability_convergence": convergence["cross_map_passed"],
        "zero_incomplete_or_numerical_failures": sum(incomplete.values()) == 0,
        "energy": maximum_energy <= value["gates"]["maximum_relative_energy_error"],
        "wall_endpoint": max(wall_errors, default=0.0)
        <= value["gates"]["maximum_wall_endpoint_error_m"],
        "earliest_event": witness_order,
        "runtime_rotation": all(
            item.dt_s
            * abs(ELECTRON_CHARGE_C)
            * campaign["field"].max_b_t
            / ELECTRON_MASS_KG
            <= campaign["config"].max_rotation_rad * (1.0 + 1.0e-14)
            for campaign in campaigns.values()
            for item in campaign["results"]
        ),
        "relativistic_phase": all(
            math.isfinite(item.accumulated_gyro_phase_rad)
            and math.isfinite(float(item.event_witness.get("observed_gamma", 1.0)))
            for item in all_results
        ),
        "material_quarantine": all(
            bool(np.all(campaign["field"].traversable_cells))
            for campaign in campaigns.values()
        ),
        "independent_repeats": all(
            row["physical_position_repeat_count"]
            >= value["launches"]["independent_repeats_per_stratum"]
            for campaign in campaigns.values()
            for row in campaign["strata"]
        ),
    }
    return {
        "checks": checks,
        "passed_before_replay": all(checks.values()),
        "incomplete_and_failure_counts": incomplete,
        "maximum_relative_energy_error": maximum_energy,
        "maximum_wall_endpoint_error_m": max(wall_errors, default=0.0),
    }


def callbacks() -> RuntimeCallbacks:
    value = protocol()
    authorities = strict_json_file(AUTHORITIES_PATH)
    frozen_launches = strict_json_file(LAUNCH_MANIFEST_PATH)
    frozen_batches = strict_json_file(BATCH_MANIFEST_PATH)
    synthetic = strict_json_file(SYNTHETIC_PREFLIGHT_PATH)
    launches = build_launches(value)
    batches = batch_records(value, launches)
    state: dict[str, Any] = {}

    def prebundle(context: Any) -> Mapping[str, Any]:
        if semantic_sha256(value) != authorities["protocol_semantic_sha256"]:
            raise ValueError("protocol semantic authority differs")
        if runtime_launch_records(launches) != frozen_launches["launches"]:
            raise ValueError("launch manifest differs from preregistered authority")
        if batches != frozen_batches["batches"]:
            raise ValueError("batch manifest differs from preregistered authority")
        if not synthetic["passed"] or synthetic["orbit_outcome_access_count"] != 0:
            raise ValueError("synthetic production preflight is invalid")
        input_authority = validate_frozen_input_authority(value)
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/authorities.json", authorities)
        context.write_json("artifacts/launch-manifest.json", frozen_launches)
        context.write_json("artifacts/batch-manifest.json", frozen_batches)
        context.write_json("artifacts/synthetic-preflight.json", synthetic)
        context.write_json("artifacts/p2-input-authority.json", input_authority)
        context.write_json(
            "artifacts/runtime.json",
            {
                "generated_at_utc": datetime.now(timezone.utc),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        )
        state["launches"] = launches
        state["batches"] = batches
        return {
            "preregistered": True,
            "launch_count": len(launches),
            "batch_count": len(batches),
            "p2_authority": input_authority,
        }

    def development(context: Any) -> Decision:
        manufactured = manufactured_gate_report(value)
        context.write_json("artifacts/manufactured-gates.json", manufactured)
        fields: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        serialized: dict[str, Any] = {}
        for role in ("primary", "refined", "enlarged"):
            context.before_expensive(
                f"p2-adapter-{role}",
                kind="solver",
                details={"role": role, "design": "divergent-exit-stack"},
            )
            fields[role], evidence[role], serialized[role] = build_regular_field(
                REPOSITORY, value, role
            )
            context.write_json(f"artifacts/fields/{role}.json", serialized[role])
            context.write_json(
                f"artifacts/field-evidence/{role}.json", evidence[role]
            )
        comparisons = {
            "primary_to_refined": compare_maps(fields["primary"], fields["refined"]),
            "refined_to_enlarged": compare_maps(fields["refined"], fields["enlarged"]),
        }
        for item in comparisons.values():
            item["b_relative_rms"] = item["b_rms_t"] / max(
                fields["primary"].max_b_t,
                fields["refined"].max_b_t,
                fields["enlarged"].max_b_t,
                np.finfo(float).tiny,
            )
        context.write_json("artifacts/field-map-convergence.json", comparisons)
        accepted = bool(
            manufactured["passed"]
            and all(item["passed"] for item in evidence.values())
            and all(
                item["b_relative_rms"]
                <= value["field_adapter"]["maximum_cross_map_b_relative_rms"]
                for item in comparisons.values()
            )
        )
        state.update(
            {
                "manufactured": manufactured,
                "fields": fields,
                "field_evidence": evidence,
                "serialized_fields": serialized,
                "map_comparisons": comparisons,
            }
        )
        return Decision(
            accepted,
            {
                "manufactured_passed": manufactured["passed"],
                "field_adapter_passed": all(item["passed"] for item in evidence.values()),
                "map_adapter_convergence_passed": all(
                    item["b_relative_rms"]
                    <= value["field_adapter"]["maximum_cross_map_b_relative_rms"]
                    for item in comparisons.values()
                ),
            },
        )

    def assessment(context: Any) -> Decision:
        campaigns: dict[tuple[str, str], dict[str, Any]] = {}
        launch_sha = content_hash(launch_records(launches))
        batch_sha = content_hash(
            {
                "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
                "batches": batches,
            }
        )
        for role in ("primary", "refined", "enlarged"):
            field = state["fields"][role]
            field_sha = content_hash(state["serialized_fields"][role])
            for timestep in ("N", "2N", "4N"):
                campaign_id = f"{value['experiment_id']}:{role}:{timestep}"
                config = orbit_config(value, role, timestep)
                config_sha = content_hash(asdict(config))
                policy_sha = content_hash(
                    {
                        "protocol_semantic_sha256": authorities[
                            "protocol_semantic_sha256"
                        ],
                        "role": role,
                        "timestep": timestep,
                    }
                )
                context.before_expensive(
                    f"orbit-{role}-{timestep}",
                    kind="label",
                    details={
                        "campaign_id": campaign_id,
                        "launch_count": len(launches),
                        "sequential_batches": True,
                    },
                )
                results: list[Any] = []
                partial_payload = None
                partial_hash = None
                ordered = sorted(launches, key=lambda item: item.launch_id)
                for batch in batches:
                    batch_launches = [
                        next(item for item in ordered if item.launch_id == entry["launch_id"])
                        for entry in batch["launches"]
                    ]
                    for item in batch_launches:
                        results.append(integrate_orbit(item, field, config))
                        if (
                            batch["batch_id"] == 0
                            and len(results)
                            == value["execution"]["partial_checkpoint_prefix_count"]
                        ):
                            partial_payload = checkpoint(
                                campaign_id,
                                (),
                                launches,
                                results,
                                batches,
                                field_identity_sha256=field_sha,
                                config_identity_sha256=config_sha,
                                policy_identity_sha256=policy_sha,
                                minimum_certificate_tightness_ratio_authority=value[
                                    "gates"
                                ]["minimum_certificate_dense_to_bound_ratio"],
                                estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                                expected_batch_manifest_sha256=batch_sha,
                                partial_current_batch={
                                    "batch_id": 0,
                                    "completed_launch_ids": [
                                        entry["launch_id"]
                                        for entry in batches[0]["launches"][
                                            : value["execution"][
                                                "partial_checkpoint_prefix_count"
                                            ]
                                        ]
                                    ],
                                },
                            )
                            temp = context.cache_root / f"{role}-{timestep}-partial.json"
                            partial_hash = write_checkpoint(
                                temp,
                                partial_payload,
                                expected_campaign_id=campaign_id,
                                expected_launches_sha256=launch_sha,
                                expected_batch_manifest_sha256=batch_sha,
                                expected_policy_sha256=policy_sha,
                                expected_estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                                expected_minimum_certificate_tightness_ratio=value[
                                    "gates"
                                ]["minimum_certificate_dense_to_bound_ratio"],
                            )
                            context.write_blob(
                                f"artifacts/checkpoints/{role}-{timestep}-partial.json.gz",
                                gzip.compress(temp.read_bytes(), mtime=0),
                            )
                summary = reduce_results(campaign_id, results)
                strata = stratum_summaries(launches, results)
                final_payload = checkpoint(
                    campaign_id,
                    tuple(range(len(batches))),
                    launches,
                    results,
                    batches,
                    field_identity_sha256=field_sha,
                    config_identity_sha256=config_sha,
                    policy_identity_sha256=policy_sha,
                    minimum_certificate_tightness_ratio_authority=value["gates"][
                        "minimum_certificate_dense_to_bound_ratio"
                    ],
                    estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                    expected_batch_manifest_sha256=batch_sha,
                    previous_checkpoint_sha256=content_hash(partial_payload),
                )
                merged_records = merge_checkpoint_results(
                    partial_payload,
                    final_payload,
                    expected_campaign_id=campaign_id,
                    expected_launches_sha256=launch_sha,
                    expected_batch_manifest_sha256=batch_sha,
                    expected_policy_sha256=policy_sha,
                    expected_estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                    expected_minimum_certificate_tightness_ratio=value["gates"][
                        "minimum_certificate_dense_to_bound_ratio"
                    ],
                )
                if len(merged_records) != len(results):
                    raise RuntimeError("partial/final checkpoint chain coverage differs")
                temp_final = context.cache_root / f"{role}-{timestep}-final.json"
                final_hash = write_checkpoint(
                    temp_final,
                    final_payload,
                    expected_campaign_id=campaign_id,
                    expected_launches_sha256=launch_sha,
                    expected_batch_manifest_sha256=batch_sha,
                    expected_policy_sha256=policy_sha,
                    expected_estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                    expected_minimum_certificate_tightness_ratio=value["gates"][
                        "minimum_certificate_dense_to_bound_ratio"
                    ],
                )
                context.write_blob(
                    f"artifacts/checkpoints/{role}-{timestep}-final.json.gz",
                    gzip.compress(temp_final.read_bytes(), mtime=0),
                )
                context.write_json(
                    f"artifacts/summaries/{role}-{timestep}.json",
                    {
                        "summary": summary.to_dict(),
                        "strata": strata,
                        "partial_checkpoint_file_sha256": partial_hash,
                        "final_checkpoint_file_sha256": final_hash,
                    },
                )
                campaigns[(role, timestep)] = {
                    "field": field,
                    "field_sha": field_sha,
                    "config": config,
                    "config_sha": config_sha,
                    "policy_sha": policy_sha,
                    "results": tuple(results),
                    "summary": summary,
                    "strata": strata,
                }
        summaries = {key: item["summary"] for key, item in campaigns.items()}
        convergence = _convergence(summaries, value)
        context.write_json("artifacts/probability-convergence.json", convergence)
        gates = _result_gate_report(
            campaigns,
            convergence,
            state["manufactured"],
            state["field_evidence"],
            state["map_comparisons"],
            value,
        )
        replay_count = 0
        handoff = None
        if gates["passed_before_replay"]:
            for (role, timestep), campaign in campaigns.items():
                field_evidence = state["field_evidence"][role]
                artifact = result_artifact(
                    campaign_id=campaign["summary"].ensemble_id,
                    field_identity_sha256=campaign["field_sha"],
                    config_identity_sha256=campaign["config_sha"],
                    policy_identity_sha256=campaign["policy_sha"],
                    minimum_certificate_tightness_ratio_authority=value["gates"][
                        "minimum_certificate_dense_to_bound_ratio"
                    ],
                    estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                    launches=launches,
                    results=campaign["results"],
                    batch_manifest=batches,
                    summary=campaign["summary"],
                    interpolation_evidence={
                        "certified_max_b_t": campaign["field"].certified_max_b_t,
                        "reference_max_b_t": campaign["field"].reference_max_b_t,
                        "runtime_max_seen_t": max(
                            item.maximum_b_t for item in campaign["results"]
                        ),
                        "dense_diagnostic_max_b_t": campaign[
                            "field"
                        ].certificate_tightness.dense_diagnostic_max_b_t,
                        "certificate_tightness_ratio": campaign[
                            "field"
                        ].certificate_tightness.dense_to_bound_ratio,
                        "minimum_certificate_tightness_ratio": value["gates"][
                            "minimum_certificate_dense_to_bound_ratio"
                        ],
                        "certificate_preflight_passed": campaign[
                            "field"
                        ].certificate_tightness.preflight_passed,
                        "material_map_sha256": campaign["field"].material_map_sha256,
                        "field_error_report": field_evidence["field_error_report"],
                        "passed": True,
                    },
                    convergence_evidence={
                        "timestep_passed": True,
                        "cross_map_passed": True,
                        "backend_parity_passed": True,
                    },
                    preregistration={
                        "protocol_id": value["schema_version"],
                        "frozen_before_outcomes": True,
                        "held_out_geometry_status": "passed",
                    },
                )
                target = context.cache_root / f"{role}-{timestep}-orbit.json"
                evidence = write_artifact(
                    target,
                    artifact,
                    field=campaign["field"],
                    config=campaign["config"],
                    expected_field_sha256=campaign["field_sha"],
                    expected_config_sha256=campaign["config_sha"],
                    expected_launches_sha256=launch_sha,
                    expected_batch_manifest_sha256=batch_sha,
                    expected_policy_sha256=campaign["policy_sha"],
                    expected_estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                    expected_minimum_certificate_tightness_ratio=value["gates"][
                        "minimum_certificate_dense_to_bound_ratio"
                    ],
                )
                verified = load_and_verify_artifact(
                    target,
                    field=campaign["field"],
                    config=campaign["config"],
                    expected_file_sha256=evidence.file_sha256,
                    expected_field_sha256=campaign["field_sha"],
                    expected_config_sha256=campaign["config_sha"],
                    expected_launches_sha256=launch_sha,
                    expected_batch_manifest_sha256=batch_sha,
                    expected_policy_sha256=campaign["policy_sha"],
                    expected_estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                    expected_minimum_certificate_tightness_ratio=value["gates"][
                        "minimum_certificate_dense_to_bound_ratio"
                    ],
                )
                replay_count += 1
                context.write_blob(
                    f"artifacts/orbits/{role}-{timestep}.json.gz",
                    gzip.compress(target.read_bytes(), mtime=0),
                )
                context.write_blob(
                    f"artifacts/orbits/{role}-{timestep}.json.sha256",
                    target.with_name(target.name + ".sha256").read_bytes(),
                )
                if role == "refined" and timestep == "4N":
                    handoff = coupling_v42_handoff(
                        verified, expected_batch_manifest_sha256=batch_sha
                    )
        gates["exact_authority_replay_count"] = replay_count
        gates["exact_authority_replay"] = replay_count == len(campaigns)
        gates["passed"] = gates["passed_before_replay"] and gates[
            "exact_authority_replay"
        ]
        context.write_json("artifacts/gates.json", gates)
        if handoff is not None:
            context.write_json("artifacts/coupling-export-only.json", handoff)
        totals = {
            f"{role}-{timestep}": {
                "trial_count": item["summary"].trial_count,
                "termination_counts": dict(item["summary"].termination_counts),
                "wall_hit": asdict(item["summary"].wall_hit),
                "reflected": asdict(item["summary"].reflected),
                "escaped": asdict(item["summary"].escaped),
                "incomplete": asdict(item["summary"].incomplete),
            }
            for (role, timestep), item in campaigns.items()
        }
        terminal = {
            "status": "accepted" if gates["passed"] else "rejected",
            "classification": value["classification"],
            "launch_count": len(launches),
            "orbit_count": sum(item["summary"].trial_count for item in campaigns.values()),
            "campaign_count": len(campaigns),
            "campaigns": totals,
            "gates": gates,
            "coupling": (
                "export_only_pending_consumer_integration"
                if handoff is not None
                else "not_exported_failed_gates"
            ),
            "limitations": value["publication_boundary"],
        }
        context.write_json("artifacts/campaign-result.json", terminal)
        return Decision(gates["passed"], terminal)

    return RuntimeCallbacks(prebundle, development, assessment)
