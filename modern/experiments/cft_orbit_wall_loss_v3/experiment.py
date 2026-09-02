"""Preregistered full-orbit campaign mechanics and shared-runtime callbacks."""

from __future__ import annotations

import gzip
import hashlib
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import (
    CanonicalizationError,
    canonical_bytes,
    semantic_sha256,
    strict_json_file,
    strict_json_loads,
)
from cft_revival.orbit_mc import (
    AnalyticField,
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    ElectronLaunch,
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
CASE_AUTHORITIES_PATH = EXPERIMENT / "case-authorities.json"
CASE_ROOT = EXPERIMENT / "cases"
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


def case_id(value: Mapping[str, Any], role: str, timestep: str) -> str:
    if role not in ("primary", "refined", "enlarged"):
        raise ValueError("unknown field-map role")
    if timestep not in ("N", "2N", "4N"):
        raise ValueError("unknown timestep policy")
    return f"{value['experiment_id']}:{role}:{timestep}"


def case_key(role: str, timestep: str) -> str:
    return f"{role}-{timestep}"


def case_matrix(value: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (role, timestep, case_id(value, role, timestep))
        for role in ("primary", "refined", "enlarged")
        for timestep in ("N", "2N", "4N")
    )


def build_case_launches(
    value: Mapping[str, Any], role: str, timestep: str
) -> tuple[ElectronLaunch, ...]:
    declaration = value["launches"]
    positions = [
        (item["flux_surface_id"], tuple(item["position_m"]))
        for item in declaration["position_seeds"]
    ]
    count = int(declaration["gyrophase_count"])
    offset = float(declaration["gyrophase_offset_rad"])
    phases = tuple(
        (offset + 2.0 * math.pi * index / count) % (2.0 * math.pi)
        for index in range(count)
    )
    return build_launch_ensemble(
        ensemble_id=case_id(value, role, timestep),
        energies_ev=declaration["energies_ev"],
        pitch_angles_rad=[
            math.radians(item) for item in declaration["pitch_angles_deg"]
        ],
        positions=positions,
        directions=declaration["directions"],
        gyrophases_rad=phases,
    )


def launch_records(launches: Sequence[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in sorted(launches, key=lambda item: item.launch_id)]


def runtime_launch_payload(
    campaign_id: str, launches: Sequence[Any]
) -> dict[str, Any]:
    records = launch_records(launches)
    for record in records:
        record["seed_id"] = str(record["seed_id"])
    return {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v3.launches/3.0.0",
        "campaign_id": campaign_id,
        "ensemble_id": campaign_id,
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "seed_encoding": "unsigned-64 decimal string",
        "launches": records,
    }


def batch_records(value: Mapping[str, Any], launches: Sequence[Any]) -> list[dict[str, Any]]:
    return frozen_batch_manifest(
        launches,
        batch_size=int(value["launches"]["batch_size"]),
        estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
    )


def runtime_batch_payload(
    value: Mapping[str, Any], campaign_id: str, launches: Sequence[Any]
) -> dict[str, Any]:
    return {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v3.batches/3.0.0",
        "campaign_id": campaign_id,
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "batches": batch_records(value, launches),
    }


def estimator_identity(
    launches: Sequence[Any], batches: Sequence[Mapping[str, Any]]
) -> str:
    weights = {
        entry["launch_id"]: entry["weight"]
        for batch in batches
        for entry in batch["launches"]
    }
    return content_hash(
        {
            "estimand_id": "campaign_wall_loss_probability",
            "policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "launches": [
                {"launch_id": launch.launch_id, "weight": weights[launch.launch_id]}
                for launch in sorted(launches, key=lambda item: item.launch_id)
            ],
        }
    )


def _decode_runtime_tags(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_runtime_tags(item) for item in value]
    if isinstance(value, dict):
        if "__cft_type__" in value:
            if set(value) != {"__cft_type__", "items"} or value["__cft_type__"] != "tuple":
                raise ValueError("unsupported or malformed reserved runtime tag")
            if not isinstance(value["items"], list):
                raise ValueError("runtime tuple tag items must be a list")
            return tuple(_decode_runtime_tags(item) for item in value["items"])
        return {key: _decode_runtime_tags(item) for key, item in value.items()}
    return value


def load_runtime_launch_payload(
    data: bytes, expected_campaign_id: str
) -> tuple[ElectronLaunch, ...]:
    """Closed typed loader used only after exact byte authority succeeds."""

    decoded = _decode_runtime_tags(strict_json_loads(data))
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema_version",
        "campaign_id",
        "ensemble_id",
        "estimator_policy",
        "seed_encoding",
        "launches",
    }:
        raise ValueError("runtime launch payload is not closed")
    if (
        decoded["schema_version"]
        != "cft-revival.cft-orbit-wall-loss-v3.launches/3.0.0"
        or decoded["campaign_id"] != expected_campaign_id
        or decoded["ensemble_id"] != expected_campaign_id
        or decoded["estimator_policy"] != EstimatorPolicy.UNWEIGHTED_BINOMIAL.value
        or decoded["seed_encoding"] != "unsigned-64 decimal string"
        or not isinstance(decoded["launches"], list)
    ):
        raise ValueError("runtime launch payload authority differs")
    launches: list[ElectronLaunch] = []
    expected_keys = {
        "launch_id",
        "seed_id",
        "kinetic_energy_ev",
        "pitch_angle_rad",
        "position_m",
        "parallel_direction",
        "gyrophase_rad",
        "flux_surface_id",
    }
    for record in decoded["launches"]:
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ValueError("runtime launch record is not closed")
        seed_text = record["seed_id"]
        if (
            not isinstance(seed_text, str)
            or not seed_text
            or not seed_text.isascii()
            or not seed_text.isdecimal()
        ):
            raise ValueError("runtime launch seed is not an unsigned decimal string")
        seed = int(seed_text)
        if seed > 2**64 - 1 or str(seed) != seed_text:
            raise ValueError("runtime launch seed is outside canonical uint64")
        position = record["position_m"]
        if not isinstance(position, tuple) or len(position) != 3:
            raise ValueError("runtime launch position did not reconstruct as a tuple")
        launches.append(
            ElectronLaunch(
                launch_id=record["launch_id"],
                seed_id=seed,
                kinetic_energy_ev=record["kinetic_energy_ev"],
                pitch_angle_rad=record["pitch_angle_rad"],
                position_m=position,
                parallel_direction=record["parallel_direction"],
                gyrophase_rad=record["gyrophase_rad"],
                flux_surface_id=record["flux_surface_id"],
            )
        )
    ordered = tuple(sorted(launches, key=lambda item: item.launch_id))
    if len({item.launch_id for item in ordered}) != len(ordered):
        raise ValueError("runtime launch IDs are not unique")
    if any(not item.launch_id.startswith(expected_campaign_id + ":") for item in ordered):
        raise ValueError("runtime launch ID is not case-prefixed")
    return ordered


def field_identity(value: Mapping[str, Any], role: str) -> str:
    declaration = value["field_adapter"]["maps"][role]
    return content_hash(
        {
            "role": role,
            "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
            "sidecar_file_sha256": declaration["sidecar_file_sha256"],
            "mesh_sha256": declaration["mesh_sha256"],
            "run_sha256": declaration["run_sha256"],
        }
    )


def policy_identity(value: Mapping[str, Any], role: str, timestep: str) -> str:
    return content_hash(
        {
            "protocol_semantic_sha256": semantic_sha256(value),
            "role": role,
            "timestep": timestep,
        }
    )


def build_case_authority(
    value: Mapping[str, Any], role: str, timestep: str
) -> dict[str, Any]:
    campaign = case_id(value, role, timestep)
    launches = build_case_launches(value, role, timestep)
    batches = batch_records(value, launches)
    launch_bytes = canonical_bytes(runtime_launch_payload(campaign, launches))
    batch_bytes = canonical_bytes(runtime_batch_payload(value, campaign, launches))
    record = {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v3.case-authority/3.0.0",
        "case_key": case_key(role, timestep),
        "campaign_id": campaign,
        "ensemble_id": campaign,
        "role": role,
        "timestep": timestep,
        "launch_count": len(launches),
        "batch_count": len(batches),
        "launch_manifest_path": (
            f"cases/{case_key(role, timestep)}/launch-manifest.json"
        ),
        "batch_manifest_path": (
            f"cases/{case_key(role, timestep)}/batch-manifest.json"
        ),
        "runtime_launch_payload_byte_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "runtime_batch_payload_byte_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "orbit_launches_sha256": content_hash(launch_records(launches)),
        "batch_manifest_sha256": content_hash(
            {
                "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
                "batches": batches,
            }
        ),
        "estimator_sha256": estimator_identity(launches, batches),
        "field_identity_sha256": field_identity(value, role),
        "config_identity_sha256": content_hash(
            asdict(orbit_config(value, role, timestep))
        ),
        "policy_identity_sha256": policy_identity(value, role, timestep),
    }
    record["case_authority_sha256"] = content_hash(
        {
            "campaign_id": campaign,
            "launches_sha256": record["orbit_launches_sha256"],
            "batch_manifest_sha256": record["batch_manifest_sha256"],
            "policy_sha256": record["policy_identity_sha256"],
            "minimum_certificate_tightness_ratio": value["gates"][
                "minimum_certificate_dense_to_bound_ratio"
            ],
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "estimator_sha256": record["estimator_sha256"],
            "replay_requirement": "deterministic_full_result_replay_required",
        }
    )
    record["authority_record_sha256"] = content_hash(record)
    return record


def build_all_case_authorities(value: Mapping[str, Any]) -> dict[str, Any]:
    cases = [
        build_case_authority(value, role, timestep)
        for role, timestep, _ in case_matrix(value)
    ]
    return {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v3.case-authorities/3.0.0",
        "case_count": len(cases),
        "total_case_launches": sum(item["launch_count"] for item in cases),
        "cases": cases,
    }


def _synthetic_checkpoint_chain(
    value: Mapping[str, Any],
    authority: Mapping[str, Any],
    launches: Sequence[ElectronLaunch],
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    synthetic_config = OrbitConfig(
        wall_radius_m=0.0009,
        wall_z_min_m=0.001,
        wall_z_max_m=0.022,
        domain_radius_m=0.001,
        domain_z_min_m=0.0,
        domain_z_max_m=0.024,
        max_time_s=1.0e-12,
        max_path_m=1.0e-6,
        max_steps=2,
        max_rotation_rad=0.16,
        event_tolerance_m=1.0e-9,
        maximum_gamma=20.0,
    )
    synthetic_field = AnalyticField(
        lambda _position: np.array([0.0, 0.0, 0.1]),
        None,
        0.1,
    )
    results = tuple(
        integrate_orbit(item, synthetic_field, synthetic_config)
        for item in launches
    )
    if any(item.termination is not Termination.INITIAL_STATE_INVALID for item in results):
        raise ValueError("synthetic result was not immediate initial-state termination")
    common = {
        "field_identity_sha256": authority["field_identity_sha256"],
        "config_identity_sha256": authority["config_identity_sha256"],
        "policy_identity_sha256": authority["policy_identity_sha256"],
        "minimum_certificate_tightness_ratio_authority": value["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        "expected_batch_manifest_sha256": authority["batch_manifest_sha256"],
    }
    partial = checkpoint(
        authority["campaign_id"],
        (),
        launches,
        results[:32],
        batches,
        partial_current_batch={
            "batch_id": 0,
            "completed_launch_ids": [
                entry["launch_id"] for entry in batches[0]["launches"][:32]
            ],
        },
        **common,
    )
    resumed = checkpoint(
        authority["campaign_id"],
        (0,),
        launches,
        results[:64],
        batches,
        previous_checkpoint_sha256=content_hash(partial),
        **common,
    )
    final = checkpoint(
        authority["campaign_id"],
        tuple(range(8)),
        launches,
        results,
        batches,
        previous_checkpoint_sha256=content_hash(resumed),
        **common,
    )
    external = {
        "expected_campaign_id": authority["campaign_id"],
        "expected_launches_sha256": authority["orbit_launches_sha256"],
        "expected_batch_manifest_sha256": authority["batch_manifest_sha256"],
        "expected_policy_sha256": authority["policy_identity_sha256"],
        "expected_estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        "expected_minimum_certificate_tightness_ratio": value["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
    }
    merged_resume = merge_checkpoint_results(partial, resumed, **external)
    merged_final = merge_checkpoint_results(resumed, final, **external)
    checks = {
        "partial_32": partial["coverage"]["completed_launches"] == 32,
        "resumed_64": len(merged_resume) == 64,
        "final_512": len(merged_final) == 512,
        "no_pending_final": final["pending_launch_ids"] == [],
        "case_authority_bound": (
            final["authority"]["campaign_identity_sha256"]
            == authority["case_authority_sha256"]
        ),
    }
    return {
        "campaign_id": authority["campaign_id"],
        "partial_checkpoint_sha256": content_hash(partial),
        "resumed_checkpoint_sha256": content_hash(resumed),
        "final_checkpoint_sha256": content_hash(final),
        "checks": checks,
        "passed": all(checks.values()),
    }


def production_synthetic_preflight(
    value: Mapping[str, Any], case_authorities: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate all nine authorities and checkpoint chains without P2 access."""

    first_authority = case_authorities["cases"][0]
    first_launches = build_case_launches(
        value, first_authority["role"], first_authority["timestep"]
    )
    first_batches = batch_records(value, first_launches)
    sample = first_launches[0]
    matrix = {
        "launch": runtime_launch_payload(first_authority["campaign_id"], (sample,)),
        "final_state": {
            "position_m": (0.0012, -2.0e-6, 0.003),
            "velocity_m_per_s": (1.0, 2.0, 3.0),
        },
        "wall_endpoint_m": (0.002, 0.0, 0.004),
        "event_witness": {
            "step_start_position_m": (0.001, 0.0, 0.003),
            "step_end_position_m": (0.0021, 0.0, 0.0032),
            "step_start_velocity_m_per_s": (1.0, 0.0, 2.0),
            "step_end_velocity_m_per_s": (0.5, 0.5, 2.0),
            "candidate_fractions": {
                "wall_hit": 0.9,
                "reflected": None,
                "domain_escape": None,
                "time_timeout": None,
                "path_timeout": None,
            },
            "reflection_bracket": None,
        },
        "gyro_averages": (
            {
                "cycle_index": 0,
                "phase_interval_rad": (0.0, 2.0 * math.pi),
                "mu_j_per_t": 1.0e-18,
            },
        ),
        "termination_counts": (
            ("wall_hit", 1),
            ("reflected", 2),
            ("domain_escape", 3),
            ("time_timeout", 4),
        ),
        "batch_checkpoint_ids": {
            "completed_batch_ids": (0, 1),
            "pending_launch_ids": [sample.launch_id],
            "checkpoint_chain": ("partial", "final"),
        },
        "p2_field_arrays": {
            "r_m": [0.0, 0.001, 0.002],
            "z_m": [0.001, 0.002, 0.003],
            "psi_wb": [[0.0, 0.0], [1.0e-8, 2.0e-8]],
            "material_id": [["plasma", "plasma"], ["plasma", "plasma"]],
        },
        "protocol_vectors": {
            "directions": tuple(value["launches"]["directions"]),
            "first_position": tuple(value["launches"]["position_seeds"][0]["position_m"]),
            "timestep_levels": list(value["orbit"]["timestep_policies"]),
        },
    }
    encoded = canonical_bytes(matrix)
    decoded = _decode_runtime_tags(strict_json_loads(encoded))
    if decoded != matrix:
        raise ValueError("synthetic production vector roundtrip differs")
    chain_reports = []
    all_v3_ids: set[str] = set()
    all_v3_seeds: set[int] = set()
    v3_positions: set[tuple[float, float, float]] = set()
    v3_phase_space: set[tuple[Any, ...]] = set()
    payload_checks = []
    for authority in case_authorities["cases"]:
        launches = build_case_launches(
            value, authority["role"], authority["timestep"]
        )
        batches = batch_records(value, launches)
        launch_bytes = canonical_bytes(
            runtime_launch_payload(authority["campaign_id"], launches)
        )
        batch_payload = runtime_batch_payload(
            value, authority["campaign_id"], launches
        )
        batch_bytes = canonical_bytes(batch_payload)
        loaded = load_runtime_launch_payload(
            launch_bytes, authority["campaign_id"]
        )
        payload_ok = (
            loaded == tuple(sorted(launches, key=lambda item: item.launch_id))
            and strict_json_loads(batch_bytes) == batch_payload
            and hashlib.sha256(launch_bytes).hexdigest()
            == authority["runtime_launch_payload_byte_sha256"]
            and hashlib.sha256(batch_bytes).hexdigest()
            == authority["runtime_batch_payload_byte_sha256"]
            and estimator_identity(launches, batches)
            == authority["estimator_sha256"]
        )
        payload_checks.append(payload_ok)
        chain_reports.append(
            _synthetic_checkpoint_chain(value, authority, launches, batches)
        )
        all_v3_ids.update(item.launch_id for item in launches)
        all_v3_seeds.update(item.seed_id for item in launches)
        v3_positions.update(item.position_m for item in launches)
        v3_phase_space.update(
            (
                item.kinetic_energy_ev,
                item.pitch_angle_rad,
                item.position_m,
                item.parallel_direction,
                item.gyrophase_rad,
            )
            for item in launches
        )
    old_positions = (
        ("cell-1-r0.60", (0.0012, 0.0, 0.003)),
        ("cell-1-r0.825", (0.00165, 0.0, 0.003)),
        ("cell-2-r0.60", (0.0012, 0.0, 0.009)),
        ("cell-2-r0.825", (0.00165, 0.0, 0.009)),
        ("cell-3-r0.60", (0.0012, 0.0, 0.015)),
        ("cell-3-r0.825", (0.00165, 0.0, 0.015)),
        ("cell-4-r0.60", (0.0012, 0.0, 0.021)),
        ("cell-4-r0.825", (0.00165, 0.0, 0.021)),
    )
    v2_launches = build_launch_ensemble(
        ensemble_id="divergent-exit-qualified-p2-wall-loss-v2",
        energies_ev=value["launches"]["energies_ev"],
        pitch_angles_rad=tuple(
            math.radians(item) for item in value["launches"]["pitch_angles_deg"]
        ),
        positions=old_positions,
        directions=value["launches"]["directions"],
        gyrophase_count=8,
    )
    v2_ids = {item.launch_id for item in v2_launches}
    v2_seeds = {item.seed_id for item in v2_launches}
    v2_positions = {item.position_m for item in v2_launches}
    v2_phase_space = {
        (
            item.kinetic_energy_ev,
            item.pitch_angle_rad,
            item.position_m,
            item.parallel_direction,
            item.gyrophase_rad,
        )
        for item in v2_launches
    }
    reserved_input_rejected = False
    try:
        canonical_bytes({"__cft_type__": "tuple", "items": []})
    except CanonicalizationError:
        reserved_input_rejected = True
    tagged_parse = strict_json_loads(canonical_bytes((1, 2, 3)))
    tagged_reencode_rejected = False
    try:
        canonical_bytes(tagged_parse)
    except CanonicalizationError:
        tagged_reencode_rejected = True
    malformed_tag_rejected = False
    try:
        _decode_runtime_tags(
            {"__cft_type__": "tuple", "items": [], "unexpected": True}
        )
    except ValueError:
        malformed_tag_rejected = True
    checks = {
        "all_vector_fields_roundtrip": True,
        "all_case_payloads_roundtrip": all(payload_checks),
        "all_nine_checkpoint_chains": (
            len(chain_reports) == 9
            and all(item["passed"] for item in chain_reports)
        ),
        "all_case_ids_and_seeds_unique": (
            len(all_v3_ids) == 9 * 512 and len(all_v3_seeds) == 9 * 512
        ),
        "zero_v2_identity_overlap": not (all_v3_ids & v2_ids),
        "zero_v2_seed_overlap": not (all_v3_seeds & v2_seeds),
        "zero_v2_position_overlap": not (v3_positions & v2_positions),
        "zero_v2_phase_space_overlap": not (v3_phase_space & v2_phase_space),
        "reserved_input_rejected": reserved_input_rejected,
        "parsed_tag_reencode_rejected": tagged_reencode_rejected,
        "malformed_tag_rejected": malformed_tag_rejected,
    }
    return {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v3.synthetic-production-preflight/3.0.0",
        "covered_fields": [
            "launch_position_and_seed",
            "final_position_and_velocity",
            "wall_endpoint",
            "event_witness_vectors_and_candidates",
            "gyro_averages",
            "termination_counts",
            "batch_and_checkpoint_ids",
            "p2_field_arrays",
            "protocol_vectors",
        ],
        "matrix_byte_sha256": hashlib.sha256(encoded).hexdigest(),
        "case_checkpoint_chains": chain_reports,
        "overlap_evidence": {
            "v2_identity_overlap_count": len(all_v3_ids & v2_ids),
            "v2_seed_overlap_count": len(all_v3_seeds & v2_seeds),
            "v2_position_overlap_count": len(v3_positions & v2_positions),
            "v2_phase_space_overlap_count": len(v3_phase_space & v2_phase_space),
            "v3_unique_case_launch_ids": len(all_v3_ids),
            "v3_unique_case_seed_ids": len(all_v3_seeds),
            "v3_unique_physical_positions": len(v3_positions),
            "v3_unique_physical_phase_space_points_per_case": len(v3_phase_space),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


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
    frozen_case_authorities = strict_json_file(CASE_AUTHORITIES_PATH)
    synthetic = strict_json_file(SYNTHETIC_PREFLIGHT_PATH)
    state: dict[str, Any] = {}

    def prebundle(context: Any) -> Mapping[str, Any]:
        if semantic_sha256(value) != authorities["protocol_semantic_sha256"]:
            raise ValueError("protocol semantic authority differs")
        expected_case_authorities = build_all_case_authorities(value)
        if (
            frozen_case_authorities != expected_case_authorities
            or semantic_sha256(frozen_case_authorities)
            != authorities["case_authorities_sha256"]
        ):
            raise ValueError("case authorities differ from preregistration")
        if (
            not synthetic["passed"]
            or synthetic["p2_field_access_count"] != 0
            or synthetic["orbit_outcome_access_count"] != 0
        ):
            raise ValueError("synthetic production preflight is invalid")
        cases: dict[tuple[str, str], dict[str, Any]] = {}
        for authority in frozen_case_authorities["cases"]:
            role = authority["role"]
            timestep = authority["timestep"]
            campaign = authority["campaign_id"]
            launches = build_case_launches(value, role, timestep)
            batches = batch_records(value, launches)
            launch_path = EXPERIMENT / authority["launch_manifest_path"]
            batch_path = EXPERIMENT / authority["batch_manifest_path"]
            actual_launch_bytes = launch_path.read_bytes()
            actual_batch_bytes = batch_path.read_bytes()
            expected_launch_bytes = canonical_bytes(
                runtime_launch_payload(campaign, launches)
            )
            expected_batch_bytes = canonical_bytes(
                runtime_batch_payload(value, campaign, launches)
            )
            if (
                actual_launch_bytes != expected_launch_bytes
                or hashlib.sha256(actual_launch_bytes).hexdigest()
                != authority["runtime_launch_payload_byte_sha256"]
                or load_runtime_launch_payload(actual_launch_bytes, campaign)
                != tuple(sorted(launches, key=lambda item: item.launch_id))
            ):
                raise ValueError(f"{campaign} launch-byte authority differs")
            if (
                actual_batch_bytes != expected_batch_bytes
                or hashlib.sha256(actual_batch_bytes).hexdigest()
                != authority["runtime_batch_payload_byte_sha256"]
                or content_hash(launch_records(launches))
                != authority["orbit_launches_sha256"]
                or content_hash(
                    {
                        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
                        "batches": batches,
                    }
                )
                != authority["batch_manifest_sha256"]
                or estimator_identity(launches, batches)
                != authority["estimator_sha256"]
            ):
                raise ValueError(f"{campaign} batch/estimator authority differs")
            context.write_blob(
                f"artifacts/{authority['launch_manifest_path']}",
                actual_launch_bytes,
            )
            context.write_blob(
                f"artifacts/{authority['batch_manifest_path']}",
                actual_batch_bytes,
            )
            cases[(role, timestep)] = {
                "authority": authority,
                "launches": launches,
                "batches": batches,
            }
        input_authority = validate_frozen_input_authority(value)
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/authorities.json", authorities)
        context.write_json(
            "artifacts/case-authorities.json", frozen_case_authorities
        )
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
        state["cases"] = cases
        return {
            "preregistered": True,
            "case_count": len(cases),
            "total_case_launches": sum(
                len(item["launches"]) for item in cases.values()
            ),
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
        for role in ("primary", "refined", "enlarged"):
            field = state["fields"][role]
            for timestep in ("N", "2N", "4N"):
                frozen_case = state["cases"][(role, timestep)]
                authority = frozen_case["authority"]
                launches = frozen_case["launches"]
                batches = frozen_case["batches"]
                campaign_id = authority["campaign_id"]
                launch_sha = authority["orbit_launches_sha256"]
                batch_sha = authority["batch_manifest_sha256"]
                field_sha = authority["field_identity_sha256"]
                config = orbit_config(value, role, timestep)
                config_sha = content_hash(asdict(config))
                policy_sha = policy_identity(value, role, timestep)
                if (
                    state["field_evidence"][role]["source_identity_sha256"]
                    != field_sha
                    or config_sha != authority["config_identity_sha256"]
                    or policy_sha != authority["policy_identity_sha256"]
                    or estimator_identity(launches, batches)
                    != authority["estimator_sha256"]
                ):
                    raise ValueError(f"{campaign_id} execution authority differs")
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
                latest_payload = None
                checkpoint_hashes: list[str] = []
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
                            latest_payload = partial_payload
                    if latest_payload is None:
                        raise RuntimeError("partial checkpoint was not created")
                    batch_id = int(batch["batch_id"])
                    batch_payload = checkpoint(
                        campaign_id,
                        tuple(range(batch_id + 1)),
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
                        previous_checkpoint_sha256=content_hash(latest_payload),
                    )
                    merged_records = merge_checkpoint_results(
                        latest_payload,
                        batch_payload,
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
                        raise RuntimeError("sequential checkpoint coverage differs")
                    temp_batch = (
                        context.cache_root
                        / f"{role}-{timestep}-batch-{batch_id:02d}.json"
                    )
                    checkpoint_hashes.append(
                        write_checkpoint(
                            temp_batch,
                            batch_payload,
                            expected_campaign_id=campaign_id,
                            expected_launches_sha256=launch_sha,
                            expected_batch_manifest_sha256=batch_sha,
                            expected_policy_sha256=policy_sha,
                            expected_estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                            expected_minimum_certificate_tightness_ratio=value[
                                "gates"
                            ]["minimum_certificate_dense_to_bound_ratio"],
                        )
                    )
                    context.write_blob(
                        (
                            f"artifacts/checkpoints/{role}-{timestep}-"
                            f"batch-{batch_id:02d}.json.gz"
                        ),
                        gzip.compress(temp_batch.read_bytes(), mtime=0),
                    )
                    latest_payload = batch_payload
                summary = reduce_results(campaign_id, results)
                strata = stratum_summaries(launches, results)
                if latest_payload is None or latest_payload["pending_launch_ids"]:
                    raise RuntimeError("final checkpoint is not complete")
                final_hash = checkpoint_hashes[-1]
                context.write_json(
                    f"artifacts/summaries/{role}-{timestep}.json",
                    {
                        "summary": summary.to_dict(),
                        "strata": strata,
                        "partial_checkpoint_file_sha256": partial_hash,
                        "final_checkpoint_file_sha256": final_hash,
                        "sequential_batch_checkpoint_file_sha256": checkpoint_hashes,
                    },
                )
                campaigns[(role, timestep)] = {
                    "authority": authority,
                    "launches": launches,
                    "batches": batches,
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
                case_launches = campaign["launches"]
                case_batches = campaign["batches"]
                case_authority = campaign["authority"]
                artifact = result_artifact(
                    campaign_id=campaign["summary"].ensemble_id,
                    field_identity_sha256=campaign["field_sha"],
                    config_identity_sha256=campaign["config_sha"],
                    policy_identity_sha256=campaign["policy_sha"],
                    minimum_certificate_tightness_ratio_authority=value["gates"][
                        "minimum_certificate_dense_to_bound_ratio"
                    ],
                    estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
                    launches=case_launches,
                    results=campaign["results"],
                    batch_manifest=case_batches,
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
                    expected_launches_sha256=case_authority[
                        "orbit_launches_sha256"
                    ],
                    expected_batch_manifest_sha256=case_authority[
                        "batch_manifest_sha256"
                    ],
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
                    expected_launches_sha256=case_authority[
                        "orbit_launches_sha256"
                    ],
                    expected_batch_manifest_sha256=case_authority[
                        "batch_manifest_sha256"
                    ],
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
                        verified,
                        expected_batch_manifest_sha256=case_authority[
                            "batch_manifest_sha256"
                        ],
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
            "launches_per_case": 512,
            "total_case_launch_count": sum(
                len(item["launches"]) for item in campaigns.values()
            ),
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
