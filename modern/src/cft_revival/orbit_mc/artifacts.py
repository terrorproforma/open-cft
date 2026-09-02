"""Canonical orbit evidence, checkpoints, identities, and coupling handoff."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import floor, isfinite, pi, sqrt
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ensemble import EnsembleSummary, result_records_identity, wilson_interval
from .models import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    LIGHT_SPEED_M_PER_S,
    AxisymmetricField,
    ElectronLaunch,
    EstimatorPolicy,
    OrbitConfig,
    OrbitResult,
    OrbitValidationError,
    Termination,
)

SCHEMA_VERSION = "cft-revival-orbit-mc-result/1.4.0"
CHECKPOINT_VERSION = "cft-revival-orbit-mc-checkpoint/1.4.0"
HANDOFF_VERSION = "cft-revival-orbit-mc-coupling-v4.2/1.3.0"
CLASSIFICATION = "test_particle_wall_loss_not_self_consistent_plasma"
REPLAY_REQUIREMENT = "deterministic_full_result_replay_required"
ESTIMAND_ID = "campaign_wall_loss_probability"
_VERIFICATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class UnverifiedOrbitArtifact:
    """Structurally validated evidence that is intentionally not publication-capable."""

    _payload_bytes: bytes
    file_sha256: str

    def _materialize(self) -> dict[str, object]:
        return json.loads(self._payload_bytes.decode("utf-8"))


@dataclass(frozen=True, slots=True, init=False)
class VerifiedOrbitEvidence:
    """Opaque evidence token created only by deterministic replay."""

    _payload_bytes: bytes
    file_sha256: str

    def __init__(
        self, payload_bytes: bytes, file_sha256: str, verification_seal: object
    ) -> None:
        if verification_seal is not _VERIFICATION_SEAL:
            raise OrbitValidationError(
                "verified evidence can only be created by deterministic replay"
            )
        object.__setattr__(self, "_payload_bytes", bytes(payload_bytes))
        object.__setattr__(self, "file_sha256", _hash("file_sha256", file_sha256))

    def _materialize(self) -> dict[str, object]:
        return json.loads(self._payload_bytes.decode("utf-8"))

    @property
    def campaign_id(self) -> str:
        return str(self._materialize()["campaign_id"])


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise OrbitValidationError("artifact is not canonical finite JSON") from error


def content_hash(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def code_identity() -> str:
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def frozen_batch_manifest(
    launches: Sequence[ElectronLaunch],
    *,
    batch_size: int,
    weights: Mapping[str, float] | None = None,
    estimator_policy: EstimatorPolicy = EstimatorPolicy.UNWEIGHTED_BINOMIAL,
) -> list[dict[str, object]]:
    if estimator_policy is not EstimatorPolicy.UNWEIGHTED_BINOMIAL:
        raise OrbitValidationError("weighted/stratified estimators are unsupported")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise OrbitValidationError("batch_size must be a positive integer")
    ordered = sorted(launches, key=lambda item: item.launch_id)
    if not ordered:
        raise OrbitValidationError("batch manifest requires launches")
    default_weight = 1.0 / len(ordered)
    manifest: list[dict[str, object]] = []
    for batch_id, start in enumerate(range(0, len(ordered), batch_size)):
        entries = []
        for order, launch in enumerate(ordered[start:start+batch_size]):
            weight = default_weight if weights is None else weights.get(launch.launch_id)
            if weight is None:
                raise OrbitValidationError("every launch requires a frozen weight")
            entries.append(
                {
                    "launch_id": launch.launch_id,
                    "order": order,
                    "weight": _real(
                        "launch weight", weight, minimum=0.0, exclusive_minimum=True
                    ),
                }
            )
        manifest.append({"batch_id": batch_id, "launches": entries})
    _validate_batch_manifest(
        manifest,
        [launch.launch_id for launch in ordered],
        estimator_policy.value,
    )
    return manifest


def _validate_batch_manifest(
    manifest: object,
    launch_ids: Sequence[str],
    estimator_policy: object,
) -> tuple[dict[int, tuple[str, ...]], dict[str, float]]:
    if estimator_policy != EstimatorPolicy.UNWEIGHTED_BINOMIAL.value:
        raise OrbitValidationError("only unweighted binomial estimation is supported")
    if not isinstance(manifest, list) or not manifest:
        raise OrbitValidationError("batch manifest must be nonempty")
    batches: dict[int, tuple[str, ...]] = {}
    weights: dict[str, float] = {}
    for expected_batch, batch in enumerate(manifest):
        if not isinstance(batch, Mapping) or set(batch) != {"batch_id", "launches"}:
            raise OrbitValidationError("batch manifest record is not closed")
        batch_id = batch["batch_id"]
        entries = batch["launches"]
        if (
            isinstance(batch_id, bool)
            or not isinstance(batch_id, int)
            or batch_id != expected_batch
            or not isinstance(entries, list)
            or not entries
        ):
            raise OrbitValidationError("batch manifest IDs/entries are invalid")
        ids: list[str] = []
        for expected_order, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) != {
                "launch_id", "order", "weight"
            }:
                raise OrbitValidationError("batch launch entry is not closed")
            launch_id = entry["launch_id"]
            if (
                not isinstance(launch_id, str)
                or entry["order"] != expected_order
                or isinstance(entry["order"], bool)
                or not isinstance(entry["order"], int)
                or launch_id in weights
            ):
                raise OrbitValidationError("batch launch identity/order is invalid")
            ids.append(launch_id)
            weights[launch_id] = _real(
                "batch launch weight",
                entry["weight"],
                minimum=0.0,
                exclusive_minimum=True,
            )
        batches[batch_id] = tuple(ids)
    flattened = [identity for batch in batches.values() for identity in batch]
    if flattened != list(launch_ids):
        raise OrbitValidationError("batch manifest does not exactly cover launch authority")
    if abs(sum(weights.values()) - 1.0) > 1.0e-12:
        raise OrbitValidationError("batch launch weights must sum to one")
    expected_weight = 1.0 / len(weights)
    if any(
        abs(weight - expected_weight)
        > 64.0*np.finfo(float).eps*max(1.0, expected_weight)
        for weight in weights.values()
    ):
        raise OrbitValidationError(
            "unweighted binomial estimator requires equal launch weights"
        )
    return batches, weights


def _estimator_identity(
    launch_ids: Sequence[str], weights: Mapping[str, float], policy: str
) -> str:
    return content_hash(
        {
            "estimand_id": ESTIMAND_ID,
            "policy": policy,
            "launches": [
                {"launch_id": launch_id, "weight": weights[launch_id]}
                for launch_id in sorted(launch_ids)
            ],
        }
    )


def _result_dict(
    result: OrbitResult,
    *,
    field_identity_sha256: str | None = None,
    config_identity_sha256: str | None = None,
    policy_identity_sha256: str | None = None,
) -> dict[str, object]:
    value = asdict(result)
    value["termination"] = result.termination.value
    if field_identity_sha256 is not None:
        value["event_witness"]["field_identity_sha256"] = field_identity_sha256
    if config_identity_sha256 is not None:
        value["event_witness"]["config_identity_sha256"] = config_identity_sha256
    if policy_identity_sha256 is not None:
        value["event_witness"]["policy_identity_sha256"] = policy_identity_sha256
    return value


def result_artifact(
    *,
    campaign_id: str,
    field_identity_sha256: str,
    config_identity_sha256: str,
    policy_identity_sha256: str,
    minimum_certificate_tightness_ratio_authority: float,
    estimator_policy: EstimatorPolicy,
    launches: Sequence[ElectronLaunch],
    results: Sequence[OrbitResult],
    batch_manifest: Sequence[Mapping[str, object]],
    summary: EnsembleSummary,
    interpolation_evidence: Mapping[str, object],
    convergence_evidence: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    if not campaign_id:
        raise OrbitValidationError("campaign_id must be non-empty")
    ordered_launches = sorted(launches, key=lambda item: item.launch_id)
    ordered_results = sorted(results, key=lambda item: item.launch_id)
    launch_records = [asdict(launch) for launch in ordered_launches]
    if estimator_policy is not EstimatorPolicy.UNWEIGHTED_BINOMIAL:
        raise OrbitValidationError("weighted/stratified estimators are unsupported")
    launch_ids = [str(record["launch_id"]) for record in launch_records]
    _, weights = _validate_batch_manifest(
        list(batch_manifest), launch_ids, estimator_policy.value
    )
    estimator_identity = _estimator_identity(
        launch_ids, weights, estimator_policy.value
    )
    batch_identity = content_hash(
        {
            "estimator_policy": estimator_policy.value,
            "batches": batch_manifest,
        }
    )
    result_records = [
        _result_dict(
            result,
            field_identity_sha256=field_identity_sha256,
            config_identity_sha256=config_identity_sha256,
            policy_identity_sha256=policy_identity_sha256,
        )
        for result in ordered_results
    ]
    summary_record = summary.to_dict()
    summary_record["result_identity_sha256"] = result_records_identity(
        result_records
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "campaign_id": campaign_id,
        "identities": {
            "field_sha256": field_identity_sha256,
            "config_sha256": config_identity_sha256,
            "policy_sha256": policy_identity_sha256,
            "code_sha256": code_identity(),
            "launches_sha256": content_hash(launch_records),
            "results_sha256": content_hash(result_records),
            "batch_manifest_sha256": batch_identity,
            "estimator_sha256": estimator_identity,
        },
        "estimator": {
            "policy": estimator_policy.value,
            "estimand_id": ESTIMAND_ID,
            "equal_launch_weights_required": True,
        },
        "verification": {
            "requirement": REPLAY_REQUIREMENT,
            "structural_status": "unverified_pending_deterministic_replay",
        },
        "launches": launch_records,
        "batch_manifest": list(batch_manifest),
        "results": result_records,
        "summary": summary_record,
        "interpolation_evidence": dict(interpolation_evidence),
        "convergence_evidence": dict(convergence_evidence),
        "preregistration": dict(preregistration),
        "limitations": [
            "Test-particle trajectories only; no self-consistent electric field.",
            "No collisions, space charge, sheath, plasma response, or PIC claim.",
            "Direct first-wall/reflection/escape outcomes are authoritative within this model.",
            "Any loss-cone value is an asymptotic comparator gated by adiabatic diagnostics.",
        ],
    }
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "payload_sha256": content_hash(payload),
    }
    validate_result_artifact(
        payload,
        expected_policy_sha256=policy_identity_sha256,
        expected_estimator_policy=estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            minimum_certificate_tightness_ratio_authority
        ),
    )
    return payload


def _hash(name: str, value: object) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OrbitValidationError(f"{name} must be lowercase SHA-256")
    return value


def _real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OrbitValidationError(f"{name} must be a real scalar")
    converted = float(value)
    if not isfinite(converted):
        raise OrbitValidationError(f"{name} must be finite")
    if minimum is not None and (
        converted <= minimum if exclusive_minimum else converted < minimum
    ):
        qualifier = "greater than" if exclusive_minimum else "at least"
        raise OrbitValidationError(f"{name} must be {qualifier} {minimum}")
    if maximum is not None and converted > maximum:
        raise OrbitValidationError(f"{name} must be at most {maximum}")
    return converted


def _validate_probability(
    name: str, estimate: object, successes: int, trials: int
) -> None:
    if not isinstance(estimate, Mapping) or set(estimate) != {
        "successes", "trials", "probability", "lower", "upper", "method"
    }:
        raise OrbitValidationError(f"summary {name} probability keys are invalid")
    if any(
        isinstance(estimate[key], bool) or not isinstance(estimate[key], int)
        for key in ("successes", "trials")
    ):
        raise OrbitValidationError(f"summary {name} counts must be integers")
    expected = wilson_interval(successes, trials)
    if (
        estimate["successes"] != successes
        or estimate["trials"] != trials
        or estimate["method"] != expected.method
    ):
        raise OrbitValidationError(f"summary {name} counts/method are inconsistent")
    values = tuple(
        _real(
            f"summary {name}.{key}",
            estimate[key],
            minimum=0.0,
            maximum=1.0,
        )
        for key in ("lower", "probability", "upper")
    )
    if (
        not values[0] <= values[1] <= values[2]
        or values
        != (expected.lower, expected.probability, expected.upper)
    ):
        raise OrbitValidationError(
            f"summary {name} confidence interval is not the exact Wilson result"
        )


def _validate_event_witness(
    result: Mapping[str, object],
    *,
    launch: Mapping[str, object],
    expected_field_sha256: str,
    expected_config_sha256: str,
    expected_policy_sha256: str,
    allow_replay_dependent_failures: bool,
) -> None:
    witness = result["event_witness"]
    witness_keys = {
        "kind", "config", "step_start_position_m", "step_end_position_m",
        "step_start_velocity_m_per_s", "step_end_velocity_m_per_s",
        "event_fraction", "candidate_fractions", "reflection_bracket",
        "start_elapsed_time_s", "start_path_length_m", "step_dt_s",
        "step_segment_length_m", "step_index", "condition",
        "observed_gamma", "observed_speed2_over_c2",
        "field_identity_sha256", "config_identity_sha256",
        "policy_identity_sha256",
    }
    config_keys = {
        "wall_radius_m", "wall_z_min_m", "wall_z_max_m", "domain_radius_m",
        "domain_z_min_m", "domain_z_max_m", "max_time_s", "max_path_m",
        "max_steps", "max_rotation_rad", "event_tolerance_m",
        "maximum_gamma", "fixed_dt_s",
    }
    candidate_keys = {
        "time_timeout", "path_timeout", "wall_hit", "reflected",
        "domain_escape",
    }
    if not isinstance(witness, Mapping) or set(witness) != witness_keys:
        raise OrbitValidationError("event witness is not closed")
    if (
        witness["kind"] != result["termination"]
        or witness["field_identity_sha256"] != expected_field_sha256
        or witness["config_identity_sha256"] != expected_config_sha256
        or witness["policy_identity_sha256"] != expected_policy_sha256
        or not isinstance(witness["condition"], str)
        or not witness["condition"]
    ):
        raise OrbitValidationError("event witness identity/kind is inconsistent")
    config_record = witness["config"]
    if not isinstance(config_record, Mapping) or set(config_record) != config_keys:
        raise OrbitValidationError("event witness config is not closed")
    try:
        config = OrbitConfig(**config_record)
    except (TypeError, ValueError) as error:
        raise OrbitValidationError("event witness config is invalid") from error
    if (
        config.max_time_s != result["configured_max_time_s"]
        or config.max_path_m != result["configured_max_path_m"]
        or config.event_tolerance_m != result["event_tolerance_m"]
    ):
        raise OrbitValidationError("event witness policy differs from result")
    vectors: dict[str, np.ndarray] = {}
    for name in (
        "step_start_position_m", "step_end_position_m",
        "step_start_velocity_m_per_s", "step_end_velocity_m_per_s",
    ):
        value = witness[name]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not isfinite(float(item))
                for item in value
            )
        ):
            raise OrbitValidationError(f"event witness {name} is invalid")
        vectors[name] = np.asarray(value, dtype=np.float64)
    fraction = _real(
        "event witness fraction", witness["event_fraction"],
        minimum=0.0, maximum=1.0,
    )
    start_time = _real(
        "event witness start time", witness["start_elapsed_time_s"], minimum=0.0
    )
    start_path = _real(
        "event witness start path", witness["start_path_length_m"], minimum=0.0
    )
    step_dt = _real("event witness step dt", witness["step_dt_s"], minimum=0.0)
    segment = _real(
        "event witness segment length",
        witness["step_segment_length_m"],
        minimum=0.0,
    )
    step_index = witness["step_index"]
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
        or step_index != result["steps"]
    ):
        raise OrbitValidationError("event witness step index is invalid")
    candidates = witness["candidate_fractions"]
    if not isinstance(candidates, Mapping) or set(candidates) != candidate_keys:
        raise OrbitValidationError("event candidate set is not closed")
    parsed_candidates: dict[str, float | None] = {}
    for name in candidate_keys:
        value = candidates[name]
        parsed_candidates[name] = (
            None if value is None else _real(
                f"event candidate {name}", value, minimum=0.0, maximum=1.0
            )
        )
    termination = str(result["termination"])
    failure_kinds = {
        Termination.INITIAL_STATE_INVALID.value,
        Termination.FIELD_FAILURE.value,
        Termination.NONFINITE_STATE.value,
        Termination.EXTREME_RELATIVITY.value,
    }
    if termination in failure_kinds:
        if any(value is not None for value in parsed_candidates.values()):
            raise OrbitValidationError("failure witness cannot claim event candidates")
        if termination in {
            Termination.FIELD_FAILURE.value,
            Termination.NONFINITE_STATE.value,
        } and not allow_replay_dependent_failures:
            raise OrbitValidationError(
                "field/nonfinite termination requires deterministic replay"
            )
        if termination == Termination.INITIAL_STATE_INVALID.value:
            launch_position = np.asarray(launch["position_m"], dtype=np.float64)
            launch_radius = float(np.hypot(launch_position[0], launch_position[1]))
            outside = (
                launch_radius >= config.domain_radius_m
                or not config.domain_z_min_m < launch_position[2] < config.domain_z_max_m
                or (
                    config.wall_z_min_m <= launch_position[2] <= config.wall_z_max_m
                    and launch_radius >= config.wall_radius_m
                )
            )
            if witness["condition"] == "launch_outside_geometry":
                if not outside:
                    raise OrbitValidationError("initial geometry witness is false")
            elif (
                witness["condition"] != "invalid_initial_field"
                or not allow_replay_dependent_failures
            ):
                raise OrbitValidationError(
                    "initial field failure requires deterministic replay"
                )
        if termination == Termination.EXTREME_RELATIVITY.value:
            gamma = witness["observed_gamma"]
            speed_ratio = witness["observed_speed2_over_c2"]
            gamma_exceeded = (
                gamma is not None
                and _real("observed gamma", gamma, minimum=1.0)
                > config.maximum_gamma
            )
            speed_exceeded = (
                speed_ratio is not None
                and _real("observed speed ratio", speed_ratio, minimum=0.0) >= 1.0
            )
            if not gamma_exceeded and not speed_exceeded:
                raise OrbitValidationError("extreme-relativity witness lacks threshold crossing")
        return
    if step_dt <= 0.0 or step_index < 1:
        raise OrbitValidationError("physical event witness requires a positive step")
    start = vectors["step_start_position_m"]
    end = vectors["step_end_position_m"]
    measured_segment = float(np.linalg.norm(end - start))
    tolerance = max(config.event_tolerance_m, 256.0*np.finfo(float).eps)
    if abs(segment - measured_segment) > tolerance:
        raise OrbitValidationError("event witness segment length is inconsistent")
    from .integrator import _geometry_event_candidates
    geometric = _geometry_event_candidates(start, end, config)
    expected_geometry = {"wall_hit": None, "domain_escape": None}
    for candidate_fraction, _, kind, _, _ in geometric:
        current = expected_geometry[kind.value]
        if current is None or candidate_fraction < current:
            expected_geometry[kind.value] = candidate_fraction
    for name, expected in expected_geometry.items():
        observed = parsed_candidates[name]
        if (expected is None) != (observed is None) or (
            expected is not None and abs(expected - observed) > 1.0e-12
        ):
            raise OrbitValidationError("geometric event witness does not replay")
    remaining_time = config.max_time_s - start_time
    expected_time = (
        max(0.0, min(1.0, remaining_time / step_dt))
        if remaining_time <= step_dt
        else None
    )
    remaining_path = config.max_path_m - start_path
    expected_path = (
        max(0.0, min(1.0, remaining_path / segment))
        if segment > 0.0 and remaining_path <= segment
        else None
    )
    for name, expected in (
        ("time_timeout", expected_time),
        ("path_timeout", expected_path),
    ):
        observed = parsed_candidates[name]
        if (expected is None) != (observed is None) or (
            expected is not None and abs(expected - observed) > 1.0e-12
        ):
            raise OrbitValidationError("deadline/path event witness does not replay")
    bracket = witness["reflection_bracket"]
    if bracket is None:
        if parsed_candidates["reflected"] is not None:
            raise OrbitValidationError("reflection candidate lacks a root bracket")
    else:
        bracket_keys = {
            "low_fraction", "high_fraction", "low_parallel_m_per_s",
            "high_parallel_m_per_s", "root_fraction", "root_parallel_m_per_s",
        }
        if not isinstance(bracket, Mapping) or set(bracket) != bracket_keys:
            raise OrbitValidationError("reflection bracket is not closed")
        low = _real("reflection low fraction", bracket["low_fraction"], minimum=0.0, maximum=1.0)
        high = _real("reflection high fraction", bracket["high_fraction"], minimum=0.0, maximum=1.0)
        root = _real("reflection root fraction", bracket["root_fraction"], minimum=0.0, maximum=1.0)
        low_value = _real("reflection low parallel", bracket["low_parallel_m_per_s"])
        high_value = _real("reflection high parallel", bracket["high_parallel_m_per_s"])
        root_value = _real("reflection root parallel", bracket["root_parallel_m_per_s"])
        if (
            not low <= root <= high
            or low_value <= 0.0
            or high_value > 0.0
            or abs(root_value) > max(1.0e-6, abs(low_value-high_value))
            or parsed_candidates["reflected"] != root
        ):
            raise OrbitValidationError("reflection root witness is inconsistent")
    priorities = {
        "time_timeout": 0, "path_timeout": 1, "wall_hit": 2,
        "reflected": 3, "domain_escape": 4,
    }
    available = [
        (value, priorities[name], name)
        for name, value in parsed_candidates.items()
        if value is not None
    ]
    if available:
        selected_fraction, _, selected_kind = min(available)
        if termination != selected_kind or abs(fraction-selected_fraction) > 1.0e-12:
            raise OrbitValidationError("event label is not the earliest candidate")
    elif termination != Termination.STEP_LIMIT.value:
        raise OrbitValidationError("termination has no supporting event candidate")
    elif step_index != config.max_steps or fraction != 1.0:
        raise OrbitValidationError("step-limit witness does not reach max_steps")
    expected_position = start + fraction*(end-start)
    start_velocity = vectors["step_start_velocity_m_per_s"]
    end_velocity = vectors["step_end_velocity_m_per_s"]
    expected_velocity = start_velocity + fraction*(end_velocity-start_velocity)
    if (
        not np.allclose(expected_position, result["final_position_m"], rtol=0.0, atol=tolerance)
        or not np.allclose(expected_velocity, result["final_velocity_m_per_s"], rtol=0.0, atol=1.0e-9)
        or abs(start_time + fraction*step_dt - float(result["elapsed_time_s"]))
        > max(1.0e-18, 64.0*np.spacing(config.max_time_s))
        or abs(start_path + fraction*segment - float(result["path_length_m"]))
        > tolerance
    ):
        raise OrbitValidationError("event witness endpoint/counters are inconsistent")


def _validate_records(
    campaign_id: str,
    launches: list[object],
    results: list[object],
    *,
    expected_field_sha256: str,
    expected_config_sha256: str,
    expected_policy_sha256: str,
    require_complete: bool = True,
    allow_replay_dependent_failures: bool = False,
) -> None:
    launch_keys = {
        "launch_id", "seed_id", "kinetic_energy_ev", "pitch_angle_rad",
        "position_m", "parallel_direction", "gyrophase_rad", "flux_surface_id",
    }
    result_keys = {
        "launch_id", "termination", "reason", "final_position_m",
        "final_velocity_m_per_s", "wall_endpoint_m", "elapsed_time_s",
        "path_length_m", "steps", "accumulated_gyro_phase_rad",
        "complete_gyrocycles", "gyro_averages", "initial_energy_j",
        "final_energy_j", "maximum_relative_energy_error",
        "maximum_instantaneous_mu_relative_variation", "transit_fraction",
        "maximum_b_t", "configured_max_time_s", "configured_max_path_m",
        "event_tolerance_m", "dt_s", "backend", "event_witness",
    }
    if not launches or (require_complete and len(launches) != len(results)):
        raise OrbitValidationError("launch/result coverage is invalid")
    if any(not isinstance(item, Mapping) or set(item) != launch_keys for item in launches):
        raise OrbitValidationError("launch records are not closed")
    if any(not isinstance(item, Mapping) or set(item) != result_keys for item in results):
        raise OrbitValidationError("result records are not closed")
    launch_ids = [str(item["launch_id"]) for item in launches]
    result_ids = [str(item["launch_id"]) for item in results]
    if (
        launch_ids != sorted(launch_ids)
        or result_ids != sorted(result_ids)
        or (require_complete and launch_ids != result_ids)
        or (not require_complete and not set(result_ids).issubset(launch_ids))
        or len(set(launch_ids)) != len(launch_ids)
        or any(not identity.startswith(campaign_id + ":") for identity in launch_ids)
    ):
        raise OrbitValidationError(
            "ordered launch/result/campaign identities are inconsistent"
        )
    for launch in launches:
        launch_id = launch["launch_id"]
        flux_id = launch["flux_surface_id"]
        position = launch["position_m"]
        if (
            not isinstance(launch_id, str)
            or not re.fullmatch(
                re.escape(campaign_id)
                + r":E[0-9]+:P[0-9]+:X[0-9]+:D[+-]1:G[0-9]+",
                launch_id,
            )
            or not isinstance(flux_id, str)
            or not flux_id
            or isinstance(launch["seed_id"], bool)
            or not isinstance(launch["seed_id"], int)
            or launch["seed_id"] < 0
            or launch["seed_id"]
            != int.from_bytes(sha256(launch_id.encode()).digest()[:8], "big")
            or _real(
                "launch kinetic_energy_ev",
                launch["kinetic_energy_ev"],
                minimum=0.0,
                exclusive_minimum=True,
            )
            <= 0.0
            or not 0.0
            <= _real("launch pitch_angle_rad", launch["pitch_angle_rad"])
            <= 0.5*pi
            or not 0.0
            <= _real("launch gyrophase_rad", launch["gyrophase_rad"])
            < 2.0*pi
            or launch["parallel_direction"] not in (-1, 1)
            or isinstance(launch["parallel_direction"], bool)
            or not isinstance(launch["parallel_direction"], int)
            or not isinstance(position, (list, tuple))
            or len(position) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not isfinite(float(item))
                for item in position
            )
        ):
            raise OrbitValidationError("launch record semantics are invalid")
        try:
            ElectronLaunch(**launch)
        except (TypeError, ValueError) as error:
            raise OrbitValidationError("launch record semantics are invalid") from error
    launch_by_id = {str(item["launch_id"]): item for item in launches}
    valid_terminations = {item.value for item in Termination}
    for result in results:
        if result["termination"] not in valid_terminations:
            raise OrbitValidationError("result termination is invalid")
        if not isinstance(result["reason"], str) or not result["reason"]:
            raise OrbitValidationError("result reason must be nonempty")
        if not isinstance(result["backend"], str) or not result["backend"]:
            raise OrbitValidationError("result backend must be nonempty")
        if (
            isinstance(result["steps"], bool)
            or not isinstance(result["steps"], int)
            or result["steps"] < 0
        ):
            raise OrbitValidationError("result step count is invalid")
        for vector_name in ("final_position_m", "final_velocity_m_per_s"):
            vector = result[vector_name]
            if (
                not isinstance(vector, (list, tuple))
                or len(vector) != 3
                or any(isinstance(item, bool) or not isinstance(item, Real) or not isfinite(float(item)) for item in vector)
            ):
                raise OrbitValidationError(f"result {vector_name} is invalid")
        gyro_averages = result["gyro_averages"]
        if (
            not isinstance(gyro_averages, (list, tuple))
            or isinstance(result["complete_gyrocycles"], bool)
            or not isinstance(result["complete_gyrocycles"], int)
            or result["complete_gyrocycles"] < 0
            or result["complete_gyrocycles"] != len(gyro_averages)
        ):
            raise OrbitValidationError("complete gyrocycle evidence is inconsistent")
        phase = _real(
            "result accumulated_gyro_phase_rad",
            result["accumulated_gyro_phase_rad"],
            minimum=0.0,
        )
        phase_tolerance = 128.0 * np.finfo(float).eps * max(1.0, phase)
        expected_cycles = floor((phase + phase_tolerance) / (2.0 * pi))
        if result["complete_gyrocycles"] != expected_cycles:
            raise OrbitValidationError("gyrocycle count differs from accumulated phase")
        for index, average in enumerate(gyro_averages):
            if not isinstance(average, Mapping) or set(average) != {
                "cycle_index", "phase_start_rad", "phase_end_rad", "mu_j_per_t"
            }:
                raise OrbitValidationError("gyro average record is not closed")
            if (
                isinstance(average["cycle_index"], bool)
                or not isinstance(average["cycle_index"], int)
                or average["cycle_index"] != index
                or _real("gyro phase start", average["phase_start_rad"], minimum=0.0)
                != 2.0 * pi * index
                or _real("gyro phase end", average["phase_end_rad"], minimum=0.0)
                != 2.0 * pi * (index + 1)
                or _real("gyro magnetic moment", average["mu_j_per_t"], minimum=0.0) < 0.0
            ):
                raise OrbitValidationError("gyro average semantics are invalid")
        wall_endpoint = result["wall_endpoint_m"]
        if (result["termination"] == Termination.WALL_HIT.value) != (
            wall_endpoint is not None
        ):
            raise OrbitValidationError("wall endpoint and termination disagree")
        if wall_endpoint is not None and (
            not isinstance(wall_endpoint, (list, tuple))
            or len(wall_endpoint) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not isfinite(float(item))
                for item in wall_endpoint
            )
        ):
            raise OrbitValidationError("wall endpoint is invalid")
        if wall_endpoint is not None and tuple(wall_endpoint) != tuple(
            result["final_position_m"]
        ):
            raise OrbitValidationError("wall endpoint differs from final position")
        for key in (
            "elapsed_time_s", "path_length_m", "accumulated_gyro_phase_rad",
            "initial_energy_j", "final_energy_j",
            "maximum_relative_energy_error", "transit_fraction", "dt_s",
            "maximum_b_t", "configured_max_time_s", "configured_max_path_m",
            "event_tolerance_m",
        ):
            _real(f"result {key}", result[key], minimum=0.0)
        if not 0.0 <= float(result["transit_fraction"]) <= 1.0:
            raise OrbitValidationError("result transit_fraction must lie in [0,1]")
        mu_variation = result["maximum_instantaneous_mu_relative_variation"]
        if mu_variation is not None:
            _real(
                "result magnetic-moment variation",
                mu_variation,
                minimum=0.0,
            )
        initial_energy = float(result["initial_energy_j"])
        final_energy = float(result["final_energy_j"])
        if initial_energy <= 0.0:
            raise OrbitValidationError("result initial energy must be positive")
        if float(result["dt_s"]) <= 0.0:
            raise OrbitValidationError("result timestep must be positive")
        if (
            float(result["configured_max_time_s"]) <= 0.0
            or float(result["configured_max_path_m"]) <= 0.0
        ):
            raise OrbitValidationError("result physical limits must be positive")
        observed_energy_change = abs(final_energy - initial_energy) / initial_energy
        if (
            float(result["maximum_relative_energy_error"])
            + 64.0 * np.finfo(float).eps
            < observed_energy_change
        ):
            raise OrbitValidationError("result energy-error envelope is inconsistent")
        maximum_phase = (
            abs(ELECTRON_CHARGE_C)
            * float(result["maximum_b_t"])
            * float(result["elapsed_time_s"])
            / ELECTRON_MASS_KG
        )
        if phase > maximum_phase + max(phase_tolerance, 1.0e-14):
            raise OrbitValidationError("result gyro phase exceeds its field/time bound")
        elapsed = float(result["elapsed_time_s"])
        path = float(result["path_length_m"])
        time_limit = float(result["configured_max_time_s"])
        path_limit = float(result["configured_max_path_m"])
        event_tolerance = float(result["event_tolerance_m"])
        time_tolerance = max(
            64.0 * np.spacing(time_limit),
            1.0e-12 * float(result["dt_s"]),
        )
        termination = result["termination"]
        if termination == Termination.TIME_TIMEOUT.value:
            if abs(elapsed - time_limit) > time_tolerance:
                raise OrbitValidationError("time-timeout endpoint is inconsistent")
        elif elapsed > time_limit + time_tolerance:
            raise OrbitValidationError("result overshoots configured time limit")
        if termination == Termination.PATH_TIMEOUT.value:
            if abs(path - path_limit) > event_tolerance + 64.0 * np.spacing(path_limit):
                raise OrbitValidationError("path-timeout endpoint is inconsistent")
        elif path > path_limit + event_tolerance:
            raise OrbitValidationError("result overshoots configured path limit")
        launch = launch_by_id[str(result["launch_id"])]
        if termination == Termination.INITIAL_STATE_INVALID.value:
            if (
                result["steps"] != 0
                or elapsed != 0.0
                or path != 0.0
                or phase != 0.0
                or tuple(result["final_position_m"]) != tuple(launch["position_m"])
                or any(float(value) != 0.0 for value in result["final_velocity_m_per_s"])
            ):
                raise OrbitValidationError("initial-state failure evidence is inconsistent")
        elif result["steps"] < 1:
            raise OrbitValidationError("non-initial result requires at least one step")
        if elapsed > result["steps"] * float(result["dt_s"]) + time_tolerance:
            raise OrbitValidationError("elapsed time exceeds step/timestep envelope")
        _validate_event_witness(
            result,
            launch=launch,
            expected_field_sha256=expected_field_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_policy_sha256=expected_policy_sha256,
            allow_replay_dependent_failures=allow_replay_dependent_failures,
        )


def validate_result_artifact(
    artifact: Mapping[str, object],
    *,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
    _allow_replay_dependent_failures: bool = False,
) -> None:
    _hash("expected_policy_sha256", expected_policy_sha256)
    if expected_estimator_policy is not EstimatorPolicy.UNWEIGHTED_BINOMIAL:
        raise OrbitValidationError("weighted/stratified estimators are unsupported")
    authority_floor = _real(
        "expected certificate tightness floor",
        expected_minimum_certificate_tightness_ratio,
        minimum=0.001,
        maximum=1.0,
    )
    expected = {
        "schema_version", "classification", "campaign_id", "identities", "launches",
        "batch_manifest", "estimator", "verification",
        "results", "summary",
        "interpolation_evidence", "convergence_evidence", "preregistration",
        "limitations", "integrity",
    }
    if set(artifact) != expected:
        raise OrbitValidationError("result artifact keys are not closed")
    if artifact["schema_version"] != SCHEMA_VERSION or artifact["classification"] != CLASSIFICATION:
        raise OrbitValidationError("unsupported result schema/classification")
    if not isinstance(artifact["campaign_id"], str) or not artifact["campaign_id"]:
        raise OrbitValidationError("campaign identity is invalid")
    identities = artifact["identities"]
    if not isinstance(identities, Mapping) or set(identities) != {
        "field_sha256", "config_sha256", "policy_sha256", "code_sha256", "launches_sha256",
        "results_sha256", "batch_manifest_sha256", "estimator_sha256",
    }:
        raise OrbitValidationError("identity set is invalid")
    for name, value in identities.items():
        _hash(f"identities.{name}", value)
    if identities["code_sha256"] != code_identity():
        raise OrbitValidationError("artifact code identity differs from runtime")
    if identities["policy_sha256"] != expected_policy_sha256:
        raise OrbitValidationError("artifact policy differs from external authority")
    estimator = artifact["estimator"]
    if not isinstance(estimator, Mapping) or dict(estimator) != {
        "policy": expected_estimator_policy.value,
        "estimand_id": ESTIMAND_ID,
        "equal_launch_weights_required": True,
    }:
        raise OrbitValidationError("artifact estimator policy is invalid")
    verification = artifact["verification"]
    if not isinstance(verification, Mapping) or dict(verification) != {
        "requirement": REPLAY_REQUIREMENT,
        "structural_status": "unverified_pending_deterministic_replay",
    }:
        raise OrbitValidationError("artifact replay requirement/status is invalid")
    if not isinstance(artifact["launches"], list) or not isinstance(artifact["results"], list):
        raise OrbitValidationError("launches/results must be arrays")
    if len(artifact["launches"]) != len(artifact["results"]):
        raise OrbitValidationError("launch/result counts differ")
    summary = artifact["summary"]
    if (
        not isinstance(summary, Mapping)
        or set(summary) != {
            "ensemble_id", "trial_count", "wall_hit", "reflected", "escaped",
            "incomplete", "termination_counts", "result_identity_sha256",
        }
        or summary.get("ensemble_id") != artifact["campaign_id"]
    ):
        raise OrbitValidationError("campaign and ensemble identities differ")
    _validate_records(
        str(artifact["campaign_id"]),
        artifact["launches"],
        artifact["results"],
        expected_field_sha256=identities["field_sha256"],
        expected_config_sha256=identities["config_sha256"],
        expected_policy_sha256=identities["policy_sha256"],
        allow_replay_dependent_failures=_allow_replay_dependent_failures,
    )
    launch_ids = [str(item["launch_id"]) for item in artifact["launches"]]
    _, weights = _validate_batch_manifest(
        artifact["batch_manifest"], launch_ids, estimator["policy"]
    )
    batch_identity = content_hash(
        {
            "estimator_policy": estimator["policy"],
            "batches": artifact["batch_manifest"],
        }
    )
    if batch_identity != identities["batch_manifest_sha256"]:
        raise OrbitValidationError("batch manifest identity SHA-256 mismatch")
    if _estimator_identity(
        launch_ids, weights, estimator["policy"]
    ) != identities["estimator_sha256"]:
        raise OrbitValidationError("estimator identity SHA-256 mismatch")
    if content_hash(artifact["launches"]) != identities["launches_sha256"]:
        raise OrbitValidationError("launches identity SHA-256 mismatch")
    if content_hash(artifact["results"]) != identities["results_sha256"]:
        raise OrbitValidationError("results identity SHA-256 mismatch")
    if summary.get("result_identity_sha256") != result_records_identity(
        artifact["results"]
    ):
        raise OrbitValidationError("summary result identity SHA-256 mismatch")
    trial_value = summary.get("trial_count")
    if (
        isinstance(trial_value, bool)
        or not isinstance(trial_value, int)
        or trial_value < 1
    ):
        raise OrbitValidationError("summary trial count is invalid")
    trial_count = trial_value
    if len(artifact["results"]) != trial_count:
        raise OrbitValidationError("summary trial count differs from result count")
    counts = summary.get("termination_counts")
    expected_terminations = {
        "wall_hit", "reflected", "domain_escape", "path_timeout", "time_timeout",
        "step_limit", "nonfinite_state", "extreme_relativity", "field_failure",
        "initial_state_invalid",
    }
    if (
        not isinstance(counts, Mapping) or set(counts) != expected_terminations
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values())
        or sum(counts.values()) != trial_count
    ):
        raise OrbitValidationError("termination counts are invalid")
    observed_counts = {name: 0 for name in expected_terminations}
    for result in artifact["results"]:
        observed_counts[result["termination"]] += 1
    if dict(counts) != observed_counts:
        raise OrbitValidationError("termination counts differ from orbit records")
    categories = {
        "wall_hit": counts["wall_hit"],
        "reflected": counts["reflected"],
        "escaped": counts["domain_escape"],
        "incomplete": trial_count-counts["wall_hit"]-counts["reflected"]-counts["domain_escape"],
    }
    for name, successes in categories.items():
        _validate_probability(name, summary.get(name), successes, trial_count)
    for key in ("interpolation_evidence", "convergence_evidence", "preregistration"):
        if not isinstance(artifact[key], Mapping):
            raise OrbitValidationError(f"{key} must be an object")
    interpolation = artifact["interpolation_evidence"]
    if set(interpolation) != {
        "certified_max_b_t", "reference_max_b_t", "runtime_max_seen_t",
        "dense_diagnostic_max_b_t", "certificate_tightness_ratio",
        "minimum_certificate_tightness_ratio", "certificate_preflight_passed",
        "material_map_sha256", "field_error_report", "passed",
    }:
        raise OrbitValidationError("interpolation evidence is not closed")
    certified = _real(
        "interpolation certified_max_b_t",
        interpolation["certified_max_b_t"],
        minimum=0.0,
        exclusive_minimum=True,
    )
    runtime_max = _real(
        "interpolation runtime_max_seen_t",
        interpolation["runtime_max_seen_t"],
        minimum=0.0,
    )
    dense_max = _real(
        "interpolation dense_diagnostic_max_b_t",
        interpolation["dense_diagnostic_max_b_t"],
        minimum=0.0,
    )
    tightness_ratio = _real(
        "interpolation certificate_tightness_ratio",
        interpolation["certificate_tightness_ratio"],
        minimum=0.0,
        maximum=1.0,
    )
    minimum_tightness = _real(
        "interpolation minimum_certificate_tightness_ratio",
        interpolation["minimum_certificate_tightness_ratio"],
        minimum=0.0,
        maximum=1.0,
    )
    observed_runtime_max = max(
        float(result["maximum_b_t"]) for result in artifact["results"]
    )
    if (
        runtime_max != observed_runtime_max
        or runtime_max > certified * (1.0 + 64.0 * np.finfo(float).eps)
        or dense_max > certified * (1.0 + 64.0 * np.finfo(float).eps)
        or tightness_ratio != dense_max / certified
        or tightness_ratio < minimum_tightness
        or minimum_tightness != authority_floor
        or interpolation["certificate_preflight_passed"] is not True
        or interpolation["passed"] is not True
    ):
        raise OrbitValidationError("interpolation bound evidence is invalid")
    error_report = interpolation["field_error_report"]
    if not isinstance(error_report, Mapping) or set(error_report) != {
        "sample_count", "psi_node_max_abs_wb", "br_max_abs_t", "bz_max_abs_t",
        "b_rms_t", "b_relative_rms",
    }:
        raise OrbitValidationError("field error report is not closed")
    if (
        isinstance(error_report["sample_count"], bool)
        or not isinstance(error_report["sample_count"], int)
        or error_report["sample_count"] < 1
        or any(
            isinstance(error_report[key], bool)
            or not isinstance(error_report[key], Real)
            or not isfinite(float(error_report[key]))
            or float(error_report[key]) < 0.0
            for key in set(error_report) - {"sample_count"}
        )
    ):
        raise OrbitValidationError("field error report values are invalid")
    _hash("material_map_sha256", interpolation["material_map_sha256"])
    reference_max = interpolation["reference_max_b_t"]
    if reference_max is not None and (
        isinstance(reference_max, bool)
        or not isinstance(reference_max, Real)
        or not isfinite(float(reference_max))
        or float(reference_max) <= 0.0
    ):
        raise OrbitValidationError("reference maximum evidence is invalid")
    convergence = artifact["convergence_evidence"]
    if set(convergence) != {
        "timestep_passed", "cross_map_passed", "backend_parity_passed"
    } or any(convergence[key] is not True for key in convergence):
        raise OrbitValidationError("convergence evidence is incomplete")
    preregistration = artifact["preregistration"]
    if set(preregistration) != {
        "protocol_id", "frozen_before_outcomes", "held_out_geometry_status"
    } or (
        not isinstance(preregistration["protocol_id"], str)
        or not preregistration["protocol_id"]
        or preregistration["frozen_before_outcomes"] is not True
        or preregistration["held_out_geometry_status"]
        not in {"pending", "passed"}
    ):
        raise OrbitValidationError("preregistration evidence is invalid")
    if (
        not isinstance(artifact["limitations"], list)
        or len(artifact["limitations"]) < 4
        or any(
            not isinstance(item, str) or not item
            for item in artifact["limitations"]
        )
    ):
        raise OrbitValidationError("limitations must be explicit")
    integrity = artifact["integrity"]
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "algorithm", "canonicalization", "payload_sha256"
    }:
        raise OrbitValidationError("integrity block is invalid")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != "json-sort-keys-compact-utf8-v1"
    ):
        raise OrbitValidationError("integrity algorithm/canonicalization is invalid")
    _hash("integrity.payload_sha256", integrity["payload_sha256"])
    expected_digest = content_hash({key: value for key, value in artifact.items() if key != "integrity"})
    if integrity.get("payload_sha256") != expected_digest:
        raise OrbitValidationError("artifact payload SHA-256 mismatch")


def validate_result_replay(
    artifact: Mapping[str, object],
    *,
    field: AxisymmetricField,
    config: OrbitConfig,
    expected_field_sha256: str,
    expected_config_sha256: str,
    expected_launches_sha256: str,
    expected_batch_manifest_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
) -> None:
    """Replay deterministic CPU results against bound launch/config/field authority."""

    validate_result_artifact(
        artifact,
        expected_policy_sha256=expected_policy_sha256,
        expected_estimator_policy=expected_estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            expected_minimum_certificate_tightness_ratio
        ),
        _allow_replay_dependent_failures=True,
    )
    _hash("expected_field_sha256", expected_field_sha256)
    _hash("expected_config_sha256", expected_config_sha256)
    _hash("expected_launches_sha256", expected_launches_sha256)
    _hash("expected_batch_manifest_sha256", expected_batch_manifest_sha256)
    identities = artifact["identities"]
    if (
        identities["field_sha256"] != expected_field_sha256
        or identities["config_sha256"] != expected_config_sha256
    ):
        raise OrbitValidationError("replay field/config authority differs")
    if identities["launches_sha256"] != expected_launches_sha256:
        raise OrbitValidationError("replay launch authority differs")
    if (
        identities["batch_manifest_sha256"]
        != expected_batch_manifest_sha256
    ):
        raise OrbitValidationError("replay batch manifest authority differs")
    from .integrator import integrate_orbit
    replayed = []
    result_by_id = {result["launch_id"]: result for result in artifact["results"]}
    for launch_record in artifact["launches"]:
        launch = ElectronLaunch(**launch_record)
        expected = result_by_id[launch.launch_id]
        replayed_result = integrate_orbit(
            launch,
            field,
            config,
            backend=str(expected["backend"]),
        )
        replayed.append(
            _result_dict(
                replayed_result,
                field_identity_sha256=expected_field_sha256,
                config_identity_sha256=expected_config_sha256,
                policy_identity_sha256=expected_policy_sha256,
            )
        )
    if canonical_bytes(replayed) != canonical_bytes(artifact["results"]):
        raise OrbitValidationError("deterministic result replay differs from artifact")


def write_artifact(
    path: str | Path,
    artifact: Mapping[str, object],
    *,
    field: AxisymmetricField,
    config: OrbitConfig,
    expected_field_sha256: str,
    expected_config_sha256: str,
    expected_launches_sha256: str,
    expected_batch_manifest_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
) -> VerifiedOrbitEvidence:
    validate_result_replay(
        artifact,
        field=field,
        config=config,
        expected_field_sha256=expected_field_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_launches_sha256=expected_launches_sha256,
        expected_batch_manifest_sha256=expected_batch_manifest_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_estimator_policy=expected_estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            expected_minimum_certificate_tightness_ratio
        ),
    )
    data = canonical_bytes(artifact)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    digest = sha256(data).hexdigest()
    target.with_name(target.name + ".sha256").write_text(
        f"{digest}  {target.name}\n", encoding="ascii"
    )
    return VerifiedOrbitEvidence(data, digest, _VERIFICATION_SEAL)


def _read_canonical_object(path: Path) -> dict[str, object]:
    def closed_pairs(pairs):
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise OrbitValidationError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=closed_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OrbitValidationError(f"nonfinite JSON constant {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OrbitValidationError("invalid checkpoint/artifact JSON") from error
    if not isinstance(loaded, dict) or path.read_bytes() != canonical_bytes(loaded):
        raise OrbitValidationError("persistent JSON is not canonical")
    return loaded


def load_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
) -> UnverifiedOrbitArtifact:
    target = Path(path)
    data = target.read_bytes()
    digest = sha256(data).hexdigest()
    _hash("expected_file_sha256", expected_file_sha256)
    if digest != expected_file_sha256:
        raise OrbitValidationError("artifact external file SHA-256 mismatch")
    sidecar = target.with_name(target.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != f"{digest}  {target.name}\n":
        raise OrbitValidationError("artifact SHA-256 sidecar mismatch")
    loaded = _read_canonical_object(target)
    validate_result_artifact(
        loaded,
        expected_policy_sha256=expected_policy_sha256,
        expected_estimator_policy=expected_estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            expected_minimum_certificate_tightness_ratio
        ),
    )
    return UnverifiedOrbitArtifact(data, digest)


def load_and_verify_artifact(
    path: str | Path,
    *,
    field: AxisymmetricField,
    config: OrbitConfig,
    expected_file_sha256: str,
    expected_field_sha256: str,
    expected_config_sha256: str,
    expected_launches_sha256: str,
    expected_batch_manifest_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
) -> VerifiedOrbitEvidence:
    structural = load_artifact(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_estimator_policy=expected_estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            expected_minimum_certificate_tightness_ratio
        ),
    )
    validate_result_replay(
        structural._materialize(),
        field=field,
        config=config,
        expected_field_sha256=expected_field_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_launches_sha256=expected_launches_sha256,
        expected_batch_manifest_sha256=expected_batch_manifest_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_estimator_policy=expected_estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            expected_minimum_certificate_tightness_ratio
        ),
    )
    return VerifiedOrbitEvidence(
        structural._payload_bytes, structural.file_sha256, _VERIFICATION_SEAL
    )


def checkpoint(
    campaign_id: str,
    completed_batch_ids: Sequence[int],
    launches: Sequence[ElectronLaunch],
    results: Sequence[OrbitResult],
    batch_manifest: Sequence[Mapping[str, object]],
    *,
    field_identity_sha256: str,
    config_identity_sha256: str,
    policy_identity_sha256: str,
    minimum_certificate_tightness_ratio_authority: float,
    estimator_policy: EstimatorPolicy,
    expected_batch_manifest_sha256: str,
    partial_current_batch: Mapping[str, object] | None = None,
    previous_checkpoint_sha256: str = "0" * 64,
) -> dict[str, object]:
    if not isinstance(campaign_id, str) or not campaign_id:
        raise OrbitValidationError("checkpoint campaign_id must be nonempty")
    _hash("previous_checkpoint_sha256", previous_checkpoint_sha256)
    _hash("expected_batch_manifest_sha256", expected_batch_manifest_sha256)
    launch_records = [
        asdict(launch) for launch in sorted(launches, key=lambda item: item.launch_id)
    ]
    result_records = [
        _result_dict(
            result,
            field_identity_sha256=field_identity_sha256,
            config_identity_sha256=config_identity_sha256,
            policy_identity_sha256=policy_identity_sha256,
        )
        for result in sorted(results, key=lambda item: item.launch_id)
    ]
    authority = {
        "field_sha256": _hash("field_identity_sha256", field_identity_sha256),
        "config_sha256": _hash("config_identity_sha256", config_identity_sha256),
        "policy_sha256": _hash("policy_identity_sha256", policy_identity_sha256),
        "code_sha256": code_identity(),
        "launches_sha256": content_hash(launch_records),
        "results_sha256": content_hash(result_records),
        "batch_manifest_sha256": "",
        "estimator_sha256": "",
        "estimator_policy": estimator_policy.value,
        "replay_requirement": REPLAY_REQUIREMENT,
        "result_verification_status": (
            "witness_validated_replay_required_for_final_artifact"
        ),
        "minimum_certificate_tightness_ratio": _real(
            "minimum certificate tightness authority",
            minimum_certificate_tightness_ratio_authority,
            minimum=0.001,
            maximum=1.0,
        ),
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in completed_batch_ids):
        raise OrbitValidationError("completed batch IDs must be nonnegative integers")
    if list(completed_batch_ids) != sorted(set(completed_batch_ids)):
        raise OrbitValidationError("completed batch IDs must be sorted and unique")
    launch_ids = [str(record["launch_id"]) for record in launch_records]
    if estimator_policy is not EstimatorPolicy.UNWEIGHTED_BINOMIAL:
        raise OrbitValidationError("weighted/stratified estimators are unsupported")
    manifest_batches, weights = _validate_batch_manifest(
        list(batch_manifest), launch_ids, estimator_policy.value
    )
    authority["batch_manifest_sha256"] = content_hash(
        {
            "estimator_policy": estimator_policy.value,
            "batches": batch_manifest,
        }
    )
    if authority["batch_manifest_sha256"] != expected_batch_manifest_sha256:
        raise OrbitValidationError(
            "checkpoint batch manifest differs from external authority"
        )
    authority["estimator_sha256"] = _estimator_identity(
        launch_ids, weights, estimator_policy.value
    )
    authority["campaign_identity_sha256"] = content_hash(
        {
            "campaign_id": campaign_id,
            "launches_sha256": authority["launches_sha256"],
            "batch_manifest_sha256": authority["batch_manifest_sha256"],
            "policy_sha256": authority["policy_sha256"],
            "minimum_certificate_tightness_ratio": authority[
                "minimum_certificate_tightness_ratio"
            ],
            "estimator_policy": authority["estimator_policy"],
            "estimator_sha256": authority["estimator_sha256"],
            "replay_requirement": authority["replay_requirement"],
        }
    )
    completed_set = set(completed_batch_ids)
    if not completed_set.issubset(manifest_batches):
        raise OrbitValidationError("completed batch ID is absent from manifest")
    covered = {
        identity
        for batch_id in completed_set
        for identity in manifest_batches[batch_id]
    }
    partial_record = None if partial_current_batch is None else dict(partial_current_batch)
    if partial_record is not None:
        if set(partial_record) != {"batch_id", "completed_launch_ids"}:
            raise OrbitValidationError("partial current batch is not closed")
        partial_id = partial_record["batch_id"]
        partial_ids = partial_record["completed_launch_ids"]
        if (
            isinstance(partial_id, bool)
            or not isinstance(partial_id, int)
            or partial_id not in manifest_batches
            or partial_id in completed_set
            or not isinstance(partial_ids, list)
            or not 0 < len(partial_ids) < len(manifest_batches[partial_id])
            or tuple(partial_ids) != manifest_batches[partial_id][:len(partial_ids)]
        ):
            raise OrbitValidationError("partial current batch coverage is invalid")
        covered.update(partial_ids)
    pending = sorted(set(launch_ids) - covered)
    if sorted(record["launch_id"] for record in result_records) != sorted(covered):
        raise OrbitValidationError("checkpoint results differ from batch coverage")
    payload = {
        "schema_version": CHECKPOINT_VERSION,
        "campaign_id": campaign_id,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "authority": authority,
        "launches": launch_records,
        "batch_manifest": list(batch_manifest),
        "completed_batch_ids": list(completed_batch_ids),
        "partial_current_batch": partial_record,
        "pending_launch_ids": pending,
        "coverage": {
            "total_launches": len(launch_ids),
            "completed_launches": len(covered),
            "pending_launches": len(pending),
            "completed_batches": len(completed_set),
        },
        "results": result_records,
    }
    payload["payload_sha256"] = content_hash(payload)
    validate_checkpoint(
        payload,
        expected_campaign_id=campaign_id,
        expected_launches_sha256=authority["launches_sha256"],
        expected_batch_manifest_sha256=authority["batch_manifest_sha256"],
        expected_policy_sha256=authority["policy_sha256"],
        expected_estimator_policy=estimator_policy,
        expected_minimum_certificate_tightness_ratio=authority[
            "minimum_certificate_tightness_ratio"
        ],
    )
    return payload


def validate_checkpoint(
    value: Mapping[str, object],
    *,
    expected_campaign_id: str,
    expected_launches_sha256: str,
    expected_batch_manifest_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
) -> None:
    if not isinstance(expected_campaign_id, str) or not expected_campaign_id:
        raise OrbitValidationError("expected checkpoint campaign must be nonempty")
    _hash("expected_launches_sha256", expected_launches_sha256)
    _hash("expected_batch_manifest_sha256", expected_batch_manifest_sha256)
    _hash("expected_policy_sha256", expected_policy_sha256)
    if expected_estimator_policy is not EstimatorPolicy.UNWEIGHTED_BINOMIAL:
        raise OrbitValidationError("weighted/stratified estimators are unsupported")
    authority_floor = _real(
        "expected certificate tightness floor",
        expected_minimum_certificate_tightness_ratio,
        minimum=0.001,
        maximum=1.0,
    )
    if set(value) != {
        "schema_version", "campaign_id", "previous_checkpoint_sha256",
        "authority", "launches", "batch_manifest", "completed_batch_ids",
        "partial_current_batch", "pending_launch_ids", "coverage", "results",
        "payload_sha256",
    }:
        raise OrbitValidationError("checkpoint keys are not closed")
    if value.get("schema_version") != CHECKPOINT_VERSION:
        raise OrbitValidationError("unsupported checkpoint schema")
    campaign_id = value.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise OrbitValidationError("checkpoint campaign_id must be nonempty")
    _hash("previous_checkpoint_sha256", value.get("previous_checkpoint_sha256"))
    authority = value.get("authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "field_sha256", "config_sha256", "policy_sha256", "code_sha256",
        "launches_sha256", "results_sha256", "batch_manifest_sha256",
        "estimator_sha256", "campaign_identity_sha256",
        "minimum_certificate_tightness_ratio", "estimator_policy",
        "replay_requirement", "result_verification_status",
    }:
        raise OrbitValidationError("checkpoint authority is not closed")
    for name, digest in authority.items():
        if name in {
            "minimum_certificate_tightness_ratio", "estimator_policy",
            "replay_requirement", "result_verification_status",
        }:
            continue
        _hash(f"checkpoint.authority.{name}", digest)
    if authority["code_sha256"] != code_identity():
        raise OrbitValidationError("checkpoint code identity differs from runtime")
    if campaign_id != expected_campaign_id:
        raise OrbitValidationError("checkpoint campaign differs from external authority")
    if authority["launches_sha256"] != expected_launches_sha256:
        raise OrbitValidationError("checkpoint launch set differs from external authority")
    if authority["batch_manifest_sha256"] != expected_batch_manifest_sha256:
        raise OrbitValidationError("checkpoint batch manifest differs from external authority")
    if authority["policy_sha256"] != expected_policy_sha256:
        raise OrbitValidationError("checkpoint policy differs from external authority")
    if (
        authority["estimator_policy"] != expected_estimator_policy.value
        or authority["replay_requirement"] != REPLAY_REQUIREMENT
        or authority["result_verification_status"]
        != "witness_validated_replay_required_for_final_artifact"
    ):
        raise OrbitValidationError("checkpoint estimator/replay authority is invalid")
    if authority["minimum_certificate_tightness_ratio"] != authority_floor:
        raise OrbitValidationError("checkpoint certificate floor differs from authority")
    expected_campaign_identity = content_hash(
        {
            "campaign_id": campaign_id,
            "launches_sha256": authority["launches_sha256"],
            "batch_manifest_sha256": authority["batch_manifest_sha256"],
            "policy_sha256": authority["policy_sha256"],
            "minimum_certificate_tightness_ratio": authority[
                "minimum_certificate_tightness_ratio"
            ],
            "estimator_policy": authority["estimator_policy"],
            "estimator_sha256": authority["estimator_sha256"],
            "replay_requirement": authority["replay_requirement"],
        }
    )
    if authority["campaign_identity_sha256"] != expected_campaign_identity:
        raise OrbitValidationError("checkpoint campaign identity hash mismatch")
    batches = value.get("completed_batch_ids")
    if (
        not isinstance(batches, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in batches
        )
        or batches != sorted(set(batches))
    ):
        raise OrbitValidationError("checkpoint batch IDs are invalid")
    launches = value.get("launches")
    results = value.get("results")
    if not isinstance(launches, list) or not isinstance(results, list):
        raise OrbitValidationError("checkpoint launches/results must be arrays")
    _validate_records(
        campaign_id,
        launches,
        results,
        expected_field_sha256=authority["field_sha256"],
        expected_config_sha256=authority["config_sha256"],
        expected_policy_sha256=authority["policy_sha256"],
        require_complete=False,
    )
    launch_ids = [str(item["launch_id"]) for item in launches]
    manifest = value.get("batch_manifest")
    manifest_batches, weights = _validate_batch_manifest(
        manifest, launch_ids, authority["estimator_policy"]
    )
    if content_hash(
        {
            "estimator_policy": authority["estimator_policy"],
            "batches": manifest,
        }
    ) != authority["batch_manifest_sha256"]:
        raise OrbitValidationError("checkpoint batch manifest authority hash mismatch")
    if _estimator_identity(
        launch_ids, weights, authority["estimator_policy"]
    ) != authority["estimator_sha256"]:
        raise OrbitValidationError("checkpoint estimator identity mismatch")
    if not set(batches).issubset(manifest_batches):
        raise OrbitValidationError("checkpoint contains unknown completed batch")
    covered = {
        identity for batch_id in batches for identity in manifest_batches[batch_id]
    }
    partial = value.get("partial_current_batch")
    if partial is not None:
        if not isinstance(partial, Mapping) or set(partial) != {
            "batch_id", "completed_launch_ids"
        }:
            raise OrbitValidationError("partial current batch is not closed")
        partial_id = partial["batch_id"]
        partial_ids = partial["completed_launch_ids"]
        if (
            isinstance(partial_id, bool)
            or not isinstance(partial_id, int)
            or partial_id not in manifest_batches
            or partial_id in batches
            or not isinstance(partial_ids, list)
            or not 0 < len(partial_ids) < len(manifest_batches[partial_id])
            or tuple(partial_ids) != manifest_batches[partial_id][:len(partial_ids)]
        ):
            raise OrbitValidationError("partial current batch coverage is invalid")
        covered.update(partial_ids)
    result_ids = [str(item["launch_id"]) for item in results]
    pending = value.get("pending_launch_ids")
    expected_pending = sorted(set(launch_ids)-covered)
    if (
        result_ids != sorted(covered)
        or not isinstance(pending, list)
        or pending != expected_pending
    ):
        raise OrbitValidationError("checkpoint result/pending coverage is inconsistent")
    coverage = value.get("coverage")
    expected_coverage = {
        "total_launches": len(launch_ids),
        "completed_launches": len(covered),
        "pending_launches": len(expected_pending),
        "completed_batches": len(batches),
    }
    if not isinstance(coverage, Mapping) or dict(coverage) != expected_coverage:
        raise OrbitValidationError("checkpoint coverage counters are inconsistent")
    if content_hash(launches) != authority["launches_sha256"]:
        raise OrbitValidationError("checkpoint launch authority hash mismatch")
    if content_hash(results) != authority["results_sha256"]:
        raise OrbitValidationError("checkpoint results authority hash mismatch")
    expected = content_hash({key: item for key, item in value.items() if key != "payload_sha256"})
    if value.get("payload_sha256") != expected:
        raise OrbitValidationError("checkpoint payload SHA-256 mismatch")


def write_checkpoint(
    path: str | Path,
    value: Mapping[str, object],
    *,
    expected_campaign_id: str,
    expected_launches_sha256: str,
    expected_batch_manifest_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
) -> str:
    validate_checkpoint(
        value,
        expected_campaign_id=expected_campaign_id,
        expected_launches_sha256=expected_launches_sha256,
        expected_batch_manifest_sha256=expected_batch_manifest_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_estimator_policy=expected_estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            expected_minimum_certificate_tightness_ratio
        ),
    )
    data = canonical_bytes(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return sha256(data).hexdigest()


def load_checkpoint(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_campaign_id: str,
    expected_launches_sha256: str,
    expected_batch_manifest_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
    expected_previous_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    target = Path(path)
    _hash("expected_file_sha256", expected_file_sha256)
    if sha256(target.read_bytes()).hexdigest() != expected_file_sha256:
        raise OrbitValidationError("checkpoint external file SHA-256 mismatch")
    loaded = _read_canonical_object(target)
    validate_checkpoint(
        loaded,
        expected_campaign_id=expected_campaign_id,
        expected_launches_sha256=expected_launches_sha256,
        expected_batch_manifest_sha256=expected_batch_manifest_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_estimator_policy=expected_estimator_policy,
        expected_minimum_certificate_tightness_ratio=(
            expected_minimum_certificate_tightness_ratio
        ),
    )
    if (
        expected_previous_checkpoint_sha256 is not None
        and loaded.get("previous_checkpoint_sha256") != expected_previous_checkpoint_sha256
    ):
        raise OrbitValidationError("checkpoint ancestry SHA-256 mismatch")
    return loaded


def merge_checkpoint_results(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    *,
    expected_campaign_id: str,
    expected_launches_sha256: str,
    expected_batch_manifest_sha256: str,
    expected_policy_sha256: str,
    expected_estimator_policy: EstimatorPolicy,
    expected_minimum_certificate_tightness_ratio: float,
) -> list[dict[str, object]]:
    """Validate a monotone checkpoint resume and return current ordered results."""

    validation = {
        "expected_campaign_id": expected_campaign_id,
        "expected_launches_sha256": expected_launches_sha256,
        "expected_batch_manifest_sha256": expected_batch_manifest_sha256,
        "expected_policy_sha256": expected_policy_sha256,
        "expected_estimator_policy": expected_estimator_policy,
        "expected_minimum_certificate_tightness_ratio": (
            expected_minimum_certificate_tightness_ratio
        ),
    }
    validate_checkpoint(previous, **validation)
    validate_checkpoint(current, **validation)
    if current["previous_checkpoint_sha256"] != content_hash(previous):
        raise OrbitValidationError("checkpoint resume ancestry is invalid")
    previous_results = {
        result["launch_id"]: result for result in previous["results"]
    }
    current_results = {
        result["launch_id"]: result for result in current["results"]
    }
    if not set(previous_results).issubset(current_results):
        raise OrbitValidationError("checkpoint resume dropped completed launches")
    if any(
        canonical_bytes(record) != canonical_bytes(current_results[launch_id])
        for launch_id, record in previous_results.items()
    ):
        raise OrbitValidationError("checkpoint resume changed completed evidence")
    if not set(previous["completed_batch_ids"]).issubset(
        current["completed_batch_ids"]
    ):
        raise OrbitValidationError("checkpoint resume dropped completed batches")
    return list(current["results"])


def coupling_v42_handoff(
    evidence: VerifiedOrbitEvidence,
    *,
    expected_batch_manifest_sha256: str,
) -> dict[str, object]:
    """Create an export-only payload; no coupling consumer is claimed."""

    if not isinstance(evidence, VerifiedOrbitEvidence):
        raise OrbitValidationError("coupling handoff requires verified orbit evidence")
    _hash("expected_batch_manifest_sha256", expected_batch_manifest_sha256)
    artifact = evidence._materialize()
    if (
        artifact["identities"]["batch_manifest_sha256"]
        != expected_batch_manifest_sha256
    ):
        raise OrbitValidationError(
            "coupling batch manifest differs from external authority"
        )
    summary = artifact["summary"]
    interval = summary["wall_hit"]
    p = float(interval["probability"])
    n = int(summary["trial_count"])
    expected = wilson_interval(int(interval["successes"]), n)
    if (
        interval["trials"] != n
        or interval != asdict(expected)
        or not 0.0 <= float(interval["lower"]) <= p <= float(interval["upper"]) <= 1.0
    ):
        raise OrbitValidationError("handoff wall probability evidence is invalid")
    result_identity = summary["result_identity_sha256"]
    _hash("result_identity_sha256", result_identity)
    standard = sqrt(p*(1.0-p)/n) if n > 0 else float("nan")
    if not isfinite(standard):
        raise OrbitValidationError("binomial standard uncertainty is nonfinite")
    return {
        "schema_version": HANDOFF_VERSION,
        "classification": CLASSIFICATION,
        "quantity": "electron_dielectric_wall_loss_probability",
        "probability": p,
        "standard_uncertainty": standard,
        "confidence_interval_95": [interval["lower"], interval["upper"]],
        "trial_count": n,
        "orbit_result_artifact_sha256": evidence.file_sha256,
        "result_identity_sha256": result_identity,
        "batch_manifest_sha256": expected_batch_manifest_sha256,
        "verification_status": "deterministic_replay_verified",
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "coupling_target": "cft-field-plasma-coupling/4.2.0",
        "integration_status": "export_only_pending_consumer_integration",
        "plasma_network_role": "export_only_pending_integration",
    }
