"""Verified topology coupling records and solver-facing projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from enum import Enum
from math import fsum, isfinite
from typing import Any

from .losses import derive_mirror_loss
from .models import (
    AcceptedFieldEvidence,
    CandidateKind,
    CouplingRecord,
    CouplingValidationError,
    ProfileRole,
    TopologyCandidate,
    TopologyPolicy,
    TopologySegment,
    TopologyStatus,
    UncertaintyModel,
)
from .profiles import extract_profiles, interpolate_profile, stable_lerp
from .topology import describe_profile
from .validation import reverify_accepted_evidence

COUPLING_SCHEMA_VERSION = "cft-field-plasma-coupling/2.0.0"
_ALGORITHM_VERSION = "verified-topology-segmentation-v2"
_MINIMUM_KINDS = frozenset(
    (CandidateKind.NULL, CandidateKind.MINIMUM, CandidateKind.PLATEAU_MINIMUM)
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not isfinite(value):
        raise CouplingValidationError("coupling records cannot contain non-finite numbers")
    return value


def _canonical_hash(label: bytes, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(label + b"\0" + encoded).hexdigest()


def coupling_model_hash(
    topology_policy: TopologyPolicy,
    uncertainty_model: UncertaintyModel,
    *,
    inner_profile_radius_m: float,
    inner_profile_role: ProfileRole,
    wall_radius_m: float,
) -> str:
    """Hash every numerical/model choice capable of changing derived values."""

    payload = {
        "algorithm": _ALGORITHM_VERSION,
        "topology": _json_value(topology_policy),
        "uncertainty": _json_value(uncertainty_model),
        "inner_profile_radius_m": inner_profile_radius_m,
        "inner_profile_role": inner_profile_role.value,
        "wall_radius_m": wall_radius_m,
    }
    return _canonical_hash(b"cft-coupling-model-v2", payload)


def _choose_cusps(
    candidates: tuple[TopologyCandidate, ...],
    policy: TopologyPolicy,
) -> tuple[tuple[TopologyCandidate, ...], tuple[TopologyCandidate, ...]]:
    eligible = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if (
                    candidate.kind in _MINIMUM_KINDS
                    or (
                        policy.allow_boundary_minima_as_cusps
                        and candidate.kind is CandidateKind.BOUNDARY_MINIMUM
                    )
                )
                and candidate.confidence >= policy.minimum_candidate_confidence
                and candidate.prominence_t
                >= policy.minimum_prominence_sigma * candidate.sigma_b_t
            ),
            key=lambda candidate: (candidate.z_m, candidate.kind.value),
        )
    )
    groups: list[list[TopologyCandidate]] = []
    for candidate in eligible:
        candidate_indices = frozenset(candidate.sample_indices)
        target = next(
            (
                group
                for group in groups
                if any(
                    candidate_indices.intersection(existing.sample_indices)
                    for existing in group
                )
            ),
            None,
        )
        if target is None:
            groups.append([candidate])
        else:
            target.append(candidate)
    selected: list[TopologyCandidate] = []
    alternatives = [candidate for candidate in candidates if candidate not in eligible]
    for group in groups:
        winner = max(
            group,
            key=lambda candidate: (
                candidate.kind is CandidateKind.NULL,
                candidate.confidence,
                -abs(candidate.z_m),
                -candidate.z_m,
            ),
        )
        selected.append(winner)
        alternatives.extend(candidate for candidate in group if candidate is not winner)
    selected.sort(key=lambda candidate: candidate.z_m)
    alternatives.sort(key=lambda candidate: (candidate.z_m, candidate.kind.value))
    return tuple(selected), tuple(alternatives)


def _record_without_hash(record: CouplingRecord) -> dict[str, Any]:
    payload = _json_value(record)
    if not isinstance(payload, dict):
        raise CouplingValidationError("coupling record serialization failed")
    payload.pop("record_hash", None)
    return payload


def build_coupling_record(
    evidence: AcceptedFieldEvidence,
    *,
    wall_radius_m: float,
    topology_policy: TopologyPolicy = TopologyPolicy(),
    uncertainty_model: UncertaintyModel = UncertaintyModel(),
    reference_time_utc: datetime | None = None,
) -> CouplingRecord:
    """Reverify immutable evidence, then build a coupling record."""

    accepted = reverify_accepted_evidence(
        evidence, reference_time_utc=reference_time_utc
    )
    inner_profile, wall_profile = extract_profiles(
        accepted.field_map,
        wall_radius_m,
        uncertainty_model=uncertainty_model,
    )
    inner = describe_profile(inner_profile, topology_policy)
    wall = describe_profile(wall_profile, topology_policy)
    candidate_pool = inner.nulls + inner.extrema
    if topology_policy.allow_boundary_minima_as_cusps:
        candidate_pool += tuple(
            candidate
            for candidate in inner.boundary_extrema
            if candidate.kind is CandidateKind.BOUNDARY_MINIMUM
        )
    selected, alternatives = _choose_cusps(candidate_pool, topology_policy)
    status = inner.topology_status
    reason = inner.topology_reason
    if (
        selected
        and topology_policy.allow_boundary_minima_as_cusps
        and status is TopologyStatus.NO_TOPOLOGY
    ):
        status = TopologyStatus.RESOLVED
        reason = "boundary minima explicitly enabled and passed all gates"
    segments: list[TopologySegment] = []
    if status is TopologyStatus.RESOLVED and selected:
        bounds = [inner_profile.z_m[0]]
        bounds.extend(
            stable_lerp(left.z_m, right.z_m, 0.5)
            for left, right in zip(selected, selected[1:])
        )
        bounds.append(inner_profile.z_m[-1])
        for index, cusp in enumerate(selected):
            _, _, wall_b, wall_independent, wall_common = interpolate_profile(
                wall_profile, cusp.z_m
            )
            mirror = derive_mirror_loss(
                cusp,
                wall_b,
                wall_independent,
                common_mode_sigma_t=wall_common,
                residual_correlation=uncertainty_model.residual_correlation,
                coverage_factor=uncertainty_model.coverage_factor,
            )
            segments.append(
                TopologySegment(
                    segment_id=f"topology-{index + 1:03d}",
                    z_start_m=bounds[index],
                    z_end_m=bounds[index + 1],
                    representative_cusp_z_m=cusp.z_m,
                    mirror_loss=mirror,
                    confidence=mirror.confidence,
                )
            )
        if any(
            segment.confidence < topology_policy.minimum_segment_confidence
            for segment in segments
        ):
            alternatives = tuple(
                sorted(
                    alternatives + selected,
                    key=lambda candidate: (candidate.z_m, candidate.kind.value),
                )
            )
            segments = []
            status = TopologyStatus.AMBIGUOUS
            reason = "one or more segments fail the declared confidence gate"
    elif candidate_pool and not selected:
        status = TopologyStatus.AMBIGUOUS
        reason = "candidates were preserved but none pass uncertainty/confidence gates"
    overall_confidence = (
        0.0
        if not segments
        else fsum(segment.confidence for segment in segments) / len(segments)
    )
    model_hash = coupling_model_hash(
        topology_policy,
        uncertainty_model,
        inner_profile_radius_m=inner_profile.sampled_r_m,
        inner_profile_role=inner_profile.role,
        wall_radius_m=float(wall_radius_m),
    )
    provisional = CouplingRecord(
        schema_version=COUPLING_SCHEMA_VERSION,
        record_hash="",
        topology_status=status,
        topology_reason=reason,
        field_map_hash=accepted.field_map.field_map_hash,
        artifact_hash=accepted.artifact_hash,
        source_hash=accepted.source_hash,
        source_map_binding_hash=accepted.source_map_binding_hash,
        artifact_schema_version=accepted.artifact_schema_version,
        model_level=accepted.model_level,
        field_model_id=accepted.field_model_id,
        field_model_hash=accepted.field_model_hash,
        code_hash=accepted.code_hash,
        config_hash=accepted.config_hash,
        backend_id=accepted.backend_id,
        backend_version=accepted.backend_version,
        adapter_id=accepted.adapter_id,
        adapter_code_hash=accepted.adapter_code_hash,
        adapter_contract_id=accepted.adapter_contract.contract_id,
        adapter_contract_version=accepted.adapter_contract.contract_version,
        adapter_input_schema_version=(
            accepted.adapter_contract.input_schema_version
        ),
        adapter_normalized_schema_version=(
            accepted.adapter_contract.normalized_schema_version
        ),
        adapter_is_migration=accepted.adapter_contract.is_migration,
        generated_at_utc=accepted.generated_at_utc,
        maximum_age_s=accepted.validation_policy.maximum_age_s,
        maximum_future_skew_s=(
            accepted.validation_policy.maximum_future_skew_s
        ),
        diagnostics=accepted.diagnostics,
        coupling_model_hash=model_hash,
        inner_profile_radius_m=inner_profile.sampled_r_m,
        inner_profile_role=inner_profile.role,
        wall_radius_m=float(wall_radius_m),
        inner_profile=inner,
        wall=wall,
        uncertainty_model=uncertainty_model,
        segments=tuple(segments),
        alternative_candidates=alternatives,
        overall_confidence=overall_confidence,
    )
    record_hash = _canonical_hash(
        b"cft-coupling-record-v2", _record_without_hash(provisional)
    )
    return replace(provisional, record_hash=record_hash)


def coupling_record_dict(record: CouplingRecord) -> dict[str, Any]:
    result = _json_value(record)
    if not isinstance(result, dict):
        raise CouplingValidationError("coupling record serialization failed")
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def global_solver_inputs(
    record: CouplingRecord,
) -> tuple[dict[str, float | str | bool | None], ...]:
    """Project resolved segments while retaining complete solver-facing identity."""

    if record.topology_status is not TopologyStatus.RESOLVED:
        return ()
    return tuple(
        {
            "record_hash": record.record_hash,
            "segment_id": segment.segment_id,
            "z_start_m": segment.z_start_m,
            "z_end_m": segment.z_end_m,
            "inner_profile_radius_m": record.inner_profile_radius_m,
            "inner_profile_role": record.inner_profile_role.value,
            "wall_radius_m": record.wall_radius_m,
            "loss_cone_probability": segment.mirror_loss.probability.value,
            "loss_cone_probability_standard_uncertainty": (
                segment.mirror_loss.probability.standard_uncertainty
            ),
            "loss_cone_probability_lower": segment.mirror_loss.probability.lower,
            "loss_cone_probability_upper": segment.mirror_loss.probability.upper,
            "input_covariance_t2": (
                segment.mirror_loss.probability.input_covariance_t2
            ),
            "input_correlation": segment.mirror_loss.probability.input_correlation,
            "mirror_ratio_high_to_low": segment.mirror_loss.mirror_ratio_high_to_low,
            "confidence": segment.confidence,
            "field_map_hash": record.field_map_hash,
            "artifact_hash": record.artifact_hash,
            "source_hash": record.source_hash,
            "source_map_binding_hash": record.source_map_binding_hash,
            "artifact_schema_version": record.artifact_schema_version,
            "model_level": record.model_level,
            "field_model_id": record.field_model_id,
            "field_model_hash": record.field_model_hash,
            "code_hash": record.code_hash,
            "config_hash": record.config_hash,
            "backend_id": record.backend_id,
            "backend_version": record.backend_version,
            "adapter_id": record.adapter_id,
            "adapter_code_hash": record.adapter_code_hash,
            "adapter_contract_id": record.adapter_contract_id,
            "adapter_contract_version": record.adapter_contract_version,
            "adapter_input_schema_version": record.adapter_input_schema_version,
            "adapter_normalized_schema_version": (
                record.adapter_normalized_schema_version
            ),
            "adapter_is_migration": record.adapter_is_migration,
            "generated_at_utc": record.generated_at_utc.isoformat().replace(
                "+00:00", "Z"
            ),
            "maximum_age_s": record.maximum_age_s,
            "maximum_future_skew_s": record.maximum_future_skew_s,
            "residual_norm": record.diagnostics.residual_norm,
            "residual_tolerance": record.diagnostics.residual_tolerance,
            "relative_residual": record.diagnostics.relative_residual,
            "relative_tolerance": record.diagnostics.relative_tolerance,
            "coupling_model_hash": record.coupling_model_hash,
        }
        for segment in record.segments
    )
