"""Fail-closed held-out validation evidence for the frozen v4 criterion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite

from .models import EvidenceVerificationError
from .validation import _validate_diagnostics, _validate_hash, _validate_identity
from .v4_models import (
    HeldOutCaseOutcome,
    HeldOutValidationAdapter,
    HeldOutValidationClaims,
    HeldOutValidationIdentity,
    HeldOutValidationPolicy,
    ValidationSetManifest,
    validation_set_manifest_hash,
)


@dataclass(frozen=True, slots=True)
class _HeldOutSnapshot:
    artifact_bytes: bytes
    claims: HeldOutValidationClaims
    adapter_id: str
    adapter_code_hash: str
    policy: HeldOutValidationPolicy


_HELD_OUT_KEY = object()


class AcceptedHeldOutValidationEvidence:
    __slots__ = ("__snapshot", "__invariant_hash")

    def __new__(
        cls,
        snapshot: _HeldOutSnapshot,
        invariant_hash: str,
        *,
        _factory_key: object | None = None,
    ) -> AcceptedHeldOutValidationEvidence:
        if _factory_key is not _HELD_OUT_KEY:
            raise TypeError("use verify_held_out_validation")
        instance = super().__new__(cls)
        object.__setattr__(
            instance, "_AcceptedHeldOutValidationEvidence__snapshot", snapshot
        )
        object.__setattr__(
            instance, "_AcceptedHeldOutValidationEvidence__invariant_hash",
            invariant_hash,
        )
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AcceptedHeldOutValidationEvidence is immutable")

    def _components(self, *, _factory_key: object) -> tuple[_HeldOutSnapshot, str]:
        if _factory_key is not _HELD_OUT_KEY:
            raise TypeError("held-out validation snapshot is private")
        return self.__snapshot, self.__invariant_hash


def _invariant(snapshot: _HeldOutSnapshot) -> str:
    claims = snapshot.claims
    payload = asdict(claims)
    payload["generated_at_utc"] = claims.generated_at_utc.isoformat()
    payload.update(
        {
            "artifact_bytes_hash": hashlib.sha256(
                snapshot.artifact_bytes
            ).hexdigest(),
            "adapter_id": snapshot.adapter_id,
            "adapter_code_hash": snapshot.adapter_code_hash,
            "policy": asdict(snapshot.policy),
        }
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(
        b"cft-v4-held-out-validation\0" + encoded
    ).hexdigest()


def _validate_manifest(
    name: str, manifest: ValidationSetManifest
) -> ValidationSetManifest:
    if not isinstance(manifest, ValidationSetManifest):
        raise EvidenceVerificationError(f"{name} manifest is malformed")
    _validate_identity(f"{name}.manifest_id", manifest.manifest_id)
    if (
        not isinstance(manifest.case_ids, tuple)
        or not isinstance(manifest.geometry_family_ids, tuple)
        or not manifest.case_ids
        or not manifest.geometry_family_ids
        or len(set(manifest.case_ids)) != len(manifest.case_ids)
        or len(set(manifest.geometry_family_ids))
        != len(manifest.geometry_family_ids)
    ):
        raise EvidenceVerificationError(
            f"{name} manifest IDs must be nonempty and unique"
        )
    for value in (*manifest.case_ids, *manifest.geometry_family_ids):
        _validate_identity(f"{name} manifest member", value)
    expected = validation_set_manifest_hash(
        manifest.manifest_id,
        manifest.case_ids,
        manifest.geometry_family_ids,
    )
    if _validate_hash(f"{name}.manifest_hash", manifest.manifest_hash) != expected:
        raise EvidenceVerificationError(f"{name} manifest hash mismatch")
    return manifest


def _validate_outcomes(
    claims: HeldOutValidationClaims,
) -> tuple[HeldOutCaseOutcome, ...]:
    outcomes = claims.outcomes
    if not isinstance(outcomes, tuple) or len(outcomes) != len(
        claims.held_out_manifest.case_ids
    ):
        raise EvidenceVerificationError(
            "held-out outcomes must cover the complete manifest"
        )
    if any(not isinstance(item, HeldOutCaseOutcome) for item in outcomes):
        raise EvidenceVerificationError("held-out outcomes are malformed")
    if {item.case_id for item in outcomes} != set(
        claims.held_out_manifest.case_ids
    ):
        raise EvidenceVerificationError(
            "held-out outcome case membership is incomplete or duplicated"
        )
    if len({item.case_id for item in outcomes}) != len(outcomes):
        raise EvidenceVerificationError("held-out outcome case IDs are duplicated")
    for outcome in outcomes:
        _validate_identity("outcome.case_id", outcome.case_id)
        _validate_identity(
            "outcome.geometry_family_id", outcome.geometry_family_id
        )
        if (
            outcome.geometry_family_id
            not in claims.held_out_manifest.geometry_family_ids
        ):
            raise EvidenceVerificationError(
                "held-out outcome family is outside the manifest"
            )
        if (
            not isinstance(outcome.three_map_hashes, tuple)
            or len(outcome.three_map_hashes) != 3
            or not isinstance(
                outcome.three_map_evidence_fingerprints,
                tuple,
            )
            or len(outcome.three_map_evidence_fingerprints) != 3
        ):
            raise EvidenceVerificationError(
                "held-out outcome requires three map hashes and fingerprints"
            )
        for value in outcome.three_map_hashes:
            _validate_hash("outcome.three_map_hash", value)
        for value in outcome.three_map_evidence_fingerprints:
            _validate_hash("outcome.three_map_evidence_fingerprint", value)
        if outcome.passed is not True:
            raise EvidenceVerificationError(
                "every held-out manifest outcome must pass"
            )
    matching = tuple(
        item
        for item in outcomes
        if item.case_id == claims.evaluated_case_id
        and item.geometry_family_id == claims.evaluated_geometry_family_id
    )
    if len(matching) != 1:
        raise EvidenceVerificationError(
            "evaluated case/family has no unique held-out outcome"
        )
    return outcomes


def _validate_snapshot(
    snapshot: _HeldOutSnapshot, reference_time_utc: datetime
) -> HeldOutValidationIdentity:
    claims = snapshot.claims
    if (
        reference_time_utc.tzinfo is None
        or reference_time_utc.utcoffset() is None
    ):
        raise EvidenceVerificationError(
            "held-out reference timestamp must be timezone-aware"
        )
    if hashlib.sha256(snapshot.artifact_bytes).hexdigest() != _validate_hash(
        "validation_artifact_hash", claims.validation_artifact_hash
    ):
        raise EvidenceVerificationError("held-out validation artifact hash mismatch")
    for name in (
        "preregistration_hash",
        "validation_code_hash",
        "validation_config_hash",
    ):
        _validate_hash(name, getattr(claims, name))
    for name in (
        "criterion_id",
        "criterion_version",
        "evaluated_case_id",
        "evaluated_geometry_family_id",
    ):
        _validate_identity(name, getattr(claims, name))
    _validate_identity("adapter_id", snapshot.adapter_id)
    _validate_hash("adapter_code_hash", snapshot.adapter_code_hash)
    _validate_diagnostics(claims.diagnostics)
    development = _validate_manifest(
        "development", claims.development_manifest
    )
    held_out = _validate_manifest("held_out", claims.held_out_manifest)
    if (
        development.manifest_hash == held_out.manifest_hash
        or set(development.case_ids) & set(held_out.case_ids)
        or set(development.geometry_family_ids)
        & set(held_out.geometry_family_ids)
    ):
        raise EvidenceVerificationError(
            "held-out case and geometry-family sets must be disjoint"
        )
    if (
        claims.evaluated_case_id not in held_out.case_ids
        or claims.evaluated_geometry_family_id
        not in held_out.geometry_family_ids
    ):
        raise EvidenceVerificationError(
            "evaluated record is outside the held-out manifest"
        )
    outcomes = _validate_outcomes(claims)
    generated = claims.generated_at_utc
    if generated.tzinfo is None or generated.utcoffset() is None:
        raise EvidenceVerificationError("held-out timestamp must be timezone-aware")
    age = (
        reference_time_utc.astimezone(timezone.utc)
        - generated.astimezone(timezone.utc)
    ).total_seconds()
    policy = snapshot.policy
    if (
        not isfinite(policy.maximum_age_s)
        or policy.maximum_age_s <= 0.0
        or not isfinite(policy.maximum_future_skew_s)
        or policy.maximum_future_skew_s < 0.0
    ):
        raise EvidenceVerificationError("held-out freshness policy is invalid")
    if age < -policy.maximum_future_skew_s or age > policy.maximum_age_s:
        raise EvidenceVerificationError("held-out validation evidence is stale/future")
    return HeldOutValidationIdentity(
        criterion_id=claims.criterion_id,
        criterion_version=claims.criterion_version,
        development_manifest=development,
        held_out_manifest=held_out,
        evaluated_case_id=claims.evaluated_case_id,
        evaluated_geometry_family_id=claims.evaluated_geometry_family_id,
        outcomes=outcomes,
        preregistration_hash=claims.preregistration_hash,
        validation_artifact_hash=claims.validation_artifact_hash,
        validation_code_hash=claims.validation_code_hash,
        validation_config_hash=claims.validation_config_hash,
        generated_at_utc=claims.generated_at_utc,
        diagnostics=claims.diagnostics,
        adapter_id=snapshot.adapter_id,
        adapter_code_hash=snapshot.adapter_code_hash,
        policy=policy,
    )


def verify_held_out_validation(
    artifact_bytes: bytes,
    adapter: HeldOutValidationAdapter,
    *,
    reference_time_utc: datetime,
    policy: HeldOutValidationPolicy = HeldOutValidationPolicy(),
) -> AcceptedHeldOutValidationEvidence:
    if not isinstance(artifact_bytes, bytes) or not artifact_bytes:
        raise EvidenceVerificationError("exact held-out artifact bytes are required")
    if not isinstance(adapter, HeldOutValidationAdapter):
        raise EvidenceVerificationError("a HeldOutValidationAdapter is required")
    _validate_identity("adapter_id", adapter.adapter_id)
    _validate_hash("adapter_code_hash", adapter.adapter_code_hash)
    claims = adapter.verify_validation_artifact(artifact_bytes)
    if not isinstance(claims, HeldOutValidationClaims):
        raise EvidenceVerificationError("validation adapter returned invalid claims")
    snapshot = _HeldOutSnapshot(
        bytes(artifact_bytes),
        claims,
        adapter.adapter_id,
        adapter.adapter_code_hash,
        policy,
    )
    _validate_snapshot(snapshot, reference_time_utc)
    return AcceptedHeldOutValidationEvidence(
        snapshot, _invariant(snapshot), _factory_key=_HELD_OUT_KEY
    )


def reverify_held_out_validation(
    value: object, *, reference_time_utc: datetime
) -> tuple[HeldOutValidationClaims, HeldOutValidationIdentity]:
    if not isinstance(value, AcceptedHeldOutValidationEvidence):
        raise EvidenceVerificationError(
            "accepted held-out validation evidence is required"
        )
    snapshot, stored = value._components(_factory_key=_HELD_OUT_KEY)
    if _invariant(snapshot) != stored:
        raise EvidenceVerificationError("held-out validation invariant was modified")
    return snapshot.claims, _validate_snapshot(snapshot, reference_time_utc)
