"""Fail-closed artifact acceptance, validation, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from typing import Iterable, Sequence

from .models import (
    _EVIDENCE_FACTORY_KEY,
    _EvidenceSnapshot,
    AcceptedArtifactAdapter,
    AcceptedArtifactClaims,
    AcceptedFieldEvidence,
    AdapterVersionContract,
    AxisymmetricFieldMapLike,
    AxisymmetricProfileLike,
    CouplingValidationError,
    EvidenceVerificationError,
    FieldProfile,
    FieldProvenance,
    MapValidationPolicy,
    ProfileRole,
    SolverDiagnosticsEvidence,
    ValidatedAxisymmetricMap,
)

_HASH_HEXDIGITS = frozenset("0123456789abcdef")


def _finite_tuple(name: str, values: Iterable[float]) -> tuple[float, ...]:
    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise CouplingValidationError(f"{name} must contain real numbers") from error
    if any(not isfinite(value) for value in converted):
        raise CouplingValidationError(f"{name} must contain only finite values")
    return converted


def _validate_hash(name: str, value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in _HASH_HEXDIGITS for character in normalized
    ):
        raise EvidenceVerificationError(
            f"{name} must be a 64-character SHA-256 hex digest"
        )
    return normalized


def _validate_identity(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise EvidenceVerificationError(f"{name} must not be empty")
    return normalized


def _reference_time(reference_time_utc: datetime | None) -> datetime:
    if reference_time_utc is None:
        return datetime.now(timezone.utc)
    if (
        reference_time_utc.tzinfo is None
        or reference_time_utc.utcoffset() is None
    ):
        raise CouplingValidationError("reference_time_utc must be timezone-aware")
    return reference_time_utc.astimezone(timezone.utc)


def validate_provenance(
    provenance: FieldProvenance,
    policy: MapValidationPolicy,
    *,
    reference_time_utc: datetime | None = None,
) -> FieldProvenance:
    """Validate map-only metadata without treating it as accepted evidence."""

    if provenance.coordinate_system != "cylindrical_axisymmetric_r_z":
        raise CouplingValidationError(
            "coordinate_system must be cylindrical_axisymmetric_r_z"
        )
    if provenance.coordinate_unit != "m" or provenance.component_unit != "T":
        raise CouplingValidationError("coupling accepts only SI metres and tesla")
    generated = provenance.generated_at_utc
    if generated.tzinfo is None or generated.utcoffset() is None:
        raise CouplingValidationError("generated_at_utc must be timezone-aware")
    generated_utc = generated.astimezone(timezone.utc)
    age_s = (_reference_time(reference_time_utc) - generated_utc).total_seconds()
    if age_s < -float(policy.maximum_future_skew_s):
        raise CouplingValidationError("field map timestamp is unacceptably in the future")
    if policy.maximum_age_s is not None and age_s > float(policy.maximum_age_s):
        raise CouplingValidationError("field map is stale under maximum_age_s")
    return FieldProvenance(
        field_model_id=_validate_identity("field_model_id", provenance.field_model_id),
        field_model_hash=_validate_hash("field_model_hash", provenance.field_model_hash),
        source_hash=_validate_hash("source_hash", provenance.source_hash),
        generated_at_utc=generated_utc,
        coordinate_system=provenance.coordinate_system,
        coordinate_unit=provenance.coordinate_unit,
        component_unit=provenance.component_unit,
    )


def _validate_policy(policy: MapValidationPolicy) -> None:
    if (
        isinstance(policy.minimum_radial_samples, bool)
        or not isinstance(policy.minimum_radial_samples, int)
        or policy.minimum_radial_samples < 2
    ):
        raise CouplingValidationError("minimum_radial_samples must be an integer >= 2")
    if (
        isinstance(policy.minimum_axial_samples, bool)
        or not isinstance(policy.minimum_axial_samples, int)
        or policy.minimum_axial_samples < 3
    ):
        raise CouplingValidationError("minimum_axial_samples must be an integer >= 3")
    numeric = (
        ("maximum_future_skew_s", policy.maximum_future_skew_s),
        ("axis_coordinate_tolerance_m", policy.axis_coordinate_tolerance_m),
        ("axis_br_absolute_tolerance_t", policy.axis_br_absolute_tolerance_t),
        ("axis_br_relative_tolerance", policy.axis_br_relative_tolerance),
    )
    for name, raw in numeric:
        value = float(raw)
        if not isfinite(value) or value < 0.0:
            raise CouplingValidationError(f"{name} must be finite and non-negative")
    if policy.maximum_age_s is not None:
        maximum_age = float(policy.maximum_age_s)
        if not isfinite(maximum_age) or maximum_age < 0.0:
            raise CouplingValidationError("maximum_age_s must be finite and non-negative")
    if not policy.current_artifact_schema.strip():
        raise CouplingValidationError("current_artifact_schema must not be empty")
    if not policy.accepted_model_levels or not all(
        item.strip() for item in policy.accepted_model_levels
    ):
        raise CouplingValidationError("accepted_model_levels must not be empty")
    if any(not item.strip() for item in policy.validated_migration_adapter_ids):
        raise CouplingValidationError(
            "validated_migration_adapter_ids must contain non-empty IDs"
        )


def _strictly_increasing(name: str, values: Sequence[float]) -> None:
    if any(right <= left for left, right in zip(values, values[1:])):
        raise CouplingValidationError(f"{name} must be strictly increasing (not inverted)")


def _canonical_map(
    r_m: Sequence[float],
    z_m: Sequence[float],
    b_r_t: Sequence[Sequence[float]],
    b_z_t: Sequence[Sequence[float]],
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    canonical_r = _finite_tuple("r_m", r_m)
    canonical_z = _finite_tuple("z_m", z_m)
    canonical_br = tuple(
        _finite_tuple(f"b_r_t[{index}]", row) for index, row in enumerate(b_r_t)
    )
    canonical_bz = tuple(
        _finite_tuple(f"b_z_t[{index}]", row) for index, row in enumerate(b_z_t)
    )
    if len(canonical_br) != len(canonical_r) or len(canonical_bz) != len(canonical_r):
        raise CouplingValidationError("field component row counts must match r_m")
    if any(len(row) != len(canonical_z) for row in canonical_br + canonical_bz):
        raise CouplingValidationError("field component row lengths must match z_m")
    return canonical_r, canonical_z, canonical_br, canonical_bz


def canonical_axisymmetric_map_bytes(
    r_m: Sequence[float],
    z_m: Sequence[float],
    b_r_t: Sequence[Sequence[float]],
    b_z_t: Sequence[Sequence[float]],
) -> bytes:
    """Canonical labelled binary64 representation used by every content hash."""

    canonical_r, canonical_z, canonical_br, canonical_bz = _canonical_map(
        r_m, z_m, b_r_t, b_z_t
    )
    chunks = [
        b"cft-coupling-axisymmetric-map-v2\0",
        struct.pack(">QQ", len(canonical_r), len(canonical_z)),
    ]
    for label, values in (
        (b"r_m\0", canonical_r),
        (b"z_m\0", canonical_z),
    ):
        chunks.append(label)
        chunks.extend(struct.pack(">d", value) for value in values)
    for label, rows in ((b"b_r_t\0", canonical_br), (b"b_z_t\0", canonical_bz)):
        chunks.append(label)
        for row in rows:
            chunks.extend(struct.pack(">d", value) for value in row)
    return b"".join(chunks)


def hash_axisymmetric_map(
    r_m: Sequence[float],
    z_m: Sequence[float],
    b_r_t: Sequence[Sequence[float]],
    b_z_t: Sequence[Sequence[float]],
) -> str:
    return hashlib.sha256(
        canonical_axisymmetric_map_bytes(r_m, z_m, b_r_t, b_z_t)
    ).hexdigest()


def source_map_binding_hash(
    map_content_hash: str, source_hash: str, artifact_hash: str
) -> str:
    """Bind accepted source identity and exact artifact bytes to map bytes."""

    map_hash = _validate_hash("map_content_hash", map_content_hash)
    source = _validate_hash("source_hash", source_hash)
    artifact = _validate_hash("artifact_hash", artifact_hash)
    payload = (
        b"cft-coupling-source-map-binding-v1\0"
        + bytes.fromhex(map_hash)
        + bytes.fromhex(source)
        + bytes.fromhex(artifact)
    )
    return hashlib.sha256(payload).hexdigest()


def validate_axisymmetric_map(
    field: AxisymmetricFieldMapLike,
    provenance: FieldProvenance,
    policy: MapValidationPolicy = MapValidationPolicy(),
    *,
    reference_time_utc: datetime | None = None,
) -> ValidatedAxisymmetricMap:
    """Validate map geometry; this does not issue accepted evidence."""

    _validate_policy(policy)
    validate_provenance(provenance, policy, reference_time_utc=reference_time_utc)
    try:
        preliminary_r = _finite_tuple("r_m", field.r_m)
        preliminary_z = _finite_tuple("z_m", field.z_m)
        if len(preliminary_r) < policy.minimum_radial_samples:
            raise CouplingValidationError("field map is undersampled radially")
        if len(preliminary_z) < policy.minimum_axial_samples:
            raise CouplingValidationError("field map is undersampled axially")
        r_m, z_m, br_rows, bz_rows = _canonical_map(
            field.r_m, field.z_m, field.b_r_t, field.b_z_t
        )
    except AttributeError as error:
        raise CouplingValidationError(
            "field must implement AxisymmetricFieldMapLike"
        ) from error
    diagnostics = getattr(field, "diagnostics", None)
    if diagnostics is not None and getattr(diagnostics, "converged", None) is not True:
        raise CouplingValidationError("field map diagnostics do not report convergence")
    _strictly_increasing("r_m", r_m)
    _strictly_increasing("z_m", z_m)
    if any(radius < 0.0 for radius in r_m):
        raise CouplingValidationError("all radial coordinates must be non-negative")
    if policy.require_axis and abs(r_m[0]) > policy.axis_coordinate_tolerance_m:
        raise CouplingValidationError("field map must include the symmetry axis r=0")
    field_scale = max(
        (abs(value) for rows in (br_rows, bz_rows) for row in rows for value in row),
        default=0.0,
    )
    scaled_axis_limit = policy.axis_br_relative_tolerance * field_scale
    if not isfinite(scaled_axis_limit):
        raise CouplingValidationError("axis regularity tolerance overflowed")
    axis_limit = max(policy.axis_br_absolute_tolerance_t, scaled_axis_limit)
    if policy.require_axis and any(abs(value) > axis_limit for value in br_rows[0]):
        raise CouplingValidationError("B_r violates axis regularity at r=0")
    return ValidatedAxisymmetricMap(
        r_m=r_m,
        z_m=z_m,
        b_r_t=br_rows,
        b_z_t=bz_rows,
        field_map_hash=hash_axisymmetric_map(r_m, z_m, br_rows, bz_rows),
    )


def _validate_diagnostics(
    diagnostics: SolverDiagnosticsEvidence,
) -> SolverDiagnosticsEvidence:
    if diagnostics.converged is not True:
        raise EvidenceVerificationError("accepted field diagnostics must be converged")
    values = (
        diagnostics.residual_norm,
        diagnostics.residual_tolerance,
        diagnostics.relative_residual,
        diagnostics.relative_tolerance,
    )
    if any(not isfinite(float(value)) for value in values):
        raise EvidenceVerificationError("field diagnostics must be finite")
    if diagnostics.residual_norm < 0.0 or diagnostics.relative_residual < 0.0:
        raise EvidenceVerificationError("field residuals must be non-negative")
    if diagnostics.residual_tolerance < 0.0 or diagnostics.relative_tolerance < 0.0:
        raise EvidenceVerificationError("field tolerances must be non-negative")
    if diagnostics.residual_norm > diagnostics.residual_tolerance:
        raise EvidenceVerificationError("absolute field residual exceeds declared tolerance")
    if diagnostics.relative_residual > diagnostics.relative_tolerance:
        raise EvidenceVerificationError("relative field residual exceeds declared tolerance")
    if (
        isinstance(diagnostics.iterations, bool)
        or not isinstance(diagnostics.iterations, int)
        or diagnostics.iterations < 0
    ):
        raise EvidenceVerificationError("field iterations must be an integer >= 0")
    return diagnostics


def _validate_adapter_contract(
    adapter_id: str,
    contract: AdapterVersionContract,
    artifact_schema_version: str,
    model_level: str,
    policy: MapValidationPolicy,
) -> AdapterVersionContract:
    if not isinstance(contract, AdapterVersionContract):
        raise EvidenceVerificationError(
            "adapter must expose an AdapterVersionContract"
        )
    normalized = AdapterVersionContract(
        contract_id=_validate_identity("adapter contract_id", contract.contract_id),
        contract_version=_validate_identity(
            "adapter contract_version", contract.contract_version
        ),
        input_schema_version=_validate_identity(
            "adapter input_schema_version", contract.input_schema_version
        ),
        normalized_schema_version=_validate_identity(
            "adapter normalized_schema_version", contract.normalized_schema_version
        ),
        model_level=_validate_identity(
            "adapter contract model_level", contract.model_level
        ),
        is_migration=contract.is_migration,
    )
    if normalized.input_schema_version != artifact_schema_version:
        raise EvidenceVerificationError(
            "adapter input schema does not match artifact schema"
        )
    if normalized.normalized_schema_version != policy.current_artifact_schema:
        raise EvidenceVerificationError(
            "adapter does not normalize to the current artifact schema"
        )
    if normalized.model_level != model_level:
        raise EvidenceVerificationError(
            "adapter contract model level does not match artifact"
        )
    if artifact_schema_version == policy.current_artifact_schema:
        if normalized.is_migration:
            raise EvidenceVerificationError(
                "current-schema adapter must not claim migration"
            )
    elif (
        not normalized.is_migration
        or adapter_id not in policy.validated_migration_adapter_ids
    ):
        raise EvidenceVerificationError(
            "legacy artifact schema requires an explicitly validated migration adapter"
        )
    return normalized


def _evidence_invariant_hash(snapshot: _EvidenceSnapshot) -> str:
    payload = {
        "field_map_hash": snapshot.field_map.field_map_hash,
        "artifact_schema_version": snapshot.artifact_schema_version,
        "model_level": snapshot.model_level,
        "artifact_hash": snapshot.artifact_hash,
        "source_hash": snapshot.source_hash,
        "source_map_binding_hash": snapshot.source_map_binding_hash,
        "backend_id": snapshot.backend_id,
        "backend_version": snapshot.backend_version,
        "field_model_id": snapshot.field_model_id,
        "field_model_hash": snapshot.field_model_hash,
        "code_hash": snapshot.code_hash,
        "config_hash": snapshot.config_hash,
        "generated_at_utc": snapshot.generated_at_utc.isoformat(),
        "diagnostics": asdict(snapshot.diagnostics),
        "adapter_id": snapshot.adapter_id,
        "adapter_code_hash": snapshot.adapter_code_hash,
        "adapter_contract": asdict(snapshot.adapter_contract),
        "validation_policy": asdict(snapshot.validation_policy),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"cft-accepted-evidence-invariant-v1\0" + encoded).hexdigest()


def verify_accepted_field_artifact(
    artifact_bytes: bytes,
    adapter: AcceptedArtifactAdapter,
    policy: MapValidationPolicy = MapValidationPolicy(),
    *,
    reference_time_utc: datetime | None = None,
) -> AcceptedFieldEvidence:
    """Issue an opaque token only after adapter and coupling-side verification."""

    _validate_policy(policy)
    if not isinstance(artifact_bytes, bytes) or not artifact_bytes:
        raise EvidenceVerificationError("non-empty immutable artifact bytes are required")
    if not isinstance(adapter, AcceptedArtifactAdapter):
        raise EvidenceVerificationError("an AcceptedArtifactAdapter is required")
    adapter_id = _validate_identity("adapter_id", adapter.adapter_id)
    adapter_code_hash = _validate_hash("adapter_code_hash", adapter.adapter_code_hash)
    adapter_contract = adapter.version_contract
    try:
        claims = adapter.verify_artifact(artifact_bytes)
    except CouplingValidationError:
        raise
    except Exception as error:
        raise EvidenceVerificationError("artifact adapter rejected the bytes") from error
    if not isinstance(claims, AcceptedArtifactClaims):
        raise EvidenceVerificationError("adapter did not return AcceptedArtifactClaims")
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    if _validate_hash("artifact_hash", claims.artifact_hash) != artifact_hash:
        raise EvidenceVerificationError("artifact hash does not match exact bytes")
    if claims.model_level not in policy.accepted_model_levels:
        raise EvidenceVerificationError("field model level is not accepted")
    checked_contract = _validate_adapter_contract(
        adapter_id,
        adapter_contract,
        claims.artifact_schema_version,
        claims.model_level,
        policy,
    )
    try:
        provenance = validate_provenance(
            FieldProvenance(
                claims.field_model_id,
                claims.field_model_hash,
                claims.source_hash,
                claims.generated_at_utc,
                claims.coordinate_system,
                claims.coordinate_unit,
                claims.component_unit,
            ),
            policy,
            reference_time_utc=reference_time_utc,
        )
        validated = validate_axisymmetric_map(
            claims.field_map,
            provenance,
            policy,
            reference_time_utc=reference_time_utc,
        )
    except EvidenceVerificationError:
        raise
    except CouplingValidationError as error:
        raise EvidenceVerificationError(str(error)) from error
    claimed_map_hash = _validate_hash("map_content_hash", claims.map_content_hash)
    if claimed_map_hash != validated.field_map_hash:
        raise EvidenceVerificationError("map content hash does not match canonical map bytes")
    expected_binding = source_map_binding_hash(
        claimed_map_hash, provenance.source_hash, artifact_hash
    )
    if (
        _validate_hash("source_map_binding_hash", claims.source_map_binding_hash)
        != expected_binding
    ):
        raise EvidenceVerificationError("source/artifact identity is not bound to map bytes")
    diagnostics = _validate_diagnostics(claims.diagnostics)
    snapshot = _EvidenceSnapshot(
        artifact_bytes=bytes(artifact_bytes),
        field_map=validated,
        artifact_schema_version=claims.artifact_schema_version,
        model_level=claims.model_level,
        artifact_hash=artifact_hash,
        source_hash=provenance.source_hash,
        source_map_binding_hash=expected_binding,
        backend_id=_validate_identity("backend_id", claims.backend_id),
        backend_version=_validate_identity("backend_version", claims.backend_version),
        field_model_id=provenance.field_model_id,
        field_model_hash=provenance.field_model_hash,
        code_hash=_validate_hash("code_hash", claims.code_hash),
        config_hash=_validate_hash("config_hash", claims.config_hash),
        generated_at_utc=provenance.generated_at_utc,
        diagnostics=diagnostics,
        adapter_id=adapter_id,
        adapter_code_hash=adapter_code_hash,
        adapter_contract=checked_contract,
        validation_policy=policy,
    )
    return AcceptedFieldEvidence(
        snapshot,
        _evidence_invariant_hash(snapshot),
        _factory_key=_EVIDENCE_FACTORY_KEY,
    )


def reverify_accepted_evidence(
    value: object,
    *,
    reference_time_utc: datetime | None = None,
) -> _EvidenceSnapshot:
    """Recompute every acceptance invariant at each record build."""

    if not isinstance(value, AcceptedFieldEvidence):
        raise EvidenceVerificationError(
            "build_coupling_record requires a verified AcceptedFieldEvidence token"
        )
    snapshot, stored_invariant_hash = value._components_for_reverification(
        _factory_key=_EVIDENCE_FACTORY_KEY
    )
    if _evidence_invariant_hash(snapshot) != stored_invariant_hash:
        raise EvidenceVerificationError(
            "accepted evidence invariant hash does not match its snapshot"
        )
    policy = snapshot.validation_policy
    _validate_policy(policy)
    artifact_bytes = snapshot.artifact_bytes
    if not isinstance(artifact_bytes, bytes) or not artifact_bytes:
        raise EvidenceVerificationError("stored artifact bytes are invalid")
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    if _validate_hash("artifact_hash", snapshot.artifact_hash) != artifact_hash:
        raise EvidenceVerificationError("stored artifact hash does not match exact bytes")
    adapter_id = _validate_identity("adapter_id", snapshot.adapter_id)
    _validate_hash("adapter_code_hash", snapshot.adapter_code_hash)
    _validate_adapter_contract(
        adapter_id,
        snapshot.adapter_contract,
        snapshot.artifact_schema_version,
        snapshot.model_level,
        policy,
    )
    if snapshot.model_level not in policy.accepted_model_levels:
        raise EvidenceVerificationError("stored field model level is not accepted")
    try:
        provenance = validate_provenance(
            FieldProvenance(
                snapshot.field_model_id,
                snapshot.field_model_hash,
                snapshot.source_hash,
                snapshot.generated_at_utc,
            ),
            policy,
            reference_time_utc=reference_time_utc,
        )
        validated = validate_axisymmetric_map(
            snapshot.field_map,
            provenance,
            policy,
            reference_time_utc=reference_time_utc,
        )
    except EvidenceVerificationError:
        raise
    except CouplingValidationError as error:
        raise EvidenceVerificationError(str(error)) from error
    if validated.field_map_hash != snapshot.field_map.field_map_hash:
        raise EvidenceVerificationError("stored field map hash is internally inconsistent")
    expected_binding = source_map_binding_hash(
        validated.field_map_hash, provenance.source_hash, artifact_hash
    )
    if (
        _validate_hash(
            "source_map_binding_hash", snapshot.source_map_binding_hash
        )
        != expected_binding
    ):
        raise EvidenceVerificationError(
            "stored source/artifact identity is not bound to map bytes"
        )
    _validate_identity("backend_id", snapshot.backend_id)
    _validate_identity("backend_version", snapshot.backend_version)
    _validate_hash("code_hash", snapshot.code_hash)
    _validate_hash("config_hash", snapshot.config_hash)
    _validate_diagnostics(snapshot.diagnostics)
    return snapshot


def validate_profile(
    profile: AxisymmetricProfileLike,
    *,
    name: str,
    sampled_r_m: float,
    role: ProfileRole = ProfileRole.CENTRELINE,
    independent_sigma_b_t: Sequence[float] | None = None,
    common_mode_sigma_t: float = 0.0,
    sigma_b_t: Sequence[float] | None = None,
    minimum_samples: int = 3,
) -> FieldProfile:
    """Canonicalize a generic profile; ``sigma_b_t`` is a legacy independent alias."""

    z_m = _finite_tuple("z_m", profile.z_m)
    b_r_t = _finite_tuple("b_r_t", profile.b_r_t)
    b_z_t = _finite_tuple("b_z_t", profile.b_z_t)
    if len(z_m) < minimum_samples:
        raise CouplingValidationError("field profile is undersampled")
    if len(b_r_t) != len(z_m) or len(b_z_t) != len(z_m):
        raise CouplingValidationError("profile component lengths must match z_m")
    _strictly_increasing("z_m", z_m)
    radius = float(sampled_r_m)
    common = float(common_mode_sigma_t)
    if not isfinite(radius) or radius < 0.0:
        raise CouplingValidationError("sampled_r_m must be finite and non-negative")
    if not isfinite(common) or common < 0.0:
        raise CouplingValidationError("common_mode_sigma_t must be finite and non-negative")
    if independent_sigma_b_t is not None and sigma_b_t is not None:
        raise CouplingValidationError("provide only one independent sigma sequence")
    supplied_sigma = (
        independent_sigma_b_t if independent_sigma_b_t is not None else sigma_b_t
    )
    independent = (
        (0.0,) * len(z_m)
        if supplied_sigma is None
        else _finite_tuple("independent_sigma_b_t", supplied_sigma)
    )
    if len(independent) != len(z_m) or any(value < 0.0 for value in independent):
        raise CouplingValidationError(
            "independent sigma must match z_m and be non-negative"
        )
    if not isinstance(role, ProfileRole):
        raise CouplingValidationError("role must be a ProfileRole")
    if not name.strip():
        raise CouplingValidationError("profile name must not be empty")
    return FieldProfile(
        name.strip(),
        role,
        radius,
        z_m,
        b_r_t,
        b_z_t,
        independent,
        common,
    )
