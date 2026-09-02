"""Strict accepted evidence for ψ-aware v3 coupling records."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite

from .models import (
    AdapterVersionContract,
    EvidenceVerificationError,
    FieldProvenance,
    MapValidationPolicy,
)
from .v3_models import (
    StabilityCase,
    TopologyStabilityStudy,
    V3ArtifactAdapter,
    V3ArtifactClaims,
    ValidatedPsiMap,
)
from .surfaces import magnetic_null_geometry
from ..fields import (
    ARTIFACT_SCHEMA_VERSION as FIELD_ARTIFACT_SCHEMA_V12,
    LEGACY_ARTIFACT_SCHEMA_VERSION as FIELD_ARTIFACT_SCHEMA_V11,
    canonical_payload_sha256 as field_canonical_payload_sha256,
    field_artifact_canonical_bytes,
    reload_field_artifact_bytes,
)
from ..fields.serialization import (
    CANONICALIZATION_V2 as FIELD_CANONICALIZATION_V2,
    parse_field_json_bytes,
)
from .validation import (
    _validate_adapter_contract,
    _validate_diagnostics,
    _validate_hash,
    _validate_identity,
    _validate_policy,
    validate_provenance,
)


@dataclass(frozen=True, slots=True)
class _V3Snapshot:
    artifact_bytes: bytes
    field_map: ValidatedPsiMap
    claims: V3ArtifactClaims
    adapter_id: str
    adapter_code_hash: str
    adapter_contract: AdapterVersionContract
    validation_policy: MapValidationPolicy
    migration_manifest_bytes: bytes | None
    migration_source_artifact_bytes: bytes | None


_V3_FACTORY_KEY = object()


class AcceptedV3FieldEvidence:
    """Private-construction immutable v3 evidence."""

    __slots__ = ("__snapshot", "__invariant_hash")

    def __new__(
        cls,
        snapshot: _V3Snapshot,
        invariant_hash: str,
        *,
        _factory_key: object | None = None,
    ) -> AcceptedV3FieldEvidence:
        if _factory_key is not _V3_FACTORY_KEY:
            raise TypeError("use verify_v3_field_artifact")
        instance = super().__new__(cls)
        object.__setattr__(instance, "_AcceptedV3FieldEvidence__snapshot", snapshot)
        object.__setattr__(
            instance, "_AcceptedV3FieldEvidence__invariant_hash", invariant_hash
        )
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AcceptedV3FieldEvidence is immutable")

    def _components(
        self, *, _factory_key: object
    ) -> tuple[_V3Snapshot, str]:
        if _factory_key is not _V3_FACTORY_KEY:
            raise TypeError("v3 evidence snapshot is private")
        return self.__snapshot, self.__invariant_hash


def canonical_psi_map_bytes(field: object) -> bytes:
    def canonical_float(value: object) -> float:
        converted = float(value)
        return 0.0 if converted == 0.0 else converted

    try:
        r = tuple(canonical_float(value) for value in field.r_m)  # type: ignore[attr-defined]
        z = tuple(canonical_float(value) for value in field.z_m)  # type: ignore[attr-defined]
        arrays = tuple(
            tuple(
                tuple(canonical_float(value) for value in row)
                for row in getattr(field, name)
            )
            for name in ("psi_wb", "b_r_t", "b_z_t")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise EvidenceVerificationError("v3 map arrays are malformed") from error
    if not r or not z or any(not isfinite(value) for value in r + z):
        raise EvidenceVerificationError("v3 map coordinates must be finite and non-empty")
    if any(len(rows) != len(r) for rows in arrays):
        raise EvidenceVerificationError("v3 map radial shapes do not match r_m")
    if any(
        len(row) != len(z) or any(not isfinite(value) for value in row)
        for rows in arrays
        for row in rows
    ):
        raise EvidenceVerificationError("v3 map arrays must be finite and match z_m")
    chunks = [
        b"cft-axisymmetric-psi-map-v3\0",
        struct.pack(">QQ", len(r), len(z)),
    ]
    for label, values in ((b"r_m\0", r), (b"z_m\0", z)):
        chunks.append(label)
        chunks.extend(struct.pack(">d", value) for value in values)
    for label, rows in zip(
        (b"psi_wb\0", b"b_r_t\0", b"b_z_t\0"), arrays, strict=True
    ):
        chunks.append(label)
        for row in rows:
            chunks.extend(struct.pack(">d", value) for value in row)
    return b"".join(chunks)


def hash_psi_map(field: object) -> str:
    return hashlib.sha256(canonical_psi_map_bytes(field)).hexdigest()


def v3_evidence_binding_hash(
    full_map_hash: str,
    source_hash: str,
    geometry_hash: str,
    material_hash: str,
    mesh_hash: str,
    domain_hash: str,
    artifact_hash: str,
) -> str:
    values = tuple(
        _validate_hash(name, value)
        for name, value in (
            ("full_map_hash", full_map_hash),
            ("source_hash", source_hash),
            ("geometry_hash", geometry_hash),
            ("material_hash", material_hash),
            ("mesh_hash", mesh_hash),
            ("domain_hash", domain_hash),
            ("artifact_hash", artifact_hash),
        )
    )
    return hashlib.sha256(
        b"cft-v3-evidence-binding\0"
        + b"".join(bytes.fromhex(value) for value in values)
    ).hexdigest()


def _validate_map(field: object, policy: MapValidationPolicy) -> ValidatedPsiMap:
    canonical_psi_map_bytes(field)
    r = tuple(float(value) for value in field.r_m)  # type: ignore[attr-defined]
    z = tuple(float(value) for value in field.z_m)  # type: ignore[attr-defined]
    psi = tuple(
        tuple(float(value) for value in row)
        for row in field.psi_wb  # type: ignore[attr-defined]
    )
    br = tuple(
        tuple(float(value) for value in row)
        for row in field.b_r_t  # type: ignore[attr-defined]
    )
    bz = tuple(
        tuple(float(value) for value in row)
        for row in field.b_z_t  # type: ignore[attr-defined]
    )
    if len(r) < policy.minimum_radial_samples or len(z) < policy.minimum_axial_samples:
        raise EvidenceVerificationError("v3 ψ map is undersampled")
    if any(right <= left for left, right in zip(r, r[1:])) or any(
        right <= left for left, right in zip(z, z[1:])
    ):
        raise EvidenceVerificationError("v3 map coordinates must increase strictly")
    coordinate_spans = (r[-1] - r[0], z[-1] - z[0])
    if any(not isfinite(value) or value <= 0.0 for value in coordinate_spans):
        raise EvidenceVerificationError(
            "v3 coordinate spans must be positively representable"
        )
    if any(value < 0.0 for value in r):
        raise EvidenceVerificationError("v3 radial coordinates must be non-negative")
    if policy.require_axis and abs(r[0]) > policy.axis_coordinate_tolerance_m:
        raise EvidenceVerificationError("v3 map must include r=0")
    scale = max(abs(value) for rows in (br, bz) for row in rows for value in row)
    axis_limit = max(
        policy.axis_br_absolute_tolerance_t,
        policy.axis_br_relative_tolerance * scale,
    )
    if not isfinite(axis_limit) or any(abs(value) > axis_limit for value in br[0]):
        raise EvidenceVerificationError("v3 map violates axis B_r regularity")
    psi_scale = max(abs(value) for row in psi for value in row)
    psi_axis_tolerance = max(1.0e-15, 1.0e-10 * psi_scale)
    if max(psi[0]) - min(psi[0]) > psi_axis_tolerance:
        raise EvidenceVerificationError("axis ψ must be gauge-constant")
    return ValidatedPsiMap(r, z, psi, br, bz, hash_psi_map(field))


def _field_map_payload(artifact: dict[str, object]) -> dict[str, object]:
    value = artifact.get("field_map")
    if not isinstance(value, dict):
        raise EvidenceVerificationError("field v1.2 artifact has no field_map")
    return value


def _verify_authoritative_field_v12(snapshot: _V3Snapshot) -> None:
    if snapshot.claims.artifact_schema_version != FIELD_ARTIFACT_SCHEMA_V12:
        return
    try:
        artifact = reload_field_artifact_bytes(
            snapshot.artifact_bytes,
            source="coupling-field-v1.2",
            allow_legacy_v1_1=False,
        )
        if field_artifact_canonical_bytes(artifact) != snapshot.artifact_bytes:
            raise EvidenceVerificationError(
                "field v1.2 bytes differ from authoritative canonical bytes"
            )
    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, EvidenceVerificationError):
            raise
        raise EvidenceVerificationError(
            "field v1.2 artifact failed authoritative canonical reload"
        ) from error
    field = _field_map_payload(artifact)
    if (
        artifact.get("model_level") != snapshot.claims.model_level
        or tuple(field.get("r_m", ())) != snapshot.field_map.r_m
        or tuple(field.get("z_m", ())) != snapshot.field_map.z_m
        or tuple(tuple(row) for row in field.get("psi_wb", ()))
        != snapshot.field_map.psi_wb
        or tuple(tuple(row) for row in field.get("b_r_t", ()))
        != snapshot.field_map.b_r_t
        or tuple(tuple(row) for row in field.get("b_z_t", ()))
        != snapshot.field_map.b_z_t
    ):
        raise EvidenceVerificationError(
            "adapter claims differ from authoritative field v1.2 reload"
        )


def _verify_field_migration(snapshot: _V3Snapshot) -> None:
    manifest_bytes = snapshot.migration_manifest_bytes
    source_bytes = snapshot.migration_source_artifact_bytes
    if manifest_bytes is None and source_bytes is None:
        return
    if (
        snapshot.claims.artifact_schema_version != FIELD_ARTIFACT_SCHEMA_V12
        or manifest_bytes is None
        or source_bytes is None
    ):
        raise EvidenceVerificationError(
            "field migration must bind v1.1 source, manifest, and v1.2 target"
        )
    try:
        source = reload_field_artifact_bytes(
            source_bytes,
            source="coupling-field-v1.1-source",
            allow_legacy_v1_1=True,
        )
        manifest = parse_field_json_bytes(
            manifest_bytes,
            source="coupling-field-migration-manifest",
            require_canonical_file_bytes=True,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise EvidenceVerificationError(
            "field migration source/manifest failed canonical reload"
        ) from error
    if source.get("schema_version") != FIELD_ARTIFACT_SCHEMA_V11:
        raise EvidenceVerificationError(
            "field migration source is not legacy v1.1"
        )
    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    integrity = manifest.get("integrity")
    if (
        manifest.get("schema_version")
        != "cft-axisymmetric-serialization-migration/1.0.0"
        or not isinstance(integrity, dict)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("canonicalization") != FIELD_CANONICALIZATION_V2
        or integrity.get("payload_sha256")
        != field_canonical_payload_sha256(payload)
    ):
        raise EvidenceVerificationError("field migration manifest is invalid")
    source_file_hash = hashlib.sha256(source_bytes).hexdigest()
    target_file_hash = hashlib.sha256(snapshot.artifact_bytes).hexdigest()
    source_payload_hash = source.get("integrity", {}).get("payload_sha256")
    target = reload_field_artifact_bytes(
        snapshot.artifact_bytes,
        source="coupling-field-v1.2-target",
        allow_legacy_v1_1=False,
    )
    target_payload_hash = target.get("integrity", {}).get("payload_sha256")
    before = manifest.get("from")
    after = manifest.get("to")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise EvidenceVerificationError("field migration manifest endpoints are invalid")
    before_artifacts = before.get("artifacts")
    after_artifacts = after.get("artifacts")
    if (
        before.get("artifact_schema") != FIELD_ARTIFACT_SCHEMA_V11
        or after.get("artifact_schema") != FIELD_ARTIFACT_SCHEMA_V12
        or not isinstance(before_artifacts, dict)
        or not isinstance(after_artifacts, dict)
    ):
        raise EvidenceVerificationError("field migration schemas are invalid")
    matches = 0
    for name in set(before_artifacts) & set(after_artifacts):
        old = before_artifacts[name]
        new = after_artifacts[name]
        if (
            isinstance(old, dict)
            and isinstance(new, dict)
            and old.get("file_sha256") == source_file_hash
            and old.get("payload_sha256") == source_payload_hash
            and new.get("file_sha256") == target_file_hash
            and new.get("payload_sha256") == target_payload_hash
        ):
            matches += 1
    if matches != 1:
        raise EvidenceVerificationError(
            "field migration manifest does not uniquely bind source and target"
        )


def _invariant_hash(snapshot: _V3Snapshot) -> str:
    claims = snapshot.claims
    payload = {
        "full_map_hash": snapshot.field_map.full_map_hash,
        "artifact_hash": claims.artifact_hash,
        "source_hash": claims.source_hash,
        "geometry_hash": claims.geometry_hash,
        "material_hash": claims.material_hash,
        "mesh_hash": claims.mesh_hash,
        "domain_hash": claims.domain_hash,
        "evidence_binding_hash": claims.evidence_binding_hash,
        "artifact_schema_version": claims.artifact_schema_version,
        "model_level": claims.model_level,
        "backend_id": claims.backend_id,
        "backend_version": claims.backend_version,
        "field_model_id": claims.field_model_id,
        "field_model_hash": claims.field_model_hash,
        "code_hash": claims.code_hash,
        "config_hash": claims.config_hash,
        "generated_at_utc": claims.generated_at_utc.isoformat(),
        "diagnostics": asdict(claims.diagnostics),
        "adapter_id": snapshot.adapter_id,
        "adapter_code_hash": snapshot.adapter_code_hash,
        "adapter_contract": asdict(snapshot.adapter_contract),
        "validation_policy": asdict(snapshot.validation_policy),
        "migration_manifest_hash": (
            hashlib.sha256(snapshot.migration_manifest_bytes).hexdigest()
            if snapshot.migration_manifest_bytes is not None
            else None
        ),
        "migration_source_artifact_hash": (
            hashlib.sha256(snapshot.migration_source_artifact_bytes).hexdigest()
            if snapshot.migration_source_artifact_bytes is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"cft-v3-evidence-invariant\0" + encoded).hexdigest()


def _verify_snapshot(
    snapshot: _V3Snapshot, *, reference_time_utc: datetime | None
) -> _V3Snapshot:
    claims = snapshot.claims
    policy = snapshot.validation_policy
    _validate_policy(policy)
    artifact_hash = hashlib.sha256(snapshot.artifact_bytes).hexdigest()
    if _validate_hash("artifact_hash", claims.artifact_hash) != artifact_hash:
        raise EvidenceVerificationError("v3 artifact bytes/hash mismatch")
    contract = _validate_adapter_contract(
        snapshot.adapter_id,
        snapshot.adapter_contract,
        claims.artifact_schema_version,
        claims.model_level,
        policy,
    )
    if contract.is_migration:
        raise EvidenceVerificationError("v3 coupling requires direct current-schema evidence")
    _verify_authoritative_field_v12(snapshot)
    _verify_field_migration(snapshot)
    if claims.model_level not in policy.accepted_model_levels:
        raise EvidenceVerificationError("v3 model level is not accepted")
    if (
        claims.coordinate_system != "cylindrical_axisymmetric_r_z"
        or claims.coordinate_unit != "m"
        or claims.flux_unit != "Wb"
        or claims.component_unit != "T"
    ):
        raise EvidenceVerificationError("v3 evidence requires SI m/Wb/T declarations")
    provenance = validate_provenance(
        FieldProvenance(
            claims.field_model_id,
            claims.field_model_hash,
            claims.source_hash,
            claims.generated_at_utc,
        ),
        policy,
        reference_time_utc=reference_time_utc,
    )
    field = _validate_map(snapshot.field_map, policy)
    if field.full_map_hash != snapshot.field_map.full_map_hash:
        raise EvidenceVerificationError("stored v3 map hash mismatch")
    if _validate_hash("full_map_hash", claims.full_map_hash) != field.full_map_hash:
        raise EvidenceVerificationError("claimed v3 full-map hash mismatch")
    for name in (
        "geometry_hash",
        "material_hash",
        "mesh_hash",
        "domain_hash",
        "code_hash",
        "config_hash",
    ):
        _validate_hash(name, getattr(claims, name))
    expected_binding = v3_evidence_binding_hash(
        field.full_map_hash,
        provenance.source_hash,
        claims.geometry_hash,
        claims.material_hash,
        claims.mesh_hash,
        claims.domain_hash,
        artifact_hash,
    )
    if _validate_hash("evidence_binding_hash", claims.evidence_binding_hash) != expected_binding:
        raise EvidenceVerificationError("v3 evidence binding mismatch")
    _validate_identity("backend_id", claims.backend_id)
    _validate_identity("backend_version", claims.backend_version)
    _validate_identity("field_model_id", claims.field_model_id)
    _validate_diagnostics(claims.diagnostics)
    return snapshot


def verify_v3_field_artifact(
    artifact_bytes: bytes,
    adapter: V3ArtifactAdapter,
    policy: MapValidationPolicy = MapValidationPolicy(),
    *,
    reference_time_utc: datetime | None = None,
    migration_manifest_bytes: bytes | None = None,
    migration_source_artifact_bytes: bytes | None = None,
) -> AcceptedV3FieldEvidence:
    if not isinstance(artifact_bytes, bytes) or not artifact_bytes:
        raise EvidenceVerificationError("non-empty v3 artifact bytes are required")
    if not isinstance(adapter, V3ArtifactAdapter):
        raise EvidenceVerificationError("a V3ArtifactAdapter is required")
    claims = adapter.verify_v3_artifact(artifact_bytes)
    if not isinstance(claims, V3ArtifactClaims):
        raise EvidenceVerificationError("v3 adapter returned invalid claims")
    snapshot = _V3Snapshot(
        bytes(artifact_bytes),
        _validate_map(claims.field_map, policy),
        claims,
        _validate_identity("adapter_id", adapter.adapter_id),
        _validate_hash("adapter_code_hash", adapter.adapter_code_hash),
        adapter.version_contract,
        policy,
        (
            bytes(migration_manifest_bytes)
            if migration_manifest_bytes is not None
            else None
        ),
        (
            bytes(migration_source_artifact_bytes)
            if migration_source_artifact_bytes is not None
            else None
        ),
    )
    _verify_snapshot(snapshot, reference_time_utc=reference_time_utc)
    return AcceptedV3FieldEvidence(
        snapshot,
        _invariant_hash(snapshot),
        _factory_key=_V3_FACTORY_KEY,
    )


def reverify_v3_evidence(
    evidence: object, *, reference_time_utc: datetime | None = None
) -> _V3Snapshot:
    if not isinstance(evidence, AcceptedV3FieldEvidence):
        raise EvidenceVerificationError("v3 build requires AcceptedV3FieldEvidence")
    snapshot, stored_hash = evidence._components(_factory_key=_V3_FACTORY_KEY)
    if _invariant_hash(snapshot) != stored_hash:
        raise EvidenceVerificationError("v3 evidence invariant was modified")
    return _verify_snapshot(snapshot, reference_time_utc=reference_time_utc)


@dataclass(frozen=True, slots=True)
class _StabilitySnapshot:
    evidence: tuple[AcceptedV3FieldEvidence, AcceptedV3FieldEvidence, AcceptedV3FieldEvidence]
    study: TopologyStabilityStudy


_STABILITY_FACTORY_KEY = object()


class AcceptedTopologyStabilityEvidence:
    """Opaque three-map mesh/domain stability evidence."""

    __slots__ = ("__snapshot", "__invariant_hash")

    def __new__(
        cls,
        snapshot: _StabilitySnapshot,
        invariant_hash: str,
        *,
        _factory_key: object | None = None,
    ) -> AcceptedTopologyStabilityEvidence:
        if _factory_key is not _STABILITY_FACTORY_KEY:
            raise TypeError("use verify_v3_topology_stability")
        instance = super().__new__(cls)
        object.__setattr__(
            instance, "_AcceptedTopologyStabilityEvidence__snapshot", snapshot
        )
        object.__setattr__(
            instance,
            "_AcceptedTopologyStabilityEvidence__invariant_hash",
            invariant_hash,
        )
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AcceptedTopologyStabilityEvidence is immutable")

    def _components(
        self, *, _factory_key: object
    ) -> tuple[_StabilitySnapshot, str]:
        if _factory_key is not _STABILITY_FACTORY_KEY:
            raise TypeError("stability evidence is private")
        return self.__snapshot, self.__invariant_hash


def _stable_cusp_z(field: ValidatedPsiMap) -> tuple[float, ...]:
    points, _ = magnetic_null_geometry(field)
    result: list[float] = []
    for _, z in points:
        if not result or abs(z - result[-1]) > 1.0e-9:
            result.append(z)
    return tuple(result)


def _stability_case(role: str, snapshot: _V3Snapshot) -> StabilityCase:
    field = snapshot.field_map
    cusps = _stable_cusp_z(field)
    return StabilityCase(
        role=role,
        artifact_hash=snapshot.claims.artifact_hash,
        full_map_hash=field.full_map_hash,
        source_hash=snapshot.claims.source_hash,
        geometry_hash=snapshot.claims.geometry_hash,
        material_hash=snapshot.claims.material_hash,
        mesh_hash=snapshot.claims.mesh_hash,
        domain_hash=snapshot.claims.domain_hash,
        evidence_binding_hash=snapshot.claims.evidence_binding_hash,
        artifact_schema_version=snapshot.claims.artifact_schema_version,
        model_level=snapshot.claims.model_level,
        field_model_id=snapshot.claims.field_model_id,
        field_model_hash=snapshot.claims.field_model_hash,
        code_hash=snapshot.claims.code_hash,
        config_hash=snapshot.claims.config_hash,
        backend_id=snapshot.claims.backend_id,
        backend_version=snapshot.claims.backend_version,
        adapter_id=snapshot.adapter_id,
        adapter_code_hash=snapshot.adapter_code_hash,
        adapter_contract=snapshot.adapter_contract,
        generated_at_utc=snapshot.claims.generated_at_utc,
        diagnostics=snapshot.claims.diagnostics,
        maximum_age_s=snapshot.validation_policy.maximum_age_s,
        maximum_future_skew_s=snapshot.validation_policy.maximum_future_skew_s,
        validation_policy=snapshot.validation_policy,
        cell_count=len(cusps),
        interior_cusp_z_m=cusps,
        radial_samples=len(field.r_m),
        axial_samples=len(field.z_m),
        radius_m=field.r_m[-1],
        z_min_m=field.z_m[0],
        z_max_m=field.z_m[-1],
    )


def _stability_invariant(study: TopologyStabilityStudy) -> str:
    encoded = json.dumps(
        asdict(study),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: (
            value.isoformat()
            if isinstance(value, datetime)
            else (_ for _ in ()).throw(TypeError(type(value).__name__))
        ),
    ).encode()
    return hashlib.sha256(b"cft-v3-topology-stability\0" + encoded).hexdigest()


def verify_v3_topology_stability(
    full_resolution: AcceptedV3FieldEvidence,
    downsampled: AcceptedV3FieldEvidence,
    enlarged_domain: AcceptedV3FieldEvidence,
    *,
    maximum_cusp_shift_m: float,
    reference_time_utc: datetime | None = None,
) -> AcceptedTopologyStabilityEvidence:
    """Verify three independently accepted maps and issue opaque stability evidence."""

    snapshots = tuple(
        reverify_v3_evidence(item, reference_time_utc=reference_time_utc)
        for item in (full_resolution, downsampled, enlarged_domain)
    )
    identity_names = (
        "source_hash",
        "geometry_hash",
        "material_hash",
        "field_model_id",
        "field_model_hash",
        "code_hash",
        "config_hash",
        "backend_id",
        "backend_version",
    )
    for name in identity_names:
        if len({getattr(item.claims, name) for item in snapshots}) != 1:
            raise EvidenceVerificationError(
                f"stability maps do not share {name}"
            )
    study = TopologyStabilityStudy(
        _stability_case("full_resolution", snapshots[0]),
        _stability_case("downsampled", snapshots[1]),
        _stability_case("enlarged_domain", snapshots[2]),
        float(maximum_cusp_shift_m),
    )
    # Local import avoids an evidence/record module cycle.
    from .v3_records import verify_topology_stability

    verify_topology_stability(
        study,
        field=snapshots[0].field_map,
        observed_cusp_z_m=study.full_resolution.interior_cusp_z_m,
    )
    snapshot = _StabilitySnapshot(
        (full_resolution, downsampled, enlarged_domain), study
    )
    return AcceptedTopologyStabilityEvidence(
        snapshot,
        _stability_invariant(study),
        _factory_key=_STABILITY_FACTORY_KEY,
    )


def reverify_v3_topology_stability(
    evidence: object,
    *,
    full_map_hash: str,
    reference_time_utc: datetime | None = None,
) -> TopologyStabilityStudy:
    if not isinstance(evidence, AcceptedTopologyStabilityEvidence):
        raise EvidenceVerificationError(
            "build requires AcceptedTopologyStabilityEvidence"
        )
    snapshot, stored_hash = evidence._components(
        _factory_key=_STABILITY_FACTORY_KEY
    )
    if _stability_invariant(snapshot.study) != stored_hash:
        raise EvidenceVerificationError("stability evidence invariant was modified")
    rebuilt = verify_v3_topology_stability(
        *snapshot.evidence,
        maximum_cusp_shift_m=snapshot.study.maximum_cusp_shift_m,
        reference_time_utc=reference_time_utc,
    )
    rebuilt_snapshot, _ = rebuilt._components(_factory_key=_STABILITY_FACTORY_KEY)
    if rebuilt_snapshot.study != snapshot.study:
        raise EvidenceVerificationError("stability evidence no longer matches maps")
    if snapshot.study.full_resolution.full_map_hash != full_map_hash:
        raise EvidenceVerificationError(
            "stability full-resolution map differs from build evidence"
        )
    return snapshot.study
