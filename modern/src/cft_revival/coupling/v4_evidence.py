"""Accepted three-map evidence boundary for v4 HEMP/CFT assessment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from .models import EvidenceVerificationError
from .v3_evidence import AcceptedV3FieldEvidence, _V3Snapshot, reverify_v3_evidence


@dataclass(frozen=True, slots=True)
class _V4MapSetSnapshot:
    evidence: tuple[
        AcceptedV3FieldEvidence,
        AcceptedV3FieldEvidence,
        AcceptedV3FieldEvidence,
    ]
    map_hashes: tuple[str, str, str]
    evidence_fingerprints: tuple[str, str, str]


_V4_SET_KEY = object()


class AcceptedV4MapSet:
    """Opaque primary/refined/enlarged accepted maps."""

    __slots__ = ("__snapshot", "__invariant_hash")

    def __new__(
        cls,
        snapshot: _V4MapSetSnapshot,
        invariant_hash: str,
        *,
        _factory_key: object | None = None,
    ) -> AcceptedV4MapSet:
        if _factory_key is not _V4_SET_KEY:
            raise TypeError("use verify_v4_map_set")
        instance = super().__new__(cls)
        object.__setattr__(instance, "_AcceptedV4MapSet__snapshot", snapshot)
        object.__setattr__(
            instance, "_AcceptedV4MapSet__invariant_hash", invariant_hash
        )
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AcceptedV4MapSet is immutable")

    def _components(
        self, *, _factory_key: object
    ) -> tuple[_V4MapSetSnapshot, str]:
        if _factory_key is not _V4_SET_KEY:
            raise TypeError("v4 map set is private")
        return self.__snapshot, self.__invariant_hash


def _set_hash(snapshot: _V4MapSetSnapshot) -> str:
    encoded = json.dumps(
        {
            "map_hashes": snapshot.map_hashes,
            "evidence_fingerprints": snapshot.evidence_fingerprints,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"cft-v4-map-set\0" + encoded).hexdigest()


def _evidence_fingerprint(snapshot: _V3Snapshot) -> str:
    claims = snapshot.claims
    payload = {
        "artifact_bytes_hash": hashlib.sha256(snapshot.artifact_bytes).hexdigest(),
        "artifact_hash": claims.artifact_hash,
        "full_map_hash": snapshot.field_map.full_map_hash,
        "source_hash": claims.source_hash,
        "geometry_hash": claims.geometry_hash,
        "material_hash": claims.material_hash,
        "mesh_hash": claims.mesh_hash,
        "domain_hash": claims.domain_hash,
        "evidence_binding_hash": claims.evidence_binding_hash,
        "artifact_schema_version": claims.artifact_schema_version,
        "model_level": claims.model_level,
        "field_model_id": claims.field_model_id,
        "field_model_hash": claims.field_model_hash,
        "code_hash": claims.code_hash,
        "config_hash": claims.config_hash,
        "backend_id": claims.backend_id,
        "backend_version": claims.backend_version,
        "adapter_id": snapshot.adapter_id,
        "adapter_code_hash": snapshot.adapter_code_hash,
        "adapter_contract": asdict(snapshot.adapter_contract),
        "generated_at_utc": claims.generated_at_utc.isoformat(),
        "diagnostics": asdict(claims.diagnostics),
        "validation_policy": asdict(snapshot.validation_policy),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(b"cft-v4-map-evidence\0" + encoded).hexdigest()


def _shared_identity(snapshots: tuple[_V3Snapshot, ...]) -> None:
    names = (
        "source_hash",
        "geometry_hash",
        "material_hash",
        "artifact_schema_version",
        "model_level",
        "field_model_id",
        "field_model_hash",
        "code_hash",
        "config_hash",
        "backend_id",
        "backend_version",
    )
    for name in names:
        if len({getattr(snapshot.claims, name) for snapshot in snapshots}) != 1:
            raise EvidenceVerificationError(f"v4 maps do not share {name}")
    for name in ("adapter_id", "adapter_code_hash", "adapter_contract", "validation_policy"):
        if len({getattr(snapshot, name) for snapshot in snapshots}) != 1:
            raise EvidenceVerificationError(f"v4 maps do not share {name}")


def verify_v4_map_set(
    primary: AcceptedV3FieldEvidence,
    refined: AcceptedV3FieldEvidence,
    enlarged: AcceptedV3FieldEvidence,
    *,
    reference_time_utc: datetime | None = None,
) -> AcceptedV4MapSet:
    snapshots = tuple(
        reverify_v3_evidence(item, reference_time_utc=reference_time_utc)
        for item in (primary, refined, enlarged)
    )
    _shared_identity(snapshots)
    base, fine, large = (snapshot.field_map for snapshot in snapshots)
    if (
        len(fine.r_m) < len(base.r_m)
        or len(fine.z_m) < len(base.z_m)
        or not (
            len(fine.r_m) > len(base.r_m)
            or len(fine.z_m) > len(base.z_m)
        )
    ):
        raise EvidenceVerificationError("v4 refined map is not higher resolution")
    if (
        fine.r_m[0] != base.r_m[0]
        or fine.r_m[-1] != base.r_m[-1]
        or fine.z_m[0] != base.z_m[0]
        or fine.z_m[-1] != base.z_m[-1]
    ):
        raise EvidenceVerificationError(
            "v4 refined map must preserve the primary domain"
        )
    if (
        large.r_m[0] > base.r_m[0]
        or large.r_m[-1] < base.r_m[-1]
        or large.z_m[0] > base.z_m[0]
        or large.z_m[-1] < base.z_m[-1]
        or not (
            large.r_m[0] < base.r_m[0]
            or large.r_m[-1] > base.r_m[-1]
            or large.z_m[0] < base.z_m[0]
            or large.z_m[-1] > base.z_m[-1]
        )
    ):
        raise EvidenceVerificationError(
            "v4 enlarged map must contain and extend the primary domain"
        )
    snapshot = _V4MapSetSnapshot(
        (primary, refined, enlarged),
        tuple(item.field_map.full_map_hash for item in snapshots),
        tuple(_evidence_fingerprint(item) for item in snapshots),
    )
    return AcceptedV4MapSet(
        snapshot, _set_hash(snapshot), _factory_key=_V4_SET_KEY
    )


def reverify_v4_map_set(
    value: object,
    *,
    reference_time_utc: datetime | None = None,
) -> tuple[_V3Snapshot, _V3Snapshot, _V3Snapshot]:
    if not isinstance(value, AcceptedV4MapSet):
        raise EvidenceVerificationError("v4 build requires AcceptedV4MapSet")
    snapshot, stored_hash = value._components(_factory_key=_V4_SET_KEY)
    if _set_hash(snapshot) != stored_hash:
        raise EvidenceVerificationError("v4 map-set invariant was modified")
    rebuilt = verify_v4_map_set(
        *snapshot.evidence, reference_time_utc=reference_time_utc
    )
    rebuilt_snapshot, _ = rebuilt._components(_factory_key=_V4_SET_KEY)
    if (
        rebuilt_snapshot.map_hashes != snapshot.map_hashes
        or rebuilt_snapshot.evidence_fingerprints
        != snapshot.evidence_fingerprints
    ):
        raise EvidenceVerificationError("v4 map set no longer matches evidence")
    return tuple(
        reverify_v3_evidence(item, reference_time_utc=reference_time_utc)
        for item in snapshot.evidence
    )


def v4_map_set_evidence_fingerprints(
    value: object,
    *,
    reference_time_utc: datetime,
) -> tuple[str, str, str]:
    """Reverify exact artifacts and return complete role-ordered fingerprints."""

    snapshots = reverify_v4_map_set(
        value,
        reference_time_utc=reference_time_utc,
    )
    return tuple(_evidence_fingerprint(snapshot) for snapshot in snapshots)
