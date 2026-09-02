"""Experiment-local trust adapter for accepted L1a artifacts (serialization v1.2).

The adapter never constructs a field map from experiment declarations.  It
loads the accepted artifact bytes and the accepted design manifest through the
public ``cft_revival.fields`` serialization v1.2 loaders (closed schema,
signed-zero-normalized values, canonical file bytes), binds the artifact to its
accepted manifest entry, and passes only the reloaded ``field_map`` through the
coupling v2 accepted-evidence factory.

The accepted schema is pinned here on purpose.  If ``cft_revival.fields`` moves
to another serialization the experiment must fail loudly and be re-audited
rather than silently follow the package default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.coupling import (
    AcceptedArtifactClaims,
    AdapterVersionContract,
    EvidenceVerificationError,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    hash_axisymmetric_map,
    source_map_binding_hash,
    verify_accepted_field_artifact,
)
from cft_revival.fields import (
    ARTIFACT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    FieldArtifactValidationError,
    canonical_payload_sha256,
    field_artifact_canonical_bytes,
    reload_field_artifact_bytes,
    validate_design_manifest,
    validate_design_manifest_file,
)
from cft_revival.fields.serialization import (
    CANONICALIZATION_V2,
    parse_field_json_bytes,
)

# Accepted field serialization pin (see spec/fields/legacy-serialization-v1.1.json
# and examples/axisymmetric/results/serialization-migration-v1.1-to-v1.2.json).
ARTIFACT_SCHEMA = "cft-axisymmetric-field-map/1.2.0"
MANIFEST_SCHEMA = "cft-axisymmetric-design-manifest/1.2.0"
FIELD_CANONICALIZATION = "field-json-sorted-utf8-signed-zero-v2"
ACCEPTANCE_TIME_UTC = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
# Canonicalization of this experiment's own sealed outputs (dataset/manifest).
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
# Coupling v2 still defaults ``current_artifact_schema`` to v1.1; the experiment
# accepts only v1.2 bytes, so its map policy names v1.2 explicitly.
ACCEPTED_MAP_POLICY = MapValidationPolicy(
    maximum_age_s=None,
    current_artifact_schema=ARTIFACT_SCHEMA,
)
_LIBRARY_DEFAULT_SCHEMA = MapValidationPolicy().current_artifact_schema


def _assert_serialization_pin() -> None:
    """Fail closed if the accepted package no longer serializes v1.2."""

    current = (ARTIFACT_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION, CANONICALIZATION_V2)
    pinned = (ARTIFACT_SCHEMA, MANIFEST_SCHEMA, FIELD_CANONICALIZATION)
    if current != pinned:
        raise ValueError(
            "cft_revival.fields serialization moved from the experiment pin "
            f"{pinned} to {current}; re-audit experiments/l1a_plasma_coupling"
        )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {label}")

    try:
        loaded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return loaded


def _verify_sidecar(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != expected:
        raise ValueError(f"invalid SHA-256 sidecar for {path.name}")
    return digest


def accepted_artifact_document(artifact_bytes: bytes, *, source: str) -> dict[str, Any]:
    """Reload exact accepted artifact bytes through the public v1.2 loader.

    ``reload_field_artifact_bytes`` performs the closed-schema validation and
    requires canonical v1.2 file bytes; legacy v1.1 reads are disabled so the
    experiment can never silently consume the historical serialization.  The
    canonical round-trip is re-checked explicitly, mirroring the accepted
    coupling v4 field adapter.
    """

    try:
        artifact = reload_field_artifact_bytes(
            artifact_bytes,
            source=source,
            allow_legacy_v1_1=False,
        )
        canonical = field_artifact_canonical_bytes(artifact)
    except FieldArtifactValidationError as error:
        raise EvidenceVerificationError(
            f"serialization v1.2 loader rejected {source}: {error}"
        ) from error
    if canonical != artifact_bytes:
        raise EvidenceVerificationError(
            f"{source} is not canonical serialization v1.2 file bytes"
        )
    if artifact["schema_version"] != ARTIFACT_SCHEMA:
        raise EvidenceVerificationError(
            f"{source} schema {artifact['schema_version']!r} is not the accepted "
            f"{ARTIFACT_SCHEMA!r}"
        )
    if artifact["integrity"]["canonicalization"] != FIELD_CANONICALIZATION:
        raise EvidenceVerificationError(
            f"{source} does not declare {FIELD_CANONICALIZATION!r}"
        )
    return artifact


def accepted_manifest_document(manifest_bytes: bytes, *, source: str) -> dict[str, Any]:
    """Parse exact accepted manifest bytes with the v1.2 canonical-bytes parser."""

    try:
        manifest = parse_field_json_bytes(
            manifest_bytes,
            source=source,
            require_canonical_file_bytes=True,
        )
        validate_design_manifest(manifest)
    except FieldArtifactValidationError as error:
        raise ValueError(
            f"serialization v1.2 loader rejected {source}: {error}"
        ) from error
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("unsupported accepted manifest schema")
    if manifest["integrity"]["canonicalization"] != FIELD_CANONICALIZATION:
        raise ValueError(f"{source} does not declare {FIELD_CANONICALIZATION!r}")
    return manifest


@dataclass(frozen=True)
class SerializedMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


def _serialized_map(artifact: Mapping[str, Any]) -> SerializedMap:
    """Carry the reloaded (signed-zero-normalized) v1.2 field map unchanged."""

    field_map = artifact["field_map"]
    return SerializedMap(
        r_m=tuple(field_map["r_m"]),
        z_m=tuple(field_map["z_m"]),
        b_r_t=tuple(tuple(row) for row in field_map["b_r_t"]),
        b_z_t=tuple(tuple(row) for row in field_map["b_z_t"]),
    )


def _producer_code_hash(modern_root: Path, provenance: Mapping[str, Any]) -> str:
    """Hash the exact accepted implementation/ledger files available at replay."""

    relative_paths = (
        "src/cft_revival/fields/artifacts.py",
        "src/cft_revival/fields/models.py",
        "src/cft_revival/fields/numerics.py",
        "src/cft_revival/fields/serialization.py",
        "src/cft_revival/fields/verification.py",
        "src/cft_revival/fields/warp_solver.py",
        "spec/fields/equation-solver-ledger-v1.json",
    )
    files = []
    for relative in relative_paths:
        path = modern_root / relative
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return stable_hash({"declared_provenance": dict(provenance), "files": files})


class AcceptedL1aV12Adapter:
    """Closed-schema v1.2 adapter bound to one accepted manifest entry."""

    adapter_id = "experiments.l1a-plasma-coupling.accepted-l1a-v1.2"
    version_contract = AdapterVersionContract(
        contract_id="cft-l1a-direct-artifact-adapter",
        contract_version="2.0.0",
        input_schema_version=ARTIFACT_SCHEMA,
        normalized_schema_version=ARTIFACT_SCHEMA,
        model_level="L1a",
    )

    def __init__(
        self,
        *,
        manifest_entry: Mapping[str, Any],
        manifest_file_sha256: str,
        manifest_payload_sha256: str,
        generated_at_utc: datetime,
        producer_code_hash: str,
    ) -> None:
        self._entry = dict(manifest_entry)
        self._manifest_file_sha256 = manifest_file_sha256
        self._manifest_payload_sha256 = manifest_payload_sha256
        self._generated_at_utc = generated_at_utc
        self._producer_code_hash = producer_code_hash
        self.adapter_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def verify_artifact(self, artifact_bytes: bytes) -> AcceptedArtifactClaims:
        artifact = accepted_artifact_document(
            artifact_bytes, source=self._entry["artifact"]
        )
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_hash != self._entry["artifact_file_sha256"]:
            raise EvidenceVerificationError(
                "artifact bytes do not match the accepted manifest file hash"
            )
        if artifact["integrity"]["payload_sha256"] != self._entry["artifact_payload_sha256"]:
            raise EvidenceVerificationError(
                "artifact payload does not match the accepted manifest"
            )
        if artifact["input"]["name"] != self._entry["name"]:
            raise EvidenceVerificationError(
                "artifact name does not match the accepted manifest"
            )
        for key in ("backend", "iterations", "relative_residual_l2"):
            if artifact["diagnostics"][key] != self._entry[key]:
                raise EvidenceVerificationError(
                    f"artifact diagnostic {key} does not match manifest"
                )
        for key in ("b_magnitude_min_t", "b_magnitude_max_t", "topology"):
            if artifact["summary"][key] != self._entry[key]:
                raise EvidenceVerificationError(
                    f"artifact summary {key} does not match manifest"
                )

        field = _serialized_map(artifact)
        map_hash = hash_axisymmetric_map(
            field.r_m, field.z_m, field.b_r_t, field.b_z_t
        )
        # Semantic identities use the v1.2 canonical payload hash of the
        # reloaded (normalized) document, the same canonicalization that seals
        # the accepted artifact itself.
        source_hash = canonical_payload_sha256(
            {
                "sources": artifact["input"]["sources"],
                "source_convention": artifact["input"]["source_convention"],
            }
        )
        config_hash = canonical_payload_sha256(
            {
                "domain": artifact["input"]["domain"],
                "outer_boundary": artifact["input"]["outer_boundary"],
                "permeability_h_per_m": artifact["input"]["permeability_h_per_m"],
                "solver": artifact["input"]["solver"],
                "field_map_downsample_stride": artifact["field_map"]["downsample_stride"],
            }
        )
        field_model_hash = canonical_payload_sha256(
            {
                "schema_version": artifact["schema_version"],
                "canonicalization": artifact["integrity"]["canonicalization"],
                "model_level": artifact["model_level"],
                "model_description": artifact["model_description"],
                "provenance": artifact["provenance"],
                "producer_code_hash": self._producer_code_hash,
            }
        )
        diagnostics = artifact["diagnostics"]
        solver = artifact["input"]["solver"]
        absolute_tolerance = max(
            float(solver["absolute_tolerance"]),
            float(solver["relative_tolerance"])
            * float(diagnostics["initial_residual_l2"]),
        )
        return AcceptedArtifactClaims(
            field_map=field,
            artifact_schema_version=artifact["schema_version"],
            model_level=artifact["model_level"],
            artifact_hash=artifact_hash,
            map_content_hash=map_hash,
            source_hash=source_hash,
            source_map_binding_hash=source_map_binding_hash(
                map_hash, source_hash, artifact_hash
            ),
            backend_id=f"cft_revival.fields/{artifact['diagnostics']['backend']}",
            backend_version="artifact-schema-1.2.0",
            field_model_id=artifact["input"]["name"],
            field_model_hash=field_model_hash,
            code_hash=self._producer_code_hash,
            config_hash=config_hash,
            generated_at_utc=self._generated_at_utc,
            diagnostics=SolverDiagnosticsEvidence(
                converged=diagnostics["converged"],
                residual_norm=float(diagnostics["final_residual_l2"]),
                residual_tolerance=absolute_tolerance,
                relative_residual=float(diagnostics["relative_residual_l2"]),
                relative_tolerance=float(solver["relative_tolerance"]),
                iterations=int(diagnostics["iterations"]),
            ),
        )

    @property
    def manifest_identity(self) -> dict[str, str]:
        return {
            "manifest_file_sha256": self._manifest_file_sha256,
            "manifest_payload_sha256": self._manifest_payload_sha256,
        }


def _pinned_policy(policy: MapValidationPolicy) -> MapValidationPolicy:
    """Pin the coupling v2 policy to the accepted v1.2 schema.

    Callers tune staleness/geometry through ``policy``; the accepted artifact
    schema is an adapter property.  The coupling v2 library default still names
    v1.1, so that default is pinned to v1.2 here; any other explicit schema is a
    caller error.
    """

    if policy.current_artifact_schema not in {ARTIFACT_SCHEMA, _LIBRARY_DEFAULT_SCHEMA}:
        raise ValueError(
            "map validation policy names an artifact schema this experiment "
            f"does not accept: {policy.current_artifact_schema!r}"
        )
    return replace(policy, current_artifact_schema=ARTIFACT_SCHEMA)


def load_accepted_evidence(
    artifact_path: Path,
    manifest_path: Path,
    *,
    generated_at_utc: datetime = ACCEPTANCE_TIME_UTC,
    policy: MapValidationPolicy = ACCEPTED_MAP_POLICY,
    reference_time_utc: datetime = ACCEPTANCE_TIME_UTC,
):
    """Verify accepted files and issue coupling v2 evidence from exact bytes.

    Order of gates (each fails closed):

    1. exact-bytes SHA-256 sidecars of the manifest and the artifact;
    2. manifest bytes through the v1.2 canonical-bytes parser and closed
       manifest schema; the artifact must be listed and its exact bytes must
       match the accepted manifest file hash;
    3. the whole accepted set through ``validate_design_manifest_file`` (the
       public v1.2 file loader cross-validates every listed artifact);
    4. the artifact bytes through the coupling v2 accepted-evidence factory
       using this experiment's v1.2 adapter.
    """

    _assert_serialization_pin()
    policy = _pinned_policy(policy)
    artifact_path = artifact_path.resolve()
    manifest_path = manifest_path.resolve()
    if artifact_path.parent != manifest_path.parent:
        raise ValueError("accepted artifact and manifest must share one directory")
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_hash = _verify_sidecar(manifest_path, manifest_bytes)
    artifact_bytes = artifact_path.read_bytes()
    artifact_file_hash = _verify_sidecar(artifact_path, artifact_bytes)
    manifest = accepted_manifest_document(manifest_bytes, source=manifest_path.name)
    entry = next(
        (
            item
            for item in manifest["designs"]
            if item["artifact"] == artifact_path.name
        ),
        None,
    )
    if entry is None:
        raise ValueError("artifact is absent from accepted manifest")
    if artifact_file_hash != entry["artifact_file_sha256"]:
        raise EvidenceVerificationError(
            "artifact bytes do not match the accepted manifest file hash"
        )
    accepted_set = validate_design_manifest_file(manifest_path)
    if accepted_set != manifest:
        raise EvidenceVerificationError(
            "accepted manifest file loader and byte parser disagree"
        )
    artifact = accepted_artifact_document(artifact_bytes, source=artifact_path.name)
    modern_root = Path(__file__).resolve().parents[2]
    adapter = AcceptedL1aV12Adapter(
        manifest_entry=entry,
        manifest_file_sha256=manifest_file_hash,
        manifest_payload_sha256=manifest["integrity"]["payload_sha256"],
        generated_at_utc=generated_at_utc,
        producer_code_hash=_producer_code_hash(modern_root, artifact["provenance"]),
    )
    evidence = verify_accepted_field_artifact(
        artifact_bytes,
        adapter,
        policy,
        reference_time_utc=reference_time_utc,
    )
    claims = adapter.verify_artifact(artifact_bytes)
    acceptance_identity = {
        **adapter.manifest_identity,
        "artifact_hash": claims.artifact_hash,
        "field_map_hash": claims.map_content_hash,
        "source_hash": claims.source_hash,
        "source_map_binding_hash": claims.source_map_binding_hash,
        "artifact_schema_version": claims.artifact_schema_version,
        "field_model_hash": claims.field_model_hash,
        "code_hash": claims.code_hash,
        "config_hash": claims.config_hash,
        "adapter_code_hash": adapter.adapter_code_hash,
        "diagnostics": {
            "converged": claims.diagnostics.converged,
            "residual_norm": claims.diagnostics.residual_norm,
            "residual_tolerance": claims.diagnostics.residual_tolerance,
            "relative_residual": claims.diagnostics.relative_residual,
            "relative_tolerance": claims.diagnostics.relative_tolerance,
            "iterations": claims.diagnostics.iterations,
        },
    }
    return evidence, acceptance_identity
