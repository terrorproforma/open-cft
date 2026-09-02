"""Experiment-local trust adapter for accepted L1a v1.1 artifacts.

The adapter never constructs a field map from experiment declarations.  It
strictly parses the accepted artifact bytes, validates their sealed payload and
accepted manifest entry, and passes only the serialized ``field_map`` through
the coupling v2 accepted-evidence factory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
from cft_revival.fields import validate_design_manifest, validate_field_artifact

ARTIFACT_SCHEMA = "cft-axisymmetric-field-map/1.1.0"
MANIFEST_SCHEMA = "cft-axisymmetric-design-manifest/1.1.0"
ACCEPTANCE_TIME_UTC = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"


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


@dataclass(frozen=True)
class SerializedMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


def _serialized_map(artifact: Mapping[str, Any]) -> SerializedMap:
    field_map = artifact["field_map"]
    return SerializedMap(
        r_m=tuple(float(value) for value in field_map["r_m"]),
        z_m=tuple(float(value) for value in field_map["z_m"]),
        b_r_t=tuple(tuple(float(value) for value in row) for row in field_map["b_r_t"]),
        b_z_t=tuple(tuple(float(value) for value in row) for row in field_map["b_z_t"]),
    )


def _producer_code_hash(modern_root: Path, provenance: Mapping[str, Any]) -> str:
    """Hash the exact accepted implementation/ledger files available at replay."""

    relative_paths = (
        "src/cft_revival/fields/artifacts.py",
        "src/cft_revival/fields/models.py",
        "src/cft_revival/fields/numerics.py",
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


class AcceptedL1aV11Adapter:
    """Closed-schema adapter bound to one accepted manifest entry."""

    adapter_id = "experiments.l1a-plasma-coupling.accepted-l1a-v1.1"
    version_contract = AdapterVersionContract(
        contract_id="cft-l1a-direct-artifact-adapter",
        contract_version="1.0.0",
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
        artifact = strict_json_bytes(artifact_bytes, label=self._entry["artifact"])
        validate_field_artifact(artifact)
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
        source_hash = stable_hash(
            {
                "sources": artifact["input"]["sources"],
                "source_convention": artifact["input"]["source_convention"],
            }
        )
        config_hash = stable_hash(
            {
                "domain": artifact["input"]["domain"],
                "outer_boundary": artifact["input"]["outer_boundary"],
                "permeability_h_per_m": artifact["input"]["permeability_h_per_m"],
                "solver": artifact["input"]["solver"],
                "field_map_downsample_stride": artifact["field_map"]["downsample_stride"],
            }
        )
        field_model_hash = stable_hash(
            {
                "schema_version": artifact["schema_version"],
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
            backend_version="artifact-schema-1.1.0",
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


def load_accepted_evidence(
    artifact_path: Path,
    manifest_path: Path,
    *,
    generated_at_utc: datetime = ACCEPTANCE_TIME_UTC,
    policy: MapValidationPolicy = MapValidationPolicy(maximum_age_s=None),
    reference_time_utc: datetime = ACCEPTANCE_TIME_UTC,
):
    """Verify accepted files and issue coupling v2 evidence from exact bytes."""

    artifact_path = artifact_path.resolve()
    manifest_path = manifest_path.resolve()
    if artifact_path.parent != manifest_path.parent:
        raise ValueError("accepted artifact and manifest must share one directory")
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_hash = _verify_sidecar(manifest_path, manifest_bytes)
    manifest = strict_json_bytes(manifest_bytes, label=manifest_path.name)
    validate_design_manifest(manifest)
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("unsupported accepted manifest schema")
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
    artifact_bytes = artifact_path.read_bytes()
    _verify_sidecar(artifact_path, artifact_bytes)
    modern_root = Path(__file__).resolve().parents[2]
    artifact_preview = strict_json_bytes(artifact_bytes, label=artifact_path.name)
    adapter = AcceptedL1aV11Adapter(
        manifest_entry=entry,
        manifest_file_sha256=manifest_file_hash,
        manifest_payload_sha256=manifest["integrity"]["payload_sha256"],
        generated_at_utc=generated_at_utc,
        producer_code_hash=_producer_code_hash(
            modern_root, artifact_preview["provenance"]
        ),
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
