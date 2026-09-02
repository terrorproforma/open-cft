"""Authoritative field-schema v1.2 adapter for coupling v4 evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from ..fields import (
    ARTIFACT_SCHEMA_VERSION,
    canonical_field_artifact_bytes,
    field_artifact_canonical_bytes,
    reload_field_artifact_bytes,
)
from ..fields.serialization import CANONICALIZATION_V2
from .models import (
    AdapterVersionContract,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
)
from .v3_evidence import (
    AcceptedV3FieldEvidence,
    v3_evidence_binding_hash,
    verify_v3_field_artifact,
)
from .v3_models import V3ArtifactClaims
from .v4_evidence import V4_FIELD_ADAPTER_ID


@dataclass(frozen=True, slots=True)
class CanonicalFieldV12Binding:
    """External identities absent from the closed field artifact itself."""

    geometry_hash: str
    code_hash: str
    backend_version: str
    generated_at_utc: datetime


@dataclass(frozen=True, slots=True)
class _ReloadedPsiMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    psi_wb: tuple[tuple[float, ...], ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


V4_FIELD_MAP_POLICY = MapValidationPolicy(
    current_artifact_schema=ARTIFACT_SCHEMA_VERSION,
)
V4_FIELD_CANONICALIZATION = CANONICALIZATION_V2
_ADAPTER_VERSION = "1.0.0"
_ADAPTER_CODE_HASH = hashlib.sha256(
    b"cft.coupling.authoritative-field-v1.2/1.0.0"
).hexdigest()


def _semantic_hash(label: bytes, value: object) -> str:
    encoded = canonical_field_artifact_bytes(value, representation="payload")
    return hashlib.sha256(label + b"\0" + encoded).hexdigest()


class CanonicalFieldV12Adapter:
    """Reload exact canonical bytes and derive coupling claims deterministically."""

    adapter_id = V4_FIELD_ADAPTER_ID
    adapter_code_hash = _ADAPTER_CODE_HASH
    version_contract = AdapterVersionContract(
        contract_id="cft-coupling-field-v1.2-direct",
        contract_version=_ADAPTER_VERSION,
        input_schema_version=ARTIFACT_SCHEMA_VERSION,
        normalized_schema_version=ARTIFACT_SCHEMA_VERSION,
        model_level="L1a",
    )

    def __init__(self, binding: CanonicalFieldV12Binding) -> None:
        self.binding = binding

    def verify_v3_artifact(self, artifact_bytes: bytes) -> V3ArtifactClaims:
        artifact = reload_field_artifact_bytes(
            artifact_bytes,
            source="coupling-v4-field-artifact",
            allow_legacy_v1_1=False,
        )
        if field_artifact_canonical_bytes(artifact) != artifact_bytes:
            raise ValueError("field artifact differs from canonical v1.2 bytes")
        field = artifact["field_map"]
        inputs = artifact["input"]
        diagnostics = artifact["diagnostics"]
        provenance = artifact["provenance"]
        solver = inputs["solver"]
        domain = inputs["domain"]
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        field_map = _ReloadedPsiMap(
            tuple(field["r_m"]),
            tuple(field["z_m"]),
            tuple(tuple(row) for row in field["psi_wb"]),
            tuple(tuple(row) for row in field["b_r_t"]),
            tuple(tuple(row) for row in field["b_z_t"]),
        )
        from .v3_evidence import hash_psi_map

        map_hash = hash_psi_map(field_map)
        source_hash = _semantic_hash(
            b"cft-field-v1.2-source",
            {
                "sources": inputs["sources"],
                "source_convention": inputs["source_convention"],
            },
        )
        material_hash = _semantic_hash(
            b"cft-field-v1.2-material",
            {"permeability_h_per_m": inputs["permeability_h_per_m"]},
        )
        mesh_hash = _semantic_hash(
            b"cft-field-v1.2-mesh",
            {
                "r_m": field["r_m"],
                "z_m": field["z_m"],
                "downsample_stride": field["downsample_stride"],
            },
        )
        domain_hash = _semantic_hash(b"cft-field-v1.2-domain", domain)
        field_model_hash = _semantic_hash(
            b"cft-field-v1.2-model",
            {
                "model_description": artifact["model_description"],
                "implementation": provenance["implementation"],
                "scalar": provenance["scalar"],
                "equation_ledger": provenance["equation_ledger"],
                "outer_boundary": inputs["outer_boundary"],
            },
        )
        config_hash = _semantic_hash(b"cft-field-v1.2-config", solver)
        absolute_tolerance = float(solver["absolute_tolerance"])
        relative_tolerance = float(solver["relative_tolerance"])
        initial_residual = float(diagnostics["initial_residual_l2"])
        residual_tolerance = max(
            absolute_tolerance,
            relative_tolerance * initial_residual,
        )
        return V3ArtifactClaims(
            field_map=field_map,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            model_level=artifact["model_level"],
            artifact_hash=artifact_hash,
            full_map_hash=map_hash,
            source_hash=source_hash,
            geometry_hash=self.binding.geometry_hash,
            material_hash=material_hash,
            mesh_hash=mesh_hash,
            domain_hash=domain_hash,
            evidence_binding_hash=v3_evidence_binding_hash(
                map_hash,
                source_hash,
                self.binding.geometry_hash,
                material_hash,
                mesh_hash,
                domain_hash,
                artifact_hash,
            ),
            backend_id=diagnostics["backend"],
            backend_version=self.binding.backend_version,
            field_model_id="cft-revival-fields-l1a-fdm",
            field_model_hash=field_model_hash,
            code_hash=self.binding.code_hash,
            config_hash=config_hash,
            generated_at_utc=self.binding.generated_at_utc,
            diagnostics=SolverDiagnosticsEvidence(
                converged=diagnostics["converged"],
                residual_norm=diagnostics["final_residual_l2"],
                residual_tolerance=residual_tolerance,
                relative_residual=diagnostics["relative_residual_l2"],
                relative_tolerance=relative_tolerance,
                iterations=diagnostics["iterations"],
            ),
        )


def verify_canonical_field_v12_artifact(
    artifact_bytes: bytes,
    binding: CanonicalFieldV12Binding,
    policy: MapValidationPolicy = V4_FIELD_MAP_POLICY,
    *,
    reference_time_utc: datetime,
    migration_manifest_bytes: bytes | None = None,
    migration_source_artifact_bytes: bytes | None = None,
) -> AcceptedV3FieldEvidence:
    """Accept current canonical bytes, optionally bound to a declared migration."""

    return verify_v3_field_artifact(
        artifact_bytes,
        CanonicalFieldV12Adapter(binding),
        policy,
        reference_time_utc=reference_time_utc,
        migration_manifest_bytes=migration_manifest_bytes,
        migration_source_artifact_bytes=migration_source_artifact_bytes,
    )
