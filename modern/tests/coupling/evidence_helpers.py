from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from cft_revival.coupling import (
    AcceptedArtifactClaims,
    AdapterVersionContract,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    hash_axisymmetric_map,
    source_map_binding_hash,
    verify_accepted_field_artifact,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


@dataclass
class AnalyticMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


class AcceptedTestAdapter:
    adapter_id = "tests.accepted-l1a-adapter"
    adapter_code_hash = "a" * 64
    version_contract = AdapterVersionContract(
        contract_id="cft-l1a-direct-adapter",
        contract_version="1.0.0",
        input_schema_version="cft-axisymmetric-field-map/1.1.0",
        normalized_schema_version="cft-axisymmetric-field-map/1.1.0",
        model_level="L1a",
    )

    def __init__(
        self,
        claims: AcceptedArtifactClaims,
        *,
        version_contract: AdapterVersionContract | None = None,
    ) -> None:
        self.claims = claims
        if version_contract is not None:
            self.version_contract = version_contract

    def verify_artifact(self, artifact_bytes: bytes) -> AcceptedArtifactClaims:
        return self.claims


def two_cusp_map(points: int = 17, *, inner_radius_m: float = 0.0) -> AnalyticMap:
    r_m = (inner_radius_m, 0.5, 1.0)
    z_m = tuple(-2.0 + 4.0 * index / (points - 1) for index in range(points))
    bz = tuple(z * z - 1.0 for z in z_m)
    return AnalyticMap(
        r_m=r_m,
        z_m=z_m,
        b_r_t=tuple(tuple(2.0 * r for _ in z_m) for r in r_m),
        b_z_t=(bz, bz, bz),
    )


def claims_for(
    field: AnalyticMap,
    artifact_bytes: bytes = b'{"accepted":"analytic-l1a"}',
) -> AcceptedArtifactClaims:
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    map_hash = hash_axisymmetric_map(
        field.r_m, field.z_m, field.b_r_t, field.b_z_t
    )
    source_hash = "2" * 64
    return AcceptedArtifactClaims(
        field_map=field,
        artifact_schema_version="cft-axisymmetric-field-map/1.1.0",
        model_level="L1a",
        artifact_hash=artifact_hash,
        map_content_hash=map_hash,
        source_hash=source_hash,
        source_map_binding_hash=source_map_binding_hash(
            map_hash, source_hash, artifact_hash
        ),
        backend_id="python-reference",
        backend_version="1.0",
        field_model_id="analytic-two-cusp",
        field_model_hash="1" * 64,
        code_hash="3" * 64,
        config_hash="4" * 64,
        generated_at_utc=NOW,
        diagnostics=SolverDiagnosticsEvidence(
            converged=True,
            residual_norm=1.0e-12,
            residual_tolerance=1.0e-10,
            relative_residual=1.0e-11,
            relative_tolerance=1.0e-9,
            iterations=12,
        ),
    )


def accepted_evidence(
    field: AnalyticMap | None = None,
    *,
    artifact_bytes: bytes = b'{"accepted":"analytic-l1a"}',
    claims: AcceptedArtifactClaims | None = None,
    policy: MapValidationPolicy = MapValidationPolicy(),
):
    selected_field = two_cusp_map() if field is None else field
    selected_claims = (
        claims_for(selected_field, artifact_bytes) if claims is None else claims
    )
    return verify_accepted_field_artifact(
        artifact_bytes,
        AcceptedTestAdapter(selected_claims),
        policy,
        reference_time_utc=NOW,
    )


def changed_claims(
    claims: AcceptedArtifactClaims, **changes: object
) -> AcceptedArtifactClaims:
    return replace(claims, **changes)
