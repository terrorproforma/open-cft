from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from cft_revival.coupling import (
    EvidenceVerificationError,
    AcceptedFieldEvidence,
    AdapterVersionContract,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    build_screening_proxy,
    verify_accepted_field_artifact,
)
from cft_revival.coupling.models import _EVIDENCE_FACTORY_KEY
from tests.coupling.evidence_helpers import (
    NOW,
    AcceptedTestAdapter,
    accepted_evidence,
    claims_for,
    two_cusp_map,
)

build_coupling_record = build_screening_proxy
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*deprecated screening_proxy.*:DeprecationWarning"
)


def test_record_builder_fails_closed_without_sealed_accepted_evidence() -> None:
    with pytest.raises(EvidenceVerificationError, match="AcceptedFieldEvidence"):
        build_coupling_record(two_cusp_map(), wall_radius_m=0.75)  # type: ignore[arg-type]
    with pytest.raises(EvidenceVerificationError, match="Adapter"):
        verify_accepted_field_artifact(
            b"artifact",
            object(),  # type: ignore[arg-type]
            reference_time_utc=NOW,
        )


def test_evidence_is_not_publicly_constructible_or_dataclass_replaceable() -> None:
    evidence = accepted_evidence()
    with pytest.raises(TypeError, match="private"):
        AcceptedFieldEvidence(None, "forged")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dataclass"):
        replace(evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact_bytes",
        "map",
        "map_hash",
        "time",
        "diagnostics",
        "residual",
        "identity",
        "binding",
    ],
)
def test_build_reverifies_replacement_forgery_invariants(mutation: str) -> None:
    evidence = accepted_evidence()
    snapshot, invariant_hash = evidence._components_for_reverification(
        _factory_key=_EVIDENCE_FACTORY_KEY
    )
    if mutation == "artifact_bytes":
        forged_snapshot = replace(snapshot, artifact_bytes=b"changed")
    elif mutation == "map":
        rows = list(snapshot.field_map.b_z_t)
        rows[0] = tuple(value + 0.125 for value in rows[0])
        forged_snapshot = replace(
            snapshot,
            field_map=replace(snapshot.field_map, b_z_t=tuple(rows)),
        )
    elif mutation == "map_hash":
        forged_snapshot = replace(
            snapshot,
            field_map=replace(snapshot.field_map, field_map_hash="f" * 64),
        )
    elif mutation == "time":
        forged_snapshot = replace(
            snapshot,
            generated_at_utc=NOW - timedelta(days=2),
        )
    elif mutation == "diagnostics":
        forged_snapshot = replace(
            snapshot,
            diagnostics=replace(snapshot.diagnostics, converged=False),
        )
    elif mutation == "residual":
        forged_snapshot = replace(
            snapshot,
            diagnostics=replace(
                snapshot.diagnostics,
                residual_norm=2.0 * snapshot.diagnostics.residual_tolerance,
            ),
        )
    elif mutation == "identity":
        forged_snapshot = replace(snapshot, code_hash="e" * 64)
    else:
        forged_snapshot = replace(snapshot, source_map_binding_hash="f" * 64)
    forged = AcceptedFieldEvidence(
        forged_snapshot,
        invariant_hash,
        _factory_key=_EVIDENCE_FACTORY_KEY,
    )
    with pytest.raises(EvidenceVerificationError):
        build_coupling_record(
            forged,
            wall_radius_m=0.75,
            reference_time_utc=NOW,
        )


def test_build_rechecks_freshness_even_after_valid_token_issuance() -> None:
    evidence = accepted_evidence()
    with pytest.raises(EvidenceVerificationError, match="stale"):
        build_coupling_record(
            evidence,
            wall_radius_m=0.75,
            reference_time_utc=NOW + timedelta(days=2),
        )


def test_l1a_11_is_default_and_10_requires_allowlisted_migration_adapter() -> None:
    artifact = b'{"accepted":"analytic-l1a"}'
    current_claims = claims_for(two_cusp_map(), artifact)
    current = verify_accepted_field_artifact(
        artifact,
        AcceptedTestAdapter(current_claims),
        reference_time_utc=NOW,
    )
    assert build_coupling_record(
        current,
        wall_radius_m=0.75,
        reference_time_utc=NOW,
    ).artifact_schema_version.endswith("/1.1.0")
    legacy_claims = replace(
        current_claims,
        artifact_schema_version="cft-axisymmetric-field-map/1.0.0",
    )
    migration_contract = AdapterVersionContract(
        contract_id="validated-l1a-10-to-11",
        contract_version="1.0.0",
        input_schema_version="cft-axisymmetric-field-map/1.0.0",
        normalized_schema_version="cft-axisymmetric-field-map/1.1.0",
        model_level="L1a",
        is_migration=True,
    )
    migration_adapter = AcceptedTestAdapter(
        legacy_claims, version_contract=migration_contract
    )
    with pytest.raises(EvidenceVerificationError, match="validated migration"):
        verify_accepted_field_artifact(
            artifact,
            migration_adapter,
            reference_time_utc=NOW,
        )
    migrated = verify_accepted_field_artifact(
        artifact,
        migration_adapter,
        MapValidationPolicy(
            validated_migration_adapter_ids=(migration_adapter.adapter_id,)
        ),
        reference_time_utc=NOW,
    )
    migrated_record = build_coupling_record(
        migrated,
        wall_radius_m=0.75,
        reference_time_utc=NOW,
    )
    assert migrated_record.artifact_schema_version.endswith("/1.0.0")
    assert migrated_record.adapter_normalized_schema_version.endswith("/1.1.0")
    assert migrated_record.adapter_is_migration


@pytest.mark.parametrize(
    ("change", "value", "message"),
    [
        ("artifact_hash", "f" * 64, "exact bytes"),
        ("map_content_hash", "f" * 64, "canonical map bytes"),
        ("source_map_binding_hash", "f" * 64, "not bound"),
        ("artifact_schema_version", "unknown/9", "schema"),
        ("model_level", "unaccepted", "model level"),
        ("backend_id", "", "backend_id"),
        ("code_hash", "bad", "code_hash"),
    ],
)
def test_claim_strings_cannot_replace_content_verification(
    change: str, value: object, message: str
) -> None:
    artifact = b'{"accepted":"analytic-l1a"}'
    claims = replace(claims_for(two_cusp_map(), artifact), **{change: value})
    with pytest.raises(EvidenceVerificationError, match=message):
        verify_accepted_field_artifact(
            artifact,
            AcceptedTestAdapter(claims),
            reference_time_utc=NOW,
        )


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (
            SolverDiagnosticsEvidence(False, 0.0, 1.0, 0.0, 1.0, 1),
            "converged",
        ),
        (
            SolverDiagnosticsEvidence(True, float("nan"), 1.0, 0.0, 1.0, 1),
            "finite",
        ),
        (
            SolverDiagnosticsEvidence(True, 2.0, 1.0, 0.0, 1.0, 1),
            "absolute",
        ),
        (
            SolverDiagnosticsEvidence(True, 0.0, 1.0, 2.0, 1.0, 1),
            "relative",
        ),
    ],
)
def test_diagnostics_must_be_finite_converged_and_under_tolerance(
    diagnostics: SolverDiagnosticsEvidence, message: str
) -> None:
    claims = replace(claims_for(two_cusp_map()), diagnostics=diagnostics)
    with pytest.raises(EvidenceVerificationError, match=message):
        verify_accepted_field_artifact(
            b'{"accepted":"analytic-l1a"}',
            AcceptedTestAdapter(claims),
            reference_time_utc=NOW,
        )


def test_freshness_is_verified_before_evidence_token_is_issued() -> None:
    claims = replace(
        claims_for(two_cusp_map()),
        generated_at_utc=NOW - timedelta(seconds=11),
    )
    with pytest.raises(EvidenceVerificationError, match="stale"):
        verify_accepted_field_artifact(
            b'{"accepted":"analytic-l1a"}',
            AcceptedTestAdapter(claims),
            MapValidationPolicy(maximum_age_s=10.0),
            reference_time_utc=NOW,
        )


def test_all_negative_radii_are_rejected_even_without_required_axis() -> None:
    field = two_cusp_map(inner_radius_m=-1.0e-15)
    claims = claims_for(field)
    with pytest.raises(EvidenceVerificationError, match="non-negative"):
        accepted_evidence(
            field,
            claims=claims,
            policy=MapValidationPolicy(require_axis=False),
        )
