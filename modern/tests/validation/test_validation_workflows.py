import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from cft_revival.validation import (
    ApplicabilityRange,
    AuditStatus,
    ClaimLevel,
    ContextOfUseLedger,
    CredibilityLevel,
    EvidenceKind,
    EvidencePartition,
    EvidenceRecord,
    EvidenceRegistry,
    ExperimentMetadata,
    ExperimentUncertainty,
    IndependenceIdentity,
    MetricObservation,
    NamedQuantity,
    Provenance,
    Quantity,
    QuantityRequirement,
    SourceAuthority,
    UncertaintyComponent,
    audit_context_of_use,
    audit_grouped_partitions,
    compare_evidence_records,
    load_published_evidence,
    render_evidence_report,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "data" / "validation" / "yeo-2020-s1-external-evidence-v2.json"


def _experiment(
    record_id: str,
    group_id: str,
    *,
    metric_name: str = "thrust",
    unit: str = "N",
    value: float = 0.1,
    model_revision: str = "model-v2",
    power_w: float = 300.0,
    hardware_article_id: str | None = None,
    campaign_id: str = "campaign-1",
    design_identity: str = "design-1",
    run_lineage_id: str | None = None,
) -> EvidenceRecord:
    hardware = hardware_article_id or group_id
    run_lineage = run_lineage_id or f"run-{record_id}"
    uncertainty = ExperimentUncertainty(
        (UncertaintyComponent("instrument", Quantity(0.001, unit), "type-b"),)
    )
    return EvidenceRecord(
        record_id=record_id,
        kind=EvidenceKind.EXPERIMENT,
        source_authority=SourceAuthority.EXPERIMENT,
        partition=EvidencePartition.VALIDATION,
        observations=(
            MetricObservation(metric_name, Quantity(value, unit), uncertainty),
        ),
        provenance=Provenance(
            "Holdout experiment",
            metric_name,
            "held-out measurement",
            f"artifact://validation/{record_id}",
            is_experimental_truth=True,
        ),
        model_context_id="model-context-v1",
        result_context_id="result-context-v1",
        group_id=group_id,
        independence_identity=IndependenceIdentity(
            design_identity,
            run_lineage,
            hardware,
            campaign_id,
            f"specimen-{hardware}",
        ),
        model_revision=model_revision,
        code_revision="code-v3",
        operating_parameters=(NamedQuantity("power", Quantity(power_w, "W")),),
        credibility=CredibilityLevel.VALIDATION_SUPPORT,
        maximum_claim=ClaimLevel.VALIDATION_EVIDENCE,
        experiment_metadata=ExperimentMetadata(
            "Facility A",
            hardware,
            campaign_id,
            "2026-09-01T10:00:00Z",
            (f"instrument-{record_id}",),
            f"artifact://validation/raw/{record_id}",
        ),
    )


def _ledger(*ids: str) -> ContextOfUseLedger:
    return ContextOfUseLedger(
        ledger_id="cou-v2",
        decision="screen candidates",
        consequence_of_wrong_decision="ranking error",
        model_role="decision support",
        quantity_requirements=(QuantityRequirement("thrust", "N", 2, 2),),
        applicability_ranges=(
            ApplicabilityRange("power", Quantity(100.0, "W"), Quantity(500.0, "W")),
        ),
        required_model_context_id="model-context-v1",
        required_result_context_id="result-context-v1",
        required_model_revision="model-v2",
        required_code_revision="code-v3",
        assumptions=("steady state",),
        limitations=("not qualification",),
        validation_evidence_ids=tuple(ids),
    )


def test_context_audit_requires_exact_quantity_units_context_and_groups() -> None:
    records = (
        _experiment(
            "e1",
            "caller-group",
            hardware_article_id="article-1",
            campaign_id="campaign-1",
            design_identity="design-1",
        ),
        _experiment(
            "e2",
            "caller-group",
            hardware_article_id="article-2",
            campaign_id="campaign-2",
            design_identity="design-2",
        ),
    )
    audit = audit_context_of_use(_ledger("e1", "e2"), EvidenceRegistry(records))
    assert audit.status is AuditStatus.PASS
    assert audit.matching_evidence_ids == ("e1", "e2")
    assert not audit.missing_dimensions


def test_context_audit_rejects_calibration_physical_overlap_despite_group_label() -> None:
    validation = _experiment("validation", "validation-caller-group")
    calibration = replace(
        validation,
        record_id="calibration",
        partition=EvidencePartition.CALIBRATION,
        group_id="different-caller-group",
        maximum_claim=ClaimLevel.DIAGNOSTIC_ONLY,
    )
    audit = audit_context_of_use(
        _ledger("validation"), EvidenceRegistry((calibration, validation))
    )
    assert audit.status is AuditStatus.FAIL
    assert any(
        "partition_independence=" in item for item in audit.missing_dimensions
    )


def test_elapsed_time_cannot_satisfy_thrust_and_missing_dimension_is_reported() -> None:
    records = (
        _experiment("elapsed", "article-1", metric_name="elapsed_time", unit="s", value=3.0),
        _experiment("thrust", "article-2"),
    )
    audit = audit_context_of_use(
        _ledger("elapsed", "thrust"), EvidenceRegistry(records)
    )
    assert audit.status is AuditStatus.FAIL
    assert "quantity:thrust:evidence_count=1/2" in audit.missing_dimensions
    assert "quantity:thrust:independent_groups=1/2" in audit.missing_dimensions


@pytest.mark.parametrize(
    ("records", "missing_fragment"),
    [
        (
            (
                _experiment("e1", "article-1"),
                _experiment("e2", "article-2", model_revision="wrong"),
            ),
            "evidence:e2:model_revision=model-v2",
        ),
        (
            (
                _experiment("e1", "article-1"),
                _experiment("e2", "article-2", unit="mN", value=100.0),
            ),
            "quantity:thrust:si_unit=N",
        ),
        (
            (
                _experiment("e1", "article-1"),
                _experiment("e2", "article-2", power_w=600.0),
            ),
            "evidence:e2:applicability_range:power",
        ),
        (
            (
                _experiment("e1", "article-1"),
                _experiment("e2", "article-1"),
            ),
            "quantity:thrust:independent_groups=1/2",
        ),
    ],
)
def test_context_audit_reports_each_missing_bound_dimension(
    records: tuple[EvidenceRecord, ...], missing_fragment: str
) -> None:
    audit = audit_context_of_use(_ledger("e1", "e2"), EvidenceRegistry(records))
    assert audit.status is AuditStatus.FAIL
    assert missing_fragment in audit.missing_dimensions


def test_partition_audit_statuses_empty_insufficient_leaking_and_pass() -> None:
    empty = audit_grouped_partitions(EvidenceRegistry())
    assert empty.status is AuditStatus.NOT_EVALUATED
    validation_only = audit_grouped_partitions(
        EvidenceRegistry((_experiment("v1", "article-1"),))
    )
    assert validation_only.status is AuditStatus.NOT_EVALUATED
    calibration = EvidenceRecord(
        record_id="cal",
        kind=EvidenceKind.EXPERIMENT,
        source_authority=SourceAuthority.EXPERIMENT,
        partition=EvidencePartition.CALIBRATION,
        observations=_experiment("base", "article-1").observations,
        provenance=_experiment("base-2", "article-1").provenance,
        model_context_id="model-context-v1",
        result_context_id="result-context-v1",
        group_id="article-1",
        independence_identity=IndependenceIdentity(
            "calibration-design",
            "calibration-run",
            "article-1",
            "calibration-campaign",
            "calibration-specimen",
        ),
        model_revision="model-v2",
        code_revision="code-v3",
        operating_parameters=(NamedQuantity("power", Quantity(300.0, "W")),),
        maximum_claim=ClaimLevel.DIAGNOSTIC_ONLY,
        experiment_metadata=ExperimentMetadata(
            "Facility A",
            "article-1",
            "calibration-campaign",
            "2026-09-01T10:00:00Z",
            ("instrument-cal",),
            "artifact://validation/raw/cal",
        ),
    )
    leaking = audit_grouped_partitions(
        EvidenceRegistry((calibration, _experiment("v2", "article-1")))
    )
    assert leaking.status is AuditStatus.FAIL
    passed = audit_grouped_partitions(
        EvidenceRegistry((calibration, _experiment("v3", "article-2")))
    )
    assert passed.status is AuditStatus.PASS


def test_report_recomputes_bound_audit_and_never_accepts_stale_audit() -> None:
    assert "partition_audit" not in inspect.signature(render_evidence_report).parameters
    empty_report = render_evidence_report("Empty", EvidenceRegistry())
    assert "Grouped split audit: NOT_EVALUATED" in empty_report
    assert "evidence assessment status is NOT_EVALUATED" in empty_report
    insufficient_report = render_evidence_report(
        "Validation only", EvidenceRegistry((_experiment("e1", "article-1"),))
    )
    assert "Grouped split audit: NOT_EVALUATED" in insufficient_report
    published_report = render_evidence_report(
        "Published", EvidenceRegistry(load_published_evidence(PUBLISHED))
    )
    assert "Grouped split audit: NOT_EVALUATED" in published_report
    assert "MDO (original)" in published_report
    assert "not authentication" in published_report


def test_published_pic_comparison_remains_cross_model_only() -> None:
    records = load_published_evidence(PUBLISHED)
    comparison = compare_evidence_records(records[0], records[1])
    assert comparison.maximum_claim is ClaimLevel.CROSS_MODEL_AGREEMENT
