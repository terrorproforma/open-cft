from math import inf, nan

import pytest

from cft_revival.validation import (
    ApplicabilityRange,
    ClaimLevel,
    ContextOfUseLedger,
    CredibilityLevel,
    EvidenceKind,
    EvidencePartition,
    EvidenceRecord,
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
    UnitError,
    ValidationError,
)


def _uncertainty(unit: str = "N") -> ExperimentUncertainty:
    return ExperimentUncertainty(
        (UncertaintyComponent("calibration", Quantity(0.001, unit), "type-b"),)
    )


def _provenance(
    label: str = "fixture", *, experimental: bool = False
) -> Provenance:
    return Provenance(
        source_title="Owned validation fixture",
        source_native_label=label,
        editorial_interpretation="test-only interpretation",
        source_locator="artifact://validation/fixture",
        is_experimental_truth=experimental,
    )


def _experiment(record_id: str = "experiment-1") -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        kind=EvidenceKind.EXPERIMENT,
        source_authority=SourceAuthority.EXPERIMENT,
        partition=EvidencePartition.VALIDATION,
        observations=(
            MetricObservation("thrust", Quantity(0.1, "N"), _uncertainty()),
        ),
        provenance=_provenance("measured thrust", experimental=True),
        model_context_id="model-context-v1",
        result_context_id="result-context-v1",
        group_id=f"article-{record_id}",
        independence_identity=IndependenceIdentity(
            design_identity="design-1",
            run_lineage_id=f"run-{record_id}",
            hardware_article_id=f"article-{record_id}",
            test_campaign_id="campaign-2026",
            specimen_id=f"specimen-{record_id}",
        ),
        model_revision="model-v2",
        code_revision="code-v3",
        operating_parameters=(NamedQuantity("power", Quantity(300.0, "W")),),
        credibility=CredibilityLevel.VALIDATION_SUPPORT,
        maximum_claim=ClaimLevel.VALIDATION_EVIDENCE,
        experiment_metadata=ExperimentMetadata(
            facility="Facility A",
            hardware_article_id=f"article-{record_id}",
            campaign_id="campaign-2026",
            acquired_at_utc="2026-09-01T10:00:00Z",
            instrument_ids=("balance-1",),
            raw_data_locator="artifact://validation/raw/experiment-1",
        ),
    )


@pytest.mark.parametrize("value", [True, nan, inf, -inf])
def test_quantity_rejects_boolean_and_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        Quantity(value, "N")


def test_units_and_source_locators_are_strict() -> None:
    assert Quantity(102.7, "mN").to("N").value == pytest.approx(0.1027)
    with pytest.raises(UnitError):
        Quantity(1.0, "N").to("s")
    with pytest.raises(ValidationError, match="locator"):
        Provenance("title", "label", "interpretation", "relative/path")


@pytest.mark.parametrize(
    "locator",
    [
        "https://exa mple.com/source",
        "https://user@example.com/source",
        "https://example.com:99999/source",
        "https://-bad.example/source",
        "https:///missing-host",
    ],
)
def test_web_source_locator_rejects_invalid_absolute_authority(locator: str) -> None:
    with pytest.raises(ValidationError):
        Provenance("title", "label", "interpretation", locator)


@pytest.mark.parametrize(
    "doi",
    [
        "10.123/x",
        "10.1234567890/suffix",
        "10.2514/",
        "10.2514/has space",
        "doi:10.2514/1.A34584",
    ],
)
def test_doi_must_match_standard_shape(doi: str) -> None:
    with pytest.raises(ValidationError, match="doi"):
        Provenance(
            "title",
            "label",
            "interpretation",
            "https://example.com/source",
            doi=doi,
        )


@pytest.mark.parametrize(
    ("kind", "authority", "forbidden_claim"),
    [
        (
            EvidenceKind.ANALYTICAL,
            SourceAuthority.ANALYTICAL_REFERENCE,
            ClaimLevel.CROSS_MODEL_AGREEMENT,
        ),
        (
            EvidenceKind.MANUFACTURED,
            SourceAuthority.MANUFACTURED_REFERENCE,
            ClaimLevel.CROSS_MODEL_AGREEMENT,
        ),
        (
            EvidenceKind.CROSS_CODE,
            SourceAuthority.INDEPENDENT_CODE,
            ClaimLevel.VALIDATION_EVIDENCE,
        ),
        (
            EvidenceKind.STOCHASTIC_PIC,
            SourceAuthority.SIMULATION,
            ClaimLevel.PREDICTIVE_VALIDITY,
        ),
        (
            EvidenceKind.PUBLISHED_EXTERNAL,
            SourceAuthority.PUBLISHED_MODEL_OUTPUT,
            ClaimLevel.VALIDATION_EVIDENCE,
        ),
    ],
)
def test_every_nonexperiment_authority_enforces_claim_ceiling(
    kind: EvidenceKind,
    authority: SourceAuthority,
    forbidden_claim: ClaimLevel,
) -> None:
    kwargs = (
        {"seed": 1, "seed_policy_id": "seed-policy-v1"}
        if kind is EvidenceKind.STOCHASTIC_PIC
        else {}
    )
    with pytest.raises(ValidationError, match="cannot support claim"):
        EvidenceRecord(
            record_id=f"bad-{kind.value}",
            kind=kind,
            source_authority=authority,
            partition=(
                EvidencePartition.REFERENCE_ONLY
                if kind is EvidenceKind.PUBLISHED_EXTERNAL
                else EvidencePartition.VALIDATION
            ),
            observations=(MetricObservation("thrust", Quantity(0.1, "N")),),
            provenance=_provenance(),
            model_context_id="model",
            result_context_id="result",
            group_id="group",
            independence_identity=IndependenceIdentity("design", "run"),
            model_revision="model-v1",
            code_revision="code-v1",
            maximum_claim=forbidden_claim,
            **kwargs,
        )


def test_pic_cannot_claim_experimental_truth_by_record_choice() -> None:
    with pytest.raises(ValidationError, match="experimental truth"):
        EvidenceRecord(
            record_id="pic-as-truth",
            kind=EvidenceKind.STOCHASTIC_PIC,
            source_authority=SourceAuthority.SIMULATION,
            partition=EvidencePartition.VALIDATION,
            observations=(MetricObservation("thrust", Quantity(0.1, "N")),),
            provenance=_provenance("PIC", experimental=True),
            model_context_id="model",
            result_context_id="result",
            group_id="group",
            independence_identity=IndependenceIdentity("design", "run"),
            model_revision="pic-v1",
            code_revision="code-v1",
            seed=3,
            seed_policy_id="seed-policy-v1",
        )


def test_experiment_requires_metadata_uncertainty_and_true_provenance() -> None:
    assert _experiment().authority_claim_ceiling is ClaimLevel.PREDICTIVE_VALIDITY
    with pytest.raises(ValidationError, match="metadata"):
        EvidenceRecord(
            record_id="incomplete",
            kind=EvidenceKind.EXPERIMENT,
            source_authority=SourceAuthority.EXPERIMENT,
            partition=EvidencePartition.VALIDATION,
            observations=(
                MetricObservation("thrust", Quantity(0.1, "N"), _uncertainty()),
            ),
            provenance=_provenance(experimental=True),
            model_context_id="model",
            result_context_id="result",
            group_id="group",
            independence_identity=IndependenceIdentity(
                "design", "run", "article", "campaign", "specimen"
            ),
            model_revision="model-v1",
            code_revision="code-v1",
        )
    with pytest.raises(ValidationError, match="validation partition"):
        replaceable = _experiment("reference-experiment")
        EvidenceRecord(
            record_id="reference-experiment",
            kind=replaceable.kind,
            source_authority=replaceable.source_authority,
            partition=EvidencePartition.REFERENCE_ONLY,
            observations=replaceable.observations,
            provenance=replaceable.provenance,
            model_context_id=replaceable.model_context_id,
            result_context_id=replaceable.result_context_id,
            group_id=replaceable.group_id,
            independence_identity=replaceable.independence_identity,
            model_revision=replaceable.model_revision,
            code_revision=replaceable.code_revision,
            operating_parameters=replaceable.operating_parameters,
            credibility=CredibilityLevel.VALIDATION_SUPPORT,
            maximum_claim=ClaimLevel.VALIDATION_EVIDENCE,
            experiment_metadata=replaceable.experiment_metadata,
        )


def test_context_of_use_contract_requires_exact_si_dimensions() -> None:
    ledger = ContextOfUseLedger(
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
        validation_evidence_ids=("experiment-1", "experiment-2"),
    )
    assert ledger.contains({"power": Quantity(300.0, "W")})
    with pytest.raises(UnitError, match="must use W"):
        ledger.contains({"power": Quantity(0.3, "kW")})
    with pytest.raises(UnitError, match="explicit SI"):
        ApplicabilityRange("power", Quantity(0.1, "kW"), Quantity(0.5, "kW"))
