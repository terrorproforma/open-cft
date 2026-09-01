from dataclasses import replace

import pytest

from cft_revival.validation import (
    ClaimLevel,
    ConservationGate,
    ConvergenceLevel,
    ConvergenceStudy,
    CredibilityLevel,
    EvidenceKind,
    EvidencePartition,
    EvidenceRecord,
    GateBinding,
    IndependenceIdentity,
    MetricObservation,
    Provenance,
    Quantity,
    ReplicateEnsemble,
    SourceAuthority,
    UnitError,
    ValidationError,
    VerificationCriterion,
    assess_verification,
    render_verification_report,
    summarize_replicates,
)


def _provenance(locator: str = "artifact://validation/run") -> Provenance:
    return Provenance(
        "Validation fixture",
        "native fixture",
        "test interpretation",
        locator,
    )


def _pic(seed: int, thrust: float = 60.0) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=f"pic-{seed}",
        kind=EvidenceKind.STOCHASTIC_PIC,
        source_authority=SourceAuthority.SIMULATION,
        partition=EvidencePartition.VALIDATION,
        observations=(MetricObservation("thrust", Quantity(thrust, "mN")),),
        provenance=_provenance(),
        model_context_id="pic-model-context",
        result_context_id="pic-result-context",
        group_id="caller-group",
        independence_identity=IndependenceIdentity("design-1", "pic-family-v1"),
        model_revision="pic-v2",
        code_revision="code-v3",
        credibility=CredibilityLevel.CORROBORATED,
        maximum_claim=ClaimLevel.CROSS_MODEL_AGREEMENT,
        seed=seed,
        seed_policy_id="independent-seeds-v1",
        mesh_id="mesh-v5",
    )


def _verification_record(
    record_id: str,
    mesh_id: str,
    *,
    error: float,
    spacing_m: float,
    residual: float = 1.0e-12,
    code_revision: str = "code-v1",
) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        kind=EvidenceKind.MANUFACTURED,
        source_authority=SourceAuthority.MANUFACTURED_REFERENCE,
        partition=EvidencePartition.VALIDATION,
        observations=(
            MetricObservation("l2_error", Quantity(error, "1")),
            MetricObservation("mesh_spacing", Quantity(spacing_m, "m")),
            MetricObservation("mass_residual", Quantity(residual, "kg/s")),
        ),
        provenance=_provenance("artifact://validation/manufactured"),
        model_context_id="equations-v1",
        result_context_id="norms-v1",
        group_id="caller-mms",
        independence_identity=IndependenceIdentity("mms-design", "mms-run-v1"),
        model_revision="model-v1",
        code_revision=code_revision,
        credibility=CredibilityLevel.CHECKED,
        maximum_claim=ClaimLevel.VERIFIED_IMPLEMENTATION,
        mesh_id=mesh_id,
    )


def _study() -> ConvergenceStudy:
    return ConvergenceStudy(
        "mms-convergence",
        (
            ConvergenceLevel(
                _verification_record("coarse", "mesh-1", error=0.04, spacing_m=0.2),
                "l2_error",
                "mesh_spacing",
            ),
            ConvergenceLevel(
                _verification_record("medium", "mesh-2", error=0.01, spacing_m=0.1),
                "l2_error",
                "mesh_spacing",
            ),
            ConvergenceLevel(
                _verification_record(
                    "candidate", "mesh-fine", error=0.0025, spacing_m=0.05
                ),
                "l2_error",
                "mesh_spacing",
            ),
        ),
        expected_minimum_order=1.9,
    )


def test_pic_ensemble_rejects_n2_and_heterogeneous_content() -> None:
    with pytest.raises(ValidationError, match="insufficient"):
        ReplicateEnsemble((_pic(1), _pic(2)))
    records = (_pic(1), _pic(2), _pic(3))
    with pytest.raises(ValidationError, match="share authority"):
        ReplicateEnsemble((records[0], records[1], replace(records[2], code_revision="other")))
    with pytest.raises(ValidationError, match="exact quantity"):
        ReplicateEnsemble(
            (
                records[0],
                records[1],
                replace(
                    records[2],
                    observations=(MetricObservation("thrust", Quantity(0.06, "N")),),
                ),
            )
        )


def test_pic_student_t_interval_for_n3_and_large_n() -> None:
    small = summarize_replicates(
        ReplicateEnsemble((_pic(1, 60.0), _pic(2, 62.0), _pic(3, 64.0))),
        "thrust",
        unit="mN",
    )
    assert small.student_t_critical_95_percent == pytest.approx(4.302653)
    expected_half_width = 4.302653 * 2.0 / 3.0**0.5
    assert small.confidence_interval_95_percent == pytest.approx(
        (62.0 - expected_half_width, 62.0 + expected_half_width)
    )
    large = summarize_replicates(
        ReplicateEnsemble(tuple(_pic(seed, 60.0 + seed % 3) for seed in range(31))),
        "thrust",
        unit="mN",
    )
    assert large.student_t_degrees_of_freedom == 30
    assert large.student_t_critical_95_percent == pytest.approx(2.042272)


def test_assessment_recomputes_raw_conservation_and_cannot_forge_pass() -> None:
    candidate = _verification_record(
        "candidate", "mesh-fine", error=0.0025, spacing_m=0.05, residual=999.0
    )
    reference = _verification_record(
        "reference", "reference-mesh", error=0.0025, spacing_m=0.05
    )
    gate = ConservationGate(
        "mass",
        GateBinding.from_record(candidate, "mass_residual"),
        Quantity(1.0, "kg/s"),
    )
    assessment = assess_verification(
        "forgery-regression",
        EvidenceKind.MANUFACTURED,
        reference,
        candidate,
        (VerificationCriterion("l2_error", Quantity(0.001, "1")),),
        conservation_gates=(gate,),
    )
    result = assessment.conservation_results[0]
    assert result.residual_si == 999.0
    assert result.threshold_si == 1.0
    assert not result.passed
    assert not assessment.passed
    report = render_verification_report("Forgery regression", assessment)
    assert "residual_si=999" in report
    assert "threshold_si=1" in report
    assert "status=FAIL" in report
    with pytest.raises(TypeError):
        assess_verification(
            "old-forged-api",
            EvidenceKind.MANUFACTURED,
            reference,
            candidate,
            (VerificationCriterion("l2_error", Quantity(0.001, "1")),),
            conservation_results=("forged-pass",),  # type: ignore[call-arg]
        )


def test_relative_conservation_requires_explicit_bound_scale() -> None:
    candidate = _study().candidate_record
    with pytest.raises(ValidationError, match="explicit valid scale"):
        ConservationGate(
            "mass",
            GateBinding.from_record(candidate, "mass_residual"),
            Quantity(1.0e-13, "kg/s"),
            relative_tolerance=1.0e-6,
        )


def test_convergence_recomputes_each_bound_level_and_order() -> None:
    study = _study()
    assert study.observed_orders == pytest.approx((2.0, 2.0))
    candidate = study.candidate_record
    reference = _verification_record(
        "reference", "reference-mesh", error=0.0025, spacing_m=0.05
    )
    assessment = assess_verification(
        "mms-case",
        EvidenceKind.MANUFACTURED,
        reference,
        candidate,
        (VerificationCriterion("l2_error", Quantity(0.001, "1")),),
        convergence_study=study,
    )
    assert assessment.passed
    report = render_verification_report("MMS", assessment)
    for level in study.levels:
        assert level.evidence_sha256 in report


def test_changed_finest_content_fails_even_with_same_id_and_mesh() -> None:
    study = _study()
    original = study.candidate_record
    changed = replace(
        original,
        observations=(
            MetricObservation("l2_error", Quantity(0.9, "1")),
            MetricObservation("mesh_spacing", Quantity(0.05, "m")),
            MetricObservation("mass_residual", Quantity(1.0e-12, "kg/s")),
        ),
    )
    reference = _verification_record(
        "reference", "reference-mesh", error=0.0025, spacing_m=0.05
    )
    with pytest.raises(ValidationError, match="candidate content hash"):
        assess_verification(
            "changed-finest",
            EvidenceKind.MANUFACTURED,
            reference,
            changed,
            (VerificationCriterion("l2_error", Quantity(1.0, "1")),),
            convergence_study=study,
        )


def test_convergence_rejects_unbound_level_revision_and_spacing_units() -> None:
    levels = list(_study().levels)
    levels[1] = ConvergenceLevel(
        replace(levels[1].evidence, code_revision="other"),
        "l2_error",
        "mesh_spacing",
    )
    with pytest.raises(ValidationError, match="share provenance"):
        ConvergenceStudy("bad-revision", tuple(levels), 1.9)
    levels = list(_study().levels)
    medium = levels[1].evidence
    levels[1] = ConvergenceLevel(
        replace(
            medium,
            observations=(
                MetricObservation("l2_error", Quantity(0.01, "1")),
                MetricObservation("mesh_spacing", Quantity(100.0, "mm")),
                MetricObservation("mass_residual", Quantity(1.0e-12, "kg/s")),
            ),
        ),
        "l2_error",
        "mesh_spacing",
    )
    with pytest.raises(UnitError, match="consistent unit"):
        ConvergenceStudy("bad-spacing-unit", tuple(levels), 1.9)
