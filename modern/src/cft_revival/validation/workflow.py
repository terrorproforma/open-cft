"""Bound partition, context-of-use, comparison, and report workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import (
    ClaimLevel,
    ContextOfUseLedger,
    CredibilityLevel,
    EvidenceKind,
    EvidencePartition,
    EvidenceRecord,
    SourceAuthority,
    ValidationError,
)
from .evidence import (
    INTEGRITY_NOTICE,
    EvidenceRegistry,
    canonical_json,
    evidence_hash,
)
from .metrics import ComparisonMetrics, VerificationAssessment, compare_observations


class AuditStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True, slots=True)
class PartitionAudit:
    registry_identity_sha256: str
    calibration_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    reference_only_ids: tuple[str, ...]
    leaking_identities: tuple[str, ...]
    status: AuditStatus
    reason: str


def audit_grouped_partitions(registry: EvidenceRegistry) -> PartitionAudit:
    """Audit the exact content-addressed registry, including sufficiency."""

    if not isinstance(registry, EvidenceRegistry):
        raise ValidationError("partition audit requires an EvidenceRegistry")
    items = registry.records
    calibration = tuple(
        sorted(
            item.record_id
            for item in items
            if item.partition is EvidencePartition.CALIBRATION
        )
    )
    validation = tuple(
        sorted(
            item.record_id
            for item in items
            if item.partition is EvidencePartition.VALIDATION
        )
    )
    reference = tuple(
        sorted(
            item.record_id
            for item in items
            if item.partition is EvidencePartition.REFERENCE_ONLY
        )
    )
    calibration_records = [
        item for item in items if item.partition is EvidencePartition.CALIBRATION
    ]
    validation_records = [
        item for item in items if item.partition is EvidencePartition.VALIDATION
    ]
    leaking = tuple(
        sorted(
            {
                f"{calibration_record.record_id}<->{validation_record.record_id}:{token}"
                for calibration_record in calibration_records
                for validation_record in validation_records
                for token in calibration_record.independence_identity.overlaps(
                    validation_record.independence_identity
                )
            }
        )
    )
    if not items:
        status = AuditStatus.NOT_EVALUATED
        reason = "registry is empty; no partition evidence was evaluated"
    elif leaking:
        status = AuditStatus.FAIL
        reason = (
            "immutable identities cross calibration and validation: "
            f"{list(leaking)}"
        )
    elif not calibration or not validation:
        status = AuditStatus.NOT_EVALUATED
        reason = "both calibration and validation partitions are required"
    else:
        status = AuditStatus.PASS
        reason = "calibration and validation groups are disjoint"
    return PartitionAudit(
        registry_identity_sha256=registry.identity_sha256,
        calibration_ids=calibration,
        validation_ids=validation,
        reference_only_ids=reference,
        leaking_identities=leaking,
        status=status,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class PairwiseEvidenceComparison:
    reference_id: str
    candidate_id: str
    metrics: tuple[tuple[str, ComparisonMetrics], ...]
    maximum_claim: ClaimLevel


def compare_evidence_records(
    reference: EvidenceRecord, candidate: EvidenceRecord
) -> PairwiseEvidenceComparison:
    if not isinstance(reference, EvidenceRecord) or not isinstance(
        candidate, EvidenceRecord
    ):
        raise ValidationError("evidence comparison requires typed records")
    if (
        reference.model_context_id != candidate.model_context_id
        or reference.result_context_id != candidate.result_context_id
    ):
        raise ValidationError(
            "cross-context evidence comparison is not meaningful"
        )
    reference_metrics = {item.name: item for item in reference.observations}
    candidate_metrics = {item.name: item for item in candidate.observations}
    if reference_metrics.keys() != candidate_metrics.keys():
        raise ValidationError("evidence metric schemas differ")
    comparisons = tuple(
        (
            name,
            compare_observations(reference_metrics[name], candidate_metrics[name]),
        )
        for name in sorted(reference_metrics)
    )
    maximum = min(
        reference.maximum_claim,
        candidate.maximum_claim,
        reference.authority_claim_ceiling,
        candidate.authority_claim_ceiling,
    )
    return PairwiseEvidenceComparison(
        reference.record_id,
        candidate.record_id,
        comparisons,
        ClaimLevel(maximum),
    )


@dataclass(frozen=True, slots=True)
class ContextOfUseAudit:
    ledger_id: str
    registry_identity_sha256: str
    matching_evidence_ids: tuple[str, ...]
    missing_dimensions: tuple[str, ...]
    status: AuditStatus

    @property
    def passed(self) -> bool:
        return self.status is AuditStatus.PASS


def _independence_component_count(records: list[EvidenceRecord]) -> int:
    """Count groups connected by any immutable design/run/physical identifier."""

    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left].independence_identity.overlaps(
                records[right].independence_identity
            ):
                left_root = root(left)
                right_root = root(right)
                if left_root != right_root:
                    parents[right_root] = left_root
    return len({root(index) for index in range(len(records))})


def audit_context_of_use(
    ledger: ContextOfUseLedger, registry: EvidenceRegistry
) -> ContextOfUseAudit:
    """Match exact experiment evidence to every context-of-use dimension."""

    if not isinstance(ledger, ContextOfUseLedger):
        raise ValidationError("context audit requires a ContextOfUseLedger")
    if not isinstance(registry, EvidenceRegistry):
        raise ValidationError("context audit requires an EvidenceRegistry")
    missing: list[str] = []
    selected: list[EvidenceRecord] = []
    for record_id in ledger.validation_evidence_ids:
        try:
            record = registry.get(record_id)
        except ValidationError:
            missing.append(f"evidence_id:{record_id}")
            continue
        selected.append(record)

    eligible: list[EvidenceRecord] = []
    calibration_records = [
        item
        for item in registry.records
        if item.partition is EvidencePartition.CALIBRATION
    ]
    for record in selected:
        prefix = f"evidence:{record.record_id}"
        if (
            record.kind is not EvidenceKind.EXPERIMENT
            or record.source_authority is not SourceAuthority.EXPERIMENT
        ):
            missing.append(f"{prefix}:experimental_authority")
            continue
        if record.partition is not EvidencePartition.VALIDATION:
            missing.append(f"{prefix}:validation_partition")
            continue
        if record.credibility < CredibilityLevel.VALIDATION_SUPPORT:
            missing.append(f"{prefix}:validation_credibility")
            continue
        if record.maximum_claim < ClaimLevel.VALIDATION_EVIDENCE:
            missing.append(f"{prefix}:validation_claim")
            continue
        exact_fields = (
            ("model_context", record.model_context_id, ledger.required_model_context_id),
            ("result_context", record.result_context_id, ledger.required_result_context_id),
            ("model_revision", record.model_revision, ledger.required_model_revision),
            ("code_revision", record.code_revision, ledger.required_code_revision),
        )
        mismatched = False
        for name, actual, expected in exact_fields:
            if actual != expected:
                missing.append(f"{prefix}:{name}={expected}")
                mismatched = True
        parameters = {item.name: item.quantity for item in record.operating_parameters}
        expected_parameters = {item.parameter for item in ledger.applicability_ranges}
        if set(parameters) != expected_parameters:
            missing.append(f"{prefix}:applicability_parameters")
            mismatched = True
        else:
            for item in ledger.applicability_ranges:
                value = parameters[item.parameter]
                if value.unit != item.minimum.unit:
                    missing.append(
                        f"{prefix}:applicability_unit:{item.parameter}={item.minimum.unit}"
                    )
                    mismatched = True
                elif not item.minimum.si_value <= value.si_value <= item.maximum.si_value:
                    missing.append(f"{prefix}:applicability_range:{item.parameter}")
                    mismatched = True
        overlaps = sorted(
            {
                token
                for calibration in calibration_records
                for token in record.independence_identity.overlaps(
                    calibration.independence_identity
                )
            }
        )
        if overlaps:
            missing.append(
                f"{prefix}:partition_independence={','.join(overlaps)}"
            )
            mismatched = True
        if not mismatched:
            eligible.append(record)

    matching_ids: set[str] = set()
    for requirement in ledger.quantity_requirements:
        named = [
            record
            for record in eligible
            if any(
                observation.name == requirement.name
                for observation in record.observations
            )
        ]
        matching = [
            record
            for record in named
            if any(
                observation.name == requirement.name
                and observation.quantity.unit == requirement.si_unit
                for observation in record.observations
            )
        ]
        independent_groups = _independence_component_count(matching)
        matching_ids.update(record.record_id for record in matching)
        if len(matching) < requirement.minimum_evidence:
            missing.append(
                f"quantity:{requirement.name}:evidence_count="
                f"{len(matching)}/{requirement.minimum_evidence}"
            )
        if independent_groups < requirement.minimum_independent_groups:
            missing.append(
                f"quantity:{requirement.name}:independent_groups="
                f"{independent_groups}/{requirement.minimum_independent_groups}"
            )
        if any(
            any(
                observation.name == requirement.name
                and observation.quantity.unit != requirement.si_unit
                for observation in record.observations
            )
            for record in named
        ):
            missing.append(f"quantity:{requirement.name}:si_unit={requirement.si_unit}")

    status = AuditStatus.PASS if not missing else AuditStatus.FAIL
    return ContextOfUseAudit(
        ledger_id=ledger.ledger_id,
        registry_identity_sha256=registry.identity_sha256,
        matching_evidence_ids=tuple(sorted(matching_ids)),
        missing_dimensions=tuple(sorted(set(missing))),
        status=status,
    )


def render_evidence_report(title: str, registry: EvidenceRegistry) -> str:
    """Recompute audits from the exact registry; stale audits cannot be injected."""

    if not isinstance(title, str) or not title.strip():
        raise ValidationError("report title must be non-empty")
    if not isinstance(registry, EvidenceRegistry):
        raise ValidationError("report requires an EvidenceRegistry")
    audit = audit_grouped_partitions(registry)
    records = registry.records
    lines = [
        f"# {title.strip()}",
        "",
        f"Contract version: `{records[0].schema_version if records else '2.0.0'}`",
        f"Registry SHA-256: `{registry.identity_sha256}`",
        f"Evidence records: {len(records)}",
        f"Grouped split audit: {audit.status.value}",
        f"Grouped split reason: {audit.reason}",
        "",
        "This report is evidence bookkeeping, not NASA-STD-7009B certification.",
        "Published model outputs are cross-model references, not experimental truth.",
        INTEGRITY_NOTICE,
        "",
        "## Evidence inventory",
        "",
    ]
    if not records:
        lines.append(
            "No evidence records; evidence assessment status is NOT_EVALUATED."
        )
    for record in records:
        lines.extend(
            [
                f"### {record.record_id}",
                f"- Source-native label: `{record.provenance.source_native_label}`",
                f"- Editorial interpretation: {record.provenance.editorial_interpretation}",
                f"- Kind: `{record.kind.value}`",
                f"- Source authority: `{record.source_authority.value}`",
                f"- Partition: `{record.partition.value}`",
                f"- Credibility: `{record.credibility.name.lower()}`",
                f"- Maximum claim: `{record.maximum_claim.name.lower()}`",
                f"- SHA-256: `{evidence_hash(record)}`",
            ]
        )
        for observation in record.observations:
            lines.append(
                f"- {observation.name}: {observation.quantity.value:g} "
                f"{observation.quantity.unit}"
            )
        lines.append("")
    report_body = "\n".join(lines).rstrip() + "\n"
    manifest = {
        "partition_audit": {
            "registry_identity_sha256": audit.registry_identity_sha256,
            "status": audit.status.value,
            "reason": audit.reason,
        },
        "record_hashes": {
            record.record_id: evidence_hash(record) for record in records
        },
        "report_body": report_body,
    }
    return report_body + f"\nReport payload: `{canonical_json(manifest)}`\n"


def render_verification_report(
    title: str, assessment: VerificationAssessment
) -> str:
    """Render only values recomputed from bound raw evidence and gate policies."""

    if not isinstance(title, str) or not title.strip():
        raise ValidationError("verification report title must be non-empty")
    if not isinstance(assessment, VerificationAssessment):
        raise ValidationError("verification report requires a typed assessment")
    metrics = assessment.metric_results
    conservation = assessment.conservation_results
    convergence = assessment.convergence_study
    status = "PASS" if assessment.passed else "FAIL"
    lines = [
        f"# {title.strip()}",
        "",
        f"Assessment status: {status}",
        f"Reference SHA-256: `{evidence_hash(assessment.reference)}`",
        f"Candidate SHA-256: `{evidence_hash(assessment.candidate)}`",
        "",
        "## Recomputed metric checks",
    ]
    for result in metrics:
        lines.append(
            f"- {result.metric_name}: absolute_error_si="
            f"{result.comparison.absolute_error_si:.17g}, "
            f"threshold_si={result.acceptance_threshold_si:.17g}, "
            f"status={'PASS' if result.passed else 'FAIL'}"
        )
    lines.append("")
    lines.append("## Recomputed conservation checks")
    if not conservation:
        lines.append("- NOT_EVALUATED: no conservation gates supplied")
    for result in conservation:
        lines.append(
            f"- {result.name}: residual_si={result.residual_si:.17g}, "
            f"threshold_si={result.threshold_si:.17g}, "
            f"status={'PASS' if result.passed else 'FAIL'}"
        )
    lines.append("")
    lines.append("## Recomputed convergence")
    if convergence is None:
        lines.append("- NOT_EVALUATED: no convergence study supplied")
    else:
        lines.append(
            "- level_hashes="
            + ",".join(level.evidence_sha256 for level in convergence.levels)
        )
        lines.append(
            "- observed_orders="
            + ",".join(f"{order:.17g}" for order in convergence.observed_orders)
        )
        lines.append(f"- status={'PASS' if convergence.passed else 'FAIL'}")
    return "\n".join(lines).rstrip() + "\n"
