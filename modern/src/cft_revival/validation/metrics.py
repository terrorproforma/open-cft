"""Bound comparison, convergence, conservation, and replicate metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
import re
from statistics import fmean, stdev

from .contracts import (
    CONTRACT_VERSION,
    EvidenceKind,
    EvidenceRecord,
    ExperimentUncertainty,
    MetricObservation,
    Quantity,
    ReplicateEnsemble,
    UnitError,
    ValidationError,
)
from .evidence import evidence_hash


@dataclass(frozen=True, slots=True)
class ComparisonMetrics:
    reference_si: float
    candidate_si: float
    signed_error_si: float
    absolute_error_si: float
    relative_error: float | None
    error_interval_si: tuple[float, float]
    intervals_overlap: bool | None
    dimension: str


def compare_quantities(
    reference: Quantity,
    candidate: Quantity,
    *,
    reference_uncertainty: ExperimentUncertainty | None = None,
    candidate_uncertainty: ExperimentUncertainty | None = None,
) -> ComparisonMetrics:
    if not isinstance(reference, Quantity) or not isinstance(candidate, Quantity):
        raise ValidationError("quantity comparison requires typed Quantity records")
    if reference.dimension != candidate.dimension:
        raise UnitError(
            f"cannot compare {reference.dimension} with {candidate.dimension}"
        )
    reference_si = reference.si_value
    candidate_si = candidate.si_value
    error = candidate_si - reference_si
    relative = None if reference_si == 0.0 else abs(error) / abs(reference_si)
    ref_interval = _interval(reference, reference_uncertainty)
    candidate_interval = _interval(candidate, candidate_uncertainty)
    error_interval = (
        candidate_interval[0] - ref_interval[1],
        candidate_interval[1] - ref_interval[0],
    )
    has_uncertainty = (
        reference_uncertainty is not None or candidate_uncertainty is not None
    )
    overlap = (
        max(ref_interval[0], candidate_interval[0])
        <= min(ref_interval[1], candidate_interval[1])
        if has_uncertainty
        else None
    )
    return ComparisonMetrics(
        reference_si=reference_si,
        candidate_si=candidate_si,
        signed_error_si=error,
        absolute_error_si=abs(error),
        relative_error=relative,
        error_interval_si=error_interval,
        intervals_overlap=overlap,
        dimension=reference.dimension,
    )


def _interval(
    quantity: Quantity, uncertainty: ExperimentUncertainty | None
) -> tuple[float, float]:
    if uncertainty is None:
        return quantity.si_value, quantity.si_value
    return uncertainty.interval(quantity)


def compare_observations(
    reference: MetricObservation, candidate: MetricObservation
) -> ComparisonMetrics:
    if not isinstance(reference, MetricObservation) or not isinstance(
        candidate, MetricObservation
    ):
        raise ValidationError("observation comparison requires typed records")
    if reference.name != candidate.name:
        raise ValidationError("metric names must match before comparison")
    return compare_quantities(
        reference.quantity,
        candidate.quantity,
        reference_uncertainty=reference.uncertainty,
        candidate_uncertainty=candidate.uncertainty,
    )


@dataclass(frozen=True, slots=True)
class GateBinding:
    evidence_id: str
    evidence_sha256: str
    model_context_id: str
    result_context_id: str
    model_revision: str
    code_revision: str
    group_id: str
    mesh_id: str
    quantity_name: str
    unit: str

    def __post_init__(self) -> None:
        for field_name in (
            "evidence_id",
            "model_context_id",
            "result_context_id",
            "model_revision",
            "code_revision",
            "group_id",
            "mesh_id",
            "quantity_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{field_name} must be non-empty")
        if not isinstance(self.evidence_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.evidence_sha256
        ):
            raise ValidationError("evidence_sha256 must be 64 lowercase hex digits")
        Quantity(0.0, self.unit)

    @classmethod
    def from_record(cls, record: EvidenceRecord, quantity_name: str) -> GateBinding:
        matches = [item for item in record.observations if item.name == quantity_name]
        if len(matches) != 1:
            raise ValidationError(f"evidence lacks unique quantity {quantity_name!r}")
        if record.mesh_id is None:
            raise ValidationError("gate-bound evidence requires a mesh_id")
        return cls(
            evidence_id=record.record_id,
            evidence_sha256=evidence_hash(record),
            model_context_id=record.model_context_id,
            result_context_id=record.result_context_id,
            model_revision=record.model_revision,
            code_revision=record.code_revision,
            group_id=record.group_id,
            mesh_id=record.mesh_id,
            quantity_name=quantity_name,
            unit=matches[0].quantity.unit,
        )

    def observation_from(self, record: EvidenceRecord) -> MetricObservation:
        expected = GateBinding.from_record(record, self.quantity_name)
        if expected != self:
            raise ValidationError(
                "gate evidence does not match bound provenance/context/quantity/"
                "unit/model/code/mesh identity"
            )
        return next(item for item in record.observations if item.name == self.quantity_name)


@dataclass(frozen=True, slots=True)
class ConservationGate:
    name: str
    binding: GateBinding
    absolute_tolerance: Quantity
    relative_tolerance: float = 0.0
    scale: Quantity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("conservation gate name must be non-empty")
        if not isinstance(self.binding, GateBinding):
            raise ValidationError("conservation gate requires a typed binding")
        if not isinstance(self.absolute_tolerance, Quantity):
            raise ValidationError("absolute conservation tolerance must be a Quantity")
        if self.absolute_tolerance.unit != self.binding.unit:
            raise UnitError("conservation tolerance must use the bound quantity unit")
        if self.absolute_tolerance.value < 0.0:
            raise ValidationError("absolute conservation tolerance must be non-negative")
        if isinstance(self.relative_tolerance, bool):
            raise ValidationError("relative conservation tolerance cannot be boolean")
        relative = float(self.relative_tolerance)
        if not isfinite(relative) or relative < 0.0:
            raise ValidationError(
                "relative conservation tolerance must be finite and non-negative"
            )
        if relative > 0.0 and self.scale is None:
            raise ValidationError(
                "relative conservation tolerance requires an explicit valid scale"
            )
        if self.scale is not None:
            if not isinstance(self.scale, Quantity):
                raise ValidationError("conservation scale must be a Quantity")
            if self.scale.unit != self.binding.unit:
                raise UnitError("conservation scale must use the bound quantity unit")
            if self.scale.si_value <= 0.0:
                raise ValidationError("conservation scale must be positive")
        threshold = self.absolute_tolerance.si_value + relative * (
            self.scale.si_value if self.scale is not None else 0.0
        )
        if threshold <= 0.0:
            raise ValidationError("conservation gate threshold must be positive")
        object.__setattr__(self, "relative_tolerance", relative)

    def evaluate(self, record: EvidenceRecord) -> ConservationGateResult:
        if not isinstance(record, EvidenceRecord):
            raise ValidationError("conservation gate requires typed evidence")
        observation = self.binding.observation_from(record)
        scale_si = self.scale.si_value if self.scale is not None else 0.0
        threshold = self.absolute_tolerance.si_value + self.relative_tolerance * scale_si
        magnitude = abs(observation.quantity.si_value)
        normalized = (
            magnitude / threshold
            if threshold > 0.0
            else (0.0 if magnitude == 0.0 else float("inf"))
        )
        return ConservationGateResult(
            name=self.name,
            binding=self.binding,
            passed=magnitude <= threshold,
            residual_si=observation.quantity.si_value,
            threshold_si=threshold,
            normalized_residual=normalized,
        )


@dataclass(frozen=True, slots=True)
class ConservationGateResult:
    name: str
    binding: GateBinding
    passed: bool
    residual_si: float
    threshold_si: float
    normalized_residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("conservation result name must be non-empty")
        if not isinstance(self.binding, GateBinding):
            raise ValidationError("conservation result requires a typed binding")
        if not isinstance(self.passed, bool):
            raise ValidationError("conservation result passed flag must be boolean")
        for field_name in ("residual_si", "threshold_si", "normalized_residual"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isfinite(float(value)):
                raise ValidationError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class ConvergenceLevel:
    """One mesh level bound to raw error and spacing observations by record hash."""

    evidence: EvidenceRecord
    error_quantity_name: str
    spacing_quantity_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EvidenceRecord):
            raise ValidationError("convergence level requires typed evidence")
        if self.evidence.mesh_id is None:
            raise ValidationError("convergence level evidence requires mesh_id")
        for field_name in ("error_quantity_name", "spacing_quantity_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{field_name} must be non-empty")
        if self.error_quantity_name == self.spacing_quantity_name:
            raise ValidationError("error and spacing observations must be distinct")
        error = self._observation(self.error_quantity_name).quantity
        spacing = self._observation(self.spacing_quantity_name).quantity
        if spacing.dimension != "length":
            raise UnitError("characteristic spacing must have length units")
        if spacing.si_value <= 0.0:
            raise ValidationError("characteristic spacing must be positive")
        if error.si_value <= 0.0:
            raise ValidationError("convergence error must be positive")

    def _observation(self, name: str) -> MetricObservation:
        matches = [item for item in self.evidence.observations if item.name == name]
        if len(matches) != 1:
            raise ValidationError(f"convergence evidence lacks unique {name!r}")
        return matches[0]

    @property
    def error(self) -> Quantity:
        return self._observation(self.error_quantity_name).quantity

    @property
    def characteristic_spacing(self) -> Quantity:
        return self._observation(self.spacing_quantity_name).quantity

    @property
    def mesh_id(self) -> str:
        assert self.evidence.mesh_id is not None
        return self.evidence.mesh_id

    @property
    def evidence_sha256(self) -> str:
        return evidence_hash(self.evidence)


@dataclass(frozen=True, slots=True)
class ConvergenceStudy:
    study_id: str
    levels: tuple[ConvergenceLevel, ...]
    expected_minimum_order: float
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValidationError("study_id must be non-empty")
        if self.schema_version != CONTRACT_VERSION:
            raise ValidationError("unsupported convergence-study schema")
        levels = tuple(self.levels)
        if len(levels) < 3 or not all(
            isinstance(level, ConvergenceLevel) for level in levels
        ):
            raise ValidationError("convergence study requires at least three typed levels")
        if len({level.mesh_id for level in levels}) != len(levels):
            raise ValidationError("convergence mesh identities must be unique")
        if len({level.evidence.record_id for level in levels}) != len(levels):
            raise ValidationError("convergence evidence identities must be unique")
        if len({level.evidence_sha256 for level in levels}) != len(levels):
            raise ValidationError("convergence evidence content hashes must be unique")
        if len({level.characteristic_spacing.unit for level in levels}) != 1:
            raise UnitError("convergence spacings must use one consistent unit")
        if len({level.error.unit for level in levels}) != 1:
            raise UnitError("convergence errors must use one exact quantity unit")
        if len({level.error_quantity_name for level in levels}) != 1:
            raise ValidationError("convergence levels must use one error quantity")
        if len({level.spacing_quantity_name for level in levels}) != 1:
            raise ValidationError("convergence levels must use one spacing quantity")
        identities = {
            (
                level.evidence.kind,
                level.evidence.source_authority,
                level.evidence.model_context_id,
                level.evidence.result_context_id,
                level.evidence.model_revision,
                level.evidence.code_revision,
                level.evidence.provenance.source_locator,
                level.evidence.independence_identity.signature,
            )
            for level in levels
        }
        if len(identities) != 1:
            raise ValidationError(
                "convergence levels must share provenance, context, revisions, "
                "authority, and immutable run/design identity"
            )
        spacings = [level.characteristic_spacing.si_value for level in levels]
        if any(left <= right for left, right in zip(spacings, spacings[1:])):
            raise ValidationError("convergence levels must be ordered coarse-to-fine")
        if isinstance(self.expected_minimum_order, bool):
            raise ValidationError("expected minimum order cannot be boolean")
        minimum_order = float(self.expected_minimum_order)
        if not isfinite(minimum_order) or minimum_order <= 0.0:
            raise ValidationError("expected minimum order must be finite and positive")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "expected_minimum_order", minimum_order)

    @property
    def candidate_record(self) -> EvidenceRecord:
        return self.levels[-1].evidence

    @property
    def observed_orders(self) -> tuple[float, ...]:
        return tuple(
            log(coarse.error.si_value / fine.error.si_value)
            / log(
                coarse.characteristic_spacing.si_value
                / fine.characteristic_spacing.si_value
            )
            for coarse, fine in zip(self.levels, self.levels[1:])
        )

    @property
    def passed(self) -> bool:
        return all(order >= self.expected_minimum_order for order in self.observed_orders)


@dataclass(frozen=True, slots=True)
class ReplicateMetricSummary:
    metric_name: str
    unit: str
    count: int
    mean: float
    sample_standard_deviation: float
    standard_error: float
    student_t_degrees_of_freedom: int
    student_t_critical_95_percent: float
    confidence_interval_95_percent: tuple[float, float]
    interval_policy: str = "two-sided conservative tabulated Student-t, alpha=0.05"


_T_975: dict[int, float] = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    14: 2.144787,
    15: 2.131450,
    16: 2.119905,
    17: 2.109816,
    18: 2.100922,
    19: 2.093024,
    20: 2.085963,
    21: 2.079614,
    22: 2.073873,
    23: 2.068658,
    24: 2.063899,
    25: 2.059539,
    26: 2.055529,
    27: 2.051831,
    28: 2.048407,
    29: 2.045230,
    30: 2.042272,
}


def _student_t_critical_95(df: int) -> float:
    if df < 1:
        raise ValidationError("Student-t interval requires at least two samples")
    if df <= 30:
        return _T_975[df]
    if df <= 40:
        return _T_975[30]
    if df <= 60:
        return 2.021075
    if df <= 120:
        return 2.000298
    return 1.979930


def summarize_replicates(
    ensemble: ReplicateEnsemble, metric_name: str, *, unit: str
) -> ReplicateMetricSummary:
    """Use a two-sided Student-t interval; n<3 is rejected by the ensemble."""

    if not isinstance(ensemble, ReplicateEnsemble):
        raise ValidationError("replicate summary requires a typed ensemble")
    values: list[float] = []
    for record in ensemble.records:
        matches = [item for item in record.observations if item.name == metric_name]
        if len(matches) != 1:
            raise ValidationError(f"replicate lacks unique metric {metric_name!r}")
        if matches[0].quantity.unit != unit:
            raise UnitError("replicate summary unit must match the exact ensemble unit")
        values.append(matches[0].quantity.value)
    count = len(values)
    mean = fmean(values)
    sample_sd = stdev(values)
    standard_error = sample_sd / sqrt(count)
    degrees_of_freedom = count - 1
    critical = _student_t_critical_95(degrees_of_freedom)
    half_width = critical * standard_error
    return ReplicateMetricSummary(
        metric_name=metric_name,
        unit=unit,
        count=count,
        mean=mean,
        sample_standard_deviation=sample_sd,
        standard_error=standard_error,
        student_t_degrees_of_freedom=degrees_of_freedom,
        student_t_critical_95_percent=critical,
        confidence_interval_95_percent=(mean - half_width, mean + half_width),
    )


@dataclass(frozen=True, slots=True)
class VerificationCriterion:
    metric_name: str
    absolute_tolerance: Quantity
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValidationError("verification metric name must be non-empty")
        if not isinstance(self.absolute_tolerance, Quantity):
            raise ValidationError("verification absolute tolerance must be a Quantity")
        if self.absolute_tolerance.value < 0.0:
            raise ValidationError("verification absolute tolerance must be non-negative")
        if isinstance(self.relative_tolerance, bool):
            raise ValidationError("verification relative tolerance cannot be boolean")
        relative = float(self.relative_tolerance)
        if not isfinite(relative) or relative < 0.0:
            raise ValidationError("verification relative tolerance must be non-negative")
        object.__setattr__(self, "relative_tolerance", relative)


@dataclass(frozen=True, slots=True)
class VerificationMetricResult:
    metric_name: str
    comparison: ComparisonMetrics
    acceptance_threshold_si: float
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValidationError("verification metric result name must be non-empty")
        if not isinstance(self.comparison, ComparisonMetrics):
            raise ValidationError("verification metric result requires typed comparison")
        if isinstance(self.acceptance_threshold_si, bool) or not isinstance(
            self.acceptance_threshold_si, (int, float)
        ):
            raise ValidationError("verification acceptance threshold must be numeric")
        threshold = float(self.acceptance_threshold_si)
        if not isfinite(threshold):
            raise ValidationError("verification acceptance threshold must be finite")
        if threshold < 0.0:
            raise ValidationError("verification acceptance threshold must be non-negative")
        object.__setattr__(self, "acceptance_threshold_si", threshold)
        if not isinstance(self.passed, bool):
            raise ValidationError("verification metric passed flag must be boolean")


@dataclass(frozen=True, slots=True)
class VerificationAssessment:
    """Raw bound inputs; all comparison and gate results are recomputed on access."""

    case_id: str
    kind: EvidenceKind
    reference: EvidenceRecord
    candidate: EvidenceRecord
    criteria: tuple[VerificationCriterion, ...]
    conservation_gates: tuple[ConservationGate, ...] = ()
    convergence_study: ConvergenceStudy | None = None
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValidationError("case_id must be non-empty")
        if self.kind not in (EvidenceKind.ANALYTICAL, EvidenceKind.MANUFACTURED):
            raise ValidationError(
                "verification assessment requires analytical or manufactured kind"
            )
        if self.schema_version != CONTRACT_VERSION:
            raise ValidationError("unsupported verification-assessment schema")
        if not isinstance(self.reference, EvidenceRecord) or not isinstance(
            self.candidate, EvidenceRecord
        ):
            raise ValidationError("verification assessment requires typed evidence")
        if self.reference.kind is not self.kind or self.candidate.kind is not self.kind:
            raise ValidationError("verification evidence does not match assessment kind")
        if (
            self.reference.model_context_id != self.candidate.model_context_id
            or self.reference.result_context_id != self.candidate.result_context_id
        ):
            raise ValidationError("verification evidence contexts differ")
        criteria = tuple(self.criteria)
        if not criteria or not all(
            isinstance(item, VerificationCriterion) for item in criteria
        ):
            raise ValidationError("verification assessment requires typed criteria")
        if len({item.metric_name for item in criteria}) != len(criteria):
            raise ValidationError("verification criteria names must be unique")
        gates = tuple(self.conservation_gates)
        if not all(isinstance(item, ConservationGate) for item in gates):
            raise ValidationError("verification assessment requires typed raw gates")
        if any(
            gate.binding.evidence_id != self.candidate.record_id
            or gate.binding.evidence_sha256 != evidence_hash(self.candidate)
            for gate in gates
        ):
            raise ValidationError("conservation gate is unrelated to candidate evidence")
        if self.convergence_study is not None and not isinstance(
            self.convergence_study, ConvergenceStudy
        ):
            raise ValidationError("verification assessment requires a typed study")
        if (
            self.convergence_study is not None
            and evidence_hash(self.convergence_study.candidate_record)
            != evidence_hash(self.candidate)
        ):
            raise ValidationError(
                "finest convergence evidence does not match candidate content hash"
            )
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "conservation_gates", gates)

    @property
    def reference_id(self) -> str:
        return self.reference.record_id

    @property
    def candidate_id(self) -> str:
        return self.candidate.record_id

    @property
    def metric_results(self) -> tuple[VerificationMetricResult, ...]:
        reference_metrics = {item.name: item for item in self.reference.observations}
        candidate_metrics = {item.name: item for item in self.candidate.observations}
        results: list[VerificationMetricResult] = []
        for criterion in self.criteria:
            if (
                criterion.metric_name not in reference_metrics
                or criterion.metric_name not in candidate_metrics
            ):
                raise ValidationError(
                    f"verification evidence lacks metric {criterion.metric_name!r}"
                )
            comparison = compare_observations(
                reference_metrics[criterion.metric_name],
                candidate_metrics[criterion.metric_name],
            )
            if criterion.absolute_tolerance.dimension != comparison.dimension:
                raise UnitError(
                    "verification tolerance dimension does not match its metric"
                )
            threshold = (
                criterion.absolute_tolerance.si_value
                + criterion.relative_tolerance * abs(comparison.reference_si)
            )
            results.append(
                VerificationMetricResult(
                    criterion.metric_name,
                    comparison,
                    threshold,
                    comparison.absolute_error_si <= threshold,
                )
            )
        return tuple(results)

    @property
    def conservation_results(self) -> tuple[ConservationGateResult, ...]:
        return tuple(gate.evaluate(self.candidate) for gate in self.conservation_gates)

    @property
    def passed(self) -> bool:
        return (
            bool(self.metric_results)
            and all(item.passed for item in self.metric_results)
            and all(item.passed for item in self.conservation_results)
            and (self.convergence_study is None or self.convergence_study.passed)
        )


def assess_verification(
    case_id: str,
    kind: EvidenceKind,
    reference: EvidenceRecord,
    candidate: EvidenceRecord,
    criteria: tuple[VerificationCriterion, ...],
    *,
    conservation_gates: tuple[ConservationGate, ...] = (),
    convergence_study: ConvergenceStudy | None = None,
) -> VerificationAssessment:
    return VerificationAssessment(
        case_id,
        kind,
        reference,
        candidate,
        tuple(criteria),
        tuple(conservation_gates),
        convergence_study,
    )
