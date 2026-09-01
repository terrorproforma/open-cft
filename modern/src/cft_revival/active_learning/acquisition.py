"""Cost-aware candidate and fidelity selection with explicit quota handling."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, hypot, log, log1p, sqrt
from statistics import fmean
from typing import Sequence

from .contracts import (
    ActiveLearningError,
    CampaignCounts,
    FidelitySource,
    PosteriorAdapter,
    PosteriorPrediction,
    finite,
    integer,
)
from .robustness import gaussian_feasibility_probability


@dataclass(frozen=True)
class AcquisitionWeights:
    predicted_improvement: float = 1.0
    feasibility: float = 1.0
    discrepancy: float = 0.35
    uncertainty: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "predicted_improvement",
            "feasibility",
            "discrepancy",
            "uncertainty",
        ):
            value = finite(f"{name} weight", getattr(self, name))
            if value < 0.0:
                raise ActiveLearningError("acquisition weights cannot be negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PendingFantasyApproximation:
    """Deterministic mean-fantasy approximation, not exact joint fantasization."""

    fantasy_incumbent: tuple[float, ...]
    spatial_penalty: float
    label: str = "asynchronous-posterior-mean-fantasy-approximation"

    def __post_init__(self) -> None:
        try:
            incumbent = tuple(
                finite("fantasy incumbent", value) for value in self.fantasy_incumbent
            )
        except TypeError as error:
            raise ActiveLearningError("fantasy incumbent must be iterable") from error
        penalty = finite("pending spatial penalty", self.spatial_penalty)
        if not incumbent or not 0.0 <= penalty <= 1.0:
            raise ActiveLearningError("pending fantasy values are invalid")
        if not isinstance(self.label, str) or not self.label:
            raise ActiveLearningError("pending fantasy method label is required")
        object.__setattr__(self, "fantasy_incumbent", incumbent)
        object.__setattr__(self, "spatial_penalty", penalty)


@dataclass(frozen=True)
class CandidateScore:
    source: FidelitySource
    predicted_improvement: float
    feasibility_probability: float
    discrepancy_signal: float
    uncertainty_signal: float
    pending_penalty: float
    raw_information_value: float
    cost_normalized_score: float
    pending_method: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, FidelitySource):
            raise ActiveLearningError("candidate score requires a fidelity source")
        for name in (
            "predicted_improvement",
            "feasibility_probability",
            "discrepancy_signal",
            "uncertainty_signal",
            "pending_penalty",
            "raw_information_value",
            "cost_normalized_score",
        ):
            value = finite(f"candidate {name}", getattr(self, name))
            if value < 0.0:
                raise ActiveLearningError(f"candidate {name} cannot be negative")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.feasibility_probability <= 1.0:
            raise ActiveLearningError("candidate feasibility probability must lie in [0, 1]")
        if not 0.0 <= self.pending_penalty <= 1.0:
            raise ActiveLearningError("candidate pending penalty must lie in [0, 1]")
        if not isinstance(self.pending_method, str) or not self.pending_method:
            raise ActiveLearningError("candidate pending method is required")


def _corrected_means(prediction: PosteriorPrediction) -> tuple[float, ...]:
    return tuple(
        finite("bias-corrected objective mean", mean + discrepancy)
        for mean, discrepancy in zip(
            prediction.objective_means,
            prediction.discrepancy_means,
            strict=True,
        )
    )


def approximate_pending_fantasization(
    design: Sequence[float],
    incumbent: Sequence[float],
    pending: Sequence[tuple[Sequence[float], PosteriorPrediction]],
    *,
    length_scale: float = 0.15,
) -> PendingFantasyApproximation:
    """Use pending posterior means as fantasies and repel nearby duplicates.

    This intentionally avoids claiming exact conditional GP fantasies. It is a
    dependency-light deterministic scheduling approximation.
    """

    length_scale = finite("pending length scale", length_scale)
    if length_scale <= 0.0:
        raise ActiveLearningError("pending length scale must be positive")
    try:
        fantasy_incumbent = tuple(finite("incumbent", value) for value in incumbent)
        candidate_design = tuple(
            finite("candidate coordinate", value) for value in design
        )
        pending_records = tuple(pending)
    except (TypeError, ValueError, AttributeError) as error:
        if isinstance(error, ActiveLearningError):
            raise
        raise ActiveLearningError("malformed pending-fantasy inputs") from error
    if not fantasy_incumbent:
        raise ActiveLearningError("incumbent must contain objective values")
    penalty = 1.0
    for record in pending_records:
        try:
            pending_design, prediction = record
            pending_coordinates = tuple(
                finite("pending coordinate", value) for value in pending_design
            )
        except (TypeError, ValueError, AttributeError) as error:
            if isinstance(error, ActiveLearningError):
                raise
            raise ActiveLearningError("malformed pending prediction record") from error
        if not isinstance(prediction, PosteriorPrediction):
            raise ActiveLearningError("pending prediction must be PosteriorPrediction")
        if len(pending_coordinates) != len(candidate_design):
            raise ActiveLearningError("pending and candidate design dimensions differ")
        pending_mean = _corrected_means(prediction)
        if len(pending_mean) != len(fantasy_incumbent):
            raise ActiveLearningError("pending and incumbent objective dimensions differ")
        fantasy_incumbent = tuple(
            max(current, fantasy)
            for current, fantasy in zip(fantasy_incumbent, pending_mean, strict=True)
        )
        squared_distance = fsum(
            (
                left / max(1.0, abs(left), abs(right))
                - right / max(1.0, abs(left), abs(right))
            )
            ** 2
            for left, right in zip(candidate_design, pending_coordinates, strict=True)
        )
        distance = sqrt(squared_distance)
        scaled_distance = min(distance / length_scale, 40.0)
        penalty *= 1.0 - exp(-0.5 * scaled_distance * scaled_distance)
        penalty = finite("pending spatial penalty", penalty)
    return PendingFantasyApproximation(fantasy_incumbent, penalty)


def score_candidate(
    design: Sequence[float],
    source: FidelitySource,
    posterior: PosteriorAdapter,
    incumbent: Sequence[float],
    *,
    pending: Sequence[tuple[Sequence[float], PosteriorPrediction]] = (),
    weights: AcquisitionWeights = AcquisitionWeights(),
    pending_length_scale: float = 0.15,
    uncertainty_scales: Sequence[float] | None = None,
) -> CandidateScore:
    """Score all-maximize objective moments and normalize by source cost."""

    if not isinstance(source, FidelitySource):
        raise ActiveLearningError("candidate source must be FidelitySource")
    if not isinstance(weights, AcquisitionWeights):
        raise ActiveLearningError("weights must be AcquisitionWeights")
    try:
        prediction = posterior.predict(design, source)
    except ActiveLearningError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ActiveLearningError("posterior adapter returned malformed prediction") from error
    if not isinstance(prediction, PosteriorPrediction):
        raise ActiveLearningError("posterior adapter must return PosteriorPrediction")
    fantasy = approximate_pending_fantasization(
        design,
        incumbent,
        pending,
        length_scale=pending_length_scale,
    )
    means = _corrected_means(prediction)
    if len(means) != len(fantasy.fantasy_incumbent):
        raise ActiveLearningError("posterior and incumbent objective dimensions differ")
    try:
        scales = (
            (1.0,) * len(means)
            if uncertainty_scales is None
            else tuple(
                finite("objective uncertainty scale", value)
                for value in uncertainty_scales
            )
        )
    except TypeError as error:
        raise ActiveLearningError("objective uncertainty scales must be iterable") from error
    if len(scales) != len(means) or any(value <= 0.0 for value in scales):
        raise ActiveLearningError(
            "objective uncertainty scales must be positive and match objectives"
        )
    improvement = fmean(
        max(
            0.0,
            mean / max(1.0, abs(mean), abs(best))
            - best / max(1.0, abs(mean), abs(best)),
        )
        / 2.0
        for mean, best in zip(means, fantasy.fantasy_incumbent, strict=True)
    )
    feasibility = min(
        (gaussian_feasibility_probability(item) for item in prediction.constraints),
        default=1.0,
    )
    discrepancy = fmean(
        fmean(
            (
                _bounded_scaled_magnitude(mean, scale),
                _bounded_scaled_magnitude(standard_deviation, scale),
                _bounded_scaled_magnitude(bias_error, scale),
            )
        )
        for mean, standard_deviation, bias_error, scale in zip(
            prediction.discrepancy_means,
            prediction.discrepancy_standard_deviations,
            prediction.discrepancy_bias_standard_errors,
            scales,
            strict=True,
        )
    )
    uncertainty = fmean(
        _bounded_scaled_quadrature(
            (epistemic, aleatoric, discrepancy_std, bias_error),
            scale,
        )
        for epistemic, aleatoric, discrepancy_std, bias_error, scale in zip(
            prediction.epistemic_standard_deviations,
            prediction.aleatoric_standard_deviations,
            prediction.discrepancy_standard_deviations,
            prediction.discrepancy_bias_standard_errors,
            scales,
            strict=True,
        )
    )
    component_values = (
        improvement * feasibility,
        feasibility,
        discrepancy,
        uncertainty,
    )
    weight_values = (
        weights.predicted_improvement,
        weights.feasibility,
        weights.discrepancy,
        weights.uncertainty,
    )
    weight_scale = max(weight_values)
    if weight_scale == 0.0:
        raise ActiveLearningError("at least one acquisition weight must be positive")
    scaled_weights = tuple(value / weight_scale for value in weight_values)
    weight_total = fsum(scaled_weights)
    information_value = finite(
        "raw information value",
        (
            fsum(
                weight * component
                for weight, component in zip(
                    scaled_weights,
                    component_values,
                    strict=True,
                )
            )
            / weight_total
        )
        * fantasy.spatial_penalty,
    )
    cost_normalized = finite(
        "cost-normalized score",
        (
            information_value / (information_value + source.cost)
            if information_value > 0.0
            else 0.0
        ),
    )
    return CandidateScore(
        source,
        improvement,
        feasibility,
        discrepancy,
        uncertainty,
        fantasy.spatial_penalty,
        information_value,
        cost_normalized,
        fantasy.label,
    )


def _bounded_scaled_magnitude(value: float, declared_scale: float) -> float:
    return _bounded_scaled_quadrature(
        (abs(finite("acquisition signal", value)),),
        declared_scale,
    )


def _bounded_scaled_quadrature(
    components: Sequence[float],
    declared_scale: float,
) -> float:
    """Normalize one total SD without saturating its components separately.

    The algebraic target is ``r = hypot(components) / declared_scale`` followed
    by ``log1p(r) / (1 + log1p(r))``. Computing in the log domain avoids
    overflow and retains ordering for finite extreme values.
    """

    values = tuple(finite("uncertainty component", value) for value in components)
    scale = finite("objective uncertainty scale", declared_scale)
    if scale <= 0.0 or any(value < 0.0 for value in values):
        raise ActiveLearningError("uncertainty components and scale are invalid")
    component_scale = max(values, default=0.0)
    if component_scale == 0.0:
        return 0.0
    normalized_norm = hypot(*(value / component_scale for value in values))
    log_ratio = log(component_scale) + log(normalized_norm) - log(scale)
    log_one_plus_ratio = (
        log_ratio + log1p(exp(-log_ratio))
        if log_ratio > 0.0
        else log1p(exp(log_ratio))
    )
    return finite(
        "normalized total predictive uncertainty",
        log_one_plus_ratio / (1.0 + log_one_plus_ratio),
    )


@dataclass(frozen=True)
class HighestFidelityQuota:
    required_successes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_successes",
            integer("highest-fidelity quota", self.required_successes),
        )


@dataclass(frozen=True)
class FidelityDecision:
    source: FidelitySource
    score: CandidateScore
    quota_forced: bool
    reason: str


def select_fidelity(
    scores: Sequence[CandidateScore],
    counts: CampaignCounts,
    quota: HighestFidelityQuota,
    *,
    remaining_evaluation_slots: int,
) -> FidelityDecision:
    """Select by cost-normalized value while reserving mandatory high-fidelity slots."""

    if not scores:
        raise ActiveLearningError("fidelity selection requires candidate scores")
    remaining_evaluation_slots = integer(
        "remaining evaluation slots",
        remaining_evaluation_slots,
        minimum=1,
    )
    if any(not isinstance(item, CandidateScore) for item in scores):
        raise ActiveLearningError("fidelity selection requires CandidateScore records")
    source_names = [item.source.name for item in scores]
    if len(source_names) != len(set(source_names)):
        raise ActiveLearningError("fidelity selection requires one score per source")
    highest_scores = tuple(item for item in scores if item.source.is_highest)
    highest_names = {item.source.name for item in highest_scores}
    if len(highest_names) != 1:
        raise ActiveLearningError("scores must identify exactly one highest-fidelity source")
    highest = highest_scores[0].source
    achieved_or_pending = counts.successful(highest) + counts.pending(highest)
    deficit = max(0, quota.required_successes - achieved_or_pending)
    if deficit > remaining_evaluation_slots:
        raise ActiveLearningError("remaining slots cannot satisfy highest-fidelity quota")
    force_highest = deficit > 0 and remaining_evaluation_slots <= deficit
    pool = highest_scores if force_highest else tuple(scores)
    best = sorted(
        pool,
        key=lambda item: (
            -item.cost_normalized_score,
            -item.source.rank,
            item.source.name,
        ),
    )[0]
    return FidelityDecision(
        best.source,
        best,
        force_highest,
        (
            "mandatory highest-fidelity quota reservation"
            if force_highest
            else "maximum cost-normalized information value"
        ),
    )
