"""Predictive-coverage diagnostics for labelled posterior uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist, fmean
from typing import Sequence

from .contracts import ActiveLearningError, PosteriorPrediction, finite, integer
from .uncertainty import decompose_prediction


@dataclass(frozen=True)
class CoverageLevel:
    nominal: float
    observed: float
    count: int
    mean_interval_width: float
    absolute_error: float
    binomial_standard_error: float
    confidence_interval: tuple[float, float]

    def __post_init__(self) -> None:
        for name in (
            "nominal",
            "observed",
            "mean_interval_width",
            "absolute_error",
            "binomial_standard_error",
        ):
            object.__setattr__(self, name, finite(f"coverage {name}", getattr(self, name)))
        object.__setattr__(self, "count", integer("coverage count", self.count, minimum=1))
        if not 0.0 < self.nominal < 1.0 or not 0.0 <= self.observed <= 1.0:
            raise ActiveLearningError("coverage probabilities are invalid")
        try:
            lower, upper = (
                finite("coverage confidence bound", value)
                for value in self.confidence_interval
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, ActiveLearningError):
                raise
            raise ActiveLearningError("coverage confidence interval is malformed") from error
        if not 0.0 <= lower <= upper <= 1.0:
            raise ActiveLearningError("coverage confidence interval is invalid")
        object.__setattr__(self, "confidence_interval", (lower, upper))


@dataclass(frozen=True)
class CalibrationDiagnostics:
    levels: tuple[CoverageLevel, ...]
    expected_calibration_error: float
    maximum_calibration_error: float
    checked: bool
    passes_tolerance: bool
    tolerance: float
    sample_count: int
    stratum: str
    confidence_level: float

    def __post_init__(self) -> None:
        try:
            level_records = tuple(self.levels)
        except TypeError as error:
            raise ActiveLearningError("calibration levels must be iterable") from error
        if not level_records or any(
            not isinstance(level, CoverageLevel) for level in level_records
        ):
            raise ActiveLearningError("calibration requires CoverageLevel records")
        object.__setattr__(self, "levels", level_records)
        for name in (
            "expected_calibration_error",
            "maximum_calibration_error",
            "tolerance",
            "confidence_level",
        ):
            object.__setattr__(
                self,
                name,
                finite(f"calibration {name}", getattr(self, name)),
            )
        if not isinstance(self.checked, bool) or not isinstance(
            self.passes_tolerance,
            bool,
        ):
            raise ActiveLearningError("calibration status fields must be boolean")
        if self.tolerance < 0.0 or not 0.0 < self.confidence_level < 1.0:
            raise ActiveLearningError("calibration tolerance or confidence is invalid")
        object.__setattr__(
            self,
            "sample_count",
            integer("calibration sample count", self.sample_count, minimum=1),
        )
        if self.stratum not in {"in-domain", "ood"}:
            raise ActiveLearningError("calibration stratum must be 'in-domain' or 'ood'")


def coverage_diagnostics(
    predictions: Sequence[PosteriorPrediction],
    truths: Sequence[Sequence[float]],
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
    tolerance: float = 0.1,
    stratum: str | None = None,
    confidence_level: float = 0.95,
) -> CalibrationDiagnostics:
    """Measure coverage within one declared in-domain or OOD stratum."""

    try:
        prediction_rows = tuple(predictions)
        truth_rows = tuple(truths)
    except TypeError as error:
        raise ActiveLearningError("calibration rows must be iterable") from error
    if len(prediction_rows) != len(truth_rows) or not prediction_rows:
        raise ActiveLearningError("coverage requires equal non-empty prediction/truth rows")
    if stratum not in {"in-domain", "ood"}:
        raise ActiveLearningError("calibration stratum must be 'in-domain' or 'ood'")
    tolerance = finite("calibration tolerance", tolerance)
    confidence_level = finite("calibration confidence level", confidence_level)
    if tolerance < 0.0:
        raise ActiveLearningError("calibration tolerance cannot be negative")
    if not 0.0 < confidence_level < 1.0:
        raise ActiveLearningError("calibration confidence level must lie in (0, 1)")
    flattened: list[tuple[float, float, float]] = []
    for prediction, truth in zip(prediction_rows, truth_rows, strict=True):
        if not isinstance(prediction, PosteriorPrediction):
            raise ActiveLearningError("calibration predictions must be PosteriorPrediction")
        try:
            truth_values = tuple(truth)
        except TypeError as error:
            raise ActiveLearningError("calibration truth row must be iterable") from error
        if len(truth_values) != len(prediction.objective_means):
            raise ActiveLearningError("truth and posterior objective dimensions differ")
        corrected = tuple(
            finite("calibration corrected mean", mean + discrepancy)
            for mean, discrepancy in zip(
                prediction.objective_means,
                prediction.discrepancy_means,
                strict=True,
            )
        )
        decomposed = decompose_prediction(prediction)
        flattened.extend(
            (
                finite("truth", observed),
                predicted,
                uncertainty.standard_deviation,
            )
            for observed, predicted, uncertainty in zip(
                truth_values,
                corrected,
                decomposed,
                strict=True,
            )
        )
    results: list[CoverageLevel] = []
    normal = NormalDist()
    for raw_level in levels:
        level = finite("coverage level", raw_level)
        if not 0.0 < level < 1.0:
            raise ActiveLearningError("coverage levels must lie strictly inside (0, 1)")
        z_score = normal.inv_cdf(0.5 + level / 2.0)
        hits = 0
        widths: list[float] = []
        for truth, mean, standard_deviation in flattened:
            half_width = z_score * standard_deviation
            widths.append(2.0 * half_width)
            hits += int(mean - half_width <= truth <= mean + half_width)
        observed_coverage = hits / len(flattened)
        standard_error = sqrt(
            observed_coverage * (1.0 - observed_coverage) / len(flattened)
        )
        confidence_interval = _wilson_interval(
            hits,
            len(flattened),
            confidence_level,
        )
        results.append(
            CoverageLevel(
                level,
                observed_coverage,
                len(flattened),
                fmean(widths),
                abs(observed_coverage - level),
                standard_error,
                confidence_interval,
            )
        )
    errors = tuple(item.absolute_error for item in results)
    maximum_error = max(errors)
    return CalibrationDiagnostics(
        tuple(results),
        fmean(errors),
        maximum_error,
        True,
        maximum_error <= tolerance,
        tolerance,
        len(flattened),
        stratum,
        confidence_level,
    )


def binomial_standard_error(level: CoverageLevel) -> float:
    """Sampling uncertainty of the measured marginal coverage rate."""

    if not isinstance(level, CoverageLevel):
        raise ActiveLearningError("coverage level record is required")
    return level.binomial_standard_error


def _wilson_interval(
    successes: int,
    count: int,
    confidence_level: float,
) -> tuple[float, float]:
    successes = integer("coverage successes", successes)
    count = integer("coverage count", count, minimum=1)
    if successes > count:
        raise ActiveLearningError("coverage successes cannot exceed count")
    probability = successes / count
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z_score * z_score / count
    center = (probability + z_score * z_score / (2.0 * count)) / denominator
    half_width = (
        z_score
        * sqrt(
            probability * (1.0 - probability) / count
            + z_score * z_score / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)
