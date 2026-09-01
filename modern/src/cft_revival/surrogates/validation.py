"""Leakage-resistant splits, OOD diagnostics, calibration, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, fsum, hypot, isfinite, sqrt
from statistics import NormalDist
from typing import Sequence

from .gp import Prediction
from .identity import canonical_hash
from .normalization import (
    InputNormalizer,
    SurrogateValidationError,
    finite_matrix,
    finite_vector,
)


def _validated_predictions(
    predictions: Sequence[Prediction],
    *,
    expected_nominal_probability: float,
) -> tuple[Prediction, ...]:
    checked = tuple(predictions)
    if not checked:
        raise SurrogateValidationError("predictions must not be empty")
    for prediction in checked:
        if (
            not isinstance(prediction, Prediction)
            or not isfinite(prediction.mean)
            or not isfinite(prediction.variance)
            or prediction.variance < 0.0
            or not isfinite(prediction.nominal_probability)
            or prediction.nominal_probability != expected_nominal_probability
        ):
            raise SurrogateValidationError(
                "prediction values or nominal probability are invalid"
            )
        prediction.interval()
    return checked


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    differences = tuple(
        a - b for a, b in zip(left, right, strict=True)
    )
    result = hypot(*differences)
    if not isfinite(result):
        raise SurrogateValidationError("distance calculation overflowed")
    return result


@dataclass(frozen=True, slots=True)
class Split:
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_groups: tuple[str, ...]
    policy: str = "grouped-spatial-coordinate-closure-v2"
    coordinate_tolerance: float = 0.0

    @property
    def split_hash(self) -> str:
        return canonical_hash(
            {
                "training_indices": self.training_indices,
                "validation_indices": self.validation_indices,
                "validation_groups": self.validation_groups,
                "policy": self.policy,
                "coordinate_tolerance": self.coordinate_tolerance,
            }
        )


def grouped_spatial_split(
    inputs: Sequence[Sequence[float]],
    groups: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
    coordinate_tolerance: float = 0.0,
) -> Split:
    """Hold out spatial blocks after exact coordinate/group transitive closure.

    Exact binary64 coordinate equality is used after signed-zero
    canonicalization. Nonzero near-coordinate tolerances are rejected because
    they require declared per-dimension units/scales and cannot be inferred.
    """
    x = finite_matrix(inputs, "inputs")
    if len(groups) != len(x):
        raise SurrogateValidationError("groups must match input rows")
    if not 0.0 < validation_fraction < 1.0:
        raise SurrogateValidationError("validation_fraction must lie in (0, 1)")
    if coordinate_tolerance != 0.0:
        raise SurrogateValidationError(
            "coordinate_tolerance must be exactly zero; near-point policy "
            "requires caller-side unit-aware canonical groups"
        )
    for group in groups:
        if not isinstance(group, str) or not group:
            raise SurrogateValidationError("group IDs must be non-empty strings")
    if len(x) < 2:
        raise SurrogateValidationError("at least two rows are required")

    parent = list(range(len(x)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_for_group: dict[str, int] = {}
    first_for_coordinate: dict[tuple[float, ...], int] = {}
    for index, (point, group) in enumerate(zip(x, groups, strict=True)):
        if group in first_for_group:
            union(index, first_for_group[group])
        else:
            first_for_group[group] = index
        if point in first_for_coordinate:
            union(index, first_for_coordinate[point])
        else:
            first_for_coordinate[point] = index

    components: dict[int, list[int]] = {}
    for index in range(len(x)):
        components.setdefault(root(index), []).append(index)
    if len(components) < 2:
        raise SurrogateValidationError(
            "coordinate/group closure leaves fewer than two independent groups"
        )
    normalizer = InputNormalizer.fit(x)
    normalized = normalizer.transform(x)
    component_ids = {
        component: canonical_hash(
            {
                "coordinates": sorted({x[index] for index in rows}),
                "caller_groups": sorted({groups[index] for index in rows}),
            }
        )
        for component, rows in components.items()
    }
    centroids = {
        component: tuple(
            fsum(normalized[row][column] / len(rows) for row in rows)
            for column in range(len(x[0]))
        )
        for component, rows in components.items()
    }
    anchor = min(
        components,
        key=lambda component: canonical_hash(
            {"seed": seed, "component_id": component_ids[component]}
        ),
    )
    count = max(
        1,
        min(
            len(components) - 1,
            round(len(components) * validation_fraction),
        ),
    )
    selected = tuple(
        sorted(
            components,
            key=lambda component: (
                _distance(centroids[component], centroids[anchor]),
                component_ids[component],
            ),
        )[:count]
    )
    selected_set = set(selected)
    validation = tuple(
        index for index in range(len(x)) if root(index) in selected_set
    )
    training = tuple(
        index for index in range(len(x)) if root(index) not in selected_set
    )
    return Split(
        training,
        validation,
        tuple(component_ids[component] for component in selected),
        coordinate_tolerance=0.0,
    )


@dataclass(frozen=True, slots=True)
class OODReport:
    nearest_training_distance: float
    domain_excess_distance: float
    threshold: float
    is_out_of_distribution: bool
    policy_version: str


class OODDetector:
    """Distance diagnostics in fitted unit-box input coordinates."""

    def __init__(
        self,
        normalizer: InputNormalizer,
        normalized_training: tuple[tuple[float, ...], ...],
        threshold: float,
        threshold_quantile: float,
        threshold_multiplier: float,
    ) -> None:
        self.normalizer = normalizer
        self._training = normalized_training
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile
        self.threshold_multiplier = threshold_multiplier

    @classmethod
    def fit(
        cls,
        inputs: Sequence[Sequence[float]],
        *,
        threshold_multiplier: float = 1.5,
        threshold_quantile: float = 0.95,
    ) -> OODDetector:
        x = finite_matrix(inputs, "inputs")
        if len(x) < 2:
            raise SurrogateValidationError("OOD detection requires at least two rows")
        if (
            not isfinite(threshold_multiplier)
            or threshold_multiplier <= 0.0
            or not isfinite(threshold_quantile)
            or not 0.0 < threshold_quantile < 1.0
        ):
            raise SurrogateValidationError("OOD threshold policy is invalid")
        normalizer = InputNormalizer.fit(x)
        normalized = normalizer.transform(x)
        nearest = sorted(
            min(
                _distance(point, other)
                for other_index, other in enumerate(normalized)
                if index != other_index
            )
            for index, point in enumerate(normalized)
        )
        quantile_index = max(
            0,
            min(
                len(nearest) - 1,
                int(threshold_quantile * (len(nearest) - 1)),
            ),
        )
        threshold = nearest[quantile_index] * threshold_multiplier
        if not isfinite(threshold):
            raise SurrogateValidationError("OOD threshold overflowed")
        return cls(
            normalizer,
            normalized,
            max(threshold, 1e-12),
            threshold_quantile,
            threshold_multiplier,
        )

    def report(self, point: Sequence[float]) -> OODReport:
        normalized = self.normalizer.transform((point,))[0]
        nearest = min(
            _distance(normalized, training) for training in self._training
        )
        excess = tuple(
            value if value < 0.0 else value - 1.0 if value > 1.0 else 0.0
            for value in normalized
        )
        domain_excess = hypot(*excess)
        if not isfinite(domain_excess):
            raise SurrogateValidationError("OOD domain distance overflowed")
        return OODReport(
            nearest,
            domain_excess,
            self.threshold,
            domain_excess > 0.0 or nearest > self.threshold,
            "unit-box-nearest-euclidean-v1",
        )


@dataclass(frozen=True, slots=True)
class VarianceCalibrator:
    variance_scale: float
    nominal_probability: float
    fit_sample_count: int
    fit_role: str = "calibration-fit"

    def __post_init__(self) -> None:
        if not isfinite(self.variance_scale) or self.variance_scale <= 0.0:
            raise SurrogateValidationError(
                "calibration variance scale must be finite and positive"
            )
        if (
            not isfinite(self.nominal_probability)
            or not 0.5 < self.nominal_probability < 1.0
        ):
            raise SurrogateValidationError(
                "calibration nominal probability is invalid"
            )
        if self.fit_sample_count < 1 or self.fit_role != "calibration-fit":
            raise SurrogateValidationError("calibration fit metadata is invalid")

    @classmethod
    def fit(
        cls,
        truth: Sequence[float],
        predictions: Sequence[Prediction],
        *,
        nominal_probability: float = 0.95,
    ) -> VarianceCalibrator:
        values = finite_vector(truth, "calibration truth")
        if len(values) != len(predictions) or not predictions:
            raise SurrogateValidationError(
                "calibration truth and predictions must have equal length"
            )
        if (
            not isfinite(nominal_probability)
            or not 0.5 < nominal_probability < 1.0
        ):
            raise SurrogateValidationError(
                "calibration nominal probability must lie in (0.5, 1)"
            )
        checked_predictions = _validated_predictions(
            predictions,
            expected_nominal_probability=nominal_probability,
        )
        standardized = []
        positive_variances = [
            prediction.variance
            for prediction in checked_predictions
            if prediction.variance > 0.0
        ]
        if not positive_variances:
            raise SurrogateValidationError(
                "calibration requires positive predictive variance"
            )
        floor = min(positive_variances) * 1e-12
        if not isfinite(floor) or floor < 0.0:
            raise SurrogateValidationError("calibration variance floor is invalid")
        for value, prediction in zip(values, checked_predictions, strict=True):
            try:
                error = abs(value - prediction.mean)
                score = error / sqrt(max(prediction.variance, floor))
            except (ArithmeticError, OverflowError) as cause:
                raise SurrogateValidationError(
                    "calibration arithmetic overflowed"
                ) from cause
            if not isfinite(error) or not isfinite(score):
                raise SurrogateValidationError(
                    "calibration residual is nonfinite"
                )
            standardized.append(score)
        standardized.sort()
        rank = min(
            len(standardized) - 1,
            max(
                0,
                ceil((len(standardized) + 1) * nominal_probability) - 1,
            ),
        )
        observed_radius = standardized[rank]
        nominal_radius = NormalDist().inv_cdf(
            0.5 + nominal_probability / 2.0
        )
        try:
            scale = max((observed_radius / nominal_radius) ** 2, 1e-12)
        except (ArithmeticError, OverflowError) as cause:
            raise SurrogateValidationError(
                "calibration scale overflowed"
            ) from cause
        if not isfinite(scale):
            raise SurrogateValidationError("calibration scale is nonfinite")
        return cls(scale, nominal_probability, len(values))

    def apply(self, prediction: Prediction) -> Prediction:
        try:
            variance = prediction.variance * self.variance_scale
        except (ArithmeticError, OverflowError) as cause:
            raise SurrogateValidationError(
                "calibrated variance overflowed"
            ) from cause
        if not isfinite(variance):
            raise SurrogateValidationError("calibrated variance is nonfinite")
        return Prediction(
            prediction.mean,
            variance,
            self.nominal_probability,
            "posterior-latent-calibrated",
        )


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    rmse: float
    output_range: float
    range_normalized_rmse: float
    rmse_acceptance_threshold: float
    rmse_accepted: bool
    mae: float
    worst_case_absolute_error: float
    interval_coverage: float
    nominal_probability: float
    coverage_target: float
    coverage_tolerance: float
    coverage_accepted: bool
    minimum_coverage_sample_count: int
    assessment_limited: bool
    model_quality_passed: bool
    sample_count: int
    assessment_role: str = "held-out-assessment"


def regression_metrics(
    truth: Sequence[float],
    predictions: Sequence[Prediction],
    *,
    nominal_probability: float = 0.95,
    coverage_target: float | None = None,
    coverage_tolerance: float = 0.05,
    minimum_coverage_sample_count: int = 30,
    rmse_acceptance_threshold: float = 0.05,
    quality_scale: float | None = None,
    assessment_role: str = "held-out-assessment",
) -> RegressionMetrics:
    values = finite_vector(truth, "assessment truth")
    if len(values) != len(predictions) or not predictions:
        raise SurrogateValidationError(
            "assessment truth and predictions must have equal nonzero length"
        )
    target = nominal_probability if coverage_target is None else coverage_target
    if (
        not isfinite(nominal_probability)
        or not 0.5 < nominal_probability < 1.0
        or not isfinite(target)
        or not 0.0 <= target <= 1.0
        or not isfinite(coverage_tolerance)
        or not 0.0 <= coverage_tolerance <= 1.0
        or isinstance(minimum_coverage_sample_count, bool)
        or minimum_coverage_sample_count < 1
        or not isfinite(rmse_acceptance_threshold)
        or not 0.0 < rmse_acceptance_threshold < 1.0
    ):
        raise SurrogateValidationError("coverage assessment policy is invalid")
    if assessment_role != "held-out-assessment":
        raise SurrogateValidationError(
            "metrics assessment role must be held-out-assessment"
        )
    checked_predictions = _validated_predictions(
        predictions,
        expected_nominal_probability=nominal_probability,
    )
    errors = []
    covered = 0
    for value, prediction in zip(values, checked_predictions, strict=True):
        try:
            error = abs(value - prediction.mean)
            lower, upper = prediction.interval()
        except (ArithmeticError, OverflowError, ValueError) as cause:
            raise SurrogateValidationError(
                "metric computation failed"
            ) from cause
        if not isfinite(error) or not isfinite(lower) or not isfinite(upper):
            raise SurrogateValidationError("metric inputs produced nonfinite values")
        if lower > upper:
            raise SurrogateValidationError("predictive interval is invalid")
        errors.append(error)
        covered += lower <= value <= upper
    try:
        rmse = hypot(*errors) / sqrt(len(errors))
        mae = fsum(error / len(errors) for error in errors)
        coverage = covered / len(errors)
    except (ArithmeticError, OverflowError) as cause:
        raise SurrogateValidationError("metric aggregation overflowed") from cause
    if not all(isfinite(value) for value in (rmse, mae, coverage)):
        raise SurrogateValidationError("metrics are nonfinite")
    output_range = max(values) - min(values)
    try:
        selected_scale = (
            output_range if quality_scale is None else float(quality_scale)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SurrogateValidationError("quality scale must be numeric") from error
    if not isfinite(selected_scale) or selected_scale <= 0.0:
        raise SurrogateValidationError(
            "range-normalized RMSE requires a finite positive quality scale"
        )
    normalized_rmse = rmse / selected_scale
    if not isfinite(normalized_rmse):
        raise SurrogateValidationError("range-normalized RMSE is nonfinite")
    assessment_limited = len(errors) < minimum_coverage_sample_count
    coverage_accepted = (
        not assessment_limited
        and abs(coverage - target) <= coverage_tolerance
    )
    rmse_accepted = normalized_rmse <= rmse_acceptance_threshold
    return RegressionMetrics(
        rmse,
        selected_scale,
        normalized_rmse,
        rmse_acceptance_threshold,
        rmse_accepted,
        mae,
        max(errors),
        coverage,
        nominal_probability,
        target,
        coverage_tolerance,
        coverage_accepted,
        minimum_coverage_sample_count,
        assessment_limited,
        rmse_accepted and coverage_accepted,
        len(errors),
        assessment_role,
    )
