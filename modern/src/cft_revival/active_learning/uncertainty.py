"""Uncertainty and cross-fidelity discrepancy decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, sqrt
from statistics import fmean
from typing import Sequence

from .contracts import ActiveLearningError, PosteriorPrediction, finite, integer


@dataclass(frozen=True)
class VarianceDecomposition:
    """Independent variance components; covariance is intentionally not invented."""

    epistemic: float
    aleatoric: float
    discrepancy: float
    bias_estimation: float = 0.0

    def __post_init__(self) -> None:
        for name in ("epistemic", "aleatoric", "discrepancy", "bias_estimation"):
            value = finite(name, getattr(self, name))
            if value < 0.0:
                raise ActiveLearningError(f"{name} variance cannot be negative")
            object.__setattr__(self, name, value)

    @property
    def total(self) -> float:
        try:
            return finite(
                "total predictive variance",
                fsum(
                    (
                        self.epistemic,
                        self.aleatoric,
                        self.discrepancy,
                        self.bias_estimation,
                    )
                ),
            )
        except OverflowError as error:
            raise ActiveLearningError("total predictive variance overflowed") from error

    @property
    def standard_deviation(self) -> float:
        return sqrt(self.total)


def decompose_prediction(
    prediction: PosteriorPrediction,
) -> tuple[VarianceDecomposition, ...]:
    """Convert explicitly labelled standard deviations into variances."""

    return tuple(
        VarianceDecomposition(
            finite("epistemic variance", epistemic * epistemic),
            finite("aleatoric variance", aleatoric * aleatoric),
            finite("discrepancy variance", discrepancy * discrepancy),
            finite("bias-estimation variance", bias_error * bias_error),
        )
        for epistemic, aleatoric, discrepancy, bias_error in zip(
            prediction.epistemic_standard_deviations,
            prediction.aleatoric_standard_deviations,
            prediction.discrepancy_standard_deviations,
            prediction.discrepancy_bias_standard_errors,
            strict=True,
        )
    )


@dataclass(frozen=True)
class DiscrepancyEstimate:
    """Additive correction ``highest - lower`` estimated from paired designs."""

    bias: tuple[float, ...]
    residual_variance: tuple[float, ...]
    paired_count: int

    def __post_init__(self) -> None:
        bias = tuple(finite("discrepancy bias", value) for value in self.bias)
        variance = tuple(
            finite("discrepancy residual variance", value)
            for value in self.residual_variance
        )
        if (
            not bias
            or len(bias) != len(variance)
            or any(value < 0.0 for value in variance)
        ):
            raise ActiveLearningError("invalid paired discrepancy estimate")
        integer("paired discrepancy count", self.paired_count, minimum=1)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "residual_variance", variance)

    @property
    def bias_standard_error(self) -> tuple[float, ...]:
        return tuple(sqrt(value / self.paired_count) for value in self.residual_variance)

    @property
    def residual_standard_deviation(self) -> tuple[float, ...]:
        """Irreducible paired-residual heterogeneity, not mean-bias uncertainty."""

        return tuple(sqrt(value) for value in self.residual_variance)

    def correct(self, lower_prediction: Sequence[float]) -> tuple[float, ...]:
        if len(lower_prediction) != len(self.bias):
            raise ActiveLearningError("prediction and discrepancy dimensions differ")
        return tuple(
            finite("corrected prediction", value + correction)
            for value, correction in zip(lower_prediction, self.bias, strict=True)
        )


def estimate_additive_discrepancy(
    lower_outputs: Sequence[Sequence[float]],
    highest_outputs: Sequence[Sequence[float]],
) -> DiscrepancyEstimate:
    """Estimate additive source bias only from paired observations."""

    if len(lower_outputs) != len(highest_outputs) or not lower_outputs:
        raise ActiveLearningError("discrepancy estimation requires equal non-empty pairs")
    dimension = len(lower_outputs[0])
    if dimension < 1 or any(
        len(row) != dimension for row in (*lower_outputs, *highest_outputs)
    ):
        raise ActiveLearningError("paired outputs must have a consistent dimension")
    residuals = [
        tuple(
            finite("paired residual", high - low)
            for low, high in zip(lower, highest, strict=True)
        )
        for lower, highest in zip(lower_outputs, highest_outputs, strict=True)
    ]
    try:
        bias = tuple(
            finite(
                "mean discrepancy bias",
                fmean(row[index] for row in residuals),
            )
            for index in range(dimension)
        )
    except (OverflowError, ValueError) as error:
        raise ActiveLearningError("mean discrepancy bias is not representable") from error
    denominator = max(1, len(residuals) - 1)
    variance_values: list[float] = []
    for index in range(dimension):
        deviations = tuple(row[index] - bias[index] for row in residuals)
        scale = max((abs(value) for value in deviations), default=0.0)
        if scale == 0.0:
            variance_values.append(0.0)
            continue
        normalized_sum = fsum((value / scale) ** 2 for value in deviations)
        variance_values.append(
            finite(
                "discrepancy residual variance",
                (normalized_sum / denominator) * scale * scale,
            )
        )
    variance = tuple(variance_values)
    return DiscrepancyEstimate(bias, variance, len(residuals))


def bias_correct_prediction(
    prediction: PosteriorPrediction,
    discrepancy: DiscrepancyEstimate,
) -> PosteriorPrediction:
    """Apply paired bias while preserving each uncertainty label."""

    if len(prediction.objective_means) != len(discrepancy.bias):
        raise ActiveLearningError("prediction and discrepancy dimensions differ")
    combined_discrepancy_std = tuple(
        finite(
            "combined discrepancy spread",
            hypot(model_std, residual_std),
        )
        for model_std, residual_std in zip(
            prediction.discrepancy_standard_deviations,
            discrepancy.residual_standard_deviation,
            strict=True,
        )
    )
    combined_bias_standard_error = tuple(
        finite(
            "combined bias standard error",
            hypot(model_error, estimate_error),
        )
        for model_error, estimate_error in zip(
            prediction.discrepancy_bias_standard_errors,
            discrepancy.bias_standard_error,
            strict=True,
        )
    )
    return PosteriorPrediction(
        prediction.objective_means,
        prediction.epistemic_standard_deviations,
        prediction.aleatoric_standard_deviations,
        tuple(
            model_bias + fitted_bias
            for model_bias, fitted_bias in zip(
                prediction.discrepancy_means,
                discrepancy.bias,
                strict=True,
            )
        ),
        combined_discrepancy_std,
        prediction.constraints,
        combined_bias_standard_error,
    )
