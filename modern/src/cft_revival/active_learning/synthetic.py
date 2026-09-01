"""Small analytical multi-fidelity problems with explicit mathematical truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import (
    ActiveLearningError,
    FidelitySource,
    GaussianConstraint,
    PosteriorPrediction,
)


SYNTHETIC_SOURCES: tuple[FidelitySource, ...] = (
    FidelitySource("F0-analytic", 0, 0.02),
    FidelitySource("F1-reduced", 1, 0.1),
    FidelitySource("F2-hybrid", 2, 0.35),
    FidelitySource("F3-truth", 3, 1.0, is_highest=True),
)

_BIASES: dict[str, tuple[float, float]] = {
    "F0-analytic": (-0.25, 0.20),
    "F1-reduced": (0.08, -0.05),
    "F2-hybrid": (-0.02, 0.015),
    "F3-truth": (0.0, 0.0),
}

_EPISTEMIC: dict[str, float] = {
    "F0-analytic": 0.18,
    "F1-reduced": 0.10,
    "F2-hybrid": 0.04,
    "F3-truth": 0.015,
}


def _coordinate(design: Sequence[float]) -> float:
    if len(design) != 1:
        raise ActiveLearningError("synthetic problem is one-dimensional")
    x = float(design[0])
    if not 0.0 <= x <= 1.0:
        raise ActiveLearningError("synthetic coordinate must lie in [0, 1]")
    return x


def analytical_truth(design: Sequence[float]) -> tuple[float, float]:
    """Two competing all-maximize parabolas with a known Pareto interval."""

    x = _coordinate(design)
    return (1.0 - (x - 0.25) ** 2, 1.0 - (x - 0.75) ** 2)


def analytical_constraint_residual(design: Sequence[float]) -> float:
    """Feasible when x is in [0.1, 0.9]."""

    return abs(_coordinate(design) - 0.5) - 0.4


def source_output(
    design: Sequence[float],
    source: FidelitySource,
) -> tuple[float, float]:
    """Deterministic source output; only ``analytical_truth`` is ground truth."""

    if source.name not in _BIASES:
        raise ActiveLearningError(f"unknown synthetic source {source.name!r}")
    truth = analytical_truth(design)
    return tuple(
        value + bias
        for value, bias in zip(truth, _BIASES[source.name], strict=True)
    )


@dataclass(frozen=True)
class AnalyticalPosterior:
    """Exact synthetic moments used to test algorithms, not performance claims."""

    include_known_discrepancy: bool = True

    def predict(
        self,
        design: Sequence[float],
        source: FidelitySource,
    ) -> PosteriorPrediction:
        means = source_output(design, source)
        source_bias = _BIASES[source.name]
        discrepancy = (
            tuple(-value for value in source_bias)
            if self.include_known_discrepancy
            else (0.0, 0.0)
        )
        epistemic = _EPISTEMIC[source.name]
        discrepancy_std = 0.5 * epistemic if not source.is_highest else 0.0
        return PosteriorPrediction(
            means,
            (epistemic, epistemic),
            (0.01, 0.01),
            discrepancy,
            (discrepancy_std, discrepancy_std),
            (
                GaussianConstraint(
                    analytical_constraint_residual(design),
                    max(0.005, epistemic * 0.1),
                ),
            ),
        )


def tolerance_response(
    design: Sequence[float],
    tolerances: dict[str, float],
) -> tuple[float, float]:
    """Analytical operating/manufacturing perturbation response."""

    shifted = (
        _coordinate(design)
        + tolerances.get("manufacturing_offset", 0.0)
        + 0.5 * tolerances.get("operating_drift", 0.0)
    )
    clipped = min(1.0, max(0.0, shifted))
    return analytical_truth((clipped,))
