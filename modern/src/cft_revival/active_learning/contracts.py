"""Dependency-light contracts for multi-fidelity active learning.

The package deliberately defines structural adapters instead of importing the
optimization campaign or any particular Gaussian-process implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Protocol, Sequence, runtime_checkable


class ActiveLearningError(ValueError):
    """An active-learning input violates a numerical or semantic invariant."""


def finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ActiveLearningError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ActiveLearningError(f"{name} must be a finite real number") from error
    if not isfinite(result):
        raise ActiveLearningError(f"{name} must be finite")
    return result


def integer(name: str, value: object, *, minimum: int = 0) -> int:
    """Return a true integer, rejecting booleans and integer-valued floats."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ActiveLearningError(f"{name} must be an integer {qualifier}")
    return value


@dataclass(frozen=True)
class FidelitySource:
    """An ordered information source with cost relative to highest fidelity."""

    name: str
    rank: int
    cost: float
    is_highest: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ActiveLearningError("source requires a non-empty name")
        object.__setattr__(self, "cost", finite("source cost", self.cost))
        object.__setattr__(self, "rank", integer("source rank", self.rank))
        if self.cost <= 0.0:
            raise ActiveLearningError("source cost must be positive")
        if not isinstance(self.is_highest, bool):
            raise ActiveLearningError("is_highest must be boolean")


@dataclass(frozen=True)
class GaussianConstraint:
    """A residual distribution using the convention residual <= 0 is feasible."""

    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", finite("constraint mean", self.mean))
        object.__setattr__(
            self,
            "standard_deviation",
            finite("constraint standard deviation", self.standard_deviation),
        )
        if self.standard_deviation < 0.0:
            raise ActiveLearningError("constraint standard deviation cannot be negative")


@dataclass(frozen=True)
class PosteriorPrediction:
    """Posterior moments with separately labelled uncertainty sources."""

    objective_means: tuple[float, ...]
    epistemic_standard_deviations: tuple[float, ...]
    aleatoric_standard_deviations: tuple[float, ...]
    discrepancy_means: tuple[float, ...] = ()
    discrepancy_standard_deviations: tuple[float, ...] = ()
    constraints: tuple[GaussianConstraint, ...] = ()
    discrepancy_bias_standard_errors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        try:
            means = tuple(finite("objective mean", value) for value in self.objective_means)
            epistemic = tuple(
                finite("epistemic standard deviation", value)
                for value in self.epistemic_standard_deviations
            )
            aleatoric = tuple(
                finite("aleatoric standard deviation", value)
                for value in self.aleatoric_standard_deviations
            )
            discrepancy_means = (
                tuple(finite("discrepancy mean", value) for value in self.discrepancy_means)
                if self.discrepancy_means
                else (0.0,) * len(means)
            )
            discrepancy_std = (
                tuple(
                    finite("discrepancy standard deviation", value)
                    for value in self.discrepancy_standard_deviations
                )
                if self.discrepancy_standard_deviations
                else (0.0,) * len(means)
            )
            bias_standard_errors = (
                tuple(
                    finite("discrepancy bias standard error", value)
                    for value in self.discrepancy_bias_standard_errors
                )
                if self.discrepancy_bias_standard_errors
                else (0.0,) * len(means)
            )
            constraints = tuple(self.constraints)
        except (TypeError, ValueError, AttributeError) as error:
            if isinstance(error, ActiveLearningError):
                raise
            raise ActiveLearningError("malformed posterior prediction") from error
        sizes = {
            len(means),
            len(epistemic),
            len(aleatoric),
            len(discrepancy_means),
            len(discrepancy_std),
            len(bias_standard_errors),
        }
        if not means or len(sizes) != 1:
            raise ActiveLearningError("all objective posterior vectors need equal non-zero length")
        if any(
            value < 0.0
            for value in (*epistemic, *aleatoric, *discrepancy_std, *bias_standard_errors)
        ):
            raise ActiveLearningError("standard deviations cannot be negative")
        if any(not isinstance(item, GaussianConstraint) for item in constraints):
            raise ActiveLearningError("constraints must be GaussianConstraint records")
        object.__setattr__(self, "objective_means", means)
        object.__setattr__(self, "epistemic_standard_deviations", epistemic)
        object.__setattr__(self, "aleatoric_standard_deviations", aleatoric)
        object.__setattr__(self, "discrepancy_means", discrepancy_means)
        object.__setattr__(self, "discrepancy_standard_deviations", discrepancy_std)
        object.__setattr__(
            self,
            "discrepancy_bias_standard_errors",
            bias_standard_errors,
        )
        object.__setattr__(self, "constraints", constraints)


@runtime_checkable
class PosteriorAdapter(Protocol):
    """Minimal adapter implemented by a GP, ensemble, or analytical posterior."""

    def predict(
        self, design: Sequence[float], source: FidelitySource
    ) -> PosteriorPrediction:
        """Return moments without requiring samples or framework-specific tensors."""


@dataclass(frozen=True)
class CampaignCounts:
    """Scheduler accounting consumed without importing campaign implementation."""

    successful_by_source: tuple[tuple[str, int], ...]
    pending_by_source: tuple[tuple[str, int], ...] = ()
    total_completed_successes: int = 0

    def __post_init__(self) -> None:
        try:
            successful = tuple(self.successful_by_source)
            pending = tuple(self.pending_by_source)
        except TypeError as error:
            raise ActiveLearningError("campaign source counts must be iterable") from error
        normalized_groups: list[tuple[tuple[str, int], ...]] = []
        for label, pairs in (
            ("successful", successful),
            ("pending", pending),
        ):
            names: set[str] = set()
            try:
                records = tuple((name, count) for name, count in pairs)
            except (TypeError, ValueError) as error:
                raise ActiveLearningError(f"{label} source counts are malformed") from error
            for name, count in records:
                if not isinstance(name, str) or not name or name in names:
                    raise ActiveLearningError(
                        f"{label} source counts must be unique and non-negative"
                    )
                integer(f"{label} count for {name}", count)
                names.add(name)
            normalized_groups.append(
                tuple(
                    (name, integer(f"{label} count for {name}", count))
                    for name, count in records
                )
            )
        object.__setattr__(self, "successful_by_source", normalized_groups[0])
        object.__setattr__(self, "pending_by_source", normalized_groups[1])
        integer("total completed successes", self.total_completed_successes)

    def successful(self, source: FidelitySource) -> int:
        return dict(self.successful_by_source).get(source.name, 0)

    def pending(self, source: FidelitySource) -> int:
        return dict(self.pending_by_source).get(source.name, 0)


@runtime_checkable
class CampaignRecordAdapter(Protocol):
    """Boundary for adapting event logs or campaign databases."""

    def counts(self) -> CampaignCounts:
        """Return immutable source-level accounting."""

    def pending_designs(self) -> Sequence[tuple[Hashable, Sequence[float], str]]:
        """Return (identity, design, source-name) records for in-flight work."""
