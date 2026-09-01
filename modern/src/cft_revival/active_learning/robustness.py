"""Tolerance propagation, risk summaries, and robust promotion."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from numbers import Real
from random import Random
from statistics import NormalDist, fmean
from typing import Callable, Protocol, Sequence

from .contracts import ActiveLearningError, GaussianConstraint, finite, integer


class Distribution(Protocol):
    """Small sampling boundary for manufacturing and operating tolerances."""

    def sample(self, rng: Random) -> float:
        """Draw one finite value from ``rng``."""


@dataclass(frozen=True)
class NormalTolerance:
    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", finite("normal mean", self.mean))
        object.__setattr__(
            self,
            "standard_deviation",
            finite("normal standard deviation", self.standard_deviation),
        )
        if self.standard_deviation < 0.0:
            raise ActiveLearningError("normal standard deviation cannot be negative")

    def sample(self, rng: Random) -> float:
        return rng.gauss(self.mean, self.standard_deviation)


@dataclass(frozen=True)
class UniformTolerance:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", finite("uniform lower", self.lower))
        object.__setattr__(self, "upper", finite("uniform upper", self.upper))
        if self.lower > self.upper:
            raise ActiveLearningError("uniform lower cannot exceed upper")

    def sample(self, rng: Random) -> float:
        return rng.uniform(self.lower, self.upper)


@dataclass(frozen=True)
class TriangularTolerance:
    lower: float
    mode: float
    upper: float

    def __post_init__(self) -> None:
        for name in ("lower", "mode", "upper"):
            object.__setattr__(self, name, finite(f"triangular {name}", getattr(self, name)))
        if not self.lower <= self.mode <= self.upper:
            raise ActiveLearningError("triangular parameters require lower <= mode <= upper")

    def sample(self, rng: Random) -> float:
        return rng.triangular(self.lower, self.upper, self.mode)


@dataclass(frozen=True)
class ToleranceVariable:
    name: str
    distribution: Distribution
    kind: str  # "manufacturing" or "operating"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.kind not in {"manufacturing", "operating"}
        ):
            raise ActiveLearningError(
                "tolerance variable needs a name and manufacturing/operating kind"
            )
        try:
            sampler = getattr(self.distribution, "sample")
        except Exception as error:
            raise ActiveLearningError(
                "tolerance distribution sample adapter is inaccessible"
            ) from error
        if not callable(sampler):
            raise ActiveLearningError(
                "tolerance distribution must implement callable sample(rng)"
            )


def quantile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated empirical quantile with endpoints included."""

    samples = sorted(finite("sample", value) for value in values)
    probability = finite("quantile probability", probability)
    if not samples or not 0.0 <= probability <= 1.0:
        raise ActiveLearningError("quantile needs samples and probability in [0, 1]")
    position = probability * (len(samples) - 1)
    lower = int(position)
    upper = min(lower + 1, len(samples) - 1)
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def cvar(
    values: Sequence[float],
    probability: float,
    *,
    tail: str = "lower",
) -> float:
    """Finite-sample expected shortfall with fractional boundary mass.

    ``probability`` is the tail mass for both tails. Each empirical observation
    has mass ``1 / n``; if the requested tail cuts an observation's mass, only
    the required fraction contributes. Sorting makes ties deterministic without
    over-counting all observations equal to the boundary value.
    """

    samples = sorted(finite("sample", value) for value in values)
    probability = finite("CVaR tail probability", probability)
    if not samples or not 0.0 < probability <= 1.0:
        raise ActiveLearningError("CVaR needs samples and probability in (0, 1]")
    if tail not in {"lower", "upper"}:
        raise ActiveLearningError("CVaR tail must be 'lower' or 'upper'")
    ordered = samples if tail == "lower" else list(reversed(samples))
    required_mass = probability * len(ordered)
    if required_mass == 0.0:
        return ordered[0]
    whole = int(required_mass)
    fractional = required_mass - whole
    weighted = [(value, 1.0) for value in ordered[:whole]]
    if fractional > 0.0:
        weighted.append((ordered[whole], fractional))
    scale = max(abs(value) for value, _ in weighted)
    if scale == 0.0:
        return 0.0
    normalized_sum = fsum((value / scale) * mass for value, mass in weighted)
    return finite("CVaR", (normalized_sum / required_mass) * scale)


@dataclass(frozen=True)
class MonteCarloSummary:
    samples: tuple[tuple[float, ...], ...]
    means: tuple[float, ...]
    quantiles: tuple[tuple[float, tuple[float, ...]], ...]
    lower_cvar: tuple[float, ...]
    upper_cvar: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        integer("Monte Carlo seed", self.seed)


def propagate_monte_carlo(
    model: Callable[[Sequence[float], dict[str, float]], Sequence[float]],
    design: Sequence[float],
    tolerances: Sequence[ToleranceVariable],
    *,
    draws: int,
    seed: int,
    quantile_probabilities: Sequence[float] = (0.05, 0.5, 0.95),
    cvar_probability: float = 0.05,
) -> MonteCarloSummary:
    """Propagate named tolerances through an arbitrary deterministic model."""

    draws = integer("Monte Carlo draws", draws, minimum=1)
    seed = integer("Monte Carlo seed", seed)
    try:
        tolerance_records = tuple(tolerances)
    except TypeError as error:
        raise ActiveLearningError("tolerances must be iterable") from error
    if any(not isinstance(item, ToleranceVariable) for item in tolerance_records):
        raise ActiveLearningError("tolerances must be ToleranceVariable records")
    names = [item.name for item in tolerance_records]
    if len(names) != len(set(names)):
        raise ActiveLearningError("tolerance variable names must be unique")
    rng = Random(seed)
    outputs: list[tuple[float, ...]] = []
    dimension: int | None = None
    for _ in range(draws):
        perturbations = {
            tolerance.name: _sample_tolerance(tolerance, rng)
            for tolerance in tolerance_records
        }
        try:
            raw_row = model(tuple(design), perturbations)
            row = tuple(
                finite("propagated model output", value) for value in raw_row
            )
        except (TypeError, ValueError, AttributeError) as error:
            if isinstance(error, ActiveLearningError):
                raise
            raise ActiveLearningError("propagated model returned malformed output") from error
        if dimension is None:
            dimension = len(row)
        if not row or len(row) != dimension:
            raise ActiveLearningError("propagated outputs need a fixed non-zero dimension")
        outputs.append(row)
    assert dimension is not None
    columns = tuple(
        tuple(row[index] for row in outputs) for index in range(dimension)
    )
    quantiles = tuple(
        (
            finite("quantile probability", probability),
            tuple(quantile(column, probability) for column in columns),
        )
        for probability in quantile_probabilities
    )
    return MonteCarloSummary(
        tuple(outputs),
        tuple(fmean(column) for column in columns),
        quantiles,
        tuple(cvar(column, cvar_probability, tail="lower") for column in columns),
        tuple(cvar(column, cvar_probability, tail="upper") for column in columns),
        seed,
    )


def _sample_tolerance(tolerance: ToleranceVariable, rng: Random) -> float:
    """Invoke a scalar distribution adapter through one fail-closed boundary."""

    try:
        sampler = getattr(tolerance.distribution, "sample")
        if not callable(sampler):
            raise TypeError("sample is not callable")
        raw_value = sampler(rng)
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ActiveLearningError(
                f"tolerance sample {tolerance.name!r} must be a non-boolean real scalar"
            )
        return finite(f"tolerance sample {tolerance.name}", raw_value)
    except ActiveLearningError:
        raise
    except Exception as error:
        raise ActiveLearningError(
            f"tolerance distribution {tolerance.name!r} failed to return a finite scalar"
        ) from error


def gaussian_feasibility_probability(constraint: GaussianConstraint) -> float:
    """Probability that a Gaussian residual is <= 0."""

    if not isinstance(constraint, GaussianConstraint):
        raise ActiveLearningError("constraint must be a GaussianConstraint")
    if constraint.standard_deviation == 0.0:
        return 1.0 if constraint.mean <= 0.0 else 0.0
    probability = NormalDist().cdf(-constraint.mean / constraint.standard_deviation)
    if not 0.0 <= probability <= 1.0:
        raise ActiveLearningError("constraint probability is invalid")
    return probability


@dataclass(frozen=True)
class PromotionAssessment:
    nominal_feasible: bool
    robust_feasible: bool
    chance_feasible: bool
    per_constraint_probabilities: tuple[float, ...]
    nondominated: bool
    eligible: bool
    requires_highest_fidelity_reevaluation: bool = True
    probability_policy: str = "every marginal must pass; no independence assumption"


def dominates(
    left: Sequence[float],
    right: Sequence[float],
    directions: Sequence[str],
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    """Return dominance after mapping maximize/minimize to all-maximize."""

    left_values, comparisons, direction_values = _validate_pareto_inputs(
        left,
        (right,),
        directions,
    )
    tolerance = finite("dominance tolerance", tolerance)
    if tolerance < 0.0:
        raise ActiveLearningError("dominance tolerance cannot be negative")
    return _dominates_validated(
        left_values,
        comparisons[0],
        direction_values,
        tolerance,
    )


def _validate_pareto_inputs(
    objectives: Sequence[float],
    comparison_objectives: Sequence[Sequence[float]],
    directions: Sequence[str],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...], tuple[str, ...]]:
    """Validate the entire Pareto domain before any feasibility/promotion logic."""

    try:
        objective_values = tuple(
            finite("promotion objective", value) for value in objectives
        )
        direction_values = tuple(directions)
        raw_comparisons = tuple(comparison_objectives)
    except (TypeError, ValueError, AttributeError) as error:
        if isinstance(error, ActiveLearningError):
            raise
        raise ActiveLearningError("malformed Pareto promotion inputs") from error
    if not objective_values or len(direction_values) != len(objective_values):
        raise ActiveLearningError("objective vectors and directions must align")
    if any(
        not isinstance(direction, str)
        or direction not in {"maximize", "minimize"}
        for direction in direction_values
    ):
        raise ActiveLearningError(
            "objective directions must be exactly 'maximize' or 'minimize'"
        )
    comparisons: list[tuple[float, ...]] = []
    for raw in raw_comparisons:
        try:
            row = tuple(finite("comparison objective", value) for value in raw)
        except (TypeError, ValueError, AttributeError) as error:
            if isinstance(error, ActiveLearningError):
                raise
            raise ActiveLearningError("malformed comparison objective vector") from error
        if len(row) != len(objective_values):
            raise ActiveLearningError("comparison objective dimensions must align")
        comparisons.append(row)
    return objective_values, tuple(comparisons), direction_values


def _dominates_validated(
    left: tuple[float, ...],
    right: tuple[float, ...],
    directions: tuple[str, ...],
    tolerance: float,
) -> bool:
    signs = tuple(1.0 if direction == "maximize" else -1.0 for direction in directions)
    transformed_left = tuple(sign * value for sign, value in zip(signs, left, strict=True))
    transformed_right = tuple(sign * value for sign, value in zip(signs, right, strict=True))
    normalized = tuple(
        (
            a / max(1.0, abs(a), abs(b), tolerance),
            b / max(1.0, abs(a), abs(b), tolerance),
            tolerance / max(1.0, abs(a), abs(b), tolerance),
        )
        for a, b in zip(transformed_left, transformed_right, strict=True)
    )
    no_worse = all(a >= b - tol for a, b, tol in normalized)
    better = any(a > b + tol for a, b, tol in normalized)
    return no_worse and better


def assess_promotion(
    objectives: Sequence[float],
    constraints: Sequence[GaussianConstraint],
    comparison_objectives: Sequence[Sequence[float]],
    directions: Sequence[str],
    *,
    robust_sigma: float = 2.0,
    chance_threshold: float = 0.95,
) -> PromotionAssessment:
    """Require nondominance plus nominal, sigma-robust, and marginal chance feasibility."""

    objective_values, comparisons, direction_values = _validate_pareto_inputs(
        objectives,
        comparison_objectives,
        directions,
    )
    robust_sigma = finite("robust sigma", robust_sigma)
    chance_threshold = finite("chance threshold", chance_threshold)
    if robust_sigma < 0.0 or not 0.0 <= chance_threshold <= 1.0:
        raise ActiveLearningError("invalid robust/chance promotion thresholds")
    try:
        constraint_values = tuple(constraints)
    except TypeError as error:
        raise ActiveLearningError("promotion constraints must be iterable") from error
    if any(not isinstance(item, GaussianConstraint) for item in constraint_values):
        raise ActiveLearningError("promotion constraints must be GaussianConstraint records")
    probabilities = tuple(
        gaussian_feasibility_probability(item) for item in constraint_values
    )
    nominal = all(item.mean <= 0.0 for item in constraint_values)
    robust_margins = tuple(
        finite(
            "robust constraint margin",
            item.mean + robust_sigma * item.standard_deviation,
        )
        for item in constraint_values
    )
    robust = all(margin <= 0.0 for margin in robust_margins)
    chance = all(value >= chance_threshold for value in probabilities)
    nondominated = not any(
        _dominates_validated(other, objective_values, direction_values, 1.0e-12)
        for other in comparisons
    )
    return PromotionAssessment(
        nominal,
        robust,
        chance,
        probabilities,
        nondominated,
        nominal and robust and chance and nondominated,
        True,
    )
