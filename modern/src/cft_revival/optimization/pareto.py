"""Deterministic constrained Pareto utilities with mixed objective directions."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, isfinite, sqrt
from typing import Iterable, Sequence

from .domain import (
    ConstraintValue,
    ContinuousConstraint,
    EvaluationStatus,
    Fidelity,
    Objective,
    Observation,
)


@dataclass(frozen=True)
class FeasibilityMetadata:
    feasible: bool
    robust_feasible: bool
    chance_feasible: bool
    minimum_constraint_feasibility_probability: float
    normalized_total_violation: float
    robust_sigma: float
    chance_threshold: float
    probability_policy: str


@dataclass(frozen=True)
class PromotionMetadata:
    observation_id: str
    pareto_rank: int
    feasibility: FeasibilityMetadata
    eligible_for_promotion: bool
    requires_high_fidelity_validation: bool
    reason: str


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def assess_feasibility(
    values: Sequence[ConstraintValue],
    definitions: Sequence[ContinuousConstraint],
    *,
    robust_sigma: float = 2.0,
    chance_threshold: float = 0.95,
) -> FeasibilityMetadata:
    if (
        not isfinite(robust_sigma)
        or not isfinite(chance_threshold)
        or robust_sigma < 0.0
        or not 0.0 <= chance_threshold <= 1.0
    ):
        raise ValueError("invalid robust/chance feasibility settings")
    by_name = {item.name: item for item in values}
    residuals: list[float] = []
    robust_residuals: list[float] = []
    marginal_probabilities: list[float] = []
    for definition in definitions:
        try:
            value = by_name[definition.name]
        except KeyError as exc:
            raise ValueError(f"missing constraint value {definition.name!r}") from exc
        if value.units != definition.units:
            raise ValueError(f"unit mismatch for constraint {definition.name!r}")
        residual = definition.normalized_residual(value.value)
        normalized_standard_error = value.standard_error / definition.violation_scale
        residuals.append(residual)
        robust_residuals.append(
            residual + robust_sigma * normalized_standard_error
        )
        if normalized_standard_error == 0.0:
            marginal_probabilities.append(float(residual <= 0.0))
        else:
            marginal_probabilities.append(
                _normal_cdf(-residual / normalized_standard_error)
            )
    minimum_probability = min(marginal_probabilities, default=1.0)
    return FeasibilityMetadata(
        feasible=all(residual <= 0.0 for residual in residuals),
        robust_feasible=all(residual <= 0.0 for residual in robust_residuals),
        chance_feasible=all(
            probability >= chance_threshold
            for probability in marginal_probabilities
        ),
        minimum_constraint_feasibility_probability=minimum_probability,
        normalized_total_violation=sum(
            max(0.0, residual) for residual in residuals
        ),
        robust_sigma=robust_sigma,
        chance_threshold=chance_threshold,
        probability_policy=(
            "each marginal constraint probability must meet threshold; "
            "no independence assumption"
        ),
    )


def objective_vector(
    observation: Observation,
    objectives: Sequence[Objective],
) -> tuple[float, ...]:
    if observation.status is not EvaluationStatus.SUCCESS:
        raise ValueError("failed observations have no objective vector")
    by_name = {item.name: item for item in observation.objectives}
    vector: list[float] = []
    for objective in objectives:
        try:
            measured = by_name[objective.name]
        except KeyError as exc:
            raise ValueError(f"missing objective {objective.name!r}") from exc
        if measured.units != objective.units:
            raise ValueError(
                f"unit mismatch for {objective.name}: {measured.units!r} != {objective.units!r}"
            )
        sign = 1.0 if objective.direction.value == "maximize" else -1.0
        vector.append(sign * measured.value)
    if not all(isfinite(value) for value in vector):
        raise ValueError("objective vector must be finite")
    return tuple(vector)


def dominates(
    left: Observation,
    right: Observation,
    objectives: Sequence[Objective],
    constraints: Sequence[ContinuousConstraint],
) -> bool:
    """Deb-style constrained dominance, with all objectives transformed to maximize."""
    if left.status is not EvaluationStatus.SUCCESS:
        return False
    if right.status is not EvaluationStatus.SUCCESS:
        return True
    left_feasibility = assess_feasibility(left.constraints, constraints)
    right_feasibility = assess_feasibility(right.constraints, constraints)
    if left_feasibility.feasible != right_feasibility.feasible:
        return left_feasibility.feasible
    if not left_feasibility.feasible:
        return (
            left_feasibility.normalized_total_violation
            < right_feasibility.normalized_total_violation - 1e-12
        )
    left_vector = objective_vector(left, objectives)
    right_vector = objective_vector(right, objectives)
    tolerances = tuple(
        max(
            objective.absolute_tolerance * objective.comparison_scale,
            objective.relative_tolerance
            * max(abs(left), abs(right), objective.comparison_scale),
        )
        for left, right, objective in zip(
            left_vector, right_vector, objectives, strict=True
        )
    )
    return all(
        left >= right - tolerance
        for left, right, tolerance in zip(
            left_vector, right_vector, tolerances, strict=True
        )
    ) and any(
        left > right + tolerance
        for left, right, tolerance in zip(
            left_vector, right_vector, tolerances, strict=True
        )
    )


def nondominated(
    observations: Iterable[Observation],
    objectives: Sequence[Objective],
    constraints: Sequence[ContinuousConstraint],
) -> tuple[Observation, ...]:
    candidates = sorted(observations, key=lambda item: item.observation_id)
    return tuple(
        candidate
        for candidate in candidates
        if not any(
            dominates(other, candidate, objectives, constraints)
            for other in candidates
            if other.observation_id != candidate.observation_id
        )
    )


def nondominated_ranks(
    observations: Iterable[Observation],
    objectives: Sequence[Objective],
    constraints: Sequence[ContinuousConstraint],
) -> dict[str, int]:
    """Return deterministic zero-based fronts keyed by immutable observation ID."""
    remaining = {item.observation_id: item for item in observations}
    ranks: dict[str, int] = {}
    rank = 0
    while remaining:
        front = nondominated(remaining.values(), objectives, constraints)
        if not front:
            raise RuntimeError("dominance relation failed to produce a front")
        for item in front:
            ranks[item.observation_id] = rank
            del remaining[item.observation_id]
        rank += 1
    return ranks


def promotion_metadata(
    observations: Iterable[Observation],
    objectives: Sequence[Objective],
    constraints: Sequence[ContinuousConstraint],
    *,
    robust_sigma: float = 2.0,
    chance_threshold: float = 0.95,
    highest_available_fidelity: Fidelity = Fidelity.F3,
) -> tuple[PromotionMetadata, ...]:
    successful = tuple(item for item in observations if item.status is EvaluationStatus.SUCCESS)
    ranks = nondominated_ranks(successful, objectives, constraints)
    result: list[PromotionMetadata] = []
    for item in sorted(successful, key=lambda value: value.observation_id):
        feasibility = assess_feasibility(
            item.constraints,
            constraints,
            robust_sigma=robust_sigma,
            chance_threshold=chance_threshold,
        )
        rank = ranks[item.observation_id]
        eligible = (
            rank == 0
            and feasibility.robust_feasible
            and feasibility.chance_feasible
        )
        requires_validation = (
            eligible
            and item.request.fidelity is not highest_available_fidelity
        )
        reason = (
            "eligible candidate requires highest-fidelity validation"
            if requires_validation
            else "eligible candidate is already at highest fidelity"
            if eligible
            else "candidate fails rank, robust, or chance-feasibility gate"
        )
        result.append(
            PromotionMetadata(
                observation_id=item.observation_id,
                pareto_rank=rank,
                feasibility=feasibility,
                eligible_for_promotion=eligible,
                requires_high_fidelity_validation=requires_validation,
                reason=reason,
            )
        )
    return tuple(result)
