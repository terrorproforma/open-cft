"""Surrogate-independent gates for safe promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, sqrt
from numbers import Real
from typing import Mapping, Sequence

from .domain import Design, Fidelity


@dataclass(frozen=True)
class ErrorBudget:
    """Keep learnable emulator error distinct from physical model discrepancy."""

    emulator_standard_error: tuple[float, ...]
    model_discrepancy_standard_error: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            emulator = tuple(self.emulator_standard_error)
            discrepancy = tuple(self.model_discrepancy_standard_error)
        except TypeError as exc:
            raise ValueError("error components must be iterable") from exc
        object.__setattr__(self, "emulator_standard_error", emulator)
        object.__setattr__(self, "model_discrepancy_standard_error", discrepancy)
        if not emulator:
            raise ValueError("error budget cannot be empty")
        if len(self.emulator_standard_error) != len(self.model_discrepancy_standard_error):
            raise ValueError("error components must have equal dimension")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            or value < 0.0
            for value in (
                self.emulator_standard_error
                + self.model_discrepancy_standard_error
            )
        ):
            raise ValueError("error components must be finite and non-negative")
        object.__setattr__(
            self,
            "emulator_standard_error",
            tuple(float(value) for value in emulator),
        )
        object.__setattr__(
            self,
            "model_discrepancy_standard_error",
            tuple(float(value) for value in discrepancy),
        )
        if any(not isfinite(value) for value in self.combined_standard_error):
            raise ValueError("combined error must remain finite")

    @property
    def combined_standard_error(self) -> tuple[float, ...]:
        return tuple(
            hypot(emulator, discrepancy)
            for emulator, discrepancy in zip(
                self.emulator_standard_error,
                self.model_discrepancy_standard_error,
                strict=True,
            )
        )


@dataclass(frozen=True)
class GuardrailPolicy:
    maximum_normalized_distance: float = 0.35
    maximum_relative_uncertainty: float = 0.20
    highest_fidelity: Fidelity = Fidelity.F3

    def __post_init__(self) -> None:
        for name in (
            "maximum_normalized_distance",
            "maximum_relative_uncertainty",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        try:
            object.__setattr__(self, "highest_fidelity", Fidelity(self.highest_fidelity))
        except ValueError as exc:
            raise ValueError("highest fidelity is invalid") from exc


@dataclass(frozen=True)
class GuardrailDecision:
    accepted_for_promotion: bool
    normalized_distance: float
    uncertainty_ratios: tuple[float, ...]
    invariant_failures: tuple[str, ...]
    reasons: tuple[str, ...]
    requires_high_fidelity_reevaluation: bool

    def __post_init__(self) -> None:
        if (
            not isfinite(self.normalized_distance)
            or self.normalized_distance < 0.0
            or any(
                not isfinite(value) or value < 0.0
                for value in self.uncertainty_ratios
            )
        ):
            raise ValueError("guardrail decision scalars must be finite and non-negative")
        object.__setattr__(
            self, "uncertainty_ratios", tuple(self.uncertainty_ratios)
        )
        object.__setattr__(
            self, "invariant_failures", tuple(self.invariant_failures)
        )
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not isinstance(self.accepted_for_promotion, bool) or not isinstance(
            self.requires_high_fidelity_reevaluation, bool
        ):
            raise ValueError("guardrail decision flags must be boolean")


def nearest_normalized_distance(candidate: Design, training_designs: Sequence[Design]) -> float:
    if not training_designs:
        raise ValueError("at least one training design is required")
    target = candidate.normalized()
    distances: list[float] = []
    for design in training_designs:
        if design.variables != candidate.variables:
            raise ValueError("training and candidate design spaces differ")
        distances.append(
            sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        target, design.normalized(), strict=True
                    )
                )
            )
            / sqrt(len(target))
        )
    return min(distances)


def evaluate_guardrails(
    candidate: Design,
    training_designs: Sequence[Design],
    error_budget: ErrorBudget,
    objective_scales: Sequence[float],
    invariant_flags: Mapping[str, bool],
    current_fidelity: Fidelity,
    policy: GuardrailPolicy = GuardrailPolicy(),
) -> GuardrailDecision:
    if len(objective_scales) != len(error_budget.combined_standard_error):
        raise ValueError("objective scales and error budget must have equal dimension")
    if any(
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not isfinite(float(scale))
        or scale <= 0.0
        for scale in objective_scales
    ):
        raise ValueError("objective scales must be finite and positive")
    if any(
        not isinstance(name, str) or not name or not isinstance(passed, bool)
        for name, passed in invariant_flags.items()
    ):
        raise ValueError("invariant flags require non-empty names and boolean values")
    try:
        current_fidelity = Fidelity(current_fidelity)
    except ValueError as exc:
        raise ValueError("current fidelity is invalid") from exc
    distance = nearest_normalized_distance(candidate, training_designs)
    ratios = tuple(
        uncertainty / float(scale)
        for uncertainty, scale in zip(
            error_budget.combined_standard_error, objective_scales, strict=True
        )
    )
    invariant_failures = tuple(
        sorted(name for name, passed in invariant_flags.items() if not passed)
    )
    reasons: list[str] = []
    if distance > policy.maximum_normalized_distance:
        reasons.append("out-of-domain distance exceeds policy")
    if any(ratio > policy.maximum_relative_uncertainty for ratio in ratios):
        reasons.append("predictive uncertainty exceeds policy")
    if invariant_failures:
        reasons.append("conservation or invariant checks failed")
    requires_high = current_fidelity is not policy.highest_fidelity
    if requires_high:
        reasons.append("Pareto promotion requires highest-available fidelity reevaluation")
    return GuardrailDecision(
        accepted_for_promotion=not reasons,
        normalized_distance=distance,
        uncertainty_ratios=ratios,
        invariant_failures=invariant_failures,
        reasons=tuple(reasons),
        requires_high_fidelity_reevaluation=requires_high,
    )
