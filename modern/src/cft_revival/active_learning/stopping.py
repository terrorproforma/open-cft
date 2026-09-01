"""Stopping gates aligned with optimization campaign specification v1.4."""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1, log
from sys import float_info
from typing import Sequence

from .contracts import ActiveLearningError, finite, integer


@dataclass(frozen=True)
class StoppingPolicyV14:
    mandatory_highest_successes: int = 12
    minimum_highest_success_fraction: float = 0.03
    verified_hypervolume_maximum_relative_improvement: float = 0.005
    verified_hypervolume_window_iterations: int = 5
    hard_equivalent_highest_cost_ceiling: float = 19.0
    schema_version: str = "1.4"

    def __post_init__(self) -> None:
        if self.schema_version != "1.4":
            raise ActiveLearningError("only stopping policy schema v1.4 is supported")
        object.__setattr__(
            self,
            "mandatory_highest_successes",
            integer(
                "mandatory highest-fidelity successes",
                self.mandatory_highest_successes,
            ),
        )
        object.__setattr__(
            self,
            "verified_hypervolume_window_iterations",
            integer(
                "verified hypervolume window iterations",
                self.verified_hypervolume_window_iterations,
                minimum=1,
            ),
        )
        for name in (
            "minimum_highest_success_fraction",
            "verified_hypervolume_maximum_relative_improvement",
        ):
            value = finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ActiveLearningError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        ceiling = finite(
            "hard equivalent-highest cost ceiling",
            self.hard_equivalent_highest_cost_ceiling,
        )
        if ceiling <= 0.0:
            raise ActiveLearningError("hard cost ceiling must be positive")
        object.__setattr__(self, "hard_equivalent_highest_cost_ceiling", ceiling)


@dataclass(frozen=True)
class StoppingEvidence:
    highest_successes: int
    total_successes: int
    pending_jobs: int
    verified_hypervolume_history: tuple[float, ...]
    surrogate_calibration_checked: bool
    promoted_candidates_pass_guardrails: bool
    acquisition_converged: bool
    iteration_acquisition_policy_satisfied: bool
    equivalent_highest_cost_spent: float
    validation_exhausted: bool = False

    def __post_init__(self) -> None:
        for name in ("highest_successes", "total_successes", "pending_jobs"):
            object.__setattr__(
                self,
                name,
                integer(name, getattr(self, name)),
            )
        if self.highest_successes > self.total_successes:
            raise ActiveLearningError("highest successes cannot exceed all successes")
        try:
            history = tuple(
                finite("verified hypervolume", value)
                for value in self.verified_hypervolume_history
            )
        except TypeError as error:
            raise ActiveLearningError("verified hypervolume history must be iterable") from error
        if any(value < 0.0 for value in history):
            raise ActiveLearningError("verified hypervolume cannot be negative")
        object.__setattr__(self, "verified_hypervolume_history", history)
        for name in (
            "surrogate_calibration_checked",
            "promoted_candidates_pass_guardrails",
            "acquisition_converged",
            "iteration_acquisition_policy_satisfied",
            "validation_exhausted",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ActiveLearningError(f"{name} must be boolean")
        cost = finite("equivalent-highest cost spent", self.equivalent_highest_cost_spent)
        if cost < 0.0:
            raise ActiveLearningError("spent cost cannot be negative")
        object.__setattr__(self, "equivalent_highest_cost_spent", cost)


@dataclass(frozen=True)
class StoppingDecision:
    should_stop: bool
    terminal_reason: str | None
    gates: tuple[tuple[str, bool], ...]
    unmet_gates: tuple[str, ...]
    highest_success_fraction: float
    verified_hypervolume_relative_improvement: float | None


def _hypervolume_stalled(
    history: Sequence[float],
    policy: StoppingPolicyV14,
) -> tuple[bool, float | None]:
    required = policy.verified_hypervolume_window_iterations + 1
    if len(history) < required:
        return False, None
    start = history[-required]
    end = history[-1]
    if end <= start:
        relative = 0.0
    elif start == 0.0:
        relative = float_info.max
    else:
        log_ratio = log(end) - log(start)
        relative = (
            float_info.max
            if log_ratio >= log(float_info.max)
            else finite("verified hypervolume relative improvement", expm1(log_ratio))
        )
    return (
        relative
        <= policy.verified_hypervolume_maximum_relative_improvement,
        relative,
    )


def evaluate_stopping_gates(
    evidence: StoppingEvidence,
    policy: StoppingPolicyV14 = StoppingPolicyV14(),
) -> StoppingDecision:
    """Evaluate all v1.4 gates; hard cost and exhaustion are terminal overrides."""

    highest_fraction = (
        evidence.highest_successes / evidence.total_successes
        if evidence.total_successes
        else 0.0
    )
    hypervolume_stalled, relative_improvement = _hypervolume_stalled(
        evidence.verified_hypervolume_history,
        policy,
    )
    gates = (
        (
            "mandatory_f3_success_count",
            evidence.highest_successes >= policy.mandatory_highest_successes,
        ),
        (
            "minimum_f3_success_fraction",
            highest_fraction >= policy.minimum_highest_success_fraction,
        ),
        ("verified_hypervolume_relative_improvement", hypervolume_stalled),
        ("pending_jobs", evidence.pending_jobs == 0),
        ("surrogate_calibration_checked", evidence.surrogate_calibration_checked),
        (
            "promoted_candidates_pass_guardrails",
            evidence.promoted_candidates_pass_guardrails,
        ),
        ("acquisition_converged", evidence.acquisition_converged),
        (
            "iteration_acquisition_policy_satisfied",
            evidence.iteration_acquisition_policy_satisfied,
        ),
    )
    cost_ceiling = (
        evidence.equivalent_highest_cost_spent
        >= policy.hard_equivalent_highest_cost_ceiling
    )
    terminal_reason = (
        "hard_equivalent_f3_cost_ceiling"
        if cost_ceiling
        else "validation_exhausted"
        if evidence.validation_exhausted
        else None
    )
    unmet = tuple(name for name, passed in gates if not passed)
    return StoppingDecision(
        not unmet or terminal_reason is not None,
        terminal_reason,
        gates,
        unmet,
        highest_fraction,
        relative_improvement,
    )
