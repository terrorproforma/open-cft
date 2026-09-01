import json
from pathlib import Path

import pytest

from cft_revival.active_learning.contracts import (
    ActiveLearningError,
    CampaignCounts,
    CampaignRecordAdapter,
    FidelitySource,
    PosteriorAdapter,
)
from cft_revival.active_learning.stopping import (
    StoppingEvidence,
    StoppingPolicyV14,
    evaluate_stopping_gates,
)
from cft_revival.active_learning.synthetic import AnalyticalPosterior


ROOT = Path(__file__).resolve().parents[2]


class RecordAdapter:
    def counts(self) -> CampaignCounts:
        return CampaignCounts((("F3", 12),), total_completed_successes=100)

    def pending_designs(self) -> tuple[tuple[str, tuple[float, ...], str], ...]:
        return ()


def complete_evidence(**overrides: object) -> StoppingEvidence:
    values: dict[str, object] = {
        "highest_successes": 12,
        "total_successes": 100,
        "pending_jobs": 0,
        "verified_hypervolume_history": (
            1.0,
            1.001,
            1.002,
            1.003,
            1.004,
            1.0049,
        ),
        "surrogate_calibration_checked": True,
        "promoted_candidates_pass_guardrails": True,
        "acquisition_converged": True,
        "iteration_acquisition_policy_satisfied": True,
        "equivalent_highest_cost_spent": 10.0,
    }
    values.update(overrides)
    return StoppingEvidence(**values)  # type: ignore[arg-type]


def test_structural_adapters_do_not_require_existing_packages() -> None:
    assert isinstance(AnalyticalPosterior(), PosteriorAdapter)
    assert isinstance(RecordAdapter(), CampaignRecordAdapter)


def test_v14_stopping_requires_every_declared_gate() -> None:
    decision = evaluate_stopping_gates(complete_evidence())
    assert decision.should_stop
    assert decision.terminal_reason is None
    assert not decision.unmet_gates
    pending = evaluate_stopping_gates(complete_evidence(pending_jobs=1))
    assert not pending.should_stop
    assert pending.unmet_gates == ("pending_jobs",)


def test_cost_ceiling_and_validation_exhaustion_are_terminal_overrides() -> None:
    incomplete = complete_evidence(
        highest_successes=1,
        total_successes=10,
        pending_jobs=2,
        equivalent_highest_cost_spent=19.0,
    )
    cost_decision = evaluate_stopping_gates(incomplete)
    assert cost_decision.should_stop
    assert cost_decision.terminal_reason == "hard_equivalent_f3_cost_ceiling"
    exhausted = evaluate_stopping_gates(
        complete_evidence(highest_successes=1, validation_exhausted=True)
    )
    assert exhausted.should_stop
    assert exhausted.terminal_reason == "validation_exhausted"


@pytest.mark.parametrize("bad", (True, 1.5, -1))
def test_stopping_and_source_integer_contracts_reject_bad_values(bad: object) -> None:
    with pytest.raises(ActiveLearningError):
        StoppingPolicyV14(mandatory_highest_successes=bad)  # type: ignore[arg-type]
    with pytest.raises(ActiveLearningError):
        StoppingPolicyV14(
            verified_hypervolume_window_iterations=bad  # type: ignore[arg-type]
        )
    with pytest.raises(ActiveLearningError):
        complete_evidence(highest_successes=bad)
    with pytest.raises(ActiveLearningError):
        FidelitySource("bad-rank", bad, 1.0)  # type: ignore[arg-type]


def test_extreme_finite_hypervolume_history_does_not_overflow() -> None:
    decision = evaluate_stopping_gates(
        complete_evidence(
            verified_hypervolume_history=(1.0e-300,) * 5 + (1.0e308,),
        )
    )
    assert not decision.should_stop
    assert decision.verified_hypervolume_relative_improvement is not None
    assert decision.verified_hypervolume_relative_improvement > 1.0


def test_active_learning_spec_matches_v14_gate_contract() -> None:
    raw = json.loads(
        (ROOT / "spec" / "active_learning" / "active-learning-v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy = StoppingPolicyV14()
    expected = {
        "mandatory_f3_success_count",
        "minimum_f3_success_fraction",
        "verified_hypervolume_relative_improvement",
        "pending_jobs",
        "surrogate_calibration_checked",
        "promoted_candidates_pass_guardrails",
        "acquisition_converged",
        "iteration_acquisition_policy_satisfied",
    }
    assert raw["independence"]["imports_existing_optimization_or_surrogate_code"] is False
    assert raw["verification"]["legacy_outputs_used_as_truth"] is False
    assert raw["verification"]["benchmark_results"] is None
    assert raw["stopping"]["aligned_optimization_schema_version"] == policy.schema_version
    assert set(raw["stopping"]["all_required_gates"]) == expected
