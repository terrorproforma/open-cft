import inspect
from math import sqrt

import pytest

from cft_revival.optimization import botorch_adapter
from cft_revival.optimization.domain import (
    ConstraintSense,
    ContinuousConstraint,
    Design,
    Fidelity,
    ObjectiveDirection,
    ObjectiveSpec,
    Variable,
)
from cft_revival.optimization.guardrails import (
    ErrorBudget,
    GuardrailPolicy,
    evaluate_guardrails,
)


VARIABLES = (Variable("x", 0.0, 1.0, "1"), Variable("y", 0.0, 1.0, "1"))
TRAINING = (Design((0.5, 0.5), VARIABLES),)


def test_error_sources_remain_separate_and_combine_in_quadrature() -> None:
    budget = ErrorBudget((3.0, 0.0), (4.0, 2.0))
    assert budget.emulator_standard_error == (3.0, 0.0)
    assert budget.model_discrepancy_standard_error == (4.0, 2.0)
    assert budget.combined_standard_error == (5.0, 2.0)


def test_guardrails_require_high_fidelity_even_for_safe_surrogate_candidate() -> None:
    budget = ErrorBudget((0.01,), (0.01,))
    low = evaluate_guardrails(
        TRAINING[0],
        TRAINING,
        budget,
        (1.0,),
        {"mass-conserved": True, "energy-bounded": True},
        Fidelity.F2,
    )
    assert low.requires_high_fidelity_reevaluation
    assert not low.accepted_for_promotion
    high = evaluate_guardrails(
        TRAINING[0],
        TRAINING,
        budget,
        (1.0,),
        {"mass-conserved": True},
        Fidelity.F3,
    )
    assert high.accepted_for_promotion
    assert high.normalized_distance == 0.0


def test_ood_uncertainty_and_invariant_failures_are_reported() -> None:
    decision = evaluate_guardrails(
        Design((1.0, 1.0), VARIABLES),
        TRAINING,
        ErrorBudget((0.3,), (0.4,)),
        (1.0,),
        {"mass-conserved": False},
        Fidelity.F3,
        GuardrailPolicy(maximum_normalized_distance=0.2, maximum_relative_uncertainty=0.2),
    )
    assert decision.normalized_distance == pytest.approx(0.5)
    assert decision.uncertainty_ratios == pytest.approx((sqrt(0.3**2 + 0.4**2),))
    assert decision.invariant_failures == ("mass-conserved",)
    assert not decision.accepted_for_promotion


def test_optional_dependency_failure_is_clean_and_actionable(monkeypatch) -> None:
    monkeypatch.setattr(botorch_adapter, "find_spec", lambda _package: None)
    assert not botorch_adapter.dependencies_available()
    with pytest.raises(botorch_adapter.OptionalDependencyError, match="torch.*botorch.*gpytorch"):
        botorch_adapter.load_api()


def test_adapter_plan_names_documented_log_acquisitions() -> None:
    plan = botorch_adapter.AcquisitionPlan()
    assert plan.primary == "qLogNEHVI"
    assert plan.fallback == "qLogNParEGO"
    assert plan.constraint_convention == "strictly-negative-is-feasible"
    assert plan.qlognparego_batch_optimizer == "optimize_acqf_list"
    assert botorch_adapter.default_model_plan(("a", "b")).observed_heteroskedastic_noise


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_error_and_guardrail_scalars_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ErrorBudget((bad,), (0.0,))
    with pytest.raises(ValueError, match="finite"):
        GuardrailPolicy(maximum_relative_uncertainty=bad)
    with pytest.raises(ValueError, match="finite"):
        evaluate_guardrails(
            TRAINING[0],
            TRAINING,
            ErrorBudget((0.01,), (0.01,)),
            (bad,),
            {"mass-conserved": True},
            Fidelity.F3,
        )


def test_guardrails_fail_closed_without_training_domain() -> None:
    with pytest.raises(ValueError, match="training design"):
        evaluate_guardrails(
            TRAINING[0],
            (),
            ErrorBudget((0.01,), (0.01,)),
            (1.0,),
            {"mass-conserved": True},
            Fidelity.F3,
        )


def test_error_budget_copies_mutable_inputs() -> None:
    emulator = [0.1]
    discrepancy = [0.2]
    budget = ErrorBudget(emulator, discrepancy)
    emulator[0] = float("nan")
    discrepancy[0] = float("inf")
    assert budget.emulator_standard_error == (0.1,)
    assert budget.model_discrepancy_standard_error == (0.2,)


def test_mixed_direction_and_strict_constraint_transforms_are_dependency_free() -> None:
    objectives = (
        ObjectiveSpec("thrust", ObjectiveDirection.MAXIMIZE, "N"),
        ObjectiveSpec("efficiency", ObjectiveDirection.MAXIMIZE, "1"),
        ObjectiveSpec("isp", ObjectiveDirection.MAXIMIZE, "s"),
        ObjectiveSpec("power", ObjectiveDirection.MINIMIZE, "W"),
    )
    assert botorch_adapter.objective_direction_signs(objectives) == (
        1.0,
        1.0,
        1.0,
        -1.0,
    )
    assert botorch_adapter.transform_objective_values(
        (1.0, 0.5, 1000.0, 200.0),
        objectives,
    ) == (1.0, 0.5, 1000.0, -200.0)
    limit = ContinuousConstraint(
        "temperature",
        ConstraintSense.LESS_THAN_OR_EQUAL,
        100.0,
        "K",
        violation_scale=10.0,
    )
    layout = botorch_adapter.ModelOutputLayout(objectives, (limit,))
    assert botorch_adapter.botorch_constraint_value(100.0, limit) < 0.0
    assert botorch_adapter.botorch_constraint_value(101.0, limit) > 0.0
    transformed = botorch_adapter.transform_model_output_values(
        (1.0, 0.5, 1000.0, 200.0, 101.0),
        layout,
    )
    assert transformed[:4] == (1.0, 0.5, 1000.0, -200.0)
    assert transformed[4] == pytest.approx(0.1 - 1e-12)
    assert layout.objective_indices == (0, 1, 2, 3)
    assert layout.constraint_indices == (4,)
    assert set(layout.objective_indices).isdisjoint(layout.constraint_indices)

    class Samples:
        def __getitem__(self, key):
            assert key[0] is Ellipsis
            return ("objective", "objective", "objective", "objective", "constraint")[
                key[1]
            ]

    selectors = botorch_adapter.constraint_output_callables(layout)
    assert tuple(selector(Samples()) for selector in selectors) == ("constraint",)


def test_adapter_source_enforces_transforms_and_implements_sequential_batch() -> None:
    model_source = inspect.getsource(botorch_adapter.build_exact_output_models)
    nehvi_source = inspect.getsource(botorch_adapter.build_qlognehvi)
    parego_source = inspect.getsource(botorch_adapter.build_qlognparego)
    batch_source = inspect.getsource(botorch_adapter.optimize_qlognparego_batch)
    assert "_transform_tensor_model_outputs" in model_source
    assert "_transform_tensor_model_variances" in model_source
    assert "IdentityMCMultiOutputObjective" in nehvi_source
    assert "constraint_output_callables" in nehvi_source
    assert "IdentityMCMultiOutputObjective" in parego_source
    assert "constraint_output_callables" in parego_source
    assert "optimize_acqf_list" in batch_source
    layout = botorch_adapter.ModelOutputLayout(
        (ObjectiveSpec("score", ObjectiveDirection.MAXIMIZE, "1"),)
    )
    with pytest.raises(ValueError, match="mixed-direction transform"):
        botorch_adapter.build_qlognparego(
            object(),
            object(),
            layout,
            model_outputs_are_direction_transformed=False,
        )


def test_constraint_variance_uses_scale_squared_and_objectives_are_unchanged() -> None:
    layout = botorch_adapter.ModelOutputLayout(
        (
            ObjectiveSpec("thrust", ObjectiveDirection.MAXIMIZE, "N"),
            ObjectiveSpec("power", ObjectiveDirection.MINIMIZE, "W"),
        ),
        (
            ContinuousConstraint(
                "temperature",
                ConstraintSense.LESS_THAN_OR_EQUAL,
                100.0,
                "K",
                violation_scale=10.0,
            ),
            ContinuousConstraint(
                "clearance",
                ConstraintSense.GREATER_THAN_OR_EQUAL,
                0.1,
                "mm",
                violation_scale=0.1,
            ),
        ),
    )
    transformed = botorch_adapter.transform_model_output_variances(
        (4.0, 9.0, 100.0, 0.04),
        layout,
    )
    assert transformed[:2] == (4.0, 9.0)
    assert transformed[2] == pytest.approx(1.0)
    assert transformed[3] == pytest.approx(4.0)


def test_multitask_known_noise_is_explicitly_rejected_before_optional_imports() -> None:
    layout = botorch_adapter.ModelOutputLayout(
        (ObjectiveSpec("score", ObjectiveDirection.MAXIMIZE, "1"),)
    )
    with pytest.raises(
        botorch_adapter.UnsupportedTaskNoiseError,
        match="does not support differing known noise",
    ):
        botorch_adapter.build_source_task_models(
            object(),
            object(),
            layout,
            train_yvar=object(),
        )
    source = inspect.getsource(botorch_adapter.build_source_task_models)
    assert "StratifiedStandardize" in source
    assert "train_Yvar=None" in source
