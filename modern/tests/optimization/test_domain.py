from dataclasses import FrozenInstanceError
from random import Random

import pytest

from cft_revival.optimization.domain import (
    ConstraintSense,
    ConstraintValue,
    ContinuousConstraint,
    Design,
    DomainError,
    EvaluationRequest,
    EvaluationStatus,
    Fidelity,
    ObjectiveDirection,
    ObjectiveSpec,
    ObjectiveValue,
    Observation,
    Provenance,
    SolverFailure,
    Variable,
)


VARIABLES = tuple(Variable(f"x{index}", 0.0, 1.0, "1") for index in range(8))
DESIGN = Design((0.5,) * 8, VARIABLES)
PROVENANCE = Provenance("solver-1", "abc123", "input-hash")
THRUST = ObjectiveSpec("thrust", ObjectiveDirection.MAXIMIZE, "N")
TEMPERATURE = ContinuousConstraint(
    "temperature",
    ConstraintSense.LESS_THAN_OR_EQUAL,
    1000.0,
    "K",
    violation_scale=100.0,
)


def request(
    fidelity: Fidelity,
    seed: int,
    replicates: int = 1,
) -> EvaluationRequest:
    return EvaluationRequest(
        DESIGN,
        fidelity,
        seed,
        replicates,
        (THRUST,),
        (TEMPERATURE,),
        (("solver_config", "v1"),),
    )


def test_design_and_evaluation_hashes_are_stable_and_seed_specific() -> None:
    clone = Design(tuple([0.5] * 8), VARIABLES, provenance="different note")
    assert clone.design_id == DESIGN.design_id
    clone_request = EvaluationRequest(
        clone, Fidelity.F1, 7, 1, (THRUST,), (TEMPERATURE,)
    )
    original_request = EvaluationRequest(
        DESIGN, Fidelity.F1, 7, 1, (THRUST,), (TEMPERATURE,)
    )
    assert original_request.evaluation_key == clone_request.evaluation_key
    assert original_request.evaluation_key != EvaluationRequest(
        DESIGN, Fidelity.F1, 8, 1, (THRUST,), (TEMPERATURE,)
    ).evaluation_key
    assert original_request.evaluation_key != EvaluationRequest(
        DESIGN, Fidelity.F1, 7, 2, (THRUST,), (TEMPERATURE,)
    ).evaluation_key
    assert original_request.evaluation_key != EvaluationRequest(
        DESIGN,
        Fidelity.F1,
        7,
        1,
        (THRUST,),
        (TEMPERATURE,),
        (("mesh", "fine"),),
    ).evaluation_key


def test_failed_observation_cannot_encode_fake_outcomes() -> None:
    evaluation = request(Fidelity.F2, 4)
    failure = SolverFailure("DIVERGED", "nonlinear solve diverged", True, "plasma")
    observation = Observation(
        evaluation,
        EvaluationStatus.FAILURE,
        (),
        (),
        PROVENANCE,
        2.0,
        0.1,
        failure,
    )
    assert observation.failure is failure
    with pytest.raises(DomainError, match="no fake outcomes"):
        Observation(
            evaluation,
            EvaluationStatus.FAILURE,
            (ObjectiveValue("thrust", -1.0, "N"),),
            (ConstraintValue("solver-status", 1.0, "category"),),
            PROVENANCE,
            2.0,
            0.1,
            failure,
        )


def test_observations_are_immutable_and_round_trip() -> None:
    original = Observation(
        request(Fidelity.F0, 0, 3),
        EvaluationStatus.SUCCESS,
        (ObjectiveValue("thrust", 1.2, "N", 0.1, 3),),
        (ConstraintValue("temperature", 900.0, "K", 2.0),),
        PROVENANCE,
        0.5,
        1 / 256,
    )
    restored = Observation.from_dict(original.to_dict())
    assert restored == original
    assert restored.observation_id == original.observation_id
    with pytest.raises(FrozenInstanceError):
        original.charged_cost = 2.0  # type: ignore[misc]


def test_nested_input_lists_are_copied_and_schema_affects_identity() -> None:
    values = [0.25] * 8
    variables = list(VARIABLES)
    design = Design(values, variables)
    objectives = [THRUST]
    constraints = [TEMPERATURE]
    context = [["mesh", "fine"]]
    evaluation = EvaluationRequest(
        design,
        Fidelity.F2,
        1,
        1,
        objectives,
        constraints,
        context,
    )
    design_id = design.design_id
    evaluation_key = evaluation.evaluation_key
    values[0] = 0.9
    variables[0] = Variable("other", 0.0, 1.0, "1")
    objectives.clear()
    constraints.clear()
    context[0][1] = "changed"
    assert design.values == (0.25,) * 8
    assert design.design_id == design_id
    assert evaluation.evaluation_key == evaluation_key
    changed_bounds = tuple(
        (Variable("x0", -1.0, 1.0, "1"), *VARIABLES[1:])
    )
    assert Design((0.25,) * 8, changed_bounds).design_id != design_id


def test_duplicate_variable_names_are_rejected_before_identity() -> None:
    with pytest.raises(DomainError, match="variable names must be unique"):
        Design(
            (0.1, 0.2),
            (
                Variable("radius", 0.0, 1.0, "m"),
                Variable("radius", 0.0, 2.0, "m"),
            ),
        )


def test_status_replicate_and_partial_schema_invariants_fail_typed() -> None:
    evaluation = request(Fidelity.F1, 2, 2)
    with pytest.raises(DomainError, match="unknown evaluation status"):
        Observation(
            evaluation,
            "mystery",  # type: ignore[arg-type]
            (),
            (),
            PROVENANCE,
            1.0,
            0.1,
        )
    with pytest.raises(DomainError, match="reported replicates"):
        Observation(
            evaluation,
            EvaluationStatus.SUCCESS,
            (ObjectiveValue("thrust", 1.0, "N", replicates=1),),
            (ConstraintValue("temperature", 900.0, "K"),),
            PROVENANCE,
            1.0,
            0.1,
        )
    with pytest.raises(DomainError, match="partial"):
        Observation(
            evaluation,
            EvaluationStatus.SUCCESS,
            (ObjectiveValue("other", 1.0, "N", replicates=2),),
            (ConstraintValue("temperature", 900.0, "K"),),
            PROVENANCE,
            1.0,
            0.1,
        )
    with pytest.raises(DomainError, match="no failure"):
        Observation(
            evaluation,
            EvaluationStatus.SUCCESS,
            (ObjectiveValue("thrust", 1.0, "N", replicates=2),),
            (ConstraintValue("temperature", 900.0, "K"),),
            PROVENANCE,
            1.0,
            0.1,
            SolverFailure("BAD", "unexpected", False, "solver"),
        )
    with pytest.raises(DomainError, match="require failure metadata"):
        Observation(
            evaluation,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            1.0,
            0.1,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_results_and_costs_are_rejected(bad: float) -> None:
    with pytest.raises(DomainError):
        ObjectiveValue("thrust", 1.0, "N", standard_error=bad)
    with pytest.raises(DomainError):
        ConstraintValue("temperature", bad, "K")
    with pytest.raises(DomainError):
        Observation(
            request(Fidelity.F0, 0),
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            1.0,
            bad,
            SolverFailure("BAD", "bad", False, "solver"),
        )


def test_randomized_identity_property_has_no_aliases() -> None:
    random = Random(20260901)
    designs = tuple(
        Design(tuple(random.random() for _ in range(8)), VARIABLES)
        for _ in range(200)
    )
    assert len({design.design_id for design in designs}) == len(designs)
    requests = tuple(
        EvaluationRequest(
            design,
            Fidelity.F0,
            index,
            1 + index % 3,
            (THRUST,),
            (TEMPERATURE,),
            (("property_case", str(index)),),
        )
        for index, design in enumerate(designs)
    )
    assert len({item.evaluation_key for item in requests}) == len(requests)
