from cft_revival.optimization.domain import (
    ConstraintSense,
    ConstraintValue,
    ContinuousConstraint,
    Design,
    EvaluationRequest,
    EvaluationStatus,
    Fidelity,
    Objective,
    ObjectiveDirection,
    ObjectiveValue,
    Observation,
    Provenance,
    Variable,
)
from cft_revival.optimization.pareto import (
    assess_feasibility,
    nondominated,
    nondominated_ranks,
    objective_vector,
    promotion_metadata,
)


VARIABLES = (Variable("x", 0.0, 1.0, "1"),)
OBJECTIVES = (
    Objective("thrust", ObjectiveDirection.MAXIMIZE, "N"),
    Objective("power", ObjectiveDirection.MINIMIZE, "W"),
    Objective("efficiency", ObjectiveDirection.MAXIMIZE, "1"),
    Objective("erosion", ObjectiveDirection.MINIMIZE, "mm/h"),
)
CONSTRAINTS = (
    ContinuousConstraint("temperature", ConstraintSense.LESS_THAN_OR_EQUAL, 10.0, "K"),
)
PROVENANCE = Provenance("test", "revision", "inputs")


def observation(x: float, values: tuple[float, ...], temperature: float) -> Observation:
    return Observation(
        EvaluationRequest(
            Design((x,), VARIABLES),
            Fidelity.F1,
            0,
            1,
            OBJECTIVES,
            CONSTRAINTS,
        ),
        EvaluationStatus.SUCCESS,
        tuple(
            ObjectiveValue(definition.name, value, definition.units)
            for definition, value in zip(OBJECTIVES, values, strict=True)
        ),
        (ConstraintValue("temperature", temperature, "K"),),
        PROVENANCE,
        1.0,
        0.1,
    )


def test_mixed_direction_hand_checked_front_and_ranks() -> None:
    a = observation(0.1, (10.0, 5.0, 0.5, 3.0), 9.0)
    b = observation(0.2, (8.0, 3.0, 0.7, 2.0), 9.0)
    c = observation(0.3, (7.0, 6.0, 0.4, 4.0), 9.0)
    infeasible = observation(0.4, (100.0, 1.0, 1.0, 0.0), 11.0)
    front = nondominated((infeasible, c, b, a), OBJECTIVES, CONSTRAINTS)
    assert {item.observation_id for item in front} == {a.observation_id, b.observation_id}
    ranks = nondominated_ranks((a, b, c, infeasible), OBJECTIVES, CONSTRAINTS)
    assert ranks[a.observation_id] == ranks[b.observation_id] == 0
    assert ranks[c.observation_id] == 1
    assert ranks[infeasible.observation_id] == 2
    assert objective_vector(a, OBJECTIVES) == (10.0, -5.0, 0.5, -3.0)


def test_robust_and_chance_feasibility_are_explicit() -> None:
    metadata = assess_feasibility(
        (ConstraintValue("temperature", 9.9, "K", 0.1),),
        CONSTRAINTS,
        robust_sigma=2.0,
        chance_threshold=0.8,
    )
    assert metadata.feasible
    assert metadata.chance_feasible
    assert not metadata.robust_feasible
    assert 0.8 < metadata.minimum_constraint_feasibility_probability < 0.9
    assert "no independence assumption" in metadata.probability_policy


def test_pareto_promotion_requires_robust_chance_feasibility() -> None:
    candidate = observation(0.1, (10.0, 5.0, 0.5, 3.0), 9.0)
    metadata = promotion_metadata((candidate,), OBJECTIVES, CONSTRAINTS)
    assert metadata[0].pareto_rank == 0
    assert metadata[0].requires_high_fidelity_validation


def test_scale_aware_tolerance_ignores_roundoff_but_not_real_improvement() -> None:
    baseline = observation(0.1, (10.0, 5.0, 0.5, 3.0), 9.0)
    roundoff = observation(0.2, (10.0 + 1e-15, 5.0, 0.5, 3.0), 9.0)
    assert len(nondominated((baseline, roundoff), OBJECTIVES, CONSTRAINTS)) == 2
    meaningful = observation(0.3, (10.0 + 1e-6, 5.0, 0.5, 3.0), 9.0)
    front = nondominated((baseline, meaningful), OBJECTIVES, CONSTRAINTS)
    assert front == (meaningful,)


def test_chance_policy_requires_each_margin_without_independence_assumption() -> None:
    definitions = (
        ContinuousConstraint(
            "a",
            ConstraintSense.LESS_THAN_OR_EQUAL,
            0.0,
            "Pa",
            violation_scale=100.0,
        ),
        ContinuousConstraint(
            "b",
            ConstraintSense.LESS_THAN_OR_EQUAL,
            0.0,
            "K",
            violation_scale=10.0,
        ),
    )
    values = (
        ConstraintValue("a", -1.7507, "Pa", 1.0),
        ConstraintValue("b", -0.17507, "K", 0.1),
    )
    metadata = assess_feasibility(values, definitions, chance_threshold=0.95)
    assert metadata.chance_feasible
    assert metadata.minimum_constraint_feasibility_probability > 0.95
    assert metadata.normalized_total_violation == 0.0


def test_highest_fidelity_candidate_does_not_request_redundant_validation() -> None:
    source = observation(0.1, (10.0, 5.0, 0.5, 3.0), 9.0)
    request = EvaluationRequest(
        source.request.design,
        Fidelity.F3,
        0,
        1,
        OBJECTIVES,
        CONSTRAINTS,
    )
    high = Observation(
        request,
        EvaluationStatus.SUCCESS,
        source.objectives,
        source.constraints,
        PROVENANCE,
        1.0,
        1.0,
    )
    metadata = promotion_metadata((high,), OBJECTIVES, CONSTRAINTS)
    assert metadata[0].eligible_for_promotion
    assert not metadata[0].requires_high_fidelity_validation
