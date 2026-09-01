from decimal import Decimal
from fractions import Fraction
from math import inf, nan

import pytest

from cft_revival.active_learning.contracts import ActiveLearningError, GaussianConstraint
from cft_revival.active_learning.robustness import (
    NormalTolerance,
    ToleranceVariable,
    TriangularTolerance,
    UniformTolerance,
    assess_promotion,
    cvar,
    dominates,
    propagate_monte_carlo,
)
from cft_revival.active_learning.synthetic import (
    analytical_constraint_residual,
    analytical_truth,
    tolerance_response,
)


def _fraction_oracle(
    values: tuple[int, ...],
    probability: Fraction,
    tail: str,
) -> Fraction:
    ordered = sorted(Fraction(value) for value in values)
    if tail == "upper":
        ordered.reverse()
    observation_mass = Fraction(1, len(ordered))
    remaining = probability
    integral = Fraction(0)
    for value in ordered:
        accepted = min(observation_mass, remaining)
        integral += value * accepted
        remaining -= accepted
        if remaining == 0:
            break
    return integral / probability


@pytest.mark.parametrize(
    ("values", "probability"),
    (
        ((0, 1, 2, 3), Fraction(2, 5)),
        ((0, 1, 1, 3), Fraction(3, 8)),
        ((-2, -2, 4, 4), Fraction(1, 4)),
        ((0, 1, 2, 3), Fraction(1, 1)),
    ),
)
def test_empirical_cvar_matches_exact_fractional_mass_oracle(
    values: tuple[int, ...],
    probability: Fraction,
) -> None:
    for tail in ("lower", "upper"):
        expected = float(_fraction_oracle(values, probability, tail))
        assert cvar(values, float(probability), tail=tail) == pytest.approx(expected)


def test_requested_fractional_boundary_example_and_symmetry() -> None:
    values = (0.0, 1.0, 2.0, 3.0)
    assert cvar(values, 0.4, tail="lower") == pytest.approx(0.375)
    assert cvar(values, 0.4, tail="upper") == pytest.approx(2.625)


@pytest.mark.parametrize("probability", (0.0, -0.1, 1.1, nan, inf))
def test_cvar_rejects_invalid_tail_probability(probability: float) -> None:
    with pytest.raises(ActiveLearningError):
        cvar((0.0, 1.0), probability)


def test_monte_carlo_is_seeded_and_reports_quantile_and_cvar() -> None:
    tolerances = (
        ToleranceVariable(
            "manufacturing_offset",
            NormalTolerance(0.0, 0.02),
            "manufacturing",
        ),
        ToleranceVariable(
            "operating_drift",
            UniformTolerance(-0.04, 0.04),
            "operating",
        ),
        ToleranceVariable(
            "unused_triangular",
            TriangularTolerance(-1.0, 0.0, 1.0),
            "operating",
        ),
    )
    first = propagate_monte_carlo(
        tolerance_response,
        (0.5,),
        tolerances,
        draws=256,
        seed=1729,
    )
    second = propagate_monte_carlo(
        tolerance_response,
        (0.5,),
        tolerances,
        draws=256,
        seed=1729,
    )
    assert first == second
    quantiles = dict(first.quantiles)
    assert first.lower_cvar[0] <= quantiles[0.05][0] <= first.means[0]
    assert first.upper_cvar[0] >= quantiles[0.95][0] >= first.means[0]


class SeededAdapter:
    def sample(self, rng: object) -> float:
        return rng.random()  # type: ignore[attr-defined,no-any-return]


def test_custom_distribution_adapter_preserves_local_seed_determinism() -> None:
    tolerance = (
        ToleranceVariable("adapter", SeededAdapter(), "operating"),
    )

    def response(
        design: tuple[float, ...],
        perturbations: dict[str, float],
    ) -> tuple[float, ...]:
        del design
        return (perturbations["adapter"],)

    first = propagate_monte_carlo(response, (0.0,), tolerance, draws=8, seed=99)
    second = propagate_monte_carlo(response, (0.0,), tolerance, draws=8, seed=99)
    different = propagate_monte_carlo(response, (0.0,), tolerance, draws=8, seed=100)
    assert first == second
    assert first.samples != different.samples


@pytest.mark.parametrize("distribution", (object(), type("Bad", (), {"sample": 1.0})()))
def test_distribution_adapter_requires_callable_sample(distribution: object) -> None:
    with pytest.raises(ActiveLearningError):
        ToleranceVariable("bad", distribution, "operating")  # type: ignore[arg-type]


class WrongSignatureDistribution:
    def sample(self) -> float:
        return 1.0


class RaisingDistribution:
    def sample(self, rng: object) -> float:
        del rng
        raise RuntimeError("adapter failure")


class ReturnedValueDistribution:
    def __init__(self, value: object) -> None:
        self.value = value

    def sample(self, rng: object) -> object:
        del rng
        return self.value


class NumericLookingObject:
    def __float__(self) -> float:
        return 1.25


@pytest.mark.parametrize(
    "distribution",
    (
        WrongSignatureDistribution(),
        RaisingDistribution(),
        ReturnedValueDistribution((1.0,)),
        ReturnedValueDistribution([1.0]),
        ReturnedValueDistribution(None),
        ReturnedValueDistribution("1.25"),
        ReturnedValueDistribution(b"1.25"),
        ReturnedValueDistribution(bytearray(b"1.25")),
        ReturnedValueDistribution(NumericLookingObject()),
        ReturnedValueDistribution(True),
        ReturnedValueDistribution(Decimal("1.25")),
        ReturnedValueDistribution(1.25 + 0.0j),
        ReturnedValueDistribution(nan),
        ReturnedValueDistribution(inf),
    ),
)
def test_distribution_invocation_and_scalar_failures_are_domain_errors(
    distribution: object,
) -> None:
    tolerance = ToleranceVariable(
        "bad",
        distribution,  # type: ignore[arg-type]
        "operating",
    )
    with pytest.raises(ActiveLearningError):
        propagate_monte_carlo(
            lambda design, perturbations: (0.0,),
            (0.0,),
            (tolerance,),
            draws=1,
            seed=0,
        )


@pytest.mark.parametrize("value", (2, 1.25))
def test_distribution_accepts_standard_finite_real_scalars(value: object) -> None:
    tolerance = ToleranceVariable(
        "valid",
        ReturnedValueDistribution(value),
        "operating",
    )
    summary = propagate_monte_carlo(
        lambda design, perturbations: (perturbations["valid"],),
        (0.0,),
        (tolerance,),
        draws=2,
        seed=7,
    )
    assert summary.samples == ((float(value),), (float(value),))


def test_distribution_accepts_numpy_real_scalars_and_rejects_numpy_bool() -> None:
    numpy = pytest.importorskip("numpy")
    for value in (numpy.float32(1.25), numpy.float64(1.5), numpy.int64(2)):
        tolerance = ToleranceVariable(
            "numpy-real",
            ReturnedValueDistribution(value),
            "operating",
        )
        summary = propagate_monte_carlo(
            lambda design, perturbations: (perturbations["numpy-real"],),
            (0.0,),
            (tolerance,),
            draws=1,
            seed=3,
        )
        assert summary.samples == ((float(value),),)
    boolean_tolerance = ToleranceVariable(
        "numpy-bool",
        ReturnedValueDistribution(numpy.bool_(True)),
        "operating",
    )
    with pytest.raises(ActiveLearningError):
        propagate_monte_carlo(
            lambda design, perturbations: (0.0,),
            (0.0,),
            (boolean_tolerance,),
            draws=1,
            seed=3,
        )


@pytest.mark.parametrize(("draws", "seed"), ((1.5, 0), (True, 0), (1, 1.5), (1, True)))
def test_monte_carlo_rejects_noninteger_draws_and_seeds(
    draws: object,
    seed: object,
) -> None:
    with pytest.raises(ActiveLearningError):
        propagate_monte_carlo(
            tolerance_response,
            (0.5,),
            (),
            draws=draws,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )


def test_known_truth_pareto_candidate_passes_robust_promotion() -> None:
    candidate = analytical_truth((0.5,))
    endpoint_front = (analytical_truth((0.25,)), analytical_truth((0.75,)))
    assessment = assess_promotion(
        candidate,
        (GaussianConstraint(analytical_constraint_residual((0.5,)), 0.01),),
        endpoint_front,
        ("maximize", "maximize"),
    )
    assert assessment.nondominated
    assert assessment.eligible
    assert assessment.requires_highest_fidelity_reevaluation


def test_empty_comparison_set_is_valid_only_after_candidate_validation() -> None:
    assessment = assess_promotion(
        (1.0, 2.0),
        (),
        (),
        ("maximize", "minimize"),
    )
    assert assessment.nondominated
    assert assessment.eligible
    with pytest.raises(ActiveLearningError):
        assess_promotion((nan,), (), (), ("maximize",))


@pytest.mark.parametrize(
    ("objectives", "comparisons", "directions"),
    (
        ((nan,), (), ("maximize",)),
        ((inf,), (), ("maximize",)),
        ((1.0,), ((1.0, 2.0),), ("maximize",)),
        ((1.0,), ((nan,),), ("maximize",)),
        ((1.0,), (None,), ("maximize",)),
        ((1.0,), (), ()),
        ((1.0,), (), ("MAXIMIZE",)),
        ((1.0,), (), (1,)),
        ((), (), ()),
    ),
)
def test_promotion_rejects_malformed_pareto_domain_before_logic(
    objectives: object,
    comparisons: object,
    directions: object,
) -> None:
    with pytest.raises(ActiveLearningError):
        assess_promotion(
            objectives,  # type: ignore[arg-type]
            (),
            comparisons,  # type: ignore[arg-type]
            directions,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("robust_sigma", "chance_threshold"),
    (
        (nan, 0.95),
        (inf, 0.95),
        (-1.0, 0.95),
        (2.0, nan),
        (2.0, inf),
        (2.0, -0.1),
        (2.0, 1.1),
        (2.0, True),
    ),
)
def test_promotion_rejects_invalid_robust_and_chance_parameters(
    robust_sigma: object,
    chance_threshold: object,
) -> None:
    with pytest.raises(ActiveLearningError):
        assess_promotion(
            (1.0,),
            (),
            (),
            ("maximize",),
            robust_sigma=robust_sigma,  # type: ignore[arg-type]
            chance_threshold=chance_threshold,  # type: ignore[arg-type]
        )


def test_direct_dominance_rejects_nonfinite_vectors_and_bad_directions() -> None:
    with pytest.raises(ActiveLearningError):
        dominates((1.0,), (nan,), ("maximize",))
    with pytest.raises(ActiveLearningError):
        dominates((1.0,), (0.0,), ("up",))


def test_distribution_validation_rejects_invalid_parameters() -> None:
    with pytest.raises(ActiveLearningError):
        NormalTolerance(0.0, -1.0)
    with pytest.raises(ActiveLearningError):
        UniformTolerance(1.0, -1.0)
    with pytest.raises(ActiveLearningError):
        TriangularTolerance(0.0, 2.0, 1.0)
