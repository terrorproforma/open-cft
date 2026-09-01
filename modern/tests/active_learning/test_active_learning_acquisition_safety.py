from math import inf, isfinite, nan
from sys import float_info

import pytest

from cft_revival.active_learning.acquisition import (
    AcquisitionWeights,
    CandidateScore,
    HighestFidelityQuota,
    approximate_pending_fantasization,
    score_candidate,
    select_fidelity,
)
from cft_revival.active_learning.contracts import (
    ActiveLearningError,
    CampaignCounts,
    FidelitySource,
    PosteriorPrediction,
)
from cft_revival.active_learning.synthetic import AnalyticalPosterior, SYNTHETIC_SOURCES


def test_cost_normalization_selects_informative_cheap_source() -> None:
    posterior = AnalyticalPosterior()
    scores = tuple(
        score_candidate((0.5,), source, posterior, incumbent=(0.7, 0.7))
        for source in SYNTHETIC_SOURCES
    )
    decision = select_fidelity(
        scores,
        CampaignCounts(tuple((source.name, 0) for source in SYNTHETIC_SOURCES)),
        HighestFidelityQuota(1),
        remaining_evaluation_slots=4,
    )
    assert decision.source.name == "F0-analytic"
    assert not decision.quota_forced
    assert all(
        isfinite(value)
        for value in (
            decision.score.predicted_improvement,
            decision.score.feasibility_probability,
            decision.score.discrepancy_signal,
            decision.score.uncertainty_signal,
            decision.score.raw_information_value,
            decision.score.cost_normalized_score,
        )
    )


def test_highest_fidelity_quota_overrides_cost_normalized_choice() -> None:
    posterior = AnalyticalPosterior()
    scores = tuple(
        score_candidate((0.5,), source, posterior, incumbent=(0.7, 0.7))
        for source in SYNTHETIC_SOURCES
    )
    counts = CampaignCounts(tuple((source.name, 0) for source in SYNTHETIC_SOURCES))
    decision = select_fidelity(
        scores,
        counts,
        HighestFidelityQuota(1),
        remaining_evaluation_slots=1,
    )
    assert decision.source.is_highest
    assert decision.quota_forced


def test_pending_mean_fantasy_is_explicit_and_repels_duplicate() -> None:
    posterior = AnalyticalPosterior()
    prediction = posterior.predict((0.5,), SYNTHETIC_SOURCES[0])
    fantasy = approximate_pending_fantasization(
        (0.5,),
        (0.0, 0.0),
        (((0.5,), prediction),),
    )
    assert fantasy.label == "asynchronous-posterior-mean-fantasy-approximation"
    assert fantasy.spatial_penalty == pytest.approx(0.0)


class ExtremePosterior:
    def predict(
        self,
        design: tuple[float, ...],
        source: FidelitySource,
    ) -> PosteriorPrediction:
        del design, source
        return PosteriorPrediction(
            (1.0e308, -1.0e308),
            (1.0e308, 1.0e308),
            (1.0e308, 1.0e308),
            (-1.0e307, 1.0e307),
            (1.0e308, 1.0e308),
            discrepancy_bias_standard_errors=(1.0e308, 1.0e308),
        )


class MomentPosterior:
    def __init__(self, prediction: PosteriorPrediction) -> None:
        self.prediction = prediction

    def predict(
        self,
        design: tuple[float, ...],
        source: FidelitySource,
    ) -> PosteriorPrediction:
        del design, source
        return self.prediction


def _uncertainty_only_score(
    prediction: PosteriorPrediction,
    *,
    scale: float = 1.0,
) -> CandidateScore:
    return score_candidate(
        (0.5,),
        SYNTHETIC_SOURCES[0],
        MomentPosterior(prediction),
        incumbent=(0.0,) * len(prediction.objective_means),
        weights=AcquisitionWeights(0.0, 0.0, 0.0, 1.0),
        uncertainty_scales=(scale,) * len(prediction.objective_means),
    )


def test_uncertainty_signal_preserves_total_quadrature_ordering() -> None:
    epistemic_only = PosteriorPrediction((0.0,), (1.0,), (0.0,))
    four_components = PosteriorPrediction(
        (0.0,),
        (0.2,),
        (0.2,),
        (0.0,),
        (0.2,),
        discrepancy_bias_standard_errors=(0.2,),
    )
    high = _uncertainty_only_score(epistemic_only)
    low = _uncertainty_only_score(four_components)
    assert high.uncertainty_signal > low.uncertainty_signal
    assert high.cost_normalized_score > low.cost_normalized_score


def test_uncertainty_signal_is_consistent_under_declared_output_rescaling() -> None:
    base = PosteriorPrediction(
        (0.0,),
        (0.3,),
        (0.4,),
        (0.0,),
        (0.5,),
        discrepancy_bias_standard_errors=(0.6,),
    )
    factor = 1.0e200
    scaled = PosteriorPrediction(
        (0.0,),
        (0.3 * factor,),
        (0.4 * factor,),
        (0.0,),
        (0.5 * factor,),
        discrepancy_bias_standard_errors=(0.6 * factor,),
    )
    base_score = _uncertainty_only_score(base)
    scaled_score = _uncertainty_only_score(scaled, scale=factor)
    assert base_score.uncertainty_signal == pytest.approx(
        scaled_score.uncertainty_signal,
        rel=1.0e-14,
    )
    assert base_score.raw_information_value == pytest.approx(
        scaled_score.raw_information_value,
        rel=1.0e-14,
    )


def test_extreme_finite_posterior_produces_only_finite_bounded_scores() -> None:
    score = score_candidate(
        (float_info.max,),
        SYNTHETIC_SOURCES[0],
        ExtremePosterior(),
        incumbent=(-1.0e308, 1.0e308),
    )
    assert 0.0 <= score.raw_information_value <= 1.0
    assert 0.0 <= score.cost_normalized_score <= 1.0
    assert all(
        isfinite(value)
        for value in (
            score.predicted_improvement,
            score.discrepancy_signal,
            score.uncertainty_signal,
            score.cost_normalized_score,
        )
    )


class OverflowingCorrectionPosterior:
    def predict(
        self,
        design: tuple[float, ...],
        source: FidelitySource,
    ) -> PosteriorPrediction:
        del design, source
        return PosteriorPrediction((float_info.max,), (1.0,), (1.0,), (float_info.max,))


@pytest.mark.parametrize(
    "posterior",
    (
        object(),
        type("WrongReturn", (), {"predict": lambda self, design, source: (1.0,)})(),
        type(
            "RaisesTypeError",
            (),
            {"predict": lambda self, design, source: (_ for _ in ()).throw(TypeError("bad"))},
        )(),
        OverflowingCorrectionPosterior(),
    ),
)
def test_malformed_or_overflowing_adapter_fails_with_domain_error(posterior: object) -> None:
    with pytest.raises(ActiveLearningError):
        score_candidate(
            (0.5,),
            SYNTHETIC_SOURCES[0],
            posterior,  # type: ignore[arg-type]
            incumbent=(0.0,),
        )


def _score(source: FidelitySource, value: float) -> CandidateScore:
    return CandidateScore(source, 0.1, 0.9, 0.1, 0.1, 1.0, value, value, "test")


def test_tie_break_is_deterministic_independent_of_input_order() -> None:
    a = _score(FidelitySource("a", 0, 0.1), 0.5)
    b = _score(FidelitySource("b", 0, 0.1), 0.5)
    highest = _score(FidelitySource("highest", 1, 1.0, True), 0.1)
    counts = CampaignCounts((("a", 0), ("b", 0), ("highest", 1)))
    quota = HighestFidelityQuota(1)
    forward = select_fidelity(
        (a, b, highest),
        counts,
        quota,
        remaining_evaluation_slots=3,
    )
    reverse = select_fidelity(
        (highest, b, a),
        counts,
        quota,
        remaining_evaluation_slots=3,
    )
    assert forward.source.name == reverse.source.name == "a"


@pytest.mark.parametrize("bad_score", (nan, inf, -inf))
def test_nonfinite_candidate_score_is_rejected_before_ranking(bad_score: float) -> None:
    with pytest.raises(ActiveLearningError):
        _score(FidelitySource("bad", 0, 1.0), bad_score)


def test_quota_detects_impossible_remaining_schedule() -> None:
    scores = tuple(
        score_candidate((0.5,), source, AnalyticalPosterior(), incumbent=(0.0, 0.0))
        for source in SYNTHETIC_SOURCES
    )
    with pytest.raises(ActiveLearningError, match="cannot satisfy"):
        select_fidelity(
            scores,
            CampaignCounts(tuple((source.name, 0) for source in SYNTHETIC_SOURCES)),
            HighestFidelityQuota(2),
            remaining_evaluation_slots=1,
        )


@pytest.mark.parametrize("bad", (True, 1.0, -1))
def test_integer_scheduler_contracts_reject_nonintegers_and_negatives(bad: object) -> None:
    with pytest.raises(ActiveLearningError):
        HighestFidelityQuota(bad)  # type: ignore[arg-type]
    with pytest.raises(ActiveLearningError):
        CampaignCounts((("F3", bad),))  # type: ignore[arg-type]
    with pytest.raises(ActiveLearningError):
        CampaignCounts((("F3", 0),), total_completed_successes=bad)  # type: ignore[arg-type]
    scores = tuple(
        score_candidate((0.5,), source, AnalyticalPosterior(), incumbent=(0.0, 0.0))
        for source in SYNTHETIC_SOURCES
    )
    with pytest.raises(ActiveLearningError):
        select_fidelity(
            scores,
            CampaignCounts(tuple((source.name, 0) for source in SYNTHETIC_SOURCES)),
            HighestFidelityQuota(0),
            remaining_evaluation_slots=bad,  # type: ignore[arg-type]
        )
