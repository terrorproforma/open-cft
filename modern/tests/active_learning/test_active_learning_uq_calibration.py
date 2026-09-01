from math import hypot, sqrt
from statistics import NormalDist

import pytest

from cft_revival.active_learning.calibration import (
    binomial_standard_error,
    coverage_diagnostics,
)
from cft_revival.active_learning.contracts import ActiveLearningError, PosteriorPrediction
from cft_revival.active_learning.synthetic import (
    SYNTHETIC_SOURCES,
    analytical_truth,
    source_output,
)
from cft_revival.active_learning.uncertainty import (
    DiscrepancyEstimate,
    bias_correct_prediction,
    decompose_prediction,
    estimate_additive_discrepancy,
)


def test_paired_discrepancy_recovers_known_analytical_bias() -> None:
    lower_source = SYNTHETIC_SOURCES[0]
    designs = tuple((index / 10.0,) for index in range(11))
    lower = tuple(source_output(design, lower_source) for design in designs)
    truth = tuple(analytical_truth(design) for design in designs)
    discrepancy = estimate_additive_discrepancy(lower, truth)
    assert discrepancy.bias == pytest.approx((0.25, -0.20))
    assert discrepancy.residual_variance == pytest.approx((0.0, 0.0), abs=1e-30)
    prediction = PosteriorPrediction(lower[4], (0.1, 0.1), (0.02, 0.02))
    corrected = bias_correct_prediction(prediction, discrepancy)
    corrected_means = tuple(
        mean + correction
        for mean, correction in zip(
            corrected.objective_means,
            corrected.discrepancy_means,
            strict=True,
        )
    )
    assert corrected_means == pytest.approx(truth[4])


def test_heterogeneous_discrepancy_separates_spread_from_bias_error() -> None:
    estimate = estimate_additive_discrepancy(
        ((0.0,), (0.0,), (0.0,)),
        ((1.0,), (3.0,), (5.0,)),
    )
    assert estimate.bias == pytest.approx((3.0,))
    assert estimate.residual_variance == pytest.approx((4.0,))
    assert estimate.residual_standard_deviation == pytest.approx((2.0,))
    assert estimate.bias_standard_error == pytest.approx((2.0 / sqrt(3.0),))

    prediction = PosteriorPrediction(
        (10.0,),
        (0.3,),
        (0.4,),
        (0.5,),
        (0.5,),
        discrepancy_bias_standard_errors=(0.25,),
    )
    corrected = bias_correct_prediction(prediction, estimate)
    assert corrected.discrepancy_means == pytest.approx((3.5,))
    assert corrected.discrepancy_standard_deviations == pytest.approx((hypot(0.5, 2.0),))
    assert corrected.discrepancy_bias_standard_errors == pytest.approx(
        (hypot(0.25, 2.0 / sqrt(3.0)),)
    )
    decomposition = decompose_prediction(corrected)[0]
    assert decomposition.discrepancy == pytest.approx(4.25)
    assert decomposition.bias_estimation == pytest.approx(
        0.25**2 + (2.0 / sqrt(3.0)) ** 2
    )


def test_variance_components_remain_separately_labelled() -> None:
    prediction = PosteriorPrediction(
        (1.0,),
        (0.3,),
        (0.4,),
        (0.0,),
        (1.2,),
        discrepancy_bias_standard_errors=(0.5,),
    )
    decomposition = decompose_prediction(prediction)[0]
    assert decomposition.epistemic == pytest.approx(0.09)
    assert decomposition.aleatoric == pytest.approx(0.16)
    assert decomposition.discrepancy == pytest.approx(1.44)
    assert decomposition.bias_estimation == pytest.approx(0.25)
    assert decomposition.standard_deviation == pytest.approx(sqrt(1.94))


def test_coverage_reports_count_uncertainty_confidence_and_stratum() -> None:
    normal = NormalDist()
    predictions = tuple(
        PosteriorPrediction((0.0,), (0.0,), (1.0,)) for _ in range(200)
    )
    truths = tuple(
        (normal.inv_cdf((index + 0.5) / 200.0),) for index in range(200)
    )
    diagnostics = coverage_diagnostics(
        predictions,
        truths,
        levels=(0.5, 0.8, 0.95),
        tolerance=0.011,
        stratum="in-domain",
        confidence_level=0.95,
    )
    assert diagnostics.checked
    assert diagnostics.passes_tolerance
    assert diagnostics.sample_count == 200
    assert diagnostics.stratum == "in-domain"
    for level in diagnostics.levels:
        assert level.count == diagnostics.sample_count
        assert level.binomial_standard_error == binomial_standard_error(level)
        assert level.confidence_interval[0] <= level.observed <= level.confidence_interval[1]


@pytest.mark.parametrize("stratum", (None, "", "mixed", "in_domain"))
def test_calibration_requires_one_declared_regime(stratum: object) -> None:
    with pytest.raises(ActiveLearningError):
        coverage_diagnostics(
            (PosteriorPrediction((0.0,), (0.1,), (0.1,)),),
            ((0.0,),),
            stratum=stratum,  # type: ignore[arg-type]
        )


def test_in_domain_and_ood_diagnostics_remain_separate() -> None:
    prediction = (PosteriorPrediction((0.0,), (0.1,), (0.1,)),)
    truth = ((0.0,),)
    in_domain = coverage_diagnostics(prediction, truth, stratum="in-domain")
    ood = coverage_diagnostics(prediction, truth, stratum="ood")
    assert in_domain.stratum != ood.stratum


@pytest.mark.parametrize("paired_count", (True, 1.0, 0, -1))
def test_discrepancy_count_requires_positive_integer(paired_count: object) -> None:
    with pytest.raises(ActiveLearningError):
        DiscrepancyEstimate((0.0,), (0.0,), paired_count)  # type: ignore[arg-type]


def test_malformed_posterior_vectors_raise_domain_error() -> None:
    with pytest.raises(ActiveLearningError):
        PosteriorPrediction(None, (1.0,), (1.0,))  # type: ignore[arg-type]
    with pytest.raises(ActiveLearningError):
        PosteriorPrediction((0.0,), None, (1.0,))  # type: ignore[arg-type]
