from __future__ import annotations

from math import isclose, ulp

import pytest

from cft_revival.coupling import (
    CandidateKind,
    CouplingValidationError,
    TopologyCandidate,
    derive_mirror_loss,
    loss_cone_probability,
)


def cusp(field_t: float, sigma_t: float = 0.0) -> TopologyCandidate:
    return TopologyCandidate(
        kind=CandidateKind.MINIMUM,
        z_m=0.0,
        b_magnitude_t=field_t,
        b_r_t=0.0,
        b_z_t=field_t,
        sigma_b_t=sigma_t,
        independent_sigma_b_t=sigma_t,
        common_mode_sigma_t=0.0,
        prominence_t=max(field_t, 1.0),
        confidence=0.9,
        bracket_z_m=(-0.1, 0.1),
        interpolation="test",
        sample_indices=(1,),
    )


def test_probability_is_stable_at_endpoints_and_tiny_ratio() -> None:
    assert loss_cone_probability(0.0, 1.0).value == 0.0
    assert loss_cone_probability(1.0, 1.0).value == 0.5
    assert isclose(
        loss_cone_probability(1.0e-18, 1.0).value,
        2.5e-19,
        rel_tol=1.0e-15,
    )
    assert loss_cone_probability(2.0e-323, 1.0).value == 5.0e-324


def test_uncertainty_propagates_to_finite_monotonic_bounds() -> None:
    result = loss_cone_probability(
        0.2,
        1.0,
        low_sigma_t=0.01,
        high_sigma_t=0.02,
    )
    assert result.lower < result.value < result.upper
    assert 0.0 < result.standard_uncertainty < result.upper - result.lower
    endpoint = loss_cone_probability(
        1.0,
        1.0,
        low_sigma_t=0.01,
        high_sigma_t=0.01,
    )
    assert endpoint.upper == 0.5
    assert 0.0 < endpoint.standard_uncertainty < 0.5


def test_covariance_common_mode_and_zero_covariance_behavior() -> None:
    independent = loss_cone_probability(
        0.2, 1.0, low_sigma_t=0.01, high_sigma_t=0.01
    )
    explicit_zero = loss_cone_probability(
        0.2,
        1.0,
        low_sigma_t=0.01,
        high_sigma_t=0.01,
        covariance_t2=0.0,
        correlation=0.0,
    )
    assert explicit_zero == independent
    common = loss_cone_probability(
        0.2, 1.0, common_mode_sigma_t=0.01
    )
    assert common.input_covariance_t2 == pytest.approx(1.0e-4)
    assert common.ratio_standard_uncertainty < independent.ratio_standard_uncertainty
    assert common.lower <= common.value <= common.upper


def test_perfectly_correlated_proportional_errors_cancel_stably() -> None:
    exact = loss_cone_probability(
        0.25,
        1.0,
        low_sigma_t=0.025,
        high_sigma_t=0.1,
        correlation=1.0,
    )
    assert exact.ratio_standard_uncertainty == 0.0
    assert exact.standard_uncertainty == 0.0
    rounded = loss_cone_probability(
        0.3,
        0.9,
        low_sigma_t=0.03,
        high_sigma_t=0.09,
        correlation=1.0,
    )
    assert rounded.ratio_standard_uncertainty <= 4.0 * ulp(0.3 / 0.9)


@pytest.mark.parametrize("correlation", [-1.01, 1.01, float("nan")])
def test_invalid_correlation_is_rejected(correlation: float) -> None:
    with pytest.raises(CouplingValidationError, match="correlation|finite"):
        loss_cone_probability(
            0.2,
            1.0,
            low_sigma_t=0.01,
            high_sigma_t=0.01,
            correlation=correlation,
        )


def test_inconsistent_covariance_is_rejected() -> None:
    with pytest.raises(CouplingValidationError, match="inconsistent"):
        loss_cone_probability(
            0.2,
            1.0,
            low_sigma_t=0.01,
            high_sigma_t=0.01,
            correlation=0.5,
            covariance_t2=0.0,
        )


def test_covariance_in_tesla_squared_can_define_correlation() -> None:
    result = loss_cone_probability(
        0.2,
        1.0,
        low_sigma_t=0.01,
        high_sigma_t=0.02,
        covariance_t2=1.0e-4,
    )
    assert result.input_covariance_t2 == pytest.approx(1.0e-4)
    assert result.input_correlation == pytest.approx(0.5)
    with pytest.raises(CouplingValidationError, match="PSD"):
        loss_cone_probability(
            0.2,
            1.0,
            low_sigma_t=0.01,
            high_sigma_t=0.02,
            covariance_t2=3.0e-4,
        )


def test_extreme_finite_scales_are_finite_or_typed() -> None:
    huge = loss_cone_probability(5.0e307, 1.0e308)
    tiny = loss_cone_probability(5.0e-324, 1.0e-323)
    assert 0.0 < huge.value < 0.5
    assert 0.0 < tiny.value < 0.5
    with pytest.raises(CouplingValidationError, match="mirror ratio overflowed"):
        derive_mirror_loss(cusp(5.0e-324), 1.0e308, 0.0)


def test_mirror_ratio_and_perfect_null_semantics_are_explicit() -> None:
    finite = derive_mirror_loss(cusp(0.2, 0.01), 1.0, 0.02)
    assert finite.mirror_ratio_high_to_low == 5.0
    assert finite.field_ratio_low_to_high == 0.2
    perfect = derive_mirror_loss(cusp(0.0), 1.0, 0.0)
    assert perfect.mirror_ratio_high_to_low is None
    assert perfect.probability.value == 0.0


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (-1.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 1.0),
        (0.1, float("inf")),
    ],
)
def test_adversarial_loss_inputs_are_rejected(low: float, high: float) -> None:
    with pytest.raises(CouplingValidationError):
        loss_cone_probability(low, high)
    if low == 2.0:
        with pytest.raises(CouplingValidationError, match="inverted"):
            derive_mirror_loss(cusp(low), high, 0.0)
