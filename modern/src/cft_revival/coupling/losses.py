"""Overflow-safe mirror and covariance-aware loss-cone relations."""

from __future__ import annotations

from math import fsum, hypot, isclose, isfinite, sqrt

from .models import (
    CouplingValidationError,
    MirrorLoss,
    TopologyCandidate,
    UncertainProbability,
)


def _stable_probability_from_ratio(ratio: float) -> float:
    if ratio == 0.0:
        return 0.0
    if ratio == 1.0:
        return 0.5
    result = 0.5 * ratio / (1.0 + sqrt(1.0 - ratio))
    if not isfinite(result):
        raise CouplingValidationError("loss-cone probability is non-finite")
    return result


def _finite_sum(*values: float) -> float:
    try:
        result = fsum(values)
    except OverflowError as error:
        raise CouplingValidationError("uncertainty arithmetic overflowed") from error
    if not isfinite(result):
        raise CouplingValidationError("uncertainty arithmetic is non-finite")
    return result


def _square(value: float, name: str) -> float:
    result = value * value
    if not isfinite(result):
        raise CouplingValidationError(f"{name} overflowed")
    return result


def _ratio_variance(
    ratio: float,
    low: float,
    high: float,
    low_independent_sigma: float,
    high_independent_sigma: float,
    common_sigma: float,
    residual_correlation: float,
) -> float:
    if ratio > 0.0:
        low_relative = low_independent_sigma / low
        high_relative = high_independent_sigma / high
        if not isfinite(low_relative) or not isfinite(high_relative):
            raise CouplingValidationError("relative uncertainty overflowed")
        relative_difference_term = ratio * (low_relative - high_relative)
        correlation_term = (
            2.0
            * ratio
            * ratio
            * low_relative
            * high_relative
            * (1.0 - residual_correlation)
        )
    else:
        relative_difference_term = low_independent_sigma / high
        correlation_term = 0.0
    if not isfinite(correlation_term):
        raise CouplingValidationError("correlated uncertainty term overflowed")
    common_term = (1.0 - ratio) * (common_sigma / high)
    terms = (
        _square(
            relative_difference_term,
            "relative-error difference uncertainty",
        ),
        correlation_term,
        _square(common_term, "common-mode uncertainty derivative"),
    )
    variance = _finite_sum(*terms)
    if variance < 0.0:
        raise CouplingValidationError("uncertainty covariance is not positive semidefinite")
    return variance


def _interval_ratios(
    low: float,
    high: float,
    low_independent_sigma: float,
    high_independent_sigma: float,
    common_sigma: float,
    coverage: float,
) -> tuple[float, float]:
    """Conservative independent-adverse/common-shared interval propagation."""

    scaled_low = coverage * low_independent_sigma
    scaled_high = coverage * high_independent_sigma
    scaled_common = coverage * common_sigma
    if any(not isfinite(value) for value in (scaled_low, scaled_high, scaled_common)):
        raise CouplingValidationError("coverage-scaled uncertainty overflowed")
    lower_candidates = []
    upper_candidates = []
    for direction in (-1.0, 1.0):
        common_shift = direction * scaled_common
        lower_numerator = max(0.0, _finite_sum(low, -scaled_low, common_shift))
        lower_denominator = _finite_sum(high, scaled_high, common_shift)
        lower_candidates.append(
            0.0
            if lower_denominator <= 0.0
            else min(1.0, lower_numerator / lower_denominator)
        )
        upper_numerator = max(0.0, _finite_sum(low, scaled_low, common_shift))
        upper_denominator = _finite_sum(high, -scaled_high, common_shift)
        upper_candidates.append(
            1.0
            if upper_denominator <= 0.0
            else min(1.0, upper_numerator / upper_denominator)
        )
    return min(lower_candidates), max(upper_candidates)


def loss_cone_probability(
    low_field_t: float,
    high_field_t: float,
    *,
    low_sigma_t: float = 0.0,
    high_sigma_t: float = 0.0,
    common_mode_sigma_t: float = 0.0,
    residual_correlation: float = 0.0,
    covariance_t2: float | None = None,
    correlation: float | None = None,
    coverage_factor: float = 1.0,
) -> UncertainProbability:
    """Return a bounded loss probability with covariance-aware delta uncertainty.

    ``low_sigma_t`` and ``high_sigma_t`` are independent/residual one-sigma
    values in tesla. ``common_mode_sigma_t`` is a shared additive one-sigma
    field error. Correlation applies to the independent residuals.
    """

    low = float(low_field_t)
    high = float(high_field_t)
    low_sigma = float(low_sigma_t)
    high_sigma = float(high_sigma_t)
    common = float(common_mode_sigma_t)
    requested_rho = float(
        residual_correlation if correlation is None else correlation
    )
    coverage = float(coverage_factor)
    values = (low, high, low_sigma, high_sigma, common, requested_rho, coverage)
    if any(not isfinite(value) for value in values):
        raise CouplingValidationError("loss-cone inputs must be finite")
    if low < 0.0 or high <= 0.0 or low > high:
        raise CouplingValidationError("mirror fields require 0 <= low <= high")
    if low_sigma < 0.0 or high_sigma < 0.0 or common < 0.0 or coverage <= 0.0:
        raise CouplingValidationError(
            "uncertainties must be non-negative and coverage_factor positive"
        )
    if not -1.0 <= requested_rho <= 1.0:
        raise CouplingValidationError("correlation must be in [-1, 1]")
    common_covariance = _square(common, "common-mode covariance")
    if covariance_t2 is not None and correlation is None:
        supplied_covariance = float(covariance_t2)
        if not isfinite(supplied_covariance):
            raise CouplingValidationError("covariance_t2 must be finite")
        residual_covariance = supplied_covariance - common_covariance
        if not isfinite(residual_covariance):
            raise CouplingValidationError("residual covariance overflowed")
        if low_sigma == 0.0 or high_sigma == 0.0:
            if residual_covariance != 0.0:
                raise CouplingValidationError(
                    "zero residual variance cannot have nonzero covariance"
                )
            rho = 0.0
        else:
            rho = residual_covariance / low_sigma / high_sigma
            if not isfinite(rho) or not -1.0 <= rho <= 1.0:
                raise CouplingValidationError(
                    "covariance_t2 is not PSD for the declared sigmas"
                )
        derived_covariance = supplied_covariance
    else:
        rho = requested_rho
        residual_covariance = rho * low_sigma * high_sigma
        if not isfinite(residual_covariance):
            raise CouplingValidationError("residual covariance overflowed")
        derived_covariance = _finite_sum(common_covariance, residual_covariance)
    if covariance_t2 is not None and correlation is not None:
        supplied_covariance = float(covariance_t2)
        if not isfinite(supplied_covariance):
            raise CouplingValidationError("covariance_t2 must be finite")
        tolerance = max(
            1.0e-300,
            32.0
            * 2.220446049250313e-16
            * max(abs(supplied_covariance), abs(derived_covariance)),
        )
        if not isclose(
            supplied_covariance,
            derived_covariance,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise CouplingValidationError(
                "covariance_t2 is inconsistent with correlation/common-mode inputs"
            )
    total_low_sigma = hypot(low_sigma, common)
    total_high_sigma = hypot(high_sigma, common)
    if total_low_sigma > 0.0 and total_high_sigma > 0.0:
        normalized_covariance = (
            derived_covariance / total_low_sigma / total_high_sigma
        )
        if abs(normalized_covariance) > 1.0 + 32.0 * 2.220446049250313e-16:
            raise CouplingValidationError("input covariance matrix is not PSD")
    elif derived_covariance != 0.0:
        raise CouplingValidationError("zero variance cannot have nonzero covariance")
    ratio = low / high
    probability = _stable_probability_from_ratio(ratio)
    ratio_lower, ratio_upper = _interval_ratios(
        low, high, low_sigma, high_sigma, common, coverage
    )
    lower_probability = _stable_probability_from_ratio(ratio_lower)
    upper_probability = _stable_probability_from_ratio(ratio_upper)
    ratio_variance = _ratio_variance(
        ratio, low, high, low_sigma, high_sigma, common, rho
    )
    ratio_sigma = sqrt(ratio_variance)
    if ratio_sigma == 0.0:
        probability_sigma = 0.0
        propagation = "exact_central_zero_delta_variance"
    elif ratio < 1.0:
        denominator = 4.0 * sqrt(1.0 - ratio)
        first_order = ratio_sigma / denominator
        interval_radius = max(
            probability - lower_probability, upper_probability - probability
        )
        probability_sigma = min(first_order, interval_radius)
        propagation = "covariance_delta_clipped_to_conservative_interval"
    else:
        probability_sigma = probability - lower_probability
        propagation = "conservative_interval_at_singular_endpoint"
    outputs = (
        probability,
        probability_sigma,
        lower_probability,
        upper_probability,
        ratio_sigma,
        derived_covariance,
    )
    if any(not isfinite(value) for value in outputs):
        raise CouplingValidationError("finite loss inputs produced non-finite outputs")
    return UncertainProbability(
        value=probability,
        standard_uncertainty=probability_sigma,
        lower=lower_probability,
        upper=upper_probability,
        ratio_standard_uncertainty=ratio_sigma,
        input_covariance_t2=derived_covariance,
        input_correlation=rho,
        coverage_factor=coverage,
        propagation=propagation,
        interval_method="independent_adverse_plus_shared_additive_common_mode",
    )


def _safe_confidence(signal: float, uncertainty: float) -> float:
    if signal <= 0.0:
        return 0.0
    if uncertainty <= 0.0:
        return 1.0
    if signal <= uncertainty:
        ratio = signal / uncertainty
        return ratio / (1.0 + ratio)
    ratio = uncertainty / signal
    return 1.0 / (1.0 + ratio)


def derive_mirror_loss(
    cusp: TopologyCandidate,
    wall_b_t: float,
    wall_independent_sigma_b_t: float,
    *,
    common_mode_sigma_t: float = 0.0,
    residual_correlation: float = 0.0,
    coverage_factor: float = 1.0,
) -> MirrorLoss:
    """Derive one mirror relation with explicit shared/residual covariance."""

    wall = float(wall_b_t)
    wall_sigma = float(wall_independent_sigma_b_t)
    common = float(common_mode_sigma_t)
    if any(not isfinite(value) for value in (wall, wall_sigma, common)):
        raise CouplingValidationError("wall field and uncertainty must be finite")
    if wall <= 0.0 or wall_sigma < 0.0 or common < 0.0:
        raise CouplingValidationError("wall field must be positive and sigmas non-negative")
    if cusp.common_mode_sigma_t != common:
        raise CouplingValidationError(
            "cusp and wall must use the same additive common-mode sigma"
        )
    if cusp.b_magnitude_t > wall:
        raise CouplingValidationError(
            "inverted magnetic mirror: inner low field exceeds wall high field"
        )
    probability = loss_cone_probability(
        cusp.b_magnitude_t,
        wall,
        low_sigma_t=cusp.independent_sigma_b_t,
        high_sigma_t=wall_sigma,
        common_mode_sigma_t=common,
        residual_correlation=residual_correlation,
        coverage_factor=coverage_factor,
    )
    ratio = cusp.b_magnitude_t / wall
    mirror_ratio = None
    if cusp.b_magnitude_t != 0.0:
        mirror_ratio = wall / cusp.b_magnitude_t
        if not isfinite(mirror_ratio):
            raise CouplingValidationError("magnetic mirror ratio overflowed")
    contrast = wall - cusp.b_magnitude_t
    total_uncertainty = hypot(
        cusp.sigma_b_t, hypot(wall_sigma, common)
    )
    return MirrorLoss(
        cusp=cusp,
        wall_b_t=wall,
        wall_independent_sigma_b_t=wall_sigma,
        common_mode_sigma_t=common,
        covariance_t2=probability.input_covariance_t2,
        correlation=probability.input_correlation,
        field_ratio_low_to_high=ratio,
        mirror_ratio_high_to_low=mirror_ratio,
        probability=probability,
        confidence=min(
            cusp.confidence, _safe_confidence(contrast, total_uncertainty)
        ),
    )
