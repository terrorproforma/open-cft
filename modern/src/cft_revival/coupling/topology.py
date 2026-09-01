"""Numerically safe, uncertainty-gated topology detection."""

from __future__ import annotations

from math import fsum, hypot, isfinite, sqrt

from .models import (
    CandidateKind,
    CouplingValidationError,
    FieldProfile,
    PlateauPolicy,
    ProfileDescriptor,
    TiePolicy,
    TopologyCandidate,
    TopologyPolicy,
    TopologyResolutionError,
    TopologyStatus,
)
from .profiles import interpolate_profile, stable_lerp

_MINIMUM_KINDS = frozenset(
    (CandidateKind.NULL, CandidateKind.MINIMUM, CandidateKind.PLATEAU_MINIMUM)
)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _safe_fraction(numerator: float, other: float) -> float:
    """Return numerator/(numerator+other) without overflowing the sum."""

    if numerator < 0.0 or other < 0.0:
        raise TopologyResolutionError("safe fraction requires non-negative inputs")
    if numerator == 0.0:
        return 0.0
    if other == 0.0:
        return 1.0
    if numerator <= other:
        ratio = numerator / other
        return ratio / (1.0 + ratio)
    ratio = other / numerator
    return 1.0 / (1.0 + ratio)


def _safe_midpoint(left: float, right: float) -> float:
    result = left * 0.5 + right * 0.5
    if not isfinite(result):
        raise TopologyResolutionError("finite coordinates produced midpoint overflow")
    return result


def _profile_is_degenerate(profile: FieldProfile, tolerance: float) -> bool:
    magnitude = profile.magnitude_t
    if max(magnitude) - min(magnitude) > tolerance:
        return False
    first_br = profile.b_r_t[0]
    first_bz = profile.b_z_t[0]
    return all(
        abs(br - first_br) <= tolerance and abs(bz - first_bz) <= tolerance
        for br, bz in zip(profile.b_r_t, profile.b_z_t, strict=True)
    )


def _scaled_tolerance(absolute: float, relative: float, scale: float) -> float:
    scaled = relative * scale
    if not isfinite(scaled):
        raise TopologyResolutionError("field-scaled tolerance overflowed")
    return max(absolute, scaled)


def _validate_policy(policy: TopologyPolicy) -> None:
    non_negative = (
        ("relative_value_tolerance", policy.relative_value_tolerance),
        ("absolute_value_tolerance_t", policy.absolute_value_tolerance_t),
        ("null_relative_tolerance", policy.null_relative_tolerance),
        ("null_absolute_tolerance_t", policy.null_absolute_tolerance_t),
        ("minimum_prominence_relative", policy.minimum_prominence_relative),
        ("minimum_prominence_sigma", policy.minimum_prominence_sigma),
        ("tie_relative_tolerance", policy.tie_relative_tolerance),
        ("tie_absolute_tolerance_t", policy.tie_absolute_tolerance_t),
    )
    for name, raw in non_negative:
        if not isfinite(float(raw)) or float(raw) < 0.0:
            raise CouplingValidationError(f"{name} must be finite and non-negative")
    for name, raw in (
        ("minimum_candidate_confidence", policy.minimum_candidate_confidence),
        ("minimum_segment_confidence", policy.minimum_segment_confidence),
    ):
        if not isfinite(float(raw)) or not 0.0 <= float(raw) <= 1.0:
            raise CouplingValidationError(f"{name} must be finite and in [0, 1]")
    if not isinstance(policy.plateau_policy, PlateauPolicy):
        raise CouplingValidationError("plateau_policy must be a PlateauPolicy")
    if not isinstance(policy.tie_policy, TiePolicy):
        raise CouplingValidationError("tie_policy must be a TiePolicy")


def _candidate(
    profile: FieldProfile,
    kind: CandidateKind,
    z_m: float,
    bracket: tuple[float, float],
    indices: tuple[int, ...],
    interpolation: str,
    confidence: float,
    prominence_t: float,
) -> TopologyCandidate:
    br, bz, magnitude, independent, common = interpolate_profile(profile, z_m)
    return TopologyCandidate(
        kind=kind,
        z_m=z_m,
        b_magnitude_t=magnitude,
        b_r_t=br,
        b_z_t=bz,
        sigma_b_t=hypot(independent, common),
        independent_sigma_b_t=independent,
        common_mode_sigma_t=common,
        prominence_t=prominence_t,
        confidence=_clamp01(confidence),
        bracket_z_m=bracket,
        interpolation=interpolation,
        sample_indices=indices,
    )


def _spacing_confidence(z_m: tuple[float, ...], left: int, right: int) -> float:
    start = max(0, left - 1)
    stop = min(len(z_m) - 1, right + 1)
    widths = []
    for index in range(start, stop):
        width = z_m[index + 1] - z_m[index]
        if not isfinite(width) or width <= 0.0:
            raise TopologyResolutionError("axial spacing is not finitely resolvable")
        widths.append(width)
    return 0.0 if not widths else min(widths) / max(widths)


def _null_confidence(
    profile: FieldProfile,
    left_index: int,
    right_index: int,
    residual_t: float,
    tolerance_t: float,
) -> float:
    spacing = _spacing_confidence(profile.z_m, left_index, right_index)
    total_sigma = max(profile.sigma_b_t[left_index : right_index + 1])
    if left_index == right_index:
        neighboring = [
            abs(profile.b_z_t[index])
            for index in (left_index - 1, right_index + 1)
            if 0 <= index < len(profile.z_m)
        ]
        signal = min(neighboring) if neighboring else 0.0
    else:
        endpoint_signal = (
            abs(profile.b_z_t[left_index]),
            abs(profile.b_z_t[right_index]),
        )
        outside = [
            abs(profile.b_z_t[index])
            for index in (left_index - 1, right_index + 1)
            if 0 <= index < len(profile.z_m)
        ]
        signal = (
            min(outside)
            if max(endpoint_signal) <= tolerance_t and outside
            else min(endpoint_signal)
        )
    signal_confidence = _safe_fraction(signal, total_sigma)
    residual_confidence = (
        1.0 if tolerance_t == 0.0 else _clamp01(1.0 - residual_t / tolerance_t)
    )
    return spacing * sqrt(signal_confidence * residual_confidence)


def locate_nulls(
    profile: FieldProfile, policy: TopologyPolicy = TopologyPolicy()
) -> tuple[TopologyCandidate, ...]:
    """Locate vector-null hypotheses without overflow-prone sign products."""

    _validate_policy(policy)
    magnitude = profile.magnitude_t
    scale = max(magnitude)
    null_tolerance = _scaled_tolerance(
        policy.null_absolute_tolerance_t, policy.null_relative_tolerance, scale
    )
    value_tolerance = _scaled_tolerance(
        policy.absolute_value_tolerance_t, policy.relative_value_tolerance, scale
    )
    if _profile_is_degenerate(profile, value_tolerance):
        return ()
    candidates: list[TopologyCandidate] = []
    index = 0
    while index < len(profile.z_m):
        if (
            abs(profile.b_z_t[index]) <= null_tolerance
            and abs(profile.b_r_t[index]) <= null_tolerance
        ):
            end = index
            while end + 1 < len(profile.z_m) and (
                abs(profile.b_z_t[end + 1]) <= null_tolerance
                and abs(profile.b_r_t[end + 1]) <= null_tolerance
            ):
                end += 1
            if end > index and policy.plateau_policy is PlateauPolicy.REJECT:
                raise TopologyResolutionError("null plateau rejected by topology policy")
            locations = (
                (profile.z_m[index], profile.z_m[end])
                if end > index and policy.plateau_policy is PlateauPolicy.BOUNDS
                else (_safe_midpoint(profile.z_m[index], profile.z_m[end]),)
            )
            residual = max(magnitude[index : end + 1])
            confidence = _null_confidence(
                profile, index, end, residual, null_tolerance
            )
            for location in locations:
                candidates.append(
                    _candidate(
                        profile,
                        CandidateKind.NULL,
                        location,
                        (profile.z_m[index], profile.z_m[end]),
                        tuple(range(index, end + 1)),
                        "plateau_bounds"
                        if len(locations) == 2
                        else "plateau_midpoint",
                        confidence,
                        max(0.0, scale - residual),
                    )
                )
            index = end + 1
            continue
        if index + 1 < len(profile.z_m):
            left = profile.b_z_t[index]
            right = profile.b_z_t[index + 1]
            crosses = (left < 0.0 < right) or (right < 0.0 < left)
            if crosses:
                fraction = _safe_fraction(abs(left), abs(right))
                location = stable_lerp(
                    profile.z_m[index], profile.z_m[index + 1], fraction
                )
                br = stable_lerp(
                    profile.b_r_t[index], profile.b_r_t[index + 1], fraction
                )
                if abs(br) <= null_tolerance:
                    candidates.append(
                        _candidate(
                            profile,
                            CandidateKind.NULL,
                            location,
                            (profile.z_m[index], profile.z_m[index + 1]),
                            (index, index + 1),
                            "linear_signed_bz_root",
                            _null_confidence(
                                profile,
                                index,
                                index + 1,
                                abs(br),
                                null_tolerance,
                            ),
                            min(abs(left), abs(right)),
                        )
                    )
        index += 1
    return tuple(sorted(candidates, key=lambda item: (item.z_m, item.kind.value)))


def _quadratic_vertex(
    x0: float, y0: float, x1: float, y1: float, x2: float, y2: float
) -> float | None:
    left_width = x1 - x0
    right_width = x2 - x1
    total_width = x2 - x0
    if (
        not isfinite(left_width)
        or not isfinite(right_width)
        or not isfinite(total_width)
        or min(left_width, right_width) <= 0.0
    ):
        raise TopologyResolutionError("quadratic interpolation spacing overflowed")
    y_scale = max(y0, y1, y2)
    if y_scale == 0.0:
        return None
    u1 = left_width / total_width
    slope01 = ((y1 / y_scale) - (y0 / y_scale)) / u1
    slope12 = ((y2 / y_scale) - (y1 / y_scale)) / (1.0 - u1)
    curvature = slope12 - slope01
    if curvature == 0.0:
        return None
    vertex_u = (curvature * u1 - slope01) / (2.0 * curvature)
    if not 0.0 <= vertex_u <= 1.0:
        return None
    return stable_lerp(x0, x2, vertex_u)


def _extremum_runs(
    values: tuple[float, ...], tolerance: float
) -> tuple[tuple[int, int], ...]:
    """Group by total run span, preventing tolerance chaining."""

    runs: list[tuple[int, int]] = []
    start = 0
    run_min = values[0]
    run_max = values[0]
    for index in range(1, len(values)):
        candidate_min = min(run_min, values[index])
        candidate_max = max(run_max, values[index])
        if candidate_max - candidate_min > tolerance:
            runs.append((start, index - 1))
            start = index
            run_min = values[index]
            run_max = values[index]
        else:
            run_min = candidate_min
            run_max = candidate_max
    runs.append((start, len(values) - 1))
    return tuple(runs)


def _confidence(
    prominence: float, sigma: float, tolerance: float, spacing: float
) -> float:
    return (
        spacing
        * _safe_fraction(prominence, sigma)
        * _safe_fraction(prominence, tolerance)
    )


def _candidate_for_run(
    profile: FieldProfile,
    policy: TopologyPolicy,
    values: tuple[float, ...],
    tolerance: float,
    start: int,
    end: int,
    kind: CandidateKind,
    prominence: float,
) -> tuple[TopologyCandidate, ...]:
    plateau = end > start
    if plateau and policy.plateau_policy is PlateauPolicy.REJECT:
        raise TopologyResolutionError("extremum plateau rejected by topology policy")
    if plateau and policy.plateau_policy is PlateauPolicy.BOUNDS:
        locations = (profile.z_m[start], profile.z_m[end])
        interpolation = "plateau_bounds"
    elif plateau:
        locations = (_safe_midpoint(profile.z_m[start], profile.z_m[end]),)
        interpolation = "plateau_midpoint"
    else:
        location = profile.z_m[start]
        interpolation = "sample"
        if 0 < start < len(values) - 1:
            vertex = _quadratic_vertex(
                profile.z_m[start - 1],
                values[start - 1],
                profile.z_m[start],
                values[start],
                profile.z_m[start + 1],
                values[start + 1],
            )
            if vertex is not None:
                location = vertex
                interpolation = "quadratic_magnitude_vertex"
        locations = (location,)
    sigma = max(profile.sigma_b_t[start : end + 1])
    confidence = _confidence(
        prominence,
        sigma,
        tolerance,
        _spacing_confidence(profile.z_m, start, end),
    )
    bracket = (
        profile.z_m[max(0, start - 1)],
        profile.z_m[min(len(values) - 1, end + 1)],
    )
    return tuple(
        _candidate(
            profile,
            kind,
            location,
            bracket,
            tuple(range(start, end + 1)),
            interpolation,
            confidence,
            prominence,
        )
        for location in locations
    )


def locate_extrema(
    profile: FieldProfile, policy: TopologyPolicy = TopologyPolicy()
) -> tuple[TopologyCandidate, ...]:
    """Locate interior extrema only; boundaries are reported separately."""

    _validate_policy(policy)
    values = profile.magnitude_t
    scale = max(values)
    tolerance = _scaled_tolerance(
        policy.absolute_value_tolerance_t, policy.relative_value_tolerance, scale
    )
    if max(values) - min(values) <= tolerance:
        return ()
    minimum_prominence = _scaled_tolerance(
        0.0, policy.minimum_prominence_relative, scale
    )
    candidates: list[TopologyCandidate] = []
    for start, end in _extremum_runs(values, tolerance):
        if start == 0 or end == len(values) - 1:
            continue
        left = values[start - 1]
        right = values[end + 1]
        is_minimum = values[start] < left - tolerance and values[end] < right - tolerance
        is_maximum = values[start] > left + tolerance and values[end] > right + tolerance
        if is_minimum:
            prominence = min(left - values[start], right - values[end])
            kind = (
                CandidateKind.PLATEAU_MINIMUM
                if end > start
                else CandidateKind.MINIMUM
            )
        elif is_maximum:
            prominence = min(values[start] - left, values[end] - right)
            kind = (
                CandidateKind.PLATEAU_MAXIMUM
                if end > start
                else CandidateKind.MAXIMUM
            )
        else:
            continue
        if prominence < minimum_prominence:
            continue
        candidates.extend(
            _candidate_for_run(
                profile, policy, values, tolerance, start, end, kind, prominence
            )
        )
    candidates.sort(key=lambda item: (item.z_m, item.kind.value))
    if policy.tie_policy is TiePolicy.PRESERVE:
        return tuple(candidates)
    kept: list[TopologyCandidate] = []
    for candidate in candidates:
        tied_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if existing.kind is candidate.kind
                and abs(existing.b_magnitude_t - candidate.b_magnitude_t)
                <= _scaled_tolerance(
                    policy.tie_absolute_tolerance_t,
                    policy.tie_relative_tolerance,
                    max(
                        abs(existing.b_magnitude_t),
                        abs(candidate.b_magnitude_t),
                    ),
                )
            ),
            None,
        )
        if tied_index is None:
            kept.append(candidate)
        elif candidate.confidence > kept[tied_index].confidence:
            kept[tied_index] = candidate
    return tuple(sorted(kept, key=lambda item: (item.z_m, item.kind.value)))


def locate_boundary_extrema(
    profile: FieldProfile, policy: TopologyPolicy = TopologyPolicy()
) -> tuple[TopologyCandidate, ...]:
    """Report strict boundary extrema without promoting them to topology."""

    _validate_policy(policy)
    if not policy.report_boundary_extrema:
        return ()
    values = profile.magnitude_t
    scale = max(values)
    tolerance = _scaled_tolerance(
        policy.absolute_value_tolerance_t, policy.relative_value_tolerance, scale
    )
    if max(values) - min(values) <= tolerance:
        return ()
    candidates = []
    for index, neighbor in ((0, 1), (len(values) - 1, len(values) - 2)):
        difference = values[index] - values[neighbor]
        if abs(difference) <= tolerance:
            continue
        kind = (
            CandidateKind.BOUNDARY_MINIMUM
            if difference < 0.0
            else CandidateKind.BOUNDARY_MAXIMUM
        )
        candidates.append(
            _candidate(
                profile,
                kind,
                profile.z_m[index],
                (
                    min(profile.z_m[index], profile.z_m[neighbor]),
                    max(profile.z_m[index], profile.z_m[neighbor]),
                ),
                (index,),
                "boundary_sample_diagnostic",
                _confidence(
                    abs(difference),
                    profile.sigma_b_t[index],
                    tolerance,
                    1.0,
                ),
                abs(difference),
            )
        )
    return tuple(candidates)


def describe_profile(
    profile: FieldProfile, policy: TopologyPolicy = TopologyPolicy()
) -> ProfileDescriptor:
    """Return resolved, ambiguous, degenerate, or no-topology diagnostics."""

    _validate_policy(policy)
    magnitude = profile.magnitude_t
    scale = max(magnitude)
    tolerance = _scaled_tolerance(
        policy.absolute_value_tolerance_t, policy.relative_value_tolerance, scale
    )
    try:
        terms = []
        for index in range(len(magnitude) - 1):
            average = magnitude[index] * 0.5 + magnitude[index + 1] * 0.5
            width = profile.z_m[index + 1] - profile.z_m[index]
            term = average * width
            if not isfinite(term):
                raise TopologyResolutionError("profile integral overflowed")
            terms.append(term)
        integral = fsum(terms)
    except OverflowError as error:
        raise TopologyResolutionError("profile integral overflowed") from error
    if not isfinite(integral):
        raise TopologyResolutionError("profile integral is non-finite")
    degenerate = _profile_is_degenerate(profile, tolerance)
    nulls = () if degenerate else locate_nulls(profile, policy)
    extrema = () if degenerate else locate_extrema(profile, policy)
    boundaries = () if degenerate else locate_boundary_extrema(profile, policy)
    interior_minima = tuple(
        candidate
        for candidate in nulls + extrema
        if candidate.kind in _MINIMUM_KINDS
    )
    supported = tuple(
        candidate
        for candidate in interior_minima
        if candidate.confidence >= policy.minimum_candidate_confidence
        and candidate.prominence_t
        >= policy.minimum_prominence_sigma * candidate.sigma_b_t
    )
    if degenerate:
        status = TopologyStatus.DEGENERATE
        reason = "profile variation is within the declared value tolerance"
    elif supported:
        status = TopologyStatus.RESOLVED
        reason = "one or more interior candidates pass uncertainty/confidence gates"
    elif interior_minima:
        status = TopologyStatus.AMBIGUOUS
        reason = "interior candidates exist but fail uncertainty/confidence gates"
    else:
        status = TopologyStatus.NO_TOPOLOGY
        reason = "no interior null or minimum was detected"
    return ProfileDescriptor(
        name=profile.name,
        role=profile.role,
        sampled_r_m=profile.sampled_r_m,
        minimum_b_t=min(magnitude),
        maximum_b_t=max(magnitude),
        integral_b_t_m=integral,
        nulls=nulls,
        extrema=extrema,
        boundary_extrema=boundaries,
        topology_status=status,
        topology_reason=reason,
    )
