from __future__ import annotations

from dataclasses import dataclass
from math import sin

import pytest

from cft_revival.coupling import (
    CandidateKind,
    PlateauPolicy,
    TiePolicy,
    TopologyPolicy,
    TopologyResolutionError,
    TopologyStatus,
    describe_profile,
    locate_extrema,
    locate_nulls,
    validate_profile,
)


@dataclass
class GenericProfile:
    z_m: tuple[float, ...]
    b_r_t: tuple[float, ...]
    b_z_t: tuple[float, ...]


def profile(
    z_m: tuple[float, ...],
    bz: tuple[float, ...],
    *,
    sigma: tuple[float, ...] | None = None,
):
    return validate_profile(
        GenericProfile(z_m, (0.0,) * len(z_m), bz),
        name="analytic",
        sampled_r_m=0.0,
        independent_sigma_b_t=sigma,
    )


def test_signed_linear_interpolation_finds_known_axis_null() -> None:
    z = (-0.5, -0.1, 0.2, 0.6, 1.0, 1.4, 1.8)
    root = 0.37
    result = locate_nulls(profile(z, tuple(value - root for value in z)))
    assert len(result) == 1
    assert result[0].z_m == pytest.approx(root, abs=1.0e-15)
    assert result[0].interpolation == "linear_signed_bz_root"
    assert result[0].confidence > 0.7


def test_quadratic_interpolation_locates_off_grid_extremum() -> None:
    z = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4)
    vertex = 0.23
    bz = tuple(1.0 + (value - vertex) ** 2 for value in z)
    extrema = locate_extrema(profile(z, bz))
    minima = [item for item in extrema if item.kind is CandidateKind.MINIMUM]
    assert len(minima) == 1
    assert minima[0].z_m == pytest.approx(vertex, abs=1.0e-14)
    assert minima[0].interpolation == "quadratic_magnitude_vertex"


def test_plateau_policy_midpoint_bounds_and_reject_are_explicit() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    sampled = profile(z, (3.0, 2.0, 1.0, 1.0, 1.0, 2.0, 3.0))
    midpoint = locate_extrema(
        sampled, TopologyPolicy(plateau_policy=PlateauPolicy.MIDPOINT)
    )
    minima = [item for item in midpoint if item.kind is CandidateKind.PLATEAU_MINIMUM]
    assert [item.z_m for item in minima] == [0.0]
    bounds = locate_extrema(
        sampled, TopologyPolicy(plateau_policy=PlateauPolicy.BOUNDS)
    )
    assert [
        item.z_m for item in bounds if item.kind is CandidateKind.PLATEAU_MINIMUM
    ] == [-1.0, 1.0]
    with pytest.raises(TopologyResolutionError, match="plateau"):
        locate_extrema(sampled, TopologyPolicy(plateau_policy=PlateauPolicy.REJECT))


def test_equal_candidates_are_preserved_unless_tie_policy_selects() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    sampled = profile(z, (3.0, 1.0, 3.0, 4.0, 3.0, 1.0, 3.0))
    preserve = locate_extrema(sampled)
    assert len([item for item in preserve if item.kind is CandidateKind.MINIMUM]) == 2
    select = locate_extrema(
        sampled, TopologyPolicy(tie_policy=TiePolicy.HIGHEST_CONFIDENCE)
    )
    assert len([item for item in select if item.kind is CandidateKind.MINIMUM]) == 1


def test_noisy_resampled_two_cusp_profile_retains_both_candidates() -> None:
    for count in (41, 81):
        z = tuple(-2.0 + 4.0 * index / (count - 1) for index in range(count))
        bz = tuple(
            (value + 0.7) * (value - 0.8) + 2.0e-4 * sin(31.0 * value)
            for value in z
        )
        descriptor = describe_profile(profile(z, bz))
        assert len(descriptor.nulls) == 2
        assert descriptor.nulls[0].z_m == pytest.approx(-0.7, abs=0.01)
        assert descriptor.nulls[1].z_m == pytest.approx(0.8, abs=0.01)
        assert descriptor.integral_b_t_m > 0.0


def test_constant_tolerance_chained_and_monotonic_profiles_are_typed() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    constant = describe_profile(profile(z, (1.0,) * len(z)))
    assert constant.topology_status is TopologyStatus.DEGENERATE
    assert constant.nulls == constant.extrema == ()
    chained = describe_profile(
        profile(z, (3.0, 1.0, 1.08, 1.16, 2.0, 2.5, 3.0)),
        TopologyPolicy(
            absolute_value_tolerance_t=0.1,
            relative_value_tolerance=0.0,
        ),
    )
    assert chained.topology_status in {
        TopologyStatus.NO_TOPOLOGY,
        TopologyStatus.AMBIGUOUS,
    }
    monotonic = describe_profile(profile(z, tuple(float(i) for i in range(1, 8))))
    assert monotonic.topology_status is TopologyStatus.NO_TOPOLOGY
    assert monotonic.nulls == monotonic.extrema == ()
    assert {item.kind for item in monotonic.boundary_extrema} == {
        CandidateKind.BOUNDARY_MINIMUM,
        CandidateKind.BOUNDARY_MAXIMUM,
    }


def test_same_sign_huge_values_are_not_nulls() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    sampled = profile(
        z,
        (1.0e308, 9.0e307, 1.0e308, 9.0e307, 1.0e308, 9.0e307, 1.0e308),
    )
    assert locate_nulls(sampled) == ()


def test_opposite_sign_huge_values_use_overflow_safe_root_interpolation() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    sampled = profile(
        z,
        (
            -1.0e308,
            1.0e308,
            1.0e308,
            1.0e308,
            1.0e308,
            1.0e308,
            1.0e308,
        ),
    )
    nulls = locate_nulls(sampled)
    assert len(nulls) == 1
    assert nulls[0].z_m == pytest.approx(-2.5)
    assert nulls[0].b_magnitude_t == 0.0
    with pytest.raises(TopologyResolutionError, match="tolerance overflowed"):
        locate_nulls(
            sampled,
            TopologyPolicy(null_relative_tolerance=1.0e308),
        )


def test_sub_nanotesla_ten_percent_difference_is_not_a_tie() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    sampled = profile(
        z,
        (3.0e-9, 1.0e-9, 3.0e-9, 4.0e-9, 3.0e-9, 1.1e-9, 3.0e-9),
    )
    selected = locate_extrema(
        sampled, TopologyPolicy(tie_policy=TiePolicy.HIGHEST_CONFIDENCE)
    )
    assert len([item for item in selected if item.kind is CandidateKind.MINIMUM]) == 2


def test_ripple_below_uncertainty_is_preserved_as_ambiguous() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    sampled = profile(
        z,
        (1.0, 0.99, 1.0, 0.99, 1.0, 0.99, 1.0),
        sigma=(0.02,) * len(z),
    )
    descriptor = describe_profile(sampled)
    assert descriptor.topology_status is TopologyStatus.AMBIGUOUS
    assert descriptor.extrema
    assert max(item.confidence for item in descriptor.extrema) < 0.5
