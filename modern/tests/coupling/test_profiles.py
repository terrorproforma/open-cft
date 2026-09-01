from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from cft_revival.coupling import (
    CouplingValidationError,
    FieldProvenance,
    extract_profiles,
    interpolate_profile,
    stable_lerp,
    validate_axisymmetric_map,
)


@dataclass
class GenericMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


def validated_linear_map():
    z_m = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    r_m = (0.0, 1.0, 2.0)
    field = GenericMap(
        r_m=r_m,
        z_m=z_m,
        b_r_t=tuple(tuple(r + z for z in z_m) for r in r_m),
        b_z_t=tuple(tuple(2.0 * r - z for z in z_m) for r in r_m),
    )
    # Axis regularity applies to Br, so use an axis-regular analytic component.
    field.b_r_t = tuple(tuple(r * (1.0 + z) for z in z_m) for r in r_m)
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    return validate_axisymmetric_map(
        field,
        FieldProvenance("linear", "3" * 64, "4" * 64, now),
        reference_time_utc=now,
    )


def test_centreline_and_wall_are_extracted_at_physical_radii() -> None:
    centre, wall = extract_profiles(
        validated_linear_map(),
        1.5,
        absolute_uncertainty_t=0.01,
        relative_uncertainty=0.02,
    )
    assert centre.sampled_r_m == 0.0
    assert wall.sampled_r_m == 1.5
    assert wall.b_r_t[2] == pytest.approx(1.5)
    assert wall.b_z_t[2] == pytest.approx(3.0)
    br, bz, magnitude, independent_sigma, common_sigma = interpolate_profile(
        wall, 0.25
    )
    assert br == pytest.approx(1.875)
    assert bz == pytest.approx(2.75)
    assert magnitude > 0.0
    assert independent_sigma > 0.01
    assert common_sigma == 0.0


@pytest.mark.parametrize("radius", [0.0, -1.0, 2.1, float("nan")])
def test_wall_radius_edge_cases_fail_closed(radius: float) -> None:
    with pytest.raises(CouplingValidationError, match="wall_radius_m"):
        extract_profiles(validated_linear_map(), radius)


def test_convex_interpolation_handles_opposite_extreme_values() -> None:
    assert stable_lerp(1.0e308, -1.0e308, 0.5) == 0.0
    assert stable_lerp(5.0e-324, 1.0e-323, 0.5) > 0.0
