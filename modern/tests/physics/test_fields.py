from decimal import Decimal, localcontext
from math import copysign, inf, isfinite, nan
from sys import float_info

import pytest

from cft_revival.physics import PhysicsValidationError, UniformAxialFieldFixture


def test_uniform_axial_field_matches_vector_potential_curl() -> None:
    fixture = UniformAxialFieldFixture(0.23)
    for radius_m in (0.0, 1.0e-12, 1.0e-5, 0.01, 1.0):
        field = fixture.magnetic_field(radius_m, axial_m=-0.4)
        assert fixture.vector_potential_phi_t_m(radius_m) == 0.5 * 0.23 * radius_m
        assert fixture.axial_field_from_cylindrical_curl(radius_m) == 0.23
        assert field.radial_t == 0.0
        assert field.axial_t == 0.23


def test_uniform_field_is_regular_on_axis() -> None:
    fixture = UniformAxialFieldFixture(-0.5)
    assert fixture.axis_regularity_residual_t_m() == 0.0
    assert fixture.vector_potential_phi_t_m(0.0) == 0.0
    assert fixture.axial_field_from_cylindrical_curl(0.0) == -0.5


@pytest.mark.parametrize("value", [-1.0, nan, inf])
def test_field_fixture_rejects_invalid_coordinates(value: float) -> None:
    fixture = UniformAxialFieldFixture(1.0)
    with pytest.raises(PhysicsValidationError):
        fixture.magnetic_field(value)
    with pytest.raises(PhysicsValidationError):
        fixture.vector_potential_phi_t_m(value)


@pytest.mark.parametrize("value", [nan, inf])
def test_field_fixture_rejects_nonfinite_strength(value: float) -> None:
    with pytest.raises(PhysicsValidationError):
        UniformAxialFieldFixture(value)


def test_vector_potential_rejects_finite_overflow() -> None:
    fixture = UniformAxialFieldFixture(float_info.max)
    assert isfinite(fixture.vector_potential_phi_t_m(1.0))
    with pytest.raises(PhysicsValidationError, match="vector_potential"):
        fixture.vector_potential_phi_t_m(float_info.max)


@pytest.mark.parametrize(
    ("b0_t", "radius_m"),
    [
        (float_info.min * float_info.epsilon, float_info.max),
        (float_info.max, float_info.min * float_info.epsilon),
        (-(float_info.min * float_info.epsilon), float_info.max),
        (0.23, 0.017),
    ],
)
def test_vector_potential_scaled_symmetry_matches_decimal_oracle(
    b0_t: float, radius_m: float
) -> None:
    with localcontext() as context:
        context.prec = 2000
        expected = float(
            Decimal.from_float(b0_t)
            * Decimal.from_float(radius_m)
            / Decimal(2)
        )
    actual = UniformAxialFieldFixture(b0_t).vector_potential_phi_t_m(radius_m)
    assert actual == expected
    assert isfinite(actual)


def test_vector_potential_preserves_representable_subnormal() -> None:
    minimum_subnormal = float_info.min * float_info.epsilon
    fixture = UniformAxialFieldFixture(2.0 * minimum_subnormal)
    assert fixture.vector_potential_phi_t_m(1.0) == minimum_subnormal


def test_minimum_subnormal_times_maximum_radius_regression() -> None:
    minimum_subnormal = float_info.min * float_info.epsilon
    result = UniformAxialFieldFixture(
        minimum_subnormal
    ).vector_potential_phi_t_m(float_info.max)
    assert result == 4.4408920985006257e-16


def test_vector_potential_canonicalizes_zero_sign() -> None:
    result = UniformAxialFieldFixture(-1.0).vector_potential_phi_t_m(0.0)
    assert result == 0.0
    assert copysign(1.0, result) == 1.0
