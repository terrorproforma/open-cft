from math import copysign, isclose

import pytest

from cft_revival.kernels import (
    calculate_performance,
    cusp_arrival_probabilities,
    cusp_arrival_probability,
    cusp_arrival_probability_python,
    legacy_cusp_fields,
)
from cft_revival.models import (
    DesignPoint,
    FieldProfile,
    PlasmaSolution,
    ValidationError,
)


def test_probability_matches_closed_form_of_matlab_integral() -> None:
    assert isclose(cusp_arrival_probability(0.2, 1.0), 0.05278640450004207)
    assert cusp_arrival_probability(0.0, 1.0) == 0.0
    assert cusp_arrival_probability(1.0, 1.0) == 0.5


def test_probability_canonicalizes_negative_zero() -> None:
    for implementation in (
        cusp_arrival_probability_python,
        cusp_arrival_probability,
    ):
        result = implementation(-0.0, 1.0)
        assert result == 0.0
        assert copysign(1.0, result) == 1.0


def test_probability_is_stable_for_tiny_ratio() -> None:
    expected = 2.5e-19
    assert isclose(
        cusp_arrival_probability_python(1.0e-18, 1.0),
        expected,
        rel_tol=1.0e-15,
        abs_tol=0.0,
    )
    assert isclose(
        cusp_arrival_probability(1.0e-18, 1.0),
        expected,
        rel_tol=1.0e-15,
        abs_tol=0.0,
    )
    assert cusp_arrival_probability_python(2.0e-323, 1.0) == 5.0e-324


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (2.0, 1.0),
        (float("inf"), 1.0),
        (1.0, float("inf")),
        (float("-inf"), 1.0),
    ],
)
def test_probability_rejects_invalid_fields(low: float, high: float) -> None:
    with pytest.raises((ValidationError, ValueError)):
        cusp_arrival_probability(low, high)


def test_native_probability_is_stable_when_extension_is_available() -> None:
    try:
        from cft_revival import _native
    except ImportError:
        pytest.skip("optional pybind11 extension is not built")
    assert isclose(
        _native.cusp_arrival_probability(1.0e-18, 1.0),
        2.5e-19,
        rel_tol=1.0e-15,
        abs_tol=0.0,
    )
    zero = _native.cusp_arrival_probability(-0.0, 1.0)
    assert zero == 0.0
    assert copysign(1.0, zero) == 1.0


def test_legacy_windows_and_probability_order_are_explicit() -> None:
    positions = (0.25, 4.7, 15.3, 20.5, 24.5)
    centre = FieldProfile(positions, (1.0, 2.0, 3.0, 4.0, 4.0))
    wall = FieldProfile(positions, (10.0, 20.0, 30.0, 40.0, 50.0))
    low, high = legacy_cusp_fields(centre, wall)
    assert low == (2.0, 2.0, 3.0, 4.0)
    assert high == (1.0, 20.0, 30.0, 40.0)
    # The unexplained cusp-4 swap can invert B_low/B_high; reject instead of
    # silently creating the complex MATLAB result.
    with pytest.raises((ValidationError, ValueError)):
        cusp_arrival_probabilities(low, high)


def test_performance_translation_uses_si_units() -> None:
    design = DesignPoint.from_sequence((300.0, 1.0, 10.0, 3.0, 8.0, 12.0, 20.0, 30.0))
    values = [0.0] * 30
    values[0] = 150.0
    values[5] = 100.0
    values[6] = 200.0
    result = calculate_performance(
        design,
        PlasmaSolution(tuple(values), converged=True, residual_norm=0.0, provenance="test"),
    )
    assert 0.0 < result.total_efficiency < 1.0
    assert result.anode_power_w == 300.0
    assert result.thrust_n > 0.0
    assert result.specific_impulse_s > 0.0
