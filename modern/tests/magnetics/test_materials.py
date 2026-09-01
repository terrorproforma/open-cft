import json
import math
import random
from fractions import Fraction

import pytest

from cft_revival.magnetics import (
    MU0_H_PER_M,
    ExtrapolationPolicy,
    LinearPermeability,
    MagneticsValidationError,
    TabulatedBHCurve,
    VectorRZ,
    canonical_json,
    checked_synthetic_smco_like_magnet,
    checked_synthetic_soft_magnetic_curve,
)


def test_linear_permeability_has_si_analytic_limits_and_energy() -> None:
    vacuum = LinearPermeability("vacuum", 1.0)
    field = 125_000.0
    flux = MU0_H_PER_M * field
    assert vacuum.permeability_h_per_m == MU0_H_PER_M
    assert vacuum.b_from_h_t(field) == pytest.approx(flux)
    assert vacuum.h_from_b_a_per_m(flux) == pytest.approx(field)
    assert vacuum.differential_permeability_h_per_m(field) == MU0_H_PER_M
    assert vacuum.secant_permeability_h_per_m(0.0) == MU0_H_PER_M
    assert vacuum.coenergy_density_j_per_m3(field) == pytest.approx(0.5 * field * flux)
    assert vacuum.energy_density_j_per_m3(flux) == pytest.approx(0.5 * field * flux)


def test_pchip_interpolates_knots_and_is_odd_symmetric() -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    for field, expected_flux in zip(curve.h_a_per_m, curve.b_t):
        assert curve.b_from_h_t(field) == pytest.approx(expected_flux, abs=2.0e-15)
        assert curve.b_from_h_t(-field) == pytest.approx(-expected_flux, abs=2.0e-15)
        assert curve.h_from_b_a_per_m(expected_flux) == pytest.approx(
            field, rel=2.0e-12, abs=2.0e-12
        )


def test_pchip_is_dense_monotone_with_positive_differential_permeability() -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    maximum = curve.h_a_per_m[-1]
    fields = tuple(maximum * index / 2000.0 for index in range(2001))
    fluxes = tuple(curve.b_from_h_t(field) for field in fields)
    derivatives = tuple(
        curve.differential_permeability_h_per_m(field) for field in fields
    )
    assert all(right > left for left, right in zip(fluxes, fluxes[1:]))
    assert min(derivatives) > 0.0


@pytest.mark.parametrize("field", [50.0, 200.0, 650.0, 2_000.0, 8_000.0, 55_000.0])
def test_pchip_reported_derivative_matches_central_difference(field: float) -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    step = max(1.0e-4, abs(field) * 1.0e-6)
    finite_difference = (
        curve.b_from_h_t(field + step) - curve.b_from_h_t(field - step)
    ) / (2.0 * step)
    assert curve.differential_permeability_h_per_m(field) == pytest.approx(
        finite_difference, rel=2.0e-7
    )


def test_secant_and_differential_permeability_are_distinct_for_nonlinear_curve() -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    field = 10_000.0
    assert curve.secant_permeability_h_per_m(field) == pytest.approx(
        curve.b_from_h_t(field) / field
    )
    assert curve.secant_permeability_h_per_m(field) != pytest.approx(
        curve.differential_permeability_h_per_m(field), rel=1.0e-2
    )
    assert curve.secant_permeability_h_per_m(0.0) == curve.knot_derivatives_h_per_m[0]


def test_energy_coenergy_are_consistent_and_nonnegative() -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    for field in (0.0, 50.0, 1_500.0, 20_000.0, 120_000.0, -20_000.0):
        flux = curve.b_from_h_t(field)
        coenergy = curve.coenergy_density_j_per_m3(field)
        energy = curve.energy_density_j_per_m3(flux)
        assert coenergy >= 0.0
        assert energy >= -2.0e-10
        assert energy + coenergy == pytest.approx(field * flux, rel=2.0e-12, abs=2.0e-12)

    field = 7_500.0
    step = 1.0e-2
    derivative = (
        curve.coenergy_density_j_per_m3(field + step)
        - curve.coenergy_density_j_per_m3(field - step)
    ) / (2.0 * step)
    assert derivative == pytest.approx(curve.b_from_h_t(field), rel=2.0e-8)


def test_extrapolation_policy_is_explicit_and_invertible() -> None:
    rejecting = TabulatedBHCurve(
        material_id="two-point",
        h_a_per_m=(0.0, 100.0),
        b_t=(0.0, 0.1),
        extrapolation=ExtrapolationPolicy.ERROR,
        provenance="test synthetic",
        is_synthetic=True,
    )
    with pytest.raises(MagneticsValidationError, match="exceeds tabulated"):
        rejecting.b_from_h_t(101.0)
    with pytest.raises(MagneticsValidationError, match="exceeds tabulated"):
        rejecting.h_from_b_a_per_m(0.11)

    extending = checked_synthetic_soft_magnetic_curve()
    field = 130_000.0
    flux = extending.b_from_h_t(field)
    expected = extending.b_t[-1] + extending.knot_derivatives_h_per_m[-1] * 30_000.0
    assert flux == pytest.approx(expected)
    assert extending.h_from_b_a_per_m(flux) == pytest.approx(field)


@pytest.mark.parametrize(
    ("h_values", "b_values"),
    [
        ((1.0, 2.0), (0.0, 0.1)),
        ((0.0, 1.0, 1.0), (0.0, 0.1, 0.2)),
        ((0.0, 1.0, 2.0), (0.0, 0.2, 0.2)),
        ((0.0, math.inf), (0.0, 0.1)),
        ((0.0, 1.0), (0.0, math.nan)),
    ],
)
def test_invalid_b_h_data_are_rejected(
    h_values: tuple[float, ...], b_values: tuple[float, ...]
) -> None:
    with pytest.raises(MagneticsValidationError):
        TabulatedBHCurve(
            "invalid",
            h_values,
            b_values,
            provenance="test synthetic",
            is_synthetic=True,
        )


def test_mutable_samples_rejected_but_sharp_valid_curve_is_accepted() -> None:
    with pytest.raises(MagneticsValidationError, match="immutable tuples"):
        TabulatedBHCurve(  # type: ignore[arg-type]
            "mutable",
            [0.0, 1.0],
            [0.0, 1.0],
            provenance="test synthetic",
        )
    sharp = TabulatedBHCurve(
        "abrupt-saturation",
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 1.01),
        provenance="test synthetic",
    )
    fields = tuple(index / 100.0 for index in range(201))
    fluxes = tuple(sharp.b_from_h_t(field) for field in fields)
    assert all(right > left for left, right in zip(fluxes, fluxes[1:]))
    assert all(
        sharp.differential_permeability_h_per_m(field) > 0.0
        for field in fields
    )


def test_smco_like_temperature_coefficients_recoil_and_magnetization() -> None:
    magnet = checked_synthetic_smco_like_magnet()
    hot = magnet.reference_temperature_k + 100.0
    assert magnet.remanence_t(hot) == pytest.approx(
        magnet.remanence_ref_t * (1.0 + 100.0 * magnet.remanence_temp_coefficient_per_k)
    )
    assert magnet.intrinsic_coercivity_a_per_m(hot) == pytest.approx(
        magnet.intrinsic_coercivity_ref_a_per_m
        * (1.0 + 100.0 * magnet.coercivity_temp_coefficient_per_k)
    )
    field = -250_000.0
    assert magnet.recoil_b_parallel_t(field, hot) == pytest.approx(
        magnet.remanence_t(hot)
        + MU0_H_PER_M * magnet.recoil_relative_permeability * field
    )
    magnetization = magnet.magnetization_a_per_m(hot, VectorRZ(3.0, 4.0))
    assert magnetization.magnitude == pytest.approx(magnet.remanence_t(hot) / MU0_H_PER_M)
    assert magnetization.radial / magnetization.axial == pytest.approx(3.0 / 4.0)


def test_smco_like_rejects_out_of_range_and_nonfinite_inputs() -> None:
    magnet = checked_synthetic_smco_like_magnet()
    with pytest.raises(MagneticsValidationError, match="outside"):
        magnet.remanence_t(magnet.valid_temperature_max_k + 0.01)
    with pytest.raises(MagneticsValidationError):
        magnet.recoil_b_parallel_t(math.inf, magnet.reference_temperature_k)
    with pytest.raises(MagneticsValidationError, match="zero vector"):
        magnet.magnetization_a_per_m(
            magnet.reference_temperature_k, VectorRZ(0.0, 0.0)
        )
    with pytest.raises(MagneticsValidationError):
        LinearPermeability("bad", 0.0)


def test_finite_extremes_reject_only_nonrepresentable_requested_results() -> None:
    linear = LinearPermeability("linear", 1.0)
    with pytest.raises(MagneticsValidationError, match="computed coenergy"):
        linear.coenergy_density_j_per_m3(float.fromhex("0x1.fffffffffffffp+1023"))
    curve = TabulatedBHCurve(
        "overflowing-slope",
        (0.0, 1.0e-320),
        (0.0, 1.0e308),
        provenance="test synthetic",
        is_synthetic=True,
    )
    assert curve.b_from_h_t(1.0e-320) == 1.0e308
    with pytest.raises(MagneticsValidationError, match="representable float range"):
        curve.differential_permeability_h_per_m(1.0e-320)


@pytest.mark.parametrize(
    ("h_scale", "b_scale"),
    [
        (5.0e-324, 5.0e-324),
        (1.0e-150, 1.0e150),
        (1.0e150, 1.0e-150),
        (1.0e300, 1.0),
        (1.0, 1.0e300),
    ],
)
def test_normalized_curve_round_trip_and_energies_across_scales(
    h_scale: float, b_scale: float
) -> None:
    curve = TabulatedBHCurve(
        f"scaled-{h_scale}-{b_scale}",
        (0.0, h_scale, 2.0 * h_scale, 3.0 * h_scale),
        (0.0, b_scale, 2.0 * b_scale, 3.0 * b_scale),
        provenance="generated deterministic scale property",
        is_synthetic=True,
    )
    for index in range(1, 4):
        field = curve.h_a_per_m[index]
        flux = curve.b_from_h_t(field)
        recovered = curve.h_from_b_a_per_m(flux)
        assert recovered == field
        assert curve.differential_permeability_h_per_m(field) > 0.0
        assert curve.energy_density_j_per_m3(flux) >= 0.0
        assert curve.coenergy_density_j_per_m3(field) >= 0.0


def test_interval_local_inverse_is_ulp_accurate_near_knots() -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    for knot in curve.h_a_per_m[1:-1]:
        for field in (math.nextafter(knot, 0.0), math.nextafter(knot, math.inf)):
            recovered = curve.h_from_b_a_per_m(curve.b_from_h_t(field))
            assert abs(recovered - field) <= 16.0 * math.ulp(field)


def test_full_binary64_linear_inverse_preserves_subnormal_oracle_values() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    curve = TabulatedBHCurve(
        "full-binary64-linear",
        (0.0, maximum),
        (0.0, maximum),
        provenance="binary64 oracle property",
        is_synthetic=True,
    )
    probes = (
        5.0e-324,
        1.0e-323,
        math.ldexp(63.0, -1074),
        math.ldexp(1.0, -1022),
        1.0,
        math.nextafter(maximum, 0.0),
    )
    for flux in probes:
        oracle = float(
            Fraction.from_float(flux)
            * Fraction.from_float(maximum)
            / Fraction.from_float(maximum)
        )
        assert curve.h_from_b_a_per_m(flux) == oracle
        assert curve.h_from_b_a_per_m(-flux) == -oracle
        assert curve.b_from_h_t(oracle) == flux
    assert curve.h_from_b_a_per_m(5.0e-324) == 5.0e-324
    assert curve.h_from_b_a_per_m(1.0e-323) == 1.0e-323
    assert curve.energy_density_j_per_m3(5.0e-324) == 0.0
    assert curve.coenergy_density_j_per_m3(5.0e-324) == 0.0


def test_endpoint_near_nonlinear_inverse_keeps_monotone_physical_bracket() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    curve = TabulatedBHCurve(
        "full-range-nonlinear",
        (0.0, maximum / 2.0, maximum),
        (0.0, maximum / 4.0, maximum),
        provenance="binary64 endpoint bracket property",
        is_synthetic=True,
    )
    previous_h = 0.0
    for count in range(1, 33):
        flux = math.ldexp(float(count), -1074)
        field = curve.h_from_b_a_per_m(flux)
        assert field >= previous_h
        assert curve.b_from_h_t(field) == flux
        previous_h = field


def test_seeded_scaled_shape_properties() -> None:
    generator = random.Random(20260901)
    scale_pairs = (
        (1.0e-250, 1.0e-50),
        (1.0e-100, 1.0e100),
        (1.0, 1.0),
        (1.0e100, 1.0e-100),
        (1.0e250, 1.0e50),
    )
    for case in range(25):
        h_scale, b_scale = scale_pairs[case % len(scale_pairs)]
        h_shape = (0.0, 1.0, 2.0, 4.0, 8.0)
        rises = tuple(10.0 ** generator.uniform(-1.0, 1.0) for _ in range(4))
        b_shape = [0.0]
        for rise in rises:
            b_shape.append(b_shape[-1] + rise)
        curve = TabulatedBHCurve(
            f"seeded-property-{case}",
            tuple(value * h_scale for value in h_shape),
            tuple(value * b_scale for value in b_shape),
            provenance="seeded generated property test",
            is_synthetic=True,
        )
        previous_flux = -1.0
        for sample in range(1, 81):
            field = curve.h_a_per_m[-1] * sample / 80.0
            flux = curve.b_from_h_t(field)
            assert flux > previous_flux
            assert curve.differential_permeability_h_per_m(field) > 0.0
            recovered = curve.h_from_b_a_per_m(flux)
            assert abs(recovered - field) <= max(
                256.0 * math.ulp(field),
                1.0e-13 * abs(field),
            )
            assert curve.energy_density_j_per_m3(flux) >= 0.0
            assert curve.coenergy_density_j_per_m3(field) >= 0.0
            previous_flux = flux


def test_synthetic_examples_are_labelled_and_serialization_is_deterministic() -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    magnet = checked_synthetic_smco_like_magnet()
    assert curve.is_synthetic and magnet.is_synthetic
    assert "not measured" in curve.provenance
    assert "not a vendor grade" in magnet.provenance
    first = canonical_json(curve)
    second = canonical_json(curve)
    assert first == second
    assert json.loads(first)["hysteresis"] == "out_of_scope"
    assert "NaN" not in first and "Infinity" not in first
