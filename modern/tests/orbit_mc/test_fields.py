from __future__ import annotations

import numpy as np
import pytest

from cft_revival.orbit_mc import OrbitValidationError, PsiBicubicField, compare_maps
from cft_revival.orbit_mc.verification import manufactured_interpolator


def test_flux_bicubic_reproduces_manufactured_divergence_free_field() -> None:
    field, report = manufactured_interpolator(8)
    assert report["psi_node_max_abs_wb"] < 4.0e-18
    assert report["br_max_abs_t"] < 1.0e-14
    assert report["bz_max_abs_t"] < 1.0e-14
    assert report["b_relative_rms"] < 1.0e-14
    assert field.field_cylindrical(0.0, 0.2)[0] == 0.0
    assert field.field_cylindrical(0.0, 0.2)[1] == pytest.approx(0.324, rel=2.0e-14)


def test_interpolant_is_c1_across_cell_boundaries() -> None:
    field, _ = manufactured_interpolator(10)
    radius = field.r_m[5]
    epsilon = 1.0e-10
    left = field.psi_gradient(float(radius-epsilon), 0.123)
    right = field.psi_gradient(float(radius+epsilon), 0.123)
    assert np.allclose(left, right, rtol=0.0, atol=2.0e-9)


def test_map_comparison_reports_resolution_and_common_domain_errors() -> None:
    coarse, _ = manufactured_interpolator(8)
    fine, _ = manufactured_interpolator(16)
    report = compare_maps(coarse, fine)
    assert report["sample_count"] == 17*31
    assert report["b_max_relative"] < 2.0e-14
    assert report["psi_max_abs_wb"] < 2.0e-15


def test_axis_and_nonfinite_contracts_fail_closed() -> None:
    r = np.linspace(0.0, 1.0, 5)
    z = np.linspace(-1.0, 1.0, 5)
    psi = np.zeros((5, 5))
    material = np.full((5, 5), "plasma", dtype=object)
    psi[0, 2] = 1.0
    with pytest.raises(OrbitValidationError, match="constant"):
        PsiBicubicField(
            r, z, psi, material_id=material, plasma_material_id="plasma"
        )
    psi[:] = 0.0
    with pytest.raises(OrbitValidationError, match="positive"):
        PsiBicubicField(
            r, z, psi, material_id=material, plasma_material_id="plasma"
        )


def test_certificate_tightness_is_reported_and_preflighted() -> None:
    field, _ = manufactured_interpolator(8)
    diagnostic = field.certificate_tightness
    assert diagnostic.preflight_passed
    assert diagnostic.minimum_ratio == 0.001
    assert 0.0 < diagnostic.dense_to_bound_ratio <= 1.0
    assert diagnostic.dense_diagnostic_max_b_t <= diagnostic.certified_max_b_t


def test_excessively_conservative_certificate_fails_before_orbits() -> None:
    b0 = 0.3
    curvature = 2.0
    r = np.linspace(0.0, 0.4, 9)
    z = np.linspace(-0.5, 0.5, 17)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = 0.5*b0*rr**2*(1.0+curvature*zz**2)
    material = np.full(psi.shape, "plasma", dtype=object)
    with pytest.raises(OrbitValidationError, match="NOT_EVALUATED"):
        PsiBicubicField(
            r,
            z,
            psi,
            material_id=material,
            plasma_material_id="plasma",
            minimum_certificate_tightness_ratio=0.99,
        )
