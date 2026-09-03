"""Koch rho and PPM descriptors on an analytic single-harmonic PPM field."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.cusp_topology_search_v3_1 import topology as T
from experiments.l1a_geometry_sweep_v3 import descriptors as DS
from experiments.l1a_geometry_sweep_v3 import experiment as E


def test_bessel_series_against_reference_values() -> None:
    assert DS.bessel_i0(1.0) == pytest.approx(1.2660658777520084, rel=1e-14)
    assert DS.bessel_i1(1.0) == pytest.approx(0.5651591039924850, rel=1e-14)
    assert DS.bessel_i1(2.0) == pytest.approx(1.5906368546373291, rel=1e-14)
    assert DS.bessel_i0(0.0) == 1.0 and DS.bessel_i1(0.0) == 0.0
    assert DS.bessel_i1(DS.X_STAR_HEMP_LIKE) == pytest.approx(1.5, rel=1e-12)
    assert DS.X_STAR_HEMP_LIKE == pytest.approx(1.937318, abs=1e-6)
    assert DS.i1_root(4.0) == pytest.approx(3.0130, abs=2e-4) and DS.i1_root(10.6) == pytest.approx(4.0909, abs=2e-4)
    prediction = DS.ppm_prediction(1.0)
    assert prediction["predicted_hemp_like"] is False and prediction["i1_over_i0_x_w"] < 1.0
    assert DS.ppm_prediction(2.0)["predicted_hemp_like"] is True


def _analytic_ppm_grid(pitch: float, wall: float, b1: float, z0: float, r_max: float, z_min: float, z_max: float, nr: int, nz: int) -> T.TracingGrid:
    kappa = math.pi / pitch
    r = np.linspace(0.0, r_max, nr + 1)
    z = np.linspace(z_min, z_max, nz + 1)
    R, Z = np.meshgrid(r, z, indexing="ij")
    i0 = np.vectorize(DS.bessel_i0)(kappa * R)
    i1 = np.vectorize(DS.bessel_i1)(kappa * R)
    psi = b1 * (R / kappa) * i1 * np.cos(kappa * (Z - z0))
    bz = b1 * i0 * np.cos(kappa * (Z - z0))
    br = b1 * i1 * np.sin(kappa * (Z - z0))
    return T.tracing_grid(r, z, psi, br, bz, wall)


@pytest.mark.parametrize("wall_mm", [1.5, 2.5])
def test_rho_equals_i1_on_an_analytic_single_harmonic_stack(wall_mm: float) -> None:
    pitch = 0.004
    wall = wall_mm * 1e-3
    centres = tuple(0.002 + index * pitch for index in range(5))  # axis maxima at the stage centres
    geometry = T.ChannelGeometry(wall_radius_m=wall, straight_z_min_m=0.0, straight_z_max_m=0.020, chamber_length_m=0.020, stage_pitch_m=pitch, stage_centres_m=centres, injector_length_m=0.0016)
    grid = _analytic_ppm_grid(pitch, wall, 0.2, centres[0], 0.003, -0.006, 0.026, 60, 320)
    value = E.protocol()
    policy = E.policy_from(value)
    window = T.axis_window(grid, geometry, policy)
    characterization = T.characterize_map(grid, geometry, policy, source_identity_sha256="0" * 64, minimum_certificate_tightness_ratio=1e-3, keep_paths=False, axis_window_m=window)
    descriptors = DS.design_descriptors(grid, geometry, characterization, policy, source_identity_sha256="0" * 64, minimum_certificate_tightness_ratio=1e-3, stage_count=5, with_profiles=True)
    x_w = math.pi * wall / pitch
    assert descriptors["x_w"] == pytest.approx(x_w)
    interior = [row for row in descriptors["cusps"] if 0.001 < row["z_c_m"] < 0.019]
    assert len(interior) == 4  # gaps at 4, 8, 12, 16 mm
    for row in interior:
        assert row["rho_conservative"] == pytest.approx(DS.bessel_i1(x_w), rel=0.02), row["cusp_id"]
        assert row["rho_downstream"] == pytest.approx(row["rho_upstream"], rel=0.02)
        assert row["rho_wall"] == pytest.approx(DS.bessel_i1(x_w) / DS.bessel_i0(x_w), rel=0.02)
        assert row["cusp_is_wall_maximum"] is False  # I_1 / I_0 < 1 for every x
        assert row["hemp_like_conservative"] == (DS.bessel_i1(x_w) >= 1.5 * 0.98)
    harmonics = descriptors["wall_harmonics"]
    assert harmonics["applies"] and harmonics["b3_over_b1"] < 0.01 and harmonics["b5_over_b1"] < 0.01
    assert harmonics["fit_rms_over_max"] < 0.01
    assert descriptors["profiles"] is not None and len(descriptors["profiles"]["z_m"]) == 241
    assert descriptors["predicted_hemp_like_i1"] == (x_w >= DS.X_STAR_HEMP_LIKE)


def test_resolution_sensitivity_reports_per_cusp_differences() -> None:
    accepted = {"cusps": [{"cusp_id": "c1", "rho_conservative": 1.0, "z_c_m": 0.001}], "hemp_like_all_cusps": False}
    refined = {"cusps": [{"cusp_id": "c1", "rho_conservative": 1.1, "z_c_m": 0.0011}], "hemp_like_all_cusps": False}
    report = DS.resolution_sensitivity(accepted, refined)
    assert report["comparable"] and report["max_relative_rho_difference"] == pytest.approx(0.1 / 1.1)
    assert report["hemp_like_flag_agrees"] is True
    assert DS.resolution_sensitivity(accepted, {"cusps": [], "hemp_like_all_cusps": False})["comparable"] is False
