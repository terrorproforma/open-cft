"""PIC model-to-model context extraction and the committed verification record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
REPO = MODERN.parent
RECORD = MODERN / "docs" / "workstreams" / "plasma-v2-verification.json"
SPEC = MODERN / "spec" / "plasma_v2" / "four-cell-sheath-closure-v2.json"
MAPS = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results" / "maps.npz"
SUMMARY = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results" / "summary.json"
MANIFEST = REPO / "paper" / "evidence" / "manifests" / "four-cell-closure.json"


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_pic_plateau_context_matches_the_run_summary() -> None:
    pytest.importorskip("numpy")
    if not MAPS.is_file() or not SUMMARY.is_file():
        pytest.skip("pic2d steady-state v2 artifacts are not present")
    from cft_revival.plasma_v2.pic_context import CUSP_PLANES_Z_M, load_pic_plateau_context

    context = load_pic_plateau_context(MAPS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    window = summary["window_currents_a"]
    # Integrated wall currents reproduce the run's own window means to < 1 % (area discretisation).
    assert context.total_wall_electron_current_a == pytest.approx(window["wall_electron_a"], rel=0.01)
    assert context.total_wall_ion_current_a == pytest.approx(window["wall_ion_a"], rel=0.01)
    assert len(context.cusps) == 3 and len(context.segments) == 4
    assert tuple(cusp.z_m for cusp in context.cusps) == CUSP_PLANES_Z_M
    for cusp in context.cusps:
        # Floating dielectric: electron and ion currents balance locally at every cusp.
        assert cusp.electron_wall_current_a == pytest.approx(cusp.ion_wall_current_a, rel=0.01)
        assert cusp.sheath_drop_v > 0.0 and cusp.near_wall_drop_v > 0.0
        assert 0.0 < cusp.near_wall_electron_temperature_ev < cusp.axis_electron_temperature_ev
    assert context.phi_max_v == pytest.approx(summary["window_maps_summary"]["phi_max_v"])
    assert all(step > 0.0 for step in context.potential_steps_v)  # staircase, not a flat interior


def test_verification_record_exists_and_is_self_consistent() -> None:
    assert RECORD.is_file(), "run python -m cft_revival.plasma_v2.verification from modern/"
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["document_type"] == "plasma-v2-verification-record"
    assert record["status"].startswith("DEVELOPMENT")
    structure = record["structure"]
    assert structure["v1_row_parity_max_abs_difference"] <= 1.0e-9
    assert structure["corrected_r27_identity_max_relative"] <= 1.0e-12
    assert structure["v1_r27_min_relative_on_same_states"] > 0.0
    for report in structure["ranks"].values():
        assert (report["rank_corrected_core"], report["rank_with_sheath_and_anode"], report["rank_full"]) == (21, 28, 31)
    summary = record["closure_grid"]["summary"]
    no_emission = summary["CL-3-sheath-limited | floating_no_emission"]
    assert no_emission["closed"] == 0
    assert no_emission["manifold_root_blocked_only_by_cusp_energy_margin"] == no_emission["cases"] == 80
    kornfeld_scl = summary["CL-1-declared (Kornfeld DM9.2 p) | space_charge_limited"]
    assert kornfeld_scl["closed"] == kornfeld_scl["cases"] == 16
    scl = summary["CL-3-sheath-limited | space_charge_limited"]
    assert scl["cases"] == 80 and scl["closed"] >= 70
    for case in record["closure_grid"]["cases"]:
        if case["converged"]:
            assert case["residual_inf_norm"] <= 1.0e-9
            assert case["jacobian_rank"] == 31
            assert case["classification"] == "closed"
        else:
            assert case["classification"] != "closed"
    binding = record["v1_package_binding"]
    assert binding["all_match"] is True


def test_v1_package_on_disk_equals_the_paper_manifest_blobs() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["executed_package"]["files"]:
        assert _lf_sha256(REPO / entry["path"]) == entry["sha256_lf"], entry["path"]


def test_spec_document_matches_the_package() -> None:
    from cft_revival.plasma_v2 import (
        FLOATING_SHEATH_COEFFICIENT,
        MASS_FLUX_RATIO,
        RESIDUAL_SIZE,
        STATE_SIZE,
        SPACE_CHARGE_LIMITED_COEFFICIENT,
    )

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["status"] == "DEVELOPMENT_NOT_ACCEPTED"
    assert spec["state_layout"]["size"] == STATE_SIZE
    assert len(spec["residual_rows"]) == RESIDUAL_SIZE
    assert [row["id"] for row in spec["residual_rows"]] == [f"R{i:02d}" for i in range(RESIDUAL_SIZE)]
    constants = spec["constants"]
    assert constants["mass_flux_ratio_K0"] == pytest.approx(MASS_FLUX_RATIO)
    assert constants["floating_sheath_coefficient"] == pytest.approx(FLOATING_SHEATH_COEFFICIENT)
    assert constants["space_charge_limited_coefficient"] == SPACE_CHARGE_LIMITED_COEFFICIENT
    assert spec["rank"]["corrected_core_rank"] == 21
    assert spec["rank"]["rank_with_sheath_and_anode_rows"] == 28
    assert spec["rank"]["full_rank_with_declared_potential_rows"] == 31
    assert spec["v1_package_policy"]["modified"] is False
