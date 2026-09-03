"""Plume dashboard generator (model v2.0): built on a tiny CPU run of the plume protocol under tmp_path."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from cft_revival.pic2d.fields import uniform_field_map
from cft_revival.pic2d.mcc import XenonCrossSections
from experiments.pic2d_cft_steady_state_v1 import run as runner

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_pic2d_cft_plume.py"
PLUME_PROTOCOL = MODERN / "experiments" / "pic2d_cft_plume_v1" / "protocol.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location("pic2d_plume_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> tuple[Path, Path]:
    """The real plume protocol shrunk to 0.25 mm cells, run 600 steps on the CPU with a uniform B field."""

    root = tmp_path_factory.mktemp("plume_dashboard")
    protocol = deepcopy(runner.load_protocol(PLUME_PROTOCOL))
    protocol["geometry"]["body_dielectric_radius_m"] = 0.0045
    protocol["case"].update({"radial_cells": 48, "axial_cells": 144, "macro_weight": 6.0e5})
    protocol["numerics"].update({"dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 100,
                                 "averaging_window_steps": 200, "frame_recorder": {"cadence_steps": 100, "precision": "float32"}})
    protocol["numerics"]["stability_reference"]["density_per_m3"] = 1.0e16
    protocol["operating_point"]["seed_plasma_density_per_m3"] = 5.0e15
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    config = runner.build_config(protocol, backend="cpu")
    results = root / "results"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=uniform_field_map(config.grid, 0.02),
                            cross_sections=XenonCrossSections.from_file(), max_steps=600, protocol_path=protocol_path, log=lambda _: None)
    return results, protocol_path


@pytest.fixture(scope="module")
def payload(tiny_run):
    results, protocol_path = tiny_run
    return GENERATOR.build_payload(results, protocol_path)


def test_payload_is_hash_bound_and_claim_bounded(payload, tiny_run) -> None:
    results, protocol_path = tiny_run
    assert payload["schema"] == GENERATOR.SCHEMA and payload["status"] == "development_screening_not_preregistered"
    statement = payload["claim_statement"].lower()
    for phrase in ("not preregistered", "not validated", "development numbers", "box size", "closure is a conservation check"):
        assert phrase in statement, phrase
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    case = payload["cases"][0]
    assert case["role"] == "headline" and case["protocol_sha256"] == summary["protocol_sha256"] == payload["protocol"]["file_sha256"]
    assert case["maps_npz_sha256"] == summary["artifacts"]["maps_npz_sha256"]
    assert case["plume"]["thrust_total_n"] == pytest.approx(case["plume"]["thrust_flux_n"] + case["plume"]["cold_gas_thrust_n"])
    assert payload["performance"]["thrust_total_n"] == case["plume"]["thrust_total_n"]
    assert payload["performance"]["specific_impulse_s"] is not None and payload["performance"]["anode_efficiency"] is not None
    for entry in payload["literature_context"]:
        assert "validation" not in entry["label"].lower() or "not" in entry["label"].lower() or "never" in entry["label"].lower()
    GENERATOR.validate_payload(payload)


def test_full_domain_maps_body_and_sampling_cover_the_l_shaped_grid(payload) -> None:
    case = payload["cases"][0]
    nr, nz = len(case["grid_r_m"]), len(case["grid_z_m"])
    assert nr == 49 and nz == 145    # 48 x 144 cells, 0.25 mm: 12 mm radius, 36 mm long (channel 24 + plume 12)
    body = case["body"]
    assert body["r_bore_m"] == pytest.approx(2e-3) and body["r_exit_m"] == pytest.approx(3e-3)
    assert body["r_body_dielectric_m"] == pytest.approx(4.5e-3) and body["r_plume_m"] == pytest.approx(12e-3)
    assert body["z_exit_m"] == pytest.approx(24e-3) and body["z_max_m"] == pytest.approx(36e-3)
    profile = np.asarray(body["wall_profile_zr_m"])
    assert profile[0].tolist() == [0.0, pytest.approx(2e-3)] and profile[-1][0] == pytest.approx(24e-3)
    assert np.all(np.diff(profile[:, 0]) >= 0) and np.all(np.diff(profile[:, 1]) >= 0)   # divergent cone: monotone stair-step
    # maps: the plume box is plasma (finite) outside the channel radius beyond the exit plane, the body is not
    n_e = case["maps"]["n_e_per_m3"]
    j_plume = nz - 2
    j_channel = nz // 3
    assert n_e[nr - 1][j_plume] is not None and n_e[nr - 1][j_channel] is None
    sampling = case["sampling"]
    assert sampling["electron_samples_source"] == "recorded" and sampling["min_samples_default"] == 20
    assert sampling["window_steps"] == 200 and sampling["window_s"] == pytest.approx(200 * 5e-12)
    assert len(sampling["electron_samples"]) == nr and len(sampling["ionization_events"][0]) == nz
    # the plume starts empty (channel-only seed): far fewer samples in the plume corner than in the channel
    channel_samples = [v for row in sampling["electron_samples"][:8] for v in row[:nz // 3] if v is not None]
    plume_corner = [v for row in sampling["electron_samples"][-8:] for v in row[-8:] if v is not None]
    assert max(channel_samples) > 10 * max(plume_corner + [0.0])


def test_histograms_axis_and_trajectories_are_consistent(payload) -> None:
    case = payload["cases"][0]
    h = case["histograms"]
    assert len(h["theta_centres_deg"]) == 90 and len(h["iedf_centres_ev"]) == 256
    assert h["theta_centres_deg"][0] == pytest.approx(0.5) and h["theta_centres_deg"][-1] == pytest.approx(89.5)
    cumulative = [v for v in h["ion_current_cumulative_fraction"] if v is not None]
    assert all(b >= a - 1e-12 for a, b in zip(cumulative, cumulative[1:]))
    assert len(case["axis"]["z_m"]) == len(case["grid_z_m"]) == len(case["axis"]["phi_v"])
    phi_axis = [v for v in case["axis"]["phi_v"] if v is not None]
    assert phi_axis[0] == pytest.approx(300.0, abs=1.0) and abs(phi_axis[-1]) < 1.0     # anode to the far plane
    tracks = case["trajectories"]
    assert 1 <= tracks["count"] <= GENERATOR.TRAJECTORY_COUNT and "not tracked particles" in tracks["method"]
    for track in tracks["tracks"]:
        z0, r0 = track["start_zr_m"]
        assert 0.0 <= r0 <= 12e-3 and 0.0 <= z0 <= 36e-3      # born in the plasma region (channel or plume ionisation cells)
        if z0 < 24e-3:
            assert r0 <= 3e-3 + 1e-9                          # inside the channel the birth cell must lie under the wall
        assert track["end"] in ("far field", "front face", "left the plasma region", "max_steps")
        assert track["final_energy_ev"] >= 0.0 and len(track["zr_m"]) <= GENERATOR.TRAJECTORY_POINTS + 1
    # a cold ion born in the channel falls through most of U_a in the mean field
    channel_born = [t["final_energy_ev"] for t in tracks["tracks"] if t["start_zr_m"][0] < 24e-3]
    if channel_born:
        assert max(channel_born) > 50.0


def test_html_is_deterministic_self_contained_and_accessible(payload, tmp_path: Path, tiny_run) -> None:
    results, protocol_path = tiny_run
    first = GENERATOR.render_html(payload)
    assert first == GENERATOR.render_html(GENERATOR.build_payload(results, protocol_path))
    output = tmp_path / "pic2d-cft-plume.html"
    GENERATOR.generate(output, results, protocol_path)
    assert output.read_text(encoding="utf-8") == first
    lowered = first.lower()
    for forbidden in ("fetch(", "xmlhttprequest", "websocket", "cdn"):
        assert forbidden not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", first, re.I)
    assert not re.search(r"\bhttps?://", first, re.I)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", first)
    for fragment in (
        'id="claim"', 'for="map"', 'for="scale"', 'for="bin"', 'for="minSamples"', 'id="tracks"', 'id="mapCaption"', 'id="performance"',
        'id="thrust"', 'id="closure"', 'id="theta"', 'id="iedf"', 'id="axis"', 'id="farfield"', 'id="literature"', "function drawOverlay",
        "function viewMatrix", "function paintView", "never a validation", "Claim boundary", "cathode emission annulus", "Maxwell stress",
        'tabindex="0"', 'role="img"', 'aria-live="polite"', "new ResizeObserver(schedule)", "createImageData",
    ):
        assert fragment in first, fragment
    assert "<svg" not in lowered
    match = re.search(r'<script id="pic2d-data" type="application/json">(.*?)</script>', first, re.DOTALL)
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(match.group(1), parse_constant=reject_constant) == payload
    node = shutil.which("node")
    if node is not None:
        scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", first, re.DOTALL)
        script = tmp_path / "plume.js"
        script.write_text(scripts[-1], encoding="utf-8")
        completed = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr


def test_tampering_and_protocol_drift_are_rejected(payload, tiny_run, tmp_path: Path) -> None:
    results, protocol_path = tiny_run
    for mutate in (
        lambda p: p.__setitem__("status", "validated"),
        lambda p: p.__setitem__("claim_statement", "Thrust prediction for the flight unit."),
        lambda p: p["cases"][0].__setitem__("plume", None),
        lambda p: p["cases"][0]["sampling"]["electron_samples"].pop(),
        lambda p: p["cases"][0]["body"].__setitem__("r_plume_m", 1e-3),
        lambda p: p["cases"][0]["plume_series"].pop("momentum_closure_fraction"),
        lambda p: p["literature_context"].append({"source": "x", "device": "y", "numbers": "z", "label": "validation"}),
    ):
        tampered = deepcopy(payload)
        mutate(tampered)
        with pytest.raises(ValueError):
            GENERATOR.validate_payload(tampered)
    drifted = tmp_path / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["numerics"]["dt_s"] = 4.9e-12
    drifted.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        GENERATOR.build_payload(results, drifted)
    # a channel-only results directory is refused (the plume dashboard needs the v2.0 geometry)
    channel = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results"
    if (channel / "summary.json").is_file():
        with pytest.raises(ValueError):
            GENERATOR.build_payload(channel, MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "protocol.json")
