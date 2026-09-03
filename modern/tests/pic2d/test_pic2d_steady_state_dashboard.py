"""Tests for the PIC-2D steady-state dashboard generator (skipped until results exist)."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_pic2d_cft_steady_state.py"
CHECKED_HTML = MODERN / "visualization" / "pic2d-cft-steady-state.html"
EXPERIMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v2"
RESULTS = EXPERIMENT / "results"


def _load_generator():
    spec = importlib.util.spec_from_file_location("pic2d_steady_state_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()
pytestmark = pytest.mark.skipif(not (RESULTS / "summary.json").is_file(), reason="steady-state v2 results are not materialised")


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_payload_is_hash_bound_and_claim_bounded(payload) -> None:
    assert payload["schema"] == GENERATOR.SCHEMA
    assert payload["status"] == "development_screening_not_preregistered"
    statement = payload["claim_statement"].lower()
    for phrase in ("not preregistered", "not validated", "single seed", "under-resolved"):
        assert phrase in statement, phrase
    assert len(payload["simplifications"]) >= 10
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    headline = payload["cases"][0]
    assert headline["role"] == "headline" and headline["id"] == summary["case"]["id"]
    assert headline["protocol_sha256"] == summary["protocol_sha256"] == payload["protocol"]["file_sha256"]
    assert headline["maps_npz_sha256"] == summary["artifacts"]["maps_npz_sha256"]
    assert headline["stop_reason"] == "plateau_reached_after_min_transit_times"
    assert headline["plateau"]["reached"] is True and headline["plateau"]["transit_times_elapsed"] >= 3
    for key in ("discharge_current_drift", "electron_count_drift", "neutral_density_drift"):
        assert abs(headline["plateau"][key]) < 0.05
    assert headline["neutral_inventory"]["cumulative_ledger_closure_relative_to_inventory"] < 1e-10
    assert headline["series"]["neutral_density_per_m3"] and headline["series"]["neutral_fixed_point_per_m3"]
    assert len(headline["maps"]["n_e_per_m3"]) == len(headline["grid_r_m"])
    assert len(headline["axial_peak_n_e_per_m3"]) == len(headline["grid_z_m"])
    assert headline["cusps"]["cusp_z_m"], "cusp planes must be located from the P2 field map"
    assert headline["resolvability_at_peak"]["dz_over_lambda_d_at_peak"] > 0
    assert {v["name"] for v in payload["variants"]} == {"seed-b", "w-half"}
    GENERATOR.validate_payload(payload)


def test_history_panels_keep_predecessors(payload) -> None:
    history = payload["history"]
    labels = [row["label"] for row in history["steady_state"]]
    assert any("v1.2 reference" in label for label in labels)
    assert any("attempt 1" in label for label in labels)
    for row in history["steady_state"]:
        assert row["plateau"]["reached"] is False
        assert len(row["summary_sha256"]) == 64
    assert history["snapshot_v2"] is not None and len(history["snapshot_v2"]["cases"]) == 4
    assert history["snapshot_v1"] is not None and "fail-closed" in history["snapshot_v1"]["lesson"]
    html = GENERATOR.render_html(payload)
    assert 'id="history"' in html and 'id="budget"' in html and 'id="verification"' in html


def test_generation_is_byte_deterministic_and_checked_html_is_current(payload, tmp_path: Path) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "pic2d-cft-steady-state.html"
    GENERATOR.generate(output)
    assert output.read_text(encoding="utf-8") == first
    assert CHECKED_HTML.read_text(encoding="utf-8") == first


def test_html_is_self_contained_offline(payload) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    assert '<script id="pic2d-data" type="application/json">' in html
    for forbidden in ("fetch(", "xmlhttprequest", "websocket", "cdn"):
        assert forbidden not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", html, re.I)
    assert not re.search(r"\bhttps?://", html, re.I)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", html)


def test_html_has_claim_panel_controls_and_accessibility(payload) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        'id="claim"', 'for="case"', 'for="map"', 'for="scale"', 'id="theme"', 'tabindex="0"', 'role="img"',
        'aria-live="polite"', 'e.key==="ArrowLeft"', 'e.key==="Home"', "new ResizeObserver(schedule)",
        "window.devicePixelRatio", "createImageData", 'id="wall"', 'id="exit"', 'id="energy"', 'id="neutral"', 'id="rates"',
        'id="axial"', 'id="convergence"', "Claim boundary", "one-cell stair-step", "reported, not hidden", "cusp planes",
    ):
        assert fragment in html, fragment
    assert "<svg" not in html.lower()


def test_embedded_json_round_trips_strictly(payload) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(r'<script id="pic2d-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(match.group(1), parse_constant=reject_constant) == payload


def test_javascript_is_valid_when_node_is_available(payload, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available for JavaScript syntax checking")
    html = GENERATOR.render_html(payload)
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    script = tmp_path / "pic2d.js"
    script.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_tampered_payload_is_rejected(payload) -> None:
    for mutate in (
        lambda p: p.__setitem__("status", "accepted"),
        lambda p: p["cases"][0].__setitem__("summary_sha256", "abc"),
        lambda p: p["cases"][0]["maps"]["phi_v"].pop(),
        lambda p: p["cases"][0].__setitem__("protocol_sha256", "0" * 64),
        lambda p: p["cases"][0].__setitem__("stop_reason", "converged"),
        lambda p: p["history"]["steady_state"][0].__setitem__("stop_reason", "converged"),
        lambda p: p["variants"][0].__setitem__("state", "done"),
        lambda p: p.__setitem__("claim_statement", "validated steady state"),
    ):
        changed = deepcopy(payload)
        mutate(changed)
        with pytest.raises(ValueError):
            GENERATOR.validate_payload(changed)


def test_protocol_drift_and_tampered_results_are_rejected(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    shutil.copytree(RESULTS, experiment / "results", ignore=shutil.ignore_patterns("checkpoint*", "*.jsonl", "*.log", "*.err", "*.pid"))
    for name in ("protocol.json", "variants.json"):
        shutil.copy(EXPERIMENT / name, experiment / name)
    # a protocol file that no longer matches the hash recorded by the run is rejected
    protocol = experiment / "protocol.json"
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol drift"):
        GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
    shutil.copy(EXPERIMENT / "protocol.json", protocol)
    GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
    # a tampered summary is rejected by its sidecar
    summary = experiment / "results" / "summary.json"
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
