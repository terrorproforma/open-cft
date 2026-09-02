"""Tests for the PIC-2D snapshot dashboard generator (skipped until results exist)."""

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
GENERATOR_PATH = MODERN / "visualization" / "generate_pic2d_cft_snapshot.py"
CHECKED_HTML = MODERN / "visualization" / "pic2d-cft-snapshot.html"
RESULTS = MODERN / "experiments" / "pic2d_cft_snapshot_v2" / "results"
HISTORY_RESULTS = MODERN / "experiments" / "pic2d_cft_snapshot_v1" / "results"


def _load_generator():
    spec = importlib.util.spec_from_file_location("pic2d_snapshot_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()
pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="snapshot results are not materialised")


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_payload_is_hash_bound_and_claim_bounded(payload) -> None:
    assert payload["schema"] == GENERATOR.SCHEMA
    assert payload["status"] == "development_screening_not_preregistered"
    assert "not preregistered" in payload["claim_statement"].lower()
    assert "not validated" in payload["claim_statement"].lower()
    assert len(payload["simplifications"]) >= 8
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in payload["cases"]] == list(manifest["cases"])
    for case in payload["cases"]:
        assert case["summary_sha256"] == manifest["cases"][case["id"]]["summary_sha256"]
        assert case["stop_reason"] in GENERATOR.STOP_REASONS
        assert len(case["maps"]["n_e_per_m3"]) == len(case["grid_r_m"])
        assert case["series"]["time_s"] and case["series"]["electrons"]
        assert case["plateau"] is not None and case["ledger"] is not None
        assert case["series"]["interval_electrode_work_j"]
    assert payload["budget"]["n_max_per_m3"] > 0
    GENERATOR.validate_payload(payload)


@pytest.mark.skipif(not (HISTORY_RESULTS / "manifest.json").is_file(), reason="v1 history results are not materialised")
def test_history_panel_keeps_v1_fail_closed_cases(payload) -> None:
    history = payload["history"]
    assert history is not None
    v1 = json.loads((HISTORY_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in history["cases"]] == list(v1["cases"])
    assert all(case["stop_reason"] == "runtime_stability_gate_stopped_run" for case in history["cases"])
    assert "fail-closed" in history["lesson"]
    html = GENERATOR.render_html(payload)
    assert 'id="history"' in html and 'id="budget"' in html


def test_generation_is_byte_deterministic_and_checked_html_is_current(payload, tmp_path: Path) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "pic2d-cft-snapshot.html"
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
        "window.devicePixelRatio", "createImageData", 'id="wall"', 'id="exit"', 'id="energy"', 'id="convergence"',
        "Claim boundary", "one-cell stair-step", "reported, not hidden",
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
    changed = deepcopy(payload)
    changed["status"] = "accepted"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["cases"][0]["summary_sha256"] = "abc"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["cases"][0]["maps"]["phi_v"].pop()
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(changed)
    if payload["history"] is not None:
        changed = deepcopy(payload)
        changed["history"]["cases"][0]["stop_reason"] = "converged"
        with pytest.raises(ValueError):
            GENERATOR.validate_payload(changed)


def test_manifest_tampering_is_detected(tmp_path: Path) -> None:
    target = tmp_path / "results"
    shutil.copytree(RESULTS, target)
    manifest = target / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(Exception):
        GENERATOR.build_payload(target)
