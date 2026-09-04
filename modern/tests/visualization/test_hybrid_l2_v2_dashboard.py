"""Tests for the offline Hybrid L2 v2 dashboard (per-cell hybrid vs the PIC base plateau).

Every check reads the hash-bound artifacts independently of the generator; the tests skip until the
preregistered base case and its assessment exist.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_hybrid_l2_v2_dashboard.py"
CHECKED_HTML = MODERN / "visualization" / "hybrid-l2-v2.html"
EXPERIMENT = MODERN / "experiments" / "hybrid_l2_v2"
RESULTS = EXPERIMENT / "results"
PIC_V2 = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results"

pytestmark = pytest.mark.skipif(not (RESULTS / "assessment.json").is_file() or not (RESULTS / "summary.json").is_file(),
                                reason="the hybrid L2 v2 base case has not been assessed")


def _load_generator():
    spec = importlib.util.spec_from_file_location("hybrid_l2_v2_dashboard", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


@pytest.fixture(scope="module")
def html(payload):
    return GENERATOR.render_html(payload)


def _embedded_payload(text: str):
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_payload_traces_to_the_hash_bound_artifacts(payload):
    assessment = _json(RESULTS / "assessment.json")
    assert payload["identity"]["assessment_sha256"] == hashlib.sha256((RESULTS / "assessment.json").read_bytes()).hexdigest()
    assert payload["identity"]["pic_base_maps_sha256"] == _json(PIC_V2 / "maps.npz.sha256.json")["byte_sha256"]
    assert payload["identity"]["l2_base_maps_sha256"] == _json(RESULTS / "maps.npz.sha256.json")["byte_sha256"]
    assert payload["verdict"] == assessment["gate_l2"]["verdict"] in GENERATOR.VERDICTS
    assert payload["metrics"] == assessment["gate_l2"]["metrics"]
    assert payload["comparison"]["outside"] == assessment["code_comparison"]["outside"]
    protocol = _json(EXPERIMENT / "protocol.json")
    assert payload["closures"]["cusp_conductance_s"] == pytest.approx(protocol["closures"]["cusp_conductance_s"], rel=1e-5)  # payload rounds to 6 digits
    for name, case in payload["cases"].items():
        directory = RESULTS if name == "base" else EXPERIMENT / f"results-{name}"
        if case["finished"]:
            summary = _json(directory / "summary.json")
            assert case["stop_reason"] == summary["stop_reason"] and case["steps"] == summary["steps_completed"]
            assert case["summary_sha256"] == hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest()
        else:
            assert not (directory / "summary.json").is_file()


def test_comparison_statuses_reproduce_from_values(payload):
    for row in payload["comparison"]["comparisons"]:
        if row["tolerance"] is None or row["reference"] in (None, 0.0) or row["value"] is None:
            assert row["status"] == "not_compared"
        else:
            rel = (row["value"] - row["reference"]) / abs(row["reference"])
            assert row["status"] == ("within" if abs(rel) <= row["tolerance"] else "outside")
            assert row["relative_difference"] == pytest.approx(rel, rel=1e-5, abs=1e-9)
    compared = [r for r in payload["comparison"]["comparisons"] if r["status"] != "not_compared"]
    assert payload["comparison"]["compared"] == len(compared)
    assert payload["metrics"]["code_comparison_passed"] == (bool(compared) and not payload["comparison"]["outside"])


def test_rendered_html_is_deterministic_offline_and_checked_in(payload, html):
    assert html == GENERATOR.render_html(payload)
    body = html.split('<script id="payload"')[1]
    assert "http://" not in body and "https://" not in body
    assert 'id="jserrors"' in html
    embedded = _embedded_payload(html)
    assert embedded == json.loads(json.dumps(payload))
    assert CHECKED_HTML.is_file(), "regenerate the dashboard: python visualization/generate_hybrid_l2_v2_dashboard.py"
    checked = CHECKED_HTML.read_bytes().replace(b"\r\n", b"\n")
    assert checked == html.encode("utf-8")
    assert len(checked) <= GENERATOR.MAX_HTML_BYTES
    for token in ("Claim boundary", "development model - not validated", "GATE-L2", payload["verdict"], "Prohibited until GATE-L2"):
        assert token in html


def test_payload_validation_rejects_tampering(payload):
    tampered = json.loads(json.dumps(payload))
    tampered["verdict"] = "accepted" if payload["verdict"] != "accepted" else "not_evaluable"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    row = next(r for r in tampered["comparison"]["comparisons"] if r["status"] != "not_compared")
    row["status"] = "within" if row["status"] == "outside" else "outside"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["paper_admission"] = "admitted"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["metrics"]["uncertainty_components"] = ["input"]
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
