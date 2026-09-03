"""Tests for the offline L1a geometry sweep v3 dashboard (HEMP-like regime campaign).

Every check reads the sealed bundle independently of the generator and compares the embedded
payload against it, so the dashboard cannot show a number that does not trace to a hash-bound
artifact. The tests skip when the v3 campaign has not executed yet.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_l1a_geometry_sweep_v3_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "l1a-geometry-sweep-v3.template.html"
CHECKED_HTML = MODERN / "visualization" / "l1a-geometry-sweep-v3.html"
EXPERIMENT = MODERN / "experiments" / "l1a_geometry_sweep_v3"
RESULTS = EXPERIMENT / "results"
CLASSIFICATION = "L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID"

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="the sweep-v3 campaign has not executed")


def _load_generator():
    spec = importlib.util.spec_from_file_location("l1a_geometry_sweep_v3_dashboard", GENERATOR_PATH)
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


@pytest.fixture(scope="module")
def dataset():
    return _json(RESULTS / "artifacts" / "sweep-dataset.json")


def _embedded_payload(text: str):
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, re.S)
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_bundle_is_byte_verified_and_payload_traces_to_it(payload, dataset):
    manifest = _json(RESULTS / "manifest.json")
    assert manifest["state"] == "accepted_result"
    for entry in manifest["artifacts"]:
        if entry["type"] == "file":
            raw = (RESULTS / entry["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == entry["byte_sha256"], entry["path"]
    assert payload["identity"]["manifest_file_sha256"] == hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest()
    assert payload["classification"] == CLASSIFICATION == dataset["classification"]
    assert payload["headline"] == dataset["headline"]
    assert len(payload["designs"]) == dataset["design_count"] == 224
    rows = {row["design_id"]: row for row in dataset["designs"]}
    for item in payload["designs"]:
        row = rows[item["id"]]
        assert item["rho"] == [r["rho_conservative"] for r in row["rho"]]
        assert item["hemp"] == row["hemp_like_all_cusps"] and item["x_w"] == row["x_w"]
        assert item["cusps"] == row["wall_cusp_count"] and item["stable"] == row["stability"]["stable"]


def test_scatter_and_curve_reproduce_from_the_rows(payload, dataset):
    points = sum(row["wall_cusp_count"] for row in dataset["designs"])
    assert len(payload["scatter"]) == points
    for point in payload["curve"]:
        assert point["i1"] == pytest.approx(GENERATOR._bessel_i(1, point["x"]), rel=1e-12)
        assert point["i1_i0"] < 1.0
    assert payload["x_star"] == pytest.approx(1.937318, abs=1e-6)
    assert payload["v2_x_range"] == pytest.approx([math.pi * 0.0014 / 0.0065, math.pi * 0.0022 / 0.0038])
    sobol = [row for row in dataset["designs"] if row["set_id"] == "sobol_v3"]
    assert sum(item["hemp"] for item in payload["designs"] if item["set"] == "sobol_v3") == dataset["headline"]["sobol_hemp_like_count"]
    assert sum(1 for row in sobol if row["five_stage_four_cusp_hemp_like"]) == dataset["headline"]["sobol_five_stage_four_cusp_hemp_like_count"]


def test_cusp_maps_are_hemp_like_representatives_with_profiles(payload):
    maps = payload["cusp_maps"]
    assert 1 <= len(maps) <= GENERATOR.MAX_CUSP_MAPS
    hemp_maps = [rep for rep in maps if rep["hemp"]]
    if payload["headline"]["sobol_hemp_like_count"]:
        assert hemp_maps
    for rep in maps:
        assert len(rep["profiles"]["z_m"]) == 241 and len(rep["profiles"]["wall_abs_b_t"]) == 241
        assert all(trace["path"] for trace in rep["traces"])
        assert len(rep["cusps"]) == len(rep["cells"]) - 1 if rep["cusps"] else len(rep["cells"]) == 1
        for cusp in rep["cusps"]:
            assert cusp["hemp"] == (cusp["rho"] >= 1.5)


def test_hypothesis_outcome_uses_the_preregistered_thresholds(payload, dataset):
    outcome = payload["hypothesis_outcome"]
    test = dataset["estimands"]["sobol_v3"]["hypothesis_test"]
    assert outcome["test"] == test
    assert outcome["h1"]["slope_in_range"] == (0.80 <= test["slope_through_origin"] <= 1.00)
    assert outcome["h1"]["band_fraction_ok"] == (test["fraction_within_band"] >= 0.80)
    assert outcome["h2"]["accuracy_ok"] == (test["prediction_accuracy"] >= 0.85)
    assert outcome["h2"]["no_hemp_like_in_v2_box"] == (dataset["headline"]["sweep_v2_region_hemp_like_count"] == 0)
    assert payload["l1b_p2_queue"]["status"] == "queued_not_run"


def test_rendered_html_is_deterministic_offline_and_checked_in(payload, html):
    assert html == GENERATOR.render_html(payload)
    assert "http://" not in html.split('<script id="payload"')[0] or "http://www.w3.org/2000/svg" in html
    body = html.split('<script id="payload"')[1]
    assert "https://" not in body
    embedded = _embedded_payload(html)
    assert embedded == json.loads(json.dumps(payload))
    assert CHECKED_HTML.is_file(), "regenerate the dashboard: python visualization/generate_l1a_geometry_sweep_v3_dashboard.py"
    checked = CHECKED_HTML.read_bytes().replace(b"\r\n", b"\n")
    assert checked == html.encode("utf-8")
    assert len(checked) <= GENERATOR.MAX_HTML_BYTES
    for token in ("no plasma", "queued_not_run", "Claim boundary", "I<sub>1</sub>"):
        assert token in html


def test_payload_validation_rejects_tampering(payload):
    tampered = json.loads(json.dumps(payload))
    tampered["designs"][0]["hemp"] = not tampered["designs"][0]["hemp"]
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["l1b_p2_queue"]["status"] = "run"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
