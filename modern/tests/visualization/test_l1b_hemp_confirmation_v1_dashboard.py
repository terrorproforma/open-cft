"""Tests for the offline L1b HEMP confirmation v1 dashboard (material-aware P2 check).

Every check reads the sealed bundle independently of the generator and compares the embedded
payload against it, so the dashboard cannot show a number that does not trace to a hash-bound
artifact. The tests skip when the confirmation campaign has not executed yet.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_l1b_hemp_confirmation_v1_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "l1b-hemp-confirmation-v1.template.html"
CHECKED_HTML = MODERN / "visualization" / "l1b-hemp-confirmation-v1.html"
EXPERIMENT = MODERN / "experiments" / "l1b_hemp_confirmation_v1_1"
RESULTS = EXPERIMENT / "results"
V1_RESULTS = MODERN / "experiments" / "l1b_hemp_confirmation_v1" / "results"
CLASSIFICATION = "P2_MATERIAL_AWARE_FIELD_CONFIRMATION_NOT_HARDWARE_VALID"

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file() or not (V1_RESULTS / "manifest.json").is_file(), reason="the L1b confirmation campaign (v1.1) has not executed")


def _load_generator():
    spec = importlib.util.spec_from_file_location("l1b_hemp_confirmation_v1_dashboard", GENERATOR_PATH)
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
    return _json(RESULTS / "artifacts" / "confirmation-dataset.json")


@pytest.fixture(scope="module")
def campaign():
    return _json(RESULTS / "artifacts" / "campaign-result.json")


def _embedded_payload(text: str):
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, re.S)
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_bundle_is_byte_verified_and_payload_traces_to_it(payload, dataset, campaign):
    manifest = _json(RESULTS / "manifest.json")
    assert manifest["state"] == "accepted_result"
    for entry in manifest["artifacts"]:
        if entry["type"] == "file":
            raw = (RESULTS / entry["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == entry["byte_sha256"], entry["path"]
    assert payload["identity"]["manifest_file_sha256"] == hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest()
    assert payload["classification"] == CLASSIFICATION == dataset["classification"]
    assert payload["headline"] == dataset["headline"] and payload["agreement_table"] == campaign["agreement_table"]
    assert payload["verdict"] == campaign["verdict"] in GENERATOR.VERDICTS
    assert len(payload["designs"]) == dataset["design_count"] == 15
    rows = {row["design_id"]: row for row in dataset["designs"]}
    for item in payload["designs"]:
        row = rows[item["id"]]
        assert item["p2_rho"] == [r["rho_conservative"] for r in row["p2_rho"]]
        assert item["l1a_rho"] == [r["rho_conservative"] for r in row["l1a"]["rho"]]
        assert item["p2_hemp"] == row["comparison"]["p2_hemp_like_all_cusps"] and item["l1a_hemp"] is True
        assert item["p2_cusps"] == row["comparison"]["p2_wall_cusp_count"] and item["l1a_cusps"] == row["comparison"]["l1a_wall_cusp_count"]
        assert item["max_shift_m"] == row["comparison"]["max_cusp_shift_m"] and item["converged"] is True
        assert item["dofs"] == [level["p2_dofs"] for level in row["p2"]["levels"]]


def test_predecessor_rejection_and_angle_gate_are_carried(payload):
    predecessor = payload["predecessor"]
    v1_manifest = _json(V1_RESULTS / "manifest.json")
    assert v1_manifest["state"] == "development_rejection" and predecessor["state"] == "development_rejection"
    assert predecessor["manifest_file_sha256"] == hashlib.sha256((V1_RESULTS / "manifest.json").read_bytes()).hexdigest()
    assert sorted(item["design_id"] for item in predecessor["failed_designs"]) == ["l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59"]
    assert predecessor["resolved_design_count"] == 13 and predecessor["reject_below_angle_deg"] == 10.0
    assert predecessor["protocol_block"]["preregistration_commit"] == predecessor["preregistration_commit_sha"]
    gate = payload["angle_gate"]
    assert gate["reject_below_angle_deg"] == 5.0 and len(gate["per_design_levels"]) == 15
    assert all(level["min_angle_deg"] >= 5.0 for levels in gate["per_design_levels"].values() for level in levels)
    assert set(gate["designs_with_elements_below_10deg"]) == {"l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59"}


def test_verdict_and_scatter_reproduce_from_the_rows(payload, dataset):
    confirmation = dataset["gates"]["confirmation"]
    b_passed = confirmation["cusp_count_unchanged"]["fraction_boundary_tolerant"] >= confirmation["cusp_count_unchanged"]["pass_threshold"]
    c_passed = confirmation["cusp_position_shift"]["all_designs_bijective"] and confirmation["cusp_position_shift"]["max_shift_over_tolerance"] <= confirmation["cusp_position_shift"]["pass_threshold"]
    expected = "CONFIRMED" if (b_passed and c_passed) else ("PARTIALLY_CONFIRMED" if (b_passed or c_passed) else "DISCONFIRMED")
    assert payload["verdict"] == expected and payload["confirmation"] == confirmation
    matched = [point for point in payload["scatter"] if "shift_m" in point]
    assert len(matched) == sum(row["comparison"]["matched_cusp_count"] for row in dataset["designs"]) == confirmation["cusp_position_shift"]["matched_cusp_count"]
    assert max(point["shift_tol"] for point in matched) == pytest.approx(confirmation["cusp_position_shift"]["max_shift_over_tolerance"])
    for point in matched:
        assert point["shift_m"] == pytest.approx(abs(point["p2_z_m"] - point["l1a_z_m"]))
        assert 0.0 <= point["z_over_L"] <= 1.0
    assert sum(item["p2_hemp"] for item in payload["designs"]) == confirmation["hemp_like_preserved"]["preserved_count"]


def test_overlays_carry_both_fields_with_profiles_and_traces(payload):
    overlays = payload["overlays"]
    assert 1 <= len(overlays) <= GENERATOR.MAX_OVERLAYS
    representatives = [item["id"] for item in payload["designs"] if item["representative"]]
    assert set(representatives) <= {overlay["id"] for overlay in overlays}
    for overlay in overlays:
        assert len(overlay["p2_profiles"]["z_m"]) == 241 and len(overlay["l1a_profiles"]["z_m"]) == 241
        assert overlay["p2_profiles"]["z_m"][0] == pytest.approx(overlay["l1a_profiles"]["z_m"][0], abs=1e-9)
        assert all(trace["path"] for trace in overlay["traces"])
        assert len(overlay["p2_cusps"]) == len(overlay["p2_cells"]) - 1
        for cusp in overlay["p2_cusps"]:
            assert cusp["hemp"] == (cusp["rho"] >= 1.5)
        assert any(region["mu_r"] > 1000 for region in overlay["regions"]) and any(region["br_t"] != 0.0 for region in overlay["regions"])


def test_rendered_html_is_deterministic_offline_and_checked_in(payload, html):
    assert html == GENERATOR.render_html(payload)
    assert "http://" not in html.split('<script id="payload"')[0] or "http://www.w3.org/2000/svg" in html
    body = html.split('<script id="payload"')[1]
    assert "https://" not in body
    embedded = _embedded_payload(html)
    assert embedded == json.loads(json.dumps(payload))
    assert CHECKED_HTML.is_file(), "regenerate the dashboard: python visualization/generate_l1b_hemp_confirmation_v1_dashboard.py"
    checked = CHECKED_HTML.read_bytes().replace(b"\r\n", b"\n")
    assert checked == html.encode("utf-8")
    assert len(checked) <= GENERATOR.MAX_HTML_BYTES
    for token in ("no plasma", "Claim boundary", "GATE (b)", "GATE (c)", "paper admission not in scope", "development rejection", payload["verdict"]):
        assert token in html


def test_payload_validation_rejects_tampering(payload):
    tampered = json.loads(json.dumps(payload))
    tampered["designs"][0]["p2_hemp"] = not tampered["designs"][0]["p2_hemp"]
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["verdict"] = "CONFIRMED" if payload["verdict"] != "CONFIRMED" else "DISCONFIRMED"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["paper_admission"] = "admitted"
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["predecessor"]["failed_designs"] = tampered["predecessor"]["failed_designs"][:1]
    with pytest.raises(ValueError):
        GENERATOR.validate_payload(tampered)
