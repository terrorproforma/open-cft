"""Tests for the offline wall-loss geometry surrogate v1 dashboard.

Every check reads the recorded results bundle independently of the generator and
compares the embedded payload against it, so the dashboard cannot show a number
that does not trace to a hash-bound artifact. The tests skip when the campaign
has not executed yet (no ``results/manifest.json``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import shutil
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_wall_loss_geometry_surrogate_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "wall-loss-geometry-surrogate-v1.template.html"
CHECKED_HTML = MODERN / "visualization" / "wall-loss-geometry-surrogate-v1.html"
EXPERIMENT = MODERN / "experiments" / "wall_loss_geometry_surrogate_v1"
RESULTS = EXPERIMENT / "results"
CLASSIFICATION = "SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "manifest.json").is_file(), reason="the surrogate campaign has not executed"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("wall_loss_geometry_surrogate_dashboard", GENERATOR_PATH)
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


def test_bundle_identity_is_byte_verified_and_state_is_recorded(payload) -> None:
    identity = payload["identity"]
    manifest = _json(RESULTS / "manifest.json")
    assert identity["manifest_file_sha256"] == hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest()
    assert identity["terminal_file_sha256"] == manifest["terminal_byte_sha256"]
    assert identity["lock_file_sha256"] == manifest["lock_byte_sha256"]
    assert identity["preregistration_commit_sha"] == _json(RESULTS / "execution-lock.json")["commit"]
    assert payload["terminal_state"] == manifest["state"] == _json(RESULTS / "terminal.json")["state"]
    files = [entry for entry in manifest["artifacts"] if entry["type"] == "file"]
    assert identity["verified_file_count"] == len(files)
    for entry in files:
        raw = (RESULTS / entry["path"]).read_bytes()
        assert identity["artifact_hashes"][entry["path"]] == hashlib.sha256(raw).hexdigest() == entry["byte_sha256"]
    assert identity["generator_sha256"] == hashlib.sha256(GENERATOR_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert identity["template_sha256"] == hashlib.sha256(TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    GENERATOR.validate_payload(payload)


def test_generator_refuses_tampered_bundles(tmp_path: Path) -> None:
    results = tmp_path / "results"
    shutil.copytree(RESULTS, results)
    victim = results / "artifacts" / "gates.json"
    raw = victim.read_bytes()
    assert b'"passed":true' in raw  # the structural gates always pass in a completed bundle
    victim.write_bytes(raw.replace(b'"passed":true', b'"passed":false', 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch|size mismatch"):
        GENERATOR.build_payload(results, EXPERIMENT)


def test_payload_carries_both_labels_and_the_recorded_verdict(payload) -> None:
    assert payload["classification"] == CLASSIFICATION
    assert payload["source_classification"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    boundary = payload["claim_boundary"]
    assert boundary["surrogate_of_screening_dataset"] is True
    assert boundary["not_physical_orbit_evidence"] is True
    assert boundary["not_performance_model"] is True
    campaign = _json(RESULTS / "artifacts" / "campaign-result.json")
    gates = _json(RESULTS / "artifacts" / "gates.json")
    assert payload["status"] == campaign["status"]
    assert (payload["status"] == "accepted_surrogate") == gates["all_binding_passed"] == (payload["terminal_state"] == "accepted_result")
    rows = {row["name"]: row for row in payload["gates"]["rows"]}
    for name, item in gates["binding"].items():
        assert rows[name]["kind"] == "binding" and rows[name]["passed"] == item["passed"]
    for name, item in gates["reported_not_binding"].items():
        if isinstance(item, dict) and "passed" in item:
            assert rows[name]["kind"] == "reported" and rows[name]["passed"] == item["passed"]


def test_scope_rows_trace_to_assessment_and_metrics(payload) -> None:
    assessment = _json(RESULTS / "artifacts" / "assessment.json")
    metrics = _json(RESULTS / "artifacts" / "metrics.json")
    partition = _json(RESULTS / "artifacts" / "partitions.json")
    for scope, role in (("interpolation", "assessment"), ("extrapolation", "extrapolation")):
        rows = payload["scopes"][scope]["rows"]
        assert [row["case_id"] for row in rows] == partition["roles"][role] == [d["case_id"] for d in assessment[scope]["designs"]]
        for row, design in zip(rows, assessment[scope]["designs"], strict=True):
            for output in payload["outputs"]:
                item = design["outputs"][output]
                assert row["outputs"][output]["truth"] == item["truth"]
                assert row["outputs"][output]["pred"] == item["predicted"]
                assert [row["outputs"][output]["lo"], row["outputs"][output]["hi"]] == item["observation_interval"]
        for output in payload["outputs"]:
            errors = [row["outputs"][output]["pred"] - row["outputs"][output]["truth"] for row in rows]
            rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
            assert payload["scopes"][scope]["per_output"][output]["rmse"] == pytest.approx(rmse, abs=1e-12)
            assert payload["scopes"][scope]["per_output"][output]["rmse"] == metrics[scope]["per_output"][output]["rmse"]
        assert payload["scopes"][scope]["gated_coverage"] == metrics[scope]["gated_coverage"]
        assert payload["scopes"][scope]["best_baseline_pooled"] == metrics[scope]["best_baseline_pooled"]


def test_headline_selection_and_sensitivity_trace_to_artifacts(payload) -> None:
    campaign = _json(RESULTS / "artifacts" / "campaign-result.json")
    selection = _json(RESULTS / "artifacts" / "selection.json")
    candidates = _json(RESULTS / "artifacts" / "candidates.json")
    sensitivity = _json(RESULTS / "artifacts" / "sensitivity.json")
    calibration = _json(RESULTS / "artifacts" / "calibration.json")
    assert payload["headline"] == campaign["headline"]
    assert payload["selection"]["selected"] == selection["selected"] == campaign["selected_candidate"]
    scores = payload["selection"]["scores"]
    assert scores == {c: candidates[c]["method_selection_rmse"]["mean_over_outputs"] for c in candidates}
    order = payload["selection"]["order"]
    assert min(order, key=lambda c: (scores[c], order.index(c))) == selection["selected"]
    assert payload["sensitivity"]["permutation"]["ranking"] == sensitivity["permutation_importance"]["ranking"]
    assert payload["sensitivity"]["ard"] == sensitivity["ard_length_scales"]
    assert payload["calibration"]["variance_scale"] == calibration["variance_scale"]
    assert len(payload["calibration"]["standardised_residuals"]) == calibration["fit_sample_count"] == 50


def test_html_is_offline_deterministic_and_within_budget(payload, html) -> None:
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert second == html
    data = html.encode("utf-8")
    assert len(data) <= GENERATOR.MAX_HTML_BYTES
    assert html.count(CLASSIFICATION) >= 2
    assert "not physical-orbit evidence" in html
    assert "<script src" not in html and '<link rel="stylesheet"' not in html
    assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", html)
    assert "__PAYLOAD_JSON__" not in html
    embedded = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert embedded is not None
    assert json.loads(embedded.group(1).replace("<\\/", "</")) == json.loads(json.dumps(payload))


def test_committed_html_matches_regeneration(html) -> None:
    if not CHECKED_HTML.is_file():
        pytest.skip("dashboard not generated yet")
    assert CHECKED_HTML.read_bytes().replace(b"\r\n", b"\n") == html.encode("utf-8").replace(b"\r\n", b"\n")
