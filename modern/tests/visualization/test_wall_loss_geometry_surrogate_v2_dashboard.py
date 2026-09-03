"""Tests for the offline wall-loss geometry surrogate v2 dashboard (with the v1 vs v2 panel).

Every check reads the recorded v2 results bundle independently of the generator
and compares the embedded payload against it, so the dashboard cannot show a
number that does not trace to a hash-bound artifact.  The v1 numbers shown are
those recorded in v2's own ``v1-comparison.json`` artifact, which the test
cross-checks against v1's committed assessment.  The tests skip when the
campaign has not executed yet (no ``results/manifest.json``).
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
GENERATOR_PATH = MODERN / "visualization" / "generate_wall_loss_geometry_surrogate_v2_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "wall-loss-geometry-surrogate-v2.template.html"
CHECKED_HTML = MODERN / "visualization" / "wall-loss-geometry-surrogate-v2.html"
EXPERIMENT = MODERN / "experiments" / "wall_loss_geometry_surrogate_v2"
RESULTS = EXPERIMENT / "results"
V1_RESULTS = MODERN / "experiments" / "wall_loss_geometry_surrogate_v1" / "results"
CLASSIFICATION = "SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "manifest.json").is_file(), reason="the v2 surrogate campaign has not executed"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("wall_loss_geometry_surrogate_v2_dashboard", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


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
    victim = results / "artifacts" / "v1-comparison.json"
    raw = victim.read_bytes()
    assert b'"v2_improves":true' in raw
    victim.write_bytes(raw.replace(b'"v2_improves":true', b'"v2_improves":false', 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch|size mismatch"):
        GENERATOR.build_payload(results, EXPERIMENT)


def test_payload_carries_both_labels_the_verdict_and_the_predictor_status(payload) -> None:
    assert payload["classification"] == CLASSIFICATION
    assert payload["source_classification"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    boundary = payload["claim_boundary"]
    assert boundary["surrogate_of_screening_dataset"] is True
    assert boundary["not_physical_orbit_evidence"] is True
    assert boundary["not_performance_model"] is True
    campaign = _json(RESULTS / "artifacts" / "campaign-result.json")
    gates = _json(RESULTS / "artifacts" / "gates.json")
    status = _json(RESULTS / "artifacts" / "predictor-status.json")
    assert payload["status"] == campaign["status"] == status["status"]
    assert payload["mdo_v2_input_status"] == campaign["mdo_v2_input_status"] == status["mdo_v2_input_status"]
    assert (payload["status"] == "accepted_surrogate") == gates["all_binding_passed"] == (payload["terminal_state"] == "accepted_result")
    assert (payload["mdo_v2_input_status"] == "usable_as_mdo_v2_input_with_screening_label") == gates["all_binding_passed"]
    if payload["status"] == "rejected_surrogate":
        assert sorted(payload["rejection_diagnosis"]["failed_binding_gates"]) == sorted(n for n, g in gates["binding"].items() if not g["passed"])
        assert payload["rejection_diagnosis"]["v3_requirements"]
    rows = {row["name"]: row for row in payload["gates"]["rows"]}
    for name, item in gates["binding"].items():
        assert rows[name]["kind"] == "binding" and rows[name]["passed"] == item["passed"]
    assert payload["partition"]["equals_v1"] is True


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
        for output in payload["outputs"]:
            errors = [row["outputs"][output]["pred"] - row["outputs"][output]["truth"] for row in rows]
            assert payload["scopes"][scope]["per_output"][output]["rmse"] == pytest.approx(_rmse(errors), abs=1e-12)
            assert payload["scopes"][scope]["per_output"][output]["rmse"] == metrics[scope]["per_output"][output]["rmse"]
        for baseline in payload["baselines"]:
            assert payload["scopes"][scope]["baselines"][baseline]["cells"] == metrics[scope]["baselines"][baseline]["cells_pooled"]
            assert payload["scopes"][scope]["baselines"][baseline]["pooled"] == metrics[scope]["baselines"][baseline]["p_wall_pooled"]
    assert payload["baselines"] == ["global-mean", "knn-3", "ridge", "gbt"]


def test_v1_comparison_pairs_the_identical_designs_and_reproduces_v1(payload) -> None:
    v1_partition = _json(V1_RESULTS / "artifacts" / "partitions.json")
    v2_partition = _json(RESULTS / "artifacts" / "partitions.json")
    assert v1_partition["roles"] == v2_partition["roles"]
    v1_assessment = _json(V1_RESULTS / "artifacts" / "assessment.json")
    comparison = _json(RESULTS / "artifacts" / "v1-comparison.json")
    for scope in ("interpolation", "extrapolation"):
        block = payload["comparison"][scope]
        assert block["design_count"] == len(payload["scopes"][scope]["rows"]) == len(v1_assessment[scope]["designs"])
        v1_by_id = {d["case_id"]: d for d in v1_assessment[scope]["designs"]}
        for output in payload["outputs"]:
            v1_errors = [float(v1_by_id[row["case_id"]]["outputs"][output]["error"]) for row in payload["scopes"][scope]["rows"]]
            v2_errors = [row["outputs"][output]["pred"] - row["outputs"][output]["truth"] for row in payload["scopes"][scope]["rows"]]
            assert block["per_output"][output]["v1"] == pytest.approx(_rmse(v1_errors), abs=1e-12)
            assert block["per_output"][output]["v2"] == pytest.approx(_rmse(v2_errors), abs=1e-12)
            assert block["per_output"][output]["v1"] == v1_assessment[scope]["per_output"][output]["rmse"]
            assert block["per_output"][output] == {
                "v1": comparison["scopes"][scope]["per_output"][output]["v1_rmse"],
                "v2": comparison["scopes"][scope]["per_output"][output]["v2_rmse"],
                "diff": comparison["scopes"][scope]["per_output"][output]["difference_v2_minus_v1"],
                "closer": comparison["scopes"][scope]["per_output"][output]["designs_v2_closer"],
                "n": comparison["scopes"][scope]["per_output"][output]["designs"],
                "improves": comparison["scopes"][scope]["per_output"][output]["v2_improves"],
            }
        assert block["cells"]["v1_rmse"] == v1_assessment[scope]["cells"]["rmse"]
    assert payload["headline"]["v1_pooled_rmse_same_designs"] == payload["comparison"]["interpolation"]["per_output"]["p_wall_pooled"]["v1"]


def test_features_learning_curve_and_tree_importance_trace_to_artifacts(payload) -> None:
    features = _json(RESULTS / "artifacts" / "features.json")
    predictor = _json(RESULTS / "artifacts" / "predictor.json")
    curve = _json(RESULTS / "artifacts" / "learning-curve.json")
    sensitivity = _json(RESULTS / "artifacts" / "sensitivity.json")
    selection = _json(RESULTS / "artifacts" / "selection.json")
    baselines = _json(RESULTS / "artifacts" / "baselines.json")
    assert payload["features"]["names"] == features["names"] == predictor["inputs"]["names"]
    assert len(payload["features"]["names"]) == 31 and features["derived_not_fitted"] is True
    assert set(payload["features"]["provenance"]) == set(payload["features"]["names"])
    assert payload["learning_curve"]["sizes"] == curve["sizes"] == [20, 30, 40, 50]
    for entry in payload["learning_curve"]["summary"]:
        index = curve["sizes"].index(entry["size"])
        pooled = [run["curve"][index]["rmse"]["p_wall_pooled"] for run in curve["runs"]]
        assert entry["pooled_rmse_mean"] == pytest.approx(sum(pooled) / len(pooled), abs=1e-12)
        assert entry["pooled_rmse_min"] == min(pooled) and entry["pooled_rmse_max"] == max(pooled)
    assert payload["learning_curve"]["extrapolation"] == curve["extrapolation"]
    assert payload["sensitivity"]["tree"] == sensitivity["tree_feature_importance"]
    assert payload["sensitivity"]["tree"]["parameters"] == selection["gbt_parameters"] == payload["selection"]["gbt_parameters"]
    assert sum(payload["sensitivity"]["tree"]["mean_over_outputs"].values()) == pytest.approx(1.0, abs=1e-9)
    assert payload["sensitivity"]["permutation"]["ranking"] == sensitivity["permutation_importance"]["ranking"]
    grid = baselines["gbt"]["grid_scores_method_selection"]
    assert min(grid, key=lambda g: (g["method_selection_rmse_mean"], g["index"]))["parameters"] == selection["gbt_parameters"]
    assert payload["selection"]["baselines_ms"]["gbt"] == baselines["gbt"]["method_selection_rmse"]


def test_html_is_offline_deterministic_and_within_budget(payload, html) -> None:
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert second == html
    data = html.encode("utf-8")
    assert len(data) <= GENERATOR.MAX_HTML_BYTES
    assert html.count(CLASSIFICATION) >= 2
    assert "not physical-orbit evidence" in html
    assert "v1 vs v2" in html and 'id="compare-table"' in html and 'id="curve"' in html and 'id="features"' in html
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
