"""Tests for the offline cusp topology search v3 dashboard (v3.1 accepted bundle, v3 lineage).

Every check reads the sealed bundles independently of the generator and compares the
embedded payload against them, so the dashboard cannot show a number that does not trace to
a hash-bound artifact. The tests skip when the v3.1 campaign has not executed yet.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import statistics
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_cusp_topology_search_v3_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "cusp-topology-search-v3.template.html"
CHECKED_HTML = MODERN / "visualization" / "cusp-topology-search-v3.html"
EXPERIMENT = MODERN / "experiments" / "cusp_topology_search_v3_1"
RESULTS = EXPERIMENT / "results"
LINEAGE_RESULTS = MODERN / "experiments" / "cusp_topology_search_v3" / "results"
CLASSIFICATION = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="the v3.1 campaign has not executed")


def _load_generator():
    spec = importlib.util.spec_from_file_location("cusp_topology_search_v3_dashboard", GENERATOR_PATH)
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
    return _json(RESULTS / "artifacts" / "topology-dataset.json")


def test_bundle_identity_is_byte_verified(payload) -> None:
    identity = payload["identity"]
    manifest = _json(RESULTS / "manifest.json")
    assert identity["state"] == "accepted_result"
    assert identity["manifest_file_sha256"] == hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest()
    assert identity["terminal_file_sha256"] == manifest["terminal_byte_sha256"]
    assert identity["lock_file_sha256"] == manifest["lock_byte_sha256"]
    assert identity["preregistration_commit_sha"] == _json(RESULTS / "execution-lock.json")["commit"]
    files = [entry for entry in manifest["artifacts"] if entry["type"] == "file"]
    assert identity["verified_file_count"] == len(files) and identity["artifact_count"] == manifest["artifact_count"]
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
    victim.write_bytes(victim.read_bytes().replace(b'"passed":true', b'"passed":false', 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch|size mismatch"):
        GENERATOR.build_payload(results, EXPERIMENT)


def test_generator_refuses_the_rejected_v3_bundle_as_the_main_source() -> None:
    with pytest.raises(ValueError, match="not accepted_result"):
        GENERATOR.build_payload(LINEAGE_RESULTS, MODERN / "experiments" / "cusp_topology_search_v3")


def test_payload_is_labelled_and_traces_to_the_dataset(payload, dataset) -> None:
    assert payload["classification"] == CLASSIFICATION
    assert payload["p2_classification"] == "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
    assert payload["claim_boundary"]["forbid_mirror_probability_publication"] is True
    assert payload["headline"] == dataset["headline"]
    assert len(payload["designs"]) == dataset["design_count"] == 281
    by_key = {(row["set_id"], row["design_id"]): row for row in dataset["designs"]}
    for item in payload["designs"]:
        row = by_key[(item["set"], item["id"])]
        assert item["cusps"] == row["wall_cusp_count"] and item["cells"] == row["cell_count"]
        assert item["z_c_m"] == [cusp["z_c_m"] for cusp in row["wall_cusps"]]
        assert item["stable"] == row["stability"]["stable"] and item["label"] == row["label"]
        assert item["wall_mirror"] == [cell["wall_mirror_ratio"] for cell in row["cells"]]


def test_headline_and_per_set_summaries_reproduce_from_rows(payload) -> None:
    rows = payload["designs"]
    histogram: dict[str, int] = {}
    for item in rows:
        histogram[str(item["cusps"])] = histogram.get(str(item["cusps"]), 0) + 1
    assert payload["headline"]["wall_cusp_count_histogram"] == dict(sorted(histogram.items(), key=lambda p: int(p[0])))
    assert payload["headline"]["stable_design_count"] == sum(item["stable"] for item in rows) == 281
    for set_id, summary in payload["by_set"].items():
        subset = [item for item in rows if item["set"] == set_id]
        assert summary["count"] == len(subset)
        assert summary["four_cusp_fraction"] == sum(item["four_cusps"] for item in subset) / len(subset)
        assert summary["four_cell_fraction"] == sum(item["four_cells"] for item in subset) / len(subset)
        z_over = sorted(value for item in subset for value in item["z_c_over_L"])
        if z_over:
            assert summary["z_c_over_L"]["values"] == z_over and summary["z_c_over_L"]["median"] == statistics.median(z_over)
    assert payload["by_set"]["four_cell_v2"]["cusp_histogram"] == {"1": 128}
    assert payload["by_set"]["p2_divergent_exit"]["cusp_histogram"] == {"3": 1}
    assert set(payload["by_set"]["sweep_v2"]["cusp_histogram"]) == {"2", "3", "4"}


def test_representative_plots_trace_to_the_sealed_records(payload, dataset) -> None:
    reps = {(rep["set"], rep["id"]): rep for rep in payload["representatives"]}
    assert len(reps) == sum(row["representative"] for row in dataset["designs"]) == 14
    for row in dataset["designs"]:
        if not row["representative"]:
            continue
        rep = reps[(row["set_id"], row["design_id"])]
        record = _json(RESULTS / row["record_path"])
        assert [c["z_m"] for c in rep["cusps"]] == [c["z_c_m"] for c in record["accepted"]["topology"]["wall_cusps"]]
        assert len(rep["traces"]) == len(record["accepted"]["separatrix_traces"])
        for trace, sealed in zip(rep["traces"], record["accepted"]["separatrix_traces"]):
            assert trace["termination"] == sealed["termination"] and len(trace["path"]) == len(sealed["path_rz_m"])
            assert trace["path"][-1][0] == round(sealed["path_rz_m"][-1][0], 10)
        assert rep["z_range_m"][0] >= record["accepted"]["grid"]["z_min_m"] and rep["z_range_m"][1] <= record["accepted"]["grid"]["z_max_m"]
    p2 = reps[("p2_divergent_exit", "divergent-exit-stack")]
    assert [round(c["z_m"] * 1e3, 3) for c in p2["cusps"]] == [6.028, 12.0, 17.972]
    assert p2["p2_consistency"]["cusp_count_equals_reference_count"] is True


def test_lineage_traces_to_the_v3_rejection_and_the_sealed_v1_v2_datasets(payload) -> None:
    lineage = payload["lineage"]
    v3 = lineage["v3_recorded_rejection"]
    manifest = _json(LINEAGE_RESULTS / "manifest.json")
    assert v3["state"] == manifest["state"] == "assessment_rejection"
    assert v3["manifest_file_sha256"] == hashlib.sha256((LINEAGE_RESULTS / "manifest.json").read_bytes()).hexdigest()
    gates = _json(LINEAGE_RESULTS / "artifacts" / "gates.json")
    assert v3["failing_gates"] == {"held_out_correspondence": gates["failing_designs"]["held_out_correspondence"]}
    assert v3["failing_design_count"] == 14
    assert v3["wall_cusp_count_histogram"] == payload["headline"]["wall_cusp_count_histogram"]
    frozen = lineage["frozen_definition_results"]
    v1 = _json(MODERN / "experiments" / "cft_topology_characterization_v1" / "results" / "dataset.json")
    v2 = _json(MODERN / "experiments" / "four_cell_topology_search_v2" / "results" / "dataset.json")
    assert frozen["characterization_v1"]["stable_eligible_cusp_count"] == v1["summary"]["stable_eligible_cusp_count"] == 0
    assert frozen["four_cell_v2"]["stable_count"] == v2["summary"]["stable_count"] == 0
    assert frozen["four_cell_v2"]["candidates"] == 128 and frozen["characterization_v1"]["cases"] == 56


def test_html_is_offline_deterministic_and_within_budget(payload, html) -> None:
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert second == html
    data = html.encode("utf-8")
    assert len(data) <= GENERATOR.MAX_HTML_BYTES
    assert html.count(CLASSIFICATION) >= 3
    assert "<script src" not in html and '<link rel="stylesheet"' not in html
    # Only the literature locators (in the JSON payload) may carry URLs; no fetched resources.
    assert not re.search(r'(src|href)="https?://', html)
    assert "__PAYLOAD_JSON__" not in html
    embedded = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert embedded is not None
    assert json.loads(embedded.group(1).replace("<\\/", "</")) == json.loads(json.dumps(payload))


def test_committed_html_matches_regeneration(html) -> None:
    if not CHECKED_HTML.is_file():
        pytest.skip("dashboard not generated yet")
    assert CHECKED_HTML.read_bytes().replace(b"\r\n", b"\n") == html.encode("utf-8").replace(b"\r\n", b"\n")
