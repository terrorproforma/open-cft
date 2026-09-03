"""Tests for the offline wall-loss-vs-geometry screening dashboard (v1).

Every check reads the sealed results bundle independently of the generator and
compares the embedded payload against it, so the dashboard cannot show a number
that does not trace to a hash-bound artifact. The tests skip when the campaign
has not executed yet (no ``results/manifest.json``).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_wall_loss_geometry_screening_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "wall-loss-geometry-screening-v1.template.html"
CHECKED_HTML = MODERN / "visualization" / "wall-loss-geometry-screening-v1.html"
EXPERIMENT = MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1"
RESULTS = EXPERIMENT / "results"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "manifest.json").is_file(), reason="the geometry screening campaign has not executed"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("wall_loss_geometry_screening_dashboard", GENERATOR_PATH)
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
    return _json(RESULTS / "artifacts" / "geometry-wall-loss-dataset.json")


def test_bundle_identity_is_byte_verified(payload) -> None:
    identity = payload["identity"]
    manifest = _json(RESULTS / "manifest.json")
    assert identity["manifest_file_sha256"] == hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest()
    assert identity["terminal_file_sha256"] == manifest["terminal_byte_sha256"]
    assert identity["lock_file_sha256"] == manifest["lock_byte_sha256"]
    assert identity["preregistration_commit_sha"] == _json(RESULTS / "execution-lock.json")["commit"]
    files = [entry for entry in manifest["artifacts"] if entry["type"] == "file"]
    assert identity["verified_file_count"] == len(files)
    assert identity["artifact_count"] == manifest["artifact_count"]
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


def test_payload_is_screening_labelled_and_evidentiary(payload, dataset) -> None:
    assert payload["classification"] == CLASSIFICATION
    assert payload["evidentiary"] is True
    assert payload["campaign_status"] == "accepted_screening_dataset"
    assert payload["claim_boundary"]["not_p2_qualified"] is True
    assert payload["claim_boundary"]["not_accepted_physical_orbit_evidence"] is True
    assert payload["design_count"] == dataset["design_count"] == len(payload["designs"])
    assert payload["field_source"]["field_status"] == "accepted_L1a_screening_not_P2_qualified"


def test_design_rows_trace_to_dataset_and_case_summaries(payload, dataset) -> None:
    by_id = {row["case_id"]: row for row in dataset["designs"]}
    for item in payload["designs"]:
        row = by_id[item["case_id"]]
        reported = row["reported"]
        assert item["p"]["wall_2N"]["p"] == reported["wall_hit"]["probability"]
        assert item["p"]["wall_2N"]["lo"] == reported["wall_hit"]["lower"]
        assert item["p"]["wall_2N"]["hi"] == reported["wall_hit"]["upper"]
        assert item["p"]["escape_2N"]["k"] == reported["domain_escape"]["successes"]
        assert item["p"]["reflected_2N"]["k"] == reported["reflected"]["successes"]
        assert item["convergence"]["converged"] == row["convergence"]["converged"]
        assert item["convergence"]["change"] == row["convergence"]["successive_change"]
        summary = _json(RESULTS / "artifacts" / "summaries" / f"{item['case_id']}--accepted-2N.json")
        assert summary["summary"]["wall_hit"]["probability"] == item["p"]["wall_2N"]["p"]
        assert summary["summary"]["termination_counts"]["reflected"] == item["reflections"]["2N"]
        coarse = _json(RESULTS / "artifacts" / "summaries" / f"{item['case_id']}--accepted-N.json")
        assert coarse["summary"]["wall_hit"]["probability"] == item["p"]["wall_N"]["p"]
        # Per-cell probabilities re-derived from the 2N strata.
        cells = {}
        for stratum in summary["strata"]:
            cell = cells.setdefault(stratum["cell_id"], [0, 0])
            cell[0] += stratum["termination_counts"]["wall_hit"]
            cell[1] += stratum["trials"]
        assert item["cell_ids"] == sorted(cells)
        assert item["per_cell_2N"] == [cells[c][0] / cells[c][1] for c in sorted(cells)]
        assert sum(s["wall"] for s in item["strata_2N"]) == item["p"]["wall_2N"]["k"]
        assert item["gates"]["structural_passed"] == row["gates"]["structural_passed"]
        # Endpoint table agrees with the summary termination counts.
        endpoints = json.loads(gzip.decompress((RESULTS / "artifacts" / "endpoints" / f"{item['case_id']}--accepted-2N.json.gz").read_bytes()))
        assert len(endpoints["rows"]) == item["p"]["wall_2N"]["n"]
        assert sum(r["termination"] == "wall_hit" for r in endpoints["rows"]) == item["p"]["wall_2N"]["k"]
        assert sum(r["termination"] == "reflected" for r in endpoints["rows"]) == item["reflections"]["2N"]


def test_headline_reproduces_from_rows(payload) -> None:
    wall = [item["p"]["wall_2N"]["p"] for item in payload["designs"]]
    headline = payload["headline"]
    assert headline["wall_hit_probability_min"] == min(wall)
    assert headline["wall_hit_probability_max"] == max(wall)
    assert headline["wall_hit_probability_median"] == statistics.median(wall)
    assert headline["converged_design_count"] == sum(item["convergence"]["converged"] for item in payload["designs"])
    assert headline["total_reflections_2N"] == sum(item["reflections"]["2N"] for item in payload["designs"])
    least = sorted(payload["designs"], key=lambda d: (d["p"]["wall_2N"]["p"], d["case_id"]))[0]["case_id"]
    assert headline["least_wall_loss_case_ids"][0] == least


def test_consumer_section_traces_to_the_consumer_record(payload) -> None:
    record = _json(RESULTS / "artifacts" / "coupling-consumer-record.json")
    assert payload["consumer"]["v4_reference"] == record["v4_reference"]["reference_row"]
    assert payload["consumer"]["v4_design_in_screening_set"] is False
    assert payload["consumer"]["screening_consumed"] + payload["consumer"]["screening_unsealed"] == len(record["screening_designs_consumed"])
    assert payload["consumer"]["v4_reference"]["not_part_of_screening_dataset"] is True


def test_html_is_offline_deterministic_and_within_budget(payload, html) -> None:
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert second == html
    data = html.encode("utf-8")
    assert len(data) <= GENERATOR.MAX_HTML_BYTES
    assert html.count(CLASSIFICATION) >= 3
    assert "<script src" not in html and "<link rel=\"stylesheet\"" not in html
    assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", html)
    assert "__PAYLOAD_JSON__" not in html
    embedded = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert embedded is not None
    assert json.loads(embedded.group(1).replace("<\\/", "</")) == json.loads(json.dumps(payload))


def test_committed_html_matches_regeneration(html) -> None:
    if not CHECKED_HTML.is_file():
        pytest.skip("dashboard not generated yet")
    assert CHECKED_HTML.read_bytes().replace(b"\r\n", b"\n") == html.encode("utf-8").replace(b"\r\n", b"\n")
