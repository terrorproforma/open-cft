"""Lifecycle-aware consistency of the sealed v3 bundle (skips before execution).

The single execution ended ``assessment_rejection``: every numerical gate true, the binding
``held_out_correspondence`` gate false for exactly 14 characterization-v1 designs because of
the reference-extraction defect audited in ``POSTHOC_AUDIT.md``. These tests bind that
RECORDED outcome; they never expect an accepted result from this bundle.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json

import pytest

from cft_revival.experiment_runtime.canonical import canonical_bytes, semantic_sha256, strict_json_file

from experiments.cusp_topology_search_v3 import catalogue as C
from experiments.cusp_topology_search_v3 import experiment as E
from experiments.cusp_topology_search_v3 import topology as T
from experiments.cusp_topology_search_v3.audit_held_out import RECORDED_FAILING_DESIGN_COUNT, RECORDED_FAILING_GATE, RECORDED_TERMINAL_STATE

RESULTS = E.RESULTS_ROOT
ARTIFACTS = RESULTS / "artifacts"
RECORDED_FAILING_DESIGNS = (
    "characterization_v1:topology-s05-p0-r0-neg",
    "characterization_v1:topology-s05-p0-r0-pos",
    "characterization_v1:topology-s06-p0-r0-neg",
    "characterization_v1:topology-s06-p0-r0-pos",
    "characterization_v1:topology-s06-p1-r1-neg",
    "characterization_v1:topology-s06-p1-r1-pos",
    "characterization_v1:topology-s07-p0-r0-neg",
    "characterization_v1:topology-s07-p0-r0-pos",
    "characterization_v1:topology-s07-p1-r1-neg",
    "characterization_v1:topology-s07-p1-r1-pos",
    "characterization_v1:topology-s08-p0-r1-neg",
    "characterization_v1:topology-s08-p0-r1-pos",
    "characterization_v1:topology-s08-p1-r1-neg",
    "characterization_v1:topology-s08-p1-r1-pos",
)

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="not executed yet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return strict_json_file(RESULTS / "manifest.json")


@pytest.fixture(scope="module")
def dataset() -> dict:
    return strict_json_file(ARTIFACTS / "topology-dataset.json")


@pytest.fixture(scope="module")
def campaign() -> dict:
    return strict_json_file(ARTIFACTS / "campaign-result.json")


@pytest.fixture(scope="module")
def gates() -> dict:
    return strict_json_file(ARTIFACTS / "gates.json")


def test_bundle_is_the_recorded_rejection_and_byte_exact(manifest: dict) -> None:
    assert manifest["state"] == RECORDED_TERMINAL_STATE == "assessment_rejection"
    terminal = strict_json_file(RESULTS / "terminal.json")
    assert terminal["state"] == RECORDED_TERMINAL_STATE and terminal["primary_error"] is None
    assert terminal["payload"]["status"] == "gates_failed"
    lock = strict_json_file(RESULTS / "execution-lock.json")
    assert lock["command"].endswith("run execute") and lock["commit"] == "691599340355818ff64d3834d45110768a751589"
    for entry in manifest["artifacts"]:
        if entry["type"] != "file":
            continue
        raw = (RESULTS / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["byte_sha256"], entry["path"]
        assert len(raw) == entry["bytes"], entry["path"]
        if entry["path"].endswith((".json", ".csv")):
            assert b"\r" not in raw, entry["path"]


def test_frozen_preregistration_files_match_the_bundle() -> None:
    assert (ARTIFACTS / "shakedown.json").read_bytes() == E.SHAKEDOWN_PATH.read_bytes()
    assert strict_json_file(ARTIFACTS / "authorities.json") == strict_json_file(E.AUTHORITIES_PATH)
    assert strict_json_file(ARTIFACTS / "design-authorities.json") == strict_json_file(E.DESIGN_AUTHORITIES_PATH)
    assert strict_json_file(ARTIFACTS / "protocol.json") == E.protocol()
    authorities = strict_json_file(E.AUTHORITIES_PATH)
    assert authorities["protocol_semantic_sha256"] == semantic_sha256(E.protocol())


def test_exactly_the_held_out_gate_failed_for_the_recorded_designs(dataset: dict, campaign: dict, gates: dict) -> None:
    assert dataset["classification"] == campaign["classification"] == E.CLASSIFICATION
    assert campaign["status"] == "gates_failed" and campaign["evidentiary"] is True and campaign["gates_passed"] is False
    assert gates["passed"] is False and gates["binding"] is True
    failing = {name for name, ok in gates["campaign"].items() if not ok}
    assert failing == {RECORDED_FAILING_GATE}
    assert tuple(gates["failing_designs"][RECORDED_FAILING_GATE]) == RECORDED_FAILING_DESIGNS
    assert len(RECORDED_FAILING_DESIGNS) == RECORDED_FAILING_DESIGN_COUNT
    for name, designs in gates["failing_designs"].items():
        if name != RECORDED_FAILING_GATE:
            assert designs == []
    assert dataset["design_count"] == campaign["design_count"] == gates["design_count"] == 281
    assert dataset["gates"]["campaign"] == gates["campaign"]
    assert campaign["headline"] == dataset["headline"]
    assert gates["replays"] and all(item["bit_identical"] and item["field_identity_equal"] and item["accepted_grid_equal"] for item in gates["replays"])
    assert dataset["headline"]["stable_design_count"] == 281
    for row in dataset["designs"]:
        assert row["label"] == (E.P2_CLASSIFICATION if row["set_id"] == "p2_divergent_exit" else E.CLASSIFICATION)
        assert row["stability"]["stable"] is True
        checks = dict(row["gate_checks"])
        expected_held_out = row["key"] not in RECORDED_FAILING_DESIGNS
        assert checks.pop("held_out_correspondence") is expected_held_out, row["key"]
        assert all(checks.values()), row["key"]
    histogram: dict[str, int] = {}
    for row in dataset["designs"]:
        histogram[str(row["wall_cusp_count"])] = histogram.get(str(row["wall_cusp_count"]), 0) + 1
    assert dataset["estimands"]["pooled_all"]["wall_cusp_count_histogram"] == dict(sorted(histogram.items(), key=lambda pair: int(pair[0])))
    for set_id, count in {"sweep_v2": 96, "four_cell_v2": 128, "characterization_v1": 56, "p2_divergent_exit": 1}.items():
        rows = [row for row in dataset["designs"] if row["set_id"] == set_id]
        assert len(rows) == count
        assert dataset["estimands"][set_id]["four_wall_cusp_fraction"] == sum(row["four_wall_cusps"] for row in rows) / count


def test_design_records_reproduce_the_dataset_rows(dataset: dict) -> None:
    value = E.protocol()
    tolerance = value["definition_v3"]["stability_tolerance_m"]
    for row in dataset["designs"]:
        record = strict_json_file(RESULTS / row["record_path"])
        assert record["status"] == "resolved" and record["key"] == row["key"]
        assert E.dataset_row(record) == row
        assert record["gate_checks"] == E.design_gate_checks(record)
        assert T.compare_resolutions(record["accepted"], record["refined"], tolerance) == record["stability"]
        payload = {key: record[key] for key in ("axis_window_m", "accepted", "refined", "stability", "held_out", "p2_consistency")}
        assert hashlib.sha256(canonical_bytes(payload)).hexdigest() == record["topology_payload_sha256"]
        grid_raw = gzip.decompress((RESULTS / record["accepted_grid_path"]).read_bytes())
        assert hashlib.sha256(grid_raw).hexdigest() == record["accepted_grid_payload_sha256"]
        grid = json.loads(grid_raw)
        assert grid["identity"] == record["identity"] and len(grid["r_m"]) == record["accepted"]["grid"]["radial_samples"]
        if record["representative"]:
            assert all(trace["path_rz_m"] is not None for trace in record["accepted"]["separatrix_traces"])
        else:
            assert all(trace["path_rz_m"] is None for trace in record["accepted"]["separatrix_traces"])


def test_csv_matches_dataset(dataset: dict) -> None:
    rows = list(csv.reader(io.StringIO((ARTIFACTS / "topology-dataset.csv").read_text(encoding="utf-8"))))
    assert rows[0] == list(E.CSV_COLUMNS)
    assert len(rows) == dataset["design_count"] + 1
    assert E.dataset_csv(dataset["designs"]) == (ARTIFACTS / "topology-dataset.csv").read_bytes()


def test_catalogue_of_the_rejected_bundle_is_refused_by_the_consumer_loader() -> None:
    with pytest.raises(ValueError, match="not an accepted result"):
        C.load_catalogue(RESULTS)
    # The sealed catalogue itself is schema-valid; only the acceptance gate refuses it.
    C.validate_catalogue(strict_json_file(ARTIFACTS / "cusp-cell-catalogue.json"))


def test_held_out_summary_records_the_partial_correspondence(dataset: dict) -> None:
    held_out = dataset["held_out"]
    assert held_out["sweep_v2"]["applies"] and held_out["sweep_v2"]["passed_count"] == 96
    assert held_out["characterization_v1"]["applies"] and held_out["characterization_v1"]["passed_count"] == 56 - RECORDED_FAILING_DESIGN_COUNT
    assert held_out["characterization_v1"]["observed_null_count"] > held_out["characterization_v1"]["reference_null_count"]
    p2 = dataset["p2_consistency"]
    assert p2 is not None and p2["role"].endswith("not a gate") and p2["cusp_count_equals_reference_count"] is True
