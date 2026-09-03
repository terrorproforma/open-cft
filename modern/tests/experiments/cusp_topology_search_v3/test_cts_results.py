"""Lifecycle-aware consistency of the sealed v3 bundle (skips before execution)."""

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

RESULTS = E.RESULTS_ROOT
ARTIFACTS = RESULTS / "artifacts"

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


def test_bundle_is_accepted_and_byte_exact(manifest: dict) -> None:
    assert manifest["state"] == "accepted_result"
    terminal = strict_json_file(RESULTS / "terminal.json")
    assert terminal["state"] == "accepted_result" and terminal["primary_error"] is None
    lock = strict_json_file(RESULTS / "execution-lock.json")
    assert lock["command"].endswith("run execute")
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


def test_labels_gates_and_headline_are_consistent(dataset: dict, campaign: dict, gates: dict) -> None:
    assert dataset["classification"] == campaign["classification"] == E.CLASSIFICATION
    assert campaign["status"] == "accepted_topology_screening" and campaign["evidentiary"] is True
    assert gates["passed"] is True and gates["binding"] is True and all(gates["campaign"].values())
    assert dataset["design_count"] == campaign["design_count"] == gates["design_count"] == 281
    assert dataset["gates"]["campaign"] == gates["campaign"]
    assert campaign["headline"] == dataset["headline"]
    assert gates["replays"] and all(item["bit_identical"] and item["field_identity_equal"] and item["accepted_grid_equal"] for item in gates["replays"])
    for row in dataset["designs"]:
        assert row["label"] == (E.P2_CLASSIFICATION if row["set_id"] == "p2_divergent_exit" else E.CLASSIFICATION)
        assert all(row["gate_checks"].values()), row["key"]
        assert row["stability"]["stable"] is True
    headline = dataset["headline"]
    assert headline["stable_design_count"] == 281
    histogram: dict[str, int] = {}
    for row in dataset["designs"]:
        histogram[str(row["wall_cusp_count"])] = histogram.get(str(row["wall_cusp_count"]), 0) + 1
    assert dataset["estimands"]["pooled_all"]["wall_cusp_count_histogram"] == dict(sorted(histogram.items(), key=lambda pair: int(pair[0])))
    for set_id, count in {"sweep_v2": 96, "four_cell_v2": 128, "characterization_v1": 56, "p2_divergent_exit": 1}.items():
        rows = [row for row in dataset["designs"] if row["set_id"] == set_id]
        assert len(rows) == count
        assert dataset["estimands"][set_id]["four_wall_cusp_fraction"] == sum(row["four_wall_cusps"] for row in rows) / count


def test_design_records_reproduce_the_dataset_rows_and_the_catalogue(dataset: dict) -> None:
    catalogue = C.load_catalogue(RESULTS)
    entries = {(entry["set_id"], entry["design_id"]): entry for entry in catalogue["entries"]}
    value = E.protocol()
    tolerance = value["definition_v3"]["stability_tolerance_m"]
    for row in dataset["designs"]:
        record = strict_json_file(RESULTS / row["record_path"])
        assert record["status"] == "resolved" and record["key"] == row["key"]
        assert E.dataset_row(record) == row
        assert record["gate_checks"] == E.design_gate_checks(record)
        stability = T.compare_resolutions(record["accepted"], record["refined"], tolerance)
        assert stability == record["stability"]
        entry = entries[(row["set_id"], row["design_id"])]
        assert C.catalogue_entry(record) == entry
        # canonical topology payload hash re-derives from the record
        payload = {key: record[key] for key in ("axis_window_m", "accepted", "refined", "stability", "held_out", "p2_consistency")}
        assert hashlib.sha256(canonical_bytes(payload)).hexdigest() == record["topology_payload_sha256"]
        # the accepted tracing grid is sealed and re-hashes
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
    by_key = {(row["set_id"], row["design_id"]): row for row in dataset["designs"]}
    count_index = rows[0].index("wall_cusp_count")
    z_index = rows[0].index("wall_cusp_z_m")
    for line in rows[1:]:
        row = by_key[(line[0], line[1])]
        assert int(line[count_index]) == row["wall_cusp_count"]
        assert line[z_index] == ";".join(repr(cusp["z_c_m"]) for cusp in row["wall_cusps"])
    assert dataset_csv_bytes_match(dataset)


def dataset_csv_bytes_match(dataset: dict) -> bool:
    return E.dataset_csv(dataset["designs"]) == (ARTIFACTS / "topology-dataset.csv").read_bytes()


def test_held_out_and_p2_consistency_are_recorded(dataset: dict) -> None:
    held_out = dataset["held_out"]
    assert held_out["characterization_v1"]["applies"] and held_out["characterization_v1"]["passed_count"] == 56
    assert held_out["sweep_v2"]["applies"] and held_out["sweep_v2"]["passed_count"] == 96
    assert held_out["characterization_v1"]["reference_null_count"] == held_out["characterization_v1"]["observed_null_count"]
    p2 = dataset["p2_consistency"]
    assert p2 is not None and p2["role"].endswith("not a gate")
    assert len(p2["cusps"]) == dataset["designs"][-1]["wall_cusp_count"] if dataset["designs"][-1]["set_id"] == "p2_divergent_exit" else True
