"""Lifecycle-aware consistency of the sealed sweep-v3 bundle (skips before execution)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json

import pytest

from cft_revival.experiment_runtime.canonical import canonical_bytes, semantic_sha256, strict_json_file

from experiments.cusp_topology_search_v3_1 import topology as T
from experiments.l1a_geometry_sweep_v3 import catalogue as C
from experiments.l1a_geometry_sweep_v3 import descriptors as DS
from experiments.l1a_geometry_sweep_v3 import experiment as E

RESULTS = E.RESULTS_ROOT
ARTIFACTS = RESULTS / "artifacts"

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="not executed yet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return strict_json_file(RESULTS / "manifest.json")


@pytest.fixture(scope="module")
def dataset() -> dict:
    return strict_json_file(ARTIFACTS / "sweep-dataset.json")


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
    assert lock["command"].endswith("run execute") and "gpu-not-used" in lock["device"]
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
    assert strict_json_file(E.AUTHORITIES_PATH)["protocol_semantic_sha256"] == semantic_sha256(E.protocol())


def test_gates_headline_and_labels_are_consistent(dataset: dict, campaign: dict, gates: dict) -> None:
    assert dataset["classification"] == campaign["classification"] == E.CLASSIFICATION
    assert campaign["status"] == "accepted_l1a_sweep_v3" and campaign["evidentiary"] is True
    assert gates["passed"] is True and gates["binding"] is True and all(gates["campaign"].values())
    assert all(designs == [] for designs in gates["failing_designs"].values())
    assert all(item["passed"] for item in gates["sweep_v2_gate_breakdown"].values())
    assert dataset["design_count"] == campaign["design_count"] == gates["design_count"] == 224
    assert dataset["gates"]["campaign"] == gates["campaign"] and campaign["headline"] == dataset["headline"]
    assert gates["replays"] and all(item["bit_identical"] and item["field_identity_equal"] and item["accepted_grid_equal"] for item in gates["replays"])
    assert len(gates["replays"]) == 3
    for row in dataset["designs"]:
        assert row["label"] == E.TOPOLOGY_LABEL and row["classification"] == E.CLASSIFICATION
        assert all(row["gate_checks"].values()), row["key"]
        assert row["stability"]["stable"] is True and row["v2_gates"]["passed"] is True
    headline = dataset["headline"]
    assert headline["stable_design_count"] == 224 and headline["v2_gates_passed_count"] == 224
    assert headline["set_counts"] == {"sobol_v3": 128, "sweep_v2": 96}
    sobol = [row for row in dataset["designs"] if row["set_id"] == "sobol_v3"]
    assert headline["sobol_hemp_like_count"] == sum(row["hemp_like_all_cusps"] for row in sobol)
    assert headline["sobol_five_stage_four_cusp_hemp_like_count"] == sum(row["five_stage_four_cusp_hemp_like"] for row in sobol)
    assert headline["sobol_predicted_hemp_like_i1_count"] == 51
    v2_region = [row for row in dataset["designs"] if row["set_id"] == "sweep_v2" or row["inside_sweep_v2_box"]]
    assert len(v2_region) == 96 + 6
    assert headline["sweep_v2_region_hemp_like_count"] == sum(row["hemp_like_all_cusps"] for row in v2_region)
    test = headline["sobol_hypothesis_test"]
    assert test["cusp_count"] == sum(row["wall_cusp_count"] for row in sobol)
    assert test["x_star_prediction"] == DS.X_STAR_HEMP_LIKE


def test_held_out_sweep_v2_reproduction_is_complete(dataset: dict) -> None:
    held_out = dataset["held_out"]
    assert held_out["applies"] and held_out["design_count"] == 96
    assert held_out["passed_count"] == 96 and held_out["qoi_replay_passed_count"] == 96 and held_out["axis_null_bijection_count"] == 96
    assert held_out["reference_null_count"] == held_out["observed_null_count"]
    assert held_out["stored_representatives_checked"] == 4
    assert held_out["max_axis_null_difference_m"] <= 2.5e-4


def test_design_records_reproduce_the_rows_and_the_catalogue(dataset: dict) -> None:
    catalogue = C.load_catalogue(RESULTS)
    entries = {(entry["set_id"], entry["design_id"]): entry for entry in catalogue["entries"]}
    assert catalogue["design_count"] == 224 and catalogue["hemp_like_design_count"] == sum(entry["hemp_like_all_cusps"] for entry in catalogue["entries"])
    value = E.protocol()
    tolerance = value["definition_v3_import"]["stability_tolerance_m"]
    for row in dataset["designs"]:
        record = strict_json_file(RESULTS / row["record_path"])
        assert record["status"] == "resolved" and record["key"] == row["key"]
        assert E.dataset_row(record) == row
        assert record["gate_checks"] == E.design_gate_checks(record)
        assert T.compare_resolutions(record["accepted"], record["refined"], tolerance) == record["stability"]
        assert C.catalogue_entry(record) == entries[(row["set_id"], row["design_id"])]
        payload = {key: record[key] for key in ("axis_window_m", "qois", "v2_gates", "accepted", "refined", "stability", "held_out", "descriptors")}
        assert hashlib.sha256(canonical_bytes(payload)).hexdigest() == record["topology_payload_sha256"]
        grid_raw = gzip.decompress((RESULTS / record["accepted_grid_path"]).read_bytes())
        assert hashlib.sha256(grid_raw).hexdigest() == record["accepted_grid_payload_sha256"]
        grid = json.loads(grid_raw)
        assert grid["identity"] == record["identity"] and len(grid["r_m"]) == 81 and len(grid["z_m"]) == 145
        descriptors = record["descriptors"]["accepted"]
        for cusp_row in descriptors["cusps"]:
            expected = cusp_row["wall_b_t"] / max(cusp_row["upstream_axis_peak_t"], cusp_row["downstream_axis_peak_t"])
            assert cusp_row["rho_conservative"] == pytest.approx(expected, rel=1e-12)
            assert cusp_row["hemp_like_conservative"] == (cusp_row["rho_conservative"] >= DS.HEMP_LIKE_RHO)
        assert descriptors["hemp_like_all_cusps"] == (bool(descriptors["cusps"]) and all(c["hemp_like_conservative"] for c in descriptors["cusps"]))
        assert descriptors["profiles"] is not None and len(descriptors["profiles"]["z_m"]) == 241
        if record["set_id"] == "sobol_v3":
            assert all(trace["path_rz_m"] is not None for trace in record["accepted"]["separatrix_traces"])
            assert record["evidence"]["derived_geometry"]["feasibility"]["feasible"] is True
        else:
            assert record["held_out"]["applies"] and record["held_out"]["qoi_replay_passed"]


def test_csv_matches_dataset(dataset: dict) -> None:
    rows = list(csv.reader(io.StringIO((ARTIFACTS / "sweep-dataset.csv").read_text(encoding="utf-8"))))
    assert rows[0] == list(E.CSV_COLUMNS)
    assert len(rows) == dataset["design_count"] + 1
    assert E.dataset_csv(dataset["designs"]) == (ARTIFACTS / "sweep-dataset.csv").read_bytes()
