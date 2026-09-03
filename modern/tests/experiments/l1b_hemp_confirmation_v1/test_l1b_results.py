"""Lifecycle-aware consistency of the sealed L1b confirmation bundle (skips before execution)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json

import pytest

from cft_revival.experiment_runtime.canonical import (
    canonical_bytes,
    semantic_sha256,
    strict_json_file,
)
from experiments.cusp_topology_search_v3_1 import topology as T
from experiments.l1a_geometry_sweep_v3 import descriptors as DS
from experiments.l1b_hemp_confirmation_v1 import designs as D
from experiments.l1b_hemp_confirmation_v1 import experiment as E

RESULTS = E.RESULTS_ROOT
ARTIFACTS = RESULTS / "artifacts"

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="not executed yet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return strict_json_file(RESULTS / "manifest.json")


@pytest.fixture(scope="module")
def dataset() -> dict:
    return strict_json_file(ARTIFACTS / "confirmation-dataset.json")


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
    assert lock["command"].endswith("run execute") and "gpu-not-used" in lock["device"] and "fem_reference" in lock["device"]
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


def test_gates_verdict_and_labels_are_consistent(dataset: dict, campaign: dict, gates: dict) -> None:
    assert dataset["classification"] == campaign["classification"] == E.CLASSIFICATION
    assert campaign["evidentiary"] is True and campaign["status"] == f"accepted_l1b_confirmation_{campaign['verdict'].lower()}"
    assert campaign["verdict"] in E.VERDICTS and campaign["verdict"] == gates["confirmation"]["verdict"] == dataset["headline"]["verdict"]
    assert gates["passed"] is True and gates["binding"] is True and all(gates["campaign"].values())
    assert all(designs == [] for designs in gates["failing_designs"].values())
    assert dataset["design_count"] == campaign["design_count"] == gates["design_count"] == 15
    assert dataset["gates"]["campaign"] == gates["campaign"] and campaign["headline"] == dataset["headline"]
    assert len(gates["replays"]) == 1 and all(item["bit_identical"] and item["field_identity_equal"] and item["accepted_grid_equal"] and item["p2_run_sha256_equal"] for item in gates["replays"])
    assert gates["peak_rss_bytes"] <= gates["ram_budget"]["budget_bytes"]
    confirmation = gates["confirmation"]
    b_passed = confirmation["cusp_count_unchanged"]["fraction_boundary_tolerant"] >= 1.0
    c_passed = confirmation["cusp_position_shift"]["all_designs_bijective"] and confirmation["cusp_position_shift"]["max_shift_over_tolerance"] <= 1.0
    assert confirmation["cusp_count_unchanged"]["passed"] == b_passed and confirmation["cusp_position_shift"]["passed"] == c_passed
    expected = "CONFIRMED" if (b_passed and c_passed) else ("PARTIALLY_CONFIRMED" if (b_passed or c_passed) else "DISCONFIRMED")
    assert confirmation["verdict"] == expected
    assert campaign["confirmation_gates"] == {"cusp_count_unchanged": b_passed, "cusp_position_shift": c_passed}
    for row in dataset["designs"]:
        assert row["label"] == E.TOPOLOGY_LABEL and row["classification"] == E.CLASSIFICATION
        assert all(row["gate_checks"].values()), row["key"]
        assert row["sampling_stability"]["stable"] is True and row["p2"]["all_levels_converged"] is True
        assert all(level["converged"] and level["relative_true_residual_l2"] <= 2.0e-10 for level in row["p2"]["levels"])
        assert row["p2"]["levels"][-1]["p2_dofs"] <= E.protocol()["p2"]["resources"]["maximum_p2_dofs"]
    assert len(campaign["agreement_table"]) == 15 and campaign["agreement_table"] == dataset["agreement_table"]


def test_design_records_reproduce_the_rows_the_comparison_and_the_sealed_l1a_reference(dataset: dict) -> None:
    value = E.protocol()
    tolerance = value["definition_v3_import"]["stability_tolerance_m"]
    for row in dataset["designs"]:
        record = strict_json_file(RESULTS / row["record_path"])
        assert record["status"] == "resolved" and record["key"] == row["key"]
        assert E.dataset_row(record) == row and E.agreement_row(record) in dataset["agreement_table"]
        assert record["gate_checks"] == E.design_gate_checks(record)
        assert T.compare_resolutions(record["accepted"], record["refined"], tolerance) == record["sampling_stability"]
        discretisation = T.compare_resolutions(record["coarse"], record["accepted"], tolerance)
        assert {key: record["p2_discretisation"][key] for key in discretisation} == discretisation
        reference = D.l1a_reference(record["design_id"])
        assert record["l1a_reference"] == reference
        geometry = T.ChannelGeometry(**{key: (tuple(item) if key == "stage_centres_m" else item) for key, item in record["geometry"].items()})
        comparison = E.compare_to_l1a(reference, record["accepted"], record["descriptors"]["accepted"], geometry, value, source_strength_scale=reference["source_strength_scale"])
        assert comparison == record["comparison"]
        payload = {key: record[key] for key in ("axis_window_m", "axis_window_reproduced", *E.MAP_ROLES, "descriptors", "sampling_stability", "p2_discretisation", "comparison")}
        assert hashlib.sha256(canonical_bytes(payload)).hexdigest() == record["comparison_payload_sha256"]
        grid_raw = gzip.decompress((RESULTS / record["accepted_grid_path"]).read_bytes())
        assert hashlib.sha256(grid_raw).hexdigest() == record["accepted_grid_payload_sha256"]
        grid = json.loads(grid_raw)
        assert grid["identity"] == record["identity"] and len(grid["r_m"]) == 33 and len(grid["psi_wb"]) == 33 and len(grid["b_r_t"]) == 33
        assert grid["r_m"][-1] == record["geometry"]["wall_radius_m"]
        descriptors = record["descriptors"]["accepted"]
        for cusp_row in descriptors["cusps"]:
            expected = cusp_row["wall_b_t"] / max(cusp_row["upstream_axis_peak_t"], cusp_row["downstream_axis_peak_t"])
            assert cusp_row["rho_conservative"] == pytest.approx(expected, rel=1e-12)
            assert cusp_row["hemp_like_conservative"] == (cusp_row["rho_conservative"] >= DS.HEMP_LIKE_RHO)
        assert descriptors["profiles"] is not None and all(trace["path_rz_m"] is not None for trace in record["accepted"]["separatrix_traces"])
        assert record["accepted"]["axis_nulls"]["window_m"] == reference["axis_window_m"]
        assert record["comparison"]["cusp_position_tolerance_m"] == E.cusp_position_tolerance_m(record["geometry"]["wall_radius_m"], value)


def test_csv_matches_dataset(dataset: dict) -> None:
    rows = list(csv.reader(io.StringIO((ARTIFACTS / "confirmation-dataset.csv").read_text(encoding="utf-8"))))
    assert rows[0] == list(E.CSV_COLUMNS)
    assert len(rows) == dataset["design_count"] + 1
    assert E.dataset_csv(dataset["designs"]) == (ARTIFACTS / "confirmation-dataset.csv").read_bytes()
