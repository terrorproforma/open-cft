"""Post-execution consistency of the sealed v2 bundle (skips before execution)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict

import pytest

from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.orbit_mc import wilson_interval

from experiments.orbit_wall_loss_geometry_screening_v2 import cells as C
from experiments.orbit_wall_loss_geometry_screening_v2 import experiment as E
from experiments.orbit_wall_loss_geometry_screening_v2.consumer import verify_handoff

RESULTS = E.RESULTS_ROOT
ARTIFACTS = RESULTS / "artifacts"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="not executed yet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return strict_json_file(RESULTS / "manifest.json")


@pytest.fixture(scope="module")
def dataset() -> dict:
    return strict_json_file(ARTIFACTS / "geometry-wall-loss-dataset-v2.json")


@pytest.fixture(scope="module")
def campaign() -> dict:
    return strict_json_file(ARTIFACTS / "campaign-result.json")


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
        if entry["path"].endswith((".json", ".sha256", ".csv")):
            assert b"\r" not in raw, entry["path"]


def test_classification_and_labels(dataset: dict, campaign: dict) -> None:
    assert dataset["classification"] == campaign["classification"] == CLASSIFICATION
    assert campaign["status"] == "accepted_screening_dataset" and campaign["evidentiary"] is True
    assert dataset["claim_boundary"]["not_p2_qualified"] is True and dataset["claim_boundary"]["p2_row_is_not_v4_replication"] is True
    for row in dataset["designs"]:
        if row["set_id"] == "sweep_v2":
            assert row["label"] == CLASSIFICATION == row["classification"]
        else:
            assert row["set_id"] == "p2_divergent_exit" and row["label"] == "P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN"
    assert strict_json_file(ARTIFACTS / "protocol.json") == E.protocol()
    assert dataset["catalogue_file_sha256"] == E.protocol()["cusp_cell_catalogue"]["catalogue_file_sha256"]


def test_case_sizes_are_wilson_exact_and_cells_pool_their_n_blocks(dataset: dict) -> None:
    value = E.protocol()
    for row in dataset["designs"]:
        by_cell: dict[str, dict[str, int]] = {}
        for key, case in row["cases"].items():
            assert E.wilson_exact_at_ends(case["trial_count"])
            summary = strict_json_file(ARTIFACTS / "summaries" / f"{key}.json")
            assert summary["summary"]["wall_hit"] == case["wall_hit"]
            assert summary["summary"]["termination_counts"] == case["termination_counts"]
            assert summary["sealed"] == case["sealed"]
            endpoints = json.loads(gzip.decompress((ARTIFACTS / "endpoints" / f"{key}.json.gz").read_bytes()))
            assert len(endpoints["rows"]) == case["trial_count"]
            assert hashlib.sha256(gzip.decompress((ARTIFACTS / "endpoints" / f"{key}.json.gz").read_bytes())).hexdigest() == case["endpoints_payload_sha256"]
            assert all(item["maximum_relative_energy_error"] == 0.0 for item in endpoints["rows"])
            assert all(item["cell_id"] == case["cell_id"] for item in endpoints["rows"])
            if case["timestep"] == "N":
                cell = by_cell.setdefault(case["cell_id"], {"trials": 0, "wall_hit": 0, "reflected": 0, "domain_escape": 0})
                cell["trials"] += case["trial_count"]
                for name in ("wall_hit", "reflected", "domain_escape"):
                    cell[name] += case["termination_counts"][name]
            if case["sealed"]:
                sidecar = (ARTIFACTS / "orbits" / f"{key}.json.sha256").read_text(encoding="ascii")
                assert sidecar.split()[0] == case["orbit_artifact_file_sha256"]
                handoff = strict_json_file(ARTIFACTS / "handoffs" / f"{key}.json")
                derived = verify_handoff(handoff)
                assert derived["probability"] == case["wall_hit"]["probability"]
                assert handoff["orbit_result_artifact_sha256"] == case["orbit_artifact_file_sha256"]
            else:
                assert not (ARTIFACTS / "handoffs" / f"{key}.json").exists()
        for cell in row["cells"]:
            pooled = by_cell[cell["cell_id"]]
            final = cell["final"]
            assert final["trials"] == pooled["trials"] and final["wall_hit"] == pooled["wall_hit"]
            assert final["reflected"] == pooled["reflected"] and final["domain_escape"] == pooled["domain_escape"]
            assert final["p_wall"] == asdict(wilson_interval(final["wall_hit"], final["trials"]))
            assert final["trials"] in (128, 512)
            assert (final["trials"] == 512) == cell["topped_up"]
            assert cell["topped_up"] == (C.wilson_width(cell["stage1"]["wall_hit"], 128) > value["allocation"]["wilson_width_threshold"])
            assert final["jeffreys_floor"] == C.jeffreys_floor(final["wall_hit"], final["trials"])
            assert final["surrogate_ready"] == (final["jeffreys_floor"] <= value["estimators"]["surrogate_readiness_floor"])
            assert cell["control"]["n_control"] == final["trials"] // 8
        strata_total = sum(item["wall_hit"] for item in row["per_stratum_final"])
        assert strata_total == sum(cell["final"]["wall_hit"] for cell in row["cells"])
        assert row["allocation_replay"]["passed"] is True
        assert row["gates"]["structural_passed"] is True
        assert row["sealed"] == row["convergence_flags"]["timestep_passed"]
        assert row["convergence_flags"]["timestep_passed"] == (abs(row["control"]["delta_p_wall"]) <= 0.02)


def test_csv_has_one_row_per_cell(dataset: dict) -> None:
    rows = list(csv.reader(io.StringIO((ARTIFACTS / "geometry-wall-loss-dataset-v2.csv").read_text(encoding="utf-8"))))
    assert rows[0] == list(E.CSV_COLUMNS)
    assert len(rows) == dataset["cell_count"] + 1
    p_index = rows[0].index("p_wall")
    n_index = rows[0].index("n_final")
    by_key = {(row["design_key"], cell["cell_id"]): cell for row in dataset["designs"] for cell in row["cells"]}
    for line in rows[1:]:
        cell = by_key[(line[0], line[6])]
        assert float(line[p_index]) == cell["final"]["p_wall"]["probability"]
        assert int(line[n_index]) == cell["final"]["trials"]


def test_gates_control_and_headline_are_consistent(dataset: dict, campaign: dict) -> None:
    gates = strict_json_file(ARTIFACTS / "gates.json")
    assert gates["passed"] is True and gates["structural_all_passed"] is True and gates["allocation_replay_all_passed"] is True
    assert gates["control_gate"]["passed"] is True and abs(gates["control_gate"]["estimated_bias_2N_minus_N"]) <= 0.02
    assert gates["design_count"] == dataset["design_count"] == campaign["design_count"]
    assert gates["sealed_case_count"] == sum(case["sealed"] for row in dataset["designs"] for case in row["cases"].values())
    headline = dataset["headline"]
    cells = [cell for row in dataset["designs"] for cell in row["cells"]]
    assert headline["cell_count"] == len(cells) == dataset["cell_count"]
    assert headline["cells_topped_up"] == sum(cell["topped_up"] for cell in cells)
    assert headline["cells_saturated_after_stage1"] == len(cells) - headline["cells_topped_up"]
    assert headline["cells_surrogate_ready"] == sum(cell["final"]["surrogate_ready"] for cell in cells)
    assert headline["control"]["n_control"] == sum(row["control"]["n_control"] for row in dataset["designs"])
    assert campaign["headline"] == headline
    allocation = strict_json_file(ARTIFACTS / "allocation-decisions.json")
    assert allocation["summary"]["topped_up_cells"] == headline["cells_topped_up"] and allocation["summary"]["replay_all_passed"] is True
    exclusions = strict_json_file(ARTIFACTS / "design-exclusions.json")
    assert exclusions["excluded"] == dataset["excluded_designs"]
    assert dataset["design_count"] + len(exclusions["excluded"]) == len(E.design_keys(E.protocol()))


def test_consumer_and_v1_comparison_records(dataset: dict) -> None:
    record = strict_json_file(ARTIFACTS / "coupling-consumer-record.json")
    assert record["v4_reference"]["passed"] is True and record["v4_reference"]["design_in_screening_set"] is False
    assert record["v4_reference"]["reference_row"]["probability"] == 330 / 512
    consumed = {(item["design_key"], item["case_key"]): item for item in record["screening_cases_consumed"]}
    for row in dataset["designs"]:
        for key, case in row["cases"].items():
            item = consumed[(row["design_key"], key)]
            assert item["sealed"] == case["sealed"]
            if case["sealed"]:
                assert item["consumed"]["passed"] is True and item["consumption_status"] == "consumed_verified_handoff"
    comparison = strict_json_file(ARTIFACTS / "v1-comparison.json")
    assert comparison["design_count"] == sum(row["set_id"] == "sweep_v2" for row in dataset["designs"])
    assert dataset["headline"]["v1_comparison"]["spearman_rank_correlation"] == comparison["spearman_rank_correlation"]
    for row in dataset["designs"]:
        if row["set_id"] == "sweep_v2":
            assert row["v1_comparison"]["v1_trials"] == 512
            assert row["v1_comparison"]["comparison"]["launches"]["v2_probability"] == row["pooled"]["launches"]["probability"]
        else:
            assert row["v1_comparison"] is None


def test_frozen_preregistration_files_match_the_bundle() -> None:
    assert (ARTIFACTS / "shakedown.json").read_bytes() == E.SHAKEDOWN_PATH.read_bytes()
    assert strict_json_file(ARTIFACTS / "authorities.json") == strict_json_file(E.AUTHORITIES_PATH)
    assert strict_json_file(ARTIFACTS / "design-authorities.json") == strict_json_file(E.DESIGN_AUTHORITIES_PATH)
