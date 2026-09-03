"""Post-execution consistency of the sealed geometry screening bundle (skips before execution)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.orbit_mc import wilson_interval

from experiments.orbit_wall_loss_geometry_screening_v1 import experiment as E
from experiments.orbit_wall_loss_geometry_screening_v1.consumer import verify_handoff

RESULTS = E.RESULTS_ROOT
ARTIFACTS = RESULTS / "artifacts"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="not executed yet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return strict_json_file(RESULTS / "manifest.json")


@pytest.fixture(scope="module")
def dataset() -> dict:
    return strict_json_file(ARTIFACTS / "geometry-wall-loss-dataset.json")


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


def test_classification_is_screening_everywhere(dataset: dict, campaign: dict) -> None:
    assert dataset["classification"] == campaign["classification"] == CLASSIFICATION
    assert campaign["status"] == "accepted_screening_dataset" and campaign["evidentiary"] is True
    assert dataset["claim_boundary"]["not_p2_qualified"] is True
    assert dataset["field_source"]["field_status"] == "accepted_L1a_screening_not_P2_qualified"
    for row in dataset["designs"]:
        assert row["classification"] == CLASSIFICATION
    protocol = strict_json_file(ARTIFACTS / "protocol.json")
    assert protocol == E.protocol()


def test_dataset_rows_agree_with_sealed_summaries_and_endpoints(dataset: dict) -> None:
    value = E.protocol()
    for row in dataset["designs"]:
        for key, case in row["cases"].items():
            summary = strict_json_file(ARTIFACTS / "summaries" / f"{row['case_id']}--{key}.json")
            assert summary["summary"]["wall_hit"] == case["wall_hit"]
            assert summary["summary"]["termination_counts"] == case["termination_counts"]
            assert summary["sealed"] == case["sealed"]
            endpoints = json.loads(gzip.decompress((ARTIFACTS / "endpoints" / f"{row['case_id']}--{key}.json.gz").read_bytes()))
            assert len(endpoints["rows"]) == case["trial_count"] == 512
            counts = {}
            for item in endpoints["rows"]:
                counts[item["termination"]] = counts.get(item["termination"], 0) + 1
            for name in ("wall_hit", "reflected", "domain_escape"):
                assert counts.get(name, 0) == case["termination_counts"][name]
            assert all(item["maximum_relative_energy_error"] == 0.0 for item in endpoints["rows"])
            assert hashlib.sha256(gzip.decompress((ARTIFACTS / "endpoints" / f"{row['case_id']}--{key}.json.gz").read_bytes())).hexdigest() == case["endpoints_payload_sha256"]
            if case["sealed"]:
                sidecar = (ARTIFACTS / "orbits" / f"{row['case_id']}--{key}.json.sha256").read_text(encoding="ascii")
                assert sidecar.split()[0] == case["orbit_artifact_file_sha256"]
                handoff = strict_json_file(ARTIFACTS / "handoffs" / f"{row['case_id']}--{key}.json")
                derived = verify_handoff(handoff)
                assert derived["probability"] == case["wall_hit"]["probability"]
                assert handoff["orbit_result_artifact_sha256"] == case["orbit_artifact_file_sha256"]
            else:
                assert not (ARTIFACTS / "handoffs" / f"{row['case_id']}--{key}.json").exists()
        coarse = row["cases"]["accepted-N"]["wall_hit"]["probability"]
        fine = row["cases"]["accepted-2N"]["wall_hit"]["probability"]
        convergence = row["convergence"]
        assert convergence["successive_change"] == abs(fine - coarse)
        assert convergence["converged"] == (
            abs(fine - coarse) <= value["gates"]["maximum_successive_probability_change"] and convergence["adjacent_wilson_overlap"]
        )
        assert row["gates"]["converged"] == convergence["converged"]
        assert row["gates"]["sealed"] == convergence["converged"]
        reported = row["reported"]["wall_hit"]
        assert reported == {**reported, **json.loads(json.dumps(wilson_interval(reported["successes"], reported["trials"]).__dict__))}
        total = sum(row["per_stratum"]["accepted-2N"][i]["wall_hit"] for i in range(32))
        assert total == reported["successes"]


def test_csv_matches_dataset(dataset: dict) -> None:
    rows = list(csv.reader(io.StringIO((ARTIFACTS / "geometry-wall-loss-dataset.csv").read_text(encoding="utf-8"))))
    assert rows[0] == list(E.CSV_COLUMNS)
    assert len(rows) == dataset["design_count"] + 1
    by_id = {row["case_id"]: row for row in dataset["designs"]}
    p_index = rows[0].index("p_wall_2N")
    label_index = rows[0].index("classification")
    for line in rows[1:]:
        assert float(line[p_index]) == by_id[line[0]]["reported"]["wall_hit"]["probability"]
        assert line[label_index] == CLASSIFICATION


def test_gates_and_headline_are_consistent(dataset: dict, campaign: dict) -> None:
    gates = strict_json_file(ARTIFACTS / "gates.json")
    assert gates["passed"] is True and gates["structural_all_passed"] is True
    assert gates["design_count"] == dataset["design_count"] == campaign["design_count"]
    assert gates["converged_design_count"] == dataset["headline"]["converged_design_count"]
    assert gates["sealed_case_count"] == sum(case["sealed"] for row in dataset["designs"] for case in row["cases"].values())
    wall = [row["reported"]["wall_hit"]["probability"] for row in dataset["designs"]]
    assert dataset["headline"]["wall_hit_probability_min"] == min(wall)
    assert dataset["headline"]["wall_hit_probability_max"] == max(wall)
    assert campaign["headline"] == dataset["headline"]
    exclusions = strict_json_file(ARTIFACTS / "design-exclusions.json")
    assert exclusions["excluded"] == dataset["excluded_designs"]
    assert dataset["design_count"] + len(exclusions["excluded"]) == len(E.design_case_ids(E.protocol()))


def test_consumer_record_consumes_v4_and_every_sealed_design(dataset: dict) -> None:
    record = strict_json_file(ARTIFACTS / "coupling-consumer-record.json")
    assert record["v4_reference"]["passed"] is True
    assert record["v4_reference"]["design_in_screening_set"] is False
    assert record["v4_reference"]["reference_row"]["probability"] == 330 / 512
    consumed = {item["case_id"]: item for item in record["screening_designs_consumed"]}
    assert set(consumed) == {row["case_id"] for row in dataset["designs"]}
    for row in dataset["designs"]:
        item = consumed[row["case_id"]]
        assert item["label"] == CLASSIFICATION
        assert item["sealed"] == row["cases"]["accepted-2N"]["sealed"]
        if item["sealed"]:
            assert item["consumed"]["passed"] is True
            assert item["consumption_status"] == "consumed_verified_handoff"


def test_frozen_preregistration_files_match_the_bundle() -> None:
    assert (ARTIFACTS / "shakedown.json").read_bytes() == E.SHAKEDOWN_PATH.read_bytes()
    assert strict_json_file(ARTIFACTS / "authorities.json") == strict_json_file(E.AUTHORITIES_PATH)
    assert strict_json_file(ARTIFACTS / "design-authorities.json") == strict_json_file(E.DESIGN_AUTHORITIES_PATH)
