"""Lifecycle-aware checks of the recorded bundle (or of its absence before execution)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from cft_revival.experiment_runtime import semantic_sha256
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.wall_loss_geometry_surrogate_v1 import data as d
from experiments.wall_loss_geometry_surrogate_v1 import experiment, models as m
from experiments.wall_loss_geometry_surrogate_v1.experiment import ALL_OUTPUTS, GATED_OUTPUTS, PARTITIONS_PATH, RESULTS_ROOT, protocol
from experiments.wall_loss_geometry_surrogate_v1.predictor import Predictor

MANIFEST = RESULTS_ROOT / "manifest.json"


def _recorded() -> dict:
    if not MANIFEST.is_file():
        pytest.skip("campaign not executed yet")
    return strict_json_file(MANIFEST)


def _artifact(name: str):
    return strict_json_file(RESULTS_ROOT / "artifacts" / name)


def test_before_execution_no_results_exist() -> None:
    if MANIFEST.is_file():
        pytest.skip("campaign executed")
    assert not RESULTS_ROOT.exists()
    assert protocol()["status"] == "preregistered_pending_single_execution"


def test_recorded_bundle_inventory_is_byte_exact() -> None:
    manifest = _recorded()
    assert manifest["experiment_id"] == "wall-loss-geometry-surrogate-v1"
    assert manifest["state"] in {"accepted_result", "assessment_rejection", "development_rejection", "runtime_failure", "prebundle_failure"}
    mismatches = []
    for entry in manifest["artifacts"]:
        if entry["type"] != "file":
            continue
        path = RESULTS_ROOT / Path(*entry["path"].split("/"))
        assert path.is_file(), entry["path"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["byte_sha256"] or len(data) != entry["bytes"]:
            mismatches.append(entry["path"])
    assert mismatches == []
    terminal = strict_json_file(RESULTS_ROOT / "terminal.json")
    assert terminal["state"] == manifest["state"]


def test_recorded_partition_and_label_access_order() -> None:
    _recorded()
    bundle_partition = _artifact("partitions.json")
    assert semantic_sha256(bundle_partition) == semantic_sha256(strict_json_file(PARTITIONS_PATH))
    value = protocol()
    rows = experiment.load_rows(value)
    recomputed = experiment.plan_partition(value, experiment.evidentiary_plan(value), rows)
    assert semantic_sha256(recomputed) == semantic_sha256(bundle_partition)
    access = []
    for path in sorted((RESULTS_ROOT / "access").glob("*.json")):
        if path.name.endswith(".sha256.json"):
            continue
        access.append(strict_json_file(path))
    labels = [row["details"]["role"] for row in sorted(access, key=lambda r: r["sequence"]) if row["kind"] == "label"]
    assert labels == ["fit", "method-selection", "calibration", "assessment", "extrapolation"]
    solvers = [row["operation"] for row in access if row["kind"] == "solver"]
    assert any(op.startswith("fit-") for op in solvers) and any(op.startswith("refit-") for op in solvers)


def test_recorded_gates_metrics_and_campaign_result_agree() -> None:
    manifest = _recorded()
    gates = _artifact("gates.json")
    result = _artifact("campaign-result.json")
    metrics = _artifact("metrics.json")
    assert gates["binding_in_this_plan"] is True
    assert gates["decision_basis"] == "all binding gates"
    assert set(gates["binding"]) == set(experiment.STRUCTURAL_GATES) | set(experiment.SCIENCE_GATES)
    assert result["all_binding_gates_passed"] == gates["all_binding_passed"] == all(item["passed"] for item in gates["binding"].values())
    assert result["binding_gate_results"] == {name: item["passed"] for name, item in gates["binding"].items()}
    assert result["status"] == ("accepted_surrogate" if gates["all_binding_passed"] else "rejected_surrogate")
    assert (manifest["state"] == "accepted_result") == gates["all_binding_passed"]
    assert result["classification"] == "SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    assert result["source_dataset_classification"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    assert "not physical-orbit evidence" in result["claim_boundary"]
    pooled = metrics["interpolation"]["per_output"]["p_wall_pooled"]
    assert gates["binding"]["interpolation_rmse_pooled"]["value"] == pooled["rmse"]
    assert gates["binding"]["interpolation_rmse_pooled"]["passed"] == (pooled["rmse"] <= 0.05)
    cells = metrics["interpolation"]["cells"]
    assert gates["binding"]["interpolation_rmse_cells"]["value"] == cells["rmse_floor_corrected"]
    assert gates["binding"]["coverage_90"]["value"] == metrics["interpolation"]["gated_coverage"]["coverage"]
    assert gates["binding"]["coverage_90"]["passed"] == (0.8 <= metrics["interpolation"]["gated_coverage"]["coverage"] <= 0.97)
    ratio = gates["binding"]["beats_best_baseline_2x"]["value"]
    assert ratio == metrics["interpolation"]["best_baseline_pooled"]["ratio_to_surrogate"]
    assert gates["binding"]["beats_best_baseline_2x"]["passed"] == (ratio >= 2.0)
    reported = gates["reported_not_binding"]
    assert reported["extrapolation_rmse_pooled"]["passed"] == (metrics["extrapolation"]["per_output"]["p_wall_pooled"]["rmse"] <= 0.1)


def test_recorded_assessment_recomputes_from_the_dataset_and_the_predictor() -> None:
    _recorded()
    value = protocol()
    rows = experiment.load_rows(value)
    by_id = {row.case_id: row for row in rows}
    partition = _artifact("partitions.json")
    assessment = _artifact("assessment.json")
    metrics = _artifact("metrics.json")
    predictor = Predictor(_artifact("predictor.json"))
    for scope, role in (("interpolation", "assessment"), ("extrapolation", "extrapolation")):
        designs = assessment[scope]["designs"]
        assert [item["case_id"] for item in designs] == partition["roles"][role]
        physical = [list(by_id[item["case_id"]].inputs) for item in designs]
        predictions = predictor.predict(physical)
        errors = {name: [] for name in ALL_OUTPUTS}
        covered = {name: 0 for name in ALL_OUTPUTS}
        for item, prediction in zip(designs, predictions, strict=True):
            row = by_id[item["case_id"]]
            for name in ALL_OUTPUTS:
                recorded = item["outputs"][name]
                successes, trials = row.counts[name]
                assert recorded["truth"] == successes / trials
                assert recorded["predicted"] == pytest.approx(prediction["outputs"][name]["probability"], abs=1e-12)
                assert recorded["observation_interval"] == pytest.approx(prediction["outputs"][name]["observation_interval"], abs=1e-12)
                errors[name].append(recorded["predicted"] - recorded["truth"])
                covered[name] += int(recorded["observation_interval"][0] <= recorded["truth"] <= recorded["observation_interval"][1])
        for name in ALL_OUTPUTS:
            recorded = metrics[scope]["per_output"][name]
            assert recorded["rmse"] == pytest.approx(float(np.sqrt(np.mean(np.square(errors[name])))), abs=1e-12)
            assert recorded["coverage_observation_interval"] == pytest.approx(covered[name] / len(designs))
            floor = d.binomial_floor([by_id[item["case_id"]].counts[name] for item in designs])
            assert recorded["binomial_floor"] == pytest.approx(floor)
            assert recorded["rmse_floor_corrected"] == pytest.approx(math.sqrt(max(recorded["rmse"] ** 2 - floor**2, 0.0)))
        cell_errors = [e for name in m.CELL_OUTPUTS for e in errors[name]]
        assert metrics[scope]["cells"]["rmse"] == pytest.approx(float(np.sqrt(np.mean(np.square(cell_errors)))), abs=1e-12)
        gated = sum(covered[name] for name in GATED_OUTPUTS)
        assert metrics[scope]["gated_coverage"]["covered"] == gated
        assert metrics[scope]["gated_coverage"]["coverage"] == pytest.approx(gated / (len(GATED_OUTPUTS) * len(designs)))


def test_recorded_predictor_contract_is_labelled_and_bound() -> None:
    _recorded()
    contract = _artifact("predictor.json")
    value = protocol()
    assert contract["classification"] == "SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    assert contract["source_dataset_classification"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    assert contract["dataset_binding"]["dataset_file_sha256"] == value["dataset"]["dataset_file_sha256"]
    assert contract["dataset_binding"]["screening_result_commit"] == value["dataset"]["screening_result_commit"]
    assert contract["inputs"]["names"] == value["inputs"]["names"]
    assert contract["evidentiary"] is True and contract["plan_kind"] == "evidentiary"
    selection = _artifact("selection.json")
    assert contract["selected_candidate"] == selection["selected"] in m.CANDIDATE_ORDER
    assert selection["labels_used"] == ["fit", "method-selection"]
    calibration = _artifact("calibration.json")
    assert contract["calibration"]["variance_scale"] == calibration["variance_scale"]
    assert calibration["fit_sample_count"] == 50
    determinism = _artifact("determinism.json")
    assert determinism["passed"] is True
    predictor = Predictor(contract)
    fit_ids = set(contract["interpolation_scope"]["fit_role_case_ids"])
    assert fit_ids == set(_artifact("partitions.json")["roles"]["fit"])
    rows = experiment.load_rows(value)
    flags = predictor.scope_flags([list(row.inputs) for row in rows])
    partition = _artifact("partitions.json")
    extrapolation = set(partition["roles"]["extrapolation"])
    for row, flag in zip(rows, flags, strict=True):
        assert flag["within_interpolation_length"] == (row.case_id not in extrapolation)
