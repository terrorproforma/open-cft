"""Frozen protocol: module consistency, claim boundary, dataset binding, code contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import semantic_sha256

from experiments.wall_loss_geometry_surrogate_v1 import data as d
from experiments.wall_loss_geometry_surrogate_v1 import experiment, models as m
from experiments.wall_loss_geometry_surrogate_v1.experiment import (
    ALL_OUTPUTS,
    GATED_OUTPUTS,
    PROTOCOL_PATH,
    code_contract_report,
    protocol,
    protocol_consistency,
    require_protocol_consistency,
    source_files,
    source_hash_report,
)


def test_protocol_is_strict_lf_json_and_consistent_with_modules() -> None:
    data = PROTOCOL_PATH.read_bytes()
    assert b"\r" not in data
    value = protocol()
    checks = protocol_consistency(value)
    assert checks and all(checks.values()), checks
    assert require_protocol_consistency(value) == checks
    assert semantic_sha256(value) == semantic_sha256(protocol())


def test_claim_boundary_is_explicit_and_carries_the_screening_label() -> None:
    value = protocol()
    assert value["classification"] == "SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    boundary = value["claim_boundary"]
    for key in ("surrogate_of_screening_dataset", "not_physical_orbit_evidence", "not_performance_model", "not_p2_qualified"):
        assert boundary[key] is True
    assert "not physical-orbit evidence" in boundary["statement"]
    assert "not a performance model" in boundary["statement"]
    assert boundary["source_dataset_classification"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    assert len(boundary["forbidden_readings"]) >= 4
    assert "l0_surrogate v9" in boundary["prior_surrogate_failures_disclosure"]
    assert value["status"] == "preregistered_pending_single_execution"


def test_gates_are_the_predeclared_physical_unit_thresholds() -> None:
    gates = protocol()["gates"]
    binding = gates["binding"]
    assert binding["interpolation_rmse_pooled"]["threshold"] == 0.05
    assert binding["interpolation_rmse_cells"]["threshold"] == 0.05
    assert binding["beats_best_baseline_2x"]["ratio_minimum"] == 2.0
    assert binding["coverage_90"]["interval"] == [0.8, 0.97]
    assert binding["coverage_90"]["nominal_probability"] == 0.9
    assert gates["reported_not_binding"]["extrapolation_rmse_pooled"]["threshold"] == 0.1
    assert gates["thresholds_tunable_after_execution"] is False
    assert set(binding) == set(experiment.STRUCTURAL_GATES) | set(experiment.SCIENCE_GATES)


def test_inputs_outputs_and_noise_model_are_declared() -> None:
    value = protocol()
    assert value["inputs"]["names"] == sorted(value["inputs"]["names"]) and len(value["inputs"]["names"]) == 11
    assert tuple(value["outputs"]["gated"]) == GATED_OUTPUTS
    assert tuple(value["outputs"]["reported_only"]) == ("p_reflect_pooled",)
    assert all(value["outputs"]["trials"][name] == (128 if "cell" in name else 512) for name in ALL_OUTPUTS)
    noise = value["outputs"]["known_noise"]
    assert noise["model"].startswith("binomial")
    assert "Haldane" in noise["logit_transform"]
    assert "orbit_mc" in value["outputs"]["no_tautology_statement"]
    assert "stage_count" in value["inputs"]["known_discontinuities"]


def test_candidates_baselines_and_partition_counts_match_the_modules() -> None:
    value = protocol()
    assert tuple(value["candidates"]["order"]) == m.CANDIDATE_ORDER
    assert set(value["candidates"]["definitions"]) == set(m.CANDIDATE_ORDER)
    assert tuple(value["candidates"]["baselines"])[:3] == m.BASELINE_ORDER
    counts = value["partition"]["counts"]
    assert counts["totals"] == {"fit": 50, "method-selection": 10, "calibration": 10, "assessment": 16, "extrapolation": 10}
    assert sum(counts["totals"].values()) == value["dataset"]["design_count"] == 96
    assert value["partition"]["extrapolation_cluster"]["count"] == math.ceil(0.1 * 96) == 10
    assert value["shakedown"]["seed_namespace"] != value["partition"]["seed_namespace"]


def test_dataset_binding_holds_against_the_screening_bundle_and_git() -> None:
    value = protocol()
    report = d.dataset_binding_report(value["dataset"], use_git=True)
    assert report["passed"], report["checks"]
    assert report["dataset_file_sha256"] == value["dataset"]["dataset_file_sha256"]
    assert report["manifest_state"] == "accepted_result"
    assert report["git"]["dataset_blob_at_result_commit"] == value["dataset"]["dataset_git_blob"]
    assert report["git"]["result_commit_is_ancestor_of_head"] is True
    assert value["dataset"]["screening_result_commit"].startswith("ab7c2897")
    assert value["dataset"]["screening_merge_commit"].startswith("22e2156b")


def test_dataset_binding_fails_closed_on_a_wrong_hash() -> None:
    value = json.loads(json.dumps(protocol()))
    value["dataset"]["dataset_file_sha256"] = "0" * 64
    report = d.dataset_binding_report(value["dataset"], use_git=False)
    assert report["passed"] is False
    assert report["checks"]["dataset_file_sha256"] is False
    with pytest.raises(d.DatasetBindingError, match="dataset_file_sha256"):
        d.require_dataset_binding(value["dataset"], use_git=False)


def test_rows_load_with_exact_count_ratios_and_non_degenerate_inputs() -> None:
    value = protocol()
    rows = experiment.load_rows(value)
    assert len(rows) == 96
    assert [row.case_id for row in rows] == sorted(row.case_id for row in rows)
    for row in rows:
        assert row.converged is True
        for name in ALL_OUTPUTS:
            successes, trials = row.counts[name]
            assert row.stored_probabilities[name] == successes / trials
        assert sum(row.counts[f"p_wall_cell{i}"][0] for i in range(1, 5)) == row.counts["p_wall_pooled"][0]
        assert abs(sum(row.counts[f"p_wall_cell{i}"][0] / 128 for i in range(1, 5)) / 4 - row.stored_probabilities["p_wall_pooled"]) < 1e-12
    degeneracy = d.degeneracy_report(rows, value["inputs"]["names"])
    assert degeneracy["passed"] and degeneracy["distinct_input_tuples"] == 96
    assert min(degeneracy["distinct_values_per_input"].values()) >= 8


def test_code_contract_scope_resolves_and_hash_is_lf_fail_closed(tmp_path: Path) -> None:
    value = protocol()
    files = source_files(value)
    names = {path.name for path in files}
    assert {"data.py", "models.py", "predictor.py", "experiment.py", "run.py", "gp.py", "validation.py", "lifecycle.py"} <= names
    report = source_hash_report(value)
    assert len(report["source_sha256"]) == 64 and report["line_endings"] == "LF"
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(b"x = 1\r\n")
    original = experiment.source_files
    try:
        experiment.source_files = lambda _value: [crlf]  # type: ignore[assignment]
        with pytest.raises(ValueError, match="carriage return"):
            experiment.source_hash_report(value)
    finally:
        experiment.source_files = original  # type: ignore[assignment]
    with pytest.raises(ValueError, match="matched nothing"):
        source_files({"code_contract": {"source_hash_scope": ["modern/does/not/exist/*.py"]}})


def test_package_contract_matches_installed_runtime_when_available() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("scipy")
    report = code_contract_report(protocol())
    assert report["matches"], report
