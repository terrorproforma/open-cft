"""Lifecycle-aware checks of the recorded bundle (or of its absence before execution)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.mdo_l0_campaign_v2 import experiment, model as m, optimizers as opt
from experiments.mdo_l0_campaign_v2.experiment import REPOSITORY, RESULTS_ROOT, protocol

MODERN = experiment.MODERN
SPEC_INDEX = MODERN / "spec" / "optimization" / "mdo-l0-campaign-v2.json"
MANIFEST = RESULTS_ROOT / "manifest.json"
CLASSIFICATION = "l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance"


def _recorded() -> dict:
    if not MANIFEST.is_file():
        pytest.skip("campaign not executed yet")
    return strict_json_file(MANIFEST)


def _accepted() -> dict:
    manifest = _recorded()
    if manifest["state"] != "accepted_result":
        pytest.skip(f"terminal state {manifest['state']}")
    return manifest


def test_spec_index_points_at_this_campaign_and_keeps_v1_4_results_null() -> None:
    index = strict_json_file(SPEC_INDEX)
    assert index["document_type"] == "cft-revival-optimization-campaign-instance"
    assert index["protocol"] == "modern/experiments/mdo_l0_campaign_v2/protocol.json"
    assert index["campaign_spec_id"] == "cft-l0-robust-catalogue96x3d-4objective-f0@1.0"
    assert index["parent_spec"] == "modern/spec/optimization/campaign-v1.json"
    assert index["predecessor"] == "modern/spec/optimization/mdo-l0-campaign-v1.json"
    assert index["classification"] == CLASSIFICATION
    v1 = strict_json_file(MODERN / "spec" / "optimization" / "campaign-v1.json")
    assert v1["benchmark"]["results"] is None
    if MANIFEST.is_file() and index["results"] is not None:
        results = index["results"]
        assert results["bundle"] == "modern/experiments/mdo_l0_campaign_v2/results"
        assert results["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        assert results["terminal_state"] == _recorded()["state"]
    elif not MANIFEST.is_file():
        assert index["results"] is None


def test_before_execution_no_results_exist() -> None:
    if MANIFEST.is_file():
        pytest.skip("campaign executed")
    assert not RESULTS_ROOT.exists()
    assert protocol()["status"] == "preregistered_pending_single_execution"


def test_recorded_bundle_inventory_is_byte_exact_and_lf() -> None:
    manifest = _recorded()
    assert manifest["experiment_id"] == "mdo-l0-campaign-v2"
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
        if entry["path"].endswith(".json"):
            assert b"\r" not in data, entry["path"]
    assert mismatches == []
    terminal = strict_json_file(RESULTS_ROOT / "terminal.json")
    assert terminal["state"] == manifest["state"]


def test_recorded_gates_and_replay_of_the_final_pareto_sets() -> None:
    _accepted()
    gates = strict_json_file(RESULTS_ROOT / "artifacts" / "gates.json")
    assert gates["all_binding_passed"] is True
    assert all(item["passed"] for item in gates["binding"].values())
    assert set(gates["binding"]) == set(protocol()["gates"]["binding"])
    assert gates["binding"]["code_hash_scope_matches_imports"]["imported_not_in_scope"] == []
    assert gates["binding"]["code_hash_scope_matches_imports"]["in_scope_not_imported"] == []
    assert all(item["duplicates"] == 0 for item in gates["binding"]["nsga3_duplicates_eliminated"]["runs"].values())
    pareto = strict_json_file(RESULTS_ROOT / "artifacts" / "pareto-sets.json")
    context = m.build_context()
    value = protocol()
    plan = experiment.evidentiary_plan(value)
    assert set(pareto) == {run_id.split(":", 1)[1] for run_id in plan.run_ids}
    for run_key, item in pareto.items():
        assert item["replay_passed"] is True and item["nondominated_recomputed"] is True
        for design in item["designs"]:
            evaluation = m.evaluate_design(design["catalogue_index"], design["values"], context)
            assert evaluation.status == "success"
            assert evaluation.design_id == design["design_id"]
            assert dict(zip(m.OBJECTIVE_NAMES, evaluation.robust_objectives, strict=True)) == design["robust_objectives"]
    metrics = strict_json_file(RESULTS_ROOT / "artifacts" / "metrics.json")
    runs_dir = RESULTS_ROOT / "artifacts" / "runs"
    for run_key in pareto:
        assert metrics["hypervolume_table"][run_key]["final_hypervolume"] >= 0.0
        strategy, seed = run_key.split(":")
        artifact = strict_json_file(runs_dir / f"{strategy}-{seed}.json")
        assert artifact["summary"]["evaluations"] == plan.evaluations_per_run
        monotone = experiment.hypervolume_monotonicity({run_key: artifact["hypervolume_curve"]})
        assert monotone["passed"] and monotone["largest_relative_decrease"] <= experiment.HYPERVOLUME_ROUNDOFF_TOLERANCE
        if strategy == "qlognehvi":
            info = artifact["optimizer"]
            assert info["acquisition"] == opt.acquisition_label(
                q=info["arguments"]["q"],
                mc_samples=info["arguments"]["mc_samples"],
                candidates_per_design=info["arguments"]["candidates_per_design"],
                refine_maxiter=info["arguments"]["refine_maxiter"],
                refine_num_restarts=info["arguments"]["refine_num_restarts"],
                sequential_candidate_stage=info["arguments"]["sequential_candidate_stage"],
            )
            assert "MixedSingleTaskGP" in info["model"] and "CategoricalKernel" in info["model"]
            assert info["torch_threads"] == value["optimizers"]["qlognehvi"]["torch_threads"]
        if strategy == "nsga3":
            info = artifact["optimizer"]
            assert info["declared_generations"] == plan.nsga3_generations
            assert info["pymoo_n_gen"] == plan.nsga3_generations + 1
            assert info["eliminate_duplicates"] is True
            assert artifact["summary"]["unique_designs"] == plan.evaluations_per_run


def test_recorded_campaign_result_carries_the_claim_boundary_and_closures() -> None:
    _accepted()
    result = strict_json_file(RESULTS_ROOT / "artifacts" / "campaign-result.json")
    assert result["evidentiary"] is True
    assert result["classification"] == CLASSIFICATION
    assert result["closure"] == m.CLOSURE_CL1 and result["sensitivity_closure"] == m.CLOSURE_CL2
    assert "no thruster-performance claim" in result["claim_boundary"]
    assert "test-particle wall-hit probability" in result["closure_identification_disclosure"]
    assert "INTEGRITY" in result["gate_semantics"]
    assert result["total_evaluations"] == protocol()["budget"]["total_evaluations"]
    assert isinstance(result["bo_beats_random"], bool) and "/" in result["bo_beats_random_wins"]
    assert result["robust_front_catalogue_designs"], "the robust front names its catalogue designs"
    for item in result["robust_front_catalogue_designs"]:
        assert 0 <= item["catalogue_index"] < 96 and len(item["cell_wall_hit_probabilities"]) == 4
        assert item["pooled_wilson_95"][0] <= item["pooled_wall_hit_probability"] <= item["pooled_wilson_95"][1]
    sensitivity = strict_json_file(RESULTS_ROOT / "artifacts" / "sensitivity.json")
    widths = {str(item["width_scale"]): item for item in sensitivity["widths"]}
    assert widths["1.0"]["is_campaign_posterior"] is True
    assert widths["1.0"]["jaccard_with_campaign_front"] == 1.0
    assert sensitivity["closure_cl2"]["closure"] == m.CLOSURE_CL2


def test_result_commit_touched_only_results_paths() -> None:
    """v1 audit F9: the commit that recorded the bundle must contain results/ files only."""

    _recorded()
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    relative = "modern/experiments/mdo_l0_campaign_v2/results/manifest.json"
    completed = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", relative], cwd=REPOSITORY, capture_output=True, text=True
    )
    commits = completed.stdout.split()
    if not commits:
        pytest.skip("results not committed yet")
    commit = commits[-1]
    changed = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit], cwd=REPOSITORY, capture_output=True, text=True
    ).stdout.splitlines()
    assert changed
    assert all(path.startswith("modern/experiments/mdo_l0_campaign_v2/results/") for path in changed), changed
