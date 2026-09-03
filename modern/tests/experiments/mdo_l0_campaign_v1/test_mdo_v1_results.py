"""Lifecycle-aware checks of the recorded bundle (or of its absence before execution)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import strict_json_loads
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.mdo_l0_campaign_v1 import experiment, model as m
from experiments.mdo_l0_campaign_v1.experiment import RESULTS_ROOT, protocol

MODERN = experiment.MODERN
SPEC_INDEX = MODERN / "spec" / "optimization" / "mdo-l0-campaign-v1.json"
MANIFEST = RESULTS_ROOT / "manifest.json"


def _recorded() -> dict:
    if not MANIFEST.is_file():
        pytest.skip("campaign not executed yet")
    return strict_json_file(MANIFEST)


def test_spec_index_points_at_this_campaign_and_keeps_v1_4_results_null() -> None:
    index = strict_json_file(SPEC_INDEX)
    assert index["document_type"] == "cft-revival-optimization-campaign-instance"
    assert index["protocol"] == "modern/experiments/mdo_l0_campaign_v1/protocol.json"
    assert index["campaign_spec_id"] == "cft-l0-robust-3d-4objective-f0@1.0"
    assert index["parent_spec"] == "modern/spec/optimization/campaign-v1.json"
    v1 = strict_json_file(MODERN / "spec" / "optimization" / "campaign-v1.json")
    assert v1["benchmark"]["results"] is None
    if MANIFEST.is_file():
        results = index["results"]
        assert results is not None
        assert results["bundle"] == "modern/experiments/mdo_l0_campaign_v1/results"
        assert results["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        assert results["terminal_state"] == _recorded()["state"]
    else:
        assert index["results"] is None


def test_before_execution_no_results_exist() -> None:
    if MANIFEST.is_file():
        pytest.skip("campaign executed")
    assert not RESULTS_ROOT.exists()
    assert protocol()["status"] == "preregistered_pending_single_execution"


def test_recorded_bundle_inventory_is_byte_exact() -> None:
    manifest = _recorded()
    assert manifest["experiment_id"] == "mdo-l0-campaign-v1"
    assert manifest["state"] in {
        "accepted_result",
        "assessment_rejection",
        "development_rejection",
        "runtime_failure",
        "prebundle_failure",
    }
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


def test_recorded_gates_and_replay_of_the_final_pareto_sets() -> None:
    manifest = _recorded()
    if manifest["state"] != "accepted_result":
        pytest.skip(f"terminal state {manifest['state']}")
    gates = strict_json_file(RESULTS_ROOT / "artifacts" / "gates.json")
    assert gates["all_binding_passed"] is True
    assert all(item["passed"] for item in gates["binding"].values())
    pareto = strict_json_file(RESULTS_ROOT / "artifacts" / "pareto-sets.json")
    sample = m.uncertain_sample()
    nominal = m.nominal_theta()
    value = protocol()
    plan = experiment.evidentiary_plan(value)
    assert set(pareto) == {run_id.split(":", 1)[1] for run_id in plan.run_ids}
    for run_key, item in pareto.items():
        assert item["replay_passed"] is True and item["nondominated_recomputed"] is True
        for design in item["designs"]:
            evaluation = m.evaluate_design(design["values"], sample, nominal=nominal)
            assert evaluation.status == "success"
            assert evaluation.design_id == design["design_id"]
            assert dict(zip(m.OBJECTIVE_NAMES, evaluation.robust_objectives, strict=True)) == design["robust_objectives"]
    metrics = strict_json_file(RESULTS_ROOT / "artifacts" / "metrics.json")
    for run_key, row in metrics["hypervolume_table"].items():
        assert row["final_hypervolume"] >= 0.0
    runs_dir = RESULTS_ROOT / "artifacts" / "runs"
    for run_key in pareto:
        strategy, seed = run_key.split(":")
        artifact = strict_json_file(runs_dir / f"{strategy}-{seed}.json")
        assert artifact["summary"]["evaluations"] == plan.evaluations_per_run
        curve = [entry["hypervolume"] for entry in artifact["hypervolume_curve"]]
        assert all(b >= a for a, b in zip(curve, curve[1:], strict=False))


def test_recorded_campaign_result_carries_the_claim_boundary() -> None:
    manifest = _recorded()
    if manifest["state"] != "accepted_result":
        pytest.skip(f"terminal state {manifest['state']}")
    result = strict_json_file(RESULTS_ROOT / "artifacts" / "campaign-result.json")
    assert result["evidentiary"] is True
    assert result["closure"] == m.CLOSURE_ID
    assert "no thruster-performance claim" in result["claim_boundary"]
    assert result["total_evaluations"] == protocol()["budget"]["total_evaluations"]
    assert isinstance(result["bo_beats_random"], bool)
    sensitivity = strict_json_file(RESULTS_ROOT / "artifacts" / "sensitivity.json")
    priors = {item["cusp_upper"]: item for item in sensitivity["priors"]}
    assert priors[0.45]["identical_to_campaign_front"] is True
