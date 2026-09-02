"""Frozen protocol: consistency with the modules, claim boundary, code contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import semantic_sha256

from experiments.mdo_l0_campaign_v1 import experiment, model
from experiments.mdo_l0_campaign_v1.experiment import (
    PROTOCOL_PATH,
    code_contract_report,
    evidentiary_plan,
    protocol,
    protocol_consistency,
    require_protocol_consistency,
    shakedown_plan,
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


def test_claim_boundary_is_explicit() -> None:
    value = protocol()
    boundary = value["claim_boundary"]
    statement = boundary["statement"]
    assert "no thruster-performance claim" in statement
    assert "DECLARED input uncertainty" in statement
    assert "ZERO reflections" in boundary["why_cusp_probabilities_are_uncertain_inputs"]
    assert "mirror" in boundary["why_cusp_probabilities_are_uncertain_inputs"]
    assert boundary["closure_disclosure"].startswith("CL-1 is a DECLARED closure")
    assert len(boundary["forbidden_readings"]) >= 4
    assert value["classification"].endswith("not_thruster_performance")
    assert value["status"] == "preregistered_pending_single_execution"


def test_design_space_is_operating_point_only_with_recorded_exclusions() -> None:
    value = protocol()
    names = [item["name"] for item in value["design_variables"]]
    assert names == [
        "discharge_voltage_v",
        "anode_current_a",
        "propellant_mass_flow_kg_per_s",
    ]
    excluded = {item["name"] for item in value["excluded_legacy_variables"]}
    assert excluded == {
        "inner_magnet_radius_mm",
        "outer_magnet_radius_mm",
        "inner_shield_radius_mm",
        "outer_shield_radius_mm",
        "outer_enclosure_radius_mm",
    }
    assert "geometry" in value["claim_boundary"]["why_geometry_variables_are_excluded"]


def test_objectives_constraints_and_uncertain_inputs_are_declared_exactly() -> None:
    value = protocol()
    assert [item["name"] for item in value["objectives"]] == list(model.OBJECTIVE_NAMES)
    directions = {item["name"]: item["direction"] for item in value["objectives"]}
    assert directions["anode_input_power_w"] == "minimize"
    assert all(
        direction == "maximize"
        for name, direction in directions.items()
        if name != "anode_input_power_w"
    )
    assert "NOT the legacy total_efficiency" in next(
        item["definition"]
        for item in value["objectives"]
        if item["name"] == "thruster_electrical_to_beam_efficiency"
    )
    uncertain = value["uncertain_inputs"]
    assert [item["name"] for item in uncertain["inputs"]] == list(model.UNCERTAIN_NAMES)
    assert uncertain["sample"]["sha256"] == model.sample_sha256(model.uncertain_sample())
    assert uncertain["sample"]["count"] == 64
    v4 = value["authority"]["wall_loss_v4"]
    assert v4["pooled_wall_hit"] == {"successes": 2962, "trials": 4608, "probability": 0.6428}
    assert v4["reflections"] == 0
    constraints = {item["name"]: item for item in value["constraints"]}
    assert constraints["robust_beam_current_margin_a"]["role"].startswith("optimised")
    assert constraints["nominal_beam_current_margin_a"]["role"].startswith("analysis")


def test_budget_and_shakedown_plans_are_arithmetically_closed() -> None:
    value = protocol()
    plan = evidentiary_plan(value)
    assert plan.kind == "evidentiary"
    assert plan.evaluations_per_run == 96
    assert plan.initial_design == 16
    assert plan.qlognehvi_batch_size * plan.qlognehvi_iterations == 80
    assert plan.nsga3_population_size * plan.nsga3_generations == 96
    assert len(plan.run_ids) == 9
    assert plan.binding_gates is True
    shakedown = shakedown_plan(value)
    assert shakedown.kind == "shakedown"
    assert shakedown.binding_gates is False
    assert shakedown.evaluations_per_run < plan.evaluations_per_run
    assert set(shakedown.seeds).isdisjoint(plan.seeds)
    assert all(seed >= 900_000 for seed in shakedown.seeds)
    assert all(seed < 1000 for seed in plan.seeds)
    with pytest.raises(ValueError):
        experiment.CampaignPlan(
            kind="evidentiary",
            seeds=(1,),
            strategies=("sobol",),
            evaluations_per_run=10,
            initial_design=4,
            qlognehvi_batch_size=4,
            qlognehvi_iterations=1,
            nsga3_population_size=4,
            nsga3_generations=2,
            dense_reference_count=8,
            binding_gates=True,
        )


def test_code_contract_scope_resolves_and_hash_is_lf_fail_closed(tmp_path: Path) -> None:
    value = protocol()
    files = source_files(value)
    names = {path.name for path in files}
    assert {"model.py", "optimizers.py", "experiment.py", "botorch_adapter.py", "campaign-v1.json"} <= names
    assert all(path.suffix in {".py", ".json"} for path in files)
    report = source_hash_report(value)
    assert len(report["source_sha256"]) == 64
    assert report["line_endings"] == "LF"
    assert {entry["path"] for entry in report["files"]} == {
        path.resolve().relative_to(experiment.REPOSITORY.resolve()).as_posix() for path in files
    }
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(b"x = 1\r\n")
    scoped = json.loads(json.dumps(value))
    scoped["code_contract"]["source_hash_scope"] = ["modern/" + "x"]
    original = experiment.source_files
    try:
        experiment.source_files = lambda _value: [crlf]  # type: ignore[assignment]
        with pytest.raises(ValueError, match="carriage return"):
            experiment.source_hash_report(scoped)
    finally:
        experiment.source_files = original  # type: ignore[assignment]
    with pytest.raises(ValueError, match="matched nothing"):
        source_files({"code_contract": {"source_hash_scope": ["modern/does/not/exist/*.py"]}})


def test_package_contract_matches_installed_runtime_when_available() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("pymoo")
    report = code_contract_report(protocol())
    assert report["matches"], report


def test_protocol_semantic_hash_is_stable() -> None:
    value = protocol()
    assert semantic_sha256(value) == semantic_sha256(protocol())
    assert "spec/optimization/campaign-v1.json" in value["prior_model_disclosure"]["campaign_spec_v1_4"]
