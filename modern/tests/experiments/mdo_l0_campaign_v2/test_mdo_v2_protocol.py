"""Frozen protocol: consistency, claim boundary, import-bound code contract, label policy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import semantic_sha256, strict_json_file
from cft_revival.provenance import blob_exists

from experiments.mdo_l0_campaign_v2 import experiment, model, optimizers as opt
from experiments.mdo_l0_campaign_v2.experiment import (
    EXPERIMENT,
    MODERN,
    PROTOCOL_PATH,
    REPOSITORY,
    code_contract_report,
    evidentiary_plan,
    import_scope_report,
    label_checks,
    protocol,
    protocol_consistency,
    require_protocol_consistency,
    shakedown_plan,
    source_files,
    source_hash_report,
)

CLASSIFICATION = "l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance"
RESULTS = EXPERIMENT / "results"
README_PATH = EXPERIMENT / "README.md"
# the commit the immutable execution lock names (preregistration = execution commit of v2)
EXECUTION_COMMIT = "99914dc2fdbe88d18ab11ca86acad634129b4e08"
# Repository files the campaign imports today that the sealed hash scope does not name. Every
# entry was added to a shared package AFTER the execution (no blob at EXECUTION_COMMIT) and is
# disclosed in README.md under "Post-hoc audit notes"; the sealed protocol, authorities and bundle
# are untouched. Growth that is not both post hoc and disclosed fails the import-trace test.
POST_HOC_IMPORTED_SHARED_MODULES = {
    # fail-closed manifest recovery for the geometry-screening-v2 EMFILE, re-exported by experiment_runtime/__init__.py
    "modern/src/cft_revival/experiment_runtime/recovery.py": "bb756418",
}


def test_protocol_is_strict_lf_json_and_consistent_with_modules() -> None:
    data = PROTOCOL_PATH.read_bytes()
    assert b"\r" not in data
    value = protocol()
    checks = protocol_consistency(value)
    assert checks and all(checks.values()), {k: v for k, v in checks.items() if not v}
    assert require_protocol_consistency(value) == checks
    assert semantic_sha256(value) == semantic_sha256(protocol())


def test_claim_boundary_and_closure_identification_are_explicit() -> None:
    value = protocol()
    assert value["classification"] == CLASSIFICATION
    assert value["status"] == "preregistered_pending_single_execution"
    boundary = value["claim_boundary"]
    assert "no thruster-performance claim" in boundary["statement"]
    assert "DIRECTLY from the accepted screening dataset" in boundary["statement"]
    assert "NOT the Kornfeld per-cusp probability" in boundary["closure_identification"]
    assert "UNDER THIS DECLARED CLOSURE ONLY" in boundary["closure_identification"]
    assert "b400d924" in boundary["why_no_surrogate"] and "a2b503be" in boundary["why_no_surrogate"]
    assert "No surrogate may be used" in boundary["why_no_surrogate"]
    assert len(boundary["forbidden_readings"]) >= 6
    closures = value["closures"]
    assert closures["CL-1"]["id"] == model.CLOSURE_CL1 and closures["CL-1"]["role"] == "campaign"
    assert "test-particle wall-hit probability" in closures["CL-1"]["identification_disclosure"]
    assert closures["CL-2"]["id"] == model.CLOSURE_CL2 and closures["CL-2"]["role"] == "sensitivity"
    assert "1 - p_pooled" in closures["CL-2"]["statement"]
    assert "no admissible root" in value["authority"]["four_cell_closure_analysis"]["finding"]
    assert value["authority"]["four_cell_closure_analysis"]["commit"] == "266d8a99"
    assert value["catalogue_binding"]["screening_result_commit"].startswith("ab7c2897")


def test_design_space_is_catalogue_times_operating_point() -> None:
    value = protocol()
    space = value["design_space"]
    assert space["catalogue"]["kind"] == "categorical" and space["catalogue"]["size"] == 96
    assert [item["name"] for item in space["operating_point"]] == list(model.CONTINUOUS_NAMES)
    bounds = {item["name"]: (item["lower"], item["upper"]) for item in space["operating_point"]}
    assert bounds["discharge_voltage_v"] == (150.0, 500.0)
    assert bounds["anode_current_a"] == (0.1, 2.5)
    assert bounds["propellant_mass_flow_kg_per_s"] == (2e-7, 2e-6)
    uncertain = value["uncertain_inputs"]
    assert all(item["distribution"] == "jeffreys-beta-posterior" for item in uncertain["per_design_inputs"])
    assert uncertain["sensitivity_widths"]["width_scales"] == [0.25, 1.0, 4.0, "point"]
    assert "no rounded probability" in uncertain["rounding_policy"]
    assert value["reference_point"] == {
        "axial_thrust_n": 0.0,
        "specific_impulse_s": 0.0,
        "thruster_electrical_to_beam_efficiency": 0.0,
        "anode_input_power_w": 1300.0,
        "normalization": value["reference_point"]["normalization"],
    }
    scales = {item["name"]: item["comparison_scale"] for item in value["objectives"]}
    assert scales == {"axial_thrust_n": 0.06, "specific_impulse_s": 3000.0, "thruster_electrical_to_beam_efficiency": 1.0, "anode_input_power_w": 1300.0}


def test_budget_and_shakedown_plans_are_arithmetically_closed() -> None:
    value = protocol()
    plan = evidentiary_plan(value)
    assert plan.evaluations_per_run == 160 and plan.initial_design == 32
    assert plan.qlognehvi_batch_size * plan.qlognehvi_iterations == 128
    assert plan.nsga3_population_size * plan.nsga3_generations == 160
    assert len(plan.run_ids) == 9 and plan.binding_gates is True
    assert value["budget"]["total_evaluations"] == 1440
    shakedown = shakedown_plan(value)
    assert shakedown.kind == "shakedown" and shakedown.binding_gates is False
    assert shakedown.evaluations_per_run < plan.evaluations_per_run
    assert set(shakedown.seeds).isdisjoint(plan.seeds)
    assert shakedown.dense_reference_points_per_design < plan.dense_reference_points_per_design
    with pytest.raises(ValueError):
        experiment.CampaignPlan("evidentiary", (1,), ("x",), 10, 4, 4, 1, 4, 2, 8, True)


def test_gate_semantics_declare_integrity_only_and_list_the_v1_disclosures() -> None:
    value = protocol()
    gates = value["gates"]
    assert "INTEGRITY" in gates["semantics"] and "NOT" in gates["semantics"]
    assert set(gates["binding"]) == {
        "replay_bit_exact",
        "l0_domain",
        "hypervolume_monotone",
        "budget_exact",
        "shared_initial_design",
        "sample_hash",
        "pareto_replay",
        "code_contract",
        "catalogue_binding",
        "code_hash_scope_matches_imports",
        "nsga3_duplicates_eliminated",
        "labels_consistent",
    }
    assert set(value["v1_audit_disclosures_closed"]) == {"F9", "F10", "F22", "F26", "F27", "F28"}
    assert value["optimizers"]["nsga3"]["eliminate_duplicates"] is True
    assert value["execution"]["result_commit_policy"].startswith("the result commit contains files under modern/experiments/mdo_l0_campaign_v2/results/ only")
    assert "count" in gates["reported_not_binding"]["bo_beats_random"]


def test_code_contract_scope_is_explicit_lf_and_fail_closed(tmp_path: Path) -> None:
    value = protocol()
    files = source_files(value)
    names = {path.resolve().relative_to(REPOSITORY.resolve()).as_posix() for path in files}
    assert {
        "modern/experiments/mdo_l0_campaign_v2/catalogue.py",
        "modern/experiments/mdo_l0_campaign_v2/model.py",
        "modern/experiments/mdo_l0_campaign_v2/optimizers.py",
        "modern/experiments/mdo_l0_campaign_v2/experiment.py",
        "modern/experiments/mdo_l0_campaign_v2/run.py",
        "modern/src/cft_revival/experiment_runtime/canonical.py",
        "modern/src/cft_revival/models.py",
        "modern/src/cft_revival/kernels.py",
        "modern/src/cft_revival/optimization/botorch_adapter.py",
        "modern/src/cft_revival/physics/reference.py",
    } <= names
    # v1 audit F10: never-imported packages must not be bound
    assert not any("/active_learning/" in name or "/surrogates/" in name for name in names)
    assert all(path.suffix == ".py" for path in files)
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
    with pytest.raises(ValueError, match="explicit path"):
        source_files({"code_contract": {"source_hash_scope": ["modern/src/cft_revival/physics/*.py"]}})
    with pytest.raises(ValueError, match="not a file"):
        source_files({"code_contract": {"source_hash_scope": ["modern/does/not/exist.py"]}})


def test_import_trace_in_a_fresh_interpreter_equals_the_hash_scope() -> None:
    """v1 audit F10: the hash scope must be exactly the set of repository files the campaign imports."""

    script = (
        "import json, sys\n"
        "import experiments.mdo_l0_campaign_v2.run\n"
        "from experiments.mdo_l0_campaign_v2 import experiment as e, model as m, optimizers as opt\n"
        "ctx = m.build_context()\n"
        "m.evaluate_design(0, (300.0, 1.0, 1e-6), ctx)\n"
        "e.dense_grid(2, 1)\n"
        "opt.shared_initial_points(1, 4)\n"
        "print(json.dumps(e.import_scope_report(e.protocol())))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(MODERN / "src"), str(MODERN)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([sys.executable, "-c", script], cwd=MODERN, env=env, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["in_scope_not_imported"] == []
    assert "modern/experiments/mdo_l0_campaign_v2/run.py" in report["imported"]
    assert "modern/src/cft_revival/experiment_runtime/canonical.py" in report["imported"]
    # The scope is sealed (protocol.json semantic hash in authorities.json) and was exact when the
    # campaign executed: the recorded binding gate says so. Growth of the live import graph since
    # then is allowed only when it is post hoc (no blob at the execution commit) AND disclosed.
    growth = report["imported_not_in_scope"]
    assert set(growth) == set(POST_HOC_IMPORTED_SHARED_MODULES), growth
    assert report["matches"] is (growth == [])
    readme = README_PATH.read_text(encoding="utf-8")
    for path, commit in POST_HOC_IMPORTED_SHARED_MODULES.items():
        assert not blob_exists(REPOSITORY, EXECUTION_COMMIT, path), f"{path} existed at the execution commit: a sealing omission, not post-hoc growth"
        assert (REPOSITORY / path).is_file(), path
        assert path.rsplit("/", 1)[1] in readme and commit in readme, f"{path} is not disclosed in README.md"
    lock = strict_json_file(RESULTS / "execution-lock.json")
    assert lock["commit"] == EXECUTION_COMMIT and lock["immutable"] is True
    recorded = strict_json_file(RESULTS / "artifacts" / "gates.json")["binding"]["code_hash_scope_matches_imports"]
    assert recorded["passed"] is True and recorded["imported_not_in_scope"] == [] and recorded["in_scope_not_imported"] == []
    assert recorded["imported_count"] == len(protocol()["code_contract"]["source_hash_scope"]) == len(report["declared"])


def test_in_process_import_scope_report_shape() -> None:
    report = import_scope_report(protocol())
    assert set(report) == {"declared", "imported", "imported_not_in_scope", "in_scope_not_imported", "matches"}
    assert set(report["declared"]) >= set(report["imported"]) - set(report["imported_not_in_scope"])


def test_package_contract_matches_installed_runtime_when_available() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("pymoo")
    report = code_contract_report(protocol())
    assert report["matches"], report


def test_hypervolume_monotonicity_gate_tolerates_roundoff_only() -> None:
    curve = [{"evaluations": 1, "hypervolume": 1e-6}, {"evaluations": 2, "hypervolume": 1e-6 * (1 - 2e-16)}, {"evaluations": 3, "hypervolume": 2e-6}]
    report = experiment.hypervolume_monotonicity({"a": curve})
    assert report["passed"] and 0.0 < report["largest_relative_decrease"] < 1e-15
    bad = [{"evaluations": 1, "hypervolume": 1e-6}, {"evaluations": 2, "hypervolume": 0.9e-6}]
    report = experiment.hypervolume_monotonicity({"b": bad})
    assert not report["passed"] and report["violations"][0]["run"] == "b"
    assert "1e-12" in protocol()["gates"]["binding"]["hypervolume_monotone"]


def test_labels_are_generated_from_arguments_and_validated(tmp_path: Path) -> None:
    value = protocol()
    plan = evidentiary_plan(value)
    arguments = {
        "q": plan.qlognehvi_batch_size,
        "mc_samples": 32,
        "candidates_per_design": 8,
        "refine_maxiter": 30,
        "refine_num_restarts": 1,
        "sequential_candidate_stage": True,
    }
    label = opt.acquisition_label(**arguments)
    assert "q=8 sequential greedy" in label and "96 catalogue designs x 8" in label and "maxiter 30" in label
    good = {
        "qlognehvi:101": {
            "arguments": arguments,
            "acquisition": label,
            "model": "ModelListGP of 5 MixedSingleTaskGP (cat_dims=[0] catalogue index; kernels CategoricalKernel, MaternKernel, ScaleKernel; outcome transform Standardize; GaussianLikelihood with noise floor 1e-06)",
            "iterations": plan.qlognehvi_iterations,
        },
        "nsga3:101": {
            "declared_generations": plan.nsga3_generations,
            "pymoo_n_gen": plan.nsga3_generations + 1,
            "pymoo_reported_evaluations": plan.evaluations_per_run,
            "eliminate_duplicates": True,
        },
        "lhs:101": {"stages": [plan.initial_design, plan.evaluations_per_run - plan.initial_design], "design": f"stage 1 = the shared {plan.initial_design}-point"},
    }
    assert label_checks(good, plan)["passed"]
    # v1's hard-coded label ("sequential greedy batch" while joint q ran) would be refused
    bad = json.loads(json.dumps(good))
    bad["qlognehvi:101"]["acquisition"] = "qLogNoisyExpectedHypervolumeImprovement, prune_baseline, sequential greedy batch"
    assert not label_checks(bad, plan)["passed"]
    bad = json.loads(json.dumps(good))
    bad["nsga3:101"]["declared_generations"] = plan.nsga3_generations + 1  # pymoo's post-incremented counter as the declared count
    assert not label_checks(bad, plan)["passed"]
    bad = json.loads(json.dumps(good))
    bad["nsga3:101"]["eliminate_duplicates"] = False
    assert not label_checks(bad, plan)["passed"]
