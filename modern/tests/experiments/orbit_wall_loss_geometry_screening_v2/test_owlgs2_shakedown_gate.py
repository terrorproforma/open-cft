"""Shakedown design and the prepare/execute shakedown gate (lifecycle-aware; ported from v1)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import canonical_bytes, semantic_sha256
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.orbit_wall_loss_geometry_screening_v2 import cells as C
from experiments.orbit_wall_loss_geometry_screening_v2 import designs as D
from experiments.orbit_wall_loss_geometry_screening_v2 import experiment, run
from experiments.orbit_wall_loss_geometry_screening_v2.experiment import (
    SHAKEDOWN_PATH,
    bind_designs,
    build_design_authorities,
    design_sha256,
    evidentiary_plan,
    protocol,
    schema,
    shakedown_disjointness,
    shakedown_plan,
    verify_shakedown_record,
)

FROZEN = (experiment.AUTHORITIES_PATH, experiment.DESIGN_AUTHORITIES_PATH)
EXECUTED = (experiment.RESULTS_ROOT / "manifest.json").is_file()
CODE_BINDING_CHECKS = {
    "orbit_mc_source_sha256_current",
    "orbit_mc_schema_versions_current",
    "field_pipeline_source_sha256_current",
    "experiment_code_sha256_current",
}


def _frozen_state() -> tuple[bool, ...]:
    return tuple(path.exists() for path in FROZEN)


@pytest.fixture(scope="module")
def value() -> dict:
    return protocol()


@pytest.fixture(scope="module")
def bound(value: dict) -> dict:
    sweep = D.load_sweep_binding(value["field_source"])
    catalogue = C.load_bound_catalogue(value["cusp_cell_catalogue"])
    return bind_designs(value, sweep, catalogue, shakedown_plan(value).design_keys)


def test_shakedown_plan_is_small_covers_p2_and_exercises_both_rule_branches(value: dict, bound: dict) -> None:
    plan = shakedown_plan(value)
    assert plan.kind == "shakedown" and plan.binding_gates is False
    assert len(plan.design_keys) == 4 and D.P2_DESIGN_ID in plan.design_keys
    assert plan.stage1_points_per_stratum == 2 and plan.stage2_points_per_stratum == 4 and plan.block_count == 2
    assert plan.wilson_width_threshold == 0.25
    # at n = 16 a saturated cell is not topped up, a mixed one is
    rule = plan.allocation_rule(value)
    assert C.allocation_decision({"c": {"wall_hit": 16, "trials": 16}}, rule)["cells"]["c"]["topped_up"] is False
    assert C.allocation_decision({"c": {"wall_hit": 0, "trials": 16}}, rule)["cells"]["c"]["topped_up"] is False
    assert C.allocation_decision({"c": {"wall_hit": 8, "trials": 16}}, rule)["cells"]["c"]["topped_up"] is True
    for key in plan.design_keys:
        for cell in bound[key].cells:
            launches = experiment.block_launches(value, plan, bound[key], cell, 0)
            assert len(launches) == 16
            assert all(item.launch_id.startswith(f"owlgs-v2-shakedown:{key}:{cell.cell_id}:stage1:N:") for item in launches)
            assert len(experiment.batch_records(plan, launches)) == 2
    assert plan.case_sizes(value) == {"block": 16, "control_of_stage1_cell": 2, "control_of_topped_up_cell": 4}


def test_shakedown_is_disjoint_from_the_evidentiary_design(value: dict, bound: dict) -> None:
    report = shakedown_disjointness(value, bound)
    assert report["proven"] is True
    assert report["shakedown_launch_count"] == 14 * 32
    assert report["shakedown_unique_launch_ids"] == report["shakedown_unique_seed_ids"] == 14 * 32
    assert set(report["reports"]["against_evidentiary_same_designs"]["overlap_counts"].values()) == {0}
    same_designs = experiment.CampaignPlan(**{**evidentiary_plan(value).__dict__, "design_keys": shakedown_plan(value).design_keys})
    assert design_sha256(value, shakedown_plan(value), bound) != design_sha256(value, same_designs, bound)


def test_shakedown_and_evidentiary_authorities_carry_distinct_policies(value: dict, bound: dict) -> None:
    shakedown = build_design_authorities(value, shakedown_plan(value), bound)
    evidentiary = build_design_authorities(value, experiment.CampaignPlan(**{**evidentiary_plan(value).__dict__, "design_keys": shakedown_plan(value).design_keys}), bound)
    assert shakedown["plan_kind"] == "shakedown" and evidentiary["plan_kind"] == "evidentiary"
    assert shakedown["cell_count"] == evidentiary["cell_count"] == 14
    assert shakedown["case_sizes"]["block"] == 16 and evidentiary["case_sizes"]["block"] == 128
    for left, right in zip(shakedown["stage1_cases"], evidentiary["stage1_cases"], strict=True):
        assert left["case_key"] == right["case_key"]
        assert left["campaign_id"] != right["campaign_id"]
        assert left["policy_identity_sha256"] != right["policy_identity_sha256"]
        assert left["orbit_launches_sha256"] != right["orbit_launches_sha256"]
        assert left["field_identity_sha256"] == right["field_identity_sha256"]
        assert left["config_identity_sha256"] == right["config_identity_sha256"]
        assert left["launch_count"] == 16 and right["launch_count"] == 128
    for row in evidentiary["designs"]:
        assert all(isinstance(seed, str) and seed.isdecimal() for seed in row["stratum_seeds"].values())
        assert row["stratum_seed_encoding"] == "unsigned-64 decimal string"
    canonical_bytes(evidentiary)  # every value is canonical JSON (uint64 seeds as strings)


# --------------------------------------------------------------------------
# the committed shakedown record
# --------------------------------------------------------------------------


def _record() -> dict:
    if not SHAKEDOWN_PATH.is_file():
        pytest.skip("shakedown.json has not been produced yet")
    return strict_json_file(SHAKEDOWN_PATH)


def _live_code_drifted() -> bool:
    if not experiment.AUTHORITIES_PATH.is_file():
        return False
    authorities = strict_json_file(experiment.AUTHORITIES_PATH)
    import cft_revival.orbit_mc as orbit_mc_package

    return (
        orbit_mc_package.__version__ != authorities["orbit_mc_package_version"]
        or experiment.orbit_mc_source_sha256() != authorities["orbit_mc_source_sha256"]
        or D.field_pipeline_source_sha256() != authorities["field_pipeline_source_sha256"]
        or experiment.experiment_code_sha256() != authorities["experiment_code_sha256"]
    )


def _refused_checks(error: BaseException) -> set[str]:
    return set(str(error).rsplit("refused: ", 1)[1].split(", "))


def test_committed_shakedown_record_opens_the_gate_for_current_protocol_and_code(value: dict, bound: dict) -> None:
    record = _record()
    if EXECUTED and _live_code_drifted():
        with pytest.raises(ValueError, match="shakedown gate refused") as info:
            verify_shakedown_record(value, record, bound)
        assert _refused_checks(info.value) <= CODE_BINDING_CHECKS
    else:
        checks = verify_shakedown_record(value, record, bound)
        assert all(checks.values()), checks
    assert record["classification"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    assert record["evidentiary"] is False and record["outcomes_enter_estimand"] is False
    assert record["passed"] is True
    plan = shakedown_plan(value)
    stage1_keys = {experiment.case_key(key, cell.cell_id, "stage1") for key in plan.design_keys for cell in bound[key].cells}
    control_keys = {experiment.case_key(key, cell.cell_id, "control") for key in plan.design_keys for cell in bound[key].cells}
    assert stage1_keys <= set(record["cases"]) and control_keys <= set(record["cases"])
    assert record["case_count"] == len(record["cases"]) >= 28
    assert record["allocation_summary"]["topped_up_cells"] > 0 and record["allocation_summary"]["saturated_cells"] > 0
    assert record["allocation_summary"]["replay_all_passed"] is True
    assert record["validators"]["failed"] == 0 and record["validators"]["passed"] > 0
    assert record["design_exclusions"] == []
    assert len(record["solver_access_records"]) == 1
    labels = [item["operation"] for item in record["label_access_records"]]
    assert set(labels) >= {f"orbit-{key}" for key in stage1_keys | control_keys}
    assert record["orbit_mc_package_version"] == "1.7.0"
    assert record["orbit_mc_source_hash_line_endings"] == "LF"
    assert record["catalogue_file_sha256"] == value["cusp_cell_catalogue"]["catalogue_file_sha256"]
    assert record["timing_projection"]["within_budget_expected"] is True
    assert record["timing_projection"]["expected"]["stage1_orbits"] == 377 * 128
    assert set(record["experiment_code_files"]) == set(experiment.EXPERIMENT_CODE_FILES)
    assert record["dataset_summary"]["design_count"] == 4 and record["dataset_summary"]["cell_count"] == 14
    for key, item in record["cases"].items():
        assert item["export_stage_ran"] is True and item["handoff_consumed"] is True
        assert item["validators"]["failed"] == 0
        assert item["launch_count"] in (16, 2, 4)
        counts = item["diagnostics"]["termination_counts"]
        assert counts["field_failure"] == 0 and counts["step_limit"] == 0 and counts["initial_state_invalid"] == 0
        assert item["diagnostics"]["maximum_relative_energy_error"] == 0.0
        assert item["diagnostics"]["final_velocity_equals_event_velocity_count"] == item["launch_count"]
    gates = record["informational_gates"]
    assert gates["binding"] is False
    assert "NON-EVIDENTIARY" in record["disclosure"]
    for design_gate in record["design_gates"].values():
        assert design_gate["structural_passed"] is True


def test_shakedown_record_hash_bindings_are_reproducible(value: dict, bound: dict) -> None:
    record = _record()
    data = SHAKEDOWN_PATH.read_bytes()
    assert b"\r" not in data
    assert json.loads(data.decode("utf-8")) == record
    if not (EXECUTED and _live_code_drifted()):
        assert record["protocol_semantic_sha256"] == semantic_sha256(value)
        assert record["shakedown_design_sha256"] == design_sha256(value, shakedown_plan(value), bound)
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(SHAKEDOWN_PATH.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# prepare refusals (must never write frozen outputs)
# --------------------------------------------------------------------------


@pytest.fixture
def frozen_contract_for_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    if not (EXECUTED and _live_code_drifted()):
        return
    authorities = strict_json_file(experiment.AUTHORITIES_PATH)
    frozen = {
        "orbit_mc": {
            "expected": dict(authorities["orbit_mc_schema_versions"]),
            "observed": dict(authorities["orbit_mc_schema_versions"]),
            "matches": True,
            "source_sha256": authorities["orbit_mc_source_sha256"],
            "code_identity_sha256": authorities["orbit_mc_code_identity_sha256"],
            "source_files": [],
        },
        "field_pipeline_source_sha256": authorities["field_pipeline_source_sha256"],
        "field_pipeline_source_files": [],
        "experiment_code_sha256": authorities["experiment_code_sha256"],
        "experiment_code_files": [],
        "catalogue_file_sha256": authorities["catalogue_file_sha256"],
        "catalogue_manifest_file_sha256": authorities["catalogue_manifest_file_sha256"],
        "v1_reused_modules": [],
    }
    monkeypatch.setattr(run, "source_binding_report", lambda _value: frozen)


def test_prepare_and_shakedown_gate_stay_closed_after_execution() -> None:
    if not EXECUTED:
        pytest.skip("the experiment has not executed yet")
    assert experiment.AUTHORITIES_PATH.is_file()
    before = _frozen_state()
    if _live_code_drifted():
        with pytest.raises((ValueError, RuntimeError)):
            run.prepare()
    else:
        with pytest.raises(RuntimeError):
            run.shakedown()
    assert _frozen_state() == before


def test_prepare_refuses_without_shakedown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frozen_contract_for_prepare: None) -> None:
    before = _frozen_state()
    monkeypatch.setattr(run, "SHAKEDOWN_PATH", tmp_path / "missing-shakedown.json")
    with pytest.raises(RuntimeError, match="shakedown.json is missing"):
        run.prepare()
    assert _frozen_state() == before


def test_prepare_refuses_on_code_hash_mismatch(monkeypatch: pytest.MonkeyPatch, frozen_contract_for_prepare: None) -> None:
    _record()
    before = _frozen_state()
    monkeypatch.setattr(experiment, "orbit_mc_source_sha256", lambda: "f" * 64)
    with pytest.raises(RuntimeError, match="orbit_mc_source_sha256_current"):
        run.prepare()
    monkeypatch.undo()
    monkeypatch.setattr(experiment, "field_pipeline_source_sha256", lambda: "e" * 64)
    with pytest.raises(RuntimeError, match="field_pipeline_source_sha256_current"):
        run.prepare()
    monkeypatch.undo()
    monkeypatch.setattr(experiment, "experiment_code_sha256", lambda: "d" * 64)
    with pytest.raises(RuntimeError, match="experiment_code_sha256_current"):
        run.prepare()
    assert _frozen_state() == before


def test_prepare_refuses_on_stale_protocol_hash(monkeypatch: pytest.MonkeyPatch, frozen_contract_for_prepare: None) -> None:
    _record()
    before = _frozen_state()
    stale = copy.deepcopy(protocol())
    stale["gates"]["maximum_relative_energy_error"] = 1e-9
    monkeypatch.setattr(run, "protocol", lambda: stale)
    with pytest.raises(RuntimeError, match="protocol_semantic_sha256_current"):
        run.prepare()
    assert _frozen_state() == before


def test_prepare_refuses_on_failed_or_tampered_shakedown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frozen_contract_for_prepare: None) -> None:
    record = _record()
    before = _frozen_state()
    path = tmp_path / "shakedown.json"
    monkeypatch.setattr(run, "SHAKEDOWN_PATH", path)
    failed = copy.deepcopy(record)
    failed["passed"] = False
    path.write_bytes(canonical_bytes(failed))
    with pytest.raises(RuntimeError, match="passed"):
        run.prepare()
    tampered = copy.deepcopy(record)
    tampered["disjointness"]["reports"]["against_evidentiary_same_designs"]["overlap_counts"]["seed_id"] = 1
    path.write_bytes(canonical_bytes(tampered))
    with pytest.raises(RuntimeError, match="disjointness_proven"):
        run.prepare()
    missing_export = copy.deepcopy(record)
    first = next(iter(missing_export["cases"]))
    missing_export["cases"][first]["handoff_consumed"] = False
    path.write_bytes(canonical_bytes(missing_export))
    with pytest.raises(RuntimeError, match="all_validators_passed"):
        run.prepare()
    one_branch = copy.deepcopy(record)
    one_branch["allocation_summary"]["saturated_cells"] = 0
    path.write_bytes(canonical_bytes(one_branch))
    with pytest.raises(RuntimeError, match="both_allocation_branches_observed"):
        run.prepare()
    wrong_catalogue = copy.deepcopy(record)
    wrong_catalogue["catalogue_file_sha256"] = "0" * 64
    path.write_bytes(canonical_bytes(wrong_catalogue))
    with pytest.raises(RuntimeError, match="catalogue_file_sha256_declared"):
        run.prepare()
    over_budget = copy.deepcopy(record)
    over_budget["timing_projection"]["within_budget_expected"] = False
    path.write_bytes(canonical_bytes(over_budget))
    with pytest.raises(RuntimeError, match="timing_within_budget"):
        run.prepare()
    excluded = copy.deepcopy(record)
    excluded["design_exclusions"] = [{"design_key": "x", "reason": "y"}]
    path.write_bytes(canonical_bytes(excluded))
    with pytest.raises(RuntimeError, match="zero_exclusions"):
        run.prepare()
    assert _frozen_state() == before


def test_verify_shakedown_record_rejects_wrong_schema_and_design(value: dict, bound: dict) -> None:
    record = _record()
    wrong_schema = copy.deepcopy(record)
    wrong_schema["schema_version"] = schema("shakedown").replace("1.0.0", "0.9.0")
    with pytest.raises(ValueError, match="schema_version"):
        verify_shakedown_record(value, wrong_schema, bound)
    wrong_design = copy.deepcopy(record)
    wrong_design["shakedown_design_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shakedown_design_sha256_current"):
        verify_shakedown_record(value, wrong_design, bound)
    evidentiary_claim = copy.deepcopy(record)
    evidentiary_claim["evidentiary"] = True
    with pytest.raises(ValueError, match="declared_non_evidentiary"):
        verify_shakedown_record(value, evidentiary_claim, bound)


def test_shakedown_refuses_after_prepare_outputs_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "authorities.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run, "FROZEN_OUTPUTS", (fake, experiment.DESIGN_AUTHORITIES_PATH))
    with pytest.raises(RuntimeError, match="only BEFORE prepare"):
        run.shakedown()
