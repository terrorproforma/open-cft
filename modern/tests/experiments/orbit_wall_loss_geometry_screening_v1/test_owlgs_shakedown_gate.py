"""Shakedown design, disjointness and the prepare/execute shakedown gate (ported from v4)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import canonical_bytes, semantic_sha256
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.orbit_wall_loss_geometry_screening_v1 import designs as D
from experiments.orbit_wall_loss_geometry_screening_v1 import experiment, run
from experiments.orbit_wall_loss_geometry_screening_v1.experiment import (
    SHAKEDOWN_PATH,
    all_plan_launches,
    batch_records,
    bind_designs,
    build_case_launches,
    build_design_authorities,
    case_matrix,
    design_sha256,
    evidentiary_plan,
    protocol,
    schema,
    shakedown_disjointness,
    shakedown_plan,
    shakedown_positions,
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
    binding = D.load_sweep_binding(value["field_source"])
    return bind_designs(value, binding, shakedown_plan(value).case_ids)


def test_shakedown_design_is_deterministic_small_and_stratified(value: dict, bound: dict) -> None:
    plan = shakedown_plan(value)
    assert plan.kind == "shakedown"
    assert plan.binding_gates is False
    assert len(plan.case_ids) == 3
    for case_id in plan.case_ids:
        geometry = bound[case_id].geometry
        assert shakedown_positions(value, geometry) == shakedown_positions(value, geometry)
        positions = shakedown_positions(value, geometry)
        assert len(positions) == 8
        low, high = value["shakedown"]["radius_fraction_range"]
        for surface, (x, y, z) in positions:
            assert y == 0.0
            assert low * geometry.wall_radius_m <= x <= high * geometry.wall_radius_m
            assert 0.0 < z < geometry.exit_start_m
            assert surface.startswith("sd-gs1-cell-")
        for role, timestep in experiment.case_roles(value, case_id):
            launches = build_case_launches(value, plan, geometry, role, timestep)
            assert len(launches) == 64
            assert all(item.launch_id.startswith(f"owlgs-v1-shakedown:{case_id}:{role}:{timestep}:") for item in launches)
            strata = Counter(
                (item.flux_surface_id.split("-r", 1)[0], item.kinetic_energy_ev, item.pitch_angle_rad, item.parallel_direction)
                for item in launches
            )
            assert len(strata) == 32 and set(strata.values()) == {2}
            assert len(batch_records(plan, launches)) == 8
    matrix = case_matrix(value, plan)
    assert len(matrix) == 7  # 3 designs x (N, 2N) + refined-N for the representative


def test_shakedown_is_disjoint_from_the_evidentiary_design(value: dict, bound: dict) -> None:
    report = shakedown_disjointness(value, bound)
    assert report["proven"] is True
    assert report["shakedown_launch_count"] == 7 * 64
    assert report["shakedown_unique_launch_ids"] == 7 * 64
    assert report["shakedown_unique_seed_ids"] == 7 * 64
    assert set(report["reports"]) == {"against_evidentiary_same_designs"}
    for item in report["reports"].values():
        assert item["disjoint"] is True
        assert set(item["overlap_counts"].values()) == {0}
    assert report["gyrophase_grids"]["disjoint"] is True
    same_designs = experiment.CampaignPlan(
        **{**evidentiary_plan(value).__dict__, "case_ids": shakedown_plan(value).case_ids}
    )
    assert design_sha256(value, shakedown_plan(value), bound) != design_sha256(value, same_designs, bound)


def test_shakedown_and_evidentiary_authorities_carry_distinct_policies(value: dict, bound: dict) -> None:
    shakedown = build_design_authorities(value, shakedown_plan(value), bound)
    evidentiary_bound = {k: v for k, v in bound.items()}
    evidentiary = build_design_authorities(
        value,
        experiment.CampaignPlan(**{**evidentiary_plan(value).__dict__, "case_ids": shakedown_plan(value).case_ids}),
        evidentiary_bound,
    )
    assert shakedown["plan_kind"] == "shakedown" and evidentiary["plan_kind"] == "evidentiary"
    for left, right in zip(shakedown["cases"], evidentiary["cases"], strict=True):
        assert left["case_key"] == right["case_key"]
        assert left["campaign_id"] != right["campaign_id"]
        assert left["policy_identity_sha256"] != right["policy_identity_sha256"]
        assert left["orbit_launches_sha256"] != right["orbit_launches_sha256"]
        assert left["field_identity_sha256"] == right["field_identity_sha256"]
        assert left["config_identity_sha256"] == right["config_identity_sha256"]
        assert left["launch_count"] == 64 and right["launch_count"] == 512


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
    return (
        experiment.orbit_mc_package.__version__ != authorities["orbit_mc_package_version"]
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
    assert record["case_count"] == 7
    assert set(record["cases"]) == {key for _, _, _, _, key in case_matrix(value, shakedown_plan(value))}
    assert record["validators"]["failed"] == 0 and record["validators"]["passed"] > 0
    assert record["design_exclusions"] == []
    assert len(record["solver_access_records"]) == 1
    assert [item["operation"] for item in record["label_access_records"]] == [
        f"orbit-{key}" for _, _, _, _, key in case_matrix(value, shakedown_plan(value))
    ]
    assert record["orbit_mc_package_version"] == "1.7.0"
    assert record["orbit_mc_source_hash_line_endings"] == "LF"
    assert record["timing_projection"]["within_budget"] is value["designs"]["extension_batch_included"]
    assert set(record["experiment_code_files"]) == set(experiment.EXPERIMENT_CODE_FILES)
    assert record["dataset_summary"]["design_count"] == 3
    for item in record["cases"].values():
        assert item["export_stage_ran"] is True and item["handoff_consumed"] is True
        assert item["validators"]["failed"] == 0
        counts = item["diagnostics"]["termination_counts"]
        assert counts["field_failure"] == 0 and counts["step_limit"] == 0 and counts["initial_state_invalid"] == 0
        assert item["diagnostics"]["maximum_relative_energy_error"] == 0.0
        assert item["diagnostics"]["final_velocity_equals_event_velocity_count"] == 64
        mu = item["diagnostics"]["magnetic_moment_variation_diagnostic"]
        assert mu["role"] == "diagnostic_only" and mu["min"] <= mu["median"] <= mu["max"]
    gates = record["informational_gates"]
    assert gates["binding"] is False
    assert not any("mu" in key.lower().split("_") or "magnetic_moment" in key.lower() for key in gates["manufactured"])
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
        assert len(all_plan_launches(value, shakedown_plan(value), bound)) == 7 * 64
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
        # The frozen outputs exist; prepare must not rewrite them (results exist -> refuse).
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
    excluded = copy.deepcopy(record)
    excluded["design_exclusions"] = [{"case_id": "x", "reason": "y"}]
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
