"""Shakedown design disjointness and the prepare/execute shakedown gate."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import canonical_bytes, semantic_sha256
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.cft_orbit_wall_loss_v4 import experiment, run
from experiments.cft_orbit_wall_loss_v4.experiment import (
    ROLES,
    SHAKEDOWN_PATH,
    TIMESTEPS,
    all_plan_launches,
    batch_records,
    build_all_case_authorities,
    build_case_launches,
    case_key,
    design_sha256,
    evidentiary_plan,
    protocol,
    schema,
    shakedown_disjointness,
    shakedown_plan,
    shakedown_positions,
    verify_shakedown_record,
)

FROZEN = (
    experiment.AUTHORITIES_PATH,
    experiment.CASE_AUTHORITIES_PATH,
    experiment.SYNTHETIC_PREFLIGHT_PATH,
    experiment.CASE_ROOT,
)


def _frozen_state() -> tuple[bool, ...]:
    return tuple(path.exists() for path in FROZEN)


# --------------------------------------------------------------------------
# shakedown design
# --------------------------------------------------------------------------


def test_shakedown_design_is_deterministic_small_and_stratified() -> None:
    value = protocol()
    plan = shakedown_plan(value)
    assert plan.kind == "shakedown"
    assert plan.binding_gates is False
    assert shakedown_positions(value) == shakedown_positions(value)
    assert len(plan.positions) == 8
    wall_radius = value["orbit"]["wall"]["radius_m"]
    low, high = value["shakedown"]["radius_fraction_range"]
    for surface, (x, y, z) in plan.positions:
        assert y == 0.0
        assert low * wall_radius <= x <= high * wall_radius
        assert value["orbit"]["domain"]["z_min_m"] < z < value["orbit"]["domain"]["z_max_m"]
        assert surface.split("-r", 1)[0].startswith("v4sd-cell-")
    for role, timestep in ((r, t) for r in ROLES for t in TIMESTEPS):
        launches = build_case_launches(value, plan, role, timestep)
        assert len(launches) == 64
        assert all(
            item.launch_id.startswith(f"cft-orbit-wall-loss-v4-shakedown:{role}:{timestep}:")
            for item in launches
        )
        strata = Counter(
            (
                item.flux_surface_id.split("-r", 1)[0],
                item.kinetic_energy_ev,
                item.pitch_angle_rad,
                item.parallel_direction,
            )
            for item in launches
        )
        assert len(strata) == 32
        assert set(strata.values()) == {2}
        batches = batch_records(plan, launches)
        assert len(batches) == 8
        assert all(len(batch["launches"]) == 8 for batch in batches)


def test_shakedown_design_is_disjoint_from_evidentiary_and_prior_designs() -> None:
    value = protocol()
    report = shakedown_disjointness(value)
    assert report["proven"] is True
    assert report["shakedown_launch_count"] == 576
    assert report["shakedown_unique_launch_ids"] == 576
    assert report["shakedown_unique_seed_ids"] == 576
    assert set(report["reports"]) == {
        "against_evidentiary_v4",
        "against_v1",
        "against_v2",
        "against_v3",
    }
    for item in report["reports"].values():
        assert item["disjoint"] is True
        assert set(item["overlap_counts"]) == {
            "launch_id",
            "seed_id",
            "position_m",
            "energy_pitch_direction_gyrophase",
            "full_phase_space",
        }
        assert set(item["overlap_counts"].values()) == {0}
    # Gyrophase grids are disjoint by construction (offset differs by pi/32).
    evidentiary = set(evidentiary_plan(value).gyrophases_rad)
    shakedown = set(shakedown_plan(value).gyrophases_rad)
    assert not (evidentiary & shakedown)
    assert design_sha256(value, shakedown_plan(value)) != design_sha256(
        value, evidentiary_plan(value)
    )


def test_shakedown_and_evidentiary_authorities_carry_distinct_policies() -> None:
    value = protocol()
    shakedown = build_all_case_authorities(value, shakedown_plan(value))
    evidentiary = build_all_case_authorities(value, evidentiary_plan(value))
    assert shakedown["plan_kind"] == "shakedown"
    assert evidentiary["plan_kind"] == "evidentiary"
    assert shakedown["total_case_launches"] == 576
    assert evidentiary["total_case_launches"] == 4608
    for left, right in zip(shakedown["cases"], evidentiary["cases"], strict=True):
        assert left["case_key"] == right["case_key"]
        assert left["campaign_id"] != right["campaign_id"]
        assert left["policy_identity_sha256"] != right["policy_identity_sha256"]
        assert left["orbit_launches_sha256"] != right["orbit_launches_sha256"]
        assert left["field_identity_sha256"] == right["field_identity_sha256"]
        assert left["config_identity_sha256"] == right["config_identity_sha256"]


# --------------------------------------------------------------------------
# the committed shakedown record
# --------------------------------------------------------------------------


def _record() -> dict:
    if not SHAKEDOWN_PATH.is_file():
        pytest.skip("shakedown.json has not been produced yet")
    return strict_json_file(SHAKEDOWN_PATH)


def test_committed_shakedown_record_opens_the_gate_for_current_protocol_and_code() -> None:
    value = protocol()
    record = _record()
    checks = verify_shakedown_record(value, record)
    assert all(checks.values()), checks
    assert record["evidentiary"] is False
    assert record["outcomes_enter_estimand"] is False
    assert record["passed"] is True
    assert record["case_count"] == 9
    assert set(record["cases"]) == {case_key(r, t) for r in ROLES for t in TIMESTEPS}
    assert record["validators"]["failed"] == 0
    assert record["validators"]["passed"] > 0
    assert len(record["p2_access_records"]) == 3
    assert [item["operation"] for item in record["label_access_records"]] == [
        f"orbit-{case_key(r, t)}" for r in ROLES for t in TIMESTEPS
    ]
    gates = record["informational_gates"]
    assert gates["binding"] is False
    assert gates["exact_authority_replay"] is True
    assert "energy" in gates["checks"]
    assert "final_velocity_equals_event_velocity" in gates["checks"]
    # mu is a diagnostic: it must not appear as any gate check.
    assert not any(
        "mu" in key.lower().split("_") or "magnetic_moment" in key.lower()
        for key in gates["checks"]
    )
    assert gates["diagnostics_not_gates"]["magnetic_moment_variation"]["binding"] is False
    # v1.6: the energy gate is satisfied exactly on the real field.
    assert record["orbit_mc_package_version"] == "1.6.0"
    assert record["orbit_mc_source_hash_line_endings"] == "LF"
    assert gates["checks"]["energy"] is True
    assert gates["checks"]["final_velocity_equals_event_velocity"] is True
    assert gates["maximum_relative_energy_error"] == 0.0
    assert gates["orbits_exceeding_energy_gate"] == 0
    assert gates["final_velocity_event_velocity_mismatches"] == 0
    assert record["energy_summary"]["orbits_with_nonzero_energy_error"] == 0
    for item in record["cases"].values():
        assert item["export_stage_ran"] is True
        assert item["validators"]["failed"] == 0
        counts = item["diagnostics"]["termination_counts"]
        assert counts["field_failure"] == 0
        assert counts["step_limit"] == 0
        assert counts["initial_state_invalid"] == 0
        assert item["diagnostics"]["tolerance_close_event_count"] > 0
        assert item["diagnostics"]["maximum_relative_energy_error"] == 0.0
        assert item["diagnostics"]["final_velocity_equals_event_velocity_count"] == 64
        mu = item["diagnostics"]["magnetic_moment_variation_diagnostic"]
        assert mu["role"] == "diagnostic_only"
        assert mu["min"] <= mu["median"] <= mu["max"]
    assert "NON-EVIDENTIARY" in record["disclosure"]


# --------------------------------------------------------------------------
# prepare refusals (must never write frozen outputs)
# --------------------------------------------------------------------------


def test_prepare_refuses_without_shakedown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    before = _frozen_state()
    monkeypatch.setattr(run, "SHAKEDOWN_PATH", tmp_path / "missing-shakedown.json")
    with pytest.raises(RuntimeError, match="shakedown.json is missing"):
        run.prepare()
    assert _frozen_state() == before


def test_prepare_refuses_on_code_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _record()
    before = _frozen_state()
    monkeypatch.setattr(
        experiment, "orbit_mc_source_sha256", lambda: "f" * 64
    )
    with pytest.raises(RuntimeError, match="orbit_mc_source_sha256_current"):
        run.prepare()
    assert _frozen_state() == before


def test_prepare_refuses_on_stale_protocol_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    _record()
    before = _frozen_state()
    stale = copy.deepcopy(protocol())
    stale["gates"]["maximum_relative_energy_error"] = 1e-9
    monkeypatch.setattr(run, "protocol", lambda: stale)
    with pytest.raises(RuntimeError, match="protocol_semantic_sha256_current"):
        run.prepare()
    assert _frozen_state() == before


def test_prepare_refuses_on_failed_or_tampered_shakedown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = _record()
    before = _frozen_state()
    failed = copy.deepcopy(record)
    failed["passed"] = False
    path = tmp_path / "shakedown.json"
    path.write_bytes(canonical_bytes(failed))
    monkeypatch.setattr(run, "SHAKEDOWN_PATH", path)
    with pytest.raises(RuntimeError, match="passed"):
        run.prepare()
    tampered = copy.deepcopy(record)
    tampered["disjointness"]["reports"]["against_v3"]["overlap_counts"]["seed_id"] = 1
    path.write_bytes(canonical_bytes(tampered))
    with pytest.raises(RuntimeError, match="disjointness_proven"):
        run.prepare()
    missing_export = copy.deepcopy(record)
    missing_export["cases"]["primary-N"]["export_stage_ran"] = False
    path.write_bytes(canonical_bytes(missing_export))
    with pytest.raises(RuntimeError, match="all_validators_passed"):
        run.prepare()
    assert _frozen_state() == before


def test_verify_shakedown_record_rejects_wrong_schema_and_design() -> None:
    value = protocol()
    record = _record()
    wrong_schema = copy.deepcopy(record)
    wrong_schema["schema_version"] = schema("shakedown").replace("4.0.0", "3.0.0")
    with pytest.raises(ValueError, match="schema_version"):
        verify_shakedown_record(value, wrong_schema)
    wrong_design = copy.deepcopy(record)
    wrong_design["shakedown_design_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shakedown_design_sha256_current"):
        verify_shakedown_record(value, wrong_design)
    evidentiary_claim = copy.deepcopy(record)
    evidentiary_claim["evidentiary"] = True
    with pytest.raises(ValueError, match="declared_non_evidentiary"):
        verify_shakedown_record(value, evidentiary_claim)


def test_shakedown_refuses_after_prepare_outputs_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_authorities = tmp_path / "authorities.json"
    fake_authorities.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        run,
        "FROZEN_OUTPUTS",
        (fake_authorities, experiment.CASE_AUTHORITIES_PATH),
    )
    with pytest.raises(RuntimeError, match="only BEFORE prepare"):
        run.shakedown()


def test_shakedown_record_hash_bindings_are_reproducible() -> None:
    value = protocol()
    record = _record()
    data = SHAKEDOWN_PATH.read_bytes()
    assert json.loads(data.decode("utf-8")) == record
    assert record["protocol_semantic_sha256"] == semantic_sha256(value)
    assert record["shakedown_design_sha256"] == design_sha256(value, shakedown_plan(value))
    assert record["evidentiary_design_sha256"] == design_sha256(value, evidentiary_plan(value))
    assert len(all_plan_launches(value, shakedown_plan(value))) == 576
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(SHAKEDOWN_PATH.read_bytes()).hexdigest()
