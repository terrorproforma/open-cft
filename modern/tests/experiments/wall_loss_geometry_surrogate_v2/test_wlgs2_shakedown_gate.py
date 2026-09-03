"""Shakedown record verification, the prepare/execute gate and lifecycle-aware authorities."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from cft_revival.experiment_runtime import semantic_sha256
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.wall_loss_geometry_surrogate_v2 import experiment, run
from experiments.wall_loss_geometry_surrogate_v2.experiment import (
    AUTHORITIES_PATH,
    PARTITIONS_PATH,
    RESULTS_ROOT,
    SHAKEDOWN_PATH,
    protocol,
    verify_shakedown_record,
)

RECORDED = RESULTS_ROOT / "manifest.json"


def _live_record() -> dict:
    if not SHAKEDOWN_PATH.is_file():
        pytest.skip("shakedown.json not yet produced")
    return strict_json_file(SHAKEDOWN_PATH)


def _require_ml_runtime() -> None:
    for name in ("torch", "botorch", "gpytorch", "scipy"):
        pytest.importorskip(name)


def _recorded_lifecycle() -> bool:
    return RECORDED.is_file()


def test_shakedown_record_passes_the_gate_or_is_bound_to_the_recorded_bundle() -> None:
    value = protocol()
    record = _live_record()
    assert b"\r" not in SHAKEDOWN_PATH.read_bytes()
    if _recorded_lifecycle():
        bundle_copy = (RESULTS_ROOT / "artifacts" / "shakedown.json").read_bytes()
        assert bundle_copy == SHAKEDOWN_PATH.read_bytes()
        authorities = strict_json_file(AUTHORITIES_PATH)
        assert authorities["shakedown_file_sha256"] == hashlib.sha256(bundle_copy).hexdigest()
        assert authorities["shakedown_semantic_sha256"] == semantic_sha256(record)
        assert authorities["protocol_semantic_sha256"] == semantic_sha256(value)
        assert record["protocol_semantic_sha256"] == semantic_sha256(value)
        assert record["passed"] is True and record["evidentiary"] is False
        return
    _require_ml_runtime()
    checks = verify_shakedown_record(value, record)
    assert checks and all(checks.values())
    assert record["evidentiary"] is False and record["outcomes_enter_estimand"] is False
    assert record["runtime"]["terminal_state"] == "accepted_result"
    assert record["runtime"]["bundle_validated"] is True
    assert set(record["development"]["candidates_fitted"]) == set(experiment.m.CANDIDATE_ORDER)
    assert record["informational_gates"]["binding_in_this_plan"] is False


@pytest.mark.parametrize(
    "tamper, expected",
    [
        (lambda r: r.__setitem__("passed", False), "passed"),
        (lambda r: r.__setitem__("evidentiary", True), "declared_non_evidentiary"),
        (lambda r: r.__setitem__("protocol_semantic_sha256", "0" * 64), "protocol_semantic_sha256_current"),
        (lambda r: r.__setitem__("source_sha256", "0" * 64), "source_sha256_current"),
        (lambda r: r["package_versions"].__setitem__("torch", "0.0.0"), "package_versions_current"),
        (lambda r: r["disjointness"].__setitem__("proven", False), "disjointness_proven"),
        (lambda r: r.__setitem__("shakedown_design_sha256", "0" * 64), "shakedown_design_sha256_current"),
        (lambda r: r.__setitem__("evidentiary_design_sha256", "0" * 64), "evidentiary_design_sha256_current"),
        (lambda r: r["development"].__setitem__("candidates_fitted", ["botorch-stgp-logit"]), "all_candidates_fitted"),
        (lambda r: r["development"].__setitem__("predictor_replay_passed", False), "predictor_replay_passed"),
        (lambda r: r["development"].__setitem__("determinism_replay_passed", False), "determinism_replay_passed"),
        (lambda r: r["runtime"].__setitem__("terminal_state", "runtime_failure"), "runtime_accepted_and_bundle_validated"),
        (lambda r: r.__setitem__("informational_gates", {}), "gates_evaluated"),
        (lambda r: r.__setitem__("schema_version", "other"), "schema_version"),
    ],
)
def test_tampered_shakedown_records_are_refused(tamper, expected) -> None:
    if _recorded_lifecycle():
        pytest.skip("live gate binding is superseded by the recorded bundle after execution")
    _require_ml_runtime()
    value = protocol()
    record = copy.deepcopy(_live_record())
    verify_shakedown_record(value, record)
    tamper(record)
    with pytest.raises(ValueError, match=expected):
        verify_shakedown_record(value, record)


def test_protocol_edit_after_shakedown_is_refused() -> None:
    if _recorded_lifecycle():
        pytest.skip("live gate binding is superseded by the recorded bundle after execution")
    _require_ml_runtime()
    record = _live_record()
    value = copy.deepcopy(protocol())
    value["gates"]["binding"]["interpolation_rmse_pooled"]["threshold"] = 0.06
    with pytest.raises(ValueError, match="protocol_semantic_sha256_current"):
        verify_shakedown_record(value, record)


def test_prepare_refuses_without_a_shakedown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run, "SHAKEDOWN_PATH", tmp_path / "missing-shakedown.json")
    with pytest.raises(RuntimeError, match="shakedown.json is missing"):
        run.shakedown_gate(protocol())


def test_prepare_refuses_a_non_object_shakedown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "shakedown.json"
    path.write_bytes(b"[]")
    monkeypatch.setattr(run, "SHAKEDOWN_PATH", path)
    with pytest.raises(RuntimeError, match="not an object"):
        run.shakedown_gate(protocol())


def test_shakedown_refuses_after_prepare_or_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    existing = tmp_path / "authorities.json"
    existing.write_bytes(b"{}")
    monkeypatch.setattr(run, "FROZEN_OUTPUTS", (existing,))
    with pytest.raises(RuntimeError, match="only BEFORE prepare"):
        run.shakedown()
    monkeypatch.setattr(run, "FROZEN_OUTPUTS", ())
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(run, "RESULTS_ROOT", results)
    with pytest.raises(RuntimeError, match="results root already exists"):
        run.shakedown()


def test_bind_preregistration_constants() -> None:
    assert run.SUBJECT == "preregister wall-loss geometry surrogate v2"
    assert run.REMOTE_BRANCH == "origin/exp/wall-loss-geometry-surrogate-v2"
    assert protocol()["execution"]["git_common_lock"] == "wall-loss-geometry-surrogate-v2.execution.lock"
    assert run.DEVICE.startswith("cpu") and "gpu-not-used" in run.DEVICE


def test_frozen_authorities_bind_protocol_partition_source_and_shakedown_when_present() -> None:
    if not AUTHORITIES_PATH.is_file():
        pytest.skip("authorities.json not yet prepared")
    value = protocol()
    authorities = strict_json_file(AUTHORITIES_PATH)
    record = _live_record()
    assert b"\r" not in AUTHORITIES_PATH.read_bytes()
    assert authorities["protocol_semantic_sha256"] == semantic_sha256(value)
    assert authorities["shakedown_file_sha256"] == hashlib.sha256(SHAKEDOWN_PATH.read_bytes()).hexdigest()
    assert authorities["shakedown_semantic_sha256"] == semantic_sha256(record)
    assert authorities["source_sha256"] == record["source_sha256"]
    assert authorities["partitions_file_sha256"] == hashlib.sha256(PARTITIONS_PATH.read_bytes()).hexdigest()
    assert authorities["partitions_semantic_sha256"] == semantic_sha256(strict_json_file(PARTITIONS_PATH))
    assert authorities["dataset_file_sha256"] == value["dataset"]["dataset_file_sha256"]
    rows = experiment.load_rows(value)
    partition = experiment.plan_partition(value, experiment.evidentiary_plan(value), rows)
    assert authorities["evidentiary_design_sha256"] == experiment.design_sha256(value, experiment.evidentiary_plan(value), partition)
    if not _recorded_lifecycle():
        assert experiment.code_contract_report(value)["source_sha256"] == authorities["source_sha256"]
    else:
        contract = strict_json_file(RESULTS_ROOT / "artifacts" / "code-contract.json")
        assert contract["source_sha256"] == authorities["source_sha256"]
