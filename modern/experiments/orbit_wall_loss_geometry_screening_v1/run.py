"""Shakedown on real re-solved fields, gated preregistration, then execute exactly once.

Lifecycle (from ``modern/``)::

    python -m experiments.orbit_wall_loss_geometry_screening_v1.run shakedown  # any HEAD, BEFORE prepare
    python -m experiments.orbit_wall_loss_geometry_screening_v1.run prepare    # refuses without a valid shakedown.json
    # commit + push the preregistration, then from a clean detached worktree:
    python -m experiments.orbit_wall_loss_geometry_screening_v1.run execute
    python -m experiments.orbit_wall_loss_geometry_screening_v1.run validate

Ported from ``experiments.cft_orbit_wall_loss_v4.run`` (accepted template).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cft_revival.experiment_runtime import (
    ExecutionAttestation,
    ExperimentRuntime,
    canonical_bytes,
    semantic_sha256,
    strict_json_loads,
    validate_bundle,
)

from .designs import field_pipeline_source_sha256, load_sweep_binding
from .experiment import (
    AUTHORITIES_PATH,
    CLASSIFICATION,
    DESIGN_AUTHORITIES_PATH,
    EXPERIMENT,
    MODERN,
    PROTOCOL_PATH,
    REPOSITORY,
    RESULTS_ROOT,
    SHAKEDOWN_PATH,
    bind_designs,
    build_callbacks,
    build_design_authorities,
    case_matrix,
    design_sha256,
    evidentiary_plan,
    experiment_code_sha256,
    load_frozen_authority,
    plan_record,
    protocol,
    require_orbit_mc_contract,
    schema,
    shakedown_disjointness,
    shakedown_plan,
    source_binding_report,
    verify_shakedown_record,
    worker_count,
)

SUBJECT = "preregister orbit wall-loss geometry screening v1"
REMOTE_BRANCH = "origin/exp/orbit-wall-loss-geometry-screening-v1"
COMMAND = "python -m experiments.orbit_wall_loss_geometry_screening_v1.run execute"
SHAKEDOWN_COMMAND = "python -m experiments.orbit_wall_loss_geometry_screening_v1.run shakedown"
DEVICE = "numpy-cpu-relativistic-boris;cpu-only;gpu-not-used"
FROZEN_OUTPUTS = (AUTHORITIES_PATH, DESIGN_AUTHORITIES_PATH)
EXPERIMENT_PREFIX = "modern/experiments/orbit_wall_loss_geometry_screening_v1/"
TESTS_PREFIX = "modern/tests/experiments/orbit_wall_loss_geometry_screening_v1/"


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=REPOSITORY, check=check, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _git_state() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    branch = _git("symbolic-ref", "-q", "--short", "HEAD", check=False)
    dirty = [line for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line.strip()]
    return {
        "head": head,
        "branch": branch or None,
        "detached": not branch,
        "dirty": bool(dirty),
        "dirty_entry_count": len(dirty),
        "dirty_entries": dirty[:200],
    }


def campaign_producer() -> None:
    """Stable producer identity for the evidentiary one-shot lifecycle."""


def shakedown_producer() -> None:
    """Stable producer identity for the non-evidentiary shakedown lifecycle."""


def _read_access_records(result_root: Path) -> list[dict[str, Any]]:
    records = []
    access_root = result_root / "access"
    if not access_root.is_dir():
        return records
    for path in sorted(access_root.glob("*.json")):
        if path.name.endswith(".sha256.json"):
            continue
        row = strict_json_loads(path.read_bytes())
        records.append(
            {
                "sequence": row["sequence"],
                "operation": row["operation"],
                "kind": row["kind"],
                "details": row["details"],
                "recorded_at_utc": row["recorded_at_utc"]["value"],
            }
        )
    return records


# --------------------------------------------------------------------------
# shakedown
# --------------------------------------------------------------------------


def shakedown() -> dict[str, Any]:
    """Drive the complete production path on three real re-solved designs."""

    existing = [path.name for path in FROZEN_OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(
            "shakedown is allowed only BEFORE prepare; frozen outputs exist: "
            + ", ".join(existing)
            + " (remove them to re-run the shakedown, then re-run prepare)"
        )
    if RESULTS_ROOT.exists():
        raise RuntimeError("shakedown refused: the evidentiary results root already exists")
    value = protocol()
    binding_report = source_binding_report(value)
    contract = binding_report["orbit_mc"]
    plan = shakedown_plan(value)
    sweep = load_sweep_binding(value["field_source"])
    bound = bind_designs(value, sweep, plan.case_ids)
    disjointness = shakedown_disjointness(value, bound)
    if not disjointness["proven"]:
        raise RuntimeError("shakedown design is not disjoint from the evidentiary design")
    git = _git_state()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    temp_root = Path(tempfile.gettempdir()) / f"owlgs-v1-shakedown-{git['head'][:12]}-{stamp}-{token}"
    result_root = temp_root / "bundle"
    cache_root = temp_root / "cache"
    runtime = ExperimentRuntime(
        experiment_id=f"{value['experiment_id']}-shakedown",
        result_root=result_root,
        cache_root=cache_root,
        attestation=ExecutionAttestation(
            attempt=1,
            commit=git["head"],
            command=SHAKEDOWN_COMMAND,
            device=f"{DEVICE};shakedown-non-evidentiary",
            clean_worktree=True,
        ),
        producer=shakedown_producer,
        source_root=MODERN,
    )
    collector: dict[str, Any] = {}
    started = time.perf_counter()
    outcome = runtime.run(build_callbacks(value, plan, frozen=None, collector=collector))
    runtime_seconds = time.perf_counter() - started
    bundle_validated = False
    bundle_error = None
    manifest: dict[str, Any] = {}
    try:
        manifest = dict(validate_bundle(result_root))
        bundle_validated = True
    except Exception as error:  # recorded, never hidden
        bundle_error = f"{type(error).__name__}: {error}"
    access = _read_access_records(result_root)
    cases = collector.get("cases", {})
    all_validators = collector.get("validators", [])
    validator_failures = [item for item in all_validators if not item["passed"]]
    assessment = collector.get("assessment", {})
    development = collector.get("development", {})
    expected_cases = {key for _, _, _, _, key in case_matrix(value, plan)}
    exclusions = development.get("exclusions", [])
    passed = bool(
        outcome.state.value == "accepted_result"
        and bundle_validated
        and disjointness["proven"]
        and set(cases) == expected_cases
        and all(item["validators"]["failed"] == 0 for item in cases.values())
        and all(item["export_stage_ran"] and item["handoff_consumed"] for item in cases.values())
        and not validator_failures
        and not exclusions
        and assessment.get("dataset_summary", {}).get("design_count") == len(plan.case_ids)
    )
    # Timing projection for the evidentiary run (recorded for the extension decision).
    per_orbit = {
        key: item["timing_s"].get("per_orbit_ms") for key, item in cases.items()
    }
    evidentiary = evidentiary_plan(value)
    evidentiary_cases = case_matrix(value, evidentiary)
    n_ms = [v for k, v in per_orbit.items() if k.endswith("-N") and v is not None]
    two_n_ms = [v for k, v in per_orbit.items() if k.endswith("-2N") and v is not None]
    mean_n = sum(n_ms) / max(1, len(n_ms))
    mean_2n = sum(two_n_ms) / max(1, len(two_n_ms))
    per_case_orbits = evidentiary.launches_per_case
    projected_cpu_s = sum(
        3.0 * per_case_orbits * (mean_2n if timestep == "2N" else mean_n) / 1000.0
        for _, _, timestep, _, _ in evidentiary_cases
    )
    projection = {
        "basis": "3 integrations per orbit (integration + write replay + verify replay) at the shakedown's mean per-orbit ms for N and 2N",
        "mean_per_orbit_ms_N": mean_n,
        "mean_per_orbit_ms_2N": mean_2n,
        "evidentiary_case_count": len(evidentiary_cases),
        "evidentiary_orbit_count": per_case_orbits * len(evidentiary_cases),
        "projected_cpu_seconds": projected_cpu_s,
        "worker_pool_size": worker_count(value),
        "projected_wall_seconds_at_pool": projected_cpu_s / max(1, worker_count(value)),
        "budget_wall_seconds": 5400.0,
        "within_budget": projected_cpu_s / max(1, worker_count(value)) <= 5400.0,
    }
    record = {
        "schema_version": schema("shakedown"),
        "classification": CLASSIFICATION,
        "evidentiary": False,
        "outcomes_enter_estimand": False,
        "disclosure": (
            "NON-EVIDENTIARY shakedown of the complete production path on three REAL re-solved "
            "L1a sweep-v2 designs with a launch design disjoint from the evidentiary design. Its "
            "outcomes never enter any estimand; it exists only to prove the code path before the freeze."
        ),
        "attestation_note": (
            "ExecutionAttestation.clean_worktree is a runtime contract constant; the actual git "
            "dirtiness at shakedown time is recorded under git.dirty_entries"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "git": git,
        "protocol_semantic_sha256": semantic_sha256(value),
        "orbit_mc_source_sha256": contract["source_sha256"],
        "orbit_mc_source_hash_line_endings": "LF",
        "orbit_mc_code_identity_sha256": contract["code_identity_sha256"],
        "orbit_mc_package_version": contract["observed"]["package_version"],
        "orbit_mc_schema_versions": contract["observed"],
        "orbit_mc_source_files": contract["source_files"],
        "field_pipeline_source_sha256": binding_report["field_pipeline_source_sha256"],
        "field_pipeline_source_files": binding_report["field_pipeline_source_files"],
        "experiment_code_sha256": binding_report["experiment_code_sha256"],
        "experiment_code_files": binding_report["experiment_code_files"],
        "sweep_binding": {
            "manifest_file_sha256": sweep.manifest_file_sha256,
            "raw_results_file_sha256": sweep.raw_file_sha256,
            "summary_file_sha256": sweep.summary_file_sha256,
        },
        "shakedown_design_sha256": design_sha256(value, plan, bound),
        "shakedown_plan": plan_record(plan),
        "disjointness": disjointness,
        "runtime": {
            "experiment_id": f"{value['experiment_id']}-shakedown",
            "result_root": str(result_root),
            "cache_root": str(cache_root),
            "terminal_state": outcome.state.value,
            "primary_error": outcome.primary_error,
            "secondary_errors": list(outcome.secondary_errors),
            "bundle_validated": bundle_validated,
            "bundle_error": bundle_error,
            "manifest_terminal_byte_sha256": manifest.get("terminal_byte_sha256"),
            "manifest_lock_byte_sha256": manifest.get("lock_byte_sha256"),
            "manifest_artifact_count": manifest.get("artifact_count"),
            "runtime_seconds": runtime_seconds,
        },
        "solver_access_records": [item for item in access if item["kind"] == "solver"],
        "label_access_records": [item for item in access if item["kind"] == "label"],
        "other_access_records": [item for item in access if item["kind"] not in {"solver", "label"}],
        "development": development,
        "design_exclusions": exclusions,
        "cases": cases,
        "validators": {
            "passed": sum(item["passed"] for item in all_validators),
            "failed": len(validator_failures),
            "all_passed": bool(all_validators) and not validator_failures,
            "failures": validator_failures,
        },
        "informational_gates": assessment.get("gates"),
        "design_gates": collector.get("design_gates"),
        "informational_gate_note": (
            "Gates are evaluated but non-binding in the shakedown; sealing uses the structural seal "
            "policy so the export/handoff/consumer path is exercised at 64 launches. Probability "
            "convergence at 64 launches is informational. Magnetic-moment variation is a diagnostic only."
        ),
        "headline_informational": assessment.get("headline"),
        "dataset_summary": assessment.get("dataset_summary"),
        "timing_projection": projection,
        "execution_mode": assessment.get("execution_mode"),
        "case_count": len(cases),
        "passed": passed,
    }
    _write(SHAKEDOWN_PATH, record)
    return {
        "passed": passed,
        "terminal_state": outcome.state.value,
        "primary_error": outcome.primary_error,
        "bundle_validated": bundle_validated,
        "bundle_error": bundle_error,
        "result_root": str(result_root),
        "case_count": len(cases),
        "validators": record["validators"]["passed"],
        "validator_failures": record["validators"]["failed"],
        "exclusions": exclusions,
        "headline_informational": assessment.get("headline"),
        "timing_projection": projection,
        "runtime_seconds": runtime_seconds,
        "shakedown_path": str(SHAKEDOWN_PATH),
        "cases": {
            key: {
                "termination_counts": item["diagnostics"]["termination_counts"],
                "tolerance_close": item["diagnostics"]["tolerance_close_event_count"],
                "validators_passed": item["validators"]["passed"],
                "validators_failed": item["validators"]["failed"],
                "export_stage_ran": item["export_stage_ran"],
                "handoff_consumed": item["handoff_consumed"],
                "integration_s": item["timing_s"].get("integration"),
                "per_orbit_ms": item["timing_s"].get("per_orbit_ms"),
                "max_energy_error": item["diagnostics"]["maximum_relative_energy_error"],
                "mu_diagnostic": item["diagnostics"].get("magnetic_moment_variation_diagnostic"),
            }
            for key, item in cases.items()
        },
    }


# --------------------------------------------------------------------------
# prepare (refuses without a valid shakedown)
# --------------------------------------------------------------------------


def shakedown_gate(value: dict[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, bool]]:
    if not SHAKEDOWN_PATH.is_file():
        raise RuntimeError(f"prepare refused: shakedown.json is missing; run `{SHAKEDOWN_COMMAND}` first")
    data = SHAKEDOWN_PATH.read_bytes()
    record = strict_json_loads(data)
    if not isinstance(record, dict):
        raise RuntimeError("prepare refused: shakedown.json is not an object")
    try:
        checks = verify_shakedown_record(value, record)
    except ValueError as error:
        raise RuntimeError(f"prepare refused: {error}") from error
    return record, data, checks


def prepare() -> dict[str, Any]:
    value = protocol()
    binding_report = source_binding_report(value)
    contract = binding_report["orbit_mc"]
    shakedown_record, shakedown_bytes, shakedown_checks = shakedown_gate(value)
    plan = evidentiary_plan(value)
    sweep = load_sweep_binding(value["field_source"])
    bound = bind_designs(value, sweep, plan.case_ids)
    design_authorities = build_design_authorities(value, plan, bound)
    authorities = {
        "schema_version": schema("authorities"),
        "classification": CLASSIFICATION,
        "protocol_semantic_sha256": semantic_sha256(value),
        "design_authorities_sha256": semantic_sha256(design_authorities),
        "evidentiary_design_sha256": design_sha256(value, plan, bound),
        "design_count": design_authorities["design_count"],
        "case_count": design_authorities["case_count"],
        "total_launches": design_authorities["total_launches"],
        "sweep_manifest_file_sha256": sweep.manifest_file_sha256,
        "sweep_raw_results_file_sha256": sweep.raw_file_sha256,
        "sweep_summary_file_sha256": sweep.summary_file_sha256,
        "minimum_certificate_dense_to_bound_ratio": value["gates"]["minimum_certificate_dense_to_bound_ratio"],
        "orbit_mc_source_sha256": contract["source_sha256"],
        "orbit_mc_source_hash_line_endings": "LF",
        "orbit_mc_code_identity_sha256": contract["code_identity_sha256"],
        "orbit_mc_package_version": contract["observed"]["package_version"],
        "orbit_mc_schema_versions": contract["observed"],
        "field_pipeline_source_sha256": binding_report["field_pipeline_source_sha256"],
        "experiment_code_sha256": binding_report["experiment_code_sha256"],
        "shakedown_file_sha256": hashlib.sha256(shakedown_bytes).hexdigest(),
        "shakedown_semantic_sha256": semantic_sha256(shakedown_record),
        "shakedown_design_sha256": shakedown_record["shakedown_design_sha256"],
        "shakedown_git_head": shakedown_record["git"]["head"],
        "shakedown_timing_projection": shakedown_record["timing_projection"],
        "shakedown_gate_checks": shakedown_checks,
    }
    _write(DESIGN_AUTHORITIES_PATH, design_authorities)
    _write(AUTHORITIES_PATH, authorities)
    return {
        "passed": all(shakedown_checks.values()),
        "design_count": authorities["design_count"],
        "case_count": authorities["case_count"],
        "total_launches": authorities["total_launches"],
        "protocol_semantic_sha256": authorities["protocol_semantic_sha256"],
        "design_authorities_sha256": authorities["design_authorities_sha256"],
        "evidentiary_design_sha256": authorities["evidentiary_design_sha256"],
        "orbit_mc_source_sha256": authorities["orbit_mc_source_sha256"],
        "field_pipeline_source_sha256": authorities["field_pipeline_source_sha256"],
        "shakedown_file_sha256": authorities["shakedown_file_sha256"],
        "shakedown_gate_checks": shakedown_checks,
        "shakedown_timing_projection": authorities["shakedown_timing_projection"],
    }


# --------------------------------------------------------------------------
# execute (one immutable attempt)
# --------------------------------------------------------------------------


def _bind_preregistration() -> str:
    head = _git("rev-parse", "HEAD")
    if _git("symbolic-ref", "-q", "HEAD", check=False):
        raise RuntimeError("execution requires detached HEAD")
    if _git("show", "-s", "--format=%s", head) != SUBJECT:
        raise RuntimeError("HEAD is not the exact preregistration commit")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", head, REMOTE_BRANCH), cwd=REPOSITORY, capture_output=True
    ).returncode:
        raise RuntimeError("preregistration commit is not pushed to the authorized branch")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("execution requires a clean detached worktree")
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
    if not changed or any(not item.startswith((EXPERIMENT_PREFIX, TESTS_PREFIX)) for item in changed):
        raise RuntimeError("preregistration commit is not experiment-path isolated")
    if any("/results/" in item for item in changed):
        raise RuntimeError("preregistration commit contains outcome artifacts")
    for path in (PROTOCOL_PATH, AUTHORITIES_PATH, DESIGN_AUTHORITIES_PATH, SHAKEDOWN_PATH):
        if not path.is_file():
            raise RuntimeError(f"missing preregistered authority: {path.name}")
    value = protocol()
    contract = require_orbit_mc_contract(value)
    frozen = load_frozen_authority()
    if frozen.authorities["orbit_mc_source_sha256"] != contract["source_sha256"]:
        raise RuntimeError("orbit_mc source hash differs from the preregistered authority")
    if frozen.authorities["orbit_mc_package_version"] != contract["observed"]["package_version"]:
        raise RuntimeError("orbit_mc package version differs from the preregistered authority")
    if frozen.authorities["orbit_mc_schema_versions"] != contract["observed"]:
        raise RuntimeError("orbit_mc schema versions differ from the preregistered authority")
    if frozen.authorities["field_pipeline_source_sha256"] != field_pipeline_source_sha256():
        raise RuntimeError("field pipeline source hash differs from the preregistered authority")
    if frozen.authorities["experiment_code_sha256"] != experiment_code_sha256():
        raise RuntimeError("experiment code hash differs from the preregistered authority")
    if frozen.authorities["shakedown_file_sha256"] != hashlib.sha256(frozen.shakedown_bytes).hexdigest():
        raise RuntimeError("shakedown.json differs from the preregistered authority")
    try:
        verify_shakedown_record(value, frozen.shakedown)
    except ValueError as error:
        raise RuntimeError(f"execution refused: {error}") from error
    return head


def _acquire_git_common_lock(commit: str) -> Path:
    common = Path(_git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (REPOSITORY / common).resolve()
    path = common / protocol()["execution"]["git_common_lock"]
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
        stream.write(commit + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def execute() -> dict[str, Any]:
    commit = _bind_preregistration()
    git_lock = _acquire_git_common_lock(commit)
    value = protocol()
    cache_root = Path(tempfile.gettempdir()) / f"owlgs-v1-{commit[:12]}-working-cache"
    runtime = ExperimentRuntime(
        experiment_id=value["experiment_id"],
        result_root=RESULTS_ROOT,
        cache_root=cache_root,
        attestation=ExecutionAttestation(attempt=1, commit=commit, command=COMMAND, device=DEVICE, clean_worktree=True),
        producer=campaign_producer,
        source_root=MODERN,
    )
    collector: dict[str, Any] = {}
    outcome = runtime.run(
        build_callbacks(value, evidentiary_plan(value), frozen=load_frozen_authority(), collector=collector)
    )
    manifest = validate_bundle(RESULTS_ROOT)
    return {
        "state": outcome.state.value,
        "manifest_state": manifest["state"],
        "primary_error": outcome.primary_error,
        "preregistration_commit": commit,
        "git_common_lock": git_lock.name,
        "single_execution": True,
        "no_patch_or_rerun": True,
        "worker_pool_size": worker_count(value),
        "headline": (collector.get("assessment") or {}).get("headline"),
        "status": (collector.get("assessment") or {}).get("status"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("shakedown", "prepare", "execute", "validate", "contract"))
    arguments = parser.parse_args(argv)
    if arguments.command == "shakedown":
        output = shakedown()
    elif arguments.command == "prepare":
        output = prepare()
    elif arguments.command == "execute":
        output = execute()
    elif arguments.command == "contract":
        output = source_binding_report(protocol())
    else:
        output = validate_bundle(RESULTS_ROOT)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
