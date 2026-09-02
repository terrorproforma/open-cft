"""Shakedown on the real fields, gated preregistration, then execute exactly once.

Lifecycle (from ``modern/``)::

    python -m experiments.cft_orbit_wall_loss_v4.run shakedown   # any HEAD, dirty ok, BEFORE prepare
    python -m experiments.cft_orbit_wall_loss_v4.run prepare     # refuses without a valid shakedown.json
    # commit + push the preregistration, then from a clean detached worktree:
    python -m experiments.cft_orbit_wall_loss_v4.run execute
    python -m experiments.cft_orbit_wall_loss_v4.run validate
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
from cft_revival.orbit_mc.artifacts import content_hash

from .experiment import (
    AUTHORITIES_PATH,
    CASE_AUTHORITIES_PATH,
    CASE_ROOT,
    EXPERIMENT,
    MODERN,
    PROTOCOL_PATH,
    REPOSITORY,
    RESULTS_ROOT,
    ROLES,
    SHAKEDOWN_PATH,
    SYNTHETIC_PREFLIGHT_PATH,
    TIMESTEPS,
    build_all_case_authorities,
    build_callbacks,
    build_case_launches,
    case_key,
    design_sha256,
    evidentiary_plan,
    load_frozen_authority,
    manufactured_gate_report,
    orbit_mc_contract_report,
    plan_record,
    production_synthetic_preflight,
    protocol,
    require_orbit_mc_contract,
    runtime_batch_payload,
    runtime_launch_payload,
    schema,
    shakedown_disjointness,
    shakedown_plan,
    verify_shakedown_record,
    worker_count,
)

SUBJECT = "preregister CFT full-orbit wall-loss v4"
REMOTE_BRANCH = "origin/exp/cft-orbit-wall-loss-v4"
COMMAND = "python -m experiments.cft_orbit_wall_loss_v4.run execute"
SHAKEDOWN_COMMAND = "python -m experiments.cft_orbit_wall_loss_v4.run shakedown"
DEVICE = "numpy-cpu-reference+warp-cpu-cuda-parity"
FROZEN_OUTPUTS = (AUTHORITIES_PATH, CASE_AUTHORITIES_PATH, SYNTHETIC_PREFLIGHT_PATH, CASE_ROOT)


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _git_state() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    branch = _git("symbolic-ref", "-q", "--short", "HEAD", check=False)
    dirty = [
        line
        for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line.strip()
    ]
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


# --------------------------------------------------------------------------
# shakedown
# --------------------------------------------------------------------------


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


def shakedown() -> dict[str, Any]:
    """Drive the complete production path on the real fields with a disjoint design."""

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
    contract = require_orbit_mc_contract(value)
    plan = shakedown_plan(value)
    evidentiary = evidentiary_plan(value)
    disjointness = shakedown_disjointness(value)
    if not disjointness["proven"]:
        raise RuntimeError("shakedown design is not disjoint from the evidentiary/prior designs")
    git = _git_state()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    temp_root = (
        Path(tempfile.gettempdir())
        / f"cft-orbit-wall-loss-v4-shakedown-{git['head'][:12]}-{stamp}-{token}"
    )
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
    expected_cases = {case_key(role, timestep) for role in ROLES for timestep in TIMESTEPS}
    passed = bool(
        outcome.state.value == "accepted_result"
        and bundle_validated
        and disjointness["proven"]
        and set(cases) == expected_cases
        and all(item["validators"]["failed"] == 0 for item in cases.values())
        and all(item["export_stage_ran"] for item in cases.values())
        and not validator_failures
    )
    record = {
        "schema_version": schema("shakedown"),
        "evidentiary": False,
        "outcomes_enter_estimand": False,
        "disclosure": (
            "NON-EVIDENTIARY shakedown of the complete production path on the real "
            "divergent-exit P2 fields with a launch design disjoint from the "
            "evidentiary v4 design and from v1/v2/v3. Its outcomes never enter any "
            "estimand; it exists only to prove the code path before the freeze."
        ),
        "attestation_note": (
            "ExecutionAttestation.clean_worktree is a runtime contract constant; the "
            "actual git dirtiness at shakedown time is recorded under git.dirty_entries"
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
        "shakedown_design_sha256": design_sha256(value, plan),
        "evidentiary_design_sha256": design_sha256(value, evidentiary),
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
        "p2_access_records": [item for item in access if item["kind"] == "solver"],
        "label_access_records": [item for item in access if item["kind"] == "label"],
        "other_access_records": [
            item for item in access if item["kind"] not in {"solver", "label"}
        ],
        "development": development,
        "cases": cases,
        "validators": {
            "passed": sum(item["passed"] for item in all_validators),
            "failed": len(validator_failures),
            "all_passed": bool(all_validators) and not validator_failures,
            "failures": validator_failures,
        },
        "informational_gates": assessment.get("gates"),
        "informational_gate_note": (
            "Gates are evaluated but non-binding in the shakedown. The energy gate "
            "(<= 1e-10) is BINDING in the evidentiary run and is expected to pass "
            "exactly (0.0) under orbit_mc v1.6. Magnetic-moment variation is a "
            "diagnostic only and is deliberately absent from the gate checks."
        ),
        "energy_summary": {
            "maximum_relative_energy_error": (assessment.get("gates") or {}).get(
                "maximum_relative_energy_error"
            ),
            "orbits_exceeding_1e-10": (assessment.get("gates") or {}).get(
                "orbits_exceeding_energy_gate"
            ),
            "orbits_with_nonzero_energy_error": sum(
                item["diagnostics"].get("orbits_with_nonzero_energy_error", 0)
                for item in cases.values()
            ),
            "final_velocity_event_velocity_mismatches": (assessment.get("gates") or {}).get(
                "final_velocity_event_velocity_mismatches"
            ),
        },
        "magnetic_moment_variation_diagnostic": {
            key: item["diagnostics"].get("magnetic_moment_variation_diagnostic")
            for key, item in cases.items()
        },
        "probability_convergence": assessment.get("convergence"),
        "execution_mode": assessment.get("execution_mode"),
        "timing_s": {
            "runtime_total": runtime_seconds,
            "development": development.get("seconds"),
            "assessment": (assessment.get("execution_mode") or {}).get("assessment_wall_s"),
            "integration_wall": (assessment.get("execution_mode") or {}).get("integration_wall_s"),
            "export_wall": (assessment.get("execution_mode") or {}).get("export_wall_s"),
        },
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
        "informational_gate_checks": (assessment.get("gates") or {}).get("checks"),
        "maximum_relative_energy_error": (assessment.get("gates") or {}).get(
            "maximum_relative_energy_error"
        ),
        "runtime_seconds": runtime_seconds,
        "shakedown_path": str(SHAKEDOWN_PATH),
        "cases": {
            key: {
                "termination_counts": item["diagnostics"]["termination_counts"],
                "tolerance_close": item["diagnostics"]["tolerance_close_event_count"],
                "validators_passed": item["validators"]["passed"],
                "validators_failed": item["validators"]["failed"],
                "export_stage_ran": item["export_stage_ran"],
                "integration_s": item["timing_s"].get("integration"),
                "per_orbit_ms": item["timing_s"].get("per_orbit_ms"),
                "max_energy_error": item["diagnostics"]["maximum_relative_energy_error"],
                "nonzero_energy_orbits": item["diagnostics"].get(
                    "orbits_with_nonzero_energy_error"
                ),
                "velocity_identity_count": item["diagnostics"].get(
                    "final_velocity_equals_event_velocity_count"
                ),
                "mu_diagnostic": item["diagnostics"].get(
                    "magnetic_moment_variation_diagnostic"
                ),
            }
            for key, item in cases.items()
        },
    }


# --------------------------------------------------------------------------
# prepare (refuses without a valid shakedown)
# --------------------------------------------------------------------------


def shakedown_gate(value: dict[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, bool]]:
    if not SHAKEDOWN_PATH.is_file():
        raise RuntimeError(
            "prepare refused: shakedown.json is missing; run "
            f"`{SHAKEDOWN_COMMAND}` first"
        )
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
    contract = require_orbit_mc_contract(value)
    shakedown_record, shakedown_bytes, shakedown_checks = shakedown_gate(value)
    plan = evidentiary_plan(value)
    case_authorities = build_all_case_authorities(value, plan)
    for authority in case_authorities["cases"]:
        launches = build_case_launches(
            value, plan, authority["role"], authority["timestep"]
        )
        _write_bytes(
            EXPERIMENT / authority["launch_manifest_path"],
            canonical_bytes(
                runtime_launch_payload(authority["campaign_id"], launches)
            ),
        )
        _write_bytes(
            EXPERIMENT / authority["batch_manifest_path"],
            canonical_bytes(
                runtime_batch_payload(plan, authority["campaign_id"], launches)
            ),
        )
    manufactured = manufactured_gate_report(value)
    production_preflight = production_synthetic_preflight(
        value, case_authorities
    )
    synthetic = {
        "schema_version": schema("synthetic-preflight"),
        "protocol_semantic_sha256": semantic_sha256(value),
        "case_authorities_sha256": semantic_sha256(case_authorities),
        "physical_stratum_design_sha256": content_hash(value["launches"]),
        "evidentiary_design_sha256": design_sha256(value, plan),
        "case_count": case_authorities["case_count"],
        "launches_per_case": plan.launches_per_case,
        "total_case_launches": case_authorities["total_case_launches"],
        "batches_per_case": plan.batches_per_case,
        "strata_per_case": plan.strata_per_case,
        "gyrophase_count": value["launches"]["gyrophase_count"],
        "manufactured_gate_report": manufactured,
        "production_preflight": production_preflight,
        "p2_field_access_count": 0,
        "orbit_outcome_access_count": 0,
        "prior_campaign_disclosure": value["prior_campaign_disclosure"],
        "shakedown_gate_checks": shakedown_checks,
        "passed": manufactured["passed"] and production_preflight["passed"],
    }
    authorities = {
        "schema_version": schema("authorities"),
        "protocol_semantic_sha256": semantic_sha256(value),
        "case_authorities_sha256": semantic_sha256(case_authorities),
        "physical_stratum_design_sha256": content_hash(value["launches"]),
        "evidentiary_design_sha256": design_sha256(value, plan),
        "case_count": case_authorities["case_count"],
        "total_case_launches": case_authorities["total_case_launches"],
        "p2_manifest_file_sha256": value["authority"]["manifest"]["file_sha256"],
        "p2_result_file_sha256": value["authority"]["result"]["file_sha256"],
        "minimum_certificate_dense_to_bound_ratio": value["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
        "orbit_mc_source_sha256": contract["source_sha256"],
        "orbit_mc_source_hash_line_endings": "LF",
        "orbit_mc_code_identity_sha256": contract["code_identity_sha256"],
        "orbit_mc_package_version": contract["observed"]["package_version"],
        "orbit_mc_schema_versions": contract["observed"],
        "shakedown_file_sha256": hashlib.sha256(shakedown_bytes).hexdigest(),
        "shakedown_semantic_sha256": semantic_sha256(shakedown_record),
        "shakedown_design_sha256": shakedown_record["shakedown_design_sha256"],
        "shakedown_git_head": shakedown_record["git"]["head"],
    }
    _write(CASE_AUTHORITIES_PATH, case_authorities)
    _write(SYNTHETIC_PREFLIGHT_PATH, synthetic)
    _write(AUTHORITIES_PATH, authorities)
    return {
        "passed": synthetic["passed"],
        "case_count": authorities["case_count"],
        "total_case_launches": authorities["total_case_launches"],
        "protocol_semantic_sha256": authorities["protocol_semantic_sha256"],
        "case_authorities_sha256": authorities["case_authorities_sha256"],
        "evidentiary_design_sha256": authorities["evidentiary_design_sha256"],
        "orbit_mc_source_sha256": authorities["orbit_mc_source_sha256"],
        "shakedown_file_sha256": authorities["shakedown_file_sha256"],
        "shakedown_gate_checks": shakedown_checks,
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
        ("git", "merge-base", "--is-ancestor", head, REMOTE_BRANCH),
        cwd=REPOSITORY,
        capture_output=True,
    ).returncode:
        raise RuntimeError("preregistration commit is not pushed to the authorized branch")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("execution requires a clean detached worktree")
    changed = _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", head
    ).splitlines()
    prefix = "modern/experiments/cft_orbit_wall_loss_v4/"
    tests_prefix = "modern/tests/experiments/cft_orbit_wall_loss_v4/"
    if not changed or any(
        not item.startswith((prefix, tests_prefix)) for item in changed
    ):
        raise RuntimeError("preregistration commit is not experiment-path isolated")
    if any("/results/" in item for item in changed):
        raise RuntimeError("preregistration commit contains outcome artifacts")
    for path in (
        PROTOCOL_PATH,
        AUTHORITIES_PATH,
        CASE_AUTHORITIES_PATH,
        SYNTHETIC_PREFLIGHT_PATH,
        SHAKEDOWN_PATH,
    ):
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
    if (
        frozen.authorities["shakedown_file_sha256"]
        != hashlib.sha256(frozen.shakedown_bytes).hexdigest()
    ):
        raise RuntimeError("shakedown.json differs from the preregistered authority")
    try:
        verify_shakedown_record(value, frozen.shakedown)
    except ValueError as error:
        raise RuntimeError(f"execution refused: {error}") from error
    for authority in frozen.case_authorities["cases"]:
        for key in ("launch_manifest_path", "batch_manifest_path"):
            if not (EXPERIMENT / authority[key]).is_file():
                raise RuntimeError(f"missing case authority: {authority[key]}")
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
    cache_root = (
        Path(tempfile.gettempdir())
        / f"cft-orbit-wall-loss-v4-{commit[:12]}-working-cache"
    )
    runtime = ExperimentRuntime(
        experiment_id=value["experiment_id"],
        result_root=RESULTS_ROOT,
        cache_root=cache_root,
        attestation=ExecutionAttestation(
            attempt=1,
            commit=commit,
            command=COMMAND,
            device=DEVICE,
            clean_worktree=True,
        ),
        producer=campaign_producer,
        source_root=MODERN,
    )
    collector: dict[str, Any] = {}
    outcome = runtime.run(
        build_callbacks(
            value, evidentiary_plan(value), frozen=load_frozen_authority(), collector=collector
        )
    )
    manifest = validate_bundle(RESULTS_ROOT)
    return {
        "state": outcome.state.value,
        "manifest_state": manifest["state"],
        "preregistration_commit": commit,
        "git_common_lock": git_lock.name,
        "single_execution": True,
        "no_patch_or_rerun": True,
        "worker_pool_size": worker_count(value),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("shakedown", "prepare", "execute", "validate", "contract")
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "shakedown":
        output = shakedown()
    elif arguments.command == "prepare":
        output = prepare()
    elif arguments.command == "execute":
        output = execute()
    elif arguments.command == "contract":
        output = orbit_mc_contract_report(protocol())
    else:
        output = validate_bundle(RESULTS_ROOT)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
