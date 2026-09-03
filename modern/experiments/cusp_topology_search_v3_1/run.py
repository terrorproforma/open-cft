"""Shakedown on real designs of every set, gated preregistration, then execute exactly once.

Lifecycle (from ``modern/``)::

    python -m experiments.cusp_topology_search_v3_1.run shakedown  # any HEAD, BEFORE prepare
    python -m experiments.cusp_topology_search_v3_1.run prepare    # refuses without a valid shakedown.json
    # commit + push the preregistration, then from a clean detached worktree:
    python -m experiments.cusp_topology_search_v3_1.run execute
    python -m experiments.cusp_topology_search_v3_1.run validate

Ported from ``experiments.orbit_wall_loss_geometry_screening_v1.run`` (accepted template).
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
    build_callbacks,
    build_design_authorities,
    evidentiary_plan,
    experiment_code_sha256,
    load_frozen_authority,
    plan_record,
    protocol,
    schema,
    shakedown_plan,
    source_binding_report,
    specs_for_plan,
    verify_shakedown_record,
    worker_count,
)
from .fields import DESIGN_SETS

SUBJECT = "preregister cusp topology search v3.1"
REMOTE_BRANCH = "origin/exp/cusp-topology-search-v3"
COMMAND = "python -m experiments.cusp_topology_search_v3_1.run execute"
SHAKEDOWN_COMMAND = "python -m experiments.cusp_topology_search_v3_1.run shakedown"
DEVICE = "cpu-only;cft_revival.fields.solve_problem_cpu;PsiBicubicField-tracing;gpu-not-used"
FROZEN_OUTPUTS = (AUTHORITIES_PATH, DESIGN_AUTHORITIES_PATH)
EXPERIMENT_PREFIX = "modern/experiments/cusp_topology_search_v3_1/"
TESTS_PREFIX = "modern/tests/experiments/cusp_topology_search_v3_1_1/"
CONTENTION_FACTOR = 1.5
BUDGET_WALL_SECONDS = 5400.0


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(("git", *arguments), cwd=REPOSITORY, check=check, capture_output=True, text=True, encoding="utf-8")
    return completed.stdout.strip()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _git_state() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    branch = _git("symbolic-ref", "-q", "--short", "HEAD", check=False)
    dirty = [line for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line.strip()]
    return {"head": head, "branch": branch or None, "detached": not branch, "dirty": bool(dirty), "dirty_entry_count": len(dirty), "dirty_entries": dirty[:200]}


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
        records.append({"sequence": row["sequence"], "operation": row["operation"], "kind": row["kind"], "details": row["details"], "recorded_at_utc": row["recorded_at_utc"]["value"]})
    return records


def _timing_projection(value: dict[str, Any], collector: dict[str, Any]) -> dict[str, Any]:
    per_design = collector.get("development", {}).get("per_design_seconds", {})
    by_set: dict[str, list[float]] = {}
    for key, timing in per_design.items():
        by_set.setdefault(key.split(":")[0], []).append(float(timing["total"]))
    evidentiary_counts = {set_id: int(value["design_sets"][set_id]["design_count"]) for set_id in DESIGN_SETS if value["design_sets"][set_id]["included"]}
    mean_by_set = {set_id: (sum(items) / len(items)) for set_id, items in by_set.items() if items}
    projected_cpu = sum(mean_by_set.get(set_id, max(mean_by_set.values(), default=0.0)) * count for set_id, count in evidentiary_counts.items())
    pool = worker_count(value)
    projected_wall = projected_cpu / max(1, pool) * CONTENTION_FACTOR
    return {
        "basis": "mean shakedown seconds per design per set (solve N + solve 2N + two characterizations) times the evidentiary design counts, divided by the worker pool, times a contention factor",
        "mean_seconds_per_design_by_set": mean_by_set,
        "evidentiary_design_counts": evidentiary_counts,
        "projected_cpu_seconds": projected_cpu,
        "worker_pool_size": pool,
        "contention_factor": CONTENTION_FACTOR,
        "projected_wall_seconds_at_pool": projected_wall,
        "budget_wall_seconds": BUDGET_WALL_SECONDS,
        "within_budget": projected_wall <= BUDGET_WALL_SECONDS,
    }


# --------------------------------------------------------------------------
# shakedown
# --------------------------------------------------------------------------


def shakedown() -> dict[str, Any]:
    existing = [path.name for path in FROZEN_OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError("shakedown is allowed only BEFORE prepare; frozen outputs exist: " + ", ".join(existing))
    if RESULTS_ROOT.exists():
        raise RuntimeError("shakedown refused: the evidentiary results root already exists")
    value = protocol()
    binding = source_binding_report(value)
    plan = shakedown_plan(value)
    git = _git_state()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    temp_root = Path(tempfile.gettempdir()) / f"cts-v31-shakedown-{git['head'][:12]}-{stamp}-{token}"
    result_root = temp_root / "bundle"
    cache_root = temp_root / "cache"
    runtime = ExperimentRuntime(
        experiment_id=f"{value['experiment_id']}-shakedown",
        result_root=result_root,
        cache_root=cache_root,
        attestation=ExecutionAttestation(attempt=1, commit=git["head"], command=SHAKEDOWN_COMMAND, device=f"{DEVICE};shakedown-non-evidentiary", clean_worktree=True),
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
    development = collector.get("development", {})
    assessment = collector.get("assessment", {})
    resolved_count = development.get("resolved_design_count", 0)
    passed = bool(
        outcome.state.value == "accepted_result"
        and bundle_validated
        and resolved_count == len(plan.design_keys)
        and not development.get("failures")
        and assessment.get("status") == "shakedown_passed"
        and all(assessment.get("gates", {}).values())
    )
    projection = _timing_projection(value, collector)
    record = {
        "schema_version": schema("shakedown"),
        "classification": CLASSIFICATION,
        "evidentiary": False,
        "outcomes_enter_estimand": False,
        "disclosure": (
            "NON-EVIDENTIARY shakedown of the complete production path on REAL designs of every set "
            "(three sweep-v2 designs, the two stored four-cell-v2 representatives, two characterization-v1 "
            "representatives and the single P2 row). Its outcomes never enter any estimand; it exists only to prove "
            "the code path before the freeze. Because the topology is a deterministic function of the fields, the "
            "shakedown designs' results are necessarily known before the freeze; the definition and every tolerance "
            "were fixed before the shakedown and are not tuned on it."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "git": git,
        "protocol_semantic_sha256": binding["protocol_semantic_sha256"],
        "experiment_code_sha256": binding["experiment_code_sha256"],
        "experiment_code_files": binding["experiment_code_files"],
        "dependency_source_sha256": binding["dependency_source_sha256"],
        "dependency_source_files": binding["dependency_source_files"],
        "field_pipeline_source_sha256": binding["field_pipeline_source_sha256"],
        "sealed_sources": binding["sealed_sources"],
        "shakedown_plan": plan_record(plan),
        "design_count": len(plan.design_keys),
        "resolved_design_count": resolved_count,
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
        "development": {key: development.get(key) for key in ("resolved_design_count", "failures", "stage_wall_s", "seconds", "accepted", "per_design_seconds")},
        "informational_gates": assessment.get("gates"),
        "failing_designs": assessment.get("failing_designs"),
        "replays": assessment.get("replays"),
        "headline_informational": assessment.get("headline"),
        "timing_projection": projection,
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
        "design_count": len(plan.design_keys),
        "resolved_design_count": resolved_count,
        "failures": development.get("failures"),
        "gates": assessment.get("gates"),
        "failing_designs": assessment.get("failing_designs"),
        "headline_informational": assessment.get("headline"),
        "timing_projection": projection,
        "runtime_seconds": runtime_seconds,
        "shakedown_path": str(SHAKEDOWN_PATH),
    }


# --------------------------------------------------------------------------
# prepare
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
    binding = source_binding_report(value)
    shakedown_record, shakedown_bytes, shakedown_checks = shakedown_gate(value)
    plan = evidentiary_plan(value)
    design_authorities = build_design_authorities(value, plan)
    authorities = {
        "schema_version": schema("authorities"),
        "classification": CLASSIFICATION,
        "protocol_semantic_sha256": binding["protocol_semantic_sha256"],
        "design_authorities_sha256": semantic_sha256(design_authorities),
        "design_count": design_authorities["design_count"],
        "set_counts": design_authorities["set_counts"],
        "experiment_code_sha256": binding["experiment_code_sha256"],
        "dependency_source_sha256": binding["dependency_source_sha256"],
        "field_pipeline_source_sha256": binding["field_pipeline_source_sha256"],
        "sealed_sources": binding["sealed_sources"],
        "shakedown_file_sha256": hashlib.sha256(shakedown_bytes).hexdigest(),
        "shakedown_semantic_sha256": semantic_sha256(shakedown_record),
        "shakedown_git_head": shakedown_record["git"]["head"],
        "shakedown_timing_projection": shakedown_record["timing_projection"],
        "shakedown_gate_checks": shakedown_checks,
    }
    _write(DESIGN_AUTHORITIES_PATH, design_authorities)
    _write(AUTHORITIES_PATH, authorities)
    return {
        "passed": all(shakedown_checks.values()),
        "design_count": authorities["design_count"],
        "set_counts": authorities["set_counts"],
        "protocol_semantic_sha256": authorities["protocol_semantic_sha256"],
        "design_authorities_sha256": authorities["design_authorities_sha256"],
        "experiment_code_sha256": authorities["experiment_code_sha256"],
        "dependency_source_sha256": authorities["dependency_source_sha256"],
        "field_pipeline_source_sha256": authorities["field_pipeline_source_sha256"],
        "shakedown_file_sha256": authorities["shakedown_file_sha256"],
        "shakedown_gate_checks": shakedown_checks,
        "shakedown_timing_projection": authorities["shakedown_timing_projection"],
    }


# --------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------


def _bind_preregistration() -> str:
    head = _git("rev-parse", "HEAD")
    if _git("symbolic-ref", "-q", "HEAD", check=False):
        raise RuntimeError("execution requires detached HEAD")
    if _git("show", "-s", "--format=%s", head) != SUBJECT:
        raise RuntimeError("HEAD is not the exact preregistration commit")
    if subprocess.run(("git", "merge-base", "--is-ancestor", head, REMOTE_BRANCH), cwd=REPOSITORY, capture_output=True).returncode:
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
    binding = source_binding_report(value)
    frozen = load_frozen_authority()
    for key in ("protocol_semantic_sha256", "experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256", "sealed_sources"):
        if frozen.authorities[key] != binding[key]:
            raise RuntimeError(f"{key} differs from the preregistered authority")
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
    cache_root = Path(tempfile.gettempdir()) / f"cts-v31-{commit[:12]}-working-cache"
    runtime = ExperimentRuntime(
        experiment_id=value["experiment_id"],
        result_root=RESULTS_ROOT,
        cache_root=cache_root,
        attestation=ExecutionAttestation(attempt=1, commit=commit, command=COMMAND, device=DEVICE, clean_worktree=True),
        producer=campaign_producer,
        source_root=MODERN,
    )
    collector: dict[str, Any] = {}
    outcome = runtime.run(build_callbacks(value, evidentiary_plan(value), frozen=load_frozen_authority(), collector=collector))
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
        "gates": (collector.get("assessment") or {}).get("gates"),
        "failing_designs": (collector.get("assessment") or {}).get("failing_designs"),
        "status": (collector.get("assessment") or {}).get("status"),
        "development": {key: (collector.get("development") or {}).get(key) for key in ("resolved_design_count", "failures", "stage_wall_s")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("shakedown", "prepare", "execute", "validate", "contract", "plan"))
    arguments = parser.parse_args(argv)
    if arguments.command == "shakedown":
        output = shakedown()
    elif arguments.command == "prepare":
        output = prepare()
    elif arguments.command == "execute":
        output = execute()
    elif arguments.command == "contract":
        output = source_binding_report(protocol())
    elif arguments.command == "plan":
        value = protocol()
        plan = evidentiary_plan(value)
        output = {"evidentiary": plan_record(plan), "shakedown": plan_record(shakedown_plan(value)), "design_count": len(specs_for_plan(value, plan))}
    else:
        output = validate_bundle(RESULTS_ROOT)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
