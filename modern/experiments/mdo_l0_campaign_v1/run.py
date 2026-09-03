"""Shakedown on the real L0 model, gated preregistration, then execute exactly once.

Lifecycle (from ``modern/`` with the ``.venv-sota`` interpreter)::

    python -m experiments.mdo_l0_campaign_v1.run shakedown   # any HEAD, dirty ok, BEFORE prepare
    python -m experiments.mdo_l0_campaign_v1.run prepare     # refuses without a valid shakedown.json
    # commit + push the preregistration, then from a clean detached worktree:
    python -m experiments.mdo_l0_campaign_v1.run execute
    python -m experiments.mdo_l0_campaign_v1.run validate
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
    EXPERIMENT,
    MODERN,
    PROTOCOL_PATH,
    REPOSITORY,
    RESULTS_ROOT,
    SHAKEDOWN_PATH,
    build_callbacks,
    code_contract_report,
    design_sha256,
    evidentiary_plan,
    load_frozen_authority,
    plan_record,
    protocol,
    require_code_contract,
    require_protocol_consistency,
    schema,
    shakedown_disjointness,
    shakedown_plan,
    verify_shakedown_record,
)

SUBJECT = "preregister MDO L0 campaign v1"
REMOTE_BRANCH = "origin/exp/mdo-l0-campaign-v1"
COMMAND = "python -m experiments.mdo_l0_campaign_v1.run execute"
SHAKEDOWN_COMMAND = "python -m experiments.mdo_l0_campaign_v1.run shakedown"
DEVICE = "cpu-l0-python-reference+cpu-gp-float64"
FROZEN_OUTPUTS = (AUTHORITIES_PATH,)


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


def shakedown() -> dict[str, Any]:
    """Drive the complete production path with a small budget and disjoint seeds."""

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
    require_protocol_consistency(value)
    contract = require_code_contract(value)
    plan = shakedown_plan(value)
    evidentiary = evidentiary_plan(value)
    disjointness = shakedown_disjointness(value)
    if not disjointness["proven"]:
        raise RuntimeError("shakedown design is not disjoint from the evidentiary design")
    git = _git_state()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    temp_root = (
        Path(tempfile.gettempdir())
        / f"mdo-l0-campaign-v1-shakedown-{git['head'][:12]}-{stamp}-{token}"
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
    assessment = collector.get("assessment", {})
    runs = assessment.get("runs", {})
    gates = assessment.get("gates", {})
    passed = bool(
        outcome.state.value == "accepted_result"
        and bundle_validated
        and disjointness["proven"]
        and set(runs) == {run_id.split(":", 1)[1] for run_id in plan.run_ids}
        and all(item["evaluations"] == item["budget"] for item in runs.values())
        and gates.get("all_binding_passed") is True
    )
    record = {
        "schema_version": schema("shakedown"),
        "evidentiary": False,
        "outcomes_enter_estimand": False,
        "disclosure": (
            "NON-EVIDENTIARY shakedown of the complete production path on the real L0 "
            "model with a small budget and a seed namespace disjoint from the "
            "evidentiary campaign. Its outcomes never enter any estimand; it exists only "
            "to prove the code path before the freeze."
        ),
        "attestation_note": (
            "ExecutionAttestation.clean_worktree is a runtime contract constant; the "
            "actual git dirtiness at shakedown time is recorded under git.dirty_entries"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "git": git,
        "protocol_semantic_sha256": semantic_sha256(value),
        "source_sha256": contract["source_sha256"],
        "source_files": contract["source_files"],
        "package_versions": contract["observed_package_versions"],
        "python": contract["python"],
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
        "development": collector.get("development"),
        "runs": runs,
        "gates": gates,
        "campaign_result": assessment.get("campaign_result"),
        "timing_s": {
            "runtime_total": runtime_seconds,
            "development": (collector.get("development") or {}).get("seconds"),
            "assessment": assessment.get("seconds"),
        },
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
        "runs": {
            key: {
                "evaluations": item["evaluations"],
                "final_hypervolume": item["final_hypervolume"],
                "pareto_set_size": item["pareto_set_size"],
                "infeasible": item["infeasible_evaluations"],
                "wall_clock_seconds": item["wall_clock_seconds"],
            }
            for key, item in runs.items()
        },
        "binding_gates": {
            name: item["passed"] for name, item in (gates.get("binding") or {}).items()
        },
        "reported": {
            "bo_beats_random": (gates.get("reported_not_binding") or {}).get("bo_beats_random", {}).get("passed"),
            "bo_beats_nsga3": (gates.get("reported_not_binding") or {}).get("bo_beats_nsga3", {}).get("passed"),
            "design_set_invariance": (gates.get("reported_not_binding") or {}).get("design_set_invariance", {}).get("passed"),
        },
        "runtime_seconds": runtime_seconds,
        "shakedown_path": str(SHAKEDOWN_PATH),
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
    require_protocol_consistency(value)
    contract = require_code_contract(value)
    shakedown_record, shakedown_bytes, shakedown_checks = shakedown_gate(value)
    plan = evidentiary_plan(value)
    authorities = {
        "schema_version": schema("authorities"),
        "protocol_semantic_sha256": semantic_sha256(value),
        "evidentiary_design_sha256": design_sha256(value, plan),
        "evidentiary_plan": plan_record(plan),
        "sample_sha256": value["uncertain_inputs"]["sample"]["sha256"],
        "source_sha256": contract["source_sha256"],
        "source_files": contract["source_files"],
        "source_hash_line_endings": "LF",
        "package_versions": contract["observed_package_versions"],
        "python": contract["python"],
        "shakedown_file_sha256": hashlib.sha256(shakedown_bytes).hexdigest(),
        "shakedown_semantic_sha256": semantic_sha256(shakedown_record),
        "shakedown_design_sha256": shakedown_record["shakedown_design_sha256"],
        "shakedown_git_head": shakedown_record["git"]["head"],
        "shakedown_gate_checks": shakedown_checks,
    }
    _write(AUTHORITIES_PATH, authorities)
    return {
        "passed": True,
        "protocol_semantic_sha256": authorities["protocol_semantic_sha256"],
        "evidentiary_design_sha256": authorities["evidentiary_design_sha256"],
        "source_sha256": authorities["source_sha256"],
        "shakedown_file_sha256": authorities["shakedown_file_sha256"],
        "shakedown_gate_checks": shakedown_checks,
        "run_count": len(plan.run_ids),
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
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
    prefix = "modern/experiments/mdo_l0_campaign_v1/"
    tests_prefix = "modern/tests/experiments/mdo_l0_campaign_v1/"
    if not changed or any(not item.startswith((prefix, tests_prefix)) for item in changed):
        raise RuntimeError("preregistration commit is not experiment-path isolated")
    if any("/results/" in item for item in changed):
        raise RuntimeError("preregistration commit contains outcome artifacts")
    for path in (PROTOCOL_PATH, AUTHORITIES_PATH, SHAKEDOWN_PATH):
        if not path.is_file():
            raise RuntimeError(f"missing preregistered authority: {path.name}")
    value = protocol()
    require_protocol_consistency(value)
    contract = require_code_contract(value)
    frozen = load_frozen_authority()
    if frozen.authorities["protocol_semantic_sha256"] != semantic_sha256(value):
        raise RuntimeError("protocol differs from the preregistered authority")
    if frozen.authorities["source_sha256"] != contract["source_sha256"]:
        raise RuntimeError("source hash differs from the preregistered authority")
    if frozen.authorities["package_versions"] != contract["observed_package_versions"]:
        raise RuntimeError("package versions differ from the preregistered authority")
    if (
        frozen.authorities["shakedown_file_sha256"]
        != hashlib.sha256(frozen.shakedown_bytes).hexdigest()
    ):
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
    cache_root = Path(tempfile.gettempdir()) / f"mdo-l0-campaign-v1-{commit[:12]}-working-cache"
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
        "primary_error": outcome.primary_error,
        "manifest_state": manifest["state"],
        "preregistration_commit": commit,
        "git_common_lock": git_lock.name,
        "single_execution": True,
        "no_patch_or_rerun": True,
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
        output = code_contract_report(protocol())
    else:
        output = validate_bundle(RESULTS_ROOT)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
