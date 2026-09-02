"""Prepare preregistration authority, then execute exactly once when detached."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cft_revival.experiment_runtime import (
    ExecutionAttestation,
    ExperimentRuntime,
    canonical_bytes,
    semantic_sha256,
    validate_bundle,
)
from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.orbit_mc.artifacts import content_hash

from .experiment import (
    AUTHORITIES_PATH,
    CASE_AUTHORITIES_PATH,
    EXPERIMENT,
    MODERN,
    PROTOCOL_PATH,
    REPOSITORY,
    SYNTHETIC_PREFLIGHT_PATH,
    build_all_case_authorities,
    build_case_launches,
    callbacks,
    manufactured_gate_report,
    production_synthetic_preflight,
    protocol,
    runtime_batch_payload,
    runtime_launch_payload,
)

SUBJECT = "preregister CFT full-orbit wall-loss v3"
REMOTE_BRANCH = "origin/exp/cft-orbit-wall-loss-v3"
COMMAND = "python -m experiments.cft_orbit_wall_loss_v3.run execute"


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


def prepare() -> dict[str, Any]:
    value = protocol()
    case_authorities = build_all_case_authorities(value)
    for authority in case_authorities["cases"]:
        launches = build_case_launches(
            value, authority["role"], authority["timestep"]
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
                runtime_batch_payload(
                    value, authority["campaign_id"], launches
                )
            ),
        )
    manufactured = manufactured_gate_report(value)
    production_preflight = production_synthetic_preflight(
        value, case_authorities
    )
    synthetic = {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v3.synthetic-preflight/3.0.0",
        "protocol_semantic_sha256": semantic_sha256(value),
        "case_authorities_sha256": semantic_sha256(case_authorities),
        "physical_stratum_design_sha256": content_hash(value["launches"]),
        "case_count": case_authorities["case_count"],
        "launches_per_case": 512,
        "total_case_launches": case_authorities["total_case_launches"],
        "batches_per_case": 8,
        "strata_per_case": 32,
        "gyrophase_count": value["launches"]["gyrophase_count"],
        "manufactured_gate_report": manufactured,
        "production_preflight": production_preflight,
        "p2_field_access_count": 0,
        "orbit_outcome_access_count": 0,
        "prior_campaign_disclosure": value["prior_campaign_disclosure"],
        "passed": manufactured["passed"] and production_preflight["passed"],
    }
    authorities = {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v3.authorities/3.0.0",
        "protocol_semantic_sha256": semantic_sha256(value),
        "case_authorities_sha256": semantic_sha256(case_authorities),
        "physical_stratum_design_sha256": content_hash(value["launches"]),
        "case_count": case_authorities["case_count"],
        "total_case_launches": case_authorities["total_case_launches"],
        "p2_manifest_file_sha256": value["authority"]["manifest"]["file_sha256"],
        "p2_result_file_sha256": value["authority"]["result"]["file_sha256"],
        "minimum_certificate_dense_to_bound_ratio": value["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
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
        "physical_stratum_design_sha256": authorities[
            "physical_stratum_design_sha256"
        ],
    }


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
    prefix = "modern/experiments/cft_orbit_wall_loss_v3/"
    if not changed or any(not item.startswith(prefix) for item in changed):
        raise RuntimeError("preregistration commit is not experiment-path isolated")
    if any("/results/" in item for item in changed):
        raise RuntimeError("preregistration commit contains outcome artifacts")
    for path in (
        PROTOCOL_PATH,
        AUTHORITIES_PATH,
        CASE_AUTHORITIES_PATH,
        SYNTHETIC_PREFLIGHT_PATH,
    ):
        if not path.is_file():
            raise RuntimeError(f"missing preregistered authority: {path.name}")
    case_authorities = strict_json_file(CASE_AUTHORITIES_PATH)
    for authority in case_authorities["cases"]:
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


def campaign_producer() -> None:
    """Stable producer identity for the shared lifecycle runtime."""


def execute() -> dict[str, Any]:
    commit = _bind_preregistration()
    git_lock = _acquire_git_common_lock(commit)
    result_root = EXPERIMENT / "results"
    cache_root = (
        Path(tempfile.gettempdir())
        / f"cft-orbit-wall-loss-v3-{commit[:12]}-working-cache"
    )
    runtime = ExperimentRuntime(
        experiment_id=protocol()["experiment_id"],
        result_root=result_root,
        cache_root=cache_root,
        attestation=ExecutionAttestation(
            attempt=1,
            commit=commit,
            command=COMMAND,
            device="numpy-cpu-reference+warp-cpu-cuda-parity",
            clean_worktree=True,
        ),
        producer=campaign_producer,
        source_root=MODERN,
    )
    outcome = runtime.run(callbacks())
    manifest = validate_bundle(result_root)
    return {
        "state": outcome.state.value,
        "manifest_state": manifest["state"],
        "preregistration_commit": commit,
        "git_common_lock": git_lock.name,
        "single_execution": True,
        "no_patch_or_rerun": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute", "validate"))
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        output = prepare()
    elif arguments.command == "execute":
        output = execute()
    else:
        output = validate_bundle(EXPERIMENT / "results")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
