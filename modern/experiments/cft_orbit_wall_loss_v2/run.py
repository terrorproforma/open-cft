"""Prepare preregistration authority, then execute exactly once when detached."""

from __future__ import annotations

import argparse
import hashlib
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
from cft_revival.orbit_mc import EstimatorPolicy
from cft_revival.orbit_mc.artifacts import content_hash

from .experiment import (
    AUTHORITIES_PATH,
    BATCH_MANIFEST_PATH,
    EXPERIMENT,
    LAUNCH_MANIFEST_PATH,
    MODERN,
    PROTOCOL_PATH,
    REPOSITORY,
    SYNTHETIC_PREFLIGHT_PATH,
    batch_records,
    build_launches,
    callbacks,
    estimator_identity,
    launch_records,
    manufactured_gate_report,
    protocol,
    runtime_batch_payload,
    runtime_launch_payload,
    synthetic_serialization_audit,
)

SUBJECT = "preregister CFT full-orbit wall-loss v2"
REMOTE_BRANCH = "origin/exp/cft-orbit-wall-loss-v2"
COMMAND = "python -m experiments.cft_orbit_wall_loss_v2.run execute"


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
    launches = build_launches(value)
    batches = batch_records(value, launches)
    launch_payload = runtime_launch_payload(launches)
    launch_bytes = canonical_bytes(launch_payload)
    batch_payload = runtime_batch_payload(value, launches)
    batch_bytes = canonical_bytes(batch_payload)
    orbit_launches_sha256 = content_hash(launch_records(launches))
    batch_manifest_sha256 = content_hash(
        {
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "batches": batches,
        }
    )
    manufactured = manufactured_gate_report(value)
    serialization_audit = synthetic_serialization_audit(value, launches, batches)
    synthetic = {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v2.synthetic-preflight/2.0.0",
        "protocol_semantic_sha256": semantic_sha256(value),
        "runtime_launch_payload_byte_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "runtime_batch_payload_byte_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "orbit_launches_sha256": orbit_launches_sha256,
        "batch_manifest_sha256": batch_manifest_sha256,
        "estimator_sha256": estimator_identity(launches, batches),
        "launch_count": len(launches),
        "batch_count": len(batches),
        "gyrophase_count": value["launches"]["gyrophase_count"],
        "both_parallel_directions": sorted(
            {item.parallel_direction for item in launches}
        )
        == [-1, 1],
        "equal_weights": all(
            entry["weight"] == 1.0 / len(launches)
            for batch in batches
            for entry in batch["launches"]
        ),
        "manufactured_gate_report": manufactured,
        "serialization_audit": serialization_audit,
        "p2_field_access_count": 0,
        "orbit_outcome_access_count": 0,
        "v1_production_access_disclosure": {
            "v1_terminal_state": "prebundle_failure",
            "v1_p2_field_access_count": 0,
            "v1_orbit_outcome_access_count": 0,
            "launch_grid_reused": True,
            "campaign_ids_and_seed_ids_recomputed": True,
        },
        "passed": manufactured["passed"] and serialization_audit["passed"],
    }
    authorities = {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v2.authorities/2.0.0",
        "protocol_semantic_sha256": semantic_sha256(value),
        "runtime_launch_payload_byte_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "runtime_batch_payload_byte_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "orbit_launches_sha256": orbit_launches_sha256,
        "batch_manifest_sha256": batch_manifest_sha256,
        "estimator_sha256": estimator_identity(launches, batches),
        "p2_manifest_file_sha256": value["authority"]["manifest"]["file_sha256"],
        "p2_result_file_sha256": value["authority"]["result"]["file_sha256"],
        "minimum_certificate_dense_to_bound_ratio": value["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
    }
    _write_bytes(LAUNCH_MANIFEST_PATH, launch_bytes)
    _write_bytes(BATCH_MANIFEST_PATH, batch_bytes)
    _write(SYNTHETIC_PREFLIGHT_PATH, synthetic)
    _write(AUTHORITIES_PATH, authorities)
    return {
        "passed": synthetic["passed"],
        "launch_count": len(launches),
        "batch_count": len(batches),
        "protocol_semantic_sha256": authorities["protocol_semantic_sha256"],
        "runtime_launch_payload_byte_sha256": authorities[
            "runtime_launch_payload_byte_sha256"
        ],
        "orbit_launches_sha256": authorities["orbit_launches_sha256"],
        "batch_manifest_sha256": authorities["batch_manifest_sha256"],
        "estimator_sha256": authorities["estimator_sha256"],
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
    prefix = "modern/experiments/cft_orbit_wall_loss_v2/"
    if not changed or any(not item.startswith(prefix) for item in changed):
        raise RuntimeError("preregistration commit is not experiment-path isolated")
    if any("/results/" in item for item in changed):
        raise RuntimeError("preregistration commit contains outcome artifacts")
    for path in (
        PROTOCOL_PATH,
        AUTHORITIES_PATH,
        LAUNCH_MANIFEST_PATH,
        BATCH_MANIFEST_PATH,
        SYNTHETIC_PREFLIGHT_PATH,
    ):
        if not path.is_file():
            raise RuntimeError(f"missing preregistered authority: {path.name}")
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
        / f"cft-orbit-wall-loss-v2-{commit[:12]}-working-cache"
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
