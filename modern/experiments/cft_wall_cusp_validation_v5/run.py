"""Sole launcher for the clean-detached v5 validation attempt."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cft_revival.experiment_runtime import ExecutionAttestation

from .experiment import EXPERIMENT_DIR, PROTOCOL, REPOSITORY_ROOT, execute_validation


COMMAND = "python -m experiments.cft_wall_cusp_validation_v5.run"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _verified_detached_commit() -> str:
    symbolic = subprocess.run(
        ("git", "symbolic-ref", "-q", "--short", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if symbolic.returncode == 0:
        raise RuntimeError("validation execution requires detached HEAD")
    commit = _git("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("unexpected Git commit identity")
    return commit


def _verify_clean() -> None:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("validation execution requires a clean worktree")


def main() -> None:
    _verify_clean()
    commit = _verified_detached_commit()
    result_root = EXPERIMENT_DIR / "results"
    cache_root = EXPERIMENT_DIR / ".working-cache"
    outcome = execute_validation(
        result_root=result_root,
        cache_root=cache_root,
        attestation=ExecutionAttestation(
            attempt=1,
            commit=commit,
            command=COMMAND,
            device=str(PROTOCOL["maps"]["solver"]["device"]),
            clean_worktree=True,
        ),
    )
    print(
        f"state={outcome.state.value} "
        f"manifest_artifacts={outcome.manifest['artifact_count']}"
    )


if __name__ == "__main__":
    main()
