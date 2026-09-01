"""Git-backed preregistration identity binding for L0 surrogate v4."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cft_revival.surrogates.identity import canonical_hash

PREREGISTRATION_SUBJECT = "preregister L0 surrogate v4 experiment"
SOURCE_PREFIX = "modern/experiments/l0_surrogate_v4/"
TEST_PREFIX = "modern/tests/experiments/l0_surrogate_v4/"
PROTOCOL_PREFIXES = (SOURCE_PREFIX, TEST_PREFIX)
REMOTE_REF = "origin/feat/sota-foundation"


class CommitBindingError(RuntimeError):
    """The live checkout is not the pushed preregistered protocol."""


@dataclass(frozen=True, slots=True)
class CommitBinding:
    observed_head_sha: str
    protocol_commit_sha: str
    remote_ref: str
    protocol_tree_hash: str
    protocol_paths: tuple[str, ...]
    intervening_unrelated_commits_allowed: bool = True
    intervening_protocol_changes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_head_sha": self.observed_head_sha,
            "protocol_commit_sha": self.protocol_commit_sha,
            "remote_ref": self.remote_ref,
            "protocol_tree_hash": self.protocol_tree_hash,
            "protocol_paths": list(self.protocol_paths),
            "intervening_unrelated_commits_allowed": (
                self.intervening_unrelated_commits_allowed
            ),
            "intervening_protocol_changes": self.intervening_protocol_changes,
        }


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CommitBindingError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _exists_as_commit(repository: Path, sha: str) -> None:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise CommitBindingError(f"{sha!r} does not identify an existing commit")


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if completed.returncode not in (0, 1):
        raise CommitBindingError("git merge-base ancestry check failed")
    return completed.returncode == 0


def bind_execution_identity(
    repository: Path,
    *,
    expected_head_sha: str | None = None,
    remote_ref: str = REMOTE_REF,
) -> CommitBinding:
    """Bind execution to the pushed commit and unchanged v4 protocol blobs.

    Unrelated commits after preregistration are allowed only when they are
    already present on ``remote_ref`` and do not alter any v4 protocol path.
    """

    repository = repository.resolve()
    observed = _git(repository, ("rev-parse", "HEAD"))
    _exists_as_commit(repository, observed)
    if expected_head_sha is not None:
        _exists_as_commit(repository, expected_head_sha)
        if expected_head_sha != observed:
            raise CommitBindingError(
                "caller-supplied expected HEAD does not equal git rev-parse HEAD"
            )
    _exists_as_commit(repository, remote_ref)
    if not _is_ancestor(repository, observed, remote_ref):
        raise CommitBindingError("observed HEAD is not present on the required remote branch")

    matches = tuple(
        line
        for line in _git(
            repository,
            (
                "log",
                "--format=%H",
                f"--grep=^{PREREGISTRATION_SUBJECT}$",
                observed,
            ),
        ).splitlines()
        if line
    )
    if len(matches) != 1:
        raise CommitBindingError("expected exactly one v4 preregistration commit in ancestry")
    protocol_commit = matches[0]
    _exists_as_commit(repository, protocol_commit)
    if not _is_ancestor(repository, protocol_commit, observed):
        raise CommitBindingError("v4 preregistration commit is not an ancestor of HEAD")

    changed = tuple(
        line
        for line in _git(
            repository,
            (
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                protocol_commit,
            ),
        ).splitlines()
        if line
    )
    if not changed or any(
        not path.startswith(PROTOCOL_PREFIXES) for path in changed
    ):
        raise CommitBindingError("preregistration commit is not exact-path isolated")
    if any("/results/" in path or path.endswith("/RESULTS_REPORT.md") for path in changed):
        raise CommitBindingError("preregistration commit already contains result artifacts")

    intervening = tuple(
        line
        for line in _git(
            repository,
            (
                "log",
                "--format=%H",
                f"{protocol_commit}..{observed}",
                "--",
                *PROTOCOL_PREFIXES,
            ),
        ).splitlines()
        if line
    )
    if intervening:
        raise CommitBindingError("an intervening commit changed the v4 protocol")

    tracked = tuple(
        sorted(
            line
            for line in _git(
                repository,
                ("ls-tree", "-r", "--name-only", protocol_commit, "--", *PROTOCOL_PREFIXES),
            ).splitlines()
            if line
        )
    )
    if not tracked or set(changed) != set(tracked):
        raise CommitBindingError(
            "v4 preregistration commit does not introduce the complete protocol tree"
        )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", observed, "--", *PROTOCOL_PREFIXES],
        cwd=repository,
        check=False,
    )
    if dirty.returncode != 0:
        raise CommitBindingError("working v4 protocol files differ from committed blobs")
    untracked = _git(
        repository,
        ("ls-files", "--others", "--exclude-standard", "--", *PROTOCOL_PREFIXES),
    )
    if untracked:
        raise CommitBindingError("untracked files exist inside the v4 protocol paths")

    entries = []
    for line in _git(
        repository,
        ("ls-tree", "-r", protocol_commit, "--", *PROTOCOL_PREFIXES),
    ).splitlines():
        metadata, path = line.split("\t", 1)
        mode, object_type, blob_sha = metadata.split()
        entries.append(
            {
                "path": path,
                "mode": mode,
                "type": object_type,
                "blob_sha": blob_sha,
            }
        )
    return CommitBinding(
        observed_head_sha=observed,
        protocol_commit_sha=protocol_commit,
        remote_ref=remote_ref,
        protocol_tree_hash=canonical_hash(entries),
        protocol_paths=tracked,
    )
