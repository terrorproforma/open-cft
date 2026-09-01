"""Detached-worktree Git and transitive-dependency identity for v5."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cft_revival.surrogates.identity import canonical_hash

SUBJECT = "preregister L0 surrogate v5 experiment"
PREFIXES = (
    "modern/experiments/l0_surrogate_v5/",
    "modern/tests/experiments/l0_surrogate_v5/",
)
REMOTE = "origin/feat/sota-foundation"


class IdentityError(RuntimeError):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and result.returncode:
        raise IdentityError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _ancestor(repo: Path, left: str, right: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", left, right],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode == 0


@dataclass(frozen=True)
class Binding:
    commit_sha: str
    remote_ref: str
    protocol_tree_hash: str
    dependency_tree_hash: str
    detached_head: bool
    clean_protocol: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "commit_sha": self.commit_sha,
            "remote_ref": self.remote_ref,
            "protocol_tree_hash": self.protocol_tree_hash,
            "dependency_tree_hash": self.dependency_tree_hash,
            "detached_head": self.detached_head,
            "clean_protocol": self.clean_protocol,
        }


def bind(repo: Path, dependency_manifest: Path) -> Binding:
    repo = repo.resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{head}^{{commit}}"],
        cwd=repo,
        check=False,
    ).returncode:
        raise IdentityError("HEAD is not an existing commit")
    if _git(repo, "symbolic-ref", "-q", "HEAD", check=False):
        raise IdentityError("v5 must execute from a detached worktree")
    if not _ancestor(repo, head, REMOTE):
        raise IdentityError("detached commit is not on the required remote branch")
    if _git(repo, "show", "-s", "--format=%s", head) != SUBJECT:
        raise IdentityError("detached HEAD is not the v5 preregistration commit")
    changed = tuple(
        line
        for line in _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head
        ).splitlines()
        if line
    )
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise IdentityError("v5 preregistration commit is not exact-path isolated")
    if any("/results/" in path or path.endswith("RESULTS_REPORT.md") for path in changed):
        raise IdentityError("v5 preregistration commit contains results")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *PREFIXES],
        cwd=repo,
        check=False,
    ).returncode:
        raise IdentityError("working v5 protocol differs from detached commit")
    if _git(repo, "ls-files", "--others", "--exclude-standard", "--", *PREFIXES):
        raise IdentityError("untracked v5 protocol files exist")
    entries = []
    for line in _git(repo, "ls-tree", "-r", head, "--", *PREFIXES).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        entries.append({"path": path, "mode": mode, "type": kind, "blob": blob})

    manifest = json.loads(dependency_manifest.read_text(encoding="utf-8"))
    declared_hash = manifest.pop("dependency_tree_hash")
    if canonical_hash(manifest) != declared_hash:
        raise IdentityError("dependency manifest hash mismatch")
    for item in manifest["files"]:
        path = repo / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise IdentityError(f"dependency content changed: {item['path']}")
    return Binding(
        head,
        REMOTE,
        canonical_hash(entries),
        declared_hash,
        True,
        True,
    )


def acquire_exclusive_lock(repo: Path, commit_sha: str) -> Path:
    common = Path(_git(repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (repo / common).resolve()
    path = common / "l0-surrogate-v5.execution.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise IdentityError("exclusive v5 execution lock already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(commit_sha + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path
