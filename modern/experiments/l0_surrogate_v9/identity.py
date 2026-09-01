"""Detached Git identity and exclusive one-run lock for v9."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v8.identity import (
    Binding,
    IdentityError,
    _git,
    runtime_closure,
)

SUBJECT = "preregister L0 surrogate v9 experiment"
REMOTE = "origin/feat/sota-foundation"
PREFIXES = (
    "modern/experiments/l0_surrogate_v9/",
    "modern/tests/experiments/l0_surrogate_v9/",
)


def bind(repo: Path, manifest_path: Path) -> Binding:
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "symbolic-ref", "-q", "HEAD", check=False):
        raise IdentityError("v9 requires detached HEAD")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, REMOTE],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode:
        raise IdentityError("v9 commit is absent from required remote")
    if _git(repo, "show", "-s", "--format=%s", head) != SUBJECT:
        raise IdentityError("HEAD is not v9 preregistration")
    changed = tuple(
        line
        for line in _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head
        ).splitlines()
        if line
    )
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise IdentityError("v9 preregistration is not exact-path isolated")
    if any("/results/" in path or path.endswith("RESULTS_REPORT.md") for path in changed):
        raise IdentityError("v9 preregistration contains results")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise IdentityError("v9 detached worktree is not clean")
    closure = runtime_closure(repo)
    external = [
        item
        for item in closure
        if not item["path"].startswith("modern/experiments/l0_surrogate_v9/")
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_hash = manifest.pop("dependency_manifest_hash")
    if canonical_hash(manifest) != declared_hash:
        raise IdentityError("dependency manifest hash mismatch")
    declared_external = list(manifest["direct_external_imported_modules"])
    for inherited in manifest["inherited_closures"]:
        inherited_path = repo / inherited["path"]
        inherited_value = json.loads(inherited_path.read_text(encoding="utf-8"))
        inherited_hash = inherited_value.pop("dependency_manifest_hash")
        if canonical_hash(inherited_value) != inherited_hash:
            raise IdentityError("inherited dependency manifest hash mismatch")
        if inherited_hash != inherited["manifest_hash"]:
            raise IdentityError("inherited dependency manifest identity mismatch")
        declared_external.extend(inherited_value["external_imported_modules"])
    declared_external.sort(key=lambda item: (item["module"], item["path"]))
    if external != declared_external:
        raise IdentityError("actual external runtime closure mismatch")
    for artifact in manifest["non_module_artifacts"]:
        listing = _git(repo, "ls-tree", "HEAD", "--", artifact["path"])
        blob = listing.split()[2]
        if blob != artifact["blob"] or _git(repo, "hash-object", artifact["path"]) != blob:
            raise IdentityError(f"non-module dependency changed: {artifact['path']}")
    entries = []
    for line in _git(repo, "ls-tree", "-r", "HEAD", "--", *PREFIXES).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        entries.append({"path": path, "mode": mode, "type": kind, "blob": blob})
    return Binding(head, canonical_hash(entries), canonical_hash(closure), tuple(closure))


def acquire_lock(repo: Path, commit: str) -> Path:
    common = Path(_git(repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (repo / common).resolve()
    path = common / "l0-surrogate-v9.execution.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise IdentityError("exclusive v9 execution lock already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(commit + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path
