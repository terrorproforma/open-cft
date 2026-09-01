"""Detached Git and actual runtime import identity for v8."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cft_revival.surrogates.identity import canonical_hash

SUBJECT = "preregister L0 surrogate v8 experiment"
REMOTE = "origin/feat/sota-foundation"
PREFIXES = (
    "modern/experiments/l0_surrogate_v8/",
    "modern/tests/experiments/l0_surrogate_v8/",
)


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


def runtime_closure(repo: Path) -> list[dict[str, str]]:
    records = {}
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "cft_revival"
            or name.startswith("cft_revival.")
            or name.startswith("experiments.l0_surrogate_")
        ):
            continue
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw)
        if path.suffix in (".pyc", ".pyo"):
            path = path.with_suffix(".py")
        try:
            relative = path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            continue
        listing = _git(repo, "ls-tree", "HEAD", "--", relative)
        if not listing:
            raise IdentityError(f"imported module is uncommitted: {name}")
        metadata, listed_path = listing.split("\t", 1)
        _, kind, blob = metadata.split()
        if kind != "blob" or listed_path != relative:
            raise IdentityError(f"invalid imported module object: {name}")
        if _git(repo, "hash-object", relative) != blob:
            raise IdentityError(f"imported module differs from HEAD: {name}")
        records[name] = {"module": name, "path": relative, "blob": blob}
    return sorted(records.values(), key=lambda item: (item["module"], item["path"]))


@dataclass(frozen=True)
class Binding:
    commit_sha: str
    protocol_tree_hash: str
    runtime_closure_hash: str
    runtime_closure: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "commit_sha": self.commit_sha,
            "remote_ref": REMOTE,
            "protocol_tree_hash": self.protocol_tree_hash,
            "runtime_closure_hash": self.runtime_closure_hash,
            "runtime_closure": list(self.runtime_closure),
            "detached_head": True,
            "clean_status": True,
        }


def bind(repo: Path, manifest_path: Path) -> Binding:
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "symbolic-ref", "-q", "HEAD", check=False):
        raise IdentityError("v8 requires detached HEAD")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, REMOTE],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode:
        raise IdentityError("v8 commit is absent from required remote")
    if _git(repo, "show", "-s", "--format=%s", head) != SUBJECT:
        raise IdentityError("HEAD is not v8 preregistration")
    changed = tuple(
        line
        for line in _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head
        ).splitlines()
        if line
    )
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise IdentityError("v8 preregistration is not exact-path isolated")
    if any("/results/" in path or path.endswith("RESULTS_REPORT.md") for path in changed):
        raise IdentityError("v8 preregistration contains results")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise IdentityError("v8 detached worktree is not clean")
    closure = runtime_closure(repo)
    external = [
        item
        for item in closure
        if not item["path"].startswith("modern/experiments/l0_surrogate_v8/")
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.pop("dependency_manifest_hash")
    if canonical_hash(manifest) != declared:
        raise IdentityError("dependency manifest hash mismatch")
    if external != manifest["external_imported_modules"]:
        raise IdentityError("actual external runtime closure mismatch")
    for artifact in manifest["non_module_artifacts"]:
        blob = _git(repo, "ls-tree", "HEAD", "--", artifact["path"]).split()[2]
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
    path = common / "l0-surrogate-v8.execution.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise IdentityError("exclusive v8 execution lock already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(commit + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path
