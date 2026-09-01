"""Exact detached identity and actual-import runtime closure for v6."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cft_revival.surrogates.identity import canonical_hash

SUBJECT = "preregister L0 surrogate v6 experiment"
REMOTE = "origin/feat/sota-foundation"
PREFIXES = (
    "modern/experiments/l0_surrogate_v6/",
    "modern/tests/experiments/l0_surrogate_v6/",
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
    repo = repo.resolve()
    records: dict[str, dict[str, str]] = {}
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
            relative = path.resolve().relative_to(repo).as_posix()
        except ValueError:
            continue
        listing = _git(repo, "ls-tree", "HEAD", "--", relative)
        if not listing:
            raise IdentityError(f"imported module is not committed: {name}")
        metadata, listed_path = listing.split("\t", 1)
        _, kind, blob = metadata.split()
        if kind != "blob" or listed_path != relative:
            raise IdentityError(f"invalid Git object for imported module: {name}")
        working_blob = _git(repo, "hash-object", relative)
        if working_blob != blob:
            raise IdentityError(f"imported module bytes differ from HEAD: {name}")
        records[name] = {"module": name, "path": relative, "blob": blob}
    return sorted(records.values(), key=lambda item: (item["module"], item["path"]))


@dataclass(frozen=True)
class Binding:
    commit_sha: str
    protocol_tree_hash: str
    runtime_closure_hash: str
    runtime_closure: tuple[dict[str, str], ...]
    clean_status: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "commit_sha": self.commit_sha,
            "remote_ref": REMOTE,
            "protocol_tree_hash": self.protocol_tree_hash,
            "runtime_closure_hash": self.runtime_closure_hash,
            "runtime_closure": list(self.runtime_closure),
            "clean_status": self.clean_status,
            "detached_head": True,
        }


def bind(repo: Path, manifest_path: Path) -> Binding:
    repo = repo.resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "symbolic-ref", "-q", "HEAD", check=False):
        raise IdentityError("v6 execution requires detached HEAD")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, REMOTE],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode:
        raise IdentityError("detached commit is absent from required remote")
    if _git(repo, "show", "-s", "--format=%s", head) != SUBJECT:
        raise IdentityError("HEAD is not the v6 preregistration commit")
    changed = tuple(
        line
        for line in _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
        if line
    )
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise IdentityError("preregistration commit is not exact-path isolated")
    if any("/results/" in path or path.endswith("RESULTS_REPORT.md") for path in changed):
        raise IdentityError("preregistration commit contains result artifacts")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise IdentityError("detached execution worktree is not completely clean")

    closure = runtime_closure(repo)
    external = [
        item
        for item in closure
        if not item["path"].startswith("modern/experiments/l0_surrogate_v6/")
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_hash = manifest.pop("dependency_manifest_hash")
    if canonical_hash(manifest) != declared_hash:
        raise IdentityError("runtime dependency manifest hash mismatch")
    if external != manifest["external_imported_modules"]:
        raise IdentityError("actual external import closure differs from preregistration")
    for artifact in manifest["non_module_artifacts"]:
        listing = _git(repo, "ls-tree", "HEAD", "--", artifact["path"])
        blob = listing.split()[2]
        if blob != artifact["blob"] or _git(repo, "hash-object", artifact["path"]) != blob:
            raise IdentityError(f"non-module dependency changed: {artifact['path']}")
    tree_entries = []
    for line in _git(repo, "ls-tree", "-r", "HEAD", "--", *PREFIXES).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        tree_entries.append({"path": path, "mode": mode, "type": kind, "blob": blob})
    return Binding(
        head,
        canonical_hash(tree_entries),
        canonical_hash(closure),
        tuple(closure),
        True,
    )


def acquire_lock(repo: Path, commit: str) -> Path:
    common = Path(_git(repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (repo / common).resolve()
    path = common / "l0-surrogate-v6.execution.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise IdentityError("exclusive v6 execution lock already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(commit + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path
