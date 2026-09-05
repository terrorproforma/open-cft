"""Recompute sealed source digests from Git blobs at a frozen commit and report live-tree drift.

Every helper here fails closed: an unresolvable commit, a recorded path without a blob at that
commit, or a CR byte inside a hashed blob raises instead of producing a digest that could be
mistaken for a mismatch. Blobs are read through one ``git cat-file --batch`` process per scope
(per-file ``git show`` costs ~170 ms each on Windows), so a 35-file scope binds in well under a
second.

The digest recipe is the one every preregistered experiment in ``modern/experiments`` uses for
its code and dependency scopes: ``sha256`` over ``path NUL bytes NUL`` for each file in the
RECORDED order. The inventory an experiment recorded (``experiment_code_files``,
``dependency_source_files``, ``field_pipeline_source_files``) is the sealed scope; it is taken
from the record, never re-derived from a live glob, so a module added to a shared package after
the campaign shows up as live drift (``added``) rather than silently entering the frozen digest.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "FrozenBlobError",
    "FrozenCommitError",
    "SealedScope",
    "blob_exists",
    "frozen_scope_report",
    "path_bytes_sha256",
    "read_blobs",
    "resolve_commit",
    "verify_sealed_scopes",
]

_HEX = re.compile(r"^[0-9a-f]{7,40}$")


class FrozenCommitError(RuntimeError):
    """The commit a record names is not a commit object of this repository (fail closed)."""


class FrozenBlobError(RuntimeError):
    """A recorded path has no blob at the frozen commit (fail closed)."""


def _git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *arguments], cwd=repo, input=input_bytes, capture_output=True, check=False)


def resolve_commit(repo: Path, commit: str) -> str:
    """Return the full SHA of ``commit`` if it is a commit object present in ``repo``."""

    if not isinstance(commit, str) or not _HEX.match(commit):
        raise FrozenCommitError(f"{commit!r} is not a hexadecimal commit identifier")
    completed = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if completed.returncode != 0:
        raise FrozenCommitError(f"commit {commit} is not available in {repo} (unreachable, shallow, or not fetched)")
    return completed.stdout.decode("ascii").strip()


def blob_exists(repo: Path, commit: str, path: str) -> bool:
    """True when ``path`` (repository-relative, posix) has a blob at ``commit``."""

    full = resolve_commit(repo, commit)
    completed = _git(repo, "cat-file", "-e", f"{full}:{path}")
    return completed.returncode == 0


def read_blobs(repo: Path, commit: str, paths: Sequence[str]) -> dict[str, bytes]:
    """Bytes of every repository-relative ``path`` at ``commit`` via one ``git cat-file --batch``."""

    full = resolve_commit(repo, commit)
    unique: list[str] = []
    for path in paths:
        if "\n" in path:
            raise FrozenBlobError(f"path {path!r} contains a newline")
        if path not in unique:
            unique.append(path)
    if not unique:
        return {}
    request = "".join(f"{full}:{path}\n" for path in unique).encode("utf-8")
    completed = _git(repo, "cat-file", "--batch", input_bytes=request)
    if completed.returncode != 0:
        raise FrozenBlobError(f"git cat-file --batch failed: {completed.stderr.decode('utf-8', 'replace').strip()}")
    output = completed.stdout
    blobs: dict[str, bytes] = {}
    position = 0
    for path in unique:
        newline = output.find(b"\n", position)
        if newline < 0:
            raise FrozenBlobError(f"truncated cat-file output before {path}")
        header = output[position:newline].decode("utf-8", "replace").split(" ")
        if header[-1] in ("missing", "ambiguous"):
            raise FrozenBlobError(f"{path} has no blob at {full[:12]} ({header[-1]})")
        if len(header) != 3 or header[1] != "blob":
            raise FrozenBlobError(f"{path} at {full[:12]} is not a blob: {' '.join(header)}")
        size = int(header[2])
        start = newline + 1
        data = output[start : start + size]
        if len(data) != size:
            raise FrozenBlobError(f"truncated blob for {path} at {full[:12]}")
        blobs[path] = data
        position = start + size + 1  # the trailing LF cat-file appends after every object
    return blobs


def path_bytes_sha256(items: Iterable[tuple[str, bytes]]) -> str:
    """SHA-256 over ``path NUL bytes NUL`` per item in order; CR bytes fail closed."""

    digest = hashlib.sha256()
    for path, data in items:
        if b"\r" in data:
            raise ValueError(f"hashed source {path} contains CR bytes")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class SealedScope:
    """One sealed digest: the recorded inventory it covers and, optionally, the live tree's value."""

    key: str
    root: str
    files: tuple[str, ...]
    sealed_sha256: str
    live_sha256: str | None = None
    live_files: tuple[str, ...] | None = None

    def repository_path(self, relative: str) -> str:
        return f"{self.root}/{relative}" if self.root else relative


def frozen_scope_report(repo: Path, commit: str, scope: SealedScope) -> dict[str, Any]:
    """Recompute ``scope`` from the blobs at ``commit``; describe the live tree's drift if given."""

    full = resolve_commit(repo, commit)
    blobs = read_blobs(repo, full, [scope.repository_path(name) for name in scope.files])
    frozen = path_bytes_sha256((name, blobs[scope.repository_path(name)]) for name in scope.files)
    report: dict[str, Any] = {
        "key": scope.key,
        "commit": full,
        "root": scope.root,
        "file_count": len(scope.files),
        "sealed_sha256": scope.sealed_sha256,
        "frozen_sha256": frozen,
        "sealed_matches_frozen_commit": frozen == scope.sealed_sha256,
        "live": None,
    }
    if scope.live_sha256 is not None:
        live_files = scope.live_files if scope.live_files is not None else scope.files
        recorded = set(scope.files)
        present = set(live_files)
        changed: list[str] = []
        missing: list[str] = []
        for name in scope.files:
            if name not in present:
                continue
            path = repo / scope.repository_path(name)
            if not path.is_file():
                missing.append(name)
            elif path.read_bytes() != blobs[scope.repository_path(name)]:
                changed.append(name)
        report["live"] = {
            "sha256": scope.live_sha256,
            "matches_sealed": scope.live_sha256 == scope.sealed_sha256,
            "drift": scope.live_sha256 != scope.sealed_sha256,
            "added": sorted(present - recorded),
            "removed": sorted(recorded - present) + missing,
            "changed": changed,
        }
    return report


def verify_sealed_scopes(
    repo: Path,
    commit: str,
    scopes: Sequence[SealedScope],
    *,
    strict_live_tree: bool = False,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    """``(checks, reports)`` for every scope.

    ``checks[f"{key}_frozen"]`` is the honest post-execution statement (sealed digest equals the
    frozen commit's blobs). With ``strict_live_tree`` the pre-execution statement
    ``checks[f"{key}_current"]`` (live tree equals the seal) is added as well - the mode a
    shakedown or ``prepare`` needs, and the only mode under which live drift is a failure.
    """

    checks: dict[str, bool] = {}
    reports: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        report = frozen_scope_report(repo, commit, scope)
        checks[f"{scope.key}_frozen"] = bool(report["sealed_matches_frozen_commit"])
        if strict_live_tree and report["live"] is not None:
            checks[f"{scope.key}_current"] = bool(report["live"]["matches_sealed"])
        reports[scope.key] = report
    return checks, reports
