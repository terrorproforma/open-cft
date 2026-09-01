"""Synthetic detached-worktree identity and exclusive-lock tests."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v5.identity import (
    IdentityError,
    SUBJECT,
    acquire_exclusive_lock,
    bind,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    detached = tmp_path / "detached"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", str(source))
    _git(source, "config", "user.name", "V5 Test")
    _git(source, "config", "user.email", "v5@example.invalid")
    _git(source, "checkout", "-b", "feat/sota-foundation")
    dependency = source / "modern/config/dependency.txt"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("dependency\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "base")
    for relative, text in (
        ("modern/experiments/l0_surrogate_v5/protocol.py", "V = 5\n"),
        ("modern/tests/experiments/l0_surrogate_v5/test_protocol.py", "def test_ok(): assert True\n"),
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(source, "add", "modern/experiments/l0_surrogate_v5")
    _git(source, "add", "modern/tests/experiments/l0_surrogate_v5")
    _git(source, "commit", "-m", SUBJECT)
    commit = _git(source, "rev-parse", "HEAD")
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "-u", "origin", "feat/sota-foundation")
    _git(source, "worktree", "add", "--detach", str(detached), commit)
    payload = {
        "document_type": "test",
        "schema_version": "1",
        "files": [
            {
                "path": "modern/config/dependency.txt",
                "sha256": sha256(
                    (detached / "modern/config/dependency.txt").read_bytes()
                ).hexdigest(),
            }
        ],
    }
    payload["dependency_tree_hash"] = canonical_hash(payload)
    manifest = detached / "dependencies.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return source, detached, commit


def test_valid_detached_binding_and_exclusive_lock(tmp_path: Path) -> None:
    _, detached, commit = _repository(tmp_path)
    binding = bind(detached, detached / "dependencies.json")
    assert binding.commit_sha == commit
    first = acquire_exclusive_lock(detached, commit)
    assert first.read_text(encoding="utf-8").strip() == commit
    with pytest.raises(IdentityError, match="already exists"):
        acquire_exclusive_lock(detached, commit)


def test_attached_head_is_rejected(tmp_path: Path) -> None:
    source, _, _ = _repository(tmp_path)
    payload = {
        "document_type": "test",
        "schema_version": "1",
        "files": [
            {
                "path": "modern/config/dependency.txt",
                "sha256": sha256(
                    (source / "modern/config/dependency.txt").read_bytes()
                ).hexdigest(),
            }
        ],
    }
    payload["dependency_tree_hash"] = canonical_hash(payload)
    manifest = source / "dependencies.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IdentityError, match="detached"):
        bind(source, manifest)


def test_changed_working_protocol_and_dependency_are_rejected(tmp_path: Path) -> None:
    _, detached, _ = _repository(tmp_path)
    manifest = detached / "dependencies.json"
    protocol = detached / "modern/experiments/l0_surrogate_v5/protocol.py"
    protocol.write_text("changed\n", encoding="utf-8")
    with pytest.raises(IdentityError, match="differs"):
        bind(detached, manifest)
    _git(detached, "checkout", "--", str(protocol.relative_to(detached)))
    dependency = detached / "modern/config/dependency.txt"
    dependency.write_text("changed\n", encoding="utf-8")
    with pytest.raises(IdentityError, match="dependency content changed"):
        bind(detached, manifest)
