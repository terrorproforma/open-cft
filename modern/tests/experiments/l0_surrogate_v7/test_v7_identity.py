"""Synthetic detached identity tests for v7."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v7.identity import IdentityError, SUBJECT, bind


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _repository(tmp_path: Path) -> Path:
    remote, source, detached = (
        tmp_path / "remote.git",
        tmp_path / "source",
        tmp_path / "detached",
    )
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", str(source))
    _git(source, "config", "user.name", "V7 Test")
    _git(source, "config", "user.email", "v7@example.invalid")
    _git(source, "checkout", "-b", "feat/sota-foundation")
    config = source / "modern/config/l0-deterministic-sweep.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "base")
    blob = _git(source, "rev-parse", "HEAD:modern/config/l0-deterministic-sweep.json")
    manifest = {
        "document_type": "test",
        "schema_version": "7.0",
        "external_imported_modules": [],
        "non_module_artifacts": [
            {"path": "modern/config/l0-deterministic-sweep.json", "blob": blob}
        ],
    }
    manifest["dependency_manifest_hash"] = canonical_hash(manifest)
    for relative, content in (
        ("modern/experiments/l0_surrogate_v7/protocol.py", "V = 7\n"),
        (
            "modern/experiments/l0_surrogate_v7/dependency-manifest.json",
            json.dumps(manifest),
        ),
        ("modern/tests/experiments/l0_surrogate_v7/test_protocol.py", "def test_ok(): assert True\n"),
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(source, "add", "modern/experiments/l0_surrogate_v7")
    _git(source, "add", "modern/tests/experiments/l0_surrogate_v7")
    _git(source, "commit", "-m", SUBJECT)
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "-u", "origin", "feat/sota-foundation")
    _git(source, "worktree", "add", "--detach", str(detached), "HEAD")
    return detached


def test_clean_detached_binding_and_dirty_rejection(tmp_path: Path) -> None:
    detached = _repository(tmp_path)
    manifest = detached / "modern/experiments/l0_surrogate_v7/dependency-manifest.json"
    binding = bind(detached, manifest)
    assert binding.runtime_closure == ()
    (detached / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(IdentityError, match="not completely clean"):
        bind(detached, manifest)
