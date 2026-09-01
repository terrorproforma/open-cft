"""Synthetic clean-worktree identity tests for v6."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v6.identity import IdentityError, SUBJECT, bind


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    detached = tmp_path / "detached"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", str(source))
    _git(source, "config", "user.name", "V6 Test")
    _git(source, "config", "user.email", "v6@example.invalid")
    _git(source, "checkout", "-b", "feat/sota-foundation")
    config = source / "modern/config/l0-deterministic-sweep.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "base")
    config_blob = _git(source, "rev-parse", "HEAD:modern/config/l0-deterministic-sweep.json")
    payload = {
        "document_type": "test",
        "schema_version": "6.0",
        "external_imported_modules": [],
        "non_module_artifacts": [
            {
                "path": "modern/config/l0-deterministic-sweep.json",
                "blob": config_blob,
            }
        ],
    }
    payload["dependency_manifest_hash"] = canonical_hash(payload)
    for relative, content in (
        ("modern/experiments/l0_surrogate_v6/protocol.py", "V = 6\n"),
        (
            "modern/experiments/l0_surrogate_v6/dependency-manifest.json",
            json.dumps(payload),
        ),
        ("modern/tests/experiments/l0_surrogate_v6/test_protocol.py", "def test_ok(): assert True\n"),
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(source, "add", "modern/experiments/l0_surrogate_v6")
    _git(source, "add", "modern/tests/experiments/l0_surrogate_v6")
    _git(source, "commit", "-m", SUBJECT)
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "-u", "origin", "feat/sota-foundation")
    _git(source, "worktree", "add", "--detach", str(detached), "HEAD")
    return source, detached


def test_clean_detached_actual_closure_binding(tmp_path: Path) -> None:
    _, detached = _repo(tmp_path)
    binding = bind(
        detached,
        detached / "modern/experiments/l0_surrogate_v6/dependency-manifest.json",
    )
    assert binding.clean_status is True
    assert binding.runtime_closure == ()


def test_any_dirty_status_is_rejected(tmp_path: Path) -> None:
    _, detached = _repo(tmp_path)
    (detached / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(IdentityError, match="not completely clean"):
        bind(
            detached,
            detached / "modern/experiments/l0_surrogate_v6/dependency-manifest.json",
        )
