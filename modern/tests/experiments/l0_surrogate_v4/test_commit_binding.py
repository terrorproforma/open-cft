"""Synthetic Git identity tests; no real L0 assessment labels are accessed.

The preflight/execute tests are lifecycle-aware: once ``results/run-manifest.json``
exists they assert that both entry points refuse to run again without touching
the immutable bundle.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from experiments.l0_surrogate_v2 import protocol as science
from experiments.l0_surrogate_v4 import protocol as v4
from experiments.l0_surrogate_v4.identity import (
    CommitBindingError,
    PREREGISTRATION_SUBJECT,
    bind_execution_identity,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", str(repository))
    _git(repository, "config", "user.name", "V4 Test")
    _git(repository, "config", "user.email", "v4@example.invalid")
    _git(repository, "checkout", "-b", "feat/sota-foundation")
    (repository / "README.md").write_text("base\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")
    (repository / "modern/experiments/l0_surrogate_v4").mkdir(parents=True)
    (repository / "modern/tests/experiments/l0_surrogate_v4").mkdir(parents=True)
    (repository / "modern/experiments/l0_surrogate_v4/protocol.py").write_text(
        "PROTOCOL = 4\n", encoding="utf-8"
    )
    (repository / "modern/experiments/l0_surrogate_v4/predeclaration.json").write_text(
        '{"schema_version":"4.0"}\n', encoding="utf-8"
    )
    (repository / "modern/tests/experiments/l0_surrogate_v4/test_protocol.py").write_text(
        "def test_protocol(): assert True\n", encoding="utf-8"
    )
    _git(repository, "add", "modern/experiments/l0_surrogate_v4")
    _git(repository, "add", "modern/tests/experiments/l0_surrogate_v4")
    _git(repository, "commit", "-m", PREREGISTRATION_SUBJECT)
    preregistration = _git(repository, "rev-parse", "HEAD")
    _git(repository, "remote", "add", "origin", str(remote))
    _git(repository, "push", "-u", "origin", "feat/sota-foundation")
    return repository, base, preregistration


def test_valid_actual_head_binding(tmp_path: Path) -> None:
    repository, _, preregistration = _repository(tmp_path)
    binding = bind_execution_identity(repository, expected_head_sha=preregistration)
    assert binding.observed_head_sha == preregistration
    assert binding.protocol_commit_sha == preregistration
    assert binding.intervening_protocol_changes == 0
    assert len(binding.protocol_tree_hash) == 64


def test_nonexistent_caller_sha_is_rejected(tmp_path: Path) -> None:
    repository, _, _ = _repository(tmp_path)
    with pytest.raises(CommitBindingError, match="does not identify"):
        bind_execution_identity(repository, expected_head_sha="f" * 40)


def test_wrong_existing_ancestor_is_rejected(tmp_path: Path) -> None:
    repository, base, _ = _repository(tmp_path)
    with pytest.raises(CommitBindingError, match="does not equal"):
        bind_execution_identity(repository, expected_head_sha=base)


def test_changed_working_v4_file_is_rejected(tmp_path: Path) -> None:
    repository, _, _ = _repository(tmp_path)
    path = repository / "modern/experiments/l0_surrogate_v4/protocol.py"
    path.write_text("PROTOCOL = 'changed'\n", encoding="utf-8")
    with pytest.raises(CommitBindingError, match="working v4 protocol"):
        bind_execution_identity(repository)


def test_intervening_unrelated_pushed_commit_is_allowed(tmp_path: Path) -> None:
    repository, _, preregistration = _repository(tmp_path)
    (repository / "README.md").write_text("unrelated\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-m", "unrelated pushed work")
    observed = _git(repository, "rev-parse", "HEAD")
    _git(repository, "push", "origin", "feat/sota-foundation")
    binding = bind_execution_identity(repository)
    assert binding.observed_head_sha == observed
    assert binding.protocol_commit_sha == preregistration


def test_intervening_protocol_commit_is_rejected(tmp_path: Path) -> None:
    repository, _, _ = _repository(tmp_path)
    path = repository / "modern/experiments/l0_surrogate_v4/protocol.py"
    path.write_text("PROTOCOL = 5\n", encoding="utf-8")
    _git(repository, "add", str(path.relative_to(repository)))
    _git(repository, "commit", "-m", "change frozen protocol")
    _git(repository, "push", "origin", "feat/sota-foundation")
    with pytest.raises(CommitBindingError, match="intervening commit"):
        bind_execution_identity(repository)


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


EXECUTED = (v4.RESULTS / "run-manifest.json").is_file()


def test_preflight_does_not_load_real_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        science,
        "load_l0_rows",
        lambda declaration: (_ for _ in ()).throw(
            AssertionError("preflight accessed real L0 rows")
        ),
    )
    if EXECUTED:
        # After the single execution the real path is occupied: preflight must
        # refuse before any synthetic work, and must not touch the bundle.
        before = _tree_digest(v4.RESULTS)
        with pytest.raises(ValueError, match="results path already exists"):
            v4.preflight(record=False)
        assert _tree_digest(v4.RESULTS) == before
        # The blind-preflight property itself is still checked against an
        # unoccupied scratch results path.
        monkeypatch.setattr(v4, "RESULTS", tmp_path / "results")
    result = v4.preflight(record=False)
    assert result["passed"] is True
    assert result["real_assessment_labels_accessed"] is False


def test_execute_refuses_to_run_again_once_results_exist() -> None:
    if not EXECUTED:
        pytest.skip("the single authorised v4 execution has not happened")
    before = _tree_digest(v4.RESULTS)
    # Either the Git identity binding refuses (the protocol paths have gained
    # post-execution commits) or the single-shot guard does; both are
    # RuntimeErrors raised before any byte is written.
    with pytest.raises(RuntimeError):
        v4.execute()
    assert _tree_digest(v4.RESULTS) == before
