"""cft_revival.provenance: sealed digests bind to a frozen commit's blobs; live drift is reported."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from cft_revival.provenance import (
    FrozenBlobError,
    FrozenCommitError,
    SealedScope,
    blob_exists,
    frozen_scope_report,
    path_bytes_sha256,
    read_blobs,
    resolve_commit,
    verify_sealed_scopes,
)

REPOSITORY = Path(__file__).resolve().parents[3]


def _git(repo: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, text=True, env=env)
    return completed.stdout.strip()


@pytest.fixture()
def scratch_repo(tmp_path: Path) -> tuple[Path, str]:
    """A throw-away repository with one committed scope of three LF files, then a drifted worktree."""

    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": str(tmp_path / "nogitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    _git(repo, "init", "-q", env=env)
    _git(repo, "config", "core.autocrlf", "false", env=env)
    package = repo / "modern" / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"# init\n")
    (package / "a.py").write_bytes(b"a = 1\n")
    (package / "b.py").write_bytes(b"b = 2\n")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "seal", env=env)
    commit = _git(repo, "rev-parse", "HEAD", env=env)
    # drift: one file changed, one removed, one added - none of it committed
    (package / "a.py").write_bytes(b"a = 2\n")
    (package / "b.py").unlink()
    (package / "c.py").write_bytes(b"c = 3\n")
    return repo, commit


def test_resolve_commit_fails_closed_on_unknown_or_malformed_identifiers() -> None:
    head = _git(REPOSITORY, "rev-parse", "HEAD")
    assert resolve_commit(REPOSITORY, head) == head
    assert resolve_commit(REPOSITORY, head[:12]) == head
    with pytest.raises(FrozenCommitError):
        resolve_commit(REPOSITORY, "0" * 40)
    with pytest.raises(FrozenCommitError):
        resolve_commit(REPOSITORY, "HEAD")  # symbolic names are not evidence of a frozen commit
    with pytest.raises(FrozenCommitError):
        resolve_commit(REPOSITORY, "")


def test_read_blobs_batches_and_matches_git_show() -> None:
    head = _git(REPOSITORY, "rev-parse", "HEAD")
    paths = ["modern/pyproject.toml", "modern/src/cft_revival/__init__.py", "modern/pyproject.toml"]
    blobs = read_blobs(REPOSITORY, head, paths)
    assert set(blobs) == set(paths)
    for path in set(paths):
        expected = subprocess.run(["git", "show", f"{head}:{path}"], cwd=REPOSITORY, check=True, capture_output=True).stdout
        assert blobs[path] == expected
    assert read_blobs(REPOSITORY, head, []) == {}
    with pytest.raises(FrozenBlobError, match="no blob"):
        read_blobs(REPOSITORY, head, ["modern/pyproject.toml", "modern/does/not/exist.py"])
    with pytest.raises(FrozenBlobError, match="not a blob"):
        read_blobs(REPOSITORY, head, ["modern/src"])
    assert blob_exists(REPOSITORY, head, "modern/pyproject.toml") is True
    assert blob_exists(REPOSITORY, head, "modern/does/not/exist.py") is False


def test_path_bytes_sha256_is_the_experiments_recipe_and_refuses_cr() -> None:
    items = [("a.py", b"a = 1\n"), ("b.py", b"")]
    expected = hashlib.sha256(b"a.py\0a = 1\n\0b.py\0\0").hexdigest()
    assert path_bytes_sha256(items) == expected
    assert path_bytes_sha256(reversed(items)) != expected  # order is part of the seal
    with pytest.raises(ValueError, match="CR bytes"):
        path_bytes_sha256([("a.py", b"a = 1\r\n")])


def test_frozen_scope_report_binds_to_the_commit_and_describes_live_drift(scratch_repo: tuple[Path, str]) -> None:
    repo, commit = scratch_repo
    recorded = ("src/pkg/__init__.py", "src/pkg/a.py", "src/pkg/b.py")
    sealed = hashlib.sha256(b"src/pkg/__init__.py\0# init\n\0src/pkg/a.py\0a = 1\n\0src/pkg/b.py\0b = 2\n\0").hexdigest()
    live_files = ("src/pkg/__init__.py", "src/pkg/a.py", "src/pkg/c.py")
    live = path_bytes_sha256((name, (repo / "modern" / name).read_bytes()) for name in live_files)
    scope = SealedScope("dependency_source_sha256", "modern", recorded, sealed, live, live_files)
    report = frozen_scope_report(repo, commit, scope)
    assert report["commit"] == commit and report["file_count"] == 3
    assert report["frozen_sha256"] == sealed and report["sealed_matches_frozen_commit"] is True
    drift = report["live"]
    assert drift["sha256"] == live and drift["matches_sealed"] is False and drift["drift"] is True
    assert drift["added"] == ["src/pkg/c.py"]
    assert drift["removed"] == ["src/pkg/b.py"]
    assert drift["changed"] == ["src/pkg/a.py"]
    # without a live digest the report is purely about the frozen commit
    frozen_only = frozen_scope_report(repo, commit, SealedScope("k", "modern", recorded, sealed))
    assert frozen_only["live"] is None and frozen_only["sealed_matches_frozen_commit"] is True
    # a wrong seal is detected against the blobs, independent of the worktree state
    bogus = frozen_scope_report(repo, commit, SealedScope("k", "modern", recorded, "0" * 64))
    assert bogus["sealed_matches_frozen_commit"] is False and bogus["frozen_sha256"] == sealed
    # a recorded path the commit does not hold fails closed
    with pytest.raises(FrozenBlobError):
        frozen_scope_report(repo, commit, SealedScope("k", "modern", recorded + ("src/pkg/c.py",), sealed))
    with pytest.raises(FrozenCommitError):
        frozen_scope_report(repo, "f" * 40, scope)


def test_verify_sealed_scopes_reports_drift_and_fails_only_in_strict_mode(scratch_repo: tuple[Path, str]) -> None:
    repo, commit = scratch_repo
    recorded = ("src/pkg/__init__.py", "src/pkg/a.py", "src/pkg/b.py")
    sealed = path_bytes_sha256((name, data) for name, data in (("src/pkg/__init__.py", b"# init\n"), ("src/pkg/a.py", b"a = 1\n"), ("src/pkg/b.py", b"b = 2\n")))
    drifted = SealedScope("dependency_source_sha256", "modern", recorded, sealed, "1" * 64, ("src/pkg/__init__.py", "src/pkg/a.py", "src/pkg/c.py"))
    stable = SealedScope("experiment_code_sha256", "modern", ("src/pkg/__init__.py",), path_bytes_sha256([("src/pkg/__init__.py", b"# init\n")]), path_bytes_sha256([("src/pkg/__init__.py", b"# init\n")]), ("src/pkg/__init__.py",))
    checks, reports = verify_sealed_scopes(repo, commit, [drifted, stable])
    assert checks == {"dependency_source_sha256_frozen": True, "experiment_code_sha256_frozen": True}
    assert reports["dependency_source_sha256"]["live"]["drift"] is True
    assert reports["experiment_code_sha256"]["live"]["drift"] is False
    strict, _reports = verify_sealed_scopes(repo, commit, [drifted, stable], strict_live_tree=True)
    assert strict == {
        "dependency_source_sha256_frozen": True,
        "dependency_source_sha256_current": False,
        "experiment_code_sha256_frozen": True,
        "experiment_code_sha256_current": True,
    }
