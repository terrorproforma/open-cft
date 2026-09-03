"""Manifest recovery for complete-but-unpublished attempts and the pinned-descriptor cap."""

from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from cft_revival.experiment_runtime import (
    MAX_PINNED_DESCRIPTORS,
    RECOVERY_KIND,
    AtomicArtifactStore,
    BundleState,
    Decision,
    ExecutionAttestation,
    ExperimentRuntime,
    LifecycleError,
    RuntimeCallbacks,
    diagnose_bundle,
    finalize_unpublished_attempt,
    strict_json_file,
    validate_bundle,
)
from cft_revival.experiment_runtime import lifecycle


def _producer() -> None:
    pass


def _runtime(tmp_path: Path, name: str = "results") -> ExperimentRuntime:
    modern = Path(__file__).resolve().parents[2]
    return ExperimentRuntime(
        experiment_id="recovery-test",
        result_root=tmp_path / name,
        cache_root=tmp_path / f"{name}-cache",
        attestation=ExecutionAttestation(attempt=1, commit="b" * 40, command="python -m fake", host="test-host", device="fake", clean_worktree=True),
        producer=_producer,
        source_root=modern,
    )


def _callbacks(files: int = 12) -> RuntimeCallbacks:
    def prebundle(context: Any) -> Mapping[str, Any]:
        context.write_json("artifacts/protocol-copy.json", {"protocol": "fake"})
        return {"ready": True}

    def development(context: Any) -> Decision:
        context.before_expensive("solve", kind="solver", details={"case": "dev"})
        for index in range(files):
            context.write_json(f"artifacts/cases/case-{index:03d}.json", {"index": index, "value": index * 0.5})
        return Decision(True, {"gate": "development"})

    def assessment(context: Any) -> Decision:
        context.before_expensive("labels", kind="label", details={"partition": "held-out"})
        context.write_json("artifacts/assessment.json", {"accepted": True})
        return Decision(True, {"gate": "assessment"})

    return RuntimeCallbacks(prebundle, development, assessment)


def _fail_sealing(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AtomicArtifactStore.seal_files

    def failing(self: AtomicArtifactStore, relatives: Any, *, limit: int | None = None) -> list[int]:
        original(self, [], limit=limit)
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(AtomicArtifactStore, "seal_files", failing)


def test_descriptor_limited_publication_leaves_a_complete_unpublished_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_sealing(monkeypatch)
    with pytest.raises(OSError, match="Too many open files"):
        _runtime(tmp_path).run(_callbacks())
    root = tmp_path / "results"
    assert (root / "terminal.json").is_file() and (root / "execution-lock.json").is_file()
    assert not (root / "manifest.json").exists()
    assert "incomplete-no-manifest" in diagnose_bundle(root)
    assert strict_json_file(root / "terminal.json")["state"] == "accepted_result"
    with pytest.raises(LifecycleError, match="no completion manifest"):
        validate_bundle(root)


def test_recovery_publishes_the_same_manifest_the_locked_attempt_would_have(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = _runtime(tmp_path, "reference").run(_callbacks())
    assert reference.state is BundleState.ACCEPTED_RESULT
    _fail_sealing(monkeypatch)
    with pytest.raises(OSError):
        _runtime(tmp_path, "results").run(_callbacks())
    monkeypatch.undo()
    record = finalize_unpublished_attempt(tmp_path / "results")
    assert record["recovery"] == RECOVERY_KIND and record["state"] == "accepted_result"
    manifest = validate_bundle(tmp_path / "results")
    assert manifest["state"] == "accepted_result"
    assert manifest["artifact_count"] == reference.manifest["artifact_count"]
    assert [item["path"] for item in manifest["artifacts"]] == [item["path"] for item in reference.manifest["artifacts"]]
    assert manifest["required_directories"] == reference.manifest["required_directories"]
    assert manifest["durability"] == reference.manifest["durability"]
    for left, right in zip(manifest["artifacts"], reference.manifest["artifacts"], strict=True):
        if left["type"] == "file" and left["path"].startswith("artifacts/"):
            assert left["byte_sha256"] == right["byte_sha256"], left["path"]
    published = (tmp_path / "results" / "manifest.json").read_bytes()
    assert json.loads(published) == record["manifest"]
    assert record["manifest_byte_sha256"] == __import__("hashlib").sha256(published).hexdigest()


def test_recovery_refuses_published_incomplete_or_tampered_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = _runtime(tmp_path, "published").run(_callbacks(files=3))
    assert outcome.state is BundleState.ACCEPTED_RESULT
    with pytest.raises(LifecycleError, match="already has a completion manifest"):
        finalize_unpublished_attempt(tmp_path / "published")

    class Crash(BaseException):
        pass

    def crash(context: Any) -> Decision:
        context.write_json("artifacts/before-crash.json", {"durable": True})
        raise Crash()

    with pytest.raises(Crash):
        _runtime(tmp_path, "crashed").run(RuntimeCallbacks(lambda _c: {}, crash, lambda _c: Decision(True)))
    with pytest.raises(LifecycleError, match="no durable terminal record"):
        finalize_unpublished_attempt(tmp_path / "crashed")

    _fail_sealing(monkeypatch)
    with pytest.raises(OSError):
        _runtime(tmp_path, "tampered").run(_callbacks(files=3))
    monkeypatch.undo()
    target = tmp_path / "tampered" / "artifacts" / "cases" / "case-001.json"
    target.write_bytes(target.read_bytes().replace(b"0.5", b"0.6"))
    with pytest.raises(Exception):
        finalize_unpublished_attempt(tmp_path / "tampered")
    assert not (tmp_path / "tampered" / "manifest.json").exists()


def test_seal_files_respects_the_descriptor_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pinned: list[int] = []
    original = AtomicArtifactStore.seal_files

    def counting(self: AtomicArtifactStore, relatives: Any, *, limit: int | None = None) -> list[int]:
        descriptors = original(self, relatives, limit=limit)
        pinned.append(len(descriptors))
        return descriptors

    monkeypatch.setattr(AtomicArtifactStore, "seal_files", counting)
    monkeypatch.setattr(lifecycle, "MAX_PINNED_DESCRIPTORS", 5)
    outcome = _runtime(tmp_path).run(_callbacks(files=20))
    assert outcome.state is BundleState.ACCEPTED_RESULT
    assert pinned == [5]
    manifest = validate_bundle(tmp_path / "results")
    assert sum(item["type"] == "file" for item in manifest["artifacts"]) > 5
    assert MAX_PINNED_DESCRIPTORS < 8192
