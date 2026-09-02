from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from cft_revival.experiment_runtime import (
    BundleState,
    Decision,
    ExecutionAttestation,
    ExistingLockError,
    ExperimentRuntime,
    FileOps,
    FilesystemSafetyError,
    RuntimeCallbacks,
    acquire_execution_lock,
    diagnose_bundle,
    preflight_result_root,
    strict_json_file,
    validate_bundle,
)


def runtime_producer() -> None:
    pass


class HardCrash(BaseException):
    pass


class RuntimeFaultOps(FileOps):
    def __init__(self, fault: str, contains: str) -> None:
        self.fault = fault
        self.contains = contains
        self.paths: dict[int, Path] = {}

    def open_exclusive(self, parent: Any, name: str, mode: int = 0o600) -> int:
        path = parent.path / name
        descriptor = super().open_exclusive(parent, name, mode)
        self.paths[descriptor] = path
        return descriptor

    def open_temporary(self, parent: Any, name: str, mode: int = 0o600) -> int:
        path = parent.path / name
        descriptor = super().open_temporary(parent, name, mode)
        self.paths[descriptor] = path
        return descriptor

    def fsync_file(self, descriptor: int) -> None:
        path = self.paths[descriptor]
        if self.fault == "fsync-file" and self.contains in str(path):
            raise OSError("injected fsync failure")
        super().fsync_file(descriptor)

    def publish(self, parent: Any, descriptor: int, source: str, target: str) -> None:
        if self.fault == "replace" and self.contains in str(parent.path / target):
            raise OSError("injected replace failure")
        super().publish(parent, descriptor, source, target)

    def remove_file(self, parent: Any, name: str) -> None:
        path = parent.path / name
        if self.fault == "cleanup" and self.contains in str(path):
            raise PermissionError("injected cleanup failure")
        super().remove_file(parent, name)

    def remove_directory(self, parent: Any, name: str) -> None:
        path = parent.path / name
        if self.fault == "cleanup" and self.contains in str(path):
            raise PermissionError("injected cleanup failure")
        super().remove_directory(parent, name)


class FakeSolver:
    def __init__(self, result_root: Path) -> None:
        self.result_root = result_root
        self.calls = 0

    def solve(self) -> Mapping[str, Any]:
        rows = sorted(
            path
            for path in self.result_root.glob("access/*.json")
            if not path.name.endswith(".sha256.json")
        )
        assert rows
        assert strict_json_file(rows[-1])["recorded_before_operation"]
        self.calls += 1
        return {"solver": "fake", "value": 7}


class FakeLabelBackend:
    def __init__(self, result_root: Path) -> None:
        self.result_root = result_root
        self.calls = 0

    def labels(self) -> Mapping[str, Any]:
        rows = sorted(
            path
            for path in self.result_root.glob("access/*.json")
            if not path.name.endswith(".sha256.json")
        )
        assert rows
        assert strict_json_file(rows[-1])["kind"] == "label"
        self.calls += 1
        return {"backend": "fake", "labels": [0, 1]}


def attestation(attempt: int = 1) -> ExecutionAttestation:
    return ExecutionAttestation(
        attempt=attempt,
        commit="a" * 40,
        command="python -m fake_experiment",
        host="test-host",
        device="fake-device",
        clean_worktree=True,
    )


def full_lock_payload() -> dict[str, Any]:
    return {
        "schema_version": "cft-revival.experiment-execution-lock/1.0.0",
        "experiment_id": "test",
        "producer_id": "producer.py:run",
        "attempt": 1,
        "commit": "a" * 40,
        "command": "python fake.py",
        "host": "test-host",
        "device": "fake-device",
        "clean_worktree_attested": True,
        "acquired_at_utc": datetime(2026, 9, 2, tzinfo=timezone.utc),
        "immutable": True,
    }


def build_runtime(
    tmp_path: Path,
    *,
    ops: FileOps | None = None,
    attempt: int = 1,
) -> ExperimentRuntime:
    modern = Path(__file__).resolve().parents[2]
    return ExperimentRuntime(
        experiment_id="test-experiment",
        result_root=tmp_path / "results",
        cache_root=tmp_path / "working-cache",
        attestation=attestation(attempt),
        producer=runtime_producer,
        source_root=modern,
        ops=ops,
    )


def accepted_callbacks(
    result_root: Path,
    *,
    development_accepts: bool = True,
    assessment_accepts: bool = True,
) -> tuple[RuntimeCallbacks, FakeSolver, FakeLabelBackend]:
    solver = FakeSolver(result_root)
    labels = FakeLabelBackend(result_root)

    def prebundle(context: Any) -> Mapping[str, Any]:
        context.write_json("artifacts/protocol-copy.json", {"protocol": "fake-v1"})
        return {"ready": True}

    def development(context: Any) -> Decision:
        context.before_expensive("development-solve", kind="solver", details={"case": "dev"})
        solved = solver.solve()
        context.write_json("artifacts/development-result.json", solved)
        return Decision(development_accepts, {"gate": "development"})

    def assessment(context: Any) -> Decision:
        context.before_expensive(
            "held-out-labels",
            kind="label",
            details={"partition": "held-out"},
        )
        observed = labels.labels()
        context.write_json("artifacts/assessment-result.json", observed)
        context.write_transcript("transcripts/fake-backend.stdout", b"fake backend complete\r\n")
        return Decision(assessment_accepts, {"gate": "assessment"})

    return RuntimeCallbacks(prebundle, development, assessment), solver, labels


def test_accepted_runtime_records_access_before_fake_solver_and_label(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    callbacks, solver, labels = accepted_callbacks(tmp_path / "results")
    outcome = runtime.run(callbacks)
    assert outcome.state is BundleState.ACCEPTED_RESULT
    assert solver.calls == 1
    assert labels.calls == 1
    assert not (tmp_path / "working-cache").exists()
    manifest = validate_bundle(tmp_path / "results")
    assert manifest["state"] == "accepted_result"
    paths = {item["path"] for item in manifest["artifacts"]}
    assert "execution-lock.json" in paths
    assert "transcripts/fake-backend.stdout" in paths
    assert "transcripts/fake-backend.stdout.sha256.json" in paths
    transcript = next(
        item for item in manifest["artifacts"] if item["path"] == "transcripts/fake-backend.stdout"
    )
    assert transcript["byte_sha256"]
    assert transcript["sidecar"] == "transcripts/fake-backend.stdout.sha256.json"


@pytest.mark.parametrize(
    ("expected", "prebundle", "development", "assessment"),
    [
        (
            BundleState.PREBUNDLE_FAILURE,
            lambda _context: (_ for _ in ()).throw(RuntimeError("prebundle failed")),
            lambda _context: Decision(True),
            lambda _context: Decision(True),
        ),
        (
            BundleState.RUNTIME_FAILURE,
            lambda _context: {},
            lambda _context: (_ for _ in ()).throw(RuntimeError("runtime failed")),
            lambda _context: Decision(True),
        ),
        (
            BundleState.DEVELOPMENT_REJECTION,
            lambda _context: {},
            lambda _context: Decision(False, {"reason": "development gate"}),
            lambda _context: Decision(True),
        ),
        (
            BundleState.ASSESSMENT_REJECTION,
            lambda _context: {},
            lambda _context: Decision(True),
            lambda _context: Decision(False, {"reason": "assessment gate"}),
        ),
        (
            BundleState.ACCEPTED_RESULT,
            lambda _context: {},
            lambda _context: Decision(True),
            lambda _context: Decision(True),
        ),
    ],
)
def test_all_first_class_bundle_states_validate_without_success_assumptions(
    tmp_path: Path,
    expected: BundleState,
    prebundle: Callable[[Any], Mapping[str, Any]],
    development: Callable[[Any], Decision],
    assessment: Callable[[Any], Decision],
) -> None:
    outcome = build_runtime(tmp_path).run(
        RuntimeCallbacks(prebundle, development, assessment)
    )
    assert outcome.state is expected
    manifest = validate_bundle(tmp_path / "results")
    assert manifest["state"] == expected.value
    terminal = strict_json_file(tmp_path / "results" / "terminal.json")
    if expected in (BundleState.PREBUNDLE_FAILURE, BundleState.RUNTIME_FAILURE):
        assert terminal["primary_error"]
    else:
        assert terminal["primary_error"] is None
    if expected in (
        BundleState.PREBUNDLE_FAILURE,
        BundleState.RUNTIME_FAILURE,
        BundleState.DEVELOPMENT_REJECTION,
    ):
        assert terminal["counts"]["assessment_access_count"] == 0
    else:
        assert terminal["counts"]["assessment_access_count"] == 1


def test_cleanup_failure_preserves_primary_and_records_secondary(tmp_path: Path) -> None:
    ops = RuntimeFaultOps("cleanup", "working-cache")
    runtime = build_runtime(tmp_path, ops=ops)

    def fail_development(_context: Any) -> Decision:
        raise ValueError("original solver failure")

    outcome = runtime.run(
        RuntimeCallbacks(lambda _context: {}, fail_development, lambda _context: Decision(True))
    )
    assert outcome.state is BundleState.RUNTIME_FAILURE
    assert outcome.primary_error == {"type": "ValueError", "message": "original solver failure"}
    assert outcome.secondary_errors == (
        {"type": "PermissionError", "message": "injected cleanup failure"},
    )
    assert (tmp_path / "results" / "secondary-failures" / "0001-cache-cleanup.json").is_file()
    validate_bundle(tmp_path / "results")


def test_cleanup_failure_without_primary_becomes_runtime_failure(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path, ops=RuntimeFaultOps("cleanup", "working-cache"))
    callbacks, _, _ = accepted_callbacks(tmp_path / "results")
    outcome = runtime.run(callbacks)
    assert outcome.state is BundleState.RUNTIME_FAILURE
    assert outcome.primary_error == {
        "type": "PermissionError",
        "message": "injected cleanup failure",
    }
    assert outcome.secondary_errors == ()
    validate_bundle(tmp_path / "results")


def test_cleanup_failure_overrides_rejection_but_preserves_rejection_payload(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(tmp_path, ops=RuntimeFaultOps("cleanup", "working-cache"))
    outcome = runtime.run(
        RuntimeCallbacks(
            lambda _context: {},
            lambda _context: Decision(False, {"reason": "development gate"}),
            lambda _context: Decision(True),
        )
    )
    assert outcome.state is BundleState.RUNTIME_FAILURE
    terminal = strict_json_file(tmp_path / "results" / "terminal.json")
    assert terminal["payload"] == {"reason": "development gate"}
    assert terminal["primary_error"]["type"] == "PermissionError"
    validate_bundle(tmp_path / "results")


def test_hard_crash_after_artifact_write_leaves_lock_and_no_false_manifest(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)

    def crash(context: Any) -> Decision:
        context.write_json("artifacts/before-crash.json", {"durable": True})
        raise HardCrash("simulated process death")

    with pytest.raises(HardCrash, match="process death"):
        runtime.run(RuntimeCallbacks(lambda _context: {}, crash, lambda _context: Decision(True)))
    assert (tmp_path / "results" / "execution-lock.json").is_file()
    assert not (tmp_path / "results" / "manifest.json").exists()
    assert "incomplete-no-manifest" in diagnose_bundle(tmp_path / "results")
    assert not (tmp_path / "working-cache").exists()
    moved = tmp_path / "results-after-crash"
    os.replace(tmp_path / "results", moved)
    os.replace(moved, tmp_path / "results")


def test_manifest_replace_failure_keeps_bundle_incomplete_and_lock_retained(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(tmp_path, ops=RuntimeFaultOps("replace", "manifest.json"))
    callbacks, _, _ = accepted_callbacks(tmp_path / "results")
    with pytest.raises(OSError, match="replace"):
        runtime.run(callbacks)
    assert (tmp_path / "results" / "execution-lock.json").is_file()
    assert not (tmp_path / "results" / "manifest.json").exists()
    diagnoses = diagnose_bundle(tmp_path / "results")
    assert "incomplete-no-manifest" in diagnoses
    assert "stale-temp" in diagnoses


def test_lock_fsync_failure_retains_fail_closed_lock(tmp_path: Path) -> None:
    root = preflight_result_root(tmp_path / "results")
    ops = RuntimeFaultOps("fsync-file", "execution-lock.json")
    payload = full_lock_payload()
    with pytest.raises(OSError, match="fsync"):
        acquire_execution_lock(root, payload, ops=ops)
    assert (root / "execution-lock.json").exists()
    with pytest.raises(ExistingLockError):
        acquire_execution_lock(root, payload)


def test_concurrent_lock_attempts_have_exactly_one_winner(tmp_path: Path) -> None:
    root = preflight_result_root(tmp_path / "results")
    payload = {**full_lock_payload(), "experiment_id": "concurrent"}

    def attempt() -> str:
        try:
            acquire_execution_lock(root, payload)
            return "won"
        except ExistingLockError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ["blocked", "won"]


def test_finalized_bundle_is_immutable_and_rerun_is_blocked(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    callbacks, _, _ = accepted_callbacks(tmp_path / "results")
    runtime.run(callbacks)
    before = (tmp_path / "results" / "execution-lock.json").read_bytes()
    with pytest.raises(FilesystemSafetyError):
        build_runtime(tmp_path, attempt=2).run(callbacks)
    assert (tmp_path / "results" / "execution-lock.json").read_bytes() == before


def test_tampering_partial_pairs_inventory_and_state_fail_validation(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    callbacks, _, _ = accepted_callbacks(tmp_path / "results")
    runtime.run(callbacks)
    artifact = tmp_path / "results" / "artifacts" / "development-result.json"
    artifact.write_bytes(b'{"changed":true}')
    with pytest.raises(Exception, match="sidecar|inventory|corrupt"):
        validate_bundle(tmp_path / "results")


def test_transcript_lf_and_crlf_are_distinct_verified_blobs(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)

    def development(context: Any) -> Decision:
        lf = context.write_transcript("transcripts/lf.log", b"a\nb\n")
        crlf = context.write_transcript("transcripts/crlf.log", b"a\r\nb\r\n")
        assert lf["byte_sha256"] != crlf["byte_sha256"]
        return Decision(False, {"transcripts_included": True})

    outcome = runtime.run(
        RuntimeCallbacks(lambda _context: {}, development, lambda _context: Decision(True))
    )
    assert outcome.state is BundleState.DEVELOPMENT_REJECTION
    validate_bundle(tmp_path / "results")
