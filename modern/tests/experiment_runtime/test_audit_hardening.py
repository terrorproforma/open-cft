from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import errno
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import pytest

import cft_revival.experiment_runtime.filesystem as filesystem_module
import cft_revival.experiment_runtime.lifecycle as lifecycle_module
import cft_revival.experiment_runtime.platformfs as platformfs_module
from cft_revival.experiment_runtime import (
    AtomicArtifactStore,
    Decision,
    ExecutionAttestation,
    ExistingLockError,
    ExperimentRuntime,
    FileOps,
    FilesystemSafetyError,
    LifecycleError,
    ManagedCache,
    RuntimeCallbacks,
    acquire_execution_lock,
    canonical_bytes,
    diagnose_bundle,
    pin_existing_root,
    preflight_result_root,
    producer_id,
    semantic_sha256,
    validate_bundle,
    verify_pair,
)
from cft_revival.experiment_runtime.canonical import CanonicalizationError
from cft_revival.experiment_runtime.contracts import validate_lock
from cft_revival.experiment_runtime.platformfs import PlatformFilesystemError


def audit_producer() -> None:
    pass


def _attestation() -> ExecutionAttestation:
    return ExecutionAttestation(
        attempt=1,
        commit="c" * 40,
        command="python -m fake_audit_experiment",
        device="fake-device",
        clean_worktree=True,
        host="audit-host",
    )


def _runtime(tmp_path: Path, ops: FileOps | None = None) -> ExperimentRuntime:
    return ExperimentRuntime(
        experiment_id="audit-experiment",
        result_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        attestation=_attestation(),
        producer=audit_producer,
        source_root=Path(__file__).resolve().parents[2],
        ops=ops,
    )


def _callbacks() -> RuntimeCallbacks:
    return RuntimeCallbacks(
        lambda _context: {"ready": True},
        lambda _context: Decision(True, {"development": "passed"}),
        lambda _context: Decision(True, {"assessment": "passed"}),
    )


def _lock_payload(experiment_id: str = "process-lock") -> dict[str, Any]:
    return {
        "schema_version": "cft-revival.experiment-execution-lock/1.0.0",
        "experiment_id": experiment_id,
        "producer_id": "tests/worker.py:run",
        "attempt": 1,
        "commit": "d" * 40,
        "command": "python fake.py",
        "host": "process-host",
        "device": "fake-device",
        "clean_worktree_attested": True,
        "acquired_at_utc": datetime(2026, 9, 2, tzinfo=timezone.utc),
        "immutable": True,
    }


def _process_lock_worker(
    root: str,
    start: Any,
    output: Any,
) -> None:
    start.wait()
    try:
        acquire_execution_lock(Path(root), _lock_payload())
        output.put("won")
    except ExistingLockError:
        output.put("blocked")


def _crash_after_lock(root: str) -> None:
    acquire_execution_lock(Path(root), _lock_payload("crash-lock"))
    os._exit(23)


def _full_runtime_process_worker(
    base: str,
    worker_id: int,
    start: Any,
    output: Any,
) -> None:
    root = Path(base)
    runtime = ExperimentRuntime(
        experiment_id="process-runtime",
        result_root=root / "results",
        cache_root=root / f"cache-{worker_id}",
        attestation=_attestation(),
        producer=_full_runtime_process_worker,
        source_root=Path(__file__).resolve().parents[2],
    )
    start.wait()
    try:
        outcome = runtime.run(_callbacks())
        output.put(outcome.state.value)
    except (ExistingLockError, FilesystemSafetyError):
        output.put("blocked")


class RecordingOps(FileOps):
    def __init__(self) -> None:
        self.calls: dict[str, int] = {
            "temporary": 0,
            "file_flush": 0,
            "publish": 0,
            "directory_flush": 0,
        }

    def open_temporary(self, parent: Any, name: str, mode: int = 0o600) -> int:
        self.calls["temporary"] += 1
        return super().open_temporary(parent, name, mode)

    def fsync_file(self, descriptor: int) -> None:
        self.calls["file_flush"] += 1
        super().fsync_file(descriptor)

    def publish(self, parent: Any, descriptor: int, source: str, target: str) -> None:
        self.calls["publish"] += 1
        super().publish(parent, descriptor, source, target)

    def fsync_directory(self, directory: Any) -> bool:
        self.calls["directory_flush"] += 1
        return super().fsync_directory(directory)


class SwapRootOnTemporaryOps(FileOps):
    def __init__(self, root: Path, moved: Path) -> None:
        self.root = root
        self.moved = moved
        self.swapped = False

    def open_temporary(self, parent: Any, name: str, mode: int = 0o600) -> int:
        if not self.swapped:
            self.swapped = True
            os.replace(self.root, self.moved)
            self.root.mkdir()
        return super().open_temporary(parent, name, mode)


class LifecycleFaultOps(FileOps):
    def __init__(self, fault: str, contains: str) -> None:
        self.fault = fault
        self.contains = contains
        self.fired = False
        self.paths: dict[int, Path] = {}
        self.short_writes = 0

    def _matches(self, operation: str, path: Path) -> bool:
        return (
            not self.fired
            and self.fault == operation
            and self.contains in str(path)
        )

    def mkdir_child(self, parent: Any, name: str) -> Any:
        path = parent.path / name
        if self._matches("mkdir", path):
            self.fired = True
            raise PermissionError("injected lifecycle mkdir denial")
        return super().mkdir_child(parent, name)

    def open_exclusive(self, parent: Any, name: str, mode: int = 0o600) -> int:
        path = parent.path / name
        if self._matches("exclusive-create", path):
            self.fired = True
            raise PermissionError("injected lifecycle exclusive-create denial")
        descriptor = super().open_exclusive(parent, name, mode)
        self.paths[descriptor] = path
        return descriptor

    def open_temporary(self, parent: Any, name: str, mode: int = 0o600) -> int:
        path = parent.path / name
        if self._matches("temp-create", path):
            self.fired = True
            raise PermissionError("injected lifecycle temporary-create denial")
        descriptor = super().open_temporary(parent, name, mode)
        self.paths[descriptor] = path
        return descriptor

    def write(self, descriptor: int, data: memoryview) -> int:
        path = self.paths[descriptor]
        if self._matches("write", path):
            self.fired = True
            raise OSError("injected lifecycle write failure")
        if self._matches("zero-write", path):
            self.fired = True
            return 0
        if self._matches("disk-full", path):
            self.fired = True
            raise OSError(errno.ENOSPC, "injected lifecycle disk full")
        if self.fault == "short-write" and self.contains in str(path) and len(data) > 1:
            self.short_writes += 1
            return super().write(descriptor, data[:1])
        return super().write(descriptor, data)

    def fsync_file(self, descriptor: int) -> None:
        path = self.paths[descriptor]
        if self._matches("file-flush", path):
            self.fired = True
            raise OSError("injected lifecycle file flush failure")
        super().fsync_file(descriptor)

    def publish(self, parent: Any, descriptor: int, source: str, target: str) -> None:
        path = parent.path / target
        if self._matches("publish", path):
            self.fired = True
            raise OSError("injected lifecycle publication failure")
        super().publish(parent, descriptor, source, target)

    def unlink(self, parent: Any, name: str) -> None:
        path = parent.path / name
        if self._matches("unlink", path):
            self.fired = True
            raise PermissionError("injected lifecycle unlink denial")
        super().unlink(parent, name)

    def fsync_directory(self, directory: Any) -> bool:
        if self._matches("directory-flush", directory.path):
            self.fired = True
            raise OSError("injected lifecycle directory flush failure")
        return super().fsync_directory(directory)


def test_pinned_root_blocks_or_detects_directory_swap(tmp_path: Path) -> None:
    safe = preflight_result_root(tmp_path / "results")
    moved = tmp_path / "moved"
    try:
        os.replace(safe.path, moved)
    except OSError:
        safe.verify()
        safe.close()
    else:
        safe.path.mkdir()
        with pytest.raises(PlatformFilesystemError):
            safe.verify()
        safe.path.rmdir()
        safe.close()
        os.replace(moved, safe.path)


def test_actual_windows_junction_is_rejected_when_available(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction test")
    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    completed = subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(junction), str(target)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        pytest.skip("junction creation unavailable")
    try:
        with pytest.raises(FilesystemSafetyError, match="reparse"):
            preflight_result_root(junction)
    finally:
        os.rmdir(junction)


def test_actual_inner_junction_blocks_candidate_manifest(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction inventory test")
    target = tmp_path / "outside"
    target.mkdir()
    probe = tmp_path / "junction-probe"
    available = subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(probe), str(target)),
        capture_output=True,
        text=True,
        check=False,
    )
    if available.returncode:
        pytest.skip("junction creation unavailable")
    os.rmdir(probe)
    junction = tmp_path / "results" / "rogue-junction"

    def development(_context: Any) -> Decision:
        created = subprocess.run(
            ("cmd", "/c", "mklink", "/J", str(junction), str(target)),
            capture_output=True,
            text=True,
            check=False,
        )
        assert created.returncode == 0
        return Decision(True)

    callbacks = RuntimeCallbacks(
        lambda _context: {},
        development,
        lambda _context: Decision(True),
    )
    with pytest.raises(FilesystemSafetyError, match="reparse"):
        _runtime(tmp_path).run(callbacks)
    assert not (tmp_path / "results" / "manifest.json").exists()
    os.rmdir(junction)


def test_injected_handle_identity_swap_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe = preflight_result_root(tmp_path / "results")
    store = AtomicArtifactStore(safe)
    original = safe.root.verify
    calls = 0

    def swapped() -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise PlatformFilesystemError("injected handle identity swap")
        original()

    monkeypatch.setattr(safe.root, "verify", swapped)
    with pytest.raises(PlatformFilesystemError, match="identity swap"):
        store.write_json("artifact.json", {"value": 1})
    assert not (safe / "manifest.json").exists()
    safe.close()


def test_inventory_enumerates_through_pinned_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe = preflight_result_root(tmp_path / "results")
    store = AtomicArtifactStore(safe)
    store.write_json("nested/artifact.json", {"value": 1})

    def path_enumeration_forbidden(_path: Path) -> Any:
        raise AssertionError("inventory must not enumerate through a path")

    monkeypatch.setattr(Path, "iterdir", path_enumeration_forbidden)
    entries = filesystem_module.scan_tree(safe)
    assert ("nested", "directory") in entries
    assert ("nested/artifact.json", "file") in entries
    safe.close()


def test_actual_path_swap_during_write_cannot_escape_pinned_root(tmp_path: Path) -> None:
    root = tmp_path / "results"
    moved = tmp_path / "moved"
    safe = preflight_result_root(root)
    store = AtomicArtifactStore(safe, SwapRootOnTemporaryOps(root, moved))
    with pytest.raises((OSError, PlatformFilesystemError)):
        store.write_json("artifact.json", {"value": 1})
    assert not (root / "artifact.json").exists()
    assert not (root / "manifest.json").exists()
    safe.close()
    if moved.exists():
        root.rmdir()
        os.replace(moved, root)


def test_process_concurrency_has_exactly_one_lock_winner(tmp_path: Path) -> None:
    safe = preflight_result_root(tmp_path / "results")
    root = str(safe.path)
    safe.close()
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(target=_process_lock_worker, args=(root, start, output))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        start.set()
        outcomes = [output.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    assert sorted(outcomes) == ["blocked", "won"]


def test_process_concurrency_crosses_full_runtime(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_full_runtime_process_worker,
            args=(str(tmp_path), worker_id, start, output),
        )
        for worker_id in range(2)
    ]
    for process in processes:
        process.start()
    try:
        start.set()
        outcomes = [output.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    assert sorted(outcomes) == ["accepted_result", "blocked"]
    validate_bundle(tmp_path / "results")


def test_process_crash_after_lock_has_no_manifest(tmp_path: Path) -> None:
    safe = preflight_result_root(tmp_path / "results")
    root = safe.path
    safe.close()
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_crash_after_lock, args=(str(root),))
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("crash worker did not terminate")
    assert process.exitcode == 23
    assert (root / "execution-lock.json").is_file()
    assert not (root / "manifest.json").exists()
    assert "incomplete-no-manifest" in diagnose_bundle(root)


@pytest.mark.parametrize(
    ("fault", "contains", "lock_expected"),
    [
        ("mkdir", "results", False),
        ("exclusive-create", "write-probe", False),
        ("write", "write-probe", False),
        ("zero-write", "write-probe", False),
        ("disk-full", "write-probe", False),
        ("file-flush", "write-probe", False),
        ("unlink", "write-probe", False),
        ("file-flush", "execution-lock", True),
        ("temp-create", "lock-acquired", True),
        ("write", "lock-acquired", True),
        ("zero-write", "lock-acquired", True),
        ("disk-full", "lock-acquired", True),
        ("file-flush", "lock-acquired", True),
        ("publish", "lock-acquired", True),
        ("directory-flush", "transitions", True),
    ],
)
def test_faults_cross_actual_production_lifecycle(
    tmp_path: Path,
    fault: str,
    contains: str,
    lock_expected: bool,
) -> None:
    ops = LifecycleFaultOps(fault, contains)
    try:
        outcome = _runtime(tmp_path, ops=ops).run(_callbacks())
    except (FilesystemSafetyError, LifecycleError, OSError):
        outcome = None
    assert ops.fired
    root = tmp_path / "results"
    assert (root / "execution-lock.json").exists() is lock_expected
    if outcome is not None:
        assert outcome.state.value != "accepted_result"
        validate_bundle(root)
    else:
        assert not (root / "manifest.json").exists()


def test_short_writes_cross_full_lifecycle_without_truncation(tmp_path: Path) -> None:
    ops = LifecycleFaultOps("short-write", "lock-acquired")
    outcome = _runtime(tmp_path, ops=ops).run(_callbacks())
    assert ops.short_writes > 0
    assert outcome.state.value == "accepted_result"
    validate_bundle(tmp_path / "results")


def test_candidate_manifest_rejection_is_invisible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lifecycle_module.validate_bundle

    def reject_candidate(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["manifest_override"] is not None
        root = args[0]
        assert not (root / "manifest.json").exists()
        raise LifecycleError("injected candidate semantic denial")

    monkeypatch.setattr(lifecycle_module, "validate_bundle", reject_candidate)
    with pytest.raises(LifecycleError, match="candidate semantic denial"):
        _runtime(tmp_path).run(_callbacks())
    assert (tmp_path / "results" / "execution-lock.json").is_file()
    assert not (tmp_path / "results" / "manifest.json").exists()
    monkeypatch.setattr(lifecycle_module, "validate_bundle", original)


def test_candidate_counter_semantic_denial_leaves_no_manifest(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)

    def assessment(_context: Any) -> Decision:
        runtime._counters["attempt_count"] = True
        return Decision(True)

    callbacks = RuntimeCallbacks(
        lambda _context: {},
        lambda _context: Decision(True),
        assessment,
    )
    with pytest.raises(LifecycleError, match="integer|counter"):
        runtime.run(callbacks)
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_candidate_access_nested_type_denial_leaves_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path)

    def malformed_access(value: Any) -> None:
        assert runtime.store is not None
        runtime._access_sequence += 1
        runtime.store.write_json(
            f"access/{runtime._access_sequence:04d}.json",
            {
                "schema_version": lifecycle_module.ACCESS_VERSION,
                "sequence": runtime._access_sequence,
                "recorded_at_utc": datetime.now(timezone.utc),
                "operation": value["operation"],
                "kind": value["kind"],
                "details": [],
                "recorded_before_operation": True,
            },
        )

    monkeypatch.setattr(runtime, "_write_access", malformed_access)
    with pytest.raises(LifecycleError, match="details must be an object"):
        runtime.run(_callbacks())
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_manifest_publication_has_no_post_publication_rejection_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lifecycle_module.validate_bundle
    calls: list[bool] = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("manifest_override") is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(lifecycle_module, "validate_bundle", observe)
    outcome = _runtime(tmp_path).run(_callbacks())
    assert outcome.state.value == "accepted_result"
    assert calls == [True]
    assert (tmp_path / "results" / "manifest.json").is_file()


def test_candidate_validation_holds_existing_files_deny_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lifecycle_module.validate_bundle
    denied = False

    def probe_seal(*args: Any, **kwargs: Any) -> Any:
        nonlocal denied
        terminal = args[0] / "terminal.json"
        try:
            terminal.write_bytes(b"attack")
        except OSError:
            denied = True
        return original(*args, **kwargs)

    monkeypatch.setattr(lifecycle_module, "validate_bundle", probe_seal)
    _runtime(tmp_path).run(_callbacks())
    assert denied


@pytest.mark.parametrize(
    "entry_kind",
    ["empty-directory", "unknown-file", "reparse", "special"],
)
def test_candidate_inventory_denial_leaves_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    rogue = tmp_path / "results" / "rogue"

    def development(context: Any) -> Decision:
        if entry_kind == "empty-directory":
            rogue.mkdir()
        else:
            rogue.write_bytes(b"uncontracted")
            if entry_kind in {"reparse", "special"}:
                original = context.store.safe_root.root.list_entries
                monkeypatch.setattr(
                    context.store.safe_root.root,
                    "list_entries",
                    lambda: [
                        (name, entry_kind if name == rogue.name else kind)
                        for name, kind in original()
                    ],
                )
        return Decision(True)

    callbacks = RuntimeCallbacks(lambda _context: {}, development, lambda _context: Decision(True))
    with pytest.raises((FilesystemSafetyError, LifecycleError)):
        _runtime(tmp_path).run(callbacks)
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_orphan_sidecar_full_lifecycle_leaves_no_manifest(tmp_path: Path) -> None:
    orphan = tmp_path / "results" / "orphan.json.sha256.json"

    def development(_context: Any) -> Decision:
        orphan.write_bytes(canonical_bytes({"artifact": "orphan.json"}))
        return Decision(True)

    callbacks = RuntimeCallbacks(
        lambda _context: {},
        development,
        lambda _context: Decision(True),
    )
    with pytest.raises(FilesystemSafetyError, match="orphan sidecar"):
        _runtime(tmp_path).run(callbacks)
    assert (tmp_path / "results" / "execution-lock.json").is_file()
    assert not (tmp_path / "results" / "manifest.json").exists()


@pytest.mark.parametrize(
    "violation",
    ["duplicate-reference", "mismatched-contract", "sidecar-of-sidecar"],
)
def test_sidecar_bijection_violations_fail_before_manifest(
    tmp_path: Path,
    violation: str,
) -> None:
    def development(context: Any) -> Decision:
        if violation == "sidecar-of-sidecar":
            nested = (
                tmp_path
                / "results"
                / "artifact.json.sha256.json.sha256.json"
            )
            nested.write_bytes(canonical_bytes({"artifact": "artifact.json"}))
            return Decision(True)
        context.write_json("first.json", {"value": 1})
        context.write_json("second.json", {"value": 2})
        second_sidecar = tmp_path / "results" / "second.json.sha256.json"
        contract = json.loads(second_sidecar.read_bytes())
        contract["artifact"] = (
            "first.json" if violation == "duplicate-reference" else "other.json"
        )
        second_sidecar.write_bytes(canonical_bytes(contract))
        return Decision(True)

    callbacks = RuntimeCallbacks(
        lambda _context: {},
        development,
        lambda _context: Decision(True),
    )
    expected = {
        "duplicate-reference": "duplicate sidecar",
        "mismatched-contract": "mismatched sidecar",
        "sidecar-of-sidecar": "sidecar-of-sidecar",
    }[violation]
    with pytest.raises(FilesystemSafetyError, match=expected):
        _runtime(tmp_path).run(callbacks)
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_bundle_validation_rejects_postpublication_orphan_sidecar(
    tmp_path: Path,
) -> None:
    _runtime(tmp_path).run(_callbacks())
    orphan = tmp_path / "results" / "orphan.json.sha256.json"
    orphan.write_bytes(canonical_bytes({"artifact": "orphan.json"}))
    with pytest.raises(LifecycleError, match="orphan sidecar"):
        validate_bundle(tmp_path / "results")


@pytest.mark.parametrize(
    ("result_suffix", "cache_suffix"),
    [
        (Path("tree/results"), Path("tree")),
        (Path("tree"), Path("tree/cache")),
        (Path("tree/results"), Path("tree/results")),
        (Path("tree/results"), Path("tree/other/../results/cache")),
    ],
)
def test_cache_result_overlap_is_rejected_in_both_directions(
    tmp_path: Path,
    result_suffix: Path,
    cache_suffix: Path,
) -> None:
    with pytest.raises(LifecycleError, match="overlap"):
        ExperimentRuntime(
            experiment_id="overlap",
            result_root=tmp_path / result_suffix,
            cache_root=tmp_path / cache_suffix,
            attestation=_attestation(),
            producer=audit_producer,
            source_root=Path(__file__).resolve().parents[2],
        )


def test_cache_result_junction_alias_is_rejected_when_available(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction alias test")
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    completed = subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(alias), str(real)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        pytest.skip("junction creation unavailable")
    try:
        with pytest.raises(LifecycleError, match="overlap"):
            ExperimentRuntime(
                experiment_id="junction-overlap",
                result_root=real / "results",
                cache_root=alias,
                attestation=_attestation(),
                producer=audit_producer,
                source_root=Path(__file__).resolve().parents[2],
            )
    finally:
        os.rmdir(alias)


def test_cleanup_identity_guard_cannot_target_result_tree(tmp_path: Path) -> None:
    safe = preflight_result_root(tmp_path / "results")
    cache = ManagedCache(tmp_path / "cache", "audit")
    cache.safe_root = safe
    with pytest.raises(FilesystemSafetyError, match="overlaps"):
        cache.cleanup(protected=safe)
    assert safe.path.is_dir()
    cache.safe_root = None
    safe.close()


def test_tuple_list_drive_relative_and_unstable_producer_collisions_close(
    tmp_path: Path,
) -> None:
    assert semantic_sha256((1, 2)) != semantic_sha256([1, 2])
    with pytest.raises(CanonicalizationError):
        canonical_bytes(PureWindowsPath("C:relative"))
    with pytest.raises(FilesystemSafetyError):
        filesystem_module.relative_path("C:relative")
    with pytest.raises(FilesystemSafetyError):
        filesystem_module.relative_path("mixed/path\\form")
    with pytest.raises(FilesystemSafetyError):
        filesystem_module.relative_path("NUL.txt")
    with pytest.raises(FilesystemSafetyError):
        filesystem_module.relative_path("trailing.")

    def nested() -> None:
        pass

    with pytest.raises(CanonicalizationError):
        producer_id(nested, tmp_path)
    with pytest.raises(CanonicalizationError):
        producer_id(lambda: None, tmp_path)


@pytest.mark.parametrize(
    "bad_value",
    [
        {**_lock_payload(), "attempt": True},
        {**_lock_payload(), "clean_worktree_attested": 1},
        {**_lock_payload(), "immutable": 1},
        {**_lock_payload(), "producer_id": "C:relative.py:run"},
        {**_lock_payload(), "producer_id": "/absolute.py:run"},
        {**_lock_payload(), "acquired_at_utc": "2026-09-02T00:00:00Z"},
        {
            **_lock_payload(),
            "acquired_at_utc": {
                "__cft_type__": "aware-utc-datetime",
                "value": "2026-09-02T00:00:00Z",
            },
        },
        {**_lock_payload(), "extra": "field"},
        {**_lock_payload(), "schema_version": "unsupported"},
    ],
)
def test_full_lock_schema_precedes_same_attempt_classification(
    tmp_path: Path,
    bad_value: dict[str, Any],
) -> None:
    safe = preflight_result_root(tmp_path / "results")
    if type(bad_value["acquired_at_utc"]) is dict:
        encoded = filesystem_module.canonical_bytes_from_parsed(bad_value)
    else:
        encoded = canonical_bytes(bad_value)
    (safe / "execution-lock.json").write_bytes(encoded)
    with pytest.raises(ExistingLockError) as error:
        acquire_execution_lock(safe, _lock_payload())
    assert error.value.classification == "malformed"
    safe.close()


def test_noncanonical_existing_lock_is_malformed(tmp_path: Path) -> None:
    safe = preflight_result_root(tmp_path / "results")
    encoded = json.dumps(
        json.loads(canonical_bytes(_lock_payload())),
        indent=2,
    ).encode("utf-8")
    (safe / "execution-lock.json").write_bytes(encoded)
    with pytest.raises(ExistingLockError) as error:
        acquire_execution_lock(safe, _lock_payload())
    assert error.value.classification == "malformed"
    safe.close()


def test_exact_runtime_types_reject_bool_int_confusion() -> None:
    with pytest.raises(LifecycleError, match="exact bool"):
        Decision(1)  # type: ignore[arg-type]
    with pytest.raises(LifecycleError, match="exact positive"):
        ExecutionAttestation(
            attempt=True,  # type: ignore[arg-type]
            commit="a" * 40,
            command="x",
            device="x",
            clean_worktree=True,
            host="x",
        )
    with pytest.raises(LifecycleError, match="exact true"):
        ExecutionAttestation(
            attempt=1,
            commit="a" * 40,
            command="x",
            device="x",
            clean_worktree=1,  # type: ignore[arg-type]
            host="x",
        )
    validate_lock(json.loads(canonical_bytes(_lock_payload())))


def test_empty_typed_bytes_pass_closed_candidate_validation(tmp_path: Path) -> None:
    callbacks = RuntimeCallbacks(
        lambda _context: {"empty": b""},
        lambda _context: Decision(True, {"empty": b""}),
        lambda _context: Decision(True, {"empty": b""}),
    )
    outcome = _runtime(tmp_path).run(callbacks)
    assert outcome.state.value == "accepted_result"
    validate_bundle(tmp_path / "results")


def test_sidecar_closed_schema_rejects_extra_keys(tmp_path: Path) -> None:
    safe = preflight_result_root(tmp_path / "results")
    store = AtomicArtifactStore(safe)
    store.write_json("artifact.json", {"value": 1})
    sidecar = safe / "artifact.json.sha256.json"
    value = json.loads(sidecar.read_bytes())
    value["extra"] = True
    sidecar.write_bytes(canonical_bytes(value))
    with pytest.raises(FilesystemSafetyError, match="schema"):
        verify_pair(store, "artifact.json")
    safe.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "unsupported"),
        ("manifest_is_sole_completion_marker", 1),
        ("canonicalization", "other"),
    ],
)
def test_manifest_closed_schema_rejects_tampering(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    _runtime(tmp_path).run(_callbacks())
    manifest_path = tmp_path / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest[field] = value
    manifest_path.write_bytes(canonical_bytes(manifest))
    with pytest.raises(LifecycleError, match="contract|canonicalization|completion"):
        validate_bundle(tmp_path / "results")


def test_windows_durability_calls_and_bounded_claim_are_recorded(tmp_path: Path) -> None:
    ops = RecordingOps()
    outcome = _runtime(tmp_path, ops=ops).run(_callbacks())
    assert all(count > 0 for count in ops.calls.values())
    durability = outcome.manifest["durability"]
    if os.name == "nt":
        assert durability["platform"] == "windows"
        assert "no POSIX directory-fsync equivalence" in durability["power_loss_claim"]
        assert "ERROR_INVALID_HANDLE" in durability["directory_metadata_flush"]
        assert "ERROR_ACCESS_DENIED" in durability["directory_metadata_flush"]
        assert "tolerated bounded limitations" in durability["directory_metadata_flush"]
        assert durability["directory_flush_supported"] is False
    else:
        assert durability["platform"] == "posix"
        assert durability["directory_flush_supported"] is True


def test_windows_temporary_files_use_write_through_nt_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows write-through API test")
    original = platformfs_module._win_nt_open_relative
    options: list[int] = []

    def record_options(*args: Any, **kwargs: Any) -> int:
        options.append(kwargs["options"])
        return original(*args, **kwargs)

    monkeypatch.setattr(platformfs_module, "_win_nt_open_relative", record_options)
    outcome = _runtime(tmp_path).run(_callbacks())
    assert outcome.state.value == "accepted_result"
    assert any(value & platformfs_module._FILE_WRITE_THROUGH for value in options)
    assert "FILE_WRITE_THROUGH" in outcome.manifest["durability"]["temporary_open"]


def test_manifest_binds_every_required_directory(tmp_path: Path) -> None:
    outcome = _runtime(tmp_path).run(_callbacks())
    directory_entries = [
        item["path"]
        for item in outcome.manifest["artifacts"]
        if item["type"] == "directory"
    ]
    assert directory_entries == outcome.manifest["required_directories"]
    assert directory_entries == sorted(set(directory_entries))


def test_inventory_is_globally_sorted_for_same_stem_directory_and_file(
    tmp_path: Path,
) -> None:
    def development(context: Any) -> Decision:
        context.write_json("x/nested/result.json", {"nested": True})
        context.write_json("x.json", {"sibling": True})
        return Decision(True)

    callbacks = RuntimeCallbacks(
        lambda _context: {"ready": True},
        development,
        lambda _context: Decision(True),
    )
    outcome = _runtime(tmp_path).run(callbacks)
    paths = [entry["path"] for entry in outcome.manifest["artifacts"]]

    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert paths.index("x.json") < paths.index("x/nested/result.json")
    assert validate_bundle(tmp_path / "results") == outcome.manifest


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("unsorted", "globally path-sorted"),
        ("duplicate", "duplicate paths"),
    ],
)
def test_manifest_rejects_unsorted_and_duplicate_artifact_paths(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    _runtime(tmp_path).run(_callbacks())
    manifest_path = tmp_path / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if defect == "unsorted":
        manifest["artifacts"][0], manifest["artifacts"][1] = (
            manifest["artifacts"][1],
            manifest["artifacts"][0],
        )
    else:
        manifest["artifacts"].append(dict(manifest["artifacts"][-1]))
        manifest["artifact_count"] = len(manifest["artifacts"])
    manifest_path.write_bytes(canonical_bytes(manifest))

    with pytest.raises(LifecycleError, match=message):
        validate_bundle(tmp_path / "results")


def test_duplicate_raw_inventory_path_leaves_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lifecycle_module.scan_tree

    def duplicate_entry(*args: Any, **kwargs: Any) -> list[tuple[str, str]]:
        entries = original(*args, **kwargs)
        duplicate = next(entry for entry in entries if entry[0] == "terminal.json")
        return [*entries, duplicate]

    monkeypatch.setattr(lifecycle_module, "scan_tree", duplicate_entry)
    with pytest.raises(FilesystemSafetyError, match="duplicate inventory path"):
        _runtime(tmp_path).run(_callbacks())
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_local_gitattributes_pin_fresh_checkout_line_endings() -> None:
    modern = Path(__file__).resolve().parents[2]
    source = modern / "src" / "cft_revival" / "experiment_runtime" / ".gitattributes"
    tests = modern / "tests" / "experiment_runtime" / ".gitattributes"
    specs = modern / "spec" / "experiment_runtime" / ".gitattributes"
    assert source.read_text(encoding="utf-8").splitlines() == ["*.py text eol=lf"]
    assert tests.read_text(encoding="utf-8").splitlines() == ["*.py text eol=lf"]
    assert specs.read_text(encoding="utf-8").splitlines() == ["*.json text eol=lf"]
    paths = (
        "modern/src/cft_revival/experiment_runtime/lifecycle.py",
        "modern/tests/experiment_runtime/test_audit_hardening.py",
        "modern/spec/experiment_runtime/state-machine-v1.json",
    )
    completed = subprocess.run(
        ("git", "check-attr", "eol", "--", *paths),
        cwd=modern.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert all(line.endswith(": eol: lf") for line in completed.stdout.splitlines())
