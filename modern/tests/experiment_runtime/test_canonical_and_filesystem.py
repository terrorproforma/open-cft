from __future__ import annotations

import errno
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

import pytest

import cft_revival.experiment_runtime.filesystem as filesystem_module

from cft_revival.experiment_runtime import (
    AtomicArtifactStore,
    CanonicalizationError,
    ExistingLockError,
    FileOps,
    FilesystemSafetyError,
    ManagedCache,
    RootPolicy,
    acquire_execution_lock,
    canonical_bytes,
    diagnose_bundle,
    preflight_result_root,
    producer_id,
    semantic_sha256,
    strict_json_loads,
    verify_pair,
)


class Mode(Enum):
    TEST = "test"


class NumericMode(IntEnum):
    TEST = 1


@dataclass(frozen=True)
class TypedRecord:
    when: datetime
    path: Path
    blob: bytes
    mode: Mode


def stable_test_producer() -> None:
    pass


class FaultOps(FileOps):
    def __init__(self, fault: str | None = None, contains: str = "") -> None:
        self.fault = fault
        self.contains = contains
        self.paths: dict[int, Path] = {}
        self.short_writes = 0

    def _matches(self, operation: str, path: Path) -> bool:
        return self.fault == operation and self.contains in str(path)

    def mkdir_child(self, parent: Any, name: str) -> Any:
        path = parent.path / name
        if self._matches("mkdir", path):
            raise PermissionError("injected mkdir denial")
        return super().mkdir_child(parent, name)

    def open_exclusive(self, parent: Any, name: str, mode: int = 0o600) -> int:
        path = parent.path / name
        if self._matches("create", path):
            raise PermissionError("injected create denial")
        descriptor = super().open_exclusive(parent, name, mode)
        self.paths[descriptor] = path
        return descriptor

    def open_temporary(self, parent: Any, name: str, mode: int = 0o600) -> int:
        path = parent.path / name
        if self._matches("create", path):
            raise PermissionError("injected create denial")
        descriptor = super().open_temporary(parent, name, mode)
        self.paths[descriptor] = path
        return descriptor

    def write(self, descriptor: int, data: memoryview) -> int:
        path = self.paths[descriptor]
        if self._matches("write", path):
            raise OSError("injected write failure")
        if self._matches("disk-full", path):
            raise OSError(errno.ENOSPC, "injected disk full")
        if self._matches("zero-write", path):
            return 0
        if self._matches("short-write", path) and len(data) > 1:
            self.short_writes += 1
            return super().write(descriptor, data[:1])
        return super().write(descriptor, data)

    def fsync_file(self, descriptor: int) -> None:
        path = self.paths[descriptor]
        if self._matches("fsync-file", path):
            raise OSError("injected file fsync failure")
        super().fsync_file(descriptor)

    def publish(self, parent: Any, descriptor: int, source: str, target: str) -> None:
        target_path = parent.path / target
        if self._matches("replace", target_path):
            raise OSError("injected replace failure")
        super().publish(parent, descriptor, source, target)

    def unlink(self, parent: Any, name: str) -> None:
        path = parent.path / name
        if self._matches("delete", path):
            raise PermissionError("injected delete denial")
        super().unlink(parent, name)

    def fsync_directory(self, directory: Any) -> bool:
        path = directory.path
        if self._matches("fsync-directory", path):
            raise OSError("injected directory fsync failure")
        return super().fsync_directory(directory)

    def remove_file(self, parent: Any, name: str) -> None:
        path = parent.path / name
        if self._matches("cleanup", path):
            raise PermissionError("injected cache cleanup denial")
        super().remove_file(parent, name)

    def remove_directory(self, parent: Any, name: str) -> None:
        path = parent.path / name
        if self._matches("cleanup", path):
            raise PermissionError("injected cache cleanup denial")
        super().remove_directory(parent, name)


def lock_payload(commit: str = "a" * 40, attempt: int = 1) -> dict[str, Any]:
    return {
        "schema_version": "cft-revival.experiment-execution-lock/1.0.0",
        "experiment_id": "matrix",
        "producer_id": "test.py:producer",
        "attempt": attempt,
        "commit": commit,
        "command": "python fake.py",
        "host": "test-host",
        "device": "fake-device",
        "clean_worktree_attested": True,
        "acquired_at_utc": datetime(2026, 9, 2, tzinfo=timezone.utc),
        "immutable": True,
    }


def test_typed_canonical_bytes_cover_time_path_bytes_enum_and_dataclass() -> None:
    local = timezone(timedelta(hours=10))
    value = TypedRecord(
        datetime(2026, 9, 2, 10, 30, 1, 7, tzinfo=local),
        Path("artifacts") / "result.json",
        b"\x00\xff",
        Mode.TEST,
    )
    first = canonical_bytes(value)
    second = canonical_bytes(value)
    assert first == second
    assert semantic_sha256(value) == hashlib.sha256(first).hexdigest()
    text = first.decode("utf-8")
    assert "2026-09-02T00:30:01.000007Z" in text
    assert "relative-posix-path" in text
    assert "bytes-base64" in text
    encoded_numeric = canonical_bytes(NumericMode.TEST)
    assert b'"__cft_type__":"enum"' in encoded_numeric
    assert b"NumericMode" in encoded_numeric


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        datetime(2026, 9, 2),
        Path("/absolute"),
        Path("a/../b"),
        Path("NUL.txt"),
        Path("trailing."),
        {1: "non-string key"},
        {"__cft_type__": "collision"},
        {1, 2},
        2**63,
        -(2**63) - 1,
    ],
)
def test_canonical_policy_rejects_ambiguous_values(value: Any) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_bytes(value)


def test_strict_json_and_lf_crlf_checkout_forms_have_one_semantic_identity() -> None:
    lf = b'{\n"a":1,\n"b":2\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    assert canonical_bytes(strict_json_loads(lf)) == canonical_bytes(strict_json_loads(crlf))
    with pytest.raises(CanonicalizationError, match="duplicate"):
        strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(CanonicalizationError, match="non-finite"):
        strict_json_loads('{"a":NaN}')


def test_producer_identity_is_relative_file_and_qualname() -> None:
    modern = Path(__file__).resolve().parents[2]
    identity = producer_id(stable_test_producer, modern)
    expected = (
        "tests/experiment_runtime/"
        "test_canonical_and_filesystem.py:stable_test_producer"
    )
    assert identity == expected
    with pytest.raises(CanonicalizationError):
        producer_id(lambda: None, modern)


def test_preflight_creates_nested_missing_parents_and_leaves_no_probe(tmp_path: Path) -> None:
    root = tmp_path / "a" / "b" / "results"
    assert preflight_result_root(root) == root.absolute()
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_preflight_accepts_only_exact_approved_placeholder(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "README.md").write_bytes(b"reserved\n")
    policy = RootPolicy({"README.md": b"reserved\n"}, allow_empty_existing=False)
    preflight_result_root(root, policy=policy)
    assert [item.name for item in root.iterdir()] == ["README.md"]
    (root / "README.md").write_bytes(b"changed\n")
    with pytest.raises(FilesystemSafetyError, match="unknown or partial"):
        preflight_result_root(root, policy=policy)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("manifest.json", b"{}", "terminal"),
        ("execution-lock.json", b"{}", "stale execution lock"),
        (".result.json.atomic.tmp", b"x", "stale temporary"),
        ("result.json.sha256.json", b"{}", "partial artifact pair"),
        ("unknown.bin", b"x", "unknown or partial"),
    ],
)
def test_preflight_rejects_terminal_stale_partial_and_unknown_states(
    tmp_path: Path, name: str, content: bytes, message: str
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / name).write_bytes(content)
    with pytest.raises(FilesystemSafetyError, match=message):
        preflight_result_root(root)


def test_preflight_rejects_file_and_link_targets(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.write_bytes(b"x")
    with pytest.raises(FilesystemSafetyError, match="occupied by file"):
        preflight_result_root(occupied)
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks require privileges on this Windows host")
    with pytest.raises(FilesystemSafetyError, match="symlink|reparse"):
        preflight_result_root(linked)


@pytest.mark.parametrize("injected_kind", ["reparse", "unknown"])
def test_preflight_rejects_injected_reparse_and_unknown_target_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_kind: str,
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    original = filesystem_module._kind
    monkeypatch.setattr(
        filesystem_module,
        "_kind",
        lambda path: injected_kind if path == root.absolute() else original(path),
    )
    with pytest.raises(FilesystemSafetyError, match=injected_kind):
        preflight_result_root(root)


@pytest.mark.parametrize(
    ("fault", "contains"),
    [
        ("mkdir", "results"),
        ("create", "write-probe"),
        ("write", "write-probe"),
        ("zero-write", "write-probe"),
        ("disk-full", "write-probe"),
        ("fsync-file", "write-probe"),
        ("delete", "write-probe"),
        ("fsync-directory", "results"),
    ],
)
def test_preflight_injected_create_write_disk_fsync_delete_failures(
    tmp_path: Path, fault: str, contains: str
) -> None:
    root = tmp_path / "results"
    with pytest.raises(FilesystemSafetyError):
        preflight_result_root(root, ops=FaultOps(fault, contains))


def test_short_writes_are_completed_not_silently_truncated(tmp_path: Path) -> None:
    root = tmp_path / "results"
    ops = FaultOps("short-write", "write-probe")
    preflight_result_root(root, ops=ops)
    assert ops.short_writes > 0
    assert list(root.iterdir()) == []


def test_atomic_pair_uses_canonical_bytes_and_valid_sidecar(tmp_path: Path) -> None:
    root = preflight_result_root(tmp_path / "results")
    store = AtomicArtifactStore(root)
    contract = store.write_json("nested/result.json", {"b": 2, "a": 1})
    artifact = root / "nested" / "result.json"
    assert artifact.read_bytes() == b'{"a":1,"b":2}'
    assert verify_pair(root, artifact) == contract
    assert contract["byte_sha256"] == contract["semantic_sha256"]


def test_atomic_store_detects_stale_temp_and_partial_pair(tmp_path: Path) -> None:
    root = preflight_result_root(tmp_path / "results")
    store = AtomicArtifactStore(root)
    stale = root / ".result.json.atomic.tmp"
    stale.write_bytes(b"old")
    with pytest.raises(FilesystemSafetyError, match="occupied or stale"):
        store.write_json("result.json", {"ok": True})
    stale.unlink()
    ops = FaultOps("replace", "sha256")
    broken = AtomicArtifactStore(root, ops)
    with pytest.raises(OSError, match="replace"):
        broken.write_json("partial.json", {"ok": True})
    assert "partial-pair:partial.json" in diagnose_bundle(root)


def test_atomic_directory_fsync_failure_leaves_diagnosable_incomplete_pair(
    tmp_path: Path,
) -> None:
    root = preflight_result_root(tmp_path / "results")
    store = AtomicArtifactStore(root, FaultOps("fsync-directory", "nested"))
    with pytest.raises(OSError, match="directory fsync"):
        store.write_json("nested/result.json", {"ok": True})
    diagnoses = diagnose_bundle(root)
    assert "partial-pair:nested/result.json" in diagnoses
    assert "stale-temp" in diagnoses


@pytest.mark.parametrize("fault", ["create", "write", "zero-write", "disk-full", "fsync-file"])
def test_atomic_write_failures_leave_no_false_completion(
    tmp_path: Path, fault: str
) -> None:
    root = preflight_result_root(tmp_path / "results")
    store = AtomicArtifactStore(root, FaultOps(fault, ".result.json.atomic.tmp"))
    with pytest.raises(OSError):
        store.write_json("result.json", {"ok": True})
    assert not (root / "manifest.json").exists()


def test_same_different_and_malformed_existing_locks_fail_closed(tmp_path: Path) -> None:
    root = preflight_result_root(tmp_path / "same")
    acquire_execution_lock(root, lock_payload())
    with pytest.raises(ExistingLockError) as same:
        acquire_execution_lock(root, lock_payload())
    assert same.value.classification == "same-attempt"

    different = preflight_result_root(tmp_path / "different")
    acquire_execution_lock(different, lock_payload())
    with pytest.raises(ExistingLockError) as other:
        acquire_execution_lock(different, lock_payload("b" * 40, 2))
    assert other.value.classification == "different-attempt"

    malformed = preflight_result_root(tmp_path / "malformed")
    (malformed / "execution-lock.json").write_bytes(b"{")
    with pytest.raises(ExistingLockError) as bad:
        acquire_execution_lock(malformed, lock_payload())
    assert bad.value.classification == "malformed"


def test_managed_cache_absent_empty_populated_and_corrupt_states(tmp_path: Path) -> None:
    absent = ManagedCache(tmp_path / "absent", "run")
    assert absent.prepare() == "absent_created"
    (absent.root / "payload.bin").write_bytes(b"x")
    assert absent.prepare() == "populated_reused"
    absent.cleanup()
    assert not absent.root.exists()

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty = ManagedCache(empty_root, "run")
    assert empty.prepare() == "empty_initialized"
    empty.cleanup()

    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    (corrupt_root / "unknown").write_bytes(b"x")
    with pytest.raises(FilesystemSafetyError, match="no marker"):
        ManagedCache(corrupt_root, "run").prepare()

    marker_root = tmp_path / "bad-marker"
    marker_root.mkdir()
    (marker_root / ".experiment-cache.json").write_bytes(b"{")
    with pytest.raises(FilesystemSafetyError, match="corrupt"):
        ManagedCache(marker_root, "run").prepare()


def test_managed_cache_rejects_injected_reparse_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = ManagedCache(tmp_path / "cache", "run")
    assert cache.prepare() == "absent_created"
    payload = cache.root / "payload.bin"
    payload.write_bytes(b"x")
    assert cache.safe_root is not None
    original = filesystem_module.PinnedDirectory.list_entries

    def injected_entries(directory: Any) -> list[tuple[str, str]]:
        return [
            (
                name,
                "reparse"
                if directory.path == cache.root and name == payload.name
                else kind,
            )
            for name, kind in original(directory)
        ]

    monkeypatch.setattr(
        filesystem_module.PinnedDirectory,
        "list_entries",
        injected_entries,
    )
    with pytest.raises(FilesystemSafetyError, match="unsafe entr"):
        cache.prepare()
