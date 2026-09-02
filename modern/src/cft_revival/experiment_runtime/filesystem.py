"""Pinned fail-closed filesystem primitives for immutable experiment bundles."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import (
    CANONICALIZATION_ID,
    WINDOWS_RESERVED_NAMES,
    canonical_bytes,
    semantic_sha256,
    strict_json_loads,
)
from .platformfs import (
    DirectoryIdentity,
    PinnedDirectory,
    PlatformFilesystemError,
    durability_contract,
    identities_overlap,
)

LOCK_NAME = "execution-lock.json"
MANIFEST_NAME = "manifest.json"
SIDECAR_SUFFIX = ".sha256.json"
TEMP_SUFFIX = ".atomic.tmp"
CACHE_MARKER = ".experiment-cache.json"


class FilesystemSafetyError(RuntimeError):
    """A filesystem state or operation cannot be proven safe."""


class PartialArtifactError(FilesystemSafetyError):
    """Only one member of a required artifact/sidecar pair exists."""


@dataclass
class SafeRoot:
    """Pinned result/cache root and its immediate pinned parent."""

    path: Path
    root: PinnedDirectory
    parent: PinnedDirectory
    directory_flush_supported: bool = True

    @property
    def identity(self) -> DirectoryIdentity:
        return self.root.identity

    def verify(self) -> None:
        self.parent.verify()
        self.root.verify()

    def close(self) -> None:
        for item in (self.root, self.parent):
            try:
                item.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __truediv__(self, other: str | Path) -> Path:
        return self.path / other

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SafeRoot):
            return self.path == other.path
        if isinstance(other, Path):
            return self.path == other
        return False

    def absolute(self) -> Path:
        return self.path

    def is_dir(self) -> bool:
        return self.path.is_dir()

    def exists(self) -> bool:
        return self.path.exists()

    def iterdir(self) -> Iterable[Path]:
        return self.path.iterdir()


class FileOps:
    """Injectable secure OS boundary used by all lifecycle mutations."""

    def pin_directory(self, path: Path) -> PinnedDirectory:
        return PinnedDirectory.open(path)

    def mkdir_child(self, parent: PinnedDirectory, name: str) -> PinnedDirectory:
        return parent.mkdir_child(name)

    def open_exclusive(
        self,
        parent: PinnedDirectory,
        name: str,
        mode: int = 0o600,
    ) -> int:
        return parent.open_file_exclusive(name, mode)

    def open_read(self, parent: PinnedDirectory, name: str) -> int:
        return parent.open_file_read(name)

    def open_temporary(
        self,
        parent: PinnedDirectory,
        name: str,
        mode: int = 0o600,
    ) -> int:
        return parent.open_temporary(name, mode)

    def write(self, descriptor: int, data: memoryview) -> int:
        return os.write(descriptor, data)

    def fsync_file(self, descriptor: int) -> None:
        os.fsync(descriptor)

    def seal_file(self, descriptor: int) -> None:
        if os.name != "nt":
            os.fchmod(descriptor, 0o444)

    def publish(
        self,
        parent: PinnedDirectory,
        descriptor: int,
        source: str,
        target: str,
    ) -> None:
        parent.publish_open_file(descriptor, source, target)

    def unlink(self, parent: PinnedDirectory, name: str) -> None:
        parent.unlink(name)

    def fsync_directory(self, directory: PinnedDirectory) -> bool:
        return directory.fsync()

    def remove_file(self, parent: PinnedDirectory, name: str) -> None:
        parent.unlink(name)

    def remove_directory(self, parent: PinnedDirectory, name: str) -> None:
        parent.remove_directory(name)


def _kind(path: Path) -> str:
    status = path.lstat()
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if path.is_symlink():
        return "symlink"
    if reparse_flag and attributes & reparse_flag:
        return "reparse"
    if stat.S_ISDIR(status.st_mode):
        return "directory"
    if stat.S_ISREG(status.st_mode):
        return "file"
    return "special"


@dataclass(frozen=True)
class RootPolicy:
    """Exact approved state for a result root which already exists."""

    approved_placeholders: Mapping[str, bytes] = field(default_factory=dict)
    approved_directories: tuple[str, ...] = ()
    allow_empty_existing: bool = True

    def __post_init__(self) -> None:
        for name in self.approved_placeholders:
            path = relative_path(name)
            if (
                path.name in {LOCK_NAME, MANIFEST_NAME}
                or path.name.endswith(SIDECAR_SUFFIX)
                or path.name.endswith(TEMP_SUFFIX)
            ):
                raise FilesystemSafetyError(
                    f"approved placeholder uses a reserved artifact name: {name}"
                )
        for name in self.approved_directories:
            relative_path(name)


def relative_path(value: str | Path) -> Path:
    text = os.fspath(value)
    path = Path(value)
    drive = getattr(path, "drive", "")
    if (
        path.is_absolute()
        or bool(drive)
        or not path.parts
        or any(
            part in ("", ".", "..")
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
            or any(ord(character) < 32 for character in part)
            for part in path.parts
        )
        or ("/" in text and "\\" in text)
        or text.endswith(("/", "\\"))
        or ":" in text
    ):
        raise FilesystemSafetyError(f"unsafe relative path: {value!s}")
    return path


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_no_link_ancestors(path: Path) -> None:
    for item in reversed((path.absolute(), *path.absolute().parents)):
        if _lexists(item) and _kind(item) != "directory":
            raise FilesystemSafetyError(f"unsafe path ancestor ({_kind(item)}): {item}")


def _pin_or_create(path: Path, ops: FileOps) -> SafeRoot:
    absolute = path.absolute()
    _assert_no_link_ancestors(absolute.parent)
    missing: list[str] = []
    current = absolute
    while not _lexists(current):
        missing.append(current.name)
        current = current.parent
    current_kind = _kind(current)
    if current_kind != "directory":
        if current == absolute:
            raise FilesystemSafetyError(
                f"result root is occupied by {current_kind}: {current}"
            )
        raise FilesystemSafetyError(
            f"nearest existing ancestor is unsafe ({current_kind}): {current}"
        )
    pin = ops.pin_directory(current)
    pins_to_close: list[PinnedDirectory] = []
    try:
        for name in reversed(missing):
            child = ops.mkdir_child(pin, name)
            pins_to_close.append(pin)
            pin = child
        if missing:
            root_pin = pin
            parent_pin = pins_to_close.pop()
        else:
            if absolute.parent == absolute:
                raise FilesystemSafetyError("filesystem root cannot be an experiment root")
            root_pin = pin
            parent_pin = ops.pin_directory(absolute.parent)
        for old in pins_to_close:
            old.close()
        safe = SafeRoot(absolute, root_pin, parent_pin)
        safe.verify()
        return safe
    except BaseException:
        pin.close()
        for old in pins_to_close:
            old.close()
        raise


def _read_all(ops: FileOps, parent: PinnedDirectory, name: str) -> bytes:
    descriptor = ops.open_read(parent, name)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _write_fully(ops: FileOps, descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = ops.write(descriptor, view[offset:])
        if type(written) is not int or written <= 0 or written > len(view) - offset:
            raise OSError("write made invalid forward progress")
        offset += written


def _exclusive_durable_write(
    ops: FileOps,
    parent: PinnedDirectory,
    name: str,
    data: bytes,
    mode: int = 0o600,
) -> None:
    descriptor = ops.open_exclusive(parent, name, mode)
    try:
        _write_fully(ops, descriptor, data)
        ops.fsync_file(descriptor)
    finally:
        os.close(descriptor)
    parent.verify()


def _open_durable_temporary(
    ops: FileOps,
    parent: PinnedDirectory,
    name: str,
    data: bytes,
    mode: int = 0o600,
) -> int:
    descriptor = ops.open_temporary(parent, name, mode)
    try:
        _write_fully(ops, descriptor, data)
        ops.fsync_file(descriptor)
        ops.seal_file(descriptor)
        parent.verify()
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _walk(
    ops: FileOps,
    directory: PinnedDirectory,
    prefix: Path = Path(),
) -> list[tuple[str, str]]:
    directory.verify()
    result: list[tuple[str, str]] = []
    for name, kind in directory.list_entries():
        relative = (prefix / name).as_posix()
        result.append((relative, kind))
        if kind == "directory":
            child = directory.open_child(name)
            try:
                result.extend(_walk(ops, child, prefix / name))
            finally:
                child.close()
    directory.verify()
    return result


def scan_tree(root: SafeRoot, ops: FileOps | None = None) -> list[tuple[str, str]]:
    selected = ops or FileOps()
    root.verify()
    return _walk(selected, root.root)


def _validate_placeholder(root: SafeRoot, policy: RootPolicy, ops: FileOps) -> None:
    entries = scan_tree(root, ops)
    if not entries:
        if policy.allow_empty_existing:
            return
        raise FilesystemSafetyError("existing empty result root is not approved")
    kinds = dict(entries)
    unsafe = {name: kind for name, kind in entries if kind not in {"file", "directory"}}
    if unsafe:
        raise FilesystemSafetyError(f"result root contains unsafe entries: {unsafe}")
    actual_directories = {name for name, kind in entries if kind == "directory"}
    if actual_directories != set(policy.approved_directories):
        raise FilesystemSafetyError("result root directories do not match approved policy")
    actual_files: dict[str, bytes] = {}
    store = AtomicArtifactStore(root, ops)
    for name, kind in entries:
        if kind == "file":
            try:
                actual_files[name] = store.read_bytes(name)
            except (OSError, PlatformFilesystemError) as error:
                raise FilesystemSafetyError(
                    f"result root changed during placeholder validation: {name}"
                ) from error
    if actual_files != dict(policy.approved_placeholders):
        names = set(actual_files)
        if MANIFEST_NAME in names:
            reason = "terminal"
        elif LOCK_NAME in names:
            reason = "stale execution lock"
        elif any(name.endswith(TEMP_SUFFIX) for name in names):
            reason = "stale temporary artifact"
        elif any(name.endswith(SIDECAR_SUFFIX) for name in names):
            reason = "partial artifact pair"
        else:
            reason = "unknown or partial content"
        raise FilesystemSafetyError(f"result root rejected: {reason}")


def preflight_result_root(
    root: Path,
    *,
    policy: RootPolicy | None = None,
    ops: FileOps | None = None,
) -> SafeRoot:
    """Pin/create the root, validate exact contents, and leave no probe."""

    selected_policy = policy or RootPolicy()
    selected_ops = ops or FileOps()
    existed = _lexists(root.absolute())
    try:
        safe = _pin_or_create(root, selected_ops)
    except Exception as error:
        if isinstance(error, FilesystemSafetyError):
            raise
        raise FilesystemSafetyError(f"cannot securely pin result root: {error}") from error
    try:
        if existed:
            _validate_placeholder(safe, selected_policy, selected_ops)
        probe = f".cft-write-probe-{uuid.uuid4().hex}"
        try:
            _exclusive_durable_write(selected_ops, safe.root, probe, b"writability")
            selected_ops.unlink(safe.root, probe)
            supported = selected_ops.fsync_directory(safe.root)
            safe.directory_flush_supported = supported
        except BaseException as error:
            if safe.root.child_kind(probe) is not None:
                try:
                    selected_ops.unlink(safe.root, probe)
                except BaseException as cleanup_error:
                    raise FilesystemSafetyError(
                        f"writability probe failed and cleanup failed: {cleanup_error}"
                    ) from error
            raise FilesystemSafetyError(f"result root is not durably writable: {error}") from error
        safe.verify()
        if safe.root.child_kind(probe) is not None:
            raise FilesystemSafetyError("writability probe left an artifact")
        return safe
    except BaseException:
        safe.close()
        raise


def pin_existing_root(root: Path, ops: FileOps | None = None) -> SafeRoot:
    selected = ops or FileOps()
    if not _lexists(root.absolute()):
        raise FilesystemSafetyError(f"root does not exist: {root}")
    safe = _pin_or_create(root, selected)
    safe.directory_flush_supported = selected.fsync_directory(safe.root)
    safe.verify()
    return safe


class AtomicArtifactStore:
    """Pinned atomic same-directory writes with mandatory hash sidecars."""

    def __init__(self, root: SafeRoot | Path, ops: FileOps | None = None) -> None:
        self.ops = ops or FileOps()
        self.safe_root = (
            root if isinstance(root, SafeRoot) else pin_existing_root(root, self.ops)
        )
        self.root = self.safe_root.path
        self.declared_directories: set[str] = set()

    @staticmethod
    def sidecar_for(path: Path) -> Path:
        return path.with_name(path.name + SIDECAR_SUFFIX)

    @staticmethod
    def temp_for(path: Path) -> Path:
        return path.with_name("." + path.name + TEMP_SUFFIX)

    def _parent(
        self,
        relative: Path,
        create: bool,
    ) -> tuple[PinnedDirectory, list[PinnedDirectory]]:
        current = self.safe_root.root
        opened: list[PinnedDirectory] = []
        prefix = Path()
        for component in relative.parts:
            prefix /= component
            path = current.child_path(component)
            child_kind = current.child_kind(component)
            if child_kind is not None:
                if child_kind != "directory":
                    raise FilesystemSafetyError(f"artifact parent is unsafe: {path}")
                child = current.open_child(component)
            elif create:
                child = self.ops.mkdir_child(current, component)
                self.ops.fsync_directory(current)
            else:
                raise FilesystemSafetyError(f"artifact parent is missing: {path}")
            opened.append(child)
            self.declared_directories.add(prefix.as_posix())
            current = child
        return current, opened

    @staticmethod
    def _close_opened(opened: list[PinnedDirectory]) -> None:
        for directory in reversed(opened):
            directory.close()

    def read_bytes(self, relative: str | Path) -> bytes:
        path = relative_path(relative)
        parent, opened = self._parent(path.parent, False)
        try:
            return _read_all(self.ops, parent, path.name)
        finally:
            self._close_opened(opened)

    def seal_files(self, relatives: Iterable[str]) -> list[int]:
        descriptors: list[int] = []
        try:
            for relative in relatives:
                path = relative_path(relative)
                parent, opened = self._parent(path.parent, False)
                try:
                    descriptors.append(self.ops.open_read(parent, path.name))
                finally:
                    self._close_opened(opened)
            self.safe_root.verify()
            return descriptors
        except BaseException:
            for descriptor in descriptors:
                os.close(descriptor)
            raise

    def write_blob(
        self,
        relative: str | Path,
        data: bytes,
        *,
        semantic_hash: str | None = None,
    ) -> dict[str, Any]:
        path = relative_path(relative)
        if path.name in (LOCK_NAME, MANIFEST_NAME) or path.name.endswith(SIDECAR_SUFFIX):
            raise FilesystemSafetyError(f"reserved artifact name: {path.name}")
        parent, opened = self._parent(path.parent, True)
        try:
            sidecar_name = path.name + SIDECAR_SUFFIX
            data_temp = "." + path.name + TEMP_SUFFIX
            sidecar_temp = "." + sidecar_name + TEMP_SUFFIX
            occupied = (path.name, sidecar_name, data_temp, sidecar_temp)
            if any(parent.child_kind(name) is not None for name in occupied):
                raise FilesystemSafetyError(f"artifact path is occupied or stale: {path}")
            byte_hash = hashlib.sha256(data).hexdigest()
            contract = {
                "schema_version": "cft-revival.experiment-artifact-sidecar/1.0.0",
                "artifact": path.as_posix(),
                "bytes": len(data),
                "byte_sha256": byte_hash,
                "semantic_sha256": semantic_hash,
                "canonicalization": CANONICALIZATION_ID if semantic_hash else None,
            }
            data_descriptor: int | None = None
            sidecar_descriptor: int | None = None
            try:
                data_descriptor = _open_durable_temporary(
                    self.ops,
                    parent,
                    data_temp,
                    data,
                    0o444,
                )
                sidecar_descriptor = _open_durable_temporary(
                    self.ops,
                    parent,
                    sidecar_temp,
                    canonical_bytes(contract),
                    0o444,
                )
                self.ops.publish(
                    parent,
                    data_descriptor,
                    data_temp,
                    path.name,
                )
                self.ops.fsync_directory(parent)
                self.ops.publish(
                    parent,
                    sidecar_descriptor,
                    sidecar_temp,
                    sidecar_name,
                )
                self.ops.fsync_directory(parent)
            finally:
                if sidecar_descriptor is not None:
                    os.close(sidecar_descriptor)
                if data_descriptor is not None:
                    os.close(data_descriptor)
            self.safe_root.verify()
            return contract
        finally:
            self._close_opened(opened)

    def write_json(self, relative: str | Path, value: Any) -> dict[str, Any]:
        data = canonical_bytes(value)
        return self.write_blob(relative, data, semantic_hash=semantic_sha256(value))

    def write_manifest_bytes(self, data: bytes) -> None:
        parsed = strict_json_loads(data)
        if canonical_bytes_from_parsed(parsed) != data:
            raise FilesystemSafetyError("candidate manifest bytes are not canonical")
        target = MANIFEST_NAME
        temporary = "." + target + TEMP_SUFFIX
        if (
            self.safe_root.root.child_kind(target) is not None
            or self.safe_root.root.child_kind(temporary) is not None
        ):
            raise FilesystemSafetyError("manifest already exists or has a stale temporary")
        descriptor = _open_durable_temporary(
            self.ops,
            self.safe_root.root,
            temporary,
            data,
            0o444,
        )
        try:
            self.ops.publish(
                self.safe_root.root,
                descriptor,
                temporary,
                target,
            )
            self.ops.fsync_directory(self.safe_root.root)
        finally:
            os.close(descriptor)
        self.safe_root.verify()


def canonical_bytes_from_parsed(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass
class ManagedCache:
    """A pinned marked working cache with secure recursive cleanup."""

    root: Path
    run_id: str
    ops: FileOps = field(default_factory=FileOps)
    safe_root: SafeRoot | None = None

    @property
    def marker(self) -> Path:
        return self.root / CACHE_MARKER

    def __post_init__(self) -> None:
        self.root = self.root.absolute()

    def prepare(self) -> str:
        existed = _lexists(self.root)
        safe = _pin_or_create(self.root, self.ops)
        self.safe_root = safe
        marker_value = {
            "schema_version": "cft-revival.experiment-working-cache/1.0.0",
            "run_id": self.run_id,
        }
        marker_data = canonical_bytes(marker_value)
        entries = scan_tree(safe, self.ops)
        unsafe = [entry for entry in entries if entry[1] not in {"file", "directory"}]
        if unsafe:
            raise FilesystemSafetyError(f"working cache contains unsafe entries: {unsafe}")
        if not entries:
            _exclusive_durable_write(self.ops, safe.root, CACHE_MARKER, marker_data)
            self.ops.fsync_directory(safe.root)
            return "empty_initialized" if existed else "absent_created"
        files = {name for name, kind in entries if kind == "file"}
        if CACHE_MARKER not in files:
            raise FilesystemSafetyError("populated working cache has no marker")
        actual = _read_all(self.ops, safe.root, CACHE_MARKER)
        try:
            parsed = strict_json_loads(actual)
        except Exception as error:
            raise FilesystemSafetyError("working cache marker is corrupt") from error
        if parsed != marker_value or actual != marker_data:
            raise FilesystemSafetyError("working cache marker identity mismatch")
        return "populated_reused"

    def _remove_contents(self, directory: PinnedDirectory) -> None:
        for name, kind in directory.list_entries():
            if kind == "file":
                self.ops.remove_file(directory, name)
            elif kind == "directory":
                child = directory.open_child(name)
                try:
                    self._remove_contents(child)
                finally:
                    child.close()
                self.ops.remove_directory(directory, name)
            else:
                raise FilesystemSafetyError(
                    f"unsafe cache cleanup entry: {directory.child_path(name)}"
                )

    def cleanup(self, *, protected: SafeRoot | None = None) -> None:
        if self.safe_root is None:
            return
        safe = self.safe_root
        safe.verify()
        if protected is not None and identities_overlap(safe.root, protected.root):
            raise FilesystemSafetyError("cache overlaps protected result tree")
        self._remove_contents(safe.root)
        safe.root.verify()
        safe.root.close()
        self.ops.remove_directory(safe.parent, safe.path.name)
        safe.parent.close()
        self.safe_root = None

    def close(self) -> None:
        if self.safe_root is not None:
            self.safe_root.close()
            self.safe_root = None


def verify_pair(
    store: AtomicArtifactStore | SafeRoot | Path,
    relative: str | Path,
) -> dict[str, Any]:
    owns_root = isinstance(store, Path)
    if isinstance(store, AtomicArtifactStore):
        selected = store
        safe: SafeRoot | None = None
    else:
        safe = store if isinstance(store, SafeRoot) else pin_existing_root(store)
        selected = AtomicArtifactStore(safe)
    relative_value = Path(relative)
    if relative_value.is_absolute():
        relative_value = relative_value.relative_to(selected.root)
    try:
        return _verify_pair_store(selected, relative_value)
    finally:
        if owns_root and safe is not None:
            safe.close()


def _verify_pair_store(
    store: AtomicArtifactStore,
    relative: str | Path,
) -> dict[str, Any]:
    path = relative_path(relative)
    sidecar_relative = path.with_name(path.name + SIDECAR_SUFFIX)
    try:
        data = store.read_bytes(path)
        sidecar_data = store.read_bytes(sidecar_relative)
    except (FileNotFoundError, OSError, PlatformFilesystemError) as error:
        raise PartialArtifactError(f"partial artifact pair: {path.as_posix()}") from error
    contract = strict_json_loads(sidecar_data)
    expected_keys = {
        "schema_version",
        "artifact",
        "bytes",
        "byte_sha256",
        "semantic_sha256",
        "canonicalization",
    }
    if (
        type(contract) is not dict
        or set(contract) != expected_keys
        or contract["schema_version"]
        != "cft-revival.experiment-artifact-sidecar/1.0.0"
        or contract["artifact"] != path.as_posix()
        or type(contract["bytes"]) is not int
        or contract["bytes"] < 0
        or contract["bytes"] != len(data)
        or type(contract["byte_sha256"]) is not str
        or contract["byte_sha256"] != hashlib.sha256(data).hexdigest()
    ):
        raise FilesystemSafetyError(f"artifact sidecar schema mismatch: {path.as_posix()}")
    semantic = contract["semantic_sha256"]
    canonicalization = contract["canonicalization"]
    if semantic is None:
        if canonicalization is not None:
            raise FilesystemSafetyError("blob sidecar cannot claim canonicalization")
    elif (
        type(semantic) is not str
        or len(semantic) != 64
        or canonicalization != CANONICALIZATION_ID
        or hashlib.sha256(data).hexdigest() != semantic
        or canonical_bytes_from_parsed(strict_json_loads(data)) != data
    ):
        raise FilesystemSafetyError(f"canonical semantic mismatch: {path.as_posix()}")
    return contract


def platform_durability_contract() -> dict[str, Any]:
    return durability_contract()
