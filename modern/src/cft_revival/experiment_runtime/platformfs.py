"""Pinned, no-follow filesystem operations for experiment result trees."""

from __future__ import annotations

import ctypes
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PlatformFilesystemError(RuntimeError):
    """The platform cannot prove a filesystem operation stayed in its root."""


@dataclass(frozen=True)
class DirectoryIdentity:
    platform: str
    volume: int
    file_id: int
    final_path: str


if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _DELETE = 0x00010000
    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _CREATE_NEW = 1
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_FLAG_WRITE_THROUGH = 0x80000000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _SYNCHRONIZE = 0x00100000
    _OBJ_CASE_INSENSITIVE = 0x00000040
    _FILE_OPEN = 1
    _FILE_CREATE = 2
    _FILE_DIRECTORY_FILE = 0x00000001
    _FILE_WRITE_THROUGH = 0x00000002
    _FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
    _FILE_NON_DIRECTORY_FILE = 0x00000040
    _FILE_OPEN_REPARSE_POINT_NT = 0x00200000
    _ERROR_ACCESS_DENIED = 5
    _ERROR_INVALID_HANDLE = 6

    class _FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", _FILETIME),
            ("access", _FILETIME),
            ("write", _FILETIME),
            ("volume_serial", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("link_count", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CreateFileW.restype = wintypes.HANDLE
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL
    _GetFileInformationByHandle = _kernel32.GetFileInformationByHandle
    _GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    ]
    _GetFileInformationByHandle.restype = wintypes.BOOL
    _GetFinalPathNameByHandleW = _kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _FlushFileBuffers = _kernel32.FlushFileBuffers
    _FlushFileBuffers.argtypes = [wintypes.HANDLE]
    _FlushFileBuffers.restype = wintypes.BOOL
    _SetFileInformationByHandle = _kernel32.SetFileInformationByHandle
    _SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _SetFileInformationByHandle.restype = wintypes.BOOL

    class _FILE_RENAME_INFO(ctypes.Structure):
        _fields_ = [
            ("replace_if_exists", wintypes.BOOLEAN),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
            ("file_name", wintypes.WCHAR * 1),
        ]

    class _FILE_DISPOSITION_INFO(ctypes.Structure):
        _fields_ = [("delete_file", wintypes.BOOLEAN)]

    class _UNICODE_STRING(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        ]

    class _OBJECT_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.ULONG),
            ("root_directory", wintypes.HANDLE),
            ("object_name", ctypes.POINTER(_UNICODE_STRING)),
            ("attributes", wintypes.ULONG),
            ("security_descriptor", ctypes.c_void_p),
            ("security_quality_of_service", ctypes.c_void_p),
        ]

    class _IO_STATUS_BLOCK(ctypes.Structure):
        _fields_ = [
            ("status", ctypes.c_void_p),
            ("information", ctypes.c_size_t),
        ]

    class _FILE_ID_BOTH_DIR_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("next_entry_offset", wintypes.ULONG),
            ("file_index", wintypes.ULONG),
            ("creation_time", ctypes.c_longlong),
            ("last_access_time", ctypes.c_longlong),
            ("last_write_time", ctypes.c_longlong),
            ("change_time", ctypes.c_longlong),
            ("end_of_file", ctypes.c_longlong),
            ("allocation_size", ctypes.c_longlong),
            ("file_attributes", wintypes.ULONG),
            ("file_name_length", wintypes.ULONG),
            ("ea_size", wintypes.ULONG),
            ("short_name_length", ctypes.c_ubyte),
            ("short_name", wintypes.WCHAR * 12),
            ("file_id", ctypes.c_longlong),
            ("file_name", wintypes.WCHAR * 1),
        ]

    _ntdll = ctypes.WinDLL("ntdll")
    _NtCreateFile = _ntdll.NtCreateFile
    _NtCreateFile.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_OBJECT_ATTRIBUTES),
        ctypes.POINTER(_IO_STATUS_BLOCK),
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _NtCreateFile.restype = wintypes.LONG
    _NtSetInformationFile = _ntdll.NtSetInformationFile
    _NtSetInformationFile.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IO_STATUS_BLOCK),
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.c_int,
    ]
    _NtSetInformationFile.restype = wintypes.LONG
    _NtQueryDirectoryFile = _ntdll.NtQueryDirectoryFile
    _NtQueryDirectoryFile.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_IO_STATUS_BLOCK),
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.c_int,
        wintypes.BOOLEAN,
        ctypes.c_void_p,
        wintypes.BOOLEAN,
    ]
    _NtQueryDirectoryFile.restype = wintypes.LONG
    _RtlNtStatusToDosError = _ntdll.RtlNtStatusToDosError
    _RtlNtStatusToDosError.argtypes = [wintypes.LONG]
    _RtlNtStatusToDosError.restype = wintypes.ULONG


def _normal_path(path: Path | str) -> str:
    text = os.path.normcase(os.path.abspath(os.fspath(path)))
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return os.path.normcase(os.path.normpath(text))


def _win_error(message: str) -> OSError:
    return ctypes.WinError(ctypes.get_last_error())  # type: ignore[name-defined]


def _win_identity(handle: int) -> DirectoryIdentity:
    information = _BY_HANDLE_FILE_INFORMATION()  # type: ignore[name-defined]
    if not _GetFileInformationByHandle(handle, information):  # type: ignore[name-defined]
        raise _win_error("GetFileInformationByHandle failed")
    if information.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:  # type: ignore[name-defined]
        raise PlatformFilesystemError("directory handle resolves to a reparse point")
    size = 512
    while True:
        buffer = ctypes.create_unicode_buffer(size)  # type: ignore[name-defined]
        length = _GetFinalPathNameByHandleW(handle, buffer, size, 0)  # type: ignore[name-defined]
        if length == 0:
            raise _win_error("GetFinalPathNameByHandleW failed")
        if length < size:
            break
        size = length + 1
    return DirectoryIdentity(
        "windows-handle-v1",
        int(information.volume_serial),
        (int(information.file_index_high) << 32) | int(information.file_index_low),
        _normal_path(buffer.value),
    )


def _win_open_directory(path: Path) -> int:
    handle = _CreateFileW(  # type: ignore[name-defined]
        str(path),
        _FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES,  # type: ignore[name-defined]
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,  # type: ignore[name-defined]
        None,
        _OPEN_EXISTING,  # type: ignore[name-defined]
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,  # type: ignore[name-defined]
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:  # type: ignore[name-defined]
        raise _win_error(f"cannot pin directory {path}")
    try:
        _win_identity(handle)
    except BaseException:
        _CloseHandle(handle)  # type: ignore[name-defined]
        raise
    return int(handle)


def _win_nt_open_relative(
    parent_handle: int,
    name: str,
    *,
    access: int,
    disposition: int,
    options: int,
    attributes: int = 0,
    share_access: int | None = None,
) -> int:
    buffer = ctypes.create_unicode_buffer(name)  # type: ignore[name-defined]
    encoded_length = len(name.encode("utf-16-le"))
    unicode_name = _UNICODE_STRING(  # type: ignore[name-defined]
        encoded_length,
        encoded_length + ctypes.sizeof(wintypes.WCHAR),  # type: ignore[name-defined]
        ctypes.cast(buffer, wintypes.LPWSTR),  # type: ignore[name-defined]
    )
    object_attributes = _OBJECT_ATTRIBUTES()  # type: ignore[name-defined]
    object_attributes.length = ctypes.sizeof(_OBJECT_ATTRIBUTES)  # type: ignore[name-defined]
    object_attributes.root_directory = parent_handle
    object_attributes.object_name = ctypes.pointer(unicode_name)  # type: ignore[name-defined]
    object_attributes.attributes = _OBJ_CASE_INSENSITIVE  # type: ignore[name-defined]
    status_block = _IO_STATUS_BLOCK()  # type: ignore[name-defined]
    handle = wintypes.HANDLE()  # type: ignore[name-defined]
    status = _NtCreateFile(  # type: ignore[name-defined]
        ctypes.byref(handle),  # type: ignore[name-defined]
        access,
        ctypes.byref(object_attributes),  # type: ignore[name-defined]
        ctypes.byref(status_block),  # type: ignore[name-defined]
        None,
        attributes,
        (
            _FILE_SHARE_READ | _FILE_SHARE_WRITE  # type: ignore[name-defined]
            if share_access is None
            else share_access
        ),
        disposition,
        options,
        None,
        0,
    )
    if status < 0:
        error = _RtlNtStatusToDosError(status)  # type: ignore[name-defined]
        if error in (80, 183):
            raise FileExistsError(error, "relative target exists", name)
        raise OSError(error, os.strerror(error), name)
    return int(handle.value)


def _win_delete_relative(parent_handle: int, name: str, *, directory: bool) -> None:
    options = (
        (
            _FILE_DIRECTORY_FILE  # type: ignore[name-defined]
            if directory
            else _FILE_NON_DIRECTORY_FILE  # type: ignore[name-defined]
        )
        | _FILE_OPEN_REPARSE_POINT_NT  # type: ignore[name-defined]
        | _FILE_SYNCHRONOUS_IO_NONALERT  # type: ignore[name-defined]
    )
    handle = _win_nt_open_relative(
        parent_handle,
        name,
        access=_DELETE | _SYNCHRONIZE,  # type: ignore[name-defined]
        disposition=_FILE_OPEN,  # type: ignore[name-defined]
        options=options,
    )
    try:
        information = _FILE_DISPOSITION_INFO(True)  # type: ignore[name-defined]
        if not _SetFileInformationByHandle(  # type: ignore[name-defined]
            handle,
            4,
            ctypes.byref(information),  # type: ignore[name-defined]
            ctypes.sizeof(information),  # type: ignore[name-defined]
        ):
            raise _win_error(f"handle-relative delete failed: {name}")
    finally:
        _CloseHandle(handle)  # type: ignore[name-defined]


class PinnedDirectory:
    """A directory kept open and identity-checked for every mutation."""

    def __init__(self, path: Path, handle: int, identity: DirectoryIdentity) -> None:
        self.path = path.absolute()
        self.handle = handle
        self.identity = identity
        self.closed = False

    @classmethod
    def open(cls, path: Path) -> "PinnedDirectory":
        absolute = path.absolute()
        if os.name == "nt":
            handle = _win_open_directory(absolute)
            return cls(absolute, handle, _win_identity(handle))
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(absolute, flags)
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode):
            os.close(descriptor)
            raise PlatformFilesystemError(f"not a directory: {absolute}")
        return cls(
            absolute,
            descriptor,
            DirectoryIdentity(
                "posix-dirfd-v1",
                status.st_dev,
                status.st_ino,
                _normal_path(absolute),
            ),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if os.name == "nt":
            if not _CloseHandle(self.handle):  # type: ignore[name-defined]
                raise _win_error("CloseHandle failed")
        else:
            os.close(self.handle)

    def __enter__(self) -> "PinnedDirectory":
        return self

    def __exit__(self, *_arguments: Any) -> None:
        self.close()

    def current_identity(self) -> DirectoryIdentity:
        if self.closed:
            raise PlatformFilesystemError("directory pin is closed")
        if os.name == "nt":
            return _win_identity(self.handle)
        status = os.fstat(self.handle)
        return DirectoryIdentity(
            "posix-dirfd-v1",
            status.st_dev,
            status.st_ino,
            self.identity.final_path,
        )

    def verify(self) -> None:
        if self.current_identity() != self.identity:
            raise PlatformFilesystemError("pinned directory handle identity changed")
        other = PinnedDirectory.open(self.path)
        try:
            if other.identity != self.identity:
                raise PlatformFilesystemError("directory path no longer names pinned identity")
        finally:
            other.close()

    def child_path(self, name: str) -> Path:
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise PlatformFilesystemError(f"unsafe child name: {name!r}")
        return self.path / name

    def list_entries(self) -> list[tuple[str, str]]:
        """Enumerate direct children through the pinned directory handle."""

        self.verify()
        entries: list[tuple[str, str]] = []
        if os.name == "nt":
            restart = True
            while True:
                buffer = ctypes.create_string_buffer(64 * 1024)  # type: ignore[name-defined]
                status_block = _IO_STATUS_BLOCK()  # type: ignore[name-defined]
                status = _NtQueryDirectoryFile(  # type: ignore[name-defined]
                    self.handle,
                    None,
                    None,
                    None,
                    ctypes.byref(status_block),  # type: ignore[name-defined]
                    buffer,
                    len(buffer),
                    37,
                    False,
                    None,
                    restart,
                )
                restart = False
                unsigned_status = status & 0xFFFFFFFF
                if unsigned_status == 0x80000006:
                    break
                if status < 0 and unsigned_status != 0x80000005:
                    error = _RtlNtStatusToDosError(status)  # type: ignore[name-defined]
                    raise OSError(error, os.strerror(error), str(self.path))
                used = int(status_block.information)
                offset = 0
                while offset < used:
                    address = ctypes.addressof(buffer) + offset  # type: ignore[name-defined]
                    information = ctypes.cast(  # type: ignore[name-defined]
                        address,
                        ctypes.POINTER(  # type: ignore[name-defined]
                            _FILE_ID_BOTH_DIR_INFORMATION  # type: ignore[name-defined]
                        ),
                    ).contents
                    name_offset = (
                        _FILE_ID_BOTH_DIR_INFORMATION  # type: ignore[name-defined]
                        .file_name.offset
                    )
                    name_bytes = ctypes.string_at(  # type: ignore[name-defined]
                        address + name_offset,
                        information.file_name_length,
                    )
                    name = name_bytes.decode("utf-16-le")
                    if name not in (".", ".."):
                        attributes = information.file_attributes
                        reparse = (
                            attributes
                            & _FILE_ATTRIBUTE_REPARSE_POINT  # type: ignore[name-defined]
                        )
                        if reparse:
                            kind = "reparse"
                        elif attributes & _FILE_ATTRIBUTE_DIRECTORY:  # type: ignore[name-defined]
                            kind = "directory"
                        else:
                            kind = "file"
                        entries.append((name, kind))
                    if information.next_entry_offset == 0:
                        break
                    offset += information.next_entry_offset
                if status == 0 and used == 0:
                    break
        else:
            with os.scandir(self.handle) as iterator:
                for entry in iterator:
                    if entry.is_symlink():
                        kind = "symlink"
                    elif entry.is_dir(follow_symlinks=False):
                        kind = "directory"
                    elif entry.is_file(follow_symlinks=False):
                        kind = "file"
                    else:
                        kind = "special"
                    entries.append((entry.name, kind))
        self.verify()
        return sorted(entries)

    def child_kind(self, name: str) -> str | None:
        self.child_path(name)
        return dict(self.list_entries()).get(name)

    def open_child(self, name: str) -> "PinnedDirectory":
        self.verify()
        if os.name == "nt":
            handle = _win_nt_open_relative(
                self.handle,
                name,
                access=(
                    _FILE_LIST_DIRECTORY  # type: ignore[name-defined]
                    | _FILE_READ_ATTRIBUTES  # type: ignore[name-defined]
                    | _SYNCHRONIZE  # type: ignore[name-defined]
                ),
                disposition=_FILE_OPEN,  # type: ignore[name-defined]
                options=_FILE_DIRECTORY_FILE
                | _FILE_OPEN_REPARSE_POINT_NT
                | _FILE_SYNCHRONOUS_IO_NONALERT,  # type: ignore[name-defined]
            )
            child = PinnedDirectory(self.child_path(name), handle, _win_identity(handle))
        else:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(name, flags, dir_fd=self.handle)
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                os.close(descriptor)
                raise PlatformFilesystemError(f"not a directory: {name}")
            child = PinnedDirectory(
                self.child_path(name),
                descriptor,
                DirectoryIdentity(
                    "posix-dirfd-v1",
                    status.st_dev,
                    status.st_ino,
                    _normal_path(self.child_path(name)),
                ),
            )
        self.verify()
        return child

    def mkdir_child(self, name: str) -> "PinnedDirectory":
        self.verify()
        if os.name == "nt":
            handle = _win_nt_open_relative(
                self.handle,
                name,
                access=(
                    _FILE_LIST_DIRECTORY  # type: ignore[name-defined]
                    | _FILE_READ_ATTRIBUTES  # type: ignore[name-defined]
                    | _SYNCHRONIZE  # type: ignore[name-defined]
                ),
                disposition=_FILE_CREATE,  # type: ignore[name-defined]
                options=_FILE_DIRECTORY_FILE
                | _FILE_OPEN_REPARSE_POINT_NT
                | _FILE_SYNCHRONOUS_IO_NONALERT,  # type: ignore[name-defined]
            )
            child = PinnedDirectory(self.child_path(name), handle, _win_identity(handle))
        else:
            os.mkdir(name, dir_fd=self.handle)
            child = self.open_child(name)
        self.verify()
        return child

    def open_file_exclusive(self, name: str, mode: int) -> int:
        self.verify()
        path = self.child_path(name)
        if os.name == "nt":
            handle = _win_nt_open_relative(
                self.handle,
                name,
                access=_GENERIC_WRITE | _SYNCHRONIZE,  # type: ignore[name-defined]
                disposition=_FILE_CREATE,  # type: ignore[name-defined]
                options=_FILE_NON_DIRECTORY_FILE
                | _FILE_WRITE_THROUGH
                | _FILE_OPEN_REPARSE_POINT_NT
                | _FILE_SYNCHRONOUS_IO_NONALERT,  # type: ignore[name-defined]
                attributes=_FILE_ATTRIBUTE_NORMAL,  # type: ignore[name-defined]
            )
            descriptor = msvcrt.open_osfhandle(  # type: ignore[name-defined]
                int(handle),
                os.O_WRONLY | os.O_BINARY,
            )
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, mode, dir_fd=self.handle)
        self.verify()
        return descriptor

    def open_temporary(self, name: str, mode: int) -> int:
        self.verify()
        if os.name == "nt":
            handle = _win_nt_open_relative(
                self.handle,
                name,
                access=_GENERIC_WRITE | _DELETE | _SYNCHRONIZE,  # type: ignore[name-defined]
                disposition=_FILE_CREATE,  # type: ignore[name-defined]
                options=_FILE_NON_DIRECTORY_FILE
                | _FILE_WRITE_THROUGH
                | _FILE_OPEN_REPARSE_POINT_NT
                | _FILE_SYNCHRONOUS_IO_NONALERT,  # type: ignore[name-defined]
                attributes=_FILE_ATTRIBUTE_NORMAL,  # type: ignore[name-defined]
            )
            descriptor = msvcrt.open_osfhandle(  # type: ignore[name-defined]
                int(handle),
                os.O_WRONLY | os.O_BINARY,
            )
        else:
            temporary_flag = getattr(os, "O_TMPFILE", 0)
            if temporary_flag == 0:
                raise PlatformFilesystemError("O_TMPFILE is required for secure publication")
            descriptor = os.open(
                ".",
                os.O_WRONLY | temporary_flag,
                mode,
                dir_fd=self.handle,
            )
        self.verify()
        return descriptor

    def publish_open_file(
        self,
        descriptor: int,
        source_name: str,
        target_name: str,
    ) -> None:
        self.verify()
        if self.child_kind(target_name) is not None:
            raise FileExistsError(f"publication target exists: {target_name}")
        if os.name == "nt":
            target_text = target_name
            encoded = target_text.encode("utf-16-le")
            offset = _FILE_RENAME_INFO.file_name.offset  # type: ignore[name-defined]
            size = (
                offset
                + len(encoded)
                + ctypes.sizeof(wintypes.WCHAR)  # type: ignore[name-defined]
            )
            buffer = ctypes.create_string_buffer(size)  # type: ignore[name-defined]
            information = ctypes.cast(  # type: ignore[name-defined]
                buffer,
                ctypes.POINTER(_FILE_RENAME_INFO),  # type: ignore[name-defined]
            ).contents
            information.replace_if_exists = False
            information.root_directory = self.handle
            information.file_name_length = len(encoded)
            ctypes.memmove(  # type: ignore[name-defined]
                ctypes.addressof(buffer) + offset,  # type: ignore[name-defined]
                encoded,
                len(encoded),
            )
            file_handle = msvcrt.get_osfhandle(descriptor)  # type: ignore[name-defined]
            status_block = _IO_STATUS_BLOCK()  # type: ignore[name-defined]
            status = _NtSetInformationFile(  # type: ignore[name-defined]
                file_handle,
                ctypes.byref(status_block),  # type: ignore[name-defined]
                buffer,
                size,
                10,
            )
            if status < 0:
                error = _RtlNtStatusToDosError(status)  # type: ignore[name-defined]
                raise OSError(
                    error,
                    os.strerror(error),
                    f"{source_name} -> {target_name}",
                )
        else:
            libc = ctypes.CDLL(None, use_errno=True)  # type: ignore[name-defined]
            linkat = libc.linkat
            linkat.argtypes = [
                ctypes.c_int,  # type: ignore[name-defined]
                ctypes.c_char_p,  # type: ignore[name-defined]
                ctypes.c_int,  # type: ignore[name-defined]
                ctypes.c_char_p,  # type: ignore[name-defined]
                ctypes.c_int,  # type: ignore[name-defined]
            ]
            linkat.restype = ctypes.c_int  # type: ignore[name-defined]
            if linkat(
                descriptor,
                b"",
                self.handle,
                os.fsencode(target_name),
                0x1000,
            ) != 0:
                error = ctypes.get_errno()  # type: ignore[name-defined]
                raise OSError(error, os.strerror(error), target_name)
        self.verify()

    def open_file_read(self, name: str) -> int:
        self.verify()
        path = self.child_path(name)
        if os.name == "nt":
            handle = _win_nt_open_relative(
                self.handle,
                name,
                access=_GENERIC_READ | _SYNCHRONIZE,  # type: ignore[name-defined]
                disposition=_FILE_OPEN,  # type: ignore[name-defined]
                options=_FILE_NON_DIRECTORY_FILE
                | _FILE_OPEN_REPARSE_POINT_NT
                | _FILE_SYNCHRONOUS_IO_NONALERT,  # type: ignore[name-defined]
                share_access=_FILE_SHARE_READ,  # type: ignore[name-defined]
            )
            information = _BY_HANDLE_FILE_INFORMATION()  # type: ignore[name-defined]
            if not _GetFileInformationByHandle(handle, information):  # type: ignore[name-defined]
                _CloseHandle(handle)  # type: ignore[name-defined]
                raise _win_error(f"cannot identify {path}")
            if (
                information.attributes
                & _FILE_ATTRIBUTE_REPARSE_POINT  # type: ignore[name-defined]
            ):
                _CloseHandle(handle)  # type: ignore[name-defined]
                raise PlatformFilesystemError(f"file is a reparse point: {path}")
            descriptor = msvcrt.open_osfhandle(  # type: ignore[name-defined]
                int(handle),
                os.O_RDONLY | os.O_BINARY,
            )
        else:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, dir_fd=self.handle)
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            os.close(descriptor)
            raise PlatformFilesystemError(f"not a regular file: {path}")
        self.verify()
        return descriptor

    def unlink(self, name: str) -> None:
        self.verify()
        if os.name == "nt":
            _win_delete_relative(self.handle, name, directory=False)
        else:
            os.unlink(name, dir_fd=self.handle)
        self.verify()

    def remove_directory(self, name: str) -> None:
        self.verify()
        if os.name == "nt":
            _win_delete_relative(self.handle, name, directory=True)
        else:
            os.rmdir(name, dir_fd=self.handle)
        self.verify()

    def fsync(self) -> bool:
        self.verify()
        if os.name == "nt":
            if _FlushFileBuffers(self.handle):  # type: ignore[name-defined]
                return True
            # Windows commonly returns ERROR_INVALID_HANDLE for directory
            # handles. File flush and FILE_WRITE_THROUGH still apply.
            if ctypes.get_last_error() in (  # type: ignore[name-defined]
                _ERROR_ACCESS_DENIED,  # type: ignore[name-defined]
                _ERROR_INVALID_HANDLE,  # type: ignore[name-defined]
            ):
                return False
            raise _win_error("directory FlushFileBuffers failed")
        os.fsync(self.handle)
        return True


def identities_overlap(left: PinnedDirectory, right: PinnedDirectory) -> bool:
    left.verify()
    right.verify()
    if (left.identity.volume, left.identity.file_id) == (
        right.identity.volume,
        right.identity.file_id,
    ):
        return True
    left_path = left.identity.final_path
    right_path = right.identity.final_path
    try:
        common = os.path.commonpath((left_path, right_path))
    except ValueError:
        return False
    return common in (left_path, right_path)


def durability_contract() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "platform": "windows",
            "temporary_open": "NtCreateFile(FILE_WRITE_THROUGH) in the pinned directory",
            "file_flush": "FlushFileBuffers via os.fsync",
            "replace": "SetFileInformationByHandle(FileRenameInfo) while source is pinned",
            "directory_metadata_flush": (
                "FlushFileBuffers attempted; ERROR_INVALID_HANDLE and "
                "ERROR_ACCESS_DENIED are tolerated bounded limitations"
            ),
            "power_loss_claim": (
                "file flush plus handle rename; "
                "no POSIX directory-fsync equivalence claimed"
            ),
        }
    return {
        "platform": "posix",
        "temporary_open": "O_TMPFILE in the pinned directory fd",
        "file_flush": "fsync",
        "replace": "linkat(AT_EMPTY_PATH) from an O_TMPFILE in the pinned dirfd",
        "directory_metadata_flush": "fsync pinned directory fd",
        "power_loss_claim": "file and containing-directory fsync",
    }
