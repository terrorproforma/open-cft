"""Strict typed canonical serialization used for both identity and persistence."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import math
import re
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

TYPE_KEY = "__cft_type__"
CANONICALIZATION_ID = "cft-typed-canonical-json-v1"
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class CanonicalizationError(ValueError):
    """A value cannot be represented by the closed canonical type policy."""


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CanonicalizationError("datetime values must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _path_text(value: PurePath) -> str:
    raw = str(value)
    drive = getattr(value, "drive", "")
    if (
        value.is_absolute()
        or bool(drive)
        or ("/" in raw and "\\" in raw)
        or raw.endswith(("/", "\\"))
        or ":" in raw
    ):
        raise CanonicalizationError("paths must be relative")
    parts = value.parts
    if not parts or any(
        part in ("", ".", "..")
        or part.endswith((" ", "."))
        or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
        or any(ord(character) < 32 for character in part)
        for part in parts
    ):
        raise CanonicalizationError("paths must be normalized, non-empty, and traversal-free")
    text = "/".join(parts)
    if "\\" in text:
        raise CanonicalizationError("paths must use portable components")
    return text


def canonical_value(value: Any) -> Any:
    """Convert supported Python values to an unambiguous JSON value.

    Plain JSON values remain plain. Types which JSON cannot distinguish are
    explicitly tagged. Mapping keys are strings and the tag key is reserved.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return {
            TYPE_KEY: "enum",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": canonical_value(value.value),
        }
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            raise CanonicalizationError("integers must fit signed 64-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are forbidden")
        return value
    if isinstance(value, datetime):
        return {TYPE_KEY: "aware-utc-datetime", "value": _utc_text(value)}
    if isinstance(value, (Path, PurePath)):
        return {TYPE_KEY: "relative-posix-path", "value": _path_text(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return {TYPE_KEY: "bytes-base64", "value": encoded}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            TYPE_KEY: "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: canonical_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("mapping keys must be strings")
            if key == TYPE_KEY:
                raise CanonicalizationError(f"{TYPE_KEY!r} is reserved")
            result[key] = canonical_value(item)
        return result
    if isinstance(value, tuple):
        return {
            TYPE_KEY: "tuple",
            "items": [canonical_value(item) for item in value],
        }
    if isinstance(value, list):
        return [canonical_value(item) for item in value]
    raise CanonicalizationError(f"unsupported canonical type: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Return the exact bytes used by both persistence and SHA-256 identity."""

    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json_loads(data: bytes | str) -> Any:
    """Parse UTF-8 JSON while rejecting duplicates, constants, and non-UTF-8."""

    text = data.decode("utf-8") if isinstance(data, bytes) else data

    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalizationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CanonicalizationError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalizationError(f"invalid strict JSON: {error}") from error


def strict_json_file(path: Path) -> Any:
    return strict_json_loads(path.read_bytes())


def producer_id(producer: Any, source_root: Path) -> str:
    """Build a stable ``relative-file:qualname`` producer identity."""

    qualname = getattr(producer, "__qualname__", None)
    source = inspect.getsourcefile(producer)
    if (
        not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
        or "<lambda>" in qualname
        or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", qualname)
    ):
        raise CanonicalizationError("producer must have a stable module-level qualname")
    if source is None:
        raise CanonicalizationError("producer source file is unavailable")
    root = source_root.resolve()
    try:
        relative = Path(source).resolve().relative_to(root)
    except ValueError as error:
        raise CanonicalizationError("producer source is outside source_root") from error
    return f"{relative.as_posix()}:{qualname}"
