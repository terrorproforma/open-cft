"""Strict canonical JSON and exact-byte persistence for validation v3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time as wall_time, timezone
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID


class CanonicalTypeError(TypeError):
    """A payload value cannot enter the canonical evidence domain."""


class CanonicalValueError(ValueError):
    """A payload value violates canonical evidence constraints."""


@dataclass(frozen=True, slots=True)
class TaggedSchema:
    """Explicit opt-in wrapper for otherwise unsupported schema values."""

    tag: str
    value: Any


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _key_path(path: str, key: str) -> str:
    return f"{path}.{key}" if _IDENTIFIER.fullmatch(key) else f"{path}[{key!r}]"


def normalize(value: Any, *, path: str = "$") -> Any:
    """Return the only representation accepted for hashing and persistence."""

    module = type(value).__module__
    if not isinstance(value, TaggedSchema) and module.split(".", 1)[0] in {
        "numpy",
        "warp",
        "torch",
        "cupy",
        "jax",
    }:
        raise CanonicalTypeError(
            f"{path}: unsupported numerical/device type "
            f"{module}.{type(value).__qualname__}"
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalValueError(f"{path}: nonfinite float is forbidden")
        return 0.0 if value == 0.0 else value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalValueError(f"{path}: naive datetime is forbidden")
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    if isinstance(value, Enum):
        return normalize(value.value, path=path)
    if isinstance(value, TaggedSchema):
        if not value.tag or not isinstance(value.tag, str):
            raise CanonicalValueError(f"{path}.tag: nonempty string required")
        tagged_encoders = {
            "bytes-hex": lambda item: item.hex() if isinstance(item, bytes) else None,
            "path-posix": lambda item: item.as_posix()
            if isinstance(item, Path)
            else None,
            "decimal-string": lambda item: str(item)
            if isinstance(item, Decimal)
            else None,
            "fraction-ratio": lambda item: [item.numerator, item.denominator]
            if isinstance(item, Fraction)
            else None,
            "uuid-string": lambda item: str(item) if isinstance(item, UUID) else None,
            "date-iso": lambda item: item.isoformat()
            if isinstance(item, date) and not isinstance(item, datetime)
            else None,
            "time-iso": lambda item: item.isoformat()
            if isinstance(item, wall_time)
            else None,
            "complex-pair": lambda item: [item.real, item.imag]
            if isinstance(item, complex)
            else None,
            "set-sorted": lambda item: sorted(
                (normalize(child, path=f"{path}.value[]") for child in item),
                key=canonical_bytes,
            )
            if isinstance(item, (set, frozenset))
            else None,
        }
        if value.tag in tagged_encoders:
            encoded = tagged_encoders[value.tag](value.value)
            if encoded is None:
                raise CanonicalTypeError(
                    f"{path}.value: value does not match tag {value.tag!r}"
                )
            return {"$type": value.tag, "value": encoded}
        return {
            "$type": value.tag,
            "value": normalize(value.value, path=f"{path}.value"),
        }
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for field in fields(value):
            result[field.name] = normalize(
                getattr(value, field.name),
                path=_key_path(path, field.name),
            )
        return result
    if isinstance(value, Mapping):
        result = {}
        seen: set[str] = set()
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalTypeError(
                    f"{path}: mapping key {key!r} has unsupported "
                    f"type {type(key).__module__}.{type(key).__qualname__}"
                )
            if key in seen:
                raise CanonicalValueError(
                    f"{_key_path(path, key)}: mapping key collision"
                )
            seen.add(key)
            result[key] = normalize(item, path=_key_path(path, key))
        return result
    if isinstance(value, (list, tuple)):
        return [
            normalize(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CanonicalTypeError(
        f"{path}: unsupported type "
        f"{type(value).__module__}.{type(value).__qualname__}; "
        "use an explicit TaggedSchema only when the protocol defines it"
    )


def canonical_bytes(value: Any) -> bytes:
    normalized = normalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize(payload)
    if "semantic_integrity" in normalized:
        raise CanonicalValueError("$.semantic_integrity: reserved envelope key")
    normalized["semantic_integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "strict-json-canonical-utf8-v3",
        "payload_sha256": hashlib.sha256(canonical_bytes(normalized)).hexdigest(),
    }
    return normalize(normalized)


def _write_exact(path: Path, data: bytes, *, exclusive: bool, atomic: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return
    if atomic:
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_canonical(
    path: Path,
    payload: Mapping[str, Any],
    *,
    exclusive: bool = False,
    atomic: bool = False,
    sealed: bool = True,
) -> dict[str, Any]:
    """Canonicalize completely, then persist those exact hashed bytes."""

    stored = seal(payload) if sealed else normalize(payload)
    data = canonical_bytes(stored)
    _write_exact(path, data, exclusive=exclusive, atomic=atomic)
    if path.read_bytes() != data:
        raise OSError(f"{path}: persisted canonical bytes differ from source")
    return stored


def write_raw(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Persist explicitly non-JSON evidence such as captured process streams."""

    if type(data) is not bytes:
        raise CanonicalTypeError("$raw: exact bytes required")
    _write_exact(path, data, exclusive=exclusive, atomic=False)


def strict_load_bytes(data: bytes, *, source: str = "$bytes") -> Any:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalValueError(
                    f"{source}.{key}: duplicate JSON key"
                )
            result[key] = value
        return result

    def reject(constant: str) -> None:
        raise CanonicalValueError(f"{source}: nonfinite constant {constant}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalValueError(f"{source}: invalid canonical JSON: {error}") from error
    normalized = normalize(value, path=source)
    if canonical_bytes(normalized) != data:
        raise CanonicalValueError(f"{source}: bytes are not canonical")
    return normalized


def load_canonical(path: Path, *, verify_seal: bool = True) -> dict[str, Any]:
    value = strict_load_bytes(path.read_bytes(), source=str(path))
    if not isinstance(value, dict):
        raise CanonicalValueError(f"{path}: object envelope required")
    if verify_seal:
        integrity = value.get("semantic_integrity")
        if not isinstance(integrity, dict):
            raise CanonicalValueError(f"{path}: semantic_integrity missing")
        payload = dict(value)
        payload.pop("semantic_integrity")
        expected = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        if integrity.get("payload_sha256") != expected:
            raise CanonicalValueError(f"{path}: semantic hash mismatch")
    return value


def diagnose_bytes(data: bytes, *, source: str) -> dict[str, Any]:
    diagnosis = {
        "source": source,
        "byte_count": len(data),
        "byte_sha256": hashlib.sha256(data).hexdigest(),
        "canonical": False,
        "error": None,
    }
    try:
        strict_load_bytes(data, source=source)
        diagnosis["canonical"] = True
    except (CanonicalTypeError, CanonicalValueError) as error:
        diagnosis["error"] = str(error)
    return diagnosis
