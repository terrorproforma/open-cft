"""Authoritative JSON normalization and canonical bytes for field artifacts."""

from __future__ import annotations

import builtins
import json
from math import copysign, isfinite
from typing import Any, Literal

from .models import FieldArtifactValidationError

CANONICALIZATION_V2 = "field-json-sorted-utf8-signed-zero-v2"
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991


def normalize_field_artifact_value(value: object, *, path: str = "$") -> Any:
    """Return the sole canonical field-serialization representation.

    Dictionaries and sequences are copied recursively, tuples become arrays,
    finite nonzero binary64 values are preserved exactly, and both signs of
    floating zero become ``+0.0``. Booleans remain booleans so closed-schema
    validation can distinguish them from numeric fields.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            raise FieldArtifactValidationError(
                f"{path} integer exceeds the exact JSON interoperability range"
            )
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise FieldArtifactValidationError(f"{path} must be finite")
        if value == 0.0:
            return 0.0
        return value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FieldArtifactValidationError(
                    f"{path} object keys must be strings"
                )
            if key in normalized:
                raise FieldArtifactValidationError(
                    f"{path} contains duplicate key {key!r}"
                )
            normalized[key] = normalize_field_artifact_value(
                item, path=f"{path}.{key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            normalize_field_artifact_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise FieldArtifactValidationError(
        f"{path} has unsupported serialization type {type(value).__name__}"
    )


def canonical_field_artifact_bytes(
    value: object,
    *,
    representation: Literal["payload", "file"],
) -> bytes:
    """Normalize recursively, then emit the authoritative UTF-8 JSON bytes."""

    normalized = normalize_field_artifact_value(value)
    try:
        if representation == "payload":
            text = json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        elif representation == "file":
            text = (
                json.dumps(
                    normalized,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        else:
            raise FieldArtifactValidationError(
                "representation must be 'payload' or 'file'"
            )
    except (builtins.ValueError, TypeError, OverflowError) as error:
        raise FieldArtifactValidationError(
            "field artifact cannot be represented as canonical JSON"
        ) from error
    return text.encode("utf-8")


def parse_field_json_bytes(
    data: bytes,
    *,
    source: str,
    require_canonical_file_bytes: bool,
) -> dict[str, object]:
    """Strictly parse, normalize, and optionally require canonical raw bytes."""

    def closed_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FieldArtifactValidationError(
                    f"duplicate JSON key {key!r} in {source}"
                )
            result[key] = value
        return result

    def reject_constant(value):
        raise FieldArtifactValidationError(
            f"nonfinite JSON constant {value!r} in {source}"
        )

    try:
        text = data.decode("utf-8")
        loaded = json.loads(
            text,
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except FieldArtifactValidationError:
        raise
    except (UnicodeError, builtins.ValueError, OverflowError) as error:
        raise FieldArtifactValidationError(
            f"invalid JSON numeric or UTF-8 value in {source}"
        ) from error
    if not isinstance(loaded, dict):
        raise FieldArtifactValidationError(
            f"{source} must contain one JSON object"
        )
    normalized = normalize_field_artifact_value(loaded)
    if require_canonical_file_bytes:
        canonical = canonical_field_artifact_bytes(
            normalized, representation="file"
        )
        if data != canonical:
            raise FieldArtifactValidationError(
                f"{source} is not canonical field-artifact file bytes"
            )
    return normalized


def contains_negative_zero(value: object) -> bool:
    """Test helper and diagnostic for nested signed floating zero."""

    if isinstance(value, float):
        return value == 0.0 and copysign(1.0, value) < 0.0
    if isinstance(value, dict):
        return any(contains_negative_zero(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_negative_zero(item) for item in value)
    return False

