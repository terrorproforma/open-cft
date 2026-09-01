"""Canonical finite JSON identity helpers for surrogate artifacts."""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from typing import Mapping

from .normalization import SurrogateValidationError


def canonical_float(value: float) -> float:
    converted = float(value)
    if not isfinite(converted):
        raise SurrogateValidationError("identity values must be finite")
    return 0.0 if converted == 0.0 else converted


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SurrogateValidationError("value is not canonical finite JSON") from error


def canonical_hash(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def strict_json_loads(payload: str) -> object:
    def reject_constant(value: str) -> object:
        raise SurrogateValidationError(f"serialized artifact contains {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SurrogateValidationError(
                    f"serialized artifact repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, RecursionError, OverflowError) as error:
        raise SurrogateValidationError("invalid serialized JSON") from error


def require_exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise SurrogateValidationError(
            f"{context} keys mismatch; missing={missing}, unknown={unknown}"
        )
