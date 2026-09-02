"""Strict identity and artifact primitives for field-surrogate v4."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DECLARATION = ROOT / "predeclaration.json"
GEOMETRY_PREFLIGHT = ROOT / "geometry-preflight.json"
PARTITIONS = ROOT / "partitions.json"
SYNTHETIC_PREFLIGHT = ROOT / "synthetic-preflight.json"
DEPENDENCY_LOCK = ROOT / "dependency-lock.json"
RESULTS = ROOT / "results"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_load(path: Path) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


PROTOCOL = strict_load(DECLARATION)
if PROTOCOL["schema_version"] != "cft-revival.l1a-field-surrogate-v4.protocol/4.0.0":
    raise ValueError("unsupported v4 protocol")
PROTOCOL_HASH = canonical_hash(PROTOCOL)


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def verify_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_name(path.name + ".sha256").read_bytes()
    if sidecar != f"{digest}  {path.name}\n".encode("ascii"):
        raise ValueError(f"invalid sidecar for {path.name}")
    return strict_load(path)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def exact_rank(count: int, probability: float) -> int:
    return min(count, math.ceil((count + 1) * probability))
