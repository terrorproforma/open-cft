"""Preregistered identities for field-surrogate v10."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime import canonical_bytes, semantic_sha256, strict_json_file

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DECLARATION = ROOT / "predeclaration.json"
GEOMETRY_PREFLIGHT = ROOT / "geometry-preflight.json"
PARTITIONS = ROOT / "partitions.json"
SYNTHETIC_PREFLIGHT = ROOT / "synthetic-preflight.json"
DEPENDENCY_LOCK = ROOT / "dependency-lock.json"
RESULTS = ROOT / "results"
CACHE = ROOT / ".runtime-cache"


def canonical_hash(value: object) -> str:
    return semantic_sha256(value)


def strict_load(path: Path) -> dict[str, Any]:
    value = strict_json_file(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


PROTOCOL = strict_load(DECLARATION)
if PROTOCOL["schema_version"] != "cft-revival.l1a-field-surrogate-v10.protocol/10.0.0":
    raise ValueError("unsupported v10 protocol")
PROTOCOL_HASH = canonical_hash(PROTOCOL)


def write_predeclared_json(path: Path, value: Mapping[str, Any]) -> str:
    """Write preparation evidence; production artifacts use RunContext."""

    data = canonical_bytes(dict(value))
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
