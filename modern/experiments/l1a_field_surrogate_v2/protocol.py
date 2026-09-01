"""Sealed protocol and byte-stable artifact helpers for v2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DECLARATION = ROOT / "predeclaration.json"
PARTITIONS = ROOT / "partitions.json"
GEOMETRY_PREFLIGHT = ROOT / "geometry-preflight.json"
SYNTHETIC_PREFLIGHT = ROOT / "synthetic-preflight.json"
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
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def load_protocol() -> dict[str, Any]:
    value = strict_load(DECLARATION)
    if value["schema_version"] != "cft-revival.l1a-field-surrogate-v2.protocol/2.0.0":
        raise ValueError("unsupported field-surrogate-v2 protocol")
    roles = value["sampling"]["roles"]
    covered = [index for bounds in roles.values() for index in range(*bounds)]
    if covered != list(range(112)) or len(set(covered)) != 112:
        raise ValueError("frozen roles are not disjoint and exhaustive")
    if value["fidelities"]["high"]["shape"] != [81, 145]:
        raise ValueError("accepted high grid changed")
    return value


PROTOCOL = load_protocol()
PROTOCOL_HASH = canonical_hash(PROTOCOL)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def verify_json(path: Path, *, git_bytes: bytes | None = None, sidecar_bytes: bytes | None = None) -> dict[str, Any]:
    data = path.read_bytes() if git_bytes is None else git_bytes
    sidecar = (
        path.with_name(path.name + ".sha256").read_bytes()
        if sidecar_bytes is None
        else sidecar_bytes
    )
    digest = hashlib.sha256(data).hexdigest()
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.replace(b"\r\n", b"\n") != expected:
        raise ValueError(f"invalid SHA-256 sidecar for {path.name}")
    value = json.loads(data.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value
