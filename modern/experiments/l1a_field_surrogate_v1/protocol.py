"""Sealed protocol and artifact helpers for the prospective L1a experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DECLARATION = ROOT / "predeclaration.json"
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
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def load_declaration() -> dict[str, Any]:
    value = strict_load(DECLARATION)
    if value["schema_version"] != "cft-revival.l1a-field-surrogate-v1.protocol/1.0.0":
        raise ValueError("unsupported L1a field-surrogate protocol")
    roles = value["sampling"]["roles"]
    covered = [index for name in roles for index in range(*roles[name])]
    if covered != list(range(value["sampling"]["rows"])) or len(set(covered)) != len(covered):
        raise ValueError("roles must be disjoint and exhaustive")
    if value["fidelities"]["high"]["shape"] != [81, 145]:
        raise ValueError("high fidelity must remain the accepted 81x145 discretization")
    return value


PROTOCOL = load_declaration()
PROTOCOL_HASH = canonical_hash(PROTOCOL)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires observations")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
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


def verify_json(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = f"{digest}  {path.name}\n"
    if path.with_name(path.name + ".sha256").read_text(encoding="ascii") != expected:
        raise ValueError(f"invalid sidecar for {path}")
    return strict_load(path)
