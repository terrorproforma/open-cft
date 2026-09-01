"""Strict preregistration and result-integrity helpers for sweep v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.json"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"

PROTOCOL_KEYS = {
    "schema_version",
    "classification",
    "execution",
    "sampling",
    "geometry",
    "field",
    "qoi_policy",
    "objectives",
    "representative_policy",
    "terminal_acceptance",
    "replay_contract",
    "failure_taxonomy",
    "claim_limits",
    "integrity",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {path.name}")

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject,
    )
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain one object")
    return loaded


def verify_sidecar(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise ValueError(f"invalid SHA-256 sidecar for {path.name}")
    return digest


def validate_sealed(value: Mapping[str, Any], *, expected_keys: set[str] | None = None) -> None:
    if expected_keys is not None and set(value) != expected_keys:
        raise ValueError("sealed object keys do not match its closed schema")
    integrity = value.get("integrity")
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "canonicalization",
        "payload_sha256",
    }:
        raise ValueError("invalid integrity declaration")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != CANONICALIZATION
    ):
        raise ValueError("unsupported integrity declaration")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if integrity["payload_sha256"] != stable_hash(payload):
        raise ValueError("canonical payload SHA-256 mismatch")


def load_protocol() -> dict[str, Any]:
    verify_sidecar(PROTOCOL_PATH)
    protocol = strict_json(PROTOCOL_PATH)
    validate_sealed(protocol, expected_keys=PROTOCOL_KEYS)
    if (
        protocol["schema_version"]
        != "cft-revival.experiment.l1a-geometry-sweep-v2.protocol/1.0.0"
    ):
        raise ValueError("unsupported sweep-v2 protocol")
    if protocol["execution"]["case_count"] != 96:
        raise ValueError("sweep-v2 protocol must remain exactly 96 cases")
    gates = protocol["terminal_acceptance"]["gates"]
    if len(gates) != 7 or len({gate["gate_id"] for gate in gates}) != 7:
        raise ValueError("sweep-v2 protocol requires exactly seven unique terminal gates")
    roles = protocol["representative_policy"]["roles"]
    if len(roles) != 5 or len({role["role"] for role in roles}) != 5:
        raise ValueError("sweep-v2 protocol requires exactly five representative roles")
    return protocol


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {
        **body,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": CANONICALIZATION,
            "payload_sha256": stable_hash(body),
        },
    }


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
        newline="\n",
    )
    return digest


def write_sealed(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value = seal(payload)
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    digest = write_bytes(path, data)
    loaded = strict_json(path)
    validate_sealed(loaded)
    verify_sidecar(path)
    return value, digest
