"""Strict preregistration and result-integrity helpers for sweep v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.json"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"


@dataclass(frozen=True)
class EolAuditedSidecar:
    """One file whose frozen sidecar recorded CRLF bytes (see POSTHOC_AUDIT.md)."""

    lf_sha256: str
    recorded_sha256: str


# Post-hoc end-of-line audit (``POSTHOC_AUDIT.md``, ``audit_sidecar_eol.py``).
# ``protocol.json.sha256`` was frozen at preregistration commit 092f5fae on a
# ``core.autocrlf=true`` checkout, so it records the SHA-256 of the CRLF working
# tree bytes, while Git stores (and, since the repo-wide ``eol=lf`` pin
# fab0eccc, checks out) the LF form. The immutable results bundle repeats that
# recorded digest as ``protocol_file_sha256``. For exactly this one file the
# recorded digest is accepted iff the LF checkout bytes hash to the audited LF
# digest AND ``sha256(bytes.replace(b"\n", b"\r\n"))`` reproduces the recorded
# digest. Every other sidecar in this experiment must be byte-exact.
EOL_AUDITED_SIDECARS: Mapping[Path, EolAuditedSidecar] = {
    PROTOCOL_PATH: EolAuditedSidecar(
        lf_sha256="2a5ba9e46c777225384539a4c453a43aa3298c956b32b022cc5ddeac72ba874c",
        recorded_sha256="64b2c58c3cecb2ea1836d2bf48e23ff83dffb114866bf21e7135b411beaa2b2c",
    ),
}

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


def eol_equivalent_digest(path: Path, data: bytes) -> str | None:
    """Return the audited recorded digest iff ``path`` is EOL-equivalent to it.

    Applies only to the files listed in :data:`EOL_AUDITED_SIDECARS`; for any
    other path, or for any byte difference other than LF→CRLF, returns ``None``.
    """

    audited = EOL_AUDITED_SIDECARS.get(path.resolve())
    if audited is None or b"\r" in data:
        return None
    if hashlib.sha256(data).hexdigest() != audited.lf_sha256:
        return None
    crlf_digest = hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest()
    if crlf_digest != audited.recorded_sha256:
        return None
    return audited.recorded_sha256


def verify_sidecar(path: Path) -> str:
    """Check ``<path>.sha256`` and return the digest it attests.

    The attested digest is the SHA-256 of the file bytes. The single audited
    exception (``EOL_AUDITED_SIDECARS``) returns the recorded CRLF-era digest,
    which is the identity the immutable results bundle binds to.
    """

    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    recorded = sidecar.read_text(encoding="ascii")
    if recorded == f"{digest}  {path.name}\n":
        return digest
    audited_digest = eol_equivalent_digest(path, data)
    if audited_digest is not None and recorded == f"{audited_digest}  {path.name}\n":
        return audited_digest
    raise ValueError(f"invalid SHA-256 sidecar for {path.name}")


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
