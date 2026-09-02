"""Read-only post-hoc audit of sweep v2: protocol sidecar EOL vs the LF checkout.

The sweep-v2 preregistration (commit 092f5fae) was made on a Windows checkout
with ``core.autocrlf=true``. ``protocol.json.sha256`` therefore records the
SHA-256 of the CRLF working-tree bytes of ``protocol.json``, and the one
authorised execution (commit f30cb42e) copied that digest into the immutable
bundle as ``protocol_file_sha256`` (manifest, raw results, execution lock).
Git stores, and since the repo-wide ``eol=lf`` pin (fab0eccc) checks out, the
LF form, whose SHA-256 differs. This script proves, without writing anything
under ``results/`` or touching any frozen file, that

* ``protocol.json`` on this checkout contains no ``\\r``,
  ``sha256(bytes.replace(b"\\n", b"\\r\\n"))`` equals the recorded sidecar
  digest and the recorded length is exactly one byte per line longer, i.e. the
  ONLY difference is the end-of-line byte,
* the sealed protocol's canonical payload SHA-256 (EOL-independent) recomputes
  and equals the ``protocol_payload_sha256`` bound by every bundle file, so the
  preregistered content is untouched,
* every file listed in ``results/manifest.json`` is byte-exact against its
  recorded ``file_sha256`` and every ``*.sha256`` sidecar under ``results/``
  attests the LF checkout bytes exactly (the run wrote its own outputs in
  binary mode), and
* the tolerance constants in ``protocol.py`` are exactly the audited digests.

Usage (from ``modern/``)::

    python -m experiments.l1a_geometry_sweep_v2.audit_sidecar_eol [--table] [--json PATH]

``--json`` may point anywhere except inside the experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .protocol import (
    CANONICALIZATION,
    EOL_AUDITED_SIDECARS,
    PROTOCOL_PATH,
    ROOT,
)

RESULTS_ROOT = ROOT / "results"
EXPECTED_EOL_ONLY_PATHS = ("protocol.json",)
PROTOCOL_LF_SHA256 = "2a5ba9e46c777225384539a4c453a43aa3298c956b32b022cc5ddeac72ba874c"
PROTOCOL_RECORDED_SHA256 = "64b2c58c3cecb2ea1836d2bf48e23ff83dffb114866bf21e7135b411beaa2b2c"
PROTOCOL_PAYLOAD_SHA256 = "da319f2271d56b0d0c883b76d3106b094359a608b560d58ac7801de1293ecbc8"
BUNDLE_FILES_BINDING_PROTOCOL_FILE = ("manifest.json", "raw-results.json", "execution-lock.json")
BUNDLE_FILES_BINDING_PROTOCOL_PAYLOAD = (
    "manifest.json",
    "raw-results.json",
    "summary.json",
    "execution-lock.json",
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256_hex(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    )


def _read_sidecar_digest(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    digest, name = sidecar.read_text(encoding="ascii").split()
    if name != path.name:
        raise ValueError(f"sidecar for {path.name} names {name!r}")
    return digest


def classify_file(relative: str, data: bytes, recorded_sha256: str) -> dict[str, Any]:
    """Compare one file's checkout bytes with the digest its sidecar recorded."""

    checkout_sha256 = sha256_hex(data)
    crlf = data.replace(b"\n", b"\r\n")
    crlf_sha256 = sha256_hex(crlf)
    if checkout_sha256 == recorded_sha256:
        status = "byte_exact"
    elif b"\r" not in data and crlf_sha256 == recorded_sha256:
        status = "eol_only"
    else:
        status = "mismatch"
    return {
        "path": relative,
        "status": status,
        "checkout_bytes": len(data),
        "crlf_bytes": len(crlf),
        "checkout_sha256": checkout_sha256,
        "recorded_sha256": recorded_sha256,
        "crlf_recomputed_sha256": crlf_sha256,
        "crlf_matches_recorded": crlf_sha256 == recorded_sha256,
        "lf_count": data.count(b"\n"),
        "contains_cr": b"\r" in data,
    }


def audit_protocol(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    data = protocol_path.read_bytes()
    row = classify_file(protocol_path.name, data, _read_sidecar_digest(protocol_path))
    protocol = json.loads(data.decode("utf-8"))
    integrity = protocol["integrity"]
    payload = {key: value for key, value in protocol.items() if key != "integrity"}
    recomputed_payload = _canonical_sha256(payload)
    return {
        **row,
        "integrity_algorithm": integrity["algorithm"],
        "integrity_canonicalization": integrity["canonicalization"],
        "recorded_payload_sha256": integrity["payload_sha256"],
        "recomputed_payload_sha256": recomputed_payload,
        "payload_recomputes": (
            integrity["algorithm"] == "sha256"
            and integrity["canonicalization"] == CANONICALIZATION
            and recomputed_payload == integrity["payload_sha256"]
        ),
        "schema_version": protocol["schema_version"],
    }


def audit_results(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    manifest_bytes = (results_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_rows = [
        classify_file(
            entry["path"],
            (results_root / entry["path"]).read_bytes(),
            entry["file_sha256"],
        )
        for entry in manifest["deterministic_files"]
    ]
    sidecar_rows = []
    for sidecar in sorted(results_root.rglob("*.sha256")):
        target = sidecar.with_name(sidecar.name[: -len(".sha256")])
        sidecar_rows.append(
            classify_file(
                target.relative_to(results_root).as_posix(),
                target.read_bytes(),
                _read_sidecar_digest(target),
            )
        )
    cr_files = [
        path.relative_to(results_root).as_posix()
        for path in sorted(results_root.rglob("*"))
        if path.is_file() and b"\r" in path.read_bytes()
    ]
    bound: dict[str, dict[str, str]] = {}
    for name in BUNDLE_FILES_BINDING_PROTOCOL_PAYLOAD:
        value = json.loads((results_root / name).read_bytes().decode("utf-8"))
        bound[name] = {"protocol_payload_sha256": value["protocol_payload_sha256"]}
        if name in BUNDLE_FILES_BINDING_PROTOCOL_FILE:
            bound[name]["protocol_file_sha256"] = value["protocol_file_sha256"]
    return {
        "manifest_sha256": sha256_hex(manifest_bytes),
        "manifest_terminal_status": manifest["terminal_status"],
        "preregistration_commit_sha": manifest["preregistration_commit_sha"],
        "manifest_entries": manifest_rows,
        "sidecar_entries": sidecar_rows,
        "files_containing_cr": cr_files,
        "protocol_bindings": bound,
    }


def audit() -> dict[str, Any]:
    protocol = audit_protocol()
    results = audit_results()
    rows = [protocol, *results["manifest_entries"], *results["sidecar_entries"]]
    counts = {"byte_exact": 0, "eol_only": 0, "mismatch": 0}
    for row in rows:
        counts[row["status"]] += 1
    eol_only_paths = tuple(row["path"] for row in rows if row["status"] == "eol_only")
    mismatch = [row for row in rows if row["status"] == "mismatch"]
    bindings = results["protocol_bindings"]
    bundle_binds_recorded_file_digest = all(
        bindings[name]["protocol_file_sha256"] == PROTOCOL_RECORDED_SHA256
        for name in BUNDLE_FILES_BINDING_PROTOCOL_FILE
    )
    bundle_binds_payload_digest = all(
        bindings[name]["protocol_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256
        for name in BUNDLE_FILES_BINDING_PROTOCOL_PAYLOAD
    )
    audited = EOL_AUDITED_SIDECARS.get(PROTOCOL_PATH)
    tolerance_bound_to_audit = (
        set(EOL_AUDITED_SIDECARS) == {PROTOCOL_PATH}
        and audited is not None
        and audited.lf_sha256 == PROTOCOL_LF_SHA256 == protocol["checkout_sha256"]
        and audited.recorded_sha256 == PROTOCOL_RECORDED_SHA256 == protocol["recorded_sha256"]
    )
    passed = (
        not mismatch
        and eol_only_paths == EXPECTED_EOL_ONLY_PATHS
        and protocol["status"] == "eol_only"
        and protocol["payload_recomputes"]
        and protocol["recorded_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256
        and protocol["crlf_bytes"] == protocol["checkout_bytes"] + protocol["lf_count"]
        and not results["files_containing_cr"]
        and bundle_binds_recorded_file_digest
        and bundle_binds_payload_digest
        and tolerance_bound_to_audit
    )
    return {
        "schema_version": "cft-revival.l1a-geometry-sweep-v2.posthoc-sidecar-eol-audit/1.0.0",
        "experiment_root": ROOT.as_posix(),
        "protocol": protocol,
        "results": {
            key: value
            for key, value in results.items()
            if key not in {"manifest_entries", "sidecar_entries"}
        },
        "counts": counts,
        "file_entries": len(rows),
        "eol_only": [row for row in rows if row["status"] == "eol_only"],
        "mismatch": mismatch,
        "expected_eol_only_paths": list(EXPECTED_EOL_ONLY_PATHS),
        "eol_only_paths_are_exactly_expected": eol_only_paths == EXPECTED_EOL_ONLY_PATHS,
        "bundle_binds_recorded_protocol_file_digest": bundle_binds_recorded_file_digest,
        "bundle_binds_protocol_payload_digest": bundle_binds_payload_digest,
        "tolerance_bound_to_audit": tolerance_bound_to_audit,
        "read_only": True,
        "passed": passed,
    }


def format_table(report: dict[str, Any]) -> str:
    header = (
        "| path | checkout bytes | recorded bytes | checkout sha256 | recorded sha256 "
        "| CRLF-recomputed sha256 | match |\n"
        "| --- | ---: | ---: | --- | --- | --- | --- |\n"
    )
    rows = [
        f"| `{row['path']}` | {row['checkout_bytes']} | {row['crlf_bytes']} "
        f"| `{row['checkout_sha256']}` | `{row['recorded_sha256']}` "
        f"| `{row['crlf_recomputed_sha256']}` "
        f"| {'CRLF == recorded' if row['crlf_matches_recorded'] else 'NO'} |"
        for row in report["eol_only"]
    ]
    return header + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--table", action="store_true", help="print the Markdown table only")
    arguments = parser.parse_args(argv)
    if arguments.json is not None:
        target = arguments.json.resolve()
        if ROOT.resolve() in target.parents or target == ROOT.resolve():
            parser.error("--json must not point inside the experiment directory")
    report = audit()
    if arguments.table:
        sys.stdout.write(format_table(report))
    else:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.json is not None:
        arguments.json.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
