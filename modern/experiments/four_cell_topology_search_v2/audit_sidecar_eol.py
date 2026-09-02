"""Read-only post-hoc audit of four-cell search v2: protocol copy EOL vs the LF checkout.

The four-cell v2 preregistration (commit d6317910) and its single execution
(commit 7120e8ed) were made on a Windows checkout with ``core.autocrlf=true``.
``experiment.py`` hashed ``protocol.json`` as it lay in that working tree
(CRLF), so the immutable bundle records the SHA-256 of the CRLF bytes as
``protocol_sha256`` (``manifest.json``, ``dataset.json``,
``execution-lock.json``), as the ``sha256``/``bytes`` of the manifest artifact
entry ``preregistered-protocol.json`` and in the sidecar
``results/preregistered-protocol.json.sha256``. Git stores, and since the
repo-wide ``eol=lf`` pin (fab0eccc) checks out, the LF form, whose SHA-256
differs. This script proves, without writing anything under ``results/`` or
touching any frozen file, that

* ``results/preregistered-protocol.json`` on this checkout contains no
  ``\\r``, is byte-identical to the frozen ``protocol.json``,
  ``sha256(bytes.replace(b"\\n", b"\\r\\n"))`` equals the recorded digest and
  the recorded length is exactly one byte per line longer, i.e. the ONLY
  difference is the end-of-line byte,
* the protocol's canonical payload SHA-256 (EOL-independent) recomputes and
  equals the ``protocol_payload_sha256`` bound by ``dataset.json``, so the
  preregistered content is untouched,
* every other file listed in ``results/manifest.json`` is byte-exact against
  its recorded ``sha256``/``bytes`` and every other ``*.sha256`` sidecar under
  ``results/`` attests the LF checkout bytes exactly (the run wrote its own
  outputs in binary mode), and
* the manifest is sealed over the recorded (CRLF) digest, so the manifest
  payload digest still recomputes.

Nothing in this module imports the experiment package; it is standard library
only and can be run from any checkout.

Usage (from ``modern/``)::

    python -m experiments.four_cell_topology_search_v2.audit_sidecar_eol [--table] [--json PATH]

``--json`` may point anywhere except inside the experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
PROTOCOL_PATH = ROOT / "protocol.json"
PROTOCOL_COPY_PATH = RESULTS_ROOT / "preregistered-protocol.json"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"

EXPECTED_EOL_ONLY_PATHS = ("preregistered-protocol.json",)
PROTOCOL_LF_SHA256 = "5c195119c7a3c3c7e8b2c2d58e2e9836ac0ece6e000e52b0fd86c4718446c1b4"
PROTOCOL_RECORDED_SHA256 = "ec2e9a732b7d0e909ff742ebbbb0215e1102909c148b812306df6f0759f48e49"
PROTOCOL_PAYLOAD_SHA256 = "bd522269b87e555fee279bd669d34e2b6a98a31540c6f4687cfcf51b40614c33"
PROTOCOL_LF_BYTES = 10580
PROTOCOL_RECORDED_BYTES = 10811
BUNDLE_FILES_BINDING_PROTOCOL_FILE = ("manifest.json", "dataset.json", "execution-lock.json")
BUNDLE_FILES_BINDING_PROTOCOL_PAYLOAD = ("dataset.json",)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
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
    """Compare one file's checkout bytes with the digest the bundle recorded."""

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


def eol_equivalent_digest(path: Path, data: bytes) -> str | None:
    """Return the recorded digest iff ``path`` is the audited copy and differs by EOL only.

    Applies only to ``results/preregistered-protocol.json``; for any other
    path, or for any byte difference other than LF->CRLF, returns ``None``.
    """

    if path.resolve() != PROTOCOL_COPY_PATH.resolve() or b"\r" in data:
        return None
    if sha256_hex(data) != PROTOCOL_LF_SHA256:
        return None
    if sha256_hex(data.replace(b"\n", b"\r\n")) != PROTOCOL_RECORDED_SHA256:
        return None
    return PROTOCOL_RECORDED_SHA256


def audit_protocol() -> dict[str, Any]:
    frozen = PROTOCOL_PATH.read_bytes()
    copy = PROTOCOL_COPY_PATH.read_bytes()
    row = classify_file(PROTOCOL_COPY_PATH.name, copy, _read_sidecar_digest(PROTOCOL_COPY_PATH))
    protocol = json.loads(copy.decode("utf-8"))
    return {
        **row,
        "frozen_protocol_sha256": sha256_hex(frozen),
        "copy_equals_frozen_protocol": frozen == copy,
        "recomputed_payload_sha256": canonical_sha256(protocol),
        "payload_recomputes": canonical_sha256(protocol) == PROTOCOL_PAYLOAD_SHA256,
        "schema_version": protocol["schema_version"],
        "status": row["status"],
        "protocol_status": protocol["status"],
        "candidate_count": protocol["sampling"]["candidate_count"],
    }


def audit_results(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    manifest_bytes = (results_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_rows = []
    for entry in manifest["artifacts"]:
        data = (results_root / entry["path"]).read_bytes()
        row = classify_file(entry["path"], data, entry["sha256"])
        row["recorded_bytes"] = entry["bytes"]
        row["recorded_bytes_match"] = (
            entry["bytes"] == (len(data) if row["status"] == "byte_exact" else row["crlf_bytes"])
        )
        manifest_rows.append(row)
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
    for name in BUNDLE_FILES_BINDING_PROTOCOL_FILE:
        value = json.loads((results_root / name).read_bytes().decode("utf-8"))
        bound[name] = {"protocol_sha256": value["protocol_sha256"]}
        if name in BUNDLE_FILES_BINDING_PROTOCOL_PAYLOAD:
            bound[name]["protocol_payload_sha256"] = value["protocol_payload_sha256"]
    manifest_payload = {k: v for k, v in manifest.items() if k != "integrity"}
    return {
        "manifest_sha256": sha256_hex(manifest_bytes),
        "manifest_payload_recomputes": (
            manifest["integrity"]["algorithm"] == "sha256"
            and manifest["integrity"]["canonicalization"] == CANONICALIZATION
            and canonical_sha256(manifest_payload) == manifest["integrity"]["payload_sha256"]
        ),
        "manifest_single_execution": manifest["single_execution"],
        "preregistration_commit_sha": manifest["preregistration_commit_sha"],
        "summary": manifest["summary"],
        "manifest_entries": manifest_rows,
        "sidecar_entries": sidecar_rows,
        "files_containing_cr": cr_files,
        "protocol_bindings": bound,
    }


def audit() -> dict[str, Any]:
    protocol = audit_protocol()
    results = audit_results()
    rows = [*results["manifest_entries"], *results["sidecar_entries"]]
    counts = {"byte_exact": 0, "eol_only": 0, "mismatch": 0}
    for row in rows:
        counts[row["status"]] += 1
    eol_only_paths = tuple(sorted({row["path"] for row in rows if row["status"] == "eol_only"}))
    mismatch = [row for row in rows if row["status"] == "mismatch"]
    bindings = results["protocol_bindings"]
    bundle_binds_recorded_file_digest = all(
        bindings[name]["protocol_sha256"] == PROTOCOL_RECORDED_SHA256
        for name in BUNDLE_FILES_BINDING_PROTOCOL_FILE
    )
    bundle_binds_payload_digest = all(
        bindings[name]["protocol_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256
        for name in BUNDLE_FILES_BINDING_PROTOCOL_PAYLOAD
    )
    manifest_entry = next(
        row for row in results["manifest_entries"] if row["path"] == PROTOCOL_COPY_PATH.name
    )
    summary = results["summary"]
    passed = (
        not mismatch
        and eol_only_paths == EXPECTED_EOL_ONLY_PATHS
        and protocol["status"] == "eol_only"
        and protocol["copy_equals_frozen_protocol"]
        and protocol["payload_recomputes"]
        and protocol["checkout_sha256"] == PROTOCOL_LF_SHA256
        and protocol["recorded_sha256"] == PROTOCOL_RECORDED_SHA256
        and protocol["checkout_bytes"] == PROTOCOL_LF_BYTES
        and protocol["crlf_bytes"] == PROTOCOL_RECORDED_BYTES
        and protocol["crlf_bytes"] == protocol["checkout_bytes"] + protocol["lf_count"]
        and manifest_entry["recorded_bytes"] == PROTOCOL_RECORDED_BYTES
        and all(row["recorded_bytes_match"] for row in results["manifest_entries"])
        and not results["files_containing_cr"]
        and results["manifest_payload_recomputes"]
        and results["manifest_single_execution"] is True
        and bundle_binds_recorded_file_digest
        and bundle_binds_payload_digest
        and summary["evaluated_count"] == 128
        and summary["stable_count"] == 0
        and eol_equivalent_digest(PROTOCOL_COPY_PATH, PROTOCOL_COPY_PATH.read_bytes())
        == PROTOCOL_RECORDED_SHA256
    )
    return {
        "schema_version": "cft-revival.four-cell-topology-search-v2.posthoc-sidecar-eol-audit/1.0.0",
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
        "read_only": True,
        "passed": passed,
    }


def format_table(report: dict[str, Any]) -> str:
    header = (
        "| path | checkout bytes | recorded bytes | checkout sha256 | recorded sha256 "
        "| CRLF-recomputed sha256 | match |\n"
        "| --- | ---: | ---: | --- | --- | --- | --- |\n"
    )
    seen: set[str] = set()
    rows = []
    for row in report["eol_only"]:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        rows.append(
            f"| `{row['path']}` | {row['checkout_bytes']} | {row['crlf_bytes']} "
            f"| `{row['checkout_sha256']}` | `{row['recorded_sha256']}` "
            f"| `{row['crlf_recomputed_sha256']}` "
            f"| {'CRLF == recorded' if row['crlf_matches_recorded'] else 'NO'} |"
        )
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
