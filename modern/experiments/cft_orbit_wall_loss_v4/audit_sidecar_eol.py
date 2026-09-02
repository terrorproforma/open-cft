"""Read-only post-hoc audit of the v4 results bundle: sidecar EOL vs manifest.

The v4 campaign ran on Windows with orbit_mc 1.6.0, whose ``write_artifact``
wrote ``<case>-orbit.json.sha256`` through the text layer without
``newline="\\n"``. The bundle's ``manifest.json`` therefore recorded the CRLF
bytes of nine ``artifacts/orbits/<case>.json.sha256`` files, while Git stores
(and every checkout receives) the LF form. This script proves, without writing
anything under ``results/``, that

* every manifest file entry other than those nine is byte-exact on this
  checkout,
* for each of the nine, ``sha256(checkout_bytes.replace(b"\\n", b"\\r\\n"))``
  equals the recorded ``byte_sha256`` and the recorded length is exactly one
  byte longer, i.e. the ONLY difference is the end-of-line byte,
* the runtime sidecar ``<path>.sha256.json`` agrees with the manifest (both
  hashed the CRLF bytes at write time, so the recording layer was internally
  consistent), and
* the nine ``.json.gz`` orbit artifacts decompress to canonical bytes whose
  SHA-256 equals the digest stated inside the affected sidecar and whose
  ``integrity.payload_sha256`` recomputes: the evidence is untouched.

Usage (from ``modern/``)::

    python -m experiments.cft_orbit_wall_loss_v4.audit_sidecar_eol [--json PATH]

``--json`` may point anywhere except inside ``results/``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cft_revival.orbit_mc.artifacts import canonical_bytes, content_hash

EXPERIMENT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENT / "results"
MANIFEST_NAME = "manifest.json"
RUNTIME_SIDECAR_SUFFIX = ".sha256.json"
CASES = (
    "enlarged-2N", "enlarged-4N", "enlarged-N",
    "primary-2N", "primary-4N", "primary-N",
    "refined-2N", "refined-4N", "refined-N",
)
EXPECTED_EOL_ONLY_PATHS = tuple(f"artifacts/orbits/{case}.json.sha256" for case in CASES)
WALL_Z_MAX_M = 0.018


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def classify_entry(entry: dict[str, Any], data: bytes) -> dict[str, Any]:
    """Compare one manifest file entry with the bytes found on this checkout."""

    checkout_sha256 = sha256_hex(data)
    recorded_sha256 = str(entry["byte_sha256"])
    crlf = data.replace(b"\n", b"\r\n")
    crlf_sha256 = sha256_hex(crlf)
    if checkout_sha256 == recorded_sha256 and len(data) == entry["bytes"]:
        status = "byte_exact"
    elif (
        b"\r" not in data
        and crlf_sha256 == recorded_sha256
        and len(crlf) == entry["bytes"]
    ):
        status = "eol_only"
    else:
        status = "mismatch"
    return {
        "path": entry["path"],
        "contract": entry.get("contract"),
        "status": status,
        "checkout_bytes": len(data),
        "recorded_bytes": entry["bytes"],
        "checkout_sha256": checkout_sha256,
        "recorded_sha256": recorded_sha256,
        "crlf_recomputed_sha256": crlf_sha256,
        "crlf_matches_recorded": crlf_sha256 == recorded_sha256,
        "lf_count": data.count(b"\n"),
    }


def _launch_cell(position_m: list[float]) -> str:
    z_mm = round(position_m[2] * 1000.0, 3)
    return f"z={z_mm:g}mm"


def audit_orbit_artifact(results_root: Path, sidecar_path: str) -> dict[str, Any]:
    """Check the .json.gz behind an affected sidecar and its runtime sidecar."""

    sidecar_bytes = (results_root / sidecar_path).read_bytes()
    stated_digest, stated_name = sidecar_bytes.decode("ascii").split()
    gz_path = results_root / (sidecar_path[: -len(".sha256")] + ".gz")
    raw = gzip.decompress(gz_path.read_bytes())
    artifact = json.loads(raw.decode("utf-8"))
    payload = {key: value for key, value in artifact.items() if key != "integrity"}
    runtime_sidecar = json.loads(
        (results_root / (sidecar_path + RUNTIME_SIDECAR_SUFFIX)).read_bytes()
    )
    terminations: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    by_cell: dict[str, Counter[str]] = {}
    launches = {item["launch_id"]: item for item in artifact["launches"]}
    for result in artifact["results"]:
        launch = launches[result["launch_id"]]
        terminations[result["termination"]] += 1
        resolutions[result["event_witness"]["event_resolution"]] += 1
        key = f"{_launch_cell(launch['position_m'])} D{launch['parallel_direction']:+d}"
        by_cell.setdefault(key, Counter())[result["termination"]] += 1
    return {
        "sidecar_path": sidecar_path,
        "artifact_path": gz_path.relative_to(results_root).as_posix(),
        "sidecar_states_name": stated_name,
        "sidecar_states_sha256": stated_digest,
        "decompressed_sha256": sha256_hex(raw),
        "content_matches_sidecar": sha256_hex(raw) == stated_digest,
        "decompressed_is_canonical": canonical_bytes(artifact) == raw,
        "integrity_recomputes": content_hash(payload) == artifact["integrity"]["payload_sha256"],
        "schema_version": artifact["schema_version"],
        "code_sha256": artifact["identities"]["code_sha256"],
        "runtime_sidecar_bytes": runtime_sidecar["bytes"],
        "runtime_sidecar_sha256": runtime_sidecar["byte_sha256"],
        "trial_count": artifact["summary"]["trial_count"],
        "terminations": dict(sorted(terminations.items())),
        "event_resolutions": dict(sorted(resolutions.items())),
        "terminations_by_launch_cell_and_direction": {
            key: dict(sorted(counter.items())) for key, counter in sorted(by_cell.items())
        },
    }


def audit(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    manifest_bytes = (results_root / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    rows = [
        classify_entry(entry, (results_root / entry["path"]).read_bytes())
        for entry in manifest["artifacts"]
        if entry["type"] == "file"
    ]
    by_status: dict[str, list[dict[str, Any]]] = {"byte_exact": [], "eol_only": [], "mismatch": []}
    for row in rows:
        by_status[row["status"]].append(row)
    eol_only_paths = tuple(row["path"] for row in by_status["eol_only"])
    orbit_checks = [audit_orbit_artifact(results_root, path) for path in eol_only_paths]
    runtime_sidecars_agree = all(
        check["runtime_sidecar_sha256"] == row["recorded_sha256"]
        and check["runtime_sidecar_bytes"] == row["recorded_bytes"]
        for row, check in zip(by_status["eol_only"], orbit_checks, strict=True)
    )
    pooled: Counter[str] = Counter()
    pooled_resolutions: Counter[str] = Counter()
    pooled_cells: dict[str, Counter[str]] = {}
    for check in orbit_checks:
        pooled.update(check["terminations"])
        pooled_resolutions.update(check["event_resolutions"])
        for key, counts in check["terminations_by_launch_cell_and_direction"].items():
            pooled_cells.setdefault(key, Counter()).update(counts)
    orbit_evidence_intact = all(
        check["content_matches_sidecar"]
        and check["decompressed_is_canonical"]
        and check["integrity_recomputes"]
        for check in orbit_checks
    )
    passed = (
        not by_status["mismatch"]
        and eol_only_paths == EXPECTED_EOL_ONLY_PATHS
        and orbit_evidence_intact
        and runtime_sidecars_agree
    )
    return {
        "schema_version": "cft-revival.cft-orbit-wall-loss-v4.posthoc-sidecar-eol-audit/1.0.0",
        "results_root": results_root.as_posix(),
        "manifest_sha256": sha256_hex(manifest_bytes),
        "manifest_state": manifest["state"],
        "manifest_artifact_count": manifest["artifact_count"],
        "file_entries": len(rows),
        "directory_entries": manifest["artifact_count"] - len(rows),
        "counts": {status: len(items) for status, items in by_status.items()},
        "eol_only": by_status["eol_only"],
        "mismatch": by_status["mismatch"],
        "expected_eol_only_paths": list(EXPECTED_EOL_ONLY_PATHS),
        "eol_only_paths_are_exactly_expected": eol_only_paths == EXPECTED_EOL_ONLY_PATHS,
        "runtime_sidecars_agree_with_manifest": runtime_sidecars_agree,
        "orbit_artifacts": orbit_checks,
        "orbit_evidence_intact": orbit_evidence_intact,
        "pooled_terminations": dict(sorted(pooled.items())),
        "pooled_event_resolutions": dict(sorted(pooled_resolutions.items())),
        "pooled_terminations_by_launch_cell_and_direction": {
            key: dict(sorted(counter.items())) for key, counter in sorted(pooled_cells.items())
        },
        "wall_z_max_m": WALL_Z_MAX_M,
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
        f"| `{row['path']}` | {row['checkout_bytes']} | {row['recorded_bytes']} "
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
        if RESULTS_ROOT.resolve() in target.parents or target == RESULTS_ROOT.resolve():
            parser.error("--json must not point inside results/ (immutable evidence)")
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
