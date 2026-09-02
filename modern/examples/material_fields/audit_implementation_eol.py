"""Read-only audit: material-fields implementation digests vs the LF checkout.

The L1b v1.4 example artifacts under ``artifacts/`` bind every raw run to the
SHA-256 of the ``cft_revival.material_fields`` source files
(``implementation_sha256`` over the solver files, ``evidence_implementation_sha256``
over the evidence files; see ``numerics._implementation_sha256``). They were
generated at commit ``8603a905`` on a Windows checkout with
``core.autocrlf=true``, so the recorded digests are hashes of the **CRLF**
working-tree bytes, while Git stores (and, since the repo-wide ``eol=lf`` pin
``fab0eccc``, checks out) the LF form. Strict validation compares the recorded
digests with the live source bytes and therefore refused every artifact on an
LF checkout (``raw run hash binding failed``).

This script proves, without writing anything, that

* every source file is CR-free on this checkout and the CRLF-era digests
  recorded at ``8603a905`` are exactly ``sha256`` over the same files with
  ``\\n`` replaced by ``\\r\\n`` (the historical proof is anchored to the Git
  blobs of that commit, so it stays valid after the artifacts are re-bound),
* the digests recorded by the artifacts on disk now identify the live source
  either byte-exactly (after re-binding with ``refresh_artifact_metadata.py``)
  or through the audited CRLF rule (before), and never anything else, and
* every artifact and viewer is byte-exact against its ``.sha256`` sidecar and
  the manifest, and every sealed payload digest recomputes.

Usage (from ``modern/``)::

    python examples/material_fields/audit_implementation_eol.py [--json PATH]

``--json`` may point anywhere except inside ``examples/material_fields``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
MODERN = HERE.parents[1]
REPO = MODERN.parent
SOURCE = MODERN / "src" / "cft_revival" / "material_fields"
SOURCE_REL = "modern/src/cft_revival/material_fields"
ARTIFACTS_REL = "modern/examples/material_fields/artifacts"

# Commit that generated the v1.4 artifacts and last touched the source
# (`record L1b v1.4 reduced screening evidence`, 2026-09-02T09:02:47+10:00).
CRLF_ERA_COMMIT = "8603a905f8b19873e9a91c1afd237864e8b31aff"
FILE_SETS: dict[str, tuple[str, ...]] = {
    "evidence": (
        "acceptance.py",
        "adapters.py",
        "artifacts.py",
        "models.py",
        "numerics.py",
        "replay.py",
        "warp_solver.py",
    ),
    "warp": ("adapters.py", "models.py", "numerics.py", "warp_solver.py"),
    "python": ("adapters.py", "models.py", "numerics.py"),
}
# The three CRLF-era digests recorded by every v1.4 artifact at 8603a905.
CRLF_ERA_DIGESTS: dict[str, str] = {
    "evidence": "d229f62d7ba6289646291d925f404785ab879b91f59185a91a90c327e92966b8",
    "warp": "dc988f4b01648e825ac7a1934b8ddca88ad53d1fa5859c8471e1dfcec745cd0b",
    "python": "734cff6aabe3964690ee6ccfa3bc5c3f9f88f2bc7184ffc9390a06b5b903e6b5",
}
DIGEST_KEYS = ("implementation_sha256", "evidence_implementation_sha256")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256_hex(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    )


def implementation_digest(read: Callable[[str], bytes], filenames: tuple[str, ...], crlf: bool) -> str:
    """Mirror ``numerics._implementation_sha256`` over arbitrary byte sources."""

    digest = hashlib.sha256()
    for filename in sorted(filenames):
        data = read(filename)
        if crlf:
            data = data.replace(b"\n", b"\r\n")
        digest.update(filename.encode("utf-8"))
        digest.update(data)
    return digest.hexdigest()


def _git_blob(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{relative}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout


def _walk_digests(value: Any, path: str = "") -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in DIGEST_KEYS and isinstance(item, str):
                found.append((f"{path}/{key}", key, item))
            found.extend(_walk_digests(item, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_digests(item, f"{path}[{index}]"))
    return found


def audit_source() -> dict[str, Any]:
    files = {}
    for name in FILE_SETS["evidence"]:
        data = (SOURCE / name).read_bytes()
        files[name] = {
            "bytes": len(data),
            "lf_count": data.count(b"\n"),
            "contains_cr": b"\r" in data,
            "sha256": sha256_hex(data),
        }
    live = {
        role: {
            "files": list(names),
            "lf_sha256": implementation_digest(lambda f: (SOURCE / f).read_bytes(), names, False),
            "crlf_sha256": implementation_digest(lambda f: (SOURCE / f).read_bytes(), names, True),
        }
        for role, names in FILE_SETS.items()
    }
    return {"files": files, "digests": live}


def audit_history() -> dict[str, Any]:
    """Anchor the CRLF-era digests to the Git blobs of the generating commit."""

    def read(name: str) -> bytes:
        return _git_blob(CRLF_ERA_COMMIT, f"{SOURCE_REL}/{name}")

    blobs = {name: sha256_hex(read(name)) for name in FILE_SETS["evidence"]}
    era = {}
    for role, names in FILE_SETS.items():
        era[role] = {
            "blob_lf_sha256": implementation_digest(read, names, False),
            "blob_crlf_sha256": implementation_digest(read, names, True),
            "recorded_at_era": CRLF_ERA_DIGESTS[role],
        }
        era[role]["crlf_reproduces_recorded"] = (
            era[role]["blob_crlf_sha256"] == CRLF_ERA_DIGESTS[role]
        )
    recorded_at_era: dict[str, int] = {}
    for relative in sorted(
        subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", CRLF_ERA_COMMIT, "--", ARTIFACTS_REL],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    ):
        if not relative.endswith(".json") or relative.endswith("manifest.json"):
            continue
        for _, _, digest in _walk_digests(json.loads(_git_blob(CRLF_ERA_COMMIT, relative))):
            recorded_at_era[digest] = recorded_at_era.get(digest, 0) + 1
    return {
        "commit": CRLF_ERA_COMMIT,
        "source_blob_sha256": blobs,
        "digests": era,
        "artifact_digests_recorded_at_era": recorded_at_era,
        "era_artifacts_recorded_exactly_the_three": set(recorded_at_era) == set(
            CRLF_ERA_DIGESTS.values()
        ),
        "all_crlf_reproduce_recorded": all(item["crlf_reproduces_recorded"] for item in era.values()),
    }


def audit_artifacts(live: dict[str, Any]) -> dict[str, Any]:
    lf_by_digest = {item["lf_sha256"]: role for role, item in live["digests"].items()}
    crlf_by_digest = {item["crlf_sha256"]: role for role, item in live["digests"].items()}
    manifest = json.loads((ARTIFACTS / "manifest.json").read_bytes())
    manifest_entries = {entry["artifact"]: entry for entry in manifest["designs"]}
    rows = []
    recorded: dict[str, dict[str, Any]] = {}
    for path in sorted(ARTIFACTS.glob("*.json")):
        data = path.read_bytes()
        sidecar_digest = (
            path.with_name(path.name + ".sha256").read_text(encoding="ascii").split()[0]
        )
        value = json.loads(data)
        payload = {key: item for key, item in value.items() if key != "integrity"}
        row = {
            "path": path.name,
            "bytes": len(data),
            "contains_cr": b"\r" in data,
            "sha256": sha256_hex(data),
            "sidecar_sha256": sidecar_digest,
            "sidecar_byte_exact": sha256_hex(data) == sidecar_digest,
            "payload_recomputes": (
                value["integrity"]["payload_sha256"] == _canonical_sha256(payload)
            ),
        }
        if path.name in manifest_entries:
            entry = manifest_entries[path.name]
            row["manifest_file_byte_exact"] = entry["artifact_file_sha256"] == row["sha256"]
            row["manifest_payload_matches"] = (
                entry["artifact_payload_sha256"] == value["integrity"]["payload_sha256"]
            )
        rows.append(row)
        for location, key, digest in _walk_digests(value):
            item = recorded.setdefault(
                digest,
                {"digest": digest, "occurrences": 0, "keys": set(), "status": None, "role": None},
            )
            item["occurrences"] += 1
            item["keys"].add(key)
    for digest, item in recorded.items():
        if digest in lf_by_digest:
            item["status"], item["role"] = "byte_exact", lf_by_digest[digest]
        elif digest in crlf_by_digest:
            item["status"], item["role"] = "eol_only", crlf_by_digest[digest]
        else:
            item["status"] = "mismatch"
        item["keys"] = sorted(item["keys"])
    counts = {"byte_exact": 0, "eol_only": 0, "mismatch": 0}
    for item in recorded.values():
        counts[item["status"]] += 1
    return {
        "manifest_sha256": sha256_hex((ARTIFACTS / "manifest.json").read_bytes()),
        "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "files": rows,
        "all_files_byte_exact": all(
            row["sidecar_byte_exact"]
            and row["payload_recomputes"]
            and not row["contains_cr"]
            and row.get("manifest_file_byte_exact", True)
            and row.get("manifest_payload_matches", True)
            for row in rows
        ),
        "recorded_implementation_digests": sorted(
            recorded.values(), key=lambda item: (item["role"] or "", item["digest"])
        ),
        "counts": counts,
    }


def audit() -> dict[str, Any]:
    live = audit_source()
    history = audit_history()
    artifacts = audit_artifacts(live)
    source_is_lf = not any(item["contains_cr"] for item in live["files"].values())
    passed = (
        source_is_lf
        and history["all_crlf_reproduce_recorded"]
        and history["era_artifacts_recorded_exactly_the_three"]
        and artifacts["all_files_byte_exact"]
        and artifacts["counts"]["mismatch"] == 0
        and len(artifacts["recorded_implementation_digests"]) == 3
    )
    return {
        "schema_version": "cft-revival.material-fields.posthoc-implementation-eol-audit/1.0.0",
        "source": live,
        "source_is_lf": source_is_lf,
        "history": history,
        "artifacts": artifacts,
        "live_state": (
            "rebound_lf"
            if artifacts["counts"] == {"byte_exact": 3, "eol_only": 0, "mismatch": 0}
            else "crlf_era"
            if artifacts["counts"] == {"byte_exact": 0, "eol_only": 3, "mismatch": 0}
            else "inconsistent"
        ),
        "read_only": True,
        "passed": passed,
    }


def format_table(report: dict[str, Any]) -> str:
    header = (
        "| digest role | files | LF sha256 (blob) | CRLF-recomputed sha256 | recorded at 8603a905 | match |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for role, names in FILE_SETS.items():
        era = report["history"]["digests"][role]
        rows.append(
            f"| `{role}` | {', '.join(f'`{name}`' for name in names)} "
            f"| `{era['blob_lf_sha256']}` | `{era['blob_crlf_sha256']}` "
            f"| `{era['recorded_at_era']}` "
            f"| {'CRLF == recorded' if era['crlf_reproduces_recorded'] else 'NO'} |"
        )
    return header + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--table", action="store_true", help="print the Markdown table only")
    arguments = parser.parse_args(argv)
    if arguments.json is not None:
        target = arguments.json.resolve()
        if HERE in target.parents or target == HERE:
            parser.error("--json must not point inside examples/material_fields")
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
