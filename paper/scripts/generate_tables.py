"""Generate hash-bound manuscript tables from accepted evidence manifests."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path("paper/evidence/l0-run-manifest.json")
OUTPUT_PATH = Path("paper/generated/l0-ranges.tex")
SIDECAR_PATH = Path("paper/generated/l0-ranges.provenance.json")

RANGE_ROWS = (
    ("axial_thrust_n", "thrust", "Axial thrust", "N"),
    ("specific_impulse_s", "isp", "Specific impulse", "s"),
    ("beam_current_a", "beamCurrent", "Beam current", "A"),
    ("anode_input_w", "anodePower", "Anode input", "W"),
    ("beam_kinetic_power_w", "beamPower", "Beam kinetic power", "W"),
    (
        "ppu_input_to_beam_efficiency",
        "ppuEfficiency",
        "PPU-input-to-beam efficiency",
        "1",
    ),
)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_bytes(repo: Path, revision: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise ValueError(
            f"cannot resolve committed input {revision}:{path}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def _embedded_payload(html: bytes) -> dict[str, Any]:
    text = html.decode("utf-8")
    match = re.search(
        r'<script id="l0-data" type="application/json">(.*?)</script>',
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("accepted HTML lacks the l0-data JSON script")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("accepted HTML payload must be an object")
    return payload


def render(repo: Path) -> tuple[bytes, dict[str, Any]]:
    manifest_file = repo / MANIFEST_PATH
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("document_type") != "paper-L0-run-evidence-manifest":
        raise ValueError("L0 manifest has the wrong document_type")
    if manifest.get("schema_version") != "1.0":
        raise ValueError("L0 manifest has an unsupported schema_version")
    revision = manifest.get("evidence_revision")
    if not isinstance(revision, str):
        raise ValueError("L0 manifest lacks evidence_revision")

    accepted = [
        source
        for source in manifest.get("source_files", [])
        if source.get("role") == "accepted-html"
    ]
    if len(accepted) != 1:
        raise ValueError("L0 manifest must bind exactly one accepted HTML artifact")
    html_source = accepted[0]
    html_bytes = git_bytes(repo, revision, html_source["path"])
    if sha256_bytes(html_bytes) != html_source.get("git_blob_sha256"):
        raise ValueError("accepted HTML SHA-256 does not match the manifest")
    payload = _embedded_payload(html_bytes)
    html_contract = manifest["accepted_html"]
    if payload.get("documentType") != html_contract["embedded_document_type"]:
        raise ValueError("accepted HTML document type does not match the manifest")
    if payload.get("schemaVersion") != html_contract["embedded_schema_version"]:
        raise ValueError("accepted HTML schema version does not match the manifest")

    raw_output = html_contract["raw_per_point_output"]
    columns = payload.get("columns")
    if not isinstance(columns, dict):
        raise ValueError("accepted HTML columns must be an object")
    if payload.get("sampleCount") != raw_output["sample_count"]:
        raise ValueError("accepted HTML sample count does not match the manifest")
    if len(columns) != raw_output["column_count"]:
        raise ValueError("accepted HTML column count does not match the manifest")
    if {len(values) for values in columns.values()} != {
        raw_output["all_column_lengths"]
    }:
        raise ValueError("accepted HTML per-point columns have inconsistent lengths")
    dataset_identity = (
        payload.get("operatingConceptGallery", {})
        .get("source", {})
        .get("dataset_identity", {})
        .get("sha256")
    )
    if dataset_identity != raw_output["dataset_sha256"]:
        raise ValueError("accepted HTML dataset identity does not match the manifest")

    rows: list[str] = []
    ranges = manifest["metrics"]["raw_ranges"]
    html_ranges = payload["ranges"]
    for manifest_key, html_key, label, unit in RANGE_ROWS:
        registered = ranges[manifest_key]
        embedded = html_ranges[html_key]
        if registered["minimum"] != embedded["minimum"]:
            raise ValueError(f"{manifest_key} minimum differs from accepted HTML")
        if registered["maximum"] != embedded["maximum"]:
            raise ValueError(f"{manifest_key} maximum differs from accepted HTML")
        rows.append(
            f"{label} ({unit}) & {registered['display_minimum']} & "
            f"{registered['display_maximum']}\\\\"
        )

    table = (
        "% Generated by paper/scripts/generate_tables.py; do not hand edit.\n"
        "\\ArtifactClaim{CLM-006}{TAB-L0-RANGES}{%\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Ranges in the accepted 8,192-point L0 evidence artifact. "
        "Inputs are hypothetical; values are not validated design predictions.}\n"
        "\\label{tab:l0-ranges}\n"
        "\\begin{tabular}{lll}\n"
        "\\toprule\n"
        "Quantity & Minimum & Maximum\\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}%\n"
        "}\n"
    ).encode("utf-8")

    generator_path = Path(__file__).resolve()
    build_config = json.loads((repo / "paper/build-config.json").read_text("utf-8"))
    sidecar = {
        "document_type": "paper-generated-artifact-provenance",
        "schema_version": "1.0",
        "artifact_id": "TAB-L0-RANGES",
        "claim_ids": ["CLM-006"],
        "evidence_revision": revision,
        "source_date_epoch": build_config["source_date_epoch"],
        "generator": {
            "path": generator_path.relative_to(repo).as_posix(),
            "sha256": sha256_bytes(generator_path.read_bytes()),
            "command": "python paper/scripts/generate_tables.py",
        },
        "manifest": {
            "path": MANIFEST_PATH.as_posix(),
            "sha256": sha256_bytes(manifest_bytes),
            "manifest_id": manifest["manifest_id"],
        },
        "inputs": [
            {
                "path": html_source["path"],
                "git_blob": html_source["git_blob"],
                "git_blob_sha256": html_source["git_blob_sha256"],
                "dataset_sha256": raw_output["dataset_sha256"],
            }
        ],
        "output": {
            "path": OUTPUT_PATH.as_posix(),
            "sha256": sha256_bytes(table),
        },
    }
    return table, sidecar


def write_generated(repo: Path) -> None:
    table, sidecar = render(repo)
    output = repo / OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(table)
    (repo / SIDECAR_PATH).write_bytes(canonical_json(sidecar))


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    try:
        write_generated(repo)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Table generation failed: {exc}")
        return 1
    print(f"Generated {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
