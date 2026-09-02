"""Refresh metadata only after strict replay under the current implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cft_revival.material_fields.acceptance import (
    RawRunObservation,
    _evidence_implementation_sha256,
    _sha,
    _solver_config_identity,
    _study_identity_hashes,
    assess_publication,
)
from cft_revival.material_fields.artifacts import (
    SCHEMA_VERSION,
    validate_artifact,
    viewer_contract,
)
from cft_revival.material_fields.numerics import _implementation_sha256


LIMITATIONS = [
    "Linear recoil PM and linear pole/yoke permeability only.",
    "Face transmissibilities use exact series resistance with linear-edge clipping; interface peaks remain screening-only.",
    "Dipole-Robin truncation remains acceptable only when both recorded nested-domain gates pass.",
    "No hysteresis, validated saturation, irreversible demagnetization, plasma response, or calibration.",
    "Hypothetical design simulation; not a validated hardware prediction.",
]


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def run_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal(value: dict[str, object]) -> None:
    payload = {key: item for key, item in value.items() if key != "integrity"}
    value["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "payload_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
    }


def write(path: Path, value: dict[str, object]) -> str:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    # newline="\n": the sidecar bytes must not depend on the producing platform
    # (Git stores LF repo-wide; a CRLF sidecar would be rewritten on commit).
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def main() -> None:
    root = Path(__file__).resolve().parent / "artifacts"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["artifact"]: entry for entry in manifest["designs"]}
    for artifact_path in sorted(root.glob("*.material-field.json")):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("schema_version") != SCHEMA_VERSION:
            continue
        artifact["limitations"] = LIMITATIONS
        evidence_hash = _evidence_implementation_sha256()
        for run in artifact["acceptance"]["raw_runs"]:
            problem = run["raw"]["problem"]
            diagnostics = run["raw"]["diagnostics"]
            problem["open_boundary_policy"] = dict(
                artifact["domain"]["open_boundary_policy"]
            )
            design_sha, registry_sha = _study_identity_hashes(problem)
            implementation_hash = _implementation_sha256(
                *(
                    ("adapters.py", "models.py", "numerics.py", "warp_solver.py")
                    if str(run["backend"]).startswith("material_fields:warp:")
                    else ("adapters.py", "models.py", "numerics.py")
                )
            )
            run["problem_sha256"] = _sha(problem)
            run["design_geometry_sha256"] = design_sha
            run["material_registry_sha256"] = registry_sha
            run["solver_config_identity_sha256"] = _solver_config_identity(
                diagnostics["run_config_json"]
            )
            run["implementation_sha256"] = implementation_hash
            diagnostics["implementation_sha256"] = implementation_hash
            run["evidence_implementation_sha256"] = evidence_hash
            run_anchors = {
                key: value
                for key, value in run.items()
                if key != "run_sha256"
            }
            run["run_sha256"] = _sha(run_anchors)
        runs = {
            run["study_id"]: RawRunObservation(**run)
            for run in artifact["acceptance"]["raw_runs"]
        }
        evidence = assess_publication(
            runs["base"],
            domain_expansions=(runs["domain-1"], runs["domain-2"]),
            mesh_fine=runs["mesh-fine"],
            mesh_third=runs["mesh-third"],
            alignment_sweeps=(runs["alignment-1"],),
            equivalent_base=runs["equivalent-base"],
            equivalent_fine=runs["equivalent-fine"],
            parity_cpu=runs["parity-cpu"],
            parity_cuda=runs["parity-cuda"],
            qualification=artifact["acceptance"]["qualification"],
        )
        artifact["acceptance"] = evidence.to_dict()
        base_run = next(
            run
            for run in artifact["acceptance"]["raw_runs"]
            if run["role"] == "base"
        )
        parity_cpu = next(
            run
            for run in artifact["acceptance"]["raw_runs"]
            if run["role"] == "backend_parity_cpu"
        )
        parity_cuda = next(
            run
            for run in artifact["acceptance"]["raw_runs"]
            if run["role"] == "backend_parity_cuda"
        )
        artifact["acceptance"]["backend_parity"]["cpu_run_sha256"] = parity_cpu[
            "run_sha256"
        ]
        artifact["acceptance"]["backend_parity"]["cuda_run_sha256"] = parity_cuda[
            "run_sha256"
        ]
        artifact["anchors"]["base_run_sha256"] = base_run["run_sha256"]
        artifact["anchors"]["problem_sha256"] = base_run["problem_sha256"]
        artifact["anchors"]["design_geometry_sha256"] = base_run[
            "design_geometry_sha256"
        ]
        artifact["anchors"]["material_registry_sha256"] = base_run[
            "material_registry_sha256"
        ]
        artifact["anchors"]["solver_config_identity_sha256"] = base_run[
            "solver_config_identity_sha256"
        ]
        artifact["anchors"]["implementation_sha256"] = base_run[
            "implementation_sha256"
        ]
        artifact["anchors"]["evidence_implementation_sha256"] = evidence_hash
        artifact["diagnostics"]["implementation_sha256"] = base_run[
            "implementation_sha256"
        ]
        artifact["summary"]["warning_codes"] = list(evidence.warning_codes)
        mesh_status = next(
            gate["status"]
            for gate in artifact["acceptance"]["gates"]
            if gate["gate_id"] == "mesh_fixed_qoi"
        )
        for peak_key in ("sampled_cell_peak", "axis_bz_peak"):
            peak = artifact["summary"][peak_key]
            peak.pop("mesh_gate_passed", None)
            peak["mesh_gate_status"] = mesh_status
        seal(artifact)
        # This reconstructs coefficients and sources and rejects migrations that
        # would change any bound numerical result under the current code.
        validate_artifact(artifact, require_accepted=False)
        file_digest = write(artifact_path, artifact)
        entry = entries[artifact_path.name]
        entry["artifact_file_sha256"] = file_digest
        entry["artifact_payload_sha256"] = artifact["integrity"]["payload_sha256"]
        viewer_path = artifact_path.with_name(
            artifact_path.name.replace(".material-field.json", ".viewer.json")
        )
        viewer = viewer_contract(artifact, validate_source=False)
        write(viewer_path, viewer)
    manifest["schema_version"] = "cft_revival.material_fields.design_manifest/1.4.0"
    seal(manifest)
    write(manifest_path, manifest)


if __name__ == "__main__":
    main()
