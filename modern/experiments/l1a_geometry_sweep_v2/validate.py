"""Strict loader for immutable sweep-v2 result bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cft_revival.fields import validate_field_artifact_file
from cft_revival.geometry import deserialize_geometry

from .experiment import CLASSIFICATION, PROTOCOL
from .protocol import strict_json, validate_sealed, verify_sidecar

MANIFEST_KEYS = {
    "schema_version",
    "classification",
    "preregistration_commit_sha",
    "protocol_file_sha256",
    "protocol_payload_sha256",
    "artifact_hash_policy",
    "raw_results_payload_sha256",
    "summary_payload_sha256",
    "representative_roles",
    "representative_artifacts",
    "deterministic_files",
    "terminal_status",
    "integrity",
}
RAW_KEYS = {
    "schema_version",
    "classification",
    "preregistration_commit_sha",
    "protocol_file_sha256",
    "protocol_payload_sha256",
    "sampling_design_ids",
    "cases",
    "parity",
    "runtime_diagnostics",
    "integrity",
}
SUMMARY_KEYS = {
    "schema_version",
    "classification",
    "screening_level",
    "preregistration_commit_sha",
    "protocol_payload_sha256",
    "environment",
    "requested_count",
    "evaluated_count",
    "failed_count",
    "nondominated_count",
    "nondominated_case_ids",
    "representative_roles",
    "unique_representative_count",
    "terminal_gates",
    "terminal_status",
    "qoi_ranges",
    "replay_contract",
    "raw_results_payload_sha256",
    "integrity",
}


def _sealed(path: Path, keys: set[str]) -> dict[str, Any]:
    verify_sidecar(path)
    value = strict_json(path)
    validate_sealed(value, expected_keys=keys)
    return value


def validate_bundle(results: Path) -> dict[str, Any]:
    results = results.resolve()
    manifest = _sealed(results / "manifest.json", MANIFEST_KEYS)
    raw = _sealed(results / "raw-results.json", RAW_KEYS)
    summary = _sealed(results / "summary.json", SUMMARY_KEYS)
    lock = _sealed(
        results / "execution-lock.json",
        {
            "schema_version",
            "state",
            "started_at_utc",
            "preregistration_commit_sha",
            "protocol_file_sha256",
            "protocol_payload_sha256",
            "case_count",
            "device",
            "integrity",
        },
    )
    if (
        manifest["classification"] != CLASSIFICATION
        or raw["classification"] != CLASSIFICATION
        or summary["classification"] != CLASSIFICATION
    ):
        raise ValueError("classification binding mismatch")
    revisions = {
        manifest["preregistration_commit_sha"],
        raw["preregistration_commit_sha"],
        summary["preregistration_commit_sha"],
        lock["preregistration_commit_sha"],
        summary["environment"]["code_revision"],
    }
    if len(revisions) != 1:
        raise ValueError("preregistration revision binding mismatch")
    protocol_hashes = {
        manifest["protocol_payload_sha256"],
        raw["protocol_payload_sha256"],
        summary["protocol_payload_sha256"],
        lock["protocol_payload_sha256"],
        PROTOCOL["integrity"]["payload_sha256"],
    }
    if len(protocol_hashes) != 1:
        raise ValueError("protocol payload binding mismatch")
    if (
        manifest["raw_results_payload_sha256"]
        != raw["integrity"]["payload_sha256"]
        or summary["raw_results_payload_sha256"]
        != raw["integrity"]["payload_sha256"]
        or manifest["summary_payload_sha256"]
        != summary["integrity"]["payload_sha256"]
    ):
        raise ValueError("result payload binding mismatch")
    if len(raw["cases"]) != 96 or len(raw["sampling_design_ids"]) != 96:
        raise ValueError("raw result does not contain exactly 96 cases")
    if len(set(raw["sampling_design_ids"])) != 96:
        raise ValueError("sampling identities are not unique")
    case_ids: set[str] = set()
    successful = 0
    for case in raw["cases"]:
        if not isinstance(case, dict) or case.get("case_id") in case_ids:
            raise ValueError("case records must be unique objects")
        case_ids.add(case["case_id"])
        if case.get("status") == "failure":
            if set(case) != {
                "case_id",
                "status",
                "failure",
                "design_id",
                "sampling_provenance",
                "design_values",
                "classification",
            }:
                raise ValueError("failed case schema is not closed")
            if "qois" in case:
                raise ValueError("failed cases cannot carry fake outcomes")
            continue
        expected = {
            "case_id",
            "status",
            "failure",
            "design_id",
            "sampling_provenance",
            "design_values",
            "derived_geometry",
            "geometry_sha256",
            "source_sha256",
            "config_sha256",
            "case_sha256",
            "backend",
            "iterations",
            "qois",
            "classification",
        }
        if set(case) != expected or case["failure"] is not None:
            raise ValueError("successful case schema is not closed")
        for key in (
            "geometry_sha256",
            "source_sha256",
            "config_sha256",
            "case_sha256",
        ):
            digest = case[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"invalid {key}")
        successful += 1
    if (
        summary["requested_count"] != 96
        or summary["evaluated_count"] != successful
        or summary["failed_count"] != 96 - successful
        or len(summary["terminal_gates"]) != 7
        or len(summary["representative_roles"]) != 5
    ):
        raise ValueError("summary count or policy mismatch")
    if summary["representative_roles"] != manifest["representative_roles"]:
        raise ValueError("representative role mapping mismatch")
    unique_role_ids = {item["case_id"] for item in summary["representative_roles"]}
    artifacts_by_id = {
        item["case_id"]: item for item in manifest["representative_artifacts"]
    }
    if set(artifacts_by_id) != unique_role_ids:
        raise ValueError("unique representative artifacts do not match role coalescence")
    if summary["unique_representative_count"] != len(unique_role_ids):
        raise ValueError("unique representative count mismatch")
    manifest_paths: set[str] = set()
    for entry in manifest["deterministic_files"]:
        if set(entry) != {"path", "kind", "file_sha256", "payload_sha256"}:
            raise ValueError("manifest file entry schema is not closed")
        if entry["path"] in manifest_paths:
            raise ValueError("manifest paths must be unique")
        manifest_paths.add(entry["path"])
        path = (results / entry["path"]).resolve()
        if not path.is_relative_to(results):
            raise ValueError("manifest path escapes result directory")
        if verify_sidecar(path) != entry["file_sha256"]:
            raise ValueError("manifest file hash mismatch")
        if entry["kind"] in {"full_field", "downsampled_field"}:
            validate_field_artifact_file(
                path,
                expected_file_sha256=entry["file_sha256"],
                expected_payload_sha256=entry["payload_sha256"],
            )
        if entry["kind"] == "geometry":
            geometry = deserialize_geometry(path.read_text(encoding="utf-8"))
            if geometry.canonical_sha256 != entry["payload_sha256"]:
                raise ValueError("geometry payload hash mismatch")
    for item in manifest["representative_artifacts"]:
        expected_roles = sorted(
            role["role"]
            for role in manifest["representative_roles"]
            if role["case_id"] == item["case_id"]
        )
        if item["roles"] != expected_roles:
            raise ValueError("artifact roles do not match role mapping")
        for kind in ("geometry", "full_field", "downsampled_field"):
            if item[kind]["path"] not in manifest_paths:
                raise ValueError("representative artifact absent from file manifest")
    if manifest["artifact_hash_policy"] != PROTOCOL["replay_contract"]["artifact_policy"]:
        raise ValueError("artifact replay policy mismatch")
    return {"manifest": manifest, "raw": raw, "summary": summary}
