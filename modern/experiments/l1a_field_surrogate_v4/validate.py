"""Strict success/failure loader with phase and cleanup verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import PROTOCOL_HASH, RESULTS, canonical_hash, verify_json


def _hash(value: Mapping[str, Any], field: str) -> None:
    payload = {key: item for key, item in value.items() if key != field}
    if value[field] != canonical_hash(payload):
        raise ValueError(f"{field} mismatch")


def _counters(value: Mapping[str, Any]) -> None:
    counters = value["access_counters"]
    roles = {"candidate", "method", "calibration", "assessment"}
    if set(counters["solver_accesses"]) != roles or set(counters["materialized"]) != roles:
        raise ValueError("role counters incomplete")
    if set(counters["checkpoint_reads"]) != roles or set(counters["label_reads"]) != roles:
        raise ValueError("read counters incomplete")
    for role in roles:
        for fidelity in ("low", "fine"):
            access = counters["solver_accesses"][role][fidelity]
            materialized = counters["materialized"][role][fidelity]
            if access not in {materialized, materialized + 1}:
                raise ValueError("solver/materialization counters inconsistent")
        if counters["checkpoint_reads"][role] != counters["label_reads"][role]:
            raise ValueError("checkpoint and label reads differ")


def validate_bundle() -> dict[str, Any]:
    lock = verify_json(RESULTS / "execution-lock.json")
    provenance = verify_json(RESULTS / "provenance-closure.json")
    cleanup = verify_json(RESULTS / "cleanup-record.json")
    _hash(cleanup, "cleanup_hash")
    if lock["protocol_hash"] != PROTOCOL_HASH or provenance["protocol_hash"] != PROTOCOL_HASH:
        raise ValueError("protocol identity mismatch")
    if provenance["solver_accesses_at_write"] != 0:
        raise ValueError("provenance was not persisted before solving")
    if provenance["git_blob_closure_hash"] != canonical_hash(provenance["git_blob_closure"]):
        raise ValueError("closure hash mismatch")
    if cleanup["working_cache_exists_after_cleanup"] or (RESULTS / ".working").exists():
        raise ValueError("working label cache was retained")
    _counters(cleanup)
    for name, expected_hash in cleanup["phase_hashes"].items():
        if name in {"method-freeze", "calibration-freeze"}:
            continue
        phase = verify_json(RESULTS / f"phase-{name}.json")
        _hash(phase, "phase_hash")
        if phase["phase_hash"] != expected_hash:
            raise ValueError("phase hash mismatch")
    failure_path = RESULTS / "failure-manifest.json"
    if failure_path.exists():
        failure = verify_json(failure_path)
        _hash(failure, "failure_hash")
        _counters(failure)
        if (
            failure["provenance_hash"] != provenance["provenance_hash"]
            or failure["rerun_performed"]
            or not failure["exclusive_lock_retained"]
        ):
            raise ValueError("failure manifest inconsistent")
        return {
            "passed": True,
            "terminal_kind": "failure",
            "status": failure["status"],
            "access_counters": failure["access_counters"],
            "cleanup": cleanup,
            "closure_entries": len(provenance["git_blob_closure"]),
        }
    terminal = verify_json(RESULTS / "terminal-result.json")
    _hash(terminal, "terminal_hash")
    _counters(terminal)
    if terminal["status"] != "failed-development-selection-gates":
        manifest = verify_json(RESULTS / "run-manifest.json")
        _hash(manifest, "manifest_hash")
        if (
            manifest["terminal_hash"] != terminal["terminal_hash"]
            or manifest["provenance_hash"] != provenance["provenance_hash"]
            or manifest["rerun_performed"]
            or not manifest["exclusive_lock_retained"]
        ):
            raise ValueError("terminal manifest inconsistent")
        verify_json(RESULTS / "selected-model.json")
        verify_json(RESULTS / "group-conformal-calibration.json")
        verify_json(RESULTS / "frozen-before-assessment.json")
        verify_json(RESULTS / "numerical-completion.json")
    return {
        "passed": True,
        "terminal_kind": "terminal",
        "status": terminal["status"],
        "access_counters": terminal["access_counters"],
        "cleanup": cleanup,
        "closure_entries": len(provenance["git_blob_closure"]),
    }
