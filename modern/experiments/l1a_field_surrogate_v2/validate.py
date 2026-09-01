"""Strict terminal loader for both successful and failed v2 bundles."""

from __future__ import annotations

from typing import Any, Mapping

from .protocol import PROTOCOL_HASH, RESULTS, canonical_hash, verify_json

COUNTER_KEYS = {
    "coarse_completed",
    "fine_completed",
    "solver_accesses",
    "model_fits",
    "method_label_accesses",
    "calibration_accesses",
    "assessment_accesses",
}


def _counters(value: Mapping[str, Any]) -> None:
    counters = value.get("counters")
    if not isinstance(counters, Mapping) or set(counters) != COUNTER_KEYS:
        raise ValueError("terminal counters are incomplete")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counters.values()):
        raise ValueError("terminal counters must be nonnegative integers")
    completed = counters["coarse_completed"] + counters["fine_completed"]
    if counters["solver_accesses"] not in {completed, completed + 1}:
        raise ValueError("solver access counter does not bound completed solves")


def validate_bundle() -> dict[str, Any]:
    lock = verify_json(RESULTS / "execution-lock.json")
    provenance = verify_json(RESULTS / "provenance-closure.json")
    if lock["protocol_hash"] != PROTOCOL_HASH or provenance["protocol_hash"] != PROTOCOL_HASH:
        raise ValueError("bundle protocol identity mismatch")
    if (
        provenance["field_solver_access_count_at_write"] != 0
        or provenance["qoi_label_access_count_at_write"] != 0
        or canonical_hash(provenance["git_blob_closure"]) != provenance["git_blob_closure_hash"]
        or not provenance["git_blob_closure"]
    ):
        raise ValueError("pre-solve provenance closure is invalid")
    for item in provenance["git_blob_closure"]:
        if set(item) != {"path", "mode", "type", "blob"} or item["type"] != "blob":
            raise ValueError("malformed Git blob closure entry")
    failure_path = RESULTS / "failure-manifest.json"
    if failure_path.exists():
        failure = verify_json(failure_path)
        _counters(failure)
        payload = {key: value for key, value in failure.items() if key != "failure_hash"}
        if (
            failure["failure_hash"] != canonical_hash(payload)
            or failure["provenance_hash"] != provenance["provenance_hash"]
            or failure["status"] != "failed-execution"
            or not failure["valid_prospective_result"]
            or failure["rerun_performed"]
            or not failure["exclusive_lock_retained"]
        ):
            raise ValueError("failure bundle is inconsistent")
        return {
            "passed": True,
            "terminal_kind": "failure",
            "status": failure["status"],
            "counters": failure["counters"],
            "closure_entries": len(provenance["git_blob_closure"]),
        }
    terminal = verify_json(RESULTS / "terminal-result.json")
    _counters(terminal)
    terminal_payload = {key: value for key, value in terminal.items() if key != "terminal_hash"}
    if terminal["terminal_hash"] != canonical_hash(terminal_payload):
        raise ValueError("terminal result hash mismatch")
    if terminal["status"] == "failed-development-selection-gates":
        if terminal["assessment_labels_accessed"] or terminal["counters"]["assessment_accesses"]:
            raise ValueError("development failure accessed assessment labels")
    else:
        manifest = verify_json(RESULTS / "run-manifest.json")
        manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if (
            manifest["manifest_hash"] != canonical_hash(manifest_payload)
            or manifest["terminal_hash"] != terminal["terminal_hash"]
            or manifest["provenance_hash"] != provenance["provenance_hash"]
            or manifest["rerun_performed"]
            or not manifest["exclusive_lock_retained"]
            or terminal["counters"]["assessment_accesses"] != 1
        ):
            raise ValueError("success/assessment manifest is inconsistent")
        for item in terminal["representatives"]:
            artifact = RESULTS / item["path"]
            verify_json(artifact)
    return {
        "passed": True,
        "terminal_kind": "terminal-result",
        "status": terminal["status"],
        "counters": terminal["counters"],
        "closure_entries": len(provenance["git_blob_closure"]),
    }
