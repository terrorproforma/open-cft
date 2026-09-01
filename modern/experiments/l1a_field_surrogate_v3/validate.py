"""Strict terminal validation for v3 success and failure bundles."""

from __future__ import annotations

from typing import Any, Mapping

from .protocol import PROTOCOL_HASH, RESULTS, canonical_hash, verify_json

COUNTERS = {
    "low_solver_accesses",
    "low_completed",
    "fine_solver_accesses",
    "fine_completed",
    "model_fit_accesses",
    "method_label_accesses",
    "calibration_label_accesses",
    "assessment_label_accesses",
}


def _validate_counters(value: Mapping[str, Any]) -> None:
    counters = value.get("stage_access_counters")
    if not isinstance(counters, Mapping) or set(counters) != COUNTERS:
        raise ValueError("stage access counters are incomplete")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counters.values()):
        raise ValueError("stage access counters are invalid")
    if counters["low_solver_accesses"] not in {counters["low_completed"], counters["low_completed"] + 1}:
        raise ValueError("low solve counters are inconsistent")
    if counters["fine_solver_accesses"] not in {counters["fine_completed"], counters["fine_completed"] + 1}:
        raise ValueError("fine solve counters are inconsistent")


def validate_bundle() -> dict[str, Any]:
    lock = verify_json(RESULTS / "execution-lock.json")
    provenance = verify_json(RESULTS / "provenance-closure.json")
    if lock["protocol_hash"] != PROTOCOL_HASH or provenance["protocol_hash"] != PROTOCOL_HASH:
        raise ValueError("terminal protocol identity mismatch")
    if (
        provenance["solver_accesses_at_write"] != 0
        or provenance["git_blob_closure_hash"] != canonical_hash(provenance["git_blob_closure"])
        or not provenance["git_blob_closure"]
    ):
        raise ValueError("pre-solve provenance closure is invalid")
    failure_path = RESULTS / "failure-manifest.json"
    if failure_path.exists():
        failure = verify_json(failure_path)
        _validate_counters(failure)
        payload = {key: value for key, value in failure.items() if key != "failure_hash"}
        if (
            failure["failure_hash"] != canonical_hash(payload)
            or failure["provenance_hash"] != provenance["provenance_hash"]
            or failure["rerun_performed"]
            or not failure["exclusive_lock_retained"]
        ):
            raise ValueError("failure bundle is inconsistent")
        return {
            "passed": True,
            "terminal_kind": "failure",
            "status": failure["status"],
            "stage_access_counters": failure["stage_access_counters"],
            "closure_entries": len(provenance["git_blob_closure"]),
        }
    terminal = verify_json(RESULTS / "terminal-result.json")
    _validate_counters(terminal)
    payload = {key: value for key, value in terminal.items() if key != "terminal_hash"}
    if terminal["terminal_hash"] != canonical_hash(payload):
        raise ValueError("terminal hash mismatch")
    counters = terminal["stage_access_counters"]
    if terminal["status"] == "failed-development-selection-gates":
        if counters["calibration_label_accesses"] or counters["assessment_label_accesses"]:
            raise ValueError("development failure accessed future roles")
    else:
        manifest = verify_json(RESULTS / "run-manifest.json")
        manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if (
            manifest["manifest_hash"] != canonical_hash(manifest_payload)
            or manifest["terminal_hash"] != terminal["terminal_hash"]
            or manifest["provenance_hash"] != provenance["provenance_hash"]
            or manifest["rerun_performed"]
            or not manifest["exclusive_lock_retained"]
            or counters["method_label_accesses"] != 1
            or counters["calibration_label_accesses"] != 1
            or counters["assessment_label_accesses"] != 1
        ):
            raise ValueError("assessment bundle is inconsistent")
        verify_json(RESULTS / "numerical-records.json")
        verify_json(RESULTS / "selected-model.json")
        verify_json(RESULTS / "group-conformal-calibration.json")
        verify_json(RESULTS / "frozen-before-assessment.json")
    return {
        "passed": True,
        "terminal_kind": "terminal-result",
        "status": terminal["status"],
        "stage_access_counters": counters,
        "closure_entries": len(provenance["git_blob_closure"]),
    }
