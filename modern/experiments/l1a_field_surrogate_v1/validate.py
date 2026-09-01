"""Strict replay-independent validation of the immutable result bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .protocol import PROTOCOL, RESULTS, canonical_hash, verify_json


def validate_bundle() -> dict[str, Any]:
    required = (
        "execution-lock.json",
        "frozen-method-selection.json",
        "selected-models.json",
        "group-conformal-calibration.json",
        "frozen-before-assessment.json",
        "final-assessment.json",
        "numerical-records.json",
        "run-manifest.json",
    )
    values = {name: verify_json(RESULTS / name) for name in required}
    frozen = values["frozen-method-selection.json"]
    calibration = values["group-conformal-calibration.json"]
    freeze = values["frozen-before-assessment.json"]
    final = values["final-assessment.json"]
    manifest = values["run-manifest.json"]
    if freeze["frozen_selection_hash"] != frozen["frozen_selection_hash"]:
        raise ValueError("method freeze identity mismatch")
    if freeze["calibration_hash"] != calibration["calibration_hash"]:
        raise ValueError("calibration freeze identity mismatch")
    if final["freeze_hash"] != freeze["freeze_hash"] or final["assessment_access_count"] != 1:
        raise ValueError("single-use assessment contract mismatch")
    if final["claim"] != PROTOCOL["classification"]:
        raise ValueError("claim boundary changed")
    for item in manifest["git_blob_closure"]:
        if len(item["blob"]) != 40 or item["type"] != "blob":
            raise ValueError("invalid Git blob closure entry")
    if canonical_hash(manifest["git_blob_closure"]) != manifest["git_blob_closure_hash"]:
        raise ValueError("Git blob closure hash mismatch")
    representative_hashes = {}
    for item in manifest["representatives"]:
        path = RESULTS / item["path"]
        verify_json(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("representative artifact hash mismatch")
        representative_hashes[item["path"]] = digest
    status_consistent = (manifest["status"] == "accepted") == bool(final["accepted"])
    if not status_consistent or not manifest["valid_prospective_result"] or manifest["rerun_performed"]:
        raise ValueError("terminal manifest status is inconsistent")
    return {
        "passed": True,
        "status": manifest["status"],
        "assessment_access_count": final["assessment_access_count"],
        "closure_entries": len(manifest["git_blob_closure"]),
        "representative_hashes": representative_hashes,
    }
