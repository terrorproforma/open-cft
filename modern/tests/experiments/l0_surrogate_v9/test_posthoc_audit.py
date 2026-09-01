"""Integrity and semantic-status tests for the v9 posthoc audit overlay."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from cft_revival.surrogates.identity import canonical_hash

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT = REPO / "modern/experiments/l0_surrogate_v9"
STATUS = EXPERIMENT / "posthoc-audit-status.json"
ALLOWED_OVERLAY = {
    "modern/experiments/l0_surrogate_v9/POSTHOC_AUDIT.md",
    "modern/experiments/l0_surrogate_v9/posthoc-audit-status.json",
    "modern/tests/experiments/l0_surrogate_v9/test_posthoc_audit.py",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def test_audit_semantics_and_hashes() -> None:
    value = json.loads(STATUS.read_text(encoding="utf-8"))
    assert value["semantic_status"] == "NUMERICAL_PASS_SCIENTIFIC_SURROGATE_FAIL"
    assert value["reviewer_evidence_summary_hash"] == canonical_hash(
        value["reviewer_evidence_summary"]
    )
    assert value["posthoc_audit_hash"] == canonical_hash(
        {key: item for key, item in value.items() if key != "posthoc_audit_hash"}
    )
    evidence = value["reviewer_evidence_summary"]
    assert evidence["numerical_and_reproducibility_gates_passed"]
    assert not evidence["scientific_surrogate_and_acceleration_claim_passed"]
    assert evidence["permitted_claim"] == (
        "Algebraic implementation plus negligible GP matched fresh same-domain evaluations."
    )
    assert value["diagnostic_speed_probe"]["speed_claim"] == "none"
    assert not value["raw_artifacts_modified"]
    assert not value["experiment_rerun"]


def test_overlay_cannot_change_raw_result_identity() -> None:
    value = json.loads(STATUS.read_text(encoding="utf-8"))
    bindings = value["bindings"]
    result_path = "modern/experiments/l0_surrogate_v9/results"
    assert _git("rev-parse", f"{bindings['result_commit']}:{result_path}") == bindings["result_tree"]
    assert _git("rev-parse", f"HEAD:{result_path}") == bindings["result_tree"]
    assert _git(
        "rev-parse",
        f"{bindings['preregistration_commit']}:modern/experiments/l0_surrogate_v9/predeclaration.json",
    ) == bindings["preregistration_blob"]
    assert _git("rev-parse", f"HEAD:{result_path}/run-manifest.json") == bindings["run_manifest_blob"]
    assert _git("rev-parse", f"HEAD:{result_path}/final-assessment.json") == bindings["final_assessment_blob"]


def test_preflight_counter_and_role_overlap_correction() -> None:
    preflight = json.loads((EXPERIMENT / "preflight.json").read_text(encoding="utf-8"))
    partitions = json.loads((EXPERIMENT / "partitions.json").read_text(encoding="utf-8"))
    final_indices = {
        int(index)
        for split in partitions["roles"]["final-calibration"].values()
        for index in split["indices"]
    }
    assessment_indices = {
        int(index)
        for split in partitions["roles"]["assessment"].values()
        for index in split["indices"]
    }
    assert preflight["analytic_reference_identity"]["points"] == 36
    assert preflight["physics_label_access_count"] == 0
    assert final_indices.intersection(range(32)) == {24}
    assert not assessment_indices.intersection(range(32))


def test_committed_overlay_is_exact_allowlist() -> None:
    if _git("show", "-s", "--format=%s", "HEAD") != "add L0 surrogate v9 posthoc audit":
        return
    changed = set(
        _git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    )
    assert changed == ALLOWED_OVERLAY
