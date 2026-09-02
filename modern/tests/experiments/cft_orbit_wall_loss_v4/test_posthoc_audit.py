"""Integrity tests for the v4 posthoc audit overlay (sidecar EOL finding).

Everything asserted here is re-derived from the immutable ``results/`` bundle
or from Git; the overlay must never change a byte under ``results/``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from cft_revival.experiment_runtime.filesystem import AtomicArtifactStore, pin_existing_root
from cft_revival.experiment_runtime.lifecycle import LifecycleError, _inventory, validate_bundle
from cft_revival.orbit_mc import wilson_interval

from experiments.cft_orbit_wall_loss_v4 import audit_sidecar_eol as audit_module
from experiments.cft_orbit_wall_loss_v4.audit_sidecar_eol import (
    CASES,
    EXPECTED_EOL_ONLY_PATHS,
    audit,
    format_table,
)

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT = REPO / "modern/experiments/cft_orbit_wall_loss_v4"
RESULTS = EXPERIMENT / "results"
AUDIT_MD = EXPERIMENT / "POSTHOC_AUDIT.md"
RESULTS_REL = "modern/experiments/cft_orbit_wall_loss_v4/results"

PREREGISTRATION_COMMIT = "757e365f9f667620c7610663574294c3b71e1f51"
RESULT_COMMIT = "6922a3cf97d261735266aa1a5a0c0c9683e021ca"
RESULTS_TREE = "447a5cf79024b85cabbaeb033d719c6d21ab28c0"
MANIFEST_SHA256 = "ef3863b0a3ba0a1d74187b05daf81d5d94d3838a7e33ecf82c485dccd162929f"
ORBIT_MC_SOURCE_SHA256 = "007c2d51a44d74f989dae6938d10538454886f2e4970f9a9867aaeac8346aa43"
ORBIT_MC_CODE_IDENTITY = "ab2acba9dd21709477bee6b61f37de881ac47f233118f9d43f607d99ad39e6b4"
OVERLAY_SUBJECT = "add CFT full-orbit wall-loss v4 posthoc audit"
ALLOWED_OVERLAY = {
    "modern/experiments/cft_orbit_wall_loss_v4/POSTHOC_AUDIT.md",
    "modern/experiments/cft_orbit_wall_loss_v4/audit_sidecar_eol.py",
    "modern/tests/experiments/cft_orbit_wall_loss_v4/test_posthoc_audit.py",
}
BINDING_GATES = {
    "campaign_preflight", "cross_map_probability_convergence", "earliest_event", "energy",
    "field_adapter", "field_map_convergence", "final_velocity_equals_event_velocity",
    "independent_repeats", "manufactured", "material_quarantine", "relativistic_phase",
    "runtime_rotation", "timestep_probability_convergence", "wall_endpoint",
    "zero_incomplete_or_numerical_failures",
}
EXPECTED_WALL_HITS = {
    "primary-N": 329, "primary-2N": 330, "primary-4N": 330,
    "refined-N": 329, "refined-2N": 330, "refined-4N": 330,
    "enlarged-N": 328, "enlarged-2N": 328, "enlarged-4N": 328,
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _load(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_bytes().decode("utf-8"))


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _utc(record: dict) -> datetime:
    return datetime.fromisoformat(record["value"].replace("Z", "+00:00"))


@pytest.fixture(scope="module")
def report() -> dict:
    before = _tree_digest(RESULTS)
    value = audit()
    assert _tree_digest(RESULTS) == before, "audit must be read-only on results/"
    return value


# --------------------------------------------------------------------------
# the finding
# --------------------------------------------------------------------------


def test_audit_passes_with_exactly_nine_eol_only_sidecars(report: dict) -> None:
    assert report["passed"] is True
    assert report["read_only"] is True
    assert report["manifest_state"] == "accepted_result"
    assert report["manifest_sha256"] == MANIFEST_SHA256
    assert report["manifest_artifact_count"] == 407
    assert report["file_entries"] == 387 and report["directory_entries"] == 20
    assert report["counts"] == {"byte_exact": 378, "eol_only": 9, "mismatch": 0}
    assert report["mismatch"] == []
    assert tuple(row["path"] for row in report["eol_only"]) == EXPECTED_EOL_ONLY_PATHS
    assert report["eol_only_paths_are_exactly_expected"] is True
    assert report["runtime_sidecars_agree_with_manifest"] is True
    for row in report["eol_only"]:
        data = (RESULTS / row["path"]).read_bytes()
        assert b"\r" not in data and data.endswith(b"\n") and data.count(b"\n") == 1
        assert row["recorded_bytes"] == row["checkout_bytes"] + 1
        assert row["checkout_sha256"] == hashlib.sha256(data).hexdigest()
        assert row["crlf_recomputed_sha256"] == hashlib.sha256(
            data.replace(b"\n", b"\r\n")
        ).hexdigest()
        assert row["crlf_recomputed_sha256"] == row["recorded_sha256"]
        assert row["checkout_sha256"] != row["recorded_sha256"]
        # one line: <64 hex>  <case>-orbit.json
        digest, name = data.decode("ascii").split()
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert name.endswith("-orbit.json")


def test_orbit_artifacts_behind_the_sidecars_are_untouched(report: dict) -> None:
    assert report["orbit_evidence_intact"] is True
    assert [item["artifact_path"] for item in report["orbit_artifacts"]] == [
        f"artifacts/orbits/{case}.json.gz" for case in CASES
    ]
    for item in report["orbit_artifacts"]:
        assert item["content_matches_sidecar"] is True
        assert item["decompressed_is_canonical"] is True
        assert item["integrity_recomputes"] is True
        assert item["decompressed_sha256"] == item["sidecar_states_sha256"]
        assert item["schema_version"] == "cft-revival-orbit-mc-result/1.6.0"
        assert item["code_sha256"] == ORBIT_MC_CODE_IDENTITY
        assert item["trial_count"] == 512
        assert item["terminations"].get("reflected", 0) == 0
        assert set(item["terminations"]) == {"wall_hit", "domain_escape"}
    contract = _load("artifacts/orbit-mc-contract.json")
    assert contract["matches"] is True
    assert contract["observed"]["package_version"] == "1.6.0"
    assert contract["source_sha256"] == ORBIT_MC_SOURCE_SHA256
    assert contract["code_identity_sha256"] == ORBIT_MC_CODE_IDENTITY


def test_manifest_inventory_reproduces_after_crlf_restoration_in_scratch_copy(
    tmp_path: Path,
) -> None:
    """The ONLY difference is EOL: restore CRLF on the nine and the inventory matches."""

    scratch = tmp_path / "results"
    shutil.copytree(RESULTS, scratch)
    manifest = json.loads((scratch / "manifest.json").read_bytes())
    with pytest.raises(LifecycleError, match="artifact sidecar schema mismatch"):
        validate_bundle(scratch)
    for relative in EXPECTED_EOL_ONLY_PATHS:
        path = scratch / relative
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    safe = pin_existing_root(scratch)
    try:
        inventory = _inventory(
            AtomicArtifactStore(safe), {}, set(manifest["required_directories"])
        )
    finally:
        safe.close()
    assert inventory == manifest["artifacts"]
    # With the inventory reproduced, validation advances to the root identity,
    # which is bound to the producing directory by design.
    with pytest.raises(LifecycleError, match="root identity"):
        validate_bundle(scratch)


def test_audit_document_table_matches_live_recomputation(report: dict) -> None:
    text = AUDIT_MD.read_text(encoding="utf-8")
    assert "\r" not in text
    table = format_table(report)
    for line in table.splitlines():
        assert line in text, line
    documented_rows = [
        line for line in text.splitlines() if line.startswith("| `artifacts/orbits/")
    ]
    assert len(documented_rows) == 9
    for row in report["eol_only"]:
        assert row["checkout_sha256"] in text
        assert row["recorded_sha256"] in text
    for needle in (
        "collisionless_prescribed_field_test_particle_wall_loss_not_pic",
        "ACCEPTED",
        PREREGISTRATION_COMMIT,
        RESULT_COMMIT,
        RESULTS_TREE,
        MANIFEST_SHA256,
        ORBIT_MC_SOURCE_SHA256,
        ORBIT_MC_CODE_IDENTITY,
        "interpretation, not evidence",
        "Duplicated `execute` invocation",
        "PID 484",
        "Results/",
        "non-evidentiary",
        "audit_sidecar_eol",
    ):
        assert needle in text, needle


# --------------------------------------------------------------------------
# run-time facts re-derived from the bundle
# --------------------------------------------------------------------------


def test_single_attempt_lock_and_terminal_timing() -> None:
    lock = _load("execution-lock.json")
    assert lock["attempt"] == 1 and lock["immutable"] is True
    assert lock["commit"] == PREREGISTRATION_COMMIT
    assert lock["clean_worktree_attested"] is True
    assert lock["command"] == "python -m experiments.cft_orbit_wall_loss_v4.run execute"
    assert lock["acquired_at_utc"]["value"] == "2026-09-02T15:03:40.072889Z"
    terminal = _load("terminal.json")
    assert terminal["state"] == "accepted_result"
    assert terminal["counts"]["attempt_count"] == 1
    assert terminal["counts"]["label_access_count"] == 9
    assert terminal["primary_error"] is None and terminal["secondary_errors"] == []
    transitions = [
        _load(f"transitions/{name}")
        for name in sorted(
            path.name
            for path in (RESULTS / "transitions").iterdir()
            if path.name.endswith(".json") and not path.name.endswith(".sha256.json")
        )
    ]
    assert [item["sequence"] for item in transitions] == list(range(1, 10))
    assert transitions[0]["transition"] == "lock-acquired"
    assert transitions[-1]["transition"] == "terminal"
    assert transitions[-1]["details"] == {"state": "accepted_result"}
    assert transitions[-1]["recorded_at_utc"]["value"] == "2026-09-02T15:14:47.662367Z"
    elapsed = (
        _utc(transitions[-1]["recorded_at_utc"]) - _utc(lock["acquired_at_utc"])
    ).total_seconds()
    assert 667.0 <= elapsed <= 668.0
    manifest = json.loads((RESULTS / "manifest.json").read_bytes())
    lock_bytes = (RESULTS / "execution-lock.json").read_bytes()
    assert manifest["lock_byte_sha256"] == hashlib.sha256(lock_bytes).hexdigest()
    assert manifest["terminal_byte_sha256"] == hashlib.sha256(
        (RESULTS / "terminal.json").read_bytes()
    ).hexdigest()


def test_all_binding_gates_validators_and_counts() -> None:
    gates = _load("artifacts/gates.json")
    assert gates["binding"] is True and gates["passed"] is True
    assert set(gates["checks"]) == BINDING_GATES and len(BINDING_GATES) == 15
    assert all(value is True for value in gates["checks"].values())
    assert gates["exact_authority_replay"] is True and gates["exact_authority_replay_count"] == 9
    assert gates["energy_gate_limit"] == 1e-10
    assert gates["maximum_relative_energy_error"] == 0.0
    assert gates["orbits_exceeding_energy_gate"] == 0
    assert gates["final_velocity_event_velocity_mismatches"] == 0
    assert gates["maximum_wall_endpoint_error_m"] < 1e-18
    assert set(gates["incomplete_and_failure_counts"].values()) == {0}
    mu = gates["diagnostics_not_gates"]["magnetic_moment_variation"]
    assert mu["binding"] is False and mu["role"] == "diagnostic_only"
    assert mu["orbit_count_with_mu"] == 4608 and mu["orbit_count_without_mu"] == 0
    assert mu["min"] == pytest.approx(0.026098, abs=1e-6)
    assert mu["median"] == pytest.approx(0.140562, abs=1e-6)
    assert mu["max"] == pytest.approx(0.628631, abs=1e-6)
    assert mu["count_above_0p1"] == 2786 and mu["count_above_0p5"] == 209
    assert mu["count_above_0p1"] / 4608 == pytest.approx(0.605, abs=0.001)
    result = _load("artifacts/campaign-result.json")
    assert result["status"] == "accepted" and result["evidentiary"] is True
    assert result["classification"] == (
        "collisionless_prescribed_field_test_particle_wall_loss_not_pic"
    )
    assert result["validators"] == {"passed": 289, "failed": 0}
    assert result["orbit_count"] == 4608 and result["campaign_count"] == 9
    pooled = {"wall_hit": 0, "domain_escape": 0, "reflected": 0}
    for case, campaign in result["campaigns"].items():
        counts = campaign["termination_counts"]
        assert campaign["trial_count"] == 512
        assert counts["wall_hit"] == EXPECTED_WALL_HITS[case]
        assert counts["reflected"] == 0
        assert sum(counts.values()) == 512
        assert counts["wall_hit"] + counts["domain_escape"] == 512
        for key in pooled:
            pooled[key] += counts[key]
        expected = wilson_interval(counts["wall_hit"], 512)
        assert campaign["wall_hit"]["probability"] == expected.probability
        assert campaign["wall_hit"]["lower"] == expected.lower
        assert campaign["wall_hit"]["upper"] == expected.upper
        assert 0.598 < campaign["wall_hit"]["lower"] < 0.603
        assert 0.680 < campaign["wall_hit"]["upper"] < 0.686
        assert campaign["incomplete"]["successes"] == 0
    assert pooled == {"wall_hit": 2962, "domain_escape": 1646, "reflected": 0}
    assert pooled["wall_hit"] / 4608 == pytest.approx(0.6428, abs=5e-4)
    mode = result["execution_mode"]
    assert mode["parallel_cases"] is True and mode["worker_pool_size"] == 9
    assert 660.0 < mode["assessment_wall_s"] < 661.0


def test_probability_convergence_changes_are_below_gate() -> None:
    convergence = _load("artifacts/probability-convergence.json")
    assert convergence["timestep_passed"] is True and convergence["cross_map_passed"] is True
    changes = [
        change
        for block in (*convergence["timestep"], *convergence["cross_map"])
        for change in block["successive_changes"]
    ]
    assert max(changes) == 0.00390625 == 2 / 512  # reported as "<= 0.0039" (rounded)
    assert round(max(changes), 4) == 0.0039
    assert max(changes) < 0.01
    assert set(changes) <= {0.0, 1 / 512, 2 / 512}
    assert all(
        all(block["adjacent_wilson_overlap"])
        for block in (*convergence["timestep"], *convergence["cross_map"])
    )


# --------------------------------------------------------------------------
# interpretation support (derived, labelled as interpretation in the document)
# --------------------------------------------------------------------------


def test_launch_cell_breakdown_supports_the_interpretation(report: dict) -> None:
    cells = report["pooled_terminations_by_launch_cell_and_direction"]
    assert set(cells) == {
        f"z={z}mm D{direction}"
        for z in ("3.5", "9.5", "15.5", "21.5")
        for direction in ("+1", "-1")
    }
    assert cells["z=9.5mm D+1"] == cells["z=9.5mm D-1"] == {"wall_hit": 576}
    assert cells["z=15.5mm D+1"] == cells["z=15.5mm D-1"] == {"wall_hit": 576}
    assert cells["z=21.5mm D+1"] == cells["z=21.5mm D-1"] == {"domain_escape": 576}
    assert cells["z=3.5mm D-1"] == {"wall_hit": 576}
    assert cells["z=3.5mm D+1"] == {"domain_escape": 494, "wall_hit": 82}
    assert 0.0215 > report["wall_z_max_m"] == 0.018
    assert report["pooled_terminations"] == {"domain_escape": 1646, "wall_hit": 2962}
    resolutions = report["pooled_event_resolutions"]
    assert resolutions == {"interpolated": 2641, "tolerance_close_fraction_zero": 1967}
    assert resolutions["tolerance_close_fraction_zero"] / 4608 == pytest.approx(0.427, abs=1e-3)


# --------------------------------------------------------------------------
# immutability of the evidence and the overlay allowlist
# --------------------------------------------------------------------------


def test_results_tree_is_unchanged_since_the_result_commit() -> None:
    assert _git("rev-parse", f"{RESULT_COMMIT}:{RESULTS_REL}") == RESULTS_TREE
    assert _git("rev-parse", f"HEAD:{RESULTS_REL}") == RESULTS_TREE
    assert _git("rev-parse", f"HEAD:{RESULTS_REL}/manifest.json") == _git(
        "rev-parse", f"{RESULT_COMMIT}:{RESULTS_REL}/manifest.json"
    )
    assert _git("status", "--porcelain", "--", RESULTS_REL) == ""
    # Working tree is LF for the bundle: no CRLF smudge anywhere under results/.
    eol = _git("ls-files", "--eol", "--", RESULTS_REL)
    assert "w/crlf" not in eol and "w/mixed" not in eol
    # The bundle is ignore-masked (case-insensitive Results/) and was force-added.
    ignored = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", f"{RESULTS_REL}/manifest.json"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
    )
    assert ignored.returncode == 0 and "Results/" in ignored.stdout
    assert _git("ls-files", "--", f"{RESULTS_REL}/manifest.json") == f"{RESULTS_REL}/manifest.json"


def test_frozen_inputs_are_the_preregistered_blobs() -> None:
    for name, blob in (
        ("protocol.json", "a9ed08f3358aadf8a56184410db42a4d43c8f48d"),
        ("authorities.json", "e954c371218b91485fc2c6a9e72d6a63ec8f68ff"),
        ("shakedown.json", "7bac8ffebbf66c24f4391b241a7f81db457e686e"),
    ):
        relative = f"modern/experiments/cft_orbit_wall_loss_v4/{name}"
        assert _git("rev-parse", f"{PREREGISTRATION_COMMIT}:{relative}") == blob
        assert _git("rev-parse", f"HEAD:{relative}") == blob
    shakedown = json.loads((EXPERIMENT / "shakedown.json").read_bytes())
    assert shakedown["evidentiary"] is False
    assert shakedown["outcomes_enter_estimand"] is False
    assert "NON-EVIDENTIARY" in shakedown["disclosure"]
    assert "cft-orbit-wall-loss-v4-shakedown" in shakedown["runtime"]["result_root"]
    assert shakedown["orbit_mc_source_sha256"] == ORBIT_MC_SOURCE_SHA256


def test_committed_overlay_is_exact_allowlist() -> None:
    if _git("show", "-s", "--format=%s", "HEAD") != OVERLAY_SUBJECT:
        pytest.skip("HEAD is not the posthoc audit overlay commit")
    changed = set(_git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines())
    assert changed == ALLOWED_OVERLAY


def test_script_refuses_to_write_inside_results(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        audit_module.main(["--json", str(RESULTS / "posthoc.json")])
    assert "must not point inside results/" in capsys.readouterr().err
    target = tmp_path / "report.json"
    assert audit_module.main(["--json", str(target)]) == 0
    written = json.loads(target.read_bytes())
    assert written["passed"] is True and written["counts"]["eol_only"] == 9
    assert not (RESULTS / "posthoc.json").exists()
