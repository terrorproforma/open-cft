"""Re-derivation of the MDO L0 campaign v1 post-hoc audit table.

Everything asserted here is recomputed from the immutable ``results/`` bundle,
from Git, or from the audit module's independent re-implementation of the
evaluation chain; the overlay must never change a byte under ``results/``.
The BoTorch replay is executed only when the pinned ML runtime is importable
(the system interpreter has no torch); the audit document records the run
that was performed once with the pinned runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from experiments.mdo_l0_campaign_v1 import audit_replay as audit_module
from experiments.mdo_l0_campaign_v1.audit_replay import (
    CLASSIFICATION,
    INDEPENDENT_RELATIVE_TOLERANCE,
    MANIFEST_SHA256,
    PREREGISTRATION_COMMIT,
    RESULT_COMMIT,
    RESULTS_TREE,
    SOURCE_SHA256,
    Bundle,
    audit,
    format_table,
    package_replay,
    wfg_hypervolume,
    wilson_interval,
)

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT = REPO / "modern/experiments/mdo_l0_campaign_v1"
RESULTS = EXPERIMENT / "results"
AUDIT_MD = EXPERIMENT / "POSTHOC_AUDIT.md"
RESULTS_REL = "modern/experiments/mdo_l0_campaign_v1/results"
OVERLAY_SUBJECT = "add MDO L0 campaign v1 posthoc audit"
ALLOWED_OVERLAY = {
    "modern/experiments/mdo_l0_campaign_v1/POSTHOC_AUDIT.md",
    "modern/experiments/mdo_l0_campaign_v1/audit_replay.py",
    "modern/tests/experiments/mdo_l0_campaign_v1/test_posthoc_audit.py",
}
EXPECTED_DISCLOSURES = {"F9", "F10", "F22", "F26", "F27", "F28"}
# rows whose observed text depends on the machine or on the optional ML stages
ENVIRONMENT_DEPENDENT_ROWS = {"F3", "F24"}
# F8's observed text ends with a statement about the CHECKOUT ("non-scoped deps unchanged=..."):
# whether the imported-but-not-hash-bound packages (F10) have moved since the preregistration.
# It was True when the document was written (e9f9af16) and has been False since bb756418 moved
# cft_revival.experiment_runtime; the document is a frozen record (its blob is pinned by the
# paper), so that clause is compared as a recorded live-tree fact and every other clause of the
# row must still match verbatim. The frozen facts behind F8 are asserted in the hash-chain test.
HEAD_DEPENDENT_CLAUSES = {"F8": re.compile(r", non-scoped deps unchanged=(True|False)")}
HAS_PYMOO = importlib.util.find_spec("pymoo") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("botorch") is not None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def _tree_digest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture(scope="module")
def report() -> dict:
    before = _tree_digest(RESULTS)
    value = audit(nsga3=HAS_PYMOO)
    assert _tree_digest(RESULTS) == before, "audit must be read-only on results/"
    return value


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


def test_audit_passes_with_exactly_the_documented_disclosures(report: dict) -> None:
    assert report["read_only"] is True
    assert report["passed"] is True
    assert report["failures"] == []
    assert set(report["disclosures"]) == EXPECTED_DISCLOSURES
    verdicts = {row["id"]: row["verdict"] for row in report["findings"]}
    assert set(verdicts.values()) <= {"PASS", "DISCLOSURE"}
    assert len(verdicts) == 28


def test_document_table_matches_live_recomputation(report: dict) -> None:
    text = AUDIT_MD.read_text(encoding="utf-8")
    assert "\r" not in text
    table = format_table(report)
    for line in table.splitlines()[2:]:
        row_id = line.split("|")[1].strip()
        if row_id in ENVIRONMENT_DEPENDENT_ROWS:
            assert f"| {row_id} |" in text, row_id
            continue
        if row_id in HEAD_DEPENDENT_CLAUSES:
            clause = HEAD_DEPENDENT_CLAUSES[row_id]
            documented = [item for item in text.splitlines() if item.startswith(f"| {row_id} |")]
            assert len(documented) == 1, row_id
            assert clause.search(line) and clause.search(documented[0]), row_id
            assert clause.sub("", line) == clause.sub("", documented[0]), line
            drift = report["preregistration"]["git"]["non_scoped_dependencies_changed_since_prereg"]
            assert clause.search(line).group(1) == str(drift == []), drift
            continue
        assert line in text, line
    for needle in (
        "ACCEPTED WITH DISCLOSURES",
        CLASSIFICATION,
        PREREGISTRATION_COMMIT,
        RESULT_COMMIT,
        RESULTS_TREE,
        MANIFEST_SHA256,
        SOURCE_SHA256,
        "137 byte-exact, 0 EOL-only, 0 mismatch",
        "audit_replay",
        "sequential greedy batch",
        "0.5711",
        "8.0618e-08",
        "sign-test",
        "eliminate_duplicates",
        "cft_revival.experiment_runtime",
        "test_mdo_v1_results.py",
        "interpretation, not evidence",
    ):
        assert needle in text, needle


# --------------------------------------------------------------------------
# bundle integrity
# --------------------------------------------------------------------------


def test_bundle_is_byte_exact_with_no_eol_cases(report: dict) -> None:
    bundle = report["bundle"]
    assert bundle["passed"] is True
    assert bundle["manifest_sha256"] == MANIFEST_SHA256
    assert bundle["manifest_state"] == "accepted_result"
    assert bundle["artifact_count"] == 143
    assert bundle["file_entries"] == 137 and bundle["directory_entries"] == 6
    assert bundle["counts"] == {"byte_exact": 137, "eol_only": 0, "mismatch": 0}
    assert bundle["not_byte_exact"] == [] and bundle["carriage_return_files"] == []
    assert bundle["contracts"] == {"hash-sidecar": 68, "hash-sidecar-metadata": 68, "immutable-lock": 1}
    assert bundle["sidecar_pairs"] == 68 and bundle["sidecar_pairs_consistent"] is True
    assert bundle["blob_artifacts_without_semantic_hash"] == ["artifacts/shakedown.json"]
    assert bundle["on_disk_not_in_manifest"] == [] and bundle["in_manifest_not_on_disk"] == []
    assert bundle["lock_byte_sha256_ok"] and bundle["terminal_byte_sha256_ok"]
    assert bundle["lock"]["attempt"] == 1 and bundle["lock"]["commit"] == PREREGISTRATION_COMMIT
    assert bundle["lock"]["command"] == "python -m experiments.mdo_l0_campaign_v1.run execute"
    assert bundle["lock_acquired_utc"] == "2026-09-02T23:34:54.979408Z"
    assert bundle["terminal_counts"] == {
        "assessment_access_count": 1, "attempt_count": 1, "development_access_count": 1,
        "expensive_operation_count": 10, "label_access_count": 0, "prebundle_access_count": 1,
    }
    assert [name for name, _time in bundle["transitions"]] == [
        "lock-acquired", "cache-prepared", "prebundle-started", "prebundle-completed", "development-started",
        "development-accepted", "assessment-started", "assessment-accepted", "terminal",
    ]
    assert bundle["transitions"][-1][1] == "2026-09-03T00:02:10.481415Z"
    assert 1635.0 <= bundle["lock_to_terminal_s"] <= 1636.0
    assert 34.0 <= bundle["development_s"] <= 35.0
    assert 1600.0 <= bundle["assessment_s"] <= 1601.0
    assert bundle["access_records"] == 13 and bundle["counter_records"] == 14
    assert bundle["access_records_before_operation"] and bundle["counter_before_access"]
    assert bundle["run_access_spacing_accommodates_wall_clocks"] is True
    for operation, spacing in bundle["run_access_spacing"].items():
        assert spacing["gap_to_next_s"] >= spacing["recorded_wall_s"], operation
        if operation.startswith("run-qlognehvi"):
            assert 500.0 < spacing["recorded_wall_s"] < 540.0
    lock = bundle["git_common_lock"]
    if lock["available"]:
        assert lock["content_is_preregistration_commit"] and lock["bytes"] == 41
        assert lock["created_before_runtime_lock"] and 0.0 <= lock["seconds_before_runtime_lock"] < 1.0


def test_manifest_entries_recompute_from_disk() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_bytes())
    for entry in manifest["artifacts"]:
        if entry["type"] != "file":
            continue
        data = (RESULTS / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["byte_sha256"], entry["path"]
        assert len(data) == entry["bytes"], entry["path"]
        assert b"\r" not in data, entry["path"]


# --------------------------------------------------------------------------
# preregistration chain
# --------------------------------------------------------------------------


def test_preregistration_hash_chain(report: dict) -> None:
    prereg = report["preregistration"]
    assert prereg["passed"] is True
    assert prereg["protocol_semantic_sha256"] == "09755b85393d3b3248941ce52f8c21edb832ce30c6f31e5c6919079c41d496ba"
    assert prereg["protocol_semantic_matches_authorities"] and prereg["protocol_semantic_matches_shakedown_record"]
    assert prereg["sealed_protocol_payload_equals_frozen_file"] and prereg["sealed_protocol_is_canonical_json"]
    assert prereg["sealed_authorities_equal_frozen_file"] and prereg["sealed_shakedown_bytes_equal_frozen_file"]
    assert prereg["shakedown_file_sha256"] == "8b5a829302e7aa800d2c60ca1146d86195a71594482c1699a2698e79d76d5c1e"
    assert prereg["shakedown_evidentiary"] is False and prereg["shakedown_outcomes_enter_estimand"] is False
    assert prereg["shakedown_passed"] is True
    assert prereg["shakedown_seeds"] == [900101, 900202]
    assert prereg["shakedown_seed_overlap"] == [] and prereg["shakedown_initial_design_overlap"] == 0
    assert prereg["shakedown_initial_design_overlap_recomputed"] == 0
    assert prereg["shakedown_seed_namespace_rule_recomputed"] is True
    assert prereg["shakedown_source_sha256_equals_authorities"] is True
    assert prereg["shakedown_git_head"] == "a1a53300cdcfcb59d9b82b75697737fe772390c4"
    # the experiment files were untracked at shakedown time; nothing else was dirty
    assert all(entry.startswith("?? modern/experiments/mdo_l0_campaign_v1/") for entry in prereg["shakedown_git_dirty_entries"])
    assert len(prereg["shakedown_git_dirty_entries"]) == 9
    assert prereg["shakedown_result_root_in_temp"] is True
    assert prereg["authorities_source_sha256"] == SOURCE_SHA256
    assert prereg["code_contract_artifact_matches_authorities"] and prereg["package_versions_declared_equal_observed"]
    assert prereg["working_tree_source_sha256_equals_authorities"] is True
    scope = prereg["import_scope"]
    assert scope["available"] is True
    assert scope["hash_scoped_packages_never_imported"] == ["cft_revival.active_learning", "cft_revival.surrogates"]
    assert scope["imported_packages_outside_hash_scope"] == ["cft_revival.experiment_runtime", "cft_revival.kernels", "cft_revival.models"]
    git = prereg["git"]
    if not git["available"]:
        pytest.skip("git history unavailable")
    assert git["prereg_subject_ok"] and git["prereg_experiment_path_isolated"] and git["prereg_contains_no_results"]
    assert len(git["prereg_changed_files"]) == 11
    assert git["prereg_pushed_to_authorized_branch"] and git["prereg_ancestor_of_result"] and git["result_parent_is_prereg"]
    assert git["result_commit_files"] == 140
    assert git["result_commit_files_outside_results"] == [
        "modern/spec/optimization/mdo-l0-campaign-v1.json",
        "modern/tests/experiments/mdo_l0_campaign_v1/test_mdo_v1_results.py",
    ]
    assert git["results_tree_unchanged"] and git["results_untouched_by_later_commits"]
    assert git["results_worktree_clean"] and git["results_worktree_lf"]
    assert git["frozen_blobs_unchanged"]
    assert git["hashed_sources_untouched_since_prereg"] and git["frozen_files_untouched_since_prereg"]
    # imported-but-not-hash-bound packages (F10): frozen between preregistration and result ...
    assert git["non_scoped_dependencies_unchanged_prereg_to_result"] is True
    # ... while their movement in the checkout since then is recorded, not asserted (they moved
    # at bb756418; the package replays above are the evidence that the movement is inert here)
    drift = git["non_scoped_dependencies_changed_since_prereg"]
    assert isinstance(drift, list) and all(item.startswith("modern/src/cft_revival/") for item in drift)
    assert git["non_scoped_dependencies_unchanged_since_prereg"] is (drift == [])
    assert _git("diff", "--name-only", PREREGISTRATION_COMMIT, "HEAD", "--", *audit_module.NON_SCOPED_DEPENDENCY_PATHS).splitlines() == drift
    assert set(git["source_hash_from_blobs"]) == {"4898d0fd", "c553124b", "e642f38c", "ba6875f6"}
    assert all(item["equals_authorities"] and item["entries_equal_authorities"] and item["files"] == 37 for item in git["source_hash_from_blobs"].values())
    assert git["shakedown_head_available"] is True
    assert git["shakedown_head_package_files_identical"] == "34/34"
    assert git["shakedown_head_is_rebased_tests_commit"] is True
    assert git["prereg_author_date"] == "2026-09-03T09:34:16+10:00"
    assert git["prereg_commit_date"] == "2026-09-03T09:34:18+10:00"


def test_result_commit_non_results_changes_are_pointer_and_test_key_fix() -> None:
    diff = _git("diff", PREREGISTRATION_COMMIT, RESULT_COMMIT, "--", "modern/spec/optimization/mdo-l0-campaign-v1.json", "modern/tests/experiments/mdo_l0_campaign_v1/test_mdo_v1_results.py")
    assert '-  "results": null' in diff
    assert f'"preregistration_commit": "{PREREGISTRATION_COMMIT}"' in diff
    assert f'"manifest_sha256": "{MANIFEST_SHA256}"' in diff
    assert '-        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:' in diff
    assert 'entry["byte_sha256"] or len(data) != entry["bytes"]' in diff
    # no source-hash-scoped path is touched between the two commits
    assert _git("diff", "--stat", PREREGISTRATION_COMMIT, RESULT_COMMIT, "--", "modern/src", "modern/spec/optimization/campaign-v1.json", "modern/experiments/mdo_l0_campaign_v1/model.py", "modern/experiments/mdo_l0_campaign_v1/optimizers.py", "modern/experiments/mdo_l0_campaign_v1/experiment.py", "modern/experiments/mdo_l0_campaign_v1/run.py") == ""


# --------------------------------------------------------------------------
# independent replay
# --------------------------------------------------------------------------


def test_independent_replay_matches_the_sealed_evaluations(report: dict) -> None:
    indep = report["independent"]
    assert indep["passed"] is True
    assert indep["sample_bit_exact"] and indep["nominal_bit_exact"] and indep["sample_size"] == 64
    assert indep["survival_max"] == pytest.approx(0.704190217251035, rel=1e-12)
    assert indep["survival_min"] == pytest.approx(0.15558587619997133, rel=1e-12)
    assert all(indep["shared_initial_design_bit_exact"].values()) and len(indep["shared_initial_design_bit_exact"]) == 9
    assert all(indep["lhs_designs_bit_exact"].values()) and len(indep["lhs_designs_bit_exact"]) == 3
    assert indep["records"] == 864
    assert indep["status_counts"] == {"success": 734, "infeasible": 130}
    assert indep["within_tolerance"] and indep["relative_tolerance"] == INDEPENDENT_RELATIVE_TOLERANCE
    assert max(indep["worst_relative_difference"].values()) < 1e-12
    assert indep["worst_relative_difference"]["robust_objectives"] < 1e-15
    assert indep["fail_closed_consistency_failures"] == [] and indep["out_of_bounds"] == [] and indep["non_finite"] == []
    assert indep["duplicate_evaluations_per_run"] == {
        "qlognehvi:101": 0, "nsga3:101": 2, "lhs:101": 0,
        "qlognehvi:202": 0, "nsga3:202": 3, "lhs:202": 0,
        "qlognehvi:303": 0, "nsga3:303": 5, "lhs:303": 0,
    }
    expected_hv = {
        "qlognehvi:101": 0.0038634857735177987, "qlognehvi:202": 0.0038773291110720137, "qlognehvi:303": 0.0038598349135775373,
        "nsga3:101": 0.002925589672975826, "nsga3:202": 0.0035050153811134006, "nsga3:303": 0.0032705234934405926,
        "lhs:101": 0.0028440082962036773, "lhs:202": 0.0032128210523537488, "lhs:303": 0.002803786563725489,
    }
    expected_pareto = {"qlognehvi:101": 28, "qlognehvi:202": 30, "qlognehvi:303": 27, "nsga3:101": 31, "nsga3:202": 24, "nsga3:303": 32, "lhs:101": 19, "lhs:202": 18, "lhs:303": 24}
    expected_infeasible = {"qlognehvi:101": 18, "qlognehvi:202": 14, "qlognehvi:303": 11, "nsga3:101": 7, "nsga3:202": 17, "nsga3:303": 8, "lhs:101": 18, "lhs:202": 18, "lhs:303": 19}
    for key, row in indep["per_run"].items():
        assert row["pareto_indices_equal"] and row["curve_monotone"] and row["evaluations"] == 96
        assert row["hypervolume_recorded"] == expected_hv[key]
        assert row["hypervolume_wfg"] == pytest.approx(expected_hv[key], rel=1e-14)
        assert row["pareto_set_size"] == expected_pareto[key]
        assert row["infeasible_evaluations"] == expected_infeasible[key]
    assert sum(expected_infeasible.values()) == 130
    assert indep["curve_spot_check_worst_relative"] < 1e-14
    assert indep["paired"]["bo_beats_lhs"]["wins"] == 3 and indep["paired"]["bo_beats_nsga3"]["wins"] == 3
    assert indep["paired"]["bo_beats_lhs"]["one_sided_sign_test_p"] == 0.125
    variance = indep["seed_variance"]
    assert variance["qlognehvi"]["mean"] == pytest.approx(0.003866883266055783, rel=1e-12)
    assert variance["qlognehvi"]["sample_std"] == pytest.approx(9.2287e-06, rel=1e-4)
    assert variance["nsga3"]["mean"] == pytest.approx(0.0032337095158432735, rel=1e-12)
    assert variance["lhs"]["mean"] == pytest.approx(0.0029535386374276384, rel=1e-12)
    pooled = indep["pooled"]
    assert pooled["unique_designs"] == 758 == 864 - 96 - 10
    assert pooled["robust_candidates"] == 644 and pooled["nominal_candidates"] == 730
    assert (pooled["robust_front_size"], pooled["nominal_front_size"], pooled["shared_designs"]) == (114, 62, 24)
    assert pooled["jaccard"] == 24 / 152 == pytest.approx(0.158, abs=5e-4)
    assert pooled["robust_front_ids_equal_recorded"] and pooled["nominal_front_ids_equal_recorded"] and pooled["jaccard_equal_recorded"]
    assert pooled["nominal_front_members_robust_feasible"] == 24
    assert pooled["robust_hypervolume_recorded"] == 0.003919578554533065
    assert pooled["robust_hypervolume_relative_difference"] < 1e-14 and pooled["nominal_hypervolume_relative_difference"] < 1e-14
    dense = indep["dense"]
    assert dense["count"] == 8192 and dense["feasible"] == 6576 and dense["infeasible"] == 1616
    assert dense["all_designs_within_bounds"] and dense["spot_replay_status_consistent"]
    assert dense["robust_front_size"] == 291 == dense["robust_front_size_recorded"]
    assert dense["nominal_front_size"] == 166 == dense["nominal_front_size_recorded"]
    assert dense["robust_hypervolume_recorded"] == 0.003797983245976796
    assert dense["robust_hypervolume_relative_difference"] < 1e-14 and dense["nominal_hypervolume_relative_difference"] < 1e-14
    assert dense["spot_replay_count"] == 256 and dense["spot_replay_worst_relative"] < 1e-12
    assert dense["attained_fraction_bo_mean"] == pytest.approx(1.018, abs=5e-4)


def test_wfg_hypervolume_agrees_with_closed_forms() -> None:
    assert wfg_hypervolume([(1.0, 1.0)]) == 1.0
    assert wfg_hypervolume([(1.0, 2.0), (2.0, 1.0)]) == 3.0
    assert wfg_hypervolume([(1.0, 2.0), (2.0, 1.0), (0.5, 0.5)]) == 3.0
    assert wfg_hypervolume([(2.0, 2.0, 2.0), (1.0, 1.0, 3.0)]) == pytest.approx(8.0 + 1.0)
    assert wfg_hypervolume([(0.0, 1.0), (-1.0, 2.0)]) == 0.0
    assert wfg_hypervolume([(1.0, 1.0), (1.0, 1.0)]) == 1.0


def test_wilson_interval_matches_the_protocol_authority() -> None:
    lower, upper = wilson_interval(330, 512)
    assert (lower, upper) == (0.6021349532568827, 0.6847749053232215)


# --------------------------------------------------------------------------
# sensitivity, statistics, claims, labels
# --------------------------------------------------------------------------


def test_sensitivity_tables_recompute(report: dict) -> None:
    sens = report["sensitivity"]
    assert sens["passed"] and sens["priors_match_recorded"] and sens["scenarios_match_recorded"]
    assert sens["design_set_invariance_on_common_set_all_priors"] is True
    table = {row["cusp_upper"]: row for row in sens["priors"]}
    assert set(table) == {0.0, 0.2, 0.45, 0.7}
    assert [table[a]["feasible"] for a in (0.0, 0.2, 0.45, 0.7)] == [397, 480, 644, 687]
    assert [table[a]["front_size"] for a in (0.0, 0.2, 0.45, 0.7)] == [74, 77, 114, 103]
    assert [table[a]["common_feasible_designs"] for a in (0.0, 0.2, 0.45, 0.7)] == [397, 480, 644, 644]
    assert all(row["identical_on_common_feasible_set"] and row["common_front_symmetric_difference"] == 0 for row in table.values())
    assert table[0.0]["jaccard_with_campaign_front"] == 0.0
    assert table[0.45]["jaccard_with_campaign_front"] == 1.0
    assert table[0.7]["jaccard_with_campaign_front"] == pytest.approx(0.4090909090909091)
    assert table[0.45]["survival_max"] == pytest.approx(0.704190217251035, rel=1e-12)
    assert table[0.0]["survival_min"] == 1.0
    scenarios = {row["id"]: row for row in sens["scenarios"]}
    assert list(scenarios) == ["no_wall_loss", "wide_prior_mean", "v4_pooled_uniform_split", "wide_prior_upper", "v4_per_cell_jeffreys"]
    jeffreys = scenarios["v4_per_cell_jeffreys"]
    assert jeffreys["survival"] == pytest.approx(6.85805567999849e-08, rel=1e-12)
    assert jeffreys["thrust_max"] == pytest.approx(2.7027199906330584e-09, rel=1e-12)
    # "thrust <= 2.70e-9 N" in the summaries is the 3-significant-digit rounding
    assert f"{jeffreys['thrust_max']:.2e}" == "2.70e-09" and jeffreys["thrust_max"] > 2.70e-9
    assert jeffreys["pareto_designs_evaluated"] == 114 and jeffreys["pareto_designs_infeasible"] == 0
    no_wall = scenarios["no_wall_loss"]
    assert no_wall["survival"] == 1.0 and no_wall["pareto_designs_evaluated"] == 4 and no_wall["pareto_designs_infeasible"] == 110
    assert scenarios["wide_prior_mean"]["survival"] == pytest.approx(0.775**4)
    assert scenarios["wide_prior_upper"]["survival"] == pytest.approx(0.55**4)
    # the Jeffreys rule as written versus the frozen numbers (disclosure F22)
    assert sens["jeffreys_rule_rounded_4dp"] == [0.5711, 0.9996, 0.9996, 0.0004]
    assert sens["jeffreys_frozen_in_protocol"] == [0.5712, 0.9996, 0.9996, 0.0004]
    assert sens["jeffreys_rule_equals_frozen"] is False
    assert sens["jeffreys_survival_frozen"] == pytest.approx(6.8581e-08, rel=1e-4)
    assert sens["jeffreys_survival_unrounded_rule"] == pytest.approx(8.0618e-08, rel=1e-4)


def test_statistics_sanity(report: dict) -> None:
    stats = report["statistics"]
    assert stats["passed"] is True
    assert stats["wilson_matches_protocol_authority"] is True
    assert stats["v4_pooled_survival"] == pytest.approx(1 - 2962 / 4608)
    assert stats["prior_implied_mean_survival"] == pytest.approx(0.775**4)
    assert stats["calibration_gap_below_0_005"] and abs(stats["calibration_gap"]) == pytest.approx(0.00355, abs=1e-5)
    assert stats["uniform_split_rounds_to_frozen"] is True and stats["uniform_split_frozen"] == 0.2269
    assert stats["binding_gate_count"] == 8 and stats["all_binding_passed"] is True
    assert set(stats["binding_gates"]) == {"replay_bit_exact", "l0_domain", "hypervolume_monotone", "budget_exact", "shared_initial_design", "sample_hash", "pareto_replay", "code_contract"}
    assert stats["replay_bit_exact_replayed"] == 864 and stats["replay_bit_exact_mismatches"] == []
    assert stats["reported_required_wins"] == 2 and stats["reported_seeds"] == 3
    assert stats["null_probability_of_passing_reported_gate"] == 0.5
    assert stats["one_sided_sign_test_p_3_of_3"] == 0.125


def test_claim_boundary_consistency(report: dict) -> None:
    claims = report["claims"]
    assert claims["passed"] is True
    assert claims["protocol_classification"] and claims["sealed_protocol_classification"] and claims["campaign_result_classification"]
    assert claims["campaign_result_claim_boundary_equals_protocol"] and claims["campaign_result_closure"]
    assert len(claims["forbidden_readings"]) == 4
    assert claims["geometry_exclusion_in_protocol"] is True
    assert claims["documents_without_classification_identifier"] == ["README.md", "spec/optimization/mdo-l0-campaign-v1.json"]
    assert all(claims["classification_or_boundary_present"].values())
    assert claims["paper_claims_with_classification"] == ["CLM-029", "CLM-030", "CLM-032", "CLM-033", "CLM-034"]
    assert claims["clm030_non_claims_cover_performance_and_geometry"] and claims["clm030_binds_result_and_prereg_commits"]
    assert claims["gate_found"] and claims["gate_kind"] == "numerical-campaign" and claims["gate_opens_level"] is None
    assert claims["spec_index_results_pointer_ok"] and claims["campaign_v1_benchmark_results_null"]


def test_artifact_labels_disclosure(report: dict) -> None:
    labels = report["labels"]
    assert labels["passed"] is True
    assert "joint q" in labels["protocol_candidate_optimizer"]
    assert labels["recorded_acquisition_label"].endswith("sequential greedy batch")
    assert labels["acquisition_label_says_sequential_while_protocol_declares_joint"] is True
    assert set(labels["nsga3_generations_completed_reported_by_pymoo"].values()) == {7}
    assert labels["nsga3_generations_declared"] == 6
    assert all(gens == list(range(6)) for gens in labels["nsga3_generation_indices_in_provenance"].values())
    assert set(labels["nsga3_pymoo_reported_evaluations"].values()) == {96}


# --------------------------------------------------------------------------
# package replays
# --------------------------------------------------------------------------


def test_package_replay_of_records_and_fronts_is_bit_exact(report: dict) -> None:
    package = report["package"]
    assert package["passed"] is True
    assert package["records_864"]["replayed"] == 864 and package["records_864"]["mismatches"] == []
    assert package["records_864"]["bit_exact"] and package["records_864"]["design_ids_recompute"]
    assert package["pooled_fronts_bit_exact"] and package["per_strategy_fronts_bit_exact"]
    assert package["sensitivity_bit_exact"]["bit_exact"] is True
    assert package["sample_sha256_matches_protocol"] is True
    if HAS_PYMOO:
        assert package["nsga3"]["bit_exact"] is True
        for seed in ("101", "202", "303"):
            run = package["nsga3"]["runs"][seed]
            assert run["record_mismatches"] == [] and run["curve_bit_exact"] and run["final_hypervolume_bit_exact"] and run["pareto_indices_equal"]
            assert run["pymoo_reported_evaluations"] == 96 and run["generations_completed_reported_by_pymoo"] == 7
    else:
        assert "skipped" in package["nsga3"]


@pytest.mark.skipif(not HAS_TORCH, reason="pinned ML runtime (torch/botorch) not importable in this interpreter")
def test_qlognehvi_seed_101_replays_bit_exactly_on_cpu() -> None:
    if os.environ.get("MDO_AUDIT_SKIP_BO") == "1":
        pytest.skip("MDO_AUDIT_SKIP_BO=1")
    before = _tree_digest(RESULTS)
    package = package_replay(Bundle(RESULTS), bo_seeds=(101,))
    assert _tree_digest(RESULTS) == before
    run = package["qlognehvi"]["runs"]["101"]
    assert run["record_mismatches"] == 0 and run["first_divergent_record_index"] is None
    assert run["curve_bit_exact"] and run["final_hypervolume_bit_exact"] and run["pareto_indices_equal"]
    assert run["final_hypervolume_recorded"] == 0.0038634857735177987
    assert run["acquisition_values_bit_exact"] is True
    assert run["iterations"] == 20


# --------------------------------------------------------------------------
# immutability of the evidence and the overlay allowlist
# --------------------------------------------------------------------------


def test_results_tree_is_unchanged_since_the_result_commit() -> None:
    assert _git("rev-parse", f"{RESULT_COMMIT}:{RESULTS_REL}") == RESULTS_TREE
    assert _git("rev-parse", f"HEAD:{RESULTS_REL}") == RESULTS_TREE
    assert _git("status", "--porcelain", "--", RESULTS_REL) == ""
    eol = _git("ls-files", "--eol", "--", RESULTS_REL)
    assert "w/crlf" not in eol and "w/mixed" not in eol
    assert hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest() == MANIFEST_SHA256


def test_frozen_inputs_are_the_preregistered_blobs() -> None:
    for name, blob in audit_module.FROZEN_BLOBS.items():
        relative = f"modern/experiments/mdo_l0_campaign_v1/{name}"
        assert _git("rev-parse", f"{PREREGISTRATION_COMMIT}:{relative}") == blob, name
        assert _git("rev-parse", f"HEAD:{relative}") == blob, name


def test_committed_overlay_is_exact_allowlist() -> None:
    if _git("show", "-s", "--format=%s", "HEAD") != OVERLAY_SUBJECT:
        pytest.skip("HEAD is not the posthoc audit overlay commit")
    changed = set(_git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines())
    assert changed == ALLOWED_OVERLAY


def test_script_refuses_to_write_inside_results(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        audit_module.main(["--json", str(RESULTS / "posthoc.json")])
    assert "must not point inside results/" in capsys.readouterr().err
    assert not (RESULTS / "posthoc.json").exists()
