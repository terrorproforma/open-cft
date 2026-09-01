"""Preregistration, role and lifecycle tests for v9."""

from __future__ import annotations

from pathlib import Path

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v7.design import group_key, normalized_design, surrogate_inputs
from experiments.l0_surrogate_v9 import protocol


def test_predeclaration_hash_and_unchanged_gates() -> None:
    value = protocol.load_declaration()
    assert value["predeclaration_hash"] == canonical_hash(
        {key: item for key, item in value.items() if key != "predeclaration_hash"}
    )
    assert value["model_selection"]["budgets"] == [64, 96, 128, 160, 224]
    assert value["gates"]["range_normalized_rmse_maximum"] == 0.05
    assert value["gates"]["worst_case_range_normalized_error_maximum"] == 0.15
    assert value["gates"]["row_coverage_upper"] == "diagnostic-only"


def test_global_roles_and_prior_assessments_are_disjoint() -> None:
    value = protocol.build_partitions()
    declaration = protocol.load_declaration()
    inputs = surrogate_inputs(normalized_design(declaration))
    roles = value["roles"]
    index_sets = []
    group_sets = []
    for role in ("candidate", "method-selection", "final-calibration", "assessment"):
        if role == "candidate":
            indices = set(roles["candidate_indices"])
        else:
            indices = {
                int(index)
                for split in roles[role].values()
                for index in split["indices"]
            }
        index_sets.append(indices)
        group_sets.append({group_key(inputs[index], declaration["partition"]) for index in indices})
    for left in range(4):
        for right in range(left + 1, 4):
            assert index_sets[left].isdisjoint(index_sets[right])
            assert group_sets[left].isdisjoint(group_sets[right])
    assert value["assessment_prior_coordinate_intersection_count"] == 0


def test_preflight_is_assessment_blind_and_reload_safe() -> None:
    value = protocol.preflight()
    assert value["passed"]
    assert value["analytic_reference_identity"]["passed"]
    assert value["model_reload_hash_valid"]
    assert value["assessment_label_access_count"] == 0


def test_results_lifecycle() -> None:
    result = Path(protocol.RESULTS) / "run-manifest.json"
    if not result.exists():
        assert not Path(protocol.RESULTS).exists()
        return
    value = protocol._load(result)
    assert value["valid_prospective_result"]
    if "assessment_metrics" not in value:
        assert not value["assessment_labels_accessed"]
    else:
        assert value["assessment_accessed_once_after_calibration_freeze"]
