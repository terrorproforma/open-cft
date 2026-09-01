"""Lifecycle, partition and information-barrier tests for v8."""

from __future__ import annotations

from pathlib import Path

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v8 import models, protocol
from experiments.l0_surrogate_v7.design import group_key, normalized_design, surrogate_inputs


def test_predeclaration_hash_and_prospective_gate_change() -> None:
    value = protocol.load_declaration()
    assert value["predeclaration_hash"] == canonical_hash(
        {key: item for key, item in value.items() if key != "predeclaration_hash"}
    )
    assert value["gates"]["row_coverage_upper"].startswith("diagnostic-only")
    assert value["gates"]["normalized_median_full_interval_width_maximum"] == 0.25
    assert value["gates"]["normalized_p90_full_interval_width_maximum"] == 0.40
    assert value["model_selection"]["budgets"] == [128, 160, 224]


def test_global_roles_and_prior_assessment_are_disjoint() -> None:
    value = protocol.build_partitions()
    roles = value["roles"]
    declaration = protocol.load_declaration()
    inputs = surrogate_inputs(normalized_design(declaration))
    index_sets = []
    group_sets = []
    for role in ("candidate", "method-selection", "final-calibration", "assessment"):
        if role == "candidate":
            index_sets.append(set(roles["candidate_indices"]))
            group_sets.append(
                {
                    group_key(inputs[index], declaration["partition"])
                    for index in roles["candidate_indices"]
                }
            )
        else:
            index_sets.append(
                {
                    index
                    for split in roles[role].values()
                    for index in split["indices"]
                }
            )
            group_sets.append(
                {
                    group
                    for split in roles[role].values()
                    for group in split["groups"]
                }
            )
    for left in range(len(index_sets)):
        for right in range(left + 1, len(index_sets)):
            assert index_sets[left].isdisjoint(index_sets[right])
            assert group_sets[left].isdisjoint(group_sets[right])
    assert value["assessment_prior_coordinate_intersection_count"] == 0


def test_physics_features_are_predeclared_and_distinct() -> None:
    row = (0.2, 0.3, 0.4, 0.5, 0.6)
    assert models.transform("raw-ard-matern52", (row,))[0] == row
    transformed = models.physics_features(row)
    assert len(transformed) == 7
    assert transformed != row


def test_preflight_is_assessment_blind() -> None:
    value = protocol.preflight()
    assert value["passed"]
    assert value["physics_label_access_count"] == 0
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
        assert value["row_coverage_upper_is_diagnostic_only"]
