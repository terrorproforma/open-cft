"""Lifecycle-aware scientific tests for v6."""

from __future__ import annotations

import json
from fractions import Fraction

from cft_revival.surrogates import Prediction
from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v6 import protocol as v6
from experiments.l0_surrogate_v6.conformal import finite_rank, fit_grouped


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_declaration_preflight_and_dependency_hashes() -> None:
    declaration = v6.load_declaration()
    dependency = _load(v6.DEPENDENCIES)
    declared = dependency.pop("dependency_manifest_hash")
    assert canonical_hash(dependency) == declared
    preflight = _load(v6.PREFLIGHT)
    assert preflight["passed"] is True
    assert preflight["v6_physics_evaluation_count"] == 0
    assert preflight["final_calibration_label_access_count"] == 0
    assert preflight["assessment_label_access_count"] == 0
    assert declaration["model_selection"]["training_budgets"] == [96, 128, 160]


def test_single_global_partition_is_strict_and_prior_disjoint() -> None:
    value = _load(v6.PARTITIONS)
    assert value["assessment_prior_coordinate_intersection_count"] == 0
    roles = value["roles"]
    seen_groups = set()
    seen_indices = set()
    for role in ("method-selection", "final-calibration", "assessment"):
        for stratum in ("interpolation", "boundary", "ood"):
            split = roles[role][stratum]
            assert len(split["groups"]) >= 16
            assert len(split["indices"]) >= (
                240 if role == "method-selection" else 192
            )
            assert not seen_groups.intersection(split["groups"])
            assert not seen_indices.intersection(split["indices"])
            seen_groups.update(split["groups"])
            seen_indices.update(split["indices"])
    assert not seen_indices.intersection(roles["candidate_indices"])
    assert "same declared L0 domain" in value["domain_disclosure"]


def test_exact_rational_rank_including_n99_regression() -> None:
    assert finite_rank(99, 9, 10) == 90
    assert finite_rank(9, 9, 10) == 9
    assert Fraction(finite_rank(99, 9, 10), 99) == Fraction(10, 11)


def test_grouped_conformal_weights_groups_equally() -> None:
    groups = {"small": (0,), "large": (1, 2, 3)}
    truth = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    predictions = {index: Prediction(0.0, 1.0) for index in truth}
    distances = {index: 1.0 for index in truth}
    fitted = fit_grouped(
        "symmetric-absolute", groups, truth, predictions, distances
    )
    identity = fitted["quantile_identity"]
    assert identity["group_count"] == 2
    assert identity["weighting"].startswith("each group mass=1/G")
    assert fitted["grouped_conformal"] is True


def test_results_are_valid_before_or_after_execution() -> None:
    manifest_path = v6.RESULTS / "run-manifest.json"
    if not manifest_path.exists():
        assert not v6.RESULTS.exists()
        return
    manifest = _load(manifest_path)
    declared = manifest.pop("run_manifest_hash")
    assert canonical_hash(manifest) == declared
    assert manifest["valid_prospective_result"] is True
    if manifest["status"] == "failed-development-selection-gates":
        assert manifest["final_calibration_labels_accessed"] is False
        assert manifest["assessment_labels_accessed"] is False
    else:
        assert manifest["assessment_prior_coordinate_intersection_count"] == 0
        assert manifest["assessment_labels_accessed_once_after_calibration_freeze"]
