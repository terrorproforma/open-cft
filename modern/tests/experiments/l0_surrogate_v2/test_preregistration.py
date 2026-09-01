"""Pre-execution protocol tests; these never load final-assessment labels."""

from __future__ import annotations

import copy
import json
from math import inf
from pathlib import Path

import pytest

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v2.protocol import (
    PARTITIONS,
    PREDECLARATION,
    SingleUseAssessmentLoader,
    TrainingOracle,
    _order_statistic,
    group_key,
    load_l0_rows,
    load_predeclaration,
    select_active_indices,
)


def test_predeclaration_is_hash_pinned_and_has_no_outcomes() -> None:
    declaration = load_predeclaration()
    assert declaration["predeclaration_hash"] == (
        "640ec66125d4de07944b012402a64e6cd7be012f1b6877b166b12c4271ff15cf"
    )
    assert declaration["campaign"]["final_rows"] == 96
    assert declaration["campaign"]["assessment_based_stopping"] is False
    assert len(declaration["partition"]["replicate_seeds"]) == 3
    assert declaration["quality_gates"]["thresholds_tunable_after_execution"] is False
    serialized = PREDECLARATION.read_text(encoding="utf-8")
    for forbidden in ('"rmse":', '"coverage":', '"selected_indices":', '"truth":'):
        assert forbidden not in serialized


def test_input_only_partitions_are_strict_and_group_disjoint() -> None:
    declaration = load_predeclaration()
    manifest = json.loads(PARTITIONS.read_text(encoding="utf-8"))
    assert manifest["partitions_hash"] == canonical_hash(
        {key: value for key, value in manifest.items() if key != "partitions_hash"}
    )
    assert "label" not in json.dumps(manifest["replicates"]).lower()
    inputs, _ = load_l0_rows(declaration)
    grouping = declaration["partition"]["grouping"]

    for replicate in manifest["replicates"]:
        candidate = set(replicate["candidate_indices"])
        used_groups: set[str] = set()
        used_indices: set[int] = set()
        used_coordinates: set[tuple[float, ...]] = set()
        for role in ("calibration", "assessment"):
            for stratum in ("interpolation", "boundary", "ood"):
                split = replicate[role][stratum]
                groups = set(split["groups"])
                indices = set(split["indices"])
                coordinates = {tuple(inputs[index]) for index in indices}
                assert len(groups) >= 6
                assert len(indices) >= 96
                assert used_groups.isdisjoint(groups)
                assert used_indices.isdisjoint(indices)
                assert used_coordinates.isdisjoint(coordinates)
                assert {
                    group_key(inputs[index], grouping) for index in indices
                } == groups
                used_groups.update(groups)
                used_indices.update(indices)
                used_coordinates.update(coordinates)
        assert candidate.isdisjoint(used_indices)
        assert {
            group_key(inputs[index], grouping) for index in candidate
        }.isdisjoint(used_groups)
        assert len(candidate) >= 96


def test_calibration_label_perturbation_cannot_change_selection() -> None:
    declaration = load_predeclaration()
    mini = copy.deepcopy(declaration)
    mini["campaign"]["initial_rows"] = 8
    mini["campaign"]["acquisition_batch_rows"] = 4
    mini["campaign"]["final_rows"] = 16
    inputs = tuple(
        tuple(((index + 1) * (dimension * 4 + 3) % 101) / 101.0 for dimension in range(5))
        for index in range(48)
    )
    training_outputs = tuple(
        (
            0.01 + 0.02 * row[0] + 0.005 * row[2],
            700.0 + 1500.0 * row[0] ** 0.5 + 80.0 * row[4],
        )
        for row in inputs
    )
    calibration_labels = [(-1.0, -1.0)] * 20
    perturbed_calibration_labels = [(1.0e12, -1.0e12)] * 20
    first = select_active_indices(
        inputs,
        tuple(range(len(inputs))),
        TrainingOracle(training_outputs, tuple(range(len(inputs)))),
        mini,
    )[0]
    assert calibration_labels != perturbed_calibration_labels
    second = select_active_indices(
        inputs,
        tuple(range(len(inputs))),
        TrainingOracle(training_outputs, tuple(range(len(inputs)))),
        mini,
    )[0]
    assert first == second


def test_outward_quantile_rounding_and_single_use_loader() -> None:
    values = (1.0, 2.0, 3.0, 4.0)
    assert _order_statistic(values, 2, -inf) < 2.0
    assert _order_statistic(values, 3, inf) > 3.0
    outputs = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    assessment = {
        "interpolation": {"indices": [0]},
        "boundary": {"indices": [1]},
        "ood": {"indices": [2]},
    }
    loader = SingleUseAssessmentLoader(outputs, assessment, "frozen")
    labels = loader.load("frozen")
    assert labels["ood"] == ((2, (5.0, 6.0)),)
    with pytest.raises(RuntimeError, match="only once"):
        loader.load("frozen")


def test_empty_results_placeholder_contains_no_assessment_results() -> None:
    path = PREDECLARATION.parent / "results-placeholder.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["status"] == "not-executed"
    assert value["assessment_labels"] == []
    assert value["results"] == []
    assert not (PREDECLARATION.parent / "results" / "run-manifest.json").exists()
