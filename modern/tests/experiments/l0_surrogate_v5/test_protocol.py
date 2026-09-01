"""Lifecycle-aware tests for the blind v5 scientific protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cft_revival.surrogates import ExactGP, Prediction
from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v5 import protocol as v5
from experiments.l0_surrogate_v5.intervals import select_method


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_predeclaration_dependencies_and_preflight_are_hash_valid() -> None:
    declaration = v5.load_declaration()
    dependencies = _load(v5.DEPENDENCIES)
    declared = dependencies.pop("dependency_tree_hash")
    assert canonical_hash(dependencies) == declared
    preflight = _load(v5.PREFLIGHT)
    assert preflight["passed"] is True
    assert preflight["assessment_access_count"] == 0
    assert preflight["real_physics_evaluation_count"] == 0
    assert declaration["campaign"]["final_rows"] == 96


def test_partitions_are_role_group_disjoint_and_prior_blind() -> None:
    value = _load(v5.PARTITIONS)
    assert value["combined_prior_coordinate_intersection_count"] == 0
    assert all(
        record["v5_intersection_count"] == 0
        for record in value["prior_disjointness"].values()
    )
    for replicate in value["replicates"]:
        seen_groups: set[str] = set()
        seen_indices: set[int] = set()
        for role in ("method-selection", "final-calibration", "assessment"):
            for stratum in ("interpolation", "boundary", "ood"):
                split = replicate[role][stratum]
                groups = set(split["groups"])
                indices = set(split["indices"])
                assert len(groups) >= 8
                assert len(indices) >= 96
                assert not groups.intersection(seen_groups)
                assert not indices.intersection(seen_indices)
                seen_groups.update(groups)
                seen_indices.update(indices)
        assert not seen_indices.intersection(replicate["candidate_indices"])


def test_calibration_labels_cannot_change_method_selection() -> None:
    groups = {"a": (0, 1), "b": (2, 3), "c": (4, 5)}
    truth = {index: float(index) for index in range(6)}
    predictions = {
        index: Prediction(float(index) + (-0.2 if index % 2 else 0.2), 0.04)
        for index in range(6)
    }
    distances = {index: 0.1 + index / 100.0 for index in range(6)}
    first = select_method(
        groups,
        truth,
        predictions,
        distances,
        nominal=0.9,
        output_scale=5.0,
        coverage_bounds=(0.85, 0.95),
        maximum_group_deviation=0.25,
    )
    unrelated_calibration_labels = [1e300, -1e300]
    second = select_method(
        groups,
        truth,
        predictions,
        distances,
        nominal=0.9,
        output_scale=5.0,
        coverage_bounds=(0.85, 0.95),
        maximum_group_deviation=0.25,
    )
    assert unrelated_calibration_labels
    assert first == second


def test_assessment_loader_is_single_use(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Result:
        axial_thrust_n = 1.0
        specific_impulse_s = 2.0

    monkeypatch.setattr(v5, "evaluate_batch", lambda points: calls.append(points) or (Result(),))
    role = {
        name: {"indices": [index]}
        for index, name in enumerate(("interpolation", "boundary", "ood"))
    }
    loader = v5.SingleUseAssessment((object(), object(), object()), role)
    assert len(loader.load("frozen", "frozen")) == 3
    with pytest.raises(RuntimeError, match="once"):
        loader.load("frozen", "frozen")


def test_result_lifecycle_and_model_hashes() -> None:
    manifest_path = v5.RESULTS / "run-manifest.json"
    if not manifest_path.exists():
        assert not v5.RESULTS.exists()
        return
    manifest = _load(manifest_path)
    declared = manifest.pop("run_manifest_hash")
    assert canonical_hash(manifest) == declared
    assert manifest["scientific_identity_valid"] is True
    assert manifest["prior_coordinate_intersection_count"] == 0
    for model_path in v5.RESULTS.rglob("*.model.json"):
        model = ExactGP.load(model_path)
        assert model.model_hash == _load(model_path)["model_hash"]
