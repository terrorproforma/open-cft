"""Lifecycle and information-barrier tests for v7."""

from __future__ import annotations

import json

import pytest

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v7 import protocol as v7


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_predeclaration_dependencies_and_preflight() -> None:
    declaration = v7.load_declaration()
    dependency = _load(v7.DEPENDENCIES)
    declared = dependency.pop("dependency_manifest_hash")
    assert canonical_hash(dependency) == declared
    preflight = _load(v7.PREFLIGHT)
    assert preflight["passed"] is True
    assert preflight["rank_regressions"] == {"n99": 90, "n239": 216}
    assert preflight["physics_label_access_count"] == 0
    assert preflight["assessment_label_access_count"] == 0
    assert declaration["conformal"]["exchangeability_unit"] == "independent spatial group"


def test_global_roles_and_fresh_assessment() -> None:
    value = _load(v7.PARTITIONS)
    assert value["assessment_prior_coordinate_intersection_count"] == 0
    roles = value["roles"]
    groups = set()
    indices = set()
    for role in ("method-selection", "final-calibration", "assessment"):
        for stratum in ("interpolation", "boundary", "ood"):
            split = roles[role][stratum]
            assert len(split["groups"]) >= 40
            assert len(split["indices"]) >= 240
            assert not groups.intersection(split["groups"])
            assert not indices.intersection(split["indices"])
            groups.update(split["groups"])
            indices.update(split["indices"])
    assert not indices.intersection(roles["candidate_indices"])
    assert "same-domain" in value["domain_disclosure"]


def test_assessment_is_exactly_once_and_hash_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        axial_thrust_n = 1.0
        specific_impulse_s = 2.0

    monkeypatch.setattr(v7, "evaluate_batch", lambda points: (Result(),))
    role = {
        name: {"indices": [index]}
        for index, name in enumerate(("interpolation", "boundary", "ood"))
    }
    loader = v7.SingleUseAssessment((object(), object(), object()), role)
    with pytest.raises(ValueError, match="hash"):
        loader.load("a", "b")
    assert loader.load("a", "a")
    with pytest.raises(RuntimeError, match="single-use"):
        loader.load("a", "a")


def test_v7_does_not_import_v5_floating_rank_code() -> None:
    source = (v7.ROOT / "protocol.py").read_text(encoding="utf-8")
    conformal = (v7.ROOT / "cluster_conformal.py").read_text(encoding="utf-8")
    assert "l0_surrogate_v5" not in source
    assert "l0_surrogate_v5" not in conformal


def test_results_lifecycle() -> None:
    path = v7.RESULTS / "run-manifest.json"
    if not path.exists():
        assert not v7.RESULTS.exists()
        return
    value = _load(path)
    declared = value.pop("run_manifest_hash")
    assert canonical_hash(value) == declared
    assert value["valid_prospective_result"] is True
    if "assessment_metrics" not in value:
        assert value["assessment_labels_accessed"] is False
    else:
        assert value["assessment_prior_coordinate_intersection_count"] == 0
        assert value["assessment_accessed_once_after_calibration_freeze"] is True
