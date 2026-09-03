"""Frozen role partition: determinism, disjointness, stratification, extrapolation cluster."""

from __future__ import annotations

import json

import pytest

from cft_revival.experiment_runtime import semantic_sha256
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.wall_loss_geometry_surrogate_v1 import data as d
from experiments.wall_loss_geometry_surrogate_v1 import experiment
from experiments.wall_loss_geometry_surrogate_v1.experiment import (
    AUTHORITIES_PATH,
    PARTITIONS_PATH,
    evidentiary_plan,
    plan_partition,
    protocol,
    shakedown_disjointness,
    shakedown_plan,
)


@pytest.fixture(scope="module")
def rows():
    return experiment.load_rows(protocol())


def test_partition_is_deterministic_disjoint_and_stratified(rows) -> None:
    value = protocol()
    first = plan_partition(value, evidentiary_plan(value), rows)
    second = plan_partition(value, evidentiary_plan(value), rows)
    assert first == second
    assert first["counts"] == value["partition"]["counts"]["totals"]
    ids = [case_id for role in d.ALL_ROLES for case_id in first["roles"][role]]
    assert len(ids) == len(set(ids)) == 96
    for stratum in ("primary", "extension"):
        declared = value["partition"]["counts"][stratum]
        for role in d.ROLES:
            assert first["stratum_counts"][stratum][role] == declared[role]
    by_id = {row.case_id: row for row in rows}
    design_ids = [by_id[c].design_id for c in ids]
    assert len(set(design_ids)) == 96
    inputs = {by_id[c].inputs for c in ids}
    assert len(inputs) == 96


def test_extrapolation_cluster_is_the_top_decile_of_chamber_length(rows) -> None:
    value = protocol()
    partition = plan_partition(value, evidentiary_plan(value), rows)
    cluster = partition["extrapolation_cluster"]
    longest = sorted(rows, key=lambda row: (-row.chamber_length_m, row.case_id))[:10]
    assert partition["roles"]["extrapolation"] == sorted(row.case_id for row in longest)
    assert cluster["stage_counts"] == [5]
    assert cluster["nondominated_in_cluster"] == value["partition"]["extrapolation_cluster"]["expected_nondominated_in_cluster"] == 3
    assert cluster["next_shorter_chamber_length_m"] < cluster["chamber_length_threshold_m"]
    by_id = {row.case_id: row for row in rows}
    for role in d.ROLES:
        assert all(by_id[c].chamber_length_m < cluster["chamber_length_threshold_m"] for c in partition["roles"][role])


def test_shakedown_partition_differs_from_the_evidentiary_partition(rows) -> None:
    value = protocol()
    report = shakedown_disjointness(value, rows)
    assert report["proven"] is True
    assert report["namespace_differs"] is True
    assert report["role_assignment_identical"] is False
    assert report["shakedown_design_sha256"] != report["evidentiary_design_sha256"]
    assert report["per_role_overlap"]["extrapolation"]["shared"] == 10
    shakedown = plan_partition(value, shakedown_plan(value), rows)
    evidentiary = plan_partition(value, evidentiary_plan(value), rows)
    assert shakedown["roles"]["assessment"] != evidentiary["roles"]["assessment"]


def test_disjointness_fails_when_the_shakedown_reuses_the_evidentiary_seed(rows) -> None:
    value = json.loads(json.dumps(protocol()))
    value["shakedown"]["seed_namespace"] = value["partition"]["seed_namespace"]
    value["shakedown"]["seed"] = value["partition"]["seed"]
    report = shakedown_disjointness(value, rows)
    assert report["proven"] is False
    assert report["role_assignment_identical"] is True


def test_partition_fails_closed_on_declared_count_mismatch(rows) -> None:
    value = json.loads(json.dumps(protocol()))
    value["partition"]["counts"]["primary"]["fit"] = 12
    with pytest.raises(ValueError, match="role counts do not sum"):
        d.build_partition(rows, value["partition"])
    value = json.loads(json.dumps(protocol()))
    value["partition"]["counts"]["primary"]["remaining_after_extrapolation"] = 21
    with pytest.raises(ValueError, match="designs remain"):
        d.build_partition(rows, value["partition"])
    value = json.loads(json.dumps(protocol()))
    value["partition"]["extrapolation_cluster"]["count"] = 9
    with pytest.raises(ValueError, match="ceil"):
        d.build_partition(rows, value["partition"])


def test_labels_for_role_returns_the_frozen_order_and_rejects_unknown_roles(rows) -> None:
    value = protocol()
    partition = plan_partition(value, evidentiary_plan(value), rows)
    fit = d.labels_for_role(rows, partition, "fit")
    assert [row.case_id for row in fit] == partition["roles"]["fit"]
    with pytest.raises(ValueError, match="unknown role"):
        d.labels_for_role(rows, partition, "test")


def test_frozen_partitions_file_equals_the_recomputation_when_present(rows) -> None:
    if not PARTITIONS_PATH.is_file():
        pytest.skip("partitions.json not yet prepared")
    value = protocol()
    frozen = strict_json_file(PARTITIONS_PATH)
    recomputed = plan_partition(value, evidentiary_plan(value), rows)
    assert semantic_sha256(frozen) == semantic_sha256(recomputed)
    assert b"\r" not in PARTITIONS_PATH.read_bytes()
    if AUTHORITIES_PATH.is_file():
        authorities = strict_json_file(AUTHORITIES_PATH)
        assert authorities["partitions_semantic_sha256"] == semantic_sha256(recomputed)
        assert authorities["partition_role_sha256"] == recomputed["role_sha256"]
