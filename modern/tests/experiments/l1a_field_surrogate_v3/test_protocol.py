from __future__ import annotations

import numpy as np

from experiments.l1a_field_surrogate_v3.experiment import (
    WeightedPOD,
    raw_designs,
    role_indices,
    stratum_indices,
)
from experiments.l1a_field_surrogate_v3.protocol import (
    DEPENDENCY_LOCK,
    GEOMETRY_PREFLIGHT,
    PARTITIONS,
    PROTOCOL,
    SYNTHETIC_PREFLIGHT,
    verify_json,
)


def test_raw_seed_and_role_sizes_are_frozen() -> None:
    left, right = raw_designs(), raw_designs()
    assert len(left) == len(right) == 512
    assert [item.design_id for item in left] == [item.design_id for item in right]
    assert len(set(item.design_id for item in left)) == 512
    assert len(role_indices("candidate")) == 128
    assert len(role_indices("method_selection")) == 16
    assert len(role_indices("final_calibration")) == 48
    assert len(role_indices("single_use_assessment")) == 48
    assert all(len(rows) == 16 for rows in stratum_indices("calibration").values())
    assert all(len(rows) == 16 for rows in stratum_indices("assessment").values())


def test_preflight_is_geometry_complete_and_label_free() -> None:
    geometry = verify_json(GEOMETRY_PREFLIGHT)
    partition = verify_json(PARTITIONS)
    synthetic = verify_json(SYNTHETIC_PREFLIGHT)
    dependency = verify_json(DEPENDENCY_LOCK)
    assert geometry["raw_count"] == 512
    assert geometry["valid_count"] + geometry["rejected_count"] == 512
    assert geometry["frozen_hash_failure_count"] == 0
    assert len(geometry["frozen_rebuild_records"]) == 240
    assert geometry["field_solver_access_count"] == geometry["qoi_label_access_count"] == 0
    assert partition["prior_role_coordinate_intersection_count"] == 0
    assert synthetic["field_solver_access_count"] == synthetic["qoi_label_access_count"] == 0
    assert dependency["python"]["version"]
    assert dependency["packages"]["numpy"]["version"] == np.__version__
    assert dependency["packages"]["warp-lang"]["version"]
    assert len(dependency["pyproject"]["git_blob"]) == 40


def test_weighted_pod_fails_closed_or_retains_declared_energy() -> None:
    rng = np.random.default_rng(7)
    snapshots = rng.normal(size=(12, 2 * 81 * 145))
    original = PROTOCOL["field_model"]["rank_cap"]
    assert original == 64
    basis = WeightedPOD.fit(snapshots)
    assert basis is not None
    assert basis.rank <= 12
    assert basis.retained >= 0.995
