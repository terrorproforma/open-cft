from __future__ import annotations

from experiments.l1a_field_surrogate_v2.experiment import (
    assessment_groups,
    high_indices,
    raw_designs,
    role_indices,
)
from experiments.l1a_field_surrogate_v2.protocol import (
    GEOMETRY_PREFLIGHT,
    PARTITIONS,
    PROTOCOL,
    SYNTHETIC_PREFLIGHT,
    verify_json,
)


def test_raw_candidates_are_fresh_deterministic_and_v1_disjoint() -> None:
    left, right = raw_designs(), raw_designs()
    assert len(left) == len(right) == 256
    assert [item.design_id for item in left] == [item.design_id for item in right]
    assert len({item.design_id for item in left}) == 256


def test_frozen_roles_and_fidelities_are_exact() -> None:
    roles = [
        set(role_indices(name))
        for name in ("candidate", "method_selection", "final_calibration", "single_use_assessment")
    ]
    assert set.union(*roles) == set(range(112))
    assert sum(map(len, roles)) == 112
    assert set.union(*(set(value) for value in assessment_groups().values())) == set(range(96, 112))
    assert len(high_indices()) == 80
    assert PROTOCOL["fidelities"]["low"]["shape"] == [41, 73]
    assert PROTOCOL["fidelities"]["high"]["shape"] == [81, 145]


def test_recorded_geometry_preflight_is_label_free_and_rebuild_exact() -> None:
    geometry = verify_json(GEOMETRY_PREFLIGHT)
    partitions = verify_json(PARTITIONS)
    synthetic = verify_json(SYNTHETIC_PREFLIGHT)
    assert geometry["raw_count"] == 256
    assert geometry["valid_count"] + geometry["rejected_count"] == 256
    assert geometry["field_solver_access_count"] == geometry["qoi_label_access_count"] == 0
    assert geometry["frozen_hash_failure_count"] == 0
    assert len(geometry["frozen_rebuild_records"]) == 112
    assert len(partitions["frozen_raw_indices"]) == 112
    assert partitions["v1_coordinate_intersection_count"] == 0
    assert synthetic["field_solver_access_count"] == synthetic["qoi_label_access_count"] == 0
    assert synthetic["passed"] is True
