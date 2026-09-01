from __future__ import annotations

import numpy as np

from experiments.l1a_field_surrogate_v1.experiment import (
    assessment_groups,
    exact_rank,
    high_indices,
    prolong_low,
    role_indices,
    sample_designs,
)
from experiments.l1a_field_surrogate_v1.protocol import PROTOCOL


def test_fresh_roles_are_deterministic_disjoint_and_nested() -> None:
    left, right = sample_designs(), sample_designs()
    assert [item.design_id for item in left] == [item.design_id for item in right]
    assert len({item.design_id for item in left}) == 112
    roles = [
        set(role_indices(name))
        for name in ("candidate", "method_selection", "final_calibration", "single_use_assessment")
    ]
    assert set.union(*roles) == set(range(112))
    assert sum(len(item) for item in roles) == len(set.union(*roles))
    assert set(high_indices()).issubset(range(112))
    assert set(range(64, 112)).issubset(high_indices())
    assert set.union(*(set(value) for value in assessment_groups().values())) == set(range(96, 112))


def test_protocol_fixes_legitimate_fidelities_and_claim_boundary() -> None:
    assert PROTOCOL["fidelities"]["low"]["shape"] == [41, 73]
    assert PROTOCOL["fidelities"]["high"]["shape"] == [81, 145]
    assert PROTOCOL["sampling"]["prior_l1a_v2_coordinate_intersection_required"] == 0
    assert PROTOCOL["execution"]["maximum_executions"] == 1
    assert "NUMERICAL_EMULATION_ONLY" in PROTOCOL["classification"]
    assert exact_rank(5, 0.8) == 5
    assert exact_rank(6, 0.8) == 6


def test_nested_prolongation_copies_coincident_nodes_exactly() -> None:
    class Field:
        b_r_t = tuple(tuple(float(i * 10 + j) for j in range(3)) for i in range(2))
        b_z_t = tuple(tuple(float(100 + i * 10 + j) for j in range(3)) for i in range(2))

    vector = prolong_low(Field())
    br = vector[:15].reshape((3, 5))
    bz = vector[15:].reshape((3, 5))
    assert np.array_equal(br[::2, ::2], np.asarray(Field.b_r_t))
    assert np.array_equal(bz[::2, ::2], np.asarray(Field.b_z_t))
