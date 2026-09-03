"""Evaluation chain: sample, closure CL-1, L0 domain, CVaR, replay, hypervolume."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from cft_revival.experiment_runtime import canonical_bytes
from cft_revival.optimization import ObjectiveDirection
from cft_revival.physics import PhysicsValidationError

from experiments.mdo_l0_campaign_v1 import model as m


def test_qmc_sample_is_deterministic_in_bounds_and_hashed() -> None:
    sample = m.uncertain_sample()
    again = m.uncertain_sample()
    assert sample == again
    assert len(sample) == 64
    for theta in sample:
        assert list(theta) == list(m.UNCERTAIN_NAMES)
        for name, lower, upper, _units in m.UNCERTAIN_INPUTS:
            assert lower <= theta[name] <= upper
    assert m.sample_sha256(sample) == "6e574ff122894e0facf951cdf89069c1b4625d6082a33b7026ff4d8a776db33e"
    rows = m.unit_qmc_rows(8, 3)
    assert all(0.0 < c < 1.0 for row in rows for c in row)
    assert m.unit_qmc_rows(8, 3) != m.unit_qmc_rows(8, 4)
    with pytest.raises(m.ModelError):
        m.unit_qmc_rows(0, 1)
    nominal = m.nominal_theta()
    assert nominal["cusp_probability_cell_1"] == 0.225
    assert nominal["ionized_number_fraction"] == pytest.approx(0.815)


def test_alternative_cusp_priors_only_rescale_cusp_columns() -> None:
    base = m.uncertain_sample()
    alt = m.uncertain_sample(cusp_upper=0.2)
    for theta_base, theta_alt in zip(base, alt, strict=True):
        for name in m.UNCERTAIN_NAMES:
            if name in m.CUSP_NAMES:
                assert theta_alt[name] == pytest.approx(theta_base[name] * 0.2 / 0.45)
            else:
                assert theta_alt[name] == theta_base[name]
    zero = m.uncertain_sample(cusp_upper=0.0)
    assert all(theta[name] == 0.0 for theta in zero for name in m.CUSP_NAMES)
    with pytest.raises(m.ModelError):
        m.uncertain_bounds(1.0)


def test_closure_cl1_is_a_multiplicative_cascade() -> None:
    theta = dict(m.nominal_theta())
    assert m.cusp_survival(theta) == pytest.approx((1 - 0.225) ** 4)
    for name in m.CUSP_NAMES:
        theta[name] = 0.0
    assert m.cusp_survival(theta) == 1.0
    theta["cusp_probability_cell_2"] = 0.9996
    assert m.cusp_survival(theta) == pytest.approx(0.0004)
    theta["cusp_probability_cell_2"] = 1.0
    with pytest.raises(m.ModelError):
        m.cusp_survival(theta)
    # Calibration of the wide prior against the v4 pooled survival.
    assert abs((1 - 0.225) ** 4 - (1 - 2962 / 4608)) < 0.005


def test_evaluate_design_success_and_fail_closed_infeasible() -> None:
    sample = m.uncertain_sample()
    good = m.evaluate_design((300.0, 1.0, 1e-6), sample)
    assert good.status == "success"
    assert good.failure_code is None
    assert good.robust_margin_a > 0.0 and good.nominal_margin_a > 0.0
    assert good.robust_objectives is not None and good.nominal_objectives is not None
    assert good.robust_objectives[3] == pytest.approx(300.0)
    assert good.robust_objectives[0] < good.nominal_objectives[0]
    assert good.robust_statistics["axial_thrust_n"]["cvar"] == good.robust_objectives[0]
    assert good.robust_statistics["axial_thrust_n"]["minimum"] <= good.robust_objectives[0]
    assert len(good.sample_result_sha256) == 64
    record = good.to_record()
    assert record["status"] == "success"
    assert set(record["robust_objectives"]) == set(m.OBJECTIVE_NAMES)

    bad = m.evaluate_design((300.0, 0.2, 1e-6), sample)
    assert bad.status == "infeasible"
    assert bad.failure_code == m.INFEASIBLE_CODE
    assert bad.robust_margin_a < 0.0
    assert bad.robust_objectives is None and bad.robust_statistics is None
    assert bad.sample_result_sha256 is None
    # Nominally feasible but robust-infeasible designs still carry nominal objectives.
    edge = m.evaluate_design((300.0, 1.05, 1.5e-6), sample)
    if edge.status == "infeasible" and edge.nominal_margin_a >= 0.0:
        assert edge.nominal_objectives is not None
    # The L0 domain is enforced by the physics package, not re-implemented here.
    theta = dict(m.nominal_theta())
    with pytest.raises(PhysicsValidationError):
        m.evaluate_l0((300.0, 0.05, 2e-6), theta)


def test_evaluation_is_bit_exact_on_replay() -> None:
    sample = m.uncertain_sample()
    rng = random.Random(7)
    for _ in range(10):
        values = (rng.uniform(150, 500), rng.uniform(0.1, 2.5), rng.uniform(2e-7, 2e-6))
        first = m.evaluate_design(values, sample).to_record()
        second = m.evaluate_design(values, sample).to_record()
        assert canonical_bytes(first) == canonical_bytes(second)


def test_cl1_objectives_are_separable_in_design_and_uncertainty() -> None:
    """f(x, theta) = g(x) h(theta): robust/nominal ratios are design independent."""

    sample = m.uncertain_sample()
    rng = random.Random(11)
    ratios = {name: set() for name in m.OBJECTIVE_NAMES}
    count = 0
    while count < 12:
        values = (rng.uniform(150, 500), rng.uniform(1.5, 2.5), rng.uniform(2e-7, 1e-6))
        evaluation = m.evaluate_design(values, sample)
        if evaluation.status != "success":
            continue
        count += 1
        for index, name in enumerate(m.OBJECTIVE_NAMES):
            ratios[name].add(round(evaluation.robust_objectives[index] / evaluation.nominal_objectives[index], 9))
    for name, values in ratios.items():
        assert len(values) == 1, (name, values)
    assert ratios["anode_input_power_w"] == {1.0}


def test_cvar_and_tail_count_definitions() -> None:
    assert m.tail_count(64, 0.25) == 16
    assert m.tail_count(24, 0.25) == 6
    assert m.tail_count(1, 0.25) == 1
    with pytest.raises(m.ModelError):
        m.tail_count(0, 0.25)
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert m.cvar(values, ObjectiveDirection.MAXIMIZE, 2) == 1.5
    assert m.cvar(values, ObjectiveDirection.MINIMIZE, 2) == 4.5
    with pytest.raises(m.ModelError):
        m.cvar(values, ObjectiveDirection.MAXIMIZE, 6)


def test_normalized_frame_maps_reference_point_to_origin() -> None:
    reference = [m.REFERENCE_POINT[name] for name in m.OBJECTIVE_NAMES]
    assert m.normalized_objectives(reference) == (0.0, 0.0, 0.0, 0.0)
    better = m.normalized_objectives([0.06, 3000.0, 1.0, 0.0])
    assert better == (1.0, 1.0, 1.0, 1.0)
    with pytest.raises(m.ModelError):
        m.normalized_objectives([1.0, 2.0])


def test_nondominated_matches_pairwise_definition_and_keeps_first_duplicate() -> None:
    rng = np.random.default_rng(3)
    for dims in (2, 3, 4):
        points = rng.random((40, dims)).tolist()
        points.append(list(points[0]))  # exact duplicate of the first point
        expected = {
            index
            for index, candidate in enumerate(points)
            if not any(
                other_index != index and m.dominates_maximize(other, candidate)
                for other_index, other in enumerate(points)
            )
        }
        expected.discard(len(points) - 1)
        assert set(m.nondominated_indices(points)) == expected
    assert m.nondominated_indices([]) == ()
    # Roundoff-aware dominance: a one-ulp power tie no longer protects the point.
    tied = [(1.0, 1.0, 1.0, 0.5), (0.9, 1.0, 1.0, 0.5 + 1e-16)]
    assert set(m.nondominated_indices(tied)) == {0, 1}
    assert set(m.nondominated_indices(tied, relative_tolerance=1e-9)) == {0}
    with pytest.raises(m.ModelError):
        m.nondominated_indices(tied, relative_tolerance=-1.0)


def test_hypervolume_matches_known_values_and_pymoo() -> None:
    assert m.hypervolume([(1.0, 1.0, 1.0, 1.0)]) == 1.0
    assert m.hypervolume([(1.0, 0.5, 0.5), (0.5, 1.0, 0.5)]) == pytest.approx(0.375)
    assert m.hypervolume([(1.0, 0.5), (0.5, 1.0)]) == pytest.approx(0.75)
    assert m.hypervolume([(1.0, -0.5), (0.0, 1.0)]) == 0.0
    assert m.hypervolume([]) == 0.0
    # Monotone under adding points.
    a = m.hypervolume([(0.5, 0.5, 0.5, 0.5)])
    b = m.hypervolume([(0.5, 0.5, 0.5, 0.5), (0.9, 0.1, 0.1, 0.1)])
    assert b >= a
    hv_module = pytest.importorskip("pymoo.indicators.hv")
    rng = np.random.default_rng(0)
    for dims in (2, 3, 4):
        for count in (5, 20, 60):
            points = rng.random((count, dims))
            reference = hv_module.HV(ref_point=np.zeros(dims))(-points)
            assert math.isclose(m.hypervolume(points.tolist()), float(reference), rel_tol=0, abs_tol=1e-12)
