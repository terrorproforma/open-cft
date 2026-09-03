"""Evaluation chain over the catalogue: closures, L0 domain, CVaR, separability, dominance, hypervolume."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cft_revival.experiment_runtime import canonical_bytes

from experiments.mdo_l0_campaign_v2 import catalogue as c
from experiments.mdo_l0_campaign_v2 import model as m
from experiments.mdo_l0_campaign_v2 import optimizers as opt


@pytest.fixture(scope="module")
def context() -> m.EvaluationContext:
    return m.build_context()


def _least_lossy(context: m.EvaluationContext) -> int:
    return min(range(m.CATALOGUE_SIZE), key=lambda k: context.designs[k].pooled_point_estimate)


def test_closures_are_declared_products_of_the_design_probabilities(context) -> None:
    design = context.designs[0]
    theta = dict(context.nominal[0])
    expected = 1.0
    for k, name in enumerate(c.CUSP_NAMES):
        expected *= 1.0 - (design.cell_wall_hits[k] + 0.5) / 129.0
    assert m.survival(theta, m.CLOSURE_CL1) == pytest.approx(expected, rel=1e-15)
    assert m.survival(theta, m.CLOSURE_CL2) == pytest.approx(1.0 - (design.pooled_wall_hits + 0.5) / 513.0, rel=1e-15)
    theta[c.CUSP_NAMES[0]] = 1.0
    with pytest.raises(m.ModelError):
        m.survival(theta, m.CLOSURE_CL1)
    with pytest.raises(m.ModelError):
        m.survival(theta, "CL-9")


def test_evaluate_design_success_infeasible_and_record_roundtrip(context) -> None:
    best = _least_lossy(context)
    good = m.evaluate_design(best, (300.0, 1.0, 1e-6), context)
    assert good.status == "success" and good.failure_code is None
    assert good.catalogue_index == best and good.case_id == context.designs[best].case_id
    assert good.robust_margin_a > 0.0 and good.nominal_margin_a > 0.0
    assert good.robust_objectives[3] == pytest.approx(300.0)
    assert good.robust_objectives[0] < good.nominal_objectives[0]
    assert good.survival_statistics["minimum"] <= good.survival_statistics["nominal"] <= good.survival_statistics["maximum"]
    assert len(good.sample_result_sha256) == 64
    record = good.to_record()
    assert record["design"]["catalogue_index"] == best and record["closure"] == m.CLOSURE_CL1
    assert record["design"]["design_id"] == m.design_id(best, good.case_id, good.values)
    again = m.evaluate_design(best, (300.0, 1.0, 1e-6), context)
    assert canonical_bytes(again.to_record()) == canonical_bytes(record)
    # a huge flow at the minimum anode current violates the L0 domain for the worst sampled draw
    bad = m.evaluate_design(best, (300.0, 0.1, 2e-6), context)
    assert bad.status == "infeasible" and bad.failure_code == m.INFEASIBLE_CODE
    assert bad.robust_margin_a < 0.0 and bad.robust_objectives is None and bad.sample_result_sha256 is None
    with pytest.raises(m.ModelError):
        m.evaluate_design(96, (300.0, 1.0, 1e-6), context)
    with pytest.raises(m.ModelError):
        m.evaluate_design(0, (100.0, 1.0, 1e-6), context)


def test_cl2_context_changes_survival_but_not_anode_power(context) -> None:
    cl2 = m.EvaluationContext(context.designs, context.sample, context.nominal, closure=m.CLOSURE_CL2)
    k = _least_lossy(context)
    a = m.evaluate_design(k, (400.0, 1.5, 8e-7), context)
    b = m.evaluate_design(k, (400.0, 1.5, 8e-7), cl2)
    assert a.status == b.status == "success"
    assert a.robust_objectives[3] == b.robust_objectives[3] == pytest.approx(600.0)
    assert a.survival_statistics != b.survival_statistics
    assert b.closure == m.CLOSURE_CL2 and b.design_id == a.design_id


def test_separability_within_a_design_ratio_is_operating_point_independent(context) -> None:
    k = _least_lossy(context)
    ratios = []
    for values in ((200.0, 0.5, 4e-7), (450.0, 2.0, 6e-7), (300.0, 1.2, 3e-7)):
        evaluation = m.evaluate_design(k, values, context)
        assert evaluation.status == "success"
        ratios.append([r / n for r, n in zip(evaluation.robust_objectives[:3], evaluation.nominal_objectives[:3], strict=True)])
    for column in range(3):
        column_values = [row[column] for row in ratios]
        assert max(column_values) - min(column_values) <= 1e-9 * max(abs(v) for v in column_values)
    other = m.evaluate_design(0, (300.0, 1.2, 3e-7), context)
    assert other.status == "success"
    assert other.robust_objectives[0] / other.nominal_objectives[0] != pytest.approx(ratios[2][0], rel=1e-6)


def test_normalization_dominance_blockwise_and_hypervolume() -> None:
    reference = [m.REFERENCE_POINT[name] for name in m.OBJECTIVE_NAMES]
    assert m.normalized_objectives(reference) == (0.0, 0.0, 0.0, 0.0)
    assert m.normalized_objectives([0.06, 3000.0, 1.0, 0.0]) == (1.0, 1.0, 1.0, 1.0)
    rng = random.Random(3)
    points = [[rng.random() for _ in range(4)] for _ in range(300)]
    brute = tuple(
        i for i, p in enumerate(points) if not any(m.dominates_maximize(q, p) for q in points) and points.index(p) == i
    )
    assert m.nondominated_indices(points) == brute
    blocks = [list(range(i, min(i + 37, 300))) for i in range(0, 300, 37)]
    assert m.nondominated_indices_blockwise(points, blocks) == brute
    two = [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25], [0.1, 0.1]]
    assert m.hypervolume(two) == pytest.approx(0.5 * 0.5 + 0.25 * 0.25 * 2)
    assert m.hypervolume([[1.0, -1.0, 0.5, 0.5]]) == 0.0
    three = [[rng.random() for _ in range(3)] for _ in range(40)]
    grid = np.linspace(0.0, 1.0, 101)
    front = np.asarray([three[i] for i in m.nondominated_indices(three)])
    cells = np.stack(np.meshgrid(grid[:-1], grid[:-1], grid[:-1], indexing="ij"), -1).reshape(-1, 3) + 0.005
    covered = np.zeros(len(cells), dtype=bool)
    for p in front:
        covered |= (cells <= p).all(axis=1)
    assert m.hypervolume(three) == pytest.approx(covered.mean(), abs=2e-2)


def test_hypervolume_matches_pymoo_exact_indicator_when_available() -> None:
    hv_module = pytest.importorskip("pymoo.indicators.hv")
    rng = np.random.default_rng(11)
    for _ in range(5):
        points = rng.random((30, 4))
        ours = m.hypervolume(points.tolist())
        theirs = float(hv_module.HV(ref_point=np.zeros(4))(-points))
        assert ours == pytest.approx(theirs, rel=1e-9)


def test_unit_mapping_and_shared_initial_design_cover_distinct_designs() -> None:
    assert opt.catalogue_index_from_unit(0.0) == 0 and opt.catalogue_index_from_unit(1.0) == 95
    assert opt.catalogue_index_from_unit(0.5) == 48
    rows = opt.shared_initial_points(101, 32)
    assert rows.shape == (32, 4)
    indices = [opt.unit_to_design(row)[0] for row in rows]
    assert len(set(indices)) == 32
    assert (rows == opt.lhs_points(101, 160, 32)[:32]).all()
    assert (opt.shared_initial_points(101, 32) != opt.shared_initial_points(202, 32)).any()
    index, values = opt.unit_to_design([0.999, 0.0, 1.0, 0.5])
    assert index == 95 and values == (150.0, 2.5, pytest.approx(1.1e-6))
    individual = opt.pymoo_individual(rows[0])
    assert individual[m.CATALOGUE_VARIABLE] == indices[0]
    assert set(individual) == {m.CATALOGUE_VARIABLE, *m.CONTINUOUS_NAMES}
