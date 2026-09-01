"""Exact-rank and adversarial efficiency tests for v8."""

from __future__ import annotations

from fractions import Fraction

from cft_revival.surrogates import Prediction
from experiments.l0_surrogate_v7.cluster_conformal import exact_rank, fit_cluster
from experiments.l0_surrogate_v8.efficiency import select_efficient_family


def _prediction(value: float) -> Prediction:
    return Prediction(value, 0.01, 0.9)


def test_exact_group_rank_regressions() -> None:
    assert exact_rank(99, Fraction(9, 10)) == 90
    assert exact_rank(239, Fraction(9, 10)) == 216
    assert exact_rank(40, Fraction(19, 20)) == 39


def test_rank_uses_group_count_not_duplicated_rows() -> None:
    groups = {f"g{group}": [group] for group in range(99)}
    groups["g0"] = list(range(1000, 1100))
    truth = {
        index: float(group + 1)
        for group, indices in enumerate(groups.values())
        for index in indices
    }
    predictions = {index: _prediction(0.0) for index in truth}
    distances = {index: 1.0 for index in truth}
    fitted = fit_cluster(
        "symmetric-absolute", groups, truth, predictions, distances
    )
    assert fitted["rank_identity"]["independent_group_count"] == 99
    assert fitted["rank_identity"]["rank"] == 90


def test_row_upper_coverage_is_not_an_efficiency_gate() -> None:
    groups = {f"g{group}": [group] for group in range(40)}
    truth = {index: 0.0 for index in range(40)}
    predictions = {index: _prediction(0.0) for index in truth}
    distances = {index: 1.0 for index in truth}
    selected = select_efficient_family(
        groups,
        truth,
        predictions,
        distances,
        output_scale=1.0,
        simultaneous_minimum=0.85,
        row_minimum=0.85,
        stability_maximum=0.2,
        median_width_maximum=0.25,
        p90_width_maximum=0.4,
    )
    assert selected["all_gates_passed"]
    assert selected["selected_diagnostics"]["equal_group_mean_row_coverage"] == 1.0
