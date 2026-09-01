"""Adversarial exact group-rank tests."""

from __future__ import annotations

from fractions import Fraction

from cft_revival.surrogates import Prediction
from experiments.l0_surrogate_v7.cluster_conformal import exact_rank, fit_cluster


def test_exact_symmetric_ranks_n99_and_n239() -> None:
    assert exact_rank(99, Fraction(9, 10)) == 90
    assert exact_rank(239, Fraction(9, 10)) == 216
    assert exact_rank(99, Fraction(19, 20)) == 95


def test_unequal_group_sizes_do_not_become_row_exchangeability_units() -> None:
    groups = {"tiny": (0,), "large": tuple(range(1, 100))}
    truth = {index: (3.0 if index == 0 else 1.0) for index in range(100)}
    predictions = {index: Prediction(0.0, 1.0) for index in truth}
    distances = {index: 1.0 for index in truth}
    fitted = fit_cluster(
        "symmetric-absolute", groups, truth, predictions, distances
    )
    assert fitted["rank_identity"]["independent_group_count"] == 2
    assert fitted["rank_identity"]["rank"] == 2
    assert len(fitted["group_scores"]) == 2


def test_correlated_row_duplication_does_not_change_group_rank_or_score() -> None:
    truth = {0: 2.0, 1: 1.0}
    predictions = {index: Prediction(0.0, 1.0) for index in truth}
    distances = {index: 1.0 for index in truth}
    original = fit_cluster(
        "symmetric-absolute",
        {"a": (0,), "b": (1,)},
        truth,
        predictions,
        distances,
    )
    duplicated = fit_cluster(
        "symmetric-absolute",
        {"a": (0, 0, 0, 0), "b": (1,)},
        truth,
        predictions,
        distances,
    )
    assert original["rank_identity"] == duplicated["rank_identity"]
    assert original["quantile"] == duplicated["quantile"]


def test_asymmetric_ranks_use_groups_and_exact_tail_fraction() -> None:
    groups = {f"g{index}": (index,) for index in range(99)}
    truth = {index: float(index % 5 - 2) for index in range(99)}
    predictions = {index: Prediction(0.0, 1.0) for index in truth}
    distances = {index: 1.0 for index in truth}
    fitted = fit_cluster(
        "asymmetric-signed-absolute", groups, truth, predictions, distances
    )
    assert fitted["lower_rank_identity"]["probability"] == [19, 20]
    assert fitted["upper_rank_identity"]["rank"] == 95
