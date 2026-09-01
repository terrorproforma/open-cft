"""Exact-rank method diagnostics and group-balanced final conformal intervals."""

from __future__ import annotations

from fractions import Fraction
from math import fsum, inf, nextafter
from statistics import pstdev
from typing import Mapping, Sequence

from cft_revival.surrogates import Prediction
from experiments.l0_surrogate_v5.intervals import (
    CANDIDATES,
    fit_parameters,
    interval,
    nearest_training_distances,
    scales_for,
)


def finite_rank(n: int, numerator: int, denominator: int) -> int:
    """Return ceil((n + 1) * p) using integer arithmetic."""
    return min(n, ((n + 1) * numerator + denominator - 1) // denominator)


def select_interval(
    groups: Mapping[str, Sequence[int]],
    truth: Mapping[int, float],
    predictions: Mapping[int, Prediction],
    distances: Mapping[int, float],
    *,
    output_scale: float,
    coverage_bounds: tuple[float, float],
    standard_deviation_maximum: float,
) -> dict[str, object]:
    candidates = []
    names = tuple(sorted(groups))
    for family in CANDIDATES:
        coverages = []
        widths = []
        for heldout in names:
            fit_indices = tuple(
                index for group in names if group != heldout for index in groups[group]
            )
            parameters = fit_parameters(
                family,
                tuple(truth[index] for index in fit_indices),
                tuple(predictions[index] for index in fit_indices),
                tuple(distances[index] for index in fit_indices),
                nominal=0.9,
            )
            hits = 0
            group_widths = []
            for index in groups[heldout]:
                lower, upper = interval(parameters, predictions[index], distances[index])
                hits += lower <= truth[index] <= upper
                group_widths.append(upper - lower)
            coverages.append(hits / len(groups[heldout]))
            widths.append(fsum(group_widths) / len(group_widths) / output_scale)
        mean = fsum(coverages) / len(coverages)
        deviation = pstdev(coverages)
        record = {
            "family": family,
            "equal_group_mean_coverage": mean,
            "equal_group_coverage_standard_deviation": deviation,
            "equal_group_mean_normalized_width": fsum(widths) / len(widths),
            "group_coverages": dict(zip(names, coverages, strict=True)),
            "coverage_gate_passed": coverage_bounds[0] <= mean <= coverage_bounds[1],
            "stability_gate_passed": deviation <= standard_deviation_maximum,
        }
        record["all_gates_passed"] = (
            record["coverage_gate_passed"] and record["stability_gate_passed"]
        )
        record["selection_key"] = [
            0 if record["all_gates_passed"] else 1,
            abs(mean - 0.9),
            deviation,
            record["equal_group_mean_normalized_width"],
            family,
        ]
        candidates.append(record)
    selected = min(candidates, key=lambda item: tuple(item["selection_key"]))
    return {
        "selected_family": selected["family"],
        "selected_diagnostics": selected,
        "candidates": candidates,
        "all_gates_passed": selected["all_gates_passed"],
    }


def _weighted_quantile(
    grouped_values: Mapping[str, Sequence[float]],
    numerator: int,
    denominator: int,
    direction: float,
) -> dict[str, object]:
    group_count = len(grouped_values)
    row_count = sum(len(values) for values in grouped_values.values())
    rank = finite_rank(row_count, numerator, denominator)
    target = Fraction(rank, row_count)
    weighted = sorted(
        (
            float(value),
            Fraction(1, group_count * len(values)),
        )
        for values in grouped_values.values()
        for value in values
    )
    cumulative = Fraction(0)
    selected = weighted[-1][0]
    for value, weight in weighted:
        cumulative += weight
        if cumulative >= target:
            selected = value
            break
    return {
        "value": nextafter(selected, direction),
        "exact_rank": rank,
        "row_count": row_count,
        "target_mass": [target.numerator, target.denominator],
        "group_count": group_count,
        "weighting": "each group mass=1/G; each row within group mass=1/(G*n_g)",
    }


def fit_grouped(
    family: str,
    groups: Mapping[str, Sequence[int]],
    truth: Mapping[int, float],
    predictions: Mapping[int, Prediction],
    distances: Mapping[int, float],
) -> dict[str, object]:
    residuals: dict[str, list[float]] = {}
    for group, indices in groups.items():
        scales = scales_for(
            family,
            tuple(predictions[index] for index in indices),
            tuple(distances[index] for index in indices),
        )
        residuals[group] = [
            (truth[index] - predictions[index].mean) / scale
            for index, scale in zip(indices, scales, strict=True)
        ]
    if family.startswith("symmetric"):
        quantile = _weighted_quantile(
            {group: [abs(value) for value in values] for group, values in residuals.items()},
            9,
            10,
            inf,
        )
        return {
            "family": family,
            "grouped_conformal": True,
            "quantile": quantile["value"],
            "quantile_identity": quantile,
        }
    lower = _weighted_quantile(residuals, 1, 20, -inf)
    upper = _weighted_quantile(residuals, 19, 20, inf)
    return {
        "family": family,
        "grouped_conformal": True,
        "lower": lower["value"],
        "upper": upper["value"],
        "lower_quantile_identity": lower,
        "upper_quantile_identity": upper,
    }
