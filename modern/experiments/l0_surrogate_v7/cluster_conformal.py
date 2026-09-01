"""Group-count-ranked conformal intervals for simultaneous within-group coverage."""

from __future__ import annotations

from fractions import Fraction
from math import fsum, hypot, inf, nextafter
from statistics import pstdev
from typing import Mapping, Sequence

from cft_revival.surrogates import Prediction

CANDIDATES = (
    "symmetric-absolute",
    "asymmetric-signed-absolute",
    "symmetric-raw-gp-sd",
    "asymmetric-signed-raw-gp-sd",
    "symmetric-input-distance",
    "asymmetric-signed-input-distance",
)


def exact_rank(group_count: int, probability: Fraction) -> int:
    """min(G, ceil((G+1)*p)) with no floating probability arithmetic."""
    if group_count < 1 or not Fraction(0) < probability < Fraction(1):
        raise ValueError("invalid exact conformal rank arguments")
    numerator = (group_count + 1) * probability.numerator
    rank = (numerator + probability.denominator - 1) // probability.denominator
    return min(group_count, rank)


def nearest_distances(
    points: Sequence[Sequence[float]],
    training: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    return tuple(
        min(
            hypot(*(left - right for left, right in zip(point, train, strict=True)))
            for train in training
        )
        for point in points
    )


def _scale(family: str, prediction: Prediction, distance: float) -> float:
    if family.endswith("absolute"):
        return 1.0
    if family.endswith("raw-gp-sd"):
        return max(prediction.standard_deviation, 1e-300)
    if family.endswith("input-distance"):
        return max(distance, 1e-300)
    raise ValueError(f"unknown family {family}")


def _order(
    values: Sequence[float],
    probability: Fraction,
    direction: float,
) -> tuple[float, dict[str, object]]:
    rank = exact_rank(len(values), probability)
    selected = sorted(float(value) for value in values)[rank - 1]
    return nextafter(selected, direction), {
        "exchangeability_unit": "group",
        "independent_group_count": len(values),
        "probability": [probability.numerator, probability.denominator],
        "rank": rank,
        "formula": "min(G, ceil((G+1)*p))",
    }


def fit_cluster(
    family: str,
    groups: Mapping[str, Sequence[int]],
    truth: Mapping[int, float],
    predictions: Mapping[int, Prediction],
    distances: Mapping[int, float],
) -> dict[str, object]:
    residuals = {
        group: tuple(
            (truth[index] - predictions[index].mean)
            / _scale(family, predictions[index], distances[index])
            for index in indices
        )
        for group, indices in groups.items()
    }
    if family.startswith("symmetric"):
        scores = tuple(max(abs(value) for value in values) for values in residuals.values())
        quantile, identity = _order(scores, Fraction(9, 10), inf)
        return {
            "family": family,
            "target": "simultaneous coverage of all rows in one future exchangeable group",
            "score": "maximum absolute normalized residual per group",
            "quantile": quantile,
            "rank_identity": identity,
            "group_scores": dict(zip(groups, scores, strict=True)),
        }
    lower_scores = tuple(max(-value for value in values) for values in residuals.values())
    upper_scores = tuple(max(value for value in values) for values in residuals.values())
    lower, lower_identity = _order(lower_scores, Fraction(19, 20), inf)
    upper, upper_identity = _order(upper_scores, Fraction(19, 20), inf)
    return {
        "family": family,
        "target": "simultaneous coverage via two 95% one-sided group bounds and union bound",
        "score": "per-group maximum lower/upper signed normalized residual",
        "lower_radius": lower,
        "upper_radius": upper,
        "lower_rank_identity": lower_identity,
        "upper_rank_identity": upper_identity,
        "lower_group_scores": dict(zip(groups, lower_scores, strict=True)),
        "upper_group_scores": dict(zip(groups, upper_scores, strict=True)),
    }


def interval(
    parameters: Mapping[str, object],
    prediction: Prediction,
    distance: float,
) -> tuple[float, float]:
    family = str(parameters["family"])
    scale = _scale(family, prediction, distance)
    if family.startswith("symmetric"):
        radius = float(parameters["quantile"]) * scale
        return nextafter(prediction.mean - radius, -inf), nextafter(
            prediction.mean + radius, inf
        )
    return (
        nextafter(
            prediction.mean - float(parameters["lower_radius"]) * scale, -inf
        ),
        nextafter(
            prediction.mean + float(parameters["upper_radius"]) * scale, inf
        ),
    )


def select_family(
    groups: Mapping[str, Sequence[int]],
    truth: Mapping[int, float],
    predictions: Mapping[int, Prediction],
    distances: Mapping[int, float],
    *,
    output_scale: float,
    simultaneous_minimum: float,
    row_coverage_minimum: float,
    group_row_standard_deviation_maximum: float,
) -> dict[str, object]:
    candidates = []
    group_names = tuple(sorted(groups))
    for family in CANDIDATES:
        simultaneous = []
        group_row_coverages = []
        widths = []
        for heldout in group_names:
            fit_groups = {
                group: groups[group] for group in group_names if group != heldout
            }
            parameters = fit_cluster(
                family, fit_groups, truth, predictions, distances
            )
            hits = []
            group_widths = []
            for index in groups[heldout]:
                lower, upper = interval(
                    parameters, predictions[index], distances[index]
                )
                hits.append(lower <= truth[index] <= upper)
                group_widths.append((upper - lower) / output_scale)
            simultaneous.append(all(hits))
            group_row_coverages.append(fsum(hits) / len(hits))
            widths.append(fsum(group_widths) / len(group_widths))
        simultaneous_coverage = fsum(simultaneous) / len(simultaneous)
        row_mean = fsum(group_row_coverages) / len(group_row_coverages)
        row_sd = pstdev(group_row_coverages)
        record = {
            "family": family,
            "leave_one_group_out_simultaneous_coverage": simultaneous_coverage,
            "equal_group_mean_row_coverage": row_mean,
            "equal_group_row_coverage_standard_deviation": row_sd,
            "equal_group_mean_normalized_width": fsum(widths) / len(widths),
            "heldout_group_simultaneous_hits": dict(
                zip(group_names, simultaneous, strict=True)
            ),
            "heldout_group_row_coverages": dict(
                zip(group_names, group_row_coverages, strict=True)
            ),
            "simultaneous_gate_passed": simultaneous_coverage
            >= simultaneous_minimum,
            "row_minimum_gate_passed": row_mean >= row_coverage_minimum,
            "stability_gate_passed": row_sd
            <= group_row_standard_deviation_maximum,
        }
        record["all_gates_passed"] = all(
            record[key]
            for key in (
                "simultaneous_gate_passed",
                "row_minimum_gate_passed",
                "stability_gate_passed",
            )
        )
        record["selection_key"] = [
            0 if record["all_gates_passed"] else 1,
            record["equal_group_mean_normalized_width"],
            abs(simultaneous_coverage - 0.9),
            row_sd,
            family,
        ]
        candidates.append(record)
    selected = min(candidates, key=lambda item: tuple(item["selection_key"]))
    return {
        "selected_family": selected["family"],
        "selected_diagnostics": selected,
        "all_gates_passed": selected["all_gates_passed"],
        "candidates": candidates,
    }
