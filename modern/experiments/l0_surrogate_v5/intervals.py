"""Blind method selection and separate final split-conformal calibration."""

from __future__ import annotations

from math import ceil, floor, fsum, hypot, inf, nextafter, sqrt
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


def _quantile(values: Sequence[float], probability: float, direction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = min(len(ordered), max(1, ceil((len(ordered) + 1) * probability)))
    return nextafter(ordered[rank - 1], direction)


def nearest_training_distances(
    points: Sequence[Sequence[float]],
    training: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    return tuple(
        min(
            hypot(*(a - b for a, b in zip(point, train, strict=True)))
            for train in training
        )
        for point in points
    )


def scales_for(
    family: str,
    predictions: Sequence[Prediction],
    distances: Sequence[float],
) -> tuple[float, ...]:
    if family.endswith("absolute"):
        return (1.0,) * len(predictions)
    if family.endswith("raw-gp-sd"):
        positive = [item.standard_deviation for item in predictions if item.variance > 0.0]
        floor_value = min(positive) * 1e-12 if positive else 1e-12
        return tuple(max(item.standard_deviation, floor_value) for item in predictions)
    if family.endswith("input-distance"):
        positive = [value for value in distances if value > 0.0]
        floor_value = min(positive) * 1e-12 if positive else 1e-12
        return tuple(max(value, floor_value) for value in distances)
    raise ValueError(f"unknown interval family {family}")


def fit_parameters(
    family: str,
    truth: Sequence[float],
    predictions: Sequence[Prediction],
    distances: Sequence[float],
    *,
    nominal: float,
) -> dict[str, object]:
    scales = scales_for(family, predictions, distances)
    residuals = tuple(
        (value - prediction.mean) / scale
        for value, prediction, scale in zip(truth, predictions, scales, strict=True)
    )
    if family.startswith("symmetric"):
        return {
            "family": family,
            "quantile": _quantile(tuple(abs(value) for value in residuals), nominal, inf),
        }
    alpha = 1.0 - nominal
    ordered = sorted(residuals)
    lower_rank = max(1, floor((len(ordered) + 1) * alpha / 2.0))
    upper_rank = min(len(ordered), ceil((len(ordered) + 1) * (1.0 - alpha / 2.0)))
    return {
        "family": family,
        "lower": nextafter(ordered[lower_rank - 1], -inf),
        "upper": nextafter(ordered[upper_rank - 1], inf),
    }


def interval(
    parameters: Mapping[str, object],
    prediction: Prediction,
    distance: float,
) -> tuple[float, float]:
    family = str(parameters["family"])
    scale = scales_for(family, (prediction,), (distance,))[0]
    if family.startswith("symmetric"):
        radius = float(parameters["quantile"]) * scale
        return nextafter(prediction.mean - radius, -inf), nextafter(
            prediction.mean + radius, inf
        )
    return (
        nextafter(prediction.mean + float(parameters["lower"]) * scale, -inf),
        nextafter(prediction.mean + float(parameters["upper"]) * scale, inf),
    )


def select_method(
    groups: Mapping[str, Sequence[int]],
    truth_by_index: Mapping[int, float],
    prediction_by_index: Mapping[int, Prediction],
    distance_by_index: Mapping[int, float],
    *,
    nominal: float,
    output_scale: float,
    coverage_bounds: tuple[float, float],
    maximum_group_deviation: float,
) -> dict[str, object]:
    candidates = []
    group_names = tuple(sorted(groups))
    for family in CANDIDATES:
        coverages = []
        widths = []
        for heldout in group_names:
            fit_indices = tuple(
                index
                for group in group_names
                if group != heldout
                for index in groups[group]
            )
            heldout_indices = tuple(groups[heldout])
            parameters = fit_parameters(
                family,
                tuple(truth_by_index[index] for index in fit_indices),
                tuple(prediction_by_index[index] for index in fit_indices),
                tuple(distance_by_index[index] for index in fit_indices),
                nominal=nominal,
            )
            hits = 0
            group_widths = []
            for index in heldout_indices:
                lower, upper = interval(
                    parameters,
                    prediction_by_index[index],
                    distance_by_index[index],
                )
                hits += lower <= truth_by_index[index] <= upper
                group_widths.append(upper - lower)
            coverages.append(hits / len(heldout_indices))
            widths.append(fsum(group_widths) / len(group_widths) / output_scale)
        mean_coverage = fsum(coverages) / len(coverages)
        mean_width = fsum(widths) / len(widths)
        record = {
            "family": family,
            "equal_group_mean_coverage": mean_coverage,
            "equal_group_coverage_standard_deviation": pstdev(coverages),
            "maximum_group_coverage_deviation": max(
                abs(value - nominal) for value in coverages
            ),
            "equal_group_mean_normalized_width": mean_width,
            "group_coverages": dict(zip(group_names, coverages, strict=True)),
            "diagnostic_coverage_passed": (
                coverage_bounds[0] <= mean_coverage <= coverage_bounds[1]
            ),
            "diagnostic_stability_passed": max(
                abs(value - nominal) for value in coverages
            )
            <= maximum_group_deviation,
        }
        record["selection_key"] = [
            abs(mean_coverage - nominal),
            record["equal_group_coverage_standard_deviation"],
            mean_width,
            family,
        ]
        candidates.append(record)
    selected = min(
        candidates,
        key=lambda item: (
            item["selection_key"][0],
            item["selection_key"][1],
            item["selection_key"][2],
            item["selection_key"][3],
        ),
    )
    return {
        "selected_family": selected["family"],
        "selected_diagnostics": selected,
        "candidates": candidates,
        "diagnostic_gates_passed": (
            selected["diagnostic_coverage_passed"]
            and selected["diagnostic_stability_passed"]
        ),
    }
