"""Method-role exact cluster validity and engineering width efficiency."""

from __future__ import annotations

from math import ceil, fsum
from statistics import median, pstdev
from typing import Mapping, Sequence

from cft_revival.surrogates import Prediction
from experiments.l0_surrogate_v7.cluster_conformal import (
    CANDIDATES,
    fit_cluster,
    interval,
)


def _p90(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[ceil(9 * len(ordered) / 10) - 1]


def select_efficient_family(
    groups: Mapping[str, Sequence[int]],
    truth: Mapping[int, float],
    predictions: Mapping[int, Prediction],
    distances: Mapping[int, float],
    *,
    output_scale: float,
    simultaneous_minimum: float,
    row_minimum: float,
    stability_maximum: float,
    median_width_maximum: float,
    p90_width_maximum: float,
) -> dict[str, object]:
    names = tuple(sorted(groups))
    candidates = []
    for family in CANDIDATES:
        simultaneous = []
        group_coverages = []
        widths = []
        for heldout in names:
            parameters = fit_cluster(
                family,
                {group: groups[group] for group in names if group != heldout},
                truth,
                predictions,
                distances,
            )
            hits = []
            for index in groups[heldout]:
                lower, upper = interval(
                    parameters, predictions[index], distances[index]
                )
                hits.append(lower <= truth[index] <= upper)
                widths.append((upper - lower) / output_scale)
            simultaneous.append(all(hits))
            group_coverages.append(fsum(hits) / len(hits))
        simultaneous_coverage = fsum(simultaneous) / len(simultaneous)
        row_mean = fsum(group_coverages) / len(group_coverages)
        row_sd = pstdev(group_coverages)
        median_width = median(widths)
        p90_width = _p90(widths)
        record = {
            "family": family,
            "leave_one_group_out_simultaneous_coverage": simultaneous_coverage,
            "equal_group_mean_row_coverage": row_mean,
            "equal_group_row_coverage_standard_deviation": row_sd,
            "normalized_median_full_interval_width": median_width,
            "normalized_p90_full_interval_width": p90_width,
            "simultaneous_gate_passed": simultaneous_coverage
            >= simultaneous_minimum,
            "row_lower_gate_passed": row_mean >= row_minimum,
            "stability_gate_passed": row_sd <= stability_maximum,
            "median_width_gate_passed": median_width <= median_width_maximum,
            "p90_width_gate_passed": p90_width <= p90_width_maximum,
        }
        record["all_gates_passed"] = all(
            record[key]
            for key in (
                "simultaneous_gate_passed",
                "row_lower_gate_passed",
                "stability_gate_passed",
                "median_width_gate_passed",
                "p90_width_gate_passed",
            )
        )
        record["selection_key"] = [
            0 if record["all_gates_passed"] else 1,
            median_width,
            p90_width,
            abs(simultaneous_coverage - 0.9),
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
