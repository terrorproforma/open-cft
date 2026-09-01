"""Preregistered raw and physics-informed ARD GP mean-model candidates."""

from __future__ import annotations

from math import log, sqrt
from typing import Mapping, Sequence

from cft_revival.surrogates import ExactGP, Prediction, SurrogateSchema

OUTPUT_NAMES = ("axial_thrust_n", "specific_impulse_s")
OUTPUT_UNITS = ("N", "s")


def physics_features(row: Sequence[float]) -> tuple[float, ...]:
    voltage_v = 150.0 + 350.0 * row[0]
    flow_ratio = 0.1 + 0.9 * row[1]
    utilization = 0.65 + 0.33 * row[2]
    double_share = 0.15 * row[3]
    axial = 0.75 + 0.23 * row[4]
    sqrt_voltage = (sqrt(voltage_v) - sqrt(150.0)) / (
        sqrt(500.0) - sqrt(150.0)
    )
    log_flow = log(flow_ratio / 0.1) / log(10.0)
    charge_factor = sqrt(1.0 + double_share)
    charge_normalized = (charge_factor - 1.0) / (sqrt(1.15) - 1.0)
    thrust_proxy = (
        flow_ratio
        * sqrt(voltage_v / 500.0)
        * utilization
        * axial
        * charge_factor
    )
    isp_proxy = sqrt(voltage_v / 500.0) * axial * charge_factor
    return (
        sqrt_voltage,
        log_flow,
        row[2],
        charge_normalized,
        row[4],
        thrust_proxy,
        isp_proxy,
    )


def transform(
    family: str, rows: Sequence[Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    if family == "raw-ard-matern52":
        return tuple(tuple(float(value) for value in row) for row in rows)
    if family == "physics-informed-v1-ard-matern52":
        return tuple(physics_features(row) for row in rows)
    raise ValueError(f"unknown mean-model family {family}")


def _schema(family: str, output: int) -> SurrogateSchema:
    features = (
        ("voltage", "flow", "ionized", "double_share", "axial")
        if family == "raw-ard-matern52"
        else (
            "sqrt_voltage",
            "log_flow",
            "utilization",
            "charge_factor",
            "axial",
            "thrust_proxy",
            "isp_proxy",
        )
    )
    return SurrogateSchema(
        features,
        (OUTPUT_NAMES[output],),
        ("1",) * len(features),
        (OUTPUT_UNITS[output],),
    )


def fit_models(
    family: str,
    selected: Sequence[int],
    inputs: Sequence[Sequence[float]],
    observed: Mapping[int, Sequence[float]],
) -> tuple[ExactGP, ExactGP]:
    transformed = transform(family, tuple(inputs[index] for index in selected))
    return tuple(  # type: ignore[return-value]
        ExactGP.fit(
            transformed,
            tuple(float(observed[index][output]) for index in selected),
            schema=_schema(family, output),
            length_scale_mode="ard",
            nominal_probability=0.9,
        )
        for output in range(2)
    )


def predict_rows(
    family: str,
    models: Sequence[ExactGP],
    indices: Sequence[int],
    inputs: Sequence[Sequence[float]],
) -> dict[int, tuple[Prediction, Prediction]]:
    points = transform(family, tuple(inputs[index] for index in indices))
    columns = tuple(model.predict(points) for model in models)
    return {
        index: (columns[0][row], columns[1][row])
        for row, index in enumerate(indices)
    }
