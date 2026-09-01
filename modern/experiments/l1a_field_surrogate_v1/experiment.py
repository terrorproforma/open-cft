"""Prospective multi-fidelity L1a field-emulation experiment mechanics."""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    SolverConfig,
    solve_problem_warp,
)
from cft_revival.optimization.sampling import initial_designs
from cft_revival.surrogates import ExactGP, PODFieldSurrogate, SurrogateSchema, TwoFidelityAR1
from cft_revival.surrogates.pod import fixed_mesh_hash
from experiments.l1a_geometry_sweep_v2.experiment import (
    VARIABLES,
    build_case,
    extract_qois,
    sample_designs as prior_v2_designs,
)

from .protocol import PROTOCOL, canonical_hash, percentile

QOIS = tuple(PROTOCOL["outputs"]["scalar_qois"])
INPUT_NAMES = tuple(variable.name for variable in VARIABLES)
HIGH_DOMAIN = AxisymmetricDomain(**PROTOCOL["fidelities"]["domain"], **{
    "radial_intervals": PROTOCOL["fidelities"]["high"]["radial_intervals"],
    "axial_intervals": PROTOCOL["fidelities"]["high"]["axial_intervals"],
})
LOW_DOMAIN = AxisymmetricDomain(**PROTOCOL["fidelities"]["domain"], **{
    "radial_intervals": PROTOCOL["fidelities"]["low"]["radial_intervals"],
    "axial_intervals": PROTOCOL["fidelities"]["low"]["axial_intervals"],
})
SOLVER = SolverConfig(**PROTOCOL["fidelities"]["solver"])


def sample_designs() -> tuple[Any, ...]:
    raw = initial_designs(VARIABLES, 256, seed=PROTOCOL["sampling"]["seed"], include_boundary_challenges=False)
    development = raw[:96]
    pool = list(raw[96:])
    normalized = {
        item.design_id: tuple(
            (value - variable.lower) / (variable.upper - variable.lower)
            for value, variable in zip(item.values, VARIABLES, strict=True)
        )
        for item in raw
    }
    boundary = sorted(
        pool,
        key=lambda item: (
            min(min(value, 1.0 - value) for value in normalized[item.design_id]),
            item.design_id,
        ),
    )[:5]
    boundary_ids = {item.design_id for item in boundary}
    remainder = [item for item in pool if item.design_id not in boundary_ids]
    candidate_rows = [normalized[item.design_id] for item in development[:64]]

    def nearest_candidate(item: Any) -> float:
        row = normalized[item.design_id]
        return min(math.dist(row, candidate) for candidate in candidate_rows)

    ood = sorted(remainder, key=lambda item: (-nearest_candidate(item), item.design_id))[:5]
    excluded = boundary_ids | {item.design_id for item in ood}
    interpolation = [item for item in pool if item.design_id not in excluded][:6]
    ordered = development + tuple(interpolation) + tuple(boundary) + tuple(ood)
    if len(ordered) != 112 or len({item.design_id for item in ordered}) != 112:
        raise RuntimeError("fresh role design is not unique")
    prior = {tuple(item.values) for item in prior_v2_designs()}
    if prior.intersection(tuple(item.values) for item in ordered):
        raise RuntimeError("fresh experiment intersects prior L1a-v2 coordinates")
    return ordered


def design_row(design: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in design.values)


def high_indices() -> tuple[int, ...]:
    return tuple(range(32)) + tuple(range(64, 112))


def role_indices(name: str) -> tuple[int, ...]:
    start, stop = PROTOCOL["sampling"]["roles"][name]
    return tuple(range(start, stop))


def assessment_groups() -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(range(*bounds))
        for name, bounds in PROTOCOL["sampling"]["assessment_strata"].items()
    }


def _coarse_problem(case: Any) -> AxisymmetricProblem:
    sources = []
    for source in case.problem.sources:
        centre = 0.5 * (source.r_inner_m + source.r_outer_m)
        half = 0.0008
        sources.append(
            AzimuthalCurrentBand(
                source.name,
                centre - half,
                centre + half,
                source.z_min_m,
                source.z_max_m,
                source.ampere_turns_a,
                source.polarity,
            )
        )
    return AxisymmetricProblem(case.problem.name + "-coarse", LOW_DOMAIN, tuple(sources))


def solve_case(design: Any, index: int, fidelity: str) -> tuple[Any, Any, dict[str, Any]]:
    built = build_case(design, index)
    if built.problem.domain != HIGH_DOMAIN:
        raise RuntimeError("accepted high-fidelity domain changed")
    if fidelity == "low":
        built = replace(built, problem=_coarse_problem(built))
    elif fidelity != "high":
        raise ValueError("fidelity must be low or high")
    field = solve_problem_warp(
        built.problem, device=PROTOCOL["execution"]["device"], config=SOLVER
    )
    qois = extract_qois(built, field)
    return built, field, qois


def field_vector(field: Any) -> np.ndarray:
    return np.concatenate(
        (np.asarray(field.b_r_t, dtype=np.float64).ravel(), np.asarray(field.b_z_t, dtype=np.float64).ravel())
    )


def prolong_low(field: Any) -> np.ndarray:
    def nested(values: np.ndarray) -> np.ndarray:
        nr, nz = values.shape
        output = np.empty((2 * nr - 1, 2 * nz - 1), dtype=np.float64)
        output[0::2, 0::2] = values
        output[1::2, 0::2] = 0.5 * (values[:-1, :] + values[1:, :])
        output[0::2, 1::2] = 0.5 * (values[:, :-1] + values[:, 1:])
        output[1::2, 1::2] = 0.25 * (
            values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:]
        )
        return output

    return np.concatenate(
        (nested(np.asarray(field.b_r_t)).ravel(), nested(np.asarray(field.b_z_t)).ravel())
    )


def mesh_hash() -> str:
    coordinates = [
        (component, i * HIGH_DOMAIN.dr_m, HIGH_DOMAIN.z_min_m + j * HIGH_DOMAIN.dz_m)
        for component in (0.0, 1.0)
        for i in range(HIGH_DOMAIN.shape[0])
        for j in range(HIGH_DOMAIN.shape[1])
    ]
    return fixed_mesh_hash(coordinates)


def topology_signature(vector: Sequence[float]) -> dict[str, Any]:
    nr, nz = HIGH_DOMAIN.shape
    bz = np.asarray(vector, dtype=np.float64)[nr * nz :].reshape((nr, nz))[0]
    scale = max(float(np.max(np.abs(bz))), 1e-300)
    tolerance = max(1e-12, 1e-6 * scale)
    nulls = []
    for j in range(nz - 1):
        left, right = float(bz[j]), float(bz[j + 1])
        if abs(left) <= tolerance:
            nulls.append(HIGH_DOMAIN.z_min_m + j * HIGH_DOMAIN.dz_m)
        elif left * right < 0.0:
            fraction = abs(left) / (abs(left) + abs(right))
            nulls.append(HIGH_DOMAIN.z_min_m + (j + fraction) * HIGH_DOMAIN.dz_m)
    cusps = [
        HIGH_DOMAIN.z_min_m + j * HIGH_DOMAIN.dz_m
        for j in range(1, nz - 1)
        if abs(bz[j]) >= abs(bz[j - 1]) and abs(bz[j]) > abs(bz[j + 1])
    ]
    return {
        "status": "resolved" if not any(abs(bz[j]) <= tolerance for j in range(1, nz - 1)) else "sampled-null",
        "null_positions_m": nulls,
        "cusp_positions_m": cusps,
        "null_count": len(nulls),
        "cusp_count": len(cusps),
    }


def topology_match(predicted: Sequence[float], truth: Sequence[float]) -> dict[str, Any]:
    left, right = topology_signature(predicted), topology_signature(truth)
    tolerance = PROTOCOL["gates"]["topology"]["position_tolerance_m"]

    def positions_match(a: Sequence[float], b: Sequence[float]) -> bool:
        return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))

    passed = (
        left["status"] == right["status"]
        and positions_match(left["null_positions_m"], right["null_positions_m"])
        and positions_match(left["cusp_positions_m"], right["cusp_positions_m"])
    )
    return {"passed": passed, "predicted": left, "truth": right, "tolerance_m": tolerance}


def field_energy(vector: Sequence[float]) -> float:
    nr, nz = HIGH_DOMAIN.shape
    array = np.asarray(vector, dtype=np.float64)
    br, bz = array[: nr * nz].reshape((nr, nz)), array[nr * nz :].reshape((nr, nz))
    radial = np.arange(nr, dtype=np.float64) * HIGH_DOMAIN.dr_m
    rw = np.ones(nr); rw[[0, -1]] = 0.5
    zw = np.ones(nz); zw[[0, -1]] = 0.5
    weighted = np.sum((br * br + bz * bz) * (radial * rw)[:, None] * zw[None, :])
    return float(math.pi * HIGH_DOMAIN.dr_m * HIGH_DOMAIN.dz_m * weighted / 1.2566370614359173e-6)


def numerical_record(index: int, built: Any, field: Any, qois: Mapping[str, Any], fidelity: str) -> dict[str, Any]:
    signed_turns = [(source.polarity, source.ampere_turns_a) for source in built.problem.sources]
    return {
        "index": index,
        "fidelity": fidelity,
        "geometry_sha256": built.geometry_sha256,
        "pairing_sha256": canonical_hash({"geometry_sha256": built.geometry_sha256, "signed_turns": signed_turns}),
        "backend": field.diagnostics.backend,
        "shape": list(built.problem.domain.shape),
        "iterations": field.diagnostics.iterations,
        "relative_residual_l2": field.diagnostics.relative_residual_l2,
        "flux_identity_t_per_m": field.diagnostics.max_flux_reconstruction_identity_t_per_m,
        "boundary_to_peak_ratio": qois["boundary_to_peak_ratio"],
        "source_representation_error": qois["source_representation_error"],
        "topology_confidence": qois["topology_confidence"],
    }


def fit_scalar_family(
    family: str,
    budget: int,
    rows: Sequence[Sequence[float]],
    low_qois: Mapping[int, Mapping[str, float]],
    high_qois: Mapping[int, Mapping[str, float]],
) -> tuple[Any, ...]:
    train = tuple(range(budget))
    schema_for = lambda name: SurrogateSchema(INPUT_NAMES, (name,))
    if family == "high_only_ard_gp":
        return tuple(
            ExactGP.fit([rows[i] for i in train], [high_qois[i][name] for i in train], schema=schema_for(name))
            for name in QOIS
        )
    if family == "ar1_low_plus_discrepancy_gp":
        return tuple(
            TwoFidelityAR1.fit(
                [rows[i] for i in role_indices("candidate")],
                [low_qois[i][name] for i in role_indices("candidate")],
                [rows[i] for i in train],
                [high_qois[i][name] for i in train],
                schema=schema_for(name),
            )
            for name in QOIS
        )
    if family == "coarse_observed_plus_discrepancy_gp":
        return tuple(
            ExactGP.fit(
                [rows[i] for i in train],
                [high_qois[i][name] - low_qois[i][name] for i in train],
                schema=SurrogateSchema(INPUT_NAMES, (name + "__high_minus_observed_coarse",)),
            )
            for name in QOIS
        )
    raise ValueError(f"unknown scalar family {family}")


def predict_scalars(
    family: str,
    models: Sequence[Any],
    indices: Sequence[int],
    rows: Sequence[Sequence[float]],
    low_qois: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for index in indices:
        values = {}
        for name, model in zip(QOIS, models, strict=True):
            if family == "coarse_observed_plus_discrepancy_gp":
                values[name] = low_qois[index][name] + model.predict([rows[index]])[0].mean
            elif family == "ar1_low_plus_discrepancy_gp":
                values[name] = model.predict([rows[index]])[0].mean
            else:
                values[name] = model.predict([rows[index]])[0].mean
        output[index] = values
    return output


def fit_field_family(
    family: str,
    budget: int,
    rows: Sequence[Sequence[float]],
    low_fields: Mapping[int, np.ndarray],
    high_fields: Mapping[int, np.ndarray],
) -> PODFieldSurrogate:
    indices = tuple(range(budget))
    snapshots = [
        high_fields[i] if family == "high_only_pod_gp" else high_fields[i] - low_fields[i]
        for i in indices
    ]
    return PODFieldSurrogate.fit(
        [rows[i] for i in indices],
        snapshots,
        rank=PROTOCOL["outputs"]["pod_requested_rank"],
        mesh_hash=mesh_hash(),
        nominal_probability=PROTOCOL["uncertainty"]["nominal_coverage"],
    )


def predict_fields(
    family: str,
    model: PODFieldSurrogate,
    indices: Sequence[int],
    rows: Sequence[Sequence[float]],
    low_fields: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    predictions = model.predict([rows[i] for i in indices], mesh_hash=mesh_hash())
    return {
        index: np.asarray(prediction.mean_field)
        + (low_fields[index] if family == "coarse_observed_plus_discrepancy_pod_gp" else 0.0)
        for index, prediction in zip(indices, predictions, strict=True)
    }


def model_metrics(
    indices: Sequence[int],
    scalar_predictions: Mapping[int, Mapping[str, float]],
    field_predictions: Mapping[int, np.ndarray],
    high_qois: Mapping[int, Mapping[str, float]],
    high_fields: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    scalar: dict[str, Any] = {}
    for name in QOIS:
        truth = [float(high_qois[i][name]) for i in indices]
        prediction = [float(scalar_predictions[i][name]) for i in indices]
        scale = max(max(truth) - min(truth), max(abs(value) for value in truth) * 1e-12, 1e-15)
        errors = [abs(a - b) / scale for a, b in zip(prediction, truth, strict=True)]
        scalar[name] = {
            "nrmse": math.sqrt(fmean(value * value for value in errors)),
            "worst_range_normalized_error": max(errors),
            "range": scale,
        }
    field_rows = []
    topology_rows = []
    for index in indices:
        predicted, truth = field_predictions[index], high_fields[index]
        relative_l2 = float(np.linalg.norm(predicted - truth) / max(np.linalg.norm(truth), 1e-300))
        energy_error = abs(field_energy(predicted) - field_energy(truth)) / max(abs(field_energy(truth)), 1e-300)
        match = topology_match(predicted, truth)
        field_rows.append({"index": index, "relative_l2": relative_l2, "relative_energy_error": energy_error})
        topology_rows.append({"index": index, **match})
    gates = PROTOCOL["gates"]
    passed = (
        all(item["nrmse"] <= gates["scalar"]["nrmse_max"] for item in scalar.values())
        and all(item["worst_range_normalized_error"] <= gates["scalar"]["worst_range_normalized_error_max"] for item in scalar.values())
        and max(item["relative_l2"] for item in field_rows) <= gates["field"]["relative_l2_max"]
        and max(item["relative_energy_error"] for item in field_rows) <= gates["field"]["relative_energy_error_max"]
        and all(item["passed"] for item in topology_rows)
    )
    return {
        "scalar": scalar,
        "field": field_rows,
        "topology": topology_rows,
        "worst_scalar_nrmse": max(item["nrmse"] for item in scalar.values()),
        "worst_scalar_error": max(item["worst_range_normalized_error"] for item in scalar.values()),
        "worst_field_relative_l2": max(item["relative_l2"] for item in field_rows),
        "worst_field_energy_error": max(item["relative_energy_error"] for item in field_rows),
        "all_gates_passed": passed,
    }


def exact_rank(count: int, probability: float) -> int:
    return min(count, math.ceil((count + 1) * probability))


def calibrate(
    predictions: Mapping[int, Mapping[str, float]],
    truth: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    groups = {
        "interpolation": tuple(range(80, 86)),
        "boundary": tuple(range(86, 91)),
        "ood": tuple(range(91, 96)),
    }
    result: dict[str, Any] = {}
    probability = PROTOCOL["uncertainty"]["nominal_coverage"]
    for group, indices in groups.items():
        result[group] = {}
        for name in QOIS:
            residuals = sorted(abs(predictions[i][name] - truth[i][name]) for i in indices)
            rank = exact_rank(len(residuals), probability)
            result[group][name] = {"count": len(residuals), "exact_rank": rank, "radius": residuals[rank - 1]}
    return result


def coverage_metrics(
    predictions: Mapping[int, Mapping[str, float]],
    truth: Mapping[int, Mapping[str, float]],
    calibration: Mapping[str, Any],
    ranges: Mapping[str, float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups = assessment_groups()
    for group, indices in groups.items():
        hits = 0
        widths = []
        total = len(indices) * len(QOIS)
        for index in indices:
            for name in QOIS:
                radius = calibration[group][name]["radius"]
                hits += int(abs(predictions[index][name] - truth[index][name]) <= radius)
                widths.append(2.0 * radius / ranges[name])
        coverage = hits / total
        output[group] = {
            "rows": len(indices),
            "scalar_intervals": total,
            "coverage": coverage,
            "median_normalized_width": percentile(widths, 0.5),
            "p95_normalized_width": percentile(widths, 0.95),
            "passed": coverage >= 0.6 and percentile(widths, 0.5) <= 0.30 and percentile(widths, 0.95) <= 0.60,
        }
    return output
