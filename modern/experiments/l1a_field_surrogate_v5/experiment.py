"""Production geometry, field transfer, GP and weighted-POD paths for v5."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.fields import AxisymmetricDomain, SolverConfig, solve_problem_warp
from cft_revival.fields.numerics import current_density_grid, source_discretization_diagnostics
from cft_revival.fields.warp_solver import solve_current_density_warp
from cft_revival.optimization.sampling import initial_designs
from experiments.l1a_geometry_sweep_v2.experiment import VARIABLES
from experiments.l1a_field_surrogate_v1.experiment import sample_designs as v1_designs
from experiments.l1a_field_surrogate_v2.experiment import (
    _geometry_attempt as production_geometry_attempt,
    raw_designs as v2_raw_designs,
)

from .protocol import PROTOCOL, PROTOCOL_HASH, REPO, canonical_hash, percentile

QOIS = tuple(PROTOCOL["models"]["qois"])
INPUT_NAMES = tuple(variable.name for variable in VARIABLES)
HIGH_DOMAIN = AxisymmetricDomain(
    **PROTOCOL["fidelities"]["domain"], radial_intervals=80, axial_intervals=144
)
LOW_DOMAIN = AxisymmetricDomain(
    **PROTOCOL["fidelities"]["domain"], radial_intervals=40, axial_intervals=72
)
SOLVER = SolverConfig(**PROTOCOL["fidelities"]["solver"])
MU0 = 1.2566370614359173e-6


def _prior_role_rows(version: int) -> set[tuple[float, ...]]:
    payload = subprocess.run(
        (
            "git",
            "show",
            f"origin/exp/l1a-field-surrogate-v{version}:"
            f"modern/experiments/l1a_field_surrogate_v{version}/partitions.json",
        ),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    value = json.loads(payload)
    return {tuple(float(item) for item in row) for row in value["frozen_design_rows"]}


def raw_designs() -> tuple[Any, ...]:
    designs = initial_designs(
        VARIABLES,
        PROTOCOL["sampling"]["raw_rows"],
        seed=PROTOCOL["sampling"]["seed"],
        include_boundary_challenges=False,
    )
    prior = {tuple(item.values) for item in v1_designs()}
    prior.update(tuple(item.values) for item in v2_raw_designs())
    prior.update(_prior_role_rows(3))
    prior.update(_prior_role_rows(4))
    if prior.intersection(tuple(item.values) for item in designs):
        raise RuntimeError("v5 coordinates overlap v1/v2/v3/v4 evidence")
    return designs


def construct_geometry(design: Any, raw_index: int) -> tuple[Any, dict[str, Any]]:
    case, record = production_geometry_attempt(design, raw_index)
    config_hash = canonical_hash(
        {
            "protocol_hash": PROTOCOL_HASH,
            "design_id": design.design_id,
            "geometry_sha256": case.geometry_sha256,
            "source_sha256": case.source_sha256,
        }
    )
    case_hash = canonical_hash(
        {
            "geometry_sha256": case.geometry_sha256,
            "source_sha256": case.source_sha256,
            "config_sha256": config_hash,
        }
    )
    case = replace(
        case,
        case_id=f"l1a-fs-v5-{raw_index:03d}",
        config_sha256=config_hash,
        case_sha256=case_hash,
    )
    return case, {
        **record,
        "v5_case_id": case.case_id,
        "v5_config_sha256": config_hash,
        "v5_case_sha256": case_hash,
    }


def preflight_candidates() -> tuple[list[dict[str, Any]], dict[int, Any]]:
    records: list[dict[str, Any]] = []
    valid: dict[int, Any] = {}
    for index, design in enumerate(raw_designs()):
        try:
            case, record = construct_geometry(design, index)
            valid[index] = case
            records.append(record)
        except Exception as error:
            records.append(
                {
                    "raw_index": index,
                    "design_id": design.design_id,
                    "valid": False,
                    "rejection_type": type(error).__name__,
                    "rejection_reason": str(error),
                }
            )
    return records, valid


def _normalized(case: Any) -> tuple[float, ...]:
    return tuple(
        (value - variable.lower) / (variable.upper - variable.lower)
        for value, variable in zip(case.design.values, VARIABLES, strict=True)
    )


def select_frozen(valid: Mapping[int, Any]) -> tuple[int, ...]:
    ordered = sorted(valid)
    fixed = ordered[:144]
    pool = ordered[144:]
    candidate = [_normalized(valid[index]) for index in fixed[:128]]

    def take(available: list[int]) -> tuple[list[int], list[int]]:
        boundary = sorted(
            available,
            key=lambda index: (
                min(min(value, 1 - value) for value in _normalized(valid[index])),
                valid[index].design.design_id,
            ),
        )[:16]
        remainder = [index for index in available if index not in boundary]
        ood = sorted(
            remainder,
            key=lambda index: (
                -min(math.dist(_normalized(valid[index]), row) for row in candidate),
                valid[index].design.design_id,
            ),
        )[:16]
        excluded = set(boundary) | set(ood)
        interpolation = [index for index in available if index not in excluded][:16]
        selected = interpolation + boundary + ood
        return selected, [index for index in available if index not in set(selected)]

    calibration, remaining = take(pool)
    assessment, _ = take(remaining)
    frozen = tuple(fixed + calibration + assessment)
    if len(frozen) != 240 or len(set(frozen)) != 240:
        raise RuntimeError("unable to freeze 240 unique valid rows")
    return frozen


def rebuild_frozen(indices: Sequence[int]) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    designs = raw_designs()
    cases, records = {}, []
    for frozen_index, raw_index in enumerate(indices):
        case, record = construct_geometry(designs[raw_index], raw_index)
        cases[frozen_index] = case
        records.append(
            {
                "frozen_index": frozen_index,
                "raw_index": raw_index,
                "geometry_sha256": case.geometry_sha256,
                "source_sha256": case.source_sha256,
                "preview_sha256": record["preview_sha256"],
            }
        )
    return cases, records


def role_indices(role: str) -> tuple[int, ...]:
    return tuple(range(*PROTOCOL["sampling"]["roles"][role]))


def stratum_indices(role: str) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(range(*bounds))
        for name, bounds in PROTOCOL["sampling"][f"{role}_strata"].items()
    }


def _fake_problem(case: Any, domain: AxisymmetricDomain) -> Any:
    return SimpleNamespace(domain=domain, sources=case.problem.sources)


def solve_fidelity(case: Any, fidelity: str) -> Any:
    if fidelity == "high":
        return solve_problem_warp(case.problem, device=PROTOCOL["execution"]["device"], config=SOLVER)
    if fidelity != "low":
        raise ValueError("fidelity must be low or high")
    source = current_density_grid(_fake_problem(case, LOW_DOMAIN))
    return solve_current_density_warp(
        LOW_DOMAIN,
        source,
        permeability_h_per_m=case.problem.permeability_h_per_m,
        device=PROTOCOL["execution"]["device"],
        config=SOLVER,
    )


def field_vector(field: Any) -> np.ndarray:
    return np.concatenate(
        (np.asarray(field.b_r_t, dtype=np.float64).ravel(), np.asarray(field.b_z_t, dtype=np.float64).ravel())
    )


def prolong_low(field: Any) -> np.ndarray:
    def nested(values: np.ndarray) -> np.ndarray:
        result = np.empty((2 * values.shape[0] - 1, 2 * values.shape[1] - 1))
        result[::2, ::2] = values
        result[1::2, ::2] = 0.5 * (values[:-1] + values[1:])
        result[::2, 1::2] = 0.5 * (values[:, :-1] + values[:, 1:])
        result[1::2, 1::2] = 0.25 * (
            values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:]
        )
        return result

    return np.concatenate((nested(np.asarray(field.b_r_t)).ravel(), nested(np.asarray(field.b_z_t)).ravel()))


def _arrays(vector: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    count = HIGH_DOMAIN.shape[0] * HIGH_DOMAIN.shape[1]
    values = np.asarray(vector, dtype=float)
    return values[:count].reshape(HIGH_DOMAIN.shape), values[count:].reshape(HIGH_DOMAIN.shape)


def field_energy(vector: Sequence[float]) -> float:
    br, bz = _arrays(vector)
    radial = np.arange(HIGH_DOMAIN.shape[0]) * HIGH_DOMAIN.dr_m
    rw = np.ones(HIGH_DOMAIN.shape[0]); rw[[0, -1]] = 0.5
    zw = np.ones(HIGH_DOMAIN.shape[1]); zw[[0, -1]] = 0.5
    weighted = np.sum((br * br + bz * bz) * (radial * rw)[:, None] * zw[None, :])
    return float(math.pi * HIGH_DOMAIN.dr_m * HIGH_DOMAIN.dz_m * weighted / MU0)


def topology(vector: Sequence[float]) -> dict[str, Any]:
    axis = _arrays(vector)[1][0]
    tolerance = max(1e-12, 1e-6 * float(np.max(np.abs(axis))))
    nulls = []
    for index in range(len(axis) - 1):
        left, right = float(axis[index]), float(axis[index + 1])
        if abs(left) <= tolerance:
            nulls.append(HIGH_DOMAIN.z_min_m + index * HIGH_DOMAIN.dz_m)
        elif left * right < 0:
            fraction = abs(left) / (abs(left) + abs(right))
            nulls.append(HIGH_DOMAIN.z_min_m + (index + fraction) * HIGH_DOMAIN.dz_m)
    cusps = [
        HIGH_DOMAIN.z_min_m + index * HIGH_DOMAIN.dz_m
        for index in range(1, len(axis) - 1)
        if abs(axis[index]) >= abs(axis[index - 1]) and abs(axis[index]) > abs(axis[index + 1])
    ]
    return {
        "status": "sampled-null" if any(abs(value) <= tolerance for value in axis[1:-1]) else "resolved",
        "nulls": nulls,
        "cusps": cusps,
    }


def topology_match(prediction: Sequence[float], truth: Sequence[float]) -> bool:
    left, right = topology(prediction), topology(truth)
    tolerance = 0.0009027777777777778
    return (
        left["status"] == right["status"]
        and len(left["nulls"]) == len(right["nulls"])
        and len(left["cusps"]) == len(right["cusps"])
        and all(abs(a - b) <= tolerance for a, b in zip(left["nulls"], right["nulls"], strict=True))
        and all(abs(a - b) <= tolerance for a, b in zip(left["cusps"], right["cusps"], strict=True))
    )


def qois(case: Any, field: Any, fidelity: str) -> dict[str, float]:
    axis = np.asarray(field.b_z_t)[0]
    domain = HIGH_DOMAIN if fidelity == "high" else LOW_DOMAIN
    nearest = lambda z: min(range(len(field.z_m)), key=lambda index: abs(field.z_m[index] - z))
    stages = [nearest(stage.center_z_m) for stage in case.geometry.stages]
    stage_values = [float(axis[index]) for index in stages]
    gradients = [
        float(axis[min(index + 1, len(axis) - 1)] - axis[max(index - 1, 0)])
        / (field.z_m[min(index + 1, len(axis) - 1)] - field.z_m[max(index - 1, 0)])
        for index in stages
    ]
    mirrors = []
    for left, right, a, b in zip(
        case.geometry.stages[:-1], case.geometry.stages[1:], stage_values[:-1], stage_values[1:], strict=True
    ):
        middle = nearest(0.5 * (left.center_z_m + right.center_z_m))
        mirrors.append(max(abs(a), abs(b)) / max(abs(float(axis[middle])), 1e-12))
    peak = max(
        math.hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    boundary = max(
        math.hypot(field.b_r_t[i][j], field.b_z_t[i][j])
        for i in range(len(field.r_m))
        for j in range(len(field.z_m))
        if i == len(field.r_m) - 1 or j in (0, len(field.z_m) - 1)
    ) / max(peak, 1e-300)
    diagnostics = source_discretization_diagnostics(_fake_problem(case, domain))
    errors = []
    for source, item in zip(case.problem.sources, diagnostics, strict=True):
        thickness = min(source.r_outer_m - source.r_inner_m, source.z_max_m - source.z_min_m)
        errors.extend(
            (
                abs(float(item["area_error_m2"])) / max(float(item["requested_area_m2"]), 1e-300),
                abs(float(item["ampere_turn_error_a"])) / max(abs(float(item["requested_signed_ampere_turns_a"])), 1e-300),
                math.hypot(float(item["centroid_r_error_m"]), float(item["centroid_z_error_m"])) / thickness,
            )
        )
    vector = field_vector(field) if fidelity == "high" else prolong_low(field)
    return {
        "centreline_mid_abs_bz_t": abs(float(axis[nearest(0.5 * case.geometry.chamber.length_m)])),
        "centreline_abs_bz_peak_t": float(np.max(np.abs(axis))),
        "minimum_mirror_ratio": min(mirrors),
        "maximum_mirror_ratio": max(mirrors),
        "stage_gradient_rms_t_per_m": math.sqrt(sum(value * value for value in gradients) / len(gradients)),
        "stage_gradient_max_abs_t_per_m": max(abs(value) for value in gradients),
        "field_energy_j": field_energy(vector),
        "boundary_to_peak_ratio": boundary,
        "source_representation_error": max(errors),
    }


def model_features(row: Sequence[float], coarse: Mapping[str, float], use_coarse: bool) -> np.ndarray:
    values = list(float(item) for item in row)
    if use_coarse:
        values.extend(math.log(max(coarse[name], 1e-15)) for name in QOIS)
    return np.asarray(values)


class SharedKernelGP:
    def __init__(self, train_x: np.ndarray, alpha: np.ndarray, lower: np.ndarray, span: np.ndarray, length: float):
        self.train_x, self.alpha, self.lower, self.span, self.length = train_x, alpha, lower, span, length

    @staticmethod
    def kernel(left: np.ndarray, right: np.ndarray, length: float) -> np.ndarray:
        radius = np.linalg.norm((left[:, None] - right[None, :]) / length, axis=2)
        scaled = math.sqrt(5) * radius
        return (1 + scaled + scaled * scaled / 3) * np.exp(-scaled)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, length: float) -> "SharedKernelGP":
        lower = np.min(x, axis=0)
        span = np.maximum(np.max(x, axis=0) - lower, 1e-12)
        normalized = (x - lower) / span
        kernel = cls.kernel(normalized, normalized, length)
        kernel.flat[:: len(kernel) + 1] += PROTOCOL["models"]["ridge"]
        return cls(normalized, np.linalg.solve(kernel, y), lower, span, length)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.kernel((x - self.lower) / self.span, self.train_x, self.length) @ self.alpha

    def to_dict(self) -> dict[str, Any]:
        value = {
            "train_x": self.train_x.tolist(),
            "alpha": self.alpha.tolist(),
            "lower": self.lower.tolist(),
            "span": self.span.tolist(),
            "length": self.length,
            "ridge": PROTOCOL["models"]["ridge"],
        }
        value["model_hash"] = canonical_hash(value)
        return value


def align_vector(vector: Sequence[float], case: Any) -> np.ndarray:
    br, bz = _arrays(vector)
    physical = np.linspace(HIGH_DOMAIN.z_min_m, HIGH_DOMAIN.z_max_m, HIGH_DOMAIN.shape[1])
    canonical = np.linspace(-1, 2, HIGH_DOMAIN.shape[1])
    chamber = case.geometry.chamber.length_m
    source = np.where(
        canonical < 0,
        -canonical * HIGH_DOMAIN.z_min_m,
        np.where(canonical <= 1, canonical * chamber, chamber + (canonical - 1) * (HIGH_DOMAIN.z_max_m - chamber)),
    )
    return np.concatenate(
        (
            np.vstack([np.interp(source, physical, row) for row in br]).ravel(),
            np.vstack([np.interp(source, physical, row) for row in bz]).ravel(),
        )
    )


def unalign_vector(vector: Sequence[float], case: Any) -> np.ndarray:
    br, bz = _arrays(vector)
    physical = np.linspace(HIGH_DOMAIN.z_min_m, HIGH_DOMAIN.z_max_m, HIGH_DOMAIN.shape[1])
    chamber = case.geometry.chamber.length_m
    target = np.where(
        physical < 0,
        physical / -HIGH_DOMAIN.z_min_m,
        np.where(physical <= chamber, physical / chamber, 1 + (physical - chamber) / (HIGH_DOMAIN.z_max_m - chamber)),
    )
    canonical = np.linspace(-1, 2, HIGH_DOMAIN.shape[1])
    return np.concatenate(
        (
            np.vstack([np.interp(target, canonical, row) for row in br]).ravel(),
            np.vstack([np.interp(target, canonical, row) for row in bz]).ravel(),
        )
    )


def cylindrical_weights() -> np.ndarray:
    radial = np.maximum(np.arange(HIGH_DOMAIN.shape[0]) * HIGH_DOMAIN.dr_m, 0.25 * HIGH_DOMAIN.dr_m)
    rw = np.ones(HIGH_DOMAIN.shape[0]); rw[[0, -1]] = 0.5
    zw = np.ones(HIGH_DOMAIN.shape[1]); zw[[0, -1]] = 0.5
    weights = (radial * rw)[:, None] * zw[None, :]
    return np.concatenate((weights.ravel(), weights.ravel()))


class WeightedPOD:
    def __init__(self, mean: np.ndarray, modes: np.ndarray, weights: np.ndarray, retained: float, rank: int):
        self.mean, self.modes, self.weights, self.retained, self.rank = mean, modes, weights, retained, rank

    @classmethod
    def fit(cls, snapshots: np.ndarray) -> "WeightedPOD | None":
        weights = cylindrical_weights()
        mean = np.mean(snapshots, axis=0)
        weighted = (snapshots - mean) * np.sqrt(weights)
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        energy = np.cumsum(singular * singular) / max(float(np.sum(singular * singular)), 1e-300)
        matches = np.flatnonzero(energy >= PROTOCOL["models"]["pod_retained_energy_min"])
        if not len(matches) or int(matches[0]) + 1 > PROTOCOL["models"]["pod_rank_cap"]:
            return None
        rank = int(matches[0]) + 1
        return cls(mean, vt[:rank] / np.sqrt(weights), weights, float(energy[rank - 1]), rank)

    def project(self, snapshots: np.ndarray) -> np.ndarray:
        return ((snapshots - self.mean) * self.weights) @ self.modes.T

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.mean + coefficients @ self.modes

    def to_dict(self) -> dict[str, Any]:
        value = {
            "mean": self.mean.tolist(),
            "modes": self.modes.tolist(),
            "retained": self.retained,
            "rank": self.rank,
            "weights_hash": canonical_hash(self.weights.tolist()),
        }
        value["basis_hash"] = canonical_hash(value)
        return value


def metric_summary(
    truth_qois: np.ndarray,
    prediction_qois: np.ndarray,
    truth_fields: Sequence[np.ndarray],
    prediction_fields: Sequence[np.ndarray],
) -> dict[str, Any]:
    scalar = {}
    for column, name in enumerate(QOIS):
        scale = max(float(np.ptp(truth_qois[:, column])), float(np.max(np.abs(truth_qois[:, column]))) * 1e-12, 1e-15)
        errors = np.abs(prediction_qois[:, column] - truth_qois[:, column]) / scale
        scalar[name] = {"nrmse": float(np.sqrt(np.mean(errors * errors))), "worst": float(np.max(errors))}
    fields = [
        {
            "l2": float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-300)),
            "energy": abs(field_energy(prediction) - field_energy(truth)) / max(field_energy(truth), 1e-300),
            "topology": topology_match(prediction, truth),
        }
        for prediction, truth in zip(prediction_fields, truth_fields, strict=True)
    ]
    return {
        "scalar": scalar,
        "worst_scalar_nrmse": max(item["nrmse"] for item in scalar.values()),
        "worst_scalar": max(item["worst"] for item in scalar.values()),
        "worst_field_l2": max(item["l2"] for item in fields),
        "worst_field_energy": max(item["energy"] for item in fields),
        "topology_matches": sum(item["topology"] for item in fields),
        "row_count": len(fields),
    }
