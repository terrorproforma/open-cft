"""Production geometry, field transfer and audited surrogate paths for v8."""

from __future__ import annotations

import json
import hashlib
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
    prior.update(_prior_role_rows(5))
    prior.update(_prior_role_rows(6))
    prior.update(_prior_role_rows(7))
    if prior.intersection(tuple(item.values) for item in designs):
        raise RuntimeError("v8 coordinates overlap v1-v7 evidence")
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
        case_id=f"l1a-fs-v8-{raw_index:04d}",
        config_sha256=config_hash,
        case_sha256=case_hash,
    )
    return case, {
        **record,
        "v8_case_id": case.case_id,
        "v8_config_sha256": config_hash,
        "v8_case_sha256": case_hash,
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
    by_stage_polarity: dict[tuple[int, int], list[int]] = {}
    for index, case in valid.items():
        key = (len(case.geometry.stages), int(case.problem.sources[0].polarity))
        by_stage_polarity.setdefault(key, []).append(index)
    cells: dict[tuple[int, str, int], list[int]] = {}
    for (stage, polarity), indices in sorted(by_stage_polarity.items()):
        ordered = sorted(
            indices,
            key=lambda index: (
                np.linalg.norm(np.asarray(_normalized(valid[index])) - 0.5),
                valid[index].design.design_id,
            ),
        )
        for stratum, values in zip(
            ("interpolation", "boundary", "ood"),
            np.array_split(np.asarray(ordered, dtype=int), 3),
            strict=True,
        ):
            pool = [int(value) for value in values]
            selected: list[int] = []
            while pool:
                choice = max(
                    pool,
                    key=lambda index: (
                        min(
                            (
                                math.dist(
                                    _normalized(valid[index]),
                                    _normalized(valid[prior]),
                                )
                                for prior in selected
                            ),
                            default=math.dist(
                                _normalized(valid[index]),
                                (0.5,) * len(VARIABLES),
                            ),
                        ),
                        valid[index].design.design_id,
                    ),
                )
                selected.append(choice)
                pool.remove(choice)
            cells[(stage, stratum, polarity)] = selected
    if len(cells) != 18 or any(len(values) < 24 for values in cells.values()):
        raise RuntimeError("insufficient balanced stage/stratum/polarity cells")
    cell_order = sorted(cells)
    candidate = [
        cells[cell][round_index]
        for round_index in range(15)
        for cell in (
            cell_order if round_index % 2 == 0 else tuple(reversed(cell_order))
        )
    ]
    later: list[int] = []
    for role_offset in range(3):
        for stratum in ("interpolation", "boundary", "ood"):
            for stage in (3, 4, 5):
                for polarity in (-1, 1):
                    cell = (stage, stratum, polarity)
                    later.extend(
                        cells[cell][15 + 3 * role_offset : 18 + 3 * role_offset]
                    )
    frozen = tuple(candidate + later)
    if len(frozen) != 432 or len(set(frozen)) != 432:
        raise RuntimeError("unable to freeze 432 balanced unique rows")
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


def input_source_representation_error(case: Any) -> float:
    diagnostics = source_discretization_diagnostics(_fake_problem(case, HIGH_DOMAIN))
    errors: list[float] = []
    for source, item in zip(case.problem.sources, diagnostics, strict=True):
        thickness = min(
            source.r_outer_m - source.r_inner_m,
            source.z_max_m - source.z_min_m,
        )
        errors.extend(
            (
                abs(float(item["area_error_m2"]))
                / max(float(item["requested_area_m2"]), 1e-300),
                abs(float(item["ampere_turn_error_a"]))
                / max(abs(float(item["requested_signed_ampere_turns_a"])), 1e-300),
                math.hypot(
                    float(item["centroid_r_error_m"]),
                    float(item["centroid_z_error_m"]),
                )
                / thickness,
            )
        )
    return max(errors)


def reconstructed_qois(
    case: Any,
    vector: Sequence[float],
    source_error: float,
) -> dict[str, float]:
    br, bz = _arrays(vector)
    z = np.linspace(HIGH_DOMAIN.z_min_m, HIGH_DOMAIN.z_max_m, HIGH_DOMAIN.shape[1])
    nearest = lambda value: int(np.argmin(np.abs(z - value)))
    stages = [nearest(stage.center_z_m) for stage in case.geometry.stages]
    stage_values = [float(bz[0, index]) for index in stages]
    gradients = [
        float(bz[0, min(index + 1, len(z) - 1)] - bz[0, max(index - 1, 0)])
        / (z[min(index + 1, len(z) - 1)] - z[max(index - 1, 0)])
        for index in stages
    ]
    mirrors = []
    for left, right, a, b in zip(
        case.geometry.stages[:-1],
        case.geometry.stages[1:],
        stage_values[:-1],
        stage_values[1:],
        strict=True,
    ):
        middle = nearest(0.5 * (left.center_z_m + right.center_z_m))
        mirrors.append(max(abs(a), abs(b)) / max(abs(float(bz[0, middle])), 1e-12))
    magnitude = np.hypot(br, bz)
    peak = float(np.max(magnitude))
    boundary = max(
        float(np.max(magnitude[-1, :])),
        float(np.max(magnitude[:, 0])),
        float(np.max(magnitude[:, -1])),
    ) / max(peak, 1e-300)
    return {
        "centreline_mid_abs_bz_t": abs(
            float(bz[0, nearest(0.5 * case.geometry.chamber.length_m)])
        ),
        "centreline_abs_bz_peak_t": float(np.max(np.abs(bz[0]))),
        "minimum_mirror_ratio": min(mirrors),
        "maximum_mirror_ratio": max(mirrors),
        "stage_gradient_rms_t_per_m": math.sqrt(
            sum(value * value for value in gradients) / len(gradients)
        ),
        "stage_gradient_max_abs_t_per_m": max(abs(value) for value in gradients),
        "field_energy_j": field_energy(vector),
        "boundary_to_peak_ratio": boundary,
        "source_representation_error": float(source_error),
    }


def qoi_transform(name: str, value: float) -> float:
    if name in {"minimum_mirror_ratio", "maximum_mirror_ratio"}:
        scale = float(PROTOCOL["models"]["mirror_transform"]["scale"])
        return math.log1p((max(float(value), 1.0) - 1.0) / scale)
    return math.log(max(float(value), 1e-15))


def qoi_inverse(name: str, value: float) -> float:
    if name in {"minimum_mirror_ratio", "maximum_mirror_ratio"}:
        scale = float(PROTOCOL["models"]["mirror_transform"]["scale"])
        return 1.0 + scale * math.expm1(float(value))
    return math.exp(float(value))


def model_features(
    row: Sequence[float],
    coarse: Mapping[str, float],
    qoi_name: str | None,
) -> np.ndarray:
    values = [float(item) for item in row]
    if qoi_name is None:
        return np.asarray(values)
    values.append(qoi_transform(qoi_name, coarse[qoi_name]))
    return np.asarray(values)


class SharedKernelGP:
    """Standardized-output ARD Mahalanobis MatÃƒÂ©rn-5/2 regressor."""

    def __init__(
        self,
        train_x: np.ndarray,
        alpha: np.ndarray,
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        ard_length: np.ndarray,
        y_mean: np.ndarray,
        y_scale: np.ndarray,
        length: float,
        output_ridge: np.ndarray,
    ):
        self.train_x = train_x
        self.alpha = alpha
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.ard_length = ard_length
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.length = length
        self.output_ridge = output_ridge

    @staticmethod
    def kernel(
        left: np.ndarray,
        right: np.ndarray,
        ard_length: np.ndarray,
    ) -> np.ndarray:
        delta = (left[:, None, :] - right[None, :, :]) / ard_length
        radius = np.sqrt(np.sum(delta * delta, axis=2))
        scaled = math.sqrt(5.0) * radius
        return (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, length: float) -> "SharedKernelGP":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        x_mean = np.mean(x, axis=0)
        x_scale = np.maximum(np.std(x, axis=0), 1e-12)
        normalized_x = (x - x_mean) / x_scale
        y_mean = np.mean(y, axis=0)
        y_scale = np.maximum(np.std(y, axis=0), 1e-12)
        normalized_y = (y - y_mean) / y_scale
        grid = [float(value) for value in PROTOCOL["models"]["ard_candidate_lengths"]]
        stage = np.digitize(x[:, 0], np.quantile(x[:, 0], (1 / 3, 2 / 3)))
        polarity_column = x[:, 10] if x.shape[1] > 10 else np.zeros(len(x))
        polarity = (polarity_column >= np.median(polarity_column)).astype(int)
        folds = (2 * stage + polarity) % 3
        cv_scores: dict[float, float] = {}
        for candidate in grid:
            scores = []
            for fold in range(3):
                train, test = folds != fold, folds == fold
                if not np.any(test):
                    continue
                train_x, test_x = normalized_x[train], normalized_x[test]
                kernel = cls.kernel(
                    train_x, train_x, np.full(x.shape[1], candidate)
                )
                kernel.flat[:: len(kernel) + 1] += PROTOCOL["models"]["ridge"]
                alpha = np.linalg.solve(kernel, normalized_y[train])
                prediction = cls.kernel(
                    test_x, train_x, np.full(x.shape[1], candidate)
                ) @ alpha
                scores.append(float(np.mean((prediction - normalized_y[test]) ** 2)))
            cv_scores[candidate] = float(np.mean(scores))
        base = min(grid, key=lambda candidate: (cv_scores[candidate], candidate))
        subset = np.linspace(
            0, len(normalized_x) - 1, min(64, len(normalized_x)), dtype=int
        )
        sx, sy = normalized_x[subset], normalized_y[subset]
        ard_length = np.full(x.shape[1], base)
        for column in range(x.shape[1]):
            choices = []
            for multiplier in (0.5, 1.0, 2.0):
                trial = ard_length.copy()
                trial[column] = base * multiplier
                kernel = cls.kernel(sx, sx, trial)
                kernel.flat[:: len(kernel) + 1] += PROTOCOL["models"]["ridge"]
                sign, logdet = np.linalg.slogdet(kernel)
                alpha = np.linalg.solve(kernel, sy)
                nll = (
                    0.5 * float(np.sum(sy * alpha))
                    + 0.5 * sy.shape[1] * float(logdet)
                    if sign > 0
                    else math.inf
                )
                choices.append((nll, trial[column]))
            ard_length[column] = min(choices)[1]
        kernel = cls.kernel(normalized_x, normalized_x, ard_length)
        output_ridge = PROTOCOL["models"]["ridge"] * (
            1.0 + np.arange(normalized_y.shape[1]) / max(normalized_y.shape[1], 1)
        )
        alpha = np.column_stack(
            [
                np.linalg.solve(
                    kernel + np.eye(len(kernel)) * output_ridge[output],
                    normalized_y[:, output],
                )
                for output in range(normalized_y.shape[1])
            ]
        )
        return cls(
            normalized_x,
            alpha,
            x_mean,
            x_scale,
            ard_length,
            y_mean,
            y_scale,
            float(base),
            output_ridge,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(x, dtype=float) - self.x_mean) / self.x_scale
        prediction = self.kernel(normalized, self.train_x, self.ard_length) @ self.alpha
        return prediction * self.y_scale + self.y_mean

    def to_dict(self) -> dict[str, Any]:
        value = {
            "train_x": self.train_x.tolist(),
            "alpha": self.alpha.tolist(),
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "ard_length": self.ard_length.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_scale": self.y_scale.tolist(),
            "length": self.length,
            "ridge": PROTOCOL["models"]["ridge"],
            "output_ridge": self.output_ridge.tolist(),
            "kernel": "mahalanobis-matern-5/2",
        }
        value["model_hash"] = canonical_hash(value)
        return value


def _alignment_landmarks(case: Any) -> tuple[np.ndarray, np.ndarray]:
    physical = sorted(
        {
            HIGH_DOMAIN.z_min_m,
            0.0,
            *(float(stage.center_z_m) for stage in case.geometry.stages),
            float(case.geometry.chamber.length_m),
            HIGH_DOMAIN.z_max_m,
        }
    )
    if physical[0] != HIGH_DOMAIN.z_min_m or physical[-1] != HIGH_DOMAIN.z_max_m:
        raise ValueError("alignment landmarks do not span the field domain")
    return np.asarray(physical), np.linspace(0.0, 1.0, len(physical))


def _polarity(case: Any) -> float:
    return float(case.problem.sources[0].polarity)


def align_vector(vector: Sequence[float], case: Any) -> np.ndarray:
    br, bz = _arrays(vector)
    physical_grid = np.linspace(
        HIGH_DOMAIN.z_min_m, HIGH_DOMAIN.z_max_m, HIGH_DOMAIN.shape[1]
    )
    physical_landmarks, canonical_landmarks = _alignment_landmarks(case)
    canonical_grid = np.linspace(0.0, 1.0, HIGH_DOMAIN.shape[1])
    source = np.interp(canonical_grid, canonical_landmarks, physical_landmarks)
    sign = _polarity(case)
    return sign * np.concatenate(
        (
            np.vstack([np.interp(source, physical_grid, row) for row in br]).ravel(),
            np.vstack([np.interp(source, physical_grid, row) for row in bz]).ravel(),
        )
    )


def unalign_vector(vector: Sequence[float], case: Any) -> np.ndarray:
    br, bz = _arrays(vector)
    physical_grid = np.linspace(
        HIGH_DOMAIN.z_min_m, HIGH_DOMAIN.z_max_m, HIGH_DOMAIN.shape[1]
    )
    physical_landmarks, canonical_landmarks = _alignment_landmarks(case)
    target = np.interp(physical_grid, physical_landmarks, canonical_landmarks)
    canonical_grid = np.linspace(0.0, 1.0, HIGH_DOMAIN.shape[1])
    sign = _polarity(case)
    return sign * np.concatenate(
        (
            np.vstack([np.interp(target, canonical_grid, row) for row in br]).ravel(),
            np.vstack([np.interp(target, canonical_grid, row) for row in bz]).ravel(),
        )
    )


def cylindrical_weights() -> np.ndarray:
    radial = np.maximum(
        np.arange(HIGH_DOMAIN.shape[0]) * HIGH_DOMAIN.dr_m,
        0.25 * HIGH_DOMAIN.dr_m,
    )
    rw = np.ones(HIGH_DOMAIN.shape[0])
    rw[[0, -1]] = 0.5
    zw = np.ones(HIGH_DOMAIN.shape[1])
    zw[[0, -1]] = 0.5
    weights = (radial * rw)[:, None] * zw[None, :]
    return np.concatenate((weights.ravel(), weights.ravel()))


class WeightedPOD:
    """Stage-local joint window/energy/axis-Bz residual basis."""

    def __init__(
        self,
        mean: np.ndarray,
        modes: np.ndarray,
        weights: np.ndarray,
        retained: float,
        rank: int,
        retained_target: float,
        rank_cap: int,
        stage_count: int,
        singular_values: np.ndarray,
    ):
        self.mean = mean
        self.modes = modes
        self.weights = weights
        self.retained = retained
        self.rank = rank
        self.retained_target = retained_target
        self.rank_cap = rank_cap
        self.stage_count = stage_count
        self.singular_values = singular_values

    @staticmethod
    def _parts(stage_count: int) -> tuple[list[np.ndarray], np.ndarray]:
        nr, nz = HIGH_DOMAIN.shape
        windows = []
        for z_values in np.array_split(np.arange(nz), stage_count + 2):
            br = np.asarray([r * nz + z for r in range(nr) for z in z_values])
            bz = nr * nz + br
            windows.append(np.concatenate((br, bz)))
        axis = nr * nz + np.arange(nz)
        return windows, axis

    @classmethod
    def _augment(cls, snapshots: np.ndarray, stage_count: int) -> np.ndarray:
        values = np.asarray(snapshots, dtype=float)
        weights = cylindrical_weights()
        energy_scale = math.sqrt(float(np.sum(weights)))
        channels = [values * np.sqrt(weights)[None, :] / energy_scale]
        windows, axis = cls._parts(stage_count)
        channels.extend(
            values[:, indices] / math.sqrt(len(indices)) for indices in windows
        )
        channels.append(2.0 * values[:, axis] / math.sqrt(len(axis)))
        return np.concatenate(channels, axis=1)

    @classmethod
    def _decode(cls, augmented: np.ndarray, stage_count: int) -> np.ndarray:
        nr, nz = HIGH_DOMAIN.shape
        size = 2 * nr * nz
        weights = cylindrical_weights()
        energy_scale = math.sqrt(float(np.sum(weights)))
        cursor = size
        energy = (
            augmented[:, :size]
            * energy_scale
            / np.sqrt(weights)[None, :]
        )
        windowed = np.zeros_like(energy)
        windows, axis = cls._parts(stage_count)
        for indices in windows:
            width = len(indices)
            windowed[:, indices] = (
                augmented[:, cursor : cursor + width] * math.sqrt(width)
            )
            cursor += width
        axis_values = (
            augmented[:, cursor : cursor + len(axis)]
            * math.sqrt(len(axis))
            / 2.0
        )
        output = 0.5 * (energy + windowed)
        output[:, axis] = (
            energy[:, axis] + windowed[:, axis] + 2.0 * axis_values
        ) / 4.0
        return output

    @classmethod
    def fit(
        cls,
        snapshots: np.ndarray,
        retained_target: float | None = None,
        rank_cap: int | None = None,
        stage_count: int | None = None,
    ) -> "WeightedPOD | None":
        target = float(
            PROTOCOL["models"]["pod_retained_energy_min"]
            if retained_target is None
            else retained_target
        )
        cap = int(
            max(PROTOCOL["models"]["pod_rank_caps"].values())
            if rank_cap is None
            else rank_cap
        )
        stages = 3 if stage_count is None else int(stage_count)
        augmented = cls._augment(np.asarray(snapshots), stages)
        mean = np.mean(augmented, axis=0)
        _, singular, vt = np.linalg.svd(augmented - mean, full_matrices=False)
        energy = np.cumsum(singular * singular) / max(
            float(np.sum(singular * singular)), 1e-300
        )
        matches = np.flatnonzero(energy >= target)
        if not len(matches) or int(matches[0]) + 1 > cap:
            return None
        rank = int(matches[0]) + 1
        return cls(
            mean,
            vt[:rank],
            cylindrical_weights(),
            float(energy[rank - 1]),
            rank,
            target,
            cap,
            stages,
            singular[:rank],
        )

    def project(self, snapshots: np.ndarray) -> np.ndarray:
        return (self._augment(np.asarray(snapshots), self.stage_count) - self.mean) @ self.modes.T

    def observed_coefficients(self, snapshots: np.ndarray) -> np.ndarray:
        return self._augment(np.asarray(snapshots), self.stage_count) @ self.modes.T

    def whiten(self, coefficients: np.ndarray) -> np.ndarray:
        return coefficients / np.maximum(self.singular_values, 1e-12)

    def unwhiten(self, coefficients: np.ndarray) -> np.ndarray:
        return coefficients * self.singular_values

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        augmented = self.mean + np.asarray(coefficients) @ self.modes
        return self._decode(augmented, self.stage_count)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "mean_shape": list(self.mean.shape),
            "mean_sha256": hashlib.sha256(
                np.ascontiguousarray(self.mean, dtype="<f8").tobytes()
            ).hexdigest(),
            "modes_shape": list(self.modes.shape),
            "modes_sha256": hashlib.sha256(
                np.ascontiguousarray(self.modes, dtype="<f8").tobytes()
            ).hexdigest(),
            "singular_values": self.singular_values.tolist(),
            "numeric_encoding": "little-endian-float64",
            "objective_channels": [
                "cylindrical-energy",
                f"{self.stage_count + 2}-axial-windows-unweighted-l2",
                "explicit-axis-bz",
            ],
            "stage_count": self.stage_count,
            "retained": self.retained,
            "retained_target": self.retained_target,
            "rank": self.rank,
            "rank_cap": self.rank_cap,
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
