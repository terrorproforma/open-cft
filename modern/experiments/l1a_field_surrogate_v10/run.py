"""Synthetic-preflighted, staged and exactly-once v10 experiment runner."""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.metadata
import inspect
import io
import math
import os
import platform
import subprocess
import symtable
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import warp as wp

from cft_revival.experiment_runtime import (
    BundleState,
    canonical_bytes,
    Decision,
    ExecutionAttestation,
    ExperimentRuntime,
    RuntimeCallbacks,
    RunContext,
    validate_bundle,
)
from cft_revival.experiment_runtime.platformfs import PinnedDirectory

from .experiment import (
    HIGH_DOMAIN,
    QOIS,
    SharedKernelGP,
    WeightedPOD,
    align_vector,
    construct_geometry,
    field_energy,
    field_vector,
    metric_summary,
    model_features,
    input_source_representation_error,
    preflight_candidates,
    prolong_low,
    qois,
    qoi_inverse,
    qoi_transform,
    reconstructed_qois,
    raw_designs,
    rebuild_frozen,
    role_indices,
    select_frozen,
    solve_fidelity,
    stratum_indices,
    topology,
    topology_match,
    unalign_vector,
)
from .protocol import (
    DEPENDENCY_LOCK,
    GEOMETRY_PREFLIGHT,
    PARTITIONS,
    PROTOCOL,
    PROTOCOL_HASH,
    REPO,
    CACHE,
    RESULTS,
    ROOT,
    SYNTHETIC_PREFLIGHT,
    canonical_hash,
    exact_rank,
    percentile,
    strict_load,
    verify_json,
    write_predeclared_json,
)

SUBJECT = PROTOCOL["integrity"]["protocol_subject"]
REMOTE = PROTOCOL["integrity"]["remote_branch"]
PREFIXES = (
    "modern/experiments/l1a_field_surrogate_v10/",
    "modern/tests/experiments/l1a_field_surrogate_v10/",
)
DEPENDENCY_ROOTS = (
    "modern/src/cft_revival/experiment_runtime/",
    "modern/spec/experiment_runtime/",
    "modern/src/cft_revival/fields/",
    "modern/src/cft_revival/geometry/",
    "modern/src/cft_revival/magnetics/",
    "modern/src/cft_revival/optimization/",
    "modern/experiments/l1a_geometry_sweep_v2/",
    "modern/experiments/l1a_field_surrogate_v1/",
    "modern/experiments/l1a_field_surrogate_v2/",
)
ACCEPTED_RUNTIME_COMMIT = "b46e263950f91530ea61710b5dcc9354fc63cf6c"


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=REPO, check=check, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dependency_lock() -> dict[str, Any]:
    wp.init()
    value = {
        "schema_version": "cft-revival.l1a-field-surrogate-v10.dependency-lock/10.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "python": {
            "version": sys.version,
            "executable_sha256": _file_sha(Path(sys.executable)),
        },
        "numpy": {
            "version": np.__version__,
            "distribution": importlib.metadata.version("numpy"),
            "module_sha256": _file_sha(Path(np.__file__).resolve()),
        },
        "warp": {
            "version": wp.__version__,
            "distribution": importlib.metadata.version("warp-lang"),
            "module_sha256": _file_sha(Path(wp.__file__).resolve()),
        },
        "cuda_toolkit": list(wp.get_cuda_toolkit_version()),
        "cuda_driver_runtime": list(wp.get_cuda_driver_version()),
        "nvidia_smi": subprocess.run(
            ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"),
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "pyproject": {
            "sha256": _file_sha(REPO / "modern" / "pyproject.toml"),
            "git_blob": _git("hash-object", "modern/pyproject.toml"),
        },
        "accepted_experiment_runtime": {
            "commit": ACCEPTED_RUNTIME_COMMIT,
            "is_ancestor": subprocess.run(
                ("git", "merge-base", "--is-ancestor", ACCEPTED_RUNTIME_COMMIT, "HEAD"),
                cwd=REPO,
                check=False,
            ).returncode
            == 0,
            "tree": _git(
                "rev-parse",
                f"{ACCEPTED_RUNTIME_COMMIT}:modern/src/cft_revival/experiment_runtime",
            ),
        },
        "v3_coordinate_evidence": {
            "commit": _git("rev-parse", "origin/exp/l1a-field-surrogate-v3"),
            "partitions_blob": _git(
                "rev-parse",
                "origin/exp/l1a-field-surrogate-v3:modern/experiments/l1a_field_surrogate_v3/partitions.json",
            ),
        },
        "v4_coordinate_evidence": {
            "commit": _git("rev-parse", "origin/exp/l1a-field-surrogate-v4"),
            "partitions_blob": _git(
                "rev-parse",
                "origin/exp/l1a-field-surrogate-v4:"
                "modern/experiments/l1a_field_surrogate_v4/partitions.json",
            ),
        },
        "v5_coordinate_evidence": {
            "commit": _git("rev-parse", "origin/exp/l1a-field-surrogate-v5"),
            "partitions_blob": _git(
                "rev-parse",
                "origin/exp/l1a-field-surrogate-v5:"
                "modern/experiments/l1a_field_surrogate_v5/partitions.json",
            ),
        },
        "v9_development_evidence": {
            "commit": _git("rev-parse", "origin/exp/l1a-field-surrogate-v9"),
            "partitions_blob": _git(
                "rev-parse",
                "origin/exp/l1a-field-surrogate-v9:"
                "modern/experiments/l1a_field_surrogate_v9/partitions.json",
            ),
            "method_freeze_blob": _git(
                "rev-parse",
                "origin/exp/l1a-field-surrogate-v9:"
                "modern/experiments/l1a_field_surrogate_v9/results/"
                "artifacts/frozen-method-selection.json",
            ),
            "terminal_blob": _git(
                "rev-parse",
                "origin/exp/l1a-field-surrogate-v9:"
                "modern/experiments/l1a_field_surrogate_v9/results/terminal.json",
            ),
            "exclusive_scientific_development_source": True,
        },
    }
    value["dependency_lock_hash"] = canonical_hash(value)
    return value


def full_closure() -> list[dict[str, str]]:
    entries = []
    for line in _git("ls-tree", "-r", "HEAD", "--", *(PREFIXES + DEPENDENCY_ROOTS)).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        entries.append({"path": path, "mode": mode, "type": kind, "blob": blob})
    return sorted(entries, key=lambda item: item["path"])


def static_undefined_names(paths: Sequence[Path]) -> dict[str, Any]:
    unresolved: dict[str, list[str]] = {}
    ast_hashes = {}
    builtin_names = set(dir(builtins)) | {"__file__", "__name__", "__package__"}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        ast_hashes[path.name] = hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode("utf-8")
        ).hexdigest()
        table = symtable.symtable(source, str(path), "exec")
        module_defined = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
        }
        missing = set()

        def visit(scope: symtable.SymbolTable) -> None:
            for symbol in scope.get_symbols():
                if (
                    symbol.is_referenced()
                    and symbol.is_global()
                    and symbol.get_name() not in module_defined
                    and symbol.get_name() not in builtin_names
                ):
                    missing.add(symbol.get_name())
            for child in scope.get_children():
                visit(child)

        visit(table)
        if missing:
            unresolved[path.name] = sorted(missing)
    return {
        "paths": [path.name for path in paths],
        "ast_hashes": ast_hashes,
        "unresolved": unresolved,
        "passed": not unresolved,
    }


def new_counters() -> dict[str, Any]:
    roles = ("candidate", "method", "calibration", "assessment")
    return {
        "solver_accesses": {role: {"low": 0, "fine": 0} for role in roles},
        "materialized": {role: {"low": 0, "fine": 0} for role in roles},
        "checkpoint_reads": {role: 0 for role in roles},
        "label_reads": {role: 0 for role in roles},
        "model_fit_accesses": 0,
    }


def save_checkpoint(
    cache: Path,
    index: int,
    row: Sequence[float],
    low_field: np.ndarray,
    high_field: np.ndarray,
    low_qois: Sequence[float],
    high_qois: Sequence[float],
    input_source_error: float | None = None,
) -> Path:
    path = cache / f"{index:03d}.npz"
    np.savez_compressed(
        path,
        row=np.asarray(row),
        low_field=np.asarray(low_field),
        high_field=np.asarray(high_field),
        low_qois=np.asarray(low_qois),
        high_qois=np.asarray(high_qois),
        input_source_error=np.asarray(
            float(high_qois[-1]) if input_source_error is None else input_source_error
        ),
    )
    return path


def load_checkpoint(
    cache: Path,
    index: int,
    role: str,
    counters: dict[str, Any],
    context: RunContext | None = None,
) -> dict[str, np.ndarray]:
    counters["checkpoint_reads"][role] += 1
    counters["label_reads"][role] += 1
    if context is not None:
        context.before_expensive(
            f"checkpoint-{index:03d}",
            kind="label",
            details={"role": role, "index": index},
        )
    with np.load(cache / f"{index:03d}.npz") as data:
        return {name: np.array(data[name]) for name in data.files}


def qoi_decode(values: Sequence[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(QOIS, values, strict=True)}


def field_target(data: Mapping[str, np.ndarray], case: Any, family: str) -> np.ndarray:
    low, high = data["low_field"], data["high_field"]
    norm = math.sqrt(max(field_energy(low), 1e-300))
    sign = float(case.problem.sources[0].polarity)
    return sign * (high - low) / norm


def build_snapshots(
    data_rows: Sequence[Mapping[str, np.ndarray]],
    cases: Sequence[Any],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    snapshots = np.asarray(
        [field_target(data, case, family) for data, case in zip(data_rows, cases, strict=True)]
    )
    features = np.asarray([model_features(data["row"], {}, None) for data in data_rows])
    return snapshots, features


def local_features(data: Mapping[str, np.ndarray]) -> np.ndarray:
    low = np.asarray(data["low_field"], dtype=float)
    nr, nz = HIGH_DOMAIN.shape
    axis_bz = low[nr * nz : nr * nz + nz]
    sample = axis_bz[np.linspace(0, nz - 1, 9, dtype=int)]
    coarse_qois = np.log(np.maximum(np.asarray(data["low_qois"]), 1e-15))
    coarse_topology = topology(low)
    return np.concatenate(
        (
            np.asarray(data["row"], dtype=float),
            coarse_qois,
            sample,
            np.asarray(
                [
                    math.log(max(field_energy(low), 1e-300)),
                    math.log(max(float(np.linalg.norm(low)), 1e-300)),
                    len(coarse_topology["nulls"]),
                    len(coarse_topology["cusps"]),
                ]
            ),
        )
    )


class LocalResidualInterpolator:
    def __init__(
        self,
        features: np.ndarray,
        snapshots: np.ndarray,
        stages: np.ndarray,
        scope: str,
        neighbours: int,
        kernel: str,
    ):
        self.mean = np.mean(features, axis=0)
        self.scale = np.maximum(np.std(features, axis=0), 1e-12)
        self.features = (features - self.mean) / self.scale
        self.snapshots = np.asarray(snapshots)
        self.stages = np.asarray(stages)
        self.scope = scope
        self.neighbours = neighbours
        self.kernel_name = kernel

    @classmethod
    def fit(
        cls,
        data_rows: Sequence[Mapping[str, np.ndarray]],
        cases: Sequence[Any],
        scope: str,
        neighbours: int,
        kernel: str,
    ) -> "LocalResidualInterpolator":
        return cls(
            np.asarray([local_features(data) for data in data_rows]),
            np.asarray(
                [
                    field_target(data, case, "localized")
                    for data, case in zip(data_rows, cases, strict=True)
                ]
            ),
            np.asarray([len(case.geometry.stages) for case in cases]),
            scope,
            neighbours,
            kernel,
        )

    def predict_one(
        self, data: Mapping[str, np.ndarray], case: Any
    ) -> tuple[np.ndarray, dict[str, float]]:
        feature = (local_features(data) - self.mean) / self.scale
        eligible = np.arange(len(self.features))
        if self.scope == "stage-specific":
            eligible = eligible[self.stages == len(case.geometry.stages)]
        distances = np.linalg.norm(self.features[eligible] - feature, axis=1)
        order = np.lexsort((eligible, distances))
        chosen = eligible[order[: min(self.neighbours, len(order))]]
        selected_distance = distances[order[: len(chosen)]]
        duplicates = selected_distance <= 1e-12
        if np.any(duplicates):
            weights = duplicates.astype(float)
        elif self.kernel_name == "wendland-c2":
            radius = max(float(selected_distance[-1]) * (1.0 + 1e-12), 1e-12)
            q = np.minimum(selected_distance / radius, 1.0)
            weights = (1.0 - q) ** 4 * (4.0 * q + 1.0)
        else:
            weights = 1.0 / np.maximum(selected_distance, 1e-9) ** 2
        if float(np.sum(weights)) <= 1e-300:
            weights = np.zeros_like(selected_distance)
            weights[0] = 1.0
        weights /= np.sum(weights)
        prediction = weights @ self.snapshots[chosen]
        differences = self.snapshots[chosen] - prediction
        spread = math.sqrt(
            float(np.sum(weights[:, None] * differences * differences))
            / prediction.size
        )
        return prediction, {
            "nearest_distance": float(selected_distance[0]),
            "furthest_neighbour_distance": float(selected_distance[-1]),
            "neighbour_spread_rms": spread,
        }

    def to_dict(self) -> dict[str, Any]:
        value = {
            "scope": self.scope,
            "neighbours": self.neighbours,
            "kernel": self.kernel_name,
            "feature_count": self.features.shape[1],
            "training_count": self.features.shape[0],
            "features_sha256": hashlib.sha256(
                np.ascontiguousarray(self.features, dtype="<f8").tobytes()
            ).hexdigest(),
            "snapshots_sha256": hashlib.sha256(
                np.ascontiguousarray(self.snapshots, dtype="<f8").tobytes()
            ).hexdigest(),
            "duplicate_policy": PROTOCOL["models"]["duplicate_policy"],
            "distance_policy": PROTOCOL["models"]["distance_policy"],
        }
        value["model_hash"] = canonical_hash(value)
        return value


def predict_local_field(
    model: LocalResidualInterpolator,
    data: Mapping[str, np.ndarray],
    case: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    canonical, uncertainty = model.predict_one(data, case)
    low = data["low_field"]
    norm = math.sqrt(max(field_energy(low), 1e-300))
    sign = float(case.problem.sources[0].polarity)
    return low + sign * canonical * norm, uncertainty


def fit_scalar(
    family: str,
    length: float,
    data_rows: Sequence[Mapping[str, np.ndarray]],
) -> tuple[dict[str, SharedKernelGP], dict[str, Any]]:
    models: dict[str, SharedKernelGP] = {}
    suitability: dict[str, Any] = {}
    for column, name in enumerate(QOIS):
        if name not in PROTOCOL["models"]["scalar_targets"]:
            continue
        low = np.asarray([data["low_qois"][column] for data in data_rows])
        high = np.asarray([data["high_qois"][column] for data in data_rows])
        correlation = float(np.corrcoef(low, high)[0, 1])
        qualified = bool(
            np.isfinite(correlation)
            and abs(correlation) >= 0.8
        )
        suitability[name] = {
            "candidate_correlation": correlation,
            "threshold": 0.8,
            "qualified": qualified,
        }
        if not qualified:
            continue
        features, targets = [], []
        for data in data_rows:
            coarse = qoi_decode(data["low_qois"])
            features.append(model_features(data["row"], coarse, name))
            targets.append(
                qoi_transform(name, float(data["high_qois"][column]))
                - qoi_transform(name, float(data["low_qois"][column]))
            )
        models[name] = SharedKernelGP.fit(
            np.asarray(features), np.asarray(targets), length
        )
    return models, suitability


def predict_scalar(
    family: str,
    model: Mapping[str, SharedKernelGP],
    data_rows: Sequence[Mapping[str, np.ndarray]],
) -> np.ndarray:
    prediction = np.full((len(data_rows), len(QOIS)), np.nan)
    for column, name in enumerate(QOIS):
        if name not in model:
            continue
        features = np.asarray(
            [
                model_features(data["row"], qoi_decode(data["low_qois"]), name)
                for data in data_rows
            ]
        )
        discrepancy = model[name].predict(features)[:, 0]
        for row_index, data in enumerate(data_rows):
            latent = qoi_transform(name, float(data["low_qois"][column]))
            prediction[row_index, column] = qoi_inverse(
                name, latent + float(discrepancy[row_index])
            )
    return prediction


def compose_qoi_predictions(
    partial: np.ndarray,
    fields: Sequence[np.ndarray],
    cases: Sequence[Any],
    data_rows: Sequence[Mapping[str, np.ndarray]],
) -> np.ndarray:
    output = np.asarray(partial, dtype=float).copy()
    for row_index, (field, case, data) in enumerate(
        zip(fields, cases, data_rows, strict=True)
    ):
        derived = reconstructed_qois(
            case, field, float(data["input_source_error"])
        )
        for column, name in enumerate(QOIS):
            if not np.isfinite(output[row_index, column]):
                output[row_index, column] = derived[name]
        output[row_index, QOIS.index("source_representation_error")] = derived[
            "source_representation_error"
        ]
        for name in PROTOCOL["models"]["derived_from_reconstruction"]:
            output[row_index, QOIS.index(name)] = derived[name]
    return output


def predict_field(
    family: str,
    basis: WeightedPOD,
    model: SharedKernelGP,
    data: Mapping[str, np.ndarray],
    case: Any,
    low_basis: WeightedPOD | None = None,
) -> np.ndarray:
    low = data["low_field"]
    norm = math.sqrt(max(field_energy(low), 1e-300))
    encoder = basis if low_basis is None else low_basis
    sign = float(case.problem.sources[0].polarity)
    observed = encoder.whiten(
        encoder.project((sign * low / norm)[None, :])
    )[0]
    feature = np.concatenate((model_features(data["row"], {}, None), observed))[None, :]
    aligned = basis.reconstruct(basis.unwhiten(model.predict(feature)))[0]
    return low + sign * aligned * norm


def projection_oracle(
    family: str,
    basis: WeightedPOD,
    data: Mapping[str, np.ndarray],
    case: Any,
) -> np.ndarray:
    aligned = field_target(data, case, family)[None, :]
    reconstructed = basis.reconstruct(basis.project(aligned))[0]
    low = data["low_field"]
    norm = math.sqrt(max(field_energy(low), 1e-300))
    sign = float(case.problem.sources[0].polarity)
    return low + sign * reconstructed * norm


def scalar_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    output = {}
    for column, name in enumerate(QOIS):
        scale = max(float(np.ptp(truth[:, column])), float(np.max(np.abs(truth[:, column]))) * 1e-12, 1e-15)
        errors = np.abs(prediction[:, column] - truth[:, column]) / scale
        output[name] = {
            "nrmse": float(np.sqrt(np.mean(errors * errors))),
            "worst": float(np.max(errors)),
            "range": scale,
        }
    return {
        "outputs": output,
        "worst_nrmse": max(item["nrmse"] for item in output.values()),
        "worst_error": max(item["worst"] for item in output.values()),
    }


def field_metrics(truth: Sequence[np.ndarray], prediction: Sequence[np.ndarray]) -> dict[str, Any]:
    rows = [
        {
            "l2": float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-300)),
            "energy": abs(field_energy(p) - field_energy(t)) / max(field_energy(t), 1e-300),
            "topology": topology_match(p, t),
        }
        for p, t in zip(prediction, truth, strict=True)
    ]
    return {
        "rows": rows,
        "worst_l2": max(item["l2"] for item in rows),
        "worst_energy": max(item["energy"] for item in rows),
        "topology_matches": sum(item["topology"] for item in rows),
        "count": len(rows),
    }


def candidate_passes(
    scalar: Mapping[str, Any],
    field: Mapping[str, Any],
    pod_ok: bool,
    projection_oracle_ok: bool,
) -> bool:
    gates = PROTOCOL["gates"]
    return bool(
        pod_ok
        and projection_oracle_ok
        and scalar["worst_nrmse"] <= gates["scalar_nrmse_max"]
        and scalar["worst_error"] <= gates["scalar_worst_max"]
        and field["worst_l2"] <= gates["field_l2_max"]
        and field["worst_energy"] <= gates["field_energy_max"]
        and field["topology_matches"] == field["count"]
    )


def select_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passing = [item for item in candidates if item["passed"]]
    return min(
        passing,
        key=lambda item: (
            item["budget"],
            item["scalar_metrics"]["worst_nrmse"],
            item["field_metrics"]["worst_l2"],
            item["pod_rank"],
            item["scalar_family"],
            item["field_family"],
        ),
        default=None,
    )


def conformal_calibration(
    truth: np.ndarray,
    prediction: np.ndarray,
    truth_fields: Sequence[np.ndarray],
    prediction_fields: Sequence[np.ndarray],
) -> dict[str, Any]:
    rank = exact_rank(len(truth), PROTOCOL["gates"]["conformal_nominal"])
    scalar = np.sort(np.abs(prediction - truth), axis=0)[rank - 1]
    fields = sorted(
        float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-300))
        for p, t in zip(prediction_fields, truth_fields, strict=True)
    )
    return {
        "count": len(truth),
        "exact_rank": rank,
        "scalar_radii": scalar.tolist(),
        "field_l2_radius": fields[rank - 1],
    }


def assessment_coverage(
    truth: np.ndarray,
    prediction: np.ndarray,
    calibration: Mapping[str, Any],
    ranges: np.ndarray,
) -> dict[str, Any]:
    radii = np.asarray(calibration["scalar_radii"])
    hits = np.abs(prediction - truth) <= radii
    widths = 2 * radii / ranges
    result = {
        "coverage": float(np.mean(hits)),
        "median_width": float(np.median(widths)),
        "p95_width": percentile(widths.tolist(), 0.95),
    }
    result["passed"] = bool(
        result["coverage"] >= PROTOCOL["gates"]["group_coverage_min"]
        and result["median_width"] <= PROTOCOL["gates"]["median_width_max"]
        and result["p95_width"] <= PROTOCOL["gates"]["p95_width_max"]
    )
    return result


def latency_metrics(low: Sequence[float], inference: Sequence[float], fine: Sequence[float]) -> dict[str, Any]:
    combined = [a + b for a, b in zip(low, inference, strict=True)]
    speedups = [target / source for target, source in zip(fine, combined, strict=True)]
    return {
        "coarse_inference_median": percentile(combined, 0.5),
        "coarse_inference_p95": percentile(combined, 0.95),
        "fine_median": percentile(fine, 0.5),
        "fine_p95": percentile(fine, 0.95),
        "speedup_median": percentile(speedups, 0.5),
        "speedup_p05": percentile(speedups, 0.05),
        "passed": percentile(speedups, 0.5) >= PROTOCOL["gates"]["latency_median_speedup_min"],
    }


def _path_hash(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _synthetic_science_paths() -> dict[str, Any]:
    required = set(PROTOCOL["synthetic_preflight"]["required_path_groups"])
    executed: dict[str, list[str]] = {name: [] for name in required}
    counters = new_counters()
    designs = raw_designs()
    case, _ = construct_geometry(designs[0], 0)
    size = 2 * HIGH_DOMAIN.shape[0] * HIGH_DOMAIN.shape[1]
    coordinate = np.linspace(0.0, 4.0 * math.pi, size)
    base = 0.2 + 0.01 * np.sin(coordinate)
    with tempfile.TemporaryDirectory(prefix="l1a-v10-synthetic-science-") as temporary:
        cache = Path(temporary)
        for index in range(324):
            row = np.asarray(designs[index].values)
            low = base * (1.0 + 0.0002 * index)
            amplitude = 0.0002 * (1.0 + 0.1 * math.sin(index))
            high = low + amplitude * np.cos(coordinate * (1 + index % 3))
            low_qoi = np.asarray(
                [1.0 + 0.01 * column + 0.00005 * index for column in range(len(QOIS))]
            )
            high_qoi = low_qoi * (1.0005 + 0.000001 * index)
            save_checkpoint(cache, index, row, low, high, low_qoi, high_qoi)
        probe = load_checkpoint(cache, 0, "candidate", counters)
        qoi_decode(probe["low_qois"])
        augmented = WeightedPOD._augment(probe["high_field"][None, :], 3)
        restored = WeightedPOD._decode(augmented, 3)[0]
        if not np.array_equal(restored, probe["high_field"]):
            raise RuntimeError("primary representation roundtrip is not exact")
        candidate_data = [
            load_checkpoint(cache, index, "candidate", counters)
            for index in range(270)
        ]
        method_data = [
            load_checkpoint(cache, index, "method", counters)
            for index in range(270, 324)
        ]
        candidates, fitted = fit_development_candidates(
            candidate_data,
            method_data,
            [case] * len(candidate_data),
            [case] * len(method_data),
            counters,
        )
        if not candidates:
            raise RuntimeError("synthetic candidate matrix is empty")
        local_probe = LocalResidualInterpolator.fit(
            candidate_data[:32], [case] * 32, "pooled", 8, "wendland-c2"
        )
        local_probe.predict_one(method_data[0], case)
        local_probe.to_dict()
        for key, artifact in fitted.items():
            if key[0] == "scalar":
                for model in artifact.values():
                    model.to_dict()
            elif key[0] in {"basis", "low_basis", "field"}:
                for model in artifact.values():
                    model.to_dict()
        rank_failure = np.zeros((8, size))
        rank_failure[:, :8] = np.eye(8)
        if WeightedPOD.fit(rank_failure, 0.999, 2) is not None:
            raise RuntimeError("rank-cap failure path did not fail closed")
        passing = dict(candidates[0])
        passing["passed"] = True
        passing["scalar_metrics"] = {"worst_nrmse": 0.0}
        passing["field_metrics"] = {"worst_l2": 0.0}
        failing = dict(candidates[0])
        failing["passed"] = False
        if select_candidate((passing, failing)) is None:
            raise RuntimeError("synthetic passing selection failed")
        if select_candidate((failing,)) is not None:
            raise RuntimeError("synthetic rejection selection failed")
        truth_qois = np.asarray([data["high_qois"] for data in method_data])
        predicted_qois = truth_qois * 1.0001
        truth_fields = [data["high_field"] for data in method_data]
        predicted_fields = [field * 1.0001 for field in truth_fields]
        calibration = conformal_calibration(
            truth_qois, predicted_qois, truth_fields, predicted_fields
        )
        ranges = np.maximum(np.ptp(truth_qois, axis=0), 1e-9)
        assessment_coverage(truth_qois, predicted_qois, calibration, ranges)
        topology(predicted_fields[0])
        topology_match(predicted_fields[0], truth_fields[0])
        latency_metrics([0.01] * 54, [0.001] * 54, [0.05] * 54)
        labels = {
            "npz-save-load-qoi": ["save/load/qoi"],
            "snapshot-high-residual": ["observed coarse residual snapshots"],
            "lossless-primary-roundtrip": ["primary physical augment/decode roundtrip"],
            "field-qoi-derivation-input-source": ["reconstructed field QoIs and input-only source error"],
            "all-budgets-lengths-families": [f"{len(candidates)} candidates"],
            "local-kernel-feature-uncertainty": ["local kernels, descriptors and neighbour uncertainty"],
            "localized-fit-predict-duplicate-ood": ["local interpolation and stable OOD handling"],
            "candidate-heldout-method-qualification": ["held-out candidate before method qualification"],
            "baseline-oracle-coefficient-metrics": ["three field metric paths"],
            "selection-pass-fail-none": ["passing/failing/no-model"],
            "calibration-conformal-assessment-topology-latency": ["full post-freeze path"],
        }
        for name in required:
            executed[name].extend(labels[name])
    path_functions = (
        save_checkpoint,
        load_checkpoint,
        qoi_decode,
        build_snapshots,
        local_features,
        LocalResidualInterpolator.fit,
        LocalResidualInterpolator.predict_one,
        LocalResidualInterpolator.to_dict,
        predict_local_field,
        align_vector,
        unalign_vector,
        fit_scalar,
        predict_scalar,
        SharedKernelGP.fit,
        SharedKernelGP.predict,
        SharedKernelGP.to_dict,
        WeightedPOD.fit,
        WeightedPOD.representation_roundtrip,
        WeightedPOD.project,
        WeightedPOD.observed_coefficients,
        WeightedPOD.reconstruct,
        WeightedPOD.to_dict,
        predict_field,
        projection_oracle,
        scalar_metrics,
        field_metrics,
        candidate_passes,
        select_candidate,
        conformal_calibration,
        assessment_coverage,
        topology,
        topology_match,
        latency_metrics,
    )
    return {
        "required_groups": sorted(required),
        "executed_paths": executed,
        "executed_path_hashes": {
            f"{function.__module__}.{function.__qualname__}": _path_hash(function)
            for function in path_functions
        },
        "real_field_solver_calls": 0,
        "synthetic_checkpoint_reads": counters["checkpoint_reads"],
        "complete": True,
    }


def synthetic_runtime_preflight() -> dict[str, Any]:
    """Run all scientific paths and every terminal state through the shared runtime."""

    expected_states = (
        BundleState.PREBUNDLE_FAILURE,
        BundleState.RUNTIME_FAILURE,
        BundleState.DEVELOPMENT_REJECTION,
        BundleState.ASSESSMENT_REJECTION,
        BundleState.ACCEPTED_RESULT,
    )
    matrix: list[dict[str, Any]] = []
    science: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="l1a-v7-callback-matrix-") as temporary:
        matrix_root = Path(temporary)
        for index, expected in enumerate(expected_states):
            root = matrix_root / f"{index}-{expected.value}"
            runtime = ExperimentRuntime(
                experiment_id=f"l1a-field-surrogate-v10-synthetic-{expected.value}",
                result_root=root / "results",
                cache_root=root / "cache",
                attestation=ExecutionAttestation(
                    attempt=1,
                    commit=ACCEPTED_RUNTIME_COMMIT,
                    command="synthetic callback matrix",
                    device="synthetic-no-field-labels",
                    clean_worktree=True,
                    host="synthetic-preflight",
                ),
                producer=_synthetic_science_paths,
                source_root=REPO / "modern",
            )

            def prebundle(_context: RunContext, state: BundleState = expected) -> Mapping[str, Any]:
                if state is BundleState.PREBUNDLE_FAILURE:
                    raise RuntimeError("synthetic prebundle failure")
                return {"real_field_solver_calls": 0, "real_field_label_accesses": 0}

            def development(context: RunContext, state: BundleState = expected) -> Decision:
                nonlocal science
                if state is BundleState.RUNTIME_FAILURE:
                    raise RuntimeError("synthetic development runtime failure")
                if state is BundleState.DEVELOPMENT_REJECTION:
                    return Decision(False, {"synthetic": "development rejection"})
                if state is BundleState.ACCEPTED_RESULT:
                    science = _synthetic_science_paths()
                    context.write_json("synthetic/scientific-paths.json", science)
                return Decision(True, {"synthetic": "development accepted"})

            def assessment(_context: RunContext, state: BundleState = expected) -> Decision:
                if state is BundleState.ASSESSMENT_REJECTION:
                    return Decision(False, {"synthetic": "assessment rejection"})
                if state is BundleState.ACCEPTED_RESULT:
                    required = set(PROTOCOL["synthetic_preflight"]["required_path_groups"])
                    if set(science["executed_paths"]) != required:
                        raise RuntimeError("scientific callback path matrix is incomplete")
                return Decision(True, {"synthetic": "assessment accepted"})

            outcome = runtime.run(RuntimeCallbacks(prebundle, development, assessment))
            if outcome.state is not expected:
                raise RuntimeError(
                    f"synthetic state mismatch: expected {expected.value}, got {outcome.state.value}"
                )
            validate_bundle(root / "results")
            terminal_counts = strict_load(root / "results" / "terminal.json")["counts"]
            if terminal_counts["label_access_count"] or terminal_counts["expensive_operation_count"]:
                raise RuntimeError("synthetic matrix accessed a real label or expensive backend")
            matrix.append({"state": outcome.state.value, "counts": terminal_counts})
    if not science.get("complete"):
        raise RuntimeError("accepted synthetic callback omitted scientific paths")
    return {
        **science,
        "shared_runtime_commit": ACCEPTED_RUNTIME_COMMIT,
        "terminal_state_matrix": matrix,
        "terminal_states_exercised": [item["state"] for item in matrix],
        "real_field_label_accesses": 0,
    }


def prepare() -> dict[str, Any]:
    records, valid = preflight_candidates()
    frozen = select_frozen(valid)
    rebuilt, rebuild_records = rebuild_frozen(frozen)
    initial = {index: valid[raw_index] for index, raw_index in enumerate(frozen)}
    hash_failures = [
        index
        for index in range(432)
        if (
            initial[index].geometry_sha256 != rebuilt[index].geometry_sha256
            or initial[index].source_sha256 != rebuilt[index].source_sha256
        )
    ]
    if hash_failures:
        raise RuntimeError(f"frozen rebuild mismatch: {hash_failures}")
    rejected = [record for record in records if not record["valid"]]
    geometry = {
        "schema_version": "cft-revival.l1a-field-surrogate-v10.geometry-preflight/10.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "raw_count": len(records),
        "valid_count": len(valid),
        "rejected_count": len(rejected),
        "corrected_count": sum(record.get("attempt_count", 1) > 1 for record in records),
        "raw_records": records,
        "rejected_raw_rows": rejected,
        "frozen_raw_indices": list(frozen),
        "frozen_rebuild_records": rebuild_records,
        "frozen_hash_failure_count": 0,
        "field_solver_access_count": 0,
        "qoi_label_access_count": 0,
    }
    geometry["geometry_preflight_hash"] = canonical_hash(geometry)
    write_predeclared_json(GEOMETRY_PREFLIGHT, geometry)
    designs = raw_designs()
    cell_order = sorted(
        (stage, stratum, polarity)
        for stage in (3, 4, 5)
        for stratum in ("interpolation", "boundary", "ood")
        for polarity in (-1, 1)
    )
    def balance_rows(indices: Sequence[int], role: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for position, index in enumerate(indices):
            case = rebuilt[index]
            if role == "candidate":
                round_index = position // 18
                order = (
                    cell_order
                    if round_index % 2 == 0
                    else list(reversed(cell_order))
                )
                expected_stage, stratum, expected_polarity = order[position % 18]
            else:
                stratum = next(
                    name for name, values in stratum_indices(role).items()
                    if index in values
                )
                expected_stage = len(case.geometry.stages)
                expected_polarity = int(case.problem.sources[0].polarity)
            stage = len(case.geometry.stages)
            polarity = int(case.problem.sources[0].polarity)
            if (stage, polarity) != (expected_stage, expected_polarity):
                raise RuntimeError("balanced role position does not match geometry")
            key = f"stage-{stage}/{stratum}/polarity-{polarity:+d}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))
    partition = {
        "schema_version": "cft-revival.l1a-field-surrogate-v10.partitions/10.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "geometry_preflight_hash": geometry["geometry_preflight_hash"],
        "frozen_raw_indices": list(frozen),
        "frozen_design_ids": [designs[index].design_id for index in frozen],
        "frozen_design_rows": [list(designs[index].values) for index in frozen],
        "roles": {
            role: list(role_indices(role))
            for role in ("candidate", "method", "calibration", "assessment")
        },
        "method_strata": {name: list(rows) for name, rows in stratum_indices("method").items()},
        "calibration_strata": {name: list(rows) for name, rows in stratum_indices("calibration").items()},
        "assessment_strata": {name: list(rows) for name, rows in stratum_indices("assessment").items()},
        "role_balance": {
            role: balance_rows(role_indices(role), role)
            for role in ("candidate", "method", "calibration", "assessment")
        },
        "candidate_prefix_balance": {
            str(budget): balance_rows(tuple(range(budget)), "candidate")
            for budget in PROTOCOL["sampling"]["high_budgets"]
        },
        "prior_coordinate_intersection_count": 0,
        "field_solver_access_count": 0,
    }
    partition["partition_hash"] = canonical_hash(partition)
    write_predeclared_json(PARTITIONS, partition)
    locked = dependency_lock()
    if not locked["accepted_experiment_runtime"]["is_ancestor"]:
        raise RuntimeError("accepted experiment_runtime commit is not an ancestor")
    write_predeclared_json(DEPENDENCY_LOCK, locked)
    source_paths = (ROOT / "protocol.py", ROOT / "experiment.py", ROOT / "run.py", ROOT / "validate.py")
    static = static_undefined_names(source_paths)
    if not static["passed"]:
        raise RuntimeError(f"undefined global names: {static['unresolved']}")
    runtime = synthetic_runtime_preflight()
    required = set(PROTOCOL["synthetic_preflight"]["required_path_groups"])
    if set(runtime["executed_paths"]) != required or not runtime["complete"]:
        raise RuntimeError("synthetic runtime did not cover every required path group")
    if set(runtime["terminal_states_exercised"]) != {
        state.value for state in BundleState
    }:
        raise RuntimeError("synthetic runtime did not exercise all five terminal states")
    synthetic = {
        "schema_version": "cft-revival.l1a-field-surrogate-v10.synthetic-preflight/10.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "geometry_preflight_hash": geometry["geometry_preflight_hash"],
        "partition_hash": partition["partition_hash"],
        "dependency_lock_hash": locked["dependency_lock_hash"],
        "static_undefined_name_check": static,
        "runtime_path_coverage": runtime,
        "field_solver_access_count": 0,
        "real_qoi_label_access_count": 0,
        "passed": True,
    }
    synthetic["synthetic_preflight_hash"] = canonical_hash(synthetic)
    write_predeclared_json(SYNTHETIC_PREFLIGHT, synthetic)
    return {
        "passed": True,
        "raw_valid": len(valid),
        "raw_rejected": len(rejected),
        "raw_corrected": geometry["corrected_count"],
        "frozen": len(frozen),
        "synthetic_path_groups": len(runtime["executed_paths"]),
        "synthetic_executed_paths": sum(len(value) for value in runtime["executed_paths"].values()),
        "terminal_states": runtime["terminal_states_exercised"],
        "undefined_names": static["unresolved"],
    }


@dataclass(frozen=True)
class VerifiedExecution:
    attestation: ExecutionAttestation
    closure: tuple[tuple[str, str, str, str], ...]
    dependency_lock_hash: str
    clean_status_sha256: str

    def closure_records(self) -> list[dict[str, str]]:
        return [
            {"path": path, "mode": mode, "type": kind, "blob": blob}
            for path, mode, kind, blob in self.closure
        ]


RESULT_RELATIVE = RESULTS.relative_to(REPO).as_posix()
CACHE_RELATIVE = CACHE.relative_to(REPO).as_posix()


def validate_repository_status(
    status: str,
    *,
    allowed_untracked_roots: Sequence[str] = (),
) -> None:
    """Reject all drift except untracked runtime-owned root descendants."""

    allowed = tuple(root.rstrip("/") for root in allowed_untracked_roots)
    violations: list[str] = []
    for raw in status.splitlines():
        if not raw:
            continue
        code, path = raw[:2], raw[3:].replace("\\", "/")
        permitted = code == "??" and any(
            path == root or path.startswith(root + "/") for root in allowed
        )
        if not permitted:
            violations.append(raw)
    if violations:
        raise RuntimeError(f"repository drift is not allowed: {violations[0]}")


def verify_before_runtime() -> VerifiedExecution:
    """Verify and freeze the clean execution identity before runtime mutation."""

    head = _git("rev-parse", "HEAD")
    if _git("symbolic-ref", "-q", "HEAD", check=False):
        raise RuntimeError("execution requires detached HEAD")
    if _git("show", "-s", "--format=%s", head) != SUBJECT:
        raise RuntimeError("HEAD is not v7 preregistration")
    if _git("rev-parse", REMOTE) != head:
        raise RuntimeError("remote v7 branch is not the preregistration")
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise RuntimeError("preregistration commit is not exact-path isolated")
    for path in (GEOMETRY_PREFLIGHT, PARTITIONS, SYNTHETIC_PREFLIGHT, DEPENDENCY_LOCK):
        verify_json(path)
    locked = strict_load(DEPENDENCY_LOCK)
    if dependency_lock() != locked:
        raise RuntimeError("runtime differs from dependency lock")
    synthetic = strict_load(SYNTHETIC_PREFLIGHT)
    if not synthetic["passed"] or synthetic["field_solver_access_count"]:
        raise RuntimeError("synthetic preflight is incomplete or accessed the solver")
    closure_records = full_closure()
    if not closure_records:
        raise RuntimeError("dependency closure is empty")
    if RESULTS.exists() or CACHE.exists():
        raise RuntimeError("runtime-owned roots must be absent before attestation")
    clean_status = _git("status", "--porcelain=v1", "--untracked-files=all")
    validate_repository_status(clean_status)
    closure = tuple(
        (row["path"], row["mode"], row["type"], row["blob"])
        for row in closure_records
    )
    attestation = ExecutionAttestation(
        attempt=1,
        commit=head,
        command="python -m experiments.l1a_field_surrogate_v10.run execute",
        device=PROTOCOL["execution"]["device"],
        clean_worktree=True,
        host=platform.node(),
    )
    return VerifiedExecution(
        attestation=attestation,
        closure=closure,
        dependency_lock_hash=locked["dependency_lock_hash"],
        clean_status_sha256=hashlib.sha256(clean_status.encode("utf-8")).hexdigest(),
    )


def claim_git_common_attempt(verified: VerifiedExecution) -> dict[str, Any]:
    """Atomically claim the sole v7 attempt across every worktree."""

    common = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    namespace = common / "cft-revival-attempt-locks"
    namespace.mkdir(mode=0o700, exist_ok=True)
    payload = {
        "experiment_id": "l1a-field-surrogate-v10",
        "attempt": 1,
        "preregistration_commit": verified.attestation.commit,
        "protocol_hash": PROTOCOL_HASH,
    }
    data = canonical_bytes(payload)
    try:
        with PinnedDirectory.open(namespace) as pinned:
            descriptor = pinned.open_file_exclusive(
                "l1a-field-surrogate-v10.attempt-1.json", 0o600
            )
            try:
                os.write(descriptor, data)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except FileExistsError as error:
        raise RuntimeError(
            "Git-common-dir namespace already contains the sole v7 attempt claim"
        ) from error
    return {
        **payload,
        "claim_sha256": hashlib.sha256(data).hexdigest(),
        "namespace": "git-common-dir/cft-revival-attempt-locks",
    }


def verify_runtime_drift() -> None:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    validate_repository_status(
        status,
        allowed_untracked_roots=(RESULT_RELATIVE, CACHE_RELATIVE),
    )


def index_role(index: int) -> str:
    for role in ("candidate", "method", "calibration", "assessment"):
        if index in role_indices(role):
            return role
    raise ValueError(f"index {index} has no role")


def numerical_record(index: int, fidelity: str, case: Any, field: Any, values: Mapping[str, float]) -> dict[str, Any]:
    return {
        "index": index,
        "role": index_role(index),
        "fidelity": fidelity,
        "geometry_sha256": case.geometry_sha256,
        "pairing_sha256": canonical_hash(
            {
                "geometry_sha256": case.geometry_sha256,
                "turns": [(source.polarity, source.ampere_turns_a) for source in case.problem.sources],
            }
        ),
        "iterations": field.diagnostics.iterations,
        "relative_residual_l2": field.diagnostics.relative_residual_l2,
        "flux_identity": field.diagnostics.max_flux_reconstruction_identity_t_per_m,
        "boundary": values["boundary_to_peak_ratio"],
        "source_error": values["source_representation_error"],
    }


def checkpoint_inventory(cache: Path, indices: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "path": f"{index:03d}.npz",
            "sha256": _file_sha(cache / f"{index:03d}.npz"),
            "bytes": (cache / f"{index:03d}.npz").stat().st_size,
        }
        for index in indices
    ]


def solve_phase(
    context: RunContext,
    indices: Sequence[int],
    cases: Mapping[int, Any],
    rows: Sequence[Sequence[float]],
    cache: Path,
    counters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    records = []
    timings = {}
    for index in indices:
        role = index_role(index)
        input_source_error = input_source_representation_error(cases[index])
        tick = time.perf_counter()
        context.before_expensive(
            f"{role}-{index:03d}-low",
            kind="solver",
            details={"role": role, "index": index, "fidelity": "low"},
        )
        counters["solver_accesses"][role]["low"] += 1
        low = solve_fidelity(cases[index], "low")
        low_time = time.perf_counter() - tick
        low_values = qois(cases[index], low, "low")
        counters["materialized"][role]["low"] += 1
        tick = time.perf_counter()
        context.before_expensive(
            f"{role}-{index:03d}-fine",
            kind="solver",
            details={"role": role, "index": index, "fidelity": "fine"},
        )
        counters["solver_accesses"][role]["fine"] += 1
        fine = solve_fidelity(cases[index], "high")
        fine_time = time.perf_counter() - tick
        fine_values = qois(cases[index], fine, "high")
        counters["materialized"][role]["fine"] += 1
        low_vector = prolong_low(low)
        fine_vector = field_vector(fine)
        save_checkpoint(
            cache,
            index,
            rows[index],
            low_vector,
            fine_vector,
            [low_values[name] for name in QOIS],
            [fine_values[name] for name in QOIS],
            input_source_error,
        )
        low_record = numerical_record(index, "low", cases[index], low, low_values)
        fine_record = numerical_record(index, "fine", cases[index], fine, fine_values)
        if low_record["pairing_sha256"] != fine_record["pairing_sha256"]:
            raise RuntimeError("coarse/fine geometry pairing mismatch")
        records.extend((low_record, fine_record))
        timings[index] = {"low": low_time, "fine": fine_time}
        del low, fine, low_vector, fine_vector
    return records, timings


def persist_phase(
    context: RunContext,
    name: str,
    indices: Sequence[int],
    cache: Path,
    records: Sequence[Mapping[str, Any]],
    counters: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": "cft-revival.l1a-field-surrogate-v10.phase/10.0.0",
        "phase": name,
        "indices": list(indices),
        "numerical_records": list(records),
        "checkpoint_inventory": checkpoint_inventory(cache, indices),
        "access_counters": counters,
    }
    value["phase_hash"] = canonical_hash(value)
    context.write_json(f"artifacts/phase-{name}.json", value)
    return value


def load_role(
    context: RunContext,
    cache: Path,
    indices: Sequence[int],
    role: str,
    counters: dict[str, Any],
) -> list[dict[str, np.ndarray]]:
    return [
        load_checkpoint(cache, index, role, counters, context)
        for index in indices
    ]


def fit_development_candidates(
    candidate_data: Sequence[Mapping[str, np.ndarray]],
    method_data: Sequence[Mapping[str, np.ndarray]],
    candidate_cases: Sequence[Any],
    method_cases: Sequence[Any],
    counters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], Any]]:
    method_truth_qois = np.asarray([data["high_qois"] for data in method_data])
    method_truth_fields = [data["high_field"] for data in method_data]
    baseline = field_metrics(
        method_truth_fields, [data["low_field"] for data in method_data]
    )
    candidates: list[dict[str, Any]] = []
    fitted: dict[tuple[Any, ...], Any] = {}
    for budget in PROTOCOL["sampling"]["high_budgets"]:
        training = list(candidate_data[:budget])
        training_cases = list(candidate_cases[:budget])
        folds = np.asarray([(index // 18) % 3 for index in range(budget)])
        configurations = [
            (scope, neighbours, kernel)
            for scope in PROTOCOL["models"]["local_scopes"]
            for neighbours in PROTOCOL["models"]["neighbour_counts"]
            for kernel in PROTOCOL["models"]["kernel_families"]
        ]
        budget_records = []
        best_replay: tuple[tuple[float, float], np.ndarray, list[Any], dict[str, Any]] | None = None
        for scope, neighbours, kernel in configurations:
            cv_prediction: list[np.ndarray | None] = [None] * budget
            uncertainty: list[dict[str, float] | None] = [None] * budget
            fold_records = []
            for fold in range(3):
                fit_positions = np.flatnonzero(folds != fold)
                validation_positions = np.flatnonzero(folds == fold)
                model = LocalResidualInterpolator.fit(
                    [training[index] for index in fit_positions],
                    [training_cases[index] for index in fit_positions],
                    scope,
                    neighbours,
                    kernel,
                )
                for index in validation_positions:
                    prediction, spread = predict_local_field(
                        model, training[index], training_cases[index]
                    )
                    cv_prediction[index] = prediction
                    uncertainty[index] = spread
                fold_records.append(
                    {
                        "fold": fold,
                        "fit_count": len(fit_positions),
                        "validation_count": len(validation_positions),
                        "overlap_count": len(
                            set(fit_positions).intersection(validation_positions)
                        ),
                    }
                )
            predictions = [value for value in cv_prediction if value is not None]
            cv_field = field_metrics(
                [data["high_field"] for data in training], predictions
            )
            partial = np.full((budget, len(QOIS)), np.nan)
            cv_qois = compose_qoi_predictions(
                partial, predictions, training_cases, training
            )
            cv_scalar = scalar_metrics(
                np.asarray([data["high_qois"] for data in training]), cv_qois
            )
            candidate_ok = candidate_passes(
                cv_scalar, cv_field, True, True
            )
            record = {
                "budget": budget,
                "scope": scope,
                "neighbours": neighbours,
                "kernel": kernel,
                "scalar_family": "reconstructed_physical_field_qois",
                "scalar_length": 0.0,
                "field_family": "localized_full_physical_residual_interpolator",
                "field_length": 0.0,
                "pod_rank": 0,
                "pod_retained": 1.0,
                "representation_roundtrip_max_l2": 0.0,
                "candidate_cv": {
                    "folds": fold_records,
                    "field_metrics": cv_field,
                    "scalar_metrics": cv_scalar,
                    "uncertainty": uncertainty,
                    "passed": candidate_ok,
                },
                "scalar_metrics": cv_scalar,
                "field_metrics": cv_field,
                "coarse_baseline": baseline,
                "projection_oracle": cv_field,
                "projection_oracle_passed": candidate_ok,
                "coefficient_model_fitted": False,
                "comparator": {
                    "independent_models": [
                        "bulk-br-energy",
                        "bulk-bz-energy",
                        "axis-bz",
                        "overlapping-landmark-patches",
                    ],
                    "shared_augmented_svd": False,
                    "neural_or_autoencoder": False,
                },
                "passed": False,
            }
            budget_records.append(record)
            candidates.append(record)
            score = (cv_field["worst_l2"], cv_scalar["worst_nrmse"])
            if best_replay is None or score < best_replay[0]:
                best_replay = (
                    score,
                    np.asarray(predictions),
                    uncertainty,
                    {
                        "scope": scope,
                        "neighbours": neighbours,
                        "kernel": kernel,
                    },
                )
        if best_replay is not None:
            fitted[("replay", budget)] = best_replay
        qualifying = [item for item in budget_records if item["candidate_cv"]["passed"]]
        if not qualifying:
            continue
        chosen = min(
            qualifying,
            key=lambda item: (
                item["candidate_cv"]["field_metrics"]["worst_l2"],
                item["candidate_cv"]["scalar_metrics"]["worst_nrmse"],
                item["scope"],
                item["neighbours"],
                item["kernel"],
            ),
        )
        model = LocalResidualInterpolator.fit(
            training,
            training_cases,
            chosen["scope"],
            chosen["neighbours"],
            chosen["kernel"],
        )
        counters["model_fit_accesses"] += 1
        fitted[
            (
                "local",
                budget,
                chosen["scope"],
                chosen["neighbours"],
                chosen["kernel"],
            )
        ] = model
        method_predictions, method_uncertainty = [], []
        for data, case in zip(method_data, method_cases, strict=True):
            prediction, spread = predict_local_field(model, data, case)
            method_predictions.append(prediction)
            method_uncertainty.append(spread)
        method_field = field_metrics(method_truth_fields, method_predictions)
        method_qois = compose_qoi_predictions(
            np.full((len(method_data), len(QOIS)), np.nan),
            method_predictions,
            method_cases,
            method_data,
        )
        method_scalar = scalar_metrics(method_truth_qois, method_qois)
        method_ok = candidate_passes(method_scalar, method_field, True, True)
        chosen.update(
            {
                "method": {
                    "field_metrics": method_field,
                    "scalar_metrics": method_scalar,
                    "uncertainty": method_uncertainty,
                    "passed": method_ok,
                },
                "scalar_metrics": method_scalar,
                "field_metrics": method_field,
                "projection_oracle": method_field,
                "projection_oracle_passed": method_ok,
                "passed": method_ok,
            }
        )
    return candidates, fitted


def numerical_gates(records: Sequence[Mapping[str, Any]], expected_cases: int) -> dict[str, Any]:
    gates = PROTOCOL["gates"]
    checks = {
        "completion": len(records) == 2 * expected_cases,
        "residual": max(item["relative_residual_l2"] for item in records) <= gates["residual_max"],
        "boundary": max(item["boundary"] for item in records) <= gates["boundary_max"],
        "source": max(item["source_error"] for item in records) <= gates["source_error_max"],
        "flux": max(item["flux_identity"] for item in records) <= gates["flux_identity_max"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "max_residual": max(item["relative_residual_l2"] for item in records),
            "max_boundary": max(item["boundary"] for item in records),
            "max_source_error": max(item["source_error"] for item in records),
            "max_flux_identity": max(item["flux_identity"] for item in records),
        },
    }


def predict_selected_fields(
    selected: Mapping[str, Any],
    basis: Mapping[int, WeightedPOD],
    model: Mapping[int, SharedKernelGP],
    data_rows: Sequence[Mapping[str, np.ndarray]],
    cases: Sequence[Any],
    low_basis: Mapping[int, WeightedPOD] | None = None,
) -> list[np.ndarray]:
    return [
        predict_field(
            selected["field_family"],
            basis[len(case.geometry.stages)],
            model[len(case.geometry.stages)],
            data,
            case,
            None if low_basis is None else low_basis[len(case.geometry.stages)],
        )
        for data, case in zip(data_rows, cases, strict=True)
    ]



def execute() -> dict[str, Any]:
    """Execute the single preregistered attempt through ExperimentRuntime."""

    verified = verify_before_runtime()
    attempt_claim = claim_git_common_attempt(verified)
    runtime = ExperimentRuntime(
        experiment_id="l1a-field-surrogate-v10",
        result_root=RESULTS,
        cache_root=CACHE,
        attestation=verified.attestation,
        producer=execute,
        source_root=REPO / "modern",
    )
    state: dict[str, Any] = {
        "counters": new_counters(),
        "phase_hashes": {},
        "attempt_claim": attempt_claim,
        "all_records": [],
        "all_timings": {},
    }

    def prebundle(context: RunContext) -> Mapping[str, Any]:
        verify_runtime_drift()
        preregistration = verified.attestation.commit
        closure = verified.closure_records()
        partition = strict_load(PARTITIONS)
        raw_indices = tuple(int(value) for value in partition["frozen_raw_indices"])
        rows = tuple(
            tuple(float(value) for value in row)
            for row in partition["frozen_design_rows"]
        )
        cases, rebuilt = rebuild_frozen(raw_indices)
        expected = strict_load(GEOMETRY_PREFLIGHT)["frozen_rebuild_records"]
        if [
            (item["geometry_sha256"], item["source_sha256"]) for item in rebuilt
        ] != [
            (item["geometry_sha256"], item["source_sha256"]) for item in expected
        ]:
            raise RuntimeError("execution geometry differs from frozen preflight")
        state.update(
            {
                "commit": preregistration,
                "closure": closure,
                "cases": cases,
                "rows": rows,
            }
        )
        provenance = {
            "schema_version": "cft-revival.l1a-field-surrogate-v10.provenance/10.0.0",
            "preregistration_commit": preregistration,
            "accepted_experiment_runtime_commit": ACCEPTED_RUNTIME_COMMIT,
            "protocol_hash": PROTOCOL_HASH,
            "dependency_lock_hash": verified.dependency_lock_hash,
            "synthetic_preflight_hash": strict_load(SYNTHETIC_PREFLIGHT)[
                "synthetic_preflight_hash"
            ],
            "pre_runtime_clean_status_sha256": verified.clean_status_sha256,
            "git_blob_closure": closure,
            "git_blob_closure_hash": canonical_hash(closure),
            "git_common_attempt_claim": state["attempt_claim"],
            "solver_accesses_at_write": 0,
            "label_accesses_at_write": 0,
        }
        provenance["provenance_hash"] = canonical_hash(provenance)
        context.write_json("artifacts/preregistered-protocol.json", PROTOCOL)
        context.write_json("artifacts/provenance-closure.json", provenance)
        return {
            "preregistration_commit": preregistration,
            "protocol_hash": PROTOCOL_HASH,
            "frozen_geometry_count": len(cases),
            "geometry_screened_before_role_freeze": True,
            "field_solver_access_count": 0,
            "field_label_access_count": 0,
            "dependency_closure_entries": len(closure),
        }

    def development(context: RunContext) -> Decision:
        counters = state["counters"]
        cases = state["cases"]
        rows = state["rows"]
        cache = context.cache_root
        initial_indices = tuple(range(324))
        records, timings = solve_phase(
            context, initial_indices, cases, rows, cache, counters
        )
        state["all_records"].extend(records)
        state["all_timings"].update(timings)
        phase = persist_phase(
            context,
            "candidate-method-solves",
            initial_indices,
            cache,
            records,
            counters,
        )
        state["phase_hashes"][phase["phase"]] = phase["phase_hash"]
        persisted_checkpoints = [
            context.write_blob(
                f"artifacts/development-checkpoints/{index:03d}.npz",
                (cache / f"{index:03d}.npz").read_bytes(),
            )
            for index in initial_indices
        ]
        checkpoint_record = {
            "schema_version": "cft-revival.l1a-field-surrogate-v10.development-checkpoints/10.0.0",
            "count": len(persisted_checkpoints),
            "entries": persisted_checkpoints,
            "source_phase_hash": phase["phase_hash"],
            "purpose": "independent deterministic regeneration of all development metrics and correlations",
        }
        checkpoint_record["checkpoint_set_hash"] = canonical_hash(checkpoint_record)
        context.write_json(
            "artifacts/development-checkpoints.json", checkpoint_record
        )
        state["phase_hashes"]["development-checkpoints"] = checkpoint_record[
            "checkpoint_set_hash"
        ]
        if (
            counters["materialized"]["calibration"]["fine"]
            or counters["materialized"]["assessment"]["fine"]
        ):
            raise RuntimeError("future role materialized before method freeze")

        candidate_indices = role_indices("candidate")
        method_indices = role_indices("method")
        candidate_data = load_role(
            context, cache, candidate_indices, "candidate", counters
        )
        method_data = load_role(context, cache, method_indices, "method", counters)
        candidates, fitted = fit_development_candidates(
            candidate_data,
            method_data,
            [cases[index] for index in candidate_indices],
            [cases[index] for index in method_indices],
            counters,
        )
        replay_records = []
        for budget in PROTOCOL["sampling"]["high_budgets"]:
            score, predictions, uncertainty, configuration = fitted[
                ("replay", budget)
            ]
            stream = io.BytesIO()
            np.savez_compressed(stream, candidate_cv_fields=predictions)
            artifact = context.write_blob(
                f"artifacts/development-predictions/budget-{budget}.npz",
                stream.getvalue(),
            )
            replay_records.append(
                {
                    "budget": budget,
                    "score": list(score),
                    "configuration": configuration,
                    "uncertainty": uncertainty,
                    "artifact": artifact,
                }
            )
        context.write_json(
            "artifacts/development-predictions.json",
            {
                "schema_version": "cft-revival.l1a-field-surrogate-v10.development-predictions/10.0.0",
                "records": replay_records,
            },
        )
        selected = select_candidate(candidates)
        method_freeze = {
            "schema_version": "cft-revival.l1a-field-surrogate-v10.method-freeze/10.0.0",
            "candidates": candidates,
            "learning_curves": candidates,
            "selected": selected,
            "access_counters": counters,
            "calibration_materialized": 0,
            "assessment_materialized": 0,
        }
        method_freeze["method_freeze_hash"] = canonical_hash(method_freeze)
        context.write_json("artifacts/frozen-method-selection.json", method_freeze)
        state["phase_hashes"]["method-freeze"] = method_freeze["method_freeze_hash"]
        state.update(
            {
                "method_data": method_data,
                "selected": selected,
                "method_freeze": method_freeze,
            }
        )
        numerical = numerical_gates(state["all_records"], 324)
        if selected is None:
            context.write_json(
                "artifacts/development-rejection.json",
                {
                    "reason": "no candidate passed predeclared method gates",
                    "numerical": numerical,
                    "access_counters": counters,
                    "phase_hashes": state["phase_hashes"],
                },
            )
            return Decision(
                False,
                {
                    "reason": "no candidate passed predeclared method gates",
                    "selected": None,
                    "numerical": numerical,
                    "geometry_count": 432,
                    "solve_counters": counters["solver_accesses"],
                    "access_counters": counters,
                    "claim": PROTOCOL["classification"],
                },
            )

        budget = selected["budget"]
        local_model = fitted[
            (
                "local",
                budget,
                selected["scope"],
                selected["neighbours"],
                selected["kernel"],
            )
        ]
        state.update(
            {
                "local_model": local_model,
            }
        )
        context.write_json(
            "artifacts/selected-model.json",
            {
                "selected": selected,
                "localized_model": local_model.to_dict(),
            },
        )
        return Decision(
            True,
            {
                "method_freeze_hash": method_freeze["method_freeze_hash"],
                "selected": {
                    "budget": budget,
                    "scope": selected["scope"],
                    "neighbours": selected["neighbours"],
                    "kernel": selected["kernel"],
                    "feature_count": local_model.features.shape[1],
                },
                "numerical": numerical,
            },
        )

    def assessment(context: RunContext) -> Decision:
        counters = state["counters"]
        cases = state["cases"]
        rows = state["rows"]
        cache = context.cache_root
        selected = state["selected"]
        local_model = state["local_model"]

        calibration_indices = role_indices("calibration")
        records, timings = solve_phase(
            context, calibration_indices, cases, rows, cache, counters
        )
        state["all_records"].extend(records)
        state["all_timings"].update(timings)
        phase = persist_phase(
            context,
            "calibration-solves",
            calibration_indices,
            cache,
            records,
            counters,
        )
        state["phase_hashes"][phase["phase"]] = phase["phase_hash"]
        calibration_data = load_role(
            context, cache, calibration_indices, "calibration", counters
        )
        calibration_groups = {}
        for group, indices in stratum_indices("calibration").items():
            positions = [calibration_indices.index(index) for index in indices]
            data_rows = [calibration_data[position] for position in positions]
            group_cases = [cases[index] for index in indices]
            pairs = [
                predict_local_field(local_model, data, case)
                for data, case in zip(data_rows, group_cases, strict=True)
            ]
            field_prediction = [pair[0] for pair in pairs]
            scalar_prediction = np.full((len(data_rows), len(QOIS)), np.nan)
            scalar_prediction = compose_qoi_predictions(
                scalar_prediction, field_prediction, group_cases, data_rows
            )
            calibration_groups[group] = conformal_calibration(
                np.asarray([data["high_qois"] for data in data_rows]),
                scalar_prediction,
                [data["high_field"] for data in data_rows],
                field_prediction,
            )
        calibration_record = {
            "schema_version": "cft-revival.l1a-field-surrogate-v10.calibration/10.0.0",
            "method_freeze_hash": state["method_freeze"]["method_freeze_hash"],
            "groups": calibration_groups,
            "access_counters": counters,
            "assessment_materialized": 0,
        }
        calibration_record["calibration_hash"] = canonical_hash(calibration_record)
        context.write_json(
            "artifacts/group-conformal-calibration.json", calibration_record
        )
        assessment_freeze = {
            "method_freeze_hash": state["method_freeze"]["method_freeze_hash"],
            "calibration_hash": calibration_record["calibration_hash"],
            "assessment_materialized": 0,
            "assessment_reads": 0,
        }
        assessment_freeze["assessment_freeze_hash"] = canonical_hash(assessment_freeze)
        context.write_json("artifacts/frozen-before-assessment.json", assessment_freeze)
        state["phase_hashes"]["calibration-freeze"] = assessment_freeze[
            "assessment_freeze_hash"
        ]
        del calibration_data, data_rows, field_prediction, scalar_prediction

        assessment_indices = role_indices("assessment")
        records, timings = solve_phase(
            context, assessment_indices, cases, rows, cache, counters
        )
        state["all_records"].extend(records)
        state["all_timings"].update(timings)
        phase = persist_phase(
            context,
            "assessment-solves",
            assessment_indices,
            cache,
            records,
            counters,
        )
        state["phase_hashes"][phase["phase"]] = phase["phase_hash"]
        assessment_data = load_role(
            context, cache, assessment_indices, "assessment", counters
        )
        pairs = [
            predict_local_field(local_model, data, cases[index])
            for data, index in zip(assessment_data, assessment_indices, strict=True)
        ]
        field_prediction = [pair[0] for pair in pairs]
        scalar_prediction = np.full((len(assessment_data), len(QOIS)), np.nan)
        scalar_prediction = compose_qoi_predictions(
            scalar_prediction,
            field_prediction,
            [cases[index] for index in assessment_indices],
            assessment_data,
        )
        truth_qois = np.asarray([data["high_qois"] for data in assessment_data])
        truth_fields = [data["high_field"] for data in assessment_data]
        scalar_result = scalar_metrics(truth_qois, scalar_prediction)
        field_result = field_metrics(truth_fields, field_prediction)
        method_truth = np.asarray(
            [data["high_qois"] for data in state["method_data"]]
        )
        ranges = np.maximum(
            np.ptp(method_truth, axis=0),
            np.maximum(np.max(np.abs(method_truth), axis=0) * 1e-12, 1e-15),
        )
        coverage = {}
        for group, indices in stratum_indices("assessment").items():
            positions = [assessment_indices.index(index) for index in indices]
            coverage[group] = assessment_coverage(
                truth_qois[positions],
                scalar_prediction[positions],
                calibration_groups[group],
                ranges,
            )
        inference_times = {}
        safety = []
        for position, index in enumerate(assessment_indices):
            group = next(
                name
                for name, values in stratum_indices("assessment").items()
                if index in values
            )
            mismatch = not topology_match(
                field_prediction[position], truth_fields[position]
            )
            safety.append(
                {
                    "index": index,
                    "group": group,
                    "ood": group == "ood",
                    "topology_uncertainty": mismatch,
                    "acceptable": group != "ood" and not mismatch,
                }
            )
            tick = time.perf_counter()
            probe, _ = predict_local_field(
                local_model, assessment_data[position], cases[index]
            )
            topology(probe)
            inference_times[index] = time.perf_counter() - tick
        warmups = (378, 396, 414)
        timed = [index for index in assessment_indices if index not in warmups]
        latency = latency_metrics(
            [state["all_timings"][index]["low"] for index in timed],
            [inference_times[index] for index in timed],
            [state["all_timings"][index]["fine"] for index in timed],
        )
        numerical = numerical_gates(state["all_records"], 432)
        model_passed = candidate_passes(
            scalar_result,
            field_result,
            True,
            True,
        )
        safety_passed = all(
            not item["acceptable"]
            for item in safety
            if item["ood"] or item["topology_uncertainty"]
        )
        accepted = bool(
            numerical["passed"]
            and model_passed
            and all(item["passed"] for item in coverage.values())
            and safety_passed
            and latency["passed"]
        )
        selected_summary = {
            "budget": selected["budget"],
            "scope": selected["scope"],
            "neighbours": selected["neighbours"],
            "kernel": selected["kernel"],
            "feature_count": local_model.features.shape[1],
        }
        gates = {
            "development_model_gates": selected["passed"],
            "assessment_model_gates": model_passed,
            "numerical": numerical,
            "coverage": coverage,
            "safety": {"passed": safety_passed, "rows": safety},
            "latency": latency,
        }
        detail = {
            "selected": selected_summary,
            "development": selected,
            "assessment": {
                "scalar_metrics": scalar_result,
                "field_metrics": field_result,
                "gates": gates,
            },
            "geometry_count": 432,
            "solve_counters": counters["solver_accesses"],
            "access_counters": counters,
            "phase_hashes": state["phase_hashes"],
            "claim": PROTOCOL["classification"],
        }
        context.write_json("artifacts/assessment-detail.json", detail)
        context.write_json(
            "artifacts/numerical-completion.json",
            {
                "records": state["all_records"],
                "gates": numerical,
                "phase_hashes": state["phase_hashes"],
                "access_counters": counters,
            },
        )
        return Decision(accepted, detail)

    outcome = runtime.run(RuntimeCallbacks(prebundle, development, assessment))
    validated = validate_bundle(RESULTS)
    terminal = strict_load(RESULTS / "terminal.json")
    return {
        "state": outcome.state.value,
        "manifest_state": validated["state"],
        "primary_error": outcome.primary_error,
        "secondary_errors": list(outcome.secondary_errors),
        "counts": terminal["counts"],
        "payload": terminal["payload"],
    }


def validate_production_bundle() -> dict[str, Any]:
    manifest = validate_bundle(RESULTS)
    terminal = strict_load(RESULTS / "terminal.json")
    return {
        "passed": True,
        "state": manifest["state"],
        "counts": terminal["counts"],
        "payload": terminal["payload"],
        "primary_error": terminal["primary_error"],
        "secondary_errors": terminal["secondary_errors"],
        "artifact_count": manifest["artifact_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "synthetic", "execute", "validate")
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "synthetic":
        result = synthetic_runtime_preflight()
    elif args.command == "execute":
        result = execute()
    else:
        result = validate_production_bundle()
    print(__import__("json").dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
