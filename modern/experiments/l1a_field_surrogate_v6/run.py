"""Synthetic-preflighted, staged and exactly-once v6 experiment runner."""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.metadata
import inspect
import math
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
    Decision,
    ExecutionAttestation,
    ExperimentRuntime,
    RuntimeCallbacks,
    RunContext,
    validate_bundle,
)

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
    preflight_candidates,
    prolong_low,
    qois,
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
    "modern/experiments/l1a_field_surrogate_v6/",
    "modern/tests/experiments/l1a_field_surrogate_v6/",
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
ACCEPTED_RUNTIME_COMMIT = "231873d23fa242b15d0c085447a3d34ad55162a7"


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
        "schema_version": "cft-revival.l1a-field-surrogate-v6.dependency-lock/6.0.0",
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
) -> Path:
    path = cache / f"{index:03d}.npz"
    np.savez_compressed(
        path,
        row=np.asarray(row),
        low_field=np.asarray(low_field),
        high_field=np.asarray(high_field),
        low_qois=np.asarray(low_qois),
        high_qois=np.asarray(high_qois),
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
    norm = math.sqrt(max(float(np.sum(low * low)), 1e-300))
    target = high - low if family.startswith("observed_coarse") else high
    return align_vector(target, case) / norm


def build_snapshots(
    data_rows: Sequence[Mapping[str, np.ndarray]],
    cases: Sequence[Any],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    residual = family.startswith("observed_coarse")
    snapshots = np.asarray(
        [field_target(data, case, family) for data, case in zip(data_rows, cases, strict=True)]
    )
    features = np.asarray(
        [
            model_features(data["row"], qoi_decode(data["low_qois"]), residual)
            for data in data_rows
        ]
    )
    return snapshots, features


def fit_scalar(
    family: str,
    length: float,
    data_rows: Sequence[Mapping[str, np.ndarray]],
) -> SharedKernelGP:
    ratio = family == "observed_coarse_log_ratio_gp"
    features, targets = [], []
    for data in data_rows:
        features.append(model_features(data["row"], qoi_decode(data["low_qois"]), ratio))
        high = np.log(np.maximum(data["high_qois"], 1e-15))
        targets.append(
            high - np.log(np.maximum(data["low_qois"], 1e-15)) if ratio else high
        )
    return SharedKernelGP.fit(np.asarray(features), np.asarray(targets), length)


def predict_scalar(
    family: str,
    model: SharedKernelGP,
    data_rows: Sequence[Mapping[str, np.ndarray]],
) -> np.ndarray:
    ratio = family == "observed_coarse_log_ratio_gp"
    features = np.asarray(
        [model_features(data["row"], qoi_decode(data["low_qois"]), ratio) for data in data_rows]
    )
    latent = model.predict(features)
    if ratio:
        latent += np.log(np.maximum(np.asarray([data["low_qois"] for data in data_rows]), 1e-15))
    return np.exp(latent)


def predict_field(
    family: str,
    basis: WeightedPOD,
    model: SharedKernelGP,
    data: Mapping[str, np.ndarray],
    case: Any,
) -> np.ndarray:
    residual = family.startswith("observed_coarse")
    feature = model_features(data["row"], qoi_decode(data["low_qois"]), residual)[None, :]
    aligned = basis.reconstruct(model.predict(feature))[0]
    low = data["low_field"]
    norm = math.sqrt(max(float(np.sum(low * low)), 1e-300))
    physical = unalign_vector(aligned, case) * norm
    return low + physical if residual else physical


def projection_oracle(
    family: str,
    basis: WeightedPOD,
    data: Mapping[str, np.ndarray],
    case: Any,
) -> np.ndarray:
    aligned = field_target(data, case, family)[None, :]
    reconstructed = basis.reconstruct(basis.project(aligned))[0]
    low = data["low_field"]
    norm = math.sqrt(max(float(np.sum(low * low)), 1e-300))
    physical = unalign_vector(reconstructed, case) * norm
    return low + physical if family.startswith("observed_coarse") else physical


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


def candidate_passes(scalar: Mapping[str, Any], field: Mapping[str, Any], pod_ok: bool) -> bool:
    gates = PROTOCOL["gates"]
    return bool(
        pod_ok
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
    coordinate = np.linspace(0, 4 * math.pi, size)
    base = 0.2 + 0.01 * np.sin(coordinate)
    with tempfile.TemporaryDirectory(prefix="l1a-v6-synthetic-science-") as temporary:
        cache = Path(temporary)
        for index in range(144):
            row = np.asarray(designs[index].values)
            low = base * (1 + 0.001 * index)
            high = low + 0.0003 * np.cos(coordinate * (1 + index % 3))
            low_qoi = np.asarray([1 + 0.01 * column + 0.0001 * index for column in range(len(QOIS))])
            high_qoi = low_qoi * (1.001 + 0.00001 * index)
            save_checkpoint(cache, index, row, low, high, low_qoi, high_qoi)
        probe = load_checkpoint(cache, 0, "candidate", counters)
        qoi_decode(probe["low_qois"])
        executed["npz-save-load-qoi"].extend(("save_checkpoint", "load_checkpoint", "qoi_decode"))
        aligned = align_vector(probe["high_field"], case)
        unalign_vector(aligned, case)
        executed["align-unalign"].extend(("align_vector", "unalign_vector"))
        candidate_data = [
            load_checkpoint(cache, index, "candidate", counters) for index in range(128)
        ]
        method_data = [
            load_checkpoint(cache, index, "method", counters) for index in range(128, 144)
        ]
        all_candidates = []
        for budget in PROTOCOL["sampling"]["high_budgets"]:
            training = candidate_data[:budget]
            for scalar_family in PROTOCOL["models"]["scalar_families"]:
                for length in PROTOCOL["models"]["length_scales"]:
                    scalar_model = fit_scalar(scalar_family, length, training)
                    scalar_prediction = predict_scalar(scalar_family, scalar_model, method_data)
                    scalar_model.to_dict()
                    executed["scalar-high-ratio"].append(scalar_family)
                    executed["shared-kernel-fit-predict-serialize"].append(
                        f"scalar:{budget}:{scalar_family}:{length}"
                    )
                    for field_family in PROTOCOL["models"]["field_families"]:
                        snapshots, features = build_snapshots(
                            training, [case] * len(training), field_family
                        )
                        executed["snapshot-high-residual"].append(
                            f"{budget}:{field_family}"
                        )
                        basis = WeightedPOD.fit(snapshots)
                        if basis is None:
                            raise RuntimeError("synthetic low-rank POD unexpectedly failed")
                        coefficients = basis.project(snapshots)
                        basis.reconstruct(coefficients)
                        basis.to_dict()
                        executed["pod-fit-project-reconstruct-serialize-rank-fail"].append(
                            f"success:{budget}:{field_family}:rank={basis.rank}"
                        )
                        for field_length in PROTOCOL["models"]["length_scales"]:
                            field_model = SharedKernelGP.fit(features, coefficients, field_length)
                            predicted_fields = [
                                predict_field(field_family, basis, field_model, data, case)
                                for data in method_data
                            ]
                            oracle_fields = [
                                projection_oracle(field_family, basis, data, case)
                                for data in method_data
                            ]
                            field_model.to_dict()
                            executed["all-budgets-lengths-families"].append(
                                f"{budget}:{scalar_family}:{length}:{field_family}:{field_length}"
                            )
                            executed["shared-kernel-fit-predict-serialize"].append(
                                f"field:{budget}:{field_family}:{field_length}"
                            )
                            executed["field-predict-projection-oracle"].append(
                                f"{field_family}:{field_length}"
                            )
                            truth_qois = np.asarray([data["high_qois"] for data in method_data])
                            truth_fields = [data["high_field"] for data in method_data]
                            baseline = field_metrics(
                                truth_fields, [data["low_field"] for data in method_data]
                            )
                            oracle = field_metrics(truth_fields, oracle_fields)
                            coefficient = field_metrics(truth_fields, predicted_fields)
                            scalar = scalar_metrics(truth_qois, scalar_prediction)
                            executed["baseline-oracle-coefficient-metrics"].append(
                                f"{baseline['count']}:{oracle['count']}:{coefficient['count']}"
                            )
                            all_candidates.append(
                                {
                                    "budget": budget,
                                    "scalar_family": scalar_family,
                                    "field_family": field_family,
                                    "pod_rank": basis.rank,
                                    "scalar_metrics": scalar,
                                    "field_metrics": coefficient,
                                    "passed": candidate_passes(scalar, coefficient, True),
                                }
                            )
        rank_failure = np.zeros((66, size))
        rank_failure[:, :66] = np.eye(66)
        if WeightedPOD.fit(rank_failure) is not None:
            raise RuntimeError("rank-cap failure path did not fail closed")
        executed["pod-fit-project-reconstruct-serialize-rank-fail"].append("rank-cap-failure")
        passing = dict(all_candidates[0])
        passing["passed"] = True
        passing["scalar_metrics"] = {"worst_nrmse": 0.0}
        passing["field_metrics"] = {"worst_l2": 0.0}
        failing = dict(all_candidates[0]); failing["passed"] = False
        if select_candidate((passing, failing)) is None or select_candidate((failing,)) is not None:
            raise RuntimeError("selection branch coverage failed")
        executed["selection-pass-fail-none"].extend(("passing", "failing", "no-model"))
        truth_qois = np.asarray([data["high_qois"] for data in method_data])
        predicted_qois = truth_qois * 1.001
        truth_fields = [data["high_field"] for data in method_data]
        predicted_fields = [field * 1.001 for field in truth_fields]
        calibration = conformal_calibration(
            truth_qois, predicted_qois, truth_fields, predicted_fields
        )
        ranges = np.maximum(np.ptp(truth_qois, axis=0), 1e-9)
        assessment_coverage(truth_qois, predicted_qois, calibration, ranges)
        topology(predicted_fields[0])
        topology_match(predicted_fields[0], truth_fields[0])
        latency_metrics([0.01] * 16, [0.001] * 16, [0.05] * 16)
        executed["calibration-conformal-assessment-topology-latency"].extend(
            (
                "conformal_calibration",
                "assessment_coverage",
                "topology",
                "topology_match",
                "latency_metrics",
            )
        )
    empty = sorted(name for name, paths in executed.items() if not paths)
    if empty:
        raise RuntimeError(f"incomplete synthetic path coverage: {empty}")
    path_functions = (
        save_checkpoint,
        load_checkpoint,
        qoi_decode,
        build_snapshots,
        align_vector,
        unalign_vector,
        fit_scalar,
        predict_scalar,
        SharedKernelGP.fit,
        SharedKernelGP.predict,
        SharedKernelGP.to_dict,
        WeightedPOD.fit,
        WeightedPOD.project,
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
    with tempfile.TemporaryDirectory(prefix="l1a-v6-callback-matrix-") as temporary:
        matrix_root = Path(temporary)
        for index, expected in enumerate(expected_states):
            root = matrix_root / f"{index}-{expected.value}"
            runtime = ExperimentRuntime(
                experiment_id=f"l1a-field-surrogate-v6-synthetic-{expected.value}",
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
        for index in range(240)
        if (
            initial[index].geometry_sha256 != rebuilt[index].geometry_sha256
            or initial[index].source_sha256 != rebuilt[index].source_sha256
        )
    ]
    if hash_failures:
        raise RuntimeError(f"frozen rebuild mismatch: {hash_failures}")
    rejected = [record for record in records if not record["valid"]]
    geometry = {
        "schema_version": "cft-revival.l1a-field-surrogate-v6.geometry-preflight/6.0.0",
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
    partition = {
        "schema_version": "cft-revival.l1a-field-surrogate-v6.partitions/6.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "geometry_preflight_hash": geometry["geometry_preflight_hash"],
        "frozen_raw_indices": list(frozen),
        "frozen_design_ids": [designs[index].design_id for index in frozen],
        "frozen_design_rows": [list(designs[index].values) for index in frozen],
        "roles": {
            role: list(role_indices(role))
            for role in ("candidate", "method", "calibration", "assessment")
        },
        "calibration_strata": {name: list(rows) for name, rows in stratum_indices("calibration").items()},
        "assessment_strata": {name: list(rows) for name, rows in stratum_indices("assessment").items()},
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
        "schema_version": "cft-revival.l1a-field-surrogate-v6.synthetic-preflight/6.0.0",
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
        raise RuntimeError("HEAD is not v6 preregistration")
    if _git("rev-parse", REMOTE) != head:
        raise RuntimeError("remote v6 branch is not the preregistration")
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
        command="python -m experiments.l1a_field_surrogate_v6.run execute",
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
        "schema_version": "cft-revival.l1a-field-surrogate-v6.phase/6.0.0",
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
    truth_qois = np.asarray([data["high_qois"] for data in method_data])
    truth_fields = [data["high_field"] for data in method_data]
    baseline = field_metrics(truth_fields, [data["low_field"] for data in method_data])
    candidates = []
    fitted: dict[tuple[Any, ...], Any] = {}
    lengths = PROTOCOL["models"]["length_scales"]
    for budget in PROTOCOL["sampling"]["high_budgets"]:
        training = candidate_data[:budget]
        training_cases = candidate_cases[:budget]
        scalar_results = {}
        for scalar_family in PROTOCOL["models"]["scalar_families"]:
            for scalar_length in lengths:
                model = fit_scalar(scalar_family, scalar_length, training)
                counters["model_fit_accesses"] += 1
                fitted[("scalar", budget, scalar_family, scalar_length)] = model
                prediction = predict_scalar(scalar_family, model, method_data)
                scalar_results[(scalar_family, scalar_length)] = scalar_metrics(truth_qois, prediction)
        field_results = {}
        for field_family in PROTOCOL["models"]["field_families"]:
            snapshots, features = build_snapshots(training, training_cases, field_family)
            basis = WeightedPOD.fit(snapshots)
            fitted[("basis", budget, field_family)] = basis
            if basis is None:
                continue
            coefficients = basis.project(snapshots)
            oracle_prediction = [
                projection_oracle(field_family, basis, data, case)
                for data, case in zip(method_data, method_cases, strict=True)
            ]
            oracle = field_metrics(truth_fields, oracle_prediction)
            for field_length in lengths:
                model = SharedKernelGP.fit(features, coefficients, field_length)
                counters["model_fit_accesses"] += 1
                fitted[("field", budget, field_family, field_length)] = model
                prediction = [
                    predict_field(field_family, basis, model, data, case)
                    for data, case in zip(method_data, method_cases, strict=True)
                ]
                field_results[(field_family, field_length)] = {
                    "coarse_baseline": baseline,
                    "projection_oracle": oracle,
                    "coefficient_regression": field_metrics(truth_fields, prediction),
                    "pod_rank": basis.rank,
                    "pod_retained": basis.retained,
                }
        for (scalar_family, scalar_length), scalar in scalar_results.items():
            for (field_family, field_length), field in field_results.items():
                regression = field["coefficient_regression"]
                candidate = {
                    "budget": budget,
                    "scalar_family": scalar_family,
                    "scalar_length": scalar_length,
                    "field_family": field_family,
                    "field_length": field_length,
                    "pod_rank": field["pod_rank"],
                    "pod_retained": field["pod_retained"],
                    "scalar_metrics": scalar,
                    "field_metrics": regression,
                    "coarse_baseline": field["coarse_baseline"],
                    "projection_oracle": field["projection_oracle"],
                }
                candidate["passed"] = candidate_passes(
                    scalar, regression, field["pod_retained"] >= PROTOCOL["models"]["pod_retained_energy_min"]
                )
                candidates.append(candidate)
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
    basis: WeightedPOD,
    model: SharedKernelGP,
    data_rows: Sequence[Mapping[str, np.ndarray]],
    cases: Sequence[Any],
) -> list[np.ndarray]:
    return [
        predict_field(selected["field_family"], basis, model, data, case)
        for data, case in zip(data_rows, cases, strict=True)
    ]



def execute() -> dict[str, Any]:
    """Execute the single preregistered attempt through ExperimentRuntime."""

    verified = verify_before_runtime()
    runtime = ExperimentRuntime(
        experiment_id="l1a-field-surrogate-v6",
        result_root=RESULTS,
        cache_root=CACHE,
        attestation=verified.attestation,
        producer=execute,
        source_root=REPO / "modern",
    )
    state: dict[str, Any] = {
        "counters": new_counters(),
        "phase_hashes": {},
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
            "schema_version": "cft-revival.l1a-field-surrogate-v6.provenance/6.0.0",
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
        initial_indices = tuple(range(144))
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
        selected = select_candidate(candidates)
        method_freeze = {
            "schema_version": "cft-revival.l1a-field-surrogate-v6.method-freeze/6.0.0",
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
        numerical = numerical_gates(state["all_records"], 144)
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
                    "geometry_count": 240,
                    "solve_counters": counters["solver_accesses"],
                    "access_counters": counters,
                    "claim": PROTOCOL["classification"],
                },
            )

        budget = selected["budget"]
        scalar_model = fitted[
            ("scalar", budget, selected["scalar_family"], selected["scalar_length"])
        ]
        basis = fitted[("basis", budget, selected["field_family"])]
        field_model = fitted[
            ("field", budget, selected["field_family"], selected["field_length"])
        ]
        if basis is None:
            raise RuntimeError("selected candidate has no POD basis")
        state.update(
            {
                "scalar_model": scalar_model,
                "basis": basis,
                "field_model": field_model,
            }
        )
        context.write_json(
            "artifacts/selected-model.json",
            {
                "selected": selected,
                "scalar_model": scalar_model.to_dict(),
                "field_basis": basis.to_dict(),
                "field_model": field_model.to_dict(),
            },
        )
        return Decision(
            True,
            {
                "method_freeze_hash": method_freeze["method_freeze_hash"],
                "selected": {
                    "budget": budget,
                    "scalar_family": selected["scalar_family"],
                    "scalar_length": selected["scalar_length"],
                    "field_family": selected["field_family"],
                    "field_length": selected["field_length"],
                    "pod_rank": basis.rank,
                    "pod_retained": basis.retained,
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
        scalar_model = state["scalar_model"]
        basis = state["basis"]
        field_model = state["field_model"]

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
            scalar_prediction = predict_scalar(
                selected["scalar_family"], scalar_model, data_rows
            )
            field_prediction = predict_selected_fields(
                selected, basis, field_model, data_rows, group_cases
            )
            calibration_groups[group] = conformal_calibration(
                np.asarray([data["high_qois"] for data in data_rows]),
                scalar_prediction,
                [data["high_field"] for data in data_rows],
                field_prediction,
            )
        calibration_record = {
            "schema_version": "cft-revival.l1a-field-surrogate-v6.calibration/6.0.0",
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
        scalar_prediction = predict_scalar(
            selected["scalar_family"], scalar_model, assessment_data
        )
        field_prediction = predict_selected_fields(
            selected,
            basis,
            field_model,
            assessment_data,
            [cases[index] for index in assessment_indices],
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
            predict_scalar(
                selected["scalar_family"],
                scalar_model,
                [assessment_data[position]],
            )
            probe = predict_field(
                selected["field_family"],
                basis,
                field_model,
                assessment_data[position],
                cases[index],
            )
            topology(probe)
            inference_times[index] = time.perf_counter() - tick
        warmups = (192, 208, 224)
        timed = [index for index in assessment_indices if index not in warmups]
        latency = latency_metrics(
            [state["all_timings"][index]["low"] for index in timed],
            [inference_times[index] for index in timed],
            [state["all_timings"][index]["fine"] for index in timed],
        )
        numerical = numerical_gates(state["all_records"], 240)
        model_passed = candidate_passes(
            scalar_result,
            field_result,
            basis.retained >= PROTOCOL["models"]["pod_retained_energy_min"],
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
            "scalar_family": selected["scalar_family"],
            "scalar_length": selected["scalar_length"],
            "field_family": selected["field_family"],
            "field_length": selected["field_length"],
            "pod_rank": basis.rank,
            "pod_retained": basis.retained,
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
            "geometry_count": 240,
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
