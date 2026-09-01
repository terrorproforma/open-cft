"""Prepare and execute the staged, exactly-once field-surrogate v3 protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import warp as wp

from .experiment import (
    HIGH_DOMAIN,
    INPUT_NAMES,
    QOIS,
    SharedKernelGP,
    WeightedPOD,
    align_vector,
    canonical_hash,
    construct_geometry,
    design_row,
    field_energy,
    field_vector,
    metric_summary,
    model_features,
    numerical_record,
    percentile,
    preflight_candidates,
    prolong_low,
    qois,
    raw_designs,
    rebuild_frozen,
    role_indices,
    scalar_predictions,
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
    RESULTS,
    ROOT,
    SYNTHETIC_PREFLIGHT,
    exact_rank,
    strict_load,
    verify_json,
    write_json,
)

SUBJECT = PROTOCOL["integrity"]["protocol_commit_subject"]
REMOTE = PROTOCOL["integrity"]["remote_branch"]
PREFIXES = (
    "modern/experiments/l1a_field_surrogate_v3/",
    "modern/tests/experiments/l1a_field_surrogate_v3/",
)
DEPENDENCY_ROOTS = (
    "modern/src/cft_revival/fields/",
    "modern/src/cft_revival/geometry/",
    "modern/src/cft_revival/magnetics/",
    "modern/src/cft_revival/optimization/",
    "modern/experiments/l1a_geometry_sweep_v2/",
    "modern/experiments/l1a_field_surrogate_v1/",
    "modern/experiments/l1a_field_surrogate_v2/",
)


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=REPO, check=check, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _closure() -> list[dict[str, str]]:
    entries = []
    for line in _git("ls-tree", "-r", "HEAD", "--", *(PREFIXES + DEPENDENCY_ROOTS)).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        entries.append({"path": path, "mode": mode, "type": kind, "blob": blob})
    return sorted(entries, key=lambda item: item["path"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dependency_lock() -> dict[str, Any]:
    wp.init()
    python_path = Path(sys.executable)
    numpy_path = Path(np.__file__).resolve()
    warp_path = Path(wp.__file__).resolve()
    pyproject = REPO / "modern" / "pyproject.toml"
    value = {
        "schema_version": "cft-revival.l1a-field-surrogate-v3.dependency-lock/3.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "python": {
            "version": sys.version,
            "executable": str(python_path),
            "executable_sha256": _file_sha256(python_path),
        },
        "packages": {
            "numpy": {
                "version": np.__version__,
                "distribution_version": importlib.metadata.version("numpy"),
                "module_file": str(numpy_path),
                "module_file_sha256": _file_sha256(numpy_path),
            },
            "warp-lang": {
                "version": wp.__version__,
                "distribution_version": importlib.metadata.version("warp-lang"),
                "module_file": str(warp_path),
                "module_file_sha256": _file_sha256(warp_path),
            },
        },
        "cuda": {
            "toolkit_version": list(wp.get_cuda_toolkit_version()),
            "driver_runtime_version": list(wp.get_cuda_driver_version()),
            "nvidia_smi": subprocess.run(
                ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"),
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
        },
        "pyproject": {
            "path": "modern/pyproject.toml",
            "sha256": _file_sha256(pyproject),
            "git_blob": _git("hash-object", "modern/pyproject.toml"),
        },
    }
    value["dependency_lock_hash"] = canonical_hash(value)
    return value


def prepare() -> dict[str, Any]:
    records, valid = preflight_candidates()
    frozen = select_frozen(valid)
    rebuilt, rebuild_records = rebuild_frozen(frozen)
    initial = {index: valid[raw_index] for index, raw_index in enumerate(frozen)}
    failures = [
        index
        for index in range(240)
        if (
            initial[index].geometry_sha256 != rebuilt[index].geometry_sha256
            or initial[index].source_sha256 != rebuilt[index].source_sha256
        )
    ]
    if failures:
        raise RuntimeError(f"frozen rebuild mismatch: {failures}")
    rejected = [record for record in records if not record["valid"]]
    geometry = {
        "schema_version": "cft-revival.l1a-field-surrogate-v3.geometry-preflight/3.0.0",
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
    write_json(GEOMETRY_PREFLIGHT, geometry)
    designs = raw_designs()
    partition = {
        "schema_version": "cft-revival.l1a-field-surrogate-v3.partitions/3.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "geometry_preflight_hash": geometry["geometry_preflight_hash"],
        "frozen_raw_indices": list(frozen),
        "frozen_design_ids": [designs[index].design_id for index in frozen],
        "frozen_design_rows": [list(designs[index].values) for index in frozen],
        "roles": {
            name: list(role_indices(name))
            for name in ("candidate", "method_selection", "final_calibration", "single_use_assessment")
        },
        "calibration_strata": {name: list(rows) for name, rows in stratum_indices("calibration").items()},
        "assessment_strata": {name: list(rows) for name, rows in stratum_indices("assessment").items()},
        "prior_role_coordinate_intersection_count": 0,
        "field_solver_access_count": 0,
    }
    partition["partition_hash"] = canonical_hash(partition)
    write_json(PARTITIONS, partition)
    lock = _dependency_lock()
    write_json(DEPENDENCY_LOCK, lock)
    synthetic = {
        "schema_version": "cft-revival.l1a-field-surrogate-v3.synthetic-preflight/3.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "geometry_preflight_hash": geometry["geometry_preflight_hash"],
        "partition_hash": partition["partition_hash"],
        "dependency_lock_hash": lock["dependency_lock_hash"],
        "frozen_rebuild_count": 240,
        "frozen_hash_failure_count": 0,
        "field_solver_access_count": 0,
        "qoi_label_access_count": 0,
        "synthetic_linear_algebra_only": True,
        "passed": True,
    }
    synthetic["synthetic_preflight_hash"] = canonical_hash(synthetic)
    write_json(SYNTHETIC_PREFLIGHT, synthetic)
    return {
        "passed": True,
        "raw_valid": len(valid),
        "raw_rejected": len(rejected),
        "raw_corrected": geometry["corrected_count"],
        "frozen": 240,
        "geometry_preflight_hash": geometry["geometry_preflight_hash"],
        "dependency_lock_hash": lock["dependency_lock_hash"],
    }


def _bind() -> tuple[str, list[dict[str, str]]]:
    head = _git("rev-parse", "HEAD")
    if _git("symbolic-ref", "-q", "HEAD", check=False):
        raise RuntimeError("execution requires detached preregistration")
    if _git("show", "-s", "--format=%s", head) != SUBJECT:
        raise RuntimeError("HEAD is not v3 preregistration")
    if _git("rev-parse", REMOTE) != head:
        raise RuntimeError("remote experiment branch does not equal preregistration")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("detached worktree is not clean")
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise RuntimeError("preregistration is not exact-path isolated")
    for path in (GEOMETRY_PREFLIGHT, PARTITIONS, SYNTHETIC_PREFLIGHT, DEPENDENCY_LOCK):
        verify_json(path)
    if _dependency_lock() != strict_load(DEPENDENCY_LOCK):
        raise RuntimeError("runtime differs from dependency lock")
    return head, _closure()


def _acquire_lock(commit: str) -> Path:
    common = Path(_git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (REPO / common).resolve()
    path = common / "l1a-field-surrogate-v3.execution.lock"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(commit + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _save_checkpoint(
    cache: Path,
    index: int,
    row: Sequence[float],
    low_field: np.ndarray,
    high_field: np.ndarray,
    low_qois: Mapping[str, float],
    high_qois: Mapping[str, float],
    chamber_length: float,
) -> None:
    np.savez_compressed(
        cache / f"{index:03d}.npz",
        row=np.asarray(row),
        low_field=low_field,
        high_field=high_field,
        low_qois=np.asarray([low_qois[name] for name in QOIS]),
        high_qois=np.asarray([high_qois[name] for name in QOIS]),
        chamber_length=np.asarray([chamber_length]),
    )


def _load(cache: Path, index: int) -> dict[str, Any]:
    with np.load(cache / f"{index:03d}.npz") as data:
        return {name: np.array(data[name]) for name in data.files}


def _qoi_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(QOIS, values, strict=True)}


def _solve_stage(
    indices: Sequence[int],
    cases: Mapping[int, Any],
    rows: Sequence[Sequence[float]],
    cache: Path,
    counters: dict[str, int],
    numerical: list[dict[str, Any]],
    timings: dict[int, dict[str, float]],
) -> None:
    for index in indices:
        tick = time.perf_counter()
        counters["low_solver_accesses"] += 1
        low = solve_fidelity(cases[index], "low")
        low_elapsed = time.perf_counter() - tick
        counters["low_completed"] += 1
        low_values = qois(cases[index], low, "low")
        tick = time.perf_counter()
        counters["fine_solver_accesses"] += 1
        high = solve_fidelity(cases[index], "high")
        high_elapsed = time.perf_counter() - tick
        counters["fine_completed"] += 1
        high_values = qois(cases[index], high, "high")
        low_vector = prolong_low(low)
        high_vector = field_vector(high)
        _save_checkpoint(
            cache, index, rows[index], low_vector, high_vector, low_values, high_values,
            cases[index].geometry.chamber.length_m,
        )
        low_record = numerical_record(cases[index], low, low_values, index, "low")
        high_record = numerical_record(cases[index], high, high_values, index, "high")
        if low_record["pairing_sha256"] != high_record["pairing_sha256"]:
            raise RuntimeError("fidelity pairing mismatch")
        numerical.extend((low_record, high_record))
        timings[index] = {"low": low_elapsed, "fine": high_elapsed}


def _case_with_length(case: Any, length: float) -> Any:
    if abs(case.geometry.chamber.length_m - length) > 1e-15:
        raise RuntimeError("checkpoint chamber identity mismatch")
    return case


def _build_snapshots(
    cache: Path,
    cases: Mapping[int, Any],
    indices: Sequence[int],
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshots, features, low_qoi_rows, high_qoi_rows = [], [], [], []
    residual = family == "observed_coarse_residual_aligned_pod_gp"
    for index in indices:
        data = _load(cache, index)
        low = data["low_field"]
        high = data["high_field"]
        norm = math.sqrt(max(float(np.sum(low * low)), 1e-300))
        target = high - low if residual else high
        snapshots.append(align_vector(target, cases[index]) / norm)
        low_dict = _qoi_dict(data["low_qois"])
        features.append(model_features(data["row"], low_dict, residual))
        low_qoi_rows.append(data["low_qois"])
        high_qoi_rows.append(data["high_qois"])
    return np.asarray(snapshots), np.asarray(features), np.asarray(low_qoi_rows), np.asarray(high_qoi_rows)


def _predict_field(
    family: str,
    basis: WeightedPOD,
    model: SharedKernelGP,
    data: Mapping[str, np.ndarray],
    case: Any,
) -> np.ndarray:
    residual = family == "observed_coarse_residual_aligned_pod_gp"
    low_dict = _qoi_dict(data["low_qois"])
    feature = model_features(data["row"], low_dict, residual)[None, :]
    coefficients = model.predict(feature)
    normalized = basis.reconstruct(coefficients)[0]
    low = data["low_field"]
    norm = math.sqrt(max(float(np.sum(low * low)), 1e-300))
    physical = unalign_vector(normalized, case) * norm
    return low + physical if residual else physical


def _projection_oracle(
    family: str,
    basis: WeightedPOD,
    data: Mapping[str, np.ndarray],
    case: Any,
) -> np.ndarray:
    residual = family == "observed_coarse_residual_aligned_pod_gp"
    low, high = data["low_field"], data["high_field"]
    norm = math.sqrt(max(float(np.sum(low * low)), 1e-300))
    target = high - low if residual else high
    normalized = align_vector(target, case)[None, :] / norm
    reconstructed = basis.reconstruct(basis.project(normalized))[0]
    physical = unalign_vector(reconstructed, case) * norm
    return low + physical if residual else physical


def _fit_scalar(
    family: str,
    length: float,
    cache: Path,
    indices: Sequence[int],
) -> SharedKernelGP:
    features, targets = [], []
    ratio = family == "observed_coarse_log_ratio_matern52_gp"
    for index in indices:
        data = _load(cache, index)
        low = _qoi_dict(data["low_qois"])
        features.append(model_features(data["row"], low, ratio))
        high_log = np.log(np.maximum(data["high_qois"], 1e-15))
        targets.append(high_log - np.log(np.maximum(data["low_qois"], 1e-15)) if ratio else high_log)
    return SharedKernelGP.fit(np.asarray(features), np.asarray(targets), length)


def _predict_scalar(
    family: str,
    model: SharedKernelGP,
    data_rows: Sequence[Mapping[str, np.ndarray]],
) -> np.ndarray:
    ratio = family == "observed_coarse_log_ratio_matern52_gp"
    features = np.asarray(
        [model_features(data["row"], _qoi_dict(data["low_qois"]), ratio) for data in data_rows]
    )
    coarse = [_qoi_dict(data["low_qois"]) for data in data_rows]
    return scalar_predictions(family, model, features, coarse)


def _passes(metrics: Mapping[str, Any], retained: bool) -> bool:
    gates = PROTOCOL["gates"]
    return bool(
        retained
        and metrics["worst_scalar_nrmse"] <= gates["scalar"]["nrmse_max"]
        and metrics["worst_scalar_error"] <= gates["scalar"]["worst_range_normalized_error_max"]
        and metrics["worst_field_l2"] <= gates["field"]["relative_l2_max"]
        and metrics["worst_field_energy"] <= gates["field"]["relative_energy_error_max"]
        and metrics["topology_matches"] == 16
    )


def _numerical_gates(records: Sequence[Mapping[str, Any]], expected_rows: int) -> dict[str, Any]:
    limits = PROTOCOL["gates"]["numerics"]
    checks = {
        "residual": max(item["relative_residual_l2"] for item in records) <= limits["relative_residual_l2_max"],
        "boundary": max(item["boundary_to_peak_ratio"] for item in records) <= limits["boundary_to_peak_ratio_max"],
        "source": max(item["source_representation_error"] for item in records) <= limits["source_representation_error_max"],
        "flux": max(item["flux_identity_t_per_m"] for item in records) <= limits["flux_identity_t_per_m_max"],
        "zero_frozen_execution_failures": len(records) == 2 * expected_rows,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "max_residual": max(item["relative_residual_l2"] for item in records),
            "max_boundary": max(item["boundary_to_peak_ratio"] for item in records),
            "max_source_error": max(item["source_representation_error"] for item in records),
            "max_flux_identity": max(item["flux_identity_t_per_m"] for item in records),
        },
    }


def execute() -> dict[str, Any]:
    commit, closure = _bind()
    lock = _acquire_lock(commit)
    counters = {
        "low_solver_accesses": 0,
        "low_completed": 0,
        "fine_solver_accesses": 0,
        "fine_completed": 0,
        "model_fit_accesses": 0,
        "method_label_accesses": 0,
        "calibration_label_accesses": 0,
        "assessment_label_accesses": 0,
    }
    stage = "binding"
    started = time.perf_counter()
    write_json(
        RESULTS / "execution-lock.json",
        {"state": "claimed-once", "commit": commit, "lock_file": lock.name, "protocol_hash": PROTOCOL_HASH},
    )
    provenance = {
        "schema_version": "cft-revival.l1a-field-surrogate-v3.provenance/3.0.0",
        "commit": commit,
        "protocol_hash": PROTOCOL_HASH,
        "partition_hash": strict_load(PARTITIONS)["partition_hash"],
        "geometry_preflight_hash": strict_load(GEOMETRY_PREFLIGHT)["geometry_preflight_hash"],
        "dependency_lock_hash": strict_load(DEPENDENCY_LOCK)["dependency_lock_hash"],
        "git_blob_closure": closure,
        "git_blob_closure_hash": canonical_hash(closure),
        "solver_accesses_at_write": 0,
    }
    provenance["provenance_hash"] = canonical_hash(provenance)
    write_json(RESULTS / "provenance-closure.json", provenance)
    verify_json(RESULTS / "provenance-closure.json")
    cache = RESULTS / ".working"
    cache.mkdir()
    numerical: list[dict[str, Any]] = []
    timings: dict[int, dict[str, float]] = {}
    try:
        partition = strict_load(PARTITIONS)
        raw_indices = tuple(int(value) for value in partition["frozen_raw_indices"])
        cases, rebuilt = rebuild_frozen(raw_indices)
        expected = strict_load(GEOMETRY_PREFLIGHT)["frozen_rebuild_records"]
        if [
            (item["geometry_sha256"], item["source_sha256"]) for item in rebuilt
        ] != [
            (item["geometry_sha256"], item["source_sha256"]) for item in expected
        ]:
            raise RuntimeError("execution rebuild differs from frozen geometry")
        rows = tuple(tuple(float(value) for value in row) for row in partition["frozen_design_rows"])

        stage = "candidate-and-method-solves"
        _solve_stage(tuple(range(144)), cases, rows, cache, counters, numerical, timings)
        if counters["calibration_label_accesses"] or counters["assessment_label_accesses"]:
            raise RuntimeError("future-role access occurred before method freeze")

        stage = "method-selection"
        method_indices = tuple(range(128, 144))
        method_data = [_load(cache, index) for index in method_indices]
        method_truth_qois = np.asarray([data["high_qois"] for data in method_data])
        method_truth_fields = [data["high_field"] for data in method_data]
        candidates = []
        fitted: dict[tuple[Any, ...], Any] = {}
        lengths = PROTOCOL["scalar_model"]["shared_kernel_length_scales"]
        for budget in PROTOCOL["sampling"]["high_training_budgets"]:
            train_indices = tuple(range(budget))
            for scalar_family in PROTOCOL["scalar_model"]["families"]:
                for scalar_length in lengths:
                    scalar_model = _fit_scalar(scalar_family, scalar_length, cache, train_indices)
                    counters["model_fit_accesses"] += 1
                    fitted[("scalar", budget, scalar_family, scalar_length)] = scalar_model
                    scalar_method = _predict_scalar(scalar_family, scalar_model, method_data)
                    for field_family in PROTOCOL["field_model"]["families"]:
                        snapshots, features, _, _ = _build_snapshots(cache, cases, train_indices, field_family)
                        basis_key = ("basis", budget, field_family)
                        if basis_key not in fitted:
                            fitted[basis_key] = WeightedPOD.fit(snapshots)
                        basis = fitted[basis_key]
                        if basis is None:
                            candidates.append(
                                {
                                    "budget": budget,
                                    "scalar_family": scalar_family,
                                    "scalar_length": scalar_length,
                                    "field_family": field_family,
                                    "field_length": None,
                                    "pod_status": "rank-cap-failed",
                                    "passed": False,
                                }
                            )
                            continue
                        coefficients = basis.project(snapshots)
                        for field_length in lengths:
                            field_model = SharedKernelGP.fit(features, coefficients, field_length)
                            counters["model_fit_accesses"] += 1
                            key = ("field", budget, field_family, field_length)
                            fitted[key] = field_model
                            predictions = [
                                _predict_field(field_family, basis, field_model, data, cases[index])
                                for index, data in zip(method_indices, method_data, strict=True)
                            ]
                            oracle = [
                                _projection_oracle(field_family, basis, data, cases[index])
                                for index, data in zip(method_indices, method_data, strict=True)
                            ]
                            metrics = metric_summary(
                                method_truth_qois, scalar_method, method_truth_fields, predictions
                            )
                            oracle_metrics = metric_summary(
                                method_truth_qois, method_truth_qois, method_truth_fields, oracle
                            )
                            baseline_metrics = metric_summary(
                                method_truth_qois,
                                np.asarray([data["low_qois"] for data in method_data]),
                                method_truth_fields,
                                [data["low_field"] for data in method_data],
                            )
                            candidate = {
                                "budget": budget,
                                "scalar_family": scalar_family,
                                "scalar_length": scalar_length,
                                "field_family": field_family,
                                "field_length": field_length,
                                "pod_rank": basis.rank,
                                "pod_retained_energy": basis.retained,
                                "coarse_baseline": baseline_metrics,
                                "pod_projection_oracle": oracle_metrics,
                                "coefficient_regression": metrics,
                            }
                            candidate["passed"] = _passes(metrics, basis.retained >= 0.995)
                            candidates.append(candidate)
        counters["method_label_accesses"] = 1
        passing = [candidate for candidate in candidates if candidate["passed"]]
        selected = min(
            passing,
            key=lambda item: (
                item["budget"],
                item["coefficient_regression"]["worst_scalar_nrmse"],
                item["coefficient_regression"]["worst_field_l2"],
                item["pod_rank"],
                item["scalar_family"],
                item["field_family"],
            ),
            default=None,
        )
        frozen = {
            "schema_version": "cft-revival.l1a-field-surrogate-v3.method-freeze/3.0.0",
            "candidates": candidates,
            "learning_curves": candidates,
            "selected": selected,
            "stage_access_counters": dict(counters),
            "calibration_label_accesses": 0,
            "assessment_label_accesses": 0,
        }
        frozen["method_freeze_hash"] = canonical_hash(frozen)
        write_json(RESULTS / "frozen-method-selection.json", frozen)
        if selected is None:
            terminal = {
                "schema_version": "cft-revival.l1a-field-surrogate-v3.terminal/3.0.0",
                "status": "failed-development-selection-gates",
                "valid_prospective_result": True,
                "stage_access_counters": counters,
                "numerical": _numerical_gates(numerical, 144),
                "calibration_accessed": False,
                "assessment_accessed": False,
            }
            terminal["terminal_hash"] = canonical_hash(terminal)
            write_json(RESULTS / "terminal-result.json", terminal)
            shutil.rmtree(cache)
            from .validate import validate_bundle

            validate_bundle()
            return terminal

        budget = selected["budget"]
        scalar_family = selected["scalar_family"]
        scalar_length = selected["scalar_length"]
        field_family = selected["field_family"]
        field_length = selected["field_length"]
        scalar_model = fitted[("scalar", budget, scalar_family, scalar_length)]
        basis = fitted[("basis", budget, field_family)]
        field_model = fitted[("field", budget, field_family, field_length)]
        write_json(
            RESULTS / "selected-model.json",
            {
                "selected": selected,
                "scalar_model": scalar_model.to_dict(),
                "field_basis": basis.to_dict(),
                "field_coefficient_model": field_model.to_dict(),
            },
        )

        stage = "calibration-solves"
        calibration_indices = role_indices("final_calibration")
        _solve_stage(calibration_indices, cases, rows, cache, counters, numerical, timings)
        counters["calibration_label_accesses"] = 1
        calibration: dict[str, Any] = {}
        for group, indices in stratum_indices("calibration").items():
            data_rows = [_load(cache, index) for index in indices]
            scalar_prediction = _predict_scalar(scalar_family, scalar_model, data_rows)
            scalar_truth = np.asarray([data["high_qois"] for data in data_rows])
            field_prediction = [
                _predict_field(field_family, basis, field_model, data, cases[index])
                for index, data in zip(indices, data_rows, strict=True)
            ]
            scalar_residual = np.abs(scalar_prediction - scalar_truth)
            rank = exact_rank(len(indices), PROTOCOL["uncertainty"]["nominal_coverage"])
            field_residual = sorted(
                float(np.linalg.norm(prediction - data["high_field"]) / max(np.linalg.norm(data["high_field"]), 1e-300))
                for prediction, data in zip(field_prediction, data_rows, strict=True)
            )
            calibration[group] = {
                "count": len(indices),
                "exact_rank": rank,
                "scalar_radii": np.sort(scalar_residual, axis=0)[rank - 1].tolist(),
                "field_relative_l2_radius": field_residual[rank - 1],
            }
        calibration_record = {
            "schema_version": "cft-revival.l1a-field-surrogate-v3.calibration/3.0.0",
            "method_freeze_hash": frozen["method_freeze_hash"],
            "groups": calibration,
            "assessment_label_accesses": 0,
            "stage_access_counters": dict(counters),
        }
        calibration_record["calibration_hash"] = canonical_hash(calibration_record)
        write_json(RESULTS / "group-conformal-calibration.json", calibration_record)
        assessment_freeze = {
            "method_freeze_hash": frozen["method_freeze_hash"],
            "calibration_hash": calibration_record["calibration_hash"],
            "assessment_label_accesses": 0,
        }
        assessment_freeze["assessment_freeze_hash"] = canonical_hash(assessment_freeze)
        write_json(RESULTS / "frozen-before-assessment.json", assessment_freeze)

        stage = "single-use-assessment-solves"
        assessment_indices = role_indices("single_use_assessment")
        _solve_stage(assessment_indices, cases, rows, cache, counters, numerical, timings)
        counters["assessment_label_accesses"] = 1
        all_data = [_load(cache, index) for index in assessment_indices]
        scalar_assessment = _predict_scalar(scalar_family, scalar_model, all_data)
        field_assessment = [
            _predict_field(field_family, basis, field_model, data, cases[index])
            for index, data in zip(assessment_indices, all_data, strict=True)
        ]
        truth_qois = np.asarray([data["high_qois"] for data in all_data])
        truth_fields = [data["high_field"] for data in all_data]
        metrics = metric_summary(truth_qois, scalar_assessment, truth_fields, field_assessment)
        method_ranges = np.maximum(
            np.ptp(method_truth_qois, axis=0),
            np.maximum(np.max(np.abs(method_truth_qois), axis=0) * 1e-12, 1e-15),
        )
        coverage = {}
        for group, indices in stratum_indices("assessment").items():
            positions = [assessment_indices.index(index) for index in indices]
            radii = np.asarray(calibration[group]["scalar_radii"])
            hits = np.abs(scalar_assessment[positions] - truth_qois[positions]) <= radii
            widths = 2.0 * radii / method_ranges
            coverage[group] = {
                "rows": 16,
                "coverage": float(np.mean(hits)),
                "median_normalized_width": float(np.median(widths)),
                "p95_normalized_width": percentile(widths.tolist(), 0.95),
                "passed": bool(
                    np.mean(hits) >= PROTOCOL["uncertainty"]["coverage_minimum"]
                    and np.median(widths) <= PROTOCOL["uncertainty"]["median_normalized_width_max"]
                    and percentile(widths.tolist(), 0.95) <= PROTOCOL["uncertainty"]["p95_normalized_width_max"]
                ),
            }
        safety = []
        inference_times = {}
        for position, index in enumerate(assessment_indices):
            group = next(name for name, values in stratum_indices("assessment").items() if index in values)
            mismatch = not topology_match(field_assessment[position], truth_fields[position])
            safety.append(
                {
                    "index": index,
                    "group": group,
                    "ood_flag": group == "ood",
                    "topology_uncertainty_flag": mismatch,
                    "surrogate_point_acceptable": group != "ood" and not mismatch,
                }
            )
            tick = time.perf_counter()
            _predict_scalar(scalar_family, scalar_model, [all_data[position]])
            probe = _predict_field(field_family, basis, field_model, all_data[position], cases[index])
            topology(probe)
            inference_times[index] = time.perf_counter() - tick
        warmups = [192, 208, 224]
        timed = [index for index in assessment_indices if index not in warmups]
        coarse_pipeline = [timings[index]["low"] + inference_times[index] for index in timed]
        fine_pipeline = [timings[index]["fine"] for index in timed]
        speedups = [fine / coarse for fine, coarse in zip(fine_pipeline, coarse_pipeline, strict=True)]
        latency = {
            "warmup_indices": warmups,
            "timed_rows": len(timed),
            "coarse_plus_inference_median_s": percentile(coarse_pipeline, 0.5),
            "coarse_plus_inference_p95_s": percentile(coarse_pipeline, 0.95),
            "fine_median_s": percentile(fine_pipeline, 0.5),
            "fine_p95_s": percentile(fine_pipeline, 0.95),
            "paired_speedup_median": percentile(speedups, 0.5),
            "paired_speedup_p05": percentile(speedups, 0.05),
            "passed": percentile(speedups, 0.5) >= PROTOCOL["gates"]["latency"]["minimum_median_speedup"],
        }
        numerical_gates = _numerical_gates(numerical, 240)
        metrics_passed = _passes(metrics, basis.retained >= 0.995)
        safety_passed = all(
            not item["surrogate_point_acceptable"]
            for item in safety
            if item["ood_flag"] or item["topology_uncertainty_flag"]
        )
        accepted = (
            numerical_gates["passed"]
            and metrics_passed
            and all(item["passed"] for item in coverage.values())
            and safety_passed
            and latency["passed"]
        )
        write_json(RESULTS / "numerical-records.json", {"records": numerical, "gates": numerical_gates})
        terminal = {
            "schema_version": "cft-revival.l1a-field-surrogate-v3.terminal/3.0.0",
            "status": "accepted" if accepted else "failed-predeclared-assessment-gates",
            "valid_prospective_result": True,
            "selected": {
                "budget": budget,
                "scalar_family": scalar_family,
                "scalar_length": scalar_length,
                "field_family": field_family,
                "field_length": field_length,
                "pod_rank": basis.rank,
                "pod_retained_energy": basis.retained,
            },
            "assessment_metrics": metrics,
            "coverage": coverage,
            "decision_safety": safety,
            "latency": latency,
            "numerical": numerical_gates,
            "stage_access_counters": counters,
            "assessment_accessed_once": True,
            "claim": PROTOCOL["classification"],
        }
        terminal["terminal_hash"] = canonical_hash(terminal)
        write_json(RESULTS / "terminal-result.json", terminal)
        shutil.rmtree(cache)
        manifest = {
            "schema_version": "cft-revival.l1a-field-surrogate-v3.manifest/3.0.0",
            "status": terminal["status"],
            "preregistration_commit_sha": commit,
            "protocol_hash": PROTOCOL_HASH,
            "provenance_hash": provenance["provenance_hash"],
            "terminal_hash": terminal["terminal_hash"],
            "stage_access_counters": counters,
            "exclusive_lock_retained": True,
            "rerun_performed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        write_json(RESULTS / "run-manifest.json", manifest)
        from .validate import validate_bundle

        validate_bundle()
        return terminal
    except Exception as error:
        failure = {
            "schema_version": "cft-revival.l1a-field-surrogate-v3.failure/3.0.0",
            "status": "failed-execution",
            "valid_prospective_result": True,
            "preregistration_commit_sha": commit,
            "protocol_hash": PROTOCOL_HASH,
            "provenance_hash": provenance["provenance_hash"],
            "stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "stage_access_counters": counters,
            "exclusive_lock_retained": True,
            "rerun_performed": False,
        }
        failure["failure_hash"] = canonical_hash(failure)
        write_json(RESULTS / "failure-manifest.json", failure)
        from .validate import validate_bundle

        validate_bundle()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute", "validate"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "execute":
        result = execute()
    else:
        from .validate import validate_bundle

        result = validate_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
