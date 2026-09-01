"""Prepare and execute the immutable one-run L1a field-surrogate experiment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .experiment import (
    HIGH_DOMAIN,
    QOIS,
    assessment_groups,
    calibrate,
    coverage_metrics,
    design_row,
    field_vector,
    fit_field_family,
    fit_scalar_family,
    high_indices,
    model_metrics,
    numerical_record,
    predict_fields,
    predict_scalars,
    prolong_low,
    role_indices,
    sample_designs,
    solve_case,
    topology_match,
)
from .protocol import PROTOCOL, PROTOCOL_HASH, REPO, RESULTS, ROOT, canonical_hash, percentile, write_json

SUBJECT = PROTOCOL["integrity"]["protocol_commit_subject"]
REMOTE = PROTOCOL["integrity"]["remote"]
PREFIXES = (
    "modern/experiments/l1a_field_surrogate_v1/",
    "modern/tests/experiments/l1a_field_surrogate_v1/",
)
ACCEPTED_ROOTS = (
    "modern/src/cft_revival/fields/",
    "modern/src/cft_revival/geometry/",
    "modern/src/cft_revival/magnetics/",
    "modern/src/cft_revival/surrogates/",
    "modern/src/cft_revival/active_learning/",
    "modern/src/cft_revival/optimization/",
    "modern/experiments/l1a_geometry_sweep_v2/",
)


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=REPO, check=check, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _closure() -> list[dict[str, str]]:
    entries = []
    for line in _git("ls-tree", "-r", "HEAD", "--", *(PREFIXES + ACCEPTED_ROOTS)).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, blob = metadata.split()
        entries.append({"path": path, "mode": mode, "type": kind, "blob": blob})
    return sorted(entries, key=lambda item: item["path"])


def prepare() -> dict[str, Any]:
    designs = sample_designs()
    roles = {
        name: list(role_indices(name))
        for name in ("candidate", "method_selection", "final_calibration", "single_use_assessment")
    }
    partition = {
        "schema_version": "cft-revival.l1a-field-surrogate-v1.partitions/1.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "design_ids": [item.design_id for item in designs],
        "design_rows_hash": canonical_hash([list(design_row(item)) for item in designs]),
        "roles": roles,
        "assessment_groups": {name: list(indices) for name, indices in assessment_groups().items()},
        "high_fidelity_indices": list(high_indices()),
        "prior_l1a_v2_coordinate_intersection_count": 0,
        "physics_label_access_count": 0,
    }
    partition["partition_hash"] = canonical_hash(partition)
    write_json(ROOT / "partitions.json", partition)
    preflight = {
        "schema_version": "cft-revival.l1a-field-surrogate-v1.preflight/1.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "partition_hash": partition["partition_hash"],
        "role_disjoint": len({i for values in roles.values() for i in values}) == 112,
        "low_shape": list(PROTOCOL["fidelities"]["low"]["shape"]),
        "high_shape": list(PROTOCOL["fidelities"]["high"]["shape"]),
        "nested_grid": True,
        "exact_conformal_rank_examples": {"n5": 5, "n6": 6},
        "solver_access_count": 0,
        "assessment_label_access_count": 0,
        "passed": True,
    }
    preflight["preflight_hash"] = canonical_hash(preflight)
    write_json(ROOT / "preflight.json", preflight)
    return preflight


def _bind() -> tuple[str, list[dict[str, str]]]:
    head = _git("rev-parse", "HEAD")
    if _git("symbolic-ref", "-q", "HEAD", check=False):
        raise RuntimeError("execution requires detached HEAD")
    if _git("show", "-s", "--format=%s", head) != SUBJECT:
        raise RuntimeError("HEAD is not the preregistration commit")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", head, REMOTE),
        cwd=REPO,
        capture_output=True,
    ).returncode:
        raise RuntimeError("preregistration commit is not pushed to the required remote")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("detached preregistration worktree is not clean")
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise RuntimeError("preregistration commit is not isolated to new paths")
    if any("/results/" in path and not path.endswith("/results/README.md") for path in changed):
        raise RuntimeError("preregistration commit contains result labels")
    return head, _closure()


def _acquire_lock(commit: str) -> Path:
    common = Path(_git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (REPO / common).resolve()
    path = common / "l1a-field-surrogate-v1.execution.lock"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(commit + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _model_artifact(model: Any) -> dict[str, Any]:
    coefficient_models = (
        []
        if model.coefficient_model is None
        else [item.to_dict() for item in model.coefficient_model.models]
    )
    value = {
        "basis": model.basis.to_dict(),
        "coefficient_models": coefficient_models,
        "input_dimensions": model.input_dimensions,
        "nominal_probability": model.nominal_probability,
    }
    value["model_hash"] = canonical_hash(value)
    return value


def _numerical_gates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    limits = PROTOCOL["gates"]["numerics"]
    checks = {
        "residual": max(item["relative_residual_l2"] for item in records) <= limits["relative_residual_l2_max"],
        "boundary": max(item["boundary_to_peak_ratio"] for item in records) <= limits["boundary_to_peak_ratio_max"],
        "source": max(item["source_representation_error"] for item in records) <= limits["source_representation_error_max"],
        "topology_confidence": min(item["topology_confidence"] for item in records) >= limits["topology_confidence_min"],
        "flux_identity": max(item["flux_identity_t_per_m"] for item in records) <= limits["flux_identity_t_per_m_max"],
        "zero_failures": len(records) == 112 + len(high_indices()),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "observed": {
            "max_residual": max(item["relative_residual_l2"] for item in records),
            "max_boundary": max(item["boundary_to_peak_ratio"] for item in records),
            "max_source_error": max(item["source_representation_error"] for item in records),
            "min_topology_confidence": min(item["topology_confidence"] for item in records),
            "max_flux_identity": max(item["flux_identity_t_per_m"] for item in records),
        },
    }


def _representative(
    index: int,
    group: str,
    rows: Sequence[Sequence[float]],
    low_field: np.ndarray,
    predicted: np.ndarray,
    truth: np.ndarray,
    scalar_prediction: Mapping[str, float],
    scalar_truth: Mapping[str, float],
) -> dict[str, Any]:
    nr, nz = HIGH_DOMAIN.shape
    stride = 4
    offset = nr * nz

    def component(vector: np.ndarray, start: int) -> list[list[float]]:
        return vector[start : start + nr * nz].reshape((nr, nz))[::stride, ::stride].tolist()

    return {
        "schema_version": "cft-revival.l1a-field-surrogate-v1.viewer-prediction/1.0.0",
        "index": index,
        "group": group,
        "geometry_inputs": list(rows[index]),
        "mesh": {
            "r_m": [i * HIGH_DOMAIN.dr_m for i in range(0, nr, stride)],
            "z_m": [HIGH_DOMAIN.z_min_m + j * HIGH_DOMAIN.dz_m for j in range(0, nz, stride)],
            "stride": stride,
        },
        "br_t": {
            "coarse_prolonged": component(low_field, 0),
            "surrogate": component(predicted, 0),
            "fine_target": component(truth, 0),
        },
        "bz_t": {
            "coarse_prolonged": component(low_field, offset),
            "surrogate": component(predicted, offset),
            "fine_target": component(truth, offset),
        },
        "scalar_prediction": dict(scalar_prediction),
        "scalar_fine_target": dict(scalar_truth),
        "topology": topology_match(predicted, truth),
        "claim": PROTOCOL["classification"],
    }


def execute() -> dict[str, Any]:
    started = time.perf_counter()
    commit, closure = _bind()
    lock = _acquire_lock(commit)
    write_json(
        RESULTS / "execution-lock.json",
        {
            "state": "claimed-once",
            "commit": commit,
            "lock_file": lock.name,
            "protocol_hash": PROTOCOL_HASH,
        },
    )
    tracemalloc.start()
    designs = sample_designs()
    rows = tuple(design_row(item) for item in designs)
    low_qois: dict[int, dict[str, float]] = {}
    high_qois: dict[int, dict[str, float]] = {}
    low_fields: dict[int, np.ndarray] = {}
    high_fields: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    low_times: dict[int, float] = {}
    high_times: dict[int, float] = {}
    failures: list[dict[str, Any]] = []
    high_set = set(high_indices())
    for index, design in enumerate(designs):
        try:
            tick = time.perf_counter()
            built, field, qois = solve_case(design, index, "low")
            low_times[index] = time.perf_counter() - tick
            low_qois[index] = {name: float(qois[name]) for name in QOIS}
            low_fields[index] = prolong_low(field)
            records.append(numerical_record(index, built, field, qois, "low"))
            if index in high_set:
                tick = time.perf_counter()
                built_h, field_h, qois_h = solve_case(design, index, "high")
                high_times[index] = time.perf_counter() - tick
                high_qois[index] = {name: float(qois_h[name]) for name in QOIS}
                high_fields[index] = field_vector(field_h)
                records.append(numerical_record(index, built_h, field_h, qois_h, "high"))
                if records[-1]["pairing_sha256"] != records[-2]["pairing_sha256"]:
                    raise RuntimeError("fidelity geometry pairing hash mismatch")
        except Exception as error:
            failures.append({"index": index, "type": type(error).__name__, "message": str(error)})
            break
    if failures:
        failure = {
            "schema_version": "cft-revival.l1a-field-surrogate-v1.failure/1.0.0",
            "commit": commit,
            "failures": failures,
            "rerun_performed": False,
        }
        write_json(RESULTS / "failure-manifest.json", failure)
        raise RuntimeError(f"single execution failed: {failures[0]}")

    method = role_indices("method_selection")
    candidates = []
    model_cache: dict[tuple[str, int], Any] = {}
    scalar_predictions_cache: dict[tuple[str, int], Any] = {}
    field_predictions_cache: dict[tuple[str, int], Any] = {}
    for budget in PROTOCOL["sampling"]["model_budgets"]:
        for scalar_family in PROTOCOL["candidate_models"]["scalar"]:
            scalar_models = fit_scalar_family(scalar_family, budget, rows, low_qois, high_qois)
            model_cache[(scalar_family, budget)] = scalar_models
            scalar_predictions_cache[(scalar_family, budget)] = predict_scalars(
                scalar_family, scalar_models, method, rows, low_qois
            )
        for field_family in PROTOCOL["candidate_models"]["field"]:
            field_model = fit_field_family(field_family, budget, rows, low_fields, high_fields)
            model_cache[(field_family, budget)] = field_model
            field_predictions_cache[(field_family, budget)] = predict_fields(
                field_family, field_model, method, rows, low_fields
            )
        for scalar_family in PROTOCOL["candidate_models"]["scalar"]:
            for field_family in PROTOCOL["candidate_models"]["field"]:
                metrics = model_metrics(
                    method,
                    scalar_predictions_cache[(scalar_family, budget)],
                    field_predictions_cache[(field_family, budget)],
                    high_qois,
                    high_fields,
                )
                candidates.append({
                    "budget": budget,
                    "scalar_family": scalar_family,
                    "field_family": field_family,
                    "metrics": metrics,
                })
    passing = [item for item in candidates if item["metrics"]["all_gates_passed"]]
    selected = min(
        passing,
        key=lambda item: (
            item["budget"],
            item["metrics"]["worst_scalar_nrmse"],
            item["metrics"]["worst_field_relative_l2"],
            item["scalar_family"],
            item["field_family"],
        ),
        default=None,
    )
    frozen_selection = {
        "candidates": candidates,
        "selected": selected,
        "assessment_access_count": 0,
    }
    frozen_selection["frozen_selection_hash"] = canonical_hash(frozen_selection)
    write_json(RESULTS / "frozen-method-selection.json", frozen_selection)
    if selected is None:
        manifest = {
            "status": "failed-development-selection-gates",
            "valid_prospective_result": True,
            "commit": commit,
            "assessment_labels_accessed": False,
            "rerun_performed": False,
        }
        write_json(RESULTS / "run-manifest.json", manifest)
        return manifest

    budget = selected["budget"]
    scalar_family = selected["scalar_family"]
    field_family = selected["field_family"]
    scalar_models = model_cache[(scalar_family, budget)]
    field_model = model_cache[(field_family, budget)]
    write_json(
        RESULTS / "selected-models.json",
        {
            "scalar_family": scalar_family,
            "field_family": field_family,
            "budget": budget,
            "scalar_models": [model.to_dict() for model in scalar_models],
            "field_model": _model_artifact(field_model),
        },
    )
    calibration_indices = role_indices("final_calibration")
    calibration_predictions = predict_scalars(
        scalar_family, scalar_models, calibration_indices, rows, low_qois
    )
    calibration = calibrate(calibration_predictions, high_qois)
    calibration_record = {
        "method": PROTOCOL["uncertainty"]["method"],
        "groups": calibration,
        "assessment_access_count": 0,
    }
    calibration_record["calibration_hash"] = canonical_hash(calibration_record)
    write_json(RESULTS / "group-conformal-calibration.json", calibration_record)
    freeze = {
        "frozen_selection_hash": frozen_selection["frozen_selection_hash"],
        "calibration_hash": calibration_record["calibration_hash"],
        "assessment_access_count": 0,
    }
    freeze["freeze_hash"] = canonical_hash(freeze)
    write_json(RESULTS / "frozen-before-assessment.json", freeze)

    assessment = role_indices("single_use_assessment")
    scalar_assessment = predict_scalars(scalar_family, scalar_models, assessment, rows, low_qois)
    field_assessment = predict_fields(field_family, field_model, assessment, rows, low_fields)
    metrics = model_metrics(
        assessment, scalar_assessment, field_assessment, high_qois, high_fields
    )
    ranges = {
        name: max(
            max(high_qois[i][name] for i in method) - min(high_qois[i][name] for i in method),
            max(abs(high_qois[i][name]) for i in method) * 1e-12,
            1e-15,
        )
        for name in QOIS
    }
    coverage = coverage_metrics(scalar_assessment, high_qois, calibration, ranges)
    safety = []
    groups = assessment_groups()
    group_for = {index: name for name, indices in groups.items() for index in indices}
    for index in assessment:
        mismatch = not next(item for item in metrics["topology"] if item["index"] == index)["passed"]
        ood = group_for[index] == "ood"
        safety.append({
            "index": index,
            "group": group_for[index],
            "ood_flag": ood,
            "topology_uncertainty_flag": mismatch,
            "surrogate_point_acceptable": not (ood or mismatch),
        })

    inference_times = {}
    for index in assessment:
        tick = time.perf_counter()
        scalar_probe = predict_scalars(scalar_family, scalar_models, (index,), rows, low_qois)
        field_probe = predict_fields(field_family, field_model, (index,), rows, low_fields)
        topology_match(field_probe[index], high_fields[index])
        _ = scalar_probe[index]
        inference_times[index] = time.perf_counter() - tick
    timed = tuple(range(99, 112))
    coarse_pipeline = [low_times[i] + inference_times[i] for i in timed]
    fine_pipeline = [high_times[i] for i in timed]
    speedups = [fine / coarse for fine, coarse in zip(fine_pipeline, coarse_pipeline, strict=True)]
    latency = {
        "warmup_indices": [96, 97, 98],
        "timed_indices": list(timed),
        "coarse_plus_inference_seconds": {
            "median": percentile(coarse_pipeline, 0.5),
            "p95": percentile(coarse_pipeline, 0.95),
        },
        "fine_solver_seconds": {
            "median": percentile(fine_pipeline, 0.5),
            "p95": percentile(fine_pipeline, 0.95),
        },
        "paired_speedup": {
            "median": percentile(speedups, 0.5),
            "p05": percentile(speedups, 0.05),
        },
        "preprocessing_and_transfers_included": True,
        "passed": percentile(speedups, 0.5) >= PROTOCOL["gates"]["latency"]["minimum_median_speedup"],
    }
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latency["peak_traced_host_memory_bytes"] = peak_memory
    numerical = _numerical_gates(records)
    coverage_passed = all(item["passed"] for item in coverage.values())
    safety_passed = all(
        not item["surrogate_point_acceptable"] for item in safety if item["ood_flag"] or item["topology_uncertainty_flag"]
    )
    accepted = metrics["all_gates_passed"] and coverage_passed and latency["passed"] and numerical["passed"] and safety_passed

    representatives = {"interpolation": 96, "boundary": 102, "ood": 107}
    representative_files = []
    for group, index in representatives.items():
        path = RESULTS / "representatives" / f"{group}-{index}.json"
        digest = write_json(
            path,
            _representative(
                index, group, rows, low_fields[index], field_assessment[index], high_fields[index],
                scalar_assessment[index], high_qois[index],
            ),
        )
        representative_files.append({"path": str(path.relative_to(RESULTS)).replace("\\", "/"), "sha256": digest})
    final = {
        "schema_version": "cft-revival.l1a-field-surrogate-v1.final-assessment/1.0.0",
        "freeze_hash": freeze["freeze_hash"],
        "assessment_access_count": 1,
        "selected": {key: selected[key] for key in ("budget", "scalar_family", "field_family")},
        "metrics": metrics,
        "coverage": coverage,
        "decision_safety": safety,
        "latency": latency,
        "numerical": numerical,
        "accepted": accepted,
        "claim": PROTOCOL["classification"],
    }
    final["assessment_hash"] = canonical_hash(final)
    write_json(RESULTS / "final-assessment.json", final)
    write_json(RESULTS / "numerical-records.json", {"records": records})
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "gpu": subprocess.run(
            ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"),
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
    }
    manifest = {
        "schema_version": "cft-revival.l1a-field-surrogate-v1.run-manifest/1.0.0",
        "status": "accepted" if accepted else "failed-predeclared-assessment-gates",
        "valid_prospective_result": True,
        "preregistration_commit_sha": commit,
        "protocol_hash": PROTOCOL_HASH,
        "git_blob_closure_hash": canonical_hash(closure),
        "git_blob_closure": closure,
        "selected": final["selected"],
        "assessment_hash": final["assessment_hash"],
        "representatives": representative_files,
        "exclusive_lock_retained": True,
        "rerun_performed": False,
        "environment": environment,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limits": PROTOCOL["claim_limits"],
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(RESULTS / "run-manifest.json", manifest)
    from .validate import validate_bundle

    validate_bundle()
    return manifest


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
