"""Two-phase preregistration preparation and exactly-once v2 execution."""

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

from experiments.l1a_field_surrogate_v1.run import _model_artifact

from .experiment import (
    HIGH_DOMAIN,
    QOIS,
    assessment_groups,
    calibrate,
    coverage_metrics,
    field_vector,
    fit_field_family,
    fit_scalar_family,
    high_indices,
    model_metrics,
    numerical_record,
    predict_fields,
    predict_scalars,
    prolong_low,
    raw_designs,
    rebuild_frozen,
    role_indices,
    select_frozen,
    solve_frozen_case,
    topology_match,
    preflight_raw_candidates,
)
from .protocol import (
    GEOMETRY_PREFLIGHT,
    PARTITIONS,
    PROTOCOL,
    PROTOCOL_HASH,
    REPO,
    RESULTS,
    ROOT,
    SYNTHETIC_PREFLIGHT,
    canonical_hash,
    percentile,
    strict_load,
    verify_json,
    write_json,
)

SUBJECT = PROTOCOL["integrity"]["protocol_commit_subject"]
REMOTE = PROTOCOL["integrity"]["remote"]
PREFIXES = (
    "modern/experiments/l1a_field_surrogate_v2/",
    "modern/tests/experiments/l1a_field_surrogate_v2/",
)
DEPENDENCY_ROOTS = (
    "modern/src/cft_revival/fields/",
    "modern/src/cft_revival/geometry/",
    "modern/src/cft_revival/magnetics/",
    "modern/src/cft_revival/surrogates/",
    "modern/src/cft_revival/active_learning/",
    "modern/src/cft_revival/optimization/",
    "modern/experiments/l1a_geometry_sweep_v2/",
    "modern/experiments/l1a_field_surrogate_v1/",
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


def prepare() -> dict[str, Any]:
    records, valid = preflight_raw_candidates()
    frozen_raw_indices = select_frozen(valid)
    rebuilt, rebuild_records = rebuild_frozen(frozen_raw_indices)
    initial = {index: valid[raw_index] for index, raw_index in enumerate(frozen_raw_indices)}
    hash_failures = [
        index
        for index in range(112)
        if (
            initial[index].geometry_sha256 != rebuilt[index].geometry_sha256
            or initial[index].source_sha256 != rebuilt[index].source_sha256
        )
    ]
    if hash_failures:
        raise RuntimeError(f"frozen geometry rebuild mismatch: {hash_failures}")
    rejected = [record for record in records if not record["valid"]]
    geometry_preflight = {
        "schema_version": "cft-revival.l1a-field-surrogate-v2.geometry-preflight/2.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "raw_count": len(records),
        "valid_count": len(valid),
        "rejected_count": len(rejected),
        "corrected_count": sum(record.get("attempt_count", 1) > 1 for record in records),
        "raw_records": records,
        "rejected_raw_rows": rejected,
        "frozen_raw_indices": list(frozen_raw_indices),
        "frozen_rebuild_records": rebuild_records,
        "frozen_hash_failure_count": len(hash_failures),
        "field_solver_access_count": 0,
        "qoi_label_access_count": 0,
    }
    geometry_preflight["geometry_preflight_hash"] = canonical_hash(geometry_preflight)
    write_json(GEOMETRY_PREFLIGHT, geometry_preflight)
    designs = raw_designs()
    partition = {
        "schema_version": "cft-revival.l1a-field-surrogate-v2.partitions/2.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "geometry_preflight_hash": geometry_preflight["geometry_preflight_hash"],
        "frozen_raw_indices": list(frozen_raw_indices),
        "frozen_design_ids": [designs[index].design_id for index in frozen_raw_indices],
        "frozen_design_rows": [list(designs[index].values) for index in frozen_raw_indices],
        "roles": {
            name: list(role_indices(name))
            for name in ("candidate", "method_selection", "final_calibration", "single_use_assessment")
        },
        "assessment_groups": {name: list(indices) for name, indices in assessment_groups().items()},
        "high_fidelity_indices": list(high_indices()),
        "v1_coordinate_intersection_count": 0,
        "label_access_count": 0,
    }
    partition["partition_hash"] = canonical_hash(partition)
    write_json(PARTITIONS, partition)
    synthetic = {
        "schema_version": "cft-revival.l1a-field-surrogate-v2.synthetic-preflight/2.0.0",
        "protocol_hash": PROTOCOL_HASH,
        "geometry_preflight_hash": geometry_preflight["geometry_preflight_hash"],
        "partition_hash": partition["partition_hash"],
        "role_disjoint": len({index for values in partition["roles"].values() for index in values}) == 112,
        "frozen_geometry_rebuild_count": 112,
        "frozen_geometry_hash_failures": 0,
        "field_solver_access_count": 0,
        "qoi_label_access_count": 0,
        "model_serialization_synthetic_only": True,
        "passed": True,
    }
    synthetic["synthetic_preflight_hash"] = canonical_hash(synthetic)
    write_json(SYNTHETIC_PREFLIGHT, synthetic)
    return {
        "passed": True,
        "valid_raw_rows": len(valid),
        "rejected_raw_rows": len(rejected),
        "corrected_raw_rows": geometry_preflight["corrected_count"],
        "frozen_rows": 112,
        "geometry_preflight_hash": geometry_preflight["geometry_preflight_hash"],
        "partition_hash": partition["partition_hash"],
    }


def _bind() -> tuple[str, list[dict[str, str]]]:
    head = _git("rev-parse", "HEAD")
    if _git("symbolic-ref", "-q", "HEAD", check=False):
        raise RuntimeError("v2 execution requires detached HEAD")
    if _git("show", "-s", "--format=%s", head) != SUBJECT:
        raise RuntimeError("HEAD is not the v2 preregistration commit")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", head, REMOTE),
        cwd=REPO,
        capture_output=True,
    ).returncode:
        raise RuntimeError("v2 preregistration is not pushed")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("v2 detached worktree is not clean")
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()
    if not changed or any(not path.startswith(PREFIXES) for path in changed):
        raise RuntimeError("v2 preregistration commit is not exact-path isolated")
    if any("/results/" in path and not path.endswith("/results/README.md") for path in changed):
        raise RuntimeError("v2 preregistration contains result labels")
    for path in (GEOMETRY_PREFLIGHT, PARTITIONS, SYNTHETIC_PREFLIGHT):
        verify_json(path)
    return head, _closure()


def _acquire_lock(commit: str) -> Path:
    common = Path(_git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (REPO / common).resolve()
    path = common / "l1a-field-surrogate-v2.execution.lock"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(commit + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _numerical_gates(records: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    limits = PROTOCOL["gates"]["numerics"]
    checks = {
        "residual": max(item["relative_residual_l2"] for item in records) <= limits["relative_residual_l2_max"],
        "boundary": max(item["boundary_to_peak_ratio"] for item in records) <= limits["boundary_to_peak_ratio_max"],
        "source": max(item["source_representation_error"] for item in records) <= limits["source_representation_error_max"],
        "topology_confidence": min(item["topology_confidence"] for item in records) >= limits["topology_confidence_min"],
        "flux_identity": max(item["flux_identity_t_per_m"] for item in records) <= limits["flux_identity_t_per_m_max"],
        "zero_frozen_execution_failures": not failures and len(records) == 112 + len(high_indices()),
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


def _viewer_prediction(
    index: int,
    group: str,
    row: Sequence[float],
    low: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    scalar_prediction: Mapping[str, float],
    scalar_truth: Mapping[str, float],
) -> dict[str, Any]:
    nr, nz = HIGH_DOMAIN.shape
    offset = nr * nz
    stride = 4

    def component(vector: np.ndarray, start: int) -> list[list[float]]:
        return vector[start : start + nr * nz].reshape((nr, nz))[::stride, ::stride].tolist()

    return {
        "schema_version": "cft-revival.l1a-field-surrogate-v2.viewer/2.0.0",
        "index": index,
        "group": group,
        "inputs": list(row),
        "mesh": {
            "r_m": [i * HIGH_DOMAIN.dr_m for i in range(0, nr, stride)],
            "z_m": [HIGH_DOMAIN.z_min_m + j * HIGH_DOMAIN.dz_m for j in range(0, nz, stride)],
            "stride": stride,
        },
        "br_t": {
            "coarse": component(low, 0),
            "prediction": component(prediction, 0),
            "fine": component(truth, 0),
        },
        "bz_t": {
            "coarse": component(low, offset),
            "prediction": component(prediction, offset),
            "fine": component(truth, offset),
        },
        "scalar_prediction": dict(scalar_prediction),
        "scalar_fine": dict(scalar_truth),
        "topology": topology_match(prediction, truth),
        "claim": PROTOCOL["classification"],
    }


def execute() -> dict[str, Any]:
    commit, closure = _bind()
    lock = _acquire_lock(commit)
    counters = {
        "coarse_completed": 0,
        "fine_completed": 0,
        "solver_accesses": 0,
        "model_fits": 0,
        "method_label_accesses": 0,
        "calibration_accesses": 0,
        "assessment_accesses": 0,
    }
    stage = "binding"
    started = time.perf_counter()
    write_json(
        RESULTS / "execution-lock.json",
        {"state": "claimed-once", "commit": commit, "lock_file": lock.name, "protocol_hash": PROTOCOL_HASH},
    )
    provenance = {
        "schema_version": "cft-revival.l1a-field-surrogate-v2.provenance-closure/2.0.0",
        "commit": commit,
        "protocol_hash": PROTOCOL_HASH,
        "partition_hash": strict_load(PARTITIONS)["partition_hash"],
        "geometry_preflight_hash": strict_load(GEOMETRY_PREFLIGHT)["geometry_preflight_hash"],
        "synthetic_preflight_hash": strict_load(SYNTHETIC_PREFLIGHT)["synthetic_preflight_hash"],
        "git_blob_closure": closure,
        "git_blob_closure_hash": canonical_hash(closure),
        "field_solver_access_count_at_write": 0,
        "qoi_label_access_count_at_write": 0,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "gpu": subprocess.run(
                ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"),
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
        },
    }
    provenance["provenance_hash"] = canonical_hash(provenance)
    write_json(RESULTS / "provenance-closure.json", provenance)
    verify_json(RESULTS / "provenance-closure.json")
    failures: list[dict[str, Any]] = []
    try:
        stage = "frozen-geometry-rebuild"
        partition = strict_load(PARTITIONS)
        raw_indices = tuple(int(value) for value in partition["frozen_raw_indices"])
        cases, rebuild = rebuild_frozen(raw_indices)
        expected = strict_load(GEOMETRY_PREFLIGHT)["frozen_rebuild_records"]
        if [
            (item["geometry_sha256"], item["source_sha256"]) for item in rebuild
        ] != [
            (item["geometry_sha256"], item["source_sha256"]) for item in expected
        ]:
            raise RuntimeError("execution geometry rebuild differs from frozen preflight")
        rows = tuple(tuple(float(value) for value in row) for row in partition["frozen_design_rows"])
        tracemalloc.start()
        low_qois: dict[int, dict[str, float]] = {}
        high_qois: dict[int, dict[str, float]] = {}
        low_fields: dict[int, np.ndarray] = {}
        high_fields: dict[int, np.ndarray] = {}
        records: list[dict[str, Any]] = []
        low_times: dict[int, float] = {}
        high_times: dict[int, float] = {}
        high_set = set(high_indices())
        stage = "field-solves"
        for index in range(112):
            tick = time.perf_counter()
            counters["solver_accesses"] += 1
            built, field, qois = solve_frozen_case(cases[index], "low")
            low_times[index] = time.perf_counter() - tick
            counters["coarse_completed"] += 1
            low_qois[index] = {name: float(qois[name]) for name in QOIS}
            low_fields[index] = prolong_low(field)
            records.append(numerical_record(index, built, field, qois, "low"))
            if index in high_set:
                tick = time.perf_counter()
                counters["solver_accesses"] += 1
                built_h, field_h, qois_h = solve_frozen_case(cases[index], "high")
                high_times[index] = time.perf_counter() - tick
                counters["fine_completed"] += 1
                high_qois[index] = {name: float(qois_h[name]) for name in QOIS}
                high_fields[index] = field_vector(field_h)
                records.append(numerical_record(index, built_h, field_h, qois_h, "high"))
                if records[-1]["pairing_sha256"] != records[-2]["pairing_sha256"]:
                    raise RuntimeError("coarse/fine pairing identity mismatch")
        numerical = _numerical_gates(records, failures)
        write_json(RESULTS / "numerical-records.json", {"records": records, "gates": numerical})

        stage = "development-model-selection"
        method = role_indices("method_selection")
        candidates = []
        model_cache: dict[tuple[str, int], Any] = {}
        scalar_cache: dict[tuple[str, int], Any] = {}
        field_cache: dict[tuple[str, int], Any] = {}
        for budget in PROTOCOL["sampling"]["model_budgets"]:
            for family in PROTOCOL["candidate_models"]["scalar"]:
                model_cache[(family, budget)] = fit_scalar_family(family, budget, rows, low_qois, high_qois)
                counters["model_fits"] += 1
                scalar_cache[(family, budget)] = predict_scalars(
                    family, model_cache[(family, budget)], method, rows, low_qois
                )
            for family in PROTOCOL["candidate_models"]["field"]:
                model_cache[(family, budget)] = fit_field_family(family, budget, rows, low_fields, high_fields)
                counters["model_fits"] += 1
                field_cache[(family, budget)] = predict_fields(
                    family, model_cache[(family, budget)], method, rows, low_fields
                )
            counters["method_label_accesses"] += 1
            for scalar_family in PROTOCOL["candidate_models"]["scalar"]:
                for field_family in PROTOCOL["candidate_models"]["field"]:
                    metrics = model_metrics(
                        method,
                        scalar_cache[(scalar_family, budget)],
                        field_cache[(field_family, budget)],
                        high_qois,
                        high_fields,
                    )
                    candidates.append(
                        {
                            "budget": budget,
                            "scalar_family": scalar_family,
                            "field_family": field_family,
                            "metrics": metrics,
                        }
                    )
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
        frozen = {"candidates": candidates, "selected": selected, "assessment_access_count": 0}
        frozen["frozen_selection_hash"] = canonical_hash(frozen)
        write_json(RESULTS / "frozen-method-selection.json", frozen)
        if selected is None:
            final = {
                "status": "failed-development-selection-gates",
                "valid_prospective_result": True,
                "counters": counters,
                "numerical": numerical,
                "assessment_labels_accessed": False,
            }
            final["terminal_hash"] = canonical_hash(final)
            write_json(RESULTS / "terminal-result.json", final)
            from .validate import validate_bundle

            validate_bundle()
            return final

        budget = selected["budget"]
        scalar_family = selected["scalar_family"]
        field_family = selected["field_family"]
        scalar_models = model_cache[(scalar_family, budget)]
        field_model = model_cache[(field_family, budget)]
        write_json(
            RESULTS / "selected-models.json",
            {
                "budget": budget,
                "scalar_family": scalar_family,
                "field_family": field_family,
                "scalar_models": [model.to_dict() for model in scalar_models],
                "field_model": _model_artifact(field_model),
            },
        )
        stage = "calibration"
        calibration_indices = role_indices("final_calibration")
        calibration_predictions = predict_scalars(
            scalar_family, scalar_models, calibration_indices, rows, low_qois
        )
        counters["calibration_accesses"] = 1
        calibration = calibrate(calibration_predictions, high_qois)
        calibration_record = {
            "method": PROTOCOL["uncertainty"]["method"],
            "groups": calibration,
            "assessment_access_count": 0,
        }
        calibration_record["calibration_hash"] = canonical_hash(calibration_record)
        write_json(RESULTS / "group-conformal-calibration.json", calibration_record)
        freeze = {
            "frozen_selection_hash": frozen["frozen_selection_hash"],
            "calibration_hash": calibration_record["calibration_hash"],
            "assessment_access_count": 0,
        }
        freeze["freeze_hash"] = canonical_hash(freeze)
        write_json(RESULTS / "frozen-before-assessment.json", freeze)

        stage = "single-use-assessment"
        assessment = role_indices("single_use_assessment")
        scalar_assessment = predict_scalars(scalar_family, scalar_models, assessment, rows, low_qois)
        field_assessment = predict_fields(field_family, field_model, assessment, rows, low_fields)
        counters["assessment_accesses"] = 1
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
        group_for = {
            index: group
            for group, indices in assessment_groups().items()
            for index in indices
        }
        safety = []
        for index in assessment:
            mismatch = not next(item for item in metrics["topology"] if item["index"] == index)["passed"]
            ood = group_for[index] == "ood"
            safety.append(
                {
                    "index": index,
                    "group": group_for[index],
                    "ood_flag": ood,
                    "topology_uncertainty_flag": mismatch,
                    "surrogate_point_acceptable": not (ood or mismatch),
                }
            )
        inference_times = {}
        for index in assessment:
            tick = time.perf_counter()
            predict_scalars(scalar_family, scalar_models, (index,), rows, low_qois)
            probe = predict_fields(field_family, field_model, (index,), rows, low_fields)
            topology_match(probe[index], high_fields[index])
            inference_times[index] = time.perf_counter() - tick
        timed = tuple(range(99, 112))
        coarse_pipeline = [low_times[index] + inference_times[index] for index in timed]
        fine_pipeline = [high_times[index] for index in timed]
        speedups = [fine / coarse for fine, coarse in zip(fine_pipeline, coarse_pipeline, strict=True)]
        latency = {
            "warmup_indices": [96, 97, 98],
            "timed_indices": list(timed),
            "coarse_plus_inference_seconds": {
                "median": percentile(coarse_pipeline, 0.5),
                "p95": percentile(coarse_pipeline, 0.95),
            },
            "fine_seconds": {
                "median": percentile(fine_pipeline, 0.5),
                "p95": percentile(fine_pipeline, 0.95),
            },
            "paired_speedup_median": percentile(speedups, 0.5),
            "paired_speedup_p05": percentile(speedups, 0.05),
            "preprocessing_and_transfers_included": True,
            "passed": percentile(speedups, 0.5) >= PROTOCOL["gates"]["latency"]["minimum_median_speedup"],
        }
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        latency["peak_traced_host_memory_bytes"] = peak_memory
        coverage_passed = all(item["passed"] for item in coverage.values())
        safety_passed = all(
            not item["surrogate_point_acceptable"]
            for item in safety
            if item["ood_flag"] or item["topology_uncertainty_flag"]
        )
        accepted = (
            numerical["passed"]
            and metrics["all_gates_passed"]
            and coverage_passed
            and safety_passed
            and latency["passed"]
        )
        representatives = {"interpolation": 96, "boundary": 102, "ood": 107}
        representative_files = []
        for group, index in representatives.items():
            path = RESULTS / "representatives" / f"{group}-{index}.json"
            digest = write_json(
                path,
                _viewer_prediction(
                    index,
                    group,
                    rows[index],
                    low_fields[index],
                    field_assessment[index],
                    high_fields[index],
                    scalar_assessment[index],
                    high_qois[index],
                ),
            )
            representative_files.append(
                {"path": str(path.relative_to(RESULTS)).replace("\\", "/"), "sha256": digest}
            )
        final = {
            "schema_version": "cft-revival.l1a-field-surrogate-v2.terminal-result/2.0.0",
            "status": "accepted" if accepted else "failed-predeclared-assessment-gates",
            "valid_prospective_result": True,
            "selected": {
                "budget": budget,
                "scalar_family": scalar_family,
                "field_family": field_family,
            },
            "metrics": metrics,
            "coverage": coverage,
            "decision_safety": safety,
            "latency": latency,
            "numerical": numerical,
            "counters": counters,
            "assessment_accessed_once_after_freeze": True,
            "representatives": representative_files,
            "claim": PROTOCOL["classification"],
        }
        final["terminal_hash"] = canonical_hash(final)
        write_json(RESULTS / "terminal-result.json", final)
        manifest = {
            "schema_version": "cft-revival.l1a-field-surrogate-v2.run-manifest/2.0.0",
            "status": final["status"],
            "preregistration_commit_sha": commit,
            "protocol_hash": PROTOCOL_HASH,
            "provenance_hash": provenance["provenance_hash"],
            "terminal_hash": final["terminal_hash"],
            "counters": counters,
            "exclusive_lock_retained": True,
            "rerun_performed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        write_json(RESULTS / "run-manifest.json", manifest)
        from .validate import validate_bundle

        validate_bundle()
        return final
    except Exception as error:
        failure = {
            "schema_version": "cft-revival.l1a-field-surrogate-v2.failure/2.0.0",
            "status": "failed-execution",
            "valid_prospective_result": True,
            "preregistration_commit_sha": commit,
            "protocol_hash": PROTOCOL_HASH,
            "provenance_hash": provenance["provenance_hash"],
            "stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "counters": counters,
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
