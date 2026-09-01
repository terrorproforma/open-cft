"""Prospective L0 validation with physics features and efficient cluster intervals."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from functools import lru_cache
from math import ceil, fsum, sqrt
from pathlib import Path
from statistics import median, pstdev
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.physics import load_l0_json
from cft_revival.surrogates import ExactGP, SurrogateSchema
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads
from experiments.l0_surrogate_v3.serialization import AtomicArtifactStore
from experiments.l0_surrogate_v7 import protocol as v7
from experiments.l0_surrogate_v7.cluster_conformal import (
    exact_rank,
    fit_cluster,
    interval,
    nearest_distances,
)
from experiments.l0_surrogate_v7.design import (
    global_partition,
    normalized_design,
    operating_points,
    surrogate_inputs,
)

from .efficiency import select_efficient_family
from .identity import acquire_lock, bind
from .models import fit_models, physics_features, predict_rows, transform

ROOT = Path(__file__).resolve().parent
MODERN = ROOT.parents[1]
REPO = MODERN.parent
DECLARATION = ROOT / "predeclaration.json"
DEPENDENCIES = ROOT / "dependency-manifest.json"
PARTITIONS = ROOT / "partitions.json"
PREFLIGHT = ROOT / "preflight.json"
RESULTS = ROOT / "results"
SOURCE_CONFIG = MODERN / "config/l0-deterministic-sweep.json"
OUTPUT_NAMES = ("axial_thrust_n", "specific_impulse_s")


def _load(path: Path) -> dict[str, object]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_declaration() -> dict[str, object]:
    value = _load(DECLARATION)
    if value["predeclaration_hash"] != canonical_hash(
        {key: item for key, item in value.items() if key != "predeclaration_hash"}
    ):
        raise ValueError("v8 predeclaration hash mismatch")
    return value


def _ranges(declaration: Mapping[str, object]) -> Mapping[str, object]:
    config = load_l0_json(SOURCE_CONFIG)
    if canonical_hash(config) != declaration["design"]["source_config_hash"]:  # type: ignore[index]
        raise ValueError("source config changed")
    return config["ranges"]


@lru_cache(maxsize=1)
def build_partitions() -> dict[str, object]:
    declaration = load_declaration()
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    roles = global_partition(inputs, declaration)
    prior, records = v7._prior_assessment_coordinates()
    v7_partition = _load(MODERN / "experiments/l0_surrogate_v7/partitions.json")
    v7_inputs = v7._prior_scrambled_inputs("v7")
    v7_indices = {
        int(index)
        for split in v7_partition["roles"]["assessment"].values()  # type: ignore[index]
        for index in split["indices"]
    }
    v7_coordinates = {v7_inputs[index] for index in v7_indices}
    prior.update(v7_coordinates)
    records["v7"] = {
        "assessment_coordinate_count": len(v7_coordinates),
        "partitions_hash": v7_partition["partitions_hash"],
    }
    assessment_indices = tuple(
        int(index)
        for split in roles["assessment"].values()  # type: ignore[union-attr]
        for index in split["indices"]
    )
    intersection = {inputs[index] for index in assessment_indices}.intersection(prior)
    if intersection:
        raise ValueError("v8 assessment reuses prior assessment coordinates")
    value: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v8-global-partition",
        "schema_version": "8.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "normalized_design_hash": canonical_hash(
            {"normalized_design": [list(row) for row in normalized]}
        ),
        "surrogate_input_hash": canonical_hash({"inputs": [list(row) for row in inputs]}),
        "rows": len(inputs),
        "roles": roles,
        "prior_assessment_evidence": records,
        "assessment_prior_coordinate_intersection_count": 0,
        "domain_disclosure": declaration["partition"]["domain_disclosure"],  # type: ignore[index]
        "label_policy": "input-only; no v8 physics labels evaluated",
    }
    value["partitions_hash"] = canonical_hash(value)
    return value


def write_partitions() -> dict[str, object]:
    value = build_partitions()
    AtomicArtifactStore(ROOT).write_json("partitions.json", value)
    return value


def preflight(*, record: bool = False) -> dict[str, object]:
    from fractions import Fraction

    declaration = load_declaration()
    partitions = build_partitions()
    if exact_rank(99, Fraction(9, 10)) != 90 or exact_rank(239, Fraction(9, 10)) != 216:
        raise ValueError("inherited exact group ranks failed")
    fixture = physics_features((0.2, 0.3, 0.4, 0.5, 0.6))
    if len(fixture) != 7 or not all(value == value for value in fixture):
        raise ValueError("physics feature preflight failed")
    with tempfile.TemporaryDirectory(prefix="l0-v8-preflight-") as temporary:
        store = AtomicArtifactStore(Path(temporary))
        model = ExactGP.fit(
            ((0.0,), (0.5,), (1.0,)),
            (0.0, 0.25, 1.0),
            schema=SurrogateSchema(("x",), ("y",), ("1",), ("1",)),
            length_scale_mode="isotropic",
        )
        for family in declaration["model_selection"]["families"]:  # type: ignore[index]
            for budget in declaration["model_selection"]["budgets"]:  # type: ignore[index]
                for output in OUTPUT_NAMES:
                    store.write_model(f"development/{family}/budget-{budget}/{output}.model.json", model)
                store.write_json(f"development/{family}/budget-{budget}/method-metrics.json", {"synthetic": True})
        for artifact in (
            "training-selection",
            "frozen-method-selection",
            "cluster-calibration",
            "frozen-before-assessment",
            "final-assessment",
            "run-manifest",
            "runtime-diagnostics",
            "failure-manifest",
        ):
            store.write_json(f"{artifact}.json", {"synthetic": True})
        temporary_files = store.temporary_files()
        files = sorted(
            str(path.relative_to(temporary)).replace("\\", "/")
            for path in Path(temporary).rglob("*")
            if path.is_file()
        )
    value: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v8-synthetic-preflight",
        "schema_version": "8.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partitions["partitions_hash"],
        "exact_group_rank_regressions": {"n99": 90, "n239": 216},
        "physics_feature_fixture": list(fixture),
        "serialization_files": files,
        "temporary_file_count": len(temporary_files),
        "physics_label_access_count": 0,
        "assessment_label_access_count": 0,
        "passed": not temporary_files,
    }
    value["preflight_hash"] = canonical_hash(value)
    if record:
        AtomicArtifactStore(ROOT).write_json("preflight.json", value)
    return value


def _point_metrics(
    family: str,
    models: Sequence[ExactGP],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    gates = declaration["gates"]
    errors = {
        scope: {name: [] for name in OUTPUT_NAMES}
        for scope in ("interpolation", "boundary", "ood", "overall")
    }
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = predict_rows(family, models, indices, inputs)
        for output, name in enumerate(OUTPUT_NAMES):
            values = [
                abs(labels[stratum][index][output] - predictions[index][output].mean)
                for index in indices
            ]
            errors[stratum][name].extend(values)
            errors["overall"][name].extend(values)
    result: dict[str, object] = {}
    for scope in errors:
        result[scope] = {}
        for name in OUTPUT_NAMES:
            values = errors[scope][name]
            scale = float(gates["quality_scales"][name])  # type: ignore[index]
            nrmse = sqrt(fsum(value * value for value in values) / len(values)) / scale
            worst = max(values) / scale
            record = {
                "rows": len(values),
                "range_normalized_rmse": nrmse,
                "worst_range_normalized_error": worst,
                "nrmse_passed": nrmse <= float(gates["range_normalized_rmse_maximum"]),  # type: ignore[index]
                "worst_error_passed": worst <= float(gates["worst_case_range_normalized_error_maximum"]),  # type: ignore[index]
            }
            record["all_gates_passed"] = record["nrmse_passed"] and record["worst_error_passed"]
            result[scope][name] = record  # type: ignore[index]
    result["all_scopes_outputs_passed"] = all(
        result[scope][name]["all_gates_passed"]  # type: ignore[index]
        for scope in ("interpolation", "boundary", "ood", "overall")
        for name in OUTPUT_NAMES
    )
    return result


def _select_intervals(
    family: str,
    models: Sequence[ExactGP],
    selected: Sequence[int],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    role: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    gates = declaration["gates"]
    transformed = transform(family, inputs)
    training = tuple(transformed[index] for index in selected)
    result: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = predict_rows(family, models, indices, inputs)
        distances = nearest_distances(tuple(transformed[index] for index in indices), training)
        groups = v7._groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        result[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            result[stratum][name] = select_efficient_family(  # type: ignore[index]
                groups,
                {index: labels[stratum][index][output] for index in indices},
                {index: predictions[index][output] for index in indices},
                dict(zip(indices, distances, strict=True)),
                output_scale=float(gates["quality_scales"][name]),  # type: ignore[index]
                simultaneous_minimum=float(gates["simultaneous_group_coverage_minimum"]),  # type: ignore[index]
                row_minimum=float(gates["row_coverage_minimum"]),  # type: ignore[index]
                stability_maximum=float(gates["equal_group_row_coverage_standard_deviation_maximum"]),  # type: ignore[index]
                median_width_maximum=float(gates["normalized_median_full_interval_width_maximum"]),  # type: ignore[index]
                p90_width_maximum=float(gates["normalized_p90_full_interval_width_maximum"]),  # type: ignore[index]
            )
    result["all_gates_passed"] = all(
        result[stratum][name]["all_gates_passed"]  # type: ignore[index]
        for stratum in ("interpolation", "boundary", "ood")
        for name in OUTPUT_NAMES
    )
    return result


def _calibrate(
    family: str,
    models: Sequence[ExactGP],
    selected: Sequence[int],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    role: Mapping[str, object],
    methods: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    transformed = transform(family, inputs)
    training = tuple(transformed[index] for index in selected)
    result: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = predict_rows(family, models, indices, inputs)
        distances = nearest_distances(tuple(transformed[index] for index in indices), training)
        groups = v7._groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        result[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            interval_family = str(methods[stratum][name]["selected_family"])  # type: ignore[index]
            result[stratum][name] = {  # type: ignore[index]
                "selected_family": interval_family,
                "parameters": fit_cluster(
                    interval_family,
                    groups,
                    {index: labels[stratum][index][output] for index in indices},
                    {index: predictions[index][output] for index in indices},
                    dict(zip(indices, distances, strict=True)),
                ),
                "groups": role[stratum]["groups"],  # type: ignore[index]
                "indices": list(indices),
                "labels": [labels[stratum][index][output] for index in indices],
            }
    return result


def _p90(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[ceil(9 * len(ordered) / 10) - 1]


def _assessment(
    family: str,
    models: Sequence[ExactGP],
    selected: Sequence[int],
    calibration: Mapping[str, object],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    role: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    gates = declaration["gates"]
    transformed = transform(family, inputs)
    training = tuple(transformed[index] for index in selected)
    raw: dict[str, object] = {}
    all_rows = {name: [] for name in OUTPUT_NAMES}
    all_groups = {name: {} for name in OUTPUT_NAMES}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = predict_rows(family, models, indices, inputs)
        distances = nearest_distances(tuple(transformed[index] for index in indices), training)
        index_distance = dict(zip(indices, distances, strict=True))
        groups = v7._groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        raw[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            rows = []
            by_index = {}
            parameters = calibration[stratum][name]["parameters"]  # type: ignore[index]
            scale = float(gates["quality_scales"][name])  # type: ignore[index]
            for index in indices:
                prediction = predictions[index][output]
                lower, upper = interval(parameters, prediction, index_distance[index])
                truth = labels[stratum][index][output]
                row = {
                    "index": index,
                    "truth": truth,
                    "predicted_mean": prediction.mean,
                    "lower": lower,
                    "upper": upper,
                    "covered": lower <= truth <= upper,
                    "normalized_width": (upper - lower) / scale,
                }
                rows.append(row)
                by_index[index] = row
            raw[stratum][name] = rows  # type: ignore[index]
            all_rows[name].extend(rows)
            for group, group_indices in groups.items():
                all_groups[name][group] = [by_index[index] for index in group_indices]

    def metric(rows, groups, name):
        scale = float(gates["quality_scales"][name])  # type: ignore[index]
        errors = [abs(row["truth"] - row["predicted_mean"]) for row in rows]
        widths = [row["normalized_width"] for row in rows]
        coverage = fsum(row["covered"] for row in rows) / len(rows)
        group_coverages = {
            group: fsum(row["covered"] for row in values) / len(values)
            for group, values in groups.items()
        }
        simultaneous = {
            group: all(row["covered"] for row in values)
            for group, values in groups.items()
        }
        result = {
            "rows": len(rows),
            "groups": len(groups),
            "range_normalized_rmse": sqrt(fsum(value * value for value in errors) / len(errors)) / scale,
            "worst_range_normalized_error": max(errors) / scale,
            "row_coverage": coverage,
            "row_coverage_above_0_95_diagnostic": coverage > 0.95,
            "simultaneous_group_coverage": fsum(simultaneous.values()) / len(simultaneous),
            "equal_group_row_coverage_standard_deviation": pstdev(group_coverages.values()),
            "normalized_median_full_interval_width": median(widths),
            "normalized_p90_full_interval_width": _p90(widths),
            "group_row_coverages": group_coverages,
            "group_simultaneous_hits": simultaneous,
        }
        result["nrmse_passed"] = result["range_normalized_rmse"] <= float(gates["range_normalized_rmse_maximum"])  # type: ignore[index]
        result["worst_error_passed"] = result["worst_range_normalized_error"] <= float(gates["worst_case_range_normalized_error_maximum"])  # type: ignore[index]
        result["row_lower_passed"] = coverage >= float(gates["row_coverage_minimum"])  # type: ignore[index]
        result["simultaneous_group_passed"] = result["simultaneous_group_coverage"] >= float(gates["simultaneous_group_coverage_minimum"])  # type: ignore[index]
        result["group_stability_passed"] = result["equal_group_row_coverage_standard_deviation"] <= float(gates["equal_group_row_coverage_standard_deviation_maximum"])  # type: ignore[index]
        result["median_width_passed"] = result["normalized_median_full_interval_width"] <= float(gates["normalized_median_full_interval_width_maximum"])  # type: ignore[index]
        result["p90_width_passed"] = result["normalized_p90_full_interval_width"] <= float(gates["normalized_p90_full_interval_width_maximum"])  # type: ignore[index]
        result["all_gates_passed"] = all(
            result[key]
            for key in (
                "nrmse_passed",
                "worst_error_passed",
                "row_lower_passed",
                "simultaneous_group_passed",
                "group_stability_passed",
                "median_width_passed",
                "p90_width_passed",
            )
        )
        return result

    metrics: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        metrics[stratum] = {}
        groups = v7._groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        for name in OUTPUT_NAMES:
            rows = raw[stratum][name]  # type: ignore[index]
            by_index = {row["index"]: row for row in rows}
            grouped = {
                group: [by_index[index] for index in group_indices]
                for group, group_indices in groups.items()
            }
            metrics[stratum][name] = metric(rows, grouped, name)  # type: ignore[index]
    metrics["overall"] = {
        name: metric(all_rows[name], all_groups[name], name) for name in OUTPUT_NAMES
    }
    metrics["all_scopes_outputs_passed"] = all(
        metrics[scope][name]["all_gates_passed"]  # type: ignore[index]
        for scope in ("interpolation", "boundary", "ood", "overall")
        for name in OUTPUT_NAMES
    )
    return raw, metrics


def _save_models(store: AtomicArtifactStore, family: str, budget: int, models: Sequence[ExactGP]) -> dict[str, str]:
    hashes = {}
    for name, model in zip(OUTPUT_NAMES, models, strict=True):
        store.write_model(f"development/{family}/budget-{budget}/{name}.model.json", model)
        hashes[name] = model.model_hash
    return hashes


def _execute_after_binding(binding, lock: Path, started: float) -> dict[str, object]:
    declaration = load_declaration()
    partitions = _load(PARTITIONS)
    preflight = _load(PREFLIGHT)
    roles = partitions["roles"]
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    points = operating_points(normalized, _ranges(declaration))
    candidate = tuple(int(index) for index in roles["candidate_indices"])
    maximum_budget = max(int(value) for value in declaration["model_selection"]["budgets"])  # type: ignore[index]
    selected_all = candidate[:maximum_budget]
    observed = v7.RoleOracle(points, candidate).observe_many(selected_all)
    store = AtomicArtifactStore(RESULTS)
    store.write_json(
        "training-selection.json",
        {
            "policy": declaration["model_selection"]["training_selection"],  # type: ignore[index]
            "selected_indices": list(selected_all),
            "selection_hash": canonical_hash(list(selected_all)),
        },
    )
    method_labels = v7._load_role(points, roles["method-selection"])
    candidates = []
    model_lookup = {}
    for family in declaration["model_selection"]["families"]:  # type: ignore[index]
        for raw_budget in declaration["model_selection"]["budgets"]:  # type: ignore[index]
            budget = int(raw_budget)
            selected = selected_all[:budget]
            models = fit_models(
                str(family),
                selected,
                inputs,
                {index: observed[index] for index in selected},
            )
            model_lookup[(str(family), budget)] = models
            point = _point_metrics(str(family), models, method_labels, inputs, declaration)
            methods = _select_intervals(
                str(family),
                models,
                selected,
                method_labels,
                roles["method-selection"],
                inputs,
                declaration,
            )
            record = {
                "family": str(family),
                "budget": budget,
                "model_hashes": _save_models(store, str(family), budget, models),
                "point_metrics": point,
                "interval_methods": methods,
                "all_selection_gates_passed": point["all_scopes_outputs_passed"]
                and methods["all_gates_passed"],
            }
            record["candidate_hash"] = canonical_hash(record)
            store.write_json(f"development/{family}/budget-{budget}/method-metrics.json", record)
            candidates.append(record)
    passing_budgets = sorted(
        {int(item["budget"]) for item in candidates if item["all_selection_gates_passed"]}
    )
    selected_candidate = None
    if passing_budgets:
        budget = passing_budgets[0]
        eligible = [
            item
            for item in candidates
            if item["budget"] == budget and item["all_selection_gates_passed"]
        ]
        selected_candidate = min(
            eligible,
            key=lambda item: (
                max(
                    item["point_metrics"]["ood"][name]["worst_range_normalized_error"]
                    for name in OUTPUT_NAMES
                ),
                sum(
                    item["point_metrics"]["overall"][name]["range_normalized_rmse"]
                    for name in OUTPUT_NAMES
                ),
                item["family"],
            ),
        )
    frozen = {
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "method_labels_hash": canonical_hash(method_labels),
        "final_calibration_access_count": 0,
        "assessment_access_count": 0,
    }
    frozen["frozen_method_hash"] = canonical_hash(frozen)
    store.write_json("frozen-method-selection.json", frozen)
    if selected_candidate is None:
        manifest = {
            "document_type": "cft-revival-l0-surrogate-v8-run-manifest",
            "schema_version": "8.0",
            "commit_binding": binding.to_dict(),
            "exclusive_lock": {"file": lock.name, "retained": True},
            "partitions_hash": partitions["partitions_hash"],
            "preflight_hash": preflight["preflight_hash"],
            "frozen_method_hash": frozen["frozen_method_hash"],
            "status": "failed-development-selection-gates",
            "final_calibration_labels_accessed": False,
            "assessment_labels_accessed": False,
            "valid_prospective_result": True,
        }
        manifest["run_manifest_hash"] = canonical_hash(manifest)
        store.write_json("run-manifest.json", manifest)
        return manifest
    family = str(selected_candidate["family"])
    budget = int(selected_candidate["budget"])
    selected = selected_all[:budget]
    models = model_lookup[(family, budget)]
    methods = selected_candidate["interval_methods"]
    calibration_labels = v7._load_role(points, roles["final-calibration"])
    calibration = _calibrate(
        family,
        models,
        selected,
        calibration_labels,
        roles["final-calibration"],
        methods,
        inputs,
        declaration,
    )
    calibration["calibration_hash"] = canonical_hash(calibration)
    store.write_json("cluster-calibration.json", calibration)
    frozen_calibration = {
        "frozen_method_hash": frozen["frozen_method_hash"],
        "calibration_hash": calibration["calibration_hash"],
        "assessment_access_count": 0,
    }
    frozen_calibration["frozen_calibration_hash"] = canonical_hash(frozen_calibration)
    store.write_json("frozen-before-assessment.json", frozen_calibration)
    assessment_labels = v7.SingleUseAssessment(points, roles["assessment"]).load(
        frozen_calibration["frozen_calibration_hash"],
        frozen_calibration["frozen_calibration_hash"],
    )
    raw, metrics = _assessment(
        family,
        models,
        selected,
        calibration,
        assessment_labels,
        roles["assessment"],
        inputs,
        declaration,
    )
    assessment = {
        "document_type": "cft-revival-l0-surrogate-v8-final-assessment",
        "schema_version": "8.0",
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "raw": raw,
        "metrics": metrics,
    }
    assessment["assessment_hash"] = canonical_hash(assessment)
    store.write_json("final-assessment.json", assessment)
    accepted = bool(metrics["all_scopes_outputs_passed"])
    manifest = {
        "document_type": "cft-revival-l0-surrogate-v8-run-manifest",
        "schema_version": "8.0",
        "commit_binding": binding.to_dict(),
        "exclusive_lock": {"file": lock.name, "retained": True, "atomic": "O_CREAT|O_EXCL"},
        "partitions_hash": partitions["partitions_hash"],
        "normalized_design_hash": partitions["normalized_design_hash"],
        "assessment_prior_coordinate_intersection_count": 0,
        "same_domain_spatial_overlap_disclosed": True,
        "preflight_hash": preflight["preflight_hash"],
        "selected_family": family,
        "selected_features": declaration["features"]["physics-informed-v1"] if family.startswith("physics") else declaration["features"]["raw"],  # type: ignore[index]
        "selected_budget": budget,
        "selected_methods": {
            stratum: {
                name: methods[stratum][name]["selected_family"]
                for name in OUTPUT_NAMES
            }
            for stratum in ("interpolation", "boundary", "ood")
        },
        "development_point_metrics": selected_candidate["point_metrics"],
        "development_interval_metrics": methods,
        "frozen_method_hash": frozen["frozen_method_hash"],
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "assessment_metrics": metrics,
        "assessment_accessed_once_after_calibration_freeze": True,
        "row_coverage_upper_is_diagnostic_only": True,
        "status": "accepted" if accepted else "failed-predeclared-assessment-gates",
        "valid_prospective_result": True,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "claim": "same-domain prospective deterministic L0 software-emulation validation only",
    }
    manifest["run_manifest_hash"] = canonical_hash(manifest)
    store.write_json("run-manifest.json", manifest)
    store.write_json("runtime-diagnostics.json", {"wall_seconds": perf_counter() - started, "diagnostic_only": True})
    return manifest


def execute() -> dict[str, object]:
    started = perf_counter()
    binding = bind(REPO, DEPENDENCIES)
    lock = acquire_lock(REPO, binding.commit_sha)
    try:
        return _execute_after_binding(binding, lock, started)
    except Exception as error:
        failure = {
            "document_type": "cft-revival-l0-surrogate-v8-execution-failure",
            "schema_version": "8.0",
            "commit_binding": binding.to_dict(),
            "exclusive_lock_retained": True,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "rerun_performed": False,
        }
        failure["failure_manifest_hash"] = canonical_hash(failure)
        AtomicArtifactStore(RESULTS).write_json("failure-manifest.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("partitions", "preflight", "execute"))
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "partitions":
        result = write_partitions()
    elif args.command == "preflight":
        result = preflight(record=args.record)
    else:
        result = execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
