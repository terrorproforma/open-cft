"""Preregistered group-exchangeable L0 surrogate validation v7."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from fractions import Fraction
from hashlib import sha256
from math import fsum, sqrt
from pathlib import Path
from statistics import pstdev
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.physics import evaluate_batch, load_l0_json
from cft_revival.surrogates import ExactGP, Prediction, SurrogateSchema
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads
from experiments.l0_surrogate_v2 import protocol as science
from experiments.l0_surrogate_v3.serialization import AtomicArtifactStore

from .cluster_conformal import (
    exact_rank,
    fit_cluster,
    interval,
    nearest_distances,
    select_family,
)
from .design import (
    global_partition,
    group_key,
    normalized_design,
    operating_points,
    scrambled_coordinate,
    surrogate_inputs,
)
from .identity import acquire_lock, bind

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
        raise ValueError("v7 predeclaration hash mismatch")
    return value


def _ranges(declaration: Mapping[str, object]) -> Mapping[str, object]:
    config = load_l0_json(SOURCE_CONFIG)
    if canonical_hash(config) != declaration["design"]["source_config_hash"]:  # type: ignore[index]
        raise ValueError("source config hash changed")
    return config["ranges"]


def _old_grid_inputs() -> tuple[tuple[float, ...], ...]:
    config = load_l0_json(SOURCE_CONFIG)
    start = 17 + int(config["seed"]) * 104_729

    def radical_inverse(index: int, base: int) -> float:
        result = 0.0
        factor = 1.0 / base
        while index:
            index, digit = divmod(index, base)
            result += digit * factor
            factor /= base
        return result

    return tuple(
        tuple(radical_inverse(start + row, base) for base in (2, 3, 5, 7, 13))
        for row in range(1, int(config["batch_size"]) + 1)
    )


def _prior_scrambled_inputs(version: str) -> tuple[tuple[float, ...], ...]:
    declaration = _load(
        MODERN / "experiments" / f"l0_surrogate_{version}" / "predeclaration.json"
    )
    policy = declaration["design"]
    bases = tuple(int(value) for value in policy["bases"])  # type: ignore[index]
    normalized = tuple(
        tuple(
            scrambled_coordinate(
                int(policy["skip"]) + row + 1,  # type: ignore[index]
                base,
                dimension,
                int(policy["scramble_seed"]),  # type: ignore[index]
                int(policy["digits_per_coordinate"]),  # type: ignore[index]
            )
            for dimension, base in enumerate(bases)
        )
        for row in range(int(policy["rows"]))  # type: ignore[index]
    )
    return surrogate_inputs(normalized)


def _prior_assessment_coordinates() -> tuple[set[tuple[float, ...]], dict[str, object]]:
    union = set()
    records = {}
    old = _old_grid_inputs()
    for version in ("v3", "v4"):
        path = MODERN / "experiments" / f"l0_surrogate_{version}" / "partitions.json"
        partition = _load(path)
        indices = {
            int(index)
            for replicate in partition["replicates"]  # type: ignore[union-attr]
            for split in replicate["assessment"].values()
            for index in split["indices"]
        }
        coordinates = {old[index] for index in indices}
        union.update(coordinates)
        records[version] = {
            "assessment_coordinate_count": len(coordinates),
            "partitions_sha256": sha256(path.read_bytes()).hexdigest(),
        }
    for version in ("v5", "v6"):
        path = MODERN / "experiments" / f"l0_surrogate_{version}" / "partitions.json"
        partition = _load(path)
        if version == "v5":
            indices = {
                int(index)
                for replicate in partition["replicates"]  # type: ignore[union-attr]
                for split in replicate["assessment"].values()
                for index in split["indices"]
            }
        else:
            indices = {
                int(index)
                for split in partition["roles"]["assessment"].values()  # type: ignore[index]
                for index in split["indices"]
            }
        prior_inputs = _prior_scrambled_inputs(version)
        coordinates = {prior_inputs[index] for index in indices}
        union.update(coordinates)
        records[version] = {
            "assessment_coordinate_count": len(coordinates),
            "partitions_sha256": sha256(path.read_bytes()).hexdigest(),
        }
    return union, records


def build_partitions() -> dict[str, object]:
    declaration = load_declaration()
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    roles = global_partition(inputs, declaration)
    prior, records = _prior_assessment_coordinates()
    assessment_indices = tuple(
        int(index)
        for split in roles["assessment"].values()  # type: ignore[union-attr]
        for index in split["indices"]
    )
    coordinates = {inputs[index] for index in assessment_indices}
    intersection = coordinates.intersection(prior)
    if intersection:
        raise ValueError("v7 assessment reuses prior assessment coordinates")
    value: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v7-global-partition",
        "schema_version": "7.0",
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
        "label_policy": "input-only; no v7 physics labels evaluated",
    }
    value["partitions_hash"] = canonical_hash(value)
    return value


def write_partitions() -> dict[str, object]:
    value = build_partitions()
    AtomicArtifactStore(ROOT).write_json("partitions.json", value)
    return value


def preflight(*, record: bool = False) -> dict[str, object]:
    declaration = load_declaration()
    partitions = build_partitions()
    if exact_rank(99, Fraction(9, 10)) != 90:
        raise ValueError("n=99 group-rank regression failed")
    if exact_rank(239, Fraction(9, 10)) != 216:
        raise ValueError("n=239 group-rank regression failed")
    groups = {"small": (0,), "large": (1, 2, 3, 4)}
    truth = {0: 2.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    predictions = {index: Prediction(0.0, 1.0) for index in truth}
    distances = {index: 1.0 for index in truth}
    adversarial = fit_cluster(
        "symmetric-absolute", groups, truth, predictions, distances
    )
    if adversarial["rank_identity"]["independent_group_count"] != 2:  # type: ignore[index]
        raise ValueError("unequal-size group regression used row count")
    with tempfile.TemporaryDirectory(prefix="l0-v7-preflight-") as temporary:
        store = AtomicArtifactStore(Path(temporary))
        model = ExactGP.fit(
            ((0.0,), (0.5,), (1.0,)),
            (0.0, 0.25, 1.0),
            schema=SurrogateSchema(("x",), ("y",), ("1",), ("1",)),
            length_scale_mode="isotropic",
        )
        for budget in declaration["model_selection"]["budgets"]:  # type: ignore[index]
            for output in OUTPUT_NAMES:
                store.write_model(f"development/budget-{budget}/{output}.model.json", model)
            store.write_json(f"development/budget-{budget}/point-metrics.json", {"synthetic": True})
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
        "document_type": "cft-revival-l0-surrogate-v7-cluster-preflight",
        "schema_version": "7.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partitions["partitions_hash"],
        "rank_regressions": {"n99": 90, "n239": 216},
        "unequal_correlated_group_regression": {
            "groups": 2,
            "rows": 5,
            "rank_uses_groups": True,
            "duplicating_correlated_rows_does_not_create_exchangeability_units": True,
        },
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


class RoleOracle:
    def __init__(self, points: Sequence[object], allowed: Sequence[int]) -> None:
        self._points = points
        self._allowed = frozenset(allowed)
        self._cache: dict[int, tuple[float, float]] = {}

    def observe(self, index: int) -> tuple[float, float]:
        if index not in self._allowed:
            raise ValueError("label access outside declared role")
        if index not in self._cache:
            result = evaluate_batch((self._points[index],))[0]  # type: ignore[arg-type]
            self._cache[index] = (result.axial_thrust_n, result.specific_impulse_s)
        return self._cache[index]

    def observe_many(self, indices: Sequence[int]) -> dict[int, tuple[float, float]]:
        missing = tuple(index for index in indices if index not in self._cache)
        if any(index not in self._allowed for index in missing):
            raise ValueError("batch label access outside declared role")
        if missing:
            values = evaluate_batch(tuple(self._points[index] for index in missing))  # type: ignore[arg-type]
            for index, result in zip(missing, values, strict=True):
                self._cache[index] = (result.axial_thrust_n, result.specific_impulse_s)
        return {index: self._cache[index] for index in indices}


class SingleUseAssessment:
    def __init__(self, points: Sequence[object], role: Mapping[str, object]) -> None:
        self._points = points
        self._role = role
        self._used = False

    def load(self, frozen: str, expected: str) -> dict[str, dict[int, tuple[float, float]]]:
        if self._used:
            raise RuntimeError("assessment is globally single-use")
        if frozen != expected:
            raise ValueError("calibration hash changed before assessment")
        self._used = True
        result = {}
        for stratum in ("interpolation", "boundary", "ood"):
            indices = tuple(int(index) for index in self._role[stratum]["indices"])  # type: ignore[index]
            values = evaluate_batch(tuple(self._points[index] for index in indices))  # type: ignore[arg-type]
            result[stratum] = {
                index: (value.axial_thrust_n, value.specific_impulse_s)
                for index, value in zip(indices, values, strict=True)
            }
        return result


def _load_role(points: Sequence[object], role: Mapping[str, object]) -> dict[str, dict[int, tuple[float, float]]]:
    result = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(int(index) for index in role[stratum]["indices"])  # type: ignore[index]
        result[stratum] = RoleOracle(points, indices).observe_many(indices)
    return result


def _groups(
    split: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, tuple[int, ...]]:
    return {
        str(group): tuple(
            int(index)
            for index in split["indices"]  # type: ignore[index]
            if group_key(inputs[int(index)], declaration["partition"]) == group  # type: ignore[arg-type]
        )
        for group in split["groups"]  # type: ignore[index]
    }


def _point_metrics(
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
        predictions = science.predict_rows(models, indices, inputs)
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
    models: Sequence[ExactGP],
    selected: Sequence[int],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    role: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    gates = declaration["gates"]
    training = tuple(inputs[index] for index in selected)
    result: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = science.predict_rows(models, indices, inputs)
        distances = nearest_distances(tuple(inputs[index] for index in indices), training)
        groups = _groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        result[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            result[stratum][name] = select_family(  # type: ignore[index]
                groups,
                {index: labels[stratum][index][output] for index in indices},
                {index: predictions[index][output] for index in indices},
                dict(zip(indices, distances, strict=True)),
                output_scale=float(gates["quality_scales"][name]),  # type: ignore[index]
                simultaneous_minimum=float(gates["method_selection_group_coverage_minimum"]),  # type: ignore[index]
                row_coverage_minimum=float(gates["method_selection_equal_group_row_coverage_minimum"]),  # type: ignore[index]
                group_row_standard_deviation_maximum=float(gates["equal_group_row_coverage_standard_deviation_maximum"]),  # type: ignore[index]
            )
    result["all_gates_passed"] = all(
        result[stratum][name]["all_gates_passed"]  # type: ignore[index]
        for stratum in ("interpolation", "boundary", "ood")
        for name in OUTPUT_NAMES
    )
    return result


def _calibrate(
    models: Sequence[ExactGP],
    selected: Sequence[int],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    role: Mapping[str, object],
    methods: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    training = tuple(inputs[index] for index in selected)
    result: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = science.predict_rows(models, indices, inputs)
        distances = nearest_distances(tuple(inputs[index] for index in indices), training)
        groups = _groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        result[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            family = str(methods[stratum][name]["selected_family"])  # type: ignore[index]
            result[stratum][name] = {  # type: ignore[index]
                "selected_family": family,
                "parameters": fit_cluster(
                    family,
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


def _assessment(
    models: Sequence[ExactGP],
    selected: Sequence[int],
    calibration: Mapping[str, object],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    role: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    gates = declaration["gates"]
    training = tuple(inputs[index] for index in selected)
    raw: dict[str, object] = {}
    scope_rows = {name: [] for name in OUTPUT_NAMES}
    scope_groups = {name: {} for name in OUTPUT_NAMES}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = science.predict_rows(models, indices, inputs)
        distances = nearest_distances(tuple(inputs[index] for index in indices), training)
        index_distance = dict(zip(indices, distances, strict=True))
        groups = _groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        raw[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            rows = []
            by_index = {}
            parameters = calibration[stratum][name]["parameters"]  # type: ignore[index]
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
                }
                rows.append(row)
                by_index[index] = row
            raw[stratum][name] = rows  # type: ignore[index]
            scope_rows[name].extend(rows)
            for group, group_indices in groups.items():
                scope_groups[name][group] = [by_index[index] for index in group_indices]

    def metric(rows, groups, name):
        scale = float(gates["quality_scales"][name])  # type: ignore[index]
        errors = [abs(row["truth"] - row["predicted_mean"]) for row in rows]
        coverage = fsum(row["covered"] for row in rows) / len(rows)
        group_coverages = {
            group: fsum(row["covered"] for row in values) / len(values)
            for group, values in groups.items()
        }
        simultaneous = {
            group: all(row["covered"] for row in values)
            for group, values in groups.items()
        }
        group_simultaneous = fsum(simultaneous.values()) / len(simultaneous)
        group_sd = pstdev(group_coverages.values())
        record = {
            "rows": len(rows),
            "groups": len(groups),
            "range_normalized_rmse": sqrt(fsum(value * value for value in errors) / len(errors)) / scale,
            "worst_range_normalized_error": max(errors) / scale,
            "row_coverage": coverage,
            "simultaneous_group_coverage": group_simultaneous,
            "equal_group_mean_row_coverage": fsum(group_coverages.values()) / len(group_coverages),
            "equal_group_row_coverage_standard_deviation": group_sd,
            "group_row_coverages": group_coverages,
            "group_simultaneous_hits": simultaneous,
        }
        low, high = gates["row_coverage_interval"]  # type: ignore[index]
        record["nrmse_passed"] = record["range_normalized_rmse"] <= float(gates["range_normalized_rmse_maximum"])  # type: ignore[index]
        record["worst_error_passed"] = record["worst_range_normalized_error"] <= float(gates["worst_case_range_normalized_error_maximum"])  # type: ignore[index]
        record["row_coverage_passed"] = float(low) <= coverage <= float(high)
        record["simultaneous_group_coverage_passed"] = group_simultaneous >= float(gates["simultaneous_group_coverage_minimum"])  # type: ignore[index]
        record["group_stability_passed"] = group_sd <= float(gates["equal_group_row_coverage_standard_deviation_maximum"])  # type: ignore[index]
        record["all_gates_passed"] = all(
            record[key]
            for key in (
                "nrmse_passed",
                "worst_error_passed",
                "row_coverage_passed",
                "simultaneous_group_coverage_passed",
                "group_stability_passed",
            )
        )
        return record

    metrics: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        metrics[stratum] = {}
        groups = _groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        for name in OUTPUT_NAMES:
            grouped_rows = {
                group: [
                    row
                    for row in raw[stratum][name]  # type: ignore[index]
                    if row["index"] in set(indices)
                ]
                for group, indices in groups.items()
            }
            metrics[stratum][name] = metric(raw[stratum][name], grouped_rows, name)  # type: ignore[index]
    metrics["overall"] = {
        name: metric(scope_rows[name], scope_groups[name], name) for name in OUTPUT_NAMES
    }
    metrics["all_scopes_outputs_passed"] = all(
        metrics[scope][name]["all_gates_passed"]  # type: ignore[index]
        for scope in ("interpolation", "boundary", "ood", "overall")
        for name in OUTPUT_NAMES
    )
    return raw, metrics


def _save_models(store: AtomicArtifactStore, budget: int, models: Sequence[ExactGP]) -> dict[str, str]:
    result = {}
    for name, model in zip(OUTPUT_NAMES, models, strict=True):
        store.write_model(f"development/budget-{budget}/{name}.model.json", model)
        result[name] = model.model_hash
    return result


def _early_manifest(binding, lock, partitions, preflight, status, frozen):
    value = {
        "document_type": "cft-revival-l0-surrogate-v7-run-manifest",
        "schema_version": "7.0",
        "commit_binding": binding.to_dict(),
        "exclusive_lock": {"file": lock.name, "retained": True, "atomic": "O_CREAT|O_EXCL"},
        "partitions_hash": partitions["partitions_hash"],
        "preflight_hash": preflight["preflight_hash"],
        "status": status,
        "frozen_method_hash": frozen["frozen_method_hash"],
        "final_calibration_labels_accessed": False,
        "assessment_labels_accessed": False,
        "valid_prospective_result": True,
    }
    value["run_manifest_hash"] = canonical_hash(value)
    AtomicArtifactStore(RESULTS).write_json("run-manifest.json", value)
    return value


def _execute_after_binding(binding, lock: Path, started: float) -> dict[str, object]:
    declaration = load_declaration()
    partitions = _load(PARTITIONS)
    preflight = _load(PREFLIGHT)
    roles = partitions["roles"]
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    points = operating_points(normalized, _ranges(declaration))
    candidate = tuple(int(index) for index in roles["candidate_indices"])
    selected_all, observed_all, rounds = science.select_active_indices(
        inputs, candidate, RoleOracle(points, candidate), declaration
    )
    store = AtomicArtifactStore(RESULTS)
    store.write_json(
        "training-selection.json",
        {
            "selected_indices": list(selected_all),
            "rounds": rounds,
            "selection_hash": canonical_hash(list(selected_all)),
        },
    )
    method_labels = _load_role(points, roles["method-selection"])
    candidates = []
    models_by_budget = {}
    for raw_budget in declaration["model_selection"]["budgets"]:  # type: ignore[index]
        budget = int(raw_budget)
        selected = selected_all[:budget]
        models = science.fit_models(
            selected, inputs, {index: observed_all[index] for index in selected}
        )
        models_by_budget[budget] = models
        record = {
            "budget": budget,
            "model_hashes": _save_models(store, budget, models),
            "point_metrics": _point_metrics(models, method_labels, inputs, declaration),
        }
        record["all_selection_gates_passed"] = record["point_metrics"]["all_scopes_outputs_passed"]
        record["candidate_hash"] = canonical_hash(record)
        store.write_json(f"development/budget-{budget}/point-metrics.json", record)
        candidates.append(record)
    passing = [item for item in candidates if item["all_selection_gates_passed"]]
    if not passing:
        frozen = {
            "mean_model_candidates": candidates,
            "selected_budget": None,
            "method_label_hash": canonical_hash(method_labels),
            "final_calibration_access_count": 0,
            "assessment_access_count": 0,
        }
        frozen["frozen_method_hash"] = canonical_hash(frozen)
        store.write_json("frozen-method-selection.json", frozen)
        return _early_manifest(
            binding, lock, partitions, preflight, "failed-mean-model-selection-gates", frozen
        )
    budget = min(int(item["budget"]) for item in passing)
    selected = selected_all[:budget]
    models = models_by_budget[budget]
    methods = _select_intervals(
        models,
        selected,
        method_labels,
        roles["method-selection"],
        inputs,
        declaration,
    )
    frozen = {
        "mean_model_candidates": candidates,
        "selected_budget": budget,
        "selected_model_hashes": next(item for item in candidates if item["budget"] == budget)["model_hashes"],
        "interval_methods": methods,
        "method_label_hash": canonical_hash(method_labels),
        "final_calibration_access_count": 0,
        "assessment_access_count": 0,
    }
    frozen["frozen_method_hash"] = canonical_hash(frozen)
    store.write_json("frozen-method-selection.json", frozen)
    if not methods["all_gates_passed"]:
        return _early_manifest(
            binding, lock, partitions, preflight, "failed-interval-method-selection-gates", frozen
        )
    calibration_labels = _load_role(points, roles["final-calibration"])
    calibration = _calibrate(
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
    assessment_labels = SingleUseAssessment(points, roles["assessment"]).load(
        frozen_calibration["frozen_calibration_hash"],
        frozen_calibration["frozen_calibration_hash"],
    )
    raw, metrics = _assessment(
        models,
        selected,
        calibration,
        assessment_labels,
        roles["assessment"],
        inputs,
        declaration,
    )
    assessment = {
        "document_type": "cft-revival-l0-surrogate-v7-final-assessment",
        "schema_version": "7.0",
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "raw": raw,
        "metrics": metrics,
    }
    assessment["assessment_hash"] = canonical_hash(assessment)
    store.write_json("final-assessment.json", assessment)
    accepted = bool(metrics["all_scopes_outputs_passed"])
    manifest = {
        "document_type": "cft-revival-l0-surrogate-v7-run-manifest",
        "schema_version": "7.0",
        "commit_binding": binding.to_dict(),
        "exclusive_lock": {"file": lock.name, "retained": True, "atomic": "O_CREAT|O_EXCL"},
        "partitions_hash": partitions["partitions_hash"],
        "normalized_design_hash": partitions["normalized_design_hash"],
        "assessment_prior_coordinate_intersection_count": 0,
        "same_domain_spatial_overlap_disclosed": True,
        "preflight_hash": preflight["preflight_hash"],
        "selected_budget": budget,
        "selected_methods": {
            stratum: {
                name: methods[stratum][name]["selected_family"]
                for name in OUTPUT_NAMES
            }
            for stratum in ("interpolation", "boundary", "ood")
        },
        "rank_semantics": declaration["conformal"],
        "frozen_method_hash": frozen["frozen_method_hash"],
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "assessment_metrics": metrics,
        "final_calibration_accessed_after_method_freeze": True,
        "assessment_accessed_once_after_calibration_freeze": True,
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
    store.write_json(
        "runtime-diagnostics.json",
        {"wall_seconds": perf_counter() - started, "diagnostic_only": True},
    )
    return manifest


def execute() -> dict[str, object]:
    started = perf_counter()
    binding = bind(REPO, DEPENDENCIES)
    lock = acquire_lock(REPO, binding.commit_sha)
    try:
        return _execute_after_binding(binding, lock, started)
    except Exception as error:
        failure = {
            "document_type": "cft-revival-l0-surrogate-v7-execution-failure",
            "schema_version": "7.0",
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
