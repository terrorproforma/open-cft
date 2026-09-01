"""Prospective, globally held-out L0 surrogate validation v6."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from hashlib import sha256
from math import fsum, sqrt
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.physics import load_l0_json
from cft_revival.surrogates import ExactGP, SurrogateSchema
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads
from experiments.l0_surrogate_v2 import protocol as science
from experiments.l0_surrogate_v3.serialization import AtomicArtifactStore
from experiments.l0_surrogate_v5 import protocol as v5
from experiments.l0_surrogate_v5.design import normalized_design as v5_design

from .conformal import (
    finite_rank,
    fit_grouped,
    nearest_training_distances,
    select_interval,
)
from .design import global_partition, normalized_design, operating_points, surrogate_inputs
from .identity import acquire_lock, bind

ROOT = Path(__file__).resolve().parent
MODERN = ROOT.parents[1]
REPO = MODERN.parent
DECLARATION = ROOT / "predeclaration.json"
DEPENDENCIES = ROOT / "dependency-manifest.json"
PARTITIONS = ROOT / "partitions.json"
PREFLIGHT = ROOT / "preflight.json"
RESULTS = ROOT / "results"
SOURCE_CONFIG = MODERN / "config" / "l0-deterministic-sweep.json"
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
        raise ValueError("v6 predeclaration hash mismatch")
    return value


def _ranges(declaration: Mapping[str, object]) -> Mapping[str, object]:
    config = load_l0_json(SOURCE_CONFIG)
    if canonical_hash(config) != declaration["design"]["source_config_hash"]:  # type: ignore[index]
        raise ValueError("source config hash mismatch")
    return config["ranges"]


def _prior_coordinates() -> tuple[set[tuple[float, ...]], dict[str, object]]:
    old = v5._old_grid_inputs()
    union: set[tuple[float, ...]] = set()
    records = {}
    for version in ("v3", "v4"):
        path = MODERN / "experiments" / f"l0_surrogate_{version}" / "partitions.json"
        indices = v5._prior_heldout_indices(path)
        coordinates = {old[index] for index in indices}
        union.update(coordinates)
        records[version] = {
            "heldout_coordinate_count": len(coordinates),
            "partition_sha256": sha256(path.read_bytes()).hexdigest(),
        }
    v5_declaration = v5.load_declaration()
    v5_inputs = surrogate_inputs(v5_design(v5_declaration))
    v5_partition = _load(MODERN / "experiments/l0_surrogate_v5/partitions.json")
    indices = set()
    for replicate in v5_partition["replicates"]:  # type: ignore[union-attr]
        for role in ("method-selection", "final-calibration", "assessment"):
            for split in replicate[role].values():
                indices.update(int(index) for index in split["indices"])
    coordinates = {v5_inputs[index] for index in indices}
    union.update(coordinates)
    records["v5"] = {
        "heldout_coordinate_count": len(coordinates),
        "partition_sha256": sha256(v5.PARTITIONS.read_bytes()).hexdigest(),
    }
    return union, records


def build_partitions() -> dict[str, object]:
    declaration = load_declaration()
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    roles = global_partition(inputs, declaration)
    prior, prior_records = _prior_coordinates()
    assessment_indices = tuple(
        int(index)
        for stratum in ("interpolation", "boundary", "ood")
        for index in roles["assessment"][stratum]["indices"]  # type: ignore[index]
    )
    assessment_coordinates = {inputs[index] for index in assessment_indices}
    intersection = assessment_coordinates.intersection(prior)
    if intersection:
        raise ValueError("v6 assessment reuses prior v3-v5 heldout coordinates")
    record: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v6-global-partition",
        "schema_version": "6.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "normalized_design_hash": canonical_hash(
            {"normalized_design": [list(row) for row in normalized]}
        ),
        "surrogate_input_hash": canonical_hash({"inputs": [list(row) for row in inputs]}),
        "rows": len(inputs),
        "roles": roles,
        "prior_heldout_evidence": prior_records,
        "assessment_prior_coordinate_intersection_count": len(intersection),
        "domain_disclosure": declaration["partition"]["domain_disclosure"],  # type: ignore[index]
        "label_policy": "input-only partition construction; zero v6 physics evaluations",
    }
    record["partitions_hash"] = canonical_hash(record)
    return record


def write_partitions() -> dict[str, object]:
    value = build_partitions()
    AtomicArtifactStore(ROOT).write_json("partitions.json", value)
    return value


def preflight(*, record: bool = False) -> dict[str, object]:
    declaration = load_declaration()
    partitions = build_partitions()
    if finite_rank(99, 9, 10) != 90:
        raise ValueError("exact n=99 rank regression failed")
    with tempfile.TemporaryDirectory(prefix="l0-v6-preflight-") as temporary:
        store = AtomicArtifactStore(Path(temporary))
        model = ExactGP.fit(
            ((0.0,), (0.5,), (1.0,)),
            (0.0, 0.25, 1.0),
            schema=SurrogateSchema(("x",), ("y",), ("1",), ("1",)),
            length_scale_mode="isotropic",
        )
        for budget in declaration["model_selection"]["training_budgets"]:  # type: ignore[index]
            for output in OUTPUT_NAMES:
                store.write_model(f"development/budget-{budget}/{output}.model.json", model)
            store.write_json(f"development/budget-{budget}/method-diagnostics.json", {"synthetic": True})
        for artifact in (
            "training-selection",
            "frozen-method-selection",
            "grouped-final-calibration",
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
        "document_type": "cft-revival-l0-surrogate-v6-synthetic-preflight",
        "schema_version": "6.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partitions["partitions_hash"],
        "exact_rank_n99_p90": 90,
        "serialization_files": files,
        "temporary_file_count": len(temporary_files),
        "v6_physics_evaluation_count": 0,
        "final_calibration_label_access_count": 0,
        "assessment_label_access_count": 0,
        "passed": not temporary_files,
    }
    value["preflight_hash"] = canonical_hash(value)
    if record:
        AtomicArtifactStore(ROOT).write_json("preflight.json", value)
    return value


def _groups(split: Mapping[str, object], inputs: Sequence[Sequence[float]], declaration: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    from experiments.l0_surrogate_v5.design import group_key

    policy = declaration["partition"]
    return {
        str(group): tuple(
            int(index)
            for index in split["indices"]  # type: ignore[index]
            if group_key(inputs[int(index)], policy) == group  # type: ignore[arg-type]
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
    errors: dict[str, dict[str, list[float]]] = {
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
            result[scope][name] = {  # type: ignore[index]
                "rows": len(values),
                "range_normalized_rmse": nrmse,
                "worst_range_normalized_error": worst,
                "nrmse_passed": nrmse <= float(gates["range_normalized_rmse_maximum"]),  # type: ignore[index]
                "worst_error_passed": worst <= float(gates["worst_case_range_normalized_error_maximum"]),  # type: ignore[index]
            }
            result[scope][name]["all_gates_passed"] = (  # type: ignore[index]
                result[scope][name]["nrmse_passed"]  # type: ignore[index]
                and result[scope][name]["worst_error_passed"]  # type: ignore[index]
            )
    result["all_scopes_outputs_passed"] = all(
        result[scope][name]["all_gates_passed"]  # type: ignore[index]
        for scope in ("interpolation", "boundary", "ood", "overall")
        for name in OUTPUT_NAMES
    )
    return result


def _method_diagnostics(
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
        distances = nearest_training_distances(tuple(inputs[index] for index in indices), training)
        groups = _groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        result[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            result[stratum][name] = select_interval(  # type: ignore[index]
                groups,
                {index: labels[stratum][index][output] for index in indices},
                {index: predictions[index][output] for index in indices},
                dict(zip(indices, distances, strict=True)),
                output_scale=float(gates["quality_scales"][name]),  # type: ignore[index]
                coverage_bounds=tuple(gates["method_equal_group_coverage_interval"]),  # type: ignore[arg-type]
                standard_deviation_maximum=float(gates["method_group_coverage_standard_deviation_maximum"]),  # type: ignore[index]
            )
    result["all_gates_passed"] = all(
        result[stratum][name]["all_gates_passed"]  # type: ignore[index]
        for stratum in ("interpolation", "boundary", "ood")
        for name in OUTPUT_NAMES
    )
    return result


def _grouped_calibration(
    models: Sequence[ExactGP],
    selected: Sequence[int],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    role: Mapping[str, object],
    method: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    training = tuple(inputs[index] for index in selected)
    result: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = science.predict_rows(models, indices, inputs)
        distances = nearest_training_distances(tuple(inputs[index] for index in indices), training)
        groups = _groups(role[stratum], inputs, declaration)  # type: ignore[arg-type]
        result[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            family = str(method[stratum][name]["selected_family"])  # type: ignore[index]
            result[stratum][name] = {  # type: ignore[index]
                "selected_family": family,
                "parameters": fit_grouped(
                    family,
                    groups,
                    {index: labels[stratum][index][output] for index in indices},
                    {index: predictions[index][output] for index in indices},
                    dict(zip(indices, distances, strict=True)),
                ),
                "groups": role[stratum]["groups"],  # type: ignore[index]
                "indices": list(indices),
                "labels": [labels[stratum][index][output] for index in indices],
                "predicted_means": [predictions[index][output].mean for index in indices],
            }
    return result


def _load_role(points: Sequence[object], role: Mapping[str, object]) -> dict[str, dict[int, tuple[float, float]]]:
    result = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(int(index) for index in role[stratum]["indices"])  # type: ignore[index]
        result[stratum] = v5.RoleOracle(points, indices).observe_many(indices)
    return result


def _save_models(store: AtomicArtifactStore, budget: int, models: Sequence[ExactGP]) -> dict[str, str]:
    hashes = {}
    for name, model in zip(OUTPUT_NAMES, models, strict=True):
        store.write_model(f"development/budget-{budget}/{name}.model.json", model)
        hashes[name] = model.model_hash
    return hashes


def _execute_after_binding(binding: object, lock: Path, started: float) -> dict[str, object]:
    declaration = load_declaration()
    partition_record = _load(PARTITIONS)
    preflight_record = _load(PREFLIGHT)
    roles = partition_record["roles"]
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    points = operating_points(normalized, _ranges(declaration))
    candidate = tuple(int(index) for index in roles["candidate_indices"])
    training_oracle = v5.RoleOracle(points, candidate)
    selected_all, observed_all, rounds = science.select_active_indices(
        inputs, candidate, training_oracle, declaration
    )
    store = AtomicArtifactStore(RESULTS)
    selection = {
        "selected_indices": list(selected_all),
        "rounds": rounds,
        "selection_hash": canonical_hash(list(selected_all)),
    }
    store.write_json("training-selection.json", selection)

    method_labels = _load_role(points, roles["method-selection"])
    candidates = []
    candidate_models = {}
    for raw_budget in declaration["model_selection"]["training_budgets"]:  # type: ignore[index]
        budget = int(raw_budget)
        selected = selected_all[:budget]
        observed = {index: observed_all[index] for index in selected}
        models = science.fit_models(selected, inputs, observed)
        candidate_models[budget] = models
        model_hashes = _save_models(store, budget, models)
        point = _point_metrics(models, method_labels, inputs, declaration)
        method = _method_diagnostics(
            models,
            selected,
            method_labels,
            roles["method-selection"],
            inputs,
            declaration,
        )
        record = {
            "budget": budget,
            "model_hashes": model_hashes,
            "point_metrics": point,
            "interval_diagnostics": method,
            "all_selection_gates_passed": point["all_scopes_outputs_passed"]
            and method["all_gates_passed"],
        }
        record["candidate_hash"] = canonical_hash(record)
        store.write_json(f"development/budget-{budget}/method-diagnostics.json", record)
        candidates.append(record)
    passing = [record for record in candidates if record["all_selection_gates_passed"]]
    frozen_method: dict[str, object] = {
        "candidates": candidates,
        "selected_budget": None if not passing else min(int(item["budget"]) for item in passing),
        "method_labels_hash": canonical_hash(method_labels),
        "final_calibration_access_count": 0,
        "assessment_access_count": 0,
    }
    frozen_method["frozen_method_hash"] = canonical_hash(frozen_method)
    store.write_json("frozen-method-selection.json", frozen_method)
    if not passing:
        manifest: dict[str, object] = {
            "document_type": "cft-revival-l0-surrogate-v6-run-manifest",
            "schema_version": "6.0",
            "commit_binding": binding.to_dict(),
            "exclusive_lock": {"file": lock.name, "retained": True, "atomic": "O_CREAT|O_EXCL"},
            "partitions_hash": partition_record["partitions_hash"],
            "preflight_hash": preflight_record["preflight_hash"],
            "frozen_method_hash": frozen_method["frozen_method_hash"],
            "status": "failed-development-selection-gates",
            "final_calibration_labels_accessed": False,
            "assessment_labels_accessed": False,
            "valid_prospective_result": True,
        }
        manifest["run_manifest_hash"] = canonical_hash(manifest)
        store.write_json("run-manifest.json", manifest)
        return manifest

    budget = int(frozen_method["selected_budget"])
    models = candidate_models[budget]
    selected = selected_all[:budget]
    selected_method = next(item for item in candidates if item["budget"] == budget)["interval_diagnostics"]
    calibration_labels = _load_role(points, roles["final-calibration"])
    calibration = _grouped_calibration(
        models,
        selected,
        calibration_labels,
        roles["final-calibration"],
        selected_method,
        inputs,
        declaration,
    )
    calibration["calibration_hash"] = canonical_hash(calibration)
    store.write_json("grouped-final-calibration.json", calibration)
    frozen_calibration = {
        "frozen_method_hash": frozen_method["frozen_method_hash"],
        "calibration_hash": calibration["calibration_hash"],
        "selected_budget": budget,
        "model_hashes": next(item for item in candidates if item["budget"] == budget)["model_hashes"],
        "assessment_access_count": 0,
    }
    frozen_calibration["frozen_calibration_hash"] = canonical_hash(frozen_calibration)
    store.write_json("frozen-before-assessment.json", frozen_calibration)

    assessment_loader = v5.SingleUseAssessment(points, roles["assessment"])
    assessment_labels = assessment_loader.load(
        frozen_calibration["frozen_calibration_hash"],
        frozen_calibration["frozen_calibration_hash"],
    )
    raw, metrics = v5._assessment(
        models, selected, calibration, assessment_labels, inputs, declaration
    )
    assessment = {
        "document_type": "cft-revival-l0-surrogate-v6-global-final-assessment",
        "schema_version": "6.0",
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "raw": raw,
        "metrics": metrics,
    }
    assessment["assessment_hash"] = canonical_hash(assessment)
    store.write_json("final-assessment.json", assessment)
    accepted = bool(metrics["all_scopes_outputs_passed"])
    manifest = {
        "document_type": "cft-revival-l0-surrogate-v6-run-manifest",
        "schema_version": "6.0",
        "commit_binding": binding.to_dict(),
        "exclusive_lock": {"file": lock.name, "retained": True, "atomic": "O_CREAT|O_EXCL"},
        "partitions_hash": partition_record["partitions_hash"],
        "normalized_design_hash": partition_record["normalized_design_hash"],
        "assessment_prior_coordinate_intersection_count": partition_record[
            "assessment_prior_coordinate_intersection_count"
        ],
        "same_domain_spatial_overlap_disclosed": True,
        "preflight_hash": preflight_record["preflight_hash"],
        "frozen_method_hash": frozen_method["frozen_method_hash"],
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "selected_budget": budget,
        "selected_methods": {
            stratum: {
                name: selected_method[stratum][name]["selected_family"]
                for name in OUTPUT_NAMES
            }
            for stratum in ("interpolation", "boundary", "ood")
        },
        "method_selection_point_metrics": next(
            item for item in candidates if item["budget"] == budget
        )["point_metrics"],
        "method_selection_interval_diagnostics_passed": selected_method["all_gates_passed"],
        "assessment_metrics": metrics,
        "final_calibration_labels_accessed_after_method_freeze": True,
        "assessment_labels_accessed_once_after_calibration_freeze": True,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "status": "accepted" if accepted else "failed-predeclared-assessment-gates",
        "valid_prospective_result": True,
        "claim": "prospective same-domain deterministic L0 software-emulation validation; not ahistorical independence or physical accuracy",
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
            "document_type": "cft-revival-l0-surrogate-v6-execution-failure",
            "schema_version": "6.0",
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
