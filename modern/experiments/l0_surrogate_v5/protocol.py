"""Preregistered blind continuous-design L0 surrogate experiment v5."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from hashlib import sha256
from math import fsum, sqrt
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.physics import evaluate_batch, load_l0_json
from cft_revival.surrogates import ExactGP, SurrogateSchema
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads
from experiments.l0_surrogate_v2 import protocol as science
from experiments.l0_surrogate_v3.serialization import AtomicArtifactStore

from .design import (
    group_key,
    normalized_design,
    operating_points,
    partitions as build_role_partitions,
    surrogate_inputs,
)
from .identity import acquire_exclusive_lock, bind
from .intervals import (
    fit_parameters,
    interval,
    nearest_training_distances,
    select_method,
)

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
    declared = value["predeclaration_hash"]
    if declared != canonical_hash(
        {key: item for key, item in value.items() if key != "predeclaration_hash"}
    ):
        raise ValueError("v5 predeclaration hash mismatch")
    return value


def _source_ranges(declaration: Mapping[str, object]) -> Mapping[str, object]:
    config = load_l0_json(SOURCE_CONFIG)
    if canonical_hash(config) != declaration["design"]["source_config_hash"]:  # type: ignore[index]
        raise ValueError("accepted source config hash changed")
    ranges = config["ranges"]
    if not isinstance(ranges, Mapping):
        raise ValueError("accepted source ranges are malformed")
    return ranges


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

    bases = (2, 3, 5, 7, 13)
    return tuple(
        tuple(radical_inverse(start + row, base) for base in bases)
        for row in range(1, int(config["batch_size"]) + 1)
    )


def _prior_heldout_indices(path: Path) -> tuple[int, ...]:
    value = _load(path)
    indices = set()
    for replicate in value["replicates"]:  # type: ignore[union-attr]
        for role in ("calibration", "assessment"):
            for split in replicate[role].values():
                indices.update(int(index) for index in split["indices"])
    return tuple(sorted(indices))


def build_partitions() -> dict[str, object]:
    declaration = load_declaration()
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    replicates = build_role_partitions(inputs, declaration)
    old = _old_grid_inputs()
    prior_records = {}
    union_coordinates = set()
    for version in ("v3", "v4"):
        path = MODERN / "experiments" / f"l0_surrogate_{version}" / "partitions.json"
        prior_indices = _prior_heldout_indices(path)
        coordinates = {old[index] for index in prior_indices}
        union_coordinates.update(coordinates)
        prior_records[version] = {
            "partition_file_sha256": sha256(path.read_bytes()).hexdigest(),
            "heldout_index_count": len(prior_indices),
            "heldout_coordinate_hash": canonical_hash(
                {"coordinates": [list(value) for value in sorted(coordinates)]}
            ),
            "v5_intersection_count": len(set(inputs).intersection(coordinates)),
        }
    if set(inputs).intersection(union_coordinates):
        raise ValueError("v5 reuses a v3/v4 heldout surrogate coordinate")
    record: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v5-input-partitions",
        "schema_version": "5.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "normalized_design_hash": canonical_hash(
            {"normalized_design": [list(row) for row in normalized]}
        ),
        "surrogate_input_hash": canonical_hash(
            {"inputs": [list(row) for row in inputs]}
        ),
        "rows": len(inputs),
        "prior_disjointness": prior_records,
        "combined_prior_coordinate_intersection_count": 0,
        "replicates": replicates,
        "label_policy": "input-only; no new v5 physics result was evaluated",
    }
    record["partitions_hash"] = canonical_hash(record)
    return record


def write_partitions() -> dict[str, object]:
    record = build_partitions()
    AtomicArtifactStore(ROOT).write_json("partitions.json", record)
    return record


def preflight(*, record: bool = False) -> dict[str, object]:
    declaration = load_declaration()
    partition_record = build_partitions()
    with tempfile.TemporaryDirectory(prefix="l0-v5-preflight-") as temporary:
        store = AtomicArtifactStore(Path(temporary))
        store.write_json("deep/a/payload.json", {"synthetic": True})
        synthetic = ExactGP.fit(
            ((0.0,), (0.5,), (1.0,)),
            (0.0, 0.25, 1.0),
            schema=SurrogateSchema(("synthetic_x",), ("synthetic_y",), ("1",), ("1",)),
            length_scale_mode="isotropic",
        )
        for seed in declaration["partition"]["replicate_seeds"]:  # type: ignore[index]
            replicate = f"split-{seed}"
            for campaign in ("active", "fixed-baseline"):
                for output in OUTPUT_NAMES:
                    store.write_model(
                        f"{replicate}/{campaign}/{output}.model.json", synthetic
                    )
                for artifact in (
                    "selection",
                    "method-selection",
                    "calibration",
                    "assessment",
                ):
                    store.write_json(
                        f"{replicate}/{campaign}/{artifact}.json",
                        {"synthetic": True, "artifact": artifact},
                    )
            store.write_json(
                f"{replicate}/frozen-before-assessment.json",
                {"synthetic": True, "assessment_access_count": 0},
            )
        store.write_json("run-manifest.json", {"synthetic": True})
        store.write_json("runtime-diagnostics.json", {"synthetic": True})
        store.write_json("failure-manifest.json", {"synthetic": True})
        temporary_files = store.temporary_files()
        files = sorted(
            str(path.relative_to(temporary)).replace("\\", "/")
            for path in Path(temporary).rglob("*")
            if path.is_file()
        )
    result: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v5-synthetic-preflight",
        "schema_version": "5.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partition_record["partitions_hash"],
        "serialization_files": files,
        "temporary_file_count_after_writes": len(temporary_files),
        "assessment_access_count": 0,
        "real_physics_evaluation_count": 0,
        "passed": not temporary_files,
    }
    result["preflight_hash"] = canonical_hash(result)
    if record:
        AtomicArtifactStore(ROOT).write_json("preflight.json", result)
    return result


class RoleOracle:
    def __init__(self, points: Sequence[object], allowed: Sequence[int]) -> None:
        self._points = points
        self._allowed = frozenset(allowed)
        self._cache: dict[int, tuple[float, float]] = {}

    def observe(self, index: int) -> tuple[float, float]:
        if index not in self._allowed:
            raise ValueError("label access outside the oracle role")
        if index not in self._cache:
            result = evaluate_batch((self._points[index],))[0]  # type: ignore[arg-type]
            self._cache[index] = (
                result.axial_thrust_n,
                result.specific_impulse_s,
            )
        return self._cache[index]

    def observe_many(self, indices: Sequence[int]) -> dict[int, tuple[float, float]]:
        missing = tuple(index for index in indices if index not in self._cache)
        if any(index not in self._allowed for index in missing):
            raise ValueError("batch label access outside the oracle role")
        if missing:
            results = evaluate_batch(tuple(self._points[index] for index in missing))  # type: ignore[arg-type]
            for index, result in zip(missing, results, strict=True):
                self._cache[index] = (
                    result.axial_thrust_n,
                    result.specific_impulse_s,
                )
        return {index: self._cache[index] for index in indices}


class SingleUseAssessment:
    def __init__(self, points: Sequence[object], role: Mapping[str, object]) -> None:
        self._points = points
        self._role = role
        self._used = False

    def load(self, frozen_hash: str, expected_hash: str) -> dict[str, dict[int, tuple[float, float]]]:
        if self._used:
            raise RuntimeError("assessment labels may be loaded once only")
        if frozen_hash != expected_hash:
            raise ValueError("frozen artifact hash changed before assessment")
        self._used = True
        result = {}
        for stratum in ("interpolation", "boundary", "ood"):
            indices = tuple(int(value) for value in self._role[stratum]["indices"])  # type: ignore[index]
            evaluated = evaluate_batch(tuple(self._points[index] for index in indices))  # type: ignore[arg-type]
            result[stratum] = {
                index: (value.axial_thrust_n, value.specific_impulse_s)
                for index, value in zip(indices, evaluated, strict=True)
            }
        return result


def _save_models(
    store: AtomicArtifactStore,
    replicate: str,
    campaign: str,
    models: Sequence[ExactGP],
) -> dict[str, str]:
    result = {}
    for name, model in zip(OUTPUT_NAMES, models, strict=True):
        store.write_model(f"{replicate}/{campaign}/{name}.model.json", model)
        result[name] = model.model_hash
    return result


def _role_groups(role: Mapping[str, object], stratum: str) -> dict[str, tuple[int, ...]]:
    split = role[stratum]
    return {
        str(group): tuple(
            int(index)
            for index in split["indices"]  # type: ignore[index]
            if group_key(_CURRENT_INPUTS[int(index)], _CURRENT_PARTITION_POLICY) == group
        )
        for group in split["groups"]  # type: ignore[index]
    }


_CURRENT_INPUTS: tuple[tuple[float, ...], ...] = ()
_CURRENT_PARTITION_POLICY: Mapping[str, object] = {}


def _method_and_calibration(
    models: Sequence[ExactGP],
    training_indices: Sequence[int],
    role_method: Mapping[str, object],
    role_calibration: Mapping[str, object],
    points: Sequence[object],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    training = tuple(inputs[index] for index in training_indices)
    gates = declaration["gates"]
    method_record: dict[str, object] = {}
    calibration_record: dict[str, object] = {}
    nominal = 0.9
    for stratum in ("interpolation", "boundary", "ood"):
        method_indices = tuple(int(value) for value in role_method[stratum]["indices"])  # type: ignore[index]
        method_truth = RoleOracle(points, method_indices).observe_many(method_indices)
        method_predictions = science.predict_rows(models, method_indices, inputs)
        method_distances = nearest_training_distances(
            tuple(inputs[index] for index in method_indices), training
        )
        groups = _role_groups(role_method, stratum)
        method_record[stratum] = {}
        calibration_record[stratum] = {}
        calibration_indices = tuple(
            int(value) for value in role_calibration[stratum]["indices"]  # type: ignore[index]
        )
        calibration_truth = RoleOracle(points, calibration_indices).observe_many(
            calibration_indices
        )
        calibration_predictions = science.predict_rows(
            models, calibration_indices, inputs
        )
        calibration_distances = nearest_training_distances(
            tuple(inputs[index] for index in calibration_indices), training
        )
        for output, name in enumerate(OUTPUT_NAMES):
            selected = select_method(
                groups,
                {index: value[output] for index, value in method_truth.items()},
                {
                    index: method_predictions[index][output]
                    for index in method_indices
                },
                dict(zip(method_indices, method_distances, strict=True)),
                nominal=nominal,
                output_scale=float(gates["quality_scales"][name]),  # type: ignore[index]
                coverage_bounds=tuple(gates["group_heldout_method_selection_coverage_interval"]),  # type: ignore[arg-type]
                maximum_group_deviation=float(
                    gates["maximum_group_coverage_deviation"]  # type: ignore[index]
                ),
            )
            method_record[stratum][name] = selected  # type: ignore[index]
            family = str(selected["selected_family"])
            parameters = fit_parameters(
                family,
                tuple(calibration_truth[index][output] for index in calibration_indices),
                tuple(
                    calibration_predictions[index][output]
                    for index in calibration_indices
                ),
                calibration_distances,
                nominal=nominal,
            )
            calibration_record[stratum][name] = {  # type: ignore[index]
                "selected_family": family,
                "parameters": parameters,
                "groups": role_calibration[stratum]["groups"],  # type: ignore[index]
                "indices": list(calibration_indices),
                "labels": [
                    calibration_truth[index][output] for index in calibration_indices
                ],
                "predicted_means": [
                    calibration_predictions[index][output].mean
                    for index in calibration_indices
                ],
                "scales": list(
                    nearest_training_distances(
                        tuple(inputs[index] for index in calibration_indices), training
                    )
                    if family.endswith("input-distance")
                    else (
                        tuple(
                            calibration_predictions[index][output].standard_deviation
                            for index in calibration_indices
                        )
                        if family.endswith("raw-gp-sd")
                        else (1.0,) * len(calibration_indices)
                    )
                ),
            }
    method_record["all_diagnostic_gates_passed"] = all(
        method_record[stratum][name]["diagnostic_gates_passed"]  # type: ignore[index]
        for stratum in ("interpolation", "boundary", "ood")
        for name in OUTPUT_NAMES
    )
    return method_record, calibration_record


def _assessment(
    models: Sequence[ExactGP],
    selected: Sequence[int],
    calibration: Mapping[str, object],
    labels: Mapping[str, Mapping[int, tuple[float, float]]],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    gates = declaration["gates"]
    train = tuple(inputs[index] for index in selected)
    raw: dict[str, object] = {}
    all_rows: dict[str, list[dict[str, float | bool]]] = {
        name: [] for name in OUTPUT_NAMES
    }
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(labels[stratum])
        predictions = science.predict_rows(models, indices, inputs)
        distances = nearest_training_distances(
            tuple(inputs[index] for index in indices), train
        )
        raw[stratum] = {}
        for output, name in enumerate(OUTPUT_NAMES):
            rows = []
            parameters = calibration[stratum][name]["parameters"]  # type: ignore[index]
            for index, distance in zip(indices, distances, strict=True):
                prediction = predictions[index][output]
                lower, upper = interval(parameters, prediction, distance)
                truth = labels[stratum][index][output]
                rows.append(
                    {
                        "index": index,
                        "truth": truth,
                        "predicted_mean": prediction.mean,
                        "predicted_standard_deviation": prediction.standard_deviation,
                        "lower": lower,
                        "upper": upper,
                        "covered": lower <= truth <= upper,
                    }
                )
            raw[stratum][name] = rows  # type: ignore[index]
            all_rows[name].extend(rows)
    metrics: dict[str, object] = {}
    for scope in ("interpolation", "boundary", "ood", "overall"):
        metrics[scope] = {}
        for name in OUTPUT_NAMES:
            rows = (
                all_rows[name]
                if scope == "overall"
                else raw[scope][name]  # type: ignore[index]
            )
            scale = float(gates["quality_scales"][name])  # type: ignore[index]
            errors = [
                abs(float(row["truth"]) - float(row["predicted_mean"]))
                for row in rows
            ]
            nrmse = sqrt(fsum(value * value for value in errors) / len(errors)) / scale
            worst = max(errors) / scale
            coverage = fsum(1.0 for row in rows if row["covered"]) / len(rows)
            coverage_bounds = gates["coverage_interval"]  # type: ignore[index]
            record = {
                "rows": len(rows),
                "range_normalized_rmse": nrmse,
                "worst_range_normalized_error": worst,
                "coverage": coverage,
                "nrmse_passed": nrmse <= float(gates["range_normalized_rmse_maximum"]),  # type: ignore[index]
                "worst_error_passed": worst
                <= float(gates["worst_case_range_normalized_error_maximum"]),  # type: ignore[index]
                "coverage_passed": float(coverage_bounds[0])
                <= coverage
                <= float(coverage_bounds[1]),
            }
            record["all_gates_passed"] = all(
                record[key]
                for key in ("nrmse_passed", "worst_error_passed", "coverage_passed")
            )
            metrics[scope][name] = record  # type: ignore[index]
    metrics["all_scopes_outputs_passed"] = all(
        metrics[scope][name]["all_gates_passed"]  # type: ignore[index]
        for scope in ("interpolation", "boundary", "ood", "overall")
        for name in OUTPUT_NAMES
    )
    return raw, metrics


def _execute_after_binding(binding: object, lock: Path, started: float) -> dict[str, object]:
    declaration = load_declaration()
    partition_record = _load(PARTITIONS)
    recorded_preflight = _load(PREFLIGHT)
    if not recorded_preflight["passed"] or recorded_preflight["assessment_access_count"]:
        raise ValueError("recorded preflight is invalid")
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    points = operating_points(normalized, _source_ranges(declaration))
    global _CURRENT_INPUTS, _CURRENT_PARTITION_POLICY
    _CURRENT_INPUTS = inputs
    _CURRENT_PARTITION_POLICY = declaration["partition"]  # type: ignore[assignment]
    store = AtomicArtifactStore(RESULTS)
    completed = []
    for replicate in partition_record["replicates"]:  # type: ignore[union-attr]
        replicate_id = str(replicate["replicate_id"])
        candidate = tuple(int(value) for value in replicate["candidate_indices"])
        training_oracle = RoleOracle(points, candidate)
        active_selected, active_observed, rounds = science.select_active_indices(
            inputs,
            candidate,
            training_oracle,  # type: ignore[arg-type]
            declaration,
        )
        baseline_selected = science.baseline_indices(candidate, declaration)
        baseline_observed = RoleOracle(points, candidate).observe_many(
            baseline_selected
        )
        selections = {
            "active": {
                "selected_indices": list(active_selected),
                "rounds": rounds,
                "selection_hash": canonical_hash(list(active_selected)),
            },
            "fixed-baseline": {
                "selected_indices": list(baseline_selected),
                "selection_hash": canonical_hash(list(baseline_selected)),
            },
        }
        models_by_campaign = {
            "active": science.fit_models(active_selected, inputs, active_observed),
            "fixed-baseline": science.fit_models(
                baseline_selected, inputs, baseline_observed
            ),
        }
        method_records = {}
        calibrations = {}
        model_hashes = {}
        for campaign, selected in (
            ("active", active_selected),
            ("fixed-baseline", baseline_selected),
        ):
            models = models_by_campaign[campaign]
            model_hashes[campaign] = _save_models(
                store, replicate_id, campaign, models
            )
            method, calibration = _method_and_calibration(
                models,
                selected,
                replicate["method-selection"],
                replicate["final-calibration"],
                points,
                inputs,
                declaration,
            )
            method["method_selection_hash"] = canonical_hash(method)
            calibration["calibration_hash"] = canonical_hash(calibration)
            method_records[campaign] = method
            calibrations[campaign] = calibration
            store.write_json(f"{replicate_id}/{campaign}/selection.json", selections[campaign])
            store.write_json(f"{replicate_id}/{campaign}/method-selection.json", method)
            store.write_json(f"{replicate_id}/{campaign}/calibration.json", calibration)
        frozen = {
            "partition_hash": replicate["replicate_partition_hash"],
            "selection_hashes": {
                key: value["selection_hash"] for key, value in selections.items()
            },
            "model_hashes": model_hashes,
            "method_selection_hashes": {
                key: value["method_selection_hash"]
                for key, value in method_records.items()
            },
            "calibration_hashes": {
                key: value["calibration_hash"]
                for key, value in calibrations.items()
            },
        }
        frozen_hash = canonical_hash(frozen)
        store.write_json(
            f"{replicate_id}/frozen-before-assessment.json",
            {**frozen, "frozen_hash": frozen_hash},
        )
        labels = SingleUseAssessment(points, replicate["assessment"]).load(
            frozen_hash, frozen_hash
        )
        campaign_results = {}
        for campaign, selected in (
            ("active", active_selected),
            ("fixed-baseline", baseline_selected),
        ):
            raw, metrics = _assessment(
                models_by_campaign[campaign],
                selected,
                calibrations[campaign],
                labels,
                inputs,
                declaration,
            )
            assessment = {
                "document_type": "cft-revival-l0-surrogate-v5-final-assessment",
                "schema_version": "5.0",
                "replicate_id": replicate_id,
                "campaign": campaign,
                "frozen_hash": frozen_hash,
                "raw": raw,
                "metrics": metrics,
            }
            assessment["assessment_hash"] = canonical_hash(assessment)
            store.write_json(f"{replicate_id}/{campaign}/assessment.json", assessment)
            campaign_results[campaign] = metrics
        result = {
            "replicate_id": replicate_id,
            "active": campaign_results["active"],
            "fixed-baseline": campaign_results["fixed-baseline"],
            "active_method_diagnostics_passed": method_records["active"][
                "all_diagnostic_gates_passed"
            ],
            "fixed_baseline_method_diagnostics_passed": method_records[
                "fixed-baseline"
            ]["all_diagnostic_gates_passed"],
            "active_passed": campaign_results["active"][
                "all_scopes_outputs_passed"
            ]
            and method_records["active"]["all_diagnostic_gates_passed"],
            "fixed_baseline_passed": campaign_results["fixed-baseline"][
                "all_scopes_outputs_passed"
            ]
            and method_records["fixed-baseline"][
                "all_diagnostic_gates_passed"
            ],
            "selected_methods": {
                campaign: {
                    stratum: {
                        name: method_records[campaign][stratum][name][
                            "selected_family"
                        ]
                        for name in OUTPUT_NAMES
                    }
                    for stratum in ("interpolation", "boundary", "ood")
                }
                for campaign in ("active", "fixed-baseline")
            },
            "frozen_hash": frozen_hash,
        }
        result["replicate_result_hash"] = canonical_hash(result)
        completed.append(result)
    accepted = all(item["active_passed"] for item in completed)
    manifest: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v5-run-manifest",
        "schema_version": "5.0",
        "commit_binding": binding.to_dict(),
        "exclusive_lock": {
            "path_role": "git-common-directory",
            "lock_file_name": lock.name,
            "atomic_create": "O_CREAT|O_EXCL",
            "retained": True,
            "content_commit_sha": binding.commit_sha,
        },
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partition_record["partitions_hash"],
        "normalized_design_hash": partition_record["normalized_design_hash"],
        "prior_coordinate_intersection_count": partition_record[
            "combined_prior_coordinate_intersection_count"
        ],
        "preflight_hash": recorded_preflight["preflight_hash"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
            "detached_worktree": True,
        },
        "replicates": completed,
        "scientific_identity_valid": True,
        "all_active_replicates_passed": accepted,
        "status": "accepted" if accepted else "failed-predeclared-gates",
        "claim": "deterministic L0 software-emulation accuracy only; no physical claim",
    }
    manifest["run_manifest_hash"] = canonical_hash(manifest)
    store.write_json("run-manifest.json", manifest)
    store.write_json(
        "runtime-diagnostics.json",
        {
            "wall_seconds": perf_counter() - started,
            "diagnostic_only": True,
            "run_manifest_hash": manifest["run_manifest_hash"],
        },
    )
    return manifest


def execute() -> dict[str, object]:
    started = perf_counter()
    binding = bind(REPO, DEPENDENCIES)
    lock = acquire_exclusive_lock(REPO, binding.commit_sha)
    try:
        return _execute_after_binding(binding, lock, started)
    except Exception as error:
        failure: dict[str, object] = {
            "document_type": "cft-revival-l0-surrogate-v5-execution-failure",
            "schema_version": "5.0",
            "commit_binding": binding.to_dict(),
            "exclusive_lock": {
                "lock_file_name": lock.name,
                "atomic_create": "O_CREAT|O_EXCL",
                "retained": True,
                "content_commit_sha": binding.commit_sha,
            },
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "wall_seconds": perf_counter() - started,
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
