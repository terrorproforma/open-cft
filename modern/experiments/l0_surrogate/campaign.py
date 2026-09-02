"""Reproducible active-learning experiment against the accepted L0 software sweep.

The 8,192 L0 evaluations are numerical truth only for software emulation.  This
module does not make a physical-accuracy claim and deliberately uses only
accepted public physics, surrogate, and active-learning APIs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from math import sqrt
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.active_learning import (
    AcquisitionWeights,
    FidelitySource,
    PosteriorPrediction,
    score_candidate,
)
from cft_revival.physics import evaluate_sweep_artifact, load_l0_json
from cft_revival.surrogates import (
    ExactGP,
    Prediction,
    SurrogateSchema,
    VarianceCalibrator,
    regression_metrics,
)
from cft_revival.surrogates.identity import (
    canonical_hash,
    strict_json_loads,
)

MODERN = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
PREDECLARATION = EXPERIMENT / "predeclared_campaign.json"
DEFAULT_ARTIFACT_DIR = EXPERIMENT / "artifacts"

OUTPUT_NAMES = ("axial_thrust_n", "specific_impulse_s")
OUTPUT_UNITS = ("N", "s")
SOURCE = FidelitySource("L0-software-emulator", rank=0, cost=1.0, is_highest=True)


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_predeclaration(path: Path = PREDECLARATION) -> dict[str, object]:
    decoded = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("campaign predeclaration must be a JSON object")
    declared_hash = decoded.get("predeclaration_hash")
    unhashed = {key: value for key, value in decoded.items() if key != "predeclaration_hash"}
    if declared_hash != canonical_hash(unhashed):
        raise ValueError("campaign predeclaration hash mismatch")
    if decoded["document_type"] != "cft-revival-l0-surrogate-campaign-predeclaration":
        raise ValueError("unexpected campaign predeclaration document type")
    return decoded


def _feature_row(
    point: Mapping[str, object],
    ranges: Mapping[str, object],
) -> tuple[float, ...]:
    raw_input = point["input"]
    if not isinstance(raw_input, Mapping):
        raise ValueError("L0 point input must be an object")
    fractions = raw_input["charge_state_number_fractions"]
    if not isinstance(fractions, Mapping):
        raise ValueError("L0 charge fractions must be an object")
    neutral = float(fractions["xe_neutral"])
    ionized = 1.0 - neutral
    double_share = float(fractions["xe_double_plus"]) / ionized
    values = (
        float(raw_input["discharge_voltage_v"]),
        float(raw_input["propellant_mass_flow_kg_per_s"]),
        ionized,
        double_share,
        float(raw_input["axial_momentum_fraction_of_ion_momentum"]),
    )
    range_names = (
        "discharge_voltage_v",
        "propellant_mass_flow_kg_per_s",
        "ionized_number_fraction",
        "xe_double_plus_fraction_of_ions",
        "axial_momentum_fraction_of_ion_momentum",
    )
    normalized: list[float] = []
    for value, name in zip(values, range_names, strict=True):
        bounds = ranges[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"invalid source range for {name}")
        lower, upper = (float(item) for item in bounds)
        coordinate = (value - lower) / (upper - lower)
        if not -1.0e-12 <= coordinate <= 1.0 + 1.0e-12:
            raise ValueError(f"source coordinate {name} lies outside declared bounds")
        normalized.append(min(1.0, max(0.0, coordinate)))
    return tuple(normalized)


def _output_row(point: Mapping[str, object]) -> tuple[float, float]:
    result = point["result"]
    if not isinstance(result, Mapping):
        raise ValueError("L0 point result must be an object")
    return float(result["axial_thrust_n"]), float(result["specific_impulse_s"])


def _group_key(row: Sequence[float], bins: int) -> str:
    coordinates = tuple(min(bins - 1, int(row[index] * bins)) for index in (0, 1, 2))
    return "-".join(str(value) for value in coordinates)


def _group_stratum(group: str, bins: int) -> str | None:
    voltage, mass_flow, ionized = (int(value) for value in group.split("-"))
    if voltage >= bins - 2 and ionized >= bins - 2:
        return "ood"
    if (
        voltage in {0, bins - 1}
        or mass_flow in {0, bins - 1}
        or ionized in {0, bins - 1}
    ):
        return "boundary"
    if all(2 <= value <= bins - 3 for value in (voltage, mass_flow, ionized)):
        return "interpolation"
    return None


def _ordered_groups(groups: Sequence[str], seed: int, role: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                sha256(f"{seed}:{role}:{group}".encode("utf-8")).hexdigest(),
                group,
            ),
        )
    )


def _take_whole_groups(
    candidates: Sequence[str],
    group_rows: Mapping[str, tuple[int, ...]],
    minimum_rows: int,
    *,
    seed: int,
    role: str,
) -> tuple[str, ...]:
    selected: list[str] = []
    count = 0
    for group in _ordered_groups(candidates, seed, role):
        selected.append(group)
        count += len(group_rows[group])
        if count >= minimum_rows:
            return tuple(selected)
    raise ValueError(f"not enough whole spatial groups for {role}")


def build_partition(
    inputs: Sequence[Sequence[float]],
    predeclaration: Mapping[str, object],
) -> dict[str, object]:
    split = predeclaration["split"]
    if not isinstance(split, Mapping):
        raise ValueError("split declaration must be an object")
    seed = int(split["seed"])
    bins = int(split["spatial_group_bins_per_dimension"])
    minimum = int(split["minimum_points_per_assessment_stratum"])
    calibration_minimum = int(split["calibration_fit_minimum_points"])

    group_lists: dict[str, list[int]] = {}
    for index, row in enumerate(inputs):
        group_lists.setdefault(_group_key(row, bins), []).append(index)
    group_rows = {group: tuple(indices) for group, indices in group_lists.items()}

    assessment_groups: dict[str, tuple[str, ...]] = {}
    reserved: set[str] = set()
    for stratum in ("interpolation", "boundary", "ood"):
        candidates = tuple(
            group
            for group in group_rows
            if group not in reserved and _group_stratum(group, bins) == stratum
        )
        chosen = _take_whole_groups(
            candidates,
            group_rows,
            minimum,
            seed=seed,
            role=f"assessment:{stratum}",
        )
        assessment_groups[stratum] = chosen
        reserved.update(chosen)

    calibration_candidates = tuple(
        group
        for group in group_rows
        if group not in reserved
        and all(1 <= int(value) <= bins - 2 for value in group.split("-"))
    )
    calibration_groups = _take_whole_groups(
        calibration_candidates,
        group_rows,
        calibration_minimum,
        seed=seed,
        role="calibration-fit",
    )
    reserved.update(calibration_groups)

    assessment_indices = {
        stratum: tuple(
            index
            for group in groups
            for index in group_rows[group]
        )
        for stratum, groups in assessment_groups.items()
    }
    calibration_indices = tuple(
        index for group in calibration_groups for index in group_rows[group]
    )
    eligible_indices = tuple(
        index
        for index, row in enumerate(inputs)
        if _group_key(row, bins) not in reserved
    )
    if len(eligible_indices) < int(predeclaration["campaign"]["maximum_training_rows"]):  # type: ignore[index]
        raise ValueError("split leaves too few campaign-eligible points")

    result: dict[str, object] = {
        "policy": split["policy"],
        "seed": seed,
        "bins_per_group_dimension": bins,
        "group_dimensions": list(split["spatial_group_dimensions"]),
        "assessment_groups": {
            stratum: list(groups) for stratum, groups in assessment_groups.items()
        },
        "assessment_indices": {
            stratum: list(indices) for stratum, indices in assessment_indices.items()
        },
        "calibration_groups": list(calibration_groups),
        "calibration_indices": list(calibration_indices),
        "eligible_indices": list(eligible_indices),
    }
    result["partition_hash"] = canonical_hash(result)
    return result


def _fit_uncalibrated_models(
    train_indices: Sequence[int],
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[Sequence[float]],
    nominal_probability: float,
) -> tuple[ExactGP, ...]:
    train_x = tuple(inputs[index] for index in train_indices)
    models = []
    for output, (name, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS, strict=True)):
        schema = SurrogateSchema(
            (
                "normalized_discharge_voltage",
                "normalized_propellant_mass_flow",
                "normalized_ionized_fraction",
                "normalized_xe_double_plus_share_of_ions",
                "normalized_axial_momentum_fraction",
            ),
            (name,),
            ("1",) * 5,
            (unit,),
        )
        models.append(
            ExactGP.fit(
                train_x,
                tuple(outputs[index][output] for index in train_indices),
                schema=schema,
                length_scale_mode="ard",
                nominal_probability=nominal_probability,
            )
        )
    return tuple(models)


def _calibrators(
    models: Sequence[ExactGP],
    calibration_indices: Sequence[int],
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[Sequence[float]],
    nominal_probability: float,
) -> tuple[VarianceCalibrator, ...]:
    calibration_x = tuple(inputs[index] for index in calibration_indices)
    return tuple(
        VarianceCalibrator.fit(
            tuple(outputs[index][output] for index in calibration_indices),
            model.predict(calibration_x),
            nominal_probability=nominal_probability,
        )
        for output, model in enumerate(models)
    )


def _predict(
    models: Sequence[ExactGP],
    calibrators: Sequence[VarianceCalibrator],
    indices: Sequence[int],
    inputs: Sequence[Sequence[float]],
) -> tuple[tuple[Prediction, ...], ...]:
    points = tuple(inputs[index] for index in indices)
    columns = tuple(
        tuple(calibrator.apply(prediction) for prediction in model.predict(points))
        for model, calibrator in zip(models, calibrators, strict=True)
    )
    return tuple(
        tuple(column[row] for column in columns)
        for row in range(len(points))
    )


def _metric_record(
    truth: Sequence[float],
    predictions: Sequence[Prediction],
    quality_scale: float,
    gates: Mapping[str, object],
    nominal_probability: float,
) -> dict[str, object]:
    metrics = regression_metrics(
        truth,
        predictions,
        nominal_probability=nominal_probability,
        coverage_target=float(gates["interval_coverage_target"]),
        coverage_tolerance=float(gates["interval_coverage_absolute_tolerance"]),
        minimum_coverage_sample_count=int(gates["minimum_coverage_sample_count"]),
        rmse_acceptance_threshold=float(gates["range_normalized_rmse_maximum"]),
        quality_scale=quality_scale,
        assessment_role="held-out-assessment",
    )
    record = asdict(metrics)
    worst_normalized = metrics.worst_case_absolute_error / quality_scale
    record["worst_case_range_normalized_absolute_error"] = worst_normalized
    record["worst_case_accepted"] = (
        worst_normalized
        <= float(gates["worst_case_range_normalized_absolute_error_maximum"])
    )
    record["all_predeclared_output_gates_passed"] = bool(
        metrics.rmse_accepted
        and metrics.coverage_accepted
        and record["worst_case_accepted"]
    )
    return record


def _evaluate_models(
    models: Sequence[ExactGP],
    calibrators: Sequence[VarianceCalibrator],
    partition: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[Sequence[float]],
    output_ranges: Sequence[float],
    gates: Mapping[str, object],
    nominal_probability: float,
) -> dict[str, object]:
    by_stratum = partition["assessment_indices"]
    if not isinstance(by_stratum, Mapping):
        raise ValueError("partition assessment indices must be an object")
    stratum_records: dict[str, object] = {}
    combined_indices: list[int] = []
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(int(value) for value in by_stratum[stratum])  # type: ignore[arg-type]
        combined_indices.extend(indices)
        predictions = _predict(models, calibrators, indices, inputs)
        stratum_records[stratum] = {
            name: _metric_record(
                tuple(outputs[index][output] for index in indices),
                tuple(row[output] for row in predictions),
                output_ranges[output],
                gates,
                nominal_probability,
            )
            for output, name in enumerate(OUTPUT_NAMES)
        }

    combined = tuple(combined_indices)
    predictions = _predict(models, calibrators, combined, inputs)
    combined_records = {
        name: _metric_record(
            tuple(outputs[index][output] for index in combined),
            tuple(row[output] for row in predictions),
            output_ranges[output],
            gates,
            nominal_probability,
        )
        for output, name in enumerate(OUTPUT_NAMES)
    }

    detector = models[0].ood_detector()
    ood_records: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        indices = tuple(int(value) for value in by_stratum[stratum])  # type: ignore[arg-type]
        reports = tuple(detector.report(inputs[index]) for index in indices)
        ood_records[stratum] = {
            "sample_count": len(reports),
            "detected_ood_count": sum(report.is_out_of_distribution for report in reports),
            "detected_ood_fraction": (
                sum(report.is_out_of_distribution for report in reports) / len(reports)
            ),
            "maximum_nearest_training_distance": max(
                report.nearest_training_distance for report in reports
            ),
            "maximum_domain_excess_distance": max(
                report.domain_excess_distance for report in reports
            ),
            "threshold": detector.threshold,
            "policy_version": reports[0].policy_version,
        }

    status = all(
        bool(record["all_predeclared_output_gates_passed"])
        for record in combined_records.values()
    )
    return {
        "combined": combined_records,
        "strata": stratum_records,
        "ood": ood_records,
        "all_predeclared_gates_passed": status,
    }


class _CachedPosterior:
    def __init__(
        self,
        rows: Mapping[tuple[float, ...], tuple[Prediction, ...]],
        previous_rows: Mapping[tuple[float, ...], tuple[Prediction, ...]] | None,
    ) -> None:
        self._rows = rows
        self._previous = previous_rows

    def predict(
        self,
        design: Sequence[float],
        source: FidelitySource,
    ) -> PosteriorPrediction:
        if source != SOURCE:
            raise ValueError("unexpected active-learning source")
        key = tuple(float(value) for value in design)
        predictions = self._rows[key]
        previous = None if self._previous is None else self._previous[key]
        return PosteriorPrediction(
            objective_means=tuple(item.mean for item in predictions),
            epistemic_standard_deviations=tuple(
                item.standard_deviation for item in predictions
            ),
            aleatoric_standard_deviations=(0.0,) * len(predictions),
            discrepancy_means=(
                (0.0,) * len(predictions)
                if previous is None
                else tuple(
                    item.mean - old.mean
                    for item, old in zip(predictions, previous, strict=True)
                )
            ),
        )


def _prediction_map(
    models: Sequence[ExactGP],
    calibrators: Sequence[VarianceCalibrator],
    indices: Sequence[int],
    inputs: Sequence[Sequence[float]],
) -> dict[tuple[float, ...], tuple[Prediction, ...]]:
    rows = _predict(models, calibrators, indices, inputs)
    return {
        tuple(inputs[index]): row
        for index, row in zip(indices, rows, strict=True)
    }


def _acquire(
    *,
    count: int,
    selected_indices: Sequence[int],
    eligible_indices: Sequence[int],
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[Sequence[float]],
    models: Sequence[ExactGP],
    calibrators: Sequence[VarianceCalibrator],
    previous_models: Sequence[ExactGP] | None,
    previous_calibrators: Sequence[VarianceCalibrator] | None,
    weights: AcquisitionWeights,
    pending_length_scale: float,
    output_ranges: Sequence[float],
) -> tuple[int, ...]:
    selected = set(selected_indices)
    remaining = tuple(index for index in eligible_indices if index not in selected)
    current = _prediction_map(models, calibrators, remaining, inputs)
    previous = (
        None
        if previous_models is None or previous_calibrators is None
        else _prediction_map(previous_models, previous_calibrators, remaining, inputs)
    )
    adapter = _CachedPosterior(current, previous)
    incumbent = tuple(
        max(outputs[index][output] for index in selected_indices)
        for output in range(len(OUTPUT_NAMES))
    )
    acquired: list[int] = []
    pending: list[tuple[Sequence[float], PosteriorPrediction]] = []
    available = set(remaining)
    for _ in range(count):
        scored = []
        for index in available:
            score = score_candidate(
                inputs[index],
                SOURCE,
                adapter,
                incumbent,
                pending=pending,
                weights=weights,
                pending_length_scale=pending_length_scale,
                uncertainty_scales=output_ranges,
            )
            scored.append((score.cost_normalized_score, index))
        _, chosen = max(scored, key=lambda item: (item[0], -item[1]))
        acquired.append(chosen)
        available.remove(chosen)
        pending.append((inputs[chosen], adapter.predict(inputs[chosen], SOURCE)))
    return tuple(acquired)


def _fit_calibrated_final_models(
    uncalibrated: Sequence[ExactGP],
    calibrators: Sequence[VarianceCalibrator],
) -> tuple[ExactGP, ...]:
    return tuple(
        ExactGP.fit(
            model.train_x,
            model.train_y,
            observation_variance=model.observation_variance,
            schema=model.schema,
            length_scale_mode="ard",
            calibration_scale=calibrator.variance_scale,
            nominal_probability=model.nominal_probability,
            calibration_source=(
                f"fixed-held-out-calibration-fit:n={calibrator.fit_sample_count}"
            ),
        )
        for model, calibrator in zip(uncalibrated, calibrators, strict=True)
    )


def run_campaign(
    *,
    output_dir: Path = DEFAULT_ARTIFACT_DIR,
    write_artifacts: bool = True,
) -> dict[str, object]:
    started = perf_counter()
    predeclared = load_predeclaration()
    source = predeclared["source"]
    campaign = predeclared["campaign"]
    model_policy = predeclared["model"]
    gates = predeclared["quality_gates"]
    if not all(isinstance(value, Mapping) for value in (source, campaign, model_policy, gates)):
        raise ValueError("predeclaration sections must be objects")

    source_path = MODERN / str(source["config_path"])
    source_config = load_l0_json(source_path)
    if canonical_hash(source_config) != source["config_hash"]:
        raise ValueError("accepted L0 source config hash mismatch")
    if int(source_config["batch_size"]) != int(source["required_batch_size"]):
        raise ValueError("accepted L0 sweep does not contain 8,192 points")
    truth_started = perf_counter()
    truth_artifact = evaluate_sweep_artifact(source_config, device="python")
    truth_seconds = perf_counter() - truth_started
    points = truth_artifact["points"]
    ranges = source_config["ranges"]
    if not isinstance(points, list) or not isinstance(ranges, Mapping):
        raise ValueError("unexpected accepted L0 sweep artifact")
    inputs = tuple(_feature_row(point, ranges) for point in points)
    outputs = tuple(_output_row(point) for point in points)
    bins = int(predeclared["split"]["spatial_group_bins_per_dimension"])  # type: ignore[index]
    groups = tuple(_group_key(row, bins) for row in inputs)
    dataset_payload = {
        "source_config_hash": source["config_hash"],
        "feature_names": predeclared["features"]["names"],  # type: ignore[index]
        "output_names": list(OUTPUT_NAMES),
        "rows": [
            {
                "index": index,
                "inputs": list(row),
                "outputs": list(outputs[index]),
                "spatial_group": groups[index],
            }
            for index, row in enumerate(inputs)
        ],
    }
    dataset_hash = canonical_hash(dataset_payload)
    partition = build_partition(inputs, predeclared)
    calibration_indices = tuple(int(value) for value in partition["calibration_indices"])  # type: ignore[arg-type]
    eligible_indices = tuple(int(value) for value in partition["eligible_indices"])  # type: ignore[arg-type]
    output_ranges = tuple(
        max(row[output] for row in outputs) - min(row[output] for row in outputs)
        for output in range(len(OUTPUT_NAMES))
    )

    initial = int(campaign["initial_training_rows"])
    increment = int(campaign["acquisition_batch_rows"])
    maximum = int(campaign["maximum_training_rows"])
    exact_limit = int(model_policy["maximum_exact_gp_training_rows"])
    if maximum > exact_limit:
        raise ValueError("campaign exceeds predeclared exact-GP row bound")
    active_indices = list(eligible_indices[:initial])
    nominal = float(model_policy["nominal_interval_probability"])
    score_policy = campaign["active_score"]
    if not isinstance(score_policy, Mapping):
        raise ValueError("active score policy must be an object")
    weights = AcquisitionWeights(
        predicted_improvement=float(score_policy["predicted_improvement_weight"]),
        feasibility=float(score_policy["feasibility_weight"]),
        discrepancy=float(score_policy["discrepancy_weight"]),
        uncertainty=float(score_policy["uncertainty_weight"]),
    )

    learning_curve: list[dict[str, object]] = []
    acquisition_rounds: list[dict[str, object]] = []
    previous_models: tuple[ExactGP, ...] | None = None
    previous_calibrators: tuple[VarianceCalibrator, ...] | None = None
    final_models: tuple[ExactGP, ...] | None = None
    final_calibrators: tuple[VarianceCalibrator, ...] | None = None

    while True:
        budget = len(active_indices)
        baseline_indices = eligible_indices[:budget]
        active_models = _fit_uncalibrated_models(
            active_indices, inputs, outputs, nominal
        )
        active_calibrators = _calibrators(
            active_models, calibration_indices, inputs, outputs, nominal
        )
        baseline_models = _fit_uncalibrated_models(
            baseline_indices, inputs, outputs, nominal
        )
        baseline_calibrators = _calibrators(
            baseline_models, calibration_indices, inputs, outputs, nominal
        )
        active_assessment = _evaluate_models(
            active_models,
            active_calibrators,
            partition,
            inputs,
            outputs,
            output_ranges,
            gates,
            nominal,
        )
        baseline_assessment = _evaluate_models(
            baseline_models,
            baseline_calibrators,
            partition,
            inputs,
            outputs,
            output_ranges,
            gates,
            nominal,
        )
        learning_curve.append(
            {
                "training_rows": budget,
                "active": active_assessment,
                "fixed_halton_like_baseline": baseline_assessment,
            }
        )
        final_models = active_models
        final_calibrators = active_calibrators
        if bool(active_assessment["all_predeclared_gates_passed"]) or budget >= maximum:
            break

        acquire_count = min(increment, maximum - budget)
        acquired = _acquire(
            count=acquire_count,
            selected_indices=active_indices,
            eligible_indices=eligible_indices,
            inputs=inputs,
            outputs=outputs,
            models=active_models,
            calibrators=active_calibrators,
            previous_models=previous_models,
            previous_calibrators=previous_calibrators,
            weights=weights,
            pending_length_scale=float(score_policy["pending_repulsion_length_scale"]),
            output_ranges=output_ranges,
        )
        acquisition_rounds.append(
            {
                "starting_training_rows": budget,
                "selected_indices": list(acquired),
                "selection_hash": canonical_hash(list(acquired)),
            }
        )
        active_indices.extend(acquired)
        previous_models = active_models
        previous_calibrators = active_calibrators

    if final_models is None or final_calibrators is None:
        raise RuntimeError("campaign produced no fitted model")
    calibrated_models = _fit_calibrated_final_models(final_models, final_calibrators)
    final_assessment = learning_curve[-1]["active"]
    baseline_assessment = learning_curve[-1]["fixed_halton_like_baseline"]
    final_status = bool(final_assessment["all_predeclared_gates_passed"])  # type: ignore[index]
    stop_reason = (
        "all-predeclared-gates-passed"
        if final_status
        else "maximum-budget-exhausted-with-gates-failed"
    )

    dataset_manifest: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-dataset-manifest",
        "schema_version": "1.0",
        "source_config_hash": source["config_hash"],
        "dataset_hash": dataset_hash,
        "row_count": len(inputs),
        "feature_names": predeclared["features"]["names"],  # type: ignore[index]
        "output_names": list(OUTPUT_NAMES),
        "output_ranges": dict(zip(OUTPUT_NAMES, output_ranges, strict=True)),
        "partition": partition,
        "truth_role": source["truth_role"],
    }
    dataset_manifest["manifest_hash"] = canonical_hash(dataset_manifest)

    benchmark: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-final-benchmark",
        "schema_version": "1.0",
        "predeclaration_hash": predeclared["predeclaration_hash"],
        "dataset_hash": dataset_hash,
        "partition_hash": partition["partition_hash"],
        "training_rows": len(active_indices),
        "active": final_assessment,
        "fixed_halton_like_baseline": baseline_assessment,
        "quality_gates": gates,
        "model_quality_passed": final_status,
        "claim": (
            "L0 software-emulation accuracy only; this is not physical or "
            "experimental predictive accuracy."
        ),
    }
    benchmark["benchmark_hash"] = canonical_hash(benchmark)

    campaign_result: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-active-learning-campaign",
        "schema_version": "1.0",
        "predeclaration_hash": predeclared["predeclaration_hash"],
        "source_config_hash": source["config_hash"],
        "dataset_hash": dataset_hash,
        "dataset_manifest_hash": dataset_manifest["manifest_hash"],
        "partition_hash": partition["partition_hash"],
        "benchmark_hash": benchmark["benchmark_hash"],
        "model_runtime": model_policy["runtime"],
        "maximum_exact_gp_training_rows": exact_limit,
        "initial_training_rows": initial,
        "final_training_rows": len(active_indices),
        "maximum_training_rows": maximum,
        "stop_reason": stop_reason,
        "all_predeclared_gates_passed": final_status,
        "selected_indices": active_indices,
        "selected_indices_hash": canonical_hash(active_indices),
        "selected_indices_unique": len(active_indices) == len(set(active_indices)),
        "acquisition_rounds": acquisition_rounds,
        "learning_curve": learning_curve,
        "final_model_hashes": {
            name: model.model_hash
            for name, model in zip(OUTPUT_NAMES, calibrated_models, strict=True)
        },
        "final_hyperparameters": {
            name: {
                **asdict(model.diagnostics),
                "length_scales": list(model.diagnostics.length_scales),
            }
            for name, model in zip(OUTPUT_NAMES, calibrated_models, strict=True)
        },
        "calibration": {
            name: asdict(calibrator)
            for name, calibrator in zip(OUTPUT_NAMES, final_calibrators, strict=True)
        },
        "wall_clock_policy": "diagnostic-only; excluded from campaign hash",
        "claim": (
            "The metrics quantify emulation of one hashed deterministic L0 "
            "software sweep, not accuracy of the underlying physics."
        ),
    }
    campaign_result["campaign_hash"] = canonical_hash(campaign_result)

    elapsed = perf_counter() - started
    runtime_diagnostics = {
        "document_type": "cft-revival-l0-surrogate-runtime-diagnostics",
        "schema_version": "1.0",
        "truth_generation_wall_seconds": truth_seconds,
        "campaign_wall_seconds": elapsed,
        "timing_role": "diagnostic-only-uncontrolled",
        "campaign_hash": campaign_result["campaign_hash"],
        "benchmark_hash": benchmark["benchmark_hash"],
        "model_hashes": campaign_result["final_model_hashes"],
    }

    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        _json_write(output_dir / "dataset_manifest.json", dataset_manifest)
        _json_write(output_dir / "benchmark.json", benchmark)
        _json_write(output_dir / "campaign.json", campaign_result)
        _json_write(output_dir / "runtime_diagnostics.json", runtime_diagnostics)
        for name, model in zip(OUTPUT_NAMES, calibrated_models, strict=True):
            model.save(output_dir / f"{name}.model.json")

    return {
        "dataset_manifest": dataset_manifest,
        "benchmark": benchmark,
        "campaign": campaign_result,
        "runtime_diagnostics": runtime_diagnostics,
        "models": calibrated_models,
    }


def main() -> int:
    result = run_campaign()
    campaign = result["campaign"]
    print(
        json.dumps(
            {
                "final_training_rows": campaign["final_training_rows"],
                "stop_reason": campaign["stop_reason"],
                "all_predeclared_gates_passed": campaign[
                    "all_predeclared_gates_passed"
                ],
                "campaign_hash": campaign["campaign_hash"],
                "benchmark_hash": campaign["benchmark_hash"],
                "model_hashes": campaign["final_model_hashes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
