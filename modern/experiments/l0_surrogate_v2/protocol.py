"""Preregistered L0 surrogate experiment v2.

Only input-derived partitions are created by ``preregister``.  ``execute`` is
single-shot and reveals final-assessment labels only after training selection,
model, and per-stratum conformal calibration artifacts have been frozen.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from math import ceil, floor, fsum, hypot, nextafter, sqrt, inf
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
from cft_revival.surrogates import ExactGP, Prediction, SurrogateSchema
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads

MODERN = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
PREDECLARATION = ROOT / "predeclaration.json"
PARTITIONS = ROOT / "partitions.json"
RESULTS = ROOT / "results"
OUTPUT_NAMES = ("axial_thrust_n", "specific_impulse_s")
OUTPUT_UNITS = ("N", "s")
SOURCE = FidelitySource("L0-software-emulator", rank=0, cost=1.0, is_highest=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json_object(path: Path) -> dict[str, object]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_predeclaration() -> dict[str, object]:
    value = load_json_object(PREDECLARATION)
    declared = value.get("predeclaration_hash")
    payload = {key: item for key, item in value.items() if key != "predeclaration_hash"}
    if declared != canonical_hash(payload):
        raise ValueError("predeclaration hash mismatch")
    return value


def _feature_row(
    point: Mapping[str, object],
    ranges: Mapping[str, object],
) -> tuple[float, ...]:
    raw = point["input"]
    if not isinstance(raw, Mapping):
        raise ValueError("L0 input row is malformed")
    fractions = raw["charge_state_number_fractions"]
    if not isinstance(fractions, Mapping):
        raise ValueError("L0 charge-state row is malformed")
    ionized = 1.0 - float(fractions["xe_neutral"])
    raw_values = (
        float(raw["discharge_voltage_v"]),
        float(raw["propellant_mass_flow_kg_per_s"]),
        ionized,
        float(fractions["xe_double_plus"]) / ionized,
        float(raw["axial_momentum_fraction_of_ion_momentum"]),
    )
    names = (
        "discharge_voltage_v",
        "propellant_mass_flow_kg_per_s",
        "ionized_number_fraction",
        "xe_double_plus_fraction_of_ions",
        "axial_momentum_fraction_of_ion_momentum",
    )
    result = []
    for raw_value, name in zip(raw_values, names, strict=True):
        bounds = ranges[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"source range {name} is malformed")
        low, high = (float(item) for item in bounds)
        coordinate = (raw_value - low) / (high - low)
        if not -1e-12 <= coordinate <= 1.0 + 1e-12:
            raise ValueError(f"source coordinate {name} is out of bounds")
        result.append(min(1.0, max(0.0, coordinate)))
    return tuple(result)


def _output_row(point: Mapping[str, object]) -> tuple[float, float]:
    result = point["result"]
    if not isinstance(result, Mapping):
        raise ValueError("L0 result row is malformed")
    return float(result["axial_thrust_n"]), float(result["specific_impulse_s"])


def load_l0_rows(
    declaration: Mapping[str, object],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, float], ...]]:
    source = declaration["source"]
    if not isinstance(source, Mapping):
        raise ValueError("source declaration is malformed")
    config = load_l0_json(MODERN / str(source["config_path"]))
    if canonical_hash(config) != source["config_hash"]:
        raise ValueError("source config hash mismatch")
    if int(config["batch_size"]) != int(source["required_rows"]):
        raise ValueError("source row count mismatch")
    artifact = evaluate_sweep_artifact(config, device="python")
    points = artifact["points"]
    ranges = config["ranges"]
    if not isinstance(points, list) or not isinstance(ranges, Mapping):
        raise ValueError("accepted L0 artifact is malformed")
    return (
        tuple(_feature_row(point, ranges) for point in points),
        tuple(_output_row(point) for point in points),
    )


def group_key(
    row: Sequence[float],
    grouping: Mapping[str, object],
) -> str:
    dimensions = tuple(int(value) for value in grouping["dimensions"])  # type: ignore[arg-type]
    bins = int(grouping["bins_per_dimension"])
    values = tuple(min(bins - 1, int(float(row[index]) * bins)) for index in dimensions)
    return ":".join(f"{dimension}={value}" for dimension, value in zip(dimensions, values))


def _group_bins(group: str) -> dict[int, int]:
    return {
        int(field.split("=")[0]): int(field.split("=")[1])
        for field in group.split(":")
    }


def group_stratum(
    group: str,
    partition_policy: Mapping[str, object],
) -> str | None:
    values = _group_bins(group)
    strata = partition_policy["strata"]
    priority = partition_policy["stratum_priority"]
    if not isinstance(strata, Mapping) or not isinstance(priority, list):
        raise ValueError("stratum policy is malformed")
    for name in priority:
        rule = strata[name]
        if not isinstance(rule, Mapping):
            raise ValueError("stratum rule is malformed")
        kind = rule["rule"]
        if kind == "all_between":
            matches = all(
                int(rule["minimum_bin"]) <= values[int(dimension)] <= int(rule["maximum_bin"])
                for dimension in rule["dimensions"]  # type: ignore[union-attr]
            )
        elif kind == "any_in":
            accepted = {int(value) for value in rule["bins"]}  # type: ignore[union-attr]
            matches = any(
                values[int(dimension)] in accepted
                for dimension in rule["dimensions"]  # type: ignore[union-attr]
            )
        elif kind == "all_dimension_in":
            matches = all(
                values[int(condition["dimension"])]
                in {int(value) for value in condition["bins"]}
                for condition in rule["conditions"]  # type: ignore[union-attr]
            )
        else:
            raise ValueError(f"unknown stratum rule {kind!r}")
        if matches:
            return str(name)
    return None


def _ordered(groups: Sequence[str], seed: int, role: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                sha256(f"{seed}:{role}:{group}".encode()).hexdigest(),
                group,
            ),
        )
    )


def _take_groups(
    groups: Sequence[str],
    rows_by_group: Mapping[str, tuple[int, ...]],
    *,
    minimum_rows: int,
    minimum_groups: int,
    seed: int,
    role: str,
) -> tuple[str, ...]:
    selected: list[str] = []
    row_count = 0
    for group in _ordered(groups, seed, role):
        selected.append(group)
        row_count += len(rows_by_group[group])
        if len(selected) >= minimum_groups and row_count >= minimum_rows:
            return tuple(selected)
    raise ValueError(f"insufficient independent groups for {role}")


def build_partitions(
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    policy = declaration["partition"]
    if not isinstance(policy, Mapping):
        raise ValueError("partition policy is malformed")
    grouping = policy["grouping"]
    if not isinstance(grouping, Mapping):
        raise ValueError("grouping policy is malformed")
    group_lists: dict[str, list[int]] = {}
    for index, row in enumerate(inputs):
        group_lists.setdefault(group_key(row, grouping), []).append(index)
    rows_by_group = {key: tuple(value) for key, value in group_lists.items()}
    group_strata = {
        group: group_stratum(group, policy)
        for group in rows_by_group
    }

    replicates = []
    for raw_seed in policy["replicate_seeds"]:  # type: ignore[union-attr]
        seed = int(raw_seed)
        reserved: set[str] = set()
        calibration: dict[str, list[object]] = {}
        assessment: dict[str, list[object]] = {}
        for stratum in ("interpolation", "boundary", "ood"):
            available = tuple(
                group
                for group, assigned in group_strata.items()
                if assigned == stratum and group not in reserved
            )
            calibration_groups = _take_groups(
                available,
                rows_by_group,
                minimum_rows=int(policy["calibration_minimum_rows_per_stratum"]),
                minimum_groups=int(policy["calibration_minimum_groups_per_stratum"]),
                seed=seed,
                role=f"{stratum}:calibration",
            )
            reserved.update(calibration_groups)
            available = tuple(group for group in available if group not in reserved)
            assessment_groups = _take_groups(
                available,
                rows_by_group,
                minimum_rows=int(policy["assessment_minimum_rows_per_stratum"]),
                minimum_groups=int(policy["assessment_minimum_groups_per_stratum"]),
                seed=seed,
                role=f"{stratum}:assessment",
            )
            reserved.update(assessment_groups)
            calibration[stratum] = [
                list(calibration_groups),
                [
                    index
                    for group in calibration_groups
                    for index in rows_by_group[group]
                ],
            ]
            assessment[stratum] = [
                list(assessment_groups),
                [
                    index
                    for group in assessment_groups
                    for index in rows_by_group[group]
                ],
            ]
        candidate_indices = [
            index
            for index, row in enumerate(inputs)
            if group_key(row, grouping) not in reserved
        ]
        replicate: dict[str, object] = {
            "replicate_id": f"group-split-{seed}",
            "seed": seed,
            "calibration": {
                name: {"groups": value[0], "indices": value[1]}
                for name, value in calibration.items()
            },
            "assessment": {
                name: {"groups": value[0], "indices": value[1]}
                for name, value in assessment.items()
            },
            "candidate_indices": candidate_indices,
        }
        replicate["replicate_partition_hash"] = canonical_hash(replicate)
        replicates.append(replicate)

    manifest: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v2-input-partitions",
        "schema_version": "2.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "source_config_hash": declaration["source"]["config_hash"],  # type: ignore[index]
        "input_row_count": len(inputs),
        "input_dataset_hash": canonical_hash(
            {"inputs": [list(row) for row in inputs]}
        ),
        "grouping": grouping,
        "replicates": replicates,
        "label_policy": "no output, calibration-label, or assessment-label fields",
    }
    manifest["partitions_hash"] = canonical_hash(manifest)
    return manifest


def preregister() -> dict[str, object]:
    declaration = load_predeclaration()
    inputs, _ = load_l0_rows(declaration)
    manifest = build_partitions(inputs, declaration)
    write_json(PARTITIONS, manifest)
    return manifest


def scalar_schema(output_index: int) -> SurrogateSchema:
    declaration = load_predeclaration()
    return SurrogateSchema(
        tuple(declaration["features"]["names"]),  # type: ignore[index]
        (OUTPUT_NAMES[output_index],),
        ("1",) * 5,
        (OUTPUT_UNITS[output_index],),
    )


def fit_models(
    selected: Sequence[int],
    inputs: Sequence[Sequence[float]],
    observed: Mapping[int, Sequence[float]],
) -> tuple[ExactGP, ExactGP]:
    train_x = tuple(inputs[index] for index in selected)
    return tuple(  # type: ignore[return-value]
        ExactGP.fit(
            train_x,
            tuple(float(observed[index][output]) for index in selected),
            schema=scalar_schema(output),
            length_scale_mode="ard",
            nominal_probability=0.9,
        )
        for output in range(2)
    )


def predict_rows(
    models: Sequence[ExactGP],
    indices: Sequence[int],
    inputs: Sequence[Sequence[float]],
) -> dict[int, tuple[Prediction, Prediction]]:
    points = tuple(inputs[index] for index in indices)
    columns = tuple(model.predict(points) for model in models)
    return {
        index: (columns[0][row], columns[1][row])
        for row, index in enumerate(indices)
    }


class TrainingOracle:
    """Reveals labels only for the declared candidate pool."""

    def __init__(
        self,
        outputs: Sequence[Sequence[float]],
        allowed_indices: Sequence[int],
    ) -> None:
        self._outputs = outputs
        self._allowed = frozenset(allowed_indices)

    def observe(self, index: int) -> tuple[float, float]:
        if index not in self._allowed:
            raise ValueError("training oracle cannot reveal held-out labels")
        row = self._outputs[index]
        return float(row[0]), float(row[1])


class _PosteriorAdapter:
    def __init__(
        self,
        current: Mapping[int, tuple[Prediction, Prediction]],
        previous: Mapping[int, tuple[Prediction, Prediction]] | None,
        index_by_input: Mapping[tuple[float, ...], int],
    ) -> None:
        self.current = current
        self.previous = previous
        self.index_by_input = index_by_input

    def predict(
        self,
        design: Sequence[float],
        source: FidelitySource,
    ) -> PosteriorPrediction:
        if source != SOURCE:
            raise ValueError("unexpected source")
        index = self.index_by_input[tuple(float(value) for value in design)]
        current = self.current[index]
        old = None if self.previous is None else self.previous[index]
        return PosteriorPrediction(
            objective_means=tuple(item.mean for item in current),
            epistemic_standard_deviations=tuple(
                item.standard_deviation for item in current
            ),
            aleatoric_standard_deviations=(0.0, 0.0),
            discrepancy_means=(
                (0.0, 0.0)
                if old is None
                else tuple(
                    item.mean - prior.mean
                    for item, prior in zip(current, old, strict=True)
                )
            ),
        )


def select_active_indices(
    inputs: Sequence[Sequence[float]],
    candidate_indices: Sequence[int],
    oracle: TrainingOracle,
    declaration: Mapping[str, object],
) -> tuple[tuple[int, ...], dict[int, tuple[float, float]], list[dict[str, object]]]:
    campaign = declaration["campaign"]
    if not isinstance(campaign, Mapping):
        raise ValueError("campaign declaration is malformed")
    score_policy = campaign["active_score"]
    if not isinstance(score_policy, Mapping):
        raise ValueError("active-score declaration is malformed")
    initial = int(campaign["initial_rows"])
    final = int(campaign["final_rows"])
    batch = int(campaign["acquisition_batch_rows"])
    selected = list(candidate_indices[:initial])
    observed = {index: oracle.observe(index) for index in selected}
    rounds: list[dict[str, object]] = []
    previous_models: tuple[ExactGP, ExactGP] | None = None

    while len(selected) < final:
        models = fit_models(selected, inputs, observed)
        remaining = tuple(index for index in candidate_indices if index not in observed)
        current = predict_rows(models, remaining, inputs)
        previous = (
            None
            if previous_models is None
            else predict_rows(previous_models, remaining, inputs)
        )
        adapter = _PosteriorAdapter(
            current,
            previous,
            {tuple(inputs[index]): index for index in remaining},
        )
        detector = models[0].ood_detector()
        scales = tuple(
            max(observed[index][output] for index in selected)
            - min(observed[index][output] for index in selected)
            for output in range(2)
        )
        scales = tuple(max(scale, 1e-300) for scale in scales)
        weights = AcquisitionWeights(
            predicted_improvement=0.0,
            feasibility=0.0,
            discrepancy=float(score_policy["posterior_change_weight"]),
            uncertainty=float(score_policy["uncertainty_weight"]),
        )
        incumbent = tuple(
            max(observed[index][output] for index in selected)
            for output in range(2)
        )
        acquired: list[int] = []
        pending: list[tuple[Sequence[float], PosteriorPrediction]] = []
        available = set(remaining)
        for _ in range(min(batch, final - len(selected))):
            scored: list[tuple[float, int]] = []
            for index in available:
                accepted = score_candidate(
                    inputs[index],
                    SOURCE,
                    adapter,
                    incumbent,
                    pending=pending,
                    weights=weights,
                    pending_length_scale=float(
                        score_policy["pending_repulsion_length_scale"]
                    ),
                    uncertainty_scales=scales,
                )
                report = detector.report(inputs[index])
                ood_signal = report.nearest_training_distance / (
                    report.nearest_training_distance + report.threshold
                )
                total = (
                    accepted.cost_normalized_score
                    + float(score_policy["input_ood_weight"]) * ood_signal
                )
                scored.append((total, index))
            _, chosen = max(scored, key=lambda item: (item[0], -item[1]))
            acquired.append(chosen)
            available.remove(chosen)
            pending.append((inputs[chosen], adapter.predict(inputs[chosen], SOURCE)))
        for index in acquired:
            observed[index] = oracle.observe(index)
        rounds.append(
            {
                "starting_rows": len(selected),
                "selected_indices": acquired,
                "selection_hash": canonical_hash(acquired),
            }
        )
        selected.extend(acquired)
        previous_models = models
    return tuple(selected), observed, rounds


def baseline_indices(
    candidate_indices: Sequence[int],
    declaration: Mapping[str, object],
) -> tuple[int, ...]:
    count = int(declaration["campaign"]["final_rows"])  # type: ignore[index]
    return tuple(candidate_indices[:count])


def _order_statistic(values: Sequence[float], rank: int, direction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not 1 <= rank <= len(ordered):
        raise ValueError("finite-sample conformal rank is out of bounds")
    return nextafter(ordered[rank - 1], direction)


def fit_conformal(
    models: Sequence[ExactGP],
    calibration: Mapping[str, object],
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[Sequence[float]],
    nominal: float,
) -> dict[str, object]:
    alpha = 1.0 - nominal
    result: dict[str, object] = {}
    for stratum in ("interpolation", "boundary", "ood"):
        split = calibration[stratum]
        if not isinstance(split, Mapping):
            raise ValueError("calibration split is malformed")
        indices = tuple(int(value) for value in split["indices"])  # type: ignore[arg-type]
        predictions = predict_rows(models, indices, inputs)
        thrust_residuals = [
            outputs[index][0] - predictions[index][0].mean for index in indices
        ]
        lower_rank = max(1, floor((len(indices) + 1) * alpha / 2.0))
        upper_rank = min(len(indices), ceil((len(indices) + 1) * (1.0 - alpha / 2.0)))
        lower = _order_statistic(thrust_residuals, lower_rank, -inf)
        upper = _order_statistic(thrust_residuals, upper_rank, inf)

        positive_sd = [
            predictions[index][1].standard_deviation
            for index in indices
            if predictions[index][1].standard_deviation > 0.0
        ]
        if not positive_sd:
            raise ValueError("Isp conformal normalization requires positive GP variance")
        scale_floor = min(positive_sd) * 1e-12
        normalized = [
            abs(outputs[index][1] - predictions[index][1].mean)
            / max(predictions[index][1].standard_deviation, scale_floor)
            for index in indices
        ]
        symmetric_rank = min(len(indices), ceil((len(indices) + 1) * nominal))
        symmetric = _order_statistic(normalized, symmetric_rank, inf)
        record: dict[str, object] = {
            "stratum": stratum,
            "groups": split["groups"],
            "indices": list(indices),
            "raw": {
                "axial_thrust_n": {
                    "labels": [outputs[index][0] for index in indices],
                    "predicted_means": [predictions[index][0].mean for index in indices],
                    "signed_residuals": thrust_residuals,
                },
                "specific_impulse_s": {
                    "labels": [outputs[index][1] for index in indices],
                    "predicted_means": [predictions[index][1].mean for index in indices],
                    "predictive_standard_deviations": [
                        predictions[index][1].standard_deviation for index in indices
                    ],
                    "absolute_normalized_residuals": normalized,
                },
            },
            "interval_parameters": {
                "axial_thrust_n": {
                    "method": "asymmetric-signed-residual",
                    "lower_rank": lower_rank,
                    "upper_rank": upper_rank,
                    "lower_residual": lower,
                    "upper_residual": upper,
                },
                "specific_impulse_s": {
                    "method": "symmetric-absolute-residual-normalized-by-raw-epistemic-sd",
                    "rank": symmetric_rank,
                    "quantile": symmetric,
                    "scale_floor": scale_floor,
                },
            },
        }
        record["stratum_calibration_hash"] = canonical_hash(record)
        result[stratum] = record
    artifact: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v2-conformal-calibration",
        "schema_version": "2.0",
        "nominal_probability": nominal,
        "strata": result,
    }
    artifact["calibration_hash"] = canonical_hash(artifact)
    return artifact


class SingleUseAssessmentLoader:
    def __init__(
        self,
        outputs: Sequence[Sequence[float]],
        assessment: Mapping[str, object],
        expected_frozen_hash: str,
    ) -> None:
        self._outputs = outputs
        self._assessment = assessment
        self._expected = expected_frozen_hash
        self._used = False

    def load(self, frozen_hash: str) -> dict[str, tuple[tuple[int, tuple[float, float]], ...]]:
        if self._used:
            raise RuntimeError("final assessment labels may be loaded only once")
        if frozen_hash != self._expected:
            raise RuntimeError("selection/model/calibration artifacts are not frozen")
        self._used = True
        result = {}
        for stratum in ("interpolation", "boundary", "ood"):
            split = self._assessment[stratum]
            if not isinstance(split, Mapping):
                raise ValueError("assessment split is malformed")
            result[stratum] = tuple(
                (int(index), tuple(self._outputs[int(index)]))
                for index in split["indices"]  # type: ignore[union-attr]
            )
        return result


def _interval(
    output: int,
    prediction: Prediction,
    parameters: Mapping[str, object],
) -> tuple[float, float]:
    if output == 0:
        return (
            prediction.mean + float(parameters["lower_residual"]),
            prediction.mean + float(parameters["upper_residual"]),
        )
    scale = max(prediction.standard_deviation, float(parameters["scale_floor"]))
    radius = float(parameters["quantile"]) * scale
    return nextafter(prediction.mean - radius, -inf), nextafter(prediction.mean + radius, inf)


def _metrics(
    rows: Sequence[tuple[float, Prediction, tuple[float, float]]],
    scale: float,
    gates: Mapping[str, object],
) -> dict[str, object]:
    errors = [abs(truth - prediction.mean) for truth, prediction, _ in rows]
    rmse = hypot(*errors) / sqrt(len(errors))
    coverage = sum(lower <= truth <= upper for truth, _, (lower, upper) in rows) / len(rows)
    normalized_rmse = rmse / scale
    normalized_worst = max(errors) / scale
    record: dict[str, object] = {
        "sample_count": len(rows),
        "rmse": rmse,
        "range_normalized_rmse": normalized_rmse,
        "mae": fsum(errors) / len(errors),
        "worst_case_absolute_error": max(errors),
        "worst_case_range_normalized_absolute_error": normalized_worst,
        "interval_coverage": coverage,
        "rmse_passed": normalized_rmse <= float(gates["range_normalized_rmse_maximum"]),
        "worst_error_passed": normalized_worst
        <= float(gates["worst_case_range_normalized_absolute_error_maximum"]),
        "coverage_passed": float(gates["coverage_minimum"])
        <= coverage
        <= float(gates["coverage_maximum"]),
    }
    record["all_gates_passed"] = all(
        bool(record[name])
        for name in ("rmse_passed", "worst_error_passed", "coverage_passed")
    )
    return record


def assess(
    models: Sequence[ExactGP],
    calibration: Mapping[str, object],
    labels: Mapping[str, Sequence[tuple[int, tuple[float, float]]]],
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    gates = declaration["quality_gates"]
    if not isinstance(gates, Mapping):
        raise ValueError("quality gates are malformed")
    scales = gates["quality_scales"]
    if not isinstance(scales, Mapping):
        raise ValueError("quality scales are malformed")
    calibration_strata = calibration["strata"]
    if not isinstance(calibration_strata, Mapping):
        raise ValueError("calibration artifact is malformed")
    raw: dict[str, object] = {}
    metrics: dict[str, object] = {}
    overall_rows: list[list[tuple[float, Prediction, tuple[float, float]]]] = [[], []]
    for stratum in ("interpolation", "boundary", "ood"):
        rows = labels[stratum]
        indices = tuple(index for index, _ in rows)
        predictions = predict_rows(models, indices, inputs)
        calibration_record = calibration_strata[stratum]
        parameters_by_output = calibration_record["interval_parameters"]  # type: ignore[index]
        raw_rows = []
        metric_rows: list[list[tuple[float, Prediction, tuple[float, float]]]] = [[], []]
        for index, truth in rows:
            row_record: dict[str, object] = {"index": index}
            for output, name in enumerate(OUTPUT_NAMES):
                prediction = predictions[index][output]
                bounds = _interval(
                    output,
                    prediction,
                    parameters_by_output[name],
                )
                item = (truth[output], prediction, bounds)
                metric_rows[output].append(item)
                overall_rows[output].append(item)
                row_record[name] = {
                    "truth": truth[output],
                    "mean": prediction.mean,
                    "raw_epistemic_variance": prediction.variance,
                    "interval": list(bounds),
                }
            raw_rows.append(row_record)
        raw[stratum] = raw_rows
        metrics[stratum] = {
            name: _metrics(
                metric_rows[output],
                float(scales[name]),
                gates,
            )
            for output, name in enumerate(OUTPUT_NAMES)
        }
    metrics["overall"] = {
        name: _metrics(overall_rows[output], float(scales[name]), gates)
        for output, name in enumerate(OUTPUT_NAMES)
    }
    metrics["all_scopes_outputs_passed"] = all(
        bool(metrics[scope][name]["all_gates_passed"])  # type: ignore[index]
        for scope in ("interpolation", "boundary", "ood", "overall")
        for name in OUTPUT_NAMES
    )
    return raw, metrics


def _save_models(path: Path, models: Sequence[ExactGP]) -> dict[str, str]:
    hashes = {}
    for name, model in zip(OUTPUT_NAMES, models, strict=True):
        model.save(path / f"{name}.model.json")
        hashes[name] = model.model_hash
    return hashes


def execute(preregistration_commit_sha: str) -> dict[str, object]:
    if not isinstance(preregistration_commit_sha, str) or len(preregistration_commit_sha) != 40:
        raise ValueError("a full preregistration commit SHA is required")
    declaration = load_predeclaration()
    partitions = load_json_object(PARTITIONS)
    if partitions["predeclaration_hash"] != declaration["predeclaration_hash"]:
        raise ValueError("partition/predeclaration identity mismatch")
    lock = RESULTS / "execution-lock.json"
    if lock.exists():
        raise RuntimeError("v2 execution is immutable and has already started")
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(
        lock,
        {
            "document_type": "cft-revival-l0-surrogate-v2-execution-lock",
            "preregistration_commit_sha": preregistration_commit_sha,
            "predeclaration_hash": declaration["predeclaration_hash"],
            "partitions_hash": partitions["partitions_hash"],
            "single_execution": True,
        },
    )
    started = perf_counter()
    inputs, outputs = load_l0_rows(declaration)
    replicate_results = []
    for replicate in partitions["replicates"]:  # type: ignore[union-attr]
        if not isinstance(replicate, Mapping):
            raise ValueError("replicate partition is malformed")
        replicate_id = str(replicate["replicate_id"])
        path = RESULTS / replicate_id
        candidate_indices = tuple(int(value) for value in replicate["candidate_indices"])  # type: ignore[arg-type]
        oracle = TrainingOracle(outputs, candidate_indices)
        active_indices, active_observed, rounds = select_active_indices(
            inputs, candidate_indices, oracle, declaration
        )
        fixed_indices = baseline_indices(candidate_indices, declaration)
        fixed_observed = {index: oracle.observe(index) for index in fixed_indices}
        active_models = fit_models(active_indices, inputs, active_observed)
        fixed_models = fit_models(fixed_indices, inputs, fixed_observed)

        selection_artifacts = {}
        model_hashes = {}
        calibrations = {}
        for campaign_name, indices, observed, models in (
            ("active", active_indices, active_observed, active_models),
            ("fixed-baseline", fixed_indices, fixed_observed, fixed_models),
        ):
            selection: dict[str, object] = {
                "document_type": "cft-revival-l0-surrogate-v2-selection",
                "replicate_id": replicate_id,
                "campaign": campaign_name,
                "selected_indices": list(indices),
                "observations": [
                    {"index": index, "outputs": list(observed[index])}
                    for index in indices
                ],
                "acquisition_rounds": rounds if campaign_name == "active" else [],
                "final_rows": len(indices),
            }
            selection["selection_hash"] = canonical_hash(selection)
            write_json(path / f"{campaign_name}.selection.json", selection)
            selection_artifacts[campaign_name] = selection
            model_hashes[campaign_name] = _save_models(path / campaign_name, models)
            calibration = fit_conformal(
                models,
                replicate["calibration"],  # type: ignore[arg-type]
                inputs,
                outputs,
                float(declaration["intervals"]["nominal_probability"]),  # type: ignore[index]
            )
            calibration["replicate_id"] = replicate_id
            calibration["campaign"] = campaign_name
            calibration["calibration_hash"] = canonical_hash(
                {
                    key: value
                    for key, value in calibration.items()
                    if key != "calibration_hash"
                }
            )
            write_json(path / f"{campaign_name}.calibration.json", calibration)
            calibrations[campaign_name] = calibration

        frozen: dict[str, object] = {
            "replicate_partition_hash": replicate["replicate_partition_hash"],
            "selection_hashes": {
                name: artifact["selection_hash"]
                for name, artifact in selection_artifacts.items()
            },
            "model_hashes": model_hashes,
            "calibration_hashes": {
                name: artifact["calibration_hash"]
                for name, artifact in calibrations.items()
            },
        }
        frozen_hash = canonical_hash(frozen)
        write_json(path / "frozen-before-assessment.json", {**frozen, "frozen_hash": frozen_hash})
        loader = SingleUseAssessmentLoader(
            outputs,
            replicate["assessment"],  # type: ignore[arg-type]
            frozen_hash,
        )
        labels = loader.load(frozen_hash)
        campaign_metrics = {}
        for campaign_name, models in (
            ("active", active_models),
            ("fixed-baseline", fixed_models),
        ):
            raw, metrics = assess(
                models,
                calibrations[campaign_name],
                labels,
                inputs,
                declaration,
            )
            assessment_artifact: dict[str, object] = {
                "document_type": "cft-revival-l0-surrogate-v2-final-assessment",
                "replicate_id": replicate_id,
                "campaign": campaign_name,
                "frozen_hash": frozen_hash,
                "raw": raw,
                "metrics": metrics,
            }
            assessment_artifact["assessment_hash"] = canonical_hash(assessment_artifact)
            write_json(path / f"{campaign_name}.assessment.json", assessment_artifact)
            campaign_metrics[campaign_name] = metrics
        result: dict[str, object] = {
            "replicate_id": replicate_id,
            "active": campaign_metrics["active"],
            "fixed-baseline": campaign_metrics["fixed-baseline"],
            "active_passed": campaign_metrics["active"]["all_scopes_outputs_passed"],  # type: ignore[index]
            "fixed_baseline_passed": campaign_metrics["fixed-baseline"]["all_scopes_outputs_passed"],  # type: ignore[index]
            "frozen_hash": frozen_hash,
        }
        result["replicate_result_hash"] = canonical_hash(result)
        replicate_results.append(result)

    accepted = all(bool(item["active_passed"]) for item in replicate_results)
    manifest: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v2-run-manifest",
        "schema_version": "2.0",
        "preregistration_commit_sha": preregistration_commit_sha,
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partitions["partitions_hash"],
        "source_config_hash": declaration["source"]["config_hash"],  # type: ignore[index]
        "replicates": replicate_results,
        "all_active_replicates_passed": accepted,
        "status": "accepted" if accepted else "failed-predeclared-gates",
        "claim": "deterministic L0 software-emulation accuracy only; no physical claim",
    }
    manifest["run_manifest_hash"] = canonical_hash(manifest)
    write_json(RESULTS / "run-manifest.json", manifest)
    write_json(
        RESULTS / "runtime-diagnostics.json",
        {
            "wall_seconds": perf_counter() - started,
            "role": "diagnostic-only-excluded-from-scientific-hashes",
            "run_manifest_hash": manifest["run_manifest_hash"],
        },
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preregister")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--preregistration-commit", required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        manifest = preregister()
        print(manifest["partitions_hash"])
    else:
        manifest = execute(args.preregistration_commit)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
