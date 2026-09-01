"""Deterministic hash-identified grouped/spatial surrogate benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .gp import (
    MODEL_SCHEMA_VERSION,
    IndependentMultiOutputGP,
    SurrogateSchema,
)
from .identity import canonical_hash
from .normalization import SurrogateValidationError, finite_matrix
from .validation import (
    RegressionMetrics,
    Split,
    grouped_spatial_split,
    regression_metrics,
)


@dataclass(frozen=True, slots=True)
class HeldoutBenchmarkReport:
    split: Split
    output_names: tuple[str, ...]
    metrics: tuple[RegressionMetrics, ...]
    dataset_hash: str
    split_hash: str
    config_hash: str
    benchmark_hash: str
    nominal_probability: float
    coverage_target: float
    coverage_tolerance: float
    minimum_coverage_sample_count: int
    rmse_acceptance_threshold: float
    software_reproducibility_passed: bool
    model_quality_passed: bool
    assessment_limited: bool
    calibration_fit_sample_count: int = 0
    assessment_role: str = "held-out-assessment"
    uncertainty_status: str = "uncalibrated-held-out-assessment"
    interpretation: str = (
        "Errors measure interpolation of the declared deterministic data generator; "
        "they are not evidence of physical accuracy."
    )


def run_heldout_benchmark(
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[Sequence[float]],
    groups: Sequence[str],
    *,
    schema: SurrogateSchema | None = None,
    validation_fraction: float = 0.2,
    seed: int = 0,
    nominal_probability: float = 0.95,
    coverage_target: float | None = None,
    coverage_tolerance: float = 0.05,
    minimum_coverage_sample_count: int = 30,
    rmse_acceptance_threshold: float = 0.05,
    length_scale_mode: str = "ard",
    expected_hashes: Mapping[str, str] | None = None,
) -> HeldoutBenchmarkReport:
    x = finite_matrix(inputs, "inputs")
    y = finite_matrix(outputs, "outputs")
    if len(x) != len(y):
        raise SurrogateValidationError("input and output rows must match")
    if len(groups) != len(x):
        raise SurrogateValidationError("benchmark groups must match rows")
    output_names = (
        schema.output_names
        if schema is not None
        else tuple(f"y{index}" for index in range(len(y[0])))
    )
    target = nominal_probability if coverage_target is None else coverage_target
    dataset_payload = {
        "inputs": [list(row) for row in x],
        "outputs": [list(row) for row in y],
        "groups": list(groups),
        "schema": None if schema is None else schema.to_dict(),
    }
    config_payload = {
        "runtime_version": MODEL_SCHEMA_VERSION,
        "split_policy": "grouped-spatial-coordinate-closure-v2",
        "validation_fraction": validation_fraction,
        "seed": seed,
        "coordinate_tolerance": 0.0,
        "nominal_probability": nominal_probability,
        "coverage_target": target,
        "coverage_tolerance": coverage_tolerance,
        "minimum_coverage_sample_count": minimum_coverage_sample_count,
        "rmse_acceptance_threshold": rmse_acceptance_threshold,
        "quality_scale_policy": "full-hashed-dataset-output-range",
        "length_scale_mode": length_scale_mode,
        "calibration_fit_sample_count": 0,
        "assessment_role": "held-out-assessment",
    }
    dataset_hash = canonical_hash(dataset_payload)
    config_hash = canonical_hash(config_payload)
    split = grouped_spatial_split(
        x,
        groups,
        validation_fraction=validation_fraction,
        seed=seed,
        coordinate_tolerance=0.0,
    )
    model = IndependentMultiOutputGP.fit(
        tuple(x[index] for index in split.training_indices),
        tuple(y[index] for index in split.training_indices),
        schema=schema,
        length_scale_mode=length_scale_mode,
        nominal_probability=nominal_probability,
    )
    predictions = model.predict(
        tuple(x[index] for index in split.validation_indices)
    )
    metrics = tuple(
        regression_metrics(
            tuple(y[index][output] for index in split.validation_indices),
            tuple(row[output] for row in predictions),
            nominal_probability=nominal_probability,
            coverage_target=target,
            coverage_tolerance=coverage_tolerance,
            minimum_coverage_sample_count=minimum_coverage_sample_count,
            rmse_acceptance_threshold=rmse_acceptance_threshold,
            quality_scale=(
                max(row[output] for row in y)
                - min(row[output] for row in y)
            ),
            assessment_role="held-out-assessment",
        )
        for output in range(len(y[0]))
    )
    benchmark_hash = canonical_hash(
        {
            "dataset_hash": dataset_hash,
            "split_hash": split.split_hash,
            "config_hash": config_hash,
            "output_names": output_names,
            "metrics": [
                {
                    "rmse": metric.rmse,
                    "output_range": metric.output_range,
                    "range_normalized_rmse": metric.range_normalized_rmse,
                    "rmse_accepted": metric.rmse_accepted,
                    "mae": metric.mae,
                    "worst_case_absolute_error": (
                        metric.worst_case_absolute_error
                    ),
                    "interval_coverage": metric.interval_coverage,
                    "coverage_accepted": metric.coverage_accepted,
                    "assessment_limited": metric.assessment_limited,
                    "model_quality_passed": metric.model_quality_passed,
                    "sample_count": metric.sample_count,
                }
                for metric in metrics
            ],
        }
    )
    actual_hashes = {
        "dataset_hash": dataset_hash,
        "split_hash": split.split_hash,
        "config_hash": config_hash,
        "benchmark_hash": benchmark_hash,
    }
    if expected_hashes is not None and set(expected_hashes) != set(actual_hashes):
        raise SurrogateValidationError(
            "expected benchmark hashes must name all four identities"
        )
    software_reproducibility_passed = (
        expected_hashes is not None
        and dict(expected_hashes) == actual_hashes
    )
    return HeldoutBenchmarkReport(
        split=split,
        output_names=output_names,
        metrics=metrics,
        dataset_hash=dataset_hash,
        split_hash=split.split_hash,
        config_hash=config_hash,
        benchmark_hash=benchmark_hash,
        nominal_probability=nominal_probability,
        coverage_target=target,
        coverage_tolerance=coverage_tolerance,
        minimum_coverage_sample_count=minimum_coverage_sample_count,
        rmse_acceptance_threshold=rmse_acceptance_threshold,
        software_reproducibility_passed=software_reproducibility_passed,
        model_quality_passed=all(
            metric.model_quality_passed for metric in metrics
        ),
        assessment_limited=any(metric.assessment_limited for metric in metrics),
    )
