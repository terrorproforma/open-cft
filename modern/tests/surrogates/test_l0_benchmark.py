"""Deterministic L0 interpolation benchmark; L0 is not physical truth."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

from cft_revival.physics.reference import evaluate_batch
from cft_revival.physics.workflows import sweep_points_from_config
from cft_revival.surrogates import SurrogateSchema, run_heldout_benchmark
from cft_revival.surrogates.identity import canonical_hash

MODERN = Path(__file__).parents[2]


def test_grouped_spatial_l0_benchmark_reports_error_and_coverage() -> None:
    config = json.loads(
        (MODERN / "config" / "l0-deterministic-sweep.json").read_text(encoding="utf-8")
    )
    config["batch_size"] = 48
    config["seed"] = 31
    assert canonical_hash(config) == (
        "194b21ec5d40e87a6ac9a934025bd0dcc848c29d367aad4fac920c9d3677def2"
    )
    points, _ = sweep_points_from_config(config)
    results = evaluate_batch(points)
    inputs = tuple(
        (
            point.discharge_voltage_v,
            point.propellant_mass_flow.kg_per_s,
            point.charge_state_fractions.ionized_fraction,
            point.charge_state_fractions.xe_double_plus,
            point.beam_divergence_factors.beam_current_fraction_of_anode_current,
            point.beam_divergence_factors.axial_momentum_fraction_of_ion_momentum,
            point.power_boundaries.cathode_input_power_w,
            point.power_boundaries.ppu_input_power_w,
        )
        for point in points
    )
    outputs = tuple(
        (result.axial_thrust_n, result.specific_impulse_s) for result in results
    )
    schema = SurrogateSchema(
        (
            "discharge_voltage_v",
            "mass_flow_kg_per_s",
            "ionized_fraction",
            "xe_double_plus_fraction",
            "beam_current_fraction",
            "axial_momentum_fraction",
            "cathode_power_w",
            "ppu_input_power_w",
        ),
        ("axial_thrust_n", "specific_impulse_s"),
    )
    report = run_heldout_benchmark(
        inputs,
        outputs,
        tuple(f"l0-design-{index}" for index in range(len(points))),
        schema=schema,
        validation_fraction=0.25,
        seed=9,
        nominal_probability=0.9,
        coverage_target=0.9,
        coverage_tolerance=0.05,
        minimum_coverage_sample_count=30,
        rmse_acceptance_threshold=0.05,
        length_scale_mode="ard",
        expected_hashes={
            "dataset_hash": (
                "2647de548f3fe3ca3cc070eeca9fac187b877904bc729061ccc42c6c5ab77006"
            ),
            "split_hash": (
                "f44fdd4b2498ebc73157c91cc9bbbc4ed33e77fc8753825217b92f1ba43c4360"
            ),
            "config_hash": (
                "100834899102d9d6a19c410f82096e72b70c36612db5fe718f9ad70185e6fc7f"
            ),
            "benchmark_hash": (
                "ccf730a32bbb0bd82399e8728e36ffe24b9e2d8a051c2040703713d436c8a566"
            ),
        },
    )

    assert report.output_names == schema.output_names
    assert len(report.split.validation_indices) == 12
    assert "not evidence of physical accuracy" in report.interpretation
    assert report.nominal_probability == 0.9
    assert report.coverage_target == 0.9
    assert report.coverage_tolerance == 0.05
    assert report.minimum_coverage_sample_count == 30
    assert report.rmse_acceptance_threshold == 0.05
    assert report.software_reproducibility_passed
    assert not report.model_quality_passed
    assert report.assessment_limited
    assert report.calibration_fit_sample_count == 0
    assert report.uncertainty_status == "uncalibrated-held-out-assessment"
    assert all(
        len(value) == 64
        for value in (
            report.dataset_hash,
            report.split_hash,
            report.config_hash,
            report.benchmark_hash,
        )
    )
    assert report.dataset_hash == (
        "2647de548f3fe3ca3cc070eeca9fac187b877904bc729061ccc42c6c5ab77006"
    )
    assert report.split_hash == (
        "f44fdd4b2498ebc73157c91cc9bbbc4ed33e77fc8753825217b92f1ba43c4360"
    )
    assert report.config_hash == (
        "100834899102d9d6a19c410f82096e72b70c36612db5fe718f9ad70185e6fc7f"
    )
    assert report.benchmark_hash == (
        "ccf730a32bbb0bd82399e8728e36ffe24b9e2d8a051c2040703713d436c8a566"
    )
    assert report.split.validation_indices == (
        0,
        2,
        12,
        13,
        14,
        18,
        20,
        24,
        26,
        28,
        36,
        38,
    )
    for metrics in report.metrics:
        assert metrics.sample_count == 12
        assert all(
            isfinite(value)
            for value in (
                metrics.rmse,
                metrics.mae,
                metrics.worst_case_absolute_error,
                metrics.interval_coverage,
            )
        )
        assert metrics.rmse >= 0.0
        assert metrics.mae >= 0.0
        assert metrics.worst_case_absolute_error >= metrics.mae
        assert not metrics.coverage_accepted
        assert not metrics.rmse_accepted
        assert not metrics.model_quality_passed
        assert metrics.assessment_limited
        assert metrics.nominal_probability == 0.9
        assert metrics.coverage_target == 0.9
        assert metrics.coverage_tolerance == 0.05
    thrust, specific_impulse = report.metrics
    assert thrust.rmse == 0.007894429684864392
    assert thrust.mae == 0.006534945269753859
    assert thrust.worst_case_absolute_error == 0.01590895031552436
    assert thrust.interval_coverage == 0.75
    assert thrust.output_range == 0.034796449298740074
    assert thrust.range_normalized_rmse == 0.22687457611228848
    assert specific_impulse.rmse == 337.7674123910935
    assert specific_impulse.mae == 270.7299834538546
    assert specific_impulse.worst_case_absolute_error == 782.2225334223967
    assert specific_impulse.interval_coverage == 11 / 12
    assert specific_impulse.output_range == 1634.9201412627206
    assert specific_impulse.range_normalized_rmse == 0.20659566413453131
