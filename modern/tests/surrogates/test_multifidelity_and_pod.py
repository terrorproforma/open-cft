from __future__ import annotations

import json
from math import cos, isfinite, sin

import pytest

import cft_revival.surrogates._linalg as linalg
from cft_revival.surrogates import (
    PODBasis,
    PODFieldSurrogate,
    Prediction,
    SurrogateError,
    SurrogateValidationError,
    TwoFidelityAR1,
    VarianceCalibrator,
    fixed_mesh_hash,
    grouped_spatial_split,
    regression_metrics,
)


def test_ar1_learns_scale_and_discrepancy_without_merging_fidelities() -> None:
    low_x = tuple((index / 15.0,) for index in range(16))
    low_y = tuple(sin(4.0 * row[0]) for row in low_x)
    high_x = low_x[::3]
    high_y = tuple(
        1.7 * sin(4.0 * row[0]) + 0.15 * row[0] ** 2 for row in high_x
    )
    model = TwoFidelityAR1.fit(low_x, low_y, high_x, high_y)
    points = ((0.2,), (0.55,), (0.9,))
    high = model.predict(points, fidelity="high")
    low = model.predict(points, fidelity="low")

    assert 1.0 < model.rho < 2.5
    assert all(item.variance >= 0.0 and isfinite(item.mean) for item in high)
    assert any(abs(a.mean - b.mean) > 0.01 for a, b in zip(high, low, strict=True))
    assert "independent" in model.diagnostics.variance_assumption
    assert TwoFidelityAR1.loads(model.dumps()).dumps() == model.dumps()
    tampered = json.loads(model.dumps())
    tampered["rho"] = 2.9
    with pytest.raises(SurrogateValidationError, match="model hash mismatch"):
        TwoFidelityAR1.loads(json.dumps(tampered))


def test_grouped_spatial_split_prevents_group_leakage_and_is_deterministic() -> None:
    inputs = tuple((index // 2 / 9.0, index % 2) for index in range(20))
    groups = tuple(f"design-{index // 2}" for index in range(20))
    first = grouped_spatial_split(inputs, groups, validation_fraction=0.3, seed=4)
    second = grouped_spatial_split(inputs, groups, validation_fraction=0.3, seed=4)
    assert first == second
    train_groups = {groups[index] for index in first.training_indices}
    validation_groups = {groups[index] for index in first.validation_indices}
    assert train_groups.isdisjoint(validation_groups)
    assert len(validation_groups) == 3


def test_calibration_and_metrics_report_required_quantities() -> None:
    truth = (0.0, 1.0, 2.0, 3.0)
    predictions = tuple(
        Prediction(value + 0.1, 0.01, nominal_probability=0.9)
        for value in truth
    )
    calibrator = VarianceCalibrator.fit(
        truth, predictions, nominal_probability=0.9
    )
    calibrated = tuple(calibrator.apply(item) for item in predictions)
    metrics = regression_metrics(
        truth,
        calibrated,
        nominal_probability=0.9,
        coverage_target=0.9,
        coverage_tolerance=0.2,
    )
    assert metrics.rmse == pytest.approx(0.1)
    assert metrics.mae == pytest.approx(0.1)
    assert metrics.worst_case_absolute_error == pytest.approx(0.1)
    assert metrics.nominal_probability == 0.9
    assert metrics.coverage_target == 0.9
    assert metrics.sample_count == 4
    assert metrics.assessment_role == "held-out-assessment"
    assert calibrator.fit_role == "calibration-fit"
    assert metrics.assessment_limited
    assert not metrics.coverage_accepted
    rejected = regression_metrics(
        truth,
        tuple(Prediction(value + 10.0, 0.01, 0.9) for value in truth),
        nominal_probability=0.9,
        coverage_target=0.9,
        coverage_tolerance=0.1,
    )
    assert rejected.interval_coverage == 0.0
    assert not rejected.coverage_accepted


def test_quality_gate_requires_predeclared_error_and_coverage_rules() -> None:
    truth = tuple(float(index) for index in range(40))
    predictions = tuple(
        Prediction(value if index < 36 else value + 1.0, 0.0, 0.9)
        for index, value in enumerate(truth)
    )
    metrics = regression_metrics(
        truth,
        predictions,
        nominal_probability=0.9,
        coverage_target=0.9,
        coverage_tolerance=0.05,
        minimum_coverage_sample_count=30,
        rmse_acceptance_threshold=0.05,
    )
    assert not metrics.assessment_limited
    assert metrics.interval_coverage == 0.9
    assert metrics.coverage_accepted
    assert metrics.rmse_accepted
    assert metrics.model_quality_passed


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Prediction(float("nan"), 1.0),
        lambda: Prediction(0.0, float("inf")),
        lambda: Prediction(0.0, -1.0),
        lambda: Prediction(0.0, 1.0, nominal_probability=float("nan")),
    ],
)
def test_invalid_prediction_values_raise_typed_surrogate_errors(factory) -> None:
    with pytest.raises(SurrogateError):
        factory()


def test_calibration_and_metrics_fail_closed_on_lengths_and_overflow() -> None:
    prediction = Prediction(-1e308, 1.0)
    with pytest.raises(SurrogateError):
        VarianceCalibrator.fit((1e308,), (prediction,))
    with pytest.raises(SurrogateError):
        regression_metrics((0.0, 1.0), (Prediction(0.0, 1.0),))
    with pytest.raises(SurrogateError):
        regression_metrics(
            (1e308,),
            (prediction,),
            nominal_probability=0.95,
        )
    corrupted = Prediction(0.0, 1.0)
    object.__setattr__(corrupted, "variance", -1.0)
    with pytest.raises(SurrogateError):
        VarianceCalibrator.fit((0.0,), (corrupted,))
    with pytest.raises(SurrogateError):
        regression_metrics((0.0,), (corrupted,))


def test_pod_field_surrogate_reconstructs_fixed_mesh_and_rejects_mesh_drift() -> None:
    mesh = tuple((index / 10.0,) for index in range(11))
    mesh_id = fixed_mesh_hash(mesh)
    train_x = tuple((index / 8.0,) for index in range(9))
    fields = tuple(
        tuple(
            (1.0 + parameter[0]) * sin(3.0 * coordinate[0])
            + 0.2 * parameter[0] * cos(5.0 * coordinate[0])
            for coordinate in mesh
        )
        for parameter in train_x
    )
    model = PODFieldSurrogate.fit(train_x, fields, rank=2, mesh_hash=mesh_id)
    prediction = model.predict(((0.45,),), mesh_hash=mesh_id)[0]
    expected = tuple(
        1.45 * sin(3.0 * coordinate[0]) + 0.09 * cos(5.0 * coordinate[0])
        for coordinate in mesh
    )
    assert max(abs(a - b) for a, b in zip(prediction.mean_field, expected, strict=True)) < 0.1
    assert len(prediction.pointwise_variance) == len(mesh)
    assert model.basis.retained_energy_fraction > 0.999
    with pytest.raises(SurrogateValidationError, match="mesh hash"):
        model.predict(((0.45,),), mesh_hash="0" * 64)


def test_zero_energy_pod_is_mean_only_and_backend_independent(monkeypatch) -> None:
    mesh = ((-0.0,), (0.5,), (1.0,))
    mesh_id = fixed_mesh_hash(mesh)
    fields = ((2.0, 3.0, 4.0),) * 4
    numpy_basis = PODBasis.fit(fields, rank=2, mesh_hash=mesh_id)
    monkeypatch.setattr(linalg, "_np", None)
    fallback_basis = PODBasis.fit(fields, rank=2, mesh_hash=mesh_id)
    assert numpy_basis == fallback_basis
    assert numpy_basis.effective_rank == 0
    assert numpy_basis.representation == "mean-only-rank-0"
    assert numpy_basis.reconstruct(()) == fields[0]
    assert PODBasis.loads(numpy_basis.dumps()) == numpy_basis
    surrogate = PODFieldSurrogate.fit(
        ((0.0,), (0.3,), (0.7,), (1.0,)),
        fields,
        rank=2,
        mesh_hash=mesh_id,
    )
    prediction = surrogate.predict(((0.4,),), mesh_hash=mesh_id)[0]
    assert prediction.mean_field == fields[0]
    assert prediction.pointwise_variance == (0.0, 0.0, 0.0)


def test_coordinate_closure_prevents_leakage_and_canonicalizes_signed_zero() -> None:
    inputs = ((-0.0, 1.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0))
    groups = ("caller-a", "caller-b", "caller-c", "caller-d")
    split = grouped_spatial_split(
        inputs, groups, validation_fraction=0.5, seed=2
    )
    assert (0 in split.training_indices) == (1 in split.training_indices)
    assert fixed_mesh_hash(((-0.0,), (1.0,))) == fixed_mesh_hash(
        ((0.0,), (1.0,))
    )
    with pytest.raises(SurrogateValidationError, match="exactly zero"):
        grouped_spatial_split(
            inputs,
            groups,
            validation_fraction=0.5,
            coordinate_tolerance=1e-9,
        )
