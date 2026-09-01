from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite, sin

import pytest

import cft_revival.surrogates._linalg as linalg
from cft_revival.surrogates import (
    BoTorchTrainingData,
    ExactGP,
    IndependentMultiOutputGP,
    OODDetector,
    SurrogateSchema,
    SurrogateValidationError,
)


def test_exact_gp_scales_predicts_and_handles_heteroskedastic_duplicates() -> None:
    x = ((0.0,), (0.25,), (0.5,), (0.5,), (0.75,), (1.0,))
    y = tuple(1.0e6 + 2.0e5 * sin(3.0 * row[0]) for row in x)
    noise = (4.0, 9.0, 16.0, 25.0, 36.0, 49.0)
    model = ExactGP.fit(
        x,
        y,
        observation_variance=noise,
        schema=SurrogateSchema(("position",), ("response",), ("m",), ("Pa",)),
    )

    prediction = model.predict(((0.4,),))[0]
    assert isfinite(prediction.mean)
    assert isfinite(prediction.variance)
    assert prediction.variance >= 0.0
    assert model.diagnostics.heteroskedastic_noise
    assert model.diagnostics.jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
    assert abs(prediction.mean - (1.0e6 + 2.0e5 * sin(1.2))) < 5.0e4


def test_ill_conditioned_cluster_uses_bounded_jitter_and_stays_finite() -> None:
    x = ((0.0,), (1e-14,), (2e-14,), (0.5,), (1.0,))
    y = (0.0, 1e-14, 2e-14, 0.5, 1.0)
    model = ExactGP.fit(x, y)
    predictions = model.predict(((1e-14,), (0.25,), (2.0,)))
    assert all(isfinite(item.mean) and item.variance >= 0.0 for item in predictions)


@pytest.mark.parametrize(
    ("x", "y", "noise"),
    [
        (((0.0,), (float("nan"),)), (0.0, 1.0), None),
        (((0.0,), (1.0,)), (0.0, float("inf")), None),
        (((0.0,), (1.0,)), (0.0, 1.0), (0.0, float("nan"))),
        (((0.0,), (1.0,)), (0.0, 1.0), (0.0, -1.0)),
    ],
)
def test_nonfinite_and_negative_noise_data_fail_closed(x, y, noise) -> None:
    with pytest.raises(SurrogateValidationError):
        ExactGP.fit(x, y, observation_variance=noise)


def test_serialization_hashes_and_reload_are_deterministic(tmp_path) -> None:
    model = ExactGP.fit(
        ((0.0,), (0.3,), (0.7,), (1.0,)),
        (0.0, 0.4, 0.8, 1.0),
        observation_variance=(0.01, 0.02, 0.03, 0.04),
    )
    path = tmp_path / "gp.json"
    model.save(path)
    reloaded = ExactGP.load(path)
    assert reloaded.dumps() == model.dumps()
    assert reloaded.predict(((0.2,), (0.8,))) == model.predict(((0.2,), (0.8,)))

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["training_data"]["raw"]["train_y"][0] = 99.0
    with pytest.raises(SurrogateValidationError, match="hash mismatch"):
        ExactGP.loads(json.dumps(tampered))


def _rehash(artifact: dict[str, object]) -> None:
    without_hash = {
        key: value for key, value in artifact.items() if key != "model_hash"
    }
    encoded = json.dumps(
        without_hash, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    artifact["model_hash"] = sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("kernel", "version", "matern-5/2-v999"),
        ("kernel", "family", "RBF"),
        ("hyperparameters", "length_scale_bounds", [0.01, 9.0]),
    ],
)
def test_full_executable_policy_is_integrity_checked(section, key, value) -> None:
    model = ExactGP.fit(
        ((0.0, 0.0), (0.5, 0.2), (1.0, 1.0)),
        (0.0, 0.4, 1.0),
        calibration_scale=1.2,
        nominal_probability=0.9,
    )
    artifact = json.loads(model.dumps())
    artifact["executable_policy"][section][key] = value
    with pytest.raises(SurrogateValidationError, match="model hash mismatch"):
        ExactGP.loads(json.dumps(artifact))
    _rehash(artifact)
    with pytest.raises(
        SurrogateValidationError,
        match="unsupported|deterministic reconstruction",
    ):
        ExactGP.loads(json.dumps(artifact))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("calibration", "variance_scale", 7.0),
        ("calibration", "nominal_probability", 0.8),
        ("ood", "threshold_multiplier", 8.0),
    ],
)
def test_configurable_policy_cannot_change_without_hash(section, key, value) -> None:
    artifact = json.loads(
        ExactGP.fit(((0.0,), (1.0,)), (0.0, 1.0)).dumps()
    )
    artifact["executable_policy"][section][key] = value
    with pytest.raises(SurrogateValidationError, match="model hash mismatch"):
        ExactGP.loads(json.dumps(artifact))


def test_unknown_policy_key_is_rejected_even_with_recomputed_hash() -> None:
    artifact = json.loads(
        ExactGP.fit(((0.0,), (1.0,)), (0.0, 1.0)).dumps()
    )
    artifact["executable_policy"]["kernel"]["unknown"] = True
    _rehash(artifact)
    with pytest.raises(SurrogateValidationError, match="keys mismatch"):
        ExactGP.loads(json.dumps(artifact))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("training_data", "normalized", "train_y", 0), 123.0),
        (("normalization", "input", "minimum", 0), -7.0),
        (("fitted_parameters", "length_scales", 0), 3.0),
        (("fitted_parameters", "jitter"), 1e-4),
        (("executable_policy", "output_semantics", "mean"), "unknown"),
    ],
)
def test_rehashed_internal_tampering_fails_reconstruction(path, value) -> None:
    artifact = json.loads(
        ExactGP.fit(
            ((0.0, 0.0), (0.3, 0.7), (1.0, 1.0)),
            (0.0, 0.5, 1.0),
        ).dumps()
    )
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _rehash(artifact)
    with pytest.raises(SurrogateValidationError):
        ExactGP.loads(json.dumps(artifact))


def test_ard_recovers_anisotropic_scales_and_isotropic_mode_is_explicit() -> None:
    x = tuple(
        (first / 7.0, second / 5.0)
        for first in range(8)
        for second in range(6)
    )
    y = tuple(sin(12.0 * row[0]) + 0.05 * row[1] for row in x)
    ard = ExactGP.fit(x, y, length_scale_mode="ard")
    isotropic = ExactGP.fit(x, y, length_scale_mode="isotropic")
    assert ard.diagnostics.length_scale_mode == "ard"
    assert len(ard.diagnostics.length_scales) == 2
    assert ard.diagnostics.length_scales[0] < ard.diagnostics.length_scales[1]
    assert isotropic.diagnostics.length_scale_mode == "isotropic"
    assert len(set(isotropic.diagnostics.length_scales)) == 1
    assert ExactGP.loads(ard.dumps()).diagnostics.length_scales == (
        ard.diagnostics.length_scales
    )


def test_extreme_finite_extrapolation_is_finite_or_typed_rejection() -> None:
    model = ExactGP.fit(((0.0,), (1.0,)), (0.0, 1.0))
    prediction = model.predict(((1e308,),))[0]
    assert isfinite(prediction.mean)
    assert isfinite(prediction.variance)
    with pytest.raises(SurrogateValidationError):
        ExactGP.fit(((-1e308,), (1e308,)), (0.0, 1.0))


def test_multioutput_and_botorch_contract_shapes() -> None:
    x = ((0.0,), (0.3,), (0.7,), (1.0,))
    y = tuple((row[0], row[0] ** 2) for row in x)
    model = IndependentMultiOutputGP.fit(x, y)
    prediction = model.predict(((0.5,),))
    assert len(prediction) == 1
    assert len(prediction[0]) == 2

    contract = BoTorchTrainingData.from_exact_gp(model.models[0])
    assert len(contract.train_x) == 4
    assert all(len(row) == 1 for row in contract.train_y)
    assert contract.schema_hash == model.models[0].schema_hash


def test_ood_reports_domain_extrapolation() -> None:
    detector = OODDetector.fit(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)))
    assert not detector.report((0.1, 0.1)).is_out_of_distribution
    report = detector.report((1.5, 0.5))
    assert report.is_out_of_distribution
    assert report.domain_excess_distance == pytest.approx(0.5)
    model = ExactGP.fit(
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)),
        (0.0, 1.0, 1.0, 2.0),
        ood_threshold_multiplier=2.0,
        ood_quantile=0.8,
    )
    governed = model.ood_detector()
    assert governed.threshold_multiplier == 2.0
    assert governed.threshold_quantile == 0.8


def test_ood_detects_drift_along_constant_training_dimension() -> None:
    detector = OODDetector.fit(((0.0, 2.0), (0.5, 2.0), (1.0, 2.0)))
    assert detector.report((0.5, 3.0)).is_out_of_distribution


def test_standard_library_cholesky_fallback_is_operational(monkeypatch) -> None:
    monkeypatch.setattr(linalg, "_np", None)
    model = ExactGP.fit(((0.0,), (0.4,), (1.0,)), (0.0, 0.5, 1.0))
    prediction = model.predict(((0.7,),))[0]
    assert isfinite(prediction.mean)
    assert prediction.variance >= 0.0
