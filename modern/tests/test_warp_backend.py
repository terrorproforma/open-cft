import json
from math import copysign, isclose
from random import Random

import pytest

import cft_revival.warp_backend as warp_backend
from cft_revival.cli import main
from cft_revival.kernels import cusp_arrival_probability_python
from cft_revival.models import ValidationError
from cft_revival.warp_backend import (
    available_warp_devices,
    cusp_arrival_probabilities_warp,
    warp_available,
    warp_device_available,
)


def _deterministic_fields(count: int) -> tuple[list[float], list[float]]:
    random = Random(20_170_032)
    low = [0.0, 1.0]
    high = [1.0, 1.0]
    for _ in range(count - 2):
        high_value = random.uniform(1.0e-8, 10.0)
        high.append(high_value)
        low.append(high_value * random.random())
    return low, high


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_matches_analytic_reference(device: str) -> None:
    if not warp_device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    low, high = _deterministic_fields(1_026)
    expected = [
        cusp_arrival_probability_python(low_value, high_value)
        for low_value, high_value in zip(low, high, strict=True)
    ]

    result = cusp_arrival_probabilities_warp(low, high, device=device)

    assert result.device == device
    assert all(
        isclose(actual, reference, rel_tol=1.0e-14, abs_tol=0.0)
        for actual, reference in zip(result.probabilities, expected, strict=True)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_preserves_tiny_ratios(device: str) -> None:
    if not warp_device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    low = [-0.0, 0.0, 1.0, 1.0e-18, 1.0e-30, 2.0e-323]
    high = [1.0] * len(low)
    expected = [0.0, 0.0, 0.5, 2.5e-19, 2.5e-31, 5.0e-324]

    result = cusp_arrival_probabilities_warp(low, high, device=device)

    assert copysign(1.0, result.probabilities[0]) == 1.0
    assert copysign(1.0, result.probabilities[1]) == 1.0
    assert all(
        isclose(actual, reference, rel_tol=1.0e-14, abs_tol=0.0)
        for actual, reference in zip(result.probabilities, expected, strict=True)
    )


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ([], []),
        ([0.1], []),
        ([-0.1], [1.0]),
        ([1.1], [1.0]),
        ([float("nan")], [1.0]),
        ([float("inf")], [1.0]),
        ([1.0], [float("inf")]),
        ([float("-inf")], [1.0]),
    ],
)
def test_warp_uses_scalar_validation_contract(
    low: list[float], high: list[float]
) -> None:
    with pytest.raises(ValidationError):
        cusp_arrival_probabilities_warp(low, high, device="cpu")


def test_warp_accepts_one_dimensional_numpy_batches() -> None:
    numpy = pytest.importorskip("numpy")
    if not warp_device_available("cpu"):
        pytest.skip("Warp CPU device is unavailable")
    low = numpy.array([0.0, 1.0e-18, 1.0], dtype=numpy.float64)
    high = numpy.ones(3, dtype=numpy.float64)

    result = cusp_arrival_probabilities_warp(low, high, device="cpu")

    assert result.probabilities[0] == 0.0
    assert isclose(result.probabilities[1], 2.5e-19, rel_tol=1.0e-14, abs_tol=0.0)
    assert result.probabilities[2] == 0.5


def test_warp_rejects_multidimensional_numpy_batches() -> None:
    numpy = pytest.importorskip("numpy")
    with pytest.raises(ValidationError, match="one-dimensional"):
        cusp_arrival_probabilities_warp(
            numpy.ones((2, 2)), numpy.ones((2, 2)), device="cpu"
        )


def test_missing_warp_fails_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(warp_backend, "wp", None)
    assert warp_backend.warp_available() is False
    assert warp_backend.available_warp_devices() == ()
    assert warp_backend.warp_device_available("cpu") is False
    with pytest.raises(RuntimeError, match="Warp is unavailable"):
        warp_backend.cusp_arrival_probabilities_warp([0.1], [1.0], device="cpu")


def test_unavailable_warp_device_fails_cleanly() -> None:
    if not warp_available():
        pytest.skip("Warp is unavailable")
    assert available_warp_devices()
    assert warp_device_available("cuda:999999") is False
    with pytest.raises(RuntimeError, match="unavailable"):
        cusp_arrival_probabilities_warp([0.1], [1.0], device="cuda:999999")


def test_benchmark_cli_labels_timing_non_authoritative(capsys) -> None:
    if not warp_device_available("cpu"):
        pytest.skip("Warp CPU device is unavailable")

    assert (
        main(
            [
                "benchmark-cusp",
                "--device",
                "cpu",
                "--batch-size",
                "16",
                "--warmup",
                "0",
                "--repeat",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output[output.find("{") :])
    assert report["device"] == "cpu"
    assert report["max_abs_error"] <= 1.0e-14
    assert report["timing_authoritative"] is False
