from math import isclose
from random import Random

import pytest

import cft_revival.physics.warp_backend as warp_backend
from cft_revival.physics import (
    OptionalDependencyError,
    PhysicsDeviceError,
    PhysicsValidationError,
    evaluate_batch,
)
from cft_revival.physics.warp_backend import (
    available_devices,
    device_available,
    evaluate_performance_warp,
)


def _deterministic_points(point_factory, count: int):
    random = Random(50902020)
    points = [
        point_factory(mass_flow_kg_per_s=1.0e-300, cathode_power_w=0.0),
        point_factory(voltage_v=1.0e-300, cathode_power_w=0.0),
        point_factory(neutral=1.0, plus=0.0, double_plus=0.0),
        point_factory(
            neutral=0.0,
            plus=0.0,
            double_plus=1.0,
            divergence_factor=0.0,
        ),
    ]
    for _ in range(count - len(points)):
        neutral = random.random()
        double_plus = random.random() * (1.0 - neutral)
        plus = 1.0 - neutral - double_plus
        points.append(
            point_factory(
                voltage_v=random.uniform(0.0, 2000.0),
                mass_flow_kg_per_s=random.uniform(0.0, 5.0e-6),
                neutral=neutral,
                plus=plus,
                double_plus=double_plus,
                beam_factor=random.uniform(0.1, 1.0),
                divergence_factor=random.random(),
                cathode_power_w=random.uniform(0.0, 100.0),
                ppu_margin_w=random.uniform(0.01, 200.0),
            )
        )
    return points


def _observables(result):
    budget = result.power_budget
    diagnostics = result.diagnostics
    return (
        result.total_xenon_particle_rate_per_s,
        result.neutral_particle_rate_per_s,
        result.xe_plus_particle_rate_per_s,
        result.xe_double_plus_particle_rate_per_s,
        result.xe_plus_speed_m_per_s,
        result.xe_double_plus_speed_m_per_s,
        result.undiverged_ion_thrust_n,
        result.axial_thrust_n,
        result.specific_impulse_s,
        budget.beam_current_a,
        budget.anode_current_a,
        budget.beam_kinetic_power_w,
        budget.anode_input_power_w,
        budget.thruster_electrical_input_power_w,
        budget.requested_ppu_input_power_w,
        budget.ppu_input_power_w,
        budget.ppu_boundary_adjustment_w,
        budget.ppu_conversion_loss_w,
        budget.anode_to_beam_efficiency,
        budget.thruster_electrical_to_beam_efficiency,
        budget.ppu_input_to_beam_efficiency,
        diagnostics.ppu_power_margin_w,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_float64_matches_python_reference(device: str, point_factory) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    points = _deterministic_points(point_factory, 257)
    expected = evaluate_batch(points)

    actual_batch = evaluate_performance_warp(points, device=device)

    assert actual_batch.device == device
    for actual, reference in zip(actual_batch.results, expected, strict=True):
        for actual_value, reference_value in zip(
            _observables(actual), _observables(reference), strict=True
        ):
            if reference_value is None:
                assert actual_value is None
            elif reference_value == 0.0:
                assert actual_value == 0.0
            else:
                assert isclose(
                    actual_value,
                    reference_value,
                    rel_tol=2.0e-14,
                    abs_tol=0.0,
                )
        diagnostics = actual.diagnostics
        assert abs(diagnostics.particle_rate_residual_particles_per_s) <= (
            4e-16 * max(1.0, actual.total_xenon_particle_rate_per_s)
        )
        assert abs(diagnostics.mass_flow_residual_kg_per_s) <= 3e-21
        assert abs(diagnostics.beam_current_residual_a) <= 2e-13
        assert abs(diagnostics.beam_power_residual_w) <= (
            5e-14 * max(1.0, actual.power_budget.beam_kinetic_power_w)
        )
        assert actual.applicability_warnings == reference.applicability_warnings


def test_cuda_alias_is_explicitly_cuda_zero(point_factory) -> None:
    if not device_available("cuda:0"):
        pytest.skip("Warp CUDA device is unavailable")
    result = evaluate_performance_warp([point_factory()], device="cuda")
    assert result.device == "cuda:0"


def test_warp_rejects_empty_multidimensional_and_wrong_entry(point_factory) -> None:
    with pytest.raises(PhysicsValidationError, match="empty"):
        evaluate_performance_warp([], device="cpu")
    with pytest.raises(PhysicsValidationError, match="XenonOperatingPoint"):
        evaluate_performance_warp([object()], device="cpu")

    numpy = pytest.importorskip("numpy")
    matrix = numpy.array(
        [[point_factory(), point_factory()], [point_factory(), point_factory()]],
        dtype=object,
    )
    with pytest.raises(PhysicsValidationError, match="one-dimensional"):
        evaluate_performance_warp(matrix, device="cpu")


def test_missing_optional_warp_fails_cleanly(monkeypatch, point_factory) -> None:
    monkeypatch.setattr(warp_backend, "wp", None)
    assert warp_backend.warp_available() is False
    assert warp_backend.available_devices() == ()
    assert warp_backend.device_available("cpu") is False
    with pytest.raises(OptionalDependencyError, match="Warp is unavailable"):
        warp_backend.evaluate_performance_warp([point_factory()], device="cpu")


@pytest.mark.parametrize("device", ["auto", "gpu", "cuda:999999"])
def test_invalid_or_unavailable_device_fails(device: str, point_factory) -> None:
    if not warp_backend.warp_available():
        pytest.skip("Warp is unavailable")
    with pytest.raises(PhysicsDeviceError):
        evaluate_performance_warp([point_factory()], device=device)
    assert device_available(device) is False


def test_available_devices_reports_cpu_and_runtime_devices() -> None:
    if not warp_backend.warp_available():
        pytest.skip("Warp is unavailable")
    devices = available_devices()
    assert "cpu" in devices
