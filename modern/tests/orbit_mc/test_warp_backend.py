from __future__ import annotations

from math import pi

import numpy as np
import pytest

from cft_revival.orbit_mc import (
    AnalyticField,
    ElectronLaunch,
    OrbitConfig,
    backend_parity,
    integrate_orbit,
    integrate_orbit_warp,
    warp_status,
)


def test_warp_cpu_relativistic_push_matches_reference() -> None:
    status = warp_status()
    if not status.cpu_available:
        pytest.skip(status.reason)
    report = backend_parity(device="cpu")
    assert report["status"] == "evaluated"
    assert report["maximum_relative_velocity_difference"] < 2.0e-14


def test_warp_cuda_relativistic_push_matches_reference_when_available() -> None:
    status = warp_status()
    if not status.cuda_available:
        pytest.skip(status.reason)
    report = backend_parity(device="cuda:0")
    assert report["status"] == "evaluated"
    assert report["maximum_relative_velocity_difference"] < 2.0e-14


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_complete_warp_orbit_matches_cpu_event_loop(device: str) -> None:
    status = warp_status()
    if not status.cpu_available or (device.startswith("cuda") and not status.cuda_available):
        pytest.skip(status.reason)
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    launch = ElectronLaunch("warp-orbit", 0, 25.0, pi/4, (0.0, 0.0, 0.0), 1, 0.3, "axis")
    dt = 2.0e-13
    config = OrbitConfig(
        0.1, -1.0, 1.0, 0.2, -2.0, 2.0, 8*dt, 1.0,
        max_steps=8, max_rotation_rad=0.1, fixed_dt_s=dt,
    )
    reference = integrate_orbit(launch, field, config)
    candidate = integrate_orbit_warp(launch, field, config, device=device)
    assert np.allclose(candidate.final_position_m, reference.final_position_m, rtol=0.0, atol=1.0e-15)
    assert np.allclose(candidate.final_velocity_m_per_s, reference.final_velocity_m_per_s, rtol=0.0, atol=1.0e-12)
    assert candidate.termination == reference.termination
