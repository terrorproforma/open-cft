"""Manufactured verification cases for fields, events, and orbit integration."""

from __future__ import annotations

from math import pi, sin, sqrt

import numpy as np

from .fields import AnalyticField, PsiBicubicField
from .integrator import _segment_event, integrate_orbit, relativistic_boris_push
from .models import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    LIGHT_SPEED_M_PER_S,
    ElectronLaunch,
    OrbitConfig,
    Termination,
)
from .warp_backend import warp_boris_push_batch, warp_status


def _config(
    duration_s: float, dt_s: float, *, wall_radius_m: float = 10.0,
    radius_m: float = 20.0, z_extent_m: float = 20.0,
) -> OrbitConfig:
    return OrbitConfig(
        wall_radius_m, -z_extent_m, z_extent_m, radius_m, -z_extent_m, z_extent_m,
        duration_s, 100.0, max_steps=int(round(duration_s/dt_s)),
        max_rotation_rad=0.5, fixed_dt_s=dt_s,
    )


def uniform_b_helix(steps_per_gyrocycle: int = 128, cycles: int = 3) -> dict[str, float]:
    b_t = 0.02
    omega_nonrelativistic = ELECTRON_CHARGE_C*b_t/ELECTRON_MASS_KG
    gamma = 1.0 + 100.0 * abs(ELECTRON_CHARGE_C) / (
        ELECTRON_MASS_KG * LIGHT_SPEED_M_PER_S**2
    )
    omega = omega_nonrelativistic/gamma
    period = 2.0*pi/abs(omega)
    dt = period/steps_per_gyrocycle
    duration = cycles*period
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, b_t]), None, b_t)
    launch = ElectronLaunch("uniform-b", 0, 100.0, pi/3, (0.0, 0.0, 0.0), 1, 0.0, "uniform")
    result = integrate_orbit(launch, field, _config(duration, dt))
    initial_v = np.asarray(result.final_velocity_m_per_s)  # recover exact start independently below
    from .integrator import launch_velocity
    v0 = launch_velocity(launch, field)
    t = result.elapsed_time_s
    exact_v = np.array([
        v0[0]*np.cos(omega*t) + v0[1]*np.sin(omega*t),
        v0[1]*np.cos(omega*t) - v0[0]*np.sin(omega*t),
        v0[2],
    ])
    exact_x = np.array([
        (v0[0]*np.sin(omega*t) + v0[1]*(1.0-np.cos(omega*t)))/omega,
        (v0[1]*np.sin(omega*t) - v0[0]*(1.0-np.cos(omega*t)))/omega,
        v0[2]*t,
    ])
    return {
        "position_error_m": float(np.linalg.norm(np.asarray(result.final_position_m)-exact_x)),
        "velocity_error_m_per_s": float(np.linalg.norm(initial_v-exact_v)),
        "relative_energy_error": result.maximum_relative_energy_error,
        "phase_error_rad": abs(
            result.accumulated_gyro_phase_rad - cycles * 2.0 * pi
        ),
        "complete_gyrocycles": float(result.complete_gyrocycles),
    }


def uniform_e_acceleration(steps: int = 200) -> dict[str, float]:
    electric = np.array([2.0e5, 0.0, 0.0])
    dt = 2.0e-13
    velocity = np.zeros(3)
    magnetic = np.zeros(3)
    for _ in range(steps):
        velocity = relativistic_boris_push(velocity, electric, magnetic, dt)
    speed2 = float(np.dot(velocity, velocity))
    gamma = 1.0/sqrt(1.0-speed2/LIGHT_SPEED_M_PER_S**2)
    momentum = ELECTRON_MASS_KG*gamma*velocity
    exact = ELECTRON_CHARGE_C*electric*(steps*dt)
    return {
        "momentum_error_kg_m_per_s": float(np.linalg.norm(momentum-exact)),
        "relative_momentum_error": float(np.linalg.norm(momentum-exact)/np.linalg.norm(exact)),
    }


def varying_e_convergence() -> dict[str, object]:
    """Verify second-order midpoint field sampling for a spatially varying E."""

    b_t = 0.01
    electric_scale = -2.0e4
    gradient_per_m = 25.0
    duration = 2.0e-10
    field = AnalyticField(
        lambda _position: np.array([0.0, 0.0, b_t]),
        lambda position, _time: np.array(
            [0.0, 0.0, electric_scale * (1.0 + gradient_per_m * position[2])]
        ),
        b_t,
    )
    launch = ElectronLaunch(
        "varying-e",
        0,
        1.0e-3,
        0.0,
        (0.0, 0.0, 0.0),
        1,
        0.0,
        "axis",
    )

    def solve(steps: int) -> float:
        dt = duration / steps
        config = OrbitConfig(
            0.2,
            -0.2,
            0.2,
            0.3,
            -0.3,
            0.3,
            duration,
            1.0,
            max_steps=steps,
            max_rotation_rad=0.5,
            fixed_dt_s=dt,
        )
        return integrate_orbit(launch, field, config).final_position_m[2]

    reference = solve(3200)
    levels = (100, 200, 400)
    values = [solve(level) for level in levels]
    errors = [abs(value - reference) for value in values]
    orders = [
        float(np.log(coarse / fine) / np.log(2.0))
        for coarse, fine in zip(errors, errors[1:])
    ]
    return {
        "steps": list(levels),
        "position_errors_m": errors,
        "observed_orders": orders,
        "reference_position_m": reference,
    }


def analytic_magnetic_bottle(
    *, steps_per_gyrocycle: int = 64, pitch_angle_rad: float = pi/3
) -> dict[str, float | str]:
    b0 = 0.02
    curvature = 10000.0
    expected_z = sqrt((1.0/sin(pitch_angle_rad)**2 - 1.0)/curvature)

    def magnetic(position: np.ndarray) -> np.ndarray:
        x, y, z = position
        return np.array([-curvature*b0*x*z, -curvature*b0*y*z, b0*(1.0+curvature*z*z)])

    max_b = b0*(1.0+curvature*0.02**2)
    field = AnalyticField(magnetic, None, max_b)
    dt = 2.0*pi*ELECTRON_MASS_KG/(abs(ELECTRON_CHARGE_C)*max_b*steps_per_gyrocycle)
    launch = ElectronLaunch("mirror", 0, 50.0, pitch_angle_rad, (2.0e-5, 0.0, 0.0), 1, 0.0, "axis-near")
    config = OrbitConfig(
        0.05, -0.02, 0.02, 0.06, -0.02, 0.02, 2.0e-8, 0.1,
        max_steps=100_000, max_rotation_rad=0.5, fixed_dt_s=dt,
    )
    result = integrate_orbit(launch, field, config)
    observed = result.final_position_m[2]
    final_magnetic = magnetic(np.asarray(result.final_position_m))
    final_parallel = float(
        np.dot(
            np.asarray(result.final_velocity_m_per_s),
            final_magnetic / np.linalg.norm(final_magnetic),
        )
    )
    return {
        "termination": result.termination.value,
        "expected_mirror_z_m": expected_z,
        "observed_reflection_z_m": observed,
        "absolute_error_m": abs(observed-expected_z),
        "relative_error": abs(observed-expected_z)/expected_z,
        "mu_relative_variation": result.maximum_instantaneous_mu_relative_variation or 0.0,
        "final_parallel_velocity_m_per_s": final_parallel,
    }


def manufactured_interpolator(intervals: int = 16) -> tuple[PsiBicubicField, dict[str, float | int]]:
    b0 = 0.3
    curvature = 2.0
    r = np.linspace(0.0, 0.4, intervals+1)
    z = np.linspace(-0.5, 0.5, 2*intervals+1)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = 0.5*b0*rr**2*(1.0+curvature*zz**2)
    br = -b0*curvature*rr*zz
    bz = b0*(1.0+curvature*zz**2)
    material = np.full(psi.shape, "plasma", dtype=object)
    field = PsiBicubicField(
        r,
        z,
        psi,
        material_id=material,
        plasma_material_id="plasma",
        reference_br_t=br,
        reference_bz_t=bz,
    )
    return field, field.reference_error().to_dict()


def wall_event_accuracy() -> dict[str, float | str]:
    config = OrbitConfig(1.0, -1.0, 1.0, 2.0, -2.0, 2.0, 1.0, 10.0)
    start = np.array([0.25, 0.0, 0.0])
    end = np.array([1.25, 0.0, 0.2])
    event = _segment_event(start, end, config)
    assert event is not None
    termination, point, fraction = event
    exact_fraction = 0.75
    exact_point = start + exact_fraction*(end-start)
    return {
        "termination": termination.value,
        "fraction_error": abs(fraction-exact_fraction),
        "endpoint_error_m": float(np.linalg.norm(point-exact_point)),
    }


def timestep_convergence() -> dict[str, object]:
    reports = [uniform_b_helix(steps) for steps in (64, 128, 256)]
    errors = [float(report["position_error_m"]) for report in reports]
    orders = [
        np.log(coarse/fine)/np.log(2.0) for coarse, fine in zip(errors, errors[1:])
    ]
    return {"steps_per_cycle": [64, 128, 256], "position_errors_m": errors, "observed_orders": orders}


def grad_b_drift_ordering(cycles: int = 30) -> dict[str, float | str]:
    """Check first-order guiding-centre grad-B drift sign and magnitude."""

    b0 = 0.05
    gradient_per_m = 5.0

    def magnetic(position: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, b0*(1.0+gradient_per_m*position[0])])

    field = AnalyticField(magnetic, None, 0.075)
    launch = ElectronLaunch("grad-b", 0, 100.0, pi/2, (0.0, 0.0, 0.0), 1, 0.0, "local")
    gamma = 1.0 + 100.0*abs(ELECTRON_CHARGE_C)/(ELECTRON_MASS_KG*LIGHT_SPEED_M_PER_S**2)
    speed = LIGHT_SPEED_M_PER_S*sqrt(1.0-gamma**-2)
    omega = abs(ELECTRON_CHARGE_C)*b0/(gamma*ELECTRON_MASS_KG)
    duration = cycles*2.0*pi/omega
    dt = (2.0*pi/omega)/128.0
    config = OrbitConfig(
        0.02, -0.1, 0.1, 0.03, -0.2, 0.2, duration, 1.0,
        max_steps=cycles*128, max_rotation_rad=0.08, fixed_dt_s=dt,
    )
    from .integrator import launch_velocity
    initial_velocity = launch_velocity(launch, field)
    result = integrate_orbit(launch, field, config)

    def guiding_centre(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        magnetic_value = magnetic(position)
        magnitude2 = float(np.dot(magnetic_value, magnetic_value))
        velocity_gamma = 1.0/sqrt(
            1.0-float(np.dot(velocity, velocity))/LIGHT_SPEED_M_PER_S**2
        )
        return position + (
            ELECTRON_MASS_KG*velocity_gamma
            /(ELECTRON_CHARGE_C*magnitude2)
        )*np.cross(velocity, magnetic_value)

    initial_centre = guiding_centre(np.zeros(3), initial_velocity)
    final_centre = guiding_centre(
        np.asarray(result.final_position_m), np.asarray(result.final_velocity_m_per_s)
    )
    observed = (final_centre[1]-initial_centre[1])/result.elapsed_time_s
    expected = (
        gamma*ELECTRON_MASS_KG*speed**2*gradient_per_m
        /(2.0*ELECTRON_CHARGE_C*b0)
    )
    return {
        "termination": result.termination.value,
        "observed_drift_m_per_s": float(observed),
        "expected_first_order_drift_m_per_s": expected,
        "relative_error": float(abs(observed-expected)/abs(expected)),
        "rho_over_gradient_scale": speed/omega*gradient_per_m,
    }


def backend_parity(*, device: str = "cpu") -> dict[str, object]:
    status = warp_status()
    if not status.import_available or (device.startswith("cuda") and not status.cuda_available):
        return {"status": "not_evaluated", "reason": status.reason, "device": device}
    velocities = np.array([[1.0e6, 2.0e6, 3.0e5], [-2.5e6, 4.0e5, 1.0e5]])
    electric = np.array([1.0e3, -2.0e3, 0.0])
    magnetic = np.array([0.01, -0.02, 0.04])
    dt = 1.0e-13
    reference = np.vstack([
        relativistic_boris_push(value, electric, magnetic, dt) for value in velocities
    ])
    candidate = warp_boris_push_batch(velocities, electric, magnetic, dt, device=device)
    difference = np.abs(candidate-reference)
    return {
        "status": "evaluated", "device": device,
        "maximum_absolute_velocity_difference_m_per_s": float(np.max(difference)),
        "maximum_relative_velocity_difference": float(np.max(difference)/np.max(np.abs(reference))),
    }
