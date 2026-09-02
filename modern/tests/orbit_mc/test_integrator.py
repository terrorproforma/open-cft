from __future__ import annotations

from math import pi

import numpy as np
import pytest

from cft_revival.orbit_mc import (
    AnalyticField,
    ElectronLaunch,
    OrbitConfig,
    OrbitValidationError,
    Termination,
    build_launch_ensemble,
    integrate_orbit,
)
from cft_revival.orbit_mc.verification import (
    analytic_magnetic_bottle,
    grad_b_drift_ordering,
    timestep_convergence,
    uniform_b_helix,
    uniform_e_acceleration,
    wall_event_accuracy,
)


def test_uniform_b_helix_energy_phase_and_complete_cycle_accounting() -> None:
    report = uniform_b_helix(128, 3)
    assert report["position_error_m"] < 6.0e-6
    assert report["velocity_error_m_per_s"] < 2.0e4
    assert report["relative_energy_error"] < 2.0e-13
    assert report["phase_error_rad"] < 2.0e-12
    assert report["complete_gyrocycles"] >= 3.0


def test_uniform_e_acceleration_matches_relativistic_momentum_solution() -> None:
    report = uniform_e_acceleration()
    assert report["relative_momentum_error"] < 1.0e-12


def test_analytic_magnetic_bottle_reflects_near_first_invariant_prediction() -> None:
    report = analytic_magnetic_bottle()
    assert report["termination"] == Termination.REFLECTED.value
    assert report["relative_error"] < 0.03
    assert report["mu_relative_variation"] < 0.02
    # Reflection detection is unchanged: the chord root is converged to 1e-9 m/s.
    assert abs(report["chord_root_parallel_velocity_m_per_s"]) < 1.0e-9
    # v1.6: the event velocity is the Boris state at the root fraction; its
    # parallel component is bounded by the chord/arc sagitta |v| theta^2 / 8
    # and the pure-B event velocity conserves energy to roundoff.
    assert abs(report["final_parallel_velocity_m_per_s"]) <= (
        report["event_velocity_parallel_bound_m_per_s"]
    )
    assert report["relative_energy_error"] <= 1.0e-12


def test_wall_event_is_first_intersection_and_exact_for_linear_segment() -> None:
    report = wall_event_accuracy()
    assert report["termination"] == Termination.WALL_HIT.value
    assert report["fraction_error"] == 0.0
    assert report["endpoint_error_m"] == 0.0


def test_n_2n_4n_helix_convergence_is_second_order() -> None:
    report = timestep_convergence()
    assert min(report["observed_orders"]) > 1.8


def test_grad_b_drift_has_guiding_centre_sign_and_ordering() -> None:
    report = grad_b_drift_ordering()
    assert report["rho_over_gradient_scale"] < 0.01
    assert report["observed_drift_m_per_s"] < 0.0
    assert report["expected_first_order_drift_m_per_s"] < 0.0
    assert report["relative_error"] < 0.08


def test_gyrophase_symmetry_and_physical_timeout_not_gyro_cap() -> None:
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.05]), None, 0.05)
    launches = build_launch_ensemble(
        ensemble_id="phase-symmetry",
        energies_ev=(20.0,),
        pitch_angles_rad=(pi/2,),
        positions=(("axis", (0.0, 0.0, 0.0)),),
        directions=(1,),
        gyrophase_count=8,
    )
    gyroperiod = 2*pi*9.1093837139e-31/(1.602176634e-19*0.05)
    config = OrbitConfig(
        0.02, -0.1, 0.1, 0.03, -0.2, 0.2,
        7.5*gyroperiod, 1.0, max_steps=4000, max_rotation_rad=0.1,
    )
    results = [integrate_orbit(launch, field, config) for launch in launches]
    assert {result.termination for result in results} == {Termination.TIME_TIMEOUT}
    assert min(result.complete_gyrocycles for result in results) >= 7
    radii = [np.hypot(result.final_position_m[0], result.final_position_m[1]) for result in results]
    assert max(radii)-min(radii) < 2.0e-12


def test_invalid_and_field_failure_taxonomy() -> None:
    with pytest.raises(OrbitValidationError):
        ElectronLaunch("bad", 0, float("nan"), 0.1, (0.0, 0.0, 0.0), 1, 0.0, "x")
    launch = ElectronLaunch("field-failure", 0, 10.0, 0.2, (0.0, 0.0, 0.0), 1, 0.0, "x")
    field = AnalyticField(
        lambda position: np.array([0.0, 0.0, 0.1]) if position[0] == 0.0 else np.full(3, np.nan),
        None, 0.1,
    )
    config = OrbitConfig(1.0, -1.0, 1.0, 2.0, -2.0, 2.0, 1.0e-8, 1.0)
    result = integrate_orbit(launch, field, config)
    assert result.termination is Termination.FIELD_FAILURE

    valid_field = AnalyticField(
        lambda _position: np.array([0.0, 0.0, 0.1]),
        lambda _position, _time: np.array([1.0e20, 0.0, 0.0]),
        0.1,
    )
    guarded = OrbitConfig(
        1.0, -1.0, 1.0, 2.0, -2.0, 2.0, 1.0e-8, 1.0,
        fixed_dt_s=1.0e-13, maximum_gamma=2.0,
    )
    assert integrate_orbit(launch, valid_field, guarded).termination is Termination.EXTREME_RELATIVITY

    nonfinite = integrate_orbit(
        launch,
        AnalyticField(lambda _position: np.array([0.0, 0.0, 0.1]), None, 0.1),
        guarded,
        velocity_pusher=lambda *_args: np.full(3, np.nan),
    )
    assert nonfinite.termination is Termination.NONFINITE_STATE
