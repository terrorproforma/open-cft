from __future__ import annotations

from math import pi

import numpy as np
import pytest

from cft_revival.orbit_mc import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    EV_J,
    LIGHT_SPEED_M_PER_S,
    AnalyticField,
    ElectronLaunch,
    OrbitConfig,
    OrbitNumericsError,
    OrbitValidationError,
    PsiBicubicField,
    Termination,
    integrate_orbit,
    preflight_campaign,
    varying_e_convergence,
    wilson_interval,
)


def _uniform_material(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, "plasma", dtype=object)


def test_deadline_and_path_events_outrank_later_wall_crossing() -> None:
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 1.0e-6]), None, 1.0e-6)
    launch = ElectronLaunch(
        "ordered-events", 0, 10.0, pi/2, (0.0, 0.0, 0.0), 1, 0.0, "axis"
    )
    dt = 1.0e-9
    time_limited = OrbitConfig(
        1.4e-3, -1.0, 1.0, 0.01, -2.0, 2.0,
        0.5*dt, 1.0, max_steps=2, fixed_dt_s=dt,
    )
    result = integrate_orbit(launch, field, time_limited)
    assert result.termination is Termination.TIME_TIMEOUT
    assert result.elapsed_time_s == pytest.approx(0.5*dt, abs=1.0e-24)
    assert result.wall_endpoint_m is None

    path_limited = OrbitConfig(
        1.4e-3, -1.0, 1.0, 0.01, -2.0, 2.0,
        2.0*dt, 4.0e-4, max_steps=2, fixed_dt_s=dt,
    )
    result = integrate_orbit(launch, field, path_limited)
    assert result.termination is Termination.PATH_TIMEOUT
    assert result.path_length_m == pytest.approx(4.0e-4, abs=1.0e-18)
    assert result.wall_endpoint_m is None


def test_outside_launch_is_typed_invalid_and_cannot_cross_into_wall() -> None:
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    launch = ElectronLaunch(
        "outside", 0, 10.0, pi/2, (0.002, 0.0, 0.0), 1, pi/2, "outside"
    )
    config = OrbitConfig(0.001, -1.0, 1.0, 0.01, -2.0, 2.0, 1.0e-8, 1.0)
    result = integrate_orbit(launch, field, config)
    assert result.termination is Termination.INITIAL_STATE_INVALID
    assert result.wall_endpoint_m is None
    assert result.steps == 0


def test_relativistic_phase_one_gamma_two_orbit_is_one_cycle() -> None:
    gamma = 2.0
    energy_ev = (gamma - 1.0) * ELECTRON_MASS_KG * LIGHT_SPEED_M_PER_S**2 / EV_J
    b_t = 0.02
    period = 2.0*pi*gamma*ELECTRON_MASS_KG/(abs(ELECTRON_CHARGE_C)*b_t)
    steps = 128
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, b_t]), None, b_t)
    launch = ElectronLaunch(
        "gamma-two", 0, energy_ev, pi/2, (0.0, 0.0, 0.0), 1, 0.0, "axis"
    )
    config = OrbitConfig(
        1.0, -10.0, 10.0, 2.0, -20.0, 20.0, period, 100.0,
        max_steps=steps, max_rotation_rad=0.1, maximum_gamma=2.1,
        fixed_dt_s=period/steps,
    )
    result = integrate_orbit(launch, field, config)
    assert result.accumulated_gyro_phase_rad == pytest.approx(2.0*pi, abs=2.0e-12)
    assert result.complete_gyrocycles == 1


def test_varying_e_midpoint_scheme_is_second_order() -> None:
    report = varying_e_convergence()
    assert min(report["observed_orders"]) > 1.8


def test_axis_regular_quartic_has_matching_cartesian_one_sided_slopes() -> None:
    r = np.linspace(0.0, 0.2, 9)
    z = np.linspace(-0.2, 0.2, 9)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = rr**2 * (0.3 + 0.7*rr**2) * (1.0 + 0.4*zz)
    field = PsiBicubicField(
        r, z, psi, material_id=_uniform_material(psi.shape),
        plasma_material_id="plasma",
    )
    epsilon = 1.0e-7
    positive = field.magnetic_cartesian(np.array([epsilon, 0.0, 0.03]))
    negative = field.magnetic_cartesian(np.array([-epsilon, 0.0, 0.03]))
    assert positive[0]/epsilon == pytest.approx(
        negative[0]/(-epsilon), rel=0.0, abs=2.0e-9
    )
    assert positive[2] == pytest.approx(negative[2], rel=0.0, abs=2.0e-12)


def test_material_interface_cells_are_quarantined() -> None:
    r = np.linspace(0.0, 0.2, 5)
    z = np.linspace(-0.2, 0.2, 5)
    rr, _ = np.meshgrid(r, z, indexing="ij")
    psi = 0.2*rr**2
    material = _uniform_material(psi.shape)
    material[:, 3:] = "dielectric"
    field = PsiBicubicField(
        r, z, psi, material_id=material, plasma_material_id="plasma"
    )
    with pytest.raises(OrbitNumericsError, match="interface|non-plasma"):
        field.field_cylindrical(0.05, 0.11)


def test_certified_bound_dominates_dense_samples_and_rejects_bad_references() -> None:
    r = np.linspace(0.0, 0.3, 7)
    z = np.linspace(-0.3, 0.3, 9)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = rr**2*(0.2 + 0.9*rr**2)*(1.0 + 0.5*zz + 0.8*zz**2)
    material = _uniform_material(psi.shape)
    field = PsiBicubicField(
        r, z, psi, material_id=material, plasma_material_id="plasma"
    )
    dense_max = max(
        np.hypot(*field.field_cylindrical(float(radius), float(axial)))
        for radius in np.linspace(0.0, 0.3, 101)
        for axial in np.linspace(-0.3, 0.3, 151)
    )
    assert dense_max <= field.certified_max_b_t
    raised = PsiBicubicField(
        r, z, psi, material_id=material, plasma_material_id="plasma",
        reference_max_b_t=2.0*field.certified_max_b_t,
    )
    assert raised.max_b_t == 2.0*field.certified_max_b_t

    br = np.zeros_like(psi)
    bz = np.full_like(psi, 10.0)
    with pytest.raises(OrbitValidationError, match="inconsistent"):
        PsiBicubicField(
            r, z, psi, material_id=material, plasma_material_id="plasma",
            reference_br_t=br, reference_bz_t=bz,
        )
    with pytest.raises(OrbitValidationError, match="underdeclared"):
        PsiBicubicField(
            r, z, psi, material_id=material, plasma_material_id="plasma",
            reference_br_t=np.zeros_like(psi),
            reference_bz_t=np.ones_like(psi),
            reference_max_b_t=0.5,
            reference_consistency_relative_tolerance=100.0,
        )
    with pytest.raises(OrbitValidationError, match="tolerance"):
        PsiBicubicField(
            r, z, psi, material_id=material, plasma_material_id="plasma",
            reference_consistency_relative_tolerance=True,
        )


def test_runtime_rejects_underdeclared_analytic_maximum() -> None:
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.02]), None, 0.01)
    launch = ElectronLaunch(
        "underdeclared", 0, 10.0, 0.2, (0.0, 0.0, 0.0), 1, 0.0, "axis"
    )
    config = OrbitConfig(1.0, -1.0, 1.0, 2.0, -2.0, 2.0, 1.0e-8, 1.0)
    result = integrate_orbit(launch, field, config)
    assert result.termination is Termination.FIELD_FAILURE
    assert "exceeds declared" in result.reason


@pytest.mark.parametrize("z", [0.0, -1.0, float("nan"), float("inf"), True])
def test_wilson_z_must_be_finite_positive(z) -> None:
    with pytest.raises(OrbitValidationError):
        wilson_interval(1, 2, z=z)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_time_s": True},
        {"max_path_m": "1.0"},
        {"max_rotation_rad": float("nan")},
        {"event_tolerance_m": -1.0},
        {"maximum_gamma": float("inf")},
        {"max_steps": 1.5},
        {"fixed_dt_s": True},
        {"fixed_dt_s": "1e-9"},
    ],
)
def test_all_orbit_policy_scalars_and_types_fail_closed(overrides) -> None:
    values = {
        "wall_radius_m": 1.0,
        "wall_z_min_m": -1.0,
        "wall_z_max_m": 1.0,
        "domain_radius_m": 2.0,
        "domain_z_min_m": -2.0,
        "domain_z_max_m": 2.0,
        "max_time_s": 1.0,
        "max_path_m": 1.0,
    }
    values.update(overrides)
    with pytest.raises(OrbitValidationError):
        OrbitConfig(**values)


class _BoundaryGuardField:
    max_b_t = 1.0e-6

    def __init__(self, radius_limit_m: float) -> None:
        self.radius_limit_m = radius_limit_m
        self.queries = 0

    def magnetic_cartesian(self, position_m: np.ndarray) -> np.ndarray:
        self.queries += 1
        if np.hypot(position_m[0], position_m[1]) >= self.radius_limit_m:
            raise OrbitNumericsError("query crossed guarded boundary")
        return np.array([0.0, 0.0, self.max_b_t])

    def electric_cartesian(
        self, position_m: np.ndarray, _time_s: float
    ) -> np.ndarray:
        if np.hypot(position_m[0], position_m[1]) >= self.radius_limit_m:
            raise OrbitNumericsError("query crossed guarded boundary")
        return np.zeros(3)


@pytest.mark.parametrize(
    ("kind", "start_radius", "start_z", "wall_radius", "domain_radius"),
    [
        (Termination.WALL_HIT, np.nextafter(1.0, 0.0), 0.0, 1.0, 2.0),
        (
            Termination.DOMAIN_ESCAPE,
            np.nextafter(2.0, 0.0),
            0.2,
            1.0,
            2.0,
        ),
    ],
)
def test_synthetic_v3_near_boundary_geometry_snaps_without_outside_query(
    kind: Termination,
    start_radius: float,
    start_z: float,
    wall_radius: float,
    domain_radius: float,
) -> None:
    boundary = wall_radius if kind is Termination.WALL_HIT else domain_radius
    field = _BoundaryGuardField(boundary)
    launch = ElectronLaunch(
        f"close-{kind.value}",
        0,
        10.0,
        pi / 2,
        (start_radius, 0.0, start_z),
        1,
        1.5 * pi,
        "near-boundary",
    )
    config = OrbitConfig(
        wall_radius,
        -0.1,
        0.1,
        domain_radius,
        -1.0,
        1.0,
        1.0e-6,
        10.0,
        max_steps=200_000,
        event_tolerance_m=1.0e-9,
        fixed_dt_s=1.0e-9,
    )
    result = integrate_orbit(launch, field, config)
    assert result.termination is kind
    assert result.termination is not Termination.FIELD_FAILURE
    assert result.steps == 1
    assert result.steps < 200_000
    assert result.event_witness["event_fraction"] == 0.0
    assert result.event_witness["event_resolution"] == (
        "tolerance_close_fraction_zero"
    )
    assert result.event_witness["step_dt_s"] == config.fixed_dt_s
    assert np.hypot(*result.final_position_m[:2]) == pytest.approx(
        boundary, abs=0.0
    )


def test_boundary_launch_remains_initial_state_invalid() -> None:
    field = AnalyticField(
        lambda _x: np.array([0.0, 0.0, 1.0e-6]), None, 1.0e-6
    )
    launch = ElectronLaunch(
        "on-wall", 0, 10.0, pi / 2, (1.0, 0.0, 0.0), 1, 1.5*pi, "wall"
    )
    config = OrbitConfig(
        1.0, -0.1, 0.1, 2.0, -1.0, 1.0, 1.0e-6, 10.0
    )
    result = integrate_orbit(launch, field, config)
    assert result.termination is Termination.INITIAL_STATE_INVALID
    assert result.steps == 0
    assert result.event_witness["event_resolution"] == "failure"


def test_campaign_preflight_exposes_launch_and_timestep_checks() -> None:
    field = AnalyticField(
        lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1
    )
    launch = ElectronLaunch(
        "preflight", 0, 10.0, 0.2, (0.0, 0.0, 0.0), 1, 0.0, "axis"
    )
    config = OrbitConfig(
        1.0, -1.0, 1.0, 2.0, -2.0, 2.0, 1.0e-8, 1.0
    )
    report = preflight_campaign((launch,), field, config)
    assert report["status"] == "passed"
    assert report["launch_count"] == 1
    with pytest.raises(OrbitValidationError, match="outside"):
        preflight_campaign(
            (
                ElectronLaunch(
                    "outside-preflight",
                    1,
                    10.0,
                    0.2,
                    (2.0, 0.0, 0.0),
                    1,
                    0.0,
                    "outside",
                ),
            ),
            field,
            config,
        )


def test_corrected_zero_progress_stops_before_reflection_field_query() -> None:
    class CountingField:
        max_b_t = 0.1

        def __init__(self) -> None:
            self.magnetic_queries = 0

        def magnetic_cartesian(self, _position_m: np.ndarray) -> np.ndarray:
            self.magnetic_queries += 1
            return np.array([0.0, 0.0, self.max_b_t])

        def electric_cartesian(
            self, _position_m: np.ndarray, _time_s: float
        ) -> np.ndarray:
            return np.zeros(3)

    field = CountingField()
    launch = ElectronLaunch(
        "zero-progress",
        0,
        10.0,
        pi / 2,
        (0.0, 0.0, 0.0),
        1,
        0.0,
        "axis",
    )
    config = OrbitConfig(
        1.0,
        -1.0,
        1.0,
        2.0,
        -2.0,
        2.0,
        1.0e-6,
        10.0,
        fixed_dt_s=1.0e-12,
    )
    result = integrate_orbit(
        launch,
        field,
        config,
        velocity_pusher=lambda velocity, *_args: -velocity,
    )
    assert result.termination is Termination.FIELD_FAILURE
    assert result.event_witness["condition"] == (
        "zero_progress_corrected_segment"
    )
    # launch, step start, and midpoint only; no reflection/end query occurs.
    assert field.magnetic_queries == 3
