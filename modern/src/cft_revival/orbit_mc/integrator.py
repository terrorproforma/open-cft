"""Relativistically guarded CPU reference full-orbit integrator."""

from __future__ import annotations

from math import hypot, isfinite, pi, sqrt
from typing import Callable, Iterable

import numpy as np

from .models import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    EV_J,
    LIGHT_SPEED_M_PER_S,
    AxisymmetricField,
    ElectronLaunch,
    GyroAverage,
    OrbitConfig,
    OrbitNumericsError,
    OrbitResult,
    OrbitValidationError,
    Termination,
    kinetic_energy_j_from_velocity,
    velocity_from_energy_ev,
)


def launch_velocity(launch: ElectronLaunch, field: AxisymmetricField) -> np.ndarray:
    """Build velocity from energy, local B, pitch, direction, and gyrophase."""

    position = np.asarray(launch.position_m, dtype=np.float64)
    magnetic = np.asarray(field.magnetic_cartesian(position), dtype=np.float64)
    magnitude = float(np.linalg.norm(magnetic))
    if not isfinite(magnitude) or magnitude <= 0.0:
        raise OrbitNumericsError("launch requires a finite nonzero magnetic field")
    b_hat = magnetic / magnitude
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, b_hat))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(b_hat, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b_hat, e1)
    speed = velocity_from_energy_ev(launch.kinetic_energy_ev)
    parallel = launch.parallel_direction * speed * np.cos(launch.pitch_angle_rad)
    perpendicular = speed * np.sin(launch.pitch_angle_rad)
    return (
        parallel * b_hat
        + perpendicular
        * (np.cos(launch.gyrophase_rad) * e1 + np.sin(launch.gyrophase_rad) * e2)
    )


def relativistic_boris_push(
    velocity_m_per_s: np.ndarray,
    electric_v_per_m: np.ndarray,
    magnetic_t: np.ndarray,
    dt_s: float,
    *,
    charge_c: float = ELECTRON_CHARGE_C,
    mass_kg: float = ELECTRON_MASS_KG,
) -> np.ndarray:
    """Relativistic momentum Boris push; suitable while the gamma guard passes."""

    velocity = np.asarray(velocity_m_per_s, dtype=np.float64)
    speed2 = float(np.dot(velocity, velocity))
    if not np.isfinite(velocity).all() or speed2 >= LIGHT_SPEED_M_PER_S**2:
        raise OrbitNumericsError("Boris input velocity is nonfinite or superluminal")
    gamma = 1.0 / sqrt(1.0 - speed2 / LIGHT_SPEED_M_PER_S**2)
    u = gamma * velocity
    u_minus = u + (charge_c * dt_s / (2.0 * mass_kg)) * electric_v_per_m
    gamma_minus = sqrt(1.0 + float(np.dot(u_minus, u_minus)) / LIGHT_SPEED_M_PER_S**2)
    t = charge_c * dt_s * magnetic_t / (2.0 * mass_kg * gamma_minus)
    s = 2.0 * t / (1.0 + float(np.dot(t, t)))
    u_prime = u_minus + np.cross(u_minus, t)
    u_plus = u_minus + np.cross(u_prime, s)
    u_new = u_plus + (charge_c * dt_s / (2.0 * mass_kg)) * electric_v_per_m
    gamma_new = sqrt(1.0 + float(np.dot(u_new, u_new)) / LIGHT_SPEED_M_PER_S**2)
    result = u_new / gamma_new
    if not np.isfinite(result).all():
        raise OrbitNumericsError("Boris push produced a nonfinite state")
    return result


def _first_cylinder_crossing(
    start: np.ndarray, end: np.ndarray, radius_m: float
) -> float | None:
    delta = end - start
    a = float(delta[0] ** 2 + delta[1] ** 2)
    b = 2.0 * float(start[0] * delta[0] + start[1] * delta[1])
    c = float(start[0] ** 2 + start[1] ** 2 - radius_m**2)
    if a == 0.0:
        return None
    discriminant = b*b - 4.0*a*c
    if discriminant < 0.0:
        return None
    roots = sorted(
        root for root in ((-b - sqrt(discriminant))/(2*a), (-b + sqrt(discriminant))/(2*a))
        if 0.0 <= root <= 1.0
    )
    return roots[0] if roots else None


def _segment_event(
    start: np.ndarray, end: np.ndarray, config: OrbitConfig
) -> tuple[Termination, np.ndarray, float] | None:
    candidates: list[tuple[float, int, Termination, np.ndarray]] = []
    wall_fraction = _first_cylinder_crossing(start, end, config.wall_radius_m)
    if wall_fraction is not None:
        point = start + wall_fraction * (end - start)
        if config.wall_z_min_m - config.event_tolerance_m <= point[2] <= config.wall_z_max_m + config.event_tolerance_m:
            candidates.append((wall_fraction, 0, Termination.WALL_HIT, point))
    domain_fraction = _first_cylinder_crossing(start, end, config.domain_radius_m)
    if domain_fraction is not None:
        point = start + domain_fraction * (end - start)
        candidates.append((domain_fraction, 1, Termination.DOMAIN_ESCAPE, point))
    dz = float(end[2] - start[2])
    if dz != 0.0:
        for plane in (config.domain_z_min_m, config.domain_z_max_m):
            fraction = (plane - float(start[2])) / dz
            if 0.0 <= fraction <= 1.0:
                point = start + fraction * (end - start)
                if hypot(float(point[0]), float(point[1])) <= config.domain_radius_m:
                    candidates.append((fraction, 1, Termination.DOMAIN_ESCAPE, point))
    if not candidates:
        return None
    fraction, _, termination, point = min(candidates, key=lambda item: (item[0], item[1]))
    return termination, point, fraction


def _geometry_event_candidates(
    start: np.ndarray,
    end: np.ndarray,
    config: OrbitConfig,
    *,
    attempted_displacement: np.ndarray | None = None,
) -> list[tuple[float, int, Termination, np.ndarray, str]]:
    candidates: list[tuple[float, int, Termination, np.ndarray, str]] = []
    delta = end - start
    direction = (
        delta
        if attempted_displacement is None
        else np.asarray(attempted_displacement, dtype=np.float64)
    )
    start_radius = hypot(float(start[0]), float(start[1]))
    scale = max(
        1.0,
        config.wall_radius_m,
        config.domain_radius_m,
        abs(config.domain_z_min_m),
        abs(config.domain_z_max_m),
    )
    close_tolerance = max(
        config.event_tolerance_m,
        256.0 * np.finfo(float).eps * scale,
    )

    def radial_outward() -> bool:
        if start_radius <= 0.0:
            return False
        return (
            float(start[0] * direction[0] + start[1] * direction[1])
            / start_radius
            > 0.0
        )

    def radial_snap(radius_m: float) -> np.ndarray:
        point = start.copy()
        point[0] = radius_m * float(start[0]) / start_radius
        point[1] = radius_m * float(start[1]) / start_radius
        return point

    wall_close = (
        0.0 < config.wall_radius_m - start_radius <= close_tolerance
        and radial_outward()
        and config.wall_z_min_m - close_tolerance
        <= float(start[2])
        <= config.wall_z_max_m + close_tolerance
    )
    if wall_close:
        candidates.append(
            (
                0.0,
                2,
                Termination.WALL_HIT,
                radial_snap(config.wall_radius_m),
                "tolerance_close_wall_radial",
            )
        )
    wall_fraction = _first_cylinder_crossing(start, end, config.wall_radius_m)
    if wall_fraction is not None and not (wall_close and wall_fraction == 0.0):
        point = start + wall_fraction * (end - start)
        if (
            config.wall_z_min_m - config.event_tolerance_m
            <= point[2]
            <= config.wall_z_max_m + config.event_tolerance_m
        ):
            candidates.append(
                (
                    wall_fraction,
                    2,
                    Termination.WALL_HIT,
                    point,
                    "interpolated_wall_radial",
                )
            )
    domain_radial_close = (
        0.0 < config.domain_radius_m - start_radius <= close_tolerance
        and radial_outward()
    )
    if domain_radial_close:
        candidates.append(
            (
                0.0,
                4,
                Termination.DOMAIN_ESCAPE,
                radial_snap(config.domain_radius_m),
                "tolerance_close_domain_radial",
            )
        )
    domain_fraction = _first_cylinder_crossing(
        start, end, config.domain_radius_m
    )
    if domain_fraction is not None and not (
        domain_radial_close and domain_fraction == 0.0
    ):
        point = start + domain_fraction * (end - start)
        candidates.append(
            (
                domain_fraction,
                4,
                Termination.DOMAIN_ESCAPE,
                point,
                "interpolated_domain_radial",
            )
        )
    for plane, condition, outward in (
        (
            config.domain_z_min_m,
            "tolerance_close_domain_z_min",
            float(direction[2]) < 0.0,
        ),
        (
            config.domain_z_max_m,
            "tolerance_close_domain_z_max",
            float(direction[2]) > 0.0,
        ),
    ):
        distance = (
            float(start[2]) - plane
            if plane == config.domain_z_min_m
            else plane - float(start[2])
        )
        if (
            0.0 < distance <= close_tolerance
            and outward
            and start_radius <= config.domain_radius_m + close_tolerance
        ):
            point = start.copy()
            point[2] = plane
            candidates.append(
                (
                    0.0,
                    4,
                    Termination.DOMAIN_ESCAPE,
                    point,
                    condition,
                )
            )
    dz = float(end[2] - start[2])
    if dz != 0.0:
        for plane in (config.domain_z_min_m, config.domain_z_max_m):
            fraction = (plane - float(start[2])) / dz
            if 0.0 <= fraction <= 1.0:
                point = start + fraction * (end - start)
                if hypot(float(point[0]), float(point[1])) <= config.domain_radius_m:
                    candidates.append(
                        (
                            fraction,
                            4,
                            Termination.DOMAIN_ESCAPE,
                            point,
                            "interpolated_domain_axial",
                        )
                    )
    return candidates


def _witness_config(config: OrbitConfig) -> dict[str, object]:
    return {
        "wall_radius_m": config.wall_radius_m,
        "wall_z_min_m": config.wall_z_min_m,
        "wall_z_max_m": config.wall_z_max_m,
        "domain_radius_m": config.domain_radius_m,
        "domain_z_min_m": config.domain_z_min_m,
        "domain_z_max_m": config.domain_z_max_m,
        "max_time_s": config.max_time_s,
        "max_path_m": config.max_path_m,
        "max_steps": config.max_steps,
        "max_rotation_rad": config.max_rotation_rad,
        "event_tolerance_m": config.event_tolerance_m,
        "maximum_gamma": config.maximum_gamma,
        "fixed_dt_s": config.fixed_dt_s,
    }


def _event_velocity(
    push: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    electric_midpoint: np.ndarray,
    magnetic_midpoint: np.ndarray,
    step_dt: float,
    fraction: float,
) -> np.ndarray:
    """Velocity at fraction ``fraction`` of a step (v1.6 contract).

    * ``fraction == 0`` returns the step-start velocity bit-for-bit (the
      tolerance-close snap path; no time elapses).
    * ``fraction == 1`` returns the full-step velocity bit-for-bit. The Boris
      push is deterministic so ``push(v0, E, B, 1.0*dt)`` would reproduce it
      anyway, but the special case removes any dependence on pusher internals.
    * interior fractions run one extra push of duration ``fraction*step_dt``
      from the start velocity with the same midpoint fields the full step used.

    The validator (``artifacts._validate_event_witness``) replays exactly this
    function with :func:`relativistic_boris_push`, so any change here is a
    contract change.
    """

    if fraction <= 0.0:
        return np.array(start_velocity, dtype=np.float64, copy=True)
    if fraction >= 1.0:
        return np.array(end_velocity, dtype=np.float64, copy=True)
    return np.asarray(
        push(start_velocity, electric_midpoint, magnetic_midpoint, fraction * step_dt),
        dtype=np.float64,
    )


def _failure_witness(
    kind: Termination,
    config: OrbitConfig,
    position: np.ndarray,
    *,
    step: int,
    elapsed_s: float,
    path_m: float,
    condition: str,
    observed_gamma: float | None = None,
    observed_speed2_over_c2: float | None = None,
) -> dict[str, object]:
    point = tuple(map(float, position))
    return {
        "kind": kind.value,
        "config": _witness_config(config),
        "step_start_position_m": point,
        "step_end_position_m": point,
        "event_position_m": point,
        "step_start_velocity_m_per_s": [0.0, 0.0, 0.0],
        "step_end_velocity_m_per_s": [0.0, 0.0, 0.0],
        "event_velocity_m_per_s": [0.0, 0.0, 0.0],
        "step_magnetic_midpoint_t": [0.0, 0.0, 0.0],
        "step_electric_midpoint_v_per_m": [0.0, 0.0, 0.0],
        "event_fraction": 0.0,
        "event_resolution": "failure",
        "candidate_fractions": {
            "time_timeout": None,
            "path_timeout": None,
            "wall_hit": None,
            "reflected": None,
            "domain_escape": None,
        },
        "reflection_bracket": None,
        "start_elapsed_time_s": elapsed_s,
        "start_path_length_m": path_m,
        "step_dt_s": 0.0,
        "step_segment_length_m": 0.0,
        "step_index": step,
        "condition": condition,
        "observed_gamma": observed_gamma,
        "observed_speed2_over_c2": observed_speed2_over_c2,
        "field_identity_sha256": "0" * 64,
        "config_identity_sha256": "0" * 64,
        "policy_identity_sha256": "0" * 64,
    }


def _initial_failure_result(
    launch: ElectronLaunch,
    config: OrbitConfig,
    backend: str,
    reason: str,
    field: AxisymmetricField,
) -> OrbitResult:
    dt = config.fixed_dt_s
    if dt is None:
        if isfinite(field.max_b_t) and field.max_b_t > 0.0:
            dt = (
                config.max_rotation_rad
                * ELECTRON_MASS_KG
                / (abs(ELECTRON_CHARGE_C) * field.max_b_t)
            )
        else:
            dt = config.max_time_s
    energy = launch.kinetic_energy_ev * EV_J
    return OrbitResult(
        launch.launch_id,
        Termination.INITIAL_STATE_INVALID,
        reason,
        launch.position_m,
        (0.0, 0.0, 0.0),
        None,
        0.0,
        0.0,
        0,
        0.0,
        0,
        (),
        energy,
        energy,
        0.0,
        None,
        0.0,
        0.0,
        config.max_time_s,
        config.max_path_m,
        config.event_tolerance_m,
        dt,
        backend,
        _failure_witness(
            Termination.INITIAL_STATE_INVALID,
            config,
            np.asarray(launch.position_m, dtype=np.float64),
            step=0,
            elapsed_s=0.0,
            path_m=0.0,
            condition=(
                "launch_outside_geometry"
                if reason.startswith("launch lies outside")
                else "invalid_initial_field"
            ),
        ),
    )


def _reflection_fraction(
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    field: AxisymmetricField,
    initial_parallel_sign: int,
) -> tuple[float, float, float, float, float, float] | None:
    def signed_parallel(fraction: float) -> float:
        position = start_position + fraction * (end_position - start_position)
        velocity = start_velocity + fraction * (end_velocity - start_velocity)
        magnetic = np.asarray(field.magnetic_cartesian(position), dtype=np.float64)
        magnitude = float(np.linalg.norm(magnetic))
        if magnitude <= 0.0:
            return float("nan")
        return initial_parallel_sign * float(np.dot(velocity, magnetic / magnitude))

    low_value = signed_parallel(0.0)
    high_value = signed_parallel(1.0)
    if not isfinite(low_value) or not isfinite(high_value):
        return None
    if low_value <= 0.0 or high_value > 0.0:
        return None
    low = 0.0
    high = 1.0
    for _ in range(52):
        middle = 0.5 * (low + high)
        value = signed_parallel(middle)
        if not isfinite(value):
            return None
        if value > 0.0:
            low = middle
        else:
            high = middle
    root = 0.5 * (low + high)
    return (
        root,
        low,
        high,
        signed_parallel(low),
        signed_parallel(high),
        signed_parallel(root),
    )


class _GyroAccumulator:
    def __init__(self) -> None:
        self.phase = 0.0
        self._cycle_progress = 0.0
        self._cycle_weight = 0.0
        self._cycle_phase = 0.0
        self.averages: list[GyroAverage] = []

    def add(self, phase_increment: float, mu_j_per_t: float | None) -> None:
        remaining = phase_increment
        while remaining > 0.0:
            needed = 2.0*pi - self._cycle_progress
            tolerance = 16.0*np.finfo(float).eps*2.0*pi
            if needed <= tolerance:
                self._finish_cycle()
                continue
            portion = min(remaining, needed)
            if mu_j_per_t is not None:
                self._cycle_weight += mu_j_per_t * portion
                self._cycle_phase += portion
            self.phase += portion
            self._cycle_progress += portion
            remaining -= portion
            if self._cycle_progress >= 2.0*pi - tolerance:
                self._finish_cycle()

    def _finish_cycle(self) -> None:
        if self._cycle_phase >= 2.0*pi * (1.0 - 1.0e-12):
            index = len(self.averages)
            self.averages.append(
                GyroAverage(index, 2.0*pi*index, 2.0*pi*(index+1),
                            self._cycle_weight / self._cycle_phase)
            )
        self._cycle_progress = 0.0
        self._cycle_weight = 0.0
        self._cycle_phase = 0.0


def _mu(velocity: np.ndarray, magnetic: np.ndarray) -> float | None:
    magnitude = float(np.linalg.norm(magnetic))
    if magnitude <= np.finfo(float).tiny:
        return None
    speed2 = float(np.dot(velocity, velocity))
    gamma = 1.0 / sqrt(1.0 - speed2 / LIGHT_SPEED_M_PER_S**2)
    momentum = ELECTRON_MASS_KG * gamma * velocity
    b_hat = magnetic / magnitude
    perpendicular = momentum - float(np.dot(momentum, b_hat)) * b_hat
    return float(np.dot(perpendicular, perpendicular) / (2.0 * ELECTRON_MASS_KG * magnitude))


def preflight_campaign(
    launches: Iterable[ElectronLaunch],
    field: AxisymmetricField,
    config: OrbitConfig,
) -> dict[str, object]:
    """Fail closed before a campaign if launches or timestep are not runnable."""

    ordered = sorted(tuple(launches), key=lambda item: item.launch_id)
    if not ordered or len({item.launch_id for item in ordered}) != len(ordered):
        raise OrbitValidationError(
            "campaign preflight requires nonempty unique launches"
        )
    if not isfinite(field.max_b_t) or field.max_b_t <= 0.0:
        raise OrbitValidationError(
            "campaign preflight requires a finite positive field bound"
        )
    magnetic_dt = (
        config.max_rotation_rad
        * ELECTRON_MASS_KG
        / (abs(ELECTRON_CHARGE_C) * field.max_b_t)
    )
    dt = config.fixed_dt_s if config.fixed_dt_s is not None else magnetic_dt
    if (
        not isfinite(dt)
        or dt <= 0.0
        or dt > magnetic_dt * (1.0 + 1.0e-14)
    ):
        raise OrbitValidationError(
            "campaign preflight timestep violates the max-B rotation bound"
        )
    maximum_launch_b_t = 0.0
    for launch in ordered:
        position = np.asarray(launch.position_m, dtype=np.float64)
        radius = hypot(float(position[0]), float(position[1]))
        if (
            radius >= config.domain_radius_m
            or not config.domain_z_min_m
            < position[2]
            < config.domain_z_max_m
            or (
                config.wall_z_min_m
                <= position[2]
                <= config.wall_z_max_m
                and radius >= config.wall_radius_m
            )
        ):
            raise OrbitValidationError(
                f"campaign preflight launch {launch.launch_id} is outside "
                "the valid plasma domain"
            )
        try:
            magnetic = np.asarray(
                field.magnetic_cartesian(position), dtype=np.float64
            )
            launch_velocity(launch, field)
        except Exception as error:
            raise OrbitValidationError(
                f"campaign preflight launch {launch.launch_id} field/state "
                f"is invalid: {type(error).__name__}: {error}"
            ) from error
        magnitude = float(np.linalg.norm(magnetic))
        if (
            magnetic.shape != (3,)
            or not np.isfinite(magnetic).all()
            or magnitude <= 0.0
            or magnitude
            > field.max_b_t * (1.0 + 64.0 * np.finfo(float).eps)
        ):
            raise OrbitValidationError(
                f"campaign preflight launch {launch.launch_id} violates "
                "the field contract"
            )
        maximum_launch_b_t = max(maximum_launch_b_t, magnitude)
    return {
        "status": "passed",
        "launch_count": len(ordered),
        "timestep_s": dt,
        "maximum_declared_b_t": float(field.max_b_t),
        "maximum_launch_b_t": maximum_launch_b_t,
    }


def integrate_orbit(
    launch: ElectronLaunch,
    field: AxisymmetricField,
    config: OrbitConfig,
    *,
    backend: str = "numpy-cpu-relativistic-boris",
    velocity_pusher: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray] | None = None,
) -> OrbitResult:
    """Second-order midpoint-field orbit integration with earliest-event ordering."""

    position = np.asarray(launch.position_m, dtype=np.float64)
    radius = hypot(float(position[0]), float(position[1]))
    inside_domain = (
        radius < config.domain_radius_m
        and config.domain_z_min_m < position[2] < config.domain_z_max_m
    )
    inside_wall_bore = not (
        config.wall_z_min_m <= position[2] <= config.wall_z_max_m
        and radius >= config.wall_radius_m
    )
    if not inside_domain or not inside_wall_bore:
        return _initial_failure_result(
            launch,
            config,
            backend,
            "launch lies outside the valid plasma domain",
            field,
        )
    try:
        velocity = launch_velocity(launch, field)
    except Exception as error:
        return _initial_failure_result(
            launch,
            config,
            backend,
            f"initial plasma field/state is invalid: {type(error).__name__}: {error}",
            field,
        )
    initial_position = position.copy()
    initial_energy = kinetic_energy_j_from_velocity(velocity)
    if field.max_b_t > 0.0:
        magnetic_dt = config.max_rotation_rad * ELECTRON_MASS_KG / (
            abs(ELECTRON_CHARGE_C) * field.max_b_t
        )
    else:
        magnetic_dt = float("inf")
    dt = config.fixed_dt_s if config.fixed_dt_s is not None else magnetic_dt
    if not isfinite(dt) or dt <= 0.0 or dt > magnetic_dt * (1.0 + 1.0e-14):
        raise OrbitNumericsError("timestep violates the declared max-B rotation bound")

    time_s = 0.0
    path_m = 0.0
    energy_error = 0.0
    maximum_b_seen = 0.0
    mu_values: list[float] = []
    gyro = _GyroAccumulator()
    wall_endpoint: tuple[float, float, float] | None = None
    termination = Termination.STEP_LIMIT
    reason = "maximum deterministic step count reached"
    initial_parallel_sign = launch.parallel_direction
    steps = 0
    event_witness = _failure_witness(
        Termination.STEP_LIMIT,
        config,
        position,
        step=0,
        elapsed_s=0.0,
        path_m=0.0,
        condition="maximum_steps_reached",
    )
    push = relativistic_boris_push if velocity_pusher is None else velocity_pusher

    for step in range(1, config.max_steps + 1):
        steps = step
        try:
            magnetic_start = np.asarray(
                field.magnetic_cartesian(position), dtype=np.float64
            )
            electric_start = np.asarray(
                field.electric_cartesian(position, time_s), dtype=np.float64
            )
            if (
                magnetic_start.shape != (3,)
                or electric_start.shape != (3,)
                or not np.isfinite(magnetic_start).all()
                or not np.isfinite(electric_start).all()
            ):
                raise OrbitNumericsError("field evaluation returned an invalid vector")
            actual_b_t = float(np.linalg.norm(magnetic_start))
            maximum_b_seen = max(maximum_b_seen, actual_b_t)
            if actual_b_t > field.max_b_t * (1.0 + 64.0 * np.finfo(float).eps):
                raise OrbitNumericsError("runtime field exceeds declared/certified maximum")
            current_speed2 = float(np.dot(velocity, velocity))
            current_gamma = 1.0 / sqrt(
                1.0 - current_speed2 / LIGHT_SPEED_M_PER_S**2
            )
            actual_rotation = (
                abs(ELECTRON_CHARGE_C) * actual_b_t * dt
                / (current_gamma * ELECTRON_MASS_KG)
            )
            if actual_rotation > config.max_rotation_rad * (
                1.0 + 64.0 * np.finfo(float).eps
            ):
                raise OrbitNumericsError("runtime cyclotron rotation bound exceeded")
            predicted_velocity = push(
                velocity, electric_start, magnetic_start, dt
            )
            predicted_position = (
                position + 0.5 * (velocity + predicted_velocity) * dt
            )
            attempted_displacement = (
                0.5 * (velocity + predicted_velocity) * dt
            )
            predicted_segment = float(
                np.linalg.norm(predicted_position - position)
            )
            preliminary = _geometry_event_candidates(
                position,
                predicted_position,
                config,
                attempted_displacement=attempted_displacement,
            )
            remaining_time = config.max_time_s - time_s
            if remaining_time <= dt:
                preliminary.append(
                    (
                        max(0.0, min(1.0, remaining_time / dt)),
                        0,
                        Termination.TIME_TIMEOUT,
                        position,
                        "physical elapsed-time deadline reached",
                    )
                )
            remaining_path = config.max_path_m - path_m
            if predicted_segment > 0.0 and remaining_path <= predicted_segment:
                preliminary.append(
                    (
                        max(0.0, min(1.0, remaining_path / predicted_segment)),
                        1,
                        Termination.PATH_TIMEOUT,
                        position,
                        "physical path-length deadline reached",
                    )
                )
            preliminary_selected = (
                min(preliminary, key=lambda item: (item[0], item[1]))
                if preliminary
                else None
            )
            immediate_zero_fraction = (
                preliminary_selected is not None
                and preliminary_selected[0] == 0.0
            )
            if immediate_zero_fraction:
                # Preserve the positive attempted timestep and do not query a
                # midpoint that may lie beyond a tolerance-close boundary.
                step_dt = dt
                new_velocity = predicted_velocity
                new_position = predicted_position
                # The full-step prediction above was pushed with the start
                # fields, so those are the "midpoint" fields this step used.
                magnetic_midpoint = magnetic_start
                electric_midpoint = electric_start
                midpoint_b_t = actual_b_t
            else:
                trial_fraction = (
                    preliminary_selected[0]
                    if preliminary_selected is not None
                    else 1.0
                )
                step_dt = dt * trial_fraction
                if trial_fraction < 1.0:
                    predicted_velocity = push(
                        velocity, electric_start, magnetic_start, step_dt
                    )
                    predicted_position = (
                        position
                        + 0.5 * (velocity + predicted_velocity) * step_dt
                    )
                    attempted_displacement = (
                        0.5 * (velocity + predicted_velocity) * step_dt
                    )
                midpoint_position = 0.5 * (position + predicted_position)
                midpoint_time = time_s + 0.5 * step_dt
                magnetic_midpoint = np.asarray(
                    field.magnetic_cartesian(midpoint_position), dtype=np.float64
                )
                electric_midpoint = np.asarray(
                    field.electric_cartesian(midpoint_position, midpoint_time),
                    dtype=np.float64,
                )
                if (
                    magnetic_midpoint.shape != (3,)
                    or electric_midpoint.shape != (3,)
                    or not np.isfinite(magnetic_midpoint).all()
                    or not np.isfinite(electric_midpoint).all()
                ):
                    raise OrbitNumericsError("midpoint field is invalid")
                midpoint_b_t = float(np.linalg.norm(magnetic_midpoint))
                maximum_b_seen = max(maximum_b_seen, midpoint_b_t)
                if midpoint_b_t > field.max_b_t * (
                    1.0 + 64.0 * np.finfo(float).eps
                ):
                    raise OrbitNumericsError(
                        "midpoint field exceeds declared/certified maximum"
                    )
                new_velocity = push(
                    velocity, electric_midpoint, magnetic_midpoint, step_dt
                )
                new_position = (
                    position + 0.5 * (velocity + new_velocity) * step_dt
                )
        except Exception as error:
            termination = Termination.FIELD_FAILURE
            reason = f"{type(error).__name__}: {error}"
            event_witness = _failure_witness(
                termination,
                config,
                position,
                step=step,
                elapsed_s=time_s,
                path_m=path_m,
                condition="field_or_pusher_exception",
            )
            break
        if not np.isfinite(new_position).all():
            termination = Termination.NONFINITE_STATE
            reason = "position became nonfinite"
            event_witness = _failure_witness(
                termination,
                config,
                position,
                step=step,
                elapsed_s=time_s,
                path_m=path_m,
                condition="nonfinite_predicted_position",
            )
            break
        speed2 = float(np.dot(new_velocity, new_velocity))
        if not isfinite(speed2):
            termination = Termination.NONFINITE_STATE
            reason = "velocity became nonfinite"
            event_witness = _failure_witness(
                termination,
                config,
                position,
                step=step,
                elapsed_s=time_s,
                path_m=path_m,
                condition="nonfinite_predicted_velocity",
            )
            break
        if speed2 >= LIGHT_SPEED_M_PER_S**2:
            termination = Termination.EXTREME_RELATIVITY
            reason = "velocity reached binary64 light-speed resolution"
            event_witness = _failure_witness(
                termination,
                config,
                position,
                step=step,
                elapsed_s=time_s,
                path_m=path_m,
                condition="speed_reached_light_speed_resolution",
                observed_speed2_over_c2=speed2 / LIGHT_SPEED_M_PER_S**2,
            )
            break
        gamma = 1.0 / sqrt(1.0 - speed2 / LIGHT_SPEED_M_PER_S**2)
        if gamma > config.maximum_gamma:
            termination = Termination.EXTREME_RELATIVITY
            reason = "maximum declared relativistic gamma exceeded"
            event_witness = _failure_witness(
                termination,
                config,
                position,
                step=step,
                elapsed_s=time_s,
                path_m=path_m,
                condition="maximum_gamma_exceeded",
                observed_gamma=gamma,
                observed_speed2_over_c2=speed2 / LIGHT_SPEED_M_PER_S**2,
            )
            break
        segment = float(np.linalg.norm(new_position - position))
        attempted_displacement = (
            0.5 * (velocity + new_velocity) * step_dt
        )
        candidates = _geometry_event_candidates(
            position,
            new_position,
            config,
            attempted_displacement=attempted_displacement,
        )
        remaining_time = config.max_time_s - time_s
        if remaining_time <= step_dt:
            fraction = max(0.0, min(1.0, remaining_time / step_dt))
            candidates.append(
                (
                    fraction,
                    0,
                    Termination.TIME_TIMEOUT,
                    position + fraction * (new_position - position),
                    "physical elapsed-time deadline reached",
                )
            )
        remaining_path = config.max_path_m - path_m
        if segment > 0.0 and remaining_path <= segment:
            fraction = max(0.0, min(1.0, remaining_path / segment))
            candidates.append(
                (
                    fraction,
                    1,
                    Termination.PATH_TIMEOUT,
                    position + fraction * (new_position - position),
                    "physical path-length deadline reached",
                )
            )

        if not candidates and (
            step_dt <= 0.0
            or segment == 0.0
            or np.array_equal(position, new_position)
        ):
            termination = Termination.FIELD_FAILURE
            reason = "corrected orbit segment made no representable progress"
            event_witness = _failure_witness(
                termination,
                config,
                position,
                step=step,
                elapsed_s=time_s,
                path_m=path_m,
                condition="zero_progress_corrected_segment",
            )
            break

        if any(
            fraction == 0.0 and condition.startswith("tolerance_close_")
            for fraction, _, _, _, condition in candidates
        ):
            reflection_evidence = None
        else:
            try:
                reflection_evidence = _reflection_fraction(
                    position,
                    new_position,
                    velocity,
                    new_velocity,
                    field,
                    initial_parallel_sign,
                )
            except Exception:
                reflection_evidence = None
        if reflection_evidence is not None:
            reflection_fraction = reflection_evidence[0]
            candidates.append(
                (
                    reflection_fraction,
                    3,
                    Termination.REFLECTED,
                    position
                    + reflection_fraction * (new_position - position),
                    "parallel velocity reached zero within the step",
                )
            )

        event_fraction = 1.0
        selected = None
        if candidates:
            selected = min(candidates, key=lambda item: (item[0], item[1]))
            event_fraction = selected[0]
        candidate_fractions: dict[str, float | None] = {
            "time_timeout": None,
            "path_timeout": None,
            "wall_hit": None,
            "reflected": None,
            "domain_escape": None,
        }
        for candidate_fraction, _, candidate_kind, _, _ in candidates:
            previous = candidate_fractions[candidate_kind.value]
            if previous is None or candidate_fraction < previous:
                candidate_fractions[candidate_kind.value] = candidate_fraction
        # v1.6: the event velocity is the Boris state at the event time, i.e. a
        # push of duration ``event_fraction * step_dt`` from the step-start
        # velocity with the SAME midpoint fields the full step used. The former
        # chord interpolation ``v0 + f*(v1 - v0)`` shortened |v| by ~(f*theta)^2/12
        # per event, which is a spurious energy error in a pure-B field. The
        # event POSITION stays on the chord because that is the geometric
        # definition the crossing candidates were solved on.
        event_velocity = _event_velocity(
            push,
            velocity,
            new_velocity,
            electric_midpoint,
            magnetic_midpoint,
            step_dt,
            event_fraction,
        )
        event_position = position + event_fraction * (new_position - position)
        event_time = event_fraction * step_dt
        event_path = event_fraction * segment
        selected_termination = selected[2] if selected is not None else Termination.STEP_LIMIT
        reflection_bracket = None
        if reflection_evidence is not None:
            (
                reflection_root,
                reflection_low,
                reflection_high,
                reflection_low_value,
                reflection_high_value,
                reflection_root_value,
            ) = reflection_evidence
            reflection_bracket = {
                "low_fraction": reflection_low,
                "high_fraction": reflection_high,
                "low_parallel_m_per_s": reflection_low_value,
                "high_parallel_m_per_s": reflection_high_value,
                "root_fraction": reflection_root,
                "root_parallel_m_per_s": reflection_root_value,
            }
        event_witness = {
            "kind": selected_termination.value,
            "config": _witness_config(config),
            "step_start_position_m": tuple(map(float, position)),
            "step_end_position_m": tuple(map(float, new_position)),
            "event_position_m": tuple(
                map(
                    float,
                    selected[3] if selected is not None else event_position,
                )
            ),
            "step_start_velocity_m_per_s": tuple(map(float, velocity)),
            "step_end_velocity_m_per_s": tuple(map(float, new_velocity)),
            "event_velocity_m_per_s": tuple(map(float, event_velocity)),
            "step_magnetic_midpoint_t": tuple(map(float, magnetic_midpoint)),
            "step_electric_midpoint_v_per_m": tuple(
                map(float, electric_midpoint)
            ),
            "event_fraction": event_fraction,
            "event_resolution": (
                "tolerance_close_fraction_zero"
                if selected is not None
                and selected[4].startswith("tolerance_close_")
                else "interpolated"
                if selected is not None
                else "completed_step"
            ),
            "candidate_fractions": candidate_fractions,
            "reflection_bracket": reflection_bracket,
            "start_elapsed_time_s": time_s,
            "start_path_length_m": path_m,
            "step_dt_s": step_dt,
            "step_segment_length_m": segment,
            "step_index": step,
            "condition": selected[4] if selected is not None else "maximum_steps_reached",
            "observed_gamma": 1.0 / sqrt(
                1.0
                - float(np.dot(event_velocity, event_velocity))
                / LIGHT_SPEED_M_PER_S**2
            ),
            "observed_speed2_over_c2": float(
                np.dot(event_velocity, event_velocity)
            ) / LIGHT_SPEED_M_PER_S**2,
            "field_identity_sha256": "0" * 64,
            "config_identity_sha256": "0" * 64,
            "policy_identity_sha256": "0" * 64,
        }
        midpoint_speed = 0.5 * (velocity + event_velocity)
        start_gamma = 1.0 / sqrt(
            1.0
            - float(np.dot(velocity, velocity)) / LIGHT_SPEED_M_PER_S**2
        )
        event_gamma = 1.0 / sqrt(
            1.0
            - float(np.dot(event_velocity, event_velocity))
            / LIGHT_SPEED_M_PER_S**2
        )
        midpoint_gamma = 0.5 * (start_gamma + event_gamma)
        gyro.add(
            abs(ELECTRON_CHARGE_C)
            * midpoint_b_t
            * event_time
            / (midpoint_gamma * ELECTRON_MASS_KG),
            _mu(midpoint_speed, magnetic_midpoint),
        )
        position = event_position
        velocity = event_velocity
        time_s += event_time
        path_m += event_path
        current_energy = kinetic_energy_j_from_velocity(velocity)
        energy_error = max(energy_error, abs(current_energy - initial_energy) / initial_energy)
        current_mu = _mu(midpoint_speed, magnetic_midpoint)
        if current_mu is not None:
            mu_values.append(current_mu)
        if selected is not None:
            _, _, termination, selected_position, reason = selected
            position = selected_position
            if termination is Termination.WALL_HIT:
                wall_endpoint = tuple(map(float, position))
            break

    final_energy = kinetic_energy_j_from_velocity(velocity)
    mu_variation = None
    if mu_values:
        mean_mu = float(np.mean(mu_values))
        if mean_mu > 0.0:
            mu_variation = max(abs(value - mean_mu) for value in mu_values) / mean_mu
    # Net axial displacement over accumulated path. A tolerance-close boundary
    # snap (<= close tolerance) is not part of the accumulated path, so bound
    # the denominator by the displacement itself to keep the ratio in [0, 1].
    axial_displacement = abs(float(position[2] - initial_position[2]))
    transit_fraction = axial_displacement / max(
        path_m, axial_displacement, np.finfo(float).tiny
    )
    return OrbitResult(
        launch.launch_id, termination, reason, tuple(map(float, position)),
        tuple(map(float, velocity)), wall_endpoint, time_s, path_m, steps,
        gyro.phase, len(gyro.averages), tuple(gyro.averages), initial_energy,
        final_energy, energy_error, mu_variation,
        transit_fraction,
        maximum_b_seen,
        config.max_time_s,
        config.max_path_m,
        config.event_tolerance_m,
        dt,
        backend,
        event_witness,
    )
