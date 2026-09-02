"""v1.6 contract: energy-consistent event velocity with replayable midpoint fields.

Before v1.6 the final velocity of a physical event at fraction ``f`` of the last
step was the chord ``v0 + f*(v1 - v0)``. In a pure magnetic field the Boris
push rotates ``v`` exactly, so the chord shortens ``|v|`` by ~(f*theta)^2/12 and
reports a spurious energy error (6.1e-4 relative on the real campaign field).
v1.6 defines the event velocity as ``boris_push(v0, E_mid, B_mid, f*step_dt)``
with the SAME midpoint fields the full step used, records those fields plus the
event velocity in the witness, and makes the validator replay them.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from math import pi

import numpy as np
import pytest

from cft_revival.orbit_mc import (
    AnalyticField,
    ElectronLaunch,
    OrbitConfig,
    OrbitResult,
    OrbitValidationError,
    Termination,
    integrate_orbit,
    launch_velocity,
    relativistic_boris_push,
)
from cft_revival.orbit_mc.artifacts import _validate_event_witness
from cft_revival.orbit_mc.models import (
    ELECTRON_CHARGE_C,
    kinetic_energy_j_from_velocity,
)

_B0 = 0.02
_CURVATURE = 1.0e4


def _bottle(position: np.ndarray) -> np.ndarray:
    """Divergence-free pure-B mirror; |B| <= 0.035 T on the test geometry."""

    x, y, z = position
    return np.array(
        [
            -_CURVATURE * _B0 * x * z,
            -_CURVATURE * _B0 * y * z,
            _B0 * (1.0 + _CURVATURE * z * z),
        ]
    )


_PURE_B = AnalyticField(_bottle, None, 0.04)
_UNIFORM_B = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.01]), None, 0.01)


def _pure_b_config(**changes: object) -> OrbitConfig:
    values: dict[str, object] = {
        "wall_radius_m": 8.0e-3,
        "wall_z_min_m": -8.0e-3,
        "wall_z_max_m": 8.0e-3,
        "domain_radius_m": 1.0e-2,
        "domain_z_min_m": -8.0e-3,
        "domain_z_max_m": 8.0e-3,
        "max_time_s": 1.0e-7,
        "max_path_m": 1.0,
        "max_steps": 200_000,
        "fixed_dt_s": 1.0e-11,
    }
    values.update(changes)
    return OrbitConfig(**values)


def _launch(
    index: int,
    *,
    energy_ev: float = 200.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch: float = 0.2,
    direction: int = 1,
    phase: float = 0.0,
) -> ElectronLaunch:
    launch_id = f"event-velocity-replay:E0:P0:X{index}:D{direction:+d}:G0"
    seed = int.from_bytes(sha256(launch_id.encode()).digest()[:8], "big")
    return ElectronLaunch(
        launch_id, seed, energy_ev, pitch, position, direction, phase,
        f"surface-{index}",
    )


def _record(result: OrbitResult) -> dict[str, object]:
    record = asdict(result)
    record["termination"] = result.termination.value
    return record


def _validate(record: dict[str, object], launch: ElectronLaunch) -> None:
    _validate_event_witness(
        record,
        launch=asdict(launch),
        expected_field_sha256="0" * 64,
        expected_config_sha256="0" * 64,
        expected_policy_sha256="0" * 64,
        allow_replay_dependent_failures=False,
    )


def _random_pure_b_launches(count: int) -> list[ElectronLaunch]:
    rng = np.random.default_rng(20260903)
    launches: list[ElectronLaunch] = []
    for index in range(count):
        radius = rng.uniform(0.0, 5.0e-3)
        angle = rng.uniform(0.0, 2.0 * pi)
        launches.append(
            _launch(
                index,
                position=(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    rng.uniform(-4.0e-3, 4.0e-3),
                ),
                pitch=rng.uniform(0.3, 1.45),
                direction=int(rng.choice([-1, 1])),
                phase=rng.uniform(0.0, 2.0 * pi),
            )
        )
    # Near-wall, nearly perpendicular launches guarantee wall hits in the mix.
    for offset in range(8):
        launches.append(
            _launch(
                count + offset,
                position=(6.8e-3 + 1.0e-4 * offset, 0.0, 2.0e-4 * offset),
                pitch=pi / 2 - 0.05,
                phase=0.25 * pi * offset,
            )
        )
    return launches


def test_pure_b_interior_events_conserve_energy_and_replay() -> None:
    """(a) many random pure-B launches: energy <= 1e-12 and the witness replays."""

    config = _pure_b_config()
    launches = _random_pure_b_launches(80)
    terminations: dict[str, int] = {}
    interior_fraction_events = 0
    chord_would_fail_gate = 0
    for launch in launches:
        result = integrate_orbit(launch, _PURE_B, config)
        terminations[result.termination.value] = (
            terminations.get(result.termination.value, 0) + 1
        )
        assert result.termination in {
            Termination.WALL_HIT, Termination.REFLECTED, Termination.DOMAIN_ESCAPE,
        }, result.reason
        assert result.maximum_relative_energy_error <= 1.0e-12
        witness = result.event_witness
        fraction = witness["event_fraction"]
        v0 = np.asarray(witness["step_start_velocity_m_per_s"])
        v1 = np.asarray(witness["step_end_velocity_m_per_s"])
        b_mid = np.asarray(witness["step_magnetic_midpoint_t"])
        e_mid = np.asarray(witness["step_electric_midpoint_v_per_m"])
        assert np.array_equal(e_mid, np.zeros(3))
        assert np.linalg.norm(b_mid) <= result.maximum_b_t
        # The witnessed event velocity IS the final velocity and IS the push.
        assert tuple(witness["event_velocity_m_per_s"]) == result.final_velocity_m_per_s
        if 0.0 < fraction < 1.0:
            interior_fraction_events += 1
            replayed = relativistic_boris_push(
                v0, e_mid, b_mid, fraction * witness["step_dt_s"]
            )
            assert np.array_equal(replayed, np.asarray(result.final_velocity_m_per_s))
            chord = v0 + fraction * (v1 - v0)
            chord_error = abs(
                kinetic_energy_j_from_velocity(chord) - result.initial_energy_j
            ) / result.initial_energy_j
            chord_would_fail_gate += chord_error > 1.0e-10
        _validate(_record(result), launch)
    # The mix must exercise every physical first-event class at interior fractions
    # and the pre-v1.6 chord must have been failing the 1e-10 protocol gate.
    assert set(terminations) == {"wall_hit", "reflected", "domain_escape"}
    assert interior_fraction_events >= 60
    assert chord_would_fail_gate >= interior_fraction_events // 2


def test_fraction_one_event_velocity_is_full_step_velocity_bitwise() -> None:
    """(b) f == 1 (completed step / STEP_LIMIT) reproduces the full-step push."""

    launch = _launch(0, energy_ev=10.0, pitch=pi / 3, phase=0.3)
    config = _pure_b_config(max_steps=1, fixed_dt_s=1.0e-11)
    result = integrate_orbit(launch, _PURE_B, config)
    assert result.termination is Termination.STEP_LIMIT
    witness = result.event_witness
    assert witness["event_fraction"] == 1.0
    assert witness["event_resolution"] == "completed_step"
    v0 = np.asarray(witness["step_start_velocity_m_per_s"])
    end = tuple(witness["step_end_velocity_m_per_s"])
    assert tuple(witness["event_velocity_m_per_s"]) == end
    assert result.final_velocity_m_per_s == end
    replayed = relativistic_boris_push(
        v0,
        np.asarray(witness["step_electric_midpoint_v_per_m"]),
        np.asarray(witness["step_magnetic_midpoint_t"]),
        witness["step_dt_s"],
    )
    assert tuple(map(float, replayed)) == end
    # Determinism check behind the special case: 1.0*dt is dt.
    assert np.array_equal(
        replayed,
        relativistic_boris_push(
            v0,
            np.asarray(witness["step_electric_midpoint_v_per_m"]),
            np.asarray(witness["step_magnetic_midpoint_t"]),
            1.0 * witness["step_dt_s"],
        ),
    )
    _validate(_record(result), launch)


def test_fraction_zero_event_velocity_is_start_velocity_bitwise() -> None:
    """(b) f == 0 (tolerance-close snap) reproduces the step-start velocity."""

    launch = _launch(
        0,
        energy_ev=10.0,
        position=(np.nextafter(1.0, 0.0), 0.0, 0.0),
        pitch=pi / 2,
        phase=1.5 * pi,
    )
    config = OrbitConfig(
        1.0, -0.1, 0.1, 2.0, -2.0, 2.0, 1.0e-6, 10.0,
        max_steps=10, event_tolerance_m=1.0e-9, fixed_dt_s=1.0e-11,
    )
    result = integrate_orbit(launch, _UNIFORM_B, config)
    assert result.termination is Termination.WALL_HIT
    witness = result.event_witness
    assert witness["event_fraction"] == 0.0
    assert witness["event_resolution"] == "tolerance_close_fraction_zero"
    start = tuple(map(float, launch_velocity(launch, _UNIFORM_B)))
    assert tuple(witness["step_start_velocity_m_per_s"]) == start
    assert tuple(witness["event_velocity_m_per_s"]) == start
    assert result.final_velocity_m_per_s == start
    # Pre-v1.6 chord at f == 0 was v0 + 0.0*(v1 - v0) == v0 as well.
    v1 = np.asarray(witness["step_end_velocity_m_per_s"])
    assert tuple(map(float, np.asarray(start) + 0.0 * (v1 - np.asarray(start)))) == start
    # The zero-fraction path pushed the prediction with the START fields, and
    # those are what the witness must carry as the step's "midpoint" fields.
    assert tuple(witness["step_magnetic_midpoint_t"]) == (0.0, 0.0, 0.01)
    assert tuple(witness["step_electric_midpoint_v_per_m"]) == (0.0, 0.0, 0.0)
    assert result.maximum_relative_energy_error == 0.0
    _validate(_record(result), launch)


def _interior_pure_b_result() -> tuple[OrbitResult, ElectronLaunch]:
    launch = _launch(3, position=(1.0e-3, 0.0, 0.0), pitch=1.3, phase=0.7)
    result = integrate_orbit(launch, _PURE_B, _pure_b_config())
    assert result.termination is Termination.REFLECTED
    assert 0.0 < result.event_witness["event_fraction"] < 1.0
    return result, launch


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("event_velocity", "event velocity does not replay"),
        ("chord_velocity", "event velocity does not replay"),
        ("magnetic_midpoint", "step-end velocity does not replay"),
        ("electric_midpoint", "step-end velocity does not replay"),
        ("magnetic_over_maximum", "exceeds the result maximum B"),
        ("final_velocity", "differs from the witnessed event velocity"),
        ("nonfinite_midpoint", "step_magnetic_midpoint_t is invalid"),
    ],
)
def test_tampered_event_velocity_or_midpoint_fields_are_rejected(
    tamper: str, message: str
) -> None:
    """(c) the validator rejects any inconsistency in the replay triple."""

    result, launch = _interior_pure_b_result()
    record = _record(result)
    witness = record["event_witness"]
    v0 = np.asarray(witness["step_start_velocity_m_per_s"])
    v1 = np.asarray(witness["step_end_velocity_m_per_s"])
    fraction = witness["event_fraction"]
    if tamper == "event_velocity":
        tampered = np.asarray(witness["event_velocity_m_per_s"]) * (1.0 + 1.0e-12)
        witness["event_velocity_m_per_s"] = tuple(map(float, tampered))
        record["final_velocity_m_per_s"] = tuple(map(float, tampered))
    elif tamper == "chord_velocity":
        chord = v0 + fraction * (v1 - v0)
        witness["event_velocity_m_per_s"] = tuple(map(float, chord))
        record["final_velocity_m_per_s"] = tuple(map(float, chord))
    elif tamper == "magnetic_midpoint":
        b_mid = np.asarray(witness["step_magnetic_midpoint_t"]) * (1.0 - 1.0e-9)
        witness["step_magnetic_midpoint_t"] = tuple(map(float, b_mid))
    elif tamper == "electric_midpoint":
        witness["step_electric_midpoint_v_per_m"] = (0.0, 0.0, 1.0)
    elif tamper == "magnetic_over_maximum":
        b_mid = np.asarray(witness["step_magnetic_midpoint_t"])
        b_mid = b_mid / np.linalg.norm(b_mid) * record["maximum_b_t"] * (1.0 + 1.0e-9)
        witness["step_magnetic_midpoint_t"] = tuple(map(float, b_mid))
    elif tamper == "final_velocity":
        final = np.asarray(record["final_velocity_m_per_s"])
        final[0] = np.nextafter(final[0], np.inf)
        record["final_velocity_m_per_s"] = tuple(map(float, final))
    else:
        witness["step_magnetic_midpoint_t"] = (0.0, 0.0, float("nan"))
    with pytest.raises(OrbitValidationError, match=message):
        _validate(record, launch)


def test_missing_replay_keys_break_witness_closure() -> None:
    result, launch = _interior_pure_b_result()
    for key in (
        "event_velocity_m_per_s",
        "step_magnetic_midpoint_t",
        "step_electric_midpoint_v_per_m",
    ):
        record = _record(result)
        del record["event_witness"][key]
        with pytest.raises(OrbitValidationError, match="event witness is not closed"):
            _validate(record, launch)


def test_failure_witness_must_carry_zero_replay_vectors() -> None:
    failing = AnalyticField(
        lambda position: (
            np.array([0.0, 0.0, 0.01])
            if np.array_equal(position, np.zeros(3))
            else np.full(3, np.nan)
        ),
        None,
        0.01,
    )
    launch = _launch(0, energy_ev=10.0)
    config = OrbitConfig(
        1.0, -0.1, 0.1, 2.0, -2.0, 2.0, 1.0e-6, 10.0, max_steps=10, fixed_dt_s=1.0e-11,
    )
    result = integrate_orbit(launch, failing, config)
    assert result.termination is Termination.FIELD_FAILURE
    witness = result.event_witness
    for key in (
        "event_velocity_m_per_s",
        "step_magnetic_midpoint_t",
        "step_electric_midpoint_v_per_m",
    ):
        assert list(witness[key]) == [0.0, 0.0, 0.0]
    record = _record(result)
    _validate_event_witness(
        record,
        launch=asdict(launch),
        expected_field_sha256="0" * 64,
        expected_config_sha256="0" * 64,
        expected_policy_sha256="0" * 64,
        allow_replay_dependent_failures=True,
    )
    record["event_witness"]["step_magnetic_midpoint_t"] = (0.0, 0.0, 0.01)
    with pytest.raises(OrbitValidationError, match="zero event velocity and midpoint"):
        _validate_event_witness(
            record,
            launch=asdict(launch),
            expected_field_sha256="0" * 64,
            expected_config_sha256="0" * 64,
            expected_policy_sha256="0" * 64,
            allow_replay_dependent_failures=True,
        )


def test_nonzero_e_field_partial_step_energy_change_replays_boris_push() -> None:
    """(d) with E != 0 the last partial step is the Boris push, not a gate."""

    e_field = np.array([0.0, 0.0, 1.0e5])
    field = AnalyticField(
        lambda _x: np.array([0.0, 0.0, 0.01]),
        lambda _x, _t: e_field,
        0.01,
    )
    launch = _launch(0, energy_ev=10.0, pitch=0.2, phase=0.4)
    config = OrbitConfig(
        1.0, -0.1, 0.1, 2.0, -2.0, 4.0e-5, 1.0e-6, 10.0,
        max_steps=100, fixed_dt_s=1.0e-11,
    )
    result = integrate_orbit(launch, field, config)
    assert result.termination is Termination.DOMAIN_ESCAPE
    witness = result.event_witness
    fraction = witness["event_fraction"]
    assert 0.0 < fraction < 1.0
    assert tuple(witness["step_electric_midpoint_v_per_m"]) == tuple(e_field)
    v0 = np.asarray(witness["step_start_velocity_m_per_s"])
    replayed = relativistic_boris_push(
        v0,
        np.asarray(witness["step_electric_midpoint_v_per_m"]),
        np.asarray(witness["step_magnetic_midpoint_t"]),
        fraction * witness["step_dt_s"],
    )
    assert np.array_equal(replayed, np.asarray(result.final_velocity_m_per_s))
    assert result.final_energy_j == kinetic_energy_j_from_velocity(
        np.asarray(result.final_velocity_m_per_s)
    )
    # Energy is NOT conserved here and the result must say so honestly.
    assert result.maximum_relative_energy_error > 1.0e-3
    # Sanity: the partial-step energy change is the work of E along the chord
    # displacement to within the O(theta, dE) averaging error of the scheme.
    start_energy = kinetic_energy_j_from_velocity(v0)
    displacement = np.asarray(witness["event_position_m"]) - np.asarray(
        witness["step_start_position_m"]
    )
    work = ELECTRON_CHARGE_C * float(np.dot(e_field, displacement))
    delta = result.final_energy_j - start_energy
    assert abs(delta - work) <= 0.1 * abs(delta)
    _validate(_record(result), launch)
