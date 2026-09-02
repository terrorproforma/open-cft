from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from math import hypot, pi

import numpy as np
import pytest

from cft_revival.orbit_mc import (
    AnalyticField,
    ElectronLaunch,
    EstimatorPolicy,
    OrbitConfig,
    OrbitValidationError,
    Termination,
    checkpoint,
    frozen_batch_manifest,
    integrate_orbit,
    load_and_verify_artifact,
    reduce_results,
    result_artifact,
    write_artifact,
)
from cft_revival.orbit_mc.artifacts import (
    _validate_event_witness,
    content_hash,
)


def _launch(
    index: int,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pitch: float = 0.2,
    phase: float = 0.0,
) -> ElectronLaunch:
    launch_id = f"termination-witnesses:E0:P0:X{index}:D+1:G0"
    seed = int.from_bytes(sha256(launch_id.encode()).digest()[:8], "big")
    return ElectronLaunch(
        launch_id,
        seed,
        10.0,
        pitch,
        position,
        1,
        phase,
        f"surface-{index}",
    )


def _base_config(**changes: object) -> OrbitConfig:
    values: dict[str, object] = {
        "wall_radius_m": 1.0,
        "wall_z_min_m": -0.1,
        "wall_z_max_m": 0.1,
        "domain_radius_m": 2.0,
        "domain_z_min_m": -2.0,
        "domain_z_max_m": 2.0,
        "max_time_s": 1.0e-6,
        "max_path_m": 10.0,
        "max_steps": 10,
        "event_tolerance_m": 1.0e-9,
        "fixed_dt_s": 1.0e-11,
    }
    values.update(changes)
    return OrbitConfig(**values)


def test_production_witnesses_checkpoint_all_ten_termination_classes() -> None:
    uniform = AnalyticField(
        lambda _x: np.array([0.0, 0.0, 0.01]), None, 0.01
    )
    launches = [
        _launch(
            0,
            position=(np.nextafter(1.0, 0.0), 0.0, 0.0),
            pitch=pi / 2,
            phase=1.5 * pi,
        ),
        _launch(
            1,
            position=(np.nextafter(2.0, 0.0), 0.0, 0.2),
            pitch=pi / 2,
            phase=1.5 * pi,
        ),
        _launch(2),
        _launch(3),
        _launch(4),
        _launch(5, position=(2.0, 0.0, 0.0)),
        _launch(6, pitch=pi / 2),
        _launch(7),
        _launch(8),
        _launch(9, position=(2.0e-5, 0.0, 0.0), pitch=pi / 3),
    ]
    results = [
        integrate_orbit(launches[0], uniform, _base_config()),
        integrate_orbit(launches[1], uniform, _base_config()),
        integrate_orbit(
            launches[2],
            uniform,
            _base_config(max_time_s=5.0e-12),
        ),
        integrate_orbit(
            launches[3],
            uniform,
            _base_config(max_path_m=1.0e-6),
        ),
        integrate_orbit(
            launches[4],
            uniform,
            _base_config(max_steps=1),
        ),
        integrate_orbit(launches[5], uniform, _base_config()),
    ]
    field_failure = AnalyticField(
        lambda position: (
            np.array([0.0, 0.0, 0.01])
            if np.array_equal(position, np.zeros(3))
            else np.full(3, np.nan)
        ),
        None,
        0.01,
    )
    results.append(
        integrate_orbit(launches[6], field_failure, _base_config())
    )
    guarded = _base_config(maximum_gamma=2.0, fixed_dt_s=1.0e-13)
    results.append(
        integrate_orbit(
            launches[7],
            uniform,
            guarded,
            velocity_pusher=lambda *_args: np.full(3, np.nan),
        )
    )
    extreme_field = AnalyticField(
        lambda _x: np.array([0.0, 0.0, 0.01]),
        lambda _x, _time: np.array([1.0e20, 0.0, 0.0]),
        0.01,
    )
    results.append(integrate_orbit(launches[8], extreme_field, guarded))

    curvature = 10000.0
    b0 = 0.02

    def bottle(position: np.ndarray) -> np.ndarray:
        x, y, z = position
        return np.array(
            [
                -curvature * b0 * x * z,
                -curvature * b0 * y * z,
                b0 * (1.0 + curvature * z*z),
            ]
        )

    bottle_field = AnalyticField(
        bottle, None, b0 * (1.0 + curvature * 0.02**2)
    )
    bottle_dt = (
        2.0
        * pi
        * 9.1093837139e-31
        / (1.602176634e-19 * bottle_field.max_b_t * 64)
    )
    results.append(
        integrate_orbit(
            launches[9],
            bottle_field,
            OrbitConfig(
                0.05,
                -0.02,
                0.02,
                0.06,
                -0.02,
                0.02,
                2.0e-8,
                0.1,
                max_steps=100_000,
                max_rotation_rad=0.5,
                fixed_dt_s=bottle_dt,
            ),
        )
    )
    results = sorted(results, key=lambda item: item.launch_id)
    launches = sorted(launches, key=lambda item: item.launch_id)
    assert {item.termination for item in results} == set(Termination)
    for result in results:
        if result.termination not in {
            Termination.INITIAL_STATE_INVALID,
            Termination.FIELD_FAILURE,
            Termination.NONFINITE_STATE,
            Termination.EXTREME_RELATIVITY,
        }:
            assert result.event_witness["step_dt_s"] > 0.0

    manifest = frozen_batch_manifest(launches, batch_size=len(launches))
    manifest_hash = content_hash(
        {
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "batches": manifest,
        }
    )
    state = checkpoint(
        "termination-witnesses",
        [0],
        launches,
        results,
        manifest,
        field_identity_sha256="a" * 64,
        config_identity_sha256="b" * 64,
        policy_identity_sha256="c" * 64,
        minimum_certificate_tightness_ratio_authority=0.001,
        estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        expected_batch_manifest_sha256=manifest_hash,
    )
    assert state["coverage"]["completed_launches"] == 10


# --- v3 campaign regression -------------------------------------------------
#
# exp/cft-orbit-wall-loss-v3 (orbit_mc v1.4, commit 25dbeaaf) failed at runtime
# with ``OrbitValidationError: physical event witness requires a positive
# step``. When an electron converged to within roundoff of the cylindrical
# wall/domain boundary, ``_first_cylinder_crossing`` returned fraction 0.0, so
# ``step_dt = 0.0``, the corrected step made no displacement, no event
# candidate existed, and the loop spun to ``max_steps`` producing a STEP_LIMIT
# witness with ``step_dt_s == 0.0``. Every launch below spins to ``max_steps``
# on v1.4 (verified out-of-tree against the frozen v3 worktree); v1.5 must end
# each one promptly on the snapped boundary with a validator-accepted witness.

_V3_GEOMETRY = {
    "wall_radius_m": 0.002,
    "wall_z_min_m": 0.001,
    "wall_z_max_m": 0.018,
    "domain_radius_m": 0.002,
    "domain_z_min_m": 0.001,
    "domain_z_max_m": 0.023,
    "max_time_s": 1.0e-8,
    "max_path_m": 0.03,
    "max_steps": 200_000,
    "max_rotation_rad": 0.16,
    "event_tolerance_m": 1.0e-9,
    "maximum_gamma": 20.0,
}
_V3_CAMPAIGN = "v3-zero-step-regression"
_UNIFORM_02T = AnalyticField(
    lambda _x: np.array([0.0, 0.0, 0.2]), None, 0.2
)


def _bottle_02t() -> AnalyticField:
    curvature = 2.0e4
    b0 = 0.2

    def bottle(position: np.ndarray) -> np.ndarray:
        x, y, z = position
        zz = z - 0.0095
        return np.array(
            [
                -curvature * b0 * x * zz,
                -curvature * b0 * y * zz,
                b0 * (1.0 + curvature * zz * zz),
            ]
        )

    return AnalyticField(bottle, None, b0 * (1.0 + curvature * 0.0135**2))


def _v3_launch(
    index: int,
    position: tuple[float, float, float],
    *,
    pitch: float,
    phase: float,
    direction: int = 1,
) -> ElectronLaunch:
    sign = "+" if direction > 0 else "-"
    launch_id = f"{_V3_CAMPAIGN}:E0:P0:X{index}:D{sign}1:G0"
    seed = int.from_bytes(sha256(launch_id.encode()).digest()[:8], "big")
    return ElectronLaunch(
        launch_id, seed, 25.0, pitch, position, direction, phase, f"v3-{index}"
    )


_V3_CASES = {
    # 1 ulp inside the dielectric wall, grazing outward approach (uniform B).
    "wall_roundoff_grazing": dict(
        field=_UNIFORM_02T,
        config={},
        launch=_v3_launch(
            0,
            (float(np.nextafter(0.002, 0.0)), 0.0, 0.005),
            pitch=pi / 2,
            phase=1.5 * pi + 0.28906 * pi,
        ),
        termination=Termination.WALL_HIT,
        condition="tolerance_close_wall_radial",
        max_steps=2,
    ),
    # 0.5 um inside the wall in a non-uniform field: the midpoint-corrected
    # segment repeatedly lands just inside the wall and converges to it.
    "wall_converging_bottle": dict(
        field=_bottle_02t(),
        config={},
        launch=_v3_launch(
            1, (0.002 - 5.0e-7, 0.0, 0.005), pitch=pi / 2, phase=0.875 * pi
        ),
        termination=Termination.WALL_HIT,
        condition=None,
        max_steps=64,
    ),
    # Same radial roundoff case outside the wall's axial extent -> domain.
    "domain_radius_roundoff": dict(
        field=_UNIFORM_02T,
        config={"wall_z_max_m": 0.004},
        launch=_v3_launch(
            2,
            (float(np.nextafter(0.002, 0.0)), 0.0, 0.005),
            pitch=pi / 2,
            phase=1.5 * pi,
        ),
        termination=Termination.DOMAIN_ESCAPE,
        condition="tolerance_close_domain_radial",
        max_steps=2,
    ),
    # 1 ulp below the +z domain plane, moving parallel to B (+z).
    "domain_z_max_roundoff": dict(
        field=_UNIFORM_02T,
        config={},
        launch=_v3_launch(
            3,
            (0.001, 0.0, float(np.nextafter(0.023, 0.0))),
            pitch=0.0,
            phase=0.0,
        ),
        termination=Termination.DOMAIN_ESCAPE,
        condition="tolerance_close_domain_z_max",
        max_steps=2,
    ),
    # 1 ulp above the -z domain plane, moving anti-parallel to B (-z).
    "domain_z_min_roundoff": dict(
        field=_UNIFORM_02T,
        config={},
        launch=_v3_launch(
            4,
            (0.001, 0.0, float(np.nextafter(0.001, 1.0))),
            pitch=0.0,
            phase=0.0,
            direction=-1,
        ),
        termination=Termination.DOMAIN_ESCAPE,
        condition="tolerance_close_domain_z_min",
        max_steps=2,
    ),
}


@pytest.mark.parametrize("case_id", sorted(_V3_CASES))
def test_v3_zero_step_wall_convergence_regression(case_id: str, tmp_path) -> None:
    case = _V3_CASES[case_id]
    config = OrbitConfig(**{**_V3_GEOMETRY, **case["config"]})
    launch = case["launch"]
    field = case["field"]

    result = integrate_orbit(launch, field, config)
    witness = result.event_witness

    # Prompt termination on the physical boundary, never a zero-step spin.
    assert result.termination is case["termination"], result.reason
    assert 1 <= result.steps <= case["max_steps"]
    assert result.steps < config.max_steps
    assert witness["step_dt_s"] > 0.0
    assert witness["step_dt_s"] == result.dt_s or witness["event_fraction"] > 0.0
    assert witness["kind"] == result.termination.value
    if case["condition"] is not None:
        assert witness["condition"] == case["condition"]
        assert witness["event_resolution"] == "tolerance_close_fraction_zero"
        assert witness["event_fraction"] == 0.0
    assert witness["event_resolution"] in {
        "tolerance_close_fraction_zero",
        "interpolated",
    }
    assert result.maximum_relative_energy_error <= 1.0e-3
    # A first-step boundary snap has zero accumulated path; the axial transit
    # ratio must still be a bounded, schema-valid fraction.
    assert 0.0 <= result.transit_fraction <= 1.0

    # Snapped endpoint lies on the claimed surface within the event tolerance.
    x, y, z = result.final_position_m
    radius = hypot(x, y)
    if result.termination is Termination.WALL_HIT:
        assert abs(radius - config.wall_radius_m) <= config.event_tolerance_m
        assert result.wall_endpoint_m == result.final_position_m
        assert config.wall_z_min_m <= z <= config.wall_z_max_m
    else:
        assert result.wall_endpoint_m is None
        if case["condition"] == "tolerance_close_domain_radial":
            assert abs(radius - config.domain_radius_m) <= config.event_tolerance_m
        elif case["condition"] == "tolerance_close_domain_z_max":
            assert abs(z - config.domain_z_max_m) <= config.event_tolerance_m
        else:
            assert abs(z - config.domain_z_min_m) <= config.event_tolerance_m
    assert tuple(witness["event_position_m"]) == result.final_position_m

    # Witness validator (campaign checkpoint path) accepts the record as-is.
    record = asdict(result)
    record["termination"] = result.termination.value
    _validate_event_witness(
        record,
        launch=asdict(launch),
        expected_field_sha256="0" * 64,
        expected_config_sha256="0" * 64,
        expected_policy_sha256="0" * 64,
        allow_replay_dependent_failures=False,
    )

    # Full artifact path: checkpoint, sealed result artifact with deterministic
    # replay, and verified reload, exactly as the campaign assessment does.
    launches = [launch]
    results = [result]
    manifest = frozen_batch_manifest(launches, batch_size=1)
    manifest_hash = content_hash(
        {
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "batches": manifest,
        }
    )
    identities = dict(
        field_identity_sha256="a" * 64,
        config_identity_sha256="b" * 64,
        policy_identity_sha256="c" * 64,
    )
    state = checkpoint(
        _V3_CAMPAIGN,
        [0],
        launches,
        results,
        manifest,
        minimum_certificate_tightness_ratio_authority=0.001,
        estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        expected_batch_manifest_sha256=manifest_hash,
        **identities,
    )
    assert state["pending_launch_ids"] == []
    artifact = result_artifact(
        campaign_id=_V3_CAMPAIGN,
        minimum_certificate_tightness_ratio_authority=0.001,
        estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        launches=launches,
        results=results,
        batch_manifest=manifest,
        summary=reduce_results(_V3_CAMPAIGN, results),
        interpolation_evidence={
            "certified_max_b_t": field.max_b_t,
            "reference_max_b_t": None,
            "runtime_max_seen_t": result.maximum_b_t,
            "dense_diagnostic_max_b_t": field.max_b_t,
            "certificate_tightness_ratio": 1.0,
            "minimum_certificate_tightness_ratio": 0.001,
            "certificate_preflight_passed": True,
            "material_map_sha256": "d" * 64,
            "field_error_report": {
                "sample_count": 1,
                "psi_node_max_abs_wb": 0.0,
                "br_max_abs_t": 0.0,
                "bz_max_abs_t": 0.0,
                "b_rms_t": 0.0,
                "b_relative_rms": 0.0,
            },
            "passed": True,
        },
        convergence_evidence={
            "timestep_passed": True,
            "cross_map_passed": True,
            "backend_parity_passed": True,
        },
        preregistration={
            "protocol_id": "v3-zero-step-regression",
            "frozen_before_outcomes": True,
            "held_out_geometry_status": "pending",
        },
        **identities,
    )
    authority = dict(
        expected_field_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_launches_sha256=artifact["identities"]["launches_sha256"],
        expected_batch_manifest_sha256=manifest_hash,
        expected_policy_sha256="c" * 64,
        expected_estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        expected_minimum_certificate_tightness_ratio=0.001,
    )
    path = tmp_path / f"{case_id}.json"
    evidence = write_artifact(path, artifact, field=field, config=config, **authority)
    verified = load_and_verify_artifact(
        path,
        field=field,
        config=config,
        expected_file_sha256=evidence.file_sha256,
        **authority,
    )
    assert verified.campaign_id == _V3_CAMPAIGN


@pytest.mark.parametrize("tamper", ["direction", "distance"])
def test_tolerance_close_witness_rejects_false_geometry(
    tamper: str,
) -> None:
    field = AnalyticField(
        lambda _x: np.array([0.0, 0.0, 0.01]), None, 0.01
    )
    launch = _launch(
        0,
        position=(np.nextafter(1.0, 0.0), 0.0, 0.0),
        pitch=pi / 2,
        phase=1.5 * pi,
    )
    result = integrate_orbit(launch, field, _base_config())
    record = asdict(result)
    record["termination"] = result.termination.value
    if tamper == "direction":
        record["event_witness"]["step_start_velocity_m_per_s"] = (
            -1.0,
            0.0,
            0.0,
        )
        record["event_witness"]["step_end_velocity_m_per_s"] = (
            -1.0,
            0.0,
            0.0,
        )
    else:
        record["event_witness"]["step_start_position_m"] = (
            1.0 - 2.0e-9,
            0.0,
            0.0,
        )
    with pytest.raises(
        OrbitValidationError,
        match="geometric event|tolerance-close|endpoint|segment length",
    ):
        _validate_event_witness(
            record,
            launch=asdict(launch),
            expected_field_sha256="0" * 64,
            expected_config_sha256="0" * 64,
            expected_policy_sha256="0" * 64,
            allow_replay_dependent_failures=True,
        )
