from __future__ import annotations

import copy
import hashlib
import json
from math import pi

import numpy as np
import pytest

from cft_revival.orbit_mc import (
    AnalyticField,
    EstimatorPolicy,
    OrbitConfig,
    OrbitValidationError,
    asymptotic_loss_cone_comparator,
    build_launch_ensemble,
    checkpoint,
    coupling_v42_handoff,
    frozen_batch_manifest,
    load_artifact,
    load_and_verify_artifact,
    load_checkpoint,
    probability_convergence,
    result_artifact,
    run_ensemble,
    validate_result_artifact,
    validate_checkpoint,
    wilson_interval,
    write_artifact,
    write_checkpoint,
)

POLICY_SHA256 = "d"*64
CERTIFICATE_FLOOR = 0.001
ESTIMATOR = EstimatorPolicy.UNWEIGHTED_BINOMIAL


def _campaign():
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    launches = build_launch_ensemble(
        ensemble_id="deterministic",
        energies_ev=(10.0, 30.0),
        pitch_angles_rad=(0.2, 0.8),
        positions=(("core", (0.0, 0.0, 0.0)),),
        directions=(-1, 1),
        gyrophase_count=4,
    )
    gyroperiod = 2*pi*9.1093837139e-31/(1.602176634e-19*0.1)
    config = OrbitConfig(
        0.1, -0.2, 0.2, 0.2, -0.3, 0.3, 1.5*gyroperiod, 1.0,
        max_steps=1000, max_rotation_rad=0.12,
    )
    return launches, field, config


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _reseal(value: dict[str, object]) -> None:
    value["integrity"]["payload_sha256"] = _hash(
        {key: item for key, item in value.items() if key != "integrity"}
    )


def test_launch_identity_and_batched_reduction_are_deterministic() -> None:
    launches, field, config = _campaign()
    first_results, first = run_ensemble("deterministic", launches, field, config, batch_size=3)
    second_results, second = run_ensemble("deterministic", reversed(launches), field, config, batch_size=11)
    assert first == second
    assert [item.launch_id for item in first_results] == [item.launch_id for item in second_results]
    assert first.trial_count == 32
    assert sum(dict(first.termination_counts).values()) == 32
    convergence = probability_convergence((first, second))
    assert convergence["successive_absolute_changes"] == [0.0]
    assert convergence["confidence_intervals_overlap"] == [True]


def test_wilson_binomial_interval_and_loss_cone_gating() -> None:
    interval = wilson_interval(5, 10)
    assert interval.lower < 0.5 < interval.upper
    blocked = asymptotic_loss_cone_comparator(
        b_min_t=0.1, b_max_t=0.4, maximum_rho_over_scale=0.2,
        maximum_mu_relative_variation=0.01, complete_gyrocycles=100,
    )
    assert not blocked["adiabatic_gates_passed"]
    assert blocked["loss_cone_probability"] is None
    accepted = asymptotic_loss_cone_comparator(
        b_min_t=0.1, b_max_t=0.4, maximum_rho_over_scale=0.01,
        maximum_mu_relative_variation=0.01, complete_gyrocycles=10,
    )
    assert accepted["adiabatic_gates_passed"]
    assert accepted["authority"].startswith("asymptotic_comparator")


def test_artifact_checkpoint_and_coupling_handoff_are_hash_bound(tmp_path) -> None:
    launches, field, config = _campaign()
    results, summary = run_ensemble("deterministic", launches, field, config)
    batches = frozen_batch_manifest(launches, batch_size=16)
    artifact = result_artifact(
        campaign_id="deterministic", field_identity_sha256="a"*64,
        config_identity_sha256="b"*64, policy_identity_sha256=POLICY_SHA256,
        minimum_certificate_tightness_ratio_authority=CERTIFICATE_FLOOR,
        estimator_policy=ESTIMATOR,
        launches=launches, results=results,
        batch_manifest=batches,
        summary=summary,
        interpolation_evidence={
            "certified_max_b_t": 0.1,
            "reference_max_b_t": None,
            "runtime_max_seen_t": 0.1,
            "dense_diagnostic_max_b_t": 0.1,
            "certificate_tightness_ratio": 1.0,
            "minimum_certificate_tightness_ratio": 0.001,
            "certificate_preflight_passed": True,
            "material_map_sha256": "c"*64,
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
            "protocol_id": "test-protocol",
            "frozen_before_outcomes": True,
            "held_out_geometry_status": "pending",
        },
    )
    batch_authority = artifact["identities"]["batch_manifest_sha256"]
    validate_result_artifact(
        artifact,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    path = tmp_path/"result.json"
    verified = write_artifact(
        path, artifact,
        field=field,
        config=config,
        expected_field_sha256="a"*64,
        expected_config_sha256="b"*64,
        expected_launches_sha256=artifact["identities"]["launches_sha256"],
        expected_batch_manifest_sha256=batch_authority,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    file_hash = verified.file_sha256
    assert hashlib.sha256(path.read_bytes()).hexdigest() == file_hash
    assert path.with_name(path.name+".sha256").is_file()
    reloaded = load_artifact(
        path, expected_file_sha256=file_hash,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    assert reloaded.file_sha256 == file_hash
    replayed = load_and_verify_artifact(
        path,
        field=field,
        config=config,
        expected_file_sha256=file_hash,
        expected_field_sha256="a"*64,
        expected_config_sha256="b"*64,
        expected_launches_sha256=artifact["identities"]["launches_sha256"],
        expected_batch_manifest_sha256=batch_authority,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    with pytest.raises(OrbitValidationError, match="external file"):
        load_artifact(
            path, expected_file_sha256="0"*64,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )
    handoff = coupling_v42_handoff(
        replayed, expected_batch_manifest_sha256=batch_authority
    )
    assert handoff["coupling_target"] == "cft-field-plasma-coupling/4.2.0"
    assert handoff["probability"] == summary.wall_hit.probability
    assert handoff["integration_status"] == (
        "export_only_pending_consumer_integration"
    )

    state = checkpoint(
        "deterministic",
        [0, 1],
        launches,
        results,
        batches,
        field_identity_sha256="a"*64,
        config_identity_sha256="b"*64,
        policy_identity_sha256=POLICY_SHA256,
        minimum_certificate_tightness_ratio_authority=CERTIFICATE_FLOOR,
        estimator_policy=ESTIMATOR,
        expected_batch_manifest_sha256=batch_authority,
    )
    checkpoint_path = tmp_path/"checkpoint.json"
    launch_hash = state["authority"]["launches_sha256"]
    batch_hash = state["authority"]["batch_manifest_sha256"]
    checkpoint_hash = write_checkpoint(
        checkpoint_path,
        state,
        expected_campaign_id="deterministic",
        expected_launches_sha256=launch_hash,
        expected_batch_manifest_sha256=batch_hash,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    reloaded_checkpoint = load_checkpoint(
        checkpoint_path,
        expected_file_sha256=checkpoint_hash,
        expected_campaign_id="deterministic",
        expected_launches_sha256=launch_hash,
        expected_batch_manifest_sha256=batch_hash,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        expected_previous_checkpoint_sha256="0"*64,
    )
    assert reloaded_checkpoint["payload_sha256"] == state["payload_sha256"]

    tampered = copy.deepcopy(artifact)
    tampered["summary"]["trial_count"] += 1
    with pytest.raises(OrbitValidationError, match="trial count|SHA-256"):
        validate_result_artifact(
            tampered, expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )

    forged_ci = copy.deepcopy(artifact)
    forged_ci["summary"]["wall_hit"]["lower"] = 0.9
    _reseal(forged_ci)
    with pytest.raises(OrbitValidationError, match="confidence interval"):
        validate_result_artifact(
            forged_ci, expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )

    forged_campaign = copy.deepcopy(artifact)
    forged_campaign["campaign_id"] = "fabricated-campaign"
    _reseal(forged_campaign)
    with pytest.raises(OrbitValidationError, match="campaign|ensemble"):
        validate_result_artifact(
            forged_campaign, expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )

    unknown_checkpoint = copy.deepcopy(state)
    unknown_checkpoint["fabricated"] = True
    unknown_checkpoint["payload_sha256"] = _hash(
        {
            key: item
            for key, item in unknown_checkpoint.items()
            if key != "payload_sha256"
        }
    )
    with pytest.raises(OrbitValidationError, match="not closed"):
        validate_checkpoint(
            unknown_checkpoint,
            expected_campaign_id="deterministic",
            expected_launches_sha256=launch_hash,
            expected_batch_manifest_sha256=batch_hash,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )

    rehashed = copy.deepcopy(state)
    rehashed["completed_batch_ids"] = [0]
    rehashed["payload_sha256"] = _hash(
        {key: item for key, item in rehashed.items() if key != "payload_sha256"}
    )
    checkpoint_path.write_text(
        json.dumps(rehashed, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(OrbitValidationError, match="external file"):
        load_checkpoint(
            checkpoint_path,
            expected_file_sha256=checkpoint_hash,
            expected_campaign_id="deterministic",
            expected_launches_sha256=launch_hash,
            expected_batch_manifest_sha256=batch_hash,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
            expected_previous_checkpoint_sha256="0"*64,
        )
