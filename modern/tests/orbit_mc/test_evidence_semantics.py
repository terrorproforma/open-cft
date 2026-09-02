from __future__ import annotations

import copy
from hashlib import sha256
from math import pi

import numpy as np
import pytest

from cft_revival.orbit_mc import (
    AnalyticField,
    EstimatorPolicy,
    ElectronLaunch,
    VerifiedOrbitEvidence,
    OrbitConfig,
    OrbitValidationError,
    build_launch_ensemble,
    checkpoint,
    coupling_v42_handoff,
    frozen_batch_manifest,
    merge_checkpoint_results,
    result_artifact,
    run_ensemble,
    validate_checkpoint,
    validate_result_artifact,
    validate_result_replay,
    load_artifact,
    write_artifact,
)
from cft_revival.orbit_mc.artifacts import canonical_bytes, content_hash
from cft_revival.orbit_mc.ensemble import result_records_identity

POLICY_SHA256 = "d"*64
CERTIFICATE_FLOOR = 0.001
ESTIMATOR = EstimatorPolicy.UNWEIGHTED_BINOMIAL


def _batch_hash(batches) -> str:
    return content_hash(
        {"estimator_policy": ESTIMATOR.value, "batches": batches}
    )


def _evidence():
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    launches = build_launch_ensemble(
        ensemble_id="evidence",
        energies_ev=(10.0,),
        pitch_angles_rad=(0.3,),
        positions=(("core", (0.0, 0.0, 0.0)),),
        directions=(1,),
        gyrophase_count=2,
    )
    gyroperiod = 2.0*pi*9.1093837139e-31/(1.602176634e-19*0.1)
    config = OrbitConfig(
        0.1, -0.2, 0.2, 0.2, -0.3, 0.3, 1.5*gyroperiod, 1.0,
        max_steps=1000, max_rotation_rad=0.12,
    )
    results, summary = run_ensemble("evidence", launches, field, config)
    batches = frozen_batch_manifest(launches, batch_size=2)
    artifact = result_artifact(
        campaign_id="evidence",
        field_identity_sha256="a"*64,
        config_identity_sha256="b"*64,
        policy_identity_sha256=POLICY_SHA256,
        minimum_certificate_tightness_ratio_authority=CERTIFICATE_FLOOR,
        estimator_policy=ESTIMATOR,
        launches=launches,
        results=results,
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
            "protocol_id": "evidence-test",
            "frozen_before_outcomes": True,
            "held_out_geometry_status": "pending",
        },
    )
    state = checkpoint(
        "evidence",
        [0],
        launches,
        results,
        batches,
        field_identity_sha256="a"*64,
        config_identity_sha256="b"*64,
        policy_identity_sha256=POLICY_SHA256,
        minimum_certificate_tightness_ratio_authority=CERTIFICATE_FLOOR,
        estimator_policy=ESTIMATOR,
        expected_batch_manifest_sha256=_batch_hash(batches),
    )
    return artifact, state


def _reseal_artifact(artifact: dict[str, object]) -> None:
    artifact["identities"]["launches_sha256"] = content_hash(artifact["launches"])
    artifact["identities"]["results_sha256"] = content_hash(artifact["results"])
    policy = artifact["estimator"]["policy"]
    artifact["identities"]["batch_manifest_sha256"] = content_hash(
        {"estimator_policy": policy, "batches": artifact["batch_manifest"]}
    )
    weight_by_id = {
        entry["launch_id"]: entry["weight"]
        for batch in artifact["batch_manifest"]
        for entry in batch["launches"]
    }
    artifact["identities"]["estimator_sha256"] = content_hash(
        {
            "estimand_id": "campaign_wall_loss_probability",
            "policy": policy,
            "launches": [
                {"launch_id": launch_id, "weight": weight_by_id[launch_id]}
                for launch_id in sorted(weight_by_id)
            ],
        }
    )
    artifact["summary"]["result_identity_sha256"] = result_records_identity(
        artifact["results"]
    )
    artifact["integrity"]["payload_sha256"] = content_hash(
        {key: value for key, value in artifact.items() if key != "integrity"}
    )


def _reseal_checkpoint(state: dict[str, object]) -> None:
    state["authority"]["launches_sha256"] = content_hash(state["launches"])
    state["authority"]["results_sha256"] = content_hash(state["results"])
    state["authority"]["batch_manifest_sha256"] = content_hash(
        {
            "estimator_policy": state["authority"]["estimator_policy"],
            "batches": state["batch_manifest"],
        }
    )
    weight_by_id = {
        entry["launch_id"]: entry["weight"]
        for batch in state["batch_manifest"]
        for entry in batch["launches"]
    }
    state["authority"]["estimator_sha256"] = content_hash(
        {
            "estimand_id": "campaign_wall_loss_probability",
            "policy": state["authority"]["estimator_policy"],
            "launches": [
                {"launch_id": launch_id, "weight": weight_by_id[launch_id]}
                for launch_id in sorted(weight_by_id)
            ],
        }
    )
    state["authority"]["campaign_identity_sha256"] = content_hash(
        {
            "campaign_id": state["campaign_id"],
            "launches_sha256": state["authority"]["launches_sha256"],
            "batch_manifest_sha256": state["authority"]["batch_manifest_sha256"],
            "policy_sha256": state["authority"]["policy_sha256"],
            "minimum_certificate_tightness_ratio": state["authority"][
                "minimum_certificate_tightness_ratio"
            ],
            "estimator_policy": state["authority"]["estimator_policy"],
            "estimator_sha256": state["authority"]["estimator_sha256"],
            "replay_requirement": state["authority"]["replay_requirement"],
        }
    )
    state["payload_sha256"] = content_hash(
        {key: value for key, value in state.items() if key != "payload_sha256"}
    )


def _r(artifact):
    return artifact["results"][0]


def _forge_malformed_identity(artifact) -> None:
    launch_id = "evidence:forged"
    artifact["launches"][0]["launch_id"] = launch_id
    artifact["launches"][0]["seed_id"] = int.from_bytes(
        sha256(launch_id.encode()).digest()[:8], "big"
    )
    artifact["results"][0]["launch_id"] = launch_id


ARTIFACT_TAMPERS = [
    lambda a: _r(a).__setitem__("transit_fraction", 1.0000001),
    lambda a: _r(a).__setitem__("maximum_instantaneous_mu_relative_variation", -1.0),
    lambda a: _r(a).__setitem__("termination", "fabricated"),
    lambda a: _r(a).__setitem__("complete_gyrocycles", _r(a)["complete_gyrocycles"] + 1),
    lambda a: (_r(a).__setitem__("termination", "wall_hit"), _r(a).__setitem__("wall_endpoint_m", None)),
    lambda a: _r(a).__setitem__("elapsed_time_s", _r(a)["configured_max_time_s"] + _r(a)["dt_s"]),
    lambda a: _r(a).__setitem__("path_length_m", _r(a)["configured_max_path_m"] + 1.0),
    lambda a: (_r(a).__setitem__("final_energy_j", 2.0*_r(a)["initial_energy_j"]), _r(a).__setitem__("maximum_relative_energy_error", 0.0)),
    lambda a: _r(a).__setitem__("unknown_evidence", True),
    lambda a: _r(a).__setitem__("transit_fraction", "0.5"),
    lambda a: a["summary"]["wall_hit"].__setitem__("lower", 1.0),
    lambda a: a["summary"]["wall_hit"].__setitem__("successes", -1),
    lambda a: a["summary"]["termination_counts"].__setitem__("time_timeout", -1),
    lambda a: a["integrity"].__setitem__("algorithm", "sha512"),
    lambda a: a["interpolation_evidence"].__setitem__("certificate_tightness_ratio", 0.5),
    lambda a: a["launches"][0].__setitem__("seed_id", a["launches"][0]["seed_id"] + 1),
    lambda a: a["launches"][0].__setitem__("gyrophase_rad", 2.0*pi),
    lambda a: a["launches"][0].__setitem__("parallel_direction", 1.0),
    _forge_malformed_identity,
]


@pytest.mark.parametrize("tamper", ARTIFACT_TAMPERS)
def test_coherently_rehashed_artifact_semantic_lies_fail(tamper) -> None:
    artifact, _ = _evidence()
    forged = copy.deepcopy(artifact)
    tamper(forged)
    _reseal_artifact(forged)
    with pytest.raises(OrbitValidationError):
        validate_result_artifact(
            forged,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )


CHECKPOINT_TAMPERS = [
    lambda c: c["results"][0].__setitem__("launch_id", "evidence:forged"),
    lambda c: c["results"][0].__setitem__("termination", "fabricated"),
    lambda c: c["launches"].pop(),
    lambda c: c["launches"].append(copy.deepcopy(c["launches"][0])),
    lambda c: c["results"].pop(),
    lambda c: c["results"][0].__setitem__("transit_fraction", 1.1),
]


@pytest.mark.parametrize("tamper", CHECKPOINT_TAMPERS)
def test_checkpoint_tamper_matrix_uses_external_launch_authority(tamper) -> None:
    _, state = _evidence()
    trusted_launches = state["authority"]["launches_sha256"]
    forged = copy.deepcopy(state)
    tamper(forged)
    _reseal_checkpoint(forged)
    with pytest.raises(OrbitValidationError):
        validate_checkpoint(
            forged,
            expected_campaign_id="evidence",
            expected_launches_sha256=trusted_launches,
            expected_batch_manifest_sha256=state["authority"][
                "batch_manifest_sha256"
            ],
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "wall_hit", "reflected", "domain_escape", "path_timeout", "step_limit",
        "field_failure", "nonfinite_state", "extreme_relativity",
        "initial_state_invalid",
    ],
)
def test_valid_enum_event_substitutions_fail_after_rehash(replacement) -> None:
    artifact, _ = _evidence()
    forged = copy.deepcopy(artifact)
    forged["results"][0]["termination"] = replacement
    forged["results"][0]["event_witness"]["kind"] = replacement
    _reseal_artifact(forged)
    with pytest.raises(OrbitValidationError):
        validate_result_artifact(
            forged,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )


def _partial_components():
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    launches = build_launch_ensemble(
        ensemble_id="partial",
        energies_ev=(10.0,),
        pitch_angles_rad=(0.3,),
        positions=(("core", (0.0, 0.0, 0.0)),),
        directions=(1,),
        gyrophase_count=4,
    )
    gyroperiod = 2.0*pi*9.1093837139e-31/(1.602176634e-19*0.1)
    config = OrbitConfig(
        0.1, -0.2, 0.2, 0.2, -0.3, 0.3, 1.5*gyroperiod, 1.0,
        max_steps=1000, max_rotation_rad=0.12,
    )
    results, _ = run_ensemble("partial", launches, field, config)
    batches = frozen_batch_manifest(launches, batch_size=2)
    return launches, results, batches


def _partial_checkpoint(completed, results, launches, batches, **kwargs):
    return checkpoint(
        "partial",
        completed,
        launches,
        results,
        batches,
        field_identity_sha256="a"*64,
        config_identity_sha256="b"*64,
        policy_identity_sha256=POLICY_SHA256,
        minimum_certificate_tightness_ratio_authority=CERTIFICATE_FLOOR,
        estimator_policy=ESTIMATOR,
        expected_batch_manifest_sha256=_batch_hash(batches),
        **kwargs,
    )


def test_partial_checkpoint_and_resume_chain_cover_exact_manifest_prefixes() -> None:
    launches, results, batches = _partial_components()
    first = _partial_checkpoint([0], results[:2], launches, batches)
    assert first["coverage"]["completed_launches"] == 2
    assert first["pending_launch_ids"] == [item.launch_id for item in launches[2:]]
    second = _partial_checkpoint(
        [0],
        results[:3],
        launches,
        batches,
        partial_current_batch={
            "batch_id": 1,
            "completed_launch_ids": [launches[2].launch_id],
        },
        previous_checkpoint_sha256=content_hash(first),
    )
    assert second["coverage"]["completed_launches"] == 3
    final = _partial_checkpoint(
        [0, 1],
        results,
        launches,
        batches,
        previous_checkpoint_sha256=content_hash(second),
    )
    assert final["pending_launch_ids"] == []
    assert final["coverage"]["completed_launches"] == 4
    authority = {
        "expected_campaign_id": "partial",
        "expected_launches_sha256": first["authority"]["launches_sha256"],
        "expected_batch_manifest_sha256": first["authority"][
            "batch_manifest_sha256"
        ],
        "expected_policy_sha256": POLICY_SHA256,
        "expected_estimator_policy": ESTIMATOR,
        "expected_minimum_certificate_tightness_ratio": CERTIFICATE_FLOOR,
    }
    assert len(merge_checkpoint_results(first, second, **authority)) == 3
    assert len(merge_checkpoint_results(second, final, **authority)) == 4
    dropped = _partial_checkpoint(
        [1],
        results[2:],
        launches,
        batches,
        previous_checkpoint_sha256=content_hash(first),
    )
    with pytest.raises(OrbitValidationError, match="dropped"):
        merge_checkpoint_results(first, dropped, **authority)


@pytest.mark.parametrize("defect", ["unknown_batch", "missing", "duplicate", "bad_partial"])
def test_partial_checkpoint_batch_corruption_fails(defect) -> None:
    launches, results, batches = _partial_components()
    with pytest.raises(OrbitValidationError):
        if defect == "unknown_batch":
            _partial_checkpoint([999], results[:2], launches, batches)
        elif defect == "missing":
            _partial_checkpoint([0], results[:1], launches, batches)
        elif defect == "duplicate":
            _partial_checkpoint([0], (results[0], results[0]), launches, batches)
        else:
            _partial_checkpoint(
                [],
                results[1:2],
                launches,
                batches,
                partial_current_batch={
                    "batch_id": 0,
                    "completed_launch_ids": [launches[1].launch_id],
                },
            )


def test_external_certificate_floor_rejects_rehashed_weakening() -> None:
    artifact, state = _evidence()
    artifact["interpolation_evidence"]["minimum_certificate_tightness_ratio"] = 0.0
    _reseal_artifact(artifact)
    with pytest.raises(OrbitValidationError, match="authority|bound evidence"):
        validate_result_artifact(
            artifact,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )
    trusted_batch = state["authority"]["batch_manifest_sha256"]
    trusted_launches = state["authority"]["launches_sha256"]
    state["authority"]["minimum_certificate_tightness_ratio"] = 0.0
    _reseal_checkpoint(state)
    with pytest.raises(OrbitValidationError, match="floor"):
        validate_checkpoint(
            state,
            expected_campaign_id="evidence",
            expected_launches_sha256=trusted_launches,
            expected_batch_manifest_sha256=trusted_batch,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )


def test_deterministic_replay_rejects_semantically_plausible_rehashed_record() -> None:
    artifact, _ = _evidence()
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    gyroperiod = 2.0*pi*9.1093837139e-31/(1.602176634e-19*0.1)
    config = OrbitConfig(
        0.1, -0.2, 0.2, 0.2, -0.3, 0.3, 1.5*gyroperiod, 1.0,
        max_steps=1000, max_rotation_rad=0.12,
    )
    validate_result_replay(
        artifact,
        field=field,
        config=config,
        expected_field_sha256="a"*64,
        expected_config_sha256="b"*64,
        expected_launches_sha256=artifact["identities"]["launches_sha256"],
        expected_batch_manifest_sha256=artifact["identities"][
            "batch_manifest_sha256"
        ],
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    forged = copy.deepcopy(artifact)
    forged["results"][0]["reason"] = "plausible but not replayed"
    _reseal_artifact(forged)
    validate_result_artifact(
        forged,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    with pytest.raises(OrbitValidationError, match="replay"):
        validate_result_replay(
            forged,
            field=field,
            config=config,
            expected_field_sha256="a"*64,
            expected_config_sha256="b"*64,
            expected_launches_sha256=artifact["identities"]["launches_sha256"],
            expected_batch_manifest_sha256=artifact["identities"][
                "batch_manifest_sha256"
            ],
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )


def test_unweighted_estimator_rejects_unequal_and_unsupported_weights() -> None:
    launches, _, _ = _partial_components()
    unequal = {
        launch.launch_id: (0.99 if index == 0 else 0.01/3.0)
        for index, launch in enumerate(launches)
    }
    with pytest.raises(OrbitValidationError, match="equal launch weights"):
        frozen_batch_manifest(launches, batch_size=2, weights=unequal)
    with pytest.raises(OrbitValidationError, match="unsupported"):
        frozen_batch_manifest(
            launches,
            batch_size=2,
            estimator_policy="weighted_stratified",  # type: ignore[arg-type]
        )


def test_coherently_rehashed_unequal_estimator_weights_fail() -> None:
    artifact, state = _evidence()
    artifact["batch_manifest"][0]["launches"][0]["weight"] = 0.99
    artifact["batch_manifest"][0]["launches"][1]["weight"] = 0.01
    _reseal_artifact(artifact)
    with pytest.raises(OrbitValidationError, match="equal launch weights"):
        validate_result_artifact(
            artifact,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )
    trusted_launches = state["authority"]["launches_sha256"]
    state["batch_manifest"][0]["launches"][0]["weight"] = 0.99
    state["batch_manifest"][0]["launches"][1]["weight"] = 0.01
    _reseal_checkpoint(state)
    with pytest.raises(OrbitValidationError):
        validate_checkpoint(
            state,
            expected_campaign_id="evidence",
            expected_launches_sha256=trusted_launches,
            expected_batch_manifest_sha256=state["authority"][
                "batch_manifest_sha256"
            ],
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )


def test_equal_weight_input_permutations_have_one_estimator_identity() -> None:
    launches, results, batches = _partial_components()
    equal = {launch.launch_id: 0.25 for launch in reversed(launches)}
    permuted = frozen_batch_manifest(
        tuple(reversed(launches)), batch_size=2, weights=equal
    )
    assert permuted == batches
    first = _partial_checkpoint([0, 1], results, launches, batches)
    second = _partial_checkpoint(
        [0, 1], tuple(reversed(results)), tuple(reversed(launches)), permuted
    )
    assert first["authority"]["estimator_sha256"] == second["authority"][
        "estimator_sha256"
    ]
    assert first["authority"]["campaign_identity_sha256"] == second["authority"][
        "campaign_identity_sha256"
    ]


def test_coherent_one_to_two_batch_repartition_fails_external_authority() -> None:
    artifact, _ = _evidence()
    original_batch_hash = artifact["identities"]["batch_manifest_sha256"]
    launches = tuple(ElectronLaunch(**record) for record in artifact["launches"])
    artifact["batch_manifest"] = frozen_batch_manifest(launches, batch_size=1)
    _reseal_artifact(artifact)
    validate_result_artifact(
        artifact,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    gyroperiod = 2.0*pi*9.1093837139e-31/(1.602176634e-19*0.1)
    config = OrbitConfig(
        0.1, -0.2, 0.2, 0.2, -0.3, 0.3, 1.5*gyroperiod, 1.0,
        max_steps=1000, max_rotation_rad=0.12,
    )
    with pytest.raises(OrbitValidationError, match="batch manifest"):
        validate_result_replay(
            artifact,
            field=field,
            config=config,
            expected_field_sha256="a"*64,
            expected_config_sha256="b"*64,
            expected_launches_sha256=artifact["identities"]["launches_sha256"],
            expected_batch_manifest_sha256=original_batch_hash,
            expected_policy_sha256=POLICY_SHA256,
            expected_estimator_policy=ESTIMATOR,
            expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
        )
    results, _ = run_ensemble("evidence", launches, field, config)
    with pytest.raises(OrbitValidationError, match="batch manifest"):
        checkpoint(
            "evidence",
            [0, 1],
            launches,
            results,
            artifact["batch_manifest"],
            field_identity_sha256="a"*64,
            config_identity_sha256="b"*64,
            policy_identity_sha256=POLICY_SHA256,
            minimum_certificate_tightness_ratio_authority=CERTIFICATE_FLOOR,
            estimator_policy=ESTIMATOR,
            expected_batch_manifest_sha256=original_batch_hash,
        )


@pytest.mark.parametrize("false_label", ["extreme_relativity", "reflected"])
def test_false_labels_fail_before_write_and_after_structural_load(
    tmp_path, false_label
) -> None:
    artifact, _ = _evidence()
    forged = copy.deepcopy(artifact)
    forged["results"][0]["termination"] = false_label
    forged["results"][0]["event_witness"]["kind"] = false_label
    _reseal_artifact(forged)
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    gyroperiod = 2.0*pi*9.1093837139e-31/(1.602176634e-19*0.1)
    config = OrbitConfig(
        0.1, -0.2, 0.2, 0.2, -0.3, 0.3, 1.5*gyroperiod, 1.0,
        max_steps=1000, max_rotation_rad=0.12,
    )
    destination = tmp_path/f"{false_label}.json"
    authority = {
        "expected_policy_sha256": POLICY_SHA256,
        "expected_estimator_policy": ESTIMATOR,
        "expected_minimum_certificate_tightness_ratio": CERTIFICATE_FLOOR,
    }
    with pytest.raises(OrbitValidationError):
        write_artifact(
            destination,
            forged,
            field=field,
            config=config,
            expected_field_sha256="a"*64,
            expected_config_sha256="b"*64,
            expected_launches_sha256=artifact["identities"]["launches_sha256"],
            expected_batch_manifest_sha256=artifact["identities"][
                "batch_manifest_sha256"
            ],
            **authority,
        )
    assert not destination.exists()
    data = canonical_bytes(forged)
    destination.write_bytes(data)
    digest = __import__("hashlib").sha256(data).hexdigest()
    destination.with_name(destination.name+".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii"
    )
    with pytest.raises(OrbitValidationError):
        load_artifact(
            destination,
            expected_file_sha256=digest,
            **authority,
        )


def test_structural_artifact_cannot_feed_coupling(tmp_path) -> None:
    artifact, _ = _evidence()
    path = tmp_path/"structural.json"
    data = canonical_bytes(artifact)
    path.write_bytes(data)
    digest = __import__("hashlib").sha256(data).hexdigest()
    path.with_name(path.name+".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    structural = load_artifact(
        path,
        expected_file_sha256=digest,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    with pytest.raises(TypeError):
        _ = structural["summary"]  # type: ignore[index]
    with pytest.raises(OrbitValidationError, match="verified"):
        coupling_v42_handoff(
            structural,  # type: ignore[arg-type]
            expected_batch_manifest_sha256=artifact["identities"][
                "batch_manifest_sha256"
            ],
        )
    with pytest.raises(OrbitValidationError, match="deterministic replay"):
        VerifiedOrbitEvidence(data, digest, object())
