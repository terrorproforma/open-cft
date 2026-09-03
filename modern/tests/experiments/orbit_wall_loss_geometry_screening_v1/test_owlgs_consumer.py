"""The coupling-v4.2 handoff consumer: verification, tamper rejection, v4 reference consumption."""

from __future__ import annotations

import copy
import math

import pytest

from cft_revival.orbit_mc import wilson_interval

from experiments.orbit_wall_loss_geometry_screening_v1 import experiment as E
from experiments.orbit_wall_loss_geometry_screening_v1.consumer import (
    HandoffConsumerError,
    consume_handoff,
    consume_v4_export,
    verify_handoff,
)


def _handoff(successes: int = 330, trials: int = 512) -> dict:
    interval = wilson_interval(successes, trials)
    p = successes / trials
    return {
        "schema_version": "cft-revival-orbit-mc-coupling-v4.2/1.3.0",
        "classification": "test_particle_wall_loss_not_self_consistent_plasma",
        "quantity": "electron_dielectric_wall_loss_probability",
        "probability": p,
        "standard_uncertainty": math.sqrt(p * (1.0 - p) / trials),
        "confidence_interval_95": [interval.lower, interval.upper],
        "trial_count": trials,
        "orbit_result_artifact_sha256": "a" * 64,
        "result_identity_sha256": "b" * 64,
        "batch_manifest_sha256": "c" * 64,
        "verification_status": "deterministic_replay_verified",
        "estimator_policy": "unweighted_binomial",
        "coupling_target": "cft-field-plasma-coupling/4.2.0",
        "integration_status": "export_only_pending_consumer_integration",
        "plasma_network_role": "export_only_pending_integration",
    }


def test_valid_handoff_is_verified_and_bound() -> None:
    derived = verify_handoff(_handoff())
    assert derived["successes"] == 330 and derived["trials"] == 512
    record = consume_handoff(_handoff(), expected_artifact_sha256="a" * 64, design_label="x")
    assert record["passed"] is True
    assert all(record["checks"].values())
    assert record["evidence_class"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda h: h.update(extra=1), "closed"),
        (lambda h: h.update(classification="plasma"), "schema constant"),
        (lambda h: h.update(orbit_result_artifact_sha256="A" * 64), "SHA-256"),
        (lambda h: h.update(trial_count=0), "positive integer"),
        (lambda h: h.update(probability=1.5), "\\[0, 1\\]"),
        (lambda h: h.update(probability=0.3), "exact binomial fraction"),
        (lambda h: h.update(confidence_interval_95=[0.6, 0.7]), "Wilson"),
        (lambda h: h.update(standard_uncertainty=0.02), "standard_uncertainty"),
    ],
)
def test_tampered_handoffs_are_rejected(mutate, message: str) -> None:
    handoff = _handoff()
    mutate(handoff)
    with pytest.raises(HandoffConsumerError, match=message):
        verify_handoff(handoff)


def test_consumer_refuses_an_unbound_artifact_hash() -> None:
    with pytest.raises(HandoffConsumerError, match="sealed artifact hash"):
        consume_handoff(_handoff(), expected_artifact_sha256="f" * 64, design_label="x")


def test_v4_export_is_consumed_as_the_absent_reference_design() -> None:
    value = E.protocol()
    record = consume_v4_export(value, E.REPOSITORY)
    assert record["passed"] is True
    assert record["design_in_screening_set"] is False
    assert record["design_id"] == "divergent-exit-stack"
    assert record["reference_row"]["probability"] == 330 / 512
    assert record["reference_row"]["trial_count"] == 512
    assert record["reference_row"]["not_part_of_screening_dataset"] is True
    assert record["reference_row"]["field_qualification"] == "NUMERICAL_P2_QUALIFIED"
    assert "absent" in record["absence_statement"]
    assert record["consumed"]["orbit_result_artifact_sha256"] == (
        "ccb85bd7738584badac5f046a5a61a21ea193d27d4dacd069c803f8387110cf1"
    )


def test_v4_export_consumption_fails_closed_on_authority_drift() -> None:
    value = copy.deepcopy(E.protocol())
    value["coupling_consumer"]["v4_export_file_sha256"] = "0" * 64
    with pytest.raises(HandoffConsumerError, match="declared authority"):
        consume_v4_export(value, E.REPOSITORY)
