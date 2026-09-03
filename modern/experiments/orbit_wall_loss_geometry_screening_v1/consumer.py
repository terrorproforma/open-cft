"""First formal consumer of the orbit_mc coupling-v4.2 export format.

The v4 campaign published ``coupling-export-only.json`` as *export only*, with
no consumer. This module consumes that format: it re-derives every derived
quantity of a handoff (Wilson interval, binomial standard uncertainty, success
count), checks the closed key set and constants of
``modern/spec/orbit_mc/coupling-v4.2-handoff-v1.schema.json``, binds the handoff
to the sealed orbit artifact hash, and emits a consumer record. It is used in
two places:

* every screening design's handoff (produced by ``coupling_v42_handoff`` in the
  sealing worker) is consumed before it enters the geometry dataset;
* the accepted v4 export for the divergent-exit design is consumed as the
  accepted-evidence reference row (that design is not a sweep-v2 design, so it
  is absent from the screening set; the record says so).
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from cft_revival.experiment_runtime.canonical import strict_json_loads
from cft_revival.orbit_mc import HANDOFF_VERSION, wilson_interval
from cft_revival.orbit_mc.artifacts import content_hash

HANDOFF_KEYS = {
    "schema_version",
    "classification",
    "quantity",
    "probability",
    "standard_uncertainty",
    "confidence_interval_95",
    "trial_count",
    "orbit_result_artifact_sha256",
    "result_identity_sha256",
    "batch_manifest_sha256",
    "verification_status",
    "estimator_policy",
    "coupling_target",
    "integration_status",
    "plasma_network_role",
}
HANDOFF_CONSTANTS = {
    "schema_version": HANDOFF_VERSION,
    "classification": "test_particle_wall_loss_not_self_consistent_plasma",
    "quantity": "electron_dielectric_wall_loss_probability",
    "verification_status": "deterministic_replay_verified",
    "estimator_policy": "unweighted_binomial",
    "coupling_target": "cft-field-plasma-coupling/4.2.0",
    "integration_status": "export_only_pending_consumer_integration",
    "plasma_network_role": "export_only_pending_integration",
}
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
CONSUMER_SCHEMA = "cft-revival.orbit-wall-loss-geometry-screening-v1.consumed-handoff/1.0.0"


class HandoffConsumerError(ValueError):
    """The handoff violates the coupling-v4.2 export contract."""


def _fail(message: str) -> None:
    raise HandoffConsumerError(message)


def verify_handoff(handoff: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive and check every quantity of a coupling-v4.2 handoff."""

    if not isinstance(handoff, Mapping) or set(handoff) != HANDOFF_KEYS:
        _fail("handoff key set is not the closed coupling-v4.2 set")
    for key, expected in HANDOFF_CONSTANTS.items():
        if handoff[key] != expected:
            _fail(f"handoff {key} differs from the schema constant")
    for key in ("orbit_result_artifact_sha256", "result_identity_sha256", "batch_manifest_sha256"):
        if not isinstance(handoff[key], str) or not HEX_64.match(handoff[key]):
            _fail(f"handoff {key} is not lowercase SHA-256")
    trials = handoff["trial_count"]
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        _fail("handoff trial_count must be a positive integer")
    probability = handoff["probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0.0 <= float(probability) <= 1.0:
        _fail("handoff probability must lie in [0, 1]")
    successes = round(float(probability) * trials)
    if successes / trials != float(probability):
        _fail("handoff probability is not an exact binomial fraction of trial_count")
    expected = wilson_interval(int(successes), int(trials))
    interval = handoff["confidence_interval_95"]
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in interval)
        or float(interval[0]) != expected.lower
        or float(interval[1]) != expected.upper
    ):
        _fail("handoff confidence_interval_95 does not reproduce the Wilson interval")
    standard = math.sqrt(float(probability) * (1.0 - float(probability)) / trials)
    declared = handoff["standard_uncertainty"]
    if isinstance(declared, bool) or not isinstance(declared, (int, float)) or float(declared) != standard:
        _fail("handoff standard_uncertainty does not reproduce sqrt(p(1-p)/n)")
    return {
        "successes": int(successes),
        "trials": int(trials),
        "probability": float(probability),
        "wilson_lower": expected.lower,
        "wilson_upper": expected.upper,
        "standard_uncertainty": standard,
    }


def consume_handoff(
    handoff: Mapping[str, Any],
    *,
    expected_artifact_sha256: str,
    design_label: str,
    evidence_class: str = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS",
) -> dict[str, Any]:
    """Verify a handoff and bind it to the sealed orbit artifact it came from."""

    derived = verify_handoff(handoff)
    if handoff["orbit_result_artifact_sha256"] != expected_artifact_sha256:
        _fail("handoff orbit_result_artifact_sha256 is not the sealed artifact hash")
    return {
        "schema_version": CONSUMER_SCHEMA,
        "design_label": design_label,
        "evidence_class": evidence_class,
        "handoff_sha256": content_hash(dict(handoff)),
        "orbit_result_artifact_sha256": handoff["orbit_result_artifact_sha256"],
        "result_identity_sha256": handoff["result_identity_sha256"],
        "batch_manifest_sha256": handoff["batch_manifest_sha256"],
        "derived": derived,
        "checks": {
            "closed_schema": True,
            "constants": True,
            "hash_patterns": True,
            "exact_binomial_fraction": True,
            "wilson_interval_reproduced": True,
            "standard_uncertainty_reproduced": True,
            "bound_to_sealed_artifact": True,
        },
        "passed": True,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def consume_v4_export(protocol: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    """Consume the accepted v4 export (divergent-exit design) as the labelled reference row."""

    declaration = protocol["coupling_consumer"]
    export_path = repository / declaration["v4_export_path"]
    sidecar_path = repository / declaration["v4_export_sidecar_path"]
    orbit_sidecar_path = repository / declaration["v4_orbit_artifact_sidecar_path"]
    data = export_path.read_bytes()
    file_sha = hashlib.sha256(data).hexdigest()
    if file_sha != declaration["v4_export_file_sha256"]:
        _fail("v4 export bytes differ from the declared authority")
    sidecar = strict_json_loads(sidecar_path.read_bytes())
    if sidecar["byte_sha256"] != file_sha or sidecar["artifact"] != "artifacts/coupling-export-only.json":
        _fail("v4 export sidecar does not attest the export bytes")
    handoff = strict_json_loads(data)
    orbit_sidecar = orbit_sidecar_path.read_text(encoding="ascii")
    expected_artifact_sha = orbit_sidecar.split()[0]
    consumed = consume_handoff(
        handoff,
        expected_artifact_sha256=expected_artifact_sha,
        design_label=declaration["v4_design_id"],
        evidence_class="collisionless_prescribed_field_test_particle_wall_loss_not_pic (accepted v4 evidence, NUMERICAL_P2_QUALIFIED field)",
    )
    return {
        "consumer_id": declaration["consumer_id"],
        "consumed_export_path": declaration["v4_export_path"],
        "consumed_export_file_sha256": file_sha,
        "v4_result_commit": declaration["v4_result_commit"],
        "design_id": declaration["v4_design_id"],
        "design_in_screening_set": bool(declaration["v4_design_in_screening_set"]),
        "absence_statement": declaration["v4_absence_statement"],
        "reference_row": {
            "design_id": declaration["v4_design_id"],
            "field_qualification": "NUMERICAL_P2_QUALIFIED",
            "evidence_class": "accepted_physical_orbit_evidence_v4",
            "probability": consumed["derived"]["probability"],
            "confidence_interval_95": [consumed["derived"]["wilson_lower"], consumed["derived"]["wilson_upper"]],
            "standard_uncertainty": consumed["derived"]["standard_uncertainty"],
            "trial_count": consumed["derived"]["trials"],
            "not_part_of_screening_dataset": True,
        },
        "consumed": consumed,
        "passed": True,
    }
