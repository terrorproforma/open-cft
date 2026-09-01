from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from cft_revival.material_fields import (
    MaterialFieldValidationError,
    validate_artifact,
    validate_artifact_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def test_generated_artifact_sidecar_hashes_match_exact_bytes() -> None:
    artifact_root = ROOT / "examples/material_fields/artifacts"
    for path in sorted(artifact_root.glob("*.json")):
        sidecar = path.with_name(path.name + ".sha256")
        declared_hash, declared_name = sidecar.read_text(encoding="ascii").split()
        assert declared_name == path.name
        assert declared_hash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_equation_ledger_closes_sign_interface_and_authority_rules() -> None:
    ledger = json.loads(
        (ROOT / "spec/material_fields/equation-ledger-v1.json").read_text(encoding="utf-8")
    )
    assert ledger["unknown"]["definition"] == "psi(r,z)=r*A_phi(r,z)"
    assert ledger["strong_form"]["equation"].startswith("-div((nu/r)*grad(psi))")
    assert ledger["discrete_form"]["face_reluctivity"].startswith("radial nu_f=")
    assert "never both" in ledger["weak_form"]["pm_authority"]
    assert ledger["interface_conditions"]["normal_B"].endswith("=0")
    assert "K_free_phi" in ledger["interface_conditions"]["tangential_H"]
    assert "not assumed" in ledger["discrete_form"]["matrix"]
    assert "minimum-eigenvalue" in ledger["discrete_form"]["matrix"]


def test_viewer_schema_is_closed_and_hash_anchored() -> None:
    schema = json.loads(
        (ROOT / "spec/material_fields/viewer-v1.schema.json").read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert {"artifact_payload_sha256", "integrity", "field_map"} <= set(schema["required"])
    assert schema["properties"]["classification"]["const"].startswith("hypothetical")


def test_result_schema_is_closed_and_generated_screening_is_not_publishable() -> None:
    schema = json.loads(
        (ROOT / "spec/material_fields/result-v1.schema.json").read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["acceptance"]["additionalProperties"] is False
    artifact = json.loads(
        (
            ROOT
            / "examples/material_fields/artifacts/"
            "historical-envelope-baseline.material-field.json"
        ).read_text(encoding="utf-8")
    )
    validate_artifact(artifact, require_accepted=False)
    with pytest.raises(MaterialFieldValidationError, match="not publication evidence"):
        validate_artifact(artifact)


def test_bundle_replay_passes_and_rejects_stale_version_mixing(tmp_path: Path) -> None:
    artifact_root = ROOT / "examples/material_fields/artifacts"
    validate_artifact_bundle(artifact_root)
    copied = tmp_path / "artifacts"
    shutil.copytree(artifact_root, copied)
    stale_path = copied / "divergent-exit-stack.material-field.json"
    stale = json.loads(stale_path.read_text(encoding="utf-8"))
    stale["schema_version"] = "cft_revival.material_fields.result/1.3.0"
    stale_path.write_text(
        json.dumps(stale, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(stale_path.read_bytes()).hexdigest()
    stale_path.with_name(stale_path.name + ".sha256").write_text(
        f"{digest}  {stale_path.name}\n", encoding="ascii"
    )
    with pytest.raises(MaterialFieldValidationError, match="schema/model"):
        validate_artifact_bundle(copied)


def _historical_artifact() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "examples/material_fields/artifacts/"
            "historical-envelope-baseline.material-field.json"
        ).read_text(encoding="utf-8")
    )


def test_reduced_resource_gates_are_all_explicitly_not_evaluated() -> None:
    artifact = _historical_artifact()
    gates = artifact["acceptance"]["gates"]
    assert len(gates) == 18
    assert {gate["status"] for gate in gates} == {"NOT_EVALUATED"}
    assert {
        gate["diagnostic_status"] for gate in gates
    } == {"MEASURED_REDUCED_RESOURCE_ONLY"}
    assert all(
        isinstance(gate["measured_value"], (int, float))
        and isinstance(gate["threshold"], (int, float))
        and "passed" not in gate
        for gate in gates
    )


def test_reduced_resource_gate_rejects_legacy_boolean_contract() -> None:
    artifact = _historical_artifact()
    legacy = copy.deepcopy(artifact)
    gate = legacy["acceptance"]["gates"][0]
    gate["passed"] = False
    del gate["status"]
    with pytest.raises(MaterialFieldValidationError, match="gates.*recomputed"):
        validate_artifact(legacy, require_accepted=False)


def test_reduced_resource_gate_rejects_forged_pass() -> None:
    artifact = _historical_artifact()
    forged = copy.deepcopy(artifact)
    forged["acceptance"]["gates"][1]["status"] = "PASS"
    with pytest.raises(MaterialFieldValidationError, match="gates.*recomputed"):
        validate_artifact(forged, require_accepted=False)


def test_viewer_uses_tri_state_without_publication_gate_payload() -> None:
    viewer = json.loads(
        (
            ROOT
            / "examples/material_fields/artifacts/"
            "historical-envelope-baseline.viewer.json"
        ).read_text(encoding="utf-8")
    )
    assert "gates" not in viewer
    assert {
        viewer["summary"][key]["mesh_gate_status"]
        for key in ("sampled_cell_peak", "axis_bz_peak")
    } == {"NOT_EVALUATED"}


def test_p2_comparison_handoff_carries_no_structured_gate_statuses() -> None:
    def all_keys(value: object):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from all_keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from all_keys(item)

    for path in sorted(
        (ROOT / "examples/fem_reference/artifacts").glob(
            "*.fem-reference.json"
        )
    ):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        comparison_keys = set(all_keys(artifact["comparisons"]))
        assert "gates" not in comparison_keys
        assert not any(key.endswith("_gate_status") for key in comparison_keys)
