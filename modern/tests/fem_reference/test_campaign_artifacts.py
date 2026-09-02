from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from cft_revival.fem_reference import replay_artifact, validate_artifact


def test_campaign_artifacts_and_viewers_replay(monkeypatch) -> None:
    monkeypatch.setattr(
        "cft_revival.fem_reference.resource_policy.available_ram_bytes",
        lambda: 16 * 1024**3,
    )
    modern = Path(__file__).resolve().parents[2]
    root = modern / "examples" / "fem_reference" / "artifacts"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "integrity"
    }
    assert manifest["integrity"]["payload_sha256"] == sha256(
        json.dumps(
            manifest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert len(manifest["designs"]) == 3
    assert manifest["artifact_authority"].startswith("legacy_v1.1_integrity_only")
    assert manifest["maximum_p2_dofs"] >= 1_500_000
    assert not manifest["resource_policy_revision"]["accuracy_gates_relaxed"]
    assert (
        manifest["resource_policy_revision"]["minimum_third_level_free_ram_bytes"]
        == 8 * 1024**3
    )
    assert manifest["resource_policy_revision"]["one_design_at_a_time"]
    assert manifest["domain_expansion_evidence"]["status"] == "not_run_screening_only"
    assert manifest["less_than_one_percent_all_designs"] == all(
        entry["convergence"]["less_than_one_percent_reached"]
        for entry in manifest["designs"]
    )
    for entry in manifest["designs"]:
        previous_file_hash = "0" * 64
        previous_mesh_hash = "0" * 64
        for anchor in entry["checkpoints"]:
            checkpoint_bytes = (root / anchor["file"]).read_bytes()
            assert sha256(checkpoint_bytes).hexdigest() == anchor["file_sha256"]
            checkpoint = json.loads(checkpoint_bytes)
            assert checkpoint["integrity"]["payload_sha256"] == anchor["payload_sha256"]
            assert checkpoint["previous_checkpoint_file_sha256"] == previous_file_hash
            assert anchor["previous_checkpoint_file_sha256"] == previous_file_hash
            if anchor["level"] > 0:
                assert anchor["parent_mesh_sha256"] == previous_mesh_hash
            previous_file_hash = anchor["file_sha256"]
            previous_mesh_hash = anchor["mesh_sha256"]
        first_anchor = entry["checkpoints"][0]
        rehashed_checkpoint = json.loads((root / first_anchor["file"]).read_bytes())
        rehashed_checkpoint["run"]["iterations"] += 1
        checkpoint_payload = {
            key: value
            for key, value in rehashed_checkpoint.items()
            if key != "integrity"
        }
        rehashed_checkpoint["integrity"]["payload_sha256"] = sha256(
            json.dumps(
                checkpoint_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        assert sha256(
            json.dumps(
                rehashed_checkpoint,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest() != first_anchor["file_sha256"]
        artifact_bytes = (root / entry["artifact"]).read_bytes()
        viewer_bytes = (root / entry["viewer"]).read_bytes()
        assert sha256(artifact_bytes).hexdigest() == entry["artifact_file_sha256"]
        assert sha256(viewer_bytes).hexdigest() == entry["viewer_file_sha256"]
        artifact = json.loads(artifact_bytes)
        validate_artifact(artifact)
        replay = replay_artifact(artifact)
        assert replay["passed"]
        assert replay["acceptance_authority"] == "legacy_integrity_only_screening"
        viewer = json.loads(viewer_bytes)
        assert viewer["artifact_payload_sha256"] == artifact["integrity"]["payload_sha256"]
        for comparison in entry["l1b_comparison"].values():
            assert comparison["identical_qoi_semantics"]

    checkpoints = sorted((root / "checkpoints").glob("*.json"))
    assert len(checkpoints) == sum(
        entry["convergence"]["completed_adaptive_levels"]
        for entry in manifest["designs"]
    )
    for path in checkpoints:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in checkpoint.items() if key != "integrity"}
        assert checkpoint["integrity"]["payload_sha256"] == sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()


def test_owned_specs_are_valid_json_and_claim_limited() -> None:
    modern = Path(__file__).resolve().parents[2]
    for path in (modern / "spec" / "fem_reference").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        assert "hardware_validation" in encoded or "schema" in encoded
