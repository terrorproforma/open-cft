"""Replay all checked-in FEM-reference artifacts and file anchors."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from cft_revival.fem_reference import replay_artifact, validate_artifact


def main() -> None:
    root = Path(__file__).resolve().parent / "artifacts"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "integrity"
    }
    manifest_hash = sha256(
        json.dumps(
            manifest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if manifest["integrity"]["payload_sha256"] != manifest_hash:
        raise RuntimeError("campaign manifest payload hash differs")
    for entry in manifest["designs"]:
        previous_file_hash = "0" * 64
        previous_mesh_hash = "0" * 64
        for anchor in entry["checkpoints"]:
            checkpoint_path = root / anchor["file"]
            checkpoint_bytes = checkpoint_path.read_bytes()
            file_hash = sha256(checkpoint_bytes).hexdigest()
            checkpoint = json.loads(checkpoint_bytes)
            if file_hash != anchor["file_sha256"]:
                raise RuntimeError(
                    f"manifest checkpoint file hash differs: {checkpoint_path.name}"
                )
            if (
                checkpoint["integrity"]["payload_sha256"]
                != anchor["payload_sha256"]
                or checkpoint["mesh_sha256"] != anchor["mesh_sha256"]
                or checkpoint["parent_mesh_sha256"]
                != anchor["parent_mesh_sha256"]
                or checkpoint["previous_checkpoint_file_sha256"]
                != previous_file_hash
                or anchor["previous_checkpoint_file_sha256"]
                != previous_file_hash
            ):
                raise RuntimeError(
                    f"manifest checkpoint authority differs: {checkpoint_path.name}"
                )
            if anchor["level"] > 0 and anchor["parent_mesh_sha256"] != previous_mesh_hash:
                raise RuntimeError(
                    f"checkpoint mesh ancestry differs: {checkpoint_path.name}"
                )
            previous_file_hash = file_hash
            previous_mesh_hash = anchor["mesh_sha256"]
        artifact_path = root / entry["artifact"]
        viewer_path = root / entry["viewer"]
        artifact_bytes = artifact_path.read_bytes()
        viewer_bytes = viewer_path.read_bytes()
        if sha256(artifact_bytes).hexdigest() != entry["artifact_file_sha256"]:
            raise RuntimeError(f"artifact file hash differs: {artifact_path.name}")
        if sha256(viewer_bytes).hexdigest() != entry["viewer_file_sha256"]:
            raise RuntimeError(f"viewer file hash differs: {viewer_path.name}")
        artifact = json.loads(artifact_bytes)
        validate_artifact(artifact)
        if not replay_artifact(artifact)["passed"]:
            raise RuntimeError(f"artifact replay differs: {artifact_path.name}")
        viewer = json.loads(viewer_bytes)
        if viewer["artifact_payload_sha256"] != artifact["integrity"]["payload_sha256"]:
            raise RuntimeError(f"viewer anchor differs: {viewer_path.name}")
    checkpoints = sorted((root / "checkpoints").glob("*.json"))
    expected_checkpoints = sum(len(entry["checkpoints"]) for entry in manifest["designs"])
    if len(checkpoints) != expected_checkpoints:
        raise RuntimeError("adaptive checkpoint count differs")
    for checkpoint_path in checkpoints:
        checkpoint = json.loads(checkpoint_path.read_bytes())
        payload = {
            key: value for key, value in checkpoint.items() if key != "integrity"
        }
        expected = sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        if checkpoint["integrity"]["payload_sha256"] != expected:
            raise RuntimeError(
                f"adaptive checkpoint hash differs: {checkpoint_path.name}"
            )
    print(
        f"replayed {len(manifest['designs'])} FEM-reference artifacts "
        f"and {len(checkpoints)} adaptive checkpoints"
    )


if __name__ == "__main__":
    main()
