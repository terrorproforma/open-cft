import json
from hashlib import sha256
from pathlib import Path

import pytest

from cft_revival.geometry import (
    GeometryValidationError,
    canonical_json,
    load_artifact_bundle,
)
from examples.geometry.generate_reference_artifacts import regenerate


def rewrite_sidecar(path: Path) -> str:
    digest = sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def rewrite_manifest(directory: Path, manifest: dict) -> None:
    path = directory / "manifest.json"
    path.write_text(canonical_json(manifest), encoding="utf-8", newline="\n")
    rewrite_sidecar(path)


def test_bundle_loader_verifies_all_files_payloads_schemas_and_sidecars(
    tmp_path: Path,
) -> None:
    regenerate(tmp_path)
    loaded = load_artifact_bundle(tmp_path)
    assert len(loaded.geometries) == 3
    assert {
        geometry.config_id for geometry in loaded.geometries
    } == {
        "historical-envelope-baseline-v1",
        "compact-high-gradient-stack-v1",
        "divergent-exit-stack-v1",
    }
    for geometry in loaded.geometries:
        path = tmp_path / f"{geometry.config_id.removesuffix('-v1')}.json"
        assert path.read_text(encoding="utf-8") == canonical_json(geometry.to_dict())
        assert not path.read_bytes().endswith(b"\n")


def test_bundle_loader_rejects_file_tamper_and_unmanifested_substitution(
    tmp_path: Path,
) -> None:
    regenerate(tmp_path)
    svg = tmp_path / "historical-envelope-baseline.svg"
    svg.write_bytes(svg.read_bytes() + b" ")
    with pytest.raises(GeometryValidationError, match="SHA-256"):
        load_artifact_bundle(tmp_path)

    regenerate(tmp_path)
    (tmp_path / "unmanifested.json").write_text("{}", encoding="utf-8")
    with pytest.raises(GeometryValidationError, match="unmanifested"):
        load_artifact_bundle(tmp_path)


def test_bundle_loader_rejects_closed_viewer_tamper_even_with_rehashed_files(
    tmp_path: Path,
) -> None:
    regenerate(tmp_path)
    viewer_name = "historical-envelope-baseline.viewer.json"
    viewer_path = tmp_path / viewer_name
    viewer = json.loads(viewer_path.read_text(encoding="utf-8"))
    viewer["external_components"][0]["injected"] = "<script>"
    viewer_path.write_text(canonical_json(viewer), encoding="utf-8", newline="\n")
    viewer_digest = rewrite_sidecar(viewer_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configurations"][0]["artifact_file_sha256"][
        viewer_name
    ] = viewer_digest
    rewrite_manifest(tmp_path, manifest)
    with pytest.raises(GeometryValidationError, match="closed projection"):
        load_artifact_bundle(tmp_path)


def test_bundle_loader_rejects_duplicate_keys_and_traversal_paths(
    tmp_path: Path,
) -> None:
    regenerate(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    tampered = text.replace(
        '"claim_limit":',
        '"claim_limit":"duplicate","claim_limit":',
        1,
    )
    manifest_path.write_text(tampered, encoding="utf-8", newline="\n")
    rewrite_sidecar(manifest_path)
    with pytest.raises(GeometryValidationError, match="duplicate JSON key"):
        load_artifact_bundle(tmp_path)

    regenerate(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest["configurations"][0]["artifact_file_sha256"]
    digest = hashes.pop("historical-envelope-baseline.svg")
    hashes["../historical-envelope-baseline.svg"] = digest
    rewrite_manifest(tmp_path, manifest)
    with pytest.raises(GeometryValidationError, match="artifact hashes|canonical"):
        load_artifact_bundle(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("generator", "unknown-generator", "identity"),
        ("generator_version", "99.0.0", "version"),
        (
            "claim_limit",
            "Build-qualified; performance proven",
            "evidence boundary",
        ),
    ),
)
def test_bundle_loader_enforces_generator_and_claim_semantic_allowlist(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    regenerate(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    rewrite_manifest(tmp_path, manifest)
    with pytest.raises(GeometryValidationError, match=message):
        load_artifact_bundle(tmp_path)
