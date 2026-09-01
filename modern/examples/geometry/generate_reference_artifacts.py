"""Regenerate deterministic geometry JSON, viewer data, SVG, and hashes."""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

MODERN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODERN / "src"))

from cft_revival.geometry import (  # noqa: E402
    ARTIFACT_CLAIM_LIMIT,
    ARTIFACT_GENERATOR_ID,
    ARTIFACT_GENERATOR_VERSION,
    canonical_json,
    compute_descriptors,
    reference_variants,
    write_reference_artifacts,
)


def regenerate(output_directory: Path | None = None) -> dict[str, object]:
    output = (
        Path(__file__).resolve().parent / "artifacts"
        if output_directory is None
        else output_directory
    )
    output.mkdir(parents=True, exist_ok=True)
    configurations: list[dict[str, object]] = []
    for geometry in reference_variants():
        files = write_reference_artifacts(geometry, output)
        configurations.append(
            {
                "config_id": geometry.config_id,
                "geometry_payload_sha256": geometry.canonical_sha256,
                "dimensions": {
                    "chamber_inner_radius_m": geometry.chamber.inner_radius_m,
                    "chamber_outer_radius_m": geometry.chamber.outer_radius_m,
                    "chamber_length_m": geometry.chamber.length_m,
                    "exit_outer_radius_m": geometry.chamber.exit_outer_radius_m,
                    "stage_count": len(geometry.stages),
                    "stage_pitch_m": geometry.stages[0].pitch_m,
                },
                "descriptors": compute_descriptors(geometry).to_dict(),
                "artifact_file_sha256": files,
            }
        )
    manifest = {
        "schema_version": "cft_revival.geometry.artifact_manifest/1.1.0",
        "generator": ARTIFACT_GENERATOR_ID,
        "generator_version": ARTIFACT_GENERATOR_VERSION,
        "configurations": configurations,
        "claim_limit": ARTIFACT_CLAIM_LIMIT,
    }
    content = canonical_json(manifest)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(content, encoding="utf-8", newline="\n")
    digest = sha256(content.encode("utf-8")).hexdigest()
    (output / "manifest.json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii", newline="\n"
    )
    return manifest


if __name__ == "__main__":
    regenerate()
