import json
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree

from cft_revival.geometry import (
    canonical_json,
    divergent_exit_stack,
    historical_envelope_baseline,
    svg_meridional_cross_section,
    viewer_data,
    write_reference_artifacts,
)


def test_svg_contains_materials_dimensions_stages_polarity_and_exit() -> None:
    geometry = divergent_exit_stack()
    svg = svg_meridional_cross_section(geometry)
    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert 'id="material-regions"' in svg
    assert 'id="stage-polarity"' in svg
    assert 'data-material="synthetic-smco-like-example-v1"' in svg
    assert 'data-polarity="1"' in svg
    assert 'data-polarity="-1"' in svg
    assert "divergent exit" in svg
    assert "pitch=" in svg
    assert geometry.canonical_sha256 in svg


def test_artifact_export_is_byte_deterministic_and_sidecars_match(tmp_path: Path) -> None:
    geometry = historical_envelope_baseline()
    first = write_reference_artifacts(geometry, tmp_path)
    first_bytes = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if not path.name.endswith(".sha256")
    }
    second = write_reference_artifacts(geometry, tmp_path)
    second_bytes = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if not path.name.endswith(".sha256")
    }
    assert first == second
    assert first_bytes == second_bytes
    for filename, digest in first.items():
        assert sha256((tmp_path / filename).read_bytes()).hexdigest() == digest
        assert (tmp_path / f"{filename}.sha256").read_text(
            encoding="ascii"
        ) == f"{digest}  {filename}\n"
        if filename.endswith(".json"):
            assert not (tmp_path / filename).read_bytes().endswith(b"\n")


def test_viewer_data_and_json_schemas_are_closed() -> None:
    geometry = divergent_exit_stack()
    data = viewer_data(geometry)
    assert data["geometry_payload_sha256"] == geometry.canonical_sha256
    assert canonical_json(data) == canonical_json(viewer_data(geometry))
    assert any(
        region["shape"] == "linear_taper_annulus" for region in data["regions"]
    )

    modern = Path(__file__).resolve().parents[2]
    geometry_schema = json.loads(
        (modern / "spec" / "geometry" / "axisymmetric-cft-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    viewer_schema = json.loads(
        (modern / "spec" / "geometry" / "viewer-data-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_schema = json.loads(
        (modern / "spec" / "geometry" / "artifact-manifest-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    solver_schema = json.loads(
        (modern / "spec" / "geometry" / "solver-neutral-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert geometry_schema["additionalProperties"] is False
    assert geometry_schema["properties"]["integrity"]["additionalProperties"] is False
    assert geometry_schema["$defs"]["region"]["additionalProperties"] is False
    assert viewer_schema["additionalProperties"] is False
    assert (
        viewer_schema["$defs"]["external"]["additionalProperties"] is False
    )
    assert manifest_schema["additionalProperties"] is False
    assert solver_schema["additionalProperties"] is False
