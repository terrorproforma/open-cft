"""Deterministic JSON, SVG, and viewer-data exports without CAD dependencies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from html import escape
from pathlib import Path
from xml.etree import ElementTree

from .descriptors import compute_descriptors
from .model import (
    AxisymmetricCFTGeometry,
    GeometryValidationError,
    MaterialKind,
    canonical_json,
    deserialize_geometry,
)
from .topology import interface_topology

VIEWER_SCHEMA_VERSION = "cft_revival.geometry.viewer_data/1.1.0"
ARTIFACT_GENERATOR_ID = (
    "modern/examples/geometry/generate_reference_artifacts.py"
)
ARTIFACT_GENERATOR_VERSION = "1.1.0"
ARTIFACT_CLAIM_LIMIT = (
    "Hypothetical geometry artifacts only; not optimized, build-qualified, "
    "or predictive of propulsion performance."
)

_COLORS = {
    MaterialKind.VACUUM_PLASMA: "#e7f4ff",
    MaterialKind.DIELECTRIC: "#e9d8a6",
    MaterialKind.PERMANENT_MAGNET: "#d1495b",
    MaterialKind.SOFT_MAGNETIC: "#495057",
    MaterialKind.NONMAGNETIC_SHIELD: "#adb5bd",
    MaterialKind.ELECTRODE: "#d97706",
}


def _xml(value: object) -> str:
    return escape(str(value), quote=True)


def viewer_data(geometry: AxisymmetricCFTGeometry) -> dict[str, object]:
    return {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "coordinate_system": geometry.coordinate_system,
        "length_unit": "m",
        "config_id": geometry.config_id,
        "geometry_payload_sha256": geometry.canonical_sha256,
        "permanent_magnet_plan": geometry.permanent_magnet_plan.to_dict(),
        "regions": [
            {
                "region_id": region.region_id,
                "role": region.role,
                "material_id": region.material_id,
                "shape": region.shape.value,
                "polygon_rz_m": [
                    [region.r_inner_start_m, region.z_min_m],
                    [region.r_outer_start_m, region.z_min_m],
                    [region.r_outer_end_m, region.z_max_m],
                    [region.r_inner_end_m, region.z_max_m],
                ],
                "polarity": region.polarity,
            }
            for region in geometry.regions
        ],
        "stage_annotations": [
            {
                "stage_id": stage.stage_id,
                "center_z_m": stage.center_z_m,
                "pitch_m": stage.pitch_m,
                "z_min_m": stage.z_min_m,
                "z_max_m": stage.z_max_m,
                "polarity": stage.magnetization.polarity,
            }
            for stage in geometry.stages
        ],
        "interfaces": [
            descriptor.to_dict() for descriptor in interface_topology(geometry)
        ],
        "external_components": [
            component.to_dict() for component in geometry.external_components
        ],
        "claim_limit": "geometry display data only; no field or performance result",
    }


def _point(
    radius_m: float,
    z_m: float,
    *,
    radial_sign: int,
    scale: float,
    margin: float,
    radial_extent_m: float,
) -> str:
    x = margin + z_m * scale
    y = margin + (radial_extent_m + radius_m * radial_sign) * scale
    return f"{x:.6f},{y:.6f}"


def svg_meridional_cross_section(geometry: AxisymmetricCFTGeometry) -> str:
    """Render both meridional halves with material, stage, polarity, and dimensions."""

    descriptors = compute_descriptors(geometry)
    radial_extent = descriptors.envelope_radius_m
    z_min = descriptors.envelope_z_min_m
    z_max = descriptors.envelope_z_max_m
    scale = 20_000.0
    margin = 45.0
    width = (z_max - z_min) * scale + 2.0 * margin
    height = 2.0 * radial_extent * scale + 2.0 * margin

    def xy(radius_m: float, z_m: float, radial_sign: int) -> str:
        return _point(
            radius_m,
            z_m - z_min,
            radial_sign=radial_sign,
            scale=scale,
            margin=margin,
            radial_extent_m=radial_extent,
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}" '
            f'height="{height:.3f}" viewBox="0 0 {width:.3f} {height:.3f}" '
            'role="img">'
        ),
        f"<title>{_xml(geometry.title)} meridional cross-section</title>",
        (
            f'<metadata data-schema="{_xml(VIEWER_SCHEMA_VERSION)}" '
            f'data-geometry-sha256="{_xml(geometry.canonical_sha256)}">'
            "Hypothetical CFT geometry; no performance prediction. TWT inspiration "
            "is limited to PPM stack hardware geometry.</metadata>"
        ),
        "<style>"
        ".region{stroke:#111;stroke-width:0.7}.axis{stroke:#111;stroke-dasharray:4 3}"
        ".dim{stroke:#1d3557;stroke-width:0.7;fill:none}.label{font:9px sans-serif;"
        "fill:#111}.small{font:7px sans-serif;fill:#111}.polarity{font:bold 10px "
        "sans-serif;fill:white;text-anchor:middle}</style>",
        (
            f'<line class="axis" x1="{margin:.6f}" y1="{height / 2:.6f}" '
            f'x2="{width - margin:.6f}" y2="{height / 2:.6f}"/>'
        ),
        '<g id="material-regions">',
    ]
    for region in geometry.regions:
        material = geometry.material_by_id(region.material_id)
        fill = _COLORS.get(material.category, "#f1f3f5")
        for sign in (-1, 1):
            points = " ".join(
                (
                    xy(region.r_inner_start_m, region.z_min_m, sign),
                    xy(region.r_outer_start_m, region.z_min_m, sign),
                    xy(region.r_outer_end_m, region.z_max_m, sign),
                    xy(region.r_inner_end_m, region.z_max_m, sign),
                )
            )
            lines.append(
                f'<polygon id="{_xml(region.region_id)}-'
                f'{"top" if sign < 0 else "bottom"}" '
                f'class="region" points="{_xml(points)}" fill="{_xml(fill)}" '
                f'data-material="{_xml(region.material_id)}" '
                f'data-role="{_xml(region.role)}"/>'
            )
    lines.append("</g>")
    lines.append('<g id="stage-polarity">')
    for stage in geometry.stages:
        magnet = geometry.region_by_id(stage.magnet_region_id)
        x = margin + (stage.center_z_m - z_min) * scale
        y = height / 2.0 - (
            (magnet.r_inner_start_m + magnet.r_outer_start_m) * 0.5 * scale
        )
        symbol = "+z" if stage.magnetization.polarity == 1 else "−z"
        lines.append(
            f'<text class="polarity" x="{x:.6f}" y="{y + 3.0:.6f}" '
            f'data-stage="{_xml(stage.stage_id)}" '
            f'data-polarity="{_xml(stage.magnetization.polarity)}">'
            f"{_xml(symbol)}</text>"
        )
    lines.append("</g>")
    dimension_y = height - 14.0
    start_x = margin + (0.0 - z_min) * scale
    end_x = margin + (geometry.chamber.length_m - z_min) * scale
    lines.extend(
        (
            '<g id="dimensions">',
            f'<line class="dim" x1="{start_x:.6f}" y1="{dimension_y:.6f}" '
            f'x2="{end_x:.6f}" y2="{dimension_y:.6f}"/>',
            f'<text class="label" x="{(start_x + end_x) / 2:.6f}" '
            f'y="{dimension_y - 3.0:.6f}" text-anchor="middle">'
            f'L={geometry.chamber.length_m * 1000.0:.3f} mm</text>',
            f'<text class="label" x="{margin:.6f}" y="12">r-axis; all dimensions SI '
            f'(labels shown in mm), pitch={geometry.stages[0].pitch_m * 1000.0:.3f} mm, '
            f"cusps={len(geometry.stages) - 1}</text>",
            f'<text class="small" x="{margin:.6f}" y="23">channel '
            f'R={geometry.chamber.outer_radius_m * 1000.0:.3f} mm; envelope '
            f'R={radial_extent * 1000.0:.3f} mm</text>',
        )
    )
    if geometry.chamber.exit_length_m > 0.0:
        exit_x = margin + (geometry.chamber.exit_start_m - z_min) * scale
        lines.append(
            f'<line class="dim" x1="{exit_x:.6f}" y1="{margin:.6f}" '
            f'x2="{exit_x:.6f}" y2="{height - margin:.6f}" stroke-dasharray="2 2"/>'
        )
        lines.append(
            f'<text class="small" x="{exit_x + 2.0:.6f}" y="{margin + 9.0:.6f}">'
            f"divergent exit to R={geometry.chamber.exit_outer_radius_m * 1000.0:.3f} mm"
            "</text>"
        )
    lines.extend(
        (
            "</g>",
            '<text class="small" x="4" y="38" transform="rotate(-90 4 38)">−r / +r</text>',
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def write_reference_artifacts(
    geometry: AxisymmetricCFTGeometry, output_directory: Path
) -> dict[str, str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = geometry.config_id.removesuffix("-v1")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", stem) or ".." in stem:
        raise GeometryValidationError("config_id cannot form a safe artifact filename")
    payloads = {
        f"{stem}.json": canonical_json(geometry.to_dict()),
        f"{stem}.viewer.json": canonical_json(viewer_data(geometry)),
        f"{stem}.svg": svg_meridional_cross_section(geometry),
    }
    hashes: dict[str, str] = {}
    for filename, content in payloads.items():
        path = output_directory / filename
        path.write_text(content, encoding="utf-8", newline="\n")
        digest = sha256(content.encode("utf-8")).hexdigest()
        hashes[filename] = digest
        (output_directory / f"{filename}.sha256").write_text(
            f"{digest}  {filename}\n", encoding="ascii", newline="\n"
        )
    return hashes


@dataclass(frozen=True, slots=True)
class LoadedArtifactBundle:
    manifest: dict[str, object]
    geometries: tuple[AxisymmetricCFTGeometry, ...]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GeometryValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise GeometryValidationError(f"non-finite JSON constant {value!r} is forbidden")


def _exact_keys(value: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GeometryValidationError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise GeometryValidationError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _safe_artifact_name(value: object) -> str:
    if not isinstance(value, str):
        raise GeometryValidationError("artifact filename must be text")
    if (
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value)
        or ".." in value
        or "/" in value
        or "\\" in value
    ):
        raise GeometryValidationError("artifact filename is not canonical and safe")
    return value


def _verified_file(
    directory: Path,
    filename: str,
    expected_digest: object,
) -> bytes:
    safe_name = _safe_artifact_name(filename)
    if (
        not isinstance(expected_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
    ):
        raise GeometryValidationError("artifact digest must be lowercase SHA-256")
    path = directory / safe_name
    sidecar_path = directory / f"{safe_name}.sha256"
    resolved_directory = directory.resolve()
    if (
        path.is_symlink()
        or sidecar_path.is_symlink()
        or path.resolve().parent != resolved_directory
        or sidecar_path.resolve().parent != resolved_directory
    ):
        raise GeometryValidationError("artifact paths must be direct non-symlink files")
    try:
        content = path.read_bytes()
        sidecar = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise GeometryValidationError(
            f"artifact or sidecar is missing for {safe_name!r}"
        ) from error
    actual = sha256(content).hexdigest()
    if not compare_digest(actual, expected_digest):
        raise GeometryValidationError(f"file SHA-256 mismatch for {safe_name!r}")
    expected_sidecar = f"{expected_digest}  {safe_name}\n"
    if sidecar != expected_sidecar:
        raise GeometryValidationError(f"invalid SHA-256 sidecar for {safe_name!r}")
    return content


def _decode_unique_json(content: bytes, name: str) -> dict[str, object]:
    try:
        text = content.decode("utf-8")
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, TypeError) as error:
        if isinstance(error, GeometryValidationError):
            raise
        raise GeometryValidationError(f"{name} is not valid strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise GeometryValidationError(f"{name} must contain a JSON object")
    if canonical_json(decoded) != text:
        raise GeometryValidationError(
            f"{name} violates the no-trailing-newline canonical JSON policy"
        )
    return decoded


def load_artifact_bundle(directory: Path) -> LoadedArtifactBundle:
    """Strictly verify a complete generated geometry artifact directory."""

    if not isinstance(directory, Path):
        raise GeometryValidationError("artifact directory must be a pathlib.Path")
    manifest_path = directory / "manifest.json"
    sidecar_path = directory / "manifest.json.sha256"
    if manifest_path.is_symlink() or sidecar_path.is_symlink():
        raise GeometryValidationError("manifest paths must not be symlinks")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_sidecar = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise GeometryValidationError("manifest or sidecar is missing") from error
    manifest_digest = sha256(manifest_bytes).hexdigest()
    if manifest_sidecar != f"{manifest_digest}  manifest.json\n":
        raise GeometryValidationError("manifest SHA-256 sidecar verification failed")
    manifest = _decode_unique_json(manifest_bytes, "manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "generator",
            "generator_version",
            "configurations",
            "claim_limit",
        },
        "manifest",
    )
    if (
        manifest["schema_version"]
        != "cft_revival.geometry.artifact_manifest/1.1.0"
    ):
        raise GeometryValidationError("unsupported geometry artifact manifest schema")
    if manifest["generator"] != ARTIFACT_GENERATOR_ID:
        raise GeometryValidationError("artifact generator identity is not allowlisted")
    if manifest["generator_version"] != ARTIFACT_GENERATOR_VERSION:
        raise GeometryValidationError("artifact generator version is not allowlisted")
    if manifest["claim_limit"] != ARTIFACT_CLAIM_LIMIT:
        raise GeometryValidationError(
            "artifact claim limit violates the accepted evidence boundary"
        )
    configurations = manifest["configurations"]
    if not isinstance(configurations, list) or not configurations:
        raise GeometryValidationError("manifest configurations must be a non-empty array")

    seen_configs: set[str] = set()
    seen_files: set[str] = set()
    geometries: list[AxisymmetricCFTGeometry] = []
    expected_directory_files = {"manifest.json", "manifest.json.sha256"}
    for index, raw_entry in enumerate(configurations):
        entry = _exact_keys(
            raw_entry,
            {
                "config_id",
                "geometry_payload_sha256",
                "dimensions",
                "descriptors",
                "artifact_file_sha256",
            },
            f"manifest configuration[{index}]",
        )
        config_id = entry["config_id"]
        if not isinstance(config_id, str) or config_id in seen_configs:
            raise GeometryValidationError("manifest config IDs must be unique strings")
        seen_configs.add(config_id)
        hashes = _exact_keys(
            entry["artifact_file_sha256"],
            {
                f"{config_id.removesuffix('-v1')}.json",
                f"{config_id.removesuffix('-v1')}.viewer.json",
                f"{config_id.removesuffix('-v1')}.svg",
            },
            f"artifact hashes for {config_id}",
        )
        for filename in hashes:
            if filename in seen_files:
                raise GeometryValidationError("manifest artifact filenames must be unique")
            seen_files.add(filename)
            expected_directory_files.update((filename, f"{filename}.sha256"))
        geometry_filename = f"{config_id.removesuffix('-v1')}.json"
        geometry_bytes = _verified_file(
            directory, geometry_filename, hashes[geometry_filename]
        )
        try:
            geometry = deserialize_geometry(geometry_bytes.decode("utf-8"))
        except UnicodeError as error:
            raise GeometryValidationError("geometry artifact must be UTF-8") from error
        if geometry.config_id != config_id:
            raise GeometryValidationError("manifest/geometry config substitution detected")
        if entry["geometry_payload_sha256"] != geometry.canonical_sha256:
            raise GeometryValidationError("manifest geometry payload hash mismatch")

        viewer_filename = f"{config_id.removesuffix('-v1')}.viewer.json"
        viewer_bytes = _verified_file(
            directory, viewer_filename, hashes[viewer_filename]
        )
        viewer = _decode_unique_json(viewer_bytes, viewer_filename)
        if canonical_json(viewer_data(geometry)) != canonical_json(viewer):
            raise GeometryValidationError(
                "viewer artifact is not the closed projection of its geometry"
            )

        svg_filename = f"{config_id.removesuffix('-v1')}.svg"
        svg_bytes = _verified_file(directory, svg_filename, hashes[svg_filename])
        try:
            ElementTree.fromstring(svg_bytes)
        except ElementTree.ParseError as error:
            raise GeometryValidationError("SVG artifact is not well-formed XML") from error
        if svg_bytes != svg_meridional_cross_section(geometry).encode("utf-8"):
            raise GeometryValidationError("SVG artifact substitution detected")

        dimensions = _exact_keys(
            entry["dimensions"],
            {
                "chamber_inner_radius_m",
                "chamber_outer_radius_m",
                "chamber_length_m",
                "exit_outer_radius_m",
                "stage_count",
                "stage_pitch_m",
            },
            f"dimensions for {config_id}",
        )
        expected_dimensions = {
            "chamber_inner_radius_m": geometry.chamber.inner_radius_m,
            "chamber_outer_radius_m": geometry.chamber.outer_radius_m,
            "chamber_length_m": geometry.chamber.length_m,
            "exit_outer_radius_m": geometry.chamber.exit_outer_radius_m,
            "stage_count": len(geometry.stages),
            "stage_pitch_m": geometry.stages[0].pitch_m,
        }
        if dimensions != expected_dimensions:
            raise GeometryValidationError("manifest dimensions do not match geometry")
        if entry["descriptors"] != compute_descriptors(geometry).to_dict():
            raise GeometryValidationError("manifest descriptors do not match geometry")
        geometries.append(geometry)

    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != expected_directory_files:
        raise GeometryValidationError(
            "artifact directory has missing or unmanifested files"
        )
    return LoadedArtifactBundle(manifest=manifest, geometries=tuple(geometries))
