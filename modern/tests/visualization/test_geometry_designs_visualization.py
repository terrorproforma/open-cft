"""Tests for the standalone accepted-geometry v1.1 viewer."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from cft_revival.geometry import (
    GeometryValidationError,
    compute_descriptors,
    load_artifact_bundle,
    viewer_data,
)

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_geometry_designs.py"
CHECKED_HTML = MODERN / "visualization" / "geometry-designs.html"
ARTIFACTS = MODERN / "examples" / "geometry" / "artifacts"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "geometry_designs_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_accepted_bundle_and_reviewed_identities_are_exact(payload) -> None:
    bundle = load_artifact_bundle(ARTIFACTS)
    assert payload["manifest"]["file_sha256"] == (
        GENERATOR.EXPECTED_MANIFEST_FILE_SHA256
    )
    assert payload["manifest"]["schema_version"] == (
        "cft_revival.geometry.artifact_manifest/1.1.0"
    )
    assert [design["id"] for design in payload["designs"]] == [
        item[0] for item in GENERATOR.EXPECTED_CONFIGURATIONS
    ]
    assert [geometry.config_id for geometry in bundle.geometries] == [
        design["id"] for design in payload["designs"]
    ]
    for design, expected in zip(
        payload["designs"], GENERATOR.EXPECTED_CONFIGURATIONS, strict=True
    ):
        assert design["identity"]["geometry_payload_sha256"] == expected[2]
        assert design["identity"]["geometry_file_sha256"] == expected[3]
        assert design["identity"]["viewer_file_sha256"] == expected[4]
    GENERATOR.validate_payload(payload)


def test_display_geometry_is_exact_accepted_projection(payload) -> None:
    bundle = load_artifact_bundle(ARTIFACTS)
    for design, geometry in zip(payload["designs"], bundle.geometries, strict=True):
        accepted = viewer_data(geometry)
        assert design["coordinate_system"] == accepted["coordinate_system"]
        assert design["interfaces"] == accepted["interfaces"]
        assert design["external_components"] == accepted["external_components"]
        assert len(design["regions"]) == len(accepted["regions"])
        for displayed, projected, model_region in zip(
            design["regions"],
            accepted["regions"],
            geometry.regions,
            strict=True,
        ):
            for key, value in projected.items():
                assert displayed[key] == value
            assert displayed["volume_m3"] == model_region.volume_m3
            assert displayed["owner_id"] == model_region.owner_id
            assert displayed["z_min_m"] == model_region.z_min_m
            assert displayed["z_max_m"] == model_region.z_max_m
            assert displayed["r_inner_start_m"] == model_region.r_inner_start_m
            assert displayed["r_inner_end_m"] == model_region.r_inner_end_m
            assert displayed["r_outer_start_m"] == model_region.r_outer_start_m
            assert displayed["r_outer_end_m"] == model_region.r_outer_end_m
            assert all(
                math.isfinite(number)
                for point in displayed["polygon_rz_m"]
                for number in point
            )


def test_descriptors_volumes_clearances_and_polarity_are_artifact_derived(
    payload,
) -> None:
    bundle = load_artifact_bundle(ARTIFACTS)
    expected_stage_counts = [3, 5, 4]
    expected_polarities = [[1, -1, 1], [1, -1, 1, -1, 1], [1, -1, 1, -1]]
    for design, geometry, stage_count, polarities in zip(
        payload["designs"],
        bundle.geometries,
        expected_stage_counts,
        expected_polarities,
        strict=True,
    ):
        descriptors = compute_descriptors(geometry).to_dict()
        for key, value in descriptors.items():
            assert design["descriptors"][key] == value
        magnets = [
            geometry.region_by_id(stage.magnet_region_id)
            for stage in geometry.stages
        ]
        assert design["descriptors"]["magnet_volume_m3"] == math.fsum(
            region.volume_m3 for region in magnets
        )
        assert len(design["stages"]) == stage_count
        assert [stage["polarity"] for stage in design["stages"]] == polarities
        assert all(design["descriptors"][key] > 0 for key in (
            "active_volume_m3",
            "channel_volume_m3",
            "magnet_volume_m3",
            "magnet_mass_estimate_kg",
            "minimum_radial_gap_m",
            "minimum_axial_gap_m",
        ))
        assert design["descriptors"]["manufacturability_warnings"]


def test_divergent_exit_and_complete_selection_records_are_preserved(payload) -> None:
    divergent = payload["designs"][2]
    tapered = [
        region for region in divergent["regions"]
        if region["shape"] == "linear_taper_annulus"
    ]
    assert {region["region_id"] for region in tapered} == {
        "channel-divergent-exit",
        "dielectric-divergent-exit",
    }
    assert all(
        region["r_outer_start_m"] != region["r_outer_end_m"]
        for region in tapered
    )
    region_ids = {region["region_id"] for region in divergent["regions"]}
    assert all(interface["region_id"] in region_ids for interface in divergent["interfaces"])
    required_interface_keys = {
        "interface_id",
        "region_id",
        "adjacent_region_id",
        "surface",
        "orientation",
        "start_rz_m",
        "end_rz_m",
        "unit_normal_rz",
        "free_surface_current_phi_a_per_m",
    }
    assert all(set(interface) == required_interface_keys for interface in divergent["interfaces"])


def test_generation_is_byte_deterministic_and_checked_html_is_current(
    payload, tmp_path: Path
) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "geometry-designs.html"
    GENERATOR.generate(output)
    assert output.read_text(encoding="utf-8") == first
    assert CHECKED_HTML.read_text(encoding="utf-8") == first


def test_html_is_zero_network_path_free_and_secret_free(payload) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    assert '<script id="geometry-data" type="application/json">' in html
    for forbidden in (
        "fetch(",
        "xmlhttprequest",
        "websocket",
        "eventsource",
        "cdn",
        "<iframe",
    ):
        assert forbidden not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", html, re.I)
    assert not re.search(r"\bhttps?://", html, re.I)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", html)
    assert not re.search(
        r"(?:api[_-]?key|access[_-]?token|client[_-]?secret)"
        r"\s*[:=]\s*['\"][^'\"]+",
        html,
        re.I,
    )


def test_html_has_compare_dimensions_accessibility_and_redraw(payload) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        'id="primary"',
        'id="secondary"',
        '<option value="side">Side by side</option>',
        '<option value="overlay">Overlay</option>',
        'id="dimensions"',
        'id="reset"',
        'aria-keyshortcuts="Escape"',
        'id="theme"',
        'tabindex="0"',
        'role="img"',
        'aria-live="polite"',
        "window.devicePixelRatio",
        "new ResizeObserver(schedule)",
        '["ArrowRight","ArrowDown"].includes(event.key)',
        '["ArrowLeft","ArrowUp"].includes(event.key)',
        'event.key==="Enter"',
        'event.key==="Home"',
        "pointIn(points,x,y)",
        "requestAnimationFrame(drawAll)",
    ):
        assert fragment in html
    assert "annular magnets alternate axial polarity" in payload["physics_boundary"]
    assert "CFT operation is distinct plasma-discharge" in payload["physics_boundary"]
    assert "TWT RF slow-wave" in payload["physics_boundary"]
    assert payload["warning"] == (
        "Hypothetical geometry artifacts only; not optimized, build-qualified, "
        "or predictive of propulsion performance."
    )


def test_reset_restores_complete_initial_view_state_without_changing_theme(
    payload,
) -> None:
    html = GENERATOR.render_html(payload)
    assert (
        'const INITIAL_VIEW_STATE=Object.freeze({primary:0,secondary:1,mode:"side",'
        "showDimensions:true,selectionDesign:0,selectionRegion:0,hover:null,"
        "cursor:null,zoom:1,panX:0,panY:0});"
    ) in html
    match = re.search(
        r"function resetView\(\)\{(.*?)\}\nfunction applyTheme",
        html,
        re.DOTALL,
    )
    assert match is not None
    reset_body = match.group(1)
    for assignment in (
        "primary=INITIAL_VIEW_STATE.primary",
        "secondary=INITIAL_VIEW_STATE.secondary",
        "mode=INITIAL_VIEW_STATE.mode",
        "showDimensions=INITIAL_VIEW_STATE.showDimensions",
        "selection={design:INITIAL_VIEW_STATE.selectionDesign,"
        "region:INITIAL_VIEW_STATE.selectionRegion}",
        "hover=INITIAL_VIEW_STATE.hover",
        "cursor=INITIAL_VIEW_STATE.cursor",
        "viewport={zoom:INITIAL_VIEW_STATE.zoom,panX:INITIAL_VIEW_STATE.panX,"
        "panY:INITIAL_VIEW_STATE.panY}",
        "layouts={}",
        "syncControls()",
        '$("details").scrollTop=0',
        "schedule()",
    ):
        assert assignment in reset_body
    assert "theme" not in reset_body.lower()
    assert '$("reset").onclick=resetView' in html
    assert (
        'if(e.key==="Escape"){e.preventDefault();resetView();return}'
    ) in html
    assert (
        'function syncControls(){$("primary").value=String(primary);'
        '$("secondary").value=String(secondary);$("mode").value=mode;'
        '$("dimensions").checked=showDimensions}'
    ) in html


def test_theme_starts_from_and_tracks_os_preference_independently_of_reset(
    payload,
) -> None:
    html = GENERATOR.render_html(payload)
    assert '<html lang="en">' in html
    assert '<html lang="en" data-theme=' not in html
    assert "@media(prefers-color-scheme:light){:root:not([data-theme])" in html
    assert (
        'const osTheme=window.matchMedia("(prefers-color-scheme: light)");'
        'let themePreference="system";'
    ) in html
    assert (
        'osTheme.addEventListener("change",()=>{if(themePreference==="system")'
        'applyTheme(osTheme.matches?"light":"dark")})'
    ) in html
    assert 'applyTheme(osTheme.matches?"light":"dark");' in html


def test_html_exposes_materials_interfaces_warnings_and_numeric_sources(payload) -> None:
    html = GENERATOR.render_html(payload)
    for role in GENERATOR.ROLE_COLORS:
        assert role in html
    for fragment in (
        "Complete region record",
        "Manufacturability warnings",
        "External components",
        "Interfaces (",
        "geometry magnet regions[].volume_m3",
        "manifest descriptors.active_volume_m3",
        "manifest descriptors.channel_volume_m3",
        "manifest descriptors.magnet_mass_estimate_kg",
        "manifest descriptors.minimum_*_gap_m",
        "source:",
    ):
        assert fragment in html
    for design in payload["designs"]:
        assert design["number_sources"]["chamber_and_regions"].endswith(
            "chamber, regions, stages, materials, manufacturing"
        )
        assert design["number_sources"]["interfaces"].endswith(": interfaces")


def test_embedded_json_round_trips_without_nonstandard_numbers(payload) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(
        r'<script id="geometry-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(match.group(1), parse_constant=reject_constant) == payload


def test_javascript_is_valid_when_node_is_available(payload, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available for JavaScript syntax checking")
    html = GENERATOR.render_html(payload)
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    script = tmp_path / "viewer.js"
    script.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run(
        [node, "--check", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_accepted_loader_rejects_sidecar_tamper_and_unmanifested_files(
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(ARTIFACTS, tampered)
    sidecar = tampered / "historical-envelope-baseline.viewer.json.sha256"
    sidecar.write_text("0" * 64 + "  historical-envelope-baseline.viewer.json\n")
    with pytest.raises(GeometryValidationError, match="sidecar|SHA-256"):
        GENERATOR.build_payload(tampered)

    extra = tmp_path / "extra"
    shutil.copytree(ARTIFACTS, extra)
    (extra / "unmanifested.txt").write_text("not accepted", encoding="utf-8")
    with pytest.raises(GeometryValidationError, match="unmanifested"):
        GENERATOR.build_payload(extra)


def test_payload_rejects_identity_and_polarity_substitution(payload) -> None:
    changed = deepcopy(payload)
    changed["manifest"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest identity"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["designs"][0]["identity"]["viewer_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="artifact identity"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["designs"][1]["stages"][2]["polarity"] = -1
    with pytest.raises(ValueError, match="not alternating"):
        GENERATOR.validate_payload(changed)
