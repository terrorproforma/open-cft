"""Isolated contract tests for the offline P2 FEM qualification dashboard."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = (
    MODERN
    / "examples"
    / "fem_reference"
    / "visualization"
    / "generate_dashboard.py"
)
CHECKED_HTML = GENERATOR_PATH.with_name("fem-reference-p2-qualification.html")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "fem_reference_p2_qualification_dashboard",
        GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_exact_accepted_statuses_and_pinned_manifest_hashes(payload) -> None:
    assert [
        (design["id"], design["status"], design["qualified"])
        for design in payload["designs"]
    ] == [
        ("historical-envelope-baseline", "SCREENING ONLY", False),
        ("compact-high-gradient-stack", "SCREENING ONLY", False),
        ("divergent-exit-stack", "NUMERICAL P2 QUALIFIED", True),
    ]
    assert [
        design["identity"]["manifest_file_sha256"]
        for design in payload["designs"]
    ] == [item[2] for item in GENERATOR.EVIDENCE]
    GENERATOR.validate_payload(payload)


def test_three_nested_levels_qois_quality_refinement_and_resources(payload) -> None:
    for design in payload["designs"]:
        assert [level["level"] for level in design["levels"]] == [0, 1, 2]
        assert all(
            right["p2_dofs"] > left["p2_dofs"]
            for left, right in zip(design["levels"], design["levels"][1:])
        )
        assert all(
            level["mesh_projection"]["source_triangles"] == level["triangles"]
            and level["mesh_projection"]["sampled_triangles"]
            <= GENERATOR.MESH_TRIANGLE_BUDGET
            for level in design["levels"]
        )
        assert all(
            level["minimum_angle_deg"] > 0
            and level["maximum_aspect_indicator"] >= 1
            and level["adjacent_area_size_growth"] <= 1.3 + 1e-12
            and level["relative_true_residual_l2"] < 1e-8
            and level["peak_working_set_bytes"] > 0
            and level["assembly_seconds"] > 0
            and level["solve_seconds"] > 0
            and level["refinement"]
            for level in design["levels"]
        )
        assert all(
            set(level["qois_bz_t"]) == set(design["qoi_names"])
            and set(level["qoi_h_m"]) == set(design["qoi_names"])
            for level in design["levels"]
        )


def test_volume_change_order_and_phase_matched_domain_evidence(payload) -> None:
    for design in payload["designs"]:
        assert len(design["successive_changes"]) == 2
        assert set(design["observed_orders"]) == set(design["qoi_names"])
        assert all(
            value < 0.01
            for change in design["successive_changes"]
            for value in change.values()
        )
        assert design["domain"]["phase_matched"] is True
        assert design["domain"]["passed"] is True
        assert [run["padding_factor"] for run in design["domain"]["runs"]] == [
            0.5,
            1.0,
            1.5,
        ]
        assert len(design["domain"]["successive_qoi_relative_changes"]) == 2
        assert all(
            value < design["domain"]["maximum_qoi_relative_change"]
            for change in design["domain"]["successive_qoi_relative_changes"]
            for value in change.values()
        )
    assert all(
        value > 0
        for value in payload["designs"][2]["observed_orders"].values()
    )
    assert any(
        value <= 0
        for design in payload["designs"][:2]
        for value in design["observed_orders"].values()
    )


def test_fields_regions_profiles_and_hash_ancestry_are_embedded(payload) -> None:
    for design in payload["designs"]:
        field = design["field"]
        cells = field["grid"]["width"] * field["grid"]["height"]
        assert set(field["fields"]) == {
            "psi_wb_per_rad",
            "b_r_t",
            "b_z_t",
            "b_magnitude_t",
        }
        assert all(len(values) == cells for values in field["fields"].values())
        assert len(field["region_raster"]) == cells
        assert field["regions"]
        assert sum(region["triangle_count"] for region in field["regions"]) == (
            field["source_triangles"]
        )
        assert set(field["profiles"]) == {"axis", "radial_slice"}
        assert all(
            len(profile["z_m"]) == GENERATOR.PROFILE_SAMPLES
            and len(profile["b_z_t"]) == GENERATOR.PROFILE_SAMPLES
            for profile in field["profiles"].values()
        )
        assert design["levels"][0]["parent_mesh_sha256"] == GENERATOR.ZERO_HASH
        assert all(
            child["parent_mesh_sha256"] == parent["mesh_sha256"]
            for parent, child in zip(design["levels"], design["levels"][1:])
        )
        for name, digest in design["identity"].items():
            if isinstance(digest, str):
                assert re.fullmatch(r"[0-9a-f]{64}", digest), name
        for chain_name in ("checkpoint_chain", "domain_checkpoint_chain"):
            assert len(design["identity"][chain_name]) == 3
            assert all(
                re.fullmatch(r"[0-9a-f]{64}", value)
                for entry in design["identity"][chain_name]
                for value in entry.values()
            )


def test_generation_is_byte_deterministic_and_checked_html_is_current(payload) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(payload)
    assert first == second
    assert CHECKED_HTML.read_text(encoding="utf-8") == first
    assert sha256(first.encode("utf-8")).hexdigest() == sha256(
        second.encode("utf-8")
    ).hexdigest()


def test_html_is_standalone_offline_path_free_secret_free_and_claim_bounded(
    payload,
) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    assert '<script id="fem-p2-data" type="application/json">' in html
    for forbidden in (
        "fetch(",
        "xmlhttprequest",
        "websocket",
        "eventsource",
        "<iframe",
        "<script src=",
        "<link ",
        "cdn.",
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
    warning = payload["warning"].lower()
    for claim_boundary in (
        "no hardware",
        "experimental",
        "material-plasma",
        "thrust",
        "efficiency",
        "device-performance",
    ):
        assert claim_boundary in warning


def test_html_has_canvas_svg_keyboard_theme_reset_and_mobile_contract(payload) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        'id="field-canvas"',
        'id="mesh-canvas"',
        'id="profile-canvas"',
        'id="change-chart"',
        'id="order-chart"',
        'id="domain-chart"',
        'id="design"',
        'id="level"',
        'id="field"',
        'id="theme"',
        'id="reset"',
        'role="img"',
        'tabindex="0"',
        'aria-live="polite"',
        "new ResizeObserver(schedule)",
        "requestAnimationFrame",
        'e.key==="Escape"',
        'e.key==="ArrowLeft"',
        'e.key==="ArrowRight"',
        'e.key==="ArrowUp"',
        'e.key==="ArrowDown"',
        "devicePixelRatio",
        "createImageData",
        "function contours(",
        "width:100%;max-width:100%;min-width:0",
        "@media(max-width:520px)",
    ):
        assert fragment in html


def test_280_to_390_pixel_containment_structure(payload) -> None:
    html = GENERATOR.render_html(payload)
    style = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    assert style is not None
    css = style.group(1)
    assert "100vw" not in css
    assert "body{min-width:320px}" not in css
    for fragment in (
        "html,body{margin:0;width:100%;max-width:100%;min-width:0;",
        ".shell{width:100%;max-width:1500px;min-width:0;",
        "padding:clamp(10px,2.5%,34px)",
        ".shell>*,.grid>*,.card>*,.controls>*,header>*,.head>*"
        "{min-width:0;max-width:100%}",
        ".controls{position:sticky;top:8px;z-index:5;width:100%;"
        "max-width:100%;min-width:0;",
        "select{width:100%;max-width:100%;min-width:0;inline-size:100%;"
        "max-inline-size:100%}",
        ".canvas-wrap{position:relative;width:100%;max-width:100%;min-width:0;",
        "canvas{display:block;width:100%;max-width:100%;min-width:0;",
        ".svg-chart{display:block;width:100%;max-width:100%;min-width:0;",
        ".controls>*,.controls label,.controls select,.controls button,.status"
        "{width:100%;max-width:100%;min-width:0}",
        ".hash{grid-template-columns:minmax(0,1fr)}",
    ):
        assert fragment in css


def test_embedded_json_round_trips_and_javascript_compiles(
    payload, tmp_path: Path
) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(
        r'<script id="fem-p2-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    assert json.loads(match.group(1)) == payload
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    javascript = tmp_path / "dashboard.js"
    javascript.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run(
        [node, "--check", str(javascript)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_payload_tampering_and_generator_compile_checks(payload) -> None:
    changed = deepcopy(payload)
    changed["designs"][2]["status"] = "SCREENING ONLY"
    with pytest.raises(ValueError, match="status label"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["designs"][0]["qualified"] = True
    with pytest.raises(ValueError, match="qualification evidence"):
        GENERATOR.validate_payload(changed)
    compile(GENERATOR_PATH.read_text(encoding="utf-8"), str(GENERATOR_PATH), "exec")
