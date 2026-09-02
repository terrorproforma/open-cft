"""Contract and offline smoke tests for the L1a sweep dashboard."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

MODERN = Path(__file__).resolve().parents[3]
GENERATOR_PATH = (
    MODERN
    / "experiments"
    / "l1a_geometry_sweep"
    / "visualization"
    / "generate_dashboard.py"
)
CHECKED_HTML = GENERATOR_PATH.with_name("l1a-geometry-sweep.html")


def _load_generator():
    spec = importlib.util.spec_from_file_location("l1a_sweep_dashboard", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_reviewed_hashes_summary_front_and_representatives_are_exact(payload) -> None:
    assert payload["manifest"] == {
        "file_sha256": GENERATOR.EXPECTED_MANIFEST_FILE_SHA256,
        "payload_sha256": GENERATOR.EXPECTED_MANIFEST_PAYLOAD_SHA256,
        "dataset_file_sha256": GENERATOR.EXPECTED_DATASET_FILE_SHA256,
        "dataset_payload_sha256": GENERATOR.EXPECTED_DATASET_PAYLOAD_SHA256,
    }
    assert (
        payload["summary"]["evaluated_count"],
        payload["summary"]["feasible_count"],
        payload["summary"]["nondominated_count"],
    ) == (96, 96, 25)
    assert len(payload["cases"]) == 96
    assert sum(case["nondominated"] for case in payload["cases"]) == 25
    assert [
        (item["label"], item["case_id"]) for item in payload["representatives"]
    ] == list(GENERATOR.EXPECTED_REPRESENTATIVES)
    GENERATOR.validate_payload(payload)


def test_all_requested_dimensions_and_directions_are_embedded(payload) -> None:
    assert [item["name"] for item in payload["metrics"]] == [
        "centreline_mid_abs_bz_t",
        "minimum_mirror_ratio",
        "axis_cusp_count",
        "axis_null_count",
        "stage_gradient_rms_t_per_m",
        "field_energy_j",
        "source_representation_error",
        "topology_confidence",
        "boundary_to_peak_ratio",
    ]
    assert [
        (item["name"], item["direction"], item["units"])
        for item in payload["objectives"]
    ] == list(GENERATOR.OBJECTIVES)
    assert {item["sense"] for item in payload["constraints"]} == {"<=", ">="}
    for case in payload["cases"]:
        assert case["feasible"] is True
        assert case["case_sha256"] == GENERATOR._case_hash(case)
        assert all(
            item["name"] in case["constraints"] for item in payload["constraints"]
        )


def test_representative_fields_profiles_geometry_and_sources_are_actual(payload) -> None:
    for representative in payload["representatives"]:
        field = representative["field"]
        nr = len(field["map"]["r_m"])
        nz = len(field["map"]["z_m"])
        assert nr >= 2 and nz >= 2
        for component in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t"):
            assert len(field["map"][component]) == nr
            assert all(len(row) == nz for row in field["map"][component])
        assert len(field["profiles"]["centreline"]["z_m"]) == 145
        assert len(field["profiles"]["wall"]["z_m"]) == 145
        assert field["sources"]
        assert all(source["polarity"] in (-1, 1) for source in field["sources"])
        assert representative["geometry"]["regions"]
        assert representative["geometry"]["stages"]
        assert field["diagnostics"]["converged"] is True
        assert field["limitations"]


def test_generation_is_byte_deterministic_and_checked_html_is_current(
    payload, tmp_path: Path
) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "dashboard.html"
    digest = GENERATOR.generate(output)
    assert output.read_text(encoding="utf-8") == first
    assert CHECKED_HTML.read_text(encoding="utf-8") == first
    assert digest == __import__("hashlib").sha256(first.encode("utf-8")).hexdigest()


def test_html_is_self_contained_offline_path_free_and_secret_free(payload) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    assert '<script id="sweep-data" type="application/json">' in html
    for forbidden in (
        "fetch(",
        "xmlhttprequest",
        "websocket",
        "eventsource",
        "<iframe",
        "<svg",
        "cdn",
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


def test_html_has_canvas_linking_filters_accessibility_and_redraw(payload) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        'id="scatter"',
        'id="parallel"',
        'id="field"',
        'id="profile"',
        'id="filters"',
        'id="reset"',
        'id="theme"',
        'id="representative"',
        'id="component"',
        'tabindex="0"',
        'role="img"',
        'aria-live="polite"',
        "window.devicePixelRatio",
        "new ResizeObserver(schedule)",
        "requestAnimationFrame(drawAll)",
        'e.key==="Escape"',
        '"ArrowLeft"',
        '"ArrowRight"',
        '"ArrowUp"',
        '"ArrowDown"',
        '"Home"',
        '"End"',
        "function contourSegments(v,rs,zs,level)",
        "createImageData",
    ):
        assert fragment in html
    assert "Full inputs" in html
    assert "Identities" in html
    assert "Residual and gates" in html
    assert "CPU/CUDA parity" in html
    assert "Field limitations" in html
    assert "no per-point DOM" in html
    assert "no material-aware permanent-magnet model" in payload["warning"]
    assert "plasma solution" in payload["warning"]
    assert "thrust" in payload["warning"]
    assert "efficiency" in payload["warning"]
    assert "hardware validity" in payload["warning"]


def test_embedded_json_round_trips_strictly(payload) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(
        r'<script id="sweep-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(match.group(1), parse_constant=reject_constant) == payload


def test_javascript_syntax_when_node_is_available(payload, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    scripts = re.findall(
        r"<script(?: [^>]*)?>(.*?)</script>",
        GENERATOR.render_html(payload),
        re.DOTALL,
    )
    assert len(scripts) == 2
    path = tmp_path / "dashboard.js"
    path.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run(
        [node, "--check", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_file_sidecar_payload_and_field_tampering_are_rejected(
    payload, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(GENERATOR.DEFAULT_MANIFEST.read_bytes() + b" ")
    source_sidecar = GENERATOR.DEFAULT_MANIFEST.with_name("manifest.json.sha256")
    manifest.with_name("manifest.json.sha256").write_text(
        source_sidecar.read_text(encoding="ascii"), encoding="ascii"
    )
    with pytest.raises(ValueError, match="file SHA-256|sidecar"):
        GENERATOR._verify_file(manifest, "manifest")

    dataset = GENERATOR._load_object(GENERATOR.RESULTS / "dataset.json", "dataset")
    dataset["limitations"][0] += " tampered"
    with pytest.raises(ValueError, match="canonical payload SHA-256"):
        GENERATOR._verify_integrity(dataset, "dataset")

    representative = payload["representatives"][0]
    binding = next(
        item
        for item in GENERATOR._load_object(
            GENERATOR.DEFAULT_MANIFEST, "manifest"
        )["representative_artifacts"]
        if item["case_id"] == representative["case_id"]
    )
    field = GENERATOR._load_object(
        GENERATOR.RESULTS / binding["downsampled_field"]["path"], "field"
    )
    field = deepcopy(field)
    field["field_map"]["b_magnitude_t"][1][1] += 0.1
    with pytest.raises(ValueError, match=r"\|B\| component identity"):
        GENERATOR._validate_field(field, "field")


def test_embedded_identity_or_representative_substitution_is_rejected(payload) -> None:
    changed = deepcopy(payload)
    changed["manifest"]["dataset_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="embedded manifest/dataset identity"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["representatives"][0]["case_id"] = changed["representatives"][1]["case_id"]
    with pytest.raises(ValueError, match="representative identity"):
        GENERATOR.validate_payload(changed)
