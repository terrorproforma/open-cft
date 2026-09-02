"""Tests for the standalone L1a axisymmetric results visualization."""

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

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_axisymmetric_results.py"
CHECKED_HTML = MODERN / "visualization" / "axisymmetric-results.html"
RESULTS = MODERN / "examples" / "axisymmetric" / "results"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "axisymmetric_results_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def _copy_results(tmp_path: Path) -> Path:
    target = tmp_path / "results"
    shutil.copytree(RESULTS, target)
    return target / "manifest-l1a-v1.json"


def test_manifest_and_artifact_identities_are_exact(payload) -> None:
    assert payload["manifest"]["file_sha256"] == (
        GENERATOR.EXPECTED_MANIFEST_FILE_SHA256
    )
    assert payload["manifest"]["payload_sha256"] == (
        GENERATOR.EXPECTED_MANIFEST_PAYLOAD_SHA256
    )
    assert payload["manifest"]["schema_version"] == (
        "cft-axisymmetric-design-manifest/1.2.0"
    )
    assert payload["migration"] == {
        "file": "serialization-migration-v1.1-to-v1.2.json",
        "file_sha256": GENERATOR.EXPECTED_MIGRATION_FILE_SHA256,
        "payload_sha256": GENERATOR.EXPECTED_MIGRATION_PAYLOAD_SHA256,
        "policy": (
            "v1.1 is read-only historical serialization; new outputs use v1.2 "
            "signed-zero normalization; no experiment artifact is migrated in place"
        ),
        "from_schema": "cft-axisymmetric-field-map/1.1.0",
        "to_schema": "cft-axisymmetric-field-map/1.2.0",
    }
    assert [design["id"] for design in payload["designs"]] == [
        expected[0] for expected in GENERATOR.EXPECTED_DESIGNS
    ]
    assert [design["file_sha256"] for design in payload["designs"]] == [
        expected[2] for expected in GENERATOR.EXPECTED_DESIGNS
    ]
    assert [design["payload_sha256"] for design in payload["designs"]] == [
        expected[3] for expected in GENERATOR.EXPECTED_DESIGNS
    ]
    GENERATOR.validate_payload(payload)


def test_v12_authoritative_roundtrip_and_signed_zero_semantics(payload) -> None:
    assert GENERATOR._canonical_payload_sha256({"value": -0.0}) == (
        GENERATOR._canonical_payload_sha256({"value": 0.0})
    )
    assert not GENERATOR.contains_negative_zero(payload)
    for expected in GENERATOR.EXPECTED_DESIGNS:
        path = RESULTS / expected[1]
        raw = path.read_bytes()
        artifact = GENERATOR.reload_field_artifact_bytes(
            raw, source=path.name, allow_legacy_v1_1=False
        )
        assert artifact["schema_version"] == "cft-axisymmetric-field-map/1.2.0"
        assert GENERATOR.field_artifact_canonical_bytes(artifact) == raw
        assert not GENERATOR.contains_negative_zero(artifact)


def test_embedded_field_arrays_are_finite_radial_major_and_physical(payload) -> None:
    for design in payload["designs"]:
        field = design["field"]
        assert field["layout"] == (
            "radial-major; values[field_r_index][field_z_index]"
        )
        nr, nz = len(field["r_m"]), len(field["z_m"])
        assert nr == 17
        assert nz == 33
        assert all(a < b for a, b in zip(field["r_m"], field["r_m"][1:]))
        assert all(a < b for a, b in zip(field["z_m"], field["z_m"][1:]))
        for key in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t"):
            assert len(field[key]) == nr
            assert all(len(row) == nz for row in field[key])
            assert all(math.isfinite(value) for row in field[key] for value in row)
        for i in range(nr):
            for j in range(nz):
                assert field["b_magnitude_t"][i][j] == pytest.approx(
                    math.hypot(field["b_r_t"][i][j], field["b_z_t"][i][j]),
                    rel=2e-12,
                    abs=2e-15,
                )


def test_source_geometry_polarity_and_dimensions_are_actual(payload) -> None:
    expected_sources = [
        [(1, 0.042, 0.057, -0.0825, -0.0525, 2200.0),
         (1, 0.042, 0.057, 0.0525, 0.0825, 2200.0)],
        [(1, 0.047, 0.067, -0.09, -0.045, 2600.0),
         (-1, 0.047, 0.067, 0.045, 0.09, 2600.0)],
        [(1, 0.06, 0.09, -0.105, -0.055, 3200.0),
         (-1, 0.032, 0.052, -0.025, 0.025, 1800.0),
         (1, 0.06, 0.09, 0.055, 0.105, 3200.0)],
    ]
    for design, expected in zip(payload["designs"], expected_sources, strict=True):
        actual = [
            (
                source["polarity"],
                source["r_inner_m"],
                source["r_outer_m"],
                source["z_min_m"],
                source["z_max_m"],
                source["ampere_turns_a"],
            )
            for source in design["input"]["sources"]
        ]
        assert actual == expected


def test_nulls_maxima_convergence_and_parity_claims_are_evidence_bounded(payload) -> None:
    assert [
        len(design["summary"]["topology"]["axis_nulls"])
        for design in payload["designs"]
    ] == [0, 1, 2]
    assert [design["summary"]["topology"]["status"] for design in payload["designs"]] == [
        "no_resolved_axis_null",
        "resolved_axis_nulls",
        "resolved_axis_nulls",
    ]
    for design in payload["designs"]:
        diagnostics = design["diagnostics"]
        assert diagnostics["converged"] is True
        assert diagnostics["stagnation_detected"] is False
        assert diagnostics["residual_history_l2"][-1] == diagnostics["final_residual_l2"]
        assert diagnostics["relative_residual_l2"] <= (
            design["input"]["solver"]["relative_tolerance"]
        )
        assert "max_flux_reconstruction_identity_t_per_m" in diagnostics
        assert "max_discrete_divergence_t_per_m" not in diagnostics
        assert design["provenance"]["backend"] == "python"
        assert design["parity"]["accepted_artifact_evidence"] is False
        assert "Python artifact provenance" in design["parity"]["artifact_statement"]
        assert "separate verification-suite evidence" in (
            design["parity"]["runtime_statement"]
        )
        assert design["max_locations"]
        assert design["shown_grid_max_t"] <= design["summary"]["b_magnitude_max_t"]
        assert design["summary"]["outer_boundary_b_magnitude_min_t"] >= 0


def test_generation_is_byte_deterministic_and_checked_html_is_current(
    payload, tmp_path: Path
) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "axisymmetric-results.html"
    GENERATOR.generate(output)
    assert output.read_text(encoding="utf-8") == first
    assert CHECKED_HTML.read_text(encoding="utf-8") == first


def test_html_is_self_contained_offline_and_has_no_secret_or_absolute_path(
    payload,
) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    assert '<script id="axisymmetric-data" type="application/json">' in html
    assert "fetch(" not in lowered
    assert "xmlhttprequest" not in lowered
    assert "websocket" not in lowered
    assert "cdn" not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", html, re.I)
    assert not re.search(r"\bhttps?://", html, re.I)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", html)
    assert not re.search(
        r"(?:api[_-]?key|access[_-]?token|client[_-]?secret)"
        r"\s*[:=]\s*['\"][^'\"]+",
        html,
        re.I,
    )


def test_javascript_is_valid_when_node_is_available(payload, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available for JavaScript syntax checking")
    html = GENERATOR.render_html(payload)
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    script = tmp_path / "axisymmetric-results.js"
    script.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run(
        [node, "--check", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_html_uses_canvas_dpr_and_real_psi_contours_without_cell_dom(payload) -> None:
    html = GENERATOR.render_html(payload)
    assert "window.devicePixelRatio" in html
    assert "createImageData" in html
    assert "function contourSegments(psi,rs,zs,level)" in html
    assert "const psi=d.field.psi_wb" in html
    assert "marching-squares isolines computed directly from ψ" in html
    assert "radial-major grid" in html
    assert 'id="field"' in html
    assert "<svg" not in html.lower()
    assert "createElement(\"div\")" not in html


def test_topology_boundary_and_limitations_have_explicit_display_semantics(payload) -> None:
    html = GENERATOR.render_html(payload)
    for status in (
        "degenerate_near_zero_field",
        "resolved_axis_nulls",
        "near_zero_axis_plateau",
        "no_resolved_axis_null",
    ):
        assert status in html
    for null_kind in (
        "sign_changing_sample",
        "sign_changing_interpolated",
        "isolated_sample",
    ):
        assert null_kind in html
    assert "finite-box boundary sample, not an interior physical null" in html
    assert "axis_plateaus" in html
    assert 'n.kind==="isolated_sample"' in html
    for element_id in ("mapLimits", "centreLimits", "wallLimits", "residualLimits"):
        assert f'id="{element_id}"' in html
    assert "not independent divergence or PDE validation" in html


def test_html_has_required_controls_accessibility_and_redraw_hooks(payload) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        'for="design"',
        'for="component"',
        'id="reset"',
        'id="theme"',
        'tabindex="0"',
        'role="img"',
        'aria-live="polite"',
        'e.key==="ArrowLeft"',
        'e.key==="ArrowRight"',
        'e.key==="ArrowDown"',
        'e.key==="ArrowUp"',
        'e.key==="Home"',
        'new ResizeObserver(schedule)',
        'requestAnimationFrame(full?drawAll:drawField)',
    ):
        assert fragment in html
    assert "Linear-vacuum equivalent-current L1a" in payload["warning"]
    assert "validated design" in payload["warning"]
    assert "Accepted CPU/CUDA artifact parity" in html
    assert "Runtime parity tests" in html
    assert "not recorded" in html
    assert "accepted v1.2 serialization" in html
    assert "signed-zero canonical bytes" in html
    assert "historical/read-only, never rewritten in place" in html


def test_embedded_json_round_trips_strictly(payload) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(
        r'<script id="axisymmetric-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    decoded = json.loads(match.group(1), parse_constant=reject_constant)
    assert decoded == payload


def _reseal(value: dict) -> None:
    payload = {key: item for key, item in value.items() if key != "integrity"}
    value["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "field-json-sorted-utf8-signed-zero-v2",
        "payload_sha256": GENERATOR._canonical_payload_sha256(payload),
    }


def _write_sealed(path: Path, value: dict) -> str:
    data = GENERATOR.canonical_field_artifact_bytes(value, representation="file")
    path.write_bytes(data)
    digest = __import__("hashlib").sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _rewrite_artifact_and_manifest(
    manifest_path: Path, manifest: dict, artifact_path: Path, artifact: dict
) -> None:
    _reseal(artifact)
    artifact_file_hash = _write_sealed(artifact_path, artifact)
    manifest["designs"][0]["artifact_file_sha256"] = artifact_file_hash
    manifest["designs"][0]["artifact_payload_sha256"] = (
        artifact["integrity"]["payload_sha256"]
    )
    _reseal(manifest)
    _write_sealed(manifest_path, manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("manifest-name", "authoritative manifest reload rejected"),
        ("superseded-schema", "authoritative manifest reload rejected"),
        ("unknown-key", "authoritative manifest reload rejected"),
        ("matrix-shape", "authoritative manifest reload rejected"),
        ("magnitude", "authoritative manifest reload rejected"),
        ("source", "authoritative manifest reload rejected"),
        ("backend", "authoritative manifest reload rejected"),
        ("nonconverged", "authoritative manifest reload rejected"),
        ("residual", "authoritative manifest reload rejected"),
        ("topology", "authoritative manifest reload rejected"),
        ("old-diagnostic", "authoritative manifest reload rejected"),
    ),
)
def test_resealed_semantic_corruption_is_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path = _copy_results(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = manifest_path.parent / manifest["designs"][0]["artifact"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if mutation in {"manifest-name", "superseded-schema"}:
        if mutation == "manifest-name":
            manifest["designs"][0]["name"] = "lookalike"
        else:
            manifest["schema_version"] = "cft-axisymmetric-design-manifest/1.1.0"
        _reseal(manifest)
        _write_sealed(manifest_path, manifest)
    else:
        if mutation == "unknown-key":
            artifact["field_map"]["unknown"] = 1
        elif mutation == "matrix-shape":
            artifact["field_map"]["psi_wb"].pop()
        elif mutation == "magnitude":
            artifact["field_map"]["b_magnitude_t"][3][4] += 1e-3
        elif mutation == "source":
            artifact["input"]["sources"][0]["r_outer_m"] = 1.0
        elif mutation == "backend":
            artifact["provenance"]["backend"] = "cuda"
        elif mutation == "nonconverged":
            artifact["diagnostics"]["converged"] = False
        elif mutation == "residual":
            artifact["diagnostics"]["final_residual_l2"] = (
                artifact["diagnostics"]["initial_residual_l2"]
            )
            artifact["diagnostics"]["relative_residual_l2"] = 1.0
        elif mutation == "topology":
            artifact["summary"]["topology"]["status"] = "degenerate_near_zero_field"
            manifest["designs"][0]["topology"] = artifact["summary"]["topology"]
        else:
            value = artifact["diagnostics"].pop(
                "max_flux_reconstruction_identity_t_per_m"
            )
            artifact["diagnostics"]["max_discrete_divergence_t_per_m"] = value
        _rewrite_artifact_and_manifest(
            manifest_path, manifest, artifact_path, artifact
        )
    with pytest.raises(ValueError, match=message):
        GENERATOR.build_payload(manifest_path)


def test_manifest_and_artifact_hash_layers_reject_tampering(tmp_path: Path) -> None:
    manifest_path = _copy_results(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = manifest_path.parent / manifest["designs"][0]["artifact"]

    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="file SHA-256|sidecar"):
        GENERATOR.build_payload(manifest_path)

    manifest_path = _copy_results(tmp_path / "payload")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_policy"] += " tampered"
    _write_sealed(manifest_path, manifest)
    with pytest.raises(ValueError, match="canonical payload SHA-256"):
        GENERATOR.build_payload(manifest_path)

    manifest_path = _copy_results(tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = manifest_path.parent / manifest["designs"][0]["artifact"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["model_description"] += " tampered"
    artifact_file_hash = _write_sealed(artifact_path, artifact)
    manifest["designs"][0]["artifact_file_sha256"] = artifact_file_hash
    _reseal(manifest)
    _write_sealed(manifest_path, manifest)
    with pytest.raises(ValueError, match="canonical payload SHA-256"):
        GENERATOR.build_payload(manifest_path)


def test_noncanonical_signed_zero_file_is_rejected(tmp_path: Path) -> None:
    manifest_path = _copy_results(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = manifest_path.parent / manifest["designs"][0]["artifact"]
    raw = artifact_path.read_bytes()
    tampered = raw.replace(b": 0.0", b": -0.0", 1)
    assert tampered != raw
    artifact_path.write_bytes(tampered)
    digest = __import__("hashlib").sha256(tampered).hexdigest()
    artifact_path.with_name(artifact_path.name + ".sha256").write_text(
        f"{digest}  {artifact_path.name}\n", encoding="ascii"
    )
    manifest["designs"][0]["artifact_file_sha256"] = digest
    _reseal(manifest)
    _write_sealed(manifest_path, manifest)
    with pytest.raises(ValueError, match="not canonical"):
        GENERATOR.build_payload(manifest_path)


def test_migration_manifest_anchor_tampering_is_rejected(tmp_path: Path) -> None:
    manifest_path = _copy_results(tmp_path)
    migration_path = manifest_path.parent / "serialization-migration-v1.1-to-v1.2.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    migration["to"]["artifacts"]["hypothetical-compact-mirror"][
        "payload_sha256"
    ] = "0" * 64
    _reseal(migration)
    _write_sealed(migration_path, migration)
    with pytest.raises(ValueError, match="accepted anchors"):
        GENERATOR.build_payload(manifest_path, migration_path)


def test_tampered_embedded_identity_or_parity_claim_is_rejected(payload) -> None:
    changed = deepcopy(payload)
    changed["designs"][0]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="embedded artifact identity"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["manifest"]["payload_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="manifest identity"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["migration"]["payload_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="migration identity"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["designs"][0]["parity"]["accepted_artifact_evidence"] = True
    with pytest.raises(ValueError, match="unrecorded artifact parity"):
        GENERATOR.validate_payload(changed)
