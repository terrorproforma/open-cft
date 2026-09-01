"""Contract and offline tests for the preregistered L1a sweep-v2 dashboard."""

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
    / "l1a_geometry_sweep_v2"
    / "visualization"
    / "generate_dashboard.py"
)
CHECKED_HTML = GENERATOR_PATH.with_name("l1a-geometry-sweep-v2.html")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "l1a_geometry_sweep_v2_dashboard", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_temporal_commits_and_all_pinned_evidence_hashes_are_exact(payload) -> None:
    identity = payload["identity"]
    assert identity["preregistration_commit_sha"] == (
        GENERATOR.PREREGISTRATION_COMMIT_SHA
    )
    assert identity["results_commit_sha"] == GENERATOR.RESULTS_COMMIT_SHA
    assert identity["relationship"] == (
        "results commit is the direct child of preregistration"
    )
    assert identity["protocol_file_sha256"] == (
        GENERATOR.EXPECTED_PROTOCOL_FILE_SHA256
    )
    assert identity["protocol_payload_sha256"] == (
        GENERATOR.EXPECTED_PROTOCOL_PAYLOAD_SHA256
    )
    assert identity["manifest_file_sha256"] == (
        GENERATOR.EXPECTED_MANIFEST_FILE_SHA256
    )
    assert identity["manifest_payload_sha256"] == (
        GENERATOR.EXPECTED_MANIFEST_PAYLOAD_SHA256
    )
    assert identity["raw_file_sha256"] == GENERATOR.EXPECTED_RAW_FILE_SHA256
    assert identity["raw_payload_sha256"] == (
        GENERATOR.EXPECTED_RAW_PAYLOAD_SHA256
    )
    assert identity["summary_file_sha256"] == (
        GENERATOR.EXPECTED_SUMMARY_FILE_SHA256
    )
    assert identity["summary_payload_sha256"] == (
        GENERATOR.EXPECTED_SUMMARY_PAYLOAD_SHA256
    )
    GENERATOR.validate_payload(payload)


def test_exact_counts_seven_gates_front_and_role_coalescence(payload) -> None:
    summary = payload["summary"]
    assert (
        summary["requested_count"],
        summary["evaluated_count"],
        summary["failed_count"],
        summary["nondominated_count"],
        summary["unique_representative_count"],
    ) == (96, 96, 0, 25, 4)
    assert len(payload["cases"]) == 96
    assert sum(case["nondominated"] for case in payload["cases"]) == 25
    assert len(payload["gates"]) == 7
    assert [gate["gate_id"] for gate in payload["gates"]] == [
        "boundary",
        "residual",
        "cpu_cuda_parity",
        "flux_identity",
        "source_representation",
        "topology_confidence",
        "manufacturability",
    ]
    assert all(
        gate["passed"] and gate["failure_count"] == 0 for gate in payload["gates"]
    )
    assert len(payload["representatives"]) == 4
    assert sum(len(item["roles"]) for item in payload["representatives"]) == 5
    assert [
        (role["role"], role["case_id"])
        for role in payload["summary"]["representative_roles"]
    ] == list(GENERATOR.EXPECTED_ROLES)


def test_bitwise_identity_and_tolerance_replay_semantics_are_distinct(payload) -> None:
    replay = payload["protocol"]["replay_contract"]
    environment = payload["summary"]["environment"]
    assert "must match bitwise" in replay["identity_policy"]
    assert "never required or claimed bitwise-identical" in replay["cuda_policy"]
    assert replay["artifact_policy"] == "artifact hashes identify this run only"
    assert set(replay["scale_aware_tolerances"]) == {
        "psi",
        "magnetic_field_t",
        "gradient_t_per_m",
        "energy_j",
        "dimensionless_qoi",
        "residual",
        "flux_identity_t_per_m",
        "manufacturing_margin_m",
    }
    assert environment["gpu"]["warp_name"] == "NVIDIA GeForce RTX 5090"
    assert environment["gpu"]["architecture"] == "sm_120"
    assert environment["warp"]["version"] == "1.14.0"
    assert environment["scalar"] == "IEEE-754 binary64"
    assert sum(case["parity"] is not None for case in payload["cases"]) == 6


def test_all_linked_metrics_objective_directions_and_case_inputs_are_embedded(
    payload,
) -> None:
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
    objectives = payload["protocol"]["objectives"]
    assert [item["direction"] for item in objectives] == [
        "maximize",
        "maximize",
        "maximize",
        "minimize",
    ]
    for case in payload["cases"]:
        assert len(case["design_values"]) == 11
        assert len(case["design_id"]) == 64
        assert len(case["geometry_sha256"]) == 64
        assert len(case["source_sha256"]) == 64
        assert len(case["config_sha256"]) == 64
        assert len(case["case_sha256"]) == 64


def test_representative_fields_profiles_geometry_sources_and_hashes_are_embedded(
    payload,
) -> None:
    assert tuple(item["case_id"] for item in payload["representatives"]) == (
        GENERATOR.EXPECTED_UNIQUE_REPRESENTATIVES
    )
    for representative in payload["representatives"]:
        field = representative["field"]
        nr, nz = len(field["map"]["r_m"]), len(field["map"]["z_m"])
        assert nr >= 2 and nz >= 2
        for component in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t"):
            assert len(field["map"][component]) == nr
            assert all(len(row) == nz for row in field["map"][component])
        assert len(field["profiles"]["centreline"]["z_m"]) == 145
        assert len(field["profiles"]["wall"]["z_m"]) == 145
        assert representative["geometry"]["regions"]
        assert representative["geometry"]["stages"]
        assert field["sources"]
        assert field["diagnostics"]["converged"] is True
        assert set(representative["identity"]) == {
            "geometry",
            "full_field",
            "downsampled_field",
        }


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
    assert '<script id="sweep-v2-data" type="application/json">' in html
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


def test_html_has_required_evidence_interaction_accessibility_and_redraw(
    payload,
) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        "Preregistration evidence and temporal chain",
        "All seven terminal gates",
        "Bitwise identity",
        "Tolerance-based CUDA replay",
        'id="scatter"',
        'id="parallel"',
        'id="filters"',
        'id="field"',
        'id="profile"',
        'id="representative"',
        'id="component"',
        'id="reset"',
        'id="theme"',
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
        "createImageData",
        "function contours(v,rs,zs,l)",
        "no DOM per point",
    ):
        assert fragment in html
    for term in (
        "no material-aware permanent-magnet model",
        "plasma solution",
        "thrust",
        "efficiency",
        "hardware validity",
    ):
        assert term in payload["warning"]


def test_controls_cannot_expand_mobile_document_and_keep_full_labels_accessible(
    payload,
) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        "html,body{max-width:100%;overflow-x:hidden}",
        "button,select,input,.controls,.controls>*,.field-head,.field-head>*"
        "{min-width:0;max-width:100%}",
        "select{inline-size:100%;max-inline-size:100%;min-inline-size:0;"
        "text-overflow:ellipsis;white-space:nowrap}",
        ".controls select{min-width:0;width:100%}",
        ".field-head>label{flex:0 1 330px;min-width:0;max-width:100%}",
        ".field-head select{display:block;width:100%;min-width:0;max-width:100%}",
        "@media(max-width:590px){.controls,.field-head{display:grid;"
        "grid-template-columns:minmax(0,1fr)",
        ".controls>*,.field-head>*,.controls label,.field-head label"
        "{width:100%;min-width:0;max-width:100%}",
        "function representativeLabel(r)",
        "o.title=fullLabel",
        'o.setAttribute("aria-label",fullLabel)',
        '$("representative").title=label',
        '$("representative").setAttribute("aria-label",'
        "`Unique representative: ${label}`)",
    ):
        assert fragment in html

    # Future role labels remain present in full for title/accessible-name use;
    # visual truncation is CSS-only and does not alter the source label.
    changed = deepcopy(payload)
    long_label = "future-" + "extremely-long-representative-role-" * 12
    changed["representatives"][0]["roles"][0] = long_label
    generated = GENERATOR.render_html(changed)
    assert long_label in generated
    assert "width:465px" not in generated
    assert "min-width:465px" not in generated


def test_embedded_json_round_trips_and_javascript_parses(payload, tmp_path: Path) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(
        r'<script id="sweep-v2-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(match.group(1), parse_constant=reject_constant) == payload
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
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


def test_payload_file_payload_and_field_tampering_are_rejected(
    payload, tmp_path: Path
) -> None:
    changed = deepcopy(payload)
    changed["identity"]["results_commit_sha"] = "0" * 40
    with pytest.raises(ValueError, match="committed identity"):
        GENERATOR.validate_payload(changed)

    raw = GENERATOR._load_object(GENERATOR.RESULTS / "raw-results.json", "raw")
    raw["runtime_diagnostics"]["policy"] += " tampered"
    with pytest.raises(ValueError, match="canonical payload SHA-256"):
        GENERATOR._verify_integrity(raw, "raw")

    copied = tmp_path / "summary.json"
    copied.write_bytes((GENERATOR.RESULTS / "summary.json").read_bytes() + b" ")
    copied.with_name("summary.json.sha256").write_text(
        (GENERATOR.RESULTS / "summary.json.sha256").read_text(encoding="ascii"),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="file SHA-256|sidecar"):
        GENERATOR._verify_file(copied, "summary")

    representative = payload["representatives"][0]
    manifest = GENERATOR._load_object(GENERATOR.RESULTS / "manifest.json", "manifest")
    binding = next(
        item
        for item in manifest["representative_artifacts"]
        if item["case_id"] == representative["case_id"]
    )
    field = GENERATOR._load_object(
        GENERATOR.RESULTS / binding["downsampled_field"]["path"], "field"
    )
    field["field_map"]["b_magnitude_t"][1][1] += 0.1
    with pytest.raises(ValueError, match=r"\|B\| component identity"):
        GENERATOR._validate_field(field, "field")
