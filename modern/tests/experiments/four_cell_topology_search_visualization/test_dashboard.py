"""Isolated contract tests for the four-cell topology-search dashboard."""

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
    / "four_cell_topology_search"
    / "visualization"
    / "generate_dashboard.py"
)
CHECKED_HTML = GENERATOR_PATH.with_name("four-cell-topology-search.html")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "four_cell_topology_dashboard", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_exact_hashes_counts_taxonomy_and_compatible_candidates(payload) -> None:
    assert payload["provenance"]["dataset_file_sha256"] == (
        GENERATOR.EXPECTED_DATASET_FILE_SHA256
    )
    assert payload["provenance"]["dataset_payload_sha256"] == (
        GENERATOR.EXPECTED_DATASET_PAYLOAD_SHA256
    )
    assert payload["provenance"]["manifest_file_sha256"] == (
        GENERATOR.EXPECTED_MANIFEST_FILE_SHA256
    )
    assert payload["provenance"]["manifest_payload_sha256"] == (
        GENERATOR.EXPECTED_MANIFEST_PAYLOAD_SHA256
    )
    assert len(payload["cases"]) == 128
    assert tuple(
        sorted(case["case_id"] for case in payload["cases"] if case["compatible"])
    ) == GENERATOR.EXPECTED_COMPATIBLE
    assert {item["code"]: item["count"] for item in payload["failure_taxonomy"]} == {
        "FIELD_GATE_FAILURE": 68,
        "TOPOLOGY_COUNT": 118,
        "BOUNDARY_LEAKAGE": 36,
        "MIRROR_INVERTED": 61,
    }
    assert payload["summary"]["plasma_residual_root_count"] == 6
    assert payload["summary"]["identifiable_state_count"] == 0
    assert payload["summary"]["performance_publication_count"] == 0
    GENERATOR.validate_payload(payload)


def test_all_case_metrics_filters_and_overlapping_failures_are_embedded(payload) -> None:
    required = {
        "stage_count",
        "segment_count",
        "confidence",
        "minimum_mirror_ratio",
        "boundary_to_peak_ratio",
        "field_peak_t",
        "relative_residual_l2",
        "source_representation_error",
        "failure_codes",
        "gates",
        "identity",
    }
    assert all(required <= set(case) for case in payload["cases"])
    assert any(len(case["failure_codes"]) >= 3 for case in payload["cases"])
    assert all(len(case["identity"]) == 6 for case in payload["cases"])


def test_two_fields_geometry_sources_topology_and_probability_intervals(payload) -> None:
    assert len(payload["representatives"]) == 2
    for representative in payload["representatives"]:
        field = representative["field"]
        nr, nz = len(field["map"]["r_m"]), len(field["map"]["z_m"])
        assert nr == 65 and nz == 193
        for component in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t"):
            assert len(field["map"][component]) == nr
            assert all(len(row) == nz for row in field["map"][component])
        assert field["sources"]
        assert representative["geometry"]["regions"]
        assert representative["geometry"]["stages"]
        assert len(representative["topology"]) == 4
        for segment in representative["topology"]:
            assert segment["mirror_ratio"] >= 1
            assert 0 <= segment["probability_lower"]
            assert segment["probability"] < 1
            assert segment["probability_upper"] < 1
            assert segment["probability_lower"] <= segment["probability_upper"]


def test_all_six_residual_roots_use_corrected_exact_semantics(payload) -> None:
    outcomes = payload["residual_roots"]
    assert len(outcomes) == 6
    assert all(item["residual_root_found"] for item in outcomes)
    assert all(item["residual_diagnostics"]["jacobian_rank"] == 22 for item in outcomes)
    assert all(
        item["identifiability"]["status"] == "non_identifiable"
        and item["identifiability"]["publication_allowed"] is False
        and item["identifiability"]["full_column_rank"] is False
        for item in outcomes
    )
    assert all(item["conservation_diagnostics"]["closures"] for item in outcomes)
    assert all(item["start_count"] == 9 for item in outcomes)
    forbidden = {
        "state",
        "raw_state",
        "powers",
        "raw_power_diagnostics",
        "screening_performance",
        "valid_state_published",
    }
    assert all(not forbidden.intersection(item) for item in outcomes)


def test_protocol_status_declared_gates_and_semantic_correction(payload) -> None:
    status = payload["protocol_status"]
    correction = payload["semantic_correction"]
    assert status["experiment_version"] == "v1"
    assert status["status"] == "development_evidence_only"
    assert status["preregistered"] is False
    assert status["valid_for_physical_mirror_claims"] is False
    assert status["valid_for_identifiable_state_claims"] is False
    assert status["valid_for_performance_claims"] is False
    assert "deprecated same-z" in " ".join(status["invalidity_reasons"])
    assert correction["kind"] == "semantic_publication_metadata_correction"
    assert correction["numerical_values_modified"] is False
    assert correction["representative_artifacts_modified"] is False
    assert correction["selection_or_ranking_modified"] is False
    assert set(payload["declared_gates"]) == {
        "derivation",
        "field",
        "topology",
        "topology_policy",
        "uncertainty",
    }


def test_cpu_cuda_parity_and_artifact_provenance(payload) -> None:
    assert len(payload["parity"]) == 8
    assert all(item["passed"] for item in payload["parity"])
    assert all(item["cpu_backend"] == "python" for item in payload["parity"])
    assert all(item["warp_backend"] == "warp:cuda:0" for item in payload["parity"])
    assert all(len(item["identity"]) == 4 for item in payload["representatives"])


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


def test_html_is_strictly_offline_path_free_secret_free_and_performance_free(
    payload,
) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    embedded = json.dumps(payload, sort_keys=True)
    for forbidden_key in (
        '"audit_raw_numerical_data"',
        '"raw_state"',
        '"raw_power_diagnostics"',
        '"valid_state_published"',
        '"screening_performance"',
    ):
        assert forbidden_key not in embedded
    assert '<script id="four-cell-data" type="application/json">' in html
    for forbidden in (
        "fetch(",
        "xmlhttprequest",
        "websocket",
        "eventsource",
        "<iframe",
        "<svg",
        "screening_performance",
        "valid_state_published",
        "raw_power_diagnostics",
        "raw_state",
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


def test_accessibility_interactions_maps_markers_and_responsive_dpr(payload) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        'id="caseRows"',
        'id="overview"',
        'id="field"',
        'id="profile"',
        'id="probability"',
        'id="reset"',
        'id="theme"',
        'id="filters"',
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
        "createImageData",
        "function contours(v,rs,zs,l)",
        "NON-IDENTIFIABLE / SCREENING ONLY",
        "Largest upper / nominal contrast",
        "V1 IS SUPERSEDED DEVELOPMENT EVIDENCE",
        "deprecated same-z",
        'data-case-id="${esc(c.case_id)}"',
        "function focusedRowId()",
        "target.focus({preventScroll:true})",
        "select(visible[p],true)",
        "e.preventDefault()",
        "html,body{max-width:100%;overflow-x:hidden}",
        ".table-wrap{max-height:520px;max-width:100%;overflow:auto",
        "word-break:break-all",
    ):
        assert fragment in html


def test_embedded_json_round_trips_and_javascript_parses(payload, tmp_path: Path) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(
        r'<script id="four-cell-data" type="application/json">(.*?)</script>',
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


def test_file_payload_field_and_embedded_identity_tampering_are_rejected(
    payload, tmp_path: Path
) -> None:
    copied = tmp_path / "dataset.json"
    copied.write_bytes((GENERATOR.RESULTS / "dataset.json").read_bytes() + b" ")
    copied.with_name("dataset.json.sha256").write_text(
        (GENERATOR.RESULTS / "dataset.json.sha256").read_text(encoding="ascii"),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="file SHA-256|sidecar"):
        GENERATOR._verify_file(copied, "dataset")

    dataset = GENERATOR._load_object(GENERATOR.RESULTS / "dataset.json", "dataset")
    dataset["summary"]["compatible_count"] = 3
    with pytest.raises(ValueError, match="canonical payload SHA-256"):
        GENERATOR._verify_integrity(dataset, "dataset")

    representative = payload["representatives"][0]
    manifest = GENERATOR._load_object(GENERATOR.RESULTS / "manifest.json", "manifest")
    binding = next(
        item
        for item in manifest["representatives"]
        if item["case_id"] == representative["case_id"]
    )
    field = GENERATOR._load_object(
        GENERATOR.RESULTS / binding["field"]["path"], "field"
    )
    field["field_map"]["b_magnitude_t"][1][1] += 0.1
    with pytest.raises(ValueError, match=r"\|B\| component identity"):
        GENERATOR._validate_field(field, "field")

    changed = deepcopy(payload)
    changed["provenance"]["manifest_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="embedded manifest identity"):
        GENERATOR.validate_payload(changed)
