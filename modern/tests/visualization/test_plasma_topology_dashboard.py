"""Contract, traceability and offline tests for the plasma-topology results dashboard."""

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
GENERATOR_PATH = MODERN / "visualization" / "generate_plasma_topology_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "plasma-topology-results.template.html"
CHECKED_HTML = MODERN / "visualization" / "plasma-topology-results.html"


def _load_generator():
    spec = importlib.util.spec_from_file_location("plasma_topology_dashboard", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


@pytest.fixture(scope="module")
def html(payload):
    return GENERATOR.render_html(payload)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Determinism and current checked output
# --------------------------------------------------------------------------
def test_generation_is_byte_deterministic_and_checked_html_is_current(payload, html, tmp_path):
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert html == second
    output = tmp_path / "plasma-topology-results.html"
    digest = GENERATOR.generate(output)
    assert output.read_bytes() == html.encode("utf-8")
    assert digest == sha256(html.encode("utf-8")).hexdigest()
    assert CHECKED_HTML.read_bytes().replace(b"\r\n", b"\n") == html.encode("utf-8")
    assert "\r" not in html


def test_size_cap_and_time_policy(payload, html):
    size = len(html.encode("utf-8"))
    assert size < GENERATOR.SIZE_CAP_BYTES
    assert size < 12 * 1024 * 1024
    provenance = payload["provenance"]
    assert provenance["evidence_snapshot_time"] == GENERATOR._git_commit_time(
        GENERATOR.ORBIT_V4_RESULTS_COMMIT
    )
    assert "no wall-clock" in provenance["time_policy"]
    assert provenance["generator_sha256"] == sha256(
        GENERATOR_PATH.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert provenance["template_sha256"] == sha256(
        TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


# --------------------------------------------------------------------------
# Traceability: every embedded number points to a hashed committed file
# --------------------------------------------------------------------------
def test_every_source_is_hashed_committed_and_recomputable(payload):
    sources = payload["sources"]
    assert len(sources) >= 60
    ids = [entry["id"] for entry in sources]
    assert len(ids) == len(set(ids))
    for entry in sources:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert re.fullmatch(r"[0-9a-f]{40}", entry["commit"])
        if entry["experiment_id"] == GENERATOR.ORBIT_V4_ID:
            raw = GENERATOR._git_show(entry["commit"], entry["path"])
        else:
            path = GENERATOR.REPO / entry["path"]
            assert path.is_file(), entry["path"]
            raw = path.read_bytes()
        assert sha256(raw).hexdigest() == entry["sha256"], entry["id"]
    semantic_only = {entry["id"] for entry in sources if not entry["identity_matches"]}
    assert semantic_only == {"wcval-v1-failure", "wcval-v1-report"}
    for entry in sources:
        if entry["id"] in semantic_only:
            assert "CRLF" in entry["note"]


def test_sections_and_claims_reference_verified_sources(payload):
    ids = {entry["id"] for entry in payload["sources"]}
    for key in (
        "characterization", "four_cell_v2", "four_cell_v1", "l1a_sweep", "axisymmetric",
        "p2_divergent_exit", "coupling_v4_validation", "orbit_wall_loss_v4",
    ):
        assert payload[key]["sources"]
        assert set(payload[key]["sources"]) <= ids
    for representative in payload["characterization"]["representatives"]:
        assert set(representative["sources"]) <= ids
    for representative in payload["l1a_sweep"]["representatives"]:
        assert set(representative["sources"]) <= ids
    for campaign in payload["orbit_wall_loss_v4"]["campaigns"]:
        assert set(campaign["sources"]) <= ids
    for claim in payload["ledger"]:
        if claim["status"] == "foundation_only_no_results":
            assert claim["sources"] == []
        else:
            assert claim["sources"] and set(claim["sources"]) <= ids


def test_headline_numbers_match_the_source_files(payload):
    char_manifest = _json(GENERATOR.CHARACTERIZATION_RESULTS / "manifest.json")
    assert payload["characterization"]["summary"] == char_manifest["summary"]
    assert payload["characterization"]["summary"]["stable_eligible_cusp_count"] == 0
    assert payload["characterization"]["summary"]["stable_eligible_cell_count"] == 0
    assert payload["characterization"]["root_class_counts"] == {"X": 520, "O": 532, "degenerate": 224}
    assert sum(payload["characterization"]["zone_counts"].values()) == 1276

    fc2_manifest = _json(GENERATOR.FOUR_CELL_V2_RESULTS / "manifest.json")
    assert payload["four_cell_v2"]["summary"] == fc2_manifest["summary"]
    assert payload["four_cell_v2"]["summary"]["stable_count"] == 0

    fc1_dataset = _json(GENERATOR.FOUR_CELL_V1_RESULTS / "dataset.json")
    assert payload["four_cell_v1"]["compatible_case_ids"] == sorted(
        case["case_id"] for case in fc1_dataset["cases"] if case["topology"]["compatible"]
    )
    assert payload["four_cell_v1"]["summary"]["failure_counts"] == fc1_dataset["summary"]["failure_counts"]

    sweep_summary = _json(GENERATOR.SWEEP_RESULTS / "summary.json")
    assert [gate["observed"] for gate in payload["l1a_sweep"]["gates"]] == [
        gate["observed"] for gate in sweep_summary["terminal_gates"]
    ]
    assert payload["l1a_sweep"]["summary"]["qoi_ranges"] == sweep_summary["qoi_ranges"]
    sweep_raw = _json(GENERATOR.SWEEP_RESULTS / "raw-results.json")
    expected_counts: dict[str, int] = {}
    for case in sweep_raw["cases"]:
        key = str(int(case["qois"]["axis_cusp_count"]))
        expected_counts[key] = expected_counts.get(key, 0) + 1
    assert payload["l1a_sweep"]["axis_cusp_count_counts"] == expected_counts
    assert sum(expected_counts.values()) == 96

    wcval_failure = _json(GENERATOR.WCVAL_V1_RESULTS / "failure.json")
    v1_run = payload["coupling_v4_validation"]["runs"][0]
    assert v1_run["read_only_reconstruction"] == wcval_failure["read_only_reconstruction_from_exact_maps"]
    assert v1_run["coverage"] == wcval_failure["coverage"]

    terminal = json.loads(
        GENERATOR._git_show(
            GENERATOR.ORBIT_V4_RESULTS_COMMIT, GENERATOR.ORBIT_V4_PREFIX + "results/terminal.json"
        )
    )
    for campaign in payload["orbit_wall_loss_v4"]["campaigns"]:
        recorded = terminal["payload"]["campaigns"][campaign["case_id"]]
        assert campaign["wall_hit"] == recorded["wall_hit"]
        assert campaign["escaped"] == recorded["escaped"]
        assert campaign["reflected"] == recorded["reflected"]
        for stratum in campaign["strata"]:
            for outcome in ("wall_hit", "reflected", "domain_escape"):
                interval = stratum[outcome]
                assert interval["method"] == "wilson-95"
                assert 0.0 <= interval["lower"] <= interval["probability"] <= interval["upper"] <= 1.0
                assert interval["probability"] == interval["successes"] / interval["trials"]
    assert payload["orbit_wall_loss_v4"]["orbit_count"] == 4608


def test_field_rasters_are_downsampled_projections_of_the_hashed_artifacts(payload):
    representative = payload["characterization"]["representatives"][2]
    source = _json(
        GENERATOR.CHARACTERIZATION_RESULTS
        / "representatives" / representative["case_id"] / "primary-field.json"
    )
    field = representative["field"]
    assert field["r_m"] == source["field_map"]["r_m"]
    assert field["z_m"] == source["field_map"]["z_m"]
    assert field["b_magnitude_t"][10][20] == GENERATOR._round(source["field_map"]["b_magnitude_t"][10][20])
    assert field["psi_wb"][3][7] == GENERATOR._round(source["field_map"]["psi_wb"][3][7])
    assert field["wall_profile"]["b_r_t"] == source["profiles"]["wall"]["b_r_t"]
    assert field["axis_topology"] == source["summary"]["topology"]
    assert "significant digits" in field["rounding"]
    assert "downsampling" in payload["provenance"]
    p2 = payload["p2_divergent_exit"]
    assert p2["raster"]["grid"] == {"width": 144, "height": 92}
    assert len(p2["raster"]["fields"]["b_magnitude_t"]) == 144 * 92
    assert p2["wall_normal_maxima"]["authority"] == (
        "dashboard_derived_display_diagnostic_not_accepted_cusp_evidence"
    )
    assert 2 <= len(p2["wall_normal_maxima"]["maxima"]) <= 4


# --------------------------------------------------------------------------
# Claim-boundary semantics
# --------------------------------------------------------------------------
def test_claim_boundaries_are_honest(payload):
    ledger = payload["ledger"]
    statuses = [claim["status"] for claim in ledger]
    assert statuses.count("accepted_numerical_evidence") == 4
    assert "preregistered_null_result" in statuses
    assert "superseded_screening_only" in statuses
    assert "rejected_failed_immutable_runs" in statuses
    assert "rejected_code_failures" in statuses
    assert "foundation_only_no_results" in statuses
    assert payload["four_cell_v1"]["protocol_status"]["valid_for_physical_mirror_claims"] is False
    assert "deprecated" in payload["four_cell_v1"]["mirror_proxy_warning"]
    assert payload["coupling_v4_validation"]["criterion"]["promotion_status"] == "not_promoted"
    for run in payload["coupling_v4_validation"]["runs"]:
        assert run["promotion"]["criterion_numerically_promoted"] is False
    orbit = payload["orbit_wall_loss_v4"]
    assert orbit["limitations"]["forbid_mirror_formula_publication"] is True
    assert orbit["limitations"]["forbid_pic_or_self_consistent_claim"] is True
    assert orbit["limitations"]["hardware_or_experimental_validation"] is False
    for version in ("v1", "v2", "v3"):
        assert orbit["prior_campaigns"][version]["terminal_state"].endswith("failure")
    for representative in payload["characterization"]["representatives"]:
        assert representative["mirror_ratio"]["status"] == "not_computed"
    for representative in payload["l1a_sweep"]["representatives"]:
        assert representative["mirror_ratio"]["status"] == "recorded_l1a_screening_qoi"
    for term in ("no PIC result", "no experimental or hardware validation", "mirror-formula"):
        assert term in payload["warning"]
    assert payload["overview"]["fidelity_ladder"][-1]["rung"] == "PIC"
    assert "no results" in payload["overview"]["fidelity_ladder"][-1]["status"]


# --------------------------------------------------------------------------
# HTML validity, offline containment, accessibility
# --------------------------------------------------------------------------
def test_html_validity_basics(html):
    assert html.startswith("<!doctype html>\n<html lang=\"en\"")
    assert html.count("<html") == 1 and html.count("</html>") == 1
    assert html.count("<head>") == 1 and html.count("</head>") == 1
    assert html.count("<body>") == 1 and html.count("</body>") == 1
    assert '<meta charset="utf-8">' in html
    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in html
    assert html.count("<title>") == 1
    markup = html.split(f'<script id="{GENERATOR.DATA_SCRIPT_ID}"')[0]
    for tag in ("header", "nav", "main", "footer", "section", "canvas", "select", "div", "h2", "h3", "p", "ul"):
        opened = len(re.findall(rf"<{tag}[\s>]", markup))
        closed = markup.count(f"</{tag}>")
        assert opened == closed, (tag, opened, closed)
    assert html.count(f'<script id="{GENERATOR.DATA_SCRIPT_ID}" type="application/json">') == 1
    ids = re.findall(r'\sid="([^"]+)"', html.split(f'<script id="{GENERATOR.DATA_SCRIPT_ID}"')[0])
    assert len(ids) == len(set(ids)), "duplicate element ids"


def test_embedded_json_round_trips_strictly(payload, html):
    match = re.search(
        rf'<script id="{GENERATOR.DATA_SCRIPT_ID}" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None

    def reject(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(match.group(1), parse_constant=reject) == payload
    assert "</" not in match.group(1).replace("<\\/", "")


def test_html_is_self_contained_offline_path_free_and_secret_free(html):
    lowered = html.lower()
    for forbidden in ("fetch(", "xmlhttprequest", "websocket", "eventsource", "<iframe", "<img", "@import", "cdn"):
        assert forbidden not in lowered, forbidden
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", html, re.I)
    assert not re.search(r"\bhttps?://", html, re.I)
    assert not re.search(r"\bwww\.", html, re.I)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", html)
    assert not re.search(
        r"(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]+", html, re.I
    )
    assert '<link rel="icon" href="data:,">' in html


def test_accessibility_landmarks_labels_and_keyboard_hooks(html):
    for fragment in (
        '<a class="skip" href="#main">',
        "<header",
        '<nav class="shell" aria-label="Sections">',
        '<main id="main"',
        "<footer",
        'role="note"',
        'aria-live="polite"',
        'aria-pressed="false"',
        'tabindex="0"',
        'e.key==="Home"',
        'e.key==="Escape"',
        'e.key==="ArrowLeft"',
        'e.key==="ArrowUp"',
        "new ResizeObserver",
        "requestAnimationFrame",
        "window.devicePixelRatio",
        "createImageData",
        "function contourSegments(v,rs,zs,level)",
    ):
        assert fragment in html, fragment
    assert html.count("<h1") == 1
    sections = re.findall(r'<section id="([^"]+)" aria-labelledby="([^"]+)">', html)
    assert len(sections) == 8
    for section_id, heading_id in sections:
        assert f'<h2 id="{heading_id}">' in html
    canvases = re.findall(r"<canvas[^>]*>", html)
    assert len(canvases) == 12
    for canvas in canvases:
        assert 'role="img"' in canvas and "aria-label=" in canvas
    selects = re.findall(r'<select id="([^"]+)"', html)
    assert selects
    for select_id in selects:
        assert f'<label for="{select_id}">' in html
    assert "html,body{max-width:100%;overflow-x:hidden}" in html
    assert "select{inline-size:100%;max-inline-size:100%;min-inline-size:0;" in html
    assert "@media(max-width:590px)" in html and "@media(max-width:900px)" in html
    assert "100vw" not in html


def test_javascript_parses_when_node_is_available(html, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    path = tmp_path / "dashboard.js"
    path.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_dashboard_text_states_downsampling_and_boundaries(html):
    for fragment in (
        "downsampling stated per design",
        "marching-squares contours",
        "foundation code only",
        "display diagnostic, not accepted cusp evidence",
        "Wilson 95",
        "Failed campaigns v1–v3 (code, never physics)",
        "Validation ledger",
        "Provenance",
    ):
        assert fragment in html, fragment


# --------------------------------------------------------------------------
# Tamper rejection
# --------------------------------------------------------------------------
def test_validate_payload_rejects_semantic_tampering(payload):
    changed = deepcopy(payload)
    changed["characterization"]["summary"]["stable_eligible_cusp_count"] = 3
    with pytest.raises(ValueError, match="characterization embedded counts"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["four_cell_v2"]["candidates"][0]["stable"] = True
    with pytest.raises(ValueError, match="four-cell v2 embedded counts"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["four_cell_v1"]["protocol_status"]["valid_for_physical_mirror_claims"] = True
    with pytest.raises(ValueError, match="four-cell v1 embedded semantics"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["coupling_v4_validation"]["criterion"]["promotion_status"] = "promoted"
    with pytest.raises(ValueError, match="coupling v4 validation"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["orbit_wall_loss_v4"]["limitations"]["forbid_mirror_formula_publication"] = False
    with pytest.raises(ValueError, match="orbit v4 embedded semantics"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["ledger"][0]["sources"] = ["missing-source"]
    with pytest.raises(ValueError, match="not traceable"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["provenance"]["evidence_base_commit"] = "0" * 40
    with pytest.raises(ValueError, match="evidence base commit"):
        GENERATOR.validate_payload(changed)


def test_file_and_payload_hash_layers_reject_tampering(tmp_path):
    source = GENERATOR.SWEEP_RESULTS / "summary.json"
    copied = tmp_path / "summary.json"
    copied.write_bytes(source.read_bytes() + b" ")
    copied.with_name("summary.json.sha256").write_text(
        source.with_name("summary.json.sha256").read_text(encoding="ascii"), encoding="ascii"
    )
    with pytest.raises(ValueError, match="file SHA-256|sidecar"):
        GENERATOR._verify_sidecar_file(copied, "summary", GENERATOR.SWEEP_SUMMARY_FILE_SHA256)
    value = GENERATOR._load_object(source, "summary")
    value["terminal_status"] = "REJECTED"
    with pytest.raises(ValueError, match="canonical payload SHA-256"):
        GENERATOR._verify_integrity(value, "summary")
    field = GENERATOR._load_object(
        GENERATOR.CHARACTERIZATION_RESULTS / "representatives" / "topology-s02-p0-r0-neg" / "primary-field.json",
        "field",
    )
    field["field_map"]["b_magnitude_t"][1][1] += 0.1
    with pytest.raises(ValueError, match=r"\|B\| component identity"):
        GENERATOR._validate_l1a_field(field, "field")


def test_characterization_manifest_tampering_is_refused(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name in ("manifest.json", "report.md"):
        shutil.copy(GENERATOR.CHARACTERIZATION_RESULTS / name, results / name)
    manifest = _json(results / "manifest.json")
    manifest["summary"]["stable_eligible_cusp_count"] = 4
    (results / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(GENERATOR, "CHARACTERIZATION_RESULTS", results)
    with pytest.raises(ValueError, match="characterization manifest file SHA-256 mismatch"):
        GENERATOR._characterization(GENERATOR.SourceLedger())


def test_orbit_v4_pinned_manifest_and_commit_chain_are_exact():
    raw = GENERATOR._git_show(
        GENERATOR.ORBIT_V4_RESULTS_COMMIT, GENERATOR.ORBIT_V4_PREFIX + "results/manifest.json"
    )
    assert sha256(raw).hexdigest() == GENERATOR.ORBIT_V4_MANIFEST_FILE_SHA256
    parents = GENERATOR._git("rev-list", "--parents", "-n", "1", GENERATOR.ORBIT_V4_RESULTS_COMMIT)
    assert parents.decode("ascii").split() == [
        GENERATOR.ORBIT_V4_RESULTS_COMMIT, GENERATOR.ORBIT_V4_PREREGISTRATION_COMMIT
    ]
    with pytest.raises(ValueError, match="Git evidence check failed"):
        GENERATOR._git_show(GENERATOR.ORBIT_V4_RESULTS_COMMIT, GENERATOR.ORBIT_V4_PREFIX + "results/missing.json")
