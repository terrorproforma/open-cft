"""Tests for the standalone CFT full-orbit wall-loss v4 results dashboard.

Every check here reads the sealed results bundle independently of the
generator and compares the embedded payload against it, so the dashboard
cannot show a number that does not trace to a hash-bound artifact.
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_wall_loss_v4_dashboard.py"
TEMPLATE_PATH = MODERN / "visualization" / "wall-loss-v4-results.template.html"
CHECKED_HTML = MODERN / "visualization" / "wall-loss-v4-results.html"
EXPERIMENT = MODERN / "experiments" / "cft_orbit_wall_loss_v4"
RESULTS = EXPERIMENT / "results"


def _load_generator():
    spec = importlib.util.spec_from_file_location("wall_loss_v4_dashboard", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


@pytest.fixture(scope="module")
def html(payload):
    return GENERATOR.render_html(payload)


def _copy_results(tmp_path: Path) -> Path:
    target = tmp_path / "results"
    shutil.copytree(RESULTS, target)
    return target


# --------------------------------------------------------------------------- #
# Bundle identity and the disclosed sidecar defect
# --------------------------------------------------------------------------- #
def test_bundle_identity_hashes_and_tolerated_sidecars_are_exact(payload) -> None:
    identity = payload["identity"]
    manifest = _json(RESULTS / "manifest.json")
    assert identity["manifest_file_sha256"] == _sha256(RESULTS / "manifest.json")
    assert identity["manifest_file_sha256"] == GENERATOR.EXPECTED_MANIFEST_SHA256
    assert identity["terminal_file_sha256"] == manifest["terminal_byte_sha256"]
    assert identity["lock_file_sha256"] == manifest["lock_byte_sha256"]
    assert identity["results_commit_sha"] == GENERATOR.RESULTS_COMMIT_SHA
    assert identity["preregistration_commit_sha"] == GENERATOR.PREREGISTRATION_COMMIT_SHA
    assert identity["execution_lock_commit"] == _json(RESULTS / "execution-lock.json")["commit"]
    assert identity["artifact_count"] == manifest["artifact_count"] == 407
    files = [entry for entry in manifest["artifacts"] if entry["type"] == "file"]
    assert identity["verified_file_count"] == len(files) == 387
    tolerated = {item["path"] for item in identity["sidecar_tolerance"]["tolerated"]}
    assert tolerated == {f"artifacts/orbits/{case}.json.sha256" for case in GENERATOR.CASES}
    assert len(tolerated) == 9
    for entry in files:
        path = RESULTS / entry["path"]
        raw = path.read_bytes()
        assert identity["artifact_hashes"][entry["path"]] == hashlib.sha256(raw).hexdigest()
        if entry["path"] in tolerated:
            assert b"\r" not in raw
            assert hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest() == entry["byte_sha256"]
            assert hashlib.sha256(raw).hexdigest() != entry["byte_sha256"]
        else:
            assert hashlib.sha256(raw).hexdigest() == entry["byte_sha256"]
    for item in identity["sidecar_tolerance"]["tolerated"]:
        assert item["recorded_bytes"] == item["checkout_bytes"] + 1
    assert identity["generator_sha256"] == hashlib.sha256(
        GENERATOR_PATH.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert identity["template_sha256"] == hashlib.sha256(
        TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    GENERATOR.validate_payload(payload)


def test_only_the_nine_orbit_sidecars_are_tolerated(tmp_path: Path) -> None:
    results = _copy_results(tmp_path)
    # A byte change elsewhere must fail even when it is a pure CRLF rewrite.
    victim = results / "artifacts" / "gates.json.sha256.json"
    victim.write_bytes(victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch|size mismatch"):
        GENERATOR.build_payload(results, EXPERIMENT)

    results = _copy_results(tmp_path / "content")
    victim = results / "artifacts" / "campaign-result.json"
    victim.write_bytes(victim.read_bytes().replace(b'"passed":289', b'"passed":290'))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        GENERATOR.build_payload(results, EXPERIMENT)

    results = _copy_results(tmp_path / "sidecar")
    victim = results / "artifacts" / "orbits" / "primary-N.json.sha256"
    victim.write_bytes(victim.read_bytes().replace(b"primary-N-orbit", b"primary-X-orbit"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        GENERATOR.build_payload(results, EXPERIMENT)

    results = _copy_results(tmp_path / "gz")
    victim = results / "artifacts" / "orbits" / "refined-4N.json.gz"
    victim.write_bytes(victim.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        GENERATOR.build_payload(results, EXPERIMENT)


def test_tampered_embedded_identity_or_gate_claim_is_rejected(payload) -> None:
    changed = deepcopy(payload)
    changed["identity"]["results_commit_sha"] = "0" * 40
    with pytest.raises(ValueError, match="committed identity"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["identity"]["sidecar_tolerance"]["tolerated"].pop()
    with pytest.raises(ValueError, match="tolerated sidecar set"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["headline"]["binding_gates_passed"] = 14
    with pytest.raises(ValueError, match="headline differs"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["cases"][0]["estimands"]["reflected"]["successes"] = 1
    with pytest.raises(ValueError, match="zero-reflection"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["diagnostics"]["mu_variation"]["pooled"]["binding"] = True
    with pytest.raises(ValueError, match="must not be a gate"):
        GENERATOR.validate_payload(changed)
    changed = deepcopy(payload)
    changed["orbits"]["columns"]["steps"].pop()
    with pytest.raises(ValueError, match="orbit columns"):
        GENERATOR.validate_payload(changed)


# --------------------------------------------------------------------------- #
# Traceability: every embedded number comes from an artifact
# --------------------------------------------------------------------------- #
def test_headline_and_case_estimands_trace_to_campaign_result(payload) -> None:
    campaign = _json(RESULTS / "artifacts" / "campaign-result.json")
    terminal = _json(RESULTS / "terminal.json")
    gates = _json(RESULTS / "artifacts" / "gates.json")
    assert terminal["payload"] == campaign
    headline = payload["headline"]
    assert headline["state"] == terminal["state"] == "accepted_result"
    assert headline["status"] == campaign["status"] == "accepted"
    assert headline["classification"] == campaign["classification"] == (
        "collisionless_prescribed_field_test_particle_wall_loss_not_pic"
    )
    assert headline["case_count"] == campaign["campaign_count"] == 9
    assert headline["launches_per_case"] == campaign["launches_per_case"] == 512
    assert headline["orbit_count"] == campaign["orbit_count"] == 4608
    assert headline["validators"] == campaign["validators"] == {"passed": 289, "failed": 0}
    assert headline["gate_checks"] == gates["checks"]
    assert len(headline["gate_checks"]) == 15 and all(headline["gate_checks"].values())
    assert headline["binding_gates_passed"] == headline["binding_gate_count"] == 15
    assert headline["attempt_count"] == terminal["counts"]["attempt_count"] == 1
    for case in payload["cases"]:
        block = campaign["campaigns"][case["id"]]
        assert case["termination_counts"] == dict(sorted(block["termination_counts"].items()))
        for key in ("wall_hit", "escaped", "reflected", "incomplete"):
            embedded = case["estimands"][key]
            for field in ("successes", "trials", "probability", "lower", "upper"):
                assert embedded[field] == block[key][field]
            assert block[key]["method"] == "wilson-95"
        assert case["estimands"]["reflected"]["successes"] == 0
        assert case["estimands"]["incomplete"]["successes"] == 0
    pooled = headline["pooled"]
    assert pooled["derived"] is True
    assert pooled["wall_hit"]["successes"] == sum(
        campaign["campaigns"][c]["termination_counts"]["wall_hit"] for c in GENERATOR.CASES
    ) == 2962
    assert pooled["escaped"]["successes"] == sum(
        campaign["campaigns"][c]["termination_counts"]["domain_escape"] for c in GENERATOR.CASES
    ) == 1646
    assert pooled["trials"] == 4608
    assert pooled["wall_hit"]["probability"] == pytest.approx(2962 / 4608)
    assert pooled["reflected"]["successes"] == 0 and pooled["incomplete"]["successes"] == 0


def test_strata_trace_to_summaries_and_show_bimodality(payload) -> None:
    for case in payload["cases"]:
        summary = _json(RESULTS / "artifacts" / "summaries" / f"{case['id']}.json")
        assert case["artifacts"]["summary_sha256"] == _sha256(
            RESULTS / "artifacts" / "summaries" / f"{case['id']}.json"
        )
        assert len(case["strata"]) == len(summary["strata"]) == 32
        for row, source in zip(case["strata"], summary["strata"], strict=True):
            assert row["cell_id"] == source["cell_id"]
            assert row["kinetic_energy_ev"] == source["kinetic_energy_ev"]
            assert row["pitch_angle_deg"] == source["pitch_angle_deg"]
            assert row["parallel_direction"] == source["parallel_direction"]
            assert row["trials"] == source["trials"] == 16
            assert row["wall_hit"] == source["termination_counts"]["wall_hit"]
            assert row["domain_escape"] == source["termination_counts"]["domain_escape"]
            assert row["wall_hit_lower"] == source["wall_hit"]["lower"]
            assert row["wall_hit_upper"] == source["wall_hit"]["upper"]
        assert sum(r["wall_hit"] for r in case["strata"]) == case["termination_counts"]["wall_hit"]
        saturated = [r for r in case["strata"] if r["cell"] in (2, 3)]
        assert saturated and all(r["wall_hit"] == 16 for r in saturated)
        exit_cell = [r for r in case["strata"] if r["cell"] == 4]
        assert exit_cell and all(r["domain_escape"] == 16 for r in exit_cell)
        cell_one_minus = [r for r in case["strata"] if r["cell"] == 1 and r["parallel_direction"] < 0]
        cell_one_plus = [r for r in case["strata"] if r["cell"] == 1 and r["parallel_direction"] > 0]
        assert all(r["wall_hit"] == 16 for r in cell_one_minus)
        assert sum(r["wall_hit"] for r in cell_one_plus) < sum(r["trials"] for r in cell_one_plus) / 2
    pooled = payload["pooled_strata"]
    assert len(pooled) == 32
    for index, row in enumerate(pooled):
        assert row["trials"] == 144
        assert row["wall_hit"] == sum(case["strata"][index]["wall_hit"] for case in payload["cases"])
    cells = payload["design"]["cells"]
    assert [round(cell["z_m"] * 1e3, 6) for cell in cells] == [3.5, 9.5, 15.5, 21.5]
    assert all([round(r * 1e3, 6) for r in cell["radii_m"]] == [1.35, 1.6] for cell in cells)
    assert cells[3]["z_m"] > payload["geometry"]["wall"]["z_max_m"]


def test_orbit_endpoints_trace_to_orbit_artifacts(payload) -> None:
    columns = payload["orbits"]["columns"]
    assert payload["orbits"]["count"] == 4608
    assert "no downsampling" in payload["orbits"]["policy"]
    codes = payload["orbits"]["termination_codes"]
    conditions = payload["orbits"]["condition_codes"]
    offset = 0
    wall = payload["geometry"]["wall"]
    for case in payload["cases"]:
        path = RESULTS / "artifacts" / "orbits" / f"{case['id']}.json.gz"
        assert case["artifacts"]["orbits_sha256"] == _sha256(path)
        raw = gzip.decompress(path.read_bytes())
        sidecar = (RESULTS / "artifacts" / "orbits" / f"{case['id']}.json.sha256").read_text("ascii")
        assert sidecar == f"{hashlib.sha256(raw).hexdigest()}  {case['id']}-orbit.json\n"
        artifact = json.loads(raw)
        assert artifact["integrity"]["payload_sha256"] == case["artifacts"]["orbits_payload_sha256"]
        results = artifact["results"]
        assert len(results) == 512
        mu_values = []
        for local, result in enumerate(results):
            i = offset + local
            assert columns["case"][i] == case["index"]
            assert codes[columns["termination"][i]] == result["termination"]
            assert conditions[columns["condition"][i]] == result["event_witness"]["condition"]
            x, y, z = result["final_position_m"]
            assert columns["final_z_mm"][i] == round(z * 1e3, 5)
            assert columns["final_r_mm"][i] == round((x * x + y * y) ** 0.5 * 1e3, 5)
            assert columns["steps"][i] == result["steps"]
            assert columns["mu_variation"][i] == pytest.approx(
                result["maximum_instantaneous_mu_relative_variation"], rel=1e-4
            )
            assert result["maximum_relative_energy_error"] == 0.0
            if result["termination"] == "wall_hit":
                assert abs(columns["final_r_mm"][i] - wall["radius_m"] * 1e3) < 1e-3
                assert wall["z_min_m"] * 1e3 - 1e-3 <= columns["final_z_mm"][i] <= wall["z_max_m"] * 1e3 + 1e-3
            mu_values.append(result["maximum_instantaneous_mu_relative_variation"])
        diagnostics = case["diagnostics"]
        assert diagnostics["mu"]["min"] == min(mu_values)
        assert diagnostics["mu"]["max"] == max(mu_values)
        assert diagnostics["mu"]["median"] == pytest.approx(statistics.median(mu_values))
        assert diagnostics["mu"]["count_above_0p1"] == sum(v > 0.1 for v in mu_values)
        assert diagnostics["steps"]["total"] == sum(r["steps"] for r in results)
        offset += 512
    assert offset == 4608
    assert set(columns["termination"]) == {0, 1}
    assert codes == ["wall_hit", "domain_escape"]


def test_convergence_field_and_diagnostics_trace_to_artifacts(payload) -> None:
    convergence = _json(RESULTS / "artifacts" / "probability-convergence.json")
    protocol = _json(EXPERIMENT / "protocol.json")
    embedded = payload["convergence"]
    assert embedded["threshold"] == protocol["gates"]["maximum_successive_probability_change"] == 0.01
    for chain, source in zip(embedded["timestep"], convergence["timestep"], strict=True):
        assert chain["map_role"] == source["map_role"]
        assert chain["probabilities"] == source["probabilities"]
        assert chain["successive_changes"] == source["successive_changes"]
        assert chain["adjacent_wilson_overlap"] == source["adjacent_wilson_overlap"]
        assert max(chain["successive_changes"]) <= embedded["threshold"]
    for chain, source in zip(embedded["cross_map"], convergence["cross_map"], strict=True):
        assert chain["timestep_policy"] == source["timestep_policy"]
        assert chain["probabilities"] == source["probabilities"]
        assert chain["successive_changes"] == source["successive_changes"]
    gates = _json(RESULTS / "artifacts" / "gates.json")
    mu = payload["diagnostics"]["mu_variation"]["pooled"]
    source_mu = gates["diagnostics_not_gates"]["magnetic_moment_variation"]
    assert (mu["min"], mu["median"], mu["max"]) == (source_mu["min"], source_mu["median"], source_mu["max"])
    assert mu["count_above_0p1"] == source_mu["count_above_0p1"] == 2786
    assert mu["count_above_0p5"] == source_mu["count_above_0p5"] == 209
    assert mu["binding"] is False and mu["role"] == "diagnostic_only"
    assert 0.60 < mu["fraction_above_0p1"] < 0.61
    tolerance_close = payload["diagnostics"]["tolerance_close"]
    assert tolerance_close["total_events"] == sum(
        _json(RESULTS / "artifacts" / "summaries" / f"{case}.json")["diagnostics"][
            "tolerance_close_event_count"
        ]
        for case in GENERATOR.CASES
    )
    assert 0.42 < tolerance_close["share"] < 0.44
    assert payload["diagnostics"]["energy"]["maximum_relative_energy_error"] == 0.0
    assert payload["diagnostics"]["energy"]["gate_limit"] == gates["energy_gate_limit"] == 1e-10
    timing = payload["diagnostics"]["timing"]
    campaign = _json(RESULTS / "artifacts" / "campaign-result.json")
    assert timing["integration_wall_s"] == campaign["execution_mode"]["integration_wall_s"]
    assert timing["assessment_wall_s"] == campaign["execution_mode"]["assessment_wall_s"]
    assert timing["export_wall_s"] == campaign["execution_mode"]["export_wall_s"]
    assert 660 < timing["lifecycle_wall_s"] < 670
    for case in payload["cases"]:
        summary = _json(RESULTS / "artifacts" / "summaries" / f"{case['id']}.json")
        assert case["timing_s"] == summary["timing_s"]
        assert case["diagnostics"]["runtime_max_b_t"] == summary["diagnostics"]["runtime_max_b_t"]
        assert case["preflight"]["maximum_declared_b_t"] == summary["preflight"]["maximum_declared_b_t"]
        assert case["diagnostics"]["runtime_max_b_t"] <= case["preflight"]["maximum_declared_b_t"]
    field = payload["field"]
    assert 0.2136 < field["runtime_max_b_t_min"] <= field["runtime_max_b_t_max"] < 0.2142
    for projection in field["maps"]:
        evidence = _json(RESULTS / "artifacts" / "field-evidence" / f"{projection['role']}.json")
        grid = _json(RESULTS / "artifacts" / "fields" / f"{projection['role']}.json")
        assert projection["artifact_sha256"] == _sha256(
            RESULTS / "artifacts" / "fields" / f"{projection['role']}.json"
        )
        assert projection["certificate"] == evidence["certificate"]
        assert projection["certified_max_b_t"] == evidence["certificate"]["certified_max_b_t"]
        assert projection["field_error_report"] == evidence["field_error_report"]
        assert projection["r_m"] == grid["r_m"] and projection["z_m"] == grid["z_m"]
        assert len(projection["b_magnitude_t"]) == len(grid["r_m"])
        assert all(len(row) == len(grid["z_m"]) for row in projection["b_magnitude_t"])
        assert projection["grid_max_b_t"] <= projection["certified_max_b_t"]
        assert all(v == pytest.approx(g, rel=1e-5, abs=1e-12) for row, src in zip(projection["psi_wb"], grid["psi_wb"]) for v, g in zip(row, src))
        for value in projection["maximum_declared_b_t_by_case"].values():
            assert value == pytest.approx(projection["certified_max_b_t"], abs=1e-15)
    assert field["cross_map_convergence"] == _json(RESULTS / "artifacts" / "field-map-convergence.json")
    p2 = _json(RESULTS / "artifacts" / "p2-input-authority.json")
    assert field["p2_authority"]["manifest_file_sha256"] == p2["manifest_file_sha256"]
    assert field["p2_authority"]["qualification_status"] == "NUMERICAL_P2_QUALIFIED"


def test_claim_boundary_and_lineage_strings_are_verbatim(payload) -> None:
    protocol = _json(EXPERIMENT / "protocol.json")
    campaign = _json(RESULTS / "artifacts" / "campaign-result.json")
    boundary = payload["claim_boundary"]
    assert boundary["classification"] == (
        "collisionless_prescribed_field_test_particle_wall_loss_not_pic"
    )
    assert boundary["limitations"] == campaign["limitations"] == protocol["publication_boundary"]
    assert boundary["limitations"]["forbid_pic_or_self_consistent_claim"] is True
    assert boundary["limitations"]["hardware_or_experimental_validation"] is False
    assert boundary["coupling"]["integration_status"] == "export_only_pending_consumer_integration"
    coupling = _json(RESULTS / "artifacts" / "coupling-export-only.json")
    assert boundary["coupling"]["probability"] == coupling["probability"]
    assert boundary["coupling"]["handoff_case"] == "refined-4N"
    readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert boundary["readme_statement"] in readme
    assert "mirror" in boundary["reflection_note"] and "bimodal" in boundary["pooled_note"]
    lineage = payload["lineage"]
    disclosure = protocol["prior_campaign_disclosure"]
    assert [p["version"] for p in lineage["prior_campaigns"]] == ["v1", "v2", "v3"]
    for prior in lineage["prior_campaigns"]:
        source = disclosure[prior["version"]]
        assert prior["terminal_state"] == source["terminal_state"]
        assert prior["primary_error_message"] == source["primary_error_message"]
        assert prior["root_cause"] == source["root_cause"]
    assert [p["primary_error_message"] for p in lineage["prior_campaigns"]] == [
        "launch manifest differs from preregistered authority",
        "ordered launch/result/campaign identities are inconsistent",
        "physical event witness requires a positive step",
    ]
    assert lineage["shakedown_rule"] == disclosure["shakedown_rule"]
    assert lineage["v1_6_fix"] == protocol["gates"]["energy_gate_note"]
    devlog = (EXPERIMENT / "DEVLOG.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert lineage["second_latent_v3_bug"] in devlog
    assert "zip(ordered[:-1], ordered[1:], strict=True)" in lineage["second_latent_v3_bug"]
    assert lineage["devlog_sha256"] == hashlib.sha256(devlog.encode("utf-8")).hexdigest()
    shakedown = _json(RESULTS / "artifacts" / "shakedown.json")
    assert lineage["shakedown"]["passed"] is True and lineage["shakedown"]["evidentiary"] is False
    assert lineage["shakedown"]["git_head"] == shakedown["git"]["head"]
    assert lineage["shakedown"]["launch_count"] == 576


# --------------------------------------------------------------------------- #
# Rendering: determinism, offline, size, accessibility
# --------------------------------------------------------------------------- #
def test_generation_is_byte_deterministic_and_checked_html_is_current(
    payload, html, tmp_path: Path
) -> None:
    again = GENERATOR.render_html(GENERATOR.build_payload())
    assert again == html
    output = tmp_path / "wall-loss-v4-results.html"
    digest = GENERATOR.generate(output)
    assert output.read_bytes() == html.encode("utf-8")
    assert digest == hashlib.sha256(html.encode("utf-8")).hexdigest()
    assert CHECKED_HTML.read_bytes() == html.encode("utf-8")
    assert b"\r" not in CHECKED_HTML.read_bytes()


def test_html_is_self_contained_offline_within_size_cap_and_free_of_paths(html) -> None:
    lowered = html.lower()
    assert '<script id="wall-loss-v4-data" type="application/json">' in html
    for forbidden in ("fetch(", "xmlhttprequest", "websocket", "eventsource", "<iframe", "cdn", "<link", "@import", "url("):
        assert forbidden not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", html, re.I)
    assert not re.search(r"\bhttps?://", html, re.I)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", html)
    assert "AppData" not in html
    assert not re.search(
        r"(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]+", html, re.I
    )
    size = len(html.encode("utf-8"))
    assert size <= GENERATOR.MAX_HTML_BYTES
    assert size > 300_000  # all 4608 endpoints and three field maps are embedded
    assert str(GENERATOR.MAX_HTML_BYTES) in html


def test_html_has_provenance_footer_controls_accessibility_and_redraw_hooks(html, payload) -> None:
    identity = payload["identity"]
    for fragment in (
        identity["results_commit_sha"],
        identity["preregistration_commit_sha"],
        identity["manifest_file_sha256"],
        identity["terminal_file_sha256"],
        identity["generator_sha256"],
        identity["template_sha256"],
        'for="case"',
        'for="map"',
        'id="reset"',
        'id="theme"',
        'aria-pressed="false"',
        'tabindex="0"',
        'role="img"',
        'aria-live="polite"',
        'aria-labelledby="h-strata"',
        'aria-labelledby="h-geom"',
        'aria-labelledby="h-field"',
        'aria-labelledby="h-diag"',
        'aria-labelledby="h-claim"',
        'aria-labelledby="h-lineage"',
        'e.key==="ArrowLeft"',
        'e.key==="ArrowRight"',
        'e.key==="ArrowUp"',
        'e.key==="ArrowDown"',
        'e.key==="Home"',
        "new ResizeObserver(",
        "requestAnimationFrame(fn)",
        "window.devicePixelRatio",
        "createImageData",
        "function contourSegments(psi,rs,zs,level)",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "@media(max-width:1000px)",
        "@media(max-width:560px)",
        "html,body{max-width:100%;overflow-x:hidden}",
        "diagnostic only",
        "non-adiabatic cusp physics",
        "domain-escape authority",
        "verbatim",
    ):
        assert fragment in html, fragment
    assert "<svg" not in html.lower()
    assert "not PIC" in payload["warning"] and "export-only" in payload["warning"]


def test_embedded_json_round_trips_and_javascript_parses(html, payload, tmp_path: Path) -> None:
    match = re.search(
        r'<script id="wall-loss-v4-data" type="application/json">(.*?)</script>', html, re.DOTALL
    )
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(match.group(1), parse_constant=reject_constant) == payload
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable for JavaScript syntax checking")
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    script = tmp_path / "wall-loss-v4-results.js"
    script.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_every_hand_visible_number_in_the_template_is_data_driven() -> None:
    """The template must carry no literal result numbers; all come from DATA."""

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    body = template.split("<body>", 1)[1].split("<script>", 1)[0]
    body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL)
    literal_numbers = re.findall(r"(?<![\w#.\-])\d+(?:\.\d+)?(?![\w%-])", re.sub(r"<[^>]+>", " ", body))
    # keyboard hints (0, 1, 9), "Wilson 95%", the 0 %/100 % colour-scale legend endpoints
    allowed = {"0", "1", "9", "95", "100"}
    assert set(literal_numbers) <= allowed, sorted(set(literal_numbers) - allowed)
