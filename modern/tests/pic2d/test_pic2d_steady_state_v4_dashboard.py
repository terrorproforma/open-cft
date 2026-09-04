"""Tests for the PIC-2D steady-state v4 (33 um refinement) dashboard generator (skipped until the v4 record exists).

Since model v2.0.6 the dashboard carries both ledger readings: the recorded assessment and the post-hoc corrected-ledger
re-read (``assessment-corrected-ledger.json``), each case's ``ledger-corrected.json`` sidecar bound to its series, and the
corrected windowed residual recomputed from the series.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from copy import deepcopy
from hashlib import sha256
from math import floor, isfinite, log10
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cft_revival.pic2d.artifacts import platform_fingerprint, read_npz, write_canonical_json

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_pic2d_cft_steady_state_v4.py"
CHECKED_HTML = MODERN / "visualization" / "pic2d-cft-steady-state-v4.html"
ANCHOR_PLATFORM = MODERN / "visualization" / "pic2d-cft-steady-state-v4.anchor-platform.json"
EXPERIMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v4"
RESULTS = EXPERIMENT / "results"
REFERENCE = MODERN / "experiments" / "pic2d_cft_steady_state_v2"
RELATIVE_FLOOR = 1e-9
MIN_RECORDED_DIGITS = 4


def _load_generator():
    spec = importlib.util.spec_from_file_location("pic2d_steady_state_v4_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()
pytestmark = pytest.mark.skipif(not (RESULTS / "assessment.json").is_file() or not (RESULTS / "assessment-corrected-ledger.json").is_file()
                                or not (REFERENCE / "results-w-0.7" / "summary.json").is_file(),
                                reason="steady-state v4 record (with its corrected-ledger re-read) or the v2 convergence pair is not materialised")


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_payload_is_hash_bound_and_carries_the_recorded_verdict(payload) -> None:
    assert payload["schema"] == GENERATOR.SCHEMA and payload["status"] == GENERATOR.STATUS
    assessment = json.loads((RESULTS / "assessment.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == assessment["verdict"] == "resolution_limited"
    assert "resolution limited" in payload["claim_statement"] and "not validated" in payload["claim_statement"].lower()
    refined, reference = payload["cases"][0], payload["cases"][1]
    assert refined["role"] == "refined" and refined["id"] == summary["case"]["id"] and refined["grid"] == {"radial_cells": 90, "axial_cells": 720, "dr_m": pytest.approx(3e-3 / 90), "dz_m": pytest.approx(24e-3 / 720)}
    assert refined["protocol_sha256"] == summary["protocol_sha256"] == payload["protocol"]["file_sha256"] == payload["execution"]["lock"]["protocol_sha256"]
    assert refined["maps_npz_sha256"] == summary["artifacts"]["maps_npz_sha256"] and refined["config_sha256"] == payload["execution"]["lock"]["config_sha256"]
    assert refined["git_head"] == payload["execution"]["lock"]["commit"] == payload["protocol"]["preregistration_commit"]
    assert refined["stop_reason"] == "plateau_reached_after_min_transit_times" and refined["ion_transit_times"] == pytest.approx(7.28e-6 / 2.4e-6)
    assert refined["plateau"]["peak_debye_soft_ok"] is True and refined["plateau"]["triad_soft_ok"] is True
    assert reference["role"] == "reference" and reference["grid"]["radial_cells"] == 60 and reference["results_dir"] == "results"
    assert [c["role"] for c in payload["cases"]] == ["refined", "reference", "band", "band"]
    # the pinned reference quantities are the v2 base artifacts (re-derived here and in the assess stage)
    for key, value in assessment["reference"].items():
        if key in reference["quantities"]:
            assert reference["quantities"][key] == pytest.approx(value, rel=1e-12), key
    assert all(entry["agree"] for entry in payload["assessment"]["reference_consistency"].values())
    GENERATOR.validate_payload(payload)


def test_comparison_table_reproduces_the_assessment_and_names_what_moved(payload) -> None:
    comparison = payload["comparison"]
    rows = {row["key"]: row for row in comparison["rows"]}
    assert list(rows) == [q[0] for q in GENERATOR.QUANTITIES]
    assessment = json.loads((RESULTS / "assessment.json").read_text(encoding="utf-8"))
    for key, row in rows.items():
        entry = assessment["c_convergence"]["quantities"][key]
        assert row["reference"] == entry["reference"] and row["refined"] == entry["value"]
        assert row["relative_difference"] == pytest.approx(entry["relative_difference"], rel=1e-12) and row["within"] is entry["within"]
        assert row["tolerance"] == entry["tolerance"] and len(row["bands"]) == 2
    assert comparison["all_within"] is False and set(comparison["failed"]) == {"discharge_current_a", "peak_n_e_window_per_m3", "t_e_peak_window_ev"}
    assert rows["discharge_current_a"]["relative_difference"] == pytest.approx(0.1035, abs=5e-4)
    assert rows["peak_n_e_window_per_m3"]["relative_difference"] == pytest.approx(-0.2142, abs=5e-4)
    assert rows["t_e_peak_window_ev"]["relative_difference"] == pytest.approx(-0.2450, abs=5e-4)
    for key in ("exit_ion_beam_a", "ionization_rate_per_s", "gross_utilisation", "neutral_density_per_m3"):
        assert rows[key]["within"] and abs(rows[key]["relative_difference"]) < 0.1
    # the 50 um particle-resolution bands (seed-b, W x0.7) are embedded beside every quantity
    seed_b = {row["key"]: row["bands"][0]["relative_difference"] for row in comparison["rows"]}
    w07 = {row["key"]: row["bands"][1]["relative_difference"] for row in comparison["rows"]}
    assert seed_b["discharge_current_a"] == pytest.approx(-0.0008, abs=5e-4) and seed_b["peak_n_e_window_per_m3"] == pytest.approx(-0.082, abs=2e-3)
    assert w07["discharge_current_a"] == pytest.approx(0.057, abs=1e-3) and w07["peak_n_e_window_per_m3"] == pytest.approx(-0.119, abs=2e-3)
    # Debye and residual rows: soft margin held at 33 um, the base sits on the CIC threshold
    debye = comparison["debye"]
    assert debye["soft_ok"] is True and debye["refined_window_gate_last"] == pytest.approx(2.154, abs=2e-3) and debye["reference_cells_per_debye_at_peak"] == pytest.approx(3.1665, abs=1e-3)
    assert debye["refined_cells_per_debye_at_peak_maps"] == pytest.approx(debye["refined_window_gate_last"], rel=1e-6)   # maps == window statistic at the stop
    residuals = comparison["residuals"]
    assert residuals["refined_windowed"] == pytest.approx(-0.07667, abs=1e-4) and residuals["refined_windowed"] < residuals["acceptance_bound"] == 0.02
    # the base's trailing 400 000-step window ends slightly on the heating side (+0.37 %: the "+0.4 %" of the attempt-8 diagnosis; the
    # protocol's pinned -0.19 % is the last window-aligned reading), seed-b and W x0.7 on the cooling side
    assert residuals["reference_windowed_recomputed"] == pytest.approx(0.0037, abs=1e-3)
    assert all(-0.05 < band["windowed_recomputed"] < 0.0 for band in residuals["bands"])


def test_windowed_residual_recomputation_matches_the_runner(payload) -> None:
    series = read_npz(RESULTS / "series.npz")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    recomputed = GENERATOR.windowed_residual(series)
    assert recomputed[:1999].tolist() == pytest.approx([float("nan")] * 1999, nan_ok=True) and np.isnan(recomputed[:1999]).all()
    assert recomputed[-1] == pytest.approx(summary["grid_heating_triad"]["windowed_energy_residual_over_electrode_work"], rel=1e-9)
    assert payload["cases"][0]["windowed_residual_recomputed"] == pytest.approx(recomputed[-1], rel=1e-12)
    for case in payload["cases"][1:]:
        assert case["windowed_residual_recomputed"] is not None and -0.15 < case["windowed_residual_recomputed"] < 0.02
    # decimated series keep the last record and every key the same length
    case = payload["cases"][0]
    assert case["series"]["time_s"][-1] == pytest.approx(7.28e-6, rel=1e-6) and len(case["series"]["time_s"]) <= GENERATOR.MAX_SERIES_POINTS + 1
    assert {len(v) for v in case["series"].values()} == {len(case["series"]["time_s"])}
    assert "peak_node_window_cells_per_debye" in case["series"] and "peak_node_window_cells_per_debye" not in payload["cases"][1]["series"]


def test_corrected_ledger_block_carries_both_readings_and_is_hash_bound(payload) -> None:
    block = payload["corrected_ledger"]
    reread_path = RESULTS / "assessment-corrected-ledger.json"
    reread = json.loads(reread_path.read_text(encoding="utf-8"))
    rr = block["reread"]
    assert rr["sha256"] == sha256(reread_path.read_bytes()).hexdigest() and rr["file"] == "assessment-corrected-ledger.json"
    assert rr["verdict_recorded"] == payload["verdict"] == "resolution_limited" and rr["verdict_on_corrected_ledger"] == "refinement_heating"
    assert rr["verdict_statement"] == reread["verdict_statement"] and "FAILED on the corrected ledger" in rr["verdict_statement"] and "25 µm (v5) pending" in rr["verdict_statement"]
    b = rr["b_residual_power"]
    assert b["recorded"]["passed"] is True and b["recorded"]["windowed_residual_over_electrode_work"] == pytest.approx(-0.07667, abs=1e-4)
    assert b["corrected"]["passed"] is False and b["corrected"]["windowed_residual_over_electrode_work"] == pytest.approx(0.02459, abs=1e-4)
    assert b["passed"] is False and b["bound"] == 0.02 and b["status_change"] == "PASS (recorded) -> FAIL (corrected)"
    assert block["thresholds"] == {"acceptance_b": 0.02, "hard_gate": 0.05, "kept": block["thresholds"]["kept"]} and "not loosened" in block["thresholds"]["kept"]
    assert all(rr["binding_checks"].values()) and len(rr["binding_checks"]) == 7
    # every embedded case carries its sidecar reading, bound to the sidecar bytes on disk, and the recomputed corrected series reproduces it
    rows = {row["results_dir"]: row for row in block["cases"]}
    assert [row["role"] for row in block["cases"]] == ["refined", "reference", "band", "band"]
    expected = {("refined", "results"): (-0.0767, 0.0246), ("reference", "results"): (0.0037, 0.1301), ("band", "results-seed-b"): (-0.014, 0.111), ("band", "results-w-0.7"): (-0.041, 0.072)}
    for case, row in zip(payload["cases"], block["cases"], strict=True):
        directory = RESULTS if case["role"] == "refined" else REFERENCE / case["results_dir"]
        sidecar_path = directory / "ledger-corrected.json"
        assert row["sidecar_sha256"] == sha256(sidecar_path.read_bytes()).hexdigest() == case["ledger_corrected"]["sidecar_sha256"]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["inputs"]["series"]["sha256"] == case["series_npz_sha256"]
        recorded, corrected = expected[(case["role"], case["results_dir"])]
        assert row["recorded_windowed"] == pytest.approx(recorded, abs=1.5e-3) and row["corrected_windowed"] == pytest.approx(corrected, abs=1.5e-3), case["label"]
        assert case["windowed_residual_corrected_recomputed"] == pytest.approx(row["corrected_windowed"], rel=1e-9)
        series = case["series"]["windowed_residual_corrected_over_electrode_work"]
        assert len(series) == len(case["series"]["time_s"]) and series[-1] == pytest.approx(row["corrected_windowed"], rel=1e-4)
        assert row["acceptance_b_recorded_passes"] is True and row["acceptance_b_corrected_passes"] is False    # every case flips PASS -> FAIL
    # the 50 um plateaus were heating at +7..+13 %: the corrected 5 % gate fires on all three, never on the 33 um run
    assert rows["results-seed-b"]["corrected_gate_0p05_first_checkpoint_time_s"] == pytest.approx(2.76e-6, abs=2e-8)
    assert rows["results-w-0.7"]["corrected_gate_0p05_first_checkpoint_time_s"] == pytest.approx(4.50e-6, abs=2e-8)
    assert block["cases"][1]["role"] == "reference" and block["cases"][1]["corrected_gate_0p05_first_checkpoint_time_s"] == pytest.approx(2.70e-6, abs=2e-8)
    assert block["cases"][0]["corrected_gate_0p05_first_checkpoint_time_s"] is None and block["cases"][0]["recorded_gate_0p05_first_checkpoint_time_s"] is None
    assert block["cases"][0]["corrected_first_checkpoint_at_or_above_0p02_time_s"] == pytest.approx(4.816e-6)
    residuals = payload["comparison"]["residuals"]
    assert residuals["refined_windowed_corrected"] == pytest.approx(0.02459, abs=1e-4) and residuals["reference_windowed_corrected"] == pytest.approx(0.1301, abs=1e-3)
    assert [band["windowed_corrected"] for band in residuals["bands"]] == pytest.approx([0.111, 0.072], abs=1.5e-3)
    statement = payload["claim_statement"]
    assert "Energy-ledger correction (model v2.0.6, post hoc)" in statement and "+2.46 %" in statement and "recorded PASS → corrected FAIL" in statement and "+13.0 %" in statement
    assert "recorded verdict resolution limited" in statement and "neither grid may be called converged" in statement
    html = GENERATOR.render_html(payload)
    for fragment in ('id="reread"', 'id="ledger"', "recorded verdict (assessment.json)", "corrected ledger, v2.0.6 post hoc", "FAILED on the corrected ledger",
                     "PASS (recorded) -&gt; FAIL (corrected)", "CORRECTED ledger (model v2.0.6 sidecars; same window)", "(corrected, v2.0.6)", "renderLedger()"):
        assert fragment in html or fragment.replace("-&gt;", "->") in html, fragment


def _embedded_payload(html: str) -> dict[str, Any]:
    match = re.search(r'<script id="pic2d-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _significant_digits(value: float) -> int:
    if value == 0.0:
        return 0
    mantissa = repr(abs(value)).split("e")[0].replace(".", "").lstrip("0").rstrip("0")
    return len(mantissa) or 1


def _leaves_agree(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        if isinstance(a, int) and isinstance(b, int) or not (isfinite(a) and isfinite(b)):
            return False
        magnitude = max(abs(a), abs(b))
        digits = max(MIN_RECORDED_DIGITS, _significant_digits(float(a)), _significant_digits(float(b)))
        unit = 10.0 ** (floor(log10(magnitude)) - digits + 1)
        return abs(a - b) <= max(RELATIVE_FLOOR * magnitude, unit) * (1 + 1e-9)
    return a == b


def payload_differences(expected: Any, actual: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any, Any]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [(path, sorted(expected), sorted(actual))]
        return [d for key in expected for d in payload_differences(expected[key], actual[key], (*path, key))]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [(path, len(expected), len(actual))]
        return [d for index, (e, a) in enumerate(zip(expected, actual, strict=True)) for d in payload_differences(e, a, (*path, index))]
    return [] if _leaves_agree(expected, actual) else [(path, expected, actual)]


def test_generation_is_byte_deterministic_and_checked_html_is_current(payload, tmp_path: Path) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "pic2d-cft-steady-state-v4.html"
    GENERATOR.generate(output)
    assert output.read_text(encoding="utf-8") == first
    sidecar = json.loads(GENERATOR.anchor_platform_path(output).read_text(encoding="utf-8"))
    assert sidecar["html_sha256"] == sha256(first.encode("utf-8")).hexdigest() and sidecar["platform"] == platform_fingerprint()
    checked = CHECKED_HTML.read_text(encoding="utf-8")
    anchor = json.loads(ANCHOR_PLATFORM.read_text(encoding="utf-8"))
    assert anchor["html_sha256"] == sha256(checked.encode("utf-8")).hexdigest(), "anchor-platform sidecar does not describe the checked-in HTML"
    if anchor["platform"]["fingerprint_sha256"] == platform_fingerprint()["fingerprint_sha256"]:
        assert checked == first, "on the anchor platform the checked-in HTML must be byte-current"
    else:
        differences = payload_differences(_embedded_payload(checked), _embedded_payload(first))
        assert not differences, f"{len(differences)} payload leaves outside the declared cross-platform tolerance: {differences[:5]}"
        strip = lambda text: re.sub(r'<script id="pic2d-data".*?</script>', "", text, flags=re.DOTALL)
        assert strip(checked) == strip(first)


def test_html_is_self_contained_offline_with_controls(payload) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    assert '<script id="pic2d-data" type="application/json">' in html
    for forbidden in ("fetch(", "xmlhttprequest", "websocket", "cdn"):
        assert forbidden not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", html, re.IGNORECASE)
    assert not re.search(r"\bhttps?://", html, re.IGNORECASE)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", html)
    for fragment in ('id="verdict"', 'id="claim"', 'id="acceptance"', 'id="comparison"', 'id="legend"', 'id="p_id"', 'id="p_res"', 'id="p_deb"',
                     'id="records"', 'id="identity"', 'for="tscale"', 'id="theme"', 'role="img"', "new ResizeObserver(schedule)", "window.devicePixelRatio",
                     "Claim boundary", "RESOLUTION-LIMITED", "hard π", "soft 2.5", "acceptance (b) +2 %"):
        assert fragment in html, fragment
    assert "<svg" not in lowered

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    assert json.loads(re.search(r'<script id="pic2d-data" type="application/json">(.*?)</script>', html, re.DOTALL).group(1), parse_constant=reject_constant) == payload


def test_javascript_is_valid_when_node_is_available(payload, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available for JavaScript syntax checking")
    html = GENERATOR.render_html(payload)
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 2
    script = tmp_path / "pic2d-v4.js"
    script.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_tampered_payload_is_rejected(payload) -> None:
    for mutate in (
        lambda p: p.__setitem__("status", "accepted"),
        lambda p: p.__setitem__("verdict", "converged"),
        lambda p: p["assessment"].__setitem__("verdict", "converged"),
        lambda p: p["comparison"]["rows"][0].__setitem__("within", True),
        lambda p: p["comparison"].__setitem__("all_within", True),
        lambda p: p["comparison"].__setitem__("failed", []),
        lambda p: p["cases"][0].__setitem__("summary_sha256", "abc"),
        lambda p: p["cases"][0].__setitem__("protocol_sha256", "0" * 64),
        lambda p: p["cases"][1].__setitem__("protocol_sha256", "0" * 64),
        lambda p: p["cases"][0].__setitem__("stop_reason", "converged"),
        lambda p: p["cases"][0]["series"]["electrons"].pop(),
        lambda p: p["execution"]["lock"].__setitem__("commit", "0" * 40),
        lambda p: p["execution"]["run_state"].__setitem__("finished", False),
        lambda p: p.__setitem__("claim_statement", "validated steady state"),
        lambda p: p.pop("comparison"),
        # the corrected-ledger block: both readings must stay consistent with the values, the bound may not move, the binding must hold
        lambda p: p.pop("corrected_ledger"),
        lambda p: p["corrected_ledger"]["reread"].__setitem__("verdict_on_corrected_ledger", "resolution_limited"),
        lambda p: p["corrected_ledger"]["reread"]["b_residual_power"]["corrected"].__setitem__("passed", True),
        lambda p: p["corrected_ledger"]["reread"]["b_residual_power"].__setitem__("passed", True),
        lambda p: p["corrected_ledger"]["thresholds"].__setitem__("acceptance_b", 0.03),
        lambda p: p["corrected_ledger"]["reread"]["b_residual_power"].__setitem__("bound", 0.03),
        lambda p: p["corrected_ledger"]["reread"].__setitem__("verdict_statement", "plateau reached; residual precondition (b) holds on the corrected ledger"),
        lambda p: p["corrected_ledger"]["reread"]["binding_checks"].__setitem__("sidecar_series_sha256_equals_summary_artifact", False),
        lambda p: p["corrected_ledger"]["cases"][0].__setitem__("corrected_windowed", 0.01),
        lambda p: p["corrected_ledger"]["cases"][1].__setitem__("acceptance_b_corrected_passes", True),
        lambda p: p["corrected_ledger"]["cases"][2].__setitem__("sidecar_sha256", "0" * 64),
        lambda p: p["cases"][0].__setitem__("windowed_residual_corrected_recomputed", 0.0),
        lambda p: p["cases"][1]["series"].pop("windowed_residual_corrected_over_electrode_work"),
        lambda p: p.__setitem__("claim_statement", p["claim_statement"].replace("Energy-ledger correction", "Ledger note")),
    ):
        changed = deepcopy(payload)
        mutate(changed)
        with pytest.raises(ValueError):
            GENERATOR.validate_payload(changed)


def test_protocol_drift_tampered_artifacts_and_inconsistent_assessment_are_rejected(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    shutil.copytree(RESULTS, experiment / "results", ignore=shutil.ignore_patterns("checkpoint", "frames", "video", "*.jsonl", "*.log", "*.err", "*.pid", "checkpoint-final.npz*"))
    shutil.copy(EXPERIMENT / "protocol.json", experiment / "protocol.json")
    protocol = experiment / "protocol.json"
    GENERATOR.build_payload(experiment / "results", protocol)
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol drift"):
        GENERATOR.build_payload(experiment / "results", protocol)
    shutil.copy(EXPERIMENT / "protocol.json", protocol)
    summary = experiment / "results" / "summary.json"
    original = summary.read_bytes()
    summary.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        GENERATOR.build_payload(experiment / "results", protocol)
    summary.write_bytes(original)     # bytes, not text: write_text would re-encode the newline on Windows and the sidecar would still refuse
    # an assessment whose verdict contradicts its own (a)-(c) outcomes is refused (the sidecar is rewritten so only the logic check fires)
    assessment = experiment / "results" / "assessment.json"
    original_assessment = assessment.read_bytes()
    original_assessment_sidecar = assessment.with_name(assessment.name + ".sha256.json").read_bytes()
    record = json.loads(assessment.read_text(encoding="utf-8"))
    record["verdict"] = "converged"
    write_canonical_json(assessment, record)
    with pytest.raises(ValueError, match="verdict"):
        GENERATOR.build_payload(experiment / "results", protocol)
    assessment.write_bytes(original_assessment)
    assessment.with_name(assessment.name + ".sha256.json").write_bytes(original_assessment_sidecar)
    GENERATOR.build_payload(experiment / "results", protocol)
    # the corrected-ledger sidecar must describe the embedded series; the re-read must bind the sidecar; both are required
    sidecar = experiment / "results" / "ledger-corrected.json"
    original_sidecar = sidecar.read_bytes()
    original_sidecar_hash = sidecar.with_name(sidecar.name + ".sha256.json").read_bytes()
    tampered = json.loads(sidecar.read_text(encoding="utf-8"))
    tampered["inputs"]["series"]["sha256"] = "0" * 64
    write_canonical_json(sidecar, tampered)
    with pytest.raises(ValueError, match="describes another series"):
        GENERATOR.build_payload(experiment / "results", protocol)
    tampered = json.loads(original_sidecar.decode("utf-8"))
    tampered["end_state_window"]["corrected_ratio"] = 0.01
    write_canonical_json(sidecar, tampered)
    with pytest.raises(ValueError, match="corrected windowed residual recomputed here"):
        GENERATOR.build_payload(experiment / "results", protocol)
    sidecar.write_bytes(original_sidecar)
    sidecar.with_name(sidecar.name + ".sha256.json").write_bytes(original_sidecar_hash)
    reread = experiment / "results" / "assessment-corrected-ledger.json"
    original_reread = reread.read_bytes()
    original_reread_hash = reread.with_name(reread.name + ".sha256.json").read_bytes()
    tampered = json.loads(reread.read_text(encoding="utf-8"))
    tampered["inputs"]["ledger_corrected"]["sha256"] = "0" * 64
    write_canonical_json(reread, tampered)
    with pytest.raises(ValueError, match="does not bind the embedded ledger-corrected.json"):
        GENERATOR.build_payload(experiment / "results", protocol)
    reread.write_bytes(original_reread)
    reread.with_name(reread.name + ".sha256.json").write_bytes(original_reread_hash)
    GENERATOR.build_payload(experiment / "results", protocol)
    reread.unlink()
    with pytest.raises(ValueError, match="assessment-corrected-ledger.json is missing"):
        GENERATOR.build_payload(experiment / "results", protocol)
    reread.write_bytes(original_reread)
    sidecar.unlink()
    with pytest.raises(ValueError, match="ledger-corrected.json is missing"):
        GENERATOR.build_payload(experiment / "results", protocol)
