"""Tests for the PIC-2D design mini-sweep v1 dashboard generator (skipped until the sweep's terminal records exist).

The dashboard embeds the four primary designs ordered by rho, each hash-bound to its record (summary / maps / series / protocol /
execution lock), the corrected-ledger sidecar recomputed from the series, the per-design assessment (plateau designs) or the
gate-stop diagnosis (interim designs), the closure targets (recorded data only), and the ss-v4 reference-grid verdict in both readings.
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

import pytest

from cft_revival.pic2d.artifacts import platform_fingerprint, write_canonical_json

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_pic2d_design_mini_sweep_v1.py"
CHECKED_HTML = MODERN / "visualization" / "pic2d-design-mini-sweep-v1.html"
ANCHOR_PLATFORM = MODERN / "visualization" / "pic2d-design-mini-sweep-v1.anchor-platform.json"
EXPERIMENT = MODERN / "experiments" / "pic2d_design_mini_sweep_v1"
RESULTS = EXPERIMENT / "results"
V4_RESULTS = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results"
RELATIVE_FLOOR = 1e-9
MIN_RECORDED_DIGITS = 4
TERMINAL = ("l1a-gs-v2-047-e3196a8aa5-channel-33um", "divergent-exit-stack-channel-33um", "l1a-gs-v3-009-d0c686b4aa-channel-33um")


def _load_generator():
    spec = importlib.util.spec_from_file_location("pic2d_design_mini_sweep_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()
pytestmark = pytest.mark.skipif(not all((RESULTS / d / "assessment.json").is_file() for d in TERMINAL) or not (V4_RESULTS / "assessment-corrected-ledger.json").is_file()
                                or not CHECKED_HTML.is_file(),
                                reason="the sweep's terminal records (047 / reference / 009 with assessments) or the v4 corrected-ledger re-read are not materialised")


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_payload_is_hash_bound_ordered_by_rho_and_carries_the_recorded_verdicts(payload) -> None:
    assert payload["schema"] == GENERATOR.SCHEMA and payload["option"] == "channel-33um"
    designs = payload["designs"]
    assert [d["label"] for d in designs] == ["047", "ref", "009", "056"]
    assert [round(d["rho"], 2) for d in designs] == [0.38, 0.6, 0.92, 2.36]
    for d in designs:
        results = RESULTS / d["results_dir"]
        summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
        assert d["summary_sha256"] == sha256((results / "summary.json").read_bytes()).hexdigest()
        assert d["protocol_sha256"] == summary["protocol_sha256"] == d["lock"]["protocol_sha256"]
        assert d["config_sha256"] == summary["provenance"]["config_sha256"] == d["lock"]["config_sha256"]
        assert d["maps_npz_sha256"] == summary["artifacts"]["maps_npz_sha256"] and d["series_npz_sha256"] == summary["artifacts"]["series_npz_sha256"]
        assert d["git_head"] == d["lock"]["commit"]
        assert d["ledger_corrected"]["sidecar_sha256"] == sha256((results / "ledger-corrected.json").read_bytes()).hexdigest()
    verdicts = {d["label"]: (d["assessment"] or {}).get("verdict") for d in designs}
    assert verdicts == {"047": "closure_quotable", "ref": "plateau_with_heating", "009": "closure_quotable", "056": None}
    statuses = {d["label"]: d["status"] for d in designs}
    assert statuses == {"047": "plateau", "ref": "plateau", "009": "plateau", "056": "gate_stopped_interim"} or statuses["056"] == "plateau"
    assert payload["protocol"]["preregistration_commit"].startswith("291a9227")
    assert payload["reference_grid_verdict"]["verdict_recorded"] == "resolution_limited" and payload["reference_grid_verdict"]["verdict_on_corrected_ledger"] == "refinement_heating"
    assert payload["reference_grid_verdict"]["b_corrected"] == pytest.approx(0.02459, abs=1e-4)
    assert "uncertified" in payload["claim_statement"] and "recorded data only" in payload["claim_statement"] and "not validated" in payload["claim_statement"].lower()
    GENERATOR.validate_payload(payload)


def test_trend_table_carries_the_plateau_values_versus_rho(payload) -> None:
    rows = {row["key"]: row for row in payload["trend"]["rows"]}
    assert list(rows) == [q[0] for q in GENERATOR.QUANTITIES]
    by_label = lambda key: {v["label"]: v["value"] for v in rows[key]["values"]}
    i_d = by_label("discharge_current_a")
    assert i_d["047"] == pytest.approx(1.925e-3, rel=2e-3) and i_d["ref"] == pytest.approx(3.805e-3, rel=2e-3) and i_d["009"] == pytest.approx(4.408e-3, rel=2e-3)
    s = by_label("ionization_rate_per_s")
    assert s["047"] == pytest.approx(1.456e16, rel=2e-3) and s["ref"] == pytest.approx(3.602e16, rel=2e-3) and s["009"] == pytest.approx(3.357e16, rel=2e-3)
    util = by_label("gross_utilisation")
    assert util["047"] == pytest.approx(0.316, abs=2e-3) and util["ref"] == pytest.approx(0.421, abs=2e-3) and util["009"] == pytest.approx(0.491, abs=2e-3)
    n_g = by_label("neutral_density_per_m3")
    assert n_g["047"] == pytest.approx(3.76e19, rel=3e-3) and n_g["ref"] == pytest.approx(3.18e19, rel=3e-3) and n_g["009"] == pytest.approx(2.80e19, rel=3e-3)
    peak = by_label("peak_n_e_window_per_m3")
    assert peak["ref"] == pytest.approx(1.277e18, rel=2e-3) and peak["047"] < peak["ref"] and peak["009"] < peak["ref"]
    centroid = by_label("ionisation_centroid_m")
    assert all(0.5 < c < 0.8 for c in centroid.values())                         # the flames sit in the downstream half of every channel
    assert centroid["009"] > centroid["ref"] > centroid["047"] * 0.9              # the mid-rho design ionises furthest downstream
    ref_row = next(v for v in rows["discharge_current_a"]["values"] if v["label"] == "ref")
    assert ref_row["relative_to_reference"] == 0.0 and ref_row["above_reference_spread"] is None
    v047 = next(v for v in rows["discharge_current_a"]["values"] if v["label"] == "047")
    assert v047["relative_to_reference"] == pytest.approx(-0.494, abs=2e-3) and v047["above_reference_spread"] is True
    assert rows["discharge_current_a"]["reference_spread"] == 0.057 and rows["ionisation_centroid_m"]["reference_spread"] is None
    interim = [v for v in rows["discharge_current_a"]["values"] if v["status"] != "plateau"]
    assert payload["trend"]["interim_designs"] == [v["id"] for v in interim] and payload["sweep"]["provisional"] is bool(interim)


def test_ledger_readings_are_the_sidecars_and_the_reference_fails_b(payload) -> None:
    rows = {d["label"]: d["ledger_corrected"] for d in payload["designs"]}
    assert rows["047"]["recorded_windowed"] == pytest.approx(-0.0711, abs=1e-3) and rows["047"]["corrected_windowed"] == pytest.approx(0.0091, abs=5e-4)
    assert rows["009"]["recorded_windowed"] == pytest.approx(-0.0762, abs=1e-3) and rows["009"]["corrected_windowed"] == pytest.approx(0.0031, abs=5e-4)
    assert rows["ref"]["recorded_windowed"] == pytest.approx(-0.0766, abs=1e-3) and rows["ref"]["corrected_windowed"] == pytest.approx(0.0247, abs=5e-4)
    assert rows["ref"]["acceptance_b_recorded_passes"] is True and rows["ref"]["acceptance_b_corrected_passes"] is False
    assert rows["ref"]["corrected_first_checkpoint_at_or_above_0p02_time_s"] == pytest.approx(4.928e-6, abs=2e-8) and rows["ref"]["corrected_gate_0p05_first_checkpoint_time_s"] is None
    assert all(r["acceptance_b_corrected_passes"] for k, r in rows.items() if k != "ref")
    for d in payload["designs"]:
        assert d["windowed_residual_corrected_recomputed"] == pytest.approx(d["ledger_corrected"]["corrected_windowed"], rel=1e-9)
        series = d["series"]["windowed_residual_corrected_over_electrode_work"]
        assert len(series) == len(d["series"]["time_s"]) and series[-1] == pytest.approx(d["ledger_corrected"]["corrected_windowed"], rel=1e-4)
    ref = next(d for d in payload["designs"] if d["role"] == "reference")
    assert ref["assessment"]["b_corrected"] == pytest.approx(ref["ledger_corrected"]["corrected_windowed"], rel=1e-12) and ref["assessment"]["b_passed"] is False
    assert "+2.47 %" in payload["sweep"]["statement"] or "+2.46 %" in payload["sweep"]["statement"]


def test_closure_targets_are_recorded_data_only(payload) -> None:
    for d in payload["designs"]:
        if d["status"] != "plateau":
            assert d["closure_targets"] is None and d["gate_stop"]["verdict"] == "SHOT_NOISE_ARTEFACT"
            continue
        t = d["closure_targets"]
        assert "RECORDED DATA ONLY" in t["note"] and len(t["file_sha256"]) == 64
        chain = t["kornfeld_chain_exit_to_anode"]
        assert all(0.0 <= k["p_transit"] <= 1.0 for k in chain) and len(chain) >= 3
        # the catalogue cells do not tile the whole domain (the reference's first cell starts at z = 1 mm; the cone / plume lies past the last cell)
        assert 0.5 < sum(c["ionisation_share"] for c in t["cells_anode_to_exit"]) <= 1.0 + 1e-9
    ref = next(d for d in payload["designs"] if d["role"] == "reference")
    assert [round(k["p_transit"], 3) for k in ref["closure_targets"]["kornfeld_chain_exit_to_anode"]] == pytest.approx([0.509, 0.368, 0.148], abs=2e-3)


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
    output = tmp_path / "pic2d-design-mini-sweep-v1.html"
    GENERATOR.generate(output)
    assert output.read_text(encoding="utf-8") == first
    sidecar = json.loads(GENERATOR.anchor_platform_path(output).read_text(encoding="utf-8"))
    assert sidecar["html_sha256"] == sha256(first.encode("utf-8")).hexdigest() and sidecar["platform"] == platform_fingerprint()
    checked = CHECKED_HTML.read_text(encoding="utf-8")
    anchor = json.loads(ANCHOR_PLATFORM.read_text(encoding="utf-8"))
    assert anchor["html_sha256"] == sha256(checked.encode("utf-8")).hexdigest(), "anchor-platform sidecar does not describe the checked-in HTML"
    checked_payload = _embedded_payload(checked)
    if checked_payload["sweep"]["interim"] != payload["sweep"]["interim"]:
        pytest.skip("the checked-in HTML was generated before a design's terminal record landed; regenerate it")
    if anchor["platform"]["fingerprint_sha256"] == platform_fingerprint()["fingerprint_sha256"]:
        assert checked == first, "on the anchor platform the checked-in HTML must be byte-current"
    else:
        differences = payload_differences(checked_payload, _embedded_payload(first))
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
    for fragment in ('id="verdict"', 'id="provisional"', 'id="claim"', 'id="trend"', 'id="acceptance"', 'id="ledger"', 'id="targets"', 'id="legend"', 'id="p_id"',
                     'id="p_res"', 'id="p_ion"', 'id="records"', 'id="identity"', 'for="tscale"', 'id="theme"', 'role="img"', "new ResizeObserver(schedule)",
                     "Claim boundary", "RECORDED DATA ONLY", "hard π", "soft 2.5", "acceptance (b) +2 %", "Design-vs-ρ trend table"):
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
    script = tmp_path / "pic2d-sweep.js"
    script.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_tampered_payload_is_rejected(payload) -> None:
    ref_index = [d["role"] for d in payload["designs"]].index("reference")
    for mutate in (
        lambda p: p.__setitem__("option", "plume-24mm"),
        lambda p: p.__setitem__("claim_statement", "validated design ranking"),
        lambda p: p.__setitem__("claim_statement", p["claim_statement"].replace("recorded data only", "calibration values")),
        lambda p: p["designs"].reverse(),
        lambda p: p["designs"][0].__setitem__("summary_sha256", "abc"),
        lambda p: p["designs"][0]["lock"].__setitem__("commit", "0" * 40),
        lambda p: p["designs"][0].__setitem__("finished", False),
        lambda p: p["designs"][0].__setitem__("stop_reason", "converged"),
        lambda p: p["designs"][0]["series"]["electrons"].pop(),
        lambda p: p["designs"][ref_index]["assessment"].__setitem__("verdict", "closure_quotable"),
        lambda p: p["designs"][ref_index]["assessment"].__setitem__("b_passed", True),
        lambda p: p["designs"][ref_index]["ledger_corrected"].__setitem__("acceptance_b_corrected_passes", True),
        lambda p: p["designs"][ref_index]["ledger_corrected"].__setitem__("corrected_windowed", 0.001),
        lambda p: p["designs"][ref_index].__setitem__("windowed_residual_corrected_recomputed", 0.0),
        lambda p: p["designs"][ref_index].__setitem__("closure_targets", None),
        lambda p: p["designs"][ref_index]["series"].pop("windowed_residual_corrected_over_electrode_work"),
        lambda p: p["trend"]["rows"][0]["values"][0].__setitem__("value", 1.0),
        lambda p: p["trend"]["rows"][0]["values"][0].__setitem__("above_reference_spread", False),
        lambda p: p["trend"]["rows"].pop(),
        lambda p: p["sweep"].__setitem__("provisional", not p["sweep"]["provisional"]),
        lambda p: p["sweep"].__setitem__("terminal_records", 9),
        lambda p: p["sweep"]["per_design_verdicts"].__setitem__(p["designs"][ref_index]["id"], "closure_quotable"),
        lambda p: p["reference_grid_verdict"].__setitem__("verdict_on_corrected_ledger", "converged"),
        lambda p: p["reference_grid_verdict"].__setitem__("assessment_sha256", "0" * 10),
        lambda p: p.pop("trend"),
    ):
        changed = deepcopy(payload)
        mutate(changed)
        with pytest.raises(ValueError):
            GENERATOR.validate_payload(changed)


def test_protocol_drift_and_tampered_artifacts_are_rejected(tmp_path: Path) -> None:
    copy = tmp_path / "results"
    shutil.copytree(RESULTS, copy, ignore=shutil.ignore_patterns("checkpoint", "frames", "video", "*.jsonl", "*.log", "*.err", "*.pid", "checkpoint-final.npz*"))
    GENERATOR.build_payload(copy)
    protocol = copy / TERMINAL[1] / "protocol.json"
    original = protocol.read_bytes()
    protocol.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="protocol drift"):
        GENERATOR.build_payload(copy)
    protocol.write_bytes(original)
    summary = copy / TERMINAL[2] / "summary.json"
    original_summary = summary.read_bytes()
    summary.write_bytes(original_summary + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        GENERATOR.build_payload(copy)
    summary.write_bytes(original_summary)
    sidecar = copy / TERMINAL[0] / "ledger-corrected.json"
    original_sidecar = sidecar.read_bytes()
    original_sidecar_hash = sidecar.with_name(sidecar.name + ".sha256.json").read_bytes()
    tampered = json.loads(original_sidecar.decode("utf-8"))
    tampered["end_state_window"]["corrected_ratio"] = 0.0001
    write_canonical_json(sidecar, tampered)
    with pytest.raises(ValueError, match="corrected windowed residual recomputed here"):
        GENERATOR.build_payload(copy)
    sidecar.write_bytes(original_sidecar)
    sidecar.with_name(sidecar.name + ".sha256.json").write_bytes(original_sidecar_hash)
    assessment = copy / TERMINAL[1] / "assessment.json"
    original_assessment = assessment.read_bytes()
    original_assessment_hash = assessment.with_name(assessment.name + ".sha256.json").read_bytes()
    record = json.loads(original_assessment.decode("utf-8"))
    record["verdict"] = "closure_quotable"
    write_canonical_json(assessment, record)
    with pytest.raises(ValueError, match="verdict"):
        GENERATOR.render_html(GENERATOR.build_payload(copy))
    assessment.write_bytes(original_assessment)
    assessment.with_name(assessment.name + ".sha256.json").write_bytes(original_assessment_hash)
    GENERATOR.render_html(GENERATOR.build_payload(copy))
    sidecar.unlink()
    with pytest.raises(ValueError, match="ledger-corrected.json is missing"):
        GENERATOR.build_payload(copy)
