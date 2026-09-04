"""Tests for the PIC-2D steady-state dashboard generator (skipped until results exist)."""

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
GENERATOR_PATH = MODERN / "visualization" / "generate_pic2d_cft_steady_state.py"
CHECKED_HTML = MODERN / "visualization" / "pic2d-cft-steady-state.html"
ANCHOR_PLATFORM = MODERN / "visualization" / "pic2d-cft-steady-state.anchor-platform.json"
EXPERIMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v2"
RESULTS = EXPERIMENT / "results"
# Payload leaves that are content hashes of arrays RE-SAMPLED at generation time (the P2 field map behind the cusp
# planes): bitwise identity on the generating platform only - provenance, compared for shape (64 hex) elsewhere.
DERIVED_HASH_LEAVES = {("cusps", "field_map_sha256")}
# Cross-platform numeric rule for the embedded payload: every recorded float was written as float(f"{v:.Ng}") with
# N in {4, 5, 6} (or is a full-precision copy of a stored value), so a last-ULP difference in a reduction on another
# CPU / BLAS / compiler can move a recorded value by at most ONE unit in its last significant digit; full-precision
# leaves get a relative floor of 1e-9 (a reduction over ~1e5 doubles carries ~1e-11 relative round-off).
RELATIVE_FLOOR = 1e-9
MIN_RECORDED_DIGITS = 4


def _load_generator():
    spec = importlib.util.spec_from_file_location("pic2d_steady_state_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()
pytestmark = pytest.mark.skipif(not (RESULTS / "summary.json").is_file(), reason="steady-state v2 results are not materialised")


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def test_payload_is_hash_bound_and_claim_bounded(payload) -> None:
    assert payload["schema"] == GENERATOR.SCHEMA
    assert payload["status"] == "development_screening_not_preregistered"
    statement = payload["claim_statement"].lower()
    for phrase in ("not preregistered", "not validated", "single seed", "under-resolved"):
        assert phrase in statement, phrase
    assert len(payload["simplifications"]) >= 10
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    headline = payload["cases"][0]
    assert headline["role"] == "headline" and headline["id"] == summary["case"]["id"]
    assert headline["protocol_sha256"] == summary["protocol_sha256"] == payload["protocol"]["file_sha256"]
    assert headline["maps_npz_sha256"] == summary["artifacts"]["maps_npz_sha256"]
    assert headline["stop_reason"] == "plateau_reached_after_min_transit_times"
    assert headline["plateau"]["reached"] is True and headline["plateau"]["transit_times_elapsed"] >= 3
    for key in ("discharge_current_drift", "electron_count_drift", "neutral_density_drift"):
        assert abs(headline["plateau"][key]) < 0.05
    assert headline["neutral_inventory"]["cumulative_ledger_closure_relative_to_inventory"] < 1e-10
    assert headline["series"]["neutral_density_per_m3"] and headline["series"]["neutral_fixed_point_per_m3"]
    assert len(headline["maps"]["n_e_per_m3"]) == len(headline["grid_r_m"])
    assert len(headline["axial_peak_n_e_per_m3"]) == len(headline["grid_z_m"])
    assert headline["cusps"]["cusp_z_m"], "cusp planes must be located from the P2 field map"
    assert headline["resolvability_at_peak"]["dz_over_lambda_d_at_peak"] > 0
    assert {v["name"] for v in payload["variants"]} == {"seed-b", "w-0.7"}
    GENERATOR.validate_payload(payload)


def test_seed_b_is_compared_to_the_headline_over_a_common_window(payload) -> None:
    seed_b = next(v for v in payload["variants"] if v["name"] == "seed-b")
    assert seed_b["state"] == "finished" and seed_b["reached_plateau"] is False and seed_b["transit_times"] < 3
    comparison = next(c for c in payload["comparisons"] if c["other_label"] == "variant seed-b")
    assert comparison["other_stop_reason"] == "wall_clock_budget_reached"
    common = comparison["windows"][0]
    assert common["t_end_s"] == pytest.approx(6.06e-6) and common["t_start_s"] == pytest.approx(0.8 * 6.06e-6)
    rows = {r["quantity"]: r for r in common["rows"]}
    for name in ("I_d", "I_beam,i", "S", "n_g", "N_e (macro)", "<T_e> (2/3 K/N)", "phi_max", "phi_min"):
        assert name in rows
    # same operating point, same time window: the seed-to-seed spread of the plateau quantities is at the per-cent level
    for name in ("I_d", "I_beam,i", "S", "n_g", "N_e (macro)"):
        assert abs(rows[name]["rel_diff"]) < 0.02, name
        assert rows[name]["samples_base"] > 1000 and rows[name]["samples_other"] > 1000
    assert rows["I_d"]["shot_noise_rel"] < abs(rows["I_d"]["rel_diff"]) < 0.01     # currents agree to < 1 %, above pure counting noise
    assert rows["n_g"]["shot_noise_rel"] is None
    # window B exists because seed-b stopped before the base plateau window
    offset = comparison["windows"][1]
    assert offset["t_start_s"] >= common["t_end_s"] and offset["other_t_end_s"] == common["t_end_s"]
    maps = comparison["maps"]["rows"]
    assert maps["wall ion flux"]["peak_z_base_m"] == maps["wall ion flux"]["peak_z_other_m"]   # same cusp-plane wall-flux peak
    assert 0 < maps["n_e"]["relative_l2_diff"] < 0.5
    html = GENERATOR.render_html(payload)
    assert '"other_label":"variant seed-b"' in html and "vs headline" in html and "shot-noise" in html


def test_ledger_correction_is_embedded_hash_bound_and_says_the_plateaus_were_heating(payload) -> None:
    block = payload["ledger_correction"]
    assert block["acceptance_bound"] == 0.02 and block["hard_gate"] == 0.05 and block["window_steps"] == 400_000
    assert "heating numerically at +7.2 to +13.0 %" in block["statement"] and "heated-grid value" in block["statement"]
    statement = payload["claim_statement"]
    assert "Energy-ledger correction (model v2.0.6, post hoc)" in statement and "HEATING numerically at +7.2 to +13.0 %" in statement
    assert "heated-grid value" in statement and "no energy-conserving or converged wording" in statement
    expected = {"results": (0.0037, 0.1301, 2.70e-6), "results-seed-b": (-0.0136, 0.1111, 2.76e-6), "results-w-0.7": (-0.0406, 0.0716, 4.50e-6)}
    assert [row["results_dir"] for row in block["cases"]] == list(expected)
    for row, case in zip(block["cases"], payload["cases"], strict=True):
        sidecar_path = EXPERIMENT / case["results_dir"] / "ledger-corrected.json"
        assert row["sidecar_sha256"] == sha256(sidecar_path.read_bytes()).hexdigest() == case["ledger_corrected"]["sidecar_sha256"]
        assert json.loads(sidecar_path.read_text(encoding="utf-8"))["inputs"]["series"]["sha256"] == case["series_npz_sha256"]
        recorded, corrected, gate = expected[case["results_dir"]]
        assert row["recorded_windowed"] == pytest.approx(recorded, abs=1e-3) and row["corrected_windowed"] == pytest.approx(corrected, abs=1e-3), case["label"]
        assert row["corrected_hard_gate_first_checkpoint_time_s"] == pytest.approx(gate, abs=2e-8)                  # every 50 um run would have been stopped
        assert row["recorded_below_bound"] is True and row["corrected_below_bound"] is False                       # (b) < 2 % PASS -> FAIL on all three
        assert row["corrected_windowed"] > block["hard_gate"] > block["acceptance_bound"]
        # the corrected windowed residual recomputed from the embedded series reproduces the sidecar; both series are embedded
        assert case["windowed_residual_corrected_recomputed"] == pytest.approx(row["corrected_windowed"], rel=1e-9)
        assert case["windowed_residual_recorded_recomputed"] == pytest.approx(row["recorded_windowed"], rel=1e-9)
        series = case["series"]
        assert len(series["windowed_residual_corrected_over_electrode_work"]) == len(series["time_s"]) == len(series["windowed_residual_recorded_over_electrode_work"])
        assert series["windowed_residual_corrected_over_electrode_work"][-1] == pytest.approx(row["corrected_windowed"], rel=1e-4)
        assert series["windowed_residual_corrected_over_electrode_work"][0] is None                                # incomplete window at the start
    html = GENERATOR.render_html(payload)
    for fragment in ('id="ledger"', 'id="residual"', 'id="residualCaption"', "the 50 µm plateaus were heating", "corrected ledger (H, model v2.0.6)",
                     "recorded (pre-v2.0.6) vs corrected (v2.0.6)", "Quotable statement"):
        assert fragment in html, fragment


def test_history_panels_keep_predecessors(payload) -> None:
    history = payload["history"]
    labels = [row["label"] for row in history["steady_state"]]
    assert any("v1.2 reference" in label for label in labels)
    assert any("attempt 1" in label for label in labels)
    for row in history["steady_state"]:
        assert row["plateau"]["reached"] is False
        assert len(row["summary_sha256"]) == 64
    assert history["snapshot_v2"] is not None and len(history["snapshot_v2"]["cases"]) == 4
    assert history["snapshot_v1"] is not None and "fail-closed" in history["snapshot_v1"]["lesson"]
    html = GENERATOR.render_html(payload)
    assert 'id="history"' in html and 'id="budget"' in html and 'id="verification"' in html


def _embedded_payload(html: str) -> dict[str, Any]:
    match = re.search(r'<script id="pic2d-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _significant_digits(value: float) -> int:
    if value == 0.0:
        return 0
    mantissa = repr(abs(value)).split("e")[0].replace(".", "").lstrip("0").rstrip("0")
    return len(mantissa) or 1


def _last_digit_unit(a: float, b: float) -> float:
    """One unit in the last recorded significant digit shared by two readings of the same quantity.

    The recorded precision is the longer of the two reprs (a trailing zero stripped by ``repr`` does not lower it)
    and never below ``MIN_RECORDED_DIGITS`` (every derived float in the payload was written with >= 4 significant
    digits), so ``0.1`` vs ``0.2`` is a real difference while ``0.364278`` vs ``0.364279`` is a last-digit flip.
    """

    magnitude = max(abs(a), abs(b))
    if magnitude == 0.0:
        return 0.0
    digits = max(MIN_RECORDED_DIGITS, _significant_digits(a), _significant_digits(b))
    return 10.0 ** (floor(log10(magnitude)) - digits + 1)


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
        tolerance = max(RELATIVE_FLOOR * max(abs(a), abs(b)), _last_digit_unit(float(a), float(b)))
        return abs(a - b) <= tolerance * (1 + 1e-9)
    return a == b


def payload_differences(expected: Any, actual: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any, Any]]:
    """Structural comparison of two dashboard payloads under the declared cross-platform rule; returns the violations."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [(path, sorted(expected), sorted(actual))]
        out: list[tuple[tuple[Any, ...], Any, Any]] = []
        for key in expected:
            out.extend(payload_differences(expected[key], actual[key], (*path, key)))
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [(path, len(expected), len(actual))]
        out = []
        for index, (e, a) in enumerate(zip(expected, actual, strict=True)):
            out.extend(payload_differences(e, a, (*path, index)))
        return out
    if path[-2:] in DERIVED_HASH_LEAVES:
        hex64 = isinstance(actual, str) and len(actual) == 64 and all(c in "0123456789abcdef" for c in actual)
        return [] if hex64 else [(path, expected, actual)]
    return [] if _leaves_agree(expected, actual) else [(path, expected, actual)]


def test_generation_is_byte_deterministic_and_checked_html_is_current(payload, tmp_path: Path) -> None:
    """Within one process / platform the generation is byte-deterministic.  Against the checked-in HTML the repo's
    replay policy applies: byte-exact only on the recorded anchor platform (OS + numpy build + SIMD dispatch + BLAS
    kernel, ``pic2d-cft-steady-state.anchor-platform.json``); elsewhere the embedded payload must agree structurally
    under the declared numeric rule (one unit in the last recorded digit, rel 1e-9 floor) and the re-sampled field
    hash is provenance.  Observed on a Lambda H100 box (Ubuntu 22.04, numpy 2.5.2 / scipy-openblas Haswell, 2026-09-04):
    generation deterministic, checked HTML (Windows anchor) differing in last digits of reductions and in
    ``cusps.field_map_sha256``."""

    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "pic2d-cft-steady-state.html"
    GENERATOR.generate(output)
    assert output.read_text(encoding="utf-8") == first
    sidecar = json.loads(GENERATOR.anchor_platform_path(output).read_text(encoding="utf-8"))
    assert sidecar["html_sha256"] == sha256(first.encode("utf-8")).hexdigest() and sidecar["platform"] == platform_fingerprint()
    checked = CHECKED_HTML.read_text(encoding="utf-8")
    anchor = json.loads(ANCHOR_PLATFORM.read_text(encoding="utf-8"))
    # the anchor sidecar must describe the checked-in bytes (it is regenerated together with them)
    assert anchor["html_sha256"] == sha256(checked.encode("utf-8")).hexdigest(), "anchor-platform sidecar does not describe the checked-in HTML"
    assert len(anchor["platform"]["fingerprint_sha256"]) == 64 and anchor["platform"]["os"] and anchor["platform"]["numpy"]
    if anchor["platform"]["fingerprint_sha256"] == platform_fingerprint()["fingerprint_sha256"]:
        assert checked == first, "on the anchor platform the checked-in HTML must be byte-current"
    else:
        differences = payload_differences(_embedded_payload(checked), _embedded_payload(first))
        assert not differences, f"{len(differences)} payload leaves outside the declared cross-platform tolerance: {differences[:5]}"
        # the HTML template around the payload is platform-independent
        assert re.sub(r'<script id="pic2d-data".*?</script>', "", checked, flags=re.DOTALL) == re.sub(r'<script id="pic2d-data".*?</script>', "", first, flags=re.DOTALL)


def test_payload_difference_rule_accepts_last_digit_flips_and_rejects_more() -> None:
    base = {"a": 0.364278, "b": [1.0, 2.5e-3], "c": {"x": 12345, "cusps": {"field_map_sha256": "a" * 64, "cusp_z_m": [0.006025]}}, "s": "t", "n": None}
    assert payload_differences(base, deepcopy(base)) == []
    flipped = deepcopy(base)
    flipped["a"] = 0.364279                                     # one unit in the 6th significant digit
    flipped["b"][1] = 2.501e-3                                  # one unit in the 4th digit of a value recorded with a trailing zero stripped
    flipped["c"]["cusps"]["field_map_sha256"] = "b" * 64        # derived-map hash: provenance, any 64-hex accepted
    assert payload_differences(base, flipped) == []
    rejected = deepcopy(base)
    rejected["a"] = 0.364280                                    # two units
    assert [d[0] for d in payload_differences(base, rejected)] == [("a",)]
    rejected = deepcopy(base)
    rejected["c"]["x"] = 12346                                  # integers are exact
    rejected["s"] = "u"
    rejected["c"]["cusps"]["field_map_sha256"] = "not-a-hash"
    assert {d[0] for d in payload_differences(base, rejected)} == {("c", "x"), ("s",), ("c", "cusps", "field_map_sha256")}
    assert payload_differences(base, {**base, "extra": 1}) and payload_differences(base, {**base, "b": [1.0]})
    # the unit of the last recorded digit: the longer repr sets the precision, never below 4 significant digits
    assert _last_digit_unit(0.364278, 0.364279) == pytest.approx(1e-6) and _last_digit_unit(0.36428, 0.364279) == pytest.approx(1e-6)
    assert _last_digit_unit(2.5e-3, 2.4e-3) == pytest.approx(1e-6) and _last_digit_unit(0.1, 0.0999999) == pytest.approx(1e-6)
    assert _last_digit_unit(12345.0, 12345.0) == pytest.approx(1.0) and _last_digit_unit(0.0, 0.0) == 0.0
    assert _leaves_agree(0.1, 0.0999999) and not _leaves_agree(0.1, 0.2) and not _leaves_agree(0.12, 0.1202) and _leaves_agree(0.12, 0.1201)
    assert _leaves_agree(1.2345678901234567e-3, 1.2345678901234580e-3) and not _leaves_agree(1.2345678901234567e-3, 1.23456790e-3)
    assert not _leaves_agree(3, 4) and _leaves_agree(3, 3.0) and not _leaves_agree(True, 1.0) and _leaves_agree(None, None)


def test_checked_html_and_anchor_platform_sidecar_are_consistent() -> None:
    anchor = json.loads(ANCHOR_PLATFORM.read_text(encoding="utf-8"))
    assert anchor["schema"] == "cft-pic2d-dashboard-anchor-platform/1.0.0" and anchor["html_file"] == CHECKED_HTML.name
    assert anchor["html_sha256"] == sha256(CHECKED_HTML.read_bytes()).hexdigest()
    platform = anchor["platform"]
    for key in ("os", "machine", "numpy", "simd_dispatch_enabled", "blas", "cpu_model", "fingerprint_sha256"):
        assert key in platform, key
    assert platform["blas"]["name"] and platform["fingerprint_sha256"] != sha256(b"").hexdigest()
    # the checked-in payload's derived-map hash is 64 hex and its declared source identities are the P2 bundle's
    checked = _embedded_payload(CHECKED_HTML.read_text(encoding="utf-8"))
    cusps = checked["cases"][0]["cusps"]
    assert len(cusps["field_map_sha256"]) == 64 and len(cusps["field_source_sha256"]) == 64
    assert cusps["checkpoint_file_sha256"] == checked["cases"][0]["field"]["provenance"]["checkpoint_file_sha256"]
    assert cusps["source_identity_sha256"] == checked["cases"][0]["field"]["provenance"]["source_identity_sha256"]


def test_html_is_self_contained_offline(payload) -> None:
    html = GENERATOR.render_html(payload)
    lowered = html.lower()
    assert '<script id="pic2d-data" type="application/json">' in html
    for forbidden in ("fetch(", "xmlhttprequest", "websocket", "cdn"):
        assert forbidden not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", html, re.IGNORECASE)
    assert not re.search(r"\bhttps?://", html, re.IGNORECASE)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", html)


def test_html_has_claim_panel_controls_and_accessibility(payload) -> None:
    html = GENERATOR.render_html(payload)
    for fragment in (
        'id="claim"', 'for="case"', 'for="map"', 'for="scale"', 'id="theme"', 'tabindex="0"', 'role="img"',
        'aria-live="polite"', 'e.key==="ArrowLeft"', 'e.key==="Home"', "new ResizeObserver(schedule)",
        "window.devicePixelRatio", "createImageData", 'id="wall"', 'id="exit"', 'id="energy"', 'id="neutral"', 'id="rates"',
        'id="axial"', 'id="convergence"', "Claim boundary", "one-cell stair-step", "reported, not hidden", "cusp planes",
    ):
        assert fragment in html, fragment
    assert "<svg" not in html.lower()


def test_embedded_json_round_trips_strictly(payload) -> None:
    html = GENERATOR.render_html(payload)
    match = re.search(r'<script id="pic2d-data" type="application/json">(.*?)</script>', html, re.DOTALL)
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
    script = tmp_path / "pic2d.js"
    script.write_text(scripts[-1], encoding="utf-8")
    completed = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_tampered_payload_is_rejected(payload) -> None:
    for mutate in (
        lambda p: p.__setitem__("status", "accepted"),
        lambda p: p["cases"][0].__setitem__("summary_sha256", "abc"),
        lambda p: p["cases"][0]["maps"]["phi_v"].pop(),
        lambda p: p["cases"][0].__setitem__("protocol_sha256", "0" * 64),
        lambda p: p["cases"][0].__setitem__("stop_reason", "converged"),
        lambda p: p["history"]["steady_state"][0].__setitem__("stop_reason", "converged"),
        lambda p: p["variants"][0].__setitem__("state", "done"),
        lambda p: p.__setitem__("claim_statement", "validated steady state"),
        # the ledger correction: present, hash-bound, values consistent, bound / gate unchanged, statement honest
        lambda p: p.pop("ledger_correction"),
        lambda p: p["ledger_correction"].__setitem__("acceptance_bound", 0.05),
        lambda p: p["ledger_correction"]["cases"][0].__setitem__("corrected_windowed", 0.01),
        lambda p: p["ledger_correction"]["cases"][0].__setitem__("corrected_below_bound", True),
        lambda p: p["ledger_correction"]["cases"][1].__setitem__("sidecar_sha256", "0" * 64),
        lambda p: p["ledger_correction"].__setitem__("statement", "the 50 um plateaus were fine"),
        lambda p: p["cases"][0].__setitem__("windowed_residual_corrected_recomputed", 0.0),
        lambda p: p["cases"][0]["series"].pop("windowed_residual_corrected_over_electrode_work"),
        lambda p: p.__setitem__("claim_statement", p["claim_statement"].replace("heated-grid value", "grid value")),
    ):
        changed = deepcopy(payload)
        mutate(changed)
        with pytest.raises(ValueError):
            GENERATOR.validate_payload(changed)


def test_protocol_drift_and_tampered_results_are_rejected(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    shutil.copytree(RESULTS, experiment / "results", ignore=shutil.ignore_patterns("checkpoint*", "*.jsonl", "*.log", "*.err", "*.pid"))
    for name in ("protocol.json", "variants.json"):
        shutil.copy(EXPERIMENT / name, experiment / name)
    # a protocol file that no longer matches the hash recorded by the run is rejected
    protocol = experiment / "protocol.json"
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol drift"):
        GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
    shutil.copy(EXPERIMENT / "protocol.json", protocol)
    GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
    # a tampered summary is rejected by its sidecar
    summary = experiment / "results" / "summary.json"
    original_summary = summary.read_bytes()
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
    summary.write_bytes(original_summary)
    # the ledger-corrected sidecar must describe the embedded series and reproduce from it; it is required
    sidecar = experiment / "results" / "ledger-corrected.json"
    original_sidecar = sidecar.read_bytes()
    tampered = json.loads(original_sidecar.decode("utf-8"))
    tampered["inputs"]["series"]["sha256"] = "0" * 64
    write_canonical_json(sidecar, tampered)
    with pytest.raises(ValueError, match="describes another series"):
        GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
    tampered = json.loads(original_sidecar.decode("utf-8"))
    tampered["end_state_window"]["corrected_ratio"] = 0.0
    write_canonical_json(sidecar, tampered)
    with pytest.raises(ValueError, match="corrected windowed residual recomputed here"):
        GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
    sidecar.unlink()
    sidecar.with_name(sidecar.name + ".sha256.json").unlink()
    with pytest.raises(ValueError, match="ledger-corrected.json is missing"):
        GENERATOR.build_payload(experiment / "results", protocol, experiment / "variants.json")
