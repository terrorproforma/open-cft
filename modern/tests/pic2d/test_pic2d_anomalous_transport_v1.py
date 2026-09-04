"""pic2d_anomalous_transport_v1: the preregistered Bohm alpha-series on the 33 um reference plateau (roadmap R1, model v2.1.0).

* composition contract: each sealed case protocol is the ss-v4 protocol with exactly the declared changes (the rotation closure at the
  case's alpha, the v2.0.6 Debye floor, K = 5, the budget, the acceptance / reference / text blocks); geometry, operating point, grid, dt,
  W, seed, cadences and the v2.0.3 gate thresholds are byte-for-byte v4's; the identities differ from v4's and from each other;
* the sealed files equal their recomposition and the campaign's listed hashes (the launch's refusal path);
* the alpha = 0 reference block equals the ss-v4 artifacts on disk and states its corrected-ledger (b) FAIL;
* the shakedown protocol shrinks cadences only;
* the per-case assessment classifies synthetic outcomes (plateau_clean / plateau_heating / no_plateau) with the shift table and the
  per-cusp report; the series assessment yields trend_confirmed / trend_not_confirmed / inconclusive by the predeclared rule;
* the launch discipline (lock, dirty worktree, wrong commit, drifted protocol, --require-mps);
* a tiny CPU end-to-end run of a shrunk case through the shared runner -> finalize -> assess (the hook is live in the runner path);
* AMENDMENT 1 (2026-09-05): the model v2.1.1 drift-member arming latch + the v2.0 ignition gate in every sealed case (identities unchanged),
  the amendment's genealogy (pre-amendment sealed hashes = the blobs at the preregistration commit 057841cf), and - once the alpha-1over16
  launch-1 record is checked out - the amended gates' reading of that extinguished run (ignition gate fails at 1.0 us; the latch never arms).
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from math import pi
from pathlib import Path

import numpy as np
import pytest
from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d import artifacts
from cft_revival.pic2d.models import PIC2DValidationError
from cft_revival.pic2d.sensitivity import ANOMALOUS_MODEL_ROTATION, AnomalousCollisionConfig
from experiments.pic2d_anomalous_transport_v1 import protocol as pm
from experiments.pic2d_anomalous_transport_v1 import run as at
from experiments.pic2d_cft_steady_state_v1 import run as runner

V4_CONFIG_SHA256_CUDA = "f10772b25b030e2ba9d86cbd48b9972b9e143cce689e0b0ee8d4d6f10ccc685f"
UNCHANGED_TOP_LEVEL = ("geometry", "operating_point", "design_id", "field_authority", "cross_sections", "case")
V4_HAS_RESULTS = (pm.V4_RESULTS / "maps.npz").is_file() and (pm.V4_RESULTS / "summary.json").is_file()


def test_case_protocols_are_the_v4_template_with_exactly_the_declared_changes():
    v4 = pm.load_v4_protocol()
    base = runner.build_config(v4, backend="warp-cuda")
    assert artifacts.config_identity(base) == V4_CONFIG_SHA256_CUDA
    identities = set()
    for case, meta in pm.CASES.items():
        p = pm.load_case_protocol(case)
        assert p["campaign"]["alpha"] == meta["alpha"] and p["campaign"]["case"] == case and p["experiment_id"] == f"{pm.EXPERIMENT_ID}-{case}"
        for key in UNCHANGED_TOP_LEVEL:
            if key == "case":
                assert {k: v for k, v in p["case"].items() if k not in ("id", "seed_note")} == {k: v for k, v in v4["case"].items() if k not in ("id", "seed_note")}
            else:
                assert p[key] == v4[key], key
        num, v4num = p["numerics"], v4["numerics"]
        assert num["anomalous_collisions"]["model"] == ANOMALOUS_MODEL_ROTATION and num["anomalous_collisions"]["alpha"] == meta["alpha"]
        assert num["performance"] == {k: v for k, v in num["performance"].items()} and num["performance"]["moment_sample_interval"] == 5
        assert num["peak_debye_gate"]["min_accumulated_macro_particle_steps_at_peak"] == 64000
        for key in ("dt_s", "ion_subcycle", "device_sync_steps", "series_interval_steps", "averaging_window_steps", "checkpoint_every_steps", "stability_limits",
                    "stability_reference", "step_graph", "frame_recorder"):
            assert num[key] == v4num[key], key
        gate_keys = {k for k in num["peak_debye_gate"] if not k.startswith("min_accumulated")}
        assert {k: num["peak_debye_gate"][k] for k in gate_keys} == v4num["peak_debye_gate"]
        stop, v4stop = p["stopping_rule"], v4["stopping_rule"]
        for key in ("plateau", "plateau_threshold", "plateau_window_fraction", "min_transit_times", "ignition_check"):
            assert stop[key] == v4stop[key], key
        # amendment 1: the triad block is v4's plus the v2.1.1 arming latch; the v2.0 ignition gate is declared beside it (both outside config_sha256)
        triad = {k: v for k, v in stop["grid_heating_triad"].items() if k not in ("note", "drift_members_arming")}
        assert triad == {k: v for k, v in v4stop["grid_heating_triad"].items() if k != "note"}
        assert stop["grid_heating_triad"]["enforced_after_transit_times"] == 1.0 and "SUPERSEDED by drift_members_arming" in stop["grid_heating_triad"]["note"]
        assert stop["grid_heating_triad"]["drift_members_arming"] == pm.DRIFT_MEMBERS_ARMING and stop["ignition_gate"] == pm.IGNITION_GATE
        arming = stop["grid_heating_triad"]["drift_members_arming"]
        assert (arming["min_transit_times"], arming["settle_quantity"], arming["settle_drift_max"]) == (2.0, "discharge_current", stop["plateau_threshold"])
        assert arming["settle_check_cadence_steps"] == num["checkpoint_every_steps"] == 40_000
        assert [c["time_s"] for c in stop["ignition_gate"]["checks"]] == [1.0e-6, 2.0e-6] and "ignition_gate" not in v4stop
        assert "AMENDMENT 1" in stop["fail_closed"] and any(change.startswith("AMENDMENT 1") for change in p["campaign"]["changes"])
        config = runner.build_config(p, backend="warp-cuda")
        assert config.anomalous == AnomalousCollisionConfig(meta["alpha"], model=ANOMALOUS_MODEL_ROTATION) and config.moment_sample_interval == 5
        assert config.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak == 64000 and config.peak_debye_gate.max_cells_per_debye == pi
        assert config.grid == base.grid and config.dt_s == base.dt_s and config.macro_weight == base.macro_weight and config.seed == base.seed == 20260903
        assert config.injection == base.injection and config.seed_plasma == base.seed_plasma and config.mcc == base.mcc and config.neutral_inventory == base.neutral_inventory
        # the physics identity differs from v4 by the closure (+ the declared K / floor) only
        mine = config.to_dict()
        theirs = base.to_dict()
        assert set(mine) - set(theirs) == {"anomalous", "moment_sample_interval"}
        assert {k: v for k, v in mine.items() if k not in ("anomalous", "moment_sample_interval", "peak_debye_gate")} == {k: v for k, v in theirs.items() if k != "peak_debye_gate"}
        identities.add(artifacts.config_identity(config))
        runner.frame_recorder_config(p).validate_alignment(sync_steps=200, checkpoint_every_steps=40_000, window_steps=400_000)
        assert p["reference_run"]["corrected_ledger"]["acceptance_b_below_0p02"] is False and p["reference_run"]["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"] == pytest.approx(0.0246, abs=1e-4)
        assert stop["acceptance"]["c_series_trend"]["hypotheses"] == pm.HYPOTHESES and set(stop["acceptance"]["d_verdict"]["series"]) == {"trend_confirmed", "trend_not_confirmed", "inconclusive"}
    assert len(identities) == 3 and V4_CONFIG_SHA256_CUDA not in identities
    assert pm.LAUNCH_PRIORITY == ("alpha-1over16", "alpha-1over64", "alpha-0.345") and set(pm.LAUNCH_PRIORITY) == set(pm.CASES)


def test_sealed_files_equal_their_recomposition_and_the_campaign_hashes(monkeypatch):
    campaign = pm.load_campaign()
    for case in pm.CASES:
        sealed = at.verify_sealed(case)
        key = f"modern/experiments/pic2d_anomalous_transport_v1/protocols/{case}.json"
        on_disk = (pm.PROTOCOLS_DIR / f"{case}.json").read_bytes()
        assert on_disk == canonical_bytes(sealed) + b"\n" and campaign["sealed_protocols"][key] == pm.protocol_sha256(sealed)
    assert campaign["design"]["closure"]["series"] == {"alpha-1over64": 1 / 64, "alpha-1over16": 1 / 16, "alpha-0.345": 0.345, "alpha-0": 0.0}
    assert campaign["design"]["closure"]["d_perp_over_kt_e_by_eb"]["alpha-0.345"] == pytest.approx(0.345 / (1 + 0.345**2))
    assert campaign["amendments"] == pm.AMENDMENTS and campaign["acceptance"] == pm.load_case_protocol("alpha-1over16")["stopping_rule"]["acceptance"]
    amendment = campaign["amendments"][0]
    assert amendment["id"] == 1 and amendment["pre_amendment_sealed_sha256"] == pm.PRE_AMENDMENT_SEALED_SHA256
    assert set(amendment["pre_amendment_sealed_sha256"]) == set(pm.CASES) and all(len(v) == 64 for v in amendment["pre_amendment_sealed_sha256"].values())
    # the amendment re-sealed every case: no sealed hash equals its pre-amendment value, and the change list names both new stopping-rule keys
    for case in pm.CASES:
        assert campaign["sealed_protocols"][f"modern/experiments/pic2d_anomalous_transport_v1/protocols/{case}.json"] != pm.PRE_AMENDMENT_SEALED_SHA256[case]
    assert any("drift_members_arming" in c for c in amendment["changes"]) and any("ignition_gate" in c for c in amendment["changes"])
    assert "NOT relaunched" in amendment["launch_1_disposition"]
    # a tampered sealed file is refused (fail closed) - via a copy, never touching the tracked file
    tampered = copy.deepcopy(pm.load_case_protocol("alpha-1over16"))
    tampered["numerics"]["anomalous_collisions"]["alpha"] = 0.07
    monkeypatch.setattr(at, "load_case_protocol", lambda case: tampered)
    with pytest.raises(PIC2DValidationError, match="differs from its recomposition"):
        at.verify_sealed("alpha-1over16")


PREREG_COMMIT = "057841cfcd72d24d09608143319079fc9f750e99"
LAUNCH1_RESULTS = pm.HERE / "results" / "alpha-1over16"


def _git_blob_sha256(commit: str, path: str) -> str | None:
    try:
        out = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, cwd=pm.MODERN, check=False)
    except OSError:
        return None
    return hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None


def test_amendment_1_genealogy_binds_the_pre_amendment_seals_to_the_preregistration_commit():
    campaign = pm.load_campaign()
    amendment = campaign["amendments"][0]
    assert amendment["preregistration_commit"] == PREREG_COMMIT[:8]
    seen = 0
    for case, recorded in pm.PRE_AMENDMENT_SEALED_SHA256.items():
        blob = _git_blob_sha256(PREREG_COMMIT, f"modern/experiments/pic2d_anomalous_transport_v1/protocols/{case}.json")
        if blob is None:
            continue
        seen += 1
        assert blob == recorded, case            # the recorded genealogy IS the sealed file at 057841cf
    if seen == 0:
        pytest.skip("preregistration commit not reachable in this checkout")
    # the launch-1 record (if checked out) was executed under the pre-amendment 1/16 seal
    if (LAUNCH1_RESULTS / "summary.json").is_file():
        summary = json.loads((LAUNCH1_RESULTS / "summary.json").read_text(encoding="utf-8"))
        assert summary["protocol_sha256"] == pm.PRE_AMENDMENT_SEALED_SHA256["alpha-1over16"]
        assert summary["stop_reason"] == "grid_heating_triad_gate_stopped_run" and summary["git_head"].startswith(PREREG_COMMIT[:12])


@pytest.mark.skipif(not (LAUNCH1_RESULTS / "series.npz").is_file(), reason="alpha-1over16 launch-1 record not checked out")
def test_amended_gates_read_the_extinguished_launch_1_as_no_ignition_and_never_arm_the_drift_members():
    """The recorded 1/16 launch (stopped at 1.0033 transits by the S / T_e,dense drift members): under the amended protocol the ignition
    gate stops it at the first checkpoint past 1.0 us on N_e (0.45 < 0.6), the latch never closes (I_d drift -0.5 ... -0.7), the residual
    member never fires (+1.15 % at the stop) - the record's own numbers, re-read with the amended rule."""

    protocol = pm.load_case_protocol("alpha-1over16")
    rule = protocol["stopping_rule"]
    series = np.load(LAUNCH1_RESULTS / "series.npz")
    arrays = {key: np.asarray(series[key], dtype=np.float64) for key in series.files}
    transit = float(runner.protocol_budget(protocol)["ion_transit_time_s"])
    assert arrays["step"][-1] == 1_720_000 and arrays["time_s"][-1] == pytest.approx(2.408e-6, rel=1e-6)
    # ignition gate at the checkpoints: passes nothing after 1.0 us
    first_fail = None
    for n in range(1, arrays["step"].size + 1):
        if int(arrays["step"][n - 1]) % 40_000 != 0:
            continue
        ignition = runner.evaluate_ignition({k: v[:n] for k, v in arrays.items()}, rule)
        if ignition["failed"]:
            first_fail = (float(arrays["time_s"][n - 1]), ignition)
            break
    assert first_fail is not None
    t_stop, ignition = first_fail
    assert 1.0e-6 <= t_stop <= 1.06e-6 and ignition["checks"][0]["electron_ratio"] < 0.6 and "no ignition" in ignition["reason"]
    # the amended triad at the recorded stop: unarmed drift members (no latch), the same drift readings, no residual failure
    triad = runner.evaluate_triad(arrays, rule, transit)
    arming = triad["drift_members_arming"]
    assert arming["latched"] is False and arming["armed"] is False and arming["current_settle_drift"] < -0.4
    assert triad["ionisation_rate_drift"] == pytest.approx(-0.618, abs=0.002) and triad["t_e_dense_drift"] == pytest.approx(0.366, abs=0.002)
    assert triad["hard_failures"] == [] and 0.0 < triad["windowed_energy_residual_over_electrode_work"] < 0.02
    # under the recorded (pre-amendment) rule the same series trips both drift members at this record - the recorded stop reproduced
    legacy = copy.deepcopy(rule)
    del legacy["grid_heating_triad"]["drift_members_arming"]
    recorded = runner.evaluate_triad(arrays, legacy, transit)
    assert recorded["enforced"] is True and len(recorded["hard_failures"]) == 2 and all("drift" in f for f in recorded["hard_failures"])


@pytest.mark.skipif(not V4_HAS_RESULTS, reason="ss-v4 artifacts not checked out")
def test_reference_block_equals_the_v4_artifacts_and_states_the_corrected_ledger_fail():
    p = pm.load_case_protocol("alpha-1over16")
    recomputed = at.reference_quantities_from_files(pm.V4_RESULTS)
    for key, value in recomputed.items():
        assert p["reference_run"]["quantities"][key] == pytest.approx(value, rel=1e-12), key
    sidecar = json.loads((pm.V4_RESULTS / "ledger-corrected.json").read_text(encoding="utf-8"))
    assert sidecar["end_state_window"]["corrected_ratio"] == p["reference_run"]["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"]
    assert sidecar["acceptance_b_residual_power_below_0p02"]["corrected_passes"] is False
    assessment = json.loads((pm.V4_RESULTS / "assessment.json").read_text(encoding="utf-8"))
    assert assessment["b_residual_power"]["passed"] is True      # the RECORDED (pre-v2.0.6) statistic passed; the corrected one fails - both stated
    cusps = at.run_quantities(pm.V4_RESULTS, runner.build_config(p, backend="cpu").grid)["per_cusp"]
    assert [round(c["z_c_m"] * 1e3, 3) for c in cusps] == [6.028, 12.0, 17.972]
    assert all(c["electron_wall_current_a"] > 0 and c["ion_wall_current_a"] > 0 for c in cusps)
    assert sum(c["electron_wall_current_a"] for c in cusps) < 1.05 * at.run_quantities(pm.V4_RESULTS)["wall_electron_a"]     # the three cusp windows hold most, not more than all


def test_shakedown_protocol_shrinks_cadences_only():
    protocol = pm.load_case_protocol("alpha-0.345")
    shake = at.shakedown_protocol(protocol)
    a = runner.build_config(protocol, backend="cpu")
    b = runner.build_config(shake, backend="cpu")
    assert a.grid == b.grid and a.dt_s == b.dt_s and a.macro_weight == b.macro_weight and a.anomalous == b.anomalous and a.moment_sample_interval == b.moment_sample_interval
    assert b.peak_debye_gate.max_cells_per_debye == pi and b.peak_debye_gate.window_steps == 40_000 and b.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak == 64000
    assert shake["numerics"]["checkpoint_every_steps"] == 4_000 and shake["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] == 40_000
    assert shake["status"].startswith("SHAKEDOWN") and shake["experiment_id"].endswith("-shakedown")


def _fake_results(tmp_path: Path, protocol: dict, *, scale: dict[str, float] | None = None, stop: str = "plateau_reached_after_min_transit_times",
                  windowed: float | None = 0.01, complete: bool = True, name: str | None = None) -> Path:
    scale = scale or {}
    ref = protocol["reference_run"]["quantities"]
    results = tmp_path / (name or f"results-{len(list(tmp_path.iterdir()))}")
    results.mkdir(parents=True)
    grid = runner.build_config(protocol, backend="cpu").grid
    n = np.zeros(grid.node_shape)
    t = np.zeros(grid.node_shape)
    n[20, 429] = ref["peak_n_e_window_per_m3"] * scale.get("peak_n_e_window_per_m3", 1.0)
    t[20, 429] = ref["t_e_peak_window_ev"] * scale.get("t_e_peak_window_ev", 1.0)
    artifacts.write_npz(results / "maps.npz", {"n_e_per_m3": n, "t_e_ev": t, "window_steps": np.array([400_000])})
    summary = {
        "stop_reason": stop, "ion_transit_times": 3.05, "steps_completed": 5_240_000, "git_head": "deadbeef", "protocol_sha256": "0" * 64,
        "provenance": {"config_sha256": "1" * 64}, "maps_kind": "window_average", "sessions": [{}],
        "final_series": {"currents_a": {"anomalous_collision_rate_per_s": 1e21}},
        "window_currents_a": {"discharge_a": ref["discharge_current_a"] * scale.get("discharge_current_a", 1.0), "exit_ion_beam_a": ref["exit_ion_beam_a"] * scale.get("exit_ion_beam_a", 1.0),
                              "wall_electron_a": 3e-3, "wall_ion_a": 3e-3},
        "neutral_inventory": {"trailing_20pct_mean_ionization_rate_per_s": ref["ionization_rate_per_s"] * scale.get("ionization_rate_per_s", 1.0),
                              "propellant_utilisation_trailing": ref["gross_utilisation"] * scale.get("gross_utilisation", 1.0),
                              "trailing_20pct_mean_density_per_m3": ref["neutral_density_per_m3"] * scale.get("neutral_density_per_m3", 1.0)},
        "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": windowed, "windowed_energy_residual_window_complete": complete, "energy_residual_over_electrode_work": 0.01},
        "peak_node_debye": {"window": {"cells_per_debye_window_last": 1.6, "trailing_20pct_mean_cells_per_debye_window": 1.6, "soft_ok": True}},
        "plateau": {"reached": stop == "plateau_reached_after_min_transit_times"},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    return results


def test_case_assessment_classifies_and_tabulates_shifts_with_the_particle_band(tmp_path: Path):
    protocol = pm.load_case_protocol("alpha-1over16")
    quiet = lambda _: None
    same = at.assess_case("alpha-1over16", results=_fake_results(tmp_path, protocol), protocol=protocol, log=quiet, reference_check=False)
    assert same["verdict"] == "plateau_clean" and all(row["status"] == "inside_band" for row in same["c_shifts_vs_alpha_0"].values())
    assert same["b_residual_power"]["reference_reads"] == pytest.approx(0.0246, abs=1e-4) and same["per_cusp_vs_alpha_0"] is None
    expected = at.assess_case("alpha-1over16", results=_fake_results(tmp_path, protocol, scale={"discharge_current_a": 1.3, "peak_n_e_window_per_m3": 0.7, "ionization_rate_per_s": 0.8,
                                                                                                 "gross_utilisation": 0.8, "t_e_peak_window_ev": 0.85, "neutral_density_per_m3": 1.1}),
                              protocol=protocol, log=quiet, reference_check=False)
    rows = expected["c_shifts_vs_alpha_0"]
    assert rows["discharge_current_a"]["status"] == "confirming" and rows["discharge_current_a"]["relative_shift"] == pytest.approx(0.3)
    assert rows["peak_n_e_window_per_m3"]["status"] == "confirming" and rows["neutral_density_per_m3"]["status"] == "confirming" and rows["exit_ion_beam_a"]["status"] == "inside_band"
    opposite = at.assess_case("alpha-1over16", results=_fake_results(tmp_path, protocol, scale={"discharge_current_a": 0.7}), protocol=protocol, log=quiet, reference_check=False)
    assert opposite["c_shifts_vs_alpha_0"]["discharge_current_a"]["status"] == "contradicting"
    heating = at.assess_case("alpha-1over16", results=_fake_results(tmp_path, protocol, windowed=0.03), protocol=protocol, log=quiet, reference_check=False)
    assert heating["verdict"] == "plateau_heating" and not heating["b_residual_power"]["passed"]
    incomplete = at.assess_case("alpha-1over16", results=_fake_results(tmp_path, protocol, windowed=0.0, complete=False), protocol=protocol, log=quiet, reference_check=False)
    assert incomplete["verdict"] == "plateau_heating"
    budget_results = _fake_results(tmp_path, protocol, stop="wall_clock_budget_reached")
    budget = at.assess_case("alpha-1over16", results=budget_results, protocol=protocol, log=quiet, reference_check=False)
    assert budget["verdict"] == "no_plateau" and "no trend contribution" in budget["verdict_rule"]
    record = artifacts.read_canonical_json(budget_results / "assessment.json")
    assert record["schema_version"] == at.ASSESSMENT_SCHEMA and record["alpha"] == 1 / 16 and record["verdict"] == "no_plateau"
    if V4_HAS_RESULTS:
        checked = at.assess_case("alpha-1over16", results=_fake_results(tmp_path, protocol), protocol=protocol, log=quiet, reference_check=True)
        assert all(v["agree"] for v in checked["reference_consistency"].values())
        bad = copy.deepcopy(protocol)
        bad["reference_run"]["quantities"]["discharge_current_a"] *= 1.01
        with pytest.raises(PIC2DValidationError, match="ss-v4 artifacts"):
            at.assess_case("alpha-1over16", results=_fake_results(tmp_path, protocol), protocol=bad, log=quiet, reference_check=True)


def test_series_assessment_applies_the_predeclared_trend_rule(tmp_path: Path):
    quiet = lambda _: None
    root = tmp_path / "results"

    def build(shifts_by_case: dict[str, dict[str, float] | None], stops: dict[str, str] | None = None) -> dict:
        for case in pm.CASES:
            case_dir = root / case
            if case_dir.exists():
                import shutil
                shutil.rmtree(case_dir)
            scale = shifts_by_case.get(case)
            if scale is None:
                continue
            p = pm.load_case_protocol(case)
            results = _fake_results(root, p, scale=scale, stop=(stops or {}).get(case, "plateau_reached_after_min_transit_times"), name=case)
            at.assess_case(case, results=results, protocol=p, log=quiet, reference_check=False)
        return at.assess_series(results_root=root, log=quiet)

    # nothing launched -> inconclusive with only the reference reached
    empty = at.assess_series(results_root=root, log=quiet)
    assert empty["verdict"] == "inconclusive" and empty["points_reached"] == ["alpha-0"] and set(empty["points_unreached"]) == set(pm.CASES)
    # monotone in the declared directions beyond the band at 3 points -> confirmed
    up = {"discharge_current_a": 1.15, "peak_n_e_window_per_m3": 0.85, "ionization_rate_per_s": 0.9, "gross_utilisation": 0.9, "t_e_peak_window_ev": 0.9}
    more = {"discharge_current_a": 1.4, "peak_n_e_window_per_m3": 0.7, "ionization_rate_per_s": 0.8, "gross_utilisation": 0.8, "t_e_peak_window_ev": 0.8}
    confirmed = build({"alpha-1over64": up, "alpha-1over16": more, "alpha-0.345": None})
    assert confirmed["verdict"] == "trend_confirmed" and confirmed["points_reached"] == ["alpha-0", "alpha-1over64", "alpha-1over16"] and confirmed["contradictions"] == []
    assert confirmed["monotonicity"]["discharge_current_a"]["monotone_in_declared_direction"] is True
    # a reversal of I_d beyond the band between 1/64 and 1/16 -> not confirmed
    reversed_ = build({"alpha-1over64": more, "alpha-1over16": up, "alpha-0.345": None})
    assert reversed_["verdict"] == "trend_not_confirmed" and reversed_["monotonicity"]["discharge_current_a"]["monotone_in_declared_direction"] is False
    # a contradicting sign at a reached point -> not confirmed even if the key pair is monotone
    contra = build({"alpha-1over64": up, "alpha-1over16": more | {"t_e_peak_window_ev": 1.3}, "alpha-0.345": None})
    assert contra["verdict"] == "trend_not_confirmed" and contra["contradictions"] == ["alpha-1over16:t_e_peak_window_ev"]
    # only two points reached (one case at budget) -> inconclusive
    two = build({"alpha-1over64": up, "alpha-1over16": more, "alpha-0.345": None}, stops={"alpha-1over16": "wall_clock_budget_reached"})
    assert two["verdict"] == "inconclusive" and two["points"]["alpha-1over16"]["verdict"] == "no_plateau"
    # everything inside the band -> inconclusive (no measurable effect), never "confirmed"
    flat = build({"alpha-1over64": {"discharge_current_a": 1.01}, "alpha-1over16": {"discharge_current_a": 1.02}, "alpha-0.345": {"discharge_current_a": 1.03}})
    assert flat["verdict"] == "inconclusive" and flat["key_shifts_all_inside_band"] is True
    record = artifacts.read_canonical_json(root / "series-assessment.json")
    assert record["schema_version"] == at.SERIES_ASSESSMENT_SCHEMA and record["launch_priority"] == list(pm.LAUNCH_PRIORITY)


def test_launch_discipline_lock_dirty_worktree_commit_protocol_and_mps(tmp_path: Path, monkeypatch):
    quiet = lambda _: None
    monkeypatch.setattr(at, "git", lambda *args, cwd=None: "f" * 40 if args[:1] == ("rev-parse",) else "")
    with pytest.raises(PIC2DValidationError, match="unknown case"):
        at.launch("alpha-1", results=tmp_path / "x", log=quiet)
    with pytest.raises(PIC2DValidationError, match="not the preregistration commit"):
        at.launch("alpha-1over16", results=tmp_path / "launch", expect_commit="0123456", log=quiet)
    monkeypatch.setattr(at, "worktree_status", lambda cwd=None: [" M modern/x.py"])
    with pytest.raises(PIC2DValidationError, match="not clean"):
        at.launch("alpha-1over16", results=tmp_path / "launch", expect_commit="fffffff", log=quiet)
    monkeypatch.setattr(at, "worktree_status", lambda cwd=None: [])
    monkeypatch.setattr(at, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else ("blob-a" if args[0] == "rev-parse" else "blob-b"))
    with pytest.raises(PIC2DValidationError, match="differs from the committed blob"):
        at.launch("alpha-1over16", results=tmp_path / "launch", log=quiet)
    monkeypatch.setattr(at, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else "same")
    monkeypatch.setattr(at, "preflight_path", lambda case: tmp_path / f"preflight-{case}.json")
    monkeypatch.setattr(at, "shakedown_path", lambda case: tmp_path / f"shakedown-{case}.json")
    with pytest.raises(PIC2DValidationError, match="preflight-alpha-1over16.json and a shakedown"):
        at.launch("alpha-1over16", results=tmp_path / "launch", log=quiet)
    (tmp_path / "preflight-alpha-1over16.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shakedown-alpha-0.345.json").write_text("{}", encoding="utf-8")       # one shakedown of the series suffices (same code path, alpha only)
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(PIC2DValidationError, match="--require-mps"):
        at.launch("alpha-1over16", results=tmp_path / "launch", require_mps=True, log=quiet)
    with pytest.raises(PIC2DValidationError, match="--resume needs"):
        at.launch("alpha-1over16", results=tmp_path / "launch", resume=True, log=quiet)
    assert not (tmp_path / "launch" / at.LOCK_NAME).exists()
    # the lock itself
    payload = {"schema_version": at.LOCK_SCHEMA, "experiment_id": "x", "commit": "a" * 40, "protocol_sha256": "b" * 64}
    at.acquire_lock(tmp_path / "results", payload)
    with pytest.raises(PIC2DValidationError, match="same-attempt"):
        at.acquire_lock(tmp_path / "results", payload)


def _tiny_case_protocol(case: str, max_steps: int) -> dict:
    """The sealed case shrunk to a 12 x 96 grid / large W for a CPU smoke of the runner path (NOT the protocol; a test fixture)."""

    p = at.shakedown_protocol(pm.load_case_protocol(case))
    p["case"]["radial_cells"], p["case"]["axial_cells"], p["case"]["macro_weight"] = 12, 96, 3.0e6
    p["operating_point"]["seed_plasma_density_per_m3"] = 2e16
    num = p["numerics"]
    num["series_interval_steps"] = num["device_sync_steps"] = 20
    num["checkpoint_every_steps"] = 100
    num["averaging_window_steps"] = 200
    num["frame_recorder"] = {"cadence_steps": 100, "precision": "float32"}
    num["peak_debye_gate"]["window_steps"] = 200
    num["peak_debye_gate"]["window_snapshot_steps"] = 100
    num["performance"]["moment_sample_interval"] = 5
    num["stability_limits"]["max_cell_debye_ratio"] = 50.0
    p["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] = 200
    p["stopping_rule"]["wall_budget_seconds"] = 600.0
    return p


def test_tiny_cpu_run_of_a_shrunk_case_through_the_runner_finalize_and_assess(tmp_path: Path):
    p = _tiny_case_protocol("alpha-0.345", 400)
    results = tmp_path / "alpha-0.345"
    results.mkdir()
    protocol_path = tmp_path / "protocol-tiny.json"
    artifacts.write_canonical_json(protocol_path, p)
    quiet = lambda _: None
    summary_path = runner.run_steady_state(p, results, backend="cpu", max_steps=400, protocol_path=protocol_path, log=quiet)
    summary = artifacts.read_canonical_json(summary_path)
    # the 12 x 96 / W 3e6 fixture heats (the residual-power gate stops it at 300 steps) - the plumbing, not the physics, is under test
    assert summary["steps_completed"] >= 200 and summary["final_series"]["ledger"]["cumulative"]["anomalous"] > 0
    assert summary["final_series"]["currents_a"]["anomalous_collision_rate_per_s"] > 0
    assert summary["provenance"]["config"]["anomalous"] == {"model": ANOMALOUS_MODEL_ROTATION, "alpha": 0.345}
    assessment = at.assess_case("alpha-0.345", results=results, protocol=p, log=quiet, reference_check=False)
    assert assessment["verdict"] == "no_plateau" and assessment["run"]["anomalous_collision_rate_per_s"] > 0 and assessment["run"]["anomalous_events_cumulative"] > 0
    assert assessment["run"]["per_cusp"] is not None and len(assessment["run"]["per_cusp"]) == 3
    series = at.assess_series(results_root=tmp_path, log=quiet)
    assert series["verdict"] == "inconclusive" and series["points"]["alpha-0.345"]["verdict"] == "no_plateau"
