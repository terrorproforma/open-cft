"""pic2d_physics_effects_v1: the preregistered SEE(BN) / xe_collision_set_v2 campaign on the 33 um reference plateau (roadmap R2 + R3).

* composition contract: each sealed case protocol is the ss-v4 protocol with exactly the declared changes (the effect block(s), the v2.0.6
  Debye floor, K = 5, the budget, the acceptance / reference / text blocks; NO anomalous-transport block); geometry, operating point (but for
  the collision-set block), grid, dt, W, seed, cadences and the v2.0.3 gate thresholds are byte-for-byte v4's; the identities differ from v4's
  and from each other; the runner builds the declared SEEConfig / CollisionSetConfig;
* the sealed files equal their recomposition and the campaign's listed hashes (the launch's refusal path);
* the reference block equals the ss-v4 artifacts on disk (incl. the quantities this campaign adds) and states its corrected-ledger (b) FAIL;
* the shakedown protocol shrinks cadences only;
* the per-case assessment classifies synthetic outcomes (plateau status + hypothesis verdict) with the shift table (relative / absolute bands, the
  '0' hypothesis), the per-cusp report and the SEE / collision readings; the campaign assessment yields the additivity statement by the predeclared rule;
* the launch discipline (lock, dirty worktree, wrong commit, drifted protocol, --require-mps);
* a tiny CPU end-to-end run of the shrunk combined case through the shared runner -> finalize -> assess (both effects live in the runner path).
"""

from __future__ import annotations

import copy
import json
import shutil
from math import pi
from pathlib import Path

import numpy as np
import pytest

from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d import artifacts
from cft_revival.pic2d.models import PIC2DValidationError
from cft_revival.pic2d.see import SEEConfig
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_physics_effects_v1 import protocol as pm
from experiments.pic2d_physics_effects_v1 import run as pe

V4_CONFIG_SHA256_CUDA = "f10772b25b030e2ba9d86cbd48b9972b9e143cce689e0b0ee8d4d6f10ccc685f"
UNCHANGED_TOP_LEVEL = ("geometry", "design_id", "field_authority")
V4_HAS_RESULTS = (pm.V4_RESULTS / "maps.npz").is_file() and (pm.V4_RESULTS / "summary.json").is_file()
QUIET = lambda _: None


def test_case_protocols_are_the_v4_template_with_exactly_the_declared_changes():
    v4 = pm.load_v4_protocol()
    base = runner.build_config(v4, backend="warp-cuda")
    assert artifacts.config_identity(base) == V4_CONFIG_SHA256_CUDA
    identities = set()
    for case, meta in pm.CASES.items():
        p = pm.load_case_protocol(case)
        assert p["campaign"]["case"] == case and p["campaign"]["effects"] == meta["effects"] and p["experiment_id"] == f"{pm.EXPERIMENT_ID}-{case}"
        for key in UNCHANGED_TOP_LEVEL:
            assert p[key] == v4[key], key
        assert {k: v for k, v in p["case"].items() if k not in ("id", "seed_note")} == {k: v for k, v in v4["case"].items() if k not in ("id", "seed_note")}
        op, v4op = p["operating_point"], v4["operating_point"]
        assert {k: v for k, v in op.items() if k != "collision_set"} == v4op
        assert ("collision_set" in op) is meta["collision_set"]
        assert (p["cross_sections"] == v4["cross_sections"]) is (not meta["collision_set"])
        num, v4num = p["numerics"], v4["numerics"]
        assert "anomalous_collisions" not in num
        assert ("see" in num) is meta["see"]
        assert num["performance"]["moment_sample_interval"] == 5
        assert num["peak_debye_gate"]["min_accumulated_macro_particle_steps_at_peak"] == 64000
        for key in ("dt_s", "ion_subcycle", "device_sync_steps", "series_interval_steps", "averaging_window_steps", "checkpoint_every_steps", "stability_limits",
                    "stability_reference", "step_graph", "frame_recorder", "poisson", "deposition"):
            assert num[key] == v4num[key], key
        gate_keys = {k for k in num["peak_debye_gate"] if not k.startswith("min_accumulated")}
        assert {k: num["peak_debye_gate"][k] for k in gate_keys} == v4num["peak_debye_gate"]
        stop, v4stop = p["stopping_rule"], v4["stopping_rule"]
        for key in ("plateau", "plateau_threshold", "plateau_window_fraction", "min_transit_times", "ignition_check"):
            assert stop[key] == v4stop[key], key
        triad = {k: v for k, v in stop["grid_heating_triad"].items() if k != "note"}
        assert triad == {k: v for k, v in v4stop["grid_heating_triad"].items() if k != "note"}
        config = runner.build_config(p, backend="warp-cuda")
        assert config.anomalous is None and config.moment_sample_interval == 5
        assert config.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak == 64000 and config.peak_debye_gate.max_cells_per_debye == pi
        assert config.grid == base.grid and config.dt_s == base.dt_s and config.macro_weight == base.macro_weight and config.seed == base.seed == 20260903
        assert config.injection == base.injection and config.seed_plasma == base.seed_plasma and config.neutral_inventory == base.neutral_inventory
        assert config.potentials == base.potentials
        if meta["see"]:
            assert config.see == SEEConfig(enabled=True, material="BN") == pm.see_config_of(case)
            assert config.see.to_dict()["constants"]["delta_max"] == 2.016 and config.see.ion_induced_yield == 0.0
        else:
            assert config.see is None and pm.see_config_of(case) is None
        if meta["collision_set"]:
            assert config.mcc.collision_set is not None and config.mcc.collision_set.to_dict()["name"] == "xe_collision_set_v2"
            assert config.mcc.collision_set.ion_neutral is not None
            assert config.mcc.neutral_density_per_m3 == base.mcc.neutral_density_per_m3 and config.mcc.neutral_temperature_k == base.mcc.neutral_temperature_k
        else:
            assert config.mcc == base.mcc
        # the physics identity differs from v4 by the effect block(s) (+ the declared K / floor) only
        mine, theirs = config.to_dict(), base.to_dict()
        expected_new = {"moment_sample_interval"} | ({"see"} if meta["see"] else set())
        assert set(mine) - set(theirs) == expected_new
        differing = {k for k in theirs if k in mine and mine[k] != theirs[k]}
        assert differing == {"peak_debye_gate"} | ({"mcc"} if meta["collision_set"] else set()), differing
        identities.add(artifacts.config_identity(config))
        runner.frame_recorder_config(p).validate_alignment(sync_steps=200, checkpoint_every_steps=40_000, window_steps=400_000)
        ref = p["reference_run"]
        assert ref["corrected_ledger"]["acceptance_b_below_0p02"] is False and ref["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"] == pytest.approx(0.0246, abs=1e-4)
        acc = stop["acceptance"]
        assert acc["c_shifts"]["hypotheses"] == pm.HYPOTHESES_BY_CASE[case] and acc["c_shifts"]["key_quantities"] == list(pm.KEY_QUANTITIES[case])
        assert set(acc["d_verdict"]["per_case_hypothesis_verdict"]) == {"confirmed", "not_confirmed", "inconclusive"}
        assert set(acc["d_verdict"]["plateau_status"]) == {"plateau_clean", "plateau_heating", "no_plateau"}
        assert "combined_vs_sum_of_parts" in acc["d_verdict"]
    assert len(identities) == 3 and V4_CONFIG_SHA256_CUDA not in identities
    assert pm.LAUNCH_PRIORITY == ("see-bn", "xe-set-v2", "see-bn+xe-set-v2") and set(pm.LAUNCH_PRIORITY) == set(pm.CASES)


def test_sealed_files_equal_their_recomposition_and_the_campaign_hashes(monkeypatch):
    campaign = pm.load_campaign()
    for case in pm.CASES:
        sealed = pe.verify_sealed(case)
        key = f"modern/experiments/pic2d_physics_effects_v1/protocols/{case}.json"
        on_disk = (pm.PROTOCOLS_DIR / f"{case}.json").read_bytes()
        assert on_disk == canonical_bytes(sealed) + b"\n" and campaign["sealed_protocols"][key] == pm.protocol_sha256(sealed)
        assert campaign["acceptance"][case] == sealed["stopping_rule"]["acceptance"]
    assert campaign["design"]["see_block"] == pm.SEE_BN_BLOCK and campaign["design"]["collision_set_block"] == pm.COLLISION_SET_V2_BLOCK
    assert campaign["amendments"] == [] and campaign["design"]["launch_priority"] == list(pm.LAUNCH_PRIORITY)
    # a tampered sealed file is refused (fail closed) - via a copy, never touching the tracked file
    tampered = copy.deepcopy(pm.load_case_protocol("see-bn"))
    tampered["numerics"]["see"]["emission_temperature_ev"] = 3.0
    monkeypatch.setattr(pe, "load_case_protocol", lambda case: tampered)
    with pytest.raises(PIC2DValidationError, match="differs from its recomposition"):
        pe.verify_sealed("see-bn")


@pytest.mark.skipif(not V4_HAS_RESULTS, reason="ss-v4 artifacts not checked out")
def test_reference_block_equals_the_v4_artifacts_and_states_the_corrected_ledger_fail():
    p = pm.load_case_protocol("see-bn+xe-set-v2")
    grid = runner.build_config(p, backend="cpu").grid
    recomputed = pe.reference_quantities_from_files(pm.V4_RESULTS, grid)
    for key in pm.QUANTITY_KEYS:
        assert recomputed[key] is not None, key
        assert p["reference_run"]["quantities"][key] == pytest.approx(recomputed[key], rel=1e-12), key
    assert 0.0 < p["reference_run"]["quantities"]["iedf_low_energy_fraction"] < 0.15        # the legacy set's exit IEDF has a small slow population
    assert p["reference_run"]["quantities"]["wall_electron_power_w"] > 0 and p["reference_run"]["quantities"]["wall_ion_mean_energy_ev"] > 0
    sidecar = json.loads((pm.V4_RESULTS / "ledger-corrected.json").read_text(encoding="utf-8"))
    assert sidecar["end_state_window"]["corrected_ratio"] == p["reference_run"]["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"]
    assert sidecar["acceptance_b_residual_power_below_0p02"]["corrected_passes"] is False
    q = pe.run_quantities(pm.V4_RESULTS, grid)
    cusps = q["per_cusp"]
    assert [round(c["z_c_m"] * 1e3, 3) for c in cusps] == [6.028, 12.0, 17.972]
    assert all(c["electron_wall_current_a"] > 0 and c["ion_wall_current_a"] > 0 and "see_effective_yield" not in c for c in cusps)
    assert all(c["wall_ion_mean_energy_ev"] > 0 for c in cusps)
    assert "see" not in q and "collision_set" not in q and q["iedf"]["total_macro_ions"] > 0


def test_shakedown_protocol_shrinks_cadences_only():
    protocol = pm.load_case_protocol("see-bn+xe-set-v2")
    shake = pe.shakedown_protocol(protocol)
    a = runner.build_config(protocol, backend="cpu")
    b = runner.build_config(shake, backend="cpu")
    assert a.grid == b.grid and a.dt_s == b.dt_s and a.macro_weight == b.macro_weight and a.see == b.see and a.mcc == b.mcc and a.moment_sample_interval == b.moment_sample_interval
    assert b.peak_debye_gate.max_cells_per_debye == pi and b.peak_debye_gate.window_steps == 40_000 and b.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak == 64000
    assert shake["numerics"]["checkpoint_every_steps"] == 4_000 and shake["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] == 40_000
    assert shake["status"].startswith("SHAKEDOWN") and shake["experiment_id"].endswith("-shakedown")


def _fake_results(root: Path, protocol: dict, *, scale: dict[str, float] | None = None, iedf_fraction: float | None = None,
                  stop: str = "plateau_reached_after_min_transit_times", windowed: float | None = 0.01, complete: bool = True, name: str | None = None,
                  see: bool = False, collision: bool = False) -> Path:
    scale = scale or {}
    ref = protocol["reference_run"]["quantities"]
    results = root / (name or f"results-{len(list(root.iterdir())) if root.exists() else 0}")
    results.mkdir(parents=True)
    grid = runner.build_config(protocol, backend="cpu").grid
    n = np.zeros(grid.node_shape)
    t = np.zeros(grid.node_shape)
    n[20, 429] = ref["peak_n_e_window_per_m3"] * scale.get("peak_n_e_window_per_m3", 1.0)
    t[20, 429] = ref["t_e_peak_window_ev"] * scale.get("t_e_peak_window_ev", 1.0)
    edges = np.linspace(0.0, 450.0, 257)
    counts = np.zeros(256)
    frac = ref["iedf_low_energy_fraction"] if iedf_fraction is None else iedf_fraction
    counts[10] = 1000.0 * frac            # centre 18.5 eV (< 30 eV)
    counts[80] = 1000.0 * (1.0 - frac)    # centre 141.6 eV
    maps = {"n_e_per_m3": n, "t_e_ev": t, "window_steps": np.array([400_000]), "iedf_ion_counts": counts, "iedf_edges_ev": edges}
    artifacts.write_npz(results / "maps.npz", maps)
    currents = {"discharge_a": ref["discharge_current_a"] * scale.get("discharge_current_a", 1.0), "exit_ion_beam_a": ref["exit_ion_beam_a"] * scale.get("exit_ion_beam_a", 1.0),
                "wall_electron_a": 3e-3, "wall_ion_a": 3e-3, "anode_ion_a": ref["anode_ion_a"] * scale.get("anode_ion_a", 1.0)}
    cumulative = {"ionizations": 1e6, "exit_ions": 5e5}
    if see:
        currents.update({"see_emission_a": 2e-3, "see_effective_yield": 0.7})
        cumulative.update({"see_impacts": 4e6, "see_electrons": 2.8e6, "ke_see_emitted_j": 1e-7})
    if collision:
        currents.update({"cex_rate_per_s": 2e15, "mex_rate_per_s": 1e15, "fast_neutral_exit_rate_per_s": 3e14, "fast_neutral_wall_rate_per_s": 1.5e15, "fast_neutral_thermal_rate_per_s": 2e14})
        cumulative.update({"cex": 5e4, "mex": 2.5e4, "fast_neutral_exit_channel": 8e3, "pz_fast_neutral_exit": 1e-12, "pz_exit_ions": 2e-10,
                           "excitations_level_1": 100.0, "excitations_level_2": 90.0, "excitations_level_3": 180.0, "excitations_level_4": 80.0})
    summary = {
        "stop_reason": stop, "ion_transit_times": 3.05, "steps_completed": 5_240_000, "simulated_time_s": 7.3e-6, "git_head": "deadbeef", "protocol_sha256": "0" * 64,
        "provenance": {"config_sha256": "1" * 64}, "maps_kind": "window_average", "sessions": [{}], "averaging_window_steps": 400_000,
        "final_series": {"time_s": 7.3e-6, "currents_a": dict(currents), "ledger": {"cumulative": cumulative}},
        "window_currents_a": currents,
        "neutral_inventory": {"trailing_20pct_mean_ionization_rate_per_s": ref["ionization_rate_per_s"] * scale.get("ionization_rate_per_s", 1.0),
                              "propellant_utilisation_trailing": ref["gross_utilisation"] * scale.get("gross_utilisation", 1.0),
                              "trailing_20pct_mean_density_per_m3": ref["neutral_density_per_m3"] * scale.get("neutral_density_per_m3", 1.0)},
        "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": windowed, "windowed_energy_residual_window_complete": complete, "energy_residual_over_electrode_work": 0.01},
        "peak_node_debye": {"window": {"cells_per_debye_window_last": 1.6, "trailing_20pct_mean_cells_per_debye_window": 1.6, "soft_ok": True}},
        "plateau": {"reached": stop == "plateau_reached_after_min_transit_times"},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    return results


def test_case_assessment_classifies_and_tabulates_shifts_with_relative_and_absolute_bands(tmp_path: Path):
    p_see = pm.load_case_protocol("see-bn")
    p_xe = pm.load_case_protocol("xe-set-v2")
    # identical to the reference: SEE case -> no contradiction, keys inside the band -> inconclusive; plateau clean
    same = pe.assess_case("see-bn", results=_fake_results(tmp_path, p_see, see=True), protocol=p_see, log=QUIET, reference_check=False)
    assert same["plateau_status"] == "plateau_clean" and same["hypothesis_verdict"] == "inconclusive"
    assert all(row["status"] in ("inside_band", "reported", "confirming", "unavailable") for row in same["c_shifts_vs_reference"].values())
    assert {k for k, row in same["c_shifts_vs_reference"].items() if row["status"] == "unavailable"} == {"wall_electron_power_w", "wall_ion_mean_energy_ev"}   # no wall profiles in the fixture
    assert same["c_shifts_vs_reference"]["iedf_low_energy_fraction"]["kind"] == "absolute" and same["c_shifts_vs_reference"]["iedf_low_energy_fraction"]["status"] == "reported"
    assert same["run"]["see"]["window_effective_yield"] == 0.7 and same["b_residual_power"]["reference_reads"] == pytest.approx(0.0246, abs=1e-4) and same["per_cusp_vs_reference"] is None
    # the audit's SEE expectation -> confirmed
    expected = pe.assess_case("see-bn", results=_fake_results(tmp_path, p_see, see=True, scale={"discharge_current_a": 1.2, "t_e_peak_window_ev": 0.85, "peak_n_e_window_per_m3": 0.9}),
                              protocol=p_see, log=QUIET, reference_check=False)
    rows = expected["c_shifts_vs_reference"]
    assert expected["hypothesis_verdict"] == "confirmed" and rows["discharge_current_a"]["status"] == "confirming" and rows["discharge_current_a"]["shift"] == pytest.approx(0.2)
    assert rows["t_e_peak_window_ev"]["status"] == "confirming" and rows["peak_n_e_window_per_m3"]["status"] == "inside_band" and rows["exit_ion_beam_a"]["status"] == "reported"
    # I_d up but T_e up beyond the band -> a contradiction -> not_confirmed
    opposite = pe.assess_case("see-bn", results=_fake_results(tmp_path, p_see, see=True, scale={"discharge_current_a": 1.2, "t_e_peak_window_ev": 1.15}), protocol=p_see, log=QUIET, reference_check=False)
    assert opposite["hypothesis_verdict"] == "not_confirmed" and opposite["c_shifts_vs_reference"]["t_e_peak_window_ev"]["status"] == "contradicting"
    # heating plateau: verdict still evaluated, status plateau_heating
    heating = pe.assess_case("see-bn", results=_fake_results(tmp_path, p_see, see=True, windowed=0.03, scale={"discharge_current_a": 1.2, "t_e_peak_window_ev": 0.85}), protocol=p_see, log=QUIET, reference_check=False)
    assert heating["plateau_status"] == "plateau_heating" and not heating["b_residual_power"]["passed"] and heating["hypothesis_verdict"] == "confirmed"
    incomplete = pe.assess_case("see-bn", results=_fake_results(tmp_path, p_see, see=True, windowed=0.0, complete=False), protocol=p_see, log=QUIET, reference_check=False)
    assert incomplete["plateau_status"] == "plateau_heating"
    # budget stop -> no_plateau, inconclusive regardless of the shifts
    budget_results = _fake_results(tmp_path, p_see, see=True, stop="wall_clock_budget_reached", scale={"discharge_current_a": 1.3, "t_e_peak_window_ev": 0.8})
    budget = pe.assess_case("see-bn", results=budget_results, protocol=p_see, log=QUIET, reference_check=False)
    assert budget["plateau_status"] == "no_plateau" and budget["hypothesis_verdict"] == "inconclusive"
    record = artifacts.read_canonical_json(budget_results / "assessment.json")
    assert record["schema_version"] == pe.ASSESSMENT_SCHEMA and record["effects"] == ["see_dielectric_v1"] and record["plateau_status"] == "no_plateau"
    # the collision-set case: the '0' hypothesis on I_d and the ABSOLUTE band on the IEDF fraction
    ref_frac = p_xe["reference_run"]["quantities"]["iedf_low_energy_fraction"]
    xe_ok = pe.assess_case("xe-set-v2", results=_fake_results(tmp_path, p_xe, collision=True, iedf_fraction=ref_frac + 0.2, scale={"ionization_rate_per_s": 0.96}), protocol=p_xe, log=QUIET, reference_check=False)
    rows = xe_ok["c_shifts_vs_reference"]
    assert xe_ok["hypothesis_verdict"] == "confirmed" and rows["iedf_low_energy_fraction"]["status"] == "confirming" and rows["iedf_low_energy_fraction"]["shift"] == pytest.approx(0.2, abs=1e-6)
    assert rows["discharge_current_a"]["status"] == "confirming" and rows["discharge_current_a"]["hypothesis_sign"] == "0" and rows["ionization_rate_per_s"]["status"] == "inside_band"
    assert xe_ok["run"]["collision_set"]["cex_over_ionization"] == pytest.approx(2e15 / (p_xe["reference_run"]["quantities"]["ionization_rate_per_s"] * 0.96))
    assert xe_ok["run"]["collision_set"]["excitation_level_shares"] == pytest.approx([100 / 450, 90 / 450, 180 / 450, 80 / 450])
    assert xe_ok["run"]["collision_set"]["run_average_rates"]["fast_neutral_exit_momentum_rate_n"] == pytest.approx(1e-12 / 7.3e-6)
    xe_id_moves = pe.assess_case("xe-set-v2", results=_fake_results(tmp_path, p_xe, collision=True, iedf_fraction=ref_frac + 0.2, scale={"discharge_current_a": 1.2}), protocol=p_xe, log=QUIET, reference_check=False)
    assert xe_id_moves["hypothesis_verdict"] == "not_confirmed" and xe_id_moves["c_shifts_vs_reference"]["discharge_current_a"]["status"] == "contradicting"
    xe_small = pe.assess_case("xe-set-v2", results=_fake_results(tmp_path, p_xe, collision=True, iedf_fraction=ref_frac + 0.01), protocol=p_xe, log=QUIET, reference_check=False)
    assert xe_small["hypothesis_verdict"] == "inconclusive" and xe_small["c_shifts_vs_reference"]["iedf_low_energy_fraction"]["status"] == "inside_band"
    xe_wrong = pe.assess_case("xe-set-v2", results=_fake_results(tmp_path, p_xe, collision=True, iedf_fraction=max(ref_frac - 0.05, 0.0)), protocol=p_xe, log=QUIET, reference_check=False)
    assert xe_wrong["hypothesis_verdict"] == "not_confirmed"
    if V4_HAS_RESULTS:
        checked = pe.assess_case("see-bn", results=_fake_results(tmp_path, p_see, see=True), protocol=p_see, log=QUIET, reference_check=True)
        assert all(v["agree"] for v in checked["reference_consistency"].values())
        bad = copy.deepcopy(p_see)
        bad["reference_run"]["quantities"]["discharge_current_a"] *= 1.01
        with pytest.raises(PIC2DValidationError, match="ss-v4 artifacts"):
            pe.assess_case("see-bn", results=_fake_results(tmp_path, p_see, see=True), protocol=bad, log=QUIET, reference_check=True)


def test_campaign_assessment_applies_the_additivity_rule(tmp_path: Path):
    root = tmp_path / "results"

    def build(by_case: dict[str, dict | None], stops: dict[str, str] | None = None) -> dict:
        for case in pm.CASES:
            case_dir = root / case
            if case_dir.exists():
                shutil.rmtree(case_dir)
            spec = by_case.get(case)
            if spec is None:
                continue
            p = pm.load_case_protocol(case)
            results = _fake_results(root, p, scale=spec.get("scale"), iedf_fraction=spec.get("iedf"), stop=(stops or {}).get(case, "plateau_reached_after_min_transit_times"),
                                    name=case, see=pm.CASES[case]["see"], collision=pm.CASES[case]["collision_set"])
            pe.assess_case(case, results=results, protocol=p, log=QUIET, reference_check=False)
        return pe.assess_campaign(results_root=root, log=QUIET)

    ref_frac = pm.load_campaign()["reference_run"]["quantities"]["iedf_low_energy_fraction"]
    empty = pe.assess_campaign(results_root=root, log=QUIET)
    assert empty["cases_reached"] == [] and empty["additivity"]["statement"] == "not_evaluable" and set(empty["cases_unreached"]) == set(pm.CASES)
    see = {"scale": {"discharge_current_a": 1.2, "t_e_peak_window_ev": 0.85, "peak_n_e_window_per_m3": 0.9}}
    xe = {"scale": {"ionization_rate_per_s": 0.96}, "iedf": ref_frac + 0.2}
    # combined = exactly the sum of the parts -> additive, every case confirmed
    both = {"scale": {"discharge_current_a": 1.2, "t_e_peak_window_ev": 0.85 - 0.0, "peak_n_e_window_per_m3": 0.9, "ionization_rate_per_s": 0.96}, "iedf": ref_frac + 0.2}
    additive = build({"see-bn": see, "xe-set-v2": xe, "see-bn+xe-set-v2": both})
    assert additive["verdicts"] == {"see-bn": "confirmed", "xe-set-v2": "confirmed", "see-bn+xe-set-v2": "confirmed"}
    assert additive["additivity"]["statement"] == "additive" and additive["additivity"]["rows"]["discharge_current_a"]["interaction"] == pytest.approx(0.0, abs=1e-12)
    # the combined I_d shift far above the sum -> super-additive; far below -> sub-additive
    more = dict(both, scale=dict(both["scale"], discharge_current_a=1.5))
    super_ = build({"see-bn": see, "xe-set-v2": xe, "see-bn+xe-set-v2": more})
    assert super_["additivity"]["statement"] == "interacting" and super_["additivity"]["rows"]["discharge_current_a"]["classification"] == "super_additive"
    assert super_["additivity"]["non_additive_quantities"] == ["discharge_current_a"]
    less = dict(both, scale=dict(both["scale"], discharge_current_a=1.05))
    sub = build({"see-bn": see, "xe-set-v2": xe, "see-bn+xe-set-v2": less})
    assert sub["additivity"]["rows"]["discharge_current_a"]["classification"] == "sub_additive"
    # a single case at budget -> the additivity is not evaluable, the other verdicts stand
    partial = build({"see-bn": see, "xe-set-v2": xe, "see-bn+xe-set-v2": both}, stops={"xe-set-v2": "wall_clock_budget_reached"})
    assert partial["additivity"]["statement"] == "not_evaluable" and partial["verdicts"]["xe-set-v2"] == "inconclusive" and partial["verdicts"]["see-bn"] == "confirmed"
    record = artifacts.read_canonical_json(root / "campaign-assessment.json")
    assert record["schema_version"] == pe.CAMPAIGN_ASSESSMENT_SCHEMA and record["launch_priority"] == list(pm.LAUNCH_PRIORITY)


def test_launch_discipline_lock_dirty_worktree_commit_protocol_and_mps(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pe, "git", lambda *args, cwd=None: "f" * 40 if args[:1] == ("rev-parse",) else "")
    with pytest.raises(PIC2DValidationError, match="unknown case"):
        pe.launch("see", results=tmp_path / "x", log=QUIET)
    with pytest.raises(PIC2DValidationError, match="not the preregistration commit"):
        pe.launch("see-bn", results=tmp_path / "launch", expect_commit="0123456", log=QUIET)
    monkeypatch.setattr(pe, "worktree_status", lambda cwd=None: [" M modern/x.py"])
    with pytest.raises(PIC2DValidationError, match="not clean"):
        pe.launch("see-bn", results=tmp_path / "launch", expect_commit="fffffff", log=QUIET)
    monkeypatch.setattr(pe, "worktree_status", lambda cwd=None: [])
    monkeypatch.setattr(pe, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else ("blob-a" if args[0] == "rev-parse" else "blob-b"))
    with pytest.raises(PIC2DValidationError, match="differs from the committed blob"):
        pe.launch("see-bn", results=tmp_path / "launch", log=QUIET)
    monkeypatch.setattr(pe, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else "same")
    monkeypatch.setattr(pe, "preflight_path", lambda case: tmp_path / f"preflight-{case}.json")
    monkeypatch.setattr(pe, "shakedown_path", lambda case: tmp_path / f"shakedown-{case}.json")
    with pytest.raises(PIC2DValidationError, match="preflight-see-bn.json and shakedown-see-bn.json"):
        pe.launch("see-bn", results=tmp_path / "launch", log=QUIET)
    (tmp_path / "preflight-see-bn.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PIC2DValidationError, match="shakedown-see-bn.json"):      # every case needs ITS OWN shakedown (different code paths)
        pe.launch("see-bn", results=tmp_path / "launch", log=QUIET)
    (tmp_path / "shakedown-see-bn.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(PIC2DValidationError, match="--require-mps"):
        pe.launch("see-bn", results=tmp_path / "launch", require_mps=True, log=QUIET)
    with pytest.raises(PIC2DValidationError, match="--resume needs"):
        pe.launch("see-bn", results=tmp_path / "launch", resume=True, log=QUIET)
    assert not (tmp_path / "launch" / pe.LOCK_NAME).exists()
    payload = {"schema_version": pe.LOCK_SCHEMA, "experiment_id": "x", "commit": "a" * 40, "protocol_sha256": "b" * 64}
    pe.acquire_lock(tmp_path / "results", payload)
    with pytest.raises(PIC2DValidationError, match="same-attempt"):
        pe.acquire_lock(tmp_path / "results", payload)


def _tiny_case_protocol(case: str) -> dict:
    """The sealed case shrunk to a 12 x 96 grid / large W for a CPU smoke of the runner path (NOT the protocol; a test fixture)."""

    p = pe.shakedown_protocol(pm.load_case_protocol(case))
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


def test_tiny_cpu_run_of_the_shrunk_combined_case_through_the_runner_finalize_and_assess(tmp_path: Path):
    p = _tiny_case_protocol("see-bn+xe-set-v2")
    results = tmp_path / "see-bn+xe-set-v2"
    results.mkdir()
    protocol_path = tmp_path / "protocol-tiny.json"
    artifacts.write_canonical_json(protocol_path, p)
    summary_path = runner.run_steady_state(p, results, backend="cpu", max_steps=400, protocol_path=protocol_path, log=QUIET)
    summary = artifacts.read_canonical_json(summary_path)
    cumulative = summary["final_series"]["ledger"]["cumulative"]
    # the 12 x 96 / W 3e6 fixture is not physics: the plumbing of both effects through the runner is under test
    assert summary["steps_completed"] >= 200
    assert summary["provenance"]["config"]["see"]["material"] == "BN" and summary["provenance"]["config"]["mcc"]["collision_set"]["name"] == "xe_collision_set_v2"
    assert "see_impacts" in cumulative and "cex" in cumulative and "excitations_level_1" in cumulative
    assert "see_emission_a" in summary["window_currents_a"] and "cex_rate_per_s" in summary["window_currents_a"]
    assessment = pe.assess_case("see-bn+xe-set-v2", results=results, protocol=p, log=QUIET, reference_check=False)
    assert assessment["plateau_status"] == "no_plateau" and assessment["hypothesis_verdict"] == "inconclusive"
    assert assessment["run"]["see"] is not None and assessment["run"]["collision_set"] is not None and assessment["run"]["iedf"] is not None
    assert assessment["run"]["per_cusp"] is not None and len(assessment["run"]["per_cusp"]) == 3
    assert all("space_charge_limited" in c and "see_effective_yield" in c for c in assessment["run"]["per_cusp"])
    campaign = pe.assess_campaign(results_root=tmp_path, log=QUIET)
    assert campaign["verdicts"]["see-bn+xe-set-v2"] == "inconclusive" and campaign["additivity"]["statement"] == "not_evaluable"
