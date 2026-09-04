"""pic2d_full_physics_v1: the preregistered R4 / R5 / full-physics campaign on the 33 um reference plateau.

* composition contract: each sealed case protocol is the ss-v4 protocol with exactly the declared changes (the effect blocks, the v2.0.6 Debye floor, K = 5,
  the v2.1.1 arming latch + the ignition gate, the budget, the acceptance / reference / text blocks); geometry, grid, dt, W, seed, cadences and the v2.0.3 gate
  thresholds are byte-for-byte v4's; the six identities differ from v4's and from each other; the runner builds the declared configs; the spatial cases carry
  the MCC ceiling above the Knudsen anode density and no 0-D inventory; the F pair differs by time_acceleration only;
* the sealed files equal their recomposition and the campaign's listed hashes (the launch's refusal path);
* the reference block equals the ss-v4 artifacts on disk (incl. the ionisation centroid) and states its corrected-ledger (b) FAIL;
* the shakedown protocol shrinks cadences only;
* the per-case assessment classifies synthetic outcomes (plateau_clean / plateau_heating / no_plateau / EXTINGUISHED; the sustain reading; the reported-only n_g of the
  spatial cases; the sign rows of an alpha case judged against the alpha = 0 record); the campaign assessment yields the sustain table, the alpha trend, the F
  qualification and the additivity statement (not_evaluable without the physics-effects records; additive / interacting with them);
* the launch discipline (lock, dirty worktree, wrong commit, drifted protocol, --require-mps);
* a tiny CPU end-to-end run of the shrunk full-physics-alpha1over16 case through the shared runner -> finalize -> assess (every effect live in the runner path).
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
from experiments.pic2d_anomalous_transport_v1 import protocol as at
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_full_physics_v1 import protocol as pm
from experiments.pic2d_full_physics_v1 import run as fp
from experiments.pic2d_physics_effects_v1 import protocol as pe_protocol

V4_CONFIG_SHA256_CUDA = "f10772b25b030e2ba9d86cbd48b9972b9e143cce689e0b0ee8d4d6f10ccc685f"
UNCHANGED_TOP_LEVEL = ("geometry", "design_id", "field_authority")
V4_HAS_RESULTS = (pm.V4_RESULTS / "maps.npz").is_file() and (pm.V4_RESULTS / "summary.json").is_file()


def QUIET(_: str) -> None:  # noqa: N802 - the silent log sink of the assessments
    return None


def test_case_protocols_are_the_v4_template_with_exactly_the_declared_changes():
    v4 = pm.load_v4_protocol()
    base = runner.build_config(v4, backend="warp-cuda")
    assert artifacts.config_identity(base) == V4_CONFIG_SHA256_CUDA
    identities = {}
    for case, meta in pm.CASES.items():
        p = pm.load_case_protocol(case)
        assert p["campaign"]["case"] == case and p["campaign"]["effects"] == meta["effects"] and p["experiment_id"] == f"{pm.EXPERIMENT_ID}-{case}"
        assert p["campaign"]["alpha"] == meta["alpha"] and p["campaign"]["time_acceleration"] == meta["time_acceleration"]
        for key in UNCHANGED_TOP_LEVEL:
            assert p[key] == v4[key], key
        assert {k: v for k, v in p["case"].items() if k not in ("id", "seed_note")} == {k: v for k, v in v4["case"].items() if k not in ("id", "seed_note")}
        op, v4op = p["operating_point"], v4["operating_point"]
        if meta["neutrals"]:
            assert "neutral_inventory" not in op and op["neutrals"]["model"] == "neutrals_spatial_v1" and op["neutrals"]["metastables"]["model"] == "metastables_v1"
            assert op["neutrals"]["time_acceleration"] == meta["time_acceleration"] and op["neutrals"]["feed_atoms_per_s"] == v4op["neutral_inventory"]["feed_atoms_per_s"]
            assert op["neutral_density_per_m3"] == pm.MCC_CEILING_SPATIAL_PER_M3 > pm.KNUDSEN_ANODE_DENSITY_PER_M3
            unchanged = {k: v for k, v in op.items() if k not in ("collision_set", "neutrals", "neutral_density_per_m3", "neutral_density_role", "neutral_model_note")}
            assert unchanged == {k: v for k, v in v4op.items() if k not in ("neutral_inventory", "neutral_density_per_m3", "neutral_density_role")}
        else:
            assert {k: v for k, v in op.items() if k != "collision_set"} == v4op
        assert ("collision_set" in op) is meta["collision_set"]
        assert (p["cross_sections"] == v4["cross_sections"]) is (not meta["collision_set"])
        num, v4num = p["numerics"], v4["numerics"]
        assert ("anomalous_collisions" in num) is (meta["alpha"] > 0)
        assert ("see" in num) is meta["see"] and ("coulomb" in num) is meta["coulomb"]
        assert num["performance"]["moment_sample_interval"] == 5
        assert num["peak_debye_gate"]["min_accumulated_macro_particle_steps_at_peak"] == 64000
        for key in ("dt_s", "ion_subcycle", "device_sync_steps", "series_interval_steps", "averaging_window_steps", "checkpoint_every_steps", "stability_limits",
                    "stability_reference", "step_graph", "frame_recorder", "poisson", "deposition"):
            assert num[key] == v4num[key], key
        gate_keys = {k for k in num["peak_debye_gate"] if not k.startswith("min_accumulated")}
        assert {k: num["peak_debye_gate"][k] for k in gate_keys} == v4num["peak_debye_gate"]
        stop, v4stop = p["stopping_rule"], v4["stopping_rule"]
        for key in ("plateau", "plateau_threshold", "plateau_window_fraction", "min_transit_times"):
            assert stop[key] == v4stop[key], key
        triad = {k: v for k, v in stop["grid_heating_triad"].items() if k not in ("note", "drift_members_arming")}
        assert triad == {k: v for k, v in v4stop["grid_heating_triad"].items() if k != "note"}
        # the alpha-series amendment's arming latch and ignition gate, numerically identical
        arming = stop["grid_heating_triad"]["drift_members_arming"]
        assert {k: arming[k] for k in ("min_transit_times", "settle_quantity", "settle_drift_max", "settle_check_cadence_steps")} == \
            {k: at.DRIFT_MEMBERS_ARMING[k] for k in ("min_transit_times", "settle_quantity", "settle_drift_max", "settle_check_cadence_steps")}
        assert {k: stop["ignition_gate"][k] for k in ("reference_window_s", "check_window_s", "checks")} == {k: at.IGNITION_GATE[k] for k in ("reference_window_s", "check_window_s", "checks")}
        config = runner.build_config(p, backend="warp-cuda")
        assert config.moment_sample_interval == 5 and config.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak == 64000 and config.peak_debye_gate.max_cells_per_debye == pi
        assert config.grid == base.grid and config.dt_s == base.dt_s and config.macro_weight == base.macro_weight and config.seed == base.seed == 20260903
        assert config.injection == base.injection and config.seed_plasma == base.seed_plasma and config.potentials == base.potentials
        assert (config.anomalous is not None) is (meta["alpha"] > 0)
        if meta["alpha"] > 0:
            assert config.anomalous.alpha == meta["alpha"] and config.anomalous.model == "bohm_perpendicular_rotation"
        assert (config.see is not None) is meta["see"] and (config.coulomb is not None) is meta["coulomb"]
        if meta["coulomb"]:
            assert config.coulomb.electron_electron and config.coulomb.electron_ion and not config.coulomb.ion_ion and config.coulomb.cycle_steps == 10
        if meta["neutrals"]:
            assert config.neutral_inventory is None and config.neutrals_spatial is not None and config.neutrals_spatial.metastables is not None
            assert config.neutrals_spatial.time_acceleration == meta["time_acceleration"] and config.neutrals_spatial.wall_recycling
            assert config.mcc.neutral_density_per_m3 == pm.MCC_CEILING_SPATIAL_PER_M3 and config.neutrals_spatial.macro_weight == pm.NEUTRAL_MACRO_WEIGHT
            assert config.neutrals_spatial.substep_steps == 200 and config.neutrals_spatial.substep_steps % config.moment_sample_interval == 0
        else:
            assert config.neutrals_spatial is None and config.neutral_inventory == base.neutral_inventory and config.mcc.neutral_density_per_m3 == base.mcc.neutral_density_per_m3
        if meta["collision_set"]:
            assert config.mcc.collision_set is not None and config.mcc.collision_set.to_dict()["name"] == "xe_collision_set_v2" and config.mcc.collision_set.ion_neutral is not None
        else:
            assert config.mcc == base.mcc
        identities[case] = artifacts.config_identity(config)
        runner.frame_recorder_config(p).validate_alignment(sync_steps=200, checkpoint_every_steps=40_000, window_steps=400_000)
        ref = p["reference_run"]
        assert ref["corrected_ledger"]["acceptance_b_below_0p02"] is False and ref["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"] == pytest.approx(0.0246, abs=1e-4)
        acc = stop["acceptance"]
        assert acc["c_shifts"]["hypotheses"] == pm.HYPOTHESES_BY_CASE[case] and acc["c_shifts"]["key_quantities"] == list(pm.KEY_QUANTITIES[case])
        assert acc["c_shifts"]["reported_only"] == (["neutral_density_per_m3"] if meta["neutrals"] else [])
        assert set(acc["d_verdict"]["plateau_status"]) == {"plateau_clean", "plateau_heating", "no_plateau", "extinguished"}
        assert set(acc["d_verdict"]["per_case_hypothesis_verdict"]) == {"confirmed", "not_confirmed", "inconclusive"}
        assert acc["e_sustain"]["applies_to"] == list(pm.FULL_PHYSICS_CASES) and acc["f_qualification"]["applies_to"] == list(pm.F_PAIR) and "g_additivity" in acc
    assert len(set(identities.values())) == 6 and V4_CONFIG_SHA256_CUDA not in identities.values()
    assert pm.LAUNCH_PRIORITY == ("full-physics-alpha0.345", "full-physics-alpha0", "neutrals-spatial", "full-physics-alpha1over16", "coulomb", "neutrals-spatial-F10")
    assert set(pm.LAUNCH_PRIORITY) == set(pm.CASES)
    # the F pair: identical but for time_acceleration (and the texts that name it)
    a, b = pm.load_case_protocol("neutrals-spatial"), pm.load_case_protocol("neutrals-spatial-F10")
    ca, cb = runner.build_config(a, backend="cpu").to_dict(), runner.build_config(b, backend="cpu").to_dict()
    assert {k for k in ca if ca[k] != cb[k]} == {"neutrals_spatial"}
    da, db = dict(ca["neutrals_spatial"]), dict(cb["neutrals_spatial"])
    assert da.pop("time_acceleration") == 1.0 and db.pop("time_acceleration") == 10.0 and da == db
    # the shared blocks are the physics-effects campaign's (SEE, collision set) byte for byte
    full = pm.load_case_protocol("full-physics-alpha0")
    assert {k: v for k, v in full["numerics"]["see"].items() if k != "see_note"} == pe_protocol.SEE_BN_BLOCK
    assert {k: v for k, v in full["operating_point"]["collision_set"].items() if k != "collision_set_note"} == pe_protocol.COLLISION_SET_V2_BLOCK
    assert {k: v for k, v in full["numerics"]["coulomb"].items() if k != "coulomb_note"} == pm.COULOMB_BLOCK


def test_sealed_files_equal_their_recomposition_and_the_campaign_hashes(monkeypatch):
    campaign = pm.load_campaign()
    for case in pm.CASES:
        sealed = fp.verify_sealed(case)
        key = f"modern/experiments/pic2d_full_physics_v1/protocols/{case}.json"
        on_disk = (pm.PROTOCOLS_DIR / f"{case}.json").read_bytes()
        assert on_disk == canonical_bytes(sealed) + b"\n" and campaign["sealed_protocols"][key] == pm.protocol_sha256(sealed)
        assert campaign["acceptance"][case] == sealed["stopping_rule"]["acceptance"]
    assert campaign["design"]["coulomb_block"] == pm.COULOMB_BLOCK and campaign["design"]["see_block"] == pe_protocol.SEE_BN_BLOCK
    assert campaign["design"]["mcc_ceiling_spatial_per_m3"] > campaign["design"]["knudsen_anode_density_per_m3"]
    assert campaign["amendments"] == [] and campaign["design"]["launch_priority"] == list(pm.LAUNCH_PRIORITY)
    tampered = copy.deepcopy(pm.load_case_protocol("full-physics-alpha0"))
    tampered["operating_point"]["neutrals"]["time_acceleration"] = 2.0
    monkeypatch.setattr(fp, "load_case_protocol", lambda case: tampered)
    with pytest.raises(PIC2DValidationError, match="differs from its recomposition"):
        fp.verify_sealed("full-physics-alpha0")


@pytest.mark.skipif(not V4_HAS_RESULTS, reason="ss-v4 artifacts not checked out")
def test_reference_block_equals_the_v4_artifacts_and_states_the_corrected_ledger_fail():
    p = pm.load_case_protocol("full-physics-alpha0")
    grid = runner.build_config(p, backend="cpu").grid
    recomputed = fp.reference_quantities_from_files(pm.V4_RESULTS, grid)
    for key in pm.QUANTITY_KEYS:
        pinned = p["reference_run"]["quantities"].get(key)
        if recomputed[key] is None:
            assert key in ("neutral_density_anode_over_exit", "neutral_depletion_fraction", "metastable_fraction_of_ground", "stepwise_fraction_of_ionization", "nu_e_spitzer_peak_over_nu_en"), key
            continue
        assert pinned == pytest.approx(recomputed[key], rel=1e-12), key
    assert 0.012 < p["reference_run"]["quantities"]["ionization_centroid_z_m"] < 0.017
    detail = p["reference_run"]["ionization_centroid_detail"]
    assert detail["total_weighted"] == pytest.approx(p["reference_run"]["quantities"]["ionization_rate_per_s"], rel=0.02)     # the node-volume-weighted map sum is S
    sidecar = json.loads((pm.V4_RESULTS / "ledger-corrected.json").read_text(encoding="utf-8"))
    assert sidecar["end_state_window"]["corrected_ratio"] == p["reference_run"]["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"]
    q = fp.run_quantities(pm.V4_RESULTS, grid)
    assert q["stop_class"] == "plateau" and q["sustain"]["late_extinction_diagnostic"]["late_extinction"] is False
    assert q["sustain"]["probes"][2]["time_s"] == 1.0e-6 and q["sustain"]["probes"][2]["electron_ratio"] > 1.3       # the ss-v4 calibration point of the ignition gate
    assert "neutrals" not in q and "coulomb" not in q and "see" not in q and len(q["per_cusp"]) == 3


def test_shakedown_protocol_shrinks_cadences_only():
    protocol = pm.load_case_protocol("full-physics-alpha0.345")
    shake = fp.shakedown_protocol(protocol)
    a = runner.build_config(protocol, backend="cpu")
    b = runner.build_config(shake, backend="cpu")
    assert a.grid == b.grid and a.dt_s == b.dt_s and a.macro_weight == b.macro_weight and a.see == b.see and a.mcc == b.mcc and a.coulomb == b.coulomb
    assert a.neutrals_spatial == b.neutrals_spatial and a.anomalous == b.anomalous and a.moment_sample_interval == b.moment_sample_interval
    assert b.peak_debye_gate.max_cells_per_debye == pi and b.peak_debye_gate.window_steps == 40_000
    assert shake["numerics"]["checkpoint_every_steps"] == 4_000 and shake["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] == 40_000
    assert shake["stopping_rule"]["ignition_gate"] == protocol["stopping_rule"]["ignition_gate"] and shake["status"].startswith("SHAKEDOWN")


def _fake_results(root: Path, protocol: dict, *, scale: dict[str, float] | None = None, stop: str = "plateau_reached_after_min_transit_times", windowed: float | None = 0.01,
                  complete: bool = True, name: str | None = None, series_shape: str = "sustained", ignition_failed: bool = False, gate_message: str | None = None) -> Path:
    """summary.json / maps.npz / series.npz of a synthetic terminal state: the plateau scalars scaled from the reference, the effect blocks the case declares."""

    scale = scale or {}
    meta = pm.CASES[protocol["campaign"]["case"]]
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
    frac = ref["iedf_low_energy_fraction"] + scale.get("iedf_shift", 0.0)
    counts[10] = 1000.0 * frac
    counts[80] = 1000.0 * (1.0 - frac)
    ion = np.zeros(grid.node_shape)
    j_c = int(round((0.0145 + scale.get("centroid_shift_m", 0.0)) / grid.dz_m))
    ion[10, j_c] = 1.0
    maps = {"n_e_per_m3": n, "t_e_ev": t, "window_steps": np.array([400_000]), "iedf_ion_counts": counts, "iedf_edges_ev": edges, "ionization_rate_per_m3_s": ion}
    if meta["neutrals"]:
        gas = np.linspace(5.5e20, 7e19, grid.cell_shape[1])[None, :] * np.ones((grid.cell_shape[0], 1))
        maps["neutral_density_per_m3"] = gas
        maps["metastable_density_per_m3"] = 0.003 * gas
    if meta["coulomb"]:
        maps["coulomb_nu_ee_per_s"] = np.full(grid.node_shape, 3e6)
        maps["coulomb_nu_ei_per_s"] = np.full(grid.node_shape, 5e6)
        maps["coulomb_electron_seconds"] = np.ones(grid.node_shape)
    artifacts.write_npz(results / "maps.npz", maps)
    s_rate = ref["ionization_rate_per_s"] * scale.get("ionization_rate_per_s", 1.0)
    currents = {"discharge_a": ref["discharge_current_a"] * scale.get("discharge_current_a", 1.0), "exit_ion_beam_a": ref["exit_ion_beam_a"] * scale.get("exit_ion_beam_a", 1.0),
                "wall_electron_a": 3e-3, "wall_ion_a": 3e-3, "anode_ion_a": ref["anode_ion_a"] * scale.get("anode_ion_a", 1.0)}
    cumulative = {"ionizations": 1e6, "exit_ions": 5e5}
    if meta["see"]:
        currents.update({"see_emission_a": 2e-3, "see_effective_yield": 0.7})
        cumulative.update({"see_impacts": 4e6, "see_electrons": 2.8e6, "ke_see_emitted_j": 1e-7})
    if meta["collision_set"]:
        currents.update({"cex_rate_per_s": 2e15, "mex_rate_per_s": 1e15, "fast_neutral_exit_rate_per_s": 3e14, "fast_neutral_wall_rate_per_s": 1.5e15, "fast_neutral_thermal_rate_per_s": 2e14})
        cumulative.update({"cex": 5e4, "mex": 2.5e4, "excitations_level_1": 100.0, "excitations_level_2": 90.0, "excitations_level_3": 180.0, "excitations_level_4": 80.0})
    if meta["coulomb"]:
        cumulative.update({"coulomb_ee_pairs": 3e9, "coulomb_ei_pairs": 5e9, "coulomb_cycles": 5e5, "pz_coulomb": 1e-29, "ke_coulomb_j": 1e-16})
    if meta["neutrals"]:
        cumulative.update({"neutral_substeps": 26000, "neutral_ionized": 1e12, "stepwise_ionizations": 3e4, "meta_ionized": 3e4})
    if meta["alpha"] > 0:
        cumulative["anomalous"] = 1e11
    inventory = {"trailing_20pct_mean_ionization_rate_per_s": s_rate, "propellant_utilisation_trailing": ref["gross_utilisation"] * scale.get("gross_utilisation", 1.0),
                 "trailing_20pct_mean_density_per_m3": ref["neutral_density_per_m3"] * scale.get("neutral_density_per_m3", 1.0)}
    if meta["neutrals"]:
        inventory.update({"model": "neutrals_spatial_v1", "time_acceleration": meta["time_acceleration"], "final_channel_mean_density_per_m3": inventory["trailing_20pct_mean_density_per_m3"],
                          "final_axis_density_anode_per_m3": 5.5e20, "final_axis_density_exit_per_m3": 7e19, "gross_utilisation_trailing": inventory["propellant_utilisation_trailing"],
                          "net_utilisation_trailing": 0.8, "metastables": {"trailing_20pct_mean_fraction_of_ground": 0.003 * scale.get("metastable", 1.0),
                                                                             "trailing_20pct_mean_stepwise_fraction_of_ionization": 0.035 * scale.get("metastable", 1.0)}})
    summary = {
        "stop_reason": stop, "stability_gate_message": gate_message, "ion_transit_times": 3.05 if stop == "plateau_reached_after_min_transit_times" else 1.0, "steps_completed": 5_240_000,
        "simulated_time_s": 7.3e-6, "git_head": "deadbeef", "protocol_sha256": "0" * 64, "provenance": {"config_sha256": "1" * 64}, "maps_kind": "window_average", "sessions": [{}],
        "averaging_window_steps": 400_000, "final_series": {"time_s": 7.3e-6, "currents_a": dict(currents), "ledger": {"cumulative": cumulative}}, "window_currents_a": currents,
        "neutral_inventory": inventory,
        "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": windowed, "windowed_energy_residual_window_complete": complete, "energy_residual_over_electrode_work": 0.01,
                               "drift_members_arming": {"armed": stop == "plateau_reached_after_min_transit_times"}},
        "peak_node_debye": {"window": {"cells_per_debye_window_last": 1.6, "trailing_20pct_mean_cells_per_debye_window": 1.6, "soft_ok": True}},
        "plateau": {"reached": stop == "plateau_reached_after_min_transit_times"},
        "ignition": {"failed": ignition_failed, "checks": [{"time_s": 1e-6, "evaluated": True, "passed": not ignition_failed}, {"time_s": 2e-6, "evaluated": not ignition_failed, "passed": not ignition_failed}]},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    # a series: sustained (growing N_e) or decaying (an extinction the latch never armed on)
    time_s = np.linspace(0.0, 7.3e-6, 400)
    if series_shape == "sustained":
        electrons = 5e5 * (1.0 + 2.0 * (1.0 - np.exp(-time_s / 2e-6)))
        i_d = np.full(time_s.shape, currents["discharge_a"])
    else:
        electrons = 6e5 * np.exp(-time_s / 0.9e-6) + 1e3
        i_d = 3.1e-3 * np.exp(-time_s / 0.9e-6) + 1e-6
    series = {"time_s": time_s, "step": (time_s / 1.4e-12).astype(np.int64), "electrons": electrons, "current_discharge_a": i_d,
              "current_ionization_rate_per_s": np.full(time_s.shape, s_rate) * (electrons / electrons[0])}
    if meta["neutrals"]:
        series["neutral_density_per_m3"] = np.linspace(2.5e20, 2.5e20 * (1.0 - 0.003 * meta["time_acceleration"]), time_s.size)
        series["neutral_ceiling_violation_fraction"] = np.full(time_s.shape, 4e-5)
        series["neutral_axis_density_anode_per_m3"] = np.full(time_s.shape, 5.5e20)
        series["neutral_axis_density_exit_per_m3"] = np.full(time_s.shape, 7e19)
    if meta["coulomb"]:
        series["coulomb_nu_ee_mean_per_s"] = np.full(time_s.shape, 3e6)
        series["coulomb_nu_e_spitzer_peak_over_nu_en"] = np.full(time_s.shape, 0.25)
        series["coulomb_nu_e_spitzer_peak_per_s"] = np.full(time_s.shape, 3e6)
    artifacts.write_npz(results / "series.npz", series)
    return results


def test_case_assessment_classifies_plateaus_extinctions_and_the_shift_table(tmp_path: Path):
    p_c = pm.load_case_protocol("coulomb")
    # the audit's Coulomb expectation -> confirmed; I_d unchanged ('0' hypothesis confirming inside the band)
    ok = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c, scale={"ionization_rate_per_s": 1.12, "gross_utilisation": 1.12}), protocol=p_c, log=QUIET, reference_check=False)
    rows = ok["c_shifts_vs_reference"]
    assert ok["plateau_status"] == "plateau_clean" and ok["hypothesis_verdict"] == "confirmed" and ok["stop_class"] == "plateau"
    assert rows["ionization_rate_per_s"]["status"] == "confirming" and rows["discharge_current_a"]["status"] == "confirming" and rows["discharge_current_a"]["hypothesis_sign"] == "0"
    assert rows["nu_e_spitzer_peak_over_nu_en"]["kind"] == "absolute_from_zero_reference" and rows["nu_e_spitzer_peak_over_nu_en"]["status"] == "reported"
    assert ok["run"]["coulomb"]["trailing_20pct_means"]["nu_e_spitzer_peak_over_nu_en"] == pytest.approx(0.25) and ok["run"]["coulomb"]["maps"]["peak_cell"]["nu_ee_pair_mean_per_s"] == 3e6
    assert ok["sustain"]["applies"] is False and ok["extinguished"] is False
    # S down beyond the band -> not_confirmed; identical -> inconclusive (inside the band)
    down = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c, scale={"ionization_rate_per_s": 0.9}), protocol=p_c, log=QUIET, reference_check=False)
    assert down["hypothesis_verdict"] == "not_confirmed" and down["c_shifts_vs_reference"]["ionization_rate_per_s"]["status"] == "contradicting"
    same = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c), protocol=p_c, log=QUIET, reference_check=False)
    assert same["hypothesis_verdict"] == "inconclusive" and same["plateau_status"] == "plateau_clean"
    # heating / budget / Debye-gate stops
    heat = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c, windowed=0.03, scale={"ionization_rate_per_s": 1.12, "gross_utilisation": 1.12}), protocol=p_c, log=QUIET, reference_check=False)
    assert heat["plateau_status"] == "plateau_heating" and heat["hypothesis_verdict"] == "confirmed"
    budget = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c, stop="wall_clock_budget_reached"), protocol=p_c, log=QUIET, reference_check=False)
    assert budget["plateau_status"] == "no_plateau" and budget["stop_class"] == "budget" and budget["hypothesis_verdict"] == "inconclusive"
    debye = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c, stop="runtime_stability_gate_stopped_run", gate_message="peak-node Debye gate: 3.2 cells per lambda_D"),
                           protocol=p_c, log=QUIET, reference_check=False)
    assert debye["plateau_status"] == "no_plateau" and debye["stop_class"] == "peak_debye_gate"
    triad = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c, stop="grid_heating_triad_gate_stopped_run", gate_message="grid-heating triad gate: windowed energy residual / electrode work 0.06"),
                           protocol=p_c, log=QUIET, reference_check=False)
    assert triad["stop_class"] == "residual_power"
    # the spatial case: n_g reported by construction; centroid / IEDF absolute bands; the R5 signs
    p_r5 = pm.load_case_protocol("neutrals-spatial")
    r5 = fp.assess_case("neutrals-spatial", results=_fake_results(tmp_path, p_r5, scale={"ionization_rate_per_s": 2.5, "gross_utilisation": 2.5, "discharge_current_a": 1.6, "peak_n_e_window_per_m3": 1.5,
                                                                                          "t_e_peak_window_ev": 0.8, "neutral_density_per_m3": 7.8, "iedf_shift": 0.2, "centroid_shift_m": -0.002}),
                        protocol=p_r5, log=QUIET, reference_check=False)
    rows = r5["c_shifts_vs_reference"]
    assert r5["hypothesis_verdict"] == "confirmed" and rows["neutral_density_per_m3"]["status"] == "reported" and rows["neutral_density_per_m3"]["shift"] == pytest.approx(6.8)
    assert rows["ionization_centroid_z_m"]["kind"] == "absolute" and rows["ionization_centroid_z_m"]["status"] == "confirming" and rows["ionization_centroid_z_m"]["shift"] < -1e-3
    assert rows["iedf_low_energy_fraction"]["status"] == "confirming" and rows["metastable_fraction_of_ground"]["status"] == "reported"
    gas = r5["run"]["neutrals"]
    assert gas["depletion_fraction"] == pytest.approx(0.9 * 0.003, rel=0.02) and gas["anode_over_exit_axis_window"] == pytest.approx(5.5e20 / 7e19) and len(gas["cusp_plane_density"]) == 3
    assert gas["metastables"]["trailing_20pct_mean_fraction_of_ground"] == pytest.approx(0.003) and r5["run"]["stepwise_fraction_of_ionization"] == pytest.approx(0.035)
    # the alpha cases: extinguished (ignition gate) -> not_confirmed; sustained plateau -> confirmed; a late decay the latch never armed on -> extinguished
    p_a = pm.load_case_protocol("full-physics-alpha0.345")
    ext = fp.assess_case("full-physics-alpha0.345", results=_fake_results(tmp_path, p_a, stop="no_ignition", ignition_failed=True, series_shape="decaying"), protocol=p_a, log=QUIET, reference_check=False)
    assert ext["plateau_status"] == "extinguished" and ext["stop_class"] == "no_ignition" and ext["hypothesis_verdict"] == "not_confirmed" and ext["sustain"]["reading"] == "extinguished"
    late = fp.assess_case("full-physics-alpha0.345", results=_fake_results(tmp_path, p_a, stop="wall_clock_budget_reached", series_shape="decaying"), protocol=p_a, log=QUIET, reference_check=False)
    assert late["plateau_status"] == "extinguished" and late["sustain"]["diagnostic"]["late_extinction_diagnostic"]["late_extinction"] is True
    sus = fp.assess_case("full-physics-alpha0.345", results=_fake_results(tmp_path, p_a, scale={"discharge_current_a": 2.0}), protocol=p_a, log=QUIET, reference_check=False)
    assert sus["plateau_status"] == "plateau_clean" and sus["sustain"]["reading"] == "sustains" and sus["hypothesis_verdict"] == "confirmed"
    assert sus["c_shifts_vs_full_physics_alpha0"] is None and all(v["status"] in ("reported", "unavailable") for v in sus["c_shifts_vs_reference"].values()) is False
    undecided = fp.assess_case("full-physics-alpha0.345", results=_fake_results(tmp_path, p_a, stop="wall_clock_budget_reached"), protocol=p_a, log=QUIET, reference_check=False)
    assert undecided["plateau_status"] == "no_plateau" and undecided["sustain"]["reading"] == "undecided" and undecided["hypothesis_verdict"] == "inconclusive"
    # with an alpha = 0 record the sign rows are judged against it: I_d DOWN with alpha contradicts -> not_confirmed even though it sustains
    p_0 = pm.load_case_protocol("full-physics-alpha0")
    root = tmp_path / "campaign"
    alpha0 = _fake_results(root, p_0, scale={"discharge_current_a": 1.8, "ionization_rate_per_s": 2.5}, name="full-physics-alpha0")
    fp.assess_case("full-physics-alpha0", results=alpha0, protocol=p_0, log=QUIET, reference_check=False)
    judged = fp.assess_case("full-physics-alpha0.345", results=_fake_results(root, p_a, scale={"discharge_current_a": 1.2, "ionization_rate_per_s": 2.0}, name="full-physics-alpha0.345"),
                            protocol=p_a, log=QUIET, reference_check=False)
    assert judged["alpha0_record"] is not None and judged["c_shifts_vs_full_physics_alpha0"]["discharge_current_a"]["status"] == "contradicting"
    assert judged["sustain"]["reading"] == "sustains" and judged["hypothesis_verdict"] == "not_confirmed"
    record = artifacts.read_canonical_json(alpha0 / "assessment.json")
    assert record["schema_version"] == fp.ASSESSMENT_SCHEMA and record["group"] == "full_physics" and record["alpha"] == 0.0
    if V4_HAS_RESULTS:
        checked = fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c), protocol=p_c, log=QUIET, reference_check=True)
        assert all(v["agree"] for v in checked["reference_consistency"].values()) and checked["per_cusp_vs_reference"] is None      # the fixture has no wall maps
        bad = copy.deepcopy(p_c)
        bad["reference_run"]["quantities"]["discharge_current_a"] *= 1.01
        with pytest.raises(PIC2DValidationError, match="ss-v4 artifacts"):
            fp.assess_case("coulomb", results=_fake_results(tmp_path, p_c), protocol=bad, log=QUIET, reference_check=True)


def test_campaign_assessment_sustain_table_alpha_trend_f_qualification_and_additivity(tmp_path: Path, monkeypatch):
    root = tmp_path / "results"

    def build(by_case: dict[str, dict | None]) -> dict:
        if root.exists():
            shutil.rmtree(root)
        for case, spec in by_case.items():
            if spec is None:
                continue
            p = pm.load_case_protocol(case)
            results = _fake_results(root, p, scale=spec.get("scale"), stop=spec.get("stop", "plateau_reached_after_min_transit_times"), name=case,
                                    series_shape=spec.get("series", "sustained"), ignition_failed=spec.get("ignition_failed", False))
            fp.assess_case(case, results=results, protocol=p, log=QUIET, reference_check=False)
        return fp.assess_campaign(results_root=root, log=QUIET)

    empty = fp.assess_campaign(results_root=tmp_path / "empty", log=QUIET)
    assert empty["cases_reached"] == [] and empty["additivity"]["statement"] == "not_evaluable" and empty["f_qualification"]["statement"] == "not_evaluable"
    assert all(v["reading"] == "pending" for v in empty["sustain"]["table"].values()) and empty["alpha_trend"]["statement"] == "inconclusive"
    r5 = {"scale": {"ionization_rate_per_s": 2.5, "gross_utilisation": 2.5, "discharge_current_a": 1.6, "peak_n_e_window_per_m3": 1.5, "t_e_peak_window_ev": 0.8, "neutral_density_per_m3": 7.8}}
    coul = {"scale": {"ionization_rate_per_s": 1.12, "gross_utilisation": 1.12}}
    full0 = {"scale": {"ionization_rate_per_s": 2.8, "gross_utilisation": 2.8, "discharge_current_a": 2.0, "peak_n_e_window_per_m3": 1.4, "t_e_peak_window_ev": 0.7, "neutral_density_per_m3": 7.8}}
    full16 = {"scale": {"ionization_rate_per_s": 2.4, "gross_utilisation": 2.4, "discharge_current_a": 2.4, "peak_n_e_window_per_m3": 1.2, "t_e_peak_window_ev": 0.65, "neutral_density_per_m3": 7.8}}
    full345 = {"scale": {"ionization_rate_per_s": 2.0, "gross_utilisation": 2.0, "discharge_current_a": 2.9, "peak_n_e_window_per_m3": 1.0, "t_e_peak_window_ev": 0.6, "neutral_density_per_m3": 7.8}}
    # F = 10 inside the band -> qualified; every alpha sustains and the trend is monotone -> trend_confirmed; additivity needs the physics-effects records
    all_in = build({"coulomb": coul, "neutrals-spatial": r5, "neutrals-spatial-F10": {"scale": dict(r5["scale"], ionization_rate_per_s=2.55)}, "full-physics-alpha0": full0,
                    "full-physics-alpha1over16": full16, "full-physics-alpha0.345": full345})
    assert set(all_in["cases_reached"]) == set(pm.CASES)
    assert all_in["f_qualification"]["statement"] == "F_qualified" and all_in["f_qualification"]["rows"]["ionization_rate_per_s"]["relative_difference"] == pytest.approx(0.02)
    assert all_in["sustain"]["statements"]["full-physics-alpha0.345"].endswith("YES") and all_in["sustain"]["table"]["full-physics-alpha1over16"]["reading"] == "sustains"
    assert all_in["alpha_trend"]["statement"] == "trend_confirmed" and all_in["alpha_trend"]["points_reached"] == ["full-physics-alpha0", "full-physics-alpha1over16", "full-physics-alpha0.345"]
    assert all_in["additivity"]["statement"] == "not_evaluable" and all_in["additivity"]["parts_reached"]["pe:xe-set-v2"] is False
    assert all_in["additivity"]["r5_operating_point_dominance"]["ionization_rate_per_s"]["shift_r5_incl_set_v2"] == pytest.approx(1.5)
    assert all_in["verdicts"]["full-physics-alpha0.345"] == "confirmed" and all_in["verdicts"]["coulomb"] == "confirmed"
    # F = 10 moves S beyond the band -> disqualified; alpha 0.345 extinguished -> its statement NO, trend inconclusive (2 points)
    mixed = build({"coulomb": coul, "neutrals-spatial": r5, "neutrals-spatial-F10": {"scale": dict(r5["scale"], ionization_rate_per_s=2.9)}, "full-physics-alpha0": full0,
                   "full-physics-alpha1over16": full16, "full-physics-alpha0.345": {"stop": "no_ignition", "ignition_failed": True, "series": "decaying"}})
    assert mixed["f_qualification"]["statement"] == "F_disqualified" and mixed["f_qualification"]["outside_band"] == ["ionization_rate_per_s"]
    assert mixed["sustain"]["statements"]["full-physics-alpha0.345"].endswith("NO (extinguished)") and mixed["plateau_status"]["full-physics-alpha0.345"] == "extinguished"
    assert mixed["alpha_trend"]["statement"] == "inconclusive" and mixed["verdicts"]["full-physics-alpha0.345"] == "not_confirmed"
    # a non-monotone I_d with three points -> trend_not_confirmed
    wrong = build({"full-physics-alpha0": full0, "full-physics-alpha1over16": {"scale": dict(full16["scale"], discharge_current_a=1.5)}, "full-physics-alpha0.345": full345})
    assert wrong["alpha_trend"]["statement"] == "trend_not_confirmed" and wrong["alpha_trend"]["rows"]["discharge_current_a"]["monotone"] is False
    # with physics-effects records present (fake, in a temp dir) the additivity is evaluated: exact sum -> additive; an excess on I_d -> interacting / super_additive
    pe_root = tmp_path / "pe"
    for case, shifts in (("see-bn+xe-set-v2", {"discharge_current_a": 0.2, "ionization_rate_per_s": 0.0, "gross_utilisation": 0.0, "exit_ion_beam_a": 0.0, "neutral_density_per_m3": 0.0,
                                               "peak_n_e_window_per_m3": -0.1, "t_e_peak_window_ev": -0.15, "iedf_low_energy_fraction": 0.2}),
                         ("xe-set-v2", {"discharge_current_a": 0.0, "ionization_rate_per_s": -0.04, "gross_utilisation": -0.04, "exit_ion_beam_a": 0.0, "neutral_density_per_m3": 0.0,
                                        "peak_n_e_window_per_m3": 0.0, "t_e_peak_window_ev": -0.04, "iedf_low_energy_fraction": 0.2})):
        (pe_root / case).mkdir(parents=True)
        artifacts.write_canonical_json(pe_root / case / "assessment.json", {"a_plateau": {"passed": True}, "c_shifts_vs_reference": {k: {"shift": v} for k, v in shifts.items()}})
    monkeypatch.setattr(fp, "PE_RESULTS", pe_root)
    # combined I_d shift = 0.2 (see+xe) + 0.0 (coulomb) + (0.6 - 0.0) (R5) = 0.8 -> exactly additive
    exact = build({"coulomb": {"scale": {"ionization_rate_per_s": 1.12, "gross_utilisation": 1.12}}, "neutrals-spatial": r5,
                   "full-physics-alpha0": {"scale": {"discharge_current_a": 1.8, "ionization_rate_per_s": 1.0 + 0.0 + 0.12 + (1.5 + 0.04), "gross_utilisation": 1.0 + 0.12 + 1.54,
                                                     "peak_n_e_window_per_m3": 0.9 + 0.5, "t_e_peak_window_ev": 1.0 - 0.15 - 0.2 + 0.04, "neutral_density_per_m3": 7.8}}})
    assert exact["additivity"]["statement"] == "additive" and exact["additivity"]["rows"]["discharge_current_a"]["interaction"] == pytest.approx(0.0, abs=1e-9)
    assert exact["additivity"]["rows"]["neutral_density_per_m3"]["classification"] == "reported" and exact["additivity"]["r5_operating_point_dominance"]["ionization_rate_per_s"]["operating_point_dominates"] is True
    more = build({"coulomb": coul, "neutrals-spatial": r5,
                  "full-physics-alpha0": {"scale": {"discharge_current_a": 2.3, "ionization_rate_per_s": 2.66, "gross_utilisation": 2.66, "peak_n_e_window_per_m3": 1.4, "t_e_peak_window_ev": 0.69, "neutral_density_per_m3": 7.8}}})
    assert more["additivity"]["statement"] == "interacting" and more["additivity"]["rows"]["discharge_current_a"]["classification"] == "super_additive"
    record = artifacts.read_canonical_json(root / "campaign-assessment.json")
    assert record["schema_version"] == fp.CAMPAIGN_ASSESSMENT_SCHEMA and record["launch_priority"] == list(pm.LAUNCH_PRIORITY)


def test_launch_discipline_lock_dirty_worktree_commit_protocol_and_mps(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fp, "git", lambda *args, cwd=None: "f" * 40 if args[:1] == ("rev-parse",) else "")
    with pytest.raises(PIC2DValidationError, match="unknown case"):
        fp.launch("full", results=tmp_path / "x", log=QUIET)
    with pytest.raises(PIC2DValidationError, match="not the preregistration commit"):
        fp.launch("coulomb", results=tmp_path / "launch", expect_commit="0123456", log=QUIET)
    monkeypatch.setattr(fp, "worktree_status", lambda cwd=None: [" M modern/x.py"])
    with pytest.raises(PIC2DValidationError, match="not clean"):
        fp.launch("coulomb", results=tmp_path / "launch", expect_commit="fffffff", log=QUIET)
    monkeypatch.setattr(fp, "worktree_status", lambda cwd=None: [])
    monkeypatch.setattr(fp, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else ("blob-a" if args[0] == "rev-parse" else "blob-b"))
    with pytest.raises(PIC2DValidationError, match="differs from the committed blob"):
        fp.launch("coulomb", results=tmp_path / "launch", log=QUIET)
    monkeypatch.setattr(fp, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else "same")
    monkeypatch.setattr(fp, "preflight_path", lambda case: tmp_path / f"preflight-{case}.json")
    monkeypatch.setattr(fp, "shakedown_path", lambda case: tmp_path / f"shakedown-{case}.json")
    with pytest.raises(PIC2DValidationError, match="preflight-coulomb.json and shakedown-coulomb.json"):
        fp.launch("coulomb", results=tmp_path / "launch", log=QUIET)
    (tmp_path / "preflight-coulomb.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shakedown-coulomb.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(PIC2DValidationError, match="--require-mps"):
        fp.launch("coulomb", results=tmp_path / "launch", require_mps=True, log=QUIET)
    with pytest.raises(PIC2DValidationError, match="--resume needs"):
        fp.launch("coulomb", results=tmp_path / "launch", resume=True, log=QUIET)
    assert not (tmp_path / "launch" / fp.LOCK_NAME).exists()


def _tiny_case_protocol(case: str) -> dict:
    """The sealed case shrunk to a 12 x 96 grid / large W (plasma AND neutral macro weights) for a CPU smoke of the runner path (NOT the protocol; a test fixture)."""

    p = fp.shakedown_protocol(pm.load_case_protocol(case))
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
    if "neutrals" in p["operating_point"]:
        p["operating_point"]["neutrals"]["substep_steps"] = 20
        p["operating_point"]["neutrals"]["macro_weight"] = 2.2e10       # ~4000 macro-neutrals on the 12 x 96 grid
    p["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] = 200
    p["stopping_rule"]["wall_budget_seconds"] = 900.0
    return p


def test_tiny_cpu_run_of_the_shrunk_full_physics_alpha_case_through_the_runner_finalize_and_assess(tmp_path: Path):
    p = _tiny_case_protocol("full-physics-alpha1over16")
    results = tmp_path / "full-physics-alpha1over16"
    results.mkdir()
    protocol_path = tmp_path / "protocol-tiny.json"
    artifacts.write_canonical_json(protocol_path, p)
    summary_path = runner.run_steady_state(p, results, backend="cpu", max_steps=400, protocol_path=protocol_path, log=QUIET)
    summary = artifacts.read_canonical_json(summary_path)
    cumulative = summary["final_series"]["ledger"]["cumulative"]
    # the 12 x 96 / W 3e6 fixture is not physics: the plumbing of every effect through the runner is under test
    assert summary["steps_completed"] >= 200
    cfg = summary["provenance"]["config"]
    assert cfg["see"]["material"] == "BN" and cfg["mcc"]["collision_set"]["name"] == "xe_collision_set_v2" and cfg["coulomb"]["cycle_steps"] == 10
    assert cfg["neutrals_spatial"]["model"] == "neutrals_spatial_v1" and cfg["neutrals_spatial"]["time_acceleration"] == 1.0 and cfg["anomalous"]["alpha"] == pytest.approx(1 / 16)
    for key in ("see_impacts", "cex", "excitations_level_1", "coulomb_ee_pairs", "anomalous", "neutral_substeps"):
        assert key in cumulative, key
    assert cumulative["neutral_substeps"] > 0 and cumulative["coulomb_ee_pairs"] > 0
    assert summary["neutral_inventory"]["model"] == "neutrals_spatial_v1" and summary["ignition"] is not None
    # reference_check=True (when the v4 artifacts are checked out) builds the per-cusp rows WITH the SEE / Coulomb / gas readings against the v4 cusps - the path the
    # first box shakedown broke on (column_frequency_profile keys its planes by name, not by index)
    assessment = fp.assess_case("full-physics-alpha1over16", results=results, protocol=p, log=QUIET, reference_check=V4_HAS_RESULTS)
    assert assessment["plateau_status"] in ("no_plateau", "extinguished") and assessment["hypothesis_verdict"] in ("inconclusive", "not_confirmed")
    run = assessment["run"]
    assert run["see"] is not None and run["collision_set"] is not None and run["coulomb"] is not None and run["neutrals"] is not None and run["anomalous"] is not None
    assert run["per_cusp"] is not None and len(run["per_cusp"]) == 3 and run["neutrals"]["axis_density_profile_per_m3"] and run["ionization_centroid_detail"] is not None
    if V4_HAS_RESULTS:
        rows = assessment["per_cusp_vs_reference"]
        assert rows is not None and len(rows) == 3
        assert all("see" in r and "neutral_gas" in r and "coulomb" in r for r in rows)
        assert all(r["coulomb"]["nu_ee_pair_mean_per_s"] is not None and r["coulomb"]["nu_e_spitzer_per_s"] is not None for r in rows)
        assert all(r["neutral_gas"]["column_mean_per_m3"] > 0 for r in rows)         # (the axis cell of a 4000-macro-neutral fixture may be empty)
    # --reuse-run (record stages on an existing run without re-stepping) refuses a run whose stored shakedown protocol is not the sealed case's (the tiny fixture)
    with pytest.raises(PIC2DValidationError, match="reuse-run"):
        fp.shakedown("full-physics-alpha1over16", results=results, backend="cpu", reuse_run=True, log=QUIET)
    with pytest.raises(PIC2DValidationError, match="needs a completed run"):
        fp.shakedown("full-physics-alpha1over16", results=tmp_path / "nothing", backend="cpu", reuse_run=True, log=QUIET)
    assert run["sustain"]["probes"] == [] or run["sustain"]["probes"][0]["time_s"] == 0.1e-6
    campaign = fp.assess_campaign(results_root=tmp_path, log=QUIET)
    assert campaign["sustain"]["table"]["full-physics-alpha1over16"]["reading"] in ("undecided", "extinguished") and campaign["additivity"]["statement"] == "not_evaluable"
