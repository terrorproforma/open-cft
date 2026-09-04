"""pic2d_cft_steady_state_v4_fast: the preregistered solver-qualification replay of the accepted v4 33.3 um plateau.

* identity contract: the configuration differs from the v4 run's (f10772b2...) by EXACTLY two keys - ``poisson`` (device-mg, 14 cycles,
  2+2 sweeps, omega 0.8, coarsest 1024) and ``moment_sample_interval`` (5); everything else in ``PIC2DConfig.to_dict()`` is equal;
* the protocol carries the v4 grid / dt / W / seed / operating point / closure / gates / cadences / plateau rule verbatim;
* the acceptance schema: (a)-(e), the seed-b-band tolerances (derived from the recorded bands and below the W x0.7 band), the four verdicts;
* the pinned reference equals the v4 artifacts on disk (summary + maps AND the v4 assessment's run block);
* the shakedown protocol shrinks cadences only (solver and K untouched);
* the pinned reference equals the v4 artifacts (summary + maps, the v4 assessment's run block AND the v2.0.6 ledger-corrected sidecar);
* synthetic outcomes classify into qualified / not_qualified / heating / no_plateau ((b) = a two-sided 1 pp band around the v4 plateau's
  v2.0.6-corrected +2.46 % residual power), (d) reads the recorded solver, a contract miss without a
  terminal state is classified from the runner log, an inconsistent reference is refused;
* the launch discipline: lock, dirty worktree, wrong commit, drifted protocol, --require-mps, wrong configuration.
"""

from __future__ import annotations

import copy
import json
from math import pi
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.frames import FrameRecorderConfig
from cft_revival.pic2d.models import PIC2DValidationError
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v4 import run as v4
from experiments.pic2d_cft_steady_state_v4_fast import run as fast

V4_CONFIG_SHA256_CUDA = "f10772b25b030e2ba9d86cbd48b9972b9e143cce689e0b0ee8d4d6f10ccc685f"
# the v4 plateau's v2.0.6 post-hoc corrected windowed residual (results/ledger-corrected.json, 02013df0): +2.46 %; the run RECORDED -7.67 %
V4_CORRECTED_WINDOWED = 0.02458578453535502
# pinned at the preregistration: any change to protocol.json that touches the configuration must be a new experiment
FAST_CONFIG_SHA256_CUDA = "a6275830aee1857d61d1af431185c68d370afcfce09ec6dbc2ea60d6e721aa51"
SEED_B_BAND = {"discharge_current_a": 0.009, "exit_ion_beam_a": 0.0068, "ionization_rate_per_s": 0.0080, "gross_utilisation": 0.0080,
               "neutral_density_per_m3": 0.0073, "peak_n_e_window_per_m3": 0.0819, "t_e_peak_window_ev": 0.011}
W07_BAND = {"discharge_current_a": 0.0568, "exit_ion_beam_a": 0.0355, "ionization_rate_per_s": 0.0464, "gross_utilisation": 0.0464,
            "neutral_density_per_m3": 0.0395, "peak_n_e_window_per_m3": 0.1189, "t_e_peak_window_ev": 0.093}


def _quiet(_: str) -> None:
    return None


def _dict_diff(a: dict, b: dict) -> set[str]:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_identity_differs_from_v4_by_exactly_the_solver_and_the_sampling_interval():
    protocol = fast.load_protocol()
    v4p = fast.load_v4_protocol()
    config = runner.build_config(protocol, backend="warp-cuda")
    base = runner.build_config(v4p, backend="warp-cuda")
    a, b = config.to_dict(), base.to_dict()
    assert _dict_diff(a, b) == {"poisson", "moment_sample_interval"}
    assert "moment_sample_interval" not in b and a["moment_sample_interval"] == 5 == config.moment_sample_interval
    assert b["poisson"]["method"] == "device-direct" and "multigrid" not in b["poisson"]
    assert a["poisson"]["method"] == "device-mg" and a["poisson"]["multigrid"] == {"cycles": 14, "pre_sweeps": 2, "post_sweeps": 2, "omega": 0.8, "coarsest_max_unknowns": 1024}
    assert a["poisson"]["relative_tolerance"] == b["poisson"]["relative_tolerance"] == 1e-10     # the SAME contract
    assert _dict_diff(a["poisson"], b["poisson"]) == {"method", "multigrid"}
    assert artifacts.config_identity(base) == V4_CONFIG_SHA256_CUDA
    assert artifacts.config_identity(config) == FAST_CONFIG_SHA256_CUDA != V4_CONFIG_SHA256_CUDA
    # the protocol file declares exactly these two changes
    assert protocol["numerics"]["poisson"] == {"method": "device-mg", "cycles": 14, "pre_sweeps": 2, "post_sweeps": 2, "omega": 0.8, "coarsest_max_unknowns": 1024}
    assert protocol["numerics"]["performance"] == {"moment_sample_interval": 5}
    assert isinstance(v4p["numerics"]["poisson"], str) and "performance" not in v4p["numerics"]
    # and the CPU backend selects the multigrid too (the same cycles in numpy)
    assert runner.build_config(protocol, backend="cpu").poisson.method == "device-mg"


def test_protocol_is_the_v4_protocol_verbatim_outside_the_declared_fields():
    protocol = fast.load_protocol()
    v4p = fast.load_v4_protocol()
    # blocks copied verbatim
    for key in ("geometry", "operating_point", "design_id", "field_authority", "cross_sections"):
        assert protocol[key] == v4p[key], key
    for key in ("radial_cells", "axial_cells", "macro_weight", "seed"):
        assert protocol["case"][key] == v4p["case"][key], key
    assert protocol["case"]["seed"] == 20260903 and protocol["case"]["replay_of"]["run_git_head"] == "392129e5"
    num, vnum = protocol["numerics"], v4p["numerics"]
    for key in vnum:
        if key not in ("poisson", "poisson_note"):
            assert num[key] == vnum[key], key
    assert set(num) - set(vnum) == {"performance", "performance_note", "poisson_note"}
    # the SAME stopping rule, gates and plateau preconditions
    rule, vrule = protocol["stopping_rule"], v4p["stopping_rule"]
    for key in ("plateau", "plateau_threshold", "plateau_window_fraction", "min_transit_times", "grid_heating_triad"):
        assert rule[key] == vrule[key], key
    assert rule["wall_budget_seconds"] > 0 and "PLACEHOLDER" not in rule["wall_budget_note"]
    assert "PLACEHOLDER" not in json.dumps(protocol["budget_v1_3"]["cost_model"])
    assert runner.protocol_budget(protocol)["ion_transit_time_s"] == 2.4e-6
    config = runner.build_config(protocol, backend="warp-cuda")
    gate = config.peak_debye_gate
    assert gate.windowed and gate.max_cells_per_debye == pi and gate.soft_cells_per_debye == 2.5 and gate.window_steps == 400_000
    frames = runner.frame_recorder_config(protocol)
    assert isinstance(frames, FrameRecorderConfig) and frames.cadence_steps == 20_000
    frames.validate_alignment(sync_steps=200, checkpoint_every_steps=40_000, window_steps=400_000)
    assert config.grid.cell_shape == (90, 720) and config.dt_s == 1.4e-12 and config.macro_weight == 26666.7
    assert protocol["status"].startswith("preregistered_solver_qualification")
    assert protocol["experiment_id"] == "pic2d-cft-steady-state-v4-fast"


def test_acceptance_schema_seed_b_band_tolerances_and_verdicts():
    protocol = fast.load_protocol()
    acceptance = protocol["stopping_rule"]["acceptance"]
    assert set(acceptance) == {"declared", "a_plateau", "b_residual_power", "c_replay_tolerances", "d_field_solve_contract", "e_verdict"}
    assert set(acceptance["e_verdict"]) == set(fast.VERDICTS) == {"qualified", "not_qualified", "heating", "no_plateau"}
    tolerances = {k: v for k, v in acceptance["c_replay_tolerances"].items() if k != "note"}
    assert tolerances == {"discharge_current_a": 0.02, "exit_ion_beam_a": 0.02, "ionization_rate_per_s": 0.02, "gross_utilisation": 0.02,
                          "neutral_density_per_m3": 0.02, "peak_n_e_window_per_m3": 0.10, "t_e_peak_window_ev": 0.03}
    assert set(tolerances) == set(fast.JUDGED)
    bands = protocol["reference_run"]["bands"]
    for key, tolerance in tolerances.items():
        # derived from the recorded bands: above the seed-b band (>= 1.2x), below the W x0.7 band, and far below the v4 convergence tolerances
        assert abs(bands["seed_b"][key]) == pytest.approx(SEED_B_BAND[key] if key != "discharge_current_a" else 0.0008, abs=1e-6)
        assert abs(bands["w_0_7"][key]) == pytest.approx(W07_BAND[key], abs=1e-6)
        assert tolerance >= 1.2 * SEED_B_BAND[key], key
        assert tolerance < W07_BAND[key], key
        assert tolerance < {"peak_n_e_window_per_m3": 0.2, "t_e_peak_window_ev": 0.2}.get(key, 0.1), key
    assert bands["seed_b"]["discharge_current_trailing_20pct"] == pytest.approx(-0.009)
    # the (b) bound is the v4 one and the (d) rule names the contract and the crash classification
    assert "<= 0.01" in acceptance["b_residual_power"] and "+0.0246" in acceptance["b_residual_power"] and "two-sided" in acceptance["b_residual_power"]
    assert "1e-10" in acceptance["d_field_solve_contract"] and "not_qualified" in acceptance["d_field_solve_contract"]
    assert fast.B_BAND == 0.01 and protocol["reference_run"]["quantities"][fast.V4_CORRECTED_KEY] == V4_CORRECTED_WINDOWED
    assert protocol["reference_run"]["quantities"]["windowed_residual_over_electrode_work_last"] == pytest.approx(-0.0767, abs=1e-4)
    assert protocol["reference_run"]["ledger_corrected_sidecar"]["acceptance_b_below_0p02"] == {"corrected_passes": False, "declared_in_protocol": True, "recorded_passes": True}
    assert "device-mg" in acceptance["e_verdict"]["qualified"] and "plume" in acceptance["e_verdict"]["qualified"]
    # the reference IS the v4 run: same quantities the v4 assessment recorded
    ref = protocol["reference_run"]
    assert ref["experiment"].endswith("pic2d_cft_steady_state_v4") and ref["commit"] == "0d228ad2" and ref["run_git_head"] == "392129e5"
    assert ref["quantities"]["discharge_current_a"] == pytest.approx(3.801e-3, rel=1e-3) and ref["quantities"]["peak_node"] == [20, 429]
    assert ref["assessment"]["verdict"] == "resolution_limited" and ref["assessment"]["run_config_sha256"] == V4_CONFIG_SHA256_CUDA
    assert "secondary_reference_run" not in protocol


@pytest.mark.skipif(not (fast.REFERENCE_RESULTS / "maps.npz").is_file(), reason="v4 artifacts not checked out")
def test_pinned_reference_equals_the_v4_artifacts_and_the_v4_assessment():
    protocol = fast.load_protocol()
    from_files = fast.reference_quantities_from_files()
    from_assessment = fast.reference_quantities_from_assessment()
    for key, value in from_files.items():
        assert protocol["reference_run"]["quantities"][key] == pytest.approx(value, rel=1e-12), key
        assert from_assessment[key] == pytest.approx(value, rel=1e-12), key
    assert from_assessment["verdict"] == "resolution_limited" and from_assessment["config_sha256"] == V4_CONFIG_SHA256_CUDA
    assert protocol["reference_run"]["quantities"]["windowed_residual_over_electrode_work_last"] == from_assessment["windowed_residual_over_electrode_work"]
    corrected = fast.reference_corrected_residual_from_sidecar()
    assert corrected is not None and corrected["windowed_corrected"] == V4_CORRECTED_WINDOWED == protocol["reference_run"]["quantities"][fast.V4_CORRECTED_KEY]
    assert corrected["windowed_recorded"] == pytest.approx(from_assessment["windowed_residual_over_electrode_work"], abs=1e-12) and corrected["window_complete"]
    assert corrected["step"] == 5_200_000 and corrected["cumulative_corrected"] == protocol["reference_run"]["quantities"]["cumulative_residual_over_electrode_work_corrected_v2_0_6"]
    assert protocol["reference_run"]["quantities"]["cells_per_debye_at_peak_window"] == pytest.approx(from_assessment["cells_per_debye_window_last"], rel=1e-12)
    assert fast._peak_from_maps(fast.REFERENCE_RESULTS / "maps.npz")["node"] == [20, 429]
    consistency = fast._consistency(protocol["reference_run"]["quantities"], fast.REFERENCE_RESULTS)
    assert consistency is not None and all(row["agree"] for row in consistency.values())
    assert fast.V4_CORRECTED_KEY in consistency and all("v4_assessment_run" in row for key, row in consistency.items() if key in fast.JUDGED)


def test_shakedown_protocol_shrinks_cadences_only():
    protocol = fast.load_protocol()
    shake = fast.shakedown_protocol(protocol)
    a = runner.build_config(protocol, backend="cpu")
    b = runner.build_config(shake, backend="cpu")
    assert a.grid == b.grid and a.dt_s == b.dt_s and a.macro_weight == b.macro_weight and a.seed_plasma == b.seed_plasma and a.seed == b.seed
    assert a.neutral_inventory == b.neutral_inventory and a.injection == b.injection and a.potentials == b.potentials
    assert a.poisson == b.poisson and b.poisson.method == "device-mg" and b.poisson.mg_cycles == 14
    assert a.moment_sample_interval == b.moment_sample_interval == 5 and b.sync_steps % 5 == 0
    assert b.peak_debye_gate.max_cells_per_debye == pi and b.peak_debye_gate.window_steps == 40_000 and b.peak_debye_gate.window_snapshot_steps == 4_000
    assert shake["numerics"]["checkpoint_every_steps"] == 4_000 and shake["numerics"]["averaging_window_steps"] == 40_000
    assert shake["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] == 40_000
    runner.frame_recorder_config(shake).validate_alignment(sync_steps=200, checkpoint_every_steps=4_000, window_steps=40_000)
    assert shake["status"].startswith("SHAKEDOWN") and shake["experiment_id"].endswith("-shakedown")


def _fake_results(tmp_path: Path, protocol: dict, *, scale: dict[str, float] | None = None, stop: str = "plateau_reached_after_min_transit_times",
                  windowed: float | None = V4_CORRECTED_WINDOWED, complete: bool = True, poisson: dict | None = None, k: int = 5) -> Path:
    """A results directory holding just what ``assess`` reads: summary.json + maps.npz with the v4 reference values scaled."""

    scale = scale or {}
    ref = protocol["reference_run"]["quantities"]
    results = tmp_path / f"results-{len(list(tmp_path.iterdir()))}"
    results.mkdir()
    grid = runner.build_config(protocol, backend="cpu").grid
    n = np.zeros(grid.node_shape)
    t = np.zeros(grid.node_shape)
    n[20, 429] = ref["peak_n_e_window_per_m3"] * scale.get("peak_n_e_window_per_m3", 1.0)
    t[20, 429] = ref["t_e_peak_window_ev"] * scale.get("t_e_peak_window_ev", 1.0)
    artifacts.write_npz(results / "maps.npz", {"n_e_per_m3": n, "t_e_ev": t, "window_steps": np.array([400_000])})
    declared = runner.build_config(protocol, backend="warp-cuda").poisson.to_dict()
    summary = {
        "stop_reason": stop, "ion_transit_times": 3.03, "steps_completed": 5_200_000, "git_head": "deadbeef", "protocol_sha256": "0" * 64,
        "provenance": {"config_sha256": "1" * 64, "config": {"poisson": declared if poisson is None else poisson} | ({} if k == 1 else {"moment_sample_interval": k})},
        "maps_kind": "window_average", "sessions": [{}], "ms_per_step_this_session": 9.0, "wall_seconds_total": 50_000.0,
        "window_currents_a": {"discharge_a": ref["discharge_current_a"] * scale.get("discharge_current_a", 1.0),
                              "exit_ion_beam_a": ref["exit_ion_beam_a"] * scale.get("exit_ion_beam_a", 1.0)},
        "neutral_inventory": {"trailing_20pct_mean_ionization_rate_per_s": ref["ionization_rate_per_s"] * scale.get("ionization_rate_per_s", 1.0),
                              "propellant_utilisation_trailing": ref["gross_utilisation"] * scale.get("gross_utilisation", 1.0),
                              "trailing_20pct_mean_density_per_m3": ref["neutral_density_per_m3"] * scale.get("neutral_density_per_m3", 1.0)},
        "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": windowed, "windowed_energy_residual_window_complete": complete,
                               "energy_residual_over_electrode_work": -0.09},
        "peak_node_debye": {"window": {"cells_per_debye_window_last": 2.15, "trailing_20pct_mean_cells_per_debye_window": 2.10, "soft_ok": True}},
        "plateau": {"reached": stop == "plateau_reached_after_min_transit_times"},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    return results


def test_assessment_classifies_synthetic_outcomes_into_the_four_verdicts(tmp_path: Path):
    protocol = fast.load_protocol()
    quiet = _quiet
    # identical to v4 -> qualified; every row within, bands recorded beside the tolerance, (d) read from the provenance
    same = fast.assess(protocol, _fake_results(tmp_path, protocol), log=quiet, reference_check=False)
    assert same["verdict"] == "qualified" and same["c_replay"]["all_within"] and same["d_field_solve_contract"]["passed"]
    assert all(v["relative_difference"] == 0.0 and v["seed_b_band"] is not None and v["w_0_7_band"] is not None for v in same["c_replay"]["quantities"].values())
    assert same["b_residual_power"]["delta_vs_v4_corrected"] == pytest.approx(0.0) and same["b_residual_power"]["v4_corrected_v2_0_6"] == V4_CORRECTED_WINDOWED
    assert same["b_residual_power"]["project_acceptance_b_below_0p02"] == {**same["b_residual_power"]["project_acceptance_b_below_0p02"], "replay": False, "v4_corrected": False}
    assert same["e_verdict"].startswith("(a) AND (b) AND every (c)") and same["schema_version"] == fast.ASSESSMENT_SCHEMA
    assert same["c_replay"]["reported_not_judged"]["cells_per_debye_window_last"]["relative_difference"] == pytest.approx(0.0, abs=2e-3)
    # inside the seed-b band on every quantity (a seed-b-like realisation) -> qualified
    seed_like = fast.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 1 - 0.009, "exit_ion_beam_a": 1.0068, "ionization_rate_per_s": 0.992,
                                                                                  "gross_utilisation": 0.992, "neutral_density_per_m3": 1.0073,
                                                                                  "peak_n_e_window_per_m3": 1 - 0.0819, "t_e_peak_window_ev": 0.989}),
                            log=quiet, reference_check=False)
    assert seed_like["verdict"] == "qualified"
    # a W x0.7-sized shift on any current / rate exceeds the band -> not_qualified (the run would have passed the v4 10 % tolerance)
    for key, factor in (("discharge_current_a", 1.0568), ("ionization_rate_per_s", 1 - 0.0464), ("neutral_density_per_m3", 1.0395),
                        ("peak_n_e_window_per_m3", 1 - 0.1189), ("t_e_peak_window_ev", 1 - 0.093), ("exit_ion_beam_a", 1.0355), ("gross_utilisation", 1.0464)):
        out = fast.assess(protocol, _fake_results(tmp_path, protocol, scale={key: factor}), log=quiet, reference_check=False)
        assert out["verdict"] == "not_qualified" and not out["c_replay"]["quantities"][key]["within"], key
        assert sum(not v["within"] for v in out["c_replay"]["quantities"].values()) == 1
    # exactly at the tolerance passes, just beyond fails (|rel| <= tol)
    assert fast.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 1.0199}), log=quiet, reference_check=False)["verdict"] == "qualified"
    assert fast.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 1.0201}), log=quiet, reference_check=False)["verdict"] == "not_qualified"
    assert fast.assess(protocol, _fake_results(tmp_path, protocol, scale={"peak_n_e_window_per_m3": 0.905}), log=quiet, reference_check=False)["verdict"] == "qualified"
    assert fast.assess(protocol, _fake_results(tmp_path, protocol, scale={"peak_n_e_window_per_m3": 0.895}), log=quiet, reference_check=False)["verdict"] == "not_qualified"
    assert fast.assess(protocol, _fake_results(tmp_path, protocol, scale={"t_e_peak_window_ev": 1.029}), log=quiet, reference_check=False)["verdict"] == "qualified"
    assert fast.assess(protocol, _fake_results(tmp_path, protocol, scale={"t_e_peak_window_ev": 1.031}), log=quiet, reference_check=False)["verdict"] == "not_qualified"
    # (b): two-sided +-1 pp band around the v4 CORRECTED +2.46 %: +3.4 % and +1.6 % pass, +3.6 % (heating side) and +1.3 % (cooling side) fail;
    # the v4-recorded -7.67 % (the pre-v2.0.6 biased value) would fail on the cooling side; an incomplete window fails
    for windowed, verdict in ((0.034, "qualified"), (0.016, "qualified"), (0.036, "heating"), (0.013, "heating"), (-0.0767, "heating"), (0.049, "heating")):
        out = fast.assess(protocol, _fake_results(tmp_path, protocol, windowed=windowed), log=quiet, reference_check=False)
        assert out["verdict"] == verdict and out["b_residual_power"]["passed"] == (verdict == "qualified"), windowed
        assert out["b_residual_power"]["side"] == ("heating" if windowed > V4_CORRECTED_WINDOWED else "cooling")
        assert out["b_residual_power"]["project_acceptance_b_below_0p02"]["replay"] == (windowed < 0.02)
    assert fast.assess(protocol, _fake_results(tmp_path, protocol, windowed=V4_CORRECTED_WINDOWED, complete=False), log=quiet, reference_check=False)["verdict"] == "heating"
    # (a): any other stop reason -> no_plateau
    for stop in ("wall_clock_budget_reached", "grid_heating_triad_gate_stopped_run", "runtime_stability_gate_stopped_run", "no_ignition"):
        budget = fast.assess(protocol, _fake_results(tmp_path, protocol, stop=stop), log=quiet, reference_check=False)
        assert budget["verdict"] == "no_plateau" and "inconclusive" in budget["e_verdict"], stop
    # (d): a run whose provenance names another solver configuration is not the declared one -> not_qualified even if every (c) row agrees
    other = fast.assess(protocol, _fake_results(tmp_path, protocol, poisson={"method": "device-direct", "relative_tolerance": 1e-10}), log=quiet, reference_check=False)
    assert other["verdict"] == "not_qualified" and not other["d_field_solve_contract"]["passed"] and other["c_replay"]["all_within"]
    fewer = fast.assess(protocol, _fake_results(tmp_path, protocol, poisson={"method": "device-mg", "multigrid": {"cycles": 12, "pre_sweeps": 2, "post_sweeps": 2, "omega": 0.8, "coarsest_max_unknowns": 1024}}),
                        log=quiet, reference_check=False)
    assert fewer["verdict"] == "not_qualified" and not fewer["d_field_solve_contract"]["passed"]
    record = artifacts.read_canonical_json(tmp_path / other["results_dir"] / "assessment.json")
    assert record["verdict"] == "not_qualified" and record["d_field_solve_contract"]["declared_poisson"]["multigrid"]["cycles"] == 14


def test_assessment_of_a_contract_miss_without_a_terminal_state_is_not_qualified(tmp_path: Path):
    protocol = fast.load_protocol()
    quiet = _quiet
    results = tmp_path / "results"
    results.mkdir()
    with pytest.raises(PIC2DValidationError, match="no summary.json"):
        fast.assess(protocol, results, log=quiet, reference_check=False)
    log_path = tmp_path / "run.log"
    log_path.write_text("[steady-state] step 812000\nsomething else\n", encoding="utf-8")
    with pytest.raises(PIC2DValidationError, match="carries no"):
        fast.assess(protocol, results, log=quiet, reference_check=False, runner_crash_log=log_path)
    log_path.write_text("[steady-state] step 812000\ncft_revival.pic2d.models.PIC2DConvergenceError: fixed-cycle multigrid solve failed its residual contract: "
                        "last true residual 2.8e-10 > 1.0e-10 ...\n", encoding="utf-8")
    out = fast.assess(protocol, results, log=quiet, reference_check=False, runner_crash_log=log_path)
    assert out["verdict"] == "not_qualified" and out["d_field_solve_contract"]["passed"] is False and out["d_field_solve_contract"]["terminal_state"] is False
    assert out["a_plateau"]["passed"] is False and out["c_replay"]["quantities"] is None and "residual contract" in out["d_field_solve_contract"]["evidence"][-1]
    assert (results / "assessment.json").is_file()


@pytest.mark.skipif(not (fast.REFERENCE_RESULTS / "maps.npz").is_file(), reason="v4 artifacts not checked out")
def test_assessment_refuses_an_inconsistent_reference(tmp_path: Path):
    protocol = fast.load_protocol()
    quiet = _quiet
    bad = copy.deepcopy(protocol)
    bad["reference_run"]["quantities"]["discharge_current_a"] *= 1.01
    with pytest.raises(PIC2DValidationError, match="v4 artifacts"):
        fast.assess(bad, _fake_results(tmp_path, protocol), log=quiet, reference_check=True)
    ok = fast.assess(protocol, _fake_results(tmp_path, protocol), log=quiet, reference_check=True)
    assert ok["verdict"] == "qualified" and all(v["agree"] for v in ok["reference_consistency"].values())


def test_launch_discipline_lock_dirty_worktree_commit_protocol_mps_and_configuration(tmp_path: Path, monkeypatch):
    protocol = fast.load_protocol()
    payload = {"schema_version": fast.LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "commit": "a" * 40, "protocol_sha256": "b" * 64}
    lock = fast.acquire_lock(tmp_path / "results", payload)
    assert lock.is_file() and json.loads(lock.read_text(encoding="utf-8"))["commit"] == "a" * 40
    with pytest.raises(PIC2DValidationError, match="same-attempt"):
        fast.acquire_lock(tmp_path / "results", payload)
    with pytest.raises(PIC2DValidationError, match="different-attempt"):
        fast.acquire_lock(tmp_path / "results", payload | {"commit": "c" * 40})
    quiet = _quiet
    monkeypatch.setattr(fast, "git", lambda *args, cwd=None: "f" * 40 if args[:1] == ("rev-parse",) else "")
    with pytest.raises(PIC2DValidationError, match="not the preregistration commit"):
        fast.launch(protocol, results=tmp_path / "launch", expect_commit="0123456", log=quiet)
    monkeypatch.setattr(fast, "worktree_status", lambda cwd=None: [" M modern/x.py"])
    with pytest.raises(PIC2DValidationError, match="not clean"):
        fast.launch(protocol, results=tmp_path / "launch", expect_commit="fffffff", log=quiet)
    assert not (tmp_path / "launch" / fast.LOCK_NAME).exists()
    monkeypatch.setattr(fast, "worktree_status", lambda cwd=None: [])
    monkeypatch.setattr(fast, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else "same")
    monkeypatch.setattr(fast, "PREFLIGHT_PATH", tmp_path / "preflight.json")
    monkeypatch.setattr(fast, "SHAKEDOWN_PATH", tmp_path / "shakedown.json")
    with pytest.raises(PIC2DValidationError, match="preflight.json and shakedown.json"):
        fast.launch(protocol, results=tmp_path / "launch", log=quiet)
    (tmp_path / "preflight.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shakedown.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(PIC2DValidationError, match="--require-mps"):
        fast.launch(protocol, results=tmp_path / "launch", require_mps=True, log=quiet)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "no-such-pipe"))
    with pytest.raises(PIC2DValidationError, match="--require-mps"):
        fast.launch(protocol, results=tmp_path / "launch", require_mps=True, log=quiet)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    with pytest.raises(PIC2DValidationError, match="--resume needs"):
        fast.launch(protocol, results=tmp_path / "launch", resume=True, require_mps=True, log=quiet)
    # only the declared fast configuration may be launched under this experiment
    v4_like = copy.deepcopy(protocol)
    v4_like["numerics"]["poisson"] = fast.load_v4_protocol()["numerics"]["poisson"]
    with pytest.raises(PIC2DValidationError, match="declared fast configuration"):
        fast.launch(v4_like, results=tmp_path / "launch", log=quiet)
    monkeypatch.setattr(fast, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else ("blob-a" if args[0] == "rev-parse" else "blob-b"))
    with pytest.raises(PIC2DValidationError, match="differs from the committed blob"):
        fast.launch(protocol, results=tmp_path / "launch", log=quiet)
    # telemetry never raises: nvidia-smi missing -> empty inventory, zero clients, None snapshot
    monkeypatch.setattr(fast.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no nvidia-smi")))
    assert fast.gpu_load_snapshot() is None and fast.gpu_inventory() == [] and fast.concurrent_mps_clients()["count"] == 0
    # the client filter: own pid and the MPS server (66 MiB) are excluded
    apps = [{"pid": 1, "used_memory_mib": 66.0}, {"pid": 2, "used_memory_mib": 1610.0}, {"pid": 3, "used_memory_mib": 1930.0}, {"pid": 4, "used_memory_mib": 1418.0}]
    assert fast.concurrent_mps_clients(apps, own_pid=3) == {**fast.concurrent_mps_clients(apps, own_pid=3), "count": 2, "pids": [2, 4]}


def test_v4_module_is_untouched_and_the_v5_and_v4_identities_still_pin():
    # the frozen v4 module is imported for helpers only: its own identity pin and reference are unchanged
    assert artifacts.config_identity(runner.build_config(v4.load_protocol(), backend="warp-cuda")) == V4_CONFIG_SHA256_CUDA
    assert v4.REFERENCE_RESULTS.name == "results" and v4.REFERENCE_RESULTS.parent.name == "pic2d_cft_steady_state_v2"
    assert fast.REFERENCE_RESULTS.parent.name == "pic2d_cft_steady_state_v4"
