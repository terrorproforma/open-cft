"""pic2d_cft_steady_state_v5: the preregistered 25 um / 1.0 ps / W 1.5e4 ladder point testing the 33.3 um (v4) plateau.

* protocol contract: the 25 um grid divides the geometry exactly, dt / W / gates as declared, the operating point and the v1.3
  closure are bit-for-bit the v2 base's and v4's, the configuration identity is pinned, the cadences keep the v4 step counts;
* the pinned PRIMARY reference quantities equal the v4 artifacts on disk and the SECONDARY ones the v2 base's;
* the shakedown protocol shrinks cadences only;
* the predeclared assessment classifies synthetic outcomes into the four declared verdicts against the 33 um reference, reports
  the 50 um column without judging it, and refuses an inconsistent reference (primary or secondary);
* the launch discipline: exclusive lock, dirty-worktree, wrong-commit and drifted-protocol refusals.
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
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import PIC2DValidationError
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v4 import run as v4
from experiments.pic2d_cft_steady_state_v5 import run as v5

MODERN = Path(__file__).resolve().parents[2]
V2_PROTOCOL = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "protocol.json"
# pinned at the preregistration: any change to protocol.json that touches the configuration must be a new experiment
V5_CONFIG_SHA256_CUDA = "efb9bb09c2d28fdfee02f7bb19f0fdd3ec12ff3a6b573659c7d17656bc9ad5b0"


def test_v5_protocol_is_the_v4_refinement_refined_by_4_3_per_axis_with_the_same_closure_and_gates():
    protocol = v5.load_protocol()
    v4p = v4.load_protocol()
    v2 = runner.load_protocol(V2_PROTOCOL)
    config = runner.build_config(protocol, backend="warp-cuda")
    fine = runner.build_config(v4p, backend="warp-cuda")
    base = runner.build_config(v2, backend="warp-cuda")
    assert config.grid.cell_shape == (120, 960) and fine.grid.cell_shape == (90, 720) and base.grid.cell_shape == (60, 480)
    assert config.grid.dr_m == pytest.approx(25e-6, rel=1e-12) and config.grid.dz_m == pytest.approx(25e-6, rel=1e-12)
    assert config.grid.dr_m == pytest.approx(base.grid.dr_m / 2, rel=1e-12) and config.grid.dr_m == pytest.approx(fine.grid.dr_m * 0.75, rel=1e-12)
    for value, spacing in ((2e-3, config.grid.dr_m), (18e-3, config.grid.dz_m), (3e-3, config.grid.dr_m), (24e-3, config.grid.dz_m)):
        assert abs(value / spacing - round(value / spacing)) < 1e-9
    assert config.dt_s == 1.0e-12 and fine.dt_s == 1.4e-12 and base.dt_s == 1.5e-12
    assert config.macro_weight == 15000.0 == pytest.approx(6e4 * (25 / 50) ** 2) and config.macro_weight == pytest.approx(fine.macro_weight / (4 / 3) ** 2, rel=2e-5)
    # same operating point, same v1.3 closure (no recycling), same seed, same injection as the base AND v4
    for other in (fine, base):
        assert config.potentials == other.potentials and config.injection == other.injection and config.seed_plasma == other.seed_plasma
        assert config.mcc == other.mcc and config.neutral_inventory == other.neutral_inventory and config.seed == other.seed == 20260903
    assert protocol["operating_point"] == {**v4p["operating_point"], "unchanged_note": protocol["operating_point"]["unchanged_note"]}
    assert not config.neutral_inventory.wall_recycling and config.cathode is None and config.plume_boundary_gate is None
    # v2.0.3 gates verbatim, cadences = the v4 step counts
    gate = config.peak_debye_gate
    assert gate == fine.peak_debye_gate and gate.windowed and gate.max_cells_per_debye == pi and gate.soft_cells_per_debye == 2.5
    assert gate.window_steps == 400_000 == protocol["numerics"]["averaging_window_steps"] == v4p["numerics"]["averaging_window_steps"]
    assert gate.window_snapshot_steps == 40_000 == protocol["numerics"]["checkpoint_every_steps"]
    assert protocol["numerics"]["series_interval_steps"] == protocol["numerics"]["device_sync_steps"] == 200
    triad = protocol["stopping_rule"]["grid_heating_triad"]
    assert {k: triad[k] for k in ("residual_window_steps", "windowed_energy_residual_over_electrode_work_max", "energy_residual_over_electrode_work_max",
                                  "enforced_after_transit_times", "soft_drift_max", "hard_drift_max")} == \
        {k: v4p["stopping_rule"]["grid_heating_triad"][k] for k in ("residual_window_steps", "windowed_energy_residual_over_electrode_work_max",
                                                                       "energy_residual_over_electrode_work_max", "enforced_after_transit_times", "soft_drift_max", "hard_drift_max")}
    rule = protocol["stopping_rule"]
    assert rule["plateau_threshold"] == 0.05 and rule["plateau_window_fraction"] == 0.2 and rule["min_transit_times"] == 3
    assert runner.protocol_budget(protocol)["ion_transit_time_s"] == 2.4e-6 and rule["wall_budget_seconds"] >= 24 * 3600
    assert "PLACEHOLDER" not in rule["wall_budget_note"]
    acceptance = rule["acceptance"]
    assert set(acceptance["d_reclassification"]) == {"converged", "resolution_limited", "refinement_heating", "no_plateau"}
    tolerances = {k: v for k, v in acceptance["c_convergence_tolerances"].items() if k != "note"}
    assert tolerances == {k: v for k, v in v4p["stopping_rule"]["acceptance"]["c_convergence_tolerances"].items() if k != "note"}
    assert set(protocol["reference_run"]["quantities"]) >= set(tolerances) and set(protocol["secondary_reference_run"]["quantities"]) >= set(tolerances)
    assert protocol["reference_run"]["experiment"].endswith("pic2d_cft_steady_state_v4") and protocol["secondary_reference_run"]["experiment"].endswith("pic2d_cft_steady_state_v2")
    # frames on and aligned; mesh statistics of the 25 um grid (the cone stair-step volume converges toward the exact 3.4558e-7 m^3)
    frames = runner.frame_recorder_config(protocol)
    assert isinstance(frames, FrameRecorderConfig) and frames.cadence_steps == 20_000
    frames.validate_alignment(sync_steps=200, checkpoint_every_steps=40_000, window_steps=400_000)
    record = build_mesh_masks(config.grid).to_dict()
    assert record["plasma_cells"] == 81480 and record["unknown_nodes"] == 82359
    assert record["plasma_volume_m3"] == pytest.approx(3.444e-7, rel=2e-3)
    # identity pinned and distinct from both rungs
    assert artifacts.config_identity(config) == V5_CONFIG_SHA256_CUDA
    assert len({artifacts.config_identity(c) for c in (config, fine, base)}) == 3
    # a-priori gate readings at the v4 and v2 peaks (25 um / lambda_D), inside the soft margin with room
    for block, expected in ((protocol["reference_run"]["quantities"], 1.615), (protocol["secondary_reference_run"]["quantities"], 1.583)):
        lam = np.sqrt(8.8541878128e-12 * block["t_e_peak_window_ev"] * 1.602176634e-19 / (block["peak_n_e_window_per_m3"] * 1.602176634e-19**2))
        assert config.grid.dz_m / lam == pytest.approx(expected, abs=0.01) and expected < gate.soft_cells_per_debye
    omega_pe = np.sqrt(protocol["reference_run"]["quantities"]["peak_n_e_window_per_m3"] * 1.602176634e-19**2 / (8.8541878128e-12 * 9.1093837015e-31))
    assert omega_pe * config.dt_s == pytest.approx(0.064, abs=0.002) and omega_pe * config.dt_s < 0.2
    assert protocol["budget_v1_3"]["expected_cells_per_debye_at_peak_window"] == pytest.approx(1.62, abs=0.01)


@pytest.mark.skipif(not (v5.REFERENCE_RESULTS / "maps.npz").is_file() or not (v5.SECONDARY_REFERENCE_RESULTS / "maps.npz").is_file(),
                    reason="v4 / v2 artifacts not checked out")
def test_pinned_reference_quantities_equal_the_v4_and_v2_artifacts():
    protocol = v5.load_protocol()
    primary = v5.reference_quantities_from_files(v5.REFERENCE_RESULTS)
    secondary = v5.reference_quantities_from_files(v5.SECONDARY_REFERENCE_RESULTS)
    for key, value in primary.items():
        assert protocol["reference_run"]["quantities"][key] == pytest.approx(value, rel=1e-12), key
    for key, value in secondary.items():
        assert protocol["secondary_reference_run"]["quantities"][key] == pytest.approx(value, rel=1e-12), key
    assert protocol["reference_run"]["quantities"]["peak_node"] == v5._peak_from_maps(v5.REFERENCE_RESULTS / "maps.npz")["node"] == [20, 429]
    assert protocol["secondary_reference_run"]["quantities"]["peak_node"] == [14, 286]
    # the primary reference IS the v4 run assessed resolution_limited against the secondary
    v4_assessment = json.loads((v5.REFERENCE_RESULTS / "assessment.json").read_text(encoding="utf-8"))
    assert v4_assessment["verdict"] == "resolution_limited" and protocol["secondary_reference_run"]["v4_verdict"].startswith("resolution_limited")
    assert protocol["reference_run"]["quantities"]["windowed_residual_over_electrode_work_last"] == v4_assessment["run"]["windowed_residual_over_electrode_work"]


def test_shakedown_protocol_shrinks_cadences_only():
    protocol = v5.load_protocol()
    shake = v5.shakedown_protocol(protocol)
    a = runner.build_config(protocol, backend="cpu")
    b = runner.build_config(shake, backend="cpu")
    assert a.grid == b.grid and a.dt_s == b.dt_s and a.macro_weight == b.macro_weight and a.seed_plasma == b.seed_plasma
    assert a.neutral_inventory == b.neutral_inventory and a.injection == b.injection and a.potentials == b.potentials
    assert b.peak_debye_gate.max_cells_per_debye == a.peak_debye_gate.max_cells_per_debye == pi
    assert b.peak_debye_gate.soft_cells_per_debye == 2.5 and b.peak_debye_gate.window_steps == 40_000 and b.peak_debye_gate.window_snapshot_steps == 4_000
    assert shake["numerics"]["checkpoint_every_steps"] == 4_000 and shake["numerics"]["averaging_window_steps"] == 40_000
    assert shake["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] == 40_000
    assert shake["stopping_rule"]["grid_heating_triad"]["windowed_energy_residual_over_electrode_work_max"] == 0.05
    runner.frame_recorder_config(shake).validate_alignment(sync_steps=200, checkpoint_every_steps=4_000, window_steps=40_000)
    assert shake["status"].startswith("SHAKEDOWN") and shake["experiment_id"].endswith("-shakedown")
    assert protocol["status"] == "preregistered_resolution_convergence_study_not_validated"


def _fake_results(tmp_path: Path, protocol: dict, *, scale: dict[str, float] | None = None, stop: str = "plateau_reached_after_min_transit_times",
                  windowed: float | None = -0.05, complete: bool = True) -> Path:
    """A results directory holding just what ``assess`` reads: summary.json + maps.npz with the PRIMARY (v4) reference values scaled."""

    scale = scale or {}
    ref = protocol["reference_run"]["quantities"]
    results = tmp_path / f"results-{len(list(tmp_path.iterdir()))}"
    results.mkdir()
    grid = runner.build_config(protocol, backend="cpu").grid
    n = np.zeros(grid.node_shape)
    t = np.zeros(grid.node_shape)
    n[27, 572] = ref["peak_n_e_window_per_m3"] * scale.get("peak_n_e_window_per_m3", 1.0)
    t[27, 572] = ref["t_e_peak_window_ev"] * scale.get("t_e_peak_window_ev", 1.0)
    artifacts.write_npz(results / "maps.npz", {"n_e_per_m3": n, "t_e_ev": t, "window_steps": np.array([400_000])})
    summary = {
        "stop_reason": stop, "ion_transit_times": 3.05, "steps_completed": 7_320_000, "git_head": "deadbeef", "protocol_sha256": "0" * 64,
        "provenance": {"config_sha256": "1" * 64}, "maps_kind": "window_average", "sessions": [{}],
        "window_currents_a": {"discharge_a": ref["discharge_current_a"] * scale.get("discharge_current_a", 1.0),
                              "exit_ion_beam_a": ref["exit_ion_beam_a"] * scale.get("exit_ion_beam_a", 1.0)},
        "neutral_inventory": {"trailing_20pct_mean_ionization_rate_per_s": ref["ionization_rate_per_s"] * scale.get("ionization_rate_per_s", 1.0),
                              "propellant_utilisation_trailing": ref["gross_utilisation"] * scale.get("gross_utilisation", 1.0),
                              "trailing_20pct_mean_density_per_m3": ref["neutral_density_per_m3"] * scale.get("neutral_density_per_m3", 1.0)},
        "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": windowed, "windowed_energy_residual_window_complete": complete,
                               "energy_residual_over_electrode_work": -0.06},
        "peak_node_debye": {"window": {"cells_per_debye_window_last": 1.6, "trailing_20pct_mean_cells_per_debye_window": 1.6, "soft_ok": True}},
        "plateau": {"reached": stop == "plateau_reached_after_min_transit_times"},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    return results


def test_assessment_judges_against_33um_reports_50um_and_refuses_an_inconsistent_reference(tmp_path: Path):
    protocol = v5.load_protocol()
    quiet = lambda _: None
    # identical to v4 -> converged vs 33 um; the 50 um column then shows v4's own shifts (I_d +10.35 %) and is NOT judged
    same = v5.assess(protocol, _fake_results(tmp_path, protocol), log=quiet, reference_check=False)
    assert same["verdict"] == "converged" and same["c_convergence"]["all_within"] and all(v["relative_difference"] == 0.0 for v in same["c_convergence"]["quantities"].values())
    secondary = same["secondary_comparison"]
    assert secondary["all_within"] is False and secondary["quantities"]["discharge_current_a"]["relative_difference"] == pytest.approx(0.10354, abs=1e-4)
    assert secondary["quantities"]["peak_n_e_window_per_m3"]["relative_difference"] == pytest.approx(-0.2142, abs=1e-3) and "not judged" in secondary["reference"]
    assert same["d_reclassification"].startswith("(a) AND (b) AND") and "ladder terminates" in same["d_reclassification"]
    # within the tolerances vs 33 um -> converged even if far from the 50 um base
    converged = v5.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 1.08, "peak_n_e_window_per_m3": 0.85, "t_e_peak_window_ev": 1.15}),
                          log=quiet, reference_check=False)
    assert converged["verdict"] == "converged" and converged["c_convergence"]["quantities"]["discharge_current_a"]["relative_difference"] == pytest.approx(0.08)
    assert not converged["secondary_comparison"]["all_within"]
    # one tolerance exceeded vs 33 um -> the 33 um plateau is also resolution-limited
    limited = v5.assess(protocol, _fake_results(tmp_path, protocol, scale={"peak_n_e_window_per_m3": 0.75}), log=quiet, reference_check=False)
    assert limited["verdict"] == "resolution_limited" and not limited["c_convergence"]["quantities"]["peak_n_e_window_per_m3"]["within"]
    assert "ALSO" in limited["d_reclassification"] and "W-only" in limited["d_reclassification"]
    # a run that agrees with the 50 um base but not with 33 um is resolution_limited: the primary reference decides
    back_to_base = v5.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 1 / 1.10354}), log=quiet, reference_check=False)
    assert back_to_base["verdict"] == "converged"    # -9.4 % vs 33 um is inside 10 %
    back_to_base = v5.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 0.88}), log=quiet, reference_check=False)
    assert back_to_base["verdict"] == "resolution_limited"
    heating = v5.assess(protocol, _fake_results(tmp_path, protocol, windowed=0.03), log=quiet, reference_check=False)
    assert heating["verdict"] == "refinement_heating" and not heating["b_residual_power"]["passed"]
    cooling = v5.assess(protocol, _fake_results(tmp_path, protocol, windowed=-0.09), log=quiet, reference_check=False)
    assert cooling["verdict"] == "converged" and cooling["b_residual_power"]["passed"]          # one-sided: the negative side is reported
    incomplete = v5.assess(protocol, _fake_results(tmp_path, protocol, windowed=0.0, complete=False), log=quiet, reference_check=False)
    assert incomplete["verdict"] == "refinement_heating"
    budget = v5.assess(protocol, _fake_results(tmp_path, protocol, stop="wall_clock_budget_reached"), log=quiet, reference_check=False)
    assert budget["verdict"] == "no_plateau" and "inconclusive" in budget["d_reclassification"]
    record = artifacts.read_canonical_json(tmp_path / budget["results_dir"] / "assessment.json")
    assert record["verdict"] == "no_plateau" and record["schema_version"] == v5.ASSESSMENT_SCHEMA and record["secondary_comparison"]["v4_verdict_against_it"].startswith("resolution_limited")
    # references that disagree with the artifacts on disk are refused (fail closed), primary and secondary
    if (v5.REFERENCE_RESULTS / "maps.npz").is_file() and (v5.SECONDARY_REFERENCE_RESULTS / "maps.npz").is_file():
        bad = copy.deepcopy(protocol)
        bad["reference_run"]["quantities"]["discharge_current_a"] *= 1.01
        with pytest.raises(PIC2DValidationError, match="v4 artifacts"):
            v5.assess(bad, _fake_results(tmp_path, protocol), log=quiet, reference_check=True)
        bad = copy.deepcopy(protocol)
        bad["secondary_reference_run"]["quantities"]["neutral_density_per_m3"] *= 1.01
        with pytest.raises(PIC2DValidationError, match="v2 artifacts"):
            v5.assess(bad, _fake_results(tmp_path, protocol), log=quiet, reference_check=True)
        ok = v5.assess(protocol, _fake_results(tmp_path, protocol), log=quiet, reference_check=True)
        assert all(v["agree"] for v in ok["reference_consistency"].values()) and all(v["agree"] for v in ok["secondary_comparison"]["reference_consistency"].values())


def test_launch_discipline_lock_dirty_worktree_and_commit(tmp_path: Path, monkeypatch):
    protocol = v5.load_protocol()
    payload = {"schema_version": v5.LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "commit": "a" * 40, "protocol_sha256": "b" * 64}
    lock = v5.acquire_lock(tmp_path / "results", payload)
    assert lock.is_file() and json.loads(lock.read_text(encoding="utf-8"))["commit"] == "a" * 40
    with pytest.raises(PIC2DValidationError, match="same-attempt"):
        v5.acquire_lock(tmp_path / "results", payload)
    with pytest.raises(PIC2DValidationError, match="different-attempt"):
        v5.acquire_lock(tmp_path / "results", payload | {"commit": "c" * 40})
    monkeypatch.setattr(v5, "git", lambda *args, cwd=None: "f" * 40 if args[:1] == ("rev-parse",) else "")
    with pytest.raises(PIC2DValidationError, match="not the preregistration commit"):
        v5.launch(protocol, results=tmp_path / "launch", expect_commit="0123456", log=lambda _: None)
    monkeypatch.setattr(v5, "worktree_status", lambda cwd=None: [" M modern/x.py"])
    with pytest.raises(PIC2DValidationError, match="not clean"):
        v5.launch(protocol, results=tmp_path / "launch", expect_commit="fffffff", log=lambda _: None)
    assert not (tmp_path / "launch" / v5.LOCK_NAME).exists()
    monkeypatch.setattr(v5, "worktree_status", lambda cwd=None: [])
    monkeypatch.setattr(v5, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else "same")
    monkeypatch.setattr(v5, "PREFLIGHT_PATH", tmp_path / "preflight.json")
    monkeypatch.setattr(v5, "SHAKEDOWN_PATH", tmp_path / "shakedown.json")
    with pytest.raises(PIC2DValidationError, match="preflight.json and shakedown.json"):
        v5.launch(protocol, results=tmp_path / "launch", log=lambda _: None)
    (tmp_path / "preflight.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shakedown.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PIC2DValidationError, match="--resume needs"):
        v5.launch(protocol, results=tmp_path / "launch", resume=True, log=lambda _: None)
    monkeypatch.setattr(v5, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else ("blob-a" if args[0] == "rev-parse" else "blob-b"))
    with pytest.raises(PIC2DValidationError, match="differs from the committed blob"):
        v5.launch(protocol, results=tmp_path / "launch", log=lambda _: None)
    # the GPU-load snapshot never raises (nvidia-smi missing -> None)
    monkeypatch.setattr(v5.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no nvidia-smi")))
    assert v5.gpu_load_snapshot() is None
