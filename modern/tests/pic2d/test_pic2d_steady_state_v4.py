"""pic2d_cft_steady_state_v4: the preregistered 33.3 um / 1.4 ps grid-refinement check of the v2 base plateau.

* protocol contract: the refined grid divides the geometry exactly, dt / W / gates as declared, the operating point and
  the v1.3 closure are bit-for-bit the v2 base's, the configuration identity is pinned;
* the pinned reference quantities equal the v2 base artifacts on disk;
* the shakedown protocol shrinks cadences only;
* the predeclared assessment classifies synthetic outcomes into the four declared verdicts and refuses an inconsistent
  reference;
* the launch discipline: exclusive lock, dirty-worktree and wrong-commit refusals.
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

MODERN = Path(__file__).resolve().parents[2]
V2_PROTOCOL = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "protocol.json"
# pinned at the preregistration: any change to protocol.json that touches the configuration must be a new experiment
V4_CONFIG_SHA256_CUDA = "f10772b25b030e2ba9d86cbd48b9972b9e143cce689e0b0ee8d4d6f10ccc685f"


def test_v4_protocol_is_the_v2_base_refined_by_1_5_per_axis_with_the_v2_0_3_gates():
    protocol = v4.load_protocol()
    v2 = runner.load_protocol(V2_PROTOCOL)
    config = runner.build_config(protocol, backend="warp-cuda")
    base = runner.build_config(v2, backend="warp-cuda")
    assert config.grid.cell_shape == (90, 720) and base.grid.cell_shape == (60, 480)
    assert config.grid.dr_m == pytest.approx(3e-3 / 90) and config.grid.dz_m == pytest.approx(24e-3 / 720)
    assert config.grid.dr_m == pytest.approx(config.grid.dz_m, rel=1e-12) and config.grid.dr_m == pytest.approx(base.grid.dr_m / 1.5, rel=1e-12)
    # every geometry line on a grid line: bore radius, cone start, exit radius
    for value, spacing in ((2e-3, config.grid.dr_m), (18e-3, config.grid.dz_m), (3e-3, config.grid.dr_m), (24e-3, config.grid.dz_m)):
        assert abs(value / spacing - round(value / spacing)) < 1e-9
    assert config.dt_s == 1.4e-12 and base.dt_s == 1.5e-12
    assert config.macro_weight == pytest.approx(6e4 / 2.25, rel=2e-5) and config.macro_weight == 26666.7
    # same operating point, same v1.3 closure (no recycling), same seed, same injection
    assert config.potentials == base.potentials and config.injection == base.injection and config.seed_plasma == base.seed_plasma
    assert config.mcc == base.mcc and config.neutral_inventory == base.neutral_inventory and config.seed == base.seed == 20260903
    assert not config.neutral_inventory.wall_recycling and config.cathode is None and config.plume_boundary_gate is None
    assert protocol["operating_point"]["neutral_inventory"]["feed_atoms_per_s"] == v2["operating_point"]["neutral_inventory"]["feed_atoms_per_s"]
    # v2.0.3 gates
    gate = config.peak_debye_gate
    assert gate.windowed and gate.max_cells_per_debye == pi and gate.soft_cells_per_debye == 2.5 and gate.min_macro_particles_at_peak == 32
    assert gate.window_steps == 400_000 == protocol["numerics"]["averaging_window_steps"]
    assert gate.window_snapshot_steps == 40_000 == protocol["numerics"]["checkpoint_every_steps"]
    triad = protocol["stopping_rule"]["grid_heating_triad"]
    assert triad["residual_window_steps"] == 400_000 and triad["windowed_energy_residual_over_electrode_work_max"] == 0.05
    assert triad["energy_residual_over_electrode_work_max"] == 0.1 and triad["enforced_after_transit_times"] == 1.0
    assert base.peak_debye_gate is None          # the v2 base had no runtime Debye gate (v1.3)
    # plateau rule as accepted, budget and acceptance declared
    rule = protocol["stopping_rule"]
    assert rule["plateau_threshold"] == 0.05 and rule["plateau_window_fraction"] == 0.2 and rule["min_transit_times"] == 3
    assert rule["wall_budget_seconds"] == 86400 and runner.protocol_budget(protocol)["ion_transit_time_s"] == 2.4e-6
    acceptance = rule["acceptance"]
    assert set(acceptance["d_reclassification"]) == {"converged", "resolution_limited", "refinement_heating", "no_plateau"}
    tolerances = {k: v for k, v in acceptance["c_convergence_tolerances"].items() if k != "note"}
    assert tolerances == {"discharge_current_a": 0.10, "ionization_rate_per_s": 0.10, "gross_utilisation": 0.10, "neutral_density_per_m3": 0.10,
                          "peak_n_e_window_per_m3": 0.20, "t_e_peak_window_ev": 0.20, "exit_ion_beam_a": 0.10}
    assert set(protocol["reference_run"]["quantities"]) >= set(tolerances)
    # frames on and aligned; mesh statistics of the refined grid
    frames = runner.frame_recorder_config(protocol)
    assert isinstance(frames, FrameRecorderConfig) and frames.cadence_steps == 20_000
    frames.validate_alignment(sync_steps=200, checkpoint_every_steps=40_000, window_steps=400_000)
    masks = build_mesh_masks(config.grid)
    record = masks.to_dict()
    assert record["plasma_cells"] == 45810 and record["unknown_nodes"] == 46469
    assert record["plasma_volume_m3"] == pytest.approx(3.432e-7, rel=5e-3)     # the v2 stair-step volume within the cone resolution
    # identity pinned
    assert artifacts.config_identity(config) == V4_CONFIG_SHA256_CUDA
    assert artifacts.config_identity(config) != artifacts.config_identity(base)
    # the expected gate reading at the v2 base peak: 33.3 um / lambda_D(1.64e18, 7.39 eV) = 2.11, inside the soft margin
    ref = protocol["reference_run"]["quantities"]
    lam = np.sqrt(8.8541878128e-12 * ref["t_e_peak_window_ev"] * 1.602176634e-19 / (ref["peak_n_e_window_per_m3"] * 1.602176634e-19**2))
    assert config.grid.dz_m / lam == pytest.approx(2.11, abs=0.02) and 2.11 < gate.soft_cells_per_debye
    assert ref["cells_per_debye_at_peak_window"] == pytest.approx(5e-5 / lam, rel=1e-6)   # the base's 3.17 on its 50 um grid


@pytest.mark.skipif(not (v4.REFERENCE_RESULTS / "maps.npz").is_file(), reason="v2 base artifacts not checked out")
def test_pinned_reference_quantities_equal_the_v2_base_artifacts():
    protocol = v4.load_protocol()
    recomputed = v4.reference_quantities_from_files()
    for key, value in recomputed.items():
        assert protocol["reference_run"]["quantities"][key] == pytest.approx(value, rel=1e-12), key
    assert protocol["reference_run"]["quantities"]["peak_node"] == v4._peak_from_maps(v4.REFERENCE_RESULTS / "maps.npz")["node"] == [14, 286]


def test_shakedown_protocol_shrinks_cadences_only():
    protocol = v4.load_protocol()
    shake = v4.shakedown_protocol(protocol)
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
    assert protocol["status"] == "preregistered_resolution_convergence_study_not_validated"   # the real protocol is untouched


def _fake_results(tmp_path: Path, protocol: dict, *, scale: dict[str, float] | None = None, stop: str = "plateau_reached_after_min_transit_times",
                  windowed: float | None = -0.003, complete: bool = True) -> Path:
    """A results directory holding just what ``assess`` reads: summary.json + maps.npz with the reference values scaled."""

    scale = scale or {}
    ref = protocol["reference_run"]["quantities"]
    results = tmp_path / f"results-{len(list(tmp_path.iterdir()))}"
    results.mkdir()
    grid = runner.build_config(protocol, backend="cpu").grid
    n = np.zeros(grid.node_shape)
    t = np.zeros(grid.node_shape)
    n[20, 300] = ref["peak_n_e_window_per_m3"] * scale.get("peak_n_e_window_per_m3", 1.0)
    t[20, 300] = ref["t_e_peak_window_ev"] * scale.get("t_e_peak_window_ev", 1.0)
    artifacts.write_npz(results / "maps.npz", {"n_e_per_m3": n, "t_e_ev": t, "window_steps": np.array([400_000])})
    summary = {
        "stop_reason": stop, "ion_transit_times": 3.2, "steps_completed": 5_485_800, "git_head": "deadbeef", "protocol_sha256": "0" * 64,
        "provenance": {"config_sha256": "1" * 64}, "maps_kind": "window_average", "sessions": [{}],
        "window_currents_a": {"discharge_a": ref["discharge_current_a"] * scale.get("discharge_current_a", 1.0),
                              "exit_ion_beam_a": ref["exit_ion_beam_a"] * scale.get("exit_ion_beam_a", 1.0)},
        "neutral_inventory": {"trailing_20pct_mean_ionization_rate_per_s": ref["ionization_rate_per_s"] * scale.get("ionization_rate_per_s", 1.0),
                              "propellant_utilisation_trailing": ref["gross_utilisation"] * scale.get("gross_utilisation", 1.0),
                              "trailing_20pct_mean_density_per_m3": ref["neutral_density_per_m3"] * scale.get("neutral_density_per_m3", 1.0)},
        "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": windowed, "windowed_energy_residual_window_complete": complete,
                               "energy_residual_over_electrode_work": -0.02},
        "peak_node_debye": {"window": {"cells_per_debye_window_last": 2.1, "trailing_20pct_mean_cells_per_debye_window": 2.1, "soft_ok": True}},
        "plateau": {"reached": stop == "plateau_reached_after_min_transit_times"},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    return results


def test_assessment_classifies_the_four_declared_outcomes_and_refuses_an_inconsistent_reference(tmp_path: Path):
    protocol = v4.load_protocol()
    quiet = lambda _: None
    converged = v4.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 1.05, "peak_n_e_window_per_m3": 0.85}), log=quiet,
                          reference_check=False)
    assert converged["verdict"] == "converged" and converged["a_plateau"]["passed"] and converged["b_residual_power"]["passed"]
    assert converged["c_convergence"]["all_within"] and converged["c_convergence"]["quantities"]["discharge_current_a"]["relative_difference"] == pytest.approx(0.05)
    assert converged["c_convergence"]["quantities"]["peak_n_e_window_per_m3"]["within"] and converged["d_reclassification"].startswith("(a) AND (b) AND")
    limited = v4.assess(protocol, _fake_results(tmp_path, protocol, scale={"discharge_current_a": 1.12}), log=quiet, reference_check=False)
    assert limited["verdict"] == "resolution_limited" and not limited["c_convergence"]["quantities"]["discharge_current_a"]["within"]
    assert "RESOLUTION-LIMITED" in limited["d_reclassification"]
    heating = v4.assess(protocol, _fake_results(tmp_path, protocol, windowed=0.03), log=quiet, reference_check=False)
    assert heating["verdict"] == "refinement_heating" and not heating["b_residual_power"]["passed"]
    cooling = v4.assess(protocol, _fake_results(tmp_path, protocol, windowed=-0.04), log=quiet, reference_check=False)
    assert cooling["verdict"] == "converged" and cooling["b_residual_power"]["passed"]          # one-sided: the negative side is reported
    incomplete = v4.assess(protocol, _fake_results(tmp_path, protocol, windowed=0.0, complete=False), log=quiet, reference_check=False)
    assert incomplete["verdict"] == "refinement_heating"                                          # no complete window -> (b) cannot pass
    budget = v4.assess(protocol, _fake_results(tmp_path, protocol, stop="wall_clock_budget_reached"), log=quiet, reference_check=False)
    assert budget["verdict"] == "no_plateau" and "inconclusive" in budget["d_reclassification"]
    assert (budget["results_dir"] and (tmp_path / budget["results_dir"] / "assessment.json").is_file())
    assert artifacts.read_canonical_json(tmp_path / budget["results_dir"] / "assessment.json")["verdict"] == "no_plateau"
    # a reference that disagrees with the v2 artifacts on disk is refused (fail closed)
    if (v4.REFERENCE_RESULTS / "maps.npz").is_file():
        bad = copy.deepcopy(protocol)
        bad["reference_run"]["quantities"]["discharge_current_a"] *= 1.01
        with pytest.raises(PIC2DValidationError, match="disagree"):
            v4.assess(bad, _fake_results(tmp_path, protocol), log=quiet, reference_check=True)
        ok = v4.assess(protocol, _fake_results(tmp_path, protocol), log=quiet, reference_check=True)
        assert ok["reference_consistency"] is not None and all(v["agree"] for v in ok["reference_consistency"].values())


def test_launch_discipline_lock_dirty_worktree_and_commit(tmp_path: Path, monkeypatch):
    protocol = v4.load_protocol()
    payload = {"schema_version": v4.LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "commit": "a" * 40, "protocol_sha256": "b" * 64}
    lock = v4.acquire_lock(tmp_path / "results", payload)
    assert lock.is_file() and json.loads(lock.read_text(encoding="utf-8"))["commit"] == "a" * 40
    with pytest.raises(PIC2DValidationError, match="same-attempt"):
        v4.acquire_lock(tmp_path / "results", payload)
    with pytest.raises(PIC2DValidationError, match="different-attempt"):
        v4.acquire_lock(tmp_path / "results", payload | {"commit": "c" * 40})
    # launch refuses a wrong expected commit and a dirty worktree before touching anything
    monkeypatch.setattr(v4, "git", lambda *args, cwd=None: "f" * 40 if args[:1] == ("rev-parse",) else "")
    with pytest.raises(PIC2DValidationError, match="not the preregistration commit"):
        v4.launch(protocol, results=tmp_path / "launch", expect_commit="0123456", log=lambda _: None)
    monkeypatch.setattr(v4, "worktree_status", lambda cwd=None: [" M modern/x.py"])
    with pytest.raises(PIC2DValidationError, match="not clean"):
        v4.launch(protocol, results=tmp_path / "launch", expect_commit="fffffff", log=lambda _: None)
    assert not (tmp_path / "launch" / v4.LOCK_NAME).exists()
    # the preflight and shakedown records must exist before a preregistered launch
    monkeypatch.setattr(v4, "worktree_status", lambda cwd=None: [])
    monkeypatch.setattr(v4, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else "same")
    monkeypatch.setattr(v4, "PREFLIGHT_PATH", tmp_path / "preflight.json")
    monkeypatch.setattr(v4, "SHAKEDOWN_PATH", tmp_path / "shakedown.json")
    with pytest.raises(PIC2DValidationError, match="preflight.json and shakedown.json"):
        v4.launch(protocol, results=tmp_path / "launch", log=lambda _: None)
    (tmp_path / "preflight.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shakedown.json").write_text("{}", encoding="utf-8")
    # --resume without a lock / without a checkpoint is refused
    with pytest.raises(PIC2DValidationError, match="--resume needs"):
        v4.launch(protocol, results=tmp_path / "launch", resume=True, log=lambda _: None)
    # a protocol on disk that differs from the committed blob is refused
    monkeypatch.setattr(v4, "git", lambda *args, cwd=None: "f" * 40 if args == ("rev-parse", "HEAD") else ("blob-a" if args[0] == "rev-parse" else "blob-b"))
    with pytest.raises(PIC2DValidationError, match="differs from the committed blob"):
        v4.launch(protocol, results=tmp_path / "launch", log=lambda _: None)
