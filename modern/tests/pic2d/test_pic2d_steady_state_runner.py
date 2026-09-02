"""Steady-state runner (model v1.2): plateau criterion, status/checkpoint cadence, bitwise resume.

* the plateau rule needs BOTH drifts under the threshold AND >= 3 transit times;
* a synthetic exponential approach is declared a plateau only once its trailing
  20 % drift falls under 5 %, a linear ramp never is, noise around a constant is;
* a short CPU run writes one status line per sync, one series record per sync,
  a checkpoint per ``checkpoint_every_steps`` chunk and a summary on stop;
* killing the run after N steps and resuming reproduces the uninterrupted run
  bitwise (particles, phi, cumulative ledger) and the resumed intervals report
  currents over the interval since the checkpoint.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import linear_psi_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.models import BoundaryPotentials, ChannelGeometry, Grid2D, PoissonConfig2D, StabilityLimits
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation
from experiments.pic2d_cft_steady_state_v1 import run as runner

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
RULE = {"plateau_threshold": 0.05, "plateau_window_fraction": 0.2, "min_transit_times": 3}
TRANSIT = 1.0


def _series(t: np.ndarray, values: np.ndarray, electrons: np.ndarray | None = None):
    return runner.evaluate_plateau(t, values, values if electrons is None else electrons, RULE, TRANSIT)


def test_plateau_requires_min_transit_times_even_when_flat():
    t = np.linspace(0.0, 2.0, 400)
    flat = np.full_like(t, 3.0e-3)
    result = _series(t, flat)
    assert result["drifts_within_threshold"] and not result["reached"]
    assert result["transit_times_elapsed"] == pytest.approx(2.0)
    t = np.linspace(0.0, 3.2, 640)
    assert _series(t, np.full_like(t, 3.0e-3))["reached"]


def test_plateau_exponential_approach_declared_only_after_settling():
    tau = 1.5
    t = np.linspace(0.0, 12.0, 2400)
    y = 1.0 - np.exp(-t / tau)
    reached_at = None
    for i in range(400, t.size, 20):
        result = _series(t[:i], y[:i])
        if result["reached"]:
            reached_at = t[i - 1]
            break
    assert reached_at is not None
    # trailing-20 % drift of 1 - exp(-t/tau) drops below 5 % at ~2.8 tau, and 3 transit times = 3.0
    assert 3.0 <= reached_at <= 4.6


def test_plateau_linear_ramp_never_and_noisy_constant_yes():
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 10.0, 5000)
    ramp = 1.0 + 0.5 * t
    assert not _series(t, ramp)["reached"]
    assert _series(t, ramp)["discharge_current_drift"] > 0.05
    # 15 % sample noise (the I_d noise per 200-step interval in v2) over ~1000 trailing samples -> ~2 % drift noise
    noisy = 1.0 + 0.15 * rng.standard_normal(t.size)
    result = _series(t, noisy)
    assert result["reached"] and abs(result["discharge_current_drift"]) < 0.05


def test_plateau_needs_both_quantities():
    t = np.linspace(0.0, 10.0, 5000)
    flat = np.ones_like(t)
    ramp = 1.0 + 0.5 * t
    assert not _series(t, flat, ramp)["reached"]
    assert not _series(t, ramp, flat)["reached"]
    assert _series(t, flat, flat)["reached"]


def test_trailing_drift_undefined_on_short_records():
    t = np.linspace(0.0, 1.0, 5)
    assert runner.trailing_time_drift(t, np.ones(5), 0.2) is None
    assert _series(t, np.ones(5))["reached"] is False


# -- cadence and resume ------------------------------------------------------

def _tiny_protocol(tmp_path: Path) -> dict:
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["case"].update({"radial_cells": 12, "axial_cells": 96, "macro_weight": 2.0e6, "seed": 3})
    protocol["operating_point"].update({
        "neutral_density_per_m3": 1.0e21, "electron_injection_current_a": 0.05, "seed_plasma_density_per_m3": 1.0e16,
    })
    protocol["numerics"].update({
        "dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 40,
        "averaging_window_steps": 80, "ion_subcycle": 1,
    })
    protocol["numerics"]["stability_limits"]["max_cell_debye_ratio"] = 4.0
    protocol["numerics"]["stability_reference"] = {"density_per_m3": 1.0e16, "electron_temperature_ev": 5.0, "max_electron_energy_ev": 400.0}
    protocol["budget_v1_2"]["ion_transit_time_s"] = 1.0e-9
    protocol["budget_v1_2"]["n_max_per_m3"] = 4.0e17
    protocol["budget_v1_2"]["n_eq_projected_per_m3"] = 1.0e17
    return protocol


@pytest.fixture
def tiny(tmp_path: Path):
    protocol = _tiny_protocol(tmp_path)
    config = runner.build_config(protocol, backend="cpu")
    field = linear_psi_field_map(config.grid, 2.0)
    xs = XenonCrossSections.from_file()
    return protocol, config, field, xs


def test_status_and_checkpoint_cadence_and_summary(tiny, tmp_path: Path):
    protocol, config, field, xs = tiny
    results = tmp_path / "results"
    summary_path = runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=200, log=lambda _: None)
    assert summary_path.is_file()
    summary = artifacts.read_canonical_json(summary_path)
    assert summary["stop_reason"] == "target_steps_reached" and summary["steps_completed"] == 200
    status_lines = [json.loads(l) for l in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [s for s in status_lines if "event" not in s]
    assert len(samples) == 200 // 20                      # one per sync
    assert [s["step"] for s in samples] == list(range(20, 201, 20))
    for key in ("time_s", "electrons", "ions", "discharge_a", "exit_ion_beam_a", "n_e_peak_node_per_m3", "n_e_mean_per_m3",
                "t_e_mean_ev", "omega_pe_dt_max", "wall_seconds_total", "ms_per_step"):
        assert key in samples[-1]
    assert samples[-1]["plateau"] is not None and "reached" in samples[-1]["plateau"]
    assert status_lines[-1] == {"event": "stop", "step": 200, "time_s": pytest.approx(200 * config.dt_s), "stop_reason": "target_steps_reached"}
    series = [json.loads(l) for l in (results / "series.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(series) == 10 and series[-1]["step"] == 200
    # checkpoint per chunk (40 steps): the live checkpoint is at the last chunk boundary
    checkpoint = artifacts.read_canonical_json(results / "checkpoint" / "checkpoint-latest.json")
    assert checkpoint["step"] == 200
    state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    assert state["finished"] and state["checkpoint_step"] == 200 and len(state["sessions"]) == 1
    assert (results / "run.pid").is_file() and (results / "maps.npz").is_file() and (results / "series.npz").is_file()
    assert summary["artifacts"]["maps_npz_sha256"] and summary["plateau"] is not None
    # the run is "on GPU-idle single process" agnostic: the summary carries the wall budget bookkeeping
    assert summary["wall_seconds_total"] > 0


def test_resume_reproduces_uninterrupted_run_bitwise(tiny, tmp_path: Path):
    protocol, config, field, xs = tiny
    reference = tmp_path / "reference"
    runner.run_steady_state(protocol, reference, backend="cpu", field_map=field, cross_sections=xs, max_steps=160, log=lambda _: None)
    interrupted = tmp_path / "interrupted"
    # session 1: stop after 80 steps (two checkpoints); session 2: resume to 160
    runner.run_steady_state(protocol, interrupted, backend="cpu", field_map=field, cross_sections=xs, max_steps=80, log=lambda _: None)
    assert artifacts.read_canonical_json(interrupted / "checkpoint" / "checkpoint-latest.json")["step"] == 80
    # simulate a crash between a sync and the next checkpoint: an extra series record past the checkpoint must be dropped
    stray = json.loads((interrupted / "series.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    stray["step"] = 100
    with (interrupted / "series.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(stray) + "\n")
    runner.run_steady_state(protocol, interrupted, backend="cpu", field_map=field, cross_sections=xs, max_steps=160, log=lambda _: None)
    a = np.load(reference / "checkpoint-final.npz")
    b = np.load(interrupted / "checkpoint-final.npz")
    for key in a.files:
        assert np.array_equal(a[key], b[key]), key
    sa = artifacts.read_canonical_json(reference / "summary.json")
    sb = artifacts.read_canonical_json(interrupted / "summary.json")
    assert sa["steps_completed"] == sb["steps_completed"] == 160
    assert len(sb["sessions"]) == 2 and sb["sessions"][1]["resumed_from_step"] == 80
    # series: same steps, same particle counts, same currents on every interval (bookkeeping re-based at the resume)
    ra = np.load(reference / "series.npz")
    rb = np.load(interrupted / "series.npz")
    assert np.array_equal(ra["step"], rb["step"]) and rb["step"].tolist() == list(range(20, 161, 20))
    assert np.array_equal(ra["electrons"], rb["electrons"]) and np.array_equal(ra["ions"], rb["ions"])
    for key in ("current_discharge_a", "current_wall_ion_a", "current_injected_electron_a", "current_exit_ion_beam_a"):
        assert np.array_equal(ra[key], rb[key]), key
    # the resumed session restarts the interval ledger: zero residual/electrode work on its first record only
    first_resumed = int(np.searchsorted(rb["step"], 100))
    assert rb["interval_electrode_work_j"][first_resumed] == 0.0 and rb["interval_residual_j"][first_resumed] == 0.0
    assert rb["interval_electrode_work_j"][first_resumed + 1] == ra["interval_electrode_work_j"][first_resumed + 1]
    status_lines = [json.loads(l) for l in (interrupted / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for s in status_lines if s.get("event") == "resume") == 1


def test_load_state_rebases_interval_bookkeeping():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    xs = XenonCrossSections.from_file()
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=4.0), series_interval_steps=20,
    )
    reference = Simulation(config, field, cross_sections=xs)
    reference.run(40)
    first = Simulation(config, field, cross_sections=xs)
    first.run(20)
    resumed = Simulation(config, field, cross_sections=xs)
    resumed.load_state(first.state)
    resumed.run(20)
    assert resumed.series[-1].currents_a == reference.series[-1].currents_a
    assert resumed.series[-1].ledger["cumulative"] == reference.series[-1].ledger["cumulative"]
    assert resumed.series[-1].ledger["interval_electrode_work_j"] == 0.0  # no previous sample in this session
