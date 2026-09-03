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


def test_plateau_tracks_neutral_density_when_present():
    t = np.linspace(0.0, 10.0, 5000)
    flat = np.ones_like(t)
    ramp = 1.0 + 0.5 * t
    without = runner.evaluate_plateau(t, flat, flat, RULE, TRANSIT)
    assert without["reached"] and without["tracked"] == ["discharge_current_drift", "electron_count_drift"]
    with_ramp = runner.evaluate_plateau(t, flat, flat, RULE, TRANSIT, neutral_density=ramp)
    assert not with_ramp["reached"] and with_ramp["neutral_density_drift"] > 0.05
    assert with_ramp["tracked"] == ["discharge_current_drift", "electron_count_drift", "neutral_density_drift"]
    with_flat = runner.evaluate_plateau(t, flat, flat, RULE, TRANSIT, neutral_density=3.4e19 * flat)
    assert with_flat["reached"] and abs(with_flat["neutral_density_drift"]) < 1e-9


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


def test_apply_case_merges_variant_and_names_results_dir(tiny):
    protocol, config, _, _ = tiny
    protocol = copy.deepcopy(protocol)
    protocol["variants"] = {
        "seed-b": {"id": "tiny-seed-b", "seed": 7, "wall_budget_seconds": 5.0, "note": "other seed"},
        "w-half": {"macro_weight": protocol["case"]["macro_weight"] / 2},
    }
    base, name = runner.apply_case(protocol, None)
    assert base is protocol and name == "results"
    seed_b, name = runner.apply_case(protocol, "seed-b")
    assert name == "results-seed-b"
    assert seed_b["case"]["seed"] == 7 and seed_b["case"]["id"] == "tiny-seed-b" and seed_b["case"]["variant"] == "seed-b"
    assert seed_b["stopping_rule"]["wall_budget_seconds"] == 5.0
    assert protocol["case"]["seed"] != 7  # the base protocol is untouched
    cfg_b = runner.build_config(seed_b, backend="cpu")
    assert cfg_b.seed == 7 and cfg_b.macro_weight == config.macro_weight
    w_half, name = runner.apply_case(protocol, "w-half")
    assert name == "results-w-half" and runner.build_config(w_half, backend="cpu").macro_weight == config.macro_weight / 2
    # the two variants have distinct config identities from the base and each other
    ids = {artifacts.config_identity(runner.build_config(p, backend="cpu")) for p in (protocol, seed_b, w_half)}
    assert len(ids) == 3
    with pytest.raises(runner.PIC2DValidationError):
        runner.apply_case(protocol, "nope")


def test_variants_load_from_sibling_file_and_keep_the_protocol_frozen(tiny, tmp_path: Path):
    protocol, _, _, _ = tiny
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    assert runner.load_variants(protocol_path) == {}
    (tmp_path / "variants.json").write_text(json.dumps({"variants": {"seed-b": {"seed": 11}}}), encoding="utf-8")
    variants = runner.load_variants(protocol_path)
    merged, name = runner.apply_case(runner.load_protocol(protocol_path), "seed-b", variants)
    assert name == "results-seed-b" and merged["case"]["seed"] == 11
    assert "variants" not in runner.load_protocol(protocol_path)
    # the checked-in v2 variants are consistent with the frozen v2 protocol
    v2 = Path(__file__).resolve().parents[2] / "experiments" / "pic2d_cft_steady_state_v2" / "protocol.json"
    v2_variants = runner.load_variants(v2)
    assert set(v2_variants) == {"seed-b", "w-0.7"}
    base = runner.load_protocol(v2)
    seed_b, _ = runner.apply_case(base, "seed-b", v2_variants)
    w_half, _ = runner.apply_case(base, "w-0.7", v2_variants)
    assert seed_b["case"]["seed"] != base["case"]["seed"] and seed_b["case"]["macro_weight"] == base["case"]["macro_weight"]
    assert w_half["case"]["seed"] == base["case"]["seed"] and w_half["case"]["macro_weight"] == pytest.approx(0.7 * base["case"]["macro_weight"])
    assert seed_b["stopping_rule"]["wall_budget_seconds"] == w_half["stopping_rule"]["wall_budget_seconds"] == 12600


def test_finalize_writes_artifacts_from_checkpoint_without_stepping(tiny, tmp_path: Path):
    protocol, config, field, xs = tiny
    results = tmp_path / "killed"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=80, log=lambda _: None)
    # a run the runner finished itself keeps its window-average maps: re-finalizing is refused
    summary_before = (results / "summary.json").read_bytes()
    with pytest.raises(runner.PIC2DValidationError, match="already finished"):
        runner.finalize(protocol, results, backend="cpu", field_map=field, cross_sections=xs, log=lambda _: None)
    assert (results / "summary.json").read_bytes() == summary_before
    # pretend the process died after the step-80 checkpoint: remove the stop artifacts and add a stray record
    for name in ("summary.json", "summary.json.sha256.json", "maps.npz", "series.npz"):
        (results / name).unlink()
    stray = json.loads((results / "series.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    stray["step"] = 100
    with (results / "series.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(stray) + "\n")
    checkpoint_before = (results / "checkpoint" / "checkpoint-latest.npz").read_bytes()
    summary_path = runner.finalize(protocol, results, backend="cpu", field_map=field, cross_sections=xs,
                                   stop_reason="finalized_no_ignition_reference", log=lambda _: None)
    summary = artifacts.read_canonical_json(summary_path)
    assert summary["steps_completed"] == 80 and summary["stop_reason"] == "finalized_no_ignition_reference"
    assert summary["maps_kind"] == "instantaneous_checkpoint" and summary["averaging_window_steps"] == 1
    assert summary["final_counts"]["electrons"] == artifacts.read_canonical_json(results / "checkpoint" / "checkpoint-latest.json")["electron_count"]
    assert summary["window_maps_summary"]["n_e_peak_per_m3"] > 0.0
    assert summary["sessions"][-1]["finalize_only"] is True
    series = np.load(results / "series.npz")
    assert series["step"].tolist() == list(range(20, 81, 20))          # stray record dropped
    assert (results / "checkpoint" / "checkpoint-latest.npz").read_bytes() == checkpoint_before   # nothing stepped
    final = np.load(results / "checkpoint-final.npz")
    latest = np.load(results / "checkpoint" / "checkpoint-latest.npz")
    for key in latest.files:
        assert np.array_equal(final[key], latest[key]), key
    state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    assert state["finished"] and state["finalized_from_step"] == 80
    status_lines = [json.loads(l) for l in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    assert status_lines[-1]["event"] == "stop" and status_lines[-1]["stop_reason"] == "finalized_no_ignition_reference"


# -- attempt-7 regression: a non-canonical diagnostic must not lose the terminal state ----------------

def test_gpu_utilisation_sample_is_none_not_nan_on_timeout_or_garbage(monkeypatch):
    """Plume attempt 7: 15 of 238 nvidia-smi calls hit the 5 s timeout under GPU contention; the old
    ``float('nan')`` fall-back made summary.json non-canonical at the wall-budget stop."""

    import subprocess
    from experiments.pic2d_cft_snapshot_v1 import run as snapshot

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(snapshot.subprocess, "run", timeout)
    assert snapshot._gpu_utilisation() is None

    class Completed:
        stdout = "nan\n"

    monkeypatch.setattr(snapshot.subprocess, "run", lambda *a, **k: Completed())
    assert snapshot._gpu_utilisation() is None
    Completed.stdout = "37\n"
    assert snapshot._gpu_utilisation() == 37.0


def _stepped_sim(config, field, xs, steps: int):
    sim = Simulation(config, field, cross_sections=xs)
    records: list[dict] = []
    sim.run(steps, accumulate_from_step=0, progress=lambda record: records.append(record.to_dict()))
    return sim, records


def test_final_summary_sanitises_gpu_samples_and_records_a_failed_canonical_write(tiny, tmp_path: Path):
    protocol, config, field, xs = tiny
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    sim, records = _stepped_sim(config, field, xs, 40)
    maps = sim.diagnostic_arrays()
    common = dict(protocol_path=protocol_path, sim=sim, config=config, field_map=field, xs_sha=xs.payload_sha256, records=records,
                  maps=maps, window_range=(0, 40), maps_kind="window_average", stop_reason="wall_clock_budget_reached", gate_error=None,
                  session={"resumed_from_step": 0}, setup_seconds=1.0, wall_session=2.0)
    # (i) NaN / None GPU samples are recorded as null, the summary is canonical and the run is finished
    results = tmp_path / "ok"
    run_state = {"wall_seconds_total": 3.0, "sessions": [{"resumed_from_step": 0}], "checkpoint_step": 40, "finished": True}
    summary_path = runner.write_final_artifacts(protocol=protocol, results=results, run_state=run_state,
                                                gpu_samples=[float("nan"), 12.0, None], **common)
    summary = artifacts.read_canonical_json(summary_path)
    assert summary["gpu_utilisation_percent_samples"] == [None, 12.0, None]
    assert json.loads((results / "run_state.json").read_text(encoding="utf-8"))["finished"] is True
    # (ii) any other non-finite value still fails closed, but the terminal record is honest: not finished, error recorded,
    # stepping artifacts present so `finalize --recover-runner-stop` can rebuild the summary
    poisoned = copy.deepcopy(protocol)
    poisoned["simplifications"] = [float("nan")]
    results = tmp_path / "poisoned"
    run_state = {"wall_seconds_total": 3.0, "sessions": [{"resumed_from_step": 0}], "checkpoint_step": 40, "finished": True}
    with pytest.raises(Exception, match="not canonical finite JSON"):
        runner.write_final_artifacts(protocol=poisoned, results=results, run_state=run_state, gpu_samples=[], **common)
    assert not (results / "summary.json").exists()
    state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    assert state["finished"] is False and state["finalization_error"]["stop_reason_at_failure"] == "wall_clock_budget_reached"
    assert "not canonical finite JSON" in state["finalization_error"]["error"]
    assert state["finalization_error"]["artifacts_written"] == ["maps.npz", "series.npz", "checkpoint-final.json", "checkpoint-final.npz"]
    for name in ("maps.npz", "series.npz", "checkpoint-final.json", "checkpoint-final.npz"):
        assert (results / name).is_file(), name


@pytest.mark.parametrize("max_steps", [80, 120])   # 80: last completed window (0, 80); 120: half-full current window (80, 120)
def test_finalize_recovers_a_runner_stop_whose_summary_write_failed(tiny, tmp_path: Path, max_steps: int):
    protocol, config, field, xs = tiny
    protocol = copy.deepcopy(protocol)
    protocol["stopping_rule"]["wall_budget_seconds"] = 100.0
    results = tmp_path / "attempt7"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=max_steps, log=lambda _: None)
    reference = artifacts.read_canonical_json(results / "summary.json")
    maps_before = (results / "maps.npz").read_bytes()
    final_before = (results / "checkpoint-final.npz").read_bytes()
    # replay the attempt-7 failure: the runner stopped on the budget, wrote maps/series/checkpoint-final, then the summary
    # write raised -> no summary.json, run_state still unfinished (as checkpointed), no stop event
    for name in ("summary.json", "summary.json.sha256.json"):
        (results / name).unlink()
    state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    state.pop("stop_reason", None)
    state.update({"finished": False, "wall_seconds_total": 50.0})
    artifacts.write_canonical_json(results / "run_state.json", state)
    status_lines = (results / "status.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(status_lines[-1])["event"] == "stop"
    (results / "status.jsonl").write_text("\n".join(status_lines[:-1]) + "\n", encoding="utf-8", newline="\n")
    (results / "run.err").write_text("...\ncft_revival.orbit_mc.models.OrbitValidationError: artifact is not canonical finite JSON\n",
                                    encoding="utf-8", newline="\n")
    # fail-closed: no evidence for a generic reason, none for the budget while the recorded wall time is under it
    with pytest.raises(runner.PIC2DValidationError, match="no on-disk evidence"):
        runner.finalize(protocol, results, backend="cpu", field_map=field, cross_sections=xs, recover_runner_stop=True, log=lambda _: None)
    with pytest.raises(runner.PIC2DValidationError, match="does not exceed the budget"):
        runner.finalize(protocol, results, backend="cpu", field_map=field, cross_sections=xs, recover_runner_stop=True,
                        stop_reason="wall_clock_budget_reached", log=lambda _: None)
    with pytest.raises(runner.PIC2DValidationError, match="does not satisfy the plateau rule"):
        runner.finalize(protocol, results, backend="cpu", field_map=field, cross_sections=xs, recover_runner_stop=True,
                        stop_reason="plateau_reached_after_min_transit_times", log=lambda _: None)
    assert not (results / "summary.json").exists()
    state["wall_seconds_total"] = 101.0
    artifacts.write_canonical_json(results / "run_state.json", state)
    summary_path = runner.finalize(protocol, results, backend="cpu", field_map=field, cross_sections=xs, recover_runner_stop=True,
                                   stop_reason="wall_clock_budget_reached", log=lambda _: None)
    summary = artifacts.read_canonical_json(summary_path)
    # the window-average maps and the final checkpoint are reused verbatim; the window range is the runner's
    assert summary["maps_kind"] == "window_average" and summary["stop_reason"] == "wall_clock_budget_reached"
    assert summary["averaging_window_step_range"] == reference["averaging_window_step_range"]
    assert summary["averaging_window_steps"] == reference["averaging_window_steps"]
    assert summary["artifacts"]["maps_npz_sha256"] == reference["artifacts"]["maps_npz_sha256"]
    assert (results / "maps.npz").read_bytes() == maps_before
    assert (results / "checkpoint-final.npz").read_bytes() == final_before
    assert summary["steps_completed"] == reference["steps_completed"] == max_steps
    assert summary["window_maps_summary"] == reference["window_maps_summary"]
    assert summary["plume"] == reference["plume"] and summary["ledger"] == reference["ledger"]
    assert summary["wall_seconds_total"] == 101.0 and summary["sessions"][-1]["recovered_runner_stop"] is True
    recovered_state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    assert recovered_state["finished"] is True and recovered_state["stop_reason"] == "wall_clock_budget_reached"
    recovery = recovered_state["finalization_recovery"]
    assert recovery["mode"] == "runner_stop_artifacts_reused" and "101.0 s > wall_budget_seconds 100 s" in recovery["stop_reason_evidence"]
    assert "not canonical finite JSON" in recovery["original_error"]["error"]
    status_lines = [json.loads(l) for l in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    assert status_lines[-1] == {"event": "stop", "step": max_steps, "time_s": pytest.approx(max_steps * config.dt_s),
                                "stop_reason": "wall_clock_budget_reached"}
    # a recovered (or any finished) run is not recovered twice
    with pytest.raises(runner.PIC2DValidationError, match="already has a summary.json"):
        runner.finalize(protocol, results, backend="cpu", field_map=field, cross_sections=xs, recover_runner_stop=True,
                        stop_reason="wall_clock_budget_reached", log=lambda _: None)


def test_v13_run_records_neutral_inventory_in_status_series_and_summary(tiny, tmp_path: Path):
    protocol, _, field, xs = tiny
    protocol = copy.deepcopy(protocol)
    from cft_revival.pic2d.neutrals import feed_for_density

    n_g0 = protocol["operating_point"]["neutral_density_per_m3"]
    feed = feed_for_density(n_g0 / 2.0, np.pi * protocol["geometry"]["exit_radius_m"] ** 2, protocol["operating_point"]["neutral_temperature_k"])
    protocol["operating_point"]["neutral_inventory"] = {"feed_atoms_per_s": feed, "relaxation_time_s": 1.0e-9}
    config = runner.build_config(protocol, backend="cpu")
    assert config.neutral_inventory is not None and config.neutral_inventory.feed_atoms_per_s == feed
    results = tmp_path / "v13"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=120, log=lambda _: None)
    samples = [json.loads(l) for l in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [s for s in samples if "event" not in s]
    assert all("n_g_per_m3" in s and "n_g_fixed_point_per_m3" in s and "effusion_rate_per_s" in s for s in samples)
    n_g = [s["n_g_per_m3"] for s in samples]
    assert n_g[0] < n_g0 and all(a > b for a, b in zip(n_g, n_g[1:]))       # relaxing toward n_g0/2 - S/c
    assert "neutral_density_drift" in samples[-1]["plateau"]
    series = np.load(results / "series.npz")
    for key in ("neutral_density_per_m3", "neutral_fixed_point_per_m3", "neutral_ledger_fed", "neutral_ledger_artificial"):
        assert key in series.files and series[key].size == 6
    summary = artifacts.read_canonical_json(results / "summary.json")
    neutral = summary["neutral_inventory"]
    assert neutral["transient_is_artificial"] is True
    assert abs(neutral["cumulative_ledger_closure_relative_to_inventory"]) < 1e-12
    assert neutral["final_density_per_m3"] == n_g[-1]
    assert summary["plateau"]["tracked"] == ["discharge_current_drift", "electron_count_drift", "neutral_density_drift"]
    assert summary["provenance"]["config"]["neutral_inventory"]["feed_atoms_per_s"] == feed
    # the checkpoint carries n_g: a resume continues from the same density
    checkpoint = artifacts.read_canonical_json(results / "checkpoint" / "checkpoint-latest.json")
    assert checkpoint["neutral_keys"][0] == "density_per_m3"
    with pytest.raises(runner.PIC2DValidationError):
        bad = copy.deepcopy(protocol)
        bad["numerics"]["series_interval_steps"] = 40
        runner.build_config(bad, backend="cpu")


# -- v1.4: recycling, peak-node gate, grid-heating triad -------------------------

TRIAD_RULE = {
    **RULE,
    "grid_heating_triad": {"energy_residual_over_electrode_work_max": 0.10, "soft_drift_max": 0.05, "hard_drift_max": 0.25,
                           "enforced_after_transit_times": 1.0},
}


def _triad_arrays(t: np.ndarray, *, residual_ratio: float, s_slope: float, te_slope: float, te_exp: float = 0.0) -> dict[str, np.ndarray]:
    n = t.size
    electrode = np.full(n, 1.0e-9)
    return {
        "step": np.arange(n, dtype=float), "time_s": t,
        "interval_residual_j": residual_ratio * electrode, "interval_electrode_work_j": electrode,
        "current_ionization_rate_per_s": 1.0e16 * np.exp(s_slope * t), "peak_omega_pe_dt": np.full(n, 0.1),
        "peak_node_t_e_dense_ev": 8.0 * (1.0 + te_slope * t) * np.exp(te_exp * t),
    }


def test_triad_absent_without_block_recorded_soft_and_hard_with_it():
    t = np.linspace(0.0, 2.0, 1000)
    arrays = _triad_arrays(t, residual_ratio=0.04, s_slope=0.0, te_slope=0.0)
    assert runner.evaluate_triad(arrays, RULE, TRANSIT) is None
    ok = runner.evaluate_triad(arrays, TRIAD_RULE, TRANSIT)
    assert ok["soft_ok"] and ok["enforced"] and ok["hard_failures"] == []
    assert ok["energy_residual_over_electrode_work"] == pytest.approx(0.04)
    assert abs(ok["ionisation_rate_drift"]) < 1e-9 and abs(ok["t_e_dense_drift"]) < 1e-9 and abs(ok["omega_pe_dt_drift"]) < 1e-9
    # (i) residual bound: recorded before one transit, fail-closed after
    bad = _triad_arrays(np.linspace(0.0, 0.5, 250), residual_ratio=0.2, s_slope=0.0, te_slope=0.0)
    early = runner.evaluate_triad(bad, TRIAD_RULE, TRANSIT)
    assert not early["enforced"] and not early["soft_ok"] and early["hard_failures"] == []
    late = runner.evaluate_triad(_triad_arrays(t, residual_ratio=0.2, s_slope=0.0, te_slope=0.0), TRIAD_RULE, TRANSIT)
    assert late["hard_failures"] and "energy residual" in late["hard_failures"][0]
    # (ii) dense-cell T_e drift: ~7 % over the window blocks the plateau (soft) but does not stop the run;
    # exponential heating (e-folding 1 transit -> ~40 % over the window) stops it
    mild = runner.evaluate_triad(_triad_arrays(t, residual_ratio=0.0, s_slope=0.0, te_slope=0.25), TRIAD_RULE, TRANSIT)
    assert 0.05 < mild["t_e_dense_drift"] < 0.25 and not mild["soft_ok"] and mild["hard_failures"] == []
    hot = runner.evaluate_triad(_triad_arrays(t, residual_ratio=0.0, s_slope=0.0, te_slope=0.0, te_exp=1.0), TRIAD_RULE, TRANSIT)
    assert hot["t_e_dense_drift"] > 0.25 and hot["hard_failures"] and "t_e_dense_drift" in hot["hard_failures"][0]
    # (ii) ionisation-rate drift alone is enough
    s_ramp = runner.evaluate_triad(_triad_arrays(t, residual_ratio=0.0, s_slope=1.0, te_slope=0.0), TRIAD_RULE, TRANSIT)
    assert s_ramp["hard_failures"] and "ionisation_rate_drift" in s_ramp["hard_failures"][0]


def _v14_protocol(protocol: dict) -> dict:
    from cft_revival.pic2d.neutrals import feed_for_density

    protocol = copy.deepcopy(protocol)
    n_g0 = protocol["operating_point"]["neutral_density_per_m3"]
    feed = feed_for_density(n_g0 / 2.0, np.pi * protocol["geometry"]["exit_radius_m"] ** 2, protocol["operating_point"]["neutral_temperature_k"])
    protocol["operating_point"]["neutral_inventory"] = {
        "feed_atoms_per_s": feed, "relaxation_time_s": 1.0e-9, "wall_recycling": True, "recombination_coefficient": 1.0,
        "wall_temperature_k": 400.0,
    }
    protocol["operating_point"]["seed_plasma_density_per_m3"] = 3.0e16
    protocol["numerics"]["peak_debye_gate"] = {"max_cells_per_debye": 50.0, "min_macro_particles_at_peak": 4, "dense_fraction": 0.5}
    protocol["numerics"]["step_graph"] = True
    protocol["stopping_rule"]["grid_heating_triad"] = TRIAD_RULE["grid_heating_triad"]
    return protocol


def test_v14_run_records_recycling_peak_node_and_triad(tiny, tmp_path: Path):
    base, _, field, xs = tiny
    protocol = _v14_protocol(base)
    config = runner.build_config(protocol, backend="cpu")
    assert config.neutral_inventory.wall_recycling and config.neutral_inventory.wall_temperature_k == 400.0
    assert config.peak_debye_gate is not None and config.peak_debye_gate.max_cells_per_debye == 50.0
    assert config.anomalous is None and runner.step_graph_flag(protocol) is True
    results = tmp_path / "v14"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=120, log=lambda _: None)
    samples = [json.loads(l) for l in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [s for s in samples if "event" not in s]
    assert all("recycled_rate_per_s" in s and "net_utilisation" in s and "peak_node" in s for s in samples)
    assert all(s["net_utilisation"] <= s["gross_utilisation"] for s in samples)
    peak = samples[-1]["peak_node"]
    assert peak["cells_per_debye"] > 0.0 and peak["gate_max_cells_per_debye"] == 50.0 and peak["n_e_peak_per_m3"] > 0.0
    assert "grid_heating_triad" in samples[-1] and "energy_residual_over_electrode_work" in samples[-1]["grid_heating_triad"]
    series = np.load(results / "series.npz")
    for key in ("neutral_recycled_rate_per_s", "neutral_net_utilisation", "neutral_ledger_recycled", "peak_node_cells_per_debye",
                "peak_node_t_e_dense_ev", "peak_node_n_e_peak_per_m3"):
        assert key in series.files and series[key].size == 6, key
    assert np.all(np.isfinite(series["neutral_recycled_rate_per_s"]))
    summary = artifacts.read_canonical_json(results / "summary.json")
    neutral = summary["neutral_inventory"]
    assert neutral["wall_recycling"] is True and neutral["wall_temperature_k"] == 400.0
    assert abs(neutral["cumulative_ledger_closure_relative_to_inventory"]) < 1e-12       # closes WITH the recycled term
    assert neutral["net_utilisation_trailing"] <= neutral["gross_utilisation_trailing"]
    assert summary["grid_heating_triad"] is not None and "triad_soft_ok" in summary["plateau"]
    assert summary["peak_node_debye"]["gate"]["max_cells_per_debye"] == 50.0
    assert summary["peak_node_debye"]["max_cells_per_debye"] >= summary["peak_node_debye"]["trailing_20pct_mean_cells_per_debye"] * 0.999
    assert summary["provenance"]["v1_4_options"]["wall_recycling"] is True
    assert summary["provenance"]["v1_4_options"]["peak_debye_gate"]["max_cells_per_debye"] == 50.0
    # a v1.3 series (no recycled key, no peak node) still loads through records_to_arrays
    old = [json.loads(l) for l in (results / "series.jsonl").read_text(encoding="utf-8").splitlines()]
    for record in old:
        record["neutral"] = {k: v for k, v in record["neutral"].items() if k not in ("recycled_rate_per_s", "net_utilisation")}
        record["neutral"]["ledger"].pop("recycled")
        record["peak_node"] = None
    arrays = runner.records_to_arrays(old)
    assert np.all(np.isnan(arrays["neutral_recycled_rate_per_s"])) and np.all(arrays["neutral_ledger_recycled"] == 0.0)
    assert "peak_node_cells_per_debye" not in arrays


def test_v14_peak_gate_stops_the_run_fail_closed(tiny, tmp_path: Path):
    base, _, field, xs = tiny
    protocol = _v14_protocol(base)
    protocol["numerics"]["peak_debye_gate"] = {"max_cells_per_debye": 0.01, "min_macro_particles_at_peak": 1}
    results = tmp_path / "v14-gate"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=120, log=lambda _: None)
    summary = artifacts.read_canonical_json(results / "summary.json")
    assert summary["stop_reason"] == "runtime_stability_gate_stopped_run"
    assert "peak-node Debye gate" in summary["stability_gate_message"]
    assert summary["steps_completed"] <= 20


# -- v2.0 plume protocol -------------------------------------------------------------------

PLUME_PROTOCOL = Path(__file__).resolve().parents[2] / "experiments" / "pic2d_cft_plume_v1" / "protocol.json"


def _tiny_plume_protocol() -> dict:
    """The real plume protocol shrunk to 0.25 mm cells (body radius moved to a 0.25 mm line) for a CPU run."""

    protocol = copy.deepcopy(runner.load_protocol(PLUME_PROTOCOL))
    protocol["geometry"]["body_dielectric_radius_m"] = 0.0045
    protocol["case"].update({"radial_cells": 48, "axial_cells": 144, "macro_weight": 6.0e5})
    protocol["numerics"].update({"dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 100,
                                 "averaging_window_steps": 200, "frame_recorder": {"cadence_steps": 100, "precision": "float32"}})
    protocol["numerics"]["stability_reference"]["density_per_m3"] = 1.0e16
    protocol["operating_point"]["seed_plasma_density_per_m3"] = 5.0e15
    return protocol


def test_plume_protocol_builds_the_v20_config_and_runs_with_the_v20_artifacts(tmp_path: Path):
    from cft_revival.pic2d.fields import uniform_field_map

    real = runner.build_config(runner.load_protocol(PLUME_PROTOCOL), backend="cpu")
    assert real.grid.geometry.has_plume and real.grid.cell_shape == (240, 720) and real.grid.dr_m == pytest.approx(5e-5)
    assert real.cathode is not None and real.cathode.current_rule == "continuity" and real.injection is None
    assert real.seed_plasma.region == "channel" and real.plume_boundary_gate.max_charge_fraction == 0.25
    # v2.0.1 (attempt 7+): the gate reads only resolved far-field nodes; the protocol declares the 32-particle floor
    assert real.plume_boundary_gate.min_macro_particles_per_node == 32 and real.plume_boundary_gate.enforce_after_s == 2.4e-6
    assert real.neutral_inventory.wall_recycling and real.peak_debye_gate is not None
    assert runner.protocol_budget(runner.load_protocol(PLUME_PROTOCOL))["ion_transit_time_s"] == 3.1e-6

    protocol = _tiny_plume_protocol()
    config = runner.build_config(protocol, backend="cpu")
    field = uniform_field_map(config.grid, 0.02)
    xs = XenonCrossSections.from_file()
    results = tmp_path / "plume"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=400, log=lambda _: None)
    samples = [json.loads(l) for l in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [s for s in samples if "event" not in s]
    assert all("thrust" in s and "plume" in s and "cathode_emission_a" in s for s in samples)
    assert all(abs(s["thrust"]["interval_ledger_residual_kg_m_s"]) < 1e-25 for s in samples)
    assert samples[-1]["plume"]["far_field_phi_max_abs_deviation_v"] == 0.0 and samples[-1]["plume"]["gate_enforced"] is False
    with np.load(results / "series.npz") as series:   # closed before the resume rewrites the file (Windows lock)
        for key in ("momentum_thrust_total_n", "momentum_closure_fraction", "momentum_electrostatic_force_thruster_n",
                    "momentum_cathode_emission_next_a", "plume_exit_plane_axis_potential_v", "plume_charge_fraction_of_peak",
                    "plume_charge_fraction_of_peak_raw", "plume_far_field_resolved_nodes"):
            assert key in series.files and series[key].size == 20, key
        assert np.all(series["plume_charge_fraction_of_peak_raw"] >= series["plume_charge_fraction_of_peak"])
    with np.load(results / "maps.npz") as maps:
        for key in ("plume_ion_current_per_sr_a", "plume_ion_counts_per_theta", "iedf_ion_counts", "iedf_edges_ev", "sample_count_e"):
            assert key in maps.files, key
        assert maps["sample_count_e"].shape == tuple(config.grid.node_shape)
    summary = artifacts.read_canonical_json(results / "summary.json")
    plume = summary["plume"]
    assert plume is not None and plume["window_step_range"] == [200, 400] and plume["window_samples"] == 10
    for key in ("thrust_flux_n", "cold_gas_thrust_n", "thrust_total_n", "thrust_balance_n", "electrostatic_force_thruster_n",
                "exit_plane_axis_potential_v", "specific_impulse_s", "anode_efficiency", "mass_flow_kg_per_s"):
        assert plume[key] is not None and np.isfinite(plume[key]), key
    assert plume["thrust_total_n"] == pytest.approx(plume["thrust_flux_n"] + plume["cold_gas_thrust_n"])
    assert plume["cold_gas_thrust_n"] > 0.0 and plume["ledger_residual_max_kg_m_s"] < 1e-25
    assert 0.0 <= plume["exit_plane_axis_potential_v"] <= 300.0
    assert summary["provenance"]["v2_0_options"]["cathode"]["current_rule"] == "continuity"
    assert summary["window_currents_a"]["cathode_emission_a"] >= 0.003 * 0.99
    # resume continues the same run (two sessions) and keeps the v2.0 blocks
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=600, log=lambda _: None)
    resumed = artifacts.read_canonical_json(results / "summary.json")
    assert resumed["steps_completed"] == 600 and len(resumed["sessions"]) == 2 and resumed["plume"] is not None
    # the frame recorder (declared in the protocol) wrote one frame per cadence across both sessions
    assert resumed["artifacts"]["frames"]["count"] == 6 and resumed["artifacts"]["frames"]["config"]["cadence_steps"] == 100
