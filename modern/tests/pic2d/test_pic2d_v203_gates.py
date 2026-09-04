"""Model v2.0.3 gates (2026-09-04, after plume attempt 8's finite-grid-heating runaway).

* Peak-node Debye gate in WINDOW mode: the gated statistic is the interval-averaged peak read from the
  window accumulators (the ``maps.npz`` construction), hard ``max(dr, dz) / lambda_D <= pi`` once the
  window is complete, a soft margin recorded per record; the single-step sample stays the witness.
  Regressions: the window statistic equals the window maps' peak exactly; a single-record spike that
  out-reads the threshold does not trip the averaged gate while a sustained state does, and only once
  the window is complete; the window is continuous across the runner's accumulator resets and restarts
  on ``load_state``; the v1.4 / v2.0.0-v2.0.2 configuration identities are unchanged.
* Windowed residual-power gate in the runner's triad: the trailing-window ledger residual over the
  electrode work (one-sided, heating) stops the run; on an attempt-8-like ramp it trips >= 1 us before the
  cumulative bound while the cumulative ratio stays recorded; on the accepted channel-only plateau runs
  (negative residual throughout) it never fires.
* Runner integration: v2.0.3 protocol blocks flow into status / series / summary and the plateau
  preconditions.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map, zero_field_map
from cft_revival.pic2d.kernels import ParticleArrays
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    EV_J,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DStabilityError,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.simulation import (
    PEAK_WINDOW_SUM_KEYS,
    InjectionConfig,
    PeakDebyeGateConfig,
    PeakDebyeWindow,
    PIC2DConfig,
    SeedPlasmaConfig,
    Simulation,
    SimulationState,
    empty_cumulative,
    window_peak_debye,
)
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v1.gpu_sampler import GpuUtilisationSampler

MODERN = Path(__file__).resolve().parents[2]
CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
DT = 1.5e-12


@pytest.fixture(autouse=True)
def _no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(runner, "GpuUtilisationSampler", functools.partial(GpuUtilisationSampler, query=lambda timeout_s: None))


def _debye(n: float, t_ev: float) -> float:
    return sqrt(EPSILON_0_F_PER_M * t_ev * EV_J / (n * ELEMENTARY_CHARGE_C**2))


def _config(grid: Grid2D, **overrides) -> PIC2DConfig:
    base = {
        "grid": grid, "potentials": BoundaryPotentials(300.0, 0.0), "dt_s": 5e-12, "macro_weight": 2e5, "seed": 3,
        "injection": InjectionConfig(0.05, 2.0), "seed_plasma": SeedPlasmaConfig(1e17, 5.0), "mcc": MCCConfig(1e21),
        "poisson": PoissonConfig2D(method="direct"), "reference_density_per_m3": 1e16, "reference_electron_temperature_ev": 5.0,
        "limits": StabilityLimits(max_cell_debye_ratio=4.0), "series_interval_steps": 10,
    }
    return PIC2DConfig(**(base | overrides))


# -- configuration contract and identities ---------------------------------------------------------------------------

def test_peak_debye_gate_window_mode_contract_and_v14_identity_unchanged():
    single = PeakDebyeGateConfig(4.5, min_macro_particles_at_peak=32)
    assert not single.windowed
    assert single.to_dict() == {"max_cells_per_debye": 4.5, "min_macro_particles_at_peak": 32, "dense_fraction": 0.5}   # v1.4 identity
    windowed = PeakDebyeGateConfig(pi, min_macro_particles_at_peak=32, window_steps=400_000, soft_cells_per_debye=2.5)
    assert windowed.windowed and windowed.window_snapshot_steps == 40_000      # default: window / 10 = the checkpoint cadence
    assert windowed.to_dict() == {"max_cells_per_debye": pi, "min_macro_particles_at_peak": 32, "dense_fraction": 0.5,
                                  "window_steps": 400_000, "window_snapshot_steps": 40_000, "soft_cells_per_debye": 2.5}
    assert PeakDebyeGateConfig(pi, window_steps=400_000).soft_cells_per_debye is None
    assert PeakDebyeGateConfig(pi, window_steps=100, window_snapshot_steps=100).window_snapshot_steps == 100
    for bad in (0, -1, 1.5, True):
        with pytest.raises(PIC2DValidationError):
            PeakDebyeGateConfig(pi, window_steps=bad)  # type: ignore[arg-type]
    for bad in (0, 401, 1.5, True):
        with pytest.raises(PIC2DValidationError):
            PeakDebyeGateConfig(pi, window_steps=400, window_snapshot_steps=bad)  # type: ignore[arg-type]
    for bad in (0.0, -1.0, pi + 1e-9, float("nan"), True):
        with pytest.raises(PIC2DValidationError):
            PeakDebyeGateConfig(pi, window_steps=400, soft_cells_per_debye=bad)  # type: ignore[arg-type]
    with pytest.raises(PIC2DValidationError, match="window mode"):
        PeakDebyeGateConfig(pi, soft_cells_per_debye=2.5)
    with pytest.raises(PIC2DValidationError, match="window mode"):
        PeakDebyeGateConfig(pi, window_snapshot_steps=10)
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    a = _config(grid, peak_debye_gate=single)
    b = _config(grid, peak_debye_gate=windowed)
    assert artifacts.config_identity(a) != artifacts.config_identity(b)
    assert a.to_dict()["peak_debye_gate"] == single.to_dict() and "window_steps" not in a.to_dict()["peak_debye_gate"]


# -- the window statistic ---------------------------------------------------------------------------------------------

def test_window_peak_debye_equals_the_window_maps_peak_and_honours_the_occupancy_floor():
    """The gated quantity is the peak of the SAME interval average maps.npz holds (n_e, moment T_e), the resolved set is
    the mean-occupancy floor on sample_count_e, and the single-step witness is still recorded."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    gate = PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=8, window_steps=40, soft_cells_per_debye=0.5, window_snapshot_steps=10)
    sim = Simulation(_config(grid, peak_debye_gate=gate), field, cross_sections=xs)
    sim.run(40, accumulate_from_step=0)
    maps = sim.diagnostic_arrays()
    assert int(maps["window_steps"][0]) == 40
    peak = sim.series[-1].peak_node
    assert peak["gate_mode"] == "window" and peak["gate_enforced"] is False and peak["gate_max_cells_per_debye"] == 1e6
    assert peak["cells_per_debye"] > 0.0 and peak["macro_particles_at_peak"] >= 8          # single-step witness still there
    window = peak["window"]
    assert window["window_steps"] == 40 and window["window_complete"] and window["gate_enforced"] and window["window_records"] == 4
    assert window["window_start_step"] == 0 and window["window_steps_required"] == 40 and window["window_snapshot_steps"] == 10
    i, j = window["node"]
    n_map, t_map = maps["n_e_per_m3"], maps["t_e_ev"]
    assert window["n_e_peak_per_m3"] == n_map[i, j] and window["t_e_peak_ev"] == t_map[i, j]
    assert window["cells_per_debye"] == pytest.approx(max(grid.dr_m, grid.dz_m) / _debye(n_map[i, j], t_map[i, j]), rel=1e-12)
    assert window["debye_length_m"] == pytest.approx(_debye(n_map[i, j], t_map[i, j]), rel=1e-12)
    occupancy = maps["sample_count_e"] / 40
    resolved = sim.masks.plasma_node & (occupancy >= 8)
    assert window["resolved_nodes"] == int(resolved.sum()) > 0
    assert window["n_e_peak_per_m3"] == float(n_map[resolved].max())                 # the densest RESOLVED node
    assert window["mean_macro_particles_at_peak"] == pytest.approx(occupancy[i, j])
    assert window["raw_peak"]["n_e_per_m3"] >= window["n_e_peak_per_m3"]
    assert window["soft_cells_per_debye"] == 0.5 and window["soft_exceeded"] is True   # ~6 cells per lambda_D here
    assert window["r_m"] == pytest.approx(i * grid.dr_m) and window["z_m"] == pytest.approx(j * grid.dz_m)
    # a partial window (15 more steps: 40..55 accumulated -> the ring base at 10 gives a 45-step window)
    sim.run(15, accumulate_from_step=0)
    later = sim.series[-1].peak_node["window"]
    assert later["window_steps"] == 45 and later["window_start_step"] == 10 and later["window_complete"]
    # the free function on explicit sums reproduces the record
    sums = sim.diagnostic_sums()
    direct = window_peak_debye(sim.masks, sim.config, {k: sums[k] for k in PEAK_WINDOW_SUM_KEYS}, int(sums["steps"][0]), min_mean_occupancy=8.0)
    assert direct["window_steps"] == 55 and direct["resolved_nodes"] > 0 and direct["cells_per_debye"] > 0.0
    empty = window_peak_debye(sim.masks, sim.config, {k: np.zeros(grid.node_shape) for k in PEAK_WINDOW_SUM_KEYS}, 0, min_mean_occupancy=8.0)
    assert empty["resolved"] is False and empty["cells_per_debye"] == 0.0 and empty["debye_length_m"] is None


def _dense_node_state(grid: Grid2D, count: int, *, t_ev: float, node=(4, 40)) -> SimulationState:
    """``count`` electron/ion pairs on one interior node with an isotropic thermal spread (field-free, quasi-neutral)."""

    rng = np.random.default_rng(11)
    i, j = node
    r = np.full(count, i * grid.dr_m)
    z = np.full(count, j * grid.dz_m)
    sigma = sqrt(t_ev * EV_J / ELECTRON_MASS_KG)
    v = rng.normal(0.0, sigma, size=(3, count))
    electrons = ParticleArrays(r, z, v[0], v[1], v[2])
    ions = ParticleArrays(r.copy(), z.copy(), np.zeros(count), np.zeros(count), np.zeros(count))
    return SimulationState(step=0, time_s=0.0, electrons=electrons, ions=ions, surface_charge_c=np.zeros(grid.node_shape),
                           phi_v=np.zeros(grid.node_shape), injection_carry=0.0, cumulative=empty_cumulative())


def _static_config(grid: Grid2D, gate: PeakDebyeGateConfig, **overrides) -> PIC2DConfig:
    base = {"potentials": BoundaryPotentials(0.0, 0.0), "injection": None, "mcc": None, "seed_plasma": None, "dt_s": 1e-13,
            "macro_weight": 1e4, "series_interval_steps": 5, "runtime_stability_check_steps": 5, "peak_debye_gate": gate,
            "reference_density_per_m3": 1e12, "max_electron_energy_ev": 1.0}
    return _config(grid, **(base | overrides))


def test_window_gate_trips_on_a_sustained_over_dense_peak_only_once_the_window_is_complete():
    """A static plasma whose peak sits above the threshold: the single-step witness exceeds it from the first record,
    the window statistic is enforced (and stops the run) only when the window is complete."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    state = _dense_node_state(grid, 800, t_ev=1.0)
    # measure the single-step ratio of this state first (a huge threshold so nothing trips)
    probe = Simulation(_static_config(grid, PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=32, window_steps=20, window_snapshot_steps=5)),
                       zero_field_map(grid), backend="cpu")
    probe.load_state(state)
    probe.run(5, accumulate_from_step=0)
    ratio = probe.series[-1].peak_node["cells_per_debye"]
    assert ratio > pi                                                           # this state is over-dense for the cell
    gate = PeakDebyeGateConfig(0.5 * ratio, min_macro_particles_at_peak=32, window_steps=20, soft_cells_per_debye=0.25 * ratio,
                               window_snapshot_steps=5)
    sim = Simulation(_static_config(grid, gate), zero_field_map(grid), backend="cpu")
    sim.load_state(state)
    sim.run(15, accumulate_from_step=0)                                        # 3 records, window 5/10/15 of 20: not enforced
    for record in sim.series:
        peak = record.peak_node
        assert peak["cells_per_debye"] > gate.max_cells_per_debye and peak["gate_enforced"] is False   # witness over, not gated
        assert peak["window"]["window_complete"] is False and peak["window"]["gate_enforced"] is False
        assert peak["window"]["soft_exceeded"] is None                          # soft is reported only when enforced
        assert peak["window"]["cells_per_debye"] == pytest.approx(ratio, rel=2e-2)   # thermal drift ~1 um on a 250 um node
    with pytest.raises(PIC2DStabilityError, match=r"peak-node Debye gate \(window\)"):
        sim.run(10, accumulate_from_step=0)                                    # the 20th accumulated step completes the window
    last = sim.series[-1].peak_node["window"]
    assert last["window_steps"] == 20 and last["window_complete"] and last["gate_enforced"] and last["soft_exceeded"] is True
    # without accumulation the window never fills and the gate cannot fire (recorded: window_steps 0, not enforced)
    idle = Simulation(_static_config(grid, gate), zero_field_map(grid), backend="cpu")
    idle.load_state(state)
    idle.run(25)
    assert idle.series[-1].peak_node["window"]["window_steps"] == 0 and idle.series[-1].peak_node["window"]["gate_enforced"] is False
    # the same threshold in single-step mode stops the run at the FIRST record (the v1.4 behaviour is unchanged)
    old = Simulation(_static_config(grid, PeakDebyeGateConfig(0.5 * ratio, min_macro_particles_at_peak=32)), zero_field_map(grid), backend="cpu")
    old.load_state(state)
    with pytest.raises(PIC2DStabilityError, match="peak-node Debye gate:"):
        old.run(5)


def test_window_gate_ignores_a_single_record_spike_that_the_single_step_gate_would_trip_on():
    """Synthetic accumulator readings: a quiet plasma at 0.9 pi cells per lambda_D with ONE record whose deposit reads
    1.6 pi (a shot-noise / transient extreme value).  The single-step gate at pi trips on that record; the window
    statistic over 20 records stays below pi (the spike is 1/20 of the window) and nothing trips."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    gate = PeakDebyeGateConfig(pi, min_macro_particles_at_peak=32, window_steps=200, soft_cells_per_debye=2.5, window_snapshot_steps=50)
    config = _static_config(grid, gate, series_interval_steps=10, runtime_stability_check_steps=10)
    cell = max(grid.dr_m, grid.dz_m)
    t_ev = 5.0
    quiet_ratio = 0.9 * pi
    n_quiet = EPSILON_0_F_PER_M * t_ev * EV_J / (ELEMENTARY_CHARGE_C**2) * (quiet_ratio / cell) ** 2   # lambda_D = cell / ratio
    node = (4, 40)
    volume = masks.shape_volume_m3[node]
    w_quiet = n_quiet * volume / config.macro_weight                       # macro-electrons per step on the node
    assert w_quiet > 32
    sigma2 = 3.0 * t_ev * EV_J / ELECTRON_MASS_KG                          # <v^2> for an isotropic Maxwellian at T_e (zero drift)

    def reading(total_steps: int, spike_steps: int = 0, spike_factor: float = 1.0) -> dict[str, np.ndarray]:
        sums = {key: np.zeros(grid.node_shape) for key in PEAK_WINDOW_SUM_KEYS}
        weight = w_quiet * (total_steps - spike_steps) + w_quiet * spike_factor * spike_steps
        sums["n_e"][node] = weight * config.macro_weight / volume
        sums["e_weight"][node] = weight
        sums["e_v2"][node] = weight * sigma2                               # the spike is denser, same temperature
        return sums

    window = PeakDebyeWindow(masks, config, gate.window_steps, gate.window_snapshot_steps, float(gate.min_macro_particles_at_peak))
    window.reset((reading(0), 0, 0), 0)
    spike_factor = (1.6 / 0.9) ** 2                                        # density x3.16 -> ratio x1.78 = 1.6 pi in that deposit
    results = []
    single_step = []
    for record in range(1, 41):                                            # 40 records of 10 steps = 400 accumulated steps
        steps = 10 * record
        spike = 10 if record >= 25 else 0                                  # record 25 (steps 241-250) is the spike
        sums = reading(steps, spike_steps=spike, spike_factor=spike_factor)
        out = window.update((sums, steps, 0), steps)
        results.append(out)
        # the single-step (per-record deposit) statistic of that record, as the v1.4 gate would read it
        previous = reading(steps - 10, spike_steps=(10 if record - 1 >= 25 else 0), spike_factor=spike_factor)
        per_record = {k: sums[k] - previous[k] for k in sums}
        single_step.append(window_peak_debye(masks, config, per_record, 10, min_mean_occupancy=32.0)["cells_per_debye"])
    single_step = np.array(single_step)
    assert single_step[:24] == pytest.approx(quiet_ratio, rel=1e-9) and single_step[24] == pytest.approx(1.6 * pi, rel=1e-6)
    assert single_step[24] > pi                                            # the v1.4 single-step gate at pi would have stopped here
    assert np.all(single_step[25:] == pytest.approx(quiet_ratio, rel=1e-9))
    gated = np.array([r["cells_per_debye"] for r in results])
    complete = np.array([r["window_complete"] for r in results])
    assert not complete[:19].any() and complete[19:].all()                 # 200 accumulated steps = 20 records
    assert gated[:24] == pytest.approx(quiet_ratio, rel=1e-9)
    # with the spike inside the window (records 25..44 of a >= 200-step window): mean density x (1 + (f-1)/20) at most
    expected_max = quiet_ratio * sqrt(1.0 + (spike_factor - 1.0) * 10 / 200)
    assert gated[24:].max() == pytest.approx(expected_max, rel=0.02) and gated.max() < pi
    assert np.all(gated[24:] > quiet_ratio) and gated[-1] > quiet_ratio    # the spike is still inside the trailing 200..250 steps
    assert all(r["resolved_nodes"] == 1 and r["node"] == list(node) for r in results)
    assert all(r["window_steps"] <= gate.window_steps + gate.window_snapshot_steps for r in results if r["window_complete"])


def test_window_is_continuous_across_runner_accumulator_resets_and_restarts_on_load_state():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    gate = PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=4, window_steps=40, window_snapshot_steps=10)
    cfg = _config(grid, peak_debye_gate=gate)
    continuous = Simulation(cfg, field, cross_sections=xs)
    continuous.run(60, accumulate_from_step=0)
    chunked = Simulation(cfg, field, cross_sections=xs)
    for _ in range(3):                       # the runner pattern: reset right after the record at a chunk end
        chunked.run(20, accumulate_from_step=0)
        chunked.backend.reset_diagnostics()
    assert chunked.backend.diagnostic_generation == 3 and chunked.diagnostic_arrays()["window_steps"][0] == 0
    a = {r.step: r.peak_node["window"] for r in continuous.series}
    b = {r.step: r.peak_node["window"] for r in chunked.series}
    assert list(a) == list(b) == [10, 20, 30, 40, 50, 60]
    for step in a:
        # snapshots every 10 steps: the trailing window is exactly 40 once complete (base = the snapshot 40 steps back)
        assert a[step]["window_steps"] == b[step]["window_steps"] == min(step, 40)
        assert a[step]["window_start_step"] == b[step]["window_start_step"] == max(step - 40, 0)
        assert a[step]["node"] == b[step]["node"] and a[step]["resolved_nodes"] == b[step]["resolved_nodes"] > 0
        for key in ("cells_per_debye", "n_e_peak_per_m3", "t_e_peak_ev", "mean_macro_particles_at_peak"):
            assert b[step][key] == pytest.approx(a[step][key], rel=1e-9), (step, key)   # carry + partial: round-off only
    assert a[40]["window_complete"] and not a[30]["window_complete"]
    # a loaded (checkpoint) state restarts the window: the first record after the load covers the accumulation since then
    resumed = Simulation(cfg, field, cross_sections=xs)
    resumed.load_state(continuous.state)
    resumed.run(10, accumulate_from_step=60)
    window = resumed.series[-1].peak_node["window"]
    assert window["window_steps"] == 10 and window["window_start_step"] == 60 and window["window_complete"] is False
    assert window["gate_enforced"] is False and window["window_records"] == 1


# -- windowed residual-power gate (runner triad) ---------------------------------------------------------------------

def _ramp_arrays(t_end_s: float, *, onset_s: float, slope_per_s: float, quiet_ratio: float = -0.005, power_w: float = 1.4,
                 interval_steps: int = 200) -> dict[str, np.ndarray]:
    """Attempt-8-like series: a quiet (slightly negative) residual ratio until ``onset_s``, then a linear ramp of the
    per-interval ratio; constant electrode power; 200-step records at dt = 1.5 ps."""

    steps = np.arange(interval_steps, round(t_end_s / DT) + 1, interval_steps, dtype=float)
    t = steps * DT
    electrode = np.full(t.size, power_w * interval_steps * DT)
    ratio = np.where(t < onset_s, quiet_ratio, quiet_ratio + slope_per_s * (t - onset_s))
    return {
        "step": steps, "time_s": t, "interval_residual_j": ratio * electrode, "interval_electrode_work_j": electrode,
        "current_ionization_rate_per_s": np.full(t.size, 1e17), "peak_omega_pe_dt": np.full(t.size, 0.15),
        "peak_node_t_e_dense_ev": np.full(t.size, 9.0),
    }


def _truncate(arrays: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {key: value[:n] for key, value in arrays.items()}


CUMULATIVE_RULE = {
    "plateau_threshold": 0.05, "plateau_window_fraction": 0.2, "min_transit_times": 3,
    "grid_heating_triad": {"energy_residual_over_electrode_work_max": 0.10, "soft_drift_max": 0.05, "hard_drift_max": 0.25,
                           "enforced_after_transit_times": 1.0},
}
WINDOWED_RULE = copy.deepcopy(CUMULATIVE_RULE)
WINDOWED_RULE["grid_heating_triad"] |= {"residual_window_steps": 400_000, "windowed_energy_residual_over_electrode_work_max": 0.05}
TRANSIT = 2.4e-6


def _first_trip_time(arrays: dict[str, np.ndarray], rule: dict, *, every_steps: int = 40_000) -> tuple[float | None, str | None]:
    """Evaluate the triad at every checkpoint (as the runner does) and return the time of the first hard failure."""

    steps = arrays["step"]
    for n in range(1, steps.size + 1):
        if int(steps[n - 1]) % every_steps != 0:
            continue
        triad = runner.evaluate_triad(_truncate(arrays, n), rule, TRANSIT)
        if triad["hard_failures"]:
            return float(arrays["time_s"][n - 1]), triad["hard_failures"][0]
    return None, None


def test_windowed_residual_gate_trips_at_least_one_microsecond_before_the_cumulative_bound_on_a_ramp():
    # attempt 8: per-window ratio -0.5 % at 2.0-2.4 us, +2.4 % (2.4-2.8), +5.8 % (2.8-3.2), ... +54.8 % (4.8-5.0 us) ->
    # a ramp of +0.035 per 0.4 us of the per-interval ratio from 2.4 us gives window means of +1.3 % / +4.8 % / +8.3 % for
    # the same three windows (the onset shape of attempt 8; its later windows accelerate beyond a linear ramp)
    arrays = _ramp_arrays(8.0e-6, onset_s=2.4e-6, slope_per_s=0.035 / 0.4e-6)
    t_cumulative, why_cumulative = _first_trip_time(arrays, CUMULATIVE_RULE)
    t_windowed, why_windowed = _first_trip_time(arrays, WINDOWED_RULE)
    assert t_cumulative is not None and "energy residual / electrode work" in why_cumulative and "windowed" not in why_cumulative
    assert t_windowed is not None and "windowed energy residual" in why_windowed and "finite-grid heating" in why_windowed
    assert t_windowed <= t_cumulative - 1.0e-6, (t_windowed, t_cumulative)
    # the windowed trip is where the trailing 0.6 us window's ratio reaches 5 %: onset + ~0.6/2 + 0.05/slope ~ 3.3 us
    assert 3.0e-6 <= t_windowed <= 3.6e-6 and t_cumulative >= 4.6e-6
    # the cumulative ratio is still recorded (witness), never a hard failure under the windowed rule, and its bound still
    # blocks the plateau (soft); the windowed record carries its window bookkeeping
    at_trip = runner.evaluate_triad(_truncate(arrays, int(np.searchsorted(arrays["time_s"], t_windowed)) + 1), WINDOWED_RULE, TRANSIT)
    assert at_trip["cumulative_residual_is_witness_only"] is True and at_trip["energy_residual_over_electrode_work"] is not None
    assert at_trip["energy_residual_over_electrode_work"] < 0.05 < at_trip["windowed_energy_residual_over_electrode_work"]
    assert at_trip["windowed_energy_residual_window_complete"] and at_trip["windowed_energy_residual_window_steps"] == 400_000
    assert at_trip["thresholds"]["windowed_energy_residual_over_electrode_work_max"] == 0.05
    assert at_trip["thresholds"]["residual_window_steps"] == 400_000 and not at_trip["soft_ok"]
    assert all("energy residual / electrode work" not in failure or "windowed" in failure for failure in at_trip["hard_failures"])
    late = runner.evaluate_triad(_truncate(arrays, int(np.searchsorted(arrays["time_s"], 7.0e-6)) + 1), WINDOWED_RULE, TRANSIT)
    assert late["energy_residual_over_electrode_work"] > 0.10                # cumulative bound exceeded ...
    assert all("windowed" in failure for failure in late["hard_failures"])   # ... but only the windowed member is a hard failure
    # before the window is complete nothing is enforced, whatever the early ratio (the seed transient of the accepted runs
    # is -12.7 % in its first window; a one-sided gate does not see it)
    early = runner.evaluate_triad(_truncate(_ramp_arrays(0.3e-6, onset_s=0.0, slope_per_s=0.0, quiet_ratio=0.5), 1000), WINDOWED_RULE, TRANSIT)
    assert early["windowed_energy_residual_window_complete"] is False and early["hard_failures"] == []
    negative = runner.evaluate_triad(_ramp_arrays(4.0e-6, onset_s=0.0, slope_per_s=0.0, quiet_ratio=-0.13), WINDOWED_RULE, TRANSIT)
    assert negative["windowed_energy_residual_over_electrode_work"] == pytest.approx(-0.13) and negative["hard_failures"] == []
    assert negative["soft_ok"] is False                                     # |cumulative| 13 % > 10 % still blocks the plateau
    # without the v2.0.3 keys the v1.4 cumulative member is the hard gate, unchanged
    old = runner.evaluate_triad(_ramp_arrays(4.0e-6, onset_s=0.0, slope_per_s=0.0, quiet_ratio=-0.13), CUMULATIVE_RULE, TRANSIT)
    assert old["hard_failures"] and "windowed_energy_residual_over_electrode_work" not in old
    assert runner.windowed_energy_residual(arrays, CUMULATIVE_RULE["grid_heating_triad"]) is None


BASE_SERIES = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results" / "series.npz"


@pytest.mark.skipif(not BASE_SERIES.is_file(), reason="accepted base plateau series not checked out")
def test_windowed_residual_gate_never_fires_on_the_accepted_channel_only_plateau():
    """Calibration on the run that succeeded (2026-09-03, 24ab82f4): the accepted 3.44 mA plateau has a NEGATIVE
    window residual throughout (-12.7 % in the seed window rising to -0.2 %), max +0.37 % - the one-sided 5 % gate
    stays silent at every checkpoint while a two-sided 5 % bound would have stopped it before 4 us."""

    series = np.load(BASE_SERIES)
    arrays = {key: np.asarray(series[key], dtype=np.float64) for key in
              ("step", "time_s", "interval_residual_j", "interval_electrode_work_j", "current_ionization_rate_per_s", "peak_omega_pe_dt")}
    assert arrays["step"][-1] == 5_120_000
    ratios = []
    for n in range(1, arrays["step"].size + 1):
        if int(arrays["step"][n - 1]) % 400_000 != 0:
            continue
        triad = runner.evaluate_triad(_truncate(arrays, n), WINDOWED_RULE, TRANSIT)
        assert all("windowed" not in failure for failure in triad["hard_failures"]), (arrays["time_s"][n - 1], triad["hard_failures"])
        if triad["windowed_energy_residual_window_complete"]:
            ratios.append(triad["windowed_energy_residual_over_electrode_work"])
    ratios = np.array(ratios)
    # complete windows at 0.8, 1.2, ..., 4.8 M steps (the (0, 0.4 M] seed window has no record before it): -11.8 % -> -0.2 %
    assert ratios.size == 11 and ratios.min() < -0.11 and ratios.max() < 0.005 and np.all(np.diff(ratios) > 0.0)
    assert (np.abs(ratios) >= 0.05).sum() >= 5                            # a two-sided bound would have killed the accepted run


# -- runner integration ------------------------------------------------------------------------------------------------

def _v203_protocol() -> dict:
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["case"].update({"radial_cells": 12, "axial_cells": 96, "macro_weight": 2.0e6, "seed": 3})
    protocol["operating_point"].update({
        "neutral_density_per_m3": 1.0e21, "electron_injection_current_a": 0.05, "seed_plasma_density_per_m3": 3.0e16,
    })
    protocol["numerics"].update({
        "dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 40,
        "averaging_window_steps": 80, "ion_subcycle": 1,
        "peak_debye_gate": {"max_cells_per_debye": 50.0, "min_macro_particles_at_peak": 4, "dense_fraction": 0.5,
                            "window_steps": 80, "window_snapshot_steps": 40, "soft_cells_per_debye": 40.0,
                            "max_cells_per_debye_note": "test"},
    })
    protocol["numerics"]["stability_limits"]["max_cell_debye_ratio"] = 4.0
    protocol["numerics"]["stability_reference"] = {"density_per_m3": 1.0e16, "electron_temperature_ev": 5.0, "max_electron_energy_ev": 400.0}
    protocol["budget_v1_2"]["ion_transit_time_s"] = 1.0e-9
    protocol["budget_v1_2"]["n_max_per_m3"] = 4.0e17
    protocol["budget_v1_2"]["n_eq_projected_per_m3"] = 1.0e17
    protocol["stopping_rule"]["grid_heating_triad"] = {
        "energy_residual_over_electrode_work_max": 0.10, "soft_drift_max": 0.05, "hard_drift_max": 0.25, "enforced_after_transit_times": 1.0,
        "residual_window_steps": 80, "windowed_energy_residual_over_electrode_work_max": 0.05,
    }
    return protocol


def test_v203_protocol_runs_with_window_gate_records_in_status_series_and_summary(tmp_path: Path):
    protocol = _v203_protocol()
    config = runner.build_config(protocol, backend="cpu")
    gate = config.peak_debye_gate
    assert gate.windowed and gate.window_steps == 80 and gate.window_snapshot_steps == 40 and gate.soft_cells_per_debye == 40.0
    assert config.to_dict()["peak_debye_gate"]["window_steps"] == 80
    field = linear_psi_field_map(config.grid, 2.0)
    xs = XenonCrossSections.from_file()
    results = tmp_path / "v203"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=160, log=lambda _: None)
    samples = [json.loads(line) for line in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [s for s in samples if "event" not in s]
    assert len(samples) == 8
    for sample in samples:
        peak = sample["peak_node"]
        assert peak["gate_mode"] == "window" and peak["gate_enforced"] is False
        assert set(peak["window"]) >= {"cells_per_debye", "window_steps", "window_complete", "gate_enforced", "soft_exceeded", "resolved_nodes"}
    assert samples[3]["peak_node"]["window"]["window_steps"] == 80 and samples[3]["peak_node"]["window"]["window_complete"] is True
    # the window bridges the runner's 80-step accumulator resets (snapshots every 40 steps): 160 -> the trailing (80, 160]
    assert samples[-1]["peak_node"]["window"]["window_steps"] == 80 and samples[-1]["peak_node"]["window"]["window_complete"]
    assert [s["peak_node"]["window"]["window_steps"] for s in samples] == [20, 40, 60, 80, 100, 80, 100, 80]
    triad = samples[-1]["grid_heating_triad"]
    assert "windowed_energy_residual_over_electrode_work" in triad and triad["windowed_energy_residual_window_complete"] is True
    assert triad["cumulative_residual_is_witness_only"] is True and "energy_residual_over_electrode_work" in triad
    assert samples[-1]["plateau"] is not None
    series = np.load(results / "series.npz")
    for key in ("peak_node_window_cells_per_debye", "peak_node_window_n_e_peak_per_m3", "peak_node_window_window_steps", "peak_node_window_resolved_nodes"):
        assert key in series.files and series[key].size == 8, key
    assert np.all(np.isfinite(series["peak_node_window_cells_per_debye"])) and np.all(series["peak_node_window_cells_per_debye"] > 0.0)
    summary = artifacts.read_canonical_json(results / "summary.json")
    debye = summary["peak_node_debye"]
    assert debye["gate_mode"] == "window" and debye["gate"]["window_steps"] == 80 and debye["gate"]["soft_cells_per_debye"] == 40.0
    window = debye["window"]
    assert window["soft_ok"] in (True, False) and window["hard_cells_per_debye"] == 50.0 and window["window_complete_last"] is True
    assert window["cells_per_debye_window_last"] > 0.0 and window["records_above_soft"] is not None
    assert "peak_debye_soft_ok" in summary["plateau"] and summary["plateau"]["peak_debye_soft_ok"] == window["soft_ok"]
    assert summary["grid_heating_triad"]["thresholds"]["residual_window_steps"] == 80
    assert summary["provenance"]["config"]["peak_debye_gate"]["window_steps"] == 80
    # records_to_arrays on a series without the window (v1.4 records) omits the window arrays
    old = [json.loads(line) for line in (results / "series.jsonl").read_text(encoding="utf-8").splitlines()]
    for record in old:
        record["peak_node"].pop("window")
    assert "peak_node_window_cells_per_debye" not in runner.records_to_arrays(old)
    # the soft margin blocks the plateau verdict: a soft level below the observed value -> peak_debye_soft_ok False
    arrays = runner.records_to_arrays([json.loads(line) for line in (results / "series.jsonl").read_text(encoding="utf-8").splitlines()])
    tight_gate = PeakDebyeGateConfig(50.0, min_macro_particles_at_peak=4, window_steps=80, window_snapshot_steps=40,
                                     soft_cells_per_debye=0.5 * float(arrays["peak_node_window_cells_per_debye"][-1]))
    evaluated = runner.evaluate_peak_debye_window(arrays, dataclasses.replace(config, peak_debye_gate=tight_gate))
    assert evaluated["soft_ok"] is False and evaluated["records_above_soft"] >= 1
    assert runner.evaluate_peak_debye_window(arrays, runner.build_config(_single_step_protocol(protocol), backend="cpu")) is None


def _single_step_protocol(protocol: dict) -> dict:
    protocol = copy.deepcopy(protocol)
    for key in ("window_steps", "window_snapshot_steps", "soft_cells_per_debye"):
        protocol["numerics"]["peak_debye_gate"].pop(key)
    return protocol


def test_v203_window_gate_stops_the_runner_fail_closed(tmp_path: Path):
    protocol = _v203_protocol()
    protocol["numerics"]["peak_debye_gate"] = {"max_cells_per_debye": 0.01, "min_macro_particles_at_peak": 1, "window_steps": 40,
                                               "window_snapshot_steps": 20}
    config = runner.build_config(protocol, backend="cpu")
    field = linear_psi_field_map(config.grid, 2.0)
    xs = XenonCrossSections.from_file()
    results = tmp_path / "v203-gate"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=160, log=lambda _: None)
    summary = artifacts.read_canonical_json(results / "summary.json")
    assert summary["stop_reason"] == "runtime_stability_gate_stopped_run"
    assert "peak-node Debye gate (window)" in summary["stability_gate_message"]
    assert summary["steps_completed"] == 40          # the first record whose window is complete, not the first record
