"""Model v2.1.2 (2026-09-05): occupancy floor for the grid-heating triad's T_e,dense drift member + the single-step members' definedness.

The external-validation v0 ``channel-20um-bohm-0.4`` launch 2 (record cd9bb41c) was stopped at 1.26 transits by
``t_e_dense_drift -0.328``: the "densest node" held 0.24-1.5 macro-electrons (W 82 467 on a 20 um grid), its moment temperature was
0 to round-off in 73 % of the trailing records and shot noise otherwise, and the member random-walked +-0.22 for 13 checkpoints after
arming before crossing the hard 0.25 - the drift of an UNDEFINED statistic - while the sibling omega_pe dt member was correctly
``None`` under its v2.0.4 floor.  v2.1.2 gives the T_e,dense statistic the same resolved-node reading (the dense set is restricted
to nodes holding >= the occupancy floor in the single-step deposit; undefined -> ``t_e_dense_resolved`` False) and makes both
single-step-deposit members drifts only where every record of the trailing window is resolved (else ``None``: recorded, never
enforced).  Diagnostic only: physics, deposition and every configuration identity are untouched.

Regressions pinned here:

* ``peak_node_debye``: a sparse deposit (no node at the floor) yields an undefined T_e,dense (0.0, resolved False, 0 dense nodes) with
  the unfloored reading kept as the witness; a resolved dense set with an unresolved single-particle neighbour above the dense
  fraction reads the resolved node's temperature while the witness is pulled toward 0; a fully resolved dense set reads the same in
  both;
* a sparse CPU simulation records the new keys, ``records_to_arrays`` carries them, ``evaluate_triad`` reads ``None`` for the member
  and never trips; warp-cpu parity of the statistic;
* an ext-val-like synthetic series (73 % zeros, shot noise otherwise) never trips under the floored reading at any checkpoint while
  its unfloored witness crosses the hard bound; the SAME member still trips on a resolved series with a real T_e runaway;
* legacy series (no flag): the exact proxy ``macro_particles_at_peak >= floor`` when the floor is passed, the recorded reading
  otherwise (``t_e_dense_definedness`` names the rule);
* recorded series: ss-v4 (accepted plateau) resolved at every record -> readings bitwise unchanged, never trips, plateau stands;
  plume attempt 8 still stopped by the residual member at the same record; the ext-val L2 record does NOT trip at 1.26 transits under
  the new reading (member ``None``, resolved share 0, unfloored witness -0.328); alpha = 1/16 stands on its S member (-0.618);
* configuration identity: the recorded ``config_sha256`` of ss-v4 and of the ext-val L2 record are reproduced by the live code;
* runner integration: a tiny CPU protocol writes the definedness block into the summary and the status lines.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import linear_psi_field_map, zero_field_map
from cft_revival.pic2d.kernels import ParticleArrays
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    EV_J,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.simulation import (
    PeakDebyeGateConfig,
    PIC2DConfig,
    Simulation,
    SimulationState,
    empty_cumulative,
    peak_node_debye,
)
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v1.gpu_sampler import GpuUtilisationSampler

MODERN = Path(__file__).resolve().parents[2]
CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
TRANSIT = 2.4e-6
DT = 1.4e-12
CADENCE = 40_000
FLOOR = 32

LEGACY_RULE = {
    "plateau_threshold": 0.05, "plateau_window_fraction": 0.2, "min_transit_times": 3,
    "grid_heating_triad": {"energy_residual_over_electrode_work_max": 0.10, "soft_drift_max": 0.05, "hard_drift_max": 0.25,
                           "enforced_after_transit_times": 1.0, "residual_window_steps": 400_000,
                           "windowed_energy_residual_over_electrode_work_max": 0.05},
}

V4_RESULTS = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results"
ATTEMPT8_SERIES = MODERN / "experiments" / "pic2d_cft_plume_v1" / "results-attempt8-grid-heating-triad-stop" / "series.npz"
ALPHA_1OVER16_SERIES = MODERN / "experiments" / "pic2d_anomalous_transport_v1" / "results" / "alpha-1over16" / "series.npz"
EXTVAL_L2_RESULTS = MODERN / "experiments" / "pic2d_external_validation_v0" / "results" / "channel-20um-bohm-0.4-launch2-triad-gate-stop"

TRIAD_KEYS = ("step", "time_s", "interval_residual_j", "interval_electrode_work_j", "current_ionization_rate_per_s", "peak_omega_pe_dt",
              "peak_node_t_e_dense_ev", "peak_node_macro_particles_at_peak", "current_discharge_a", "electrons")


@pytest.fixture(autouse=True)
def _no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(runner, "GpuUtilisationSampler", functools.partial(GpuUtilisationSampler, query=lambda timeout_s: None))


def _load(path: Path, keys: tuple[str, ...] = TRIAD_KEYS) -> dict[str, np.ndarray]:
    series = np.load(path)
    return {key: np.asarray(series[key], dtype=np.float64) for key in keys if key in series.files}


def _truncate(arrays: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {key: value[:n] for key, value in arrays.items()}


def _checkpoints(arrays: dict[str, np.ndarray], cadence: int = CADENCE, from_step: int = 0):
    steps = arrays["step"]
    for n in range(1, steps.size + 1):
        if int(steps[n - 1]) % cadence == 0 and steps[n - 1] >= from_step:
            yield n


# -- peak_node_debye ---------------------------------------------------------------------------------------------------------

def _config(grid: Grid2D, *, dt: float = 1e-12, macro_weight: float = 1e4, gate: PeakDebyeGateConfig | None = None,
            series_interval: int = 5) -> PIC2DConfig:
    return PIC2DConfig(grid=grid, potentials=BoundaryPotentials(0.0, 0.0), dt_s=dt, macro_weight=macro_weight, seed=3, injection=None, seed_plasma=None, mcc=None,
                       poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e12, reference_electron_temperature_ev=5.0, max_electron_energy_ev=1.0,
                       limits=StabilityLimits(max_cell_debye_ratio=1e6, max_omega_pe_dt=10.0), series_interval_steps=series_interval,
                       runtime_stability_check_steps=series_interval, peak_debye_gate=gate)


def _moments(shape: tuple[int, int], placements: list[tuple[tuple[int, int], float, float]]) -> list[np.ndarray]:
    """Node moments (weight, sum v_r, sum v_theta, sum v_z, sum v^2) for drift-free electrons: ``placements`` = [((i, j), count, T_eV)]."""

    weight, vr, vt, vz, v2 = (np.zeros(shape) for _ in range(5))
    for (i, j), count, t_ev in placements:
        weight[i, j] += count
        v2[i, j] += count * 3.0 * t_ev * EV_J / ELECTRON_MASS_KG       # <v^2> = 3 k T / m_e per particle
    return [weight, vr, vt, vz, v2]


def test_peak_node_debye_floors_the_dense_set_and_keeps_the_unfloored_witness():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    config = _config(grid, macro_weight=8e4)
    volumes = np.where(masks.plasma_node, masks.shape_volume_m3, np.inf)
    tiny = tuple(int(k) for k in np.unravel_index(int(np.argmin(volumes)), volumes.shape))      # the axis corner node
    big = (6, 40)
    # (a) sparse: nothing at the floor -> undefined statistic, unfloored witness kept
    sparse = peak_node_debye(masks, config, *_moments(grid.node_shape, [(tiny, 2.0, 4.0), ((3, 30), 3.0, 6.0), ((7, 60), 5.0, 9.0)]),
                             dense_fraction=0.5, min_particles=FLOOR)
    assert sparse["t_e_dense_resolved"] is False and sparse["t_e_dense_resolved_node_count"] == 0 and sparse["dense_node_count"] == 0
    assert sparse["t_e_dense_ev"] == 0.0 and sparse["macro_particles_at_peak"] < FLOOR and sparse["min_particles_for_peak"] == FLOOR
    assert sparse["t_e_dense_raw_ev"] > 0.0 and sparse["dense_node_count_raw"] >= 1            # the pre-v2.1.2 reading: shot noise, not 0
    # (b) a resolved dense node (40 macro-electrons at 7 eV) + ONE cold macro-electron on the axis corner node that out-reads half the peak
    t_dense = 7.0
    n_big = 40.0 * config.macro_weight / masks.shape_volume_m3[big]
    assert 1.0 * config.macro_weight / masks.shape_volume_m3[tiny] > 0.5 * n_big               # the single particle IS above the dense fraction
    mixed = peak_node_debye(masks, config, *_moments(grid.node_shape, [(big, 40.0, t_dense), (tiny, 1.0, 0.0)]), dense_fraction=0.5, min_particles=FLOOR)
    assert mixed["t_e_dense_resolved"] is True and mixed["dense_node_count"] == 1 and mixed["t_e_dense_resolved_node_count"] == 1
    assert mixed["t_e_dense_ev"] == pytest.approx(t_dense, rel=1e-12)                          # floored: the resolved node's temperature
    assert mixed["dense_node_count_raw"] == 2 and 0.0 < mixed["t_e_dense_raw_ev"] < 0.5 * t_dense   # unfloored: dragged toward the cold particle
    assert mixed["node"] == list(big) and mixed["macro_particles_at_peak"] == 40.0             # the peak node itself was always floored (v1.4)
    # (c) a fully resolved dense set: the two readings coincide (the accepted runs are unchanged by construction)
    full = peak_node_debye(masks, config, *_moments(grid.node_shape, [(big, 40.0, 7.0), ((6, 41), 36.0, 5.0), ((2, 10), 3.0, 1.0)]),
                           dense_fraction=0.5, min_particles=FLOOR)
    assert full["t_e_dense_resolved"] is True and full["dense_node_count"] == 2 and full["dense_node_count_raw"] == 2
    assert full["t_e_dense_ev"] == full["t_e_dense_raw_ev"] and 5.0 < full["t_e_dense_ev"] < 7.0
    # (d) exactly at the floor counts; one below does not
    at_floor = peak_node_debye(masks, config, *_moments(grid.node_shape, [(big, float(FLOOR), 3.0)]), dense_fraction=0.5, min_particles=FLOOR)
    below = peak_node_debye(masks, config, *_moments(grid.node_shape, [(big, float(FLOOR - 1), 3.0)]), dense_fraction=0.5, min_particles=FLOOR)
    assert at_floor["t_e_dense_resolved"] is True and at_floor["t_e_dense_ev"] == pytest.approx(3.0, rel=1e-12)
    assert below["t_e_dense_resolved"] is False and below["t_e_dense_ev"] == 0.0 and below["t_e_dense_raw_ev"] == pytest.approx(3.0, rel=1e-12)
    # (e) no electrons at all: everything undefined, nothing raises
    empty = peak_node_debye(masks, config, *_moments(grid.node_shape, []), dense_fraction=0.5, min_particles=FLOOR)
    assert empty["t_e_dense_resolved"] is False and empty["t_e_dense_ev"] == 0.0 and empty["t_e_dense_raw_ev"] == 0.0 and empty["dense_node_count_raw"] == 0


# -- simulation records ------------------------------------------------------------------------------------------------------

def _thermal_state(grid: Grid2D, placements: list[tuple[tuple[int, int], int, float]], seed: int = 11) -> SimulationState:
    """Electron/ion pairs placed exactly on nodes with Maxwellian electron velocities at T_eV (ions cold)."""

    rng = np.random.default_rng(seed)
    r = np.concatenate([np.full(count, i * grid.dr_m) for (i, _), count, _ in placements])
    z = np.concatenate([np.full(count, j * grid.dz_m) for (_, j), count, _ in placements])
    sigma = np.concatenate([np.full(count, np.sqrt(t_ev * EV_J / ELECTRON_MASS_KG)) for _, count, t_ev in placements])
    velocities = [rng.normal(0.0, 1.0, r.size) * sigma for _ in range(3)]
    electrons = ParticleArrays(r, z, *velocities)
    zeros = np.zeros_like(r)
    ions = ParticleArrays(r.copy(), z.copy(), zeros.copy(), zeros.copy(), zeros.copy())
    return SimulationState(step=0, time_s=0.0, electrons=electrons, ions=ions, surface_charge_c=np.zeros(grid.node_shape), phi_v=np.zeros(grid.node_shape),
                           injection_carry=0.0, cumulative=empty_cumulative())


def _gate() -> PeakDebyeGateConfig:
    return PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=FLOOR, window_steps=400, window_snapshot_steps=40)


def test_sparse_simulation_records_an_undefined_t_e_dense_and_the_runner_member_reads_none():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    # 3-6 macro-electrons per occupied node: nowhere near the floor - the ext-val class (median 0.6 macro-electrons on the densest node)
    placements = [((3, 30), 4, 5.0), ((6, 45), 6, 8.0), ((8, 70), 3, 3.0), ((2, 15), 5, 6.0)]
    sim = Simulation(_config(grid, macro_weight=1e4, gate=_gate(), series_interval=1), zero_field_map(grid), backend="cpu")
    sim.load_state(_thermal_state(grid, placements))
    sim.run(60, accumulate_from_step=0)
    peaks = [record.peak_node for record in sim.series]
    assert peaks and all(p["t_e_dense_resolved"] is False and p["t_e_dense_ev"] == 0.0 and p["t_e_dense_resolved_node_count"] == 0 for p in peaks)
    assert all(p["macro_particles_at_peak"] < FLOOR for p in peaks) and any(p["t_e_dense_raw_ev"] > 0.0 for p in peaks)
    record = sim.series[-1].to_dict()
    assert {"t_e_dense_resolved", "t_e_dense_resolved_node_count", "t_e_dense_raw_ev", "dense_node_count_raw"} <= set(record["peak_node"])
    assert sim.to_provenance()["v1_4_options"]["t_e_dense_statistic"] == {"statistic": "resolved_dense_set_single_step", "min_macro_particles": FLOOR, "dense_fraction": 0.5}
    # the runner's arrays carry the flag; the triad member is None (undefined), never a hard failure, even with the members armed
    arrays = runner.records_to_arrays([r.to_dict() for r in sim.series])
    assert np.all(arrays["peak_node_t_e_dense_resolved"] == 0.0) and np.all(arrays["peak_node_min_particles_for_peak"] == FLOOR)
    rule = copy.deepcopy(LEGACY_RULE)
    rule["grid_heating_triad"].update({"enforced_after_transit_times": 0.0, "residual_window_steps": 4})
    triad = runner.evaluate_triad(arrays, rule, transit_time_s=1e-12, t_e_dense_min_macro_particles=FLOOR)
    assert triad["enforced"] is True and triad["t_e_dense_drift"] is None and not any("t_e_dense" in f for f in triad["hard_failures"])
    definedness = triad["single_step_members_definedness"]
    assert definedness["t_e_dense_resolved_share_of_window"] == 0.0 and definedness["t_e_dense_definedness"] == "record_flag"
    assert definedness["t_e_dense_floor_macro_particles"] == FLOOR and triad["omega_pe_dt_drift"] is None
    assert definedness["omega_pe_dt_resolved_share_of_window"] == 0.0            # no node at the floor -> the v2.0.4 member is undefined too
    assert triad["soft_ok"] is False                                           # an undefined member never certifies a plateau (v1.4 semantics)
    # the status line carries the definedness
    line = runner.status_from_record(record, sim.config, 1.0, wall_seconds_total=1.0, ms_per_step=1.0, plateau=None)
    assert line["peak_node"]["t_e_dense_resolved"] is False and line["peak_node"]["t_e_dense_resolved_node_count"] == 0


def test_resolved_simulation_reads_the_same_as_the_unfloored_statistic_and_the_member_is_a_drift():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    placements = [((6, 45), 60, 8.0), ((6, 46), 48, 7.0), ((2, 15), 3, 2.0)]      # two resolved nodes + a stray triple below the dense fraction
    sim = Simulation(_config(grid, macro_weight=1e4, gate=_gate(), series_interval=1), zero_field_map(grid), backend="cpu")
    sim.load_state(_thermal_state(grid, placements))
    sim.run(60, accumulate_from_step=0)
    peaks = [record.peak_node for record in sim.series]
    assert all(p["t_e_dense_resolved"] is True and p["t_e_dense_resolved_node_count"] >= 1 for p in peaks)
    # while every dense node is resolved the floored and unfloored readings coincide (bitwise: the same nodes, the same weights)
    assert all(p["t_e_dense_ev"] == p["t_e_dense_raw_ev"] for p in peaks if p["dense_node_count"] == p["dense_node_count_raw"])
    assert any(p["dense_node_count"] == p["dense_node_count_raw"] for p in peaks) and all(3.0 < p["t_e_dense_ev"] < 12.0 for p in peaks)
    arrays = runner.records_to_arrays([r.to_dict() for r in sim.series])
    assert np.all(arrays["peak_node_t_e_dense_resolved"] == 1.0)
    rule = copy.deepcopy(LEGACY_RULE)
    rule["grid_heating_triad"].update({"enforced_after_transit_times": 0.0, "residual_window_steps": 4})
    triad = runner.evaluate_triad(arrays, rule, transit_time_s=1e-12, t_e_dense_min_macro_particles=FLOOR)
    assert triad["t_e_dense_drift"] is not None and triad["single_step_members_definedness"]["t_e_dense_resolved_share_of_window"] == 1.0


def test_warp_cpu_backend_reports_the_same_floored_statistic():
    pytest.importorskip("warp")
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    placements = [((6, 45), 60, 8.0), ((5, 44), 40, 6.0), ((0, 3), 2, 4.0)]
    readings = {}
    for backend in ("cpu", "warp-cpu"):
        sim = Simulation(_config(grid, macro_weight=1e4, gate=_gate(), series_interval=5), zero_field_map(grid), backend=backend)
        sim.load_state(_thermal_state(grid, placements))
        sim.run(5, accumulate_from_step=0)
        peak = sim.series[-1].peak_node
        readings[backend] = (peak["t_e_dense_resolved"], peak["t_e_dense_resolved_node_count"], peak["t_e_dense_ev"], peak["t_e_dense_raw_ev"], peak["dense_node_count_raw"])
    cpu, warp = readings["cpu"], readings["warp-cpu"]
    assert cpu[0] is True and cpu[:2] == warp[:2] and cpu[4] == warp[4]
    assert warp[2] == pytest.approx(cpu[2], rel=1e-9) and warp[3] == pytest.approx(cpu[3], rel=1e-9)


# -- runner: synthetic series ---------------------------------------------------------------------------------------------------

def _series(t_end_s: float, *, te: np.ndarray | None = None, te_resolved: float | None = None, macro_particles: float | None = None,
            interval_steps: int = 2000) -> dict[str, np.ndarray]:
    """A quiet alpha = 0-like plateau (constant I_d, N_e, S, omega_pe dt; cooling residual) with a prescribed T_e,dense series."""

    steps = np.arange(interval_steps, round(t_end_s / DT) + 1, interval_steps, dtype=float)
    t = steps * DT
    electrode = np.full(t.size, 1.2 * interval_steps * DT)
    arrays = {
        "step": steps, "time_s": t, "current_discharge_a": np.full(t.size, 3.8e-3), "electrons": np.full(t.size, 2.0e6),
        "current_ionization_rate_per_s": np.full(t.size, 3.6e16), "peak_omega_pe_dt": np.full(t.size, 0.09),
        "peak_node_t_e_dense_ev": np.full(t.size, 6.5) if te is None else te,
        "interval_residual_j": -0.005 * electrode, "interval_electrode_work_j": electrode,
    }
    if te_resolved is not None:
        arrays["peak_node_t_e_dense_resolved"] = np.full(t.size, te_resolved)
    if macro_particles is not None:
        arrays["peak_node_macro_particles_at_peak"] = np.full(t.size, macro_particles)
    return arrays


def _ext_val_like_te(n: int, seed: int = 20260905) -> np.ndarray:
    """73 % zeros (the temperature of a node holding ~one macro-electron), uniform 0-30 eV shot noise otherwise (record cd9bb41c)."""

    rng = np.random.default_rng(seed)
    te = np.where(rng.random(n) < 0.733, 0.0, rng.uniform(0.0, 30.0, n))
    return te


def test_an_undefined_ext_val_like_series_never_trips_while_its_unfloored_witness_does():
    t_end = 3.0 * TRANSIT
    n = round(t_end / DT) // 2000
    te = _ext_val_like_te(n)
    flagged = _series(t_end, te=te, te_resolved=0.0, macro_particles=0.62)
    witness_trips = []
    for k in _checkpoints(flagged):
        triad = runner.evaluate_triad(_truncate(flagged, k), LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
        assert triad["t_e_dense_drift"] is None and not any("t_e_dense" in f for f in triad["hard_failures"])
        assert triad["single_step_members_definedness"]["t_e_dense_resolved_share_of_window"] == 0.0
        assert triad["single_step_members_definedness"]["t_e_dense_definedness"] == "record_flag"
        if triad["enforced"]:
            unfloored = runner.trailing_time_drift(flagged["time_s"][:k], te[:k], 0.2)
            witness_trips.append(abs(unfloored) >= 0.25)
    assert any(witness_trips), "the unfloored random walk should cross the hard bound at some armed checkpoint (as the record did)"
    # the same numbers WITHOUT the flag: (i) the exact legacy proxy with the floor reads None too; (ii) without the floor the legacy reading
    # is reproduced (the recorded stop is reproducible) and the record names the rule
    legacy = _series(t_end, te=te, macro_particles=0.62)
    proxy = runner.evaluate_triad(legacy, LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
    assert proxy["t_e_dense_drift"] is None and proxy["single_step_members_definedness"]["t_e_dense_definedness"] == "legacy_proxy"
    assert proxy["single_step_members_definedness"]["t_e_dense_drift_unfloored"] == pytest.approx(runner.trailing_time_drift(legacy["time_s"], te, 0.2))
    unfloored = runner.evaluate_triad(legacy, LEGACY_RULE, TRANSIT)
    assert unfloored["single_step_members_definedness"]["t_e_dense_definedness"] == "legacy_unfloored"
    assert unfloored["t_e_dense_drift"] == pytest.approx(runner.trailing_time_drift(legacy["time_s"], te, 0.2))
    # a resolved-flag series with the same shot noise IS read as a drift: the floor gates definedness, it does not switch the member off
    resolved = _series(t_end, te=te, te_resolved=1.0, macro_particles=200.0)
    trips = [bool(any("t_e_dense" in f for f in runner.evaluate_triad(_truncate(resolved, k), LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)["hard_failures"]))
             for k in _checkpoints(resolved)]
    assert any(trips)


def test_a_real_t_e_runaway_on_a_resolved_series_still_trips_and_a_window_with_one_unresolved_record_does_not():
    t_end = 3.0 * TRANSIT
    n = round(t_end / DT) // 2000
    t = np.arange(1, n + 1) * 2000 * DT
    ramp = 6.5 * np.where(t > 2.0 * TRANSIT, 1.0 + 0.8 * (t - 2.0 * TRANSIT) / TRANSIT, 1.0)       # +80 % per transit after 2 transits
    hot = _series(t_end, te=ramp, te_resolved=1.0)
    final = runner.evaluate_triad(hot, LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
    assert final["t_e_dense_drift"] > 0.25 and any("t_e_dense_drift" in f for f in final["hard_failures"])
    # one unresolved record inside the trailing window makes the statistic undefined over that window (fail-safe: None, not a drift)
    flag = np.ones(n)
    flag[-3] = 0.0
    hot["peak_node_t_e_dense_resolved"] = flag
    one_hole = runner.evaluate_triad(hot, LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
    assert one_hole["t_e_dense_drift"] is None and one_hole["hard_failures"] == []
    share = one_hole["single_step_members_definedness"]["t_e_dense_resolved_share_of_window"]
    assert 0.99 < share < 1.0
    # the same hole in the omega_pe dt member (a 0.0 = unresolved record, v2.0.4 encoding) -> None as well
    hole = _series(t_end)
    hole["peak_omega_pe_dt"][-3] = 0.0
    assert runner.evaluate_triad(hole, LEGACY_RULE, TRANSIT)["omega_pe_dt_drift"] is None
    assert runner.evaluate_triad(_series(t_end), LEGACY_RULE, TRANSIT)["omega_pe_dt_drift"] == pytest.approx(0.0, abs=1e-12)
    # a resume mixes flagged and unflagged records: each is read by its own rule
    mixed = _series(t_end, te=ramp, macro_particles=200.0)
    mixed["peak_node_t_e_dense_resolved"] = np.where(np.arange(n) < n // 2, np.nan, 1.0)
    read = runner.evaluate_triad(mixed, LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
    assert read["single_step_members_definedness"]["t_e_dense_definedness"] == "record_flag+legacy_proxy" and read["t_e_dense_drift"] == pytest.approx(final["t_e_dense_drift"])


def test_resolved_trailing_drift_contract():
    t = np.arange(1, 101, dtype=float) * 1e-9
    y = np.linspace(1.0, 2.0, t.size)
    drift, share = runner.resolved_trailing_drift(t, y, np.ones(t.size, dtype=bool), 0.2)
    assert share == 1.0 and drift == pytest.approx(runner.trailing_time_drift(t, y, 0.2))
    mask = np.ones(t.size, dtype=bool)
    mask[-1] = False
    in_window = int(np.sum(t >= t[-1] - 0.2 * t[-1]))                     # the window trailing_time_drift fits
    assert runner.resolved_trailing_drift(t, y, mask, 0.2) == (None, pytest.approx(1.0 - 1.0 / in_window))
    assert runner.resolved_trailing_drift(np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), 0.2) == (None, None)
    mask[:] = False
    mask[: t.size // 2] = True                                  # resolved early, unresolved over the whole trailing window
    assert runner.resolved_trailing_drift(t, y, mask, 0.2) == (None, 0.0)


# -- recorded series -----------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (V4_RESULTS / "series.npz").is_file(), reason="ss-v4 series not checked out")
def test_the_accepted_v4_plateau_is_resolved_at_every_record_and_its_readings_are_unchanged():
    arrays = _load(V4_RESULTS / "series.npz")
    assert arrays["step"][-1] == 5_200_000 and float(arrays["peak_node_macro_particles_at_peak"].min()) >= FLOOR
    for k in _checkpoints(arrays, cadence=200_000):
        floored = runner.evaluate_triad(_truncate(arrays, k), LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
        legacy = runner.evaluate_triad(_truncate(arrays, k), LEGACY_RULE, TRANSIT)
        assert floored["hard_failures"] == [] and legacy["hard_failures"] == []
        assert floored["t_e_dense_drift"] == legacy["t_e_dense_drift"] and floored["omega_pe_dt_drift"] == legacy["omega_pe_dt_drift"]   # bitwise
        assert floored["single_step_members_definedness"]["t_e_dense_resolved_share_of_window"] == 1.0
        assert floored["single_step_members_definedness"]["t_e_dense_definedness"] == "legacy_proxy"
    final = runner.evaluate_triad(arrays, LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
    assert final["soft_ok"] is True and abs(final["t_e_dense_drift"]) < 0.05 and final["t_e_dense_drift"] is not None   # the plateau stands
    assert final["single_step_members_definedness"]["t_e_dense_drift_unfloored"] == final["t_e_dense_drift"]


@pytest.mark.skipif(not ATTEMPT8_SERIES.is_file(), reason="plume attempt-8 series not checked out")
def test_attempt_8_is_still_stopped_by_the_residual_member_at_the_same_record():
    arrays = _load(ATTEMPT8_SERIES)
    transit = 3.1e-6
    assert float(arrays["peak_node_macro_particles_at_peak"].min()) >= FLOOR      # the heating runaway was dense: resolved everywhere

    def first_trip(floor: int | None) -> tuple[float, list[str]]:
        for k in _checkpoints(arrays):
            triad = runner.evaluate_triad(_truncate(arrays, k), LEGACY_RULE, transit, t_e_dense_min_macro_particles=floor)
            if triad["hard_failures"]:
                return float(arrays["time_s"][k - 1]), triad["hard_failures"]
        raise AssertionError("no trip")

    t_floored, why_floored = first_trip(FLOOR)
    t_legacy, why_legacy = first_trip(None)
    assert t_floored == t_legacy and why_floored == why_legacy and any("windowed energy residual" in w for w in why_floored)
    assert 3.0e-6 <= t_floored <= 3.4e-6


@pytest.mark.skipif(not (EXTVAL_L2_RESULTS / "series.npz").is_file(), reason="ext-val bohm-0.4 launch-2 record not checked out")
def test_the_ext_val_l2_record_does_not_trip_under_the_floored_reading():
    """Record cd9bb41c re-read (read-only): the member that stopped the run reads None; its unfloored witness reproduces -0.328."""

    protocol = json.loads((EXTVAL_L2_RESULTS / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((EXTVAL_L2_RESULTS / "summary.json").read_text(encoding="utf-8"))
    rule = protocol["stopping_rule"]
    transit = float(runner.protocol_budget(protocol)["ion_transit_time_s"])
    floor = int(protocol["numerics"]["peak_debye_gate"]["min_macro_particles_at_peak"])
    assert floor == FLOOR and summary["stop_reason"] == "grid_heating_triad_gate_stopped_run"
    assert summary["grid_heating_triad"]["hard_failures"] == ["t_e_dense_drift -0.328 exceeds 0.25"]
    arrays = _load(EXTVAL_L2_RESULTS / "series.npz")
    assert arrays["step"][-1] == 2_520_000 and float(arrays["peak_node_macro_particles_at_peak"].max()) < FLOOR   # no node ever reached the floor
    # the recorded reading reproduces without the floor (the record's numbers are reproducible) ...
    recorded = runner.evaluate_triad(arrays, rule, transit)
    assert recorded["t_e_dense_drift"] == pytest.approx(summary["grid_heating_triad"]["t_e_dense_drift"], rel=1e-9)
    assert recorded["hard_failures"] == summary["grid_heating_triad"]["hard_failures"]
    # ... and under the v2.1.2 reading the member is undefined at the recorded stop and at every checkpoint from the sealed arming on
    final = runner.evaluate_triad(arrays, rule, transit, t_e_dense_min_macro_particles=floor)
    definedness = final["single_step_members_definedness"]
    assert final["t_e_dense_drift"] is None and final["hard_failures"] == [] and final["enforced"] is True
    assert definedness["t_e_dense_resolved_share_of_window"] == 0.0 and definedness["t_e_dense_definedness"] == "legacy_proxy"
    assert definedness["t_e_dense_drift_unfloored"] == pytest.approx(-0.3285, abs=1e-3)
    assert final["omega_pe_dt_drift"] is None and definedness["omega_pe_dt_resolved_share_of_window"] == 0.0
    assert final["ionisation_rate_drift"] == pytest.approx(0.0015, abs=1e-3)
    assert 0.0 < final["windowed_energy_residual_over_electrode_work"] < 0.01                     # +0.25 %: numerically clean
    assert final["soft_ok"] is False                                                              # undefined members do not certify a plateau either
    for k in _checkpoints(arrays, from_step=2_000_000):
        assert runner.evaluate_triad(_truncate(arrays, k), rule, transit, t_e_dense_min_macro_particles=floor)["hard_failures"] == []


@pytest.mark.skipif(not ALPHA_1OVER16_SERIES.is_file(), reason="alpha-1over16 record not checked out")
def test_the_alpha_1over16_extinction_stands_on_its_s_member_under_the_floored_reading():
    arrays = _load(ALPHA_1OVER16_SERIES)
    floored = runner.evaluate_triad(arrays, LEGACY_RULE, TRANSIT, t_e_dense_min_macro_particles=FLOOR)
    assert floored["t_e_dense_drift"] is None and floored["omega_pe_dt_drift"] is None
    assert floored["ionisation_rate_drift"] == pytest.approx(-0.618, abs=0.002)
    assert len(floored["hard_failures"]) == 1 and "ionisation_rate_drift" in floored["hard_failures"][0]
    assert floored["single_step_members_definedness"]["t_e_dense_drift_unfloored"] == pytest.approx(0.366, abs=0.002)   # the recorded reading


def test_v4_configuration_identity_is_unchanged():
    """The change is diagnostic only: the preregistered ss-v4 identity (pinned at its preregistration) is reproduced by the live code."""

    from experiments.pic2d_cft_steady_state_v4 import run as v4
    from tests.pic2d.test_pic2d_steady_state_v4 import V4_CONFIG_SHA256_CUDA
    config = runner.build_config(v4.load_protocol(), backend="warp-cuda")
    assert artifacts.config_identity(config) == V4_CONFIG_SHA256_CUDA
    assert "t_e_dense" not in json.dumps(config.to_dict())          # nothing of the new statistic enters the identity


# -- runner integration -----------------------------------------------------------------------------------------------------------

def _tiny_protocol() -> dict:
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["case"].update({"radial_cells": 12, "axial_cells": 96, "macro_weight": 2.0e6, "seed": 3})
    protocol["operating_point"].update({"neutral_density_per_m3": 1.0e21, "electron_injection_current_a": 0.05, "seed_plasma_density_per_m3": 3.0e16})
    protocol["numerics"].update({
        "dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 40, "averaging_window_steps": 80, "ion_subcycle": 1,
        "peak_debye_gate": {"max_cells_per_debye": 50.0, "min_macro_particles_at_peak": 4, "dense_fraction": 0.5, "window_steps": 80,
                            "window_snapshot_steps": 40, "soft_cells_per_debye": 40.0, "max_cells_per_debye_note": "test"},
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


def test_runner_records_the_definedness_in_summary_and_status(tmp_path: Path):
    from cft_revival.pic2d.mcc import XenonCrossSections
    protocol = _tiny_protocol()
    config = runner.build_config(protocol, backend="cpu")
    field = linear_psi_field_map(config.grid, 2.0)
    results = tmp_path / "results"
    summary_path = runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=XenonCrossSections.from_file(),
                                           max_steps=160, log=lambda _: None)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    triad = summary["grid_heating_triad"]
    definedness = triad["single_step_members_definedness"]
    assert definedness["t_e_dense_definedness"] == "record_flag" and definedness["t_e_dense_floor_macro_particles"] == 4
    assert definedness["t_e_dense_resolved_share_of_window"] is not None and "t_e_dense_drift_unfloored" in definedness
    assert summary["peak_node_debye"]["trailing_20pct_t_e_dense_resolved_fraction"] is not None
    assert summary["provenance"]["v1_4_options"]["t_e_dense_statistic"]["min_macro_particles"] == 4
    status = [json.loads(line) for line in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    assert status and all("t_e_dense_resolved" in s["peak_node"] for s in status if s.get("peak_node") is not None)
    with_triad = [s for s in status if s.get("grid_heating_triad") is not None]
    assert with_triad and all("single_step_members_definedness" in s["grid_heating_triad"] for s in with_triad)
    records = [json.loads(line) for line in (results / "series.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all({"t_e_dense_resolved", "t_e_dense_raw_ev"} <= set(r["peak_node"]) for r in records)
