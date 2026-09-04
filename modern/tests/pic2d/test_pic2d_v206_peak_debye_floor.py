"""Model v2.0.6: the window-mode peak-Debye gate's resolved set in ACCUMULATED macro-electron-steps.

The v2.0.3 window gate reads the densest node whose MEAN occupancy over the 400 000-step window is >= 32 macro-electrons.
That floor is blind to the small axis nodes: at 20 um / W 82 467 an axis node holds 0.76 macro-electrons per step at
1e19 m^-3, so the external-validation launch 1 ran its axis column at 2.9-3.3 cells per lambda_D (past pi) while the
gate read 2.26 on the densest node meeting the occupancy floor.  Over a 400 000-step window that column has ~300 000
macro-electron-steps of accumulation - a resolved estimate of <n_e> and T_e.  v2.0.6 adds
``PeakDebyeGateConfig.min_accumulated_macro_particle_steps_at_peak`` (the v2.0.2 plume-gate construction, default
64 000 for new protocols = 32 samples x 2000 steps): the gated peak is the densest node with that many accumulated
macro-electron-steps; the v2.0.3 occupancy-floor peak stays recorded as the witness.

Regressions: an axis-peaked synthetic column past pi trips the new gate and not the old (pure function, Simulation and
warp-cpu parity); the accepted ss-v4 plateau map still passes (2.15, same node under both floors); attempt 8's final
window trips (3.61); the ext-val end-state map resolves the near-axis nodes the old floor excluded; identity: the key
enters ``to_dict`` / ``config_sha256`` only when declared, so every v2.0.3-v2.0.5 identity is unchanged; the runner
passes the protocol key through and records the witness arrays.
"""

from __future__ import annotations

import copy
import functools
import json
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import linear_psi_field_map, zero_field_map
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
    PEAK_DEBYE_GATE_DEFAULT_ACCUMULATED_FLOOR,
    PEAK_WINDOW_SUM_KEYS,
    InjectionConfig,
    PeakDebyeGateConfig,
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


# -- configuration contract and identity ---------------------------------------------------------------------------------

def test_accumulated_floor_contract_and_the_v203_identity_is_unchanged_without_it():
    v203 = PeakDebyeGateConfig(pi, min_macro_particles_at_peak=32, window_steps=400_000, soft_cells_per_debye=2.5)
    assert not v203.accumulated_floor and "min_accumulated_macro_particle_steps_at_peak" not in v203.to_dict()
    assert v203.to_dict() == {"max_cells_per_debye": pi, "min_macro_particles_at_peak": 32, "dense_fraction": 0.5,
                              "window_steps": 400_000, "window_snapshot_steps": 40_000, "soft_cells_per_debye": 2.5}    # v2.0.3 identity
    v206 = PeakDebyeGateConfig(pi, min_macro_particles_at_peak=32, window_steps=400_000, soft_cells_per_debye=2.5,
                               min_accumulated_macro_particle_steps_at_peak=PEAK_DEBYE_GATE_DEFAULT_ACCUMULATED_FLOOR)
    assert v206.accumulated_floor and v206.to_dict()["min_accumulated_macro_particle_steps_at_peak"] == 64_000
    assert PEAK_DEBYE_GATE_DEFAULT_ACCUMULATED_FLOOR == 64_000                    # = the v2.0.2 plume-gate floor (32 samples x 2000 steps)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(PIC2DValidationError, match="macro-electron-steps"):
            PeakDebyeGateConfig(pi, window_steps=400, min_accumulated_macro_particle_steps_at_peak=bad)  # type: ignore[arg-type]
    with pytest.raises(PIC2DValidationError, match="window mode"):
        PeakDebyeGateConfig(pi, min_accumulated_macro_particle_steps_at_peak=64_000)          # single-step gate: no window to accumulate
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    a, b = _config(grid, peak_debye_gate=v203), _config(grid, peak_debye_gate=v206)
    assert artifacts.config_identity(a) != artifacts.config_identity(b)             # the floor is part of config_sha256 when declared
    assert a.to_dict()["peak_debye_gate"] == v203.to_dict()


# -- the statistic: an axis-peaked column past pi ------------------------------------------------------------------------

def _synthetic_sums(masks, steps: int, *, axis_n: float, axis_t_ev: float, axis_occupancy: float, interior_n: float,
                    interior_t_ev: float, interior_occupancy: float, axis_j: slice = slice(30, 40), interior=(6, 50)):
    """Window sums with an axis column (i = 0) at ``axis_n`` / ``axis_occupancy`` per step and one interior node."""

    shape = masks.grid.node_shape
    sums = {key: np.zeros(shape) for key in PEAK_WINDOW_SUM_KEYS}

    def put(index, n, t_ev, occupancy):
        w = occupancy * steps
        sums["n_e"][index] = n * steps
        sums["e_weight"][index] = w
        sums["e_v2"][index] = w * 3.0 * t_ev * EV_J / ELECTRON_MASS_KG          # T_e = m <v^2> / 3e with zero drift

    put((0, axis_j), axis_n, axis_t_ev, axis_occupancy)
    put(interior, interior_n, interior_t_ev, interior_occupancy)
    return sums


def test_axis_column_past_pi_is_gated_by_the_accumulated_floor_and_invisible_to_the_occupancy_floor():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    cell = max(grid.dr_m, grid.dz_m)
    steps = 400_000
    # the external-validation picture scaled to this grid: the axis column at 1.2 pi with 0.8 macro-electrons per step
    # (320 000 macro-electron-steps), the densest node the occupancy floor admits at 0.7 pi with 100 per step
    t_axis, t_interior = 9.0, 6.0
    axis_n = t_axis * EV_J * EPSILON_0_F_PER_M * (1.2 * pi / cell) ** 2 / ELEMENTARY_CHARGE_C**2
    interior_n = t_interior * EV_J * EPSILON_0_F_PER_M * (0.7 * pi / cell) ** 2 / ELEMENTARY_CHARGE_C**2
    assert cell / _debye(axis_n, t_axis) == pytest.approx(1.2 * pi) and cell / _debye(interior_n, t_interior) == pytest.approx(0.7 * pi)
    sums = _synthetic_sums(masks, steps, axis_n=axis_n, axis_t_ev=t_axis, axis_occupancy=0.8, interior_n=interior_n,
                           interior_t_ev=t_interior, interior_occupancy=100.0)
    old = window_peak_debye(masks, None, sums, steps, min_mean_occupancy=32.0)
    assert old["node"] == [6, 50] and old["cells_per_debye"] == pytest.approx(0.7 * pi, rel=1e-9) and old["resolved_nodes"] == 1
    assert old["raw_peak"]["node"][0] == 0 and old["raw_peak"]["mean_macro_particles"] == pytest.approx(0.8)     # the axis IS the raw peak
    assert "occupancy_floor_peak" not in old and "accumulated_macro_particle_steps_at_peak" not in old            # v2.0.3 layout
    new = window_peak_debye(masks, None, sums, steps, min_mean_occupancy=32.0, min_accumulated_particle_steps=64_000.0)
    assert new["node"][0] == 0 and 30 <= new["node"][1] < 40 and new["cells_per_debye"] == pytest.approx(1.2 * pi, rel=1e-9)
    assert new["cells_per_debye"] > pi > old["cells_per_debye"]
    assert new["mean_macro_particles_at_peak"] == pytest.approx(0.8) and new["accumulated_macro_particle_steps_at_peak"] == pytest.approx(320_000.0)
    assert new["min_accumulated_macro_particle_steps_at_peak"] == 64_000.0 and new["resolved_nodes"] == 11      # 10 axis nodes + the interior node
    witness = new["occupancy_floor_peak"]
    assert witness["node"] == [6, 50] and witness["cells_per_debye"] == pytest.approx(old["cells_per_debye"]) and witness["resolved_nodes"] == 1
    # a short window: the same column has not accumulated the floor yet -> the interior node gates, the axis is unresolved
    short = window_peak_debye(masks, 50_000, {k: v * (50_000 / steps) for k, v in sums.items()}, 50_000,
                              min_mean_occupancy=32.0, min_accumulated_particle_steps=64_000.0)
    assert short["node"] == [6, 50] and short["resolved_nodes"] == 1 and short["accumulated_macro_particle_steps_at_peak"] == pytest.approx(5e6)
    # K-sampled moments (v2.0.5): the accumulated steps fold the sample count in (occupancy x steps, not sum w)
    sampled = {k: v.copy() for k, v in sums.items()}
    for key in ("e_weight", "e_vr", "e_vt", "e_vz", "e_v2"):
        sampled[key] = sampled[key] / 5.0
    sampled["moment_samples"] = np.array([steps // 5])
    k5 = window_peak_debye(masks, None, sampled, steps, min_mean_occupancy=32.0, min_accumulated_particle_steps=64_000.0)
    assert k5["window_moment_samples"] == steps // 5 and k5["node"][0] == 0
    assert k5["accumulated_macro_particle_steps_at_peak"] == pytest.approx(320_000.0) and k5["cells_per_debye"] == pytest.approx(1.2 * pi, rel=1e-9)
    # empty window keeps the extended layout
    empty = window_peak_debye(masks, None, {k: np.zeros(grid.node_shape) for k in PEAK_WINDOW_SUM_KEYS}, 0, min_mean_occupancy=32.0,
                              min_accumulated_particle_steps=64_000.0)
    assert empty["resolved"] is False and empty["accumulated_macro_particle_steps_at_peak"] == 0.0 and empty["occupancy_floor_peak"]["resolved"] is False


def _axis_state(grid: Grid2D, *, per_node: int, nodes: range, t_ev: float, seed: int = 4) -> SimulationState:
    """``per_node`` warm electron/ion pairs on each axis node of ``nodes`` (r = 0): a thin axis column."""

    rng = np.random.default_rng(seed)
    count = per_node * len(nodes)
    r = np.zeros(count)
    z = np.repeat([j * grid.dz_m for j in nodes], per_node)
    sigma = sqrt(t_ev * EV_J / ELECTRON_MASS_KG)
    v = rng.normal(0.0, sigma, size=(3, count))
    electrons = ParticleArrays(r, z, v[0], v[1], v[2])
    ions = ParticleArrays(r.copy(), z.copy(), np.zeros(count), np.zeros(count), np.zeros(count))
    return SimulationState(step=0, time_s=0.0, electrons=electrons, ions=ions, surface_charge_c=np.zeros(grid.node_shape),
                           phi_v=np.zeros(grid.node_shape), injection_carry=0.0, cumulative=empty_cumulative())


def _static_config(grid: Grid2D, gate: PeakDebyeGateConfig, *, macro_weight: float, backend_series: int = 5) -> PIC2DConfig:
    return _config(grid, potentials=BoundaryPotentials(0.0, 0.0), injection=None, mcc=None, seed_plasma=None, dt_s=1e-14,
                   macro_weight=macro_weight, series_interval_steps=backend_series, runtime_stability_check_steps=backend_series,
                   peak_debye_gate=gate, reference_density_per_m3=1e12, max_electron_energy_ev=1.0,
                   limits=StabilityLimits(max_cell_debye_ratio=1e6, max_omega_pe_dt=10.0))


def _axis_over_dense_weight(grid: Grid2D, per_node: int, t_ev: float, target_ratio: float) -> float:
    """The macro weight that puts ``per_node`` electrons on an axis node at ``target_ratio`` cells per lambda_D."""

    masks = build_mesh_masks(grid)
    volume = float(masks.shape_volume_m3[0, 40])
    cell = max(grid.dr_m, grid.dz_m)
    n_target = t_ev * EV_J * EPSILON_0_F_PER_M * (target_ratio / cell) ** 2 / ELEMENTARY_CHARGE_C**2
    return n_target * volume / per_node


@pytest.mark.parametrize("backend", ["cpu", "warp-cpu"])
def test_simulation_axis_column_trips_the_accumulated_floor_gate_and_not_the_occupancy_floor(backend: str):
    if backend != "cpu":
        pytest.importorskip("warp")
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    per_node, t_ev = 3, 2.0                                                  # 3 macro-electrons per step: far below the 32 floor
    w = _axis_over_dense_weight(grid, per_node, t_ev, 1.3 * pi)
    state = _axis_state(grid, per_node=per_node, nodes=range(36, 44), t_ev=t_ev)
    # (a) v2.0.3 occupancy floor: the axis column is unresolved, nothing is gated, the run continues
    old_gate = PeakDebyeGateConfig(pi, min_macro_particles_at_peak=32, window_steps=20, window_snapshot_steps=5, soft_cells_per_debye=2.5)
    old = Simulation(_static_config(grid, old_gate, macro_weight=w), zero_field_map(grid), backend=backend)
    old.load_state(state)
    old.run(30, accumulate_from_step=0)
    last = old.series[-1].peak_node["window"]
    assert last["window_complete"] and last["resolved"] is False and last["gate_enforced"] is False
    assert last["raw_peak"]["node"][0] == 0 and last["raw_peak"]["mean_macro_particles"] == pytest.approx(per_node, rel=0.35)
    # (b) v2.0.6 accumulated floor (test scale: 40 macro-electron-steps = 32 samples over a 20-step window at ~2-3 per step):
    #     the column is resolved once the window is complete and the gate fails closed on it
    new_gate = PeakDebyeGateConfig(pi, min_macro_particles_at_peak=32, window_steps=20, window_snapshot_steps=5, soft_cells_per_debye=2.5,
                                   min_accumulated_macro_particle_steps_at_peak=40)
    new = Simulation(_static_config(grid, new_gate, macro_weight=w), zero_field_map(grid), backend=backend)
    new.load_state(state)
    new.run(15, accumulate_from_step=0)                                       # window incomplete: recorded, not enforced
    for record in new.series:
        window = record.peak_node["window"]
        assert window["window_complete"] is False and window["gate_enforced"] is False
        assert window["min_accumulated_macro_particle_steps_at_peak"] == 40.0 and "occupancy_floor_peak" in window
    with pytest.raises(PIC2DStabilityError, match=r"peak-node Debye gate \(window\)"):
        new.run(10, accumulate_from_step=0)
    tripped = new.series[-1].peak_node["window"]
    assert tripped["window_complete"] and tripped["gate_enforced"] and tripped["node"][0] == 0 and tripped["cells_per_debye"] > pi
    assert tripped["mean_macro_particles_at_peak"] < 32 and tripped["accumulated_macro_particle_steps_at_peak"] >= 40.0
    assert tripped["occupancy_floor_peak"]["resolved"] is False                # the witness: the old floor saw nothing
    assert tripped["soft_exceeded"] is True
    # the single-step witness is still recorded and still not enforced in window mode
    assert new.series[-1].peak_node["gate_mode"] == "window" and new.series[-1].peak_node["gate_enforced"] is False


def test_cpu_and_warp_cpu_read_the_same_accumulated_floor_statistic():
    pytest.importorskip("warp")
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    gate = PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=32, window_steps=20, window_snapshot_steps=5, soft_cells_per_debye=1e6,
                               min_accumulated_macro_particle_steps_at_peak=40)
    w = _axis_over_dense_weight(grid, 3, 2.0, 1.3 * pi)
    readings = {}
    for backend in ("cpu", "warp-cpu"):
        sim = Simulation(_static_config(grid, gate, macro_weight=w), zero_field_map(grid), backend=backend)
        sim.load_state(_axis_state(grid, per_node=3, nodes=range(36, 44), t_ev=2.0))
        sim.run(25, accumulate_from_step=0)
        readings[backend] = sim.series[-1].peak_node["window"]
    a, b = readings["cpu"], readings["warp-cpu"]
    assert a["node"] == b["node"] and a["resolved_nodes"] == b["resolved_nodes"] > 0 and a["gate_enforced"] and b["gate_enforced"]
    for key in ("cells_per_debye", "n_e_peak_per_m3", "t_e_peak_ev", "mean_macro_particles_at_peak", "accumulated_macro_particle_steps_at_peak"):
        assert a[key] == pytest.approx(b[key], rel=1e-9), key
    assert a["occupancy_floor_peak"]["resolved_nodes"] == b["occupancy_floor_peak"]["resolved_nodes"]


# -- recorded maps: the accepted v4 plateau passes, attempt 8 trips, the ext-val axis nodes become resolved ---------------

def _masks_from_summary(summary: dict) -> object:
    grid = summary["provenance"]["config"]["grid"]
    geometry = grid["geometry"]
    extra = {key: geometry[key] for key in ("plume_radius_m", "plume_length_m", "body_dielectric_radius_m") if key in geometry}
    return build_mesh_masks(Grid2D(ChannelGeometry(geometry["bore_radius_m"], geometry["z_min_m"], geometry["z_max_m"], geometry["cone_start_z_m"],
                                                   geometry["exit_radius_m"], **extra), grid["radial_cells"], grid["axial_cells"]))


def _map_statistics(results: Path) -> tuple[dict, dict, dict]:
    """(v2.0.3 reading, v2.0.6 reading, maps) from a recorded final-window ``maps.npz`` (sample_count_e = sum of weights)."""

    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    masks = _masks_from_summary(summary)
    maps = np.load(results / "maps.npz")
    steps = int(maps["window_steps"][0])
    w = maps["sample_count_e"]
    sums = {"n_e": maps["n_e_per_m3"] * steps, "e_weight": w, "e_vr": np.zeros_like(w), "e_vt": np.zeros_like(w), "e_vz": np.zeros_like(w),
            "e_v2": w * maps["t_e_ev"] * 3.0 * EV_J / ELECTRON_MASS_KG}
    old = window_peak_debye(masks, None, sums, steps, min_mean_occupancy=32.0)
    new = window_peak_debye(masks, None, sums, steps, min_mean_occupancy=32.0, min_accumulated_particle_steps=float(PEAK_DEBYE_GATE_DEFAULT_ACCUMULATED_FLOOR))
    return old, new, {"steps": steps, "occupancy": w / steps, "masks": masks}


V4 = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results"
ATTEMPT8 = MODERN / "experiments" / "pic2d_cft_plume_v1" / "results-attempt8-grid-heating-triad-stop"
EXT_VAL = MODERN / "experiments" / "pic2d_external_validation_v0" / "results" / "channel-20um-launch1-triad-gate-stop"


@pytest.mark.skipif(not (V4 / "maps.npz").is_file(), reason="ss-v4 maps not checked out")
def test_accepted_v4_plateau_still_passes_under_the_accumulated_floor():
    old, new, info = _map_statistics(V4)
    assert old["cells_per_debye"] == pytest.approx(2.154, abs=0.01) and new["cells_per_debye"] == pytest.approx(2.154, abs=0.01)
    assert new["node"] == old["node"] == [20, 429] and new["cells_per_debye"] <= 2.5 < pi           # soft margin holds, hard gate silent
    assert new["resolved_nodes"] > 2 * old["resolved_nodes"]                                          # 42 130 vs 19 650 nodes
    assert new["occupancy_floor_peak"]["cells_per_debye"] == pytest.approx(old["cells_per_debye"])
    # the densest axis node (0.38 macro-electrons per step, 151 661 macro-electron-steps) IS resolved under the accumulated floor
    # - the occupancy floor never saw it - and reads 0.79 cells per lambda_D: the 33 um plateau's axis is far from pi
    n_e = np.load(V4 / "maps.npz")["n_e_per_m3"]
    t_e = np.load(V4 / "maps.npz")["t_e_ev"]
    j = int(np.argmax(n_e[0, :]))
    axis_acc = info["occupancy"][0, j] * info["steps"]
    assert info["occupancy"][0, j] < 1.0 and axis_acc > PEAK_DEBYE_GATE_DEFAULT_ACCUMULATED_FLOOR
    assert max(info["masks"].grid.dr_m, info["masks"].grid.dz_m) / _debye(n_e[0, j], t_e[0, j]) < 1.0


@pytest.mark.skipif(not (ATTEMPT8 / "maps.npz").is_file(), reason="attempt-8 maps not checked out")
def test_attempt_8_final_window_trips_under_both_floors():
    old, new, _ = _map_statistics(ATTEMPT8)
    assert old["cells_per_debye"] > pi and new["cells_per_debye"] > pi                               # 3.61 at node (14, 285)
    assert new["cells_per_debye"] == pytest.approx(3.608, abs=0.01) and new["node"] == old["node"]


@pytest.mark.skipif(not (EXT_VAL / "maps.npz").is_file(), reason="ext-val maps not checked out")
def test_ext_val_end_state_map_resolves_the_near_axis_nodes_the_occupancy_floor_excluded():
    old, new, info = _map_statistics(EXT_VAL)
    assert old["resolved_nodes"] < 10_000 and new["resolved_nodes"] > 40_000                          # 8 843 -> 42 373 nodes
    assert new["mean_macro_particles_at_peak"] < 32 <= old["mean_macro_particles_at_peak"]           # the gated node was invisible before
    assert new["accumulated_macro_particle_steps_at_peak"] > 1e6 and new["node"][0] < old["node"][0]  # closer to the axis
    axis_acc = info["occupancy"][0, :] * info["steps"]
    assert axis_acc.max() > 1.5e5 > PEAK_DEBYE_GATE_DEFAULT_ACCUMULATED_FLOOR                       # 172 000 macro-electron-steps on the axis
    # the axis column of THIS 240 000-step window average reads 2.18 (the diagnosis' 28-ns frames read 2.9-3.3 at the stop): the
    # accumulated floor makes the column gate-able; the window average's lag on a 0.24-us-doubling density is a separate limitation
    assert 2.0 < new["cells_per_debye"] < 2.5 and abs(new["cells_per_debye"] - old["cells_per_debye"]) < 0.1


# -- runner ---------------------------------------------------------------------------------------------------------------

def _v206_protocol() -> dict:
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["case"].update({"radial_cells": 12, "axial_cells": 96, "macro_weight": 2.0e6, "seed": 3})
    protocol["operating_point"].update({"neutral_density_per_m3": 1.0e21, "electron_injection_current_a": 0.05, "seed_plasma_density_per_m3": 3.0e16})
    protocol["numerics"].update({
        "dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 40, "averaging_window_steps": 80, "ion_subcycle": 1,
        "peak_debye_gate": {"max_cells_per_debye": 50.0, "min_macro_particles_at_peak": 4, "dense_fraction": 0.5, "window_steps": 80,
                            "window_snapshot_steps": 40, "soft_cells_per_debye": 40.0, "min_accumulated_macro_particle_steps_at_peak": 100,
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


def test_runner_passes_the_floor_through_and_records_the_gated_and_witness_statistics(tmp_path: Path):
    protocol = _v206_protocol()
    config = runner.build_config(protocol, backend="cpu")
    gate = config.peak_debye_gate
    assert gate.accumulated_floor and gate.min_accumulated_macro_particle_steps_at_peak == 100
    assert config.to_dict()["peak_debye_gate"]["min_accumulated_macro_particle_steps_at_peak"] == 100
    results = tmp_path / "v206"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=linear_psi_field_map(config.grid, 2.0), cross_sections=XenonCrossSections.from_file(),
                            max_steps=160, log=lambda _: None)
    series = np.load(results / "series.npz")
    for key in ("peak_node_window_cells_per_debye", "peak_node_window_accumulated_macro_particle_steps_at_peak", "peak_node_window_occupancy_floor_cells_per_debye",
                "interval_inelastic_loss_j"):
        assert key in series.files and series[key].size == 8, key
    assert np.all(np.isfinite(series["peak_node_window_accumulated_macro_particle_steps_at_peak"]))
    assert np.all(np.isfinite(series["peak_node_window_occupancy_floor_cells_per_debye"])) and np.all(np.isfinite(series["interval_inelastic_loss_j"]))
    # the gated statistic reads at least the witness: the accumulated floor admits a superset of the occupancy floor's nodes once the
    # window holds >= 100 / 4 = 25 steps (4 per step at the occupancy floor)
    complete = series["peak_node_window_window_steps"] >= 25
    assert np.all(series["peak_node_window_cells_per_debye"][complete] >= series["peak_node_window_occupancy_floor_cells_per_debye"][complete] - 1e-12)
    samples = [json.loads(line) for line in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    window = [s for s in samples if "event" not in s][-1]["peak_node"]["window"]
    assert window["min_accumulated_macro_particle_steps_at_peak"] == 100.0 and set(window["occupancy_floor_peak"]) >= {"cells_per_debye", "resolved_nodes", "node"}
    summary = artifacts.read_canonical_json(results / "summary.json")
    assert summary["peak_node_debye"]["gate"]["min_accumulated_macro_particle_steps_at_peak"] == 100
    assert summary["provenance"]["config"]["peak_debye_gate"]["min_accumulated_macro_particle_steps_at_peak"] == 100
    # without the key the protocol builds the v2.0.3 gate (identity unchanged) and the witness arrays are NaN
    protocol["numerics"]["peak_debye_gate"].pop("min_accumulated_macro_particle_steps_at_peak")
    v203 = runner.build_config(protocol, backend="cpu")
    assert not v203.peak_debye_gate.accumulated_floor and "min_accumulated_macro_particle_steps_at_peak" not in v203.to_dict()["peak_debye_gate"]
    records = [json.loads(line) for line in (results / "series.jsonl").read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["peak_node"]["window"].pop("occupancy_floor_peak")
        record["peak_node"]["window"].pop("accumulated_macro_particle_steps_at_peak")
    arrays = runner.records_to_arrays(records)
    assert np.all(np.isnan(arrays["peak_node_window_occupancy_floor_cells_per_debye"])) and np.all(np.isnan(arrays["peak_node_window_accumulated_macro_particle_steps_at_peak"]))
