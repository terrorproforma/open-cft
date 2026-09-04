"""Model v2.0.4 (2026-09-04, found by the external-validation v0 launch-box preflight / shakedown at 20 um / W 8.2e4).

The runtime omega_pe dt gate used to read the peak over EVERY plasma node of the single-step electron deposit: on a
small-volume axis node ONE macro-electron reads 1.3e19 m^-3 at 20 um / W 82 467 (omega_pe dt 0.14 at 0.7 ps), two read
0.20 - the gate tripped on shot noise at 60 000 electrons in 53 000 nodes (peak 5.5e18 while the mean was 5e14) and stopped
the synthetic 12 M-particle timing seed at omega_pe dt 0.212 before its first step.  Same class as the plume-boundary lesson:
a max-over-nodes single-step deposit statistic is an extreme value decided by the smallest node.

v2.0.4: the gate statistic is the peak over the RESOLVED nodes (deposit >= the peak-Debye gate's macro-particle floor, 32
under the v2.0.3 gates; 16 without a gate); the unfloored peak is recorded alongside (``peak_omega_pe_dt_raw``).  Physics is
untouched (a run that never trips replays bitwise); only the stop decision and the recorded statistic change.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from cft_revival.pic2d.fields import zero_field_map
from cft_revival.pic2d.kernels import ParticleArrays
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DStabilityError,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.simulation import (
    PeakDebyeGateConfig,
    PIC2DConfig,
    Simulation,
    SimulationState,
    StepTally,
    empty_cumulative,
    omega_pe_gate_min_macro_particles,
    peak_deposit_densities,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)


def _omega_pe_dt(density: float, dt: float) -> float:
    return sqrt(density * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ELECTRON_MASS_KG)) * dt


def _config(grid: Grid2D, *, dt: float, macro_weight: float, gate: PeakDebyeGateConfig | None, limit: float = 0.2) -> PIC2DConfig:
    return PIC2DConfig(grid=grid, potentials=BoundaryPotentials(0.0, 0.0), dt_s=dt, macro_weight=macro_weight, seed=3, injection=None, seed_plasma=None, mcc=None,
                       poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e12, reference_electron_temperature_ev=5.0, max_electron_energy_ev=1.0,
                       limits=StabilityLimits(max_cell_debye_ratio=1e6, max_omega_pe_dt=limit), series_interval_steps=5, runtime_stability_check_steps=5, peak_debye_gate=gate)


def _state(grid: Grid2D, placements: list[tuple[tuple[int, int], int]]) -> SimulationState:
    """Cold electron/ion pairs placed exactly on nodes: ``placements`` = [((i, j), count), ...]."""

    r = np.concatenate([np.full(count, i * grid.dr_m) for (i, _), count in placements])
    z = np.concatenate([np.full(count, j * grid.dz_m) for (_, j), count in placements])
    zeros = np.zeros_like(r)
    electrons = ParticleArrays(r, z, zeros.copy(), zeros.copy(), zeros.copy())
    ions = ParticleArrays(r.copy(), z.copy(), zeros.copy(), zeros.copy(), zeros.copy())
    return SimulationState(step=0, time_s=0.0, electrons=electrons, ions=ions, surface_charge_c=np.zeros(grid.node_shape), phi_v=np.zeros(grid.node_shape),
                           injection_carry=0.0, cumulative=empty_cumulative())


def test_peak_deposit_densities_floors_the_gate_statistic_and_keeps_the_raw_peak():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    volume = masks.shape_volume_m3[masks.plasma_node]
    w = 8e4
    q = np.zeros(volume.size)
    tiny = int(np.argmin(volume))                    # the axis corner node
    big = int(np.argmax(volume))
    q[tiny] = 1.0 * ELEMENTARY_CHARGE_C * w          # one macro-electron on the smallest node
    q[big] = 40.0 * ELEMENTARY_CHARGE_C * w          # forty on the largest
    resolved, raw = peak_deposit_densities(q, volume, macro_weight=w, min_macro_particles=32)
    assert raw == pytest.approx(w / volume[tiny]) and resolved == pytest.approx(40.0 * w / volume[big]) and resolved < raw
    # no node at the floor -> the gate statistic is 0 while the raw peak is still recorded
    q[big] = 31.0 * ELEMENTARY_CHARGE_C * w
    resolved, raw = peak_deposit_densities(q, volume, macro_weight=w, min_macro_particles=32)
    assert resolved == 0.0 and raw > 0.0
    # exactly at the floor counts
    q[big] = 32.0 * ELEMENTARY_CHARGE_C * w
    assert peak_deposit_densities(q, volume, macro_weight=w, min_macro_particles=32)[0] == pytest.approx(32.0 * w / volume[big])
    assert peak_deposit_densities(np.zeros(0), np.zeros(0), macro_weight=w, min_macro_particles=32) == (0.0, 0.0)
    assert StepTally(1, 0.1, 0.0, 1, 1).max_omega_pe_dt_raw == 0.0            # pre-v2.0.4 constructor still valid


def test_floor_follows_the_peak_debye_gate_and_defaults_to_sixteen():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    assert omega_pe_gate_min_macro_particles(_config(grid, dt=1e-13, macro_weight=1e4, gate=None)) == 16
    gate = PeakDebyeGateConfig(3.14, min_macro_particles_at_peak=32, window_steps=400, window_snapshot_steps=40)
    cfg = _config(grid, dt=1e-13, macro_weight=1e4, gate=gate)
    assert omega_pe_gate_min_macro_particles(cfg) == 32
    sim = Simulation(cfg, zero_field_map(grid), backend="cpu")
    assert sim.to_provenance()["v1_4_options"]["omega_pe_dt_gate"] == {"statistic": "resolved_node_single_step_peak", "min_macro_particles": 32, "limit": 0.2}


def test_one_macro_electron_on_an_axis_node_no_longer_stops_the_run_but_is_recorded_raw():
    """The external-validation configuration class: a single macro-electron on the smallest node out-reads the gate."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    volumes = np.where(masks.plasma_node, masks.shape_volume_m3, np.inf)
    tiny = tuple(int(k) for k in np.unravel_index(int(np.argmin(volumes)), volumes.shape))
    w = 2e8                                                                  # on THIS coarse test grid (167 x 250 um) one macro-electron at 2e8 out-reads the gate
    dt = 1e-12
    one_particle_density = w / volumes[tiny]
    assert _omega_pe_dt(one_particle_density, dt) > 0.2                     # the raw statistic is over the limit with ONE particle
    gate = PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=32, window_steps=400, window_snapshot_steps=40)
    sim = Simulation(_config(grid, dt=dt, macro_weight=w, gate=gate), zero_field_map(grid), backend="cpu")
    sim.load_state(_state(grid, [(tiny, 1)]))
    sim.run(5, accumulate_from_step=0)                                       # v2.0.3 would have raised here
    record = sim.series[-1]
    assert record.peak_omega_pe_dt == 0.0 and record.peak_omega_pe_dt_raw > 0.2
    assert record.to_dict()["peak_omega_pe_dt_raw"] == record.peak_omega_pe_dt_raw and record.to_dict()["peak_omega_pe_dt"] == 0.0


def test_a_resolved_over_dense_node_still_stops_the_run_fail_closed_with_the_raw_witness():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    node = (6, 40)
    volume = masks.shape_volume_m3[node]
    w = 2e8
    count = 40                                                               # >= the 32 floor
    dt = 1e-12
    dense = count * w / volume
    assert _omega_pe_dt(dense, dt) > 0.2
    gate = PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=32, window_steps=400, window_snapshot_steps=40)
    sim = Simulation(_config(grid, dt=dt, macro_weight=w, gate=gate), zero_field_map(grid), backend="cpu")
    sim.load_state(_state(grid, [(node, count)]))
    with pytest.raises(PIC2DStabilityError, match=r"holding >= 32 macro-electrons \(raw single-node peak"):
        sim.run(5, accumulate_from_step=0)
    # the same node with 31 particles is below the floor: no stop, resolved statistic 0
    sim2 = Simulation(_config(grid, dt=dt, macro_weight=w, gate=gate), zero_field_map(grid), backend="cpu")
    sim2.load_state(_state(grid, [(node, 31)]))
    sim2.run(5, accumulate_from_step=0)
    assert sim2.series[-1].peak_omega_pe_dt == 0.0 and sim2.series[-1].peak_omega_pe_dt_raw == pytest.approx(_omega_pe_dt(31 * w / volume, dt), rel=1e-9)


def test_warp_cpu_backend_reports_the_same_resolved_and_raw_statistics():
    pytest.importorskip("warp")
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    volumes = np.where(masks.plasma_node, masks.shape_volume_m3, np.inf)
    tiny = tuple(int(k) for k in np.unravel_index(int(np.argmin(volumes)), volumes.shape))
    node = (6, 40)
    w = 2e8
    dt = 1e-12
    gate = PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=32, window_steps=400, window_snapshot_steps=40)
    placements = [(tiny, 1), (node, 35)]
    results = {}
    for backend in ("cpu", "warp-cpu"):
        cfg = _config(grid, dt=dt, macro_weight=w, gate=gate, limit=10.0)
        sim = Simulation(cfg, zero_field_map(grid), backend=backend)
        sim.load_state(_state(grid, placements))
        sim.run(5, accumulate_from_step=0)
        results[backend] = (sim.series[-1].peak_omega_pe_dt, sim.series[-1].peak_omega_pe_dt_raw)
    (res_cpu, raw_cpu), (res_warp, raw_warp) = results["cpu"], results["warp-cpu"]
    assert res_cpu > 0.0 and raw_cpu > res_cpu
    assert res_warp == pytest.approx(res_cpu, rel=1e-9) and raw_warp == pytest.approx(raw_cpu, rel=1e-9)
