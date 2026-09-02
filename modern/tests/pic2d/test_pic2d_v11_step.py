"""v1.1 all-GPU step: device direct solve parity, ion subcycling, electrode-work ledger.

Every new piece of the v1.1 time step has a test here:

* the device block-Thomas solve (``poisson.method = "device-direct"``) equals the
  host direct solve to round-off on the first step and keeps its true-residual
  contract at every host sync;
* per-block tile reductions in the push/MCC kernels reproduce the exact
  absorbed/collision tallies of the numpy reference (counts are exact integers);
* ion subcycling ``k`` vs ``2k`` leaves the ion population and the field
  statistically unchanged (ions move a small fraction of a cell per ``k`` steps);
* the energy ledger closes once electrode work is tracked: the cumulative
  residual is a small fraction of the electrode-work term that dominates the
  energy budget of a 300 V discharge.
"""

from __future__ import annotations

import numpy as np
import pytest

from cft_revival.pic2d.fields import linear_psi_field_map
from cft_revival.pic2d.models import (
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)


def _config(grid: Grid2D, *, method: str = "direct", ion_subcycle: int = 1, series: int = 20, injection: bool = True) -> PIC2DConfig:
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0) if injection else None, seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=None,
        poisson=PoissonConfig2D(method=method), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=series, ion_subcycle=ion_subcycle,
    )


def _warp_cuda() -> bool:
    try:
        from cft_revival.pic2d.warp_backend import device_available
    except ImportError:  # pragma: no cover
        return False
    return device_available("cuda:0")


needs_cuda = pytest.mark.skipif(not _warp_cuda(), reason="CUDA Warp device unavailable")


@needs_cuda
def test_device_direct_matches_host_direct_to_roundoff():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    cpu = Simulation(_config(grid, method="direct"), field, backend="cpu")
    gpu = Simulation(_config(grid, method="device-direct"), field, backend="warp-cuda")
    cpu.run(1)
    gpu.run(1)
    a, b = cpu.state, gpu.state
    assert a.electrons.count == b.electrons.count and a.ions.count == b.ions.count
    scale = float(np.max(np.abs(a.phi_v)))
    assert float(np.max(np.abs(a.phi_v - b.phi_v))) <= 1e-10 * scale
    # the particles saw the same field: positions agree to round-off after the first push
    assert np.max(np.abs(a.electrons.r_m - b.electrons.r_m)) <= 1e-15
    assert np.max(np.abs(a.electrons.z_m - b.electrons.z_m)) <= 1e-15


@needs_cuda
def test_device_direct_residual_contract_enforced_at_sync():
    from cft_revival.pic2d.warp_backend import WarpBlockThomas

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    sim = Simulation(_config(grid, method="device-direct"), field, backend="warp-cuda")
    backend = sim.backend
    assert isinstance(backend.device_direct, WarpBlockThomas)
    sim.run(2 * sim.config.sync_steps)
    backend.device_direct.queue_residual_check()
    residual, tolerance = backend.device_direct.verify()
    assert residual <= tolerance
    # the direct solve is far inside the tolerance (it is exact to round-off)
    assert residual <= 1e-3 * tolerance


@needs_cuda
def test_tile_reduced_tallies_are_exact_integers():
    """Absorbed-particle counts from the block-reduced GPU kernels equal the numpy reference."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    # no injection: the injected sample differs between the numpy and Philox streams
    cpu = Simulation(_config(grid, method="direct", series=10, injection=False), field, backend="cpu")
    gpu = Simulation(_config(grid, method="direct", series=10, injection=False), field, backend="warp-cuda")
    cpu.run(10)
    gpu.run(10)
    a = cpu.series[-1].ledger["cumulative"]
    b = gpu.series[-1].ledger["cumulative"]
    absorbed = 0.0
    for key in ("anode_electrons", "exit_electrons", "wall_electrons", "anode_ions", "exit_ions", "wall_ions"):
        assert a[key] == b[key], key
        assert float(b[key]).is_integer()
        absorbed += b[key]
    assert absorbed > 0.0
    # the block-reduced work sum agrees with the sequential numpy sum to round-off of the gross work
    gross = float(cpu.series[-1].kinetic_electron_j)
    assert abs(a["field_work_j"] - b["field_work_j"]) <= 1e-9 * gross


def test_ion_subcycling_k_vs_2k_is_insensitive():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    results = {}
    for k in (4, 8):
        sim = Simulation(_config(grid, ion_subcycle=k, series=40), field, backend="cpu")
        sim.run(400)
        state = sim.state
        results[k] = (
            state.ions.count,
            float(np.mean(state.ions.z_m)),
            float(np.mean(state.ions.r_m)),
            float(np.mean(state.phi_v)),
            float(sim.series[-1].kinetic_ion_j),
        )
    a, b = results[4], results[8]
    assert a[0] == b[0]  # no ion reached a boundary differently
    assert abs(a[1] - b[1]) <= 1e-6 * grid.geometry.z_max_m
    assert abs(a[2] - b[2]) <= 1e-6 * grid.geometry.bore_radius_m
    assert abs(a[3] - b[3]) <= 1e-3 * abs(a[3])
    # ion velocities are staggered by k*dt/2: cold seed ions accelerating from rest
    # differ in kinetic energy by ~ (k dt)/t = 1 % after 2 ns; positions agree above
    assert abs(a[4] - b[4]) <= 2.5e-2 * max(abs(a[4]), 1e-30)


def test_ion_subcycle_is_recorded_in_config_provenance():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _config(grid, ion_subcycle=8, series=40)
    assert config.to_dict()["ion_subcycle"] == 8
    assert config.sync_steps % 1 == 0 and config.sync_steps >= 1


def test_ledger_closes_with_electrode_work():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    sim = Simulation(_config(grid, series=20), field, backend="cpu")
    sim.run(200)
    series = sim.series
    total_change = series[-1].ledger["total_energy_j"] - series[0].ledger["total_energy_j"]
    residual = sum(r.ledger["interval_residual_j"] for r in series[1:])
    electrode = sum(r.ledger["interval_electrode_work_j"] for r in series[1:])
    assert electrode > 0.0
    # without the electrode term the ledger would miss the whole 300 V supply work
    assert abs(electrode) > 0.5 * abs(total_change)
    # with it the cumulative residual is a small fraction of the budget
    assert abs(residual) <= 0.15 * abs(total_change)
    assert abs(residual) <= 0.15 * abs(electrode)
    assert all(np.isfinite(r.ledger["anode_induced_charge_c"]) for r in series)
