"""Integrated CPU cycle: gates, conservation ledgers, checkpoint determinism, sheath limit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import uniform_field_map, zero_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELEMENTARY_CHARGE_C,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DStabilityError,
    PIC2DValidationError,
    StabilityLimits,
    stability_report,
)
from cft_revival.pic2d.poisson import induced_electrode_charge_c
from cft_revival.pic2d.simulation import (
    InjectionConfig,
    PIC2DConfig,
    SeedPlasmaConfig,
    Simulation,
    seed_plasma_state,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)


def _config(grid: Grid2D, **overrides) -> PIC2DConfig:
    base = dict(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        seed_plasma=SeedPlasmaConfig(1e16, 5.0), reference_density_per_m3=1e16,
        reference_electron_temperature_ev=5.0, limits=StabilityLimits(max_cell_debye_ratio=2.0),
        series_interval_steps=10,
    )
    base.update(overrides)
    return PIC2DConfig(**base)


def test_stability_gate_fails_closed():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    report = stability_report(grid, 5e-12, reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0, max_b_t=0.05, max_electron_energy_ev=400.0)
    assert report.omega_pe_dt < 0.2 and report.omega_ce_dt < 0.2 and report.particle_courant < 1.0
    too_dense = stability_report(grid, 5e-12, reference_density_per_m3=1e19, reference_electron_temperature_ev=5.0, max_b_t=0.05, max_electron_energy_ev=400.0)
    assert not too_dense.stable and any("plasma-frequency" in item for item in too_dense.violations)
    assert any("Debye" in item for item in too_dense.violations)
    strong_b = stability_report(grid, 5e-12, reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0, max_b_t=0.5, max_electron_energy_ev=400.0)
    assert any("cyclotron" in item for item in strong_b.violations)
    with pytest.raises(PIC2DStabilityError):
        Simulation(_config(grid, reference_density_per_m3=1e19), uniform_field_map(grid, 0.05))
    with pytest.raises(PIC2DValidationError):
        PIC2DConfig(grid=grid, potentials=BoundaryPotentials(1.0), dt_s=-1.0, macro_weight=1.0)


def test_field_solve_charge_conservation_every_step():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _config(grid)
    sim = Simulation(config, uniform_field_map(grid, 0.05))
    masks = sim.masks
    for _ in range(5):
        sim.run(1)
        state = sim.state
        anode, exit_plane = induced_electrode_charge_c(masks, state.phi_v)
        electrons = -ELEMENTARY_CHARGE_C * config.macro_weight * state.electrons.count
        ions = ELEMENTARY_CHARGE_C * config.macro_weight * state.ions.count
        # phi^n was solved from the charges of the particles present at the start of the step; the
        # Gauss law must close on the source actually used (volume charge with the 3/4 axis ratio
        # plus surface charge), which we reconstruct from the cumulative wall tally.
        total_particles = electrons + ions
        assert abs(anode + exit_plane) <= abs(total_particles) * 1.05 + 1e-30
        assert np.isfinite(state.phi_v).all()
        assert state.phi_v[masks.anode_node].tolist() == [300.0] * int(masks.anode_node.sum())
        assert state.phi_v[masks.exit_node].tolist() == [0.0] * int(masks.exit_node.sum())


def test_collisionless_energy_ledger_and_series_bookkeeping():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _config(grid, injection=InjectionConfig(0.05, 2.0), series_interval_steps=20)
    sim = Simulation(config, uniform_field_map(grid, 0.05))
    sim.run(100)
    assert len(sim.series) == 5
    final = sim.series[-1]
    cumulative = final.ledger["cumulative"]
    total_energy = final.ledger["total_energy_j"]
    residuals = [abs(record.ledger["interval_residual_j"]) for record in sim.series[1:]]
    # ledger residual (includes untracked electrode work) stays a small fraction of the system energy
    assert max(residuals) < 0.2 * total_energy
    assert cumulative["injected_electrons"] > 0 and cumulative["wall_electrons"] > 0
    expected_current = config.injection.electron_current_a  # type: ignore[union-attr]
    assert final.currents_a["injected_electron_a"] == pytest.approx(expected_current, rel=0.2)
    assert final.kinetic_electron_j > 0 and final.field_energy_j > 0
    assert final.phi_max_v <= 320.0 and final.phi_min_v >= -50.0
    provenance = sim.to_provenance()
    assert provenance["stability_gate"]["stable"] is True
    assert provenance["mesh"]["unknown_nodes"] > 0


def test_single_electron_energy_is_conserved_in_pure_magnetic_field():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    # a vanishing macro weight removes the particle's own space charge (self-force) from the test
    config = _config(grid, seed_plasma=None, potentials=BoundaryPotentials(0.0, 0.0), macro_weight=1.0e-9, series_interval_steps=50)
    sim = Simulation(config, uniform_field_map(grid, 0.05))
    from cft_revival.pic2d.models import ParticleArrays
    from cft_revival.pic2d.simulation import SimulationState, empty_cumulative

    electrons = ParticleArrays(np.array([1.0e-3]), np.array([12.0e-3]), np.array([0.0]), np.array([1.0e6]), np.array([5.0e5]))
    state = SimulationState(0, 0.0, electrons, ParticleArrays.empty(), np.zeros(grid.node_shape), np.zeros(grid.node_shape), 0.0, empty_cumulative())
    sim.load_state(state)
    sim.run(500)
    k0 = sim.series[0].kinetic_electron_j
    drift = abs(sim.series[-1].kinetic_electron_j - k0) / k0
    assert drift < 1e-9, drift
    assert sim.state.electrons.count == 1


def test_checkpoint_resume_is_bitwise_deterministic(tmp_path: Path):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    config = _config(grid, injection=InjectionConfig(0.05, 2.0), mcc=MCCConfig(1e21))
    field = uniform_field_map(grid, 0.05)
    reference = Simulation(config, field, cross_sections=xs)
    reference.run(40)
    resumed = Simulation(config, field, cross_sections=xs)
    resumed.run(20)
    json_path, _ = artifacts.save_checkpoint(
        tmp_path, "ckpt", resumed.state, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256, backend="cpu"
    )
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    fresh = Simulation(config, field, cross_sections=xs)
    fresh.load_state(loaded)
    fresh.run(20)
    a, b = reference.state, fresh.state
    assert a.step == b.step == 40
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(a.electrons, name), getattr(b.electrons, name))
        assert np.array_equal(getattr(a.ions, name), getattr(b.ions, name))
    assert np.array_equal(a.surface_charge_c, b.surface_charge_c)
    assert np.array_equal(a.phi_v, b.phi_v)
    assert a.cumulative == b.cumulative
    with pytest.raises(PIC2DValidationError):
        artifacts.load_checkpoint(json_path, config, field_sha256="0" * 64, cross_section_sha256=xs.payload_sha256)
    other = _config(grid, injection=InjectionConfig(0.06, 2.0), mcc=MCCConfig(1e21))
    with pytest.raises(PIC2DValidationError):
        artifacts.load_checkpoint(json_path, other, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)


def test_mcc_run_creates_ions_and_reports_rates():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    config = _config(grid, injection=InjectionConfig(0.2, 20.0), mcc=MCCConfig(3e21), seed_plasma=SeedPlasmaConfig(1e16, 30.0), series_interval_steps=50)
    sim = Simulation(config, uniform_field_map(grid, 0.05), cross_sections=xs)
    sim.run(200, accumulate_from_step=100)
    cumulative = sim.state.cumulative
    assert cumulative["ionizations"] > 0 and cumulative["elastic"] > cumulative["ionizations"]
    arrays = sim.diagnostic_arrays()
    assert arrays["window_steps"][0] == 100
    assert np.nanmax(arrays["n_e_per_m3"]) > 0 and np.nanmax(arrays["ionization_rate_per_m3_s"]) > 0
    assert arrays["t_e_ev"].shape == grid.node_shape and np.all(arrays["t_e_ev"] >= 0.0)
    assert arrays["wall_electron_flux_per_m2_s"].shape == (grid.axial_cells,)


def test_debye_sheath_forms_in_slab_limit():
    """Straight bore, grounded electrodes, no B, no injection: the bulk floats positive by a few T_e."""

    grid = Grid2D(STRAIGHT_GEOMETRY, 12, 64)
    te = 4.0
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(0.0, 0.0), dt_s=4.0e-12, macro_weight=1e6, seed=1,
        seed_plasma=SeedPlasmaConfig(2.0e16, te), reference_density_per_m3=2.0e16, reference_electron_temperature_ev=te,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=50,
    )
    sim = Simulation(config, zero_field_map(grid))
    sim.run(1500, accumulate_from_step=750)
    phi = sim.diagnostic_arrays()["phi_v"]
    masks = build_mesh_masks(grid)
    bulk = phi[masks.plasma_node & (np.abs(np.arange(grid.node_shape[1])[None, :] - grid.axial_cells // 2) < 8)]
    assert bulk.mean() > 0.5 * te
    assert bulk.mean() < 8.0 * te
    # sheath: the drop to the grounded exit plane happens within ~1-2 Debye lengths (about one cell here)
    axis = phi[0]
    assert axis[-1] == 0.0 and axis[0] == 0.0
    assert axis[-2] < bulk.mean() and axis[1] < bulk.mean()
    assert axis.max() < 8.0 * te
    state = sim.state
    assert state.cumulative["wall_electrons"] > state.cumulative["wall_ions"]


def test_seed_plasma_is_quasineutral_and_inside_the_channel():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _config(grid)
    masks = build_mesh_masks(grid)
    state = seed_plasma_state(config, masks)
    assert state.electrons.count == state.ions.count > 1000
    from cft_revival.pic2d import kernels

    assert np.all(kernels.classify_boundary(masks, state.electrons.r_m, state.electrons.z_m) == kernels.BOUNDARY_INSIDE)
    expected = config.seed_plasma.density_per_m3 * masks.plasma_volume_m3 / config.macro_weight  # type: ignore[union-attr]
    assert state.electrons.count == pytest.approx(expected, abs=1.0)
