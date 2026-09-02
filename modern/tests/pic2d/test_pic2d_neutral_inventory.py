"""Model v1.3: quasi-steady 0-D neutral inventory.

* atom ledger closes to round-off on every update: fed - ionised - effused - artificial
  = V (n_g,1 - n_g,0), also across many steps with a varying prescribed S;
* the fixed point of the integrator equals the analytic balance n_g* = (Q_in - S)/c
  and the artificial term vanishes there; the approach rate is 1/tau_g + c/V;
* the MCC real-collision rate scales with n_g / n_g0 on the CPU reference (and the
  GPU kernel receives the scaled density): ionisation and elastic counts at
  scale 0.5 are half those at scale 1 within counting statistics;
* fail-closed: feed above the null-collision ceiling, scale > 1, exhaustion;
* the neutral state is in the checkpoint (hash-bound arrays), a resume is bitwise
  identical to the uninterrupted run including n_g and the ledgers, and a
  checkpoint without the inventory cannot be loaded into an inventory config.
"""

from __future__ import annotations

from math import exp, pi
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map
from cft_revival.pic2d.mcc import MCCConfig, NullCollisionMCC, XenonCrossSections
from cft_revival.pic2d.models import (
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    ParticleArrays,
    PIC2DStabilityError,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
    xenon_ion_species,
)
from cft_revival.pic2d.neutrals import (
    NEUTRAL_LEDGER_KEYS,
    NeutralInventory,
    NeutralInventoryConfig,
    NeutralState,
    effusion_coefficient_m3_per_s,
    feed_for_density,
    mass_flow_mg_per_s,
    mean_thermal_speed_m_per_s,
)
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
EXIT_AREA = pi * 3.0e-3**2
VOLUME = 3.43e-7


def _inventory(feed: float, tau: float = 3.0e-8, ceiling: float = 5.0e19, temperature: float = 300.0) -> NeutralInventory:
    return NeutralInventory(
        NeutralInventoryConfig(feed, tau), ceiling_density_per_m3=ceiling, exit_area_m2=EXIT_AREA, temperature_k=temperature, volume_m3=VOLUME
    )


def test_effusion_coefficient_and_feed_numbers():
    v_bar = mean_thermal_speed_m_per_s(300.0)
    assert v_bar == pytest.approx(220.0, rel=1e-3)              # xenon at 300 K: sqrt(8 k T / (pi m))
    c = effusion_coefficient_m3_per_s(EXIT_AREA, 300.0)
    assert c == pytest.approx(0.25 * v_bar * EXIT_AREA)
    feed = feed_for_density(5.0e19, EXIT_AREA, 300.0)
    assert feed == pytest.approx(5.0e19 * c)
    assert mass_flow_mg_per_s(feed) == pytest.approx(feed * 2.1801714e-25 * 1e6)
    inventory = _inventory(feed)
    assert inventory.zero_ionization_density == pytest.approx(5.0e19)
    assert inventory.physical_time_constant_s == pytest.approx(VOLUME / c)
    assert 1e-4 < inventory.physical_time_constant_s < 1e-3   # ~0.2 ms: why the relaxation is artificial


def test_ledger_closes_to_roundoff_with_varying_source():
    feed = feed_for_density(5.0e19, EXIT_AREA, 300.0)
    inventory = _inventory(feed)
    state = NeutralState.initial(5.0e19)
    rng = np.random.default_rng(0)
    for k in range(500):
        s = float(rng.uniform(0.0, 0.9 * feed)) if k % 7 else 0.0
        before = state
        result = inventory.advance(state, s, 3.0e-10)
        state = result.state
        balance = (
            (state.ledger["fed"] - before.ledger["fed"]) - (state.ledger["ionized"] - before.ledger["ionized"])
            - (state.ledger["effused"] - before.ledger["effused"]) - (state.ledger["artificial"] - before.ledger["artificial"])
        )
        assert abs(balance - VOLUME * (state.density_per_m3 - before.density_per_m3)) <= 1e-9 * VOLUME * 5.0e19
        assert abs(result.ledger_residual_atoms) <= 1e-9 * VOLUME * 5.0e19
    # cumulative closure over the whole history
    total = state.ledger["fed"] - state.ledger["ionized"] - state.ledger["effused"] - state.ledger["artificial"]
    assert total == pytest.approx(VOLUME * (state.density_per_m3 - 5.0e19), abs=1e-9 * VOLUME * 5.0e19)
    assert 0.0 < state.density_per_m3 <= 5.0e19 * (1 + 1e-12)


def test_fixed_point_matches_analytic_balance_and_rate():
    feed = feed_for_density(5.0e19, EXIT_AREA, 300.0)
    tau = 3.0e-8
    inventory = _inventory(feed, tau)
    s = 0.3 * feed
    n_star = (feed - s) / inventory.effusion_coefficient
    assert inventory.fixed_point(s) == pytest.approx(n_star)
    assert n_star == pytest.approx(3.5e19)
    state = NeutralState.initial(5.0e19)
    dt = 3.0e-10
    rate = 1.0 / tau + inventory.effusion_coefficient / VOLUME
    # one step: exact exponential approach with the documented rate
    first = inventory.advance(state, s, dt)
    assert first.state.density_per_m3 == pytest.approx(n_star + (5.0e19 - n_star) * exp(-rate * dt), rel=1e-12)
    initial_artificial = abs(first.artificial_rate_per_s)
    assert initial_artificial > 0.1 * feed                          # the transient is dominated by the artificial term
    for _ in range(2000):        # 600 ns = 20 tau
        state = inventory.advance(state, s, dt).state
    assert state.density_per_m3 == pytest.approx(n_star, rel=1e-8)
    result = inventory.advance(state, s, dt)
    assert abs(result.artificial_rate_per_s) <= 1e-8 * initial_artificial   # vanishes at the fixed point (e^-20)
    assert result.effusion_rate_per_s == pytest.approx(feed - s, rel=1e-8)   # feed = ionisation + effusion
    assert result.state.ledger["artificial"] > 0.0                  # the transient removed atoms artificially
    assert inventory.scale(result.state) == pytest.approx(0.7)


def test_fail_closed_feed_above_ceiling_scale_and_exhaustion():
    with pytest.raises(PIC2DValidationError):
        _inventory(feed_for_density(5.1e19, EXIT_AREA, 300.0), ceiling=5.0e19)
    with pytest.raises(PIC2DValidationError):
        NeutralInventoryConfig(0.0, 1e-8)
    with pytest.raises(PIC2DValidationError):
        NeutralInventoryConfig(1e17, 0.0)
    xs = XenonCrossSections.from_file()
    mcc = NullCollisionMCC(xs, MCCConfig(1e21), xenon_ion_species(1e6))
    with pytest.raises(PIC2DValidationError):
        mcc.set_neutral_scale(1.5)
    inventory = _inventory(feed_for_density(5.0e19, EXIT_AREA, 300.0), tau=1e-12)
    with pytest.raises(PIC2DStabilityError):
        inventory.advance(NeutralState.initial(1.0e17), 1e3 * inventory.config.feed_atoms_per_s, 1e-9)


def test_mcc_rates_scale_with_neutral_density():
    xs = XenonCrossSections.from_file()
    mcc = NullCollisionMCC(xs, MCCConfig(1e21), xenon_ion_species(1e6))
    n = 400_000
    speed = 6.0e6 * np.ones(n)   # ~100 eV electrons: well above the ionisation threshold
    electrons = ParticleArrays(np.full(n, 1e-3), np.full(n, 1e-2), speed, np.zeros(n), np.zeros(n))
    counts = {}
    for scale in (1.0, 0.5):
        mcc.set_neutral_scale(scale)
        result = mcc.apply(electrons, 2e-11, np.random.default_rng(11))
        counts[scale] = (result.tally.ionization, result.tally.elastic, result.tally.excitation, result.tally.candidates)
    # the candidate count (null ceiling) is unchanged; every real channel halves within 5 sigma
    assert counts[1.0][3] == counts[0.5][3]
    for i in range(3):
        full, half = counts[1.0][i], counts[0.5][i]
        assert full > 500
        assert abs(half - 0.5 * full) <= 5.0 * np.sqrt(0.5 * full)
    assert mcc.neutral_density_per_m3 == pytest.approx(5e20)


def _config(grid: Grid2D, *, inventory: NeutralInventoryConfig | None, series: int = 20) -> PIC2DConfig:
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=4.0), series_interval_steps=series, neutral_inventory=inventory,
    )


def test_simulation_updates_inventory_and_records_it():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    # a feed that would sustain only 1/3 of the ceiling: n_g must fall every interval
    feed = feed_for_density(1e21 / 3, EXIT_AREA, 300.0)
    config = _config(grid, inventory=NeutralInventoryConfig(feed, 1e-9))
    sim = Simulation(config, field, cross_sections=xs)
    sim.run(100)
    records = sim.series
    assert len(records) == 5 and all(r.neutral is not None for r in records)
    n_g = [r.neutral["density_per_m3"] for r in records]
    assert all(a > b for a, b in zip(n_g, n_g[1:]))
    assert records[-1].neutral["scale"] == pytest.approx(n_g[-1] / 1e21)
    assert sim.backend.mcc.neutral_scale == pytest.approx(n_g[-1] / 1e21)
    ledger = records[-1].neutral["ledger"]
    assert set(ledger) == set(NEUTRAL_LEDGER_KEYS)
    volume = sim.neutrals.volume_m3
    closure = ledger["fed"] - ledger["ionized"] - ledger["effused"] - ledger["artificial"]
    assert closure == pytest.approx(volume * (n_g[-1] - 1e21), abs=1e-9 * volume * 1e21)
    # the ionised atoms equal the MCC tally (macro-particles x weight)
    assert ledger["ionized"] == pytest.approx(records[-1].ledger["cumulative"]["ionizations"] * config.macro_weight, rel=1e-12)
    assert sim.state.neutral is not None and sim.state.neutral.density_per_m3 == n_g[-1]
    provenance = sim.to_provenance()
    assert provenance["neutral_inventory"]["transient_is_artificial"] is True
    assert provenance["config"]["neutral_inventory"]["feed_atoms_per_s"] == feed
    assert "neutral_inventory" not in _config(grid, inventory=None).to_dict()   # v1.0-1.2 identities unchanged


def test_checkpoint_carries_neutral_state_and_resume_is_bitwise(tmp_path: Path):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    feed = feed_for_density(5e20, EXIT_AREA, 300.0)
    config = _config(grid, inventory=NeutralInventoryConfig(feed, 1e-9))
    reference = Simulation(config, field, cross_sections=xs)
    reference.run(40)
    first = Simulation(config, field, cross_sections=xs)
    first.run(20)
    json_path, npz_path = artifacts.save_checkpoint(
        tmp_path, "ckpt", first.state, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256, backend="cpu"
    )
    arrays = np.load(npz_path)
    assert "neutral" in arrays.files and arrays["neutral"].shape == (5,)
    assert artifacts.read_canonical_json(json_path)["neutral_keys"] == ["density_per_m3", *NEUTRAL_LEDGER_KEYS]
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    assert loaded.neutral is not None and loaded.neutral.density_per_m3 == first.state.neutral.density_per_m3
    resumed = Simulation(config, field, cross_sections=xs)
    resumed.load_state(loaded)
    assert resumed.backend.mcc.neutral_scale == pytest.approx(loaded.neutral.density_per_m3 / 1e21)
    resumed.run(20)
    a, b = reference.state, resumed.state
    assert a.neutral.density_per_m3 == b.neutral.density_per_m3
    assert a.neutral.ledger == b.neutral.ledger
    assert a.electrons.count == b.electrons.count and a.ions.count == b.ions.count
    assert np.array_equal(a.electrons.r_m, b.electrons.r_m) and np.array_equal(a.ions.vz_m_per_s, b.ions.vz_m_per_s)
    assert reference.series[-1].neutral == resumed.series[-1].neutral
    # tampering with n_g in the arrays breaks the hash (fail closed)
    tampered = dict(arrays)
    tampered["neutral"] = arrays["neutral"] * 1.01
    np.savez(npz_path, **tampered)
    with pytest.raises(PIC2DValidationError):
        artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)


def test_static_checkpoint_cannot_load_into_inventory_config(tmp_path: Path):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    static = _config(grid, inventory=None)
    sim = Simulation(static, field, cross_sections=xs)
    sim.run(20)
    json_path, _ = artifacts.save_checkpoint(
        tmp_path, "ckpt", sim.state, static, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256, backend="cpu"
    )
    assert artifacts.read_canonical_json(json_path)["neutral_keys"] is None
    with_inventory = _config(grid, inventory=NeutralInventoryConfig(feed_for_density(5e20, EXIT_AREA, 300.0), 1e-9))
    with pytest.raises(PIC2DValidationError):
        artifacts.load_checkpoint(json_path, with_inventory, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    with pytest.raises(PIC2DValidationError):
        Simulation(with_inventory, field, cross_sections=xs).load_state(sim.state)
    with pytest.raises(PIC2DValidationError):
        PIC2DConfig(
            grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6,
            neutral_inventory=NeutralInventoryConfig(1e17, 1e-9),
        )


def _warp_cuda() -> bool:
    try:
        from cft_revival.pic2d.warp_backend import device_available
    except ImportError:  # pragma: no cover
        return False
    return device_available("cuda:0")


@pytest.mark.skipif(not _warp_cuda(), reason="CUDA Warp device unavailable")
def test_gpu_backend_applies_the_same_neutral_scale():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = linear_psi_field_map(grid, 2.0)
    feed = feed_for_density(1e21 / 3, EXIT_AREA, 300.0)
    config = _config(grid, inventory=NeutralInventoryConfig(feed, 1e-9))
    cpu = Simulation(config, field, cross_sections=xs, backend="cpu")
    gpu = Simulation(config, field, cross_sections=xs, backend="warp-cuda")
    cpu.run(20)
    gpu.run(20)
    # the inventory update is host-side and deterministic given the ionisation count;
    # both backends fed the scaled density to their MCC after the first interval
    assert gpu.backend.mcc.neutral_scale < 1.0
    assert gpu.series[-1].neutral is not None
    assert gpu.series[-1].neutral["fixed_point_per_m3"] == pytest.approx(cpu.series[-1].neutral["fixed_point_per_m3"], rel=0.5)
    assert gpu.state.neutral.density_per_m3 == pytest.approx(gpu.series[-1].neutral["density_per_m3"])
