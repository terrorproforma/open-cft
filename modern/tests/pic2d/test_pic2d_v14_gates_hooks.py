"""Model v1.4: peak-node Debye gate, wall-ion recycling in the simulation, sensitivity hooks.

* the peak-node sample reproduces a known uniform plasma (density, T_e, lambda_D, cells per
  lambda_D) on both backends and is recorded at every series record;
* the gate fails closed (PIC2DStabilityError) when cells per lambda_D at the peak exceed the
  declared value, is recorded-only below the particle floor, and leaves v1.3 config
  identities unchanged when absent;
* the simulation's recycled source equals wall + anode ion absorption x W / interval and the
  ledger closes with the recycled term; gross and net utilisation are recorded;
* Bohm scattering: speed preserved to round-off, the number of scatterings matches
  alpha omega_ce dt N within counting statistics, energy ledger untouched, alpha bracket
  values accepted, default OFF (no "anomalous" tally without the hook);
* SEE scaffold: Vaughan yield shape (zero below threshold, delta_max at E_max, first crossover),
  the virtual wall-yield diagnostic, and enabled=True refused by the configuration.
"""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import uniform_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
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
from cft_revival.pic2d.neutrals import NEUTRAL_LEDGER_KEYS, NeutralInventoryConfig, feed_for_density
from cft_revival.pic2d.sensitivity import (
    BN_VAUGHAN,
    BOHM_ALPHA_BRACKET,
    AnomalousCollisionConfig,
    SEEConfig,
    apply_bohm_scattering,
    bohm_collision_probability,
    first_crossover_ev,
    vaughan_yield,
    virtual_wall_yield,
)
from cft_revival.pic2d.simulation import (
    InjectionConfig,
    PIC2DConfig,
    PeakDebyeGateConfig,
    SeedPlasmaConfig,
    Simulation,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
EXIT_AREA = pi * 3.0e-3**2


def _config(grid: Grid2D, **overrides) -> PIC2DConfig:
    base = dict(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=4.0), series_interval_steps=20,
    )
    return PIC2DConfig(**(base | overrides))


def _debye(n: float, t_ev: float) -> float:
    return sqrt(EPSILON_0_F_PER_M * t_ev * EV_J / (n * ELEMENTARY_CHARGE_C**2))


def test_peak_node_sample_reproduces_the_seed_plasma_and_is_recorded():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    n0, t0 = 1e17, 5.0
    config = _config(grid, seed_plasma=SeedPlasmaConfig(n0, t0), injection=None, mcc=None, macro_weight=2e5,
                     peak_debye_gate=PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=64))
    sim = Simulation(config, field, cross_sections=xs)
    sample = sim.backend.peak_node_sample()
    # uniform seed: among nodes with >= 64 particles the peak is a ~10 % fluctuation of the uniform density,
    # T_e is the seed temperature; the unrestricted maximum is single-particle noise on an axis node
    assert 1.0 < sample["n_e_peak_per_m3"] / n0 < 1.6
    assert sample["raw_peak"]["n_e_per_m3"] > sample["n_e_peak_per_m3"] and sample["raw_peak"]["macro_particles"] < 64
    assert sample["t_e_peak_ev"] == pytest.approx(t0, rel=0.35)
    assert sample["t_e_dense_ev"] == pytest.approx(t0, rel=0.1)
    assert sample["macro_particles_at_peak"] >= 64 and sample["min_particles_for_peak"] == 64
    lam = _debye(sample["n_e_peak_per_m3"], sample["t_e_peak_ev"])
    assert sample["debye_length_m"] == pytest.approx(lam)
    assert sample["cells_per_debye"] == pytest.approx(max(grid.dr_m, grid.dz_m) / lam)
    assert sample["dz_per_debye"] == pytest.approx(grid.dz_m / lam) and sample["dr_per_debye"] == pytest.approx(grid.dr_m / lam)
    i, j = sample["node"]
    assert sample["r_m"] == pytest.approx(i * grid.dr_m) and sample["z_m"] == pytest.approx(j * grid.dz_m)
    sim.run(20)
    record = sim.series[-1]
    assert record.peak_node is not None and record.peak_node["cells_per_debye"] > 0.0
    assert record.peak_node["gate_max_cells_per_debye"] == 1e6 and record.peak_node["gate_enforced"] is True
    assert record.to_dict()["peak_node"]["node"] == record.peak_node["node"]
    plain = _config(grid)
    assert "peak_debye_gate" not in plain.to_dict()                  # v1.3 identity unchanged
    sim = Simulation(plain, field, cross_sections=xs)
    sim.run(20)
    assert "gate_max_cells_per_debye" not in sim.series[-1].peak_node   # no gate configured: recorded only


def test_peak_debye_gate_fails_closed_and_records_the_threshold():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    dense = dict(seed_plasma=SeedPlasmaConfig(1e17, 5.0), macro_weight=2e5)
    loose = _config(grid, peak_debye_gate=PeakDebyeGateConfig(1e6, min_macro_particles_at_peak=32), **dense)
    sim = Simulation(loose, field, cross_sections=xs)
    sim.run(20)
    peak = sim.series[-1].peak_node
    assert peak["gate_max_cells_per_debye"] == 1e6 and peak["gate_enforced"] is True
    assert loose.to_dict()["peak_debye_gate"] == {"max_cells_per_debye": 1e6, "min_macro_particles_at_peak": 32, "dense_fraction": 0.5}
    # a threshold below the observed ratio trips the gate at the first record
    tight = _config(grid, peak_debye_gate=PeakDebyeGateConfig(0.5 * peak["cells_per_debye"], min_macro_particles_at_peak=32), **dense)
    with pytest.raises(PIC2DStabilityError, match="peak-node Debye gate"):
        Simulation(tight, field, cross_sections=xs).run(20)
    # below the particle floor the gate is recorded but not enforced
    floor = _config(grid, peak_debye_gate=PeakDebyeGateConfig(0.5 * peak["cells_per_debye"], min_macro_particles_at_peak=10**6), **dense)
    sim = Simulation(floor, field, cross_sections=xs)
    sim.run(20)
    assert sim.series[-1].peak_node["gate_enforced"] is False
    with pytest.raises(PIC2DValidationError):
        PeakDebyeGateConfig(0.0)
    with pytest.raises(PIC2DValidationError):
        PeakDebyeGateConfig(2.0, dense_fraction=0.0)


def test_simulation_recycles_wall_and_anode_ions_into_the_inventory():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    feed = feed_for_density(5e20, EXIT_AREA, 300.0)
    inventory = NeutralInventoryConfig(feed, 1e-9, wall_recycling=True, wall_temperature_k=500.0)
    # dense seed with hot ions (50 eV: 8.6 km/s) so that the ions born within ~4 um of the wall reach it in 100 steps
    config = _config(grid, neutral_inventory=inventory, seed_plasma=SeedPlasmaConfig(1.5e17, 5.0, 50.0), macro_weight=2e5)
    sim = Simulation(config, field, cross_sections=xs)
    sim.run(60)
    records = sim.series
    absorbed = [r.neutral["wall_ion_absorption_rate_per_s"] for r in records]
    assert any(a > 0.0 for a in absorbed), "ions must reach the wall/anode within the run"
    previous = 0.0
    for record in records:
        cum = record.ledger["cumulative"]
        interval = 20 * config.dt_s
        expected = ((cum["wall_ions"] + cum["anode_ions"]) - previous) * config.macro_weight / interval
        previous = cum["wall_ions"] + cum["anode_ions"]
        assert record.neutral["wall_ion_absorption_rate_per_s"] == pytest.approx(expected, rel=1e-12, abs=1e-6)
        assert record.neutral["recycled_rate_per_s"] == pytest.approx(expected, rel=1e-12, abs=1e-6)    # gamma = 1
        s = record.neutral["ionization_rate_per_s"]
        assert record.neutral["gross_utilisation"] == pytest.approx(s / feed)
        assert record.neutral["net_utilisation"] == pytest.approx((s - expected) / feed)
        assert record.neutral["effusion_coefficient_m3_per_s"] >= sim.neutrals.effusion_coefficient   # hotter recycled atoms effuse faster
    ledger = records[-1].neutral["ledger"]
    assert set(ledger) == set(NEUTRAL_LEDGER_KEYS) and ledger["recycled"] > 0.0
    volume = sim.neutrals.volume_m3
    closure = ledger["fed"] + ledger["recycled"] - ledger["ionized"] - ledger["effused"] - ledger["artificial"]
    assert closure == pytest.approx(volume * (records[-1].neutral["density_per_m3"] - 1e21), abs=1e-9 * volume * 1e21)
    provenance = sim.to_provenance()
    assert provenance["neutral_inventory"]["wall_recycling"] is True and provenance["neutral_inventory"]["wall_temperature_k"] == 500.0
    assert provenance["config"]["neutral_inventory"]["wall_recycling"] is True
    assert provenance["v1_4_options"]["peak_debye_gate"] is None and provenance["v1_4_options"]["step_graph"] is False


def test_bohm_scattering_preserves_speed_and_matches_the_rate():
    rng = np.random.default_rng(5)
    n = 200_000
    v = rng.normal(0.0, 1e6, size=(3, n))
    b = 0.05
    dt = 5e-12
    alpha = 1.0 / 16.0
    omega_ce = ELEMENTARY_CHARGE_C * b / ELECTRON_MASS_KG
    p = float(bohm_collision_probability(alpha, b, dt))
    assert p == pytest.approx(1.0 - np.exp(-alpha * omega_ce * dt))
    vr, vt, vz, count = apply_bohm_scattering(alpha, v[0], v[1], v[2], np.full(n, b), dt, rng)
    assert abs(count - n * p) <= 5.0 * sqrt(n * p)
    speed_before = np.sqrt((v**2).sum(axis=0))
    speed_after = np.sqrt(vr**2 + vt**2 + vz**2)
    np.testing.assert_allclose(speed_after, speed_before, rtol=1e-12)
    changed = (vr != v[0]) | (vt != v[1]) | (vz != v[2])
    assert int(changed.sum()) == count
    # the scattered directions are isotropic: mean cos(theta) ~ 0
    cos_t = vz[changed] / speed_after[changed]
    assert abs(float(cos_t.mean())) < 5.0 / sqrt(max(count, 1))
    assert BOHM_ALPHA_BRACKET == (1.0 / 64.0, 1.0 / 16.0)
    for alpha in BOHM_ALPHA_BRACKET:
        assert AnomalousCollisionConfig(alpha).to_dict()["alpha"] == alpha
    with pytest.raises(PIC2DValidationError):
        AnomalousCollisionConfig(0.0)


def test_bohm_hook_in_the_simulation_is_off_by_default_and_tallied_when_on(tmp_path):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    field = uniform_field_map(grid, 0.05)
    off = Simulation(_config(grid), field, cross_sections=xs)
    off.run(20)
    assert "anomalous" not in off.state.cumulative and "anomalous_collision_rate_per_s" not in off.series[-1].currents_a
    assert "anomalous" not in _config(grid).to_dict()
    config = _config(grid, anomalous=AnomalousCollisionConfig(1.0 / 16.0))
    on = Simulation(config, field, cross_sections=xs)
    on.run(20)
    hits = on.state.cumulative["anomalous"]
    n_e = on.state.electrons.count
    omega_ce = ELEMENTARY_CHARGE_C * 0.05 / ELECTRON_MASS_KG
    expected = 20 * n_e * (1.0 - np.exp(-omega_ce * config.dt_s / 16.0))
    assert abs(hits - expected) <= 5.0 * sqrt(expected) + 0.05 * expected     # counts drift as N_e changes over 20 steps
    assert on.series[-1].currents_a["anomalous_collision_rate_per_s"] == pytest.approx(hits * config.macro_weight / (20 * config.dt_s))
    assert config.to_dict()["anomalous"] == {"model": "bohm_isotropic_scattering", "alpha": 1.0 / 16.0}
    # the tally survives a checkpoint round trip (extra ledger keys are hash-bound like the rest)
    json_path, _ = artifacts.save_checkpoint(tmp_path, "an", on.state, config, field_sha256=field.sha256,
                                             cross_section_sha256=xs.payload_sha256, backend="cpu")
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    assert loaded.cumulative["anomalous"] == hits
    # energy ledger: scattering is a pure rotation, the interval residual stays at the collisionless level
    assert abs(on.series[-1].ledger["interval_residual_j"]) < 1e-3 * abs(on.series[-1].ledger["total_energy_j"])


def test_see_scaffold_vaughan_yield_and_virtual_wall_diagnostic():
    """v1.4 scaffold behaviour kept under model v2.2.0: the Vaughan curve shape (with the v1.4 BN constants as explicit
    overrides), the first crossover, the angular factor and the virtual per-column yield diagnostic.  ``enabled=True``
    is no longer refused (emission is implemented: tests/pic2d/test_pic2d_v22_see.py)."""

    see = SEEConfig(enabled=False, material="BN", overrides={**BN_VAUGHAN, "low_energy_elastic_peak": 0.0, "k_rise": 0.56})
    assert see.enabled is False and see.material == "BN"
    assert float(see.yield_at(5.0)) == 0.0                                       # below the (overridden) threshold
    assert float(see.yield_at(BN_VAUGHAN["energy_max_ev"])) == pytest.approx(BN_VAUGHAN["delta_max"])   # the component shares are inside the total
    e = np.linspace(13.0, 2000.0, 400)
    d = see.yield_at(e)
    assert np.all(np.diff(d[e < BN_VAUGHAN["energy_max_ev"]]) > 0.0)              # rising branch
    assert np.all(np.diff(d[e > BN_VAUGHAN["energy_max_ev"]]) < 0.0)              # falling branch
    crossover = first_crossover_ev(see)
    assert 20.0 < crossover < 60.0 and float(see.yield_at(crossover)) == pytest.approx(1.0, abs=1e-6)
    # oblique incidence raises the yield (Vaughan angular factor)
    assert float(see.yield_at(100.0, 1.0)) > float(see.yield_at(100.0, 0.0))
    assert float(vaughan_yield(100.0, delta_max=1.0, energy_max_ev=300.0, energy_threshold_ev=12.5)) < 1.0
    # virtual wall yield from column fluxes: 30 eV mean impact is below the crossover, 200 eV is above
    flux = np.array([0.0, 1e20, 1e20])
    energy_flux = flux * np.array([0.0, 30.0, 200.0]) * ELEMENTARY_CHARGE_C
    virtual = virtual_wall_yield(see, flux, energy_flux)
    assert virtual["yield_per_column"][0] == 0.0
    assert virtual["yield_per_column"][1] < 1.0 < virtual["yield_per_column"][2]
    assert virtual["columns_above_space_charge_limit"] == 1
    assert virtual["flux_weighted_yield"] == pytest.approx(0.5 * (virtual["yield_per_column"][1] + virtual["yield_per_column"][2]))
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    assert _config(grid, see=see).to_dict()["see"]["enabled"] is False
    assert "see" not in _config(grid).to_dict()
    assert _config(grid, see=SEEConfig(enabled=True)).see_active is True
    with pytest.raises(PIC2DValidationError):
        SEEConfig(overrides={"energy_threshold_ev": 400.0})
