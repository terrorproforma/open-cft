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


def _recycling_inventory(feed: float, *, tau: float | None = 3.0e-8, gamma: float = 1.0, t_wall: float | None = None) -> NeutralInventory:
    return NeutralInventory(
        NeutralInventoryConfig(feed, tau, wall_recycling=True, recombination_coefficient=gamma, wall_temperature_k=t_wall),
        ceiling_density_per_m3=5.5e19, exit_area_m2=EXIT_AREA, temperature_k=300.0, volume_m3=VOLUME,
    )


def test_v14_recycling_ledger_closes_and_fixed_point_includes_the_wall_source():
    """fed + recycled - ionised - effused - artificial = V dn (round-off), n_g* = (Q_in + R - S)/c."""

    feed = feed_for_density(5.5e19, EXIT_AREA, 300.0)
    inventory = _recycling_inventory(feed)
    assert NEUTRAL_LEDGER_KEYS == ("fed", "ionized", "effused", "artificial", "recycled")
    state = NeutralState.initial(5.5e19)
    rng = np.random.default_rng(1)
    for k in range(400):
        s = float(rng.uniform(0.2, 0.7) * feed)
        r = float(rng.uniform(0.3, 0.7)) * s                      # 30-70 % of the ionisation hits the wall
        before = state
        result = inventory.advance(state, s, 3.0e-10, r)
        state = result.state
        d = {key: state.ledger[key] - before.ledger[key] for key in NEUTRAL_LEDGER_KEYS}
        closure = d["fed"] + d["recycled"] - d["ionized"] - d["effused"] - d["artificial"]
        assert abs(closure - VOLUME * (state.density_per_m3 - before.density_per_m3)) <= 1e-9 * VOLUME * 5.5e19
        assert abs(result.ledger_residual_atoms) <= 1e-9 * VOLUME * 5.5e19
        assert result.recycled_rate_per_s == pytest.approx(r)
        assert result.fixed_point_per_m3 == pytest.approx((feed + r - s) / inventory.effusion_coefficient)
    assert state.ledger["recycled"] > 0.0
    # the plateau numbers of steady-state v2: S = 3.93e16 /s, wall+anode ions 2.355e16 /s -> n_g* rises from 2.97e19 to 4.49e19
    v2 = NeutralInventory(NeutralInventoryConfig(8.551102004120011e16, 3e-8, wall_recycling=True),
                          ceiling_density_per_m3=5.5e19, exit_area_m2=EXIT_AREA, temperature_k=300.0, volume_m3=3.432268513863189e-07)
    assert v2.fixed_point(3.93e16) == pytest.approx(2.97e19, rel=0.01)
    assert v2.fixed_point(3.93e16, 2.355e16) == pytest.approx(4.49e19, rel=0.01)


def test_v14_recycling_converges_to_the_recycled_fixed_point_with_gross_and_net_utilisation():
    feed = feed_for_density(5.5e19, EXIT_AREA, 300.0)
    inventory = _recycling_inventory(feed)
    s, r = 0.5 * feed, 0.3 * feed
    state = NeutralState.initial(5.5e19)
    for _ in range(3000):        # 900 ns = 30 tau
        state = inventory.advance(state, s, 3.0e-10, r).state
    result = inventory.advance(state, s, 3.0e-10, r)
    n_star = (feed + r - s) / inventory.effusion_coefficient
    assert result.state.density_per_m3 == pytest.approx(n_star, rel=1e-8)
    assert result.effusion_rate_per_s == pytest.approx(feed + r - s, rel=1e-8)   # feed + recycled = ionisation + effusion
    assert abs(result.artificial_rate_per_s) <= 1e-8 * feed
    gross, net = s / feed, (s - r) / feed
    assert gross == pytest.approx(0.5) and net == pytest.approx(0.2)
    # recombination coefficient scales the source; recycling off ignores it entirely
    half = _recycling_inventory(feed, gamma=0.5).advance(NeutralState.initial(5.5e19), s, 3.0e-10, r)
    assert half.recycled_rate_per_s == pytest.approx(0.5 * r)
    off = _inventory(feed, ceiling=5.5e19).advance(NeutralState.initial(5.5e19), s, 3.0e-10, r)
    assert off.recycled_rate_per_s == 0.0 and off.fixed_point_per_m3 == pytest.approx((feed - s) / inventory.effusion_coefficient)


def test_v14_wall_temperature_mixture_effusion_and_relaxation_off():
    feed = feed_for_density(5.5e19, EXIT_AREA, 300.0)
    hot = _recycling_inventory(feed, t_wall=500.0)
    c_g, c_w = hot.effusion_coefficient, hot.wall_effusion_coefficient
    assert c_w / c_g == pytest.approx((500.0 / 300.0) ** 0.5)
    r = 0.25 * feed
    assert hot.effective_effusion_coefficient(0.0) == c_g
    assert hot.effective_effusion_coefficient(r) == pytest.approx((feed * c_g + r * c_w) / (feed + r))
    assert hot.fixed_point(0.4 * feed, r) == pytest.approx((feed + r - 0.4 * feed) / hot.effective_effusion_coefficient(r))
    result = hot.advance(NeutralState.initial(5.5e19), 0.4 * feed, 3.0e-10, r)
    assert result.effusion_coefficient_m3_per_s == pytest.approx(hot.effective_effusion_coefficient(r))
    assert hot.to_dict()["wall_temperature_k"] == 500.0 and hot.to_dict()["wall_recycling"] is True
    with pytest.raises(PIC2DValidationError):
        NeutralInventoryConfig(feed, 1e-8, wall_temperature_k=500.0)          # wall temperature needs recycling
    with pytest.raises(PIC2DValidationError):
        NeutralInventoryConfig(feed, 1e-8, wall_recycling=True, recombination_coefficient=1.5)
    # relaxation OFF: the physical effusion time scale only (rate c/V), artificial ledger stays zero
    physical = _recycling_inventory(feed, tau=None)
    assert physical.relaxation_on is False and physical.to_dict()["transient_is_artificial"] is False
    state = NeutralState.initial(5.5e19)
    s = 0.3 * feed
    result = physical.advance(state, s, 1.0e-6, 0.0)
    rate = physical.effusion_coefficient / VOLUME
    n_star = (feed - s) / physical.effusion_coefficient
    assert result.state.density_per_m3 == pytest.approx(n_star + (5.5e19 - n_star) * exp(-rate * 1.0e-6), rel=1e-12)
    assert result.artificial_rate_per_s == 0.0 and result.state.ledger["artificial"] == 0.0
    assert abs(result.ledger_residual_atoms) <= 1e-9 * VOLUME * 5.5e19
    # after 1 us the physical inventory has moved < 1 % of the way: why a plateau without relaxation needs >> us
    assert abs(result.state.density_per_m3 - 5.5e19) < 0.01 * abs(n_star - 5.5e19)


def test_v14_state_array_reads_v13_and_v14_layouts():
    v13 = NeutralState.from_array(np.array([3.0e19, 1.0, 2.0, 3.0, 4.0]))
    assert v13.ledger == {"fed": 1.0, "ionized": 2.0, "effused": 3.0, "artificial": 4.0, "recycled": 0.0}
    v14 = NeutralState(2.0e19, {"fed": 1.0, "ionized": 2.0, "effused": 3.0, "artificial": 4.0, "recycled": 5.0})
    assert NeutralState.from_array(v14.to_array()).ledger == v14.ledger
    with pytest.raises(PIC2DValidationError):
        NeutralState.from_array(np.array([1.0, 2.0]))
    # v1.3 config identity is unchanged (no recycling keys)
    assert NeutralInventoryConfig(1e17, 3e-8).to_dict() == {"feed_atoms_per_s": 1e17, "relaxation_time_s": 3e-8}


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


def test_v20_artificial_relaxation_is_suspended_when_ionisation_exceeds_the_sources():
    """Plume run attempt 4 (2026-09-04): S peaked at 1.26 x Q for two 30 ns intervals; relaxing toward the NEGATIVE fixed
    point emptied the channel (5.5e19 -> 4e18) in one interval and the discharge collapsed. Without a fixed point the
    relaxation is suspended and the inventory follows the conservative balance (decay by (S - Q) dt / V only)."""

    n_g0 = 5.5e19
    feed = feed_for_density(n_g0, EXIT_AREA, 300.0)
    inventory = _inventory(feed, tau=3.0e-8, ceiling=2.0 * n_g0)
    state = NeutralState.initial(n_g0)
    dt = 3.0e-8
    burst = inventory.advance(state, 1.26 * feed, dt)
    assert burst.artificial_relaxation_suspended and burst.fixed_point_per_m3 < 0.0 and burst.artificial_rate_per_s == 0.0
    expected_loss = ((1.26 * feed - feed) * dt + inventory.effusion_coefficient * n_g0 * dt) / VOLUME
    assert burst.state.density_per_m3 == pytest.approx(n_g0 - expected_loss, rel=1e-3)
    assert burst.state.density_per_m3 > 0.99 * n_g0            # ~0.1 ms to empty the channel, not 30 ns
    assert burst.state.ledger["artificial"] == 0.0
    assert abs(burst.ledger_residual_atoms) < 1e-6 * feed * dt
    # below the sources the relaxation acts as before
    calm = inventory.advance(burst.state, 0.5 * feed, dt)
    assert not calm.artificial_relaxation_suspended and calm.artificial_rate_per_s != 0.0
    # the old behaviour: with the relaxation toward n* < 0 the density would have collapsed in one interval
    assert (1.0 - exp(-dt / 3.0e-8)) > 0.6


def test_v20_recycling_transient_above_q_over_c_needs_ceiling_headroom():
    """Plume run launch 1 (2026-09-03): with recycling on and the ceiling AT Q/c, the seeded ions returning as atoms
    before the ionisation catches up (R > S) push n_g above Q/c and the run stops fail-closed. A ceiling with declared
    headroom and the inventory started at Q/c (initial_density_per_m3) carries the transient; identities of
    ceiling-started configs are unchanged."""

    n_g0 = 5.5e19
    feed = feed_for_density(n_g0, EXIT_AREA, 300.0)
    tau = 3.0e-8
    # ceiling at Q/c (v1.4 default): the recycled source with S = 0 makes n* = (Q + R)/c > Q/c -> fail-closed
    tight = NeutralInventory(NeutralInventoryConfig(feed, tau, wall_recycling=True, wall_temperature_k=400.0),
                             ceiling_density_per_m3=n_g0, exit_area_m2=EXIT_AREA, temperature_k=300.0, volume_m3=VOLUME)
    assert tight.initial_density == n_g0 and tight.ceiling_headroom == pytest.approx(1.0)
    state = NeutralState.initial(tight.initial_density)
    with pytest.raises(PIC2DStabilityError, match="exceeds the null-collision ceiling"):
        for _ in range(20):
            state = tight.advance(state, 0.0, 1.0e-9, recycled_ion_rate_per_s=2.0e16).state
    # ceiling 2 x Q/c, inventory started at Q/c: the relaxation tracks the rate-based fixed point (Q + R)/c_mix = 1.23 Q/c
    # (the launch-1 artefact), which now stays inside the ceiling; the MCC scale starts at 1/2 (real collision rate unchanged)
    roomy_config = NeutralInventoryConfig(feed, tau, wall_recycling=True, wall_temperature_k=400.0, initial_density_per_m3=n_g0)
    roomy = NeutralInventory(roomy_config, ceiling_density_per_m3=2.0 * n_g0, exit_area_m2=EXIT_AREA, temperature_k=300.0, volume_m3=VOLUME)
    assert roomy.ceiling_headroom == pytest.approx(2.0) and roomy.initial_density == n_g0
    state = NeutralState.initial(roomy.initial_density)
    assert roomy.scale(state) == pytest.approx(0.5)
    for _ in range(600):   # 20 relaxation times
        state = roomy.advance(state, 0.0, 1.0e-9, recycled_ion_rate_per_s=2.0e16).state
    assert 1.15 * n_g0 < state.density_per_m3 < 1.3 * n_g0
    assert state.density_per_m3 == pytest.approx(roomy.fixed_point(0.0, 2.0e16), rel=1e-6)
    # with R <= S (steady state: every recycled ion was ionised) the fixed point is below Q/c
    assert roomy.fixed_point(3.0e16, 2.0e16) < n_g0
    assert roomy.to_dict()["initial_density_per_m3"] == n_g0 and roomy.to_dict()["ceiling_headroom_over_zero_ionization"] == pytest.approx(2.0)
    # identity: the key appears only when set; an initial density above the ceiling is refused
    assert "initial_density_per_m3" not in NeutralInventoryConfig(feed, tau, wall_recycling=True).to_dict()
    assert roomy_config.to_dict()["initial_density_per_m3"] == n_g0
    with pytest.raises(PIC2DValidationError, match="exceeds the null-collision ceiling"):
        NeutralInventory(NeutralInventoryConfig(feed, tau, initial_density_per_m3=2.2 * n_g0), ceiling_density_per_m3=2.0 * n_g0,
                         exit_area_m2=EXIT_AREA, temperature_k=300.0, volume_m3=VOLUME)
    with pytest.raises(PIC2DValidationError):
        NeutralInventoryConfig(feed, tau, initial_density_per_m3=0.0)


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
    # v2.0: a declared start density below the ceiling sets the MCC scale from step 0 and the ledger closes from it
    below = _config(grid, inventory=NeutralInventoryConfig(feed, 1e-9, initial_density_per_m3=5e20))
    sim2 = Simulation(below, field, cross_sections=xs)
    assert sim2.neutral_state.density_per_m3 == 5e20 and sim2.backend.mcc.neutral_scale == pytest.approx(0.5)
    sim2.run(40)
    ledger2 = sim2.series[-1].neutral["ledger"]
    closure2 = ledger2["fed"] - ledger2["ionized"] - ledger2["effused"] - ledger2["artificial"]
    assert closure2 == pytest.approx(volume * (sim2.neutral_state.density_per_m3 - 5e20), abs=1e-9 * volume * 1e21)
    assert artifacts.config_identity(below) != artifacts.config_identity(config)


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
    assert "neutral" in arrays.files and arrays["neutral"].shape == (6,)   # density + 5 ledgers (v1.4 adds recycled)
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
