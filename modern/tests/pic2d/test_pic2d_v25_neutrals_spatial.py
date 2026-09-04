"""Model v2.5.0 - ``neutrals_spatial_v1`` (test-particle neutrals) + ``metastables_v1`` (R5 of the physics audit).

* transport: the Clausing transmission probability of a diffuse tube and the Knudsen density gradient of a long tube;
* identity: ``inventory-0d`` replays origin head 8e02db57 bitwise on cpu and warp-cpu (pinned state hashes and identities);
  the spatial block enters ``config_sha256``; the protocol block builds the configuration;
* the atom ledger: exact removal of the plasma's demanded atoms with the debt carry; recycling spawned at the impact cell;
  the CEX hand-off from the ion MCC; the interval identity on cpu and warp-cpu with every channel active (plus the particle-side
  energy identity with the metastable channels); weight consistency of the ionisation sink against the plasma's own count;
* metastables: BEB / superelastic tables, the MCC channel rates against ``n sigma v dt N`` on a monoenergetic beam, production /
  wall de-excitation / radiative decay in the transport, the metastable ledger identity;
* checkpoint round trip + bitwise resume; diagnostic maps carry the neutral fields only when the model is on;
* CUDA (box only): graph vs direct bitwise with the spatial model + metastables on.
"""

from __future__ import annotations

import hashlib
import platform
from itertools import pairwise
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.cross_sections_xe import CollisionSetConfig
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map
from cft_revival.pic2d.ion_mcc import IonNeutralCrossSections, IonNeutralMCCConfig, IonNullCollisionMCC
from cft_revival.pic2d.mcc import EV_J, MCCConfig, NullCollisionMCC, XenonCrossSections, electron_speed_from_energy
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    BOLTZMANN_J_PER_K,
    XENON_MASS_KG,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DValidationError,
    ParticleArrays,
    PoissonConfig2D,
    StabilityLimits,
    xenon_ion_species,
)
from cft_revival.pic2d.neutrals import NeutralInventoryConfig, effusion_coefficient_m3_per_s, feed_for_density, mean_thermal_speed_m_per_s
from cft_revival.pic2d.neutrals_spatial import (
    NEUTRAL_SPATIAL_LEDGER_KEYS,
    STATE_GROUND,
    STATE_METASTABLE,
    CellSinks,
    MetastableConfig,
    MetastableProcessTable,
    SpatialNeutralConfig,
    SpatialNeutrals,
    beb_cross_section_m2,
    clausing_factor,
    knudsen_profile_per_m3,
)
from cft_revival.pic2d.simulation import DiagnosticAccumulator, InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation
from experiments.pic2d_cft_steady_state_v1 import run as runner

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
BRANCHING = (0.45, 0.35, 0.5, 0.35)
# origin head 8e02db57: the inventory-0d discharge (legacy set and xe_collision_set_v2) after 150 steps on cpu / warp-cpu
ORIGIN_PINS = {
    "legacy": {
        "config_sha256": "931a6a045dd3ead69658129328c04ca8f86af6f2686c894bee5e422c6cb75683",
        "cpu": ("4605199d9cd4e2e1b3642351308dc658472ebcaaa1c821ca35042edbcdacca3a", 1502, 1691, 21.0, 10.0),
        "warp-cpu": ("dd68d1f4fed34632749625e0a06bb39bf43a21516c8158214ea73980ff4cd9fe", 1498, 1690, 20.0, 17.0),
    },
    "xe_v2": {
        "config_sha256": "c269ab72085d4320e179db9613e4353c9eacb33a07b320443a78d69a0834ba84",
        "cpu": ("193888c28e4eab97572b9dfefe959242c3b2fb7e0e8ff4b67e0822560bc7027b", 1502, 1691, 21.0, 11.0),
        "warp-cpu": ("f7aefe74d42de2ff244b629975b533151d830fdb7462e27b95fbcea665f7dc7b", 1498, 1690, 20.0, 17.0),
    },
}


def _warp_backends() -> list[str]:
    try:
        from cft_revival.pic2d.warp_backend import device_available
    except ImportError:  # pragma: no cover
        return []
    return [name for name, device in (("warp-cpu", "cpu"), ("warp-cuda", "cuda:0")) if device_available(device)]


BACKENDS = ["cpu", *_warp_backends()]
CUDA = [b for b in BACKENDS if b == "warp-cuda"]
PARITY = [b for b in BACKENDS if b != "warp-cuda"]


@pytest.fixture(scope="module")
def collision_set() -> CollisionSetConfig:
    return CollisionSetConfig.xe_collision_set_v2()


@pytest.fixture(scope="module")
def xs(collision_set: CollisionSetConfig) -> XenonCrossSections:
    return collision_set.load_electron_cross_sections()


def _tube(length_over_diameter: float, nr: int = 8) -> tuple[Grid2D, float, float]:
    a = 1.0e-3
    length = length_over_diameter * 2.0 * a
    grid = Grid2D(ChannelGeometry(a, 0.0, length, length, a), nr, int(round(nr * length / a)))
    return grid, a, length


def _neutral_only(grid: Grid2D, feed: float, particles: float, *, acceleration: float, substeps: int = 1000, dt: float = 1e-12,
                  profile: str = "knudsen", **kwargs) -> tuple[SpatialNeutrals, CellSinks]:
    geometry = grid.geometry
    n_exit = feed / effusion_coefficient_m3_per_s(pi * geometry.exit_radius_m**2, 300.0)
    volume = pi * geometry.bore_radius_m**2 * geometry.channel_length_m
    config = SpatialNeutralConfig(feed, n_exit * volume / particles, substeps, time_acceleration=acceleration, initial_profile=profile, **kwargs)
    model = SpatialNeutrals(config, build_mesh_masks(grid), temperature_k=300.0, ceiling_density_per_m3=50.0 * n_exit, dt_s=dt, ion_macro_weight=1.0)
    return model, CellSinks.zeros(model.n_cells, 1.0)


def _spatial_config(grid: Grid2D, collision_set, *, metastables: bool = True, substep: int = 5, acceleration: float = 50.0, seed: int = 3,
                    feed_scale: float = 1.0, particles: float = 20000.0, **extra) -> PIC2DConfig:
    """The warp-parity discharge (300 V, injection, W 2e6) with spatial neutrals at a Knudsen profile below the ceiling."""

    geometry = grid.geometry
    feed = feed_scale * feed_for_density(1.0e20, pi * geometry.exit_radius_m**2, 300.0)
    n_exit = feed / effusion_coefficient_m3_per_s(pi * geometry.exit_radius_m**2, 300.0)
    volume = pi * geometry.bore_radius_m**2 * geometry.channel_length_m
    meta = MetastableConfig(BRANCHING) if metastables else None
    neutrals = SpatialNeutralConfig(feed, n_exit * volume / particles, substep, time_acceleration=acceleration, wall_temperature_k=500.0,
                                    metastables=meta)
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=seed,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0, 1.0), mcc=MCCConfig(2.0e21, collision_set=collision_set),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=4, neutrals_spatial=neutrals, **extra,
    )


def _inventory_config(grid: Grid2D, collision_set) -> PIC2DConfig:
    n_g = 1.0e21
    feed = feed_for_density(0.8 * n_g, pi * grid.geometry.exit_radius_m**2, 300.0)
    neutral = NeutralInventoryConfig(feed, None, wall_recycling=True, wall_temperature_k=500.0, initial_density_per_m3=0.9 * n_g)
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0, 1.0), mcc=MCCConfig(n_g, collision_set=collision_set),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=4, neutral_inventory=neutral,
    )


def _state_sha256(state) -> str:
    h = hashlib.sha256()
    for arr in (state.electrons.r_m, state.electrons.z_m, state.electrons.vr_m_per_s, state.electrons.vt_m_per_s, state.electrons.vz_m_per_s,
                state.ions.r_m, state.ions.z_m, state.ions.vr_m_per_s, state.ions.vt_m_per_s, state.ions.vz_m_per_s, state.phi_v, state.surface_charge_c):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# 1. free-molecular transport
# ----------------------------------------------------------------------------------------------------------------------

def test_clausing_conductance_of_a_diffuse_tube_within_statistics():
    """A tube closed by the anode and fed there transmits Q = K Phi_wall with K the Clausing factor: K = Q / (Q + anode re-emission)."""

    grid, a, length = _tube(2.0)
    feed = 1.0e16
    model, sinks = _neutral_only(grid, feed, 16000.0, acceleration=1.0, substeps=1, dt=0.4e-6)   # 0.4 us of neutral time per sub-step (~1 cell)
    rng = np.random.default_rng(1)
    state = model.initial_state(rng)
    transit = length / mean_thermal_speed_m_per_s(300.0)
    n_sub = int(round(30 * transit / model.substep_dt_s))
    anode_hits = fed = 0.0
    for k in range(n_sub):
        tally = model.substep(state, sinks, rng).values
        if k >= n_sub // 2:
            anode_hits += tally["neutral_anode_hits"]
            fed += tally["neutral_fed"]
    k_measured = fed / (fed + anode_hits)
    assert clausing_factor(2.0) == pytest.approx(0.3564)
    assert k_measured == pytest.approx(clausing_factor(2.0), rel=0.05), k_measured
    # the population is stationary and finite; everything fed eventually leaves (true count = particles + carries - debts)
    assert state.particles.count > 5000 and abs(state.true_ground_atoms() - state.particles.atoms(STATE_GROUND) - state.pending_feed.sum()) < 1e-6 * state.true_ground_atoms()


def test_knudsen_gradient_and_anode_to_exit_ratio_of_a_long_tube():
    """The steady density falls linearly with the Knudsen slope 3 Q / (2 pi a^3 v_bar) away from the ends (L/D = 4)."""

    grid, a, length = _tube(4.0, nr=6)
    feed = 1.0e16
    model, sinks = _neutral_only(grid, feed, 16000.0, acceleration=1.0, substeps=1, dt=0.6e-6)
    rng = np.random.default_rng(2)
    state = model.initial_state(rng)
    transit = length / mean_thermal_speed_m_per_s(300.0)
    n_sub = int(round(30 * transit / model.substep_dt_s))
    profile = np.zeros(model.nz)
    count = 0
    for k in range(n_sub):
        model.substep(state, sinks, rng)
        if k >= n_sub // 2:
            profile += np.mean(state.density_per_m3.reshape(model.nr, model.nz)[: model.nr // 2], axis=0)
            count += 1
    profile /= count
    z_mid = (np.arange(model.nz) + 0.5) * grid.dz_m
    inner = slice(model.nz // 4, 3 * model.nz // 4)
    slope = np.polyfit(z_mid[inner], profile[inner], 1)[0]
    knudsen_slope = -3.0 * feed / (2.0 * pi * a**3 * mean_thermal_speed_m_per_s(300.0))
    assert slope == pytest.approx(knudsen_slope, rel=0.25), (slope, knudsen_slope)   # end effects steepen a L/D = 4 tube by ~15 %
    # the anode density exceeds the exit density by at least the Knudsen factor ~1 + 3 L / (8 a); the last cell's density is below
    # the isotropic effusion estimate 4 Q / (v_bar A) because the exit distribution is forward-peaked (end correction), so the
    # measured ratio lies between the Knudsen estimate and ~2.5x it
    analytic = knudsen_profile_per_m3(grid.geometry, np.array([0.0, length]), feed, 300.0)
    assert analytic[0] / analytic[1] < profile[0] / profile[-1] < 2.5 * analytic[0] / analytic[1]
    assert profile[0] == pytest.approx(analytic[0], rel=0.15)        # the anode-end density itself follows the Knudsen profile
    assert clausing_factor(0.0) == 1.0 and clausing_factor(40.0) < clausing_factor(20.0) < clausing_factor(10.0)


# ----------------------------------------------------------------------------------------------------------------------
# 2. identity
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("backend", PARITY)
@pytest.mark.parametrize("label", ["legacy", "xe_v2"])
def test_inventory_0d_replays_origin_head_bitwise(backend: str, label: str, collision_set: CollisionSetConfig, xs: XenonCrossSections):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    cs = None if label == "legacy" else collision_set
    config = _inventory_config(grid, cs)
    assert config.neutral_model == "inventory-0d" and "neutrals_spatial" not in config.to_dict()
    assert artifacts.config_identity(config) == ORIGIN_PINS[label]["config_sha256"]
    sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=XenonCrossSections.from_file() if cs is None else xs)
    sim.run(150)
    state = sim.state
    assert state.neutral_particles is None and not any(key in state.cumulative for key in NEUTRAL_SPATIAL_LEDGER_KEYS)
    assert state.neutral is not None and "neutrals_spatial" not in sim.to_provenance()["config"]
    if platform.system() != "Windows":
        # the digests are anchor-platform pins (Windows / MSVC libm; Linux differs at ULP level in the same run - see the
        # cross-platform identity lesson): elsewhere the identity and the record layout are what this test checks
        pytest.skip("origin-head state digests are Windows anchor pins")
    digest, electrons, ions, ionizations, excitations = ORIGIN_PINS[label][backend]
    assert (state.electrons.count, state.ions.count) == (electrons, ions)
    assert (state.cumulative["ionizations"], state.cumulative["excitations"]) == (ionizations, excitations)
    assert _state_sha256(state) == digest


def test_spatial_block_enters_the_identity_and_is_validated(collision_set: CollisionSetConfig):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    base = _spatial_config(grid, collision_set)
    record = base.to_dict()["neutrals_spatial"]
    assert record["model"] == "neutrals_spatial_v1" and record["metastables"]["model"] == "metastables_v1"
    assert base.neutral_model == "neutrals_spatial_v1" and base.spatial_neutrals_active and base.metastables_active
    identities = {artifacts.config_identity(base)}
    for variant in (
        _spatial_config(grid, collision_set, metastables=False),
        _spatial_config(grid, collision_set, acceleration=20.0),
        _spatial_config(grid, collision_set, substep=25),
        _spatial_config(grid, collision_set, feed_scale=1.1),
    ):
        identities.add(artifacts.config_identity(variant))
    assert len(identities) == 5, "every declared parameter enters config_sha256"
    meta = MetastableConfig(BRANCHING, stepwise_scale=2.0)
    assert MetastableConfig(BRANCHING).to_dict() != meta.to_dict()
    with pytest.raises(PIC2DValidationError):
        _spatial_config(grid, collision_set, substep=7)          # must divide the sync / series interval
    with pytest.raises(PIC2DValidationError):
        _spatial_config(grid, collision_set, neutral_inventory=NeutralInventoryConfig(1e16, None))   # mutually exclusive
    with pytest.raises(PIC2DValidationError):
        _spatial_config(grid, None)                                # metastables need the declared collision set
    with pytest.raises(PIC2DValidationError):
        MetastableConfig((0.5, 1.5, 0.5, 0.5))
    with pytest.raises(PIC2DValidationError):
        SpatialNeutralConfig(1e16, 1e6, 10, time_acceleration=0.5)
    # the protocol block of the shared runner builds the same configuration
    block = {"model": "neutrals_spatial_v1", **{k: v for k, v in record.items() if k not in ("model", "metastables")},
             "metastables": {**record["metastables"], "branching_note": "documentation"}, "feed_note": "documentation"}
    rebuilt = runner.spatial_neutral_config_from_protocol(block)
    assert rebuilt.to_dict() == record
    with pytest.raises(PIC2DValidationError):
        runner.spatial_neutral_config_from_protocol({**block, "model": "inventory-0d"})


# ----------------------------------------------------------------------------------------------------------------------
# 3. the atom ledger contract (unit level)
# ----------------------------------------------------------------------------------------------------------------------

def test_depletion_removes_exactly_the_demanded_atoms_and_carries_a_debt():
    grid, a, length = _tube(2.0, nr=4)
    model, sinks = _neutral_only(grid, 1.0e16, 4000.0, acceleration=3.0, substeps=1, dt=1e-9, profile="uniform")
    rng = np.random.default_rng(5)
    state = model.initial_state(rng)
    cells = model._cells(state.particles)
    populated = int(np.bincount(cells, minlength=model.n_cells).argmax())
    weight_before = float(state.particles.weight[cells == populated].sum())
    demand = 0.3 * weight_before / 3.0            # real atoms; x F = 30 % of the cell
    sinks.ground_ionization[populated] = demand
    empty = int(np.flatnonzero(model.cell_volume > 0.0)[np.argmin(np.bincount(cells, minlength=model.n_cells)[model.cell_volume > 0.0])])
    before_true = state.true_ground_atoms()
    tally = model.substep(state, sinks, rng).values
    assert tally["neutral_ionized"] == pytest.approx(3.0 * demand) and tally["neutral_removed_ground"] == pytest.approx(3.0 * demand, rel=1e-12)
    assert state.debt_ground.sum() == 0.0
    # identity: d(true) = fed - ionized - effused (no recycling / CEX / metastables here)
    after_true = state.true_ground_atoms()
    assert after_true - before_true == pytest.approx(tally["neutral_fed"] - tally["neutral_ionized"] - tally["neutral_effused"], abs=1e-6 * before_true)
    # a demand on a cell without atoms is carried as debt, then removed when atoms are present (uniform profile: the cell fills)
    state2 = model.initial_state(np.random.default_rng(6))
    state2.particles = state2.particles.select(model._cells(state2.particles) != empty)
    model._deposit(state2)
    sinks.ground_ionization[empty] = 1000.0
    t1 = model.substep(state2, sinks, np.random.default_rng(7)).values
    assert t1["neutral_removed_ground"] < 3000.0 and state2.debt_ground[empty] == pytest.approx(3000.0 - t1["neutral_removed_ground"])
    debt = state2.debt_ground[empty]
    total_removed = t1["neutral_removed_ground"]
    for k in range(200):
        t = model.substep(state2, sinks, np.random.default_rng(100 + k)).values
        total_removed += t["neutral_removed_ground"]
        if state2.debt_ground[empty] == 0.0:
            break
    assert state2.debt_ground[empty] == 0.0 and total_removed == pytest.approx(3000.0, rel=1e-9), "the debt is paid when atoms arrive"
    assert debt > 0.0


def test_recycling_spawns_thermal_atoms_at_the_impact_cell_and_the_cex_hand_off_adds_fast_particles():
    grid, a, length = _tube(2.0, nr=4)
    model, sinks = _neutral_only(grid, 1.0e16, 2000.0, acceleration=2.0, substeps=1, dt=1e-9, profile="empty", wall_temperature_k=600.0)
    rng = np.random.default_rng(8)
    state = model.initial_state(rng)
    assert state.particles.count == 0
    wall_cell = (model.nr - 1) * model.nz + model.nz // 2
    w_n = model.config.macro_weight
    sinks.recycle[wall_cell] = 3.7 * w_n / 2.0          # real atoms: x F = 3.7 macro-weights -> 3 spawned, 0.7 carried
    fast = ParticleArrays(np.array([0.5e-3]), np.array([0.3 * length]), np.array([0.0]), np.array([0.0]), np.array([20000.0]))
    sinks.fast_neutrals = fast
    tally = model.substep(state, sinks, rng).values
    assert tally["neutral_recycled"] == pytest.approx(3.7 * w_n)
    particles = state.particles
    recycled = np.abs(particles.weight - w_n) < 1e-9 * w_n
    assert int(recycled.sum()) == 3 and state.pending_recycle[wall_cell] == pytest.approx(0.7 * w_n)
    assert state.true_ground_atoms() == pytest.approx(3.7 * w_n + tally["neutral_fast_in"] + tally["neutral_fed"] - tally["neutral_effused"], rel=1e-12)
    # the fast neutral carries the ion's velocity and F x W atoms (here W = 1 x F = 2)
    fast_mask = ~recycled
    assert tally["neutral_fast_in"] == pytest.approx(2.0) and int(fast_mask.sum()) == 1
    assert particles.vz_m_per_s[fast_mask][0] == pytest.approx(20000.0) and particles.state[fast_mask][0] == STATE_GROUND
    # recycled atoms: spawned inside the impact cell (before the flight) at the wall temperature - check the speed statistics on many
    sinks.recycle[wall_cell] = 4000.0 * w_n / 2.0
    state = model.initial_state(np.random.default_rng(9))
    model.substep(state, sinks, np.random.default_rng(10))
    speeds2 = state.particles.vr_m_per_s**2 + state.particles.vt_m_per_s**2 + state.particles.vz_m_per_s**2
    assert np.mean(speeds2) == pytest.approx(3.0 * BOLTZMANN_J_PER_K * 600.0 / XENON_MASS_KG, rel=0.08)


def test_ion_mcc_hands_the_cex_fast_neutral_over_instead_of_marching(collision_set: CollisionSetConfig):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    ion_set = IonNeutralCrossSections.synthetic_for_tests(cex_m2=5e-19, mex_m2=0.0)
    ion_config = IonNeutralMCCConfig("synthetic", ion_set.payload_sha256, (("cex", "charge_exchange"), ("mex", "momentum_transfer")))
    mcc_config = MCCConfig(1.0e21)
    operator = IonNullCollisionMCC(ion_set, mcc_config, ion_config, xenon_ion_species(2e6), masks)
    count = 20000
    ions = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 12.0e-3), np.zeros(count), np.zeros(count), np.full(count, 20000.0))
    local = np.full(count, 1.0e21)
    moments = (np.zeros(count), np.zeros(count), np.full(count, 150.0), np.full(count, 200.0))     # drifting gas at 150 m/s, v_th 200 m/s
    result = operator.apply(ions, 1e-9, np.random.default_rng(11), density_per_particle=local, neutral_moments=moments, hand_off_fast_neutrals=True)
    tally = result.tally
    assert tally.cex > 50 and result.fast_neutrals.count == tally.cex == result.cex_r_m.size
    assert tally.fast_neutral_exit_channel == tally.fast_neutral_wall == tally.fast_neutral_thermal == 0
    assert np.allclose(result.fast_neutrals.vz_m_per_s, 20000.0) and np.allclose(result.fast_neutrals.z_m, 12.0e-3)
    # the slow ions sampled the local drifting gas: mean v_z ~ the drift, not zero
    changed = result.ions.vz_m_per_s < 10000.0
    assert int(changed.sum()) == tally.cex and abs(float(np.mean(result.ions.vz_m_per_s[changed])) - 150.0) < 5.0 * 200.0 / sqrt(tally.cex)
    # the legacy call path (no hand-off) still marches the fates
    legacy = operator.apply(ions, 1e-9, np.random.default_rng(11), density_per_particle=local)
    assert legacy.fast_neutrals.count == 0 and legacy.tally.fast_neutral_exit_channel + legacy.tally.fast_neutral_wall + legacy.tally.fast_neutral_thermal == legacy.tally.cex
    assert "handed over" in operator.to_dict(spatial_neutrals=True)["fast_neutral_fate"]


# ----------------------------------------------------------------------------------------------------------------------
# 4. metastables
# ----------------------------------------------------------------------------------------------------------------------

def test_metastable_tables_and_mcc_channel_rates_against_analytic(xs: XenonCrossSections):
    config = MetastableConfig(BRANCHING)
    table = MetastableProcessTable.build(xs, config, ground_ceiling_per_m3=1e21, energy_step_ev=0.05, energy_max_ev=2000.0)
    assert table.stepwise_threshold_ev == pytest.approx(12.13 - 8.315)
    # BEB: zero below the binding energy, peak a few binding energies above it, 1 / E ln E tail; the Xe(6s) peak lies in the
    # audit's 1e-20..3e-19 m2 bracket
    energies = np.array([3.0, 3.815, 5.0, 10.0, 15.0, 30.0, 100.0, 1000.0])
    sigma = beb_cross_section_m2(energies, table.stepwise_threshold_ev, table.stepwise_threshold_ev)
    assert sigma[0] == 0.0 and sigma[1] == 0.0 and sigma[2] > 0.0
    assert 1e-20 < sigma.max() < 3e-19 and sigma[-1] < 0.2 * sigma.max()
    assert 8.0 < table.to_dict()["stepwise_peak_energy_ev"] < 25.0
    # superelastic by detailed balance: g E' sigma_super(E') = g0 E sigma_exc(E) with the metastable share of the lumped level
    e_prime = 5.0
    exc = xs.excitation_levels[0].at(np.array([e_prime + 8.315]))[0]
    assert table.table.lookup(np.array([e_prime]))[1, 0] == pytest.approx(0.2 * BRANCHING[0] * (e_prime + 8.315) / e_prime * exc, rel=0.02)
    # the MCC channels on a monoenergetic beam: n_m sigma v dt N events each, the ground channels untouched by n_m
    operator = NullCollisionMCC(xs, MCCConfig(1.0e21, collision_set=CollisionSetConfig.xe_collision_set_v2(ion_neutral=False)), xenon_ion_species(1.0), table)
    assert operator.nu_max > operator.nu_max_ground
    count, energy_ev, dt = 200_000, 20.0, 2.0e-11
    speed = float(electron_speed_from_energy(np.array([energy_ev]))[0])
    electrons = ParticleArrays(np.full(count, 1.0e-3), np.full(count, 12.0e-3), np.zeros(count), np.zeros(count), np.full(count, speed))
    n_g, n_m = 2.0e20, 4.0e19
    result = operator.apply(electrons, dt, np.random.default_rng(12), density_per_particle=np.full(count, n_g),
                            metastable_density_per_particle=np.full(count, n_m), neutral_moments=(np.zeros(count), np.zeros(count), np.zeros(count), np.zeros(count)))
    tally = result.tally
    sig = table.table.lookup(np.array([energy_ev]))[:, 0]
    expected_step = n_m * sig[0] * speed * dt * count
    expected_super = n_m * sig[1] * speed * dt * count
    assert abs(tally.stepwise_ionization - expected_step) < 4.0 * sqrt(expected_step) + 1.0, (tally.stepwise_ionization, expected_step)
    assert abs(tally.superelastic - expected_super) < 4.0 * sqrt(expected_super) + 1.0, (tally.superelastic, expected_super)
    sig_g = operator.table.lookup(np.array([energy_ev]))[:, 0]
    expected_ion = n_g * sig_g[-1] * speed * dt * count
    assert abs(tally.ionization - expected_ion) < 4.0 * sqrt(expected_ion)
    # energy bookkeeping: stepwise removes E_iz - E_m, superelastic hands E_m back; births = ground + stepwise
    levels = sum(n * t for n, t in zip(tally.excitation_levels, operator.table.excitation_thresholds_ev))
    expected_loss = (levels + tally.ionization * 12.13 + tally.stepwise_ionization * table.stepwise_threshold_ev - tally.superelastic * 8.315) * EV_J
    assert tally.inelastic_energy_loss_j == pytest.approx(expected_loss, rel=1e-12)
    assert result.new_ions.count == tally.ionization + tally.stepwise_ionization == result.ionization_r_m.size + result.stepwise_r_m.size
    assert result.superelastic_r_m.size == tally.superelastic and result.excitation_level.size == tally.excitation
    gained = 0.5 * 9.1093837139e-31 * (result.electrons.vr_m_per_s**2 + result.electrons.vt_m_per_s**2 + result.electrons.vz_m_per_s**2) / EV_J
    assert np.count_nonzero(np.abs(gained - (energy_ev + 8.315)) < 1e-6) == tally.superelastic
    assert "stepwise_ionization" in operator.to_dict()["kinematics"] and tally.to_dict()["superelastic"] == tally.superelastic


def test_metastable_production_wall_deexcitation_and_radiative_decay_in_the_transport():
    grid, a, length = _tube(2.0, nr=4)
    meta = MetastableConfig((1.0,), weight_ratio=0.1, radiative_decay_rate_per_s=4.0e5, wall_deexcitation_probability=1.0)
    model, sinks = _neutral_only(grid, 1.0e16, 3000.0, acceleration=1.0, substeps=1, dt=50e-9, profile="uniform", metastables=meta)
    rng = np.random.default_rng(13)
    state = model.initial_state(rng)
    w_m = model.metastable_weight
    assert w_m == pytest.approx(0.1 * model.config.macro_weight)
    cell = model.nz // 2                        # an axis cell far from the walls
    sinks.ground_excitation[cell] = 2000.0 * w_m
    before_g, before_m = state.true_ground_atoms(), state.true_meta_atoms()
    tally = model.substep(state, sinks, rng).values
    produced = tally["neutral_excited_to_pool"]
    assert produced == pytest.approx(2000.0 * w_m) and tally["meta_produced"] == produced
    meta_mask = state.particles.state == STATE_METASTABLE
    # radiative decay over one sub-step: survival exp(-A dt) within statistics
    survival = np.exp(-4.0e5 * model.substep_dt_s)
    n_meta = int(meta_mask.sum())
    assert abs(n_meta - 2000 * survival) < 4.0 * sqrt(2000 * survival * (1 - survival)) + 2
    assert tally["meta_radiative"] == pytest.approx((2000 - n_meta) * w_m)
    # ledger identities for both species (no plasma sinks on the pool, no effusion of metastables this early)
    d_g = state.true_ground_atoms() - before_g
    d_m = state.true_meta_atoms() - before_m
    assert d_g == pytest.approx(tally["neutral_fed"] + tally["meta_radiative"] + tally["meta_wall_deexcited"] - produced - tally["neutral_effused"], abs=1e-6 * before_g)
    assert d_m == pytest.approx(produced - tally["meta_radiative"] - tally["meta_wall_deexcited"] - tally["meta_effused"], abs=1e-6 * max(before_g, 1.0))
    # let them fly to the walls: every metastable eventually de-excites (wall) or radiates; the pool empties, atoms are conserved
    total_before = state.true_ground_atoms() + state.true_meta_atoms()
    ledger = {k: tally[k] for k in NEUTRAL_SPATIAL_LEDGER_KEYS}       # incl. the production sub-step's radiative loss
    for k in range(300):
        t = model.substep(state, sinks, np.random.default_rng(200 + k)).values
        for key in ledger:
            ledger[key] += t[key]
    assert state.particles.atoms(STATE_METASTABLE) < 0.02 * produced
    assert ledger["meta_wall_deexcited"] > 0.0 and ledger["meta_wall_deexcited"] + ledger["meta_radiative"] + ledger["meta_effused"] == pytest.approx(
        produced - state.true_meta_atoms(), rel=1e-9)
    total_after = state.true_ground_atoms() + state.true_meta_atoms()
    later = {k: ledger[k] - tally[k] for k in ledger}
    assert total_after - total_before == pytest.approx(later["neutral_fed"] - later["neutral_effused"] - later["meta_effused"], abs=1e-6 * total_before)


# ----------------------------------------------------------------------------------------------------------------------
# 5. the coupled discharge on both backends
# ----------------------------------------------------------------------------------------------------------------------

def _ledger_terms(sim: Simulation) -> dict[str, np.ndarray]:
    out = {k: [] for k in ("dke", "rhs")}
    for a, b in pairwise(sim.series):
        ca, cb = a.ledger["cumulative"], b.ledger["cumulative"]
        d = lambda key: float(cb.get(key, 0.0) - ca.get(key, 0.0))  # noqa: E731
        out["dke"].append((b.kinetic_electron_j + b.kinetic_ion_j) - (a.kinetic_electron_j + a.kinetic_ion_j))
        out["rhs"].append(d("field_work_j") + d("ke_injected_j") - d("ke_absorbed_anode_j") - d("ke_absorbed_exit_j") - d("ke_absorbed_wall_j")
                          + d("ke_born_ions_j") - d("inelastic_loss_j") - d("ion_neutral_loss_j"))
    return {k: np.asarray(v) for k, v in out.items()}


@pytest.mark.parametrize("backend", PARITY)
def test_discharge_atom_and_energy_identities_close_with_every_channel_active(backend: str, collision_set: CollisionSetConfig, xs: XenonCrossSections):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _spatial_config(grid, collision_set, acceleration=200.0)
    sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs)
    sim.run(200, accumulate_from_step=100)
    records = sim.series
    assert len(records) == 8
    for record in records:
        neutral = record.neutral
        assert neutral["model"] == "neutrals_spatial_v1" and neutral["metastables"]["model"] == "metastables_v1"
        scale = neutral["atoms_ground"]
        assert abs(neutral["interval_ledger_residual_atoms"]) < 1e-9 * scale, neutral["interval_ledger_residual_atoms"]
        assert abs(neutral["interval_meta_ledger_residual_atoms"]) < 1e-9 * scale
        assert neutral["sink_consistency_atoms"] == 0.0, "the ionisation sink equals F x W x (ground ionisations) exactly"
        assert set(NEUTRAL_SPATIAL_LEDGER_KEYS) <= set(neutral["ledger"]) and neutral["ceiling_violation_fraction"] <= 1e-3
        assert neutral["axis_density_anode_per_m3"] > neutral["axis_density_exit_per_m3"] > 0.0      # the Knudsen profile
    cumulative = records[-1].ledger["cumulative"]
    assert cumulative["ionizations"] > 0 and cumulative["neutral_ionized"] > 0 and cumulative["neutral_substeps"] == 40.0
    assert cumulative["neutral_ionized"] == pytest.approx(200.0 * 2e6 * (cumulative["ionizations"] - cumulative.get("stepwise_ionizations", 0.0)))
    assert cumulative["neutral_wall_hits"] > 0 and cumulative["neutral_effused"] > 0 and cumulative["neutral_fed"] > 0
    assert cumulative["meta_produced"] > 0 and cumulative["neutral_excited_to_pool"] == cumulative["meta_produced"]
    # the particle-side energy identity with the metastable channels in the inelastic sink
    terms = _ledger_terms(sim)
    scale = max(float(np.max(np.abs(terms["dke"]))), float(records[-1].kinetic_electron_j))
    assert float(np.max(np.abs(terms["dke"] - terms["rhs"]))) <= 1e-6 * scale
    # the record carries the real-time rates the 0-D balance speaks of
    assert records[-1].neutral["ionization_rate_per_s"] == records[-1].currents_a["ionization_rate_per_s"]
    assert records[-1].neutral["gross_utilisation"] == pytest.approx(records[-1].neutral["ionization_rate_per_s"] / config.neutrals_spatial.feed_atoms_per_s)
    provenance = sim.to_provenance()
    assert provenance["config"]["neutrals_spatial"]["model"] == "neutrals_spatial_v1"
    # the maps carry the cell-centred neutral fields (only with the model on)
    maps = sim.backend.diagnostic_arrays()
    assert maps["neutral_density_per_m3"].shape == grid.cell_shape and maps["metastable_density_per_m3"].shape == grid.cell_shape
    assert float(maps["neutral_density_per_m3"].max()) > 0.0 and int(maps["neutral_samples"][0]) > 0
    legacy_maps = DiagnosticAccumulator(build_mesh_masks(grid)).to_arrays(2e6, 5e-12)
    assert "neutral_density_per_m3" not in legacy_maps


def test_cpu_and_warp_agree_in_distribution(collision_set: CollisionSetConfig, xs: XenonCrossSections):
    if len(PARITY) < 2:
        pytest.skip("Warp unavailable")
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    finals = {}
    for backend in PARITY:
        config = _spatial_config(grid, collision_set, acceleration=200.0)
        sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs)
        sim.run(150)
        finals[backend] = sim.series[-1].neutral
    a, b = (finals[name] for name in PARITY)
    for key in ("atoms_ground", "density_per_m3", "axis_density_anode_per_m3"):
        assert abs(a[key] - b[key]) < 0.05 * (abs(a[key]) + abs(b[key])), key
    assert abs(a["ledger"]["neutral_effused"] - b["ledger"]["neutral_effused"]) < 0.25 * (a["ledger"]["neutral_effused"] + b["ledger"]["neutral_effused"])
    assert a["ledger"]["neutral_fed"] == pytest.approx(b["ledger"]["neutral_fed"], rel=1e-12)     # the same feed (summation order differs)


def test_checkpoint_round_trip_and_bitwise_resume_with_spatial_neutrals(tmp_path: Path, collision_set: CollisionSetConfig, xs: XenonCrossSections):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _spatial_config(grid, collision_set)
    field = linear_psi_field_map(grid, 2.0)
    sim = Simulation(config, field, backend="cpu", cross_sections=xs)
    sim.run(50)
    state = sim.state
    assert state.neutral_particles is not None and state.neutral_particles.particles.count > 1000
    json_path, npz_path = artifacts.save_checkpoint(tmp_path, "ck", state, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256, backend="cpu", field=field)
    # the files carry the REQUESTED name (v2.5.0 wrote them as thermal_speed.* - the last neutral cell key shadowed ``name`` - so the runner's
    # checkpoint-latest / checkpoint-final of a spatial run could never be found for a resume or a finalize; full-physics v1 box shakedown)
    assert json_path == tmp_path / "ck.json" and npz_path == tmp_path / "ck.npz" and (tmp_path / "ck.field.npz").is_file()
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix in (".json", ".npz") and not p.name.endswith(".sha256.json")) == ["ck.field.npz", "ck.json", "ck.npz"]
    metadata = artifacts.read_canonical_json(json_path)
    assert metadata["arrays_file"] == "ck.npz" and metadata["field_anchor_file"] == "ck.field.npz"
    assert metadata["neutral_particle_count"] == state.neutral_particles.particles.count and "neutral_fed" in metadata["cumulative_extra_keys"]
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    assert np.array_equal(loaded.neutral_particles.particles.weight, state.neutral_particles.particles.weight)
    assert np.array_equal(loaded.neutral_particles.density_per_m3, state.neutral_particles.density_per_m3)
    sim.run(50)
    resumed = Simulation(config, field, backend="cpu", cross_sections=xs)
    resumed.load_state(loaded)
    resumed.run(50)
    assert np.array_equal(resumed.state.ions.vz_m_per_s, sim.state.ions.vz_m_per_s) and np.array_equal(resumed.state.phi_v, sim.state.phi_v)
    assert np.array_equal(resumed.state.neutral_particles.particles.r_m, sim.state.neutral_particles.particles.r_m)
    assert resumed.state.cumulative["neutral_effused"] == sim.state.cumulative["neutral_effused"]
    # a checkpoint of the spatial model does not load into the 0-D configuration and vice versa
    with pytest.raises(PIC2DValidationError):
        artifacts.load_checkpoint(json_path, _inventory_config(grid, collision_set), field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)


@pytest.mark.parametrize("backend", CUDA)
def test_cuda_graph_and_direct_launch_are_bitwise_with_spatial_neutrals(backend: str, collision_set: CollisionSetConfig, xs: XenonCrossSections):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    states = []
    tallies = []
    for step_graph in (True, False):
        config = _spatial_config(grid, collision_set, acceleration=200.0)
        sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs, step_graph=step_graph)
        sim.run(150)
        states.append(sim.state)
        tallies.append(sim.series[-1].ledger["cumulative"])
    assert sim.step_graph_state() is False and states[0].electrons.count == states[1].electrons.count
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(states[0].electrons, name), getattr(states[1].electrons, name))
        assert np.array_equal(getattr(states[0].ions, name), getattr(states[1].ions, name))
    assert np.array_equal(states[0].phi_v, states[1].phi_v)
    a, b = states[0].neutral_particles, states[1].neutral_particles
    assert a.particles.count == b.particles.count
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s", "weight", "state"):
        assert np.array_equal(getattr(a.particles, name), getattr(b.particles, name)), name
    assert np.array_equal(a.density_per_m3, b.density_per_m3) and np.array_equal(a.debt_ground, b.debt_ground)
    for key in ("ionizations", "cex", "neutral_substeps", "stepwise_ionizations", "superelastic"):
        assert tallies[0][key] == tallies[1][key], key
    assert tallies[0]["neutral_ionized"] > 0


def test_a_plume_domain_runs_with_spatial_neutrals_and_effuses_through_the_far_field(collision_set: CollisionSetConfig, xs: XenonCrossSections):
    geometry = ChannelGeometry(2.0e-3, 0.0, 12.0e-3, 9.0e-3, 3.0e-3, plume_radius_m=6.0e-3, plume_length_m=6.0e-3, body_dielectric_radius_m=4.0e-3)
    grid = Grid2D(geometry, 12, 36)
    feed = feed_for_density(1.0e20, pi * geometry.exit_radius_m**2, 300.0)
    n_exit = feed / effusion_coefficient_m3_per_s(pi * geometry.exit_radius_m**2, 300.0)
    volume = pi * geometry.bore_radius_m**2 * geometry.channel_length_m
    neutrals = SpatialNeutralConfig(feed, n_exit * volume / 6000.0, 5, time_acceleration=100.0, wall_temperature_k=500.0)
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e15, 5.0, 1.0, region="channel"),
        mcc=MCCConfig(6.0e20, collision_set=collision_set), poisson=PoissonConfig2D(method="direct"),
        reference_density_per_m3=1e15, reference_electron_temperature_ev=5.0, limits=StabilityLimits(max_cell_debye_ratio=2.0),
        series_interval_steps=25, ion_subcycle=4, neutrals_spatial=neutrals,
    )
    sim = Simulation(config, uniform_field_map(grid, 0.02), backend="cpu", cross_sections=xs)
    sim.run(100, accumulate_from_step=50)
    neutral = sim.series[-1].neutral
    assert neutral["ceiling_violation_fraction"] <= 1e-3 and neutral["ledger"]["neutral_wall_hits"] > 0.0
    assert abs(neutral["interval_ledger_residual_atoms"]) < 1e-9 * neutral["atoms_ground"]
    maps = sim.backend.diagnostic_arrays()
    density = maps["neutral_density_per_m3"]
    j_exit = int(round(geometry.channel_length_m / grid.dz_m))
    assert density[0, :j_exit].mean() > 3.0 * density[0, j_exit:].mean() > 0.0     # the plume gas is the effused cone, thinner than the channel
    # neutral-only flight over a few tube transits: the atoms leave through the far-field boundary (z = z_max and r = R_plume), not the
    # exit plane, and the body front face reflects them
    spatial = sim.backend.spatial
    state = spatial.initial_state(np.random.default_rng(3))
    sinks = CellSinks.zeros(spatial.n_cells, 2e6)
    ledger = {k: 0.0 for k in NEUTRAL_SPATIAL_LEDGER_KEYS}
    spatial.substep_dt_s = 0.5e-6                                     # 0.5 us of neutral time per sub-step (test-only override)
    for k in range(60):
        t = spatial.substep(state, sinks, np.random.default_rng(300 + k)).values
        for key in ledger:
            ledger[key] += t[key]
    assert ledger["neutral_effused"] > 0.0 and ledger["neutral_wall_hits"] > 0.0
    particles = state.particles
    assert np.all(particles.z_m < geometry.domain_z_max_m) and np.all(particles.r_m < geometry.max_radius_m)
    in_channel = particles.z_m < geometry.z_max_m
    assert np.all(particles.r_m[in_channel] <= geometry.wall_radius_m(particles.z_m[in_channel]) + grid.dr_m)
