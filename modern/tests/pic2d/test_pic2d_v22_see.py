"""Model v2.2.0 ``see_dielectric_v1``: secondary electron emission from the dielectric wall (R2 of the physics audit).

Regressions:
* material constants reproduce their cited anchors (BN: Villemant 2019 fit -> Dunaevsky 2003 crossover 35 eV and
  sigma_0 0.54; critical temperature between Sydorenko's 18.3 eV and Dunaevsky's 19.3 eV; Al2O3 more emissive);
* the yield curve is reproduced statistically by the emission sampler at sampled energies and angles; the integer yield
  sampling is unbiased; emission energy / angle distributions (elastic keeps the speed, inelastic uniform in energy, true
  secondaries at mean 2 T_see with the cosine law) are right; the wall-crossing geometry places the secondaries inside
  the last plasma cell with the inward normal;
* the particle-side energy identity closes to round-off with SEE on (cpu, warp-cpu, cuda), the emitted energy being an
  injected term; the electron count identity holds; the surface charge is absorbed minus emitted;
* SEE off reproduces v2.0.6: config identity, ledger keys, window sums, frame profile keys and the recorded tallies of a
  fixed run are unchanged (the whole-state bitwise comparison against the pre-v2.2.0 tree was made offline for cpu and
  warp-cpu: particles, phi, surface charge, cumulative ledger and maps identical - the integer tallies below are the
  in-repo witness);
* the CUDA graph replays the direct launches bitwise with SEE on; warp-cpu / cuda agree with the numpy reference in
  distribution;
* the 1-D-like sheath test (THE physics check): a floating dielectric wall facing a Maxwellian plasma - the wall drop
  falls monotonically with the yield, the fall follows the Hobbs-Wesson factor ``T_e ln(1 / (1 - delta_eff))`` and at
  delta >= 1 the space-charge-limited drop ~1.02 T_e emerges from the PIC (no cap imposed), with the effective
  (wall-defined) yield saturating below 1.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cft_revival.pic2d import artifacts, kernels
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map, zero_field_map
from cft_revival.pic2d.frames import PROFILE_KEYS, interval_maps
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EV_J,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    ParticleArrays,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.see import (
    HOBBS_WESSON_CRITICAL_YIELD_XE,
    HOBBS_WESSON_SCL_DROP_TE,
    MATERIALS,
    NORMAL_MINUS_R,
    NORMAL_MINUS_Z,
    NORMAL_PLUS_Z,
    SEEConfig,
    classical_sheath_drop_te,
    critical_temperature_ev,
    emit_secondaries,
    first_crossover_ev,
    maxwellian_flux_average_yield,
    sample_integer_yield,
    wall_crossing,
)
from cft_revival.pic2d.sensitivity import AnomalousCollisionConfig
from cft_revival.pic2d.simulation import (
    CUMULATIVE_KEYS,
    SEE_KEYS,
    DiagnosticAccumulator,
    InjectionConfig,
    PIC2DConfig,
    SeedPlasmaConfig,
    Simulation,
    dielectric_wall_nodes,
    empty_cumulative,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)


def _warp_backends() -> list[str]:
    try:
        from cft_revival.pic2d.warp_backend import device_available
    except ImportError:  # pragma: no cover
        return []
    return [name for name, device in (("warp-cpu", "cpu"), ("warp-cuda", "cuda:0")) if device_available(device)]


WARP_BACKENDS = _warp_backends()
BACKENDS = ["cpu", *WARP_BACKENDS]
FAST_BACKEND = "warp-cpu" if "warp-cpu" in WARP_BACKENDS else "cpu"


def _discharge_config(grid: Grid2D, *, see: SEEConfig | None, series: int = 25, seed: int = 3, **extra) -> PIC2DConfig:
    """The warp-parity discharge (300 V, injection, MCC at 1e21, W 2e6) with an optional emitting wall."""

    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=seed,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=series, see=see, **extra,
    )


# -- model constants ------------------------------------------------------------------------------------------------------

def test_material_constants_reproduce_the_cited_anchors_and_the_identity_carries_them():
    bn = SEEConfig(material="BN")
    al = SEEConfig(material="Al2O3")
    # BN (Villemant 2019 fit): maximum 2.016 at 299 eV; first crossover = Dunaevsky 2003 power-fit E_1 = 35 eV (grade HP);
    # delta(10 eV) ~ Dunaevsky's linear-fit sigma_0 = 0.54; critical temperature between Sydorenko 18.3 and Dunaevsky 19.3 eV
    assert float(bn.yield_at(MATERIALS["BN"].energy_max_ev)) == pytest.approx(2.016)
    assert first_crossover_ev(bn) == pytest.approx(35.0, abs=1.5)
    assert float(bn.yield_at(10.0)) == pytest.approx(0.54, abs=0.05)
    assert 18.0 < critical_temperature_ev(bn) < 21.0
    # Maxwellian-flux-averaged yield at the plateau temperatures (declared in the spec) and the space-charge limit constants
    assert maxwellian_flux_average_yield(bn, 7.0) == pytest.approx(0.575, abs=0.01)
    assert HOBBS_WESSON_CRITICAL_YIELD_XE == pytest.approx(0.983, abs=1e-3) and HOBBS_WESSON_SCL_DROP_TE == 1.02
    assert bn.space_charge_limit_yield == pytest.approx(HOBBS_WESSON_CRITICAL_YIELD_XE)
    # the floating-sheath formula of the audit appendix
    assert [round(classical_sheath_drop_te(d), 2) for d in (0.0, 0.5, 0.9, 0.983)] == [5.27, 4.58, 2.97, 1.2]
    # Al2O3 is the more emissive wall at every plateau temperature (audit hypothesis "Al2O3 > BN in every effect")
    for t in (5.0, 7.0, 10.0):
        assert maxwellian_flux_average_yield(al, t) > maxwellian_flux_average_yield(bn, t)
    assert critical_temperature_ev(al) < critical_temperature_ev(bn)
    assert float(al.yield_at(5.0)) > 0.0        # the low-energy elastic bump keeps the yield finite below the Vaughan threshold
    # components: elastic + inelastic below the total; the total at normal incidence is the Vaughan curve (+ bump)
    total, elastic, inelastic = bn.yield_components(np.array([5.0, 50.0, 500.0]))
    assert np.all(elastic + inelastic < total) and np.all(elastic > 0.0) and np.all(inelastic > 0.0)
    assert np.allclose(elastic / total, 0.03) and np.allclose(inelastic / total, 0.07)
    # oblique incidence raises the yield (Vaughan angular factors)
    assert float(bn.yield_at(100.0, 1.0)) > float(bn.yield_at(100.0, 0.0))
    # identity: every constant and its provenance enter the configuration dict; overrides are declared and validated
    record = bn.to_dict()
    assert record["model"] == "see_dielectric_v1" and record["constants"]["delta_max"] == 2.016 and "Villemant" in record["constants"]["source"]
    assert record["constants"]["energy_threshold_ev"] == 0.0 and record["emission"]["space_charge_limit"].startswith("not imposed")
    over = SEEConfig(material="BN", overrides={"delta_max": 2.9})
    assert over.to_dict()["constants"]["delta_max"] == 2.9 and over.to_dict() != record
    with pytest.raises(PIC2DValidationError):
        SEEConfig(material="steel")
    with pytest.raises(PIC2DValidationError):
        SEEConfig(overrides={"nonsense": 1.0})
    with pytest.raises(PIC2DValidationError):
        SEEConfig(yield_model="constant", constant_yield=9.0, max_emitted_per_impact=8)
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    assert artifacts.config_identity(_discharge_config(grid, see=bn)) != artifacts.config_identity(_discharge_config(grid, see=al))
    assert artifacts.config_identity(_discharge_config(grid, see=bn)) != artifacts.config_identity(_discharge_config(grid, see=None))


# -- sampling: yield curve, integer yield, distributions, geometry ---------------------------------------------------------

def _impacts(grid: Grid2D, n: int, energy_ev: float, theta: float, rng: np.random.Generator) -> tuple[ParticleArrays, ParticleArrays]:
    """``n`` electrons that crossed the straight-bore wall (outer grid line) at energy E and incidence angle theta."""

    speed = float(np.sqrt(2.0 * energy_ev * EV_J / ELECTRON_MASS_KG))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(2.0e-3, 6.0e-3, n)
    r_wall = grid.geometry.bore_radius_m
    old = ParticleArrays(np.full(n, r_wall - 0.4 * grid.dr_m), z, np.zeros(n), np.zeros(n), np.zeros(n))
    vr = np.full(n, speed * np.cos(theta))
    vt = speed * np.sin(theta) * np.cos(phi)
    vz = speed * np.sin(theta) * np.sin(phi)
    hit = ParticleArrays(np.full(n, r_wall + 0.3 * grid.dr_m), z + 1e-6, vr, vt, vz)
    return old, hit


def test_yield_curve_is_reproduced_statistically_at_sampled_energies_and_angles():
    grid = Grid2D(STRAIGHT_GEOMETRY, 12, 64)
    masks = build_mesh_masks(grid)
    see = SEEConfig(material="BN")
    rng = np.random.default_rng(11)
    n = 20000
    for energy, theta in ((8.0, 0.0), (35.0, 0.0), (100.0, 0.0), (299.0, 0.0), (100.0, 1.0), (1500.0, 0.5)):
        old, hit = _impacts(grid, n, energy, theta, rng)
        ke = 0.5 * ELECTRON_MASS_KG * hit.speed_squared() * 2e6
        out = emit_secondaries(see, grid, masks.plasma_cell, is_electron=True, old=old, hit=hit, impact_kinetic_energy_j=ke,
                               macro_weight=2e6, rng=rng)
        delta = float(see.yield_at(energy, theta))
        frac = delta - np.floor(delta)
        sigma = float(np.sqrt(max(frac * (1.0 - frac), 1e-6) / n))
        assert out.impacts == n and out.yield_sum == pytest.approx(n * delta, rel=1e-9)
        assert out.emitted / n == pytest.approx(delta, abs=max(5.0 * sigma, 0.01)), (energy, theta, out.emitted / n, delta)
        # every secondary sits inside the plasma (the last cell before the wall) and moves inward
        assert np.all(kernels.classify_boundary(masks, out.particles.r_m, out.particles.z_m) == kernels.BOUNDARY_INSIDE)
        assert np.all(out.particles.vr_m_per_s < 0.0)
        assert np.all(out.particles.r_m == pytest.approx(grid.geometry.bore_radius_m - 1e-6 * grid.dr_m))
    # the wall's charge bookkeeping: +1 unit per emitted electron per impact
    assert out.emitted_per_impact.sum() == out.emitted
    # ion-induced yield: constant, true secondaries only
    ion = SEEConfig(material="BN", ion_induced_yield=0.3)
    old, hit = _impacts(grid, n, 50.0, 0.0, rng)
    out_i = emit_secondaries(ion, grid, masks.plasma_cell, is_electron=False, old=old, hit=hit, impact_kinetic_energy_j=np.zeros(n),
                             macro_weight=2e6, rng=rng)
    assert out_i.impacts == 0 and out_i.backscattered == 0 and out_i.emitted / n == pytest.approx(0.3, abs=0.02)


def test_integer_yield_sampling_is_unbiased_and_clamped():
    rng = np.random.default_rng(5)
    n = 200000
    for delta in (0.0, 0.3, 1.0, 1.7, 2.5):
        counts = sample_integer_yield(np.full(n, delta), rng.random(n), 8)
        frac = delta - np.floor(delta)
        assert counts.min() >= int(np.floor(delta)) and counts.max() <= int(np.floor(delta)) + 1
        assert counts.mean() == pytest.approx(delta, abs=5.0 * np.sqrt(max(frac * (1 - frac), 1e-9) / n) + 1e-12)
    assert np.all(sample_integer_yield(np.full(10, 12.7), rng.random(10), 8) == 8)


def test_emission_energy_and_angle_distributions_follow_the_declared_samplers():
    grid = Grid2D(STRAIGHT_GEOMETRY, 12, 64)
    masks = build_mesh_masks(grid)
    rng = np.random.default_rng(3)
    n = 40000
    energy = 60.0
    old, hit = _impacts(grid, n, energy, 0.3, rng)
    ke = 0.5 * ELECTRON_MASS_KG * hit.speed_squared() * 2e6

    def emitted(config: SEEConfig):
        return emit_secondaries(config, grid, masks.plasma_cell, is_electron=True, old=old, hit=hit, impact_kinetic_energy_j=ke,
                                macro_weight=2e6, rng=rng)

    # elastic only: the impact speed is kept exactly; the direction follows the cosine law about the inward normal (<cos> = 2/3)
    elastic = emitted(SEEConfig(yield_model="constant", constant_yield=1.0, overrides={"elastic_fraction": 1.0, "inelastic_fraction": 0.0}))
    speed = np.sqrt(elastic.particles.speed_squared())
    assert elastic.backscattered == elastic.emitted == n
    assert np.allclose(speed, np.sqrt(hit.speed_squared()), rtol=1e-12)
    cos_out = -elastic.particles.vr_m_per_s / speed
    assert np.all(cos_out > 0.0) and cos_out.mean() == pytest.approx(2.0 / 3.0, abs=0.01)
    assert np.mean(cos_out**2) == pytest.approx(0.5, abs=0.01)                    # cosine law: <cos^2> = 1/2
    # inelastic only: energy uniform in (0, E) -> mean E/2, still cosine-directed
    inelastic = emitted(SEEConfig(yield_model="constant", constant_yield=1.0, overrides={"elastic_fraction": 0.0, "inelastic_fraction": 1.0}))
    e_in = 0.5 * ELECTRON_MASS_KG * inelastic.particles.speed_squared() / EV_J
    assert inelastic.backscattered == n and e_in.mean() == pytest.approx(energy / 2.0, rel=0.02) and e_in.max() <= energy * (1 + 1e-9)
    # true secondaries: flux half-Maxwellian at T_see -> energy E e^(-E/T)/T^2 (mean 2 T, variance 2 T^2), cosine law
    t_see = 2.0
    true = emitted(SEEConfig(yield_model="constant", constant_yield=1.0, emission_temperature_ev=t_see,
                             overrides={"elastic_fraction": 0.0, "inelastic_fraction": 0.0}))
    e_true = 0.5 * ELECTRON_MASS_KG * true.particles.speed_squared() / EV_J
    assert true.backscattered == 0
    assert e_true.mean() == pytest.approx(2.0 * t_see, rel=0.02) and e_true.std() == pytest.approx(np.sqrt(2.0) * t_see, rel=0.03)
    cos_true = -true.particles.vr_m_per_s / np.sqrt(true.particles.speed_squared())
    assert np.all(cos_true > 0.0) and cos_true.mean() == pytest.approx(2.0 / 3.0, abs=0.01)
    assert true.kinetic_energy_j == pytest.approx(float(e_true.sum()) * EV_J * 2e6, rel=1e-6)
    # the material split: BN at 60 eV emits 10 % backscattered (r_e + r_i) within statistics
    bn = emitted(SEEConfig(material="BN"))
    assert bn.backscattered / bn.emitted == pytest.approx(0.10, abs=0.01)


def test_wall_crossing_geometry_finds_the_face_and_the_inward_normal():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)          # bore 2 mm to z = 18 mm, then a stair-step cone to 3 mm
    masks = build_mesh_masks(grid)
    dr, dz = grid.dr_m, grid.dz_m
    r_w = CFT_GEOMETRY.bore_radius_m
    # straight bore: an outward radial crossing lands on the outer face of the last plasma cell, normal -r
    r_e, z_e, code = wall_crossing(grid, masks.plasma_cell, np.array([r_w - 0.3 * dr]), np.array([5.0e-3]), np.array([r_w + 0.2 * dr]), np.array([5.05e-3]))
    assert code[0] == NORMAL_MINUS_R and r_e[0] == pytest.approx(r_w - 1e-6 * dr) and 5.0e-3 < z_e[0] < 5.05e-3
    # cone: the stair-step wall has axial faces; a particle in the first cone cell moving toward the anode across the step
    # face of the bore (z decreasing, r above the bore radius) hits an axial face with normal +z
    i_step = int((r_w + 0.3 * dr) / dr)                                   # the first radial cell above the bore wall
    j_step = int(np.flatnonzero(masks.plasma_cell[i_step])[0])            # first cone column whose stair includes it
    z_face = j_step * dz
    r_e, z_e, code = wall_crossing(grid, masks.plasma_cell, np.array([r_w + 0.3 * dr]), np.array([z_face + 0.2 * dz]),
                                   np.array([r_w + 0.3 * dr]), np.array([z_face - 0.3 * dz]))
    assert masks.plasma_cell[i_step, j_step] and not masks.plasma_cell[i_step, j_step - 1]
    assert code[0] == NORMAL_PLUS_Z and z_e[0] == pytest.approx(z_face + 1e-6 * dz) and r_e[0] == pytest.approx(r_w + 0.3 * dr)
    # a forward axial crossing out of the exit-side end of a stair (normal -z) when the cell beyond is solid: construct it on
    # a wall cell of the cone where the next column is not plasma at that radius (none exists in a widening cone) -> the
    # code is exercised through the fallback branch of a pure radial move instead
    r_e, z_e, code = wall_crossing(grid, masks.plasma_cell, np.array([r_w - 0.5 * dr]), np.array([5.0e-3]), np.array([r_w]), np.array([5.0e-3]))
    assert code[0] == NORMAL_MINUS_R and r_e[0] == pytest.approx(r_w - 1e-6 * dr)
    # every emission point is inside the plasma
    for r0, z0, r1, z1 in ((r_w - 0.3 * dr, 5.0e-3, r_w + 0.2 * dr, 5.05e-3), (r_w + 0.3 * dr, z_face + 0.2 * dz, r_w + 0.3 * dr, z_face - 0.3 * dz)):
        r_e, z_e, _ = wall_crossing(grid, masks.plasma_cell, np.array([r0]), np.array([z0]), np.array([r1]), np.array([z1]))
        assert kernels.classify_boundary(masks, r_e, z_e)[0] == kernels.BOUNDARY_INSIDE
    assert NORMAL_MINUS_Z == 1


# -- ledger identities with SEE on ------------------------------------------------------------------------------------------

def _interval_terms(sim: Simulation) -> dict[str, np.ndarray]:
    keys = ("field_work_j", "ke_injected_j", "ke_absorbed_anode_j", "ke_absorbed_exit_j", "ke_absorbed_wall_j", "ke_born_ions_j",
            "inelastic_loss_j", "ke_see_emitted_j")
    out: dict[str, list[float]] = {k: [] for k in ("dke", "rhs", "rhs_without_see", "residual", "h", "see")}
    for a, b in pairwise(sim.series):
        ca, cb = a.ledger["cumulative"], b.ledger["cumulative"]
        d = {key: float(cb.get(key, 0.0) - ca.get(key, 0.0)) for key in keys}
        base = (d["field_work_j"] + d["ke_injected_j"] - d["ke_absorbed_anode_j"] - d["ke_absorbed_exit_j"] - d["ke_absorbed_wall_j"]
                + d["ke_born_ions_j"] - d["inelastic_loss_j"])
        out["dke"].append((b.kinetic_electron_j + b.kinetic_ion_j) - (a.kinetic_electron_j + a.kinetic_ion_j))
        out["rhs"].append(base + d["ke_see_emitted_j"])
        out["rhs_without_see"].append(base)
        out["residual"].append(b.ledger["interval_residual_j"])
        out["h"].append(d["field_work_j"] + (b.field_energy_j - a.field_energy_j) - b.ledger["interval_electrode_work_j"])
        out["see"].append(d["ke_see_emitted_j"])
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


@pytest.mark.parametrize("backend", BACKENDS)
def test_particle_side_identity_closes_to_round_off_with_see_on_and_the_counts_balance(backend: str):
    """dKE = field work + injected - absorbed + born - W(n E)e + EMITTED per record, the recorded residual is H, the
    electron count is seed + injected + ionisations + emitted - absorbed, and the surface charge is absorbed minus emitted."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    see = SEEConfig(material="BN", ion_induced_yield=0.1)
    sim = Simulation(_discharge_config(grid, see=see), linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs)
    seed_count = sim.state.electrons.count
    sim.run(200)
    state = sim.state
    c = state.cumulative
    assert c["see_impacts"] > 50 and c["see_electrons"] > 20 and c["ke_see_emitted_j"] > 0.0
    assert c["see_impacts"] == c["wall_electrons"]                       # channel-only: every wall impact is on the dielectric
    assert 0.3 < c["see_electrons"] / c["see_impacts"] < 1.2
    assert abs(c["see_electrons"] - c["see_yield_sum"]) < 4.0 * np.sqrt(c["see_impacts"]) + 1.0   # unbiased integer yield
    # electron count identity
    expected = seed_count + c["injected_electrons"] + c["ionizations"] + c["see_electrons"] + c["see_ion_induced_electrons"] \
        - c["anode_electrons"] - c["exit_electrons"] - c["wall_electrons"]
    assert state.electrons.count == expected
    # surface charge = absorbed minus emitted (fixed-point deposit: 2^-40 units per contribution)
    quantum = ELEMENTARY_CHARGE_C * 2e6
    expected_surface = quantum * (c["wall_ions"] - c["wall_electrons"] + c["see_electrons"] + c["see_ion_induced_electrons"])
    assert float(state.surface_charge_c.sum()) == pytest.approx(expected_surface, abs=1e-9 * quantum * (c["wall_electrons"] + 1))
    # energy identity
    terms = _interval_terms(sim)
    scale = max(float(np.max(np.abs(terms["dke"]))), float(sim.series[-1].kinetic_electron_j))
    closure = np.abs(terms["dke"] - terms["rhs"])
    assert float(closure.max()) <= 1e-6 * scale, (closure.max(), scale)
    assert np.allclose(terms["residual"], terms["h"], rtol=0.0, atol=1e-6 * scale)
    assert float(terms["see"].sum()) > 1e-3 * scale                    # the emitted energy is a visible term ...
    assert float(np.abs(terms["dke"] - terms["rhs_without_see"]).max()) > 1e3 * float(closure.max())   # ... without which it does not close
    # the record carries the SEE sample and the extra ledger keys; none of them is in the fixed checkpoint key set
    record = sim.series[-1].to_dict()
    assert record["ledger"]["interval_see_emitted_j"] == pytest.approx(terms["see"][-1])
    assert set(SEE_KEYS) <= set(c) and not (set(SEE_KEYS) & set(CUMULATIVE_KEYS)) and not (set(SEE_KEYS) & set(empty_cumulative()))
    assert record["see"]["interval_effective_yield"] >= 0.0 and record["currents_a"]["see_emission_a"] >= 0.0
    assert record["see"]["space_charge_limit_yield"] == pytest.approx(HOBBS_WESSON_CRITICAL_YIELD_XE)


def test_diagnostics_gain_see_profiles_and_frames_only_when_the_wall_emits():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    xs = XenonCrossSections.from_file()
    sim = Simulation(_discharge_config(grid, see=SEEConfig(material="BN")), linear_psi_field_map(grid, 2.0), backend="cpu", cross_sections=xs)
    sim.run(200, accumulate_from_step=50)
    arrays = sim.diagnostic_arrays()
    for key in ("wall_see_flux_per_m2_s", "wall_see_effective_yield", "wall_see_mean_energy_ev"):
        assert key in PROFILE_KEYS and arrays[key].shape == (grid.axial_cells,)
    yields = arrays["wall_see_effective_yield"]
    assert np.all(yields >= 0.0) and np.any(yields > 0.0)
    emitted_total = float(np.sum(arrays["wall_see_flux_per_m2_s"] * 2.0 * np.pi * grid.geometry.wall_radius_m(grid.z_m[:-1] + 0.5 * grid.dz_m) * grid.dz_m
                                 * arrays["window_s"][0])) / 2e6
    assert emitted_total == pytest.approx(sim.backend.diagnostics.wall_see_electrons.sum(), rel=1e-9)
    # frames: the SEE sums are differenced like the others; the interval maps carry the SEE profiles
    sums = sim.diagnostic_sums()
    assert all(key in sums for key in DiagnosticAccumulator.SEE_SUM_KEYS)
    maps = interval_maps(sums, None, masks, 2e6, 5e-12)
    assert np.array_equal(maps["wall_see_effective_yield"], yields)
    # no SEE: nothing added (sums, maps, frames, ledger keys, record)
    off = Simulation(_discharge_config(grid, see=None), linear_psi_field_map(grid, 2.0), backend="cpu", cross_sections=xs)
    off.run(50, accumulate_from_step=25)
    assert not any(key in off.diagnostic_sums() for key in DiagnosticAccumulator.SEE_SUM_KEYS)
    assert not any("see" in key for key in off.diagnostic_arrays())
    assert not any(key in off.state.cumulative for key in SEE_KEYS)
    record = off.series[-1].to_dict()
    assert "see" not in record and "interval_see_emitted_j" not in record["ledger"] and "see_emission_a" not in record["currents_a"]
    # the wall-node set of the SEE diagnostic covers the straight bore's outer grid row
    straight = build_mesh_masks(Grid2D(STRAIGHT_GEOMETRY, 12, 64))
    assert straight.wall_node.sum() == 0 and dielectric_wall_nodes(straight).sum() == straight.unknown_node[-1].sum() > 0


# -- SEE off reproduces v2.0.6 --------------------------------------------------------------------------------------------

def test_see_off_reproduces_the_v2_0_6_identity_ledger_and_tallies():
    """Identity, ledger keys and the integer tallies of a fixed 300-step run equal the pre-v2.2.0 tree (recorded from the
    0901138a checkout on the anchor platform; the full-state comparison - particles, phi, surface charge, cumulative
    ledger, maps - was bitwise on cpu and warp-cpu)."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _discharge_config(grid, see=None, anomalous=AnomalousCollisionConfig(1.0 / 16.0), ion_subcycle=4)
    assert "see" not in config.to_dict() and config.see_active is False
    assert artifacts.config_identity(config) == "9690a3bfb1683749117ac51161a4580e1d55bd2d80ee69e19a7a3fda6cd3caf2"
    assert sorted(empty_cumulative()) == sorted(
        ["anode_electrons", "anode_ions", "exit_electrons", "exit_ions", "wall_electrons", "wall_ions", "injected_electrons",
         "ionizations", "excitations", "elastic", "ke_injected_j", "ke_absorbed_anode_j", "ke_absorbed_exit_j", "ke_absorbed_wall_j",
         "inelastic_loss_j", "ke_born_ions_j", "field_work_j", "pz_impulse", "pz_impulse_electric", "pz_collisions", "pz_born",
         "pz_injected", "pz_exit_electrons", "pz_exit_ions", "pz_wall_electrons", "pz_wall_ions", "pz_anode_electrons", "pz_anode_ions",
         "body_face_electrons", "body_face_ions", "ionizations_plume", "inelastic_loss_per_weight_j"]
    )
    assert tuple(DiagnosticAccumulator.SUM_KEYS) == (
        "n_e", "n_i", "phi", "e_weight", "e_vr", "e_vt", "e_vz", "e_v2", "ionization", "wall_electrons", "wall_ions",
        "wall_electron_energy_j", "wall_ion_energy_j", "exit_ions", "exit_electrons", "side_ions", "side_electrons", "theta_ions", "iedf_ions",
    )
    xs = XenonCrossSections.from_file()
    pins = {
        "cpu": (1418, 1738, {"anode_electrons": 94, "anode_ions": 0, "exit_electrons": 115, "exit_ions": 0, "wall_electrons": 345, "wall_ions": 0,
                             "injected_electrons": 234, "ionizations": 68, "excitations": 61, "elastic": 565, "anomalous": 1322}),
        "warp-cpu": (1438, 1751, {"anode_electrons": 86, "anode_ions": 0, "exit_electrons": 114, "exit_ions": 0, "wall_electrons": 347, "wall_ions": 0,
                                  "injected_electrons": 234, "ionizations": 81, "excitations": 76, "elastic": 553, "anomalous": 1339}),
    }
    for backend, (electrons, ions, tallies) in pins.items():
        if backend != "cpu" and backend not in WARP_BACKENDS:
            continue
        sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs)
        sim.run(300)
        c = sim.state.cumulative
        assert (sim.state.electrons.count, sim.state.ions.count) == (electrons, ions), backend
        assert {key: int(c[key]) for key in tallies} == tallies, backend
        assert sorted(c) == sorted(empty_cumulative()) + ["anomalous"] or sorted(c) == sorted([*empty_cumulative(), "anomalous"])
    # the seed streams of MCC / injection / anomalous scattering / ion-neutral MCC (columns 0-3) are unchanged by the SEE
    # column (4 = stream id 5): the table row is read per column, so a wider row leaves every earlier column's seeds as they were
    try:
        from cft_revival.pic2d.warp_backend import SEED_STREAM_IDS, SEED_STREAMS, stream_seed
        from cft_revival.pic2d.warp_see import SEE_STREAM
    except ImportError:  # pragma: no cover
        return
    assert SEED_STREAMS == 5 and SEED_STREAM_IDS == (1, 2, 3, 4, 5) and SEE_STREAM == 4
    assert len({stream_seed(3, 17, s) for s in SEED_STREAM_IDS}) == 5


# -- Warp: graph bitwise with SEE on; parity with the numpy reference ---------------------------------------------------------

def _assert_same_state(a, b) -> None:
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(a.electrons, name), getattr(b.electrons, name)), name
        assert np.array_equal(getattr(a.ions, name), getattr(b.ions, name)), name
    assert np.array_equal(a.surface_charge_c, b.surface_charge_c)
    assert np.array_equal(a.phi_v, b.phi_v)
    for key, value in a.cumulative.items():
        assert value == pytest.approx(b.cumulative[key], rel=1e-9, abs=1e-300), key


@pytest.mark.skipif("warp-cuda" not in WARP_BACKENDS, reason="CUDA graphs need a CUDA device")
def test_cuda_graph_step_is_bitwise_identical_to_the_direct_launches_with_see_on():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=7,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="device-direct", relative_tolerance=1e-10),
        reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=4, device_sync_steps=25,
        see=SEEConfig(material="BN", ion_induced_yield=0.1),
    )
    direct = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=False)
    graph = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=True)
    direct.run(200)
    graph.run(200)
    assert graph.backend.step_graph_active
    a, b = direct.state, graph.state
    assert a.cumulative["see_electrons"] > 0 and a.cumulative["ionizations"] > 0
    _assert_same_state(a, b)
    # the SEE sample: counts and potentials exact, float-atomic sums (emitted energy / momentum) at round-off
    for x, y in zip(direct.series, graph.series, strict=True):
        for key, value in x.to_dict()["see"].items():
            other = y.to_dict()["see"][key]
            if isinstance(value, bool):
                assert value == other, key
            else:
                assert value == pytest.approx(other, rel=1e-9, abs=1e-12), key


@pytest.mark.parametrize("backend", WARP_BACKENDS)
def test_warp_backends_agree_with_the_numpy_reference_in_distribution_with_see_on(backend: str):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    xs = XenonCrossSections.from_file()
    config = _discharge_config(grid, see=SEEConfig(material="BN", ion_induced_yield=0.1))
    cpu = Simulation(config, field, backend="cpu", cross_sections=xs)
    warp = Simulation(config, field, backend=backend, cross_sections=xs)
    cpu.run(200, accumulate_from_step=100)
    warp.run(200, accumulate_from_step=100)
    a, b = cpu.state.cumulative, warp.state.cumulative
    for key in ("see_impacts", "see_electrons", "wall_electrons"):
        assert b[key] > 0 and abs(a[key] - b[key]) <= 0.25 * max(a[key], b[key]) + 4.0 * np.sqrt(max(a[key], b[key])), key
    assert a["see_electrons"] / a["see_impacts"] == pytest.approx(b["see_electrons"] / b["see_impacts"], abs=0.15)
    assert a["see_backscattered"] / a["see_electrons"] == pytest.approx(b["see_backscattered"] / b["see_electrons"], abs=0.1)
    mean_a = a["ke_see_emitted_j"] / a["see_electrons"]
    mean_b = b["ke_see_emitted_j"] / b["see_electrons"]
    assert mean_a == pytest.approx(mean_b, rel=0.35)
    assert warp.state.electrons.count == pytest.approx(cpu.state.electrons.count, rel=0.1)
    for key in ("wall_see_flux_per_m2_s", "wall_see_effective_yield"):
        assert warp.diagnostic_arrays()[key].shape == cpu.diagnostic_arrays()[key].shape


# -- the physics check: floating dielectric wall facing a Maxwellian plasma --------------------------------------------------

def _slab_drop(delta: float | None, *, backend: str, steps: int = 3000) -> dict[str, float]:
    """Straight bore (r_w 2 mm = 19 lambda_D at 2e16 / 4 eV), grounded end plates, no B, no source: an afterglow slab whose
    dielectric wall floats.  Returns the window-averaged mid-bore bulk-to-wall potential drop, the density-weighted bulk
    T_e, and the wall-defined effective yield (emitted / impacting) over the last third of the run."""

    grid = Grid2D(STRAIGHT_GEOMETRY, 24, 64)
    masks = build_mesh_masks(grid)
    te = 4.0
    see = None if delta is None else SEEConfig(
        yield_model="constant", constant_yield=delta, constant_yield_threshold_ev=3.0, emission_temperature_ev=0.5,
        overrides={"elastic_fraction": 0.0, "inelastic_fraction": 0.0},
    )
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(0.0, 0.0), dt_s=4.0e-12, macro_weight=2.5e5, seed=1,
        seed_plasma=SeedPlasmaConfig(2.0e16, te), reference_density_per_m3=2.0e16, reference_electron_temperature_ev=te,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=100, see=see,
    )
    sim = Simulation(config, zero_field_map(grid), backend=backend)
    sim.run(steps, accumulate_from_step=steps - steps // 3)
    arrays = sim.diagnostic_arrays()
    phi, t_map, weight = arrays["phi_v"], arrays["t_e_ev"], arrays["sample_count_e"]
    j = np.arange(grid.node_shape[1])[None, :]
    mid = np.abs(j - grid.axial_cells // 2) < 8
    bulk = masks.plasma_node & mid & (grid.r_m[:, None] < 0.5 * STRAIGHT_GEOMETRY.bore_radius_m)
    wall = dielectric_wall_nodes(masks) & mid
    c = sim.state.cumulative
    terms = _interval_terms(sim)
    return {
        "drop_v": float(phi[bulk].mean() - phi[wall].mean()),
        "t_e_ev": float(np.sum(t_map[bulk] * weight[bulk]) / np.sum(weight[bulk])),
        "effective_yield": float(c.get("see_electrons", 0.0) / max(c.get("see_impacts", 0.0), 1.0)),
        "closure_j": float(np.max(np.abs(terms["dke"] - terms["rhs"]))), "kinetic_j": float(sim.series[-1].kinetic_electron_j),
    }


def test_floating_dielectric_wall_sheath_drop_follows_hobbs_wesson_with_and_without_see():
    """THE physics check (declared tolerances).  In the afterglow slab the wall charges on the electron time scale, so the
    drop at equal time obeys ``Delta phi(delta) = Delta phi(0) - T_e ln(1 / (1 - delta_eff))`` - the same ``(1 - delta)``
    factor as the Hobbs-Wesson floating-sheath formula (the ion-flux equilibrium value 5.27 T_e needs the ion transit, ~1 us,
    and is not reached here; the DIFFERENCES between yields are the test).  At delta > 1 no cap is imposed: the PIC must form
    the virtual cathode itself, the wall-defined effective yield saturates below 1 and the drop settles at the Hobbs-Wesson
    space-charge-limited value ~1.02 T_e of the primary electrons.

    Tolerances: strict monotone decrease with delta; |Delta phi(0) - Delta phi(delta) - T_e ln(1/(1 - delta_eff))| <= 0.35 T_e
    for delta 0.5 and 0.9; for delta 1.5 and 3.0: drop in [0.7, 1.5] x T_e(delta = 0), both below 0.6 x the delta = 0.9 drop
    (saturation) and within 40 % of each other (the shot noise of a 2000-step window on ~8000 macro-particles is ~0.4 V),
    effective yield in [0.8, 1.0); the particle-side energy identity closed to 1e-9 of the kinetic energy in every case.
    """

    backend = FAST_BACKEND
    results = {delta: _slab_drop(delta, backend=backend) for delta in (None, 0.5, 0.9, 1.5, 3.0)}
    for delta, r in results.items():
        assert r["closure_j"] <= 1e-9 * r["kinetic_j"], (delta, r)
    t_e = results[None]["t_e_ev"]                # the primary population's temperature (the secondaries dilute the bulk T_e)
    drops = [results[d]["drop_v"] for d in (None, 0.5, 0.9, 1.5, 3.0)]
    assert all(a > b for a, b in pairwise(drops)), drops
    assert 2.5 * t_e < drops[0] < 5.5 * t_e, (drops[0], t_e)         # classical (unsaturated) charging: several T_e
    for delta in (0.5, 0.9):
        eff = results[delta]["effective_yield"]
        assert 0.2 < eff < 1.0 and results[None]["effective_yield"] == 0.0
        predicted = t_e * np.log(1.0 / (1.0 - eff))
        measured = drops[0] - results[delta]["drop_v"]
        assert abs(measured - predicted) <= 0.35 * t_e, (delta, eff, measured, predicted, t_e)
    for delta in (1.5, 3.0):
        r = results[delta]
        assert 0.8 <= r["effective_yield"] < 1.0, r                    # space-charge-limited: the wall returns the excess
        assert 0.7 * t_e <= r["drop_v"] <= 1.5 * t_e, (delta, r["drop_v"] / t_e, HOBBS_WESSON_SCL_DROP_TE)
        assert r["drop_v"] <= 0.6 * results[0.9]["drop_v"]              # saturated well below the emitting (delta < 1) sheath
    assert results[1.5]["drop_v"] == pytest.approx(results[3.0]["drop_v"], rel=0.4)
