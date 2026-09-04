"""Model v2.4.0 ``coulomb_v1``: Coulomb collisions by binary pairing per cell (R4 of the physics audit).

Regressions:
* pair kinematics: the centre-of-mass rotation preserves the relative speed, the pair momentum and the classical pair
  energy to round-off, the sampled deflection angle is the one applied, and Nanbu's cumulative angle reproduces
  ``<1 - cos chi> = 1 - exp(-s)`` on every branch;
* the Coulomb logarithms reproduce the NRL values; the config identity carries every parameter; Coulomb off keeps the
  v2.0.6 / v2.2.0 identity, ledger keys, window sums and the pinned integer tallies of a fixed run (bitwise witness);
* relaxation of an anisotropic Maxwellian towards isotropy at the Trubnikov rate (numpy reference AND the Warp stage on
  the CPU device through the standalone harness; the CUDA device when present), the Spitzer / Lorentz slowing-down of a
  drifting electron population on cold xenon ions, the two-temperature electron-ion energy exchange at the Spitzer rate
  across a mass ladder (10, 100, 1000 m_e) with the realised exchange equal to the Landau expectation of the formed
  pairs, and the heavy-ion bound on the per-collision electron energy change;
* per-step conservation in a discharge: ``pz_coulomb`` is round-off, ``ke_coulomb_j`` is the O(v^2/c^2) relativistic
  remainder, the particle-side energy identity closes to round-off with the operator on (cpu, warp-cpu, cuda);
* sub-cycle invariance: k = 1 and k = 2 relax at the same rate within statistics; the same seed replays the same state;
* the CUDA graph replays the direct launches bitwise with Coulomb on; warp-cpu / cuda agree with numpy in distribution;
* the runner accepts ``numerics.coulomb`` and carries the series columns; frames / maps gain the Coulomb maps only when
  the operator is on.
"""

from __future__ import annotations

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.coulomb import (
    COULOMB_KEYS,
    COULOMB_RNG_STREAM,
    CoulombConfig,
    CoulombOperator,
    cell_volumes_m3,
    coulomb_frequencies,
    coulomb_log_ee,
    coulomb_log_ei,
    coulomb_log_ii,
    deflection_parameter,
    nanbu_cos_chi,
    scatter_pairs,
    spitzer_electron_ion_momentum_rate,
    temperature_equilibration_rate,
    trubnikov_isotropization_rate,
)
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map
from cft_revival.pic2d.frames import COULOMB_MAP_KEYS, interval_maps
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    EV_J,
    XENON_MASS_KG,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    ParticleArrays,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
    electron_species,
    xenon_ion_species,
)
from cft_revival.pic2d.sensitivity import AnomalousCollisionConfig
from cft_revival.pic2d.simulation import (
    CUMULATIVE_KEYS,
    DiagnosticAccumulator,
    InjectionConfig,
    PIC2DConfig,
    SeedPlasmaConfig,
    Simulation,
    empty_cumulative,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)
LNL = 10.0


def _warp_backends() -> list[str]:
    try:
        from cft_revival.pic2d.warp_backend import device_available
    except ImportError:  # pragma: no cover
        return []
    return [name for name, device in (("warp-cpu", "cpu"), ("warp-cuda", "cuda:0")) if device_available(device)]


WARP_BACKENDS = _warp_backends()
BACKENDS = ["cpu", *WARP_BACKENDS]
# the standalone relaxation harness runs the numpy reference and the Warp stage on every available device
HARNESSES = ["numpy", *("warp:" + ("cpu" if b == "warp-cpu" else "cuda:0") for b in WARP_BACKENDS)]


# -- helpers --------------------------------------------------------------------------------------------------------------

class _Box:
    """A collisional box: particles frozen in space in a straight bore (no fields, no walls), the operator applied per cycle."""

    def __init__(self, harness: str, config: CoulombConfig, *, ion_mass_kg: float = XENON_MASS_KG, density: float = 1.0e21,
                 electrons: int, radial_cells: int = 8, axial_cells: int = 64, seed: int = 1) -> None:
        self.grid = Grid2D(STRAIGHT_GEOMETRY, radial_cells, axial_cells)
        self.masks = build_mesh_masks(self.grid)
        self.rng = np.random.default_rng(seed)
        self.density = density
        volume = float(np.sum(self.masks.plasma_cell * cell_volumes_m3(self.grid)[:, None]))
        self.weight = density * volume / electrons
        self.ion_mass = ion_mass_kg
        self.harness = harness
        self.step = 0
        if harness == "numpy":
            self.operator = CoulombOperator(config, self.grid, self.masks, self.weight, ion_mass_kg=ion_mass_kg)
        else:
            import warp as wp

            from cft_revival.pic2d.warp_backend import SEED_STREAMS, resolve_device
            from cft_revival.pic2d.warp_coulomb import WarpCoulombStage

            wp.init()
            ion = xenon_ion_species(self.weight)
            ion = type(ion)(ion.name, ion.charge_c, ion_mass_kg, self.weight)
            self.stage = WarpCoulombStage(config, self.grid, self.masks, self.weight, resolve_device(harness.split(":", 1)[1]),
                                          electron=electron_species(self.weight), ion=ion, seed_streams=SEED_STREAMS)

    def positions(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        r = STRAIGHT_GEOMETRY.bore_radius_m * np.sqrt(self.rng.random(n)) * (1.0 - 1e-9)
        z = STRAIGHT_GEOMETRY.z_max_m * self.rng.random(n) * (1.0 - 1e-9)
        return r, z

    def maxwellian(self, n: int, t_par_ev: float, t_perp_ev: float, mass_kg: float, drift_z: float = 0.0) -> ParticleArrays:
        r, z = self.positions(n)
        s_par = np.sqrt(t_par_ev * EV_J / mass_kg)
        s_perp = np.sqrt(t_perp_ev * EV_J / mass_kg)
        return ParticleArrays(r, z, self.rng.normal(0.0, s_perp, n), self.rng.normal(0.0, s_perp, n), self.rng.normal(0.0, s_par, n) + drift_z)

    def cycle(self, electrons: ParticleArrays, ions: ParticleArrays, dt_c: float) -> tuple[ParticleArrays, ParticleArrays, dict[str, float]]:
        self.step += 1
        if self.harness == "numpy":
            result = self.operator.apply(electrons, ions, dt_c, np.random.default_rng([7, self.step, COULOMB_RNG_STREAM]))
            return result.electrons, result.ions, result.tally.to_cumulative()
        return self.stage.apply_host(electrons, ions, dt_c, seed=7, step=self.step)


def _temperatures(p: ParticleArrays, mass_kg: float) -> tuple[float, float]:
    """(T_parallel = z, T_perp) in eV of a particle set (no drift subtraction: the boxes have none)."""

    t_par = mass_kg * float(np.mean(p.vz_m_per_s**2)) / EV_J
    t_perp = mass_kg * 0.5 * float(np.mean(p.vr_m_per_s**2 + p.vt_m_per_s**2)) / EV_J
    return t_par, t_perp


def _discharge_config(grid: Grid2D, *, coulomb: CoulombConfig | None, series: int = 25, seed: int = 3, **extra) -> PIC2DConfig:
    """The warp-parity discharge (300 V, injection, MCC at 1e21, W 2e6) with an optional Coulomb operator."""

    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=seed,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=series, coulomb=coulomb, **extra,
    )


# -- kinematics and sampling ----------------------------------------------------------------------------------------------

def test_pair_rotation_conserves_momentum_relative_speed_and_energy_and_applies_the_sampled_angle():
    rng = np.random.default_rng(0)
    n = 100_000
    va = rng.normal(size=(n, 3)) * 1.0e6
    vb = rng.normal(size=(n, 3)) * 3.0e5
    cos_chi = 1.0 - 2.0 * rng.random(n)
    phi = 2.0 * np.pi * rng.random(n)
    ma, mb = ELECTRON_MASS_KG, 3.0 * ELECTRON_MASS_KG
    va2, vb2 = scatter_pairs(va, vb, ma, mb, cos_chi, phi)
    u, u2 = va - vb, va2 - vb2
    g, g2 = np.linalg.norm(u, axis=1), np.linalg.norm(u2, axis=1)
    assert np.max(np.abs(g2 / g - 1.0)) < 1e-13
    assert np.max(np.abs(np.sum(u * u2, axis=1) / (g * g2) - cos_chi)) < 1e-12
    scale = np.max(np.abs(ma * va + mb * vb))
    assert np.max(np.abs((ma * va2 + mb * vb2) - (ma * va + mb * vb))) < 1e-14 * scale
    ke = 0.5 * ma * np.sum(va**2, 1) + 0.5 * mb * np.sum(vb**2, 1)
    ke2 = 0.5 * ma * np.sum(va2**2, 1) + 0.5 * mb * np.sum(vb2**2, 1)
    assert np.max(np.abs(ke2 - ke) / ke) < 1e-13
    # the azimuth is uniform about u (the two perpendicular components of u' have zero mean and equal variance)
    e1 = np.cross(u, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2, axis=1)[:, None]
    p1, p2 = np.sum(u2 * e1, 1) / g, np.sum(u2 * e2, 1) / g
    assert abs(p1.mean()) < 0.01 and abs(p2.mean()) < 0.01 and abs((p1**2).mean() / (p2**2).mean() - 1.0) < 0.03
    # a relative velocity along z (u_perp = 0) is handled by the special branch, still exactly
    va3 = np.tile([0.0, 0.0, 2.0e6], (5, 1))
    vb3 = np.zeros((5, 3))
    a3, b3 = scatter_pairs(va3, vb3, ma, mb, np.full(5, 0.3), np.linspace(0.0, 6.0, 5))
    assert np.allclose(np.linalg.norm(a3 - b3, axis=1), 2.0e6, rtol=1e-13) and np.allclose(ma * a3 + mb * b3, ma * va3, atol=1e-30)
    # zero relative speed: nothing happens
    a4, b4 = scatter_pairs(va3, va3.copy(), ma, ma, np.full(5, 0.3), np.zeros(5))
    assert np.array_equal(a4, va3) and np.array_equal(b4, va3)


def test_nanbu_cumulative_angle_has_the_exact_mean_on_every_branch_and_the_small_s_limit():
    rng = np.random.default_rng(1)
    u = rng.random(400_000)
    for s in (1e-4, 1e-2, 0.1, 0.19, 0.25, 1.0, 2.9, 3.5, 5.9, 8.0):
        cos_chi = nanbu_cos_chi(np.full(u.size, s), u)
        assert np.all(np.abs(cos_chi) <= 1.0)
        expected = 1.0 - np.exp(-s)
        tolerance = 0.015 * expected + 3.0 * np.sqrt(np.var(cos_chi) / u.size)
        assert abs((1.0 - cos_chi.mean()) - expected) < tolerance, (s, 1.0 - cos_chi.mean(), expected)
    # small s: <chi^2> = 2 s (the Takizuka-Abe variance <delta^2> = s / 2 with delta = tan(chi/2))
    s = 1e-3
    chi = np.arccos(nanbu_cos_chi(np.full(u.size, s), u))
    assert np.mean(chi**2) == pytest.approx(2.0 * s, rel=0.02)
    assert np.mean(np.tan(chi / 2.0) ** 2) == pytest.approx(s / 2.0, rel=0.02)
    # s -> infinity: isotropic
    cos_iso = nanbu_cos_chi(np.full(u.size, 50.0), u)
    assert abs(cos_iso.mean()) < 0.01 and np.mean(cos_iso**2) == pytest.approx(1.0 / 3.0, rel=0.02)
    # the deflection parameter is Nanbu's s = (lnL / 4 pi) (q_a q_b / eps0 m_ab)^2 n dt / g^3
    g = np.array([1.0e6])
    s_ee = deflection_parameter(g, charge_a=-ELEMENTARY_CHARGE_C, charge_b=-ELEMENTARY_CHARGE_C, mass_a=ELECTRON_MASS_KG, mass_b=ELECTRON_MASS_KG,
                                n_field_per_m3=1e18, coulomb_log=LNL, dt_s=1e-12)
    expected = LNL / (4.0 * np.pi) * (ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * 0.5 * ELECTRON_MASS_KG)) ** 2 * 1e18 * 1e-12 / 1e18
    assert float(s_ee[0]) == pytest.approx(expected, rel=1e-12)


def test_coulomb_logarithms_reproduce_the_nrl_formulary_and_the_reference_rates_their_numeric_forms():
    # NRL: lnL_ee = 23.5 - ln(n^1/2 T^-5/4) - [1e-5 + (ln T - 2)^2/16]^1/2 ; lnL_ei = 23 - ln(n^1/2 Z T^-3/2) (T < 10 eV) ; 24 - ln(n^1/2 / T)
    n_cm3, t = 1e12, 7.0     # 1e18 m^-3, 7 eV: the plateau of the audit
    assert float(coulomb_log_ee(1e18, t)) == pytest.approx(23.5 - np.log(np.sqrt(n_cm3) * t**-1.25) - np.sqrt(1e-5 + (np.log(t) - 2.0) ** 2 / 16.0), abs=1e-12)
    assert float(coulomb_log_ei(1e18, t)) == pytest.approx(23.0 - np.log(np.sqrt(n_cm3) * t**-1.5), abs=1e-12)
    assert float(coulomb_log_ei(1e18, 20.0)) == pytest.approx(24.0 - np.log(np.sqrt(n_cm3) / 20.0), abs=1e-12)
    assert float(coulomb_log_ii(1e18, 0.5)) == pytest.approx(23.0 - np.log(np.sqrt(2.0 * n_cm3 / 0.5) / 0.5), abs=1e-12)
    assert 11.5 < float(coulomb_log_ee(1e18, 7.0)) < 12.5 and 11.5 < float(coulomb_log_ei(1e18, 7.0)) < 12.5   # ~12 at the plateau
    assert float(coulomb_log_ee(1e30, 0.01, floor=2.0)) == 2.0     # the floor
    # reference rates in NRL numeric form (n cm^-3, T eV): nu_T(ee, A -> 0) = 8.2e-7 n lnL T^-3/2 ; nu_e = 2.91e-6 n lnL T^-3/2 ;
    # equilibration 1.8e-19 (m_a m_b)^1/2 n lnL / (m_a T_b + m_b T_a)^3/2 with masses in grams
    assert trubnikov_isotropization_rate(1e21, 8.0, 8.0, ELECTRON_MASS_KG, LNL) == pytest.approx(8.2e-7 * 1e15 * LNL * 8.0**-1.5, rel=0.01)
    assert trubnikov_isotropization_rate(1e21, 8.0, 4.0, ELECTRON_MASS_KG, LNL) / trubnikov_isotropization_rate(1e21, 8.0, 8.0, ELECTRON_MASS_KG, LNL) == pytest.approx(1.742, rel=0.01)
    assert spitzer_electron_ion_momentum_rate(1e21, 5.0, LNL) == pytest.approx(2.91e-6 * 1e15 * LNL * 5.0**-1.5, rel=0.01)
    m_e_g, m_i_g = ELECTRON_MASS_KG * 1e3, 100.0 * ELECTRON_MASS_KG * 1e3
    nrl = 1.8e-19 * np.sqrt(m_e_g * m_i_g) * 1e12 * LNL / (m_e_g * 1.0 + m_i_g * 5.0) ** 1.5
    # (NRL's two-digit 1.8e-19 rounds the exact 1.754e-19 up by 2.6 %)
    assert temperature_equilibration_rate(1e18, ELECTRON_MASS_KG, 100.0 * ELECTRON_MASS_KG, 5.0, 1.0, LNL) == pytest.approx(nrl, rel=0.03)


# -- identity and the off state -------------------------------------------------------------------------------------------

def test_config_identity_carries_every_parameter_and_validates():
    base = CoulombConfig()
    record = base.to_dict()
    assert record["model"] == "coulomb_v1" and record["cycle_steps"] == 10 and record["coulomb_log"] == "nrl_local_cell"
    assert record["electron_electron"] and record["electron_ion"] and not record["ion_ion"] and record["method"]["rng_stream"] == 6
    assert CoulombConfig(cycle_steps=5).to_dict() != record and CoulombConfig(coulomb_log_fixed=10.0).to_dict()["coulomb_log"] == "fixed"
    assert CoulombConfig(ion_ion=True).to_dict() != record and CoulombConfig(coulomb_log_floor=3.0).to_dict() != record
    for bad in ({"cycle_steps": 0}, {"cycle_steps": 2.0}, {"coulomb_log_floor": 0.0}, {"coulomb_log_fixed": -1.0}, {"min_temperature_ev": 0.0},
                {"electron_electron": False, "electron_ion": False}, {"enabled": "yes"}):
        with pytest.raises(PIC2DValidationError):
            CoulombConfig(**bad)   # type: ignore[arg-type]
    assert not CoulombConfig(enabled=False, electron_electron=False, electron_ion=False).active
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    on = _discharge_config(grid, coulomb=CoulombConfig())
    off = _discharge_config(grid, coulomb=None)
    assert "coulomb" in on.to_dict() and "coulomb" not in off.to_dict()
    assert on.coulomb_active and not off.coulomb_active
    assert [on.coulomb_step(s) for s in range(10)] == [False] * 9 + [True] and not off.coulomb_step(9)
    assert artifacts.config_identity(on) != artifacts.config_identity(off)
    disabled = _discharge_config(grid, coulomb=CoulombConfig(enabled=False))
    assert not disabled.coulomb_active and artifacts.config_identity(disabled) not in (artifacts.config_identity(on), artifacts.config_identity(off))
    with pytest.raises(PIC2DValidationError):
        _discharge_config(grid, coulomb="on")   # type: ignore[arg-type]


def test_coulomb_off_reproduces_the_v2_2_0_identity_ledger_keys_and_tallies():
    """The v2.2.0 SEE-off pin (test_pic2d_v22_see) re-asserted with the Coulomb field present: identity, ledger keys, window sums and
    the integer tallies of the fixed 300-step run on cpu and warp-cpu are the pre-v2.4.0 values (the seed table gained a 6th
    column without changing columns 0-4)."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    config = _discharge_config(grid, coulomb=None, anomalous=AnomalousCollisionConfig(1.0 / 16.0), ion_subcycle=4)
    assert artifacts.config_identity(config) == "9690a3bfb1683749117ac51161a4580e1d55bd2d80ee69e19a7a3fda6cd3caf2"
    assert not (set(COULOMB_KEYS) & set(CUMULATIVE_KEYS)) and not (set(COULOMB_KEYS) & set(empty_cumulative()))
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
        sim.run(300, accumulate_from_step=200)
        c = sim.state.cumulative
        assert (sim.state.electrons.count, sim.state.ions.count) == (electrons, ions), backend
        assert {key: int(c[key]) for key in tallies} == tallies, backend
        assert not any(key in c for key in COULOMB_KEYS)
        assert not any("coulomb" in key for key in sim.diagnostic_sums()) and not any("coulomb" in key for key in sim.diagnostic_arrays())
        record = sim.series[-1].to_dict()
        assert "coulomb" not in record and "interval_coulomb_ke_j" not in record["ledger"] and not any("coulomb" in k for k in record["currents_a"])
        assert getattr(sim.backend, "coulomb", None) is None
    try:
        from cft_revival.pic2d.warp_backend import SEED_STREAM_IDS, SEED_STREAMS, stream_seed
        from cft_revival.pic2d.warp_coulomb import COULOMB_STREAM
    except ImportError:  # pragma: no cover
        return
    assert SEED_STREAMS == 7 and SEED_STREAM_IDS == (1, 2, 3, 4, 5, 6, 7) and COULOMB_STREAM == 5 and COULOMB_RNG_STREAM == 6   # v2.5.0 appends the neutral column 6
    assert len({stream_seed(3, 17, s) for s in SEED_STREAM_IDS}) == 7


# -- relaxation physics (numpy reference and the Warp stage) --------------------------------------------------------------------

@pytest.mark.parametrize("harness", HARNESSES)
def test_anisotropic_maxwellian_relaxes_towards_isotropy_at_the_trubnikov_rate(harness: str):
    """Electrons only, T_par 8 / T_perp 4 eV (A = -0.5) and 5.5 / 4.75 eV (A = -0.14) at 1e21 m^-3, lnL fixed 10: the log decay of
    T_perp - T_par over the run equals the Trubnikov / NRL rate integrated along the measured temperatures (d(T_perp - T_par)/dt =
    -3 nu_T(A) (T_perp - T_par)) within 10 % - with mean s per pair ~0.05 (the deviation from a bi-Maxwellian path at large |A| and
    the finite-s bias are both inside that band; the numpy reference gave 0.96 / 1.05 at N = 200k)."""

    n_e = 60_000 if harness == "numpy" else 40_000
    cycles = 120
    for t_par, t_perp in ((8.0, 4.0), (5.5, 4.75)):
        box = _Box(harness, CoulombConfig(electron_ion=False, coulomb_log_fixed=LNL, cycle_steps=1), electrons=n_e, seed=1)
        electrons = box.maxwellian(n_e, t_par, t_perp, ELECTRON_MASS_KG)
        ions = ParticleArrays.empty()
        dt_c = 5.0e-4 / trubnikov_isotropization_rate(box.density, t_par, t_perp, ELECTRON_MASS_KG, LNL)
        history = []
        s_sum = pairs = 0.0
        for _ in range(cycles):
            history.append(_temperatures(electrons, ELECTRON_MASS_KG))
            electrons, ions, tally = box.cycle(electrons, ions, dt_c)
            s_sum += tally["coulomb_ee_s_sum"]
            pairs += tally["coulomb_ee_pairs"]
        t_par_h, t_perp_h = np.array(history).T
        diff = t_perp_h - t_par_h
        rates = np.array([trubnikov_isotropization_rate(box.density, a, b, ELECTRON_MASS_KG, LNL) for a, b in zip(t_par_h, t_perp_h)])
        reference = -3.0 * float(np.sum(rates[:-1] * dt_c))
        measured = float(np.log(diff[-1] / diff[0]))
        assert 0.15 < -reference < 0.6                                   # the run relaxes by a measurable, not saturated, amount
        assert measured / reference == pytest.approx(1.0, abs=0.10), (harness, t_par, t_perp, measured, reference)
        assert 0.01 < s_sum / pairs < 0.2                              # small-angle regime per cycle
        assert pairs >= cycles * (n_e // 2)                              # every electron paired each cycle (triplets add sub-pairs)
        # total energy conserved by the operator: T_par + 2 T_perp constant to round-off (no field, no motion)
        total = t_par_h + 2.0 * t_perp_h
        assert np.max(np.abs(total - total[0])) < 1e-9 * total[0]


@pytest.mark.parametrize("harness", HARNESSES)
def test_drifting_electrons_slow_down_on_cold_xenon_ions_at_the_spitzer_lorentz_rate(harness: str):
    """e-i only, Maxwellian electrons at 5 eV drifting at 0.3 v_th against cold Xe+ at 1e21 m^-3: the momentum lost per cycle equals
    the Lorentz-gas expectation of the actual distribution, -<(1 - exp(-s_k)) (m_ei / m_e) v_z,k> with the per-electron deflection
    parameter s_k = (lnL / 4 pi) (e^2 / eps0 m_ei)^2 n_i dt_c / v_k^3 (the bounded form: the linear sum_k nu_s(v_k) v_z,k has an
    infinite-variance 1/v^3 tail), within 10 %; the initial slope of the shifted Maxwellian is the Braginskii rate 2.91e-6 n lnL
    T^-3/2 within 10 % (the saturated slow tail holds ~2 % of the linear integral); the momentum the electrons lose is the momentum
    the ions gain (pairwise conservation)."""

    n = 100_000 if harness == "numpy" else 50_000
    cycles = 150
    t_e = 5.0
    box = _Box(harness, CoulombConfig(electron_electron=False, coulomb_log_fixed=LNL, cycle_steps=1), electrons=n, seed=2)
    v_th = np.sqrt(t_e * EV_J / ELECTRON_MASS_KG)
    electrons = box.maxwellian(n, t_e, t_e, ELECTRON_MASS_KG, drift_z=0.3 * v_th)
    r, z = box.positions(n)
    ions = ParticleArrays(r, z, np.zeros(n), np.zeros(n), np.zeros(n))
    nu_ei = spitzer_electron_ion_momentum_rate(box.density, t_e, LNL)
    dt_c = 1.0e-3 / nu_ei
    reduced = ELECTRON_MASS_KG * XENON_MASS_KG / (ELECTRON_MASS_KG + XENON_MASS_KG)
    s_factor = LNL / (4.0 * np.pi) * (ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * reduced)) ** 2 * box.density * dt_c
    predicted = measured = 0.0
    pz0 = ELECTRON_MASS_KG * float(np.sum(electrons.vz_m_per_s)) + XENON_MASS_KG * float(np.sum(ions.vz_m_per_s))
    first_slope = None
    for _ in range(cycles):
        speed = np.sqrt(electrons.speed_squared())
        s = s_factor / speed**3
        expected_change = -float(np.mean(-np.expm1(-s) * (reduced / ELECTRON_MASS_KG) * electrons.vz_m_per_s))   # expectation of the operator
        if first_slope is None:
            first_slope = -expected_change / dt_c / float(np.mean(electrons.vz_m_per_s))
        before = float(np.mean(electrons.vz_m_per_s))
        electrons, ions, tally = box.cycle(electrons, ions, dt_c)
        predicted += expected_change
        measured += float(np.mean(electrons.vz_m_per_s)) - before
        assert tally["coulomb_ei_pairs"] == n                            # every electron collides once per cycle
    assert first_slope == pytest.approx(nu_ei, rel=0.10)                # Maxwellian initial slope = Braginskii nu_ei
    assert 0.06 < -predicted / (0.3 * v_th) < 0.5                        # a measurable decay (the distribution deforms; not a single exponential)
    assert measured / predicted == pytest.approx(1.0, abs=0.10), (harness, measured, predicted)
    pz1 = ELECTRON_MASS_KG * float(np.sum(electrons.vz_m_per_s)) + XENON_MASS_KG * float(np.sum(ions.vz_m_per_s))
    assert abs(pz1 - pz0) < 1e-10 * abs(ELECTRON_MASS_KG * 0.3 * v_th * n)
    assert float(np.sum(ions.vz_m_per_s)) > 0.0                          # the ions took the drift momentum


def test_two_temperature_equilibration_follows_the_spitzer_rate_across_a_mass_ladder_and_the_heavy_ion_bound():
    """e-i only, Maxwellian electrons (5 eV) against ions at equal density.  Mass ladder m_i / m_e in {10, 100, 1000} with cold ions:
    the Landau integral the formed pairs realise in expectation per cycle, -sum_pairs s m_ab V.u (linear in dt, exact for any s),
    equals Spitzer's nu_eps (T_i - T_e) (3/2) N k within 8 % - the mass scaling sqrt(m_e m_i) / (m_i T_e + m_e T_i)^3/2 over a factor
    100 in mass (the electrons' energy loss rate scales as 1/m_i).  With warm ions (1 eV) at m_i = 10 m_e and small s the realised
    exchange over 60 cycles equals the expectation -sum (1 - exp(-s)) m_ab V.u within 15 % (angle sampling), and that expectation is
    Spitzer's within the finite-s factor.  Heavy-ion bound (xenon, ions at rest): a collision changes an electron's energy by at most
    4 m_e / M of it (exact kinematics) and keeps its speed to 1e-4; with moving ions the per-collision change carries the ion-motion
    term 2 (1 - cos chi) v_i . v_e / v_e^2 that averages out (Spitzer's T_i term)."""

    n = 60_000
    t_e = 5.0

    def instrument(operator: CoulombOperator, sums: dict[str, float]) -> None:
        original = operator._collide

        def instrumented(va, vb, ma, mb, qa, qb, n_field, lnl, dt, rng, _orig=original):
            u = va - vb
            g = np.linalg.norm(u, axis=1)
            total = ma + mb
            centre_dot_u = np.sum(((ma * va + mb * vb) / total) * u, axis=1)
            s = deflection_parameter(g, charge_a=qa, charge_b=qb, mass_a=ma, mass_b=mb, n_field_per_m3=n_field, coulomb_log=lnl, dt_s=1.0) * dt
            sums["landau"] += -float(np.sum(s * (ma * mb / total) * centre_dot_u))
            sums["expected"] += -float(np.sum((1.0 - np.exp(-s)) * (ma * mb / total) * centre_dot_u))
            va2, vb2, s2 = _orig(va, vb, ma, mb, qa, qb, n_field, lnl, dt, rng)
            sums["realised"] += 0.5 * ma * float(np.sum(np.sum(va2**2, 1) - np.sum(va**2, 1)))
            return va2, vb2, s2

        operator._collide = instrumented       # type: ignore[method-assign]

    for ratio in (10.0, 100.0, 1000.0):
        m_i = ratio * ELECTRON_MASS_KG
        box = _Box("numpy", CoulombConfig(electron_electron=False, coulomb_log_fixed=LNL, cycle_steps=1), ion_mass_kg=m_i, electrons=n, seed=int(ratio))
        electrons = box.maxwellian(n, t_e, t_e, ELECTRON_MASS_KG)
        r, z = box.positions(n)
        ions = ParticleArrays(r, z, np.zeros(n), np.zeros(n), np.zeros(n))
        sums = {"landau": 0.0, "expected": 0.0, "realised": 0.0}
        instrument(box.operator, sums)
        dt_c = 2.0e-4 / temperature_equilibration_rate(box.density, ELECTRON_MASS_KG, m_i, t_e, 0.0, LNL)
        spitzer = 0.0
        for _ in range(20):
            te_now = ELECTRON_MASS_KG * float(np.mean(electrons.speed_squared())) / (3.0 * EV_J)
            ti_now = m_i * float(np.mean(ions.speed_squared())) / (3.0 * EV_J)
            spitzer += temperature_equilibration_rate(box.density, ELECTRON_MASS_KG, m_i, te_now, ti_now, LNL) * (ti_now - te_now) * dt_c * 1.5 * EV_J * n
            electrons, ions, _ = box.cycle(electrons, ions, dt_c)
        assert sums["landau"] / spitzer == pytest.approx(1.0, abs=0.08), (ratio, sums, spitzer)
        assert sums["realised"] < 0.0 and float(np.mean(ions.speed_squared())) > 0.0     # the electrons cooled, the ions warmed
    # warm ions, light mass, small s: the realised exchange is the pair expectation
    m_i = 10.0 * ELECTRON_MASS_KG
    box = _Box("numpy", CoulombConfig(electron_electron=False, coulomb_log_fixed=LNL, cycle_steps=1), ion_mass_kg=m_i, electrons=n, seed=3)
    electrons = box.maxwellian(n, t_e, t_e, ELECTRON_MASS_KG)
    ions = box.maxwellian(n, 1.0, 1.0, m_i)
    sums = {"landau": 0.0, "expected": 0.0, "realised": 0.0}
    instrument(box.operator, sums)
    dt_c = 2.0e-4 / temperature_equilibration_rate(box.density, ELECTRON_MASS_KG, m_i, t_e, 1.0, LNL)
    spitzer = 0.0
    for _ in range(60):
        te_now = ELECTRON_MASS_KG * float(np.mean(electrons.speed_squared())) / (3.0 * EV_J)
        ti_now = m_i * float(np.mean(ions.speed_squared())) / (3.0 * EV_J)
        spitzer += temperature_equilibration_rate(box.density, ELECTRON_MASS_KG, m_i, te_now, ti_now, LNL) * (ti_now - te_now) * dt_c * 1.5 * EV_J * n
        electrons, ions, _ = box.cycle(electrons, ions, dt_c)
    assert sums["realised"] / sums["expected"] == pytest.approx(1.0, abs=0.15), sums
    assert sums["landau"] / spitzer == pytest.approx(1.0, abs=0.08) and 0.9 < sums["expected"] / sums["landau"] <= 1.0
    # heavy-ion bound with the xenon mass and ions at rest: |dE_e| / E_e <= 4 m_e / M per collision with an ion at rest (exact
    # kinematics); an ion hit by a second electron in the same cycle already moves at ~(m_e / M) v_e, which adds the ion-motion term of
    # the same order - so the bound over the cycle is a few times 4 m_e / M; the electron population loses energy, the ions gain it
    box = _Box("numpy", CoulombConfig(electron_electron=False, coulomb_log_fixed=LNL, cycle_steps=1), electrons=20_000, seed=5)
    electrons = box.maxwellian(20_000, t_e, t_e, ELECTRON_MASS_KG)
    r, z = box.positions(20_000)
    ions = ParticleArrays(r, z, np.zeros(20_000), np.zeros(20_000), np.zeros(20_000))
    before = electrons.speed_squared()
    after, ions_after, tally = box.cycle(electrons, ions, 1.0e-11)
    change = np.abs(after.speed_squared() - before) / before
    assert tally["coulomb_ei_pairs"] == 20_000
    assert float(np.max(change)) <= 4.0 * 4.0 * ELECTRON_MASS_KG / XENON_MASS_KG and float(np.max(change)) < 1e-4
    assert float(np.sum(after.speed_squared())) < float(np.sum(before)) and float(np.sum(ions_after.speed_squared())) > 0.0
    assert np.mean(after.speed_squared() <= before * (1.0 + 1e-12)) > 0.97       # almost every electron only lost energy


def test_sub_cycle_invariance_and_same_seed_replay():
    """k = 1 with dt_c and k = 2 with 2 dt_c relax an anisotropic electron population at the same rate within statistics (the operator
    is applied at the accumulated interval); the same seed and step replay the same state bitwise; a different step differs."""

    n = 80_000
    decays = {}
    for k in (1, 2):
        box = _Box("numpy", CoulombConfig(electron_ion=False, coulomb_log_fixed=LNL, cycle_steps=k), electrons=n, seed=11)
        electrons = box.maxwellian(n, 8.0, 4.0, ELECTRON_MASS_KG)     # the same initial particles for both k (same seed)
        dt = 4.0e-4 / trubnikov_isotropization_rate(box.density, 8.0, 4.0, ELECTRON_MASS_KG, LNL)
        d0 = np.subtract(*_temperatures(electrons, ELECTRON_MASS_KG)[::-1])
        for _ in range(200 // k):
            electrons, _, _ = box.cycle(electrons, ParticleArrays.empty(), k * dt)
        decays[k] = float(np.log(np.subtract(*_temperatures(electrons, ELECTRON_MASS_KG)[::-1]) / d0))
    assert decays[1] < -0.15 and decays[1] == pytest.approx(decays[2], abs=0.04), decays
    box = _Box("numpy", CoulombConfig(coulomb_log_fixed=None, cycle_steps=1, ion_ion=True), electrons=3000, seed=12)
    electrons = box.maxwellian(3000, 5.0, 5.0, ELECTRON_MASS_KG)
    ions = box.maxwellian(3000, 0.2, 0.2, XENON_MASS_KG)
    rng_a = np.random.default_rng([7, 3, COULOMB_RNG_STREAM])
    rng_b = np.random.default_rng([7, 3, COULOMB_RNG_STREAM])
    a = box.operator.apply(electrons, ions, 1e-12, rng_a)
    b = box.operator.apply(electrons, ions, 1e-12, rng_b)
    c = box.operator.apply(electrons, ions, 1e-12, np.random.default_rng([7, 4, COULOMB_RNG_STREAM]))
    for name in ("vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(a.electrons, name), getattr(b.electrons, name)) and np.array_equal(getattr(a.ions, name), getattr(b.ions, name))
        assert not np.array_equal(getattr(a.electrons, name), getattr(c.electrons, name))
    assert a.tally.ii_pairs > 0 and a.tally.ee_pairs > 0 and a.tally.ei_pairs > 0 and a.tally.to_cumulative() == b.tally.to_cumulative()
    # the frequency helper: nu_ee = 2 sum s / (N_e dt_c), nu_ei = sum s / (N_e dt_c)
    freq = coulomb_frequencies(a.tally.to_cumulative(), 1e-12)
    assert freq["nu_ee_mean_per_s"] == pytest.approx(2.0 * a.tally.ee_s_sum / (3000 * 1e-12)) and freq["nu_ei_mean_per_s"] == pytest.approx(a.tally.ei_s_sum / (3000 * 1e-12))


# -- the operator inside the PIC cycle -------------------------------------------------------------------------------------------

def _interval_terms(sim: Simulation) -> dict[str, np.ndarray]:
    """Per-record particle-side terms: dKE vs field work + injected - absorbed + born - W n E + Coulomb (relativistic remainder)."""

    out: dict[str, list[float]] = {"dke": [], "rhs": [], "coulomb": []}
    for a, b in zip(sim.series[:-1], sim.series[1:]):
        d = {key: b.ledger["cumulative"][key] - a.ledger["cumulative"].get(key, 0.0) for key in b.ledger["cumulative"]}
        dke = (b.kinetic_electron_j + b.kinetic_ion_j) - (a.kinetic_electron_j + a.kinetic_ion_j)
        rhs = (d["field_work_j"] + d["ke_injected_j"] - d["ke_absorbed_anode_j"] - d["ke_absorbed_exit_j"] - d["ke_absorbed_wall_j"]
               + d["ke_born_ions_j"] - d["inelastic_loss_j"] + d.get("ke_coulomb_j", 0.0))
        out["dke"].append(dke)
        out["rhs"].append(rhs)
        out["coulomb"].append(d.get("ke_coulomb_j", 0.0))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


@pytest.mark.parametrize("backend", BACKENDS)
def test_conservation_per_step_in_a_discharge_and_the_ledger_terms(backend: str):
    """pz_coulomb is round-off of the represented momentum, ke_coulomb_j is the O(v^2/c^2) relativistic remainder, the particle-side
    energy identity closes to round-off with the operator on, the counts / keys / record block are present."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    config = _discharge_config(grid, coulomb=CoulombConfig(cycle_steps=2), ion_subcycle=4)
    sim = Simulation(config, linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs)
    sim.run(200, accumulate_from_step=100)
    state = sim.state
    c = state.cumulative
    assert set(COULOMB_KEYS) <= set(c) and c["coulomb_cycles"] == 100 and c["coulomb_ee_pairs"] > 1000 and c["coulomb_ei_pairs"] > 1000
    assert c["coulomb_ii_pairs"] == 0 and c["coulomb_electron_cycles"] > c["coulomb_ee_pairs"]
    momentum_scale = ELECTRON_MASS_KG * 2e6 * float(np.sum(np.abs(state.electrons.vz_m_per_s))) * c["coulomb_cycles"]
    assert abs(c["pz_coulomb"]) < 1e-9 * momentum_scale
    kinetic = sim.series[-1].kinetic_electron_j
    assert abs(c["ke_coulomb_j"]) < 1e-4 * kinetic           # O(v^2/c^2 x redistributed fraction) per cycle, summed over 100 cycles
    terms = _interval_terms(sim)
    scale = max(float(np.max(np.abs(terms["dke"]))), kinetic)
    assert float(np.max(np.abs(terms["dke"] - terms["rhs"]))) <= 1e-6 * scale
    record = sim.series[-1].to_dict()
    block = record["coulomb"]
    assert block["interval_cycles"] > 0 and block["nu_ee_mean_per_s"] > 0.0 and block["nu_ei_mean_per_s"] > 0.0 and block["nu_ii_mean_per_s"] == 0.0
    assert 5.0 < block["mean_coulomb_log_ee"] < 20.0 and 5.0 < block["mean_coulomb_log_ei"] < 20.0
    assert 0.0 <= block["fraction_large_s_ee"] <= 1.0 and block["nu_en_elastic_mean_per_s"] > 0.0 and block["nu_ee_over_nu_en"] > 0.0
    # the NRL electron collision rate at the record's peak node (the audit's definition) beside the operator's pair-mean rate
    peak = record["peak_node"]
    expected_spitzer = spitzer_electron_ion_momentum_rate(peak["n_e_peak_per_m3"], peak["t_e_peak_ev"], float(coulomb_log_ee(peak["n_e_peak_per_m3"], peak["t_e_peak_ev"])))
    assert block["nu_e_spitzer_peak_per_s"] == pytest.approx(expected_spitzer) and block["nu_e_spitzer_peak_over_nu_en"] > 0.0
    assert block["cycle_dt_s"] == pytest.approx(2.0 * config.dt_s) and record["ledger"]["interval_coulomb_ke_j"] == pytest.approx(terms["coulomb"][-1])
    assert record["currents_a"]["coulomb_nu_ee_mean_per_s"] == block["nu_ee_mean_per_s"]
    # window maps: cell-layout Coulomb frequencies (node-shaped arrays, last row / column zero), positive where electrons were
    arrays = sim.diagnostic_arrays()
    for key in COULOMB_MAP_KEYS:
        assert arrays[key].shape == grid.node_shape and np.all(arrays[key][-1, :] == 0.0) and np.all(arrays[key][:, -1] == 0.0)
    assert np.any(arrays["coulomb_nu_ee_per_s"] > 0.0) and np.all(arrays["coulomb_nu_ee_per_s"] >= 0.0)
    sums = sim.diagnostic_sums()
    assert all(key in sums for key in DiagnosticAccumulator.COULOMB_SUM_KEYS)
    maps = interval_maps(sums, None, build_mesh_masks(grid), 2e6, 5e-12)
    assert np.allclose(maps["coulomb_nu_ee_per_s"], arrays["coulomb_nu_ee_per_s"])
    prov = sim.to_provenance()
    assert prov["coulomb"]["config"]["model"] == "coulomb_v1" and prov["v1_4_options"]["coulomb"]["cycle_steps"] == 2


@pytest.mark.parametrize("backend", WARP_BACKENDS)
def test_warp_backends_agree_with_the_numpy_reference_in_distribution_with_coulomb_on(backend: str):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 2.0)
    xs = XenonCrossSections.from_file()
    config = _discharge_config(grid, coulomb=CoulombConfig(cycle_steps=2), ion_subcycle=4)
    cpu = Simulation(config, field, backend="cpu", cross_sections=xs)
    warp = Simulation(config, field, backend=backend, cross_sections=xs)
    cpu.run(200, accumulate_from_step=100)
    warp.run(200, accumulate_from_step=100)
    a, b = cpu.state.cumulative, warp.state.cumulative
    for key in ("coulomb_ee_pairs", "coulomb_ei_pairs", "coulomb_electron_cycles"):
        assert b[key] > 0 and abs(a[key] - b[key]) <= 0.1 * max(a[key], b[key]), key
    assert a["coulomb_ee_lnl_sum"] / a["coulomb_ee_pairs"] == pytest.approx(b["coulomb_ee_lnl_sum"] / b["coulomb_ee_pairs"], rel=0.05)
    ra, rb = cpu.series[-1].to_dict()["coulomb"], warp.series[-1].to_dict()["coulomb"]
    assert rb["nu_ee_mean_per_s"] == pytest.approx(ra["nu_ee_mean_per_s"], rel=0.6) and rb["nu_ei_mean_per_s"] == pytest.approx(ra["nu_ei_mean_per_s"], rel=0.6)
    assert warp.state.electrons.count == pytest.approx(cpu.state.electrons.count, rel=0.1)
    assert warp.diagnostic_arrays()["coulomb_nu_ee_per_s"].shape == cpu.diagnostic_arrays()["coulomb_nu_ee_per_s"].shape


def _assert_same_state(a, b) -> None:
    for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
        assert np.array_equal(getattr(a.electrons, name), getattr(b.electrons, name)), name
        assert np.array_equal(getattr(a.ions, name), getattr(b.ions, name)), name
    assert np.array_equal(a.surface_charge_c, b.surface_charge_c)
    assert np.array_equal(a.phi_v, b.phi_v)
    for key, value in a.cumulative.items():
        assert value == pytest.approx(b.cumulative[key], rel=1e-9, abs=1e-300), key


@pytest.mark.skipif("warp-cuda" not in WARP_BACKENDS, reason="CUDA graphs need a CUDA device")
def test_cuda_graph_step_is_bitwise_identical_to_the_direct_launches_with_coulomb_on():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    config = PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=7,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="device-direct", relative_tolerance=1e-10),
        reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=4, device_sync_steps=25,
        coulomb=CoulombConfig(cycle_steps=3, ion_ion=True),
    )
    direct = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=False)
    graph = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=True)
    direct.run(200)
    graph.run(200)
    assert graph.backend.step_graph_active
    a, b = direct.state, graph.state
    assert a.cumulative["coulomb_ee_pairs"] > 0 and a.cumulative["coulomb_ei_pairs"] > 0 and a.cumulative["coulomb_ii_pairs"] > 0
    _assert_same_state(a, b)
    for x, y in zip(direct.series, graph.series, strict=True):
        for key, value in x.to_dict()["coulomb"].items():
            other = y.to_dict()["coulomb"][key]
            if value is None or other is None:
                assert value == other, key
            else:
                assert value == pytest.approx(other, rel=1e-9, abs=1e-12), key
    # the same seed replays the same state (a second direct run)
    again = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=False)
    again.run(200)
    _assert_same_state(a, again.state)


@pytest.mark.skipif("warp-cuda" not in WARP_BACKENDS, reason="CUDA parity of the standalone stage")
def test_cuda_stage_matches_the_cpu_stage_in_distribution_and_replays_its_own_seed():
    n = 30_000
    results = {}
    for device in ("cpu", "cuda:0"):
        box = _Box("warp:" + device, CoulombConfig(coulomb_log_fixed=None, cycle_steps=1, ion_ion=True), electrons=n, seed=21)
        electrons = box.maxwellian(n, 8.0, 4.0, ELECTRON_MASS_KG)
        ions = box.maxwellian(n, 0.3, 0.3, XENON_MASS_KG)
        a, _, ta = box.stage.apply_host(electrons, ions, 2e-12, seed=9, step=4)
        b, _, tb = box.stage.apply_host(electrons, ions, 2e-12, seed=9, step=4)
        assert np.array_equal(a.vr_m_per_s, b.vr_m_per_s) and ta["coulomb_ee_pairs"] == tb["coulomb_ee_pairs"]
        results[device] = ta
    for key in ("coulomb_ee_pairs", "coulomb_ei_pairs", "coulomb_ii_pairs"):
        assert results["cpu"][key] == pytest.approx(results["cuda:0"][key], rel=0.02), key
    for key in ("coulomb_ee_s_sum", "coulomb_ei_s_sum", "coulomb_ee_lnl_sum"):
        assert results["cpu"][key] == pytest.approx(results["cuda:0"][key], rel=0.2), key


# -- runner ---------------------------------------------------------------------------------------------------------------------

def test_runner_reads_the_coulomb_block_and_carries_the_series_columns():
    from experiments.pic2d_cft_steady_state_v1 import run as runner
    from experiments.pic2d_cft_steady_state_v4 import run as v4

    protocol = runner.load_protocol(v4.PROTOCOL_PATH)
    off = runner.build_config(protocol, backend="cpu")
    assert off.coulomb is None
    protocol["numerics"]["coulomb"] = {"enabled": True, "cycle_steps": 10, "ion_ion": False, "coulomb_note": "R4 shakedown"}
    on = runner.build_config(protocol, backend="cpu")
    assert on.coulomb is not None and on.coulomb.cycle_steps == 10 and on.coulomb_active
    assert artifacts.config_identity(on) != artifacts.config_identity(off)
    record = {
        "step": 10, "time_s": 1e-11, "electrons": 5, "ions": 5, "phi_mean_v": 1.0, "phi_min_v": 0.0, "phi_max_v": 2.0, "kinetic_electron_j": 1.0,
        "kinetic_ion_j": 1.0, "field_energy_j": 1.0, "surface_charge_c": 0.0, "peak_omega_pe_dt": 0.1, "poisson_iterations": 0,
        "currents_a": {"discharge_a": 1.0}, "ledger": {key: 0.0 for key in runner.LEDGER_SCALARS} | {"interval_coulomb_ke_j": 1e-20},
        "coulomb": {key: 1.0 for key in runner.COULOMB_SCALARS},
    }
    arrays = runner.records_to_arrays([record])
    assert all(f"coulomb_{key}" in arrays for key in runner.COULOMB_SCALARS) and arrays["interval_coulomb_ke_j"][0] == 1e-20
    plain = dict(record)
    del plain["coulomb"]
    plain["ledger"] = {key: 0.0 for key in runner.LEDGER_SCALARS}
    arrays = runner.records_to_arrays([plain])
    assert not any(key.startswith("coulomb_") for key in arrays) and np.isnan(arrays["interval_coulomb_ke_j"][0])
