"""Model v2.0 (plume block): L-shaped domain, cathode region, two-zone neutrals, momentum/thrust ledgers, gates."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from cft_revival.pic2d import kernels
from cft_revival.pic2d.fields import uniform_field_map, zero_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    EPSILON_0_F_PER_M,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DStabilityError,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.neutrals import NeutralInventoryConfig
from cft_revival.pic2d.poisson import BlockTridiagonalSolver, Poisson2D, apply_operator, electric_field_nodes, induced_electrode_charge_c
from cft_revival.pic2d.simulation import (
    MOMENTUM_KEYS,
    CathodeConfig,
    InjectionConfig,
    PIC2DConfig,
    PlumeBoundaryGateConfig,
    SeedPlasmaConfig,
    Simulation,
    boundary_forces_n,
    cathode_sample,
    empty_cumulative,
    neutral_shape_cells,
    plume_neutral_shape,
)

ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
XENON_MASS_KG = 2.1801714e-25

# small L-shaped development geometry: 2 mm bore, 8 mm channel with a 2 mm cone to 3 mm,
# plume box 6 mm x 4 mm, dielectric front face to 4 mm, grounded beyond (all on 0.25 mm lines)
PLUME_GEOMETRY = ChannelGeometry(
    2.0e-3, 0.0, 8.0e-3, 6.0e-3, 3.0e-3, plume_radius_m=6.0e-3, plume_length_m=4.0e-3, body_dielectric_radius_m=4.0e-3
)
CHANNEL_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 6.0e-3, 3.0e-3)
UA = 300.0


def plume_grid(cells_per_mm: int = 4) -> Grid2D:
    return Grid2D(PLUME_GEOMETRY, 6 * cells_per_mm, 12 * cells_per_mm)


def cathode(current_a: float = 1e-3, rule: str = "fixed", **extra) -> CathodeConfig:
    return CathodeConfig(3.5e-3, 4.5e-3, 9.0e-3, 10.0e-3, 2.0, current_a, current_rule=rule, **extra)  # type: ignore[arg-type]


def config(grid: Grid2D, *, cathode_config: CathodeConfig | None = None, mcc: bool = False, seed_density: float = 1e15,
           gate: PlumeBoundaryGateConfig | None = None, inventory: NeutralInventoryConfig | None = None, **overrides) -> PIC2DConfig:
    interval = overrides.pop("series_interval_steps", 10)
    keywords: dict = dict(
        dt_s=5e-12, macro_weight=2e5, seed=7, reference_density_per_m3=1e15, reference_electron_temperature_ev=5.0,
        series_interval_steps=interval, runtime_stability_check_steps=interval,
    ) | overrides
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(UA, 0.0), cathode=cathode_config,
        seed_plasma=SeedPlasmaConfig(seed_density, 5.0) if seed_density > 0 else None,
        mcc=MCCConfig(1.5e19) if mcc else None, poisson=PoissonConfig2D(method="direct", relative_tolerance=1e-10),
        limits=StabilityLimits(max_cell_debye_ratio=2.0), plume_boundary_gate=gate, neutral_inventory=inventory, **keywords,
    )


# -- geometry and mask contracts ----------------------------------------------------------

def test_plume_geometry_contract():
    assert PLUME_GEOMETRY.has_plume and PLUME_GEOMETRY.max_radius_m == 6.0e-3 and PLUME_GEOMETRY.domain_z_max_m == 12.0e-3
    assert PLUME_GEOMETRY.channel_length_m == 8.0e-3 and PLUME_GEOMETRY.length_m == 12.0e-3
    assert not CHANNEL_GEOMETRY.has_plume and CHANNEL_GEOMETRY.length_m == 8.0e-3
    with pytest.raises(PIC2DValidationError):
        ChannelGeometry(2e-3, 0.0, 8e-3, 6e-3, 3e-3, plume_radius_m=6e-3)  # radius without length
    with pytest.raises(PIC2DValidationError):
        ChannelGeometry(2e-3, 0.0, 8e-3, 6e-3, 3e-3, plume_radius_m=3e-3, plume_length_m=4e-3)  # no front face
    with pytest.raises(PIC2DValidationError):
        ChannelGeometry(2e-3, 0.0, 8e-3, 6e-3, 3e-3, plume_radius_m=6e-3, plume_length_m=4e-3, body_dielectric_radius_m=7e-3)
    with pytest.raises(PIC2DValidationError):
        ChannelGeometry(2e-3, 0.0, 8e-3, 6e-3, 3e-3, body_dielectric_radius_m=4e-3)
    # wall radius jumps to the box radius at the exit plane
    radius = PLUME_GEOMETRY.wall_radius_m(np.array([1e-3, 7e-3, 8e-3, 10e-3]))
    assert radius[0] == 2e-3 and radius[1] == pytest.approx(2.5e-3) and radius[2] == 6e-3 and radius[3] == 6e-3
    # channel-only identity is unchanged by the optional keys
    assert set(CHANNEL_GEOMETRY.to_dict()) == {"bore_radius_m", "z_min_m", "z_max_m", "cone_start_z_m", "exit_radius_m"}
    assert PLUME_GEOMETRY.to_dict()["body_dielectric_radius_m"] == 4.0e-3


def test_plume_grid_requires_exit_plane_lip_and_face_split_on_grid_lines():
    Grid2D(PLUME_GEOMETRY, 24, 48)
    with pytest.raises(PIC2DValidationError):
        Grid2D(PLUME_GEOMETRY, 24, 50)  # exit plane off a z line
    Grid2D(PLUME_GEOMETRY, 30, 48)  # 0.2 mm radial cells: bore 10, lip 15, face split 20 -> all on lines
    with pytest.raises(PIC2DValidationError):
        Grid2D(ChannelGeometry(2e-3, 0.0, 8e-3, 6e-3, 3e-3, plume_radius_m=6e-3, plume_length_m=4e-3, body_dielectric_radius_m=4.1e-3), 24, 48)


def test_l_shaped_masks_are_consistent():
    grid = plume_grid()
    masks = build_mesh_masks(grid)
    nr, nz = grid.cell_shape
    j_exit, i_exit, i_body = 32, 12, 16
    assert masks.has_plume
    # plume cells: everything downstream of the exit plane; channel cells only inside the (stair-step) wall
    assert masks.plume_cell[:, j_exit:].all() and not masks.plume_cell[:, :j_exit].any()
    assert masks.plasma_cell[:8, :24].all() and not masks.plasma_cell[8:, :24].any()
    assert masks.plasma_cell[:, j_exit:].all()
    # far field: the whole outer plume boundary; body face: exit-plane row outside the lip, conductor beyond the dielectric split
    assert masks.far_field_node[:, nz].all() and masks.far_field_node[nr, j_exit:].all() and not masks.far_field_node[nr, :j_exit].any()
    assert masks.body_face_node[i_exit + 1:nr, j_exit].all() and not masks.body_face_node[:i_exit + 1, j_exit].any()
    assert masks.body_conductor_node[i_body + 1:nr, j_exit].all() and not masks.body_conductor_node[:i_body + 1, j_exit].any()
    assert not masks.body_face_node[:, j_exit + 1:].any()
    # the dielectric part of the face is a wall (surface charge) node, the conductor part is Dirichlet
    assert masks.wall_node[i_exit + 1:i_body + 1, j_exit].all()
    assert np.array_equal(masks.dirichlet_node, masks.anode_node | masks.far_field_node | masks.body_conductor_node)
    assert not (masks.dirichlet_node & masks.wall_node).any()
    # the channel walls are internal boundaries: wall nodes upstream of the exit plane, plasma both sides of the exit lip
    assert masks.wall_node[8, 1:24].all() and masks.plasma_node[12, j_exit] and masks.plasma_node[12, j_exit + 1]
    # volumes partition
    plume_volume = pi * (6e-3) ** 2 * 4e-3
    assert masks.plasma_volume_m3 == pytest.approx(masks.channel_volume_m3 + plume_volume, rel=1e-9)
    assert 0.95 * (pi * (2e-3) ** 2 * 6e-3 + pi / 3 * 2e-3 * ((2e-3) ** 2 + 2e-3 * 3e-3 + (3e-3) ** 2)) < masks.channel_volume_m3
    assert masks.unknown_count == int((masks.plasma_node & ~masks.dirichlet_node).sum())
    record = masks.to_dict()
    assert record["plasma_volume_m3"] == masks.plasma_volume_m3


def test_boundary_classification_on_the_l_shaped_domain():
    grid = plume_grid()
    masks = build_mesh_masks(grid)
    r = np.array([1.0e-3, 5.0e-3, 6.0e-3, 5.0e-3, 2.1e-3, 1.0e-3, 5.5e-3, 2.6e-3])
    z = np.array([4.0e-3, 10.0e-3, 10.0e-3, 12.0e-3, 4.0e-3, -1e-6, 7.9e-3, 4.0e-3])
    codes = kernels.classify_boundary(masks, r, z)
    assert codes[0] == kernels.BOUNDARY_INSIDE           # channel interior
    assert codes[1] == kernels.BOUNDARY_INSIDE           # plume interior (would be a wall hit without the plume)
    assert codes[2] == kernels.BOUNDARY_EXIT             # far field r = R_plume
    assert codes[3] == kernels.BOUNDARY_EXIT             # far field z = z_max
    assert codes[4] == kernels.BOUNDARY_WALL             # channel dielectric wall (internal boundary)
    assert codes[5] == kernels.BOUNDARY_ANODE
    assert codes[6] == kernels.BOUNDARY_WALL             # crossed the front face back into the thruster body
    assert codes[7] == kernels.BOUNDARY_INVALID          # two cells into the dielectric: Courant violation, fails closed


# -- Poisson on the masked L-shaped domain ----------------------------------------------------

@pytest.mark.parametrize("method", ["direct", "pcg"])
def test_poisson_manufactured_solution_on_the_l_shaped_domain(method: str):
    """A smooth target field that takes the Dirichlet values on the electrodes is recovered to round-off
    from its own discrete source (volume charge on interior nodes, surface charge on the wall nodes)."""

    grid = plume_grid()
    masks = build_mesh_masks(grid)
    geometry = grid.geometry
    rr, zz = np.meshgrid(grid.r_m, grid.z_m, indexing="ij")
    zeta = zz / geometry.domain_z_max_m
    target = UA * (1.0 - zeta) * (1.0 - 0.5 * (rr / geometry.max_radius_m) ** 2) + 40.0 * np.sin(pi * zeta) * np.cos(0.5 * pi * rr / geometry.max_radius_m)
    target[masks.anode_node] = UA                                       # the electrodes hold their potentials exactly
    target[masks.far_field_node | masks.body_conductor_node] = 0.0
    target[~masks.plasma_node] = 0.0
    source = apply_operator(masks, target)                             # exact discrete source (volume + surface)
    source[~masks.unknown_node] = 0.0
    potentials = BoundaryPotentials(UA, 0.0)
    if method == "direct":
        result = BlockTridiagonalSolver(masks).solve(source, potentials)
    else:
        result = Poisson2D(masks, PoissonConfig2D(method="pcg", relative_tolerance=1e-13)).solve(source, potentials)  # type: ignore[arg-type]
    assert result.diagnostics.converged
    error = np.abs(result.phi_v - target)[masks.plasma_node]
    assert error.max() < (1e-9 if method == "direct" else 1e-6) * UA
    # Gauss: the induced charges on the electrodes balance the source; the far field and body conductor are part of "exit"
    anode, exit_plane = induced_electrode_charge_c(masks, result.phi_v)
    assert anode + exit_plane == pytest.approx(-float(source.sum()), rel=1e-9)


def test_discrete_operator_is_second_order_on_the_l_shaped_mesh():
    """Truncation error of the masked operator on the whole L-shaped interior (channel, exit aperture,
    plume): (A phi)_n / V_n against -eps0 Laplacian(phi) for a smooth axisymmetric field, second order.
    Wall/face nodes carry the surface term and Dirichlet nodes are excluded; the exit-lip corner and
    the stair-step cone nodes are excluded too (first-order metrics there, as in the channel-only mesh)."""

    errors = []
    for cells_per_mm in (2, 4, 8):
        grid = plume_grid(cells_per_mm)
        masks = build_mesh_masks(grid)
        geometry = grid.geometry
        rr, zz = np.meshgrid(grid.r_m, grid.z_m, indexing="ij")
        big_r, length = geometry.max_radius_m, geometry.domain_z_max_m
        amplitude = 1.0e5
        radial = rr**2 - rr**4 / (2.0 * big_r**2)
        phi = amplitude * radial * np.sin(pi * zz / length) + 50.0 * np.cos(2.0 * pi * zz / length)
        laplacian = amplitude * ((4.0 - 8.0 * rr**2 / big_r**2) - (pi / length) ** 2 * radial) * np.sin(pi * zz / length) \
            - 50.0 * (2.0 * pi / length) ** 2 * np.cos(2.0 * pi * zz / length)
        with np.errstate(divide="ignore", invalid="ignore"):   # solid nodes have zero volume and are masked below
            discrete = apply_operator(masks, phi) / masks.geometric_volume_m3
        exact = -EPSILON_0_F_PER_M * laplacian
        interior = masks.unknown_node & ~masks.wall_node & ~masks.body_face_node
        interior &= ~((zz >= geometry.cone_start_z_m - 1e-12) & (zz <= geometry.z_max_m + 1e-12) & (rr >= geometry.bore_radius_m - 1e-12))
        errors.append(float(np.abs(discrete - exact)[interior].max()) / float(np.abs(exact).max()))
    orders = [np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
    assert all(order > 1.8 for order in orders), (errors, orders)


# -- neutrals: two-zone shape -----------------------------------------------------------------

def test_plume_neutral_shape_is_a_capped_cosine_effusion_cone():
    geometry = PLUME_GEOMETRY
    z_exit = geometry.z_max_m
    r_exit = geometry.exit_radius_m
    assert plume_neutral_shape(geometry, np.array([0.0, 1e-3]), np.array([1e-3, 7.9e-3])).tolist() == [1.0, 1.0]
    # on the axis: 2 A cos / (3 pi^2 rho^2), capped at 1/2
    rho = np.array([0.5e-3, 1e-3, 2e-3, 4e-3])
    axis = plume_neutral_shape(geometry, np.zeros_like(rho), z_exit + rho)
    expected = np.minimum(0.5, 2.0 * pi * r_exit**2 / (3.0 * pi**2 * rho**2))
    assert axis == pytest.approx(expected)
    assert axis[0] == 0.5 and axis[1] == 0.5 and axis[2] < 0.5 and axis[3] < axis[2]   # capped near the lip, then 1/rho^2
    # cosine law: zero along the front face (theta = 90 deg), maximal on the axis at equal distance
    face = plume_neutral_shape(geometry, np.array([3e-3, 5e-3]), np.array([z_exit, z_exit]))
    assert face.tolist() == [0.0, 0.0]
    off_axis = plume_neutral_shape(geometry, np.array([2e-3]), np.array([z_exit + 2e-3]))
    assert 0.0 < off_axis[0] < plume_neutral_shape(geometry, np.array([0.0]), np.array([z_exit + 2e-3 * sqrt(2.0)]))[0] * 2.0
    # channel-only geometries: uniform
    assert plume_neutral_shape(CHANNEL_GEOMETRY, np.array([1e-3]), np.array([5e-3])).tolist() == [1.0]
    grid = plume_grid()
    masks = build_mesh_masks(grid)
    cells = neutral_shape_cells(masks)
    assert cells.shape == grid.cell_shape and np.all(cells[masks.plasma_cell] >= 0.0) and np.all(cells[~masks.plasma_cell] == 0.0)
    assert np.all(cells[:8, :24] == 1.0) and cells[:, 32:].max() <= 0.5


def test_mcc_uses_the_local_neutral_density_shape():
    from cft_revival.pic2d.kernels import ParticleArrays
    from cft_revival.pic2d.mcc import NullCollisionMCC
    from cft_revival.pic2d.models import xenon_ion_species

    xs = XenonCrossSections.from_file()
    mcc = NullCollisionMCC(xs, MCCConfig(1.5e19), xenon_ion_species(1e5))
    rng = np.random.default_rng(1)
    n = 20000
    speed = sqrt(2.0 * 30.0 * ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG)
    electrons = lambda: ParticleArrays(np.full(n, 1e-3), np.full(n, 4e-3), np.zeros(n), np.zeros(n), np.full(n, speed))  # noqa: E731
    dt = 5e-9  # null-collision probability ~5 %: ~1000 candidates so the 1/2 ratio is resolved
    full = mcc.apply(electrons(), dt, np.random.default_rng(1))
    half = mcc.apply(electrons(), dt, np.random.default_rng(1), density_shape=np.full(n, 0.5))
    none = mcc.apply(electrons(), dt, np.random.default_rng(1), density_shape=np.zeros(n))
    total = lambda result: result.tally.elastic + result.tally.excitation + result.tally.ionization  # noqa: E731
    assert total(none) == 0
    assert total(full) > 0 and 0.35 < total(half) / total(full) < 0.65
    with pytest.raises(PIC2DValidationError):
        mcc.apply(electrons(), dt, rng, density_shape=np.full(n, 1.5))


# -- cathode emission region -----------------------------------------------------------------

def test_cathode_config_contract_and_sampling():
    with pytest.raises(PIC2DValidationError):
        CathodeConfig(4e-3, 3e-3, 9e-3, 10e-3, 2.0, 1e-3)
    with pytest.raises(PIC2DValidationError):
        CathodeConfig(3e-3, 4e-3, 9e-3, 10e-3, 2.0, 1e-3, current_rule="continuity")  # needs max_current_a
    with pytest.raises(PIC2DValidationError):
        CathodeConfig(3e-3, 4e-3, 9e-3, 10e-3, 2.0, 1e-3, max_current_a=2e-3)          # fixed rule: no ceiling
    grid = plume_grid()
    with pytest.raises(PIC2DValidationError):
        config(grid, cathode_config=cathode(), injection=InjectionConfig(1e-3, 2.0))   # not both
    with pytest.raises(PIC2DValidationError):
        config(Grid2D(CHANNEL_GEOMETRY, 12, 32), cathode_config=cathode())             # needs a plume
    with pytest.raises(PIC2DValidationError):
        config(grid, cathode_config=CathodeConfig(3.5e-3, 4.5e-3, 7.0e-3, 9.0e-3, 2.0, 1e-3))  # annulus inside the channel
    cfg = config(grid, cathode_config=cathode(2e-3))
    assert cfg.emission_peak_current_a == 2e-3 and cfg.emission_temperature_ev == 2.0
    assert cfg.initial_emission_rate_per_step == pytest.approx(2e-3 * cfg.dt_s / (ELEMENTARY_CHARGE_C * cfg.macro_weight))
    assert "cathode" in cfg.to_dict() and "injection" in cfg.to_dict()
    n = 40000
    sample = cathode_sample(cfg, np.random.default_rng(3).random((7, n)))
    assert np.all((sample.r_m >= 3.5e-3) & (sample.r_m <= 4.5e-3)) and np.all((sample.z_m >= 9e-3) & (sample.z_m <= 10e-3))
    # uniform in volume: mean r^2 is the mid-point of [r_in^2, r_out^2]
    assert np.mean(sample.r_m**2) == pytest.approx(0.5 * (3.5e-3**2 + 4.5e-3**2), rel=0.01)
    v_th2 = 2.0 * ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG  # kT/m at 2 eV
    for component in (sample.vr_m_per_s, sample.vt_m_per_s, sample.vz_m_per_s):
        assert abs(np.mean(component)) < 0.03 * sqrt(v_th2)          # isotropic (no directed flux)
        assert np.mean(component**2) == pytest.approx(v_th2, rel=0.03)


def test_fixed_cathode_emits_the_declared_current_into_the_plume():
    grid = plume_grid()
    cfg = config(grid, cathode_config=cathode(4e-3), seed_density=0.0, series_interval_steps=20, runtime_stability_check_steps=20)
    sim = Simulation(cfg, zero_field_map(grid), backend="cpu")
    sim.run(40)
    record = sim.series[-1]
    unit = ELEMENTARY_CHARGE_C * cfg.macro_weight
    expected = 4e-3 * 20 * cfg.dt_s / unit
    assert record.ledger["cumulative"]["injected_electrons"] == pytest.approx(2 * expected, abs=2.0)
    assert record.currents_a["cathode_emission_a"] == pytest.approx(4e-3, rel=0.1)
    state = sim.state
    born_in_plume = state.electrons.z_m >= grid.geometry.z_max_m
    assert born_in_plume.all() and state.electrons.count > 0
    assert sim.to_provenance()["v2_0_options"]["cathode"]["current_rule"] == "fixed"
    assert record.momentum is not None and record.plume is not None
    assert abs(record.momentum["interval_ledger_residual_kg_m_s"]) <= 1e-9 * max(abs(record.momentum["momentum_z_kg_m_s"]), 1e-30) + 1e-40


def test_continuity_cathode_tracks_the_discharge_current_with_floor_and_ceiling():
    grid = plume_grid()
    cfg = config(grid, cathode_config=cathode(1e-4, "continuity", max_current_a=5e-3, continuity_relaxation_intervals=1.0),
                 seed_density=2e15, series_interval_steps=20, runtime_stability_check_steps=20, dt_s=1e-11)
    sim = Simulation(cfg, zero_field_map(grid), backend="cpu")
    unit = cfg.dt_s / (ELEMENTARY_CHARGE_C * cfg.macro_weight)
    assert sim.backend.emission_rate_per_step == pytest.approx(1e-4 * unit)
    sim.run(200)
    # plume record: the rate the interval ran with; momentum record: the rate set for the NEXT interval from this target
    rates = [r.momentum["cathode_rate_per_step"] for r in sim.series]
    targets = [r.momentum["cathode_target_rate_per_step"] for r in sim.series]
    assert sim.series[0].plume["cathode_rate_per_step"] == pytest.approx(1e-4 * unit)
    for previous, record in zip(sim.series, sim.series[1:]):
        assert record.plume["cathode_rate_per_step"] == pytest.approx(previous.momentum["cathode_rate_per_step"])
    floor, ceiling = 1e-4 * unit, 5e-3 * unit
    assert all(floor - 1e-12 <= rate <= ceiling + 1e-12 for rate in rates)
    # with relaxation 1 the new rate IS the clamped target of the interval
    for rate, target in zip(rates, targets):
        assert rate == pytest.approx(min(max(target, floor), ceiling))
    assert any(rate > floor for rate in rates), "the seed electrons reach the 300 V anode: emission must follow I_d"
    assert sim.series[-1].momentum["cathode_emission_next_a"] == pytest.approx(rates[-1] / unit)
    # the target IS the interval discharge current in macro-particles per step
    unit_a = ELEMENTARY_CHARGE_C * cfg.macro_weight / cfg.dt_s
    for record in sim.series[1:]:
        assert record.momentum["cathode_target_rate_per_step"] * unit_a == pytest.approx(record.currents_a["discharge_a"], rel=1e-9, abs=1e-12)


# -- momentum ledger, thrust and force closure -----------------------------------------------------

def test_momentum_ledger_closes_to_round_off_with_fields_and_boundaries():
    grid = plume_grid()
    cfg = config(grid, cathode_config=cathode(2e-3), seed_density=3e15, series_interval_steps=10, runtime_stability_check_steps=10, dt_s=1e-11)
    sim = Simulation(cfg, uniform_field_map(grid, 0.01), backend="cpu")
    sim.run(60)
    assert len(sim.series) == 6
    for record in sim.series[1:]:
        momentum = record.momentum
        assert momentum is not None
        scale = max(abs(momentum["interval_dp_kg_m_s"]), abs(momentum["field_impulse_rate_n"]) * 10 * cfg.dt_s, 1e-30)
        assert abs(momentum["interval_ledger_residual_kg_m_s"]) < 1e-9 * scale
        assert set(MOMENTUM_KEYS) <= set(record.ledger["cumulative"])
        # the magnetic force does no axial work only in the sense of energy; here it can exchange momentum: recorded
        assert np.isfinite(momentum["magnetic_impulse_rate_n"])
        assert momentum["thrust_total_n"] == pytest.approx(momentum["thrust_flux_n"] + momentum["cold_gas_thrust_n"])
        assert momentum["thrust_balance_n"] == pytest.approx(-momentum["force_on_thruster_n"])
    assert "pz_exit_ions" in empty_cumulative()


def test_boundary_forces_balance_the_plasma_force_on_a_synthetic_charge_cloud():
    """Newton's third law for the discrete field: the Maxwell-stress force on all solid boundaries
    (anode, dielectric incl. its surface charge and the tangential-field pressure on the front-face
    ring and cone steps, front-face conductor, far field) balances the force on the plasma charge,
    up to the first-order discretisation error of the one-sided boundary fields."""

    residuals = []
    for cells_per_mm in (4, 8, 16):
        grid = plume_grid(cells_per_mm)
        masks = build_mesh_masks(grid)
        rr, zz = np.meshgrid(grid.r_m, grid.z_m, indexing="ij")
        # smooth positive cloud straddling the exit plane plus a negative surface charge on the channel wall
        cloud = 2e-13 * np.exp(-((rr / 1.5e-3) ** 2) - ((zz - 8.5e-3) / 2e-3) ** 2)
        interior = masks.unknown_node & ~masks.wall_node
        volume_charge = np.where(interior, cloud * masks.shape_volume_m3 / masks.shape_volume_m3.max(), 0.0)
        surface = np.where(masks.wall_node, -1e-14, 0.0)
        source = volume_charge + surface
        phi = BlockTridiagonalSolver(masks).solve(source, BoundaryPotentials(UA, 0.0)).phi_v
        _, e_z = electric_field_nodes(masks, phi)
        plasma_force = float(np.sum(volume_charge * e_z))
        forces = boundary_forces_n(masks, phi)
        total = plasma_force + forces["thruster_n"] + forces["far_field_n"]
        scale = abs(plasma_force) + abs(forces["thruster_n"]) + abs(forces["far_field_n"])
        residuals.append(abs(total) / scale)
        assert forces["thruster_n"] == pytest.approx(forces["dielectric_n"] + forces["anode_n"] + forces["body_conductor_n"])
    assert residuals[-1] < 0.05 and residuals[-1] < residuals[0], residuals


def test_thrust_from_momentum_flux_matches_the_ion_beam_on_a_synthetic_case():
    """A cold ion beam launched in the plume with no fields: the momentum-flux thrust over the interval in
    which the beam leaves equals m v_z W N / interval and the ledger's exit tally carries it."""

    from cft_revival.pic2d.kernels import ParticleArrays
    from cft_revival.pic2d.simulation import SimulationState

    grid = plume_grid()
    # the a-priori Courant gate uses the reference electron thermal speed (5 eV): 2e-11 s keeps it under 1 cell/step
    cfg = config(grid, cathode_config=None, seed_density=0.0, series_interval_steps=10, runtime_stability_check_steps=10, dt_s=2e-11,
                 macro_weight=1e4)
    sim = Simulation(cfg, zero_field_map(grid), backend="cpu")
    n = 500
    v = 2.0e4
    ions = ParticleArrays(np.linspace(0.2e-3, 2.0e-3, n), np.full(n, 11.99e-3), np.zeros(n), np.zeros(n), np.full(n, v))
    empty = ParticleArrays(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    sim.load_state(SimulationState(step=0, time_s=0.0, electrons=empty, ions=ions, surface_charge_c=np.zeros(grid.node_shape),
                                   phi_v=np.zeros(grid.node_shape), injection_carry=0.0, cumulative=empty_cumulative()))
    sim.run(60)  # 10 um to the far plane at 0.4 um per step: all ions leave inside the run
    exit_momentum = sum(r.momentum["beam_momentum_rate_ions_n"] * 10 * cfg.dt_s for r in sim.series)
    assert sim.state.ions.count == 0
    assert exit_momentum == pytest.approx(XENON_MASS_KG * v * cfg.macro_weight * n, rel=1e-9)
    # the electric force on the boundaries is zero (no charge left, no potential) and the thrust closure is trivially the beam
    record = sim.series[-1]
    assert record.momentum["beam_momentum_rate_electrons_n"] == 0.0
    assert sim.series[-1].ledger["cumulative"]["exit_ions"] == n


# -- gates -------------------------------------------------------------------------------

def test_plume_boundary_gate_fails_closed_on_charge_pile_up():
    from cft_revival.pic2d.kernels import ParticleArrays
    from cft_revival.pic2d.simulation import SimulationState

    grid = plume_grid()
    with pytest.raises(PIC2DValidationError):
        config(Grid2D(CHANNEL_GEOMETRY, 12, 32), gate=PlumeBoundaryGateConfig(0.1))
    with pytest.raises(PIC2DValidationError):
        PlumeBoundaryGateConfig(0.0)
    cfg = config(grid, seed_density=0.0, gate=PlumeBoundaryGateConfig(max_charge_fraction=0.05), series_interval_steps=5,
                 runtime_stability_check_steps=5, dt_s=1e-12, macro_weight=1e3)
    sim = Simulation(cfg, zero_field_map(grid), backend="cpu")
    n = 2000
    # ions parked in the last cell before the far plane (pile-up), a few electrons deep in the channel (the peak reference)
    ions = ParticleArrays(np.linspace(0.1e-3, 5.5e-3, n), np.full(n, 11.99e-3), np.zeros(n), np.zeros(n), np.zeros(n))
    electrons = ParticleArrays(np.full(20, 1e-3), np.full(20, 3e-3), np.zeros(20), np.zeros(20), np.zeros(20))
    piled = lambda: SimulationState(step=0, time_s=0.0, electrons=electrons, ions=ions, surface_charge_c=np.zeros(grid.node_shape),  # noqa: E731
                                    phi_v=np.zeros(grid.node_shape), injection_carry=0.0, cumulative=empty_cumulative())
    sim.load_state(piled())
    with pytest.raises(PIC2DStabilityError, match="plume-boundary gate"):
        sim.run(5)
    # without the gate the same state is recorded, not rejected
    cfg2 = config(grid, seed_density=0.0, series_interval_steps=5, runtime_stability_check_steps=5, dt_s=1e-12, macro_weight=1e3)
    sim2 = Simulation(cfg2, zero_field_map(grid), backend="cpu")
    sim2.load_state(piled())
    sim2.run(5)
    plume = sim2.series[-1].plume
    assert plume["charge_fraction_of_peak"] > 0.05 and "gate_enforced" not in plume
    assert plume["far_field_phi_max_abs_deviation_v"] == 0.0     # Dirichlet nodes hold the reference potential exactly
    # a real pile-up is RESOLVED: ~90 macro-ions per far-plane node (>= the 32 floor), and the raw statistic bounds it
    assert plume["far_field_resolved_nodes"] > 0 and plume["min_macro_particles_per_node"] == 32
    assert plume["charge_fraction_of_peak_raw"] >= plume["charge_fraction_of_peak"] > 0.05


def test_plume_boundary_gate_ignores_single_macro_particle_shot_noise_on_the_axis_corner_node():
    """Regression for plume attempt 6 (2026-09-04): ONE macro-ion approaching the axis corner node of the
    far plane (bilinear shape volume pi dr^2 dz / 6) read 0.259 of the peak electron density and stopped the run
    66 ns after the gate armed, while the interval-averaged far-field charge fraction was 0.03.  The gate
    must read only far-field nodes holding >= min_macro_particles_per_node macro-particles; the raw
    single-deposit statistic stays recorded, and the pre-fix behaviour is reproduced with a floor of 1."""

    from cft_revival.pic2d.kernels import ParticleArrays
    from cft_revival.pic2d.simulation import PLUME_GATE_MIN_MACRO_PARTICLES_PER_NODE, SimulationState

    grid = plume_grid()
    masks = build_mesh_masks(grid)
    nr, nz = grid.cell_shape
    corner_volume = masks.shape_volume_m3[0, nz]
    assert corner_volume == pytest.approx(pi * grid.dr_m**2 * grid.dz_m / 6.0, rel=1e-12)   # the smallest far-field node
    assert corner_volume == masks.shape_volume_m3[masks.far_field_node].min()
    weight = 1e3
    # ions 10 um off axis, 10 um before the far plane: bilinear weight (1 - 0.04)^2 = 0.92 each on node (0, nz);
    # the peak reference is 20 electrons deep in the channel exactly on a node (r = 4 dr, z = 12 dz)
    electrons = ParticleArrays(np.full(20, 4 * grid.dr_m), np.full(20, 12 * grid.dz_m), np.zeros(20), np.zeros(20), np.zeros(20))

    def state(n_ions: int) -> SimulationState:
        ions = ParticleArrays(np.full(n_ions, 0.01e-3), np.full(n_ions, grid.geometry.domain_z_max_m - 0.01e-3),
                              np.zeros(n_ions), np.zeros(n_ions), np.zeros(n_ions))
        return SimulationState(step=0, time_s=0.0, electrons=electrons, ions=ions, surface_charge_c=np.zeros(grid.node_shape),
                               phi_v=np.zeros(grid.node_shape), injection_carry=0.0, cumulative=empty_cumulative())

    one_particle_over_peak = (weight / corner_volume) / (20 * weight / masks.shape_volume_m3[4, 12])
    assert one_particle_over_peak > 1.0          # a single macro-ion there out-reads the whole 20-particle peak node

    def run(gate: PlumeBoundaryGateConfig, n_ions: int) -> dict:
        cfg = config(grid, seed_density=0.0, gate=gate, series_interval_steps=5, runtime_stability_check_steps=5, dt_s=1e-12,
                     macro_weight=weight)
        sim = Simulation(cfg, zero_field_map(grid), backend="cpu")
        sim.load_state(state(n_ions))
        sim.run(5)
        return sim.series[-1].plume

    # (a) default floor (32, as the protocol): ONE ion (the attempt-6 reading, 0.92 macro-particles on the corner
    # node) leaves it unresolved -> the gate reads 0 and does not fire
    plume = run(PlumeBoundaryGateConfig(max_charge_fraction=0.25), 1)
    assert plume["gate_enforced"] and plume["min_macro_particles_per_node"] == PLUME_GATE_MIN_MACRO_PARTICLES_PER_NODE == 32
    assert plume["far_field_resolved_nodes"] == 0 and plume["charge_fraction_of_peak"] == 0.0
    assert plume["far_field_net_charge_density_max_per_m3"] == 0.0
    # ... while the raw single-deposit statistic records it: one ion at the corner node, far above the threshold
    assert plume["far_field_raw_max_node"] == [0, nz]
    assert plume["far_field_raw_max_macro_particles"] == pytest.approx(0.96 * 0.96, abs=2e-3)
    assert plume["charge_fraction_of_peak_raw"] == pytest.approx(0.96 * 0.96 * one_particle_over_peak, rel=1e-3)
    assert plume["charge_fraction_of_peak_raw"] > 0.25
    # (b) the pre-fix gate had no sample-size floor: with the smallest admissible floor (1) two macro-ions on the
    # corner node (1.84 macro-particles) stop the run; with the default floor the same state is recorded, not rejected
    with pytest.raises(PIC2DStabilityError, match="plume-boundary gate"):
        run(PlumeBoundaryGateConfig(max_charge_fraction=0.25, min_macro_particles_per_node=1), 2)
    plume2 = run(PlumeBoundaryGateConfig(max_charge_fraction=0.25), 2)
    assert plume2["far_field_resolved_nodes"] == 0 and plume2["charge_fraction_of_peak"] == 0.0
    assert plume2["charge_fraction_of_peak_raw"] == pytest.approx(2 * plume["charge_fraction_of_peak_raw"], rel=1e-9)
    # (c) the floor is part of the configuration identity and validated
    assert PlumeBoundaryGateConfig(0.25, 2.4e-6).to_dict() == {"max_charge_fraction": 0.25, "enforce_after_s": 2.4e-6, "min_macro_particles_per_node": 32}
    for bad in (0, -1, 1.5, True):
        with pytest.raises(PIC2DValidationError):
            PlumeBoundaryGateConfig(0.25, 0.0, bad)  # type: ignore[arg-type]


def test_plume_record_reports_exit_plane_potential_and_acceleration_region():
    grid = plume_grid()
    cfg = config(grid, cathode_config=cathode(1e-3), seed_density=1e15, series_interval_steps=10, runtime_stability_check_steps=10)
    sim = Simulation(cfg, zero_field_map(grid), backend="cpu")
    sim.run(10)
    plume = sim.series[-1].plume
    assert plume is not None
    assert 0.0 <= plume["exit_plane_axis_potential_v"] <= UA
    assert plume["axis_phi_max_v"] == pytest.approx(UA, abs=1e-6) and plume["axis_phi_max_z_m"] == 0.0
    assert 0.0 < plume["acceleration_z90_m"] < plume["acceleration_z10_m"] <= grid.geometry.domain_z_max_m
    assert plume["acceleration_width_m"] == pytest.approx(plume["acceleration_z10_m"] - plume["acceleration_z90_m"])
    assert plume["body_conductor_induced_charge_c"] != 0.0
    provenance = sim.to_provenance()["v2_0_options"]
    assert provenance["plume_radius_m"] == 6e-3 and provenance["plume_boundary_gate"] is None
    assert provenance["cathode"]["current_rule"] == "fixed" and not provenance["legacy_exit_plane_injection"]


def test_channel_only_seed_leaves_the_plume_empty_and_keeps_the_default_identity():
    from cft_revival.pic2d.simulation import seed_plasma_state

    grid = plume_grid()
    masks = build_mesh_masks(grid)
    channel = config(grid, cathode_config=cathode(1e-3), seed_density=0.0)
    channel = PIC2DConfig(**{**{k: getattr(channel, k) for k in channel.__dataclass_fields__}, "seed_plasma": SeedPlasmaConfig(2e15, 5.0, region="channel")})  # type: ignore[attr-defined]
    state = seed_plasma_state(channel, masks)
    assert state.electrons.count == round(2e15 * masks.channel_volume_m3 / channel.macro_weight) > 0
    assert np.all(state.electrons.z_m < grid.geometry.channel_length_m) and np.all(state.electrons.r_m < grid.geometry.exit_radius_m)
    everywhere = seed_plasma_state(config(grid, cathode_config=cathode(1e-3), seed_density=2e15), masks)
    assert np.any(everywhere.electrons.z_m > grid.geometry.channel_length_m)
    assert "region" not in SeedPlasmaConfig(2e15, 5.0).to_dict() and SeedPlasmaConfig(2e15, 5.0, region="channel").to_dict()["region"] == "channel"
    with pytest.raises(PIC2DValidationError):
        SeedPlasmaConfig(2e15, 5.0, region="plume")  # type: ignore[arg-type]


def test_two_zone_inventory_uses_the_channel_volume_and_ignores_front_face_ions():
    grid = plume_grid()
    inventory = NeutralInventoryConfig(2e16, 1e-7, wall_recycling=True, wall_temperature_k=400.0)
    cfg = config(grid, cathode_config=cathode(1e-3), mcc=True, seed_density=2e15, inventory=inventory, series_interval_steps=10,
                 runtime_stability_check_steps=10, reference_density_per_m3=1e14)
    sim = Simulation(cfg, zero_field_map(grid), backend="cpu", cross_sections=XenonCrossSections.from_file())
    assert sim.neutrals is not None
    assert sim.neutrals.volume_m3 == pytest.approx(sim.masks.channel_volume_m3)
    sim.run(30)
    record = sim.series[-1]
    assert "plume_ionization_rate_per_s" in record.currents_a and "body_face_ion_a" in record.currents_a
    assert record.neutral is not None and abs(record.neutral["interval_ledger_residual_atoms"]) < 1e-6 * cfg.mcc.neutral_density_per_m3 * sim.neutrals.volume_m3
    assert record.neutral["ionization_rate_per_s"] <= record.currents_a["ionization_rate_per_s"] + 1e-9
