"""poisson_gmg_v1: fixed-cycle geometric multigrid field solve (host reference + Warp).

CPU-only by default (numpy + the Warp CPU device); the CUDA cases run only where a CUDA device
exists (the Lambda box) and are skipped elsewhere.
"""

from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DConvergenceError,
    PIC2DValidationError,
    PoissonConfig2D,
)
from cft_revival.pic2d.poisson import BlockTridiagonalSolver, Poisson2D, apply_operator, induced_electrode_charge_c
from cft_revival.pic2d.poisson_mg import (
    MultigridPoisson2D,
    build_hierarchy,
    coarse_axis,
    level_apply,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)
# v2.0 plume geometry (bore 2 / exit 3 / plume 12 x 12 mm / dielectric flange to 4.4 mm) on a 200 um test grid
PLUME_GEOMETRY = ChannelGeometry(
    2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3, plume_radius_m=12.0e-3, plume_length_m=12.0e-3, body_dielectric_radius_m=4.4e-3
)
POTENTIALS = BoundaryPotentials(300.0, 0.0)
MG = PoissonConfig2D(method="device-mg")
EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
# the accepted v4 33 um plateau and the two plume-50 attempts whose window maps are tracked in the repository
REAL_MAPS = [
    ("channel-33um-v4-plateau", CFT_GEOMETRY, 90, 720, EXPERIMENTS / "pic2d_cft_steady_state_v4" / "results" / "maps.npz"),
    ("plume-50um-attempt7", PLUME_GEOMETRY, 240, 720,
     EXPERIMENTS / "pic2d_cft_plume_v1" / "results-attempt7-wall-budget-no-plateau" / "maps.npz"),
    ("plume-50um-attempt8", PLUME_GEOMETRY, 240, 720,
     EXPERIMENTS / "pic2d_cft_plume_v1" / "results-attempt8-grid-heating-triad-stop" / "maps.npz"),
]


def manufactured(grid: Grid2D) -> tuple[np.ndarray, np.ndarray]:
    """phi = A (r^2 - r^4/(2R^2)) sin(pi zeta) + Ua (1 - zeta) (the test_pic2d_mesh_poisson solution)."""

    geometry = grid.geometry
    radius = geometry.bore_radius_m
    length = geometry.length_m
    rr, zz = np.meshgrid(grid.r_m, grid.z_m, indexing="ij")
    zeta = (zz - geometry.z_min_m) / length
    radial = rr**2 - rr**4 / (2.0 * radius**2)
    phi = 1.0e5 * radial * np.sin(pi * zeta) + 300.0 * (1.0 - zeta)
    laplacian = 1.0e5 * ((4.0 - 8.0 * rr**2 / radius**2) - (pi / length) ** 2 * radial) * np.sin(pi * zeta)
    return phi, -EPSILON_0_F_PER_M * laplacian


def random_source(masks, seed: int, *, surface: bool = True) -> np.ndarray:
    generator = np.random.default_rng(seed)
    source = np.zeros(masks.grid.node_shape)
    source[masks.unknown_node] = 1.0e-15 * generator.standard_normal(masks.unknown_count)
    if surface:
        source[masks.wall_node] += -3.0e-16
    return source


def maps_source(masks, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Node charge (x the finite-volume ratio) from the window-averaged density maps, and the averaged phi."""

    data = np.load(path)
    charge = ELEMENTARY_CHARGE_C * (data["n_i_per_m3"] - data["n_e_per_m3"]) * masks.shape_volume_m3
    charge[~masks.plasma_node] = 0.0
    source = charge * masks.charge_to_source
    source[~masks.unknown_node] = 0.0
    return source, np.asarray(data["phi_v"], dtype=np.float64)


# ----------------------------------------------------------------------------- hierarchy

def test_coarse_axis_keeps_the_last_node_of_an_odd_axis():
    assert coarse_axis(61).tolist() == list(range(0, 61, 2))                 # 60 cells -> 30
    assert coarse_axis(46).tolist() == list(range(0, 45, 2)) + [45]           # 45 cells -> 22 + the wall node
    assert coarse_axis(3).tolist() == [0, 2] and coarse_axis(4).tolist() == [0, 2, 3]
    with pytest.raises(PIC2DValidationError):
        coarse_axis(1)


@pytest.mark.parametrize("geometry, nr, nz", [(CFT_GEOMETRY, 30, 240), (PLUME_GEOMETRY, 60, 180), (CFT_GEOMETRY, 45, 360)])
def test_hierarchy_preserves_constants_and_symmetry_on_every_level(geometry, nr, nz):
    """Operator-dependent P reproduces constants on every pure-Neumann row (the wall reflection, including
    the concave stair-step corners where a parent is a solid), the Galerkin operators are symmetric with
    positive diagonals, and A 1 = 0 on Neumann rows of every level; the coarsest dense inverse exists."""

    masks = build_mesh_masks(Grid2D(geometry, nr, nz))
    hierarchy = build_hierarchy(masks)
    assert hierarchy.depth >= 3
    assert hierarchy.levels[0].active_count == masks.unknown_count
    for level in hierarchy.levels:
        diag = level.coef[:, 4]
        assert np.all(diag[level.active] > 0.0) and not np.any(diag[~level.active] != 0.0)
        row_sum = level.coef.sum(axis=1)
        neumann = level.active & (np.abs(row_sum) <= 1e-9 * diag.max())
        assert neumann.sum() > 0.8 * level.active_count
        ones = np.where(level.active, 1.0, 0.0)
        assert np.abs(level_apply(level, ones)[neumann]).max() <= 1e-12 * diag.max()
        if level.p_idx is not None:
            p_sum = level.p_w.sum(axis=1)
            assert np.abs(p_sum[neumann] - 1.0).max() <= 1e-12, "constants must be preserved on Neumann rows"
            assert np.all(p_sum[level.active] <= 1.0 + 1e-12) and np.all(p_sum[level.active] > 0.0)
            assert not np.any(level.p_idx[~level.active] >= 0)
        # symmetric: coef[n, k] == coef[nbr(n, k), 8 - k]
        for k in range(4):
            target = level.nbr[:, k]
            ok = target >= 0
            assert np.allclose(level.coef[ok, k], level.coef[target[ok], 8 - k], rtol=0.0, atol=1e-12 * diag.max())
    coarsest = hierarchy.levels[-1]
    assert hierarchy.coarsest_active_index.size == coarsest.active_count <= 1024
    identity = hierarchy.coarsest_inverse @ hierarchy.coarsest_inverse.T  # SPD inverse
    assert np.all(np.linalg.eigvalsh(identity) > 0.0)


def test_hierarchy_levels_match_the_production_grids():
    """The production grids coarsen to a few hundred unknowns in 4-5 levels (the launch count of the device solve)."""

    expectations = {
        (CFT_GEOMETRY, 60, 480): [(61, 481), (31, 241), (16, 121), (9, 61)],
        (CFT_GEOMETRY, 90, 720): [(91, 721), (46, 361), (24, 181), (13, 91)],
        (PLUME_GEOMETRY, 240, 720): [(241, 721), (121, 361), (61, 181), (31, 91), (16, 46)],
    }
    for (geometry, nr, nz), shapes in expectations.items():
        hierarchy = build_hierarchy(build_mesh_masks(Grid2D(geometry, nr, nz)))
        assert [(level.ni, level.nj) for level in hierarchy.levels] == shapes
        assert hierarchy.coarsest_active_index.size <= 1024


# ----------------------------------------------------------------------------- host solver

def test_multigrid_second_order_convergence_and_axis_regularity():
    errors = []
    for nr in (8, 16, 32):
        grid = Grid2D(STRAIGHT_GEOMETRY, nr, 4 * nr)
        masks = build_mesh_masks(grid)
        phi_exact, rho = manufactured(grid)
        source = rho * masks.geometric_volume_m3
        source[~masks.plasma_node] = 0.0
        result = Poisson2D(masks, PoissonConfig2D(method="device-mg", relative_tolerance=1e-12, mg_cycles=16)).solve(
            source, POTENTIALS
        )
        assert result.diagnostics.converged and result.diagnostics.iterations == 16
        errors.append(float(np.abs(result.phi_v - phi_exact)[masks.plasma_node].max()))
        assert np.isfinite(result.phi_v[0]).all()
    orders = [np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
    assert all(order > 1.9 for order in orders), orders


@pytest.mark.parametrize("geometry, nr, nz", [(CFT_GEOMETRY, 30, 240), (PLUME_GEOMETRY, 60, 180)])
def test_multigrid_matches_block_thomas_and_honours_dirichlet_exactly(geometry, nr, nz):
    grid = Grid2D(geometry, nr, nz)
    masks = build_mesh_masks(grid)
    source = random_source(masks, 5)
    reference = BlockTridiagonalSolver(masks).solve(source, POTENTIALS).phi_v
    solver = MultigridPoisson2D(masks, MG)
    result = solver.solve(source, POTENTIALS)
    assert result.diagnostics.converged
    # zero start: the contract (1e-10 |rhs|) bounds the potential error at the 1e-8 V level on the 300 V scale
    assert np.abs(result.phi_v - reference).max() <= 1.0e-8
    assert np.array_equal(result.phi_v[masks.anode_node], np.full(masks.anode_node.sum(), 300.0))
    assert np.array_equal(result.phi_v[masks.exit_node], np.zeros(masks.exit_node.sum()))
    assert np.array_equal(result.phi_v[~masks.plasma_node], np.zeros((~masks.plasma_node).sum()))
    if geometry.has_plume:
        assert masks.body_conductor_node.sum() > 0 and masks.far_field_node.sum() > 0
        assert np.all(result.phi_v[masks.body_conductor_node] == 0.0) and np.all(result.phi_v[masks.far_field_node] == 0.0)
    # warm start from the previous solution (the production situation: every step starts from the previous
    # potential): one cycle already meets the contract and the full cycle count is at the 1e-9 V parity level
    phi_warm, history, rhs_norm = solver.run_cycles(source, POTENTIALS, initial_phi_v=result.phi_v, cycles=1)
    assert history[-1] <= 1e-10 * rhs_norm
    phi_warm = solver.solve(source, POTENTIALS, initial_phi_v=result.phi_v).phi_v
    assert np.abs(phi_warm - reference).max() <= 1.0e-9


def test_discrete_gauss_law_with_volume_and_surface_charge_under_multigrid():
    grid = Grid2D(CFT_GEOMETRY, 12, 72)
    masks = build_mesh_masks(grid)
    generator = np.random.default_rng(11)
    volume = np.zeros(grid.node_shape)
    volume[masks.unknown_node] = 2.0e-15 * generator.random(masks.unknown_count)
    surface = np.zeros(grid.node_shape)
    surface[masks.wall_node] = -3.0e-15
    source = volume + surface
    phi = Poisson2D(masks, MG).solve(source, POTENTIALS).phi_v
    anode, exit_plane = induced_electrode_charge_c(masks, phi)
    assert anode + exit_plane == pytest.approx(-float(source.sum()), rel=1e-9)
    unknown_source = source[masks.unknown_node]
    residual = apply_operator(masks, phi)[masks.unknown_node] - unknown_source
    assert np.abs(residual).max() <= 1e-9 * np.abs(unknown_source).max()     # scale-aware, as the direct-solve test


def test_multigrid_fails_closed_when_the_fixed_cycle_count_misses_the_contract():
    masks = build_mesh_masks(Grid2D(CFT_GEOMETRY, 30, 240))       # three levels (12 x 96 is one dense level: exact)
    assert build_hierarchy(masks).depth == 3
    source = random_source(masks, 2)
    starved = Poisson2D(masks, PoissonConfig2D(method="device-mg", mg_cycles=2))
    with pytest.raises(PIC2DConvergenceError):
        starved.solve(source, POTENTIALS)
    bad = source.copy()
    bad[1, 1] = np.nan
    with pytest.raises(PIC2DValidationError):
        Poisson2D(masks, MG).solve(bad, POTENTIALS)


@pytest.mark.parametrize("label, geometry, nr, nz, path", REAL_MAPS, ids=[case[0] for case in REAL_MAPS])
def test_fixed_cycle_count_meets_the_contract_on_the_production_masks_with_real_charge_maps(label, geometry, nr, nz, path):
    """The default 12 V(2,2) cycles reach the 1e-10 contract from a ZERO start on the real charge of the accepted
    v4 33 um plateau and of the plume-50 attempts 7/8 (window-averaged n_i - n_e), with a uniform contraction
    (no slow mode at the stair-stepped cone: the measured factor stays <= 0.2 on every cycle)."""

    if not path.is_file():
        pytest.skip(f"{path} is not available")
    masks = build_mesh_masks(Grid2D(geometry, nr, nz))
    source, phi_maps = maps_source(masks, path)
    assert np.abs(source).max() > 0.0
    solver = MultigridPoisson2D(masks, MG)
    phi, history, rhs_norm = solver.run_cycles(source, POTENTIALS)
    relative = np.asarray(history) / rhs_norm
    factors = relative[1:] / relative[:-1]
    assert relative[-1] <= 1e-10, relative.tolist()
    assert relative[-1] <= 0.25e-10, "the default cycle count must keep a margin under the contract"
    assert np.all(factors <= 0.2), factors.tolist()
    # warm start from the (pessimistic: window-averaged) recorded potential still meets the contract
    _, warm, _ = solver.run_cycles(source, POTENTIALS, initial_phi_v=phi_maps)
    assert warm[-1] / rhs_norm <= 1e-10
    assert warm[0] < history[0]


# ----------------------------------------------------------------------------- identity

def test_solver_selection_is_part_of_the_configuration_identity_and_legacy_identities_are_unchanged():
    legacy = PoissonConfig2D(method="device-direct").to_dict()
    assert "multigrid" not in legacy and set(legacy) == {"method", "relative_tolerance", "absolute_tolerance", "max_iterations", "preconditioner"}
    record = PoissonConfig2D(method="device-mg", mg_cycles=12).to_dict()
    assert record["multigrid"] == {"cycles": 12, "pre_sweeps": 2, "post_sweeps": 2, "omega": 0.8, "coarsest_max_unknowns": 1024}
    assert artifacts.content_hash(legacy) != artifacts.content_hash(record)
    assert artifacts.content_hash(record) != artifacts.content_hash(PoissonConfig2D(method="device-mg", mg_cycles=14).to_dict())
    with pytest.raises(PIC2DValidationError):
        PoissonConfig2D(method="device-mg", mg_omega=1.5)
    with pytest.raises(PIC2DValidationError):
        PoissonConfig2D(method="device-mg", mg_pre_sweeps=0, mg_post_sweeps=0)
    with pytest.raises(PIC2DValidationError):
        PoissonConfig2D(method="gmg")


# ----------------------------------------------------------------------------- Warp

warp = pytest.importorskip("warp")

from cft_revival.pic2d.fields import uniform_field_map  # noqa: E402
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections  # noqa: E402
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation, StabilityLimits  # noqa: E402
from cft_revival.pic2d.warp_backend import device_available, resolve_device  # noqa: E402
from cft_revival.pic2d.warp_poisson_mg import WarpPoissonMG  # noqa: E402

DEVICES = [device for device in ("cpu", "cuda:0") if device_available(device)]


def _device_solve(masks, config: PoissonConfig2D, device: str, source_parts, *, use_graph: bool, initial_phi=None):
    resolved = resolve_device(device)
    q_e, q_i, surface = source_parts
    to_device = lambda a: warp.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), dtype=warp.float64, device=resolved)  # noqa: E731
    d_qe, d_qi, d_surface = to_device(q_e), to_device(q_i), to_device(surface)
    phi0 = np.zeros(masks.grid.node_shape) if initial_phi is None else initial_phi
    d_phi = to_device(phi0)
    solver = WarpPoissonMG(masks, POTENTIALS, config, resolved, use_graph=use_graph)
    solver.bind(d_qe, d_qi, d_surface, d_phi)
    warp.copy(d_phi, to_device(phi0))            # the bind warm-up wrote phi: restore the requested start
    solver.solve(d_qe, d_qi, d_surface, d_phi)
    warp.synchronize_device(resolved)
    return solver, d_phi.numpy().reshape(masks.grid.node_shape).copy()


def _split_source(masks, seed: int):
    generator = np.random.default_rng(seed)
    q_e = np.zeros(masks.grid.node_shape)
    q_i = np.zeros(masks.grid.node_shape)
    surface = np.zeros(masks.grid.node_shape)
    q_e[masks.plasma_node] = -1.0e-15 * generator.random(masks.plasma_node.sum())
    q_i[masks.plasma_node] = 1.0e-15 * generator.random(masks.plasma_node.sum())
    surface[masks.wall_node] = -2.0e-16
    source = (q_e + q_i) * masks.charge_to_source + surface
    source[~masks.unknown_node] = 0.0
    return (q_e, q_i, surface), source


@pytest.mark.skipif(not DEVICES, reason="no Warp device available")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("geometry, nr, nz", [(CFT_GEOMETRY, 30, 240), (PLUME_GEOMETRY, 60, 180)])
def test_warp_multigrid_matches_the_numpy_multigrid_and_the_block_thomas_solve(device, geometry, nr, nz):
    masks = build_mesh_masks(Grid2D(geometry, nr, nz))
    parts, source = _split_source(masks, 9)
    solver, phi_device = _device_solve(masks, MG, device, parts, use_graph=(device != "cpu"))
    true_residual, tolerance = solver.verify()
    assert true_residual <= tolerance and 0.0 < solver.last_worst_ratio <= 1.0
    assert solver.launches_per_solve == 12 + 12 * ((len(solver.levels) - 1) * 6 + 1)
    assert solver.host_memory_bytes <= 1024**2 * 8 and solver.device_memory_bytes < 64 * 1024**2
    host = MultigridPoisson2D(masks, MG).solve(source, POTENTIALS).phi_v          # same cycles, zero start
    reference = BlockTridiagonalSolver(masks).solve(source, POTENTIALS).phi_v
    assert np.abs(phi_device - host).max() <= 1.0e-9                                # same algorithm: summation order only
    assert np.abs(phi_device - reference).max() <= 1.0e-8                           # zero start: contract level
    assert np.all(phi_device[masks.anode_node] == 300.0) and np.all(phi_device[masks.exit_node] == 0.0)
    assert np.all(phi_device[~masks.plasma_node] == 0.0)
    # warm start from the exact potential (the production situation) -> 1e-9 V parity with the direct solve
    _, phi_warm = _device_solve(masks, MG, device, parts, use_graph=(device != "cpu"), initial_phi=reference)
    assert np.abs(phi_warm - reference).max() <= 1.0e-9


@pytest.mark.skipif(not DEVICES, reason="no Warp device available")
@pytest.mark.parametrize("device", DEVICES)
def test_warp_multigrid_verify_fails_closed_on_too_few_cycles(device):
    masks = build_mesh_masks(Grid2D(CFT_GEOMETRY, 30, 240))
    parts, _ = _split_source(masks, 4)
    solver, _ = _device_solve(masks, PoissonConfig2D(method="device-mg", mg_cycles=2), device, parts, use_graph=False)
    with pytest.raises(PIC2DConvergenceError):
        solver.verify()


def _step_config(grid: Grid2D, poisson: PoissonConfig2D) -> PIC2DConfig:
    return PIC2DConfig(
        grid=grid, potentials=POTENTIALS, dt_s=5e-12, macro_weight=2e6, seed=7,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21), poisson=poisson,
        reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=25, ion_subcycle=4, device_sync_steps=25,
    )


@pytest.mark.skipif(not DEVICES, reason="no Warp device available")
@pytest.mark.parametrize("device", DEVICES)
def test_backend_step_with_device_mg_agrees_with_the_direct_solve_and_verifies_at_the_sync(device):
    """The production step through the backend hook (Class C one-step parity): from the SAME state (positions,
    velocities, surface charge and the previous potential as the warm start) one step under the multigrid gives
    the block-Thomas potential to 1e-9 V; the first step of a run (zero start) is at the contract level; the
    residual contract is enforced at the host sync; a 50-step run with ionisation and injection keeps the integer
    tallies of the direct run (round-off-different fields do not change a particle's fate within 50 steps here)."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    backend = "warp-cpu" if device == "cpu" else "warp-cuda"
    direct = Simulation(_step_config(grid, PoissonConfig2D(method="device-direct")), field, cross_sections=xs, backend=backend, device=device)
    multigrid = Simulation(_step_config(grid, MG), field, cross_sections=xs, backend=backend, device=device)
    assert type(multigrid.backend.device_direct).__name__ == "WarpPoissonMG"
    assert multigrid.backend.device_direct.bound_inputs is not None
    direct.run(1)
    multigrid.run(1)
    assert np.abs(direct.state.phi_v - multigrid.state.phi_v).max() <= 1.0e-8      # zero start (phi^0 = 0)
    direct.run(24)
    state = direct.state
    # one-step parity from a common state: identical deposit, warm start from the same previous potential
    direct_again = Simulation(_step_config(grid, PoissonConfig2D(method="device-direct")), field, cross_sections=xs, backend=backend, device=device)
    direct_again.load_state(state)
    multigrid.load_state(state)
    direct_again.run(1)
    multigrid.run(1)
    assert np.abs(direct_again.state.phi_v - multigrid.state.phi_v).max() <= 1.0e-9
    multigrid.run(24)
    direct_again.run(24)
    residual, tolerance = multigrid.backend.device_direct.verify()
    assert residual <= tolerance
    a, b = direct_again.state, multigrid.state
    assert a.step == b.step == 50 and a.electrons.count == b.electrons.count and a.ions.count == b.ions.count
    assert np.abs(a.phi_v - b.phi_v).max() <= 1.0e-6 * 300.0
    for key in ("anode_electrons", "exit_electrons", "wall_electrons", "anode_ions", "exit_ions", "wall_ions", "ionizations"):
        assert a.cumulative[key] == b.cumulative[key], key
    # the configuration identity records the solver: a checkpoint of one is refused by the other
    assert artifacts.config_identity(direct.config) != artifacts.config_identity(multigrid.config)


@pytest.mark.skipif("cuda:0" not in DEVICES, reason="CUDA graphs need a CUDA device")
def test_cuda_graph_step_with_device_mg_is_bitwise_identical_to_the_direct_launches():
    """The multigrid solve is a fixed kernel sequence: captured inside the step graph it replays the uncaptured
    step bitwise (positions, velocities, phi, surface charge) over 100 steps with ionisation and injection."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = uniform_field_map(grid, 0.05)
    xs = XenonCrossSections.from_file()
    config = _step_config(grid, MG)
    direct = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=False)
    graph = Simulation(config, field, cross_sections=xs, backend="warp-cuda", step_graph=True)
    assert direct.backend.step_graph is False and graph.backend.step_graph is True
    direct.run(100)
    graph.run(100)
    assert graph.backend.step_graph_active and graph.backend.graph_captures >= 2
    a, b = direct.state, graph.state
    assert a.step == b.step == 100 and a.cumulative["ionizations"] > 0
    for species in ("electrons", "ions"):
        for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s"):
            assert np.array_equal(getattr(getattr(a, species), name), getattr(getattr(b, species), name)), (species, name)
    assert np.array_equal(a.phi_v, b.phi_v) and np.array_equal(a.surface_charge_c, b.surface_charge_c)
    residual, tolerance = graph.backend.device_direct.verify()
    assert residual <= tolerance


@pytest.mark.skipif("cuda:0" not in DEVICES, reason="needs the CUDA device")
@pytest.mark.parametrize("geometry, nr, nz", [(CFT_GEOMETRY, 30, 240), (PLUME_GEOMETRY, 60, 180)])
def test_warp_cpu_and_cuda_multigrid_agree(geometry, nr, nz):
    masks = build_mesh_masks(Grid2D(geometry, nr, nz))
    parts, _ = _split_source(masks, 21)
    _, phi_cpu = _device_solve(masks, MG, "cpu", parts, use_graph=False)
    _, phi_cuda = _device_solve(masks, MG, "cuda:0", parts, use_graph=True)
    assert np.abs(phi_cpu - phi_cuda).max() <= 1.0e-9
