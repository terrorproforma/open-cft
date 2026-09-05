"""L2 v2: per-cell Poisson-Boltzmann field step and the kinetic ion population."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from cft_revival.hybrid.cells import synthetic_partition
from cft_revival.hybrid.ions import (
    IonPopulation,
    births_this_step,
    sample_births,
    uniform_seed_ions,
)
from cft_revival.hybrid.models import HybridValidationError
from cft_revival.hybrid.pb_solver import (
    HybridConvergenceError,
    PBConfig,
    PoissonBoltzmannSolver,
    electrode_face_areas,
    wall_effective_areas,
)
from cft_revival.pic2d.fields import uniform_field_map, zero_field_map
from cft_revival.pic2d.kernels import deposit_node_charge
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELEMENTARY_CHARGE_C,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    ParticleArrays,
    PIC2DValidationError,
    PoissonConfig2D,
    xenon_ion_species,
)
from cft_revival.pic2d.poisson import Poisson2D, apply_operator


def small_grid(nr: int = 10, nz: int = 60, cone: bool = False) -> Grid2D:
    geometry = ChannelGeometry(0.002, 0.0, 0.012, 0.009 if cone else 0.012, 0.003 if cone else 0.002)
    return Grid2D(geometry, nr if not cone else 15, nz)


def solver_for(grid: Grid2D, cusps=(0.006,), config: PBConfig | None = None) -> tuple[PoissonBoltzmannSolver, np.ndarray]:
    masks = build_mesh_masks(grid)
    part = synthetic_partition(grid.geometry.z_min_m, grid.geometry.domain_z_max_m, list(cusps))
    node_cell = part.node_cells(grid)
    field = uniform_field_map(grid, 0.1)
    _, _, effective, _ = wall_effective_areas(masks, field.b_r_t, field.b_z_t, access_floor=1.0)
    anode, exit_plane = electrode_face_areas(masks)
    solver = PoissonBoltzmannSolver(masks, node_cell, wall_effective_area_m2=effective, electrode_effective_area_m2=anode + exit_plane,
                                    config=config or PBConfig())
    return solver, node_cell


def test_wall_areas_reproduce_the_bore_surface_and_live_on_wall_nodes() -> None:
    grid = small_grid()
    masks = build_mesh_masks(grid)
    field = uniform_field_map(grid, 0.1)
    area_r, area_z, effective, access = wall_effective_areas(masks, field.b_r_t, field.b_z_t)
    # the bore surface minus the two half-faces attributed to the electrode corner nodes (Dirichlet, not wall)
    assert np.isclose(area_r.sum(), 2.0 * pi * 0.002 * (0.012 - grid.dz_m), rtol=1e-12)
    assert area_z.sum() == 0.0
    assert np.all(effective[~masks.wall_node] == 0.0)
    assert np.all(access[masks.wall_node] == 0.0)          # purely axial B: no access without a floor
    _, _, floored, _ = wall_effective_areas(masks, field.b_r_t, field.b_z_t, access_floor=0.5)
    assert np.isclose(floored.sum(), 0.5 * area_r.sum())
    anode, exit_plane = electrode_face_areas(masks)
    assert np.isclose(anode.sum(), pi * 0.002**2) and np.isclose(exit_plane.sum(), pi * 0.002**2)
    cone_grid = small_grid(cone=True)
    cone_masks = build_mesh_masks(cone_grid)
    area_r2, area_z2, _, _ = wall_effective_areas(cone_masks, np.zeros(cone_grid.node_shape), np.ones(cone_grid.node_shape))
    assert area_z2.sum() > 0.0 and area_r2.sum() > 2.0 * pi * 0.002 * 0.009


def test_negligible_electrons_recover_the_linear_poisson_solution() -> None:
    grid = small_grid()
    solver, _ = solver_for(grid, config=PBConfig(publish_relative_residual=1e-6))
    masks = solver.masks
    ion = np.where(masks.unknown_node, 2.0e-16, 0.0)          # ~1e15 m^-3 of ions
    potentials = BoundaryPotentials(100.0, 0.0)
    result = solver.solve(ion_source_c=ion, surface_charge_c=np.zeros(grid.node_shape), temperature_ev=np.array([5.0, 5.0]),
                          count=np.array([1.0, 1.0]), potentials=potentials, dt_s=0.0)
    linear = Poisson2D(masks, PoissonConfig2D(method="direct")).solve(ion, potentials).phi_v
    assert np.allclose(result.phi_v, linear, rtol=0.0, atol=1e-6 * 100.0)
    assert result.gauss_residual_c <= 1e-7 * result.gauss_source_norm_c


def test_boltzmann_cells_are_quasineutral_and_sheathed() -> None:
    grid = small_grid()
    solver, _node_cell = solver_for(grid)
    masks = solver.masks
    density = 1.0e17
    ion = np.where(masks.unknown_node, ELEMENTARY_CHARGE_C * density * masks.geometric_volume_m3, 0.0)
    counts = np.array([density * solver.cell_volume_m3[k] for k in range(2)])
    result = solver.solve(ion_source_c=ion, surface_charge_c=np.zeros(grid.node_shape), temperature_ev=np.array([5.0, 5.0]), count=counts,
                          potentials=BoundaryPotentials(100.0, 0.0), dt_s=1e-9)
    n_e = result.electron_density_per_m3
    # bulk = away from the electrode sheaths, the wall sheath and the double layer at the cell interface (column 30)
    interior = masks.unknown_node.copy()
    interior[:, :4] = False
    interior[:, -4:] = False
    interior[-3:, :] = False
    interior[:, 28:33] = False
    assert np.allclose(n_e[interior], density, rtol=0.15)     # quasi-neutral bulk
    # the interface carries a double layer: densities differ across it while each cell keeps its count
    assert not np.isclose(n_e[3, 29], n_e[3, 31], rtol=0.15)
    # the dielectric wall charges negative through the implicit electron deposit and sits below the adjacent plasma
    wall = masks.wall_node
    assert np.all(result.surface_charge_c[wall] < 0.0)
    assert np.all(result.wall_electron_flux_per_s[wall] > 0.0)
    assert result.constraint_residual_max <= 1e-7
    # every published check is re-verifiable from the outputs
    total = np.where(masks.unknown_node, ion + result.surface_charge_c - ELEMENTARY_CHARGE_C * n_e * solver.volume, 0.0)
    gauss = apply_operator(masks, result.phi_v) - total
    assert np.linalg.norm(gauss[masks.unknown_node]) <= 1e-7 * np.linalg.norm(np.abs(total[masks.unknown_node]))
    assert abs(total.sum() + result.anode_induced_charge_c + result.exit_induced_charge_c) <= 1e-7 * np.abs(total).sum()


def test_field_step_fails_closed() -> None:
    grid = small_grid()
    solver, node_cell = solver_for(grid)
    masks = solver.masks
    ion = np.where(masks.unknown_node, ELEMENTARY_CHARGE_C * 1e17 * masks.geometric_volume_m3, 0.0)
    zeros = np.zeros(grid.node_shape)
    with pytest.raises(HybridValidationError):
        solver.solve(ion_source_c=ion, surface_charge_c=zeros, temperature_ev=np.array([5.0, 5.0]), count=np.array([1e9, -1.0]),
                     potentials=BoundaryPotentials(100.0), dt_s=1e-9)
    with pytest.raises(HybridValidationError):
        solver.solve(ion_source_c=ion, surface_charge_c=zeros, temperature_ev=np.array([5.0, 0.0]), count=np.array([1e9, 1e9]),
                     potentials=BoundaryPotentials(100.0), dt_s=1e-9)
    strict, _ = solver_for(grid, config=PBConfig(max_iterations=1))
    with pytest.raises(HybridConvergenceError):
        strict.solve(ion_source_c=ion, surface_charge_c=zeros, temperature_ev=np.array([5.0, 5.0]),
                     count=np.array([1e17 * solver.cell_volume_m3[0], 1e17 * solver.cell_volume_m3[1]]), potentials=BoundaryPotentials(100.0), dt_s=1e-9)
    # a cell without any populated node is refused at construction
    populated = masks.plasma_node & (node_cell == 0)
    with pytest.raises(HybridValidationError, match="populated"):
        PoissonBoltzmannSolver(masks, node_cell, wall_effective_area_m2=np.zeros(grid.node_shape), populated_node=populated)


def test_births_carry_and_positions_follow_the_source() -> None:
    carry = 0.0
    total = 0
    for _ in range(1000):
        n, carry = births_this_step(0.37, carry)
        total += n
    assert total in (369, 370)
    with pytest.raises(HybridValidationError):
        births_this_step(-1.0, 0.0)
    grid = small_grid()
    masks = build_mesh_masks(grid)
    weights = np.zeros(grid.node_shape)
    weights[3, 30] = 1.0
    rng = np.random.default_rng(3)
    born = sample_births(masks, weights, 200, rng, mass_kg=2.18e-25, temperature_k=300.0)
    assert born.count == 200
    assert np.all(np.abs(born.z_m - 30 * grid.dz_m) <= 0.5 * grid.dz_m + 1e-12)
    assert np.all(np.abs(born.r_m - 3 * grid.dr_m) <= 0.5 * grid.dr_m + 1e-12)
    assert abs(float(np.mean(born.speed_squared())) - 3.0 * 1.380649e-23 * 300.0 / 2.18e-25) < 0.3 * 3.0 * 1.380649e-23 * 300.0 / 2.18e-25
    with pytest.raises(HybridValidationError):
        sample_births(masks, np.zeros(grid.node_shape), 5, rng, mass_kg=2.18e-25, temperature_k=300.0)


def test_ion_push_energy_wall_deposit_and_courant_guard() -> None:
    grid = small_grid()
    masks = build_mesh_masks(grid)
    species = xenon_ion_species(1000.0)
    part = synthetic_partition(0.0, 0.012, [0.006])
    # one ion on the axis in a uniform axial field gains q E dz per step (zero B)
    ions = IonPopulation(species, ParticleArrays(np.array([0.0005]), np.array([0.003]), np.zeros(1), np.zeros(1), np.zeros(1)))
    e_z = np.full(grid.node_shape, -1.0e4)      # E_z = -1e4 V/m pushes ions to +z
    zero = np.zeros(grid.node_shape)
    field = zero_field_map(grid)
    ke0 = ions.kinetic_energy_j()
    tally = ions.push(masks, e_r_nodes=zero, e_z_nodes=e_z, b_r_nodes=field.b_r_t, b_z_nodes=field.b_z_t, dt_s=1e-9, partition=part)
    assert tally.wall == 0 and ions.count == 1
    assert np.isclose(tally.field_work_j, ions.kinetic_energy_j() - ke0)
    # a radially moving ion is absorbed by the wall and its charge lands on the wall nodes
    fast = IonPopulation(species, ParticleArrays(np.array([0.00199]), np.array([0.003]), np.array([2.0e4]), np.zeros(1), np.zeros(1)))
    tally = fast.push(masks, e_r_nodes=zero, e_z_nodes=zero, b_r_nodes=field.b_r_t, b_z_nodes=field.b_z_t, dt_s=1e-9, partition=part)
    assert tally.wall == 1 and fast.count == 0
    assert np.isclose(tally.surface_deposit_c.sum(), species.charge_c * species.macro_weight)
    assert tally.wall_hits_per_cell.tolist() == [1.0, 0.0]
    # a jump of more than one cell fails closed (Courant)
    jumper = IonPopulation(species, ParticleArrays(np.array([0.0015]), np.array([0.003]), np.array([1.0e6]), np.zeros(1), np.zeros(1)))
    with pytest.raises(PIC2DValidationError):
        jumper.push(masks, e_r_nodes=zero, e_z_nodes=zero, b_r_nodes=field.b_r_t, b_z_nodes=field.b_z_t, dt_s=1e-9, partition=part)
    charge = deposit_node_charge(masks, species, ions.particles, fixed_point=False)
    assert np.isclose(charge.sum(), species.charge_c * species.macro_weight)


def test_uniform_seed_respects_an_acceptance_mask() -> None:
    grid = small_grid()
    masks = build_mesh_masks(grid)
    accept = masks.plasma_node.copy()
    accept[:, 30:] = False
    rng = np.random.default_rng(1)
    ions = uniform_seed_ions(masks, 1e16, 1e4, rng, mass_kg=2.18e-25, temperature_k=300.0, accept_node=accept, accept_volume_m3=0.5 * masks.plasma_volume_m3)
    assert ions.count == round(1e16 * 0.5 * masks.plasma_volume_m3 / 1e4)
    assert np.all(ions.z_m < 30.5 * grid.dz_m)
