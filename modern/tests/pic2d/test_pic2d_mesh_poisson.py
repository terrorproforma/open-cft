"""Mesh classification and cylindrical Poisson verification."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    EPSILON_0_F_PER_M,
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DConvergenceError,
    PIC2DValidationError,
    PoissonConfig2D,
)
from cft_revival.pic2d.poisson import (
    BlockTridiagonalSolver,
    Poisson2D,
    apply_operator,
    dense_reference_solve,
    electric_field_nodes,
    field_energy_j,
    induced_electrode_charge_c,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)
UA = 300.0
AMPLITUDE = 1.0e5


def manufactured(grid: Grid2D) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """phi = A (r^2 - r^4/(2R^2)) sin(pi zeta) + Ua (1 - zeta): Neumann at r=R, Dirichlet at both z ends."""

    geometry = grid.geometry
    radius = geometry.bore_radius_m
    length = geometry.length_m
    rr, zz = np.meshgrid(grid.r_m, grid.z_m, indexing="ij")
    zeta = (zz - geometry.z_min_m) / length
    radial = rr**2 - rr**4 / (2.0 * radius**2)
    phi = AMPLITUDE * radial * np.sin(pi * zeta) + UA * (1.0 - zeta)
    laplacian = AMPLITUDE * ((4.0 - 8.0 * rr**2 / radius**2) - (pi / length) ** 2 * radial) * np.sin(pi * zeta)
    rho = -EPSILON_0_F_PER_M * laplacian
    e_r = -AMPLITUDE * (2.0 * rr - 2.0 * rr**3 / radius**2) * np.sin(pi * zeta)
    e_z = -(AMPLITUDE * radial * np.cos(pi * zeta) * pi / length - UA / length)
    return phi, rho, e_r, e_z


def test_masks_partition_volume_and_classify_the_cone():
    grid = Grid2D(CFT_GEOMETRY, 30, 240)
    masks = build_mesh_masks(grid)
    r = grid.r_m
    exact_volume = pi * (2.0e-3) ** 2 * 18.0e-3 + pi / 3.0 * 6.0e-3 * ((2.0e-3) ** 2 + 2.0e-3 * 3.0e-3 + (3.0e-3) ** 2)
    # stair-step cone lies inside the true cone: volume slightly below the exact frustum
    assert 0.97 * exact_volume < masks.plasma_volume_m3 <= exact_volume
    assert masks.plasma_cell[:20, :180].all() and not masks.plasma_cell[20:, :180].any()
    assert masks.top_plasma_cell[0] == 19 and masks.top_plasma_cell[-1] == 28
    assert np.all(np.diff(masks.top_plasma_cell) >= 0)
    assert masks.axis_node[0].all() and not masks.wall_node[0].any()
    assert masks.wall_node[20, 1:180].all()
    assert masks.dirichlet_node[:, 0].sum() == 21 and masks.dirichlet_node[:, -1].sum() == 30
    assert masks.charge_to_source[0, 5] == pytest.approx(0.75)
    assert masks.charge_to_source[5, 5] == pytest.approx(1.0)
    assert np.all(masks.shape_volume_m3[masks.plasma_node] > 0.0)
    assert r[20] == pytest.approx(2.0e-3)


def test_grid_requires_bore_on_a_radial_grid_line():
    with pytest.raises(PIC2DValidationError):
        Grid2D(CFT_GEOMETRY, 10, 24)


@pytest.mark.parametrize("method", ["direct", "pcg"])
def test_poisson_second_order_convergence_and_axis_regularity(method: str):
    errors = []
    axis_errors = []
    for nr in (8, 16, 32):
        grid = Grid2D(STRAIGHT_GEOMETRY, nr, 4 * nr)
        masks = build_mesh_masks(grid)
        phi_exact, rho, _, _ = manufactured(grid)
        source = rho * masks.geometric_volume_m3
        source[~masks.plasma_node] = 0.0
        result = Poisson2D(masks, PoissonConfig2D(method=method, relative_tolerance=1e-12)).solve(  # type: ignore[arg-type]
            source, BoundaryPotentials(UA, 0.0)
        )
        assert result.diagnostics.converged
        error = np.abs(result.phi_v - phi_exact)[masks.plasma_node]
        errors.append(float(error.max()))
        axis_errors.append(float(np.abs(result.phi_v[0] - phi_exact[0]).max()))
        assert np.isfinite(result.phi_v[0]).all()
    orders = [np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
    assert all(order > 1.9 for order in orders), orders
    assert axis_errors[-1] < axis_errors[0] / 10.0


def test_direct_pcg_and_dense_solvers_agree():
    grid = Grid2D(CFT_GEOMETRY, 9, 54)
    masks = build_mesh_masks(grid)
    generator = np.random.default_rng(5)
    source = np.zeros(grid.node_shape)
    source[masks.unknown_node] = 1.0e-15 * generator.standard_normal(masks.unknown_count)
    potentials = BoundaryPotentials(UA, 0.0)
    direct = BlockTridiagonalSolver(masks).solve(source, potentials).phi_v
    pcg = Poisson2D(masks, PoissonConfig2D(method="pcg", relative_tolerance=1e-12)).solve(source, potentials).phi_v
    dense = dense_reference_solve(masks, source, potentials)
    assert np.max(np.abs(direct - dense)) < 1e-9
    assert np.max(np.abs(pcg - dense)) < 1e-7


def test_discrete_gauss_law_with_volume_and_surface_charge():
    grid = Grid2D(CFT_GEOMETRY, 12, 72)
    masks = build_mesh_masks(grid)
    generator = np.random.default_rng(11)
    volume = np.zeros(grid.node_shape)
    volume[masks.unknown_node] = 2.0e-15 * generator.random(masks.unknown_count)
    surface = np.zeros(grid.node_shape)
    surface[masks.wall_node] = -3.0e-15
    source = volume + surface
    phi = Poisson2D(masks).solve(source, BoundaryPotentials(UA, 0.0)).phi_v
    anode, exit_plane = induced_electrode_charge_c(masks, phi)
    total = float(source.sum())
    assert anode + exit_plane == pytest.approx(-total, rel=1e-9)
    # A phi reproduces the source on every unknown node
    assert np.allclose(apply_operator(masks, phi)[masks.unknown_node], source[masks.unknown_node], rtol=1e-9, atol=1e-30)
    assert field_energy_j(masks, phi) > 0.0


def test_electric_field_reconstruction_converges():
    errors = []
    for nr in (8, 16, 32):
        grid = Grid2D(STRAIGHT_GEOMETRY, nr, 4 * nr)
        masks = build_mesh_masks(grid)
        phi_exact, _, e_r_exact, e_z_exact = manufactured(grid)
        e_r, e_z = electric_field_nodes(masks, phi_exact)
        interior = masks.plasma_node.copy()
        error = np.hypot(e_r - e_r_exact, e_z - e_z_exact)[interior]
        errors.append(float(error.max()) / float(np.abs(e_z_exact).max()))
        assert np.all(e_r[0] == 0.0)
    orders = [np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
    assert all(order > 1.8 for order in orders), (errors, orders)


def test_poisson_fails_closed_on_nonfinite_or_unconverged_input():
    grid = Grid2D(STRAIGHT_GEOMETRY, 6, 12)
    masks = build_mesh_masks(grid)
    solver = Poisson2D(masks)
    bad = np.zeros(grid.node_shape)
    bad[1, 1] = np.nan
    with pytest.raises(PIC2DValidationError):
        solver.solve(bad, BoundaryPotentials(1.0, 0.0))
    starved = Poisson2D(masks, PoissonConfig2D(method="pcg", max_iterations=1, relative_tolerance=1e-14))
    source = np.zeros(grid.node_shape)
    source[masks.unknown_node] = 1e-15
    with pytest.raises(PIC2DConvergenceError):
        starved.solve(source, BoundaryPotentials(1.0, 0.0))


def test_geometry_contract_rejections():
    with pytest.raises(PIC2DValidationError):
        ChannelGeometry(2e-3, 0.0, 24e-3, 25e-3, 3e-3)
    with pytest.raises(PIC2DValidationError):
        ChannelGeometry(2e-3, 0.0, 24e-3, 18e-3, 1e-3)
    with pytest.raises(PIC2DValidationError):
        ChannelGeometry(2e-3, 0.0, 24e-3, 24e-3, 3e-3)
