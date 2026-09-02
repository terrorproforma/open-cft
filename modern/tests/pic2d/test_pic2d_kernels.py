"""Particle kernel verification: deposition, gather, Boris push, boundaries, orbits vs orbit_mc."""

from __future__ import annotations

from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from cft_revival.orbit_mc.fields import AnalyticField
from cft_revival.orbit_mc.integrator import integrate_orbit, relativistic_boris_push
from cft_revival.orbit_mc.models import ElectronLaunch, OrbitConfig, Termination, velocity_from_energy_ev
from cft_revival.pic2d import kernels
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EV_J,
    ChannelGeometry,
    Grid2D,
    ParticleArrays,
    electron_species,
    xenon_ion_species,
)

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)


def _uniform_particles(masks, count: int, generator: np.random.Generator) -> ParticleArrays:
    grid = masks.grid
    r_list, z_list = [], []
    accepted = 0
    while accepted < count:
        r = grid.geometry.max_radius_m * np.sqrt(generator.random(2 * count))
        z = grid.geometry.z_min_m + grid.geometry.length_m * generator.random(2 * count)
        keep = kernels.classify_boundary(masks, r, z) == kernels.BOUNDARY_INSIDE
        r_list.append(r[keep])
        z_list.append(z[keep])
        accepted += int(keep.sum())
    r = np.concatenate(r_list)[:count]
    z = np.concatenate(z_list)[:count]
    zeros = np.zeros(count)
    return ParticleArrays(r, z, zeros, zeros.copy(), zeros.copy())


def test_deposition_conserves_charge_and_fixed_point_matches_float():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    generator = np.random.default_rng(1)
    species = electron_species(1.0e5)
    particles = _uniform_particles(masks, 20_000, generator)
    exact = species.charge_c * species.macro_weight * particles.count
    q_float = kernels.deposit_node_charge(masks, species, particles, fixed_point=False)
    q_fixed = kernels.deposit_node_charge(masks, species, particles, fixed_point=True)
    assert q_float.sum() == pytest.approx(exact, rel=1e-13)
    assert q_fixed.sum() == pytest.approx(exact, rel=1e-11)
    assert np.max(np.abs(q_fixed - q_float)) <= particles.count * abs(species.charge_c) * species.macro_weight * 2.0**-38
    assert not q_float[~masks.plasma_node].any()
    # order independence of the fixed-point path
    permutation = generator.permutation(particles.count)
    shuffled = ParticleArrays(particles.r_m[permutation], particles.z_m[permutation], particles.vr_m_per_s, particles.vt_m_per_s, particles.vz_m_per_s)
    assert np.array_equal(kernels.deposit_node_charge(masks, species, shuffled, fixed_point=True), q_fixed)


def test_uniform_density_deposits_uniform_density_with_shape_volumes():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    generator = np.random.default_rng(2)
    count = 400_000
    particles = _uniform_particles(masks, count, generator)
    density = count / masks.plasma_volume_m3
    species = xenon_ion_species(1.0)
    q = kernels.deposit_node_charge(masks, species, particles, fixed_point=False)
    rho = q / np.where(masks.plasma_node, masks.shape_volume_m3, np.inf) / species.charge_c
    interior = masks.plasma_node & ~masks.wall_node & (np.arange(grid.node_shape[1])[None, :] > 5) & (np.arange(grid.node_shape[1])[None, :] < 90)
    ratio = rho / density
    assert abs(ratio[interior].mean() - 1.0) < 0.01
    # Axis nodes hold ~20 particles each; their mean over ~85 nodes has ~2.5%%
    # noise, which cleanly separates shape-volume normalisation (1.0) from the
    # 4/3 bias that geometric volumes would give.
    axis = interior[0]
    assert abs(ratio[0, axis].mean() - 1.0) < 0.07
    assert np.abs(ratio[interior] - 1.0).max() < 0.5


def test_deposition_and_gather_are_adjoint():
    grid = Grid2D(CFT_GEOMETRY, 9, 72)
    masks = build_mesh_masks(grid)
    generator = np.random.default_rng(3)
    particles = _uniform_particles(masks, 5_000, generator)
    species = electron_species(1.0)
    phi = generator.standard_normal(grid.node_shape)
    q = kernels.deposit_node_charge(masks, species, particles, fixed_point=False)
    gathered = kernels.gather_nodes(grid, phi, particles.r_m, particles.z_m)
    assert float((q * phi).sum()) == pytest.approx(species.charge_c * float(gathered.sum()), rel=1e-12)


def test_boris_push_matches_orbit_mc_relativistic_push_to_roundoff():
    generator = np.random.default_rng(4)
    count = 2_000
    velocity = generator.standard_normal((count, 3)) * 3.0e6
    e_field = generator.standard_normal((count, 2)) * 2.0e4
    b_field = generator.standard_normal((count, 2)) * 0.2
    dt = 3.0e-12
    vx, vy, vz = kernels.boris_push(
        velocity[:, 0], velocity[:, 1], velocity[:, 2], e_field[:, 0], e_field[:, 1], b_field[:, 0], b_field[:, 1],
        -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, dt,
    )
    for k in range(0, count, 7):
        reference = relativistic_boris_push(
            velocity[k], np.array([e_field[k, 0], 0.0, e_field[k, 1]]), np.array([b_field[k, 0], 0.0, b_field[k, 1]]), dt
        )
        scale = np.linalg.norm(reference)
        assert abs(vx[k] - reference[0]) <= 8.0 * np.finfo(float).eps * scale
        assert abs(vy[k] - reference[1]) <= 8.0 * np.finfo(float).eps * scale
        assert abs(vz[k] - reference[2]) <= 8.0 * np.finfo(float).eps * scale


def test_magnetic_only_push_preserves_speed_and_e_only_push_is_exact():
    generator = np.random.default_rng(5)
    count = 1_000
    v = generator.standard_normal((count, 3)) * 2.0e6
    zeros = np.zeros(count)
    vx, vy, vz = kernels.boris_push(v[:, 0], v[:, 1], v[:, 2], zeros, zeros, np.full(count, 0.1), np.full(count, 0.2), -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, 1e-12)
    speed_before = np.sqrt((v**2).sum(axis=1))
    speed_after = np.sqrt(vx**2 + vy**2 + vz**2)
    assert np.max(np.abs(speed_after / speed_before - 1.0)) < 1e-13
    # ions: non-relativistic to machine precision, uniform E gives v += (q/m) E dt
    ion = xenon_ion_species(1.0)
    ez = np.full(count, 1.0e4)
    ivx, ivy, ivz = kernels.boris_push(v[:, 0] * 1e-3, v[:, 1] * 1e-3, v[:, 2] * 1e-3, zeros, ez, zeros, zeros, ion.charge_c, ion.mass_kg, 1e-9)
    assert np.allclose(ivz - v[:, 2] * 1e-3, ion.charge_to_mass * 1e4 * 1e-9, rtol=1e-12)
    assert np.allclose(ivx, v[:, 0] * 1e-3) and np.allclose(ivy, v[:, 1] * 1e-3)


def test_advance_positions_handles_axis_crossing():
    r = np.array([1.0e-4, 5.0e-4])
    z = np.zeros(2)
    vr = np.array([-1.0e6, 0.0])
    vt = np.array([0.0, 1.0e6])
    vz = np.zeros(2)
    r_new, z_new, vr_new, vt_new, cos_a, sin_a = kernels.advance_positions(r, z, vr, vt, vz, 1.0e-9)
    assert r_new[0] == pytest.approx(9.0e-4)  # passed through the axis: r = |1e-4 - 1e-3|
    assert vr_new[0] == pytest.approx(1.0e6)  # radial velocity flips sign in the new frame
    assert np.all(r_new >= 0.0)
    assert r_new[1] == pytest.approx(np.hypot(5.0e-4, 1.0e-3))
    assert vr_new[1] ** 2 + vt_new[1] ** 2 == pytest.approx(1.0e12)
    assert cos_a[1] ** 2 + sin_a[1] ** 2 == pytest.approx(1.0)


def test_boundary_classification_and_wall_surface_deposit():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    masks = build_mesh_masks(grid)
    dr = grid.dr_m
    # r = 2.7 mm at z = 23.9 mm lies inside the stair-step cone (last column keeps
    # cells with outer radius <= r_wall(23.75 mm) = 2.958 mm); 2.9 mm would be a wall hit.
    r = np.array([1.0e-3, 1.0e-3, 1.0e-3, 2.0e-3 + 0.5 * dr, 2.7e-3, 2.0e-3 + 2.5 * dr, 2.0e-3 + 0.5 * dr])
    z = np.array([12.0e-3, -1.0e-9, 24.0e-3, 12.0e-3, 23.9e-3, 12.0e-3, 22.0e-3])
    codes = kernels.classify_boundary(masks, r, z)
    assert codes.tolist() == [
        kernels.BOUNDARY_INSIDE, kernels.BOUNDARY_ANODE, kernels.BOUNDARY_EXIT, kernels.BOUNDARY_WALL,
        kernels.BOUNDARY_INSIDE, kernels.BOUNDARY_INVALID, kernels.BOUNDARY_INSIDE,
    ]
    wall = codes == kernels.BOUNDARY_WALL
    charge = np.full(int(wall.sum()), -ELEMENTARY_CHARGE_C * 1e5)
    surface = kernels.wall_surface_deposit(masks, r[wall], z[wall], charge, fixed_point=False, quantum_c=ELEMENTARY_CHARGE_C * 1e5)
    assert surface.sum() == pytest.approx(charge.sum(), rel=1e-13)
    assert not surface[~masks.wall_node].any()
    fixed = kernels.wall_surface_deposit(masks, r[wall], z[wall], charge, fixed_point=True, quantum_c=ELEMENTARY_CHARGE_C * 1e5)
    assert np.allclose(fixed, surface, rtol=1e-10, atol=1e-30)


def _pic_orbit(b_z: float, e_r: float, energy_ev: float, dt: float, steps: int):
    """Single electron leapfrog in uniform B_z and radial E; returns Cartesian positions."""

    speed = velocity_from_energy_ev(energy_ev)
    r = np.array([1.0e-3])
    z = np.array([12.0e-3])
    vr = np.array([0.0])
    vt = np.array([speed * 0.8])
    vz = np.array([speed * 0.6])
    theta = 0.0
    # backward half step to leapfrog stagger
    vr, vt, vz = kernels.boris_push(vr, vt, vz, np.array([e_r]), np.array([0.0]), np.array([0.0]), np.array([b_z]), -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, -0.5 * dt)
    positions = []
    for _ in range(steps):
        vr, vt, vz = kernels.boris_push(vr, vt, vz, np.array([e_r * r[0] / r[0]]), np.array([0.0]), np.array([0.0]), np.array([b_z]), -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, dt)
        r, z, vr, vt, cos_a, sin_a = kernels.advance_positions(r, z, vr, vt, vz, dt)
        theta += float(np.arctan2(sin_a[0], cos_a[0]))
        positions.append((r[0] * cos(theta), r[0] * sin(theta), z[0]))
    return np.asarray(positions), speed


def test_uniform_field_orbit_matches_orbit_mc_within_1e6():
    """Leapfrog Boris vs orbit_mc's synchronous midpoint scheme agree to O(theta^2/8)."""

    b_z = 0.05
    omega = ELEMENTARY_CHARGE_C * b_z / ELECTRON_MASS_KG
    theta = 1.0e-3
    dt = theta / omega
    steps = 4000
    pic, speed = _pic_orbit(b_z, 0.0, 5.0, dt, steps)
    field = AnalyticField(lambda position: np.array([0.0, 0.0, b_z]), None, b_z)
    launch = ElectronLaunch("uniform", 1, 5.0, pi / 2, (1.0e-3, 0.0, 12.0e-3), 1, 0.0, "test")
    # orbit_mc launches v = v_par b + v_perp (cos(phase) e1 + sin(phase) e2); build the same initial velocity
    from cft_revival.orbit_mc.integrator import launch_velocity

    v0 = launch_velocity(launch, field)
    # Replace the PIC initial velocity by orbit_mc's so both start identically.
    r = np.array([1.0e-3]); z = np.array([12.0e-3])
    vr = np.array([v0[0]]); vt = np.array([v0[1]]); vz = np.array([v0[2]])
    vr, vt, vz = kernels.boris_push(vr, vt, vz, np.zeros(1), np.zeros(1), np.zeros(1), np.array([b_z]), -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, -0.5 * dt)
    theta_acc = 0.0
    for _ in range(steps):
        vr, vt, vz = kernels.boris_push(vr, vt, vz, np.zeros(1), np.zeros(1), np.zeros(1), np.array([b_z]), -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, dt)
        r, z, vr, vt, cos_a, sin_a = kernels.advance_positions(r, z, vr, vt, vz, dt)
        theta_acc += float(np.arctan2(sin_a[0], cos_a[0]))
    pic_final = np.array([r[0] * cos(theta_acc), r[0] * sin(theta_acc), z[0]])
    config = OrbitConfig(
        wall_radius_m=1.0, wall_z_min_m=-1.0, wall_z_max_m=1.0, domain_radius_m=1.0, domain_z_min_m=-1.0, domain_z_max_m=1.0,
        max_time_s=steps * dt, max_path_m=10.0, max_steps=steps + 10, max_rotation_rad=0.5, fixed_dt_s=dt,
    )
    result = integrate_orbit(launch, field, config)
    assert result.termination is Termination.TIME_TIMEOUT
    assert result.steps in (steps, steps + 1)  # a roundoff-sized final deadline fraction may add one step
    assert result.elapsed_time_s == pytest.approx(steps * dt, rel=1e-12)
    reference = np.asarray(result.final_position_m)
    gyro_radius = speed / omega
    assert np.linalg.norm(pic_final - reference) / gyro_radius < 1.0e-6
    # kinetic energy is exactly conserved by the magnetic rotation
    assert abs(sqrt(vr[0] ** 2 + vt[0] ** 2 + vz[0] ** 2) / speed - 1.0) < 1e-12


def test_exb_drift_velocity():
    b_z = 0.02
    e_r = 2.0e3
    omega = ELEMENTARY_CHARGE_C * b_z / ELECTRON_MASS_KG
    dt = 0.02 / omega
    steps = int(round(2.0 * pi / (omega * dt))) * 20
    # E along +r, B along +z -> E x B along -theta (azimuthal drift), magnitude E/B
    speed = velocity_from_energy_ev(1.0)
    # large radius: cylindrical curvature effects on the drift are ~ rho_L/r ~ 1e-4
    r = np.array([1.0]); z = np.array([12.0e-3])
    vr = np.array([0.0]); vt = np.array([speed]); vz = np.array([0.0])
    theta = 0.0
    for _ in range(steps):
        vr, vt, vz = kernels.boris_push(vr, vt, vz, np.array([e_r]), np.zeros(1), np.zeros(1), np.array([b_z]), -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, dt)
        r, z, vr, vt, cos_a, sin_a = kernels.advance_positions(r, z, vr, vt, vz, dt)
        theta += float(np.arctan2(sin_a[0], cos_a[0]))
    drift = r[0] * theta / (steps * dt)  # mean azimuthal speed (r nearly constant)
    # residual ~ v_perp * (Boris phase error over 20 gyrations) / v_d ~ 4e-3
    assert drift == pytest.approx(-e_r / b_z, rel=6e-3)


def test_kinetic_energy_uses_stable_relativistic_form():
    species = electron_species(1.0)
    particles = ParticleArrays(np.array([1e-3]), np.array([1e-3]), np.array([velocity_from_energy_ev(10.0)]), np.zeros(1), np.zeros(1))
    assert kernels.kinetic_energy_j(species, particles) == pytest.approx(10.0 * EV_J, rel=1e-12)
