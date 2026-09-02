"""Prescribed magnetic field: P2 binding, bicubic/bilinear consistency, orbits in the P2 field."""

from __future__ import annotations

from math import cos, hypot, pi, sin
from pathlib import Path

import numpy as np
import pytest

from cft_revival.orbit_mc.fields import AnalyticField
from cft_revival.orbit_mc.integrator import integrate_orbit, launch_velocity
from cft_revival.orbit_mc.models import ElectronLaunch, OrbitConfig, Termination
from cft_revival.pic2d import kernels
from cft_revival.pic2d.fields import (
    DEFAULT_AUTHORITY_PATH,
    MagneticFieldMap,
    build_p2_psi_field,
    linear_psi_field_map,
    load_authority,
    sample_field_map,
    uniform_field_map,
)
from cft_revival.pic2d.models import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, ChannelGeometry, Grid2D, PIC2DValidationError

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = (
    REPOSITORY_ROOT / "modern" / "examples" / "fem_reference" / "artifacts" / "third-level"
    / "divergent-exit-stack" / "checkpoints" / "divergent-exit-stack.level-1.json"
)
p2_required = pytest.mark.skipif(
    not (CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 1_000_000),
    reason="qualified P2 divergent-exit checkpoint is not materialised",
)


@pytest.fixture(scope="module")
def p2_field():
    return build_p2_psi_field(REPOSITORY_ROOT, role="primary")


def test_linear_psi_map_is_divergence_free_and_bilinear_exact():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    field = linear_psi_field_map(grid, 3.0)
    generator = np.random.default_rng(0)
    r = 3.0e-3 * generator.random(500)
    z = 24.0e-3 * generator.random(500)
    assert np.allclose(kernels.gather_nodes(grid, field.b_r_t, r, z), -3.0 * r, rtol=1e-12, atol=1e-15)
    assert np.allclose(kernels.gather_nodes(grid, field.b_z_t, r, z), 6.0 * z, rtol=1e-12, atol=1e-15)
    # discrete divergence (1/r) d(r B_r)/dr + dB_z/dz vanishes to roundoff at interior nodes
    rr = grid.r_m[:, None]
    div = np.gradient(rr * field.b_r_t, grid.dr_m, axis=0)[1:-1, 1:-1] / rr[1:-1] + np.gradient(field.b_z_t, grid.dz_m, axis=1)[1:-1, 1:-1]
    assert np.max(np.abs(div)) < 1e-9


def test_field_map_contract_rejects_axis_radial_field():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    bad = np.zeros(grid.node_shape)
    bad[0, 3] = 0.1
    with pytest.raises(PIC2DValidationError):
        MagneticFieldMap(grid, bad, np.zeros(grid.node_shape), {})


@p2_required
def test_p2_authority_hashes_are_bound_and_tampering_fails(p2_field):
    field, evidence = p2_field
    authority = load_authority(DEFAULT_AUTHORITY_PATH)
    declaration = authority["maps"]["primary"]
    assert evidence["checkpoint_file_sha256"] == declaration["checkpoint_file_sha256"]
    assert evidence["design_id"] == "divergent-exit-stack"
    assert evidence["withheld_midcell_error"]["b_relative_rms"] < 0.01
    assert 0.29 < field.certified_max_b_t < 0.32
    tampered = dict(authority)
    tampered["maps"] = {"primary": dict(declaration, checkpoint_file_sha256="0" * 64)}
    with pytest.raises(PIC2DValidationError):
        build_p2_psi_field(REPOSITORY_ROOT, role="primary", authority=tampered)


@p2_required
def test_p2_node_samples_equal_bicubic_and_bilinear_error_is_second_order(p2_field):
    field, evidence = p2_field
    coarse = Grid2D(CFT_GEOMETRY, 12, 96)
    fine = Grid2D(CFT_GEOMETRY, 24, 192)
    coarse_map = sample_field_map(field, coarse, evidence)
    fine_map = sample_field_map(field, fine, evidence)
    assert coarse_map.b_r_t[0].tolist() == [0.0] * (coarse.axial_cells + 1)
    for i in (3, 8):
        for j in (10, 50, 90):
            br, bz = field.field_cylindrical(float(coarse.r_m[i]), float(coarse.z_m[j]))
            assert coarse_map.b_r_t[i, j] == br and coarse_map.b_z_t[i, j] == bz
    assert 0.2 < coarse_map.max_b_t < 0.3
    generator = np.random.default_rng(7)
    r = 1.9e-3 * generator.random(300) + 0.05e-3
    z = 22.0e-3 * generator.random(300) + 1.0e-3
    exact = np.array([field.field_cylindrical(float(a), float(b)) for a, b in zip(r, z)])
    errors = []
    for field_map, grid in ((coarse_map, coarse), (fine_map, fine)):
        br = kernels.gather_nodes(grid, field_map.b_r_t, r, z)
        bz = kernels.gather_nodes(grid, field_map.b_z_t, r, z)
        errors.append(float(np.sqrt(np.mean((br - exact[:, 0]) ** 2 + (bz - exact[:, 1]) ** 2))))
    assert errors[0] / errors[1] > 3.0  # second-order bilinear interpolation of the bicubic field
    assert errors[1] / field.max_b_t < 5e-3


def _pic_orbit_in_field_map(field_map: MagneticFieldMap, position, velocity, dt: float, steps: int) -> np.ndarray:
    grid = field_map.grid
    r = np.array([hypot(position[0], position[1])])
    theta = float(np.arctan2(position[1], position[0]))
    z = np.array([position[2]])
    c, s = cos(theta), sin(theta)
    vr = np.array([velocity[0] * c + velocity[1] * s])
    vt = np.array([-velocity[0] * s + velocity[1] * c])
    vz = np.array([velocity[2]])
    zeros = np.zeros(1)

    def fields(rr, zz):
        return kernels.gather_nodes(grid, field_map.b_r_t, rr, zz), kernels.gather_nodes(grid, field_map.b_z_t, rr, zz)

    br, bz = fields(r, z)
    vr, vt, vz = kernels.boris_push(vr, vt, vz, zeros, zeros, br, bz, -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, -0.5 * dt)
    for _ in range(steps):
        br, bz = fields(r, z)
        vr, vt, vz = kernels.boris_push(vr, vt, vz, zeros, zeros, br, bz, -ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, dt)
        r, z, vr, vt, cos_a, sin_a = kernels.advance_positions(r, z, vr, vt, vz, dt)
        theta += float(np.arctan2(sin_a[0], cos_a[0]))
    return np.array([r[0] * cos(theta), r[0] * sin(theta), z[0]])


def _orbit_mc_reference(field, launch: ElectronLaunch, dt: float, steps: int) -> np.ndarray:
    config = OrbitConfig(
        wall_radius_m=2.0e-3, wall_z_min_m=1.0e-3, wall_z_max_m=18.0e-3, domain_radius_m=2.0e-3,
        domain_z_min_m=0.5e-3, domain_z_max_m=23.5e-3, max_time_s=steps * dt, max_path_m=1.0, max_steps=steps + 10,
        max_rotation_rad=0.5, fixed_dt_s=dt,
    )
    result = integrate_orbit(launch, field, config)
    assert result.termination is Termination.TIME_TIMEOUT, result.reason
    return np.asarray(result.final_position_m)


@p2_required
def test_p2_orbit_converges_to_orbit_mc_with_grid_refinement(p2_field):
    """A collisionless electron in the PIC-sampled P2 field approaches the bicubic orbit_mc orbit as O(dx^2)."""

    field, evidence = p2_field
    launch = ElectronLaunch("p2-orbit", 3, 5.0, 1.1, (1.2e-3, 0.0, 9.25e-3), 1, 0.7, "cell-2")
    velocity = launch_velocity(launch, field)
    omega = ELEMENTARY_CHARGE_C * field.max_b_t / ELECTRON_MASS_KG
    dt = 0.02 / omega
    steps = 600
    reference = _orbit_mc_reference(field, launch, dt, steps)
    errors = []
    for nr in (12, 24, 48):
        grid = Grid2D(CFT_GEOMETRY, nr, 8 * nr)
        final = _pic_orbit_in_field_map(sample_field_map(field, grid, evidence), launch.position_m, velocity, dt, steps)
        errors.append(float(np.linalg.norm(final - reference)))
    speed = float(np.linalg.norm(velocity))
    gyro_radius = speed / omega
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] > 2.5, errors
    assert errors[2] / gyro_radius < 0.05, (errors, gyro_radius)


def test_uniform_map_orbit_matches_orbit_mc_to_1e6():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    b_z = 0.08
    field_map = uniform_field_map(grid, b_z)
    analytic = AnalyticField(lambda position: np.array([0.0, 0.0, b_z]), None, b_z)
    launch = ElectronLaunch("uniform-map", 2, 8.0, 1.0, (1.0e-3, 0.0, 12.0e-3), -1, 2.1, "test")
    velocity = launch_velocity(launch, analytic)
    omega = ELEMENTARY_CHARGE_C * b_z / ELECTRON_MASS_KG
    dt = 1.0e-3 / omega
    steps = 3000
    reference = _orbit_mc_reference(analytic, launch, dt, steps)
    final = _pic_orbit_in_field_map(field_map, launch.position_m, velocity, dt, steps)
    gyro_radius = float(np.linalg.norm(velocity)) / omega
    assert np.linalg.norm(final - reference) / gyro_radius < 1.0e-6
