"""Synthetic preflight of the P2 regular-grid sampler and the RAM budget (no real design, no solve)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cft_revival.fem_reference import Domain, ResourceBlockedError, build_body_fitted_mesh
from cft_revival.fem_reference.solver import field_at
from experiments.l1b_hemp_confirmation_v1 import p2_fields as P

# A(r, z) = a r + b r z + c z^2 + d r^2 is exactly representable by P2; c = 0 keeps A = 0 on the axis
# (a physical A_phi), so the sampler's forced axis values psi = B_r = 0 are the analytic ones.
COEFFICIENTS = (0.7, -3.0, 0.0, 4.0)


def _analytic(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, d = COEFFICIENTS
    a_phi = a * r + b * r * z + c * z * z + d * r * r
    d_dr = a + b * z + 2.0 * d * r
    d_dz = b * r + 2.0 * c * z
    psi = r * a_phi
    b_r = -d_dz
    with np.errstate(divide="ignore", invalid="ignore"):
        b_z = np.where(r > 0.0, a_phi / np.where(r > 0.0, r, 1.0) + d_dr, 2.0 * d_dr)
    return psi, b_r, b_z


@pytest.fixture(scope="module")
def synthetic_result() -> SimpleNamespace:
    domain = Domain(0.0, 0.004, -0.003, 0.009)
    mesh = build_body_fitted_mesh(domain, (), lambda _r, _z: "ambient-background", radial_divisions=5, axial_divisions=9)
    r = mesh.p2_nodes_rz_m[:, 0]
    z = mesh.p2_nodes_rz_m[:, 1]
    a, b, c, d = COEFFICIENTS
    a_phi = a * r + b * r * z + c * z * z + d * r * r
    return SimpleNamespace(mesh=mesh, a_phi_dofs_t_m=a_phi, domain=domain)


def test_sampler_reproduces_a_quadratic_field_exactly(synthetic_result: SimpleNamespace) -> None:
    r_nodes = np.linspace(0.0, 0.003, 13)
    z_nodes = np.linspace(-0.0025, 0.0085, 23)
    sampled = P.sample_regular_grid(synthetic_result, r_nodes, z_nodes)
    rr, zz = np.meshgrid(r_nodes, z_nodes, indexing="ij")
    psi, b_r, b_z = _analytic(rr, zz)
    assert np.allclose(sampled.psi_wb, psi, rtol=0.0, atol=1.0e-13)
    assert np.allclose(sampled.b_r_t, b_r, rtol=0.0, atol=1.0e-10)
    assert np.allclose(sampled.b_z_t, b_z, rtol=0.0, atol=1.0e-10)
    assert np.all(sampled.psi_wb[0] == 0.0) and np.all(sampled.b_r_t[0] == 0.0)
    assert np.all(sampled.hit_counts >= 1) and sampled.report()["max_hits_per_node"] >= 2
    assert sampled.report()["radial_samples"] == 13 and sampled.report()["axial_samples"] == 23


def test_sampler_matches_the_reference_point_evaluator(synthetic_result: SimpleNamespace) -> None:
    r_nodes = np.linspace(0.0, 0.0035, 8)
    z_nodes = np.linspace(-0.002, 0.008, 11)
    sampled = P.sample_regular_grid(synthetic_result, r_nodes, z_nodes, scale=0.75)
    for i in (1, 3, 7):
        for j in (0, 5, 10):
            psi, b_r, b_z = field_at(synthetic_result, float(r_nodes[i]), float(z_nodes[j]))
            assert sampled.psi_wb[i, j] == pytest.approx(0.75 * psi, abs=1.0e-13)
            assert sampled.b_r_t[i, j] == pytest.approx(0.75 * b_r, abs=1.0e-9)
            assert sampled.b_z_t[i, j] == pytest.approx(0.75 * b_z, abs=1.0e-9)
    assert sampled.report()["source_strength_scale_applied"] == 0.75


def test_sampler_rejects_nodes_outside_the_mesh_and_bad_grids(synthetic_result: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="outside the FEM mesh"):
        P.sample_regular_grid(synthetic_result, np.linspace(0.0, 0.003, 5), np.linspace(-0.002, 0.02, 5))
    with pytest.raises(ValueError, match="start on the axis"):
        P.sample_regular_grid(synthetic_result, np.linspace(0.001, 0.003, 5), np.linspace(-0.002, 0.008, 5))
    with pytest.raises(ValueError, match="strictly increasing"):
        P.sample_regular_grid(synthetic_result, np.asarray([0.0, 0.002, 0.001]), np.linspace(-0.002, 0.008, 5))


def test_sampled_tracing_grid_ends_exactly_on_the_wall(synthetic_result: SimpleNamespace) -> None:
    wall = 0.0031
    value = {"p2": {"sampling": {"radial_intervals": 32, "axial_inset_steps": 1.0, "refinement": 2}}}
    r_nodes, z_nodes = P.sampling_nodes(value, wall, synthetic_result.domain.to_dict(), refinement=1)
    assert r_nodes[0] == 0.0 and r_nodes[-1] == wall and len(r_nodes) == 33
    step = wall / 32
    assert z_nodes[0] == pytest.approx(synthetic_result.domain.z_min_m + step) and z_nodes[-1] == pytest.approx(synthetic_result.domain.z_max_m - step)
    assert step * 0.99 <= z_nodes[1] - z_nodes[0] <= step * 1.05
    fine_r, fine_z = P.sampling_nodes(value, wall, synthetic_result.domain.to_dict(), refinement=2)
    assert len(fine_r) == 65 and fine_z[0] == z_nodes[0] and fine_z[-1] == z_nodes[-1] and len(fine_z) >= 2 * len(z_nodes) - 1
    sampled = P.sample_regular_grid(synthetic_result, r_nodes, z_nodes)
    grid = P.sampled_tracing_grid(sampled, wall)
    assert float(grid.r_m[-1]) == wall and grid.psi_wb.shape == (33, len(z_nodes))


def test_ram_budget_and_allocation_preflight_fail_closed(synthetic_result: SimpleNamespace) -> None:
    value = {"p2": {"resources": {"ram_budget_fraction_of_free_at_start": 0.4, "maximum_p2_dofs": 600000}}}
    budget = P.ram_budget(value, free_bytes=10 * 1024**3)
    assert budget.budget_bytes == int(0.4 * 10 * 1024**3) and budget.policy().maximum_p2_dofs == 600000
    report = P.allocation_preflight(synthetic_result.mesh, budget, phase="level-0")
    assert report["passed"] is True and report["phase"] == "level-0"
    starved = P.RamBudget(1024, 0.4, 600000)
    big_mesh = SimpleNamespace(p2_nodes_rz_m=np.zeros((200000, 2)), triangles=np.zeros((100000, 3), dtype=np.int64), boundary_edges={"outer_radial": np.zeros(10, dtype=np.int64), "z_min": np.zeros(10, dtype=np.int64), "z_max": np.zeros(10, dtype=np.int64)})
    with pytest.raises(ResourceBlockedError):
        P.allocation_preflight(big_mesh, starved, phase="level-1")
    capped = P.RamBudget(10 * 1024**3, 0.4, 100000)
    with pytest.raises(ResourceBlockedError, match="exceeds"):
        P.allocation_preflight(big_mesh, capped, phase="level-1")
