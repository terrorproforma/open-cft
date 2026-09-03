"""Definition-v3 machinery on manufactured axisymmetric fields with known nulls and separatrices.

psi(r, z) = (B0/2) r^2 [sin(k z) + c r^2 cos(k z)] is an exact flux function (div B = 0).
Its axis nulls sit at z_k = n pi / k and the g = 0 separatrix of each null meets the wall
cylinder r = r_w at z_c = z_k - atan(c r_w^2) / k, i.e. a curved separatrix with a known
wall intersection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cft_revival.experiment_runtime.canonical import canonical_bytes

from experiments.cusp_topology_search_v3 import topology as T

B0 = 0.2
K = math.pi / 0.006
C = 5.0e4
WALL = 0.002
POLICY = T.TopologyPolicy()


def manufactured_grid(dr: float, dz: float, *, z_min: float = -0.005, z_max: float = 0.029, c: float = C) -> T.TracingGrid:
    r = np.arange(0.0, 0.0025 + 0.5 * dr, dr)
    z = np.arange(z_min, z_max + 0.5 * dz, dz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = 0.5 * B0 * rr**2 * (np.sin(K * zz) + c * rr**2 * np.cos(K * zz))
    br = -0.5 * B0 * rr * (K * np.cos(K * zz) - c * rr**2 * K * np.sin(K * zz))
    bz = B0 * (np.sin(K * zz) + 2.0 * c * rr**2 * np.cos(K * zz))
    return T.tracing_grid(r, z, psi, br, bz, WALL)


GEOMETRY = T.ChannelGeometry(
    wall_radius_m=WALL,
    straight_z_min_m=0.001,
    straight_z_max_m=0.023,
    chamber_length_m=0.024,
    stage_pitch_m=0.006,
    stage_centres_m=(0.003, 0.009, 0.015, 0.021),
    injector_length_m=0.001,
)


def _characterize(grid: T.TracingGrid, **kwargs):
    return T.characterize_map(
        grid,
        GEOMETRY,
        POLICY,
        source_identity_sha256="a" * 64,
        minimum_certificate_tightness_ratio=1.0e-3,
        keep_paths=True,
        **kwargs,
    )


@pytest.fixture(scope="module")
def coarse():
    return _characterize(manufactured_grid(1.25e-4, 2.5e-4))


@pytest.fixture(scope="module")
def fine():
    return _characterize(manufactured_grid(6.25e-5, 1.25e-4), axis_window_m=tuple(T.axis_window(manufactured_grid(1.25e-4, 2.5e-4), GEOMETRY, POLICY)))


def test_tracing_grid_restricts_to_the_wall_and_rejects_bad_inputs() -> None:
    grid = manufactured_grid(1.25e-4, 2.5e-4)
    assert grid.r_m[0] == 0.0 and grid.r_m[-1] >= WALL and grid.r_m[-2] < WALL
    assert grid.psi_wb.shape == (len(grid.r_m), len(grid.z_m))
    with pytest.raises(ValueError, match="wall radius lies outside"):
        T.tracing_grid(grid.r_m, grid.z_m, grid.psi_wb, grid.b_r_t, grid.b_z_t, 0.01)
    with pytest.raises(ValueError, match="fewer than three radial cells"):
        T.tracing_grid(grid.r_m, grid.z_m, grid.psi_wb, grid.b_r_t, grid.b_z_t, 1.0e-4)
    shifted = grid.r_m + 1.0e-4
    with pytest.raises(ValueError, match="start on the axis"):
        T.tracing_grid(shifted, grid.z_m, grid.psi_wb, grid.b_r_t, grid.b_z_t, WALL)


def test_axis_nulls_are_found_converged_and_x_type(coarse) -> None:
    nulls = coarse["axis_nulls"]["nulls"]
    expected = [0.0, 0.006, 0.012, 0.018, 0.024]
    assert [round(n["z_m"], 6) for n in nulls] == expected
    for null, z_expected in zip(nulls, expected, strict=True):
        assert abs(null["z_m"] - z_expected) <= 2.0e-6
        assert null["converged"] and null["classification"] == "X"
        assert null["analytic_jacobian"]["classification"] == "X"
        assert null["analytic_jacobian"]["separatrix_direction_is_radial"]
        assert null["analytic_jacobian"]["divergence_identity_relative_residual"] < 1.0e-3
        assert null["v1_local_topology"]["jacobian_converged"] is True
    # z = 24 mm is the chamber length itself (<= L), hence the divergent-exit zone, not downstream.
    assert [n["zone"] for n in nulls] == ["anode_side", "channel", "channel", "channel", "divergent_exit"]
    assert coarse["axis_nulls"]["all_converged"] and coarse["axis_nulls"]["all_x_type"] and coarse["axis_nulls"]["all_classifications_agree"]
    # Nulls sit at the inter-magnet gap centres of the declared stack.
    assert all(n["distance_to_nearest_stage_gap_m"] <= 2.0e-6 for n in nulls)


def test_separatrices_reach_the_wall_at_the_analytic_intersection(coarse) -> None:
    shift = math.atan(C * WALL * WALL) / K
    assert shift > 3.0e-4  # the manufactured separatrix is genuinely curved
    traces = coarse["separatrix_traces"]
    assert len(traces) == 5 and all(trace["termination"] == "wall" for trace in traces)
    for trace, z_k in zip(traces, [0.0, 0.006, 0.012, 0.018, 0.024], strict=True):
        assert abs(trace["z_c_m"] - (z_k - shift)) <= 3.0e-5, (trace["z_c_m"], z_k - shift)
        assert trace["flux_root_consistent"] and trace["flux_root_difference_m"] <= POLICY.trace_flux_root_tolerance_m
        assert abs(trace["psi_drift_wb"]) <= 1.0e-9
        assert trace["path_rz_m"] is not None and trace["path_rz_m"][-1][0] == WALL
        assert trace["v4_bilinear_termination"] == "wall" and trace["v4_bilinear_difference_m"] <= 1.0e-4
        assert 0.0 <= trace["angle_to_wall_normal_deg"] <= 90.0
    assert coarse["all_traces_terminate_cleanly"] and coarse["all_wall_traces_flux_consistent"]


def test_cusps_and_cells_tile_the_straight_dielectric(coarse) -> None:
    topology = coarse["topology"]
    assert topology["wall_cusp_count"] == 3
    assert [c["zone"] for c in topology["outside_intersections"]] == ["anode_side", "divergent_exit"]
    assert [c["cusp_id"] for c in topology["wall_cusps"]] == ["wall-cusp-01", "wall-cusp-02", "wall-cusp-03"]
    cells = topology["cells"]
    assert [c["kind"] for c in cells] == ["anode_partial", "interior", "interior", "exit_partial"]
    assert cells[0]["z_start_m"] == GEOMETRY.straight_z_min_m and cells[-1]["z_end_m"] == GEOMETRY.straight_z_max_m
    for left, right in zip(cells[:-1], cells[1:]):
        assert left["z_end_m"] == right["z_start_m"]
    for cell in cells:
        assert cell["wall_b_min_t"] > 0.0 and cell["wall_mirror_ratio"] >= 1.0 - 1.0e-12
        assert cell["axis_bz_peak_t"] > 0.0 and cell["axis_mirror_ratio"] > 0.0
    assert topology["four_cells"] is True and topology["four_wall_cusps"] is False
    assert not topology["any_boundary_ambiguous"]


def test_refinement_stability_and_determinism(coarse, fine) -> None:
    stability = T.compare_resolutions(coarse, fine, 2.5e-4)
    assert stability["stable"] and stability["axis_null_count_equal"] and stability["wall_reaching_count_equal"]
    assert stability["max_wall_intersection_shift_m"] <= 3.0e-5 and stability["max_axis_null_shift_m"] <= 2.0e-6
    again = _characterize(manufactured_grid(1.25e-4, 2.5e-4))
    assert canonical_bytes(again) == canonical_bytes(coarse)


def test_window_override_keeps_both_resolutions_on_the_same_window() -> None:
    coarse_grid = manufactured_grid(1.25e-4, 2.5e-4)
    fine_grid = manufactured_grid(6.25e-5, 1.25e-4)
    window = T.axis_window(coarse_grid, GEOMETRY, POLICY)
    assert T.axis_window(fine_grid, GEOMETRY, POLICY) != window  # own margins differ ...
    assert T.axis_window(fine_grid, GEOMETRY, POLICY, mesh_scale_m=coarse_grid.mesh_scale_m) == window  # ... unless shared


def test_sign_change_brackets_ignore_tangencies_and_accept_node_zeros() -> None:
    samples = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert T._sign_change_brackets(samples, [1.0, 0.0, 1.0, 1.0, 1.0]) == []  # tangency
    assert T._sign_change_brackets(samples, [1.0, 0.0, -1.0, -1.0, 1.0]) == [1.0, (3.0, 4.0)]
    assert T._sign_change_brackets(samples, [-1.0, -0.5, 0.5, 1.0, 1.0]) == [(1.0, 2.0)]


def test_field_without_axis_null_has_one_unbounded_cell() -> None:
    dr, dz = 1.25e-4, 2.5e-4
    r = np.arange(0.0, 0.0025 + 0.5 * dr, dr)
    z = np.arange(-0.005, 0.029 + 0.5 * dz, dz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = 0.5 * B0 * rr**2 * (2.0 + np.sin(K * zz))  # B_z never changes sign
    br = -0.5 * B0 * rr * K * np.cos(K * zz)
    bz = B0 * (2.0 + np.sin(K * zz))
    grid = T.tracing_grid(r, z, psi, br, bz, WALL)
    record = _characterize(grid)
    assert record["axis_nulls"]["count"] == 0 and record["separatrix_traces"] == []
    topology = record["topology"]
    assert topology["wall_cusp_count"] == 0 and topology["cell_count"] == 1 and topology["cells"][0]["kind"] == "unbounded"
    assert topology["cells"][0]["wall_mirror_ratio"] is None and topology["cells"][0]["axis_mirror_ratio"] is None


def test_policy_round_trips_through_the_protocol_block() -> None:
    from experiments.cusp_topology_search_v3.experiment import protocol

    declaration = protocol()["definition_v3"]["numerical_parameters"]
    assert set(declaration) == set(T.TopologyPolicy.__dataclass_fields__)
    assert T.TopologyPolicy.from_protocol(declaration) == POLICY
