"""Model v2.1: the axially extended plume box as a configuration.

The far plane, the side wall, the far-field gate node set, the cathode flux-tube tracer and the exit-plane
diagnostics must all follow ``geometry.plume_length_m`` (nothing is keyed on a fixed box length), the
configuration identity must change with the box, and every v2.0.x identity (configuration and field) must be
bit-identical to the pre-v2.1 code.  The v2 field extension (the domain-padding-1.5 P2 solution) is bound and
cross-checked against the qualified channel field.
"""

from __future__ import annotations

import json
from math import atan2, degrees, pi
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts, kernels
from cft_revival.pic2d.fieldlines import annulus_connectivity, channel_connected_flux_tube, trace_field_line
from cft_revival.pic2d.fields import (
    DEFAULT_PLUME_EXTENSION_PATH,
    PLUME_EXTENSION_V2_PATH,
    load_plume_extension,
    p2_plume_field_map,
    uniform_field_map,
    zero_field_map,
)
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import BoundaryPotentials, ChannelGeometry, Grid2D, PIC2DValidationError, PoissonConfig2D, StabilityLimits
from cft_revival.pic2d.simulation import (
    CathodeConfig,
    DiagnosticAccumulator,
    PIC2DConfig,
    PlumeBoundaryGateConfig,
    SeedPlasmaConfig,
    Simulation,
)
from experiments.pic2d_cft_steady_state_v1 import run as runner

MODERN = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = MODERN.parent
PLUME_V20_PROTOCOL = MODERN / "experiments" / "pic2d_cft_plume_v1" / "protocol.json"
PLUME_V21_PROTOCOL = MODERN / "experiments" / "pic2d_cft_plume_v2_1" / "protocol.json"

# v2.0.2 plume protocol (0251ff10) configuration identities: any change here is a deliberate identity change of the
# v2.0.x record family (attempts 7-8 are v2.0.1; attempt 9+ v2.0.2) and must be declared in the spec
V20_CONFIG_SHA256_CUDA = "1937f3790426696008e18eee87080630bef90a556f1ce868250a3383a5bcf1e6"
V20_CONFIG_SHA256_CPU = "4c969bff274c7f33dc1f9c65744285956fc840b12ff7a13829b7c11777b5d96c"
# v2.0 field identity (p2-field-plume-extension-v1, direct node sample) on the coarse 60 x 72 grid of the real
# geometry (dr 0.2 mm, dz 0.5 mm), computed on the pre-v2.1 code: the v1 provenance must stay bit-for-bit
V20_COARSE_FIELD_SHA256 = "d30d2d24c9d0d8d6f126f11cdb678fec056cfbb01cad9efe68d2ecae1c6479e0"

PRODUCTION = dict(bore_radius_m=0.002, z_min_m=0.0, z_max_m=0.024, cone_start_z_m=0.018, exit_radius_m=0.003)


def tiny_geometry(plume_length_m: float) -> ChannelGeometry:
    """The v2.0 test geometry (2 mm bore, 8 mm channel, 6 mm plume radius) with a variable plume length."""

    return ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 6.0e-3, 3.0e-3, plume_radius_m=6.0e-3, plume_length_m=plume_length_m,
                           body_dielectric_radius_m=4.0e-3)


def tiny_grid(plume_length_m: float, cells_per_mm: int = 4) -> Grid2D:
    return Grid2D(tiny_geometry(plume_length_m), 6 * cells_per_mm, int(round((8.0e-3 + plume_length_m) * 1e3)) * cells_per_mm)


def tiny_config(grid: Grid2D, *, seed_density: float = 1e15, cathode: CathodeConfig | None = None, **overrides) -> PIC2DConfig:
    keywords: dict = dict(dt_s=5e-12, macro_weight=2e5, seed=7, reference_density_per_m3=1e15, reference_electron_temperature_ev=5.0,
                          series_interval_steps=10, runtime_stability_check_steps=10) | overrides
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), cathode=cathode,
        seed_plasma=SeedPlasmaConfig(seed_density, 5.0) if seed_density > 0 else None,
        poisson=PoissonConfig2D(method="direct", relative_tolerance=1e-10), limits=StabilityLimits(max_cell_debye_ratio=2.0),
        plume_boundary_gate=PlumeBoundaryGateConfig(0.25, 1.0, window_steps=20, min_accumulated_macro_particles_per_node=1.0), **keywords,
    )


# -- mesh: the exit plane is found by index, whatever the box length does to the node coordinates ------------------

@pytest.mark.parametrize("plume_length_m, axial_cells", [(0.012, 720), (0.020, 880), (0.024, 960), (0.036, 1200), (0.028, 1040)])
def test_exit_plane_column_is_found_by_index_for_every_box_length(plume_length_m: float, axial_cells: int):
    geometry = ChannelGeometry(**PRODUCTION, plume_radius_m=0.012, plume_length_m=plume_length_m, body_dielectric_radius_m=0.0044)
    grid = Grid2D(geometry, 240, axial_cells)
    masks = build_mesh_masks(grid)
    j_exit = 480
    # 0.044 / 880 and 0.06 / 1200 put node 480 at 0.024 - 7e-14 dz: the v2.0 coordinate comparison lost the first plume column
    assert masks.plume_cell[:, j_exit].all() and masks.plasma_cell[:, j_exit].all()
    assert int(masks.plume_cell.sum()) == 240 * (axial_cells - j_exit)
    assert int((masks.plasma_cell & ~masks.plume_cell).sum()) == 20340       # the channel is untouched by the plume length
    assert masks.far_field_node[:, axial_cells].all() and int(masks.far_field_node.sum()) == 241 + (axial_cells - j_exit)
    assert grid.geometry.domain_z_max_m == pytest.approx(0.024 + plume_length_m)


def test_v2_0_production_masks_are_unchanged_by_the_index_rule():
    """The v2.0 box (0.036 / 720 rounds the other way) keeps its recorded counts: 77,940 plasma cells, 78,228 unknowns, 481 far nodes."""

    geometry = ChannelGeometry(**PRODUCTION, plume_radius_m=0.012, plume_length_m=0.012, body_dielectric_radius_m=0.0044)
    masks = build_mesh_masks(Grid2D(geometry, 240, 720))
    record = masks.to_dict()
    assert record["plasma_cells"] == 77940 and record["unknown_nodes"] == 78228 and record["plume"]["far_field_nodes"] == 481
    assert record["plume"]["plume_cells"] == 57600 and record["plume"]["channel_cells"] == 20340


# -- everything keyed on the far plane follows plume_length_m -------------------------------------------------------

def test_far_plane_side_wall_and_particle_classification_follow_the_plume_length():
    short, long = tiny_grid(4.0e-3), tiny_grid(8.0e-3)
    ms, ml = build_mesh_masks(short), build_mesh_masks(long)
    j_exit = 32
    # the far plane is the last node column of each box; the side wall gains one node per added cell
    assert ms.far_field_node[:, short.axial_cells].all() and ml.far_field_node[:, long.axial_cells].all()
    assert int(ml.far_field_node[ml.grid.radial_cells, :].sum()) - int(ms.far_field_node[ms.grid.radial_cells, :].sum()) == 16
    assert int(ml.far_field_node.sum()) == int(ms.far_field_node.sum()) + 16
    # the exit plane, the front face and the channel do not move
    assert np.array_equal(ms.body_face_node[:, j_exit], ml.body_face_node[:, j_exit]) and not ml.body_face_node[:, j_exit + 1:].any()
    assert np.array_equal(ms.plasma_cell[:, :j_exit], ml.plasma_cell[:, :j_exit]) and ms.channel_volume_m3 == ml.channel_volume_m3
    assert ml.plasma_volume_m3 - ms.plasma_volume_m3 == pytest.approx(pi * (6e-3) ** 2 * 4e-3, rel=1e-9)
    # a particle at z = 12 mm has LEFT the 4 mm box through its far plane and is INSIDE the 8 mm box; z = 16 mm leaves the long one
    r = np.array([1.0e-3, 1.0e-3, 6.0e-3, 5.0e-3])
    z = np.array([12.0e-3, 16.0e-3, 14.0e-3, 8.0e-3 - 1e-6])
    short_codes = kernels.classify_boundary(ms, r[:1], z[:1])
    long_codes = kernels.classify_boundary(ml, r, z)
    assert short_codes[0] == kernels.BOUNDARY_EXIT
    assert long_codes[0] == kernels.BOUNDARY_INSIDE and long_codes[1] == kernels.BOUNDARY_EXIT
    assert long_codes[2] == kernels.BOUNDARY_EXIT          # side wall at r = R_plume, z beyond the old far plane
    assert long_codes[3] == kernels.BOUNDARY_WALL          # front face still where it was


def test_far_field_histograms_bin_crossings_against_the_actual_far_plane():
    """An ion crossing the far plane at r = 1 mm is a 14 deg crossing in the 4 mm box and a 7.1 deg crossing in the 8 mm box;
    a side crossing at z beyond the old far plane lands in the long box's side profile."""

    short, long = tiny_grid(4.0e-3), tiny_grid(8.0e-3)
    for grid in (short, long):
        acc = DiagnosticAccumulator(build_mesh_masks(grid))
        z_far = grid.geometry.domain_z_max_m
        acc.record_exit(False, np.array([1.0e-3]), np.array([z_far]), np.array([100.0]))
        theta = degrees(atan2(1.0e-3, z_far - grid.geometry.z_max_m))
        expected_bin = int(theta * 90 / 90.0)
        assert acc.theta_ions.sum() == 1.0 and acc.theta_ions[expected_bin] == 1.0, (z_far, theta)
        assert acc.exit_ions[int(1.0e-3 / grid.dr_m)] == 1.0 and acc.side_ions.sum() == 0.0
    acc = DiagnosticAccumulator(build_mesh_masks(long))
    acc.record_exit(False, np.array([6.0e-3]), np.array([14.0e-3]), np.array([50.0]))     # side wall, past the short box
    assert acc.side_ions[int(14.0e-3 / long.dz_m)] == 1.0 and acc.exit_ions.sum() == 0.0
    arrays = acc.to_arrays(2e5, 5e-12)
    assert arrays["side_ion_current_density_a_per_m2"].shape == (long.axial_cells,)


def test_plume_record_and_far_field_gate_read_the_extended_far_plane():
    """Two boxes, same channel: the exit-plane axis potential is read at the same column, the acceleration end point and the
    far-field induced charge come from each box's own far plane, and the gate statistic covers every far-field node."""

    records = {}
    for length in (4.0e-3, 8.0e-3):
        grid = tiny_grid(length)
        sim = Simulation(tiny_config(grid), zero_field_map(grid), backend="cpu")
        sim.run(20)
        record = sim.series[-1]
        masks = sim.masks
        j_exit = int(round(grid.geometry.channel_length_m / grid.dz_m))
        assert record.plume is not None
        assert record.plume["exit_plane_axis_potential_v"] == pytest.approx(float(sim.state.phi_v[0, j_exit]))
        assert record.plume["acceleration_z10_m"] <= grid.geometry.domain_z_max_m + 1e-12
        assert record.plume["far_field_phi_max_abs_deviation_v"] == 0.0            # Dirichlet on the new far plane too
        window = sim._far_field_window
        assert window is not None and window.far.sum() == masks.far_field_node.sum()
        records[length] = (record, masks)
    (short, ms), (long, ml) = records[4.0e-3], records[8.0e-3]
    assert ml.far_field_node.sum() == ms.far_field_node.sum() + 16
    # the potential structure differs (the 0 V plane moved), the exit column is the same index in both
    assert short.plume["exit_plane_axis_potential_v"] != long.plume["exit_plane_axis_potential_v"]
    assert int(round(ms.grid.geometry.channel_length_m / ms.grid.dz_m)) == int(round(ml.grid.geometry.channel_length_m / ml.grid.dz_m)) == 32


def test_cathode_flux_tube_tracer_and_connectivity_gate_follow_the_far_plane():
    """On a uniform axial field the forward half-line from a cathode sample inside the aperture ends on the far plane of
    whichever box it is traced in, the backward half enters the channel (connected in both boxes), and the flux-tube
    probe planes extend to the new far plane."""

    for length in (4.0e-3, 8.0e-3):
        grid = tiny_grid(length)
        masks = build_mesh_masks(grid)
        field = uniform_field_map(grid, 0.02)
        z_far = grid.geometry.domain_z_max_m
        line = trace_field_line(field, masks, 1.0e-3, 9.0e-3, direction=1)
        assert line.termination == "far_field" and line.points[-1, 1] == pytest.approx(z_far, abs=grid.dz_m)
        result = annulus_connectivity(field, masks, 0.5e-3, 1.5e-3, 8.5e-3, 9.5e-3, n_r=3, n_z=2)
        assert result["connected_fraction"] == 1.0 and result["terminations"] == {"far_field": 6, "channel": 6}
        tube = channel_connected_flux_tube(field, masks, n_lines=4)
        probes = sorted(float(key) for key in tube["bands_by_probe_z_m"])
        assert probes[-1] > z_far - 8 * grid.dz_m - 1e-12 and probes[-1] < z_far
        assert tube["terminations"] == {"far_field": 4}
    # the runner's gate uses the same tracer: the connectivity summary carries probe planes up to the far plane in use
    grid = tiny_grid(8.0e-3)
    protocol = {"operating_point": {"cathode": {"r_inner_m": 0.5e-3, "r_outer_m": 1.5e-3, "z_start_m": 8.5e-3, "z_end_m": 9.5e-3,
                                                "require_channel_connected_fraction": 1.0}}}
    summary = runner.cathode_connectivity_check(protocol, uniform_field_map(grid, 0.02), build_mesh_masks(grid))
    assert summary is not None and summary["connected_fraction"] == 1.0
    # probe planes every 8 cells from 2 cells behind the exit: 8.5, 10.5, 12.5, 14.5 mm on the 8 mm box (past the 12 mm far
    # plane of the 4 mm box, whose probes stop at 10.5 mm)
    assert max(float(k) for k in summary["channel_flux_tube"]["bands_by_probe_z_m"]) == pytest.approx(14.5e-3)


# -- configuration identity ------------------------------------------------------------------------------------------

def test_configuration_identity_changes_with_the_plume_length_and_the_v2_0_identities_are_unchanged():
    short, long = tiny_config(tiny_grid(4.0e-3)), tiny_config(tiny_grid(8.0e-3))
    assert artifacts.config_identity(short) != artifacts.config_identity(long)
    assert short.to_dict()["grid"]["geometry"]["plume_length_m"] == 4.0e-3 and long.to_dict()["grid"]["geometry"]["plume_length_m"] == 8.0e-3
    # no new geometry keys: a channel-only and a v2.0 plume geometry serialise exactly as before
    assert set(ChannelGeometry(**PRODUCTION).to_dict()) == {"bore_radius_m", "z_min_m", "z_max_m", "cone_start_z_m", "exit_radius_m"}
    assert set(tiny_geometry(4.0e-3).to_dict()) == set(ChannelGeometry(**PRODUCTION).to_dict()) | {"plume_radius_m", "plume_length_m", "body_dielectric_radius_m"}
    protocol = runner.load_protocol(PLUME_V20_PROTOCOL)
    assert artifacts.config_identity(runner.build_config(protocol, backend="warp-cuda")) == V20_CONFIG_SHA256_CUDA
    assert artifacts.config_identity(runner.build_config(protocol, backend="cpu")) == V20_CONFIG_SHA256_CPU
    assert runner.plume_extension_path(protocol) is None            # v2.0 protocols keep the default (v1) extension


def test_v2_1_protocol_builds_the_extended_box_with_a_new_identity_and_its_own_field_extension():
    v20 = runner.load_protocol(PLUME_V20_PROTOCOL)
    v21 = runner.load_protocol(PLUME_V21_PROTOCOL)
    c20 = runner.build_config(v20, backend="warp-cuda")
    c21 = runner.build_config(v21, backend="warp-cuda")
    assert c21.grid.cell_shape == (240, 960) and c21.grid.dr_m == pytest.approx(5e-5) and c21.grid.dz_m == pytest.approx(5e-5)
    assert c21.grid.geometry.domain_z_max_m == pytest.approx(0.048) and c21.grid.geometry.plume_radius_m == 0.012
    assert c21.grid.geometry.z_max_m == c20.grid.geometry.z_max_m == 0.024          # the channel and its exit plane are unchanged
    assert artifacts.config_identity(c21) != artifacts.config_identity(c20)
    masks = build_mesh_masks(c21.grid)
    record = masks.to_dict()
    assert record["plasma_cells"] == 135540 and record["unknown_nodes"] == 135828 and record["plume"]["far_field_nodes"] == 721
    # everything else of the operating point is the v2.0.2 one (same cathode, gate semantics, neutrals, seed)
    assert c21.cathode == c20.cathode and c21.neutral_inventory == c20.neutral_inventory and c21.seed_plasma == c20.seed_plasma
    assert c21.plume_boundary_gate.window_steps == c20.plume_boundary_gate.window_steps == 400_000
    assert c21.plume_boundary_gate.min_accumulated_macro_particles_per_node == 64_000.0
    assert c21.plume_boundary_gate.enforce_after_s == pytest.approx(3.8e-6)             # one ion transit of the new box
    budget = runner.protocol_budget(v21)
    assert budget["ion_transit_time_s"] == pytest.approx(3.8e-6) and "cost_table_alternatives" in budget
    assert v21["field_plume_extension"] == "modern/spec/pic2d/p2-field-plume-extension-v2.json"
    assert runner.plume_extension_path(v21) == PLUME_EXTENSION_V2_PATH.resolve()
    assert v21["model_spec"] == "modern/spec/pic2d/pic2d-model-v2.1.json" and (MODERN / "spec" / "pic2d" / "pic2d-model-v2.1.json").is_file()
    assert "NOT_LAUNCHED" in v21["status"]
    # a protocol naming a missing extension file fails closed
    with pytest.raises(PIC2DValidationError, match="field_plume_extension"):
        runner.plume_extension_path({"field_plume_extension": "modern/spec/pic2d/does-not-exist.json"})


# -- the field source ---------------------------------------------------------------------------------------------

def test_v2_field_extension_declaration_is_bound_and_validated():
    ext = load_plume_extension(PLUME_EXTENSION_V2_PATH)
    assert ext["schema"] == "cft.pic2d.p2-field-plume-extension.v2"
    assert ext["map"]["checkpoint_path"].endswith("divergent-exit-stack.domain-padding-1.5.json")
    checkpoint = REPOSITORY_ROOT / ext["map"]["checkpoint_path"]
    assert checkpoint.is_file()
    meta = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert meta["integrity"]["payload_sha256"] == ext["map"]["checkpoint_payload_sha256"]
    assert meta["mesh_sha256"] == ext["map"]["mesh_sha256"] and meta["run_sha256"] == ext["map"]["run_sha256"]
    assert meta["domain_study"] == {"padding_factor": 1.5}
    assert ext["bounding_box"]["z_max_m"] == pytest.approx(0.06075) and ext["supported_pic_box"]["z_max_m"] == 0.060
    v1 = load_plume_extension(DEFAULT_PLUME_EXTENSION_PATH)
    assert v1["schema"] == "cft.pic2d.p2-field-plume-extension.v1" and "map" not in v1
    # a v2 file without its map is refused
    broken = dict(ext)
    broken.pop("map")
    path = Path(__file__).resolve().parent / "_broken_extension.json"
    try:
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(PIC2DValidationError, match="map"):
            load_plume_extension(path)
    finally:
        path.unlink(missing_ok=True)


def test_v2_field_extension_serves_the_48_mm_box_and_agrees_with_the_qualified_channel_field():
    """Coarse (0.2 x 0.5 mm) grids of the real geometry: the v1 extension refuses a far plane beyond 36 mm, the v2 extension
    evaluates the padded P2 solution over the 48 mm box, passes the 0.02 T channel cross-check, records its source, and
    the v2.0 box on the v1 extension keeps its pre-v2.1 field identity."""

    v20 = Grid2D(ChannelGeometry(**PRODUCTION, plume_radius_m=0.012, plume_length_m=0.012, body_dielectric_radius_m=0.0044), 60, 72)
    v21 = Grid2D(ChannelGeometry(**PRODUCTION, plume_radius_m=0.012, plume_length_m=0.024, body_dielectric_radius_m=0.0044), 60, 96)
    with pytest.raises(PIC2DValidationError, match="exceeds the declared P2 plume-extension bounding box"):
        p2_plume_field_map(REPOSITORY_ROOT, v21)
    old = p2_plume_field_map(REPOSITORY_ROOT, v20)
    assert old.sha256 == V20_COARSE_FIELD_SHA256
    assert "field_source" not in old.provenance and old.provenance["bounding_box"]["z_max_m"] == 0.036
    new = p2_plume_field_map(REPOSITORY_ROOT, v21, extension_path=PLUME_EXTENSION_V2_PATH)
    assert new.provenance["field_source"] == "plume-extension-v2" and new.provenance["plume_extension_path"] == "p2-field-plume-extension-v2.json"
    assert new.provenance["checkpoint_path"].endswith("domain-padding-1.5.json") and new.provenance["map_declaration"]["fem_level"] == 0
    assert new.provenance["plasma_nodes_sampled"] == int(build_mesh_masks(v21).plasma_node.sum())
    check = new.provenance["channel_cross_check"]
    assert check["max_abs_diff_t"] < 0.002 and check["rms_diff_t"] < 0.001            # measured 1.06 / 0.45 mT; the gate is 20 mT
    assert check["channel_field_map_sha256"] == old.provenance["channel_cross_check"]["channel_field_map_sha256"]
    # the field is live on the extended far plane (axis |B_z| ~2.6 mT at 48 mm) and the far plume is not empty of field
    j_far = v21.axial_cells
    assert 1.5e-3 < abs(new.b_z_t[0, j_far]) < 4.0e-3
    assert np.all(np.isfinite(new.b_r_t)) and new.b_r_t[0, :].max() == 0.0
    # the v2.0 box on the v2 source: same node values as the 48 mm box on the shared nodes (one solution, no seam)
    same_box_new = p2_plume_field_map(REPOSITORY_ROOT, v20, extension_path=PLUME_EXTENSION_V2_PATH, cross_check=False)
    shared = build_mesh_masks(v20).plasma_node & ~build_mesh_masks(v20).far_field_node
    assert np.array_equal(same_box_new.b_z_t[shared], new.b_z_t[:, : v20.axial_cells + 1][shared])
    assert same_box_new.sha256 != old.sha256
    # the v2 extension refuses a box outside its supported region even though the FEM mesh would cover it
    too_wide = Grid2D(ChannelGeometry(**PRODUCTION, plume_radius_m=0.0486, plume_length_m=0.024, body_dielectric_radius_m=0.0044), 243, 96)
    with pytest.raises(PIC2DValidationError, match="supported box"):
        p2_plume_field_map(REPOSITORY_ROOT, too_wide, extension_path=PLUME_EXTENSION_V2_PATH, cross_check=False)


def test_cathode_region_stays_channel_connected_on_the_padded_field_over_the_extended_box():
    """The v2.1 protocol's cathode region (r 0.5-2 mm, z 24.3-25 mm) traced on the coarse padded field of the 48 mm box:
    every sample enters the channel (the runner's fail-closed launch gate), and the flux-tube probes reach the far plane."""

    v21 = runner.load_protocol(PLUME_V21_PROTOCOL)
    grid = Grid2D(ChannelGeometry(**PRODUCTION, plume_radius_m=0.012, plume_length_m=0.024, body_dielectric_radius_m=0.0044), 60, 96)
    field = p2_plume_field_map(REPOSITORY_ROOT, grid, extension_path=PLUME_EXTENSION_V2_PATH, cross_check=False)
    summary = runner.cathode_connectivity_check(v21, field, build_mesh_masks(grid))
    assert summary is not None and summary["connected_fraction"] == 1.0 and summary["samples"] == 24
    probes = [float(k) for k in summary["channel_flux_tube"]["bands_by_probe_z_m"]]
    assert max(probes) > 0.040 and min(probes) > 0.024


def test_synthetic_extended_domain_runs_with_a_cathode_and_records_the_v2_1_plume_block():
    """A tiny run on the 8 mm box with a cathode inside the aperture tube on a uniform field: the plume record, the gate window
    and the beam tallies exist and refer to the extended far plane (no code path is keyed on the v2.0 length)."""

    grid = tiny_grid(8.0e-3)
    cathode = CathodeConfig(0.5e-3, 1.5e-3, 8.3e-3, 9.0e-3, 2.0, 1e-3, current_rule="fixed")
    sim = Simulation(tiny_config(grid, cathode=cathode, seed_density=1e15), uniform_field_map(grid, 0.02), backend="cpu")
    sim.run(40, accumulate_from_step=0)
    record = sim.series[-1]
    assert record.plume is not None and record.plume["far_field_window_steps"] >= 20 and record.plume["far_field_window_complete"]
    assert record.plume["gate_armed"] is False and record.momentum is not None
    assert 0.5e-3 < record.currents_a["cathode_emission_a"] < 2.0e-3      # fixed rule, 10-step interval: integer-macro rounding
    assert sim.masks.far_field_node.sum() == 25 + 32          # 25 far-plane nodes + 32 side nodes on the 8 mm box (4 cells/mm)
    # the final ion count is finite and the exit-plane column is where the geometry says
    assert sim.state.ions.count > 0 and int(round(grid.geometry.channel_length_m / grid.dz_m)) == 32


def test_resume_state_and_history_helper_contract():
    """Unit contract of the resume-hygiene helper (the integration test lives in test_pic2d_steady_state_runner.py)."""

    state = {"wall_seconds_total": 10.0, "sessions": [{}], "checkpoint_step": 40, "finished": True, "stop_reason": "wall_clock_budget_reached",
             "finalized_from_step": 40, "finalization_recovery": {"mode": "runner_stop_artifacts_reused"}, "ignition": {"failed": False}}
    entry = runner._demote_terminal_state(state, event="resume", step=40, utc="2026-09-04T00:00:00+00:00", summary_present=True)
    assert entry is not None and entry["stop_reason"] == "wall_clock_budget_reached" and entry["finalization_recovery"]["mode"] == "runner_stop_artifacts_reused"
    assert entry["event"] == "resume" and entry["step"] == 40 and entry["superseded_summary_json_on_disk"] is True
    assert state["finished"] is False and "stop_reason" not in state and "finalized_from_step" not in state and "finalization_recovery" not in state
    assert state["history"] == [entry] and state["ignition"] == {"failed": False} and state["checkpoint_step"] == 40
    # a live (unfinished) state has nothing to demote and gains no history entry
    live = {"wall_seconds_total": 1.0, "sessions": [], "checkpoint_step": 0, "finished": False}
    assert runner._demote_terminal_state(live, event="resume", step=0, utc="", summary_present=False) is None
    assert "history" not in live and live["finished"] is False
    # a second demotion appends
    state.update({"finished": True, "stop_reason": "target_steps_reached"})
    runner._demote_terminal_state(state, event="finalize", step=80, utc="", summary_present=False)
    assert [e["event"] for e in state["history"]] == ["resume", "finalize"] and state["history"][1]["stop_reason"] == "target_steps_reached"
