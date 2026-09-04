"""PIC design mini-sweep v1 (DRAFT): design list, catalogue identity, PIC mapping, field bindings, protocol composition,
cost anchors, closure-target extraction and the whole-set preflight.  No GPU, no stepping."""

from __future__ import annotations

import json
from math import pi

import numpy as np
import pytest

from cft_revival.pic2d.models import PIC2DValidationError
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_design_mini_sweep_v1 import closure, cost, designs, fields, preflight, protocol
from experiments.pic2d_design_mini_sweep_v1 import run as sweep_run

PRIMARY = [d for d in designs.SWEEP_DESIGNS if not d.optional]
OPTIONAL = [d for d in designs.SWEEP_DESIGNS if d.optional]


def _bindings_present() -> bool:
    return all(fields.binding_path(d.design_id).is_file() for d in designs.SWEEP_DESIGNS)


needs_bindings = pytest.mark.skipif(not _bindings_present(), reason="field bindings not produced (run `fields`)")


# -- design list and catalogue numbers -------------------------------------------------------------------------------


def test_design_list_spans_rho_with_the_reference_cusp_count():
    assert len(designs.SWEEP_DESIGNS) == 5 and len(PRIMARY) == 4 and len(OPTIONAL) == 1
    assert [d.priority for d in designs.SWEEP_DESIGNS] == [1, 2, 3, 4, 5]
    assert len({d.role for d in designs.SWEEP_DESIGNS}) == 5
    rho = {d.role: designs.design_summary(d.design_id)["min_rho_conservative"] for d in designs.SWEEP_DESIGNS}
    assert rho["low-rho"] < rho["reference"] < rho["mid-rho"] < 1.5 <= rho["hemp-like"] < rho["hemp-like-four-cusp"]
    for d in PRIMARY:
        summary = designs.design_summary(d.design_id)
        assert summary["wall_cusp_count"] == 3 and summary["cell_count"] == 4, d.design_id
    optional = designs.design_summary(OPTIONAL[0].design_id)
    assert optional["wall_cusp_count"] == 4 and optional["cell_count"] == 5 and optional["hemp_like_all_cusps"]


def test_catalogue_numbers_match_the_sealed_records():
    reference = designs.design_summary(designs.REFERENCE_DESIGN_ID)
    assert [round(c["z_c_m"] * 1e3, 3) for c in reference["cusps_l1a_or_p2"]] == [6.028, 12.0, 17.972]
    assert 0.59 < reference["min_rho_conservative"] < 0.62 and not reference["hemp_like_all_cusps"]
    assert reference["screening_v2_p_wall"]["cell-02"]["p_wall"] == 1.0 and reference["screening_v2_p_wall"]["cell-03"]["p_wall"] == 1.0
    hemp = designs.design_summary("l1a-gs-v3-056-effcbc8686")
    assert hemp["hemp_like_all_cusps"] and hemp["exit_length_m"] == 0.0
    assert abs(hemp["l1b_v1_1"]["p2_min_rho_conservative"] - 2.372) < 0.005
    low = designs.design_summary("l1a-gs-v2-047-e3196a8aa5")
    assert all(low["screening_v2_p_wall"][c]["p_wall"] == 1.0 for c in ("cell-01", "cell-02", "cell-03"))
    assert 0.34 < low["min_rho_conservative"] < 0.36
    mid = designs.design_summary("l1a-gs-v3-009-d0c686b4aa")
    assert "l1b_v1_1" not in mid and "screening_v2_p_wall" not in mid and mid["exit_length_m"] == 0.0


def test_identity_proof_and_pic_mapping_for_every_design_and_domain():
    for d in designs.SWEEP_DESIGNS:
        built = designs.build_design(d.design_id)
        checks = built.identity.get("identity_checks")
        if d.design_id != designs.REFERENCE_DESIGN_ID:
            assert checks and all(checks.values()), d.design_id
        for domain in designs.DOMAIN_OPTIONS:
            mapping = designs.pic_geometry(built, domain)
            grid = mapping.grid
            assert abs(grid.dr_m - designs.TARGET_CELL_M) <= 0.02 * designs.TARGET_CELL_M
            assert abs(grid.dz_m - designs.TARGET_CELL_M) <= 0.02 * designs.TARGET_CELL_M
            for key, snap in mapping.snaps.items():
                if isinstance(snap, dict):
                    spacing = grid.dr_m if "radius" in key else grid.dz_m
                    assert abs(snap["error_m"]) <= 0.5 * spacing + 1e-12, (d.design_id, key)
            assert grid.geometry.has_plume == (domain != "channel")
    reference = designs.pic_geometry(designs.build_design(designs.REFERENCE_DESIGN_ID), "plume-24mm")
    assert reference.grid.node_shape == (241, 961) and all(s["error_m"] == 0.0 for s in reference.snaps.values() if isinstance(s, dict))
    assert designs.pic_geometry(designs.build_design(designs.REFERENCE_DESIGN_ID), "channel").grid.node_shape == (61, 481)


# -- field bindings ------------------------------------------------------------------------------------------------


@needs_bindings
def test_field_bindings_verify_and_their_production_gates_passed():
    for d in designs.SWEEP_DESIGNS:
        binding = fields.load_binding(d.design_id)
        assert fields.verify_binding(binding)["passed"]
        assert binding["gates"]["all_passed"], d.design_id
        if d.design_id == designs.REFERENCE_DESIGN_ID:
            continue
        topology = binding["topology_under_iron"]
        catalogue = designs.catalogue_entry(d.design_id)
        agreement = binding["gates"]["topology_agreement"]
        assert agreement["p2_interior_cusp_count"] == agreement["catalogue_interior_cusp_count"] == catalogue["wall_cusp_count"]
        assert agreement["positions_within_tolerance"] and agreement["max_shift_m"] <= agreement["tolerance_m"]
        if d.design_id == "l1a-gs-v2-047-e3196a8aa5":
            # the iron moves the anode-side axis null from -1.40 to -0.11 mm and its separatrix reaches the dielectric 0.07 mm
            # from the anode: a boundary cusp under the v3.1 ambiguity tolerance, excluded from the cell map and disclosed
            assert not agreement["count_equal_strict"] and len(agreement["boundary_cusps_excluded_z_m"]["p2"]) == 1
            assert agreement["boundary_cusps_excluded_z_m"]["p2"][0] < agreement["boundary_ambiguity_tolerance_m"]
            assert 0.37 < topology["min_rho_conservative_interior"] < 0.39
        else:
            assert agreement["count_equal_strict"] and len(topology["wall_cusps"]) == catalogue["wall_cusp_count"]
        assert binding["gates"]["solver_converged"]["relative_true_residual_l2"] <= fields.RELATIVE_TOLERANCE
        assert binding["mesh_preflight"]["minimum_angle_deg"] >= fields.REJECT_BELOW_ANGLE_DEG
        need = fields.coverage_requirement(designs.build_design(d.design_id))
        assert binding["supported_pic_box"]["z_max_m"] >= need["z_max_m"] - fields.TRUNCATION_MARGIN_M - 1e-12
        assert binding["supported_pic_box"]["r_max_m"] >= fields.COVER_R_M
        l1b = designs.l1b_record(d.design_id)
        if l1b is not None:
            assert abs(topology["min_rho_conservative"] / l1b["comparison"]["p2_min_rho_conservative"] - 1.0) < 0.03
            l1b_gate = binding["gates"]["l1b_accepted_grid_agreement"]
            assert l1b_gate["passed"] and l1b_gate["max_abs_diff_channel_t"] < 0.005   # level-0 padded vs level-1 padding-0.5: a few mT at 0.3 T


@needs_bindings
def test_design_field_map_binds_the_checkpoint_and_scale_and_fails_closed():
    design_id = "l1a-gs-v3-009-d0c686b4aa"
    built = designs.build_design(design_id)
    binding = fields.load_binding(design_id)
    mapping = designs.pic_geometry(built, "channel", target_cell_m=2.5e-4)
    field_map = fields.design_field_map(mapping, binding)
    assert field_map.provenance["kind"] == "p2-direct-node-sample-design-mini-sweep"
    assert field_map.provenance["checkpoint_file_sha256"] == binding["map"]["checkpoint_file_sha256"]
    assert field_map.provenance["source_strength_scale"] == binding["source_strength_scale"] == built.source_strength_scale
    assert np.all(field_map.b_r_t[0, :] == 0.0) and 0.05 < field_map.max_b_t < 1.0
    tampered = json.loads(json.dumps(binding))
    tampered["map"]["checkpoint_file_sha256"] = "0" * 64
    with pytest.raises(PIC2DValidationError):
        fields.design_field_map(mapping, tampered)
    too_long = designs.pic_geometry(built, "plume-24mm", target_cell_m=1.0e-3, plume_length_m=0.040)
    with pytest.raises(PIC2DValidationError):
        fields.design_field_map(too_long, binding)


def test_reference_field_map_uses_the_existing_pipeline(p2_available):
    if not p2_available:
        pytest.skip("P2 checkpoint not available")
    built = designs.build_design(designs.REFERENCE_DESIGN_ID)
    binding = fields.write_reference_binding()
    channel = fields.design_field_map(designs.pic_geometry(built, "channel", target_cell_m=2.5e-4), binding)
    assert channel.provenance["kind"] == "p2-psi-bicubic-node-sample"
    plume = fields.design_field_map(designs.pic_geometry(built, "plume-12mm", target_cell_m=5.0e-4), binding)
    assert plume.provenance["kind"] == "p2-direct-node-sample-plume-domain" and "field_source" not in plume.provenance


# -- protocol composition ------------------------------------------------------------------------------------------


def test_per_design_protocol_reproduces_the_reference_and_scales_the_feed():
    reference_channel, _ = protocol.build_protocol(designs.REFERENCE_DESIGN_ID, "channel")
    assert reference_channel["status"] == protocol.STATUS and "DRAFT" in reference_channel["status"]
    assert abs(reference_channel["operating_point"]["neutral_inventory"]["feed_atoms_per_s"] / 8.551102004120011e16 - 1.0) < 1e-9
    assert reference_channel["case"]["radial_cells"] == 60 and reference_channel["case"]["axial_cells"] == 480
    assert sum(1 for key in reference_channel if key.startswith("budget")) == 1
    assert "field_authority" not in reference_channel and "field_plume_extension" not in reference_channel
    runner.build_config(reference_channel, backend="cpu")
    reference_plume, _ = protocol.build_protocol(designs.REFERENCE_DESIGN_ID, "plume-24mm")
    cathode = reference_plume["operating_point"]["cathode"]
    assert (cathode["r_inner_m"], cathode["r_outer_m"]) == (0.0005, 0.002)
    assert abs(cathode["z_start_m"] - 0.0243) < 1e-12 and abs(cathode["z_end_m"] - 0.025) < 1e-12
    assert abs(reference_plume["budget_design_mini_sweep"]["ion_transit_time_s"] - 3.81e-6) < 0.02e-6
    assert reference_plume["numerics"]["plume_boundary_gate"]["enforce_after_s"] == reference_plume["budget_design_mini_sweep"]["ion_transit_time_s"]
    hemp, mapping = protocol.build_protocol("l1a-gs-v3-056-effcbc8686", "channel")
    ratio = hemp["operating_point"]["neutral_inventory"]["feed_atoms_per_s"] / reference_channel["operating_point"]["neutral_inventory"]["feed_atoms_per_s"]
    assert abs(ratio - (mapping.geometry.exit_radius_m / 0.003) ** 2) < 1e-9
    assert hemp["stopping_rule"]["wall_budget_seconds"] >= 3600.0 and hemp["stopping_rule"]["wall_budget_seconds"] % 600.0 == 0.0
    assert hemp["case"]["macro_weight"] == 60000.0
    runner.build_config(hemp, backend="cpu")
    weight, _ = protocol.macro_weight_for(12.0)
    assert weight == pytest.approx(60000.0 * 12.0 / 8.0)


def test_draft_protocol_document_on_disk_matches_the_generator_and_is_marked_draft():
    document = protocol.draft_protocol_document()
    assert "DRAFT" in document["status"] and "NOT preregistered" in document["preregistration"]["state"]
    assert [d["design_id"] for d in document["designs"]] == list(designs.design_ids())
    assert len(document["closure_targets"]) >= 15
    on_disk = json.loads(protocol.DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    for key in ("designs", "operating_point_policy", "grid_policy", "stopping_rule", "replication_policy", "closure_targets", "cost_table_hours_to_3_transits", "recommended_schedule"):
        assert on_disk[key] == document[key], key


# -- cost model ----------------------------------------------------------------------------------------------------


def test_cost_model_reproduces_the_measured_anchors_and_the_v21_spec_row():
    residuals = {row["case"]: row["relative_error"] for row in cost.anchor_residuals()}
    assert abs(residuals["plume attempt 8 (241 x 721)"]) < 0.01
    assert all(0.0 <= value < 0.15 for name, value in residuals.items() if "channel" in name)
    built = designs.build_design(designs.REFERENCE_DESIGN_ID)
    row = cost.design_cost(designs.pic_geometry(built, "plume-24mm"))
    assert abs(row["ms_per_step"] / 8.22 - 1.0) < 0.03 and abs(row["steps_to_transits"] / 7.62e6 - 1.0) < 0.01
    assert abs(row["wall_hours"] / 17.4 - 1.0) < 0.05 and abs(row["factorisation_s"] / 60.0 - 11.8) < 0.3
    channel = cost.design_cost(designs.pic_geometry(built, "channel"))
    assert abs(channel["transit_s"] - 2.4e-6) < 1e-12 and channel["wall_hours"] < 3.5
    table = cost.cost_table([designs.build_design(d.design_id) for d in designs.SWEEP_DESIGNS])
    schedule = cost.serial_schedule(table, option="channel", replicate_design_ids=("l1a-gs-v3-056-effcbc8686",), extra=((designs.REFERENCE_DESIGN_ID, "plume-24mm"),))
    assert len(schedule["items"]) == 7 and schedule["total_hours"] == pytest.approx(sum(i["wall_hours"] for i in schedule["items"]))


# -- closure targets -----------------------------------------------------------------------------------------------


def test_kornfeld_mapping_for_three_and_four_cusps():
    cells = [{"cell_id": "cell-01", "kind": "anode_partial", "z_start_m": 0.0, "z_end_m": 0.006}, {"cell_id": "cell-02", "kind": "interior", "z_start_m": 0.006, "z_end_m": 0.012},
             {"cell_id": "cell-03", "kind": "interior", "z_start_m": 0.012, "z_end_m": 0.018}, {"cell_id": "cell-04", "kind": "exit_partial", "z_start_m": 0.018, "z_end_m": 0.024}]
    three = closure.kornfeld_mapping([0.006, 0.012, 0.018], cells)
    assert three["cusp_slots_z_m"] == {"cusp_1": None, "cusp_2": 0.018, "cusp_3": 0.012, "cusp_4_anode": 0.006}
    assert three["model_cell_of_catalogue_cell"]["cell-04"] == "model cell 1" and three["model_cell_of_catalogue_cell"]["cell-01"] == "model cell 4"
    four = closure.kornfeld_mapping([0.004, 0.008, 0.012, 0.016], cells)
    assert four["cusp_slots_z_m"]["cusp_1"] == 0.016 and four["cusp_slots_z_m"]["cusp_4_anode"] == 0.004
    with pytest.raises(ValueError):
        closure.kornfeld_mapping([0.002, 0.004, 0.006, 0.008, 0.010], cells)


def test_closure_target_extraction_recovers_a_synthetic_plateau():
    built = designs.build_design(designs.REFERENCE_DESIGN_ID)
    mapping = designs.pic_geometry(built, "channel", target_cell_m=5.0e-4)      # 6 x 48 cells
    grid = mapping.grid
    nr1, nz1 = grid.node_shape
    dz = grid.dz_m
    cusps = [0.006, 0.012, 0.018]
    cells = [{"cell_id": "cell-01", "kind": "anode_partial", "z_start_m": 0.0, "z_end_m": 0.006}, {"cell_id": "cell-02", "kind": "interior", "z_start_m": 0.006, "z_end_m": 0.012},
             {"cell_id": "cell-03", "kind": "interior", "z_start_m": 0.012, "z_end_m": 0.018}, {"cell_id": "cell-04", "kind": "exit_partial", "z_start_m": 0.018, "z_end_m": 0.024}]
    z_cells = (np.arange(grid.axial_cells) + 0.5) * dz
    wall_e = np.zeros(grid.axial_cells)
    for z_c, flux in zip(cusps, (1.0e22, 2.0e22, 3.0e22)):
        wall_e[np.abs(z_cells - z_c) <= 1.0e-3] = flux
    wall_i = np.full(grid.axial_cells, 5.0e21)
    phi = np.tile(np.linspace(300.0, 0.0, nz1), (nr1, 1))
    maps = {
        "phi_v": phi, "t_e_ev": np.full((nr1, nz1), 8.0), "n_e_per_m3": np.full((nr1, nz1), 1.0e17),
        "ionization_rate_per_m3_s": np.full((nr1, nz1), 1.0e23), "wall_electron_flux_per_m2_s": wall_e, "wall_ion_flux_per_m2_s": wall_i,
        "wall_electron_mean_energy_ev": np.full(grid.axial_cells, 20.0), "window_steps": np.array([400000]),
    }
    out = closure.extract_targets(maps, mapping, cusps, cells, injected_electron_current_a=3.0e-3)
    assert out["kornfeld_mapping"]["cusp_slots_z_m"]["cusp_4_anode"] == 0.006
    e = closure.ELEMENTARY_CHARGE_C
    geometry = grid.geometry
    for cusp, flux in zip(out["cusps"], (1.0e22, 2.0e22, 3.0e22)):
        window = np.abs(z_cells - cusp["z_c_m"]) <= 1.0e-3
        # wall area per cell: 2 pi r_wall(z) dz x slant (the last cusp's window reaches into the divergent cone)
        expected = e * flux * sum(2.0 * pi * closure._wall_radius(geometry, z) * dz * closure._slant(geometry, z) for z in z_cells[window])
        assert cusp["electron_wall_current_a"] == pytest.approx(expected, rel=1e-9)
        assert cusp["sheath_drop_v"] == pytest.approx(0.0) and cusp["near_wall_electron_temperature_ev"] == pytest.approx(8.0)
        assert cusp["leak_width_fwhm_m"] == pytest.approx(int(window.sum()) * dz)
    shares = [c["ionisation_share"] for c in out["cells"]]
    assert sum(shares) == pytest.approx(1.0) and all(0.0 < s < 1.0 for s in shares)
    assert out["diffuse_non_cusp_electron_wall_current_a"] == 0.0
    chain = out["kornfeld_chain"]["cusps_exit_to_anode"]
    assert [round(row["z_c_m"], 6) for row in chain] == [0.018, 0.012, 0.006]
    je = 3.0e-3 + out["cells"][3]["ionisation_current_a"]
    assert chain[0]["je_arriving_a"] == pytest.approx(je) and chain[0]["p_transit"] == pytest.approx(chain[0]["electron_wall_current_a"] / je)
    assert chain[1]["je_arriving_a"] == pytest.approx(je - chain[0]["electron_wall_current_a"] + out["cells"][2]["ionisation_current_a"])
    for cell in out["cells"]:
        assert cell["ion_wall_loss_fraction"] > 0.0 and cell["density_weighted_electron_temperature_ev"] == pytest.approx(8.0)
    assert len(out["potential_steps_v"]) == 3 and all(step < 0.0 for step in out["potential_steps_v"])
    assert len(closure.closure_target_table()) == len(closure.CLOSURE_TARGETS)


# -- preflight and the launch guard --------------------------------------------------------------------------------


@needs_bindings
def test_whole_set_preflight_over_every_design_passes_for_the_channel_option():
    report = preflight.preflight_all("channel", log=lambda text: None)
    failures = {r["design_id"]: {k: v for k, v in r["gates"].items() if not v["passed"]} for r in report["designs"] if not r["passed"]}
    assert report["all_passed"], failures
    assert report["design_count"] == len(designs.SWEEP_DESIGNS)
    for record in report["designs"]:
        assert set(record["gates"]) == {"identity", "field_binding", "grid", "field_map", "mesh_masks", "protocol", "cathode_connectivity", "cost"}
        assert record["gates"]["field_map"]["stability"]["stable"]
        assert record["gates"]["field_map"]["stability"]["omega_ce_dt"] <= 0.2


def test_run_refuses_to_launch_without_the_draft_flag(capsys):
    assert sweep_run.main(["run", "--design", designs.REFERENCE_DESIGN_ID, "--domain", "channel"]) == 2
    assert "REFUSED" in capsys.readouterr().err
