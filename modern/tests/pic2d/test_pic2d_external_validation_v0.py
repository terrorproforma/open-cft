"""External validation v0 (DRAFT): the Brandt 2016 reference record, the geometry mapping onto the v1.1 contract and the PIC grid, the
field binding and its published-anchor gates, the ASME V&V 20 comparison spec (schema + metric), the protocol composition on the
steady-state v4 template (static neutrals, 20 um, dt policy, W cap, budget), the grid argument and the whole-set preflight.  No GPU,
no stepping, no P2 solve (the bound checkpoint is read where it exists)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from cft_revival.geometry import AxisymmetricCFTGeometry
from cft_revival.pic2d.models import PIC2DValidationError
from cft_revival.validation.contracts import ValidationError, validate_doi
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_external_validation_v0 import (
    comparison,
    fields,
    geometry,
    preflight,
    protocol,
    reference,
)
from experiments.pic2d_external_validation_v0 import run as ev_run

needs_binding = pytest.mark.skipif(not fields.binding_path().is_file(), reason="field binding not produced (run `fields`)")


def _binding_is_real() -> bool:
    """The checkpoint bundle is a Git LFS object; on a checkout without LFS content it is a pointer file."""

    if not fields.binding_path().is_file():
        return False
    try:
        fields.verify_binding(fields.load_binding())
    except (PIC2DValidationError, OSError, KeyError, ValueError):
        return False
    return True


needs_checkpoint = pytest.mark.skipif(not _binding_is_real(), reason="bound checkpoint not present as bytes (LFS pointer or not produced)")


# -- reference record ---------------------------------------------------------------------------------------------------------


def test_reference_doi_and_record_are_complete():
    assert reference.DOI == "10.2322/tastj.14.Pb_235" == validate_doi("https://doi.org/10.2322/tastj.14.Pb_235")
    with pytest.raises(ValidationError):
        validate_doi("not a doi")
    document = reference.reference_document()
    for key in ("channel_radius_m", "channel_length_m", "anode_voltage_v", "mass_flow", "neutral_background", "magnet_stack", "field_anchors", "grid", "time_step_s", "self_similarity_scaling"):
        assert key in document["setup"] and document["setup"][key]["source"], key
    assert document["setup"]["channel_radius_m"]["value"] == 1.5e-3 and document["setup"]["channel_length_m"]["value"] == 14.0e-3
    assert document["setup"]["anode_voltage_v"]["value"] == 400.0 and document["setup"]["neutral_background"]["value"]["mean_density_per_m3"] == 2.0e20
    stack = document["setup"]["magnet_stack"]["value"]
    assert stack["magnet_count"] == 3 and stack["distance_ring_count"] == 5 and stack["magnet_axial_length_m"] == 5e-3 and stack["distance_ring_axial_length_m"] == 0.5e-3
    assert stack["magnet_outer_radius_m"] == 15e-3 and stack["distance_ring_outer_radius_m"] == 8e-3 and stack["magnet_inner_radius_m"] == 2.5e-3
    assert len(document["alternatives_considered"]) >= 4 and all(validate_doi(a["doi"]) for a in document["alternatives_considered"])
    assert reference.THESIS["urn"].startswith("urn:nbn:de:gbv:8-diss-")


def test_reported_quantities_carry_kind_source_and_uncertainty_budget():
    for qid, row in reference.REPORTED.items():
        assert row["kind"] and row["source"] and row["unit"], qid
        d, u_d, components = reference.reported_value_and_u_d(qid)
        assert math.isfinite(d) and u_d > 0.0 and components, qid
        assert abs(u_d - math.sqrt(sum(c["standard"] ** 2 for c in components))) < 1e-15 * max(1.0, u_d)
        for c in components:
            assert c["method"] and c["standard"] >= 0.0
    # the paper-vs-thesis spread of the reference's own two runs enters u_D where both exist
    assert reference.REPORTED["anode_electron_current_a"]["second_run"]["value"] == 4.7e-3
    assert any(c["name"] == "reference_variability" and abs(c["standard"] - 0.2e-3) < 1e-12 for c in reference.REPORTED["anode_electron_current_a"]["u_d"])
    assert any(c["name"] == "reference_variability" and c["standard"] == 5.0 for c in reference.REPORTED["plume_peak_angle_deg"]["u_d"])
    # figure-read quantities are marked as such
    assert "figure" in reference.REPORTED["wall_ion_energy_max_ev"]["kind"] and "figure" in reference.REPORTED["ion_density_typical_per_m3"]["kind"]
    assert "text" in reference.REPORTED["anode_electron_current_a"]["kind"]
    # net ionisation is Brandt's I_a / e Q_in
    assert abs(4.3e-3 / reference.I_FEED_A - 0.244) < 0.002


# -- geometry mapping ---------------------------------------------------------------------------------------------------------


def test_reconstruction_builds_under_the_v1_1_contract_and_is_deterministic():
    g1, g2 = geometry.brandt_micro_hempt_geometry(), geometry.brandt_micro_hempt_geometry()
    assert isinstance(g1, AxisymmetricCFTGeometry) and g1.canonical_sha256 == g2.canonical_sha256
    assert g1.schema_version.endswith("/1.1.0") and g1.config_id == geometry.CONFIG_ID
    assert len(g1.stages) == 3 and [s.magnetization.polarity for s in g1.stages] == [1, -1, 1]
    magnets = [r for r in g1.regions if r.role == "permanent_magnet"]
    poles = [r for r in g1.regions if r.role == "pole_piece"]
    assert [(m.z_min_m, m.z_max_m) for m in magnets] == [(0.0, 0.005), (0.0055, 0.0105), (0.011, 0.016)]
    assert [(p.z_min_m, p.z_max_m) for p in poles] == [(0.005, 0.0055), (0.0105, 0.011)]
    assert all(m.r_inner_start_m == 2.5e-3 and m.r_outer_start_m == 15e-3 for m in magnets)
    # anode frame: magnet centres at 0 / 5.5 / 11 mm, rings at 2.75 / 8.25 mm, channel exit at 14 mm
    assert [round(geometry.anode_frame_z(s.center_z_m) * 1e3, 3) for s in g1.stages] == [0.0, 5.5, 11.0]
    assert [round(geometry.anode_frame_z(0.5 * (p.z_min_m + p.z_max_m)) * 1e3, 3) for p in poles] == [2.75, 8.25]
    assert abs(geometry.anode_frame_z(g1.chamber.length_m) - 0.0165 + 0.0025 - 0.014) < 1e-15 or abs(g1.chamber.length_m - geometry.AXIAL_OFFSET_M - 0.014) < 1e-15
    assert g1.chamber.outer_radius_m == 1.5e-3 and g1.chamber.injector_length_m == geometry.AXIAL_OFFSET_M
    # the yoke is the inert placeholder, the rings are the linear iron
    assert g1.material_by_id(next(r for r in g1.regions if r.role == "yoke").material_id).relative_permeability == 1.0
    assert g1.material_by_id(poles[0].material_id).relative_permeability == 4000.0


def test_sensitivity_variant_removes_only_the_rings():
    base, variant = geometry.brandt_micro_hempt_geometry(), geometry.brandt_micro_hempt_geometry(pole_vacuum=True)
    assert variant.canonical_sha256 != base.canonical_sha256 and variant.config_id.endswith("no-rings-sensitivity")
    assert variant.material_by_id(next(r for r in variant.regions if r.role == "pole_piece").material_id).relative_permeability == 1.0
    same = [(r.region_id, r.z_min_m, r.z_max_m, r.r_inner_start_m, r.r_outer_start_m) for r in base.regions]
    assert same == [(r.region_id, r.z_min_m, r.z_max_m, r.r_inner_start_m, r.r_outer_start_m) for r in variant.regions]


def test_approximations_are_listed_and_the_mapping_table_names_them():
    ids = [a["id"] for a in geometry.APPROXIMATIONS]
    assert ids == [f"A{i}" for i in range(1, 10)]
    for a in geometry.APPROXIMATIONS:
        assert a["reference"] and a["represented"] and a["effect"] and a["quantified_by"], a["id"]
    table = geometry.mapping_table()
    assert table["geometry_sha256"] == geometry.brandt_micro_hempt_geometry().canonical_sha256
    named = {a for row in table["rows"] if row.get("approximation") for a in row["approximation"].replace(",", " ").split() if a.startswith("A")}
    assert {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A9"} <= named
    assert table["frames"]["axial_offset_m"] == 2.5e-3
    stack = table["stack"]
    assert sum(1 for r in stack if r["role"] == "permanent_magnet") == 3 and sum(1 for r in stack if r["role"] == "pole_piece") == 4   # 2 represented + 2 not represented


def test_pic_mapping_hits_every_reference_line_at_20um():
    channel = geometry.pic_mapping("channel")
    assert (channel.grid.radial_cells, channel.grid.axial_cells) == (75, 700) and channel.grid.node_shape == (76, 701)
    assert abs(channel.grid.dr_m - 20e-6) < 1e-15 and abs(channel.grid.dz_m - 20e-6) < 1e-15
    assert geometry.worst_snap_in_cells(channel) == 0.0 and not channel.geometry.has_plume
    plume = geometry.pic_mapping("plume-brandt")
    assert (plume.grid.radial_cells, plume.grid.axial_cells) == (256, 1024)          # the published grid exactly
    assert plume.geometry.has_plume and abs(plume.geometry.plume_radius_m - 5.12e-3) < 1e-15 and abs(plume.geometry.body_dielectric_radius_m - 2.5e-3) < 1e-15
    assert abs(plume.geometry.domain_z_max_m - 20.48e-3) < 1e-12 and geometry.worst_snap_in_cells(plume) <= 1e-9
    assert abs(geometry.channel_volume_m3(channel) - math.pi * 1.5e-3**2 * 14e-3) < 1e-18
    with pytest.raises(ValueError):
        geometry.pic_mapping("plume-24mm")
    coarse = geometry.pic_mapping("channel", target_cell_m=protocol.GRIDS["33um"]["cell_m"])
    assert (coarse.grid.radial_cells, coarse.grid.axial_cells) == (45, 420)


# -- comparison spec --------------------------------------------------------------------------------------------------------------


def test_comparison_spec_validates_and_covers_the_reported_quantities():
    document = comparison.comparison_document()
    assert comparison.validate_comparison_spec(document) == []
    ids = [q["quantity_id"] for q in document["quantities"]]
    assert set(ids) == set(reference.REPORTED)
    channel = [q for q in document["quantities"] if "channel" in q["comparable_under"]]
    plume_only = [q for q in document["quantities"] if q["comparable_under"] == ["plume-brandt"]]
    assert len(channel) >= 8 and {q["quantity_id"] for q in plume_only} == {"plume_peak_angle_deg", "electron_energy_near_exit_cusp_ev"}
    assert document["reference"]["claim_ceiling"] == "CROSS_MODEL_AGREEMENT" and document["reference"]["doi"] == reference.DOI
    assert len(document["closure_differences"]) >= 8 and len(document["inconclusive_conditions"]) >= 8
    for q in document["quantities"]:
        assert q["u_input"]["propagated"] is False and q["u_input"]["components"]
        assert q["u_num_predicted"]["grid_caveat"]
    # the schema validator catches a broken row
    broken = json.loads(json.dumps(document))
    broken["quantities"][0]["u_D"]["standard"] *= 2.0
    del broken["quantities"][1]["tolerance"]
    broken["quantities"][2]["comparable_under"] = ["nowhere"]
    problems = comparison.validate_comparison_spec(broken)
    assert any("root-sum-square" in p for p in problems) and any("tolerance" in p for p in problems) and any("comparable_under" in p for p in problems)


def test_validation_metric_forms_e_and_u_val_and_the_predeclared_statement():
    rows = {q["quantity_id"]: q for q in comparison.comparison_rows()}
    q = rows["anode_electron_current_a"]
    # S = D -> agreement within u_val; u_val = sqrt(u_num^2 + u_D^2) with u_input = 0 (not propagated)
    m = comparison.validation_metric(q, 4.3e-3)
    assert m["E"] == 0.0 and m["verdict"] == "agreement_within_u_val" and m["u_input"] == 0.0 and m["u_input_propagated"] is False
    assert abs(m["u_val"] - math.sqrt((0.057 * 4.3e-3) ** 2 + q["u_D"]["standard"] ** 2)) < 1e-15
    # just outside 2 u_val but inside the 20 % tolerance
    s = 4.3e-3 + 2.0 * m["u_val"] + 1e-6
    assert comparison.validation_metric(q, s)["verdict"] == "agreement_within_tolerance"
    assert comparison.validation_metric(q, 4.3e-3 * 1.25)["verdict"] == "discrepancy"
    # a measured u_num replaces the predicted class value
    assert comparison.validation_metric(q, 4.3e-3, u_num_measured=1e-3)["u_num"] == 1e-3
    # log-scale row: E in dex
    density = comparison.validation_metric(rows["ion_density_typical_per_m3"], 2e19)
    assert density["scale"] == "log10" and abs(density["E"] - math.log10(2.0)) < 1e-12 and density["verdict"] != "discrepancy"
    assert comparison.validation_metric(rows["ion_density_typical_per_m3"], 3e20)["verdict"] == "discrepancy"
    # potential steps: absolute u_num and tolerance
    step = comparison.validation_metric(rows["potential_drop_first_cusp_v"], 14.0)
    assert step["u_num"] == 2.0 and step["tolerance"] == 5.0 and step["verdict"] == "agreement_within_u_val"
    # u_D 2.5 V + u_num 2 V -> 2 u_val = 6.4 V exceeds the 5 V tolerance: the row cannot discriminate at the tolerance level and says so
    assert step["tolerance_below_expanded_u_val"] and step["u_val_dominated_by"] == "u_D"
    assert comparison.validation_metric(rows["potential_drop_first_cusp_v"], 17.0)["verdict"] == "discrepancy"
    assert not m["tolerance_below_expanded_u_val"]


# -- protocol composition (template-only, no field) ---------------------------------------------------------------------------


def test_primary_protocol_is_the_v4_template_with_the_reference_operating_point():
    p, mapping = protocol.build_protocol("base", "20um")
    template = protocol.load_template()
    assert p["template_protocol"]["experiment_id"] == template["experiment_id"] == "pic2d-cft-steady-state-v4"
    assert p["status"].startswith("DRAFT_NOT_PREREGISTERED") and "preregistration" not in p and "reference_run" not in p
    op = p["operating_point"]
    assert op["anode_potential_v"] == 400.0 and op["exit_plane_potential_v"] == 0.0 and op["neutral_density_per_m3"] == 2e20 and op["neutral_temperature_k"] == 500.0
    assert op["neutral_inventory"] is None and op["electron_injection_current_a"] == 1.8e-3 and op["electron_injection_temperature_ev"] == 1.0
    assert op["seed_plasma_density_per_m3"] == template["operating_point"]["seed_plasma_density_per_m3"]
    num = p["numerics"]
    assert num["dt_s"] == 0.7e-12 and num["stability_reference"] == {"density_per_m3": 1e19, "electron_temperature_ev": 10.0, "max_electron_energy_ev": 400.0}
    assert num["stability_limits"]["max_cell_debye_ratio"] == math.pi and num["stability_limits"]["max_omega_pe_dt"] == 0.2
    assert num["peak_debye_gate"]["max_cells_per_debye"] == math.pi and num["peak_debye_gate"]["soft_cells_per_debye"] == 2.5 and num["peak_debye_gate"]["window_steps"] == 400000
    assert num["frame_recorder"] == {"cadence_steps": 40000, "precision": "float32"} and "anomalous_collisions" not in num
    assert p["stopping_rule"]["grid_heating_triad"]["windowed_energy_residual_over_electrode_work_max"] == 0.05
    case = p["case"]
    assert (case["radial_cells"], case["axial_cells"]) == (75, 700) and case["seed"] == 20260903
    # W: parity would give ~100 M particles at the declared mean density -> raised to the 12 M cap (disclosed)
    assert case["macro_weight"] > protocol.macro_weight_policy(mapping)[1]["parity_weight"] and "cap" in case["macro_weight_policy"]["rule"]
    assert abs(p["budget_external_validation_v0"]["particles_projected_m"] - 12.0) < 1e-9
    # transit 1.4 us; 3 transits = 6.0 M steps; budget 1.5x the MPS-4 projection, rounded to 10 min
    assert abs(p["budget_external_validation_v0"]["ion_transit_time_s"] - 1.4e-6) < 1e-15
    assert abs(p["budget_external_validation_v0"]["steps_to_3_transits"] - 6.0e6) < 1.0
    budget = p["stopping_rule"]["wall_budget_seconds"]
    assert budget % 600 == 0 and budget >= 1.5 * p["budget_external_validation_v0"]["hours_to_3_transits_mps4"] * 3600 - 600
    # a-priori numbers: omega_pe dt gate density at 0.7 ps ~2.6e19; the hard Debye level on 20 um at 10 eV binds first, at 1.36e19 (= n_max of the budget block)
    assert 2.5e19 < protocol.density_at_omega_pe_dt(0.2, 0.7e-12) < 2.6e19
    hard = protocol.density_at_cells_per_debye(math.pi, 20e-6, 10.0)
    assert 1.3e19 < hard < 1.4e19 and runner.protocol_budget(p)["n_max_per_m3"] == hard
    # the shared runner accepts the composed protocol with static neutrals
    config = runner.build_config(p, backend="cpu")
    assert config.neutral_inventory is None and config.mcc is not None and config.mcc.neutral_density_per_m3 == 2e20 and config.anomalous is None
    assert config.injection.electron_current_a == 1.8e-3 and config.potentials.anode_v == 400.0


def test_bohm_variant_switches_only_the_anomalous_hook():
    base, _ = protocol.build_protocol("base", "20um")
    bohm, _ = protocol.build_protocol("bohm-0.4", "20um")
    assert bohm["numerics"]["anomalous_collisions"]["alpha"] == 0.4 and "anomalous_collisions" not in base["numerics"]
    assert runner.build_config(bohm, backend="cpu").anomalous.alpha == 0.4
    for key in ("operating_point", "geometry", "stopping_rule"):
        assert json.dumps(base[key], sort_keys=True) == json.dumps(bohm[key], sort_keys=True), key
    assert base["case"]["macro_weight"] == bohm["case"]["macro_weight"] and base["option"] == "channel-20um" and bohm["option"] == "channel-20um-bohm-0.4"


def test_grid_argument_rejects_33um_and_admits_20um_at_the_published_density():
    argument = protocol.grid_argument()
    rows = {r["grid"]: r for r in argument["rows"]}
    assert abs(argument["debye_length_at_published_m"] - 7.434e-6) < 1e-8
    assert 2.6 < rows["20um"]["cells_per_debye_at_published"] < 2.8 and rows["20um"]["admissible_hard_pi"] and not rows["20um"]["soft_2p5_met"]
    assert rows["33um"]["cells_per_debye_at_published"] > 4.0 and not rows["33um"]["admissible_hard_pi"]
    assert rows["15um"]["soft_2p5_met"] and rows["15um"]["relative_cell_count"] > 1.7
    assert rows["20um"]["cells"] == [75, 700] and rows["33um"]["cells"] == [45, 420]
    assert all(r["electron_courant_400ev"] < 1.0 and r["omega_pe_dt_at_published"] < 0.2 for r in argument["rows"])
    assert "PRIMARY = 20 um" in argument["decision"]
    assert protocol.admissible_dt(0.7e-12, 0.7, 0.2)[0] == 0.7e-12                     # 0.086 at the channel maximum
    reduced, policy = protocol.admissible_dt(0.7e-12, 2.0, 0.2)                        # a 2 T pole face would force a reduction
    assert reduced < 0.7e-12 and policy["omega_ce_dt"] <= 0.19


def test_cost_row_scales_the_h100_mps4_anchor_by_cells_and_particles():
    mapping = geometry.pic_mapping("channel")
    row = protocol.cost_row(mapping, dt_s=0.7e-12, macro_weight=1e5, projected_total_m=12.0)
    assert row["nodes"] == [76, 701] and abs(row["transit_s"] - 1.4e-6) < 1e-15 and abs(row["steps_to_3_transits"] - 6.0e6) < 1.0
    assert 15.0 < row["ms_per_step_h100_mps4_per_process"] < 22.0 and row["ms_per_step_h100_solo_equivalent"] < row["ms_per_step_h100_mps4_per_process"]
    assert 25.0 < row["hours_to_3_transits_mps4"] < 40.0 and row["hours_to_reference_time_mps4"] > 400.0
    assert 15.0 < row["device_gb_projected"] < 20.0


# -- field binding (needs the produced field) -------------------------------------------------------------------------------


@needs_binding
def test_field_binding_records_the_calibration_gates_and_genealogy():
    binding = fields.load_binding()
    gates = binding["gates"]
    assert gates["all_passed"] is True and binding["geometry"]["axial_offset_m"] == 2.5e-3
    assert binding["geometry"]["geometry_sha256"] == geometry.brandt_micro_hempt_geometry().canonical_sha256
    g1 = gates["G1_scale"]
    assert g1["passed"] and 0.8 <= g1["scale"] <= 1.2 and abs(g1["implied_remanence_t"] - 1.05 * g1["scale"]) < 1e-12 and binding["source_strength_scale"] == g1["scale"]
    assert gates["G2_interior_nulls"]["passed"] and len(gates["G2_interior_nulls"]["interior_nulls_m"]) == 2
    for match in gates["G2_interior_nulls"]["matches"]:
        assert match["within"] and match["shift_m"] <= 0.5e-3
    assert gates["G3_exit_null"]["passed"] and abs(gates["G3_exit_null"]["nearest_null_m"] - 16e-3) <= 1.5e-3
    assert gates["G4_exit_point"]["passed"] and abs(gates["G4_exit_point"]["b_t"] - 0.05) <= 0.025
    assert gates["G5_axis_maximum"]["passed"] and abs(gates["G5_axis_maximum"]["axis_max_t"] - 0.7) <= 0.07
    assert all(m["within"] for m in gates["G5_axis_maximum"]["magnet_centre_matches"])
    assert gates["G7_mesh_solver_coverage"]["passed"] and binding["mesh_preflight"]["minimum_angle_deg"] >= 5.0 and binding["solve"]["relative_true_residual_l2"] <= 2e-10
    # D6 is a descriptor, never gated; the genealogy records the withdrawn G5 / G6 rules and their outcomes
    d6 = gates["D6_wall_cusp_field"]
    assert d6["gated"] is False and "passed" not in d6 and len(d6["cusps"]) == 2
    assert all(0.0 < r["low_field_contour_radius_m"] < 1.5e-3 for r in d6["cusps"])
    assert {g["gate"] for g in gates["genealogy"]} == {"G5", "G6"} and all("FAIL" in g["outcome_on_first_solve"] for g in gates["genealogy"])
    # the no-ring sensitivity bracket exists and moved the wall field but not the null positions materially
    bracket = binding["sensitivity_no_rings"]["bracket"]
    assert max(bracket["interior_nulls_m"]["shifts_m"]) < 0.5e-3 and all(b > 0.0 for b in bracket["wall_cusp_b_t"]["no_rings"])
    # the supported PIC box (anode frame) covers both PIC boxes
    supported = binding["supported_pic_box_anode_frame"]
    plume = geometry.pic_mapping("plume-brandt")
    assert supported["r_max_m"] >= plume.geometry.max_radius_m and supported["z_max_m"] >= plume.geometry.domain_z_max_m and supported["z_min_m"] <= 0.0


@needs_checkpoint
def test_field_map_applies_offset_and_scale_and_refuses_a_box_beyond_the_field():
    binding = fields.load_binding()
    mapping = geometry.pic_mapping("channel")
    fm = fields.brandt_field_map(mapping, binding)
    assert fm.provenance["axial_offset_m"] == 2.5e-3 and fm.provenance["source_strength_scale"] == binding["source_strength_scale"]
    assert fm.provenance["kind"] == "p2-direct-node-sample-external-validation-v0" and fm.provenance["reference_doi"] == reference.DOI
    # the axis field at the published anchor (z = 11 mm -> node 550) equals 0.6 T by calibration, the interior nulls sit where the gates say
    z = np.asarray(mapping.grid.z_m)
    bz_axis = np.asarray(fm.b_z_t)[0] if hasattr(fm, "b_z_t") else np.asarray(fm.b_z)[0]
    k11 = int(np.argmin(np.abs(z - 11e-3)))
    assert abs(abs(bz_axis[k11]) - 0.6) < 0.01
    crossings = [float(z[k]) for k in range(len(z) - 1) if bz_axis[k] * bz_axis[k + 1] < 0.0]
    for null in binding["gates"]["G2_interior_nulls"]["interior_nulls_m"]:
        assert min(abs(c - null) for c in crossings) < 2 * mapping.grid.dz_m
    assert 0.6 < fm.max_b_t < 0.8
    # tampering with the scale is caught by the binding verification
    tampered = json.loads(json.dumps(binding))
    tampered["geometry"]["axial_offset_m"] = 0.0
    with pytest.raises(PIC2DValidationError):
        fields.brandt_field_map(mapping, tampered)
    tampered = json.loads(json.dumps(binding))
    tampered["map"]["checkpoint_file_sha256"] = "0" * 64
    with pytest.raises(PIC2DValidationError):
        fields.verify_binding(tampered)
    # a box beyond the supported field is refused
    too_far = geometry.PicMapping(geometry.CONFIG_ID, "channel", mapping.geometry, mapping.grid, mapping.snaps)
    big = geometry.pic_mapping("plume-brandt")
    fake = json.loads(json.dumps(binding))
    fake["supported_pic_box_anode_frame"]["z_max_m"] = 0.015
    with pytest.raises(PIC2DValidationError):
        fields.brandt_field_map(big, fake)
    assert too_far.grid.node_shape == (76, 701)


@needs_checkpoint
def test_regate_reproduces_the_binding_gates_from_the_checkpoint():
    regated = fields.regate_field()
    binding = fields.load_binding()
    assert regated["scale_matches_binding"] and regated["all_passed"]
    for key in ("G2_interior_nulls", "G3_exit_null", "G4_exit_point", "G5_axis_maximum"):
        assert regated[key]["passed"] == binding["gates"][key]["passed"], key
    assert np.allclose(regated["G2_interior_nulls"]["interior_nulls_m"], binding["gates"]["G2_interior_nulls"]["interior_nulls_m"], rtol=1e-9)


@needs_checkpoint
def test_composed_protocol_with_field_keeps_dt_and_the_preflight_passes_the_launch_set():
    p, mapping, fm = protocol.compose_run_protocol("base", "20um")
    assert p["numerics"]["dt_s"] == 0.7e-12 and p["numerics"]["dt_policy"]["omega_ce"]["omega_ce_dt"] < 0.1
    assert mapping.grid.node_shape == fm.grid.node_shape == (76, 701)
    report = preflight.preflight_all((("base", "20um"),), log=lambda text: None)
    assert report["all_passed"] and report["launch_set_passed"] and report["options"][0]["option"] == "channel-20um"
    gates = report["options"][0]["gates"]
    assert all(g["passed"] for g in gates.values()), {k: v.get("error") for k, v in gates.items() if not v["passed"]}
    assert gates["field_map"]["hard_pi_met"] and not gates["field_map"]["soft_margin_2p5_met"] and gates["protocol"]["static_neutrals"]
    assert gates["grid"]["worst_snap_in_cells"] == 0.0 and gates["comparison_spec"]["plume_only_rows"] == ["plume_peak_angle_deg", "electron_energy_near_exit_cusp_ev"]
    if preflight.preflight_path().is_file():
        on_disk = json.loads(preflight.preflight_path().read_text(encoding="utf-8"))
        assert on_disk["all_passed"] and {r["option"] for r in on_disk["options"]} >= {"channel-20um", "channel-20um-bohm-0.4"}


# -- run guards -------------------------------------------------------------------------------------------------------------------


def test_run_and_launch_guards(capsys):
    assert ev_run.main(["launch"]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert ev_run.main(["run"]) == 2
    assert "allow-launch" in capsys.readouterr().err


def test_sealed_protocols_equal_recomposition_when_present():
    for variant in ("base", "bohm-0.4"):
        path = protocol.composed_protocol_path(variant, "20um")
        if not path.is_file() or not _binding_is_real():
            pytest.skip("draft protocols not composed in this checkout or checkpoint not present")
        sealed = json.loads(path.read_bytes())
        recomposed, _, _ = protocol.compose_run_protocol(variant, "20um")
        for key in ("case", "operating_point", "geometry", "stopping_rule", "variant"):
            assert json.dumps(sealed[key], sort_keys=True) == json.dumps(recomposed[key], sort_keys=True), (variant, key)
        assert sealed["numerics"]["dt_s"] == recomposed["numerics"]["dt_s"]
