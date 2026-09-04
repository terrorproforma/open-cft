"""PIC design mini-sweep v1: design list, catalogue identity, PIC mapping, field bindings, protocol composition (the draft 50 um
option and the PREREGISTERED channel-33um option on the steady-state v4 template), cost anchors (5090 model + H100 MPS-4),
closure-target extraction, the whole-set preflight and the launch guards.  No GPU, no stepping."""

from __future__ import annotations

import json
from math import pi

import numpy as np
import pytest

from cft_revival.pic2d.models import PIC2DValidationError
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_design_mini_sweep_v1 import (
    closure,
    cost,
    designs,
    fields,
    preflight,
    protocol,
)
from experiments.pic2d_design_mini_sweep_v1 import run as sweep_run

PRIMARY = [d for d in designs.SWEEP_DESIGNS if not d.optional]
OPTIONAL = [d for d in designs.SWEEP_DESIGNS if d.optional]


def _bindings_present() -> bool:
    return all(fields.binding_path(d.design_id).is_file() for d in designs.SWEEP_DESIGNS)


needs_bindings = pytest.mark.skipif(not _bindings_present(), reason="field bindings not produced (run `fields`)")
needs_sealed = pytest.mark.skipif(not (sweep_run.MPS_REPLAY_PATH.is_file() and protocol.PROTOCOLS_DIR.is_dir()),
                                  reason="the channel-33um option is not sealed in this checkout (compose + records missing)")


def _approx_json(a, b, *, rtol: float = 1e-9, path: str = "") -> list[str]:
    """Structural JSON comparison with a relative tolerance on floats (cross-platform: CPU-derived floats differ at ULP level)."""

    problems: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            problems.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for key in set(a) & set(b):
            problems += _approx_json(a[key], b[key], rtol=rtol, path=f"{path}/{key}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} != {len(b)}")
        for index, (x, y) in enumerate(zip(a, b)):
            problems += _approx_json(x, y, rtol=rtol, path=f"{path}[{index}]")
    elif isinstance(a, float) and isinstance(b, (int, float)) or isinstance(b, float) and isinstance(a, (int, float)):
        if not (a == b or abs(float(a) - float(b)) <= rtol * max(abs(float(a)), abs(float(b)))):
            problems.append(f"{path}: {a} != {b}")
    elif a != b:
        problems.append(f"{path}: {a!r} != {b!r}")
    return problems


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


# -- the preregistered channel-33um option --------------------------------------------------------------------------


def test_channel_33um_option_is_the_v4_configuration_with_the_v2_0_3_gates_and_parity_weight():
    reference, mapping = protocol.build_protocol(designs.REFERENCE_DESIGN_ID, "channel", grid="33um")
    v4 = json.loads(protocol.REFINED_CHANNEL_TEMPLATE.read_text(encoding="utf-8"))
    assert reference["status"] == protocol.STATUS_PREREGISTERED and reference["option"] == "channel-33um"
    assert reference["template_protocol"]["path"].endswith("pic2d_cft_steady_state_v4/protocol.json") and reference["template_protocol"]["model_version"] == v4["model_version"]
    # the reference reproduces steady-state v4 exactly: grid, dt, W, seed, operating point
    assert (reference["case"]["radial_cells"], reference["case"]["axial_cells"]) == (90, 720) == (v4["case"]["radial_cells"], v4["case"]["axial_cells"])
    assert mapping.grid.dr_m == pytest.approx(0.002 / 60) and mapping.grid.dz_m == pytest.approx(0.024 / 720)
    assert reference["case"]["macro_weight"] == 26666.7 == v4["case"]["macro_weight"]
    assert reference["numerics"]["dt_s"] == 1.4e-12 == v4["numerics"]["dt_s"] and reference["case"]["seed"] == v4["case"]["seed"] == 20260903
    for key in ("anode_potential_v", "exit_plane_potential_v", "neutral_density_per_m3", "electron_injection_current_a", "electron_injection_temperature_ev", "seed_plasma_density_per_m3"):
        assert reference["operating_point"][key] == v4["operating_point"][key], key
    assert reference["operating_point"]["neutral_inventory"]["feed_atoms_per_s"] == pytest.approx(v4["operating_point"]["neutral_inventory"]["feed_atoms_per_s"], rel=1e-12)
    assert not reference["operating_point"]["neutral_inventory"].get("wall_recycling", False)          # v1.3 closure: no recycling
    # v2.0.3 gates and the plateau preconditions, verbatim from v4
    gate = reference["numerics"]["peak_debye_gate"]
    assert gate["max_cells_per_debye"] == pi and gate["soft_cells_per_debye"] == 2.5 and gate["window_steps"] == 400000 and gate["window_snapshot_steps"] == 40000
    triad = reference["stopping_rule"]["grid_heating_triad"]
    assert triad["residual_window_steps"] == 400000 and triad["windowed_energy_residual_over_electrode_work_max"] == 0.05 and triad["hard_drift_max"] == 0.25
    assert reference["numerics"]["frame_recorder"] == {"cadence_steps": 20000, "precision": "float32"}
    assert reference["stopping_rule"]["min_transit_times"] == 3 and reference["stopping_rule"]["plateau_threshold"] == 0.05
    assert "peak_debye_soft_ok" in reference["stopping_rule"]["plateau"] and "5 142 858" not in reference["stopping_rule"]["plateau"]
    # sweep acceptance replaces the v4 convergence acceptance; the v4 verdict is cited as a caveat
    acceptance = reference["stopping_rule"]["acceptance"]
    assert set(acceptance) == {"declared", "a_plateau", "b_residual_power", "c_closure_targets", "d_verdicts", "e_design_effect", "f_convergence_caveat", "g_design_specific"}
    assert set(acceptance["d_verdicts"]) == {"closure_quotable", "plateau_with_heating", "no_plateau"}
    assert "392129e5" in acceptance["f_convergence_caveat"] and "PENDING" in acceptance["f_convergence_caveat"]
    assert "reference_run" not in reference and "preregistration" not in reference and sum(1 for k in reference if k.startswith("budget")) == 1
    budget = reference["budget_design_mini_sweep"]
    assert budget["platform"] == "h100-mps4" and budget["ms_per_step_projected"] == pytest.approx(8.71, rel=1e-3) and budget["wall_budget_factor"] == 1.5
    assert budget["particles_projected_m"]["total_m"] == pytest.approx(4.5, rel=1e-3) and budget["steps_to_3_transits"] == pytest.approx(3 * 2.4e-6 / 1.4e-12)
    assert reference["stopping_rule"]["wall_budget_seconds"] >= 1.5 * budget["hours_to_3_transits_projected"] * 3600.0
    assert reference["stopping_rule"]["wall_budget_seconds"] % 600.0 == 0.0
    assert reference["execution"]["gpu"]["model"] == "NVIDIA H100 80GB HBM3" and "MPS" in reference["execution"]["gpu"]["concurrency"]
    config = runner.build_config(reference, backend="cpu")
    assert config.macro_weight == 26666.7 and config.peak_debye_gate.window_steps == 400000
    # every primary design: parity weight within 1 % of 6e4 / 2.25, 1.4 ps, budget on the MPS-4 rate, 047's disclosure carried
    for design in PRIMARY:
        composed, m = protocol.build_protocol(design.design_id, "channel", grid="33um")
        assert abs(composed["case"]["macro_weight"] / (60000.0 / 2.25) - 1.0) < 0.01, design.design_id
        assert composed["case"]["macro_weight"] == cost.parity_macro_weight(m) and composed["case"]["macro_weight_policy"].startswith("W = 6e4 x dr dz")
        assert composed["numerics"]["dt_s"] == 1.4e-12 and composed["budget_design_mini_sweep"]["platform"] == "h100-mps4"
        assert abs(m.grid.dr_m - 0.024 / 720) <= 0.02 * 0.024 / 720 and abs(m.grid.dz_m - 0.024 / 720) <= 0.02 * 0.024 / 720
        assert composed["budget_design_mini_sweep"]["particles_projected_m"]["total_m"] <= protocol.MAX_PROJECTED_PARTICLES_M_H100
        runner.build_config(composed, backend="cpu")
        specific = composed["stopping_rule"]["acceptance"]["g_design_specific"]
        assert ("anode-edge" in specific) == (design.design_id == "l1a-gs-v2-047-e3196a8aa5")
    replicate, _ = protocol.build_protocol("l1a-gs-v3-056-effcbc8686", "channel", grid="33um", case="seed-replicate")
    assert replicate["case"]["seed"] == 20260904 and replicate["case"]["id"].endswith("-seed-replicate")
    assert protocol.composed_protocol_path("l1a-gs-v3-056-effcbc8686", case="seed-replicate").name == "l1a-gs-v3-056-effcbc8686-channel-33um-seed-replicate.json"
    with pytest.raises(ValueError):
        protocol.build_protocol(designs.REFERENCE_DESIGN_ID, "channel", grid="33um", case="nope")


def test_admissible_dt_records_platform_stable_floats_and_reduces_only_above_the_margin():
    dt, policy = protocol.admissible_dt(1.4e-12, 0.2913456789123456, 0.2)
    assert dt == 1.4e-12 and policy["max_b_t"] == float(f"{0.2913456789123456:.9g}") and policy["rule"].startswith("template dt kept")
    dt2, policy2 = protocol.admissible_dt(1.5e-12, 0.821, 0.2)
    assert dt2 == pytest.approx(1.3e-12) and policy2["omega_ce_dt"] <= 0.95 * 0.2 and policy2["rule"].startswith("dt reduced")


@needs_sealed
def test_experiment_protocol_document_on_disk_matches_the_generator_and_is_preregistered():
    document = protocol.experiment_protocol_document()
    assert document["status"] == protocol.STATUS_PREREGISTERED and "PREREGISTERED" in document["preregistration"]["state"]
    assert document["preregistration"]["option"] == "channel-33um" and document["preregistration"]["launch_set"] == [d.design_id for d in PRIMARY]
    assert [d["design_id"] for d in document["designs"]] == list(designs.design_ids())
    assert len(document["closure_targets"]) >= 15
    assert set(document["launch_projection"]) == set(document["preregistration"]["launch_set"])
    on_disk = json.loads(protocol.DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    for key in ("status", "designs", "operating_point_policy", "grid_policy", "stopping_rule", "replication_policy", "closure_targets", "cost_table_hours_to_3_transits",
                "recommended_schedule", "refined_grid_schedule", "launch_projection", "h100_mps4_anchor", "domain_options"):
        problems = _approx_json(on_disk[key], document[key])
        assert not problems, (key, problems[:5])
    for key in ("state", "option", "launch_set", "decisions", "convergence_caveat", "records", "sealed_run_protocols"):
        problems = _approx_json(on_disk["preregistration"][key], document["preregistration"][key])
        assert not problems, (key, problems[:5])
    # the records the launch requires are bound by hash in the experiment protocol
    records = on_disk["preregistration"]["records"]
    for name in ("preflight", "shakedown", "mps_replay"):
        assert records.get(name) is not None and len(records[name]["sha256"]) == 64, name
        assert (protocol.REPOSITORY / records[name]["path"]).is_file(), name
    assert set(on_disk["preregistration"]["sealed_run_protocols"]) >= {protocol.composed_protocol_path(d.design_id).relative_to(protocol.REPOSITORY).as_posix() for d in designs.SWEEP_DESIGNS}


@needs_bindings
@needs_sealed
def test_sealed_run_protocols_match_the_recomposition_on_the_design_fields():
    """The launch enforces byte equality on the launch platform; here (any platform) structural equality with a float tolerance."""

    on_disk_document = json.loads(protocol.DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    sealed = on_disk_document["preregistration"]["sealed_run_protocols"]
    assert sealed, "no sealed run protocols recorded"
    for relative, digest in sealed.items():
        path = protocol.REPOSITORY / relative
        assert path.is_file(), relative
        assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == digest, relative
    for design in PRIMARY:
        path = protocol.composed_protocol_path(design.design_id)
        recomposed, _, _ = protocol.compose_run_protocol(design.design_id, "channel", "33um", "base")
        problems = _approx_json(json.loads(path.read_text(encoding="utf-8")), recomposed)
        assert not problems, (design.design_id, problems[:8])
        assert recomposed["numerics"]["dt_s"] == 1.4e-12 and "dt_policy" in recomposed["numerics"]


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
    # the H100 / MPS-4 anchor: the v4 configuration (91 x 721, 4.5 M at parity W) costs exactly the measured 8.71 ms/step per process
    refined = {row["design_id"]: row for row in table[cost.REFINED_CHANNEL_KEY]}
    reference_33 = refined[designs.REFERENCE_DESIGN_ID]
    assert reference_33["nodes"] == [91, 721] and reference_33["macro_weight"] == 26666.7 and reference_33["platform"] == "h100-mps4"
    assert reference_33["particles_projected_m"]["total_m"] == pytest.approx(4.5, rel=1e-3) and reference_33["ms_per_step"] == pytest.approx(8.71, rel=1e-3)
    assert reference_33["ms_per_step_h100_mps4_per_process"] == reference_33["ms_per_step"] and reference_33["ms_per_step_rtx5090_model"] < reference_33["ms_per_step"]
    assert cost.h100_mps4_ms_per_step((91, 721), 4.5) == pytest.approx(8.71) and cost.parity_macro_weight(designs.pic_geometry(built, "channel", target_cell_m=cost.REFINED_CHANNEL_CELL_M)) == 26666.7
    assert cost.parity_macro_weight(designs.pic_geometry(built, "channel")) == 60000.0
    assert cost.projected_particles_m(designs.pic_geometry(built, "channel"), macro_weight=30000.0)["total_m"] == pytest.approx(4.0, rel=1e-3)
    with pytest.raises(ValueError):
        cost.platform_ms_per_step((91, 721), 4.5, "tpu")


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
    assert out["anode_edge_band_m"] == 2.5e-4 and out["anode_edge_electron_wall_current_a"] == 0.0
    wall_e_edge = wall_e.copy()
    wall_e_edge[z_cells <= 2.5e-4] = 4.0e22           # electrons lost at the anode-edge band (the 047 boundary-cusp disclosure) are visible, not cusp-counted
    edge = closure.extract_targets({**maps, "wall_electron_flux_per_m2_s": wall_e_edge}, mapping, cusps, cells, injected_electron_current_a=3.0e-3)
    assert edge["anode_edge_electron_wall_current_a"] > 0.0 and edge["anode_edge_electron_wall_current_a"] == pytest.approx(edge["diffuse_non_cusp_electron_wall_current_a"])
    assert [c["electron_wall_current_a"] for c in edge["cusps"]] == [c["electron_wall_current_a"] for c in out["cusps"]]
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


@needs_bindings
def test_whole_set_preflight_over_every_design_passes_for_the_preregistered_channel_33um_option():
    report = preflight.preflight_all("channel", grid="33um", log=lambda text: None)
    failures = {r["design_id"]: {k: v for k, v in r["gates"].items() if not v["passed"]} for r in report["designs"] if not r["passed"]}
    assert report["all_passed"], failures
    assert report["option"] == "channel-33um" and report["schema_version"].endswith("/1.0.0") and report["platform"]["numpy"]
    for record in report["designs"]:
        gates = record["gates"]
        assert gates["field_map"]["dt_s"] == 1.4e-12 and not gates["field_map"]["dt_admissibility"]["reduced_from_template"], record["design_id"]
        assert gates["field_map"]["dt_admissibility"]["omega_ce_dt_at_composed_dt"] <= 0.2 and len(gates["field_map"]["field_source_sha256"]) == 64
        if record["design_id"] in {d.design_id for d in PRIMARY}:
            assert abs(gates["protocol"]["macro_weight"] / (60000.0 / 2.25) - 1.0) < 0.01 and gates["protocol"]["particles_projected_m"] <= protocol.MAX_PROJECTED_PARTICLES_M_H100
        else:   # the optional four-cusp design projects 13.9 M particles at parity: the 12 M cap scales W (disclosed)
            assert gates["protocol"]["macro_weight"] > 60000.0 / 2.25 and gates["protocol"]["macro_weight_policy"].startswith("W scaled from the parity value")
            assert gates["protocol"]["particles_projected_m"] == pytest.approx(protocol.MAX_PROJECTED_PARTICLES_M_H100, rel=1e-4)
        assert gates["protocol"]["platform"] == "h100-mps4"
        assert gates["protocol"]["gates_v2_0_3"] == {"peak_debye_window_mode": True, "peak_debye_hard": pi, "peak_debye_soft": 2.5, "windowed_residual_power_max": 0.05, "residual_window_steps": 400000}
        assert gates["protocol"]["frame_recorder"] == {"cadence_steps": 20000, "precision": "float32"} and not gates["protocol"]["wall_recycling"]
        assert gates["cathode_connectivity"]["skipped"].startswith("channel-only")
        assert gates["cost"]["platform"] == "h100-mps4" and gates["cost"]["macro_weight"] == gates["protocol"]["macro_weight"]
    reference = next(r for r in report["designs"] if r["design_id"] == designs.REFERENCE_DESIGN_ID)
    assert reference["gates"]["protocol"]["cells"] == [90, 720] and reference["gates"]["protocol"]["macro_weight"] == 26666.7
    assert preflight.preflight_path("channel", "33um").name == "preflight-channel-33um.json"


def test_run_refuses_to_launch_without_the_draft_flag(capsys):
    assert sweep_run.main(["run", "--design", designs.REFERENCE_DESIGN_ID, "--domain", "channel"]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert sweep_run.main(["run", "--design", designs.REFERENCE_DESIGN_ID, "--domain", "channel", "--grid", "33um"]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_launch_refuses_an_unexpected_commit_and_a_non_preregistered_option():
    with pytest.raises(PIC2DValidationError, match="not the preregistration commit"):
        sweep_run.launch(designs.REFERENCE_DESIGN_ID, "channel", "33um", expect_commit="0000000000000000000000000000000000000000")
    with pytest.raises(PIC2DValidationError, match="only the preregistered option|not clean"):
        sweep_run.launch(designs.REFERENCE_DESIGN_ID, "channel", "50um", allow_dirty=False)
    with pytest.raises(PIC2DValidationError, match="only the preregistered option"):
        sweep_run.launch(designs.REFERENCE_DESIGN_ID, "channel", "50um", allow_dirty=True)
    assert sweep_run.results_dir("l1a-gs-v3-056-effcbc8686", "channel", "33um", "seed-replicate").name == "l1a-gs-v3-056-effcbc8686-channel-33um-seed-replicate"
    assert sweep_run.results_dir(designs.REFERENCE_DESIGN_ID, "channel", "33um").name == "divergent-exit-stack-channel-33um"


def _fake_run(root, *, ledger_bump: float = 0.0, electrons_bump: int = 0, t_e_bump: float = 0.0, n_e_bump: float = 0.0):
    root.mkdir(parents=True)
    records = [{"step": 200 * (i + 1), "electrons": 1000 + i + electrons_bump, "currents_a": {"anode_electron": 1e-3 * (i + 1)},
                "ledger": {"interval_residual_j": 1e-9 * (1.0 + ledger_bump), "cumulative": {"field_work_j": 2e-7 * (1.0 + ledger_bump)}},
                "peak_node": {"t_e_peak_ev": 7.0 * (1.0 + t_e_bump)}} for i in range(3)]
    (root / "series.jsonl").write_bytes(b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in records))
    np.savez(root / "maps.npz", n_e_per_m3=np.full((3, 4), 1e17 * (1.0 + n_e_bump)), t_e_ev=np.full((3, 4), 7.0 * (1.0 + t_e_bump)), window_steps=np.array([600]),
             wall_electron_flux_per_m2_s=np.full(4, 1e22), wall_electron_mean_energy_ev=np.full(4, 20.0 * (1.0 + t_e_bump)))
    np.savez(root / "checkpoint-final.npz", positions=np.arange(6.0), cumulative=np.array([1.0 + ledger_bump]))
    np.savez(root / "series.npz", electrons=np.array([1000, 1001, 1002]) + electrons_bump, interval_residual_j=np.array([1e-9 * (1.0 + ledger_bump)] * 3))
    (root / "summary.json").write_text(json.dumps({"final_counts": {"electrons": 1002 + electrons_bump}, "steps_completed": 600, "ms_per_step_this_session": 2.0,
                                                   "window_currents_a": {"discharge_a": 3e-3}}), encoding="utf-8")


def test_replay_comparison_separates_physics_from_float_atomic_diagnostics(tmp_path):
    _fake_run(tmp_path / "a")
    _fake_run(tmp_path / "b")
    same = sweep_run._compare_runs(tmp_path / "a", tmp_path / "b")
    assert same["all_bitwise"] and same["physics_bitwise"] and same["passed"]
    _fake_run(tmp_path / "c", ledger_bump=1e-14, t_e_bump=1e-15)       # round-off in the float-atomic diagnostics only
    diag = sweep_run._compare_runs(tmp_path / "a", tmp_path / "c")
    assert not diag["all_bitwise"] and diag["physics_bitwise"] and diag["diagnostics_within_rtol"] and diag["passed"]
    assert set(diag["series_records"]["differing_keys_max_rel"]) == {"ledger/interval_residual_j", "ledger/cumulative/field_work_j", "peak_node/t_e_peak_ev"}
    assert diag["maps.npz"]["physics_keys_bitwise"] and set(diag["maps.npz"]["diagnostic_differing_keys_max_rel"]) == {"t_e_ev", "wall_electron_mean_energy_ev"}
    assert diag["checkpoint-final.npz"]["physics_keys_bitwise"] and set(diag["checkpoint-final.npz"]["diagnostic_differing_keys_max_rel"]) == {"cumulative"}
    assert diag["series.npz"]["physics_keys_bitwise"]
    _fake_run(tmp_path / "d", electrons_bump=1)                          # a physics difference fails
    phys = sweep_run._compare_runs(tmp_path / "a", tmp_path / "d")
    assert not phys["physics_bitwise"] and not phys["passed"] and "electrons" in phys["series_records"]["physics_differing_keys"]
    _fake_run(tmp_path / "e", n_e_bump=1e-15)                            # a deposited map is physics, however small the difference
    assert not sweep_run._compare_runs(tmp_path / "a", tmp_path / "e")["passed"]
    _fake_run(tmp_path / "f", ledger_bump=1e-3)                           # a diagnostic beyond the tolerance fails too
    assert not sweep_run._compare_runs(tmp_path / "a", tmp_path / "f")["passed"]


def test_shrunk_protocol_keeps_grid_dt_weight_and_gate_thresholds():
    full, _ = protocol.build_protocol(designs.REFERENCE_DESIGN_ID, "channel", grid="33um")
    shrunk = sweep_run.shrunk_protocol(full, "shakedown")
    assert shrunk["numerics"]["checkpoint_every_steps"] == 4000 and shrunk["numerics"]["averaging_window_steps"] == 40000
    assert shrunk["numerics"]["peak_debye_gate"]["window_steps"] == 40000 and shrunk["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] == 40000
    assert shrunk["numerics"]["frame_recorder"]["cadence_steps"] == 2000 and "non_evidentiary" in shrunk["status"]
    for key in ("dt_s", "stability_limits"):
        assert shrunk["numerics"][key] == full["numerics"][key]
    assert shrunk["case"]["macro_weight"] == full["case"]["macro_weight"] and shrunk["numerics"]["peak_debye_gate"]["max_cells_per_debye"] == pi
    runner.build_config(shrunk, backend="cpu")
    assert sweep_run.steady_state_v4_verdict()["status"] in ("available", "pending")


# -- assess: acceptance (b) on the corrected ledger (model v2.0.6) and the v4 reference caveat with both readings --------------------------------

def _fake_assessable_run(tmp_path, *, name: str, recorded_windowed: float = -0.075, stop: str = "plateau_reached_after_min_transit_times",
                         corrected_windowed: float | None = None, series_sha: str = "a" * 64, sidecar_series_sha: str | None = None):
    """summary.json + maps.npz (what assess_run reads) and, optionally, a ledger-corrected.json sidecar with the given corrected reading."""

    from cft_revival.pic2d import artifacts

    results = tmp_path / name
    results.mkdir()
    n = np.zeros((5, 7))
    t = np.zeros((5, 7))
    n[2, 3] = 1e18
    t[2, 3] = 6.0
    artifacts.write_npz(results / "maps.npz", {"n_e_per_m3": n, "t_e_ev": t})
    summary = {
        "stop_reason": stop, "ion_transit_times": 3.1, "steps_completed": 5_000_000, "plateau": {"reached": stop.startswith("plateau")},
        "window_currents_a": {"discharge_a": 2e-3, "exit_ion_beam_a": 1e-3},
        "neutral_inventory": {"trailing_20pct_mean_ionization_rate_per_s": 1.5e16, "propellant_utilisation_trailing": 0.3, "net_utilisation_trailing": 0.2,
                              "trailing_20pct_mean_density_per_m3": 3.7e19},
        "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": recorded_windowed, "windowed_energy_residual_window_complete": True,
                               "energy_residual_over_electrode_work": recorded_windowed},
        "peak_node_debye": {"window": {"cells_per_debye_window_last": 1.7, "trailing_20pct_mean_cells_per_debye_window": 1.7, "soft_ok": True}},
        "sessions": [{}], "git_head": "deadbeef", "protocol_sha256": "1" * 64, "provenance": {"config_sha256": "2" * 64, "runtime": {}},
        "artifacts": {"series_npz_sha256": series_sha},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    if corrected_windowed is not None:
        sidecar = {
            "schema": "cft.pic2d.ledger-corrected/1.0.0", "generated_by": "python -m cft_revival.pic2d.ledger_recompute",
            "inputs": {"series": {"file": "series.npz", "sha256": sidecar_series_sha or series_sha}},
            "end_state_window": {"recorded_ratio": recorded_windowed, "corrected_ratio": corrected_windowed, "window_complete": True,
                                 "omitted_ratio": corrected_windowed - recorded_windowed, "recorded_ratio_matches_summary": True},
            "cumulative": {"corrected_over_electrode": corrected_windowed - 0.002},
            "max_over_complete_windows": {"corrected": {"ratio": corrected_windowed, "step": 5_000_000, "time_s": 7e-6}},
            "threshold_crossings": {"0.05": {"corrected_first_crossing_at_checkpoint": None if corrected_windowed < 0.05 else {"step": 1, "time_s": 1e-6, "ratio": 0.05}}},
            "cross_check_vs_final_counts": {"already_w_scaled": False},
        }
        artifacts.write_canonical_json(results / sweep_run.LEDGER_SIDECAR_NAME, sidecar)
    return results


@needs_bindings
def test_assess_evaluates_b_on_the_corrected_ledger_when_the_sidecar_exists_and_keeps_the_recorded_reading(tmp_path):
    from cft_revival.pic2d import artifacts

    proto, _ = protocol.build_protocol(designs.REFERENCE_DESIGN_ID, "channel", grid="33um")
    quiet = lambda _: None
    # no sidecar: the recorded statistic decides and the record says a sidecar is missing
    plain = sweep_run.assess_run(proto, _fake_assessable_run(tmp_path, name="plain"), log=quiet)
    assert plain["verdict"] == "closure_quotable" and plain["b_residual_power"]["passed"] is True and plain["b_residual_power"]["corrected"] is None
    assert "NO ledger-corrected.json" in plain["b_residual_power"]["basis"] and plain["b_residual_power"]["passed_recorded_statistic"] is True
    assert plain["b_residual_power"]["recorded"]["windowed_residual_over_electrode_work"] == -0.075 and plain["b_residual_power"]["bound"] == 0.02
    # a sidecar whose corrected reading is above the bound flips (b) and the verdict; the recorded reading stays beside it
    heating = sweep_run.assess_run(proto, _fake_assessable_run(tmp_path, name="heating", corrected_windowed=0.03), log=quiet)
    assert heating["verdict"] == "plateau_with_heating" and heating["b_residual_power"]["passed"] is False and heating["closure_targets_quotable"] is False
    assert heating["b_residual_power"]["passed_recorded_statistic"] is True and heating["b_residual_power"]["recorded"]["passed"] is True
    assert heating["b_residual_power"]["corrected"]["passed"] is False and heating["b_residual_power"]["corrected"]["windowed_residual_over_electrode_work"] == 0.03
    assert heating["b_residual_power"]["basis"].startswith("corrected statistic")
    # a sidecar below the bound keeps the pass, on the corrected basis
    clean = sweep_run.assess_run(proto, _fake_assessable_run(tmp_path, name="clean", corrected_windowed=0.009), log=quiet)
    assert clean["verdict"] == "closure_quotable" and clean["b_residual_power"]["passed"] is True and clean["b_residual_power"]["corrected"]["passed"] is True
    assert clean["b_residual_power"]["corrected"]["sha256"] == artifacts.read_canonical_json(tmp_path / "clean" / "ledger-corrected.json.sha256.json")["byte_sha256"]
    written = artifacts.read_canonical_json(tmp_path / "clean" / "assessment.json")
    assert written["b_residual_power"]["corrected"]["windowed_residual_over_electrode_work"] == 0.009
    # a sidecar that describes another series is refused (fail closed)
    with pytest.raises(PIC2DValidationError, match="another series"):
        sweep_run.assess_run(proto, _fake_assessable_run(tmp_path, name="foreign", corrected_windowed=0.009, sidecar_series_sha="f" * 64), log=quiet)
    # no plateau stays no_plateau whatever the ledger says
    stopped = sweep_run.assess_run(proto, _fake_assessable_run(tmp_path, name="stopped", stop="grid_heating_triad_gate_stopped_run", corrected_windowed=0.009), log=quiet)
    assert stopped["verdict"] == "no_plateau"


def test_v4_reference_caveat_carries_both_readings(monkeypatch, tmp_path):
    v4 = sweep_run.steady_state_v4_verdict()
    if v4["status"] != "available" or v4.get("corrected_ledger") is None:
        pytest.skip("the steady-state v4 record with its corrected-ledger re-read is not checked out")
    corrected = v4["corrected_ledger"]
    assert v4["verdict"] == corrected["verdict_recorded"] == "resolution_limited" and corrected["verdict_on_corrected_ledger"] == "refinement_heating"
    assert corrected["binds_recorded_assessment"] is True and len(corrected["sha256"]) == 64
    assert corrected["b_recorded"] == pytest.approx(-0.0767, abs=1e-3) and corrected["b_recorded_passed"] is True
    assert corrected["b_corrected"] == pytest.approx(0.0246, abs=1e-3) and corrected["b_corrected_passed"] is False
    assert "FAILED on the corrected ledger" in corrected["verdict_statement"] and "NOT a clean reference" in corrected["verdict_statement"]
    # the assessment record carries the recorded statement and the corrected-ledger statement side by side
    if _bindings_present():
        proto, _ = protocol.build_protocol(designs.REFERENCE_DESIGN_ID, "channel", grid="33um")
        record = sweep_run.assess_run(proto, _fake_assessable_run(tmp_path, name="ref", corrected_windowed=0.009), log=lambda _: None)
        assert record["steady_state_v4_verdict"]["corrected_ledger"]["verdict_on_corrected_ledger"] == "refinement_heating"
        assert record["convergence_statement"].startswith("the 33 um values of this design are the resolved numbers but carry NO grid band")
        statement = record["convergence_statement_corrected_ledger"]
        assert "+2.46 % (FAIL; recorded -7.67 % pass)" in statement and "'refinement_heating' beside the recorded 'resolution_limited'" in statement
        assert "NOT a clean reference" in statement and "the reference grid is not certified" in statement
    # without the re-read file the recorded reading is flagged as the pre-v2.0.6 statistic
    monkeypatch.setattr(sweep_run, "STEADY_STATE_V4_CORRECTED_LEDGER", tmp_path / "absent.json")
    pending = sweep_run.steady_state_v4_verdict()
    assert pending["corrected_ledger"] is None and "not present in this checkout" in pending["corrected_ledger_note"]
