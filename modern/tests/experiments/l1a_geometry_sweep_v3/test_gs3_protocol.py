"""Protocol integrity, plans, code binding and design-space declarations of sweep v3."""

from __future__ import annotations

import math
import re

import pytest

from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file

from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.l1a_geometry_sweep_v3 import descriptors as DS
from experiments.l1a_geometry_sweep_v3 import designs as D
from experiments.l1a_geometry_sweep_v3 import experiment as E

HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


def test_labels_and_claim_boundary(value: dict) -> None:
    assert value["experiment_id"] == "l1a-geometry-sweep-v3"
    assert value["classification"] == E.CLASSIFICATION == sweep.CLASSIFICATION
    assert value["catalogue"]["label"] == E.TOPOLOGY_LABEL == "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
    boundary = value["claim_boundary"]
    for key in ("forbid_plasma_performance_publication", "forbid_mirror_probability_publication", "mirror_ratios_are_field_descriptors_not_probabilities", "shakedown_outcomes_are_not_evidence"):
        assert boundary[key] is True
    assert boundary["l1b_p2_confirmation"]["status"] == "queued_not_run"
    assert "r_w / L > 0.5" in boundary["l1b_p2_confirmation"]["statement"]
    assert "beb4772c" in value["purpose"]


def test_v3_box_contains_the_sweep_v2_box(value: dict) -> None:
    v3 = {item["name"]: item for item in value["sampling"]["variables"]}
    assert set(v3) == {variable.name for variable in sweep.VARIABLES}
    for variable in sweep.VARIABLES:
        assert v3[variable.name]["lower"] <= variable.lower and v3[variable.name]["upper"] >= variable.upper, variable.name
    # the extension is exactly where the protocol says it is
    assert v3["chamber_outer_radius_m"]["upper"] == 0.0042 and v3["stage_pitch_m"]["lower"] == 0.0034
    assert v3["radial_clearance_m"]["upper"] == 0.0016 and v3["magnet_radial_thickness_m"]["upper"] == 0.005
    coverage = value["sampling"]["regime_coverage"]
    low = v3["chamber_outer_radius_m"]["lower"] / v3["stage_pitch_m"]["upper"]
    high = v3["chamber_outer_radius_m"]["upper"] / v3["stage_pitch_m"]["lower"]
    assert coverage["wall_radius_over_pitch"] == [round(low, 4), round(high, 4)]
    assert coverage["x_w"] == [round(math.pi * low, 4), round(math.pi * high, 4)]
    assert high > 1.2 > 0.62 > low


def test_hypothesis_and_threshold_numbers_are_declared(value: dict) -> None:
    prediction = value["descriptors_v3"]["ppm_prediction"]
    assert "1.937318" in prediction["i1_threshold"] and "0.616668" in prediction["i1_threshold"]
    assert abs(DS.X_STAR_HEMP_LIKE - 1.937318) < 1.0e-6 and abs(DS.RW_OVER_L_STAR_HEMP_LIKE - 0.616668) < 1.0e-6
    hypothesis = value["descriptors_v3"]["hypothesis"]
    assert hypothesis["agreement_band_relative"] == 0.25
    assert "[0.80, 1.00]" in hypothesis["statement"] and "0.85" in hypothesis["statement"]
    assert "REPORTED" in hypothesis["statement"]
    assert value["descriptors_v3"]["hemp_like_rule"].startswith("a design is HEMP-like iff")
    assert "IEPC-2007-110" in value["descriptors_v3"]["koch_rho"]["citation"]


def test_field_declaration_equals_sweep_v2(value: dict) -> None:
    assert value["field"]["domain"] == sweep.PROTOCOL["field"]["domain"]
    assert value["field"]["solver"] == sweep.PROTOCOL["field"]["solver"]
    assert value["field"]["preview"]["radial_smear_thickness_m"] == sweep.SMEAR_THICKNESS_M
    assert value["field"]["refinement"] == 2
    D.solver_config(value)
    for key in ("exit_minimum_length_m", "radial_tolerance_m", "axial_tolerance_m", "minimum_thickness_m", "minimum_clearance_m", "thermal_clearance_m"):
        assert value["geometry"][key] == sweep.PROTOCOL["geometry"][key]


def test_definition_v3_import_matches_the_v3_1_protocol(value: dict) -> None:
    policy = E.policy_from(value)
    assert policy.axis_root_bracket_tolerance_m == 1.0e-12 and policy.path_max_points == 400
    cts = strict_json_file(E.CTS_PROTOCOL_PATH)["definition_v3"]
    for key in ("stability_tolerance_m", "held_out_tolerance_m", "minimum_certificate_dense_to_bound_ratio"):
        assert value["definition_v3_import"][key] == cts[key]


def test_design_sets_and_plans(value: dict) -> None:
    sets = value["design_sets"]
    assert set(sets) == set(D.DESIGN_SETS) == {"sobol_v3", "sweep_v2"}
    assert sets["sobol_v3"]["design_count"] == 128 and sets["sweep_v2"]["design_count"] == 96
    plan = E.evidentiary_plan(value)
    assert plan.kind == "evidentiary" and plan.binding_gates and len(plan.design_keys) == 224
    assert plan.design_keys[0].startswith("sobol_v3:l1a-gs-v3-000-") and plan.design_keys[128].startswith("sweep_v2:l1a-gs-v2-000-")
    shakedown = E.shakedown_plan(value)
    assert shakedown.kind == "shakedown" and not shakedown.binding_gates and set(shakedown.design_keys) <= set(plan.design_keys)
    assert len(shakedown.design_keys) == 9
    assert E.replay_keys(value, shakedown) == (shakedown.design_keys[0],)
    assert set(E.replay_keys(value, plan)) == {"sobol_v3:l1a-gs-v3-000-78dcc2bb4c", "sobol_v3:l1a-gs-v3-106-ccec1c8b2f", "sweep_v2:l1a-gs-v2-000-48d2ccedd5"}
    assert E.worker_count(value) <= 6


def test_gates_declare_every_binding_check_used_by_the_assessment(value: dict) -> None:
    binding = value["gates"]["binding_integrity"]
    assert set(binding) == {"all_declared_designs_resolved", "determinism_replay", "hash_bindings", *E.GATE_NAMES}
    assert set(E.V2_GATES_APPLIED) == {gate["gate_id"] for gate in sweep.TERMINAL_GATES} - {E.V2_GATE_NOT_APPLICABLE}
    for gate in E.v2_gate_definitions():
        assert gate == next(item for item in sweep.TERMINAL_GATES if item["gate_id"] == gate["gate_id"])
    assert "hypothesis tests H1 and H2" in value["gates"]["reported_not_binding"]


def test_code_and_dependency_hashes_are_lf_bound(value: dict) -> None:
    binding = E.source_binding_report(value)
    for key in ("experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256", "protocol_semantic_sha256"):
        assert HEX64.match(binding[key]), key
    assert binding["protocol_semantic_sha256"] == semantic_sha256(value)
    for name in E.EXPERIMENT_CODE_FILES:
        assert b"\r" not in (E.EXPERIMENT / name).read_bytes(), name
    assert b"\r" not in E.PROTOCOL_PATH.read_bytes()
    files = binding["dependency_source_files"]
    assert "experiments/cusp_topology_search_v3_1/topology.py" in files
    assert "experiments/cusp_topology_search_v3_1/protocol.json" in files
    assert "experiments/orbit_wall_loss_geometry_screening_v1/designs.py" in files
    assert "experiments/l1a_geometry_sweep_v2/experiment.py" in binding["field_pipeline_source_files"]
    assert set(binding["sealed_sources"]) == {"sweep_v2"}
    assert binding["sealed_sources"]["sweep_v2"]["preregistration_commit"].startswith("092f5fae")


def test_verify_shakedown_record_rejects_tampering(value: dict) -> None:
    if not E.SHAKEDOWN_PATH.is_file():
        pytest.skip("shakedown not recorded yet")
    record = strict_json_file(E.SHAKEDOWN_PATH)
    checks = E.verify_shakedown_record(value, record)
    assert all(checks.values())
    tampered = dict(record)
    tampered["passed"] = False
    with pytest.raises(ValueError, match="passed"):
        E.verify_shakedown_record(value, tampered)
    tampered = dict(record)
    tampered["protocol_semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol_semantic_sha256_current"):
        E.verify_shakedown_record(value, tampered)
