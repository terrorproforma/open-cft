"""Protocol integrity, plans, code binding, sealed-source binding and real-input preflight (no P2 solve)."""

from __future__ import annotations

import re

import pytest

from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file
from cft_revival.fem_reference import graded_mesh_geometry
from experiments.l1a_geometry_sweep_v3 import experiment as v3_experiment
from experiments.l1b_hemp_confirmation_v1_1 import designs as D
from experiments.l1b_hemp_confirmation_v1_1 import experiment as E
from experiments.l1b_hemp_confirmation_v1_1 import p2_fields as P

HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


def test_labels_and_claim_boundary(value: dict) -> None:
    assert value["experiment_id"] == "l1b-hemp-confirmation-v1-1"
    assert value["classification"] == E.CLASSIFICATION and E.TOPOLOGY_LABEL.startswith("SCREENING_P2_MATERIAL")
    boundary = value["claim_boundary"]
    for key in ("forbid_plasma_performance_publication", "forbid_mirror_probability_publication", "mirror_ratios_are_field_descriptors_not_probabilities", "shakedown_outcomes_are_not_evidence"):
        assert boundary[key] is True
    assert "NOT in scope" in boundary["paper_admission"]
    assert "2cfe8223" in value["purpose"] and "beb4772c" in value["purpose"]
    assert value["p2"]["resources"]["cpu_only"] is True and value["p2"]["resources"]["worker_pool_size"] == 1
    assert value["p2"]["resources"]["maximum_p2_dofs"] <= 1_500_000 and value["p2"]["resources"]["ram_budget_fraction_of_free_at_start"] == 0.4


def test_design_ids_are_exactly_the_sealed_hemp_like_set(value: dict) -> None:
    entries = D.hemp_like_catalogue_entries()
    assert len(entries) == 15 == D.v3_catalogue_sealed()["hemp_like_design_count"]
    assert [entry["design_id"] for entry in entries] == value["design_sets"][D.SET_HEMP]["design_ids"]
    assert all(entry["hemp_like_all_cusps"] and entry["set_id"] == "sobol_v3" and entry["stable"] for entry in entries)
    specs = E.all_specs(value)
    assert len(specs) == 15 and [spec.design_id for spec in specs] == value["design_sets"][D.SET_HEMP]["design_ids"]
    assert sum(spec.representative for spec in specs) == 4
    tampered = {**value, "design_sets": {D.SET_HEMP: {**value["design_sets"][D.SET_HEMP], "design_ids": list(reversed(value["design_sets"][D.SET_HEMP]["design_ids"]))}}}
    with pytest.raises(ValueError, match="order-sensitive"):
        D.design_specs(tampered)


def test_definition_v3_import_matches_the_v3_1_protocol(value: dict) -> None:
    policy = E.policy_from(value)
    assert policy.axis_root_bracket_tolerance_m == 1.0e-12 and policy.boundary_ambiguity_tolerance_m == 2.5e-4
    cts = strict_json_file(E.CTS_PROTOCOL_PATH)["definition_v3"]
    for key in ("stability_tolerance_m", "minimum_certificate_dense_to_bound_ratio"):
        assert value["definition_v3_import"][key] == cts[key]
    v3 = v3_experiment.protocol()
    assert value["definition_v3_import"]["numerical_parameters"] == v3["definition_v3_import"]["numerical_parameters"]
    assert value["comparison"]["l1a_dz_m"] == pytest.approx((v3["field"]["domain"]["z_max_m"] - v3["field"]["domain"]["z_min_m"]) / v3["field"]["domain"]["axial_intervals"])


def test_p2_controls_match_the_fem_reference_qualification(value: dict) -> None:
    p2 = value["p2"]
    assert p2["solver"]["relative_tolerance"] == 2.0e-10 and p2["solver"]["max_iterations"] == 16000
    assert p2["mesh"]["bore_elements"] == 8 and p2["mesh"]["feature_elements"] == 3 and p2["mesh"]["padding_factor"] == 0.5
    assert p2["mesh"]["reject_below_angle_deg"] == 5.0 and "978c71be" in p2["mesh"]["angle_gate_disclosure"]
    assert value["predecessor"]["terminal_state"] == "development_rejection" and value["predecessor"]["preregistration_commit"].startswith("b9449ee5")
    assert p2["adaptivity"] == {"levels": 2, "dorfler_theta": 0.5, "maximum_adjacent_size_growth": 1.3, "statement": p2["adaptivity"]["statement"]}
    assert p2["sampling"]["radial_intervals"] == 32 and p2["sampling"]["refinement"] == 2
    assert p2["materials"]["soft_iron_relative_permeability"] == 4000.0 and p2["materials"]["magnet_recoil_relative_permeability"] == 1.05


def test_gates_declare_every_binding_check_used_by_the_assessment(value: dict) -> None:
    binding = value["gates"]["binding_integrity"]
    assert set(binding) == {"all_declared_designs_resolved", "determinism_replay", "hash_bindings", *E.GATE_NAMES}
    assert "mesh_preflight covers every declared design" in value["shakedown"]["prepare_requires"][-1]
    confirmation = value["gates"]["confirmation"]
    assert confirmation["cusp_count_unchanged"]["pass_threshold"] == 1.0 and confirmation["cusp_count_unchanged"]["comparator"] == ">="
    assert confirmation["cusp_position_shift"]["pass_threshold"] == 1.0 and confirmation["cusp_position_shift"]["comparator"] == "<="
    assert confirmation["hemp_like_preserved"]["pass_threshold"] is None
    assert "CONFIRMED iff (b) and (c) pass" in confirmation["verdict_rule"]
    assert set(E.VERDICTS) == {"CONFIRMED", "PARTIALLY_CONFIRMED", "DISCONFIRMED"}


def test_plans_and_replays(value: dict) -> None:
    plan = E.evidentiary_plan(value)
    assert plan.kind == "evidentiary" and plan.binding_gates and len(plan.design_keys) == 15
    assert plan.design_keys[0] == "hemp_like_v3:l1a-gs-v3-000-78dcc2bb4c"
    shakedown = E.shakedown_plan(value)
    assert shakedown.kind == "shakedown" and not shakedown.binding_gates and len(shakedown.design_keys) == 5
    assert set(shakedown.design_keys) <= set(plan.design_keys)
    assert {"hemp_like_v3:l1a-gs-v3-028-f012c0bf33", "hemp_like_v3:l1a-gs-v3-048-aabacb3a59"} <= set(shakedown.design_keys)
    assert E.replay_keys(value, shakedown) == (shakedown.design_keys[0],)
    assert E.replay_keys(value, plan) == ("hemp_like_v3:l1a-gs-v3-015-3ce63a512e",)
    assert E.worker_count(value) == 1


def test_code_dependency_and_sealed_source_bindings(value: dict) -> None:
    binding = E.source_binding_report(value)
    for key in ("experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256", "protocol_semantic_sha256"):
        assert HEX64.match(binding[key]), key
    assert binding["protocol_semantic_sha256"] == semantic_sha256(value)
    for name in E.EXPERIMENT_CODE_FILES:
        assert b"\r" not in (E.EXPERIMENT / name).read_bytes(), name
    assert b"\r" not in E.PROTOCOL_PATH.read_bytes()
    files = binding["dependency_source_files"]
    for expected in ("experiments/cusp_topology_search_v3_1/topology.py", "experiments/l1a_geometry_sweep_v3/descriptors.py", "experiments/l1a_geometry_sweep_v3/design-authorities.json", "src/cft_revival/fem_reference/solver.py", "src/cft_revival/fem_reference/mesh.py", "src/cft_revival/fem_reference/adaptivity.py"):
        assert expected in files, expected
    sealed = binding["sealed_sources"]["l1a_geometry_sweep_v3"]
    assert sealed["preregistration_commit"].startswith("1923ef76") and sealed["hemp_like_design_count"] == 15
    manifest = D.v3_manifest()
    assert sealed["terminal_byte_sha256"] == manifest["terminal_byte_sha256"]


def test_sealed_v3_artifacts_are_byte_bound() -> None:
    D.sealed_v3_json("artifacts/campaign-result.json")
    with pytest.raises(ValueError, match="not an artifact"):
        D.sealed_v3_json("artifacts/does-not-exist.json")


@pytest.mark.parametrize("design_id", ["l1a-gs-v3-000-78dcc2bb4c", "l1a-gs-v3-005-0e7f21e31d"])
def test_real_design_rebuild_identity_reference_and_level0_mesh_fit_the_cap(value: dict, design_id: str) -> None:
    case = D.rebuild_case(design_id)
    reference = D.l1a_reference(design_id)
    assert reference["hemp_like_all_cusps"] and reference["wall_cusp_count"] >= 2 and reference["cell_count"] == reference["wall_cusp_count"] + 1
    assert reference["identity"]["geometry_sha256"] == case.geometry_sha256 and reference["identity"]["case_sha256"] == case.case_sha256
    geometry = E.channel_geometry(case)
    assert geometry.to_dict() == reference["geometry"]
    assert all("upstream_wall_max_b_t" in row for row in reference["rho"])
    identity = D.design_identity_without_solving(E.all_specs(value)[0] if design_id.startswith("l1a-gs-v3-000") else E.all_specs(value)[1])
    assert identity["l1a_record_byte_sha256"] == reference["record_byte_sha256"] and identity["design_id"] == design_id
    mesh_declaration = value["p2"]["mesh"]
    problem, mesh = graded_mesh_geometry(case.geometry, bore_elements=mesh_declaration["bore_elements"], feature_elements=mesh_declaration["feature_elements"], padding_factor=mesh_declaration["padding_factor"])
    assert 4 * len(mesh.p2_nodes_rz_m) <= value["p2"]["resources"]["maximum_p2_dofs"], "level-1 red closure would exceed the DOF cap"
    iron = [region for region in problem.regions if region.material_id == "soft-iron-assumed"]
    assert len(iron) == reference["stage_count"] - 1 + 1  # poles between magnets + return yoke
    assert all(abs(1.0 / (region.reluctivity_per_m_h * 4.0e-7 * 3.141592653589793) - 4000.0) < 1.0e-6 for region in iron)
    assert not problem.sheets and sum(1 for region in problem.regions if region.remanence_z_t != 0.0) == reference["stage_count"]
    budget = P.ram_budget(value, free_bytes=4 * 1024**3)
    assert P.allocation_preflight(mesh, budget, phase="level-0")["passed"]
    r_nodes, z_nodes = P.sampling_nodes(value, geometry.wall_radius_m, problem.domain.to_dict())
    margin = E.policy_from(value).axis_window_margin_mesh_factor * max(r_nodes[1] - r_nodes[0], z_nodes[1] - z_nodes[0])
    assert z_nodes[0] + margin <= reference["axis_window_m"][0] and reference["axis_window_m"][1] <= z_nodes[-1] - margin


def test_whole_set_mesh_preflight_passes_the_declared_gate_and_cap(value: dict) -> None:
    """The real preflight v1 lacked: every design's level-0 mesh through the angle gate and the DOF cap (no solve)."""

    budget = P.ram_budget(value, free_bytes=4 * 1024**3)
    report = E.mesh_preflight(value, E.evidentiary_plan(value), budget)
    assert report["design_count"] == 15 and report["all_passed"] and report["failed_designs"] == []
    assert report["minimum_angle_deg"] >= 5.0 and report["max_level1_red_closure_p2_dof_upper_bound"] <= value["p2"]["resources"]["maximum_p2_dofs"]
    rows = {row["design_id"]: row for row in report["designs"]}
    # the two v1 casualties are exactly the designs with elements below the qualification's 10 deg
    assert set(report["designs_with_elements_below_10deg"]) == {"l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59"}
    assert rows["l1a-gs-v3-028-f012c0bf33"]["minimum_angle_deg"] < 10.0 and rows["l1a-gs-v3-048-aabacb3a59"]["minimum_angle_deg"] < 10.0
    assert "dielectric-divergent-exit" in rows["l1a-gs-v3-028-f012c0bf33"]["sliver"]["regions_below_threshold"]
    assert all(row["reject_below_angle_deg"] == 5.0 for row in report["designs"])


def test_verify_shakedown_record_rejects_tampering(value: dict) -> None:
    if not E.SHAKEDOWN_PATH.is_file():
        pytest.skip("shakedown not recorded yet")
    record = strict_json_file(E.SHAKEDOWN_PATH)
    checks = E.verify_shakedown_record(value, record)
    assert all(checks.values()) and "mesh_preflight_covers_every_design_and_passed" in checks
    assert record["mesh_preflight"]["design_count"] == 15 and record["mesh_preflight"]["all_passed"]
    tampered = dict(record)
    tampered["passed"] = False
    with pytest.raises(ValueError, match="passed"):
        E.verify_shakedown_record(value, tampered)
    tampered = dict(record)
    tampered["protocol_semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol_semantic_sha256_current"):
        E.verify_shakedown_record(value, tampered)
    tampered = dict(record)
    tampered["mesh_preflight"] = {**record["mesh_preflight"], "passed_count": 14}
    with pytest.raises(ValueError, match="mesh_preflight"):
        E.verify_shakedown_record(value, tampered)
