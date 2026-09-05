"""Protocol integrity, plans, code binding and design-set declarations of cusp topology search v3."""

from __future__ import annotations

import re

import pytest

from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file
from cft_revival.provenance import FrozenCommitError

from experiments.cusp_topology_search_v3_1 import experiment as E
from experiments.cusp_topology_search_v3_1 import fields as F
from experiments.cusp_topology_search_v3_1 import frozen_contract as FC
from experiments.cusp_topology_search_v3_1.topology import TopologyPolicy

HEX64 = re.compile(r"^[0-9a-f]{64}$")
DOI = re.compile(r"^10\.\d{4,9}/\S+$")
# the commit the immutable execution lock names (preregistration = execution commit of v3.1)
FROZEN_COMMIT = "1600cfd3b102980eeba4b070930667d232a1105c"


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


def test_protocol_labels_and_boundary(value: dict) -> None:
    assert value["experiment_id"] == "cusp-topology-search-v3.1"
    assert value["classification"] == E.CLASSIFICATION == "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
    assert value["p2_row_classification"] == E.P2_CLASSIFICATION == "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
    boundary = value["claim_boundary"]
    assert boundary["forbid_plasma_performance_publication"] is True
    assert boundary["forbid_mirror_probability_publication"] is True
    assert boundary["mirror_ratios_are_field_descriptors_not_probabilities"] is True
    assert boundary["shakedown_outcomes_are_not_evidence"] is True
    assert "Section 8" in value["relation_to_prior_nulls"]
    assert "assessment_rejection" in value["relation_to_v3"]


def test_v3_lineage_is_disclosed_and_the_held_out_reference_is_defined_by_member_method(value: dict) -> None:
    v3 = value["prior_campaign_disclosure"]["v3"]
    assert v3["preregistration_commit"] == "691599340355818ff64d3834d45110768a751589"
    assert v3["result_commit"] == "8cbcdbe6ede6c55156f300f82d9c85133f06c0dd"
    assert v3["terminal_state"] == "assessment_rejection" and v3["failing_gate"] == "held_out_correspondence"
    assert v3["failing_design_count"] == 14
    assert "r_m == 0.0" in v3["root_cause"]
    held_out = value["definition_v3"]["stability"]["held_out_checks"]
    assert "axis_sign_change or axis_grid" in held_out and "NOT a root whose clustered centroid" in held_out
    assert F.V1_AXIS_DETECTION_METHODS == ("axis_sign_change", "axis_grid")
    assert "topology-s05-p0-r0-neg" in value["shakedown"]["designs"]["characterization_v1"]


def test_definition_cites_the_literature_with_dois(value: dict) -> None:
    basis = value["definition_v3"]["literature_basis"]
    keys = {item["key"] for item in basis}
    assert {"gildea2012", "kornfeld2007", "koch2011", "lewerentz2023", "parnell1996", "haynes2010", "murphy2015"} <= keys
    for item in basis:
        assert item["citation"] and item["used_for"]
        if "doi" in item:
            assert DOI.match(item["doi"]), item["doi"]
        else:
            assert item["locator"].startswith("http")
    assert any(item.get("doi") == "10.3390/app13063491" for item in basis)  # Lewerentz & Schneider 2023
    definition = value["definition_v3"]
    for block in ("axis_null", "separatrix", "stability"):
        assert definition[block]["statement"]
    for key in ("cusp", "cell", "mirror_descriptors", "boundary_ambiguity"):
        assert definition["wall_cusp_and_cell"][key]
    assert definition["wall_cusp_and_cell"]["legacy_target"].startswith("four_wall_cusps")
    assert "zero at the nulls by definition" in definition["wall_cusp_and_cell"]["mirror_descriptors"]


def test_numerical_parameters_match_the_policy_dataclass(value: dict) -> None:
    declaration = value["definition_v3"]["numerical_parameters"]
    assert set(declaration) == set(TopologyPolicy.__dataclass_fields__)
    policy = TopologyPolicy.from_protocol(declaration)
    assert policy.axis_root_bracket_tolerance_m == 1.0e-12
    assert value["definition_v3"]["stability_tolerance_m"] == 2.5e-4
    assert value["definition_v3"]["held_out_tolerance_m"] == 2.5e-4


def test_design_sets_are_declared_and_sum_to_281(value: dict) -> None:
    sets = value["design_sets"]
    assert set(sets) == set(F.DESIGN_SETS)
    assert all(sets[set_id]["included"] for set_id in F.DESIGN_SETS)
    assert {set_id: sets[set_id]["design_count"] for set_id in F.DESIGN_SETS} == {
        "sweep_v2": 96,
        "four_cell_v2": 128,
        "characterization_v1": 56,
        "p2_divergent_exit": 1,
    }
    assert sum(sets[set_id]["design_count"] for set_id in F.DESIGN_SETS) == 281
    for set_id in F.DESIGN_SETS:
        assert sets[set_id]["why"]
    assert sets["four_cell_v2"]["representative_ids"] == ["v2-006", "v2-010"]
    references = sets["p2_divergent_exit"]["consistency_references"]
    assert references["topology_dashboard_wall_abs_br_maxima_m"] == [0.00605, 0.01205, 0.01815]
    assert references["pic_axis_null_planes_m"] == [0.006, 0.012, 0.01795]
    assert "not a gate" in references["role"]


def test_gates_declare_every_binding_check_used_by_the_assessment(value: dict) -> None:
    binding = value["gates"]["binding_integrity"]
    assert set(binding) == {
        "all_declared_designs_resolved",
        "every_null_converged",
        "every_trace_terminates_cleanly",
        "every_wall_trace_flux_consistent",
        "refinement_stability",
        "held_out_correspondence",
        "determinism_replay",
        "hash_bindings",
    }
    assert "four_wall_cusp_fraction" in value["gates"]["reported_not_binding"]


def test_plans_cover_every_declared_design_and_the_shakedown_subset(value: dict) -> None:
    plan = E.evidentiary_plan(value)
    assert plan.kind == "evidentiary" and plan.binding_gates and len(plan.design_keys) == 281
    counts: dict[str, int] = {}
    for key in plan.design_keys:
        counts[key.split(":")[0]] = counts.get(key.split(":")[0], 0) + 1
    assert counts == {"sweep_v2": 96, "four_cell_v2": 128, "characterization_v1": 56, "p2_divergent_exit": 1}
    shakedown = E.shakedown_plan(value)
    assert shakedown.kind == "shakedown" and not shakedown.binding_gates
    assert set(shakedown.design_keys) <= set(plan.design_keys)
    assert len(shakedown.design_keys) == 9
    assert E.replay_keys(value, plan) == (
        "sweep_v2:l1a-gs-v2-000-48d2ccedd5",
        "four_cell_v2:v2-006",
        "characterization_v1:topology-s04-p0-r0-neg",
        "p2_divergent_exit:divergent-exit-stack",
    )
    assert E.replay_keys(value, shakedown) == (shakedown.design_keys[0],)
    specs = E.specs_for_plan(value, plan)
    assert sum(spec.representative for spec in specs) == 4 + 2 + 7 + 1


def test_code_and_dependency_hashes_are_lf_bound(value: dict) -> None:
    binding = E.source_binding_report(value)
    for key in ("experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256", "protocol_semantic_sha256"):
        assert HEX64.match(binding[key]), key
    assert binding["protocol_semantic_sha256"] == semantic_sha256(value)
    assert set(binding["experiment_code_files"]) == set(E.EXPERIMENT_CODE_FILES)
    for name in E.EXPERIMENT_CODE_FILES:
        assert b"\r" not in (E.EXPERIMENT / name).read_bytes(), name
    assert b"\r" not in E.PROTOCOL_PATH.read_bytes()
    files = binding["dependency_source_files"]
    assert "experiments/cft_topology_characterization_v1/experiment.py" in files
    assert "experiments/four_cell_topology_search_v2/experiment.py" in files
    assert "experiments/cft_orbit_wall_loss_v4/adapter.py" in files
    assert "experiments/orbit_wall_loss_geometry_screening_v1/designs.py" in files
    assert any(item.startswith("src/cft_revival/coupling/") for item in files)
    assert "src/cft_revival/orbit_mc/fields.py" in files
    sealed = binding["sealed_sources"]
    assert set(sealed) == set(F.DESIGN_SETS)
    assert sealed["four_cell_v2"]["recorded_protocol_sha256"] != sealed["four_cell_v2"]["lf_protocol_sha256"]


def _recorded_shakedown() -> dict:
    """The frozen shakedown record, or skip while the lifecycle has not reached it."""

    if not E.SHAKEDOWN_PATH.is_file():
        pytest.skip("shakedown not recorded yet")
    if not FC.EXECUTION_LOCK_PATH.is_file():
        pytest.skip("not executed yet: the live-tree gate E.verify_shakedown_record applies until then")
    return strict_json_file(E.SHAKEDOWN_PATH)


def test_shakedown_record_binds_to_the_frozen_commit_and_reports_live_drift(value: dict) -> None:
    """The sealed digests are evidence about the EXECUTION commit, not about today's worktree.

    (a) every sealed digest must recompute from that commit's blobs (the seal was honest);
    (b) the live tree's digest is RECORDED with a drift flag, never asserted equal - shared
    packages legitimately move on after a campaign (experiment_runtime at bb756418).
    """

    record = _recorded_shakedown()
    report = FC.verify_recorded_shakedown(value, record)
    assert all(report["checks"].values()), report["checks"]
    assert report["frozen_commit"] == FROZEN_COMMIT
    for key in FC.SEALED_HASH_KEYS:
        scope = report["scopes"][key]
        assert scope["commit"] == FROZEN_COMMIT and scope["sealed_sha256"] == record[key], key
        assert scope["sealed_matches_frozen_commit"], key
        assert scope["file_count"] >= (6 if key == "experiment_code_sha256" else 30), key
        live = scope["live"]
        assert HEX64.match(live["sha256"]) and live["sha256"] == report["live_tree"][f"{key}_current"], key
        assert live["drift"] == (live["sha256"] != record[key]) == bool(live["added"] or live["removed"] or live["changed"]), key
    assert report["live_tree"]["drift"] is any(report["scopes"][key]["live"]["drift"] for key in FC.SEALED_HASH_KEYS)
    assert report["live_tree"]["drifted"] == sorted(key for key in FC.SEALED_HASH_KEYS if report["scopes"][key]["live"]["drift"])
    # the pre-execution (strict) mode is the only one under which drift is a failure
    if report["live_tree"]["drift"]:
        with pytest.raises(ValueError, match="_current"):
            FC.verify_recorded_shakedown(value, record, strict_live_tree=True)
    else:
        assert FC.verify_recorded_shakedown(value, record, strict_live_tree=True)["checks"]


def test_verify_shakedown_record_rejects_tampering(value: dict) -> None:
    record = _recorded_shakedown()
    for field, bogus, check in (
        ("passed", False, "passed"),
        ("protocol_semantic_sha256", "0" * 64, "protocol_semantic_sha256_current"),
        ("experiment_code_sha256", "0" * 64, "experiment_code_sha256_frozen"),
        ("dependency_source_sha256", "0" * 64, "dependency_source_sha256_frozen"),
        ("field_pipeline_source_sha256", "0" * 64, "field_pipeline_source_sha256_frozen"),
        ("dependency_source_files", record["dependency_source_files"][:-1], "bundle_dependency_inventory_equals_record"),
        ("shakedown_plan", {**record["shakedown_plan"], "design_keys": record["shakedown_plan"]["design_keys"][:-1]}, "plan_matches_protocol"),
    ):
        with pytest.raises(ValueError, match=check):
            FC.verify_recorded_shakedown(value, {**record, field: bogus})
    # a record naming a commit this repository does not hold fails closed rather than passing vacuously
    with pytest.raises(FrozenCommitError):
        FC.verify_recorded_shakedown(value, record, commit="0" * 40)
    # the sealed pre-execution gate keeps refusing a tampered record as well
    with pytest.raises(ValueError, match="passed"):
        E.verify_shakedown_record(value, {**record, "passed": False})
