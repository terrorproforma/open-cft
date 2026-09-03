"""Design-set binding: every declared design rebuilds with identity against its sealed record."""

from __future__ import annotations

import pytest

from experiments.cusp_topology_search_v3 import experiment as E
from experiments.cusp_topology_search_v3 import fields as F
from experiments.four_cell_topology_search_v2 import experiment as four_cell_v2


@pytest.fixture(scope="module")
def value() -> dict:
    return E.protocol()


@pytest.fixture(scope="module")
def specs(value: dict):
    return F.design_specs(value)


def test_design_specs_are_unique_ordered_and_flag_representatives(specs) -> None:
    keys = [spec.key for spec in specs]
    assert len(set(keys)) == len(keys) == 281
    for set_id in F.DESIGN_SETS:
        ordinals = [spec.ordinal for spec in specs if spec.set_id == set_id]
        assert ordinals == list(range(len(ordinals)))
    assert {spec.design_id for spec in specs if spec.set_id == F.SET_SWEEP and spec.representative} == {
        "l1a-gs-v2-000-48d2ccedd5",
        "l1a-gs-v2-032-570ad83ba6",
        "l1a-gs-v2-065-9e98f08f3b",
        "l1a-gs-v2-068-375d1b1b13",
    }
    assert {spec.design_id for spec in specs if spec.set_id == F.SET_FOUR_CELL and spec.representative} == {"v2-006", "v2-010"}
    assert sum(spec.representative for spec in specs if spec.set_id == F.SET_CHARACTERIZATION) == 7
    assert [spec.design_id for spec in specs if spec.set_id == F.SET_P2] == [F.P2_DESIGN_ID]


def test_every_design_identity_rebuilds_against_its_sealed_record(specs, value: dict) -> None:
    v2_dataset = F.v2_dataset()
    v1_dataset = F.v1_dataset()
    v2_records = {case["candidate_id"]: case for case in v2_dataset["cases"]}
    v1_records = {case["case_id"]: case for case in v1_dataset["cases"]}
    sweep = F.sweep_binding()
    for spec in specs:
        identity = F.design_identity_without_solving(spec, value)
        assert identity["set_id"] == spec.set_id and identity["design_id"] == spec.design_id
        if spec.set_id == F.SET_SWEEP:
            recorded = sweep.cases_by_id[spec.design_id]
            assert identity["case_sha256"] == recorded["case_sha256"]
            assert identity["geometry_sha256"] == recorded["geometry_sha256"]
        elif spec.set_id == F.SET_FOUR_CELL:
            recorded = v2_records[spec.design_id]
            assert identity["geometry_sha256"] == recorded["geometry_sha256"]
            assert identity["source_sha256"] == recorded["source_sha256"]
            assert identity["material_sha256"] == recorded["material_sha256"]
        elif spec.set_id == F.SET_CHARACTERIZATION:
            recorded = v1_records[spec.design_id]
            assert identity["geometry_sha256"] == recorded["geometry_sha256"]
            assert identity["source_semantic_sha256"] == recorded["source_semantic_sha256"]
        else:
            assert set(identity["maps"]) == {"primary", "refined"}


def test_v2_protocol_hash_substitution_is_scoped_and_documented(value: dict) -> None:
    original = four_cell_v2.PROTOCOL_SHA256
    recorded = F.v2_dataset()["protocol_sha256"]
    assert recorded != original  # LF checkout vs the sealed CRLF-era byte hash
    with F._recorded_v2_protocol_hash(recorded):
        assert four_cell_v2.PROTOCOL_SHA256 == recorded
    assert four_cell_v2.PROTOCOL_SHA256 == original
    candidate = next(item for item in four_cell_v2.sample_candidates() if item["candidate_id"] == "v2-000")
    lf_hash = four_cell_v2.build_candidate(candidate).geometry_sha256
    with F._recorded_v2_protocol_hash(recorded):
        sealed_hash = four_cell_v2.build_candidate(candidate).geometry_sha256
    record = next(case for case in F.v2_dataset()["cases"] if case["candidate_id"] == "v2-000")
    assert sealed_hash == record["geometry_sha256"] != lf_hash
    assert "CRLF" in value["design_sets"]["four_cell_v2"]["why"]


def test_sealed_source_binding_reports_every_set() -> None:
    binding = F.sealed_source_binding()
    assert set(binding) == set(F.DESIGN_SETS)
    assert binding["sweep_v2"]["preregistration_commit"] == "092f5fae692ee7d6711e0c7e1c94dac6a345f37c"
    assert binding["four_cell_v2"]["preregistration_commit"] == "d6317910703de91ca6dc25c4d4d855e36cc3b14d"
    assert binding["characterization_v1"]["preregistration_commit"] == "af88470b86fd95882ae7fddc48e2860cbfba1219"
    assert set(binding["p2_divergent_exit"]["maps"]) == {"primary", "refined"}


def test_one_characterization_case_resolves_and_matches_its_sealed_axis_root(value: dict) -> None:
    """Real re-solve of the smallest v1 case (~10 s): identity, stored map, held-out null, stability."""

    spec = F.DesignSpec(F.SET_CHARACTERIZATION, "topology-s02-p0-r0-neg", 0, True)
    resolved = F.resolve_design(spec, value)
    assert resolved.evidence["identity_proven"] and resolved.evidence["stored_representative"]["passed"]
    assert resolved.geometry.wall_radius_m == 0.0065 and len(resolved.geometry.stage_centres_m) == 2
    record = E.characterize_resolved(resolved, value, keep_paths=False)
    channel = [null for null in record["accepted"]["axis_nulls"]["nulls"] if null["zone"] == "channel"]
    sealed = [root for root in resolved.reference["v1_primary_axis_roots"] if root["zone"] == "plasma_channel"]
    assert len(channel) == len(sealed) == 1
    assert abs(channel[0]["z_m"] - sealed[0]["z_m"]) <= value["definition_v3"]["held_out_tolerance_m"]
    assert record["held_out"]["passed"] and record["stability"]["stable"]
    assert record["accepted"]["topology"]["wall_cusp_count"] == 1
