from __future__ import annotations

from collections import Counter
import hashlib

import pytest

from cft_revival.experiment_runtime import canonical_bytes
from cft_revival.experiment_runtime.canonical import (
    CanonicalizationError,
    strict_json_file,
)

from experiments.cft_orbit_wall_loss_v3.experiment import (
    CASE_AUTHORITIES_PATH,
    EXPERIMENT,
    batch_records,
    build_all_case_authorities,
    build_case_launches,
    case_matrix,
    load_runtime_launch_payload,
    manufactured_gate_report,
    production_synthetic_preflight,
    protocol,
    runtime_launch_payload,
)


def test_only_qualified_divergent_exit_design_is_authorized() -> None:
    value = protocol()
    assert value["authority"]["design_id"] == "divergent-exit-stack"
    assert value["authority"]["required_qualification"] == "NUMERICAL_P2_QUALIFIED"
    assert set(value["authority"]["excluded_designs"]) == {
        "historical-envelope-baseline",
        "compact-high-gradient-stack",
    }
    assert value["classification"].endswith("not_pic")
    assert value["publication_boundary"]["hardware_or_experimental_validation"] is False


def test_nine_cases_have_exact_prefixed_equal_weight_launch_authorities() -> None:
    value = protocol()
    all_ids: set[str] = set()
    all_seeds: set[int] = set()
    for role, timestep, campaign_id in case_matrix(value):
        launches = build_case_launches(value, role, timestep)
        assert len(launches) == 512
        assert all(item.launch_id.startswith(campaign_id + ":") for item in launches)
        assert len({item.launch_id for item in launches}) == 512
        assert len({item.seed_id for item in launches}) == 512
        assert {item.parallel_direction for item in launches} == {-1, 1}
        strata = Counter(
            (
                item.flux_surface_id.split("-r", 1)[0],
                item.kinetic_energy_ev,
                item.pitch_angle_rad,
                item.parallel_direction,
            )
            for item in launches
        )
        assert len(strata) == 32
        assert set(strata.values()) == {16}
        batches = batch_records(value, launches)
        assert len(batches) == 8
        assert {
            entry["weight"]
            for batch in batches
            for entry in batch["launches"]
        } == {1.0 / 512}
        all_ids.update(item.launch_id for item in launches)
        all_seeds.update(item.seed_id for item in launches)
    assert len(all_ids) == 4608
    assert len(all_seeds) == 4608


def test_every_case_file_is_exact_bytes_and_typed_roundtrip() -> None:
    value = protocol()
    authorities = strict_json_file(CASE_AUTHORITIES_PATH)
    assert authorities == build_all_case_authorities(value)
    for authority in authorities["cases"]:
        launches = build_case_launches(
            value, authority["role"], authority["timestep"]
        )
        expected = canonical_bytes(
            runtime_launch_payload(authority["campaign_id"], launches)
        )
        actual = (EXPERIMENT / authority["launch_manifest_path"]).read_bytes()
        assert actual == expected
        assert (
            hashlib.sha256(actual).hexdigest()
            == authority["runtime_launch_payload_byte_sha256"]
        )
        assert load_runtime_launch_payload(
            actual, authority["campaign_id"]
        ) == tuple(sorted(launches, key=lambda item: item.launch_id))


def test_all_checkpoint_chains_and_fresh_grid_preflight_pass() -> None:
    value = protocol()
    audit = production_synthetic_preflight(
        value, build_all_case_authorities(value)
    )
    assert audit["passed"]
    assert len(audit["covered_fields"]) == 9
    assert len(audit["case_checkpoint_chains"]) == 9
    assert all(item["passed"] for item in audit["case_checkpoint_chains"])
    assert audit["overlap_evidence"] == {
        "v2_identity_overlap_count": 0,
        "v2_seed_overlap_count": 0,
        "v2_position_overlap_count": 0,
        "v2_phase_space_overlap_count": 0,
        "v3_unique_case_launch_ids": 4608,
        "v3_unique_case_seed_ids": 4608,
        "v3_unique_physical_positions": 8,
        "v3_unique_physical_phase_space_points_per_case": 512,
    }
    assert all(audit["checks"].values())
    with pytest.raises(CanonicalizationError, match="reserved"):
        canonical_bytes({"__cft_type__": "tuple", "items": []})


def test_three_map_three_timestep_protocol_and_all_required_gates() -> None:
    value = protocol()
    assert set(value["field_adapter"]["maps"]) == {
        "primary",
        "refined",
        "enlarged",
    }
    assert set(value["orbit"]["timestep_policies"]) == {"N", "2N", "4N"}
    assert value["gates"]["maximum_successive_probability_change"] == 0.01
    assert value["gates"]["maximum_relative_energy_error"] == 1e-10
    assert value["gates"]["minimum_helix_position_order"] == 1.8
    assert value["gates"]["minimum_varying_e_position_order"] == 1.8
    assert value["gates"]["maximum_mirror_point_relative_error"] == 0.03
    assert value["gates"]["maximum_wall_endpoint_error_m"] == 1e-8
    assert value["gates"]["maximum_cpu_cuda_relative_velocity_difference"] == 1e-11
    assert value["gates"]["minimum_certificate_dense_to_bound_ratio"] == 0.001
    assert value["prior_campaign_disclosure"]["v1_orbit_outcome_access_count"] == 0
    assert value["prior_campaign_disclosure"][
        "v2_orbit_results_constructed_before_failure"
    ] == 32
    assert value["prior_campaign_disclosure"]["v3_launch_grid_reused"] is False


def test_manufactured_production_preflight_passes_without_outcomes() -> None:
    report = manufactured_gate_report(protocol())
    assert report["passed"]
    assert all(report["checks"].values())
