from __future__ import annotations

from collections import Counter
import hashlib

import pytest

from cft_revival.experiment_runtime import canonical_bytes
from cft_revival.experiment_runtime.canonical import CanonicalizationError
from cft_revival.orbit_mc import EstimatorPolicy
from cft_revival.orbit_mc.artifacts import content_hash

from experiments.cft_orbit_wall_loss_v2.experiment import (
    LAUNCH_MANIFEST_PATH,
    batch_records,
    build_launches,
    load_runtime_launch_payload,
    launch_records,
    manufactured_gate_report,
    protocol,
    runtime_launch_payload,
    synthetic_serialization_audit,
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


def test_launches_are_equal_weight_deterministic_and_repeated_by_stratum() -> None:
    value = protocol()
    launches = build_launches(value)
    assert len(launches) == 512
    assert len({item.launch_id for item in launches}) == len(launches)
    assert len({item.seed_id for item in launches}) == len(launches)
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
    assert set(strata.values()) == {16}
    batches = batch_records(value, launches)
    assert len(batches) == 8
    weights = [
        entry["weight"] for batch in batches for entry in batch["launches"]
    ]
    assert set(weights) == {1.0 / len(launches)}
    assert content_hash(launch_records(launches))
    assert content_hash(
        {
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "batches": batches,
        }
    )


def test_runtime_launch_authority_is_exact_bytes_and_typed_roundtrip() -> None:
    launches = build_launches(protocol())
    expected = canonical_bytes(runtime_launch_payload(launches))
    actual = LAUNCH_MANIFEST_PATH.read_bytes()
    assert actual == expected
    assert hashlib.sha256(actual).hexdigest()
    assert load_runtime_launch_payload(actual) == tuple(
        sorted(launches, key=lambda item: item.launch_id)
    )


def test_serialization_audit_covers_vectors_and_rejects_reserved_tags() -> None:
    value = protocol()
    launches = build_launches(value)
    audit = synthetic_serialization_audit(
        value, launches, batch_records(value, launches)
    )
    assert audit["passed"]
    assert len(audit["covered_fields"]) == 9
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
    assert value["prior_campaign_disclosure"]["launch_grid_reused"] is True


def test_manufactured_production_preflight_passes_without_outcomes() -> None:
    report = manufactured_gate_report(protocol())
    assert report["passed"]
    assert all(report["checks"].values())
