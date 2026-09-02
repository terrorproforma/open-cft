from __future__ import annotations

import json
from pathlib import Path

from cft_revival.orbit_mc import HANDOFF_VERSION


SPEC_ROOT = Path(__file__).resolve().parents[2] / "spec" / "orbit_mc"


def test_orbit_specs_are_closed_draft_2020_12_json() -> None:
    schemas = sorted(SPEC_ROOT.glob("*.schema.json"))
    assert len(schemas) == 3
    for path in schemas:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["$schema"].endswith("2020-12/schema")
        assert value["type"] == "object"
        assert value["additionalProperties"] is False
        assert set(value["required"]) == set(value["properties"])
        assert all(
            definition.get("additionalProperties") is False
            for definition in value.get("$defs", {}).values()
            if definition.get("type") == "object"
        )


def test_validation_protocol_preregisters_authority_and_convergence() -> None:
    protocol = json.loads((SPEC_ROOT/"validation-protocol-v1.json").read_text(encoding="utf-8"))
    assert protocol["authority"]["primary"] == "direct_first-event_full-orbit_ensemble"
    assert protocol["launch_strata"]["minimum_deterministic_gyrophases"] >= 8
    assert protocol["numerical_gates"]["field_maps"] == [
        "primary", "refined_resolution", "enlarged_domain"
    ]
    assert protocol["numerical_gates"]["timestep_levels"] == ["N", "2N", "4N"]
    assert protocol["publication_gates"]["held_out_geometry_family_required"]
    assert protocol["numerical_gates"]["require_certified_cellwise_max_b_bound"]
    assert protocol["numerical_gates"]["require_relativistic_phase_gamma"]
    assert protocol["numerical_gates"]["require_certificate_tightness_preflight"]
    assert protocol["numerical_gates"][
        "require_external_checkpoint_campaign_and_launch_authority"
    ]
    assert protocol["numerical_gates"][
        "require_runtime_schema_semantic_equivalence"
    ]
    assert protocol["numerical_gates"]["estimator_policy"] == (
        "unweighted_binomial"
    )
    assert protocol["numerical_gates"]["require_equal_launch_weights"]
    assert protocol["numerical_gates"][
        "artifact_sealing_requires_deterministic_replay"
    ]
    assert protocol["numerical_gates"][
        "require_external_batch_manifest_sha256_for_artifact_write"
    ]
    assert protocol["integration"]["status"] == (
        "export_only_pending_consumer_integration"
    )
    assert "pic" in protocol["authority"]["excluded_claims"]


def test_result_and_checkpoint_schemas_expose_runtime_authority_fields() -> None:
    result_schema = json.loads(
        (SPEC_ROOT/"result-v1.schema.json").read_text(encoding="utf-8")
    )
    result_required = set(result_schema["$defs"]["result"]["required"])
    assert {
        "configured_max_time_s", "configured_max_path_m", "event_tolerance_m"
    } <= result_required
    assert "event_witness" in result_required
    assert result_schema["properties"]["schema_version"]["const"].endswith(
        "/1.4.0"
    )
    assert result_schema["$defs"]["result"]["properties"]["transit_fraction"][
        "maximum"
    ] == 1
    interpolation_required = set(
        result_schema["$defs"]["interpolationEvidence"]["required"]
    )
    assert {
        "dense_diagnostic_max_b_t", "certificate_tightness_ratio",
        "minimum_certificate_tightness_ratio", "certificate_preflight_passed",
    } <= interpolation_required
    checkpoint_schema = json.loads(
        (SPEC_ROOT/"checkpoint-v1.schema.json").read_text(encoding="utf-8")
    )
    assert "launches" in checkpoint_schema["required"]
    assert {
        "batch_manifest", "partial_current_batch", "pending_launch_ids", "coverage"
    } <= set(checkpoint_schema["required"])
    assert result_schema["$defs"]["estimator"]["properties"]["policy"][
        "const"
    ] == "unweighted_binomial"
    assert "campaign_identity_sha256" in checkpoint_schema["properties"][
        "authority"
    ]["required"]


def test_handoff_version_agrees_across_runtime_schema_protocol_and_docs() -> None:
    handoff_schema = json.loads(
        (SPEC_ROOT/"coupling-v4.2-handoff-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = json.loads(
        (SPEC_ROOT/"validation-protocol-v1.json").read_text(encoding="utf-8")
    )
    integration_doc = (
        SPEC_ROOT.parents[1]/"docs"/"workstreams"/"orbit-mc-integration.md"
    ).read_text(encoding="utf-8")
    assert HANDOFF_VERSION == (
        "cft-revival-orbit-mc-coupling-v4.2/1.3.0"
    )
    assert handoff_schema["properties"]["schema_version"]["const"] == (
        HANDOFF_VERSION
    )
    assert protocol["integration"]["handoff_schema"] == HANDOFF_VERSION
    assert f"`{HANDOFF_VERSION}`" in integration_doc
