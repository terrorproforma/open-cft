from __future__ import annotations

import json
from pathlib import Path

from cft_revival.coupling import (
    COUPLING_SCHEMA_VERSION,
    COUPLING_V2_SCHEMA_VERSION,
    COUPLING_V3_SCHEMA_VERSION,
    COUPLING_V4_SCHEMA_VERSION,
)

SPEC = Path(__file__).parents[2] / "spec" / "coupling"


def test_record_schema_is_closed_and_matches_runtime_version() -> None:
    schema = json.loads((SPEC / "coupling-record-v4.schema.json").read_text())
    assert schema["$id"] == COUPLING_SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema["$defs"].values()
        if definition.get("type") == "object"
    )
    held_out = schema["$defs"]["heldOutValidation"]
    assert "disjoint_from_development" not in held_out["properties"]
    assert "all_required_cases_passed" not in held_out["properties"]
    assert {
        "development_manifest",
        "held_out_manifest",
        "outcomes",
        "validation_config_hash",
    } <= set(held_out["required"])
    assert {
        "validation_registration",
        "evidence_fingerprints",
        "orbit_identity",
    } <= set(schema["required"])
    assert "three_map_evidence_fingerprints" in (
        schema["$defs"]["heldOutOutcome"]["required"]
    )


def test_equation_ledger_has_unique_traceable_relations_and_prohibitions() -> None:
    ledger = json.loads((SPEC / "equation-ledger-v4.json").read_text())
    identifiers = [relation["id"] for relation in ledger["relations"]]
    assert len(identifiers) == len(set(identifiers))
    assert {
        "CPL-004-001",
        "CPL-004-003",
        "CPL-004-005",
        "CPL-004-007",
        "CPL-004-010",
        "CPL-004-011",
        "CPL-004-014",
        "CPL-004-015",
        "CPL-004-016",
        "CPL-004-017",
        "CPL-004-018",
    } <= set(identifiers)
    assert any("fixed axial windows" in item for item in ledger["prohibited_shortcuts"])
    assert ledger["coordinate_convention"]["coordinate_unit"] == "m"
    assert ledger["coordinate_convention"]["field_component_unit"] == "T"
    assert ledger["coordinate_convention"]["flux_unit"] == "Wb"
    assert ledger["criterion_status"]["assessed_56_case_role"] == (
        "development_non_validation"
    )
    citations = ledger["source_basis"]
    assert len(citations) >= 4
    assert all(
        item["url"].startswith("https://") and item["supports"]
        for item in citations
    )


def test_v3_closed_contour_schema_remains_historical_and_separate() -> None:
    schema = json.loads((SPEC / "coupling-record-v3.schema.json").read_text())
    assert schema["$id"] == "cft-field-plasma-coupling/3.0.0"
    assert len(
        {
            COUPLING_V2_SCHEMA_VERSION,
            COUPLING_V3_SCHEMA_VERSION,
            COUPLING_V4_SCHEMA_VERSION,
        }
    ) == 3
    assert COUPLING_SCHEMA_VERSION == COUPLING_V4_SCHEMA_VERSION
