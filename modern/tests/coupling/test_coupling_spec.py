from __future__ import annotations

import json
from pathlib import Path

from cft_revival.coupling import COUPLING_SCHEMA_VERSION

SPEC = Path(__file__).parents[2] / "spec" / "coupling"


def test_record_schema_is_closed_and_matches_runtime_version() -> None:
    schema = json.loads((SPEC / "coupling-record-v2.schema.json").read_text())
    assert schema["$id"] == COUPLING_SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema["$defs"].values()
        if definition.get("type") == "object"
    )


def test_equation_ledger_has_unique_traceable_relations_and_prohibitions() -> None:
    ledger = json.loads((SPEC / "equation-ledger-v2.json").read_text())
    identifiers = [relation["id"] for relation in ledger["relations"]]
    assert len(identifiers) == len(set(identifiers))
    assert {
        "CPL-002-002",
        "CPL-002-005",
        "CPL-002-006",
        "CPL-002-010",
    } <= set(identifiers)
    assert any("fixed axial windows" in item for item in ledger["prohibited_shortcuts"])
    assert ledger["coordinate_convention"]["coordinate_unit"] == "m"
    assert ledger["coordinate_convention"]["field_component_unit"] == "T"
    assert ledger["coordinate_convention"]["covariance_unit"] == "T^2"
