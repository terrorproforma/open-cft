import json
from pathlib import Path


MODERN_ROOT = Path(__file__).resolve().parents[2]


def test_equation_ledger_declares_scope_sources_and_unresolved_physics() -> None:
    ledger = json.loads(
        (MODERN_ROOT / "spec" / "hybrid" / "equation-ledger-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["schema_version"] == "1.0.0"
    equation_ids = {entry["id"] for entry in ledger["equations"]}
    assert {
        "HYB-PUSH-001",
        "HYB-TIME-001",
        "HYB-DEP-001",
        "HYB-COLL-001",
        "HYB-CX-001",
        "HYB-ELEC-001",
        "HYB-SRC-001",
    } <= equation_ids
    fixture = next(
        entry for entry in ledger["equations"] if entry["id"] == "HYB-COLL-FIXTURE-001"
    )
    assert fixture["source_url"] is None
    assert "synthetic" in fixture["method"]
    assert "anomalous electron mobility" in ledger["explicitly_unresolved"]
    assert "not a self-consistent" in ledger["claim"]


def test_checkpoint_schema_pins_rng_and_format_versions() -> None:
    schema = json.loads(
        (MODERN_ROOT / "spec" / "hybrid" / "checkpoint-schema-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload = schema["properties"]["payload"]
    assert payload["properties"]["schema_version"]["const"] == "hybrid-checkpoint-v1"
    assert (
        payload["properties"]["rng"]["properties"]["algorithm"]["const"]
        == "splitmix64-counter-v1"
    )
    species = payload["properties"]["particles"]["items"]["properties"]["species"]
    assert species["properties"]["symbol"]["enum"] == [
        "Xe",
        "Xe+",
        "Xe2+",
    ]
    assert payload["additionalProperties"] is False
    assert species["additionalProperties"] is False
    assert schema["additionalProperties"] is False
    assert payload["properties"]["rng"]["additionalProperties"] is False
    assert (
        payload["properties"]["time_levels"]["additionalProperties"] is False
    )
    assert (
        payload["properties"]["particles"]["items"]["additionalProperties"]
        is False
    )
    assert (
        payload["properties"]["provenance"]["additionalProperties"] is False
    )
    assert (
        payload["properties"]["time_levels"]["properties"]["velocity"]["const"]
        == "n_minus_one_half"
    )
    assert "not authenticity" in schema["properties"]["sha256"]["description"]
    assert "duplicate-key rejection" in schema["description"]
    charge_alternatives = species["oneOf"]
    assert [
        alternative["properties"]["charge_c"]["const"]
        for alternative in charge_alternatives
    ] == [0.0, 1.602176634e-19, 3.204353268e-19]
