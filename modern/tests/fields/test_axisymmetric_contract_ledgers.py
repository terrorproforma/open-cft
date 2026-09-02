import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_equation_solver_ledger_is_machine_readable_and_explicitly_fdm() -> None:
    ledger = json.loads(
        (ROOT / "spec/fields/equation-solver-ledger-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["schema_version"] == "1.2.0"
    assert "FDM" in ledger["classification"]
    assert ledger["equations"][2]["expression"] == "-div((1/r) grad psi) = mu J_phi"
    assert "not a permanent magnet" in ledger["source_convention"]["interpretation"]
    assert len(ledger["publication_gates"]["required_before_l1_production"]) >= 7


def test_dashboard_contract_requires_l1a_and_integrity() -> None:
    contract = json.loads(
        (ROOT / "spec/fields/field-map-contract-v1.json").read_text(encoding="utf-8")
    )
    assert contract["properties"]["model_level"]["const"] == "L1a"
    required = contract["properties"]["field_map"]["required"]
    assert {"r_m", "z_m", "b_r_t", "b_z_t", "b_magnitude_t"} <= set(required)
    assert "integrity" in contract["required"]


def test_published_manifest_and_artifacts_pass_closed_runtime_contract() -> None:
    from cft_revival.fields import (
        contains_negative_zero,
        validate_design_manifest_file,
    )

    result_directory = ROOT / "examples/axisymmetric/results"
    manifest = validate_design_manifest_file(
        result_directory / "manifest-l1a-v1.json"
    )
    assert manifest["model_level"] == "L1a"
    assert len(manifest["designs"]) == 3
    assert manifest["schema_version"] == "cft-axisymmetric-design-manifest/1.2.0"
    assert not contains_negative_zero(manifest)


def test_serialization_migration_manifest_anchors_legacy_and_current_hashes() -> None:
    result_directory = ROOT / "examples/axisymmetric/results"
    path = result_directory / "serialization-migration-v1.1-to-v1.2.json"
    migration = json.loads(path.read_text(encoding="utf-8"))
    assert migration["from"]["manifest_schema"].endswith("/1.1.0")
    assert migration["to"]["manifest_schema"].endswith("/1.2.0")
    assert migration["from"]["manifest_file_sha256"] == (
        "8444389efc87f89495e34d46ccf2deedcc44ee65614dfdd660beecf84cedc3b4"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256").read_text(encoding="ascii")
    assert sidecar == f"{digest}  {path.name}\n"
