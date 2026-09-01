from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from cft_revival.pic import lxcat_source_hash


SPEC_ROOT = Path(__file__).parents[2] / "spec" / "pic"


def test_foundation_ledger_is_honest_and_has_all_acceptance_gates() -> None:
    ledger = json.loads((SPEC_ROOT / "pic-foundation-v1.json").read_text("utf-8"))
    assert ledger["model_level"] == "L3-foundation"
    assert ledger["status"] == "reduced-kernel-correctness-verified-not-predictive-cft"
    assert ledger["integration_gate"]["warpx_picmi"].startswith("adapter protocol only")
    assert ledger["integration_gate"]["cft_predictive_outputs"].startswith("prohibited")
    refinement = ledger["energy_refinement_contract"]
    assert refinement["acceptance"] == "coarse envelope > 1.5 * fine envelope"
    assert refinement["observed_current_fixture_ratio"] == 1.93
    assert "no general convergence order" in refinement["claim_exclusions"]
    assert ledger["charge_representability"]["rejection"].startswith(
        "unrepresentable volumetric density"
    )
    cases = set(ledger["verification_cases"])
    assert {
        "manufactured periodic Poisson",
        "cold-plasma quarter period",
        "MCC collision-rate statistics",
        "explicit Poisson nonconvergence",
        "Nyquist face-field preservation and Poisson energy identity",
        "late-particle MCC rollback of particles, RNG, and counters",
    } <= cases


def test_only_synthetic_cross_sections_are_shipped() -> None:
    fixture = json.loads(
        (SPEC_ROOT / "synthetic-cross-sections-v1.json").read_text("utf-8")
    )
    assert fixture["purpose"] == "verification-only"
    assert fixture["physical_validation"] is False
    assert all(
        table["source"].startswith("synthetic-verification:")
        for table in fixture["tables"]
    )
    assert "not xenon collision data" in fixture["warning"]


def test_lxcat_boundary_hashes_exact_source_bytes(tmp_path) -> None:
    raw = b"Energy (eV),Cross section (m2)\r\n1.0,2.0e-20\r\n"
    source = tmp_path / "lxcat-export.txt"
    source.write_bytes(raw)
    assert lxcat_source_hash(source) == sha256(raw).hexdigest()
