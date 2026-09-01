import json
from pathlib import Path


SPEC_ROOT = Path(__file__).parents[2] / "spec" / "physics"


def test_equation_ledger_is_machine_readable_and_traceable() -> None:
    ledger = json.loads((SPEC_ROOT / "equation-ledger.json").read_text(encoding="utf-8"))
    equations = ledger["equations"]
    identifiers = [equation["id"] for equation in equations]
    assert len(identifiers) == len(set(identifiers))
    assert {"PHY-L0-010", "PHY-FIELD-MMS-002"} <= set(identifiers)
    for equation in equations:
        assert equation["expression"]
        assert equation["symbols"]
        assert all(unit for unit in equation["symbols"].values())
        assert equation["provenance"]
        assert equation["validity_domain"]
        assert equation["confidence"]
        assert "total_efficiency" not in equation["expression"]


def test_2020_outputs_are_labeled_external_evidence_not_truth() -> None:
    fixtures = json.loads(
        (SPEC_ROOT / "external-regression-fixtures.json").read_text(encoding="utf-8")
    )
    assert "not fitted truth" in fixtures["policy"]
    assert fixtures["source"]["doi_url"] == "https://doi.org/10.2514/1.A34584"
    assert len(fixtures["fixtures"]) == 3
    for fixture in fixtures["fixtures"]:
        assert fixture["role"].startswith("external_regression_only")
        assert fixture["role"] != "fitted_truth"
