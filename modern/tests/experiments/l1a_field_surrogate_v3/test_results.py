from __future__ import annotations

import pytest

from experiments.l1a_field_surrogate_v3.protocol import RESULTS
from experiments.l1a_field_surrogate_v3.validate import validate_bundle


def test_terminal_bundle_when_present_is_strict() -> None:
    if not (RESULTS / "execution-lock.json").exists():
        pytest.skip("v3 is prospective before execution")
    result = validate_bundle()
    assert result["passed"] is True
    assert result["closure_entries"] > 0


def test_result_commit_parent_contract_is_recorded() -> None:
    if not (RESULTS / "terminal-result.json").exists():
        pytest.skip("terminal result not yet committed")
    result = validate_bundle()
    assert result["status"] in {
        "accepted",
        "failed-development-selection-gates",
        "failed-predeclared-assessment-gates",
    }
