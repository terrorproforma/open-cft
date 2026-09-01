from __future__ import annotations

import pytest

from experiments.l1a_field_surrogate_v2.protocol import RESULTS
from experiments.l1a_field_surrogate_v2.validate import validate_bundle


def test_terminal_bundle_when_present_validates_success_or_failure() -> None:
    if not (RESULTS / "execution-lock.json").exists():
        pytest.skip("prospective v2 bundle does not exist before execution")
    result = validate_bundle()
    assert result["passed"] is True
    assert result["terminal_kind"] in {"failure", "terminal-result"}
    assert result["closure_entries"] > 0


def test_failure_bundle_has_required_access_counters() -> None:
    if not (RESULTS / "failure-manifest.json").exists():
        pytest.skip("no terminal execution failure")
    result = validate_bundle()
    counters = result["counters"]
    assert set(counters) == {
        "coarse_completed",
        "fine_completed",
        "solver_accesses",
        "model_fits",
        "method_label_accesses",
        "calibration_accesses",
        "assessment_accesses",
    }
