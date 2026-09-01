from __future__ import annotations

from pathlib import Path

import pytest

from experiments.l1a_field_surrogate_v1.protocol import RESULTS
from experiments.l1a_field_surrogate_v1.validate import validate_bundle


def test_result_bundle_when_present_is_strictly_valid() -> None:
    if not (RESULTS / "run-manifest.json").exists():
        pytest.skip("prospective result bundle does not exist before execution")
    validation = validate_bundle()
    assert validation["passed"] is True
    assert validation["assessment_access_count"] == 1
    assert len(validation["representative_hashes"]) == 3


def test_result_sidecars_are_not_orphaned() -> None:
    for sidecar in RESULTS.rglob("*.sha256"):
        target = Path(str(sidecar)[: -len(".sha256")])
        assert target.is_file()
