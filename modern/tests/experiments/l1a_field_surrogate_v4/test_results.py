from pathlib import Path

import pytest

from experiments.l1a_field_surrogate_v4.protocol import RESULTS, verify_json, write_json
from experiments.l1a_field_surrogate_v4.run import new_counters
from experiments.l1a_field_surrogate_v4.validate import _counters, validate_bundle


def test_counter_validator_rejects_unproved_completion() -> None:
    counters = new_counters()
    counters["materialized"]["method"]["fine"] = 2
    with pytest.raises(ValueError, match="solver/materialization"):
        _counters({"access_counters": counters})


def test_sidecar_rejects_byte_change(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    write_json(path, {"value": 1})
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="invalid sidecar"):
        verify_json(path)


def test_terminal_bundle_when_present_has_no_cache() -> None:
    if not (RESULTS / "execution-lock.json").exists():
        pytest.skip("prospective execution has not occurred")
    result = validate_bundle()
    assert result["passed"]
    assert not (RESULTS / ".working").exists()
