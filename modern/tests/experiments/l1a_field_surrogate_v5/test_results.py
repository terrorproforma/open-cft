import pytest

from cft_revival.experiment_runtime import BundleState, strict_json_file, validate_bundle
from experiments.l1a_field_surrogate_v5.protocol import CACHE, RESULTS


def test_terminal_bundle_when_present_is_shared_runtime_bundle() -> None:
    if not (RESULTS / "execution-lock.json").exists():
        pytest.skip("prospective execution has not occurred")
    manifest = validate_bundle(RESULTS)
    terminal = strict_json_file(RESULTS / "terminal.json")
    assert manifest["state"] in {state.value for state in BundleState}
    assert terminal["state"] == manifest["state"]
    assert terminal["counts"]["attempt_count"] == 1
    assert not CACHE.exists()
