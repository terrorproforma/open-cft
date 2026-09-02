from __future__ import annotations

import json
from pathlib import Path

from cft_revival.experiment_runtime import BundleState, EVENT_TRANSITION_PAIRS


SPEC_ROOT = Path(__file__).resolve().parents[2] / "spec" / "experiment_runtime"


def test_all_runtime_specs_are_strict_json_objects() -> None:
    paths = sorted(SPEC_ROOT.glob("*.json"))
    assert {path.name for path in paths} == {
        "execution-lock-v1.schema.json",
        "manifest-v1.schema.json",
        "state-machine-v1.json",
        "terminal-v1.schema.json",
    }
    for path in paths:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
        assert isinstance(value, dict)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key}")
        result[key] = value
    return result


def test_state_machine_and_manifest_schema_match_runtime_terminal_states() -> None:
    state_machine = json.loads((SPEC_ROOT / "state-machine-v1.json").read_text(encoding="utf-8"))
    schema = json.loads((SPEC_ROOT / "manifest-v1.schema.json").read_text(encoding="utf-8"))
    runtime_states = {state.value for state in BundleState}
    assert set(state_machine["terminal_bundle_states"]) == runtime_states
    assert set(schema["properties"]["state"]["enum"]) == runtime_states
    assert schema["properties"]["schema_version"]["const"].endswith("/1.0.0")
    transitions = {tuple(item) for item in state_machine["transitions"]}
    event_pairs = {tuple(item) for item in state_machine["event_transition_pairs"]}
    assert event_pairs == EVENT_TRANSITION_PAIRS
    assert ("development_started", "development_rejection") in transitions
    assert ("development_accepted", "assessment_started") in transitions
    assert ("assessment_started", "accepted_result") in transitions
    assert ("development_rejection", "assessment_started") not in transitions
