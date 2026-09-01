from __future__ import annotations

import json
from pathlib import Path

import pytest

from cft_revival.cli import main
from cft_revival.optimization.spec import (
    CampaignSpecError,
    campaign_spec_artifact,
    load_json_strict,
    validate_campaign_spec,
)

ROOT = Path(__file__).parents[2]
SPEC_PATH = ROOT / "spec" / "optimization" / "campaign-v1.json"


def _raw_spec() -> dict[str, object]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_checked_campaign_spec_matches_executable_policy() -> None:
    validated = validate_campaign_spec(load_json_strict(SPEC_PATH))
    assert validated.campaign_spec_id.endswith("@1.4")
    assert len(validated.variables) == 8
    assert len(validated.objectives) == 4
    assert validated.campaign_config.highest_fidelity_attempt_limit == 16
    assert validated.campaign_config.reserved_high_fidelity_retries == 4


def test_initial_design_manifest_is_deterministic_and_botorch_free() -> None:
    raw = load_json_strict(SPEC_PATH)
    first = campaign_spec_artifact(raw, initial_design_count=32, seed=19)
    second = campaign_spec_artifact(raw, initial_design_count=32, seed=19)
    assert first == second
    assert len(first["designs"]) == 32
    assert first["generator"]["requires_botorch"] is False
    assert len({item["design_id"] for item in first["designs"]}) == 32


def test_campaign_spec_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"1.4","schema_version":"1.5"}')
    with pytest.raises(CampaignSpecError, match="duplicate key"):
        load_json_strict(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw.__setitem__("unknown_top_level", True),
            "unknown field",
        ),
        (
            lambda raw: raw["decision_space"]["variables"][0].__setitem__(
                "legacy_alias", "Ua"
            ),
            "unknown field",
        ),
        (
            lambda raw: raw["iteration_policy"]["promotion"].__setitem__(
                "unknown_gate", True
            ),
            "unknown field",
        ),
        (
            lambda raw: raw["iteration_policy"].__setitem__(
                "botorch_output_transform", [1.0, -1.0, 1.0, -1.0]
            ),
            "botorch_output_transform",
        ),
        (
            lambda raw: raw["iteration_policy"]["acquisition_mix"][0].__setitem__(
                "fraction", 2.0
            ),
            "fraction must lie",
        ),
        (
            lambda raw: raw["iteration_policy"]["acquisition_mix"][0].__setitem__(
                "fraction", 0.5
            ),
            "sum to one",
        ),
        (
            lambda raw: raw["iteration_policy"]["acquisition_mix"][0].__setitem__(
                "fraction", "0.6"
            ),
            "real number",
        ),
        (
            lambda raw: raw["iteration_policy"][
                "cheap_medium_evaluations"
            ].__setitem__("minimum", 17),
            "minimum <= maximum",
        ),
        (
            lambda raw: raw["highest_fidelity_attempt_policy"].__setitem__(
                "total_attempt_limit", 17
            ),
            "attempt limit",
        ),
        (
            lambda raw: raw["stopping_gates"][
                "all_must_hold_unless_cost_ceiling_reached"
            ][0].__setitem__("value", 11),
            "must match",
        ),
        (
            lambda raw: raw["iteration_policy"].__setitem__(
                "asynchronous_pending_aware", False
            ),
            "must be true",
        ),
        (
            lambda raw: raw["objectives"][3].__setitem__(
                "direction", "maximize"
            ),
            "ordered as",
        ),
        (
            lambda raw: raw["benchmark"].__setitem__("results", {}),
            "must remain null",
        ),
    ],
)
def test_campaign_spec_rejects_adversarial_policy_mutations(
    mutate, message: str
) -> None:
    raw = _raw_spec()
    mutate(raw)
    with pytest.raises(CampaignSpecError, match=message):
        validate_campaign_spec(raw)


def test_campaign_spec_accepts_nearby_valid_acquisition_mix() -> None:
    raw = _raw_spec()
    mix = raw["iteration_policy"]["acquisition_mix"]
    mix[0]["fraction"] = 0.5
    mix[1]["fraction"] = 0.25
    mix[2]["fraction"] = 0.25
    validated = validate_campaign_spec(raw)
    assert validated.campaign_spec_id.endswith("@1.4")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "1e999"])
def test_campaign_spec_loader_rejects_nonfinite_numbers(
    tmp_path: Path, literal: str
) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value":{literal}}}', encoding="utf-8")
    with pytest.raises(CampaignSpecError, match="non-finite|contains"):
        load_json_strict(path)


def test_campaign_cli_validates_and_generates_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate-campaign-spec", str(SPEC_PATH)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["document_type"] == (
        "cft-revival-optimization-campaign-validation"
    )
    assert validated["valid"] is True
    assert validated["summary"]["dimensions"] == 8

    output = tmp_path / "initial.json"
    assert (
        main(
            [
                "generate-initial-design",
                str(SPEC_PATH),
                "--count",
                "24",
                "--seed",
                "5",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(json.loads(output.read_text())["designs"]) == 24
