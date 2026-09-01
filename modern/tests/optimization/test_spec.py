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
