from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import cft_revival
from cft_revival.cli import main
from cft_revival.optimization import Design as OptimizationDesign
from cft_revival.physics import (
    L0_MODEL_FIDELITY,
    MAX_L0_SWEEP_BATCH_SIZE,
    PhysicsConfigurationError,
    XenonOperatingPoint,
    evaluate_operating_point_artifact,
    evaluate_sweep_artifact,
    load_l0_json,
)

ROOT = Path(__file__).parents[1]


def _config(name: str) -> dict[str, object]:
    return json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_public_exports_keep_legacy_l0_and_optimization_domains_distinct() -> None:
    assert cft_revival.L0XenonOperatingPoint is XenonOperatingPoint
    assert cft_revival.OptimizationDesign is OptimizationDesign
    assert cft_revival.DesignPoint is not cft_revival.OptimizationDesign
    assert cft_revival.calculate_performance is not cft_revival.evaluate_l0_performance


def test_checked_point_artifact_has_complete_named_boundaries() -> None:
    artifact = evaluate_operating_point_artifact(
        _config("l0-representative-point.json")
    )
    assert artifact["document_type"] == "cft-revival-l0-result"
    assert artifact["model_fidelity"] == L0_MODEL_FIDELITY
    result = artifact["result"]
    assert isinstance(result, dict)
    assert result["axial_thrust_n"] > 0.0
    assert result["specific_impulse_s"] > 0.0
    assert set(result["power_budget"]) >= {
        "anode_input_power_w",
        "thruster_electrical_input_power_w",
        "ppu_input_power_w",
        "ppu_conversion_loss_w",
    }
    assert set(result["diagnostics"]) == {
        "particle_rate_residual_particles_per_s",
        "mass_flow_residual_kg_per_s",
        "beam_current_residual_a",
        "beam_power_residual_w",
        "ppu_power_margin_w",
    }
    assert result["applicability_warnings"]


def test_point_config_rejects_unlabelled_non_hypothetical_inputs() -> None:
    raw = _config("l0-representative-point.json")
    raw["hypothetical_inputs"] = False
    with pytest.raises(PhysicsConfigurationError, match="must be true"):
        evaluate_operating_point_artifact(raw)


def test_python_sweep_is_deterministic_and_reports_exact_reference_parity() -> None:
    raw = _config("l0-deterministic-sweep.json")
    raw["batch_size"] = 32
    raw["seed"] = 7
    first = evaluate_sweep_artifact(raw, device="python")
    second = evaluate_sweep_artifact(raw, device="python")
    assert [point["input"] for point in first["points"]] == [
        point["input"] for point in second["points"]
    ]
    assert first["cpu_reference_parity"]["compared_count"] == 32
    assert first["cpu_reference_parity"]["mismatch_count"] == 0
    assert first["cpu_reference_parity"]["within_documented_binary64_tolerance"]
    assert first["summary"]["output_ranges"]["axial_thrust_n"]["minimum"] > 0.0
    assert not first["runtime"]["timing_controlled"]


def test_cli_writes_machine_readable_point_and_sweep_artifacts(
    tmp_path: Path,
) -> None:
    point_output = tmp_path / "point.json"
    assert (
        main(
            [
                "l0-evaluate",
                str(ROOT / "config" / "l0-representative-point.json"),
                "--output",
                str(point_output),
            ]
        )
        == 0
    )
    assert json.loads(point_output.read_text())["document_type"] == (
        "cft-revival-l0-result"
    )

    sweep = _config("l0-deterministic-sweep.json")
    sweep["batch_size"] = 16
    sweep_path = tmp_path / "sweep-config.json"
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    sweep_output = tmp_path / "sweep.json"
    assert (
        main(
            [
                "l0-sweep",
                str(sweep_path),
                "--device",
                "python",
                "--output",
                str(sweep_output),
            ]
        )
        == 0
    )
    assert len(json.loads(sweep_output.read_text())["points"]) == 16


def test_cli_failure_names_typed_configuration_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["l0-evaluate", str(ROOT / "config" / "default.json")])
    assert caught.value.code == 2
    assert "PhysicsConfigurationError" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("root", "unexpected", True, "unknown field"),
        ("inputs", "legacy_voltage", 300.0, "unknown field"),
        ("fractions", "xe_triple_plus", 0.0, "unknown field"),
    ],
)
def test_point_schema_rejects_unknown_fields(
    target: str, field: str, value: object, message: str
) -> None:
    raw = _config("l0-representative-point.json")
    if target == "root":
        destination = raw
    elif target == "inputs":
        destination = raw["inputs"]
    else:
        destination = raw["inputs"]["charge_state_number_fractions"]
    destination[field] = value
    with pytest.raises(PhysicsConfigurationError, match=message):
        evaluate_operating_point_artifact(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw.__setitem__("unknown_top_level", True),
            "unknown field",
        ),
        (
            lambda raw: raw["ranges"].__setitem__("legacy_flow_sccm", [1, 2]),
            "unknown field",
        ),
        (
            lambda raw: raw["ranges"].pop("discharge_voltage_v"),
            "missing required",
        ),
        (
            lambda raw: raw["ranges"].__setitem__(
                "discharge_voltage_v", [300.0]
            ),
            "must be \\[minimum, maximum\\]",
        ),
        (
            lambda raw: raw["ranges"].__setitem__(
                "discharge_voltage_v", [500.0, 150.0]
            ),
            "must increase",
        ),
        (
            lambda raw: raw["ranges"].__setitem__(
                "ppu_efficiency_fraction", [0.8, "0.9"]
            ),
            "real number",
        ),
        (
            lambda raw: raw.__setitem__("batch_size", True),
            "must be an integer",
        ),
        (
            lambda raw: raw.__setitem__("batch_size", 0),
            "batch_size must lie",
        ),
        (
            lambda raw: raw.__setitem__(
                "batch_size", MAX_L0_SWEEP_BATCH_SIZE + 1
            ),
            "batch_size must lie",
        ),
        (
            lambda raw: raw.__setitem__("hypothetical_inputs", False),
            "must be true",
        ),
    ],
)
def test_sweep_schema_rejects_adversarial_mutations(
    mutate, message: str
) -> None:
    raw = _config("l0-deterministic-sweep.json")
    mutate(raw)
    with pytest.raises(PhysicsConfigurationError, match=message):
        evaluate_sweep_artifact(raw, device="python")


@pytest.mark.parametrize("device", ["auto", "gpu", "cuda:-1", "cuda:01", "", None])
def test_sweep_rejects_malformed_device_before_evaluation(device: object) -> None:
    raw = _config("l0-deterministic-sweep.json")
    raw["batch_size"] = 1
    with pytest.raises(PhysicsConfigurationError, match="device must be"):
        evaluate_sweep_artifact(raw, device=device)


def test_sweep_schema_accepts_nearby_valid_boundaries() -> None:
    raw = _config("l0-deterministic-sweep.json")
    raw["batch_size"] = 1
    raw["ranges"]["ppu_efficiency_fraction"] = [0.95, 1.0]
    raw["ranges"]["xe_double_plus_fraction_of_ions"] = [0.0, 0.01]
    artifact = evaluate_sweep_artifact(raw, device="python")
    assert len(artifact["points"]) == 1


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "1e999"])
def test_l0_loader_rejects_nonfinite_numbers_with_typed_error(
    tmp_path: Path, literal: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(f'{{"value":{literal}}}', encoding="utf-8")
    with pytest.raises(PhysicsConfigurationError, match="contains"):
        load_l0_json(path)


def test_l0_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"1.0","schema_version":"2.0"}')
    with pytest.raises(PhysicsConfigurationError, match="duplicate key"):
        load_l0_json(path)


def test_documented_no_install_core_commands_work_from_modern() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    json_commands = (
        (
            "l0-evaluate",
            "config/l0-representative-point.json",
        ),
        (
            "validate-campaign-spec",
            "spec/optimization/campaign-v1.json",
        ),
        (
            "generate-initial-design",
            "spec/optimization/campaign-v1.json",
            "--count",
            "32",
            "--seed",
            "7",
        ),
    )
    for arguments in json_commands:
        completed = subprocess.run(
            [sys.executable, "-m", "cft_revival", *arguments],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["document_type"]

    plain_commands = (
        ("validate-config", "config/default.json"),
        ("cusp-probability", "--low-t", "0.02", "--high-t", "0.2"),
    )
    for arguments in plain_commands:
        completed = subprocess.run(
            [sys.executable, "-m", "cft_revival", *arguments],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip()

    for documentation in (
        ROOT.parent / "README.md",
        ROOT / "README.md",
        ROOT / "docs" / "FIRST_RESULTS.md",
    ):
        text = documentation.read_text(encoding="utf-8")
        command_blocks = [
            block
            for block in re.findall(
                r"```powershell\n(.*?)```", text, flags=re.DOTALL
            )
            if "python -m cft_revival" in block
        ]
        assert command_blocks
        assert all(
            '$env:PYTHONPATH = "$PWD\\src"' in block
            for block in command_blocks
        )
    assert "python -m cft_revival" not in (
        ROOT.parent / "ROADMAP.md"
    ).read_text(encoding="utf-8")
