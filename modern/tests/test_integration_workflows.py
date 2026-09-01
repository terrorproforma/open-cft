from __future__ import annotations

import json
from pathlib import Path

import pytest

import cft_revival
from cft_revival.cli import main
from cft_revival.optimization import Design as OptimizationDesign
from cft_revival.physics import (
    L0_MODEL_FIDELITY,
    PhysicsConfigurationError,
    XenonOperatingPoint,
    evaluate_operating_point_artifact,
    evaluate_sweep_artifact,
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
