from __future__ import annotations

import json

import pytest

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    FieldDeviceError,
    SolverConfig,
    device_available,
    field_artifact,
    max_field_difference,
    solve_problem_cpu,
    solve_problem_warp,
    validate_field_artifact,
    write_field_artifact,
)


def _problem() -> AxisymmetricProblem:
    return AxisymmetricProblem(
        "parity",
        AxisymmetricDomain(0.12, -0.15, 0.15, 24, 48),
        (
            AzimuthalCurrentBand("upstream", 0.035, 0.055, -0.07, -0.03, 1800, 1),
            AzimuthalCurrentBand("downstream", 0.045, 0.070, 0.025, 0.065, 1300, -1),
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_field_and_diagnostics_match_python(device: str) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} unavailable")
    config = SolverConfig(relative_tolerance=1.0e-10)
    reference = solve_problem_cpu(_problem(), config)
    actual = solve_problem_warp(_problem(), device=device, config=config)
    differences = max_field_difference(reference, actual)
    assert max(
        differences["psi_scale_relative"],
        differences["br_scale_relative"],
        differences["bz_scale_relative"],
    ) < 2.0e-12
    assert actual.diagnostics.converged
    assert actual.diagnostics.relative_residual_l2 <= config.relative_tolerance
    assert abs(actual.diagnostics.iterations - reference.diagnostics.iterations) <= 1
    assert actual.diagnostics.max_flux_reconstruction_identity_t_per_m < 1.0e-10


def test_invalid_warp_device_is_typed() -> None:
    with pytest.raises(FieldDeviceError):
        solve_problem_warp(_problem(), device="gpu")


def test_versioned_artifact_round_trip_and_shape_validation(tmp_path) -> None:
    config = SolverConfig()
    field = solve_problem_cpu(_problem(), config)
    artifact = field_artifact(_problem(), config, field, map_stride=3, wall_radius_m=0.08)
    validate_field_artifact(artifact)
    assert artifact["model_level"] == "L1a"
    assert "not permanent-magnet" in artifact["model_description"]
    destination = tmp_path / "field.json"
    write_field_artifact(destination, artifact)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    validate_field_artifact(loaded)
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_artifact_rejects_nonfinite_and_shape_damage() -> None:
    config = SolverConfig()
    artifact = field_artifact(_problem(), config, solve_problem_cpu(_problem(), config))
    artifact["field_map"]["b_z_t"][0].pop()
    with pytest.raises(ValueError, match="shape"):
        validate_field_artifact(artifact)
