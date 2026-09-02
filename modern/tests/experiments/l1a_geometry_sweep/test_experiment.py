from __future__ import annotations

import hashlib

import pytest

from cft_revival.fields import (
    device_available,
    field_artifact,
    max_field_difference,
    solve_problem_cpu,
    solve_problem_warp,
    validate_field_artifact_file,
    write_field_artifact,
)
from cft_revival.geometry import canonical_json, deserialize_geometry
from experiments.l1a_geometry_sweep.experiment import (
    CLASSIFICATION,
    CONSTRAINTS,
    SOLVER,
    build_case,
    dominates,
    nondominated,
    sample_designs,
    stable_hash,
    validate_experiment_bundle,
)


def test_sampling_is_deterministic_bounded_and_unique() -> None:
    left = sample_designs()
    right = sample_designs()
    assert len(left) == len(right) == 96
    assert [item.design_id for item in left] == [item.design_id for item in right]
    assert len({item.design_id for item in left}) == 96
    for design in left:
        assert all(
            variable.lower <= value <= variable.upper
            for value, variable in zip(design.values, design.variables, strict=True)
        )
        assert "halton" in design.provenance or "boundary-challenge" in design.provenance


def test_geometry_preview_manufacturing_and_hashes_are_valid() -> None:
    for index, design in enumerate(sample_designs(16)):
        case = build_case(design, index)
        assert case.preview.authoritative is False
        assert case.geometry.canonical_sha256 == case.geometry_sha256
        assert case.derived["stage_count"] in (3, 4, 5)
        assert case.derived["worst_case_radial_manufacturing_margin_m"] >= 0.0
        assert case.derived["worst_case_axial_manufacturing_margin_m"] >= 0.0
        assert len(case.problem.sources) == 2 * case.derived["stage_count"]
        assert len(case.geometry_sha256) == 64
        assert len(case.source_sha256) == 64
        assert len(case.config_sha256) == 64
        assert case.case_sha256 == stable_hash(
            {
                "geometry_sha256": case.geometry_sha256,
                "source_sha256": case.source_sha256,
                "config_sha256": case.config_sha256,
            }
        )


def _constraints(**overrides: float) -> dict[str, float]:
    values = {
        definition["name"]: (
            0.0 if definition["sense"] == ">=" else definition["threshold"]
        )
        for definition in CONSTRAINTS
    }
    values["topology_confidence"] = 1.0
    values.update(overrides)
    return values


def _rank_case(
    case_id: str,
    centre: float,
    mirror: float,
    gradient: float,
    energy: float,
    **constraint_overrides: float,
):
    constraints = _constraints(**constraint_overrides)
    feasible = all(
        constraints[item["name"]] <= item["threshold"]
        if item["sense"] == "<="
        else constraints[item["name"]] >= item["threshold"]
        for item in CONSTRAINTS
    )
    return {
        "case_id": case_id,
        "status": "success",
        "qois": {
            "centreline_mid_abs_bz_t": centre,
            "minimum_mirror_ratio": mirror,
            "stage_gradient_rms_t_per_m": gradient,
            "field_energy_j": energy,
        },
        "constraints": constraints,
        "feasible": feasible,
    }


def test_exact_constrained_ranking_and_no_failure_penalty() -> None:
    strong = _rank_case("strong", 2.0, 3.0, 4.0, 1.0)
    weak = _rank_case("weak", 1.0, 2.0, 3.0, 2.0)
    tradeoff = _rank_case("tradeoff", 3.0, 1.0, 5.0, 3.0)
    infeasible = _rank_case(
        "infeasible",
        100.0,
        100.0,
        100.0,
        0.01,
        boundary_to_peak_ratio=0.2,
    )
    failed = {
        "case_id": "failed",
        "status": "failure",
        "failure": {"code": "SOLVER_FAILURE"},
    }
    assert dominates(strong, weak)
    assert dominates(strong, infeasible)
    assert {item["case_id"] for item in nondominated((strong, weak, tradeoff, infeasible, failed))} == {
        "strong",
        "tradeoff",
    }
    with pytest.raises(ValueError, match="successful"):
        dominates(strong, failed)


def test_cpu_artifact_and_geometry_reload(tmp_path) -> None:
    case = build_case(sample_designs(1)[0], 0)
    field = solve_problem_cpu(case.problem, SOLVER)
    artifact = field_artifact(
        case.problem,
        SOLVER,
        field,
        map_stride=1,
        wall_radius_m=case.geometry.chamber.outer_radius_m,
    )
    path = tmp_path / "field.json"
    digest = write_field_artifact(path, artifact)
    loaded = validate_field_artifact_file(
        path,
        expected_file_sha256=digest,
        expected_payload_sha256=artifact["integrity"]["payload_sha256"],
    )
    assert loaded["model_level"] == "L1a"
    geometry_path = tmp_path / "geometry.json"
    geometry_bytes = canonical_json(case.geometry.to_dict()).encode("utf-8")
    geometry_path.write_bytes(geometry_bytes)
    assert hashlib.sha256(geometry_bytes).hexdigest()
    reloaded = deserialize_geometry(geometry_path.read_text(encoding="utf-8"))
    assert reloaded.canonical_sha256 == case.geometry_sha256


@pytest.mark.skipif(not device_available("cuda:0"), reason="CUDA Warp device unavailable")
def test_cpu_cuda_parity_for_representative_case() -> None:
    case = build_case(sample_designs(1)[0], 0)
    cpu = solve_problem_cpu(case.problem, SOLVER)
    cuda = solve_problem_warp(case.problem, device="cuda:0", config=SOLVER)
    differences = max_field_difference(cpu, cuda)
    assert differences["psi_scale_relative"] <= 2.0e-9
    assert differences["br_scale_relative"] <= 2.0e-8
    assert differences["bz_scale_relative"] <= 2.0e-8


def test_generated_bundle_reload_if_present() -> None:
    results = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "experiments"
        / "l1a_geometry_sweep"
        / "results"
    )
    if not (results / "manifest.json").is_file():
        pytest.skip("real experiment bundle has not been generated yet")
    loaded = validate_experiment_bundle(results)
    assert loaded["dataset"]["classification"] == CLASSIFICATION
    assert loaded["dataset"]["summary"]["failed_count"] == 0
