from __future__ import annotations

import math
from pathlib import Path

import pytest

from cft_revival.fields import validate_field_artifact_file
from experiments.l1a_geometry_sweep_v2.experiment import PROTOCOL, build_case, sample_designs
from experiments.l1a_geometry_sweep_v2.protocol import stable_hash
from experiments.l1a_geometry_sweep_v2.validate import validate_bundle

RESULTS = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "l1a_geometry_sweep_v2"
    / "results"
)


def _bundle() -> dict:
    if not (RESULTS / "manifest.json").is_file():
        pytest.skip("the authorized one-time run has not executed")
    return validate_bundle(RESULTS)


def _comparison(left: float, right: float, definition: dict) -> int:
    tolerance = max(
        definition["absolute_tolerance"],
        definition["relative_tolerance"] * max(abs(left), abs(right), 1e-300),
    )
    if definition["direction"] == "minimize":
        left, right = -left, -right
    if left > right + tolerance:
        return 1
    if right > left + tolerance:
        return -1
    return 0


def _independent_front(cases: list[dict]) -> list[str]:
    successful = sorted(
        (case for case in cases if case["status"] == "success"),
        key=lambda case: case["case_id"],
    )

    def dominates(left: dict, right: dict) -> bool:
        results = [
            _comparison(
                float(left["qois"][objective["name"]]),
                float(right["qois"][objective["name"]]),
                objective,
            )
            for objective in PROTOCOL["objectives"]
        ]
        return all(value >= 0 for value in results) and any(value > 0 for value in results)

    return [
        candidate["case_id"]
        for candidate in successful
        if not any(
            other["case_id"] != candidate["case_id"] and dominates(other, candidate)
            for other in successful
        )
    ]


def _independent_roles(cases: list[dict], front_ids: list[str]) -> list[dict[str, str]]:
    by_id = {case["case_id"]: case for case in cases}
    front = [by_id[case_id] for case_id in front_ids]
    records = []
    for role in PROTOCOL["representative_policy"]["roles"]:
        ordered = sorted(front, key=lambda case: case["case_id"])
        choose = max if role["selection"] == "maximum" else min
        selected = choose(ordered, key=lambda case: float(case["qois"][role["qoi"]]))
        records.append({"role": role["role"], "case_id": selected["case_id"]})
    return records


def _independent_gates(cases: list[dict], parity: list[dict]) -> list[dict]:
    successful = [case for case in cases if case["status"] == "success"]
    records = []
    for definition in PROTOCOL["terminal_acceptance"]["gates"]:
        gate_id = definition["gate_id"]
        failed: list[str] = []
        if gate_id == "cpu_cuda_parity":
            by_id = {item["case_id"]: item for item in parity}
            expected = [
                cases[index]["case_id"]
                for index in PROTOCOL["execution"]["parity_case_indices"]
                if cases[index]["status"] == "success"
            ]
            limits = definition["limits"]
            for case_id in expected:
                item = by_id.get(case_id)
                if item is None or (
                    item["differences"]["psi_scale_relative"] > limits["psi"]
                    or item["differences"]["br_scale_relative"] > limits["br"]
                    or item["differences"]["bz_scale_relative"] > limits["bz"]
                ):
                    failed.append(case_id)
        elif gate_id == "manufacturability":
            for case in successful:
                margin = min(
                    case["derived_geometry"]["worst_case_radial_manufacturing_margin_m"],
                    case["derived_geometry"]["worst_case_axial_manufacturing_margin_m"],
                )
                if margin < definition["limit"]:
                    failed.append(case["case_id"])
        else:
            for case in successful:
                value = float(case["qois"][definition["metric"]])
                failed_gate = (
                    value > definition["limit"]
                    if definition["comparator"] == "<="
                    else value < definition["limit"]
                )
                if failed_gate:
                    failed.append(case["case_id"])
        records.append(
            {
                "gate_id": gate_id,
                "failed_case_ids": sorted(failed),
                "failure_count": len(failed),
                "passed": not failed,
            }
        )
    return records


def test_all_sampling_geometry_source_and_config_identities_recompute_exactly() -> None:
    bundle = _bundle()
    raw = bundle["raw"]
    designs = sample_designs()
    assert raw["sampling_design_ids"] == [design.design_id for design in designs]
    for index, (design, record) in enumerate(zip(designs, raw["cases"], strict=True)):
        if record["status"] != "success":
            continue
        case = build_case(design, index)
        assert record["geometry_sha256"] == case.geometry_sha256
        assert record["source_sha256"] == case.source_sha256
        assert record["config_sha256"] == case.config_sha256
        assert record["case_sha256"] == stable_hash(
            {
                "geometry_sha256": record["geometry_sha256"],
                "source_sha256": record["source_sha256"],
                "config_sha256": record["config_sha256"],
            }
        )


def test_nondominated_set_and_roles_recompute_independently() -> None:
    bundle = _bundle()
    cases = bundle["raw"]["cases"]
    expected_front = _independent_front(cases)
    assert bundle["summary"]["nondominated_case_ids"] == expected_front
    expected_roles = _independent_roles(cases, expected_front)
    assert bundle["summary"]["representative_roles"] == expected_roles
    assert bundle["manifest"]["representative_roles"] == expected_roles
    unique_ids = {item["case_id"] for item in expected_roles}
    assert bundle["summary"]["unique_representative_count"] == len(unique_ids)
    assert {
        item["case_id"] for item in bundle["manifest"]["representative_artifacts"]
    } == unique_ids


def test_zero_failure_condition_and_all_seven_gates_recompute_independently() -> None:
    bundle = _bundle()
    raw, summary = bundle["raw"], bundle["summary"]
    recomputed = _independent_gates(raw["cases"], raw["parity"])
    recorded = [
        {
            "gate_id": item["gate_id"],
            "failed_case_ids": item["failed_case_ids"],
            "failure_count": item["failure_count"],
            "passed": item["passed"],
        }
        for item in summary["terminal_gates"]
    ]
    assert recorded == recomputed
    failed_count = sum(case["status"] == "failure" for case in raw["cases"])
    accepted = failed_count == 0 and all(item["passed"] for item in recomputed)
    assert summary["failed_count"] == failed_count
    assert summary["terminal_status"] == ("ACCEPTED" if accepted else "FAILED")
    assert len(recomputed) == 7


def test_representative_artifacts_reload_and_role_coalescence_is_exact() -> None:
    bundle = _bundle()
    manifest = bundle["manifest"]
    roles_by_case: dict[str, list[str]] = {}
    for role in manifest["representative_roles"]:
        roles_by_case.setdefault(role["case_id"], []).append(role["role"])
    for item in manifest["representative_artifacts"]:
        assert item["roles"] == sorted(roles_by_case[item["case_id"]])
        for field_kind in ("full_field", "downsampled_field"):
            field = item[field_kind]
            validate_field_artifact_file(
                RESULTS / field["path"],
                expected_file_sha256=field["file_sha256"],
                expected_payload_sha256=field["payload_sha256"],
            )
    assert len(manifest["representative_artifacts"]) == len(roles_by_case)


def test_replay_contract_and_environment_are_recorded_without_bitwise_cuda_claim() -> None:
    bundle = _bundle()
    summary = bundle["summary"]
    contract = summary["replay_contract"]
    environment = summary["environment"]
    assert contract == PROTOCOL["replay_contract"]
    assert "never required or claimed bitwise" in contract["cuda_policy"]
    assert contract["artifact_policy"] == "artifact hashes identify this run only"
    assert environment["gpu"]["warp_name"] == "NVIDIA GeForce RTX 5090"
    assert environment["warp"]["version"]
    assert environment["warp"]["cuda_toolkit_version"]
    assert environment["warp"]["cuda_driver_runtime_version"]
    assert environment["gpu"]["driver_version"]
    assert environment["scalar"] == "IEEE-754 binary64"
    assert environment["code_revision"] == summary["preregistration_commit_sha"]
    assert len(bundle["raw"]["parity"]) == 6
    assert all(
        math.isfinite(value)
        for item in bundle["raw"]["parity"]
        for value in item["differences"].values()
    )


def test_no_physical_performance_claims_or_fake_failure_penalties() -> None:
    bundle = _bundle()
    raw = bundle["raw"]
    forbidden = {"thrust", "efficiency", "isp", "specific_impulse"}
    for case in raw["cases"]:
        if case["status"] == "failure":
            assert "qois" not in case
        else:
            assert forbidden.isdisjoint(name.lower() for name in case["qois"])
