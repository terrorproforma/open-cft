from __future__ import annotations

from experiments.l1a_geometry_sweep_v2.experiment import (
    OBJECTIVES,
    PARITY_INDICES,
    PROTOCOL,
    build_case,
    dominates,
    nondominated,
    representative_roles,
    sample_designs,
)
from experiments.l1a_geometry_sweep_v2.protocol import (
    PROTOCOL_PATH,
    load_protocol,
    stable_hash,
    verify_sidecar,
)


def test_protocol_is_sealed_complete_and_unambiguous() -> None:
    protocol = load_protocol()
    assert verify_sidecar(PROTOCOL_PATH)
    assert protocol["execution"]["case_count"] == 96
    assert protocol["execution"]["maximum_executions"] == 1
    assert len(PARITY_INDICES) == len(set(PARITY_INDICES)) == 6
    gates = protocol["terminal_acceptance"]["gates"]
    assert protocol["terminal_acceptance"]["requires_zero_case_failures"] is True
    assert [gate["gate_id"] for gate in gates] == [
        "boundary",
        "residual",
        "cpu_cuda_parity",
        "flux_identity",
        "source_representation",
        "topology_confidence",
        "manufacturability",
    ]
    assert len(protocol["representative_policy"]["roles"]) == 5
    assert "never pad" in protocol["representative_policy"]["coalescence"]
    assert "never required or claimed bitwise" in protocol["replay_contract"]["cuda_policy"]
    assert len(OBJECTIVES) == 4
    assert {item["direction"] for item in OBJECTIVES} == {"maximize", "minimize"}


def test_sampling_and_all_geometry_identities_are_deterministic() -> None:
    left = sample_designs()
    right = sample_designs()
    assert len(left) == len(right) == 96
    assert [item.design_id for item in left] == [item.design_id for item in right]
    assert len({item.design_id for item in left}) == 96
    for index, design in enumerate(left):
        assert all(
            variable.lower <= value <= variable.upper
            for variable, value in zip(design.variables, design.values, strict=True)
        )
        case = build_case(design, index)
        assert case.preview.authoritative is False
        assert case.derived["stage_count"] in (3, 4, 5)
        assert case.derived["worst_case_radial_manufacturing_margin_m"] >= 0.0
        assert case.derived["worst_case_axial_manufacturing_margin_m"] >= 0.0
        assert case.case_sha256 == stable_hash(
            {
                "geometry_sha256": case.geometry_sha256,
                "source_sha256": case.source_sha256,
                "config_sha256": case.config_sha256,
            }
        )


def _rank_case(
    case_id: str,
    centre: float,
    mirror: float,
    gradient: float,
    energy: float,
    boundary: float = 0.01,
) -> dict:
    return {
        "case_id": case_id,
        "status": "success",
        "qois": {
            "centreline_mid_abs_bz_t": centre,
            "minimum_mirror_ratio": mirror,
            "stage_gradient_rms_t_per_m": gradient,
            "field_energy_j": energy,
            "boundary_to_peak_ratio": boundary,
        },
    }


def test_tolerant_nondominance_is_deterministic() -> None:
    strong = _rank_case("strong", 2.0, 3.0, 4.0, 1.0)
    weak = _rank_case("weak", 1.0, 2.0, 3.0, 2.0)
    tradeoff = _rank_case("tradeoff", 3.0, 1.0, 5.0, 3.0)
    within_tolerance = _rank_case("within", 2.0 + 1e-14, 3.0, 4.0, 1.0)
    assert dominates(strong, weak)
    assert not dominates(strong, within_tolerance)
    assert not dominates(within_tolerance, strong)
    assert [case["case_id"] for case in nondominated((weak, strong, tradeoff))] == [
        "strong",
        "tradeoff",
    ]


def test_representative_roles_coalesce_without_padding() -> None:
    a = _rank_case("a", 5.0, 2.0, 3.0, 1.0, boundary=0.001)
    b = _rank_case("b", 4.0, 6.0, 3.5, 2.0, boundary=0.002)
    c = _rank_case("c", 3.0, 3.0, 8.0, 3.0, boundary=0.003)
    d = _rank_case("d", 2.0, 4.0, 5.0, 4.0, boundary=0.004)
    roles = representative_roles((a, b, c, d))
    assert len(roles) == 5
    assert {item["case_id"] for item in roles} == {"a", "b", "c"}
    by_role = {item["role"]: item["case_id"] for item in roles}
    assert by_role["lowest-field-energy"] == "a"
    assert by_role["best-boundary-isolation"] == "a"
    assert all(not item["role"].startswith("additional") for item in roles)


def test_protocol_identity_is_part_of_case_config_hash() -> None:
    case = build_case(sample_designs()[0], 0)
    assert PROTOCOL["integrity"]["payload_sha256"]
    assert len(case.config_sha256) == 64
