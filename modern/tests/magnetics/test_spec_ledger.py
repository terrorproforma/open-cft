import json
from pathlib import Path

from cft_revival.magnetics import (
    checked_synthetic_smco_like_magnet,
    checked_synthetic_soft_magnetic_curve,
)


ROOT = Path(__file__).parents[2]


def test_material_source_ledger_is_machine_readable_and_complete() -> None:
    ledger = json.loads(
        (ROOT / "spec/magnetics/material-source-ledger-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["schema_version"] == "1.0.0"
    assert "hysteresis loops, minor loops, and loss" in ledger["scope"][
        "explicitly_excluded"
    ]
    equation_ids = {equation["id"] for equation in ledger["constitutive_equations"]}
    assert {
        "MAG-MAT-001",
        "MAG-MAT-002",
        "MAG-MAT-003",
        "MAG-PM-001",
        "MAG-SRC-001",
        "MAG-SRC-002",
        "MAG-IFACE-001",
    } <= equation_ids
    assert ledger["nonlinear_curve_policy"]["hysteresis"] == "out_of_scope"
    assert "interval-local" in ledger["nonlinear_curve_policy"]["inverse"]
    assert "H dB" in ledger["nonlinear_curve_policy"]["energy"]
    assert "2^-1074" in ledger["nonlinear_curve_policy"]["endpoint_oracle"]
    assert ledger["permanent_magnet_representation"]["rule"].startswith("exactly one")
    assert (
        ledger["permanent_magnet_representation"][
            "magnetization_relative_tolerance"
        ]
        < 1.0e-14
    )
    assert "canonical SHA-256" in ledger["solver_handoff"]["requirements"][-1]
    assert "exact infinite boundary" in ledger["open_boundary_policy"]["claim_limit"]


def test_ledger_synthetic_examples_match_runtime_factories() -> None:
    ledger = json.loads(
        (ROOT / "spec/magnetics/material-source-ledger-v1.json").read_text(
            encoding="utf-8"
        )
    )
    datasets = {dataset["dataset_id"]: dataset for dataset in ledger["datasets"]}
    curve = checked_synthetic_soft_magnetic_curve()
    magnet = checked_synthetic_smco_like_magnet()

    curve_data = datasets[curve.material_id]
    assert curve_data["classification"] == "synthetic checked algorithm example"
    assert curve_data["measured"] is False
    assert tuple(curve_data["H_A_per_m"]) == curve.h_a_per_m
    assert tuple(curve_data["B_T"]) == curve.b_t

    magnet_data = datasets[magnet.material_id]["parameters"]
    assert magnet_data["Br_ref_T"] == magnet.remanence_ref_t
    assert magnet_data["Hci_ref_A_per_m"] == magnet.intrinsic_coercivity_ref_a_per_m
    assert magnet_data["alpha_Br_per_K"] == magnet.remanence_temp_coefficient_per_k
    assert magnet_data["alpha_Hci_per_K"] == magnet.coercivity_temp_coefficient_per_k
