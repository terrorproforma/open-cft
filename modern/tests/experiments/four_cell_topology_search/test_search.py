from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.four_cell_topology_search.experiment import (
    AXIAL_INTERVALS,
    FIELD_GATES,
    PROTOCOL_STATUS,
    TOPOLOGY_GATES,
    TOPOLOGY_POLICY,
    build_case,
    load_sealed_json,
    sample_designs,
    serialize_plasma,
    stable_hash,
    validate_bundle,
)
from cft_revival.plasma import (
    PlasmaMultiStartResult,
    PlasmaState,
    SolverOptions,
    XenonGlobalInputs,
    solve_global_discharge,
)

MODERN = Path(__file__).resolve().parents[3]
RESULTS = MODERN / "experiments" / "four_cell_topology_search" / "results"


def _dataset():
    return validate_bundle(RESULTS)["dataset"]


def test_sampling_is_deterministic_unique_and_strict_geometry_valid() -> None:
    first = sample_designs(128)
    second = sample_designs(128)
    assert first == second
    assert len({design.design_id for design in first}) == 128
    cases = [build_case(design, index) for index, design in enumerate(first)]
    assert len(cases) == 128
    assert {case.derived["stage_count"] for case in cases} == {4, 5, 6}
    assert all(
        case.derived["source_polarities"][::2]
        == [
            case.derived["source_polarities"][0] * (-1 if index % 2 else 1)
            for index in range(case.derived["stage_count"])
        ]
        for case in cases
    )
    assert all(
        all(
            outer == -inner
            for inner, outer in zip(
                case.derived["source_polarities"][::2],
                case.derived["source_polarities"][1::2],
                strict=True,
            )
        )
        for case in cases
    )


def test_gates_are_derived_from_direct_four_segment_semantics() -> None:
    dataset = _dataset()
    declaration = dataset["declared_gates"]
    assert declaration["field"] == FIELD_GATES
    assert declaration["topology"] == TOPOLOGY_GATES
    assert declaration["topology_policy"]["allow_boundary_minima_as_cusps"] is False
    assert declaration["topology_policy"]["minimum_candidate_confidence"] == (
        TOPOLOGY_POLICY.minimum_candidate_confidence
    )
    compatible = [
        case for case in dataset["cases"] if case.get("topology", {}).get("compatible")
    ]
    assert compatible
    for case in compatible:
        assert case["topology"]["segment_count"] == 4
        assert all(case["field_gates"].values())
        assert all(case["topology"]["gates"].values())
        assert len(case["plasma"]) == len(dataset["plasma_policy"]["operating_points"])


def test_selected_topology_has_no_boundary_leakage_or_inverted_fields() -> None:
    dataset = _dataset()
    for case in dataset["cases"]:
        topology = case.get("topology")
        if not topology or not topology["compatible"]:
            continue
        rows = topology["segments"]
        positions = [row["cusp"]["z_m"] for row in rows]
        assert positions == sorted(positions)
        for row in rows:
            indices = row["cusp"]["sample_indices"]
            assert min(indices) >= 2
            assert max(indices) <= AXIAL_INTERVALS - 2
            assert row["cusp"]["bracket_z_m"][0] > rows[0]["z_start_m"]
            assert row["cusp"]["bracket_z_m"][1] < rows[-1]["z_end_m"]
            assert 0.0 < row["field_ratio_low_to_high"] <= 1.0
            assert row["mirror_ratio_high_to_low"] >= 1.0
            assert row["wall_b_t"] > row["cusp"]["b_magnitude_t"] > 0.0


def test_case_and_coupling_identity_are_hash_bound() -> None:
    dataset = _dataset()
    representatives = {
        item["case_id"]: item for item in dataset["representatives"]
    }
    for case in dataset["cases"]:
        if case["status"] != "field_evaluated":
            continue
        identity = case["identity"]
        assert identity["case_sha256"] == stable_hash(
            {
                "geometry_sha256": identity["geometry_sha256"],
                "source_sha256": identity["source_sha256"],
                "config_sha256": identity["config_sha256"],
            }
        )
        coupling = case["topology"]["coupling_identity"]
        if coupling is not None:
            for key in (
                "record_hash",
                "field_map_hash",
                "artifact_hash",
                "source_hash",
                "source_map_binding_hash",
                "field_model_hash",
                "code_hash",
                "config_hash",
                "adapter_code_hash",
                "coupling_model_hash",
            ):
                assert len(coupling[key]) == 64
        if case["case_id"] in representatives:
            field = RESULTS / representatives[case["case_id"]]["field"]["path"]
            assert hashlib.sha256(field.read_bytes()).hexdigest() == (
                identity["artifact_sha256"]
            )


def test_cpu_warp_parity_subset_passes_declared_gates() -> None:
    dataset = _dataset()
    assert dataset["summary"]["parity_count"] >= 1
    assert dataset["summary"]["parity_failure_count"] == 0
    for item in dataset["parity"]:
        assert item["passed"]
        assert item["differences"]["psi_scale_relative"] <= (
            item["gates"]["psi_scale_relative_max"]
        )
        assert item["differences"]["br_scale_relative"] <= (
            item["gates"]["br_scale_relative_max"]
        )
        assert item["differences"]["bz_scale_relative"] <= (
            item["gates"]["bz_scale_relative_max"]
        )


def test_v1_protocol_and_root_counts_prohibit_publication_claims() -> None:
    dataset = _dataset()
    summary = dataset["summary"]
    assert dataset["protocol_status"] == PROTOCOL_STATUS
    assert not dataset["protocol_status"]["preregistered"]
    assert not dataset["protocol_status"]["valid_for_physical_mirror_claims"]
    assert not dataset["protocol_status"]["valid_for_performance_claims"]
    assert summary["plasma_residual_root_count"] == 6
    assert summary["plasma_residual_root_candidate_count"] == 2
    assert summary["identifiable_state_count"] == 0
    assert summary["performance_publication_count"] == 0
    roots = []
    for case in dataset["cases"]:
        for plasma in case.get("plasma", []):
            if plasma["residual_root_found"]:
                roots.append(plasma)
            assert plasma["identifiable_state"] is False
            assert plasma["identifiability"]["publication_allowed"] is False
    assert len(roots) == 6
    assert all(root["identifiability"]["jacobian_rank"] == 22 for root in roots)
    assert all(root["identifiability"]["state_dimension"] == 25 for root in roots)
    assert all(
        root["outcome_classification"]
        == "non_identifiable_screening_equation_residual_root"
        for root in roots
    )


def test_outcomes_have_only_permitted_diagnostics_and_raw_values_are_audited() -> None:
    dataset = _dataset()
    prohibited = {
        "valid_state",
        "valid_state_published",
        "screening_performance",
        "state",
        "powers",
    }
    for collection in ("cases", "ranking"):
        for case in dataset[collection]:
            for plasma in case.get("plasma", []):
                assert prohibited.isdisjoint(plasma)
                for attempt in plasma["attempts"]:
                    assert prohibited.isdisjoint(attempt)
                    conservation = attempt["conservation_diagnostics"]
                    assert conservation is None or set(conservation) == {"closures"}
    audit = dataset["audit_raw_numerical_data"]
    assert audit["numeric_values_modified"] is False
    assert audit["prior_semantic_labels"] == {
        "performance_publication_count": 6,
        "plasma_converged_candidate_count": 2,
        "plasma_converged_state_count": 6,
    }
    assert len(audit["records"]) == 42
    assert any(record.get("raw_state") is not None for record in audit["records"])
    assert any(
        record.get("raw_power_diagnostics") is not None
        for record in audit["records"]
    )
    assert all(record["numeric_values_modified"] is False for record in audit["records"])


def test_forced_tolerance_failure_cannot_publish_a_state() -> None:
    inputs = XenonGlobalInputs(1000.0, 1.0, (0.060, 0.119, 0.160, 0.254))
    rounded = PlasmaState(
        plasma_potential_v=(14.1, 1000.0, 1000.0, 1000.0),
        electron_temperature_ev=(8.9, 100.1, 43.1, 23.5),
        ionization_source_current_a=(0.008, 0.543, 0.310, 0.157),
        electron_current_a=(0.106, 0.107, 0.637, 0.845, 1.002),
        ion_current_a=(0.894, 0.893, 0.363, 0.155, -0.002),
        cusp_ion_current_a=(0.007, 0.013, 0.102),
    )
    failed = solve_global_discharge(
        inputs,
        rounded,
        options=SolverOptions(max_iterations=2, residual_tolerance=1.0e-12),
    )
    serialized = serialize_plasma(
        PlasmaMultiStartResult(
            best=failed,
            attempts=(failed,),
            selected_start_index=0,
            residual_floor=failed.diagnostics.residual_inf_norm,
        )
    )
    assert not serialized["residual_root_found"]
    assert serialized["identifiable_state"] is False
    assert serialized["identifiability"]["publication_allowed"] is False
    assert serialized["attempts"][0]["residual_root_found"] is False
    assert serialized["attempts"][0]["conservation_diagnostics"] is None
    assert len(serialized["attempts"][0]["residual_rows"]) == 28


def test_bundle_replay_and_ranking_are_deterministic() -> None:
    first = validate_bundle(RESULTS)
    second = validate_bundle(RESULTS)
    assert first == second
    dataset = load_sealed_json(RESULTS / "dataset.json")
    manifest = load_sealed_json(RESULTS / "manifest.json")
    assert manifest["dataset_payload_sha256"] == (
        dataset["integrity"]["payload_sha256"]
    )
    assert manifest["protocol_status"] == PROTOCOL_STATUS
    correction = manifest["semantic_correction"]
    assert correction["numerical_simulations_rerun"] is False
    assert correction["numerical_values_modified"] is False
    assert correction["selection_or_ranking_modified"] is False
    assert correction["representative_artifacts_modified"] is False
    assert correction["supersession_required"] == (
        PROTOCOL_STATUS["supersession_required"]
    )
    assert dataset["ranking"] == sorted(
        dataset["ranking"], key=lambda item: item["rank"]
    )
    compatible_ranks = [
        item["rank"] for item in dataset["ranking"] if item["topology"]["compatible"]
    ]
    incompatible_ranks = [
        item["rank"] for item in dataset["ranking"] if not item["topology"]["compatible"]
    ]
    if compatible_ranks and incompatible_ranks:
        assert max(compatible_ranks) < min(incompatible_ranks)
