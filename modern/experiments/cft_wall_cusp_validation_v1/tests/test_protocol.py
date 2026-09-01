from __future__ import annotations

import ast
import math
import subprocess

from cft_revival.coupling import CFT_V4_DEVELOPMENT_MANIFEST
from experiments.cft_wall_cusp_validation_v1.experiment import (
    ACCEPTED_COUPLING_COMMIT,
    EXPERIMENT_DIR,
    PROTOCOL,
    PROTOCOL_SEMANTIC_SHA256,
    AdiabaticPathOrbitAdapter,
    _strict_json,
    build_case,
    case_definitions,
    held_out_manifest,
    normalized_text_hash,
    policies_for,
    registrations_for,
    semantic_hash,
    validate_results,
)


def test_protocol_freezes_schema_criterion_and_accepted_commit() -> None:
    assert PROTOCOL["classification"] == "preregistered_held_out_numerical_validation"
    assert PROTOCOL["accepted_coupling"]["commit_sha"] == ACCEPTED_COUPLING_COMMIT
    assert PROTOCOL["accepted_coupling"]["record_schema"].endswith("/4.1.0")
    assert PROTOCOL["accepted_coupling"]["criterion_version"] == "4.0.0"
    assert PROTOCOL["development_evidence"]["role"] == "development_non_validation"
    assert not PROTOCOL["publication_boundary"]["experimental_truth_claim"]
    assert not PROTOCOL["publication_boundary"]["hardware_validation_claim"]
    assert PROTOCOL["execution"]["single_execution"]
    assert PROTOCOL["execution"]["no_patch_or_rerun"]
    assert semantic_hash(_strict_json(EXPERIMENT_DIR / "protocol.json")) == (
        PROTOCOL_SEMANTIC_SHA256
    )


def test_24_case_held_out_family_is_explicit_and_disjoint() -> None:
    definitions = case_definitions()
    manifest = held_out_manifest()
    assert len(definitions) == 24
    assert len(manifest.case_ids) == 24
    assert {item.stage_count for item in definitions} == {4, 6, 8}
    assert {item.pitch_m for item in definitions} == {0.0051, 0.0068}
    assert {item.chamber_radius_m for item in definitions} == {0.0074, 0.0102}
    assert {item.first_polarity for item in definitions} == {-1, 1}
    assert len({item.geometry_id for item in definitions}) == 24
    assert len({item.family_semantic_sha256 for item in definitions}) == 24
    assert not set(manifest.case_ids) & set(CFT_V4_DEVELOPMENT_MANIFEST.case_ids)
    assert not set(manifest.geometry_family_ids) & set(
        CFT_V4_DEVELOPMENT_MANIFEST.geometry_family_ids
    )
    assert PROTOCOL["development_evidence"]["manifest_hash"] == (
        CFT_V4_DEVELOPMENT_MANIFEST.manifest_hash
    )


def test_every_case_builds_strict_geometry_maps_and_registrations() -> None:
    for definition in case_definitions():
        case = build_case(definition)
        assert case.geometry.config_id == definition.geometry_id
        assert len(case.sources) == 2 * definition.stage_count
        assert len(registrations_for(case)) == definition.stage_count
        assert all(
            len(registration.seeds[0].electron_samples) == 3
            for registration in registrations_for(case)
        )
        policy = policies_for(case)
        assert policy["cusp_policy"].minimum_prominence_t > 0.0
        assert policy["trace_policy"].maximum_psi_drift_wb > 0.0
        assert policy["axial_policy"].minimum_mean_axial_fraction > 0.0
        assert policy["uncertainty_model"].coverage_factor == 2.0


def test_manufactured_orbit_adapter_binds_path_sample_and_identity() -> None:
    adapter = AdiabaticPathOrbitAdapter()
    case = build_case(case_definitions()[0])
    sample = registrations_for(case)[0].seeds[0].electron_samples[0]
    path = tuple((0.001 * index, 0.0002 * index**2) for index in range(12))
    claims = adapter.verify_orbit(path, "a" * 64, sample)
    assert claims.path_hash == "a" * 64
    assert claims.sample_id == sample.sample_id
    assert claims.converged
    assert math.isfinite(claims.maximum_mu_relative_variation)
    assert claims.maximum_mu_relative_variation <= sample.maximum_mu_relative_variation
    assert len(claims.adapter_code_hash) == 64
    assert len(claims.orbit_code_hash) == 64
    assert len(claims.orbit_config_hash) == 64
    assert len(claims.convergence_config_hash) == 64


def test_numerical_targets_cover_atomic_v4_gates_and_topology_diagnostics() -> None:
    targets = PROTOCOL["numerical_targets"]
    assert targets["cross_map_cusp_count_agreement"]
    assert targets["cross_map_cell_count_agreement"]
    assert targets["all_registered_paths_wall_connected"]
    assert targets["same_line_extrema_required"]
    assert targets["all_uncertainty_bounds_finite_ordered"]
    assert targets["all_energy_pitch_direction_orbits_adiabatic"]
    assert targets["all_held_out_cases_must_pass_for_promotion"]
    assert set(PROTOCOL["topology_diagnostics"]["reported_separately"]) == {
        "magnetic_null",
        "X",
        "O",
        "degenerate",
    }


def test_code_does_not_import_prior_experiment_proxy_or_v3_records() -> None:
    tree = ast.parse((EXPERIMENT_DIR / "experiment.py").read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("cft_topology_characterization_v1" in module for module in imports)
    assert not any("four_cell_topology_search" in module for module in imports)
    assert "cft_revival.coupling.v3_records" not in imports
    assert "cft_revival.coupling.proxies" not in imports


def test_accepted_dependency_trees_match_exact_commit() -> None:
    result = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            ACCEPTED_COUPLING_COMMIT,
            "HEAD",
            "--",
            "modern/src/cft_revival/coupling",
            "modern/src/cft_revival/fields",
            "modern/src/cft_revival/geometry",
            "modern/src/cft_revival/magnetics",
            "modern/spec",
            "modern/pyproject.toml",
        ),
        cwd=EXPERIMENT_DIR.parents[2],
    )
    assert result.returncode == 0


def test_semantic_hashes_are_newline_safe() -> None:
    assert normalized_text_hash("a\nb\n") == normalized_text_hash("a\r\nb\r\n")
    assert semantic_hash({"b": 2, "a": 1}) == semantic_hash({"a": 1, "b": 2})


def test_result_lifecycle_before_or_after_single_run() -> None:
    results = EXPERIMENT_DIR / "results"
    if not results.exists():
        assert PROTOCOL["execution"]["preflight"] == "manufactured tests only before execution"
    else:
        validated = validate_results(results)
        assert validated["dataset"]["summary"]["case_count"] == 24
        assert validated["dataset"]["summary"]["map_count"] == 72

