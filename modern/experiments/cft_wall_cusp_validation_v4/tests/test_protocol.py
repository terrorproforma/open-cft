from __future__ import annotations

import ast
import subprocess

from cft_revival.coupling import (
    CFT_V4_DEVELOPMENT_MANIFEST,
    COUPLING_V4_SCHEMA_VERSION,
)
from cft_revival.fields import ARTIFACT_SCHEMA_VERSION
from experiments.cft_wall_cusp_validation_v4.experiment import (
    EXPERIMENT_DIR,
    FOUNDATION_COMMIT,
    PROTOCOL,
    build_case,
    case_definitions,
    held_out_manifest,
    run_serialization_preflight,
)


def test_protocol_freezes_current_artifact_and_coupling_contracts() -> None:
    assert PROTOCOL["accepted_coupling"]["commit_sha"] == FOUNDATION_COMMIT
    assert PROTOCOL["accepted_coupling"]["record_schema"] == COUPLING_V4_SCHEMA_VERSION
    assert PROTOCOL["field_artifact_pipeline"]["schema_version"] == (
        ARTIFACT_SCHEMA_VERSION
    )
    assert not PROTOCOL["field_artifact_pipeline"]["legacy_v1_1_allowed"]
    assert PROTOCOL["execution"]["shared_runtime"] == (
        "cft_revival.experiment_runtime"
    )
    assert PROTOCOL["execution"]["single_execution"]
    assert PROTOCOL["execution"]["no_patch_or_rerun"]
    assert not PROTOCOL["publication_boundary"]["hardware_validity_is_a_gate"]


def test_new_family_is_disjoint_from_development_and_v1_v3_accesses() -> None:
    definitions = case_definitions()
    manifest = held_out_manifest()
    excluded = PROTOCOL["held_out_family"]["excluded_accessed_evidence"]
    assert len(definitions) == 8
    assert {item.stage_count for item in definitions} == {4, 8}
    assert {item.pitch_m for item in definitions} == {0.0053, 0.0067}
    assert {item.chamber_radius_m for item in definitions} == {0.0097}
    assert not set(manifest.case_ids) & set(CFT_V4_DEVELOPMENT_MANIFEST.case_ids)
    assert not set(manifest.geometry_family_ids) & set(
        CFT_V4_DEVELOPMENT_MANIFEST.geometry_family_ids
    )
    prior_ids = set().union(
        excluded["v1_accessed_case_ids"],
        excluded["v2_accessed_case_ids"],
        excluded["v3_accessed_case_ids"],
    )
    assert not prior_ids & set(manifest.case_ids)
    prior_coordinates = {
        tuple(item)
        for key in (
            "v1_accessed_coordinate_tuples",
            "v2_accessed_coordinate_tuples",
            "v3_accessed_coordinate_tuples",
        )
        for item in excluded[key]
    }
    assert not prior_coordinates & {
        (
            item.stage_count,
            item.pitch_m,
            item.chamber_radius_m,
            item.first_polarity,
        )
        for item in definitions
    }


def test_every_preregistered_geometry_builds_without_solver_access() -> None:
    for definition in case_definitions():
        built = build_case(definition)
        assert built.definition == definition
        assert len(built.sources) == 2 * definition.stage_count


def test_synthetic_serializer_and_orbit_preflight_has_zero_held_out_access() -> None:
    report = run_serialization_preflight()
    assert report["status"] == "passed"
    assert report["prior_validation_held_out_map_access_count"] == 0
    assert report["field_signed_zero_normalized"]
    assert report["field_subnormals_preserved"]
    assert not report["missing_dataclass_types"]
    assert report["orbit_diagnostic_count"] > 0


def test_experiment_uses_shared_runtime_without_bespoke_lifecycle_writes() -> None:
    tree = ast.parse((EXPERIMENT_DIR / "experiment.py").read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "cft_revival.experiment_runtime" in imports
    assert not any(
        module.endswith(".canonical") or module.endswith(".control")
        for module in imports
        if module.startswith("experiments.")
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"write_text", "write_bytes", "mkdir"} & calls


def test_foundation_packages_and_fyp_are_untouched() -> None:
    result = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            FOUNDATION_COMMIT,
            "HEAD",
            "--",
            "modern/src/cft_revival",
            "modern/spec",
            "modern/pyproject.toml",
            "FYP",
        ),
        cwd=EXPERIMENT_DIR.parents[2],
    )
    assert result.returncode == 0
