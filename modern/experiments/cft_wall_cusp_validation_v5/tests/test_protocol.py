from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

from cft_revival.coupling import (
    CanonicalFieldV12Adapter,
    CFT_V4_DEVELOPMENT_MANIFEST,
    COUPLING_V4_SCHEMA_VERSION,
    V4_FIELD_CANONICALIZATION,
)
from cft_revival.experiment_runtime import AtomicArtifactStore, preflight_result_root
from cft_revival.fields import ARTIFACT_SCHEMA_VERSION
from experiments.cft_wall_cusp_validation_v5.experiment import (
    EXPERIMENT_DIR,
    FOUNDATION_COMMIT,
    PROTOCOL,
    build_case,
    case_definitions,
    held_out_manifest,
    map_policy,
    run_production_field_pipeline_preflight,
    run_production_path_static_preflight,
    run_serialization_preflight,
    _runtime_identity,
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
    policy = map_policy()
    assert policy.current_artifact_schema == ARTIFACT_SCHEMA_VERSION
    assert policy.accepted_model_levels == ("L1a",)
    assert policy.validated_migration_adapter_ids == ()
    assert policy.maximum_age_s == 3600.0
    assert CanonicalFieldV12Adapter.version_contract.input_schema_version == (
        ARTIFACT_SCHEMA_VERSION
    )
    assert CanonicalFieldV12Adapter.version_contract.normalized_schema_version == (
        ARTIFACT_SCHEMA_VERSION
    )
    assert not CanonicalFieldV12Adapter.version_contract.is_migration


def test_new_family_is_disjoint_from_development_and_v1_v4_accesses() -> None:
    definitions = case_definitions()
    manifest = held_out_manifest()
    excluded = PROTOCOL["held_out_family"]["excluded_accessed_evidence"]
    assert len(definitions) == 8
    assert {item.stage_count for item in definitions} == {5, 9}
    assert {item.pitch_m for item in definitions} == {0.0054, 0.0069}
    assert {item.chamber_radius_m for item in definitions} == {0.0103}
    assert not set(manifest.case_ids) & set(CFT_V4_DEVELOPMENT_MANIFEST.case_ids)
    assert not set(manifest.geometry_family_ids) & set(
        CFT_V4_DEVELOPMENT_MANIFEST.geometry_family_ids
    )
    prior_ids = set().union(
        excluded["v1_accessed_case_ids"],
        excluded["v2_accessed_case_ids"],
        excluded["v3_accessed_case_ids"],
        excluded["v4_accessed_case_ids"],
    )
    assert not prior_ids & set(manifest.case_ids)
    prior_coordinates = {
        tuple(item)
        for key in (
            "v1_accessed_coordinate_tuples",
            "v2_accessed_coordinate_tuples",
            "v3_accessed_coordinate_tuples",
            "v4_accessed_coordinate_tuples",
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


def test_static_preflight_rejects_implicit_policy_and_legacy_reload_paths() -> None:
    report = run_production_path_static_preflight()
    assert report["status"] == "passed"
    assert report["map_validation_policy_call_count"] == 2
    assert report["legacy_disabled_reload_call_count"] >= 3
    assert not report["implicit_policy_defaults"]
    assert not report["legacy_v1_1_reload_possible"]


def test_actual_production_v12_pipeline_preflight(tmp_path: Path) -> None:
    safe = preflight_result_root(tmp_path / "artifacts")
    store = AtomicArtifactStore(safe)

    class Context:
        def __init__(self) -> None:
            self.store = store
            self.accesses: list[dict[str, Any]] = []

        def before_expensive(
            self,
            operation: str,
            *,
            kind: str,
            details: dict[str, Any],
        ) -> None:
            self.accesses.append(
                {"operation": operation, "kind": kind, "details": details}
            )

        def write_blob(self, relative: str, data: bytes) -> dict[str, Any]:
            return self.store.write_blob(relative, data)

    context = Context()
    try:
        report = run_production_field_pipeline_preflight(
            context,  # type: ignore[arg-type]
            {"closure_semantic_sha256": "a" * 64},
            _runtime_identity(),
        )
        assert report["status"] == "passed"
        assert report["held_out_case_access_count"] == 0
        assert report["field_schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert report["field_canonicalization"] == V4_FIELD_CANONICALIZATION
        assert report["coupling_schema_version"] == COUPLING_V4_SCHEMA_VERSION
        assert report["criterion_version"] == "4.0.0"
        assert all(report["byte_equality_by_role"].values())
        assert report["migration_manifest_hashes"] == (None, None, None)
        assert report["migration_source_artifact_hashes"] == (None, None, None)
        assert len(context.accesses) == 3
        assert all(not item["details"]["held_out"] for item in context.accesses)
    finally:
        safe.close()


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
