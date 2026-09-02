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
from cft_revival.experiment_runtime import (
    AtomicArtifactStore,
    Decision,
    ExecutionAttestation,
    ExperimentRuntime,
    RootPolicy,
    RuntimeCallbacks,
    preflight_result_root,
    validate_bundle,
)
from cft_revival.fields import ARTIFACT_SCHEMA_VERSION
from experiments.cft_wall_cusp_validation_v6.experiment import (
    EXPERIMENT_DIR,
    FOUNDATION_COMMIT,
    PROTOCOL,
    RESULT_ATTRIBUTES,
    _write_callback_json,
    build_case,
    case_definitions,
    held_out_manifest,
    map_policy,
    registrations_for,
    run_callback_summary_preflight,
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


def test_new_family_is_disjoint_from_development_and_v1_v5_accesses() -> None:
    definitions = case_definitions()
    manifest = held_out_manifest()
    excluded = PROTOCOL["held_out_family"]["excluded_accessed_evidence"]
    assert len(definitions) == 8
    assert {item.stage_count for item in definitions} == {5}
    assert {item.pitch_m for item in definitions} == {0.0056, 0.007}
    assert {item.chamber_radius_m for item in definitions} == {0.0107, 0.0111}
    assert not set(manifest.case_ids) & set(CFT_V4_DEVELOPMENT_MANIFEST.case_ids)
    assert not set(manifest.geometry_family_ids) & set(
        CFT_V4_DEVELOPMENT_MANIFEST.geometry_family_ids
    )
    prior_ids = set().union(
        excluded["v1_accessed_case_ids"],
        excluded["v2_accessed_case_ids"],
        excluded["v3_accessed_case_ids"],
        excluded["v4_accessed_case_ids"],
        excluded["v5_accessed_case_ids"],
    )
    assert not prior_ids & set(manifest.case_ids)
    prior_coordinates = {
        tuple(item)
        for key in (
            "v1_accessed_coordinate_tuples",
            "v2_accessed_coordinate_tuples",
            "v3_accessed_coordinate_tuples",
            "v4_accessed_coordinate_tuples",
            "v5_accessed_coordinate_tuples",
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
        assert len(registrations_for(built)) + 1 == 6


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
    assert not report["precanonicalized_callback_payloads"]
    assert report["direct_context_write_json_call_count"] == 1


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

        def write_json(self, relative: str, value: Any) -> dict[str, Any]:
            return self.store.write_json(relative, value)

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
        callback = run_callback_summary_preflight(context)  # type: ignore[arg-type]
        assert callback["status"] == "passed"
        assert callback["resolved_status"] == "resolved"
        assert callback["ambiguous_status"] == "ambiguous"
        assert callback["ambiguous_cell_count"] == 0
        assert callback["ambiguous_orbit_count"] == 0
        assert callback["boundary_diagnostic_count"] == 2
        assert callback["assessment_rejection_passed"]
        raw = store.read_bytes("preflight/callback-summary-matrix.json")
        assert b'"__cft_type__"' not in raw
    finally:
        safe.close()


def test_result_line_endings_are_binary_and_same_stem_paths_are_avoided() -> None:
    assert (EXPERIMENT_DIR / "results/.gitattributes").read_bytes() == b"* -text\n"
    source = (EXPERIMENT_DIR / "experiment.py").read_text(encoding="utf-8")
    assert "preflight/production-fields/" in source
    assert "preflight/production-field-pipeline/" not in source


def test_five_of_six_cusps_finalizes_as_valid_assessment_rejection(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / ".gitattributes").write_bytes(RESULT_ATTRIBUTES)
    attestation = ExecutionAttestation(
        attempt=1,
        commit="a" * 40,
        command="manufactured-v6-callback-finalization",
        device="cpu",
        clean_worktree=True,
    )
    runtime = ExperimentRuntime(
        experiment_id="manufactured-v6-callback-finalization",
        result_root=result_root,
        cache_root=tmp_path / "cache",
        attestation=attestation,
        producer=test_five_of_six_cusps_finalizes_as_valid_assessment_rejection,
        source_root=EXPERIMENT_DIR.parents[1],
        root_policy=RootPolicy(
            approved_placeholders={".gitattributes": RESULT_ATTRIBUTES}
        ),
    )

    def prebundle(context: Any) -> dict[str, Any]:
        return run_callback_summary_preflight(context)

    def development(_context: Any) -> Decision:
        return Decision(True, {"thresholds_frozen": True})

    def assessment(context: Any) -> Decision:
        topology = {
            "detected_cusp_count": 5,
            "expected_cusp_count": 6,
            "cells": [],
            "paths": [],
            "orbits": [],
            "status": "ambiguous",
        }
        outcome = {
            "case_id": "manufactured-five-of-six-cusps",
            "topology": topology,
            "failures": ["WALL_CUSP_UNRESOLVED"],
            "assessment_complete": True,
            "passed": False,
        }
        _write_callback_json(context, "topology/five-of-six.json", topology)
        _write_callback_json(context, "outcomes/five-of-six.json", outcome)
        return Decision(False, {"promotion": False, "outcome": outcome})

    outcome = runtime.run(
        RuntimeCallbacks(prebundle, development, assessment)
    )
    assert outcome.state.value == "assessment_rejection"
    manifest = validate_bundle(
        result_root,
        approved_placeholders={".gitattributes": RESULT_ATTRIBUTES},
    )
    assert manifest["state"] == "assessment_rejection"
    paths = [item["path"] for item in manifest["artifacts"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))


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
