"""Post-execution verification of the sealed shakedown record against the FROZEN commit.

``experiment.verify_shakedown_record`` is the PRE-execution gate: it demands that the live
worktree equals the code the shakedown proved, which is exactly what ``prepare`` and the one
``execute`` need (and it stays that way; it is sealed under ``experiment_code_sha256``). After
the terminal bundle exists that demand holds only until the next commit to a shared package -
``cft_revival.experiment_runtime`` moved at ``bb756418`` (descriptor cap + ``recovery.py`` for
the geometry-screening-v2 EMFILE) and the record's ``dependency_source_sha256`` stopped equalling
the live tree although nothing about the evidence changed.

This module asks the post-execution question instead: do the sealed digests describe the code at
the commit the immutable execution lock names? It recomputes them from Git blobs at that commit
using the inventories the records themselves carry, asserts equality, and REPORTS the live
tree's drift (added / removed / changed files) without failing on it. ``strict_live_tree=True``
restores the pre-execution semantics for shakedown-time use. Nothing under ``results/`` or in the
frozen files is read for anything but its recorded content; nothing is written.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file
from cft_revival.provenance import SealedScope, verify_sealed_scopes

from experiments.orbit_wall_loss_geometry_screening_v1.designs import (
    field_pipeline_source_files,
    field_pipeline_source_sha256,
)

from . import experiment as E

EXECUTION_LOCK_PATH = E.RESULTS_ROOT / "execution-lock.json"
SOURCE_BINDING_RELATIVE = "artifacts/source-binding.json"
SEALED_HASH_KEYS = ("experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256")


def frozen_commit() -> str:
    """The commit the immutable execution lock names: the code that produced the bundle."""

    lock = strict_json_file(EXECUTION_LOCK_PATH)
    if lock.get("immutable") is not True or not isinstance(lock.get("commit"), str):
        raise ValueError("execution lock is not an immutable record naming its commit")
    return lock["commit"]


def recorded_source_binding() -> dict[str, Any]:
    """``artifacts/source-binding.json`` of the bundle, checked against its hash sidecar."""

    path = E.RESULTS_ROOT / SOURCE_BINDING_RELATIVE
    data = path.read_bytes()
    sidecar = strict_json_file(path.with_name(path.name + ".sha256.json"))
    if sidecar.get("artifact") != SOURCE_BINDING_RELATIVE or sidecar.get("byte_sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("source-binding artifact disagrees with its sidecar")
    return strict_json_file(path)


def _relative(paths: Any) -> tuple[str, ...]:
    return tuple(path.relative_to(E.MODERN).as_posix() for path in paths)


def sealed_scopes(record: Mapping[str, Any], source_binding: Mapping[str, Any]) -> list[SealedScope]:
    """The three sealed digests with the inventories the records carry and the live tree's values."""

    return [
        SealedScope(
            "experiment_code_sha256",
            f"modern/experiments/{E.EXPERIMENT.name}",
            tuple(record["experiment_code_files"]),
            record["experiment_code_sha256"],
            E.experiment_code_sha256(),
            tuple(E.EXPERIMENT_CODE_FILES),
        ),
        SealedScope(
            "dependency_source_sha256",
            "modern",
            tuple(record["dependency_source_files"]),
            record["dependency_source_sha256"],
            E.dependency_source_sha256(),
            _relative(E.dependency_source_files()),
        ),
        SealedScope(
            "field_pipeline_source_sha256",
            "modern",
            tuple(source_binding["field_pipeline_source_files"]),
            record["field_pipeline_source_sha256"],
            field_pipeline_source_sha256(),
            _relative(field_pipeline_source_files()),
        ),
    ]


def verify_recorded_shakedown(
    value: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    commit: str | None = None,
    strict_live_tree: bool = False,
) -> dict[str, Any]:
    """Verify the shakedown record against the protocol and the frozen commit; report live drift.

    Raises ``ValueError`` naming every failed check when the record does not prove the executed
    protocol/code (or, with ``strict_live_tree``, when the live tree has drifted from the seal).
    """

    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == E.schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False and record.get("outcomes_enter_estimand") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = record.get("protocol_semantic_sha256") == semantic_sha256(value)
    try:
        checks["plan_matches_protocol"] = record.get("shakedown_plan") == E.plan_record(E.shakedown_plan(value))
    except Exception:
        checks["plan_matches_protocol"] = False
    checks["every_design_resolved"] = bool(record.get("design_count")) and record.get("design_count") == record.get("resolved_design_count")
    checks["timing_projection_present"] = isinstance(record.get("timing_projection"), dict) and "projected_wall_seconds_at_pool" in record.get("timing_projection", {})
    source_binding = recorded_source_binding()
    checks["bundle_source_binding_equals_record"] = all(source_binding.get(key) == record.get(key) for key in SEALED_HASH_KEYS)
    checks["bundle_dependency_inventory_equals_record"] = source_binding.get("dependency_source_files") == record.get("dependency_source_files")
    frozen = commit if commit is not None else frozen_commit()
    scope_checks, reports = verify_sealed_scopes(E.REPOSITORY, frozen, sealed_scopes(record, source_binding), strict_live_tree=strict_live_tree)
    checks.update(scope_checks)
    if not all(checks.values()):
        failed = sorted(name for name, ok in checks.items() if not ok)
        raise ValueError(f"shakedown record does not prove the executed protocol/code: {failed}")
    live = {key: report["live"] for key, report in reports.items()}
    return {
        "checks": checks,
        "frozen_commit": reports[SEALED_HASH_KEYS[0]]["commit"],
        "scopes": reports,
        "live_tree": {
            "drift": any(item["drift"] for item in live.values()),
            "drifted": sorted(key for key, item in live.items() if item["drift"]),
            **{f"{key}_current": item["sha256"] for key, item in live.items()},
        },
    }
