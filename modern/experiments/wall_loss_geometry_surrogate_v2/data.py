"""v2 data layer: v1's dataset binding, labels and transforms, plus the binding of v1's frozen partition and result.

v2 fits on DERIVED features (:mod:`.features`) but keeps v1's role assignment
exactly (same seed namespace, seed, strata and counts; ``partitions.json`` bound
by byte hash, Git blob and recomputation), so the sixteen assessment designs and
the ten extrapolation designs are identical to v1's and the two campaigns are
directly comparable.  v1's recorded assessment artifact is bound by hash so the
reported (non-gated) v1-vs-v2 comparison reads exactly the committed v1 numbers.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file

from ..wall_loss_geometry_surrogate_v1 import data as v1d
from ..wall_loss_geometry_surrogate_v1.data import (
    ALL_ROLES,
    EXTRAPOLATION,
    ROLES,
    SOURCE_CLASSIFICATION,
    DatasetBindingError,
    DesignRow,
    binomial_floor,
    build_partition,
    floor_corrected,
    labels_for_role,
    observation_noise_at,
    partition_overlap,
    require_dataset_binding,
    to_working,
    working_to_probability,
)
from .features import FEATURE_NAMES, feature_degeneracy_report, load_feature_rows

__all__ = [
    "ALL_ROLES",
    "EXTRAPOLATION",
    "REPOSITORY",
    "ROLES",
    "SOURCE_CLASSIFICATION",
    "DatasetBindingError",
    "DesignRow",
    "V1BindingError",
    "binomial_floor",
    "build_partition",
    "feature_degeneracy_report",
    "floor_corrected",
    "labels_for_role",
    "load_feature_rows",
    "load_v1_assessment",
    "observation_noise_at",
    "partition_overlap",
    "require_dataset_binding",
    "require_v1_binding",
    "to_working",
    "v1_binding_report",
    "v1_partition",
    "working_to_probability",
]

REPOSITORY = v1d.REPOSITORY


class V1BindingError(RuntimeError):
    """The v1 campaign files on disk are not the committed v1 preregistration / result."""


def _git(*arguments: str) -> str:
    completed = subprocess.run(("git", *arguments), cwd=REPOSITORY, check=True, capture_output=True, text=True, encoding="utf-8")
    return completed.stdout.strip()


def _blob_of(commit: str, relative: str) -> str | None:
    try:
        return _git("rev-parse", f"{commit}:{relative}")
    except subprocess.CalledProcessError:
        return None


def _is_ancestor(commit: str) -> bool | None:
    try:
        completed = subprocess.run(("git", "merge-base", "--is-ancestor", commit, "HEAD"), cwd=REPOSITORY, capture_output=True)
    except OSError:
        return None
    return completed.returncode == 0


def v1_binding_report(spec: Mapping[str, Any], rows: Sequence[DesignRow], *, use_git: bool = True) -> dict[str, Any]:
    """Byte hashes, Git blobs and recomputation of v1's frozen partition and recorded result."""

    checks: dict[str, bool] = {}
    hashes: dict[str, str] = {}
    git: dict[str, Any] = {"used": use_git}
    for key, entry in spec["files"].items():
        path = REPOSITORY / entry["path"]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hashes[key] = digest
        checks[f"{key}_file_sha256"] = digest == entry["sha256"] and len(data) == int(entry["bytes"])
        if use_git:
            blob = _blob_of(spec["result_commit"], entry["path"])
            working = _git("hash-object", str(path))
            git[key] = {"blob_at_result_commit": blob, "blob_working_tree": working}
            checks[f"{key}_git_blob"] = blob == entry["git_blob"] == working
    v1_protocol = strict_json_file(REPOSITORY / spec["files"]["protocol"]["path"])
    v1_partitions = strict_json_file(REPOSITORY / spec["files"]["partitions"]["path"])
    checks["v1_partitions_semantic_sha256"] = semantic_sha256(v1_partitions) == spec["partitions_semantic_sha256"]
    recomputed = build_partition(rows, v1_protocol["partition"])
    checks["v1_partition_recomputed_from_v1_rule"] = semantic_sha256(recomputed) == semantic_sha256(v1_partitions)
    checks["v1_partition_seed"] = (
        v1_protocol["partition"]["seed_namespace"] == spec["partition_seed_namespace"]
        and int(v1_protocol["partition"]["seed"]) == int(spec["partition_seed"])
    )
    v1_result = strict_json_file(REPOSITORY / spec["files"]["campaign_result"]["path"])
    checks["v1_status_recorded"] = v1_result.get("status") == spec["recorded_status"] and v1_result.get("selected_candidate") == spec["recorded_selected_candidate"]
    v1_manifest = strict_json_file(REPOSITORY / spec["files"]["results_manifest"]["path"])
    checks["v1_manifest_state"] = v1_manifest.get("state") == spec["results_manifest_state"]
    if use_git:
        git["result_commit_is_ancestor_of_head"] = _is_ancestor(spec["result_commit"])
        checks["v1_result_commit_is_ancestor"] = git["result_commit_is_ancestor_of_head"] is True
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "file_sha256": hashes,
        "git": git,
        "result_commit": spec["result_commit"],
        "partitions_semantic_sha256": semantic_sha256(v1_partitions),
        "v1_selected_candidate": v1_result.get("selected_candidate"),
        "v1_status": v1_result.get("status"),
    }


def require_v1_binding(spec: Mapping[str, Any], rows: Sequence[DesignRow], *, use_git: bool = True) -> dict[str, Any]:
    report = v1_binding_report(spec, rows, use_git=use_git)
    if not report["passed"]:
        failed = sorted(name for name, ok in report["checks"].items() if not ok)
        raise V1BindingError("v1 binding failed: " + ", ".join(failed))
    return report


def v1_partition(spec: Mapping[str, Any]) -> dict[str, Any]:
    return strict_json_file(REPOSITORY / spec["files"]["partitions"]["path"])


def load_v1_assessment(spec: Mapping[str, Any]) -> dict[str, Any]:
    """v1's per-design predictions and metrics for both scopes (from the committed, hash-bound artifacts)."""

    assessment = strict_json_file(REPOSITORY / spec["files"]["assessment"]["path"])
    metrics = strict_json_file(REPOSITORY / spec["files"]["metrics"]["path"])
    scopes: dict[str, Any] = {}
    for scope in ("interpolation", "extrapolation"):
        block = assessment[scope]
        scopes[scope] = {
            "case_ids": [design["case_id"] for design in block["designs"]],
            "designs": {design["case_id"]: design["outputs"] for design in block["designs"]},
            "per_output_rmse": {name: float(item["rmse"]) for name, item in block["per_output"].items()},
            "cells_rmse": float(block["cells"]["rmse"]),
            "cells_rmse_floor_corrected": float(block["cells"]["rmse_floor_corrected"]),
            "gated_coverage": float(block["gated_coverage"]["coverage"]),
            "best_baseline_pooled": dict(block["best_baseline_pooled"]),
            "baselines": {baseline: {k: float(v) for k, v in scores.items()} for baseline, scores in block["baselines"].items()},
        }
    return {"selected_candidate": metrics["selected_candidate"], "scopes": scopes, "variance_scale": float(metrics["variance_scale"])}
