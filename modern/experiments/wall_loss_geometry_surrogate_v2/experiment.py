"""Preregistered wall-loss geometry surrogate v2: contracts, plans, callbacks and gates.

Classification ``SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS``.  v2 keeps
v1's lifecycle, roles and gates verbatim (the scoring helpers are imported from v1
and hash-bound) and changes the input representation to derived physical features,
adds the per-stage-count GP mixture, a tree baseline, a learning curve and the
reported paired comparison against v1's committed assessment.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file, strict_json_loads
from cft_revival.surrogates import Prediction, VarianceCalibrator

from ..wall_loss_geometry_surrogate_v1 import experiment as v1x
from ..wall_loss_geometry_surrogate_v1.experiment import (
    ALL_OUTPUTS,
    GATED_OUTPUTS,
    SCIENCE_GATES,
    STRUCTURAL_GATES,
    CampaignPlan,
    _access_records,
    _plain,
    plan_record,
    score_baseline_on_rows,
    score_candidate_on_rows,
)
from . import data as d
from . import features as f
from . import models as m
from .predictor import CLASSIFICATION, NOT_USABLE_LABEL, PREDICTOR_SCHEMA, SOURCE_CLASSIFICATION, USABLE_LABEL, Predictor

__all__ = [
    "ALL_OUTPUTS",
    "AUTHORITIES_PATH",
    "CLASSIFICATION",
    "EXPERIMENT",
    "GATED_OUTPUTS",
    "MODERN",
    "PARTITIONS_PATH",
    "PROTOCOL_PATH",
    "REPOSITORY",
    "RESULTS_ROOT",
    "SCIENCE_GATES",
    "SHAKEDOWN_PATH",
    "STRUCTURAL_GATES",
    "CampaignPlan",
    "FrozenAuthority",
    "build_callbacks",
    "code_contract_report",
    "design_sha256",
    "evidentiary_plan",
    "load_frozen_authority",
    "load_rows",
    "plan_partition",
    "plan_record",
    "protocol",
    "protocol_consistency",
    "require_code_contract",
    "require_protocol_consistency",
    "schema",
    "shakedown_disjointness",
    "shakedown_plan",
    "v1_comparison",
    "verify_shakedown_record",
]

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
PARTITIONS_PATH = EXPERIMENT / "partitions.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"

VERSION_TAG = "cft-revival.wall-loss-geometry-surrogate-v2"
PACKAGES = ("torch", "botorch", "gpytorch", "numpy", "scipy", "sklearn")


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/1.0.0"


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    if value["classification"] != CLASSIFICATION:
        raise ValueError("protocol classification must be the surrogate label")
    return value


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Protocol <-> module consistency
# --------------------------------------------------------------------------


def protocol_consistency(value: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["classification"] = value["classification"] == CLASSIFICATION
    checks["source_classification"] = (
        value["dataset"]["classification"] == SOURCE_CLASSIFICATION
        and value["claim_boundary"]["source_dataset_classification"] == SOURCE_CLASSIFICATION
    )
    checks["claim_boundary_flags"] = all(
        value["claim_boundary"][key] is True
        for key in ("surrogate_of_screening_dataset", "not_physical_orbit_evidence", "not_performance_model", "not_p2_qualified")
    )
    checks["candidate_order"] = tuple(value["candidates"]["order"]) == m.CANDIDATE_ORDER
    checks["candidate_transforms"] = all((("logit" in name) == (m.TRANSFORM_OF[name] == "logit")) for name in m.CANDIDATE_ORDER)
    checks["mixture_rule"] = int(value["candidates"]["mixture_rule"]["minimum_fit_designs_per_stage_count"]) == m.MIXTURE_MINIMUM_PER_COUNT
    checks["outputs"] = (
        tuple(value["outputs"]["gated"]) == GATED_OUTPUTS
        and tuple(value["outputs"]["reported_only"]) == ("p_reflect_pooled",)
        and tuple(value["outputs"]["cells"]) == ("gs1-cell-1", "gs1-cell-2", "gs1-cell-3", "gs1-cell-4")
        and all(int(value["outputs"]["trials"][name]) == (128 if "cell" in name else 512) for name in ALL_OUTPUTS)
    )
    checks["inputs"] = (
        tuple(value["inputs"]["names"]) == f.FEATURE_NAMES
        and int(value["inputs"]["count"]) == len(f.FEATURE_NAMES)
        and value["inputs"]["derived_not_fitted"] is True
    )
    counts = value["partition"]["counts"]
    checks["partition_counts"] = (
        all(
            sum(int(counts[stratum][role]) for role in d.ROLES) == int(counts[stratum]["remaining_after_extrapolation"])
            for stratum in ("primary", "extension")
        )
        and all(int(counts["totals"][role]) == sum(int(counts[stratum][role]) for stratum in ("primary", "extension")) for role in d.ROLES)
        and int(counts["totals"]["extrapolation"]) == int(value["partition"]["extrapolation_cluster"]["count"])
        and sum(int(counts["totals"][role]) for role in d.ALL_ROLES) == int(value["dataset"]["design_count"])
        and int(value["partition"]["extrapolation_cluster"]["count"]) == math.ceil(0.10 * int(value["dataset"]["design_count"]))
    )
    checks["partition_inherited_from_v1"] = (
        value["partition"]["inherited_from_v1"] is True
        and value["partition"]["seed_namespace"] == value["v1_binding"]["partition_seed_namespace"]
        and int(value["partition"]["seed"]) == int(value["v1_binding"]["partition_seed"])
        and value["v1_binding"]["files"]["partitions"]["sha256"] == value["v1_binding"]["partitions_semantic_sha256"]
    )
    checks["baselines"] = (
        tuple(value["candidates"]["baselines"])[:4] == m.BASELINE_ORDER
        and [float(p) for p in m.RIDGE_PENALTIES] == [0.0001, 0.001, 0.01, 0.1, 1.0]
        and len(m.GBT_GRID) == 6
    )
    gates = value["gates"]["binding"]
    checks["gate_thresholds"] = (
        value["gates"]["verbatim_from_v1"] is True
        and float(gates["interpolation_rmse_pooled"]["threshold"]) == 0.05
        and float(gates["interpolation_rmse_cells"]["threshold"]) == 0.05
        and float(gates["beats_best_baseline_2x"]["ratio_minimum"]) == 2.0
        and [float(v) for v in gates["coverage_90"]["interval"]] == [0.8, 0.97]
        and float(gates["coverage_90"]["nominal_probability"]) == 0.9 == float(value["candidates"]["calibration"]["nominal_probability"])
        and float(value["gates"]["reported_not_binding"]["extrapolation_rmse_pooled"]["threshold"]) == 0.1
        and value["gates"]["thresholds_tunable_after_execution"] is False
        and "v1_comparison" in value["gates"]["reported_not_binding"]
        and "learning_curve" in value["gates"]["reported_not_binding"]
    )
    checks["gates_verbatim_v1"] = _gates_match_v1(value)
    checks["shakedown_namespace_differs"] = (
        value["shakedown"]["seed_namespace"] != value["partition"]["seed_namespace"]
        and int(value["shakedown"]["seed"]) != int(value["partition"]["seed"])
    )
    scope = value["code_contract"]["source_hash_scope"]
    checks["code_contract_scope"] = (
        all(f"modern/experiments/wall_loss_geometry_surrogate_v2/{name}" in scope for name in ("__init__.py", "features.py", "data.py", "models.py", "predictor.py", "experiment.py", "run.py"))
        and all(f"modern/experiments/wall_loss_geometry_surrogate_v1/{name}" in scope for name in ("data.py", "models.py", "predictor.py", "experiment.py"))
        and "modern/src/cft_revival/surrogates/*.py" in scope
        and "modern/src/cft_revival/experiment_runtime/*.py" in scope
    )
    checks["torch_policy"] = (
        value["candidates"]["determinism"]["torch"]["device"] == "cpu" and int(value["candidates"]["determinism"]["torch"]["threads"]) <= 8
    )
    curve = value["learning_curve"]
    checks["learning_curve"] = (
        curve["enabled"] is True
        and [int(s) for s in curve["sizes"]] == [20, 30, 40, 50]
        and curve["pool_role"] == "fit"
        and curve["evaluation_role"] == "method-selection"
        and value["active_learning_addon"]["enabled"] is False
    )
    checks["packages_declared"] = tuple(sorted(value["code_contract"]["package_versions"])) == tuple(sorted(PACKAGES))
    return checks


def _gates_match_v1(value: Mapping[str, Any]) -> bool:
    """The binding science thresholds equal v1's committed protocol thresholds."""

    try:
        v1_protocol = strict_json_file(REPOSITORY / value["v1_binding"]["files"]["protocol"]["path"])
    except Exception:
        return False
    ours = value["gates"]["binding"]
    theirs = v1_protocol["gates"]["binding"]
    return (
        float(ours["interpolation_rmse_pooled"]["threshold"]) == float(theirs["interpolation_rmse_pooled"]["threshold"])
        and float(ours["interpolation_rmse_cells"]["threshold"]) == float(theirs["interpolation_rmse_cells"]["threshold"])
        and float(ours["beats_best_baseline_2x"]["ratio_minimum"]) == float(theirs["beats_best_baseline_2x"]["ratio_minimum"])
        and list(ours["coverage_90"]["interval"]) == list(theirs["coverage_90"]["interval"])
        and float(value["gates"]["reported_not_binding"]["extrapolation_rmse_pooled"]["threshold"])
        == float(v1_protocol["gates"]["reported_not_binding"]["extrapolation_rmse_pooled"]["threshold"])
        and set(ours) == set(theirs)
    )


def require_protocol_consistency(value: Mapping[str, Any]) -> dict[str, bool]:
    checks = protocol_consistency(value)
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError("protocol/module mismatch: " + ", ".join(failed))
    return checks


# --------------------------------------------------------------------------
# Code contract
# --------------------------------------------------------------------------


def package_versions() -> dict[str, str]:
    return {name: str(getattr(importlib.import_module(name), "__version__")) for name in PACKAGES}


def code_contract_report(value: Mapping[str, Any]) -> dict[str, Any]:
    sources = v1x.source_hash_report(value)
    declared = dict(value["code_contract"]["package_versions"])
    observed = package_versions()
    python_ok = sys.version_info[:2] == tuple(int(part) for part in value["code_contract"]["python"].split("."))
    return {
        "source_sha256": sources["source_sha256"],
        "source_files": sources["files"],
        "source_line_endings": sources["line_endings"],
        "declared_package_versions": declared,
        "observed_package_versions": observed,
        "package_versions_match": observed == declared,
        "python": sys.version,
        "python_minor_matches": python_ok,
        "matches": observed == declared and python_ok,
    }


def require_code_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    report = code_contract_report(value)
    if not report["matches"]:
        raise ValueError(
            "code contract mismatch: observed "
            f"{report['observed_package_versions']} vs declared {report['declared_package_versions']}; python ok={report['python_minor_matches']}"
        )
    return report


# --------------------------------------------------------------------------
# Plans and the frozen (inherited) partition
# --------------------------------------------------------------------------


def evidentiary_plan(value: Mapping[str, Any]) -> CampaignPlan:
    return CampaignPlan("evidentiary", value["partition"]["seed_namespace"], int(value["partition"]["seed"]), True)


def shakedown_plan(value: Mapping[str, Any]) -> CampaignPlan:
    return CampaignPlan("shakedown", value["shakedown"]["seed_namespace"], int(value["shakedown"]["seed"]), False)


def load_rows(value: Mapping[str, Any]) -> tuple[d.DesignRow, ...]:
    return d.load_feature_rows(value["dataset"], value["outputs"])


def plan_partition(value: Mapping[str, Any], plan: CampaignPlan, rows: Sequence[d.DesignRow]) -> dict[str, Any]:
    return d.build_partition(rows, value["partition"], namespace=plan.partition_namespace, seed=plan.partition_seed)


def design_sha256(value: Mapping[str, Any], plan: CampaignPlan, partition: Mapping[str, Any]) -> str:
    return semantic_sha256(
        {
            "plan": plan_record(plan),
            "partition_role_sha256": dict(partition["role_sha256"]),
            "dataset_file_sha256": value["dataset"]["dataset_file_sha256"],
            "v1_partitions_semantic_sha256": value["v1_binding"]["partitions_semantic_sha256"],
            "inputs": list(value["inputs"]["names"]),
            "outputs": list(ALL_OUTPUTS),
        }
    )


def shakedown_disjointness(value: Mapping[str, Any], rows: Sequence[d.DesignRow]) -> dict[str, Any]:
    evidentiary = plan_partition(value, evidentiary_plan(value), rows)
    shakedown = plan_partition(value, shakedown_plan(value), rows)
    overlap = d.partition_overlap(evidentiary, shakedown)
    namespace_differs = (
        value["shakedown"]["seed_namespace"] != value["partition"]["seed_namespace"]
        and int(value["shakedown"]["seed"]) != int(value["partition"]["seed"])
    )
    inherits_v1 = semantic_sha256(evidentiary) == value["v1_binding"]["partitions_semantic_sha256"]
    proven = bool(namespace_differs and not overlap["identical"] and evidentiary["role_sha256"] != shakedown["role_sha256"] and inherits_v1)
    return {
        "proven": proven,
        "namespace_differs": namespace_differs,
        "evidentiary_partition_equals_v1": inherits_v1,
        "role_assignment_identical": overlap["identical"],
        "designs_with_same_role": overlap["designs_with_same_role"],
        "per_role_overlap": overlap["per_role"],
        "extrapolation_cluster_shared_by_construction": True,
        "note": "the extrapolation cluster is a deterministic function of the data (top-decile chamber length) and is therefore identical in both plans and in v1; the four interpolation roles are reshuffled in the shakedown only",
        "shakedown_design_sha256": design_sha256(value, shakedown_plan(value), shakedown),
        "evidentiary_design_sha256": design_sha256(value, evidentiary_plan(value), evidentiary),
    }


# --------------------------------------------------------------------------
# Shakedown record verification
# --------------------------------------------------------------------------


def verify_shakedown_record(value: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = record.get("protocol_semantic_sha256") == semantic_sha256(value)
    try:
        contract = code_contract_report(value)
        checks["source_sha256_current"] = record.get("source_sha256") == contract["source_sha256"]
        checks["package_versions_current"] = record.get("package_versions") == contract["observed_package_versions"] and contract["matches"]
    except Exception:
        checks["source_sha256_current"] = False
        checks["package_versions_current"] = False
    disjointness = record.get("disjointness")
    checks["disjointness_proven"] = isinstance(disjointness, Mapping) and disjointness.get("proven") is True
    try:
        rows = load_rows(value)
        recomputed = shakedown_disjointness(value, rows)
        checks["shakedown_design_sha256_current"] = record.get("shakedown_design_sha256") == recomputed["shakedown_design_sha256"]
        checks["evidentiary_design_sha256_current"] = record.get("evidentiary_design_sha256") == recomputed["evidentiary_design_sha256"]
        checks["disjointness_recomputed"] = recomputed["proven"]
    except Exception:
        checks["shakedown_design_sha256_current"] = False
        checks["evidentiary_design_sha256_current"] = False
        checks["disjointness_recomputed"] = False
    development = record.get("development")
    checks["all_candidates_fitted"] = (
        isinstance(development, Mapping)
        and set(development.get("candidates_fitted") or ()) == set(m.CANDIDATE_ORDER)
        and development.get("selected") in m.CANDIDATE_ORDER
    )
    checks["predictor_replay_passed"] = isinstance(development, Mapping) and development.get("predictor_replay_passed") is True
    checks["determinism_replay_passed"] = isinstance(development, Mapping) and development.get("determinism_replay_passed") is True
    runtime = record.get("runtime")
    checks["runtime_accepted_and_bundle_validated"] = (
        isinstance(runtime, Mapping) and runtime.get("terminal_state") == "accepted_result" and runtime.get("bundle_validated") is True
    )
    gates = record.get("informational_gates")
    checks["gates_evaluated"] = isinstance(gates, Mapping) and isinstance(gates.get("binding"), Mapping) and bool(gates["binding"])
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError("shakedown gate refused: " + ", ".join(failed))
    return checks


@dataclass(frozen=True)
class FrozenAuthority:
    authorities: Mapping[str, Any]
    partitions: Mapping[str, Any]
    shakedown: Mapping[str, Any]
    shakedown_bytes: bytes
    partitions_bytes: bytes


def load_frozen_authority() -> FrozenAuthority:
    return FrozenAuthority(
        strict_json_file(AUTHORITIES_PATH),
        strict_json_file(PARTITIONS_PATH),
        strict_json_file(SHAKEDOWN_PATH),
        SHAKEDOWN_PATH.read_bytes(),
        PARTITIONS_PATH.read_bytes(),
    )


# --------------------------------------------------------------------------
# No-tautology checks (v2: leave-one-out ridge, because 32 coefficients on 50 designs overfit in-sample)
# --------------------------------------------------------------------------


def no_tautology_report(rows: Sequence[d.DesignRow], input_names: Sequence[str], fit_rows: Sequence[d.DesignRow]) -> dict[str, Any]:
    counts_ok = all(row.stored_probabilities[name] == row.counts[name][0] / row.counts[name][1] for row in rows for name in ALL_OUTPUTS)
    affine_max_r2 = 0.0
    affine_argmax = None
    x_all = m.physical_matrix(rows)
    for name in GATED_OUTPUTS:
        y = np.asarray([row.counts[name][0] / row.counts[name][1] for row in rows])
        for column in range(x_all.shape[1]):
            design = np.column_stack([np.ones(len(rows)), x_all[:, column]])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            total = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
            if r2 > affine_max_r2:
                affine_max_r2 = r2
                affine_argmax = {"output": name, "feature": input_names[column]}
    table = m.TrainingTable.build(fit_rows, input_names, GATED_OUTPUTS, {name: fit_rows[0].counts[name][1] for name in GATED_OUTPUTS})
    loo_errors: dict[str, list[float]] = {name: [] for name in GATED_OUTPUTS}
    for index in range(len(fit_rows)):
        held = fit_rows[index]
        others = [row for i, row in enumerate(fit_rows) if i != index]
        sub = m.subset_table(table, others)
        ridge = m.fit_ridge(sub, 1e-6)
        x_held = table.normaliser.transform(np.asarray([held.inputs], dtype=float))
        for name in GATED_OUTPUTS:
            loo_errors[name].append(float(ridge.predict(x_held, name)[0]) - held.counts[name][0] / held.counts[name][1])
    ridge_rmse = {name: m.rmse(loo_errors[name]) for name in GATED_OUTPUTS}
    floors = {name: d.binomial_floor([row.counts[name] for row in fit_rows]) for name in GATED_OUTPUTS}
    ridge_above_floor = all(ridge_rmse[name] > floors[name] for name in GATED_OUTPUTS)
    return {
        "stored_probabilities_equal_count_ratios": counts_ok,
        "max_single_input_affine_r2": affine_max_r2,
        "max_single_input_affine_r2_at": affine_argmax,
        "single_input_affine_r2_below_0p99": affine_max_r2 < 0.99,
        "ridge_leave_one_out_rmse_fit_role": ridge_rmse,
        "ridge_penalty": 1e-6,
        "binomial_floor_fit_role": floors,
        "ridge_stays_above_binomial_floor": ridge_above_floor,
        "statement": "targets are orbit-termination counts from orbit_mc integration in L1a re-solved fields; no algebraic function of the derived features produces them; the leave-one-out ridge check replaces v1's in-sample check because 32 coefficients on 50 designs overfit in-sample",
        "passed": bool(counts_ok and affine_max_r2 < 0.99 and ridge_above_floor),
    }


# --------------------------------------------------------------------------
# v1 comparison (reported, never gated)
# --------------------------------------------------------------------------


def v1_comparison(v1: Mapping[str, Any], scopes: Mapping[str, Mapping[str, Any]], baselines_selected: Mapping[str, Any]) -> dict[str, Any]:
    """Paired v1-vs-v2 errors on the identical designs of each scope."""

    report: dict[str, Any] = {"v1_selected_candidate": v1["selected_candidate"], "gated": False, "scopes": {}}
    for scope_name, v2_scope in scopes.items():
        v1_scope = v1["scopes"][scope_name]
        v2_ids = [design["case_id"] for design in v2_scope["designs"]]
        identical = sorted(v2_ids) == sorted(v1_scope["case_ids"])
        paired = [design for design in v2_scope["designs"] if design["case_id"] in v1_scope["designs"]]
        per_output = {}
        for name in ALL_OUTPUTS:
            v1_errors = []
            v2_errors = []
            wins = 0
            for design in paired:
                v1_item = v1_scope["designs"][design["case_id"]][name]
                v2_item = design["outputs"][name]
                if v1_item["truth"] != v2_item["truth"]:
                    raise ValueError(f"{design['case_id']}: v1 and v2 truths differ for {name}")
                v1_errors.append(float(v1_item["error"]))
                v2_errors.append(float(v2_item["error"]))
                wins += int(abs(v2_item["error"]) < abs(v1_item["error"]))
            v1_rmse = m.rmse(v1_errors)
            v2_rmse = m.rmse(v2_errors)
            if identical and not math.isclose(v1_rmse, v1_scope["per_output_rmse"][name], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"v1 recorded RMSE for {name} does not reproduce from its per-design errors")
            per_output[name] = {
                "v1_rmse": v1_rmse,
                "v1_recorded_rmse": v1_scope["per_output_rmse"][name],
                "v2_rmse": v2_rmse,
                "difference_v2_minus_v1": v2_rmse - v1_rmse,
                "ratio_v1_over_v2": (v1_rmse / v2_rmse) if v2_rmse > 0.0 else 1e300,
                "designs_v2_closer": wins,
                "designs": len(v2_errors),
                "v2_improves": v2_rmse < v1_rmse,
            }
        v1_cells = v1_scope["cells_rmse"] if identical else m.rmse([e for name in m.CELL_OUTPUTS for e in [float(v1_scope["designs"][d_["case_id"]][name]["error"]) for d_ in paired]])
        v2_cells = v2_scope["cells"]["rmse"] if identical else m.rmse([float(d_["outputs"][name]["error"]) for name in m.CELL_OUTPUTS for d_ in paired])
        report["scopes"][scope_name] = {
            "identical_designs": identical,
            "paired_design_count": len(paired),
            "pairing_note": "paired on the identical v1 designs" if identical else "SHAKEDOWN ONLY: the shakedown roles differ from v1's, so the comparison pairs the designs both scopes share; the evidentiary run pairs all designs",
            "design_count": len(v2_ids),
            "per_output": per_output,
            "cells": {"v1_rmse": v1_cells, "v2_rmse": v2_cells, "difference_v2_minus_v1": v2_cells - v1_cells, "v2_improves": v2_cells < v1_cells},
            "coverage_90": {"v1": v1_scope["gated_coverage"], "v2": v2_scope["gated_coverage"]["coverage"]},
            "best_baseline_pooled": {"v1": v1_scope["best_baseline_pooled"], "v2": v2_scope["best_baseline_pooled"]},
            "v1_baselines_pooled_rmse": {b: s["p_wall_pooled"] for b, s in v1_scope["baselines"].items()},
            "v2_baselines_pooled_rmse": {b: s["p_wall_pooled"] for b, s in v2_scope["baselines"].items()},
            "tree_vs_selected_gp": {
                "gbt_pooled_rmse": v2_scope["baselines"]["gbt"]["p_wall_pooled"],
                "gbt_cells_rmse": v2_scope["baselines"]["gbt"]["cells_pooled"],
                "gp_pooled_rmse": v2_scope["per_output"]["p_wall_pooled"]["rmse"],
                "gp_cells_rmse": v2_cells,
                "tree_beats_gp_pooled": v2_scope["baselines"]["gbt"]["p_wall_pooled"] < v2_scope["per_output"]["p_wall_pooled"]["rmse"],
                "tree_beats_gp_cells": v2_scope["baselines"]["gbt"]["cells_pooled"] < v2_cells,
            },
        }
    interpolation = report["scopes"]["interpolation"]
    report["headline"] = {
        "v1_pooled_rmse_16_designs": interpolation["per_output"]["p_wall_pooled"]["v1_rmse"],
        "v2_pooled_rmse_16_designs": interpolation["per_output"]["p_wall_pooled"]["v2_rmse"],
        "v2_improves_pooled": interpolation["per_output"]["p_wall_pooled"]["v2_improves"],
        "v1_cells_rmse": interpolation["cells"]["v1_rmse"],
        "v2_cells_rmse": interpolation["cells"]["v2_rmse"],
        "v2_improves_cells": interpolation["cells"]["v2_improves"],
        "designs_v2_closer_pooled": interpolation["per_output"]["p_wall_pooled"]["designs_v2_closer"],
        "gbt_parameters": {k: v for k, v in baselines_selected.items() if k in ("max_depth", "n_estimators", "learning_rate")},
        "tree_beats_gp_pooled": interpolation["tree_vs_selected_gp"]["tree_beats_gp_pooled"],
    }
    return report


def _rejection_diagnosis(binding: Mapping[str, Any], reported: Mapping[str, Any], curve: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    failed = [name for name, item in binding.items() if not item["passed"]]
    reasons = []
    for name in failed:
        item = binding[name]
        if "value" in item and "threshold" in item:
            reasons.append(f"{name}: {item['value']:.4f} vs threshold {item['threshold']}")
        elif "value" in item and "ratio_minimum" in item:
            reasons.append(f"{name}: ratio {item['value']:.3f} vs minimum {item['ratio_minimum']} (best baseline {item['best_baseline']} at {item['best_baseline_rmse']:.4f})")
        elif "value" in item and "interval" in item:
            reasons.append(f"{name}: {item['value']:.3f} outside {item['interval']}")
        else:
            reasons.append(f"{name}: structural failure")
    extrapolation = curve.get("extrapolation") or {}
    tree = comparison["headline"]
    needs = []
    if extrapolation.get("fitted") and extrapolation.get("designs_needed_for_target") is not None:
        needs.append(f"learning curve: {extrapolation['statement']}")
    elif extrapolation.get("fitted"):
        needs.append(f"learning curve: {extrapolation['statement']}; v3 needs a different model class or target definition, not just more designs")
    if tree.get("tree_beats_gp_pooled"):
        needs.append("the tree baseline beat the selected GP on pooled P(wall): v3 should make a step-capable model (trees / additive GP with categorical kernel) the primary candidate")
    else:
        needs.append("the selected GP beat the tree baseline on pooled P(wall): the remaining error is not the step structure; v3 needs lower label noise (more launches per design) or more designs")
    if "beats_best_baseline_2x" in failed:
        needs.append("the 2x-baseline gate is limited by the small pooled-P variance across designs (a global mean already scores about 0.056); a v3 gate should be stated as floor-corrected RMSE against the binomial floor rather than a ratio to a mean baseline")
    return {"failed_binding_gates": failed, "reasons": reasons, "v3_requirements": needs}


# --------------------------------------------------------------------------
# Runtime callbacks
# --------------------------------------------------------------------------


def build_callbacks(
    value: Mapping[str, Any],
    plan: CampaignPlan,
    *,
    frozen: FrozenAuthority | None,
    collector: dict[str, Any],
) -> RuntimeCallbacks:
    if (plan.kind == "evidentiary") != (frozen is not None):
        raise ValueError("evidentiary runs require frozen authorities; shakedowns forbid them")
    state: dict[str, Any] = {}
    collector.setdefault("plan_kind", plan.kind)
    input_names = tuple(value["inputs"]["names"])
    trials = {name: int(value["outputs"]["trials"][name]) for name in ALL_OUTPUTS}
    threads = int(value["candidates"]["determinism"]["torch"]["threads"])
    torch_seed = int(value["candidates"]["determinism"]["torch"]["seed"])

    def prebundle(context: Any) -> Mapping[str, Any]:
        consistency = require_protocol_consistency(value)
        contract = require_code_contract(value)
        binding = d.require_dataset_binding(value["dataset"])
        rows = load_rows(value)
        v1_binding = d.require_v1_binding(value["v1_binding"], rows)
        degeneracy = d.feature_degeneracy_report(rows)
        if not degeneracy["passed"]:
            raise ValueError("derived features are degenerate: " + ", ".join(degeneracy["failures"]))
        partition = plan_partition(value, plan, rows)
        disjointness = shakedown_disjointness(value, rows)
        if not disjointness["proven"]:
            raise ValueError("shakedown/evidentiary partitions are not disjoint or the evidentiary partition is not v1's")
        if plan.kind == "evidentiary" and semantic_sha256(partition) != value["v1_binding"]["partitions_semantic_sha256"]:
            raise ValueError("evidentiary partition differs from v1's committed partitions.json")
        if frozen is not None:
            authorities = frozen.authorities
            if authorities["protocol_semantic_sha256"] != semantic_sha256(value):
                raise ValueError("protocol semantic authority differs")
            if authorities["source_sha256"] != contract["source_sha256"]:
                raise ValueError("source hash differs from the preregistered authority")
            if authorities["package_versions"] != contract["observed_package_versions"]:
                raise ValueError("package versions differ from the preregistered authority")
            if authorities["partitions_semantic_sha256"] != semantic_sha256(partition):
                raise ValueError("recomputed partition differs from the preregistered partitions.json")
            if semantic_sha256(frozen.partitions) != semantic_sha256(partition):
                raise ValueError("partitions.json differs from the recomputed partition")
            if authorities["evidentiary_design_sha256"] != design_sha256(value, plan, partition):
                raise ValueError("evidentiary design differs from the preregistered authority")
            if authorities["dataset_file_sha256"] != binding["dataset_file_sha256"]:
                raise ValueError("dataset hash differs from the preregistered authority")
            if hashlib.sha256(frozen.shakedown_bytes).hexdigest() != authorities["shakedown_file_sha256"] or semantic_sha256(frozen.shakedown) != authorities["shakedown_semantic_sha256"]:
                raise ValueError("shakedown record differs from the preregistered authority")
            verify_shakedown_record(value, frozen.shakedown)
        features_artifact = {
            **f.feature_manifest(),
            "per_design": [{"case_id": row.case_id, "stage_count": row.stage_count, "batch": row.batch, "features": dict(zip(input_names, row.inputs, strict=True))} for row in rows],
            "degeneracy": degeneracy,
        }
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/campaign-plan.json", plan_record(plan))
        context.write_json("artifacts/code-contract.json", contract)
        context.write_json("artifacts/protocol-consistency.json", consistency)
        context.write_json("artifacts/dataset-binding.json", _plain({**binding, "v1_binding": v1_binding, "feature_degeneracy": degeneracy}))
        context.write_json("artifacts/features.json", _plain(features_artifact))
        context.write_json("artifacts/partitions.json", partition)
        context.write_json(
            "artifacts/runtime.json",
            {
                "generated_at_utc": datetime.now(timezone.utc),
                "python": sys.version,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "package_versions": contract["observed_package_versions"],
                "torch_threads_declared": threads,
            },
        )
        if frozen is not None:
            context.write_json("artifacts/authorities.json", frozen.authorities)
            context.write_blob("artifacts/shakedown.json", frozen.shakedown_bytes)
        else:
            context.write_json(
                "artifacts/shakedown-disclosure.json",
                {"evidentiary": False, "outcomes_enter_estimand": False, "statement": value["shakedown"]["purpose"], "disjointness": disjointness},
            )
        state.update({"rows": rows, "partition": partition, "contract": contract, "binding": binding, "v1_binding": v1_binding})
        collector["prebundle"] = {"source_sha256": contract["source_sha256"], "dataset_file_sha256": binding["dataset_file_sha256"], "partition_sha256": semantic_sha256(partition)}
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "source_sha256": contract["source_sha256"],
            "dataset_file_sha256": binding["dataset_file_sha256"],
            "partition_semantic_sha256": semantic_sha256(partition),
            "partition_equals_v1": semantic_sha256(partition) == value["v1_binding"]["partitions_semantic_sha256"],
        }

    def development(context: Any) -> Decision:
        started = time.perf_counter()
        rows: Sequence[d.DesignRow] = state["rows"]
        partition = state["partition"]
        _log("development: torch cpu probe")
        environment = m.torch_environment(threads)
        context.write_json("artifacts/device-probe.json", environment)
        role_rows: dict[str, tuple[d.DesignRow, ...]] = {}
        for role in ("fit", "method-selection", "calibration"):
            context.before_expensive(f"labels-{role}", kind="label", details={"role": role, "count": len(partition["roles"][role]), "role_sha256": partition["role_sha256"][role]})
            role_rows[role] = d.labels_for_role(rows, partition, role)
        fit_rows = role_rows["fit"]
        selection_rows = role_rows["method-selection"]
        table = m.TrainingTable.build(fit_rows, input_names, ALL_OUTPUTS, trials)
        context.write_json(
            "artifacts/training-table.json",
            _plain(
                {
                    "input_names": list(input_names),
                    "output_names": list(ALL_OUTPUTS),
                    "normaliser": table.normaliser.to_dict(),
                    "fit_case_ids": [row.case_id for row in fit_rows],
                    "fit_inputs_physical": table.physical,
                    "fit_stage_counts": {str(k): sum(row.stage_count == k for row in fit_rows) for k in f.STAGE_COUNTS},
                    "fit_counts": {name: [list(row.counts[name]) for row in fit_rows] for name in ALL_OUTPUTS},
                    "working": {
                        transform: {name: {"value": table.working(name, transform)[0], "noise_variance": table.working(name, transform)[1]} for name in ALL_OUTPUTS}
                        for transform in ("logit", "direct")
                    },
                    "method_selection_case_ids": [row.case_id for row in selection_rows],
                    "calibration_case_ids": [row.case_id for row in role_rows["calibration"]],
                }
            ),
        )
        # ---- candidates -----------------------------------------------------
        candidates: dict[str, m.FittedCandidate] = {}
        candidate_report: dict[str, Any] = {}
        for candidate_id in m.CANDIDATE_ORDER:
            context.before_expensive(f"fit-{candidate_id}", kind="solver", details={"candidate": candidate_id, "fit_rows": len(fit_rows)})
            _log(f"development: fit {candidate_id}")
            fitted = m.fit_candidate(candidate_id, table, threads=threads, seed=torch_seed)
            candidates[candidate_id] = fitted
            candidate_report[candidate_id] = _plain(
                {
                    "transform": fitted.transform,
                    "diagnostics": fitted.diagnostics,
                    "method_selection_rmse": score_candidate_on_rows(fitted, table, selection_rows, GATED_OUTPUTS),
                    "fit_role_in_sample_rmse": score_candidate_on_rows(fitted, table, fit_rows, GATED_OUTPUTS),
                    "blocks": sorted(fitted.blocks),
                }
            )
            context.write_json(f"artifacts/models/{candidate_id}.json", _plain({"candidate": candidate_id, "transform": fitted.transform, "blocks": fitted.blocks, "outputs": fitted.outputs}))
        selected_id = min(m.CANDIDATE_ORDER, key=lambda c: (candidate_report[c]["method_selection_rmse"]["mean_over_outputs"], m.CANDIDATE_ORDER.index(c)))
        selected = candidates[selected_id]
        _log(f"development: selected {selected_id}")
        # ---- baselines (ridge penalty and gbt grid chosen on the method-selection role) ----
        ridge_scores = {penalty: score_baseline_on_rows(m.fit_ridge(table, penalty), table, selection_rows, GATED_OUTPUTS)["mean_over_outputs"] for penalty in m.RIDGE_PENALTIES}
        ridge_penalty = min(m.RIDGE_PENALTIES, key=lambda p: (ridge_scores[p], p))
        _log("development: gbt grid")
        gbt_scores = []
        for index, parameters in enumerate(m.GBT_GRID):
            score = score_baseline_on_rows(m.fit_gbt(table, parameters), table, selection_rows, GATED_OUTPUTS)["mean_over_outputs"]
            gbt_scores.append({"index": index, "parameters": dict(parameters), "method_selection_rmse_mean": score})
        gbt_choice = min(gbt_scores, key=lambda item: (item["method_selection_rmse_mean"], item["index"]))
        baselines = {
            "global-mean": m.fit_global_mean(table),
            "knn-3": m.fit_knn(table, 3),
            "ridge": m.fit_ridge(table, ridge_penalty),
            "gbt": m.fit_gbt(table, gbt_choice["parameters"]),
        }
        baseline_report = {
            baseline_id: {
                "parameters": _plain({k: v for k, v in baseline.parameters.items() if k not in ("weights", "feature_importances")}),
                "method_selection_rmse": score_baseline_on_rows(baseline, table, selection_rows, GATED_OUTPUTS),
            }
            for baseline_id, baseline in baselines.items()
        }
        baseline_report["ridge"]["penalty_scores_method_selection"] = {str(p): s for p, s in ridge_scores.items()}
        baseline_report["ridge"]["weights"] = _plain(baselines["ridge"].parameters["weights"])
        baseline_report["gbt"]["grid_scores_method_selection"] = gbt_scores
        baseline_report["gbt"]["feature_importances"] = _plain(baselines["gbt"].parameters["feature_importances"])
        # ---- calibration ----------------------------------------------------
        calibration_rows = role_rows["calibration"]
        calibration_physical = m.physical_matrix(calibration_rows)
        calibration_normalized = table.normaliser.transform(calibration_physical)
        truths_working = []
        predictions_working = []
        residual_records = []
        for name in GATED_OUTPUTS:
            spec = selected.output_spec(name)
            mean, variance = selected.latent(calibration_normalized, name)
            for row, mu, var in zip(calibration_rows, mean, variance, strict=True):
                y_working, _ = d.to_working(*row.counts[name], spec["transform"])
                total = float(var) + d.observation_noise_at(float(mu), int(spec["trials"]), spec["transform"])
                truths_working.append(y_working)
                predictions_working.append(Prediction(float(mu), total, 0.9))
                residual_records.append({"case_id": row.case_id, "output": name, "working_truth": y_working, "working_mean": float(mu), "total_variance": total, "standardised_residual": (y_working - float(mu)) / math.sqrt(total)})
        calibrator = VarianceCalibrator.fit(truths_working, predictions_working, nominal_probability=0.9)
        calibration = {
            "role": "calibration",
            "nominal_probability": 0.9,
            "variance_scale": calibrator.variance_scale,
            "fit_sample_count": calibrator.fit_sample_count,
            "residuals": residual_records,
            "rank_rule": value["candidates"]["calibration"]["rank_rule"],
        }
        # ---- predictor contract (status decided in assessment; written provisional here) ----
        extrapolation_threshold = float(partition["extrapolation_cluster"]["chamber_length_threshold_m"])
        contract = _plain(
            {
                "schema_version": PREDICTOR_SCHEMA,
                "classification": CLASSIFICATION,
                "source_dataset_classification": SOURCE_CLASSIFICATION,
                "classification_statement": value["classification_statement"],
                "claim_boundary": value["claim_boundary"],
                "experiment_id": value["experiment_id"],
                "plan_kind": plan.kind,
                "evidentiary": plan.kind == "evidentiary",
                "mdo_v2_input_status": NOT_USABLE_LABEL,
                "mdo_v2_input_status_note": "provisional at development time: the final status is written to campaign-result.json after the single-use assessment (usable only if every binding gate passes); a consumer must read campaign-result.json.mdo_v2_input_status",
                "dataset_binding": {key: state["binding"][key] for key in ("dataset_file_sha256", "manifest_file_sha256", "screening_result_commit", "screening_preregistration_commit", "screening_merge_commit")},
                "inputs": {
                    "names": list(input_names),
                    "units": list(f.FEATURE_UNITS),
                    "derived_not_fitted": True,
                    "feature_manifest": f.feature_manifest(),
                    "normaliser": table.normaliser.to_dict(),
                    "known_discontinuities": value["inputs"]["known_discontinuities"],
                },
                "interpolation_scope": {
                    "statement": "features inside the fit-role unit box and realised chamber length (stage_count x stage_pitch_m) below the extrapolation threshold (top-decile chamber lengths were held out); predictions outside carry the extrapolation caveat",
                    "chamber_length_max_m": extrapolation_threshold,
                    "fit_role_case_ids": [row.case_id for row in fit_rows],
                },
                "outputs": selected.outputs,
                "models": selected.blocks,
                "selected_candidate": selected_id,
                "calibration": {"variance_scale": calibrator.variance_scale, "nominal_probability": 0.9, "fit_sample_count": calibrator.fit_sample_count, "applies_to": "total predictive variance (latent + prediction-time binomial noise)"},
                "observation_noise_rule": value["outputs"]["known_noise"]["prediction_time_observation_noise"],
                "prediction_rule": "probability = inverse transform of the working latent mean; latent interval = inverse transform of mean +/- z_0.90 sqrt(scale * latent variance); observation interval = inverse transform of mean +/- z_0.90 sqrt(scale * (latent variance + binomial noise at the predicted mean)); outputs with a dispatch rule route each design to the block of its stage_count feature",
            }
        )
        context.write_json("artifacts/predictor.json", contract)
        predictor = Predictor(strict_json_loads(context.store.read_bytes("artifacts/predictor.json")))
        # ---- predictor replay against the native posterior --------------------
        replay_inputs = np.vstack([table.physical, calibration_physical])
        replay_normalized = table.normaliser.transform(replay_inputs)
        working = predictor.predict_working(replay_inputs.tolist())
        replay_max = 0.0
        for name in ALL_OUTPUTS:
            native_mean, native_variance = m.native_latent(selected, table, replay_inputs, name)
            replay_max = max(replay_max, float(np.max(np.abs(working[name]["mean"] - native_mean))), float(np.max(np.abs(working[name]["variance"] - native_variance))))
        predictor_replay = {"max_abs_difference_working_space": replay_max, "tolerance": 1e-9, "passed": replay_max <= 1e-9, "inputs": int(replay_inputs.shape[0])}
        # ---- determinism replay ----------------------------------------------
        context.before_expensive(f"refit-{selected_id}", kind="solver", details={"candidate": selected_id, "purpose": "determinism replay"})
        refitted = m.fit_candidate(selected_id, table, threads=threads, seed=torch_seed)
        hyper_a = selected.hyperparameter_vector()
        hyper_b = refitted.hyperparameter_vector()
        hyper_rel = float(np.max(np.abs(hyper_a - hyper_b) / np.maximum(np.abs(hyper_a), 1e-300))) if hyper_a.size else 0.0
        pred_max = 0.0
        for name in ALL_OUTPUTS:
            mean_a, var_a = selected.latent(replay_normalized, name)
            mean_b, var_b = refitted.latent(replay_normalized, name)
            pred_max = max(pred_max, float(np.max(np.abs(mean_a - mean_b))), float(np.max(np.abs(var_a - var_b))))
        determinism = {
            "candidate": selected_id,
            "hyperparameters_bit_exact": bool(np.array_equal(hyper_a, hyper_b)),
            "hyperparameters_max_relative_difference": hyper_rel,
            "predictions_bit_exact": pred_max == 0.0,
            "predictions_max_abs_difference_working_space": pred_max,
            "tolerance_abs": 1e-9,
            "tolerance_rel": 1e-9,
            "passed": bool(pred_max <= 1e-9 and hyper_rel <= 1e-9),
        }
        # ---- sensitivity ----------------------------------------------------
        permutation = m.permutation_importance(selected, table, GATED_OUTPUTS, repeats=int(value["sensitivity"]["permutation_importance"]["repeats"]), namespace="wall-loss-geometry-surrogate-v2:permutation")
        sensitivity = {
            "candidate": selected_id,
            "ard_length_scales": m.ard_length_scales(selected, input_names),
            "permutation_importance": permutation,
            "tree_feature_importance": {
                "baseline": "gbt",
                "parameters": gbt_choice["parameters"],
                "mean_over_outputs": baselines["gbt"].parameters["feature_importance_mean_over_outputs"],
                "ranking": baselines["gbt"].parameters["feature_ranking_mean_over_outputs"],
            },
            "interpretation": "sensitivities of the SCREENING dataset's wall-loss probability to DERIVED geometry / field features; not physical sensitivities",
        }
        # ---- learning curve ----------------------------------------------------
        curve_spec = value["learning_curve"]
        context.before_expensive("learning-curve", kind="solver", details={"candidate": selected_id, "sizes": list(curve_spec["sizes"]), "seeds": list(curve_spec["seeds"])})
        _log("development: learning curve")
        curve = m.learning_curve(
            selected_id,
            fit_rows,
            selection_rows,
            input_names,
            GATED_OUTPUTS,
            trials,
            sizes=[int(s) for s in curve_spec["sizes"]],
            seeds=[int(s) for s in curve_spec["seeds"]],
            threads=threads,
            torch_seed=torch_seed,
            namespace="wall-loss-geometry-surrogate-v2:learning-curve",
        )
        curve["evaluation_role"] = curve_spec["evaluation_role"]
        # ---- publish ----------------------------------------------------------
        tautology = no_tautology_report(rows, input_names, fit_rows)
        context.write_json("artifacts/candidates.json", _plain(candidate_report))
        context.write_json("artifacts/baselines.json", _plain(baseline_report))
        context.write_json(
            "artifacts/selection.json",
            _plain(
                {
                    "role": "method-selection",
                    "criterion": value["candidates"]["selection_criterion"],
                    "scores": {c: candidate_report[c]["method_selection_rmse"]["mean_over_outputs"] for c in m.CANDIDATE_ORDER},
                    "per_output_scores": {c: candidate_report[c]["method_selection_rmse"] for c in m.CANDIDATE_ORDER},
                    "baseline_scores": {b: baseline_report[b]["method_selection_rmse"]["mean_over_outputs"] for b in baselines},
                    "selected": selected_id,
                    "ridge_penalty": ridge_penalty,
                    "gbt_parameters": gbt_choice["parameters"],
                    "labels_used": ["fit", "method-selection"],
                }
            ),
        )
        context.write_json("artifacts/calibration.json", _plain(calibration))
        context.write_json("artifacts/determinism.json", _plain(determinism))
        context.write_json("artifacts/predictor-replay-development.json", _plain(predictor_replay))
        context.write_json("artifacts/sensitivity.json", _plain(sensitivity))
        context.write_json("artifacts/learning-curve.json", _plain(curve))
        context.write_json("artifacts/no-tautology.json", _plain(tautology))
        state.update(
            {
                "table": table,
                "selected": selected,
                "selected_id": selected_id,
                "predictor": predictor,
                "baselines": baselines,
                "gbt_parameters": gbt_choice["parameters"],
                "determinism": determinism,
                "predictor_replay": predictor_replay,
                "tautology": tautology,
                "calibration": calibration,
                "role_rows": role_rows,
                "curve": curve,
            }
        )
        accepted = bool(environment["float64_cholesky_probe"] and len(candidates) == len(m.CANDIDATE_ORDER) and predictor_replay["passed"] and determinism["passed"] and tautology["passed"])
        collector["development"] = {
            "seconds": time.perf_counter() - started,
            "accepted": accepted,
            "candidates_fitted": sorted(candidates),
            "selected": selected_id,
            "selection_scores": {c: candidate_report[c]["method_selection_rmse"]["mean_over_outputs"] for c in m.CANDIDATE_ORDER},
            "baseline_selection_scores": {b: baseline_report[b]["method_selection_rmse"]["mean_over_outputs"] for b in baselines},
            "ridge_penalty": ridge_penalty,
            "gbt_parameters": gbt_choice["parameters"],
            "variance_scale": calibrator.variance_scale,
            "predictor_replay_passed": predictor_replay["passed"],
            "determinism_replay_passed": determinism["passed"],
            "no_tautology_passed": tautology["passed"],
            "learning_curve_summary": curve["summary"],
        }
        return Decision(
            accepted,
            {
                "candidates_fitted": len(candidates),
                "selected": selected_id,
                "predictor_replay_passed": predictor_replay["passed"],
                "determinism_replay_passed": determinism["passed"],
                "no_tautology_passed": tautology["passed"],
            },
        )

    def assessment(context: Any) -> Decision:
        started = time.perf_counter()
        rows: Sequence[d.DesignRow] = state["rows"]
        partition = state["partition"]
        table: m.TrainingTable = state["table"]
        predictor: Predictor = state["predictor"]
        selected: m.FittedCandidate = state["selected"]
        baselines = state["baselines"]
        scopes = {}
        for role, scope_name in (("assessment", "interpolation"), ("extrapolation", "extrapolation")):
            context.before_expensive(f"labels-{role}", kind="label", details={"role": role, "count": len(partition["roles"][role]), "role_sha256": partition["role_sha256"][role], "single_use": True})
            scope_rows = d.labels_for_role(rows, partition, role)
            scopes[scope_name] = v1x.scope_assessment(predictor, scope_rows, ALL_OUTPUTS, GATED_OUTPUTS, baselines, table, scope_name)
        interpolation = scopes["interpolation"]
        extrapolation = scopes["extrapolation"]
        # ---- predictor replay on the assessment + extrapolation inputs -------
        assessment_physical = np.vstack([m.physical_matrix(d.labels_for_role(rows, partition, "assessment")), m.physical_matrix(d.labels_for_role(rows, partition, "extrapolation"))])
        working = predictor.predict_working(assessment_physical.tolist())
        replay_max = 0.0
        for name in ALL_OUTPUTS:
            native_mean, native_variance = m.native_latent(selected, table, assessment_physical, name)
            replay_max = max(replay_max, float(np.max(np.abs(working[name]["mean"] - native_mean))), float(np.max(np.abs(working[name]["variance"] - native_variance))))
        predictor_replay = {"max_abs_difference_working_space": replay_max, "tolerance": 1e-9, "passed": replay_max <= 1e-9, "inputs": int(assessment_physical.shape[0])}
        # ---- v1 comparison (v1 artifacts read only now) -------------------------
        context.before_expensive("v1-assessment-artifact", kind="label", details={"role": "v1-comparison", "source": value["v1_binding"]["files"]["assessment"]["path"], "sha256": value["v1_binding"]["files"]["assessment"]["sha256"]})
        comparison = v1_comparison(d.load_v1_assessment(value["v1_binding"]), scopes, state["gbt_parameters"])
        # ---- label access order -----------------------------------------------
        access = _access_records(context)
        label_sequence = [item["details"].get("role") for item in access if item["kind"] == "label"]
        single_use = (
            label_sequence.count("assessment") == 1
            and label_sequence.count("extrapolation") == 1
            and label_sequence.index("assessment") > max(label_sequence.index(r) for r in ("fit", "method-selection", "calibration"))
            and label_sequence.index("v1-comparison") > label_sequence.index("assessment")
        )
        frozen_matches = frozen is None or semantic_sha256(frozen.partitions) == semantic_sha256(partition)
        equals_v1 = semantic_sha256(partition) == value["v1_binding"]["partitions_semantic_sha256"]
        # ---- gates ------------------------------------------------------------
        thresholds = value["gates"]["binding"]
        reported_spec = value["gates"]["reported_not_binding"]
        pooled = interpolation["per_output"]["p_wall_pooled"]
        coverage = interpolation["gated_coverage"]["coverage"]
        low, high = (float(v) for v in thresholds["coverage_90"]["interval"])
        binding = {
            "interpolation_rmse_pooled": {"value": pooled["rmse"], "threshold": float(thresholds["interpolation_rmse_pooled"]["threshold"]), "passed": pooled["rmse"] <= float(thresholds["interpolation_rmse_pooled"]["threshold"])},
            "interpolation_rmse_cells": {"value": interpolation["cells"]["rmse_floor_corrected"], "raw": interpolation["cells"]["rmse"], "floor": interpolation["cells"]["binomial_floor"], "threshold": float(thresholds["interpolation_rmse_cells"]["threshold"]), "passed": interpolation["cells"]["rmse_floor_corrected"] <= float(thresholds["interpolation_rmse_cells"]["threshold"])},
            "beats_best_baseline_2x": {"value": interpolation["best_baseline_pooled"]["ratio_to_surrogate"], "best_baseline": interpolation["best_baseline_pooled"]["baseline"], "best_baseline_rmse": interpolation["best_baseline_pooled"]["rmse"], "ratio_minimum": 2.0, "passed": interpolation["best_baseline_pooled"]["ratio_to_surrogate"] >= 2.0},
            "coverage_90": {"value": coverage, "covered": interpolation["gated_coverage"]["covered"], "count": interpolation["gated_coverage"]["count"], "interval": [low, high], "passed": low <= coverage <= high},
            "no_tautology": {"passed": state["tautology"]["passed"], "report": state["tautology"]},
            "determinism_replay": {"passed": state["determinism"]["passed"], "report": {k: v for k, v in state["determinism"].items() if k != "candidate"}},
            "dataset_binding": {"passed": bool(state["binding"]["passed"] and state["v1_binding"]["passed"]), "checks": state["binding"]["checks"], "v1_checks": state["v1_binding"]["checks"]},
            "partition_frozen_single_use": {
                "passed": bool(single_use and frozen_matches and (plan.kind != "evidentiary" or equals_v1)),
                "label_access_sequence": label_sequence,
                "single_use": single_use,
                "predictor_written_before_assessment": True,
                "frozen_partition_matches": frozen_matches,
                "partition_equals_v1": equals_v1,
            },
            "predictor_contract_replay": {"passed": predictor_replay["passed"] and state["predictor_replay"]["passed"], "assessment_inputs": predictor_replay, "development_inputs": state["predictor_replay"]},
            "code_contract": {"passed": bool(state["contract"]["matches"] and (frozen is None or frozen.authorities["source_sha256"] == state["contract"]["source_sha256"]))},
        }
        ext_pooled = extrapolation["per_output"]["p_wall_pooled"]
        reported = {
            "extrapolation_rmse_pooled": {"value": ext_pooled["rmse"], "threshold": float(reported_spec["extrapolation_rmse_pooled"]["threshold"]), "passed": ext_pooled["rmse"] <= float(reported_spec["extrapolation_rmse_pooled"]["threshold"])},
            "extrapolation_rmse_cells": {"value": extrapolation["cells"]["rmse_floor_corrected"], "raw": extrapolation["cells"]["rmse"], "threshold": 0.1, "passed": extrapolation["cells"]["rmse_floor_corrected"] <= 0.1},
            "extrapolation_coverage_90": {"value": extrapolation["gated_coverage"]["coverage"], "interval": [low, high], "passed": low <= extrapolation["gated_coverage"]["coverage"] <= high},
            "interpolation_rmse_cells_raw": {"value": interpolation["cells"]["rmse"], "threshold": 0.05, "passed": interpolation["cells"]["rmse"] <= 0.05},
            "beats_best_baseline_2x_cells": {"value": interpolation["best_baseline_cells"]["ratio_to_surrogate"], "best_baseline": interpolation["best_baseline_cells"]["baseline"], "passed": interpolation["best_baseline_cells"]["ratio_to_surrogate"] >= 2.0},
            "reflect_output": {"interpolation": interpolation["per_output"]["p_reflect_pooled"], "extrapolation": extrapolation["per_output"]["p_reflect_pooled"]},
            "extrapolation_best_baseline_pooled": extrapolation["best_baseline_pooled"],
            "v1_comparison": {"value": comparison["headline"]["v2_pooled_rmse_16_designs"] - comparison["headline"]["v1_pooled_rmse_16_designs"], "passed": comparison["headline"]["v2_improves_pooled"], "headline": comparison["headline"]},
            "learning_curve": {"summary": state["curve"]["summary"], "extrapolation": state["curve"]["extrapolation"]},
        }
        all_binding = all(item["passed"] for item in binding.values())
        structural_passed = all(binding[name]["passed"] for name in STRUCTURAL_GATES)
        decision_accepted = all_binding if plan.binding_gates else structural_passed
        gates = {
            "binding": binding,
            "reported_not_binding": reported,
            "all_binding_passed": all_binding,
            "structural_gates": list(STRUCTURAL_GATES),
            "structural_all_passed": structural_passed,
            "binding_in_this_plan": plan.binding_gates,
            "decision_basis": "all binding gates" if plan.binding_gates else "structural gates only (shakedown; science gates informational)",
        }
        metrics = {
            "selected_candidate": state["selected_id"],
            "interpolation": {k: v for k, v in interpolation.items() if k != "designs"},
            "extrapolation": {k: v for k, v in extrapolation.items() if k != "designs"},
            "variance_scale": state["calibration"]["variance_scale"],
            "v1_comparison_headline": comparison["headline"],
        }
        if plan.kind == "evidentiary":
            status = "accepted_surrogate" if all_binding else "rejected_surrogate"
            mdo_status = USABLE_LABEL if all_binding else NOT_USABLE_LABEL
        else:
            status = "shakedown_passed" if structural_passed else "shakedown_failed"
            mdo_status = NOT_USABLE_LABEL
        diagnosis = None if all_binding else _rejection_diagnosis(binding, reported, state["curve"], comparison)
        context.write_json("artifacts/assessment.json", _plain({"interpolation": interpolation, "extrapolation": extrapolation}))
        context.write_json("artifacts/v1-comparison.json", _plain(comparison))
        context.write_json("artifacts/metrics.json", _plain(metrics))
        context.write_json("artifacts/gates.json", _plain(gates))
        context.write_json(
            "artifacts/predictor-status.json",
            {
                "predictor_artifact": "artifacts/predictor.json",
                "predictor_file_sha256": hashlib.sha256(context.store.read_bytes("artifacts/predictor.json")).hexdigest(),
                "mdo_v2_input_status": mdo_status,
                "status": status,
                "rule": value["gates"]["acceptance"],
                "note": "predictor.json is written (immutably) in development with a provisional not-usable status; this record, written after the single-use assessment, carries the final status",
            },
        )
        campaign_result = _plain(
            {
                "schema_version": schema("campaign-result"),
                "experiment_id": value["experiment_id"],
                "plan_kind": plan.kind,
                "evidentiary": plan.kind == "evidentiary",
                "status": status,
                "mdo_v2_input_status": mdo_status,
                "classification": CLASSIFICATION,
                "source_dataset_classification": SOURCE_CLASSIFICATION,
                "claim_boundary": value["claim_boundary"]["statement"],
                "selected_candidate": state["selected_id"],
                "gbt_parameters": state["gbt_parameters"],
                "partition_counts": partition["counts"],
                "partition_equals_v1": equals_v1,
                "headline": {
                    "interpolation_rmse_pooled": pooled["rmse"],
                    "interpolation_rmse_pooled_floor_corrected": pooled["rmse_floor_corrected"],
                    "interpolation_rmse_cells_raw": interpolation["cells"]["rmse"],
                    "interpolation_rmse_cells_floor_corrected": interpolation["cells"]["rmse_floor_corrected"],
                    "interpolation_coverage_90": coverage,
                    "best_baseline_pooled": interpolation["best_baseline_pooled"],
                    "gbt_pooled_rmse": interpolation["baselines"]["gbt"]["p_wall_pooled"],
                    "extrapolation_rmse_pooled": ext_pooled["rmse"],
                    "extrapolation_coverage_90": extrapolation["gated_coverage"]["coverage"],
                    "variance_scale": state["calibration"]["variance_scale"],
                    "v1_pooled_rmse_same_designs": comparison["headline"]["v1_pooled_rmse_16_designs"],
                    "v1_cells_rmse_same_designs": comparison["headline"]["v1_cells_rmse"],
                },
                "all_binding_gates_passed": all_binding,
                "structural_all_passed": structural_passed,
                "decision_basis": gates["decision_basis"],
                "binding_gate_results": {name: item["passed"] for name, item in binding.items()},
                "reported_gate_results": {name: item.get("passed") for name, item in reported.items() if "passed" in item},
                "learning_curve": {"summary": state["curve"]["summary"], "extrapolation": state["curve"]["extrapolation"]},
                "rejection_diagnosis": diagnosis,
                "assessment_seconds": time.perf_counter() - started,
            }
        )
        context.write_json("artifacts/campaign-result.json", campaign_result)
        collector["assessment"] = {"gates": gates, "metrics": metrics, "campaign_result": campaign_result, "headline": campaign_result["headline"], "status": status, "seconds": time.perf_counter() - started}
        return Decision(
            decision_accepted,
            {
                "all_binding_gates_passed": all_binding,
                "structural_all_passed": structural_passed,
                "decision_basis": gates["decision_basis"],
                "binding_gate_results": {name: item["passed"] for name, item in binding.items()},
                "selected": state["selected_id"],
                "interpolation_rmse_pooled": pooled["rmse"],
                "coverage_90": coverage,
                "extrapolation_rmse_pooled": ext_pooled["rmse"],
                "v2_improves_pooled_over_v1": comparison["headline"]["v2_improves_pooled"],
            },
        )

    return RuntimeCallbacks(prebundle=prebundle, development=development, assessment=assessment)
