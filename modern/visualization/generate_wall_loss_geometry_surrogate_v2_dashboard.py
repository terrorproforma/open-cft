"""Generate the offline wall-loss geometry surrogate v2 results dashboard (with the v1 vs v2 panel).

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/wall_loss_geometry_surrogate_v2`` (or from its committed
protocol for verbatim strings).  The bundle verification, per-design
recomputation and rendering helpers are v1's (imported); v2 adds the derived
feature manifest, the tree baseline, the paired v1 comparison (read from v2's
own ``v1-comparison.json`` artifact, which was produced from v1's hash-bound
assessment), the learning curve and the predictor status.  No wall-clock
timestamps or machine paths: identical inputs produce identical bytes.

Classification carried everywhere:
``SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`` — a surrogate of a
SCREENING dataset; not physical-orbit evidence; not a performance model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_wall_loss_geometry_surrogate_dashboard import (  # noqa: E402
    ALLOWED_STATES,
    CELLS,
    CLASSIFICATION,
    GATED,
    MAX_HTML_BYTES,
    OUTPUTS,
    SOURCE_CLASSIFICATION,
    _artifact,
    _rmse,
    _scope_rows,
    verify_bundle,
)

MODERN = HERE.parent
EXPERIMENT = MODERN / "experiments" / "wall_loss_geometry_surrogate_v2"
RESULTS = EXPERIMENT / "results"
TEMPLATE_PATH = HERE / "wall-loss-geometry-surrogate-v2.template.html"
DEFAULT_OUTPUT = HERE / "wall-loss-geometry-surrogate-v2.html"
SCHEMA = "cft-revival.wall-loss-geometry-surrogate-v2-dashboard/1.0.0"
BASELINES = ("global-mean", "knn-3", "ridge", "gbt")


def _comparison_rows(comparison: Mapping[str, Any], scope_rows: Mapping[str, Any], scope_name: str) -> dict[str, Any]:
    """Re-derive the paired v1/v2 per-output RMSE from the comparison artifact and cross-check against v2's designs."""

    scope = comparison["scopes"][scope_name]
    if scope["identical_designs"] is not True or scope["paired_design_count"] != scope_rows["design_count"]:
        raise ValueError(f"{scope_name}: the v1 comparison is not paired on the identical designs")
    per_output = {}
    for output in OUTPUTS:
        item = scope["per_output"][output]
        v2_errors = [row["outputs"][output]["pred"] - row["outputs"][output]["truth"] for row in scope_rows["rows"]]
        if abs(_rmse(v2_errors) - item["v2_rmse"]) > 1e-12:
            raise ValueError(f"{scope_name} {output}: v2 RMSE in the comparison does not reproduce from the designs")
        if abs(item["v1_rmse"] - item["v1_recorded_rmse"]) > 1e-12:
            raise ValueError(f"{scope_name} {output}: v1 RMSE in the comparison differs from v1's recorded RMSE")
        if item["v2_improves"] != (item["v2_rmse"] < item["v1_rmse"]):
            raise ValueError(f"{scope_name} {output}: v2_improves flag does not reproduce")
        per_output[output] = {"v1": item["v1_rmse"], "v2": item["v2_rmse"], "diff": item["difference_v2_minus_v1"], "closer": item["designs_v2_closer"], "n": item["designs"], "improves": item["v2_improves"]}
    return {
        "per_output": per_output,
        "cells": scope["cells"],
        "coverage_90": scope["coverage_90"],
        "best_baseline_pooled": scope["best_baseline_pooled"],
        "v1_baselines_pooled": scope["v1_baselines_pooled_rmse"],
        "v2_baselines_pooled": scope["v2_baselines_pooled_rmse"],
        "tree_vs_gp": scope["tree_vs_selected_gp"],
        "design_count": scope["paired_design_count"],
    }


def build_payload(results: Path = RESULTS, experiment: Path = EXPERIMENT) -> dict[str, Any]:
    identity = verify_bundle(results)
    campaign = _artifact(results, "campaign-result.json")
    gates = _artifact(results, "gates.json")
    metrics = _artifact(results, "metrics.json")
    assessment = _artifact(results, "assessment.json")
    candidates = _artifact(results, "candidates.json")
    baselines = _artifact(results, "baselines.json")
    selection = _artifact(results, "selection.json")
    calibration = _artifact(results, "calibration.json")
    sensitivity = _artifact(results, "sensitivity.json")
    curve = _artifact(results, "learning-curve.json")
    comparison = _artifact(results, "v1-comparison.json")
    features = _artifact(results, "features.json")
    partitions = _artifact(results, "partitions.json")
    binding = _artifact(results, "dataset-binding.json")
    predictor = _artifact(results, "predictor.json")
    predictor_status = _artifact(results, "predictor-status.json")
    protocol = _artifact(results, "protocol.json")
    determinism = _artifact(results, "determinism.json")
    tautology = _artifact(results, "no-tautology.json")
    contract = _artifact(results, "code-contract.json")
    if campaign["classification"] != CLASSIFICATION or predictor["classification"] != CLASSIFICATION:
        raise ValueError("bundle classification is not the surrogate label")
    if campaign["source_dataset_classification"] != SOURCE_CLASSIFICATION:
        raise ValueError("bundle does not carry the screening source label")
    if campaign["evidentiary"] is not True or campaign["plan_kind"] != "evidentiary":
        raise ValueError("bundle is not the evidentiary plan")
    expected_status = "accepted_surrogate" if identity["terminal_state"] == "accepted_result" else "rejected_surrogate"
    if campaign["status"] != expected_status or campaign["all_binding_gates_passed"] != gates["all_binding_passed"]:
        raise ValueError("campaign status disagrees with the gates and the terminal state")
    if gates["all_binding_passed"] != all(item["passed"] for item in gates["binding"].values()):
        raise ValueError("all_binding_passed does not reproduce from the binding gates")
    if selection["selected"] != campaign["selected_candidate"] != predictor["selected_candidate"]:
        raise ValueError("selected candidate differs between artifacts")
    if predictor_status["mdo_v2_input_status"] != campaign["mdo_v2_input_status"] or predictor_status["status"] != campaign["status"]:
        raise ValueError("predictor status disagrees with the campaign result")
    if (campaign["mdo_v2_input_status"] == "usable_as_mdo_v2_input_with_screening_label") != gates["all_binding_passed"]:
        raise ValueError("mdo_v2_input_status does not follow the gates")
    if predictor_status["predictor_file_sha256"] != identity["artifact_hashes"]["artifacts/predictor.json"]:
        raise ValueError("predictor status hash differs from the manifest")
    scores = {c: candidates[c]["method_selection_rmse"]["mean_over_outputs"] for c in candidates}
    order = list(protocol["candidates"]["order"])
    if min(order, key=lambda c: (scores[c], order.index(c))) != selection["selected"]:
        raise ValueError("selection does not reproduce from the method-selection scores")
    grid = baselines["gbt"]["grid_scores_method_selection"]
    if min(grid, key=lambda g: (g["method_selection_rmse_mean"], g["index"]))["parameters"] != selection["gbt_parameters"]:
        raise ValueError("gbt parameters do not reproduce from the grid scores")
    interpolation = _scope_rows(assessment["interpolation"], metrics["interpolation"], "interpolation")
    extrapolation = _scope_rows(assessment["extrapolation"], metrics["extrapolation"], "extrapolation")
    if interpolation["design_count"] != partitions["counts"]["assessment"] or extrapolation["design_count"] != partitions["counts"]["extrapolation"]:
        raise ValueError("scope design counts differ from the partition")
    if binding["v1_binding"]["passed"] is not True or binding["v1_binding"]["partitions_semantic_sha256"] != protocol["v1_binding"]["partitions_semantic_sha256"]:
        raise ValueError("v1 binding is not recorded as passed")
    if campaign["partition_equals_v1"] is not True:
        raise ValueError("the partition is not recorded as v1's")
    headline = campaign["headline"]
    if headline["interpolation_rmse_pooled"] != interpolation["per_output"]["p_wall_pooled"]["rmse"]:
        raise ValueError("headline pooled RMSE differs from the metrics")
    if headline["v1_pooled_rmse_same_designs"] != comparison["headline"]["v1_pooled_rmse_16_designs"]:
        raise ValueError("headline v1 RMSE differs from the comparison artifact")
    if features["names"] != predictor["inputs"]["names"] or features["derived_not_fitted"] is not True or predictor["inputs"]["derived_not_fitted"] is not True:
        raise ValueError("feature manifest disagrees with the predictor inputs")
    summary = curve["summary"]
    for entry in summary:
        pooled = [run["curve"][[c["size"] for c in run["curve"]].index(entry["size"])]["rmse"]["p_wall_pooled"] for run in curve["runs"]]
        if abs(sum(pooled) / len(pooled) - entry["pooled_rmse_mean"]) > 1e-12:
            raise ValueError("learning-curve summary does not reproduce from the runs")
    protocol_text = (experiment / "protocol.json").read_bytes().replace(b"\r\n", b"\n")
    gate_rows = []
    for name, item in gates["binding"].items():
        gate_rows.append({"name": name, "kind": "binding", "passed": item["passed"], "value": item.get("value"), "threshold": item.get("threshold", item.get("ratio_minimum", item.get("interval"))), "extra": {k: v for k, v in item.items() if k in ("raw", "floor", "best_baseline", "best_baseline_rmse", "covered", "count")}})
    for name, item in gates["reported_not_binding"].items():
        if isinstance(item, dict) and "passed" in item:
            gate_rows.append({"name": name, "kind": "reported", "passed": item["passed"], "value": item.get("value"), "threshold": item.get("threshold", item.get("ratio_minimum", item.get("interval"))), "extra": {k: v for k, v in item.items() if k in ("raw", "best_baseline")}})
    payload = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "source_classification": SOURCE_CLASSIFICATION,
        "classification_statement": protocol["classification_statement"],
        "claim_boundary": protocol["claim_boundary"],
        "terminal_state": identity["terminal_state"],
        "status": campaign["status"],
        "mdo_v2_input_status": campaign["mdo_v2_input_status"],
        "rejection_diagnosis": campaign["rejection_diagnosis"],
        "identity": {
            **identity,
            "protocol_file_sha256_lf": sha256(protocol_text).hexdigest(),
            "source_sha256": contract["source_sha256"],
            "package_versions": contract["observed_package_versions"],
            "dataset_file_sha256": binding["dataset_file_sha256"],
            "manifest_file_sha256_screening": binding["manifest_file_sha256"],
            "screening_result_commit": binding["screening_result_commit"],
            "screening_preregistration_commit": binding["screening_preregistration_commit"],
            "screening_merge_commit": binding["screening_merge_commit"],
            "dataset_git_blob": binding["git"]["dataset_blob_at_result_commit"],
            "v1_result_commit": binding["v1_binding"]["result_commit"],
            "v1_file_sha256": binding["v1_binding"]["file_sha256"],
            "v1_partitions_semantic_sha256": binding["v1_binding"]["partitions_semantic_sha256"],
            "generator_sha256": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "template_sha256": sha256(TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        },
        "headline": headline,
        "gates": {"rows": gate_rows, "all_binding_passed": gates["all_binding_passed"], "structural_all_passed": gates["structural_all_passed"], "decision_basis": gates["decision_basis"], "structural": list(gates["structural_gates"])},
        "partition": {
            "counts": partitions["counts"],
            "stratum_counts": partitions["stratum_counts"],
            "seed_namespace": partitions["seed_namespace"],
            "seed": partitions["seed"],
            "extrapolation_cluster": {k: v for k, v in partitions["extrapolation_cluster"].items() if k != "case_ids"},
            "extrapolation_case_ids": partitions["roles"]["extrapolation"],
            "role_sha256": partitions["role_sha256"],
            "label_access_sequence": gates["binding"]["partition_frozen_single_use"]["label_access_sequence"],
            "equals_v1": campaign["partition_equals_v1"],
        },
        "selection": {
            "selected": selection["selected"],
            "criterion": selection["criterion"],
            "order": order,
            "scores": scores,
            "ridge_penalty": selection["ridge_penalty"],
            "gbt_parameters": selection["gbt_parameters"],
            "gbt_grid": grid,
            "candidates": {
                c: {
                    "transform": candidates[c]["transform"],
                    "ms_rmse": candidates[c]["method_selection_rmse"],
                    "fit_rmse": candidates[c]["fit_role_in_sample_rmse"],
                    "fit_seconds": candidates[c]["diagnostics"]["fit_seconds"],
                    "library": candidates[c]["diagnostics"].get("library"),
                    "definition": protocol["candidates"]["definitions"][c],
                    "mixture": {k: candidates[c]["diagnostics"][k] for k in ("fit_designs_per_stage_count", "served_stage_counts", "fallback_stage_counts") if k in candidates[c]["diagnostics"]},
                }
                for c in order
            },
            "baselines_ms": {b: baselines[b]["method_selection_rmse"] for b in BASELINES},
            "baseline_definitions": {b: protocol["candidates"]["baselines"][b] for b in BASELINES},
        },
        "scopes": {"interpolation": interpolation, "extrapolation": extrapolation},
        "outputs": list(OUTPUTS),
        "gated_outputs": list(GATED),
        "baselines": list(BASELINES),
        "calibration": {
            "variance_scale": calibration["variance_scale"],
            "nominal_probability": calibration["nominal_probability"],
            "fit_sample_count": calibration["fit_sample_count"],
            "standardised_residuals": [item["standardised_residual"] for item in calibration["residuals"]],
            "rank_rule": calibration["rank_rule"],
        },
        "sensitivity": {
            "candidate": sensitivity["candidate"],
            "ard": sensitivity["ard_length_scales"],
            "permutation": {"ranking": sensitivity["permutation_importance"]["ranking"], "increase": sensitivity["permutation_importance"]["increase_by_input"], "baseline_rmse": sensitivity["permutation_importance"]["baseline_rmse"], "repeats": sensitivity["permutation_importance"]["repeats"]},
            "tree": sensitivity["tree_feature_importance"],
            "interpretation": sensitivity["interpretation"],
            "input_names": predictor["inputs"]["names"],
        },
        "features": {
            "names": features["names"],
            "units": features["units"],
            "kinds": features["kinds"],
            "provenance": features["provenance"],
            "excluded": features["excluded_recorded_quantities"],
            "statement": features["statement"],
            "distinct": features["degeneracy"]["distinct_values_per_feature"],
        },
        "comparison": {
            "v1_selected": comparison["v1_selected_candidate"],
            "headline": comparison["headline"],
            "interpolation": _comparison_rows(comparison, interpolation, "interpolation"),
            "extrapolation": _comparison_rows(comparison, extrapolation, "extrapolation"),
        },
        "learning_curve": {
            "sizes": curve["sizes"],
            "seeds": curve["seeds"],
            "summary": summary,
            "runs": [{"seed": run["seed"], "points": [{"size": c["size"], "pooled": c["rmse"]["p_wall_pooled"], "mean": c["rmse"]["mean_over_outputs"], "stages": c["stage_counts"]} for c in run["curve"]]} for run in curve["runs"]],
            "extrapolation": curve["extrapolation"],
            "evaluation_role": curve["evaluation_role"],
            "candidate": curve["candidate"],
        },
        "determinism": determinism,
        "no_tautology": {k: tautology[k] for k in ("stored_probabilities_equal_count_ratios", "max_single_input_affine_r2", "max_single_input_affine_r2_at", "ridge_stays_above_binomial_floor", "statement", "passed")},
        "predictor": {
            "path": "modern/experiments/wall_loss_geometry_surrogate_v2/results/artifacts/predictor.json",
            "schema_version": predictor["schema_version"],
            "models": {model_id: {"family": block["family"], "outputs": block["outputs"], "tasks": len(block["task_covariance"]), "rows": len(block["train"]["x"])} for model_id, block in predictor["models"].items()},
            "outputs": predictor["outputs"],
            "interpolation_scope": {k: v for k, v in predictor["interpolation_scope"].items() if k != "fit_role_case_ids"},
            "prediction_rule": predictor["prediction_rule"],
            "known_discontinuities": predictor["inputs"]["known_discontinuities"],
            "status_note": predictor_status["note"],
        },
        "prior_failures": protocol["claim_boundary"]["prior_surrogate_failures_disclosure"],
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA or payload["classification"] != CLASSIFICATION:
        raise ValueError("payload schema/classification is invalid")
    boundary = payload["claim_boundary"]
    if not (boundary["surrogate_of_screening_dataset"] is True and boundary["not_physical_orbit_evidence"] is True and boundary["not_performance_model"] is True):
        raise ValueError("claim boundary flags are missing")
    if payload["terminal_state"] not in ALLOWED_STATES:
        raise ValueError("terminal state is not a completed assessment")
    if (payload["status"] == "accepted_surrogate") != (payload["terminal_state"] == "accepted_result"):
        raise ValueError("status/terminal state mismatch")
    if (payload["mdo_v2_input_status"] == "usable_as_mdo_v2_input_with_screening_label") != (payload["status"] == "accepted_surrogate"):
        raise ValueError("mdo status / campaign status mismatch")
    for scope in payload["scopes"].values():
        for row in scope["rows"]:
            for item in row["outputs"].values():
                if not 0.0 <= item["lo"] <= item["pred"] <= item["hi"] <= 1.0 or not 0.0 <= item["truth"] <= 1.0:
                    raise ValueError(f"{row['case_id']}: malformed interval")
    if len(payload["features"]["names"]) != 31:
        raise ValueError("feature count is not 31")
    text = json.dumps(payload)
    if "http://" in text or "https://" in text:
        raise ValueError("payload must not reference network resources")
    if any(not math.isfinite(v) for v in _numbers(payload)):
        raise ValueError("payload contains non-finite numbers")


def _numbers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)


def render_html(payload: Mapping[str, Any], template_path: Path = TEMPLATE_PATH) -> str:
    template = template_path.read_text(encoding="utf-8")
    if template.count("__PAYLOAD_JSON__") != 1:
        raise ValueError("template must contain exactly one payload slot")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = template.replace("__PAYLOAD_JSON__", encoded)
    if "__PAYLOAD_JSON__" in html:
        raise ValueError("payload slot was not replaced")
    data = html.encode("utf-8")
    if len(data) > MAX_HTML_BYTES:
        raise ValueError(f"dashboard exceeds {MAX_HTML_BYTES} bytes ({len(data)})")
    return html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    payload = build_payload(arguments.results, arguments.experiment)
    html = render_html(payload)
    arguments.output.write_bytes(html.encode("utf-8"))
    print(json.dumps({"output": str(arguments.output), "bytes": len(html.encode("utf-8")), "terminal_state": payload["terminal_state"], "status": payload["status"], "mdo_v2_input_status": payload["mdo_v2_input_status"], "headline": payload["headline"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
