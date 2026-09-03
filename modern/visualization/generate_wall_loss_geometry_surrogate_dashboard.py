"""Generate the offline wall-loss geometry surrogate v1 results dashboard.

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/wall_loss_geometry_surrogate_v1`` (or from its committed
protocol for verbatim strings). The generator verifies every file of the bundle
against ``results/manifest.json`` (byte SHA-256 and sizes), recomputes the
headline metrics from the per-design assessment records and refuses to render on
any inconsistency. It emits no wall-clock timestamps or machine paths, so
identical inputs produce identical bytes.

Classification carried everywhere:
``SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`` — a surrogate of a
SCREENING dataset; not physical-orbit evidence; not a performance model. The
recorded terminal state (accepted or rejected) is shown as recorded.
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
MODERN = HERE.parent
EXPERIMENT = MODERN / "experiments" / "wall_loss_geometry_surrogate_v1"
RESULTS = EXPERIMENT / "results"
TEMPLATE_PATH = HERE / "wall-loss-geometry-surrogate-v1.template.html"
DEFAULT_OUTPUT = HERE / "wall-loss-geometry-surrogate-v1.html"

SCHEMA = "cft-revival.wall-loss-geometry-surrogate-v1-dashboard/1.0.0"
CLASSIFICATION = "SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
SOURCE_CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
MAX_HTML_BYTES = 1_200_000
ALLOWED_STATES = {"accepted_result", "assessment_rejection"}
OUTPUTS = ("p_wall_cell1", "p_wall_cell2", "p_wall_cell3", "p_wall_cell4", "p_wall_pooled", "p_reflect_pooled")
CELLS = OUTPUTS[:4]
GATED = OUTPUTS[:5]


# --------------------------------------------------------------------------- #
# Strict loading and bundle verification
# --------------------------------------------------------------------------- #
def _load_json_bytes(raw: bytes, label: str) -> Any:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {label}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=closed_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = _load_json_bytes(path.read_bytes(), label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def verify_bundle(results: Path) -> dict[str, Any]:
    """Byte-verify every manifest entry; return identity facts (state as recorded)."""

    manifest = _load_object(results / "manifest.json", "manifest.json")
    if manifest.get("state") not in ALLOWED_STATES:
        raise ValueError(f"bundle state is {manifest.get('state')!r}, not a completed assessment")
    hashes: dict[str, str] = {}
    verified = 0
    for entry in manifest["artifacts"]:
        if entry.get("type") != "file":
            continue
        path = results / entry["path"]
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != entry["byte_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {entry['path']}")
        if len(raw) != entry["bytes"]:
            raise ValueError(f"size mismatch for {entry['path']}")
        hashes[entry["path"]] = digest
        verified += 1
    terminal_raw = (results / "terminal.json").read_bytes()
    lock_raw = (results / "execution-lock.json").read_bytes()
    if sha256(terminal_raw).hexdigest() != manifest["terminal_byte_sha256"]:
        raise ValueError("terminal.json does not match the manifest")
    if sha256(lock_raw).hexdigest() != manifest["lock_byte_sha256"]:
        raise ValueError("execution-lock.json does not match the manifest")
    lock = _load_json_bytes(lock_raw, "execution-lock.json")
    terminal = _load_json_bytes(terminal_raw, "terminal.json")
    if terminal["state"] != manifest["state"]:
        raise ValueError("terminal state differs from the manifest state")
    return {
        "manifest_file_sha256": sha256((results / "manifest.json").read_bytes()).hexdigest(),
        "terminal_file_sha256": manifest["terminal_byte_sha256"],
        "lock_file_sha256": manifest["lock_byte_sha256"],
        "experiment_id": manifest["experiment_id"],
        "terminal_state": manifest["state"],
        "preregistration_commit_sha": lock["commit"],
        "execution_command": lock["command"],
        "artifact_count": manifest["artifact_count"],
        "verified_file_count": verified,
        "artifact_hashes": hashes,
    }


def _artifact(results: Path, relative: str) -> dict[str, Any]:
    return _load_object(results / "artifacts" / relative, relative)


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _scope_rows(scope: Mapping[str, Any], metrics_scope: Mapping[str, Any], name: str) -> dict[str, Any]:
    designs = scope["designs"]
    errors: dict[str, list[float]] = {output: [] for output in OUTPUTS}
    covered: dict[str, int] = {output: 0 for output in OUTPUTS}
    rows = []
    for design in designs:
        row = {"case_id": design["case_id"], "short": design["case_id"].split("-")[3], "batch": design["batch"], "L": design["chamber_length_m"], "stages": design["stage_count"], "outputs": {}}
        for output in OUTPUTS:
            item = design["outputs"][output]
            if item["error"] != item["predicted"] - item["truth"]:
                raise ValueError(f"{name} {design['case_id']} {output}: recorded error is not predicted - truth")
            low, high = item["observation_interval"]
            inside = low <= item["truth"] <= high
            if inside != item["covered"]:
                raise ValueError(f"{name} {design['case_id']} {output}: coverage flag disagrees with the interval")
            errors[output].append(item["error"])
            covered[output] += int(inside)
            row["outputs"][output] = {"truth": item["truth"], "pred": item["predicted"], "lo": low, "hi": high, "llo": item["latent_interval"][0], "lhi": item["latent_interval"][1], "n": item["trials"], "covered": inside}
        rows.append(row)
    for output in OUTPUTS:
        recorded = metrics_scope["per_output"][output]
        if abs(recorded["rmse"] - _rmse(errors[output])) > 1e-12:
            raise ValueError(f"{name} {output}: metrics RMSE does not reproduce from the designs")
        if abs(recorded["coverage_observation_interval"] - covered[output] / len(designs)) > 1e-12:
            raise ValueError(f"{name} {output}: metrics coverage does not reproduce from the designs")
    cell_errors = [e for output in CELLS for e in errors[output]]
    if abs(metrics_scope["cells"]["rmse"] - _rmse(cell_errors)) > 1e-12:
        raise ValueError(f"{name}: cell RMSE does not reproduce")
    gated_covered = sum(covered[output] for output in GATED)
    if metrics_scope["gated_coverage"]["covered"] != gated_covered:
        raise ValueError(f"{name}: gated coverage count does not reproduce")
    return {
        "name": name,
        "design_count": len(designs),
        "rows": rows,
        "per_output": {
            output: {
                "rmse": metrics_scope["per_output"][output]["rmse"],
                "floor": metrics_scope["per_output"][output]["binomial_floor"],
                "rmse_corrected": metrics_scope["per_output"][output]["rmse_floor_corrected"],
                "mae": metrics_scope["per_output"][output]["mae"],
                "worst": metrics_scope["per_output"][output]["worst_abs_error"],
                "coverage": metrics_scope["per_output"][output]["coverage_observation_interval"],
                "coverage_latent": metrics_scope["per_output"][output]["coverage_latent_interval"],
            }
            for output in OUTPUTS
        },
        "cells": metrics_scope["cells"],
        "gated_coverage": metrics_scope["gated_coverage"],
        "baselines": {
            baseline: {"pooled": item["p_wall_pooled"], "cells": item["cells_pooled"], "per_output": {output: item[output] for output in GATED}}
            for baseline, item in metrics_scope["baselines"].items()
        },
        "best_baseline_pooled": metrics_scope["best_baseline_pooled"],
        "best_baseline_cells": metrics_scope["best_baseline_cells"],
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
    active_learning = _artifact(results, "active-learning.json")
    partitions = _artifact(results, "partitions.json")
    binding = _artifact(results, "dataset-binding.json")
    predictor = _artifact(results, "predictor.json")
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
    scores = {c: candidates[c]["method_selection_rmse"]["mean_over_outputs"] for c in candidates}
    order = list(protocol["candidates"]["order"])
    if min(order, key=lambda c: (scores[c], order.index(c))) != selection["selected"]:
        raise ValueError("selection does not reproduce from the method-selection scores")
    interpolation = _scope_rows(assessment["interpolation"], metrics["interpolation"], "interpolation")
    extrapolation = _scope_rows(assessment["extrapolation"], metrics["extrapolation"], "extrapolation")
    if interpolation["design_count"] != partitions["counts"]["assessment"] or extrapolation["design_count"] != partitions["counts"]["extrapolation"]:
        raise ValueError("scope design counts differ from the partition")
    headline = campaign["headline"]
    if headline["interpolation_rmse_pooled"] != interpolation["per_output"]["p_wall_pooled"]["rmse"]:
        raise ValueError("headline pooled RMSE differs from the metrics")
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
        },
        "selection": {
            "selected": selection["selected"],
            "criterion": selection["criterion"],
            "order": order,
            "scores": scores,
            "ridge_penalty": selection["ridge_penalty"],
            "candidates": {
                c: {
                    "transform": candidates[c]["transform"],
                    "ms_rmse": candidates[c]["method_selection_rmse"],
                    "fit_rmse": candidates[c]["fit_role_in_sample_rmse"],
                    "fit_seconds": candidates[c]["diagnostics"]["fit_seconds"],
                    "library": candidates[c]["diagnostics"].get("library"),
                    "definition": protocol["candidates"]["definitions"][c],
                }
                for c in order
            },
            "baselines_ms": {b: baselines[b]["method_selection_rmse"] for b in baselines},
            "baseline_definitions": {b: protocol["candidates"]["baselines"][b] for b in baselines},
        },
        "scopes": {"interpolation": interpolation, "extrapolation": extrapolation},
        "outputs": list(OUTPUTS),
        "gated_outputs": list(GATED),
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
            "interpretation": sensitivity["interpretation"],
            "input_names": predictor["inputs"]["names"],
        },
        "active_learning": {k: active_learning[k] for k in ("comparison", "budgets", "pool_size", "initial_count", "acquisition", "model", "active_better_at_every_budget", "gated")},
        "determinism": determinism,
        "no_tautology": {k: tautology[k] for k in ("stored_probabilities_equal_count_ratios", "max_single_input_affine_r2", "ridge_stays_above_binomial_floor", "statement", "passed")},
        "predictor": {
            "path": "modern/experiments/wall_loss_geometry_surrogate_v1/results/artifacts/predictor.json",
            "schema_version": predictor["schema_version"],
            "models": {model_id: {"family": block["family"], "outputs": block["outputs"], "tasks": len(block["task_covariance"]), "rows": len(block["train"]["x"])} for model_id, block in predictor["models"].items()},
            "outputs": predictor["outputs"],
            "interpolation_scope": {k: v for k, v in predictor["interpolation_scope"].items() if k != "fit_role_case_ids"},
            "prediction_rule": predictor["prediction_rule"],
            "known_discontinuities": predictor["inputs"]["known_discontinuities"],
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
    for scope in payload["scopes"].values():
        for row in scope["rows"]:
            for item in row["outputs"].values():
                if not 0.0 <= item["lo"] <= item["pred"] <= item["hi"] <= 1.0 or not 0.0 <= item["truth"] <= 1.0:
                    raise ValueError(f"{row['case_id']}: malformed interval")
    text = json.dumps(payload)
    if "http://" in text or "https://" in text:
        raise ValueError("payload must not reference network resources")


def render_html(payload: Mapping[str, Any], template_path: Path = TEMPLATE_PATH) -> str:
    template = template_path.read_text(encoding="utf-8")
    if template.count("__PAYLOAD_JSON__") != 1:
        raise ValueError("template must contain exactly one payload slot")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace("</", "<\\/")
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
    print(json.dumps({"output": str(arguments.output), "bytes": len(html.encode("utf-8")), "terminal_state": payload["terminal_state"], "status": payload["status"], "headline": payload["headline"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
