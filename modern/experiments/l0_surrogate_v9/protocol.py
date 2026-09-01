"""Prospective v9 validation of exact-leading-mean L0 GP corrections."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from functools import lru_cache
from math import isclose
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.physics import evaluate_batch, load_l0_json
from cft_revival.surrogates import ExactGP
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads
from experiments.l0_surrogate_v3.serialization import AtomicArtifactStore
from experiments.l0_surrogate_v7 import protocol as v7
from experiments.l0_surrogate_v7.cluster_conformal import exact_rank
from experiments.l0_surrogate_v7.design import (
    global_partition,
    normalized_design,
    operating_points,
    surrogate_inputs,
)
from experiments.l0_surrogate_v8 import protocol as v8

from .identity import acquire_lock, bind
from .models import (
    OUTPUT_NAMES,
    RAW_FAMILY,
    LeadingCorrectionGP,
    analytic_outputs,
    analytic_quantities,
    fit_models,
)

ROOT = Path(__file__).resolve().parent
MODERN = ROOT.parents[1]
REPO = MODERN.parent
DECLARATION = ROOT / "predeclaration.json"
DEPENDENCIES = ROOT / "dependency-manifest.json"
PARTITIONS = ROOT / "partitions.json"
PREFLIGHT = ROOT / "preflight.json"
RESULTS = ROOT / "results"
SOURCE_CONFIG = MODERN / "config/l0-deterministic-sweep.json"


def _load(path: Path) -> dict[str, object]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_declaration() -> dict[str, object]:
    value = _load(DECLARATION)
    payload = {key: item for key, item in value.items() if key != "predeclaration_hash"}
    if value["predeclaration_hash"] != canonical_hash(payload):
        raise ValueError("v9 predeclaration hash mismatch")
    return value


def _ranges(declaration: Mapping[str, object]) -> Mapping[str, object]:
    config = load_l0_json(SOURCE_CONFIG)
    if canonical_hash(config) != declaration["design"]["source_config_hash"]:  # type: ignore[index]
        raise ValueError("source config changed")
    return config["ranges"]


@lru_cache(maxsize=1)
def build_partitions() -> dict[str, object]:
    declaration = load_declaration()
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    roles = global_partition(inputs, declaration)
    prior, records = v7._prior_assessment_coordinates()
    for version in ("v7", "v8"):
        partition = _load(MODERN / "experiments" / f"l0_surrogate_{version}" / "partitions.json")
        previous = v7._prior_scrambled_inputs(version)
        indices = {
            int(index)
            for split in partition["roles"]["assessment"].values()  # type: ignore[index]
            for index in split["indices"]
        }
        coordinates = {previous[index] for index in indices}
        prior.update(coordinates)
        records[version] = {
            "assessment_coordinate_count": len(coordinates),
            "partitions_hash": partition["partitions_hash"],
        }
    assessment_indices = tuple(
        int(index)
        for split in roles["assessment"].values()  # type: ignore[union-attr]
        for index in split["indices"]
    )
    if {inputs[index] for index in assessment_indices}.intersection(prior):
        raise ValueError("v9 assessment reuses prior assessment coordinates")
    value: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v9-global-partition",
        "schema_version": "9.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "normalized_design_hash": canonical_hash({"normalized_design": [list(row) for row in normalized]}),
        "surrogate_input_hash": canonical_hash({"inputs": [list(row) for row in inputs]}),
        "rows": len(inputs),
        "roles": roles,
        "prior_assessment_evidence": records,
        "assessment_prior_coordinate_intersection_count": 0,
        "domain_disclosure": declaration["partition"]["domain_disclosure"],  # type: ignore[index]
        "label_policy": "input-only; no v9 physics labels evaluated",
    }
    value["partitions_hash"] = canonical_hash(value)
    return value


def write_partitions() -> dict[str, object]:
    value = build_partitions()
    AtomicArtifactStore(ROOT).write_json("partitions.json", value)
    return value


def _identity_fixture(declaration: Mapping[str, object]) -> dict[str, object]:
    extremes = (
        (0.0,) * 8,
        (1.0,) * 8,
        (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    )
    random_rows = normalized_design({**declaration, "design": {**declaration["design"], "rows": 32}})  # type: ignore[index]
    rows = extremes + random_rows
    points = operating_points(rows, _ranges(declaration))
    accepted = evaluate_batch(points)
    maximum_relative = 0.0
    for row, result in zip(rows, accepted, strict=True):
        analytic = analytic_quantities(row)
        reference = {
            "axial_thrust_n": result.axial_thrust_n,
            "specific_impulse_s": result.specific_impulse_s,
            "beam_current_a": result.power_budget.beam_current_a,
            "anode_current_a": result.power_budget.anode_current_a,
            "beam_kinetic_power_w": result.power_budget.beam_kinetic_power_w,
            "anode_input_power_w": result.power_budget.anode_input_power_w,
            "cathode_input_power_w": result.power_budget.cathode_input_power_w,
            "thruster_electrical_input_power_w": result.power_budget.thruster_electrical_input_power_w,
            "ppu_input_power_w": result.power_budget.ppu_input_power_w,
        }
        for name in analytic:
            denominator = max(abs(reference[name]), 1e-300)
            relative = abs(analytic[name] - reference[name]) / denominator
            maximum_relative = max(maximum_relative, relative)
            if not isclose(analytic[name], reference[name], rel_tol=2e-14, abs_tol=1e-14):
                raise ValueError(f"analytic identity mismatch: {name}")
    return {
        "points": len(rows),
        "extreme_points": len(extremes),
        "deterministic_random_points": len(random_rows),
        "maximum_relative_error": maximum_relative,
        "passed": True,
    }


def preflight(*, record: bool = False) -> dict[str, object]:
    from fractions import Fraction

    declaration = load_declaration()
    partitions = build_partitions()
    identity = _identity_fixture(declaration)
    if exact_rank(99, Fraction(9, 10)) != 90:
        raise ValueError("exact rank regression failed")
    with tempfile.TemporaryDirectory(prefix="l0-v9-preflight-") as temporary:
        store = AtomicArtifactStore(Path(temporary))
        selected = (0, 1, 2, 3, 4, 5, 6, 7)
        rows = tuple((index / 8, 0.2, 0.3, 0.4, 0.5) for index in range(8))
        observed = {
            index: analytic_outputs(row) for index, row in enumerate(rows)
        }
        models = fit_models("exact-leading-ratio-ard-matern52", selected, rows, observed)
        for output, model in zip(OUTPUT_NAMES, models, strict=True):
            payload = model.to_dict()
            reloaded = LeadingCorrectionGP.from_dict(payload)
            if reloaded.model_hash != model.model_hash:
                raise ValueError("leading correction model reload hash mismatch")
            store.write_json(f"models/{output}.model.json", payload)
        for name in ("frozen-method-selection", "cluster-calibration", "frozen-before-assessment", "final-assessment", "run-manifest", "failure-manifest"):
            store.write_json(f"{name}.json", {"synthetic": True})
        temporary_files = store.temporary_files()
    value: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v9-synthetic-preflight",
        "schema_version": "9.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partitions["partitions_hash"],
        "analytic_reference_identity": identity,
        "exact_group_rank_n99": 90,
        "model_reload_hash_valid": True,
        "temporary_file_count": len(temporary_files),
        "physics_label_access_count": 0,
        "assessment_label_access_count": 0,
        "passed": not temporary_files,
    }
    value["preflight_hash"] = canonical_hash(value)
    if record:
        AtomicArtifactStore(ROOT).write_json("preflight.json", value)
    return value


def _save_models(store: AtomicArtifactStore, family: str, budget: int, models: Sequence[object]) -> dict[str, str]:
    hashes = {}
    for name, model in zip(OUTPUT_NAMES, models, strict=True):
        store.write_json(f"development/{family}/budget-{budget}/{name}.model.json", model.to_dict())
        hashes[name] = model.model_hash
    return hashes


def _execute_after_binding(binding, lock: Path, started: float) -> dict[str, object]:
    declaration = load_declaration()
    partitions = _load(PARTITIONS)
    preflight = _load(PREFLIGHT)
    roles = partitions["roles"]
    normalized = normalized_design(declaration)
    inputs = surrogate_inputs(normalized)
    points = operating_points(normalized, _ranges(declaration))
    candidate = tuple(int(index) for index in roles["candidate_indices"])
    maximum_budget = max(int(value) for value in declaration["model_selection"]["budgets"])  # type: ignore[index]
    selected_all = candidate[:maximum_budget]
    observed = v7.RoleOracle(points, candidate).observe_many(selected_all)
    store = AtomicArtifactStore(RESULTS)
    store.write_json("training-selection.json", {
        "policy": declaration["model_selection"]["training_selection"],  # type: ignore[index]
        "selected_indices": list(selected_all),
        "selection_hash": canonical_hash(list(selected_all)),
    })
    method_labels = v7._load_role(points, roles["method-selection"])
    candidates = []
    model_lookup = {}
    for family in declaration["model_selection"]["families"]:  # type: ignore[index]
        for raw_budget in declaration["model_selection"]["budgets"]:  # type: ignore[index]
            budget = int(raw_budget)
            selected = selected_all[:budget]
            models = fit_models(str(family), selected, inputs, {index: observed[index] for index in selected})
            model_lookup[(str(family), budget)] = models
            point = v8._point_metrics(RAW_FAMILY, models, method_labels, inputs, declaration)
            methods = v8._select_intervals(
                RAW_FAMILY, models, selected, method_labels, roles["method-selection"], inputs, declaration
            )
            record = {
                "family": str(family),
                "budget": budget,
                "model_hashes": _save_models(store, str(family), budget, models),
                "point_metrics": point,
                "interval_methods": methods,
                "all_selection_gates_passed": point["all_scopes_outputs_passed"] and methods["all_gates_passed"],
            }
            record["candidate_hash"] = canonical_hash(record)
            store.write_json(f"development/{family}/budget-{budget}/method-metrics.json", record)
            candidates.append(record)
    passing_budgets = sorted({int(item["budget"]) for item in candidates if item["all_selection_gates_passed"]})
    selected_candidate = None
    if passing_budgets:
        budget = passing_budgets[0]
        eligible = [item for item in candidates if item["budget"] == budget and item["all_selection_gates_passed"]]
        selected_candidate = min(eligible, key=lambda item: (
            max(item["point_metrics"]["ood"][name]["worst_range_normalized_error"] for name in OUTPUT_NAMES),
            sum(item["point_metrics"]["overall"][name]["range_normalized_rmse"] for name in OUTPUT_NAMES),
            item["family"],
        ))
    frozen = {
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "method_labels_hash": canonical_hash(method_labels),
        "final_calibration_access_count": 0,
        "assessment_access_count": 0,
    }
    frozen["frozen_method_hash"] = canonical_hash(frozen)
    store.write_json("frozen-method-selection.json", frozen)
    if selected_candidate is None:
        manifest = {
            "document_type": "cft-revival-l0-surrogate-v9-run-manifest",
            "schema_version": "9.0",
            "commit_binding": binding.to_dict(),
            "exclusive_lock": {"file": lock.name, "retained": True},
            "partitions_hash": partitions["partitions_hash"],
            "preflight_hash": preflight["preflight_hash"],
            "frozen_method_hash": frozen["frozen_method_hash"],
            "status": "failed-development-selection-gates",
            "final_calibration_labels_accessed": False,
            "assessment_labels_accessed": False,
            "valid_prospective_result": True,
        }
        manifest["run_manifest_hash"] = canonical_hash(manifest)
        store.write_json("run-manifest.json", manifest)
        return manifest
    family = str(selected_candidate["family"])
    budget = int(selected_candidate["budget"])
    selected = selected_all[:budget]
    models = model_lookup[(family, budget)]
    methods = selected_candidate["interval_methods"]
    calibration_labels = v7._load_role(points, roles["final-calibration"])
    calibration = v8._calibrate(
        RAW_FAMILY, models, selected, calibration_labels, roles["final-calibration"], methods, inputs, declaration
    )
    calibration["calibration_hash"] = canonical_hash(calibration)
    store.write_json("cluster-calibration.json", calibration)
    frozen_calibration = {
        "frozen_method_hash": frozen["frozen_method_hash"],
        "calibration_hash": calibration["calibration_hash"],
        "assessment_access_count": 0,
    }
    frozen_calibration["frozen_calibration_hash"] = canonical_hash(frozen_calibration)
    store.write_json("frozen-before-assessment.json", frozen_calibration)
    assessment_labels = v7.SingleUseAssessment(points, roles["assessment"]).load(
        frozen_calibration["frozen_calibration_hash"], frozen_calibration["frozen_calibration_hash"]
    )
    raw, metrics = v8._assessment(
        RAW_FAMILY, models, selected, calibration, assessment_labels, roles["assessment"], inputs, declaration
    )
    assessment = {
        "document_type": "cft-revival-l0-surrogate-v9-final-assessment",
        "schema_version": "9.0",
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "raw": raw,
        "metrics": metrics,
    }
    assessment["assessment_hash"] = canonical_hash(assessment)
    store.write_json("final-assessment.json", assessment)
    accepted = bool(metrics["all_scopes_outputs_passed"])
    manifest = {
        "document_type": "cft-revival-l0-surrogate-v9-run-manifest",
        "schema_version": "9.0",
        "commit_binding": binding.to_dict(),
        "exclusive_lock": {"file": lock.name, "retained": True, "atomic": "O_CREAT|O_EXCL"},
        "partitions_hash": partitions["partitions_hash"],
        "normalized_design_hash": partitions["normalized_design_hash"],
        "assessment_prior_coordinate_intersection_count": 0,
        "same_domain_spatial_overlap_disclosed": True,
        "preflight_hash": preflight["preflight_hash"],
        "selected_family": family,
        "selected_features": "raw five output-relevant inputs plus exact analytic leading output correction",
        "selected_budget": budget,
        "selected_methods": {
            stratum: {name: methods[stratum][name]["selected_family"] for name in OUTPUT_NAMES}
            for stratum in ("interpolation", "boundary", "ood")
        },
        "development_point_metrics": selected_candidate["point_metrics"],
        "development_interval_metrics": methods,
        "frozen_method_hash": frozen["frozen_method_hash"],
        "frozen_calibration_hash": frozen_calibration["frozen_calibration_hash"],
        "assessment_metrics": metrics,
        "assessment_accessed_once_after_calibration_freeze": True,
        "row_coverage_upper_is_diagnostic_only": True,
        "status": "accepted" if accepted else "failed-predeclared-assessment-gates",
        "valid_prospective_result": True,
        "environment": {"python": sys.version, "platform": platform.platform(), "executable": sys.executable, "cwd": str(Path.cwd())},
        "claim": "same-domain prospective deterministic L0 software-emulation validation only",
    }
    manifest["run_manifest_hash"] = canonical_hash(manifest)
    store.write_json("run-manifest.json", manifest)
    store.write_json("runtime-diagnostics.json", {"wall_seconds": perf_counter() - started, "diagnostic_only": True})
    return manifest


def execute() -> dict[str, object]:
    started = perf_counter()
    binding = bind(REPO, DEPENDENCIES)
    lock = acquire_lock(REPO, binding.commit_sha)
    try:
        return _execute_after_binding(binding, lock, started)
    except Exception as error:
        failure = {
            "document_type": "cft-revival-l0-surrogate-v9-execution-failure",
            "schema_version": "9.0",
            "commit_binding": binding.to_dict(),
            "exclusive_lock_retained": True,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "rerun_performed": False,
        }
        failure["failure_manifest_hash"] = canonical_hash(failure)
        AtomicArtifactStore(RESULTS).write_json("failure-manifest.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("partitions", "preflight", "execute"))
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    result = write_partitions() if args.command == "partitions" else preflight(record=args.record) if args.command == "preflight" else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
