"""Preregistered MDO L0 campaign v2: contract binding, plans, callbacks and gates.

One :class:`CampaignPlan` drives both the evidentiary campaign and the disclosed
NON-EVIDENTIARY shakedown so the shakedown exercises exactly the production code
(catalogue binding, model, optimiser adapters, parallel dense reference, metrics, gates,
export).  Changes against v1 that close the v1 post-hoc audit disclosures are marked
``(v1 audit Fnn)`` where they occur.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import canonical_bytes, semantic_sha256, strict_json_file
from cft_revival.optimization.sampling import initial_designs

from . import catalogue as cat
from . import model as m
from . import optimizers as opt

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
SOURCE_ROOT = MODERN / "src" / "cft_revival"
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"

VERSION_TAG = "cft-revival.mdo-l0-campaign-v2"
PACKAGES = ("torch", "botorch", "gpytorch", "pymoo", "numpy", "scipy")
TIE_TOLERANCE = 1e-9
MAX_WORKERS = 12


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/1.0.0"


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    return value


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Protocol <-> module consistency (the frozen document is the authority)
# --------------------------------------------------------------------------


def protocol_consistency(value: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    space = value["design_space"]
    checks["catalogue_variable"] = (
        space["catalogue"]["name"] == m.CATALOGUE_VARIABLE
        and int(space["catalogue"]["size"]) == m.CATALOGUE_SIZE
        and space["catalogue"]["source_classification"] == cat.SOURCE_CLASSIFICATION
    )
    declared_variables = [
        (item["name"], float(item["lower"]), float(item["upper"]), item["units"]) for item in space["operating_point"]
    ]
    checks["design_variables"] = declared_variables == [
        (v.name, v.lower, v.upper, v.units) for v in m.DESIGN_VARIABLES
    ]
    checks["catalogue_binding"] = dict(value["catalogue_binding"]) == dict(cat.DATASET_BINDING)
    uncertain = value["uncertain_inputs"]
    checks["cell_inputs"] = [item["name"] for item in uncertain["per_design_inputs"]] == list(cat.CUSP_NAMES) and all(
        item["distribution"] == "jeffreys-beta-posterior" and item["trials"] == cat.CELL_TRIALS
        for item in uncertain["per_design_inputs"]
    )
    checks["pooled_input"] = (
        uncertain["pooled_input"]["name"] == cat.POOLED_NAME
        and uncertain["pooled_input"]["trials"] == cat.POOLED_TRIALS
        and uncertain["pooled_input"]["used_by"] == m.CLOSURE_CL2
    )
    declared_shared = [
        (item["name"], float(item["lower"]), float(item["upper"]), item["units"]) for item in uncertain["shared_inputs"]
    ]
    checks["shared_inputs"] = declared_shared == list(cat.SHARED_UNCERTAIN_INPUTS) and all(
        item["distribution"] == "uniform" for item in uncertain["shared_inputs"]
    )
    sample_spec = uncertain["sample"]
    checks["sample_parameters"] = (
        tuple(sample_spec["bases"]) == cat.QMC_BASES
        and sample_spec["seed"] == cat.QMC_SEED
        and sample_spec["count"] == cat.QMC_SAMPLE_SIZE
        and sample_spec["frozen"] is True
        and float(sample_spec["jeffreys_prior"]) == cat.JEFFREYS_PRIOR
    )
    checks["unit_rows_sha256"] = cat.unit_rows_sha256() == sample_spec["unit_rows_sha256"]
    checks["objectives"] = [
        (item["name"], item["direction"], item["units"], float(item["comparison_scale"])) for item in value["objectives"]
    ] == [(o.name, o.direction.value, o.units, o.comparison_scale) for o in m.OBJECTIVES]
    checks["reference_point"] = all(
        float(value["reference_point"][name]) == m.REFERENCE_POINT[name] for name in m.OBJECTIVE_NAMES
    )
    constraints = {item["name"]: item for item in value["constraints"]}
    checks["constraints"] = set(constraints) == {m.ROBUST_CONSTRAINT.name, m.NOMINAL_CONSTRAINT.name} and all(
        item["sense"] == ">=" and float(item["threshold"]) == 0.0 and float(item["violation_scale"]) == spec.violation_scale
        for item, spec in (
            (constraints[m.ROBUST_CONSTRAINT.name], m.ROBUST_CONSTRAINT),
            (constraints[m.NOMINAL_CONSTRAINT.name], m.NOMINAL_CONSTRAINT),
        )
    )
    robust = value["robust_formulation"]
    checks["robust_formulation"] = (
        robust["risk_measure"] == "CVaR"
        and float(robust["tail_fraction"]) == m.CVAR_TAIL_FRACTION
        and int(robust["tail_count"]) == m.tail_count(cat.QMC_SAMPLE_SIZE)
    )
    closures = value["closures"]
    checks["closures"] = (
        closures["CL-1"]["id"] == m.CLOSURE_CL1
        and closures["CL-1"]["role"] == "campaign"
        and closures["CL-2"]["id"] == m.CLOSURE_CL2
        and closures["CL-2"]["role"] == "sensitivity"
        and {key: float(item) for key, item in closures["fixed"].items() if key != "note"} == m.FIXED_CLOSURES
    )
    budget = value["budget"]
    checks["budget_arithmetic"] = (
        budget["initial_design"] + budget["qlognehvi_batch_size"] * budget["qlognehvi_iterations"]
        == budget["evaluations_per_run"]
        and budget["nsga3_population_size"] * budget["nsga3_generations"] == budget["evaluations_per_run"]
        and budget["nsga3_population_size"] == budget["initial_design"]
        and budget["total_evaluations"]
        == budget["evaluations_per_run"] * len(budget["seeds"]) * len(budget["strategies"])
        and tuple(budget["strategies"]) == opt.STRATEGIES
    )
    shakedown = value["shakedown"]
    checks["shakedown_arithmetic"] = (
        shakedown["initial_design"] + shakedown["qlognehvi_batch_size"] * shakedown["qlognehvi_iterations"]
        == shakedown["evaluations_per_run"]
        and shakedown["nsga3_population_size"] * shakedown["nsga3_generations"] == shakedown["evaluations_per_run"]
        and shakedown["nsga3_population_size"] == shakedown["initial_design"]
    )
    checks["seed_namespaces"] = all(seed < 1000 for seed in budget["seeds"]) and all(
        seed >= 900_000 for seed in shakedown["seeds"]
    )
    q = value["optimizers"]["qlognehvi"]
    checks["qlognehvi_arguments"] = (
        int(q["batch_size"]) == int(budget["qlognehvi_batch_size"])
        and q["device"] == "cpu"
        and q["dtype"] == "float64"
        and 1 <= int(q["torch_threads"]) <= MAX_WORKERS
        and int(q["candidates_per_design"]) >= 1
        and int(q["refine_maxiter"]) >= 1
        and int(q["mc_samples"]) >= 1
    )
    n = value["optimizers"]["nsga3"]
    checks["nsga3_arguments"] = (
        n["eliminate_duplicates"] is True
        and int(n["population_size"]) == int(budget["nsga3_population_size"])
        and int(n["generations"]) == int(budget["nsga3_generations"])
    )
    dense = value["dense_reference"]
    checks["dense_reference"] = (
        int(dense["points_per_design"]) >= 1
        and int(dense["designs"]) == m.CATALOGUE_SIZE
        and int(dense["total_evaluations"]) == int(dense["points_per_design"]) * m.CATALOGUE_SIZE
        and 1 <= int(dense["max_workers"]) <= MAX_WORKERS
    )
    checks["gate_semantics_declared"] = "integrity" in value["gates"]["semantics"].lower()
    checks["source_scope_explicit_files"] = all(
        "*" not in entry and entry.startswith("modern/") for entry in value["code_contract"]["source_hash_scope"]
    )
    return checks


def require_protocol_consistency(value: Mapping[str, Any]) -> dict[str, bool]:
    checks = protocol_consistency(value)
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError("protocol/module mismatch: " + ", ".join(failed))
    return checks


# --------------------------------------------------------------------------
# Code contract (explicit file scope bound to the modules actually imported; v1 audit F10)
# --------------------------------------------------------------------------


def source_files(value: Mapping[str, Any]) -> list[Path]:
    files: list[Path] = []
    for entry in value["code_contract"]["source_hash_scope"]:
        if not entry.startswith("modern/") or "*" in entry:
            raise ValueError(f"source scope entry must be an explicit path under modern/: {entry}")
        path = REPOSITORY.joinpath(*entry.split("/"))
        if not path.is_file():
            raise ValueError(f"source scope entry is not a file: {entry}")
        files.append(path)
    if len({path.resolve() for path in files}) != len(files):
        raise ValueError("source scope lists a file twice")
    return files


def source_hash_report(value: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    digest = hashlib.sha256()
    for path in source_files(value):
        data = path.read_bytes()
        if b"\r" in data:
            raise ValueError(f"hashed source contains a carriage return (CRLF checkout?): {path}")
        relative = path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
        file_sha = hashlib.sha256(data).hexdigest()
        entries.append({"path": relative, "sha256": file_sha, "bytes": len(data)})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {"source_sha256": digest.hexdigest(), "files": entries, "line_endings": "LF"}


def imported_repository_files() -> list[str]:
    """Repository-relative paths of every ``cft_revival`` and experiment module currently imported.

    The entry module (``python -m ...run``) is imported as ``__main__``; it is mapped by its
    file, so ``run.py`` is covered whichever way it was loaded.
    """

    roots = (SOURCE_ROOT.resolve(), EXPERIMENT.resolve())
    found: set[str] = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        try:
            path = Path(file).resolve()
        except OSError:
            continue
        if path.suffix != ".py":
            continue
        if any(root == path or root in path.parents for root in roots):
            found.add(path.relative_to(REPOSITORY.resolve()).as_posix())
    return sorted(found)


def import_scope_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the declared hash scope with the modules actually imported in this process."""

    declared = sorted(value["code_contract"]["source_hash_scope"])
    imported = imported_repository_files()
    return {
        "declared": declared,
        "imported": imported,
        "imported_not_in_scope": sorted(set(imported) - set(declared)),
        "in_scope_not_imported": sorted(set(declared) - set(imported)),
        "matches": set(imported) == set(declared),
    }


def package_versions() -> dict[str, str]:
    observed = {}
    for name in PACKAGES:
        module = importlib.import_module(name)
        observed[name] = str(getattr(module, "__version__"))
    return observed


def code_contract_report(value: Mapping[str, Any]) -> dict[str, Any]:
    sources = source_hash_report(value)
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
            f"{report['observed_package_versions']} vs declared {report['declared_package_versions']}; "
            f"python ok={report['python_minor_matches']}"
        )
    return report


# --------------------------------------------------------------------------
# Campaign plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPlan:
    kind: str
    seeds: tuple[int, ...]
    strategies: tuple[str, ...]
    evaluations_per_run: int
    initial_design: int
    qlognehvi_batch_size: int
    qlognehvi_iterations: int
    nsga3_population_size: int
    nsga3_generations: int
    dense_reference_points_per_design: int
    binding_gates: bool

    def __post_init__(self) -> None:
        if self.kind not in {"evidentiary", "shakedown"}:
            raise ValueError("plan kind must be evidentiary or shakedown")
        if (
            self.initial_design + self.qlognehvi_batch_size * self.qlognehvi_iterations != self.evaluations_per_run
            or self.nsga3_population_size * self.nsga3_generations != self.evaluations_per_run
            or self.nsga3_population_size != self.initial_design
        ):
            raise ValueError("plan budget arithmetic is inconsistent")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("plan seeds must be unique and non-empty")
        if self.dense_reference_points_per_design < 1:
            raise ValueError("dense reference needs at least one operating point per design")

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(f"{self.kind}:{strategy}:{seed}" for seed in self.seeds for strategy in self.strategies)


def evidentiary_plan(value: Mapping[str, Any]) -> CampaignPlan:
    budget = value["budget"]
    return CampaignPlan(
        kind="evidentiary",
        seeds=tuple(int(seed) for seed in budget["seeds"]),
        strategies=tuple(budget["strategies"]),
        evaluations_per_run=int(budget["evaluations_per_run"]),
        initial_design=int(budget["initial_design"]),
        qlognehvi_batch_size=int(budget["qlognehvi_batch_size"]),
        qlognehvi_iterations=int(budget["qlognehvi_iterations"]),
        nsga3_population_size=int(budget["nsga3_population_size"]),
        nsga3_generations=int(budget["nsga3_generations"]),
        dense_reference_points_per_design=int(value["dense_reference"]["points_per_design"]),
        binding_gates=True,
    )


def shakedown_plan(value: Mapping[str, Any]) -> CampaignPlan:
    shakedown = value["shakedown"]
    return CampaignPlan(
        kind="shakedown",
        seeds=tuple(int(seed) for seed in shakedown["seeds"]),
        strategies=tuple(value["budget"]["strategies"]),
        evaluations_per_run=int(shakedown["evaluations_per_run"]),
        initial_design=int(shakedown["initial_design"]),
        qlognehvi_batch_size=int(shakedown["qlognehvi_batch_size"]),
        qlognehvi_iterations=int(shakedown["qlognehvi_iterations"]),
        nsga3_population_size=int(shakedown["nsga3_population_size"]),
        nsga3_generations=int(shakedown["nsga3_generations"]),
        dense_reference_points_per_design=int(shakedown["dense_reference_points_per_design"]),
        binding_gates=False,
    )


def plan_record(plan: CampaignPlan) -> dict[str, Any]:
    """Plain-JSON plan record (lists, not tuples, so re-parsed records re-hash)."""

    record = {key: list(item) if isinstance(item, tuple) else item for key, item in asdict(plan).items()}
    record["run_ids"] = list(plan.run_ids)
    return record


def design_sha256(value: Mapping[str, Any], plan: CampaignPlan) -> str:
    """Identity of a plan's complete evaluation design (seeds, budgets, catalogue, sample)."""

    return semantic_sha256(
        {
            "plan": plan_record(plan),
            "unit_rows_sha256": value["uncertain_inputs"]["sample"]["unit_rows_sha256"],
            "catalogue_sample_sha256": value["uncertain_inputs"]["sample"]["catalogue_sample_sha256"],
            "catalogue_sha256": value["catalogue_binding_identity"]["catalogue_sha256"],
            "design_variables": [asdict(v) for v in m.DESIGN_VARIABLES],
            "catalogue_size": m.CATALOGUE_SIZE,
            "objectives": list(m.OBJECTIVE_NAMES),
            "reference_point": m.REFERENCE_POINT,
            "closure": m.CLOSURE_CL1,
        }
    )


def shakedown_disjointness(value: Mapping[str, Any]) -> dict[str, Any]:
    evidentiary = evidentiary_plan(value)
    shakedown = shakedown_plan(value)
    seed_overlap = sorted(set(evidentiary.seeds) & set(shakedown.seeds))
    run_overlap = sorted(set(evidentiary.run_ids) & set(shakedown.run_ids))
    namespace = all(seed < 1000 for seed in evidentiary.seeds) and all(seed >= 900_000 for seed in shakedown.seeds)
    initial_overlap = 0
    evidentiary_rows = {
        tuple(row) for seed in evidentiary.seeds for row in opt.shared_initial_points(seed, evidentiary.initial_design).tolist()
    }
    for seed in shakedown.seeds:
        for row in opt.shared_initial_points(seed, shakedown.initial_design).tolist():
            if tuple(row) in evidentiary_rows:
                initial_overlap += 1
    proven = bool(
        not seed_overlap
        and not run_overlap
        and namespace
        and initial_overlap == 0
        and shakedown.evaluations_per_run < evidentiary.evaluations_per_run
        and design_sha256(value, shakedown) != design_sha256(value, evidentiary)
    )
    return {
        "proven": proven,
        "seed_overlap": seed_overlap,
        "run_id_overlap": run_overlap,
        "seed_namespace_rule_holds": namespace,
        "initial_design_overlap_count": initial_overlap,
        "shakedown_budget_smaller": shakedown.evaluations_per_run < evidentiary.evaluations_per_run,
        "shakedown_design_sha256": design_sha256(value, shakedown),
        "evidentiary_design_sha256": design_sha256(value, evidentiary),
    }


# --------------------------------------------------------------------------
# Shakedown record verification (prepare, execute and prebundle)
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
        checks["package_versions_current"] = (
            record.get("package_versions") == contract["observed_package_versions"] and contract["matches"]
        )
    except Exception:
        checks["source_sha256_current"] = False
        checks["package_versions_current"] = False
    disjointness = record.get("disjointness")
    checks["disjointness_proven"] = isinstance(disjointness, Mapping) and disjointness.get("proven") is True
    try:
        checks["shakedown_design_sha256_current"] = record.get("shakedown_design_sha256") == design_sha256(
            value, shakedown_plan(value)
        )
        checks["evidentiary_design_sha256_current"] = record.get("evidentiary_design_sha256") == design_sha256(
            value, evidentiary_plan(value)
        )
        checks["disjointness_recomputed"] = shakedown_disjointness(value)["proven"]
    except Exception:
        checks["shakedown_design_sha256_current"] = False
        checks["evidentiary_design_sha256_current"] = False
        checks["disjointness_recomputed"] = False
    runs = record.get("runs")
    expected_runs = (
        {run_id.split(":", 1)[1] for run_id in shakedown_plan(value).run_ids}
        if checks["shakedown_design_sha256_current"]
        else set()
    )
    checks["all_runs_present"] = isinstance(runs, Mapping) and set(runs) == expected_runs and bool(runs)
    checks["all_runs_budget_exact"] = (
        isinstance(runs, Mapping) and bool(runs) and all(item.get("evaluations") == item.get("budget") for item in runs.values())
    )
    gates = record.get("gates")
    checks["all_binding_gates_passed"] = (
        isinstance(gates, Mapping)
        and isinstance(gates.get("binding"), Mapping)
        and bool(gates["binding"])
        and all(item.get("passed") is True for item in gates["binding"].values())
    )
    scope = record.get("import_scope")
    checks["import_scope_matched"] = isinstance(scope, Mapping) and scope.get("matches") is True and sorted(
        scope.get("declared", [])
    ) == sorted(value["code_contract"]["source_hash_scope"])
    runtime = record.get("runtime")
    checks["runtime_accepted_and_bundle_validated"] = (
        isinstance(runtime, Mapping)
        and runtime.get("terminal_state") == "accepted_result"
        and runtime.get("bundle_validated") is True
    )
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError("shakedown gate refused: " + ", ".join(failed))
    return checks


@dataclass(frozen=True)
class FrozenAuthority:
    authorities: Mapping[str, Any]
    shakedown: Mapping[str, Any]
    shakedown_bytes: bytes


def load_frozen_authority() -> FrozenAuthority:
    return FrozenAuthority(strict_json_file(AUTHORITIES_PATH), strict_json_file(SHAKEDOWN_PATH), SHAKEDOWN_PATH.read_bytes())


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------


def _robust_vector(record: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(record["robust_objectives"][name] for name in m.OBJECTIVE_NAMES)


def _nominal_vector(record: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(record["nominal_objectives"][name] for name in m.OBJECTIVE_NAMES)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95 % interval (the screening dataset's convention)."""

    if trials < 1:
        raise ValueError("trials must be positive")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def catalogue_design_summary(design: cat.CatalogueDesign) -> dict[str, Any]:
    """Screening P(wall) (point, Wilson-95, posterior mean) and sealed geometry of one design."""

    cells = []
    for k in range(len(cat.CELLS)):
        s, n = design.cell_wall_hits[k], design.cell_trials[k]
        lower, upper = wilson_interval(s, n)
        cells.append(
            {
                "cell": cat.CELLS[k],
                "wall_hits": s,
                "trials": n,
                "probability": s / n,
                "wilson_95": [lower, upper],
                "posterior_mean": cat.posterior_mean(s, n),
            }
        )
    lower, upper = wilson_interval(design.pooled_wall_hits, design.pooled_trials)
    nominal = cat.design_nominal_theta(design)
    return {
        "catalogue_index": design.index,
        "case_id": design.case_id,
        "screening_design_id": design.design_id,
        "geometry_sha256": design.geometry_sha256,
        "cells": cells,
        "pooled": {
            "wall_hits": design.pooled_wall_hits,
            "trials": design.pooled_trials,
            "probability": design.pooled_point_estimate,
            "wilson_95": [lower, upper],
            "posterior_mean": cat.posterior_mean(design.pooled_wall_hits, design.pooled_trials),
            "reflected": design.reflected,
        },
        "nominal_survival_cl1": m.survival(nominal, m.CLOSURE_CL1),
        "nominal_survival_cl2": m.survival(nominal, m.CLOSURE_CL2),
        "geometry": dict(design.geometry),
        "design_values": dict(design.design_values),
    }


def _front_membership(front: Sequence[Mapping[str, Any]], designs: Sequence[cat.CatalogueDesign]) -> list[dict[str, Any]]:
    by_index: dict[int, list[Mapping[str, Any]]] = {}
    for record in front:
        by_index.setdefault(int(record["design"]["catalogue_index"]), []).append(record)
    out = []
    for index in sorted(by_index):
        summary = catalogue_design_summary(designs[index])
        summary["front_members"] = len(by_index[index])
        summary["operating_points"] = [record["design"]["values"] for record in by_index[index]]
        out.append(summary)
    return out


def pooled_fronts(records: Sequence[Mapping[str, Any]], designs: Sequence[cat.CatalogueDesign]) -> dict[str, Any]:
    """Robust and nominal nondominated sets of a pool of evaluation records, with catalogue membership."""

    unique: dict[str, Mapping[str, Any]] = {}
    for record in records:
        unique.setdefault(record["design"]["design_id"], record)
    ordered = list(unique.values())
    robust_rows = [
        (record, m.normalized_objectives(_robust_vector(record))) for record in ordered if record["status"] == "success"
    ]
    nominal_rows = [
        (record, m.normalized_objectives(_nominal_vector(record)))
        for record in ordered
        if record["nominal_objectives"] is not None
    ]
    robust_front = [robust_rows[index][0] for index in m.nondominated_indices([row[1] for row in robust_rows])]
    nominal_front = [nominal_rows[index][0] for index in m.nondominated_indices([row[1] for row in nominal_rows])]
    robust_ids = {record["design"]["design_id"] for record in robust_front}
    nominal_ids = {record["design"]["design_id"] for record in nominal_front}
    nominal_feasible_robust = sum(
        1 for record in nominal_front if record["constraints"][m.ROBUST_CONSTRAINT.name] >= 0.0
    )

    def objective_ranges(front: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        out = {}
        for name in m.OBJECTIVE_NAMES:
            column = [record[key][name] for record in front]
            out[name] = {"minimum": min(column), "maximum": max(column)} if column else None
        return out

    def design_rows(front: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "design_id": record["design"]["design_id"],
                "catalogue_index": record["design"]["catalogue_index"],
                "case_id": record["design"]["case_id"],
                "values": record["design"]["values"],
                "robust_objectives": record["robust_objectives"],
                "nominal_objectives": record["nominal_objectives"],
                "constraints": record["constraints"],
                "survival_statistics": record["survival_statistics"],
            }
            for record in front
        ]

    return {
        "unique_designs": len(ordered),
        "distinct_catalogue_designs": len({record["design"]["catalogue_index"] for record in ordered}),
        "robust": {
            "candidates": len(robust_rows),
            "front_size": len(robust_front),
            "hypervolume": m.hypervolume([row[1] for row in robust_rows]),
            "design_ids": sorted(robust_ids),
            "catalogue_indices": sorted({record["design"]["catalogue_index"] for record in robust_front}),
            "objective_ranges": objective_ranges(robust_front, "robust_objectives"),
            "designs": design_rows(robust_front),
            "catalogue_membership": _front_membership(robust_front, designs),
        },
        "nominal": {
            "candidates": len(nominal_rows),
            "front_size": len(nominal_front),
            "hypervolume": m.hypervolume([row[1] for row in nominal_rows]),
            "design_ids": sorted(nominal_ids),
            "catalogue_indices": sorted({record["design"]["catalogue_index"] for record in nominal_front}),
            "objective_ranges": objective_ranges(nominal_front, "nominal_objectives"),
            "robust_feasible_members": nominal_feasible_robust,
            "designs": design_rows(nominal_front),
            "catalogue_membership": _front_membership(nominal_front, designs),
        },
        "shared_design_ids": sorted(robust_ids & nominal_ids),
        "jaccard_robust_nominal": (
            len(robust_ids & nominal_ids) / len(robust_ids | nominal_ids) if robust_ids | nominal_ids else 1.0
        ),
    }


# ---- dense reference (parallel over designs; every worker evaluates a pure function) -------

_WORKER_CONTEXT: dict[str, Any] = {}


def _dense_worker_init(closure: str) -> None:
    _WORKER_CONTEXT["context"] = m.build_context(closure=closure)


def dense_grid(points_per_design: int, seed: int) -> list[tuple[float, ...]]:
    """The fixed operating-point grid shared by every catalogue design (v1's shifted-Halton construction)."""

    return [tuple(float(v) for v in design.values) for design in initial_designs(m.DESIGN_VARIABLES, points_per_design, seed=seed)]


def _dense_rows_for_design(index: int, grid: Sequence[Sequence[float]], context: m.EvaluationContext) -> list[dict[str, Any]]:
    rows = []
    for position, values in enumerate(grid):
        evaluation = m.evaluate_design(index, values, context)
        record = evaluation.to_record()
        record["grid_index"] = position
        rows.append(record)
    return rows


def _dense_worker(arguments: tuple[int, list[tuple[float, ...]]]) -> tuple[int, list[dict[str, Any]]]:
    index, grid = arguments
    return index, _dense_rows_for_design(index, grid, _WORKER_CONTEXT["context"])


def compact_columns(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Columnar form of evaluation records (hashed for the dense reference; stored for fronts)."""

    def column(getter: Any) -> list[Any]:
        return [getter(record) for record in records]

    return {
        "count": len(records),
        "catalogue_index": column(lambda r: r["design"]["catalogue_index"]),
        "values": column(lambda r: r["design"]["values"]),
        "design_id": column(lambda r: r["design"]["design_id"]),
        "status": column(lambda r: r["status"]),
        m.ROBUST_CONSTRAINT.name: column(lambda r: r["constraints"][m.ROBUST_CONSTRAINT.name]),
        m.NOMINAL_CONSTRAINT.name: column(lambda r: r["constraints"][m.NOMINAL_CONSTRAINT.name]),
        "robust_objectives": column(
            lambda r: None if r["robust_objectives"] is None else [r["robust_objectives"][name] for name in m.OBJECTIVE_NAMES]
        ),
        "nominal_objectives": column(
            lambda r: None if r["nominal_objectives"] is None else [r["nominal_objectives"][name] for name in m.OBJECTIVE_NAMES]
        ),
        "sample_result_sha256": column(lambda r: r["sample_result_sha256"]),
    }


def dense_reference(
    context: m.EvaluationContext, points_per_design: int, seed: int, *, max_workers: int
) -> dict[str, Any]:
    """Every catalogue design x the fixed operating-point grid, evaluated in parallel by design."""

    grid = dense_grid(points_per_design, seed)
    tick = time.perf_counter()
    per_design_records: dict[int, list[dict[str, Any]]] = {}
    workers = max(1, min(int(max_workers), MAX_WORKERS, os.cpu_count() or 1))
    if workers == 1 or len(context.designs) == 1:
        for design in context.designs:
            per_design_records[design.index] = _dense_rows_for_design(design.index, grid, context)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_dense_worker_init, initargs=(context.closure,)) as pool:
            for index, rows in pool.map(_dense_worker, [(design.index, grid) for design in context.designs]):
                per_design_records[index] = rows
    seconds = time.perf_counter() - tick
    records: list[dict[str, Any]] = []
    blocks: list[list[int]] = []
    per_design_summary = []
    for design in context.designs:
        rows = per_design_records[design.index]
        start = len(records)
        records.extend(rows)
        blocks.append(list(range(start, start + len(rows))))
        robust_points = [m.normalized_objectives(_robust_vector(r)) for r in rows if r["status"] == "success"]
        nominal_points = [m.normalized_objectives(_nominal_vector(r)) for r in rows if r["nominal_objectives"] is not None]
        per_design_summary.append(
            {
                "catalogue_index": design.index,
                "case_id": design.case_id,
                "feasible": len(robust_points),
                "infeasible": len(rows) - len(robust_points),
                "robust_front_size": len(m.nondominated_indices(robust_points)) if robust_points else 0,
                "robust_hypervolume": m.hypervolume(robust_points),
                "nominal_front_size": len(m.nondominated_indices(nominal_points)) if nominal_points else 0,
                "nominal_hypervolume": m.hypervolume(nominal_points),
                "nominal_survival_cl1": m.survival(context.nominal[design.index], context.closure),
            }
        )
    robust_index = [i for i, r in enumerate(records) if r["status"] == "success"]
    robust_points_all = [m.normalized_objectives(_robust_vector(records[i])) for i in robust_index]
    robust_blocks = _reindex_blocks(blocks, robust_index)
    robust_front_local = m.nondominated_indices_blockwise(robust_points_all, robust_blocks)
    robust_front = [records[robust_index[j]] for j in robust_front_local]
    nominal_index = [i for i, r in enumerate(records) if r["nominal_objectives"] is not None]
    nominal_points_all = [m.normalized_objectives(_nominal_vector(records[i])) for i in nominal_index]
    nominal_blocks = _reindex_blocks(blocks, nominal_index)
    nominal_front_local = m.nondominated_indices_blockwise(nominal_points_all, nominal_blocks)
    nominal_front = [records[nominal_index[j]] for j in nominal_front_local]
    front_tick = time.perf_counter()
    robust_hv = m.hypervolume([robust_points_all[j] for j in robust_front_local])
    nominal_hv = m.hypervolume([nominal_points_all[j] for j in nominal_front_local])
    columns = compact_columns(records)
    return {
        "designs": len(context.designs),
        "points_per_design": points_per_design,
        "count": len(records),
        "seed": seed,
        "closure": context.closure,
        "workers": workers,
        "evaluation_seconds": seconds,
        "front_and_hypervolume_seconds": time.perf_counter() - front_tick,
        "feasible": len(robust_index),
        "infeasible": len(records) - len(robust_index),
        "grid": [list(values) for values in grid],
        "columns_sha256": hashlib.sha256(canonical_bytes(columns)).hexdigest(),
        "per_design": per_design_summary,
        "fronts": {
            "robust": {
                "front_size": len(robust_front),
                "hypervolume": robust_hv,
                "catalogue_indices": sorted({r["design"]["catalogue_index"] for r in robust_front}),
                "catalogue_membership": _front_membership(robust_front, context.designs),
                "records": robust_front,
            },
            "nominal": {
                "front_size": len(nominal_front),
                "hypervolume": nominal_hv,
                "catalogue_indices": sorted({r["design"]["catalogue_index"] for r in nominal_front}),
                "records": nominal_front,
            },
        },
        "records": records,
    }


def _reindex_blocks(blocks: Sequence[Sequence[int]], kept: Sequence[int]) -> list[list[int]]:
    position = {original: local for local, original in enumerate(kept)}
    out = []
    for block in blocks:
        local = [position[i] for i in block if i in position]
        if local:
            out.append(local)
    return out


def separability_check(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Under CL-1 the robust/nominal ratio of each objective depends on the catalogue design only.

    ``f_j(k, x, theta) = g_j(x) h_j(k, theta)`` with ``g_j > 0`` (anode power has ``h = 1``), so
    within one design the ratio CVaR/nominal is constant over operating points; across designs
    it varies with the design's own posterior.  Reported per design; the gate-free expectation
    is a relative spread <= 1e-9 within every design.
    """

    per_design: dict[int, dict[str, list[float]]] = {}
    for record in records:
        if record["status"] != "success" or record["nominal_objectives"] is None:
            continue
        ratios = per_design.setdefault(int(record["design"]["catalogue_index"]), {name: [] for name in m.OBJECTIVE_NAMES})
        for name in m.OBJECTIVE_NAMES:
            nominal = record["nominal_objectives"][name]
            if nominal != 0.0:
                ratios[name].append(record["robust_objectives"][name] / nominal)
    report = []
    passed = True
    for index in sorted(per_design):
        entry: dict[str, Any] = {"catalogue_index": index}
        for name, column in per_design[index].items():
            if not column:
                entry[name] = None
                continue
            low, high = min(column), max(column)
            spread = (high - low) / max(abs(high), 1e-300)
            entry[name] = {"ratio_min": low, "ratio_max": high, "relative_spread": spread, "count": len(column)}
            passed = passed and spread <= 1e-9
        report.append(entry)
    return {"passed": passed and bool(report), "per_design": report, "tolerance_relative_spread": 1e-9}


def replay_records(records: Sequence[Mapping[str, Any]], context: m.EvaluationContext) -> dict[str, Any]:
    """Recompute every record through the model and compare bit-exactly."""

    mismatches = []
    for record in records:
        if record["closure"] != context.closure:
            mismatches.append({"index": record.get("index"), "key": "closure"})
            continue
        evaluation = m.evaluate_design(record["design"]["catalogue_index"], record["design"]["values"], context)
        replayed = evaluation.to_record()
        for key in (
            "design",
            "closure",
            "constraints",
            "status",
            "failure_code",
            "robust_objectives",
            "robust_statistics",
            "nominal_objectives",
            "survival_statistics",
            "sample_result_sha256",
        ):
            if canonical_bytes(replayed[key]) != canonical_bytes(record[key]):
                mismatches.append({"index": record.get("index"), "key": key})
                break
    return {"replayed": len(records), "mismatches": mismatches, "passed": not mismatches}


def _front_ids(rows: Mapping[str, tuple[float, ...]], tolerance: float = 0.0) -> set[str]:
    keys = list(rows)
    return {keys[i] for i in m.nondominated_indices([rows[key] for key in keys], relative_tolerance=tolerance)}


def _reevaluate_pool(
    designs: Sequence[tuple[str, int, Sequence[float]]], context: m.EvaluationContext
) -> tuple[dict[str, tuple[float, ...]], dict[str, Mapping[str, Any]], int]:
    rows: dict[str, tuple[float, ...]] = {}
    records: dict[str, Mapping[str, Any]] = {}
    infeasible = 0
    for design_id, index, values in designs:
        evaluation = m.evaluate_design(index, values, context)
        records[design_id] = evaluation.to_record()
        if evaluation.status == "success":
            rows[design_id] = m.normalized_objectives(evaluation.robust_objectives)
        else:
            infeasible += 1
    return rows, records, infeasible


def closure_and_width_sensitivity(
    records: Sequence[Mapping[str, Any]],
    value: Mapping[str, Any],
    campaign_front_ids: Sequence[str],
    base_context: m.EvaluationContext,
) -> dict[str, Any]:
    """Re-evaluate the pooled designs under CL-2 and under rescaled posterior widths (reported, not gated)."""

    unique: dict[str, Mapping[str, Any]] = {}
    for record in records:
        unique.setdefault(record["design"]["design_id"], record)
    designs = [
        (key, int(record["design"]["catalogue_index"]), record["design"]["values"]) for key, record in unique.items()
    ]
    campaign_set = set(campaign_front_ids)
    campaign_rows = {
        key: m.normalized_objectives(_robust_vector(record)) for key, record in unique.items() if record["status"] == "success"
    }
    campaign_hv = m.hypervolume(list(campaign_rows.values()))

    def compare(rows: Mapping[str, tuple[float, ...]], infeasible: int, front_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        front = _front_ids(rows)
        union = front | campaign_set
        common = set(rows) & set(campaign_rows)
        alternative_common = _front_ids({k: rows[k] for k in common}, TIE_TOLERANCE)
        campaign_common = _front_ids({k: campaign_rows[k] for k in common}, TIE_TOLERANCE)
        by_catalogue = sorted({unique[k]["design"]["catalogue_index"] for k in front})
        return {
            "feasible": len(rows),
            "infeasible": infeasible,
            "front_size": len(front),
            "hypervolume": m.hypervolume(list(rows.values())),
            "campaign_hypervolume": campaign_hv,
            "shared_with_campaign_front": len(front & campaign_set),
            "jaccard_with_campaign_front": len(front & campaign_set) / len(union) if union else 1.0,
            "common_feasible_designs": len(common),
            "identical_on_common_feasible_set_up_to_ties": alternative_common == campaign_common,
            "common_front_symmetric_difference": len(alternative_common ^ campaign_common),
            "front_catalogue_indices": by_catalogue,
            "campaign_front_catalogue_indices": sorted({unique[k]["design"]["catalogue_index"] for k in campaign_set if k in unique}),
            "front_design_ids": sorted(front),
            "front_members": [
                {
                    "design_id": k,
                    "catalogue_index": unique[k]["design"]["catalogue_index"],
                    "case_id": unique[k]["design"]["case_id"],
                    "values": unique[k]["design"]["values"],
                    "robust_objectives": front_records[k]["robust_objectives"],
                    "survival_statistics": front_records[k]["survival_statistics"],
                }
                for k in sorted(front)
            ],
        }

    # CL-2: pooled survival, campaign posterior width.
    cl2_context = m.EvaluationContext(
        designs=base_context.designs,
        sample=base_context.sample,
        nominal=base_context.nominal,
        closure=m.CLOSURE_CL2,
        tail_fraction=base_context.tail_fraction,
    )
    rows, recs, infeasible = _reevaluate_pool(designs, cl2_context)
    closure_report = {
        "closure": m.CLOSURE_CL2,
        "statement": value["closures"]["CL-2"]["statement"],
        **compare(rows, infeasible, recs),
    }
    widths = []
    for width in value["uncertain_inputs"]["sensitivity_widths"]["width_scales"]:
        scale = None if width == "point" else float(width)
        context = m.build_context(base_context.designs, closure=m.CLOSURE_CL1, width_scale=scale, tail_fraction=base_context.tail_fraction)
        rows, recs, infeasible = _reevaluate_pool(designs, context)
        survivals = [theta_survival for design_rows in context.sample for theta_survival in (m.survival(t, m.CLOSURE_CL1) for t in design_rows)]
        widths.append(
            {
                "width_scale": width,
                "meaning": (
                    "posterior replaced by its mean (no binomial uncertainty in the cell probabilities)"
                    if scale is None
                    else f"Beta(s*w + 1/2, (n - s)*w + 1/2) with w = {scale} (standard deviation ~ 1/sqrt(w) of the campaign posterior)"
                ),
                "is_campaign_posterior": scale == 1.0,
                "survival_min": min(survivals),
                "survival_max": max(survivals),
                **compare(rows, infeasible, recs),
            }
        )
    return {"unique_designs": len(designs), "closure_cl2": closure_report, "widths": widths}


HYPERVOLUME_ROUNDOFF_TOLERANCE = 1e-12


def hypervolume_monotonicity(curves: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Every curve must be non-decreasing up to floating-point roundoff.

    The exact slicing hypervolume of a growing nondominated set is monotone in exact
    arithmetic; a new point with a negligible exclusive contribution changes the slicing
    order and can move the floating-point result by an ulp (observed: -2.1e-16 relative in
    the shakedown).  The gate therefore allows a relative decrease of at most
    ``HYPERVOLUME_ROUNDOFF_TOLERANCE`` and records the largest decrease seen.
    """

    worst = 0.0
    violations = []
    for key, curve in curves.items():
        for i in range(1, len(curve)):
            previous = float(curve[i - 1]["hypervolume"])
            current = float(curve[i]["hypervolume"])
            if current < previous:
                relative = (previous - current) / previous if previous > 0.0 else math.inf
                worst = max(worst, relative)
                if relative > HYPERVOLUME_ROUNDOFF_TOLERANCE:
                    violations.append({"run": key, "evaluations": curve[i]["evaluations"], "relative_decrease": relative})
    return {
        "passed": not violations,
        "relative_tolerance": HYPERVOLUME_ROUNDOFF_TOLERANCE,
        "largest_relative_decrease": worst,
        "violations": violations,
    }


def _paired_test(summaries: Mapping[str, Mapping[str, Any]], seeds: Sequence[int], left: str, right: str) -> dict[str, Any]:
    wins = 0
    pairs = []
    for seed in seeds:
        a = summaries[f"{left}:{seed}"]["final_hypervolume"]
        b = summaries[f"{right}:{seed}"]["final_hypervolume"]
        pairs.append({"seed": seed, left: a, right: b, "left_wins": a > b})
        wins += int(a > b)
    required = math.ceil(2 * len(seeds) / 3) if len(seeds) != 3 else 2
    return {
        "left": left,
        "right": right,
        "wins": wins,
        "seeds": len(seeds),
        "required_wins": required,
        "passed": wins >= required,
        "statement": (
            f"{left} final robust hypervolume > {right} in {wins} of {len(seeds)} seeds (paired by seed); "
            "a count, not a significance statement: with three seeds the >= 2/3 rule passes with probability 1/2 under a no-difference null"
        ),
        "pairs": pairs,
    }


def _seed_variance(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def label_checks(run_infos: Mapping[str, Mapping[str, Any]], plan: CampaignPlan) -> dict[str, Any]:
    """Descriptive labels must be reproducible from the recorded arguments (v1 audit F28)."""

    checks: dict[str, bool] = {}
    for key, info in run_infos.items():
        strategy = key.split(":", 1)[0]
        if strategy == "qlognehvi":
            arguments = info["arguments"]
            checks[f"{key}:acquisition_label"] = info["acquisition"] == opt.acquisition_label(
                q=arguments["q"],
                mc_samples=arguments["mc_samples"],
                candidates_per_design=arguments["candidates_per_design"],
                refine_maxiter=arguments["refine_maxiter"],
                refine_num_restarts=arguments["refine_num_restarts"],
                sequential_candidate_stage=arguments["sequential_candidate_stage"],
            )
            checks[f"{key}:model_label"] = all(
                token in str(info["model"]) for token in ("MixedSingleTaskGP", "CategoricalKernel", "MaternKernel", "Standardize")
            )
            checks[f"{key}:batch_size"] = arguments["q"] == plan.qlognehvi_batch_size and info["iterations"] == plan.qlognehvi_iterations
        elif strategy == "nsga3":
            checks[f"{key}:generations"] = (
                info["declared_generations"] == plan.nsga3_generations
                and isinstance(info["pymoo_n_gen"], int)
                and info["pymoo_reported_evaluations"] == plan.evaluations_per_run
                and info["eliminate_duplicates"] is True
            )
        elif strategy == "lhs":
            checks[f"{key}:stages"] = info["stages"] == [plan.initial_design, plan.evaluations_per_run - plan.initial_design] and str(
                plan.initial_design
            ) in info["design"]
    return {"checks": checks, "passed": bool(checks) and all(checks.values())}


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

    def prebundle(context: Any) -> Mapping[str, Any]:
        consistency = require_protocol_consistency(value)
        contract = require_code_contract(value)
        binding = cat.require_binding(value["catalogue_binding"])
        designs = cat.load_catalogue(value["catalogue_binding"])
        catalogue_sha = cat.catalogue_sha256(designs)
        if catalogue_sha != value["catalogue_binding_identity"]["catalogue_sha256"]:
            raise ValueError("catalogue identity differs from the protocol")
        evaluation_context = m.build_context(designs)
        sample_sha = evaluation_context.sample_sha256
        if sample_sha != value["uncertain_inputs"]["sample"]["catalogue_sample_sha256"]:
            raise ValueError("frozen catalogue sample hash differs from the protocol")
        disjointness = shakedown_disjointness(value)
        if not disjointness["proven"]:
            raise ValueError("shakedown/evidentiary designs are not disjoint")
        scope = import_scope_report(value)
        if frozen is not None:
            authorities = frozen.authorities
            if authorities["protocol_semantic_sha256"] != semantic_sha256(value):
                raise ValueError("protocol semantic authority differs")
            if authorities["source_sha256"] != contract["source_sha256"]:
                raise ValueError("source hash differs from the preregistered authority")
            if authorities["package_versions"] != contract["observed_package_versions"]:
                raise ValueError("package versions differ from the preregistered authority")
            if authorities["evidentiary_design_sha256"] != design_sha256(value, plan):
                raise ValueError("evidentiary design differs from the preregistered authority")
            if (
                hashlib.sha256(frozen.shakedown_bytes).hexdigest() != authorities["shakedown_file_sha256"]
                or semantic_sha256(frozen.shakedown) != authorities["shakedown_semantic_sha256"]
            ):
                raise ValueError("shakedown record differs from the preregistered authority")
            verify_shakedown_record(value, frozen.shakedown)
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/campaign-plan.json", plan_record(plan))
        context.write_json("artifacts/code-contract.json", {**contract, "import_scope_at_prebundle": scope})
        context.write_json("artifacts/protocol-consistency.json", consistency)
        context.write_json("artifacts/catalogue-binding.json", binding)
        context.write_json(
            "artifacts/catalogue.json",
            {"catalogue_sha256": catalogue_sha, "designs": [catalogue_design_summary(d) for d in designs]},
        )
        context.write_json(
            "artifacts/uncertain-sample.json",
            {
                "unit_rows_sha256": cat.unit_rows_sha256(),
                "catalogue_sample_sha256": sample_sha,
                "unit_rows": [list(row) for row in cat.unit_qmc_rows()],
                "nominal": [dict(theta) for theta in evaluation_context.nominal],
                "sample": [[dict(theta) for theta in rows] for rows in evaluation_context.sample],
            },
        )
        context.write_json(
            "artifacts/runtime.json",
            {
                "generated_at_utc": datetime.now(timezone.utc),
                "python": sys.version,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "package_versions": contract["observed_package_versions"],
            },
        )
        if frozen is not None:
            context.write_json("artifacts/authorities.json", frozen.authorities)
            context.write_blob("artifacts/shakedown.json", frozen.shakedown_bytes)
        else:
            context.write_json(
                "artifacts/shakedown-disclosure.json",
                {
                    "evidentiary": False,
                    "outcomes_enter_estimand": False,
                    "statement": value["shakedown"]["purpose"],
                    "disjointness": disjointness,
                },
            )
        state["context"] = evaluation_context
        state["contract"] = contract
        collector["prebundle"] = {
            "source_sha256": contract["source_sha256"],
            "catalogue_sample_sha256": sample_sha,
            "catalogue_sha256": catalogue_sha,
        }
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "source_sha256": contract["source_sha256"],
            "catalogue_sample_sha256": sample_sha,
            "run_count": len(plan.run_ids),
        }

    def development(context: Any) -> Decision:
        started = time.perf_counter()
        _log("development: device probes")
        cpu_probe = opt.torch_environment("cpu")
        cuda_probe: dict[str, Any]
        try:
            cuda_probe = opt.torch_environment("cuda:0")
            cuda_probe["available"] = True
        except Exception as error:  # recorded, never hidden
            cuda_probe = {"available": False, "error": f"{type(error).__name__}: {error}"}
        context.write_json("artifacts/device-probes.json", {"cpu": cpu_probe, "cuda": cuda_probe})
        evaluation_context: m.EvaluationContext = state["context"]
        total = plan.dense_reference_points_per_design * m.CATALOGUE_SIZE
        _log(f"development: dense reference ({m.CATALOGUE_SIZE} designs x {plan.dense_reference_points_per_design} = {total})")
        context.before_expensive(
            "dense-reference",
            kind="solver",
            details={"designs": m.CATALOGUE_SIZE, "points_per_design": plan.dense_reference_points_per_design, "plan_kind": plan.kind},
        )
        reference = dense_reference(
            evaluation_context,
            plan.dense_reference_points_per_design,
            int(value["dense_reference"]["seed"]),
            max_workers=int(value["dense_reference"]["max_workers"]),
        )
        _log(
            f"development: dense reference done in {reference['evaluation_seconds']:.1f}s "
            f"(fronts {reference['front_and_hypervolume_seconds']:.1f}s; robust front {reference['fronts']['robust']['front_size']})"
        )
        separability = separability_check(reference["records"])
        stride = max(1, len(reference["records"]) // 64)
        replay = replay_records(reference["records"][::stride], evaluation_context)
        context.write_json(
            "artifacts/dense-reference.json",
            {key: item for key, item in reference.items() if key != "records"},
        )
        context.write_json(
            "artifacts/dense-reference-summary.json",
            {
                "designs": reference["designs"],
                "points_per_design": reference["points_per_design"],
                "count": reference["count"],
                "feasible": reference["feasible"],
                "infeasible": reference["infeasible"],
                "workers": reference["workers"],
                "evaluation_seconds": reference["evaluation_seconds"],
                "front_and_hypervolume_seconds": reference["front_and_hypervolume_seconds"],
                "columns_sha256": reference["columns_sha256"],
                "robust_hypervolume": reference["fronts"]["robust"]["hypervolume"],
                "nominal_hypervolume": reference["fronts"]["nominal"]["hypervolume"],
                "robust_front_size": reference["fronts"]["robust"]["front_size"],
                "nominal_front_size": reference["fronts"]["nominal"]["front_size"],
                "robust_front_catalogue_indices": reference["fronts"]["robust"]["catalogue_indices"],
                "nominal_front_catalogue_indices": reference["fronts"]["nominal"]["catalogue_indices"],
                "separability": {"passed": separability["passed"], "tolerance_relative_spread": separability["tolerance_relative_spread"]},
                "replay": replay,
            },
        )
        context.write_json("artifacts/dense-reference-separability.json", separability)
        accepted = bool(
            cpu_probe["float64_cholesky_probe"] and reference["feasible"] > 0 and separability["passed"] and replay["passed"]
        )
        state["reference"] = {key: item for key, item in reference.items() if key != "records"}
        collector["development"] = {
            "seconds": time.perf_counter() - started,
            "accepted": accepted,
            "cuda_available": cuda_probe.get("available"),
            "dense_feasible": reference["feasible"],
            "dense_robust_hypervolume": reference["fronts"]["robust"]["hypervolume"],
            "dense_robust_front_catalogue_indices": reference["fronts"]["robust"]["catalogue_indices"],
            "separability_passed": separability["passed"],
            "replay_passed": replay["passed"],
        }
        return Decision(
            accepted,
            {
                "cpu_probe_passed": True,
                "cuda_probe_available": bool(cuda_probe.get("available")),
                "dense_feasible": reference["feasible"],
                "separability_passed": separability["passed"],
                "replay_passed": replay["passed"],
            },
        )

    def assessment(context: Any) -> Decision:
        started = time.perf_counter()
        evaluation_context: m.EvaluationContext = state["context"]
        qparams = value["optimizers"]["qlognehvi"]
        run_summaries: dict[str, dict[str, Any]] = {}
        run_infos: dict[str, dict[str, Any]] = {}
        all_records: list[dict[str, Any]] = []
        curves: dict[str, list[dict[str, Any]]] = {}
        checkpoint_root = context.cache_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for seed in plan.seeds:
            for strategy in plan.strategies:
                run_key = f"{strategy}:{seed}"
                context.before_expensive(
                    f"run-{strategy}-{seed}",
                    kind="solver",
                    details={"strategy": strategy, "seed": seed, "budget": plan.evaluations_per_run},
                )
                _log(f"assessment: run {run_key} start")

                def checkpoint(ledger: opt.RunLedger, key: str = run_key) -> None:
                    path = checkpoint_root / f"{key.replace(':', '-')}.json"
                    path.write_bytes(
                        canonical_bytes(
                            {
                                "run": key,
                                "evaluations": len(ledger.records),
                                "records": ledger.records,
                                "hypervolume_curve": ledger.hypervolume_curve,
                            }
                        )
                    )

                ledger = opt.RunLedger(
                    strategy=strategy, seed=seed, budget=plan.evaluations_per_run, context=evaluation_context, checkpoint=checkpoint
                )
                if strategy == "qlognehvi":
                    info = opt.run_qlognehvi(
                        ledger,
                        initial_count=plan.initial_design,
                        batch_size=plan.qlognehvi_batch_size,
                        device=qparams["device"],
                        torch_threads=int(qparams["torch_threads"]),
                        mc_samples=int(qparams["mc_samples"]),
                        candidates_per_design=int(qparams["candidates_per_design"]),
                        refine_maxiter=int(qparams["refine_maxiter"]),
                        fit_noise_floor=float(qparams["fit_noise_floor"]),
                        progress=lambda entry, key=run_key: _log(
                            f"  {key} it {entry['iteration']} n={entry['evaluations']} fit {entry['fit_seconds']:.1f}s "
                            f"cand {entry['candidate_stage_seconds']:.1f}s refine {entry['refinement_seconds']:.1f}s "
                            f"hv {entry['hypervolume']:.3e}"
                        ),
                    )
                elif strategy == "nsga3":
                    info = opt.run_nsga3(
                        ledger,
                        initial_count=plan.initial_design,
                        population_size=plan.nsga3_population_size,
                        generations=plan.nsga3_generations,
                        reference_direction_seed=int(value["optimizers"]["nsga3"]["reference_direction_seed"]),
                    )
                elif strategy == "lhs":
                    info = opt.run_lhs(ledger, initial_count=plan.initial_design)
                else:  # pragma: no cover - protocol consistency forbids this
                    raise ValueError(f"unknown strategy {strategy}")
                summary = ledger.summary()
                run_summaries[run_key] = summary
                run_infos[run_key] = info
                curves[run_key] = list(ledger.hypervolume_curve)
                for record in ledger.records:
                    record_copy = dict(record)
                    record_copy["run"] = run_key
                    all_records.append(record_copy)
                context.write_json(
                    f"artifacts/runs/{strategy}-{seed}.json",
                    {
                        "run": run_key,
                        "strategy": strategy,
                        "seed": seed,
                        "summary": summary,
                        "optimizer": info,
                        "hypervolume_curve": ledger.hypervolume_curve,
                        "records": ledger.records,
                    },
                )
                _log(
                    f"assessment: run {run_key} done hv={summary['final_hypervolume']:.3e} pareto={summary['pareto_set_size']} "
                    f"designs={summary['distinct_catalogue_designs']} infeasible={summary['infeasible_evaluations']} "
                    f"wall={summary['wall_clock_seconds']:.1f}s"
                )

        _log("assessment: metrics and gates")
        reference = state["reference"]
        reference_hv = reference["fronts"]["robust"]["hypervolume"]
        pooled = pooled_fronts(all_records, evaluation_context.designs)
        per_strategy = {
            strategy: pooled_fronts(
                [record for record in all_records if record["run"].startswith(strategy + ":")], evaluation_context.designs
            )
            for strategy in plan.strategies
        }
        variance = {
            strategy: _seed_variance([run_summaries[f"{strategy}:{seed}"]["final_hypervolume"] for seed in plan.seeds])
            for strategy in plan.strategies
        }
        timing = {
            key: {
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "evaluation_seconds": math.fsum(record["evaluation_seconds"] for record in all_records if record["run"] == key),
                "bo_fit_seconds": math.fsum(entry["fit_seconds"] for entry in run_infos[key].get("iteration_log", [])),
                "bo_acquisition_seconds": math.fsum(
                    entry["acquisition_seconds"] for entry in run_infos[key].get("iteration_log", [])
                ),
            }
            for key, summary in run_summaries.items()
        }
        _log("assessment: closure and width sensitivity")
        sensitivity = closure_and_width_sensitivity(all_records, value, pooled["robust"]["design_ids"], evaluation_context)
        separability = separability_check(all_records)

        # ---- binding gates (recording-integrity gates; acceptance != efficacy) --------
        replay = replay_records(all_records, evaluation_context)
        l0_domain_ok = all(
            (
                record["status"] == "success"
                and record["constraints"][m.ROBUST_CONSTRAINT.name] >= 0.0
                and record["robust_objectives"] is not None
                and all(math.isfinite(v) for v in record["robust_objectives"].values())
            )
            or (
                record["status"] == "infeasible"
                and record["constraints"][m.ROBUST_CONSTRAINT.name] < 0.0
                and record["failure_code"] == m.INFEASIBLE_CODE
                and record["robust_objectives"] is None
            )
            for record in all_records
        )
        monotone = hypervolume_monotonicity(curves)
        monotone_ok = monotone["passed"]
        budget_ok = all(summary["evaluations"] == plan.evaluations_per_run for summary in run_summaries.values())
        shared_ok = all(
            len(
                {
                    canonical_bytes(
                        [
                            [record["design"]["catalogue_index"], record["design"]["values"]]
                            for record in all_records
                            if record["run"] == f"{strategy}:{seed}"
                        ][: plan.initial_design]
                    )
                    for strategy in plan.strategies
                }
            )
            == 1
            for seed in plan.seeds
        )
        sample_ok = (
            evaluation_context.sample_sha256 == value["uncertain_inputs"]["sample"]["catalogue_sample_sha256"]
            and cat.unit_rows_sha256() == value["uncertain_inputs"]["sample"]["unit_rows_sha256"]
        )
        pareto_ok = True
        pareto_sets = {}
        for key, summary in run_summaries.items():
            run_records = [record for record in all_records if record["run"] == key]
            by_index = {record["index"]: record for record in run_records}
            members = [by_index[i] for i in summary["pareto_record_indices"]]
            member_replay = replay_records(members, evaluation_context)
            successful = [
                (record["index"], m.normalized_objectives(_robust_vector(record))) for record in run_records if record["status"] == "success"
            ]
            front = {successful[i][0] for i in m.nondominated_indices([row[1] for row in successful])}
            nondominated_ok = set(summary["pareto_record_indices"]) == front
            pareto_ok = pareto_ok and member_replay["passed"] and nondominated_ok
            pareto_sets[key] = {
                "size": len(members),
                "replay_passed": member_replay["passed"],
                "nondominated_recomputed": nondominated_ok,
                "catalogue_indices": summary["pareto_catalogue_indices"],
                "designs": [
                    {
                        "index": record["index"],
                        "design_id": record["design"]["design_id"],
                        "catalogue_index": record["design"]["catalogue_index"],
                        "case_id": record["design"]["case_id"],
                        "values": record["design"]["values"],
                        "robust_objectives": record["robust_objectives"],
                        "nominal_objectives": record["nominal_objectives"],
                    }
                    for record in members
                ],
            }
        contract_ok = bool(
            state["contract"]["matches"]
            and (
                frozen is None
                or (
                    frozen.authorities["source_sha256"] == state["contract"]["source_sha256"]
                    and frozen.authorities["protocol_semantic_sha256"] == semantic_sha256(value)
                )
            )
        )
        binding_report = cat.binding_report(value["catalogue_binding"])
        scope = import_scope_report(value)
        duplicates = {
            key: {
                "evaluations": summary["evaluations"],
                "unique_designs": summary["unique_designs"],
                "duplicates": summary["evaluations"] - summary["unique_designs"],
            }
            for key, summary in run_summaries.items()
            if key.startswith("nsga3:")
        }
        duplicates_ok = all(item["duplicates"] == 0 for item in duplicates.values())
        labels = label_checks(run_infos, plan)
        binding = {
            "replay_bit_exact": {"passed": replay["passed"], "replayed": replay["replayed"], "mismatches": replay["mismatches"]},
            "l0_domain": {"passed": l0_domain_ok},
            "hypervolume_monotone": monotone,
            "budget_exact": {"passed": budget_ok},
            "shared_initial_design": {"passed": shared_ok},
            "sample_hash": {"passed": sample_ok},
            "pareto_replay": {"passed": pareto_ok},
            "code_contract": {"passed": contract_ok},
            "catalogue_binding": {"passed": binding_report["passed"], "checks": binding_report["checks"]},
            "code_hash_scope_matches_imports": {
                "passed": scope["matches"],
                "imported_not_in_scope": scope["imported_not_in_scope"],
                "in_scope_not_imported": scope["in_scope_not_imported"],
                "imported_count": len(scope["imported"]),
            },
            "nsga3_duplicates_eliminated": {"passed": duplicates_ok, "runs": duplicates},
            "labels_consistent": labels,
        }
        reported = {
            "bo_beats_random": _paired_test(run_summaries, plan.seeds, "qlognehvi", "lhs"),
            "bo_beats_nsga3": _paired_test(run_summaries, plan.seeds, "qlognehvi", "nsga3"),
            "hypervolume_vs_dense_reference": {
                key: summary["final_hypervolume"] / reference_hv if reference_hv > 0 else None
                for key, summary in run_summaries.items()
            },
            "robust_vs_nominal": {
                "robust_front_size": pooled["robust"]["front_size"],
                "nominal_front_size": pooled["nominal"]["front_size"],
                "shared_designs": len(pooled["shared_design_ids"]),
                "jaccard": pooled["jaccard_robust_nominal"],
                "nominal_front_members_robust_feasible": pooled["nominal"]["robust_feasible_members"],
                "robust_hypervolume": pooled["robust"]["hypervolume"],
                "nominal_hypervolume": pooled["nominal"]["hypervolume"],
                "robust_front_catalogue_indices": pooled["robust"]["catalogue_indices"],
                "nominal_front_catalogue_indices": pooled["nominal"]["catalogue_indices"],
            },
            "closure_cl1_vs_cl2": {
                key: sensitivity["closure_cl2"][key]
                for key in (
                    "feasible",
                    "infeasible",
                    "front_size",
                    "hypervolume",
                    "campaign_hypervolume",
                    "shared_with_campaign_front",
                    "jaccard_with_campaign_front",
                    "front_catalogue_indices",
                    "campaign_front_catalogue_indices",
                )
            },
            "uncertainty_width_sensitivity": [
                {
                    key: item[key]
                    for key in (
                        "width_scale",
                        "is_campaign_posterior",
                        "survival_min",
                        "survival_max",
                        "feasible",
                        "infeasible",
                        "front_size",
                        "hypervolume",
                        "shared_with_campaign_front",
                        "jaccard_with_campaign_front",
                        "common_feasible_designs",
                        "identical_on_common_feasible_set_up_to_ties",
                        "front_catalogue_indices",
                    )
                }
                for item in sensitivity["widths"]
            ],
            "robust_front_catalogue_designs": pooled["robust"]["catalogue_membership"],
            "per_design_separability": {"passed": separability["passed"], "designs": len(separability["per_design"])},
        }
        all_binding = all(item["passed"] for item in binding.values())
        gates = {
            "semantics": value["gates"]["semantics"],
            "binding": binding,
            "reported_not_binding": reported,
            "all_binding_passed": all_binding,
            "binding_in_this_plan": plan.binding_gates,
        }
        metrics = {
            "runs": run_summaries,
            "hypervolume_table": {
                key: {
                    "final_hypervolume": summary["final_hypervolume"],
                    "attained_fraction_of_dense_reference": summary["final_hypervolume"] / reference_hv if reference_hv > 0 else None,
                    "pareto_set_size": summary["pareto_set_size"],
                    "pareto_catalogue_indices": summary["pareto_catalogue_indices"],
                    "distinct_catalogue_designs": summary["distinct_catalogue_designs"],
                    "unique_designs": summary["unique_designs"],
                    "infeasible_evaluations": summary["infeasible_evaluations"],
                    "wall_clock_seconds": summary["wall_clock_seconds"],
                }
                for key, summary in run_summaries.items()
            },
            "dense_reference": {
                "designs": reference["designs"],
                "points_per_design": reference["points_per_design"],
                "count": reference["count"],
                "robust_hypervolume": reference_hv,
                "nominal_hypervolume": reference["fronts"]["nominal"]["hypervolume"],
                "robust_front_size": reference["fronts"]["robust"]["front_size"],
                "nominal_front_size": reference["fronts"]["nominal"]["front_size"],
                "robust_front_catalogue_indices": reference["fronts"]["robust"]["catalogue_indices"],
            },
            "seed_variance": variance,
            "timing": timing,
            "per_strategy_pooled": {
                strategy: {
                    "robust_front_size": item["robust"]["front_size"],
                    "robust_hypervolume": item["robust"]["hypervolume"],
                    "nominal_front_size": item["nominal"]["front_size"],
                    "robust_front_catalogue_indices": item["robust"]["catalogue_indices"],
                    "distinct_catalogue_designs": item["distinct_catalogue_designs"],
                }
                for strategy, item in per_strategy.items()
            },
        }
        context.write_json("artifacts/hypervolume-curves.json", curves)
        context.write_json("artifacts/metrics.json", metrics)
        context.write_json("artifacts/pareto-sets.json", pareto_sets)
        context.write_json("artifacts/pooled-fronts.json", pooled)
        context.write_json("artifacts/per-strategy-fronts.json", per_strategy)
        context.write_json("artifacts/sensitivity.json", sensitivity)
        context.write_json("artifacts/separability.json", separability)
        context.write_json("artifacts/import-scope.json", scope)
        context.write_json("artifacts/gates.json", gates)
        campaign_result = {
            "schema_version": schema("campaign-result"),
            "experiment_id": value["experiment_id"],
            "plan_kind": plan.kind,
            "evidentiary": plan.kind == "evidentiary",
            "classification": value["classification"],
            "claim_boundary": value["claim_boundary"]["statement"],
            "closure": value["closures"]["CL-1"]["id"],
            "closure_identification_disclosure": value["closures"]["CL-1"]["identification_disclosure"],
            "sensitivity_closure": value["closures"]["CL-2"]["id"],
            "gate_semantics": value["gates"]["semantics"],
            "runs": len(run_summaries),
            "total_evaluations": len(all_records),
            "infeasible_evaluations": sum(1 for record in all_records if record["status"] != "success"),
            "hypervolume_table": metrics["hypervolume_table"],
            "dense_reference": metrics["dense_reference"],
            "seed_variance": variance,
            "bo_beats_random": reported["bo_beats_random"]["passed"],
            "bo_beats_random_wins": f"{reported['bo_beats_random']['wins']}/{reported['bo_beats_random']['seeds']}",
            "bo_beats_nsga3": reported["bo_beats_nsga3"]["passed"],
            "bo_beats_nsga3_wins": f"{reported['bo_beats_nsga3']['wins']}/{reported['bo_beats_nsga3']['seeds']}",
            "robust_vs_nominal": reported["robust_vs_nominal"],
            "closure_cl1_vs_cl2": reported["closure_cl1_vs_cl2"],
            "uncertainty_width_sensitivity": reported["uncertainty_width_sensitivity"],
            "robust_front_catalogue_designs": [
                {
                    "catalogue_index": item["catalogue_index"],
                    "case_id": item["case_id"],
                    "front_members": item["front_members"],
                    "pooled_wall_hit_probability": item["pooled"]["probability"],
                    "pooled_wilson_95": item["pooled"]["wilson_95"],
                    "cell_wall_hit_probabilities": [cell["probability"] for cell in item["cells"]],
                    "nominal_survival_cl1": item["nominal_survival_cl1"],
                    "geometry": item["geometry"],
                }
                for item in pooled["robust"]["catalogue_membership"]
            ],
            "all_binding_gates_passed": all_binding,
            "assessment_seconds": time.perf_counter() - started,
        }
        context.write_json("artifacts/campaign-result.json", campaign_result)
        collector["assessment"] = {
            "runs": run_summaries,
            "gates": gates,
            "metrics": metrics,
            "campaign_result": campaign_result,
            "import_scope": scope,
            "seconds": time.perf_counter() - started,
        }
        return Decision(
            all_binding,
            {
                "all_binding_gates_passed": all_binding,
                "binding_gate_results": {name: item["passed"] for name, item in binding.items()},
                "bo_beats_random": reported["bo_beats_random"]["passed"],
                "bo_beats_nsga3": reported["bo_beats_nsga3"]["passed"],
                "runs": len(run_summaries),
                "total_evaluations": len(all_records),
            },
        )

    return RuntimeCallbacks(prebundle=prebundle, development=development, assessment=assessment)
