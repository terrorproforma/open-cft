"""Preregistered MDO L0 campaign v1: contract binding, plans, callbacks and gates.

One :class:`CampaignPlan` drives both the evidentiary campaign and the disclosed
NON-EVIDENTIARY shakedown so the shakedown exercises exactly the production
code (model, optimiser adapters, metrics, gates, export).
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import (
    canonical_bytes,
    semantic_sha256,
    strict_json_file,
)
from cft_revival.optimization import ObjectiveDirection
from cft_revival.optimization.sampling import initial_designs

from . import model as m
from . import optimizers as opt

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"

VERSION_TAG = "cft-revival.mdo-l0-campaign-v1"
PACKAGES = ("torch", "botorch", "gpytorch", "pymoo", "numpy", "scipy")
TIE_TOLERANCE = 1e-9


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
    declared_variables = [
        (item["name"], float(item["lower"]), float(item["upper"]), item["units"])
        for item in value["design_variables"]
    ]
    checks["design_variables"] = declared_variables == [
        (v.name, v.lower, v.upper, v.units) for v in m.DESIGN_VARIABLES
    ]
    declared_uncertain = [
        (item["name"], float(item["lower"]), float(item["upper"]), item["units"])
        for item in value["uncertain_inputs"]["inputs"]
    ]
    checks["uncertain_inputs"] = declared_uncertain == list(m.UNCERTAIN_INPUTS)
    checks["uniform_priors"] = all(
        item["distribution"] == "uniform" for item in value["uncertain_inputs"]["inputs"]
    )
    sample_spec = value["uncertain_inputs"]["sample"]
    checks["sample_parameters"] = (
        tuple(sample_spec["bases"]) == m.QMC_BASES
        and sample_spec["seed"] == m.QMC_SEED
        and sample_spec["count"] == m.QMC_SAMPLE_SIZE
        and sample_spec["frozen"] is True
    )
    checks["sample_sha256"] = m.sample_sha256(m.uncertain_sample()) == sample_spec["sha256"]
    checks["objectives"] = [
        (item["name"], item["direction"], item["units"], float(item["comparison_scale"]))
        for item in value["objectives"]
    ] == [(o.name, o.direction.value, o.units, o.comparison_scale) for o in m.OBJECTIVES]
    checks["reference_point"] = all(
        float(value["reference_point"][name]) == m.REFERENCE_POINT[name]
        for name in m.OBJECTIVE_NAMES
    )
    constraints = {item["name"]: item for item in value["constraints"]}
    checks["constraints"] = (
        set(constraints) == {m.ROBUST_CONSTRAINT.name, m.NOMINAL_CONSTRAINT.name}
        and all(
            item["sense"] == ">="
            and float(item["threshold"]) == 0.0
            and float(item["violation_scale"]) == spec.violation_scale
            for item, spec in (
                (constraints[m.ROBUST_CONSTRAINT.name], m.ROBUST_CONSTRAINT),
                (constraints[m.NOMINAL_CONSTRAINT.name], m.NOMINAL_CONSTRAINT),
            )
        )
    )
    robust = value["robust_formulation"]
    checks["robust_formulation"] = (
        robust["risk_measure"] == "CVaR"
        and float(robust["tail_fraction"]) == m.CVAR_TAIL_FRACTION
        and int(robust["tail_count"]) == m.tail_count(m.QMC_SAMPLE_SIZE)
    )
    checks["closures"] = (
        value["closures"]["CL-1"]["id"] == m.CLOSURE_ID
        and {
            key: float(item)
            for key, item in value["closures"]["fixed"].items()
            if key != "note"
        }
        == m.FIXED_CLOSURES
    )
    budget = value["budget"]
    checks["budget_arithmetic"] = (
        budget["initial_design"] + budget["qlognehvi_batch_size"] * budget["qlognehvi_iterations"]
        == budget["evaluations_per_run"]
        and budget["nsga3_population_size"] * budget["nsga3_generations"]
        == budget["evaluations_per_run"]
        and budget["nsga3_population_size"] == budget["initial_design"]
        and budget["total_evaluations"]
        == budget["evaluations_per_run"] * len(budget["seeds"]) * len(budget["strategies"])
        and tuple(budget["strategies"]) == opt.STRATEGIES
    )
    shakedown = value["shakedown"]
    checks["shakedown_arithmetic"] = (
        shakedown["initial_design"]
        + shakedown["qlognehvi_batch_size"] * shakedown["qlognehvi_iterations"]
        == shakedown["evaluations_per_run"]
        and shakedown["nsga3_population_size"] * shakedown["nsga3_generations"]
        == shakedown["evaluations_per_run"]
        and shakedown["nsga3_population_size"] == shakedown["initial_design"]
    )
    checks["seed_namespaces"] = all(seed < 1000 for seed in budget["seeds"]) and all(
        seed >= 900_000 for seed in shakedown["seeds"]
    )
    checks["cusp_prior_calibration"] = (
        abs(
            (1.0 - 0.5 * m.UNCERTAIN_INPUTS[0][2]) ** 4
            - (1.0 - 2962 / 4608)
        )
        < 0.005
    )
    return checks


def require_protocol_consistency(value: Mapping[str, Any]) -> dict[str, bool]:
    checks = protocol_consistency(value)
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError("protocol/module mismatch: " + ", ".join(failed))
    return checks


# --------------------------------------------------------------------------
# Code contract
# --------------------------------------------------------------------------


def source_files(value: Mapping[str, Any]) -> list[Path]:
    files: list[Path] = []
    for pattern in value["code_contract"]["source_hash_scope"]:
        if not pattern.startswith("modern/"):
            raise ValueError(f"source scope entry must start with modern/: {pattern}")
        relative = pattern[len("modern/") :]
        directory, _, name = relative.rpartition("/")
        matches = sorted((MODERN / directory).glob(name))
        if not matches:
            raise ValueError(f"source scope entry matched nothing: {pattern}")
        files.extend(matches)
    return files


def source_hash_report(value: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    digest = hashlib.sha256()
    for path in source_files(value):
        data = path.read_bytes()
        if b"\r" in data:
            raise ValueError(
                f"hashed source contains a carriage return (CRLF checkout?): {path}"
            )
        relative = path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
        file_sha = hashlib.sha256(data).hexdigest()
        entries.append({"path": relative, "sha256": file_sha, "bytes": len(data)})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {"source_sha256": digest.hexdigest(), "files": entries, "line_endings": "LF"}


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
    python_ok = sys.version_info[:2] == tuple(
        int(part) for part in value["code_contract"]["python"].split(".")
    )
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
            f"{report['observed_package_versions']} vs declared "
            f"{report['declared_package_versions']}; python ok={report['python_minor_matches']}"
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
    dense_reference_count: int
    binding_gates: bool

    def __post_init__(self) -> None:
        if self.kind not in {"evidentiary", "shakedown"}:
            raise ValueError("plan kind must be evidentiary or shakedown")
        if (
            self.initial_design + self.qlognehvi_batch_size * self.qlognehvi_iterations
            != self.evaluations_per_run
            or self.nsga3_population_size * self.nsga3_generations != self.evaluations_per_run
            or self.nsga3_population_size != self.initial_design
        ):
            raise ValueError("plan budget arithmetic is inconsistent")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("plan seeds must be unique and non-empty")

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.kind}:{strategy}:{seed}" for seed in self.seeds for strategy in self.strategies
        )


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
        dense_reference_count=int(value["dense_reference"]["count"]),
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
        dense_reference_count=int(shakedown["dense_reference_count"]),
        binding_gates=False,
    )


def plan_record(plan: CampaignPlan) -> dict[str, Any]:
    """Plain-JSON plan record (lists, not tuples, so re-parsed records re-hash)."""

    record = {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in asdict(plan).items()
    }
    record["run_ids"] = list(plan.run_ids)
    return record


def design_sha256(value: Mapping[str, Any], plan: CampaignPlan) -> str:
    """Identity of a plan's complete evaluation design (seeds, budgets, sample)."""

    return semantic_sha256(
        {
            "plan": plan_record(plan),
            "sample_sha256": value["uncertain_inputs"]["sample"]["sha256"],
            "design_variables": [asdict(v) for v in m.DESIGN_VARIABLES],
            "objectives": list(m.OBJECTIVE_NAMES),
            "reference_point": m.REFERENCE_POINT,
        }
    )


def shakedown_disjointness(value: Mapping[str, Any]) -> dict[str, Any]:
    evidentiary = evidentiary_plan(value)
    shakedown = shakedown_plan(value)
    seed_overlap = sorted(set(evidentiary.seeds) & set(shakedown.seeds))
    run_overlap = sorted(set(evidentiary.run_ids) & set(shakedown.run_ids))
    namespace = all(seed < 1000 for seed in evidentiary.seeds) and all(
        seed >= 900_000 for seed in shakedown.seeds
    )
    # Different seeds give different LHS initial designs; prove it.
    initial_overlap = 0
    evidentiary_rows = {
        tuple(row)
        for seed in evidentiary.seeds
        for row in opt.shared_initial_points(seed, evidentiary.initial_design).tolist()
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


def verify_shakedown_record(
    value: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = (
        record.get("protocol_semantic_sha256") == semantic_sha256(value)
    )
    try:
        contract = code_contract_report(value)
        checks["source_sha256_current"] = (
            record.get("source_sha256") == contract["source_sha256"]
        )
        checks["package_versions_current"] = (
            record.get("package_versions") == contract["observed_package_versions"]
            and contract["matches"]
        )
    except Exception:
        checks["source_sha256_current"] = False
        checks["package_versions_current"] = False
    disjointness = record.get("disjointness")
    checks["disjointness_proven"] = (
        isinstance(disjointness, Mapping) and disjointness.get("proven") is True
    )
    try:
        checks["shakedown_design_sha256_current"] = (
            record.get("shakedown_design_sha256") == design_sha256(value, shakedown_plan(value))
        )
        checks["evidentiary_design_sha256_current"] = (
            record.get("evidentiary_design_sha256")
            == design_sha256(value, evidentiary_plan(value))
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
    checks["all_runs_budget_exact"] = isinstance(runs, Mapping) and bool(runs) and all(
        item.get("evaluations") == item.get("budget") for item in runs.values()
    )
    gates = record.get("gates")
    checks["all_binding_gates_passed"] = (
        isinstance(gates, Mapping)
        and isinstance(gates.get("binding"), Mapping)
        and bool(gates["binding"])
        and all(item.get("passed") is True for item in gates["binding"].values())
    )
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
    return FrozenAuthority(
        strict_json_file(AUTHORITIES_PATH),
        strict_json_file(SHAKEDOWN_PATH),
        SHAKEDOWN_PATH.read_bytes(),
    )


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------


def _robust_vector(record: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(record["robust_objectives"][name] for name in m.OBJECTIVE_NAMES)


def _nominal_vector(record: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(record["nominal_objectives"][name] for name in m.OBJECTIVE_NAMES)


def pooled_fronts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Robust and nominal nondominated sets of a pool of evaluation records."""

    unique: dict[str, Mapping[str, Any]] = {}
    for record in records:
        unique.setdefault(record["design"]["design_id"], record)
    ordered = list(unique.values())
    robust_rows = [
        (record, m.normalized_objectives(_robust_vector(record)))
        for record in ordered
        if record["status"] == "success"
    ]
    nominal_rows = [
        (record, m.normalized_objectives(_nominal_vector(record)))
        for record in ordered
        if record["nominal_objectives"] is not None
    ]
    robust_front = [
        robust_rows[index][0] for index in m.nondominated_indices([row[1] for row in robust_rows])
    ]
    nominal_front = [
        nominal_rows[index][0]
        for index in m.nondominated_indices([row[1] for row in nominal_rows])
    ]
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

    return {
        "unique_designs": len(ordered),
        "robust": {
            "candidates": len(robust_rows),
            "front_size": len(robust_front),
            "hypervolume": m.hypervolume([row[1] for row in robust_rows]),
            "design_ids": sorted(robust_ids),
            "objective_ranges": objective_ranges(robust_front, "robust_objectives"),
            "designs": [
                {
                    "design_id": record["design"]["design_id"],
                    "values": record["design"]["values"],
                    "robust_objectives": record["robust_objectives"],
                    "nominal_objectives": record["nominal_objectives"],
                    "constraints": record["constraints"],
                }
                for record in robust_front
            ],
        },
        "nominal": {
            "candidates": len(nominal_rows),
            "front_size": len(nominal_front),
            "hypervolume": m.hypervolume([row[1] for row in nominal_rows]),
            "design_ids": sorted(nominal_ids),
            "objective_ranges": objective_ranges(nominal_front, "nominal_objectives"),
            "robust_feasible_members": nominal_feasible_robust,
            "designs": [
                {
                    "design_id": record["design"]["design_id"],
                    "values": record["design"]["values"],
                    "nominal_objectives": record["nominal_objectives"],
                    "robust_objectives": record["robust_objectives"],
                    "constraints": record["constraints"],
                }
                for record in nominal_front
            ],
        },
        "shared_design_ids": sorted(robust_ids & nominal_ids),
        "jaccard_robust_nominal": (
            len(robust_ids & nominal_ids) / len(robust_ids | nominal_ids)
            if robust_ids | nominal_ids
            else 1.0
        ),
    }


def dense_reference(count: int, seed: int) -> dict[str, Any]:
    """Robust and nominal reference fronts from a dense shifted-Halton design."""

    sample = m.uncertain_sample()
    nominal = m.nominal_theta()
    designs = initial_designs(m.DESIGN_VARIABLES, count, seed=seed)
    tick = time.perf_counter()
    records = []
    for index, design in enumerate(designs):
        evaluation = m.evaluate_design(design.values, sample, nominal=nominal)
        record = evaluation.to_record()
        record["index"] = index
        record["provenance"] = design.provenance
        records.append(record)
    seconds = time.perf_counter() - tick
    fronts = pooled_fronts(records)
    return {
        "count": len(records),
        "seed": seed,
        "evaluation_seconds": seconds,
        "feasible": sum(1 for record in records if record["status"] == "success"),
        "infeasible": sum(1 for record in records if record["status"] != "success"),
        "fronts": fronts,
        "records": records,
    }


def compact_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Columnar form of evaluation records for the large dense-reference artifact."""

    def column(getter: Any) -> list[Any]:
        return [getter(record) for record in records]

    return {
        "count": len(records),
        "design_variables": [variable.name for variable in m.DESIGN_VARIABLES],
        "objectives": list(m.OBJECTIVE_NAMES),
        "values": column(lambda r: r["design"]["values"]),
        "design_id": column(lambda r: r["design"]["design_id"]),
        "status": column(lambda r: r["status"]),
        m.ROBUST_CONSTRAINT.name: column(lambda r: r["constraints"][m.ROBUST_CONSTRAINT.name]),
        m.NOMINAL_CONSTRAINT.name: column(lambda r: r["constraints"][m.NOMINAL_CONSTRAINT.name]),
        "robust_objectives": column(
            lambda r: None
            if r["robust_objectives"] is None
            else [r["robust_objectives"][name] for name in m.OBJECTIVE_NAMES]
        ),
        "nominal_objectives": column(
            lambda r: None
            if r["nominal_objectives"] is None
            else [r["nominal_objectives"][name] for name in m.OBJECTIVE_NAMES]
        ),
        "sample_result_sha256": column(lambda r: r["sample_result_sha256"]),
    }


def separability_check(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Under CL-1 robust/nominal objective ratios are design-independent (except power)."""

    ratios: dict[str, list[float]] = {name: [] for name in m.OBJECTIVE_NAMES}
    for record in records:
        if record["status"] != "success" or record["nominal_objectives"] is None:
            continue
        for name in m.OBJECTIVE_NAMES:
            nominal = record["nominal_objectives"][name]
            robust = record["robust_objectives"][name]
            if nominal != 0.0:
                ratios[name].append(robust / nominal)
    report = {}
    passed = True
    for name, column in ratios.items():
        if not column:
            report[name] = None
            passed = False
            continue
        low, high = min(column), max(column)
        spread = (high - low) / max(abs(high), 1e-300)
        report[name] = {"ratio_min": low, "ratio_max": high, "relative_spread": spread, "count": len(column)}
        passed = passed and spread <= 1e-9
    return {"passed": passed, "ratios": report, "tolerance_relative_spread": 1e-9}


def replay_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute every record through the model and compare bit-exactly."""

    sample = m.uncertain_sample()
    nominal = m.nominal_theta()
    mismatches = []
    for record in records:
        evaluation = m.evaluate_design(record["design"]["values"], sample, nominal=nominal)
        replayed = evaluation.to_record()
        for key in (
            "design",
            "constraints",
            "status",
            "failure_code",
            "robust_objectives",
            "robust_statistics",
            "nominal_objectives",
            "sample_result_sha256",
        ):
            if canonical_bytes(replayed[key]) != canonical_bytes(record[key]):
                mismatches.append({"index": record.get("index"), "key": key})
                break
    return {"replayed": len(records), "mismatches": mismatches, "passed": not mismatches}


def cusp_sensitivity(
    records: Sequence[Mapping[str, Any]],
    value: Mapping[str, Any],
    campaign_design_ids: Sequence[str],
) -> dict[str, Any]:
    """Re-evaluate the pooled designs under alternative cusp priors and scenarios."""

    unique: dict[str, Mapping[str, Any]] = {}
    for record in records:
        unique.setdefault(record["design"]["design_id"], record)
    designs = [(key, record["design"]["values"]) for key, record in unique.items()]
    campaign_set = set(campaign_design_ids)
    campaign_rows = {
        record["design"]["design_id"]: m.normalized_objectives(_robust_vector(record))
        for record in unique.values()
        if record["status"] == "success"
    }

    def front_of(rows: Mapping[str, tuple[float, ...]], tolerance: float = 0.0) -> set[str]:
        keys = list(rows)
        return {
            keys[index]
            for index in m.nondominated_indices(
                [rows[key] for key in keys], relative_tolerance=tolerance
            )
        }

    priors = []
    for upper in value["uncertain_inputs"]["sensitivity_priors"]["cusp_upper_bounds"]:
        sample = m.uncertain_sample(cusp_upper=float(upper))
        nominal = m.nominal_theta(cusp_upper=float(upper))
        rows: dict[str, tuple[float, ...]] = {}
        infeasible = 0
        for design_id, values in designs:
            evaluation = m.evaluate_design(values, sample, nominal=nominal)
            if evaluation.status == "success":
                rows[design_id] = m.normalized_objectives(evaluation.robust_objectives)
            else:
                infeasible += 1
        front_ids = front_of(rows)
        union = front_ids | campaign_set
        # The predeclared expectation applies on the COMMON feasible set: the
        # feasible set itself moves with the prior (max beam current changes).
        common = set(rows) & set(campaign_rows)
        alternative_common_front = front_of({key: rows[key] for key in common})
        campaign_common_front = front_of({key: campaign_rows[key] for key in common})
        # Exact set equality can break on floating-point ties: L0 recomputes the
        # anode power as Ua * beam_current / beam_fraction, which differs from
        # Ua * Ia by an ulp depending on theta, so two designs with equal Ua*Ia
        # may tie in one frame and differ by one ulp in the other.  The
        # roundoff-aware fronts (relative tolerance 1e-9) remove that artefact.
        differences = alternative_common_front ^ campaign_common_front
        tolerant_alternative = front_of({key: rows[key] for key in common}, TIE_TOLERANCE)
        tolerant_campaign = front_of({key: campaign_rows[key] for key in common}, TIE_TOLERANCE)
        priors.append(
            {
                "cusp_upper": float(upper),
                "survival_min": min(m.cusp_survival(theta) for theta in sample),
                "survival_max": max(m.cusp_survival(theta) for theta in sample),
                "survival_mean": statistics.fmean(m.cusp_survival(theta) for theta in sample),
                "feasible": len(rows),
                "infeasible": infeasible,
                "front_size": len(front_ids),
                "hypervolume": m.hypervolume(list(rows.values())),
                "identical_to_campaign_front": front_ids == campaign_set,
                "jaccard_with_campaign_front": (
                    len(front_ids & campaign_set) / len(union) if union else 1.0
                ),
                "common_feasible_designs": len(common),
                "common_front_size": len(campaign_common_front),
                "identical_on_common_feasible_set": (
                    alternative_common_front == campaign_common_front
                ),
                "common_front_symmetric_difference": len(differences),
                "tolerant_common_front_size": len(tolerant_campaign),
                "tolerant_common_front_symmetric_difference": len(
                    tolerant_alternative ^ tolerant_campaign
                ),
                "identical_on_common_feasible_set_up_to_ties": (
                    tolerant_alternative == tolerant_campaign
                ),
                "tie_tolerance_relative": TIE_TOLERANCE,
                "front_design_ids": sorted(front_ids),
            }
        )
    scenarios = []
    base_nominal = m.nominal_theta()
    pareto = [(key, unique[key]["design"]["values"]) for key in campaign_design_ids if key in unique]
    for scenario in value["uncertain_inputs"]["sensitivity_scenarios"]:
        theta = dict(base_nominal)
        for name, probability in zip(m.CUSP_NAMES, scenario["cusp_probabilities"], strict=True):
            theta[name] = float(probability)
        survival = m.cusp_survival(theta)
        rows = []
        infeasible = 0
        for design_id, values in pareto:
            anode_current = values[1]
            margin = anode_current - m.beam_current_a(values[2], theta)
            if margin < 0.0:
                infeasible += 1
                continue
            objectives = m.evaluate_l0(values, theta)
            rows.append({"design_id": design_id, "objectives": dict(zip(m.OBJECTIVE_NAMES, objectives, strict=True))})
        ranges = {}
        for name in m.OBJECTIVE_NAMES:
            column = [row["objectives"][name] for row in rows]
            ranges[name] = {"minimum": min(column), "maximum": max(column)} if column else None
        scenarios.append(
            {
                "id": scenario["id"],
                "cusp_probabilities": list(scenario["cusp_probabilities"]),
                "survival": survival,
                "pareto_designs_evaluated": len(rows),
                "pareto_designs_infeasible": infeasible,
                "objective_ranges": ranges,
                "hypervolume": m.hypervolume(
                    [m.normalized_objectives([row["objectives"][name] for name in m.OBJECTIVE_NAMES]) for row in rows]
                ),
                "rows": rows,
            }
        )
    return {"priors": priors, "scenarios": scenarios, "unique_designs": len(designs)}


def _paired_test(
    summaries: Mapping[str, Mapping[str, Any]], seeds: Sequence[int], left: str, right: str
) -> dict[str, Any]:
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
        "pairs": pairs,
    }


def _seed_variance(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


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
        disjointness = shakedown_disjointness(value)
        if not disjointness["proven"]:
            raise ValueError("shakedown/evidentiary designs are not disjoint")
        sample = m.uncertain_sample()
        sample_sha = m.sample_sha256(sample)
        if sample_sha != value["uncertain_inputs"]["sample"]["sha256"]:
            raise ValueError("frozen QMC sample hash differs from the protocol")
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
                hashlib.sha256(frozen.shakedown_bytes).hexdigest()
                != authorities["shakedown_file_sha256"]
                or semantic_sha256(frozen.shakedown) != authorities["shakedown_semantic_sha256"]
            ):
                raise ValueError("shakedown record differs from the preregistered authority")
            verify_shakedown_record(value, frozen.shakedown)
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/campaign-plan.json", plan_record(plan))
        context.write_json("artifacts/code-contract.json", contract)
        context.write_json("artifacts/protocol-consistency.json", consistency)
        context.write_json(
            "artifacts/uncertain-sample.json",
            {"sha256": sample_sha, "nominal": m.nominal_theta(), "sample": [dict(theta) for theta in sample]},
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
        state["sample"] = sample
        state["contract"] = contract
        collector["prebundle"] = {"source_sha256": contract["source_sha256"], "sample_sha256": sample_sha}
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "source_sha256": contract["source_sha256"],
            "sample_sha256": sample_sha,
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
        _log(f"development: dense reference ({plan.dense_reference_count})")
        context.before_expensive(
            "dense-reference",
            kind="solver",
            details={"count": plan.dense_reference_count, "plan_kind": plan.kind},
        )
        reference = dense_reference(plan.dense_reference_count, int(value["dense_reference"]["seed"]))
        separability = separability_check(reference["records"])
        replay_subset = reference["records"][:: max(1, len(reference["records"]) // 32)]
        replay = replay_records(replay_subset)
        context.write_json(
            "artifacts/dense-reference.json",
            {
                **{key: item for key, item in reference.items() if key != "records"},
                "records": compact_records(reference["records"]),
            },
        )
        context.write_json(
            "artifacts/dense-reference-summary.json",
            {
                "count": reference["count"],
                "feasible": reference["feasible"],
                "infeasible": reference["infeasible"],
                "evaluation_seconds": reference["evaluation_seconds"],
                "robust_hypervolume": reference["fronts"]["robust"]["hypervolume"],
                "nominal_hypervolume": reference["fronts"]["nominal"]["hypervolume"],
                "robust_front_size": reference["fronts"]["robust"]["front_size"],
                "nominal_front_size": reference["fronts"]["nominal"]["front_size"],
                "separability": separability,
                "replay": replay,
            },
        )
        accepted = bool(
            cpu_probe["float64_cholesky_probe"]
            and reference["feasible"] > 0
            and separability["passed"]
            and replay["passed"]
        )
        state["reference"] = reference
        collector["development"] = {
            "seconds": time.perf_counter() - started,
            "accepted": accepted,
            "cuda_available": cuda_probe.get("available"),
            "dense_feasible": reference["feasible"],
            "dense_robust_hypervolume": reference["fronts"]["robust"]["hypervolume"],
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
        sample = state["sample"]
        nominal = m.nominal_theta()
        qparams = value["optimizers"]["qlognehvi"]
        nparams = value["optimizers"]["nsga3"]
        device = qparams["device"]
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
                    strategy=strategy,
                    seed=seed,
                    budget=plan.evaluations_per_run,
                    sample=sample,
                    nominal=nominal,
                    tail_fraction=m.CVAR_TAIL_FRACTION,
                    checkpoint=checkpoint,
                )
                if strategy == "qlognehvi":
                    info = opt.run_qlognehvi(
                        ledger,
                        initial_count=plan.initial_design,
                        batch_size=plan.qlognehvi_batch_size,
                        device=device,
                        num_restarts=4,
                        raw_samples=128,
                        mc_samples=int(qparams["mc_samples"]),
                        fit_noise_floor=1e-6,
                        sequential=False,
                        maxiter=100,
                        progress=lambda entry, key=run_key: _log(
                            f"  {key} it {entry['iteration']} n={entry['evaluations']} "
                            f"fit {entry['fit_seconds']:.1f}s acq {entry['acquisition_seconds']:.1f}s "
                            f"hv {entry['hypervolume']:.5f}"
                        ),
                    )
                elif strategy == "nsga3":
                    info = opt.run_nsga3(
                        ledger,
                        initial_count=plan.initial_design,
                        population_size=plan.nsga3_population_size,
                        generations=plan.nsga3_generations,
                        reference_direction_seed=1,
                    )
                elif strategy == "lhs":
                    opt.run_lhs(ledger, initial_count=plan.initial_design)
                    info = {"points": plan.evaluations_per_run, "stages": [plan.initial_design, plan.evaluations_per_run - plan.initial_design]}
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
                    f"assessment: run {run_key} done hv={summary['final_hypervolume']:.5f} "
                    f"pareto={summary['pareto_set_size']} infeasible={summary['infeasible_evaluations']} "
                    f"wall={summary['wall_clock_seconds']:.1f}s"
                )

        _log("assessment: metrics and gates")
        reference = state["reference"]
        reference_hv = reference["fronts"]["robust"]["hypervolume"]
        pooled = pooled_fronts(all_records)
        per_strategy = {
            strategy: pooled_fronts([record for record in all_records if record["run"].startswith(strategy + ":")])
            for strategy in plan.strategies
        }
        variance = {
            strategy: _seed_variance(
                [run_summaries[f"{strategy}:{seed}"]["final_hypervolume"] for seed in plan.seeds]
            )
            for strategy in plan.strategies
        }
        timing = {
            key: {
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "evaluation_seconds": math.fsum(
                    record["evaluation_seconds"] for record in all_records if record["run"] == key
                ),
                "bo_fit_seconds": math.fsum(
                    entry["fit_seconds"] for entry in run_infos[key].get("iteration_log", [])
                ),
                "bo_acquisition_seconds": math.fsum(
                    entry["acquisition_seconds"] for entry in run_infos[key].get("iteration_log", [])
                ),
            }
            for key, summary in run_summaries.items()
        }
        sensitivity = cusp_sensitivity(all_records, value, pooled["robust"]["design_ids"])

        # ---- binding gates -------------------------------------------------
        replay = replay_records(all_records)
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
        monotone_ok = all(
            all(
                curve[index]["hypervolume"] >= curve[index - 1]["hypervolume"]
                for index in range(1, len(curve))
            )
            for curve in curves.values()
        )
        budget_ok = all(
            summary["evaluations"] == plan.evaluations_per_run for summary in run_summaries.values()
        )
        shared_ok = all(
            len(
                {
                    canonical_bytes(
                        [
                            record["design"]["values"]
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
        sample_ok = m.sample_sha256(sample) == value["uncertain_inputs"]["sample"]["sha256"]
        pareto_ok = True
        pareto_sets = {}
        for key, summary in run_summaries.items():
            run_records = [record for record in all_records if record["run"] == key]
            by_index = {record["index"]: record for record in run_records}
            members = [by_index[index] for index in summary["pareto_record_indices"]]
            member_replay = replay_records(members)
            successful = [
                (record["index"], m.normalized_objectives(_robust_vector(record)))
                for record in run_records
                if record["status"] == "success"
            ]
            front = {successful[index][0] for index in m.nondominated_indices([row[1] for row in successful])}
            nondominated_ok = set(summary["pareto_record_indices"]) == front
            pareto_ok = pareto_ok and member_replay["passed"] and nondominated_ok
            pareto_sets[key] = {
                "size": len(members),
                "replay_passed": member_replay["passed"],
                "nondominated_recomputed": nondominated_ok,
                "designs": [
                    {
                        "index": record["index"],
                        "design_id": record["design"]["design_id"],
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
        binding = {
            "replay_bit_exact": {"passed": replay["passed"], "replayed": replay["replayed"], "mismatches": replay["mismatches"]},
            "l0_domain": {"passed": l0_domain_ok},
            "hypervolume_monotone": {"passed": monotone_ok},
            "budget_exact": {"passed": budget_ok},
            "shared_initial_design": {"passed": shared_ok},
            "sample_hash": {"passed": sample_ok},
            "pareto_replay": {"passed": pareto_ok},
            "code_contract": {"passed": contract_ok},
        }
        reported = {
            "bo_beats_random": _paired_test(run_summaries, plan.seeds, "qlognehvi", "lhs"),
            "bo_beats_nsga3": _paired_test(run_summaries, plan.seeds, "qlognehvi", "nsga3"),
            "design_set_invariance": {
                "passed": all(
                    item["identical_on_common_feasible_set_up_to_ties"]
                    for item in sensitivity["priors"]
                ),
                "definition": (
                    "for every alternative cusp prior the robust nondominated design set of the "
                    "designs feasible under BOTH priors equals the campaign's nondominated set "
                    "restricted to the same designs, both computed with roundoff-aware "
                    "dominance (relative tolerance 1e-9); the exact-arithmetic sets are reported too"
                ),
                "per_prior": [
                    {
                        k: item[k]
                        for k in (
                            "cusp_upper",
                            "identical_on_common_feasible_set",
                            "identical_on_common_feasible_set_up_to_ties",
                            "common_front_symmetric_difference",
                            "tolerant_common_front_symmetric_difference",
                            "common_feasible_designs",
                            "common_front_size",
                            "identical_to_campaign_front",
                            "jaccard_with_campaign_front",
                            "front_size",
                            "feasible",
                        )
                    }
                    for item in sensitivity["priors"]
                ],
            },
            "robust_vs_nominal": {
                "robust_front_size": pooled["robust"]["front_size"],
                "nominal_front_size": pooled["nominal"]["front_size"],
                "shared_designs": len(pooled["shared_design_ids"]),
                "jaccard": pooled["jaccard_robust_nominal"],
                "nominal_front_members_robust_feasible": pooled["nominal"]["robust_feasible_members"],
                "robust_hypervolume": pooled["robust"]["hypervolume"],
                "nominal_hypervolume": pooled["nominal"]["hypervolume"],
            },
        }
        all_binding = all(item["passed"] for item in binding.values())
        gates = {"binding": binding, "reported_not_binding": reported, "all_binding_passed": all_binding, "binding_in_this_plan": plan.binding_gates}
        metrics = {
            "runs": run_summaries,
            "hypervolume_table": {
                key: {
                    "final_hypervolume": summary["final_hypervolume"],
                    "attained_fraction_of_dense_reference": (
                        summary["final_hypervolume"] / reference_hv if reference_hv > 0 else None
                    ),
                    "pareto_set_size": summary["pareto_set_size"],
                    "infeasible_evaluations": summary["infeasible_evaluations"],
                    "wall_clock_seconds": summary["wall_clock_seconds"],
                }
                for key, summary in run_summaries.items()
            },
            "dense_reference": {
                "count": reference["count"],
                "robust_hypervolume": reference_hv,
                "nominal_hypervolume": reference["fronts"]["nominal"]["hypervolume"],
                "robust_front_size": reference["fronts"]["robust"]["front_size"],
                "nominal_front_size": reference["fronts"]["nominal"]["front_size"],
            },
            "seed_variance": variance,
            "timing": timing,
            "per_strategy_pooled": {
                strategy: {
                    "robust_front_size": item["robust"]["front_size"],
                    "robust_hypervolume": item["robust"]["hypervolume"],
                    "nominal_front_size": item["nominal"]["front_size"],
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
        context.write_json("artifacts/gates.json", gates)
        campaign_result = {
            "schema_version": schema("campaign-result"),
            "experiment_id": value["experiment_id"],
            "plan_kind": plan.kind,
            "evidentiary": plan.kind == "evidentiary",
            "classification": value["classification"],
            "claim_boundary": value["claim_boundary"]["statement"],
            "closure": value["closures"]["CL-1"]["id"],
            "runs": len(run_summaries),
            "total_evaluations": len(all_records),
            "infeasible_evaluations": sum(1 for record in all_records if record["status"] != "success"),
            "hypervolume_table": metrics["hypervolume_table"],
            "seed_variance": variance,
            "bo_beats_random": reported["bo_beats_random"]["passed"],
            "bo_beats_nsga3": reported["bo_beats_nsga3"]["passed"],
            "design_set_invariance": reported["design_set_invariance"]["passed"],
            "robust_vs_nominal": reported["robust_vs_nominal"],
            "all_binding_gates_passed": all_binding,
            "assessment_seconds": time.perf_counter() - started,
        }
        context.write_json("artifacts/campaign-result.json", campaign_result)
        collector["assessment"] = {
            "runs": run_summaries,
            "gates": gates,
            "metrics": metrics,
            "campaign_result": campaign_result,
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
