"""Campaign mechanics of cusp topology search v3 (literature cusp/cell definition).

Follows the accepted one-shot template (``cft_orbit_wall_loss_v4`` /
``orbit_wall_loss_geometry_screening_v1``): one :class:`CampaignPlan` drives the evidentiary
campaign and the disclosed NON-EVIDENTIARY shakedown; the shakedown must pass on real designs
of every set before ``prepare`` freezes the authorities; one detached execution publishes
through the shared :class:`ExperimentRuntime`.

Per design (worker task): rebuild + identity proof + CPU solves at the accepted and refined
resolutions (:mod:`.fields`), definition-v3 characterization of both maps (:mod:`.topology`),
refinement stability, held-out comparison against the sealed v1 / sweep axis nulls, P2
consistency references. The assessment evaluates the binding integrity gates, the reported
estimands, and emits the dataset, CSV and the consumer catalogue.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import multiprocessing
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import canonical_bytes, semantic_sha256, strict_json_file

from experiments.orbit_wall_loss_geometry_screening_v1.designs import field_pipeline_source_files, field_pipeline_source_sha256

from . import catalogue as catalogue_module
from . import fields as field_module
from . import topology as topology_module
from .fields import DESIGN_SETS, SET_CHARACTERIZATION, SET_P2, SET_SWEEP, DesignSpec, ResolvedDesign
from .topology import TopologyPolicy

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
DESIGN_AUTHORITIES_PATH = EXPERIMENT / "design-authorities.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"

VERSION_TAG = "cft-revival.cusp-topology-search-v3"
CLASSIFICATION = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
P2_CLASSIFICATION = "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
EXPERIMENT_CODE_FILES = ("__init__.py", "catalogue.py", "experiment.py", "fields.py", "run.py", "topology.py")
CSV_COLUMNS = (
    "set_id",
    "design_id",
    "label",
    "stable",
    "axis_null_count",
    "channel_axis_null_count",
    "wall_cusp_count",
    "cell_count",
    "four_wall_cusps",
    "four_cells",
    "wall_cusp_z_m",
    "wall_cusp_z_over_length",
    "cell_lengths_m",
    "wall_mirror_ratio_min",
    "wall_mirror_ratio_max",
    "axis_mirror_ratio_min",
    "axis_mirror_ratio_max",
    "angle_to_wall_normal_deg_min",
    "angle_to_wall_normal_deg_max",
    "max_wall_intersection_shift_m",
    "max_axis_null_shift_m",
    "wall_radius_m",
    "chamber_length_m",
    "stage_pitch_m",
    "stage_count",
)


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/1.0.0"


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    if value["classification"] != CLASSIFICATION or value["p2_row_classification"] != P2_CLASSIFICATION:
        raise ValueError("protocol classification labels differ from the experiment constants")
    return value


def label_for(set_id: str, value: Mapping[str, Any]) -> str:
    return value["p2_row_classification"] if set_id == SET_P2 else value["classification"]


def policy_from(value: Mapping[str, Any]) -> TopologyPolicy:
    return TopologyPolicy.from_protocol(value["definition_v3"]["numerical_parameters"])


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


# --------------------------------------------------------------------------
# Source binding
# --------------------------------------------------------------------------


def experiment_code_sha256() -> str:
    digest = hashlib.sha256()
    for name in EXPERIMENT_CODE_FILES:
        data = (EXPERIMENT / name).read_bytes()
        if b"\r" in data:
            raise RuntimeError(f"experiment source {name} contains CR bytes")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def source_binding_report(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_code_sha256": experiment_code_sha256(),
        "experiment_code_files": list(EXPERIMENT_CODE_FILES),
        "dependency_source_sha256": field_module.dependency_source_sha256(),
        "dependency_source_files": [path.relative_to(MODERN).as_posix() for path in field_module.dependency_source_files()],
        "field_pipeline_source_sha256": field_pipeline_source_sha256(),
        "field_pipeline_source_files": [path.relative_to(MODERN).as_posix() for path in field_pipeline_source_files()],
        "sealed_sources": field_module.sealed_source_binding(),
        "protocol_semantic_sha256": semantic_sha256(value),
    }


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPlan:
    kind: str
    design_keys: tuple[str, ...]
    binding_gates: bool

    def __post_init__(self) -> None:
        if self.kind not in ("evidentiary", "shakedown"):
            raise ValueError("unknown campaign plan kind")
        if not self.design_keys or len(set(self.design_keys)) != len(self.design_keys):
            raise ValueError("plan designs must be unique and non-empty")
        if (self.kind == "evidentiary") != self.binding_gates:
            raise ValueError("binding gates are exactly the evidentiary plan's")


def all_specs(value: Mapping[str, Any]) -> tuple[DesignSpec, ...]:
    return field_module.design_specs(value)


def evidentiary_plan(value: Mapping[str, Any]) -> CampaignPlan:
    return CampaignPlan("evidentiary", tuple(spec.key for spec in all_specs(value)), True)


def shakedown_plan(value: Mapping[str, Any]) -> CampaignPlan:
    declaration = value["shakedown"]
    if declaration["evidentiary"] is not False or declaration["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown must be declared non-evidentiary")
    known = {spec.key for spec in all_specs(value)}
    keys: list[str] = []
    for set_id in DESIGN_SETS:
        for design_id in declaration["designs"].get(set_id, ()):
            key = f"{set_id}:{design_id}"
            if key not in known:
                raise ValueError(f"shakedown design {key} is not a declared design")
            keys.append(key)
    return CampaignPlan("shakedown", tuple(keys), False)


def plan_record(plan: CampaignPlan) -> dict[str, Any]:
    record = asdict(plan)
    record["design_keys"] = list(plan.design_keys)
    return record


def specs_for_plan(value: Mapping[str, Any], plan: CampaignPlan) -> tuple[DesignSpec, ...]:
    by_key = {spec.key: spec for spec in all_specs(value)}
    return tuple(by_key[key] for key in plan.design_keys)


def replay_keys(value: Mapping[str, Any], plan: CampaignPlan) -> tuple[str, ...]:
    declared = [f"{set_id}:{design_id}" for set_id, ids in value["execution"]["replay_designs"].items() for design_id in ids]
    if plan.kind == "shakedown":
        # the shakedown replays the first design of its own plan
        return (plan.design_keys[0],)
    return tuple(key for key in declared if key in plan.design_keys)


def worker_count(value: Mapping[str, Any]) -> int:
    execution = value["execution"]
    if not execution["parallel_designs"]:
        return 1
    return max(1, min(int(execution["max_design_workers"]), os.cpu_count() or 1))


def run_stage(tasks: Sequence[Mapping[str, Any]], function: Callable[[Mapping[str, Any]], dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Run design tasks in submission order; results are returned in the same order."""

    if workers <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        return [future.result() for future in futures]


# --------------------------------------------------------------------------
# Design authorities (no solving)
# --------------------------------------------------------------------------


def build_design_authorities(value: Mapping[str, Any], plan: CampaignPlan) -> dict[str, Any]:
    specs = specs_for_plan(value, plan)
    designs = [
        {**field_module.design_identity_without_solving(spec, value), "key": spec.key, "ordinal": spec.ordinal}
        for spec in specs
    ]
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.set_id] = counts.get(spec.set_id, 0) + 1
    return {
        "schema_version": schema("design-authorities"),
        "plan_kind": plan.kind,
        "design_count": len(designs),
        "set_counts": counts,
        "designs": designs,
    }


# --------------------------------------------------------------------------
# Held-out comparisons and P2 consistency
# --------------------------------------------------------------------------


def _match_sorted(reference: Sequence[float], observed: Sequence[float], tolerance: float) -> dict[str, Any]:
    reference = sorted(reference)
    observed = sorted(observed)
    pairs: list[dict[str, Any]] = []
    unmatched_reference: list[float] = []
    remaining = list(observed)
    for value in reference:
        if remaining:
            nearest = min(remaining, key=lambda item: abs(item - value))
            if abs(nearest - value) <= tolerance:
                pairs.append({"reference_z_m": value, "observed_z_m": nearest, "difference_m": abs(nearest - value)})
                remaining.remove(nearest)
                continue
        unmatched_reference.append(value)
    return {
        "tolerance_m": tolerance,
        "reference_count": len(reference),
        "observed_count": len(observed),
        "matched": pairs,
        "unmatched_reference_z_m": unmatched_reference,
        "unmatched_observed_z_m": remaining,
        "max_difference_m": max((pair["difference_m"] for pair in pairs), default=None),
        "bijection": not unmatched_reference and not remaining,
    }


def held_out_comparison(resolved: ResolvedDesign, accepted: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(value["definition_v3"]["held_out_tolerance_m"])
    nulls = accepted["axis_nulls"]["nulls"]
    if resolved.spec.set_id == SET_CHARACTERIZATION:
        reference = [
            root for root in resolved.reference["v1_primary_axis_roots"]
            if root["zone"] in ("plasma_channel", "channel_axial_margin")
        ]
        observed = [null for null in nulls if null["zone"] == "channel"]
        match = _match_sorted([root["z_m"] for root in reference], [null["z_m"] for null in observed], tolerance)
        classifications_x = all(root["classification"] == "X" for root in reference) and all(null["classification"] == "X" for null in observed)
        return {
            "kind": "characterization_v1_axis_roots",
            "statement": "sealed v1 primary-map axis roots inside the channel vs v3 channel axis nulls",
            "reference_classifications": [root["classification"] for root in reference],
            **match,
            "classifications_agree": classifications_x,
            "passed": bool(match["bijection"] and classifications_x),
            "applies": True,
        }
    if resolved.spec.set_id == SET_SWEEP:
        low, high = accepted["axis_nulls"]["window_m"]
        reference = [z for z in resolved.reference["sweep_axis_null_positions_m"] if low <= z <= high]
        match = _match_sorted(reference, [null["z_m"] for null in nulls], tolerance)
        return {
            "kind": "sweep_v2_axis_nulls",
            "statement": "sealed sweep-v2 axis_null_positions_m inside the v3 window vs v3 axis nulls",
            "reference_outside_window_m": [z for z in resolved.reference["sweep_axis_null_positions_m"] if not low <= z <= high],
            **match,
            "passed": bool(match["bijection"]),
            "applies": True,
        }
    return {"kind": "none", "statement": "no sealed axis-null reference for this set", "passed": True, "applies": False}


def p2_consistency(resolved: ResolvedDesign, accepted: Mapping[str, Any]) -> dict[str, Any] | None:
    if resolved.spec.set_id != SET_P2:
        return None
    references = resolved.reference["p2_consistency_references"]
    cusps = accepted["topology"]["wall_cusps"]
    rows = []
    for cusp in cusps:
        z_c = cusp["z_c_m"]
        wall_ref = min(references["topology_dashboard_wall_abs_br_maxima_m"], key=lambda z: abs(z - z_c))
        pic_ref = min(references["pic_axis_null_planes_m"], key=lambda z: abs(z - z_c))
        rows.append(
            {
                "cusp_id": cusp["cusp_id"],
                "z_c_m": z_c,
                "axis_null_z_m": cusp["axis_null_z_m"],
                "nearest_dashboard_wall_abs_br_maximum_m": wall_ref,
                "difference_to_dashboard_maximum_m": z_c - wall_ref,
                "nearest_pic_axis_null_plane_m": pic_ref,
                "difference_axis_null_to_pic_plane_m": cusp["axis_null_z_m"] - pic_ref,
            }
        )
    return {
        "role": references["role"],
        "references": references,
        "cusps": rows,
        "cusp_count_equals_reference_count": len(cusps) == len(references["pic_axis_null_planes_m"]) == len(references["topology_dashboard_wall_abs_br_maxima_m"]),
        "max_abs_difference_to_dashboard_maximum_m": max((abs(row["difference_to_dashboard_maximum_m"]) for row in rows), default=None),
        "max_abs_difference_axis_null_to_pic_plane_m": max((abs(row["difference_axis_null_to_pic_plane_m"]) for row in rows), default=None),
    }


# --------------------------------------------------------------------------
# Per-design worker
# --------------------------------------------------------------------------


def _nulls_converged(report: Mapping[str, Any]) -> bool:
    nulls = report["axis_nulls"]["nulls"]
    return bool(
        report["axis_nulls"]["all_converged"]
        and report["axis_nulls"]["all_x_type"]
        and report["axis_nulls"]["all_classifications_agree"]
        and all(null["v1_local_topology"].get("jacobian_converged") is True for null in nulls)
    )


def design_gate_checks(record: Mapping[str, Any]) -> dict[str, bool]:
    accepted = record["accepted"]
    refined = record["refined"]
    return {
        "identity_proven": bool(record["evidence"]["identity_proven"]),
        "every_null_converged": _nulls_converged(accepted) and _nulls_converged(refined),
        "every_trace_terminates_cleanly": bool(accepted["all_traces_terminate_cleanly"] and refined["all_traces_terminate_cleanly"]),
        "every_wall_trace_flux_consistent": bool(accepted["all_wall_traces_flux_consistent"] and refined["all_wall_traces_flux_consistent"]),
        "refinement_stability": bool(record["stability"]["stable"]),
        "held_out_correspondence": bool(record["held_out"]["passed"]),
    }


def characterize_resolved(resolved: ResolvedDesign, value: Mapping[str, Any], *, keep_paths: bool) -> dict[str, Any]:
    """Definition-v3 characterization of both maps plus stability and held-out checks (pure)."""

    policy = policy_from(value)
    tightness = float(value["definition_v3"]["minimum_certificate_dense_to_bound_ratio"])
    window = topology_module.axis_window(resolved.accepted, resolved.geometry, policy)
    sweep_peaks = resolved.reference.get("sweep_axis_bz_peak_positions_m")
    accepted = topology_module.characterize_map(
        resolved.accepted,
        resolved.geometry,
        policy,
        source_identity_sha256=resolved.accepted_identity_sha256,
        minimum_certificate_tightness_ratio=tightness,
        keep_paths=keep_paths,
        sweep_axis_bz_peaks_m=sweep_peaks,
        axis_window_m=window,
    )
    refined = topology_module.characterize_map(
        resolved.refined,
        resolved.geometry,
        policy,
        source_identity_sha256=resolved.refined_identity_sha256,
        minimum_certificate_tightness_ratio=tightness,
        keep_paths=False,
        sweep_axis_bz_peaks_m=sweep_peaks,
        axis_window_m=window,
    )
    stability = topology_module.compare_resolutions(accepted, refined, float(value["definition_v3"]["stability_tolerance_m"]))
    held_out = held_out_comparison(resolved, accepted, value)
    consistency = p2_consistency(resolved, accepted)
    payload = _plain(
        {
            "axis_window_m": [window[0], window[1]],
            "accepted": accepted,
            "refined": refined,
            "stability": stability,
            "held_out": held_out,
            "p2_consistency": consistency,
        }
    )
    payload["topology_payload_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def run_design_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Worker: resolve one design and characterize it under definition v3."""

    spec = DesignSpec(**task["spec"])
    value = task["protocol"]
    started = time.perf_counter()
    try:
        resolved = field_module.resolve_design(spec, value)
    except Exception as error:  # recorded, never hidden
        return _plain(
            {
                "key": spec.key,
                "set_id": spec.set_id,
                "design_id": spec.design_id,
                "ordinal": spec.ordinal,
                "representative": spec.representative,
                "label": label_for(spec.set_id, value),
                "status": "failed",
                "stage": "resolve",
                "reason": f"{type(error).__name__}: {error}",
                "timing_s": {"total": time.perf_counter() - started},
            }
        )
    solve_seconds = resolved.solve_seconds
    characterize_started = time.perf_counter()
    try:
        topology = characterize_resolved(resolved, value, keep_paths=bool(task["keep_paths"]))
    except Exception as error:  # recorded, never hidden
        return _plain(
            {
                "key": spec.key,
                "set_id": spec.set_id,
                "design_id": spec.design_id,
                "ordinal": spec.ordinal,
                "representative": spec.representative,
                "label": label_for(spec.set_id, value),
                "status": "failed",
                "stage": "characterize",
                "reason": f"{type(error).__name__}: {error}",
                "identity": resolved.identity,
                "timing_s": {"solve": solve_seconds, "total": time.perf_counter() - started},
            }
        )
    characterize_seconds = time.perf_counter() - characterize_started
    record = {
        "schema_version": schema("design-record"),
        "key": spec.key,
        "set_id": spec.set_id,
        "design_id": spec.design_id,
        "ordinal": spec.ordinal,
        "representative": spec.representative,
        "label": label_for(spec.set_id, value),
        "status": "resolved",
        "identity": resolved.identity,
        "evidence": resolved.evidence,
        "reference": resolved.reference,
        "geometry": resolved.geometry.to_dict(),
        **topology,
        "timing_s": {"solve": solve_seconds, "characterize": characterize_seconds, "total": time.perf_counter() - started},
        "accepted_grid": resolved.accepted.to_dict(),
    }
    record["gate_checks"] = design_gate_checks(record)
    record["record_path"] = f"artifacts/designs/{spec.set_id}/{spec.design_id}.json"
    return _plain(record)


# --------------------------------------------------------------------------
# Estimands and dataset
# --------------------------------------------------------------------------


def _range(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(clean), "min": min(clean), "median": statistics.median(clean), "max": max(clean)}


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in values:
        counts[str(int(item))] = counts.get(str(int(item)), 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: int(pair[0])))


def set_estimands(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cusps = [cusp for record in records for cusp in record["accepted"]["topology"]["wall_cusps"]]
    interior_cells = [cell for record in records for cell in record["accepted"]["topology"]["cells"] if cell["kind"] == "interior"]
    all_cells = [cell for record in records for cell in record["accepted"]["topology"]["cells"]]
    counts = [record["accepted"]["topology"]["wall_cusp_count"] for record in records]
    return {
        "design_count": len(records),
        "stable_design_count": sum(record["stability"]["stable"] for record in records),
        "wall_cusp_count_histogram": _histogram(counts),
        "cell_count_histogram": _histogram([record["accepted"]["topology"]["cell_count"] for record in records]),
        "axis_null_count_histogram": _histogram([record["accepted"]["axis_nulls"]["count"] for record in records]),
        "channel_axis_null_count_histogram": _histogram([record["accepted"]["axis_nulls"]["channel_count"] for record in records]),
        "four_wall_cusp_count": sum(record["accepted"]["topology"]["four_wall_cusps"] for record in records),
        "four_wall_cusp_fraction": (sum(record["accepted"]["topology"]["four_wall_cusps"] for record in records) / len(records)) if records else None,
        "four_cell_count": sum(record["accepted"]["topology"]["four_cells"] for record in records),
        "four_cell_fraction": (sum(record["accepted"]["topology"]["four_cells"] for record in records) / len(records)) if records else None,
        "designs_with_at_least_one_cusp": sum(count > 0 for count in counts),
        "z_c_m": _range([cusp["z_c_m"] for cusp in cusps]),
        "z_c_over_length": _range([cusp["z_c_over_length"] for cusp in cusps]),
        "axis_to_wall_shift_m": _range([cusp["axis_to_wall_shift_m"] for cusp in cusps]),
        "distance_to_nearest_stage_gap_m": _range([cusp["distance_to_nearest_stage_gap_m"] for cusp in cusps]),
        "distance_to_nearest_stage_centre_m": _range([cusp["distance_to_nearest_stage_centre_m"] for cusp in cusps]),
        "angle_to_wall_normal_deg": _range([cusp["angle_to_wall_normal_deg"] for cusp in cusps]),
        "wall_b_at_cusp_t": _range([cusp["wall_b_t"] for cusp in cusps]),
        "interior_cell_length_m": _range([cell["length_m"] for cell in interior_cells]),
        "interior_cell_length_over_pitch": _range([cell["length_over_pitch"] for cell in interior_cells]),
        "interior_wall_mirror_ratio": _range([cell["wall_mirror_ratio"] for cell in interior_cells]),
        "interior_axis_mirror_ratio": _range([cell["axis_mirror_ratio"] for cell in interior_cells]),
        "all_cells_wall_mirror_ratio": _range([cell["wall_mirror_ratio"] for cell in all_cells]),
        "all_cells_axis_mirror_ratio": _range([cell["axis_mirror_ratio"] for cell in all_cells]),
        "boundary_ambiguous_cusp_count": sum(cusp["boundary_ambiguous"] for cusp in cusps),
        "max_wall_intersection_shift_m": _range([record["stability"]["max_wall_intersection_shift_m"] for record in records]),
        "max_axis_null_shift_m": _range([record["stability"]["max_axis_null_shift_m"] for record in records]),
        "v4_bilinear_difference_m": _range(
            [trace["v4_bilinear_difference_m"] for record in records for trace in record["accepted"]["separatrix_traces"] if trace.get("v4_bilinear_difference_m") is not None]
        ),
        "outside_intersection_zones": _histogram_str([row["zone"] for record in records for row in record["accepted"]["topology"]["outside_intersections"]]),
    }


def _histogram_str(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in values:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def dataset_row(record: Mapping[str, Any]) -> dict[str, Any]:
    accepted = record["accepted"]
    topology = accepted["topology"]
    return {
        "key": record["key"],
        "set_id": record["set_id"],
        "design_id": record["design_id"],
        "ordinal": record["ordinal"],
        "representative": record["representative"],
        "label": record["label"],
        "record_path": record["record_path"],
        "geometry": record["geometry"],
        "identity": {key: record["identity"][key] for key in ("accepted_field_identity_sha256", "refined_field_identity_sha256")},
        "axis_nulls": [{"null_id": n["null_id"], "z_m": n["z_m"], "zone": n["zone"], "classification": n["classification"]} for n in accepted["axis_nulls"]["nulls"]],
        "axis_null_count": accepted["axis_nulls"]["count"],
        "channel_axis_null_count": accepted["axis_nulls"]["channel_count"],
        "wall_cusps": [
            {key: cusp[key] for key in ("cusp_id", "axis_null_z_m", "z_c_m", "z_c_over_length", "wall_b_t", "wall_b_r_t", "angle_to_wall_normal_deg", "boundary_ambiguous", "distance_to_nearest_stage_gap_m", "distance_to_nearest_stage_centre_m")}
            for cusp in topology["wall_cusps"]
        ],
        "outside_intersections": [{key: row[key] for key in ("cusp_id", "z_c_m", "zone", "wall_b_t")} for row in topology["outside_intersections"]],
        "cells": [
            {key: cell[key] for key in ("cell_id", "kind", "z_start_m", "z_end_m", "length_m", "length_over_pitch", "wall_b_min_t", "wall_mirror_ratio", "axis_bz_peak_t", "axis_mirror_ratio", "sweep_axis_bz_peaks_inside", "stage_centres_inside")}
            for cell in topology["cells"]
        ],
        "wall_cusp_count": topology["wall_cusp_count"],
        "cell_count": topology["cell_count"],
        "four_wall_cusps": topology["four_wall_cusps"],
        "four_cells": topology["four_cells"],
        "stability": {key: record["stability"][key] for key in ("stable", "axis_null_count_equal", "wall_reaching_count_equal", "wall_cusp_count_equal", "max_axis_null_shift_m", "max_wall_intersection_shift_m")},
        "held_out": {key: record["held_out"].get(key) for key in ("kind", "applies", "passed", "reference_count", "observed_count", "max_difference_m")},
        "p2_consistency": record.get("p2_consistency"),
        "gate_checks": record["gate_checks"],
        "grid": {key: accepted["grid"][key] for key in ("radial_samples", "axial_samples", "dr_m", "dz_m", "radial_cells_across_bore")},
        "refined_grid": {key: record["refined"]["grid"][key] for key in ("radial_samples", "axial_samples", "dr_m", "dz_m")},
        "timing_s": record["timing_s"],
    }


def dataset_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        cusps = row["wall_cusps"]
        cells = row["cells"]
        wall_ratios = [cell["wall_mirror_ratio"] for cell in cells if cell["wall_mirror_ratio"] is not None]
        axis_ratios = [cell["axis_mirror_ratio"] for cell in cells if cell["axis_mirror_ratio"] is not None]
        angles = [cusp["angle_to_wall_normal_deg"] for cusp in cusps]
        writer.writerow(
            [
                row["set_id"],
                row["design_id"],
                row["label"],
                row["stability"]["stable"],
                row["axis_null_count"],
                row["channel_axis_null_count"],
                row["wall_cusp_count"],
                row["cell_count"],
                row["four_wall_cusps"],
                row["four_cells"],
                ";".join(repr(cusp["z_c_m"]) for cusp in cusps),
                ";".join(repr(cusp["z_c_over_length"]) for cusp in cusps),
                ";".join(repr(cell["length_m"]) for cell in cells),
                repr(min(wall_ratios)) if wall_ratios else "",
                repr(max(wall_ratios)) if wall_ratios else "",
                repr(min(axis_ratios)) if axis_ratios else "",
                repr(max(axis_ratios)) if axis_ratios else "",
                repr(min(angles)) if angles else "",
                repr(max(angles)) if angles else "",
                repr(row["stability"]["max_wall_intersection_shift_m"]) if row["stability"]["max_wall_intersection_shift_m"] is not None else "",
                repr(row["stability"]["max_axis_null_shift_m"]) if row["stability"]["max_axis_null_shift_m"] is not None else "",
                repr(row["geometry"]["wall_radius_m"]),
                repr(row["geometry"]["chamber_length_m"]),
                repr(row["geometry"]["stage_pitch_m"]),
                len(row["geometry"]["stage_centres_m"]),
            ]
        )
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Shakedown record verification
# --------------------------------------------------------------------------


def verify_shakedown_record(value: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False and record.get("outcomes_enter_estimand") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = record.get("protocol_semantic_sha256") == semantic_sha256(value)
    checks["experiment_code_sha256_current"] = record.get("experiment_code_sha256") == experiment_code_sha256()
    checks["dependency_source_sha256_current"] = record.get("dependency_source_sha256") == field_module.dependency_source_sha256()
    checks["field_pipeline_source_sha256_current"] = record.get("field_pipeline_source_sha256") == field_pipeline_source_sha256()
    try:
        checks["plan_matches_protocol"] = record.get("shakedown_plan") == plan_record(shakedown_plan(value))
    except Exception:
        checks["plan_matches_protocol"] = False
    checks["every_design_resolved"] = bool(record.get("design_count")) and record.get("design_count") == record.get("resolved_design_count")
    checks["timing_projection_present"] = isinstance(record.get("timing_projection"), dict) and "projected_wall_seconds_at_pool" in record.get("timing_projection", {})
    if not all(checks.values()):
        failed = sorted(name for name, ok in checks.items() if not ok)
        raise ValueError(f"shakedown record does not prove the current protocol/code: {failed}")
    return checks


@dataclass(frozen=True)
class FrozenAuthority:
    authorities: Mapping[str, Any]
    design_authorities: Mapping[str, Any]
    shakedown: Mapping[str, Any]
    shakedown_bytes: bytes


def load_frozen_authority() -> FrozenAuthority:
    return FrozenAuthority(
        strict_json_file(AUTHORITIES_PATH),
        strict_json_file(DESIGN_AUTHORITIES_PATH),
        strict_json_file(SHAKEDOWN_PATH),
        SHAKEDOWN_PATH.read_bytes(),
    )


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
        binding = source_binding_report(value)
        design_authorities = build_design_authorities(value, plan)
        if frozen is not None:
            if binding["protocol_semantic_sha256"] != frozen.authorities["protocol_semantic_sha256"]:
                raise ValueError("protocol semantic authority differs")
            if frozen.design_authorities != design_authorities or semantic_sha256(frozen.design_authorities) != frozen.authorities["design_authorities_sha256"]:
                raise ValueError("design authorities differ from preregistration")
            for key in ("experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
                if frozen.authorities[key] != binding[key]:
                    raise ValueError(f"{key} differs from the preregistered authority")
            if frozen.authorities["sealed_sources"] != binding["sealed_sources"]:
                raise ValueError("sealed source identities differ from the preregistered authority")
            if hashlib.sha256(frozen.shakedown_bytes).hexdigest() != frozen.authorities["shakedown_file_sha256"] or semantic_sha256(frozen.shakedown) != frozen.authorities["shakedown_semantic_sha256"]:
                raise ValueError("shakedown record differs from preregistered authority")
            verify_shakedown_record(value, frozen.shakedown)
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/design-authorities.json", design_authorities)
        context.write_json("artifacts/campaign-plan.json", plan_record(plan))
        context.write_json("artifacts/source-binding.json", binding)
        if frozen is not None:
            context.write_json("artifacts/authorities.json", frozen.authorities)
            context.write_blob("artifacts/shakedown.json", frozen.shakedown_bytes)
        else:
            context.write_json(
                "artifacts/shakedown-disclosure.json",
                {"evidentiary": False, "outcomes_enter_estimand": False, "statement": value["shakedown"]["purpose"]},
            )
        context.write_json(
            "artifacts/runtime.json",
            {
                "generated_at_utc": datetime.now(timezone.utc),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "cpu_count": os.cpu_count(),
                "worker_pool_size": worker_count(value),
                "backend": "cft_revival.fields.solve_problem_cpu + PsiBicubicField tracing (CPU only)",
            },
        )
        state.update({"binding": binding, "design_authorities": design_authorities})
        collector["prebundle"] = {"design_count": design_authorities["design_count"], "set_counts": design_authorities["set_counts"]}
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "classification": CLASSIFICATION,
            "design_count": design_authorities["design_count"],
            "set_counts": design_authorities["set_counts"],
            "experiment_code_sha256": binding["experiment_code_sha256"],
            "dependency_source_sha256": binding["dependency_source_sha256"],
            "field_pipeline_source_sha256": binding["field_pipeline_source_sha256"],
        }

    def development(context: Any) -> Decision:
        started = time.perf_counter()
        workers = worker_count(value)
        specs = specs_for_plan(value, plan)
        context.before_expensive(
            "resolve-and-characterize-all-designs",
            kind="solver",
            details={"design_count": len(specs), "worker_pool_size": workers, "plan_kind": plan.kind, "solver": "cft_revival.fields.solve_problem_cpu"},
        )
        tasks = [{"spec": spec.to_dict(), "protocol": dict(value), "keep_paths": spec.representative} for spec in specs]
        stage_started = time.perf_counter()
        outcomes = run_stage(tasks, run_design_task, workers)
        stage_wall = time.perf_counter() - stage_started
        authorities = {item["key"]: item for item in state["design_authorities"]["designs"]}
        records: dict[str, dict[str, Any]] = {}
        grids: dict[str, Any] = {}
        failures: list[dict[str, Any]] = []
        for task, outcome in zip(tasks, outcomes, strict=True):
            if outcome["key"] != f"{task['spec']['set_id']}:{task['spec']['design_id']}":
                raise RuntimeError("design results returned out of order")
            if outcome["status"] != "resolved":
                failures.append({"key": outcome["key"], "stage": outcome.get("stage"), "reason": outcome.get("reason")})
                continue
            authority = authorities[outcome["key"]]
            for hash_key in ("geometry_sha256", "source_sha256", "case_sha256", "source_semantic_sha256", "material_sha256", "material_semantic_sha256"):
                if hash_key in authority and authority[hash_key] != outcome["identity"].get(hash_key):
                    raise ValueError(f"{outcome['key']}: resolved {hash_key} differs from the design authority")
            records[outcome["key"]] = outcome
            grid = outcome.pop("accepted_grid")
            grids[outcome["key"]] = grid["psi_wb"]
            field_bytes = canonical_bytes({"key": outcome["key"], "identity": outcome["identity"], **grid})
            outcome["accepted_grid_payload_sha256"] = hashlib.sha256(field_bytes).hexdigest()
            outcome["accepted_grid_path"] = f"artifacts/fields/{outcome['set_id']}/{outcome['design_id']}.json.gz"
            context.write_blob(outcome["accepted_grid_path"], gzip.compress(field_bytes, compresslevel=9, mtime=0))
            context.write_json(outcome["record_path"], outcome)
        context.write_json(
            "artifacts/design-failures.json",
            {"schema_version": schema("design-failures"), "rule": value["gates"]["binding_integrity"]["all_declared_designs_resolved"], "failed": failures},
        )
        accepted = bool(records) and not failures
        state.update({"records": records, "grids": grids, "failures": failures, "stage_wall_s": stage_wall})
        collector["development"] = {
            "resolved_design_count": len(records),
            "failures": failures,
            "stage_wall_s": stage_wall,
            "seconds": time.perf_counter() - started,
            "accepted": accepted,
            "per_design_seconds": {key: item["timing_s"] for key, item in records.items()},
        }
        return Decision(accepted, {"resolved_design_count": len(records), "failed_design_count": len(failures), "stage_wall_s": stage_wall})

    def assessment(context: Any) -> Decision:
        started = time.perf_counter()
        records = state["records"]
        ordered = [records[key] for key in plan.design_keys if key in records]
        # Determinism / replay in the main process.
        replays = []
        for key in replay_keys(value, plan):
            if key not in records:
                continue
            spec = next(item for item in specs_for_plan(value, plan) if item.key == key)
            context.before_expensive("replay-resolve-and-characterize", kind="solver", details={"key": key})
            resolved = field_module.resolve_design(spec, value)
            replay = characterize_resolved(resolved, value, keep_paths=spec.representative)
            replays.append(
                {
                    "key": key,
                    "worker_topology_payload_sha256": records[key]["topology_payload_sha256"],
                    "replay_topology_payload_sha256": replay["topology_payload_sha256"],
                    "field_identity_equal": resolved.accepted_identity_sha256 == records[key]["identity"]["accepted_field_identity_sha256"],
                    "accepted_grid_equal": _plain(resolved.accepted.to_dict()["psi_wb"]) == state["grids"][key],
                    "bit_identical": replay["topology_payload_sha256"] == records[key]["topology_payload_sha256"],
                }
            )
        replay_passed = bool(replays) and all(item["bit_identical"] and item["field_identity_equal"] and item["accepted_grid_equal"] for item in replays)
        per_design_gates = {key: record["gate_checks"] for key, record in records.items()}
        gate_names = ("identity_proven", "every_null_converged", "every_trace_terminates_cleanly", "every_wall_trace_flux_consistent", "refinement_stability", "held_out_correspondence")
        campaign_gates = {
            "all_declared_designs_resolved": not state["failures"] and len(records) == len(plan.design_keys),
            **{name: all(checks[name] for checks in per_design_gates.values()) for name in gate_names},
            "determinism_replay": replay_passed,
            # Enforced fail-closed in the prebundle (a mismatch raises -> prebundle_failure); recorded here.
            "hash_bindings": True,
        }
        hash_binding_note = (
            "verified against the frozen authorities in the prebundle" if frozen is not None else "shakedown: recorded, no frozen authority to compare"
        )
        failing_designs = {name: sorted(key for key, checks in per_design_gates.items() if not checks[name]) for name in gate_names}
        gates_passed = all(campaign_gates.values())
        by_set: dict[str, list[dict[str, Any]]] = {set_id: [] for set_id in DESIGN_SETS}
        for record in ordered:
            by_set[record["set_id"]].append(record)
        estimands = {set_id: set_estimands(items) for set_id, items in by_set.items() if items}
        estimands["pooled_l1a_sets"] = set_estimands([record for set_id, items in by_set.items() if set_id != SET_P2 for record in items])
        estimands["pooled_all"] = set_estimands(ordered)
        rows = [dataset_row(record) for record in ordered]
        held_out = {}
        for set_id, items in by_set.items():
            if not items:
                continue
            applies = any(record["held_out"]["applies"] for record in items)
            differences = [record["held_out"].get("max_difference_m") for record in items if record["held_out"].get("max_difference_m") is not None]
            held_out[set_id] = {
                "applies": applies,
                "passed_count": sum(record["held_out"]["passed"] for record in items),
                "design_count": len(items),
                "max_difference_m": max(differences) if (applies and differences) else None,
                "reference_null_count": sum(record["held_out"].get("reference_count") or 0 for record in items),
                "observed_null_count": sum(record["held_out"].get("observed_count") or 0 for record in items),
            }
        p2 = next((record["p2_consistency"] for record in ordered if record.get("p2_consistency")), None)
        protocol_hash = semantic_sha256(value)
        catalogue = catalogue_module.build_catalogue(value, ordered, protocol_semantic_sha256=protocol_hash)
        headline = {
            "design_count": len(ordered),
            "set_counts": {set_id: len(items) for set_id, items in by_set.items() if items},
            "stable_design_count": sum(record["stability"]["stable"] for record in ordered),
            "wall_cusp_count_histogram": estimands["pooled_all"]["wall_cusp_count_histogram"],
            "wall_cusp_count_histogram_by_set": {set_id: estimands[set_id]["wall_cusp_count_histogram"] for set_id in estimands if set_id in by_set},
            "four_wall_cusp_fraction_by_set": {set_id: estimands[set_id]["four_wall_cusp_fraction"] for set_id in estimands if set_id in by_set},
            "four_cell_fraction_by_set": {set_id: estimands[set_id]["four_cell_fraction"] for set_id in estimands if set_id in by_set},
            "z_c_over_length": estimands["pooled_all"]["z_c_over_length"],
            "interior_wall_mirror_ratio": estimands["pooled_all"]["interior_wall_mirror_ratio"],
            "interior_axis_mirror_ratio": estimands["pooled_all"]["interior_axis_mirror_ratio"],
            "angle_to_wall_normal_deg": estimands["pooled_all"]["angle_to_wall_normal_deg"],
            "max_wall_intersection_shift_m": estimands["pooled_all"]["max_wall_intersection_shift_m"]["max"],
            "held_out": held_out,
            "p2_consistency": None
            if p2 is None
            else {
                "cusp_count_equals_reference_count": p2["cusp_count_equals_reference_count"],
                "max_abs_difference_to_dashboard_maximum_m": p2["max_abs_difference_to_dashboard_maximum_m"],
                "max_abs_difference_axis_null_to_pic_plane_m": p2["max_abs_difference_axis_null_to_pic_plane_m"],
            },
        }
        dataset = {
            "schema_version": schema("topology-dataset"),
            "experiment_id": value["experiment_id"],
            "classification": CLASSIFICATION,
            "p2_row_classification": P2_CLASSIFICATION,
            "classification_statement": value["classification_statement"],
            "claim_boundary": value["claim_boundary"],
            "definition_v3": value["definition_v3"],
            "plan_kind": plan.kind,
            "evidentiary": plan.kind == "evidentiary",
            "protocol_semantic_sha256": protocol_hash,
            "experiment_code_sha256": state["binding"]["experiment_code_sha256"],
            "dependency_source_sha256": state["binding"]["dependency_source_sha256"],
            "field_pipeline_source_sha256": state["binding"]["field_pipeline_source_sha256"],
            "sealed_sources": state["binding"]["sealed_sources"],
            "design_count": len(rows),
            "designs": rows,
            "estimands": estimands,
            "held_out": held_out,
            "p2_consistency": p2,
            "headline": headline,
            "gates": {"campaign": campaign_gates, "failing_designs": failing_designs, "passed": gates_passed},
        }
        gates_record = {
            "schema_version": schema("gates"),
            "binding": plan.binding_gates,
            "campaign": campaign_gates,
            "hash_bindings_note": hash_binding_note,
            "failing_designs": failing_designs,
            "per_design": per_design_gates,
            "replays": replays,
            "passed": gates_passed,
            "design_count": len(records),
            "definitions": value["gates"],
        }
        context.write_json("artifacts/gates.json", gates_record)
        context.write_json("artifacts/topology-dataset.json", dataset)
        context.write_blob("artifacts/topology-dataset.csv", dataset_csv(rows))
        context.write_json(catalogue_module.CATALOGUE_RELATIVE_PATH, catalogue)
        status = "accepted_topology_screening" if (gates_passed and plan.kind == "evidentiary") else ("shakedown_passed" if gates_passed else "gates_failed")
        campaign_result = {
            "schema_version": schema("campaign-result"),
            "experiment_id": value["experiment_id"],
            "classification": CLASSIFICATION,
            "p2_row_classification": P2_CLASSIFICATION,
            "plan_kind": plan.kind,
            "evidentiary": plan.kind == "evidentiary",
            "status": status,
            "gates_passed": gates_passed,
            "campaign_gates": campaign_gates,
            "design_count": len(records),
            "set_counts": headline["set_counts"],
            "headline": headline,
            "execution_mode": {"worker_pool_size": worker_count(value), "stage_wall_s": state["stage_wall_s"], "assessment_wall_s": time.perf_counter() - started},
            "protocol_semantic_sha256": protocol_hash,
        }
        context.write_json("artifacts/campaign-result.json", campaign_result)
        collector["assessment"] = {"headline": headline, "gates": campaign_gates, "failing_designs": failing_designs, "status": status, "replays": replays, "estimands": estimands}
        return Decision(bool(gates_passed), {"status": status, "design_count": len(records), "stable_design_count": headline["stable_design_count"], "gates": campaign_gates})

    return RuntimeCallbacks(prebundle=prebundle, development=development, assessment=assessment)
