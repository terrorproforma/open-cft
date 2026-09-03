"""Campaign mechanics of L1a geometry sweep v3 (HEMP-like wall-radius-to-pitch regime).

Follows the accepted one-shot template (``cusp_topology_search_v3_1`` /
``orbit_wall_loss_geometry_screening_v1``): one :class:`CampaignPlan` drives the
evidentiary campaign and the disclosed NON-EVIDENTIARY shakedown; the shakedown must pass on
real designs of both sets before ``prepare`` freezes the authorities; one detached execution
publishes through the shared :class:`ExperimentRuntime`.

Per design (worker task): build/rebuild + identity + CPU solves at the accepted (sweep-v2)
resolution and at 2x (:mod:`.designs`); the sweep-v2 QoIs and the six sweep-v2 metric gates
verbatim (imported from ``experiments.l1a_geometry_sweep_v2.experiment``); the cusp
topology search v3.1 characterization of both maps (imported definition), refinement
stability and, for the held-out sweep-v2 set, the axis-null correspondence; the v3
descriptors (Koch rho per cusp, PPM I_1(x_w) prediction, wall harmonics, HEMP-like flags)
on both maps (:mod:`.descriptors`). The assessment evaluates the binding integrity gates,
the reported hypothesis test and estimands, and emits the dataset, CSV and the v3 catalogue.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
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

from experiments.cusp_topology_search_v3_1 import experiment as cts_experiment
from experiments.cusp_topology_search_v3_1 import topology as topology_module
from experiments.cusp_topology_search_v3_1.topology import ChannelGeometry, TopologyPolicy, tracing_grid
from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.orbit_wall_loss_geometry_screening_v1.designs import field_pipeline_source_files, field_pipeline_source_sha256

from . import catalogue as catalogue_module
from . import descriptors as descriptor_module
from . import designs as design_module
from .designs import DESIGN_SETS, SET_SOBOL, SET_SWEEP, DesignSpec, ResolvedDesign

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
DESIGN_AUTHORITIES_PATH = EXPERIMENT / "design-authorities.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"
CTS_PROTOCOL_PATH = MODERN / "experiments" / "cusp_topology_search_v3_1" / "protocol.json"

VERSION_TAG = "cft-revival.l1a-geometry-sweep-v3"
CLASSIFICATION = "L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID"
TOPOLOGY_LABEL = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
EXPERIMENT_CODE_FILES = ("__init__.py", "catalogue.py", "descriptors.py", "designs.py", "experiment.py", "run.py", "sampling.py")
V2_GATES_APPLIED = ("boundary", "residual", "flux_identity", "source_representation", "topology_confidence", "manufacturability")
V2_GATE_NOT_APPLICABLE = "cpu_cuda_parity"
CSV_COLUMNS = (
    "set_id",
    "design_id",
    "label",
    "inside_sweep_v2_box",
    "stage_count",
    "stage_pitch_m",
    "wall_radius_m",
    "wall_radius_over_pitch",
    "x_w",
    "i1_x_w",
    "predicted_hemp_like_i1",
    "magnet_inner_radius_m",
    "magnet_radial_thickness_m",
    "magnet_axial_fraction",
    "source_strength_scale",
    "exit_length_m",
    "axis_null_count",
    "wall_cusp_count",
    "cell_count",
    "four_wall_cusps",
    "min_rho_conservative",
    "max_rho_conservative",
    "rho_conservative_per_cusp",
    "rho_downstream_per_cusp",
    "rho_wall_per_cusp",
    "hemp_like_all_cusps",
    "five_stage_four_cusp_hemp_like",
    "wall_b3_over_b1",
    "wall_b5_over_b1",
    "angle_to_wall_normal_deg_max",
    "stable",
    "max_wall_intersection_shift_m",
    "rho_resolution_sensitivity_max",
    "v2_gates_passed",
    "boundary_to_peak_ratio",
    "topology_confidence",
    "relative_residual_l2",
    "centreline_abs_bz_peak_t",
    "field_peak_t",
)


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/1.0.0"


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    if value["classification"] != CLASSIFICATION or value["catalogue"]["label"] != TOPOLOGY_LABEL:
        raise ValueError("protocol classification labels differ from the experiment constants")
    if value["sampling"]["design_count"] < 128:
        raise ValueError("the v3 protocol requires at least 128 Sobol designs")
    return value


def policy_from(value: Mapping[str, Any]) -> TopologyPolicy:
    """The imported definition-v3 numerical parameters; must equal the v3.1 protocol's."""

    declared = value["definition_v3_import"]["numerical_parameters"]
    cts = strict_json_file(CTS_PROTOCOL_PATH)["definition_v3"]["numerical_parameters"]
    if dict(declared) != dict(cts):
        raise ValueError("definition-v3 numerical parameters differ from the v3.1 protocol")
    return TopologyPolicy.from_protocol(declared)


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


def dependency_source_files() -> list[Path]:
    """Imported experiment modules whose bytes determine the topology and the held-out binding."""

    cts = MODERN / "experiments" / "cusp_topology_search_v3_1"
    files = [
        cts / "topology.py",
        cts / "catalogue.py",
        cts / "experiment.py",
        cts / "fields.py",
        cts / "protocol.json",
        MODERN / "experiments" / "cft_topology_characterization_v1" / "experiment.py",
        MODERN / "experiments" / "cft_topology_characterization_v1" / "protocol.json",
        MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "designs.py",
        MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "protocol.json",
        MODERN / "experiments" / "cft_orbit_wall_loss_v4" / "adapter.py",
    ]
    import cft_revival.coupling as coupling_package
    import cft_revival.experiment_runtime as runtime_package
    import cft_revival.orbit_mc as orbit_mc_package

    for package in (coupling_package, runtime_package):
        root = Path(package.__file__).resolve().parent
        if root.parent != (MODERN / "src" / "cft_revival").resolve():
            raise RuntimeError(f"{package.__name__} is imported from {root}, not from this worktree")
        files.extend(sorted(root.glob("*.py")))
    orbit_root = Path(orbit_mc_package.__file__).resolve().parent
    files.extend(orbit_root / name for name in ("fields.py", "models.py"))
    return files


def dependency_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in dependency_source_files():
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(f"dependency source {path.relative_to(MODERN).as_posix()} contains CR bytes")
        digest.update(path.relative_to(MODERN).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def source_binding_report(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_code_sha256": experiment_code_sha256(),
        "experiment_code_files": list(EXPERIMENT_CODE_FILES),
        "dependency_source_sha256": dependency_source_sha256(),
        "dependency_source_files": [path.relative_to(MODERN).as_posix() for path in dependency_source_files()],
        "field_pipeline_source_sha256": field_pipeline_source_sha256(),
        "field_pipeline_source_files": [path.relative_to(MODERN).as_posix() for path in field_pipeline_source_files()],
        "sealed_sources": design_module.sealed_source_binding(),
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
    return design_module.design_specs(value)


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
        return (plan.design_keys[0],)
    return tuple(key for key in declared if key in plan.design_keys)


def worker_count(value: Mapping[str, Any]) -> int:
    execution = value["execution"]
    if not execution["parallel_designs"]:
        return 1
    return max(1, min(int(execution["max_design_workers"]), os.cpu_count() or 1))


def run_stage(tasks: Sequence[Mapping[str, Any]], function: Callable[[Mapping[str, Any]], dict[str, Any]], workers: int) -> list[dict[str, Any]]:
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
    designs = [{**design_module.design_identity_without_solving(spec, value), "key": spec.key, "ordinal": spec.ordinal} for spec in specs]
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.set_id] = counts.get(spec.set_id, 0) + 1
    return {
        "schema_version": schema("design-authorities"),
        "plan_kind": plan.kind,
        "design_count": len(designs),
        "set_counts": counts,
        "sobol_inside_sweep_v2_box_count": sum(1 for item in designs if item["set_id"] == SET_SOBOL and item["inside_sweep_v2_box"]),
        "sobol_predicted_hemp_like_count": sum(1 for item in designs if item["set_id"] == SET_SOBOL and item["x_w"] >= descriptor_module.X_STAR_HEMP_LIKE),
        "designs": designs,
    }


# --------------------------------------------------------------------------
# Sweep-v2 QoIs and gates (verbatim definitions)
# --------------------------------------------------------------------------


def v2_gate_definitions() -> tuple[dict[str, Any], ...]:
    return tuple(dict(gate) for gate in sweep.TERMINAL_GATES if gate["gate_id"] in V2_GATES_APPLIED)


def evaluate_v2_gates(case: sweep.BuiltCase, qois: Mapping[str, Any]) -> dict[str, Any]:
    """The six sweep-v2 metric gates on one design (cpu_cuda_parity is CPU-only here: not applicable)."""

    results = {}
    for definition in v2_gate_definitions():
        gate_id = definition["gate_id"]
        if gate_id == "manufacturability":
            observed = min(case.derived["worst_case_radial_manufacturing_margin_m"], case.derived["worst_case_axial_manufacturing_margin_m"])
        else:
            observed = float(qois[definition["metric"]])
        passed = observed <= definition["limit"] if definition["comparator"] == "<=" else observed >= definition["limit"]
        results[gate_id] = {"observed": observed, "limit": definition["limit"], "comparator": definition["comparator"], "passed": bool(passed)}
    return {
        "gates": results,
        "passed": all(item["passed"] for item in results.values()),
        "not_applicable": {V2_GATE_NOT_APPLICABLE: "the campaign is CPU-only (no CUDA solve to compare); replaced by the determinism_replay gate (bitwise CPU re-solve)"},
    }


# --------------------------------------------------------------------------
# Held-out comparison (sweep-v2 axis nulls)
# --------------------------------------------------------------------------


def held_out_comparison(resolved: ResolvedDesign, accepted: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
    if resolved.spec.set_id != SET_SWEEP:
        return {"kind": "none", "statement": "no sealed reference for a new v3 design", "passed": True, "applies": False}
    tolerance = float(value["definition_v3_import"]["held_out_tolerance_m"])
    low, high = accepted["axis_nulls"]["window_m"]
    reference = [z for z in resolved.reference["sweep_axis_null_positions_m"] if low <= z <= high]
    match = cts_experiment._match_sorted(reference, [null["z_m"] for null in accepted["axis_nulls"]["nulls"]], tolerance)
    qoi_replay = resolved.evidence["qoi_replay"]
    return {
        "kind": "sweep_v2_reproduction",
        "statement": "sealed sweep-v2 QoIs replay within the sweep-v2 tolerances AND every sealed axis null inside the v3 window matches a v3 null (bijection)",
        "qoi_replay_passed": bool(qoi_replay["passed"]),
        "qoi_replay_checks": dict(qoi_replay["checks"]),
        "stored_representative_passed": None if resolved.evidence["stored_representative"] is None else bool(resolved.evidence["stored_representative"]["passed"]),
        **match,
        "passed": bool(match["bijection"] and qoi_replay["passed"] and (resolved.evidence["stored_representative"] is None or resolved.evidence["stored_representative"]["passed"])),
        "applies": True,
    }


# --------------------------------------------------------------------------
# Per-design worker
# --------------------------------------------------------------------------


def _channel_geometry(case: sweep.BuiltCase) -> ChannelGeometry:
    chamber = case.geometry.chamber
    return ChannelGeometry(
        wall_radius_m=float(chamber.outer_radius_m),
        straight_z_min_m=0.0,
        straight_z_max_m=float(chamber.exit_start_m),
        chamber_length_m=float(chamber.length_m),
        stage_pitch_m=float(case.derived["represented_stage_pitch_m"]),
        stage_centres_m=tuple(float(stage.center_z_m) for stage in case.geometry.stages),
        injector_length_m=float(chamber.injector_length_m),
    )


def design_gate_checks(record: Mapping[str, Any]) -> dict[str, bool]:
    accepted = record["accepted"]
    refined = record["refined"]
    return {
        "identity_proven": bool(record["evidence"]["identity_proven"]),
        "sweep_v2_gates_verbatim": bool(record["v2_gates"]["passed"]),
        "solver_converged_both_maps": bool(record["evidence"]["accepted_solve"]["converged"] and record["evidence"]["refined_solve"]["converged"]),
        "every_null_converged": cts_experiment._nulls_converged(accepted) and cts_experiment._nulls_converged(refined),
        "every_trace_terminates_cleanly": bool(accepted["all_traces_terminate_cleanly"] and refined["all_traces_terminate_cleanly"]),
        "every_wall_trace_flux_consistent": bool(accepted["all_wall_traces_flux_consistent"] and refined["all_wall_traces_flux_consistent"]),
        "refinement_stability": bool(record["stability"]["stable"]),
        "held_out_sweep_v2_reproduction": bool(record["held_out"]["passed"]),
    }


def characterize_resolved(resolved: ResolvedDesign, value: Mapping[str, Any], *, keep_paths: bool) -> dict[str, Any]:
    """Sweep QoIs + gates, definition-v3 topology of both maps, stability, held-out, descriptors (pure)."""

    policy = policy_from(value)
    tightness = float(value["definition_v3_import"]["minimum_certificate_dense_to_bound_ratio"])
    geometry = _channel_geometry(resolved.case)
    qois = sweep.extract_qois(resolved.case, resolved.accepted)
    v2_gates = evaluate_v2_gates(resolved.case, qois)
    accepted_grid = tracing_grid(resolved.accepted.r_m, resolved.accepted.z_m, resolved.accepted.psi_wb, resolved.accepted.b_r_t, resolved.accepted.b_z_t, geometry.wall_radius_m)
    refined_grid = tracing_grid(resolved.refined.r_m, resolved.refined.z_m, resolved.refined.psi_wb, resolved.refined.b_r_t, resolved.refined.b_z_t, geometry.wall_radius_m)
    window = topology_module.axis_window(accepted_grid, geometry, policy)
    sweep_peaks = resolved.reference.get("sweep_axis_bz_peak_positions_m") or list(qois["axis_cusp_positions_m"])
    accepted = topology_module.characterize_map(
        accepted_grid, geometry, policy, source_identity_sha256=resolved.identity["accepted_field_identity_sha256"], minimum_certificate_tightness_ratio=tightness, keep_paths=keep_paths, sweep_axis_bz_peaks_m=sweep_peaks, axis_window_m=window
    )
    refined = topology_module.characterize_map(
        refined_grid, geometry, policy, source_identity_sha256=resolved.identity["refined_field_identity_sha256"], minimum_certificate_tightness_ratio=tightness, keep_paths=False, sweep_axis_bz_peaks_m=sweep_peaks, axis_window_m=window
    )
    stability = topology_module.compare_resolutions(accepted, refined, float(value["definition_v3_import"]["stability_tolerance_m"]))
    held_out = held_out_comparison(resolved, accepted, value)
    stage_count = int(resolved.case.derived["stage_count"])
    descriptors_accepted = descriptor_module.design_descriptors(
        accepted_grid, geometry, accepted, policy, source_identity_sha256=resolved.identity["accepted_field_identity_sha256"], minimum_certificate_tightness_ratio=tightness, stage_count=stage_count, with_profiles=True
    )
    descriptors_refined = descriptor_module.design_descriptors(
        refined_grid, geometry, refined, policy, source_identity_sha256=resolved.identity["refined_field_identity_sha256"], minimum_certificate_tightness_ratio=tightness, stage_count=stage_count, with_profiles=False
    )
    descriptors_accepted["x_m_inner"] = math.pi * float(resolved.case.derived["magnet_inner_radius_m"]) / geometry.stage_pitch_m
    descriptors_refined["x_m_inner"] = descriptors_accepted["x_m_inner"]
    payload = _plain(
        {
            "axis_window_m": [window[0], window[1]],
            "qois": qois,
            "v2_gates": v2_gates,
            "accepted": accepted,
            "refined": refined,
            "stability": stability,
            "held_out": held_out,
            "descriptors": {
                "accepted": descriptors_accepted,
                "refined": descriptors_refined,
                "resolution_sensitivity": descriptor_module.resolution_sensitivity(descriptors_accepted, descriptors_refined),
            },
        }
    )
    payload["topology_payload_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def run_design_task(task: Mapping[str, Any]) -> dict[str, Any]:
    spec = DesignSpec(**task["spec"])
    value = task["protocol"]
    started = time.perf_counter()
    try:
        resolved = design_module.resolve_design(spec, value)
    except Exception as error:  # recorded, never hidden
        return _plain({"key": spec.key, "set_id": spec.set_id, "design_id": spec.design_id, "ordinal": spec.ordinal, "representative": spec.representative, "label": TOPOLOGY_LABEL, "status": "failed", "stage": "resolve", "reason": f"{type(error).__name__}: {error}", "timing_s": {"total": time.perf_counter() - started}})
    characterize_started = time.perf_counter()
    try:
        topology = characterize_resolved(resolved, value, keep_paths=bool(task["keep_paths"]))
    except Exception as error:  # recorded, never hidden
        return _plain({"key": spec.key, "set_id": spec.set_id, "design_id": spec.design_id, "ordinal": spec.ordinal, "representative": spec.representative, "label": TOPOLOGY_LABEL, "status": "failed", "stage": "characterize", "reason": f"{type(error).__name__}: {error}", "identity": resolved.identity, "timing_s": {"solve": resolved.solve_seconds, "total": time.perf_counter() - started}})
    record = {
        "schema_version": schema("design-record"),
        "key": spec.key,
        "set_id": spec.set_id,
        "design_id": spec.design_id,
        "ordinal": spec.ordinal,
        "representative": spec.representative,
        "label": TOPOLOGY_LABEL,
        "classification": CLASSIFICATION,
        "status": "resolved",
        "identity": resolved.identity,
        "evidence": resolved.evidence,
        "reference": resolved.reference,
        "geometry": _channel_geometry(resolved.case).to_dict(),
        **topology,
        "timing_s": {"solve": resolved.solve_seconds, "characterize": time.perf_counter() - characterize_started, "total": time.perf_counter() - started},
        "accepted_grid": {
            "r_m": list(resolved.accepted.r_m),
            "z_m": list(resolved.accepted.z_m),
            "psi_wb": [list(row) for row in resolved.accepted.psi_wb],
        },
    }
    record["gate_checks"] = design_gate_checks(record)
    record["record_path"] = f"artifacts/designs/{spec.set_id}/{spec.design_id}.json"
    return _plain(record)


# --------------------------------------------------------------------------
# Estimands
# --------------------------------------------------------------------------


def _range(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(clean), "min": min(clean), "median": statistics.median(clean), "max": max(clean)}


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in values:
        counts[str(int(item))] = counts.get(str(int(item)), 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: int(pair[0])))


def hypothesis_test(records: Sequence[Mapping[str, Any]], *, band: float) -> dict[str, Any]:
    """rho_conservative vs I_1(x_w) over every wall cusp of the given designs (reported)."""

    pairs = []
    for record in records:
        descriptors = record["descriptors"]["accepted"]
        i1 = descriptors["ppm_prediction"]["i1_x_w"]
        for row in descriptors["cusps"]:
            if row["rho_conservative"] is not None:
                pairs.append((i1, row["rho_conservative"], descriptors["x_w"], record["key"], row["cusp_id"]))
    if not pairs:
        return {"cusp_count": 0}
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    slope = float(np.sum(x * y) / np.sum(x * x))
    fitted = slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ratios = y / x
    within = np.abs(ratios - 1.0) <= band
    design_level = []
    for record in records:
        descriptors = record["descriptors"]["accepted"]
        if descriptors["wall_cusp_count"] == 0:
            continue
        design_level.append((bool(descriptors["predicted_hemp_like_i1"]), bool(descriptors["hemp_like_all_cusps"]), descriptors["x_w"]))
    confusion = {
        "predicted_and_realised": sum(1 for p, r, _ in design_level if p and r),
        "predicted_not_realised": sum(1 for p, r, _ in design_level if p and not r),
        "not_predicted_but_realised": sum(1 for p, r, _ in design_level if not p and r),
        "neither": sum(1 for p, r, _ in design_level if not p and not r),
    }
    realised_x = sorted(xw for _, r, xw in design_level if r)
    not_realised_x = sorted(xw for _, r, xw in design_level if not r)
    threshold_from_slope = descriptor_module.i1_root(descriptor_module.HEMP_LIKE_RHO / slope) if slope > 0.0 else None
    return {
        "cusp_count": len(pairs),
        "design_count_with_cusps": len(design_level),
        "slope_through_origin": slope,
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None,
        "rho_over_i1": _range(ratios.tolist()),
        "fraction_within_band": float(np.mean(within)),
        "band": band,
        "confusion_predicted_i1_vs_realised": confusion,
        "prediction_accuracy": ((confusion["predicted_and_realised"] + confusion["neither"]) / len(design_level)) if design_level else None,
        "smallest_x_w_realised_hemp_like": realised_x[0] if realised_x else None,
        "largest_x_w_not_hemp_like": not_realised_x[-1] if not_realised_x else None,
        "x_star_prediction": descriptor_module.X_STAR_HEMP_LIKE,
        "x_star_from_fitted_slope": threshold_from_slope,
        "wall_radius_over_pitch_star_from_fitted_slope": (threshold_from_slope / math.pi) if threshold_from_slope else None,
    }


def set_estimands(records: Sequence[Mapping[str, Any]], *, band: float) -> dict[str, Any]:
    descriptors = [record["descriptors"]["accepted"] for record in records]
    with_cusps = [d for d in descriptors if d["wall_cusp_count"] > 0]
    cusps = [row for d in descriptors for row in d["cusps"]]
    by_stage: dict[str, dict[str, Any]] = {}
    for d in descriptors:
        bucket = by_stage.setdefault(str(d["stage_count"]), {"designs": 0, "hemp_like": 0, "n_minus_1_cusps": 0, "x_w": []})
        bucket["designs"] += 1
        bucket["hemp_like"] += int(d["hemp_like_all_cusps"])
        bucket["n_minus_1_cusps"] += int(d["wall_cusp_count"] == d["expected_interior_cusps_n_minus_1"])
        bucket["x_w"].append(d["x_w"])
    for bucket in by_stage.values():
        bucket["x_w"] = _range(bucket["x_w"])
    hemp = [d for d in descriptors if d["hemp_like_all_cusps"]]
    return {
        "design_count": len(records),
        "stable_design_count": sum(record["stability"]["stable"] for record in records),
        "v2_gates_passed_count": sum(record["v2_gates"]["passed"] for record in records),
        "wall_cusp_count_histogram": _histogram([d["wall_cusp_count"] for d in descriptors]),
        "axis_null_count_histogram": _histogram([record["accepted"]["axis_nulls"]["count"] for record in records]),
        "n_minus_1_cusp_fraction": (sum(d["wall_cusp_count"] == d["expected_interior_cusps_n_minus_1"] for d in descriptors) / len(descriptors)) if descriptors else None,
        "hemp_like_count": len(hemp),
        "hemp_like_fraction": (len(hemp) / len(records)) if records else None,
        "hemp_like_fraction_among_designs_with_cusps": (len(hemp) / len(with_cusps)) if with_cusps else None,
        "predicted_hemp_like_i1_count": sum(d["predicted_hemp_like_i1"] for d in descriptors),
        "five_stage_four_cusp_hemp_like_count": sum(d["five_stage_four_cusp_hemp_like"] for d in descriptors),
        "four_wall_cusp_count": sum(d["four_wall_cusps"] for d in descriptors),
        "by_stage_count": by_stage,
        "x_w": _range([d["x_w"] for d in descriptors]),
        "wall_radius_over_pitch": _range([d["wall_radius_over_pitch"] for d in descriptors]),
        "rho_conservative": _range([row["rho_conservative"] for row in cusps]),
        "rho_downstream": _range([row["rho_downstream"] for row in cusps]),
        "rho_wall": _range([row["rho_wall"] for row in cusps]),
        "cusp_is_wall_maximum_count": sum(row["cusp_is_wall_maximum"] for row in cusps),
        "cusp_count": len(cusps),
        "angle_to_wall_normal_deg": _range([row["angle_to_wall_normal_deg"] for row in cusps]),
        "wall_b3_over_b1": _range([d["wall_harmonics"].get("b3_over_b1") for d in descriptors if d["wall_harmonics"].get("applies")]),
        "wall_b5_over_b1": _range([d["wall_harmonics"].get("b5_over_b1") for d in descriptors if d["wall_harmonics"].get("applies")]),
        "rho_resolution_sensitivity_max": _range([record["descriptors"]["resolution_sensitivity"]["max_relative_rho_difference"] for record in records]),
        "max_wall_intersection_shift_m": _range([record["stability"]["max_wall_intersection_shift_m"] for record in records]),
        "hemp_like_region": None
        if not hemp
        else {
            "x_w": _range([d["x_w"] for d in hemp]),
            "wall_radius_over_pitch": _range([d["wall_radius_over_pitch"] for d in hemp]),
            "x_m_inner": _range([d["x_m_inner"] for d in hemp]),
            "stage_counts": _histogram([d["stage_count"] for d in hemp]),
            "wall_cusp_counts": _histogram([d["wall_cusp_count"] for d in hemp]),
            "min_rho_conservative": _range([d["min_rho_conservative"] for d in hemp]),
        },
        "hypothesis_test": hypothesis_test(records, band=band),
    }


def dataset_row(record: Mapping[str, Any]) -> dict[str, Any]:
    accepted = record["accepted"]
    topology = accepted["topology"]
    descriptors = record["descriptors"]["accepted"]
    qois = record["qois"]
    derived = record["evidence"]["derived_geometry"]
    return {
        "key": record["key"],
        "set_id": record["set_id"],
        "design_id": record["design_id"],
        "ordinal": record["ordinal"],
        "representative": record["representative"],
        "label": record["label"],
        "classification": record["classification"],
        "record_path": record["record_path"],
        "design_values": dict(record["evidence"]["design_values"]),
        "sampling_provenance": record["evidence"]["sampling_provenance"],
        "inside_sweep_v2_box": bool(derived.get("inside_sweep_v2_box", record["set_id"] == SET_SWEEP)),
        "geometry": record["geometry"],
        "derived": {key: derived[key] for key in ("stage_count", "represented_stage_pitch_m", "magnet_inner_radius_m", "magnet_outer_radius_m", "magnet_axial_thickness_m", "represented_exit_length_m", "represented_exit_outer_radius_m", "chamber_length_m")},
        "identity": {key: record["identity"][key] for key in ("accepted_field_identity_sha256", "refined_field_identity_sha256", "case_sha256")},
        "x_w": descriptors["x_w"],
        "x_m_inner": descriptors["x_m_inner"],
        "wall_radius_over_pitch": descriptors["wall_radius_over_pitch"],
        "ppm_prediction": descriptors["ppm_prediction"],
        "axis_nulls": [{"null_id": n["null_id"], "z_m": n["z_m"], "zone": n["zone"], "classification": n["classification"]} for n in accepted["axis_nulls"]["nulls"]],
        "axis_null_count": accepted["axis_nulls"]["count"],
        "wall_cusps": [{key: cusp[key] for key in ("cusp_id", "axis_null_z_m", "z_c_m", "z_c_over_length", "wall_b_t", "wall_b_r_t", "angle_to_wall_normal_deg", "boundary_ambiguous", "distance_to_nearest_stage_gap_m")} for cusp in topology["wall_cusps"]],
        "outside_intersections": [{key: row[key] for key in ("cusp_id", "z_c_m", "zone", "wall_b_t")} for row in topology["outside_intersections"]],
        "cells": [{key: cell[key] for key in ("cell_id", "kind", "z_start_m", "z_end_m", "length_over_pitch", "wall_b_min_t", "wall_mirror_ratio", "axis_bz_peak_t", "axis_mirror_ratio")} for cell in topology["cells"]],
        "wall_cusp_count": topology["wall_cusp_count"],
        "cell_count": topology["cell_count"],
        "four_wall_cusps": topology["four_wall_cusps"],
        "rho": [{key: row[key] for key in ("cusp_id", "z_c_m", "wall_b_t", "upstream_axis_peak_t", "downstream_axis_peak_t", "rho_upstream", "rho_downstream", "rho_conservative", "rho_wall", "hemp_like_conservative", "cusp_is_wall_maximum")} for row in descriptors["cusps"]],
        "min_rho_conservative": descriptors["min_rho_conservative"],
        "hemp_like_all_cusps": descriptors["hemp_like_all_cusps"],
        "predicted_hemp_like_i1": descriptors["predicted_hemp_like_i1"],
        "five_stage_four_cusp_hemp_like": descriptors["five_stage_four_cusp_hemp_like"],
        "wall_harmonics": {key: descriptors["wall_harmonics"].get(key) for key in ("applies", "b3_over_b1", "b5_over_b1", "fit_rms_over_max", "wall_b_r_max_abs_t")},
        "resolution_sensitivity": {key: record["descriptors"]["resolution_sensitivity"][key] for key in ("comparable", "max_relative_rho_difference", "hemp_like_flag_agrees") if key in record["descriptors"]["resolution_sensitivity"]},
        "qois": {key: qois[key] for key in ("centreline_abs_bz_peak_t", "centreline_mid_abs_bz_t", "minimum_mirror_ratio", "maximum_mirror_ratio", "axis_cusp_count", "axis_null_count", "stage_gradient_rms_t_per_m", "boundary_to_peak_ratio", "field_energy_j", "source_representation_error", "topology_confidence", "field_peak_t", "relative_residual_l2", "flux_reconstruction_identity_t_per_m")},
        "v2_gates": {"passed": record["v2_gates"]["passed"], "gates": {gate_id: item["passed"] for gate_id, item in record["v2_gates"]["gates"].items()}},
        "stability": {key: record["stability"][key] for key in ("stable", "axis_null_count_equal", "wall_reaching_count_equal", "wall_cusp_count_equal", "max_axis_null_shift_m", "max_wall_intersection_shift_m")},
        "held_out": {key: record["held_out"].get(key) for key in ("kind", "applies", "passed", "qoi_replay_passed", "reference_count", "observed_count", "max_difference_m")},
        "gate_checks": record["gate_checks"],
        "grid": {key: accepted["grid"][key] for key in ("radial_samples", "axial_samples", "dr_m", "dz_m", "radial_cells_across_bore")},
        "refined_grid": {key: record["refined"]["grid"][key] for key in ("radial_samples", "axial_samples", "dr_m", "dz_m")},
        "timing_s": record["timing_s"],
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def dataset_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        rho = [item["rho_conservative"] for item in row["rho"] if item["rho_conservative"] is not None]
        angles = [cusp["angle_to_wall_normal_deg"] for cusp in row["wall_cusps"]]
        values = row["design_values"]
        writer.writerow(
            [
                row["set_id"],
                row["design_id"],
                row["label"],
                _csv_value(row["inside_sweep_v2_box"]),
                row["derived"]["stage_count"],
                _csv_value(row["derived"]["represented_stage_pitch_m"]),
                _csv_value(row["geometry"]["wall_radius_m"]),
                _csv_value(row["wall_radius_over_pitch"]),
                _csv_value(row["x_w"]),
                _csv_value(row["ppm_prediction"]["i1_x_w"]),
                _csv_value(row["predicted_hemp_like_i1"]),
                _csv_value(row["derived"]["magnet_inner_radius_m"]),
                _csv_value(values["magnet_radial_thickness_m"]),
                _csv_value(values["magnet_axial_fraction"]),
                _csv_value(values["source_strength_scale"]),
                _csv_value(row["derived"]["represented_exit_length_m"]),
                row["axis_null_count"],
                row["wall_cusp_count"],
                row["cell_count"],
                _csv_value(row["four_wall_cusps"]),
                _csv_value(min(rho) if rho else None),
                _csv_value(max(rho) if rho else None),
                ";".join(repr(v) for v in rho),
                ";".join(_csv_value(item["rho_downstream"]) for item in row["rho"]),
                ";".join(_csv_value(item["rho_wall"]) for item in row["rho"]),
                _csv_value(row["hemp_like_all_cusps"]),
                _csv_value(row["five_stage_four_cusp_hemp_like"]),
                _csv_value(row["wall_harmonics"]["b3_over_b1"]),
                _csv_value(row["wall_harmonics"]["b5_over_b1"]),
                _csv_value(max(angles) if angles else None),
                _csv_value(row["stability"]["stable"]),
                _csv_value(row["stability"]["max_wall_intersection_shift_m"]),
                _csv_value(row["resolution_sensitivity"].get("max_relative_rho_difference")),
                _csv_value(row["v2_gates"]["passed"]),
                _csv_value(row["qois"]["boundary_to_peak_ratio"]),
                _csv_value(row["qois"]["topology_confidence"]),
                _csv_value(row["qois"]["relative_residual_l2"]),
                _csv_value(row["qois"]["centreline_abs_bz_peak_t"]),
                _csv_value(row["qois"]["field_peak_t"]),
            ]
        )
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Shakedown record verification and frozen authorities
# --------------------------------------------------------------------------


def verify_shakedown_record(value: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False and record.get("outcomes_enter_estimand") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = record.get("protocol_semantic_sha256") == semantic_sha256(value)
    checks["experiment_code_sha256_current"] = record.get("experiment_code_sha256") == experiment_code_sha256()
    checks["dependency_source_sha256_current"] = record.get("dependency_source_sha256") == dependency_source_sha256()
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
    return FrozenAuthority(strict_json_file(AUTHORITIES_PATH), strict_json_file(DESIGN_AUTHORITIES_PATH), strict_json_file(SHAKEDOWN_PATH), SHAKEDOWN_PATH.read_bytes())


# --------------------------------------------------------------------------
# Runtime callbacks
# --------------------------------------------------------------------------


GATE_NAMES = (
    "identity_proven",
    "sweep_v2_gates_verbatim",
    "solver_converged_both_maps",
    "every_null_converged",
    "every_trace_terminates_cleanly",
    "every_wall_trace_flux_consistent",
    "refinement_stability",
    "held_out_sweep_v2_reproduction",
)


def build_callbacks(value: Mapping[str, Any], plan: CampaignPlan, *, frozen: FrozenAuthority | None, collector: dict[str, Any]) -> RuntimeCallbacks:
    if (plan.kind == "evidentiary") != (frozen is not None):
        raise ValueError("evidentiary runs require frozen authorities; shakedowns forbid them")
    state: dict[str, Any] = {}
    collector.setdefault("plan_kind", plan.kind)
    band = float(value["descriptors_v3"]["hypothesis"]["agreement_band_relative"])

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
            context.write_json("artifacts/shakedown-disclosure.json", {"evidentiary": False, "outcomes_enter_estimand": False, "statement": value["shakedown"]["purpose"]})
        context.write_json(
            "artifacts/runtime.json",
            {
                "generated_at_utc": datetime.now(timezone.utc),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "cpu_count": os.cpu_count(),
                "worker_pool_size": worker_count(value),
                "backend": "cft_revival.fields.solve_problem_cpu + PsiBicubicField tracing (CPU only; GPU not used)",
            },
        )
        state.update({"binding": binding, "design_authorities": design_authorities})
        collector["prebundle"] = {"design_count": design_authorities["design_count"], "set_counts": design_authorities["set_counts"]}
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "classification": CLASSIFICATION,
            "topology_label": TOPOLOGY_LABEL,
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
        context.before_expensive("build-solve-characterize-all-designs", kind="solver", details={"design_count": len(specs), "worker_pool_size": workers, "plan_kind": plan.kind, "solver": "cft_revival.fields.solve_problem_cpu"})
        keep_paths_sets = set(value["execution"]["keep_paths_for_sets"])
        tasks = [{"spec": spec.to_dict(), "protocol": dict(value), "keep_paths": bool(spec.set_id in keep_paths_sets or spec.representative)} for spec in specs]
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
            for hash_key in ("geometry_sha256", "source_sha256", "config_sha256", "case_sha256"):
                if authority[hash_key] != outcome["identity"].get(hash_key):
                    raise ValueError(f"{outcome['key']}: resolved {hash_key} differs from the design authority")
            records[outcome["key"]] = outcome
            grid = outcome.pop("accepted_grid")
            grids[outcome["key"]] = grid["psi_wb"]
            field_bytes = canonical_bytes({"key": outcome["key"], "identity": outcome["identity"], **grid})
            outcome["accepted_grid_payload_sha256"] = hashlib.sha256(field_bytes).hexdigest()
            outcome["accepted_grid_path"] = f"artifacts/fields/{outcome['set_id']}/{outcome['design_id']}.json.gz"
            context.write_blob(outcome["accepted_grid_path"], gzip.compress(field_bytes, compresslevel=9, mtime=0))
            context.write_json(outcome["record_path"], outcome)
        context.write_json("artifacts/design-failures.json", {"schema_version": schema("design-failures"), "rule": value["gates"]["binding_integrity"]["all_declared_designs_resolved"], "failed": failures})
        accepted = bool(records) and not failures
        state.update({"records": records, "grids": grids, "failures": failures, "stage_wall_s": stage_wall})
        collector["development"] = {"resolved_design_count": len(records), "failures": failures, "stage_wall_s": stage_wall, "seconds": time.perf_counter() - started, "accepted": accepted, "per_design_seconds": {key: item["timing_s"] for key, item in records.items()}}
        return Decision(accepted, {"resolved_design_count": len(records), "failed_design_count": len(failures), "stage_wall_s": stage_wall})

    def assessment(context: Any) -> Decision:
        started = time.perf_counter()
        records = state["records"]
        ordered = [records[key] for key in plan.design_keys if key in records]
        replays = []
        for key in replay_keys(value, plan):
            if key not in records:
                continue
            spec = next(item for item in specs_for_plan(value, plan) if item.key == key)
            context.before_expensive("replay-build-solve-characterize", kind="solver", details={"key": key})
            resolved = design_module.resolve_design(spec, value)
            keep = bool(spec.set_id in set(value["execution"]["keep_paths_for_sets"]) or spec.representative)
            replay = characterize_resolved(resolved, value, keep_paths=keep)
            replays.append(
                {
                    "key": key,
                    "worker_topology_payload_sha256": records[key]["topology_payload_sha256"],
                    "replay_topology_payload_sha256": replay["topology_payload_sha256"],
                    "field_identity_equal": resolved.identity["accepted_field_identity_sha256"] == records[key]["identity"]["accepted_field_identity_sha256"],
                    "accepted_grid_equal": _plain([list(row) for row in resolved.accepted.psi_wb]) == state["grids"][key],
                    "bit_identical": replay["topology_payload_sha256"] == records[key]["topology_payload_sha256"],
                }
            )
        replay_passed = bool(replays) and all(item["bit_identical"] and item["field_identity_equal"] and item["accepted_grid_equal"] for item in replays)
        per_design_gates = {key: record["gate_checks"] for key, record in records.items()}
        campaign_gates = {
            "all_declared_designs_resolved": not state["failures"] and len(records) == len(plan.design_keys),
            **{name: all(checks[name] for checks in per_design_gates.values()) for name in GATE_NAMES},
            "determinism_replay": replay_passed,
            "hash_bindings": True,
        }
        definitions_by_id = v2_gate_definitions_by_id()
        v2_gate_breakdown = {}
        for gate_id in V2_GATES_APPLIED:
            comparator = definitions_by_id[gate_id]["comparator"]
            observed = [record["v2_gates"]["gates"][gate_id]["observed"] for record in ordered]
            v2_gate_breakdown[gate_id] = {
                "passed": all(record["v2_gates"]["gates"][gate_id]["passed"] for record in ordered),
                "failed_designs": sorted(record["key"] for record in ordered if not record["v2_gates"]["gates"][gate_id]["passed"]),
                "observed_extreme": (max(observed) if comparator == "<=" else min(observed)) if observed else None,
                "limit": definitions_by_id[gate_id]["limit"],
                "comparator": comparator,
            }
        failing_designs = {name: sorted(key for key, checks in per_design_gates.items() if not checks[name]) for name in GATE_NAMES}
        gates_passed = all(campaign_gates.values())
        by_set: dict[str, list[dict[str, Any]]] = {set_id: [] for set_id in DESIGN_SETS}
        for record in ordered:
            by_set[record["set_id"]].append(record)
        estimands = {set_id: set_estimands(items, band=band) for set_id, items in by_set.items() if items}
        estimands["pooled_all"] = set_estimands(ordered, band=band)
        v2_region = [record for record in ordered if record["set_id"] == SET_SWEEP or record["evidence"]["derived_geometry"].get("inside_sweep_v2_box")]
        estimands["sweep_v2_region_pooled"] = set_estimands(v2_region, band=band) if v2_region else None
        rows = [dataset_row(record) for record in ordered]
        held_out = None
        if by_set[SET_SWEEP]:
            items = by_set[SET_SWEEP]
            differences = [record["held_out"].get("max_difference_m") for record in items if record["held_out"].get("max_difference_m") is not None]
            held_out = {
                "applies": True,
                "design_count": len(items),
                "passed_count": sum(record["held_out"]["passed"] for record in items),
                "qoi_replay_passed_count": sum(record["held_out"]["qoi_replay_passed"] for record in items),
                "axis_null_bijection_count": sum(record["held_out"]["bijection"] for record in items),
                "max_axis_null_difference_m": max(differences) if differences else None,
                "reference_null_count": sum(record["held_out"].get("reference_count") or 0 for record in items),
                "observed_null_count": sum(record["held_out"].get("observed_count") or 0 for record in items),
                "stored_representatives_checked": sum(1 for record in items if record["held_out"].get("stored_representative_passed") is not None),
            }
        protocol_hash = semantic_sha256(value)
        catalogue = catalogue_module.build_catalogue(value, ordered, protocol_semantic_sha256=protocol_hash)
        sobol = estimands.get(SET_SOBOL)
        headline = {
            "design_count": len(ordered),
            "set_counts": {set_id: len(items) for set_id, items in by_set.items() if items},
            "stable_design_count": sum(record["stability"]["stable"] for record in ordered),
            "v2_gates_passed_count": sum(record["v2_gates"]["passed"] for record in ordered),
            "sobol_hemp_like_count": None if sobol is None else sobol["hemp_like_count"],
            "sobol_hemp_like_fraction": None if sobol is None else sobol["hemp_like_fraction"],
            "sobol_predicted_hemp_like_i1_count": None if sobol is None else sobol["predicted_hemp_like_i1_count"],
            "sobol_five_stage_four_cusp_hemp_like_count": None if sobol is None else sobol["five_stage_four_cusp_hemp_like_count"],
            "sobol_wall_cusp_count_histogram": None if sobol is None else sobol["wall_cusp_count_histogram"],
            "sobol_rho_conservative": None if sobol is None else sobol["rho_conservative"],
            "sobol_hypothesis_test": None if sobol is None else {key: sobol["hypothesis_test"].get(key) for key in ("cusp_count", "slope_through_origin", "r_squared", "fraction_within_band", "band", "confusion_predicted_i1_vs_realised", "prediction_accuracy", "smallest_x_w_realised_hemp_like", "largest_x_w_not_hemp_like", "x_star_prediction", "x_star_from_fitted_slope", "wall_radius_over_pitch_star_from_fitted_slope")},
            "sobol_hemp_like_region": None if sobol is None else sobol["hemp_like_region"],
            "sweep_v2_region_hemp_like_count": None if estimands["sweep_v2_region_pooled"] is None else estimands["sweep_v2_region_pooled"]["hemp_like_count"],
            "sweep_v2_region_max_rho_conservative": None if estimands["sweep_v2_region_pooled"] is None else estimands["sweep_v2_region_pooled"]["rho_conservative"]["max"],
            "pooled_rho_conservative": estimands["pooled_all"]["rho_conservative"],
            "pooled_rho_wall": estimands["pooled_all"]["rho_wall"],
            "pooled_cusp_is_wall_maximum_count": estimands["pooled_all"]["cusp_is_wall_maximum_count"],
            "pooled_cusp_count": estimands["pooled_all"]["cusp_count"],
            "held_out": held_out,
            "max_wall_intersection_shift_m": estimands["pooled_all"]["max_wall_intersection_shift_m"]["max"],
            "rho_resolution_sensitivity_max": estimands["pooled_all"]["rho_resolution_sensitivity_max"]["max"],
        }
        dataset = {
            "schema_version": schema("sweep-dataset"),
            "experiment_id": value["experiment_id"],
            "classification": CLASSIFICATION,
            "topology_label": TOPOLOGY_LABEL,
            "classification_statement": value["classification_statement"],
            "claim_boundary": value["claim_boundary"],
            "hypothesis": value["descriptors_v3"]["hypothesis"],
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
            "headline": headline,
            "gates": {"campaign": campaign_gates, "sweep_v2_gate_breakdown": v2_gate_breakdown, "failing_designs": failing_designs, "passed": gates_passed},
        }
        gates_record = {
            "schema_version": schema("gates"),
            "binding": plan.binding_gates,
            "campaign": campaign_gates,
            "sweep_v2_gate_breakdown": v2_gate_breakdown,
            "sweep_v2_gate_not_applicable": {V2_GATE_NOT_APPLICABLE: "CPU-only campaign; replaced by determinism_replay"},
            "hash_bindings_note": "verified against the frozen authorities in the prebundle" if frozen is not None else "shakedown: recorded, no frozen authority to compare",
            "failing_designs": failing_designs,
            "per_design": per_design_gates,
            "replays": replays,
            "passed": gates_passed,
            "design_count": len(records),
            "definitions": value["gates"],
        }
        context.write_json("artifacts/gates.json", gates_record)
        context.write_json("artifacts/sweep-dataset.json", dataset)
        context.write_blob("artifacts/sweep-dataset.csv", dataset_csv(rows))
        context.write_json(catalogue_module.CATALOGUE_RELATIVE_PATH, catalogue)
        status = "accepted_l1a_sweep_v3" if (gates_passed and plan.kind == "evidentiary") else ("shakedown_passed" if gates_passed else "gates_failed")
        campaign_result = {
            "schema_version": schema("campaign-result"),
            "experiment_id": value["experiment_id"],
            "classification": CLASSIFICATION,
            "topology_label": TOPOLOGY_LABEL,
            "plan_kind": plan.kind,
            "evidentiary": plan.kind == "evidentiary",
            "status": status,
            "gates_passed": gates_passed,
            "campaign_gates": campaign_gates,
            "design_count": len(records),
            "set_counts": headline["set_counts"],
            "headline": headline,
            "l1b_p2_confirmation_queue": value["claim_boundary"]["l1b_p2_confirmation"],
            "execution_mode": {"worker_pool_size": worker_count(value), "stage_wall_s": state["stage_wall_s"], "assessment_wall_s": time.perf_counter() - started},
            "protocol_semantic_sha256": protocol_hash,
        }
        context.write_json("artifacts/campaign-result.json", campaign_result)
        collector["assessment"] = {"headline": headline, "gates": campaign_gates, "failing_designs": failing_designs, "status": status, "replays": replays, "estimands": estimands}
        return Decision(bool(gates_passed), {"status": status, "design_count": len(records), "stable_design_count": headline["stable_design_count"], "gates": campaign_gates})

    return RuntimeCallbacks(prebundle=prebundle, development=development, assessment=assessment)


def v2_gate_definitions_by_id() -> dict[str, dict[str, Any]]:
    return {gate["gate_id"]: gate for gate in v2_gate_definitions()}
