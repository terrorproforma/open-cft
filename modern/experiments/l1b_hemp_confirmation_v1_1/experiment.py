"""Campaign mechanics of the L1b HEMP confirmation v1.1 (material-aware P2 check of 15 L1a designs).

v1.1 differs from the recorded v1 (development_rejection, commit 978c71be) in two declared
points only: the level-0 mesh minimum-angle gate is 5 deg instead of the qualification's 10 deg
(two designs have 5.3 / 9.3 deg meshes from geometric near-coincidences; the sliver statistics
are recorded per level) and the shakedown records a whole-set mesh preflight (every design's
level-0 mesh built and gated before the freeze) that ``prepare`` and ``execute`` verify.

Follows the accepted one-shot template (``l1a_geometry_sweep_v3`` / ``cusp_topology_search_v3_1``):
one :class:`CampaignPlan` drives the evidentiary campaign and the disclosed NON-EVIDENTIARY
shakedown; the shakedown must pass on real designs before ``prepare`` freezes the
authorities; one detached execution publishes through the shared :class:`ExperimentRuntime`.

Per design (sequential, one at a time): rebuild the sweep-v3 case with identity proof and load
the sealed L1a record (:mod:`.designs`); solve the material-aware P2 field on two nested
adaptive levels under the RAM guard and sample it on regular grids (:mod:`.p2_fields`); apply
the cusp topology search v3.1 definition (imported unchanged) to the coarse (level 0, 1x),
accepted (level 1, 1x) and refined (level 1, 2x) maps on the sealed L1a axis window; compute
the sweep-v3 descriptors (Koch rho, HEMP-like flag); compare cusp count, cusp positions, wall
|B| and rho with the L1a record. The assessment evaluates the binding integrity gates, the
predeclared confirmation gates (b) and (c), the reported estimands, and emits the dataset,
CSV, gates and campaign result.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import canonical_bytes, semantic_sha256, strict_json_file
from cft_revival.fem_reference import ResourceBlockedError

from experiments.cusp_topology_search_v3_1 import experiment as cts_experiment
from experiments.cusp_topology_search_v3_1 import topology as topology_module
from experiments.cusp_topology_search_v3_1.topology import ChannelGeometry, TopologyPolicy, TracingGrid
from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.l1a_geometry_sweep_v3 import descriptors as descriptor_module
from experiments.orbit_wall_loss_geometry_screening_v1.designs import field_pipeline_source_files, field_pipeline_source_sha256

from . import designs as design_module
from . import p2_fields
from .designs import DESIGN_SETS, SET_HEMP, DesignSpec

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
DESIGN_AUTHORITIES_PATH = EXPERIMENT / "design-authorities.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"
CTS_PROTOCOL_PATH = MODERN / "experiments" / "cusp_topology_search_v3_1" / "protocol.json"

VERSION_TAG = "cft-revival.l1b-hemp-confirmation-v1-1"
CLASSIFICATION = "P2_MATERIAL_AWARE_FIELD_CONFIRMATION_NOT_HARDWARE_VALID"
TOPOLOGY_LABEL = "SCREENING_P2_MATERIAL_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
EXPERIMENT_CODE_FILES = ("__init__.py", "designs.py", "experiment.py", "p2_fields.py", "run.py")
MAP_ROLES = ("coarse", "accepted", "refined")
VERDICTS = ("CONFIRMED", "PARTIALLY_CONFIRMED", "DISCONFIRMED")
CSV_COLUMNS = (
    "design_id",
    "stage_count",
    "x_w",
    "wall_radius_m",
    "stage_pitch_m",
    "source_strength_scale",
    "l1a_axis_null_count",
    "p2_axis_null_count",
    "l1a_wall_cusp_count",
    "p2_wall_cusp_count",
    "l1a_cell_count",
    "p2_cell_count",
    "count_agreement_strict",
    "count_agreement_boundary_tolerant",
    "cusp_bijection",
    "matched_cusp_count",
    "max_cusp_shift_m",
    "max_cusp_shift_over_tolerance",
    "cusp_position_tolerance_m",
    "max_channel_axis_null_shift_m",
    "peak_wall_b_ratio_p2_over_l1a",
    "peak_wall_b_ratio_unscaled",
    "axis_peak_b_ratio_p2_over_l1a",
    "cusp_wall_b_ratios",
    "l1a_min_rho_conservative",
    "p2_min_rho_conservative",
    "rho_conservative_ratios",
    "l1a_hemp_like",
    "p2_hemp_like",
    "p2_level0_dofs",
    "p2_level1_dofs",
    "p2_all_levels_converged",
    "p2_discretisation_max_cusp_shift_m",
    "sampling_stable",
    "p2_total_seconds",
    "peak_rss_bytes",
)


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/1.0.0"


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    if value["classification"] != CLASSIFICATION:
        raise ValueError("protocol classification differs from the experiment constant")
    if int(value["design_sets"][SET_HEMP]["design_count"]) != 15:
        raise ValueError("the v1 protocol confirms exactly the 15 HEMP-like designs")
    if int(value["p2"]["adaptivity"]["levels"]) != 2 or int(value["execution"]["max_design_workers"]) != 1:
        raise ValueError("the v1 protocol is a two-level, one-design-at-a-time campaign")
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
    """Imported experiment modules and packages whose bytes determine the P2 field and the topology."""

    cts = MODERN / "experiments" / "cusp_topology_search_v3_1"
    v3 = MODERN / "experiments" / "l1a_geometry_sweep_v3"
    files = [
        cts / "topology.py",
        cts / "experiment.py",
        cts / "fields.py",
        cts / "catalogue.py",
        cts / "protocol.json",
        v3 / "designs.py",
        v3 / "descriptors.py",
        v3 / "catalogue.py",
        v3 / "experiment.py",
        v3 / "sampling.py",
        v3 / "protocol.json",
        v3 / "design-authorities.json",
        MODERN / "experiments" / "cft_topology_characterization_v1" / "experiment.py",
        MODERN / "experiments" / "cft_topology_characterization_v1" / "protocol.json",
        MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "designs.py",
        MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "protocol.json",
        MODERN / "experiments" / "cft_orbit_wall_loss_v4" / "adapter.py",
    ]
    import cft_revival.coupling as coupling_package
    import cft_revival.experiment_runtime as runtime_package
    import cft_revival.fem_reference as fem_package
    import cft_revival.orbit_mc as orbit_mc_package

    for package in (coupling_package, runtime_package, fem_package):
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
    if execution["parallel_designs"] or int(execution["max_design_workers"]) != 1:
        raise ValueError("the confirmation runs one P2 design at a time")
    return 1


def run_stage(tasks: Sequence[Mapping[str, Any]], function: Callable[[Mapping[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    """Sequential, in-process (one P2 design at a time; the RSS of this process is the measured RSS)."""

    return [function(task) for task in tasks]


# --------------------------------------------------------------------------
# Design authorities (no solving)
# --------------------------------------------------------------------------


def build_design_authorities(value: Mapping[str, Any], plan: CampaignPlan) -> dict[str, Any]:
    specs = specs_for_plan(value, plan)
    designs = [{**design_module.design_identity_without_solving(spec), "key": spec.key} for spec in specs]
    return {
        "schema_version": schema("design-authorities"),
        "plan_kind": plan.kind,
        "design_count": len(designs),
        "set_counts": {SET_HEMP: len(designs)},
        "l1a_wall_cusp_count_histogram": _histogram([item["l1a_wall_cusp_count"] for item in designs]),
        "designs": designs,
    }


def mesh_preflight(value: Mapping[str, Any], plan: CampaignPlan, budget: p2_fields.RamBudget) -> dict[str, Any]:
    """Whole-set level-0 mesh preflight (no solve): angle gate, DOF cap and allocation for every design."""

    rows = []
    for spec in specs_for_plan(value, plan):
        case = design_module.rebuild_case(spec.design_id)
        report = p2_fields.mesh_preflight_for_geometry(case.geometry, value, budget)
        rows.append({"key": spec.key, "design_id": spec.design_id, **report})
    return {
        "design_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "failed_designs": [row["design_id"] for row in rows if not row["passed"]],
        "minimum_angle_deg": min(row["minimum_angle_deg"] for row in rows) if rows else None,
        "max_level1_red_closure_p2_dof_upper_bound": max(row["level1_red_closure_p2_dof_upper_bound"] for row in rows) if rows else None,
        "designs_with_elements_below_10deg": [row["design_id"] for row in rows if row["sliver"]["elements_below_threshold"] > 0],
        "all_passed": bool(rows) and all(row["passed"] for row in rows),
        "designs": rows,
    }


# --------------------------------------------------------------------------
# Comparison with the sealed L1a record
# --------------------------------------------------------------------------


def cusp_position_tolerance_m(wall_radius_m: float, value: Mapping[str, Any]) -> float:
    """max(r_w / bore_elements, L1a dz): one level-0 P2 bore element, never below the L1a axial step."""

    return max(float(wall_radius_m) / int(value["p2"]["mesh"]["bore_elements"]), float(value["comparison"]["l1a_dz_m"]))


def _near_boundary(z: float, geometry: ChannelGeometry, tolerance_m: float) -> bool:
    return min(abs(z - geometry.straight_z_min_m), abs(z - geometry.straight_z_max_m)) <= tolerance_m


def _peak_wall_b(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [max(float(row["upstream_wall_max_b_t"]), float(row["downstream_wall_max_b_t"])) for row in rows if row.get("upstream_wall_max_b_t") is not None]
    return max(values) if values else None


def _peak_axis_b(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [max(float(row["upstream_axis_peak_t"]), float(row["downstream_axis_peak_t"])) for row in rows if row.get("upstream_axis_peak_t") is not None]
    return max(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator) / float(denominator)


def compare_to_l1a(
    reference: Mapping[str, Any],
    accepted: Mapping[str, Any],
    descriptors: Mapping[str, Any],
    geometry: ChannelGeometry,
    value: Mapping[str, Any],
    *,
    source_strength_scale: float,
) -> dict[str, Any]:
    """Cusp count, cusp positions, axis nulls, wall |B|, axis |B_z| and rho: P2 accepted map vs sealed L1a."""

    policy = policy_from(value)
    tolerance = cusp_position_tolerance_m(geometry.wall_radius_m, value)
    l1a_cusps = list(reference["wall_cusps"])
    p2_cusps = list(accepted["topology"]["wall_cusps"])
    match = cts_experiment._match_sorted([float(c["z_c_m"]) for c in l1a_cusps], [float(c["z_c_m"]) for c in p2_cusps], tolerance)
    l1a_by_z = {float(c["z_c_m"]): c for c in l1a_cusps}
    p2_by_z = {float(c["z_c_m"]): c for c in p2_cusps}
    l1a_rho = {row["cusp_id"]: row for row in reference["rho"]}
    p2_rho = {row["cusp_id"]: row for row in descriptors["cusps"]}
    pairs = []
    for pair in match["matched"]:
        left = l1a_by_z[pair["reference_z_m"]]
        right = p2_by_z[pair["observed_z_m"]]
        left_rho = l1a_rho.get(left["cusp_id"], {})
        right_rho = p2_rho.get(right["cusp_id"], {})
        pairs.append(
            {
                "l1a_cusp_id": left["cusp_id"],
                "p2_cusp_id": right["cusp_id"],
                "l1a_z_c_m": float(left["z_c_m"]),
                "p2_z_c_m": float(right["z_c_m"]),
                "shift_m": float(pair["difference_m"]),
                "shift_over_tolerance": float(pair["difference_m"]) / tolerance,
                "l1a_wall_b_t": float(left["wall_b_t"]),
                "p2_wall_b_t": float(right["wall_b_t"]),
                "p2_wall_b_unscaled_t": float(right["wall_b_t"]) / source_strength_scale,
                "wall_b_ratio_p2_over_l1a": _ratio(right["wall_b_t"], left["wall_b_t"]),
                "l1a_angle_to_wall_normal_deg": float(left["angle_to_wall_normal_deg"]),
                "p2_angle_to_wall_normal_deg": float(right["angle_to_wall_normal_deg"]),
                "l1a_rho_conservative": left_rho.get("rho_conservative"),
                "p2_rho_conservative": right_rho.get("rho_conservative"),
                "rho_conservative_ratio_p2_over_l1a": _ratio(right_rho.get("rho_conservative"), left_rho.get("rho_conservative")),
                "l1a_hemp_like_conservative": left_rho.get("hemp_like_conservative"),
                "p2_hemp_like_conservative": right_rho.get("hemp_like_conservative"),
                "l1a_boundary_ambiguous": bool(left["boundary_ambiguous"]),
                "p2_boundary_ambiguous": bool(right["boundary_ambiguous"]),
            }
        )
    unmatched = [("l1a", z) for z in match["unmatched_reference_z_m"]] + [("p2", z) for z in match["unmatched_observed_z_m"]]
    unmatched_rows = [{"side": side, "z_c_m": float(z), "near_straight_section_end": _near_boundary(float(z), geometry, policy.boundary_ambiguity_tolerance_m)} for side, z in unmatched]
    strict = len(p2_cusps) == len(l1a_cusps)
    boundary_tolerant = strict or all(row["near_straight_section_end"] for row in unmatched_rows)
    shifts = [pair["shift_m"] for pair in pairs]
    l1a_window = [float(reference["axis_window_m"][0]), float(reference["axis_window_m"][1])]
    l1a_nulls = [float(null["z_m"]) for null in reference["axis_nulls"]]
    p2_nulls = [float(null["z_m"]) for null in accepted["axis_nulls"]["nulls"]]
    null_match = cts_experiment._match_sorted(l1a_nulls, p2_nulls, tolerance)
    l1a_channel_nulls = [float(null["z_m"]) for null in reference["axis_nulls"] if null["zone"] == "channel"]
    p2_channel_nulls = [float(null["z_m"]) for null in accepted["axis_nulls"]["nulls"] if null["zone"] == "channel"]
    channel_null_match = cts_experiment._match_sorted(l1a_channel_nulls, p2_channel_nulls, tolerance)
    channel_null_shifts = [abs(a - b) for a, b in zip(sorted(l1a_channel_nulls), sorted(p2_channel_nulls))] if len(l1a_channel_nulls) == len(p2_channel_nulls) else None
    outside_l1a = [z for z in l1a_nulls if z not in l1a_channel_nulls]
    outside_p2 = [z for z in p2_nulls if z not in p2_channel_nulls]
    # Separatrix lean: axial distance from the generating axis null to the wall cusp, both maps.
    l1a_lean = [abs(float(c["z_c_m"]) - float(c["axis_null_z_m"])) for c in l1a_cusps]
    p2_lean = [abs(float(c["z_c_m"]) - float(c["axis_null_z_m"])) for c in p2_cusps]
    l1a_peak_wall = _peak_wall_b(reference["rho"])
    p2_peak_wall = _peak_wall_b(descriptors["cusps"])
    l1a_peak_axis = _peak_axis_b(reference["rho"])
    p2_peak_axis = _peak_axis_b(descriptors["cusps"])
    return {
        "cusp_position_tolerance_m": tolerance,
        "tolerance_rule": value["comparison"]["cusp_position_tolerance_rule"],
        "l1a_wall_cusp_count": int(reference["wall_cusp_count"]),
        "p2_wall_cusp_count": int(accepted["topology"]["wall_cusp_count"]),
        "l1a_cell_count": int(reference["cell_count"]),
        "p2_cell_count": int(accepted["topology"]["cell_count"]),
        "count_agreement_strict": bool(strict),
        "count_agreement_boundary_tolerant": bool(boundary_tolerant),
        "cell_count_agreement": int(reference["cell_count"]) == int(accepted["topology"]["cell_count"]),
        "cusp_match": {key: match[key] for key in ("reference_count", "observed_count", "unmatched_reference_z_m", "unmatched_observed_z_m", "max_difference_m", "bijection")},
        "matched_cusps": pairs,
        "unmatched_cusps": unmatched_rows,
        "matched_cusp_count": len(pairs),
        "max_cusp_shift_m": max(shifts) if shifts else None,
        "median_cusp_shift_m": statistics.median(shifts) if shifts else None,
        "max_cusp_shift_over_tolerance": (max(shifts) / tolerance) if shifts else None,
        "all_matched_within_tolerance": all(pair["shift_over_tolerance"] <= 1.0 for pair in pairs),
        "position_gate_passed": bool(match["bijection"] and pairs and all(pair["shift_over_tolerance"] <= 1.0 for pair in pairs)),
        "axis_window_m": l1a_window,
        "p2_axis_window_m": [float(accepted["axis_nulls"]["window_m"][0]), float(accepted["axis_nulls"]["window_m"][1])],
        "l1a_axis_null_count": len(l1a_nulls),
        "p2_axis_null_count": len(p2_nulls),
        "axis_null_match": {key: null_match[key] for key in ("reference_count", "observed_count", "unmatched_reference_z_m", "unmatched_observed_z_m", "max_difference_m", "bijection")},
        "channel_axis_null_match": {key: channel_null_match[key] for key in ("reference_count", "observed_count", "unmatched_reference_z_m", "unmatched_observed_z_m", "max_difference_m", "bijection")},
        "channel_axis_nulls": {
            "l1a_z_m": sorted(l1a_channel_nulls),
            "p2_z_m": sorted(p2_channel_nulls),
            "count_equal": len(l1a_channel_nulls) == len(p2_channel_nulls),
            "sorted_shifts_m": channel_null_shifts,
            "max_sorted_shift_m": max(channel_null_shifts) if channel_null_shifts else None,
            "note": "sorted pairing of the channel axis nulls regardless of the tolerance; the wall cusps are the gated quantity, the axis nulls are where the iron moves the separatrix root",
        },
        "separatrix_lean_m": {
            "l1a_axis_null_to_cusp": l1a_lean,
            "p2_axis_null_to_cusp": p2_lean,
            "l1a_max": max(l1a_lean) if l1a_lean else None,
            "p2_max": max(p2_lean) if p2_lean else None,
        },
        "outside_channel_axis_nulls": {
            "l1a_z_m": outside_l1a,
            "p2_z_m": outside_p2,
            "count_equal": len(outside_l1a) == len(outside_p2),
            "shifts_m": [abs(a - b) for a, b in zip(sorted(outside_l1a), sorted(outside_p2))] if len(outside_l1a) == len(outside_p2) else None,
            "note": "axis nulls outside the straight section (anode side / exit / downstream) sit in the un-cancelled end field where the iron yoke and poles act most strongly; they are reported, not gated",
        },
        "l1a_peak_wall_b_t": l1a_peak_wall,
        "p2_peak_wall_b_t": p2_peak_wall,
        "p2_peak_wall_b_unscaled_t": None if p2_peak_wall is None else p2_peak_wall / source_strength_scale,
        "peak_wall_b_ratio_p2_over_l1a": _ratio(p2_peak_wall, l1a_peak_wall),
        "peak_wall_b_ratio_unscaled": _ratio(None if p2_peak_wall is None else p2_peak_wall / source_strength_scale, l1a_peak_wall),
        "l1a_axis_peak_b_t": l1a_peak_axis,
        "p2_axis_peak_b_t": p2_peak_axis,
        "axis_peak_b_ratio_p2_over_l1a": _ratio(p2_peak_axis, l1a_peak_axis),
        "wall_b_ratio_band_descriptive": [0.5, 2.0],
        "peak_wall_b_ratio_in_band": None if _ratio(p2_peak_wall, l1a_peak_wall) is None else bool(0.5 <= _ratio(p2_peak_wall, l1a_peak_wall) <= 2.0),
        "l1a_min_rho_conservative": reference["min_rho_conservative"],
        "p2_min_rho_conservative": descriptors["min_rho_conservative"],
        "l1a_hemp_like_all_cusps": bool(reference["hemp_like_all_cusps"]),
        "p2_hemp_like_all_cusps": bool(descriptors["hemp_like_all_cusps"]),
        "hemp_like_preserved": bool(reference["hemp_like_all_cusps"]) and bool(descriptors["hemp_like_all_cusps"]),
        "source_strength_scale": float(source_strength_scale),
    }


# --------------------------------------------------------------------------
# Per-design worker
# --------------------------------------------------------------------------


def channel_geometry(case: sweep.BuiltCase) -> ChannelGeometry:
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


def window_contained(grid: TracingGrid, window: Sequence[float], policy: TopologyPolicy) -> bool:
    margin = policy.axis_window_margin_mesh_factor * grid.mesh_scale_m
    return bool(float(grid.z_m[0]) + margin <= float(window[0]) and float(window[1]) <= float(grid.z_m[-1]) - margin)


@dataclass(frozen=True)
class ResolvedDesign:
    spec: DesignSpec
    case: sweep.BuiltCase
    reference: dict[str, Any]
    solution: p2_fields.P2Solution
    grids: dict[str, TracingGrid]
    sampled: dict[str, p2_fields.SampledField]
    identity: dict[str, Any]
    evidence: dict[str, Any]
    solve_seconds: float


def resolve_design(spec: DesignSpec, value: Mapping[str, Any], budget: p2_fields.RamBudget) -> ResolvedDesign:
    """Rebuild with identity proof, load the sealed L1a record, solve two P2 levels, sample three maps."""

    started = time.perf_counter()
    case = design_module.rebuild_case(spec.design_id)
    reference = design_module.l1a_reference(spec.design_id)
    geometry = channel_geometry(case)
    if geometry.to_dict() != reference["geometry"]:
        raise ValueError(f"{spec.design_id}: rebuilt channel geometry differs from the sealed L1a record")
    if case.geometry_sha256 != reference["identity"]["geometry_sha256"] or case.case_sha256 != reference["identity"]["case_sha256"]:
        raise ValueError(f"{spec.design_id}: rebuilt hashes differ from the sealed L1a record identity")
    scale = float(reference["source_strength_scale"])
    solution = p2_fields.solve_two_level(case.geometry, value, budget)
    refinement = int(value["p2"]["sampling"]["refinement"])
    nodes_1x = p2_fields.sampling_nodes(value, geometry.wall_radius_m, solution.domain, refinement=1)
    nodes_2x = p2_fields.sampling_nodes(value, geometry.wall_radius_m, solution.domain, refinement=refinement)
    sampled = {
        "coarse": p2_fields.sample_regular_grid(solution.coarse, *nodes_1x, scale=scale),
        "accepted": p2_fields.sample_regular_grid(solution.accepted, *nodes_1x, scale=scale),
        "refined": p2_fields.sample_regular_grid(solution.accepted, *nodes_2x, scale=scale),
    }
    grids = {role: p2_fields.sampled_tracing_grid(field, geometry.wall_radius_m) for role, field in sampled.items()}
    base_identity = {
        "set_id": spec.set_id,
        "design_id": spec.design_id,
        "case_sha256": case.case_sha256,
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "sampling_design_id": case.design.design_id,
        "l1a_record_byte_sha256": reference["record_byte_sha256"],
        "l1a_accepted_field_identity_sha256": reference["identity"]["accepted_field_identity_sha256"],
        "p2_problem_id": solution.problem_id,
        "p2_geometry_sha256": solution.geometry_sha256,
        "p2_magnetics_sha256": solution.magnetics_sha256,
        "p2_level_run_sha256": [level.result.run_sha256 for level in solution.levels],
        "p2_level_mesh_sha256": [level.result.mesh.sha256 for level in solution.levels],
        "solver": "cft_revival.fem_reference.solve (two nested adaptive levels) + p2_fields.sample_regular_grid",
        "solver_controls": dict(value["p2"]["solver"]),
        "sampling": dict(value["p2"]["sampling"]),
        "source_strength_scale": scale,
    }
    identity = {
        **base_identity,
        **{
            f"{role}_field_identity_sha256": p2_fields.field_identity({**base_identity, "role": role, "level": (0 if role == "coarse" else len(solution.levels) - 1), "refinement": (refinement if role == "refined" else 1)})
            for role in MAP_ROLES
        },
    }
    evidence = {
        "identity_proven": True,
        "identity_basis": "case rebuilt from the preregistered sweep-v3 Sobol design; geometry / source / config / case hashes equal the sealed sweep-v3 design authorities and the sealed L1a record; L1a record bytes equal the sweep-v3 manifest",
        "design_values": dict(reference["design_values"]),
        "sampling_provenance": case.design.provenance,
        "derived_geometry": {key: case.derived[key] for key in ("stage_count", "represented_stage_pitch_m", "magnet_inner_radius_m", "magnet_outer_radius_m", "magnet_axial_thickness_m", "represented_exit_length_m", "represented_exit_outer_radius_m", "chamber_length_m", "x_w", "wall_radius_over_pitch")},
        "p2": solution.evidence(),
        "sampling": {role: field.report() for role, field in sampled.items()},
        "ram_budget": budget.to_dict(),
    }
    return ResolvedDesign(spec, case, reference, solution, grids, sampled, identity, evidence, time.perf_counter() - started)


def _nulls_converged(report: Mapping[str, Any]) -> bool:
    return cts_experiment._nulls_converged(report)


def design_gate_checks(record: Mapping[str, Any]) -> dict[str, bool]:
    maps = [record[role] for role in MAP_ROLES]
    p2 = record["evidence"]["p2"]
    budget = record["evidence"]["ram_budget"]
    return {
        "identity_proven": bool(record["evidence"]["identity_proven"]),
        "solver_converged_all_levels": bool(p2["all_levels_converged"] and p2["level_count"] == 2 and all(level["allocation_preflight"]["passed"] for level in p2["levels"])),
        "every_null_converged": all(_nulls_converged(item) for item in maps),
        "every_trace_terminates_cleanly": all(bool(item["all_traces_terminate_cleanly"]) for item in maps),
        "every_wall_trace_flux_consistent": all(bool(item["all_wall_traces_flux_consistent"]) for item in maps),
        "sampling_stability": bool(record["sampling_stability"]["stable"]),
        "axis_window_reproduced": bool(record["axis_window_reproduced"]),
        "ram_policy_respected": bool(p2["peak_rss_bytes"] <= budget["budget_bytes"]),
    }


def characterize_resolved(resolved: ResolvedDesign, value: Mapping[str, Any], *, keep_paths: bool) -> dict[str, Any]:
    """Definition-v3 topology of the three P2 maps on the L1a window, descriptors, stability, comparison (pure)."""

    policy = policy_from(value)
    tightness = float(value["definition_v3_import"]["minimum_certificate_dense_to_bound_ratio"])
    stability_tolerance = float(value["definition_v3_import"]["stability_tolerance_m"])
    geometry = channel_geometry(resolved.case)
    window = (float(resolved.reference["axis_window_m"][0]), float(resolved.reference["axis_window_m"][1]))
    contained = {role: window_contained(grid, window, policy) for role, grid in resolved.grids.items()}
    if not all(contained.values()):
        raise ValueError(f"{resolved.spec.design_id}: the P2 sampling grid does not contain the sealed L1a axis window with the v3.1 margin: {contained}")
    characterizations = {
        role: topology_module.characterize_map(
            resolved.grids[role],
            geometry,
            policy,
            source_identity_sha256=resolved.identity[f"{role}_field_identity_sha256"],
            minimum_certificate_tightness_ratio=tightness,
            keep_paths=bool(keep_paths and role == "accepted"),
            sweep_axis_bz_peaks_m=None,
            axis_window_m=window,
        )
        for role in MAP_ROLES
    }
    stage_count = int(resolved.case.derived["stage_count"])
    descriptors = {
        role: descriptor_module.design_descriptors(
            resolved.grids[role],
            geometry,
            characterizations[role],
            policy,
            source_identity_sha256=resolved.identity[f"{role}_field_identity_sha256"],
            minimum_certificate_tightness_ratio=tightness,
            stage_count=stage_count,
            with_profiles=(role == "accepted"),
        )
        for role in MAP_ROLES
    }
    x_m_inner = float(resolved.case.derived["x_m_inner"])
    for item in descriptors.values():
        item["x_m_inner"] = x_m_inner
    sampling_stability = topology_module.compare_resolutions(characterizations["accepted"], characterizations["refined"], stability_tolerance)
    discretisation = topology_module.compare_resolutions(characterizations["coarse"], characterizations["accepted"], stability_tolerance)
    discretisation["rho_sensitivity"] = descriptor_module.resolution_sensitivity(descriptors["coarse"], descriptors["accepted"])
    comparison = compare_to_l1a(resolved.reference, characterizations["accepted"], descriptors["accepted"], geometry, value, source_strength_scale=float(resolved.reference["source_strength_scale"]))
    payload = _plain(
        {
            "axis_window_m": [window[0], window[1]],
            "axis_window_reproduced": all(contained.values()) and all(list(item["axis_nulls"]["window_m"]) == [window[0], window[1]] for item in characterizations.values()),
            **characterizations,
            "descriptors": {**descriptors, "sampling_sensitivity": descriptor_module.resolution_sensitivity(descriptors["accepted"], descriptors["refined"])},
            "sampling_stability": sampling_stability,
            "p2_discretisation": discretisation,
            "comparison": comparison,
        }
    )
    payload["comparison_payload_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def run_design_task(task: Mapping[str, Any]) -> dict[str, Any]:
    spec = DesignSpec(**task["spec"])
    value = task["protocol"]
    budget = p2_fields.RamBudget(int(task["ram_budget"]["free_at_start_bytes"]), float(task["ram_budget"]["fraction"]), int(task["ram_budget"]["maximum_p2_dofs"]))
    started = time.perf_counter()
    base = {"key": spec.key, "set_id": spec.set_id, "design_id": spec.design_id, "ordinal": spec.ordinal, "representative": spec.representative, "label": TOPOLOGY_LABEL}
    try:
        resolved = resolve_design(spec, value, budget)
    except ResourceBlockedError as error:
        return _plain({**base, "status": "failed", "stage": "resolve", "resource_blocked": True, "reason": f"{type(error).__name__}: {error}", "timing_s": {"total": time.perf_counter() - started}})
    except Exception as error:  # recorded, never hidden
        return _plain({**base, "status": "failed", "stage": "resolve", "resource_blocked": False, "reason": f"{type(error).__name__}: {error}", "timing_s": {"total": time.perf_counter() - started}})
    characterize_started = time.perf_counter()
    try:
        topology = characterize_resolved(resolved, value, keep_paths=bool(task["keep_paths"]))
    except Exception as error:  # recorded, never hidden
        return _plain({**base, "status": "failed", "stage": "characterize", "resource_blocked": False, "reason": f"{type(error).__name__}: {error}", "identity": resolved.identity, "timing_s": {"solve": resolved.solve_seconds, "total": time.perf_counter() - started}})
    accepted_field = resolved.sampled["accepted"]
    record = {
        "schema_version": schema("design-record"),
        **base,
        "classification": CLASSIFICATION,
        "status": "resolved",
        "identity": resolved.identity,
        "evidence": resolved.evidence,
        "l1a_reference": resolved.reference,
        "geometry": channel_geometry(resolved.case).to_dict(),
        **topology,
        "timing_s": {"solve_and_sample": resolved.solve_seconds, "characterize": time.perf_counter() - characterize_started, "total": time.perf_counter() - started},
        "accepted_grid": {
            "r_m": list(accepted_field.r_m),
            "z_m": list(accepted_field.z_m),
            "psi_wb": [list(row) for row in accepted_field.psi_wb],
            "b_r_t": [list(row) for row in accepted_field.b_r_t],
            "b_z_t": [list(row) for row in accepted_field.b_z_t],
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
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"count": len(clean), "min": min(clean), "median": statistics.median(clean), "mean": statistics.fmean(clean), "max": max(clean)}


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in values:
        counts[str(int(item))] = counts.get(str(int(item)), 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: int(pair[0])))


def confirmation_gates(records: Sequence[Mapping[str, Any]], value: Mapping[str, Any]) -> dict[str, Any]:
    """GATE (b), GATE (c) and the reported (d) with their predeclared thresholds; verdict per the protocol rule."""

    declaration = value["gates"]["confirmation"]
    comparisons = [record["comparison"] for record in records]
    count = len(comparisons)
    fraction_bt = (sum(item["count_agreement_boundary_tolerant"] for item in comparisons) / count) if count else None
    fraction_strict = (sum(item["count_agreement_strict"] for item in comparisons) / count) if count else None
    shifts = [pair["shift_m"] for item in comparisons for pair in item["matched_cusps"]]
    normalised = [pair["shift_over_tolerance"] for item in comparisons for pair in item["matched_cusps"]]
    max_normalised = max(normalised) if normalised else None
    all_bijective = all(item["cusp_match"]["bijection"] for item in comparisons) if comparisons else False
    b_threshold = float(declaration["cusp_count_unchanged"]["pass_threshold"])
    c_threshold = float(declaration["cusp_position_shift"]["pass_threshold"])
    b_passed = bool(fraction_bt is not None and fraction_bt >= b_threshold)
    c_passed = bool(all_bijective and max_normalised is not None and max_normalised <= c_threshold)
    hemp_preserved = sum(item["hemp_like_preserved"] for item in comparisons)
    if b_passed and c_passed:
        verdict = "CONFIRMED"
    elif b_passed or c_passed:
        verdict = "PARTIALLY_CONFIRMED"
    else:
        verdict = "DISCONFIRMED"
    return {
        "cusp_count_unchanged": {
            "id": declaration["cusp_count_unchanged"]["id"],
            "fraction_boundary_tolerant": fraction_bt,
            "fraction_strict": fraction_strict,
            "agreeing_designs_boundary_tolerant": sum(item["count_agreement_boundary_tolerant"] for item in comparisons),
            "agreeing_designs_strict": sum(item["count_agreement_strict"] for item in comparisons),
            "disagreeing_designs": [record["design_id"] for record in records if not record["comparison"]["count_agreement_boundary_tolerant"]],
            "design_count": count,
            "pass_threshold": b_threshold,
            "comparator": declaration["cusp_count_unchanged"]["comparator"],
            "passed": b_passed,
        },
        "cusp_position_shift": {
            "id": declaration["cusp_position_shift"]["id"],
            "matched_cusp_count": len(shifts),
            "all_designs_bijective": all_bijective,
            "non_bijective_designs": [record["design_id"] for record in records if not record["comparison"]["cusp_match"]["bijection"]],
            "shift_m": _range(shifts),
            "shift_over_tolerance": _range(normalised),
            "max_shift_over_tolerance": max_normalised,
            "designs_exceeding_tolerance": [record["design_id"] for record in records if record["comparison"]["matched_cusps"] and not record["comparison"]["all_matched_within_tolerance"]],
            "tolerance_m": _range([item["cusp_position_tolerance_m"] for item in comparisons]),
            "pass_threshold": c_threshold,
            "comparator": declaration["cusp_position_shift"]["comparator"],
            "passed": c_passed,
        },
        "hemp_like_preserved": {
            "id": declaration["hemp_like_preserved"]["id"],
            "preserved_count": hemp_preserved,
            "design_count": count,
            "fraction": (hemp_preserved / count) if count else None,
            "lost_designs": [record["design_id"] for record in records if not record["comparison"]["hemp_like_preserved"]],
            "p2_min_rho_conservative": _range([item["p2_min_rho_conservative"] for item in comparisons]),
            "l1a_min_rho_conservative": _range([item["l1a_min_rho_conservative"] for item in comparisons]),
            "rho_conservative_ratio_p2_over_l1a": _range([pair["rho_conservative_ratio_p2_over_l1a"] for item in comparisons for pair in item["matched_cusps"]]),
            "wall_b_ratio_p2_over_l1a_per_cusp": _range([pair["wall_b_ratio_p2_over_l1a"] for item in comparisons for pair in item["matched_cusps"]]),
            "peak_wall_b_ratio_p2_over_l1a": _range([item["peak_wall_b_ratio_p2_over_l1a"] for item in comparisons]),
            "peak_wall_b_ratio_unscaled": _range([item["peak_wall_b_ratio_unscaled"] for item in comparisons]),
            "axis_peak_b_ratio_p2_over_l1a": _range([item["axis_peak_b_ratio_p2_over_l1a"] for item in comparisons]),
            "peak_wall_b_ratio_in_band_count": sum(1 for item in comparisons if item["peak_wall_b_ratio_in_band"]),
            "pass_threshold": None,
            "passed": None,
        },
        "verdict": verdict,
        "verdict_rule": declaration["verdict_rule"],
    }


def set_estimands(records: Sequence[Mapping[str, Any]], value: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = [record["comparison"] for record in records]
    p2 = [record["evidence"]["p2"] for record in records]
    return {
        "design_count": len(records),
        "confirmation": confirmation_gates(records, value),
        "l1a_wall_cusp_count_histogram": _histogram([item["l1a_wall_cusp_count"] for item in comparisons]),
        "p2_wall_cusp_count_histogram": _histogram([item["p2_wall_cusp_count"] for item in comparisons]),
        "l1a_axis_null_count_histogram": _histogram([item["l1a_axis_null_count"] for item in comparisons]),
        "p2_axis_null_count_histogram": _histogram([item["p2_axis_null_count"] for item in comparisons]),
        "axis_null_bijection_count": sum(item["axis_null_match"]["bijection"] for item in comparisons),
        "max_axis_null_shift_m": _range([item["axis_null_match"]["max_difference_m"] for item in comparisons]),
        "channel_axis_null_bijection_count": sum(item["channel_axis_null_match"]["bijection"] for item in comparisons),
        "channel_axis_null_shift_m": _range([item["channel_axis_null_match"]["max_difference_m"] for item in comparisons]),
        "channel_axis_null_count_equal_count": sum(item["channel_axis_nulls"]["count_equal"] for item in comparisons),
        "channel_axis_null_sorted_shift_m": _range([shift for item in comparisons for shift in (item["channel_axis_nulls"]["sorted_shifts_m"] or [])]),
        "separatrix_lean_l1a_m": _range([lean for item in comparisons for lean in item["separatrix_lean_m"]["l1a_axis_null_to_cusp"]]),
        "separatrix_lean_p2_m": _range([lean for item in comparisons for lean in item["separatrix_lean_m"]["p2_axis_null_to_cusp"]]),
        "outside_channel_axis_null_shift_m": _range([shift for item in comparisons for shift in (item["outside_channel_axis_nulls"]["shifts_m"] or [])]),
        "p2_discretisation_max_wall_intersection_shift_m": _range([record["p2_discretisation"]["max_wall_intersection_shift_m"] for record in records]),
        "p2_discretisation_max_axis_null_shift_m": _range([record["p2_discretisation"]["max_axis_null_shift_m"] for record in records]),
        "p2_discretisation_stable_count": sum(record["p2_discretisation"]["stable"] for record in records),
        "p2_discretisation_rho_sensitivity_max": _range([record["p2_discretisation"]["rho_sensitivity"].get("max_relative_rho_difference") for record in records]),
        "sampling_stable_count": sum(record["sampling_stability"]["stable"] for record in records),
        "sampling_max_wall_intersection_shift_m": _range([record["sampling_stability"]["max_wall_intersection_shift_m"] for record in records]),
        "p2_level_dofs": {"level_0": _range([item["levels"][0]["p2_dofs"] for item in p2]), "level_1": _range([item["levels"][-1]["p2_dofs"] for item in p2])},
        "p2_iterations": {"level_0": _range([item["levels"][0]["iterations"] for item in p2]), "level_1": _range([item["levels"][-1]["iterations"] for item in p2])},
        "p2_relative_true_residual_max": max(level["relative_true_residual_l2"] for item in p2 for level in item["levels"]) if p2 else None,
        "p2_total_seconds": _range([item["total_seconds"] for item in p2]),
        "p2_peak_rss_bytes": _range([item["peak_rss_bytes"] for item in p2]),
        "p2_wall_b3_over_b1": _range([record["descriptors"]["accepted"]["wall_harmonics"].get("b3_over_b1") for record in records if record["descriptors"]["accepted"]["wall_harmonics"].get("applies")]),
        "l1a_wall_b3_over_b1": _range([record["l1a_reference"]["wall_harmonics"].get("b3_over_b1") for record in records if record["l1a_reference"]["wall_harmonics"].get("applies")]),
        "angle_to_wall_normal_deg_p2": _range([pair["p2_angle_to_wall_normal_deg"] for item in comparisons for pair in item["matched_cusps"]]),
        "angle_to_wall_normal_deg_l1a": _range([pair["l1a_angle_to_wall_normal_deg"] for item in comparisons for pair in item["matched_cusps"]]),
    }


def agreement_row(record: Mapping[str, Any]) -> dict[str, Any]:
    comparison = record["comparison"]
    p2 = record["evidence"]["p2"]
    return {
        "design_id": record["design_id"],
        "stage_count": int(record["evidence"]["derived_geometry"]["stage_count"]),
        "x_w": float(record["evidence"]["derived_geometry"]["x_w"]),
        "l1a_wall_cusp_count": comparison["l1a_wall_cusp_count"],
        "p2_wall_cusp_count": comparison["p2_wall_cusp_count"],
        "l1a_cell_count": comparison["l1a_cell_count"],
        "p2_cell_count": comparison["p2_cell_count"],
        "count_agreement_strict": comparison["count_agreement_strict"],
        "count_agreement_boundary_tolerant": comparison["count_agreement_boundary_tolerant"],
        "cusp_bijection": comparison["cusp_match"]["bijection"],
        "max_cusp_shift_m": comparison["max_cusp_shift_m"],
        "max_cusp_shift_over_tolerance": comparison["max_cusp_shift_over_tolerance"],
        "cusp_position_tolerance_m": comparison["cusp_position_tolerance_m"],
        "max_channel_axis_null_shift_m": comparison["channel_axis_nulls"]["max_sorted_shift_m"],
        "peak_wall_b_ratio_p2_over_l1a": comparison["peak_wall_b_ratio_p2_over_l1a"],
        "cusp_wall_b_ratios": [pair["wall_b_ratio_p2_over_l1a"] for pair in comparison["matched_cusps"]],
        "l1a_min_rho_conservative": comparison["l1a_min_rho_conservative"],
        "p2_min_rho_conservative": comparison["p2_min_rho_conservative"],
        "p2_hemp_like": comparison["p2_hemp_like_all_cusps"],
        "p2_level1_dofs": p2["levels"][-1]["p2_dofs"],
        "p2_all_levels_converged": p2["all_levels_converged"],
        "p2_discretisation_max_cusp_shift_m": record["p2_discretisation"]["max_wall_intersection_shift_m"],
    }


def dataset_row(record: Mapping[str, Any]) -> dict[str, Any]:
    accepted = record["accepted"]
    topology = accepted["topology"]
    descriptors = record["descriptors"]["accepted"]
    p2 = record["evidence"]["p2"]
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
        "geometry": record["geometry"],
        "derived": dict(record["evidence"]["derived_geometry"]),
        "identity": {key: record["identity"][key] for key in ("case_sha256", "geometry_sha256", "l1a_record_byte_sha256", "l1a_accepted_field_identity_sha256", "accepted_field_identity_sha256", "coarse_field_identity_sha256", "refined_field_identity_sha256")},
        "p2": {
            "problem_id": p2["problem_id"],
            "domain": p2["domain"],
            "regions": p2["regions"],
            "levels": [{key: level[key] for key in ("level", "p2_dofs", "triangles", "converged", "iterations", "relative_true_residual_l2", "energy_action_relative", "backend", "solve_wall_seconds", "rss_after_solve_bytes", "mesh_sha256", "run_sha256")} for level in p2["levels"]],
            "all_levels_converged": p2["all_levels_converged"],
            "peak_rss_bytes": p2["peak_rss_bytes"],
            "total_seconds": p2["total_seconds"],
        },
        "sampling": record["evidence"]["sampling"],
        "axis_window_m": record["axis_window_m"],
        "p2_axis_nulls": [{"null_id": n["null_id"], "z_m": n["z_m"], "zone": n["zone"], "classification": n["classification"]} for n in accepted["axis_nulls"]["nulls"]],
        "p2_wall_cusps": [{key: cusp[key] for key in ("cusp_id", "axis_null_z_m", "z_c_m", "z_c_over_length", "wall_b_t", "wall_b_r_t", "angle_to_wall_normal_deg", "boundary_ambiguous")} for cusp in topology["wall_cusps"]],
        "p2_outside_intersections": [{key: row[key] for key in ("cusp_id", "z_c_m", "zone", "wall_b_t")} for row in topology["outside_intersections"]],
        "p2_cells": [{key: cell[key] for key in ("cell_id", "kind", "z_start_m", "z_end_m", "length_over_pitch", "wall_b_min_t", "wall_mirror_ratio", "axis_bz_peak_t", "axis_mirror_ratio")} for cell in topology["cells"]],
        "p2_rho": [{key: row[key] for key in ("cusp_id", "z_c_m", "wall_b_t", "upstream_axis_peak_t", "downstream_axis_peak_t", "rho_upstream", "rho_downstream", "rho_conservative", "rho_wall", "hemp_like_conservative", "cusp_is_wall_maximum")} for row in descriptors["cusps"]],
        "p2_wall_harmonics": {key: descriptors["wall_harmonics"].get(key) for key in ("applies", "b3_over_b1", "b5_over_b1", "fit_rms_over_max", "wall_b_r_max_abs_t")},
        "l1a": {key: record["l1a_reference"][key] for key in ("record_path", "record_byte_sha256", "axis_nulls", "axis_null_count", "wall_cusps", "wall_cusp_count", "cell_count", "cells", "rho", "min_rho_conservative", "hemp_like_all_cusps", "x_w", "grid", "wall_harmonics", "stability")},
        "comparison": record["comparison"],
        "sampling_stability": {key: record["sampling_stability"][key] for key in ("stable", "axis_null_count_equal", "wall_reaching_count_equal", "wall_cusp_count_equal", "max_axis_null_shift_m", "max_wall_intersection_shift_m")},
        "p2_discretisation": {key: record["p2_discretisation"][key] for key in ("stable", "axis_null_count_equal", "wall_reaching_count_equal", "wall_cusp_count_equal", "max_axis_null_shift_m", "max_wall_intersection_shift_m")},
        "p2_discretisation_rho_sensitivity": {key: record["p2_discretisation"]["rho_sensitivity"].get(key) for key in ("comparable", "max_relative_rho_difference", "hemp_like_flag_agrees")},
        "gate_checks": record["gate_checks"],
        "grid": {key: accepted["grid"][key] for key in ("radial_samples", "axial_samples", "dr_m", "dz_m", "radial_cells_across_bore", "max_b_t")},
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
        comparison = row["comparison"]
        p2 = row["p2"]
        writer.writerow(
            [
                row["design_id"],
                row["derived"]["stage_count"],
                _csv_value(row["derived"]["x_w"]),
                _csv_value(row["geometry"]["wall_radius_m"]),
                _csv_value(row["derived"]["represented_stage_pitch_m"]),
                _csv_value(comparison["source_strength_scale"]),
                comparison["l1a_axis_null_count"],
                comparison["p2_axis_null_count"],
                comparison["l1a_wall_cusp_count"],
                comparison["p2_wall_cusp_count"],
                comparison["l1a_cell_count"],
                comparison["p2_cell_count"],
                _csv_value(comparison["count_agreement_strict"]),
                _csv_value(comparison["count_agreement_boundary_tolerant"]),
                _csv_value(comparison["cusp_match"]["bijection"]),
                comparison["matched_cusp_count"],
                _csv_value(comparison["max_cusp_shift_m"]),
                _csv_value(comparison["max_cusp_shift_over_tolerance"]),
                _csv_value(comparison["cusp_position_tolerance_m"]),
                _csv_value(comparison["channel_axis_nulls"]["max_sorted_shift_m"]),
                _csv_value(comparison["peak_wall_b_ratio_p2_over_l1a"]),
                _csv_value(comparison["peak_wall_b_ratio_unscaled"]),
                _csv_value(comparison["axis_peak_b_ratio_p2_over_l1a"]),
                ";".join(_csv_value(pair["wall_b_ratio_p2_over_l1a"]) for pair in comparison["matched_cusps"]),
                _csv_value(comparison["l1a_min_rho_conservative"]),
                _csv_value(comparison["p2_min_rho_conservative"]),
                ";".join(_csv_value(pair["rho_conservative_ratio_p2_over_l1a"]) for pair in comparison["matched_cusps"]),
                _csv_value(comparison["l1a_hemp_like_all_cusps"]),
                _csv_value(comparison["p2_hemp_like_all_cusps"]),
                p2["levels"][0]["p2_dofs"],
                p2["levels"][-1]["p2_dofs"],
                _csv_value(p2["all_levels_converged"]),
                _csv_value(row["p2_discretisation"]["max_wall_intersection_shift_m"]),
                _csv_value(row["sampling_stability"]["stable"]),
                _csv_value(p2["total_seconds"]),
                p2["peak_rss_bytes"],
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
    preflight = record.get("mesh_preflight")
    try:
        evidentiary_count = len(evidentiary_plan(value).design_keys)
    except Exception:
        evidentiary_count = -1
    checks["mesh_preflight_covers_every_design_and_passed"] = bool(
        isinstance(preflight, dict)
        and preflight.get("all_passed") is True
        and preflight.get("design_count") == evidentiary_count
        and preflight.get("passed_count") == evidentiary_count
        and all(row["reject_below_angle_deg"] == float(value["p2"]["mesh"]["reject_below_angle_deg"]) for row in preflight.get("designs", []))
    )
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
    "solver_converged_all_levels",
    "every_null_converged",
    "every_trace_terminates_cleanly",
    "every_wall_trace_flux_consistent",
    "sampling_stability",
    "axis_window_reproduced",
    "ram_policy_respected",
)


def build_callbacks(value: Mapping[str, Any], plan: CampaignPlan, *, frozen: FrozenAuthority | None, collector: dict[str, Any]) -> RuntimeCallbacks:
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
        budget = p2_fields.ram_budget(value)
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
                "blas_threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
                "ram_budget": budget.to_dict(),
                "process_rss_at_start_bytes": p2_fields.current_process_rss_bytes(),
                "backend": "cft_revival.fem_reference.solve (numpy CSR PCG, CPU only; GPU not used) + p2_fields.sample_regular_grid + PsiBicubicField tracing",
            },
        )
        state.update({"binding": binding, "design_authorities": design_authorities, "budget": budget})
        collector["prebundle"] = {"design_count": design_authorities["design_count"], "ram_budget": budget.to_dict()}
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "classification": CLASSIFICATION,
            "topology_label": TOPOLOGY_LABEL,
            "design_count": design_authorities["design_count"],
            "experiment_code_sha256": binding["experiment_code_sha256"],
            "dependency_source_sha256": binding["dependency_source_sha256"],
            "field_pipeline_source_sha256": binding["field_pipeline_source_sha256"],
            "ram_budget_bytes": budget.budget_bytes,
        }

    def development(context: Any) -> Decision:
        started = time.perf_counter()
        specs = specs_for_plan(value, plan)
        budget: p2_fields.RamBudget = state["budget"]
        context.before_expensive("rebuild-solve-p2-sample-characterize-all-designs", kind="solver", details={"design_count": len(specs), "worker_pool_size": 1, "plan_kind": plan.kind, "solver": "cft_revival.fem_reference.solve", "ram_budget_bytes": budget.budget_bytes})
        keep_paths_sets = set(value["execution"]["keep_paths_for_sets"])
        tasks = [{"spec": spec.to_dict(), "protocol": dict(value), "keep_paths": bool(spec.set_id in keep_paths_sets or spec.representative), "ram_budget": budget.to_dict()} for spec in specs]
        stage_started = time.perf_counter()
        outcomes = run_stage(tasks, run_design_task)
        stage_wall = time.perf_counter() - stage_started
        authorities = {item["key"]: item for item in state["design_authorities"]["designs"]}
        records: dict[str, dict[str, Any]] = {}
        grids: dict[str, Any] = {}
        failures: list[dict[str, Any]] = []
        for task, outcome in zip(tasks, outcomes, strict=True):
            if outcome["key"] != f"{task['spec']['set_id']}:{task['spec']['design_id']}":
                raise RuntimeError("design results returned out of order")
            if outcome["status"] != "resolved":
                failures.append({"key": outcome["key"], "stage": outcome.get("stage"), "resource_blocked": outcome.get("resource_blocked"), "reason": outcome.get("reason")})
                continue
            authority = authorities[outcome["key"]]
            for hash_key in ("geometry_sha256", "source_sha256", "config_sha256", "case_sha256", "l1a_record_byte_sha256"):
                if authority[hash_key] != outcome["identity"].get(hash_key):
                    raise ValueError(f"{outcome['key']}: resolved {hash_key} differs from the design authority")
            records[outcome["key"]] = outcome
            grid = outcome.pop("accepted_grid")
            grids[outcome["key"]] = grid
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
        budget: p2_fields.RamBudget = state["budget"]
        ordered = [records[key] for key in plan.design_keys if key in records]
        replays = []
        for key in replay_keys(value, plan):
            if key not in records:
                continue
            spec = next(item for item in specs_for_plan(value, plan) if item.key == key)
            context.before_expensive("replay-rebuild-solve-p2-sample-characterize", kind="solver", details={"key": key})
            resolved = resolve_design(spec, value, budget)
            keep = bool(spec.set_id in set(value["execution"]["keep_paths_for_sets"]) or spec.representative)
            replay = characterize_resolved(resolved, value, keep_paths=keep)
            replay_grid = _plain({"r_m": list(resolved.sampled["accepted"].r_m), "z_m": list(resolved.sampled["accepted"].z_m), "psi_wb": [list(row) for row in resolved.sampled["accepted"].psi_wb], "b_r_t": [list(row) for row in resolved.sampled["accepted"].b_r_t], "b_z_t": [list(row) for row in resolved.sampled["accepted"].b_z_t]})
            replays.append(
                {
                    "key": key,
                    "worker_comparison_payload_sha256": records[key]["comparison_payload_sha256"],
                    "replay_comparison_payload_sha256": replay["comparison_payload_sha256"],
                    "field_identity_equal": resolved.identity["accepted_field_identity_sha256"] == records[key]["identity"]["accepted_field_identity_sha256"],
                    "p2_run_sha256_equal": resolved.identity["p2_level_run_sha256"] == records[key]["identity"]["p2_level_run_sha256"],
                    "accepted_grid_equal": replay_grid == state["grids"][key],
                    "bit_identical": replay["comparison_payload_sha256"] == records[key]["comparison_payload_sha256"],
                }
            )
        replay_passed = bool(replays) and all(item["bit_identical"] and item["field_identity_equal"] and item["accepted_grid_equal"] for item in replays)
        peak_rss = max([record["evidence"]["p2"]["peak_rss_bytes"] for record in ordered] + [p2_fields.current_process_rss_bytes()])
        per_design_gates = {key: record["gate_checks"] for key, record in records.items()}
        campaign_gates = {
            "all_declared_designs_resolved": not state["failures"] and len(records) == len(plan.design_keys),
            **{name: all(checks[name] for checks in per_design_gates.values()) for name in GATE_NAMES},
            "determinism_replay": replay_passed,
            "hash_bindings": True,
        }
        campaign_gates["ram_policy_respected"] = bool(campaign_gates["ram_policy_respected"] and peak_rss <= budget.budget_bytes)
        failing_designs = {name: sorted(key for key, checks in per_design_gates.items() if not checks[name]) for name in GATE_NAMES}
        gates_passed = all(campaign_gates.values())
        estimands = set_estimands(ordered, value)
        confirmation = estimands["confirmation"]
        rows = [dataset_row(record) for record in ordered]
        agreement_table = [agreement_row(record) for record in ordered]
        protocol_hash = semantic_sha256(value)
        headline = {
            "design_count": len(ordered),
            "verdict": confirmation["verdict"],
            "gate_b_cusp_count_unchanged": {key: confirmation["cusp_count_unchanged"][key] for key in ("fraction_boundary_tolerant", "fraction_strict", "agreeing_designs_boundary_tolerant", "agreeing_designs_strict", "disagreeing_designs", "pass_threshold", "passed")},
            "gate_c_cusp_position_shift": {key: confirmation["cusp_position_shift"][key] for key in ("matched_cusp_count", "all_designs_bijective", "non_bijective_designs", "shift_m", "shift_over_tolerance", "max_shift_over_tolerance", "designs_exceeding_tolerance", "tolerance_m", "pass_threshold", "passed")},
            "reported_d_hemp_like_preserved": {key: confirmation["hemp_like_preserved"][key] for key in ("preserved_count", "fraction", "lost_designs", "p2_min_rho_conservative", "l1a_min_rho_conservative", "rho_conservative_ratio_p2_over_l1a", "wall_b_ratio_p2_over_l1a_per_cusp", "peak_wall_b_ratio_p2_over_l1a", "peak_wall_b_ratio_unscaled", "axis_peak_b_ratio_p2_over_l1a", "peak_wall_b_ratio_in_band_count")},
            "axis_null_bijection_count": estimands["axis_null_bijection_count"],
            "max_axis_null_shift_m": estimands["max_axis_null_shift_m"],
            "channel_axis_null_bijection_count": estimands["channel_axis_null_bijection_count"],
            "channel_axis_null_shift_m": estimands["channel_axis_null_shift_m"],
            "channel_axis_null_count_equal_count": estimands["channel_axis_null_count_equal_count"],
            "channel_axis_null_sorted_shift_m": estimands["channel_axis_null_sorted_shift_m"],
            "separatrix_lean_l1a_m": estimands["separatrix_lean_l1a_m"],
            "separatrix_lean_p2_m": estimands["separatrix_lean_p2_m"],
            "outside_channel_axis_null_shift_m": estimands["outside_channel_axis_null_shift_m"],
            "p2_discretisation_max_wall_intersection_shift_m": estimands["p2_discretisation_max_wall_intersection_shift_m"],
            "p2_discretisation_stable_count": estimands["p2_discretisation_stable_count"],
            "sampling_stable_count": estimands["sampling_stable_count"],
            "p2_level_dofs": estimands["p2_level_dofs"],
            "p2_relative_true_residual_max": estimands["p2_relative_true_residual_max"],
            "p2_total_seconds": estimands["p2_total_seconds"],
            "peak_rss_bytes": peak_rss,
            "ram_budget_bytes": budget.budget_bytes,
            "ram_budget_fraction_used": (peak_rss / budget.budget_bytes) if budget.budget_bytes else None,
        }
        dataset = {
            "schema_version": schema("confirmation-dataset"),
            "experiment_id": value["experiment_id"],
            "classification": CLASSIFICATION,
            "topology_label": TOPOLOGY_LABEL,
            "classification_statement": value["classification_statement"],
            "claim_boundary": value["claim_boundary"],
            "plan_kind": plan.kind,
            "evidentiary": plan.kind == "evidentiary",
            "protocol_semantic_sha256": protocol_hash,
            "experiment_code_sha256": state["binding"]["experiment_code_sha256"],
            "dependency_source_sha256": state["binding"]["dependency_source_sha256"],
            "field_pipeline_source_sha256": state["binding"]["field_pipeline_source_sha256"],
            "sealed_sources": state["binding"]["sealed_sources"],
            "design_count": len(rows),
            "designs": rows,
            "agreement_table": agreement_table,
            "estimands": estimands,
            "headline": headline,
            "gates": {"campaign": campaign_gates, "confirmation": confirmation, "failing_designs": failing_designs, "passed": gates_passed},
        }
        gates_record = {
            "schema_version": schema("gates"),
            "binding": plan.binding_gates,
            "campaign": campaign_gates,
            "confirmation": confirmation,
            "hash_bindings_note": "verified against the frozen authorities in the prebundle" if frozen is not None else "shakedown: recorded, no frozen authority to compare",
            "failing_designs": failing_designs,
            "per_design": per_design_gates,
            "replays": replays,
            "peak_rss_bytes": peak_rss,
            "ram_budget": budget.to_dict(),
            "passed": gates_passed,
            "design_count": len(records),
            "definitions": value["gates"],
        }
        context.write_json("artifacts/gates.json", gates_record)
        context.write_json("artifacts/confirmation-dataset.json", dataset)
        context.write_blob("artifacts/confirmation-dataset.csv", dataset_csv(rows))
        if gates_passed and plan.kind == "evidentiary":
            status = f"accepted_l1b_confirmation_{confirmation['verdict'].lower()}"
        elif gates_passed:
            status = "shakedown_passed"
        else:
            status = "gates_failed"
        campaign_result = {
            "schema_version": schema("campaign-result"),
            "experiment_id": value["experiment_id"],
            "classification": CLASSIFICATION,
            "topology_label": TOPOLOGY_LABEL,
            "plan_kind": plan.kind,
            "evidentiary": plan.kind == "evidentiary",
            "status": status,
            "verdict": confirmation["verdict"],
            "gates_passed": gates_passed,
            "campaign_gates": campaign_gates,
            "confirmation_gates": {key: confirmation[key]["passed"] for key in ("cusp_count_unchanged", "cusp_position_shift")},
            "design_count": len(records),
            "headline": headline,
            "agreement_table": agreement_table,
            "paper_admission": value["claim_boundary"]["paper_admission"],
            "execution_mode": {"worker_pool_size": 1, "stage_wall_s": state["stage_wall_s"], "assessment_wall_s": time.perf_counter() - started},
            "protocol_semantic_sha256": protocol_hash,
        }
        context.write_json("artifacts/campaign-result.json", campaign_result)
        collector["assessment"] = {"headline": headline, "gates": campaign_gates, "confirmation": confirmation, "failing_designs": failing_designs, "status": status, "replays": replays, "estimands": estimands, "agreement_table": agreement_table}
        return Decision(bool(gates_passed), {"status": status, "verdict": confirmation["verdict"], "design_count": len(records), "gates": campaign_gates})

    return RuntimeCallbacks(prebundle=prebundle, development=development, assessment=assessment)
