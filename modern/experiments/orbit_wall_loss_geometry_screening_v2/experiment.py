"""Wall-loss-vs-geometry screening v2: catalogue cells, scrambled Sobol, two-stage allocation.

Classification ``SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`` for the 96 sweep-v2 designs
(L1a fields, not P2-qualified) plus one ``P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN`` row.

Mechanics follow the accepted v1 screening (``experiments.orbit_wall_loss_geometry_screening_v1``)
and the v4 template (``experiments.cft_orbit_wall_loss_v4``): one :class:`CampaignPlan` drives
the evidentiary campaign and the disclosed NON-EVIDENTIARY shakedown; the shakedown must pass
on real fields before ``prepare`` freezes the authorities; one detached execution publishes
through the shared :class:`ExperimentRuntime`. Generic pieces are imported with attribution.

Case structure (new in v2). One orbit_mc case is one (design, catalogue cell, block of
``stage1_points_per_stratum`` scrambled-Sobol indices per stratum) at one time step:

* ``stage1``   block 0 of every cell (128 launches; frozen authority);
* ``stage2b1..3`` blocks 1..3 of the cells the frozen allocation rule tops up (128 each;
  rule evaluated by code inside the worker from the cell's stage-1 counts);
* ``control``  a frozen-seed subset of one eighth of the cell's final launches at 2N.

Every case therefore has 128, 64 or 16 launches. This matters: orbit_mc v1.7's artifact
validator requires ``lower <= p <= upper`` verbatim, and ``wilson_interval(0, n).lower`` is a
positive round-off for many ``n`` (384 among them) while ``wilson_interval(n, n).upper`` is
``1 - ulp`` for others (512, 640, ...); a zero-count category (timeouts are always zero) at such
an ``n`` would abort sealing. 128, 64 and 16 are exact at both ends (checked by the tests).

Per design all cells run inside one worker task (stage 1 -> rule -> stage 2 -> control ->
flags -> sealing), so designs are independent. The main process replays the allocation rule and
the control selection from the endpoint terminations (``allocation_rule_replay`` gate), binds the
cells to the catalogue (``catalogue_binding`` gate) and evaluates the pooled 2N control gate.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
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
    strict_json_loads,
)
from cft_revival.orbit_mc import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    ElectronLaunch,
    EstimatorPolicy,
    OrbitConfig,
    Termination,
    analytic_magnetic_bottle,
    backend_parity,
    frozen_batch_manifest,
    timestep_convergence,
    uniform_b_helix,
    varying_e_convergence,
    wall_event_accuracy,
    wilson_interval,
)
from cft_revival.orbit_mc.artifacts import content_hash
from cft_revival.orbit_mc.integrator import integrate_orbit

# Reused v4 mechanics (accepted campaign template), imported with attribution.
from experiments.cft_orbit_wall_loss_v4.experiment import (
    ValidatorLedger,
    _decode_runtime_tags,
    _final_velocity_equals_event_velocity,
    _plain,
    estimator_identity,
    launch_records,
    orbit_mc_contract_report as _v4_orbit_mc_contract_report,
    orbit_mc_source_files,
    orbit_mc_source_sha256,
    result_diagnostics,
    result_record,
    run_case_export,
    run_case_integration,
    run_stage,
)

from . import cells as cell_module
from . import designs as design_module
from .cells import (
    CatalogueBinding,
    LaunchCell,
    allocation_decision,
    build_launches,
    candidate_sha256,
    catalogue_entry,
    cell_counts_from_terminations,
    cell_id_of_key,
    control_selection,
    design_cells,
    design_pooled,
    key_cell_index,
    key_index,
    key_of_launch,
    load_bound_catalogue,
    pooled_cell_row,
    strata,
    stratum_seed,
)
from .consumer import consume_handoff, consume_v4_export, load_v1_dataset
from .designs import (
    LABEL_P2,
    LABEL_SWEEP,
    P2_DESIGN_ID,
    SET_P2,
    SET_SWEEP,
    BoundDesign,
    SweepBinding,
    bind_p2_design,
    bind_sweep_design,
    field_pipeline_source_sha256,
    load_sweep_binding,
    orbit_config_for_design,
    resolve_design,
    resolve_p2_design,
)

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
DESIGN_AUTHORITIES_PATH = EXPERIMENT / "design-authorities.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"

VERSION_TAG = "cft-revival.orbit-wall-loss-geometry-screening-v2"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
STAGE1 = "stage1"
CONTROL = "control"
NUMERICAL_FAILURES = (
    Termination.STEP_LIMIT,
    Termination.NONFINITE_STATE,
    Termination.EXTREME_RELATIVITY,
    Termination.FIELD_FAILURE,
    Termination.INITIAL_STATE_INVALID,
)
TIMEOUTS = (Termination.TIME_TIMEOUT, Termination.PATH_TIMEOUT)
ESCAPE_TOLERANCE_M = 1.0e-8
POSITION_CLASSES = ("anode_side", "interior", "exit_side", "unbounded")
AUTHORITY_COMPARED_KEYS = (
    "runtime_launch_payload_byte_sha256", "runtime_batch_payload_byte_sha256", "orbit_launches_sha256",
    "batch_manifest_sha256", "estimator_sha256", "launch_keys_sha256", "launch_count", "case_authority_sha256",
)


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/1.0.0"


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    if value["classification"] != CLASSIFICATION:
        raise ValueError("protocol classification must be the screening label")
    return value


def stage2_stage(block: int) -> str:
    if block < 1:
        raise ValueError("stage-2 blocks start at 1")
    return f"stage2b{block}"


def stage_block(stage: str) -> int:
    if stage == STAGE1:
        return 0
    if stage.startswith("stage2b"):
        return int(stage[len("stage2b"):])
    raise ValueError(f"{stage} is not an N-step block stage")


def stage_timestep(stage: str) -> str:
    if stage == CONTROL:
        return "2N"
    stage_block(stage)
    return "N"


def wilson_exact_at_ends(n: int) -> bool:
    """True iff orbit_mc's ordering check ``lower <= p <= upper`` holds at k = 0 and k = n for this case size."""

    low = wilson_interval(0, n)
    high = wilson_interval(n, n)
    return low.lower <= low.probability and high.upper >= high.probability


# --------------------------------------------------------------------------
# orbit_mc + field pipeline + experiment code binding
# --------------------------------------------------------------------------


def orbit_mc_contract_report(value: Mapping[str, Any]) -> dict[str, Any]:
    return _v4_orbit_mc_contract_report(value)


def require_orbit_mc_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    report = orbit_mc_contract_report(value)
    if not report["matches"]:
        raise ValueError(
            "orbit_mc contract (package version / schema versions) differs from protocol: "
            f"expected {report['expected']}, observed {report['observed']}"
        )
    return report


EXPERIMENT_CODE_FILES = ("cells.py", "consumer.py", "designs.py", "experiment.py", "run.py", "sobol.py", "__init__.py")


def experiment_code_sha256() -> str:
    """SHA-256 over the LF bytes of this experiment's own code."""

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
    contract = require_orbit_mc_contract(value)
    catalogue = load_bound_catalogue(value["cusp_cell_catalogue"])
    return {
        "orbit_mc": contract,
        "field_pipeline_source_sha256": field_pipeline_source_sha256(),
        "field_pipeline_source_files": [path.relative_to(MODERN).as_posix() for path in design_module.field_pipeline_source_files()],
        "experiment_code_sha256": experiment_code_sha256(),
        "experiment_code_files": list(EXPERIMENT_CODE_FILES),
        "catalogue_file_sha256": catalogue.file_sha256,
        "catalogue_manifest_file_sha256": catalogue.manifest_file_sha256,
        "v1_reused_modules": [
            "experiments.orbit_wall_loss_geometry_screening_v1.designs",
            "experiments.orbit_wall_loss_geometry_screening_v1.consumer",
            "experiments.cft_orbit_wall_loss_v4.experiment",
            "experiments.cft_orbit_wall_loss_v4.adapter",
            "experiments.cusp_topology_search_v3_1.catalogue",
        ],
    }


# --------------------------------------------------------------------------
# Campaign plans
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPlan:
    kind: str
    campaign_id_prefix: str
    seed_namespace: str
    control_seed_namespace: str
    design_keys: tuple[str, ...]
    stage1_points_per_stratum: int
    stage2_points_per_stratum: int
    wilson_width_threshold: float
    control_fraction: float
    batch_size: int
    partial_checkpoint_prefix_count: int
    control_partial_checkpoint_prefix_count: int
    binding_gates: bool

    def __post_init__(self) -> None:
        if self.kind not in ("evidentiary", "shakedown"):
            raise ValueError("unknown campaign plan kind")
        for name in ("campaign_id_prefix", "seed_namespace", "control_seed_namespace"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if ":" in self.campaign_id_prefix:
            raise ValueError("campaign_id_prefix must be colon-free")
        if self.seed_namespace == self.control_seed_namespace:
            raise ValueError("control seeds must live in their own namespace")
        if len(set(self.design_keys)) != len(self.design_keys) or not self.design_keys:
            raise ValueError("plan designs must be unique and non-empty")
        if not 0 < self.stage1_points_per_stratum < self.stage2_points_per_stratum:
            raise ValueError("stage points must satisfy 0 < stage1 < stage2")
        if self.stage2_points_per_stratum % self.stage1_points_per_stratum:
            raise ValueError("stage-2 points must be a whole number of stage-1 blocks")
        if not 0.0 < self.wilson_width_threshold < 1.0 or not 0.0 < self.control_fraction <= 1.0:
            raise ValueError("threshold and control fraction must lie in (0, 1)")
        if not 0 < self.partial_checkpoint_prefix_count < self.batch_size:
            raise ValueError("partial prefix must lie strictly inside batch 0")
        if not 0 < self.control_partial_checkpoint_prefix_count < self.batch_size:
            raise ValueError("control partial prefix must lie strictly inside batch 0")

    @property
    def block_count(self) -> int:
        return self.stage2_points_per_stratum // self.stage1_points_per_stratum

    def block_range(self, block: int) -> tuple[int, int]:
        if not 0 <= block < self.block_count:
            raise ValueError("block outside the plan")
        return (block * self.stage1_points_per_stratum, (block + 1) * self.stage1_points_per_stratum)

    def launches_per_block(self, value: Mapping[str, Any]) -> int:
        return self.stage1_points_per_stratum * int(value["launches"]["strata_per_cell"])

    def control_count(self, final_launches: int) -> int:
        count = final_launches * self.control_fraction
        if abs(count - round(count)) > 1.0e-9 or round(count) < 1:
            raise ValueError("control fraction must give a whole positive count per cell")
        return int(round(count))

    def launch_rule(self, value: Mapping[str, Any]) -> dict[str, Any]:
        rule = dict(value["launches"])
        rule["stage1_points_per_stratum"] = self.stage1_points_per_stratum
        rule["stage2_points_per_stratum"] = self.stage2_points_per_stratum
        rule["stage1_launches_per_cell"] = self.launches_per_block(value)
        rule["final_launches_per_topped_up_cell"] = self.stage2_points_per_stratum * int(rule["strata_per_cell"])
        rule["batch_size"] = self.batch_size
        return rule

    def allocation_rule(self, value: Mapping[str, Any]) -> dict[str, Any]:
        rule = dict(value["allocation"])
        rule["wilson_width_threshold"] = self.wilson_width_threshold
        rule["stage1_points_per_stratum"] = self.stage1_points_per_stratum
        rule["stage2_points_per_stratum"] = self.stage2_points_per_stratum
        rule["stage1_launches_per_cell"] = self.launches_per_block(value)
        rule["stage2_launches_per_cell"] = self.stage2_points_per_stratum * int(value["launches"]["strata_per_cell"])
        rule["strata_per_cell"] = int(value["launches"]["strata_per_cell"])
        return rule

    def case_sizes(self, value: Mapping[str, Any]) -> dict[str, int]:
        block = self.launches_per_block(value)
        return {
            "block": block,
            "control_of_stage1_cell": self.control_count(block),
            "control_of_topped_up_cell": self.control_count(block * self.block_count),
        }


def design_keys(value: Mapping[str, Any]) -> tuple[str, ...]:
    declaration = value["designs"]
    keys = list(declaration["sweep_case_ids"])
    if len(set(keys)) != len(keys) or len(keys) != int(declaration["sweep_design_count"]):
        raise ValueError("sweep design ids are not unique or differ from the declared count")
    if declaration["p2_design"]["included"]:
        keys.append(declaration["p2_design"]["design_key"])
    return tuple(sorted(keys))


def design_set(value: Mapping[str, Any], design_key: str) -> str:
    if design_key in value["designs"]["sweep_case_ids"]:
        return SET_SWEEP
    if value["designs"]["p2_design"]["included"] and design_key == value["designs"]["p2_design"]["design_key"]:
        return SET_P2
    raise ValueError(f"{design_key} is not a declared design")


def evidentiary_plan(value: Mapping[str, Any]) -> CampaignPlan:
    launches = value["launches"]
    plan = CampaignPlan(
        kind="evidentiary",
        campaign_id_prefix=launches["campaign_id_prefix"],
        seed_namespace=launches["seed_namespace"],
        control_seed_namespace=value["control"]["seed_namespace"],
        design_keys=design_keys(value),
        stage1_points_per_stratum=int(launches["stage1_points_per_stratum"]),
        stage2_points_per_stratum=int(launches["stage2_points_per_stratum"]),
        wilson_width_threshold=float(value["allocation"]["wilson_width_threshold"]),
        control_fraction=float(value["control"]["fraction_per_cell"]),
        batch_size=int(launches["batch_size"]),
        partial_checkpoint_prefix_count=int(value["execution"]["partial_checkpoint_prefix_count"]),
        control_partial_checkpoint_prefix_count=int(value["execution"]["control_partial_checkpoint_prefix_count"]),
        binding_gates=True,
    )
    require_safe_case_sizes(value, plan)
    return plan


def shakedown_plan(value: Mapping[str, Any]) -> CampaignPlan:
    declaration = value["shakedown"]
    if declaration["evidentiary"] is not False or declaration["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown must be declared non-evidentiary")
    keys = list(declaration["design_case_ids"])
    if declaration["include_p2_design"]:
        if not value["designs"]["p2_design"]["included"]:
            raise ValueError("shakedown includes the P2 design but the campaign does not")
        keys.append(value["designs"]["p2_design"]["design_key"])
    if any(key not in design_keys(value) for key in keys):
        raise ValueError("shakedown designs must be declared designs")
    plan = CampaignPlan(
        kind="shakedown",
        campaign_id_prefix=declaration["campaign_id_prefix"],
        seed_namespace=declaration["seed_namespace"],
        control_seed_namespace=declaration["control_seed_namespace"],
        design_keys=tuple(sorted(keys)),
        stage1_points_per_stratum=int(declaration["stage1_points_per_stratum"]),
        stage2_points_per_stratum=int(declaration["stage2_points_per_stratum"]),
        wilson_width_threshold=float(declaration["wilson_width_threshold"]),
        control_fraction=float(declaration["control_fraction_per_cell"]),
        batch_size=int(declaration["batch_size"]),
        partial_checkpoint_prefix_count=int(declaration["partial_checkpoint_prefix_count"]),
        control_partial_checkpoint_prefix_count=int(declaration["control_partial_checkpoint_prefix_count"]),
        binding_gates=False,
    )
    require_safe_case_sizes(value, plan)
    return plan


def require_safe_case_sizes(value: Mapping[str, Any], plan: CampaignPlan) -> dict[str, int]:
    """Every case size of the plan must be exact at both Wilson ends (orbit_mc v1.7 defect guard)."""

    sizes = plan.case_sizes(value)
    for name, n in sizes.items():
        if not wilson_exact_at_ends(n):
            raise ValueError(f"case size {name} = {n} is not Wilson-exact at k = 0 / k = n (orbit_mc v1.7 would refuse a zero-count category)")
        if n <= (plan.control_partial_checkpoint_prefix_count if name.startswith("control") else plan.partial_checkpoint_prefix_count):
            raise ValueError(f"case size {name} = {n} does not reach the partial checkpoint prefix")
    return sizes


def plan_record(plan: CampaignPlan) -> dict[str, Any]:
    record = asdict(plan)
    record["design_keys"] = list(plan.design_keys)
    return record


# --------------------------------------------------------------------------
# Bound designs with catalogue cells
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundCells:
    design: BoundDesign
    cells: tuple[LaunchCell, ...]
    catalogue_entry: Mapping[str, Any]

    @property
    def design_key(self) -> str:
        return self.design.design_key

    def cell(self, cell_id: str) -> LaunchCell:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(cell_id)


def catalogue_design_id(design: BoundDesign) -> str:
    """The catalogue keys sweep designs by their sweep case id, the P2 design by its design id."""

    return design.design_key if design.set_id == SET_SWEEP else design.design_id


def bind_designs(value: Mapping[str, Any], sweep: SweepBinding, catalogue: CatalogueBinding, keys: Sequence[str]) -> dict[str, BoundCells]:
    """Geometry + identities + catalogue cells for every design (no field solve)."""

    representatives = set(value["designs"]["representative_case_ids"])
    rule = value["launches"]
    output: dict[str, BoundCells] = {}
    for key in keys:
        set_id = design_set(value, key)
        if set_id == SET_SWEEP:
            design = bind_sweep_design(sweep, key, value["field_source"], representative=key in representatives)
        else:
            design = bind_p2_design(value["designs"]["p2_design"])
        entry = catalogue_entry(catalogue, set_id, catalogue_design_id(design))
        geometry = entry["geometry"]
        if (
            abs(float(geometry["wall_radius_m"]) - design.wall_radius_m) > 1.0e-12
            or abs(float(geometry["straight_z_min_m"]) - design.straight_z_min_m) > 1.0e-12
            or abs(float(geometry["straight_z_max_m"]) - design.straight_z_max_m) > 1.0e-12
            or abs(float(geometry["chamber_length_m"]) - design.chamber_length_m) > 1.0e-12
        ):
            raise ValueError(f"{key}: catalogue geometry differs from the rebuilt design geometry")
        cells = design_cells(entry, injector_length_m=design.injector_length_m, rule=rule)
        for cell in cells:
            if not design.domain_z_min_m < cell.launch_z_m < design.domain_z_max_m:
                raise ValueError(f"{key} {cell.cell_id}: launch plane outside the orbit domain")
        output[key] = BoundCells(design, cells, entry)
    return output


# --------------------------------------------------------------------------
# Cases, identities, launches, payloads
# --------------------------------------------------------------------------


def case_key(design_key: str, cell_id: str, stage: str) -> str:
    return f"{design_key}--{cell_id}--{stage}-{stage_timestep(stage)}"


def campaign_id(plan: CampaignPlan, design_key: str, cell_id: str, stage: str) -> str:
    return f"{plan.campaign_id_prefix}:{design_key}:{cell_id}:{stage}:{stage_timestep(stage)}"


def orbit_config(value: Mapping[str, Any], design: BoundDesign, timestep: str) -> OrbitConfig:
    rule = value["orbit_geometry_rule"]
    return orbit_config_for_design(design, rule, rule["timestep_policies"][timestep])


def policy_identity(value: Mapping[str, Any], plan: CampaignPlan, design_key: str, cell_id: str, stage: str) -> str:
    return content_hash(
        {
            "protocol_semantic_sha256": semantic_sha256(value),
            "plan_kind": plan.kind,
            "design_key": design_key,
            "cell_id": cell_id,
            "stage": stage,
            "timestep": stage_timestep(stage),
            "stage1_points_per_stratum": plan.stage1_points_per_stratum,
            "stage2_points_per_stratum": plan.stage2_points_per_stratum,
            "wilson_width_threshold": plan.wilson_width_threshold,
            "control_fraction": plan.control_fraction,
        }
    )


def block_launches(value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells, cell: LaunchCell, block: int) -> tuple[ElectronLaunch, ...]:
    """The 128 launches of one cell block (block 0 = stage 1, blocks 1.. = stage 2)."""

    stage = STAGE1 if block == 0 else stage2_stage(block)
    return build_launches(
        campaign_id(plan, bound.design_key, cell.cell_id, stage),
        namespace=plan.seed_namespace,
        design_key=bound.design_key,
        cells=bound.cells,
        rule=plan.launch_rule(value),
        wall_radius_m=bound.design.wall_radius_m,
        index_ranges={cell.cell_id: plan.block_range(block)},
    )


def final_keys(value: Mapping[str, Any], plan: CampaignPlan, cell: LaunchCell, topped_up: bool) -> list[str]:
    rule = plan.launch_rule(value)
    stop = plan.stage2_points_per_stratum if topped_up else plan.stage1_points_per_stratum
    return sorted(cell_module.launch_key(cell.index, stratum, index) for stratum in strata(rule) for index in range(stop))


def control_launches(value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells, cell: LaunchCell, keys: Sequence[str]) -> tuple[ElectronLaunch, ...]:
    if not keys:
        raise ValueError(f"{cell.cell_id}: empty control selection")
    return build_launches(
        campaign_id(plan, bound.design_key, cell.cell_id, CONTROL),
        namespace=plan.seed_namespace,
        design_key=bound.design_key,
        cells=bound.cells,
        rule=plan.launch_rule(value),
        wall_radius_m=bound.design.wall_radius_m,
        index_ranges={cell.cell_id: (0, max(key_index(key) for key in keys) + 1)},
        selected_keys=set(keys),
    )


def cell_control_selection(plan: CampaignPlan, design_key: str, cell: LaunchCell, keys: Sequence[str], rounding: str) -> list[str]:
    selection = control_selection(
        namespace=plan.control_seed_namespace,
        design_key=design_key,
        cell_keys={cell.cell_id: list(keys)},
        fraction=plan.control_fraction,
        rounding=rounding,
    )
    return selection[cell.cell_id]


def runtime_launch_payload(campaign: str, launches: Sequence[Any]) -> dict[str, Any]:
    records = launch_records(launches)
    for record in records:
        record["seed_id"] = str(record["seed_id"])
    return {
        "schema_version": schema("launches"),
        "campaign_id": campaign,
        "ensemble_id": campaign,
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "seed_encoding": "unsigned-64 decimal string",
        "launches": records,
    }


def batch_records(plan: CampaignPlan, launches: Sequence[Any]) -> list[dict[str, Any]]:
    return frozen_batch_manifest(launches, batch_size=plan.batch_size, estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL)


def runtime_batch_payload(campaign: str, batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": schema("batches"),
        "campaign_id": campaign,
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "batches": list(batches),
    }


def load_runtime_launch_payload(data: bytes, expected_campaign_id: str) -> tuple[ElectronLaunch, ...]:
    """Closed typed loader (v4/v1 logic, this campaign's schema tag)."""

    decoded = _decode_runtime_tags(strict_json_loads(data))
    if not isinstance(decoded, dict) or set(decoded) != {"schema_version", "campaign_id", "ensemble_id", "estimator_policy", "seed_encoding", "launches"}:
        raise ValueError("runtime launch payload is not closed")
    if (
        decoded["schema_version"] != schema("launches")
        or decoded["campaign_id"] != expected_campaign_id
        or decoded["ensemble_id"] != expected_campaign_id
        or decoded["estimator_policy"] != EstimatorPolicy.UNWEIGHTED_BINOMIAL.value
        or decoded["seed_encoding"] != "unsigned-64 decimal string"
        or not isinstance(decoded["launches"], list)
    ):
        raise ValueError("runtime launch payload authority differs")
    launches: list[ElectronLaunch] = []
    expected_keys = {"launch_id", "seed_id", "kinetic_energy_ev", "pitch_angle_rad", "position_m", "parallel_direction", "gyrophase_rad", "flux_surface_id"}
    for record in decoded["launches"]:
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ValueError("runtime launch record is not closed")
        seed_text = record["seed_id"]
        if not isinstance(seed_text, str) or not seed_text.isascii() or not seed_text.isdecimal():
            raise ValueError("runtime launch seed is not an unsigned decimal string")
        seed = int(seed_text)
        if seed > 2**64 - 1 or str(seed) != seed_text:
            raise ValueError("runtime launch seed is outside canonical uint64")
        position = record["position_m"]
        if not isinstance(position, tuple) or len(position) != 3:
            raise ValueError("runtime launch position did not reconstruct as a tuple")
        launches.append(
            ElectronLaunch(
                launch_id=record["launch_id"],
                seed_id=seed,
                kinetic_energy_ev=record["kinetic_energy_ev"],
                pitch_angle_rad=record["pitch_angle_rad"],
                position_m=position,
                parallel_direction=record["parallel_direction"],
                gyrophase_rad=record["gyrophase_rad"],
                flux_surface_id=record["flux_surface_id"],
            )
        )
    ordered = tuple(sorted(launches, key=lambda item: item.launch_id))
    if len({item.launch_id for item in ordered}) != len(ordered):
        raise ValueError("runtime launch IDs are not unique")
    if any(not item.launch_id.startswith(expected_campaign_id + ":") for item in ordered):
        raise ValueError("runtime launch ID is not case-prefixed")
    return ordered


def case_authority_record(
    *,
    plan_kind: str,
    design_key: str,
    cell_id: str,
    stage: str,
    campaign: str,
    launches: Sequence[ElectronLaunch],
    batches: Sequence[Mapping[str, Any]],
    config: OrbitConfig,
    field_sha: str,
    policy_sha: str,
    tightness_floor: float,
) -> dict[str, Any]:
    launch_bytes = canonical_bytes(runtime_launch_payload(campaign, launches))
    batch_bytes = canonical_bytes(runtime_batch_payload(campaign, batches))
    record = {
        "schema_version": schema("case-authority"),
        "plan_kind": plan_kind,
        "case_key": case_key(design_key, cell_id, stage),
        "campaign_id": campaign,
        "ensemble_id": campaign,
        "design_key": design_key,
        "cell_id": cell_id,
        "stage": stage,
        "timestep": stage_timestep(stage),
        "launch_count": len(launches),
        "batch_count": len(batches),
        "launch_keys_sha256": content_hash(sorted(key_of_launch(item) for item in launches)),
        "runtime_launch_payload_byte_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "runtime_batch_payload_byte_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "orbit_launches_sha256": content_hash(launch_records(launches)),
        "batch_manifest_sha256": content_hash({"estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value, "batches": list(batches)}),
        "estimator_sha256": estimator_identity(launches, batches),
        "field_identity_sha256": field_sha,
        "config": asdict(config),
        "config_identity_sha256": content_hash(asdict(config)),
        "policy_identity_sha256": policy_sha,
    }
    record["case_authority_sha256"] = content_hash(
        {
            "campaign_id": campaign,
            "launches_sha256": record["orbit_launches_sha256"],
            "batch_manifest_sha256": record["batch_manifest_sha256"],
            "policy_sha256": record["policy_identity_sha256"],
            "minimum_certificate_tightness_ratio": tightness_floor,
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "estimator_sha256": record["estimator_sha256"],
            "replay_requirement": "deterministic_full_result_replay_required",
        }
    )
    record["case_authority_record_sha256"] = content_hash(record)
    return record


def build_stage1_authority(value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells, cell: LaunchCell) -> dict[str, Any]:
    launches = block_launches(value, plan, bound, cell, 0)
    return case_authority_record(
        plan_kind=plan.kind,
        design_key=bound.design_key,
        cell_id=cell.cell_id,
        stage=STAGE1,
        campaign=campaign_id(plan, bound.design_key, cell.cell_id, STAGE1),
        launches=launches,
        batches=batch_records(plan, launches),
        config=orbit_config(value, bound.design, "N"),
        field_sha=bound.design.accepted_field_identity,
        policy_sha=policy_identity(value, plan, bound.design_key, cell.cell_id, STAGE1),
        tightness_floor=float(value["gates"]["minimum_certificate_dense_to_bound_ratio"]),
    )


def design_authority_row(value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells) -> dict[str, Any]:
    rule = plan.launch_rule(value)
    design = bound.design
    cells = [cell.to_dict() for cell in bound.cells]
    # unsigned 64-bit seeds as decimal strings (canonical JSON integers are signed 64-bit)
    seeds = {
        f"{cell.cell_id}:{stratum['stratum_id']}": str(stratum_seed(plan.seed_namespace, design.design_key, cell.cell_id, stratum["stratum_id"]))
        for cell in bound.cells
        for stratum in strata(rule)
    }
    stages = [STAGE1] + [stage2_stage(block) for block in range(1, plan.block_count)] + [CONTROL]
    return {
        "design_key": design.design_key,
        "set_id": design.set_id,
        "design_id": design.design_id,
        "label": design.label,
        "field_level": design.field_level,
        "representative": design.representative,
        "sweep_index": design.sweep_index,
        "design_values": design.design_values,
        "geometry": design.geometry,
        "identities": design.identities,
        "accepted_field_identity_sha256": design.accepted_field_identity,
        "refined_field_identity_sha256": design.refined_field_identity,
        "catalogue": {
            "label": bound.catalogue_entry["label"],
            "record_path": bound.catalogue_entry["record_path"],
            "accepted_field_identity_sha256": bound.catalogue_entry["accepted_field_identity_sha256"],
            "wall_cusp_count": bound.catalogue_entry["wall_cusp_count"],
            "cell_count": bound.catalogue_entry["cell_count"],
            "wall_cusps_z_m": [cusp["z_c_m"] for cusp in bound.catalogue_entry["wall_cusps"]],
            "identity_note": "the catalogue's field identity is v3.1's own scheme; this campaign's field identity (accepted_field_identity_sha256) is v1's; the cells are bound by the catalogue bytes, the field by the re-solve identity proof",
        },
        "cells": cells,
        "cell_count": len(cells),
        "strata": list(strata(rule)),
        "stratum_seeds": seeds,
        "stratum_seed_encoding": "unsigned-64 decimal string",
        "control_seed_namespace": plan.control_seed_namespace,
        "candidate_launches_sha256": candidate_sha256(namespace=plan.seed_namespace, design_key=design.design_key, cells=bound.cells, rule=rule, wall_radius_m=design.wall_radius_m),
        "candidate_launch_count": len(cells) * plan.stage2_points_per_stratum * int(rule["strata_per_cell"]),
        "stage1_launch_count": len(cells) * plan.launches_per_block(value),
        "orbit_config": {step: asdict(orbit_config(value, design, step)) for step in ("N", "2N")},
        "policy_identity_sha256": {cell.cell_id: {stage: policy_identity(value, plan, design.design_key, cell.cell_id, stage) for stage in stages} for cell in bound.cells},
    }


def build_design_authorities(value: Mapping[str, Any], plan: CampaignPlan, bound_designs: Mapping[str, BoundCells]) -> dict[str, Any]:
    """Frozen per-design authority (cells, seeds, stage-1 case authorities; no field solve, no outcomes)."""

    design_rows = []
    case_rows = []
    for key in plan.design_keys:
        bound = bound_designs[key]
        design_rows.append(design_authority_row(value, plan, bound))
        for cell in bound.cells:
            case_rows.append(build_stage1_authority(value, plan, bound, cell))
    return {
        "schema_version": schema("design-authorities"),
        "plan_kind": plan.kind,
        "protocol_semantic_sha256": semantic_sha256(value),
        "catalogue_file_sha256": value["cusp_cell_catalogue"]["catalogue_file_sha256"],
        "design_count": len(design_rows),
        "cell_count": sum(row["cell_count"] for row in design_rows),
        "stage1_case_count": len(case_rows),
        "stage1_launches": sum(item["launch_count"] for item in case_rows),
        "candidate_launches": sum(row["candidate_launch_count"] for row in design_rows),
        "case_sizes": plan.case_sizes(value),
        "allocation_rule": plan.allocation_rule(value),
        "control_rule": {**dict(value["control"]), "fraction_per_cell": plan.control_fraction, "seed_namespace": plan.control_seed_namespace},
        "designs": design_rows,
        "stage1_cases": case_rows,
    }


def all_stage1_launches(value: Mapping[str, Any], plan: CampaignPlan, bound_designs: Mapping[str, BoundCells]) -> tuple[ElectronLaunch, ...]:
    return tuple(launch for key in plan.design_keys for cell in bound_designs[key].cells for launch in block_launches(value, plan, bound_designs[key], cell, 0))


def design_sha256(value: Mapping[str, Any], plan: CampaignPlan, bound_designs: Mapping[str, BoundCells]) -> str:
    return content_hash(launch_records(all_stage1_launches(value, plan, bound_designs)))


# --------------------------------------------------------------------------
# Disjointness (shakedown vs evidentiary on the same designs)
# --------------------------------------------------------------------------


def _signature(launches: Sequence[ElectronLaunch]) -> dict[str, set[Any]]:
    return {
        "launch_id": {item.launch_id for item in launches},
        "seed_id": {item.seed_id for item in launches},
        "position_m": {item.position_m for item in launches},
        "energy_pitch_direction_gyrophase": {(item.kinetic_energy_ev, item.pitch_angle_rad, item.parallel_direction, item.gyrophase_rad) for item in launches},
    }


def disjointness_report(left: Sequence[ElectronLaunch], right: Sequence[ElectronLaunch], *, left_name: str, right_name: str) -> dict[str, Any]:
    ls, rs = _signature(left), _signature(right)
    overlaps = {name: len(ls[name] & rs[name]) for name in ls}
    return {"left": left_name, "right": right_name, "left_launch_count": len(left), "right_launch_count": len(right), "overlap_counts": overlaps, "disjoint": all(count == 0 for count in overlaps.values())}


def _candidate_launches(value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells) -> tuple[ElectronLaunch, ...]:
    return build_launches(
        campaign_id(plan, bound.design_key, "all-cells", STAGE1),
        namespace=plan.seed_namespace,
        design_key=bound.design_key,
        cells=bound.cells,
        rule=plan.launch_rule(value),
        wall_radius_m=bound.design.wall_radius_m,
        index_ranges={cell.cell_id: (0, plan.stage2_points_per_stratum) for cell in bound.cells},
    )


def shakedown_disjointness(value: Mapping[str, Any], bound_designs: Mapping[str, BoundCells]) -> dict[str, Any]:
    """Shakedown candidates (all indices) vs evidentiary candidates (all indices) on the same designs."""

    shakedown = shakedown_plan(value)
    evidentiary = evidentiary_plan(value)
    left: list[ElectronLaunch] = []
    right: list[ElectronLaunch] = []
    for key in shakedown.design_keys:
        left.extend(_candidate_launches(value, shakedown, bound_designs[key]))
        right.extend(_candidate_launches(value, evidentiary, bound_designs[key]))
    report = disjointness_report(left, right, left_name="shakedown-candidates", right_name="evidentiary-candidates-same-designs")
    return {
        "shakedown_launch_count": len(left),
        "shakedown_unique_launch_ids": len({item.launch_id for item in left}),
        "shakedown_unique_seed_ids": len({item.seed_id for item in left}),
        "reports": {"against_evidentiary_same_designs": report},
        "namespaces": {
            "shakedown": shakedown.seed_namespace,
            "evidentiary": evidentiary.seed_namespace,
            "distinct": shakedown.seed_namespace != evidentiary.seed_namespace and shakedown.campaign_id_prefix != evidentiary.campaign_id_prefix,
        },
        "against_v1_v2_v3_v4": "disjoint by construction: v1-v4 launched at exactly 0.675/0.800 r_w with 8-point gyrophase grids and campaign-prefixed ids/seeds outside this namespace; v2 draws radii inside bands and gyrophases from scrambled Sobol (coincidence has probability zero)",
        "proven": (
            len({item.launch_id for item in left}) == len(left)
            and len({item.seed_id for item in left}) == len(left)
            and report["disjoint"]
            and shakedown.seed_namespace != evidentiary.seed_namespace
            and shakedown.campaign_id_prefix != evidentiary.campaign_id_prefix
        ),
    }


# --------------------------------------------------------------------------
# Manufactured gates (CPU only), per-case gate facts, endpoint rows
# --------------------------------------------------------------------------


def manufactured_gate_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """v4's manufactured checks without the CUDA parity leg (GPU occupied, CUDA unused)."""

    limits = value["gates"]
    helix = timestep_convergence()
    varying = varying_e_convergence()
    mirror = analytic_magnetic_bottle()
    energy = uniform_b_helix()
    wall = wall_event_accuracy()
    cpu = backend_parity(device="cpu")
    checks = {
        "uniform_b_energy": energy["relative_energy_error"] <= limits["maximum_relative_energy_error"],
        "helix_order": min(helix["observed_orders"]) >= limits["minimum_helix_position_order"],
        "varying_e_order": min(varying["observed_orders"]) >= limits["minimum_varying_e_position_order"],
        "mirror_smoke": mirror["relative_error"] <= limits["maximum_mirror_point_relative_error"],
        "wall_endpoint": wall["endpoint_error_m"] <= limits["maximum_wall_endpoint_error_m"],
        "cpu_parity": cpu["status"] == "evaluated" and cpu["maximum_relative_velocity_difference"] <= limits["maximum_cpu_cuda_relative_velocity_difference"],
    }
    return _plain(
        {
            "checks": checks,
            "passed": all(checks.values()),
            "cuda_parity": {"status": "not_evaluated", "reason": limits["backend_parity_scope"]},
            "uniform_b": energy,
            "helix_convergence": helix,
            "varying_e_convergence": varying,
            "mirror": mirror,
            "wall_event": wall,
            "cpu_parity": cpu,
        }
    )


def escape_subclass(result: Any, config: OrbitConfig) -> str | None:
    if result.termination is not Termination.DOMAIN_ESCAPE:
        return None
    x, y, z = (float(item) for item in result.final_position_m)
    radius = math.hypot(x, y)
    if abs(z - config.domain_z_min_m) <= ESCAPE_TOLERANCE_M:
        return "upstream_anode_plane"
    if abs(z - config.domain_z_max_m) <= ESCAPE_TOLERANCE_M:
        return "exit_plane"
    if abs(radius - config.domain_radius_m) <= ESCAPE_TOLERANCE_M and z > config.wall_z_max_m:
        return "divergent_section_radial"
    return "unclassified"


def case_gate_facts(results: Sequence[Any], field: Any, config: OrbitConfig) -> dict[str, Any]:
    """Compact per-case facts so the main process never needs the orbit results (v1 logic)."""

    witness_order = all(
        item.event_witness["event_fraction"]
        <= min([c for c in item.event_witness["candidate_fractions"].values() if c is not None] or [item.event_witness["event_fraction"]]) + 64.0 * np.finfo(float).eps
        for item in results
        if "candidate_fractions" in item.event_witness
    )
    wall_errors = [abs(math.hypot(*item.wall_endpoint_m[:2]) - config.wall_radius_m) for item in results if item.wall_endpoint_m is not None]
    subclasses: dict[str, int] = {}
    for item in results:
        label = escape_subclass(item, config)
        if label is not None:
            subclasses[label] = subclasses.get(label, 0) + 1
    return _plain(
        {
            "earliest_event_ordering": bool(witness_order),
            "runtime_rotation_bound": all(item.dt_s * abs(ELECTRON_CHARGE_C) * field.max_b_t / ELECTRON_MASS_KG <= config.max_rotation_rad * (1.0 + 1.0e-14) for item in results),
            "relativistic_phase_finite": all(math.isfinite(item.accumulated_gyro_phase_rad) and math.isfinite(float(item.event_witness.get("observed_gamma", 1.0))) for item in results),
            "maximum_relative_energy_error": max(item.maximum_relative_energy_error for item in results),
            "orbits_exceeding_energy_gate": None,
            "final_velocity_event_velocity_mismatches": sum(not _final_velocity_equals_event_velocity(item) for item in results),
            "maximum_wall_endpoint_error_m": max(wall_errors, default=0.0),
            "numerical_failure_counts": {termination.value: sum(item.termination is termination for item in results) for termination in NUMERICAL_FAILURES},
            "timeout_counts": {termination.value: sum(item.termination is termination for item in results) for termination in TIMEOUTS},
            "domain_escape_subclasses": dict(sorted(subclasses.items())),
            "material_quarantine": bool(np.all(field.traversable_cells)),
        }
    )


def endpoint_rows(launches: Sequence[ElectronLaunch], results: Sequence[Any], config: OrbitConfig, stage: str) -> list[dict[str, Any]]:
    by_id = {item.launch_id: item for item in launches}
    rows = []
    for result in sorted(results, key=lambda item: item.launch_id):
        launch = by_id[result.launch_id]
        x, y, z = (float(item) for item in result.final_position_m)
        key = key_of_launch(launch)
        rows.append(
            {
                "launch_id": result.launch_id,
                "launch_key": key,
                "stage": stage,
                "cell_id": launch.flux_surface_id.split("-r", 1)[0],
                "cell_index": key_cell_index(key),
                "sobol_index": key_index(key),
                "flux_surface_id": launch.flux_surface_id,
                "band_id": launch.flux_surface_id.rsplit("-", 1)[1],
                "kinetic_energy_ev": launch.kinetic_energy_ev,
                "pitch_angle_deg": round(math.degrees(launch.pitch_angle_rad), 12),
                "parallel_direction": launch.parallel_direction,
                "gyrophase_rad": launch.gyrophase_rad,
                "launch_r_m": math.hypot(launch.position_m[0], launch.position_m[1]),
                "launch_z_m": launch.position_m[2],
                "termination": result.termination.value,
                "escape_subclass": escape_subclass(result, config),
                "final_r_m": math.hypot(x, y),
                "final_z_m": z,
                "steps": result.steps,
                "elapsed_time_s": result.elapsed_time_s,
                "path_length_m": result.path_length_m,
                "maximum_relative_energy_error": result.maximum_relative_energy_error,
                "mu_relative_variation": result.maximum_instantaneous_mu_relative_variation,
                "complete_gyrocycles": result.complete_gyrocycles,
                "event_resolution": result.event_witness.get("event_resolution"),
                "tolerance_close": result.event_witness.get("event_resolution") == "tolerance_close_fraction_zero",
            }
        )
    return rows


# --------------------------------------------------------------------------
# Per-design worker: per cell stage 1 -> rule -> stage-2 blocks -> control -> flags -> sealing
# --------------------------------------------------------------------------


def _terminations(launches: Sequence[ElectronLaunch], results: Sequence[Any]) -> dict[str, str]:
    by_id = {item.launch_id: item for item in launches}
    return {key_of_launch(by_id[result.launch_id]): result.termination.value for result in results}


def paired_control(terminations_n: Mapping[str, str], terminations_2n: Mapping[str, str], maximum_change: float, cells: Sequence[LaunchCell]) -> dict[str, Any]:
    """Paired N vs 2N comparison over identical launches (the control subset of one design)."""

    keys = sorted(terminations_2n)
    if any(key not in terminations_n for key in keys):
        raise ValueError("control launch without an N-step partner")
    n = len(keys)
    wall_n = sum(terminations_n[key] == "wall_hit" for key in keys)
    wall_2n = sum(terminations_2n[key] == "wall_hit" for key in keys)
    discordant = sum(terminations_n[key] != terminations_2n[key] for key in keys)
    delta = (wall_2n - wall_n) / n if n else 0.0
    per_cell: dict[str, dict[str, Any]] = {}
    for key in keys:
        cell = per_cell.setdefault(cell_id_of_key(key, cells), {"n_control": 0, "wall_N": 0, "wall_2N": 0, "discordant": 0})
        cell["n_control"] += 1
        cell["wall_N"] += terminations_n[key] == "wall_hit"
        cell["wall_2N"] += terminations_2n[key] == "wall_hit"
        cell["discordant"] += terminations_n[key] != terminations_2n[key]
    for cell in per_cell.values():
        cell["delta_p_wall"] = (cell["wall_2N"] - cell["wall_N"]) / cell["n_control"]
        cell["quantum"] = 1.0 / cell["n_control"]
    return {
        "n_control": n,
        "wall_N": wall_n,
        "wall_2N": wall_2n,
        "p_wall_N": wall_n / n if n else None,
        "p_wall_2N": wall_2n / n if n else None,
        "delta_p_wall": delta,
        "quantum": 1.0 / n if n else None,
        "discordant": discordant,
        "discordance_rate": discordant / n if n else None,
        "maximum_allowed_change": maximum_change,
        "passed": bool(n > 0 and abs(delta) <= maximum_change),
        "per_cell": dict(sorted(per_cell.items())),
    }


def _rebuild_cells(rows: Sequence[Mapping[str, Any]]) -> tuple[LaunchCell, ...]:
    return tuple(LaunchCell(**dict(row)) for row in rows)


def _case_output(
    *,
    task: Mapping[str, Any],
    case: Mapping[str, Any],
    integration: Mapping[str, Any],
    flags: Mapping[str, bool],
    sealable: bool,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    results = integration["results"]
    config = case["config"]
    field = case["field"]
    stage = case["stage"]
    ordered = sorted(case["launches"], key=lambda item: item.launch_id)
    by_id = {item.launch_id: item for item in results}
    determinism = {launch.launch_id: content_hash(result_record(by_id[launch.launch_id])) for launch in ordered[:2]}
    export = None
    consumed = None
    ledger = ValidatorLedger()
    if sealable:
        export_task = dict(case)
        export_task["results"] = results
        export_task["summary"] = integration["summary"]
        export_task["convergence_evidence"] = dict(flags)
        export_task["preregistration"] = dict(case["preregistration"])
        export_task["export_handoff"] = True
        export = run_case_export(export_task)
        consumed = ledger.run(case["case_key"], "coupling_handoff_consumer", consume_handoff, export["handoff"], expected_artifact_sha256=export["artifact_file_sha256"], design_label=case["case_key"], evidence_class=task["label"])
    rows = endpoint_rows(case["launches"], results, config, stage)
    endpoints_payload = canonical_bytes(
        {
            "schema_version": schema("endpoints"),
            "classification": CLASSIFICATION,
            "label": task["label"],
            "campaign_id": case["campaign_id"],
            "case_key": case["case_key"],
            "design_key": task["design_key"],
            "cell_id": case["cell_id"],
            "stage": stage,
            "timestep": stage_timestep(stage),
            "sealed": sealable,
            "orbit_artifact_file_sha256": None if export is None else export["artifact_file_sha256"],
            "rows": rows,
        }
    )
    facts = case_gate_facts(results, field, config)
    facts["orbits_exceeding_energy_gate"] = sum(item.maximum_relative_energy_error > float(task["maximum_relative_energy_error"]) for item in results)
    validators = list(integration["validators"]) + ([] if export is None else list(export["validators"])) + ledger.records
    return {
        "case_key": case["case_key"],
        "campaign_id": case["campaign_id"],
        "design_key": task["design_key"],
        "cell_id": case["cell_id"],
        "stage": stage,
        "timestep": stage_timestep(stage),
        "process_id": os.getpid(),
        "authority": _plain(case["authority"]),
        "preflight": integration["preflight"],
        "summary": integration["summary"].to_dict(),
        "strata": integration["strata"],
        "checkpoints": [{key: item[key] for key in ("stage", "batch_id", "completed_launches", "file_sha256", "artifact_name")} for item in integration["checkpoints"]],
        "partial_checkpoint_file_sha256": integration["partial_checkpoint_file_sha256"],
        "final_checkpoint_file_sha256": integration["final_checkpoint_file_sha256"],
        "diagnostics": integration["diagnostics"],
        "gate_facts": facts,
        "determinism_hashes": determinism,
        "sealed": sealable,
        "artifact_path": None if export is None else export["artifact_path"],
        "artifact_sidecar_bytes": None if export is None else Path(export["artifact_sidecar_path"]).read_bytes(),
        "artifact_file_sha256": None if export is None else export["artifact_file_sha256"],
        "verified_file_sha256": None if export is None else export["verified_file_sha256"],
        "handoff": None if export is None else export["handoff"],
        "consumed_handoff": consumed,
        "endpoints_gz": gzip.compress(endpoints_payload, mtime=0),
        "endpoints_payload_sha256": hashlib.sha256(endpoints_payload).hexdigest(),
        "endpoint_row_count": len(rows),
        "terminations": {row["launch_key"]: row["termination"] for row in rows},
        "validators": validators,
        "timing_s": {
            **integration["timing_s"],
            "export_write_replay": None if export is None else export["timing_s"]["write_artifact_replay"],
            "export_verify_replay": None if export is None else export["timing_s"]["load_and_verify_replay"],
            "case_total": time.perf_counter() - case_started,
        },
    }


def _adaptive_case(task: Mapping[str, Any], template: Mapping[str, Any], plan: CampaignPlan, launches: Sequence[ElectronLaunch]) -> dict[str, Any]:
    batches = frozen_batch_manifest(launches, batch_size=plan.batch_size, estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL)
    authority = case_authority_record(
        plan_kind=plan.kind,
        design_key=task["design_key"],
        cell_id=template["cell_id"],
        stage=template["stage"],
        campaign=template["campaign_id"],
        launches=launches,
        batches=batches,
        config=template["config"],
        field_sha=template["field_sha"],
        policy_sha=template["policy_sha"],
        tightness_floor=float(task["tightness_floor"]),
    )
    return {**dict(template), "launches": tuple(launches), "batches": list(batches), "authority": authority, "launch_sha": authority["orbit_launches_sha256"], "batch_sha": authority["batch_manifest_sha256"]}


def run_design_full(task: Mapping[str, Any]) -> dict[str, Any]:
    """Integrate one design end to end inside a spawn worker (orbit results never leave it).

    Per cell: stage-1 block (frozen authority) -> allocation rule -> stage-2 blocks (rule-bound)
    -> control subset at 2N (rule-bound). Then the design's convergence flags and sealing
    (``write_artifact`` replay + ``load_and_verify_artifact`` replay + handoff + consumer) of
    every case iff the flags hold.
    """

    started = time.perf_counter()
    plan = CampaignPlan(**task["plan"])
    launch_rule = task["launch_rule"]
    allocation_rule = task["allocation_rule"]
    control_rule = task["control_rule"]
    design_key = task["design_key"]
    cells = _rebuild_cells(task["cells"])
    wall_radius = float(task["wall_radius_m"])
    cases: dict[str, dict[str, Any]] = {}
    integrations: dict[str, dict[str, Any]] = {}
    terminations_n: dict[str, str] = {}
    terminations_2n: dict[str, str] = {}
    stage1_counts: dict[str, dict[str, int]] = {}
    per_cell_decisions: dict[str, dict[str, Any]] = {}
    stage2_authorities: dict[str, list[dict[str, Any]]] = {}
    control_authorities: dict[str, dict[str, Any]] = {}
    selections: dict[str, list[str]] = {}
    for cell in cells:
        templates = task["cell_templates"][cell.cell_id]
        # ---- stage 1 (frozen) ----------------------------------------------
        stage1_case = task["stage1_cases"][cell.cell_id]
        key1 = stage1_case["case_key"]
        cases[key1] = dict(stage1_case)
        integrations[key1] = run_case_integration(stage1_case)
        cell_terminations = _terminations(stage1_case["launches"], integrations[key1]["results"])
        terminations_n.update(cell_terminations)
        counts = cell_counts_from_terminations(cell_terminations, cells)[cell.cell_id]
        stage1_counts[cell.cell_id] = counts
        # ---- allocation rule (code, frozen) ---------------------------------
        decision = allocation_decision({cell.cell_id: {"wall_hit": counts["wall_hit"], "trials": counts["trials"]}}, allocation_rule)
        per_cell_decisions[cell.cell_id] = decision["cells"][cell.cell_id]
        topped = bool(decision["cells"][cell.cell_id]["topped_up"])
        if topped:
            stage2_authorities[cell.cell_id] = []
            for block in range(1, plan.block_count):
                template = templates[stage2_stage(block)]
                launches = build_launches(
                    template["campaign_id"],
                    namespace=plan.seed_namespace,
                    design_key=design_key,
                    cells=cells,
                    rule=launch_rule,
                    wall_radius_m=wall_radius,
                    index_ranges={cell.cell_id: plan.block_range(block)},
                )
                case = _adaptive_case(task, template, plan, launches)
                cases[case["case_key"]] = case
                integrations[case["case_key"]] = run_case_integration(case)
                terminations_n.update(_terminations(launches, integrations[case["case_key"]]["results"]))
                stage2_authorities[cell.cell_id].append(case["authority"])
        # ---- control subset at 2N (frozen seed) -------------------------------
        cell_keys = sorted(key for key in terminations_n if cell_id_of_key(key, cells) == cell.cell_id)
        selection = control_selection(namespace=plan.control_seed_namespace, design_key=design_key, cell_keys={cell.cell_id: cell_keys}, fraction=plan.control_fraction, rounding=control_rule["rounding"])[cell.cell_id]
        selections[cell.cell_id] = selection
        template = templates[CONTROL]
        control_set = build_launches(
            template["campaign_id"],
            namespace=plan.seed_namespace,
            design_key=design_key,
            cells=cells,
            rule=launch_rule,
            wall_radius_m=wall_radius,
            index_ranges={cell.cell_id: (0, max(key_index(k) for k in selection) + 1)},
            selected_keys=set(selection),
        )
        case = _adaptive_case(task, template, plan, control_set)
        cases[case["case_key"]] = case
        integrations[case["case_key"]] = run_case_integration(case)
        terminations_2n.update(_terminations(control_set, integrations[case["case_key"]]["results"]))
        control_authorities[cell.cell_id] = case["authority"]
    stage2_counts = cell_counts_from_terminations({key: value for key, value in terminations_n.items() if key_index(key) >= plan.stage1_points_per_stratum}, cells)
    control = paired_control(terminations_n, terminations_2n, float(control_rule["maximum_paired_probability_change"]), cells)
    control["selection"] = selections
    control["selection_sha256"] = content_hash(selections)
    cell_rows = [
        pooled_cell_row(cell, stage1_counts[cell.cell_id], stage2_counts.get(cell.cell_id), per_cell_decisions[cell.cell_id], control["per_cell"].get(cell.cell_id), readiness_floor=float(task["readiness_floor"]))
        for cell in cells
    ]
    topped_ids = sorted(cell_id for cell_id, item in per_cell_decisions.items() if item["topped_up"])
    allocation = {
        "rule": allocation_rule["statement"],
        "wilson_width_threshold": float(allocation_rule["wilson_width_threshold"]),
        "cells": per_cell_decisions,
        "topped_up_cell_ids": topped_ids,
        "topped_up_cell_count": len(topped_ids),
        "saturated_cell_count": len(cells) - len(topped_ids),
        "stage2_blocks_per_topped_up_cell": plan.block_count - 1,
        "stage2_launch_count": sum(item["launch_count"] for items in stage2_authorities.values() for item in items),
    }
    # ---- convergence-evidence flags and sealing ------------------------------
    if task["seal_policy"] == "converged":
        timestep_flag = bool(control["passed"])
        seal_basis = "evidentiary: timestep_passed is the design's paired N->2N control check (|Delta P| <= 0.02 on its control subset)"
    elif task["seal_policy"] == "structural":
        timestep_flag = all(item["preflight"]["status"] == "passed" and all(result.termination not in NUMERICAL_FAILURES for result in item["results"]) for item in integrations.values())
        seal_basis = "shakedown: timestep_passed is a structural check (preflight passed, zero numerical failures across all cases); the paired control is informational at 16 launches per cell"
    else:
        raise ValueError("unknown seal policy")
    flags = {"timestep_passed": timestep_flag, "cross_map_passed": bool(task["field_adapter_passed"]), "backend_parity_passed": bool(task["cpu_parity_passed"])}
    sealable = all(flags.values())
    outputs = [_case_output(task=task, case=cases[key], integration=integrations[key], flags=flags, sealable=sealable) for key in cases]
    return {
        "design_key": design_key,
        "process_id": os.getpid(),
        "allocation": _plain(allocation),
        "stage2_authorities": _plain(stage2_authorities),
        "control": _plain(control),
        "control_authorities": _plain(control_authorities),
        "cell_rows": _plain(cell_rows),
        "convergence_flags": flags,
        "seal_policy": task["seal_policy"],
        "seal_basis": seal_basis,
        "sealed": sealable,
        "cases": outputs,
        "timing_s": {"design_total": time.perf_counter() - started},
    }


def resolve_design_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Worker: bind, re-solve (or sample through the v4 adapter) one design; return field + evidence (or the exclusion)."""

    value = task["protocol"]
    started = time.perf_counter()
    try:
        if task["set_id"] == SET_SWEEP:
            binding = load_sweep_binding(value["field_source"])
            resolved = resolve_design(binding, task["design_key"], value, include_refined=True)
            geometry = resolved.geometry.to_dict()
            field = resolved.accepted.field
            serialized = _plain(resolved.accepted.serialized)
            evidence = _plain(resolved.evidence)
        elif task["set_id"] == SET_P2:
            resolved_p2 = resolve_p2_design(value["designs"]["p2_design"], value["field_source"]["adapter_gates"])
            geometry = None
            field = resolved_p2["field"]
            serialized = _plain(resolved_p2["serialized"])
            evidence = _plain(resolved_p2["evidence"])
        else:
            raise ValueError(f"unknown design set {task['set_id']}")
    except Exception as error:  # recorded as an exclusion, never hidden
        return {"design_key": task["design_key"], "status": "excluded", "reason": f"{type(error).__name__}: {error}"[:4096], "seconds": time.perf_counter() - started}
    return {
        "design_key": task["design_key"],
        "status": "resolved" if evidence["passed"] else "excluded",
        "reason": None if evidence["passed"] else "field adapter gates failed",
        "geometry": geometry,
        "accepted_field": field,
        "accepted_serialized": serialized,
        "evidence": evidence,
        "seconds": time.perf_counter() - started,
    }


def worker_count(value: Mapping[str, Any]) -> int:
    execution = value["execution"]
    if not execution["parallel_designs"]:
        return 1
    return max(1, min(int(execution["max_case_workers"]), os.cpu_count() or 1))


# --------------------------------------------------------------------------
# Main-process assessment: replay, gates, dataset rows
# --------------------------------------------------------------------------


def _authority_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left[key] == right[key] for key in AUTHORITY_COMPARED_KEYS)


def replay_allocation(value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells, outcome: Mapping[str, Any], stage1_terminations: Mapping[str, str]) -> dict[str, Any]:
    """Recompute every cell's top-up decision, stage-2 authorities and control authority from the stage-1 terminations."""

    counts = cell_counts_from_terminations(stage1_terminations, bound.cells)
    decision = allocation_decision({cell: {"wall_hit": c["wall_hit"], "trials": c["trials"]} for cell, c in counts.items()}, plan.allocation_rule(value))
    tightness = float(value["gates"]["minimum_certificate_dense_to_bound_ratio"])
    checks: dict[str, bool] = {
        "topped_up_set_reproduced": list(decision["topped_up_cell_ids"]) == list(outcome["allocation"]["topped_up_cell_ids"]),
        "per_cell_decisions_reproduced": all(
            decision["cells"][cell.cell_id]["topped_up"] == outcome["allocation"]["cells"][cell.cell_id]["topped_up"]
            and decision["cells"][cell.cell_id]["stage1_wall_hit"] == outcome["allocation"]["cells"][cell.cell_id]["stage1_wall_hit"]
            for cell in bound.cells
        ),
    }
    stage2_ok = True
    control_ok = True
    selection_ok = True
    expected_stage2 = 0
    expected_control = 0
    expected_selection: dict[str, list[str]] = {}
    for cell in bound.cells:
        topped = decision["cells"][cell.cell_id]["topped_up"]
        realised = outcome["stage2_authorities"].get(cell.cell_id)
        if topped:
            expected = []
            for block in range(1, plan.block_count):
                launches = block_launches(value, plan, bound, cell, block)
                expected.append(
                    case_authority_record(
                        plan_kind=plan.kind,
                        design_key=bound.design_key,
                        cell_id=cell.cell_id,
                        stage=stage2_stage(block),
                        campaign=campaign_id(plan, bound.design_key, cell.cell_id, stage2_stage(block)),
                        launches=launches,
                        batches=batch_records(plan, launches),
                        config=orbit_config(value, bound.design, "N"),
                        field_sha=bound.design.accepted_field_identity,
                        policy_sha=policy_identity(value, plan, bound.design_key, cell.cell_id, stage2_stage(block)),
                        tightness_floor=tightness,
                    )
                )
            expected_stage2 += sum(item["launch_count"] for item in expected)
            stage2_ok = stage2_ok and realised is not None and len(realised) == len(expected) and all(_authority_equal(a, b) for a, b in zip(realised, expected))
        else:
            stage2_ok = stage2_ok and not realised
        keys = final_keys(value, plan, cell, topped)
        selection = cell_control_selection(plan, bound.design_key, cell, keys, value["control"]["rounding"])
        expected_selection[cell.cell_id] = selection
        selection_ok = selection_ok and list(outcome["control"]["selection"].get(cell.cell_id, [])) == selection
        control = control_launches(value, plan, bound, cell, selection)
        expected_authority = case_authority_record(
            plan_kind=plan.kind,
            design_key=bound.design_key,
            cell_id=cell.cell_id,
            stage=CONTROL,
            campaign=campaign_id(plan, bound.design_key, cell.cell_id, CONTROL),
            launches=control,
            batches=batch_records(plan, control),
            config=orbit_config(value, bound.design, "2N"),
            field_sha=bound.design.accepted_field_identity,
            policy_sha=policy_identity(value, plan, bound.design_key, cell.cell_id, CONTROL),
            tightness_floor=tightness,
        )
        expected_control += expected_authority["launch_count"]
        realised_control = outcome["control_authorities"].get(cell.cell_id)
        control_ok = control_ok and realised_control is not None and _authority_equal(realised_control, expected_authority)
        checks[f"{cell.cell_id}:stage2_keys_inside_candidate_superset"] = all(0 <= key_index(key) < plan.stage2_points_per_stratum and key_cell_index(key) == cell.index for key in keys)
    checks["stage2_authorities_reproduced"] = stage2_ok
    checks["control_selection_reproduced"] = selection_ok
    checks["control_authorities_reproduced"] = control_ok
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "replayed_decision": decision,
        "expected_stage2_launch_count": expected_stage2,
        "expected_control_launch_count": expected_control,
        "control_selection_sha256": content_hash(expected_selection),
    }


def design_gates(
    value: Mapping[str, Any],
    design_evidence: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    outcome: Mapping[str, Any],
    replay: Mapping[str, Any],
    catalogue_bound: bool,
    stage1_frozen: bool,
    cell_count: int,
    *,
    binding: bool,
) -> dict[str, Any]:
    limits = value["gates"]
    facts = {key: item["gate_facts"] for key, item in cases.items()}
    flags = outcome["convergence_flags"]
    stages = [item["stage"] for item in cases.values()]
    checks = {
        "field_adapter": bool(design_evidence["passed"]),
        "catalogue_binding": bool(catalogue_bound),
        "stage1_authority_frozen": bool(stage1_frozen),
        "allocation_rule_replay": bool(replay["passed"]),
        "campaign_preflight": all(item["preflight"]["status"] == "passed" and item["preflight"]["maximum_launch_b_t"] <= item["preflight"]["maximum_declared_b_t"] for item in cases.values()),
        "zero_numerical_failures": all(sum(f["numerical_failure_counts"].values()) == 0 for f in facts.values()),
        "energy": all(f["maximum_relative_energy_error"] <= limits["maximum_relative_energy_error"] for f in facts.values()),
        "final_velocity_equals_event_velocity": all(f["final_velocity_event_velocity_mismatches"] == 0 for f in facts.values()),
        "wall_endpoint": all(f["maximum_wall_endpoint_error_m"] <= limits["maximum_wall_endpoint_error_m"] for f in facts.values()),
        "earliest_event": all(f["earliest_event_ordering"] for f in facts.values()),
        "runtime_rotation": all(f["runtime_rotation_bound"] for f in facts.values()),
        "relativistic_phase": all(f["relativistic_phase_finite"] for f in facts.values()),
        "material_quarantine": all(f["material_quarantine"] for f in facts.values()),
        "independent_repeats": all(row["physical_position_repeat_count"] >= limits["minimum_independent_repeats_per_stratum"] for item in cases.values() for row in item["strata"]),
        "exact_authority_replay_when_sealed": all((not item["sealed"]) or item["artifact_file_sha256"] == item["verified_file_sha256"] for item in cases.values()),
        "sealed_iff_convergence_flags": all(item["sealed"] == bool(flags["timestep_passed"] and flags["cross_map_passed"] and flags["backend_parity_passed"]) for item in cases.values()),
        "seal_policy_matches_plan": outcome["seal_policy"] == ("converged" if binding else "structural"),
        "cross_process_determinism": all(item["determinism_sample"]["passed"] for item in cases.values()),
        "handoff_consumed_when_sealed": all((not item["sealed"]) or (item["consumed_handoff"] is not None and item["consumed_handoff"]["passed"]) for item in cases.values()),
        "expected_cases_present": stages.count(STAGE1) == cell_count and stages.count(CONTROL) == cell_count and sum(s.startswith("stage2b") for s in stages) == outcome["allocation"]["topped_up_cell_count"] * outcome["allocation"]["stage2_blocks_per_topped_up_cell"],
        "case_sizes_wilson_exact": all(wilson_exact_at_ends(int(item["summary"]["trial_count"])) for item in cases.values()),
    }
    timeouts = sum(sum(f["timeout_counts"].values()) for f in facts.values())
    return {
        "checks": checks,
        "structural_passed": all(checks.values()),
        "control_flag": bool(outcome["control"]["passed"]),
        "sealed": all(item["sealed"] for item in cases.values()),
        "timeout_free": timeouts == 0,
        "timeout_count": timeouts,
        "passed": bool(all(checks.values())),
    }


def _case_totals(case: Mapping[str, Any]) -> dict[str, Any]:
    summary = case["summary"]
    return {
        "campaign_id": case["campaign_id"],
        "cell_id": case["cell_id"],
        "stage": case["stage"],
        "timestep": case["timestep"],
        "trial_count": summary["trial_count"],
        "termination_counts": dict(summary["termination_counts"]),
        "wall_hit": summary["wall_hit"],
        "reflected": summary["reflected"],
        "domain_escape": summary["escaped"],
        "timeout": summary["incomplete"],
        "domain_escape_subclasses": case["gate_facts"]["domain_escape_subclasses"],
        "timeout_counts": case["gate_facts"]["timeout_counts"],
        "sealed": case["sealed"],
        "orbit_artifact_file_sha256": case["artifact_file_sha256"],
        "handoff_sha256": None if case["handoff"] is None else content_hash(case["handoff"]),
        "endpoints_payload_sha256": case["endpoints_payload_sha256"],
        "batch_count": case["authority"]["batch_count"],
        "runtime_launch_payload_byte_sha256": case["authority"]["runtime_launch_payload_byte_sha256"],
        "orbit_launches_sha256": case["authority"]["orbit_launches_sha256"],
        "steps": case["diagnostics"]["steps"],
        "per_orbit_ms": case["timing_s"]["per_orbit_ms"],
        "tolerance_close_event_count": case["diagnostics"]["tolerance_close_event_count"],
        "mu": case["diagnostics"]["magnetic_moment_variation_diagnostic"],
    }


def _stratum_table(cases: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Final (stage 1 + stage 2) counts per (cell, energy, pitch, direction) stratum."""

    table: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case in cases.values():
        if case["timestep"] != "N":
            continue
        for row in case["strata"]:
            key = (row["cell_id"], row["kinetic_energy_ev"], row["pitch_angle_deg"], row["parallel_direction"])
            item = table.setdefault(key, {"cell_id": key[0], "kinetic_energy_ev": key[1], "pitch_angle_deg": key[2], "parallel_direction": key[3], "trials": 0, "wall_hit": 0, "reflected": 0, "domain_escape": 0, "timeout": 0})
            counts = row["termination_counts"]
            item["trials"] += int(row["trials"])
            item["wall_hit"] += int(counts["wall_hit"])
            item["reflected"] += int(counts["reflected"])
            item["domain_escape"] += int(counts["domain_escape"])
            item["timeout"] += int(row["trials"]) - int(counts["wall_hit"]) - int(counts["reflected"]) - int(counts["domain_escape"])
    output = []
    for key in sorted(table):
        item = table[key]
        item["p_wall"] = item["wall_hit"] / item["trials"]
        output.append(item)
    return output


def _field_summary(value: Mapping[str, Any], design: BoundDesign, evidence: Mapping[str, Any]) -> dict[str, Any]:
    accepted = evidence["accepted_bore_field"]
    cross = evidence.get("cross_resolution")
    return {
        "field_level": design.field_level,
        "status": value["field_source"]["field_status"] if design.set_id == SET_SWEEP else evidence.get("field_level"),
        "bore_max_b_t": accepted["max_b_t"],
        "bore_grid": accepted["bore_grid"],
        "interpolation_b_relative_rms": accepted["interpolation_error_report"]["b_relative_rms"],
        "cross_resolution_b_relative_rms": None if cross is None else cross["b_relative_rms"],
        "cross_resolution_evaluated": cross is not None,
        "sweep_qois": evidence.get("sweep_record"),
    }


def v1_comparison_row(v1_rows: Mapping[str, Mapping[str, Any]], design_key: str, pooled: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    v1 = v1_rows.get(design_key)
    if v1 is None:
        return None
    left = v1["reported"]["wall_hit"]
    rows = {}
    for name, item in pooled.items():
        rows[name] = {
            "v2_probability": item["probability"],
            "difference_v2_minus_v1": item["probability"] - left["probability"],
            "intervals_overlap": max(left["lower"], item["lower"]) <= min(left["upper"], item["upper"]),
        }
    return {
        "v1_case": "accepted-2N",
        "v1_probability": left["probability"],
        "v1_interval": [left["lower"], left["upper"]],
        "v1_trials": left["trials"],
        "v1_cells_z_m": [cell["axial_center_m"] for cell in v1["launch_design"]["cells"]],
        "v1_per_cell_p_wall": [v1["per_cell"]["accepted-2N"][cell]["wall_hit"]["probability"] for cell in sorted(v1["per_cell"]["accepted-2N"])],
        "comparison": rows,
        "statement": "same field, different cells and launch distributions; pooled values only",
    }


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        output = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            rank = 0.5 * (position + end) + 1.0
            for index in order[position : end + 1]:
                output[index] = rank
            position = end + 1
        return output

    a = ranks(left)
    b = ranks(right)
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0.0 or var_b == 0.0:
        return None
    return cov / math.sqrt(var_a * var_b)


def dataset_row(
    value: Mapping[str, Any],
    authority_row: Mapping[str, Any],
    bound: BoundCells,
    design_evidence: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    outcome: Mapping[str, Any],
    replay: Mapping[str, Any],
    gates: Mapping[str, Any],
    v1_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    design = bound.design
    cell_rows = list(outcome["cell_rows"])
    pooled = {"wall_area": design_pooled(cell_rows, weight="wall_area"), "launches": design_pooled(cell_rows, weight="launches")}
    n_cases = {key: item for key, item in cases.items() if item["timestep"] == "N"}
    n_final = sum(int(row["final"]["trials"]) for row in cell_rows)
    stage1_cases = [item for item in cases.values() if item["stage"] == STAGE1]
    return {
        "design_key": design.design_key,
        "set_id": design.set_id,
        "design_id": design.design_id,
        "label": design.label,
        "classification": CLASSIFICATION if design.set_id == SET_SWEEP else LABEL_P2,
        "sweep_index": design.sweep_index,
        "representative": design.representative,
        "design_values": design.design_values,
        "geometry": design.geometry,
        "identities": {
            **design.identities,
            "accepted_field_identity_sha256": design.accepted_field_identity,
            "refined_field_identity_sha256": design.refined_field_identity,
            "catalogue_accepted_field_identity_sha256": authority_row["catalogue"]["accepted_field_identity_sha256"],
        },
        "field": _field_summary(value, design, design_evidence),
        "catalogue": authority_row["catalogue"],
        "launch_design": {
            "cell_count": len(cell_rows),
            "strata_per_cell": int(value["launches"]["strata_per_cell"]),
            "radius_bands_of_wall": value["launches"]["radius_bands_of_wall"],
            "launch_radii_m": [[(band["centre_of_wall"] - band["half_width_of_wall"]) * design.wall_radius_m, (band["centre_of_wall"] + band["half_width_of_wall"]) * design.wall_radius_m] for band in value["launches"]["radius_bands_of_wall"]],
            "launch_planes_z_m": [row["launch_z_m"] for row in cell_rows],
            "stage1_launches": sum(int(item["summary"]["trial_count"]) for item in stage1_cases),
            "stage2_launches": sum(int(item["summary"]["trial_count"]) for item in cases.values() if item["stage"].startswith("stage2b")),
            "control_launches": sum(int(item["summary"]["trial_count"]) for item in cases.values() if item["stage"] == CONTROL),
            "final_launches": n_final,
            "candidate_launches_sha256": authority_row["candidate_launches_sha256"],
        },
        "allocation": outcome["allocation"],
        "allocation_replay": {"checks": replay["checks"], "passed": replay["passed"]},
        "cells": cell_rows,
        "per_stratum_final": _stratum_table(cases),
        "pooled": pooled,
        "control": {key: item for key, item in outcome["control"].items() if key != "selection"},
        "convergence_flags": outcome["convergence_flags"],
        "seal_policy": outcome["seal_policy"],
        "seal_basis": outcome["seal_basis"],
        "sealed": outcome["sealed"],
        "cases": {key: _case_totals(item) for key, item in cases.items()},
        "orbit_config": authority_row["orbit_config"],
        "v1_comparison": v1_comparison_row(v1_rows, design.design_key, pooled) if design.set_id == SET_SWEEP else None,
        "diagnostics": {
            "magnetic_moment_variation": {
                "role": "diagnostic_only",
                "binding": False,
                "median_of_case_medians": statistics.median(item["diagnostics"]["magnetic_moment_variation_diagnostic"]["median"] for item in n_cases.values() if item["diagnostics"]["magnetic_moment_variation_diagnostic"]["median"] is not None),
                "max": max(item["diagnostics"]["magnetic_moment_variation_diagnostic"]["max"] for item in n_cases.values() if item["diagnostics"]["magnetic_moment_variation_diagnostic"]["max"] is not None),
                "count_above_0p5": sum(item["diagnostics"]["magnetic_moment_variation_diagnostic"]["count_above_0p5"] for item in n_cases.values()),
            },
            "tolerance_close_share": sum(item["diagnostics"]["tolerance_close_event_count"] for item in n_cases.values()) / max(1, n_final),
            "steps_median_of_case_medians": statistics.median(item["diagnostics"]["steps"]["median"] for item in n_cases.values()),
            "steps_max": max(item["diagnostics"]["steps"]["max"] for item in n_cases.values()),
            "maximum_relative_energy_error": max(item["gate_facts"]["maximum_relative_energy_error"] for item in cases.values()),
            "reflections_final_n": sum(int(row["final"]["reflected"]) for row in cell_rows),
            "reflections_control_2n": sum(int(item["summary"]["termination_counts"]["reflected"]) for item in cases.values() if item["stage"] == CONTROL),
            "domain_escape_subclasses_final_n": {key: sum(item["gate_facts"]["domain_escape_subclasses"].get(key, 0) for item in n_cases.values()) for key in value["diagnostics"]["domain_escape_subclasses"]},
        },
        "gates": gates,
    }


CSV_COLUMNS = (
    "design_key", "design_id", "set_id", "label", "sweep_index", "representative",
    "cell_id", "cell_index", "kind", "position_class", "z_start_m", "z_end_m", "length_m", "launch_z_m",
    "wall_area_m2", "start_cusp_z_m", "end_cusp_z_m", "length_over_pitch", "wall_mirror_ratio", "axis_mirror_ratio",
    "wall_b_min_t", "axis_bz_peak_t", "boundary_ambiguous", "short_cell", "launch_plane_inside_injector_zone",
    "stage_count_selector", "stage_pitch_m", "magnet_axial_fraction", "chamber_outer_radius_m",
    "dielectric_thickness_m", "radial_clearance_m", "magnet_radial_thickness_m", "source_strength_scale",
    "exit_length_fraction", "exit_expansion_descriptor", "first_polarity_selector",
    "wall_radius_m", "chamber_length_m", "injector_length_m", "exit_start_m", "stage_count", "stage_pitch_represented_m", "has_divergent_exit",
    "bore_max_b_t", "interpolation_b_relative_rms", "cross_resolution_b_relative_rms",
    "n_stage1", "wall_stage1", "wilson_width_stage1", "topped_up", "n_stage2", "wall_stage2",
    "n_final", "wall_final", "reflected_final", "escape_final", "timeout_final",
    "p_wall", "p_wall_lower", "p_wall_upper", "p_reflected", "p_escape", "p_timeout",
    "wilson_width_final", "binomial_floor", "jeffreys_floor", "surrogate_ready",
    "n_control", "wall_N_control", "wall_2N_control", "delta_p_control", "discordant_control",
    "design_p_wall_area_weighted", "design_p_wall_launch_weighted", "design_control_delta_p", "design_timestep_passed",
    "v1_p_wall_2N", "sealed", "structural_gates_passed", "accepted_field_identity_sha256", "catalogue_accepted_field_identity_sha256",
)
DESIGN_VALUE_NAMES = (
    "stage_count_selector", "stage_pitch_m", "magnet_axial_fraction", "chamber_outer_radius_m",
    "dielectric_thickness_m", "radial_clearance_m", "magnet_radial_thickness_m", "source_strength_scale",
    "exit_length_fraction", "exit_expansion_descriptor", "first_polarity_selector",
)


def dataset_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """One CSV row per CELL (the dataset's unit); design-level columns repeat."""

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        values = row["design_values"] or {}
        geometry = row["geometry"]
        field = row["field"]
        v1 = row["v1_comparison"]
        for cell in row["cells"]:
            final = cell["final"]
            control = cell["control"] or {}
            writer.writerow(
                [
                    row["design_key"], row["design_id"], row["set_id"], row["label"], row["sweep_index"], row["representative"],
                    cell["cell_id"], cell["index"], cell["kind"], cell["position_class"], cell["z_start_m"], cell["z_end_m"], cell["length_m"], cell["launch_z_m"],
                    cell["wall_area_m2"], cell["start_cusp_z_m"], cell["end_cusp_z_m"], cell["length_over_pitch"], cell["wall_mirror_ratio"], cell["axis_mirror_ratio"],
                    cell["wall_b_min_t"], cell["axis_bz_peak_t"], cell["boundary_ambiguous"], cell["short_cell"], cell["launch_plane_inside_injector_zone"],
                    *[values.get(name) for name in DESIGN_VALUE_NAMES],
                    geometry["wall_radius_m"], geometry["chamber_length_m"], geometry["injector_length_m"], geometry["exit_start_m"], geometry["stage_count"], geometry["stage_pitch_m"], geometry["has_divergent_exit"],
                    field["bore_max_b_t"], field["interpolation_b_relative_rms"], field["cross_resolution_b_relative_rms"],
                    cell["stage1"]["trials"], cell["stage1"]["wall_hit"], cell["stage1"]["wilson_width"], cell["topped_up"],
                    0 if cell["stage2"] is None else cell["stage2"]["trials"], 0 if cell["stage2"] is None else cell["stage2"]["wall_hit"],
                    final["trials"], final["wall_hit"], final["reflected"], final["domain_escape"], final["timeout"],
                    final["p_wall"]["probability"], final["p_wall"]["lower"], final["p_wall"]["upper"],
                    final["p_reflected"]["probability"], final["p_escape"]["probability"], final["p_timeout"]["probability"],
                    final["wilson_width"], final["binomial_floor"], final["jeffreys_floor"], final["surrogate_ready"],
                    control.get("n_control", 0), control.get("wall_N", 0), control.get("wall_2N", 0), control.get("delta_p_wall"), control.get("discordant", 0),
                    row["pooled"]["wall_area"]["probability"], row["pooled"]["launches"]["probability"], row["control"]["delta_p_wall"], row["convergence_flags"]["timestep_passed"],
                    None if v1 is None else v1["v1_probability"], row["sealed"], row["gates"]["structural_passed"],
                    row["identities"]["accepted_field_identity_sha256"], row["identities"]["catalogue_accepted_field_identity_sha256"],
                ]
            )
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Shakedown record verification (prepare, execute, prebundle)
# --------------------------------------------------------------------------


def verify_shakedown_record(value: Mapping[str, Any], record: Mapping[str, Any], bound_designs: Mapping[str, BoundCells] | None = None) -> dict[str, bool]:
    """Fail closed unless the shakedown proves the current protocol, orbit_mc, field pipeline, code and catalogue."""

    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = record.get("protocol_semantic_sha256") == semantic_sha256(value)
    checks["orbit_mc_source_sha256_current"] = record.get("orbit_mc_source_sha256") == orbit_mc_source_sha256()
    checks["field_pipeline_source_sha256_current"] = record.get("field_pipeline_source_sha256") == field_pipeline_source_sha256()
    checks["experiment_code_sha256_current"] = record.get("experiment_code_sha256") == experiment_code_sha256()
    checks["catalogue_file_sha256_declared"] = record.get("catalogue_file_sha256") == value["cusp_cell_catalogue"]["catalogue_file_sha256"]
    contract = orbit_mc_contract_report(value)
    checks["orbit_mc_schema_versions_current"] = record.get("orbit_mc_schema_versions") == contract["observed"] and contract["matches"]
    disjointness = record.get("disjointness")
    checks["disjointness_proven"] = (
        isinstance(disjointness, Mapping)
        and disjointness.get("proven") is True
        and set(disjointness.get("reports", {})) == {"against_evidentiary_same_designs"}
        and all(report["disjoint"] is True and all(count == 0 for count in report["overlap_counts"].values()) for report in disjointness.get("reports", {}).values())
    )
    plan = shakedown_plan(value)
    cases = record.get("cases")
    checks["expected_cases"] = isinstance(cases, Mapping) and all(
        any(key.startswith(f"{design}--") and key.endswith(f"--{STAGE1}-N") for key in cases) and any(key.startswith(f"{design}--") and key.endswith(f"--{CONTROL}-2N") for key in cases)
        for design in plan.design_keys
    ) and all(key.split("--")[0] in plan.design_keys for key in cases)
    try:
        if bound_designs is None:
            sweep = load_sweep_binding(value["field_source"])
            catalogue = load_bound_catalogue(value["cusp_cell_catalogue"])
            bound_designs = bind_designs(value, sweep, catalogue, plan.design_keys)
        checks["shakedown_design_sha256_current"] = record.get("shakedown_design_sha256") == design_sha256(value, plan, bound_designs)
        checks["disjointness_recomputed"] = shakedown_disjointness(value, bound_designs)["proven"]
    except Exception:
        checks["shakedown_design_sha256_current"] = False
        checks["disjointness_recomputed"] = False
    validators = record.get("validators")
    checks["all_validators_passed"] = (
        isinstance(validators, Mapping)
        and validators.get("all_passed") is True
        and validators.get("failed") == 0
        and isinstance(cases, Mapping)
        and all(item.get("validators", {}).get("failed") == 0 and item.get("validators", {}).get("passed", 0) > 0 and item.get("export_stage_ran") is True and item.get("handoff_consumed") is True for item in cases.values())
    )
    allocation = record.get("allocation_summary")
    checks["both_allocation_branches_observed"] = (
        isinstance(allocation, Mapping) and int(allocation.get("topped_up_cells", 0)) > 0 and int(allocation.get("saturated_cells", 0)) > 0 and allocation.get("replay_all_passed") is True
    )
    checks["zero_exclusions"] = record.get("design_exclusions") == []
    runtime = record.get("runtime")
    checks["runtime_accepted_and_bundle_validated"] = isinstance(runtime, Mapping) and runtime.get("terminal_state") == "accepted_result" and runtime.get("bundle_validated") is True
    checks["dataset_assembled"] = isinstance(record.get("dataset_summary"), Mapping) and record["dataset_summary"].get("design_count") == len(plan.design_keys)
    projection = record.get("timing_projection")
    checks["timing_within_budget"] = isinstance(projection, Mapping) and projection.get("within_budget_expected") is True
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError("shakedown gate refused: " + ", ".join(failed))
    return checks


# --------------------------------------------------------------------------
# Shared runtime callbacks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenAuthority:
    authorities: Mapping[str, Any]
    design_authorities: Mapping[str, Any]
    shakedown: Mapping[str, Any]
    shakedown_bytes: bytes


def load_frozen_authority() -> FrozenAuthority:
    return FrozenAuthority(strict_json_file(AUTHORITIES_PATH), strict_json_file(DESIGN_AUTHORITIES_PATH), strict_json_file(SHAKEDOWN_PATH), SHAKEDOWN_PATH.read_bytes())


def _position_class_summary(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for position in POSITION_CLASSES:
        rows = [row for row in cell_rows if row["position_class"] == position]
        if not rows:
            output[position] = {"cell_count": 0}
            continue
        p = sorted(row["final"]["p_wall"]["probability"] for row in rows)
        output[position] = {
            "cell_count": len(rows),
            "p_wall_min": p[0],
            "p_wall_q1": p[len(p) // 4],
            "p_wall_median": statistics.median(p),
            "p_wall_q3": p[(3 * len(p)) // 4],
            "p_wall_max": p[-1],
            "p_wall_mean": statistics.fmean(p),
            "p_reflected_mean": statistics.fmean(row["final"]["p_reflected"]["probability"] for row in rows),
            "p_escape_mean": statistics.fmean(row["final"]["p_escape"]["probability"] for row in rows),
            "topped_up_count": sum(row["topped_up"] for row in rows),
            "saturated_count": sum(row["saturated_after_stage1"] for row in rows),
            "surrogate_ready_count": sum(row["final"]["surrogate_ready"] for row in rows),
            "saturated_at_zero": sum(row["final"]["wall_hit"] == 0 for row in rows),
            "saturated_at_one": sum(row["final"]["wall_hit"] == row["final"]["trials"] for row in rows),
        }
    return output


def _cell_templates(*, value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells, cell: LaunchCell, common: Mapping[str, Any], configs: Mapping[str, OrbitConfig]) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for stage in [stage2_stage(block) for block in range(1, plan.block_count)] + [CONTROL]:
        step = stage_timestep(stage)
        templates[stage] = {
            **common,
            "case_key": case_key(bound.design_key, cell.cell_id, stage),
            "cell_id": cell.cell_id,
            "stage": stage,
            "timestep": step,
            "campaign_id": campaign_id(plan, bound.design_key, cell.cell_id, stage),
            "config": configs[step],
            "config_sha": content_hash(asdict(configs[step])),
            "policy_sha": policy_identity(value, plan, bound.design_key, cell.cell_id, stage),
            "partial_checkpoint_prefix_count": plan.control_partial_checkpoint_prefix_count if stage == CONTROL else plan.partial_checkpoint_prefix_count,
        }
    return templates


def build_callbacks(value: Mapping[str, Any], plan: CampaignPlan, *, frozen: FrozenAuthority | None, collector: dict[str, Any]) -> RuntimeCallbacks:
    if (plan.kind == "evidentiary") != (frozen is not None):
        raise ValueError("evidentiary runs require frozen authorities; shakedowns forbid them")
    state: dict[str, Any] = {}
    tightness_floor = float(value["gates"]["minimum_certificate_dense_to_bound_ratio"])
    collector.setdefault("plan_kind", plan.kind)
    collector.setdefault("cases", {})
    collector.setdefault("validators", [])

    def prebundle(context: Any) -> Mapping[str, Any]:
        binding_report = source_binding_report(value)
        contract = binding_report["orbit_mc"]
        sweep = load_sweep_binding(value["field_source"])
        catalogue = load_bound_catalogue(value["cusp_cell_catalogue"])
        bound = bind_designs(value, sweep, catalogue, plan.design_keys)
        design_authorities = build_design_authorities(value, plan, bound)
        if frozen is not None:
            if semantic_sha256(value) != frozen.authorities["protocol_semantic_sha256"]:
                raise ValueError("protocol semantic authority differs")
            if frozen.design_authorities != design_authorities or semantic_sha256(frozen.design_authorities) != frozen.authorities["design_authorities_sha256"]:
                raise ValueError("design authorities differ from preregistration")
            if frozen.authorities["orbit_mc_source_sha256"] != contract["source_sha256"]:
                raise ValueError("orbit_mc source differs from preregistered authority")
            if frozen.authorities["field_pipeline_source_sha256"] != binding_report["field_pipeline_source_sha256"]:
                raise ValueError("field pipeline source differs from preregistered authority")
            if frozen.authorities["catalogue_file_sha256"] != catalogue.file_sha256:
                raise ValueError("catalogue differs from preregistered authority")
            if hashlib.sha256(frozen.shakedown_bytes).hexdigest() != frozen.authorities["shakedown_file_sha256"] or semantic_sha256(frozen.shakedown) != frozen.authorities["shakedown_semantic_sha256"]:
                raise ValueError("shakedown record differs from preregistered authority")
            verify_shakedown_record(value, frozen.shakedown, bind_designs(value, sweep, catalogue, shakedown_plan(value).design_keys))
        by_key = {item["case_key"]: item for item in design_authorities["stage1_cases"]}
        stage1_cases: dict[str, dict[str, dict[str, Any]]] = {}
        for key in plan.design_keys:
            stage1_cases[key] = {}
            for cell in bound[key].cells:
                authority = by_key[case_key(key, cell.cell_id, STAGE1)]
                campaign = campaign_id(plan, key, cell.cell_id, STAGE1)
                launches = block_launches(value, plan, bound[key], cell, 0)
                batches = batch_records(plan, launches)
                launch_bytes = canonical_bytes(runtime_launch_payload(campaign, launches))
                batch_bytes = canonical_bytes(runtime_batch_payload(campaign, batches))
                if (
                    authority["campaign_id"] != campaign
                    or hashlib.sha256(launch_bytes).hexdigest() != authority["runtime_launch_payload_byte_sha256"]
                    or hashlib.sha256(batch_bytes).hexdigest() != authority["runtime_batch_payload_byte_sha256"]
                    or load_runtime_launch_payload(launch_bytes, campaign) != tuple(sorted(launches, key=lambda item: item.launch_id))
                    or content_hash(launch_records(launches)) != authority["orbit_launches_sha256"]
                    or estimator_identity(launches, batches) != authority["estimator_sha256"]
                ):
                    raise ValueError(f"{campaign} stage-1 launch/batch authority differs")
                stage1_cases[key][cell.cell_id] = {"authority": authority, "launches": launches, "batches": batches}
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/design-authorities.json", design_authorities)
        context.write_json("artifacts/campaign-plan.json", plan_record(plan))
        context.write_json("artifacts/orbit-mc-contract.json", contract)
        context.write_json(
            "artifacts/field-pipeline-binding.json",
            {
                "field_pipeline_source_sha256": binding_report["field_pipeline_source_sha256"],
                "field_pipeline_source_files": binding_report["field_pipeline_source_files"],
                "sweep_manifest_file_sha256": sweep.manifest_file_sha256,
                "sweep_raw_results_file_sha256": sweep.raw_file_sha256,
                "sweep_summary_file_sha256": sweep.summary_file_sha256,
                "field_status": value["field_source"]["field_status"],
                "v1_reused_modules": binding_report["v1_reused_modules"],
            },
        )
        context.write_json(
            "artifacts/catalogue-binding.json",
            {
                "catalogue_file_sha256": catalogue.file_sha256,
                "manifest_file_sha256": catalogue.manifest_file_sha256,
                "declaration": value["cusp_cell_catalogue"],
                "design_count": catalogue.catalogue["design_count"],
                "cells_bound": {key: [cell.cell_id for cell in bound[key].cells] for key in plan.design_keys},
            },
        )
        if frozen is not None:
            context.write_json("artifacts/authorities.json", frozen.authorities)
            context.write_blob("artifacts/shakedown.json", frozen.shakedown_bytes)
        else:
            context.write_json("artifacts/shakedown-disclosure.json", {"evidentiary": False, "outcomes_enter_estimand": False, "statement": value["shakedown"]["purpose"], "disjointness": shakedown_disjointness(value, bound)})
        context.write_json(
            "artifacts/runtime.json",
            {"generated_at_utc": datetime.now(timezone.utc), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "cpu_count": os.cpu_count(), "worker_pool_size": worker_count(value), "backend": "numpy-cpu-relativistic-boris"},
        )
        state.update({"stage1_cases": stage1_cases, "bound": bound, "design_authorities": design_authorities, "sweep": sweep, "catalogue": catalogue})
        collector["prebundle"] = {"design_count": len(plan.design_keys), "stage1_case_count": design_authorities["stage1_case_count"], "orbit_mc_contract": contract, "catalogue_file_sha256": catalogue.file_sha256}
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "classification": CLASSIFICATION,
            "design_count": len(plan.design_keys),
            "cell_count": design_authorities["cell_count"],
            "stage1_launches": design_authorities["stage1_launches"],
            "case_sizes": design_authorities["case_sizes"],
            "orbit_mc_source_sha256": contract["source_sha256"],
            "field_pipeline_source_sha256": binding_report["field_pipeline_source_sha256"],
            "catalogue_file_sha256": catalogue.file_sha256,
        }

    def development(context: Any) -> Decision:
        started = time.perf_counter()
        workers = worker_count(value)
        manufactured = manufactured_gate_report(value)
        context.write_json("artifacts/manufactured-gates.json", manufactured)
        context.before_expensive(
            "resolve-all-designs",
            kind="solver",
            details={
                "design_count": len(plan.design_keys),
                "solver": "cft_revival.fields.solve_problem_cpu (sweep designs, accepted + 2x refined for every design); cft_orbit_wall_loss_v4.adapter.build_regular_field (P2 row, level-1 + level-2)",
                "worker_pool_size": workers,
                "plan_kind": plan.kind,
            },
        )
        tasks = [{"design_key": key, "set_id": state["bound"][key].design.set_id, "protocol": dict(value)} for key in plan.design_keys]
        stage_started = time.perf_counter()
        resolved = run_stage(tasks, resolve_design_task, workers)
        resolve_wall = time.perf_counter() - stage_started
        fields: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        exclusions: list[dict[str, Any]] = []
        for task, outcome in zip(tasks, resolved, strict=True):
            if outcome["design_key"] != task["design_key"]:
                raise RuntimeError("design results returned out of order")
            if outcome["status"] != "resolved":
                exclusions.append({"design_key": outcome["design_key"], "reason": outcome["reason"], "evidence": outcome.get("evidence")})
                continue
            key = outcome["design_key"]
            bound = state["bound"][key]
            if outcome["evidence"]["accepted_bore_field"]["source_identity_sha256"] != bound.design.accepted_field_identity:
                raise ValueError(f"{key}: resolved field identity differs from the bound design")
            if outcome["geometry"] is not None and outcome["geometry"] != bound.design.geometry:
                raise ValueError(f"{key}: resolved geometry differs from the bound design")
            fields[key] = outcome["accepted_field"]
            evidence[key] = outcome["evidence"]
            context.write_json(f"artifacts/fields/{key}.json", outcome["accepted_serialized"])
            context.write_json(f"artifacts/field-evidence/{key}.json", outcome["evidence"])
        context.write_json("artifacts/design-exclusions.json", {"schema_version": schema("design-exclusions"), "rule": value["designs"]["fallback_rule"], "excluded": exclusions})
        accepted = bool(manufactured["passed"] and fields and (plan.binding_gates or not exclusions))
        state.update({"manufactured": manufactured, "fields": fields, "field_evidence": evidence, "exclusions": exclusions})
        collector["development"] = {
            "manufactured_checks": manufactured["checks"],
            "resolved_design_count": len(fields),
            "exclusions": exclusions,
            "resolve_wall_s": resolve_wall,
            "seconds": time.perf_counter() - started,
            "accepted": accepted,
            "field_evidence_summary": {
                key: {
                    "interpolation_b_relative_rms": item["accepted_bore_field"]["interpolation_error_report"]["b_relative_rms"],
                    "cross_resolution_b_relative_rms": None if item.get("cross_resolution") is None else item["cross_resolution"]["b_relative_rms"],
                    "bore_max_b_t": item["accepted_bore_field"]["max_b_t"],
                    "passed": item["passed"],
                }
                for key, item in evidence.items()
            },
        }
        return Decision(accepted, {"manufactured_passed": manufactured["passed"], "resolved_design_count": len(fields), "excluded_design_count": len(exclusions), "resolve_wall_s": resolve_wall})

    def assessment(context: Any) -> Decision:
        started = time.perf_counter()
        workers = worker_count(value)
        main_ledger = ValidatorLedger()
        launch_rule = plan.launch_rule(value)
        allocation_rule = plan.allocation_rule(value)
        control_rule = {**dict(value["control"]), "fraction_per_cell": plan.control_fraction}
        design_tasks: list[dict[str, Any]] = []
        preregistration = {"protocol_id": value["schema_version"] if plan.binding_gates else f"{schema('shakedown')}:NON-EVIDENTIARY", "frozen_before_outcomes": True, "held_out_geometry_status": "pending"}
        for key in plan.design_keys:
            if key not in state["fields"]:
                continue
            bound = state["bound"][key]
            field = state["fields"][key]
            configs = {step: orbit_config(value, bound.design, step) for step in ("N", "2N")}
            bore_evidence = state["field_evidence"][key]["accepted_bore_field"]
            common = {
                "design_key": key,
                "field": field,
                "field_sha": bound.design.accepted_field_identity,
                "tightness_floor": tightness_floor,
                "work_dir": str(context.cache_root / "cases" / key),
                "field_evidence": {"field_error_report": bore_evidence["interpolation_error_report"]},
                "preregistration": preregistration,
            }
            stage1_cases: dict[str, dict[str, Any]] = {}
            cell_templates: dict[str, dict[str, dict[str, Any]]] = {}
            for cell in bound.cells:
                frozen_case = state["stage1_cases"][key][cell.cell_id]
                authority = frozen_case["authority"]
                if (
                    field.source_identity_sha256 != authority["field_identity_sha256"]
                    or content_hash(asdict(configs["N"])) != authority["config_identity_sha256"]
                    or policy_identity(value, plan, key, cell.cell_id, STAGE1) != authority["policy_identity_sha256"]
                    or estimator_identity(frozen_case["launches"], frozen_case["batches"]) != authority["estimator_sha256"]
                ):
                    raise ValueError(f"{key} {cell.cell_id} stage-1 execution authority differs")
                stage1_cases[cell.cell_id] = {
                    **common,
                    "case_key": case_key(key, cell.cell_id, STAGE1),
                    "cell_id": cell.cell_id,
                    "stage": STAGE1,
                    "timestep": "N",
                    "campaign_id": campaign_id(plan, key, cell.cell_id, STAGE1),
                    "authority": authority,
                    "launches": frozen_case["launches"],
                    "batches": frozen_case["batches"],
                    "config": configs["N"],
                    "config_sha": authority["config_identity_sha256"],
                    "policy_sha": authority["policy_identity_sha256"],
                    "launch_sha": authority["orbit_launches_sha256"],
                    "batch_sha": authority["batch_manifest_sha256"],
                    "partial_checkpoint_prefix_count": plan.partial_checkpoint_prefix_count,
                }
                cell_templates[cell.cell_id] = _cell_templates(value=value, plan=plan, bound=bound, cell=cell, common=common, configs=configs)
            design_tasks.append(
                {
                    "design_key": key,
                    "label": bound.design.label,
                    "plan": plan_record(plan) | {"design_keys": tuple(plan.design_keys)},
                    "launch_rule": launch_rule,
                    "allocation_rule": allocation_rule,
                    "control_rule": control_rule,
                    "cells": [cell.to_dict() for cell in bound.cells],
                    "wall_radius_m": bound.design.wall_radius_m,
                    "stage1_cases": stage1_cases,
                    "cell_templates": cell_templates,
                    "tightness_floor": tightness_floor,
                    "readiness_floor": float(value["estimators"]["surrogate_readiness_floor"]),
                    "maximum_relative_energy_error": value["gates"]["maximum_relative_energy_error"],
                    "field_adapter_passed": bool(state["field_evidence"][key]["passed"]),
                    "cpu_parity_passed": bool(state["manufactured"]["checks"]["cpu_parity"]),
                    "seal_policy": "converged" if plan.binding_gates else "structural",
                }
            )
        for task in design_tasks:
            for cell_id, stage1_case in task["stage1_cases"].items():
                context.before_expensive(
                    f"orbit-{stage1_case['case_key']}",
                    kind="label",
                    details={"campaign_id": stage1_case["campaign_id"], "launch_count": len(stage1_case["launches"]), "stage": STAGE1, "frozen_authority": True, "worker_pool_size": workers, "plan_kind": plan.kind},
                )
                for stage, template in task["cell_templates"][cell_id].items():
                    context.before_expensive(
                        f"orbit-{template['case_key']}",
                        kind="label",
                        details={
                            "campaign_id": template["campaign_id"],
                            "stage": stage,
                            "adaptive": True,
                            "rule": control_rule["statement"] if stage == CONTROL else allocation_rule["statement"],
                            "launch_count": "determined inside the worker by the frozen rule from the cell's stage-1 counts (stage 2) or by the frozen control seed (control); replayed by the main process",
                            "plan_kind": plan.kind,
                        },
                    )
        stage_started = time.perf_counter()
        design_outcomes = run_stage(design_tasks, run_design_full, workers)
        cases_wall = time.perf_counter() - stage_started
        publish_full = {key for key in plan.design_keys if state["bound"][key].design.representative}
        cases: dict[str, dict[str, Any]] = {}
        replays: dict[str, dict[str, Any]] = {}
        v1_rows = load_v1_dataset(value["v1_comparison"], REPOSITORY)
        dataset_rows: list[dict[str, Any]] = []
        design_gate_records: dict[str, Any] = {}
        allocation_records: dict[str, Any] = {}
        for design_task, design_outcome in zip(design_tasks, design_outcomes, strict=True):
            key = design_task["design_key"]
            if design_outcome["design_key"] != key:
                raise RuntimeError("design results returned out of order")
            bound = state["bound"][key]
            design_cases: dict[str, dict[str, Any]] = {}
            stage1_terminations: dict[str, str] = {}
            for outcome in design_outcome["cases"]:
                stage = outcome["stage"]
                cell = bound.cell(outcome["cell_id"])
                if stage == STAGE1:
                    task_case = design_task["stage1_cases"][cell.cell_id]
                    launches = task_case["launches"]
                    config = task_case["config"]
                    stage1_terminations.update(outcome["terminations"])
                else:
                    template = design_task["cell_templates"][cell.cell_id][stage]
                    config = template["config"]
                    launches = rebuild_adaptive_launches(value, plan, bound, cell, stage, outcome)
                ordered = sorted(launches, key=lambda item: item.launch_id)

                def determinism_sample(sample_launches: Sequence[Any] = ordered[:2], sample_field: Any = design_task["stage1_cases"][cell.cell_id]["field"], sample_config: Any = config, expected: Mapping[str, str] = outcome["determinism_hashes"]) -> dict[str, Any]:
                    compared = 0
                    for launch in sample_launches:
                        local = integrate_orbit(launch, sample_field, sample_config)
                        if content_hash(result_record(local)) != expected[launch.launch_id]:
                            raise RuntimeError(f"cross-process determinism differs for {launch.launch_id}")
                        compared += 1
                    return {"compared": compared, "passed": True}

                sample = main_ledger.run(outcome["case_key"], "cross_process_determinism_sample", determinism_sample)
                context.write_json(
                    f"artifacts/summaries/{outcome['case_key']}.json",
                    {
                        "classification": CLASSIFICATION,
                        "label": design_task["label"],
                        "design_key": key,
                        "cell_id": cell.cell_id,
                        "stage": stage,
                        "timestep": outcome["timestep"],
                        "campaign_id": outcome["campaign_id"],
                        "authority": outcome["authority"],
                        "summary": outcome["summary"],
                        "strata": outcome["strata"],
                        "preflight": outcome["preflight"],
                        "config": asdict(config),
                        "checkpoint_chain": outcome["checkpoints"],
                        "partial_checkpoint_file_sha256": outcome["partial_checkpoint_file_sha256"],
                        "final_checkpoint_file_sha256": outcome["final_checkpoint_file_sha256"],
                        "sealed": outcome["sealed"],
                        "orbit_artifact_file_sha256": outcome["artifact_file_sha256"],
                        "verified_file_sha256": outcome["verified_file_sha256"],
                        "endpoints_payload_sha256": outcome["endpoints_payload_sha256"],
                        "diagnostics": outcome["diagnostics"],
                        "gate_facts": outcome["gate_facts"],
                        "timing_s": outcome["timing_s"],
                        "worker_process_id": outcome["process_id"],
                        "determinism_sample": sample,
                    },
                )
                context.write_blob(f"artifacts/endpoints/{outcome['case_key']}.json.gz", outcome["endpoints_gz"])
                if outcome["sealed"]:
                    context.write_blob(f"artifacts/orbits/{outcome['case_key']}.json.sha256", outcome["artifact_sidecar_bytes"])
                    if key in publish_full:
                        context.write_blob(f"artifacts/orbits/{outcome['case_key']}.json.gz", gzip.compress(Path(outcome["artifact_path"]).read_bytes(), mtime=0))
                    context.write_json(f"artifacts/handoffs/{outcome['case_key']}.json", outcome["handoff"])
                design_cases[outcome["case_key"]] = {
                    **{k: outcome[k] for k in ("case_key", "campaign_id", "design_key", "cell_id", "stage", "timestep", "authority", "preflight", "summary", "strata", "diagnostics", "gate_facts", "sealed", "artifact_file_sha256", "verified_file_sha256", "handoff", "consumed_handoff", "endpoints_payload_sha256", "timing_s", "process_id", "terminations")},
                    "config": asdict(config),
                    "validators": list(outcome["validators"]),
                    "determinism_sample": sample,
                }
                cases[outcome["case_key"]] = design_cases[outcome["case_key"]]
                collector["cases"][outcome["case_key"]] = _collect_case(design_cases[outcome["case_key"]])
            replay = replay_allocation(value, plan, bound, design_outcome, stage1_terminations)
            replays[key] = replay
            stage1_frozen = all(design_cases[case_key(key, cell.cell_id, STAGE1)]["authority"] == design_task["stage1_cases"][cell.cell_id]["authority"] for cell in bound.cells)
            catalogue_cells = design_cells(catalogue_entry(state["catalogue"], bound.design.set_id, catalogue_design_id(bound.design)), injector_length_m=bound.design.injector_length_m, rule=value["launches"])
            catalogue_bound = [cell.to_dict() for cell in catalogue_cells] == design_task["cells"] == [{k: row[k] for k in design_task["cells"][0]} for row in design_outcome["cell_rows"]]
            gates = design_gates(value, state["field_evidence"][key], design_cases, design_outcome, replay, catalogue_bound, stage1_frozen, len(bound.cells), binding=plan.binding_gates)
            design_gate_records[key] = gates
            authority_row = next(row for row in state["design_authorities"]["designs"] if row["design_key"] == key)
            dataset_rows.append(dataset_row(value, authority_row, bound, state["field_evidence"][key], design_cases, design_outcome, replay, gates, v1_rows))
            allocation_records[key] = {
                "worker_decision": design_outcome["allocation"],
                "replay": replay,
                "control": {k: v for k, v in design_outcome["control"].items() if k not in ("per_cell", "selection")},
                "stage2_authority_sha256": {cell: [item["case_authority_sha256"] for item in items] for cell, items in design_outcome["stage2_authorities"].items()},
                "control_authority_sha256": {cell: item["case_authority_sha256"] for cell, item in design_outcome["control_authorities"].items()},
            }
        all_validators = list(main_ledger.records)
        for item in cases.values():
            all_validators.extend(item["validators"])
        validator_failures = [item for item in all_validators if not item["passed"]]
        structural_all = all(item["structural_passed"] for item in design_gate_records.values())
        # ---- pooled control gate (campaign-binding) ----------------------------
        pooled_n = sum(row["control"]["n_control"] for row in dataset_rows)
        pooled_wall_n = sum(row["control"]["wall_N"] for row in dataset_rows)
        pooled_wall_2n = sum(row["control"]["wall_2N"] for row in dataset_rows)
        pooled_discordant = sum(row["control"]["discordant"] for row in dataset_rows)
        pooled_delta = (pooled_wall_2n - pooled_wall_n) / pooled_n if pooled_n else None
        paired_diffs: list[float] = []
        for row in dataset_rows:
            n = row["control"]["n_control"]
            plus = row["control"]["wall_2N"] - min(row["control"]["wall_2N"], row["control"]["wall_N"])
            minus = row["control"]["wall_N"] - min(row["control"]["wall_2N"], row["control"]["wall_N"])
            paired_diffs.extend([1.0] * plus + [-1.0] * minus + [0.0] * (n - plus - minus))
        bias_se = (statistics.pstdev(paired_diffs) / math.sqrt(len(paired_diffs))) if len(paired_diffs) > 1 else None
        maximum_change = float(value["control"]["maximum_paired_probability_change"])
        control_gate = {
            "n_control": pooled_n,
            "wall_N": pooled_wall_n,
            "wall_2N": pooled_wall_2n,
            "p_wall_N": None if not pooled_n else pooled_wall_n / pooled_n,
            "p_wall_2N": None if not pooled_n else pooled_wall_2n / pooled_n,
            "estimated_bias_2N_minus_N": pooled_delta,
            "estimated_bias_standard_error": bias_se,
            "discordant": pooled_discordant,
            "discordance_rate": None if not pooled_n else pooled_discordant / pooled_n,
            "maximum_allowed_change": maximum_change,
            "passed": bool(pooled_n and abs(pooled_delta) <= maximum_change),
            "designs_with_control_flag_false": [row["design_key"] for row in dataset_rows if not row["convergence_flags"]["timestep_passed"]],
            "note": "the per-design flag decides sealing (quantum 1/n_control); the pooled paired difference is the campaign-binding gate and the N -> 2N bias estimate of the reported N values",
        }
        # ---- v1 comparison (reported, never gated) -----------------------------
        sweep_rows = [row for row in dataset_rows if row["set_id"] == SET_SWEEP and row["v1_comparison"] is not None]
        weights = ("wall_area", "launches")
        v1_comparison = {
            "schema_version": schema("v1-comparison"),
            "declaration": value["v1_comparison"],
            "design_count": len(sweep_rows),
            "statement": value["estimators"]["v1_comparison"],
            "spearman_rank_correlation": {w: spearman([row["v1_comparison"]["v1_probability"] for row in sweep_rows], [row["pooled"][w]["probability"] for row in sweep_rows]) for w in weights},
            "mean_difference_v2_minus_v1": {w: (statistics.fmean(row["v1_comparison"]["comparison"][w]["difference_v2_minus_v1"] for row in sweep_rows) if sweep_rows else None) for w in weights},
            "mean_absolute_difference": {w: (statistics.fmean(abs(row["v1_comparison"]["comparison"][w]["difference_v2_minus_v1"]) for row in sweep_rows) if sweep_rows else None) for w in weights},
            "interval_overlap_fraction": {w: (sum(row["v1_comparison"]["comparison"][w]["intervals_overlap"] for row in sweep_rows) / len(sweep_rows) if sweep_rows else None) for w in weights},
            "per_design": [{"design_key": row["design_key"], "v1": row["v1_comparison"]["v1_probability"], "v2_wall_area": row["pooled"]["wall_area"]["probability"], "v2_launches": row["pooled"]["launches"]["probability"]} for row in sweep_rows],
        }
        context.write_json("artifacts/v1-comparison.json", _plain(v1_comparison))
        # ---- consumer record ----------------------------------------------------
        consumer = consume_v4_export(value, REPOSITORY)
        consumer_record = {
            "schema_version": schema("coupling-consumer-record"),
            "consumer_id": value["coupling_consumer"]["consumer_id"],
            "classification": CLASSIFICATION,
            "v4_reference": consumer,
            "catalogue_consumed": {"catalogue_file_sha256": state["catalogue"].file_sha256, "designs": len(dataset_rows), "cells": sum(len(row["cells"]) for row in dataset_rows)},
            "screening_cases_consumed": [
                {
                    "design_key": row["design_key"],
                    "label": row["label"],
                    "case_key": key,
                    "cell_id": case["cell_id"],
                    "stage": case["stage"],
                    "sealed": case["sealed"],
                    "handoff_sha256": case["handoff_sha256"],
                    "probability": case["wall_hit"]["probability"],
                    "confidence_interval_95": [case["wall_hit"]["lower"], case["wall_hit"]["upper"]],
                    "trial_count": case["wall_hit"]["trials"],
                    "consumed": cases[key]["consumed_handoff"],
                    "consumption_status": "consumed_verified_handoff" if case["sealed"] else "not_consumable_unsealed_design (reported through summaries/endpoints only)",
                }
                for row in dataset_rows
                for key, case in row["cases"].items()
            ],
            "per_cell_statement": "every case is one cell block, so a stage-1 handoff carries that cell's block-0 estimate and a stage-2 handoff one of its top-up blocks; the dataset's per-cell values pool the N blocks of the cell and are cross-checked against the sealed strata",
            "statement": value["coupling_consumer"]["v4_absence_statement"],
        }
        context.write_json("artifacts/coupling-consumer-record.json", _plain(consumer_record))
        # ---- dataset --------------------------------------------------------------
        all_cells = [cell for row in dataset_rows for cell in row["cells"]]
        sweep_cells = [cell for row in dataset_rows if row["set_id"] == SET_SWEEP for cell in row["cells"]]
        design_wall = [row["pooled"]["wall_area"]["probability"] for row in dataset_rows if row["set_id"] == SET_SWEEP]
        headline = {
            "design_count": len(dataset_rows),
            "sweep_design_count": sum(row["set_id"] == SET_SWEEP for row in dataset_rows),
            "p2_row_present": any(row["set_id"] == SET_P2 for row in dataset_rows),
            "cell_count": len(all_cells),
            "sweep_cell_count": len(sweep_cells),
            "cells_topped_up": sum(cell["topped_up"] for cell in all_cells),
            "cells_saturated_after_stage1": sum(cell["saturated_after_stage1"] for cell in all_cells),
            "fraction_cells_saturated": (sum(cell["saturated_after_stage1"] for cell in all_cells) / len(all_cells)) if all_cells else None,
            "cells_surrogate_ready": sum(cell["final"]["surrogate_ready"] for cell in all_cells),
            "fraction_cells_surrogate_ready": (sum(cell["final"]["surrogate_ready"] for cell in all_cells) / len(all_cells)) if all_cells else None,
            "sweep_cells_surrogate_ready": sum(cell["final"]["surrogate_ready"] for cell in sweep_cells),
            "fraction_sweep_cells_surrogate_ready": (sum(cell["final"]["surrogate_ready"] for cell in sweep_cells) / len(sweep_cells)) if sweep_cells else None,
            "final_n_per_cell_counts": {str(n): sum(cell["final"]["trials"] == n for cell in all_cells) for n in sorted({cell["final"]["trials"] for cell in all_cells})},
            "jeffreys_floor_median": statistics.median(cell["final"]["jeffreys_floor"] for cell in all_cells) if all_cells else None,
            "jeffreys_floor_max": max((cell["final"]["jeffreys_floor"] for cell in all_cells), default=None),
            "stage1_launches": sum(row["launch_design"]["stage1_launches"] for row in dataset_rows),
            "stage2_launches": sum(row["launch_design"]["stage2_launches"] for row in dataset_rows),
            "control_launches": sum(row["launch_design"]["control_launches"] for row in dataset_rows),
            "total_orbits": sum(item["summary"]["trial_count"] for item in cases.values()),
            "per_cell_by_position": _position_class_summary(sweep_cells),
            "design_pooled_wall_area_min": min(design_wall, default=None),
            "design_pooled_wall_area_median": statistics.median(design_wall) if design_wall else None,
            "design_pooled_wall_area_max": max(design_wall, default=None),
            "least_wall_loss_design_keys": [row["design_key"] for row in sorted((r for r in dataset_rows if r["set_id"] == SET_SWEEP), key=lambda r: (r["pooled"]["wall_area"]["probability"], r["design_key"]))[:3]],
            "most_wall_loss_design_keys": [row["design_key"] for row in sorted((r for r in dataset_rows if r["set_id"] == SET_SWEEP), key=lambda r: (-r["pooled"]["wall_area"]["probability"], r["design_key"]))[:3]],
            "designs_with_reflections": [row["design_key"] for row in dataset_rows if row["diagnostics"]["reflections_final_n"] > 0],
            "total_reflections_final_n": sum(row["diagnostics"]["reflections_final_n"] for row in dataset_rows),
            "reflection_fraction_final_n": (sum(row["diagnostics"]["reflections_final_n"] for row in dataset_rows) / max(1, sum(row["launch_design"]["final_launches"] for row in dataset_rows))),
            "control": {k: control_gate[k] for k in ("n_control", "estimated_bias_2N_minus_N", "estimated_bias_standard_error", "discordant", "discordance_rate", "passed")},
            "control_flag_true_design_count": sum(row["convergence_flags"]["timestep_passed"] for row in dataset_rows),
            "sealed_design_count": sum(row["sealed"] for row in dataset_rows),
            "timeout_free_design_count": sum(item["timeout_free"] for item in design_gate_records.values()),
            "structural_gates_all_passed": structural_all,
            "allocation_replay_all_passed": all(item["passed"] for item in replays.values()),
            "v1_comparison": {k: v1_comparison[k] for k in ("spearman_rank_correlation", "mean_difference_v2_minus_v1", "mean_absolute_difference", "interval_overlap_fraction")},
            "p2_row": next(
                (
                    {"design_key": row["design_key"], "label": row["label"], "cells": [{"cell_id": c["cell_id"], "kind": c["kind"], "n": c["final"]["trials"], "p_wall": c["final"]["p_wall"]["probability"]} for c in row["cells"]], "pooled_wall_area": row["pooled"]["wall_area"]["probability"]}
                    for row in dataset_rows
                    if row["set_id"] == SET_P2
                ),
                None,
            ),
        }
        dataset = {
            "schema_version": schema("geometry-wall-loss-dataset-v2"),
            "classification": CLASSIFICATION,
            "classification_statement": value["classification_statement"],
            "claim_boundary": value["claim_boundary"],
            "plan_kind": plan.kind,
            "evidentiary": plan.binding_gates,
            "generated_at_utc": datetime.now(timezone.utc),
            "protocol_semantic_sha256": semantic_sha256(value),
            "orbit_mc_source_sha256": collector["prebundle"]["orbit_mc_contract"]["source_sha256"],
            "field_pipeline_source_sha256": field_pipeline_source_sha256(),
            "catalogue_file_sha256": state["catalogue"].file_sha256,
            "field_source": {
                "experiment": value["field_source"]["experiment"],
                "field_status": value["field_source"]["field_status"],
                "manifest_file_sha256": value["field_source"]["manifest_file_sha256"],
                "raw_results_file_sha256": value["field_source"]["raw_results_file_sha256"],
                "refined_diagnostic_coverage": value["field_source"]["refined_diagnostic"]["coverage"],
            },
            "cusp_cell_catalogue": {k: value["cusp_cell_catalogue"][k] for k in ("experiment", "experiment_id", "catalogue_file_sha256", "manifest_file_sha256", "result_commit", "cell_definition", "label_statement")},
            "launch_design": {k: value["launches"][k] for k in ("stratification_statement", "radius_rule", "launch_plane_statement", "stage1_launches_per_cell", "final_launches_per_topped_up_cell", "sobol", "launch_id_rule")},
            "case_structure": value["cases"],
            "allocation_rule": value["allocation"]["statement"],
            "control_rule": value["control"]["statement"],
            "estimators": value["estimators"],
            "orbit_geometry_rule": value["orbit_geometry_rule"],
            "design_count": len(dataset_rows),
            "cell_count": len(all_cells),
            "excluded_designs": state["exclusions"],
            "reported_case": value["cases"]["reported_probability_case"],
            "headline": headline,
            "control_gate": control_gate,
            "designs": dataset_rows,
        }
        context.write_json("artifacts/geometry-wall-loss-dataset-v2.json", _plain(dataset))
        csv_bytes = dataset_csv(dataset_rows)
        context.write_blob("artifacts/geometry-wall-loss-dataset-v2.csv", csv_bytes)
        context.write_json(
            "artifacts/allocation-decisions.json",
            _plain(
                {
                    "schema_version": schema("allocation-decisions"),
                    "rule": allocation_rule,
                    "control_rule": control_rule,
                    "evaluated_by": "run_design_full (worker, per cell) and replay_allocation (main process, from the stage-1 endpoint terminations)",
                    "designs": allocation_records,
                    "summary": {
                        "cells": len(all_cells),
                        "topped_up_cells": headline["cells_topped_up"],
                        "saturated_cells": headline["cells_saturated_after_stage1"],
                        "stage2_launches": headline["stage2_launches"],
                        "control_launches": headline["control_launches"],
                        "replay_all_passed": headline["allocation_replay_all_passed"],
                    },
                }
            ),
        )
        gates = {
            "binding": plan.binding_gates,
            "manufactured": state["manufactured"]["checks"],
            "per_design": design_gate_records,
            "structural_all_passed": structural_all,
            "allocation_replay_all_passed": headline["allocation_replay_all_passed"],
            "control_gate": {k: control_gate[k] for k in ("n_control", "estimated_bias_2N_minus_N", "discordant", "maximum_allowed_change", "passed")},
            "control_flag_true_design_count": headline["control_flag_true_design_count"],
            "timeout_free_design_count": headline["timeout_free_design_count"],
            "design_count": len(design_gate_records),
            "validator_failures": len(validator_failures),
            "exact_authority_replay_count": sum(item["sealed"] and item["artifact_file_sha256"] == item["verified_file_sha256"] for item in cases.values()),
            "sealed_case_count": sum(item["sealed"] for item in cases.values()),
            "case_count": len(cases),
            "diagnostics_not_gates": {"magnetic_moment_variation": "per design in the dataset; never a gate", "v1_comparison": "reported, never gated", "timeouts": "reported, never gated"},
            "passed": bool(structural_all and state["manufactured"]["passed"] and not validator_failures and cases and control_gate["passed"]),
        }
        context.write_json("artifacts/gates.json", _plain(gates))
        if plan.binding_gates:
            accepted = bool(gates["passed"])
            status = "accepted_screening_dataset" if accepted else "rejected"
        else:
            accepted = bool(gates["passed"] and not state["exclusions"] and headline["cells_topped_up"] > 0 and headline["cells_saturated_after_stage1"] > 0)
            status = "shakedown_passed" if accepted else "shakedown_failed"
        terminal = {
            "status": status,
            "plan_kind": plan.kind,
            "evidentiary": plan.binding_gates,
            "classification": CLASSIFICATION,
            "design_count": len(dataset_rows),
            "excluded_design_count": len(state["exclusions"]),
            "case_count": len(cases),
            "orbit_count": headline["total_orbits"],
            "headline": headline,
            "gates": _plain(gates),
            "validators": {"passed": sum(item["passed"] for item in all_validators), "failed": len(validator_failures)},
            "execution_mode": {"parallel_designs": workers > 1, "worker_pool_size": workers, "cases_wall_s": cases_wall, "assessment_wall_s": time.perf_counter() - started},
            "coupling": "consumer_record_published",
            "limitations": value["claim_boundary"],
        }
        context.write_json("artifacts/campaign-result.json", _plain(terminal))
        collector["assessment"] = {
            "gates": _plain(gates),
            "execution_mode": terminal["execution_mode"],
            "status": status,
            "accepted": accepted,
            "headline": _plain(headline),
            "control_gate": _plain(control_gate),
            "dataset_summary": {"design_count": len(dataset_rows), "cell_count": len(all_cells), "csv_bytes": len(csv_bytes), "consumer_v4_passed": consumer["passed"]},
            "allocation_summary": {"topped_up_cells": headline["cells_topped_up"], "saturated_cells": headline["cells_saturated_after_stage1"], "replay_all_passed": headline["allocation_replay_all_passed"]},
        }
        collector["validators"] = all_validators
        collector["design_gates"] = design_gate_records
        return Decision(accepted, _plain(terminal))

    return RuntimeCallbacks(prebundle, development, assessment)


def rebuild_adaptive_launches(value: Mapping[str, Any], plan: CampaignPlan, bound: BoundCells, cell: LaunchCell, stage: str, outcome: Mapping[str, Any]) -> tuple[ElectronLaunch, ...]:
    """Rebuild an adaptive case's launches from its realised terminations for the determinism sample; bind to its authority."""

    authority = outcome["authority"]
    keys = sorted(outcome["terminations"])
    if stage == CONTROL:
        launches = control_launches(value, plan, bound, cell, keys)
    else:
        launches = block_launches(value, plan, bound, cell, stage_block(stage))
    if content_hash(launch_records(launches)) != authority["orbit_launches_sha256"] or len(launches) != authority["launch_count"] or sorted(key_of_launch(item) for item in launches) != keys:
        raise ValueError(f"{outcome['case_key']}: rebuilt launches differ from the realised authority")
    return launches


def _collect_case(case: Mapping[str, Any]) -> dict[str, Any]:
    validators = case["validators"]
    return {
        "campaign_id": case["campaign_id"],
        "design_key": case["design_key"],
        "cell_id": case["cell_id"],
        "stage": case["stage"],
        "timestep": case["timestep"],
        "launch_count": case["summary"]["trial_count"],
        "preflight": case["preflight"],
        "diagnostics": case["diagnostics"],
        "gate_facts": case["gate_facts"],
        "timing_s": dict(case["timing_s"]),
        "validators": {"passed": sum(item["passed"] for item in validators), "failed": sum(not item["passed"] for item in validators), "failures": [item for item in validators if not item["passed"]], "names": [item["validator"] for item in validators]},
        "export_stage_ran": bool(case["sealed"]),
        "handoff_consumed": bool(case["consumed_handoff"] is not None and case["consumed_handoff"]["passed"]),
        "artifact_file_sha256": case["artifact_file_sha256"],
        "determinism_sample": case["determinism_sample"],
        "worker_process_id": case["process_id"],
        "summary": case["summary"],
    }
