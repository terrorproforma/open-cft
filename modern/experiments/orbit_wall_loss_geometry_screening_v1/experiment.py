"""Wall-loss-vs-geometry screening campaign mechanics (orbit_mc v1.7 on L1a sweep-v2 fields).

Classification: ``SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS``. The fields are the
accepted L1a linear-vacuum equivalent-current maps of the geometry sweep v2, not
P2-qualified FEM; nothing here is accepted physical-orbit evidence.

The mechanics follow the accepted v4 template (``experiments.cft_orbit_wall_loss_v4``):
one :class:`CampaignPlan` drives the evidentiary campaign and the disclosed
NON-EVIDENTIARY shakedown; the shakedown must pass on real re-solved fields
before ``prepare`` freezes the authorities; a single detached execution publishes
through the shared :class:`ExperimentRuntime`. Generic pieces are imported from
v4 with attribution (validator ledger, per-case integration/export workers,
stratum summaries, diagnostics, orbit_mc source/contract binding, runtime tag
decoding); the P2 adapter is replaced by :mod:`.designs` (L1a re-solve) and the
case matrix is per design.
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

import cft_revival.orbit_mc as orbit_mc_package
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
    backend_parity,
    analytic_magnetic_bottle,
    build_launch_ensemble,
    coupling_v42_handoff,
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
    gyrophase_grid,
    launch_records,
    mu_diagnostic,
    orbit_mc_contract_report as _v4_orbit_mc_contract_report,
    orbit_mc_source_files,
    orbit_mc_source_sha256,
    result_diagnostics,
    result_record,
    run_case_export,
    run_case_integration,
    run_stage,
    stratum_summaries,
)

from . import designs as design_module
from .consumer import consume_handoff, consume_v4_export, verify_handoff
from .designs import (
    DesignGeometry,
    SweepBinding,
    design_geometry,
    field_identity,
    field_pipeline_source_sha256,
    launch_cells,
    launch_positions,
    load_sweep_binding,
    orbit_config_for,
    rebuild_case,
    resolve_design,
)

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
DESIGN_AUTHORITIES_PATH = EXPERIMENT / "design-authorities.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"

VERSION_TAG = "cft-revival.orbit-wall-loss-geometry-screening-v1"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
TIMESTEPS = ("N", "2N")
ROLES = ("accepted", "refined")
NUMERICAL_FAILURES = (
    Termination.STEP_LIMIT,
    Termination.NONFINITE_STATE,
    Termination.EXTREME_RELATIVITY,
    Termination.FIELD_FAILURE,
    Termination.INITIAL_STATE_INVALID,
)
TIMEOUTS = (Termination.TIME_TIMEOUT, Termination.PATH_TIMEOUT)
ESCAPE_TOLERANCE_M = 1.0e-8


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/1.0.0"


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    if value["classification"] != CLASSIFICATION:
        raise ValueError("protocol classification must be the screening label")
    return value


# --------------------------------------------------------------------------
# orbit_mc + field pipeline contract binding
# --------------------------------------------------------------------------


def orbit_mc_contract_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """v4 contract report re-keyed on this protocol (same observed quantities)."""

    return _v4_orbit_mc_contract_report(value)


def require_orbit_mc_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    report = orbit_mc_contract_report(value)
    if not report["matches"]:
        raise ValueError(
            "orbit_mc contract (package version / schema versions) differs from protocol: "
            f"expected {report['expected']}, observed {report['observed']}"
        )
    return report


EXPERIMENT_CODE_FILES = ("consumer.py", "designs.py", "experiment.py", "run.py", "__init__.py")


def experiment_code_sha256() -> str:
    """SHA-256 over the LF bytes of this experiment's own code (stricter than v4)."""

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
    return {
        "orbit_mc": contract,
        "field_pipeline_source_sha256": field_pipeline_source_sha256(),
        "field_pipeline_source_files": [
            path.relative_to(MODERN).as_posix()
            for path in design_module.field_pipeline_source_files()
        ],
        "experiment_code_sha256": experiment_code_sha256(),
        "experiment_code_files": list(EXPERIMENT_CODE_FILES),
    }


# --------------------------------------------------------------------------
# Campaign plans
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPlan:
    kind: str
    campaign_id_prefix: str
    case_ids: tuple[str, ...]
    gyrophases_rad: tuple[float, ...]
    batch_size: int
    partial_checkpoint_prefix_count: int
    launches_per_case: int
    batches_per_case: int
    strata_per_case: int
    independent_repeats_per_stratum: int
    binding_gates: bool

    def __post_init__(self) -> None:
        if self.kind not in ("evidentiary", "shakedown"):
            raise ValueError("unknown campaign plan kind")
        if not self.campaign_id_prefix or ":" in self.campaign_id_prefix:
            raise ValueError("campaign_id_prefix must be non-empty and colon-free")
        if len(set(self.case_ids)) != len(self.case_ids) or not self.case_ids:
            raise ValueError("plan designs must be unique and non-empty")
        if len(set(self.gyrophases_rad)) != len(self.gyrophases_rad):
            raise ValueError("plan gyrophases must be unique")
        if not 0 < self.partial_checkpoint_prefix_count < self.batch_size:
            raise ValueError("partial prefix must lie strictly inside batch 0")


def design_case_ids(value: Mapping[str, Any]) -> tuple[str, ...]:
    declaration = value["designs"]
    ids = list(declaration["primary_case_ids"])
    if declaration["extension_batch_included"]:
        ids.extend(declaration["extension_case_ids"])
    if len(set(ids)) != len(ids):
        raise ValueError("design batches overlap")
    return tuple(sorted(ids))


def design_batch(value: Mapping[str, Any], case_id: str) -> str:
    if case_id in value["designs"]["primary_case_ids"]:
        return "primary"
    if case_id in value["designs"]["extension_case_ids"]:
        return "extension"
    raise ValueError(f"{case_id} is not a declared design")


def evidentiary_plan(value: Mapping[str, Any]) -> CampaignPlan:
    declaration = value["launches"]
    return CampaignPlan(
        kind="evidentiary",
        campaign_id_prefix=declaration["campaign_id_prefix"],
        case_ids=design_case_ids(value),
        gyrophases_rad=gyrophase_grid(
            declaration["gyrophase_offset_rad"], declaration["gyrophase_count"]
        ),
        batch_size=int(declaration["batch_size"]),
        partial_checkpoint_prefix_count=int(
            value["execution"]["partial_checkpoint_prefix_count"]
        ),
        launches_per_case=int(declaration["launches_per_case"]),
        batches_per_case=int(declaration["batches_per_case"]),
        strata_per_case=int(declaration["strata_per_case"]),
        independent_repeats_per_stratum=int(declaration["independent_repeats_per_stratum"]),
        binding_gates=True,
    )


def shakedown_plan(value: Mapping[str, Any]) -> CampaignPlan:
    declaration = value["shakedown"]
    if declaration["evidentiary"] is not False or declaration["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown must be declared non-evidentiary")
    ids = tuple(declaration["design_case_ids"])
    if any(case_id not in design_case_ids(value) for case_id in ids):
        raise ValueError("shakedown designs must be declared designs")
    return CampaignPlan(
        kind="shakedown",
        campaign_id_prefix=declaration["campaign_id_prefix"],
        case_ids=tuple(sorted(ids)),
        gyrophases_rad=gyrophase_grid(
            declaration["gyrophase_offset_rad"], declaration["gyrophase_count"]
        ),
        batch_size=int(declaration["batch_size"]),
        partial_checkpoint_prefix_count=int(declaration["partial_checkpoint_prefix_count"]),
        launches_per_case=int(declaration["launches_per_case"]),
        batches_per_case=int(declaration["batches_per_case"]),
        strata_per_case=int(declaration["strata_per_case"]),
        independent_repeats_per_stratum=int(declaration["independent_repeats_per_stratum"]),
        binding_gates=False,
    )


def plan_record(plan: CampaignPlan) -> dict[str, Any]:
    record = asdict(plan)
    record["case_ids"] = list(plan.case_ids)
    record["gyrophases_rad"] = list(plan.gyrophases_rad)
    return record


def shakedown_positions(
    value: Mapping[str, Any], geometry: DesignGeometry
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Deterministic RNG positions per design in a namespace disjoint from the design."""

    declaration = value["shakedown"]
    seed = int.from_bytes(
        hashlib.sha256(
            f"{declaration['seed_namespace']}:{geometry.case_id}:positions".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    low, high = (float(item) for item in declaration["radius_fraction_range"])
    span = geometry.exit_start_m - geometry.injector_length_m
    half_width = float(declaration["axial_half_width_fraction_of_straight_span"]) * span
    repeats = int(declaration["positions_per_cell"])
    labels = "abcdefgh"
    if not 0 < repeats <= len(labels):
        raise ValueError("positions_per_cell is out of range")
    positions: list[tuple[str, tuple[float, float, float]]] = []
    for cell in launch_cells(geometry, value["launches"]):
        for index in range(repeats):
            fraction = float(rng.uniform(low, high))
            axial_offset = float(rng.uniform(-half_width, half_width))
            positions.append(
                (
                    f"sd-{cell['cell_id']}-r{fraction:.6f}-{labels[index]}",
                    (fraction * geometry.wall_radius_m, 0.0, cell["axial_center_m"] + axial_offset),
                )
            )
    return tuple(positions)


def plan_positions(
    value: Mapping[str, Any], plan: CampaignPlan, geometry: DesignGeometry
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if plan.kind == "evidentiary":
        return launch_positions(geometry, value["launches"])
    return shakedown_positions(value, geometry)


# --------------------------------------------------------------------------
# Case matrix, identities, launches, payloads
# --------------------------------------------------------------------------


def case_roles(value: Mapping[str, Any], case_id: str) -> tuple[tuple[str, str], ...]:
    roles = value["cases"]["roles_per_design"]
    matrix = [("accepted", step) for step in roles["accepted"]]
    if case_id in value["designs"]["representative_case_ids"]:
        matrix.extend(("refined", step) for step in roles["refined"])
    return tuple(matrix)


def case_key(case_id: str, role: str, timestep: str) -> str:
    if role not in ROLES or timestep not in TIMESTEPS:
        raise ValueError("unknown role or timestep")
    return f"{case_id}--{role}-{timestep}"


def campaign_id(plan: CampaignPlan, case_id: str, role: str, timestep: str) -> str:
    return f"{plan.campaign_id_prefix}:{case_id}:{role}:{timestep}"


def case_matrix(
    value: Mapping[str, Any], plan: CampaignPlan
) -> tuple[tuple[str, str, str, str, str], ...]:
    """(case_id, role, timestep, campaign_id, case_key) for every case of the plan."""

    return tuple(
        (case_id, role, timestep, campaign_id(plan, case_id, role, timestep), case_key(case_id, role, timestep))
        for case_id in plan.case_ids
        for role, timestep in case_roles(value, case_id)
    )


def build_case_launches(
    value: Mapping[str, Any], plan: CampaignPlan, geometry: DesignGeometry, role: str, timestep: str
) -> tuple[ElectronLaunch, ...]:
    declaration = value["launches"]
    launches = build_launch_ensemble(
        ensemble_id=campaign_id(plan, geometry.case_id, role, timestep),
        energies_ev=declaration["energies_ev"],
        pitch_angles_rad=[math.radians(item) for item in declaration["pitch_angles_deg"]],
        positions=plan_positions(value, plan, geometry),
        directions=declaration["directions"],
        gyrophases_rad=plan.gyrophases_rad,
    )
    if len(launches) != plan.launches_per_case:
        raise ValueError(
            f"{plan.kind} plan produced {len(launches)} launches, expected {plan.launches_per_case}"
        )
    return launches


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
    batches = frozen_batch_manifest(
        launches, batch_size=plan.batch_size, estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL
    )
    if len(batches) != plan.batches_per_case:
        raise ValueError(
            f"{plan.kind} plan produced {len(batches)} batches, expected {plan.batches_per_case}"
        )
    return batches


def runtime_batch_payload(plan: CampaignPlan, campaign: str, launches: Sequence[Any]) -> dict[str, Any]:
    return {
        "schema_version": schema("batches"),
        "campaign_id": campaign,
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "batches": batch_records(plan, launches),
    }


def load_runtime_launch_payload(data: bytes, expected_campaign_id: str) -> tuple[ElectronLaunch, ...]:
    """Closed typed loader (v4 logic, this campaign's schema tag)."""

    decoded = _decode_runtime_tags(strict_json_loads(data))
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema_version", "campaign_id", "ensemble_id", "estimator_policy", "seed_encoding", "launches",
    }:
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
    expected_keys = {
        "launch_id", "seed_id", "kinetic_energy_ev", "pitch_angle_rad", "position_m",
        "parallel_direction", "gyrophase_rad", "flux_surface_id",
    }
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


def policy_identity(value: Mapping[str, Any], plan: CampaignPlan, case_id: str, role: str, timestep: str) -> str:
    return content_hash(
        {
            "protocol_semantic_sha256": semantic_sha256(value),
            "plan_kind": plan.kind,
            "case_id": case_id,
            "role": role,
            "timestep": timestep,
        }
    )


def orbit_config(value: Mapping[str, Any], geometry: DesignGeometry, timestep: str) -> OrbitConfig:
    rule = value["orbit_geometry_rule"]
    return orbit_config_for(geometry, rule, rule["timestep_policies"][timestep])


@dataclass(frozen=True)
class BoundDesign:
    """A design's sweep case and geometry (no field solve)."""

    case_id: str
    geometry: DesignGeometry
    case_sha256: str
    geometry_sha256: str
    source_sha256: str
    config_sha256: str
    design_id: str
    design_values: dict[str, float]
    accepted_field_identity: str
    refined_field_identity: str


def bind_designs(value: Mapping[str, Any], binding: SweepBinding, case_ids: Sequence[str]) -> dict[str, BoundDesign]:
    declaration = value["field_source"]
    output: dict[str, BoundDesign] = {}
    for case_id in case_ids:
        case = rebuild_case(binding, case_id)
        output[case_id] = BoundDesign(
            case_id=case_id,
            geometry=design_geometry(case),
            case_sha256=case.case_sha256,
            geometry_sha256=case.geometry_sha256,
            source_sha256=case.source_sha256,
            config_sha256=case.config_sha256,
            design_id=case.design.design_id,
            design_values=design_module.sweep.design_values(case.design),
            accepted_field_identity=field_identity(case, declaration, "accepted"),
            refined_field_identity=field_identity(case, declaration, "refined"),
        )
    return output


def build_case_authority(
    value: Mapping[str, Any], plan: CampaignPlan, bound: BoundDesign, role: str, timestep: str
) -> dict[str, Any]:
    campaign = campaign_id(plan, bound.case_id, role, timestep)
    launches = build_case_launches(value, plan, bound.geometry, role, timestep)
    batches = batch_records(plan, launches)
    launch_bytes = canonical_bytes(runtime_launch_payload(campaign, launches))
    batch_bytes = canonical_bytes(runtime_batch_payload(plan, campaign, launches))
    config = orbit_config(value, bound.geometry, timestep)
    field_sha = bound.accepted_field_identity if role == "accepted" else bound.refined_field_identity
    record = {
        "schema_version": schema("case-authority"),
        "plan_kind": plan.kind,
        "case_key": case_key(bound.case_id, role, timestep),
        "campaign_id": campaign,
        "ensemble_id": campaign,
        "case_id": bound.case_id,
        "role": role,
        "timestep": timestep,
        "launch_count": len(launches),
        "batch_count": len(batches),
        "runtime_launch_payload_byte_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "runtime_batch_payload_byte_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "orbit_launches_sha256": content_hash(launch_records(launches)),
        "batch_manifest_sha256": content_hash(
            {"estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value, "batches": batches}
        ),
        "estimator_sha256": estimator_identity(launches, batches),
        "field_identity_sha256": field_sha,
        "config": asdict(config),
        "config_identity_sha256": content_hash(asdict(config)),
        "policy_identity_sha256": policy_identity(value, plan, bound.case_id, role, timestep),
    }
    record["case_authority_sha256"] = content_hash(
        {
            "campaign_id": campaign,
            "launches_sha256": record["orbit_launches_sha256"],
            "batch_manifest_sha256": record["batch_manifest_sha256"],
            "policy_sha256": record["policy_identity_sha256"],
            "minimum_certificate_tightness_ratio": value["gates"]["minimum_certificate_dense_to_bound_ratio"],
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "estimator_sha256": record["estimator_sha256"],
            "replay_requirement": "deterministic_full_result_replay_required",
        }
    )
    record["case_authority_record_sha256"] = content_hash(record)
    return record


def build_design_authorities(
    value: Mapping[str, Any], plan: CampaignPlan, bound_designs: Mapping[str, BoundDesign]
) -> dict[str, Any]:
    """Frozen per-design and per-case authority (no field solve, no outcomes)."""

    design_rows = []
    case_rows = []
    for case_id in plan.case_ids:
        bound = bound_designs[case_id]
        positions = plan_positions(value, plan, bound.geometry)
        design_rows.append(
            {
                "case_id": case_id,
                "design_id": bound.design_id,
                "batch": design_batch(value, case_id),
                "representative": case_id in value["designs"]["representative_case_ids"],
                "design_values": bound.design_values,
                "geometry": bound.geometry.to_dict(),
                "geometry_sha256": bound.geometry_sha256,
                "source_sha256": bound.source_sha256,
                "config_sha256": bound.config_sha256,
                "case_sha256": bound.case_sha256,
                "accepted_field_identity_sha256": bound.accepted_field_identity,
                "refined_field_identity_sha256": bound.refined_field_identity,
                "cells": launch_cells(bound.geometry, value["launches"]),
                "positions": [
                    {"flux_surface_id": surface, "position_m": list(position)}
                    for surface, position in positions
                ],
                "roles": [list(item) for item in case_roles(value, case_id)],
            }
        )
        for role, timestep in case_roles(value, case_id):
            case_rows.append(build_case_authority(value, plan, bound, role, timestep))
    return {
        "schema_version": schema("design-authorities"),
        "plan_kind": plan.kind,
        "protocol_semantic_sha256": semantic_sha256(value),
        "design_count": len(design_rows),
        "case_count": len(case_rows),
        "total_launches": sum(item["launch_count"] for item in case_rows),
        "designs": design_rows,
        "cases": case_rows,
    }


def all_plan_launches(
    value: Mapping[str, Any], plan: CampaignPlan, bound_designs: Mapping[str, BoundDesign]
) -> tuple[ElectronLaunch, ...]:
    return tuple(
        launch
        for case_id, role, timestep, _, _ in case_matrix(value, plan)
        for launch in build_case_launches(value, plan, bound_designs[case_id].geometry, role, timestep)
    )


def design_sha256(value: Mapping[str, Any], plan: CampaignPlan, bound_designs: Mapping[str, BoundDesign]) -> str:
    return content_hash(launch_records(all_plan_launches(value, plan, bound_designs)))


# --------------------------------------------------------------------------
# Disjointness (shakedown vs evidentiary on the same designs; vs v1-v4 by construction)
# --------------------------------------------------------------------------


def _signature(launches: Sequence[ElectronLaunch]) -> dict[str, set[Any]]:
    return {
        "launch_id": {item.launch_id for item in launches},
        "seed_id": {item.seed_id for item in launches},
        "position_m": {item.position_m for item in launches},
        "energy_pitch_direction_gyrophase": {
            (item.kinetic_energy_ev, item.pitch_angle_rad, item.parallel_direction, item.gyrophase_rad)
            for item in launches
        },
    }


def disjointness_report(left: Sequence[ElectronLaunch], right: Sequence[ElectronLaunch], *, left_name: str, right_name: str) -> dict[str, Any]:
    ls, rs = _signature(left), _signature(right)
    overlaps = {name: len(ls[name] & rs[name]) for name in ls}
    return {
        "left": left_name,
        "right": right_name,
        "left_launch_count": len(left),
        "right_launch_count": len(right),
        "overlap_counts": overlaps,
        "disjoint": all(count == 0 for count in overlaps.values()),
    }


def _prior_gyrophase_offsets(value: Mapping[str, Any]) -> dict[str, float]:
    disclosure = value["prior_campaign_disclosure"]
    return {
        "v1_v2": 0.0,
        "v3": math.pi / 16.0,
        "v4": float(disclosure["v4"]["gyrophase_offset_rad"]),
        "v4_shakedown": float(disclosure["v4"]["shakedown_gyrophase_offset_rad"]),
    }


def gyrophase_grid_disjointness(value: Mapping[str, Any]) -> dict[str, Any]:
    """Offsets differ mod pi/4 (8-point grids) from every prior campaign and each other."""

    count = int(value["launches"]["gyrophase_count"])
    period = 2.0 * math.pi / count
    offsets = dict(_prior_gyrophase_offsets(value))
    offsets["evidentiary"] = float(value["launches"]["gyrophase_offset_rad"])
    offsets["shakedown"] = float(value["shakedown"]["gyrophase_offset_rad"])
    names = sorted(offsets)
    pairs = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            separation = abs((offsets[left] - offsets[right]) % period)
            separation = min(separation, period - separation)
            pairs[f"{left}|{right}"] = separation
    return {"offsets_rad": offsets, "minimum_separation_mod_period_rad": min(pairs.values()), "pairs": pairs, "disjoint": min(pairs.values()) > 1.0e-9}


def shakedown_disjointness(value: Mapping[str, Any], bound_designs: Mapping[str, BoundDesign]) -> dict[str, Any]:
    shakedown = all_plan_launches(value, shakedown_plan(value), bound_designs)
    evidentiary_same_designs = tuple(
        launch
        for case_id, role, timestep, _, _ in case_matrix(value, shakedown_plan(value))
        for launch in build_case_launches(value, evidentiary_plan(value), bound_designs[case_id].geometry, role, timestep)
    )
    reports = {
        "against_evidentiary_same_designs": disjointness_report(
            shakedown, evidentiary_same_designs, left_name="shakedown", right_name="evidentiary-same-designs"
        )
    }
    grids = gyrophase_grid_disjointness(value)
    return {
        "shakedown_launch_count": len(shakedown),
        "shakedown_unique_launch_ids": len({item.launch_id for item in shakedown}),
        "shakedown_unique_seed_ids": len({item.seed_id for item in shakedown}),
        "reports": reports,
        "gyrophase_grids": grids,
        "against_v1_v2_v3_v4": "disjoint by construction: prior campaigns launched only in the divergent-exit-stack P2 field at positions/ids/seeds outside every sweep-v2 design; gyrophase grids are disjoint mod pi/4 (see gyrophase_grids)",
        "proven": (
            len({item.launch_id for item in shakedown}) == len(shakedown)
            and len({item.seed_id for item in shakedown}) == len(shakedown)
            and all(item["disjoint"] for item in reports.values())
            and grids["disjoint"]
        ),
    }


# --------------------------------------------------------------------------
# Manufactured gates (CPU only) and per-case gate facts
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
        "cpu_parity": cpu["status"] == "evaluated"
        and cpu["maximum_relative_velocity_difference"] <= limits["maximum_cpu_cuda_relative_velocity_difference"],
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
    """Compact per-case facts so the main process never needs the orbit results."""

    witness_order = all(
        item.event_witness["event_fraction"]
        <= min(
            [c for c in item.event_witness["candidate_fractions"].values() if c is not None]
            or [item.event_witness["event_fraction"]]
        )
        + 64.0 * np.finfo(float).eps
        for item in results
        if "candidate_fractions" in item.event_witness
    )
    wall_errors = [
        abs(math.hypot(*item.wall_endpoint_m[:2]) - config.wall_radius_m)
        for item in results
        if item.wall_endpoint_m is not None
    ]
    subclasses: dict[str, int] = {}
    for item in results:
        label = escape_subclass(item, config)
        if label is not None:
            subclasses[label] = subclasses.get(label, 0) + 1
    return {
        "earliest_event_ordering": bool(witness_order),
        "runtime_rotation_bound": all(
            item.dt_s * abs(ELECTRON_CHARGE_C) * field.max_b_t / ELECTRON_MASS_KG
            <= config.max_rotation_rad * (1.0 + 1.0e-14)
            for item in results
        ),
        "relativistic_phase_finite": all(
            math.isfinite(item.accumulated_gyro_phase_rad)
            and math.isfinite(float(item.event_witness.get("observed_gamma", 1.0)))
            for item in results
        ),
        "maximum_relative_energy_error": max(item.maximum_relative_energy_error for item in results),
        "orbits_exceeding_energy_gate": None,
        "final_velocity_event_velocity_mismatches": sum(
            not _final_velocity_equals_event_velocity(item) for item in results
        ),
        "maximum_wall_endpoint_error_m": max(wall_errors, default=0.0),
        "numerical_failure_counts": {
            termination.value: sum(item.termination is termination for item in results)
            for termination in NUMERICAL_FAILURES
        },
        "timeout_counts": {
            termination.value: sum(item.termination is termination for item in results)
            for termination in TIMEOUTS
        },
        "domain_escape_subclasses": dict(sorted(subclasses.items())),
        "material_quarantine": bool(np.all(field.traversable_cells)),
    }


def endpoint_rows(launches: Sequence[ElectronLaunch], results: Sequence[Any], config: OrbitConfig) -> list[dict[str, Any]]:
    by_id = {item.launch_id: item for item in launches}
    rows = []
    for result in sorted(results, key=lambda item: item.launch_id):
        launch = by_id[result.launch_id]
        x, y, z = (float(item) for item in result.final_position_m)
        rows.append(
            {
                "launch_id": result.launch_id,
                "cell_id": launch.flux_surface_id.split("-r", 1)[0],
                "flux_surface_id": launch.flux_surface_id,
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


def _convergence_from_summaries(
    coarse: Mapping[str, Any], fine: Mapping[str, Any], maximum: float, require_overlap: bool
) -> dict[str, Any]:
    change = abs(float(fine["probability"]) - float(coarse["probability"]))
    overlap = max(coarse["lower"], fine["lower"]) <= min(coarse["upper"], fine["upper"])
    return {
        "probabilities": {"accepted-N": coarse["probability"], "accepted-2N": fine["probability"]},
        "successive_change": change,
        "maximum_allowed_change": maximum,
        "adjacent_wilson_overlap": bool(overlap),
        "converged": bool(change <= maximum and (overlap or not require_overlap)),
    }


def run_design_full(task: Mapping[str, Any]) -> dict[str, Any]:
    """Integrate every case of one design, assess N/2N convergence, seal, hand off.

    Runs inside a spawn worker. Orbit results never leave the worker: the main
    process receives strata, diagnostics, gate facts, endpoint rows, artifact
    hashes and the coupling handoffs. Sealing (``write_artifact`` deterministic
    replay + ``load_and_verify_artifact`` replay) requires the orbit_mc
    convergence-evidence contract (all three flags true); ``timestep_passed`` is
    the design's own N->2N screening check, ``cross_map_passed`` its field-adapter
    gates, ``backend_parity_passed`` the CPU parity check. A design that fails the
    N->2N check is therefore reported (summaries + endpoints) but NOT sealed, and
    carries no handoff; this is recorded, never hidden.
    """

    started = time.perf_counter()
    integrations: dict[str, dict[str, Any]] = {}
    for case in task["cases"]:
        integrations[case["case_key"]] = run_case_integration(case)
    by_role = {(case["role"], case["timestep"]): case["case_key"] for case in task["cases"]}
    coarse = integrations[by_role[("accepted", "N")]]["summary"].wall_hit
    fine = integrations[by_role[("accepted", "2N")]]["summary"].wall_hit
    convergence = _convergence_from_summaries(
        asdict(coarse), asdict(fine), float(task["maximum_successive_probability_change"]), bool(task["require_adjacent_wilson_overlap"])
    )
    if task["seal_policy"] == "converged":
        timestep_flag = bool(convergence["converged"])
        seal_basis = "evidentiary: timestep_passed is the design's own N->2N screening convergence check"
    elif task["seal_policy"] == "structural":
        timestep_flag = all(
            item["preflight"]["status"] == "passed"
            and all(result.termination not in NUMERICAL_FAILURES for result in item["results"])
            for item in integrations.values()
        )
        seal_basis = (
            "shakedown: timestep_passed is a structural check (preflight passed, zero numerical "
            "failures across N/2N); probability convergence is informational at 64 launches"
        )
    else:
        raise ValueError("unknown seal policy")
    flags = {
        "timestep_passed": timestep_flag,
        "cross_map_passed": bool(task["field_adapter_passed"]),
        "backend_parity_passed": bool(task["cpu_parity_passed"]),
    }
    sealable = all(flags.values())
    outputs: list[dict[str, Any]] = []
    for case in task["cases"]:
        case_started = time.perf_counter()
        integration = integrations[case["case_key"]]
        results = integration["results"]
        config = case["config"]
        field = case["field"]
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
            consumed = ledger.run(
                case["case_key"],
                "coupling_handoff_consumer",
                consume_handoff,
                export["handoff"],
                expected_artifact_sha256=export["artifact_file_sha256"],
                design_label=case["case_key"],
            )
        rows = endpoint_rows(case["launches"], results, config)
        endpoints_payload = canonical_bytes(
            {
                "schema_version": schema("endpoints"),
                "classification": CLASSIFICATION,
                "campaign_id": case["campaign_id"],
                "case_key": case["case_key"],
                "case_id": case["case_id"],
                "role": case["role"],
                "timestep": case["timestep"],
                "sealed": sealable,
                "orbit_artifact_file_sha256": None if export is None else export["artifact_file_sha256"],
                "rows": rows,
            }
        )
        facts = case_gate_facts(results, field, config)
        facts["orbits_exceeding_energy_gate"] = sum(
            item.maximum_relative_energy_error > float(task["maximum_relative_energy_error"]) for item in results
        )
        validators = list(integration["validators"]) + ([] if export is None else list(export["validators"])) + ledger.records
        outputs.append(
            {
                "case_key": case["case_key"],
                "campaign_id": case["campaign_id"],
                "case_id": case["case_id"],
                "role": case["role"],
                "timestep": case["timestep"],
                "process_id": os.getpid(),
                "preflight": integration["preflight"],
                "summary": integration["summary"].to_dict(),
                "strata": integration["strata"],
                "checkpoints": [
                    {key: item[key] for key in ("stage", "batch_id", "completed_launches", "file_sha256", "artifact_name")}
                    for item in integration["checkpoints"]
                ],
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
                "validators": validators,
                "timing_s": {
                    **integration["timing_s"],
                    "export_write_replay": None if export is None else export["timing_s"]["write_artifact_replay"],
                    "export_verify_replay": None if export is None else export["timing_s"]["load_and_verify_replay"],
                    "case_total": time.perf_counter() - case_started,
                },
            }
        )
    return {
        "case_id": task["case_id"],
        "process_id": os.getpid(),
        "convergence": convergence,
        "convergence_flags": flags,
        "seal_policy": task["seal_policy"],
        "seal_basis": seal_basis,
        "sealed": sealable,
        "cases": outputs,
        "timing_s": {"design_total": time.perf_counter() - started},
    }


def resolve_design_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Worker: bind the sweep, re-solve one design, return bore fields + evidence (or the exclusion)."""

    value = task["protocol"]
    started = time.perf_counter()
    try:
        binding = load_sweep_binding(value["field_source"])
        resolved = resolve_design(binding, task["case_id"], value, include_refined=task["include_refined"])
    except Exception as error:  # recorded as an exclusion, never hidden
        return {
            "case_id": task["case_id"],
            "status": "excluded",
            "reason": f"{type(error).__name__}: {error}"[:4096],
            "seconds": time.perf_counter() - started,
        }
    evidence = _plain(resolved.evidence)
    return {
        "case_id": task["case_id"],
        "status": "resolved" if evidence["passed"] else "excluded",
        "reason": None if evidence["passed"] else "field adapter gates failed",
        "geometry": resolved.geometry,
        "accepted_field": resolved.accepted.field,
        "accepted_serialized": _plain(resolved.accepted.serialized),
        "refined_field": None if resolved.refined is None else resolved.refined.field,
        "refined_serialized": None if resolved.refined is None else _plain(resolved.refined.serialized),
        "evidence": evidence,
        "seconds": time.perf_counter() - started,
    }


def worker_count(value: Mapping[str, Any]) -> int:
    execution = value["execution"]
    if not execution["parallel_cases"]:
        return 1
    return max(1, min(int(execution["max_case_workers"]), os.cpu_count() or 1))


# --------------------------------------------------------------------------
# Per-design assessment: convergence, gates, dataset rows
# --------------------------------------------------------------------------


def _probability(successes: int, trials: int) -> dict[str, Any]:
    return asdict(wilson_interval(successes, trials))


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return max(left["lower"], right["lower"]) <= min(left["upper"], right["upper"])


def design_convergence(value: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    maximum = float(value["gates"]["maximum_successive_probability_change"])
    coarse = cases["accepted-N"]["summary"]["wall_hit"]
    fine = cases["accepted-2N"]["summary"]["wall_hit"]
    record = _convergence_from_summaries(
        coarse, fine, maximum, bool(value["gates"]["require_adjacent_wilson_overlap"])
    )
    if "refined-N" in cases:
        refined = cases["refined-N"]["summary"]["wall_hit"]
        record["field_resolution_sensitivity"] = {
            "probabilities": {"accepted-N": coarse["probability"], "refined-N": refined["probability"]},
            "change": abs(float(refined["probability"]) - float(coarse["probability"])),
            "adjacent_wilson_overlap": _overlap(coarse, refined),
            "within_screening_change": abs(float(refined["probability"]) - float(coarse["probability"])) <= maximum,
            "binding": False,
        }
    return record


def design_gates(
    value: Mapping[str, Any],
    design_evidence: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    convergence: Mapping[str, Any],
    *,
    binding: bool,
) -> dict[str, Any]:
    limits = value["gates"]
    facts = {key: item["gate_facts"] for key, item in cases.items()}
    checks = {
        "field_adapter": bool(design_evidence["passed"]),
        "campaign_preflight": all(
            item["preflight"]["status"] == "passed"
            and item["preflight"]["maximum_launch_b_t"] <= item["preflight"]["maximum_declared_b_t"]
            for item in cases.values()
        ),
        "zero_numerical_failures": all(sum(f["numerical_failure_counts"].values()) == 0 for f in facts.values()),
        "energy": all(f["maximum_relative_energy_error"] <= limits["maximum_relative_energy_error"] for f in facts.values()),
        "final_velocity_equals_event_velocity": all(f["final_velocity_event_velocity_mismatches"] == 0 for f in facts.values()),
        "wall_endpoint": all(f["maximum_wall_endpoint_error_m"] <= limits["maximum_wall_endpoint_error_m"] for f in facts.values()),
        "earliest_event": all(f["earliest_event_ordering"] for f in facts.values()),
        "runtime_rotation": all(f["runtime_rotation_bound"] for f in facts.values()),
        "relativistic_phase": all(f["relativistic_phase_finite"] for f in facts.values()),
        "material_quarantine": all(f["material_quarantine"] for f in facts.values()),
        "independent_repeats": all(
            row["physical_position_repeat_count"] >= limits["minimum_independent_repeats_per_stratum"]
            for item in cases.values()
            for row in item["strata"]
        ),
        "exact_authority_replay_when_sealed": all(
            (not item["sealed"]) or item["artifact_file_sha256"] == item["verified_file_sha256"] for item in cases.values()
        ),
        "sealed_iff_convergence_flag": all(
            item["sealed"] == bool(convergence["worker_flags"]["timestep_passed"] and convergence["worker_flags"]["cross_map_passed"] and convergence["worker_flags"]["backend_parity_passed"])
            for item in cases.values()
        ),
        "seal_policy_matches_plan": convergence["seal_policy"] == ("converged" if binding else "structural"),
        "cross_process_determinism": all(item["determinism_sample"]["passed"] for item in cases.values()),
        "handoff_consumed_when_sealed": all(
            (not item["sealed"]) or (item["consumed_handoff"] is not None and item["consumed_handoff"]["passed"])
            for item in cases.values()
        ),
    }
    timeouts = sum(sum(f["timeout_counts"].values()) for f in facts.values())
    return {
        "checks": checks,
        "structural_passed": all(checks.values()),
        "converged": bool(convergence["converged"]),
        "sealed": all(item["sealed"] for item in cases.values()),
        "timeout_free": timeouts == 0,
        "timeout_count": timeouts,
        "passed": bool(all(checks.values()) and convergence["converged"]),
    }


def per_cell_table(strata: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, int]] = {}
    for row in strata:
        cell = cells.setdefault(row["cell_id"], {"trials": 0, "wall_hit": 0, "reflected": 0, "domain_escape": 0, "timeout": 0})
        counts = row["termination_counts"]
        cell["trials"] += int(row["trials"])
        cell["wall_hit"] += int(counts["wall_hit"])
        cell["reflected"] += int(counts["reflected"])
        cell["domain_escape"] += int(counts["domain_escape"])
        cell["timeout"] += int(row["trials"]) - int(counts["wall_hit"]) - int(counts["reflected"]) - int(counts["domain_escape"])
    return {
        cell: {
            "trials": item["trials"],
            "counts": {k: item[k] for k in ("wall_hit", "reflected", "domain_escape", "timeout")},
            "wall_hit": _probability(item["wall_hit"], item["trials"]),
            "domain_escape": _probability(item["domain_escape"], item["trials"]),
            "reflected": _probability(item["reflected"], item["trials"]),
            "timeout": _probability(item["timeout"], item["trials"]),
        }
        for cell, item in sorted(cells.items())
    }


def _case_totals(case: Mapping[str, Any]) -> dict[str, Any]:
    summary = case["summary"]
    return {
        "campaign_id": case["campaign_id"],
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
        "steps": case["diagnostics"]["steps"],
        "per_orbit_ms": case["timing_s"]["per_orbit_ms"],
        "tolerance_close_event_count": case["diagnostics"]["tolerance_close_event_count"],
        "mu": case["diagnostics"]["magnetic_moment_variation_diagnostic"],
    }


def dataset_row(
    value: Mapping[str, Any],
    design_authority: Mapping[str, Any],
    design_evidence: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    convergence: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    reported = cases["accepted-2N"]
    summary = reported["summary"]
    trials = int(summary["trial_count"])
    accepted_evidence = design_evidence["accepted_bore_field"]
    return {
        "case_id": design_authority["case_id"],
        "design_id": design_authority["design_id"],
        "sweep_index": design_module.case_index(design_authority["case_id"]),
        "batch": design_authority["batch"],
        "representative": design_authority["representative"],
        "classification": CLASSIFICATION,
        "design_values": design_authority["design_values"],
        "geometry": design_authority["geometry"],
        "identities": {
            "geometry_sha256": design_authority["geometry_sha256"],
            "case_sha256": design_authority["case_sha256"],
            "accepted_field_identity_sha256": design_authority["accepted_field_identity_sha256"],
            "refined_field_identity_sha256": design_authority["refined_field_identity_sha256"],
        },
        "field": {
            "status": value["field_source"]["field_status"],
            "bore_max_b_t": accepted_evidence["max_b_t"],
            "bore_grid": accepted_evidence["bore_grid"],
            "interpolation_b_relative_rms": accepted_evidence["interpolation_error_report"]["b_relative_rms"],
            "cross_resolution_b_relative_rms": (
                None if design_evidence["cross_resolution"] is None else design_evidence["cross_resolution"]["b_relative_rms"]
            ),
            "sweep_qois": design_evidence["sweep_record"],
        },
        "launch_design": {
            "cells": design_authority["cells"],
            "radius_fractions_of_wall": value["launches"]["radius_fractions_of_wall"],
            "launch_radii_m": [f * design_authority["geometry"]["wall_radius_m"] for f in value["launches"]["radius_fractions_of_wall"]],
            "launches_per_case": trials,
            "cell_to_field": [
                {
                    "cell_id": cell["cell_id"],
                    "axial_center_m": cell["axial_center_m"],
                    "nearest_axis_null_distance_m": min(
                        (abs(cell["axial_center_m"] - z) for z in design_evidence["sweep_record"]["axis_null_positions_m"]),
                        default=None,
                    ),
                    "nearest_axis_bz_peak_distance_m": min(
                        (abs(cell["axial_center_m"] - z) for z in design_evidence["sweep_record"]["axis_cusp_positions_m"]),
                        default=None,
                    ),
                }
                for cell in design_authority["cells"]
            ],
        },
        "orbit_config": {key: cases[key]["config"] for key in cases},
        "reported": {
            "case": "accepted-2N",
            "wall_hit": summary["wall_hit"],
            "domain_escape": summary["escaped"],
            "reflected": summary["reflected"],
            "timeout": summary["incomplete"],
            "domain_escape_subclasses": reported["gate_facts"]["domain_escape_subclasses"],
        },
        "cases": {key: _case_totals(item) for key, item in cases.items()},
        "convergence": convergence,
        "per_stratum": {
            key: [
                {
                    "cell_id": row["cell_id"],
                    "kinetic_energy_ev": row["kinetic_energy_ev"],
                    "pitch_angle_deg": row["pitch_angle_deg"],
                    "parallel_direction": row["parallel_direction"],
                    "trials": row["trials"],
                    "wall_hit": row["termination_counts"]["wall_hit"],
                    "reflected": row["termination_counts"]["reflected"],
                    "domain_escape": row["termination_counts"]["domain_escape"],
                    "timeout": row["trials"]
                    - row["termination_counts"]["wall_hit"]
                    - row["termination_counts"]["reflected"]
                    - row["termination_counts"]["domain_escape"],
                    "p_wall": row["wall_hit"]["probability"],
                }
                for row in item["strata"]
            ]
            for key, item in cases.items()
        },
        "per_stratum_note": "counts only; Wilson intervals per stratum live in artifacts/summaries/<case_key>.json",
        "per_cell": {key: per_cell_table(item["strata"]) for key, item in cases.items()},
        "diagnostics": {
            "magnetic_moment_variation": reported["diagnostics"]["magnetic_moment_variation_diagnostic"],
            "tolerance_close_share": reported["diagnostics"]["tolerance_close_event_count"] / trials,
            "tolerance_close_conditions": reported["diagnostics"]["tolerance_close_conditions"],
            "event_resolution_counts": reported["diagnostics"]["event_resolution_counts"],
            "steps": reported["diagnostics"]["steps"],
            "maximum_relative_energy_error": reported["gate_facts"]["maximum_relative_energy_error"],
            "reflection_counts": {key: item["summary"]["termination_counts"]["reflected"] for key, item in cases.items()},
        },
        "gates": gates,
    }


CSV_COLUMNS = (
    "case_id", "design_id", "sweep_index", "batch", "representative", "classification",
    "stage_count_selector", "stage_pitch_m", "magnet_axial_fraction", "chamber_outer_radius_m",
    "dielectric_thickness_m", "radial_clearance_m", "magnet_radial_thickness_m", "source_strength_scale",
    "exit_length_fraction", "exit_expansion_descriptor", "first_polarity_selector",
    "wall_radius_m", "chamber_length_m", "injector_length_m", "exit_start_m", "exit_length_m",
    "exit_outer_radius_m", "stage_count", "stage_pitch_represented_m", "has_divergent_exit",
    "bore_max_b_t", "centreline_mid_abs_bz_t", "centreline_abs_bz_peak_t", "minimum_mirror_ratio",
    "stage_gradient_rms_t_per_m", "field_energy_j", "axis_cusp_count", "axis_null_count",
    "interpolation_b_relative_rms", "cross_resolution_b_relative_rms",
    "trials_2N", "p_wall_2N", "p_wall_2N_lower", "p_wall_2N_upper", "p_escape_2N", "p_reflected_2N", "p_timeout_2N",
    "p_wall_N", "convergence_change", "converged",
    "p_wall_2N_cell1", "p_wall_2N_cell2", "p_wall_2N_cell3", "p_wall_2N_cell4",
    "escape_upstream_anode_plane_2N", "escape_exit_plane_2N", "escape_divergent_section_radial_2N", "escape_unclassified_2N",
    "reflections_N", "reflections_2N", "timeouts_2N",
    "mu_variation_median", "mu_variation_max", "tolerance_close_share", "steps_median_2N",
    "structural_gates_passed", "gates_passed", "accepted_field_identity_sha256", "orbit_artifact_2N_sha256",
)


def dataset_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        values = row["design_values"]
        geometry = row["geometry"]
        qois = row["field"]["sweep_qois"]
        reported = row["reported"]
        cells = row["per_cell"]["accepted-2N"]
        cell_ids = sorted(cells)
        escapes = reported["domain_escape_subclasses"]
        writer.writerow(
            [
                row["case_id"], row["design_id"], row["sweep_index"], row["batch"], row["representative"], row["classification"],
                *[values[name] for name in (
                    "stage_count_selector", "stage_pitch_m", "magnet_axial_fraction", "chamber_outer_radius_m",
                    "dielectric_thickness_m", "radial_clearance_m", "magnet_radial_thickness_m", "source_strength_scale",
                    "exit_length_fraction", "exit_expansion_descriptor", "first_polarity_selector",
                )],
                geometry["wall_radius_m"], geometry["chamber_length_m"], geometry["injector_length_m"], geometry["exit_start_m"],
                geometry["exit_length_m"], geometry["exit_outer_radius_m"], geometry["stage_count"], geometry["stage_pitch_m"],
                geometry["has_divergent_exit"],
                row["field"]["bore_max_b_t"], qois["centreline_mid_abs_bz_t"], qois["centreline_abs_bz_peak_t"], qois["minimum_mirror_ratio"],
                qois["stage_gradient_rms_t_per_m"], qois["field_energy_j"], len(qois["axis_cusp_positions_m"]), len(qois["axis_null_positions_m"]),
                row["field"]["interpolation_b_relative_rms"], row["field"]["cross_resolution_b_relative_rms"],
                reported["wall_hit"]["trials"], reported["wall_hit"]["probability"], reported["wall_hit"]["lower"], reported["wall_hit"]["upper"],
                reported["domain_escape"]["probability"], reported["reflected"]["probability"], reported["timeout"]["probability"],
                row["convergence"]["probabilities"]["accepted-N"], row["convergence"]["successive_change"], row["convergence"]["converged"],
                *[cells[cell]["wall_hit"]["probability"] for cell in cell_ids],
                escapes.get("upstream_anode_plane", 0), escapes.get("exit_plane", 0), escapes.get("divergent_section_radial", 0), escapes.get("unclassified", 0),
                row["diagnostics"]["reflection_counts"]["accepted-N"], row["diagnostics"]["reflection_counts"]["accepted-2N"],
                sum(row["cases"]["accepted-2N"]["timeout_counts"].values()),
                row["diagnostics"]["magnetic_moment_variation"]["median"], row["diagnostics"]["magnetic_moment_variation"]["max"],
                row["diagnostics"]["tolerance_close_share"], row["diagnostics"]["steps"]["median"],
                row["gates"]["structural_passed"], row["gates"]["passed"], row["identities"]["accepted_field_identity_sha256"],
                row["cases"]["accepted-2N"]["orbit_artifact_file_sha256"],
            ]
        )
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Shakedown record verification (prepare, execute, prebundle)
# --------------------------------------------------------------------------


def verify_shakedown_record(value: Mapping[str, Any], record: Mapping[str, Any], bound_designs: Mapping[str, BoundDesign] | None = None) -> dict[str, bool]:
    """Fail closed unless the shakedown proves the current protocol, orbit_mc and field pipeline."""

    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = record.get("protocol_semantic_sha256") == semantic_sha256(value)
    checks["orbit_mc_source_sha256_current"] = record.get("orbit_mc_source_sha256") == orbit_mc_source_sha256()
    checks["field_pipeline_source_sha256_current"] = record.get("field_pipeline_source_sha256") == field_pipeline_source_sha256()
    checks["experiment_code_sha256_current"] = record.get("experiment_code_sha256") == experiment_code_sha256()
    contract = orbit_mc_contract_report(value)
    checks["orbit_mc_schema_versions_current"] = record.get("orbit_mc_schema_versions") == contract["observed"] and contract["matches"]
    disjointness = record.get("disjointness")
    checks["disjointness_proven"] = (
        isinstance(disjointness, Mapping)
        and disjointness.get("proven") is True
        and all(
            report["disjoint"] is True and all(count == 0 for count in report["overlap_counts"].values())
            for report in disjointness.get("reports", {}).values()
        )
        and set(disjointness.get("reports", {})) == {"against_evidentiary_same_designs"}
        and isinstance(disjointness.get("gyrophase_grids"), Mapping)
        and disjointness["gyrophase_grids"].get("disjoint") is True
    )
    plan = shakedown_plan(value)
    expected_keys = {key for _, _, _, _, key in case_matrix(value, plan)}
    cases = record.get("cases")
    checks["expected_cases"] = isinstance(cases, Mapping) and set(cases) == expected_keys
    try:
        if bound_designs is None:
            binding = load_sweep_binding(value["field_source"])
            bound_designs = bind_designs(value, binding, plan.case_ids)
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
        and all(
            item.get("validators", {}).get("failed") == 0
            and item.get("validators", {}).get("passed", 0) > 0
            and item.get("export_stage_ran") is True
            and item.get("handoff_consumed") is True
            for item in cases.values()
        )
    )
    checks["zero_exclusions"] = record.get("design_exclusions") == []
    runtime = record.get("runtime")
    checks["runtime_accepted_and_bundle_validated"] = (
        isinstance(runtime, Mapping)
        and runtime.get("terminal_state") == "accepted_result"
        and runtime.get("bundle_validated") is True
    )
    checks["dataset_assembled"] = isinstance(record.get("dataset_summary"), Mapping) and record["dataset_summary"].get("design_count") == len(plan.case_ids)
    projection = record.get("timing_projection")
    checks["extension_decision_consistent"] = (
        isinstance(projection, Mapping)
        and isinstance(projection.get("within_budget"), bool)
        and projection["within_budget"] == bool(value["designs"]["extension_batch_included"])
    )
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
    return FrozenAuthority(
        strict_json_file(AUTHORITIES_PATH),
        strict_json_file(DESIGN_AUTHORITIES_PATH),
        strict_json_file(SHAKEDOWN_PATH),
        SHAKEDOWN_PATH.read_bytes(),
    )


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
    tightness_floor = value["gates"]["minimum_certificate_dense_to_bound_ratio"]
    collector.setdefault("plan_kind", plan.kind)
    collector.setdefault("cases", {})
    collector.setdefault("validators", [])

    def prebundle(context: Any) -> Mapping[str, Any]:
        binding_report = source_binding_report(value)
        contract = binding_report["orbit_mc"]
        binding = load_sweep_binding(value["field_source"])
        bound = bind_designs(value, binding, plan.case_ids)
        design_authorities = build_design_authorities(value, plan, bound)
        if frozen is not None:
            if semantic_sha256(value) != frozen.authorities["protocol_semantic_sha256"]:
                raise ValueError("protocol semantic authority differs")
            if (
                frozen.design_authorities != design_authorities
                or semantic_sha256(frozen.design_authorities) != frozen.authorities["design_authorities_sha256"]
            ):
                raise ValueError("design authorities differ from preregistration")
            if frozen.authorities["orbit_mc_source_sha256"] != contract["source_sha256"]:
                raise ValueError("orbit_mc source differs from preregistered authority")
            if frozen.authorities["field_pipeline_source_sha256"] != binding_report["field_pipeline_source_sha256"]:
                raise ValueError("field pipeline source differs from preregistered authority")
            if (
                hashlib.sha256(frozen.shakedown_bytes).hexdigest() != frozen.authorities["shakedown_file_sha256"]
                or semantic_sha256(frozen.shakedown) != frozen.authorities["shakedown_semantic_sha256"]
            ):
                raise ValueError("shakedown record differs from preregistered authority")
            verify_shakedown_record(value, frozen.shakedown, bind_designs(value, binding, shakedown_plan(value).case_ids))
        by_key = {item["case_key"]: item for item in design_authorities["cases"]}
        cases: dict[str, dict[str, Any]] = {}
        for case_id, role, timestep, campaign, key in case_matrix(value, plan):
            authority = by_key[key]
            launches = build_case_launches(value, plan, bound[case_id].geometry, role, timestep)
            batches = batch_records(plan, launches)
            launch_bytes = canonical_bytes(runtime_launch_payload(campaign, launches))
            batch_bytes = canonical_bytes(runtime_batch_payload(plan, campaign, launches))
            if (
                authority["campaign_id"] != campaign
                or hashlib.sha256(launch_bytes).hexdigest() != authority["runtime_launch_payload_byte_sha256"]
                or hashlib.sha256(batch_bytes).hexdigest() != authority["runtime_batch_payload_byte_sha256"]
                or load_runtime_launch_payload(launch_bytes, campaign) != tuple(sorted(launches, key=lambda item: item.launch_id))
                or content_hash(launch_records(launches)) != authority["orbit_launches_sha256"]
                or estimator_identity(launches, batches) != authority["estimator_sha256"]
            ):
                raise ValueError(f"{campaign} launch/batch authority differs")
            cases[key] = {"authority": authority, "launches": launches, "batches": batches, "case_id": case_id, "role": role, "timestep": timestep}
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/design-authorities.json", design_authorities)
        context.write_json("artifacts/campaign-plan.json", plan_record(plan))
        context.write_json("artifacts/orbit-mc-contract.json", contract)
        context.write_json(
            "artifacts/field-pipeline-binding.json",
            {
                "field_pipeline_source_sha256": binding_report["field_pipeline_source_sha256"],
                "field_pipeline_source_files": binding_report["field_pipeline_source_files"],
                "sweep_manifest_file_sha256": binding.manifest_file_sha256,
                "sweep_raw_results_file_sha256": binding.raw_file_sha256,
                "sweep_summary_file_sha256": binding.summary_file_sha256,
                "field_status": value["field_source"]["field_status"],
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
                    "disjointness": shakedown_disjointness(value, bound),
                },
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
                "backend": "numpy-cpu-relativistic-boris",
            },
        )
        state.update({"cases": cases, "bound": bound, "design_authorities": design_authorities, "binding": binding})
        collector["prebundle"] = {"design_count": len(plan.case_ids), "case_count": len(cases), "orbit_mc_contract": contract}
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "classification": CLASSIFICATION,
            "design_count": len(plan.case_ids),
            "case_count": len(cases),
            "total_launches": sum(len(item["launches"]) for item in cases.values()),
            "orbit_mc_source_sha256": contract["source_sha256"],
            "field_pipeline_source_sha256": binding_report["field_pipeline_source_sha256"],
        }

    def development(context: Any) -> Decision:
        started = time.perf_counter()
        workers = worker_count(value)
        manufactured = manufactured_gate_report(value)
        context.write_json("artifacts/manufactured-gates.json", manufactured)
        context.before_expensive(
            "l1a-resolve-all-designs",
            kind="solver",
            details={
                "design_count": len(plan.case_ids),
                "solver": "cft_revival.fields.solve_problem_cpu",
                "refined_designs": [c for c in plan.case_ids if c in value["designs"]["representative_case_ids"]],
                "worker_pool_size": workers,
                "plan_kind": plan.kind,
            },
        )
        tasks = [
            {
                "case_id": case_id,
                "protocol": dict(value),
                "include_refined": case_id in value["designs"]["representative_case_ids"],
            }
            for case_id in plan.case_ids
        ]
        stage_started = time.perf_counter()
        resolved = run_stage(tasks, resolve_design_task, workers)
        resolve_wall = time.perf_counter() - stage_started
        fields: dict[str, dict[str, Any]] = {}
        evidence: dict[str, Any] = {}
        exclusions: list[dict[str, Any]] = []
        for task, outcome in zip(tasks, resolved, strict=True):
            if outcome["case_id"] != task["case_id"]:
                raise RuntimeError("design results returned out of order")
            if outcome["status"] != "resolved":
                exclusions.append({"case_id": outcome["case_id"], "reason": outcome["reason"], "evidence": outcome.get("evidence")})
                continue
            case_id = outcome["case_id"]
            bound = state["bound"][case_id]
            if (
                outcome["evidence"]["accepted_bore_field"]["source_identity_sha256"] != bound.accepted_field_identity
                or outcome["geometry"] != bound.geometry
            ):
                raise ValueError(f"{case_id}: resolved field identity or geometry differs from the bound design")
            fields[case_id] = {"accepted": outcome["accepted_field"], "refined": outcome["refined_field"]}
            evidence[case_id] = outcome["evidence"]
            context.write_json(f"artifacts/fields/{case_id}.json", outcome["accepted_serialized"])
            context.write_json(f"artifacts/field-evidence/{case_id}.json", outcome["evidence"])
        context.write_json(
            "artifacts/design-exclusions.json",
            {"schema_version": schema("design-exclusions"), "rule": value["designs"]["fallback_rule"], "excluded": exclusions},
        )
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
                case_id: {
                    "interpolation_b_relative_rms": item["accepted_bore_field"]["interpolation_error_report"]["b_relative_rms"],
                    "cross_resolution_b_relative_rms": None if item["cross_resolution"] is None else item["cross_resolution"]["b_relative_rms"],
                    "bore_max_b_t": item["accepted_bore_field"]["max_b_t"],
                    "passed": item["passed"],
                }
                for case_id, item in evidence.items()
            },
        }
        return Decision(
            accepted,
            {
                "manufactured_passed": manufactured["passed"],
                "resolved_design_count": len(fields),
                "excluded_design_count": len(exclusions),
                "resolve_wall_s": resolve_wall,
            },
        )

    def assessment(context: Any) -> Decision:
        started = time.perf_counter()
        workers = worker_count(value)
        main_ledger = ValidatorLedger()
        case_tasks: list[dict[str, Any]] = []
        for case_id, role, timestep, campaign, key in case_matrix(value, plan):
            if case_id not in state["fields"]:
                continue
            frozen_case = state["cases"][key]
            authority = frozen_case["authority"]
            field = state["fields"][case_id][role]
            if field is None:
                raise RuntimeError(f"{key}: field role {role} was not resolved")
            config = orbit_config(value, state["bound"][case_id].geometry, timestep)
            config_sha = content_hash(asdict(config))
            policy_sha = policy_identity(value, plan, case_id, role, timestep)
            if (
                field.source_identity_sha256 != authority["field_identity_sha256"]
                or config_sha != authority["config_identity_sha256"]
                or policy_sha != authority["policy_identity_sha256"]
                or estimator_identity(frozen_case["launches"], frozen_case["batches"]) != authority["estimator_sha256"]
            ):
                raise ValueError(f"{campaign} execution authority differs")
            bore_evidence = state["field_evidence"][case_id][f"{role}_bore_field"]
            case_tasks.append(
                {
                    "case_key": key,
                    "case_id": case_id,
                    "role": role,
                    "timestep": timestep,
                    "campaign_id": campaign,
                    "authority": authority,
                    "launches": frozen_case["launches"],
                    "batches": frozen_case["batches"],
                    "field": field,
                    "config": config,
                    "field_sha": authority["field_identity_sha256"],
                    "config_sha": config_sha,
                    "policy_sha": policy_sha,
                    "launch_sha": authority["orbit_launches_sha256"],
                    "batch_sha": authority["batch_manifest_sha256"],
                    "tightness_floor": tightness_floor,
                    "partial_checkpoint_prefix_count": plan.partial_checkpoint_prefix_count,
                    "work_dir": str(context.cache_root / "cases" / key),
                    "field_evidence": {"field_error_report": bore_evidence["interpolation_error_report"]},
                    "preregistration": {
                        "protocol_id": value["schema_version"] if plan.binding_gates else f"{schema('shakedown')}:NON-EVIDENTIARY",
                        "frozen_before_outcomes": True,
                        "held_out_geometry_status": "pending",
                    },
                }
            )
        for task in case_tasks:
            context.before_expensive(
                f"orbit-{task['case_key']}",
                kind="label",
                details={
                    "campaign_id": task["campaign_id"],
                    "launch_count": len(task["launches"]),
                    "sequential_batches_within_case": True,
                    "parallel_designs": workers > 1,
                    "worker_pool_size": workers,
                    "plan_kind": plan.kind,
                    "classification": CLASSIFICATION,
                },
            )
        design_tasks = [
            {
                "case_id": case_id,
                "cases": [task for task in case_tasks if task["case_id"] == case_id],
                "maximum_successive_probability_change": value["gates"]["maximum_successive_probability_change"],
                "require_adjacent_wilson_overlap": value["gates"]["require_adjacent_wilson_overlap"],
                "maximum_relative_energy_error": value["gates"]["maximum_relative_energy_error"],
                "field_adapter_passed": bool(state["field_evidence"][case_id]["passed"]),
                "cpu_parity_passed": bool(state["manufactured"]["checks"]["cpu_parity"]),
                "seal_policy": "converged" if plan.binding_gates else "structural",
            }
            for case_id in plan.case_ids
            if case_id in state["fields"]
        ]
        stage_started = time.perf_counter()
        design_outcomes = run_stage(design_tasks, run_design_full, workers)
        cases_wall = time.perf_counter() - stage_started
        publish_full = set(value["designs"]["representative_case_ids"])
        cases: dict[str, dict[str, Any]] = {}
        design_worker_records: dict[str, dict[str, Any]] = {}
        tasks_by_key = {task["case_key"]: task for task in case_tasks}
        flat_outcomes: list[dict[str, Any]] = []
        for design_task, design_outcome in zip(design_tasks, design_outcomes, strict=True):
            if design_outcome["case_id"] != design_task["case_id"]:
                raise RuntimeError("design results returned out of order")
            design_worker_records[design_task["case_id"]] = {
                "convergence": design_outcome["convergence"],
                "convergence_flags": design_outcome["convergence_flags"],
                "seal_policy": design_outcome["seal_policy"],
                "seal_basis": design_outcome["seal_basis"],
                "sealed": design_outcome["sealed"],
                "timing_s": design_outcome["timing_s"],
                "process_id": design_outcome["process_id"],
            }
            flat_outcomes.extend(design_outcome["cases"])
        for outcome in flat_outcomes:
            key = outcome["case_key"]
            task = tasks_by_key[key]
            ordered = sorted(task["launches"], key=lambda item: item.launch_id)

            def determinism_sample(
                sample_launches: Sequence[Any] = ordered[:2],
                sample_field: Any = task["field"],
                sample_config: Any = task["config"],
                expected: Mapping[str, str] = outcome["determinism_hashes"],
            ) -> dict[str, Any]:
                compared = 0
                for launch in sample_launches:
                    local = integrate_orbit(launch, sample_field, sample_config)
                    if content_hash(result_record(local)) != expected[launch.launch_id]:
                        raise RuntimeError(f"cross-process determinism differs for {launch.launch_id}")
                    compared += 1
                return {"compared": compared, "passed": True}

            sample = main_ledger.run(key, "cross_process_determinism_sample", determinism_sample)
            context.write_json(
                f"artifacts/summaries/{key}.json",
                {
                    "classification": CLASSIFICATION,
                    "case_id": task["case_id"],
                    "role": task["role"],
                    "timestep": task["timestep"],
                    "campaign_id": task["campaign_id"],
                    "summary": outcome["summary"],
                    "strata": outcome["strata"],
                    "preflight": outcome["preflight"],
                    "config": asdict(task["config"]),
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
            context.write_blob(f"artifacts/endpoints/{key}.json.gz", outcome["endpoints_gz"])
            if outcome["sealed"]:
                context.write_blob(f"artifacts/orbits/{key}.json.sha256", outcome["artifact_sidecar_bytes"])
                if task["case_id"] in publish_full:
                    context.write_blob(
                        f"artifacts/orbits/{key}.json.gz",
                        gzip.compress(Path(outcome["artifact_path"]).read_bytes(), mtime=0),
                    )
                context.write_json(f"artifacts/handoffs/{key}.json", outcome["handoff"])
            cases[key] = {
                **{
                    k: outcome[k]
                    for k in (
                        "case_key", "campaign_id", "case_id", "role", "timestep", "preflight", "summary", "strata",
                        "diagnostics", "gate_facts", "sealed", "artifact_file_sha256", "verified_file_sha256", "handoff",
                        "consumed_handoff", "endpoints_payload_sha256", "timing_s", "process_id",
                    )
                },
                "config": asdict(task["config"]),
                "validators": list(outcome["validators"]),
                "determinism_sample": sample,
            }
            collector["cases"][key] = _collect_case(cases[key])
        # Per-design assessment
        designs_by_id = {item["case_id"]: item for item in state["design_authorities"]["designs"]}
        dataset_rows: list[dict[str, Any]] = []
        design_gate_records: dict[str, Any] = {}
        for case_id in plan.case_ids:
            if case_id not in state["fields"]:
                continue
            design_cases = {
                f"{role}-{timestep}": cases[case_key(case_id, role, timestep)]
                for role, timestep in case_roles(value, case_id)
            }
            convergence = design_convergence(value, design_cases)
            worker_record = design_worker_records[case_id]
            if (
                worker_record["convergence"]["converged"] != convergence["converged"]
                or abs(worker_record["convergence"]["successive_change"] - convergence["successive_change"]) > 1e-15
            ):
                raise RuntimeError(f"{case_id}: worker and main-process convergence assessments differ")
            convergence["worker_flags"] = worker_record["convergence_flags"]
            convergence["seal_policy"] = worker_record["seal_policy"]
            convergence["seal_basis"] = worker_record["seal_basis"]
            convergence["sealed"] = worker_record["sealed"]
            gates = design_gates(
                value, state["field_evidence"][case_id], design_cases, convergence, binding=plan.binding_gates
            )
            design_gate_records[case_id] = gates
            dataset_rows.append(
                dataset_row(value, designs_by_id[case_id], state["field_evidence"][case_id], design_cases, convergence, gates)
            )
        all_validators = list(main_ledger.records)
        for item in cases.values():
            all_validators.extend(item["validators"])
        validator_failures = [item for item in all_validators if not item["passed"]]
        structural_all = all(item["structural_passed"] for item in design_gate_records.values())
        converged_count = sum(item["converged"] for item in design_gate_records.values())
        timeout_free_count = sum(item["timeout_free"] for item in design_gate_records.values())
        consumer = consume_v4_export(value, REPOSITORY)
        consumer_record = {
            "schema_version": schema("coupling-consumer-record"),
            "consumer_id": value["coupling_consumer"]["consumer_id"],
            "classification": CLASSIFICATION,
            "v4_reference": consumer,
            "screening_designs_consumed": [
                {
                    "case_id": row["case_id"],
                    "case": row["reported"]["case"],
                    "sealed": row["cases"]["accepted-2N"]["sealed"],
                    "handoff_sha256": row["cases"]["accepted-2N"]["handoff_sha256"],
                    "probability": row["reported"]["wall_hit"]["probability"],
                    "confidence_interval_95": [row["reported"]["wall_hit"]["lower"], row["reported"]["wall_hit"]["upper"]],
                    "trial_count": row["reported"]["wall_hit"]["trials"],
                    "consumed": cases[case_key(row["case_id"], "accepted", "2N")]["consumed_handoff"],
                    "consumption_status": (
                        "consumed_verified_handoff"
                        if row["cases"]["accepted-2N"]["sealed"]
                        else "not_consumable_unsealed_nonconverged_design (reported through summaries/endpoints only)"
                    ),
                    "label": CLASSIFICATION,
                }
                for row in dataset_rows
            ],
            "statement": value["coupling_consumer"]["v4_absence_statement"],
        }
        context.write_json("artifacts/coupling-consumer-record.json", consumer_record)
        wall_probabilities = [row["reported"]["wall_hit"]["probability"] for row in dataset_rows]
        dataset = {
            "schema_version": schema("geometry-wall-loss-dataset"),
            "classification": CLASSIFICATION,
            "classification_statement": value["classification_statement"],
            "claim_boundary": value["claim_boundary"],
            "plan_kind": plan.kind,
            "evidentiary": plan.binding_gates,
            "generated_at_utc": datetime.now(timezone.utc),
            "protocol_semantic_sha256": semantic_sha256(value),
            "orbit_mc_source_sha256": collector["prebundle"]["orbit_mc_contract"]["source_sha256"],
            "field_pipeline_source_sha256": field_pipeline_source_sha256(),
            "field_source": {
                "experiment": value["field_source"]["experiment"],
                "field_status": value["field_source"]["field_status"],
                "manifest_file_sha256": value["field_source"]["manifest_file_sha256"],
                "raw_results_file_sha256": value["field_source"]["raw_results_file_sha256"],
            },
            "launch_rule": value["launches"]["cell_rule"],
            "orbit_geometry_rule": value["orbit_geometry_rule"],
            "design_count": len(dataset_rows),
            "excluded_designs": state["exclusions"],
            "reported_case": value["cases"]["reported_probability_case"],
            "headline": {
                "wall_hit_probability_min": min(wall_probabilities, default=None),
                "wall_hit_probability_max": max(wall_probabilities, default=None),
                "wall_hit_probability_median": statistics.median(wall_probabilities) if wall_probabilities else None,
                "least_wall_loss_case_ids": [row["case_id"] for row in sorted(dataset_rows, key=lambda r: (r["reported"]["wall_hit"]["probability"], r["case_id"]))[:3]],
                "most_wall_loss_case_ids": [row["case_id"] for row in sorted(dataset_rows, key=lambda r: (-r["reported"]["wall_hit"]["probability"], r["case_id"]))[:3]],
                "designs_with_reflections": [row["case_id"] for row in dataset_rows if any(v > 0 for v in row["diagnostics"]["reflection_counts"].values())],
                "total_reflections": sum(sum(row["diagnostics"]["reflection_counts"].values()) for row in dataset_rows),
                "converged_design_count": converged_count,
                "sealed_design_count": sum(item["sealed"] for item in design_gate_records.values()),
                "timeout_free_design_count": timeout_free_count,
                "structural_gates_all_passed": structural_all,
            },
            "designs": dataset_rows,
        }
        context.write_json("artifacts/geometry-wall-loss-dataset.json", dataset)
        csv_bytes = dataset_csv(dataset_rows)
        context.write_blob("artifacts/geometry-wall-loss-dataset.csv", csv_bytes)
        gates = {
            "binding": plan.binding_gates,
            "manufactured": state["manufactured"]["checks"],
            "per_design": design_gate_records,
            "structural_all_passed": structural_all,
            "converged_design_count": converged_count,
            "timeout_free_design_count": timeout_free_count,
            "design_count": len(design_gate_records),
            "validator_failures": len(validator_failures),
            "exact_authority_replay_count": sum(
                item["sealed"] and item["artifact_file_sha256"] == item["verified_file_sha256"] for item in cases.values()
            ),
            "sealed_case_count": sum(item["sealed"] for item in cases.values()),
            "case_count": len(cases),
            "diagnostics_not_gates": {"magnetic_moment_variation": "per design in the dataset; never a gate"},
            "passed": bool(structural_all and state["manufactured"]["passed"] and not validator_failures and cases),
        }
        context.write_json("artifacts/gates.json", _plain(gates))
        if plan.binding_gates:
            accepted = bool(gates["passed"])
            status = "accepted_screening_dataset" if accepted else "rejected"
        else:
            accepted = bool(gates["passed"] and not state["exclusions"])
            status = "shakedown_passed" if accepted else "shakedown_failed"
        terminal = {
            "status": status,
            "plan_kind": plan.kind,
            "evidentiary": plan.binding_gates,
            "classification": CLASSIFICATION,
            "design_count": len(dataset_rows),
            "excluded_design_count": len(state["exclusions"]),
            "case_count": len(cases),
            "orbit_count": sum(item["summary"]["trial_count"] for item in cases.values()),
            "headline": dataset["headline"],
            "gates": _plain(gates),
            "validators": {"passed": sum(item["passed"] for item in all_validators), "failed": len(validator_failures)},
            "execution_mode": {
                "parallel_cases": workers > 1,
                "worker_pool_size": workers,
                "cases_wall_s": cases_wall,
                "assessment_wall_s": time.perf_counter() - started,
            },
            "coupling": "consumer_record_published",
            "limitations": value["claim_boundary"],
        }
        context.write_json("artifacts/campaign-result.json", terminal)
        collector["assessment"] = {
            "gates": _plain(gates),
            "execution_mode": terminal["execution_mode"],
            "status": status,
            "accepted": accepted,
            "headline": dataset["headline"],
            "dataset_summary": {"design_count": len(dataset_rows), "csv_bytes": len(csv_bytes), "consumer_v4_passed": consumer["passed"]},
        }
        collector["validators"] = all_validators
        collector["design_gates"] = design_gate_records
        return Decision(accepted, _plain(terminal))

    return RuntimeCallbacks(prebundle, development, assessment)


def _collect_case(case: Mapping[str, Any]) -> dict[str, Any]:
    validators = case["validators"]
    return {
        "campaign_id": case["campaign_id"],
        "case_id": case["case_id"],
        "role": case["role"],
        "timestep": case["timestep"],
        "preflight": case["preflight"],
        "diagnostics": case["diagnostics"],
        "gate_facts": case["gate_facts"],
        "timing_s": dict(case["timing_s"]),
        "validators": {
            "passed": sum(item["passed"] for item in validators),
            "failed": sum(not item["passed"] for item in validators),
            "failures": [item for item in validators if not item["passed"]],
            "names": [item["validator"] for item in validators],
        },
        "export_stage_ran": bool(case["sealed"]),
        "handoff_consumed": bool(case["consumed_handoff"] is not None and case["consumed_handoff"]["passed"]),
        "artifact_file_sha256": case["artifact_file_sha256"],
        "determinism_sample": case["determinism_sample"],
        "worker_process_id": case["process_id"],
        "summary": case["summary"],
    }
