"""Preregistered full-orbit campaign mechanics, shakedown plan and shared-runtime callbacks.

v4 differences from v3:

* One :class:`CampaignPlan` drives both the evidentiary campaign and the
  disclosed NON-EVIDENTIARY shakedown, so the shakedown exercises exactly the
  production code (adapter, ``preflight_campaign``, integration, checkpoint
  chain, validators, estimators, gate report, export, publication).
* ``orbit_mc_source_sha256`` binds the orbit_mc package sources and spec files
  into ``shakedown.json`` and ``authorities.json``.
* Cases run in a spawn process pool after all nine label accesses have been
  recorded in case order; a main-process determinism sample and the
  deterministic replay validator both re-integrate orbits from the main
  process/field object, so parallel results are proven identical.
"""

from __future__ import annotations

import gzip
import hashlib
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

import cft_revival.orbit_mc as orbit_mc_package
from cft_revival.experiment_runtime import Decision, RuntimeCallbacks
from cft_revival.experiment_runtime.canonical import (
    CanonicalizationError,
    canonical_bytes,
    semantic_sha256,
    strict_json_file,
    strict_json_loads,
)
from cft_revival.orbit_mc import (
    CHECKPOINT_VERSION,
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    HANDOFF_VERSION,
    SCHEMA_VERSION,
    AnalyticField,
    ElectronLaunch,
    EstimatorPolicy,
    OrbitConfig,
    Termination,
    analytic_magnetic_bottle,
    backend_parity,
    build_launch_ensemble,
    checkpoint,
    code_identity,
    compare_maps,
    coupling_v42_handoff,
    frozen_batch_manifest,
    load_and_verify_artifact,
    merge_checkpoint_results,
    preflight_campaign,
    reduce_results,
    result_artifact,
    timestep_convergence,
    uniform_b_helix,
    varying_e_convergence,
    wall_event_accuracy,
    wilson_interval,
    write_artifact,
    write_checkpoint,
)
from cft_revival.orbit_mc.artifacts import content_hash
from cft_revival.orbit_mc.integrator import integrate_orbit

from .adapter import build_regular_field, file_sha256

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
AUTHORITIES_PATH = EXPERIMENT / "authorities.json"
CASE_AUTHORITIES_PATH = EXPERIMENT / "case-authorities.json"
CASE_ROOT = EXPERIMENT / "cases"
SYNTHETIC_PREFLIGHT_PATH = EXPERIMENT / "synthetic-preflight.json"
SHAKEDOWN_PATH = EXPERIMENT / "shakedown.json"
RESULTS_ROOT = EXPERIMENT / "results"

VERSION_TAG = "cft-revival.cft-orbit-wall-loss-v4"
ROLES = ("primary", "refined", "enlarged")
TIMESTEPS = ("N", "2N", "4N")
PRIOR_VERSIONS = ("v1", "v2", "v3")


def schema(name: str) -> str:
    return f"{VERSION_TAG}.{name}/4.0.0"


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def protocol() -> dict[str, Any]:
    value = strict_json_file(PROTOCOL_PATH)
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    return value


# --------------------------------------------------------------------------
# orbit_mc source/contract binding
# --------------------------------------------------------------------------


def orbit_mc_source_files() -> list[Path]:
    """Every orbit_mc Python source plus every orbit_mc spec JSON in this worktree."""

    package = Path(orbit_mc_package.__file__).resolve().parent
    expected = (MODERN / "src" / "cft_revival" / "orbit_mc").resolve()
    if package != expected:
        raise RuntimeError(
            f"orbit_mc is imported from {package}, not from this worktree ({expected})"
        )
    spec = MODERN / "spec" / "orbit_mc"
    files = sorted(package.glob("*.py")) + sorted(spec.glob("*.json"))
    if not any(path.suffix == ".py" for path in files) or not any(
        path.suffix == ".json" for path in files
    ):
        raise RuntimeError("orbit_mc source hash scope is incomplete")
    return files


def orbit_mc_source_sha256() -> str:
    """SHA-256 over (posix path, LF bytes) of every orbit_mc source and spec file.

    The repository pins ``eol=lf`` for hash-bound files (fab0eccc). A CR byte in
    any scoped file means the working tree was smudged before the pin and the
    hash would not be reproducible from the committed blobs, so fail closed.
    """

    digest = hashlib.sha256()
    for path in orbit_mc_source_files():
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(
                f"orbit_mc source file {path.relative_to(MODERN).as_posix()} contains "
                "CR bytes; the source hash is defined over LF working-tree bytes"
            )
        digest.update(path.relative_to(MODERN).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _validation_protocol_schema_version() -> str:
    spec = strict_json_file(MODERN / "spec" / "orbit_mc" / "validation-protocol-v1.json")
    if not isinstance(spec, Mapping):
        raise ValueError("orbit_mc validation protocol spec must be an object")
    return str(spec["schema_version"])


def orbit_mc_contract_report(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = value["orbit_mc_contract"]
    observed = {
        "package_version": orbit_mc_package.__version__,
        "result_schema_version": SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_VERSION,
        "validation_protocol_schema_version": _validation_protocol_schema_version(),
        "handoff_schema_version": HANDOFF_VERSION,
    }
    return {
        "expected": {key: contract[key] for key in observed},
        "observed": observed,
        "matches": all(observed[key] == contract[key] for key in observed),
        "source_sha256": orbit_mc_source_sha256(),
        "code_identity_sha256": code_identity(),
        "source_files": [
            path.relative_to(MODERN).as_posix() for path in orbit_mc_source_files()
        ],
    }


def require_orbit_mc_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    report = orbit_mc_contract_report(value)
    if not report["matches"]:
        raise ValueError(
            "orbit_mc contract (package version / schema versions) differs from protocol: "
            f"expected {report['expected']}, observed {report['observed']}"
        )
    return report


# --------------------------------------------------------------------------
# Campaign plans (evidentiary and shakedown)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignPlan:
    kind: str
    campaign_id_prefix: str
    positions: tuple[tuple[str, tuple[float, float, float]], ...]
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
        if len({item[0] for item in self.positions}) != len(self.positions):
            raise ValueError("plan positions must have unique flux-surface IDs")
        if len(set(self.gyrophases_rad)) != len(self.gyrophases_rad):
            raise ValueError("plan gyrophases must be unique")
        if not 0 < self.partial_checkpoint_prefix_count < self.batch_size:
            raise ValueError("partial prefix must lie strictly inside batch 0")


def gyrophase_grid(offset_rad: float, count: int) -> tuple[float, ...]:
    return tuple(
        (float(offset_rad) + 2.0 * math.pi * index / count) % (2.0 * math.pi)
        for index in range(int(count))
    )


def evidentiary_plan(value: Mapping[str, Any]) -> CampaignPlan:
    declaration = value["launches"]
    return CampaignPlan(
        kind="evidentiary",
        campaign_id_prefix=declaration["campaign_id_prefix"],
        positions=tuple(
            (item["flux_surface_id"], tuple(float(x) for x in item["position_m"]))
            for item in declaration["position_seeds"]
        ),
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
        independent_repeats_per_stratum=int(
            declaration["independent_repeats_per_stratum"]
        ),
        binding_gates=True,
    )


def shakedown_positions(
    value: Mapping[str, Any],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Deterministic RNG positions in a seed namespace disjoint from the design."""

    declaration = value["shakedown"]
    seed = int.from_bytes(
        hashlib.sha256(
            (declaration["seed_namespace"] + ":positions").encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    wall_radius = float(value["orbit"]["wall"]["radius_m"])
    low, high = (float(item) for item in declaration["radius_fraction_range"])
    half_width = float(declaration["axial_half_width_m"])
    repeats = int(declaration["positions_per_cell"])
    labels = "abcdefgh"
    if not 0 < repeats <= len(labels):
        raise ValueError("positions_per_cell is out of range")
    positions: list[tuple[str, tuple[float, float, float]]] = []
    for cell in declaration["cells"]:
        for index in range(repeats):
            fraction = float(rng.uniform(low, high))
            axial_offset = float(rng.uniform(-half_width, half_width))
            position = (
                fraction * wall_radius,
                0.0,
                float(cell["axial_center_m"]) + axial_offset,
            )
            positions.append(
                (f"{cell['cell_id']}-r{fraction:.6f}-{labels[index]}", position)
            )
    return tuple(positions)


def shakedown_plan(value: Mapping[str, Any]) -> CampaignPlan:
    declaration = value["shakedown"]
    if declaration["evidentiary"] is not False or declaration["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown must be declared non-evidentiary")
    return CampaignPlan(
        kind="shakedown",
        campaign_id_prefix=declaration["campaign_id_prefix"],
        positions=shakedown_positions(value),
        gyrophases_rad=gyrophase_grid(
            declaration["gyrophase_offset_rad"], declaration["gyrophase_count"]
        ),
        batch_size=int(declaration["batch_size"]),
        partial_checkpoint_prefix_count=int(
            declaration["partial_checkpoint_prefix_count"]
        ),
        launches_per_case=int(declaration["launches_per_case"]),
        batches_per_case=int(declaration["batches_per_case"]),
        strata_per_case=int(declaration["strata_per_case"]),
        independent_repeats_per_stratum=int(
            declaration["independent_repeats_per_stratum"]
        ),
        binding_gates=False,
    )


def plan_record(plan: CampaignPlan) -> dict[str, Any]:
    record = asdict(plan)
    record["positions"] = [
        {"flux_surface_id": surface, "position_m": list(position)}
        for surface, position in plan.positions
    ]
    record["gyrophases_rad"] = list(plan.gyrophases_rad)
    return record


# --------------------------------------------------------------------------
# Case identities, launches, payloads
# --------------------------------------------------------------------------


def case_id(plan: CampaignPlan, role: str, timestep: str) -> str:
    if role not in ROLES:
        raise ValueError("unknown field-map role")
    if timestep not in TIMESTEPS:
        raise ValueError("unknown timestep policy")
    return f"{plan.campaign_id_prefix}:{role}:{timestep}"


def case_key(role: str, timestep: str) -> str:
    return f"{role}-{timestep}"


def case_matrix(plan: CampaignPlan) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (role, timestep, case_id(plan, role, timestep))
        for role in ROLES
        for timestep in TIMESTEPS
    )


def build_case_launches(
    value: Mapping[str, Any], plan: CampaignPlan, role: str, timestep: str
) -> tuple[ElectronLaunch, ...]:
    declaration = value["launches"]
    launches = build_launch_ensemble(
        ensemble_id=case_id(plan, role, timestep),
        energies_ev=declaration["energies_ev"],
        pitch_angles_rad=[
            math.radians(item) for item in declaration["pitch_angles_deg"]
        ],
        positions=plan.positions,
        directions=declaration["directions"],
        gyrophases_rad=plan.gyrophases_rad,
    )
    if len(launches) != plan.launches_per_case:
        raise ValueError(
            f"{plan.kind} plan produced {len(launches)} launches, "
            f"expected {plan.launches_per_case}"
        )
    return launches


def launch_records(launches: Sequence[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in sorted(launches, key=lambda item: item.launch_id)]


def runtime_launch_payload(
    campaign_id: str, launches: Sequence[Any]
) -> dict[str, Any]:
    records = launch_records(launches)
    for record in records:
        record["seed_id"] = str(record["seed_id"])
    return {
        "schema_version": schema("launches"),
        "campaign_id": campaign_id,
        "ensemble_id": campaign_id,
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "seed_encoding": "unsigned-64 decimal string",
        "launches": records,
    }


def batch_records(plan: CampaignPlan, launches: Sequence[Any]) -> list[dict[str, Any]]:
    batches = frozen_batch_manifest(
        launches,
        batch_size=plan.batch_size,
        estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
    )
    if len(batches) != plan.batches_per_case:
        raise ValueError(
            f"{plan.kind} plan produced {len(batches)} batches, "
            f"expected {plan.batches_per_case}"
        )
    return batches


def runtime_batch_payload(
    plan: CampaignPlan, campaign_id: str, launches: Sequence[Any]
) -> dict[str, Any]:
    return {
        "schema_version": schema("batches"),
        "campaign_id": campaign_id,
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
        "batches": batch_records(plan, launches),
    }


def estimator_identity(
    launches: Sequence[Any], batches: Sequence[Mapping[str, Any]]
) -> str:
    weights = {
        entry["launch_id"]: entry["weight"]
        for batch in batches
        for entry in batch["launches"]
    }
    return content_hash(
        {
            "estimand_id": "campaign_wall_loss_probability",
            "policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "launches": [
                {"launch_id": launch.launch_id, "weight": weights[launch.launch_id]}
                for launch in sorted(launches, key=lambda item: item.launch_id)
            ],
        }
    )


def _decode_runtime_tags(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_runtime_tags(item) for item in value]
    if isinstance(value, dict):
        if "__cft_type__" in value:
            if set(value) != {"__cft_type__", "items"} or value["__cft_type__"] != "tuple":
                raise ValueError("unsupported or malformed reserved runtime tag")
            if not isinstance(value["items"], list):
                raise ValueError("runtime tuple tag items must be a list")
            return tuple(_decode_runtime_tags(item) for item in value["items"])
        return {key: _decode_runtime_tags(item) for key, item in value.items()}
    return value


def load_runtime_launch_payload(
    data: bytes, expected_campaign_id: str
) -> tuple[ElectronLaunch, ...]:
    """Closed typed loader used only after exact byte authority succeeds."""

    decoded = _decode_runtime_tags(strict_json_loads(data))
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema_version",
        "campaign_id",
        "ensemble_id",
        "estimator_policy",
        "seed_encoding",
        "launches",
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
        "launch_id",
        "seed_id",
        "kinetic_energy_ev",
        "pitch_angle_rad",
        "position_m",
        "parallel_direction",
        "gyrophase_rad",
        "flux_surface_id",
    }
    for record in decoded["launches"]:
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ValueError("runtime launch record is not closed")
        seed_text = record["seed_id"]
        if (
            not isinstance(seed_text, str)
            or not seed_text
            or not seed_text.isascii()
            or not seed_text.isdecimal()
        ):
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


def field_identity(value: Mapping[str, Any], role: str) -> str:
    declaration = value["field_adapter"]["maps"][role]
    return content_hash(
        {
            "role": role,
            "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
            "sidecar_file_sha256": declaration["sidecar_file_sha256"],
            "mesh_sha256": declaration["mesh_sha256"],
            "run_sha256": declaration["run_sha256"],
        }
    )


def policy_identity(
    value: Mapping[str, Any], plan: CampaignPlan, role: str, timestep: str
) -> str:
    return content_hash(
        {
            "protocol_semantic_sha256": semantic_sha256(value),
            "plan_kind": plan.kind,
            "role": role,
            "timestep": timestep,
        }
    )


def orbit_config(value: Mapping[str, Any], role: str, timestep: str) -> OrbitConfig:
    del role
    declaration = value["orbit"]
    wall = declaration["wall"]
    domain = declaration["domain"]
    return OrbitConfig(
        wall_radius_m=wall["radius_m"],
        wall_z_min_m=wall["z_min_m"],
        wall_z_max_m=wall["z_max_m"],
        domain_radius_m=domain["radius_m"],
        domain_z_min_m=domain["z_min_m"],
        domain_z_max_m=domain["z_max_m"],
        max_time_s=declaration["max_time_s"],
        max_path_m=declaration["max_path_m"],
        max_steps=declaration["max_steps"],
        max_rotation_rad=declaration["timestep_policies"][timestep][
            "max_rotation_rad"
        ],
        event_tolerance_m=declaration["event_tolerance_m"],
        maximum_gamma=declaration["maximum_gamma"],
    )


def build_case_authority(
    value: Mapping[str, Any], plan: CampaignPlan, role: str, timestep: str
) -> dict[str, Any]:
    campaign = case_id(plan, role, timestep)
    launches = build_case_launches(value, plan, role, timestep)
    batches = batch_records(plan, launches)
    launch_bytes = canonical_bytes(runtime_launch_payload(campaign, launches))
    batch_bytes = canonical_bytes(runtime_batch_payload(plan, campaign, launches))
    record = {
        "schema_version": schema("case-authority"),
        "plan_kind": plan.kind,
        "case_key": case_key(role, timestep),
        "campaign_id": campaign,
        "ensemble_id": campaign,
        "role": role,
        "timestep": timestep,
        "launch_count": len(launches),
        "batch_count": len(batches),
        "launch_manifest_path": (
            f"cases/{case_key(role, timestep)}/launch-manifest.json"
        ),
        "batch_manifest_path": (
            f"cases/{case_key(role, timestep)}/batch-manifest.json"
        ),
        "runtime_launch_payload_byte_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "runtime_batch_payload_byte_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "orbit_launches_sha256": content_hash(launch_records(launches)),
        "batch_manifest_sha256": content_hash(
            {
                "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
                "batches": batches,
            }
        ),
        "estimator_sha256": estimator_identity(launches, batches),
        "field_identity_sha256": field_identity(value, role),
        "config_identity_sha256": content_hash(
            asdict(orbit_config(value, role, timestep))
        ),
        "policy_identity_sha256": policy_identity(value, plan, role, timestep),
    }
    record["case_authority_sha256"] = content_hash(
        {
            "campaign_id": campaign,
            "launches_sha256": record["orbit_launches_sha256"],
            "batch_manifest_sha256": record["batch_manifest_sha256"],
            "policy_sha256": record["policy_identity_sha256"],
            "minimum_certificate_tightness_ratio": value["gates"][
                "minimum_certificate_dense_to_bound_ratio"
            ],
            "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
            "estimator_sha256": record["estimator_sha256"],
            "replay_requirement": "deterministic_full_result_replay_required",
        }
    )
    record["authority_record_sha256"] = content_hash(record)
    return record


def build_all_case_authorities(
    value: Mapping[str, Any], plan: CampaignPlan
) -> dict[str, Any]:
    cases = [
        build_case_authority(value, plan, role, timestep)
        for role, timestep, _ in case_matrix(plan)
    ]
    return {
        "schema_version": schema("case-authorities"),
        "plan_kind": plan.kind,
        "case_count": len(cases),
        "total_case_launches": sum(item["launch_count"] for item in cases),
        "cases": cases,
    }


def all_plan_launches(value: Mapping[str, Any], plan: CampaignPlan) -> tuple[ElectronLaunch, ...]:
    return tuple(
        launch
        for role, timestep, _ in case_matrix(plan)
        for launch in build_case_launches(value, plan, role, timestep)
    )


def design_sha256(value: Mapping[str, Any], plan: CampaignPlan) -> str:
    return content_hash(launch_records(all_plan_launches(value, plan)))


# --------------------------------------------------------------------------
# Prior campaign designs and disjointness proofs
# --------------------------------------------------------------------------


def prior_design_launches(
    value: Mapping[str, Any], version: str
) -> tuple[ElectronLaunch, ...]:
    """Rebuild the exact v1/v2/v3 launch designs from the disclosed parameters."""

    design = value["prior_campaign_disclosure"][version]["design"]
    positions = tuple(
        (str(surface), tuple(float(x) for x in position))
        for surface, position in design["positions_m"]
    )
    phases = gyrophase_grid(design["gyrophase_offset_rad"], design["gyrophase_count"])
    if "ensemble_id" in design:
        ensemble_ids: tuple[str, ...] = (design["ensemble_id"],)
    else:
        ensemble_ids = tuple(
            f"{design['campaign_id_prefix']}:{role}:{timestep}"
            for role in ROLES
            for timestep in TIMESTEPS
        )
    declaration = value["launches"]
    return tuple(
        launch
        for ensemble_id in ensemble_ids
        for launch in build_launch_ensemble(
            ensemble_id=ensemble_id,
            energies_ev=declaration["energies_ev"],
            pitch_angles_rad=[
                math.radians(item) for item in declaration["pitch_angles_deg"]
            ],
            positions=positions,
            directions=declaration["directions"],
            gyrophases_rad=phases,
        )
    )


def design_signature(launches: Sequence[ElectronLaunch]) -> dict[str, set[Any]]:
    return {
        "launch_id": {item.launch_id for item in launches},
        "seed_id": {item.seed_id for item in launches},
        "position_m": {item.position_m for item in launches},
        "energy_pitch_direction_gyrophase": {
            (
                item.kinetic_energy_ev,
                item.pitch_angle_rad,
                item.parallel_direction,
                item.gyrophase_rad,
            )
            for item in launches
        },
        "full_phase_space": {
            (
                item.kinetic_energy_ev,
                item.pitch_angle_rad,
                item.position_m,
                item.parallel_direction,
                item.gyrophase_rad,
            )
            for item in launches
        },
    }


def disjointness_report(
    left: Sequence[ElectronLaunch],
    right: Sequence[ElectronLaunch],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_signature = design_signature(left)
    right_signature = design_signature(right)
    overlaps = {
        name: len(left_signature[name] & right_signature[name])
        for name in left_signature
    }
    return {
        "left": left_name,
        "right": right_name,
        "left_launch_count": len(left),
        "right_launch_count": len(right),
        "overlap_counts": overlaps,
        "disjoint": all(count == 0 for count in overlaps.values()),
    }


def shakedown_disjointness(value: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the shakedown design shares nothing with v4 evidentiary or v1/v2/v3."""

    shakedown = all_plan_launches(value, shakedown_plan(value))
    evidentiary = all_plan_launches(value, evidentiary_plan(value))
    reports = {
        "against_evidentiary_v4": disjointness_report(
            shakedown, evidentiary, left_name="shakedown", right_name="evidentiary-v4"
        )
    }
    for version in PRIOR_VERSIONS:
        reports[f"against_{version}"] = disjointness_report(
            shakedown,
            prior_design_launches(value, version),
            left_name="shakedown",
            right_name=version,
        )
    return {
        "shakedown_launch_count": len(shakedown),
        "shakedown_unique_launch_ids": len({item.launch_id for item in shakedown}),
        "shakedown_unique_seed_ids": len({item.seed_id for item in shakedown}),
        "reports": reports,
        "proven": (
            len({item.launch_id for item in shakedown}) == len(shakedown)
            and len({item.seed_id for item in shakedown}) == len(shakedown)
            and all(item["disjoint"] for item in reports.values())
        ),
    }


def evidentiary_disjointness(value: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the evidentiary v4 design shares nothing with v1/v2/v3 or the shakedown."""

    evidentiary = all_plan_launches(value, evidentiary_plan(value))
    reports = {
        f"against_{version}": disjointness_report(
            evidentiary,
            prior_design_launches(value, version),
            left_name="evidentiary-v4",
            right_name=version,
        )
        for version in PRIOR_VERSIONS
    }
    reports["against_shakedown"] = disjointness_report(
        evidentiary,
        all_plan_launches(value, shakedown_plan(value)),
        left_name="evidentiary-v4",
        right_name="shakedown",
    )
    return {
        "evidentiary_launch_count": len(evidentiary),
        "evidentiary_unique_launch_ids": len({item.launch_id for item in evidentiary}),
        "evidentiary_unique_seed_ids": len({item.seed_id for item in evidentiary}),
        "evidentiary_unique_positions": len({item.position_m for item in evidentiary}),
        "reports": reports,
        "proven": (
            len({item.launch_id for item in evidentiary}) == len(evidentiary)
            and len({item.seed_id for item in evidentiary}) == len(evidentiary)
            and all(item["disjoint"] for item in reports.values())
        ),
    }


# --------------------------------------------------------------------------
# Synthetic production preflight (no P2 access, no outcomes)
# --------------------------------------------------------------------------


def _synthetic_checkpoint_chain(
    value: Mapping[str, Any],
    plan: CampaignPlan,
    authority: Mapping[str, Any],
    launches: Sequence[ElectronLaunch],
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    synthetic_config = OrbitConfig(
        wall_radius_m=0.0009,
        wall_z_min_m=0.001,
        wall_z_max_m=0.022,
        domain_radius_m=0.001,
        domain_z_min_m=0.0,
        domain_z_max_m=0.024,
        max_time_s=1.0e-12,
        max_path_m=1.0e-6,
        max_steps=2,
        max_rotation_rad=0.16,
        event_tolerance_m=1.0e-9,
        maximum_gamma=20.0,
    )
    synthetic_field = AnalyticField(
        lambda _position: np.array([0.0, 0.0, 0.1]),
        None,
        0.1,
    )
    results = tuple(
        integrate_orbit(item, synthetic_field, synthetic_config)
        for item in launches
    )
    if any(item.termination is not Termination.INITIAL_STATE_INVALID for item in results):
        raise ValueError("synthetic result was not immediate initial-state termination")
    common = {
        "field_identity_sha256": authority["field_identity_sha256"],
        "config_identity_sha256": authority["config_identity_sha256"],
        "policy_identity_sha256": authority["policy_identity_sha256"],
        "minimum_certificate_tightness_ratio_authority": value["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        "expected_batch_manifest_sha256": authority["batch_manifest_sha256"],
    }
    prefix = plan.partial_checkpoint_prefix_count
    two_batches = 2 * plan.batch_size
    partial = checkpoint(
        authority["campaign_id"],
        (),
        launches,
        results[:prefix],
        batches,
        partial_current_batch={
            "batch_id": 0,
            "completed_launch_ids": [
                entry["launch_id"] for entry in batches[0]["launches"][:prefix]
            ],
        },
        **common,
    )
    resumed = checkpoint(
        authority["campaign_id"],
        (0,),
        launches,
        results[: plan.batch_size],
        batches,
        previous_checkpoint_sha256=content_hash(partial),
        **common,
    )
    two_batch = checkpoint(
        authority["campaign_id"],
        (0, 1),
        launches,
        results[:two_batches],
        batches,
        previous_checkpoint_sha256=content_hash(resumed),
        **common,
    )
    final = checkpoint(
        authority["campaign_id"],
        tuple(range(len(batches))),
        launches,
        results,
        batches,
        previous_checkpoint_sha256=content_hash(two_batch),
        **common,
    )
    external = {
        "expected_campaign_id": authority["campaign_id"],
        "expected_launches_sha256": authority["orbit_launches_sha256"],
        "expected_batch_manifest_sha256": authority["batch_manifest_sha256"],
        "expected_policy_sha256": authority["policy_identity_sha256"],
        "expected_estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        "expected_minimum_certificate_tightness_ratio": value["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
    }
    merged_resume = merge_checkpoint_results(partial, resumed, **external)
    merged_two = merge_checkpoint_results(resumed, two_batch, **external)
    merged_final = merge_checkpoint_results(two_batch, final, **external)
    checks = {
        "partial_prefix": partial["coverage"]["completed_launches"] == prefix,
        "resumed_one_batch": len(merged_resume) == plan.batch_size,
        "resumed_two_batches": len(merged_two) == two_batches,
        "final_complete": len(merged_final) == plan.launches_per_case,
        "no_pending_final": final["pending_launch_ids"] == [],
        "case_authority_bound": (
            final["authority"]["campaign_identity_sha256"]
            == authority["case_authority_sha256"]
        ),
    }
    return {
        "campaign_id": authority["campaign_id"],
        "partial_checkpoint_sha256": content_hash(partial),
        "resumed_checkpoint_sha256": content_hash(resumed),
        "final_checkpoint_sha256": content_hash(final),
        "checks": checks,
        "passed": all(checks.values()),
    }


def production_synthetic_preflight(
    value: Mapping[str, Any], case_authorities: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate all nine evidentiary authorities and chains without P2 access."""

    plan = evidentiary_plan(value)
    if case_authorities["plan_kind"] != "evidentiary":
        raise ValueError("synthetic preflight requires evidentiary case authorities")
    first_authority = case_authorities["cases"][0]
    first_launches = build_case_launches(
        value, plan, first_authority["role"], first_authority["timestep"]
    )
    sample = first_launches[0]
    matrix = {
        "launch": runtime_launch_payload(first_authority["campaign_id"], (sample,)),
        "final_state": {
            "position_m": (0.0012, -2.0e-6, 0.003),
            "velocity_m_per_s": (1.0, 2.0, 3.0),
        },
        "wall_endpoint_m": (0.002, 0.0, 0.004),
        "event_witness": {
            "step_start_position_m": (0.001, 0.0, 0.003),
            "step_end_position_m": (0.0021, 0.0, 0.0032),
            "event_position_m": (0.002, 0.0, 0.00318),
            "step_start_velocity_m_per_s": (1.0, 0.0, 2.0),
            "step_end_velocity_m_per_s": (0.5, 0.5, 2.0),
            "event_velocity_m_per_s": (1.0, 0.0, 2.0),
            "step_magnetic_midpoint_t": (0.0, -0.01, 0.2),
            "step_electric_midpoint_v_per_m": (0.0, 0.0, 0.0),
            "event_resolution": "tolerance_close_fraction_zero",
            "condition": "tolerance_close_wall_radial",
            "candidate_fractions": {
                "wall_hit": 0.9,
                "reflected": None,
                "domain_escape": None,
                "time_timeout": None,
                "path_timeout": None,
            },
            "reflection_bracket": None,
        },
        "failure_event_witness_v1_6_zero_vectors": {
            "event_velocity_m_per_s": (0.0, 0.0, 0.0),
            "step_magnetic_midpoint_t": (0.0, 0.0, 0.0),
            "step_electric_midpoint_v_per_m": (0.0, 0.0, 0.0),
            "event_resolution": "failure",
            "event_fraction": 0.0,
        },
        "gyro_averages": (
            {
                "cycle_index": 0,
                "phase_interval_rad": (0.0, 2.0 * math.pi),
                "mu_j_per_t": 1.0e-18,
            },
        ),
        "termination_counts": (
            ("wall_hit", 1),
            ("reflected", 2),
            ("domain_escape", 3),
            ("time_timeout", 4),
        ),
        "batch_checkpoint_ids": {
            "completed_batch_ids": (0, 1),
            "pending_launch_ids": [sample.launch_id],
            "checkpoint_chain": ("partial", "final"),
        },
        "p2_field_arrays": {
            "r_m": [0.0, 0.001, 0.002],
            "z_m": [0.001, 0.002, 0.003],
            "psi_wb": [[0.0, 0.0], [1.0e-8, 2.0e-8]],
            "material_id": [["plasma", "plasma"], ["plasma", "plasma"]],
        },
        "protocol_vectors": {
            "directions": tuple(value["launches"]["directions"]),
            "first_position": tuple(value["launches"]["position_seeds"][0]["position_m"]),
            "timestep_levels": list(value["orbit"]["timestep_policies"]),
            "shakedown_gyrophases": shakedown_plan(value).gyrophases_rad,
        },
    }
    encoded = canonical_bytes(matrix)
    decoded = _decode_runtime_tags(strict_json_loads(encoded))
    if decoded != matrix:
        raise ValueError("synthetic production vector roundtrip differs")
    chain_reports = []
    payload_checks = []
    for authority in case_authorities["cases"]:
        launches = build_case_launches(
            value, plan, authority["role"], authority["timestep"]
        )
        batches = batch_records(plan, launches)
        launch_bytes = canonical_bytes(
            runtime_launch_payload(authority["campaign_id"], launches)
        )
        batch_payload = runtime_batch_payload(
            plan, authority["campaign_id"], launches
        )
        batch_bytes = canonical_bytes(batch_payload)
        loaded = load_runtime_launch_payload(
            launch_bytes, authority["campaign_id"]
        )
        payload_ok = (
            loaded == tuple(sorted(launches, key=lambda item: item.launch_id))
            and strict_json_loads(batch_bytes) == batch_payload
            and hashlib.sha256(launch_bytes).hexdigest()
            == authority["runtime_launch_payload_byte_sha256"]
            and hashlib.sha256(batch_bytes).hexdigest()
            == authority["runtime_batch_payload_byte_sha256"]
            and estimator_identity(launches, batches)
            == authority["estimator_sha256"]
        )
        payload_checks.append(payload_ok)
        chain_reports.append(
            _synthetic_checkpoint_chain(value, plan, authority, launches, batches)
        )
    disjointness = evidentiary_disjointness(value)
    reserved_input_rejected = False
    try:
        canonical_bytes({"__cft_type__": "tuple", "items": []})
    except CanonicalizationError:
        reserved_input_rejected = True
    tagged_parse = strict_json_loads(canonical_bytes((1, 2, 3)))
    tagged_reencode_rejected = False
    try:
        canonical_bytes(tagged_parse)
    except CanonicalizationError:
        tagged_reencode_rejected = True
    malformed_tag_rejected = False
    try:
        _decode_runtime_tags(
            {"__cft_type__": "tuple", "items": [], "unexpected": True}
        )
    except ValueError:
        malformed_tag_rejected = True
    contract = orbit_mc_contract_report(value)
    expected_total = 9 * plan.launches_per_case
    checks = {
        "all_vector_fields_roundtrip": True,
        "all_case_payloads_roundtrip": all(payload_checks),
        "all_nine_checkpoint_chains": (
            len(chain_reports) == 9
            and all(item["passed"] for item in chain_reports)
        ),
        "all_case_ids_and_seeds_unique": (
            disjointness["evidentiary_unique_launch_ids"] == expected_total
            and disjointness["evidentiary_unique_seed_ids"] == expected_total
        ),
        "zero_v1_overlap": disjointness["reports"]["against_v1"]["disjoint"],
        "zero_v2_overlap": disjointness["reports"]["against_v2"]["disjoint"],
        "zero_v3_overlap": disjointness["reports"]["against_v3"]["disjoint"],
        "zero_shakedown_overlap": disjointness["reports"]["against_shakedown"]["disjoint"],
        "orbit_mc_contract_matches": contract["matches"],
        "reserved_input_rejected": reserved_input_rejected,
        "parsed_tag_reencode_rejected": tagged_reencode_rejected,
        "malformed_tag_rejected": malformed_tag_rejected,
    }
    return {
        "schema_version": schema("synthetic-production-preflight"),
        "covered_fields": [
            "launch_position_and_seed",
            "final_position_and_velocity",
            "wall_endpoint",
            "event_witness_vectors_and_candidates",
            "event_witness_v1_6_event_velocity_and_midpoint_fields",
            "failure_event_witness_v1_6_zero_vectors",
            "gyro_averages",
            "termination_counts",
            "batch_and_checkpoint_ids",
            "p2_field_arrays",
            "protocol_vectors",
        ],
        "matrix_byte_sha256": hashlib.sha256(encoded).hexdigest(),
        "case_checkpoint_chains": chain_reports,
        "overlap_evidence": disjointness,
        "orbit_mc_contract": contract,
        "checks": checks,
        "passed": all(checks.values()),
    }


def manufactured_gate_report(value: Mapping[str, Any]) -> dict[str, Any]:
    limits = value["gates"]
    helix = timestep_convergence()
    varying = varying_e_convergence()
    mirror = analytic_magnetic_bottle()
    energy = uniform_b_helix()
    wall = wall_event_accuracy()
    cpu = backend_parity(device="cpu")
    cuda = backend_parity(device="cuda:0")
    checks = {
        "uniform_b_energy": energy["relative_energy_error"]
        <= limits["maximum_relative_energy_error"],
        "helix_order": min(helix["observed_orders"])
        >= limits["minimum_helix_position_order"],
        "varying_e_order": min(varying["observed_orders"])
        >= limits["minimum_varying_e_position_order"],
        "mirror_smoke": mirror["relative_error"]
        <= limits["maximum_mirror_point_relative_error"],
        "wall_endpoint": wall["endpoint_error_m"]
        <= limits["maximum_wall_endpoint_error_m"],
        "cpu_parity": cpu["status"] == "evaluated"
        and cpu["maximum_relative_velocity_difference"]
        <= limits["maximum_cpu_cuda_relative_velocity_difference"],
        "cuda_parity": cuda["status"] == "evaluated"
        and cuda["maximum_relative_velocity_difference"]
        <= limits["maximum_cpu_cuda_relative_velocity_difference"],
    }
    return _plain({
        "checks": checks,
        "passed": all(checks.values()),
        "uniform_b": energy,
        "helix_convergence": helix,
        "varying_e_convergence": varying,
        "mirror": mirror,
        "wall_event": wall,
        "cpu_parity": cpu,
        "cuda_parity": cuda,
    })


def validate_frozen_input_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    authority = value["authority"]
    manifest_path = REPOSITORY / authority["manifest"]["path"]
    result_path = REPOSITORY / authority["result"]["path"]
    if (
        file_sha256(manifest_path) != authority["manifest"]["file_sha256"]
        or file_sha256(result_path) != authority["result"]["file_sha256"]
    ):
        raise ValueError("frozen P2 manifest/result bytes differ")
    manifest = strict_json_file(manifest_path)
    rows = [
        item
        for item in manifest["designs"]
        if item["config_id"] == "divergent-exit-stack-v1"
    ]
    if (
        len(rows) != 1
        or rows[0]["qualification_status"] != authority["required_qualification"]
        or set(authority["excluded_designs"])
        != {"historical-envelope-baseline", "compact-high-gradient-stack"}
    ):
        raise ValueError("P2 qualification/design authority differs")
    return {
        "manifest_file_sha256": authority["manifest"]["file_sha256"],
        "result_file_sha256": authority["result"]["file_sha256"],
        "qualification_status": rows[0]["qualification_status"],
        "design_id": authority["design_id"],
    }


# --------------------------------------------------------------------------
# Estimators, convergence and gates
# --------------------------------------------------------------------------


def _probability(successes: int, trials: int) -> dict[str, Any]:
    return asdict(wilson_interval(successes, trials))


def stratum_summaries(
    launches: Sequence[Any], results: Sequence[Any]
) -> list[dict[str, Any]]:
    authority = {item.launch_id: item for item in launches}
    groups: dict[tuple[Any, ...], list[Any]] = {}
    for result in results:
        launch = authority[result.launch_id]
        cell_id = launch.flux_surface_id.split("-r", 1)[0]
        key = (
            cell_id,
            launch.kinetic_energy_ev,
            round(math.degrees(launch.pitch_angle_rad), 12),
            launch.parallel_direction,
        )
        groups.setdefault(key, []).append(result)
    output = []
    for key in sorted(groups):
        rows = groups[key]
        counts = {
            termination.value: sum(item.termination is termination for item in rows)
            for termination in Termination
        }
        n = len(rows)
        timeout = n - counts["wall_hit"] - counts["reflected"] - counts["domain_escape"]
        repeat_ids = {
            authority[item.launch_id].flux_surface_id for item in rows
        }
        output.append(
            {
                "cell_id": key[0],
                "kinetic_energy_ev": key[1],
                "pitch_angle_deg": key[2],
                "parallel_direction": key[3],
                "trials": n,
                "physical_position_repeat_count": len(repeat_ids),
                "termination_counts": counts,
                "wall_hit": _probability(counts["wall_hit"], n),
                "reflected": _probability(counts["reflected"], n),
                "domain_escape": _probability(counts["domain_escape"], n),
                "timeout": _probability(timeout, n),
            }
        )
    return output


def _interval_overlap(left: Any, right: Any) -> bool:
    return max(left.wall_hit.lower, right.wall_hit.lower) <= min(
        left.wall_hit.upper, right.wall_hit.upper
    )


def _convergence(
    summaries: Mapping[tuple[str, str], Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    maximum = value["gates"]["maximum_successive_probability_change"]
    timestep_rows = []
    for role in ROLES:
        ordered = [summaries[(role, step)] for step in TIMESTEPS]
        # v3 zipped ``ordered`` against ``ordered[1:]`` with strict=True, which
        # always raises; v3 never reached this line. Found by the v4 shakedown.
        changes = [
            abs(right.wall_hit.probability - left.wall_hit.probability)
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        overlaps = [
            _interval_overlap(left, right)
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        timestep_rows.append(
            {
                "map_role": role,
                "probabilities": [item.wall_hit.probability for item in ordered],
                "successive_changes": changes,
                "adjacent_wilson_overlap": overlaps,
                "passed": max(changes) <= maximum and all(overlaps),
            }
        )
    map_rows = []
    for step in TIMESTEPS:
        ordered = [summaries[(role, step)] for role in ROLES]
        changes = [
            abs(right.wall_hit.probability - left.wall_hit.probability)
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        overlaps = [
            _interval_overlap(left, right)
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        map_rows.append(
            {
                "timestep_policy": step,
                "probabilities": [item.wall_hit.probability for item in ordered],
                "successive_changes": changes,
                "adjacent_wilson_overlap": overlaps,
                "passed": max(changes) <= maximum and all(overlaps),
            }
        )
    return {
        "timestep": timestep_rows,
        "cross_map": map_rows,
        "timestep_passed": all(item["passed"] for item in timestep_rows),
        "cross_map_passed": all(item["passed"] for item in map_rows),
    }


def _result_gate_report(
    campaigns: Mapping[tuple[str, str], Mapping[str, Any]],
    convergence: Mapping[str, Any],
    manufactured: Mapping[str, Any],
    field_evidence: Mapping[str, Mapping[str, Any]],
    map_comparisons: Mapping[str, Mapping[str, Any]],
    value: Mapping[str, Any],
    plan: CampaignPlan,
) -> dict[str, Any]:
    all_results = [
        item for campaign in campaigns.values() for item in campaign["results"]
    ]
    incomplete = {
        termination.value: sum(item.termination is termination for item in all_results)
        for termination in (
            Termination.PATH_TIMEOUT,
            Termination.TIME_TIMEOUT,
            Termination.STEP_LIMIT,
            Termination.NONFINITE_STATE,
            Termination.EXTREME_RELATIVITY,
            Termination.FIELD_FAILURE,
            Termination.INITIAL_STATE_INVALID,
        )
    }
    wall_errors = [
        abs(math.hypot(*item.wall_endpoint_m[:2]) - value["orbit"]["wall"]["radius_m"])
        for item in all_results
        if item.wall_endpoint_m is not None
    ]
    witness_order = all(
        item.event_witness["event_fraction"]
        <= min(
            [
                candidate
                for candidate in item.event_witness["candidate_fractions"].values()
                if candidate is not None
            ]
            or [item.event_witness["event_fraction"]]
        )
        + 64.0 * np.finfo(float).eps
        for item in all_results
        if "candidate_fractions" in item.event_witness
    )
    maximum_energy = max(item.maximum_relative_energy_error for item in all_results)
    velocity_mismatches = sum(
        not _final_velocity_equals_event_velocity(item) for item in all_results
    )
    checks = {
        "manufactured": manufactured["passed"],
        "field_adapter": all(item["passed"] for item in field_evidence.values()),
        "field_map_convergence": all(
            item["b_relative_rms"]
            <= value["field_adapter"]["maximum_cross_map_b_relative_rms"]
            for item in map_comparisons.values()
        ),
        "campaign_preflight": all(
            campaign["preflight"]["status"] == "passed"
            and campaign["preflight"]["maximum_launch_b_t"]
            <= campaign["preflight"]["maximum_declared_b_t"]
            for campaign in campaigns.values()
        ),
        "timestep_probability_convergence": convergence["timestep_passed"],
        "cross_map_probability_convergence": convergence["cross_map_passed"],
        "zero_incomplete_or_numerical_failures": sum(incomplete.values()) == 0,
        "energy": maximum_energy <= value["gates"]["maximum_relative_energy_error"],
        "final_velocity_equals_event_velocity": (
            not value["gates"]["require_final_velocity_equals_event_velocity"]
            or velocity_mismatches == 0
        ),
        "wall_endpoint": max(wall_errors, default=0.0)
        <= value["gates"]["maximum_wall_endpoint_error_m"],
        "earliest_event": witness_order,
        "runtime_rotation": all(
            item.dt_s
            * abs(ELECTRON_CHARGE_C)
            * campaign["field"].max_b_t
            / ELECTRON_MASS_KG
            <= campaign["config"].max_rotation_rad * (1.0 + 1.0e-14)
            for campaign in campaigns.values()
            for item in campaign["results"]
        ),
        "relativistic_phase": all(
            math.isfinite(item.accumulated_gyro_phase_rad)
            and math.isfinite(float(item.event_witness.get("observed_gamma", 1.0)))
            for item in all_results
        ),
        "material_quarantine": all(
            bool(np.all(campaign["field"].traversable_cells))
            for campaign in campaigns.values()
        ),
        "independent_repeats": all(
            row["physical_position_repeat_count"]
            >= plan.independent_repeats_per_stratum
            for campaign in campaigns.values()
            for row in campaign["strata"]
        ),
    }
    return {
        "checks": checks,
        "binding": plan.binding_gates,
        "passed_before_replay": all(checks.values()),
        "incomplete_and_failure_counts": incomplete,
        "maximum_relative_energy_error": maximum_energy,
        "energy_gate_limit": value["gates"]["maximum_relative_energy_error"],
        "orbits_exceeding_energy_gate": sum(
            item.maximum_relative_energy_error
            > value["gates"]["maximum_relative_energy_error"]
            for item in all_results
        ),
        "final_velocity_event_velocity_mismatches": velocity_mismatches,
        "maximum_wall_endpoint_error_m": max(wall_errors, default=0.0),
        "diagnostics_not_gates": {
            "magnetic_moment_variation": mu_diagnostic(all_results),
        },
    }


def _final_velocity_equals_event_velocity(result: Any) -> bool:
    """v1.6 contract: the result's final velocity IS the witnessed event velocity."""

    witness = result.event_witness
    if "event_velocity_m_per_s" not in witness:
        return False
    return tuple(map(float, result.final_velocity_m_per_s)) == tuple(
        map(float, witness["event_velocity_m_per_s"])
    )


def mu_diagnostic(results: Sequence[Any]) -> dict[str, Any]:
    """Magnetic-moment variation summary. DIAGNOSTIC ONLY: never a gate.

    Non-adiabatic mu near the cusps is the measured physics of the divergent-exit
    field; no threshold here accepts or rejects anything.
    """

    values = sorted(
        float(item.maximum_instantaneous_mu_relative_variation)
        for item in results
        if item.maximum_instantaneous_mu_relative_variation is not None
    )
    return {
        "role": "diagnostic_only",
        "binding": False,
        "informational_gate": False,
        "orbit_count_with_mu": len(values),
        "orbit_count_without_mu": len(results) - len(values),
        "min": values[0] if values else None,
        "median": statistics.median(values) if values else None,
        "max": values[-1] if values else None,
        "count_above_0p1": sum(item > 0.1 for item in values),
        "count_above_0p5": sum(item > 0.5 for item in values),
    }


# --------------------------------------------------------------------------
# Validator ledger and per-case workers (run inside a spawn process pool)
# --------------------------------------------------------------------------


class ValidatorLedger:
    """Record every validator call; failures are recorded and re-raised."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def run(
        self, case: str, validator: str, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        started = time.perf_counter()
        try:
            output = function(*args, **kwargs)
        except Exception as error:
            self.records.append(
                {
                    "case_key": case,
                    "validator": validator,
                    "passed": False,
                    "message": f"{type(error).__name__}: {error}"[:4096],
                    "seconds": time.perf_counter() - started,
                }
            )
            raise
        self.records.append(
            {
                "case_key": case,
                "validator": validator,
                "passed": True,
                "message": "",
                "seconds": time.perf_counter() - started,
            }
        )
        return output


def result_record(result: Any) -> dict[str, Any]:
    value = asdict(result)
    value["termination"] = result.termination.value
    return value


def _counts(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def result_diagnostics(results: Sequence[Any]) -> dict[str, Any]:
    steps = [item.steps for item in results]
    conditions: dict[str, int] = {}
    tolerance_close = 0
    for item in results:
        if item.event_witness.get("event_resolution") == "tolerance_close_fraction_zero":
            tolerance_close += 1
            condition = str(item.event_witness.get("condition"))
            conditions[condition] = conditions.get(condition, 0) + 1
    return {
        "termination_counts": {
            termination.value: sum(item.termination is termination for item in results)
            for termination in Termination
        },
        "tolerance_close_event_count": tolerance_close,
        "tolerance_close_conditions": dict(sorted(conditions.items())),
        "steps": {
            "min": min(steps),
            "median": statistics.median(steps),
            "max": max(steps),
            "total": sum(steps),
        },
        "maximum_relative_energy_error": max(
            item.maximum_relative_energy_error for item in results
        ),
        "orbits_with_nonzero_energy_error": sum(
            item.maximum_relative_energy_error != 0.0 for item in results
        ),
        "final_velocity_equals_event_velocity_count": sum(
            _final_velocity_equals_event_velocity(item) for item in results
        ),
        "event_resolution_counts": _counts(
            str(item.event_witness.get("event_resolution")) for item in results
        ),
        "magnetic_moment_variation_diagnostic": mu_diagnostic(results),
        "runtime_max_b_t": max(item.maximum_b_t for item in results),
        "wall_endpoint_error_max_m": max(
            (
                abs(math.hypot(*item.wall_endpoint_m[:2]) - item.event_witness["config"]["wall_radius_m"])
                for item in results
                if item.wall_endpoint_m is not None
            ),
            default=0.0,
        ),
    }


def _checkpoint_authority_kwargs(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "field_identity_sha256": task["field_sha"],
        "config_identity_sha256": task["config_sha"],
        "policy_identity_sha256": task["policy_sha"],
        "minimum_certificate_tightness_ratio_authority": task["tightness_floor"],
        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        "expected_batch_manifest_sha256": task["batch_sha"],
    }


def _external_expectations(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_campaign_id": task["campaign_id"],
        "expected_launches_sha256": task["launch_sha"],
        "expected_batch_manifest_sha256": task["batch_sha"],
        "expected_policy_sha256": task["policy_sha"],
        "expected_estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        "expected_minimum_certificate_tightness_ratio": task["tightness_floor"],
    }


def run_case_integration(task: Mapping[str, Any]) -> dict[str, Any]:
    """Preflight, integrate one case in frozen batch order and build its checkpoint chain."""

    ledger = ValidatorLedger()
    key = task["case_key"]
    campaign_id = task["campaign_id"]
    launches = task["launches"]
    batches = task["batches"]
    field = task["field"]
    config = task["config"]
    prefix = int(task["partial_checkpoint_prefix_count"])
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    preflight = ledger.run(key, "preflight_campaign", preflight_campaign, launches, field, config)
    preflight_seconds = time.perf_counter() - started
    ordered = sorted(launches, key=lambda item: item.launch_id)
    by_id = {item.launch_id: item for item in ordered}
    results: list[Any] = []
    partial_hash = None
    latest_payload = None
    checkpoint_records: list[dict[str, Any]] = []
    integration_seconds = 0.0
    checkpoint_seconds = 0.0
    authority_kwargs = _checkpoint_authority_kwargs(task)
    external = _external_expectations(task)
    for batch in batches:
        batch_id = int(batch["batch_id"])
        for entry in batch["launches"]:
            item = by_id[entry["launch_id"]]
            tick = time.perf_counter()
            results.append(integrate_orbit(item, field, config))
            integration_seconds += time.perf_counter() - tick
            if batch_id == 0 and len(results) == prefix:
                tick = time.perf_counter()
                partial_payload = ledger.run(
                    key,
                    "checkpoint_partial_build",
                    checkpoint,
                    campaign_id,
                    (),
                    launches,
                    results,
                    batches,
                    partial_current_batch={
                        "batch_id": 0,
                        "completed_launch_ids": [
                            row["launch_id"] for row in batches[0]["launches"][:prefix]
                        ],
                    },
                    **authority_kwargs,
                )
                partial_path = work_dir / f"{key}-partial.json"
                partial_hash = ledger.run(
                    key,
                    "checkpoint_partial_write",
                    write_checkpoint,
                    partial_path,
                    partial_payload,
                    **external,
                )
                checkpoint_records.append(
                    {
                        "stage": "partial",
                        "batch_id": 0,
                        "completed_launches": prefix,
                        "path": str(partial_path),
                        "file_sha256": partial_hash,
                        "artifact_name": f"{key}-partial.json.gz",
                    }
                )
                latest_payload = partial_payload
                checkpoint_seconds += time.perf_counter() - tick
        if latest_payload is None:
            raise RuntimeError("partial checkpoint was not created")
        tick = time.perf_counter()
        batch_payload = ledger.run(
            key,
            f"checkpoint_batch_{batch_id:02d}_build",
            checkpoint,
            campaign_id,
            tuple(range(batch_id + 1)),
            launches,
            results,
            batches,
            previous_checkpoint_sha256=content_hash(latest_payload),
            **authority_kwargs,
        )
        merged = ledger.run(
            key,
            f"checkpoint_batch_{batch_id:02d}_merge",
            merge_checkpoint_results,
            latest_payload,
            batch_payload,
            **external,
        )
        if len(merged) != len(results):
            raise RuntimeError("sequential checkpoint coverage differs")
        batch_path = work_dir / f"{key}-batch-{batch_id:02d}.json"
        batch_hash = ledger.run(
            key,
            f"checkpoint_batch_{batch_id:02d}_write",
            write_checkpoint,
            batch_path,
            batch_payload,
            **external,
        )
        checkpoint_records.append(
            {
                "stage": "batch",
                "batch_id": batch_id,
                "completed_launches": len(results),
                "path": str(batch_path),
                "file_sha256": batch_hash,
                "artifact_name": f"{key}-batch-{batch_id:02d}.json.gz",
            }
        )
        latest_payload = batch_payload
        checkpoint_seconds += time.perf_counter() - tick
    if latest_payload is None or latest_payload["pending_launch_ids"]:
        raise RuntimeError("final checkpoint is not complete")
    if len(results) != len(launches):
        raise RuntimeError("case integration did not cover every launch")
    summary = ledger.run(key, "reduce_results", reduce_results, campaign_id, results)
    strata = stratum_summaries(launches, results)
    return {
        "case_key": key,
        "campaign_id": campaign_id,
        "process_id": os.getpid(),
        "preflight": preflight,
        "results": tuple(results),
        "summary": summary,
        "strata": strata,
        "checkpoints": checkpoint_records,
        "partial_checkpoint_file_sha256": partial_hash,
        "final_checkpoint_file_sha256": checkpoint_records[-1]["file_sha256"],
        "validators": ledger.records,
        "diagnostics": result_diagnostics(results),
        "timing_s": {
            "preflight": preflight_seconds,
            "integration": integration_seconds,
            "checkpoints": checkpoint_seconds,
            "total": time.perf_counter() - started,
            "per_orbit_ms": 1000.0 * integration_seconds / max(1, len(results)),
        },
    }


def run_case_export(task: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one case artifact with deterministic replay, verified reload and handoff."""

    ledger = ValidatorLedger()
    key = task["case_key"]
    field = task["field"]
    config = task["config"]
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    field_evidence = task["field_evidence"]
    results = task["results"]
    artifact = ledger.run(
        key,
        "result_artifact",
        result_artifact,
        campaign_id=task["campaign_id"],
        field_identity_sha256=task["field_sha"],
        config_identity_sha256=task["config_sha"],
        policy_identity_sha256=task["policy_sha"],
        minimum_certificate_tightness_ratio_authority=task["tightness_floor"],
        estimator_policy=EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        launches=task["launches"],
        results=results,
        batch_manifest=task["batches"],
        summary=task["summary"],
        interpolation_evidence={
            "certified_max_b_t": field.certified_max_b_t,
            "reference_max_b_t": field.reference_max_b_t,
            "runtime_max_seen_t": max(item.maximum_b_t for item in results),
            "dense_diagnostic_max_b_t": field.certificate_tightness.dense_diagnostic_max_b_t,
            "certificate_tightness_ratio": field.certificate_tightness.dense_to_bound_ratio,
            "minimum_certificate_tightness_ratio": task["tightness_floor"],
            "certificate_preflight_passed": field.certificate_tightness.preflight_passed,
            "material_map_sha256": field.material_map_sha256,
            "field_error_report": field_evidence["field_error_report"],
            "passed": True,
        },
        convergence_evidence=dict(task["convergence_evidence"]),
        preregistration=dict(task["preregistration"]),
    )
    target = work_dir / f"{key}-orbit.json"
    replay_kwargs = {
        "field": field,
        "config": config,
        "expected_field_sha256": task["field_sha"],
        "expected_config_sha256": task["config_sha"],
        "expected_launches_sha256": task["launch_sha"],
        "expected_batch_manifest_sha256": task["batch_sha"],
        "expected_policy_sha256": task["policy_sha"],
        "expected_estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL,
        "expected_minimum_certificate_tightness_ratio": task["tightness_floor"],
    }
    tick = time.perf_counter()
    evidence = ledger.run(
        key, "write_artifact_deterministic_replay", write_artifact, target, artifact, **replay_kwargs
    )
    write_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    verified = ledger.run(
        key,
        "load_and_verify_artifact_replay",
        load_and_verify_artifact,
        target,
        expected_file_sha256=evidence.file_sha256,
        **replay_kwargs,
    )
    verify_seconds = time.perf_counter() - tick
    handoff = None
    if task["export_handoff"]:
        handoff = ledger.run(
            key,
            "coupling_v42_handoff",
            coupling_v42_handoff,
            verified,
            expected_batch_manifest_sha256=task["batch_sha"],
        )
    return {
        "case_key": key,
        "campaign_id": task["campaign_id"],
        "process_id": os.getpid(),
        "artifact_path": str(target),
        "artifact_sidecar_path": str(target.with_name(target.name + ".sha256")),
        "artifact_file_sha256": evidence.file_sha256,
        "verified_file_sha256": verified.file_sha256,
        "handoff": handoff,
        "validators": ledger.records,
        "timing_s": {
            "write_artifact_replay": write_seconds,
            "load_and_verify_replay": verify_seconds,
            "total": time.perf_counter() - started,
        },
    }


def run_stage(
    tasks: Sequence[Mapping[str, Any]],
    function: Callable[[Mapping[str, Any]], dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    """Run case tasks in submission order; results are returned in the same order."""

    if workers <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        return [future.result() for future in futures]


def worker_count(value: Mapping[str, Any]) -> int:
    execution = value["execution"]
    if not execution["parallel_cases"]:
        return 1
    return max(1, min(int(execution["max_case_workers"]), os.cpu_count() or 1))


# --------------------------------------------------------------------------
# Shakedown record verification (used by prepare, execute and the prebundle)
# --------------------------------------------------------------------------


def verify_shakedown_record(
    value: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the shakedown proves the current protocol and orbit_mc."""

    checks: dict[str, bool] = {}
    checks["schema_version"] = record.get("schema_version") == schema("shakedown")
    checks["declared_non_evidentiary"] = record.get("evidentiary") is False
    checks["passed"] = record.get("passed") is True
    checks["protocol_semantic_sha256_current"] = (
        record.get("protocol_semantic_sha256") == semantic_sha256(value)
    )
    checks["orbit_mc_source_sha256_current"] = (
        record.get("orbit_mc_source_sha256") == orbit_mc_source_sha256()
    )
    contract = orbit_mc_contract_report(value)
    checks["orbit_mc_schema_versions_current"] = (
        record.get("orbit_mc_schema_versions") == contract["observed"] and contract["matches"]
    )
    disjointness = record.get("disjointness")
    checks["disjointness_proven"] = (
        isinstance(disjointness, Mapping)
        and disjointness.get("proven") is True
        and all(
            report["disjoint"] is True
            and all(count == 0 for count in report["overlap_counts"].values())
            for report in disjointness.get("reports", {}).values()
        )
        and set(disjointness.get("reports", {}))
        == {"against_evidentiary_v4", "against_v1", "against_v2", "against_v3"}
    )
    try:
        checks["shakedown_design_sha256_current"] = (
            record.get("shakedown_design_sha256")
            == design_sha256(value, shakedown_plan(value))
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
    cases = record.get("cases")
    checks["nine_cases"] = isinstance(cases, Mapping) and set(cases) == {
        case_key(role, timestep) for role in ROLES for timestep in TIMESTEPS
    }
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
            for item in cases.values()
        )
    )
    runtime = record.get("runtime")
    checks["runtime_accepted_and_bundle_validated"] = (
        isinstance(runtime, Mapping)
        and runtime.get("terminal_state") == "accepted_result"
        and runtime.get("bundle_validated") is True
    )
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ValueError(
            "shakedown gate refused: " + ", ".join(failed)
        )
    return checks


# --------------------------------------------------------------------------
# Shared runtime callbacks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenAuthority:
    authorities: Mapping[str, Any]
    case_authorities: Mapping[str, Any]
    synthetic: Mapping[str, Any]
    shakedown: Mapping[str, Any]
    shakedown_bytes: bytes


def load_frozen_authority() -> FrozenAuthority:
    return FrozenAuthority(
        strict_json_file(AUTHORITIES_PATH),
        strict_json_file(CASE_AUTHORITIES_PATH),
        strict_json_file(SYNTHETIC_PREFLIGHT_PATH),
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
        contract = require_orbit_mc_contract(value)
        if frozen is not None:
            if semantic_sha256(value) != frozen.authorities["protocol_semantic_sha256"]:
                raise ValueError("protocol semantic authority differs")
            expected_case_authorities = build_all_case_authorities(value, plan)
            if (
                frozen.case_authorities != expected_case_authorities
                or semantic_sha256(frozen.case_authorities)
                != frozen.authorities["case_authorities_sha256"]
            ):
                raise ValueError("case authorities differ from preregistration")
            if (
                not frozen.synthetic["passed"]
                or frozen.synthetic["p2_field_access_count"] != 0
                or frozen.synthetic["orbit_outcome_access_count"] != 0
            ):
                raise ValueError("synthetic production preflight is invalid")
            if frozen.authorities["orbit_mc_source_sha256"] != contract["source_sha256"]:
                raise ValueError("orbit_mc source differs from preregistered authority")
            if (
                hashlib.sha256(frozen.shakedown_bytes).hexdigest()
                != frozen.authorities["shakedown_file_sha256"]
                or semantic_sha256(frozen.shakedown)
                != frozen.authorities["shakedown_semantic_sha256"]
            ):
                raise ValueError("shakedown record differs from preregistered authority")
            verify_shakedown_record(value, frozen.shakedown)
            case_authorities = frozen.case_authorities

            def manifest_bytes(authority: Mapping[str, Any]) -> tuple[bytes, bytes]:
                return (
                    (EXPERIMENT / authority["launch_manifest_path"]).read_bytes(),
                    (EXPERIMENT / authority["batch_manifest_path"]).read_bytes(),
                )

        else:
            case_authorities = build_all_case_authorities(value, plan)

            def manifest_bytes(authority: Mapping[str, Any]) -> tuple[bytes, bytes]:
                launches = build_case_launches(
                    value, plan, authority["role"], authority["timestep"]
                )
                campaign = authority["campaign_id"]
                return (
                    canonical_bytes(runtime_launch_payload(campaign, launches)),
                    canonical_bytes(runtime_batch_payload(plan, campaign, launches)),
                )

        cases: dict[tuple[str, str], dict[str, Any]] = {}
        for authority in case_authorities["cases"]:
            role = authority["role"]
            timestep = authority["timestep"]
            campaign = authority["campaign_id"]
            if campaign != case_id(plan, role, timestep) or authority["plan_kind"] != plan.kind:
                raise ValueError(f"{campaign} does not belong to the {plan.kind} plan")
            launches = build_case_launches(value, plan, role, timestep)
            batches = batch_records(plan, launches)
            actual_launch_bytes, actual_batch_bytes = manifest_bytes(authority)
            expected_launch_bytes = canonical_bytes(
                runtime_launch_payload(campaign, launches)
            )
            expected_batch_bytes = canonical_bytes(
                runtime_batch_payload(plan, campaign, launches)
            )
            if (
                actual_launch_bytes != expected_launch_bytes
                or hashlib.sha256(actual_launch_bytes).hexdigest()
                != authority["runtime_launch_payload_byte_sha256"]
                or load_runtime_launch_payload(actual_launch_bytes, campaign)
                != tuple(sorted(launches, key=lambda item: item.launch_id))
            ):
                raise ValueError(f"{campaign} launch-byte authority differs")
            if (
                actual_batch_bytes != expected_batch_bytes
                or hashlib.sha256(actual_batch_bytes).hexdigest()
                != authority["runtime_batch_payload_byte_sha256"]
                or content_hash(launch_records(launches))
                != authority["orbit_launches_sha256"]
                or content_hash(
                    {
                        "estimator_policy": EstimatorPolicy.UNWEIGHTED_BINOMIAL.value,
                        "batches": batches,
                    }
                )
                != authority["batch_manifest_sha256"]
                or estimator_identity(launches, batches)
                != authority["estimator_sha256"]
            ):
                raise ValueError(f"{campaign} batch/estimator authority differs")
            context.write_blob(
                f"artifacts/{authority['launch_manifest_path']}",
                actual_launch_bytes,
            )
            context.write_blob(
                f"artifacts/{authority['batch_manifest_path']}",
                actual_batch_bytes,
            )
            cases[(role, timestep)] = {
                "authority": authority,
                "launches": launches,
                "batches": batches,
            }
        input_authority = validate_frozen_input_authority(value)
        context.write_json("artifacts/protocol.json", value)
        context.write_json("artifacts/case-authorities.json", case_authorities)
        context.write_json("artifacts/campaign-plan.json", plan_record(plan))
        context.write_json("artifacts/orbit-mc-contract.json", contract)
        if frozen is not None:
            context.write_json("artifacts/authorities.json", frozen.authorities)
            context.write_json("artifacts/synthetic-preflight.json", frozen.synthetic)
            context.write_blob("artifacts/shakedown.json", frozen.shakedown_bytes)
        else:
            context.write_json(
                "artifacts/shakedown-disclosure.json",
                {
                    "evidentiary": False,
                    "outcomes_enter_estimand": False,
                    "statement": value["shakedown"]["purpose"],
                    "disjointness": shakedown_disjointness(value),
                },
            )
        context.write_json("artifacts/p2-input-authority.json", input_authority)
        context.write_json(
            "artifacts/runtime.json",
            {
                "generated_at_utc": datetime.now(timezone.utc),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "cpu_count": os.cpu_count(),
                "worker_pool_size": worker_count(value),
            },
        )
        state["cases"] = cases
        state["case_authorities"] = case_authorities
        collector["prebundle"] = {
            "case_count": len(cases),
            "orbit_mc_contract": contract,
        }
        return {
            "preregistered": frozen is not None,
            "plan_kind": plan.kind,
            "case_count": len(cases),
            "total_case_launches": sum(
                len(item["launches"]) for item in cases.values()
            ),
            "p2_authority": input_authority,
            "orbit_mc_source_sha256": contract["source_sha256"],
        }

    def development(context: Any) -> Decision:
        started = time.perf_counter()
        manufactured = manufactured_gate_report(value)
        context.write_json("artifacts/manufactured-gates.json", manufactured)
        fields: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        serialized: dict[str, Any] = {}
        adapter_seconds: dict[str, float] = {}
        for role in ROLES:
            context.before_expensive(
                f"p2-adapter-{role}",
                kind="solver",
                details={
                    "role": role,
                    "design": "divergent-exit-stack",
                    "plan_kind": plan.kind,
                },
            )
            tick = time.perf_counter()
            fields[role], evidence[role], serialized[role] = build_regular_field(
                REPOSITORY, value, role
            )
            adapter_seconds[role] = time.perf_counter() - tick
            context.write_json(f"artifacts/fields/{role}.json", serialized[role])
            context.write_json(
                f"artifacts/field-evidence/{role}.json", evidence[role]
            )
        comparisons = {
            "primary_to_refined": compare_maps(fields["primary"], fields["refined"]),
            "refined_to_enlarged": compare_maps(fields["refined"], fields["enlarged"]),
        }
        for item in comparisons.values():
            item["b_relative_rms"] = item["b_rms_t"] / max(
                fields["primary"].max_b_t,
                fields["refined"].max_b_t,
                fields["enlarged"].max_b_t,
                np.finfo(float).tiny,
            )
        context.write_json("artifacts/field-map-convergence.json", comparisons)
        map_convergence_passed = all(
            item["b_relative_rms"]
            <= value["field_adapter"]["maximum_cross_map_b_relative_rms"]
            for item in comparisons.values()
        )
        accepted = bool(
            manufactured["passed"]
            and all(item["passed"] for item in evidence.values())
            and map_convergence_passed
        )
        state.update(
            {
                "manufactured": manufactured,
                "fields": fields,
                "field_evidence": evidence,
                "serialized_fields": serialized,
                "map_comparisons": comparisons,
            }
        )
        collector["development"] = {
            "manufactured_checks": manufactured["checks"],
            "field_evidence": evidence,
            "map_comparisons": comparisons,
            "adapter_seconds": adapter_seconds,
            "seconds": time.perf_counter() - started,
            "accepted": accepted,
        }
        return Decision(
            accepted,
            {
                "manufactured_passed": manufactured["passed"],
                "field_adapter_passed": all(item["passed"] for item in evidence.values()),
                "map_adapter_convergence_passed": map_convergence_passed,
            },
        )

    def assessment(context: Any) -> Decision:
        started = time.perf_counter()
        workers = worker_count(value)
        main_ledger = ValidatorLedger()
        tasks: list[dict[str, Any]] = []
        for role in ROLES:
            field = state["fields"][role]
            for timestep in TIMESTEPS:
                frozen_case = state["cases"][(role, timestep)]
                authority = frozen_case["authority"]
                launches = frozen_case["launches"]
                batches = frozen_case["batches"]
                campaign_id = authority["campaign_id"]
                config = orbit_config(value, role, timestep)
                config_sha = content_hash(asdict(config))
                policy_sha = policy_identity(value, plan, role, timestep)
                if (
                    state["field_evidence"][role]["source_identity_sha256"]
                    != authority["field_identity_sha256"]
                    or config_sha != authority["config_identity_sha256"]
                    or policy_sha != authority["policy_identity_sha256"]
                    or estimator_identity(launches, batches)
                    != authority["estimator_sha256"]
                ):
                    raise ValueError(f"{campaign_id} execution authority differs")
                key = case_key(role, timestep)
                tasks.append(
                    {
                        "case_key": key,
                        "role": role,
                        "timestep": timestep,
                        "campaign_id": campaign_id,
                        "authority": authority,
                        "launches": launches,
                        "batches": batches,
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
                        "field_evidence": state["field_evidence"][role],
                    }
                )
        # Every label access is recorded, in case order, before any orbit runs.
        for task in tasks:
            context.before_expensive(
                f"orbit-{task['case_key']}",
                kind="label",
                details={
                    "campaign_id": task["campaign_id"],
                    "launch_count": len(task["launches"]),
                    "sequential_batches_within_case": True,
                    "parallel_cases": workers > 1,
                    "worker_pool_size": workers,
                    "plan_kind": plan.kind,
                },
            )
        stage_started = time.perf_counter()
        integrated = run_stage(tasks, run_case_integration, workers)
        integration_wall = time.perf_counter() - stage_started
        campaigns: dict[tuple[str, str], dict[str, Any]] = {}
        for task, outcome in zip(tasks, integrated, strict=True):
            if outcome["case_key"] != task["case_key"]:
                raise RuntimeError("case results returned out of order")
            key = task["case_key"]
            role, timestep = task["role"], task["timestep"]
            for record in outcome["checkpoints"]:
                context.write_blob(
                    f"artifacts/checkpoints/{record['artifact_name']}",
                    gzip.compress(Path(record["path"]).read_bytes(), mtime=0),
                )
            summary = outcome["summary"]
            context.write_json(
                f"artifacts/summaries/{key}.json",
                {
                    "summary": summary.to_dict(),
                    "strata": outcome["strata"],
                    "preflight": outcome["preflight"],
                    "partial_checkpoint_file_sha256": outcome["partial_checkpoint_file_sha256"],
                    "final_checkpoint_file_sha256": outcome["final_checkpoint_file_sha256"],
                    "sequential_batch_checkpoint_file_sha256": [
                        record["file_sha256"]
                        for record in outcome["checkpoints"]
                        if record["stage"] == "batch"
                    ],
                    "diagnostics": outcome["diagnostics"],
                    "timing_s": outcome["timing_s"],
                    "worker_process_id": outcome["process_id"],
                },
            )
            # Main-process determinism sample: re-integrate the first two
            # launches of every case with the main-process field object and
            # compare canonical result records with the worker's results.
            ordered = sorted(task["launches"], key=lambda item: item.launch_id)
            worker_by_id = {item.launch_id: item for item in outcome["results"]}

            def determinism_sample(
                sample_launches: Sequence[Any] = ordered[:2],
                sample_field: Any = task["field"],
                sample_config: Any = task["config"],
                sample_results: Mapping[str, Any] = worker_by_id,
            ) -> dict[str, Any]:
                compared = 0
                for launch in sample_launches:
                    local = integrate_orbit(launch, sample_field, sample_config)
                    if content_hash(result_record(local)) != content_hash(
                        result_record(sample_results[launch.launch_id])
                    ):
                        raise RuntimeError(
                            f"cross-process determinism differs for {launch.launch_id}"
                        )
                    compared += 1
                return {"compared": compared}

            sample = main_ledger.run(key, "cross_process_determinism_sample", determinism_sample)
            campaigns[(role, timestep)] = campaign = {
                "case_key": key,
                "task": task,
                "authority": task["authority"],
                "launches": task["launches"],
                "batches": task["batches"],
                "field": task["field"],
                "field_sha": task["field_sha"],
                "config": task["config"],
                "config_sha": task["config_sha"],
                "policy_sha": task["policy_sha"],
                "results": outcome["results"],
                "summary": summary,
                "strata": outcome["strata"],
                "preflight": outcome["preflight"],
                "diagnostics": outcome["diagnostics"],
                "timing_s": dict(outcome["timing_s"]),
                "validators": list(outcome["validators"]),
                "determinism_sample": sample,
                "worker_process_id": outcome["process_id"],
            }
            # Record now so a later failure still leaves per-case diagnostics.
            collector["cases"][key] = _collect_case(campaign, export_ran=False)
            collector["validators"] = list(main_ledger.records) + [
                item for entry in campaigns.values() for item in entry["validators"]
            ]
        summaries = {key: item["summary"] for key, item in campaigns.items()}
        convergence = _convergence(summaries, value)
        context.write_json("artifacts/probability-convergence.json", convergence)
        gates = _result_gate_report(
            campaigns,
            convergence,
            state["manufactured"],
            state["field_evidence"],
            state["map_comparisons"],
            value,
            plan,
        )
        parity_passed = bool(
            state["manufactured"]["checks"]["cpu_parity"]
            and state["manufactured"]["checks"]["cuda_parity"]
        )
        if plan.binding_gates:
            replay_condition = bool(gates["passed_before_replay"])
            convergence_flags = {
                "timestep_passed": bool(convergence["timestep_passed"]),
                "cross_map_passed": bool(convergence["cross_map_passed"]),
                "backend_parity_passed": parity_passed,
            }
            flag_basis = "campaign probability convergence gates (binding)"
        else:
            convergence_flags = {
                "timestep_passed": bool(
                    gates["checks"]["campaign_preflight"]
                    and gates["checks"]["runtime_rotation"]
                    and gates["checks"]["zero_incomplete_or_numerical_failures"]
                ),
                "cross_map_passed": bool(
                    gates["checks"]["field_adapter"]
                    and gates["checks"]["field_map_convergence"]
                ),
                "backend_parity_passed": parity_passed,
            }
            replay_condition = all(convergence_flags.values())
            flag_basis = (
                "shakedown-scale structural checks: preflight+rotation bound+zero "
                "failures across N/2N/4N, adapter+cross-map field convergence, "
                "backend parity; probability/energy gates are informational"
            )
        gates["replay_condition"] = replay_condition
        gates["artifact_convergence_flags"] = convergence_flags
        gates["artifact_convergence_flag_basis"] = flag_basis
        replay_count = 0
        handoff = None
        export_wall = 0.0
        exports: dict[str, dict[str, Any]] = {}
        if replay_condition:
            export_tasks = []
            for (role, timestep), campaign in campaigns.items():
                task = dict(campaign["task"])
                task["results"] = campaign["results"]
                task["summary"] = campaign["summary"]
                task["convergence_evidence"] = convergence_flags
                task["preregistration"] = {
                    "protocol_id": (
                        value["schema_version"]
                        if plan.binding_gates
                        else f"{schema('shakedown')}:NON-EVIDENTIARY"
                    ),
                    "frozen_before_outcomes": True,
                    "held_out_geometry_status": "passed" if plan.binding_gates else "pending",
                }
                task["export_handoff"] = role == "refined" and timestep == "4N"
                export_tasks.append(task)
            context.before_expensive(
                "exact-authority-replay-all-cases",
                kind="backend",
                details={
                    "case_count": len(export_tasks),
                    "worker_pool_size": workers,
                    "plan_kind": plan.kind,
                },
            )
            stage_started = time.perf_counter()
            exported = run_stage(export_tasks, run_case_export, workers)
            export_wall = time.perf_counter() - stage_started
            for task, outcome in zip(export_tasks, exported, strict=True):
                if outcome["case_key"] != task["case_key"]:
                    raise RuntimeError("export results returned out of order")
                key = task["case_key"]
                replay_count += 1
                context.write_blob(
                    f"artifacts/orbits/{key}.json.gz",
                    gzip.compress(Path(outcome["artifact_path"]).read_bytes(), mtime=0),
                )
                context.write_blob(
                    f"artifacts/orbits/{key}.json.sha256",
                    Path(outcome["artifact_sidecar_path"]).read_bytes(),
                )
                if outcome["handoff"] is not None:
                    handoff = outcome["handoff"]
                exports[key] = outcome
                campaign = campaigns[(task["role"], task["timestep"])]
                campaign["validators"].extend(outcome["validators"])
                campaign["timing_s"].update(outcome["timing_s"])
                campaign["artifact_file_sha256"] = outcome["artifact_file_sha256"]
        gates["exact_authority_replay_count"] = replay_count
        gates["exact_authority_replay"] = replay_count == len(campaigns)
        gates["passed"] = bool(gates["passed_before_replay"] and gates["exact_authority_replay"])
        context.write_json("artifacts/gates.json", _plain(gates))
        if handoff is not None:
            context.write_json("artifacts/coupling-export-only.json", handoff)
        totals = {
            campaign["case_key"]: {
                "trial_count": campaign["summary"].trial_count,
                "termination_counts": dict(campaign["summary"].termination_counts),
                "wall_hit": asdict(campaign["summary"].wall_hit),
                "reflected": asdict(campaign["summary"].reflected),
                "escaped": asdict(campaign["summary"].escaped),
                "incomplete": asdict(campaign["summary"].incomplete),
            }
            for campaign in campaigns.values()
        }
        all_validators = list(main_ledger.records)
        for campaign in campaigns.values():
            all_validators.extend(campaign["validators"])
        validator_failures = [item for item in all_validators if not item["passed"]]
        if plan.binding_gates:
            accepted = bool(gates["passed"])
            status = "accepted" if accepted else "rejected"
        else:
            accepted = bool(
                replay_condition
                and gates["exact_authority_replay"]
                and not validator_failures
            )
            status = "shakedown_passed" if accepted else "shakedown_failed"
        terminal = {
            "status": status,
            "plan_kind": plan.kind,
            "evidentiary": plan.binding_gates,
            "gates_binding": plan.binding_gates,
            "classification": value["classification"],
            "launches_per_case": plan.launches_per_case,
            "total_case_launch_count": sum(
                len(item["launches"]) for item in campaigns.values()
            ),
            "orbit_count": sum(item["summary"].trial_count for item in campaigns.values()),
            "campaign_count": len(campaigns),
            "campaigns": totals,
            "gates": _plain(gates),
            "validators": {
                "passed": sum(item["passed"] for item in all_validators),
                "failed": len(validator_failures),
            },
            "execution_mode": {
                "parallel_cases": workers > 1,
                "worker_pool_size": workers,
                "integration_wall_s": integration_wall,
                "export_wall_s": export_wall,
                "assessment_wall_s": time.perf_counter() - started,
            },
            "coupling": (
                "export_only_pending_consumer_integration"
                if handoff is not None and plan.binding_gates
                else "shakedown_export_constructed_not_published"
                if handoff is not None
                else "not_exported_failed_gates"
            ),
            "limitations": value["publication_boundary"],
        }
        context.write_json("artifacts/campaign-result.json", terminal)
        collector["assessment"] = {
            "gates": _plain(gates),
            "convergence": convergence,
            "execution_mode": terminal["execution_mode"],
            "status": status,
            "accepted": accepted,
        }
        for campaign in campaigns.values():
            collector["cases"][campaign["case_key"]] = _collect_case(
                campaign, export_ran=campaign["case_key"] in exports
            )
        collector["validators"] = all_validators
        return Decision(accepted, _plain(terminal))

    return RuntimeCallbacks(prebundle, development, assessment)


def _collect_case(campaign: Mapping[str, Any], *, export_ran: bool) -> dict[str, Any]:
    validators = campaign["validators"]
    return {
        "campaign_id": campaign["authority"]["campaign_id"],
        "launch_count": len(campaign["launches"]),
        "preflight": campaign["preflight"],
        "diagnostics": campaign["diagnostics"],
        "timing_s": dict(campaign["timing_s"]),
        "validators": {
            "passed": sum(item["passed"] for item in validators),
            "failed": sum(not item["passed"] for item in validators),
            "failures": [item for item in validators if not item["passed"]],
            "names": [item["validator"] for item in validators],
        },
        "export_stage_ran": export_ran,
        "artifact_file_sha256": campaign.get("artifact_file_sha256"),
        "determinism_sample": campaign["determinism_sample"],
        "worker_process_id": campaign["worker_process_id"],
        "summary": campaign["summary"].to_dict(),
    }


def callbacks() -> RuntimeCallbacks:
    """Production callbacks bound to the frozen preregistration files."""

    value = protocol()
    return build_callbacks(
        value, evidentiary_plan(value), frozen=load_frozen_authority(), collector={}
    )
