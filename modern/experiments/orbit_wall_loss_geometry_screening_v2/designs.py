"""Design binding for v2: the 96 sweep-v2 designs (v1 pipeline, reused by import) + the P2 row.

Sweep-v2 designs are rebuilt, re-solved and identity-proven exactly as in
``experiments.orbit_wall_loss_geometry_screening_v1.designs`` (imported, not forked): the
rebuilt geometry/source/config/case hashes must equal the sealed sweep record, the re-solved
QoIs must replay within the sweep tolerances, the representatives must reproduce their
stored maps node-wise. v2 differs from v1 in coverage only: the 2x refined re-solve and the
cross-resolution diagnostic run for EVERY design (v1 ran them for the four representatives).

The P2 divergent-exit design (the v4 campaign's NUMERICAL_P2_QUALIFIED field) enters as a
97th, separately labelled row through the hash-bound v4 adapter
(``experiments.cft_orbit_wall_loss_v4.adapter.build_regular_field``): level-1 as the
accepted map, level-2 as the refined map. Its launch design is this campaign's (catalogue
cells, scrambled Sobol), so its row is a screening launch design on a qualified field, not a
replication of v4's evidence.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.orbit_mc import OrbitConfig, compare_maps

from experiments.cft_orbit_wall_loss_v4 import adapter as v4_adapter
from experiments.orbit_wall_loss_geometry_screening_v1 import designs as v1_designs
from experiments.orbit_wall_loss_geometry_screening_v1.designs import (  # re-exported, reused by import
    DesignGeometry,
    SweepBinding,
    case_index,
    design_geometry,
    field_identity,
    field_pipeline_source_files,
    field_pipeline_source_sha256,
    load_sweep_binding,
    rebuild_case,
    resolve_design,
)

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
V4_PROTOCOL_PATH = MODERN / "experiments" / "cft_orbit_wall_loss_v4" / "protocol.json"

SET_SWEEP = "sweep_v2"
SET_P2 = "p2_divergent_exit"
P2_DESIGN_ID = "divergent-exit-stack"
LABEL_SWEEP = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
LABEL_P2 = "P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN"
ELECTRON_MASS_KG = v1_designs.ELECTRON_MASS_KG
EV_J = v1_designs.EV_J

__all__ = [
    "DesignGeometry", "SweepBinding", "case_index", "design_geometry", "field_identity",
    "field_pipeline_source_files", "field_pipeline_source_sha256", "load_sweep_binding", "rebuild_case",
    "resolve_design", "BoundDesign", "bind_sweep_design", "bind_p2_design", "orbit_config_for_design",
    "resolve_p2_design", "v4_protocol", "SET_SWEEP", "SET_P2", "P2_DESIGN_ID", "LABEL_SWEEP", "LABEL_P2",
]


@dataclass(frozen=True)
class BoundDesign:
    """One design's geometry and identities (no field solve)."""

    design_key: str
    set_id: str
    design_id: str
    label: str
    field_level: str
    representative: bool
    sweep_index: int | None
    design_values: dict[str, float] | None
    wall_radius_m: float
    chamber_length_m: float
    injector_length_m: float
    straight_z_min_m: float
    straight_z_max_m: float
    domain_z_min_m: float
    domain_z_max_m: float
    geometry: dict[str, Any]
    identities: dict[str, str]
    accepted_field_identity: str
    refined_field_identity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bind_sweep_design(binding: SweepBinding, case_id: str, declaration: Mapping[str, Any], *, representative: bool) -> BoundDesign:
    case = rebuild_case(binding, case_id)
    geometry = design_geometry(case)
    return BoundDesign(
        design_key=case_id,
        set_id=SET_SWEEP,
        design_id=case.design.design_id,
        label=LABEL_SWEEP,
        field_level=declaration["field_status"],
        representative=representative,
        sweep_index=case_index(case_id),
        design_values=v1_designs.sweep.design_values(case.design),
        wall_radius_m=geometry.wall_radius_m,
        chamber_length_m=geometry.chamber_length_m,
        injector_length_m=geometry.injector_length_m,
        straight_z_min_m=0.0,
        straight_z_max_m=geometry.exit_start_m,
        domain_z_min_m=0.0,
        domain_z_max_m=geometry.chamber_length_m,
        geometry=geometry.to_dict(),
        identities={
            "geometry_sha256": case.geometry_sha256,
            "source_sha256": case.source_sha256,
            "config_sha256": case.config_sha256,
            "case_sha256": case.case_sha256,
        },
        accepted_field_identity=field_identity(case, declaration, "accepted"),
        refined_field_identity=field_identity(case, declaration, "refined"),
    )


def v4_protocol(declaration: Mapping[str, Any]) -> dict[str, Any]:
    data = V4_PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != declaration["v4_protocol_file_sha256"]:
        raise ValueError("v4 protocol bytes differ from the declared authority")
    value = strict_json_file(V4_PROTOCOL_PATH)
    if value["authority"]["design_id"] != P2_DESIGN_ID:
        raise ValueError("v4 protocol does not describe the divergent-exit-stack design")
    for role in ("primary", "refined"):
        for key in ("checkpoint_file_sha256", "sidecar_file_sha256", "mesh_sha256", "run_sha256"):
            if value["field_adapter"]["maps"][role][key] != declaration["maps"][role][key]:
                raise ValueError(f"v4 {role} map {key} differs from the declared authority")
    return value


def _p2_field_identity(v4: Mapping[str, Any], role: str) -> str:
    from cft_revival.orbit_mc.artifacts import content_hash

    declaration = v4["field_adapter"]["maps"][role]
    return content_hash(
        {
            "role": role,
            "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
            "sidecar_file_sha256": declaration["sidecar_file_sha256"],
            "mesh_sha256": declaration["mesh_sha256"],
            "run_sha256": declaration["run_sha256"],
        }
    )


def bind_p2_design(declaration: Mapping[str, Any]) -> BoundDesign:
    """The v4 divergent-exit design from its generator + the v4 wall/domain authority."""

    from cft_revival.geometry.generators import divergent_exit_stack

    v4 = v4_protocol(declaration)
    design = divergent_exit_stack()
    chamber = design.chamber
    wall = v4["orbit"]["wall"]
    domain = v4["orbit"]["domain"]
    if float(wall["radius_m"]) != float(chamber.outer_radius_m) or abs(float(wall["z_max_m"]) - float(chamber.exit_start_m)) > 1.0e-12:
        raise ValueError("v4 straight-wall authority differs from the divergent-exit-stack geometry")
    geometry = {
        "case_id": None,
        "design_id": P2_DESIGN_ID,
        "wall_radius_m": float(chamber.outer_radius_m),
        "chamber_length_m": float(chamber.length_m),
        "injector_length_m": float(chamber.injector_length_m),
        "exit_start_m": float(wall["z_max_m"]),
        "exit_length_m": float(chamber.exit_length_m),
        "exit_outer_radius_m": float(chamber.exit_outer_radius_m),
        "dielectric_thickness_m": float(chamber.dielectric_thickness_m),
        "stage_count": len(design.stages),
        "stage_pitch_m": float(design.stages[0].pitch_m),
        "stage_centers_m": [float(stage.center_z_m) for stage in design.stages],
        "has_divergent_exit": bool(chamber.exit_length_m > 0.0),
        "straight_wall_scope": wall["scope"],
        "domain_scope": domain["scope"],
        "regular_plasma_domain": dict(v4["field_adapter"]["regular_plasma_domain"]),
    }
    return BoundDesign(
        design_key=P2_DESIGN_ID,
        set_id=SET_P2,
        design_id=P2_DESIGN_ID,
        label=LABEL_P2,
        field_level=declaration["field_level"],
        representative=True,
        sweep_index=None,
        design_values=None,
        wall_radius_m=float(wall["radius_m"]),
        chamber_length_m=float(chamber.length_m),
        injector_length_m=float(chamber.injector_length_m),
        straight_z_min_m=float(wall["z_min_m"]),
        straight_z_max_m=float(wall["z_max_m"]),
        domain_z_min_m=float(domain["z_min_m"]),
        domain_z_max_m=float(domain["z_max_m"]),
        geometry=geometry,
        identities={
            "geometry_sha256": design.canonical_sha256,
            "v4_protocol_file_sha256": declaration["v4_protocol_file_sha256"],
            "primary_checkpoint_file_sha256": v4["field_adapter"]["maps"]["primary"]["checkpoint_file_sha256"],
            "refined_checkpoint_file_sha256": v4["field_adapter"]["maps"]["refined"]["checkpoint_file_sha256"],
        },
        accepted_field_identity=_p2_field_identity(v4, "primary"),
        refined_field_identity=_p2_field_identity(v4, "refined"),
    )


def orbit_config_for_design(design: BoundDesign, rule: Mapping[str, Any], timestep_policy: Mapping[str, Any]) -> OrbitConfig:
    """The v1 orbit rule (max_path = 2 L, max_time = 2 max_path / v(5 eV)) on the design's own wall/domain authority."""

    max_path = float(rule["max_path_channel_lengths"]) * design.chamber_length_m
    slowest = math.sqrt(2.0 * float(rule["slowest_energy_ev"]) * EV_J / ELECTRON_MASS_KG)
    max_time = float(rule["max_time_transit_factor"]) * max_path / slowest
    return OrbitConfig(
        wall_radius_m=design.wall_radius_m,
        wall_z_min_m=design.straight_z_min_m,
        wall_z_max_m=design.straight_z_max_m,
        domain_radius_m=design.wall_radius_m,
        domain_z_min_m=design.domain_z_min_m,
        domain_z_max_m=design.domain_z_max_m,
        max_time_s=max_time,
        max_path_m=max_path,
        max_steps=int(rule["max_steps"]),
        max_rotation_rad=float(timestep_policy["max_rotation_rad"]),
        event_tolerance_m=float(rule["event_tolerance_m"]),
        maximum_gamma=float(rule["maximum_gamma"]),
    )


def resolve_p2_design(declaration: Mapping[str, Any], adapter_gates: Mapping[str, Any]) -> dict[str, Any]:
    """Level-1 (accepted) and level-2 (refined) P2 maps through the v4 adapter + cross-map diagnostic."""

    v4 = v4_protocol(declaration)
    accepted_field, accepted_evidence, accepted_serialized = v4_adapter.build_regular_field(REPOSITORY, v4, "primary")
    refined_field, refined_evidence, _ = v4_adapter.build_regular_field(REPOSITORY, v4, "refined")
    cross = dict(compare_maps(accepted_field, refined_field))
    cross["b_relative_rms"] = float(cross["b_rms_t"]) / max(accepted_field.max_b_t, refined_field.max_b_t, float(np.finfo(float).tiny))
    checks = {
        "identity_proven": True,
        "adapter_primary_passed": bool(accepted_evidence["passed"]),
        "adapter_refined_passed": bool(refined_evidence["passed"]),
        "interpolation_b_relative_rms": accepted_evidence["field_error_report"]["b_relative_rms"] <= float(adapter_gates["maximum_b_relative_rms"]),
        "interpolation_b_component_abs": max(
            accepted_evidence["field_error_report"]["br_max_abs_t"], accepted_evidence["field_error_report"]["bz_max_abs_t"]
        )
        <= float(adapter_gates["maximum_b_component_absolute_error_t"]),
        "cross_resolution_b_relative_rms": cross["b_relative_rms"] <= float(adapter_gates["maximum_cross_resolution_b_relative_rms"]),
        "certificate_preflight": bool(accepted_field.certificate_tightness.preflight_passed),
    }
    grid = accepted_evidence["regular_grid"]
    bore = {
        "radial_samples": grid["radial_samples"],
        "axial_samples": grid["axial_samples"],
        "r_max_m": grid["r_max_m"],
        "z_min_m": grid["z_min_m"],
        "z_max_m": grid["z_max_m"],
        "dr_m": (grid["r_max_m"] - grid["r_min_m"]) / (grid["radial_samples"] - 1),
        "dz_m": (grid["z_max_m"] - grid["z_min_m"]) / (grid["axial_samples"] - 1),
        "wall_radius_m": grid["r_max_m"],
        "radial_cells_across_bore": float(grid["radial_samples"] - 1),
    }
    evidence = {
        "case_id": None,
        "design_id": P2_DESIGN_ID,
        "set_id": SET_P2,
        "field_level": declaration["field_level"],
        "v4_protocol_file_sha256": declaration["v4_protocol_file_sha256"],
        "adapter": {"primary": accepted_evidence, "refined": refined_evidence},
        "accepted_bore_field": {
            "role": "accepted",
            "source_identity_sha256": accepted_evidence["source_identity_sha256"],
            "bore_grid": bore,
            "interpolation_error_report": accepted_evidence["field_error_report"],
            "certificate": accepted_evidence["certificate"],
            "max_b_t": float(accepted_field.max_b_t),
            "material_map_sha256": accepted_evidence["material_map_sha256"],
        },
        "refined_bore_field": {
            "role": "refined",
            "source_identity_sha256": refined_evidence["source_identity_sha256"],
            "interpolation_error_report": refined_evidence["field_error_report"],
            "max_b_t": float(refined_field.max_b_t),
        },
        "cross_resolution": cross,
        "cross_resolution_note": "level-1 vs level-2 FEM checkpoints sampled on the regular grid (v4's primary_to_refined comparison), not a re-solve of one discrete problem",
        "checks": checks,
        "passed": all(checks.values()),
    }
    return {"field": accepted_field, "serialized": accepted_serialized, "evidence": evidence}
