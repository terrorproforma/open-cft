"""Design list, identity proof and sealed L1a references of the L1b HEMP confirmation.

The design set is the sealed L1a geometry sweep v3 catalogue's HEMP-like set: every
``sobol_v3`` entry with ``hemp_like_all_cusps == true`` (15 of 128; the rule is the
catalogue's ``hemp_like_rule``). Each design is rebuilt from the preregistered Sobol design
(seed, index) with the sweep-v3 builder; its geometry / source / config / case hashes must
equal the sealed sweep-v3 design authorities (identity proof), and its sealed sweep-v3
design record (byte hash bound to the sweep-v3 results manifest) is the L1a reference the
material-aware map is compared with.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.l1a_geometry_sweep_v3 import catalogue as v3_catalogue
from experiments.l1a_geometry_sweep_v3 import designs as v3_designs
from experiments.l1a_geometry_sweep_v3 import experiment as v3_experiment

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
V3_EXPERIMENT = MODERN / "experiments" / "l1a_geometry_sweep_v3"
V3_RESULTS = V3_EXPERIMENT / "results"
V3_MANIFEST_PATH = V3_RESULTS / "manifest.json"
V3_PROTOCOL_PATH = V3_EXPERIMENT / "protocol.json"
V3_DESIGN_AUTHORITIES_PATH = V3_EXPERIMENT / "design-authorities.json"

SET_HEMP = "hemp_like_v3"
DESIGN_SETS = (SET_HEMP,)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DesignSpec:
    set_id: str
    design_id: str
    ordinal: int
    representative: bool

    @property
    def key(self) -> str:
        return f"{self.set_id}:{self.design_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Sealed sweep-v3 sources
# --------------------------------------------------------------------------


_V3_MANIFEST: dict[str, Any] | None = None
_V3_PROTOCOL: dict[str, Any] | None = None
_V3_CATALOGUE: dict[str, Any] | None = None
_V3_AUTHORITIES: dict[str, Any] | None = None


def v3_manifest() -> dict[str, Any]:
    global _V3_MANIFEST
    if _V3_MANIFEST is None:
        value = strict_json_file(V3_MANIFEST_PATH)
        if value["state"] != "accepted_result" or value["experiment_id"] != "l1a-geometry-sweep-v3":
            raise ValueError("the sweep-v3 results manifest is not an accepted l1a-geometry-sweep-v3 bundle")
        _V3_MANIFEST = value
    return _V3_MANIFEST


def v3_manifest_entry(relative_path: str) -> dict[str, Any]:
    for entry in v3_manifest()["artifacts"]:
        if entry["path"] == relative_path and entry["type"] == "file":
            return entry
    raise ValueError(f"{relative_path} is not an artifact of the sweep-v3 bundle")


def sealed_v3_json(relative_path: str) -> dict[str, Any]:
    """Load a sweep-v3 artifact after checking its byte hash against the sweep-v3 manifest."""

    entry = v3_manifest_entry(relative_path)
    path = V3_RESULTS / relative_path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["byte_sha256"] or len(raw) != int(entry["bytes"]):
        raise ValueError(f"{relative_path}: bytes differ from the sweep-v3 manifest")
    from cft_revival.experiment_runtime.canonical import strict_json_loads

    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{relative_path}: sealed artifact is not an object")
    return value


def v3_protocol() -> dict[str, Any]:
    global _V3_PROTOCOL
    if _V3_PROTOCOL is None:
        value = v3_experiment.protocol()
        sealed = sealed_v3_json("artifacts/protocol.json")
        if sealed != value:
            raise ValueError("the checked-out sweep-v3 protocol differs from the sealed bundle copy")
        _V3_PROTOCOL = value
    return _V3_PROTOCOL


def v3_catalogue_sealed() -> dict[str, Any]:
    global _V3_CATALOGUE
    if _V3_CATALOGUE is None:
        value = sealed_v3_json(v3_catalogue.CATALOGUE_RELATIVE_PATH)
        if value["hemp_like_design_count"] != sum(1 for entry in value["entries"] if entry["hemp_like_all_cusps"]):
            raise ValueError("sweep-v3 catalogue HEMP-like count differs from its entries")
        _V3_CATALOGUE = value
    return _V3_CATALOGUE


def v3_design_authorities() -> dict[str, Any]:
    global _V3_AUTHORITIES
    if _V3_AUTHORITIES is None:
        value = strict_json_file(V3_DESIGN_AUTHORITIES_PATH)
        sealed = sealed_v3_json("artifacts/design-authorities.json")
        if sealed != value:
            raise ValueError("the checked-out sweep-v3 design authorities differ from the sealed bundle copy")
        _V3_AUTHORITIES = value
    return _V3_AUTHORITIES


def hemp_like_catalogue_entries() -> list[dict[str, Any]]:
    """The sealed HEMP-like entries (sobol_v3, hemp_like_all_cusps) in catalogue order."""

    return [entry for entry in v3_catalogue_sealed()["entries"] if entry["set_id"] == v3_designs.SET_SOBOL and entry["hemp_like_all_cusps"]]


def sealed_source_binding() -> dict[str, Any]:
    manifest = v3_manifest()
    return {
        "l1a_geometry_sweep_v3": {
            "manifest_file_sha256": _file_sha256(V3_MANIFEST_PATH),
            "terminal_byte_sha256": manifest["terminal_byte_sha256"],
            "lock_byte_sha256": manifest["lock_byte_sha256"],
            "catalogue_byte_sha256": v3_manifest_entry(v3_catalogue.CATALOGUE_RELATIVE_PATH)["byte_sha256"],
            "protocol_semantic_sha256": v3_catalogue_sealed()["protocol_semantic_sha256"],
            "design_authorities_file_sha256": _file_sha256(V3_DESIGN_AUTHORITIES_PATH),
            "preregistration_commit": strict_json_file(V3_RESULTS / "execution-lock.json")["commit"],
            "hemp_like_design_count": int(v3_catalogue_sealed()["hemp_like_design_count"]),
        }
    }


# --------------------------------------------------------------------------
# Design specifications
# --------------------------------------------------------------------------


def design_specs(value: Mapping[str, Any]) -> tuple[DesignSpec, ...]:
    """The declared design ids, checked against the sealed catalogue rule (order = catalogue order)."""

    declaration = value["design_sets"][SET_HEMP]
    if not declaration["included"]:
        raise ValueError("the HEMP-like design set must be included")
    entries = hemp_like_catalogue_entries()
    catalogue_ids = [entry["design_id"] for entry in entries]
    declared = list(declaration["design_ids"])
    if declared != catalogue_ids:
        raise ValueError("declared design ids differ from the sealed catalogue's HEMP-like set (order-sensitive)")
    if len(declared) != int(declaration["design_count"]):
        raise ValueError("declared design count differs from the design id list")
    representatives = set(declaration["representative_ids"])
    if not representatives <= set(declared):
        raise ValueError("representative ids must be declared designs")
    return tuple(DesignSpec(SET_HEMP, design_id, index, design_id in representatives) for index, design_id in enumerate(declared))


def v3_spec_for(design_id: str) -> v3_designs.DesignSpec:
    protocol = v3_protocol()
    for spec in v3_designs.design_specs(protocol):
        if spec.set_id == v3_designs.SET_SOBOL and spec.design_id == design_id:
            return spec
    raise ValueError(f"{design_id} is not a sweep-v3 Sobol design")


def rebuild_case(design_id: str) -> sweep.BuiltCase:
    """Rebuild the sweep-v3 case from the preregistered Sobol design and prove its identity."""

    protocol = v3_protocol()
    spec = v3_spec_for(design_id)
    case = v3_designs.sobol_case(spec, protocol)
    authority = next(item for item in v3_design_authorities()["designs"] if item["set_id"] == v3_designs.SET_SOBOL and item["design_id"] == design_id)
    checks = {
        "geometry_sha256": case.geometry_sha256 == authority["geometry_sha256"],
        "source_sha256": case.source_sha256 == authority["source_sha256"],
        "config_sha256": case.config_sha256 == authority["config_sha256"],
        "case_sha256": case.case_sha256 == authority["case_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"{design_id}: rebuilt case differs from the sealed sweep-v3 design authority: {checks}")
    return case


def l1a_reference(design_id: str) -> dict[str, Any]:
    """The sealed sweep-v3 design record (byte-bound) reduced to what the comparison needs."""

    entry = next(item for item in hemp_like_catalogue_entries() if item["design_id"] == design_id)
    record = sealed_v3_json(entry["record_path"])
    if record["design_id"] != design_id or record["status"] != "resolved" or record["set_id"] != v3_designs.SET_SOBOL:
        raise ValueError(f"{design_id}: sealed sweep-v3 record does not describe this design")
    if not record["descriptors"]["accepted"]["hemp_like_all_cusps"]:
        raise ValueError(f"{design_id}: sealed sweep-v3 record is not HEMP-like")
    accepted = record["accepted"]
    descriptors = record["descriptors"]["accepted"]
    return {
        "record_path": entry["record_path"],
        "record_byte_sha256": v3_manifest_entry(entry["record_path"])["byte_sha256"],
        "topology_payload_sha256": record["topology_payload_sha256"],
        "identity": dict(record["identity"]),
        "geometry": dict(record["geometry"]),
        "axis_window_m": [float(record["axis_window_m"][0]), float(record["axis_window_m"][1])],
        "source_strength_scale": float(record["evidence"]["design_values"]["source_strength_scale"]),
        "design_values": dict(record["evidence"]["design_values"]),
        "stage_count": int(record["evidence"]["derived_geometry"]["stage_count"]),
        "grid": {key: accepted["grid"][key] for key in ("radial_samples", "axial_samples", "dr_m", "dz_m", "radial_cells_across_bore", "max_b_t")},
        "axis_nulls": [{"null_id": null["null_id"], "z_m": null["z_m"], "zone": null["zone"], "classification": null["classification"]} for null in accepted["axis_nulls"]["nulls"]],
        "axis_null_count": int(accepted["axis_nulls"]["count"]),
        "wall_cusps": [
            {key: cusp[key] for key in ("cusp_id", "null_id", "axis_null_z_m", "z_c_m", "wall_b_t", "wall_b_r_t", "wall_b_z_t", "angle_to_wall_normal_deg", "boundary_ambiguous")}
            for cusp in accepted["topology"]["wall_cusps"]
        ],
        "outside_intersections": [{key: row[key] for key in ("cusp_id", "z_c_m", "zone", "wall_b_t")} for row in accepted["topology"]["outside_intersections"]],
        "wall_cusp_count": int(accepted["topology"]["wall_cusp_count"]),
        "cell_count": int(accepted["topology"]["cell_count"]),
        "cells": [{key: cell[key] for key in ("cell_id", "kind", "z_start_m", "z_end_m", "wall_b_min_t", "wall_mirror_ratio", "axis_bz_peak_t", "axis_mirror_ratio")} for cell in accepted["topology"]["cells"]],
        "rho": [
            {key: row[key] for key in ("cusp_id", "z_c_m", "wall_b_t", "upstream_axis_peak_t", "downstream_axis_peak_t", "upstream_wall_max_b_t", "downstream_wall_max_b_t", "rho_conservative", "rho_downstream", "rho_upstream", "rho_wall", "hemp_like_conservative")}
            for row in descriptors["cusps"]
        ],
        "min_rho_conservative": descriptors["min_rho_conservative"],
        "hemp_like_all_cusps": bool(descriptors["hemp_like_all_cusps"]),
        "x_w": float(descriptors["x_w"]),
        "wall_radius_over_pitch": float(descriptors["wall_radius_over_pitch"]),
        "ppm_prediction": dict(descriptors["ppm_prediction"]),
        "wall_harmonics": {key: descriptors["wall_harmonics"].get(key) for key in ("applies", "b3_over_b1", "b5_over_b1", "wall_b_r_max_abs_t")},
        "profiles": dict(descriptors["profiles"]) if descriptors.get("profiles") else None,
        "stability": {key: record["stability"][key] for key in ("stable", "max_axis_null_shift_m", "max_wall_intersection_shift_m")},
        "qois": {key: record["qois"][key] for key in ("centreline_abs_bz_peak_t", "field_peak_t", "boundary_to_peak_ratio", "relative_residual_l2")},
    }


def design_identity_without_solving(spec: DesignSpec) -> dict[str, Any]:
    case = rebuild_case(spec.design_id)
    reference = l1a_reference(spec.design_id)
    return {
        "set_id": spec.set_id,
        "design_id": spec.design_id,
        "ordinal": spec.ordinal,
        "representative": spec.representative,
        "sampling_design_id": case.design.design_id,
        "sampling_provenance": case.design.provenance,
        "case_sha256": case.case_sha256,
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "l1a_record_path": reference["record_path"],
        "l1a_record_byte_sha256": reference["record_byte_sha256"],
        "l1a_topology_payload_sha256": reference["topology_payload_sha256"],
        "stage_count": reference["stage_count"],
        "wall_radius_m": float(case.geometry.chamber.outer_radius_m),
        "stage_pitch_m": float(case.derived["represented_stage_pitch_m"]),
        "x_w": reference["x_w"],
        "source_strength_scale": reference["source_strength_scale"],
        "l1a_wall_cusp_count": reference["wall_cusp_count"],
        "l1a_cell_count": reference["cell_count"],
        "l1a_min_rho_conservative": reference["min_rho_conservative"],
    }
