"""Closed, deterministic L1b artifact and viewer contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from math import hypot, isfinite
from pathlib import Path
from re import fullmatch
from typing import Any

from .acceptance import assess_publication, validate_publication_evidence
from .models import MaterialFieldResult, MaterialFieldValidationError

SCHEMA_VERSION = "cft_revival.material_fields.result/1.4.0"
VIEWER_SCHEMA_VERSION = "cft_revival.material_fields.viewer/1.4.0"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
CLASSIFICATION = "hypothetical_design_simulation_not_validated_hardware_prediction"
MANIFEST_SCHEMA_VERSION = "cft_revival.material_fields.design_manifest/1.4.0"


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise MaterialFieldValidationError("artifact contains nonfinite/unsupported data") from error


def _seal(payload: dict[str, object]) -> dict[str, object]:
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": CANONICALIZATION,
            "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        },
    }


def _indices(length: int, stride: int) -> tuple[int, ...]:
    values = list(range(0, length, stride))
    if values[-1] != length - 1:
        values.append(length - 1)
    return tuple(values)


def topology_descriptors(result: MaterialFieldResult) -> dict[str, object]:
    bz = result.field.b_z_t[0]
    z = result.field.z_m
    peak = max(
        hypot(br, axial)
        for br_row, bz_row in zip(result.field.b_r_t, result.field.b_z_t)
        for br, axial in zip(br_row, bz_row)
    )
    tolerance = max(1.0e-14, 1.0e-9 * peak)
    nulls: list[float] = []
    for j in range(1, len(z) - 1):
        if abs(bz[j]) <= tolerance:
            nulls.append(z[j])
        elif bz[j - 1] * bz[j] < 0.0:
            fraction = abs(bz[j - 1]) / (abs(bz[j - 1]) + abs(bz[j]))
            nulls.append(z[j - 1] + fraction * (z[j] - z[j - 1]))
    cusps = [
        {"z_m": z[j], "b_z_t": bz[j]}
        for j in range(1, len(z) - 1)
        if abs(bz[j]) >= abs(bz[j - 1]) and abs(bz[j]) > abs(bz[j + 1])
    ]
    return {
        "axis_null_z_m": nulls,
        "axis_cusps": cusps,
        "null_tolerance_t": tolerance,
        "classification": "sampled_axis_topology_not_continuous_critical_point_proof",
    }


def material_field_artifact(
    result: MaterialFieldResult,
    *,
    domain_expansions: tuple[MaterialFieldResult, ...],
    mesh_fine: MaterialFieldResult,
    mesh_third: MaterialFieldResult,
    alignment_sweeps: tuple[MaterialFieldResult, ...],
    equivalent_base: MaterialFieldResult,
    equivalent_fine: MaterialFieldResult,
    parity_cpu: MaterialFieldResult,
    parity_cuda: MaterialFieldResult,
    downsample_stride: int = 4,
    precomputed_evidence: PublicationEvidence | None = None,
    qualification: dict[str, object] | None = None,
    validate_value: bool = True,
) -> dict[str, object]:
    if isinstance(downsample_stride, bool) or not isinstance(downsample_stride, int) or downsample_stride < 1:
        raise MaterialFieldValidationError("downsample_stride must be an integer >= 1")
    publication_evidence = precomputed_evidence or assess_publication(
            result,
            domain_expansions=domain_expansions,
            mesh_fine=mesh_fine,
            mesh_third=mesh_third,
            alignment_sweeps=alignment_sweeps,
            equivalent_base=equivalent_base,
            equivalent_fine=equivalent_fine,
            parity_cpu=parity_cpu,
            parity_cuda=parity_cuda,
            qualification=qualification,
        )
    field = result.field
    ri = _indices(len(field.r_m), downsample_stride)
    zj = _indices(len(field.z_m), downsample_stride)
    magnitude = [
        [hypot(field.b_r_t[i][j], field.b_z_t[i][j]) for j in range(len(field.z_m))]
        for i in range(len(field.r_m))
    ]
    gate_statuses = {
        gate_id: status
        for gate_id, status, _, _, _ in publication_evidence.gates
    }
    sampled_peak = max(value for row in magnitude for value in row)
    axis_peak = max(abs(value) for value in field.b_z_t[0])
    diagnostics = asdict(result.diagnostics)
    diagnostics["residual_history_l2"] = list(result.diagnostics.residual_history_l2)
    diagnostics["rasterization"] = [asdict(item) for item in result.problem.raster_diagnostics]
    diagnostics["weak_source_action"] = [
        asdict(item) for item in result.problem.weak_action_diagnostics
    ]
    diagnostics["free_current_represented_a"] = sum(
        result.problem.free_current_phi_a_per_m2
    ) * result.problem.domain.dr_m * result.problem.domain.dz_m
    diagnostics["pm_bound_current_represented_a"] = sum(
        result.problem.pm_bound_current_phi_a_per_m2
    ) * result.problem.domain.dr_m * result.problem.domain.dz_m
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_level": "L1b",
        "classification": CLASSIFICATION,
        "acceptance": publication_evidence.to_dict(),
        "anchors": {
            "problem_id": result.problem.problem_id,
            "geometry_sha256": result.problem.geometry_sha256,
            "magnetics_sha256": result.problem.magnetics_sha256,
            "design_geometry_sha256": publication_evidence.raw_runs[0].design_geometry_sha256,
            "material_registry_sha256": publication_evidence.raw_runs[0].material_registry_sha256,
            "base_run_sha256": publication_evidence.raw_runs[0].run_sha256,
            "config_sha256": result.diagnostics.run_config_sha256,
            "solver_config_identity_sha256": publication_evidence.raw_runs[0].solver_config_identity_sha256,
            "implementation_sha256": result.diagnostics.implementation_sha256,
            "evidence_implementation_sha256": publication_evidence.raw_runs[0].evidence_implementation_sha256,
            "grid_sha256": publication_evidence.raw_runs[0].grid_sha256,
            "domain_sha256": publication_evidence.raw_runs[0].domain_sha256,
            "problem_sha256": publication_evidence.raw_runs[0].problem_sha256,
        },
        "formulation": {
            "unknown": "psi=r*A_phi [Wb/rad]",
            "operator": "-div((nu/r)*grad(psi))",
            "rhs": "J_free_phi+J_bound_phi-div(nu*Br_z,-nu*Br_r)",
            "face_treatment": "exact_series_resistance_with_radial_1_over_r",
            "source_discretization": "face_integrated_discrete_adjoint",
            "outer_boundary": result.problem.outer_boundary_kind,
            "pm_authority": result.problem.authority,
            "nonlinear_status": result.problem.nonlinear_status,
        },
        "domain": {
            "radius_m": result.problem.domain.radius_m,
            "z_min_m": result.problem.domain.z_min_m,
            "z_max_m": result.problem.domain.z_max_m,
            "radial_intervals": result.problem.domain.radial_intervals,
            "axial_intervals": result.problem.domain.axial_intervals,
            "dr_m": result.problem.domain.dr_m,
            "dz_m": result.problem.domain.dz_m,
            "tolerances_m": list(result.problem.tolerances_m),
            "open_boundary_policy": dict(result.problem.open_boundary_policy),
        },
        "handoff_provenance": {
            "pm_region_count": result.problem.pm_region_count,
            "material_region_count": result.problem.authoritative_material_region_count,
            "free_current_source_count": result.problem.authoritative_free_current_source_count,
            "interface_count": result.problem.handoff_interface_count,
            "geometry_regions": [
                {
                    "region_id": item[0],
                    "shape": item[1],
                    "represented_in_magnetics_handoff": item[2],
                    "disposition": item[3],
                }
                for item in result.problem.geometry_region_provenance
            ],
            "geometry_schema_version": result.problem.geometry_schema_version,
            "geometry_payload_sha256": result.problem.geometry_sha256,
        },
        "diagnostics": diagnostics,
        "summary": {
            "sampled_cell_peak": {
                "value_t": sampled_peak,
                "classification": "SCREENING_ONLY_INTERFACE_SENSITIVE",
                "mesh_gate_status": gate_statuses["mesh_fixed_qoi"],
            },
            "axis_bz_peak": {
                "value_t": axis_peak,
                "classification": "SAMPLED_AXIS_EXTREMUM_SCREENING_ONLY",
                "mesh_gate_status": gate_statuses["mesh_fixed_qoi"],
            },
            "fixed_qois_bz_t": publication_evidence.studies[0]["fixed_qois_bz_t"],
            "topology": topology_descriptors(result),
            "warning_codes": list(publication_evidence.warning_codes),
            "pm_model_form_comparison": publication_evidence.pm_model_form_comparison,
        },
        "full_field_map": {
            "layout": "radial-major",
            "r_m": list(field.r_m),
            "z_m": list(field.z_m),
            "psi_wb": [list(row) for row in field.psi_wb],
            "b_r_t": [list(row) for row in field.b_r_t],
            "b_z_t": [list(row) for row in field.b_z_t],
            "b_magnitude_t": magnitude,
            "material_id": [list(row) for row in result.material_ids],
            "free_current_phi_a_per_m2": [
                list(row) for row in result.source_phi_a_per_m2
            ],
            "pm_bound_current_phi_a_per_m2": _rows(
                result.problem.pm_bound_current_phi_a_per_m2, result.problem.domain.shape
            ),
            "remanence_r_t": [list(row) for row in result.remanence_r_t],
            "remanence_z_t": [list(row) for row in result.remanence_z_t],
        },
        "downsampled_field_map": {
            "stride": downsample_stride,
            "r_m": [field.r_m[i] for i in ri],
            "z_m": [field.z_m[j] for j in zj],
            "psi_wb": [[field.psi_wb[i][j] for j in zj] for i in ri],
            "b_r_t": [[field.b_r_t[i][j] for j in zj] for i in ri],
            "b_z_t": [[field.b_z_t[i][j] for j in zj] for i in ri],
            "b_magnitude_t": [[magnitude[i][j] for j in zj] for i in ri],
            "material_id": [[result.material_ids[i][j] for j in zj] for i in ri],
            "free_current_phi_a_per_m2": [
                [result.source_phi_a_per_m2[i][j] for j in zj] for i in ri
            ],
            "pm_bound_current_phi_a_per_m2": [
                [result.problem.pm_bound_current_phi_a_per_m2[i * len(field.z_m) + j] for j in zj]
                for i in ri
            ],
        },
        "limitations": [
            "Linear recoil PM and linear pole/yoke permeability only.",
            "Face transmissibilities use exact series resistance with linear-edge clipping; interface peaks remain screening-only.",
            "Dipole-Robin truncation remains acceptable only when both recorded nested-domain gates pass.",
            "No hysteresis, validated saturation, irreversible demagnetization, plasma response, or calibration.",
            "Hypothetical design simulation; not a validated hardware prediction.",
        ],
    }
    artifact = _seal(payload)
    if validate_value:
        validate_artifact(artifact, require_accepted=False)
    return artifact


def _rows(values, shape):
    nr, nz = shape
    return [[values[i * nz + j] for j in range(nz)] for i in range(nr)]


def viewer_contract(
    artifact: dict[str, object], *, validate_source: bool = True
) -> dict[str, object]:
    if validate_source:
        validate_artifact(artifact, require_accepted=False)
    payload = {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "model_level": "L1b",
        "artifact_payload_sha256": artifact["integrity"]["payload_sha256"],
        "classification": artifact["classification"],
        "acceptance_status": artifact["acceptance"]["status"],
        "anchors": artifact["anchors"],
        "summary": artifact["summary"],
        "field_map": artifact["downsampled_field_map"],
        "units": {
            "r_m": "m", "z_m": "m", "psi_wb": "Wb/rad", "b_r_t": "T",
            "b_z_t": "T", "b_magnitude_t": "T",
            "free_current_phi_a_per_m2": "A/m^2",
            "pm_bound_current_phi_a_per_m2": "A/m^2",
        },
    }
    viewer = _seal(payload)
    if validate_source:
        validate_viewer_contract(viewer, artifact=artifact)
    return viewer


def _closed(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise MaterialFieldValidationError(
            f"{name} keys differ: missing={sorted(keys-actual)}, extra={sorted(actual-keys)}"
        )
    return value


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise MaterialFieldValidationError(f"{name} must be finite numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise MaterialFieldValidationError(f"{name} is below its minimum")
    return result


def _integrity(value: dict[str, object], name: str) -> None:
    integrity = _closed(
        value.get("integrity"), f"{name}.integrity",
        {"algorithm", "canonicalization", "payload_sha256"},
    )
    if integrity["algorithm"] != "sha256" or integrity["canonicalization"] != CANONICALIZATION:
        raise MaterialFieldValidationError(f"{name} integrity declaration is unsupported")
    digest = integrity["payload_sha256"]
    if not isinstance(digest, str) or not fullmatch(r"[0-9a-f]{64}", digest):
        raise MaterialFieldValidationError(f"{name} digest is not canonical SHA-256")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if digest != hashlib.sha256(_canonical(payload)).hexdigest():
        raise MaterialFieldValidationError(f"{name} payload SHA-256 mismatch")


def _matrix(value: object, name: str, rows: int, columns: int, *, strings: bool = False) -> None:
    if not isinstance(value, list) or len(value) != rows:
        raise MaterialFieldValidationError(f"{name} row shape mismatch")
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise MaterialFieldValidationError(f"{name} column shape mismatch")
        if strings:
            if any(not isinstance(item, str) or not item for item in row):
                raise MaterialFieldValidationError(f"{name} contains invalid IDs")
        else:
            for item in row:
                _number(item, name)


def _validate_map(value: object, name: str, *, downsampled: bool) -> None:
    keys = {
        "r_m", "z_m", "psi_wb", "b_r_t", "b_z_t", "b_magnitude_t", "material_id",
        "free_current_phi_a_per_m2", "pm_bound_current_phi_a_per_m2",
    }
    if downsampled:
        keys.add("stride")
    else:
        keys.update({"layout", "remanence_r_t", "remanence_z_t"})
    mapping = _closed(value, name, keys)
    r = mapping["r_m"]
    z = mapping["z_m"]
    if not isinstance(r, list) or not isinstance(z, list) or len(r) < 2 or len(z) < 2:
        raise MaterialFieldValidationError(f"{name} coordinates are invalid")
    for item in (*r, *z):
        _number(item, name)
    if any(right <= left for left, right in zip(r, r[1:])) or any(
        right <= left for left, right in zip(z, z[1:])
    ):
        raise MaterialFieldValidationError(f"{name} coordinates must increase")
    for key in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t", "free_current_phi_a_per_m2", "pm_bound_current_phi_a_per_m2"):
        _matrix(mapping[key], f"{name}.{key}", len(r), len(z))
    _matrix(mapping["material_id"], f"{name}.material_id", len(r), len(z), strings=True)
    if not downsampled:
        if mapping["layout"] != "radial-major":
            raise MaterialFieldValidationError("full map layout is unsupported")
        _matrix(mapping["remanence_r_t"], f"{name}.remanence_r_t", len(r), len(z))
        _matrix(mapping["remanence_z_t"], f"{name}.remanence_z_t", len(r), len(z))
    else:
        if isinstance(mapping["stride"], bool) or not isinstance(mapping["stride"], int) or mapping["stride"] < 1:
            raise MaterialFieldValidationError("downsample stride is invalid")
    for i in range(len(r)):
        for j in range(len(z)):
            expected = hypot(mapping["b_r_t"][i][j], mapping["b_z_t"][i][j])
            actual = mapping["b_magnitude_t"][i][j]
            if abs(actual - expected) > max(1.0e-14, 1.0e-12 * expected):
                raise MaterialFieldValidationError(f"{name} field magnitude is inconsistent")


def _validate_acceptance(
    value: object, *, policy: dict[str, float | int]
) -> str:
    return validate_publication_evidence(value, policy=policy)


def validate_artifact(
    artifact: dict[str, object], *, require_accepted: bool = True
) -> None:
    top = _closed(
        artifact, "artifact",
        {"schema_version", "model_level", "classification", "acceptance", "anchors",
         "formulation", "domain", "handoff_provenance", "diagnostics", "summary",
         "full_field_map", "downsampled_field_map", "limitations", "integrity"},
    )
    if top["schema_version"] != SCHEMA_VERSION or top["model_level"] != "L1b" or top["classification"] != CLASSIFICATION:
        raise MaterialFieldValidationError("artifact schema/model/classification is unsupported")
    anchors = _closed(
        top["anchors"],
        "anchors",
        {
            "problem_id", "geometry_sha256", "magnetics_sha256",
            "design_geometry_sha256", "material_registry_sha256",
            "base_run_sha256", "config_sha256", "implementation_sha256",
            "solver_config_identity_sha256",
            "evidence_implementation_sha256",
            "grid_sha256", "domain_sha256", "problem_sha256",
        },
    )
    if not isinstance(anchors["problem_id"], str) or not anchors["problem_id"]:
        raise MaterialFieldValidationError("problem anchor is invalid")
    for key in (
        "geometry_sha256", "magnetics_sha256", "design_geometry_sha256",
        "material_registry_sha256", "base_run_sha256",
        "config_sha256", "solver_config_identity_sha256", "implementation_sha256", "evidence_implementation_sha256", "grid_sha256",
        "domain_sha256", "problem_sha256",
    ):
        if not isinstance(anchors[key], str) or not fullmatch(r"[0-9a-f]{64}", anchors[key]):
            raise MaterialFieldValidationError(f"{key} is not canonical SHA-256")
    formulation = _closed(
        top["formulation"], "formulation",
        {"unknown", "operator", "rhs", "face_treatment", "source_discretization",
         "outer_boundary", "pm_authority", "nonlinear_status"},
    )
    if (
        formulation["face_treatment"]
        != "exact_series_resistance_with_radial_1_over_r"
        or formulation["source_discretization"]
        != "face_integrated_discrete_adjoint"
        or formulation["outer_boundary"] != "dipole_robin_psi"
        or formulation["pm_authority"] not in {
        "recoil_remanence_constitutive", "equivalent_bound_current"
        }
    ):
        raise MaterialFieldValidationError("formulation identifiers are unsupported")
    domain = _closed(
        top["domain"], "domain",
        {"radius_m", "z_min_m", "z_max_m", "radial_intervals", "axial_intervals",
         "dr_m", "dz_m", "tolerances_m", "open_boundary_policy"},
    )
    for key in ("radius_m", "z_min_m", "z_max_m", "dr_m", "dz_m"):
        _number(domain[key], f"domain.{key}")
    for key in ("radial_intervals", "axial_intervals"):
        if isinstance(domain[key], bool) or not isinstance(domain[key], int) or domain[key] < 4:
            raise MaterialFieldValidationError(f"domain.{key} is invalid")
    if (
        domain["radius_m"] <= 0.0
        or domain["z_max_m"] <= domain["z_min_m"]
        or domain["dr_m"] != domain["radius_m"] / domain["radial_intervals"]
        or domain["dz_m"] != (domain["z_max_m"] - domain["z_min_m"]) / domain["axial_intervals"]
    ):
        raise MaterialFieldValidationError("domain extents/spacings are inconsistent")
    if not isinstance(domain["tolerances_m"], list) or len(domain["tolerances_m"]) != 2:
        raise MaterialFieldValidationError("domain tolerances are invalid")
    for item in domain["tolerances_m"]:
        _number(item, "domain tolerance", minimum=0.0)
    policy = _closed(
        domain["open_boundary_policy"], "open_boundary_policy",
        {"minimum_padding_characteristic_lengths", "maximum_boundary_to_peak_field_ratio",
         "domain_expansion_factor", "required_expansion_comparisons", "maximum_qoi_relative_change"},
    )
    for key, value in policy.items():
        _number(value, f"open_boundary_policy.{key}", minimum=0.0)
    status = _validate_acceptance(top["acceptance"], policy=policy)
    if require_accepted and status != "ACCEPTED_PUBLICATION_EVIDENCE":
        raise MaterialFieldValidationError("screening artifact is not publication evidence")
    provenance = _closed(
        top["handoff_provenance"], "handoff_provenance",
        {"pm_region_count", "material_region_count", "free_current_source_count",
         "interface_count", "geometry_regions", "geometry_schema_version",
         "geometry_payload_sha256"},
    )
    if not isinstance(provenance["geometry_regions"], list) or not provenance["geometry_regions"]:
        raise MaterialFieldValidationError("geometry provenance must be a list")
    region_ids: set[str] = set()
    for item in provenance["geometry_regions"]:
        region = _closed(item, "geometry region", {"region_id", "shape", "represented_in_magnetics_handoff", "disposition"})
        if (
            not isinstance(region["region_id"], str)
            or not region["region_id"]
            or region["shape"] not in {"rectangular_annulus", "linear_taper_annulus"}
            or not isinstance(region["represented_in_magnetics_handoff"], bool)
            or not isinstance(region["disposition"], str)
            or not region["disposition"]
        ):
            raise MaterialFieldValidationError("geometry region provenance is invalid")
        if region["region_id"] in region_ids:
            raise MaterialFieldValidationError("geometry region provenance is duplicated")
        region_ids.add(region["region_id"])
    for key in ("pm_region_count", "material_region_count", "free_current_source_count", "interface_count"):
        if isinstance(provenance[key], bool) or not isinstance(provenance[key], int) or provenance[key] < 0:
            raise MaterialFieldValidationError(f"handoff {key} is invalid")
    if provenance["geometry_schema_version"] != "cft_revival.geometry.axisymmetric_cft/1.1.0":
        raise MaterialFieldValidationError("geometry schema binding is unsupported")
    if provenance["geometry_payload_sha256"] != anchors["geometry_sha256"]:
        raise MaterialFieldValidationError("geometry payload binding disagrees with anchor")
    diagnostics = _closed(
        top["diagnostics"], "diagnostics",
        {"converged", "iterations", "initial_residual_l2", "final_true_residual_l2",
         "relative_true_residual_l2", "residual_history_l2", "true_residual_restarts",
         "backend", "magnetic_energy_j", "source_coenergy_j", "energy_balance_relative",
         "run_config_sha256", "implementation_sha256", "run_config_json",
         "host_synchronization_count", "convergence_check_interval",
         "rasterization", "weak_source_action", "free_current_represented_a",
         "pm_bound_current_represented_a"},
    )
    if (
        diagnostics["converged"] is not True
        or not isinstance(diagnostics["backend"], str)
        or not fullmatch(r"material_fields:(python|warp:(cpu|cuda:[0-9]+))", diagnostics["backend"])
    ):
        raise MaterialFieldValidationError("solver convergence/backend metadata is invalid")
    for key in ("iterations", "true_residual_restarts"):
        if isinstance(diagnostics[key], bool) or not isinstance(diagnostics[key], int) or diagnostics[key] < 0:
            raise MaterialFieldValidationError(f"diagnostics.{key} is invalid")
    for key in ("initial_residual_l2", "final_true_residual_l2", "relative_true_residual_l2",
                "magnetic_energy_j", "source_coenergy_j", "energy_balance_relative",
                "free_current_represented_a", "pm_bound_current_represented_a"):
        _number(diagnostics[key], f"diagnostics.{key}")
    if not isinstance(diagnostics["residual_history_l2"], list) or not diagnostics["residual_history_l2"]:
        raise MaterialFieldValidationError("residual history is required")
    if not isinstance(diagnostics["rasterization"], list) or not diagnostics["rasterization"] or not isinstance(diagnostics["weak_source_action"], list) or not diagnostics["weak_source_action"]:
        raise MaterialFieldValidationError("raster/source diagnostics are required")
    for key in ("run_config_sha256", "implementation_sha256"):
        if not isinstance(diagnostics[key], str) or not fullmatch(r"[0-9a-f]{64}", diagnostics[key]):
            raise MaterialFieldValidationError(f"diagnostics.{key} is invalid")
    if (
        not isinstance(diagnostics["run_config_json"], str)
        or hashlib.sha256(diagnostics["run_config_json"].encode("utf-8")).hexdigest()
        != diagnostics["run_config_sha256"]
    ):
        raise MaterialFieldValidationError("diagnostics.run_config_json binding is invalid")
    for key in ("host_synchronization_count", "convergence_check_interval"):
        if (
            isinstance(diagnostics[key], bool)
            or not isinstance(diagnostics[key], int)
            or diagnostics[key] < 0
        ):
            raise MaterialFieldValidationError(f"diagnostics.{key} is invalid")
    raster_keys = {
        "item_id", "requested_volume_m3", "represented_volume_m3", "relative_volume_error",
        "requested_source_measure", "represented_source_measure", "relative_source_error",
    }
    for item in diagnostics["rasterization"]:
        raster = _closed(item, "raster diagnostic", raster_keys)
        if not isinstance(raster["item_id"], str) or not raster["item_id"]:
            raise MaterialFieldValidationError("raster diagnostic ID is invalid")
        for key in raster_keys - {"item_id"}:
            _number(raster[key], f"raster.{key}")
    action_keys = {
        "basis_id", "analytical_action_a", "rasterized_action_a",
        "absolute_bias_a", "relative_bias",
    }
    for item in diagnostics["weak_source_action"]:
        action = _closed(item, "weak action", action_keys)
        if not isinstance(action["basis_id"], str) or not action["basis_id"]:
            raise MaterialFieldValidationError("weak-action basis ID is invalid")
        for key in action_keys - {"basis_id"}:
            _number(action[key], f"weak_action.{key}")
    summary = _closed(
        top["summary"], "summary",
        {"sampled_cell_peak", "axis_bz_peak", "fixed_qois_bz_t", "topology",
         "warning_codes", "pm_model_form_comparison"},
    )
    sampled = _closed(summary["sampled_cell_peak"], "sampled peak", {"value_t", "classification", "mesh_gate_status"})
    axis = _closed(summary["axis_bz_peak"], "axis peak", {"value_t", "classification", "mesh_gate_status"})
    if sampled["classification"] != "SCREENING_ONLY_INTERFACE_SENSITIVE" or axis["classification"] != "SAMPLED_AXIS_EXTREMUM_SCREENING_ONLY":
        raise MaterialFieldValidationError("peak classifications are unsupported")
    for peak in (sampled, axis):
        _number(peak["value_t"], "peak.value_t", minimum=0.0)
        if peak["mesh_gate_status"] not in {"PASS", "FAIL", "NOT_EVALUATED"}:
            raise MaterialFieldValidationError("peak mesh gate status is invalid")
    topology = _closed(summary["topology"], "topology", {"axis_null_z_m", "axis_cusps", "null_tolerance_t", "classification"})
    if not isinstance(topology["axis_null_z_m"], list) or not isinstance(topology["axis_cusps"], list):
        raise MaterialFieldValidationError("topology arrays are invalid")
    for item in topology["axis_null_z_m"]:
        _number(item, "topology null")
    for item in topology["axis_cusps"]:
        cusp = _closed(item, "axis cusp", {"z_m", "b_z_t"})
        _number(cusp["z_m"], "cusp.z_m")
        _number(cusp["b_z_t"], "cusp.b_z_t")
    _number(topology["null_tolerance_t"], "topology tolerance", minimum=0.0)
    if topology["classification"] != "sampled_axis_topology_not_continuous_critical_point_proof":
        raise MaterialFieldValidationError("topology classification is unsupported")
    if not isinstance(summary["warning_codes"], list) or summary["warning_codes"] != top["acceptance"]["warning_codes"]:
        raise MaterialFieldValidationError("summary warnings are not acceptance-derived")
    if not isinstance(summary["fixed_qois_bz_t"], dict) or not summary["fixed_qois_bz_t"]:
        raise MaterialFieldValidationError("fixed physical QoIs are required")
    for value in summary["fixed_qois_bz_t"].values():
        _number(value, "fixed QoI")
    comparison = _closed(
        summary["pm_model_form_comparison"], "pm_model_form_comparison",
        {
            "base_fixed_qoi_relative_difference",
            "fine_fixed_qoi_relative_difference",
            "discrepancy_change",
        },
    )
    for key, value in comparison.items():
        _number(value, f"pm_model_form_comparison.{key}", minimum=0.0)
    _validate_map(top["full_field_map"], "full_field_map", downsampled=False)
    _validate_map(top["downsampled_field_map"], "downsampled_field_map", downsampled=True)
    full_map = top["full_field_map"]
    downsampled = top["downsampled_field_map"]
    stride = downsampled["stride"]
    selected_r = _indices(len(full_map["r_m"]), stride)
    selected_z = _indices(len(full_map["z_m"]), stride)
    if downsampled["r_m"] != [full_map["r_m"][i] for i in selected_r] or downsampled[
        "z_m"
    ] != [full_map["z_m"][j] for j in selected_z]:
        raise MaterialFieldValidationError("downsampled coordinates are inconsistent")
    if len(full_map["r_m"]) != domain["radial_intervals"] + 1 or len(
        full_map["z_m"]
    ) != domain["axial_intervals"] + 1:
        raise MaterialFieldValidationError("full map shape disagrees with domain")
    acceptance = top["acceptance"]
    base_raw = next(
        item for item in acceptance["raw_runs"] if item["role"] == "base"
    )
    from .replay import replay_raw_run

    replay = replay_raw_run(base_raw["raw"], backend=base_raw["backend"])
    if (
        [list(row) for row in replay.b_r_t] != full_map["b_r_t"]
        or [list(row) for row in replay.b_z_t] != full_map["b_z_t"]
    ):
        raise MaterialFieldValidationError(
            "full field map differs from deterministic base-run replay"
        )
    bound_counts = base_raw["raw"]["problem"]["counts"]
    gates_by_id = {item["gate_id"]: item for item in acceptance["gates"]}
    if (
        gates_by_id["energy_balance"]["measured_value"] != diagnostics["energy_balance_relative"]
        or gates_by_id["true_equation_residual"]["measured_value"]
        != diagnostics["relative_true_residual_l2"]
        or sampled["mesh_gate_status"] != gates_by_id["mesh_fixed_qoi"]["status"]
        or axis["mesh_gate_status"] != gates_by_id["mesh_fixed_qoi"]["status"]
        or acceptance["studies"][0]["interior_cellwise_max_t"] != sampled["value_t"]
        or acceptance["studies"][0]["fixed_qois_bz_t"] != summary["fixed_qois_bz_t"]
        or gates_by_id["boundary_field_ratio"]["threshold"]
        != policy["maximum_boundary_to_peak_field_ratio"]
        or gates_by_id["successive_fixed_qoi"]["threshold"]
        != policy["maximum_qoi_relative_change"]
        or provenance["material_region_count"] != bound_counts["material_regions"]
        or provenance["pm_region_count"] != bound_counts["pm_regions"]
        or provenance["free_current_source_count"] != bound_counts["free_current_sources"]
        or provenance["interface_count"] != bound_counts["interfaces"]
        or anchors["base_run_sha256"] != base_raw["run_sha256"]
        or anchors["config_sha256"] != base_raw["config_sha256"]
        or anchors["solver_config_identity_sha256"]
        != base_raw["solver_config_identity_sha256"]
        or anchors["implementation_sha256"] != base_raw["implementation_sha256"]
        or anchors["evidence_implementation_sha256"]
        != base_raw["evidence_implementation_sha256"]
        or anchors["grid_sha256"] != base_raw["grid_sha256"]
        or anchors["domain_sha256"] != base_raw["domain_sha256"]
        or anchors["problem_sha256"] != base_raw["problem_sha256"]
        or anchors["geometry_sha256"] != base_raw["geometry_sha256"]
        or anchors["magnetics_sha256"] != base_raw["material_sha256"]
        or anchors["design_geometry_sha256"] != base_raw["design_geometry_sha256"]
        or anchors["material_registry_sha256"] != base_raw["material_registry_sha256"]
        or len(diagnostics["rasterization"])
        != provenance["material_region_count"] + provenance["pm_region_count"]
    ):
        raise MaterialFieldValidationError("artifact gates disagree with diagnostics/policy")
    if not isinstance(top["limitations"], list) or not top["limitations"] or any(
        not isinstance(item, str) or not item for item in top["limitations"]
    ):
        raise MaterialFieldValidationError("limitations must be a non-empty string list")
    _integrity(artifact, "artifact")


def validate_viewer_contract(
    viewer: dict[str, object], *, artifact: dict[str, object] | None = None
) -> None:
    top = _closed(
        viewer, "viewer",
        {"schema_version", "model_level", "artifact_payload_sha256", "classification",
         "acceptance_status", "anchors", "summary", "field_map", "units", "integrity"},
    )
    if top["schema_version"] != VIEWER_SCHEMA_VERSION or top["model_level"] != "L1b" or top["classification"] != CLASSIFICATION:
        raise MaterialFieldValidationError("viewer schema/model/classification is unsupported")
    if top["acceptance_status"] not in {"ACCEPTED_PUBLICATION_EVIDENCE", "SCREENING_NOT_ACCEPTED"}:
        raise MaterialFieldValidationError("viewer acceptance status is unsupported")
    if not isinstance(top["artifact_payload_sha256"], str) or not fullmatch(r"[0-9a-f]{64}", top["artifact_payload_sha256"]):
        raise MaterialFieldValidationError("viewer artifact anchor is invalid")
    anchors = _closed(
        top["anchors"],
        "viewer.anchors",
        {
            "problem_id", "geometry_sha256", "magnetics_sha256",
            "design_geometry_sha256", "material_registry_sha256",
            "base_run_sha256", "config_sha256", "implementation_sha256",
            "solver_config_identity_sha256",
            "evidence_implementation_sha256",
            "grid_sha256", "domain_sha256", "problem_sha256",
        },
    )
    if not isinstance(anchors["problem_id"], str) or not anchors["problem_id"]:
        raise MaterialFieldValidationError("viewer problem anchor is invalid")
    for key in (
        "geometry_sha256", "magnetics_sha256", "design_geometry_sha256",
        "material_registry_sha256", "base_run_sha256",
        "config_sha256", "solver_config_identity_sha256", "implementation_sha256", "evidence_implementation_sha256", "grid_sha256",
        "domain_sha256", "problem_sha256",
    ):
        if not isinstance(anchors[key], str) or not fullmatch(r"[0-9a-f]{64}", anchors[key]):
            raise MaterialFieldValidationError("viewer hash anchor is invalid")
    summary = _closed(top["summary"], "viewer.summary", {"sampled_cell_peak", "axis_bz_peak", "fixed_qois_bz_t", "topology", "warning_codes", "pm_model_form_comparison"})
    sampled = _closed(summary["sampled_cell_peak"], "viewer.sampled_peak", {"value_t", "classification", "mesh_gate_status"})
    axis = _closed(summary["axis_bz_peak"], "viewer.axis_peak", {"value_t", "classification", "mesh_gate_status"})
    for peak in (sampled, axis):
        _number(peak["value_t"], "viewer peak", minimum=0.0)
        if (
            not isinstance(peak["classification"], str)
            or peak["mesh_gate_status"] not in {"PASS", "FAIL", "NOT_EVALUATED"}
        ):
            raise MaterialFieldValidationError("viewer peak metadata is invalid")
    topology = _closed(summary["topology"], "viewer.topology", {"axis_null_z_m", "axis_cusps", "null_tolerance_t", "classification"})
    if not isinstance(topology["axis_null_z_m"], list) or not isinstance(topology["axis_cusps"], list):
        raise MaterialFieldValidationError("viewer topology arrays are invalid")
    for item in topology["axis_null_z_m"]:
        _number(item, "viewer axis null")
    for item in topology["axis_cusps"]:
        cusp = _closed(item, "viewer cusp", {"z_m", "b_z_t"})
        _number(cusp["z_m"], "viewer cusp z")
        _number(cusp["b_z_t"], "viewer cusp Bz")
    comparison = _closed(
        summary["pm_model_form_comparison"], "viewer.pm_comparison",
        {"base_fixed_qoi_relative_difference", "fine_fixed_qoi_relative_difference",
         "discrepancy_change"},
    )
    for value in comparison.values():
        _number(value, "viewer PM comparison", minimum=0.0)
    if not isinstance(summary["warning_codes"], list) or not summary["warning_codes"]:
        raise MaterialFieldValidationError("viewer warnings are invalid")
    if not isinstance(summary["fixed_qois_bz_t"], dict) or not summary["fixed_qois_bz_t"]:
        raise MaterialFieldValidationError("viewer fixed QoIs are invalid")
    _validate_map(top["field_map"], "viewer.field_map", downsampled=True)
    units = _closed(
        top["units"], "viewer.units",
        {"r_m", "z_m", "psi_wb", "b_r_t", "b_z_t", "b_magnitude_t",
         "free_current_phi_a_per_m2", "pm_bound_current_phi_a_per_m2"},
    )
    if any(not isinstance(item, str) or not item for item in units.values()):
        raise MaterialFieldValidationError("viewer units are invalid")
    _integrity(viewer, "viewer")
    if artifact is None:
        raise MaterialFieldValidationError(
            "referenced artifact is required to validate viewer acceptance"
        )
    validate_artifact(artifact, require_accepted=False)
    if (
        top["artifact_payload_sha256"] != artifact["integrity"]["payload_sha256"]
        or top["acceptance_status"] != artifact["acceptance"]["status"]
        or top["classification"] != artifact["classification"]
        or top["anchors"] != artifact["anchors"]
        or top["summary"] != artifact["summary"]
        or top["field_map"] != artifact["downsampled_field_map"]
    ):
        raise MaterialFieldValidationError("viewer is not the hash-bound artifact projection")


def write_json(
    path: str | Path,
    value: dict[str, object],
    *,
    referenced_artifact: dict[str, object] | None = None,
    validate_value: bool = True,
) -> str:
    if validate_value and value.get("schema_version") == SCHEMA_VERSION:
        validate_artifact(value, require_accepted=False)
    elif validate_value and value.get("schema_version") == VIEWER_SCHEMA_VERSION:
        validate_viewer_contract(value, artifact=referenced_artifact)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = data.encode("utf-8")
    target.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    target.with_name(target.name + ".sha256").write_bytes(
        f"{digest}  {target.name}\n".encode("ascii")
    )
    return digest


def validate_artifact_bundle(root: str | Path) -> dict[str, object]:
    """Strictly replay a complete, sidecar-bound three-design screening bundle."""
    directory = Path(root)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    top = _closed(
        manifest,
        "manifest",
        {"schema_version", "model_level", "designs", "limitations", "integrity"},
    )
    if (
        top["schema_version"] != MANIFEST_SCHEMA_VERSION
        or top["model_level"] != "L1b"
    ):
        raise MaterialFieldValidationError("manifest schema/model is unsupported")
    _integrity(manifest, "manifest")
    expected = {
        "historical-envelope-baseline-v1": "historical-envelope-baseline",
        "compact-high-gradient-stack-v1": "compact-high-gradient-stack",
        "divergent-exit-stack-v1": "divergent-exit-stack",
    }
    designs = top["designs"]
    if (
        not isinstance(designs, list)
        or len(designs) != len(expected)
        or {item.get("config_id") for item in designs if isinstance(item, dict)}
        != set(expected)
    ):
        raise MaterialFieldValidationError(
            "manifest design cardinality/identity is invalid"
        )
    common_identities: set[tuple[str, str]] = set()
    for entry in designs:
        keys = {
            "config_id",
            "artifact",
            "artifact_file_sha256",
            "artifact_payload_sha256",
            "geometry_sha256",
            "acceptance_status",
            "sampled_cell_peak_t",
            "axis_bz_peak_t",
            "relative_true_residual_l2",
            "energy_balance_relative",
            "boundary_to_peak_ratios",
            "fixed_qois_bz_t",
            "pm_model_form_comparison",
            "classification",
            "qualification",
        }
        _closed(entry, "manifest design", keys)
        stem = expected[entry["config_id"]]
        if entry["artifact"] != f"{stem}.material-field.json":
            raise MaterialFieldValidationError("manifest artifact name is invalid")
        artifact_path = directory / entry["artifact"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        validate_artifact(artifact, require_accepted=False)
        viewer_path = directory / f"{stem}.viewer.json"
        viewer = json.loads(viewer_path.read_text(encoding="utf-8"))
        validate_viewer_contract(viewer, artifact=artifact)
        artifact_file_sha = _validate_sidecar(artifact_path)
        _validate_sidecar(viewer_path)
        if (
            artifact["schema_version"] != SCHEMA_VERSION
            or artifact["acceptance"]["status"] != "SCREENING_NOT_ACCEPTED"
            or artifact["acceptance"]["qualification"] != entry["qualification"]
            or entry["acceptance_status"] != artifact["acceptance"]["status"]
            or entry["artifact_file_sha256"] != artifact_file_sha
            or entry["artifact_payload_sha256"]
            != artifact["integrity"]["payload_sha256"]
            or entry["geometry_sha256"] != artifact["anchors"]["geometry_sha256"]
            or entry["classification"] != artifact["classification"]
            or len(artifact["acceptance"]["raw_runs"]) != 10
        ):
            raise MaterialFieldValidationError(
                "manifest projection or v1.4 screening status is inconsistent"
            )
        common_identities.add(
            (
                artifact["anchors"]["solver_config_identity_sha256"],
                artifact["anchors"]["evidence_implementation_sha256"],
            )
        )
    if len(common_identities) != 1:
        raise MaterialFieldValidationError(
            "bundle mixes solver configuration or evidence-code versions"
        )
    _validate_sidecar(manifest_path)
    return manifest


def _validate_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    try:
        declared, name = sidecar.read_text(encoding="ascii").split()
    except (OSError, ValueError) as error:
        raise MaterialFieldValidationError(
            f"invalid sidecar for {path.name}"
        ) from error
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if name != path.name or declared != digest:
        raise MaterialFieldValidationError(
            f"sidecar hash mismatch for {path.name}"
        )
    return digest
