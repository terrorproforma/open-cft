"""Hash-bound sweep-v2 design binding: geometry, accepted L1a re-solve, bore ψ-grid.

Field provenance
----------------
The accepted L1a geometry sweep v2 (``modern/experiments/l1a_geometry_sweep_v2``)
stored full-field maps for its four representative designs only; every other
design is recorded as QoIs. This module therefore rebuilds every design with the
sweep's own ``build_case`` (same protocol, same geometry v1.1 generator, same
equivalent-current preview, same sampling seed) and re-solves it with the accepted
L1a CPU solver ``cft_revival.fields.solve_problem_cpu`` at the sweep-v2 protocol
resolution. Identity is proven three ways before any orbit is integrated:

* ``geometry_sha256``, ``source_sha256``, ``config_sha256`` and ``case_sha256`` of
  the rebuilt case must equal the values recorded in the sweep's sealed
  ``raw-results.json`` (bytes verified through the sweep's own sidecars);
* the re-solved QoIs must reproduce the recorded QoIs within the sweep's
  preregistered scale-aware replay tolerances;
* for the representatives, the re-solved ψ/B nodes must agree with the stored
  full-field artifact within the declared node tolerances.

The bore sub-grid handed to :class:`cft_revival.orbit_mc.PsiBicubicField` is the
accepted map restricted to the channel bore (``r <= wall radius``, ``0 <= z <=
chamber length``, padded to the enclosing grid nodes). Everything here is L1a
linear-vacuum equivalent-current screening: not P2-qualified, not hardware-valid.

Reuses (by import, with attribution) ``experiments.l1a_geometry_sweep_v2`` and
``experiments.cft_orbit_wall_loss_v4.adapter.file_sha256``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cft_revival.fields as fields_package
import cft_revival.geometry as geometry_package
import cft_revival.magnetics as magnetics_package
import cft_revival.optimization as optimization_package
from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    FieldMap,
    SolverConfig,
    solve_problem_cpu,
)
from cft_revival.orbit_mc import OrbitConfig, PsiBicubicField, compare_maps
from cft_revival.orbit_mc.artifacts import content_hash

from experiments.cft_orbit_wall_loss_v4.adapter import file_sha256
from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.l1a_geometry_sweep_v2 import protocol as sweep_protocol

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
SWEEP_ROOT = MODERN / "experiments" / "l1a_geometry_sweep_v2"
SWEEP_RESULTS = SWEEP_ROOT / "results"

PLASMA_MATERIAL_ID = "l1a-vacuum-bore-screening"
ELECTRON_MASS_KG = 9.1093837139e-31
EV_J = 1.602176634e-19


# --------------------------------------------------------------------------
# Field-pipeline source binding (solver version + inputs by hash)
# --------------------------------------------------------------------------


def field_pipeline_source_files() -> list[Path]:
    """Every source file whose bytes determine the re-solved fields."""

    packages = (
        fields_package,
        geometry_package,
        magnetics_package,
        optimization_package,
    )
    files: list[Path] = []
    for package in packages:
        root = Path(package.__file__).resolve().parent
        expected_root = (MODERN / "src" / "cft_revival").resolve()
        if root.parent != expected_root:
            raise RuntimeError(
                f"{package.__name__} is imported from {root}, not from this worktree"
            )
        files.extend(sorted(root.rglob("*.py")))
    files.extend(sorted((MODERN / "spec" / "fields").glob("*.json")))
    files.extend(
        SWEEP_ROOT / name
        for name in ("experiment.py", "protocol.py", "protocol.json", "protocol.json.sha256")
    )
    if len(files) < 8:
        raise RuntimeError("field pipeline source scope is incomplete")
    return files


def field_pipeline_source_sha256() -> str:
    """SHA-256 over (posix path, LF bytes) of the field pipeline sources (fail closed on CR)."""

    digest = hashlib.sha256()
    for path in field_pipeline_source_files():
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(
                f"field pipeline source {path.relative_to(MODERN).as_posix()} contains CR "
                "bytes; the hash is defined over LF working-tree bytes"
            )
        digest.update(path.relative_to(MODERN).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Sweep-v2 results binding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepBinding:
    manifest: Mapping[str, Any]
    raw: Mapping[str, Any]
    summary: Mapping[str, Any]
    cases_by_id: Mapping[str, Mapping[str, Any]]
    manifest_file_sha256: str
    raw_file_sha256: str
    summary_file_sha256: str


def load_sweep_binding(declaration: Mapping[str, Any]) -> SweepBinding:
    """Load the sweep-v2 bundle, verifying its sidecars and the declared hashes."""

    manifest_path = SWEEP_RESULTS / "manifest.json"
    raw_path = SWEEP_RESULTS / "raw-results.json"
    summary_path = SWEEP_RESULTS / "summary.json"
    manifest_sha = sweep_protocol.verify_sidecar(manifest_path)
    raw_sha = sweep_protocol.verify_sidecar(raw_path)
    summary_sha = sweep_protocol.verify_sidecar(summary_path)
    manifest = sweep_protocol.strict_json(manifest_path)
    raw = sweep_protocol.strict_json(raw_path)
    summary = sweep_protocol.strict_json(summary_path)
    sweep_protocol.validate_sealed(manifest)
    sweep_protocol.validate_sealed(raw)
    sweep_protocol.validate_sealed(summary)
    expected = {
        "manifest_file_sha256": manifest_sha,
        "raw_results_file_sha256": raw_sha,
        "summary_file_sha256": summary_sha,
        "raw_results_payload_sha256": raw["integrity"]["payload_sha256"],
        "summary_payload_sha256": summary["integrity"]["payload_sha256"],
        "protocol_payload_sha256": sweep.PROTOCOL["integrity"]["payload_sha256"],
        "preregistration_commit": manifest["preregistration_commit_sha"],
    }
    for key, value in expected.items():
        if declaration[key] != value:
            raise ValueError(f"sweep-v2 field source authority differs: {key}")
    if (
        manifest["terminal_status"] != "ACCEPTED"
        or summary["terminal_status"] != "ACCEPTED"
        or manifest["raw_results_payload_sha256"] != raw["integrity"]["payload_sha256"]
        or manifest["summary_payload_sha256"] != summary["integrity"]["payload_sha256"]
        or manifest["classification"] != declaration["classification"]
    ):
        raise ValueError("sweep-v2 bundle is not the accepted, sealed field source")
    cases = {case["case_id"]: case for case in raw["cases"]}
    if len(cases) != 96 or any(case["status"] != "success" for case in cases.values()):
        raise ValueError("sweep-v2 raw results must contain 96 successful cases")
    return SweepBinding(manifest, raw, summary, cases, manifest_sha, raw_sha, summary_sha)


def representative_case_ids(binding: SweepBinding) -> tuple[str, ...]:
    return tuple(sorted(item["case_id"] for item in binding.manifest["representative_artifacts"]))


def case_index(case_id: str) -> int:
    parts = case_id.split("-")
    if len(parts) != 5 or parts[:3] != ["l1a", "gs", "v2"]:
        raise ValueError(f"not a sweep-v2 case id: {case_id}")
    return int(parts[3])


# --------------------------------------------------------------------------
# Geometry → orbit configuration rule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignGeometry:
    case_id: str
    design_id: str
    wall_radius_m: float
    chamber_length_m: float
    injector_length_m: float
    exit_start_m: float
    exit_length_m: float
    exit_outer_radius_m: float
    dielectric_thickness_m: float
    stage_count: int
    stage_pitch_m: float
    stage_centers_m: tuple[float, ...]
    magnet_axial_thickness_m: float
    magnet_inner_radius_m: float
    magnet_outer_radius_m: float
    first_polarity: int
    has_divergent_exit: bool
    straight_wall_scope: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stage_centers_m"] = list(self.stage_centers_m)
        return value


def design_geometry(case: sweep.BuiltCase) -> DesignGeometry:
    chamber = case.geometry.chamber
    stages = case.geometry.stages
    derived = case.derived
    values = sweep.design_values(case.design)
    return DesignGeometry(
        case_id=case.case_id,
        design_id=case.design.design_id,
        wall_radius_m=float(chamber.outer_radius_m),
        chamber_length_m=float(chamber.length_m),
        injector_length_m=float(chamber.injector_length_m),
        exit_start_m=float(chamber.exit_start_m),
        exit_length_m=float(chamber.exit_length_m),
        exit_outer_radius_m=float(chamber.exit_outer_radius_m),
        dielectric_thickness_m=float(chamber.dielectric_thickness_m),
        stage_count=int(derived["stage_count"]),
        stage_pitch_m=float(derived["represented_stage_pitch_m"]),
        stage_centers_m=tuple(float(stage.center_z_m) for stage in stages),
        magnet_axial_thickness_m=float(derived["magnet_axial_thickness_m"]),
        magnet_inner_radius_m=float(derived["magnet_inner_radius_m"]),
        magnet_outer_radius_m=float(derived["magnet_outer_radius_m"]),
        first_polarity=1 if values["first_polarity_selector"] < 0.5 else -1,
        has_divergent_exit=bool(chamber.exit_length_m > 0.0),
        straight_wall_scope=(
            "straight_cylindrical_dielectric_section_only; radial exit for "
            "z > exit_start_m is domain_escape (divergent_section_radial)"
            if chamber.exit_length_m > 0.0
            else "full-length straight cylindrical dielectric (no divergent exit)"
        ),
    )


def orbit_config_for(
    geometry: DesignGeometry, rule: Mapping[str, Any], timestep_policy: Mapping[str, Any]
) -> OrbitConfig:
    """Design-dependent :class:`OrbitConfig` from the preregistered geometry rule."""

    max_path = float(rule["max_path_channel_lengths"]) * geometry.chamber_length_m
    slowest = math.sqrt(2.0 * float(rule["slowest_energy_ev"]) * EV_J / ELECTRON_MASS_KG)
    max_time = float(rule["max_time_transit_factor"]) * max_path / slowest
    return OrbitConfig(
        wall_radius_m=geometry.wall_radius_m,
        wall_z_min_m=float(rule["wall_z_min_m"]),
        wall_z_max_m=geometry.exit_start_m,
        domain_radius_m=geometry.wall_radius_m,
        domain_z_min_m=float(rule["domain_z_min_m"]),
        domain_z_max_m=geometry.chamber_length_m,
        max_time_s=max_time,
        max_path_m=max_path,
        max_steps=int(rule["max_steps"]),
        max_rotation_rad=float(timestep_policy["max_rotation_rad"]),
        event_tolerance_m=float(rule["event_tolerance_m"]),
        maximum_gamma=float(rule["maximum_gamma"]),
    )


def launch_cells(geometry: DesignGeometry, rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Four axial cells at fixed fractions of the channel-straight span."""

    z_low = geometry.injector_length_m
    z_high = geometry.exit_start_m
    if not z_high > z_low:
        raise ValueError(f"{geometry.case_id} has no straight channel span")
    cells = []
    for index, fraction in enumerate(rule["cell_fractions_of_straight_span"]):
        cells.append(
            {
                "cell_id": f"{rule['cell_id_prefix']}-{index + 1}",
                "fraction": float(fraction),
                "axial_center_m": z_low + float(fraction) * (z_high - z_low),
            }
        )
    return cells


def launch_positions(
    geometry: DesignGeometry, rule: Mapping[str, Any]
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    positions: list[tuple[str, tuple[float, float, float]]] = []
    for cell in launch_cells(geometry, rule):
        for fraction in rule["radius_fractions_of_wall"]:
            positions.append(
                (
                    f"{cell['cell_id']}-r{float(fraction):.3f}",
                    (float(fraction) * geometry.wall_radius_m, 0.0, cell["axial_center_m"]),
                )
            )
    return tuple(positions)


# --------------------------------------------------------------------------
# Field re-solve and bore grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BoreField:
    field: PsiBicubicField
    serialized: dict[str, Any]
    evidence: dict[str, Any]


def rebuild_case(binding: SweepBinding, case_id: str) -> sweep.BuiltCase:
    """Rebuild one sweep-v2 case and prove identity against the sealed raw record."""

    recorded = binding.cases_by_id[case_id]
    index = case_index(case_id)
    designs = sweep.sample_designs()
    design = designs[index]
    if design.design_id != recorded["design_id"]:
        raise ValueError(f"{case_id}: sampled design id differs from the sealed record")
    case = sweep.build_case(design, index)
    if case.case_id != case_id:
        raise ValueError(f"{case_id}: rebuilt case id differs")
    for key in ("geometry_sha256", "source_sha256", "config_sha256", "case_sha256"):
        if getattr(case, key) != recorded[key]:
            raise ValueError(f"{case_id}: rebuilt {key} differs from the sealed record")
    if sweep.design_values(design) != recorded["design_values"]:
        raise ValueError(f"{case_id}: design values differ from the sealed record")
    return case


def _within(left: float, right: float, tolerance: Mapping[str, float]) -> bool:
    return abs(left - right) <= tolerance["absolute"] + tolerance["relative"] * max(
        abs(left), abs(right)
    )


def verify_resolved_qois(
    case: sweep.BuiltCase, field: FieldMap, recorded: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-solved QoIs must reproduce the sealed record under the sweep replay tolerances."""

    tolerances = sweep.PROTOCOL["replay_contract"]["scale_aware_tolerances"]
    record = sweep.case_record(case, field)
    qois = record["qois"]
    recorded_qois = recorded["qois"]
    checks = {
        "centreline_mid_abs_bz_t": _within(
            qois["centreline_mid_abs_bz_t"],
            recorded_qois["centreline_mid_abs_bz_t"],
            tolerances["magnetic_field_t"],
        ),
        "centreline_abs_bz_peak_t": _within(
            qois["centreline_abs_bz_peak_t"],
            recorded_qois["centreline_abs_bz_peak_t"],
            tolerances["magnetic_field_t"],
        ),
        "field_peak_t": _within(
            qois["field_peak_t"], recorded_qois["field_peak_t"], tolerances["magnetic_field_t"]
        ),
        "minimum_mirror_ratio": _within(
            qois["minimum_mirror_ratio"],
            recorded_qois["minimum_mirror_ratio"],
            tolerances["dimensionless_qoi"],
        ),
        "stage_gradient_rms_t_per_m": _within(
            qois["stage_gradient_rms_t_per_m"],
            recorded_qois["stage_gradient_rms_t_per_m"],
            tolerances["gradient_t_per_m"],
        ),
        "field_energy_j": _within(
            qois["field_energy_j"], recorded_qois["field_energy_j"], tolerances["energy_j"]
        ),
        "axis_cusp_positions_m": qois["axis_cusp_positions_m"]
        == recorded_qois["axis_cusp_positions_m"],
        "axis_null_count": qois["axis_null_count"] == recorded_qois["axis_null_count"],
        "relative_residual_l2_within_gate": float(qois["relative_residual_l2"])
        <= float(sweep.PROTOCOL["terminal_acceptance"]["gates"][1]["limit"]),
        "geometry_sha256": record["geometry_sha256"] == recorded["geometry_sha256"],
        "case_sha256": record["case_sha256"] == recorded["case_sha256"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "resolved": {
            key: qois[key]
            for key in (
                "centreline_mid_abs_bz_t",
                "centreline_abs_bz_peak_t",
                "field_peak_t",
                "minimum_mirror_ratio",
                "maximum_mirror_ratio",
                "stage_gradient_rms_t_per_m",
                "field_energy_j",
                "axis_cusp_count",
                "axis_cusp_positions_m",
                "axis_null_count",
                "axis_null_positions_m",
                "boundary_to_peak_ratio",
                "topology_confidence",
                "relative_residual_l2",
            )
        },
        "recorded": {
            key: recorded_qois[key]
            for key in (
                "centreline_mid_abs_bz_t",
                "field_peak_t",
                "minimum_mirror_ratio",
                "field_energy_j",
                "axis_cusp_positions_m",
                "axis_null_positions_m",
            )
        },
        "backend": field.diagnostics.backend,
        "iterations": field.diagnostics.iterations,
        "recorded_backend": recorded["backend"],
    }


def verify_stored_representative(
    case_id: str, field: FieldMap, node_tolerance: Mapping[str, float]
) -> dict[str, Any] | None:
    """Node-wise agreement with the stored full-field artifact (representatives only)."""

    path = SWEEP_RESULTS / "representatives" / f"{case_id}.field-full.json"
    if not path.is_file():
        return None
    file_sha = sweep_protocol.verify_sidecar(path)
    artifact = sweep_protocol.strict_json(path)
    mapping = artifact["field_map"]
    if list(mapping["r_m"]) != list(field.r_m) or list(mapping["z_m"]) != list(field.z_m):
        raise ValueError(f"{case_id}: stored representative grid differs from the re-solve")
    psi_difference = float(np.max(np.abs(np.asarray(mapping["psi_wb"]) - np.asarray(field.psi_wb))))
    b_difference = float(
        max(
            np.max(np.abs(np.asarray(mapping["b_r_t"]) - np.asarray(field.b_r_t))),
            np.max(np.abs(np.asarray(mapping["b_z_t"]) - np.asarray(field.b_z_t))),
        )
    )
    checks = {
        "psi_nodes": psi_difference <= float(node_tolerance["psi_max_abs_wb"]),
        "b_nodes": b_difference <= float(node_tolerance["b_max_abs_t"]),
    }
    return {
        "path": path.relative_to(MODERN).as_posix(),
        "file_sha256": file_sha,
        "payload_sha256": artifact["integrity"]["payload_sha256"],
        "psi_max_abs_difference_wb": psi_difference,
        "b_max_abs_difference_t": b_difference,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _bore_indices(
    field: FieldMap, geometry: DesignGeometry, rule: Mapping[str, Any]
) -> tuple[int, int, int]:
    r = np.asarray(field.r_m)
    z = np.asarray(field.z_m)
    i_max = int(np.searchsorted(r, geometry.wall_radius_m, side="left"))
    if i_max >= len(r):
        raise ValueError(f"{geometry.case_id}: wall radius lies outside the field grid")
    z_low = float(rule["domain_z_min_m"])
    j0 = int(np.searchsorted(z, z_low, side="right") - 1)
    j1 = int(np.searchsorted(z, geometry.chamber_length_m, side="left"))
    if j0 < 0 or j1 >= len(z) or j1 - j0 < 3 or i_max < 3:
        raise ValueError(f"{geometry.case_id}: bore grid is too small for bicubic interpolation")
    return i_max, j0, j1


def bore_field(
    field: FieldMap,
    geometry: DesignGeometry,
    rule: Mapping[str, Any],
    *,
    role: str,
    source_identity_sha256: str,
    minimum_certificate_tightness_ratio: float,
) -> BoreField:
    """Restrict the accepted map to the channel bore and build the C1 ψ interpolant."""

    i_max, j0, j1 = _bore_indices(field, geometry, rule)
    r = np.asarray(field.r_m)[: i_max + 1]
    z = np.asarray(field.z_m)[j0 : j1 + 1]
    psi = np.asarray(field.psi_wb)[: i_max + 1, j0 : j1 + 1]
    br = np.asarray(field.b_r_t)[: i_max + 1, j0 : j1 + 1]
    bz = np.asarray(field.b_z_t)[: i_max + 1, j0 : j1 + 1]
    material = np.full(psi.shape, PLASMA_MATERIAL_ID, dtype=object)
    interpolant = PsiBicubicField(
        r,
        z,
        psi,
        material_id=material,
        plasma_material_id=PLASMA_MATERIAL_ID,
        reference_br_t=br,
        reference_bz_t=bz,
        minimum_certificate_tightness_ratio=minimum_certificate_tightness_ratio,
        source_identity_sha256=source_identity_sha256,
    )
    report = interpolant.reference_error().to_dict()
    serialized = {
        "role": role,
        "case_id": geometry.case_id,
        "r_m": r.tolist(),
        "z_m": z.tolist(),
        "psi_wb": psi.tolist(),
        "b_r_t": br.tolist(),
        "b_z_t": bz.tolist(),
        "material_id": PLASMA_MATERIAL_ID,
        "source_identity_sha256": source_identity_sha256,
    }
    evidence = {
        "role": role,
        "case_id": geometry.case_id,
        "source_identity_sha256": source_identity_sha256,
        "bore_grid": {
            "radial_samples": int(len(r)),
            "axial_samples": int(len(z)),
            "r_max_m": float(r[-1]),
            "z_min_m": float(z[0]),
            "z_max_m": float(z[-1]),
            "dr_m": float(r[1] - r[0]),
            "dz_m": float(z[1] - z[0]),
            "wall_radius_m": geometry.wall_radius_m,
            "radial_cells_across_bore": geometry.wall_radius_m / float(r[1] - r[0]),
        },
        "interpolation_error_report": report,
        "certificate": interpolant.certificate_tightness.to_dict(),
        "max_b_t": float(interpolant.max_b_t),
        "material_map_sha256": interpolant.material_map_sha256,
    }
    return BoreField(interpolant, serialized, evidence)


def refined_problem(case: sweep.BuiltCase, refinement: int) -> AxisymmetricProblem:
    domain = case.problem.domain
    refined = AxisymmetricDomain(
        domain.radius_m,
        domain.z_min_m,
        domain.z_max_m,
        int(domain.radial_intervals) * int(refinement),
        int(domain.axial_intervals) * int(refinement),
    )
    return AxisymmetricProblem(case.problem.name, refined, case.problem.sources)


def field_identity(case: sweep.BuiltCase, declaration: Mapping[str, Any], role: str) -> str:
    """Identity of a re-solved field: sweep case identity + solver inputs + role."""

    return content_hash(
        {
            "case_id": case.case_id,
            "case_sha256": case.case_sha256,
            "geometry_sha256": case.geometry_sha256,
            "source_sha256": case.source_sha256,
            "config_sha256": case.config_sha256,
            "role": role,
            "solver": "cft_revival.fields.solve_problem_cpu",
            "domain": dict(declaration["resolve"]["domain"])
            if role == "accepted"
            else {
                **dict(declaration["resolve"]["domain"]),
                "radial_intervals": int(declaration["resolve"]["domain"]["radial_intervals"])
                * int(declaration["refined_diagnostic"]["refinement"]),
                "axial_intervals": int(declaration["resolve"]["domain"]["axial_intervals"])
                * int(declaration["refined_diagnostic"]["refinement"]),
            },
            "solver_config": dict(declaration["resolve"]["solver_config"]),
            "sweep_protocol_payload_sha256": declaration["protocol_payload_sha256"],
        }
    )


@dataclass(frozen=True)
class ResolvedDesign:
    case_id: str
    case: sweep.BuiltCase
    geometry: DesignGeometry
    accepted: BoreField
    refined: BoreField | None
    evidence: dict[str, Any]


def resolve_design(
    binding: SweepBinding,
    case_id: str,
    protocol: Mapping[str, Any],
    *,
    include_refined: bool,
) -> ResolvedDesign:
    """Rebuild, re-solve, verify and restrict one design's field to its bore."""

    declaration = protocol["field_source"]
    rule = protocol["orbit_geometry_rule"]
    tightness = float(protocol["gates"]["minimum_certificate_dense_to_bound_ratio"])
    recorded = binding.cases_by_id[case_id]
    case = rebuild_case(binding, case_id)
    geometry = design_geometry(case)
    domain = case.problem.domain
    if (
        domain.radius_m != declaration["resolve"]["domain"]["radius_m"]
        or domain.z_min_m != declaration["resolve"]["domain"]["z_min_m"]
        or domain.z_max_m != declaration["resolve"]["domain"]["z_max_m"]
        or domain.radial_intervals != declaration["resolve"]["domain"]["radial_intervals"]
        or domain.axial_intervals != declaration["resolve"]["domain"]["axial_intervals"]
        or asdict(sweep.SOLVER) != dict(declaration["resolve"]["solver_config"])
    ):
        raise ValueError(f"{case_id}: sweep solver inputs differ from the declared authority")
    field = solve_problem_cpu(case.problem, sweep.SOLVER)
    qoi_report = verify_resolved_qois(case, field, recorded)
    if not qoi_report["passed"]:
        raise ValueError(f"{case_id}: re-solved QoIs differ from the sealed sweep record")
    stored = verify_stored_representative(
        case_id, field, declaration["resolve"]["stored_map_node_tolerance"]
    )
    if stored is not None and not stored["passed"]:
        raise ValueError(f"{case_id}: re-solved field differs from the stored representative map")
    accepted = bore_field(
        field,
        geometry,
        rule,
        role="accepted",
        source_identity_sha256=field_identity(case, declaration, "accepted"),
        minimum_certificate_tightness_ratio=tightness,
    )
    adapter_limits = declaration["adapter_gates"]
    refined: BoreField | None = None
    cross_resolution: dict[str, Any] | None = None
    if include_refined:
        refinement = int(declaration["refined_diagnostic"]["refinement"])
        refined_field = solve_problem_cpu(refined_problem(case, refinement), sweep.SOLVER)
        refined = bore_field(
            refined_field,
            geometry,
            rule,
            role="refined",
            source_identity_sha256=field_identity(case, declaration, "refined"),
            minimum_certificate_tightness_ratio=tightness,
        )
        cross_resolution = dict(compare_maps(accepted.field, refined.field))
        cross_resolution["b_relative_rms"] = float(cross_resolution["b_rms_t"]) / max(
            accepted.field.max_b_t, refined.field.max_b_t, float(np.finfo(float).tiny)
        )
    checks = {
        "identity_proven": True,
        "qois_reproduced": bool(qoi_report["passed"]),
        "stored_map_reproduced": True if stored is None else bool(stored["passed"]),
        "interpolation_b_relative_rms": (
            accepted.evidence["interpolation_error_report"]["b_relative_rms"]
            <= float(adapter_limits["maximum_b_relative_rms"])
        ),
        "interpolation_b_component_abs": (
            max(
                accepted.evidence["interpolation_error_report"]["br_max_abs_t"],
                accepted.evidence["interpolation_error_report"]["bz_max_abs_t"],
            )
            <= float(adapter_limits["maximum_b_component_absolute_error_t"])
        ),
        "cross_resolution_b_relative_rms": (
            True
            if cross_resolution is None
            else cross_resolution["b_relative_rms"]
            <= float(adapter_limits["maximum_cross_resolution_b_relative_rms"])
        ),
        "certificate_preflight": bool(accepted.field.certificate_tightness.preflight_passed),
    }
    evidence = {
        "case_id": case_id,
        "design_id": case.design.design_id,
        "design_values": sweep.design_values(case.design),
        "geometry": geometry.to_dict(),
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "case_sha256": case.case_sha256,
        "sweep_record": {
            "backend": recorded["backend"],
            "iterations": recorded["iterations"],
            "axis_cusp_positions_m": list(recorded["qois"]["axis_cusp_positions_m"]),
            "axis_null_positions_m": list(recorded["qois"]["axis_null_positions_m"]),
            "centreline_mid_abs_bz_t": recorded["qois"]["centreline_mid_abs_bz_t"],
            "centreline_abs_bz_peak_t": recorded["qois"]["centreline_abs_bz_peak_t"],
            "minimum_mirror_ratio": recorded["qois"]["minimum_mirror_ratio"],
            "maximum_mirror_ratio": recorded["qois"]["maximum_mirror_ratio"],
            "stage_gradient_rms_t_per_m": recorded["qois"]["stage_gradient_rms_t_per_m"],
            "field_energy_j": recorded["qois"]["field_energy_j"],
            "field_peak_t": recorded["qois"]["field_peak_t"],
            "boundary_to_peak_ratio": recorded["qois"]["boundary_to_peak_ratio"],
            "topology_confidence": recorded["qois"]["topology_confidence"],
        },
        "resolve": {
            "solver": "cft_revival.fields.solve_problem_cpu",
            "backend": field.diagnostics.backend,
            "iterations": field.diagnostics.iterations,
            "relative_residual_l2": field.diagnostics.relative_residual_l2,
            "converged": field.diagnostics.converged,
            "qoi_replay": qoi_report,
            "stored_representative": stored,
        },
        "accepted_bore_field": accepted.evidence,
        "refined_bore_field": None if refined is None else refined.evidence,
        "cross_resolution": cross_resolution,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return ResolvedDesign(case_id, case, geometry, accepted, refined, evidence)


def design_field_identities(
    binding: SweepBinding, case_ids: Sequence[str], declaration: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Field identities for every design without solving anything (used by prepare)."""

    output: dict[str, dict[str, str]] = {}
    for case_id in case_ids:
        case = rebuild_case(binding, case_id)
        output[case_id] = {
            "accepted": field_identity(case, declaration, "accepted"),
            "refined": field_identity(case, declaration, "refined"),
        }
    return output
