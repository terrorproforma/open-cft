"""Publication gates recomputed from immutable, hash-bound raw solver runs."""

from __future__ import annotations

import hashlib
import json
import base64
import struct
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from math import fsum, hypot, isclose, isfinite
from typing import Any

from .models import MaterialFieldResult, MaterialFieldValidationError
from .numerics import assemble_rhs
from .numerics import _implementation_sha256


def _evidence_implementation_sha256() -> str:
    return _implementation_sha256(
        "acceptance.py",
        "adapters.py",
        "artifacts.py",
        "models.py",
        "numerics.py",
        "replay.py",
        "warp_solver.py",
    )

WARNING_CODES = (
    "CELLWISE_MAXIMUM_SCREENING_ONLY",
    "HOST_MEMORY_LIMITED_QUALIFICATION",
    "LINEAR_IRON_SATURATION_UNASSESSED",
    "PM_IRREVERSIBLE_DEMAGNETIZATION_UNASSESSED",
    "UNDERRESOLVED_GEOMETRIC_FEATURE",
    "STRUCTURED_GRID_METHOD_INSUFFICIENT",
)
GATE_LIMITS = {
    "true_equation_residual": 1.0e-8,
    "energy_balance": 1.0e-8,
    "current_balance": 1.0e-12,
    "weak_source_action": 1.0e-3,
    "minimum_effective_feature_cells": 12.0,
    "mesh_fixed_qoi": 1.0e-2,
    "alignment_fixed_qoi": 1.0e-2,
    "pm_form_discrepancy_convergence": 2.0e-2,
    "backend_parity": 1.0e-8,
    "structured_observed_order": 1.5,
}
_REPLAY_CACHE: OrderedDict[str, object] = OrderedDict()


def _replay_cached(run: "RawRunObservation"):
    cached = _REPLAY_CACHE.get(run.run_sha256)
    if cached is not None:
        _REPLAY_CACHE.move_to_end(run.run_sha256)
        return cached
    from .replay import replay_raw_run

    report = replay_raw_run(run.raw, backend=run.backend)
    _REPLAY_CACHE[run.run_sha256] = report
    # Replay reports contain full-resolution Br/Bz matrices, so retain only
    # the immediately reused report and keep high-resolution validation
    # bounded in host memory.
    while len(_REPLAY_CACHE) > 1:
        _REPLAY_CACHE.popitem(last=False)
    return report


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _solver_config_identity(run_config_json: str) -> str:
    payload = dict(json.loads(run_config_json)["config"])
    payload.pop("allow_underresolved_screening", None)
    return _sha(payload)


def _array_sha(*arrays) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(struct.pack("<Q", len(values)))
        digest.update(struct.pack(f"<{len(values)}d", *values))
    return digest.hexdigest()


def _encoded_solution(values: tuple[float, ...]) -> dict[str, object]:
    binary = struct.pack(f"<{len(values)}d", *values)
    return {
        "codec": "zlib-base64",
        "dtype": "float64-little-endian",
        "layout": "radial-major",
        "count": len(values),
        "uncompressed_sha256": hashlib.sha256(binary).hexdigest(),
        "data_base64": base64.b64encode(zlib.compress(binary, level=9)).decode("ascii"),
    }


def _study_identity_hashes(problem_payload: dict[str, object]) -> tuple[str, str]:
    geometry_identity = json.loads(problem_payload["geometry_bundle_json"])
    geometry_identity.pop("permanent_magnet_plan", None)
    geometry_identity.pop("integrity", None)
    handoff_envelope = json.loads(problem_payload["magnetics_bundle_json"])
    materials = sorted(
        (
            material
            for material in handoff_envelope["content"]["materials"]
            if not material["material_id"].endswith("-equivalent-host")
        ),
        key=lambda item: item["material_id"],
    )
    return _sha(geometry_identity), _sha(materials)


def _relative(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-300)


def _interpolate(result: MaterialFieldResult, r: float, z: float) -> float:
    domain = result.problem.domain
    x = min(max(r / domain.dr_m, 0.0), domain.radial_intervals)
    y = min(max((z - domain.z_min_m) / domain.dz_m, 0.0), domain.axial_intervals)
    i0, j0 = min(int(x), domain.radial_intervals - 1), min(int(y), domain.axial_intervals - 1)
    fr, fz = x - i0, y - j0
    values = result.field.b_z_t
    return (
        (1.0 - fr) * (1.0 - fz) * values[i0][j0]
        + fr * (1.0 - fz) * values[i0 + 1][j0]
        + (1.0 - fr) * fz * values[i0][j0 + 1]
        + fr * fz * values[i0 + 1][j0 + 1]
    )


def _problem_payload(result: MaterialFieldResult) -> dict[str, object]:
    problem = result.problem
    return {
        "problem_id": problem.problem_id,
        "geometry_schema_version": problem.geometry_schema_version,
        "geometry_sha256": problem.geometry_sha256,
        "magnetics_sha256": problem.magnetics_sha256,
        "authority": problem.authority,
        "coefficient_sha256": _array_sha(
            problem.radial_face_reluctivity_per_m_h,
            problem.axial_face_reluctivity_per_m_h,
            problem.remanence_g_r_face_a_per_m,
            problem.remanence_g_z_face_a_per_m,
        ),
        "source_sha256": _array_sha(assemble_rhs(problem)),
        "outer_boundary_kind": problem.outer_boundary_kind,
        "geometry_bundle_json": problem.geometry_bundle_json,
        "magnetics_bundle_json": problem.magnetics_bundle_json,
        "raster_diagnostics": [asdict(item) for item in problem.raster_diagnostics],
        "weak_action_diagnostics": [asdict(item) for item in problem.weak_action_diagnostics],
        "source_envelope_m": list(problem.source_envelope_m),
        "feature_effective_cells": [list(item) for item in problem.feature_effective_cells],
        "qoi_locations_rz_m": [list(item) for item in problem.qoi_locations_rz_m],
        "qoi_bore_windows_m": [list(item) for item in problem.qoi_bore_windows_m],
        "open_boundary_policy": dict(problem.open_boundary_policy),
        "counts": {
            "material_regions": problem.authoritative_material_region_count,
            "pm_regions": problem.pm_region_count,
            "free_current_sources": problem.authoritative_free_current_source_count,
            "interfaces": problem.handoff_interface_count,
        },
    }


@dataclass(frozen=True, slots=True)
class RawRunObservation:
    study_id: str
    role: str
    run_sha256: str
    config_sha256: str
    solver_config_identity_sha256: str
    geometry_sha256: str
    material_sha256: str
    design_geometry_sha256: str
    material_registry_sha256: str
    implementation_sha256: str
    evidence_implementation_sha256: str
    backend: str
    grid_sha256: str
    domain_sha256: str
    problem_sha256: str
    raw: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def raw_run_observation(
    result: MaterialFieldResult, *, study_id: str, role: str
) -> RawRunObservation:
    domain = result.problem.domain
    domain_payload = {
        "radius_m": domain.radius_m,
        "z_min_m": domain.z_min_m,
        "z_max_m": domain.z_max_m,
    }
    grid_payload = {
        **domain_payload,
        "radial_intervals": domain.radial_intervals,
        "axial_intervals": domain.axial_intervals,
        "dr_m": domain.dr_m,
        "dz_m": domain.dz_m,
    }
    solver_psi = [list(row) for row in result.field.psi_wb]
    if result.problem.outer_boundary_kind == "dipole_robin_psi":
        for row in solver_psi:
            row[0] = 0.0
            row[-1] = 0.0
        solver_psi[-1] = [0.0] * len(solver_psi[-1])
    flat_solver_psi = tuple(value for row in solver_psi for value in row)
    raw: dict[str, object] = {
        "domain": grid_payload,
        "solution": _encoded_solution(flat_solver_psi),
        "diagnostics": asdict(result.diagnostics),
        "problem": _problem_payload(result),
    }
    problem_sha = _sha(raw["problem"])
    design_geometry_sha, material_registry_sha = _study_identity_hashes(
        raw["problem"]
    )
    anchors = {
        "study_id": study_id,
        "role": role,
        "config_sha256": result.diagnostics.run_config_sha256,
        "solver_config_identity_sha256": _solver_config_identity(
            result.diagnostics.run_config_json
        ),
        "geometry_sha256": result.problem.geometry_sha256,
        "material_sha256": result.problem.magnetics_sha256,
        "design_geometry_sha256": design_geometry_sha,
        "material_registry_sha256": material_registry_sha,
        "implementation_sha256": result.diagnostics.implementation_sha256,
        "evidence_implementation_sha256": _evidence_implementation_sha256(),
        "backend": result.diagnostics.backend,
        "grid_sha256": _sha(grid_payload),
        "domain_sha256": _sha(domain_payload),
        "problem_sha256": problem_sha,
        "raw": raw,
    }
    return RawRunObservation(
        study_id,
        role,
        _sha(anchors),
        result.diagnostics.run_config_sha256,
        _solver_config_identity(result.diagnostics.run_config_json),
        result.problem.geometry_sha256,
        result.problem.magnetics_sha256,
        design_geometry_sha,
        material_registry_sha,
        result.diagnostics.implementation_sha256,
        _evidence_implementation_sha256(),
        result.diagnostics.backend,
        _sha(grid_payload),
        _sha(domain_payload),
        problem_sha,
        raw,
    )


def _metrics(run: RawRunObservation) -> dict[str, object]:
    raw, problem = run.raw, run.raw["problem"]
    domain = raw["domain"]
    replay = _replay_cached(run)
    if not replay.passed:
        raise MaterialFieldValidationError("run metrics require successful replay")
    br, bz = replay.b_r_t, replay.b_z_t
    nr, nz = len(br), len(br[0])
    magnitudes = [hypot(br[i][j], bz[i][j]) for i in range(nr) for j in range(nz)]
    boundary = [
        hypot(br[i][j], bz[i][j])
        for i in range(nr)
        for j in range(nz)
        if i == nr - 1 or j in (0, nz - 1)
    ]
    peak, boundary_peak = max(magnitudes), max(boundary)
    envelope = problem["source_envelope_m"]
    padding = min(
        domain["radius_m"] - envelope[0],
        envelope[1] - domain["z_min_m"],
        domain["z_max_m"] - envelope[2],
    ) / envelope[3]
    qois = {
        item[0]: _interpolate_matrix(domain, bz, float(item[1]), float(item[2]))
        for item in problem["qoi_locations_rz_m"]
    }
    for name, radius, z_min, z_max in problem["qoi_bore_windows_m"]:
        qois[name] = _bore_average(
            domain, bz, float(radius), float(z_min), float(z_max)
        )
    return {
        "study_id": run.study_id,
        "role": run.role,
        "run_sha256": run.run_sha256,
        "domain_sha256": run.domain_sha256,
        "grid_sha256": run.grid_sha256,
        "padding_characteristic_lengths": padding,
        "boundary_b_max_t": boundary_peak,
        "interior_cellwise_max_t": peak,
        "boundary_to_peak_ratio": boundary_peak / peak,
        "fixed_qois_bz_t": qois,
        "minimum_effective_feature_cells": min(
            min(float(item[1]), float(item[2]))
            for item in problem["feature_effective_cells"]
        ),
    }


def _interpolate_matrix(
    domain: dict[str, Any], values, r: float, z: float
) -> float:
    x = min(max(r / domain["dr_m"], 0.0), domain["radial_intervals"])
    y = min(max((z - domain["z_min_m"]) / domain["dz_m"], 0.0), domain["axial_intervals"])
    i0 = min(int(x), domain["radial_intervals"] - 1)
    j0 = min(int(y), domain["axial_intervals"] - 1)
    fr, fz = x - i0, y - j0
    return (
        (1 - fr) * (1 - fz) * values[i0][j0]
        + fr * (1 - fz) * values[i0 + 1][j0]
        + (1 - fr) * fz * values[i0][j0 + 1]
        + fr * fz * values[i0 + 1][j0 + 1]
    )


_GAUSS_2 = (
    (-0.5773502691896258, 1.0),
    (0.5773502691896258, 1.0),
)


def _bore_average(
    domain: dict[str, Any],
    values,
    radius: float,
    z_min: float,
    z_max: float,
) -> float:
    """Composite cell-intersection Gauss average of the bilinear field."""
    numerator = 0.0
    denominator = 0.0
    dr, dz = float(domain["dr_m"]), float(domain["dz_m"])
    radial_intervals = int(domain["radial_intervals"])
    axial_intervals = int(domain["axial_intervals"])
    for i in range(min(radial_intervals, int(radius / dr) + 1)):
        cell_r0, cell_r1 = i * dr, min((i + 1) * dr, radius)
        if cell_r1 <= cell_r0:
            continue
        first_j = max(0, int((z_min - float(domain["z_min_m"])) / dz))
        last_j = min(
            axial_intervals - 1,
            int((z_max - float(domain["z_min_m"])) / dz),
        )
        for j in range(first_j, last_j + 1):
            grid_z0 = float(domain["z_min_m"]) + j * dz
            cell_z0, cell_z1 = max(grid_z0, z_min), min(grid_z0 + dz, z_max)
            if cell_z1 <= cell_z0:
                continue
            jacobian = 0.25 * (cell_r1 - cell_r0) * (cell_z1 - cell_z0)
            for radial_node, radial_weight in _GAUSS_2:
                radial = 0.5 * (
                    (cell_r1 - cell_r0) * radial_node + cell_r1 + cell_r0
                )
                for axial_node, axial_weight in _GAUSS_2:
                    axial = 0.5 * (
                        (cell_z1 - cell_z0) * axial_node + cell_z1 + cell_z0
                    )
                    weight = radial_weight * axial_weight * jacobian * radial
                    numerator += weight * _interpolate_matrix(
                        domain, values, radial, axial
                    )
                    denominator += weight
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class PublicationEvidence:
    status: str
    qualification: dict[str, object]
    raw_runs: tuple[RawRunObservation, ...]
    studies: tuple[dict[str, object], ...]
    gates: tuple[tuple[str, str, float, float, str], ...]
    warning_codes: tuple[str, ...]
    pm_model_form_comparison: dict[str, float]
    structured_convergence: dict[str, object]
    backend_parity: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "qualification": dict(self.qualification),
            "raw_runs": [item.to_dict() for item in self.raw_runs],
            "studies": [dict(item) for item in self.studies],
            "gates": [
                {
                    "gate_id": key,
                    "status": status,
                    "measured_value": measured_value,
                    "threshold": threshold,
                    "diagnostic_status": diagnostic_status,
                }
                for (
                    key,
                    status,
                    measured_value,
                    threshold,
                    diagnostic_status,
                ) in self.gates
            ],
            "warning_codes": list(self.warning_codes),
            "pm_model_form_comparison": dict(self.pm_model_form_comparison),
            "structured_convergence": dict(self.structured_convergence),
            "backend_parity": dict(self.backend_parity),
        }


def _max_qoi_change(left: dict[str, object], right: dict[str, object]) -> float:
    lq, rq = left["fixed_qois_bz_t"], right["fixed_qois_bz_t"]
    if set(lq) != set(rq) or not lq:
        raise MaterialFieldValidationError("fixed QoI sets must be identical and nonempty")
    return max(_relative(float(lq[key]), float(rq[key])) for key in lq)


def _source_error(base: MaterialFieldResult) -> float:
    problem = base.problem
    diagnostics = problem.raster_diagnostics
    expected = problem.authoritative_material_region_count + problem.pm_region_count
    if (
        not diagnostics
        or len(diagnostics) != expected
        or len({item.item_id for item in diagnostics}) != len(diagnostics)
        or not problem.geometry_region_provenance
    ):
        return 1.0e300
    return max(abs(item.relative_source_error) for item in diagnostics)


def _observed_order(
    h0: float, h1: float, h2: float, q0: float, q1: float, q2: float
) -> float:
    d01, d12 = q0 - q1, q1 - q2
    if d01 * d12 <= 0.0 or d12 == 0.0:
        return 0.0
    target = abs(d01 / d12)

    def ratio(order: float) -> float:
        return (h0**order - h1**order) / (h1**order - h2**order)

    lower, upper = 0.05, 8.0
    if not min(ratio(lower), ratio(upper)) <= target <= max(
        ratio(lower), ratio(upper)
    ):
        return 0.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if (ratio(middle) < target) == (ratio(lower) < target):
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _structured_diagnostics(
    base: dict[str, object],
    fine: dict[str, object],
    third: dict[str, object],
    run_by_id: dict[str, RawRunObservation],
) -> dict[str, object]:
    runs = [run_by_id[item["study_id"]] for item in (base, fine, third)]
    spacings = [
        max(float(run.raw["domain"]["dr_m"]), float(run.raw["domain"]["dz_m"]))
        for run in runs
    ]
    qois: dict[str, object] = {}
    for key in base["fixed_qois_bz_t"]:
        values = [
            float(item["fixed_qois_bz_t"][key]) for item in (base, fine, third)
        ]
        order = _observed_order(*spacings, *values)
        refinement = spacings[1] / spacings[2]
        richardson = (
            values[2] + (values[2] - values[1]) / (refinement**order - 1.0)
            if order > 0.0 and refinement**order != 1.0
            else values[2]
        )
        qois[key] = {
            "observed_order": order,
            "richardson_bz_t": richardson,
            "values_bz_t": values,
        }
    minimum_order = min(
        float(item["observed_order"]) for item in qois.values()
    )
    return {
        "minimum_observed_order": minimum_order,
        "method_assessment": (
            "STRUCTURED_GRID_ADEQUATE"
            if minimum_order >= GATE_LIMITS["structured_observed_order"]
            else "STRUCTURED_GRID_L1B_INSUFFICIENT"
        ),
        "qois": qois,
    }


def _backend_parity_metrics(
    cpu_run: RawRunObservation, cuda_run: RawRunObservation
) -> dict[str, object]:
    cpu, cuda = _replay_cached(cpu_run), _replay_cached(cuda_run)
    numerator = 0.0
    denominator = 0.0
    for left_rows, right_rows in (
        (cpu.b_r_t, cuda.b_r_t),
        (cpu.b_z_t, cuda.b_z_t),
    ):
        for left_row, right_row in zip(left_rows, right_rows):
            for left, right in zip(left_row, right_row):
                numerator += (left - right) ** 2
                denominator += left * left
    relative_l2 = (numerator / max(denominator, 1.0e-300)) ** 0.5
    return {
        "cpu_run_sha256": cpu_run.run_sha256,
        "cuda_run_sha256": cuda_run.run_sha256,
        "relative_field_l2": relative_l2,
    }


def _validate_qualification(
    value: object, *, base_grid: list[int]
) -> dict[str, object]:
    keys = {
        "schema_version",
        "study_scope",
        "status",
        "reason_code",
        "required_role_count",
        "completed_role_count",
        "not_evaluated_roles",
        "requested_base_grid",
        "executed_base_grid",
        "estimated_requested_raster_bytes",
        "safe_raster_bytes",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise MaterialFieldValidationError("qualification contract is not closed")
    if value["schema_version"] != "cft_revival.material_fields.qualification/1.4.0":
        raise MaterialFieldValidationError("qualification schema is unsupported")
    if (
        value["required_role_count"] != 10
        or value["completed_role_count"] != 10
        or value["not_evaluated_roles"] != []
        or value["executed_base_grid"] != base_grid
    ):
        raise MaterialFieldValidationError(
            "qualification role/grid cardinality is invalid"
        )
    for key in ("requested_base_grid", "executed_base_grid"):
        grid = value[key]
        if (
            not isinstance(grid, list)
            or len(grid) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 4
                for item in grid
            )
        ):
            raise MaterialFieldValidationError("qualification grid is invalid")
    for key in ("estimated_requested_raster_bytes", "safe_raster_bytes"):
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] < 1
        ):
            raise MaterialFieldValidationError(
                "qualification memory bound is invalid"
            )
    memory_limited = (
        value["study_scope"] == "MEMORY_LIMITED_REDUCED_SCREENING"
    )
    if memory_limited:
        if (
            value["status"] != "NOT_EVALUATED"
            or value["reason_code"] != "HOST_MEMORY_LIMIT"
            or value["estimated_requested_raster_bytes"]
            <= value["safe_raster_bytes"]
            or value["requested_base_grid"] == value["executed_base_grid"]
        ):
            raise MaterialFieldValidationError(
                "memory-limited qualification must fail closed"
            )
    elif (
        value["study_scope"] != "PREREGISTERED_HIGH_RESOLUTION"
        or value["status"] != "EVALUATED"
        or value["reason_code"] != "NONE"
        or value["requested_base_grid"] != value["executed_base_grid"]
    ):
        raise MaterialFieldValidationError(
            "qualification status is inconsistent"
        )
    return dict(value)


def _publication_gate(
    gate_id: str,
    passed: bool,
    measured_value: float,
    threshold: float,
    *,
    high_resolution_evaluated: bool,
) -> tuple[str, str, float, float, str]:
    return (
        gate_id,
        (
            "PASS"
            if high_resolution_evaluated and passed
            else "FAIL"
            if high_resolution_evaluated
            else "NOT_EVALUATED"
        ),
        measured_value,
        threshold,
        (
            "MEASURED_HIGH_RESOLUTION"
            if high_resolution_evaluated
            else "MEASURED_REDUCED_RESOURCE_ONLY"
        ),
    )


def assess_publication(
    base: MaterialFieldResult | RawRunObservation,
    *,
    domain_expansions: tuple[MaterialFieldResult | RawRunObservation, ...],
    mesh_fine: MaterialFieldResult | RawRunObservation,
    mesh_third: MaterialFieldResult | RawRunObservation,
    alignment_sweeps: tuple[MaterialFieldResult | RawRunObservation, ...] = (),
    equivalent_base: MaterialFieldResult | RawRunObservation | None = None,
    equivalent_fine: MaterialFieldResult | RawRunObservation | None = None,
    parity_cpu: MaterialFieldResult | RawRunObservation | None = None,
    parity_cuda: MaterialFieldResult | RawRunObservation | None = None,
    qualification: dict[str, object] | None = None,
) -> PublicationEvidence:
    """Derive all status, warnings, extents, hashes, and gates from raw runs."""

    raw_mode = isinstance(base, RawRunObservation)
    policy = dict(
        base.raw["problem"]["open_boundary_policy"]
        if raw_mode
        else base.problem.open_boundary_policy
    )
    required = int(policy["required_expansion_comparisons"])
    if len(domain_expansions) != required or len(alignment_sweeps) != 1:
        raise MaterialFieldValidationError(
            "publication requires exact domain/alignment study cardinality"
        )
    if any(
        item is None
        for item in (equivalent_base, equivalent_fine, parity_cpu, parity_cuda)
    ):
        raise MaterialFieldValidationError(
            "publication requires both PM-form and CPU/CUDA parity pairs"
        )
    model_results = (equivalent_base, equivalent_fine)
    parity_results = (parity_cpu, parity_cuda)
    results = (
        base,
        *domain_expansions,
        mesh_fine,
        mesh_third,
        *alignment_sweeps,
        *model_results,
        *parity_results,
    )
    if any(
        (
            item.geometry_sha256 if raw_mode else item.problem.geometry_sha256
        )
        != (base.geometry_sha256 if raw_mode else base.problem.geometry_sha256)
        for item in (
            base,
            *domain_expansions,
            mesh_fine,
            mesh_third,
            *alignment_sweeps,
            parity_cpu,
            parity_cuda,
        )
    ):
        raise MaterialFieldValidationError("study geometry hashes differ")
    roles = (
        ("base", "base"),
        *((f"domain-{i}", "domain_expansion") for i in range(1, len(domain_expansions) + 1)),
        ("mesh-fine", "mesh_refinement"),
        ("mesh-third", "mesh_refinement_2"),
        *((f"alignment-{i}", "grid_alignment") for i in range(1, len(alignment_sweeps) + 1)),
        ("equivalent-base", "model_form"),
        ("equivalent-fine", "model_form"),
        ("parity-cpu", "backend_parity_cpu"),
        ("parity-cuda", "backend_parity_cuda"),
    )
    raw_runs = tuple(
        (
            replace(result, study_id=identity, role=role, run_sha256="")
            if raw_mode
            else raw_run_observation(result, study_id=identity, role=role)
        )
        for result, (identity, role) in zip(results, roles)
    )
    if raw_mode:
        raw_runs = tuple(
            replace(
                run,
                run_sha256=_sha(
                    {
                        key: value
                        for key, value in run.to_dict().items()
                        if key != "run_sha256"
                    }
                ),
            )
            for run in raw_runs
        )
    base_grid = [
        int(raw_runs[0].raw["domain"]["radial_intervals"]),
        int(raw_runs[0].raw["domain"]["axial_intervals"]),
    ]
    if qualification is None:
        estimated = (base_grid[0] + 1) * (base_grid[1] + 1) * 2048
        qualification = {
            "schema_version": "cft_revival.material_fields.qualification/1.4.0",
            "study_scope": "PREREGISTERED_HIGH_RESOLUTION",
            "status": "EVALUATED",
            "reason_code": "NONE",
            "required_role_count": len(raw_runs),
            "completed_role_count": len(raw_runs),
            "not_evaluated_roles": [],
            "requested_base_grid": base_grid,
            "executed_base_grid": base_grid,
            "estimated_requested_raster_bytes": estimated,
            "safe_raster_bytes": estimated,
        }
    qualification = _validate_qualification(
        qualification, base_grid=base_grid
    )
    common_identity = {
        (
            run.design_geometry_sha256,
            run.material_registry_sha256,
            run.solver_config_identity_sha256,
            run.evidence_implementation_sha256,
        )
        for run in raw_runs
    }
    if len(common_identity) != 1:
        raise MaterialFieldValidationError(
            "all publication studies require one geometry/material/config/evidence-code identity"
        )
    main_runs = raw_runs[:-2]
    if (
        len({run.backend for run in main_runs}) != 1
        or not raw_runs[-2].backend.endswith("python")
        or "warp:cuda:" not in raw_runs[-1].backend
    ):
        raise MaterialFieldValidationError(
            "main studies require one backend and parity requires CPU then CUDA"
        )
    if len({item.run_sha256 for item in raw_runs}) != len(raw_runs):
        raise MaterialFieldValidationError("study runs must be distinct")
    studies = tuple(_metrics(run) for run in raw_runs)
    domains = tuple(item for item in studies if item["role"] in {"base", "domain_expansion"})
    domain_runs = tuple(item for item in raw_runs if item.role in {"base", "domain_expansion"})
    raw_domains = [item.raw["domain"] for item in domain_runs]
    base_domain = raw_domains[0]
    for left, right in zip(raw_domains, raw_domains[1:]):
        if not (
            right["radius_m"] > left["radius_m"]
            and right["z_min_m"] < left["z_min_m"]
            and right["z_max_m"] > left["z_max_m"]
        ):
            raise MaterialFieldValidationError("domain expansions must strictly contain predecessors")
        radial_cells = right["radius_m"] / base_domain["dr_m"]
        axial_offset = (base_domain["z_min_m"] - right["z_min_m"]) / base_domain["dz_m"]
        if (
            not isclose(right["dr_m"], base_domain["dr_m"], rel_tol=1.0e-12)
            or not isclose(right["dz_m"], base_domain["dz_m"], rel_tol=1.0e-12)
            or abs(radial_cells - round(radial_cells)) > 1.0e-9
            or abs(axial_offset - round(axial_offset)) > 1.0e-9
        ):
            raise MaterialFieldValidationError(
                "domain expansions must preserve phase-locked spacing and coordinates"
            )
    expansion_ratios = [
        float(right["padding_characteristic_lengths"])
        / float(left["padding_characteristic_lengths"])
        for left, right in zip(domains, domains[1:])
    ]
    domain_qoi_changes = [_max_qoi_change(left, right) for left, right in zip(domains, domains[1:])]
    base_metrics = next(item for item in studies if item["role"] == "base")
    fine_metrics = next(item for item in studies if item["role"] == "mesh_refinement")
    third_metrics = next(item for item in studies if item["role"] == "mesh_refinement_2")
    mesh_change = max(
        _max_qoi_change(base_metrics, fine_metrics),
        _max_qoi_change(fine_metrics, third_metrics),
    )
    alignment_metrics = tuple(item for item in studies if item["role"] == "grid_alignment")
    alignment_change = max(
        (_max_qoi_change(base_metrics, item) for item in alignment_metrics), default=1.0e300
    )
    base_problem_payload = raw_runs[0].raw["problem"]
    base_diagnostics = raw_runs[0].raw["diagnostics"]
    weak_error = max(
        (
            float(item["relative_bias"])
            for item in base_problem_payload["weak_action_diagnostics"]
            if float(item["absolute_bias_a"]) > 1.0e-8
        ),
        default=0.0,
    )
    raster = base_problem_payload["raster_diagnostics"]
    counts = base_problem_payload["counts"]
    complete = (
        len(raster) == counts["material_regions"] + counts["pm_regions"]
        and len({item["item_id"] for item in raster}) == len(raster)
    )
    source_error = (
        max(abs(float(item["relative_source_error"])) for item in raster)
        if complete and raster
        else 1.0e300
    )
    min_cells = min(
        float(item["minimum_effective_feature_cells"])
        for item in studies
        if not str(item["role"]).startswith("backend_parity")
    )
    run_by_id = {run.study_id: run for run in raw_runs}
    structured = _structured_diagnostics(
        base_metrics, fine_metrics, third_metrics, run_by_id
    )
    parity = _backend_parity_metrics(raw_runs[-2], raw_runs[-1])
    comparison = {
        "base_fixed_qoi_relative_difference": 1.0e300,
        "fine_fixed_qoi_relative_difference": 1.0e300,
        "discrepancy_change": 1.0e300,
    }
    if equivalent_base is not None and equivalent_fine is not None:
        recoil_base = base_metrics
        recoil_fine = fine_metrics
        model_metrics = tuple(item for item in studies if item["role"] == "model_form")
        eq_base, eq_fine = model_metrics
        base_gap = _max_qoi_change(recoil_base, eq_base)
        fine_gap = _max_qoi_change(recoil_fine, eq_fine)
        comparison = {
            "base_fixed_qoi_relative_difference": base_gap,
            "fine_fixed_qoi_relative_difference": fine_gap,
            "discrepancy_change": _relative(base_gap, fine_gap),
        }
    evaluated_gates = (
        (
            "high_resolution_qualification",
            qualification["status"] == "EVALUATED",
            1.0 if qualification["status"] == "EVALUATED" else 0.0,
            1.0,
        ),
        ("required_domain_expansions", len(domain_expansions) >= required, float(len(domain_expansions)), float(required)),
        ("minimum_base_padding", float(domains[0]["padding_characteristic_lengths"]) >= float(policy["minimum_padding_characteristic_lengths"]), float(domains[0]["padding_characteristic_lengths"]), float(policy["minimum_padding_characteristic_lengths"])),
        ("domain_expansion_factor", min(expansion_ratios) >= float(policy["domain_expansion_factor"]) * (1.0 - 1.0e-12), min(expansion_ratios), float(policy["domain_expansion_factor"])),
        ("domain_phase_lock", True, 1.0, 1.0),
        ("boundary_field_ratio", max(float(item["boundary_to_peak_ratio"]) for item in domains) <= float(policy["maximum_boundary_to_peak_field_ratio"]), max(float(item["boundary_to_peak_ratio"]) for item in domains), float(policy["maximum_boundary_to_peak_field_ratio"])),
        ("successive_fixed_qoi", max(domain_qoi_changes[-required:]) <= float(policy["maximum_qoi_relative_change"]), max(domain_qoi_changes[-required:]), float(policy["maximum_qoi_relative_change"])),
        ("mesh_fixed_qoi", mesh_change <= GATE_LIMITS["mesh_fixed_qoi"], mesh_change, GATE_LIMITS["mesh_fixed_qoi"]),
        ("alignment_fixed_qoi", bool(alignment_metrics) and alignment_change <= GATE_LIMITS["alignment_fixed_qoi"], alignment_change, GATE_LIMITS["alignment_fixed_qoi"]),
        ("minimum_effective_feature_cells", min_cells >= GATE_LIMITS["minimum_effective_feature_cells"], min_cells, GATE_LIMITS["minimum_effective_feature_cells"]),
        ("true_equation_residual", bool(base_diagnostics["converged"]) and float(base_diagnostics["relative_true_residual_l2"]) <= GATE_LIMITS["true_equation_residual"], float(base_diagnostics["relative_true_residual_l2"]), GATE_LIMITS["true_equation_residual"]),
        ("energy_balance", float(base_diagnostics["energy_balance_relative"]) <= GATE_LIMITS["energy_balance"], float(base_diagnostics["energy_balance_relative"]), GATE_LIMITS["energy_balance"]),
        ("numerical_energy_definiteness", float(base_diagnostics["magnetic_energy_j"]) > 0.0, float(base_diagnostics["magnetic_energy_j"]), 0.0),
        ("current_balance", source_error <= GATE_LIMITS["current_balance"], source_error, GATE_LIMITS["current_balance"]),
        ("weak_source_action", weak_error <= GATE_LIMITS["weak_source_action"], weak_error, GATE_LIMITS["weak_source_action"]),
        ("pm_form_discrepancy_convergence", comparison["fine_fixed_qoi_relative_difference"] <= GATE_LIMITS["pm_form_discrepancy_convergence"], comparison["fine_fixed_qoi_relative_difference"], GATE_LIMITS["pm_form_discrepancy_convergence"]),
        ("backend_parity", parity["relative_field_l2"] <= GATE_LIMITS["backend_parity"], parity["relative_field_l2"], GATE_LIMITS["backend_parity"]),
        ("structured_observed_order", structured["minimum_observed_order"] >= GATE_LIMITS["structured_observed_order"], structured["minimum_observed_order"], GATE_LIMITS["structured_observed_order"]),
    )
    high_resolution_evaluated = qualification["status"] == "EVALUATED"
    gates = tuple(
        _publication_gate(
            gate_id,
            passed,
            measured_value,
            threshold,
            high_resolution_evaluated=high_resolution_evaluated,
        )
        for gate_id, passed, measured_value, threshold in evaluated_gates
    )
    warning_codes = [
        "CELLWISE_MAXIMUM_SCREENING_ONLY",
        "LINEAR_IRON_SATURATION_UNASSESSED",
    ]
    if counts["pm_regions"]:
        warning_codes.append("PM_IRREVERSIBLE_DEMAGNETIZATION_UNASSESSED")
    if qualification["status"] == "NOT_EVALUATED":
        warning_codes.append("HOST_MEMORY_LIMITED_QUALIFICATION")
    if min_cells < GATE_LIMITS["minimum_effective_feature_cells"]:
        warning_codes.append("UNDERRESOLVED_GEOMETRIC_FEATURE")
    if structured["minimum_observed_order"] < GATE_LIMITS["structured_observed_order"]:
        warning_codes.append("STRUCTURED_GRID_METHOD_INSUFFICIENT")
    return PublicationEvidence(
        (
            "ACCEPTED_PUBLICATION_EVIDENCE"
            if all(item[1] == "PASS" for item in gates)
            else "SCREENING_NOT_ACCEPTED"
        ),
        qualification,
        raw_runs,
        studies,
        gates,
        tuple(warning_codes),
        comparison,
        structured,
        parity,
    )


def study_metrics(
    result: MaterialFieldResult,
    *,
    study_id: str,
    padding_characteristic_lengths: float | None = None,
    expansion_factor_from_previous: float | None = None,
) -> dict[str, object]:
    """Compatibility projection; padding/factors are intentionally ignored and derived."""
    return _metrics(raw_run_observation(result, study_id=study_id, role="diagnostic"))


def validate_publication_evidence(
    value: object, *, policy: dict[str, float | int]
) -> str:
    """Recompute hashes, metrics, warnings, and every gate from embedded raw runs."""
    if not isinstance(value, dict) or set(value) != {
        "status", "qualification", "raw_runs", "studies", "gates", "warning_codes",
        "pm_model_form_comparison", "structured_convergence", "backend_parity",
    }:
        raise MaterialFieldValidationError("acceptance contract is not closed")
    from .numerics import _implementation_sha256

    serialized = value["raw_runs"]
    required = int(policy["required_expansion_comparisons"])
    if not isinstance(serialized, list) or len(serialized) != required + 8:
        raise MaterialFieldValidationError("complete raw domain/mesh/alignment/model runs required")
    runs: list[RawRunObservation] = []
    run_keys = {
        "study_id", "role", "run_sha256", "config_sha256", "geometry_sha256",
        "solver_config_identity_sha256",
        "material_sha256", "design_geometry_sha256", "material_registry_sha256",
        "implementation_sha256", "evidence_implementation_sha256", "backend", "grid_sha256",
        "domain_sha256", "problem_sha256", "raw",
    }
    for item in serialized:
        if not isinstance(item, dict) or set(item) != run_keys:
            raise MaterialFieldValidationError("raw run contract is not closed")
        run = RawRunObservation(**item)
        raw = run.raw
        if not isinstance(raw, dict) or set(raw) != {
            "domain", "solution", "diagnostics", "problem"
        }:
            raise MaterialFieldValidationError("raw observation payload is not closed")
        domain = raw["domain"]
        if not isinstance(domain, dict) or set(domain) != {
            "radius_m", "z_min_m", "z_max_m", "radial_intervals",
            "axial_intervals", "dr_m", "dz_m",
        }:
            raise MaterialFieldValidationError("raw domain is not closed")
        diagnostics = raw["diagnostics"]
        if not isinstance(diagnostics, dict) or set(diagnostics) != {
            "converged", "iterations", "initial_residual_l2",
            "final_true_residual_l2", "relative_true_residual_l2",
            "residual_history_l2", "true_residual_restarts", "backend",
            "magnetic_energy_j", "source_coenergy_j", "energy_balance_relative",
            "run_config_sha256", "implementation_sha256", "run_config_json",
            "host_synchronization_count", "convergence_check_interval",
        }:
            raise MaterialFieldValidationError("raw solver diagnostics are not closed")
        problem = raw["problem"]
        if not isinstance(problem, dict) or set(problem) != {
            "problem_id", "geometry_schema_version", "geometry_sha256",
            "magnetics_sha256", "authority", "coefficient_sha256",
            "source_sha256", "outer_boundary_kind", "geometry_bundle_json",
            "magnetics_bundle_json",
            "raster_diagnostics", "weak_action_diagnostics", "source_envelope_m",
            "feature_effective_cells", "qoi_locations_rz_m", "counts",
            "qoi_bore_windows_m", "open_boundary_policy",
        }:
            raise MaterialFieldValidationError("raw coefficient problem is not closed")
        counts = problem["counts"]
        if not isinstance(counts, dict) or set(counts) != {
            "material_regions", "pm_regions", "free_current_sources", "interfaces"
        }:
            raise MaterialFieldValidationError("raw authoritative counts are not closed")
        raster_keys = {
            "item_id", "requested_volume_m3", "represented_volume_m3",
            "relative_volume_error", "requested_source_measure",
            "represented_source_measure", "relative_source_error",
        }
        action_keys = {
            "basis_id", "analytical_action_a", "rasterized_action_a",
            "absolute_bias_a", "relative_bias",
        }
        if any(
            not isinstance(entry, dict) or set(entry) != raster_keys
            for entry in problem["raster_diagnostics"]
        ) or any(
            not isinstance(entry, dict) or set(entry) != action_keys
            for entry in problem["weak_action_diagnostics"]
        ):
            raise MaterialFieldValidationError("raw source diagnostics are not closed")
        weak_prefix = (
            "recoil-gradient-"
            if problem["authority"] == "recoil_remanence_constitutive"
            else "equivalent-value-"
        )
        expected_weak_ids = {
            weak_prefix + basis for basis in ("one", "r", "z", "r_z")
        }
        if {
            entry["basis_id"] for entry in problem["weak_action_diagnostics"]
        } != expected_weak_ids:
            raise MaterialFieldValidationError(
                "weak-action diagnostic enumeration is incomplete"
            )
        nr = int(domain["radial_intervals"]) + 1
        nz = int(domain["axial_intervals"]) + 1
        solution = raw["solution"]
        if (
            not isinstance(solution, dict)
            or set(solution) != {
                "codec", "dtype", "layout", "count", "uncompressed_sha256",
                "data_base64",
            }
            or solution["count"] != nr * nz
        ):
            raise MaterialFieldValidationError("raw field shape disagrees with grid")
        domain_payload = {
            "radius_m": domain["radius_m"],
            "z_min_m": domain["z_min_m"],
            "z_max_m": domain["z_max_m"],
        }
        anchors = {
            "study_id": run.study_id,
            "role": run.role,
            "config_sha256": run.config_sha256,
            "solver_config_identity_sha256": run.solver_config_identity_sha256,
            "geometry_sha256": run.geometry_sha256,
            "material_sha256": run.material_sha256,
            "design_geometry_sha256": run.design_geometry_sha256,
            "material_registry_sha256": run.material_registry_sha256,
            "implementation_sha256": run.implementation_sha256,
            "evidence_implementation_sha256": run.evidence_implementation_sha256,
            "backend": run.backend,
            "grid_sha256": run.grid_sha256,
            "domain_sha256": run.domain_sha256,
            "problem_sha256": run.problem_sha256,
            "raw": raw,
        }
        design_geometry_sha, material_registry_sha = _study_identity_hashes(problem)
        if (
            run.problem_sha256 != _sha(raw["problem"])
            or run.grid_sha256 != _sha(domain)
            or run.domain_sha256 != _sha(domain_payload)
            or run.run_sha256 != _sha(anchors)
            or run.evidence_implementation_sha256
            != _evidence_implementation_sha256()
            or run.design_geometry_sha256 != design_geometry_sha
            or run.material_registry_sha256 != material_registry_sha
            or run.solver_config_identity_sha256
            != _solver_config_identity(diagnostics["run_config_json"])
        ):
            raise MaterialFieldValidationError("raw run hash binding failed")
        expected_implementation = _implementation_sha256(
            *(
                ("adapters.py", "models.py", "numerics.py", "warp_solver.py")
                if str(run.backend).startswith("material_fields:warp:")
                else ("adapters.py", "models.py", "numerics.py")
            )
        )
        if run.implementation_sha256 != expected_implementation:
            raise MaterialFieldValidationError("run implementation hash is not current")
        for digest in (
            run.run_sha256, run.config_sha256, run.geometry_sha256,
            run.solver_config_identity_sha256,
            run.material_sha256, run.design_geometry_sha256,
            run.material_registry_sha256, run.implementation_sha256, run.grid_sha256,
            run.evidence_implementation_sha256,
            run.domain_sha256, run.problem_sha256,
        ):
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise MaterialFieldValidationError("raw run digest is invalid")
        runs.append(run)
        if not _replay_cached(run).passed:
            raise MaterialFieldValidationError("deterministic run replay failed")
    if len(
        {
            (
                run.design_geometry_sha256,
                run.material_registry_sha256,
                run.solver_config_identity_sha256,
                run.evidence_implementation_sha256,
            )
            for run in runs
        }
    ) != 1:
        raise MaterialFieldValidationError(
            "study runs do not share one geometry/material/config/evidence-code identity"
        )
    expected_sequence = [
        ("base", "base"),
        *((f"domain-{index}", "domain_expansion") for index in range(1, required + 1)),
        ("mesh-fine", "mesh_refinement"),
        ("mesh-third", "mesh_refinement_2"),
        ("alignment-1", "grid_alignment"),
        ("equivalent-base", "model_form"),
        ("equivalent-fine", "model_form"),
        ("parity-cpu", "backend_parity_cpu"),
        ("parity-cuda", "backend_parity_cuda"),
    ]
    if [(run.study_id, run.role) for run in runs] != expected_sequence:
        raise MaterialFieldValidationError(
            "raw study role order/cardinality/multiplicity is invalid"
        )
    base_grid = [
        int(runs[0].raw["domain"]["radial_intervals"]),
        int(runs[0].raw["domain"]["axial_intervals"]),
    ]
    qualification = _validate_qualification(
        value["qualification"], base_grid=base_grid
    )
    if (
        len({run.backend for run in runs[:-2]}) != 1
        or not runs[-2].backend.endswith("python")
        or "warp:cuda:" not in runs[-1].backend
    ):
        raise MaterialFieldValidationError("CPU/CUDA parity backend order is invalid")
    if len({item.run_sha256 for item in runs}) != len(runs):
        raise MaterialFieldValidationError("raw runs are duplicated")
    recomputed_studies = [_metrics(item) for item in runs]
    if value["studies"] != recomputed_studies:
        raise MaterialFieldValidationError("study metrics are not derived from raw runs")
    domains = [item for item in recomputed_studies if item["role"] in {"base", "domain_expansion"}]
    domain_runs = [item for item in runs if item.role in {"base", "domain_expansion"}]
    if len(domains) != required + 1:
        raise MaterialFieldValidationError("wrong number of successive domain runs")
    for left, right in zip(domain_runs, domain_runs[1:]):
        ld, rd = left.raw["domain"], right.raw["domain"]
        if not (
            rd["radius_m"] > ld["radius_m"]
            and rd["z_min_m"] < ld["z_min_m"]
            and rd["z_max_m"] > ld["z_max_m"]
        ):
            raise MaterialFieldValidationError("domain sequence is not strictly increasing")
        base_domain = domain_runs[0].raw["domain"]
        radial_cells = rd["radius_m"] / base_domain["dr_m"]
        axial_offset = (base_domain["z_min_m"] - rd["z_min_m"]) / base_domain["dz_m"]
        if (
            not isclose(rd["dr_m"], base_domain["dr_m"], rel_tol=1.0e-12)
            or not isclose(rd["dz_m"], base_domain["dz_m"], rel_tol=1.0e-12)
            or abs(radial_cells - round(radial_cells)) > 1.0e-9
            or abs(axial_offset - round(axial_offset)) > 1.0e-9
        ):
            raise MaterialFieldValidationError("domain sequence is not phase locked")
    base = next(item for item in recomputed_studies if item["role"] == "base")
    fine = next(item for item in recomputed_studies if item["role"] == "mesh_refinement")
    third = next(item for item in recomputed_studies if item["role"] == "mesh_refinement_2")
    alignments = [item for item in recomputed_studies if item["role"] == "grid_alignment"]
    models = [item for item in recomputed_studies if item["role"] == "model_form"]
    if len(alignments) != 1 or len(models) != 2:
        raise MaterialFieldValidationError("alignment and PM model-form runs are required")
    expansion_ratios = [
        float(right["padding_characteristic_lengths"])
        / float(left["padding_characteristic_lengths"])
        for left, right in zip(domains, domains[1:])
    ]
    domain_changes = [_max_qoi_change(left, right) for left, right in zip(domains, domains[1:])]
    mesh_change = max(_max_qoi_change(base, fine), _max_qoi_change(fine, third))
    alignment_change = max(_max_qoi_change(base, item) for item in alignments)
    problem = runs[0].raw["problem"]
    raster = problem["raster_diagnostics"]
    counts = problem["counts"]
    complete = (
        isinstance(raster, list)
        and len(raster) == counts["material_regions"] + counts["pm_regions"]
        and len({item["item_id"] for item in raster}) == len(raster)
    )
    source_error = (
        max(abs(float(item["relative_source_error"])) for item in raster)
        if complete and raster else 1.0e300
    )
    weak = problem["weak_action_diagnostics"]
    weak_error = max(
        (
            float(item["relative_bias"])
            for item in weak
            if float(item["absolute_bias_a"]) > 1.0e-8
        ),
        default=0.0,
    )
    min_cells = min(
        float(item["minimum_effective_feature_cells"])
        for item in recomputed_studies
        if not str(item["role"]).startswith("backend_parity")
    )
    diagnostics = runs[0].raw["diagnostics"]
    base_gap = _max_qoi_change(base, models[0])
    fine_gap = _max_qoi_change(fine, models[1])
    comparison = {
        "base_fixed_qoi_relative_difference": base_gap,
        "fine_fixed_qoi_relative_difference": fine_gap,
        "discrepancy_change": _relative(base_gap, fine_gap),
    }
    if value["pm_model_form_comparison"] != comparison:
        raise MaterialFieldValidationError("PM comparison is not derived from raw runs")
    run_by_id = {run.study_id: run for run in runs}
    structured = _structured_diagnostics(base, fine, third, run_by_id)
    parity = _backend_parity_metrics(runs[-2], runs[-1])
    if value["structured_convergence"] != structured:
        raise MaterialFieldValidationError("structured convergence is not derived")
    if value["backend_parity"] != parity:
        raise MaterialFieldValidationError("backend parity is not derived")
    expected = (
        (
            "high_resolution_qualification",
            1.0 if qualification["status"] == "EVALUATED" else 0.0,
            1.0,
            ">=",
        ),
        ("required_domain_expansions", float(len(domains) - 1), float(required), ">="),
        ("minimum_base_padding", float(base["padding_characteristic_lengths"]), float(policy["minimum_padding_characteristic_lengths"]), ">="),
        ("domain_expansion_factor", min(expansion_ratios), float(policy["domain_expansion_factor"]), ">="),
        ("domain_phase_lock", 1.0, 1.0, ">="),
        ("boundary_field_ratio", max(float(item["boundary_to_peak_ratio"]) for item in domains), float(policy["maximum_boundary_to_peak_field_ratio"]), "<="),
        ("successive_fixed_qoi", max(domain_changes[-required:]), float(policy["maximum_qoi_relative_change"]), "<="),
        ("mesh_fixed_qoi", mesh_change, GATE_LIMITS["mesh_fixed_qoi"], "<="),
        ("alignment_fixed_qoi", alignment_change, GATE_LIMITS["alignment_fixed_qoi"], "<="),
        ("minimum_effective_feature_cells", min_cells, GATE_LIMITS["minimum_effective_feature_cells"], ">="),
        ("true_equation_residual", float(diagnostics["relative_true_residual_l2"]), GATE_LIMITS["true_equation_residual"], "<="),
        ("energy_balance", float(diagnostics["energy_balance_relative"]), GATE_LIMITS["energy_balance"], "<="),
        ("numerical_energy_definiteness", float(diagnostics["magnetic_energy_j"]), 0.0, ">"),
        ("current_balance", source_error, GATE_LIMITS["current_balance"], "<="),
        ("weak_source_action", weak_error, GATE_LIMITS["weak_source_action"], "<="),
        ("pm_form_discrepancy_convergence", comparison["fine_fixed_qoi_relative_difference"], GATE_LIMITS["pm_form_discrepancy_convergence"], "<="),
        ("backend_parity", float(parity["relative_field_l2"]), GATE_LIMITS["backend_parity"], "<="),
        ("structured_observed_order", float(structured["minimum_observed_order"]), GATE_LIMITS["structured_observed_order"], ">="),
    )
    expected_gates = [
        {
            "gate_id": key,
            "status": (
                "PASS"
                if qualification["status"] == "EVALUATED"
                and (
                    observed >= limit * (1.0 - 1.0e-12)
                    if comparison_op == ">="
                    else observed > limit
                    if comparison_op == ">"
                    else observed <= limit
                )
                else "FAIL"
                if qualification["status"] == "EVALUATED"
                else "NOT_EVALUATED"
            ),
            "measured_value": observed,
            "threshold": limit,
            "diagnostic_status": (
                "MEASURED_HIGH_RESOLUTION"
                if qualification["status"] == "EVALUATED"
                else "MEASURED_REDUCED_RESOURCE_ONLY"
            ),
        }
        for key, observed, limit, comparison_op in expected
    ]
    if value["gates"] != expected_gates:
        raise MaterialFieldValidationError("gates are not recomputed from raw evidence")
    warnings = [
        "CELLWISE_MAXIMUM_SCREENING_ONLY",
        "LINEAR_IRON_SATURATION_UNASSESSED",
    ]
    if counts["pm_regions"]:
        warnings.append("PM_IRREVERSIBLE_DEMAGNETIZATION_UNASSESSED")
    if qualification["status"] == "NOT_EVALUATED":
        warnings.append("HOST_MEMORY_LIMITED_QUALIFICATION")
    if min_cells < GATE_LIMITS["minimum_effective_feature_cells"]:
        warnings.append("UNDERRESOLVED_GEOMETRIC_FEATURE")
    if structured["minimum_observed_order"] < GATE_LIMITS["structured_observed_order"]:
        warnings.append("STRUCTURED_GRID_METHOD_INSUFFICIENT")
    if value["warning_codes"] != warnings or any(item not in WARNING_CODES for item in warnings):
        raise MaterialFieldValidationError("warning codes are not evidence-derived")
    status = (
        "ACCEPTED_PUBLICATION_EVIDENCE"
        if all(item["status"] == "PASS" for item in expected_gates)
        else "SCREENING_NOT_ACCEPTED"
    )
    if value["status"] != status:
        raise MaterialFieldValidationError("acceptance status is not evidence-derived")
    return status
