"""Reproducible field-only L1a geometry design-space experiment.

The experiment deliberately uses the accepted non-authoritative geometry-to-L1a
current-equivalent preview.  It is a screening study, not a material-aware
permanent-magnet solve and not a hardware-valid performance prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    FieldMap,
    SolverConfig,
    field_artifact,
    max_field_difference,
    solve_problem_cpu,
    solve_problem_warp,
    source_discretization_diagnostics,
    validate_field_artifact,
    validate_field_artifact_file,
    write_field_artifact,
)
from cft_revival.geometry import (
    EvidenceNote,
    GeometryValidationError,
    MaterialKind,
    PPMStackParameters,
    canonical_json,
    compute_descriptors,
    deserialize_geometry,
    generate_twt_inspired_ppm_stack,
    to_l1a_current_equivalent_preview,
)
from cft_revival.magnetics import (
    LinearPermeability,
    checked_synthetic_smco_like_magnet,
)
from cft_revival.optimization import Design, Variable
from cft_revival.optimization.sampling import initial_designs

SCHEMA_VERSION = "cft-revival.experiment.l1a-geometry-sweep/1.0.0"
MANIFEST_VERSION = "cft-revival.experiment.l1a-geometry-sweep-manifest/1.0.0"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
CLASSIFICATION = "L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID"
RUNTIME_POLICY = (
    "wall times are uncontrolled diagnostics under concurrent GPU load; "
    "they are not benchmark evidence"
)

VARIABLES = (
    Variable("stage_count_selector", 0.0, 1.0, "1"),
    Variable("stage_pitch_m", 0.0038, 0.0065, "m"),
    Variable("magnet_axial_fraction", 0.48, 0.72, "1"),
    Variable("chamber_outer_radius_m", 0.0014, 0.0022, "m"),
    Variable("dielectric_thickness_m", 0.0005, 0.0009, "m"),
    Variable("radial_clearance_m", 0.00035, 0.0008, "m"),
    Variable("magnet_radial_thickness_m", 0.0022, 0.0040, "m"),
    Variable("source_strength_scale", 0.75, 1.30, "1"),
    Variable("exit_length_fraction", 0.0, 0.28, "1"),
    Variable("exit_expansion_descriptor", 0.0, 1.0, "1"),
    Variable("first_polarity_selector", 0.0, 1.0, "1"),
)

DOMAIN = AxisymmetricDomain(
    radius_m=0.030,
    z_min_m=-0.015,
    z_max_m=0.050,
    radial_intervals=80,
    axial_intervals=144,
)
SOLVER = SolverConfig(
    relative_tolerance=1.0e-10,
    absolute_tolerance=1.0e-13,
    max_iterations=20_000,
    residual_history_stride=10,
    max_true_residual_restarts=2,
)
SMEAR_THICKNESS_M = 8.0e-4

OBJECTIVES = (
    {"name": "centreline_mid_abs_bz_t", "direction": "maximize", "units": "T"},
    {"name": "minimum_mirror_ratio", "direction": "maximize", "units": "1"},
    {
        "name": "stage_gradient_rms_t_per_m",
        "direction": "maximize",
        "units": "T/m",
    },
    {"name": "field_energy_j", "direction": "minimize", "units": "J"},
)

CONSTRAINTS = (
    {
        "name": "boundary_to_peak_ratio",
        "sense": "<=",
        "threshold": 0.05,
        "units": "1",
    },
    {
        "name": "relative_residual_l2",
        "sense": "<=",
        "threshold": 1.0e-10,
        "units": "1",
    },
    {
        "name": "flux_reconstruction_identity_t_per_m",
        "sense": "<=",
        "threshold": 1.0e-8,
        "units": "T/m",
    },
    {
        "name": "source_representation_error",
        "sense": "<=",
        "threshold": 0.25,
        "units": "1",
    },
    {
        "name": "topology_confidence",
        "sense": ">=",
        "threshold": 0.50,
        "units": "1",
    },
    {
        "name": "worst_case_radial_manufacturing_margin_m",
        "sense": ">=",
        "threshold": 0.0,
        "units": "m",
    },
    {
        "name": "worst_case_axial_manufacturing_margin_m",
        "sense": ">=",
        "threshold": 0.0,
        "units": "m",
    },
)

FAILURE_TAXONOMY = {
    "SAMPLING_FAILURE": "accepted bounded sampler could not construct a design",
    "GEOMETRY_INVALID": "geometry v1.1 or manufacturability validation failed",
    "PREVIEW_INVALID": "non-authoritative current-equivalent preview failed",
    "SOURCE_INVALID": "scaled L1a source/domain contract failed",
    "SOLVER_FAILURE": "Warp solve raised or did not satisfy true residual",
    "ARTIFACT_INVALID": "accepted strict L1a artifact validation failed",
    "BOUNDARY_GATE_FAILURE": "finite-domain boundary-to-peak gate failed",
    "PARITY_FAILURE": "CPU/CUDA parity gate failed",
}


@dataclass(frozen=True)
class BuiltCase:
    case_id: str
    design: Design
    geometry: Any
    preview: Any
    problem: AxisymmetricProblem
    geometry_sha256: str
    source_sha256: str
    config_sha256: str
    case_sha256: str
    derived: Mapping[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {
        **body,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": CANONICALIZATION,
            "payload_sha256": stable_hash(body),
        },
    }


def _strict_load(path: Path) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject,
    )
    if not isinstance(loaded, dict):
        raise ValueError("sealed artifact must contain an object")
    return loaded


def _verify_sidecar(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = f"{digest}  {path.name}\n"
    if path.with_name(path.name + ".sha256").read_text(encoding="ascii") != expected:
        raise ValueError(f"invalid sidecar for {path.name}")
    return digest


def validate_sealed_file(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    value = _strict_load(path)
    integrity = value.get("integrity")
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "canonicalization",
        "payload_sha256",
    }:
        raise ValueError("invalid integrity object")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != CANONICALIZATION
    ):
        raise ValueError("unsupported integrity declaration")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if integrity["payload_sha256"] != stable_hash(payload):
        raise ValueError("payload SHA-256 mismatch")
    return value


def _write_bytes_with_sidecar(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def write_sealed_json(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value = seal(payload)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    digest = _write_bytes_with_sidecar(path, encoded)
    validate_sealed_file(path)
    return value, digest


def sample_designs(count: int = 96, seed: int = 20260902) -> tuple[Design, ...]:
    """Accepted deterministic shifted-Halton sampling with bounded challenges."""

    return initial_designs(
        VARIABLES,
        count,
        seed=seed,
        include_boundary_challenges=True,
    )


def _design_values(design: Design) -> dict[str, float]:
    return {
        variable.name: value
        for variable, value in zip(design.variables, design.values, strict=True)
    }


def _contract_stable_pitch(requested: float, stage_count: int) -> float:
    """Move by a few ULPs, inward, until geometry's pitch identity is stable."""

    pitch = requested
    target = 0.5 * (
        next(item.lower for item in VARIABLES if item.name == "stage_pitch_m")
        + next(item.upper for item in VARIABLES if item.name == "stage_pitch_m")
    )
    for _ in range(64):
        centers = tuple((stage + 0.5) * pitch for stage in range(stage_count))
        if all(
            abs((right - left) - pitch)
            <= 2.0 * max(math.ulp(right - left), math.ulp(pitch))
            for left, right in zip(centers, centers[1:])
        ):
            return pitch
        pitch = math.nextafter(pitch, target)
    raise GeometryValidationError("could not represent a contract-stable axial pitch")


def _ulp_close(left: float, right: float, limit: float) -> bool:
    return abs(left - right) <= limit * max(math.ulp(left), math.ulp(right))


def _contract_stable_exit_radius(
    chamber_radius: float,
    requested: float,
    dielectric: float,
    exit_length: float,
) -> float:
    """Move a tapered endpoint inward by ULPs until wall slopes remain identical."""

    radius = requested
    for _ in range(128):
        channel = (radius - chamber_radius) / exit_length
        wall_inner = ((radius) - chamber_radius) / exit_length
        wall_outer = (
            (radius + dielectric) - (chamber_radius + dielectric)
        ) / exit_length
        if (
            _ulp_close((chamber_radius + dielectric) - chamber_radius, dielectric, 2.0)
            and _ulp_close((radius + dielectric) - radius, dielectric, 2.0)
            and _ulp_close(channel, wall_inner, 4.0)
            and _ulp_close(channel, wall_outer, 4.0)
        ):
            return radius
        radius = math.nextafter(radius, chamber_radius)
    raise GeometryValidationError("could not represent contract-stable divergent slopes")


def _material_registry(geometry: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for material in geometry.materials:
        if material.category is MaterialKind.PERMANENT_MAGNET:
            resolved = checked_synthetic_smco_like_magnet()
            if resolved.material_id != material.material_id:
                raise GeometryValidationError("accepted PM registry ID mismatch")
        else:
            resolved = LinearPermeability(
                material.material_id, material.relative_permeability
            )
        result[material.material_id] = resolved
    return result


def build_case(design: Design, index: int) -> BuiltCase:
    values = _design_values(design)
    stage_count = min(5, 3 + int(values["stage_count_selector"] * 3.0))
    requested_pitch = values["stage_pitch_m"]
    pitch = _contract_stable_pitch(requested_pitch, stage_count)
    magnet_axial_thickness = pitch * values["magnet_axial_fraction"]
    chamber_radius = values["chamber_outer_radius_m"]
    dielectric = values["dielectric_thickness_m"]
    exit_fraction = values["exit_length_fraction"]
    chamber_length = stage_count * pitch
    requested_exit_length = chamber_length * exit_fraction
    exit_length = (
        0.0
        if requested_exit_length < 2.5e-4
        else requested_exit_length
    )
    requested_exit_radius = (
        chamber_radius
        if exit_length == 0.0
        else chamber_radius
        * (1.15 + 0.35 * values["exit_expansion_descriptor"])
    )
    exit_radius = (
        requested_exit_radius
        if exit_length == 0.0
        else _contract_stable_exit_radius(
            chamber_radius,
            requested_exit_radius,
            dielectric,
            exit_length,
        )
    )
    radial_clearance = values["radial_clearance_m"]
    magnet_inner = max(chamber_radius, exit_radius) + dielectric + radial_clearance
    magnet_outer = magnet_inner + values["magnet_radial_thickness_m"]
    stage_centers = tuple((stage + 0.5) * pitch for stage in range(stage_count))
    case_id = f"l1a-gs-{index:03d}-{design.design_id[:10]}"
    geometry = generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id=f"{case_id}-v1",
            title=f"L1a geometry sweep case {index:03d} (screening only)",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=chamber_radius,
            chamber_length_m=chamber_length,
            injector_length_m=0.08 * chamber_length,
            dielectric_thickness_m=dielectric,
            thermal_clearance_m=2.5e-4,
            magnet_inner_radius_m=magnet_inner,
            magnet_outer_radius_m=magnet_outer,
            stage_pitch_m=pitch,
            stage_centers_m=stage_centers,
            magnet_axial_thicknesses_m=(magnet_axial_thickness,) * stage_count,
            shield_outer_radius_m=magnet_outer + 7.5e-4,
            yoke_outer_radius_m=magnet_outer + 1.75e-3,
            exit_length_m=exit_length,
            exit_outer_radius_m=exit_radius,
            first_polarity=(
                1 if values["first_polarity_selector"] < 0.5 else -1
            ),
            radial_tolerance_m=2.5e-5,
            axial_tolerance_m=2.5e-5,
            minimum_thickness_m=2.5e-4,
            minimum_clearance_m=1.0e-4,
        ),
        evidence=(
            EvidenceNote(
                f"case-{index:03d}-screening",
                "assumption",
                "Bounded deterministic design-space sample for L1a field-only screening.",
                "Experiment-local shifted-Halton parameterization.",
            ),
        ),
    )
    descriptors = compute_descriptors(geometry)
    preview = to_l1a_current_equivalent_preview(
        geometry,
        material_registry=_material_registry(geometry),
        radial_smear_thickness_m=SMEAR_THICKNESS_M,
    )
    source_scale = values["source_strength_scale"]
    bands = tuple(
        replace(band, ampere_turns_a=band.ampere_turns_a * source_scale)
        for band in preview.bands
    )
    problem = AxisymmetricProblem(case_id, DOMAIN, bands)
    source_payload = {
        **preview.to_dict(),
        "source_strength_scale": source_scale,
        "scaled_bands": [asdict(band) for band in bands],
    }
    source_sha256 = stable_hash(source_payload)
    radial_margin = (
        descriptors.minimum_radial_gap_m
        - 2.0 * geometry.manufacturing.radial_tolerance_m
        - geometry.manufacturing.thermal_clearance_m
    )
    axial_margin = (
        descriptors.minimum_axial_gap_m
        - 2.0 * geometry.manufacturing.axial_tolerance_m
        - geometry.manufacturing.minimum_clearance_m
    )
    derived = {
        "stage_count": stage_count,
        "requested_stage_pitch_m": requested_pitch,
        "represented_stage_pitch_m": pitch,
        "stage_centers_m": list(stage_centers),
        "magnet_axial_thickness_m": magnet_axial_thickness,
        "magnet_inner_radius_m": magnet_inner,
        "magnet_outer_radius_m": magnet_outer,
        "chamber_length_m": chamber_length,
        "requested_exit_length_m": requested_exit_length,
        "represented_exit_length_m": exit_length,
        "requested_exit_outer_radius_m": requested_exit_radius,
        "exit_outer_radius_m": exit_radius,
        "worst_case_radial_manufacturing_margin_m": radial_margin,
        "worst_case_axial_manufacturing_margin_m": axial_margin,
        "geometry_descriptors": descriptors.to_dict(),
    }
    config_payload = {
        "schema_version": SCHEMA_VERSION,
        "design_id": design.design_id,
        "domain": asdict(DOMAIN),
        "solver": asdict(SOLVER),
        "smear_thickness_m": SMEAR_THICKNESS_M,
        "objectives": OBJECTIVES,
        "constraints": CONSTRAINTS,
        "qoi_policy": qoi_policy(),
    }
    config_sha256 = stable_hash(config_payload)
    case_sha256 = stable_hash(
        {
            "geometry_sha256": geometry.canonical_sha256,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
        }
    )
    return BuiltCase(
        case_id,
        design,
        geometry,
        preview,
        problem,
        geometry.canonical_sha256,
        source_sha256,
        config_sha256,
        case_sha256,
        derived,
    )


def qoi_policy() -> dict[str, str]:
    return {
        "fixed_locations": (
            "nearest grid nodes at axis/chamber midpoint, axis/injector exit, "
            "axis/divergent-exit start, and chamber-radius/chamber midpoint"
        ),
        "mirror_ratio": (
            "for each adjacent stage pair: larger absolute axis Bz at the two "
            "stage centres divided by max(abs(Bz at gap midpoint), topology tolerance)"
        ),
        "stage_gradient": "centred axis-Bz finite difference at each stage centre",
        "boundary_ratio": "maximum sampled non-axis outer-boundary |B| / global sampled peak |B|",
        "field_energy": "axisymmetric trapezoidal integral of B^2/(2*mu0) over the domain",
        "source_representation_error": (
            "maximum of per-band relative ampere-turn error, relative overlap-area "
            "error, and centroid error divided by minimum band thickness"
        ),
        "topology_confidence": (
            "clamp(1 - 0.25*boundary_ratio/0.05 - 0.25*source_error/0.25 "
            "- 0.25*dz/min_pitch - 0.25*plateau_indicator, 0, 1)"
        ),
        "claim_limit": (
            "sampled-grid descriptors only; null interpolation is not a continuous "
            "critical-point proof"
        ),
    }


def _nearest(values: Sequence[float], target: float) -> int:
    return min(range(len(values)), key=lambda index: abs(values[index] - target))


def _field_peak(field: FieldMap) -> float:
    return max(
        math.hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )


def _boundary_ratio(field: FieldMap, peak: float) -> float:
    nr, nz = len(field.r_m), len(field.z_m)
    boundary = (
        math.hypot(field.b_r_t[i][j], field.b_z_t[i][j])
        for i in range(nr)
        for j in range(nz)
        if i == nr - 1 or j == 0 or j == nz - 1
    )
    return max(boundary) / max(peak, 1.0e-300)


def _field_energy(field: FieldMap) -> float:
    dr = field.r_m[1] - field.r_m[0]
    dz = field.z_m[1] - field.z_m[0]
    total = 0.0
    for i, radius in enumerate(field.r_m):
        wr = 0.5 if i in (0, len(field.r_m) - 1) else 1.0
        for j in range(len(field.z_m)):
            wz = 0.5 if j in (0, len(field.z_m) - 1) else 1.0
            b2 = field.b_r_t[i][j] ** 2 + field.b_z_t[i][j] ** 2
            total += wr * wz * radius * b2
    return math.pi * dr * dz * total / 1.2566370614359173e-6


def _source_error(case: BuiltCase) -> tuple[float, list[dict[str, Any]]]:
    diagnostics = [dict(item) for item in source_discretization_diagnostics(case.problem)]
    errors: list[float] = []
    for source, item in zip(case.problem.sources, diagnostics, strict=True):
        area = float(item["requested_area_m2"])
        current = abs(float(item["requested_signed_ampere_turns_a"]))
        thickness = min(
            source.r_outer_m - source.r_inner_m,
            source.z_max_m - source.z_min_m,
        )
        errors.extend(
            (
                abs(float(item["area_error_m2"])) / max(area, 1.0e-300),
                abs(float(item["ampere_turn_error_a"])) / max(current, 1.0e-300),
                math.hypot(
                    float(item["centroid_r_error_m"]),
                    float(item["centroid_z_error_m"]),
                )
                / thickness,
            )
        )
    return max(errors, default=0.0), diagnostics


def extract_qois(
    case: BuiltCase, field: FieldMap, accepted_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    peak = _field_peak(field)
    boundary_ratio = _boundary_ratio(field, peak)
    source_error, source_diagnostics = _source_error(case)
    axis = field.b_z_t[0]
    chamber = case.geometry.chamber
    fixed_targets = {
        "axis_chamber_mid": (0.0, 0.5 * chamber.length_m),
        "axis_injector_exit": (0.0, chamber.injector_length_m),
        "axis_exit_start": (0.0, chamber.exit_start_m),
        "channel_wall_chamber_mid": (
            chamber.outer_radius_m,
            0.5 * chamber.length_m,
        ),
    }
    fixed: dict[str, dict[str, float]] = {}
    for name, (radius, axial) in fixed_targets.items():
        i = _nearest(field.r_m, radius)
        j = _nearest(field.z_m, axial)
        fixed[name] = {
            "requested_r_m": radius,
            "requested_z_m": axial,
            "sampled_r_m": field.r_m[i],
            "sampled_z_m": field.z_m[j],
            "b_z_t": field.b_z_t[i][j],
        }
    topology = accepted_artifact["summary"]["topology"]
    topology_tolerance = float(topology["null_tolerance_t"])
    stage_indices = [_nearest(field.z_m, stage.center_z_m) for stage in case.geometry.stages]
    stage_bz = [axis[index] for index in stage_indices]
    stage_gradients: list[float] = []
    for index in stage_indices:
        index = min(max(1, index), len(axis) - 2)
        stage_gradients.append(
            (axis[index + 1] - axis[index - 1])
            / (field.z_m[index + 1] - field.z_m[index - 1])
        )
    mirror_ratios: list[float] = []
    for left, right, left_bz, right_bz in zip(
        case.geometry.stages[:-1],
        case.geometry.stages[1:],
        stage_bz[:-1],
        stage_bz[1:],
        strict=True,
    ):
        gap = 0.5 * (left.center_z_m + right.center_z_m)
        gap_bz = abs(axis[_nearest(field.z_m, gap)])
        mirror_ratios.append(
            max(abs(left_bz), abs(right_bz))
            / max(gap_bz, topology_tolerance, 1.0e-300)
        )
    cusp_indices = [
        index
        for index in range(1, len(axis) - 1)
        if 0.0 <= field.z_m[index] <= chamber.length_m
        and abs(axis[index]) >= abs(axis[index - 1])
        and abs(axis[index]) > abs(axis[index + 1])
    ]
    min_pitch = min(stage.pitch_m for stage in case.geometry.stages)
    plateau_indicator = 1.0 if topology["axis_plateaus"] else 0.0
    confidence = min(
        1.0,
        max(
            0.0,
            1.0
            - 0.25 * boundary_ratio / 0.05
            - 0.25 * source_error / 0.25
            - 0.25 * DOMAIN.dz_m / min_pitch
            - 0.25 * plateau_indicator,
        ),
    )
    gradient_rms = math.sqrt(
        sum(value * value for value in stage_gradients) / len(stage_gradients)
    )
    return {
        "centreline_bz_min_t": min(axis),
        "centreline_bz_max_t": max(axis),
        "centreline_abs_bz_peak_t": max(abs(value) for value in axis),
        "centreline_mid_abs_bz_t": abs(fixed["axis_chamber_mid"]["b_z_t"]),
        "fixed_location_bz": fixed,
        "stage_centre_bz_t": stage_bz,
        "mirror_ratios": mirror_ratios,
        "minimum_mirror_ratio": min(mirror_ratios),
        "maximum_mirror_ratio": max(mirror_ratios),
        "axis_cusp_count": len(cusp_indices),
        "axis_cusp_positions_m": [field.z_m[index] for index in cusp_indices],
        "axis_null_count": len(topology["axis_nulls"]),
        "axis_null_positions_m": [
            float(item["z_m"]) for item in topology["axis_nulls"]
        ],
        "topology_status": topology["status"],
        "topology_confidence": confidence,
        "stage_gradients_t_per_m": stage_gradients,
        "stage_gradient_rms_t_per_m": gradient_rms,
        "stage_gradient_max_abs_t_per_m": max(
            abs(value) for value in stage_gradients
        ),
        "boundary_to_peak_ratio": boundary_ratio,
        "field_energy_j": _field_energy(field),
        "source_representation_error": source_error,
        "source_discretization": source_diagnostics,
        "field_peak_t": peak,
        "relative_residual_l2": field.diagnostics.relative_residual_l2,
        "flux_reconstruction_identity_t_per_m": (
            field.diagnostics.max_flux_reconstruction_identity_t_per_m
        ),
    }


def constraint_values(case: BuiltCase, qois: Mapping[str, Any]) -> dict[str, float]:
    return {
        "boundary_to_peak_ratio": float(qois["boundary_to_peak_ratio"]),
        "relative_residual_l2": float(qois["relative_residual_l2"]),
        "flux_reconstruction_identity_t_per_m": float(
            qois["flux_reconstruction_identity_t_per_m"]
        ),
        "source_representation_error": float(qois["source_representation_error"]),
        "topology_confidence": float(qois["topology_confidence"]),
        "worst_case_radial_manufacturing_margin_m": float(
            case.derived["worst_case_radial_manufacturing_margin_m"]
        ),
        "worst_case_axial_manufacturing_margin_m": float(
            case.derived["worst_case_axial_manufacturing_margin_m"]
        ),
    }


def feasible(values: Mapping[str, float]) -> bool:
    for definition in CONSTRAINTS:
        value = values[definition["name"]]
        if definition["sense"] == "<=" and value > definition["threshold"]:
            return False
        if definition["sense"] == ">=" and value < definition["threshold"]:
            return False
    return True


def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Exact deterministic constrained dominance with no failure penalties."""

    if left["status"] != "success" or right["status"] != "success":
        raise ValueError("only successful cases can enter field-objective ranking")
    left_feasible = bool(left["feasible"])
    right_feasible = bool(right["feasible"])
    if left_feasible != right_feasible:
        return left_feasible
    if not left_feasible:
        def violation(case: Mapping[str, Any]) -> float:
            total = 0.0
            for definition in CONSTRAINTS:
                value = case["constraints"][definition["name"]]
                threshold = definition["threshold"]
                scale = max(abs(float(threshold)), 1.0)
                raw = (
                    value - threshold
                    if definition["sense"] == "<="
                    else threshold - value
                )
                total += max(0.0, raw / scale)
            return total

        return violation(left) < violation(right)
    comparisons: list[tuple[float, float]] = []
    for objective in OBJECTIVES:
        left_value = float(left["qois"][objective["name"]])
        right_value = float(right["qois"][objective["name"]])
        if objective["direction"] == "minimize":
            left_value, right_value = -left_value, -right_value
        comparisons.append((left_value, right_value))
    return all(a >= b for a, b in comparisons) and any(
        a > b for a, b in comparisons
    )


def nondominated(cases: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    successful = sorted(
        (case for case in cases if case["status"] == "success"),
        key=lambda case: case["case_id"],
    )
    return tuple(
        candidate
        for candidate in successful
        if not any(
            other["case_id"] != candidate["case_id"]
            and dominates(other, candidate)
            for other in successful
        )
    )


def _range(records: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    values = [float(record["qois"][name]) for record in records]
    return [min(values), max(values)]


def _select_representatives(
    front: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if not front:
        return []
    specifications = (
        ("strongest-centreline", "centreline_mid_abs_bz_t", max),
        ("strongest-mirror", "minimum_mirror_ratio", max),
        ("steepest-stage-gradient", "stage_gradient_rms_t_per_m", max),
        ("lowest-field-energy", "field_energy_j", min),
        ("best-boundary-isolation", "boundary_to_peak_ratio", min),
    )
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, name, function in specifications:
        candidate = function(front, key=lambda item: float(item["qois"][name]))
        if candidate["case_id"] not in seen:
            seen.add(candidate["case_id"])
            selected.append({"label": label, "case_id": candidate["case_id"]})
    for candidate in front:
        if len(selected) >= 5:
            break
        if candidate["case_id"] not in seen:
            seen.add(candidate["case_id"])
            selected.append(
                {"label": f"additional-nondominated-{len(selected)+1}", "case_id": candidate["case_id"]}
            )
    return selected


def _case_record(
    case: BuiltCase,
    field: FieldMap,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    qois = extract_qois(case, field, artifact)
    constraints = constraint_values(case, qois)
    return {
        "case_id": case.case_id,
        "status": "success",
        "failure": None,
        "design_id": case.design.design_id,
        "sampling_provenance": case.design.provenance,
        "design_values": _design_values(case.design),
        "derived_geometry": dict(case.derived),
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "case_sha256": case.case_sha256,
        "backend": field.diagnostics.backend,
        "iterations": field.diagnostics.iterations,
        "qois": qois,
        "constraints": constraints,
        "feasible": feasible(constraints),
        "classification": CLASSIFICATION,
    }


def _parity_record(case: BuiltCase, cuda: FieldMap) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    cpu = solve_problem_cpu(case.problem, SOLVER)
    elapsed = perf_counter() - started
    differences = max_field_difference(cpu, cuda)
    passed = (
        differences["psi_scale_relative"] <= 2.0e-9
        and differences["br_scale_relative"] <= 2.0e-8
        and differences["bz_scale_relative"] <= 2.0e-8
        and cpu.diagnostics.converged
        and cuda.diagnostics.converged
    )
    return (
        {
            "case_id": case.case_id,
            "cpu_backend": cpu.diagnostics.backend,
            "cuda_backend": cuda.diagnostics.backend,
            "differences": differences,
            "gates": {
                "psi_scale_relative_max": 2.0e-9,
                "br_scale_relative_max": 2.0e-8,
                "bz_scale_relative_max": 2.0e-8,
            },
            "passed": passed,
        },
        elapsed,
    )


def _write_geometry(path: Path, geometry: Any) -> str:
    encoded = canonical_json(geometry.to_dict()).encode("utf-8")
    digest = _write_bytes_with_sidecar(path, encoded)
    loaded = deserialize_geometry(path.read_text(encoding="utf-8"))
    if loaded.canonical_sha256 != geometry.canonical_sha256:
        raise ValueError("geometry artifact reload hash mismatch")
    return digest


def _representative_artifacts(
    output: Path,
    selections: Sequence[Mapping[str, str]],
    built_by_id: Mapping[str, BuiltCase],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    entries: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    for selection in selections:
        case = built_by_id[selection["case_id"]]
        started = perf_counter()
        field = solve_problem_warp(case.problem, device="cuda:0", config=SOLVER)
        elapsed = perf_counter() - started
        elapsed_values.append(elapsed)
        stem = case.case_id
        geometry_path = output / "representatives" / f"{stem}.geometry.json"
        geometry_file_sha = _write_geometry(geometry_path, case.geometry)
        full = field_artifact(
            case.problem,
            SOLVER,
            field,
            map_stride=1,
            wall_radius_m=case.geometry.chamber.outer_radius_m,
        )
        downsampled = field_artifact(
            case.problem,
            SOLVER,
            field,
            map_stride=4,
            wall_radius_m=case.geometry.chamber.outer_radius_m,
        )
        full_path = output / "representatives" / f"{stem}.field-full.json"
        down_path = output / "representatives" / f"{stem}.field-downsampled.json"
        full_file_sha = write_field_artifact(full_path, full)
        down_file_sha = write_field_artifact(down_path, downsampled)
        validate_field_artifact_file(
            full_path,
            expected_file_sha256=full_file_sha,
            expected_payload_sha256=full["integrity"]["payload_sha256"],
        )
        validate_field_artifact_file(
            down_path,
            expected_file_sha256=down_file_sha,
            expected_payload_sha256=downsampled["integrity"]["payload_sha256"],
        )
        entries.append(
            {
                "label": selection["label"],
                "case_id": case.case_id,
                "geometry": {
                    "path": str(geometry_path.relative_to(output)).replace("\\", "/"),
                    "file_sha256": geometry_file_sha,
                    "payload_sha256": case.geometry_sha256,
                },
                "full_field": {
                    "path": str(full_path.relative_to(output)).replace("\\", "/"),
                    "file_sha256": full_file_sha,
                    "payload_sha256": full["integrity"]["payload_sha256"],
                    "stride": 1,
                },
                "downsampled_field": {
                    "path": str(down_path.relative_to(output)).replace("\\", "/"),
                    "file_sha256": down_file_sha,
                    "payload_sha256": downsampled["integrity"]["payload_sha256"],
                    "stride": 4,
                },
            }
        )
        runtime.append(
            {
                "case_id": case.case_id,
                "phase": "representative-artifact-rerun",
                "wall_time_seconds": elapsed,
            }
        )
    return entries, runtime, elapsed_values


def _report_text(dataset: Mapping[str, Any]) -> str:
    summary = dataset["summary"]
    ranges = summary["qoi_ranges"]
    representatives = dataset["representatives"]
    lines = [
        "# L1a axisymmetric geometry sweep",
        "",
        f"- Classification: `{CLASSIFICATION}`",
        f"- Screening level: `{summary['screening_level']}`",
        f"- Evaluated: {summary['evaluated_count']}",
        f"- Failed: {summary['failed_count']}",
        f"- Feasible: {summary['feasible_count']}",
        f"- Nondominated: {summary['nondominated_count']}",
        f"- Boundary-gate failures: {summary['boundary_gate_failure_count']}",
        f"- Solver-gate failures: {summary['solver_gate_failure_count']}",
        "",
        "## QoI ranges",
        "",
    ]
    for name in sorted(ranges):
        lines.append(f"- `{name}`: {ranges[name][0]:.12g} to {ranges[name][1]:.12g}")
    lines.extend(("", "## Representatives", ""))
    for item in representatives:
        lines.append(f"- `{item['label']}`: `{item['case_id']}`")
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "This is a deterministic L1a field-only screening experiment using a",
            "non-authoritative thin-band current-equivalent preview. It is not a",
            "material-aware permanent-magnet model, FEM result, propulsion model,",
            "thrust/efficiency calculation, build qualification, or hardware-valid prediction.",
            "",
            "Timing is stored separately as uncontrolled diagnostics and is not benchmark evidence.",
            "",
        )
    )
    return "\n".join(lines)


def run_experiment(
    output: Path,
    *,
    count: int = 96,
    seed: int = 20260902,
    parity_count: int = 6,
) -> dict[str, Any]:
    if count < 64 or count > 128:
        raise ValueError("the declared design-space experiment requires 64..128 cases")
    if parity_count < 1 or parity_count > count:
        raise ValueError("parity_count must lie in 1..count")
    output.mkdir(parents=True, exist_ok=True)
    designs = sample_designs(count, seed)
    cases: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    runtime_cases: list[dict[str, Any]] = []
    failure_counts = {key: 0 for key in FAILURE_TAXONOMY}
    built_by_id: dict[str, BuiltCase] = {}
    parity_indices = {
        round(index * (count - 1) / (parity_count - 1))
        if parity_count > 1
        else 0
        for index in range(parity_count)
    }
    for index, design in enumerate(designs):
        stage = "GEOMETRY_INVALID"
        try:
            case = build_case(design, index)
            built_by_id[case.case_id] = case
            stage = "SOLVER_FAILURE"
            started = perf_counter()
            field = solve_problem_warp(case.problem, device="cuda:0", config=SOLVER)
            elapsed = perf_counter() - started
            runtime_cases.append(
                {
                    "case_id": case.case_id,
                    "phase": "primary-cuda-solve",
                    "wall_time_seconds": elapsed,
                }
            )
            stage = "ARTIFACT_INVALID"
            artifact = field_artifact(
                case.problem,
                SOLVER,
                field,
                map_stride=8,
                wall_radius_m=case.geometry.chamber.outer_radius_m,
            )
            validate_field_artifact(artifact)
            record = _case_record(case, field, artifact)
            if record["qois"]["boundary_to_peak_ratio"] > 0.05:
                failure_counts["BOUNDARY_GATE_FAILURE"] += 1
            if record["qois"]["relative_residual_l2"] > 1.0e-10:
                failure_counts["SOLVER_FAILURE"] += 1
            cases.append(record)
            if index in parity_indices:
                parity_item, parity_elapsed = _parity_record(case, field)
                parity.append(parity_item)
                runtime_cases.append(
                    {
                        "case_id": case.case_id,
                        "phase": "cpu-parity-solve",
                        "wall_time_seconds": parity_elapsed,
                    }
                )
                if not parity_item["passed"]:
                    failure_counts["PARITY_FAILURE"] += 1
        except Exception as error:
            failure_counts[stage] += 1
            cases.append(
                {
                    "case_id": f"l1a-gs-{index:03d}",
                    "status": "failure",
                    "failure": {
                        "code": stage,
                        "message": str(error),
                        "retryable": stage == "SOLVER_FAILURE",
                    },
                    "design_id": design.design_id,
                    "sampling_provenance": design.provenance,
                    "design_values": _design_values(design),
                    "classification": CLASSIFICATION,
                }
            )
    successful = [case for case in cases if case["status"] == "success"]
    front = nondominated(successful)
    selections = _select_representatives(front)
    representative_entries, representative_runtime, _ = _representative_artifacts(
        output, selections, built_by_id
    )
    runtime_cases.extend(representative_runtime)
    range_names = (
        "centreline_bz_min_t",
        "centreline_bz_max_t",
        "centreline_abs_bz_peak_t",
        "centreline_mid_abs_bz_t",
        "minimum_mirror_ratio",
        "maximum_mirror_ratio",
        "axis_cusp_count",
        "axis_null_count",
        "stage_gradient_rms_t_per_m",
        "stage_gradient_max_abs_t_per_m",
        "boundary_to_peak_ratio",
        "field_energy_j",
        "source_representation_error",
        "topology_confidence",
        "field_peak_t",
        "relative_residual_l2",
        "flux_reconstruction_identity_t_per_m",
    )
    summary = {
        "screening_level": "L1a_field_only_design_space_screening",
        "requested_count": count,
        "evaluated_count": len(successful),
        "failed_count": count - len(successful),
        "feasible_count": sum(bool(case["feasible"]) for case in successful),
        "nondominated_count": len(front),
        "nondominated_case_ids": [case["case_id"] for case in front],
        "boundary_gate_failure_count": failure_counts["BOUNDARY_GATE_FAILURE"],
        "solver_gate_failure_count": failure_counts["SOLVER_FAILURE"],
        "parity_failure_count": failure_counts["PARITY_FAILURE"],
        "qoi_ranges": {
            name: _range(successful, name) for name in range_names
        }
        if successful
        else {},
        "failure_counts": failure_counts,
    }
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "model_level": "L1a",
        "sampling": {
            "algorithm": "accepted deterministic shifted-Halton with boundary challenges",
            "seed": seed,
            "count": count,
            "variables": [asdict(variable) for variable in VARIABLES],
        },
        "domain": asdict(DOMAIN),
        "solver": asdict(SOLVER),
        "current_equivalent_preview": {
            "authoritative": False,
            "radial_smear_thickness_m": SMEAR_THICKNESS_M,
            "source_strength_scale_is_experiment_variable": True,
        },
        "objectives": list(OBJECTIVES),
        "constraints": list(CONSTRAINTS),
        "qoi_policy": qoi_policy(),
        "failure_taxonomy": FAILURE_TAXONOMY,
        "cases": cases,
        "parity": parity,
        "representatives": selections,
        "summary": summary,
        "limitations": [
            "No thrust, efficiency, Isp, plasma, thermal, or structural quantity is calculated.",
            "Equivalent current bands are a non-authoritative geometry preview.",
            "Finite Dirichlet-boundary FDM and sampled topology remain screening-level.",
            "No result is hardware-valid or build-qualified.",
        ],
    }
    dataset, dataset_file_sha = write_sealed_json(
        output / "dataset.json", dataset_payload
    )
    report = _report_text(dataset)
    report_file_sha = _write_bytes_with_sidecar(
        output / "report.md", report.encode("utf-8")
    )
    runtime_payload = {
        "schema_version": f"{SCHEMA_VERSION}.runtime-diagnostics",
        "runtime_policy": RUNTIME_POLICY,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "requested_device": "cuda:0",
        },
        "records": runtime_cases,
    }
    write_sealed_json(output / "runtime-diagnostics.json", runtime_payload)
    deterministic_files = [
        {
            "path": "dataset.json",
            "kind": "sealed_dataset",
            "file_sha256": dataset_file_sha,
            "payload_sha256": dataset["integrity"]["payload_sha256"],
        },
        {
            "path": "report.md",
            "kind": "deterministic_report",
            "file_sha256": report_file_sha,
            "payload_sha256": None,
        },
    ]
    for item in representative_entries:
        for kind in ("geometry", "full_field", "downsampled_field"):
            artifact = item[kind]
            deterministic_files.append(
                {
                    "path": artifact["path"],
                    "kind": kind,
                    "file_sha256": artifact["file_sha256"],
                    "payload_sha256": artifact["payload_sha256"],
                }
            )
    manifest_payload = {
        "schema_version": MANIFEST_VERSION,
        "classification": CLASSIFICATION,
        "dataset_payload_sha256": dataset["integrity"]["payload_sha256"],
        "deterministic_files": deterministic_files,
        "representative_artifacts": representative_entries,
        "runtime_diagnostics": {
            "path": "runtime-diagnostics.json",
            "included_in_deterministic_manifest_hash": False,
            "policy": RUNTIME_POLICY,
        },
    }
    manifest, _ = write_sealed_json(output / "manifest.json", manifest_payload)
    validate_experiment_bundle(output)
    if (
        summary["failed_count"]
        or summary["boundary_gate_failure_count"]
        or summary["solver_gate_failure_count"]
        or summary["parity_failure_count"]
    ):
        raise RuntimeError(
            "predeclared experiment gates failed; inspect the sealed dataset and runtime diagnostics"
        )
    return {"dataset": dataset, "manifest": manifest}


def validate_experiment_bundle(output: Path) -> dict[str, Any]:
    manifest = validate_sealed_file(output / "manifest.json")
    expected_manifest_keys = {
        "schema_version",
        "classification",
        "dataset_payload_sha256",
        "deterministic_files",
        "representative_artifacts",
        "runtime_diagnostics",
        "integrity",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("experiment manifest is not a closed schema")
    if manifest["schema_version"] != MANIFEST_VERSION:
        raise ValueError("unsupported experiment manifest")
    dataset = validate_sealed_file(output / "dataset.json")
    expected_dataset_keys = {
        "schema_version",
        "classification",
        "model_level",
        "sampling",
        "domain",
        "solver",
        "current_equivalent_preview",
        "objectives",
        "constraints",
        "qoi_policy",
        "failure_taxonomy",
        "cases",
        "parity",
        "representatives",
        "summary",
        "limitations",
        "integrity",
    }
    if set(dataset) != expected_dataset_keys:
        raise ValueError("experiment dataset is not a closed schema")
    if (
        dataset["schema_version"] != SCHEMA_VERSION
        or dataset["classification"] != CLASSIFICATION
        or dataset["model_level"] != "L1a"
        or dataset["objectives"] != list(OBJECTIVES)
        or dataset["constraints"] != list(CONSTRAINTS)
        or dataset["failure_taxonomy"] != FAILURE_TAXONOMY
    ):
        raise ValueError("dataset policy/schema binding mismatch")
    if manifest["dataset_payload_sha256"] != dataset["integrity"]["payload_sha256"]:
        raise ValueError("manifest/dataset payload mismatch")
    cases = dataset["cases"]
    if not isinstance(cases, list) or len(cases) != dataset["sampling"]["count"]:
        raise ValueError("dataset case count disagrees with sampling declaration")
    successful = 0
    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or case.get("case_id") in seen_case_ids:
            raise ValueError("dataset case records must be unique objects")
        seen_case_ids.add(case["case_id"])
        if case.get("status") == "failure":
            if "qois" in case or "constraints" in case or case.get("failure") is None:
                raise ValueError("failed cases must not carry fake outcomes")
            continue
        expected_success_keys = {
            "case_id",
            "status",
            "failure",
            "design_id",
            "sampling_provenance",
            "design_values",
            "derived_geometry",
            "geometry_sha256",
            "source_sha256",
            "config_sha256",
            "case_sha256",
            "backend",
            "iterations",
            "qois",
            "constraints",
            "feasible",
            "classification",
        }
        if set(case) != expected_success_keys or case["failure"] is not None:
            raise ValueError("successful case is not a closed record")
        for key in (
            "geometry_sha256",
            "source_sha256",
            "config_sha256",
            "case_sha256",
        ):
            digest = case[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"invalid case digest {key}")
        expected_case_hash = stable_hash(
            {
                "geometry_sha256": case["geometry_sha256"],
                "source_sha256": case["source_sha256"],
                "config_sha256": case["config_sha256"],
            }
        )
        if case["case_sha256"] != expected_case_hash:
            raise ValueError("case hash does not bind geometry/source/config")
        if set(case["constraints"]) != {
            definition["name"] for definition in CONSTRAINTS
        }:
            raise ValueError("case constraint schema mismatch")
        if case["feasible"] is not feasible(case["constraints"]):
            raise ValueError("case feasibility flag is inconsistent")
        successful += 1
    summary = dataset["summary"]
    if (
        summary["evaluated_count"] != successful
        or summary["failed_count"] != len(cases) - successful
        or summary["feasible_count"]
        != sum(case.get("feasible") is True for case in cases)
    ):
        raise ValueError("dataset summary counts are inconsistent")
    if any(not item.get("passed") for item in dataset["parity"]):
        raise ValueError("dataset contains a failed parity gate")
    paths: set[str] = set()
    for entry in manifest["deterministic_files"]:
        if set(entry) != {"path", "kind", "file_sha256", "payload_sha256"}:
            raise ValueError("manifest file entry is not closed")
        if entry["path"] in paths:
            raise ValueError("manifest file paths must be unique")
        paths.add(entry["path"])
        path = output / entry["path"]
        if path.resolve().is_relative_to(output.resolve()) is False:
            raise ValueError("manifest path escapes experiment output")
        digest = _verify_sidecar(path)
        if digest != entry["file_sha256"]:
            raise ValueError(f"manifest file hash mismatch for {entry['path']}")
        if entry["kind"] in {"full_field", "downsampled_field"}:
            validate_field_artifact_file(
                path,
                expected_file_sha256=entry["file_sha256"],
                expected_payload_sha256=entry["payload_sha256"],
            )
        elif entry["kind"] == "geometry":
            geometry = deserialize_geometry(path.read_text(encoding="utf-8"))
            if geometry.canonical_sha256 != entry["payload_sha256"]:
                raise ValueError("geometry payload hash mismatch")
    validate_sealed_file(output / "runtime-diagnostics.json")
    return {"manifest": manifest, "dataset": dataset}
