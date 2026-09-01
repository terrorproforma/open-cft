"""Preregistered L1a field-only geometry sweep v2 core."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    FieldMap,
    SolverConfig,
    field_artifact,
    source_discretization_diagnostics,
    validate_field_artifact,
)
from cft_revival.geometry import (
    EvidenceNote,
    GeometryValidationError,
    MaterialKind,
    PPMStackParameters,
    compute_descriptors,
    generate_twt_inspired_ppm_stack,
    to_l1a_current_equivalent_preview,
)
from cft_revival.magnetics import (
    LinearPermeability,
    checked_synthetic_smco_like_magnet,
)
from cft_revival.optimization import Design, Variable
from cft_revival.optimization.sampling import initial_designs

from .protocol import load_protocol, stable_hash

PROTOCOL = load_protocol()
CLASSIFICATION = PROTOCOL["classification"]
VARIABLES = tuple(Variable(**item) for item in PROTOCOL["sampling"]["variables"])
DOMAIN = AxisymmetricDomain(**PROTOCOL["field"]["domain"])
SOLVER = SolverConfig(**PROTOCOL["field"]["solver"])
OBJECTIVES = tuple(PROTOCOL["objectives"])
TERMINAL_GATES = tuple(PROTOCOL["terminal_acceptance"]["gates"])
REPRESENTATIVE_ROLES = tuple(PROTOCOL["representative_policy"]["roles"])
PARITY_INDICES = tuple(PROTOCOL["execution"]["parity_case_indices"])
SMEAR_THICKNESS_M = PROTOCOL["field"]["preview"]["radial_smear_thickness_m"]


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


def sample_designs() -> tuple[Design, ...]:
    sampling = PROTOCOL["sampling"]
    return initial_designs(
        VARIABLES,
        PROTOCOL["execution"]["case_count"],
        seed=sampling["seed"],
        include_boundary_challenges=sampling["include_boundary_challenges"],
    )


def design_values(design: Design) -> dict[str, float]:
    return {
        variable.name: value
        for variable, value in zip(design.variables, design.values, strict=True)
    }


def _ulp_close(left: float, right: float, count: float) -> bool:
    return abs(left - right) <= count * max(math.ulp(left), math.ulp(right))


def _stable_pitch(requested: float, stage_count: int) -> float:
    bounds = next(variable for variable in VARIABLES if variable.name == "stage_pitch_m")
    target = 0.5 * (bounds.lower + bounds.upper)
    pitch = requested
    for _ in range(64):
        centres = tuple((index + 0.5) * pitch for index in range(stage_count))
        if all(
            _ulp_close(right - left, pitch, 2.0)
            for left, right in zip(centres, centres[1:])
        ):
            return pitch
        pitch = math.nextafter(pitch, target)
    raise GeometryValidationError("pitch could not satisfy preregistered ULP policy")


def _stable_exit_radius(
    chamber_radius: float,
    requested: float,
    dielectric: float,
    exit_length: float,
) -> float:
    radius = requested
    for _ in range(128):
        channel_slope = (radius - chamber_radius) / exit_length
        wall_slope = (
            (radius + dielectric) - (chamber_radius + dielectric)
        ) / exit_length
        if (
            _ulp_close((chamber_radius + dielectric) - chamber_radius, dielectric, 2.0)
            and _ulp_close((radius + dielectric) - radius, dielectric, 2.0)
            and _ulp_close(channel_slope, wall_slope, 4.0)
        ):
            return radius
        radius = math.nextafter(radius, chamber_radius)
    raise GeometryValidationError("exit radius could not satisfy preregistered ULP policy")


def _material_registry(geometry: Any) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for definition in geometry.materials:
        if definition.category is MaterialKind.PERMANENT_MAGNET:
            material = checked_synthetic_smco_like_magnet()
            if material.material_id != definition.material_id:
                raise GeometryValidationError("permanent-magnet registry ID mismatch")
        else:
            material = LinearPermeability(
                definition.material_id, definition.relative_permeability
            )
        registry[definition.material_id] = material
    return registry


def build_case(design: Design, index: int) -> BuiltCase:
    values = design_values(design)
    geometry_policy = PROTOCOL["geometry"]
    stage_count = min(5, 3 + int(values["stage_count_selector"] * 3.0))
    requested_pitch = values["stage_pitch_m"]
    pitch = _stable_pitch(requested_pitch, stage_count)
    chamber_length = stage_count * pitch
    requested_exit_length = chamber_length * values["exit_length_fraction"]
    exit_length = (
        0.0
        if requested_exit_length < geometry_policy["exit_minimum_length_m"]
        else requested_exit_length
    )
    chamber_radius = values["chamber_outer_radius_m"]
    dielectric = values["dielectric_thickness_m"]
    requested_exit_radius = (
        chamber_radius
        if exit_length == 0.0
        else chamber_radius
        * (1.15 + 0.35 * values["exit_expansion_descriptor"])
    )
    exit_radius = (
        requested_exit_radius
        if exit_length == 0.0
        else _stable_exit_radius(
            chamber_radius, requested_exit_radius, dielectric, exit_length
        )
    )
    magnet_inner = (
        max(chamber_radius, exit_radius)
        + dielectric
        + values["radial_clearance_m"]
    )
    magnet_outer = magnet_inner + values["magnet_radial_thickness_m"]
    magnet_axial_thickness = pitch * values["magnet_axial_fraction"]
    stage_centres = tuple((index + 0.5) * pitch for index in range(stage_count))
    case_id = f"l1a-gs-v2-{index:03d}-{design.design_id[:10]}"
    geometry = generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id=f"{case_id}-v1",
            title=f"Preregistered L1a geometry sweep v2 case {index:03d}",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=chamber_radius,
            chamber_length_m=chamber_length,
            injector_length_m=0.08 * chamber_length,
            dielectric_thickness_m=dielectric,
            thermal_clearance_m=geometry_policy["thermal_clearance_m"],
            magnet_inner_radius_m=magnet_inner,
            magnet_outer_radius_m=magnet_outer,
            stage_pitch_m=pitch,
            stage_centers_m=stage_centres,
            magnet_axial_thicknesses_m=(magnet_axial_thickness,) * stage_count,
            shield_outer_radius_m=magnet_outer + 7.5e-4,
            yoke_outer_radius_m=magnet_outer + 1.75e-3,
            exit_length_m=exit_length,
            exit_outer_radius_m=exit_radius,
            first_polarity=1 if values["first_polarity_selector"] < 0.5 else -1,
            radial_tolerance_m=geometry_policy["radial_tolerance_m"],
            axial_tolerance_m=geometry_policy["axial_tolerance_m"],
            minimum_thickness_m=geometry_policy["minimum_thickness_m"],
            minimum_clearance_m=geometry_policy["minimum_clearance_m"],
        ),
        evidence=(
            EvidenceNote(
                f"v2-case-{index:03d}-screening",
                "assumption",
                "Preregistered bounded L1a field-only design-space sample.",
                "Sweep-v2 protocol and preregistration commit.",
            ),
        ),
    )
    descriptors = compute_descriptors(geometry)
    preview = to_l1a_current_equivalent_preview(
        geometry,
        material_registry=_material_registry(geometry),
        radial_smear_thickness_m=SMEAR_THICKNESS_M,
    )
    bands = tuple(
        replace(
            band,
            ampere_turns_a=band.ampere_turns_a * values["source_strength_scale"],
        )
        for band in preview.bands
    )
    problem = AxisymmetricProblem(case_id, DOMAIN, bands)
    source_payload = {
        **preview.to_dict(),
        "source_strength_scale": values["source_strength_scale"],
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
        "stage_centers_m": list(stage_centres),
        "magnet_axial_thickness_m": magnet_axial_thickness,
        "magnet_inner_radius_m": magnet_inner,
        "magnet_outer_radius_m": magnet_outer,
        "chamber_length_m": chamber_length,
        "requested_exit_length_m": requested_exit_length,
        "represented_exit_length_m": exit_length,
        "requested_exit_outer_radius_m": requested_exit_radius,
        "represented_exit_outer_radius_m": exit_radius,
        "worst_case_radial_manufacturing_margin_m": radial_margin,
        "worst_case_axial_manufacturing_margin_m": axial_margin,
        "geometry_descriptors": descriptors.to_dict(),
    }
    config_payload = {
        "protocol_payload_sha256": PROTOCOL["integrity"]["payload_sha256"],
        "design_id": design.design_id,
        "domain": PROTOCOL["field"]["domain"],
        "solver": PROTOCOL["field"]["solver"],
        "preview": PROTOCOL["field"]["preview"],
        "objectives": PROTOCOL["objectives"],
        "terminal_acceptance": PROTOCOL["terminal_acceptance"],
        "replay_contract": PROTOCOL["replay_contract"],
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
    return max(
        math.hypot(field.b_r_t[i][j], field.b_z_t[i][j])
        for i in range(nr)
        for j in range(nz)
        if i == nr - 1 or j in (0, nz - 1)
    ) / max(peak, 1.0e-300)


def _field_energy(field: FieldMap) -> float:
    dr = field.r_m[1] - field.r_m[0]
    dz = field.z_m[1] - field.z_m[0]
    weighted = 0.0
    for i, radius in enumerate(field.r_m):
        radial_weight = 0.5 if i in (0, len(field.r_m) - 1) else 1.0
        for j in range(len(field.z_m)):
            axial_weight = 0.5 if j in (0, len(field.z_m) - 1) else 1.0
            b2 = field.b_r_t[i][j] ** 2 + field.b_z_t[i][j] ** 2
            weighted += radial_weight * axial_weight * radius * b2
    return math.pi * dr * dz * weighted / 1.2566370614359173e-6


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
                abs(float(item["area_error_m2"])) / max(area, 1e-300),
                abs(float(item["ampere_turn_error_a"])) / max(current, 1e-300),
                math.hypot(
                    float(item["centroid_r_error_m"]),
                    float(item["centroid_z_error_m"]),
                )
                / thickness,
            )
        )
    return max(errors, default=0.0), diagnostics


def extract_qois(case: BuiltCase, field: FieldMap) -> dict[str, Any]:
    screening_artifact = field_artifact(
        case.problem,
        SOLVER,
        field,
        map_stride=8,
        wall_radius_m=case.geometry.chamber.outer_radius_m,
    )
    validate_field_artifact(screening_artifact)
    topology = screening_artifact["summary"]["topology"]
    peak = _field_peak(field)
    boundary_ratio = _boundary_ratio(field, peak)
    source_error, source_diagnostics = _source_error(case)
    chamber = case.geometry.chamber
    axis = field.b_z_t[0]
    targets = {
        "axis_chamber_mid": (0.0, 0.5 * chamber.length_m),
        "axis_injector_exit": (0.0, chamber.injector_length_m),
        "axis_exit_start": (0.0, chamber.exit_start_m),
        "channel_wall_chamber_mid": (
            chamber.outer_radius_m,
            0.5 * chamber.length_m,
        ),
    }
    fixed: dict[str, dict[str, float]] = {}
    for name, (requested_r, requested_z) in targets.items():
        i = _nearest(field.r_m, requested_r)
        j = _nearest(field.z_m, requested_z)
        fixed[name] = {
            "requested_r_m": requested_r,
            "requested_z_m": requested_z,
            "sampled_r_m": field.r_m[i],
            "sampled_z_m": field.z_m[j],
            "b_z_t": field.b_z_t[i][j],
        }
    stage_indices = [
        _nearest(field.z_m, stage.center_z_m) for stage in case.geometry.stages
    ]
    stage_bz = [axis[index] for index in stage_indices]
    gradients = [
        (axis[min(index + 1, len(axis) - 1)] - axis[max(index - 1, 0)])
        / (
            field.z_m[min(index + 1, len(axis) - 1)]
            - field.z_m[max(index - 1, 0)]
        )
        for index in stage_indices
    ]
    topology_tolerance = float(topology["null_tolerance_t"])
    mirror_ratios = []
    for left, right, left_bz, right_bz in zip(
        case.geometry.stages[:-1],
        case.geometry.stages[1:],
        stage_bz[:-1],
        stage_bz[1:],
        strict=True,
    ):
        midpoint = 0.5 * (left.center_z_m + right.center_z_m)
        midpoint_bz = abs(axis[_nearest(field.z_m, midpoint)])
        mirror_ratios.append(
            max(abs(left_bz), abs(right_bz))
            / max(midpoint_bz, topology_tolerance, 1e-300)
        )
    cusp_indices = [
        index
        for index in range(1, len(axis) - 1)
        if 0.0 <= field.z_m[index] <= chamber.length_m
        and abs(axis[index]) >= abs(axis[index - 1])
        and abs(axis[index]) > abs(axis[index + 1])
    ]
    min_pitch = min(stage.pitch_m for stage in case.geometry.stages)
    plateau = 1.0 if topology["axis_plateaus"] else 0.0
    confidence = min(
        1.0,
        max(
            0.0,
            1.0
            - 0.25 * boundary_ratio / 0.05
            - 0.25 * source_error / 0.25
            - 0.25 * DOMAIN.dz_m / min_pitch
            - 0.25 * plateau,
        ),
    )
    gradient_rms = math.sqrt(sum(value * value for value in gradients) / len(gradients))
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
        "axis_null_positions_m": [float(item["z_m"]) for item in topology["axis_nulls"]],
        "topology_status": topology["status"],
        "topology_confidence": confidence,
        "stage_gradients_t_per_m": gradients,
        "stage_gradient_rms_t_per_m": gradient_rms,
        "stage_gradient_max_abs_t_per_m": max(abs(value) for value in gradients),
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


def case_record(case: BuiltCase, field: FieldMap) -> dict[str, Any]:
    qois = extract_qois(case, field)
    return {
        "case_id": case.case_id,
        "status": "success",
        "failure": None,
        "design_id": case.design.design_id,
        "sampling_provenance": case.design.provenance,
        "design_values": design_values(case.design),
        "derived_geometry": dict(case.derived),
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "case_sha256": case.case_sha256,
        "backend": field.diagnostics.backend,
        "iterations": field.diagnostics.iterations,
        "qois": qois,
        "classification": CLASSIFICATION,
    }


def _objective_cmp(left: float, right: float, objective: Mapping[str, Any]) -> int:
    tolerance = max(
        objective["absolute_tolerance"],
        objective["relative_tolerance"] * max(abs(left), abs(right), 1e-300),
    )
    transformed_left = left if objective["direction"] == "maximize" else -left
    transformed_right = right if objective["direction"] == "maximize" else -right
    if transformed_left > transformed_right + tolerance:
        return 1
    if transformed_right > transformed_left + tolerance:
        return -1
    return 0


def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    comparisons = [
        _objective_cmp(
            float(left["qois"][objective["name"]]),
            float(right["qois"][objective["name"]]),
            objective,
        )
        for objective in OBJECTIVES
    ]
    return all(value >= 0 for value in comparisons) and any(
        value > 0 for value in comparisons
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
            other["case_id"] != candidate["case_id"] and dominates(other, candidate)
            for other in successful
        )
    )


def representative_roles(
    front: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    selected: list[dict[str, str]] = []
    for definition in REPRESENTATIVE_ROLES:
        ordered = sorted(front, key=lambda case: case["case_id"])
        function = max if definition["selection"] == "maximum" else min
        candidate = function(
            ordered, key=lambda case: float(case["qois"][definition["qoi"]])
        )
        selected.append({"role": definition["role"], "case_id": candidate["case_id"]})
    return tuple(selected)


def evaluate_terminal_gates(
    cases: Sequence[Mapping[str, Any]],
    parity: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    successful = [case for case in cases if case["status"] == "success"]
    gate_results: list[dict[str, Any]] = []
    for definition in TERMINAL_GATES:
        gate_id = definition["gate_id"]
        failed: list[str] = []
        observed: Any
        if gate_id == "cpu_cuda_parity":
            limits = definition["limits"]
            by_case = {item["case_id"]: item for item in parity}
            expected = [
                cases[index]["case_id"]
                for index in PARITY_INDICES
                if cases[index]["status"] == "success"
            ]
            for case_id in expected:
                item = by_case.get(case_id)
                if item is None or (
                    item["differences"]["psi_scale_relative"] > limits["psi"]
                    or item["differences"]["br_scale_relative"] > limits["br"]
                    or item["differences"]["bz_scale_relative"] > limits["bz"]
                ):
                    failed.append(case_id)
            observed = {
                name: max(
                    (item["differences"][f"{name}_scale_relative"] for item in parity),
                    default=float("inf"),
                )
                for name in ("psi", "br", "bz")
            }
        elif gate_id == "manufacturability":
            values = {
                case["case_id"]: min(
                    case["derived_geometry"][
                        "worst_case_radial_manufacturing_margin_m"
                    ],
                    case["derived_geometry"][
                        "worst_case_axial_manufacturing_margin_m"
                    ],
                )
                for case in successful
            }
            failed = [
                case_id
                for case_id, value in values.items()
                if value < definition["limit"]
            ]
            observed = min(values.values(), default=float("-inf"))
        else:
            qoi_name = definition["metric"]
            values = {
                case["case_id"]: float(case["qois"][qoi_name])
                for case in successful
            }
            if definition["comparator"] == "<=":
                failed = [
                    case_id
                    for case_id, value in values.items()
                    if value > definition["limit"]
                ]
                observed = max(values.values(), default=float("inf"))
            else:
                failed = [
                    case_id
                    for case_id, value in values.items()
                    if value < definition["limit"]
                ]
                observed = min(values.values(), default=float("-inf"))
        gate_results.append(
            {
                "gate_id": gate_id,
                "definition": definition,
                "observed": observed,
                "failed_case_ids": sorted(failed),
                "failure_count": len(failed),
                "passed": not failed,
            }
        )
    return tuple(gate_results)
