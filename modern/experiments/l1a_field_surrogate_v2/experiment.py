"""Input-safe geometry freezing and v2 L1a field execution mechanics."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any, Mapping

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    SolverConfig,
    solve_problem_warp,
)
from cft_revival.geometry import (
    EvidenceNote,
    GeometryValidationError,
    PPMStackParameters,
    compute_descriptors,
    generate_twt_inspired_ppm_stack,
    to_l1a_current_equivalent_preview,
)
from cft_revival.optimization.sampling import initial_designs
from experiments.l1a_geometry_sweep_v2.experiment import (
    BuiltCase,
    VARIABLES,
    _material_registry,
    _stable_pitch,
    extract_qois,
)
from experiments.l1a_field_surrogate_v1.experiment import (
    QOIS,
    calibrate,
    coverage_metrics,
    exact_rank,
    field_energy,
    field_vector,
    fit_field_family,
    fit_scalar_family,
    mesh_hash,
    model_metrics,
    numerical_record,
    predict_fields,
    predict_scalars,
    prolong_low,
    topology_match,
    topology_signature,
)
from experiments.l1a_field_surrogate_v1.experiment import (
    sample_designs as v1_designs,
)

from .protocol import PROTOCOL, PROTOCOL_HASH, canonical_hash

HIGH_DOMAIN = AxisymmetricDomain(
    **PROTOCOL["fidelities"]["domain"],
    radial_intervals=PROTOCOL["fidelities"]["high"]["radial_intervals"],
    axial_intervals=PROTOCOL["fidelities"]["high"]["axial_intervals"],
)
LOW_DOMAIN = AxisymmetricDomain(
    **PROTOCOL["fidelities"]["domain"],
    radial_intervals=PROTOCOL["fidelities"]["low"]["radial_intervals"],
    axial_intervals=PROTOCOL["fidelities"]["low"]["axial_intervals"],
)
SOLVER = SolverConfig(**PROTOCOL["fidelities"]["solver"])
INPUT_NAMES = tuple(variable.name for variable in VARIABLES)


def raw_designs() -> tuple[Any, ...]:
    designs = initial_designs(
        VARIABLES,
        PROTOCOL["sampling"]["raw_rows"],
        seed=PROTOCOL["sampling"]["seed"],
        include_boundary_challenges=False,
    )
    prior = {tuple(item.values) for item in v1_designs()}
    overlap = prior.intersection(tuple(item.values) for item in designs)
    if overlap:
        raise RuntimeError("v2 raw candidates overlap v1 coordinates")
    return designs


def design_row(design: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in design.values)


def _geometry_attempt(design: Any, raw_index: int) -> tuple[BuiltCase, dict[str, Any]]:
    values = dict(zip(INPUT_NAMES, design.values, strict=True))
    stage_count = min(5, 3 + int(values["stage_count_selector"] * 3.0))
    pitch = _stable_pitch(values["stage_pitch_m"], stage_count)
    chamber_length = stage_count * pitch
    requested_exit_length = chamber_length * values["exit_length_fraction"]
    exit_length = 0.0 if requested_exit_length < 0.00025 else requested_exit_length
    chamber_radius = values["chamber_outer_radius_m"]
    dielectric = values["dielectric_thickness_m"]
    requested_exit_radius = (
        chamber_radius
        if exit_length == 0.0
        else chamber_radius * (1.15 + 0.35 * values["exit_expansion_descriptor"])
    )
    magnet_inner = (
        max(chamber_radius, requested_exit_radius)
        + dielectric
        + values["radial_clearance_m"]
    )
    magnet_outer = magnet_inner + values["magnet_radial_thickness_m"]
    magnet_axial = pitch * values["magnet_axial_fraction"]
    stage_centres = tuple((index + 0.5) * pitch for index in range(stage_count))
    radius = requested_exit_radius
    attempts: list[dict[str, Any]] = []
    geometry = None
    preview = None
    maximum = PROTOCOL["geometry_preflight"]["maximum_nextafter_steps"]
    for step in range(maximum + 1):
        try:
            geometry = generate_twt_inspired_ppm_stack(
                PPMStackParameters(
                    config_id=f"l1a-fs-v2-raw-{raw_index:03d}",
                    title=f"L1a field-surrogate v2 raw geometry {raw_index:03d}",
                    chamber_inner_radius_m=0.0,
                    chamber_outer_radius_m=chamber_radius,
                    chamber_length_m=chamber_length,
                    injector_length_m=0.08 * chamber_length,
                    dielectric_thickness_m=dielectric,
                    thermal_clearance_m=0.00025,
                    magnet_inner_radius_m=magnet_inner,
                    magnet_outer_radius_m=magnet_outer,
                    stage_pitch_m=pitch,
                    stage_centers_m=stage_centres,
                    magnet_axial_thicknesses_m=(magnet_axial,) * stage_count,
                    shield_outer_radius_m=magnet_outer + 0.00075,
                    yoke_outer_radius_m=magnet_outer + 0.00175,
                    exit_length_m=exit_length,
                    exit_outer_radius_m=radius,
                    first_polarity=1 if values["first_polarity_selector"] < 0.5 else -1,
                    radial_tolerance_m=0.000025,
                    axial_tolerance_m=0.000025,
                    minimum_thickness_m=0.00025,
                    minimum_clearance_m=0.0001,
                ),
                evidence=(
                    EvidenceNote(
                        f"v2-raw-{raw_index:03d}",
                        "assumption",
                        "Fresh preregistered L1a numerical-emulation geometry.",
                        "L1a field-surrogate v2 protocol.",
                    ),
                ),
            )
            preview = to_l1a_current_equivalent_preview(
                geometry,
                material_registry=_material_registry(geometry),
                radial_smear_thickness_m=PROTOCOL["fidelities"]["high"]["radial_source_smear_m"],
            )
            break
        except GeometryValidationError as error:
            message = str(error)
            attempts.append({"step": step, "radius_m": radius, "reason": message})
            if exit_length == 0.0 or not message.startswith("divergent wall "):
                raise
            radius = math.nextafter(radius, chamber_radius)
    if geometry is None or preview is None:
        reason = attempts[-1]["reason"] if attempts else "constructor produced no geometry"
        raise GeometryValidationError(f"nextafter sequence exhausted: {reason}")
    bands = tuple(
        replace(band, ampere_turns_a=band.ampere_turns_a * values["source_strength_scale"])
        for band in preview.bands
    )
    problem = AxisymmetricProblem(f"l1a-fs-v2-{raw_index:03d}", HIGH_DOMAIN, bands)
    descriptors = compute_descriptors(geometry)
    source_payload = {
        **preview.to_dict(),
        "source_strength_scale": values["source_strength_scale"],
        "scaled_bands": [asdict(band) for band in bands],
    }
    source_hash = canonical_hash(source_payload)
    config_hash = canonical_hash(
        {
            "protocol_hash": PROTOCOL_HASH,
            "design_id": design.design_id,
            "domain": PROTOCOL["fidelities"]["domain"],
            "fidelities": PROTOCOL["fidelities"],
            "gates": PROTOCOL["gates"],
        }
    )
    case_hash = canonical_hash(
        {
            "geometry_sha256": geometry.canonical_sha256,
            "source_sha256": source_hash,
            "config_sha256": config_hash,
        }
    )
    derived = {
        "stage_count": stage_count,
        "requested_stage_pitch_m": values["stage_pitch_m"],
        "represented_stage_pitch_m": pitch,
        "stage_centers_m": list(stage_centres),
        "magnet_axial_thickness_m": magnet_axial,
        "magnet_inner_radius_m": magnet_inner,
        "magnet_outer_radius_m": magnet_outer,
        "chamber_length_m": chamber_length,
        "requested_exit_length_m": requested_exit_length,
        "represented_exit_length_m": exit_length,
        "requested_exit_outer_radius_m": requested_exit_radius,
        "represented_exit_outer_radius_m": radius,
        "geometry_descriptors": descriptors.to_dict(),
    }
    case = BuiltCase(
        f"l1a-fs-v2-{raw_index:03d}",
        design,
        geometry,
        preview,
        problem,
        geometry.canonical_sha256,
        source_hash,
        config_hash,
        case_hash,
        derived,
    )
    record = {
        "raw_index": raw_index,
        "design_id": design.design_id,
        "valid": True,
        "attempt_count": len(attempts) + 1,
        "continuity_rejections": attempts,
        "requested_exit_outer_radius_m": requested_exit_radius,
        "represented_exit_outer_radius_m": radius,
        "geometry_sha256": geometry.canonical_sha256,
        "preview_sha256": canonical_hash(preview.to_dict()),
        "source_sha256": source_hash,
    }
    return case, record


def preflight_raw_candidates() -> tuple[list[dict[str, Any]], dict[int, BuiltCase]]:
    records: list[dict[str, Any]] = []
    valid: dict[int, BuiltCase] = {}
    for raw_index, design in enumerate(raw_designs()):
        try:
            case, record = _geometry_attempt(design, raw_index)
            valid[raw_index] = case
            records.append(record)
        except Exception as error:
            records.append(
                {
                    "raw_index": raw_index,
                    "design_id": design.design_id,
                    "valid": False,
                    "rejection_type": type(error).__name__,
                    "rejection_reason": str(error),
                }
            )
    return records, valid


def select_frozen(valid: Mapping[int, BuiltCase]) -> tuple[int, ...]:
    ordered = sorted(valid)
    if len(ordered) < 112:
        raise RuntimeError("geometry preflight produced fewer than 112 valid rows")
    development = ordered[:96]
    pool = ordered[96:]

    def normalized(index: int) -> tuple[float, ...]:
        return tuple(
            (value - variable.lower) / (variable.upper - variable.lower)
            for value, variable in zip(valid[index].design.values, VARIABLES, strict=True)
        )

    boundary = sorted(
        pool,
        key=lambda index: (
            min(min(value, 1.0 - value) for value in normalized(index)),
            valid[index].design.design_id,
        ),
    )[:5]
    remainder = [index for index in pool if index not in boundary]
    candidate_rows = [normalized(index) for index in development[:64]]
    ood = sorted(
        remainder,
        key=lambda index: (
            -min(math.dist(normalized(index), candidate) for candidate in candidate_rows),
            valid[index].design.design_id,
        ),
    )[:5]
    excluded = set(boundary) | set(ood)
    interpolation = [index for index in pool if index not in excluded][:6]
    frozen = tuple(development + interpolation + boundary + ood)
    if len(frozen) != 112 or len(set(frozen)) != 112:
        raise RuntimeError("frozen selection is not exactly 112 unique rows")
    return frozen


def rebuild_frozen(raw_indices: tuple[int, ...]) -> tuple[dict[int, BuiltCase], list[dict[str, Any]]]:
    designs = raw_designs()
    rebuilt: dict[int, BuiltCase] = {}
    records = []
    for frozen_index, raw_index in enumerate(raw_indices):
        case, record = _geometry_attempt(designs[raw_index], raw_index)
        rebuilt[frozen_index] = case
        records.append(
            {
                "frozen_index": frozen_index,
                "raw_index": raw_index,
                "geometry_sha256": case.geometry_sha256,
                "preview_sha256": record["preview_sha256"],
                "source_sha256": case.source_sha256,
            }
        )
    return rebuilt, records


def role_indices(name: str) -> tuple[int, ...]:
    return tuple(range(*PROTOCOL["sampling"]["roles"][name]))


def assessment_groups() -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(range(*bounds))
        for name, bounds in PROTOCOL["sampling"]["assessment_strata"].items()
    }


def high_indices() -> tuple[int, ...]:
    return tuple(range(32)) + tuple(range(64, 112))


def _coarse_case(case: BuiltCase) -> BuiltCase:
    sources = []
    for source in case.problem.sources:
        centre = 0.5 * (source.r_inner_m + source.r_outer_m)
        half = 0.5 * PROTOCOL["fidelities"]["low"]["radial_source_smear_m"]
        sources.append(
            AzimuthalCurrentBand(
                source.name,
                centre - half,
                centre + half,
                source.z_min_m,
                source.z_max_m,
                source.ampere_turns_a,
                source.polarity,
            )
        )
    return replace(
        case,
        case_id=case.case_id + "-coarse",
        problem=AxisymmetricProblem(case.problem.name + "-coarse", LOW_DOMAIN, tuple(sources)),
    )


def solve_frozen_case(case: BuiltCase, fidelity: str) -> tuple[BuiltCase, Any, dict[str, Any]]:
    selected = _coarse_case(case) if fidelity == "low" else case
    if fidelity not in {"low", "high"}:
        raise ValueError("fidelity must be low or high")
    field = solve_problem_warp(
        selected.problem,
        device=PROTOCOL["execution"]["device"],
        config=SOLVER,
    )
    return selected, field, extract_qois(selected, field)
