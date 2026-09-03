"""Design space, case builder and field solves of L1a geometry sweep v3.

The v3 builder is the sweep-v2 builder (``experiments.l1a_geometry_sweep_v2.experiment
.build_case``) with the same eleven design variables and the same derivation rules, on a
wider box: the channel radius extends to 4.2 mm, the pitch down to 3.4 mm, the magnet
radial clearance to 1.6 mm and the magnet radial thickness to 5.0 mm, so that the
wall-radius-to-pitch ratio r_w / L reaches 1.24 (x_w = pi r_w / L up to 3.9) while the
sweep-v2 box (r_w / L = 0.215-0.579) is a strict subset. Geometry schema v1.1 is kept: no
new ring-magnet parameter is needed because the inner radius is already
``r_w + dielectric + clearance`` and the thickness is a variable.

Two design sets:

* ``sobol_v3`` - 128 scrambled-Sobol designs of this box, built here and solved with the
  accepted CPU solver at the sweep-v2 domain/resolution (accepted map) and at 2x
  (refined map);
* ``sweep_v2`` - the 96 accepted sweep-v2 designs, rebuilt and re-solved through the
  identity-proven pipeline of the wall-loss geometry screening
  (``experiments.orbit_wall_loss_geometry_screening_v1.designs``: rebuilt hashes equal the
  sealed raw record, QoIs replay within the sweep-v2 tolerances, stored representatives
  reproduce node-wise). They are the held-out reproduction check of the pipeline.

Everything is L1a linear-vacuum equivalent-current screening: not P2-qualified, not
hardware-valid, not a plasma claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime.canonical import semantic_sha256
from cft_revival.fields import AxisymmetricDomain, AxisymmetricProblem, FieldMap, SolverConfig, solve_problem_cpu
from cft_revival.geometry import (
    EvidenceNote,
    GeometryValidationError,
    PPMStackParameters,
    compute_descriptors,
    generate_twt_inspired_ppm_stack,
    to_l1a_current_equivalent_preview,
)
from cft_revival.magnetics import checked_synthetic_smco_like_magnet
from cft_revival.optimization import Design, Variable

from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.orbit_wall_loss_geometry_screening_v1 import designs as screening_designs

from .sampling import sobol_designs

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
SCREENING_PROTOCOL_PATH = MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "protocol.json"

SET_SOBOL = "sobol_v3"
SET_SWEEP = "sweep_v2"
DESIGN_SETS = (SET_SOBOL, SET_SWEEP)
CASE_PREFIX = "l1a-gs-v3"


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Design space
# --------------------------------------------------------------------------


def variables_from_protocol(value: Mapping[str, Any]) -> tuple[Variable, ...]:
    return tuple(Variable(**item) for item in value["sampling"]["variables"])


def sobol_design_list(value: Mapping[str, Any]) -> tuple[Design, ...]:
    sampling = value["sampling"]
    return sobol_designs(variables_from_protocol(value), int(sampling["design_count"]), seed=int(sampling["seed"]), scramble=bool(sampling["scramble"]))


def design_values(design: Design) -> dict[str, float]:
    return {variable.name: value for variable, value in zip(design.variables, design.values, strict=True)}


def v2_box_contains(values: Mapping[str, float]) -> bool:
    """True iff a v3 design value vector lies inside the sweep-v2 variable box."""

    for variable in sweep.VARIABLES:
        if not variable.lower <= values[variable.name] <= variable.upper:
            return False
    return True


def stage_count_for(selector: float) -> int:
    return min(5, 3 + int(selector * 3.0))


# --------------------------------------------------------------------------
# Case builder (sweep-v2 rules on the v3 box)
# --------------------------------------------------------------------------


def _ulp_close(left: float, right: float, count: float) -> bool:
    return abs(left - right) <= count * max(math.ulp(left), math.ulp(right))


def _stable_pitch(requested: float, stage_count: int, variables: Sequence[Variable]) -> float:
    bounds = next(variable for variable in variables if variable.name == "stage_pitch_m")
    target = 0.5 * (bounds.lower + bounds.upper)
    pitch = requested
    for _ in range(64):
        centres = tuple((index + 0.5) * pitch for index in range(stage_count))
        if all(_ulp_close(right - left, pitch, 2.0) for left, right in zip(centres, centres[1:])):
            return pitch
        pitch = math.nextafter(pitch, target)
    raise GeometryValidationError("pitch could not satisfy the ULP policy")


def _stable_exit_radius(chamber_radius: float, requested: float, dielectric: float, exit_length: float) -> float:
    radius = requested
    for _ in range(128):
        channel_slope = (radius - chamber_radius) / exit_length
        wall_slope = ((radius + dielectric) - (chamber_radius + dielectric)) / exit_length
        if (
            _ulp_close((chamber_radius + dielectric) - chamber_radius, dielectric, 2.0)
            and _ulp_close((radius + dielectric) - radius, dielectric, 2.0)
            and _ulp_close(channel_slope, wall_slope, 4.0)
        ):
            return radius
        radius = math.nextafter(radius, chamber_radius)
    raise GeometryValidationError("exit radius could not satisfy the ULP policy")


LENGTH_QUANTUM_M = 2.0**-40  # 0.91 pm: every radial length is an exact multiple, so radial sums are exact in binary64


def quantize_length(value: float) -> float:
    """Nearest multiple of 2**-40 m (exact in binary64 for lengths below 2**-7 m = 7.8 mm...2**13 quanta of margin)."""

    return round(value / LENGTH_QUANTUM_M) * LENGTH_QUANTUM_M


def derived_geometry_values(values: Mapping[str, float], geometry_policy: Mapping[str, Any], variables: Sequence[Variable]) -> dict[str, Any]:
    """Closed-form derived dimensions (the sweep-v2 rules) before any geometry object exists.

    Binary64 policy (v3, documented in protocol.json#geometry.length_binary64_policy): the
    four radial variables and the pitch are represented by their nearest multiple of
    2**-40 m before the sweep-v2 rules are applied. The sweep-v2 builder required
    ``(r_w + d) - r_w`` to equal ``d`` within 2 ULP for a tapered exit; on the wider v3
    box 22 of the 128 raw Sobol values violate that identity purely by rounding (the sum
    crosses 2**-8 m where the ULP is 8x the dielectric's), which no exit-radius walk can
    repair. With every radial length an exact multiple of 2**-40 m all radial sums,
    differences and stage-centre products are exact, so the v1.1 identities hold with
    zero ULP error; the represented values differ from the requested ones by < 0.5 pm.
    """

    stage_count = stage_count_for(values["stage_count_selector"])
    pitch = _stable_pitch(quantize_length(values["stage_pitch_m"]), stage_count, variables)
    chamber_length = stage_count * pitch
    requested_exit_length = chamber_length * values["exit_length_fraction"]
    exit_length = 0.0 if requested_exit_length < geometry_policy["exit_minimum_length_m"] else requested_exit_length
    chamber_radius = quantize_length(values["chamber_outer_radius_m"])
    dielectric = quantize_length(values["dielectric_thickness_m"])
    clearance = quantize_length(values["radial_clearance_m"])
    thickness = quantize_length(values["magnet_radial_thickness_m"])
    requested_exit_radius = chamber_radius if exit_length == 0.0 else quantize_length(chamber_radius * (1.15 + 0.35 * values["exit_expansion_descriptor"]))
    exit_radius = requested_exit_radius if exit_length == 0.0 else _stable_exit_radius(chamber_radius, requested_exit_radius, dielectric, exit_length)
    magnet_inner = max(chamber_radius, exit_radius) + dielectric + clearance
    magnet_outer = magnet_inner + thickness
    return {
        "stage_count": stage_count,
        "length_quantum_m": LENGTH_QUANTUM_M,
        "requested_stage_pitch_m": values["stage_pitch_m"],
        "represented_stage_pitch_m": pitch,
        "requested_chamber_outer_radius_m": values["chamber_outer_radius_m"],
        "represented_chamber_outer_radius_m": chamber_radius,
        "requested_dielectric_thickness_m": values["dielectric_thickness_m"],
        "represented_dielectric_thickness_m": dielectric,
        "requested_radial_clearance_m": values["radial_clearance_m"],
        "represented_radial_clearance_m": clearance,
        "requested_magnet_radial_thickness_m": values["magnet_radial_thickness_m"],
        "represented_magnet_radial_thickness_m": thickness,
        "stage_centers_m": [(index + 0.5) * pitch for index in range(stage_count)],
        "magnet_axial_thickness_m": pitch * values["magnet_axial_fraction"],
        "magnet_inner_radius_m": magnet_inner,
        "magnet_outer_radius_m": magnet_outer,
        "shield_outer_radius_m": magnet_outer + 7.5e-4,
        "yoke_outer_radius_m": magnet_outer + 1.75e-3,
        "chamber_length_m": chamber_length,
        "requested_exit_length_m": requested_exit_length,
        "represented_exit_length_m": exit_length,
        "requested_exit_outer_radius_m": requested_exit_radius,
        "represented_exit_outer_radius_m": exit_radius,
        "wall_radius_over_pitch": chamber_radius / pitch,
        "x_w": math.pi * chamber_radius / pitch,
        "x_m_inner": math.pi * magnet_inner / pitch,
        "first_polarity": 1 if values["first_polarity_selector"] < 0.5 else -1,
    }


def feasibility_report(values: Mapping[str, float], derived: Mapping[str, Any], geometry_policy: Mapping[str, Any], domain: AxisymmetricDomain, smear_thickness_m: float) -> dict[str, Any]:
    """Constructive feasibility of the geometry v1.1 rules and of the field domain (recorded per design).

    Every check is implied by the variable bounds of the protocol (the box is feasible by
    construction); the report exists so that a future bound change cannot silently produce
    an unbuildable design.
    """

    pitch = derived["represented_stage_pitch_m"]
    pole_gap = pitch * (1.0 - values["magnet_axial_fraction"])
    checks = {
        "magnet_axial_thickness_ge_minimum": derived["magnet_axial_thickness_m"] >= geometry_policy["minimum_thickness_m"],
        "pole_gap_ge_minimum_thickness": pole_gap >= geometry_policy["minimum_thickness_m"],
        "pole_gap_gt_axial_clearance_stack": pole_gap > geometry_policy["minimum_clearance_m"] + 2.0 * geometry_policy["axial_tolerance_m"],
        "radial_clearance_gt_thermal_stack": derived["represented_radial_clearance_m"] > geometry_policy["thermal_clearance_m"] + 2.0 * geometry_policy["radial_tolerance_m"],
        "dielectric_ge_minimum_thickness": derived["represented_dielectric_thickness_m"] >= geometry_policy["minimum_thickness_m"],
        "radial_sums_exact": (derived["represented_chamber_outer_radius_m"] + derived["represented_dielectric_thickness_m"]) - derived["represented_chamber_outer_radius_m"] == derived["represented_dielectric_thickness_m"],
        "yoke_inside_domain": derived["yoke_outer_radius_m"] < domain.radius_m,
        "chamber_inside_domain": domain.z_min_m < 0.0 and derived["chamber_length_m"] < domain.z_max_m,
        "smear_fits_magnet": 2.0 * smear_thickness_m < (derived["magnet_outer_radius_m"] - derived["magnet_inner_radius_m"]),
    }
    return {"checks": checks, "feasible": all(checks.values()), "pole_gap_m": pole_gap}


def build_case(design: Design, index: int, value: Mapping[str, Any]) -> sweep.BuiltCase:
    """Build one v3 case: geometry v1.1, L1a current-equivalent preview, source scale, hashes."""

    values = design_values(design)
    geometry_policy = value["geometry"]
    variables = variables_from_protocol(value)
    derived = derived_geometry_values(values, geometry_policy, variables)
    stage_count = derived["stage_count"]
    pitch = derived["represented_stage_pitch_m"]
    case_id = f"{CASE_PREFIX}-{index:03d}-{design.design_id[:10]}"
    geometry = generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id=f"{case_id}-v1",
            title=f"Preregistered L1a geometry sweep v3 case {index:03d}",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=derived["represented_chamber_outer_radius_m"],
            chamber_length_m=derived["chamber_length_m"],
            injector_length_m=0.08 * derived["chamber_length_m"],
            dielectric_thickness_m=derived["represented_dielectric_thickness_m"],
            thermal_clearance_m=geometry_policy["thermal_clearance_m"],
            magnet_inner_radius_m=derived["magnet_inner_radius_m"],
            magnet_outer_radius_m=derived["magnet_outer_radius_m"],
            stage_pitch_m=pitch,
            stage_centers_m=tuple(derived["stage_centers_m"]),
            magnet_axial_thicknesses_m=(derived["magnet_axial_thickness_m"],) * stage_count,
            shield_outer_radius_m=derived["shield_outer_radius_m"],
            yoke_outer_radius_m=derived["yoke_outer_radius_m"],
            exit_length_m=derived["represented_exit_length_m"],
            exit_outer_radius_m=derived["represented_exit_outer_radius_m"],
            first_polarity=derived["first_polarity"],
            radial_tolerance_m=geometry_policy["radial_tolerance_m"],
            axial_tolerance_m=geometry_policy["axial_tolerance_m"],
            minimum_thickness_m=geometry_policy["minimum_thickness_m"],
            minimum_clearance_m=geometry_policy["minimum_clearance_m"],
        ),
        evidence=(
            EvidenceNote(
                f"v3-case-{index:03d}-screening",
                "assumption",
                "Preregistered bounded L1a field-only design-space sample extending the sweep-v2 box into the HEMP-like wall-radius-to-pitch regime.",
                "Sweep-v3 protocol and preregistration commit.",
            ),
        ),
    )
    descriptors = compute_descriptors(geometry)
    preview = to_l1a_current_equivalent_preview(
        geometry,
        material_registry=sweep._material_registry(geometry),
        radial_smear_thickness_m=value["field"]["preview"]["radial_smear_thickness_m"],
    )
    bands = tuple(replace(band, ampere_turns_a=band.ampere_turns_a * values["source_strength_scale"]) for band in preview.bands)
    domain = AxisymmetricDomain(**value["field"]["domain"])
    problem = AxisymmetricProblem(case_id, domain, bands)
    source_payload = {**preview.to_dict(), "source_strength_scale": values["source_strength_scale"], "scaled_bands": [asdict(band) for band in bands]}
    source_sha256 = _stable_hash(source_payload)
    radial_margin = descriptors.minimum_radial_gap_m - 2.0 * geometry.manufacturing.radial_tolerance_m - geometry.manufacturing.thermal_clearance_m
    axial_margin = descriptors.minimum_axial_gap_m - 2.0 * geometry.manufacturing.axial_tolerance_m - geometry.manufacturing.minimum_clearance_m
    derived_record = {
        **derived,
        "worst_case_radial_manufacturing_margin_m": radial_margin,
        "worst_case_axial_manufacturing_margin_m": axial_margin,
        "geometry_descriptors": descriptors.to_dict(),
        "inside_sweep_v2_box": v2_box_contains(values),
        "feasibility": feasibility_report(values, derived, geometry_policy, domain, value["field"]["preview"]["radial_smear_thickness_m"]),
        "magnet_remanence_t": checked_synthetic_smco_like_magnet().remanence_t(checked_synthetic_smco_like_magnet().reference_temperature_k),
        "magnet_material_id": checked_synthetic_smco_like_magnet().material_id,
    }
    config_payload = {
        "protocol_semantic_sha256": semantic_sha256(value),
        "design_id": design.design_id,
        "domain": value["field"]["domain"],
        "solver": value["field"]["solver"],
        "preview": value["field"]["preview"],
        "refinement": value["field"]["refinement"],
    }
    config_sha256 = _stable_hash(config_payload)
    case_sha256 = _stable_hash({"geometry_sha256": geometry.canonical_sha256, "source_sha256": source_sha256, "config_sha256": config_sha256})
    return sweep.BuiltCase(case_id, design, geometry, preview, problem, geometry.canonical_sha256, source_sha256, config_sha256, case_sha256, derived_record)


# --------------------------------------------------------------------------
# Field solves
# --------------------------------------------------------------------------


def refined_problem(problem: AxisymmetricProblem, refinement: int) -> AxisymmetricProblem:
    domain = problem.domain
    refined = AxisymmetricDomain(domain.radius_m, domain.z_min_m, domain.z_max_m, int(domain.radial_intervals) * int(refinement), int(domain.axial_intervals) * int(refinement))
    return AxisymmetricProblem(problem.name, refined, problem.sources)


def solver_evidence(field: FieldMap) -> dict[str, Any]:
    return {
        "backend": field.diagnostics.backend,
        "iterations": int(field.diagnostics.iterations),
        "converged": bool(field.diagnostics.converged),
        "relative_residual_l2": float(field.diagnostics.relative_residual_l2),
        "flux_reconstruction_identity_t_per_m": float(field.diagnostics.max_flux_reconstruction_identity_t_per_m),
        "true_residual_restarts": int(field.diagnostics.true_residual_restarts),
        "stagnation_detected": bool(field.diagnostics.stagnation_detected),
    }


def field_identity(case: sweep.BuiltCase, value: Mapping[str, Any], role: str, set_id: str) -> str:
    domain = case.problem.domain
    refinement = int(value["field"]["refinement"]) if role == "refined" else 1
    return _stable_hash(
        {
            "set_id": set_id,
            "case_id": case.case_id,
            "case_sha256": case.case_sha256,
            "geometry_sha256": case.geometry_sha256,
            "source_sha256": case.source_sha256,
            "config_sha256": case.config_sha256,
            "role": role,
            "solver": "cft_revival.fields.solve_problem_cpu",
            "solver_config": dict(value["field"]["solver"]),
            "domain": {
                "radius_m": domain.radius_m,
                "z_min_m": domain.z_min_m,
                "z_max_m": domain.z_max_m,
                "radial_intervals": int(domain.radial_intervals) * refinement,
                "axial_intervals": int(domain.axial_intervals) * refinement,
            },
        }
    )


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


@dataclass(frozen=True)
class ResolvedDesign:
    spec: DesignSpec
    case: sweep.BuiltCase
    accepted: FieldMap
    refined: FieldMap
    identity: dict[str, Any]
    evidence: dict[str, Any]
    reference: dict[str, Any]
    solve_seconds: float


def solver_config(value: Mapping[str, Any]) -> SolverConfig:
    """The v3 solver configuration; the protocol declares it equal to the sweep-v2 one."""

    config = SolverConfig(**value["field"]["solver"])
    if asdict(config) != asdict(sweep.SOLVER):
        raise ValueError("v3 solver configuration differs from the sweep-v2 solver configuration")
    return config


def _check_domain(case: sweep.BuiltCase, value: Mapping[str, Any]) -> None:
    domain = case.problem.domain
    declared = value["field"]["domain"]
    if (
        domain.radius_m != declared["radius_m"]
        or domain.z_min_m != declared["z_min_m"]
        or domain.z_max_m != declared["z_max_m"]
        or domain.radial_intervals != declared["radial_intervals"]
        or domain.axial_intervals != declared["axial_intervals"]
    ):
        raise ValueError(f"{case.case_id}: field domain differs from the v3 protocol declaration")
    v2_domain = {key: getattr(sweep.DOMAIN, key) for key in ("radius_m", "z_min_m", "z_max_m", "radial_intervals", "axial_intervals")}
    if v2_domain != dict(declared):
        raise ValueError("v3 field domain differs from the sweep-v2 domain (the v2 QoIs would not be verbatim)")


_SWEEP_BINDING: screening_designs.SweepBinding | None = None


def sweep_binding() -> screening_designs.SweepBinding:
    global _SWEEP_BINDING
    if _SWEEP_BINDING is None:
        screening_protocol = screening_designs.sweep_protocol.strict_json(SCREENING_PROTOCOL_PATH)
        _SWEEP_BINDING = screening_designs.load_sweep_binding(screening_protocol["field_source"])
    return _SWEEP_BINDING


def design_specs(value: Mapping[str, Any]) -> tuple[DesignSpec, ...]:
    """Every declared design of both sets in a fixed order (Sobol first, then sorted sweep ids)."""

    specs: list[DesignSpec] = []
    sets = value["design_sets"]
    if sets[SET_SOBOL]["included"]:
        designs = sobol_design_list(value)
        if len(designs) != int(sets[SET_SOBOL]["design_count"]):
            raise ValueError("Sobol design count differs from the protocol")
        representatives = set(int(item) for item in sets[SET_SOBOL]["representative_indices"])
        for index, design in enumerate(designs):
            case_id = f"{CASE_PREFIX}-{index:03d}-{design.design_id[:10]}"
            specs.append(DesignSpec(SET_SOBOL, case_id, index, index in representatives))
    if sets[SET_SWEEP]["included"]:
        binding = sweep_binding()
        ids = sorted(binding.cases_by_id)
        if len(ids) != int(sets[SET_SWEEP]["design_count"]):
            raise ValueError("sweep-v2 design count differs from the protocol")
        representatives = set(screening_designs.representative_case_ids(binding))
        specs.extend(DesignSpec(SET_SWEEP, case_id, index, case_id in representatives) for index, case_id in enumerate(ids))
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("design keys are not unique")
    return tuple(specs)


def sobol_case(spec: DesignSpec, value: Mapping[str, Any]) -> sweep.BuiltCase:
    designs = sobol_design_list(value)
    design = designs[spec.ordinal]
    case = build_case(design, spec.ordinal, value)
    if case.case_id != spec.design_id:
        raise ValueError(f"{spec.design_id}: rebuilt case id differs ({case.case_id})")
    return case


def resolve_design(spec: DesignSpec, value: Mapping[str, Any]) -> ResolvedDesign:
    """Build (or rebuild with identity proof), solve the accepted and refined maps."""

    refinement = int(value["field"]["refinement"])
    config = solver_config(value)
    started = time.perf_counter()
    if spec.set_id == SET_SOBOL:
        case = sobol_case(spec, value)
        _check_domain(case, value)
        accepted = solve_problem_cpu(case.problem, config)
        refined = solve_problem_cpu(refined_problem(case.problem, refinement), config)
        evidence = {
            "design_values": design_values(case.design),
            "sampling_provenance": case.design.provenance,
            "derived_geometry": dict(case.derived),
            "accepted_solve": solver_evidence(accepted),
            "refined_solve": solver_evidence(refined),
            "identity_proven": True,
            "identity_basis": "case built from the preregistered Sobol design (seed, index) and the protocol; hashes recorded in design-authorities.json",
        }
        reference: dict[str, Any] = {}
    elif spec.set_id == SET_SWEEP:
        binding = sweep_binding()
        recorded = binding.cases_by_id[spec.design_id]
        case = screening_designs.rebuild_case(binding, spec.design_id)
        _check_domain(case, value)
        accepted = solve_problem_cpu(case.problem, config)
        qoi_report = screening_designs.verify_resolved_qois(case, accepted, recorded)
        if not qoi_report["passed"]:
            raise ValueError(f"{spec.design_id}: re-solved QoIs differ from the sealed sweep-v2 record")
        stored = screening_designs.verify_stored_representative(spec.design_id, accepted, value["design_sets"][SET_SWEEP]["stored_map_node_tolerance"])
        if stored is not None and not stored["passed"]:
            raise ValueError(f"{spec.design_id}: re-solved field differs from the stored representative map")
        refined = solve_problem_cpu(refined_problem(case.problem, refinement), config)
        evidence = {
            "design_values": sweep.design_values(case.design),
            "sampling_provenance": case.design.provenance,
            "derived_geometry": dict(case.derived),
            "accepted_solve": solver_evidence(accepted),
            "refined_solve": solver_evidence(refined),
            "qoi_replay": {"passed": qoi_report["passed"], "checks": qoi_report["checks"]},
            "stored_representative": stored,
            "identity_proven": True,
            "identity_basis": "sweep-v2 case rebuilt by the wall-loss geometry screening pipeline; geometry/source/config/case hashes equal the sealed raw record; QoIs replay within the sweep-v2 tolerances",
        }
        reference = {
            "sweep_axis_null_positions_m": list(recorded["qois"]["axis_null_positions_m"]),
            "sweep_axis_bz_peak_positions_m": list(recorded["qois"]["axis_cusp_positions_m"]),
            "sweep_minimum_mirror_ratio": recorded["qois"]["minimum_mirror_ratio"],
            "sweep_recorded_qois": {key: recorded["qois"][key] for key in ("centreline_mid_abs_bz_t", "centreline_abs_bz_peak_t", "field_peak_t", "minimum_mirror_ratio", "stage_gradient_rms_t_per_m", "field_energy_j", "boundary_to_peak_ratio", "topology_confidence")},
        }
    else:
        raise ValueError(f"unknown design set {spec.set_id}")
    solve_seconds = time.perf_counter() - started
    identity = {
        "set_id": spec.set_id,
        "design_id": spec.design_id,
        "case_sha256": case.case_sha256,
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "sampling_design_id": case.design.design_id,
        "accepted_field_identity_sha256": field_identity(case, value, "accepted", spec.set_id),
        "refined_field_identity_sha256": field_identity(case, value, "refined", spec.set_id),
    }
    return ResolvedDesign(spec, case, accepted, refined, identity, evidence, reference, solve_seconds)


def design_identity_without_solving(spec: DesignSpec, value: Mapping[str, Any]) -> dict[str, Any]:
    if spec.set_id == SET_SOBOL:
        case = sobol_case(spec, value)
    elif spec.set_id == SET_SWEEP:
        case = screening_designs.rebuild_case(sweep_binding(), spec.design_id)
    else:
        raise ValueError(f"unknown design set {spec.set_id}")
    values = design_values(case.design)
    return {
        "set_id": spec.set_id,
        "design_id": spec.design_id,
        "sampling_design_id": case.design.design_id,
        "sampling_provenance": case.design.provenance,
        "case_sha256": case.case_sha256,
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "representative": spec.representative,
        "stage_count": int(case.derived["stage_count"]),
        "wall_radius_m": float(case.geometry.chamber.outer_radius_m),
        "stage_pitch_m": float(case.derived["represented_stage_pitch_m"]),
        "x_w": math.pi * float(case.geometry.chamber.outer_radius_m) / float(case.derived["represented_stage_pitch_m"]),
        "inside_sweep_v2_box": v2_box_contains(values) if spec.set_id == SET_SOBOL else True,
    }


def sealed_source_binding() -> dict[str, Any]:
    """Byte identities of the sealed sweep-v2 records the held-out set is bound to."""

    binding = sweep_binding()
    return {
        "sweep_v2": {
            "manifest_file_sha256": binding.manifest_file_sha256,
            "raw_results_file_sha256": binding.raw_file_sha256,
            "summary_file_sha256": binding.summary_file_sha256,
            "preregistration_commit": binding.manifest["preregistration_commit_sha"],
            "protocol_payload_sha256": sweep.PROTOCOL["integrity"]["payload_sha256"],
            "screening_protocol_file_sha256": hashlib.sha256(SCREENING_PROTOCOL_PATH.read_bytes()).hexdigest(),
        }
    }
