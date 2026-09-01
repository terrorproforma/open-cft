"""Development-only topology-targeted L1a field/coupling/global-plasma search.

This v1 search was not preregistered and used coupling v2's now-deprecated
same-z mirror proxy with roundoff-scale null lows.  Its preserved numerical
outputs are development evidence only: residual roots are non-identifiable
screening-equation diagnostics, not physical mirror or performance results.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.coupling import (
    AcceptedArtifactClaims,
    AdapterVersionContract,
    CandidateKind,
    CouplingValidationError,
    MapValidationPolicy,
    ProfileRole,
    SolverDiagnosticsEvidence,
    TopologyPolicy,
    TopologyStatus,
    UncertaintyModel,
    build_coupling_record,
    coupling_record_dict,
    global_solver_inputs,
    hash_axisymmetric_map,
    source_map_binding_hash,
    verify_accepted_field_artifact,
)
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
from cft_revival.plasma import (
    PlasmaMultiStartResult,
    PlasmaState,
    SolverOptions,
    XenonGlobalInputs,
    solve_global_discharge_multistart,
)

SCHEMA_VERSION = "cft-revival.experiment.four-cell-topology-search/1.0.1"
MANIFEST_VERSION = "cft-revival.experiment.four-cell-topology-search-manifest/1.0.1"
CLASSIFICATION = (
    "DEVELOPMENT_EVIDENCE_INVALID_FOR_PHYSICAL_MIRROR_OR_PERFORMANCE_CLAIMS"
)
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
DEFAULT_CASE_COUNT = 128
DEFAULT_SEED = 20260902
ACCEPTANCE_TIME_UTC = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
STATE_DIMENSION = 25
PROTOCOL_STATUS = {
    "experiment_version": "v1",
    "status": "development_evidence_only",
    "preregistered": False,
    "valid_for_physical_mirror_claims": False,
    "valid_for_performance_claims": False,
    "valid_for_identifiable_state_claims": False,
    "permitted_use": "audit-preserved non-identifiable screening-equation diagnostics",
    "invalidity_reasons": [
        "the search protocol and gates were not preregistered before execution",
        "coupling v2 used the deprecated same-z centreline-low/wall-high mirror proxy",
        "selected null lows are roundoff-scale and not physical mirror minima",
    ],
    "supersession_required": {
        "coupling_model": "coupling v3",
        "search_protocol": "preregistered four-cell topology search v2",
        "requirement": (
            "both coupling v3 and a preregistered v2 search are required before "
            "physical mirror, identifiable-state, or performance claims"
        ),
    },
}

VARIABLES = (
    Variable("stage_count_selector", 0.0, 1.0, "1"),
    Variable("stage_pitch_m", 0.0045, 0.0070, "m"),
    Variable("stack_axial_offset_fraction", 0.52, 0.90, "1"),
    Variable("magnet_axial_fraction", 0.42, 0.66, "1"),
    Variable("chamber_wall_radius_m", 0.0014, 0.0024, "m"),
    Variable("dielectric_thickness_m", 0.0005, 0.0010, "m"),
    Variable("radial_clearance_m", 0.00035, 0.00085, "m"),
    Variable("magnet_radial_thickness_m", 0.0025, 0.0050, "m"),
    Variable("source_strength_scale", 0.75, 1.30, "1"),
    Variable("alternating_strength_ratio", 0.20, 0.55, "1"),
    Variable("first_polarity_selector", 0.0, 1.0, "1"),
    Variable("upstream_padding_pitch", 2.0, 4.5, "1"),
    Variable("downstream_padding_pitch", 2.0, 4.5, "1"),
    Variable("radial_padding_m", 0.0050, 0.0100, "m"),
)

SOLVER = SolverConfig(
    relative_tolerance=1.0e-10,
    absolute_tolerance=1.0e-13,
    max_iterations=20_000,
    residual_history_stride=10,
    max_true_residual_restarts=2,
)
RADIAL_INTERVALS = 64
AXIAL_INTERVALS = 192
SMEAR_THICKNESS_M = 9.0e-4

UNCERTAINTY = UncertaintyModel(
    absolute_independent_sigma_t=2.0e-5,
    relative_independent_sigma=0.01,
    common_mode_sigma_t=1.0e-5,
    residual_correlation=0.25,
    coverage_factor=2.0,
)
TOPOLOGY_POLICY = TopologyPolicy(
    relative_value_tolerance=1.0e-8,
    absolute_value_tolerance_t=1.0e-18,
    null_relative_tolerance=1.0e-7,
    null_absolute_tolerance_t=1.0e-18,
    minimum_prominence_relative=1.0e-5,
    minimum_prominence_sigma=2.0,
    minimum_candidate_confidence=0.95,
    minimum_segment_confidence=0.50,
    report_boundary_extrema=True,
    allow_boundary_minima_as_cusps=False,
)
FIELD_GATES = {
    "relative_residual_l2_max": 1.0e-10,
    "flux_reconstruction_identity_t_per_m_max": 1.0e-8,
    "boundary_to_peak_ratio_max": 0.05,
    "source_representation_error_max": 0.25,
}
TOPOLOGY_GATES = {
    "required_status": "resolved",
    "required_segment_count": 4,
    "required_order": "strictly increasing upstream-to-downstream",
    "boundary_exclusion_grid_cells": 2,
    "required_inner_role": "centreline",
    "required_wall_role": "wall",
    "minimum_candidate_confidence": 0.95,
    "minimum_segment_confidence": 0.50,
    "minimum_prominence_sigma": 2.0,
    "mirror_ratio": "finite and >= 1; wall high field strictly exceeds positive cusp low field",
    "probability": "finite in [0, 1)",
}
PLASMA_OPTIONS = SolverOptions(
    max_iterations=250,
    residual_tolerance=1.0e-9,
    gradient_tolerance=1.0e-10,
    step_tolerance=1.0e-12,
    initial_damping=1.0e-3,
)
PLASMA_START_COUNT = 9
OPERATING_POINTS = (
    {"operating_point_id": "hypothetical-300V-1A", "anode_voltage_v": 300.0, "anode_current_a": 1.0},
    {"operating_point_id": "hypothetical-500V-1p5A", "anode_voltage_v": 500.0, "anode_current_a": 1.5},
    {"operating_point_id": "hypothetical-1000V-1A", "anode_voltage_v": 1000.0, "anode_current_a": 1.0},
)
FAILURE_TAXONOMY = {
    "GEOMETRY_INVALID": "accepted geometry v1.1 construction or strict validation failed",
    "SOURCE_INVALID": "L1a current-equivalent source construction failed",
    "FIELD_SOLVER_FAILURE": "Warp field solve or strict L1a artifact validation failed",
    "FIELD_GATE_FAILURE": "field residual, identity, source, or boundary gate failed",
    "COUPLING_REJECTED": "coupling v2 rejected evidence or mirror projection",
    "TOPOLOGY_STATUS": "coupling topology was not resolved",
    "TOPOLOGY_COUNT": "resolved coupling topology did not contain exactly four segments",
    "BOUNDARY_LEAKAGE": "a selected cusp depended on a finite-domain boundary sample or margin",
    "ROLE_MISMATCH": "centreline or wall profile role/radius was invalid",
    "ORDERING_FAILURE": "segments or cusps were not strictly ordered and contiguous",
    "MIRROR_INVERTED": "wall/cusp fields or mirror ratio were inverted, zero, or nonfinite",
    "CONFIDENCE_FAILURE": "candidate, prominence, segment, or overall confidence failed",
    "PROBABILITY_INVALID": "coupling-v2 loss probability was nonfinite or outside [0,1)",
    "PARITY_FAILURE": "selected CPU/Warp parity case exceeded the declared scale-relative gates",
    "PLASMA_NONCONVERGENCE": (
        "strict deterministic global-plasma multi-start found no residual root"
    ),
}

RESIDUAL_NAMES = (
    "cathode-emission-current",
    "electron-transport-cell-1",
    "electron-transport-cell-2",
    "electron-transport-cell-3",
    "ionization-source-cell-1",
    "ionization-source-cell-2",
    "ionization-source-cell-3",
    "ionization-source-cell-4",
    "ion-transport-cell-1",
    "ion-transport-cell-2",
    "ion-transport-cell-3",
    "anode-ion-current",
    "electron-power-transport-cell-2",
    "electron-power-transport-cell-3",
    "electron-power-transport-cell-4",
    "interface-current-0",
    "interface-current-1",
    "interface-current-2",
    "interface-current-3",
    "interface-current-4",
    "cusp-current-1",
    "cusp-current-2",
    "cusp-current-3",
    "cell-energy-1",
    "cell-energy-2",
    "cell-energy-3",
    "cell-energy-4",
    "global-energy",
)


@dataclass(frozen=True)
class SerializedMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class BuiltCase:
    case_id: str
    design: Design
    geometry: Any
    problem: AxisymmetricProblem
    geometry_sha256: str
    source_sha256: str
    config_sha256: str
    case_sha256: str
    values: Mapping[str, float]
    derived: Mapping[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _json_value(dict(payload))
    return {
        **body,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": CANONICALIZATION,
            "payload_sha256": stable_hash(body),
        },
    }


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def write_sealed_json(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value = _seal(payload)
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return value, _write_bytes(path, data)


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _verify_sidecar(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = f"{digest}  {path.name}\n"
    if path.with_name(path.name + ".sha256").read_text(encoding="ascii") != expected:
        raise ValueError(f"invalid SHA-256 sidecar for {path.name}")
    return digest


def load_sealed_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    value = _strict_json(path)
    integrity = value.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError(f"{path.name} has no integrity object")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("payload_sha256") != stable_hash(payload)
    ):
        raise ValueError(f"{path.name} payload integrity mismatch")
    return value


def sample_designs(
    count: int = DEFAULT_CASE_COUNT, seed: int = DEFAULT_SEED
) -> tuple[Design, ...]:
    if count < 1 or count > 256:
        raise ValueError("count must be in [1, 256]")
    return initial_designs(
        VARIABLES,
        count,
        seed=seed,
        include_boundary_challenges=False,
    )


def _design_values(design: Design) -> dict[str, float]:
    return {
        variable.name: value
        for variable, value in zip(design.variables, design.values, strict=True)
    }


def _stable_pitch_and_centres(
    requested: float, stage_count: int, offset_fraction: float
) -> tuple[float, tuple[float, ...]]:
    pitch = requested
    target = 0.5 * (
        next(item.lower for item in VARIABLES if item.name == "stage_pitch_m")
        + next(item.upper for item in VARIABLES if item.name == "stage_pitch_m")
    )
    for _ in range(64):
        first = pitch * offset_fraction
        centres = tuple(first + index * pitch for index in range(stage_count))
        if all(
            abs((right - left) - pitch)
            <= 2.0 * max(math.ulp(right - left), math.ulp(pitch))
            for left, right in zip(centres, centres[1:])
        ):
            return pitch, centres
        pitch = math.nextafter(pitch, target)
    raise GeometryValidationError("could not represent a contract-stable pitch")


def _stable_grid_upper(lower: float, requested: float, intervals: int) -> float:
    upper = requested
    for _ in range(256):
        spacing = (upper - lower) / intervals
        if lower + intervals * spacing == upper:
            return upper
        upper = math.nextafter(upper, lower)
    raise GeometryValidationError("could not represent a contract-stable grid extent")


def _materials(geometry: Any) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for material in geometry.materials:
        if material.category is MaterialKind.PERMANENT_MAGNET:
            resolved = checked_synthetic_smco_like_magnet()
            if resolved.material_id != material.material_id:
                raise GeometryValidationError("permanent-magnet registry ID mismatch")
        else:
            resolved = LinearPermeability(
                material.material_id, material.relative_permeability
            )
        registry[material.material_id] = resolved
    return registry


def build_case(design: Design, index: int) -> BuiltCase:
    values = _design_values(design)
    stage_count = min(6, 4 + int(values["stage_count_selector"] * 3.0))
    pitch, centres = _stable_pitch_and_centres(
        values["stage_pitch_m"],
        stage_count,
        values["stack_axial_offset_fraction"],
    )
    first_centre = centres[0]
    magnet_thickness = pitch * values["magnet_axial_fraction"]
    downstream_lead = pitch * (1.42 - values["stack_axial_offset_fraction"])
    chamber_length = centres[-1] + downstream_lead
    chamber_radius = values["chamber_wall_radius_m"]
    dielectric = values["dielectric_thickness_m"]
    magnet_inner = chamber_radius + dielectric + values["radial_clearance_m"]
    magnet_outer = magnet_inner + values["magnet_radial_thickness_m"]
    case_id = f"four-cell-{index:03d}-{design.design_id[:10]}"
    geometry = generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id=f"{case_id}-geometry-v1",
            title=f"Four-cell topology search case {index:03d}",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=chamber_radius,
            chamber_length_m=chamber_length,
            injector_length_m=min(0.08 * chamber_length, 0.5 * first_centre),
            dielectric_thickness_m=dielectric,
            thermal_clearance_m=2.5e-4,
            magnet_inner_radius_m=magnet_inner,
            magnet_outer_radius_m=magnet_outer,
            stage_pitch_m=pitch,
            stage_centers_m=centres,
            magnet_axial_thicknesses_m=(magnet_thickness,) * stage_count,
            shield_outer_radius_m=magnet_outer + 7.5e-4,
            yoke_outer_radius_m=magnet_outer + 1.75e-3,
            first_polarity=1 if values["first_polarity_selector"] < 0.5 else -1,
            radial_tolerance_m=2.5e-5,
            axial_tolerance_m=2.5e-5,
            minimum_thickness_m=2.5e-4,
            minimum_clearance_m=1.0e-4,
        ),
        evidence=(
            EvidenceNote(
                f"{case_id}-search",
                "assumption",
                "Deterministic topology-targeted L1a screening case.",
                "Experiment-local shifted-Halton design declaration.",
            ),
        ),
    )
    preview = to_l1a_current_equivalent_preview(
        geometry,
        material_registry=_materials(geometry),
        radial_smear_thickness_m=SMEAR_THICKNESS_M,
    )
    scale = values["source_strength_scale"]
    alternating = values["alternating_strength_ratio"]
    bands = tuple(
        replace(
            band,
            ampere_turns_a=band.ampere_turns_a
            * scale
            * (alternating if (band_index // 2) % 2 else 1.0),
        )
        for band_index, band in enumerate(preview.bands)
    )
    z_min = -values["upstream_padding_pitch"] * pitch
    radius = _stable_grid_upper(
        0.0,
        magnet_outer + values["radial_padding_m"],
        RADIAL_INTERVALS,
    )
    z_max = _stable_grid_upper(
        z_min,
        chamber_length + values["downstream_padding_pitch"] * pitch,
        AXIAL_INTERVALS,
    )
    domain = AxisymmetricDomain(
        radius_m=radius,
        z_min_m=z_min,
        z_max_m=z_max,
        radial_intervals=RADIAL_INTERVALS,
        axial_intervals=AXIAL_INTERVALS,
    )
    problem = AxisymmetricProblem(case_id, domain, bands)
    source_payload = {
        "preview": preview.to_dict(),
        "scaled_sources": [asdict(source) for source in bands],
        "strength_scale": scale,
        "alternating_strength_ratio": alternating,
    }
    source_sha = stable_hash(source_payload)
    config_payload = {
        "domain": asdict(domain),
        "solver": asdict(SOLVER),
        "wall_radius_m": chamber_radius,
        "topology_policy": asdict(TOPOLOGY_POLICY),
        "uncertainty": asdict(UNCERTAINTY),
        "field_gates": FIELD_GATES,
        "topology_gates": TOPOLOGY_GATES,
    }
    config_sha = stable_hash(config_payload)
    case_sha = stable_hash(
        {
            "geometry_sha256": geometry.canonical_sha256,
            "source_sha256": source_sha,
            "config_sha256": config_sha,
        }
    )
    derived = {
        "stage_count": stage_count,
        "represented_stage_pitch_m": pitch,
        "stage_centres_m": list(centres),
        "stack_axial_offset_m": first_centre,
        "magnet_axial_thickness_m": magnet_thickness,
        "chamber_length_m": chamber_length,
        "chamber_wall_radius_m": chamber_radius,
        "magnet_inner_radius_m": magnet_inner,
        "magnet_outer_radius_m": magnet_outer,
        "domain": asdict(domain),
        "source_polarities": [source.polarity for source in bands],
        "source_ampere_turns_a": [source.ampere_turns_a for source in bands],
        "source_stage_strength_scales": [
            scale * (alternating if stage % 2 else 1.0)
            for stage in range(stage_count)
        ],
    }
    return BuiltCase(
        case_id,
        design,
        geometry,
        problem,
        geometry.canonical_sha256,
        source_sha,
        config_sha,
        case_sha,
        values,
        derived,
    )


def _artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _producer_code_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    relative = (
        "src/cft_revival/fields/artifacts.py",
        "src/cft_revival/fields/numerics.py",
        "src/cft_revival/fields/warp_solver.py",
        "src/cft_revival/coupling/records.py",
        "src/cft_revival/coupling/topology.py",
        "src/cft_revival/plasma/solver.py",
        "experiments/four_cell_topology_search/experiment.py",
    )
    return stable_hash(
        [
            {
                "path": item,
                "sha256": hashlib.sha256((root / item).read_bytes()).hexdigest(),
            }
            for item in relative
        ]
    )


class GeneratedL1aAdapter:
    """Accept one exact, strict experiment-produced L1a artifact."""

    adapter_id = "experiments.four-cell-topology-search.generated-l1a-v1.1"
    version_contract = AdapterVersionContract(
        contract_id="four-cell-search-direct-l1a-adapter",
        contract_version="1.0.0",
        input_schema_version="cft-axisymmetric-field-map/1.1.0",
        normalized_schema_version="cft-axisymmetric-field-map/1.1.0",
        model_level="L1a",
    )

    def __init__(self, expected_artifact_hash: str, expected_case: BuiltCase) -> None:
        self.expected_artifact_hash = expected_artifact_hash
        self.expected_case = expected_case
        self.adapter_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def verify_artifact(self, artifact_bytes: bytes) -> AcceptedArtifactClaims:
        if hashlib.sha256(artifact_bytes).hexdigest() != self.expected_artifact_hash:
            raise CouplingValidationError("artifact bytes differ from accepted case bytes")
        artifact = json.loads(artifact_bytes)
        validate_field_artifact(artifact)
        if artifact["input"]["name"] != self.expected_case.case_id:
            raise CouplingValidationError("artifact/case identity mismatch")
        if artifact["input"]["sources"] != [
            asdict(source) for source in self.expected_case.problem.sources
        ]:
            raise CouplingValidationError("artifact/source identity mismatch")
        raw = artifact["field_map"]
        field = SerializedMap(
            tuple(raw["r_m"]),
            tuple(raw["z_m"]),
            tuple(tuple(row) for row in raw["b_r_t"]),
            tuple(tuple(row) for row in raw["b_z_t"]),
        )
        map_hash = hash_axisymmetric_map(
            field.r_m, field.z_m, field.b_r_t, field.b_z_t
        )
        source_hash = stable_hash(
            {
                "sources": artifact["input"]["sources"],
                "source_convention": artifact["input"]["source_convention"],
            }
        )
        diagnostics = artifact["diagnostics"]
        solver = artifact["input"]["solver"]
        residual_tolerance = max(
            solver["absolute_tolerance"],
            solver["relative_tolerance"] * diagnostics["initial_residual_l2"],
        )
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        producer_hash = _producer_code_hash()
        return AcceptedArtifactClaims(
            field_map=field,
            artifact_schema_version=artifact["schema_version"],
            model_level=artifact["model_level"],
            artifact_hash=artifact_hash,
            map_content_hash=map_hash,
            source_hash=source_hash,
            source_map_binding_hash=source_map_binding_hash(
                map_hash, source_hash, artifact_hash
            ),
            backend_id=f"cft_revival.fields/{diagnostics['backend']}",
            backend_version="warp-1.14-artifact-v1.1",
            field_model_id=self.expected_case.case_id,
            field_model_hash=stable_hash(
                {
                    "model_description": artifact["model_description"],
                    "provenance": artifact["provenance"],
                    "producer_code_hash": producer_hash,
                }
            ),
            code_hash=producer_hash,
            config_hash=self.expected_case.config_sha256,
            generated_at_utc=ACCEPTANCE_TIME_UTC,
            diagnostics=SolverDiagnosticsEvidence(
                converged=diagnostics["converged"],
                residual_norm=diagnostics["final_residual_l2"],
                residual_tolerance=residual_tolerance,
                relative_residual=diagnostics["relative_residual_l2"],
                relative_tolerance=solver["relative_tolerance"],
                iterations=diagnostics["iterations"],
            ),
        )


def _field_quality(case: BuiltCase, field: FieldMap) -> dict[str, float]:
    peak = max(
        math.hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    boundary = max(
        math.hypot(field.b_r_t[i][j], field.b_z_t[i][j])
        for i in range(len(field.r_m))
        for j in range(len(field.z_m))
        if i == len(field.r_m) - 1 or j in (0, len(field.z_m) - 1)
    )
    source_errors: list[float] = []
    for source, item in zip(
        case.problem.sources,
        source_discretization_diagnostics(case.problem),
        strict=True,
    ):
        area = float(item["requested_area_m2"])
        current = abs(float(item["requested_signed_ampere_turns_a"]))
        thickness = min(
            source.r_outer_m - source.r_inner_m,
            source.z_max_m - source.z_min_m,
        )
        source_errors.extend(
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
    return {
        "field_peak_t": peak,
        "boundary_to_peak_ratio": boundary / max(peak, 1.0e-300),
        "source_representation_error": max(source_errors, default=0.0),
        "relative_residual_l2": field.diagnostics.relative_residual_l2,
        "flux_reconstruction_identity_t_per_m": (
            field.diagnostics.max_flux_reconstruction_identity_t_per_m
        ),
    }


def evaluate_field_gates(quality: Mapping[str, float]) -> dict[str, bool]:
    return {
        "relative_residual": quality["relative_residual_l2"]
        <= FIELD_GATES["relative_residual_l2_max"],
        "flux_identity": quality["flux_reconstruction_identity_t_per_m"]
        <= FIELD_GATES["flux_reconstruction_identity_t_per_m_max"],
        "boundary": quality["boundary_to_peak_ratio"]
        <= FIELD_GATES["boundary_to_peak_ratio_max"],
        "source": quality["source_representation_error"]
        <= FIELD_GATES["source_representation_error_max"],
    }


def evaluate_topology_gates(
    record: Any, *, axial_sample_count: int | None = None
) -> dict[str, bool]:
    """Apply only gates implied by coupling-v2 and the fixed four-cell model."""

    segments = tuple(record.segments)
    cusps = [segment.mirror_loss.cusp for segment in segments]
    domain_start = min(
        (segment.z_start_m for segment in segments), default=float("-inf")
    )
    domain_end = max(
        (segment.z_end_m for segment in segments), default=float("inf")
    )
    boundary_cells = int(TOPOLOGY_GATES["boundary_exclusion_grid_cells"])
    no_boundary_samples = all(
        cusp.sample_indices
        and min(cusp.sample_indices) >= boundary_cells
        and (
            axial_sample_count is None
            or max(cusp.sample_indices)
            <= axial_sample_count - boundary_cells - 1
        )
        for cusp in cusps
    )
    # The index upper bound is checked from each candidate's bracket against the
    # record domain because coupling records intentionally omit the raw grid.
    no_boundary_brackets = all(
        cusp.bracket_z_m[0] > domain_start and cusp.bracket_z_m[1] < domain_end
        for cusp in cusps
    )
    ordered = all(
        right.representative_cusp_z_m > left.representative_cusp_z_m
        for left, right in zip(segments, segments[1:])
    )
    contiguous = all(
        left.z_end_m == right.z_start_m
        for left, right in zip(segments, segments[1:])
    )
    mirrors = all(
        segment.mirror_loss.mirror_ratio_high_to_low is not None
        and math.isfinite(segment.mirror_loss.mirror_ratio_high_to_low)
        and segment.mirror_loss.mirror_ratio_high_to_low >= 1.0
        and math.isfinite(segment.mirror_loss.field_ratio_low_to_high)
        and 0.0 < segment.mirror_loss.field_ratio_low_to_high <= 1.0
        and segment.mirror_loss.wall_b_t
        > segment.mirror_loss.cusp.b_magnitude_t
        > 0.0
        for segment in segments
    )
    confidence = all(
        cusp.confidence >= TOPOLOGY_POLICY.minimum_candidate_confidence
        and cusp.prominence_t
        >= TOPOLOGY_POLICY.minimum_prominence_sigma * cusp.sigma_b_t
        and segment.confidence >= TOPOLOGY_POLICY.minimum_segment_confidence
        for cusp, segment in zip(cusps, segments, strict=True)
    ) and (
        not segments
        or record.overall_confidence >= TOPOLOGY_POLICY.minimum_segment_confidence
    )
    probabilities = all(
        math.isfinite(segment.mirror_loss.probability.value)
        and 0.0 <= segment.mirror_loss.probability.value < 1.0
        for segment in segments
    )
    return {
        "status": record.topology_status is TopologyStatus.RESOLVED,
        "segment_count": len(segments) == 4,
        "no_boundary_leakage": no_boundary_samples and no_boundary_brackets,
        "roles": (
            record.inner_profile_role is ProfileRole.CENTRELINE
            and record.inner_profile.role is ProfileRole.CENTRELINE
            and record.inner_profile_radius_m == 0.0
            and record.wall.role is ProfileRole.WALL
            and record.wall_radius_m > 0.0
        ),
        "ordered_contiguous": ordered and contiguous,
        "mirror_ordering": mirrors,
        "confidence_prominence": confidence,
        "probabilities": probabilities,
    }


def topology_compatible(gates: Mapping[str, bool]) -> bool:
    return bool(gates) and all(gates.values())


def _failure_codes(
    field_gates: Mapping[str, bool], topology_gates: Mapping[str, bool]
) -> list[str]:
    result = []
    if not all(field_gates.values()):
        result.append("FIELD_GATE_FAILURE")
    mapping = {
        "status": "TOPOLOGY_STATUS",
        "segment_count": "TOPOLOGY_COUNT",
        "no_boundary_leakage": "BOUNDARY_LEAKAGE",
        "roles": "ROLE_MISMATCH",
        "ordered_contiguous": "ORDERING_FAILURE",
        "mirror_ordering": "MIRROR_INVERTED",
        "confidence_prominence": "CONFIDENCE_FAILURE",
        "probabilities": "PROBABILITY_INVALID",
    }
    result.extend(code for gate, code in mapping.items() if not topology_gates.get(gate, False))
    return result


def _residual_rows(
    normalized: Sequence[float], raw: Sequence[float] | None
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "name": name,
            "normalized": float(normalized[index]),
            "raw_si": None if raw is None else float(raw[index]),
        }
        for index, name in enumerate(RESIDUAL_NAMES)
    ]


def serialize_plasma(result: PlasmaMultiStartResult) -> dict[str, Any]:
    attempts = []
    for index, attempt in enumerate(result.attempts):
        diagnostics = attempt.diagnostics
        raw = None if attempt.evaluation is None else attempt.evaluation.raw
        residual_root = attempt.state is not None and diagnostics.converged
        attempts.append(
            {
                "start_index": index,
                "residual_root_found": residual_root,
                "outcome_classification": (
                    "non_identifiable_screening_equation_residual_root"
                    if residual_root
                    else "no_residual_root"
                ),
                "diagnostics": _json_value(diagnostics),
                "residual_rows": _residual_rows(diagnostics.normalized_residuals, raw),
                "conservation_diagnostics": (
                    None
                    if attempt.evaluation is None
                    else {
                        "closures": _json_value(attempt.evaluation.closures),
                    }
                ),
            }
        )
    residual_root = (
        result.best.state is not None and result.best.diagnostics.converged
    )
    selected_rank = result.best.diagnostics.jacobian_rank
    return {
        "residual_root_found": residual_root,
        "outcome_classification": (
            "non_identifiable_screening_equation_residual_root"
            if residual_root
            else "no_residual_root"
        ),
        "identifiable_state": False,
        "identifiability": {
            "status": "non_identifiable",
            "jacobian_rank": selected_rank,
            "state_dimension": STATE_DIMENSION,
            "full_column_rank": selected_rank == STATE_DIMENSION,
            "publication_allowed": False,
            "reason": (
                "v1 residual roots are screening-equation diagnostics only; "
                "the observed roots are rank 22/25 and the protocol is invalid "
                "for identifiable-state claims"
            ),
        },
        "selected_start_index": result.selected_start_index,
        "residual_floor": result.residual_floor,
        "attempts": attempts,
    }


def _run_plasma(probabilities: Sequence[float]) -> list[dict[str, Any]]:
    results = []
    for point in OPERATING_POINTS:
        inputs = XenonGlobalInputs(
            point["anode_voltage_v"],
            point["anode_current_a"],
            tuple(float(value) for value in probabilities),
        )
        solved = solve_global_discharge_multistart(
            inputs,
            start_count=PLASMA_START_COUNT,
            options=PLASMA_OPTIONS,
            use_analytic_jacobian=True,
        )
        results.append(
            {
                "operating_point": dict(point),
                "input_hash": stable_hash(_json_value(inputs)),
                **serialize_plasma(solved),
            }
        )
    return results


def _parity(case: BuiltCase, warp_field: FieldMap) -> dict[str, Any]:
    cpu = solve_problem_cpu(case.problem, SOLVER)
    differences = max_field_difference(cpu, warp_field)
    gates = {
        "psi_scale_relative_max": 2.0e-9,
        "br_scale_relative_max": 2.0e-8,
        "bz_scale_relative_max": 2.0e-8,
    }
    passed = (
        cpu.diagnostics.converged
        and warp_field.diagnostics.converged
        and differences["psi_scale_relative"] <= gates["psi_scale_relative_max"]
        and differences["br_scale_relative"] <= gates["br_scale_relative_max"]
        and differences["bz_scale_relative"] <= gates["bz_scale_relative_max"]
    )
    return {
        "case_id": case.case_id,
        "cpu_backend": cpu.diagnostics.backend,
        "warp_backend": warp_field.diagnostics.backend,
        "differences": differences,
        "gates": gates,
        "passed": passed,
    }


def _rank_key(case: Mapping[str, Any]) -> tuple[Any, ...]:
    topology = case.get("topology")
    if not isinstance(topology, dict):
        return (1, 99, 1, 1.0, 1.0, case["case_id"])
    compatible = topology["compatible"]
    count = topology.get("segment_count", 0)
    confidence = topology.get("overall_confidence", 0.0)
    quality = case["field_quality"]
    return (
        0 if compatible else 1,
        abs(count - 4),
        0 if topology.get("status") == "resolved" else 1,
        -confidence,
        quality["boundary_to_peak_ratio"],
        quality["source_representation_error"],
        quality["relative_residual_l2"],
        case["case_id"],
    )


def _case_result(case: BuiltCase, field: FieldMap) -> dict[str, Any]:
    quality = _field_quality(case, field)
    field_gates = evaluate_field_gates(quality)
    artifact = field_artifact(
        case.problem,
        SOLVER,
        field,
        map_stride=1,
        wall_radius_m=case.geometry.chamber.outer_radius_m,
    )
    validate_field_artifact(artifact)
    artifact_bytes = _artifact_bytes(artifact)
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    adapter = GeneratedL1aAdapter(artifact_hash, case)
    evidence = verify_accepted_field_artifact(
        artifact_bytes,
        adapter,
        MapValidationPolicy(maximum_age_s=None),
        reference_time_utc=ACCEPTANCE_TIME_UTC,
    )
    base = {
        "case_id": case.case_id,
        "status": "field_evaluated",
        "design_id": case.design.design_id,
        "sampling_provenance": case.design.provenance,
        "design_values": dict(case.values),
        "derived_geometry": dict(case.derived),
        "identity": {
            "geometry_sha256": case.geometry_sha256,
            "source_sha256": case.source_sha256,
            "config_sha256": case.config_sha256,
            "case_sha256": case.case_sha256,
            "artifact_sha256": artifact_hash,
            "artifact_payload_sha256": artifact["integrity"]["payload_sha256"],
        },
        "backend": field.diagnostics.backend,
        "field_quality": quality,
        "field_gates": field_gates,
        "_artifact_bytes": artifact_bytes,
    }
    try:
        record = build_coupling_record(
            evidence,
            wall_radius_m=case.geometry.chamber.outer_radius_m,
            topology_policy=TOPOLOGY_POLICY,
            uncertainty_model=UNCERTAINTY,
            reference_time_utc=ACCEPTANCE_TIME_UTC,
        )
    except CouplingValidationError as error:
        return {
            **base,
            "topology": {
                "status": "coupling_rejected",
                "reason": str(error),
                "segment_count": 0,
                "overall_confidence": 0.0,
                "compatible": False,
                "gates": {},
                "coupling_identity": None,
                "segments": [],
            },
            "plasma": [],
            "failure_codes": [
                *(["FIELD_GATE_FAILURE"] if not all(field_gates.values()) else []),
                "COUPLING_REJECTED",
            ],
        }
    gates = evaluate_topology_gates(
        record, axial_sample_count=len(field.z_m)
    )
    compatible = all(field_gates.values()) and topology_compatible(gates)
    record_dict = coupling_record_dict(record)
    rows = [
        {
            **dict(row),
            "cusp": {
                "kind": segment.mirror_loss.cusp.kind.value,
                "z_m": segment.mirror_loss.cusp.z_m,
                "b_magnitude_t": segment.mirror_loss.cusp.b_magnitude_t,
                "prominence_t": segment.mirror_loss.cusp.prominence_t,
                "sigma_b_t": segment.mirror_loss.cusp.sigma_b_t,
                "confidence": segment.mirror_loss.cusp.confidence,
                "sample_indices": list(segment.mirror_loss.cusp.sample_indices),
                "bracket_z_m": list(segment.mirror_loss.cusp.bracket_z_m),
            },
            "wall_b_t": segment.mirror_loss.wall_b_t,
            "field_ratio_low_to_high": (
                segment.mirror_loss.field_ratio_low_to_high
            ),
        }
        for row, segment in zip(
            global_solver_inputs(record), record.segments, strict=True
        )
    ]
    probabilities = [float(row["loss_cone_probability"]) for row in rows]
    plasma = _run_plasma(probabilities) if compatible else []
    topology = {
        "status": record.topology_status.value,
        "reason": record.topology_reason,
        "segment_count": len(record.segments),
        "overall_confidence": record.overall_confidence,
        "compatible": compatible,
        "gates": gates,
        "coupling_identity": {
            key: record_dict[key]
            for key in (
                "record_hash",
                "field_map_hash",
                "artifact_hash",
                "source_hash",
                "source_map_binding_hash",
                "field_model_hash",
                "code_hash",
                "config_hash",
                "adapter_code_hash",
                "coupling_model_hash",
                "diagnostics",
            )
        },
        "segments": rows,
    }
    failure_codes = _failure_codes(field_gates, gates)
    if plasma and not any(item["residual_root_found"] for item in plasma):
        failure_codes.append("PLASMA_NONCONVERGENCE")
    return {
        **base,
        "topology": topology,
        "plasma": plasma,
        "failure_codes": failure_codes,
    }


def _write_geometry(path: Path, geometry: Any) -> str:
    data = canonical_json(geometry.to_dict()).encode("utf-8")
    digest = _write_bytes(path, data)
    if deserialize_geometry(path.read_text(encoding="utf-8")).canonical_sha256 != (
        geometry.canonical_sha256
    ):
        raise ValueError("representative geometry replay hash mismatch")
    return digest


def _representatives(
    output: Path,
    ranked: Sequence[Mapping[str, Any]],
    built: Mapping[str, BuiltCase],
    staging: Path,
) -> list[dict[str, Any]]:
    selected = list(ranked[: min(3, len(ranked))])
    entries = []
    for rank, result in enumerate(selected, start=1):
        case = built[result["case_id"]]
        artifact_bytes = (staging / f"{case.case_id}.field-full.json").read_bytes()
        artifact = json.loads(artifact_bytes)
        validate_field_artifact(artifact)
        field_path = output / "representatives" / f"{case.case_id}.field-full.json"
        geometry_path = output / "representatives" / f"{case.case_id}.geometry.json"
        field_hash = _write_bytes(field_path, artifact_bytes)
        geometry_hash = _write_geometry(geometry_path, case.geometry)
        if field_hash != result["identity"]["artifact_sha256"]:
            raise ValueError("representative field replay changed artifact identity")
        entries.append(
            {
                "rank": rank,
                "case_id": case.case_id,
                "field": {
                    "path": str(field_path.relative_to(output)).replace("\\", "/"),
                    "file_sha256": field_hash,
                    "payload_sha256": artifact["integrity"]["payload_sha256"],
                    "stride": 1,
                },
                "geometry": {
                    "path": str(geometry_path.relative_to(output)).replace("\\", "/"),
                    "file_sha256": geometry_hash,
                    "payload_sha256": case.geometry_sha256,
                },
            }
        )
    return entries


def _report(dataset: Mapping[str, Any]) -> str:
    summary = dataset["summary"]
    lines = [
        "# Four-cell topology-targeted L1a search",
        "",
        f"- Classification: `{CLASSIFICATION}`",
        "- Protocol status: `development_evidence_only`",
        "- Preregistered: `False`",
        "- Physical mirror claims valid: `False`",
        "- Performance claims valid: `False`",
        f"- Requested/evaluated: {summary['requested_count']}/{summary['evaluated_count']}",
        f"- Topology compatible: {summary['compatible_count']}",
        f"- Residual-root candidates: {summary['plasma_residual_root_candidate_count']}",
        f"- Plasma residual roots: {summary['plasma_residual_root_count']}",
        f"- Identifiable states: {summary['identifiable_state_count']}",
        f"- Performance publications: {summary['performance_publication_count']}",
        f"- CPU/Warp parity cases: {summary['parity_count']} "
        f"(failures: {summary['parity_failure_count']})",
        "",
        "## Best configurations",
        "",
    ]
    for item in dataset["ranking"][:5]:
        probabilities = [
            row["loss_cone_probability"] for row in item["topology"]["segments"]
        ]
        floors = [
            plasma["residual_floor"] for plasma in item["plasma"]
        ]
        lines.extend(
            (
                f"### {item['case_id']}",
                "",
                f"- Compatible: `{item['topology']['compatible']}`",
                f"- Stages/pitch: `{item['derived_geometry']['stage_count']}` / "
                f"`{item['derived_geometry']['represented_stage_pitch_m']:.9g} m`",
                f"- Segments/confidence: `{item['topology']['segment_count']}` / "
                f"`{item['topology']['overall_confidence']:.9g}`",
                f"- Probabilities: `{probabilities}`",
                f"- Plasma residual floors: `{floors}`",
                (
                    "- Plasma classification: "
                    "`non-identifiable screening equations only`"
                    if item["plasma"]
                    else "- Plasma classification: `not attempted; incompatible gates`"
                ),
                "",
            )
        )
    lines.extend(("", "## Failure taxonomy", ""))
    for code, count in sorted(summary["failure_counts"].items()):
        lines.append(f"- `{code}`: {count} — {FAILURE_TAXONOMY[code]}")
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "Version 1 is non-preregistered development evidence. Coupling v2 used",
            "a deprecated same-z mirror proxy and roundoff-scale null lows. Therefore",
            "its nominal mirror ratios/probabilities are not physical mirror results.",
            "The six rank-22/25 outcomes are non-identifiable screening-equation",
            "residual roots, not plasma-state or performance publications. Raw state",
            "and power values are retained only in the audit archive and are prohibited",
            "from physical/performance use. Coupling v3 plus a preregistered search v2",
            "are required before any such claims. Timings are not claimed.",
            "",
        )
    )
    return "\n".join(lines)


def run_experiment(
    output: Path,
    *,
    count: int = DEFAULT_CASE_COUNT,
    seed: int = DEFAULT_SEED,
    parity_count: int = 8,
) -> dict[str, Any]:
    if parity_count < 1 or parity_count > count:
        raise ValueError("parity_count must be in [1, count]")
    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    designs = sample_designs(count, seed)
    parity_indices = {
        round(index * (count - 1) / (parity_count - 1))
        if parity_count > 1
        else 0
        for index in range(parity_count)
    }
    cases: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    built: dict[str, BuiltCase] = {}
    for index, design in enumerate(designs):
        try:
            case = build_case(design, index)
            built[case.case_id] = case
        except Exception as error:
            cases.append(
                {
                    "case_id": f"four-cell-{index:03d}",
                    "status": "failure",
                    "design_id": design.design_id,
                    "design_values": _design_values(design),
                    "failure_codes": ["GEOMETRY_INVALID"],
                    "failure_message": str(error),
                }
            )
            continue
        try:
            field = solve_problem_warp(case.problem, device="cuda:0", config=SOLVER)
            result = _case_result(case, field)
            artifact_bytes = result.pop("_artifact_bytes")
            (staging / f"{case.case_id}.field-full.json").write_bytes(artifact_bytes)
            cases.append(result)
            if index in parity_indices:
                parity.append(_parity(case, field))
        except Exception as error:
            cases.append(
                {
                    "case_id": case.case_id,
                    "status": "failure",
                    "design_id": design.design_id,
                    "design_values": dict(case.values),
                    "failure_codes": ["FIELD_SOLVER_FAILURE"],
                    "failure_message": str(error),
                }
            )
    evaluated = [case for case in cases if case["status"] == "field_evaluated"]
    ranked = sorted(evaluated, key=_rank_key)
    ranking = []
    for rank, case in enumerate(ranked, start=1):
        copy = dict(case)
        copy["rank"] = rank
        ranking.append(copy)
    representatives = _representatives(output, ranked, built, staging)
    for path in staging.iterdir():
        path.unlink()
    staging.rmdir()
    failure_counts = {code: 0 for code in FAILURE_TAXONOMY}
    for case in cases:
        for code in case["failure_codes"]:
            failure_counts[code] += 1
    failure_counts["PARITY_FAILURE"] = sum(not item["passed"] for item in parity)
    compatible = [
        case for case in evaluated if case["topology"]["compatible"]
    ]
    plasma_converged = [
        case
        for case in compatible
        if any(item["residual_root_found"] for item in case["plasma"])
    ]
    summary = {
        "requested_count": count,
        "evaluated_count": len(evaluated),
        "compatible_count": len(compatible),
        "plasma_attempted_candidate_count": len(compatible),
        "plasma_residual_root_candidate_count": len(plasma_converged),
        "plasma_residual_root_count": sum(
            item["residual_root_found"]
            for case in compatible
            for item in case["plasma"]
        ),
        "identifiable_state_count": 0,
        "performance_publication_count": 0,
        "parity_count": len(parity),
        "parity_failure_count": sum(not item["passed"] for item in parity),
        "failure_counts": failure_counts,
    }
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "protocol_status": PROTOCOL_STATUS,
        "model_chain": [
            "accepted geometry v1.1",
            "L1a current-equivalent Warp field",
            "coupling v2 deprecated same-z mirror proxy",
            "four-cell global plasma screening equations",
        ],
        "sampling": {
            "algorithm": "deterministic shifted-Halton low-discrepancy sequence",
            "seed": seed,
            "count": count,
            "variables": [asdict(variable) for variable in VARIABLES],
        },
        "declared_gates": {
            "field": FIELD_GATES,
            "topology": TOPOLOGY_GATES,
            "topology_policy": _json_value(TOPOLOGY_POLICY),
            "uncertainty": _json_value(UNCERTAINTY),
            "derivation": (
                "coupling v2 emits one ordered segment per accepted centreline "
                "minimum/null and derives wall-high/cusp-low loss-cone probability; "
                "the corrected global model accepts exactly four direct probabilities"
            ),
        },
        "plasma_policy": {
            "operating_points": list(OPERATING_POINTS),
            "solver_options": _json_value(PLASMA_OPTIONS),
            "start_count": PLASMA_START_COUNT,
            "probability_tuning": False,
            "closure_tuning": False,
            "publication": "none",
            "outcome_classification": (
                "rank-deficient residual roots are non-identifiable "
                "screening-equation diagnostics only"
            ),
        },
        "failure_taxonomy": FAILURE_TAXONOMY,
        "cases": cases,
        "ranking": ranking,
        "parity": parity,
        "representatives": representatives,
        "summary": summary,
        "limitations": [
            "L1a is a finite homogeneous-Dirichlet equivalent-current field model.",
            "Topology is sampled-grid and uncertainty-gated, not a continuous proof.",
            "Operating points are hypothetical and not experiment-calibrated.",
            "Global plasma does not provide validated thrust, efficiency, or lifetime.",
            "Version 1 was not preregistered.",
            "Coupling v2 used a deprecated same-z mirror proxy and roundoff null lows.",
            "Residual roots are rank-deficient and are not identifiable plasma states.",
            "No uncontrolled timing or hardware-performance claim is made.",
        ],
    }
    dataset, dataset_hash = write_sealed_json(output / "dataset.json", dataset_payload)
    report_hash = _write_bytes(output / "report.md", _report(dataset).encode("utf-8"))
    files = [
        {
            "path": "dataset.json",
            "kind": "dataset",
            "file_sha256": dataset_hash,
            "payload_sha256": dataset["integrity"]["payload_sha256"],
        },
        {
            "path": "report.md",
            "kind": "report",
            "file_sha256": report_hash,
            "payload_sha256": None,
        },
    ]
    for item in representatives:
        for kind in ("field", "geometry"):
            entry = item[kind]
            files.append(
                {
                    "path": entry["path"],
                    "kind": kind,
                    "file_sha256": entry["file_sha256"],
                    "payload_sha256": entry["payload_sha256"],
                }
            )
    manifest_payload = {
        "schema_version": MANIFEST_VERSION,
        "classification": CLASSIFICATION,
        "protocol_status": PROTOCOL_STATUS,
        "dataset_payload_sha256": dataset["integrity"]["payload_sha256"],
        "deterministic_files": files,
        "representatives": representatives,
    }
    manifest, _ = write_sealed_json(output / "manifest.json", manifest_payload)
    validate_bundle(output)
    return {"dataset": dataset, "manifest": manifest}


def validate_bundle(output: Path) -> dict[str, Any]:
    manifest = load_sealed_json(output / "manifest.json")
    dataset = load_sealed_json(output / "dataset.json")
    if manifest["schema_version"] != MANIFEST_VERSION:
        raise ValueError("unsupported four-cell search manifest")
    if dataset["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported four-cell search dataset")
    if (
        dataset["classification"] != CLASSIFICATION
        or manifest["classification"] != CLASSIFICATION
    ):
        raise ValueError("semantic classification mismatch")
    if manifest["dataset_payload_sha256"] != dataset["integrity"]["payload_sha256"]:
        raise ValueError("manifest/dataset identity mismatch")
    if len(dataset["cases"]) != dataset["sampling"]["count"]:
        raise ValueError("sampling/case count mismatch")
    evaluated = [case for case in dataset["cases"] if case["status"] == "field_evaluated"]
    compatible = [
        case for case in evaluated if case["topology"]["compatible"]
    ]
    summary = dataset["summary"]
    if (
        summary["evaluated_count"] != len(evaluated)
        or summary["compatible_count"] != len(compatible)
        or summary["performance_publication_count"] != 0
        or summary["identifiable_state_count"] != 0
        or summary["plasma_residual_root_count"] != 6
        or summary["plasma_residual_root_count"]
        != sum(
            plasma["residual_root_found"]
            for case in compatible
            for plasma in case["plasma"]
        )
    ):
        raise ValueError("summary counts are inconsistent")
    if dataset["protocol_status"] != PROTOCOL_STATUS:
        raise ValueError("dataset protocol status is inconsistent")
    if manifest["protocol_status"] != PROTOCOL_STATUS:
        raise ValueError("manifest protocol status is inconsistent")
    correction = dataset.get("semantic_correction")
    if not isinstance(correction, dict) or any(
        correction.get(key) is not False
        for key in (
            "numerical_simulations_rerun",
            "numerical_values_modified",
            "selection_or_ranking_modified",
            "representative_artifacts_modified",
        )
    ):
        raise ValueError("semantic correction audit is missing or invalid")
    if manifest.get("semantic_correction") != correction:
        raise ValueError("manifest/dataset semantic correction mismatch")
    expected_selection_hash = stable_hash(
        [
            {"case_id": case["case_id"], "rank": case["rank"]}
            for case in dataset["ranking"]
        ]
    )
    if correction.get("selection_identity_sha256") != expected_selection_hash:
        raise ValueError("semantic correction changed selection/ranking identity")
    audit = dataset.get("audit_raw_numerical_data")
    if (
        not isinstance(audit, dict)
        or audit.get("numeric_values_modified") is not False
        or audit.get("prior_semantic_labels")
        != {
            "performance_publication_count": 6,
            "plasma_converged_candidate_count": 2,
            "plasma_converged_state_count": 6,
        }
    ):
        raise ValueError("raw numerical audit preservation metadata is invalid")
    for case in evaluated:
        expected_case_hash = stable_hash(
            {
                "geometry_sha256": case["identity"]["geometry_sha256"],
                "source_sha256": case["identity"]["source_sha256"],
                "config_sha256": case["identity"]["config_sha256"],
            }
        )
        if case["identity"]["case_sha256"] != expected_case_hash:
            raise ValueError("case identity is not geometry/source/config bound")
        for plasma in case["plasma"]:
            prohibited = {
                "valid_state",
                "screening_performance",
                "state",
                "powers",
                "valid_state_published",
            }
            if prohibited.intersection(plasma):
                raise ValueError("plasma outcome contains publication fields")
            if plasma["identifiable_state"] is not False:
                raise ValueError("v1 plasma outcome cannot claim an identifiable state")
            if plasma["residual_root_found"] and (
                plasma["identifiability"]["jacobian_rank"] != 22
                or plasma["identifiability"]["state_dimension"] != STATE_DIMENSION
                or plasma["identifiability"]["publication_allowed"] is not False
            ):
                raise ValueError("v1 residual root identifiability metadata is invalid")
            for attempt in plasma["attempts"]:
                if prohibited.intersection(attempt):
                    raise ValueError("plasma attempt contains publication fields")
                diagnostics = attempt.get("conservation_diagnostics")
                if isinstance(diagnostics, dict) and "powers" in diagnostics:
                    raise ValueError("outcome conservation contains power fields")
    for parity in dataset["parity"]:
        if not parity["passed"]:
            raise ValueError("bundle contains failed CPU/Warp parity")
    seen: set[str] = set()
    for entry in manifest["deterministic_files"]:
        if entry["path"] in seen:
            raise ValueError("manifest paths are not unique")
        seen.add(entry["path"])
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("manifest path escapes output")
        if _verify_sidecar(path) != entry["file_sha256"]:
            raise ValueError(f"manifest hash mismatch for {entry['path']}")
        if entry["kind"] == "field":
            validate_field_artifact_file(
                path,
                expected_file_sha256=entry["file_sha256"],
                expected_payload_sha256=entry["payload_sha256"],
            )
        elif entry["kind"] == "geometry":
            geometry = deserialize_geometry(path.read_text(encoding="utf-8"))
            if geometry.canonical_sha256 != entry["payload_sha256"]:
                raise ValueError("representative geometry payload mismatch")
    return {"manifest": manifest, "dataset": dataset}
