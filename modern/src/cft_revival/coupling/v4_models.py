"""Frozen v4 HEMP/CFT wall-cusp and inter-cusp-cell contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from .models import MapValidationPolicy, SolverDiagnosticsEvidence, UncertaintyModel
from .v3_models import V3EvidenceIdentity


class V4Status(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"
    NONADIABATIC = "nonadiabatic"
    ORBIT_UNVERIFIED = "orbit_unverified"
    UNCERTAINTY_DOMINATED = "uncertainty_dominated"


@dataclass(frozen=True, slots=True)
class CFTGeometry:
    channel_wall_radius_m: float
    plasma_z_min_m: float
    plasma_z_max_m: float
    core_radius_m: float
    geometry_id: str


@dataclass(frozen=True, slots=True)
class WallCuspPolicy:
    minimum_prominence_t: float = 1.0e-4
    prominence_support_half_width_m: float = 0.25
    minimum_cusp_separation_m: float = 0.1
    minimum_wall_radial_fraction: float = 0.6
    minimum_bundle_paths: int = 1
    endpoint_plane_tolerance_m: float = 0.15
    axial_boundary_margin_m: float = 0.05
    minimum_endpoint_high_field_fraction: float = 0.8


@dataclass(frozen=True, slots=True)
class FieldLineTracePolicy:
    step_m: float = 0.005
    maximum_steps: int = 4000
    wall_tolerance_m: float = 1.0e-5
    maximum_psi_drift_wb: float = 1.0e-5
    minimum_b_t: float = 1.0e-12
    interpolation_relative_error: float = 0.01
    path_relative_error: float = 0.01
    uncertainty_dominance_factor: float = 100.0


@dataclass(frozen=True, slots=True)
class CFTStabilityPolicy:
    maximum_cusp_shift_m: float = 0.02
    maximum_cusp_strength_relative_change: float = 0.15
    maximum_endpoint_shift_m: float = 0.03
    maximum_cell_bound_shift_m: float = 0.02
    maximum_axial_metric_change: float = 0.1


@dataclass(frozen=True, slots=True)
class AxialDominancePolicy:
    pointwise_axial_fraction_threshold: float = 0.75
    minimum_passing_fraction: float = 0.7
    minimum_mean_axial_fraction: float = 0.8


@dataclass(frozen=True, slots=True)
class ElectronOrbitSample:
    sample_id: str
    kinetic_energy_ev: float
    pitch_angle_rad: float
    maximum_rho_over_scale: float = 0.1
    maximum_mu_relative_variation: float = 0.05


@dataclass(frozen=True, slots=True)
class FieldLineSeed:
    seed_id: str
    r_m: float
    z_m: float
    electron_samples: tuple[ElectronOrbitSample, ...]


@dataclass(frozen=True, slots=True)
class CFTCellRegistration:
    cell_id: str
    seeds: tuple[FieldLineSeed, ...]


@dataclass(frozen=True, slots=True)
class OrbitVerificationClaims:
    path_hash: str
    sample_id: str
    converged: bool
    maximum_mu_relative_variation: float
    adapter_id: str
    adapter_version: str
    adapter_code_hash: str
    orbit_model_id: str
    orbit_model_version: str
    orbit_code_hash: str
    orbit_config_hash: str
    convergence_id: str
    convergence_version: str
    convergence_config_hash: str


@dataclass(frozen=True, slots=True)
class OrbitVerificationIdentity:
    adapter_id: str
    adapter_version: str
    adapter_code_hash: str
    orbit_model_id: str
    orbit_model_version: str
    orbit_code_hash: str
    orbit_config_hash: str
    convergence_id: str
    convergence_version: str
    convergence_config_hash: str


@runtime_checkable
class OrbitVerificationAdapter(Protocol):
    adapter_id: str
    adapter_version: str
    adapter_code_hash: str
    orbit_model_id: str
    orbit_model_version: str
    orbit_code_hash: str
    orbit_config_hash: str
    convergence_id: str
    convergence_version: str
    convergence_config_hash: str

    def verify_orbit(
        self,
        path_points_rz_m: tuple[tuple[float, float], ...],
        path_hash: str,
        sample: ElectronOrbitSample,
    ) -> OrbitVerificationClaims:
        """Verify a prescribed particle sample against this exact path."""


@dataclass(frozen=True, slots=True)
class ValidationSetManifest:
    manifest_id: str
    case_ids: tuple[str, ...]
    geometry_family_ids: tuple[str, ...]
    manifest_hash: str


def validation_set_manifest_hash(
    manifest_id: str,
    case_ids: tuple[str, ...],
    geometry_family_ids: tuple[str, ...],
) -> str:
    encoded = json.dumps(
        {
            "manifest_id": manifest_id,
            "case_ids": case_ids,
            "geometry_family_ids": geometry_family_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"cft-v4-validation-set\0" + encoded).hexdigest()


_DEVELOPMENT_CASE_IDS = tuple(
    sorted(
        f"topology-s{stage:02d}-p{pitch}-r{radius}-{polarity}"
        for stage in range(2, 9)
        for pitch in range(2)
        for radius in range(2)
        for polarity in ("neg", "pos")
    )
)
_DEVELOPMENT_FAMILY_IDS = (
    "cft-topology-characterization-v1-cartesian-family",
)
CFT_V4_DEVELOPMENT_MANIFEST = ValidationSetManifest(
    manifest_id="assessed-56-case-characterization",
    case_ids=_DEVELOPMENT_CASE_IDS,
    geometry_family_ids=_DEVELOPMENT_FAMILY_IDS,
    manifest_hash=validation_set_manifest_hash(
        "assessed-56-case-characterization",
        _DEVELOPMENT_CASE_IDS,
        _DEVELOPMENT_FAMILY_IDS,
    ),
)


@dataclass(frozen=True, slots=True)
class HeldOutValidationPolicy:
    maximum_age_s: float = 86_400.0
    maximum_future_skew_s: float = 1.0


@dataclass(frozen=True, slots=True)
class HeldOutCaseRegistration:
    case_id: str
    geometry_family_id: str


@dataclass(frozen=True, slots=True)
class HeldOutValidationRegistration:
    development_manifest: ValidationSetManifest
    held_out_manifest: ValidationSetManifest
    evaluated_case_id: str
    evaluated_geometry_family_id: str
    required_case_count: int
    required_outcomes: tuple[HeldOutCaseRegistration, ...]
    validation_adapter_id: str
    validation_adapter_code_hash: str
    validation_code_hash: str
    validation_config_hash: str
    policy: HeldOutValidationPolicy = HeldOutValidationPolicy()


@dataclass(frozen=True, slots=True)
class HeldOutCaseOutcome:
    case_id: str
    geometry_family_id: str
    three_map_hashes: tuple[str, str, str]
    three_map_evidence_fingerprints: tuple[str, str, str]
    passed: bool


@dataclass(frozen=True, slots=True)
class HeldOutValidationClaims:
    criterion_id: str
    criterion_version: str
    development_manifest: ValidationSetManifest
    held_out_manifest: ValidationSetManifest
    evaluated_case_id: str
    evaluated_geometry_family_id: str
    outcomes: tuple[HeldOutCaseOutcome, ...]
    preregistration_hash: str
    validation_artifact_hash: str
    validation_code_hash: str
    validation_config_hash: str
    generated_at_utc: datetime
    diagnostics: SolverDiagnosticsEvidence


@runtime_checkable
class HeldOutValidationAdapter(Protocol):
    adapter_id: str
    adapter_code_hash: str

    def verify_validation_artifact(
        self, artifact_bytes: bytes
    ) -> HeldOutValidationClaims:
        """Verify exact held-out validation artifact bytes."""


@dataclass(frozen=True, slots=True)
class HeldOutValidationIdentity:
    criterion_id: str
    criterion_version: str
    development_manifest: ValidationSetManifest
    held_out_manifest: ValidationSetManifest
    evaluated_case_id: str
    evaluated_geometry_family_id: str
    outcomes: tuple[HeldOutCaseOutcome, ...]
    preregistration_hash: str
    validation_artifact_hash: str
    validation_code_hash: str
    validation_config_hash: str
    generated_at_utc: datetime
    diagnostics: SolverDiagnosticsEvidence
    adapter_id: str
    adapter_code_hash: str
    policy: HeldOutValidationPolicy


@dataclass(frozen=True, slots=True)
class WallCusp:
    cusp_id: str
    z_m: float
    wall_br_t: float
    wall_b_t: float
    prominence_t: float
    radial_fraction: float
    bundle_endpoint_count: int
    stable: bool


@dataclass(frozen=True, slots=True)
class AxialDominanceMetrics:
    mean_axial_fraction: float
    passing_fraction: float
    sample_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class OrbitAssessment:
    sample: ElectronOrbitSample
    path_hash: str
    rho_over_scale: float | None
    maximum_mu_relative_variation: float | None
    adapter_id: str | None
    adapter_version: str | None
    adapter_code_hash: str | None
    orbit_model_id: str | None
    orbit_model_version: str | None
    orbit_code_hash: str | None
    orbit_config_hash: str | None
    convergence_id: str | None
    convergence_version: str | None
    convergence_config_hash: str | None
    status: V4Status
    reason: str


@dataclass(frozen=True, slots=True)
class WallConnectedMirrorPath:
    seed_id: str
    direction: int
    points_rz_m: tuple[tuple[float, float], ...]
    wall_endpoint_rz_m: tuple[float, float] | None
    wall_endpoint_error_m: float | None
    termination: str
    path_length_m: float
    psi_start_wb: float
    maximum_psi_drift_wb: float
    b_low_t: float
    b_high_t: float
    b_low_location_rz_m: tuple[float, float]
    b_high_location_rz_m: tuple[float, float]
    b_low_lower_t: float
    b_high_upper_t: float
    field_scale_length_m: float | None
    interpolation_error_t: float
    mirror_probability: float | None
    probability_lower: float | None
    probability_upper: float | None
    path_hash: str
    orbit_assessments: tuple[OrbitAssessment, ...]
    status: V4Status
    reason: str


@dataclass(frozen=True, slots=True)
class SeedPathOutcome:
    seed: FieldLineSeed
    negative_path: WallConnectedMirrorPath
    positive_path: WallConnectedMirrorPath
    status: V4Status
    reason: str


@dataclass(frozen=True, slots=True)
class ClosedIslandDiagnostic:
    psi_wb: float
    component_count: int
    cell_id: str


@dataclass(frozen=True, slots=True)
class CFTCell:
    cell_id: str
    z_start_m: float
    z_end_m: float
    upstream_cusp_id: str
    downstream_cusp_id: str
    axial_metrics: AxialDominanceMetrics
    seed_outcomes: tuple[SeedPathOutcome, ...]
    closed_islands: tuple[ClosedIslandDiagnostic, ...]
    status: V4Status
    reason: str


@dataclass(frozen=True, slots=True)
class V4MapAssessment:
    role: str
    identity: V3EvidenceIdentity
    validation_policy: MapValidationPolicy
    cusps: tuple[WallCusp, ...]
    cells: tuple[CFTCell, ...]
    status: V4Status
    reason: str
    detected_cusp_count: int
    expected_cusp_count: int


@dataclass(frozen=True, slots=True)
class V4StabilityAssessment:
    primary: V4MapAssessment
    refined: V4MapAssessment
    enlarged: V4MapAssessment
    cusp_assignment: tuple[tuple[int, int, int], ...]
    cusp_counts: tuple[int, int, int]
    maximum_cusp_shift_m: float | None
    maximum_cusp_strength_relative_change: float | None
    maximum_endpoint_shift_m: float | None
    maximum_cell_bound_shift_m: float | None
    maximum_axial_metric_change: float | None
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class V4Criterion:
    criterion_id: str = "cft-hemp-wall-cusp-v4"
    criterion_version: str = "4.0.0"
    development_evidence_id: str = "assessed-56-case-characterization"
    development_evidence_role: str = "development_non_validation"
    held_out_validation_status: str = "awaiting_new_geometry_family"


@dataclass(frozen=True, slots=True)
class V4CouplingRecord:
    schema_version: str
    record_hash: str
    status: V4Status
    reason: str
    criterion: V4Criterion
    geometry: CFTGeometry
    cusp_policy: WallCuspPolicy
    trace_policy: FieldLineTracePolicy
    axial_policy: AxialDominancePolicy
    stability_policy: CFTStabilityPolicy
    uncertainty_model: UncertaintyModel
    registrations: tuple[CFTCellRegistration, ...]
    validation_registration: HeldOutValidationRegistration
    evidence_fingerprints: tuple[str, str, str]
    stability: V4StabilityAssessment
    orbit_identity: OrbitVerificationIdentity | None
    held_out_validation: HeldOutValidationIdentity | None
