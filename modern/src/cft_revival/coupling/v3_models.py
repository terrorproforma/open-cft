"""Contracts for physically meaningful axisymmetric flux-surface coupling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from .models import (
    AdapterVersionContract,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    TopologyStatus,
    UncertaintyModel,
)


@runtime_checkable
class AxisymmetricPsiFieldMapLike(Protocol):
    r_m: Sequence[float]
    z_m: Sequence[float]
    psi_wb: Sequence[Sequence[float]]
    b_r_t: Sequence[Sequence[float]]
    b_z_t: Sequence[Sequence[float]]


@dataclass(frozen=True, slots=True)
class V3ArtifactClaims:
    field_map: AxisymmetricPsiFieldMapLike
    artifact_schema_version: str
    model_level: str
    artifact_hash: str
    full_map_hash: str
    source_hash: str
    geometry_hash: str
    material_hash: str
    mesh_hash: str
    domain_hash: str
    evidence_binding_hash: str
    backend_id: str
    backend_version: str
    field_model_id: str
    field_model_hash: str
    code_hash: str
    config_hash: str
    generated_at_utc: datetime
    diagnostics: SolverDiagnosticsEvidence
    coordinate_system: str = "cylindrical_axisymmetric_r_z"
    coordinate_unit: str = "m"
    flux_unit: str = "Wb"
    component_unit: str = "T"


@runtime_checkable
class V3ArtifactAdapter(Protocol):
    adapter_id: str
    adapter_code_hash: str
    version_contract: AdapterVersionContract

    def verify_v3_artifact(self, artifact_bytes: bytes) -> V3ArtifactClaims:
        """Parse and verify one exact v3-capable field artifact."""


@dataclass(frozen=True, slots=True)
class ValidatedPsiMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    psi_wb: tuple[tuple[float, ...], ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]
    full_map_hash: str


@dataclass(frozen=True, slots=True)
class StabilityCase:
    role: str
    artifact_hash: str
    full_map_hash: str
    source_hash: str
    geometry_hash: str
    material_hash: str
    mesh_hash: str
    domain_hash: str
    evidence_binding_hash: str
    artifact_schema_version: str
    model_level: str
    field_model_id: str
    field_model_hash: str
    code_hash: str
    config_hash: str
    backend_id: str
    backend_version: str
    adapter_id: str
    adapter_code_hash: str
    adapter_contract: AdapterVersionContract
    generated_at_utc: datetime
    diagnostics: SolverDiagnosticsEvidence
    maximum_age_s: float | None
    maximum_future_skew_s: float
    validation_policy: MapValidationPolicy
    cell_count: int
    interior_cusp_z_m: tuple[float, ...]
    radial_samples: int
    axial_samples: int
    radius_m: float
    z_min_m: float
    z_max_m: float


@dataclass(frozen=True, slots=True)
class TopologyStabilityStudy:
    full_resolution: StabilityCase
    downsampled: StabilityCase
    enlarged_domain: StabilityCase
    maximum_cusp_shift_m: float


@dataclass(frozen=True, slots=True)
class CellRegistration:
    cell_id: str
    flux_quantiles: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ElectronAdiabaticInputs:
    kinetic_energy_ev: float
    perpendicular_energy_fraction: float = 1.0
    maximum_gyroradius_to_scale_length: float = 0.1


@dataclass(frozen=True, slots=True)
class FluxSurfacePolicy:
    psi_absolute_tolerance_wb: float = 1.0e-12
    psi_relative_tolerance: float = 1.0e-8
    connectivity_tolerance_m: float = 1.0e-9
    boundary_exclusion_cells: int = 1
    minimum_contour_points: int = 4
    interpolation_relative_error: float = 0.01
    surface_relative_error: float = 0.01
    uncertainty_dominance_factor: float = 100.0
    null_field_absolute_floor_t: float = 1.0e-15
    null_field_relative_floor: float = 1.0e-12
    segment_bound_absolute_tolerance_t: float = 1.0e-14
    segment_bound_relative_tolerance: float = 0.01
    segment_max_depth: int = 20
    saddle_tie_policy: str = "reject"


class SurfaceStatus(str, Enum):
    VALID = "valid"
    OPEN_BOUNDARY = "open_boundary"
    DISCONNECTED = "disconnected"
    EXACT_NULL = "exact_null"
    MISSING_ADIABATIC_INPUTS = "missing_adiabatic_inputs"
    NONADIABATIC = "nonadiabatic"
    UNCERTAINTY_DOMINATED = "uncertainty_dominated"
    NUMERICALLY_INVALID = "numerically_invalid"
    PHYSICALLY_INVALID = "physically_invalid"


@dataclass(frozen=True, slots=True)
class FluxContour:
    psi_wb: float
    points_rz_m: tuple[tuple[float, float], ...]
    closed: bool
    touches_boundary: bool
    maximum_psi_residual_wb: float
    connectivity_gap_m: float
    simple: bool
    topology_reason: str
    unique_vertex_count: int
    edge_count: int


@dataclass(frozen=True, slots=True)
class ContourFieldCertificate:
    certified_b_low_lower_t: float
    sampled_b_low_upper_t: float
    sampled_b_high_lower_t: float
    certified_b_high_upper_t: float
    maximum_gradient_t_per_m: float
    sampled_points_rz_m: tuple[tuple[float, float], ...]
    sampled_b_t: tuple[float, ...]
    subdivisions: int
    maximum_depth_reached: int
    regular: bool
    reason: str


@dataclass(frozen=True, slots=True)
class BoundedMirrorProbability:
    status: SurfaceStatus
    reason: str
    nominal_probability: float | None
    probability_lower: float | None
    probability_upper: float | None
    field_ratio_lower: float | None
    field_ratio_upper: float | None
    mirror_ratio_lower: float | None
    mirror_ratio_upper: float | None
    uncertainty_method: str


@dataclass(frozen=True, slots=True)
class FluxSurfaceMirror:
    cell_id: str
    flux_quantile: float
    psi_wb: float
    contour_component: int
    contour: FluxContour
    certificate: ContourFieldCertificate
    b_low_t: float
    b_high_t: float
    b_low_location_rz_m: tuple[float, float]
    b_high_location_rz_m: tuple[float, float]
    field_scale_length_m: float | None
    electron_gyroradius_m: float | None
    gyroradius_to_scale_length: float | None
    adiabatic_valid: bool
    probability: BoundedMirrorProbability


@dataclass(frozen=True, slots=True)
class FluxCell:
    cell_id: str
    z_start_m: float
    z_end_m: float
    separatrix_z_m: float
    separatrix_psi_wb: float
    surfaces: tuple[FluxSurfaceMirror, ...]
    quantile_outcomes: tuple[FluxQuantileOutcome, ...]
    status: SurfaceStatus
    reason: str


@dataclass(frozen=True, slots=True)
class BoundaryNullDiagnostic:
    z_m: float
    boundary: str
    b_magnitude_t: float


@dataclass(frozen=True, slots=True)
class FluxQuantileOutcome:
    flux_quantile: float
    requested_psi_wb: float
    status: SurfaceStatus
    reason: str
    contour_components: tuple[int, ...]
    accepted_components: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class V3EvidenceIdentity:
    artifact_hash: str
    full_map_hash: str
    source_hash: str
    geometry_hash: str
    material_hash: str
    mesh_hash: str
    domain_hash: str
    evidence_binding_hash: str
    artifact_schema_version: str
    model_level: str
    field_model_id: str
    field_model_hash: str
    code_hash: str
    config_hash: str
    backend_id: str
    backend_version: str
    adapter_id: str
    adapter_code_hash: str
    adapter_contract: AdapterVersionContract
    generated_at_utc: datetime
    diagnostics: SolverDiagnosticsEvidence
    validation_policy: MapValidationPolicy


@dataclass(frozen=True, slots=True)
class V3CouplingRecord:
    schema_version: str
    record_hash: str
    topology_status: TopologyStatus
    topology_reason: str
    identity: V3EvidenceIdentity
    stability_study: TopologyStabilityStudy
    surface_policy: FluxSurfacePolicy
    uncertainty_model: UncertaintyModel
    electron_inputs: ElectronAdiabaticInputs | None
    cell_registrations: tuple[CellRegistration, ...]
    boundary_nulls: tuple[BoundaryNullDiagnostic, ...]
    interior_cusp_z_m: tuple[float, ...]
    cells: tuple[FluxCell, ...]
