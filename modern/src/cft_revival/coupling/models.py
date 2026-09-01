"""Immutable contracts for verified field-to-plasma coupling evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import hypot
from typing import Protocol, Sequence, runtime_checkable


class CouplingError(Exception):
    """Base error for the field-to-plasma coupling workstream."""


class CouplingValidationError(CouplingError, ValueError):
    """An input or policy violates the published coupling contract."""


class EvidenceVerificationError(CouplingValidationError):
    """Artifact evidence is missing, untrusted, or not content-bound."""


class TopologyResolutionError(CouplingError, RuntimeError):
    """A numerical topology operation cannot produce a finite result."""


@runtime_checkable
class AxisymmetricFieldMapLike(Protocol):
    """Structural map carried inside adapter-verified artifact claims."""

    r_m: Sequence[float]
    z_m: Sequence[float]
    b_r_t: Sequence[Sequence[float]]
    b_z_t: Sequence[Sequence[float]]


@runtime_checkable
class AxisymmetricProfileLike(Protocol):
    z_m: Sequence[float]
    b_r_t: Sequence[float]
    b_z_t: Sequence[float]


@dataclass(frozen=True, slots=True)
class FieldProvenance:
    """Map-only metadata; insufficient to authorize record construction."""

    field_model_id: str
    field_model_hash: str
    source_hash: str
    generated_at_utc: datetime
    coordinate_system: str = "cylindrical_axisymmetric_r_z"
    coordinate_unit: str = "m"
    component_unit: str = "T"


@dataclass(frozen=True, slots=True)
class SolverDiagnosticsEvidence:
    converged: bool
    residual_norm: float
    residual_tolerance: float
    relative_residual: float
    relative_tolerance: float
    iterations: int


@dataclass(frozen=True, slots=True)
class AcceptedArtifactClaims:
    """Claims emitted only after an external format adapter accepts bytes."""

    field_map: AxisymmetricFieldMapLike
    artifact_schema_version: str
    model_level: str
    artifact_hash: str
    map_content_hash: str
    source_hash: str
    source_map_binding_hash: str
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
    component_unit: str = "T"


@dataclass(frozen=True, slots=True)
class AdapterVersionContract:
    """Explicit schema/version normalization contract for one adapter."""

    contract_id: str
    contract_version: str
    input_schema_version: str
    normalized_schema_version: str
    model_level: str
    is_migration: bool = False


@runtime_checkable
class AcceptedArtifactAdapter(Protocol):
    """Trust boundary implemented later by each stable artifact loader."""

    adapter_id: str
    adapter_code_hash: str
    version_contract: AdapterVersionContract

    def verify_artifact(self, artifact_bytes: bytes) -> AcceptedArtifactClaims:
        """Parse, schema-check, and accept exact artifact bytes."""


@dataclass(frozen=True, slots=True)
class MapValidationPolicy:
    minimum_radial_samples: int = 2
    minimum_axial_samples: int = 7
    maximum_age_s: float | None = 86_400.0
    maximum_future_skew_s: float = 1.0
    require_axis: bool = True
    axis_coordinate_tolerance_m: float = 1.0e-12
    axis_br_absolute_tolerance_t: float = 1.0e-10
    axis_br_relative_tolerance: float = 1.0e-8
    current_artifact_schema: str = "cft-axisymmetric-field-map/1.1.0"
    accepted_model_levels: tuple[str, ...] = ("L1a",)
    validated_migration_adapter_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidatedAxisymmetricMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]
    field_map_hash: str


@dataclass(frozen=True, slots=True)
class _EvidenceSnapshot:
    artifact_bytes: bytes
    field_map: ValidatedAxisymmetricMap
    artifact_schema_version: str
    model_level: str
    artifact_hash: str
    source_hash: str
    source_map_binding_hash: str
    backend_id: str
    backend_version: str
    field_model_id: str
    field_model_hash: str
    code_hash: str
    config_hash: str
    generated_at_utc: datetime
    diagnostics: SolverDiagnosticsEvidence
    adapter_id: str
    adapter_code_hash: str
    adapter_contract: AdapterVersionContract
    validation_policy: MapValidationPolicy


_EVIDENCE_FACTORY_KEY = object()


class AcceptedFieldEvidence:
    """Immutable evidence wrapper; only the verifier can construct one."""

    __slots__ = ("__snapshot", "__invariant_hash")

    def __new__(
        cls,
        snapshot: _EvidenceSnapshot,
        invariant_hash: str,
        *,
        _factory_key: object | None = None,
    ) -> AcceptedFieldEvidence:
        if _factory_key is not _EVIDENCE_FACTORY_KEY:
            raise TypeError(
                "AcceptedFieldEvidence is private; use verify_accepted_field_artifact"
            )
        instance = super().__new__(cls)
        object.__setattr__(instance, "_AcceptedFieldEvidence__snapshot", snapshot)
        object.__setattr__(
            instance,
            "_AcceptedFieldEvidence__invariant_hash",
            invariant_hash,
        )
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AcceptedFieldEvidence is immutable")

    def __repr__(self) -> str:
        return "AcceptedFieldEvidence(<verified immutable snapshot>)"

    def _components_for_reverification(
        self, *, _factory_key: object
    ) -> tuple[_EvidenceSnapshot, str]:
        if _factory_key is not _EVIDENCE_FACTORY_KEY:
            raise TypeError("evidence snapshot is private")
        return self.__snapshot, self.__invariant_hash


class ProfileRole(str, Enum):
    CENTRELINE = "centreline"
    INNER_RADIAL_PROFILE = "inner_radial_profile"
    WALL = "wall"


@dataclass(frozen=True, slots=True)
class UncertaintyModel:
    """Field magnitude uncertainty inputs, explicitly in tesla."""

    absolute_independent_sigma_t: float = 0.0
    relative_independent_sigma: float = 0.0
    common_mode_sigma_t: float = 0.0
    residual_correlation: float = 0.0
    coverage_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class FieldProfile:
    name: str
    role: ProfileRole
    sampled_r_m: float
    z_m: tuple[float, ...]
    b_r_t: tuple[float, ...]
    b_z_t: tuple[float, ...]
    independent_sigma_b_t: tuple[float, ...]
    common_mode_sigma_t: float

    @property
    def magnitude_t(self) -> tuple[float, ...]:
        return tuple(hypot(br, bz) for br, bz in zip(self.b_r_t, self.b_z_t, strict=True))

    @property
    def sigma_b_t(self) -> tuple[float, ...]:
        return tuple(
            hypot(value, self.common_mode_sigma_t)
            for value in self.independent_sigma_b_t
        )


class CandidateKind(str, Enum):
    NULL = "null"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    PLATEAU_MINIMUM = "plateau_minimum"
    PLATEAU_MAXIMUM = "plateau_maximum"
    BOUNDARY_MINIMUM = "boundary_minimum"
    BOUNDARY_MAXIMUM = "boundary_maximum"


class PlateauPolicy(str, Enum):
    MIDPOINT = "midpoint"
    BOUNDS = "bounds"
    REJECT = "reject"


class TiePolicy(str, Enum):
    PRESERVE = "preserve"
    HIGHEST_CONFIDENCE = "highest_confidence"


class TopologyStatus(str, Enum):
    RESOLVED = "resolved"
    NO_TOPOLOGY = "no_topology"
    DEGENERATE = "degenerate"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class TopologyPolicy:
    relative_value_tolerance: float = 1.0e-8
    absolute_value_tolerance_t: float = 1.0e-18
    null_relative_tolerance: float = 1.0e-7
    null_absolute_tolerance_t: float = 1.0e-18
    minimum_prominence_relative: float = 1.0e-5
    minimum_prominence_sigma: float = 2.0
    minimum_candidate_confidence: float = 0.5
    minimum_segment_confidence: float = 0.5
    report_boundary_extrema: bool = True
    allow_boundary_minima_as_cusps: bool = False
    plateau_policy: PlateauPolicy = PlateauPolicy.MIDPOINT
    tie_policy: TiePolicy = TiePolicy.PRESERVE
    tie_relative_tolerance: float = 1.0e-9
    tie_absolute_tolerance_t: float = 1.0e-18


@dataclass(frozen=True, slots=True)
class TopologyCandidate:
    kind: CandidateKind
    z_m: float
    b_magnitude_t: float
    b_r_t: float
    b_z_t: float
    sigma_b_t: float
    independent_sigma_b_t: float
    common_mode_sigma_t: float
    prominence_t: float
    confidence: float
    bracket_z_m: tuple[float, float]
    interpolation: str
    sample_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProfileDescriptor:
    name: str
    role: ProfileRole
    sampled_r_m: float
    minimum_b_t: float
    maximum_b_t: float
    integral_b_t_m: float
    nulls: tuple[TopologyCandidate, ...]
    extrema: tuple[TopologyCandidate, ...]
    boundary_extrema: tuple[TopologyCandidate, ...]
    topology_status: TopologyStatus
    topology_reason: str


@dataclass(frozen=True, slots=True)
class UncertainProbability:
    value: float
    standard_uncertainty: float
    lower: float
    upper: float
    ratio_standard_uncertainty: float
    input_covariance_t2: float
    input_correlation: float
    coverage_factor: float
    propagation: str
    interval_method: str


@dataclass(frozen=True, slots=True)
class MirrorLoss:
    cusp: TopologyCandidate
    wall_b_t: float
    wall_independent_sigma_b_t: float
    common_mode_sigma_t: float
    covariance_t2: float
    correlation: float
    field_ratio_low_to_high: float
    mirror_ratio_high_to_low: float | None
    probability: UncertainProbability
    confidence: float


@dataclass(frozen=True, slots=True)
class TopologySegment:
    segment_id: str
    z_start_m: float
    z_end_m: float
    representative_cusp_z_m: float
    mirror_loss: MirrorLoss
    confidence: float


@dataclass(frozen=True, slots=True)
class CouplingRecord:
    schema_version: str
    record_hash: str
    topology_status: TopologyStatus
    topology_reason: str
    field_map_hash: str
    artifact_hash: str
    source_hash: str
    source_map_binding_hash: str
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
    adapter_contract_id: str
    adapter_contract_version: str
    adapter_input_schema_version: str
    adapter_normalized_schema_version: str
    adapter_is_migration: bool
    generated_at_utc: datetime
    maximum_age_s: float | None
    maximum_future_skew_s: float
    diagnostics: SolverDiagnosticsEvidence
    coupling_model_hash: str
    inner_profile_radius_m: float
    inner_profile_role: ProfileRole
    wall_radius_m: float
    inner_profile: ProfileDescriptor
    wall: ProfileDescriptor
    uncertainty_model: UncertaintyModel
    segments: tuple[TopologySegment, ...]
    alternative_candidates: tuple[TopologyCandidate, ...]
    overall_confidence: float
