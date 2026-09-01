"""Closed L1b contracts for material-aware axisymmetric magnetostatics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from re import fullmatch

from cft_revival.fields import AxisymmetricDomain, FieldMap, SolverConfig


class MaterialFieldError(Exception):
    """Base exception for the L1b workstream."""


class MaterialFieldValidationError(MaterialFieldError, ValueError):
    """A handoff, rasterization, or solve request is invalid."""


class MaterialFieldConvergenceError(MaterialFieldError, RuntimeError):
    """A solve failed its true-residual acceptance contract."""


@dataclass(frozen=True, slots=True)
class RasterDiagnostic:
    item_id: str
    requested_volume_m3: float
    represented_volume_m3: float
    relative_volume_error: float
    requested_source_measure: float = 0.0
    represented_source_measure: float = 0.0
    relative_source_error: float = 0.0


@dataclass(frozen=True, slots=True)
class WeakActionDiagnostic:
    basis_id: str
    analytical_action_a: float
    rasterized_action_a: float
    absolute_bias_a: float
    relative_bias: float


@dataclass(frozen=True, slots=True)
class RasterizedMaterialProblem:
    """Node-centred finite-volume coefficients and immutable provenance."""

    problem_id: str
    domain: AxisymmetricDomain
    geometry_sha256: str
    magnetics_sha256: str
    authority: str
    material_ids: tuple[str, ...]
    temperatures_k: tuple[float | None, ...]
    polarities: tuple[int | None, ...]
    reluctivity_per_m_h: tuple[float, ...]
    remanence_r_t: tuple[float, ...]
    remanence_z_t: tuple[float, ...]
    free_current_phi_a_per_m2: tuple[float, ...]
    raster_diagnostics: tuple[RasterDiagnostic, ...]
    tolerances_m: tuple[float, float]
    nonlinear_status: str = "linear_only_nonlinear_iron_gated"
    pm_bound_current_phi_a_per_m2: tuple[float, ...] = ()
    weak_action_diagnostics: tuple[WeakActionDiagnostic, ...] = ()
    geometry_region_provenance: tuple[tuple[str, str, bool, str], ...] = ()
    pm_region_count: int = 0
    handoff_interface_count: int = 0
    open_boundary_policy: tuple[tuple[str, float | int], ...] = ()
    geometry_schema_version: str = "cft_revival.geometry.axisymmetric_cft/1.1.0"
    source_envelope_m: tuple[float, float, float, float] = ()
    feature_effective_cells: tuple[tuple[str, float, float], ...] = ()
    qoi_locations_rz_m: tuple[tuple[str, float, float], ...] = ()
    qoi_bore_windows_m: tuple[tuple[str, float, float, float], ...] = ()
    radial_face_reluctivity_per_m_h: tuple[float, ...] = ()
    axial_face_reluctivity_per_m_h: tuple[float, ...] = ()
    authoritative_material_region_count: int = 0
    authoritative_free_current_source_count: int = 0
    remanence_g_r_face_a_per_m: tuple[float, ...] = ()
    remanence_g_z_face_a_per_m: tuple[float, ...] = ()
    outer_boundary_kind: str = "homogeneous_dirichlet_psi"
    robin_radial_q: tuple[float, ...] = ()
    robin_z_min_q: tuple[float, ...] = ()
    robin_z_max_q: tuple[float, ...] = ()
    geometry_bundle_json: str = ""
    magnetics_bundle_json: str = ""

    def __post_init__(self) -> None:
        count = self.domain.shape[0] * self.domain.shape[1]
        if not self.pm_bound_current_phi_a_per_m2:
            object.__setattr__(
                self, "pm_bound_current_phi_a_per_m2", (0.0,) * count
            )
        nr, nz = self.domain.shape
        if not self.radial_face_reluctivity_per_m_h:
            object.__setattr__(
                self,
                "radial_face_reluctivity_per_m_h",
                tuple(
                    _stable_harmonic(
                        self.reluctivity_per_m_h[i * nz + j],
                        self.reluctivity_per_m_h[(i + 1) * nz + j],
                    )
                    for i in range(nr - 1)
                    for j in range(nz)
                ),
            )
        if not self.axial_face_reluctivity_per_m_h:
            object.__setattr__(
                self,
                "axial_face_reluctivity_per_m_h",
                tuple(
                    _stable_harmonic(
                        self.reluctivity_per_m_h[i * nz + j],
                        self.reluctivity_per_m_h[i * nz + j + 1],
                    )
                    for i in range(nr)
                    for j in range(nz - 1)
                ),
            )
        if not self.remanence_g_r_face_a_per_m:
            object.__setattr__(
                self, "remanence_g_r_face_a_per_m", (0.0,) * ((nr - 1) * nz)
            )
        if not self.remanence_g_z_face_a_per_m:
            object.__setattr__(
                self, "remanence_g_z_face_a_per_m", (0.0,) * (nr * (nz - 1))
            )
        if not self.robin_radial_q:
            object.__setattr__(self, "robin_radial_q", (0.0,) * nz)
        if not self.robin_z_min_q:
            object.__setattr__(self, "robin_z_min_q", (0.0,) * nr)
        if not self.robin_z_max_q:
            object.__setattr__(self, "robin_z_max_q", (0.0,) * nr)
        arrays = (
            self.material_ids,
            self.temperatures_k,
            self.polarities,
            self.reluctivity_per_m_h,
            self.remanence_r_t,
            self.remanence_z_t,
            self.free_current_phi_a_per_m2,
            self.pm_bound_current_phi_a_per_m2,
        )
        if any(len(values) != count for values in arrays):
            raise MaterialFieldValidationError("raster arrays do not match domain shape")
        numeric = (
            self.reluctivity_per_m_h,
            self.remanence_r_t,
            self.remanence_z_t,
            self.free_current_phi_a_per_m2,
            self.pm_bound_current_phi_a_per_m2,
        )
        if any(not isfinite(value) for values in numeric for value in values):
            raise MaterialFieldValidationError("raster arrays must be finite")
        if any(value <= 0.0 for value in self.reluctivity_per_m_h):
            raise MaterialFieldValidationError("reluctivity must be strictly positive")
        if len(self.radial_face_reluctivity_per_m_h) != (nr - 1) * nz or len(
            self.axial_face_reluctivity_per_m_h
        ) != nr * (nz - 1):
            raise MaterialFieldValidationError("face reluctivity arrays have invalid shape")
        if any(
            not isfinite(value) or value <= 0.0
            for values in (
                self.radial_face_reluctivity_per_m_h,
                self.axial_face_reluctivity_per_m_h,
            )
            for value in values
        ):
            raise MaterialFieldValidationError("face reluctivity must be finite and positive")
        if len(self.remanence_g_r_face_a_per_m) != (nr - 1) * nz or len(
            self.remanence_g_z_face_a_per_m
        ) != nr * (nz - 1):
            raise MaterialFieldValidationError("remanence face arrays have invalid shape")
        if any(
            not isfinite(value)
            for values in (
                self.remanence_g_r_face_a_per_m,
                self.remanence_g_z_face_a_per_m,
            )
            for value in values
        ):
            raise MaterialFieldValidationError("remanence face arrays must be finite")
        if self.outer_boundary_kind not in {
            "homogeneous_dirichlet_psi",
            "dipole_robin_psi",
        }:
            raise MaterialFieldValidationError("unsupported outer boundary treatment")
        if (
            len(self.robin_radial_q) != nz
            or len(self.robin_z_min_q) != nr
            or len(self.robin_z_max_q) != nr
            or any(
                not isfinite(value) or value < 0.0
                for values in (
                    self.robin_radial_q,
                    self.robin_z_min_q,
                    self.robin_z_max_q,
                )
                for value in values
            )
        ):
            raise MaterialFieldValidationError("Robin boundary factors are invalid")
        if self.authority not in {
            "recoil_remanence_constitutive",
            "equivalent_bound_current",
        }:
            raise MaterialFieldValidationError("unsupported permanent-magnet authority")
        if not fullmatch(r"[0-9a-f]{64}", self.geometry_sha256) or not fullmatch(
            r"[0-9a-f]{64}", self.magnetics_sha256
        ):
            raise MaterialFieldValidationError("provenance hashes must be canonical SHA-256")
        if any(not isinstance(value, str) or not value for value in self.material_ids):
            raise MaterialFieldValidationError("material IDs must be non-empty strings")
        for temperature in self.temperatures_k:
            if temperature is not None and (
                isinstance(temperature, bool)
                or not isfinite(float(temperature))
                or float(temperature) <= 0.0
            ):
                raise MaterialFieldValidationError("temperatures must be finite kelvin values")
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1))
            for value in self.polarities
        ):
            raise MaterialFieldValidationError("polarities must be exactly -1, +1, or null")
        if (
            len(self.tolerances_m) != 2
            or any(not isfinite(value) or value < 0.0 for value in self.tolerances_m)
        ):
            raise MaterialFieldValidationError("tolerances must be finite and non-negative")
        has_remanence = any(
            radial != 0.0 or axial != 0.0
            for radial, axial in zip(self.remanence_r_t, self.remanence_z_t)
        )
        has_pm_bound_current = any(
            value != 0.0 for value in self.pm_bound_current_phi_a_per_m2
        )
        if self.authority == "recoil_remanence_constitutive":
            if self.pm_region_count > 0 and not has_remanence:
                raise MaterialFieldValidationError("recoil PM authority requires remanence")
            if has_pm_bound_current:
                raise MaterialFieldValidationError(
                    "recoil PM authority forbids equivalent bound-current sources"
                )
        else:
            if has_remanence:
                raise MaterialFieldValidationError(
                    "equivalent-current PM authority requires zero remanence"
                )
            if self.pm_region_count > 0 and not has_pm_bound_current:
                raise MaterialFieldValidationError(
                    "equivalent-current PM authority requires a bound-current source"
                )
        if isinstance(self.pm_region_count, bool) or self.pm_region_count < 0:
            raise MaterialFieldValidationError("PM region count must be a non-negative integer")
        if isinstance(self.handoff_interface_count, bool) or self.handoff_interface_count < 0:
            raise MaterialFieldValidationError("interface count must be non-negative")
        if self.geometry_schema_version != "cft_revival.geometry.axisymmetric_cft/1.1.0":
            raise MaterialFieldValidationError("unsupported geometry schema version")
        if bool(self.geometry_bundle_json) != bool(self.magnetics_bundle_json):
            raise MaterialFieldValidationError("replay bundles must be supplied together")
        for name in ("geometry_bundle_json", "magnetics_bundle_json"):
            serialized = getattr(self, name)
            if serialized:
                try:
                    value = json.loads(serialized)
                except (TypeError, ValueError) as error:
                    raise MaterialFieldValidationError(f"{name} is not valid JSON") from error
                if not isinstance(value, dict):
                    raise MaterialFieldValidationError(f"{name} must encode an object")
        if self.source_envelope_m:
            if len(self.source_envelope_m) != 4 or any(
                not isfinite(value) for value in self.source_envelope_m
            ):
                raise MaterialFieldValidationError("source envelope is invalid")
            r_max, z_min, z_max, characteristic = self.source_envelope_m
            if r_max <= 0.0 or z_max <= z_min or characteristic <= 0.0:
                raise MaterialFieldValidationError("source envelope is not positive")
        for item_id, radial_cells, axial_cells in self.feature_effective_cells:
            if (
                not item_id
                or not isfinite(radial_cells)
                or not isfinite(axial_cells)
                or radial_cells <= 0.0
                or axial_cells <= 0.0
            ):
                raise MaterialFieldValidationError("feature-resolution metadata is invalid")
        for name, radial, axial in self.qoi_locations_rz_m:
            if not name or not isfinite(radial) or not isfinite(axial):
                raise MaterialFieldValidationError("QoI location metadata is invalid")
        for name, radius, z_min, z_max in self.qoi_bore_windows_m:
            if (
                not name
                or not isfinite(radius)
                or not isfinite(z_min)
                or not isfinite(z_max)
                or radius <= 0.0
                or z_max <= z_min
            ):
                raise MaterialFieldValidationError("bore-average QoI metadata is invalid")
        for name in (
            "authoritative_material_region_count",
            "authoritative_free_current_source_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MaterialFieldValidationError(f"{name} must be a non-negative integer")
        if self.nonlinear_status != "linear_only_nonlinear_iron_gated":
            raise MaterialFieldValidationError("unvalidated nonlinear iron must remain gated")


def _stable_harmonic(left: float, right: float) -> float:
    smaller, larger = (left, right) if left <= right else (right, left)
    return smaller / (0.5 + 0.5 * smaller / larger)


@dataclass(frozen=True, slots=True)
class MaterialSolverDiagnostics:
    converged: bool
    iterations: int
    initial_residual_l2: float
    final_true_residual_l2: float
    relative_true_residual_l2: float
    residual_history_l2: tuple[float, ...]
    true_residual_restarts: int
    backend: str
    magnetic_energy_j: float
    source_coenergy_j: float
    energy_balance_relative: float
    run_config_sha256: str = "0" * 64
    implementation_sha256: str = "0" * 64
    run_config_json: str = ""
    host_synchronization_count: int = 0
    convergence_check_interval: int = 0


@dataclass(frozen=True, slots=True)
class MaterialFieldResult:
    field: FieldMap
    diagnostics: MaterialSolverDiagnostics
    material_ids: tuple[tuple[str, ...], ...]
    source_phi_a_per_m2: tuple[tuple[float, ...], ...]
    remanence_r_t: tuple[tuple[float, ...], ...]
    remanence_z_t: tuple[tuple[float, ...], ...]
    problem: RasterizedMaterialProblem
    model_level: str = "L1b"


@dataclass(frozen=True, slots=True)
class MaterialSolveConfig:
    linear: SolverConfig = SolverConfig(relative_tolerance=1.0e-9, absolute_tolerance=1.0e-12)
    nonlinear_enabled: bool = False
    minimum_effective_feature_cells: float = 12.0
    allow_underresolved_screening: bool = False

    def __post_init__(self) -> None:
        if self.nonlinear_enabled:
            raise MaterialFieldValidationError(
                "nonlinear iron is gated: no independently validated nonlinear energy path"
            )
        if (
            not isfinite(self.minimum_effective_feature_cells)
            or self.minimum_effective_feature_cells < 12.0
        ):
            raise MaterialFieldValidationError(
                "minimum effective feature resolution cannot be below twelve cells"
            )
        if not isinstance(self.allow_underresolved_screening, bool):
            raise MaterialFieldValidationError(
                "allow_underresolved_screening must be boolean"
            )
