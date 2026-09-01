"""Material, interface, demagnetization, and open-boundary integration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .common import (
    MagneticsValidationError,
    VectorRZ,
    finite_float,
    nonempty_identifier,
)
from .materials import LinearPermeability, SmCoPermanentMagnet, TabulatedBHCurve
from .sources import (
    AxisymmetricBounds,
    PermanentMagnetRepresentation,
    UniformAxisymmetricMagnetizationSource,
)


class WarningSeverity(str, Enum):
    WARNING = "warning"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ValidityWarning:
    """Machine-readable warning that cannot be mistaken for solver success."""

    code: str
    severity: WarningSeverity
    message: str

    def __post_init__(self) -> None:
        nonempty_identifier("code", self.code)
        nonempty_identifier("message", self.message)
        try:
            severity = WarningSeverity(self.severity)
        except ValueError as error:
            raise MagneticsValidationError("unsupported warning severity") from error
        object.__setattr__(self, "severity", severity)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
        }


class ConstitutiveLawKind(str, Enum):
    LINEAR_ISOTROPIC = "linear_isotropic"
    TABULATED_SINGLE_VALUED = "tabulated_single_valued"
    PERMANENT_MAGNET_RECOIL = "permanent_magnet_recoil"


@dataclass(frozen=True, slots=True)
class MaterialRegionContract:
    """Bind one constitutive law identifier to a meridional material region."""

    region_id: str
    constitutive_law_id: str
    constitutive_law_kind: ConstitutiveLawKind
    bounds: AxisymmetricBounds
    priority: int = 0
    permanent_magnet_representation: PermanentMagnetRepresentation | None = None
    magnetization_direction_rz: VectorRZ | None = None

    def __post_init__(self) -> None:
        nonempty_identifier("region_id", self.region_id)
        nonempty_identifier("constitutive_law_id", self.constitutive_law_id)
        try:
            kind = ConstitutiveLawKind(self.constitutive_law_kind)
        except ValueError as error:
            raise MagneticsValidationError("unsupported constitutive law kind") from error
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise MagneticsValidationError("material region priority must be an integer")
        representation = self.permanent_magnet_representation
        if representation is not None:
            try:
                representation = PermanentMagnetRepresentation(representation)
            except ValueError as error:
                raise MagneticsValidationError(
                    "unsupported permanent-magnet representation"
                ) from error
        if kind is ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL:
            if representation is not PermanentMagnetRepresentation.RECOIL_REMANENCE:
                raise MagneticsValidationError(
                    "permanent-magnet recoil regions require recoil-remanence authority"
                )
            if self.magnetization_direction_rz is None:
                raise MagneticsValidationError(
                    "permanent-magnet recoil regions require a magnetization direction"
                )
            direction = self.magnetization_direction_rz.normalized()
            if self.bounds.r_inner_m == 0.0 and direction.radial != 0.0:
                raise MagneticsValidationError(
                    "radial magnetization is not regular on a region touching r=0"
                )
            object.__setattr__(self, "magnetization_direction_rz", direction)
        elif representation is not None or self.magnetization_direction_rz is not None:
            raise MagneticsValidationError(
                "only permanent-magnet recoil regions declare remanence authority/direction"
            )
        object.__setattr__(self, "constitutive_law_kind", kind)
        object.__setattr__(self, "permanent_magnet_representation", representation)

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "constitutive_law_id": self.constitutive_law_id,
            "constitutive_law_kind": self.constitutive_law_kind.value,
            "bounds": self.bounds.to_dict(),
            "priority": self.priority,
            "permanent_magnet_representation": (
                None
                if self.permanent_magnet_representation is None
                else self.permanent_magnet_representation.value
            ),
            "magnetization_direction_rz": (
                None
                if self.magnetization_direction_rz is None
                else self.magnetization_direction_rz.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class MaterialInterfaceContract:
    """Maxwell jump contract from ``minus`` to ``plus`` along ``normal``.

    The solver must enforce ``n·(B_plus-B_minus)=0`` and
    ``(n×(H_plus-H_minus))_phi=K_free_phi``. Bound current due to a permanent
    magnet is constitutive/source data and must not be inserted as free current.
    """

    interface_id: str
    minus_region_id: str
    plus_region_id: str
    normal_minus_to_plus_rz: VectorRZ
    free_surface_current_phi_a_per_m: float = 0.0

    def __post_init__(self) -> None:
        nonempty_identifier("interface_id", self.interface_id)
        nonempty_identifier("minus_region_id", self.minus_region_id)
        nonempty_identifier("plus_region_id", self.plus_region_id)
        if self.minus_region_id == self.plus_region_id:
            raise MagneticsValidationError("an interface must join two different regions")
        normal = self.normal_minus_to_plus_rz.normalized()
        current = finite_float(
            "free_surface_current_phi_a_per_m",
            self.free_surface_current_phi_a_per_m,
        )
        object.__setattr__(self, "normal_minus_to_plus_rz", normal)
        object.__setattr__(self, "free_surface_current_phi_a_per_m", current)

    def residuals(
        self,
        *,
        b_minus_t: VectorRZ,
        b_plus_t: VectorRZ,
        h_minus_a_per_m: VectorRZ,
        h_plus_a_per_m: VectorRZ,
    ) -> tuple[float, float]:
        """Return normal-B residual [T] and tangential-H jump residual [A/m]."""

        normal = self.normal_minus_to_plus_rz
        delta_b = VectorRZ(
            b_plus_t.radial - b_minus_t.radial,
            b_plus_t.axial - b_minus_t.axial,
        )
        delta_h = VectorRZ(
            h_plus_a_per_m.radial - h_minus_a_per_m.radial,
            h_plus_a_per_m.axial - h_minus_a_per_m.axial,
        )
        normal_b_residual = normal.dot(delta_b)
        cross_phi = normal.axial * delta_h.radial - normal.radial * delta_h.axial
        return (
            normal_b_residual,
            cross_phi - self.free_surface_current_phi_a_per_m,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "interface_id": self.interface_id,
            "minus_region_id": self.minus_region_id,
            "plus_region_id": self.plus_region_id,
            "normal_minus_to_plus_rz": self.normal_minus_to_plus_rz.to_dict(),
            "free_surface_current_phi_a_per_m": self.free_surface_current_phi_a_per_m,
            "required_jump_conditions": {
                "normal_b": "dot(n, B_plus - B_minus) = 0",
                "tangential_h": (
                    "cross(n, H_plus - H_minus)_phi = K_free_phi"
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class DemagnetizationAssessment:
    """Screening result against temperature-adjusted intrinsic coercivity."""

    status: str
    opposing_field_a_per_m: float
    intrinsic_coercivity_a_per_m: float | None
    coercivity_margin_a_per_m: float | None
    warnings: tuple[ValidityWarning, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "opposing_field_a_per_m": self.opposing_field_a_per_m,
            "intrinsic_coercivity_a_per_m": self.intrinsic_coercivity_a_per_m,
            "coercivity_margin_a_per_m": self.coercivity_margin_a_per_m,
            "warnings": tuple(warning.to_dict() for warning in self.warnings),
            "limitations": self.limitations,
        }


def assess_demagnetization(
    *,
    material: SmCoPermanentMagnet,
    temperature_k: float,
    magnetization_direction: VectorRZ,
    local_h_a_per_m: VectorRZ,
    warning_fraction_of_hci: float = 0.8,
) -> DemagnetizationAssessment:
    """Screen reverse ``H`` against ``Hci(T)`` without claiming irreversible proof."""

    temperature = finite_float("temperature_k", temperature_k)
    threshold = finite_float(
        "warning_fraction_of_hci", warning_fraction_of_hci
    )
    if not 0.0 < threshold < 1.0:
        raise MagneticsValidationError("warning_fraction_of_hci must lie in (0, 1)")
    direction = magnetization_direction.normalized()
    opposing = max(0.0, -local_h_a_per_m.dot(direction))
    limitations = (
        "Intrinsic-coercivity comparison is a screening check, not a full load-line "
        "or irreversible-demagnetization model.",
        "Local temperature and reverse field must include credible worst-case "
        "spatial and operating-condition extrema.",
    )
    if not (
        material.valid_temperature_min_k
        <= temperature
        <= material.valid_temperature_max_k
    ):
        warning = ValidityWarning(
            code="temperature_outside_material_validity",
            severity=WarningSeverity.INVALID,
            message=(
                f"{temperature} K lies outside material interval "
                f"[{material.valid_temperature_min_k}, "
                f"{material.valid_temperature_max_k}] K."
            ),
        )
        return DemagnetizationAssessment(
            status="indeterminate",
            opposing_field_a_per_m=opposing,
            intrinsic_coercivity_a_per_m=None,
            coercivity_margin_a_per_m=None,
            warnings=(warning,),
            limitations=limitations,
        )

    coercivity = material.intrinsic_coercivity_a_per_m(temperature)
    margin = coercivity - opposing
    warnings: list[ValidityWarning] = []
    if opposing >= coercivity:
        warnings.append(
            ValidityWarning(
                code="intrinsic_coercivity_exceeded",
                severity=WarningSeverity.INVALID,
                message=(
                    f"Opposing H={opposing} A/m meets or exceeds "
                    f"Hci(T)={coercivity} A/m."
                ),
            )
        )
        status = "invalid"
    elif opposing >= threshold * coercivity:
        warnings.append(
            ValidityWarning(
                code="demagnetization_margin_low",
                severity=WarningSeverity.WARNING,
                message=(
                    f"Opposing H uses {opposing / coercivity:.6g} of Hci(T), "
                    f"above warning fraction {threshold:.6g}."
                ),
            )
        )
        status = "warning"
    else:
        status = "within_screening_limit"
    return DemagnetizationAssessment(
        status=status,
        opposing_field_a_per_m=opposing,
        intrinsic_coercivity_a_per_m=coercivity,
        coercivity_margin_a_per_m=margin,
        warnings=tuple(warnings),
        limitations=limitations,
    )


@dataclass(frozen=True, slots=True)
class AxisymmetricTruncationDomain:
    """Finite meridional box used to approximate an unbounded exterior."""

    radius_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self) -> None:
        radius = finite_float("radius_m", self.radius_m)
        z_min = finite_float("z_min_m", self.z_min_m)
        z_max = finite_float("z_max_m", self.z_max_m)
        if radius <= 0.0 or z_max <= z_min:
            raise MagneticsValidationError("domain radius and axial span must be positive")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "z_min_m", z_min)
        object.__setattr__(self, "z_max_m", z_max)

    def to_dict(self) -> dict[str, object]:
        return {
            "radius_m": self.radius_m,
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
        }


@dataclass(frozen=True, slots=True)
class OpenBoundaryDomainPolicy:
    """Evidence policy for accepting a finite-box open-boundary approximation."""

    minimum_padding_characteristic_lengths: float = 3.0
    maximum_boundary_to_peak_field_ratio: float = 1.0e-3
    domain_expansion_factor: float = 1.5
    required_expansion_comparisons: int = 2
    maximum_qoi_relative_change: float = 1.0e-3

    def __post_init__(self) -> None:
        positive_names = (
            "minimum_padding_characteristic_lengths",
            "maximum_boundary_to_peak_field_ratio",
            "domain_expansion_factor",
            "maximum_qoi_relative_change",
        )
        values = {
            name: finite_float(name, getattr(self, name)) for name in positive_names
        }
        if values["minimum_padding_characteristic_lengths"] <= 0.0:
            raise MagneticsValidationError("minimum padding must be positive")
        if not 0.0 < values["maximum_boundary_to_peak_field_ratio"] < 1.0:
            raise MagneticsValidationError("boundary field ratio must lie in (0, 1)")
        if values["domain_expansion_factor"] <= 1.0:
            raise MagneticsValidationError("domain expansion factor must exceed 1")
        if not 0.0 < values["maximum_qoi_relative_change"] < 1.0:
            raise MagneticsValidationError("QoI relative-change limit must lie in (0, 1)")
        if (
            isinstance(self.required_expansion_comparisons, bool)
            or not isinstance(self.required_expansion_comparisons, int)
            or self.required_expansion_comparisons < 1
        ):
            raise MagneticsValidationError(
                "required_expansion_comparisons must be an integer >= 1"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def assess(
        self,
        *,
        domain: AxisymmetricTruncationDomain,
        source_bounds: tuple[AxisymmetricBounds, ...],
        maximum_boundary_field_t: float,
        maximum_interior_field_t: float,
        domain_expansion_factors: tuple[float, ...],
        qoi_relative_changes_on_expansion: tuple[float, ...],
    ) -> tuple[ValidityWarning, ...]:
        """Return all failed acceptance criteria; an empty tuple means accepted."""

        if not source_bounds:
            raise MagneticsValidationError("source_bounds must not be empty")
        if not isinstance(source_bounds, tuple):
            raise MagneticsValidationError("source_bounds must be an immutable tuple")
        boundary_field = finite_float(
            "maximum_boundary_field_t", maximum_boundary_field_t
        )
        interior_field = finite_float(
            "maximum_interior_field_t", maximum_interior_field_t
        )
        if boundary_field < 0.0 or interior_field <= 0.0:
            raise MagneticsValidationError(
                "boundary field must be non-negative and interior peak positive"
            )
        if boundary_field > interior_field:
            raise MagneticsValidationError(
                "maximum_boundary_field_t cannot exceed the supplied global interior peak"
            )
        if not isinstance(qoi_relative_changes_on_expansion, tuple):
            raise MagneticsValidationError("expansion changes must be an immutable tuple")
        if not isinstance(domain_expansion_factors, tuple):
            raise MagneticsValidationError("expansion factors must be an immutable tuple")
        factors = tuple(
            finite_float(f"domain_expansion_factors[{index}]", value)
            for index, value in enumerate(domain_expansion_factors)
        )
        changes = tuple(
            finite_float(f"qoi_relative_changes[{index}]", value)
            for index, value in enumerate(qoi_relative_changes_on_expansion)
        )
        if len(factors) != len(changes):
            raise MagneticsValidationError(
                "each QoI change must have one observed domain expansion factor"
            )
        if any(value <= 1.0 for value in factors):
            raise MagneticsValidationError("observed domain expansion factors must exceed 1")
        if any(value < 0.0 for value in changes):
            raise MagneticsValidationError("QoI relative changes must be non-negative")

        outer_radius = max(bounds.r_outer_m for bounds in source_bounds)
        source_z_min = min(bounds.z_min_m for bounds in source_bounds)
        source_z_max = max(bounds.z_max_m for bounds in source_bounds)
        if (
            outer_radius >= domain.radius_m
            or source_z_min <= domain.z_min_m
            or source_z_max >= domain.z_max_m
        ):
            raise MagneticsValidationError("all sources must lie strictly inside the domain")
        characteristic = max(
            outer_radius,
            source_z_max - source_z_min,
            max(bounds.radial_thickness_m for bounds in source_bounds),
        )
        minimum_padding = min(
            domain.radius_m - outer_radius,
            source_z_min - domain.z_min_m,
            domain.z_max_m - source_z_max,
        )

        warnings: list[ValidityWarning] = []
        if minimum_padding / characteristic < self.minimum_padding_characteristic_lengths:
            warnings.append(
                ValidityWarning(
                    code="open_boundary_padding_insufficient",
                    severity=WarningSeverity.WARNING,
                    message=(
                        "Minimum finite-box padding is "
                        f"{minimum_padding / characteristic:.6g} characteristic lengths; "
                        f"policy requires {self.minimum_padding_characteristic_lengths:.6g}."
                    ),
                )
            )
        field_ratio = boundary_field / interior_field
        if field_ratio > self.maximum_boundary_to_peak_field_ratio:
            warnings.append(
                ValidityWarning(
                    code="boundary_field_not_negligible",
                    severity=WarningSeverity.WARNING,
                    message=(
                        f"Boundary/peak field ratio {field_ratio:.6g} exceeds "
                        f"{self.maximum_boundary_to_peak_field_ratio:.6g}."
                    ),
                )
            )
        if len(changes) < self.required_expansion_comparisons:
            warnings.append(
                ValidityWarning(
                    code="domain_expansion_evidence_missing",
                    severity=WarningSeverity.INVALID,
                    message=(
                        f"Received {len(changes)} expansion comparisons; policy requires "
                        f"{self.required_expansion_comparisons}."
                    ),
                )
            )
        else:
            trailing_factors = factors[-self.required_expansion_comparisons :]
            trailing_changes = changes[-self.required_expansion_comparisons :]
            if any(factor < self.domain_expansion_factor for factor in trailing_factors):
                warnings.append(
                    ValidityWarning(
                        code="domain_expansion_factor_insufficient",
                        severity=WarningSeverity.INVALID,
                        message=(
                            "At least one required trailing domain expansion is below "
                            f"factor {self.domain_expansion_factor:.6g}."
                        ),
                    )
                )
            if any(
                change > self.maximum_qoi_relative_change
                for change in trailing_changes
            ):
                warnings.append(
                    ValidityWarning(
                        code="domain_expansion_not_converged",
                        severity=WarningSeverity.WARNING,
                        message=(
                            "At least one required trailing QoI change exceeds "
                            f"{self.maximum_qoi_relative_change:.6g}."
                        ),
                    )
                )
        return tuple(warnings)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "finite_box_open_boundary_acceptance_policy",
            "minimum_padding_characteristic_lengths": (
                self.minimum_padding_characteristic_lengths
            ),
            "maximum_boundary_to_peak_field_ratio": (
                self.maximum_boundary_to_peak_field_ratio
            ),
            "domain_expansion_factor": self.domain_expansion_factor,
            "required_expansion_comparisons": self.required_expansion_comparisons,
            "maximum_qoi_relative_change": self.maximum_qoi_relative_change,
            "claim_limit": (
                "Passing this policy supports finite-domain convergence only; it is not "
                "an exact infinite-element or analytic open-boundary condition."
            ),
        }


@dataclass(frozen=True, slots=True)
class AxisymmetricMaterialProblemContract:
    """Versioned hand-off payload for an axisymmetric solver worker."""

    problem_id: str
    materials: tuple[
        LinearPermeability | TabulatedBHCurve | SmCoPermanentMagnet, ...
    ]
    regions: tuple[MaterialRegionContract, ...]
    interfaces: tuple[MaterialInterfaceContract, ...]
    magnetization_sources: tuple[UniformAxisymmetricMagnetizationSource, ...]
    open_boundary_policy: OpenBoundaryDomainPolicy
    contract_version: str = "1.0.0"

    def __post_init__(self) -> None:
        nonempty_identifier("problem_id", self.problem_id)
        if self.contract_version != "1.0.0":
            raise MagneticsValidationError("only contract_version 1.0.0 is supported")
        tuple_fields = (
            "materials",
            "regions",
            "interfaces",
            "magnetization_sources",
        )
        if any(not isinstance(getattr(self, name), tuple) for name in tuple_fields):
            raise MagneticsValidationError("integration contract collections must be tuples")
        material_types = (LinearPermeability, TabulatedBHCurve, SmCoPermanentMagnet)
        if any(not isinstance(material, material_types) for material in self.materials):
            raise MagneticsValidationError(
                "every material must be a typed magnetics constitutive model"
            )
        material_ids = {material.material_id for material in self.materials}
        if len(material_ids) != len(self.materials):
            raise MagneticsValidationError("material identifiers must be unique")
        region_ids = {region.region_id for region in self.regions}
        if len(region_ids) != len(self.regions):
            raise MagneticsValidationError("material region identifiers must be unique")
        if any(region.constitutive_law_id not in material_ids for region in self.regions):
            raise MagneticsValidationError("every region must reference a supplied material")
        materials_by_id = {material.material_id: material for material in self.materials}
        for region in self.regions:
            material = materials_by_id[region.constitutive_law_id]
            if isinstance(material, LinearPermeability):
                material_kind = ConstitutiveLawKind.LINEAR_ISOTROPIC
            elif isinstance(material, TabulatedBHCurve):
                material_kind = ConstitutiveLawKind.TABULATED_SINGLE_VALUED
            else:
                material_kind = ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL
            if region.constitutive_law_kind is not material_kind:
                raise MagneticsValidationError(
                    f"region {region.region_id} kind is incompatible with material "
                    f"{material.material_id}"
                )
        interface_ids = {interface.interface_id for interface in self.interfaces}
        if len(interface_ids) != len(self.interfaces):
            raise MagneticsValidationError("interface identifiers must be unique")
        for interface in self.interfaces:
            if (
                interface.minus_region_id not in region_ids
                or interface.plus_region_id not in region_ids
            ):
                raise MagneticsValidationError(
                    "every interface region must reference a supplied region"
                )
        source_ids = {source.source_id for source in self.magnetization_sources}
        if len(source_ids) != len(self.magnetization_sources):
            raise MagneticsValidationError("source identifiers must be unique")
        regions_by_id = {region.region_id: region for region in self.regions}
        source_pm_ids: set[str] = set()
        for source in self.magnetization_sources:
            pm_material = materials_by_id.get(source.permanent_magnet_material_id)
            if not isinstance(pm_material, SmCoPermanentMagnet):
                raise MagneticsValidationError(
                    f"source {source.source_id} must reference a permanent-magnet material"
                )
            source.validate_against(pm_material)
            source_region = regions_by_id.get(source.region_id)
            if source_region is None:
                raise MagneticsValidationError(
                    f"source {source.source_id} must reference a supplied region"
                )
            host_material = materials_by_id[source_region.constitutive_law_id]
            if not isinstance(host_material, LinearPermeability):
                raise MagneticsValidationError(
                    f"equivalent-current source {source.source_id} requires a linear "
                    "recoil-permeability host region"
                )
            if host_material.relative_permeability != pm_material.recoil_relative_permeability:
                raise MagneticsValidationError(
                    f"source {source.source_id} host relative permeability must match "
                    "the permanent-magnet recoil relative permeability"
                )
            if source_region.bounds != source.bounds:
                raise MagneticsValidationError(
                    f"source {source.source_id} bounds must equal its host region bounds"
                )
            source_pm_ids.add(source.permanent_magnet_material_id)
        recoil_pm_ids = {
            region.constitutive_law_id
            for region in self.regions
            if region.constitutive_law_kind
            is ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL
        }
        duplicated_authority = source_pm_ids & recoil_pm_ids
        if duplicated_authority:
            raise MagneticsValidationError(
                "permanent-magnet materials cannot use recoil-remanence and equivalent "
                "bound-current authority in the same handoff"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "coordinate_system": "right_handed_cylindrical_r_phi_z",
            "problem_id": self.problem_id,
            "materials": tuple(material.to_dict() for material in self.materials),
            "regions": tuple(region.to_dict() for region in self.regions),
            "interfaces": tuple(interface.to_dict() for interface in self.interfaces),
            "magnetization_sources": tuple(
                source.to_dict() for source in self.magnetization_sources
            ),
            "open_boundary_policy": self.open_boundary_policy.to_dict(),
            "solver_requirements": {
                "material_assignment": (
                    "deterministic highest-priority region; reject equal-priority overlap"
                ),
                "nonlinear_iteration": (
                    "use differential permeability/Jacobian and report independently "
                    "recomputed constitutive plus algebraic residuals"
                ),
                "current_semantics": (
                    "free current and bound magnetization current are distinct; do not "
                    "double-count permanent-magnet sheets and remanence"
                ),
                "permanent_magnet_authority": (
                    "each magnet uses exactly one of recoil-remanence constitutive or "
                    "equivalent-bound-current authority"
                ),
                "publication": (
                    "reject nonfinite state and retain domain-expansion evidence"
                ),
            },
        }
