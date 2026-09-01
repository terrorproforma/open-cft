"""Adapters from geometry into accepted magnetics and L1a field contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Protocol

from cft_revival.fields import AzimuthalCurrentBand
from cft_revival.magnetics import (
    AxisymmetricBounds,
    AxisymmetricMaterialProblemContract,
    ConstitutiveLawKind,
    LinearPermeability,
    MaterialInterfaceContract,
    MaterialRegionContract,
    OpenBoundaryDomainPolicy,
    PermanentMagnetRepresentation,
    SmCoPermanentMagnet,
    TabulatedBHCurve,
    UniformAxisymmetricMagnetizationSource,
    VectorRZ,
)

from .model import (
    AxisymmetricCFTGeometry,
    GeometryValidationError,
    MaterialKind,
    MeridionalRegion,
    PermanentMagnetAuthority,
    RegionShape,
)
from .topology import interface_topology

MagneticsMaterial = LinearPermeability | TabulatedBHCurve | SmCoPermanentMagnet


class GeometryRegionProvider(Protocol):
    """Structural input expected by downstream material-aware workers."""

    def solver_neutral_contract(self) -> "SolverNeutralGeometryContract":
        """Return material regions, interfaces, and source descriptors."""


@dataclass(frozen=True, slots=True)
class SolverNeutralGeometryContract:
    contract_version: str
    coordinate_system: str
    config_id: str
    material_regions: tuple[dict[str, object], ...]
    interfaces: tuple[dict[str, object], ...]
    magnetization_sources: tuple[dict[str, object], ...]
    external_components: tuple[dict[str, object], ...]
    permanent_magnet_plan: dict[str, object]
    permanent_magnet_authority_rule: str

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "coordinate_system": self.coordinate_system,
            "config_id": self.config_id,
            "material_regions": list(self.material_regions),
            "interfaces": list(self.interfaces),
            "magnetization_sources": list(self.magnetization_sources),
            "external_components": list(self.external_components),
            "permanent_magnet_plan": self.permanent_magnet_plan,
            "permanent_magnet_authority_rule": self.permanent_magnet_authority_rule,
        }


@dataclass(frozen=True, slots=True)
class GeometryAdapter:
    geometry: AxisymmetricCFTGeometry

    def solver_neutral_contract(self) -> SolverNeutralGeometryContract:
        envelope_radius = max(
            max(region.r_outer_start_m, region.r_outer_end_m)
            for region in self.geometry.regions
        )
        z_min = min(region.z_min_m for region in self.geometry.regions)
        z_max = max(region.z_max_m for region in self.geometry.regions)
        padding = max(envelope_radius, z_max - z_min) * 4.0
        if not isfinite(padding):
            raise GeometryValidationError(
                "solver-neutral ambient padding is not representable"
            )
        ambient: dict[str, object] = {
            "region_id": "ambient-background",
            "owner_id": "solver-domain",
            "role": "ambient_background",
            "material_id": "vacuum-plasma-placeholder",
            "shape": "rectangular_annulus",
            "r_inner_start_m": 0.0,
            "r_inner_end_m": 0.0,
            "r_outer_start_m": envelope_radius + padding,
            "r_outer_end_m": envelope_radius + padding,
            "z_min_m": z_min - padding,
            "z_max_m": z_max + padding,
            "polarity": None,
            "priority": 0,
            "units": {
                "r_inner_start_m": "m",
                "r_inner_end_m": "m",
                "r_outer_start_m": "m",
                "r_outer_end_m": "m",
                "z_min_m": "m",
                "z_max_m": "m",
            },
        }
        regions = (ambient,) + tuple(
            {
                **region.to_dict(),
                "priority": 10,
                "units": {
                    "r_inner_start_m": "m",
                    "r_inner_end_m": "m",
                    "r_outer_start_m": "m",
                    "r_outer_end_m": "m",
                    "z_min_m": "m",
                    "z_max_m": "m",
                },
            }
            for region in self.geometry.regions
        )
        interfaces = tuple(
            {
                **descriptor.to_dict(),
                "bound_current_semantics": (
                    "derived from the geometry permanent-magnet plan; never free current"
                ),
            }
            for descriptor in interface_topology(self.geometry)
        )
        sources = tuple(
            {
                "source_id": f"{stage.magnet_region_id}-magnetization",
                "region_id": stage.magnet_region_id,
                "material_id": self.geometry.region_by_id(
                    stage.magnet_region_id
                ).material_id,
                "direction_rz": {
                    "radial": 0.0,
                    "axial": float(stage.magnetization.polarity),
                },
                "magnitude_semantics": "derive from Br(T)/mu0 in magnetic material contract",
            }
            for stage in self.geometry.stages
        )
        return SolverNeutralGeometryContract(
            contract_version="cft_revival.geometry.solver_neutral/1.1.0",
            coordinate_system=self.geometry.coordinate_system,
            config_id=self.geometry.config_id,
            material_regions=regions,
            interfaces=tuple(interfaces),
            magnetization_sources=sources,
            external_components=tuple(
                component.to_dict() for component in self.geometry.external_components
            ),
            permanent_magnet_plan=self.geometry.permanent_magnet_plan.to_dict(),
            permanent_magnet_authority_rule=(
                "Select exactly one of recoil-remanence constitutive authority or "
                "equivalent-bound-current authority for every permanent-magnet region."
            ),
        )


def _rectangular_bounds(region: MeridionalRegion) -> AxisymmetricBounds:
    if region.shape is not RegionShape.RECTANGULAR_ANNULUS:
        raise GeometryValidationError(
            "accepted magnetics v1 adapter requires rectangular material regions"
        )
    return AxisymmetricBounds(
        region.r_inner_start_m,
        region.r_outer_start_m,
        region.z_min_m,
        region.z_max_m,
    )


def _validated_registry(
    geometry: AxisymmetricCFTGeometry,
    material_registry: Mapping[str, MagneticsMaterial],
) -> dict[str, MagneticsMaterial]:
    if not isinstance(material_registry, Mapping):
        raise GeometryValidationError("material_registry must be a mapping")
    registry = dict(material_registry)
    for definition in geometry.materials:
        material = registry.get(definition.material_id)
        if material is None:
            raise GeometryValidationError(
                f"material registry is missing {definition.material_id!r}"
            )
        if material.material_id != definition.material_id:
            raise GeometryValidationError(
                "material registry key and material_id must match exactly"
            )
        if definition.category is MaterialKind.PERMANENT_MAGNET:
            if not isinstance(material, SmCoPermanentMagnet):
                raise GeometryValidationError(
                    f"permanent-magnet material {definition.material_id!r} must "
                    "resolve to SmCoPermanentMagnet"
                )
            relative = material.recoil_relative_permeability
        else:
            if isinstance(material, SmCoPermanentMagnet):
                raise GeometryValidationError(
                    f"non-PM material {definition.material_id!r} cannot resolve "
                    "to a permanent magnet"
                )
            if isinstance(material, LinearPermeability):
                relative = material.relative_permeability
            else:
                # Geometry cannot collapse a nonlinear law to one scalar. The
                # serialized relative permeability is therefore only a display
                # hint and is not used to replace the supplied B-H law.
                relative = definition.relative_permeability
        if isinstance(material, LinearPermeability) or isinstance(
            material, SmCoPermanentMagnet
        ):
            if relative != definition.relative_permeability:
                raise GeometryValidationError(
                    f"registry permeability for {definition.material_id!r} "
                    "does not match serialized ownership"
                )
    return registry


def _constitutive_kind(material: MagneticsMaterial) -> ConstitutiveLawKind:
    if isinstance(material, LinearPermeability):
        return ConstitutiveLawKind.LINEAR_ISOTROPIC
    if isinstance(material, TabulatedBHCurve):
        return ConstitutiveLawKind.TABULATED_SINGLE_VALUED
    return ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL


def to_magnetics_handoff(
    geometry: AxisymmetricCFTGeometry,
    *,
    material_registry: Mapping[str, MagneticsMaterial],
) -> AxisymmetricMaterialProblemContract:
    """Adapt magnetic solids to the accepted material-aware contract.

    Plasma and BN regions are omitted because both are mu_r=1 in this geometry
    workstream.  Tapered exit boundaries remain in the solver-neutral contract;
    they do not alter this magnetostatic material map.
    """

    registry = _validated_registry(geometry, material_registry)
    selected = geometry.permanent_magnet_plan.authority
    pm_definitions = tuple(
        material
        for material in geometry.materials
        if material.category is MaterialKind.PERMANENT_MAGNET
    )
    if len(pm_definitions) != 1:
        raise GeometryValidationError(
            "geometry must declare exactly one permanent-magnet material"
        )
    permanent = registry[pm_definitions[0].material_id]
    if not isinstance(permanent, SmCoPermanentMagnet):
        raise GeometryValidationError("resolved permanent magnet has invalid type")
    authoritative_pm_regions = tuple(
        region
        for region in geometry.regions
        if region.role == "permanent_magnet"
    )
    if any(
        region.shape is not RegionShape.RECTANGULAR_ANNULUS
        for region in authoritative_pm_regions
    ):
        raise GeometryValidationError(
            "magnetics handoff cannot represent non-rectangular permanent magnets"
        )
    supported_roles = {
        "anode",
        "injector_plasma",
        "channel_plasma",
        "dielectric_wall",
        "permanent_magnet",
        "pole_piece",
        "shield",
        "yoke",
    }
    selected_regions = tuple(
        region
        for region in geometry.regions
        if region.role in supported_roles
        and region.shape is RegionShape.RECTANGULAR_ANNULUS
    )
    background = LinearPermeability("ambient-vacuum", 1.0)
    envelope_radius = max(
        max(region.r_outer_start_m, region.r_outer_end_m)
        for region in geometry.regions
    )
    z_min = min(region.z_min_m for region in geometry.regions)
    z_max = max(region.z_max_m for region in geometry.regions)
    padding = max(envelope_radius, z_max - z_min) * 4.0
    if not isfinite(padding):
        raise GeometryValidationError(
            "magnetics handoff ambient padding is not representable"
        )
    background_region = MaterialRegionContract(
        region_id="ambient-background",
        constitutive_law_id=background.material_id,
        constitutive_law_kind=ConstitutiveLawKind.LINEAR_ISOTROPIC,
        bounds=AxisymmetricBounds(
            0.0,
            envelope_radius + padding,
            z_min - padding,
            z_max + padding,
        ),
        priority=0,
    )
    contract_regions: list[MaterialRegionContract] = [background_region]
    sources: list[UniformAxisymmetricMagnetizationSource] = []
    referenced_material_ids = {
        region.material_id for region in selected_regions
    }
    materials: list[MagneticsMaterial] = [
        background,
        *(
            registry[material_id]
            for material_id in sorted(referenced_material_ids)
        ),
    ]
    equivalent_host: LinearPermeability | None = None
    if selected is PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT:
        equivalent_host = LinearPermeability(
            f"{permanent.material_id}-equivalent-host",
            permanent.recoil_relative_permeability,
        )
        materials.append(equivalent_host)
    for region in selected_regions:
        bounds = _rectangular_bounds(region)
        if region.role != "permanent_magnet":
            material = registry[region.material_id]
            contract_regions.append(
                MaterialRegionContract(
                    region.region_id,
                    region.material_id,
                    _constitutive_kind(material),
                    bounds,
                    priority=10,
                )
            )
            continue
        direction = VectorRZ(0.0, float(region.polarity))
        if selected is PermanentMagnetAuthority.RECOIL_REMANENCE:
            contract_regions.append(
                MaterialRegionContract(
                    region.region_id,
                    permanent.material_id,
                    ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL,
                    bounds,
                    priority=20,
                    permanent_magnet_representation=(
                        PermanentMagnetRepresentation.RECOIL_REMANENCE
                    ),
                    magnetization_direction_rz=direction,
                )
            )
        else:
            assert equivalent_host is not None
            contract_regions.append(
                MaterialRegionContract(
                    region.region_id,
                    equivalent_host.material_id,
                    ConstitutiveLawKind.LINEAR_ISOTROPIC,
                    bounds,
                    priority=20,
                )
            )
            sources.append(
                UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
                    source_id=f"{region.region_id}-equivalent-current",
                    region_id=region.region_id,
                    material=permanent,
                    bounds=bounds,
                    direction=direction,
                    temperature_k=permanent.reference_temperature_k,
                )
            )
    handed_off_pm_ids = {
        region.region_id
        for region in contract_regions
        if region.region_id
        in {pm_region.region_id for pm_region in authoritative_pm_regions}
    }
    authoritative_pm_ids = {
        region.region_id for region in authoritative_pm_regions
    }
    if handed_off_pm_ids != authoritative_pm_ids:
        raise GeometryValidationError(
            "magnetics handoff must account for every authoritative PM region"
        )
    if (
        selected is PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT
        and len(sources) != len(authoritative_pm_regions)
    ):
        raise GeometryValidationError(
            "equivalent-current handoff source count must equal PM region count"
        )
    contract_region_ids = {region.region_id for region in contract_regions}
    interfaces_list: list[MaterialInterfaceContract] = []
    for descriptor in interface_topology(geometry):
        if descriptor.region_id not in contract_region_ids:
            continue
        neighbor = descriptor.adjacent_region_id
        if neighbor == "symmetry-axis":
            continue
        if neighbor not in contract_region_ids:
            neighbor = "ambient-background"
        interfaces_list.append(
            MaterialInterfaceContract(
                interface_id=descriptor.interface_id,
                minus_region_id=descriptor.region_id,
                plus_region_id=neighbor,
                normal_minus_to_plus_rz=VectorRZ(*descriptor.unit_normal_rz),
                free_surface_current_phi_a_per_m=0.0,
            )
        )
    interfaces = tuple(interfaces_list)
    return AxisymmetricMaterialProblemContract(
        problem_id=f"geometry-{geometry.config_id}-{selected.value}",
        materials=tuple(materials),
        regions=tuple(contract_regions),
        interfaces=interfaces,
        magnetization_sources=tuple(sources),
        open_boundary_policy=OpenBoundaryDomainPolicy(),
    )


@dataclass(frozen=True, slots=True)
class L1aCurrentEquivalentPreview:
    preview_id: str
    representation_plan_id: str
    authoritative: bool
    approximation: str
    bands: tuple[AzimuthalCurrentBand, ...]

    def __post_init__(self) -> None:
        if self.authoritative is not False:
            raise GeometryValidationError("L1a previews must be non-authoritative")
        if not isinstance(self.bands, tuple):
            raise GeometryValidationError("L1a preview bands must be an immutable tuple")

    def to_dict(self) -> dict[str, object]:
        return {
            "preview_id": self.preview_id,
            "representation_plan_id": self.representation_plan_id,
            "authoritative": self.authoritative,
            "approximation": self.approximation,
            "bands": [
                {
                    "name": band.name,
                    "r_inner_m": band.r_inner_m,
                    "r_outer_m": band.r_outer_m,
                    "z_min_m": band.z_min_m,
                    "z_max_m": band.z_max_m,
                    "ampere_turns_a": band.ampere_turns_a,
                    "polarity": band.polarity,
                }
                for band in self.bands
            ],
        }


def to_l1a_current_equivalent_preview(
    geometry: AxisymmetricCFTGeometry,
    *,
    material_registry: Mapping[str, MagneticsMaterial],
    radial_smear_thickness_m: float = 2.5e-4,
) -> L1aCurrentEquivalentPreview:
    """Approximate axial-magnetization surface sheets as L1a current bands.

    This is explicitly an L1a vacuum-field preview.  It cannot represent pole
    permeability, recoil response, demagnetization, or a finite-thickness sheet
    exactly, and must not be combined with recoil-remanence magnet regions.
    """

    if isinstance(radial_smear_thickness_m, bool):
        raise GeometryValidationError("radial_smear_thickness_m must be positive")
    smear = float(radial_smear_thickness_m)
    if not isfinite(smear) or smear <= 0.0:
        raise GeometryValidationError("radial_smear_thickness_m must be positive")
    registry = _validated_registry(geometry, material_registry)
    pm_ids = {
        region.material_id
        for region in geometry.regions
        if region.role == "permanent_magnet"
    }
    if len(pm_ids) != 1:
        raise GeometryValidationError(
            "L1a preview requires exactly one permanent-magnet material"
        )
    permanent = registry[next(iter(pm_ids))]
    if not isinstance(permanent, SmCoPermanentMagnet):
        raise GeometryValidationError("L1a preview PM registry entry has invalid type")
    magnetization = (
        permanent.remanence_t(permanent.reference_temperature_k)
        / 1.2566370614359173e-6
    )
    bands: list[AzimuthalCurrentBand] = []
    for stage in geometry.stages:
        region = geometry.region_by_id(stage.magnet_region_id)
        if 2.0 * smear >= region.minimum_radial_thickness_m:
            raise GeometryValidationError("L1a smear bands must fit inside each magnet")
        ampere_turns = magnetization * region.axial_thickness_m
        polarity = stage.magnetization.polarity
        bands.extend(
            (
                AzimuthalCurrentBand(
                    name=f"{region.region_id}-inner-sheet-l1a-preview",
                    r_inner_m=region.r_inner_start_m,
                    r_outer_m=region.r_inner_start_m + smear,
                    z_min_m=region.z_min_m,
                    z_max_m=region.z_max_m,
                    ampere_turns_a=ampere_turns,
                    polarity=-polarity,
                ),
                AzimuthalCurrentBand(
                    name=f"{region.region_id}-outer-sheet-l1a-preview",
                    r_inner_m=region.r_outer_start_m - smear,
                    r_outer_m=region.r_outer_start_m,
                    z_min_m=region.z_min_m,
                    z_max_m=region.z_max_m,
                    ampere_turns_a=ampere_turns,
                    polarity=polarity,
                ),
            )
        )
    return L1aCurrentEquivalentPreview(
        preview_id=f"{geometry.config_id}-l1a-current-equivalent-preview-v1",
        representation_plan_id=geometry.permanent_magnet_plan.plan_id,
        authoritative=False,
        approximation=(
            "thin-volume-band smear of cylindrical bound-current sheets; "
            "not a material-aware solver handoff"
        ),
        bands=tuple(bands),
    )
