"""Deterministic hypothetical PPM-stack CFT geometry generators."""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    AxisymmetricCFTGeometry,
    ChamberDefinition,
    ElectrodeDefinition,
    EvidenceNote,
    ExternalComponent,
    GeometryValidationError,
    MagnetizationDirection,
    ManufacturingRules,
    MaterialDefinition,
    MaterialKind,
    MeridionalRegion,
    PPMStage,
    PermanentMagnetAuthority,
    PermanentMagnetRepresentationPlan,
    RegionShape,
)


@dataclass(frozen=True, slots=True)
class PPMStackParameters:
    config_id: str
    title: str
    chamber_inner_radius_m: float
    chamber_outer_radius_m: float
    chamber_length_m: float
    injector_length_m: float
    dielectric_thickness_m: float
    thermal_clearance_m: float
    magnet_inner_radius_m: float
    magnet_outer_radius_m: float
    stage_pitch_m: float
    stage_centers_m: tuple[float, ...]
    magnet_axial_thicknesses_m: tuple[float, ...]
    shield_outer_radius_m: float
    yoke_outer_radius_m: float
    exit_length_m: float = 0.0
    exit_outer_radius_m: float | None = None
    first_polarity: int = 1
    radial_tolerance_m: float = 5.0e-5
    axial_tolerance_m: float = 5.0e-5
    minimum_thickness_m: float = 2.5e-4
    minimum_clearance_m: float = 1.0e-4
    permanent_magnet_authority: PermanentMagnetAuthority = (
        PermanentMagnetAuthority.RECOIL_REMANENCE
    )

    def __post_init__(self) -> None:
        if len(self.stage_centers_m) != len(self.magnet_axial_thicknesses_m):
            raise GeometryValidationError(
                "each stage center requires one magnet axial thickness"
            )
        if len(self.stage_centers_m) < 2:
            raise GeometryValidationError("a PPM stack requires at least two stages")
        if self.first_polarity not in (-1, 1):
            raise GeometryValidationError("first_polarity must be -1 or +1")
        try:
            authority = PermanentMagnetAuthority(self.permanent_magnet_authority)
        except ValueError as error:
            raise GeometryValidationError(
                "unsupported generator permanent-magnet authority"
            ) from error
        object.__setattr__(self, "permanent_magnet_authority", authority)


def _region(
    region_id: str,
    owner_id: str,
    role: str,
    material_id: str,
    r_inner_start_m: float,
    r_outer_start_m: float,
    z_min_m: float,
    z_max_m: float,
    *,
    r_inner_end_m: float | None = None,
    r_outer_end_m: float | None = None,
    polarity: int | None = None,
) -> MeridionalRegion:
    inner_end = r_inner_start_m if r_inner_end_m is None else r_inner_end_m
    outer_end = r_outer_start_m if r_outer_end_m is None else r_outer_end_m
    shape = (
        RegionShape.RECTANGULAR_ANNULUS
        if inner_end == r_inner_start_m and outer_end == r_outer_start_m
        else RegionShape.LINEAR_TAPER_ANNULUS
    )
    return MeridionalRegion(
        region_id=region_id,
        owner_id=owner_id,
        role=role,
        material_id=material_id,
        shape=shape,
        r_inner_start_m=r_inner_start_m,
        r_inner_end_m=inner_end,
        r_outer_start_m=r_outer_start_m,
        r_outer_end_m=outer_end,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        polarity=polarity,
    )


def standard_materials() -> tuple[MaterialDefinition, ...]:
    """Material identifiers and screening properties, not procurement grades."""

    return (
        MaterialDefinition(
            "vacuum-plasma-placeholder",
            MaterialKind.VACUUM_PLASMA,
            1.0,
            None,
            "Solver-neutral geometry placeholder; plasma constitutive response is out of scope.",
            True,
        ),
        MaterialDefinition(
            "bn-dielectric-assumed",
            MaterialKind.DIELECTRIC,
            1.0,
            2100.0,
            "Density is an explicit generic BN assumption; no vendor grade selected.",
            True,
        ),
        MaterialDefinition(
            "synthetic-smco-like-example-v1",
            MaterialKind.PERMANENT_MAGNET,
            1.05,
            8300.0,
            "Density 8300 kg/m^3 is an assumed representative SmCo value; magnetic "
            "parameters are delegated to the accepted synthetic SmCo-like magnet contract.",
            True,
        ),
        MaterialDefinition(
            "soft-iron-assumed",
            MaterialKind.SOFT_MAGNETIC,
            4000.0,
            7870.0,
            "Linear mu_r and density are screening assumptions, not a selected "
            "grade or B-H curve.",
            True,
        ),
        MaterialDefinition(
            "al6061-assumed",
            MaterialKind.NONMAGNETIC_SHIELD,
            1.0,
            2700.0,
            "Material family follows legacy FEMMrun.m; properties are generic assumptions.",
            True,
        ),
        MaterialDefinition(
            "copper-anode-assumed",
            MaterialKind.ELECTRODE,
            1.0,
            8960.0,
            "Generic copper geometry placeholder; thermal/electrical qualification is absent.",
            True,
        ),
    )


def generate_twt_inspired_ppm_stack(
    parameters: PPMStackParameters,
    *,
    evidence: tuple[EvidenceNote, ...],
) -> AxisymmetricCFTGeometry:
    """Build a CFT geometry using only the shared PPM/pole-stack hardware pattern.

    "TWT-inspired" applies to the alternating annular magnet/pole layout only.
    This generator contains no slow-wave structure or RF gain model.
    """

    exit_outer = (
        parameters.chamber_outer_radius_m
        if parameters.exit_outer_radius_m is None
        else parameters.exit_outer_radius_m
    )
    chamber = ChamberDefinition(
        inner_radius_m=parameters.chamber_inner_radius_m,
        outer_radius_m=parameters.chamber_outer_radius_m,
        length_m=parameters.chamber_length_m,
        injector_length_m=parameters.injector_length_m,
        dielectric_thickness_m=parameters.dielectric_thickness_m,
        exit_length_m=parameters.exit_length_m,
        exit_outer_radius_m=exit_outer,
    )
    rules = ManufacturingRules(
        minimum_thickness_m=parameters.minimum_thickness_m,
        minimum_clearance_m=parameters.minimum_clearance_m,
        radial_tolerance_m=parameters.radial_tolerance_m,
        axial_tolerance_m=parameters.axial_tolerance_m,
        thermal_clearance_m=parameters.thermal_clearance_m,
    )
    regions: list[MeridionalRegion] = [
        _region(
            "anode",
            "anode",
            "anode",
            "copper-anode-assumed",
            chamber.inner_radius_m,
            chamber.outer_radius_m,
            -max(5.0e-4, rules.minimum_thickness_m),
            0.0,
        ),
        _region(
            "injector-zone",
            "injector",
            "injector_plasma",
            "vacuum-plasma-placeholder",
            chamber.inner_radius_m,
            chamber.outer_radius_m,
            0.0,
            chamber.injector_length_m,
        ),
    ]
    straight_end = chamber.exit_start_m if chamber.exit_length_m > 0.0 else chamber.length_m
    regions.append(
        _region(
            "channel-straight",
            "chamber",
            "channel_plasma",
            "vacuum-plasma-placeholder",
            chamber.inner_radius_m,
            chamber.outer_radius_m,
            chamber.injector_length_m,
            straight_end,
        )
    )
    if chamber.exit_length_m > 0.0:
        regions.extend(
            (
                _region(
                    "channel-divergent-exit",
                    "exit",
                    "channel_plasma",
                    "vacuum-plasma-placeholder",
                    chamber.inner_radius_m,
                    chamber.outer_radius_m,
                    straight_end,
                    chamber.length_m,
                    r_inner_end_m=chamber.inner_radius_m,
                    r_outer_end_m=chamber.exit_outer_radius_m,
                ),
                _region(
                    "dielectric-divergent-exit",
                    "dielectric-wall",
                    "dielectric_wall",
                    "bn-dielectric-assumed",
                    chamber.outer_radius_m,
                    chamber.outer_radius_m + chamber.dielectric_thickness_m,
                    straight_end,
                    chamber.length_m,
                    r_inner_end_m=chamber.exit_outer_radius_m,
                    r_outer_end_m=(
                        chamber.exit_outer_radius_m + chamber.dielectric_thickness_m
                    ),
                ),
            )
        )
    regions.append(
        _region(
            "dielectric-straight",
            "dielectric-wall",
            "dielectric_wall",
            "bn-dielectric-assumed",
            chamber.outer_radius_m,
            chamber.outer_radius_m + chamber.dielectric_thickness_m,
            0.0,
            straight_end,
        )
    )

    magnet_regions: list[MeridionalRegion] = []
    stage_data: list[PPMStage] = []
    magnet_bounds: list[tuple[float, float]] = []
    for index, (center, thickness) in enumerate(
        zip(parameters.stage_centers_m, parameters.magnet_axial_thicknesses_m)
    ):
        polarity = parameters.first_polarity * (-1 if index % 2 else 1)
        z_min = center - thickness / 2.0
        z_max = center + thickness / 2.0
        if z_min < 0.0 or z_max > chamber.length_m:
            raise GeometryValidationError("magnet stages must lie within the axial envelope")
        magnet_id = f"magnet-{index + 1:02d}"
        pole_id = f"pole-{index + 1:02d}" if index < len(parameters.stage_centers_m) - 1 else None
        magnet_regions.append(
            _region(
                magnet_id,
                f"stage-{index + 1:02d}",
                "permanent_magnet",
                "synthetic-smco-like-example-v1",
                parameters.magnet_inner_radius_m,
                parameters.magnet_outer_radius_m,
                z_min,
                z_max,
                polarity=polarity,
            )
        )
        magnet_bounds.append((z_min, z_max))
        stage_z_max = (
            parameters.stage_centers_m[index + 1]
            - parameters.magnet_axial_thicknesses_m[index + 1] / 2.0
            if index < len(parameters.stage_centers_m) - 1
            else z_max
        )
        stage_data.append(
            PPMStage(
                stage_id=f"stage-{index + 1:02d}",
                index=index,
                center_z_m=center,
                pitch_m=parameters.stage_pitch_m,
                z_min_m=z_min,
                z_max_m=stage_z_max,
                magnet_region_id=magnet_id,
                pole_after_region_id=pole_id,
                magnetization=(
                    MagnetizationDirection.AXIAL_POSITIVE
                    if polarity == 1
                    else MagnetizationDirection.AXIAL_NEGATIVE
                ),
            )
        )
    regions.extend(magnet_regions)
    for index, ((_, left_max), (right_min, _)) in enumerate(
        zip(magnet_bounds, magnet_bounds[1:])
    ):
        if right_min <= left_max:
            raise GeometryValidationError("adjacent magnet stages overlap or leave no pole gap")
        regions.append(
            _region(
                f"pole-{index + 1:02d}",
                f"stage-{index + 1:02d}",
                "pole_piece",
                "soft-iron-assumed",
                parameters.magnet_inner_radius_m,
                parameters.magnet_outer_radius_m,
                left_max,
                right_min,
            )
        )
    regions.extend(
        (
            _region(
                "shield-shell",
                "shield",
                "shield",
                "al6061-assumed",
                parameters.magnet_outer_radius_m,
                parameters.shield_outer_radius_m,
                0.0,
                chamber.length_m,
            ),
            _region(
                "return-yoke",
                "yoke",
                "yoke",
                "soft-iron-assumed",
                parameters.shield_outer_radius_m,
                parameters.yoke_outer_radius_m,
                0.0,
                chamber.length_m,
            ),
        )
    )
    regions.sort(
        key=lambda region: (
            region.z_min_m,
            region.r_inner_start_m,
            region.r_inner_end_m,
            region.region_id,
        )
    )
    physics_limitation = EvidenceNote(
        "twt-physics-boundary",
        "limitation",
        "TWT RF slow-wave electron-beam amplification physics is not CFT plasma "
        "propulsion physics; only the annular alternating PPM/pole focusing-stack "
        "geometry is shared here.",
        "Workstream claim boundary.",
    )
    external = (
        ExternalComponent(
            "external-cathode",
            "cathode",
            "external_non_axisymmetric_metadata",
            "Downstream/off-axis placement to be defined by integration study.",
            False,
        ),
        ExternalComponent(
            "external-neutralizer",
            "neutralizer",
            "external_non_axisymmetric_metadata",
            "External plume-neutralization hardware; no body of revolution assumed.",
            False,
        ),
    )
    return AxisymmetricCFTGeometry(
        config_id=parameters.config_id,
        title=parameters.title,
        classification="hypothetical_not_optimized_not_build_qualified",
        chamber=chamber,
        electrodes=ElectrodeDefinition(
            anode_region_id="anode",
            anode_thickness_m=max(5.0e-4, rules.minimum_thickness_m),
            injector_region_id="injector-zone",
        ),
        manufacturing=rules,
        permanent_magnet_plan=PermanentMagnetRepresentationPlan(
            plan_id=(
                f"{parameters.config_id}-"
                f"{parameters.permanent_magnet_authority.value}-v1"
            ),
            authority=parameters.permanent_magnet_authority,
        ),
        materials=standard_materials(),
        regions=tuple(regions),
        stages=tuple(stage_data),
        external_components=external,
        evidence=(physics_limitation, *evidence),
        design_variable_order=(
            "chamber_outer_radius_m",
            "chamber_length_m",
            "dielectric_thickness_m",
            "magnet_inner_radius_m",
            "magnet_outer_radius_m",
            "stage_pitch_m",
            "exit_length_m",
            "exit_outer_radius_m",
        ),
    )


def historical_envelope_baseline() -> AxisymmetricCFTGeometry:
    """Map the traceable legacy FEMM envelope, retaining explicit assumptions."""

    return generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id="historical-envelope-baseline-v1",
            title="Historical-envelope baseline (hypothetical reconstruction)",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=0.002,
            chamber_length_m=0.021,
            injector_length_m=0.001,
            dielectric_thickness_m=0.001,
            thermal_clearance_m=0.00025,
            magnet_inner_radius_m=0.0034,
            magnet_outer_radius_m=0.009,
            stage_pitch_m=0.008,
            stage_centers_m=(0.002, 0.010, 0.018),
            magnet_axial_thicknesses_m=(0.004, 0.010, 0.004),
            shield_outer_radius_m=0.011,
            yoke_outer_radius_m=0.013,
            radial_tolerance_m=5.0e-5,
            axial_tolerance_m=5.0e-5,
        ),
        evidence=(
            EvidenceNote(
                "legacy-fixed-envelope",
                "traceable",
                "The legacy axisymmetric FEMM script fixes bore radius a=2 mm, "
                "axial p=21 mm, and magnet spans 0–4, 5–15, and 16–20 mm.",
                "FYP/FEMMrun.m lines 24–42 and 47–68.",
            ),
            EvidenceNote(
                "legacy-commented-radii",
                "traceable",
                "Legacy comments show example b=3 mm, d=9 mm, e=11 mm, while the "
                "actual values are optimization variables x(4:6), not qualified dimensions.",
                "FYP/FEMMrun.m lines 24–34.",
            ),
            EvidenceNote(
                "historical-clearance-assumption",
                "assumption",
                "A 0.40 mm nominal radial gap shifts the magnet inner radius from "
                "the commented b=3 mm to 3.40 mm, preserving 0.25 mm thermal "
                "clearance after tolerance; 13 mm yoke outer radius is assumed.",
                "Authored workstream assumption; not present in the FYP.",
            ),
        ),
    )


def compact_high_gradient_stack() -> AxisymmetricCFTGeometry:
    return generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id="compact-high-gradient-stack-v1",
            title="Compact high-gradient PPM stack (hypothetical)",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=0.0015,
            chamber_length_m=0.020,
            injector_length_m=0.001,
            dielectric_thickness_m=0.00075,
            thermal_clearance_m=0.00025,
            magnet_inner_radius_m=0.0026,
            magnet_outer_radius_m=0.006,
            stage_pitch_m=0.004,
            stage_centers_m=(0.002, 0.006, 0.010, 0.014, 0.018),
            magnet_axial_thicknesses_m=(0.0025,) * 5,
            shield_outer_radius_m=0.00675,
            yoke_outer_radius_m=0.008,
            radial_tolerance_m=2.5e-5,
            axial_tolerance_m=2.5e-5,
        ),
        evidence=(
            EvidenceNote(
                "compact-assumption",
                "assumption",
                "All dimensions are a compact parametric screening choice; 'high-gradient' "
                "describes the short pitch intent and is not a computed field claim.",
                "Authored hypothetical variant.",
            ),
        ),
    )


def divergent_exit_stack() -> AxisymmetricCFTGeometry:
    return generate_twt_inspired_ppm_stack(
        PPMStackParameters(
            config_id="divergent-exit-stack-v1",
            title="Divergent-exit PPM stack (hypothetical)",
            chamber_inner_radius_m=0.0,
            chamber_outer_radius_m=0.002,
            chamber_length_m=0.024,
            injector_length_m=0.0015,
            dielectric_thickness_m=0.001,
            thermal_clearance_m=0.00025,
            magnet_inner_radius_m=0.0044,
            magnet_outer_radius_m=0.009,
            stage_pitch_m=0.006,
            stage_centers_m=(0.003, 0.009, 0.015, 0.021),
            magnet_axial_thicknesses_m=(0.004,) * 4,
            shield_outer_radius_m=0.010,
            yoke_outer_radius_m=0.012,
            exit_length_m=0.006,
            exit_outer_radius_m=0.003,
        ),
        evidence=(
            EvidenceNote(
                "divergent-assumption",
                "assumption",
                "The final 6 mm linear divergence from 2 to 3 mm channel outer radius "
                "is a geometry study variable, not a performance-backed nozzle design.",
                "Authored hypothetical variant.",
            ),
        ),
    )


def reference_variants() -> tuple[AxisymmetricCFTGeometry, ...]:
    return (
        historical_envelope_baseline(),
        compact_high_gradient_stack(),
        divergent_exit_stack(),
    )
