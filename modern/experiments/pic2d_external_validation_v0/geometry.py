"""Brandt 2016 micro-HEMPT mapped onto the parametric CFT geometry v1.1 and onto the pic2d grid.

Two frames:

* the FEM frame of the geometry model (``cft_revival.geometry`` v1.1): magnets and poles must lie inside ``[0, chamber.length_m]``;
* the anode frame of the reference and of the PIC (``z = 0`` at the anode surface, the channel exit at 14 mm).

Brandt places the anode surface at the mid-plane of the first magnet, so magnet 1 spans -2.5..2.5 mm in the anode frame.  The
FEM geometry therefore starts ``AXIAL_OFFSET_M = 2.5 mm`` behind the anode (``z_FEM = z + 2.5 mm``); the 2.5 mm "injector zone"
of the geometry model stands for the neutral-gas inlet section behind the anode that the paper says exists but does not model.
The PIC node map applies the offset when it evaluates the bound P2 solution (``fields.brandt_field_map``).

Every approximation of the mapping is listed in ``APPROXIMATIONS`` with its expected effect and how it is quantified.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from cft_revival.geometry import AxisymmetricCFTGeometry
from cft_revival.geometry.generators import PPMStackParameters, generate_twt_inspired_ppm_stack
from cft_revival.geometry.model import EvidenceNote, MaterialDefinition, MaterialKind
from cft_revival.pic2d.models import ChannelGeometry, Grid2D

from . import reference

CONFIG_ID = "brandt2016-micro-hempt-v1"
AXIAL_OFFSET_M = 2.5e-3                    # z_FEM = z_anode + offset (magnet 1 straddles the anode plane in the reference)
CHANNEL_RADIUS_M = 1.5e-3
CHANNEL_LENGTH_M = 14.0e-3
FEM_CHAMBER_LENGTH_M = CHANNEL_LENGTH_M + AXIAL_OFFSET_M          # 16.5 mm: magnet 3 ends at 16.0 mm (13.5 mm anode frame)
MAGNET_LENGTH_M = 5.0e-3
MAGNET_INNER_RADIUS_M = 2.5e-3
MAGNET_OUTER_RADIUS_M = 15.0e-3
RING_LENGTH_M = 0.5e-3
RING_OUTER_RADIUS_REFERENCE_M = 8.0e-3     # NOT representable (poles share the magnet radii in v1.1) - approximation A3
STAGE_PITCH_M = MAGNET_LENGTH_M + RING_LENGTH_M                    # 5.5 mm
STAGE_CENTRES_FEM_M = (2.5e-3, 8.0e-3, 13.5e-3)                    # magnet centres in the FEM frame (0.0, 5.5, 11.0 mm anode frame)
DIELECTRIC_THICKNESS_REFERENCE_M = 1.0e-3
DIELECTRIC_THICKNESS_MODEL_M = 0.9e-3      # 0.1 mm clearance to the magnet bore demanded by the manufacturing rules - approximation A5 (mu_r 1 either way)
SHIELD_OUTER_RADIUS_M = 15.5e-3
YOKE_OUTER_RADIUS_M = 16.0e-3
YOKE_PLACEHOLDER_MATERIAL_ID = "nonmagnetic-yoke-placeholder-mu1"
POLE_VACUUM_MATERIAL_ID = "pole-placeholder-mu1-sensitivity"

# the reference's PIC box (anode frame)
PLUME_RADIUS_M = 5.12e-3
PLUME_LENGTH_M = 20.48e-3 - CHANNEL_LENGTH_M                        # 6.48 mm
BODY_DIELECTRIC_RADIUS_M = 2.5e-3
REFERENCE_CELL_M = 20.0e-6
DOMAIN_OPTIONS = ("channel", "plume-brandt")

# interior distance-ring centres and the exit null the field gates test (anode frame)
RING_CENTRES_ANODE_M = (2.75e-3, 8.25e-3)
EXIT_RING_ANODE_M = (13.5e-3, 14.0e-3)


APPROXIMATIONS: tuple[dict[str, Any], ...] = (
    {"id": "A1", "item": "anode inside the first magnet", "reference": "anode surface at the mid-plane of magnet 1 (magnet 1 spans -2.5..2.5 mm)",
     "represented": "exactly, by an axial offset: the FEM geometry starts 2.5 mm behind the anode (injector zone 0..2.5 mm FEM = the unmodelled inlet section); "
                    "the PIC grid samples the field at z_FEM = z + 2.5 mm", "effect": "none on B (exact geometry); the PIC anode plane sits at the magnet-1 mid-plane as in the reference",
     "quantified_by": "field gate G5 (axis |B| maximum at the magnet-3 centre z = 11.0 mm)"},
    {"id": "A2", "item": "end distance rings", "reference": "five 0.5 mm soft-iron rings; the layout implies rings at both stack ends (anode side at -3.0..-2.5 mm, exit side 13.5..14.0 mm) "
                                                             "plus two interior rings; the fifth ring's position is not stated (stack 17.5 vs stated 18 mm)",
     "represented": "the two INTERIOR rings only (v1.1 pole pieces exist between adjacent magnets); no end rings",
     "effect": "the exit-side ring would pull the exit null toward the channel exit and raise the wall field at 13.5-14 mm; the anode-side ring is behind the anode plane",
     "quantified_by": "field gates G3 (exit null 16 +- 1.5 mm) and G4 (|B| 0.05 +- 0.025 T at z = 17 mm); the no-ring sensitivity solve brackets the ring effect"},
    {"id": "A3", "item": "distance-ring outer radius", "reference": "8 mm (magnets 15 mm)", "represented": "15 mm (v1.1 poles share the magnet radii)",
     "effect": "extra iron between the magnets at 8-15 mm: more flux shorted in the outer region, marginally stronger cusp focusing",
     "quantified_by": "no-ring sensitivity solve (pole mu_r -> 1): the represented ring lies between 'no ring' and 'full-radius ring'; the shift of the gate quantities between the two solves is the bracket"},
    {"id": "A4", "item": "iron constitutive law", "reference": "FEMM 'Carbon steel forgings, annealed' (nonlinear B-H, saturates ~2 T)", "represented": "linear mu_r 4000 (the L1b v1.1 material)",
     "effect": "a 0.5 mm ring between two 12.5 mm-thick SmCo pole faces saturates in reality; the linear ring over-focuses -> cusp wall field over-predicted",
     "quantified_by": "gate G6 (wall |B| at the cusps within 0.1-0.35 T of the reference's 'about 0.2 T and lower'); the no-ring solve is the lower bracket"},
    {"id": "A5", "item": "dielectric tube thickness", "reference": "1.0 mm (Al2O3, r 1.5-2.5 mm)", "represented": "0.9 mm in the FEM geometry (manufacturing rule: strictly positive clearance to the magnet bore); "
                                                                                                          "the PIC front face keeps 2.5 mm as the body-dielectric radius",
     "effect": "none on B (mu_r 1 either side of the gap); none on the PIC (front face radius set directly)", "quantified_by": "not needed"},
    {"id": "A6", "item": "return yoke / housing", "reference": "none in the FEMM model (magnets + distance rings only); housing material unspecified", "represented": "the mandatory v1.1 yoke role filled by a mu_r = 1 placeholder ring (magnetically inert); Al shield mu_r 1",
     "effect": "none", "quantified_by": "not needed (mu_r 1 regions do not enter the mesh grading or the field)"},
    {"id": "A7", "item": "magnet remanence", "reference": "SmCo, grade not stated (FEMM library SmCo 20-32 MGOe: B_r 0.9-1.15 T)", "represented": "the repository's SmCo-like contract (B_r 1.05 T, recoil mu_r 1.05), "
                                                                                                                                       "then ONE linear post-scale so that |B|(r = 0, z = 11 mm) = 0.6 T (the published anchor)",
     "effect": "the absolute field level is calibrated on the published anchor, not predicted; the scale must stay inside the SmCo grade band",
     "quantified_by": "gate G1 (scale in [0.80, 1.20] of nominal = remanence 0.84-1.26 T); the anchor's own precision ('about 0.6 T', +-0.05 T) is carried as the field's u_input"},
    {"id": "A8", "item": "magnetisation uniformity / recoil", "reference": "FEMM linear-demagnetisation SmCo", "represented": "uniform axial remanence with recoil mu_r 1.05 (same model class)", "effect": "none expected", "quantified_by": "not needed"},
    {"id": "A9", "item": "channel exit and plume box", "reference": "domain 20.48 x 5.12 mm: channel + 6.48 mm plume, grounded body 2.5 <= r <= 5.12 mm, 0 V far boundaries",
     "represented": "channel-only option (z <= 14 mm, Dirichlet 0 V exit plane, the ss-v4 template) as the PRIMARY protocol; the reference box as the 'plume-brandt' mapping (256 x 1024 cells at 20 um, "
                    "the published grid exactly) for the costed plume option",
     "effect": "channel-only forces the whole 400 V drop inside the channel; the exit cell, the exit cusp region and every plume quantity are not comparable (comparison spec 'comparable_under')",
     "quantified_by": "declared per comparison row; the plume option is the follow-up"},
)


def _yoke_placeholder() -> MaterialDefinition:
    return MaterialDefinition(YOKE_PLACEHOLDER_MATERIAL_ID, MaterialKind.SOFT_MAGNETIC, 1.0, 7870.0,
                              "Placeholder for the mandatory yoke role: the reference (Brandt 2017 ch. 7 FEMM model) has no return yoke; mu_r 1 makes the region magnetically inert.", True)


def _pole_vacuum_placeholder() -> MaterialDefinition:
    return MaterialDefinition(POLE_VACUUM_MATERIAL_ID, MaterialKind.SOFT_MAGNETIC, 1.0, 7870.0,
                              "Sensitivity variant only: the distance rings removed (mu_r 1) to bracket approximations A2-A4.", True)


def brandt_micro_hempt_geometry(*, pole_vacuum: bool = False) -> AxisymmetricCFTGeometry:
    """The reconstructed micro-HEMPT stack in the v1.1 geometry contract (FEM frame; see the module docstring for the frames).

    ``pole_vacuum=True`` is the sensitivity variant (distance rings at mu_r 1) used to bracket approximations A2-A4; it is NOT the
    field the PIC uses.
    """

    evidence = (
        EvidenceNote("brandt-2016-channel", "traceable", "Discharge channel Z_thr = 14 mm from the anode surface, R_thr = 1.5 mm; Al2O3 tube 1.5-2.5 mm; grounded body 2.5-5.12 mm.",
                     f"Brandt et al. 2016, doi:{reference.DOI}, pp. Pb_235-Pb_237."),
        EvidenceNote("brandt-2017-magnet-stack", "traceable", "Three SmCo ring magnets 5 mm long (r 2.5-15 mm) alternating in polarity, separated by 0.5 mm soft-iron distance rings (r 2.5-8 mm); "
                     "anode at the mid-plane of the first magnet; stack length 18 mm (3 x 5 + 5 x 0.5 = 17.5 mm; 0.5 mm discrepancy).",
                     f"Brandt 2017 thesis ({reference.THESIS['urn']}), chapters 6-7."),
        EvidenceNote("frame-offset", "assumption", "The FEM geometry starts 2.5 mm behind the anode so that magnet 1 (which straddles the anode plane) lies inside the v1.1 axial envelope; "
                     "the PIC samples the field at z_FEM = z + 2.5 mm.", "Approximation A1 of experiments/pic2d_external_validation_v0/geometry.py."),
        EvidenceNote("ring-radius-and-ends", "limitation", "The v1.1 pole pieces share the magnet radii (15 mm vs the reference's 8 mm rings) and exist only between magnets (the reference's "
                     "end rings are not represented); the iron is linear mu_r 4000 where the reference used a nonlinear FEMM steel.",
                     "Approximations A2-A4; bracketed by the no-ring sensitivity solve."),
        EvidenceNote("dielectric-clearance", "assumption", "Dielectric thickness 0.9 mm instead of 1.0 mm so that the geometry contract's clearance rule holds; the material is mu_r 1, so B is unaffected.",
                     "Approximation A5."),
        EvidenceNote("no-yoke", "assumption", "The reference's FEMM model has no return yoke; the mandatory yoke role is filled with a mu_r 1 placeholder.", "Approximation A6."),
    )
    parameters = PPMStackParameters(
        config_id=CONFIG_ID if not pole_vacuum else f"{CONFIG_ID}-no-rings-sensitivity",
        title="Brandt 2016 down-scaled HEMP thruster (micro-HEMPT) - reconstruction for external validation v0" + (" - NO-RING SENSITIVITY VARIANT" if pole_vacuum else ""),
        chamber_inner_radius_m=0.0,
        chamber_outer_radius_m=CHANNEL_RADIUS_M,
        chamber_length_m=FEM_CHAMBER_LENGTH_M,
        injector_length_m=AXIAL_OFFSET_M,
        dielectric_thickness_m=DIELECTRIC_THICKNESS_MODEL_M,
        thermal_clearance_m=5.0e-5,
        magnet_inner_radius_m=MAGNET_INNER_RADIUS_M,
        magnet_outer_radius_m=MAGNET_OUTER_RADIUS_M,
        stage_pitch_m=STAGE_PITCH_M,
        stage_centers_m=STAGE_CENTRES_FEM_M,
        magnet_axial_thicknesses_m=(MAGNET_LENGTH_M,) * 3,
        shield_outer_radius_m=SHIELD_OUTER_RADIUS_M,
        yoke_outer_radius_m=YOKE_OUTER_RADIUS_M,
        radial_tolerance_m=1.0e-5,
        axial_tolerance_m=1.0e-5,
        minimum_clearance_m=5.0e-5,
    )
    base = generate_twt_inspired_ppm_stack(parameters, evidence=evidence)
    materials = base.materials + (_yoke_placeholder(),)
    regions = [dataclasses.replace(r, material_id=YOKE_PLACEHOLDER_MATERIAL_ID) if r.role == "yoke" else r for r in base.regions]
    if pole_vacuum:
        materials = materials + (_pole_vacuum_placeholder(),)
        regions = [dataclasses.replace(r, material_id=POLE_VACUUM_MATERIAL_ID) if r.role == "pole_piece" else r for r in regions]
    return dataclasses.replace(base, materials=materials, regions=tuple(regions))


def fem_frame_z(z_anode_m: float) -> float:
    return float(z_anode_m) + AXIAL_OFFSET_M


def anode_frame_z(z_fem_m: float) -> float:
    return float(z_fem_m) - AXIAL_OFFSET_M


def stack_table() -> list[dict[str, Any]]:
    """The represented stack in BOTH frames next to the reference's layout (the geometry mapping table)."""

    geometry = brandt_micro_hempt_geometry()
    rows = []
    for region in geometry.regions:
        if region.role not in ("permanent_magnet", "pole_piece"):
            continue
        rows.append({"region_id": region.region_id, "role": region.role, "polarity": region.polarity,
                     "z_fem_m": [region.z_min_m, region.z_max_m], "z_anode_m": [anode_frame_z(region.z_min_m), anode_frame_z(region.z_max_m)],
                     "r_m": [region.r_inner_start_m, region.r_outer_start_m],
                     "reference_r_m": [MAGNET_INNER_RADIUS_M, MAGNET_OUTER_RADIUS_M if region.role == "permanent_magnet" else RING_OUTER_RADIUS_REFERENCE_M]})
    rows.append({"region_id": "exit-ring (not represented)", "role": "pole_piece", "polarity": None, "z_fem_m": [fem_frame_z(EXIT_RING_ANODE_M[0]), fem_frame_z(EXIT_RING_ANODE_M[1])],
                 "z_anode_m": list(EXIT_RING_ANODE_M), "r_m": None, "reference_r_m": [MAGNET_INNER_RADIUS_M, RING_OUTER_RADIUS_REFERENCE_M]})
    rows.append({"region_id": "anode-side ring (not represented)", "role": "pole_piece", "polarity": None, "z_fem_m": [-0.5e-3, 0.0], "z_anode_m": [-3.0e-3, -2.5e-3], "r_m": None,
                 "reference_r_m": [MAGNET_INNER_RADIUS_M, RING_OUTER_RADIUS_REFERENCE_M]})
    return rows


# --------------------------------------------------------------------------------------------------------------------------
# PIC mapping (anode frame)
# --------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PicMapping:
    """PIC geometry and grid for one domain option; every grid line of the reference is hit exactly at 20 um (snaps recorded anyway)."""

    design_id: str
    domain: str
    geometry: ChannelGeometry
    grid: Grid2D
    snaps: dict[str, Any]
    axial_offset_m: float = AXIAL_OFFSET_M

    def to_dict(self) -> dict[str, Any]:
        return {"design_id": self.design_id, "domain": self.domain, "geometry": self.geometry.to_dict(), "grid": self.grid.to_dict(), "snaps": self.snaps,
                "axial_offset_m": self.axial_offset_m}


def _snap(value: float, spacing: float) -> dict[str, float | int]:
    cells = round(value / spacing)
    return {"value_m": float(value), "snapped_m": cells * spacing, "cells": cells, "error_m": cells * spacing - float(value)}


def pic_mapping(domain: str = "channel", *, target_cell_m: float = REFERENCE_CELL_M) -> PicMapping:
    """Channel-only (75 x 700 at 20 um) or the reference's full box (256 x 1024 at 20 um: the published grid exactly)."""

    if domain not in DOMAIN_OPTIONS:
        raise ValueError(f"unknown domain option {domain!r}; known {DOMAIN_OPTIONS}")
    n_r = max(2, round(CHANNEL_RADIUS_M / target_cell_m))
    n_z = max(2, round(CHANNEL_LENGTH_M / target_cell_m))
    dr, dz = CHANNEL_RADIUS_M / n_r, CHANNEL_LENGTH_M / n_z
    snaps: dict[str, Any] = {"dr_m": dr, "dz_m": dz, "bore_cells": n_r, "channel_cells": n_z, "target_cell_m": target_cell_m,
                             "exit_radius": _snap(CHANNEL_RADIUS_M, dr), "cone_start": _snap(CHANNEL_LENGTH_M, dz)}
    if domain == "channel":
        geometry = ChannelGeometry(CHANNEL_RADIUS_M, 0.0, CHANNEL_LENGTH_M, CHANNEL_LENGTH_M, CHANNEL_RADIUS_M)
        return PicMapping(CONFIG_ID, domain, geometry, Grid2D(geometry, n_r, n_z), snaps)
    body = _snap(BODY_DIELECTRIC_RADIUS_M, dr)
    plume_r = _snap(PLUME_RADIUS_M, dr)
    plume_z = _snap(PLUME_LENGTH_M, dz)
    snaps.update({"body_dielectric_radius": body, "plume_radius": plume_r, "plume_length": plume_z})
    geometry = ChannelGeometry(CHANNEL_RADIUS_M, 0.0, CHANNEL_LENGTH_M, CHANNEL_LENGTH_M, CHANNEL_RADIUS_M, plume_radius_m=plume_r["snapped_m"],
                               plume_length_m=plume_z["snapped_m"], body_dielectric_radius_m=body["snapped_m"])
    return PicMapping(CONFIG_ID, domain, geometry, Grid2D(geometry, plume_r["cells"], n_z + plume_z["cells"]), snaps)


def channel_volume_m3(mapping: PicMapping) -> float:
    g = mapping.geometry
    return math.pi * g.bore_radius_m**2 * (g.z_max_m - g.z_min_m)


def worst_snap_in_cells(mapping: PicMapping) -> float:
    return max(abs(v["error_m"]) / (mapping.grid.dr_m if "radius" in k else mapping.grid.dz_m) for k, v in mapping.snaps.items() if isinstance(v, dict))


def mapping_table() -> dict[str, Any]:
    """The geometry-mapping record (README table + protocol.json): reference value, represented value, approximation id."""

    geometry = brandt_micro_hempt_geometry()
    return {
        "config_id": geometry.config_id, "geometry_sha256": geometry.canonical_sha256, "schema_version": geometry.schema_version,
        "frames": {"axial_offset_m": AXIAL_OFFSET_M, "rule": "z_FEM = z_anode + offset; the PIC and the reference use the anode frame"},
        "rows": [
            {"item": "channel radius", "reference_m": CHANNEL_RADIUS_M, "represented_m": geometry.chamber.outer_radius_m, "approximation": None},
            {"item": "channel length (anode -> exit)", "reference_m": CHANNEL_LENGTH_M, "represented_m": anode_frame_z(geometry.chamber.length_m) + 0.0, "approximation": "A1 (FEM chamber 16.5 mm = 2.5 mm inlet zone + 14 mm channel)"},
            {"item": "dielectric thickness", "reference_m": DIELECTRIC_THICKNESS_REFERENCE_M, "represented_m": geometry.chamber.dielectric_thickness_m, "approximation": "A5"},
            {"item": "magnet count / polarity", "reference": "3, alternating", "represented": f"{len(geometry.stages)}, alternating (first +)", "approximation": None},
            {"item": "magnet axial length", "reference_m": MAGNET_LENGTH_M, "represented_m": MAGNET_LENGTH_M, "approximation": None},
            {"item": "magnet inner / outer radius", "reference_m": [MAGNET_INNER_RADIUS_M, MAGNET_OUTER_RADIUS_M], "represented_m": [MAGNET_INNER_RADIUS_M, MAGNET_OUTER_RADIUS_M], "approximation": None},
            {"item": "magnet centres (anode frame)", "reference_m": [anode_frame_z(z) for z in STAGE_CENTRES_FEM_M], "represented_m": [anode_frame_z(s.center_z_m) for s in geometry.stages], "approximation": "A1"},
            {"item": "distance rings", "reference": "5 x 0.5 mm, r 2.5-8 mm, annealed carbon steel (nonlinear)", "represented": "2 interior x 0.5 mm, r 2.5-15 mm, linear mu_r 4000", "approximation": "A2, A3, A4"},
            {"item": "distance-ring centres (anode frame)", "reference_m": list(RING_CENTRES_ANODE_M) + ["exit ring 13.5-14.0", "anode-side ring -3.0..-2.5"],
             "represented_m": [anode_frame_z(0.5 * (r.z_min_m + r.z_max_m)) for r in geometry.regions if r.role == "pole_piece"], "approximation": "A2"},
            {"item": "return yoke", "reference": "none", "represented": "mu_r 1 placeholder (inert)", "approximation": "A6"},
            {"item": "remanence", "reference": "SmCo, grade not stated", "represented": "1.05 T nominal, post-scaled to the 0.6 T axis anchor", "approximation": "A7"},
            {"item": "PIC box (primary)", "reference": "20.48 x 5.12 mm incl. 6.48 mm plume", "represented": "channel-only 14 x 1.5 mm, 75 x 700 cells at 20 um", "approximation": "A9"},
            {"item": "PIC box (plume option)", "reference": "20.48 x 5.12 mm, 1024 x 256 cells", "represented": "1024 x 256 cells at 20 um, body dielectric 2.5 mm, plume radius 5.12 mm", "approximation": None},
        ],
        "approximations": list(APPROXIMATIONS),
        "stack": stack_table(),
    }


__all__ = [
    "APPROXIMATIONS",
    "AXIAL_OFFSET_M",
    "BODY_DIELECTRIC_RADIUS_M",
    "CHANNEL_LENGTH_M",
    "CHANNEL_RADIUS_M",
    "CONFIG_ID",
    "DOMAIN_OPTIONS",
    "EXIT_RING_ANODE_M",
    "FEM_CHAMBER_LENGTH_M",
    "MAGNET_INNER_RADIUS_M",
    "MAGNET_OUTER_RADIUS_M",
    "PLUME_LENGTH_M",
    "PLUME_RADIUS_M",
    "REFERENCE_CELL_M",
    "RING_CENTRES_ANODE_M",
    "STAGE_CENTRES_FEM_M",
    "PicMapping",
    "anode_frame_z",
    "brandt_micro_hempt_geometry",
    "channel_volume_m3",
    "fem_frame_z",
    "mapping_table",
    "pic_mapping",
    "stack_table",
    "worst_snap_in_cells",
]
