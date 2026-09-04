"""Design list of the PIC design mini-sweep v1 and the catalogue -> PIC geometry mapping.

Every sweep design is one of the accepted catalogue designs, rebuilt through the sealed pipelines the earlier
campaigns proved (identity hashes recorded next to the geometry):

* ``divergent-exit-stack`` - the fem_reference reference design every pic2d run so far used
  (``cft_revival.geometry.generators.divergent_exit_stack``; cusp/cell catalogue entry of cusp topology
  search v3.1, set ``p2_divergent_exit``);
* ``l1a-gs-v2-NNN-...`` - accepted L1a sweep-v2 designs, rebuilt by the wall-loss geometry screening
  pipeline (``experiments.orbit_wall_loss_geometry_screening_v1.designs.rebuild_case``, identity proven
  against the sealed raw record) - the held-out set of sweep v3 and the design set of screening v2;
* ``l1a-gs-v3-NNN-...`` - accepted L1a sweep-v3 Sobol designs, rebuilt from the preregistered (seed, index)
  with the sweep-v3 builder and proven against the sealed sweep-v3 design authorities.

The PIC mesh (``cft_revival.pic2d.models``) represents a straight bore, an optional linear divergent exit
cone and (v2.0+) an L-shaped plume box on a uniform node grid.  A catalogue design is mapped onto it by
``pic_geometry``: the bore radius sets ``dr`` exactly (``dr = r_w / n_r`` with ``n_r = round(r_w / 50 um)``),
the channel length sets ``dz`` exactly, and the three radii the grid must hit (exit radius, dielectric
front-face radius = magnet inner radius, plume radius = return-yoke outer radius) and the cone start are
SNAPPED to the nearest grid line; every snap is recorded in metres and in cells.  The record is the
disclosure the preregistration must carry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.geometry import AxisymmetricCFTGeometry
from cft_revival.geometry.generators import divergent_exit_stack
from cft_revival.pic2d.models import ChannelGeometry, Grid2D

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
FIELDS_DIR = EXPERIMENT / "fields"

V3_CATALOGUE_PATH = MODERN / "experiments" / "l1a_geometry_sweep_v3" / "results" / "artifacts" / "cusp-cell-catalogue-v3.json"
V3_MANIFEST_PATH = MODERN / "experiments" / "l1a_geometry_sweep_v3" / "results" / "manifest.json"
V31_CATALOGUE_PATH = MODERN / "experiments" / "cusp_topology_search_v3_1" / "results" / "artifacts" / "cusp-cell-catalogue.json"
L1B_DATASET_PATH = MODERN / "experiments" / "l1b_hemp_confirmation_v1_1" / "results" / "artifacts" / "confirmation-dataset.json"
L1B_RESULTS = MODERN / "experiments" / "l1b_hemp_confirmation_v1_1" / "results"
SCREENING_V2_DATASET_PATH = MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v2" / "results" / "artifacts" / "geometry-wall-loss-dataset-v2.json"

REFERENCE_DESIGN_ID = "divergent-exit-stack"
TARGET_CELL_M = 5.0e-5          # the 50 um node spacing of every accepted pic2d run (v1.3 fine cases, v2.0, v2.1)
DEFAULT_PLUME_LENGTHS_M = {"channel": None, "plume-12mm": 0.012, "plume-24mm": 0.024}
DOMAIN_OPTIONS = tuple(DEFAULT_PLUME_LENGTHS_M)


@dataclass(frozen=True)
class SweepDesign:
    """One sweep design: catalogue identity, role in the rho ladder and run priority."""

    design_id: str
    role: str                 # reference | low-rho | mid-rho | hemp-like | hemp-like-four-cusp
    source: str               # p2_divergent_exit | sweep_v2 | sobol_v3
    priority: int             # GPU serial order (1 first)
    optional: bool = False    # the fifth design runs only if the budget allows
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The proposal (design-selection section of the README): four primary designs spanning Koch rho with the SAME
# wall-cusp count as the reference (3 cusps -> 4 cells, a 1:1 map onto the four-cell plasma network v2), plus
# the strongest four-cusp HEMP-like design as an optional fifth (fills all four Kornfeld probability slots).
SWEEP_DESIGNS: tuple[SweepDesign, ...] = (
    SweepDesign(REFERENCE_DESIGN_ID, "reference", "p2_divergent_exit", 1, False,
                "every pic2d run so far; rho_cons 0.60-0.62 under the qualified P2 field; fields exist (authority level-1, padding-1.5)"),
    SweepDesign("l1a-gs-v3-056-effcbc8686", "hemp-like", "sobol_v3", 2, False,
                "3 cusps / 4 cells, no exit taper, rho 1.99 (L1a) -> 2.37 (L1b iron), wall |B| 0.21 T; L1b level-0 mesh 33.6 deg"),
    SweepDesign("l1a-gs-v2-047-e3196a8aa5", "low-rho", "sweep_v2", 3, False,
                "4 stages, 3 cusps / 4 cells, rho 0.349 (L1a), r_w/L 0.24, 1.68 mm exit taper; screening-v2 P(wall) exists"),
    SweepDesign("l1a-gs-v3-009-d0c686b4aa", "mid-rho", "sobol_v3", 4, False,
                "4 stages, 3 cusps / 4 cells, no exit taper, rho 0.899 (L1a), r_w/L 0.47; no L1b / screening record (new P2 solve gives rho under iron)"),
    SweepDesign("l1a-gs-v3-106-ccec1c8b2f", "hemp-like-four-cusp", "sobol_v3", 5, True,
                "5 stages, 4 cusps / 5 cells (the five-stage four-cusp HEMP-like class), no exit taper, rho 2.56 (L1a) -> 2.93 (L1b iron), wall |B| 0.31 T under iron"),
)


def sweep_design(design_id: str) -> SweepDesign:
    for design in SWEEP_DESIGNS:
        if design.design_id == design_id:
            return design
    raise KeyError(f"{design_id} is not a sweep design")


def design_ids(*, include_optional: bool = True) -> tuple[str, ...]:
    return tuple(d.design_id for d in SWEEP_DESIGNS if include_optional or not d.optional)


# --------------------------------------------------------------------------
# Sealed catalogue lookups (read-only)
# --------------------------------------------------------------------------

_CACHE: dict[str, Any] = {}


def _cached_json(path: Path) -> dict[str, Any]:
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = strict_json_file(path)
    return _CACHE[key]


def v3_catalogue() -> dict[str, Any]:
    """The sealed sweep-v3 cusp/cell catalogue (224 entries: 128 sobol_v3 + 96 sweep_v2), byte-checked against its manifest."""

    value = _cached_json(V3_CATALOGUE_PATH)
    manifest = _cached_json(V3_MANIFEST_PATH)
    entry = next(item for item in manifest["artifacts"] if item["path"] == "artifacts/cusp-cell-catalogue-v3.json" and item["type"] == "file")
    if hashlib.sha256(V3_CATALOGUE_PATH.read_bytes()).hexdigest() != entry["byte_sha256"]:
        raise ValueError("the sweep-v3 catalogue bytes differ from the sweep-v3 manifest")
    return value


def v31_catalogue() -> dict[str, Any]:
    return _cached_json(V31_CATALOGUE_PATH)


def catalogue_entry(design_id: str) -> dict[str, Any]:
    """Cusp/cell catalogue entry: sweep-v3 catalogue for L1a designs, topology v3.1 for the P2 reference."""

    if design_id == REFERENCE_DESIGN_ID:
        for entry in v31_catalogue()["entries"]:
            if entry["set_id"] == "p2_divergent_exit" and entry["design_id"] == design_id:
                return entry
        raise KeyError("the P2 reference is missing from the v3.1 catalogue")
    for entry in v3_catalogue()["entries"]:
        if entry["design_id"] == design_id:
            return entry
    raise KeyError(f"{design_id} is not in the sweep-v3 catalogue")


def l1b_record(design_id: str) -> dict[str, Any] | None:
    """The L1b v1.1 material-aware confirmation record (15 HEMP-like designs) or None."""

    dataset = _cached_json(L1B_DATASET_PATH)
    for row in dataset["designs"]:
        if row["design_id"] == design_id:
            return row
    return None


def screening_v2_p_wall(design_id: str) -> dict[str, dict[str, Any]] | None:
    """Per-cell collisionless P(wall) of screening v2 (97 designs: 96 sweep_v2 + the P2 reference) or None."""

    dataset = _cached_json(SCREENING_V2_DATASET_PATH)
    for row in dataset["designs"]:
        record_path = row["catalogue"].get("record_path", "")
        if row["design_id"] == design_id or design_id in record_path:
            out: dict[str, dict[str, Any]] = {}
            for cell in row["cells"]:
                final = cell["final"]
                out[cell["cell_id"]] = {
                    "kind": cell["kind"], "z_start_m": cell.get("z_start_m"), "z_end_m": cell.get("z_end_m"),
                    "p_wall": final["p_wall"]["probability"], "lower": final["p_wall"]["lower"], "upper": final["p_wall"]["upper"],
                    "trials": final["trials"], "reflected": final["reflected"], "domain_escape": final["domain_escape"],
                }
            return out
    return None


def rho_conservative_from_entry(entry: Mapping[str, Any]) -> list[dict[str, float]]:
    """Koch rho_conservative per cusp = wall |B| at the cusp / max(adjacent axis |B_z| peaks) (sweep-v3 descriptor definition).

    Sweep-v3 entries carry the descriptor rows verbatim; the v3.1 P2 reference entry carries the cells' axis peaks
    and the cusps' wall fields, from which the same ratio is formed here (definition: l1a_geometry_sweep_v3/descriptors.py).
    """

    if entry.get("rho"):
        return [{"cusp_id": row["cusp_id"], "z_c_m": row["z_c_m"], "wall_b_t": row["wall_b_t"], "rho_conservative": row["rho_conservative"]} for row in entry["rho"]]
    cells = entry["cells"]
    rows = []
    for index, cusp in enumerate(entry["wall_cusps"]):
        up = abs(float(cells[index]["axis_bz_peak_t"]))
        down = abs(float(cells[index + 1]["axis_bz_peak_t"]))
        rows.append({"cusp_id": cusp["cusp_id"], "z_c_m": cusp["z_c_m"], "wall_b_t": cusp["wall_b_t"], "rho_conservative": float(cusp["wall_b_t"]) / max(up, down)})
    return rows


# --------------------------------------------------------------------------
# Geometry rebuild (identity-proven pipelines)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltDesign:
    design: SweepDesign
    geometry: AxisymmetricCFTGeometry
    source_strength_scale: float
    identity: dict[str, Any]
    derived: dict[str, Any]

    @property
    def design_id(self) -> str:
        return self.design.design_id


def build_design(design_id: str) -> BuiltDesign:
    """Rebuild the accepted geometry of a sweep design with its identity proof."""

    design = sweep_design(design_id)
    if design.source == "p2_divergent_exit":
        geometry = divergent_exit_stack()
        return BuiltDesign(design, geometry, 1.0,
                           {"geometry_sha256": geometry.canonical_sha256, "config_id": geometry.config_id,
                            "identity_basis": "cft_revival.geometry.generators.divergent_exit_stack (the fem_reference qualification design)"},
                           {"stage_count": len(geometry.stages), "represented_stage_pitch_m": geometry.stages[0].pitch_m})
    from experiments.l1a_geometry_sweep_v3 import designs as v3_designs
    from experiments.l1a_geometry_sweep_v3 import experiment as v3_experiment

    protocol = v3_experiment.protocol()
    spec = next(s for s in v3_designs.design_specs(protocol) if s.design_id == design_id)
    if design.source == "sobol_v3":
        case = v3_designs.sobol_case(spec, protocol)
        authorities = strict_json_file(v3_designs.EXPERIMENT / "design-authorities.json")
        authority = next(item for item in authorities["designs"] if item["set_id"] == v3_designs.SET_SOBOL and item["design_id"] == design_id)
        checks = {key: getattr(case, key) == authority[key] for key in ("geometry_sha256", "source_sha256", "config_sha256", "case_sha256")}
        basis = "sweep-v3 Sobol (seed, index) rebuild; hashes equal the sealed sweep-v3 design authorities"
        scale = float(v3_designs.design_values(case.design)["source_strength_scale"])
    elif design.source == "sweep_v2":
        from experiments.orbit_wall_loss_geometry_screening_v1 import designs as screening_designs

        binding = v3_designs.sweep_binding()
        case = screening_designs.rebuild_case(binding, design_id)
        recorded = binding.cases_by_id[design_id]
        checks = {key: getattr(case, key) == recorded[key] for key in ("geometry_sha256", "source_sha256", "config_sha256", "case_sha256") if key in recorded}
        basis = "sweep-v2 case rebuilt by the wall-loss geometry screening pipeline; hashes equal the sealed raw record"
        from experiments.l1a_geometry_sweep_v2 import experiment as sweep_v2

        scale = float(sweep_v2.design_values(case.design)["source_strength_scale"])
    else:
        raise ValueError(f"unknown design source {design.source}")
    if not all(checks.values()):
        raise ValueError(f"{design_id}: rebuilt case differs from its sealed authority: {checks}")
    identity = {"case_sha256": case.case_sha256, "geometry_sha256": case.geometry_sha256, "source_sha256": case.source_sha256,
                "config_sha256": case.config_sha256, "sampling_design_id": case.design.design_id, "identity_checks": checks, "identity_basis": basis}
    return BuiltDesign(design, case.geometry, scale, identity, dict(case.derived))


# --------------------------------------------------------------------------
# Catalogue geometry -> PIC ChannelGeometry / Grid2D (with a snapping record)
# --------------------------------------------------------------------------


def body_radii(geometry: AxisymmetricCFTGeometry) -> dict[str, float]:
    """Radii of the thruster body the PIC front face needs: dielectric outer, magnet inner (= conductor start), yoke outer."""

    magnets = [r for r in geometry.regions if r.role == "permanent_magnet"]
    return {
        "dielectric_outer_m": float(geometry.chamber.exit_outer_radius_m + geometry.chamber.dielectric_thickness_m),
        "magnet_inner_m": float(min(r.r_inner_start_m for r in magnets)),
        "yoke_outer_m": float(max(max(r.r_outer_start_m, r.r_outer_end_m) for r in geometry.regions)),
    }


def _snap(value: float, spacing: float) -> tuple[int, float, float]:
    cells = int(round(value / spacing))
    snapped = cells * spacing
    return cells, snapped, snapped - value


@dataclass(frozen=True)
class PicMapping:
    """PIC geometry and grid for one design and domain option, with every snap recorded."""

    design_id: str
    domain: str
    geometry: ChannelGeometry
    grid: Grid2D
    snaps: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"design_id": self.design_id, "domain": self.domain, "geometry": self.geometry.to_dict(), "grid": self.grid.to_dict(), "snaps": self.snaps}


def pic_geometry(built: BuiltDesign, domain: str, *, target_cell_m: float = TARGET_CELL_M, plume_length_m: float | None = None) -> PicMapping:
    """Map the catalogue geometry onto the PIC's straight-bore + cone (+ plume box) representation.

    * ``dr = r_w / n_r`` exactly (n_r = round(r_w / target)); ``dz = L_channel / n_z`` exactly;
    * exit radius, cone start, front-face dielectric radius (= magnet inner radius: the 0.25-0.8 mm clearance gap
      between the dielectric tube and the magnets is treated as dielectric front face, as the reference's 4.4 mm does)
      and the plume radius (= return-yoke outer radius, the thruster's own envelope, as the reference's 12 mm) are
      snapped to grid lines; ``snaps`` records value, snapped value, cells and the error of each;
    * the reference design snaps with zero error (2 / 3 / 4.4 / 12 mm and 18 / 24 mm on 50 um).
    """

    if domain not in DOMAIN_OPTIONS:
        raise ValueError(f"unknown domain option {domain!r}; known {DOMAIN_OPTIONS}")
    chamber = built.geometry.chamber
    r_w = float(chamber.outer_radius_m)
    length = float(chamber.length_m)
    n_r = max(2, int(round(r_w / target_cell_m)))
    dr = r_w / n_r
    n_z = max(2, int(round(length / target_cell_m)))
    dz = length / n_z
    snaps: dict[str, Any] = {"dr_m": dr, "dz_m": dz, "bore_cells": n_r, "channel_cells": n_z, "target_cell_m": target_cell_m}
    if chamber.exit_length_m > 0.0:
        n_exit, exit_snapped, exit_error = _snap(float(chamber.exit_outer_radius_m), dr)
        n_cone, cone_snapped, cone_error = _snap(length - float(chamber.exit_length_m), dz)
        if n_exit <= n_r:
            n_exit, exit_snapped, exit_error = n_r + 1, (n_r + 1) * dr, (n_r + 1) * dr - float(chamber.exit_outer_radius_m)
        snaps["exit_radius"] = {"value_m": float(chamber.exit_outer_radius_m), "snapped_m": exit_snapped, "cells": n_exit, "error_m": exit_error}
        snaps["cone_start"] = {"value_m": length - float(chamber.exit_length_m), "snapped_m": cone_snapped, "cells": n_cone, "error_m": cone_error}
    else:
        n_exit, exit_snapped = n_r, r_w
        n_cone, cone_snapped = n_z, length
        snaps["exit_radius"] = {"value_m": r_w, "snapped_m": r_w, "cells": n_r, "error_m": 0.0}
        snaps["cone_start"] = {"value_m": length, "snapped_m": length, "cells": n_z, "error_m": 0.0}
    plume_length = DEFAULT_PLUME_LENGTHS_M[domain] if plume_length_m is None else plume_length_m
    if plume_length is None:
        geometry = ChannelGeometry(r_w, 0.0, length, cone_snapped, exit_snapped)
        grid = Grid2D(geometry, n_exit, n_z)
        return PicMapping(built.design_id, domain, geometry, grid, snaps)
    radii = body_radii(built.geometry)
    n_body, body_snapped, body_error = _snap(radii["magnet_inner_m"], dr)
    n_plume, plume_snapped, plume_error = _snap(radii["yoke_outer_m"], dr)
    n_body = max(n_body, n_exit)                      # the front-face dielectric must reach the exit lip
    n_plume = max(n_plume, n_body + 1)
    body_snapped, plume_snapped = n_body * dr, n_plume * dr
    n_plume_z, plume_len_snapped, plume_len_error = _snap(float(plume_length), dz)
    snaps["body_dielectric_radius"] = {"value_m": radii["magnet_inner_m"], "snapped_m": body_snapped, "cells": n_body, "error_m": body_snapped - radii["magnet_inner_m"]}
    snaps["plume_radius"] = {"value_m": radii["yoke_outer_m"], "snapped_m": plume_snapped, "cells": n_plume, "error_m": plume_snapped - radii["yoke_outer_m"]}
    snaps["plume_length"] = {"value_m": float(plume_length), "snapped_m": plume_len_snapped, "cells": n_plume_z, "error_m": plume_len_error}
    geometry = ChannelGeometry(r_w, 0.0, length, cone_snapped, exit_snapped, plume_radius_m=plume_snapped, plume_length_m=plume_len_snapped,
                               body_dielectric_radius_m=body_snapped)
    grid = Grid2D(geometry, n_plume, n_z + n_plume_z)
    return PicMapping(built.design_id, domain, geometry, grid, snaps)


def exit_area_m2(mapping: PicMapping) -> float:
    return math.pi * float(mapping.geometry.exit_radius_m) ** 2


def channel_volume_m3(mapping: PicMapping) -> float:
    """Volume of the PIC channel (bore + linear cone), the volume the macro-particle count scales with."""

    g = mapping.geometry
    straight = math.pi * g.bore_radius_m**2 * (g.cone_start_z_m - g.z_min_m)
    h = g.z_max_m - g.cone_start_z_m
    cone = math.pi * h * (g.bore_radius_m**2 + g.bore_radius_m * g.exit_radius_m + g.exit_radius_m**2) / 3.0 if h > 0.0 else 0.0
    return straight + cone


def design_summary(design_id: str) -> dict[str, Any]:
    """The design-table row: catalogue numbers (cusps, cells, rho, wall |B|), L1b iron numbers and screening P(wall) where they exist."""

    built = build_design(design_id)
    entry = catalogue_entry(design_id)
    chamber = built.geometry.chamber
    rho_rows = rho_conservative_from_entry(entry)
    l1b = l1b_record(design_id)
    summary: dict[str, Any] = {
        "design": built.design.to_dict(),
        "catalogue_label": entry["label"],
        "stage_count": int(len(built.geometry.stages)),
        "wall_cusp_count": int(entry["wall_cusp_count"]),
        "cell_count": int(entry["cell_count"]),
        "wall_radius_m": float(chamber.outer_radius_m),
        "chamber_length_m": float(chamber.length_m),
        "stage_pitch_m": float(built.geometry.stages[0].pitch_m),
        "wall_radius_over_pitch": float(chamber.outer_radius_m) / float(built.geometry.stages[0].pitch_m),
        "exit_length_m": float(chamber.exit_length_m),
        "exit_outer_radius_m": float(chamber.exit_outer_radius_m),
        "dielectric_thickness_m": float(chamber.dielectric_thickness_m),
        "body_radii_m": body_radii(built.geometry),
        "source_strength_scale": built.source_strength_scale,
        "cusps_l1a_or_p2": [{"cusp_id": c["cusp_id"], "z_c_m": c["z_c_m"], "wall_b_t": c["wall_b_t"]} for c in entry["wall_cusps"]],
        "cells": [{"cell_id": c["cell_id"], "kind": c["kind"], "z_start_m": c["z_start_m"], "z_end_m": c["z_end_m"], "axis_bz_peak_t": c["axis_bz_peak_t"]} for c in entry["cells"]],
        "rho_conservative": rho_rows,
        "min_rho_conservative": min(row["rho_conservative"] for row in rho_rows),
        "rho_field": "P2 qualified level-1 (material-aware)" if design_id == REFERENCE_DESIGN_ID else "L1a linear-vacuum equivalent-current",
        "hemp_like_all_cusps": bool(entry.get("hemp_like_all_cusps", False)),
        "identity": built.identity,
    }
    if l1b is not None:
        cmp = l1b["comparison"]
        summary["l1b_v1_1"] = {
            "p2_min_rho_conservative": cmp["p2_min_rho_conservative"], "p2_hemp_like_all_cusps": cmp["p2_hemp_like_all_cusps"],
            "p2_wall_cusp_count": cmp["p2_wall_cusp_count"], "max_cusp_shift_m": cmp["max_cusp_shift_m"],
            "cusp_position_tolerance_m": cmp["cusp_position_tolerance_m"], "peak_wall_b_ratio_p2_over_l1a": cmp["peak_wall_b_ratio_p2_over_l1a"],
            "p2_cusps": [{"cusp_id": m["p2_cusp_id"], "z_c_m": m["p2_z_c_m"], "wall_b_t": m["p2_wall_b_t"], "rho_conservative": m["p2_rho_conservative"]} for m in cmp["matched_cusps"]],
            "record_path": l1b["record_path"],
        }
    p_wall = screening_v2_p_wall(design_id)
    if p_wall is not None:
        summary["screening_v2_p_wall"] = p_wall
    return summary


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


__all__ = [
    "BuiltDesign", "DEFAULT_PLUME_LENGTHS_M", "DOMAIN_OPTIONS", "FIELDS_DIR", "PicMapping", "REFERENCE_DESIGN_ID", "SWEEP_DESIGNS",
    "SweepDesign", "TARGET_CELL_M", "body_radii", "build_design", "canonical_sha256", "catalogue_entry", "channel_volume_m3",
    "design_ids", "design_summary", "exit_area_m2", "l1b_record", "pic_geometry", "rho_conservative_from_entry", "screening_v2_p_wall",
    "sweep_design", "v3_catalogue", "v31_catalogue",
]
