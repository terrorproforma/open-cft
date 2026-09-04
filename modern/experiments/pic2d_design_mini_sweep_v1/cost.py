"""Cost model and GPU schedule of the design mini-sweep (projection, anchored on measured pic2d runs).

Model (spec/pic2d/pic2d-model-v2.1.json ``cost_table_v2_1``): per step

    ms/step = fixed(grid) + PARTICLE_SLOPE_MS_PER_M * N_particles[M]
    fixed   = 2 (nr+1) row-block launches x 5 us            (sequential block-Thomas launches)
            + (nr+1) x 2 (nz+1)^2 x 8 B / 1.6 TB/s          (inverse-block reads per solve)
            + 0.15 ms x nodes / 173,761                     (node kernels, scaled from the v2.0 grid)

Anchors (RTX 5090, CUDA-graph step): channel 61 x 481 nodes, 2.0 M particles -> 1.98 ms/step (steady-state v2 base,
5.12 M steps in 10,141 s); 2.7 M -> 2.44 (W x 0.7); plume 241 x 721, 4.45 M -> 7.08 (attempt 8).  The model
reproduces the plume anchor by construction and over-predicts the channel anchors by 13-14 % (kept: conservative).
The particle count of a design is projected from the reference plateau at EQUAL mean electron density and equal
macro weight (N proportional to the channel volume); for the plume options the channel count carries the 1.75x the
v2.0 model showed over the v1.3 channel-only plateau (attempt 8: 3.5 M of 4.45 M in the channel) plus the plume
particles measured in attempt 7 (21 % of the ions in the 12 mm box; +0.16 M per further 12 mm for a constant-current
beam), scaled by the design's channel volume.  Every number is a projection until the design's own ignition check
measures it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .designs import DOMAIN_OPTIONS, BuiltDesign, PicMapping, channel_volume_m3, pic_geometry

PARTICLE_SLOPE_MS_PER_M = 0.733
LAUNCH_US = 5.0
DEVICE_BANDWIDTH_B_PER_S = 1.6e12
NODE_KERNEL_MS_AT_V20 = 0.15
V20_NODES = 173_761
DT_S = 1.5e-12
REFERENCE_CHANNEL_RESIDENCE_S = 2.4e-6          # measured N_i / L of the v2 fine cases (24 mm channel)
REFERENCE_CHANNEL_LENGTH_M = 0.024
PLUME_ION_SPEED_M_PER_S = 1.7e4                  # ~200 eV Xe+ (v2.1 transit rule)
REFERENCE_CHANNEL_PARTICLES_M = 2.0              # electrons + ions at the steady-state v2 plateau (W = 6e4, 3.44 mA)
REFERENCE_CHANNEL_VOLUME_M3 = 3.456e-7           # bore + cone of the reference (designs.channel_volume_m3)
REFERENCE_PLUME_12MM_PARTICLES_M = 1.0           # attempt 7: 21 % of 2.47 M ions in the 12 mm box, plus their electrons
PLUME_PARTICLES_PER_12MM_M = 0.16                # +6.5k macro-ions/mm + electrons for a constant-current beam (spec v2.1)
PLUME_MODEL_CHANNEL_DENSITY_FACTOR = 1.75        # the v2.0 plume model's channel holds 1.75x the v1.3 channel-only plateau count (attempt 8: 3.5 M of 4.45 M in the channel at I_d 6 mA vs 3.4 mA)
FACTORISATION_REFERENCE_S = 300.0                # 5 min at 241 rows x 721 nodes (measured)
FACTORISATION_REFERENCE_ROWS_M3 = 241 * 721**3
FIELD_MAP_S_PER_NODE = 1.5e-4                    # direct P2 node evaluation (~30 s for 231k nodes)
DEVICE_GB_BASE = 1.5
DEVICE_GB_PER_M_PARTICLES = 1.3

ANCHORS = (
    {"case": "steady-state v2 base (channel 61 x 481)", "nodes": (61, 481), "particles_m": 2.0, "ms_per_step_measured": 1.98},
    {"case": "steady-state v2 W x 0.7 (channel 61 x 481)", "nodes": (61, 481), "particles_m": 2.7, "ms_per_step_measured": 2.44},
    {"case": "plume attempt 8 (241 x 721)", "nodes": (241, 721), "particles_m": 4.45, "ms_per_step_measured": 7.08},
)


def fixed_ms_per_step(node_shape: tuple[int, int]) -> dict[str, float]:
    nr1, nz1 = int(node_shape[0]), int(node_shape[1])
    launches_ms = 2.0 * nr1 * LAUNCH_US * 1e-3
    block_bytes = nr1 * 2 * nz1 * nz1 * 8
    reads_ms = block_bytes / DEVICE_BANDWIDTH_B_PER_S * 1e3
    kernels_ms = NODE_KERNEL_MS_AT_V20 * (nr1 * nz1) / V20_NODES
    return {"launches_ms": launches_ms, "inverse_block_reads_ms": reads_ms, "node_kernels_ms": kernels_ms, "fixed_ms": launches_ms + reads_ms + kernels_ms,
            "inverse_blocks_gb": nr1 * nz1 * nz1 * 8 / 1e9, "sequential_launches": 2 * nr1}


def ms_per_step(node_shape: tuple[int, int], particles_m: float) -> float:
    return fixed_ms_per_step(node_shape)["fixed_ms"] + PARTICLE_SLOPE_MS_PER_M * particles_m


def anchor_residuals() -> list[dict[str, Any]]:
    rows = []
    for anchor in ANCHORS:
        projected = ms_per_step(anchor["nodes"], anchor["particles_m"])
        rows.append({**anchor, "ms_per_step_projected": projected, "relative_error": projected / anchor["ms_per_step_measured"] - 1.0})
    return rows


def projected_particles_m(mapping: PicMapping) -> dict[str, float]:
    volume_ratio = channel_volume_m3(mapping) / REFERENCE_CHANNEL_VOLUME_M3
    channel = REFERENCE_CHANNEL_PARTICLES_M * volume_ratio
    plume = 0.0
    geometry = mapping.geometry
    if geometry.has_plume:
        channel *= PLUME_MODEL_CHANNEL_DENSITY_FACTOR
        length_mm = float(geometry.plume_length_m) * 1e3
        plume = (REFERENCE_PLUME_12MM_PARTICLES_M + PLUME_PARTICLES_PER_12MM_M * max(0.0, (length_mm - 12.0) / 12.0)) * volume_ratio
    return {"channel_m": channel, "plume_m": plume, "total_m": channel + plume, "channel_volume_ratio": volume_ratio}


def transit_time_s(mapping: PicMapping) -> float:
    geometry = mapping.geometry
    residence = REFERENCE_CHANNEL_RESIDENCE_S * geometry.channel_length_m / REFERENCE_CHANNEL_LENGTH_M
    plume = 0.0 if not geometry.has_plume else float(geometry.plume_length_m) / PLUME_ION_SPEED_M_PER_S
    return residence + plume


def design_cost(mapping: PicMapping, *, transits: float = 3.0, dt_s: float = DT_S) -> dict[str, Any]:
    nodes = mapping.grid.node_shape
    fixed = fixed_ms_per_step(nodes)
    particles = projected_particles_m(mapping)
    ms = fixed["fixed_ms"] + PARTICLE_SLOPE_MS_PER_M * particles["total_m"]
    transit = transit_time_s(mapping)
    steps = transits * transit / dt_s
    factorisation_s = FACTORISATION_REFERENCE_S * (nodes[0] * nodes[1] ** 3) / FACTORISATION_REFERENCE_ROWS_M3
    field_map_s = FIELD_MAP_S_PER_NODE * nodes[0] * nodes[1]
    stepping_h = steps * ms / 3.6e6
    return {
        "design_id": mapping.design_id, "domain": mapping.domain, "nodes": [nodes[0], nodes[1]], "cells": [mapping.grid.radial_cells, mapping.grid.axial_cells],
        "dr_um": mapping.grid.dr_m * 1e6, "dz_um": mapping.grid.dz_m * 1e6, "dt_s": dt_s,
        **fixed, "particles_projected_m": particles, "ms_per_step": ms,
        "transit_s": transit, "transits": transits, "steps_to_transits": steps,
        "factorisation_s": factorisation_s, "field_map_s": field_map_s,
        "stepping_hours": stepping_h, "wall_hours": stepping_h + (factorisation_s + field_map_s) / 3600.0,
        "device_gb_projected": DEVICE_GB_BASE + DEVICE_GB_PER_M_PARTICLES * particles["total_m"] + fixed["inverse_blocks_gb"],
    }


REFINED_CHANNEL_CELL_M = 3.33e-5     # attempt-8 verdict (ac248e05): Delta <= 32.4 um keeps Delta/lambda_D <= 0.8 pi at the recorded peak; dt <= 1.48 ps
REFINED_CHANNEL_DT_S = 1.4e-12
REFINED_CHANNEL_KEY = "channel-33um-1.4ps"


def cost_table(built_designs: list[BuiltDesign], *, transits: float = 3.0) -> dict[str, list[dict[str, Any]]]:
    """Rows per domain option at the 50 um / 1.5 ps grid, plus the grid-refinement channel variant of the attempt-8 verdict."""

    table = {domain: [design_cost(pic_geometry(built, domain), transits=transits) for built in built_designs] for domain in DOMAIN_OPTIONS}
    table[REFINED_CHANNEL_KEY] = [
        {**design_cost(pic_geometry(built, "channel", target_cell_m=REFINED_CHANNEL_CELL_M), transits=transits, dt_s=REFINED_CHANNEL_DT_S), "domain": REFINED_CHANNEL_KEY}
        for built in built_designs
    ]
    return table


@dataclass(frozen=True)
class ScheduleItem:
    design_id: str
    domain: str
    kind: str
    wall_hours: float


def serial_schedule(table: dict[str, list[dict[str, Any]]], *, option: str, replicate_design_ids: tuple[str, ...] = (), extra: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    """GPU serial schedule: every design once under ``option`` (+ replicates + extra (design, domain) runs); no concurrency (one GPU, one factorisation at a time)."""

    items = [ScheduleItem(row["design_id"], option, "base", row["wall_hours"]) for row in table[option]]
    by_id = {row["design_id"]: row for row in table[option]}
    for design_id in replicate_design_ids:
        items.append(ScheduleItem(design_id, option, "seed-replicate", by_id[design_id]["wall_hours"]))
    for design_id, domain in extra:
        row = next(r for r in table[domain] if r["design_id"] == design_id)
        items.append(ScheduleItem(design_id, domain, "extra", row["wall_hours"]))
    total = sum(item.wall_hours for item in items)
    return {"option": option, "items": [item.__dict__ for item in items], "total_hours": total, "total_days_at_24h": total / 24.0,
            "note": "serial on one RTX 5090; +~10 % if the particle count keeps growing past the projected plateau (attempt 8 added 0.6 M ions/us); "
                    "budgets should be set at 1.25x the projected wall so the plateau rule can be evaluated past 3 transits"}


__all__ = ["ANCHORS", "PARTICLE_SLOPE_MS_PER_M", "REFINED_CHANNEL_CELL_M", "REFINED_CHANNEL_DT_S", "REFINED_CHANNEL_KEY", "anchor_residuals", "cost_table",
           "design_cost", "fixed_ms_per_step", "ms_per_step", "projected_particles_m", "serial_schedule", "transit_time_s"]
