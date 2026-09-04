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

# Lambda H100 80GB SXM under CUDA MPS with FOUR concurrent PIC processes (modern/tools/cloud/bench.sh, bench-mps, 2026-09-04):
# the steady-state v4 configuration (91 x 721 nodes, ~4.5 M macro-particles at the re-seeded production load) stepped at
# 3.37 ms/step alone, 8.71 ms/step per process with N = 4 (aggregate 1.54x).  Every per-process H100/MPS projection below is
# the 5090 cost model scaled by 8.71 / model(91 x 721, 4.5 M): the model's grid and particle dependence, the box's measured level.
H100_MPS4_ANCHOR = {"gpu": "NVIDIA H100 80GB HBM3", "mps_slots": 4, "nodes": (91, 721), "particles_m": 4.5, "ms_per_step_per_process": 8.71,
                    "ms_per_step_solo": 3.37, "source": "modern/tools/cloud/bench.sh channel-33um production load, bench-mps N=4 (2026-09-04 14:35 AEST)"}
PLATFORMS = ("rtx5090", "h100-mps4")
MACRO_WEIGHT_REFERENCE = 6.0e4                   # the macro weight of every 50 um anchor (particle counts scale with 6e4 / W)


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


def h100_mps4_ms_per_step(node_shape: tuple[int, int], particles_m: float) -> float:
    """Per-process ms/step on the H100 with four MPS slots busy: the 5090 model scaled to the measured N = 4 anchor."""

    anchor = H100_MPS4_ANCHOR
    return ms_per_step(node_shape, particles_m) * anchor["ms_per_step_per_process"] / ms_per_step(anchor["nodes"], anchor["particles_m"])


def platform_ms_per_step(node_shape: tuple[int, int], particles_m: float, platform: str) -> float:
    if platform == "rtx5090":
        return ms_per_step(node_shape, particles_m)
    if platform == "h100-mps4":
        return h100_mps4_ms_per_step(node_shape, particles_m)
    raise ValueError(f"unknown platform {platform!r}; known {PLATFORMS}")


def anchor_residuals() -> list[dict[str, Any]]:
    rows = []
    for anchor in ANCHORS:
        projected = ms_per_step(anchor["nodes"], anchor["particles_m"])
        rows.append({**anchor, "ms_per_step_projected": projected, "relative_error": projected / anchor["ms_per_step_measured"] - 1.0})
    return rows


def projected_particles_m(mapping: PicMapping, *, macro_weight: float = MACRO_WEIGHT_REFERENCE) -> dict[str, float]:
    """Macro-particle count at the reference plateau's mean density, scaled with the channel volume and with 6e4 / W."""

    volume_ratio = channel_volume_m3(mapping) / REFERENCE_CHANNEL_VOLUME_M3
    weight_ratio = MACRO_WEIGHT_REFERENCE / float(macro_weight)
    channel = REFERENCE_CHANNEL_PARTICLES_M * volume_ratio * weight_ratio
    plume = 0.0
    geometry = mapping.geometry
    if geometry.has_plume:
        channel *= PLUME_MODEL_CHANNEL_DENSITY_FACTOR
        length_mm = float(geometry.plume_length_m) * 1e3
        plume = (REFERENCE_PLUME_12MM_PARTICLES_M + PLUME_PARTICLES_PER_12MM_M * max(0.0, (length_mm - 12.0) / 12.0)) * volume_ratio * weight_ratio
    return {"channel_m": channel, "plume_m": plume, "total_m": channel + plume, "channel_volume_ratio": volume_ratio, "macro_weight": float(macro_weight),
            "weight_ratio_6e4_over_w": weight_ratio}


def transit_time_s(mapping: PicMapping) -> float:
    geometry = mapping.geometry
    residence = REFERENCE_CHANNEL_RESIDENCE_S * geometry.channel_length_m / REFERENCE_CHANNEL_LENGTH_M
    plume = 0.0 if not geometry.has_plume else float(geometry.plume_length_m) / PLUME_ION_SPEED_M_PER_S
    return residence + plume


def design_cost(mapping: PicMapping, *, transits: float = 3.0, dt_s: float = DT_S, macro_weight: float = MACRO_WEIGHT_REFERENCE,
                platform: str = "rtx5090") -> dict[str, Any]:
    """Projected cost of one design run: ms/step (5090 model, and the H100/MPS-4 per-process level), steps and hours to ``transits``.

    ``macro_weight`` scales the projected particle count (6e4 / W); ``platform`` selects which ms/step the hours are formed with
    (``rtx5090`` = the anchored model; ``h100-mps4`` = one of four MPS slots on the H100).  Both ms/step values are recorded.
    """

    nodes = mapping.grid.node_shape
    fixed = fixed_ms_per_step(nodes)
    particles = projected_particles_m(mapping, macro_weight=macro_weight)
    ms_5090 = fixed["fixed_ms"] + PARTICLE_SLOPE_MS_PER_M * particles["total_m"]
    ms_h100 = h100_mps4_ms_per_step(nodes, particles["total_m"])
    ms = platform_ms_per_step(nodes, particles["total_m"], platform)
    transit = transit_time_s(mapping)
    steps = transits * transit / dt_s
    factorisation_s = FACTORISATION_REFERENCE_S * (nodes[0] * nodes[1] ** 3) / FACTORISATION_REFERENCE_ROWS_M3
    field_map_s = FIELD_MAP_S_PER_NODE * nodes[0] * nodes[1]
    stepping_h = steps * ms / 3.6e6
    return {
        "design_id": mapping.design_id, "domain": mapping.domain, "nodes": [nodes[0], nodes[1]], "cells": [mapping.grid.radial_cells, mapping.grid.axial_cells],
        "dr_um": mapping.grid.dr_m * 1e6, "dz_um": mapping.grid.dz_m * 1e6, "dt_s": dt_s, "macro_weight": float(macro_weight), "platform": platform,
        **fixed, "particles_projected_m": particles, "ms_per_step": ms, "ms_per_step_rtx5090_model": ms_5090, "ms_per_step_h100_mps4_per_process": ms_h100,
        "transit_s": transit, "transits": transits, "steps_to_transits": steps,
        "factorisation_s": factorisation_s, "field_map_s": field_map_s,
        "stepping_hours": stepping_h, "wall_hours": stepping_h + (factorisation_s + field_map_s) / 3600.0,
        "device_gb_projected": DEVICE_GB_BASE + DEVICE_GB_PER_M_PARTICLES * particles["total_m"] + fixed["inverse_blocks_gb"],
    }


# The refined channel grid = the preregistered steady-state v4 grid (392129e5): target 24 mm / 720 = 33.333 um so that the reference
# design reproduces v4's 90 x 720 cells EXACTLY (dr = 2 mm / 60, dz = 24 mm / 720; the draft's 3.33e-5 gave 90 x 721).  Attempt-8
# verdict (ac248e05): Delta <= 32.4 um keeps Delta/lambda_D <= 0.8 pi at the recorded peak; dt <= 1.48 ps.
REFINED_CHANNEL_CELL_M = 0.024 / 720.0
REFINED_CHANNEL_DT_S = 1.4e-12
REFINED_CHANNEL_KEY = "channel-33um-1.4ps"


def parity_macro_weight(mapping: PicMapping, *, reference_weight: float = MACRO_WEIGHT_REFERENCE, reference_cell_m: float = 5.0e-5) -> float:
    """W with the SAME macro-particles per cell as the 50 um / W 6e4 runs at equal density: W = 6e4 x (dr dz) / (50 um)^2, to 0.1.

    The steady-state v4 rule (W = 6e4 / 2.25 = 26 666.7 on 33.33 um): the reference design reproduces v4's value exactly; the other
    designs' snapped dr / dz give W within +-1 % of it.
    """

    return round(reference_weight * (mapping.grid.dr_m * mapping.grid.dz_m) / (reference_cell_m * reference_cell_m), 1)


def cost_table(built_designs: list[BuiltDesign], *, transits: float = 3.0) -> dict[str, list[dict[str, Any]]]:
    """Rows per domain option at the 50 um / 1.5 ps grid (W 6e4, 5090 model), plus the refined channel variant at the v4 grid
    with particles-per-cell parity (W = 6e4 / 2.25) costed for one of four H100 MPS slots (the preregistered option)."""

    table = {domain: [design_cost(pic_geometry(built, domain), transits=transits) for built in built_designs] for domain in DOMAIN_OPTIONS}
    refined = []
    for built in built_designs:
        mapping = pic_geometry(built, "channel", target_cell_m=REFINED_CHANNEL_CELL_M)
        refined.append({**design_cost(mapping, transits=transits, dt_s=REFINED_CHANNEL_DT_S, macro_weight=parity_macro_weight(mapping), platform="h100-mps4"),
                        "domain": REFINED_CHANNEL_KEY})
    table[REFINED_CHANNEL_KEY] = refined
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


__all__ = ["ANCHORS", "H100_MPS4_ANCHOR", "MACRO_WEIGHT_REFERENCE", "PARTICLE_SLOPE_MS_PER_M", "PLATFORMS", "REFINED_CHANNEL_CELL_M", "REFINED_CHANNEL_DT_S",
           "REFINED_CHANNEL_KEY", "anchor_residuals", "cost_table", "design_cost", "fixed_ms_per_step", "h100_mps4_ms_per_step", "ms_per_step",
           "parity_macro_weight", "platform_ms_per_step", "projected_particles_m", "serial_schedule", "transit_time_s"]
