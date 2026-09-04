"""Closure targets of the design mini-sweep: what each PIC run measures and where it enters plasma-network v2.

The reduced model is ``cft_revival.plasma_v2`` (sheath-closed four-cell power balance: rows R00-R26 of the Kornfeld
ledger, sheath rows R28-R30, anode row R31, potential closure R32-R34, cusp-loss rows R35-R37).  Its design-dependent
inputs are (``plasma_v2.models``):

* ``SheathClosureInputs.declared_cusp_probabilities`` p_1..p_3 and ``anode_cusp_probability`` p_4 (CL-1) - the
  per-cusp-TRANSIT electron loss probabilities of Kornfeld's chain je_k = je_{k-1} (1 - p_k) + I_k;
* ``CuspSheathSpec.area_ratio`` rho_k = A_e,k / A_i,k (sheath rows: Delta phi_s,k = T_k ln(K0 rho_k)),
  ``access_fraction`` A_k (CL-3), ``electron_density_per_m3`` n_k and ``wall_field_t`` B_w,k (CL-4),
  ``regime`` / ``emission_yield`` (floating / emitting / space-charge-limited sheath);
* ``SheathClosureInputs.leak_width_prefactor`` (CL-4: leak width = prefactor x hybrid gyroradius) and ``wall_radius_m``;
* ``PotentialClosure`` interior steps phi_3 - phi_2, phi_4 - phi_3, anode fall, cathode coupling (CL-3-potentials);
* the state itself (``PlasmaState``): phi_k, T_k, I_k (ionisation source current per cell), je_k, ji_k, cusp ion
  currents - the quantities a calibrated closure must REPRODUCE, not fit.

``plasma_v2.pic_context`` already extracts these for the reference design at its three cusp planes (17.95 / 12.0 /
6.0 mm) with the declared mapping "Kornfeld cell 1 = plume/cone cell, cells 2-4 = channel cells, p_4 = anode-most
cusp".  ``extract_targets`` below is that extraction generalised to any sweep design (cusp planes and cell bounds
from the design's own material-aware topology, wall radius from the PIC geometry) and extended with the ionisation
share per cell, the per-cell ion wall-loss fraction, the Kornfeld chain probabilities p_k from the window currents,
the diffuse (non-cusp) wall loss and the plume-side quantities where the run has a plume box.

Nothing here is a fit: the targets are the estimands the preregistration will freeze; their acceptance rule is the
plateau rule of the accepted steady-state runs (README, section "Plateau / acceptance").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .designs import PicMapping

ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class ClosureTarget:
    name: str
    unit: str
    per: str                 # cusp | cell | design | plume
    pic_observable: str
    plasma_network_v2_parameter: str
    role: str                # calibration | reproduction | disclosure

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CLOSURE_TARGETS: tuple[ClosureTarget, ...] = (
    ClosureTarget("cusp_transit_loss_probability", "-", "cusp",
                  "L_k / je_arriving,k: window-averaged electron wall current in the cusp window (|z - z_c| <= w) over the electron current arriving at the cusp plane from the exit side, from the Kornfeld chain je_k = je_{k-1} - L_k + I_k seeded with the injected/cathode current that enters the channel",
                  "SheathClosureInputs.declared_cusp_probabilities[k] (CL-1) / anode_cusp_probability (anode-most cusp)", "calibration"),
    ClosureTarget("cusp_electron_wall_current", "A", "cusp", "e x sum over the cusp window of wall_electron_flux x wall area (maps.npz wall_electron_flux_per_m2_s)",
                  "L_k in rows R23-R26 / R35-R37 (the lost electron current)", "calibration"),
    ClosureTarget("cusp_ion_wall_current", "A", "cusp", "e x sum over the cusp window of wall_ion_flux x wall area",
                  "PlasmaState.cusp_ion_current_a[k] (ambipolar balance of the sheath rows)", "reproduction"),
    ClosureTarget("effective_cusp_loss_coefficient", "1/s", "cusp", "L_k / (e N_e,cell) with N_e,cell the window-averaged electron inventory of the cell upstream of the cusp: the electron loss frequency the reduced model implies",
                  "derived: p_k x (je_arriving / e N_e) - the cross-field transport the four-cell model has no explicit slot for (disclosure: it enters only through p_k)", "disclosure"),
    ClosureTarget("cusp_leak_width", "m", "cusp", "FWHM of the wall electron flux profile about z_c",
                  "SheathClosureInputs.leak_width_prefactor (CL-4: width / hybrid gyroradius at the cusp wall field)", "calibration"),
    ClosureTarget("cusp_sheath_drop", "V", "cusp", "phi(axis, z_c) - phi(wall, z_c) and the near-wall drop phi(wall - 3 dr) - phi(wall) (pic_context proxy)",
                  "SheathClosureState.sheath_drop_v[k]; with T_k gives c_s,k = Delta phi / T_k -> CuspSheathSpec.area_ratio via ln(K0 rho_k) and the regime", "calibration"),
    ClosureTarget("cusp_near_wall_temperature", "eV", "cusp", "density-weighted T_e in the last 0.5 mm before the wall over the cusp window", "PlasmaState.electron_temperature_ev[k] (sheath coefficient argument)", "reproduction"),
    ClosureTarget("cusp_wall_field", "T", "cusp", "|B| at (r_w, z_c) of the bound P2 map (scaled)", "CuspSheathSpec.wall_field_t (CL-4)", "input"),
    ClosureTarget("cusp_electron_density", "1/m^3", "cusp", "window-averaged n_e in the near-wall band of the cusp window", "CuspSheathSpec.electron_density_per_m3 (CL-4)", "calibration"),
    ClosureTarget("cell_ionisation_share", "-", "cell", "S_k / S: volume integral of ionization_rate_per_m3_s over the cell (2 pi r dr dz) over the domain total (the renderer's cusp-plane 'flames')",
                  "PlasmaState.ionization_source_current_a[k] / sum (e S_k = I_k)", "reproduction"),
    ClosureTarget("cell_ion_wall_loss_fraction", "-", "cell", "e-weighted wall ion current of the cell over the ionisation current of the cell (bounded above by the collisionless screening P(wall) = 1)",
                  "closes ji_k in the ion current rows R06-R11; the quantity the saturated screening-v2 label could not resolve", "calibration"),
    ClosureTarget("cell_potential", "V", "cell", "n_e-weighted phi over the cell", "PlasmaState.plasma_potential_v[k]; steps phi_{k+1} - phi_k -> PotentialClosure.interior_step_3_v / _4_v (CL-3-potentials)", "calibration"),
    ClosureTarget("cell_temperature", "eV", "cell", "n_e-weighted T_e over the cell", "PlasmaState.electron_temperature_ev[k]", "reproduction"),
    ClosureTarget("discharge_current", "A", "design", "window anode electron + ion current (summary.window_currents_a.discharge_a)", "SheathClosureInputs.anode_current_a (input) / row R11 closure", "reproduction"),
    ClosureTarget("ionisation_rate", "1/s", "design", "window S (neutral_inventory ionization rate)", "sum_k I_k / e", "reproduction"),
    ClosureTarget("utilisation", "-", "design", "gross e S / (e Q_in) and net (S - recycled) / Q_in (v1.4 ledger)", "mass utilisation of the reduced model's beam current ji_1 / (e Q_in)", "reproduction"),
    ClosureTarget("anode_ion_fraction", "-", "design", "window anode ion current / discharge current", "anode row R31 (anode sheath: identifies phi_4 - Ua from the anode ion fraction)", "calibration"),
    ClosureTarget("beam_current", "A", "plume", "far-field ion current (summary.plume / window exit_ion_beam_a)", "ji_1 (beam ion current leaving cell 1)", "reproduction"),
    ClosureTarget("divergence_half_angles", "deg", "plume", "50 / 90 / 95 % current half-angles (plume_ion_current_per_sr_a histogram)", "none in the four-cell model (disclosure; beam efficiency input of any thrust estimate)", "disclosure"),
    ClosureTarget("thrust", "N", "plume", "thrust_total_n of the momentum ledger (development number until the plateau rule holds)", "none (downstream performance estimate; the model's beam power P_beam = Ua x ji_1 is the comparable)", "disclosure"),
)


def closure_target_table() -> list[dict[str, Any]]:
    return [target.to_dict() for target in CLOSURE_TARGETS]


# --------------------------------------------------------------------------
# Kornfeld mapping of a design's cusps / cells
# --------------------------------------------------------------------------


def kornfeld_mapping(cusps_z_m: Sequence[float], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Declared map of a design's wall cusps and cells onto the four-cell model's slots.

    Cusps are ordered from the EXIT toward the anode (Kornfeld numbers his cusps from the plume side).  With three
    wall cusps (the reference and the three primary sweep designs) model cusp 1 (plume-side) has no magnetic
    counterpart, model cusps 2 and 3 are the two exit-side wall cusps and p_4 is the anode-most cusp - the
    ``plasma_v2.pic_context`` mapping.  With four wall cusps every slot p_1..p_4 is a wall cusp.  Cells are the
    catalogue cells (anode_partial, interior..., exit_partial); model cell 1 = exit_partial (+ cone + plume), model
    cell 4 = anode_partial.  Designs with more than four wall cusps are not mappable and raise.
    """

    ordered = sorted(float(z) for z in cusps_z_m)
    exit_to_anode = ordered[::-1]
    if len(exit_to_anode) == 3:
        slots = {"cusp_1": None, "cusp_2": exit_to_anode[0], "cusp_3": exit_to_anode[1], "cusp_4_anode": exit_to_anode[2]}
    elif len(exit_to_anode) == 4:
        slots = {"cusp_1": exit_to_anode[0], "cusp_2": exit_to_anode[1], "cusp_3": exit_to_anode[2], "cusp_4_anode": exit_to_anode[3]}
    else:
        raise ValueError(f"the four-cell model maps 3 or 4 wall cusps, not {len(exit_to_anode)}")
    cells_sorted = sorted(cells, key=lambda c: float(c["z_start_m"]))
    return {
        "cusp_slots_z_m": slots,
        "cells_anode_to_exit": [{"cell_id": c["cell_id"], "kind": c["kind"], "z_start_m": float(c["z_start_m"]), "z_end_m": float(c["z_end_m"])} for c in cells_sorted],
        "model_cell_of_catalogue_cell": {c["cell_id"]: f"model cell {len(cells_sorted) - index}" for index, c in enumerate(cells_sorted)},
        "note": "model cell 1 = exit-side partial cell + cone + plume; model cell N = anode-side partial cell; cusps numbered from the exit (Kornfeld)",
    }


# --------------------------------------------------------------------------
# Extraction from window-averaged maps (generalised plasma_v2.pic_context)
# --------------------------------------------------------------------------


def _wall_radius(geometry, z: float) -> float:
    return float(geometry.wall_radius_m(z)) if z < geometry.z_max_m else float(geometry.exit_radius_m)


def _slant(geometry, z: float) -> float:
    if z <= geometry.cone_start_z_m or geometry.cone_start_z_m >= geometry.z_max_m:
        return 1.0
    slope = (geometry.exit_radius_m - geometry.bore_radius_m) / (geometry.z_max_m - geometry.cone_start_z_m)
    return sqrt(1.0 + slope * slope)


def extract_targets(
    maps: Mapping[str, np.ndarray],
    mapping: PicMapping,
    cusps_z_m: Sequence[float],
    cells: Sequence[Mapping[str, Any]],
    *,
    window_currents: Mapping[str, float] | None = None,
    injected_electron_current_a: float | None = None,
    cusp_half_width_m: float | None = None,
    near_wall_band_m: float = 5.0e-4,
    anode_edge_band_m: float = 2.5e-4,
) -> dict[str, Any]:
    """Per-cusp / per-cell closure targets from the window-averaged maps of one finished run (pure numpy).

    ``anode_edge_band_m``: the wall electron current within this distance of the anode plane is reported separately
    (``anode_edge_electron_wall_current_a``) - the v3.1 boundary-ambiguity tolerance (0.25 mm); design 047's disclosed
    anode-edge boundary cusp (separatrix at the wall 0.073 mm from the anode under iron) falls in this band, so its
    electron loss is visible without being counted as an interior cusp.
    """

    grid = mapping.grid
    geometry = grid.geometry
    dr, dz = grid.dr_m, grid.dz_m
    phi = np.asarray(maps["phi_v"], dtype=float)
    t_e = np.asarray(maps["t_e_ev"], dtype=float)
    n_e = np.asarray(maps["n_e_per_m3"], dtype=float)
    ionisation = np.asarray(maps["ionization_rate_per_m3_s"], dtype=float)
    wall_e = np.asarray(maps["wall_electron_flux_per_m2_s"], dtype=float)
    wall_i = np.asarray(maps["wall_ion_flux_per_m2_s"], dtype=float)
    wall_e_energy = np.asarray(maps.get("wall_electron_mean_energy_ev", np.zeros_like(wall_e)), dtype=float)
    nr1, nz1 = phi.shape
    r_nodes = np.arange(nr1) * dr
    z_nodes = geometry.z_min_m + np.arange(nz1) * dz
    channel_cells = int(round(geometry.channel_length_m / dz))
    z_cells = geometry.z_min_m + (np.arange(min(wall_e.size, channel_cells)) + 0.5) * dz
    area = np.array([2.0 * pi * _wall_radius(geometry, z) * dz * _slant(geometry, z) for z in z_cells])
    current_e = ELEMENTARY_CHARGE_C * wall_e[: z_cells.size] * area
    current_i = ELEMENTARY_CHARGE_C * wall_i[: z_cells.size] * area
    pitch = None
    ordered = sorted(float(z) for z in cusps_z_m)
    if len(ordered) >= 2:
        pitch = min(b - a for a, b in zip(ordered, ordered[1:]))
    half_width = cusp_half_width_m if cusp_half_width_m is not None else (1.0e-3 if pitch is None else min(1.0e-3, 0.25 * pitch))
    radial_weight = np.where(r_nodes > 0.0, r_nodes, 0.25 * dr)[:, None] * dr * dz * 2.0 * pi   # node volumes (axis node -> dr/4 surrogate)
    inside = np.zeros_like(phi, dtype=bool)
    for j, z in enumerate(z_nodes):
        inside[:, j] = r_nodes <= _wall_radius(geometry, z) + 1e-12 if z <= geometry.z_max_m else r_nodes <= geometry.max_radius_m + 1e-12
    weight_volume = radial_weight * inside
    total_ionisation = float((ionisation * weight_volume).sum())

    def cusp_block(z_c: float) -> dict[str, Any]:
        j = int(round((z_c - geometry.z_min_m) / dz))
        j = max(0, min(nz1 - 1, j))
        wall_index = min(int(round(_wall_radius(geometry, z_c) / dr)), nr1 - 1)
        axis_phi, wall_phi = float(phi[0, j]), float(phi[wall_index, j])
        near_wall_phi = float(phi[max(0, wall_index - 3), j])
        band = max(1, int(round(near_wall_band_m / dr)))
        j_lo, j_hi = max(0, int(round((z_c - half_width - geometry.z_min_m) / dz))), min(nz1, int(round((z_c + half_width - geometry.z_min_m) / dz)) + 1)
        block_t, block_n = t_e[max(0, wall_index - band):wall_index, j_lo:j_hi], n_e[max(0, wall_index - band):wall_index, j_lo:j_hi]
        weight = float(block_n.sum())
        near_wall_t = float((block_t * block_n).sum() / weight) if weight > 0 else float("nan")
        near_wall_n = float(block_n.mean()) if block_n.size else float("nan")
        mask = (z_cells >= z_c - half_width) & (z_cells <= z_c + half_width)
        i_e, i_i = float(current_e[mask].sum()), float(current_i[mask].sum())
        energy_weight = float((wall_e[: z_cells.size] * area)[mask].sum())
        mean_energy = float(((wall_e[: z_cells.size] * area * wall_e_energy[: z_cells.size])[mask]).sum() / energy_weight) if energy_weight > 0 else float("nan")
        profile = current_e[mask]
        fwhm = None
        if profile.size and profile.max() > 0.0:
            above = np.flatnonzero(profile >= 0.5 * profile.max())
            fwhm = float((above[-1] - above[0] + 1) * dz)
        drop = axis_phi - wall_phi
        return {"z_c_m": z_c, "window_half_width_m": half_width, "axis_potential_v": axis_phi, "near_wall_potential_v": near_wall_phi, "wall_potential_v": wall_phi,
                "sheath_drop_v": drop, "near_wall_drop_v": near_wall_phi - wall_phi, "near_wall_electron_temperature_ev": near_wall_t,
                "axis_electron_temperature_ev": float(t_e[0, j]), "sheath_drop_over_near_wall_temperature": (drop / near_wall_t if near_wall_t > 0 else float("nan")),
                "near_wall_electron_density_per_m3": near_wall_n, "electron_wall_current_a": i_e, "ion_wall_current_a": i_i,
                "electron_wall_mean_energy_ev": mean_energy, "leak_width_fwhm_m": fwhm}

    cusps = [cusp_block(z) for z in ordered]
    cusp_mask_total = np.zeros(z_cells.size, dtype=bool)
    for z in ordered:
        cusp_mask_total |= (z_cells >= z - half_width) & (z_cells <= z + half_width)
    diffuse_e = float(current_e[~cusp_mask_total].sum())
    anode_edge_mask = (z_cells <= geometry.z_min_m + anode_edge_band_m) & ~cusp_mask_total
    anode_edge_e = float(current_e[anode_edge_mask].sum())

    cell_rows = []
    ordered_cells = sorted(cells, key=lambda c: float(c["z_start_m"]))
    # every node column belongs to exactly one cell (half-open intervals; the last boundary column joins the last cell)
    boundaries = np.array([float(c["z_start_m"]) for c in ordered_cells] + [float(ordered_cells[-1]["z_end_m"])])
    column_cell = np.clip(np.searchsorted(boundaries, z_nodes + 1e-12, side="right") - 1, -1, len(ordered_cells) - 1)
    column_cell[z_nodes > boundaries[-1] + 1e-12] = -1
    for index, cell in enumerate(ordered_cells):
        z_lo, z_hi = float(cell["z_start_m"]), float(cell["z_end_m"])
        cols = column_cell == index
        wv = weight_volume[:, cols]
        block_n, block_phi, block_t = n_e[:, cols], phi[:, cols], t_e[:, cols]
        w = block_n * wv
        total = float(w.sum())
        s_cell = float((ionisation[:, cols] * wv).sum())
        wall_mask = (z_cells >= z_lo) & (z_cells < z_hi)
        ion_wall = float(current_i[wall_mask].sum())
        cell_rows.append({
            "cell_id": cell["cell_id"], "kind": cell["kind"], "z_start_m": z_lo, "z_end_m": z_hi,
            "density_weighted_potential_v": float((block_phi * w).sum() / total) if total > 0 else float("nan"),
            "density_weighted_electron_temperature_ev": float((block_t * w).sum() / total) if total > 0 else float("nan"),
            "electron_inventory": float((block_n * wv).sum()), "peak_electron_density_per_m3": float(block_n.max()) if block_n.size else float("nan"),
            "ionisation_rate_per_s": s_cell, "ionisation_share": (s_cell / total_ionisation) if total_ionisation > 0 else float("nan"),
            "ionisation_current_a": ELEMENTARY_CHARGE_C * s_cell, "ion_wall_current_a": ion_wall,
            "ion_wall_loss_fraction": (ion_wall / (ELEMENTARY_CHARGE_C * s_cell)) if s_cell > 0 else float("nan"),
            "electron_wall_current_a": float(current_e[wall_mask].sum()),
        })
    steps = [b["density_weighted_potential_v"] - a["density_weighted_potential_v"] for a, b in zip(cell_rows, cell_rows[1:])]

    # Kornfeld chain from the exit side: je arriving at cusp k = entering current + ionisation of the cells passed - losses at the cusps passed
    chain = None
    entering = None
    if injected_electron_current_a is not None:
        entering = injected_electron_current_a
    elif window_currents is not None and "injected_electron_a" in window_currents:
        entering = float(window_currents["injected_electron_a"]) - float(window_currents.get("exit_electron_a", 0.0))
    if entering is not None and cell_rows:
        rows = []
        je = entering
        exit_to_anode_cells = cell_rows[::-1]
        for index, cusp in enumerate(cusps[::-1]):     # exit-most cusp first
            je_arriving = je + exit_to_anode_cells[index]["ionisation_current_a"]
            p = cusp["electron_wall_current_a"] / je_arriving if je_arriving > 0 else float("nan")
            rows.append({"z_c_m": cusp["z_c_m"], "je_arriving_a": je_arriving, "electron_wall_current_a": cusp["electron_wall_current_a"], "p_transit": p,
                         "effective_loss_frequency_per_s": (cusp["electron_wall_current_a"] / (ELEMENTARY_CHARGE_C * exit_to_anode_cells[index]["electron_inventory"]))
                         if exit_to_anode_cells[index]["electron_inventory"] > 0 else float("nan")})
            je = je_arriving - cusp["electron_wall_current_a"]
        chain = {"entering_electron_current_a": entering, "cusps_exit_to_anode": rows, "electron_current_reaching_anode_cell_a": je + exit_to_anode_cells[-1]["ionisation_current_a"] if len(exit_to_anode_cells) > len(cusps) else je,
                 "note": "Kornfeld per-transit probabilities from window currents: p_k = L_k / je_arriving,k; the model's je_k = je_{k-1}(1-p_k) + I_k"}
    return {
        "window_steps": int(np.asarray(maps["window_steps"]).ravel()[0]) if "window_steps" in maps else None,
        "cusp_half_width_m": half_width,
        "cusps": cusps, "cells": cell_rows, "potential_steps_v": steps,
        "total_wall_electron_current_a": float(current_e.sum()), "total_wall_ion_current_a": float(current_i.sum()),
        "diffuse_non_cusp_electron_wall_current_a": diffuse_e, "total_ionisation_rate_per_s": total_ionisation,
        "anode_edge_band_m": anode_edge_band_m, "anode_edge_electron_wall_current_a": anode_edge_e,
        "anode_edge_note": "wall electron current within anode_edge_band_m of the anode plane outside every cusp window (part of the diffuse current); "
                           "the disclosed 047 anode-edge boundary cusp lives here",
        "kornfeld_chain": chain, "kornfeld_mapping": kornfeld_mapping(ordered, cells),
        "phi_max_v": float(phi.max()), "phi_min_v": float(phi.min()),
    }


def load_maps(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


__all__ = ["CLOSURE_TARGETS", "ClosureTarget", "closure_target_table", "extract_targets", "kornfeld_mapping", "load_maps"]
