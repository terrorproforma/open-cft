"""PIC-side inputs of L2 v2: the two closures and the reference quantities with their particle bands.

Everything here is read from the ACCEPTED steady-state v2 artifacts (``results/`` base plateau, ``results-seed-b``,
``results-w-0.7``) through the design mini-sweep's extraction ``extract_targets`` (the same per-cell / per-cusp
accounting the sweep applies to every PIC design), on the L2 cell partition (v3.1 catalogue cells extended to the
electrodes).  Two numbers per cusp become L2 CLOSURES:

* the effective cusp conductance ``G_k = e F_k / drive_k`` with ``F_k`` the electron current PASSING cusp k
  (Kornfeld chain: arriving minus lost) and ``drive_k`` the thermalised-potential difference of the two cells
  ``(phi_k - phi_k+1) - (n_k T_k - n_k+1 T_k+1) / mean(n)`` - the generalised Ohm's law L2 uses across the cusp;
* the leak half-width ``w_k = FWHM_k / 2`` of the PIC's wall electron flux about the cusp, which sets the flux tubes
  L2 populates with electrons.

Everything else the PIC gives is a REFERENCE for the comparison gate (never an input): I_d, S, utilisation, n_g,
peak n_e, beam current, wall currents, per-cell potentials / steps / temperatures / ionisation shares / ion wall
losses, per-cusp ion wall currents and sheath drops.  Bands are the largest relative deviation of the two PIC
convergence pairs (seed-b, W x 0.7) from the base plateau; the predeclared tolerance is ``clip(2 x band, 5 %, 12 %)``
and a quantity whose own PIC band exceeds 12 % is ``not_compared`` (the reference is then less precise than the cap).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.hybrid.cells import CellPartition
from cft_revival.pic2d.models import ChannelGeometry, Grid2D
from experiments.pic2d_design_mini_sweep_v1.closure import extract_targets, load_maps

ELEMENTARY_CHARGE_C = 1.602176634e-19
TOLERANCE_FLOOR = 0.05
TOLERANCE_CAP = 0.12


class _Mapping:
    def __init__(self, grid: Grid2D) -> None:
        self.grid = grid


def pic_grid(summary: Mapping[str, Any]) -> Grid2D:
    g = summary["provenance"]["config"]["grid"]
    geometry = g["geometry"]
    return Grid2D(ChannelGeometry(geometry["bore_radius_m"], geometry["z_min_m"], geometry["z_max_m"], geometry["cone_start_z_m"],
                                  geometry["exit_radius_m"]), int(g["radial_cells"]), int(g["axial_cells"]))


def partition_cells(partition: CellPartition) -> list[dict[str, Any]]:
    return [{"cell_id": c, "kind": k, "z_start_m": a, "z_end_m": b}
            for c, k, a, b in zip(partition.cell_ids, partition.kinds, partition.z_start_m, partition.z_end_m, strict=True)]


def pic_run_targets(results_dir: Path, partition: CellPartition) -> dict[str, Any]:
    """Per-cell / per-cusp extraction of one finished PIC run on the L2 partition, plus the scalar window quantities."""

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    maps = load_maps(results_dir / "maps.npz")
    grid = pic_grid(summary)
    targets = extract_targets(maps, _Mapping(grid), list(partition.cusp_z_m), partition_cells(partition), window_currents=summary["window_currents_a"])
    n_e = np.asarray(maps["n_e_per_m3"])
    peak = int(np.nanargmax(n_e))
    i, j = np.unravel_index(peak, n_e.shape)
    return {
        "results_dir": results_dir.name,
        "maps_sha256": json.loads((results_dir / "maps.npz.sha256.json").read_text(encoding="utf-8"))["byte_sha256"],
        "summary_sha256": json.loads((results_dir / "summary.json.sha256.json").read_text(encoding="utf-8"))["byte_sha256"],
        "window_currents_a": summary["window_currents_a"],
        "neutral_density_per_m3": summary["neutral_inventory"]["trailing_20pct_mean_density_per_m3"],
        "ionization_rate_per_s": summary["neutral_inventory"]["trailing_20pct_mean_ionization_rate_per_s"],
        "gross_utilisation": summary["neutral_inventory"]["propellant_utilisation_trailing"],
        "peak_n_e_per_m3": float(n_e[i, j]),
        "peak_node": [int(i), int(j)],
        "t_e_peak_ev": float(np.asarray(maps["t_e_ev"])[i, j]),
        "targets": targets,
    }


def cusp_conductances(targets: Mapping[str, Any]) -> tuple[list[float], list[float], list[float]]:
    """``(G_k, passing current, drive)`` per cusp (anode -> exit order) from the chain and the cell rows."""

    cells = targets["cells"]
    chain = targets["kornfeld_chain"]["cusps_exit_to_anode"]
    by_z = {round(row["z_c_m"], 9): row for row in chain}
    volumes = cell_volumes_m3(targets)
    conductances, passing, drives = [], [], []
    for k, cusp in enumerate(targets["cusps"]):
        row = by_z[round(cusp["z_c_m"], 9)]
        f_pass = row["je_arriving_a"] - row["electron_wall_current_a"]
        a, b = cells[k], cells[k + 1]
        n_a, n_b = a["electron_inventory"] / volumes[k], b["electron_inventory"] / volumes[k + 1]
        t_a, t_b = a["density_weighted_electron_temperature_ev"], b["density_weighted_electron_temperature_ev"]
        drive = (a["density_weighted_potential_v"] - b["density_weighted_potential_v"]) - (n_a * t_a - n_b * t_b) / (0.5 * (n_a + n_b))
        conductances.append(f_pass / drive)
        passing.append(f_pass)
        drives.append(drive)
    return conductances, passing, drives


def cell_volumes_m3(targets: Mapping[str, Any], *, bore_radius_m: float = 0.002, cone_start_z_m: float = 0.018, exit_radius_m: float = 0.003,
                    z_max_m: float = 0.024) -> list[float]:
    """Geometric volume of every partition cell (straight bore plus the divergent cone share)."""

    volumes = []
    for cell in targets["cells"]:
        a, b = float(cell["z_start_m"]), float(cell["z_end_m"])
        straight = np.pi * bore_radius_m**2 * (min(b, cone_start_z_m) - a) if a < cone_start_z_m else 0.0
        cone = 0.0
        lo, hi = max(a, cone_start_z_m), min(b, z_max_m)
        if hi > lo:
            slope = (exit_radius_m - bore_radius_m) / (z_max_m - cone_start_z_m)
            r_lo = bore_radius_m + slope * (lo - cone_start_z_m)
            r_hi = bore_radius_m + slope * (hi - cone_start_z_m)
            cone = np.pi / 3.0 * (hi - lo) * (r_lo**2 + r_lo * r_hi + r_hi**2)
        volumes.append(float(straight + cone))
    return volumes


def leak_half_widths_m(targets: Mapping[str, Any]) -> list[float]:
    return [0.5 * float(cusp["leak_width_fwhm_m"]) for cusp in targets["cusps"]]


def scalar_quantities(run: Mapping[str, Any]) -> dict[str, float]:
    """The scalar and per-cell reference quantities of one PIC run, in the L2 comparison's naming."""

    t = run["targets"]
    wc = run["window_currents_a"]
    out: dict[str, float] = {
        "discharge_current_a": wc["discharge_a"], "ionization_rate_per_s": run["ionization_rate_per_s"],
        "gross_utilisation": run["gross_utilisation"], "neutral_density_per_m3": run["neutral_density_per_m3"],
        "peak_n_e_per_m3": run["peak_n_e_per_m3"], "exit_ion_beam_a": wc["exit_ion_beam_a"],
        "wall_ion_current_a": t["total_wall_ion_current_a"], "wall_electron_current_a": t["total_wall_electron_current_a"],
        "exit_electron_current_a": wc["exit_electron_a"], "anode_ion_fraction": wc["anode_ion_a"] / wc["discharge_a"],
    }
    for k, cell in enumerate(t["cells"]):
        out[f"cell{k}_potential_v"] = cell["density_weighted_potential_v"]
        out[f"cell{k}_temperature_ev"] = cell["density_weighted_electron_temperature_ev"]
        out[f"cell{k}_ionisation_share"] = cell["ionisation_share"]
        out[f"cell{k}_ion_wall_loss_fraction"] = cell["ion_wall_loss_fraction"]
        out[f"cell{k}_electron_inventory"] = cell["electron_inventory"]
    for k, step in enumerate(t["potential_steps_v"]):
        out[f"step{k}_v"] = step
    for k, cusp in enumerate(t["cusps"]):
        out[f"cusp{k}_ion_wall_current_a"] = cusp["ion_wall_current_a"]
        out[f"cusp{k}_electron_wall_current_a"] = cusp["electron_wall_current_a"]
        out[f"cusp{k}_near_wall_drop_v"] = cusp["near_wall_drop_v"]
    return out


def reference_with_bands(base: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reference values (base plateau), particle bands (max |pair - base| / |base|) and predeclared tolerances."""

    base_q = scalar_quantities(base)
    pair_q = [scalar_quantities(run) for run in pairs]
    table: dict[str, Any] = {}
    for key, value in base_q.items():
        deviations = [abs(q[key] - value) / abs(value) for q in pair_q if value != 0.0 and np.isfinite(q[key]) and np.isfinite(value)]
        band = max(deviations) if deviations else None
        if band is None or not np.isfinite(value) or value == 0.0:
            tolerance, status = None, "not_compared_no_band"
        elif band > TOLERANCE_CAP:
            tolerance, status = None, "not_compared_pic_band_exceeds_cap"
        else:
            tolerance, status = float(min(max(2.0 * band, TOLERANCE_FLOOR), TOLERANCE_CAP)), "compared"
        table[key] = {"reference": float(value), "pairs": [float(q[key]) for q in pair_q], "band": band, "tolerance": tolerance, "status": status}
    return table


def build_pic_reference(partition: CellPartition, v2_dir: Path) -> dict[str, Any]:
    base = pic_run_targets(v2_dir / "results", partition)
    pairs = [pic_run_targets(v2_dir / name, partition) for name in ("results-seed-b", "results-w-0.7")]
    g, passing, drives = cusp_conductances(base["targets"])
    g_pairs = [cusp_conductances(run["targets"])[0] for run in pairs]
    widths = leak_half_widths_m(base["targets"])
    width_pairs = [leak_half_widths_m(run["targets"]) for run in pairs]
    return {
        "source": "experiments/pic2d_cft_steady_state_v2 (accepted base plateau + convergence pairs seed-b, W x 0.7)",
        "base": {k: v for k, v in base.items() if k != "targets"}, "pairs": [{k: v for k, v in run.items() if k != "targets"} for run in pairs],
        "closures": {
            "cusp_conductance_s": g, "cusp_conductance_pairs_s": g_pairs, "passing_current_a": passing, "drive_v": drives,
            "leak_half_width_m": widths, "leak_half_width_pairs_m": width_pairs,
            "definition": "G_k = e F_pass,k / drive_k with drive_k = (phi_k - phi_k+1) - (n_k T_k - n_k+1 T_k+1) / mean(n); w_k = FWHM_k / 2",
        },
        "quantities": reference_with_bands(base, pairs),
        "base_targets": base["targets"],
    }


__all__ = ["TOLERANCE_CAP", "TOLERANCE_FLOOR", "build_pic_reference", "cell_volumes_m3", "cusp_conductances", "leak_half_widths_m",
           "partition_cells", "pic_run_targets", "reference_with_bands", "scalar_quantities"]
