"""Post-mortem of the v1 snapshot: trip cell, source/loss balance, 0-D equilibrium, step cost.

From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_snapshot_v1.diagnose            # (a) + (b) from the artifacts
    python -m experiments.pic2d_cft_snapshot_v1.diagnose --profile  # (c) per-phase GPU step cost

Writes ``results/diagnosis.json`` (canonical JSON + sha256 sidecar).  Every
number is derived from the committed artifacts (series, maps, final
checkpoints) or from the bound cross-section table; nothing here is a
physics claim beyond the v1 model.
"""

from __future__ import annotations

import argparse
import json
from math import pi, sqrt
import sys
import time
from typing import Any

import numpy as np

from cft_revival.pic2d import artifacts, kernels
from cft_revival.pic2d.fields import build_p2_psi_field, sample_field_map
from cft_revival.pic2d.mcc import UniformSigmaTable, XenonCrossSections, electron_speed_from_energy
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, EPSILON_0_F_PER_M, EV_J, electron_species, xenon_ion_species

from .run import HERE, REPOSITORY_ROOT, RESULTS, build_config, load_protocol

XE_MASS_KG = xenon_ion_species(1.0).mass_kg


def plasma_frequency(density: float) -> float:
    return sqrt(density * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ELECTRON_MASS_KG))


def debye_length(density: float, t_e_ev: float) -> float:
    return sqrt(EPSILON_0_F_PER_M * t_e_ev / (density * ELEMENTARY_CHARGE_C))


def density_from_omega_dt(omega_dt: float, dt: float) -> float:
    omega = omega_dt / dt
    return omega**2 * EPSILON_0_F_PER_M * ELECTRON_MASS_KG / ELEMENTARY_CHARGE_C**2


def maxwellian_rate_coefficient(table: UniformSigmaTable, process: int, t_e_ev: float) -> float:
    """``<sigma v>`` for an isotropic Maxwellian of temperature ``t_e_ev`` (m^3/s)."""

    energy = np.arange(table.point_count, dtype=np.float64) * table.energy_step_ev
    speed = electron_speed_from_energy(energy)
    sigma = table.table_m2[process]
    # f(E) dE = 2 sqrt(E/pi) T^-3/2 exp(-E/T) dE
    weights = 2.0 * np.sqrt(np.maximum(energy, 0.0) / pi) * t_e_ev ** -1.5 * np.exp(-energy / t_e_ev)
    return float(np.trapezoid(weights * sigma * speed, energy))


def loss_geometry(protocol: dict[str, Any]) -> dict[str, float]:
    g = protocol["geometry"]
    r0, r1 = g["bore_radius_m"], g["exit_radius_m"]
    z0, z1, z2 = g["z_min_m"], g["cone_start_z_m"], g["z_max_m"]
    straight = 2.0 * pi * r0 * (z1 - z0)
    slant = sqrt((r1 - r0) ** 2 + (z2 - z1) ** 2)
    cone = pi * (r0 + r1) * slant
    anode = pi * r0**2
    exit_plane = pi * r1**2
    volume = pi * r0**2 * (z1 - z0) + pi * (z2 - z1) * (r0**2 + r0 * r1 + r1**2) / 3.0
    return {
        "wall_area_m2": straight + cone, "anode_area_m2": anode, "exit_area_m2": exit_plane,
        "total_loss_area_m2": straight + cone + anode + exit_plane, "volume_m3": volume,
    }


def trip_cell(case: str, protocol: dict[str, Any]) -> dict[str, Any]:
    """Peak-node electron density of the final checkpoint versus the window-averaged map peak."""

    config, _spec = build_config(protocol, case)
    masks = build_mesh_masks(config.grid)
    out = RESULTS / case
    summary = artifacts.read_canonical_json(out / "summary.json")
    arrays = artifacts.read_npz(out / "checkpoint-final.npz")
    electrons = artifacts._particles_from_arrays("electrons", arrays)
    species = electron_species(config.macro_weight)
    q_e = kernels.deposit_node_charge(masks, species, electrons, fixed_point=True)
    n_e = np.zeros(config.grid.node_shape)
    n_e[masks.plasma_node] = np.abs(q_e[masks.plasma_node]) / (ELEMENTARY_CHARGE_C * masks.shape_volume_m3[masks.plasma_node])
    i, j = np.unravel_index(int(np.argmax(n_e)), n_e.shape)
    peak = float(n_e[i, j])
    order = np.argsort(n_e.ravel())[::-1][:5]
    top = [
        {"i": int(k // n_e.shape[1]), "j": int(k % n_e.shape[1]), "r_mm": float(config.grid.r_m[k // n_e.shape[1]] * 1e3),
         "z_mm": float(config.grid.z_m[k % n_e.shape[1]] * 1e3), "n_e_per_m3": float(n_e.ravel()[k]),
         "omega_pe_dt": plasma_frequency(float(n_e.ravel()[k])) * config.dt_s}
        for k in order
    ]
    maps = artifacts.read_npz(out / "maps.npz")
    map_peak = float(np.nanmax(maps["n_e_per_m3"][masks.plasma_node]))
    series = artifacts.read_npz(out / "series.npz")
    gate_omega = float(series["peak_omega_pe_dt"][-1])
    # count of nodes above the gate density
    n_gate = density_from_omega_dt(config.limits.max_omega_pe_dt, config.dt_s)
    above = int(np.count_nonzero(n_e > n_gate))
    axis_fraction = float(np.count_nonzero(n_e[0] > 0.5 * peak)) / max(int(np.count_nonzero(n_e > 0.5 * peak)), 1)
    return {
        "case": case,
        "steps_completed": summary["steps_completed"],
        "stop_reason": summary["stop_reason"],
        "gate_message": summary.get("stability_gate_message"),
        "gate_density_per_m3": n_gate,
        "final_checkpoint_peak_node": top[0],
        "final_checkpoint_top5_nodes": top,
        "nodes_above_gate_density": above,
        "fraction_of_hot_nodes_on_axis": axis_fraction,
        "window_average_map_peak_n_e_per_m3": map_peak,
        "peak_to_window_ratio": peak / map_peak if map_peak > 0 else None,
        "last_series_peak_omega_pe_dt": gate_omega,
        "last_series_implied_peak_density_per_m3": density_from_omega_dt(gate_omega, config.dt_s),
        "note": (
            "the runtime gate uses the instantaneous peak node density (single node, shape-volume weighted); "
            "the reported map peak is a window average and therefore lower"
        ),
    }


def source_loss(case: str, protocol: dict[str, Any]) -> dict[str, Any]:
    config, _spec = build_config(protocol, case)
    out = RESULTS / case
    summary = artifacts.read_canonical_json(out / "summary.json")
    cumulative = summary["final_series"]["ledger"]["cumulative"]
    w = config.macro_weight
    ionised = cumulative["ionizations"] * w
    lost = (cumulative["anode_ions"] + cumulative["exit_ions"] + cumulative["wall_ions"]) * w
    series = artifacts.read_npz(out / "series.npz")
    t = series["time_s"]
    ionisation_rate = series["current_ionization_rate_per_s"]
    ion_loss_rate = (series["current_wall_ion_a"] + series["current_anode_ion_a"] + series["current_exit_ion_beam_a"]) / ELEMENTARY_CHARGE_C
    # exponential growth rate of the electron count over the second half of the run
    n = series["electrons"]
    half = n.size // 2
    slope = np.polyfit(t[half:], np.log(np.maximum(n[half:], 1.0)), 1)[0] if n.size > 4 else float("nan")
    return {
        "case": case,
        "simulated_time_s": summary["simulated_time_s"],
        "ions_created": ionised,
        "ions_lost_total": lost,
        "ions_lost_wall": cumulative["wall_ions"] * w,
        "ions_lost_anode": cumulative["anode_ions"] * w,
        "ions_lost_exit": cumulative["exit_ions"] * w,
        "loss_to_source_ratio": lost / ionised if ionised > 0 else None,
        "final_ionisation_rate_per_s": float(ionisation_rate[-1]),
        "final_ion_loss_rate_per_s": float(ion_loss_rate[-1]),
        "final_loss_to_source_rate_ratio": float(ion_loss_rate[-1] / ionisation_rate[-1]) if ionisation_rate[-1] > 0 else None,
        "electron_count_e_folding_time_s": float(1.0 / slope) if slope > 0 else None,
        "ion_transit_estimate_s": {
            "note": "axial 24 mm at the 300 V Xe+ speed (sqrt(2 e U / M)); radial 2 mm at the Bohm speed for T_e = 18 eV",
            "axial_300v_s": 0.024 / sqrt(2.0 * ELEMENTARY_CHARGE_C * 300.0 / XE_MASS_KG),
            "radial_bohm_18ev_s": 0.002 / sqrt(ELEMENTARY_CHARGE_C * 18.0 / XE_MASS_KG),
        },
    }


def equilibrium_table(protocol: dict[str, Any], t_e_ev: float, i_d_a: float, u_a_v: float, epsilon_ev: float | None = None) -> dict[str, Any]:
    """0-D particle balance (amplification) and power balance (density) for several n_g."""

    cross_sections = XenonCrossSections.from_file()
    table = UniformSigmaTable.build(cross_sections)
    geometry = loss_geometry(protocol)
    k_iz = maxwellian_rate_coefficient(table, 2, t_e_ev)
    k_exc = maxwellian_rate_coefficient(table, 1, t_e_ev)
    k_el = maxwellian_rate_coefficient(table, 0, t_e_ev)
    v_bohm = sqrt(ELEMENTARY_CHARGE_C * t_e_ev / XE_MASS_KG)
    a_loss = geometry["total_loss_area_m2"]
    volume = geometry["volume_m3"]
    # collisional energy cost per electron-ion pair (Lieberman & Lichtenberg eq. 3.5.8 form)
    eps_c = (k_iz * 12.13 + k_exc * 8.32 + k_el * 3.0 * ELECTRON_MASS_KG / XE_MASS_KG * t_e_ev) / k_iz
    # energy carried out per pair: ions ~ (sheath ~ 5 T_e) + electrons 2 T_e (Bohm/Maxwellian wall losses)
    eps_pair = eps_c + 7.0 * t_e_ev if epsilon_ev is None else epsilon_ev
    power_w = i_d_a * u_a_v
    rows = []
    for n_g in (5.0e20, 1.0e20, 5.0e19):
        source_rate_per_electron = n_g * k_iz
        loss_rate_per_electron = v_bohm * a_loss / volume
        amplification = source_rate_per_electron / loss_rate_per_electron
        # power balance: P = n_eq v_B A_loss e eps_pair
        n_eq = power_w / (v_bohm * a_loss * ELEMENTARY_CHARGE_C * eps_pair)
        # particle-balance temperature: n_g K_iz(T) V = v_B(T) A  (solve on a grid)
        grid = np.linspace(1.0, 60.0, 591)
        residual = [n_g * maxwellian_rate_coefficient(table, 2, te) * volume - sqrt(ELEMENTARY_CHARGE_C * te / XE_MASS_KG) * a_loss for te in grid]
        residual = np.asarray(residual)
        sign = np.sign(residual)
        crossings = np.nonzero(np.diff(sign) != 0)[0]
        t_balance = float(grid[crossings[0]]) if crossings.size else None
        rows.append({
            "n_g_per_m3": n_g,
            "k_iz_m3_per_s_at_t_e": k_iz,
            "ionisation_rate_per_electron_per_s": source_rate_per_electron,
            "bohm_loss_rate_per_electron_per_s": loss_rate_per_electron,
            "amplification_source_over_loss": amplification,
            "particle_balance_t_e_ev": t_balance,
            "n_eq_power_balance_per_m3": n_eq,
            "lambda_d_m_at_n_eq": debye_length(n_eq, t_e_ev),
            "omega_pe_rad_per_s_at_n_eq": plasma_frequency(n_eq),
            "omega_pe_dt_at_n_eq_for_dt_2e-12": plasma_frequency(n_eq) * 2.0e-12,
            "omega_pe_dt_at_n_eq_for_dt_1e-12": plasma_frequency(n_eq) * 1.0e-12,
        })
    return {
        "assumptions": {
            "t_e_ev": t_e_ev, "discharge_current_a": i_d_a, "anode_potential_v": u_a_v, "power_w": power_w,
            "collisional_cost_per_pair_ev": eps_c, "energy_per_pair_ev": eps_pair,
            "loss_area_m2": a_loss, "volume_m3": volume, "bohm_speed_m_per_s": v_bohm,
            "note": (
                "particle balance n_g <sigma_iz v> n_e V = n_e v_B A_loss cancels n_e: at the observed T_e it gives the "
                "source/loss amplification (>1 means the avalanche cannot saturate by ion loss alone); the density scale "
                "comes from power balance P = I_d U_a = n_eq v_B A_loss e (eps_c + 7 T_e). Unmagnetised Bohm loss to all "
                "surfaces is an upper bound on the loss rate, so n_eq is a lower bound for a magnetised discharge."
            ),
        },
        "geometry": geometry,
        "rows": rows,
    }


def profile_step(case: str, protocol: dict[str, Any], *, steps: int, method: str, subcycle: int | None = None,
                 phases: bool = True) -> dict[str, Any]:
    """Per-phase GPU cost of one time step restarted from the final checkpoint.

    ``phases=False`` skips the per-phase device synchronisations so the total is
    the true pipelined throughput (each synchronisation costs ~ms on WDDM).
    """

    from dataclasses import replace

    from cft_revival.pic2d.simulation import Simulation
    from cft_revival.pic2d.models import PoissonConfig2D

    config, _spec = build_config(protocol, case)
    poisson = PoissonConfig2D(method=method, relative_tolerance=config.poisson.relative_tolerance if method == "direct" else 1.0e-6)
    config = replace(config, poisson=poisson)
    if subcycle is not None:
        config = replace(config, ion_subcycle=subcycle)
    psi_field, evidence = build_p2_psi_field(REPOSITORY_ROOT, role="primary")
    field_map = sample_field_map(psi_field, config.grid, evidence)
    cross_sections = XenonCrossSections.from_file()
    sim = Simulation(config, field_map, cross_sections=cross_sections, backend="warp-cuda")
    arrays = artifacts.read_npz(RESULTS / case / "checkpoint-final.npz")
    metadata = artifacts.read_canonical_json(RESULTS / case / "checkpoint-final.json")
    from cft_revival.pic2d.simulation import CUMULATIVE_KEYS, SimulationState

    cumulative = {key: float(value) for key, value in zip(CUMULATIVE_KEYS, arrays["cumulative"], strict=True)}
    state = SimulationState(
        int(metadata["step"]), float(metadata["time_s"]),
        artifacts._particles_from_arrays("electrons", arrays), artifacts._particles_from_arrays("ions", arrays),
        np.asarray(arrays["surface_charge_c"]), np.asarray(arrays["phi_v"]), float(metadata["injection_carry"]), cumulative,
    )
    # keep the gate from firing during the profile (the checkpoint is the tripped state)
    sim.load_state(state)
    backend = sim.backend
    # warm-up (kernel compilation, graph capture)
    for _ in range(5):
        backend.step(False)
    backend.profile = {} if phases else None
    import warp as wp

    wp.synchronize_device(backend.device)
    backend._profile_clock = time.perf_counter()
    t0 = time.perf_counter()
    iterations = []
    syncs_before = backend.sync_count
    for _ in range(steps):
        backend.step(False)
        iterations.append(backend.last_iterations)
    backend.flush()
    wp.synchronize_device(backend.device)
    wall = time.perf_counter() - t0
    profile = {key: value / steps * 1e3 for key, value in (backend.profile or {}).items()}
    counts = {"electrons": int(state.electrons.count), "ions": int(state.ions.count)}
    return {
        "case": case, "poisson_method": method, "steps": steps, "ms_per_step_total": wall / steps * 1e3,
        "ms_per_step_by_phase": profile, "mean_poisson_iterations": float(np.mean(iterations)),
        "particles": counts, "host_syncs_per_step": (backend.sync_count - syncs_before) / steps,
        "ion_subcycle": config.ion_subcycle, "device_sync_steps": config.sync_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="time the GPU step phases on the fine case")
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--profile-method", default="direct")
    parser.add_argument("--profile-case", default="fine-w2.5e4")
    parser.add_argument("--profile-subcycle", type=int, default=None)
    parser.add_argument("--no-phases", action="store_true", help="total throughput only (no per-phase device syncs)")
    args = parser.parse_args()
    protocol = load_protocol()
    if args.profile:
        report = profile_step(args.profile_case, protocol, steps=args.profile_steps, method=args.profile_method,
                              subcycle=args.profile_subcycle, phases=not args.no_phases)
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    cases = list(protocol["cases"])
    fine = "fine-w2.5e4"
    fine_summary = artifacts.read_canonical_json(RESULTS / fine / "summary.json")
    t_e = float(fine_summary["window_maps_summary"]["t_e_density_weighted_mean_ev"])
    i_d = float(fine_summary["final_series"]["currents_a"]["discharge_a"])
    report = {
        "schema_version": "cft-revival.pic2d-cft-snapshot-v1.diagnosis/0.1.0",
        "experiment_id": protocol["experiment_id"],
        "trip_cells": {case: trip_cell(case, protocol) for case in cases},
        "source_loss": {case: source_loss(case, protocol) for case in cases},
        "equilibrium_0d": {
            "at_observed_t_e_and_i_d": equilibrium_table(protocol, t_e, i_d, protocol["operating_point"]["anode_potential_v"]),
            "at_observed_t_e_and_injected_current": equilibrium_table(
                protocol, t_e, protocol["operating_point"]["electron_injection_current_a"], protocol["operating_point"]["anode_potential_v"]
            ),
        },
        "claim_boundary": "diagnostic post-mortem of a development run; 0-D estimates are order-of-magnitude budgets, not predictions",
    }
    artifacts.write_canonical_json(RESULTS / "diagnosis.json", report)
    for case in cases:
        tc = report["trip_cells"][case]
        sl = report["source_loss"][case]
        print(f"{case}: peak node n_e={tc['final_checkpoint_peak_node']['n_e_per_m3']:.3g} at r={tc['final_checkpoint_peak_node']['r_mm']:.2f} mm "
              f"z={tc['final_checkpoint_peak_node']['z_mm']:.2f} mm (map peak {tc['window_average_map_peak_n_e_per_m3']:.3g}); "
              f"ions created {sl['ions_created']:.3g}, lost {sl['ions_lost_total']:.3g} (ratio {sl['loss_to_source_ratio']:.3f})")
    for row in report["equilibrium_0d"]["at_observed_t_e_and_i_d"]["rows"]:
        print(f"n_g={row['n_g_per_m3']:.1e}: amplification={row['amplification_source_over_loss']:.2f} T_e,balance={row['particle_balance_t_e_ev']} "
              f"n_eq={row['n_eq_power_balance_per_m3']:.3g} lambda_D={row['lambda_d_m_at_n_eq']*1e6:.1f} um w_pe={row['omega_pe_rad_per_s_at_n_eq']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
