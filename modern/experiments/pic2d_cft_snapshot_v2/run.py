"""Snapshot v2 (model v1.1): run one case, or summarise all cases.

From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_snapshot_v2.run budget
    python -m experiments.pic2d_cft_snapshot_v2.run case coarse-w2.4e5 --max-wall-seconds 3600
    python -m experiments.pic2d_cft_snapshot_v2.run summarize

Differences from v1: the v1.1 operating point (3 mA, n_g = 1e20 m^-3, dt = 1.5 ps),
the all-GPU step (device block-Thomas Poisson, ion subcycling k = 8, host sync
every 200 steps), the electrode-work ledger, and a plateau stopping rule after a
minimum of one ion transit time.  Still a development/screening experiment: not
preregistered, no validated physics claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import build_p2_psi_field, sample_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.models import (
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DStabilityError,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation, SeriesRecord
from experiments.pic2d_cft_snapshot_v1.run import _exit_areas, _file_sha256, _gpu_utilisation, git_head

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
PROTOCOL_PATH = HERE / "protocol.json"
RESULTS = HERE / "results"
ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_0 = 8.8541878128e-12
ELECTRON_MASS_KG = 9.1093837139e-31
XE_MASS_KG = 2.1801714e-25


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def build_config(protocol: dict[str, Any], case: str, *, backend: str = "warp-cuda") -> tuple[PIC2DConfig, dict[str, Any]]:
    geometry = protocol["geometry"]
    spec = protocol["cases"][case]
    operating = protocol["operating_point"]
    numerics = protocol["numerics"]
    grid = Grid2D(
        ChannelGeometry(
            geometry["bore_radius_m"], geometry["z_min_m"], geometry["z_max_m"],
            geometry["cone_start_z_m"], geometry["exit_radius_m"],
        ),
        int(spec["radial_cells"]), int(spec["axial_cells"]),
    )
    limit_values = dict(numerics["stability_limits"])
    if case.startswith("coarse"):
        limit_values["max_cell_debye_ratio"] = float(numerics["coarse_debye_ratio_limit"])
    limits = StabilityLimits(**limit_values)
    config = PIC2DConfig(
        grid=grid,
        potentials=BoundaryPotentials(operating["anode_potential_v"], operating["exit_plane_potential_v"]),
        dt_s=float(numerics["dt_s"]),
        macro_weight=float(spec["macro_weight"]),
        seed=20260903,
        injection=InjectionConfig(operating["electron_injection_current_a"], operating["electron_injection_temperature_ev"]),
        seed_plasma=SeedPlasmaConfig(
            operating["seed_plasma_density_per_m3"], operating["seed_electron_temperature_ev"], operating["seed_ion_temperature_ev"]
        ),
        mcc=MCCConfig(operating["neutral_density_per_m3"], operating["neutral_temperature_k"]),
        poisson=PoissonConfig2D(method="device-direct" if backend == "warp-cuda" else "direct", relative_tolerance=1.0e-10),
        limits=limits,
        reference_density_per_m3=numerics["stability_reference"]["density_per_m3"],
        reference_electron_temperature_ev=numerics["stability_reference"]["electron_temperature_ev"],
        max_electron_energy_ev=numerics["stability_reference"]["max_electron_energy_ev"],
        series_interval_steps=int(numerics["series_interval_steps"]),
        runtime_stability_check_steps=int(numerics["device_sync_steps"]),
        ion_subcycle=int(numerics["ion_subcycle"]),
        device_sync_steps=int(numerics["device_sync_steps"]),
    )
    return config, spec


def series_to_arrays(series: list[SeriesRecord]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[float]] = {
        "step": [], "time_s": [], "electrons": [], "ions": [], "phi_mean_v": [], "phi_min_v": [], "phi_max_v": [],
        "kinetic_electron_j": [], "kinetic_ion_j": [], "field_energy_j": [], "surface_charge_c": [],
        "peak_omega_pe_dt": [], "poisson_iterations": [], "total_energy_j": [], "interval_residual_j": [],
        "interval_sources_j": [], "interval_electrode_work_j": [], "interval_field_work_j": [],
        "anode_induced_charge_c": [], "exit_induced_charge_c": [],
    }
    current_keys = sorted(series[0].currents_a) if series else []
    for key in current_keys:
        arrays[f"current_{key}"] = []
    for record in series:
        for key in ("step", "time_s", "electrons", "ions", "phi_mean_v", "phi_min_v", "phi_max_v", "kinetic_electron_j",
                    "kinetic_ion_j", "field_energy_j", "surface_charge_c", "peak_omega_pe_dt", "poisson_iterations"):
            arrays[key].append(float(getattr(record, key)))
        for key in ("total_energy_j", "interval_residual_j", "interval_sources_j", "interval_electrode_work_j",
                    "interval_field_work_j", "anode_induced_charge_c", "exit_induced_charge_c"):
            arrays[key].append(float(record.ledger[key]))
        for key in current_keys:
            arrays[f"current_{key}"].append(record.currents_a[key])
    return {key: np.asarray(values, dtype=np.float64) for key, values in arrays.items()}


def trailing_drift(steps: np.ndarray, values: np.ndarray, fraction: float) -> float | None:
    """Relative drift of a linear fit over the trailing ``fraction`` of the record."""
    if steps.size < 8:
        return None
    start = steps[-1] - fraction * (steps[-1] - steps[0])
    mask = steps >= start
    if mask.sum() < 8:
        return None
    x = steps[mask].astype(np.float64)
    y = values[mask].astype(np.float64)
    mean = float(np.mean(y))
    if not np.isfinite(mean) or abs(mean) < 1e-300:
        return None
    slope = float(np.polyfit(x - x[0], y, 1)[0])
    return slope * float(x[-1] - x[0]) / abs(mean)


def plateau_status(series: list[SeriesRecord], protocol: dict[str, Any]) -> dict[str, Any]:
    rule = protocol["stopping_rule"]
    steps = np.asarray([r.step for r in series], dtype=np.float64)
    i_d = np.asarray([r.currents_a["discharge_a"] for r in series])
    electrons = np.asarray([r.electrons for r in series], dtype=np.float64)
    drift_i = trailing_drift(steps, i_d, float(rule["plateau_window_fraction"]))
    drift_n = trailing_drift(steps, electrons, float(rule["plateau_window_fraction"]))
    threshold = float(rule["plateau_threshold"])
    reached = drift_i is not None and drift_n is not None and abs(drift_i) < threshold and abs(drift_n) < threshold
    return {
        "reached": bool(reached),
        "discharge_current_drift": drift_i,
        "electron_count_drift": drift_n,
        "threshold": threshold,
        "window_fraction": float(rule["plateau_window_fraction"]),
    }


def ledger_summary(series: list[SeriesRecord]) -> dict[str, Any]:
    if len(series) < 2:
        return {}
    residual = np.asarray([r.ledger["interval_residual_j"] for r in series[1:]])
    electrode = np.asarray([r.ledger["interval_electrode_work_j"] for r in series[1:]])
    sources = np.asarray([r.ledger["interval_sources_j"] for r in series[1:]])
    total = np.asarray([r.ledger["total_energy_j"] for r in series])
    change = float(total[-1] - total[0])
    gross = float(np.sum(np.abs(sources)))
    return {
        "cumulative_residual_j": float(residual.sum()),
        "cumulative_electrode_work_j": float(electrode.sum()),
        "cumulative_sources_j": float(sources.sum()),
        "total_energy_change_j": change,
        "gross_source_turnover_j": gross,
        "cumulative_residual_over_gross_turnover": float(residual.sum() / gross) if gross > 0 else None,
        "cumulative_residual_over_electrode_work": float(residual.sum() / electrode.sum()) if abs(electrode.sum()) > 0 else None,
        "interval_residual_rms_j": float(np.sqrt(np.mean(residual**2))),
        "final_total_energy_j": float(total[-1]),
    }


def run_case(case: str, *, backend: str, max_wall_seconds: float, max_steps: int | None, checkpoint_every: int,
             min_steps: int | None = None) -> Path:
    protocol = load_protocol()
    config, spec = build_config(protocol, case, backend=backend)
    target_steps = int(spec["target_steps"]) if max_steps is None else int(max_steps)
    minimum = int(spec["min_steps"]) if min_steps is None else int(min_steps)
    minimum = min(minimum, target_steps)
    window_steps = int(protocol["numerics"]["averaging_window_steps"])
    window_steps -= window_steps % config.sync_steps
    out = RESULTS / case
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    psi_field, evidence = build_p2_psi_field(REPOSITORY_ROOT, role="primary")
    field_map = sample_field_map(psi_field, config.grid, evidence)
    cross_sections = XenonCrossSections.from_file()
    field_seconds = time.perf_counter() - t0
    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend)  # a-priori stability gate inside
    print(f"[{case}] stability gate: {json.dumps(sim.stability.to_dict())}", flush=True)
    print(f"[{case}] mesh: {sim.masks.to_dict()}", flush=True)
    stop_reason = "target_steps_reached"
    t_run = time.perf_counter()
    step = 0
    last_print = time.perf_counter()
    gpu_samples: list[float | None] = []
    plateau: dict[str, Any] = {"reached": False}

    def progress(record: SeriesRecord) -> None:
        nonlocal last_print
        now = time.perf_counter()
        if now - last_print > 30.0:
            elapsed = now - t_run
            rate = record.step / max(elapsed, 1e-9)
            print(
                f"[{case}] step {record.step}/{target_steps} t={record.time_s*1e9:.0f} ns  e={record.electrons} i={record.ions} "
                f"phi[{record.phi_min_v:.0f},{record.phi_max_v:.0f}] V  I_d={record.currents_a['discharge_a']*1e3:.2f} mA "
                f"I_beam={record.currents_a['exit_ion_beam_a']*1e3:.2f} mA  w_pe*dt={record.peak_omega_pe_dt:.3f}  {rate:.0f} steps/s",
                flush=True,
            )
            last_print = now
            gpu_samples.append(_gpu_utilisation())

    chunk = config.sync_steps * 10
    window_start = 0
    completed_window: dict[str, np.ndarray] | None = None
    completed_window_range: tuple[int, int] | None = None
    gate_error: str | None = None
    while step < target_steps:
        this_chunk = min(chunk, target_steps - step, window_start + window_steps - step)
        try:
            sim.run(this_chunk, accumulate_from_step=window_start, progress=progress)
        except PIC2DStabilityError as error:
            gate_error = str(error)
            stop_reason = "runtime_stability_gate_stopped_run"
            step = sim.state.step
            print(f"[{case}] fail-closed stop at step {step}: {gate_error}", flush=True)
            break
        step += this_chunk
        elapsed = time.perf_counter() - t_run
        if step - window_start >= window_steps:
            completed_window = sim.diagnostic_arrays()
            completed_window_range = (window_start, step)
            sim.backend.reset_diagnostics()
            window_start = step
        if checkpoint_every and step % checkpoint_every == 0:
            artifacts.save_checkpoint(out, "checkpoint-latest", sim.state, config, field_sha256=field_map.sha256,
                                      cross_section_sha256=cross_sections.payload_sha256, backend=sim.backend.name)
        if step >= minimum and step % window_steps == 0:
            plateau = plateau_status(sim.series, protocol)
            print(f"[{case}] plateau check at step {step}: {json.dumps(plateau)}", flush=True)
            if plateau["reached"]:
                stop_reason = "plateau_reached_after_min_steps"
                break
        if elapsed > max_wall_seconds:
            stop_reason = "wall_clock_budget_reached"
            break
    wall = time.perf_counter() - t_run
    state = sim.state
    if not plateau.get("discharge_current_drift"):
        plateau = plateau_status(sim.series, protocol)
    partial = sim.diagnostic_arrays()
    if int(partial["window_steps"][0]) >= window_steps // 2 or completed_window is None:
        maps = partial
        window_range = (window_start, state.step)
    else:
        maps = completed_window
        window_range = completed_window_range
    series = series_to_arrays(sim.series)
    maps_sha = artifacts.write_npz(out / "maps.npz", maps)
    series_sha = artifacts.write_npz(out / "series.npz", series)
    checkpoint_json, checkpoint_npz = artifacts.save_checkpoint(
        out, "checkpoint-final", state, config, field_sha256=field_map.sha256,
        cross_section_sha256=cross_sections.payload_sha256, backend=sim.backend.name,
    )
    final = sim.series[-1]
    window = int(maps["window_steps"][0])
    plasma = sim.masks.plasma_node
    # window-averaged currents from the series over the map window
    steps_arr = series["step"]
    in_window = (steps_arr > window_range[0]) & (steps_arr <= window_range[1])
    window_currents = {
        key[len("current_"):]: float(np.mean(series[key][in_window])) if in_window.any() else None
        for key in series if key.startswith("current_")
    }
    summary = {
        "schema_version": "cft-revival.pic2d-cft-snapshot-v2.case-summary/0.1.0",
        "experiment_id": protocol["experiment_id"],
        "model_version": protocol["model_version"],
        "status": protocol["status"],
        "classification": protocol["classification"],
        "case": case,
        "case_spec": spec,
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "git_head": git_head(),
        "started_utc": started,
        "backend": sim.backend.name,
        "steps_completed": int(state.step),
        "min_steps": minimum,
        "target_steps": target_steps,
        "simulated_time_s": float(state.time_s),
        "ion_transit_times": float(state.time_s) / float(protocol["budget_v1_1"]["ion_transit_time_s"]),
        "stop_reason": stop_reason,
        "stability_gate_message": gate_error,
        "plateau": plateau,
        "averaging_window_step_range": list(window_range),
        "wall_seconds_run": wall,
        "wall_seconds_field_setup": field_seconds,
        "steps_per_second": state.step / max(wall, 1e-9),
        "ms_per_step": 1e3 * wall / max(state.step, 1),
        "gpu_utilisation_percent_samples": gpu_samples,
        "averaging_window_steps": window,
        "final_counts": {"electrons": final.electrons, "ions": final.ions},
        "peak_counts": {"electrons": int(series["electrons"].max()), "ions": int(series["ions"].max())},
        "final_series": final.to_dict(),
        "window_currents_a": window_currents,
        "ledger": ledger_summary(sim.series),
        "window_maps_summary": {
            "n_e_peak_per_m3": float(np.nanmax(maps["n_e_per_m3"][plasma])) if window else None,
            "n_e_mean_per_m3": float(np.nanmean(maps["n_e_per_m3"][plasma])) if window else None,
            "n_i_peak_per_m3": float(np.nanmax(maps["n_i_per_m3"][plasma])) if window else None,
            "phi_min_v": float(np.nanmin(maps["phi_v"][plasma])) if window else None,
            "phi_max_v": float(np.nanmax(maps["phi_v"][plasma])) if window else None,
            "t_e_max_ev": float(np.nanmax(maps["t_e_ev"][plasma])) if window else None,
            "t_e_density_weighted_mean_ev": (
                float(np.nansum(maps["t_e_ev"][plasma] * maps["n_e_per_m3"][plasma]) / max(np.nansum(maps["n_e_per_m3"][plasma]), 1e-300))
                if window else None
            ),
            "ionization_rate_peak_per_m3_s": float(np.nanmax(maps["ionization_rate_per_m3_s"][plasma])) if window else None,
            "wall_ion_flux_peak_per_m2_s": float(np.nanmax(maps["wall_ion_flux_per_m2_s"])) if window else None,
            "wall_electron_flux_peak_per_m2_s": float(np.nanmax(maps["wall_electron_flux_per_m2_s"])) if window else None,
            "exit_ion_current_a": float(np.sum(maps["exit_ion_current_density_a_per_m2"] * _exit_areas(config.grid))) if window else None,
        },
        "budget_check": {
            "n_e_peak_over_n_max": (float(np.nanmax(maps["n_e_per_m3"][plasma])) / float(protocol["budget_v1_1"]["n_max_per_m3"])) if window else None,
            "max_observed_omega_pe_dt": float(series["peak_omega_pe_dt"].max()),
        },
        "artifacts": {
            "maps_npz_sha256": maps_sha,
            "series_npz_sha256": series_sha,
            "checkpoint_json": checkpoint_json.name,
            "checkpoint_npz": checkpoint_npz.name,
        },
        "provenance": sim.to_provenance() | {"runtime": artifacts.runtime_identity(), "config_sha256": artifacts.config_identity(config)},
        "simplifications": protocol["simplifications"],
        "claim_boundary": (
            "development/screening PIC-MCC snapshot (model v1.1); not preregistered; not validated against experiment; "
            "not a thruster performance prediction; operating point scaled to a resolvable density"
        ),
    }
    artifacts.write_canonical_json(out / "summary.json", summary)
    print(f"[{case}] done: {state.step} steps in {wall:.0f} s ({stop_reason}); summary at {out / 'summary.json'}", flush=True)
    return out


def budget(protocol: dict[str, Any]) -> dict[str, Any]:
    """Recompute the v1.1 resolvability budget from the protocol numbers (a-priori, no run data)."""
    b = protocol["budget_v1_1"]
    numerics = protocol["numerics"]
    geometry = protocol["geometry"]
    n_max = float(b["n_max_per_m3"])
    t_e = float(b["t_e_for_lambda_d_ev"])
    dt = float(numerics["dt_s"])
    lambda_d = np.sqrt(EPSILON_0 * t_e / (n_max * ELEMENTARY_CHARGE_C))
    omega_pe = np.sqrt(n_max * ELEMENTARY_CHARGE_C**2 / (EPSILON_0 * ELECTRON_MASS_KG))
    omega_pi = omega_pe * np.sqrt(ELECTRON_MASS_KG / XE_MASS_KG)
    omega_ce = ELEMENTARY_CHARGE_C * float(b["b_max_t"]) / ELECTRON_MASS_KG
    rows = {}
    for case, spec in protocol["cases"].items():
        dr = geometry["bore_radius_m"] / spec["radial_cells"]
        dz = (geometry["z_max_m"] - geometry["z_min_m"]) / spec["axial_cells"]
        rows[case] = {
            "dr_m": dr, "dz_m": dz, "max_cell_over_lambda_d": max(dr, dz) / lambda_d,
            "macro_weight": spec["macro_weight"],
            "debye_ratio_limit": numerics["coarse_debye_ratio_limit"] if case.startswith("coarse") else numerics["stability_limits"]["max_cell_debye_ratio"],
        }
    return {
        "n_max_per_m3": n_max, "t_e_ev": t_e, "dt_s": dt,
        "lambda_d_min_m": float(lambda_d), "omega_pe_max_rad_per_s": float(omega_pe), "omega_pe_dt": float(omega_pe * dt),
        "omega_ce_dt": float(omega_ce * dt), "ion_subcycle": numerics["ion_subcycle"],
        "omega_pi_k_dt": float(omega_pi * numerics["ion_subcycle"] * dt),
        "cases": rows,
    }


def summarize() -> Path:
    protocol = load_protocol()
    cases: dict[str, Any] = {}
    for case in protocol["cases"]:
        path = RESULTS / case / "summary.json"
        if path.is_file():
            cases[case] = artifacts.read_canonical_json(path)
    convergence: dict[str, Any] = {}
    if cases:
        keys = ("n_e_peak_per_m3", "n_e_mean_per_m3", "phi_max_v", "phi_min_v", "t_e_density_weighted_mean_ev", "exit_ion_current_a")
        for key in keys:
            values = {case: summary["window_maps_summary"].get(key) for case, summary in cases.items()}
            finite = {k: v for k, v in values.items() if v is not None}
            spread = (max(finite.values()) - min(finite.values())) / max(abs(np.mean(list(finite.values()))), 1e-300) if len(finite) > 1 else None
            convergence[key] = {"values": values, "relative_spread": spread}
        for key in ("discharge_a", "exit_ion_beam_a", "wall_ion_a"):
            values = {case: summary["window_currents_a"].get(key) for case, summary in cases.items()}
            finite = {k: v for k, v in values.items() if v is not None}
            spread = (max(finite.values()) - min(finite.values())) / max(abs(np.mean(list(finite.values()))), 1e-300) if len(finite) > 1 else None
            convergence[f"window_{key}"] = {"values": values, "relative_spread": spread}
    manifest = {
        "schema_version": "cft-revival.pic2d-cft-snapshot-v2.manifest/0.1.0",
        "experiment_id": protocol["experiment_id"],
        "model_version": protocol["model_version"],
        "status": protocol["status"],
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "budget_v1_1": budget(protocol),
        "cases": {
            case: {
                "summary_sha256": _file_sha256(RESULTS / case / "summary.json"),
                "steps_completed": summary["steps_completed"],
                "simulated_time_s": summary["simulated_time_s"],
                "ion_transit_times": summary["ion_transit_times"],
                "stop_reason": summary["stop_reason"],
                "plateau": summary["plateau"],
                "wall_seconds_run": summary["wall_seconds_run"],
                "ms_per_step": summary["ms_per_step"],
                "ledger": summary["ledger"],
                "maps_npz_sha256": summary["artifacts"]["maps_npz_sha256"],
                "series_npz_sha256": summary["artifacts"]["series_npz_sha256"],
            }
            for case, summary in cases.items()
        },
        "convergence": convergence,
        "claim_boundary": protocol["classification"],
        "simplifications": protocol["simplifications"],
    }
    target = RESULTS / "manifest.json"
    artifacts.write_canonical_json(target, manifest)
    print(json.dumps(manifest["convergence"], indent=1))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    case_parser = sub.add_parser("case")
    case_parser.add_argument("name")
    case_parser.add_argument("--backend", default="warp-cuda")
    case_parser.add_argument("--max-wall-seconds", type=float, default=7200.0)
    case_parser.add_argument("--max-steps", type=int, default=None)
    case_parser.add_argument("--min-steps", type=int, default=None)
    case_parser.add_argument("--checkpoint-every", type=int, default=100000)
    sub.add_parser("summarize")
    sub.add_parser("budget")
    args = parser.parse_args()
    if args.command == "case":
        run_case(args.name, backend=args.backend, max_wall_seconds=args.max_wall_seconds, max_steps=args.max_steps,
                 checkpoint_every=args.checkpoint_every, min_steps=args.min_steps)
    elif args.command == "budget":
        print(json.dumps(budget(load_protocol()), indent=1))
    else:
        summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
