"""Run one snapshot case or summarise all cases.

From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_snapshot_v1.run case coarse-w1e5 --max-wall-seconds 1800
    python -m experiments.pic2d_cft_snapshot_v1.run summarize

Every case writes canonical JSON (+ ``.sha256.json`` sidecars) and npz arrays
under ``results/<case>/``.  This is a development/screening experiment: it is
not preregistered and makes no validated physics claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import build_p2_psi_field, sample_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.models import BoundaryPotentials, ChannelGeometry, Grid2D, PIC2DStabilityError, StabilityLimits
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation, SeriesRecord

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
PROTOCOL_PATH = HERE / "protocol.json"
RESULTS = HERE / "results"


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def build_config(protocol: dict[str, Any], case: str) -> tuple[PIC2DConfig, dict[str, Any]]:
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
    limits = StabilityLimits(**numerics["stability_limits"])
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
        limits=limits,
        reference_density_per_m3=numerics["stability_reference"]["density_per_m3"],
        reference_electron_temperature_ev=numerics["stability_reference"]["electron_temperature_ev"],
        max_electron_energy_ev=numerics["stability_reference"]["max_electron_energy_ev"],
        series_interval_steps=int(numerics["series_interval_steps"]),
    )
    return config, spec


def series_to_arrays(series: list[SeriesRecord]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[float]] = {
        "step": [], "time_s": [], "electrons": [], "ions": [], "phi_mean_v": [], "phi_min_v": [], "phi_max_v": [],
        "kinetic_electron_j": [], "kinetic_ion_j": [], "field_energy_j": [], "surface_charge_c": [],
        "peak_omega_pe_dt": [], "poisson_iterations": [], "total_energy_j": [], "interval_residual_j": [],
        "interval_sources_j": [],
    }
    current_keys = sorted(series[0].currents_a) if series else []
    for key in current_keys:
        arrays[f"current_{key}"] = []
    for record in series:
        for key in ("step", "time_s", "electrons", "ions", "phi_mean_v", "phi_min_v", "phi_max_v", "kinetic_electron_j",
                    "kinetic_ion_j", "field_energy_j", "surface_charge_c", "peak_omega_pe_dt", "poisson_iterations"):
            arrays[key].append(float(getattr(record, key)))
        arrays["total_energy_j"].append(record.ledger["total_energy_j"])
        arrays["interval_residual_j"].append(record.ledger["interval_residual_j"])
        arrays["interval_sources_j"].append(record.ledger["interval_sources_j"])
        for key in current_keys:
            arrays[f"current_{key}"].append(record.currents_a[key])
    return {key: np.asarray(values, dtype=np.float64) for key, values in arrays.items()}


def run_case(case: str, *, backend: str, max_wall_seconds: float, max_steps: int | None, checkpoint_every: int) -> Path:
    protocol = load_protocol()
    config, spec = build_config(protocol, case)
    target_steps = int(spec["target_steps"]) if max_steps is None else int(max_steps)
    window_steps = max(int(target_steps * float(protocol["numerics"]["averaging_window_fraction"])), config.series_interval_steps)
    window_steps -= window_steps % config.series_interval_steps
    out = RESULTS / case
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    psi_field, evidence = build_p2_psi_field(REPOSITORY_ROOT, role="primary")
    field_map = sample_field_map(psi_field, config.grid, evidence)
    cross_sections = XenonCrossSections.from_file()
    field_seconds = time.perf_counter() - t0
    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend)  # stability gate inside
    print(f"[{case}] stability gate: {json.dumps(sim.stability.to_dict())}", flush=True)
    print(f"[{case}] mesh: {sim.masks.to_dict()}", flush=True)
    stop_reason = "target_steps_reached"
    t_run = time.perf_counter()
    step = 0
    last_print = time.perf_counter()
    gpu_samples: list[float] = []

    def progress(record: SeriesRecord) -> None:
        nonlocal last_print
        now = time.perf_counter()
        if now - last_print > 20.0:
            elapsed = now - t_run
            rate = record.step / max(elapsed, 1e-9)
            print(
                f"[{case}] step {record.step}/{target_steps} t={record.time_s*1e9:.1f} ns  e={record.electrons} i={record.ions} "
                f"phi[{record.phi_min_v:.0f},{record.phi_max_v:.0f}] V  I_d={record.currents_a['discharge_a']*1e3:.1f} mA "
                f"I_beam={record.currents_a['exit_ion_beam_a']*1e3:.1f} mA  w_pe*dt={record.peak_omega_pe_dt:.3f}  {rate:.0f} steps/s",
                flush=True,
            )
            last_print = now
            gpu_samples.append(_gpu_utilisation())

    # Time averaging uses consecutive windows of `window_steps`; the reported maps are
    # the last window that completed (or the current one if it is at least half full),
    # so a wall-clock stop or a fail-closed stability stop still leaves a valid window.
    chunk = config.series_interval_steps * 5
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
        if elapsed > max_wall_seconds:
            stop_reason = "wall_clock_budget_reached"
            break
    wall = time.perf_counter() - t_run
    state = sim.state
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
    summary = {
        "schema_version": "cft-revival.pic2d-cft-snapshot-v1.case-summary/0.1.0",
        "experiment_id": protocol["experiment_id"],
        "status": protocol["status"],
        "classification": protocol["classification"],
        "case": case,
        "case_spec": spec,
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "git_head": git_head(),
        "started_utc": started,
        "backend": sim.backend.name,
        "steps_completed": int(state.step),
        "target_steps": target_steps,
        "simulated_time_s": float(state.time_s),
        "stop_reason": stop_reason,
        "stability_gate_message": gate_error,
        "averaging_window_step_range": list(window_range),
        "wall_seconds_run": wall,
        "wall_seconds_field_setup": field_seconds,
        "steps_per_second": state.step / max(wall, 1e-9),
        "gpu_utilisation_percent_samples": gpu_samples,
        "averaging_window_steps": window,
        "final_counts": {"electrons": final.electrons, "ions": final.ions},
        "final_series": final.to_dict(),
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
        "artifacts": {
            "maps_npz_sha256": maps_sha,
            "series_npz_sha256": series_sha,
            "checkpoint_json": checkpoint_json.name,
            "checkpoint_npz": checkpoint_npz.name,
        },
        "provenance": sim.to_provenance() | {"runtime": artifacts.runtime_identity(), "config_sha256": artifacts.config_identity(config)},
        "simplifications": protocol["simplifications"],
        "claim_boundary": (
            "development/screening PIC-MCC snapshot; not preregistered; not validated against experiment; "
            "not a thruster performance prediction"
        ),
    }
    artifacts.write_canonical_json(out / "summary.json", summary)
    print(f"[{case}] done: {state.step} steps in {wall:.0f} s ({stop_reason}); summary at {out / 'summary.json'}", flush=True)
    return out


def _exit_areas(grid: Grid2D) -> np.ndarray:
    r = grid.r_m
    return np.pi * (r[1:] ** 2 - r[:-1] ** 2)


def _file_sha256(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def _gpu_utilisation() -> float:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5
        )
        return float(completed.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def summarize() -> Path:
    protocol = load_protocol()
    cases: dict[str, Any] = {}
    for case in protocol["cases"]:
        path = RESULTS / case / "summary.json"
        if path.is_file():
            cases[case] = artifacts.read_canonical_json(path)
    convergence: dict[str, Any] = {}
    if cases:
        keys = ("n_e_peak_per_m3", "n_e_mean_per_m3", "phi_max_v", "t_e_density_weighted_mean_ev", "exit_ion_current_a")
        for key in keys:
            values = {case: summary["window_maps_summary"].get(key) for case, summary in cases.items()}
            finite = {k: v for k, v in values.items() if v is not None}
            spread = (max(finite.values()) - min(finite.values())) / max(abs(np.mean(list(finite.values()))), 1e-300) if len(finite) > 1 else None
            convergence[key] = {"values": values, "relative_spread": spread}
        discharge = {case: summary["final_series"]["currents_a"]["discharge_a"] for case, summary in cases.items()}
        convergence["discharge_current_a"] = {"values": discharge}
    manifest = {
        "schema_version": "cft-revival.pic2d-cft-snapshot-v1.manifest/0.1.0",
        "experiment_id": protocol["experiment_id"],
        "status": protocol["status"],
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "cases": {
            case: {
                "summary_sha256": _file_sha256(RESULTS / case / "summary.json"),
                "steps_completed": summary["steps_completed"],
                "simulated_time_s": summary["simulated_time_s"],
                "stop_reason": summary["stop_reason"],
                "wall_seconds_run": summary["wall_seconds_run"],
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
    case_parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    case_parser.add_argument("--max-steps", type=int, default=None)
    case_parser.add_argument("--checkpoint-every", type=int, default=20000)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "case":
        run_case(args.name, backend=args.backend, max_wall_seconds=args.max_wall_seconds, max_steps=args.max_steps, checkpoint_every=args.checkpoint_every)
    else:
        summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
