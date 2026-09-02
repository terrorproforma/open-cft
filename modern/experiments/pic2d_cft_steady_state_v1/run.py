"""Detached, checkpointed, resumable steady-state run (model v1.2).

From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_steady_state_v1.run run          # start, or resume if a checkpoint exists
    python -m experiments.pic2d_cft_steady_state_v1.run status       # last status line + projections

Detached launch (PowerShell, from ``modern/``)::

    Start-Process python -ArgumentList "-m","experiments.pic2d_cft_steady_state_v1.run","run" `
        -WindowStyle Hidden -RedirectStandardOutput results\\run.log -RedirectStandardError results\\run.err

The run writes, under ``results/``:

* ``status.jsonl`` - one machine-readable line per 200-step diagnostic sync
  (t, steps, N_e, N_i, I_d, I_beam,i, peak-node / mean n_e, <T_e>, max omega_pe dt,
  cumulative wall time, ms/step, latest plateau evaluation);
* ``series.jsonl`` - the full series record per sync (the source of ``series.npz``);
* ``checkpoint/checkpoint-latest.{json,npz}`` - rewritten atomically every
  ``checkpoint_every_steps`` (bitwise-resumable dynamical state);
* ``run_state.json`` - cumulative wall time, sessions, last checkpoint step;
* ``run.pid`` - PID of the running process;
* on any stop: ``summary.json``, ``series.npz``, ``maps.npz``, ``checkpoint-final.*``.

Stop conditions: plateau (relative drift of I_d and N_e < 5 % over the trailing
20 % of elapsed simulated time, only after >= 3 ion transit times), the cumulative
wall budget, the fail-closed stability gate, or an explicit ``--max-steps``.
Every stop exits 0 with the artifacts written.  Development/screening run: not
preregistered, no validated physics claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

import numpy as np

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import MagneticFieldMap, build_p2_psi_field, sample_field_map
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.models import (
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DStabilityError,
    PIC2DValidationError,
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

SERIES_SCALARS = (
    "step", "time_s", "electrons", "ions", "phi_mean_v", "phi_min_v", "phi_max_v", "kinetic_electron_j",
    "kinetic_ion_j", "field_energy_j", "surface_charge_c", "peak_omega_pe_dt", "poisson_iterations",
)
LEDGER_SCALARS = (
    "total_energy_j", "interval_residual_j", "interval_sources_j", "interval_electrode_work_j",
    "interval_field_work_j", "anode_induced_charge_c", "exit_induced_charge_c",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_config(protocol: dict[str, Any], *, backend: str = "warp-cuda") -> PIC2DConfig:
    geometry = protocol["geometry"]
    case = protocol["case"]
    operating = protocol["operating_point"]
    numerics = protocol["numerics"]
    grid = Grid2D(
        ChannelGeometry(
            geometry["bore_radius_m"], geometry["z_min_m"], geometry["z_max_m"],
            geometry["cone_start_z_m"], geometry["exit_radius_m"],
        ),
        int(case["radial_cells"]), int(case["axial_cells"]),
    )
    sync = int(numerics["device_sync_steps"])
    checkpoint_every = int(numerics["checkpoint_every_steps"])
    window = int(numerics["averaging_window_steps"])
    if checkpoint_every % sync != 0 or window % checkpoint_every != 0:
        raise PIC2DValidationError("checkpoint_every_steps must be a multiple of device_sync_steps and divide averaging_window_steps")
    mcc = MCCConfig(operating["neutral_density_per_m3"], operating["neutral_temperature_k"]) if operating["neutral_density_per_m3"] > 0 else None
    return PIC2DConfig(
        grid=grid,
        potentials=BoundaryPotentials(operating["anode_potential_v"], operating["exit_plane_potential_v"]),
        dt_s=float(numerics["dt_s"]),
        macro_weight=float(case["macro_weight"]),
        seed=int(case.get("seed", 20260903)),
        injection=InjectionConfig(operating["electron_injection_current_a"], operating["electron_injection_temperature_ev"]),
        seed_plasma=SeedPlasmaConfig(
            operating["seed_plasma_density_per_m3"], operating["seed_electron_temperature_ev"], operating["seed_ion_temperature_ev"]
        ),
        mcc=mcc,
        poisson=PoissonConfig2D(method="device-direct" if backend == "warp-cuda" else "direct", relative_tolerance=1.0e-10),
        limits=StabilityLimits(**numerics["stability_limits"]),
        reference_density_per_m3=numerics["stability_reference"]["density_per_m3"],
        reference_electron_temperature_ev=numerics["stability_reference"]["electron_temperature_ev"],
        max_electron_energy_ev=numerics["stability_reference"]["max_electron_energy_ev"],
        series_interval_steps=int(numerics["series_interval_steps"]),
        runtime_stability_check_steps=sync,
        ion_subcycle=int(numerics["ion_subcycle"]),
        device_sync_steps=sync,
    )


# -- plateau criterion ------------------------------------------------------

def trailing_time_drift(time_s: np.ndarray, values: np.ndarray, fraction: float) -> float | None:
    """Relative drift of a linear fit over the trailing ``fraction`` of the elapsed time.

    drift = slope * window / |mean|; ``None`` if fewer than 8 samples fall in the window
    or the mean is not usable.
    """

    if time_s.size < 8:
        return None
    t_end = float(time_s[-1])
    start = t_end - fraction * t_end
    mask = time_s >= start
    if int(mask.sum()) < 8:
        return None
    x = time_s[mask].astype(np.float64)
    y = values[mask].astype(np.float64)
    mean = float(np.mean(y))
    if not np.isfinite(mean) or abs(mean) < 1e-300:
        return None
    slope = float(np.polyfit(x - x[0], y, 1)[0])
    return slope * float(x[-1] - x[0]) / abs(mean)


def evaluate_plateau(
    time_s: np.ndarray, discharge_a: np.ndarray, electrons: np.ndarray, rule: dict[str, Any], transit_time_s: float
) -> dict[str, Any]:
    """The stopping rule: both drifts below the threshold AND >= min_transit_times elapsed."""

    fraction = float(rule["plateau_window_fraction"])
    threshold = float(rule["plateau_threshold"])
    min_transits = float(rule["min_transit_times"])
    elapsed = float(time_s[-1]) if time_s.size else 0.0
    transits = elapsed / transit_time_s
    drift_i = trailing_time_drift(time_s, discharge_a, fraction)
    drift_n = trailing_time_drift(time_s, electrons, fraction)
    drifts_ok = drift_i is not None and drift_n is not None and abs(drift_i) < threshold and abs(drift_n) < threshold
    return {
        "reached": bool(drifts_ok and transits >= min_transits),
        "drifts_within_threshold": bool(drifts_ok),
        "transit_times_elapsed": transits,
        "min_transit_times": min_transits,
        "discharge_current_drift": drift_i,
        "electron_count_drift": drift_n,
        "threshold": threshold,
        "window_fraction": fraction,
    }


# -- records ----------------------------------------------------------------

def records_to_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[float]] = {key: [] for key in SERIES_SCALARS + LEDGER_SCALARS}
    current_keys = sorted(records[0]["currents_a"]) if records else []
    for key in current_keys:
        arrays[f"current_{key}"] = []
    for record in records:
        for key in SERIES_SCALARS:
            arrays[key].append(float(record[key]))
        for key in LEDGER_SCALARS:
            arrays[key].append(float(record["ledger"][key]))
        for key in current_keys:
            arrays[f"current_{key}"].append(float(record["currents_a"][key]))
    return {key: np.asarray(values, dtype=np.float64) for key, values in arrays.items()}


def status_from_record(
    record: dict[str, Any], config: PIC2DConfig, plasma_volume_m3: float, *, wall_seconds_total: float,
    ms_per_step: float | None, plateau: dict[str, Any] | None,
) -> dict[str, Any]:
    n_e = record["electrons"] * config.macro_weight
    omega = record["peak_omega_pe_dt"] / config.dt_s
    peak_node = omega * omega * EPSILON_0 * ELECTRON_MASS_KG / ELEMENTARY_CHARGE_C**2
    t_e = (2.0 / 3.0) * record["kinetic_electron_j"] / (max(n_e, 1.0) * ELEMENTARY_CHARGE_C)
    return {
        "step": record["step"],
        "time_s": record["time_s"],
        "electrons": record["electrons"],
        "ions": record["ions"],
        "discharge_a": record["currents_a"]["discharge_a"],
        "exit_ion_beam_a": record["currents_a"]["exit_ion_beam_a"],
        "n_e_peak_node_per_m3": peak_node,
        "n_e_mean_per_m3": n_e / plasma_volume_m3,
        "t_e_mean_ev": t_e,
        "omega_pe_dt_max": record["peak_omega_pe_dt"],
        "phi_max_v": record["phi_max_v"],
        "wall_seconds_total": wall_seconds_total,
        "ms_per_step": ms_per_step,
        "plateau": None if plateau is None else {
            "reached": plateau["reached"],
            "discharge_current_drift": plateau["discharge_current_drift"],
            "electron_count_drift": plateau["electron_count_drift"],
            "transit_times_elapsed": plateau["transit_times_elapsed"],
        },
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    # append-only logs (not canonical artifacts): a NaN in a diagnostic must not end a 12 h run
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


# -- checkpointing ----------------------------------------------------------

CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_NAME = "checkpoint-latest"


def save_checkpoint_atomic(results: Path, sim: Simulation, config: PIC2DConfig, field_sha256: str, xs_sha256: str | None) -> Path:
    """Write the checkpoint into a fresh directory, then swap it in (old copy kept until the swap is done)."""

    tmp = results / f"{CHECKPOINT_DIR}-tmp"
    old = results / f"{CHECKPOINT_DIR}-old"
    live = results / CHECKPOINT_DIR
    for stale in (tmp, old):
        if stale.exists():
            shutil.rmtree(stale)
    tmp.mkdir(parents=True)
    artifacts.save_checkpoint(tmp, CHECKPOINT_NAME, sim.state, config, field_sha256=field_sha256,
                              cross_section_sha256=xs_sha256, backend=sim.backend.name)
    if live.exists():
        live.rename(old)
    tmp.rename(live)
    if old.exists():
        shutil.rmtree(old)
    return live / f"{CHECKPOINT_NAME}.json"


def find_checkpoint(results: Path) -> Path | None:
    for name in (CHECKPOINT_DIR, f"{CHECKPOINT_DIR}-old"):
        candidate = results / name / f"{CHECKPOINT_NAME}.json"
        if candidate.is_file():
            return candidate
    return None


# -- the run ------------------------------------------------------------------

def run_steady_state(
    protocol: dict[str, Any],
    results: Path,
    *,
    backend: str = "warp-cuda",
    field_map: MagneticFieldMap | None = None,
    cross_sections: XenonCrossSections | None = None,
    max_steps: int | None = None,
    wall_budget_seconds: float | None = None,
    require_same_code: bool = True,
    log: Callable[[str], None] = lambda text: print(text, flush=True),
) -> Path:
    """Start or resume the run; returns the path of ``summary.json`` when the run stops."""

    config = build_config(protocol, backend=backend)
    numerics = protocol["numerics"]
    rule = protocol["stopping_rule"]
    budget = protocol["budget_v1_2"]
    transit_time = float(budget["ion_transit_time_s"])
    checkpoint_every = int(numerics["checkpoint_every_steps"])
    window_steps = int(numerics["averaging_window_steps"])
    wall_budget = float(rule["wall_budget_seconds"]) if wall_budget_seconds is None else float(wall_budget_seconds)
    results.mkdir(parents=True, exist_ok=True)
    (results / "run.pid").write_text(f"{os.getpid()}\n", encoding="utf-8", newline="\n")

    t0 = time.perf_counter()
    if field_map is None:
        psi_field, evidence = build_p2_psi_field(REPOSITORY_ROOT, role="primary")
        field_map = sample_field_map(psi_field, config.grid, evidence)
    if cross_sections is None and config.mcc is not None:
        cross_sections = XenonCrossSections.from_file()
    xs_sha = cross_sections.payload_sha256 if cross_sections is not None else None
    setup_seconds = time.perf_counter() - t0

    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend)  # a-priori gate inside
    log(f"[steady-state] stability gate: {json.dumps(sim.stability.to_dict())}")
    log(f"[steady-state] mesh: {sim.masks.to_dict()}")
    plasma_volume = float(sim.masks.to_dict()["plasma_volume_m3"])

    series_path = results / "series.jsonl"
    status_path = results / "status.jsonl"
    state_path = results / "run_state.json"
    run_state: dict[str, Any] = {"wall_seconds_total": 0.0, "sessions": [], "checkpoint_step": 0, "finished": False}
    records: list[dict[str, Any]] = []
    checkpoint = find_checkpoint(results)
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "resumed_from_step": 0, "pid": os.getpid()}
    if checkpoint is not None:
        state = artifacts.load_checkpoint(checkpoint, config, field_sha256=field_map.sha256, cross_section_sha256=xs_sha,
                                          require_same_code=require_same_code)
        sim.load_state(state)
        if state_path.is_file():
            run_state = json.loads(state_path.read_text(encoding="utf-8"))
        records = [r for r in _read_jsonl(series_path) if r["step"] <= state.step]
        _write_jsonl(series_path, records)  # drop records past the checkpoint (process died between sync and checkpoint)
        session["resumed_from_step"] = int(state.step)
        _append_jsonl(status_path, {"event": "resume", "step": int(state.step), "time_s": float(state.time_s), "utc": session["started_utc"]})
        log(f"[steady-state] resumed from step {state.step} (t = {state.time_s*1e9:.1f} ns), {len(records)} series records kept")
    run_state["sessions"].append(session)
    wall_before = float(run_state["wall_seconds_total"])

    stop_reason = "target_steps_reached"
    gate_error: str | None = None
    t_session = time.perf_counter()
    last_status_wall = t_session
    last_status_step = sim.backend.step_index
    last_print = t_session
    last_plateau: dict[str, Any] | None = None
    gpu_samples: list[float] = []

    def wall_total() -> float:
        return wall_before + (time.perf_counter() - t_session)

    def progress(record: SeriesRecord) -> None:
        nonlocal last_status_wall, last_status_step, last_print
        payload = record.to_dict()
        records.append(payload)
        _append_jsonl(series_path, payload)
        now = time.perf_counter()
        ms = 1e3 * (now - last_status_wall) / max(record.step - last_status_step, 1)
        last_status_wall, last_status_step = now, record.step
        _append_jsonl(status_path, status_from_record(payload, config, plasma_volume, wall_seconds_total=wall_total(),
                                                      ms_per_step=ms, plateau=last_plateau))
        if now - last_print > 60.0:
            last_print = now
            gpu_samples.append(_gpu_utilisation())
            log(f"[steady-state] step {record.step} t={record.time_s*1e6:.3f} us e={record.electrons} i={record.ions} "
                f"I_d={record.currents_a['discharge_a']*1e3:.2f} mA I_beam={record.currents_a['exit_ion_beam_a']*1e3:.2f} mA "
                f"w_pe*dt={record.peak_omega_pe_dt:.3f} {ms:.2f} ms/step wall={wall_total()/3600:.2f} h")

    step = sim.backend.step_index
    window_start = step
    completed_window: dict[str, np.ndarray] | None = None
    completed_range: tuple[int, int] | None = None
    while True:
        chunk = min(checkpoint_every, window_start + window_steps - step)
        if max_steps is not None:
            chunk = min(chunk, max_steps - step)
        if chunk <= 0:
            stop_reason = "target_steps_reached"
            break
        try:
            sim.run(chunk, accumulate_from_step=window_start, progress=progress)
        except PIC2DStabilityError as error:
            gate_error = str(error)
            stop_reason = "runtime_stability_gate_stopped_run"
            log(f"[steady-state] fail-closed stop at step {sim.backend.step_index}: {gate_error}")
            break
        step = sim.backend.step_index
        if step - window_start >= window_steps:
            completed_window = sim.diagnostic_arrays()
            completed_range = (window_start, step)
            sim.backend.reset_diagnostics()
            window_start = step
        # a chunk is at most checkpoint_every steps: checkpoint after every chunk
        save_checkpoint_atomic(results, sim, config, field_map.sha256, xs_sha)
        run_state.update({"wall_seconds_total": wall_total(), "checkpoint_step": step, "checkpoint_time_s": sim.state.time_s})
        artifacts.write_canonical_json(state_path, run_state)
        arrays = records_to_arrays(records)
        last_plateau = evaluate_plateau(arrays["time_s"], arrays["current_discharge_a"], arrays["electrons"], rule, transit_time)
        if last_plateau["reached"]:
            stop_reason = "plateau_reached_after_min_transit_times"
            break
        if wall_total() > wall_budget:
            stop_reason = "wall_clock_budget_reached"
            break
        if max_steps is not None and step >= max_steps:
            stop_reason = "target_steps_reached"
            break

    state = sim.state
    wall_session = time.perf_counter() - t_session
    run_state.update({"wall_seconds_total": wall_before + wall_session, "finished": True, "stop_reason": stop_reason})
    arrays = records_to_arrays(records) if records else {}
    plateau = evaluate_plateau(arrays["time_s"], arrays["current_discharge_a"], arrays["electrons"], rule, transit_time) if records else None
    partial = sim.diagnostic_arrays()
    if int(partial["window_steps"][0]) >= window_steps // 2 or completed_window is None:
        maps, window_range = partial, (window_start, state.step)
    else:
        maps, window_range = completed_window, completed_range  # type: ignore[assignment]
    maps_sha = artifacts.write_npz(results / "maps.npz", maps)
    series_sha = artifacts.write_npz(results / "series.npz", arrays) if arrays else None
    checkpoint_json, checkpoint_npz = artifacts.save_checkpoint(
        results, "checkpoint-final", state, config, field_sha256=field_map.sha256, cross_section_sha256=xs_sha, backend=sim.backend.name
    )
    window = int(maps["window_steps"][0])
    plasma = sim.masks.plasma_node
    window_currents: dict[str, float | None] = {}
    if arrays:
        steps_arr = arrays["step"]
        in_window = (steps_arr > window_range[0]) & (steps_arr <= window_range[1])
        window_currents = {
            key[len("current_"):]: float(np.mean(arrays[key][in_window])) if in_window.any() else None
            for key in arrays if key.startswith("current_")
        }
    ledger: dict[str, Any] = {}
    if len(records) >= 2:
        residual = arrays["interval_residual_j"][1:]
        electrode = arrays["interval_electrode_work_j"][1:]
        sources = arrays["interval_sources_j"][1:]
        total = arrays["total_energy_j"]
        gross = float(np.sum(np.abs(sources)))
        ledger = {
            "cumulative_residual_j": float(residual.sum()),
            "cumulative_electrode_work_j": float(electrode.sum()),
            "total_energy_change_j": float(total[-1] - total[0]),
            "gross_source_turnover_j": gross,
            "cumulative_residual_over_electrode_work": float(residual.sum() / electrode.sum()) if abs(electrode.sum()) > 0 else None,
            "interval_residual_rms_j": float(np.sqrt(np.mean(residual**2))),
            "note": "intervals restart at every resume (the first record of a session has zero residual and electrode work)",
        }

    def stat(name: str, fn: Callable[[np.ndarray], float]) -> float | None:
        return _finite(fn(maps[name][plasma])) if window else None

    summary = {
        "schema_version": "cft-revival.pic2d-cft-steady-state-v1.summary/0.1.0",
        "experiment_id": protocol["experiment_id"],
        "model_version": protocol["model_version"],
        "status": protocol["status"],
        "classification": protocol["classification"],
        "case": protocol["case"],
        "protocol_sha256": _file_sha256(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None,
        "git_head": git_head(),
        "backend": sim.backend.name,
        "steps_completed": int(state.step),
        "simulated_time_s": float(state.time_s),
        "ion_transit_times": float(state.time_s) / transit_time,
        "stop_reason": stop_reason,
        "stability_gate_message": gate_error,
        "plateau": plateau,
        "sessions": run_state["sessions"],
        "wall_seconds_total": run_state["wall_seconds_total"],
        "wall_seconds_setup_this_session": setup_seconds,
        "ms_per_step_this_session": 1e3 * wall_session / max(state.step - session["resumed_from_step"], 1),
        "gpu_utilisation_percent_samples": gpu_samples,
        "averaging_window_steps": window,
        "averaging_window_step_range": list(window_range),
        "final_counts": {"electrons": state.electrons.count, "ions": state.ions.count},
        "peak_counts": {"electrons": int(arrays["electrons"].max()), "ions": int(arrays["ions"].max())} if arrays else None,
        "final_series": records[-1] if records else None,
        "window_currents_a": window_currents,
        "ledger": ledger,
        "window_maps_summary": {
            "n_e_peak_per_m3": stat("n_e_per_m3", np.nanmax),
            "n_e_mean_per_m3": stat("n_e_per_m3", np.nanmean),
            "n_i_peak_per_m3": stat("n_i_per_m3", np.nanmax),
            "phi_min_v": stat("phi_v", np.nanmin),
            "phi_max_v": stat("phi_v", np.nanmax),
            "t_e_max_ev": stat("t_e_ev", np.nanmax),
            "t_e_density_weighted_mean_ev": (
                _finite(np.nansum(maps["t_e_ev"][plasma] * maps["n_e_per_m3"][plasma]) / max(np.nansum(maps["n_e_per_m3"][plasma]), 1e-300))
                if window else None
            ),
            "ionization_rate_peak_per_m3_s": stat("ionization_rate_per_m3_s", np.nanmax),
            "wall_ion_flux_peak_per_m2_s": _finite(np.nanmax(maps["wall_ion_flux_per_m2_s"])) if window else None,
            "exit_ion_current_a": _finite(np.sum(maps["exit_ion_current_density_a_per_m2"] * _exit_areas(config.grid))) if window else None,
        },
        "budget_check": {
            "n_e_peak_over_n_max": (stat("n_e_per_m3", np.nanmax) or 0.0) / float(budget["n_max_per_m3"]) if window else None,
            "n_e_mean_over_projected_n_eq": (stat("n_e_per_m3", np.nanmean) or 0.0) / float(budget["n_eq_projected_per_m3"]) if window else None,
            "max_observed_omega_pe_dt": float(arrays["peak_omega_pe_dt"].max()) if arrays else None,
        },
        "artifacts": {
            "maps_npz_sha256": maps_sha,
            "series_npz_sha256": series_sha,
            "checkpoint_json": checkpoint_json.name,
            "checkpoint_npz": checkpoint_npz.name,
            "status_jsonl": status_path.name,
            "series_jsonl": series_path.name,
        },
        "provenance": sim.to_provenance() | {"runtime": artifacts.runtime_identity(), "config_sha256": artifacts.config_identity(config)},
        "simplifications": protocol["simplifications"],
        "claim_boundary": (
            "development/screening PIC-MCC steady-state run (model v1.2); not preregistered; not validated against experiment; "
            "not a thruster performance prediction; sub-critical operating point sized from measured kinetics"
        ),
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    artifacts.write_canonical_json(state_path, run_state)
    _append_jsonl(status_path, {"event": "stop", "step": int(state.step), "time_s": float(state.time_s), "stop_reason": stop_reason})
    log(f"[steady-state] done: {state.step} steps, t = {state.time_s*1e6:.3f} us, {stop_reason}; summary at {results / 'summary.json'}")
    return results / "summary.json"


# -- status -----------------------------------------------------------------

def status(results: Path = RESULTS, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or load_protocol()
    lines = _read_jsonl(results / "status.jsonl")
    samples = [line for line in lines if "event" not in line]
    if not samples:
        return {"status": "no samples yet"}
    last = samples[-1]
    transit = float(protocol["budget_v1_2"]["ion_transit_time_s"])
    steps_per_transit = transit / float(protocol["numerics"]["dt_s"])
    recent = samples[-50:]
    ms = float(np.nanmean([s["ms_per_step"] for s in recent if s["ms_per_step"] is not None]))
    remaining = {f"{k}_transit_times": {"steps": int(k * steps_per_transit), "hours_from_now": (k * steps_per_transit - last["step"]) * ms / 3.6e6}
                 for k in (3, 5, 10)}
    pid_file = results / "run.pid"
    return {
        "last": last,
        "samples": len(samples),
        "recent_ms_per_step": ms,
        "transit_times_elapsed": last["time_s"] / transit,
        "projection": remaining,
        "pid": int(pid_file.read_text().strip()) if pid_file.is_file() else None,
        "finished": (results / "summary.json").is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--backend", default="warp-cuda")
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--wall-budget-seconds", type=float, default=None)
    run_parser.add_argument("--ignore-code-identity", action="store_true", help="resume even if the package code hash changed")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "run":
        run_steady_state(load_protocol(), RESULTS, backend=args.backend, max_steps=args.max_steps,
                         wall_budget_seconds=args.wall_budget_seconds, require_same_code=not args.ignore_code_identity)
    else:
        print(json.dumps(status(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
