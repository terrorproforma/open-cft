"""Steady-state v4-fast: preregistered SOLVER-QUALIFICATION replay of the accepted v4 33.3 um plateau under the multigrid field
solve (poisson_gmg_v1, ``device-mg`` 14 cycles) and the model v2.0.5 electron-moment sampling interval K = 5.

The accepted ``pic2d_cft_steady_state_v4`` run (one execution at ``392129e5``, record ``0d228ad2``: I_d 3.801 mA, S 3.595e16 /s,
utilisation 0.4204, n_g 3.188e19, I_beam 2.459 mA, peak n_e 1.287e18, T_e,peak 5.577 eV at 3.03 transits) is replayed bit-for-bit
- grid 90 x 720, dt 1.4 ps, W 26 666.7, operating point, v1.3 closure, seed 20260903, frames ON, v2.0.3 gates - EXCEPT
``numerics.poisson = {"method": "device-mg", ...}`` and ``numerics.performance.moment_sample_interval = 5``.  Both enter
``config_sha256`` (a different identity, disclosed).  This is Class C item 4 of ``docs/pic2d-performance-audit.md`` section 9: only a
``qualified`` verdict here lets a later preregistered protocol name the fast solver.

Stages (from ``modern/``, ``PYTHONPATH=src:.``)::

    python -m experiments.pic2d_cft_steady_state_v4_fast.run preflight [--no-compare-v4]   # box: fast AND v4 protocol timed under the same MPS load
    python -m experiments.pic2d_cft_steady_state_v4_fast.run shakedown                     # 100k real-input steps through finalize + assess
    python -m experiments.pic2d_cft_steady_state_v4_fast.run launch --expect-commit SHA --require-mps [--resume]
    python -m experiments.pic2d_cft_steady_state_v4_fast.run status
    python -m experiments.pic2d_cft_steady_state_v4_fast.run finalize [...]                # only for an externally stopped run (shared runner)
    python -m experiments.pic2d_cft_steady_state_v4_fast.run assess [--runner-crash-log PATH]   # predeclared acceptance (a)-(e)

The stepping itself is the shared runner ``experiments.pic2d_cft_steady_state_v1.run`` (the multigrid is selected by the protocol's
``numerics.poisson`` object, K by ``numerics.performance``); the stages are the v4 / v5 preregistration discipline with this
experiment's paths, reference (the v4 run itself) and assessment schema.  The v4 module is frozen with its executed run and is
imported only for its host helpers.  On the box the launch goes through ``tools/cloud/schedule.py`` (job ``ss33-fast``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d import artifacts
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    EV_J,
    PIC2DValidationError,
)
from cft_revival.pic2d.simulation import Simulation
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v4.run import (
    _time_steps,
    device_memory,
    git,
    peak_working_set_bytes,
    utc_now,
    worktree_status,
)

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
PROTOCOL_PATH = HERE / "protocol.json"
RESULTS = HERE / "results"
PREFLIGHT_PATH = HERE / "preflight.json"
SHAKEDOWN_PATH = HERE / "shakedown.json"
LOCK_NAME = "execution-lock.json"
LOCK_SCHEMA = "cft-revival.pic2d-steady-state-execution-lock/1.0.0"
ASSESSMENT_SCHEMA = "cft-revival.pic2d-cft-steady-state-v4-fast.assessment/1.0.0"
# the run being replayed: the accepted v4 33.3 um plateau (its protocol builds the block-Thomas / K = 1 configuration)
V4_DIR = MODERN / "experiments" / "pic2d_cft_steady_state_v4"
V4_PROTOCOL_PATH = V4_DIR / "protocol.json"
REFERENCE_RESULTS = V4_DIR / "results"
V4_VERDICT_TIME_S = 7.28e-6
CONTRACT_MISS_MARKER = "failed its residual contract"
VERDICTS = ("qualified", "not_qualified", "heating", "no_plateau")
# (b): the replay's natively corrected (v2.0.6) windowed residual must lie within this band of the v4 plateau's post-hoc corrected value
B_BAND = 0.01
V4_CORRECTED_KEY = "windowed_residual_over_electrode_work_corrected_v2_0_6"
V4_LEDGER_SIDECAR = REFERENCE_RESULTS / "ledger-corrected.json"
JUDGED = ("discharge_current_a", "exit_ion_beam_a", "ionization_rate_per_s", "gross_utilisation", "neutral_density_per_m3",
          "peak_n_e_window_per_m3", "t_e_peak_window_ev")

# shakedown: the real protocol with only the cadences shrunk (every gate, the grid, dt, W, field, seed, solver and K are the real ones)
SHAKEDOWN_OVERRIDES = {
    "series_interval_steps": 200, "device_sync_steps": 200, "checkpoint_every_steps": 4000, "averaging_window_steps": 40000,
    "frame_cadence_steps": 2000, "peak_debye_window_steps": 40000, "peak_debye_window_snapshot_steps": 4000,
    "residual_window_steps": 40000, "max_steps": 100000,
}


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return runner.load_protocol(path)


def load_v4_protocol(path: Path = V4_PROTOCOL_PATH) -> dict[str, Any]:
    return runner.load_protocol(path)


# -- GPU / MPS telemetry (optional, never raises) -----------------------------------------------------------------------------

def _nvidia_smi(query: str, timeout_s: float = 10.0) -> list[list[str]]:
    try:
        out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception:  # noqa: BLE001 - telemetry is optional
        return []
    if out.returncode != 0:
        return []
    return [[cell.strip() for cell in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


def _float_or_nan(text: str) -> float:
    try:
        return float(text)
    except ValueError:      # Windows reports "[N/A]" per process; NaN fails every memory filter, so such rows never count as clients
        return float("nan")


def compute_apps() -> list[dict[str, Any]]:
    """Every CUDA compute process on the box (under MPS the server itself is one entry, ~66 MiB)."""

    return [{"pid": int(row[0]), "used_memory_mib": _float_or_nan(row[1])} for row in _nvidia_smi("--query-compute-apps=pid,used_memory")
            if len(row) >= 2 and row[0].isdigit()]


def concurrent_mps_clients(apps: list[dict[str, Any]] | None = None, *, own_pid: int | None = None, min_mib: float = 200.0) -> dict[str, Any]:
    """The OTHER PIC clients sharing the GPU: every compute process but this one and the MPS server (memory filter)."""

    apps = compute_apps() if apps is None else apps
    own = os.getpid() if own_pid is None else own_pid
    others = [a for a in apps if a["pid"] != own and a["used_memory_mib"] > min_mib]
    return {"count": len(others), "pids": [a["pid"] for a in others], "compute_apps": apps,
            "mps_pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
            "mps_pipe_present": bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY") and Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"]).exists())}


def gpu_inventory() -> list[dict[str, Any]]:
    return [{"name": row[0], "uuid": row[1], "driver_version": row[2], "memory_total_mib": float(row[3])}
            for row in _nvidia_smi("--query-gpu=name,uuid,driver_version,memory.total") if len(row) >= 4]


def gpu_load_snapshot(timeout_s: float = 5.0) -> dict[str, Any] | None:
    rows = _nvidia_smi("--query-gpu=utilization.gpu,memory.used,memory.total", timeout_s=timeout_s)
    if not rows or len(rows[0]) < 3:
        return None
    try:
        return {"utilization_percent": float(rows[0][0]), "memory_used_mib": float(rows[0][1]), "memory_total_mib": float(rows[0][2]),
                "note": "sampled before this process touched the device; utilisation >> 0 means other processes share the GPU and the ms/step is an upper bound"}
    except ValueError:
        return None


# -- preflight -------------------------------------------------------------------------------------------------------------

def multigrid_solver_stats(sim: Simulation) -> dict[str, Any] | None:
    """Launch count, device / host footprint, hierarchy and the last interval's worst contract ratio of the device multigrid (None otherwise)."""

    solver = getattr(sim.backend, "device_direct", None)
    if solver is None or not hasattr(solver, "last_worst_ratio"):
        return None
    levels = getattr(solver, "levels", None)
    return {
        "method": "device-mg", "cycles": getattr(solver, "cycles", None), "launches_per_solve": getattr(solver, "launches_per_solve", None),
        "device_memory_bytes": getattr(solver, "device_memory_bytes", None), "host_memory_bytes": getattr(solver, "host_memory_bytes", None),
        "levels": None if levels is None else len(levels), "last_interval_worst_contract_ratio": float(solver.last_worst_ratio),
    }


def _time_steps_tracking_contract(sim: Simulation, steps: int, *, warmup: int, chunk: int = 200) -> dict[str, Any]:
    """``_time_steps`` in sync-sized chunks so the multigrid's interval-worst contract ratio is read after every host sync.

    ``verify()`` runs at each 200-step sync inside ``sim.run`` and resets the running maximum; reading ``last_worst_ratio`` after each
    chunk keeps the maximum over the whole timed span.  The chunk loop adds ~10 Python iterations to 2000 steps (negligible).
    """

    start = sim.backend.step_index
    sim.run(warmup, accumulate_from_step=start)
    worst = 0.0
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        n = min(chunk, steps - done)
        sim.run(n, accumulate_from_step=start)
        done += n
        stats = multigrid_solver_stats(sim)
        if stats is not None:
            worst = max(worst, stats["last_interval_worst_contract_ratio"])
    elapsed = time.perf_counter() - t0
    return {"steps": steps, "seconds": elapsed, "ms_per_step": 1e3 * elapsed / steps, "accumulation": True,
            "worst_contract_ratio_over_timed_steps": worst}


def _expected_gate_reading(grid_spacing_m: float, dt_s: float, n_peak: float, t_e_ev: float, config: Any) -> dict[str, float]:
    lam = np.sqrt(EPSILON_0_F_PER_M * t_e_ev * EV_J / (n_peak * ELEMENTARY_CHARGE_C**2))
    omega = np.sqrt(n_peak * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ELECTRON_MASS_KG))
    return {"peak_n_e_per_m3": float(n_peak), "t_e_peak_ev": float(t_e_ev), "debye_length_m": float(lam), "cells_per_debye": float(grid_spacing_m / lam),
            "omega_pe_dt": float(omega * dt_s), "soft_gate": config.peak_debye_gate.soft_cells_per_debye, "hard_gate": config.peak_debye_gate.max_cells_per_debye}


def _timed_pair(protocol: dict[str, Any], field_map: Any, cross_sections: Any, *, backend: str, timing_steps: int, loaded_seed_density: float,
                track_contract: bool, log: Callable[[str], None], label: str) -> dict[str, Any]:
    """Seed-load and synthetic plateau-load timing of one protocol (the fast one or the v4 one) on the shared field / cross sections."""

    config = runner.build_config(protocol, backend=backend)
    before = device_memory()
    t0 = time.perf_counter()
    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend, step_graph=runner.step_graph_flag(protocol))
    build_seconds = time.perf_counter() - t0
    seed_state = sim.state
    record: dict[str, Any] = {
        "label": label, "config_sha256": artifacts.config_identity(config), "poisson": config.poisson.to_dict(),
        "moment_sample_interval": config.moment_sample_interval, "solver_build_seconds": build_seconds,
        "stability_gate": sim.stability.to_dict(), "v1_4_options": sim.to_provenance()["v1_4_options"],
        "seed_particles": {"electrons": seed_state.electrons.count, "ions": seed_state.ions.count},
    }
    timer = _time_steps_tracking_contract if track_contract else _time_steps
    timing_seed = timer(sim, timing_steps, warmup=200)
    after = device_memory()
    timing_seed.update({"electrons_after": sim.state.electrons.count, "ions_after": sim.state.ions.count, "step_graph": sim.step_graph_state()})
    record["timing_seed_load"] = timing_seed
    record["multigrid"] = multigrid_solver_stats(sim)
    record["last_series_record"] = sim.series[-1].to_dict() if sim.series else None
    log(f"[preflight] {label}: build {build_seconds:.1f} s; seed load {timing_seed['ms_per_step']:.3f} ms/step over {timing_steps} steps "
        f"({sim.state.electrons.count} e-); multigrid {record['multigrid']}")
    loaded = copy.deepcopy(protocol)
    loaded["operating_point"]["seed_plasma_density_per_m3"] = loaded_seed_density
    loaded_config = runner.build_config(loaded, backend=backend)
    del sim
    sim2 = Simulation(loaded_config, field_map, cross_sections=cross_sections, backend=backend, step_graph=runner.step_graph_flag(protocol))
    loaded_state = sim2.state
    timing_loaded = timer(sim2, timing_steps, warmup=200)
    after_loaded = device_memory()
    timing_loaded.update({"seed_density_per_m3": loaded_seed_density, "electrons": loaded_state.electrons.count, "ions": loaded_state.ions.count,
                          "electrons_after": sim2.state.electrons.count, "ions_after": sim2.state.ions.count, "step_graph": sim2.step_graph_state()})
    record["timing_plateau_load"] = timing_loaded
    record["multigrid_plateau_load"] = multigrid_solver_stats(sim2)
    record["memory"] = {
        "device_before": before, "device_after_seed_run": after, "device_after_loaded_run": after_loaded,
        "device_used_by_seed_run_bytes": None if before is None or after is None else before["free_bytes"] - after["free_bytes"],
        "device_used_by_loaded_run_bytes": None if before is None or after_loaded is None else before["free_bytes"] - after_loaded["free_bytes"],
    }
    per_m = (timing_loaded["ms_per_step"] - timing_seed["ms_per_step"]) / max(
        (loaded_state.electrons.count + loaded_state.ions.count - seed_state.electrons.count - seed_state.ions.count) / 1e6, 1e-9)
    record["ms_per_step_per_million_particles"] = per_m
    log(f"[preflight] {label}: plateau load {timing_loaded['ms_per_step']:.3f} ms/step over {timing_steps} steps ({loaded_state.electrons.count} e- + "
        f"{loaded_state.ions.count} i); {per_m:.3f} ms per M particles")
    del sim2
    return record


def preflight(protocol: dict[str, Any], *, backend: str = "warp-cuda", timing_steps: int = 2000, loaded_seed_density: float = 1.75e17,
              compare_v4: bool = True, output: Path = PREFLIGHT_PATH, log: Callable[[str], None] = lambda text: print(text, flush=True)) -> dict[str, Any]:
    """Real inputs (P2 field on the 33 um grid, mesh, cross sections), multigrid build, memory and ms/step at two loads - for THIS protocol
    and (``compare_v4``) for the v4 protocol under the same GPU load: the honest contended A/B.  The concurrent MPS clients are recorded.

    ``loaded_seed_density`` seeds a synthetic plasma at the expected plateau load (2 x 1.75e17 x 3.44e-7 / 2.667e4 = 4.5 M macro-particles).
    Non-evidentiary: nothing here is a result.
    """

    config = runner.build_config(protocol, backend=backend)
    grid = config.grid
    gpu_before = gpu_load_snapshot()
    clients_before = concurrent_mps_clients()
    record: dict[str, Any] = {
        "schema_version": "cft-revival.pic2d-cft-steady-state-v4-fast.preflight/1.0.0", "utc": utc_now(), "git_head": runner.git_head(),
        "protocol_sha256": runner._file_sha256(PROTOCOL_PATH), "config_sha256": artifacts.config_identity(config), "backend": backend,
        "host": socket.gethostname(), "python": sys.version.split()[0], "non_evidentiary": True, "gpu": gpu_inventory(),
        "gpu_load_before": gpu_before, "concurrent_mps_clients_before": clients_before,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "grid": {"cells": list(grid.cell_shape), "nodes": list(grid.node_shape), "dr_m": grid.dr_m, "dz_m": grid.dz_m, "dr_over_dz": grid.dr_m / grid.dz_m},
        "dt_s": config.dt_s, "macro_weight": config.macro_weight, "timing_steps": timing_steps,
    }
    t0 = time.perf_counter()
    field_map, cross_sections = runner.load_inputs(config, None, None, protocol=protocol)
    record["field"] = {"sha256": field_map.sha256, "source_sha256": field_map.source_sha256, "max_b_t": field_map.max_b_t,
                       "seconds": time.perf_counter() - t0, "evidence": field_map.to_dict().get("evidence") or field_map.to_dict().get("source")}
    record["cross_sections_sha256"] = cross_sections.payload_sha256 if cross_sections is not None else None
    masks = build_mesh_masks(grid)
    record["mesh"] = masks.to_dict()
    log(f"[preflight] grid {grid.cell_shape} dr {grid.dr_m*1e6:.3f} um; field {field_map.sha256[:12]} max |B| {field_map.max_b_t:.3f} T; mesh "
        f"{masks.to_dict()['plasma_cells']} plasma cells; GPU before: {gpu_before}; other MPS clients: {clients_before['count']} {clients_before['pids']}")
    record["fast"] = _timed_pair(protocol, field_map, cross_sections, backend=backend, timing_steps=timing_steps, loaded_seed_density=loaded_seed_density,
                                 track_contract=True, log=log, label="fast (device-mg 14 cycles, K = 5)")
    if compare_v4:
        v4p = load_v4_protocol()
        record["v4"] = _timed_pair(v4p, field_map, cross_sections, backend=backend, timing_steps=timing_steps, loaded_seed_density=loaded_seed_density,
                                   track_contract=False, log=log, label="v4 (block-Thomas device-direct, K = 1)")
        record["contended_ratio_fast_over_v4"] = {
            "seed_load": record["fast"]["timing_seed_load"]["ms_per_step"] / record["v4"]["timing_seed_load"]["ms_per_step"],
            "plateau_load": record["fast"]["timing_plateau_load"]["ms_per_step"] / record["v4"]["timing_plateau_load"]["ms_per_step"],
            "note": ("both protocols timed one after the other under the SAME background (the concurrent MPS clients above; contention can drift "
                     "between the two); a ratio > 1 means the fast configuration is SLOWER while contended - the latency-bound multigrid's "
                     "278 dependent launches wait for SM shares (audit section 12.3) - and says nothing about the solo cost"),
        }
    record["concurrent_mps_clients_after"] = concurrent_mps_clients()
    record["host_peak_working_set_bytes"] = peak_working_set_bytes()
    budget = runner.protocol_budget(protocol)
    transit = float(budget["ion_transit_time_s"])
    steps_3 = 3.0 * transit / config.dt_s
    steps_v4_verdict = V4_VERDICT_TIME_S / config.dt_s
    ms = record["fast"]["timing_plateau_load"]["ms_per_step"]
    ms_seed = record["fast"]["timing_seed_load"]["ms_per_step"]
    hours_3 = steps_3 * ms / 3.6e6
    record["projection"] = {
        "steps_per_transit": transit / config.dt_s, "steps_to_3_transits": steps_3, "steps_to_v4_verdict_time_7_28_us": steps_v4_verdict,
        "hours_to_3_transits_at_plateau_load": hours_3, "hours_to_v4_verdict_time_at_plateau_load": steps_v4_verdict * ms / 3.6e6,
        "hours_to_3_transits_at_seed_load": steps_3 * ms_seed / 3.6e6,
        "ms_per_step_per_million_particles": record["fast"]["ms_per_step_per_million_particles"],
        "wall_budget_seconds": float(protocol["stopping_rule"]["wall_budget_seconds"]),
        "budget_over_3_transit_time": float(protocol["stopping_rule"]["wall_budget_seconds"]) / max(steps_3 * ms / 1e3, 1e-9),
        "budget_rule_1_5x_contended_seconds": 1.5 * hours_3 * 3600.0,
        "note": ("the particle count grows from the seed (1.3 M) to the plateau (~4.5 M) over the first ~1.5 us, so the wall time to 3 transits lies "
                 f"between the two projections, close to the plateau-load one; measured with {clients_before['count']} other MPS client(s) on the GPU - "
                 "the per-process rate changes as they finish or others start; the budget rule is 1.5 x the contended plateau-load projection"),
    }
    ref = protocol["reference_run"]["quantities"]
    record["expected_at_v4_peak"] = _expected_gate_reading(max(grid.dr_m, grid.dz_m), config.dt_s, ref["peak_n_e_window_per_m3"], ref["t_e_peak_window_ev"], config)
    artifacts.write_canonical_json(output, record)
    log(f"[preflight] projection: {hours_3:.1f} h to 3 transits at the contended plateau load (1.5x rule -> {record['projection']['budget_rule_1_5x_contended_seconds']:.0f} s); "
        f"budget/3-transit {record['projection']['budget_over_3_transit_time']:.2f}; written {output}")
    return record


# -- shakedown -------------------------------------------------------------------------------------------------------------

def shakedown_protocol(protocol: dict[str, Any], overrides: dict[str, Any] = SHAKEDOWN_OVERRIDES) -> dict[str, Any]:
    """The real protocol with every cadence shrunk (NON-EVIDENTIARY): grid, dt, W, field, seed, solver, K and gate thresholds untouched."""

    p = copy.deepcopy(protocol)
    num = p["numerics"]
    num["series_interval_steps"] = overrides["series_interval_steps"]
    num["device_sync_steps"] = overrides["device_sync_steps"]
    num["checkpoint_every_steps"] = overrides["checkpoint_every_steps"]
    num["averaging_window_steps"] = overrides["averaging_window_steps"]
    num["frame_recorder"] = {"cadence_steps": overrides["frame_cadence_steps"], "precision": "float32"}
    num["peak_debye_gate"]["window_steps"] = overrides["peak_debye_window_steps"]
    num["peak_debye_gate"]["window_snapshot_steps"] = overrides["peak_debye_window_snapshot_steps"]
    p["stopping_rule"]["grid_heating_triad"]["residual_window_steps"] = overrides["residual_window_steps"]
    p["status"] = "SHAKEDOWN_non_evidentiary_shrunk_cadences"
    p["experiment_id"] = protocol["experiment_id"] + "-shakedown"
    return p


def shakedown(protocol: dict[str, Any], *, results: Path | None = None, backend: str = "warp-cuda", output: Path = SHAKEDOWN_PATH,
              log: Callable[[str], None] = lambda text: print(text, flush=True)) -> dict[str, Any]:
    results = HERE / "results-shakedown" if results is None else results
    if results.exists():
        shutil.rmtree(results)
    p = shakedown_protocol(protocol)
    shake_protocol_path = results / "protocol-shakedown.json"
    results.mkdir(parents=True)
    artifacts.write_canonical_json(shake_protocol_path, p)
    clients_before = concurrent_mps_clients()
    t0 = time.perf_counter()
    summary_path = runner.run_steady_state(p, results, backend=backend, max_steps=SHAKEDOWN_OVERRIDES["max_steps"], protocol_path=shake_protocol_path, log=log)
    run_seconds = time.perf_counter() - t0
    summary = artifacts.read_canonical_json(summary_path)
    assessment = assess(p, results, log=log, reference_check=True)
    status_lines = runner._read_jsonl(results / "status.jsonl")
    samples = [s for s in status_lines if "event" not in s]
    windows = [s["peak_node"]["window"] for s in samples if s.get("peak_node", {}).get("window") is not None]
    enforced = [w for w in windows if w["gate_enforced"]]
    triads = [s["grid_heating_triad"] for s in samples if s.get("grid_heating_triad") is not None]
    complete = [t for t in triads if t.get("windowed_energy_residual_window_complete")]
    record = {
        "schema_version": "cft-revival.pic2d-cft-steady-state-v4-fast.shakedown/1.0.0", "utc": utc_now(), "git_head": runner.git_head(),
        "non_evidentiary": True, "overrides": SHAKEDOWN_OVERRIDES, "results_dir": results.name, "run_seconds": run_seconds,
        "host": socket.gethostname(), "gpu": gpu_inventory(), "concurrent_mps_clients_before": clients_before, "concurrent_mps_clients_after": concurrent_mps_clients(),
        "steps_completed": summary["steps_completed"], "stop_reason": summary["stop_reason"], "ms_per_step": summary["ms_per_step_this_session"],
        "final_counts": summary["final_counts"], "frames": summary["artifacts"]["frames"]["count"] if summary["artifacts"]["frames"] else 0,
        "poisson_provenance": summary["provenance"]["config"]["poisson"], "moment_sample_interval": summary["provenance"]["config"].get("moment_sample_interval", 1),
        "config_sha256": summary["provenance"]["config_sha256"],
        "peak_debye_window": {
            "records": len(windows), "enforced_records": len(enforced), "last": windows[-1] if windows else None,
            "max_cells_per_debye_enforced": max((w["cells_per_debye"] for w in enforced), default=None),
        },
        "windowed_residual": {
            "records_with_complete_window": len(complete),
            "last": None if not complete else {k: complete[-1][k] for k in complete[-1] if k.startswith("windowed") or k == "energy_residual_over_electrode_work"},
        },
        "plateau_keys": sorted(summary["plateau"]) if summary["plateau"] else None,
        "summary_keys_present": {key: (summary.get(key) is not None) for key in ("plateau", "grid_heating_triad", "peak_node_debye", "neutral_inventory", "ledger")},
        "assessment": {k: assessment[k] for k in ("verdict", "a_plateau", "b_residual_power", "c_replay", "d_field_solve_contract", "reference_consistency")},
        "artifacts": {k: summary["artifacts"][k] for k in ("maps_npz_sha256", "series_npz_sha256")},
        "gate_not_inert_check": {
            "peak_window_enforced_at_least_once": bool(enforced), "peak_window_resolved_nodes_last": windows[-1]["resolved_nodes"] if windows else None,
            "residual_window_completed_at_least_once": bool(complete),
            # K = 5 evidence: the final series record's window gate block carries the sample count only when K != 1 (v2.0.5 identity rule)
            "window_moment_samples_final_record": (((summary.get("final_series") or {}).get("peak_node") or {}).get("window") or {}).get("window_moment_samples"),
            "window_steps_final_record": (((summary.get("final_series") or {}).get("peak_node") or {}).get("window") or {}).get("window_steps"),
        },
    }
    artifacts.write_canonical_json(output, record)
    log(f"[shakedown] {summary['steps_completed']} steps, {summary['stop_reason']}, {summary['ms_per_step_this_session']:.2f} ms/step with "
        f"{clients_before['count']} other MPS client(s), {record['frames']} frames, peak window enforced in {len(enforced)}/{len(windows)} records "
        f"(max {record['peak_debye_window']['max_cells_per_debye_enforced']}), residual window complete in {len(complete)} records; poisson "
        f"{record['poisson_provenance']['method']} K {record['moment_sample_interval']}; assessment verdict {assessment['verdict']}; written {output}")
    return record


# -- launch -------------------------------------------------------------------------------------------------------------------

def acquire_lock(results: Path, payload: dict[str, Any]) -> Path:
    """O_EXCL canonical lock in the results directory; refuses to overwrite (same-attempt / different-attempt classified)."""

    results.mkdir(parents=True, exist_ok=True)
    path = results / LOCK_NAME
    data = canonical_bytes(payload) + b"\n"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        same = all(existing.get(k) == payload.get(k) for k in ("experiment_id", "commit", "protocol_sha256"))
        raise PIC2DValidationError(f"execution lock already exists at {path} ({'same-attempt' if same else 'different-attempt'}: "
                                   f"commit {existing.get('commit', '?')[:12]}, acquired {existing.get('acquired_at_utc')}); refusing to launch")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def launch(protocol: dict[str, Any], *, results: Path = RESULTS, backend: str = "warp-cuda", expect_commit: str | None = None,
           resume: bool = False, allow_dirty: bool = False, require_mps: bool = False, wall_budget_seconds: float | None = None,
           log: Callable[[str], None] = lambda text: print(text, flush=True)) -> Path:
    """Preregistered execution: clean worktree, expected commit, records present, MPS pipe (``--require-mps``), exclusive lock, shared runner."""

    head = git("rev-parse", "HEAD")
    if expect_commit is not None and not head.startswith(expect_commit):
        raise PIC2DValidationError(f"HEAD {head[:12]} is not the preregistration commit {expect_commit}")
    dirty = worktree_status()
    if dirty and not allow_dirty:
        raise PIC2DValidationError(f"worktree is not clean ({len(dirty)} entries, e.g. {dirty[0]!r}); the preregistered launch requires a clean checkout")
    protocol_sha = runner._file_sha256(PROTOCOL_PATH)
    relative = PROTOCOL_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    if git("rev-parse", f"HEAD:{relative}") != git("hash-object", "--", str(PROTOCOL_PATH)):
        raise PIC2DValidationError("protocol.json on disk differs from the committed blob at HEAD")
    if not PREFLIGHT_PATH.is_file() or not SHAKEDOWN_PATH.is_file():
        raise PIC2DValidationError("preflight.json and shakedown.json must exist (and be committed) before a preregistered launch")
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if require_mps and not (mps_pipe and Path(mps_pipe).exists()):
        raise PIC2DValidationError(f"--require-mps: CUDA_MPS_PIPE_DIRECTORY {mps_pipe!r} is not set or does not exist in this environment")
    config = runner.build_config(protocol, backend=backend)
    if config.poisson.method != "device-mg" or config.moment_sample_interval != 5:
        raise PIC2DValidationError("this experiment launches only the declared fast configuration (device-mg, K = 5)")
    clients = concurrent_mps_clients()
    payload = {
        "schema_version": LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "commit": head, "protocol_sha256": protocol_sha,
        "config_sha256": artifacts.config_identity(config), "backend": backend,
        "command": " ".join(sys.argv), "host": socket.gethostname(), "pid": os.getpid(), "acquired_at_utc": utc_now(),
        "clean_worktree_attested": not dirty, "worktree": str(REPOSITORY_ROOT), "immutable": True,
        "gpu": gpu_inventory(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_mps_pipe_directory": mps_pipe,
        "mps_required": require_mps, "concurrent_mps_clients_at_launch": clients["count"], "concurrent_mps_client_pids_at_launch": clients["pids"],
        "poisson": config.poisson.to_dict(), "moment_sample_interval": config.moment_sample_interval,
    }
    lock = results / LOCK_NAME
    if resume:
        if not lock.is_file():
            raise PIC2DValidationError("--resume needs the execution lock of the first session")
        existing = json.loads(lock.read_text(encoding="utf-8"))
        if existing.get("commit") != head or existing.get("protocol_sha256") != protocol_sha:
            raise PIC2DValidationError("--resume refused: the lock names a different commit / protocol")
        if runner.find_checkpoint(results) is None:
            raise PIC2DValidationError("--resume refused: no checkpoint to resume from")
        log(f"[launch] resuming under the existing lock (commit {head[:12]}, acquired {existing.get('acquired_at_utc')})")
    else:
        if runner.find_checkpoint(results) is not None:
            raise PIC2DValidationError(f"{results} already holds a checkpoint; use --resume for a new session under the same lock")
        acquire_lock(results, payload)
        log(f"[launch] execution lock acquired: commit {head[:12]}, protocol {protocol_sha[:12]}, clean worktree {not dirty}, MPS pipe {mps_pipe}, "
            f"{clients['count']} other MPS client(s) {clients['pids']}, poisson {config.poisson.method} x{config.poisson.mg_cycles}, K {config.moment_sample_interval}")
    return runner.run_steady_state(protocol, results, backend=backend, protocol_path=PROTOCOL_PATH, wall_budget_seconds=wall_budget_seconds, log=log)


# -- assessment ----------------------------------------------------------------------------------------------------------------

def _peak_from_maps(maps_path: Path) -> dict[str, Any]:
    maps = artifacts.read_npz(maps_path)
    n = np.asarray(maps["n_e_per_m3"])
    t = np.asarray(maps["t_e_ev"])
    i, j = np.unravel_index(int(np.nanargmax(n)), n.shape)
    return {"peak_n_e_window_per_m3": float(n[i, j]), "t_e_peak_window_ev": float(t[i, j]), "node": [int(i), int(j)]}


def reference_quantities_from_files(results: Path = REFERENCE_RESULTS) -> dict[str, float] | None:
    """Recompute the pinned v4 numbers from the v4 results directory (summary.json + maps.npz) - the v4 assessment's own extraction."""

    summary_path = results / "summary.json"
    if not summary_path.is_file() or not (results / "maps.npz").is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    peak = _peak_from_maps(results / "maps.npz")
    return {
        "discharge_current_a": summary["window_currents_a"]["discharge_a"],
        "exit_ion_beam_a": summary["window_currents_a"]["exit_ion_beam_a"],
        "ionization_rate_per_s": summary["neutral_inventory"]["trailing_20pct_mean_ionization_rate_per_s"],
        "gross_utilisation": summary["neutral_inventory"]["propellant_utilisation_trailing"],
        "neutral_density_per_m3": summary["neutral_inventory"]["trailing_20pct_mean_density_per_m3"],
        "peak_n_e_window_per_m3": peak["peak_n_e_window_per_m3"],
        "t_e_peak_window_ev": peak["t_e_peak_window_ev"],
    }


def reference_corrected_residual_from_sidecar(results: Path = REFERENCE_RESULTS) -> dict[str, Any] | None:
    """The v2.0.6 post-hoc ledger correction of the v4 run (results/ledger-corrected.json): the corrected windowed / cumulative ratios."""

    path = results / "ledger-corrected.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    end = record["end_state_window"]
    return {"windowed_corrected": float(end["corrected_ratio"]), "windowed_recorded": float(end["recorded_ratio"]),
            "cumulative_corrected": float(record["cumulative"]["corrected_over_electrode"]), "step": int(end["step"]),
            "window_complete": bool(end["window_complete"]), "recorded_ratio_matches_summary": bool(end["recorded_ratio_matches_summary"]),
            "max_corrected_over_complete_windows": record["max_over_complete_windows"]["corrected"], "schema": record.get("schema")}


def reference_quantities_from_assessment(results: Path = REFERENCE_RESULTS) -> dict[str, Any] | None:
    """The v4 assessment.json ``run`` block (the second, independent copy of the reference numbers)."""

    path = results / "assessment.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return {k: record["run"][k] for k in JUDGED} | {"verdict": record["verdict"], "config_sha256": record["run"]["config_sha256"],
                                                    "windowed_residual_over_electrode_work": record["run"]["windowed_residual_over_electrode_work"],
                                                    "cells_per_debye_window_last": record["run"]["cells_per_debye_window_last"]}


def run_quantities(results: Path) -> dict[str, Any]:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    peak = _peak_from_maps(results / "maps.npz")
    triad = summary.get("grid_heating_triad") or {}
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
    provenance = summary.get("provenance") or {}
    config = provenance.get("config") or {}
    return {
        "discharge_current_a": summary["window_currents_a"]["discharge_a"],
        "exit_ion_beam_a": summary["window_currents_a"]["exit_ion_beam_a"],
        "ionization_rate_per_s": summary["neutral_inventory"]["trailing_20pct_mean_ionization_rate_per_s"],
        "gross_utilisation": summary["neutral_inventory"]["propellant_utilisation_trailing"],
        "neutral_density_per_m3": summary["neutral_inventory"]["trailing_20pct_mean_density_per_m3"],
        "peak_n_e_window_per_m3": peak["peak_n_e_window_per_m3"],
        "t_e_peak_window_ev": peak["t_e_peak_window_ev"],
        "peak_node": peak["node"],
        "stop_reason": summary["stop_reason"],
        "ion_transit_times": summary["ion_transit_times"],
        "steps_completed": summary["steps_completed"],
        "plateau": summary.get("plateau"),
        "windowed_residual_over_electrode_work": triad.get("windowed_energy_residual_over_electrode_work"),
        "windowed_residual_window_complete": triad.get("windowed_energy_residual_window_complete"),
        "cumulative_residual_over_electrode_work": triad.get("energy_residual_over_electrode_work"),
        "cells_per_debye_window_last": debye.get("cells_per_debye_window_last"),
        "cells_per_debye_window_trailing_mean": debye.get("trailing_20pct_mean_cells_per_debye_window"),
        "peak_debye_soft_ok": debye.get("soft_ok"),
        "maps_kind": summary.get("maps_kind"),
        "sessions": len(summary.get("sessions") or []),
        "ms_per_step_this_session": summary.get("ms_per_step_this_session"), "wall_seconds_total": summary.get("wall_seconds_total"),
        "git_head": summary.get("git_head"), "protocol_sha256": summary.get("protocol_sha256"),
        "config_sha256": provenance.get("config_sha256"),
        "poisson": config.get("poisson"), "moment_sample_interval": config.get("moment_sample_interval", 1),
    }


def _compare(run: dict[str, Any], reference: dict[str, float], tolerances: dict[str, float], bands: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    rows: dict[str, Any] = {}
    all_within = True
    for key, tolerance in tolerances.items():
        ref = float(reference[key])
        value = float(run[key])
        rel = (value - ref) / abs(ref) if ref != 0.0 else float("inf")
        within = abs(rel) <= tolerance
        all_within = all_within and within
        rows[key] = {"reference": ref, "value": value, "relative_difference": rel, "tolerance": tolerance, "within": within,
                     "seed_b_band": bands.get("seed_b", {}).get(key), "w_0_7_band": bands.get("w_0_7", {}).get(key)}
    return rows, all_within


def _consistency(pinned: dict[str, Any], results: Path) -> dict[str, Any] | None:
    recomputed = reference_quantities_from_files(results)
    if recomputed is None:
        return None
    rows = {key: {"pinned": float(pinned[key]), "recomputed": float(recomputed[key]),
                  "agree": abs(float(recomputed[key]) - float(pinned[key])) <= 1e-9 * max(abs(float(pinned[key])), 1e-300)} for key in recomputed}
    assessed = reference_quantities_from_assessment(results)
    if assessed is not None:
        for key in JUDGED:
            rows[key]["v4_assessment_run"] = float(assessed[key])
            rows[key]["agree"] = bool(rows[key]["agree"] and abs(float(assessed[key]) - float(pinned[key])) <= 1e-9 * max(abs(float(pinned[key])), 1e-300))
    corrected = reference_corrected_residual_from_sidecar(results)
    if corrected is not None and V4_CORRECTED_KEY in pinned:
        pin = float(pinned[V4_CORRECTED_KEY])
        rows[V4_CORRECTED_KEY] = {"pinned": pin, "recomputed": corrected["windowed_corrected"],
                                  "agree": abs(corrected["windowed_corrected"] - pin) <= 1e-9 * max(abs(pin), 1e-300)
                                  and corrected["window_complete"] and corrected["recorded_ratio_matches_summary"]
                                  and abs(corrected["windowed_recorded"] - float(pinned["windowed_residual_over_electrode_work_last"])) <= 1e-9}
    return rows


def _contract_from_run(run: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """(d): a runner terminal state exists (no PIC2DConvergenceError escaped) AND the provenance names the declared multigrid configuration."""

    declared = runner.build_config(protocol, backend="warp-cuda").poisson.to_dict()
    recorded = run.get("poisson") or {}
    same = recorded.get("method") == declared["method"] and recorded.get("multigrid") == declared.get("multigrid")
    return {"passed": bool(same), "declared_poisson": declared, "recorded_poisson": recorded, "terminal_state": True,
            "moment_sample_interval_recorded": run.get("moment_sample_interval"),
            "rule": protocol["stopping_rule"]["acceptance"]["d_field_solve_contract"]}


def _crash_log_contract(log_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    lines = [line for line in text.splitlines() if CONTRACT_MISS_MARKER in line]
    if not lines:
        raise PIC2DValidationError(f"{log_path} carries no '{CONTRACT_MISS_MARKER}' line: a missing summary.json without a contract miss is not an "
                                   "assessable outcome (finalize the run or report the stop separately)")
    return {"passed": False, "terminal_state": False, "runner_crash_log": str(log_path), "evidence": lines[-3:],
            "declared_poisson": runner.build_config(protocol, backend="warp-cuda").poisson.to_dict(), "recorded_poisson": None,
            "rule": protocol["stopping_rule"]["acceptance"]["d_field_solve_contract"]}


def assess(protocol: dict[str, Any], results: Path = RESULTS, *, output: Path | None = None, reference_check: bool = True,
           runner_crash_log: Path | None = None, log: Callable[[str], None] = lambda text: print(text, flush=True)) -> dict[str, Any]:
    """Predeclared acceptance (a)-(e) against protocol.reference_run.quantities = the accepted v4 33.3 um plateau.

    ``runner_crash_log``: the ONLY path for a run without summary.json - the log must carry the multigrid contract-miss line; the record
    then states (d) failed and verdict ``not_qualified``.
    """

    acceptance = protocol["stopping_rule"]["acceptance"]
    reference = protocol["reference_run"]["quantities"]
    bands = protocol["reference_run"].get("bands") or {}
    tolerances = {k: float(v) for k, v in acceptance["c_replay_tolerances"].items() if k != "note"}
    consistency = None
    if reference_check:
        consistency = _consistency(reference, REFERENCE_RESULTS)
        if consistency is not None and not all(entry["agree"] for entry in consistency.values()):
            raise PIC2DValidationError("protocol.reference_run.quantities disagree with the v4 artifacts on disk: " + json.dumps(
                {k: v for k, v in consistency.items() if not v["agree"]}))
    if not (results / "summary.json").is_file():
        if runner_crash_log is None:
            raise PIC2DValidationError(f"{results} has no summary.json to assess (pass --runner-crash-log for a multigrid contract miss)")
        contract = _crash_log_contract(runner_crash_log, protocol)
        checkpoint = runner.find_checkpoint(results)
        record = {
            "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "results_dir": results.name,
            "git_head_now": runner.git_head(), "run": {"summary": None, "last_checkpoint": None if checkpoint is None else str(checkpoint)},
            "reference": reference, "reference_consistency": consistency,
            "a_plateau": {"passed": False, "stop_reason": None, "rule": acceptance["a_plateau"]},
            "b_residual_power": {"passed": None, "rule": acceptance["b_residual_power"]},
            "c_replay": {"all_within": None, "quantities": None, "rule": acceptance["c_replay_tolerances"]["note"]},
            "d_field_solve_contract": contract, "e_verdict": acceptance["e_verdict"]["not_qualified"], "verdict": "not_qualified",
            "claim_boundary": protocol["claim_boundary"],
        }
        artifacts.write_canonical_json(output or (results / "assessment.json"), record)
        log(f"[assess] {results.name}: verdict not_qualified - multigrid contract missed (no terminal state; {runner_crash_log})")
        return record
    run = run_quantities(results)
    a_plateau = run["stop_reason"] == "plateau_reached_after_min_transit_times"
    windowed = run["windowed_residual_over_electrode_work"]      # v2.0.6 code: the corrected statistic H / electrode work, natively
    ref_corrected = float(reference[V4_CORRECTED_KEY])
    b_delta = None if windowed is None else windowed - ref_corrected
    b_ok = windowed is not None and bool(run["windowed_residual_window_complete"]) and abs(b_delta) <= B_BAND
    replay, all_within = _compare(run, reference, tolerances, bands)
    contract = _contract_from_run(run, protocol)
    if a_plateau and b_ok and all_within and contract["passed"]:
        verdict = "qualified"
    elif a_plateau and b_ok:
        verdict = "not_qualified"
    elif a_plateau:
        verdict = "heating"
    else:
        verdict = "no_plateau"
    ref_windowed = reference.get("windowed_residual_over_electrode_work_last")     # what v4 RECORDED (biased, pre-v2.0.6)
    ref_debye = reference.get("cells_per_debye_at_peak_window")
    record = {
        "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "results_dir": results.name,
        "git_head_now": runner.git_head(), "run": run, "reference": reference, "reference_consistency": consistency,
        "a_plateau": {"passed": a_plateau, "stop_reason": run["stop_reason"], "ion_transit_times": run["ion_transit_times"],
                      "plateau": run["plateau"], "rule": acceptance["a_plateau"]},
        "b_residual_power": {"passed": b_ok, "windowed_residual_over_electrode_work": windowed, "window_complete": run["windowed_residual_window_complete"],
                             "statistic": "v2.0.6 corrected (H / electrode work) recorded natively by the replay's code",
                             "v4_corrected_v2_0_6": ref_corrected, "band": B_BAND, "two_sided": True, "delta_vs_v4_corrected": b_delta,
                             "side": None if b_delta is None else ("heating" if b_delta > 0 else "cooling"),
                             "cumulative_witness": run["cumulative_residual_over_electrode_work"],
                             "v4_cumulative_corrected_v2_0_6": reference.get("cumulative_residual_over_electrode_work_corrected_v2_0_6"),
                             "v4_recorded_pre_v2_0_6": ref_windowed,
                             "project_acceptance_b_below_0p02": {"replay": None if windowed is None else bool(windowed < 0.02), "v4_corrected": bool(ref_corrected < 0.02),
                                                                 "note": "the project's plateau acceptance (b) < +2 % (gate_recalibration_v2_0_6), reported, not judged here: v4 itself reads +2.46 %"},
                             "rule": acceptance["b_residual_power"]},
        "c_replay": {"all_within": all_within, "quantities": replay, "rule": acceptance["c_replay_tolerances"]["note"],
                     "reference": "pic2d_cft_steady_state_v4/results (the accepted 33.3 um plateau; the run being replayed)",
                     "reported_not_judged": {
                         "cells_per_debye_window_last": {"value": run["cells_per_debye_window_last"], "v4": ref_debye,
                                                         "relative_difference": None if run["cells_per_debye_window_last"] is None or not ref_debye
                                                         else (run["cells_per_debye_window_last"] - ref_debye) / abs(ref_debye)},
                         "windowed_residual_over_electrode_work": {"value": windowed, "v4_recorded_pre_v2_0_6": ref_windowed, "v4_corrected_v2_0_6": ref_corrected},
                         "peak_node": {"value": run["peak_node"], "v4": reference.get("peak_node")},
                     }},
        "d_field_solve_contract": contract,
        "e_verdict": acceptance["e_verdict"][verdict],
        "verdict": verdict,
        "peak_debye_window": {"cells_per_debye_last": run["cells_per_debye_window_last"], "trailing_mean": run["cells_per_debye_window_trailing_mean"],
                              "soft_ok": run["peak_debye_soft_ok"]},
        "claim_boundary": protocol["claim_boundary"],
    }
    artifacts.write_canonical_json(output or (results / "assessment.json"), record)
    log(f"[assess] {results.name}: verdict {verdict} (a {a_plateau}, b {b_ok} [{windowed} vs v4 corrected {ref_corrected:+.4f}, band {B_BAND}], "
        f"c all_within {all_within}, d {contract['passed']}); "
        + ", ".join(f"{k} {v['relative_difference']*100:+.2f}% (tol {v['tolerance']*100:.0f}%)" for k, v in replay.items()))
    return record


# -- CLI ------------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--backend", default="warp-cuda")
    pre.add_argument("--timing-steps", type=int, default=2000)
    pre.add_argument("--loaded-seed-density", type=float, default=1.75e17)
    pre.add_argument("--no-compare-v4", action="store_true", help="skip the v4 (block-Thomas, K = 1) timing under the same load")
    shake = sub.add_parser("shakedown")
    shake.add_argument("--backend", default="warp-cuda")
    la = sub.add_parser("launch")
    la.add_argument("--backend", default="warp-cuda")
    la.add_argument("--expect-commit", default=None)
    la.add_argument("--resume", action="store_true")
    la.add_argument("--allow-dirty", action="store_true", help="development only; never for the preregistered execution")
    la.add_argument("--require-mps", action="store_true", help="refuse unless CUDA_MPS_PIPE_DIRECTORY is set and exists (the four-slot H100 configuration)")
    la.add_argument("--wall-budget-seconds", type=float, default=None)
    sub.add_parser("status")
    fin = sub.add_parser("finalize")
    fin.add_argument("--backend", default="warp-cuda")
    fin.add_argument("--stop-reason", default="finalized_from_checkpoint")
    fin.add_argument("--allow-refinalize", action="store_true")
    fin.add_argument("--recover-runner-stop", action="store_true")
    ass = sub.add_parser("assess")
    ass.add_argument("--results", default=None)
    ass.add_argument("--runner-crash-log", default=None, help="a run log carrying the multigrid contract-miss line (run without a terminal state)")
    args = parser.parse_args(argv)
    protocol = load_protocol()
    if args.command == "preflight":
        preflight(protocol, backend=args.backend, timing_steps=args.timing_steps, loaded_seed_density=args.loaded_seed_density, compare_v4=not args.no_compare_v4)
    elif args.command == "shakedown":
        shakedown(protocol, backend=args.backend)
    elif args.command == "launch":
        launch(protocol, backend=args.backend, expect_commit=args.expect_commit, resume=args.resume, allow_dirty=args.allow_dirty,
               require_mps=args.require_mps, wall_budget_seconds=args.wall_budget_seconds)
    elif args.command == "status":
        print(json.dumps(runner.status(RESULTS, protocol), indent=1))
    elif args.command == "finalize":
        runner.finalize(protocol, RESULTS, backend=args.backend, stop_reason=args.stop_reason, protocol_path=PROTOCOL_PATH,
                        allow_refinalize=args.allow_refinalize, recover_runner_stop=args.recover_runner_stop)
    else:
        assess(protocol, RESULTS if args.results is None else Path(args.results),
               runner_crash_log=None if args.runner_crash_log is None else Path(args.runner_crash_log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
