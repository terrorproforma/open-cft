"""Steady-state v4: preregistered grid-refinement check of the v2 base plateau (33.3 um / 1.4 ps / W 2.667e4, v1.3 closure,
v2.0.3 gates).

Stages (from ``modern/``, ``$env:PYTHONPATH="$PWD\\src;$PWD"``)::

    python -m experiments.pic2d_cft_steady_state_v4.run preflight            # real field + mesh + memory + ms/step -> preflight.json
    python -m experiments.pic2d_cft_steady_state_v4.run shakedown            # short real-input run through finalize + assess -> shakedown.json
    python -m experiments.pic2d_cft_steady_state_v4.run launch [--expect-commit SHA] [--resume]   # clean worktree, exclusive lock, run
    python -m experiments.pic2d_cft_steady_state_v4.run status
    python -m experiments.pic2d_cft_steady_state_v4.run finalize [...]       # only for an externally stopped run (shared runner)
    python -m experiments.pic2d_cft_steady_state_v4.run assess               # predeclared acceptance (a)-(d) -> results/assessment.json

The stepping itself is the shared runner ``experiments.pic2d_cft_steady_state_v1.run`` (models v1.2-v2.1); this module adds the
preregistration discipline the PIC development runs did not have: a preflight on the real inputs, a shakedown of every stage on
a short real run, a launch that refuses a dirty worktree / an unexpected commit / an existing execution lock, and the predeclared
assessment against ``protocol.reference_run.quantities``.  Detached launch (PowerShell, from ``modern/``)::

    $res = "experiments\\pic2d_cft_steady_state_v4\\results"; New-Item -ItemType Directory -Force $res | Out-Null
    Start-Process python -ArgumentList "-u","-m","experiments.pic2d_cft_steady_state_v4.run","launch","--expect-commit","<prereg sha>" `
        -WorkingDirectory $PWD -WindowStyle Hidden -RedirectStandardOutput "$res\\run.log" -RedirectStandardError "$res\\run.err"
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import ctypes.wintypes
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
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

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
PROTOCOL_PATH = HERE / "protocol.json"
RESULTS = HERE / "results"
PREFLIGHT_PATH = HERE / "preflight.json"
SHAKEDOWN_PATH = HERE / "shakedown.json"
LOCK_NAME = "execution-lock.json"
LOCK_SCHEMA = "cft-revival.pic2d-steady-state-execution-lock/1.0.0"
ASSESSMENT_SCHEMA = "cft-revival.pic2d-cft-steady-state-v4.assessment/1.0.0"
REFERENCE_RESULTS = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results"

# shakedown: the real protocol with only the cadences shrunk (every gate, the grid, dt, W, field and seed are the real ones)
SHAKEDOWN_OVERRIDES = {
    "series_interval_steps": 200, "device_sync_steps": 200, "checkpoint_every_steps": 4000, "averaging_window_steps": 40000,
    "frame_cadence_steps": 2000, "peak_debye_window_steps": 40000, "peak_debye_window_snapshot_steps": 4000,
    "residual_window_steps": 40000, "max_steps": 100000,
}


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return runner.load_protocol(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def peak_working_set_bytes() -> int | None:
    """Peak resident set of this process (Windows: GetProcessMemoryInfo; elsewhere resource.getrusage)."""

    if platform.system() == "Windows":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        for library, name in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"), (ctypes.windll.psapi, "GetProcessMemoryInfo")):
            function = getattr(library, name, None)
            if function is None:
                continue
            function.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(Counters), ctypes.wintypes.DWORD]
            function.restype = ctypes.wintypes.BOOL
            if function(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, OSError, AttributeError):  # pragma: no cover
        return None


def device_memory(device: str = "cuda:0") -> dict[str, int] | None:
    """Total / free device memory from Warp (None when Warp or the device is unavailable)."""

    try:
        import warp as wp

        wp.init()
        dev = wp.get_device(device)
        return {"total_bytes": int(dev.total_memory), "free_bytes": int(dev.free_memory)}
    except (ImportError, RuntimeError, AttributeError, KeyError, ValueError):
        return None


# -- preflight -------------------------------------------------------------------------------------------------------------

def _time_steps(sim: Simulation, steps: int, *, warmup: int) -> dict[str, float]:
    """Wall time per step over ``steps`` after ``warmup`` steps (graph capture, first allocations) on the live simulation.

    The window accumulation is ON from the first step (as in the runner: every production step deposits the diagnostic
    moments and every series record reads the window sums), so the measured step is the production step.
    """

    start = sim.backend.step_index
    sim.run(warmup, accumulate_from_step=start)
    t0 = time.perf_counter()
    sim.run(steps, accumulate_from_step=start)
    elapsed = time.perf_counter() - t0
    return {"steps": steps, "seconds": elapsed, "ms_per_step": 1e3 * elapsed / steps, "accumulation": True}


def preflight(protocol: dict[str, Any], *, backend: str = "warp-cuda", timing_steps: int = 2000, loaded_seed_density: float = 1.75e17,
              output: Path = PREFLIGHT_PATH, log: Callable[[str], None] = lambda text: print(text, flush=True)) -> dict[str, Any]:
    """Real inputs (P2 field on the refined grid, mesh, cross sections), factorisation, memory and ms/step at two loads.

    ``loaded_seed_density`` seeds a synthetic plasma at the expected plateau load (2 x 1.75e17 x 3.44e-7 / 2.667e4 = 4.5 M
    macro-particles) so the loaded step cost is measured, not extrapolated.  Non-evidentiary: nothing here is a result.
    """

    config = runner.build_config(protocol, backend=backend)
    grid = config.grid
    record: dict[str, Any] = {
        "schema_version": "cft-revival.pic2d-cft-steady-state-v4.preflight/1.0.0", "utc": utc_now(), "git_head": runner.git_head(),
        "protocol_sha256": runner._file_sha256(PROTOCOL_PATH), "config_sha256": artifacts.config_identity(config), "backend": backend,
        "host": socket.gethostname(), "python": sys.version.split()[0], "non_evidentiary": True,
        "grid": {"cells": list(grid.cell_shape), "nodes": list(grid.node_shape), "dr_m": grid.dr_m, "dz_m": grid.dz_m,
                 "dr_over_dz": grid.dr_m / grid.dz_m},
        "dt_s": config.dt_s, "macro_weight": config.macro_weight,
    }
    t0 = time.perf_counter()
    field_map, cross_sections = runner.load_inputs(config, None, None, protocol=protocol)
    # sha256 is the content hash of the sampled arrays (bitwise identity on THIS platform, provenance only); the
    # platform-independent binding the checkpoints verify is source_sha256 (P2 bundle file hashes + grid)
    record["field"] = {"sha256": field_map.sha256, "source_sha256": field_map.source_sha256, "max_b_t": field_map.max_b_t,
                       "seconds": time.perf_counter() - t0, "evidence": field_map.to_dict().get("evidence") or field_map.to_dict().get("source")}
    record["cross_sections_sha256"] = cross_sections.payload_sha256 if cross_sections is not None else None
    masks = build_mesh_masks(grid)
    record["mesh"] = masks.to_dict()
    before = device_memory()
    t0 = time.perf_counter()
    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend, step_graph=runner.step_graph_flag(protocol))
    record["factorisation_seconds"] = time.perf_counter() - t0
    record["stability_gate"] = sim.stability.to_dict()
    record["v1_4_options"] = sim.to_provenance()["v1_4_options"]
    seed_state = sim.state
    record["seed_particles"] = {"electrons": seed_state.electrons.count, "ions": seed_state.ions.count}
    log(f"[preflight] grid {grid.cell_shape} dr {grid.dr_m*1e6:.3f} um dz {grid.dz_m*1e6:.3f} um; field {field_map.sha256[:12]} max |B| "
        f"{field_map.max_b_t:.3f} T; mesh {masks.to_dict()['plasma_cells']} plasma cells; factorisation {record['factorisation_seconds']:.1f} s; "
        f"seed {seed_state.electrons.count} e-")
    timing_seed = _time_steps(sim, timing_steps, warmup=200)
    after = device_memory()
    timing_seed["electrons_after"] = sim.state.electrons.count
    timing_seed["ions_after"] = sim.state.ions.count
    timing_seed["step_graph"] = sim.step_graph_state()
    record["timing_seed_load"] = timing_seed
    record["last_series_record"] = sim.series[-1].to_dict() if sim.series else None
    log(f"[preflight] seed load: {timing_seed['ms_per_step']:.3f} ms/step over {timing_steps} steps ({sim.state.electrons.count} e-)")
    # loaded step: the same configuration seeded at the expected plateau density (synthetic, timing only)
    loaded = copy.deepcopy(protocol)
    loaded["operating_point"]["seed_plasma_density_per_m3"] = loaded_seed_density
    loaded_config = runner.build_config(loaded, backend=backend)
    del sim
    sim2 = Simulation(loaded_config, field_map, cross_sections=cross_sections, backend=backend, step_graph=runner.step_graph_flag(protocol))
    loaded_state = sim2.state
    timing_loaded = _time_steps(sim2, timing_steps, warmup=200)
    after_loaded = device_memory()
    timing_loaded.update({"seed_density_per_m3": loaded_seed_density, "electrons": loaded_state.electrons.count, "ions": loaded_state.ions.count,
                          "electrons_after": sim2.state.electrons.count, "ions_after": sim2.state.ions.count, "step_graph": sim2.step_graph_state()})
    record["timing_plateau_load"] = timing_loaded
    log(f"[preflight] plateau load: {timing_loaded['ms_per_step']:.3f} ms/step over {timing_steps} steps ({loaded_state.electrons.count} e- + "
        f"{loaded_state.ions.count} i)")
    record["memory"] = {
        "device_before": before, "device_after_seed_run": after, "device_after_loaded_run": after_loaded,
        "device_used_by_seed_run_bytes": None if before is None or after is None else before["free_bytes"] - after["free_bytes"],
        "device_used_by_loaded_run_bytes": None if before is None or after_loaded is None else before["free_bytes"] - after_loaded["free_bytes"],
        "host_peak_working_set_bytes": peak_working_set_bytes(),
    }
    budget = runner.protocol_budget(protocol)
    transit = float(budget["ion_transit_time_s"])
    steps_3 = 3.0 * transit / config.dt_s
    steps_v2_plateau = 7.68e-6 / config.dt_s
    ms = timing_loaded["ms_per_step"]
    ms_seed = timing_seed["ms_per_step"]
    per_m = (ms - ms_seed) / max((loaded_state.electrons.count + loaded_state.ions.count - seed_state.electrons.count - seed_state.ions.count) / 1e6, 1e-9)
    record["projection"] = {
        "steps_per_transit": transit / config.dt_s, "steps_to_3_transits": steps_3, "steps_to_v2_plateau_time_7_68_us": steps_v2_plateau,
        "hours_to_3_transits_at_plateau_load": steps_3 * ms / 3.6e6, "hours_to_v2_plateau_time_at_plateau_load": steps_v2_plateau * ms / 3.6e6,
        "hours_to_3_transits_at_seed_load": steps_3 * ms_seed / 3.6e6,
        "ms_per_step_per_million_particles": per_m,
        "wall_budget_seconds": float(protocol["stopping_rule"]["wall_budget_seconds"]),
        "budget_over_3_transit_time": float(protocol["stopping_rule"]["wall_budget_seconds"]) / max(steps_3 * ms / 1e3, 1e-9),
        "note": "the particle count grows from the seed (1.3 M) to the plateau (~4.5 M) over the first ~1.5 us, so the wall time to 3 "
                "transits lies between the two projections, close to the plateau-load one",
    }
    # the expected gate reading at the v2 base peak on this grid
    ref = protocol["reference_run"]["quantities"]
    lam = np.sqrt(EPSILON_0_F_PER_M * ref["t_e_peak_window_ev"] * EV_J / (ref["peak_n_e_window_per_m3"] * ELEMENTARY_CHARGE_C**2))
    omega = np.sqrt(ref["peak_n_e_window_per_m3"] * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ELECTRON_MASS_KG))
    record["expected_at_v2_base_peak"] = {
        "debye_length_m": float(lam), "cells_per_debye": float(max(grid.dr_m, grid.dz_m) / lam), "omega_pe_dt": float(omega * config.dt_s),
        "soft_gate": config.peak_debye_gate.soft_cells_per_debye, "hard_gate": config.peak_debye_gate.max_cells_per_debye,
    }
    artifacts.write_canonical_json(output, record)
    log(f"[preflight] projection: {record['projection']['hours_to_3_transits_at_plateau_load']:.1f} h to 3 transits at the plateau load; "
        f"budget/3-transit {record['projection']['budget_over_3_transit_time']:.2f}; written {output}")
    return record


# -- shakedown -------------------------------------------------------------------------------------------------------------

def shakedown_protocol(protocol: dict[str, Any], overrides: dict[str, Any] = SHAKEDOWN_OVERRIDES) -> dict[str, Any]:
    """The real protocol with every cadence shrunk (NON-EVIDENTIARY): the grid, dt, W, field, seed and gate thresholds are untouched."""

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
        "schema_version": "cft-revival.pic2d-cft-steady-state-v4.shakedown/1.0.0", "utc": utc_now(), "git_head": runner.git_head(),
        "non_evidentiary": True, "overrides": SHAKEDOWN_OVERRIDES, "results_dir": results.name, "run_seconds": run_seconds,
        "steps_completed": summary["steps_completed"], "stop_reason": summary["stop_reason"], "ms_per_step": summary["ms_per_step_this_session"],
        "final_counts": summary["final_counts"], "frames": summary["artifacts"]["frames"]["count"] if summary["artifacts"]["frames"] else 0,
        "peak_debye_window": {
            "records": len(windows), "enforced_records": len(enforced),
            "last": windows[-1] if windows else None,
            "max_cells_per_debye_enforced": max((w["cells_per_debye"] for w in enforced), default=None),
        },
        "windowed_residual": {
            "records_with_complete_window": len(complete),
            "last": None if not complete else {k: complete[-1][k] for k in complete[-1] if k.startswith("windowed") or k == "energy_residual_over_electrode_work"},
        },
        "plateau_keys": sorted(summary["plateau"]) if summary["plateau"] else None,
        "summary_keys_present": {key: (summary.get(key) is not None) for key in ("plateau", "grid_heating_triad", "peak_node_debye", "neutral_inventory", "ledger")},
        "assessment": {k: assessment[k] for k in ("verdict", "a_plateau", "b_residual_power", "c_convergence", "reference_consistency")},
        "artifacts": {k: summary["artifacts"][k] for k in ("maps_npz_sha256", "series_npz_sha256")},
        "gate_not_inert_check": {
            "peak_window_enforced_at_least_once": bool(enforced), "peak_window_resolved_nodes_last": windows[-1]["resolved_nodes"] if windows else None,
            "residual_window_completed_at_least_once": bool(complete),
        },
    }
    artifacts.write_canonical_json(output, record)
    log(f"[shakedown] {summary['steps_completed']} steps, {summary['stop_reason']}, {summary['ms_per_step_this_session']:.2f} ms/step, "
        f"{record['frames']} frames, peak window enforced in {len(enforced)}/{len(windows)} records (max {record['peak_debye_window']['max_cells_per_debye_enforced']}), "
        f"residual window complete in {len(complete)} records; assessment verdict {assessment['verdict']}; written {output}")
    return record


# -- launch -------------------------------------------------------------------------------------------------------------------

def worktree_status(cwd: Path = REPOSITORY_ROOT) -> list[str]:
    return [line for line in git("status", "--porcelain", "--untracked-files=normal", cwd=cwd).splitlines() if line.strip()]


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
           resume: bool = False, allow_dirty: bool = False, wall_budget_seconds: float | None = None,
           log: Callable[[str], None] = lambda text: print(text, flush=True)) -> Path:
    """Preregistered execution: clean worktree, expected commit, exclusive lock, then the shared runner (blocking)."""

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
    payload = {
        "schema_version": LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "commit": head, "protocol_sha256": protocol_sha,
        "config_sha256": artifacts.config_identity(runner.build_config(protocol, backend=backend)), "backend": backend,
        "command": " ".join(sys.argv), "host": socket.gethostname(), "pid": os.getpid(), "acquired_at_utc": utc_now(),
        "clean_worktree_attested": not dirty, "worktree": str(REPOSITORY_ROOT), "immutable": True,
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
        log(f"[launch] execution lock acquired: commit {head[:12]}, protocol {protocol_sha[:12]}, clean worktree {not dirty}")
    return runner.run_steady_state(protocol, results, backend=backend, protocol_path=PROTOCOL_PATH, wall_budget_seconds=wall_budget_seconds, log=log)


# -- assessment ----------------------------------------------------------------------------------------------------------------

def _peak_from_maps(maps_path: Path) -> dict[str, float]:
    maps = artifacts.read_npz(maps_path)
    n = np.asarray(maps["n_e_per_m3"])
    t = np.asarray(maps["t_e_ev"])
    flat = int(np.nanargmax(n))
    i, j = np.unravel_index(flat, n.shape)
    return {"peak_n_e_window_per_m3": float(n[i, j]), "t_e_peak_window_ev": float(t[i, j]), "node": [int(i), int(j)]}


def reference_quantities_from_files(results: Path = REFERENCE_RESULTS) -> dict[str, float] | None:
    """Recompute the pinned reference numbers from the v2 base artifacts (consistency check of protocol.reference_run.quantities)."""

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


def run_quantities(results: Path) -> dict[str, Any]:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    peak = _peak_from_maps(results / "maps.npz")
    triad = summary.get("grid_heating_triad") or {}
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
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
        "git_head": summary.get("git_head"), "protocol_sha256": summary.get("protocol_sha256"),
        "config_sha256": (summary.get("provenance") or {}).get("config_sha256"),
    }


def assess(protocol: dict[str, Any], results: Path = RESULTS, *, output: Path | None = None, reference_check: bool = True,
           log: Callable[[str], None] = lambda text: print(text, flush=True)) -> dict[str, Any]:
    """Predeclared acceptance (protocol.stopping_rule.acceptance) against protocol.reference_run.quantities."""

    if not (results / "summary.json").is_file():
        raise PIC2DValidationError(f"{results} has no summary.json to assess")
    acceptance = protocol["stopping_rule"]["acceptance"]
    reference = protocol["reference_run"]["quantities"]
    tolerances = {k: float(v) for k, v in acceptance["c_convergence_tolerances"].items() if k != "note"}
    run = run_quantities(results)
    a_plateau = run["stop_reason"] == "plateau_reached_after_min_transit_times"
    windowed = run["windowed_residual_over_electrode_work"]
    b_ok = windowed is not None and bool(run["windowed_residual_window_complete"]) and windowed < 0.02
    convergence: dict[str, Any] = {}
    all_within = True
    for key, tolerance in tolerances.items():
        ref = float(reference[key])
        value = float(run[key])
        rel = (value - ref) / abs(ref) if ref != 0.0 else float("inf")
        within = abs(rel) <= tolerance
        all_within = all_within and within
        convergence[key] = {"reference": ref, "value": value, "relative_difference": rel, "tolerance": tolerance, "within": within}
    if a_plateau and b_ok and all_within:
        verdict = "converged"
    elif a_plateau and b_ok:
        verdict = "resolution_limited"
    elif a_plateau:
        verdict = "refinement_heating"
    else:
        verdict = "no_plateau"
    consistency = None
    if reference_check:
        recomputed = reference_quantities_from_files()
        if recomputed is not None:
            consistency = {key: {"pinned": float(reference[key]), "recomputed": float(recomputed[key]),
                                 "agree": abs(float(recomputed[key]) - float(reference[key])) <= 1e-9 * max(abs(float(reference[key])), 1e-300)}
                           for key in recomputed}
            if not all(entry["agree"] for entry in consistency.values()):
                raise PIC2DValidationError("protocol.reference_run.quantities disagree with the v2 base artifacts on disk: " + json.dumps(
                    {k: v for k, v in consistency.items() if not v["agree"]}))
    record = {
        "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "results_dir": results.name,
        "git_head_now": runner.git_head(), "run": run, "reference": reference, "reference_consistency": consistency,
        "a_plateau": {"passed": a_plateau, "stop_reason": run["stop_reason"], "ion_transit_times": run["ion_transit_times"],
                      "plateau": run["plateau"], "rule": acceptance["a_plateau"]},
        "b_residual_power": {"passed": b_ok, "windowed_residual_over_electrode_work": windowed, "window_complete": run["windowed_residual_window_complete"],
                             "bound": 0.02, "one_sided": True, "cumulative_witness": run["cumulative_residual_over_electrode_work"], "rule": acceptance["b_residual_power"]},
        "c_convergence": {"all_within": all_within, "quantities": convergence, "rule": acceptance["c_convergence_tolerances"]["note"]},
        "d_reclassification": acceptance["d_reclassification"][verdict],
        "verdict": verdict,
        "peak_debye_window": {"cells_per_debye_last": run["cells_per_debye_window_last"], "trailing_mean": run["cells_per_debye_window_trailing_mean"],
                              "soft_ok": run["peak_debye_soft_ok"]},
        "claim_boundary": protocol["claim_boundary"],
    }
    artifacts.write_canonical_json(output or (results / "assessment.json"), record)
    log(f"[assess] {results.name}: verdict {verdict} (a {a_plateau}, b {b_ok} [{windowed}], c all_within {all_within}); "
        + ", ".join(f"{k} {v['relative_difference']*100:+.1f}%" for k, v in convergence.items()))
    return record


# -- CLI ------------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--backend", default="warp-cuda")
    pre.add_argument("--timing-steps", type=int, default=2000)
    pre.add_argument("--loaded-seed-density", type=float, default=1.75e17)
    shake = sub.add_parser("shakedown")
    shake.add_argument("--backend", default="warp-cuda")
    la = sub.add_parser("launch")
    la.add_argument("--backend", default="warp-cuda")
    la.add_argument("--expect-commit", default=None)
    la.add_argument("--resume", action="store_true")
    la.add_argument("--allow-dirty", action="store_true", help="development only; never for the preregistered execution")
    la.add_argument("--wall-budget-seconds", type=float, default=None)
    sub.add_parser("status")
    fin = sub.add_parser("finalize")
    fin.add_argument("--backend", default="warp-cuda")
    fin.add_argument("--stop-reason", default="finalized_from_checkpoint")
    fin.add_argument("--allow-refinalize", action="store_true")
    fin.add_argument("--recover-runner-stop", action="store_true")
    ass = sub.add_parser("assess")
    ass.add_argument("--results", default=None)
    args = parser.parse_args(argv)
    protocol = load_protocol()
    if args.command == "preflight":
        preflight(protocol, backend=args.backend, timing_steps=args.timing_steps, loaded_seed_density=args.loaded_seed_density)
    elif args.command == "shakedown":
        shakedown(protocol, backend=args.backend)
    elif args.command == "launch":
        launch(protocol, backend=args.backend, expect_commit=args.expect_commit, resume=args.resume, allow_dirty=args.allow_dirty,
               wall_budget_seconds=args.wall_budget_seconds)
    elif args.command == "status":
        print(json.dumps(runner.status(RESULTS, protocol), indent=1))
    elif args.command == "finalize":
        runner.finalize(protocol, RESULTS, backend=args.backend, stop_reason=args.stop_reason, protocol_path=PROTOCOL_PATH,
                        allow_refinalize=args.allow_refinalize, recover_runner_stop=args.recover_runner_stop)
    else:
        assess(protocol, RESULTS if args.results is None else Path(args.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
