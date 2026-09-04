"""Anomalous cross-field transport v1: the preregistered Bohm alpha-series on the 33 um reference plateau (roadmap R1, model v2.1.0).

Three cases (``protocols/alpha-1over64.json``, ``alpha-1over16.json``, ``alpha-0.345.json``) = the ss-v4 protocol with the v2.1.0
perpendicular-rotation Bohm closure at ``alpha``, the v2.0.6 gates and K = 5; the alpha = 0 point of the series is the RECORDED ss-v4
plateau (``pic2d_cft_steady_state_v4/results``, 0d228ad2), which fails its own acceptance (b) at +2.46 % on the corrected ledger.

Stages (from ``modern/`` with ``PYTHONPATH=src:.``)::

    python -m experiments.pic2d_anomalous_transport_v1.run compose [--budget-from-preflight]     # (re)write protocols/*.json + protocol.json
    python -m experiments.pic2d_anomalous_transport_v1.run preflight --case alpha-1over16 [--gpu-timing]   # -> preflight-<case>.json
    python -m experiments.pic2d_anomalous_transport_v1.run shakedown --case alpha-1over16                  # 100k steps -> finalize -> assess -> shakedown-<case>.json
    python -m experiments.pic2d_anomalous_transport_v1.run launch --case alpha-1over16 --expect-commit SHA [--require-mps] [--resume]
    python -m experiments.pic2d_anomalous_transport_v1.run status
    python -m experiments.pic2d_anomalous_transport_v1.run finalize --case alpha-1over16 [...]          # externally stopped run only
    python -m experiments.pic2d_anomalous_transport_v1.run assess --case alpha-1over16                  # per-case verdict -> results/<case>/assessment.json
    python -m experiments.pic2d_anomalous_transport_v1.run assess --series                              # trend verdict -> results/series-assessment.json

The stepping is the shared runner ``experiments.pic2d_cft_steady_state_v1.run``; the stages follow the v4 / v5 preregistration
discipline (clean worktree, expected commit, sealed protocol == recomposition, O_EXCL execution lock, preflight + shakedown records).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d import artifacts
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import ELEMENTARY_CHARGE_C, PIC2DValidationError
from cft_revival.pic2d.simulation import Simulation
from experiments.pic2d_anomalous_transport_v1 import protocol as protocol_module
from experiments.pic2d_anomalous_transport_v1.protocol import (
    CASES,
    CUSP_HALF_WIDTH_M,
    CUSP_PLANES_M,
    EXPERIMENT_ID,
    HYPOTHESES,
    LAUNCH_PRIORITY,
    MONOTONE_QUANTITIES,
    PARTICLE_BAND,
    REFERENCE_CASE,
    STEPS_TO_3_TRANSITS,
    compose_campaign,
    compose_case_protocol,
    load_campaign,
    load_case_protocol,
    protocol_sha256,
    write_sealed_protocols,
)
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v4.run import (
    _time_steps,
    device_memory,
    git,
    peak_working_set_bytes,
    utc_now,
    worktree_status,
)
from experiments.pic2d_cft_steady_state_v5.run import (
    _peak_from_maps,
    acquire_lock,
    gpu_load_snapshot,
)

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
RESULTS = HERE / "results"
LOCK_NAME = "execution-lock.json"
LOCK_SCHEMA = "cft-revival.pic2d-steady-state-execution-lock/1.0.0"
ASSESSMENT_SCHEMA = "cft-revival.pic2d-anomalous-transport-v1.assessment/1.0.0"
SERIES_ASSESSMENT_SCHEMA = "cft-revival.pic2d-anomalous-transport-v1.series-assessment/1.0.0"
PREFLIGHT_SCHEMA = "cft-revival.pic2d-anomalous-transport-v1.preflight/1.0.0"
SHAKEDOWN_SCHEMA = "cft-revival.pic2d-anomalous-transport-v1.shakedown/1.0.0"
REFERENCE_RESULTS = protocol_module.V4_RESULTS
QUANTITY_KEYS = ("discharge_current_a", "exit_ion_beam_a", "ionization_rate_per_s", "gross_utilisation", "neutral_density_per_m3", "peak_n_e_window_per_m3", "t_e_peak_window_ev")

# shakedown: the real protocol with only the cadences shrunk (every gate, the grid, dt, W, alpha, field and seed are the real ones)
SHAKEDOWN_OVERRIDES = {
    "series_interval_steps": 200, "device_sync_steps": 200, "checkpoint_every_steps": 4000, "averaging_window_steps": 40000,
    "frame_cadence_steps": 2000, "peak_debye_window_steps": 40000, "peak_debye_window_snapshot_steps": 4000,
    "residual_window_steps": 40000, "max_steps": 100000,
}


def preflight_path(case: str) -> Path:
    return HERE / f"preflight-{case}.json"


def shakedown_path(case: str) -> Path:
    return HERE / f"shakedown-{case}.json"


def case_results(case: str, results: Path = RESULTS) -> Path:
    return results / case


def _log(text: str) -> None:
    print(text, flush=True)


def compute_apps() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else []

    def _mib(text: str) -> float:      # Windows reports "[N/A]" for per-process memory
        try:
            return float(text)
        except ValueError:
            return float("nan")

    return [{"pid": int(r[0]), "used_memory_mib": _mib(r[1])} for r in rows if len(r) >= 2 and r[0].isdigit()]


# -- compose ---------------------------------------------------------------------------------------------------------------

def compose(*, budget_from_preflight: bool = False, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """Write the sealed per-case protocols and the campaign protocol; ``--budget-from-preflight`` derives each budget from its preflight record."""

    cases: dict[str, dict[str, Any]] = {}
    budgets: dict[str, dict[str, Any]] = {}
    for case in CASES:
        budget = None
        note = None
        path = preflight_path(case)
        if budget_from_preflight and path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            derived = record.get("budget_derivation")
            if derived is not None:
                budget = float(derived["wall_budget_seconds"])
                note = derived["note"]
        cases[case] = compose_case_protocol(case, wall_budget_seconds=budget, budget_note=note)
        budgets[case] = {"wall_budget_seconds": cases[case]["stopping_rule"]["wall_budget_seconds"], "note": cases[case]["stopping_rule"]["wall_budget_note"],
                         "from_preflight": budget is not None}
    campaign = compose_campaign(cases, budgets=budgets)
    written = write_sealed_protocols(cases, campaign)
    for path in written:
        log(f"[compose] {path.relative_to(MODERN).as_posix()} {protocol_sha256(json.loads(path.read_text(encoding='utf-8')))[:12]}")
    return campaign


def verify_sealed(case: str) -> dict[str, Any]:
    """The sealed protocol must equal its recomposition (with its own recorded budget) and the campaign's listed hash."""

    sealed = load_case_protocol(case)
    recomposed = compose_case_protocol(case, wall_budget_seconds=sealed["stopping_rule"]["wall_budget_seconds"], budget_note=sealed["stopping_rule"]["wall_budget_note"])
    if canonical_bytes(sealed) != canonical_bytes(recomposed):
        raise PIC2DValidationError(f"sealed protocol for {case} differs from its recomposition; refusing")
    campaign = load_campaign()
    key = f"modern/experiments/pic2d_anomalous_transport_v1/protocols/{case}.json"
    on_disk = protocol_sha256(sealed)
    if campaign["sealed_protocols"].get(key) != on_disk:
        raise PIC2DValidationError(f"campaign protocol.json lists {campaign['sealed_protocols'].get(key)} for {case}, on disk {on_disk}")
    return sealed


# -- preflight --------------------------------------------------------------------------------------------------------------

def preflight(case: str, *, backend: str = "warp-cuda", timing_steps: int = 2000, loaded_seed_density: float = 1.75e17, gpu_timing: bool = True,
              log: Callable[[str], None] = _log) -> dict[str, Any]:
    """Real inputs on the launch box: field, mesh, factorisation, memory, ms/step at the seed and plateau loads; the budget derivation. Non-evidentiary."""

    protocol = load_case_protocol(case)
    config = runner.build_config(protocol, backend=backend)
    grid = config.grid
    gpu_before = gpu_load_snapshot()
    apps = compute_apps()
    own = os.getpid()
    others = [a for a in apps if a["pid"] != own and a["used_memory_mib"] > 200.0]
    record: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA, "utc": utc_now(), "git_head": runner.git_head(), "case": case, "alpha": protocol["campaign"]["alpha"],
        "protocol_sha256": protocol_sha256(protocol), "config_sha256": artifacts.config_identity(config), "backend": backend,
        "host": socket.gethostname(), "python": sys.version.split()[0], "non_evidentiary": True, "gpu_load_before": gpu_before,
        "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), "concurrent_mps_clients": len(others), "concurrent_mps_client_pids": [a["pid"] for a in others],
        "grid": {"cells": list(grid.cell_shape), "nodes": list(grid.node_shape), "dr_m": grid.dr_m, "dz_m": grid.dz_m},
        "dt_s": config.dt_s, "macro_weight": config.macro_weight, "anomalous": config.anomalous.to_dict() if config.anomalous is not None else None,
        "moment_sample_interval": config.moment_sample_interval,
        "peak_debye_floor": config.peak_debye_gate.to_dict() if config.peak_debye_gate is not None else None,
    }
    t0 = time.perf_counter()
    field_map, cross_sections = runner.load_inputs(config, None, None, protocol=protocol)
    record["field"] = {"sha256": field_map.sha256, "source_sha256": getattr(field_map, "source_sha256", None), "max_b_t": field_map.max_b_t, "seconds": time.perf_counter() - t0}
    record["cross_sections_sha256"] = cross_sections.payload_sha256 if cross_sections is not None else None
    masks = build_mesh_masks(grid)
    record["mesh"] = masks.to_dict()
    # the closure's per-step event probability at the field extremes (the hook's own admissibility statement)
    alpha = float(protocol["campaign"]["alpha"])
    omega_max = ELEMENTARY_CHARGE_C / 9.1093837015e-31 * field_map.max_b_t
    record["anomalous_rates"] = {"nu_an_at_max_b_per_s": alpha * omega_max, "nu_an_dt_at_max_b": alpha * omega_max * config.dt_s,
                                 "omega_ce_dt_at_max_b": omega_max * config.dt_s, "d_perp_factor": alpha / (1.0 + alpha**2),
                                 "note": "nu_an dt << 1 keeps the per-step Poisson probability far from saturation; the ratio to omega_ce dt is alpha"}
    if not gpu_timing:
        artifacts.write_canonical_json(preflight_path(case), record)
        log(f"[preflight] {case}: inputs + mesh only (no GPU timing); written {preflight_path(case)}")
        return record
    before = device_memory()
    t0 = time.perf_counter()
    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend, step_graph=runner.step_graph_flag(protocol))
    record["factorisation_seconds"] = time.perf_counter() - t0
    record["stability_gate"] = sim.stability.to_dict()
    record["v1_4_options"] = sim.to_provenance()["v1_4_options"]
    seed_state = sim.state
    record["seed_particles"] = {"electrons": seed_state.electrons.count, "ions": seed_state.ions.count}
    log(f"[preflight] {case} alpha {alpha:.5g}: grid {grid.cell_shape}; field {field_map.sha256[:12]} max |B| {field_map.max_b_t:.3f} T; "
        f"factorisation {record['factorisation_seconds']:.1f} s; seed {seed_state.electrons.count} e-; MPS clients before: {len(others)}; GPU before: {gpu_before}")
    timing_seed = _time_steps(sim, timing_steps, warmup=200)
    after = device_memory()
    timing_seed.update({"electrons_after": sim.state.electrons.count, "ions_after": sim.state.ions.count, "step_graph": sim.step_graph_state(),
                        "anomalous_events": sim.state.cumulative.get("anomalous")})
    record["timing_seed_load"] = timing_seed
    record["last_series_record"] = sim.series[-1].to_dict() if sim.series else None
    log(f"[preflight] seed load: {timing_seed['ms_per_step']:.3f} ms/step over {timing_steps} steps ({sim.state.electrons.count} e-, {timing_seed['anomalous_events']} anomalous events)")
    loaded = copy.deepcopy(protocol)
    loaded["operating_point"]["seed_plasma_density_per_m3"] = loaded_seed_density
    loaded_config = runner.build_config(loaded, backend=backend)
    del sim
    sim2 = Simulation(loaded_config, field_map, cross_sections=cross_sections, backend=backend, step_graph=runner.step_graph_flag(protocol))
    loaded_state = sim2.state
    timing_loaded = _time_steps(sim2, timing_steps, warmup=200)
    after_loaded = device_memory()
    timing_loaded.update({"seed_density_per_m3": loaded_seed_density, "electrons": loaded_state.electrons.count, "ions": loaded_state.ions.count,
                          "electrons_after": sim2.state.electrons.count, "ions_after": sim2.state.ions.count, "step_graph": sim2.step_graph_state(),
                          "anomalous_events": sim2.state.cumulative.get("anomalous")})
    record["timing_plateau_load"] = timing_loaded
    log(f"[preflight] plateau load: {timing_loaded['ms_per_step']:.3f} ms/step over {timing_steps} steps ({loaded_state.electrons.count} e- + {loaded_state.ions.count} i)")
    record["memory"] = {
        "device_before": before, "device_after_seed_run": after, "device_after_loaded_run": after_loaded,
        "device_used_by_loaded_run_bytes": None if before is None or after_loaded is None else before["free_bytes"] - after_loaded["free_bytes"],
        "host_peak_working_set_bytes": peak_working_set_bytes(),
    }
    ms = timing_loaded["ms_per_step"]
    hours_3 = STEPS_TO_3_TRANSITS * ms / 3.6e6
    budget = float(np.ceil(1.5 * STEPS_TO_3_TRANSITS * ms / 1e3 / 600.0) * 600.0)      # 1.5 x, rounded up to 10 min
    record["projection"] = {"steps_to_3_transits": STEPS_TO_3_TRANSITS, "hours_to_3_transits_at_plateau_load": hours_3,
                            "hours_to_3_transits_at_seed_load": STEPS_TO_3_TRANSITS * timing_seed["ms_per_step"] / 3.6e6,
                            "ms_per_step_per_million_particles": (ms - timing_seed["ms_per_step"]) / max((loaded_state.electrons.count + loaded_state.ions.count
                                                                                                         - seed_state.electrons.count - seed_state.ions.count) / 1e6, 1e-9)}
    record["budget_derivation"] = {"wall_budget_seconds": budget, "basis_ms_per_step": ms, "factor": 1.5,
                                   "note": f"{budget / 3600:.1f} h = 1.5 x {hours_3:.2f} h (launch-box plateau-load preflight {ms:.2f} ms/step with {len(others)} other MPS "
                                           f"clients x {STEPS_TO_3_TRANSITS} steps to 3 transits), rounded up to 10 min; preflight-{case}.json"}
    ref = protocol["reference_run"]["quantities"]
    record["expected_at_v4_peak"] = {"peak_n_e_per_m3": ref["peak_n_e_window_per_m3"], "t_e_peak_ev": ref["t_e_peak_window_ev"],
                                     "cells_per_debye": ref["cells_per_debye_at_peak_window"], "hypothesis": "lower at alpha > 0 (hypotheses peak_n_e / T_e signs)"}
    artifacts.write_canonical_json(preflight_path(case), record)
    log(f"[preflight] projection {hours_3:.2f} h to 3 transits at the plateau load; budget {budget / 3600:.1f} h; written {preflight_path(case)}")
    return record


# -- shakedown --------------------------------------------------------------------------------------------------------------

def shakedown_protocol(protocol: dict[str, Any], overrides: dict[str, Any] = SHAKEDOWN_OVERRIDES) -> dict[str, Any]:
    """The real protocol with every cadence shrunk (NON-EVIDENTIARY): grid, dt, W, alpha, field, seed and gate thresholds untouched."""

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


def shakedown(case: str, *, results: Path | None = None, backend: str = "warp-cuda", max_steps: int = SHAKEDOWN_OVERRIDES["max_steps"],
              log: Callable[[str], None] = _log) -> dict[str, Any]:
    protocol = load_case_protocol(case)
    results = HERE / "results-shakedown" / case if results is None else results
    if results.exists():
        shutil.rmtree(results)
    p = shakedown_protocol(protocol)
    results.mkdir(parents=True)
    shake_protocol_path = results / "protocol-shakedown.json"
    artifacts.write_canonical_json(shake_protocol_path, p)
    clients_before = compute_apps()
    t0 = time.perf_counter()
    summary_path = runner.run_steady_state(p, results, backend=backend, max_steps=max_steps, protocol_path=shake_protocol_path, log=log)
    run_seconds = time.perf_counter() - t0
    summary = artifacts.read_canonical_json(summary_path)
    assessment = assess_case(case, results=results, protocol=p, log=log, reference_check=True)
    series = assess_series(results_root=results.parent, cases_override={case: results}, log=log, output=results / "series-assessment.json")
    t1 = time.perf_counter()
    runner.finalize(p, results, backend=backend, stop_reason="shakedown_refinalize", protocol_path=shake_protocol_path, allow_refinalize=True)
    refinalize_seconds = time.perf_counter() - t1
    samples = [s for s in runner._read_jsonl(results / "status.jsonl") if "event" not in s]
    windows = [s["peak_node"]["window"] for s in samples if (s.get("peak_node") or {}).get("window") is not None]
    enforced = [w for w in windows if w.get("gate_enforced")]
    triads = [s["grid_heating_triad"] for s in samples if s.get("grid_heating_triad") is not None]
    complete = [t for t in triads if t.get("windowed_energy_residual_window_complete")]
    own = os.getpid()
    others = [a for a in clients_before if a["pid"] != own and a["used_memory_mib"] > 200.0]
    anomalous = _cumulative_anomalous(summary)
    record = {
        "schema_version": SHAKEDOWN_SCHEMA, "utc": utc_now(), "git_head": runner.git_head(), "non_evidentiary": True, "case": case, "alpha": protocol["campaign"]["alpha"],
        "host": socket.gethostname(), "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), "concurrent_mps_clients": len(others),
        "concurrent_mps_client_pids": [a["pid"] for a in others], "overrides": {**SHAKEDOWN_OVERRIDES, "max_steps": max_steps}, "results_dir": results.relative_to(HERE).as_posix(),
        "run_seconds": run_seconds, "refinalize_seconds": refinalize_seconds, "steps_completed": summary["steps_completed"], "stop_reason": summary["stop_reason"],
        "ms_per_step": summary["ms_per_step_this_session"], "final_counts": summary["final_counts"], "anomalous_events_cumulative": anomalous,
        "anomalous_collision_rate_per_s_last": (summary.get("final_series") or {}).get("currents_a", {}).get("anomalous_collision_rate_per_s"),
        "frames": summary["artifacts"]["frames"]["count"] if summary["artifacts"].get("frames") else 0,
        "peak_debye_window": {"records": len(windows), "enforced_records": len(enforced), "last": windows[-1] if windows else None,
                              "max_cells_per_debye_enforced": max((w["cells_per_debye"] for w in enforced), default=None),
                              "floor_kind": "accumulated_particle_steps" if windows and windows[-1].get("min_accumulated_macro_particle_steps_at_peak") else "mean_occupancy"},
        "windowed_residual": {"records_with_complete_window": len(complete),
                              "last": None if not complete else {k: complete[-1][k] for k in complete[-1] if k.startswith("windowed") or k == "energy_residual_over_electrode_work"}},
        "plateau_keys": sorted(summary["plateau"]) if summary.get("plateau") else None,
        "summary_keys_present": {key: (summary.get(key) is not None) for key in ("plateau", "grid_heating_triad", "peak_node_debye", "neutral_inventory", "ledger", "window_currents_a")},
        "assessment": {k: assessment[k] for k in ("verdict", "a_plateau", "b_residual_power", "reference_consistency")},
        "series_assessment": {"verdict": series["verdict"], "points_reached": series["points_reached"]},
        "artifacts": {k: summary["artifacts"].get(k) for k in ("maps_npz_sha256", "series_npz_sha256")},
        "gate_not_inert_check": {"peak_window_enforced_at_least_once": bool(enforced), "peak_window_resolved_nodes_last": windows[-1].get("resolved_nodes") if windows else None,
                                 "residual_window_completed_at_least_once": bool(complete), "anomalous_events_nonzero": bool(anomalous)},
    }
    artifacts.write_canonical_json(shakedown_path(case), record)
    log(f"[shakedown] {case}: {summary['steps_completed']} steps, {summary['stop_reason']}, {summary['ms_per_step_this_session']:.2f} ms/step, {record['frames']} frames, "
        f"{anomalous} anomalous events, peak window enforced {len(enforced)}/{len(windows)} (max {record['peak_debye_window']['max_cells_per_debye_enforced']}), "
        f"residual windows complete {len(complete)}; case verdict {assessment['verdict']}, series {series['verdict']}; written {shakedown_path(case)}")
    return record


# -- launch -----------------------------------------------------------------------------------------------------------------

def launch(case: str, *, results: Path | None = None, backend: str = "warp-cuda", expect_commit: str | None = None, resume: bool = False,
           allow_dirty: bool = False, require_mps: bool = False, wall_budget_seconds: float | None = None, log: Callable[[str], None] = _log) -> Path:
    """Preregistered execution of one case: clean worktree, expected commit, sealed == recomposed, exclusive lock, then the shared runner (blocking)."""

    if case not in CASES:
        raise PIC2DValidationError(f"unknown case {case!r}; cases: {sorted(CASES)}")
    results = case_results(case) if results is None else results
    head = git("rev-parse", "HEAD")
    if expect_commit is not None and not head.startswith(expect_commit):
        raise PIC2DValidationError(f"HEAD {head[:12]} is not the preregistration commit {expect_commit}")
    dirty = worktree_status()
    if dirty and not allow_dirty:
        raise PIC2DValidationError(f"worktree is not clean ({len(dirty)} entries, e.g. {dirty[0]!r}); the preregistered launch requires a clean checkout")
    protocol = verify_sealed(case)
    protocol_path = protocol_module.PROTOCOLS_DIR / f"{case}.json"
    relative = protocol_path.relative_to(REPOSITORY_ROOT).as_posix()
    if git("rev-parse", f"HEAD:{relative}") != git("hash-object", "--", str(protocol_path)):
        raise PIC2DValidationError(f"{relative} on disk differs from the committed blob at HEAD")
    if not preflight_path(case).is_file() or not any(shakedown_path(c).is_file() for c in CASES):
        raise PIC2DValidationError(f"preflight-{case}.json and a shakedown-<case>.json must exist (and be committed) before a preregistered launch")
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if require_mps and not (mps_pipe and Path(mps_pipe).exists()):
        raise PIC2DValidationError(f"--require-mps: CUDA_MPS_PIPE_DIRECTORY {mps_pipe!r} is not set or does not exist in this environment")
    protocol_sha = protocol_sha256(protocol)
    payload = {
        "schema_version": LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "case": case, "alpha": protocol["campaign"]["alpha"], "commit": head,
        "protocol_sha256": protocol_sha, "config_sha256": artifacts.config_identity(runner.build_config(protocol, backend=backend)), "backend": backend,
        "command": " ".join(sys.argv), "host": socket.gethostname(), "pid": os.getpid(), "acquired_at_utc": utc_now(), "clean_worktree_attested": not dirty,
        "worktree": str(REPOSITORY_ROOT), "immutable": True, "cuda_mps_pipe_directory": mps_pipe, "mps_required": require_mps,
        "concurrent_mps_clients_at_launch": [a for a in compute_apps() if a["pid"] != os.getpid() and a["used_memory_mib"] > 200.0],
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
        log(f"[launch] {case}: resuming under the existing lock (commit {head[:12]}, acquired {existing.get('acquired_at_utc')})")
    else:
        if runner.find_checkpoint(results) is not None:
            raise PIC2DValidationError(f"{results} already holds a checkpoint; use --resume for a new session under the same lock")
        acquire_lock(results, payload)
        log(f"[launch] {case} (alpha {protocol['campaign']['alpha']:.5g}): execution lock acquired: commit {head[:12]}, protocol {protocol_sha[:12]}, clean worktree {not dirty}, "
            f"MPS clients {len(payload['concurrent_mps_clients_at_launch'])}")
    return runner.run_steady_state(protocol, results, backend=backend, protocol_path=protocol_path, wall_budget_seconds=wall_budget_seconds, log=log)


# -- assessment -------------------------------------------------------------------------------------------------------------

def _cumulative_anomalous(summary: dict[str, Any]) -> float | None:
    """The hook's cumulative event count (macro events) from the final series record's ledger."""

    ledger = (summary.get("final_series") or {}).get("ledger") or {}
    return (ledger.get("cumulative") or {}).get("anomalous")


def _wall_area_m2(grid, z_cells: np.ndarray) -> np.ndarray:
    geometry = grid.geometry
    radius = np.array([float(geometry.wall_radius_m(z)) if z < geometry.z_max_m else float(geometry.exit_radius_m) for z in z_cells])
    if geometry.cone_start_z_m < geometry.z_max_m:
        slope = (geometry.exit_radius_m - geometry.bore_radius_m) / (geometry.z_max_m - geometry.cone_start_z_m)
        slant = np.where(z_cells > geometry.cone_start_z_m, np.sqrt(1.0 + slope * slope), 1.0)
    else:
        slant = np.ones_like(z_cells)
    return 2.0 * pi * radius * grid.dz_m * slant


def per_cusp_report(maps: dict[str, np.ndarray], grid, planes_m=CUSP_PLANES_M, half_width_m: float = CUSP_HALF_WIDTH_M) -> list[dict[str, Any]]:
    """Per-cusp wall currents (+-half_width of each plane), axis-to-wall potential drop and near-wall T_e at the plane, from the window maps."""

    geometry = grid.geometry
    phi = np.asarray(maps["phi_v"], dtype=float)
    t_e = np.asarray(maps["t_e_ev"], dtype=float)
    wall_e = np.asarray(maps["wall_electron_flux_per_m2_s"], dtype=float)
    wall_i = np.asarray(maps["wall_ion_flux_per_m2_s"], dtype=float)
    n_wall = min(wall_e.size, round(geometry.channel_length_m / grid.dz_m))
    z_cells = geometry.z_min_m + (np.arange(n_wall) + 0.5) * grid.dz_m
    area = _wall_area_m2(grid, z_cells)
    current_e = ELEMENTARY_CHARGE_C * wall_e[:n_wall] * area
    current_i = ELEMENTARY_CHARGE_C * wall_i[:n_wall] * area
    rows = []
    for z_c in planes_m:
        j = max(0, min(phi.shape[1] - 1, round((z_c - geometry.z_min_m) / grid.dz_m)))
        wall_index = min(round(float(geometry.wall_radius_m(min(z_c, geometry.z_max_m - 1e-12))) / grid.dr_m), phi.shape[0] - 1)
        mask = (z_cells >= z_c - half_width_m) & (z_cells <= z_c + half_width_m)
        near = t_e[max(0, wall_index - 3):wall_index, j]
        rows.append({"z_c_m": float(z_c), "electron_wall_current_a": float(current_e[mask].sum()), "ion_wall_current_a": float(current_i[mask].sum()),
                     "axis_potential_v": float(phi[0, j]), "wall_potential_v": float(phi[wall_index, j]), "sheath_drop_v": float(phi[0, j] - phi[wall_index, j]),
                     "near_wall_t_e_ev": float(np.nanmean(near)) if near.size else float("nan"), "axis_t_e_ev": float(t_e[0, j])})
    return rows


def run_quantities(results: Path, grid=None) -> dict[str, Any]:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    peak = _peak_from_maps(results / "maps.npz")
    triad = summary.get("grid_heating_triad") or {}
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
    out = {
        "discharge_current_a": summary["window_currents_a"]["discharge_a"],
        "exit_ion_beam_a": summary["window_currents_a"]["exit_ion_beam_a"],
        "ionization_rate_per_s": summary["neutral_inventory"]["trailing_20pct_mean_ionization_rate_per_s"],
        "gross_utilisation": summary["neutral_inventory"]["propellant_utilisation_trailing"],
        "neutral_density_per_m3": summary["neutral_inventory"]["trailing_20pct_mean_density_per_m3"],
        "peak_n_e_window_per_m3": peak["peak_n_e_window_per_m3"], "t_e_peak_window_ev": peak["t_e_peak_window_ev"], "peak_node": peak["node"],
        "wall_electron_a": summary["window_currents_a"].get("wall_electron_a"), "wall_ion_a": summary["window_currents_a"].get("wall_ion_a"),
        "anomalous_collision_rate_per_s": (summary.get("final_series") or {}).get("currents_a", {}).get("anomalous_collision_rate_per_s"),
        "anomalous_events_cumulative": _cumulative_anomalous(summary),
        "stop_reason": summary["stop_reason"], "ion_transit_times": summary["ion_transit_times"], "steps_completed": summary["steps_completed"],
        "plateau": summary.get("plateau"),
        "windowed_residual_over_electrode_work": triad.get("windowed_energy_residual_over_electrode_work"),
        "windowed_residual_window_complete": triad.get("windowed_energy_residual_window_complete"),
        "cumulative_residual_over_electrode_work": triad.get("energy_residual_over_electrode_work"),
        "cells_per_debye_window_last": debye.get("cells_per_debye_window_last"), "cells_per_debye_window_trailing_mean": debye.get("trailing_20pct_mean_cells_per_debye_window"),
        "peak_debye_soft_ok": debye.get("soft_ok"), "maps_kind": summary.get("maps_kind"), "sessions": len(summary.get("sessions") or []),
        "git_head": summary.get("git_head"), "protocol_sha256": summary.get("protocol_sha256"), "config_sha256": (summary.get("provenance") or {}).get("config_sha256"),
    }
    if grid is not None:
        with np.load(results / "maps.npz") as archive:
            maps = {k: np.asarray(archive[k]) for k in archive.files}
        if all(k in maps for k in ("phi_v", "t_e_ev", "wall_electron_flux_per_m2_s", "wall_ion_flux_per_m2_s")):
            out["per_cusp"] = per_cusp_report(maps, grid)
    return out


def reference_quantities_from_files(results: Path = REFERENCE_RESULTS) -> dict[str, float] | None:
    """Recompute the pinned alpha = 0 numbers from the ss-v4 results directory (fail-closed consistency check)."""

    if not (results / "summary.json").is_file() or not (results / "maps.npz").is_file():
        return None
    q = run_quantities(results)
    return {k: q[k] for k in QUANTITY_KEYS}


def _shift_rows(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key in QUANTITY_KEYS:
        ref = float(reference[key])
        value = float(run[key])
        rel = (value - ref) / abs(ref) if ref != 0.0 else float("inf")
        band = PARTICLE_BAND.get(key)
        sign = HYPOTHESES.get(key, {}).get("sign")
        if band is None or sign is None:
            status = "reported"
        elif abs(rel) <= band:
            status = "inside_band"
        elif (rel > 0) == (sign == "+"):
            status = "confirming"
        else:
            status = "contradicting"
        rows[key] = {"reference": ref, "value": value, "relative_shift": rel, "particle_band": band, "hypothesis_sign": sign, "status": status}
    return rows


def _consistency(pinned: dict[str, Any], results: Path) -> dict[str, Any] | None:
    recomputed = reference_quantities_from_files(results)
    if recomputed is None:
        return None
    return {key: {"pinned": float(pinned[key]), "recomputed": float(recomputed[key]),
                  "agree": abs(float(recomputed[key]) - float(pinned[key])) <= 1e-9 * max(abs(float(pinned[key])), 1e-300)} for key in recomputed}


def assess_case(case: str, *, results: Path | None = None, protocol: dict[str, Any] | None = None, output: Path | None = None, reference_check: bool = True,
                log: Callable[[str], None] = _log) -> dict[str, Any]:
    """Per-case verdict (plateau_clean / plateau_heating / no_plateau) and the shift table against the alpha = 0 reference."""

    results = case_results(case) if results is None else results
    protocol = load_case_protocol(case) if protocol is None else protocol
    if not (results / "summary.json").is_file():
        raise PIC2DValidationError(f"{results} has no summary.json to assess")
    acceptance = protocol["stopping_rule"]["acceptance"]
    reference = protocol["reference_run"]["quantities"]
    grid = runner.build_config(protocol, backend="cpu").grid
    run = run_quantities(results, grid)
    a_plateau = run["stop_reason"] == "plateau_reached_after_min_transit_times"
    windowed = run["windowed_residual_over_electrode_work"]
    b_ok = windowed is not None and bool(run["windowed_residual_window_complete"]) and windowed < 0.02
    verdict = "plateau_clean" if (a_plateau and b_ok) else "plateau_heating" if a_plateau else "no_plateau"
    shifts = _shift_rows(run, reference)
    consistency = None
    reference_cusps = None
    if reference_check:
        consistency = _consistency(reference, REFERENCE_RESULTS)
        if consistency is not None and not all(entry["agree"] for entry in consistency.values()):
            raise PIC2DValidationError("reference_run.quantities disagree with the ss-v4 artifacts on disk: " + json.dumps({k: v for k, v in consistency.items() if not v["agree"]}))
        if (REFERENCE_RESULTS / "maps.npz").is_file():
            reference_cusps = run_quantities(REFERENCE_RESULTS, grid).get("per_cusp")
    cusp_rows = None
    if run.get("per_cusp") is not None and reference_cusps is not None:
        cusp_rows = []
        for mine, ref in zip(run["per_cusp"], reference_cusps, strict=True):
            cusp_rows.append({"z_c_m": mine["z_c_m"],
                              "electron_wall_current_a": {"value": mine["electron_wall_current_a"], "reference": ref["electron_wall_current_a"],
                                                          "relative_shift": (mine["electron_wall_current_a"] - ref["electron_wall_current_a"]) / abs(ref["electron_wall_current_a"]) if ref["electron_wall_current_a"] else None,
                                                          "hypothesis_sign": HYPOTHESES["cusp_electron_wall_current_a"]["sign"]},
                              "ion_wall_current_a": {"value": mine["ion_wall_current_a"], "reference": ref["ion_wall_current_a"]},
                              "sheath_drop_v": {"value": mine["sheath_drop_v"], "reference": ref["sheath_drop_v"], "difference_v": mine["sheath_drop_v"] - ref["sheath_drop_v"],
                                                "hypothesis_sign": HYPOTHESES["cusp_sheath_drop_v"]["sign"]},
                              "near_wall_t_e_ev": {"value": mine["near_wall_t_e_ev"], "reference": ref["near_wall_t_e_ev"]}})
    record = {
        "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "case": case, "alpha": protocol["campaign"]["alpha"],
        "results_dir": results.relative_to(HERE).as_posix() if results.is_relative_to(HERE) else str(results), "git_head_now": runner.git_head(), "run": run,
        "reference": reference, "reference_case": REFERENCE_CASE, "reference_corrected_ledger": protocol["reference_run"]["corrected_ledger"], "reference_consistency": consistency,
        "a_plateau": {"passed": a_plateau, "stop_reason": run["stop_reason"], "ion_transit_times": run["ion_transit_times"], "plateau": run["plateau"], "rule": acceptance["a_plateau"]},
        "b_residual_power": {"passed": b_ok, "windowed_residual_over_electrode_work": windowed, "window_complete": run["windowed_residual_window_complete"], "bound": 0.02,
                             "one_sided": True, "ledger": "v2.0.6 W-corrected (native)", "cumulative_witness": run["cumulative_residual_over_electrode_work"],
                             "reference_reads": protocol["reference_run"]["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"], "rule": acceptance["b_residual_power"]},
        "c_shifts_vs_alpha_0": shifts, "per_cusp_vs_alpha_0": cusp_rows,
        "verdict": verdict, "verdict_rule": acceptance["d_verdict"]["per_case"][verdict],
        "peak_debye_window": {"cells_per_debye_last": run["cells_per_debye_window_last"], "trailing_mean": run["cells_per_debye_window_trailing_mean"], "soft_ok": run["peak_debye_soft_ok"]},
        "claim_boundary": protocol["claim_boundary"],
    }
    artifacts.write_canonical_json(output or (results / "assessment.json"), record)
    log(f"[assess] {case} (alpha {protocol['campaign']['alpha']:.5g}): {verdict} (a {a_plateau}, b {b_ok} [{windowed}]); shifts vs alpha 0: "
        + ", ".join(f"{k} {v['relative_shift']*100:+.1f}% {v['status']}" for k, v in shifts.items()))
    return record


def _monotone(values: list[tuple[float, float]], sign: str, band: float) -> bool:
    """Ordered in the declared direction over increasing alpha; a reversal smaller than band x |previous| is a tie."""

    ordered = sorted(values)
    for (_, a), (_, b) in itertools.pairwise(ordered):
        step = (b - a) / abs(a) if a != 0.0 else 0.0
        if (sign == "+" and step < -band) or (sign == "-" and step > band):
            return False
    return True


def assess_series(*, results_root: Path = RESULTS, cases_override: dict[str, Path] | None = None, output: Path | None = None, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """The trend verdict over the reached points {alpha = 0 (reference), cases with an assessment}; unreached cases are listed."""

    campaign = load_campaign()
    reference = campaign["reference_run"]["quantities"]
    points: dict[str, dict[str, Any]] = {REFERENCE_CASE: {"alpha": 0.0, "reached": True, "verdict": "reference (recorded ss-v4 plateau; (b) FAIL at +2.46 % corrected)",
                                                          "quantities": {k: float(reference[k]) for k in QUANTITY_KEYS}}}
    for case in CASES:
        results = (cases_override or {}).get(case, case_results(case, results_root))
        assessment_path = results / "assessment.json"
        entry: dict[str, Any] = {"alpha": float(CASES[case]["alpha"]), "reached": False, "verdict": None, "quantities": None, "results_dir": str(results)}
        if assessment_path.is_file():
            assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
            entry.update({"verdict": assessment["verdict"], "reached": assessment["a_plateau"]["passed"], "quantities": {k: float(assessment["run"][k]) for k in QUANTITY_KEYS},
                          "shifts": assessment["c_shifts_vs_alpha_0"], "b_passed": assessment["b_residual_power"]["passed"], "per_cusp": assessment.get("per_cusp_vs_alpha_0")})
        elif (results / "summary.json").is_file():
            entry["verdict"] = "not assessed (summary present; run `assess --case`)"
        points[case] = entry
    reached = {k: v for k, v in points.items() if v["reached"]}
    monotone: dict[str, Any] = {}
    for key in MONOTONE_QUANTITIES:
        sign = HYPOTHESES[key]["sign"]
        values = [(v["alpha"], v["quantities"][key]) for v in reached.values()]
        monotone[key] = {"hypothesis_sign": sign, "values_by_alpha": {str(a): q for a, q in sorted(values)}, "monotone_in_declared_direction": _monotone(values, sign, PARTICLE_BAND[key]) if len(values) >= 2 else None}
    contradictions = [f"{case}:{k}" for case, v in reached.items() if case != REFERENCE_CASE for k, row in v["shifts"].items() if k in MONOTONE_QUANTITIES and row["status"] == "contradicting"]
    key_shifts_all_inside = all(row["status"] == "inside_band" for case, v in reached.items() if case != REFERENCE_CASE
                                for k, row in v["shifts"].items() if k in ("discharge_current_a", "peak_n_e_window_per_m3")) if len(reached) > 1 else True
    if len(reached) < 3 or key_shifts_all_inside:
        verdict = "inconclusive"
    elif monotone["discharge_current_a"]["monotone_in_declared_direction"] and monotone["peak_n_e_window_per_m3"]["monotone_in_declared_direction"] and not contradictions:
        verdict = "trend_confirmed"
    else:
        verdict = "trend_not_confirmed"
    record = {
        "schema_version": SERIES_ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": EXPERIMENT_ID, "git_head_now": runner.git_head(), "points": points,
        "points_reached": sorted(reached, key=lambda c: points[c]["alpha"]), "points_unreached": [c for c in points if not points[c]["reached"]],
        "monotonicity": monotone, "contradictions": contradictions, "key_shifts_all_inside_band": key_shifts_all_inside,
        "verdict": verdict, "verdict_rule": campaign["acceptance"]["d_verdict"]["series"][verdict], "hypotheses": HYPOTHESES, "launch_priority": list(LAUNCH_PRIORITY),
        "note": "the verdict evaluates the reached points only; it is provisional until every case has a terminal state and final once all four points are in",
    }
    output = output or (results_root / "series-assessment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.write_canonical_json(output, record)
    log(f"[assess --series] {verdict}: reached {record['points_reached']}, unreached {record['points_unreached']}; monotone I_d "
        f"{monotone['discharge_current_a']['monotone_in_declared_direction']}, peak n_e {monotone['peak_n_e_window_per_m3']['monotone_in_declared_direction']}; contradictions {contradictions}")
    return record


# -- CLI ---------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    comp = sub.add_parser("compose")
    comp.add_argument("--budget-from-preflight", action="store_true")
    pre = sub.add_parser("preflight")
    pre.add_argument("--case", required=True, choices=sorted(CASES))
    pre.add_argument("--backend", default="warp-cuda")
    pre.add_argument("--timing-steps", type=int, default=2000)
    pre.add_argument("--loaded-seed-density", type=float, default=1.75e17)
    pre.add_argument("--gpu-timing", action="store_true", help="time the step on the launch GPU (default: inputs + mesh only)")
    shake = sub.add_parser("shakedown")
    shake.add_argument("--case", required=True, choices=sorted(CASES))
    shake.add_argument("--backend", default="warp-cuda")
    shake.add_argument("--max-steps", type=int, default=SHAKEDOWN_OVERRIDES["max_steps"])
    la = sub.add_parser("launch")
    la.add_argument("--case", required=True, choices=sorted(CASES))
    la.add_argument("--backend", default="warp-cuda")
    la.add_argument("--expect-commit", default=None)
    la.add_argument("--resume", action="store_true")
    la.add_argument("--allow-dirty", action="store_true", help="development only; never for the preregistered execution")
    la.add_argument("--require-mps", action="store_true", help="refuse unless CUDA_MPS_PIPE_DIRECTORY is set and exists (the four-slot H100 configuration)")
    la.add_argument("--wall-budget-seconds", type=float, default=None)
    sub.add_parser("status")
    fin = sub.add_parser("finalize")
    fin.add_argument("--case", required=True, choices=sorted(CASES))
    fin.add_argument("--backend", default="warp-cuda")
    fin.add_argument("--stop-reason", default="finalized_from_checkpoint")
    fin.add_argument("--allow-refinalize", action="store_true")
    fin.add_argument("--recover-runner-stop", action="store_true")
    ass = sub.add_parser("assess")
    ass.add_argument("--case", default=None, choices=sorted(CASES))
    ass.add_argument("--series", action="store_true")
    ass.add_argument("--results", default=None)
    args = parser.parse_args(argv)
    if args.command == "compose":
        compose(budget_from_preflight=args.budget_from_preflight)
    elif args.command == "preflight":
        preflight(args.case, backend=args.backend, timing_steps=args.timing_steps, loaded_seed_density=args.loaded_seed_density, gpu_timing=args.gpu_timing)
    elif args.command == "shakedown":
        shakedown(args.case, backend=args.backend, max_steps=args.max_steps)
    elif args.command == "launch":
        launch(args.case, backend=args.backend, expect_commit=args.expect_commit, resume=args.resume, allow_dirty=args.allow_dirty, require_mps=args.require_mps,
               wall_budget_seconds=args.wall_budget_seconds)
    elif args.command == "status":
        for case in CASES:
            results = case_results(case)
            print(json.dumps({"case": case, "status": runner.status(results, load_case_protocol(case)) if results.exists() else "not launched"}, indent=1, default=str))
    elif args.command == "finalize":
        protocol = load_case_protocol(args.case)
        runner.finalize(protocol, case_results(args.case), backend=args.backend, stop_reason=args.stop_reason, protocol_path=protocol_module.PROTOCOLS_DIR / f"{args.case}.json",
                        allow_refinalize=args.allow_refinalize, recover_runner_stop=args.recover_runner_stop)
    else:
        if args.series or args.case is None:
            assess_series()
        else:
            assess_case(args.case, results=None if args.results is None else Path(args.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
