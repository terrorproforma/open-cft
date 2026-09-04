"""Physics effects v1: the preregistered SEE(BN) / xenon-collision-set-v2 campaign on the 33 um reference plateau (roadmap R2 + R3).

Three cases (``protocols/see-bn.json``, ``xe-set-v2.json``, ``see-bn+xe-set-v2.json``) = the ss-v4 protocol with model v2.2.0 SEE from the BN
wall and / or model v2.3.0 ``xe_collision_set_v2``, the v2.0.6 gates and K = 5, alpha = 0; the reference point of every case is the RECORDED
ss-v4 plateau (``pic2d_cft_steady_state_v4/results``, 0d228ad2), which fails its own acceptance (b) at +2.46 % on the corrected ledger.

Stages (from ``modern/`` with ``PYTHONPATH=src:.``)::

    python -m experiments.pic2d_physics_effects_v1.run compose [--budget-from-preflight]      # (re)write protocols/*.json + protocol.json
    python -m experiments.pic2d_physics_effects_v1.run preflight --case see-bn [--gpu-timing]   # -> preflight-<case>.json
    python -m experiments.pic2d_physics_effects_v1.run shakedown --case see-bn                  # 100k steps -> finalize -> assess -> shakedown-<case>.json
    python -m experiments.pic2d_physics_effects_v1.run launch --case see-bn --expect-commit SHA [--require-mps] [--resume]
    python -m experiments.pic2d_physics_effects_v1.run status
    python -m experiments.pic2d_physics_effects_v1.run finalize --case see-bn [...]            # externally stopped run only
    python -m experiments.pic2d_physics_effects_v1.run assess --case see-bn                    # per-case verdict -> results/<case>/assessment.json
    python -m experiments.pic2d_physics_effects_v1.run assess --campaign                       # three verdicts + additivity -> results/campaign-assessment.json

The stepping is the shared runner ``experiments.pic2d_cft_steady_state_v1.run``; the stages follow the v4 / v5 / alpha-series preregistration
discipline (clean worktree, expected commit, sealed protocol == recomposition, O_EXCL execution lock, preflight + shakedown records).
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
from cft_revival.pic2d.models import ELEMENTARY_CHARGE_C, PIC2DValidationError
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
from experiments.pic2d_cft_steady_state_v5.run import (
    _peak_from_maps,
    acquire_lock,
    gpu_load_snapshot,
)
from experiments.pic2d_physics_effects_v1 import protocol as protocol_module
from experiments.pic2d_physics_effects_v1.protocol import (
    ABSOLUTE_BAND,
    CASES,
    CUSP_HALF_WIDTH_M,
    CUSP_PLANES_M,
    EXPERIMENT_ID,
    HYPOTHESES_BY_CASE,
    KEY_QUANTITIES,
    LAUNCH_PRIORITY,
    PARTICLE_BAND,
    QUANTITY_KEYS,
    REFERENCE_CASE,
    STEPS_TO_3_TRANSITS,
    channel_wall_cells,
    compose_campaign,
    compose_case_protocol,
    iedf_low_energy_fraction,
    load_campaign,
    load_case_protocol,
    protocol_sha256,
    wall_power_and_ion_energy,
    write_sealed_protocols,
)

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
RESULTS = HERE / "results"
LOCK_NAME = "execution-lock.json"
LOCK_SCHEMA = "cft-revival.pic2d-steady-state-execution-lock/1.0.0"
ASSESSMENT_SCHEMA = "cft-revival.pic2d-physics-effects-v1.assessment/1.0.0"
CAMPAIGN_ASSESSMENT_SCHEMA = "cft-revival.pic2d-physics-effects-v1.campaign-assessment/1.0.0"
PREFLIGHT_SCHEMA = "cft-revival.pic2d-physics-effects-v1.preflight/1.0.0"
SHAKEDOWN_SCHEMA = "cft-revival.pic2d-physics-effects-v1.shakedown/1.0.0"
REFERENCE_RESULTS = protocol_module.V4_RESULTS
COMBINED_CASE = "see-bn+xe-set-v2"
SINGLE_CASES = ("see-bn", "xe-set-v2")
SEE_CUMULATIVE_KEYS = ("see_impacts", "see_electrons", "see_ion_induced_electrons", "see_backscattered", "see_yield_clamped", "ke_see_emitted_j")
XE_CUMULATIVE_KEYS = ("cex", "mex", "cex_plume", "ion_mcc_candidates", "ion_mcc_null", "ion_mcc_ceiling_violations", "fast_neutral_exit_channel",
                      "fast_neutral_wall", "fast_neutral_thermal", "fast_neutral_unresolved", "ion_neutral_loss_j", "pz_ion_collisions", "pz_fast_neutral_exit",
                      "pz_fast_neutral_wall", "ke_fast_neutral_exit_j", "excitations", "excitations_level_1", "excitations_level_2", "excitations_level_3",
                      "excitations_level_4", "ionizations", "exit_ions")
XE_CURRENT_KEYS = ("cex_rate_per_s", "mex_rate_per_s", "cex_plume_rate_per_s", "fast_neutral_exit_rate_per_s", "fast_neutral_wall_rate_per_s",
                   "fast_neutral_thermal_rate_per_s", "ion_mcc_candidate_rate_per_s")
SEE_SERIES_KEYS = ("see_interval_effective_yield", "see_cumulative_effective_yield", "see_interval_mean_yield", "see_emission_current_a", "see_wall_impact_current_a",
                   "see_backscattered_fraction", "see_mean_emitted_energy_ev", "see_emitted_power_w", "see_wall_potential_mean_v", "see_wall_potential_min_v",
                   "see_wall_potential_max_v", "see_plasma_minus_wall_mean_v", "see_interval_clamped_impacts")

# shakedown: the real protocol with only the cadences shrunk (every gate, the grid, dt, W, the effect blocks, field and seed are the real ones)
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


def _effect_identity(config) -> dict[str, Any]:
    return {"see": None if config.see is None else config.see.to_dict(),
            "collision_set": None if config.mcc is None or config.mcc.collision_set is None else config.mcc.collision_set.to_dict(),
            "anomalous": None if config.anomalous is None else config.anomalous.to_dict()}


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
    key = f"modern/experiments/pic2d_physics_effects_v1/protocols/{case}.json"
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
        "schema_version": PREFLIGHT_SCHEMA, "utc": utc_now(), "git_head": runner.git_head(), "case": case, "effects": protocol["campaign"]["effects"],
        "protocol_sha256": protocol_sha256(protocol), "config_sha256": artifacts.config_identity(config), "backend": backend,
        "host": socket.gethostname(), "python": sys.version.split()[0], "non_evidentiary": True, "gpu_load_before": gpu_before,
        "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), "concurrent_mps_clients": len(others), "concurrent_mps_client_pids": [a["pid"] for a in others],
        "grid": {"cells": list(grid.cell_shape), "nodes": list(grid.node_shape), "dr_m": grid.dr_m, "dz_m": grid.dz_m},
        "dt_s": config.dt_s, "macro_weight": config.macro_weight, "effect_identity": _effect_identity(config),
        "moment_sample_interval": config.moment_sample_interval,
        "peak_debye_floor": config.peak_debye_gate.to_dict() if config.peak_debye_gate is not None else None,
    }
    t0 = time.perf_counter()
    field_map, cross_sections = runner.load_inputs(config, None, None, protocol=protocol)
    record["field"] = {"sha256": field_map.sha256, "source_sha256": getattr(field_map, "source_sha256", None), "max_b_t": field_map.max_b_t, "seconds": time.perf_counter() - t0}
    record["cross_sections_sha256"] = cross_sections.payload_sha256 if cross_sections is not None else None
    record["cross_sections_processes"] = [getattr(p, "identifier", str(p)) for p in getattr(cross_sections, "processes", [])] if cross_sections is not None else None
    masks = build_mesh_masks(grid)
    record["mesh"] = masks.to_dict()
    if config.see is not None:
        material = config.see.resolved_material()
        record["see_admissibility"] = {"material": config.see.material, "flux_averaged_yield_note": "spec v2.2: BN 0.48 / 0.58 / 0.69 at T_e 5 / 7 / 10 eV, critical T_e 20.3 eV",
                                      "delta_max": material.delta_max, "energy_max_ev": material.energy_max_ev, "space_charge_limit_yield": config.see.space_charge_limit_yield}
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
    log(f"[preflight] {case} ({', '.join(protocol['campaign']['effects'])}): grid {grid.cell_shape}; field {field_map.sha256[:12]} max |B| {field_map.max_b_t:.3f} T; "
        f"factorisation {record['factorisation_seconds']:.1f} s; seed {seed_state.electrons.count} e-; MPS clients before: {len(others)}; GPU before: {gpu_before}")
    timing_seed = _time_steps(sim, timing_steps, warmup=200)
    after = device_memory()
    timing_seed.update({"electrons_after": sim.state.electrons.count, "ions_after": sim.state.ions.count, "step_graph": sim.step_graph_state(),
                        "effect_events": _effect_events(sim.state.cumulative)})
    record["timing_seed_load"] = timing_seed
    record["last_series_record"] = sim.series[-1].to_dict() if sim.series else None
    log(f"[preflight] seed load: {timing_seed['ms_per_step']:.3f} ms/step over {timing_steps} steps ({sim.state.electrons.count} e-; events {timing_seed['effect_events']})")
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
                          "effect_events": _effect_events(sim2.state.cumulative)})
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
                                     "cells_per_debye": ref["cells_per_debye_at_peak_window"],
                                     "hypothesis": {k: v["sign"] for k, v in HYPOTHESES_BY_CASE[case].items() if k in ("peak_n_e_window_per_m3", "t_e_peak_window_ev")}}
    artifacts.write_canonical_json(preflight_path(case), record)
    log(f"[preflight] projection {hours_3:.2f} h to 3 transits at the plateau load; budget {budget / 3600:.1f} h; written {preflight_path(case)}")
    return record


def _effect_events(cumulative: dict[str, Any]) -> dict[str, Any]:
    return {k: cumulative.get(k) for k in ("see_impacts", "see_electrons", "cex", "mex", "fast_neutral_exit_channel", "excitations_level_1") if k in cumulative}


# -- shakedown --------------------------------------------------------------------------------------------------------------

def shakedown_protocol(protocol: dict[str, Any], overrides: dict[str, Any] = SHAKEDOWN_OVERRIDES) -> dict[str, Any]:
    """The real protocol with every cadence shrunk (NON-EVIDENTIARY): grid, dt, W, effect blocks, field, seed and gate thresholds untouched."""

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
    campaign = assess_campaign(results_root=results.parent, cases_override={case: results}, log=log, output=results / "campaign-assessment.json")
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
    cumulative = ((summary.get("final_series") or {}).get("ledger") or {}).get("cumulative") or {}
    run = assessment["run"]
    record = {
        "schema_version": SHAKEDOWN_SCHEMA, "utc": utc_now(), "git_head": runner.git_head(), "non_evidentiary": True, "case": case, "effects": protocol["campaign"]["effects"],
        "host": socket.gethostname(), "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), "concurrent_mps_clients": len(others),
        "concurrent_mps_client_pids": [a["pid"] for a in others], "overrides": {**SHAKEDOWN_OVERRIDES, "max_steps": max_steps}, "results_dir": results.relative_to(HERE).as_posix(),
        "run_seconds": run_seconds, "refinalize_seconds": refinalize_seconds, "steps_completed": summary["steps_completed"], "stop_reason": summary["stop_reason"],
        "ms_per_step": summary["ms_per_step_this_session"], "final_counts": summary["final_counts"], "config_sha256": (summary.get("provenance") or {}).get("config_sha256"),
        "frames": summary["artifacts"]["frames"]["count"] if summary["artifacts"].get("frames") else 0,
        "effect_cumulative": {k: cumulative.get(k) for k in SEE_CUMULATIVE_KEYS + XE_CUMULATIVE_KEYS if k in cumulative},
        "see_readings": run.get("see"), "collision_readings": run.get("collision_set"), "iedf_exit_plane": run.get("iedf"), "per_cusp": run.get("per_cusp"),
        "peak_debye_window": {"records": len(windows), "enforced_records": len(enforced), "last": windows[-1] if windows else None,
                              "max_cells_per_debye_enforced": max((w["cells_per_debye"] for w in enforced), default=None),
                              "floor_kind": "accumulated_particle_steps" if windows and windows[-1].get("min_accumulated_macro_particle_steps_at_peak") else "mean_occupancy"},
        "windowed_residual": {"records_with_complete_window": len(complete),
                              "last": None if not complete else {k: complete[-1][k] for k in complete[-1] if k.startswith("windowed") or k == "energy_residual_over_electrode_work"}},
        "plateau_keys": sorted(summary["plateau"]) if summary.get("plateau") else None,
        "summary_keys_present": {key: (summary.get(key) is not None) for key in ("plateau", "grid_heating_triad", "peak_node_debye", "neutral_inventory", "ledger", "window_currents_a")},
        "assessment": {k: assessment[k] for k in ("plateau_status", "hypothesis_verdict", "a_plateau", "b_residual_power", "reference_consistency")},
        "campaign_assessment": {"cases_reached": campaign["cases_reached"], "additivity": campaign["additivity"]["statement"]},
        "artifacts": {k: summary["artifacts"].get(k) for k in ("maps_npz_sha256", "series_npz_sha256")},
        "gate_not_inert_check": {"peak_window_enforced_at_least_once": bool(enforced), "peak_window_resolved_nodes_last": windows[-1].get("resolved_nodes") if windows else None,
                                 "residual_window_completed_at_least_once": bool(complete),
                                 "see_events_nonzero": (cumulative.get("see_electrons", 0) or 0) > 0 if CASES[case]["see"] else None,
                                 "cex_events_nonzero": (cumulative.get("cex", 0) or 0) > 0 if CASES[case]["collision_set"] else None,
                                 "level_split_nonzero": all((cumulative.get(f"excitations_level_{i}", 0) or 0) > 0 for i in (1, 2, 3, 4)) if CASES[case]["collision_set"] else None},
    }
    artifacts.write_canonical_json(shakedown_path(case), record)
    log(f"[shakedown] {case}: {summary['steps_completed']} steps, {summary['stop_reason']}, {summary['ms_per_step_this_session']:.2f} ms/step, {record['frames']} frames, "
        f"events {record['effect_cumulative']}, peak window enforced {len(enforced)}/{len(windows)} (max {record['peak_debye_window']['max_cells_per_debye_enforced']}), "
        f"residual windows complete {len(complete)}; plateau {assessment['plateau_status']}, verdict {assessment['hypothesis_verdict']}; written {shakedown_path(case)}")
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
    if not preflight_path(case).is_file() or not shakedown_path(case).is_file():
        raise PIC2DValidationError(f"preflight-{case}.json and shakedown-{case}.json must exist (and be committed) before a preregistered launch")
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if require_mps and not (mps_pipe and Path(mps_pipe).exists()):
        raise PIC2DValidationError(f"--require-mps: CUDA_MPS_PIPE_DIRECTORY {mps_pipe!r} is not set or does not exist in this environment")
    protocol_sha = protocol_sha256(protocol)
    payload = {
        "schema_version": LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "case": case, "effects": protocol["campaign"]["effects"], "commit": head,
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
        log(f"[launch] {case} ({', '.join(protocol['campaign']['effects'])}): execution lock acquired: commit {head[:12]}, protocol {protocol_sha[:12]}, clean worktree {not dirty}, "
            f"MPS clients {len(payload['concurrent_mps_clients_at_launch'])}")
    return runner.run_steady_state(protocol, results, backend=backend, protocol_path=protocol_path, wall_budget_seconds=wall_budget_seconds, log=log)


# -- diagnostics --------------------------------------------------------------------------------------------------------------

def per_cusp_report(maps: dict[str, np.ndarray], grid, planes_m=CUSP_PLANES_M, half_width_m: float = CUSP_HALF_WIDTH_M,
                    space_charge_limit_yield: float | None = None) -> list[dict[str, Any]]:
    """Per-cusp wall currents (+-half_width of each plane), axis-to-wall and near-wall potential drops, near-wall T_e, wall-ion impact energy and - when the
    wall emits - the effective SEE yield, SEE current, mean emitted energy and the space-charge-limit flag, from the window maps."""

    geometry = grid.geometry
    phi = np.asarray(maps["phi_v"], dtype=float)
    t_e = np.asarray(maps["t_e_ev"], dtype=float)
    wall_e = np.asarray(maps["wall_electron_flux_per_m2_s"], dtype=float)
    wall_i = np.asarray(maps["wall_ion_flux_per_m2_s"], dtype=float)
    n_wall, z_cells, area = channel_wall_cells(maps, grid)
    current_e = ELEMENTARY_CHARGE_C * np.nan_to_num(wall_e[:n_wall]) * area
    current_i = ELEMENTARY_CHARGE_C * np.nan_to_num(wall_i[:n_wall]) * area
    ion_energy = np.nan_to_num(np.asarray(maps["wall_ion_mean_energy_ev"], dtype=float)[:n_wall]) if "wall_ion_mean_energy_ev" in maps else None
    see_flux = np.nan_to_num(np.asarray(maps["wall_see_flux_per_m2_s"], dtype=float)[:n_wall]) if "wall_see_flux_per_m2_s" in maps else None
    see_energy = np.nan_to_num(np.asarray(maps["wall_see_mean_energy_ev"], dtype=float)[:n_wall]) if "wall_see_mean_energy_ev" in maps else None
    limit = space_charge_limit_yield
    rows = []
    for z_c in planes_m:
        j = max(0, min(phi.shape[1] - 1, round((z_c - geometry.z_min_m) / grid.dz_m)))
        wall_index = min(round(float(geometry.wall_radius_m(min(z_c, geometry.z_max_m - 1e-12))) / grid.dr_m), phi.shape[0] - 1)
        mask = (z_cells >= z_c - half_width_m) & (z_cells <= z_c + half_width_m)
        near = t_e[max(0, wall_index - 3):wall_index, j]
        row: dict[str, Any] = {
            "z_c_m": float(z_c), "electron_wall_current_a": float(current_e[mask].sum()), "ion_wall_current_a": float(current_i[mask].sum()),
            "axis_potential_v": float(phi[0, j]), "wall_potential_v": float(phi[wall_index, j]), "sheath_drop_v": float(phi[0, j] - phi[wall_index, j]),
            "near_wall_drop_v": float(phi[max(wall_index - 3, 0), j] - phi[wall_index, j]),
            "near_wall_t_e_ev": float(np.nanmean(near)) if near.size else float("nan"), "axis_t_e_ev": float(t_e[0, j]),
        }
        if ion_energy is not None:
            weight = float(current_i[mask].sum())
            row["wall_ion_mean_energy_ev"] = float(np.sum(current_i[mask] * ion_energy[mask]) / weight) if weight > 0.0 else None
        if see_flux is not None:
            impacting = float(np.sum(np.nan_to_num(wall_e[:n_wall])[mask] * area[mask]))
            emitted = float(np.sum(see_flux[mask] * area[mask]))
            eff = emitted / impacting if impacting > 0.0 else None
            row["see_current_a"] = ELEMENTARY_CHARGE_C * emitted
            row["see_effective_yield"] = eff
            row["see_mean_emitted_energy_ev"] = (float(np.sum(see_flux[mask] * area[mask] * see_energy[mask]) / emitted) if see_energy is not None and emitted > 0.0 else None)
            row["space_charge_limited"] = bool((eff is not None and limit is not None and eff >= limit) or row["near_wall_drop_v"] < 0.0)
            row["space_charge_limit_rule"] = f"effective yield >= {limit} OR near-wall drop < 0 (non-monotonic sheath)"
        rows.append(row)
    return rows


def iedf_report(maps: dict[str, np.ndarray], anode_v: float) -> dict[str, Any] | None:
    if "iedf_ion_counts" not in maps or "iedf_edges_ev" not in maps:
        return None
    counts = np.asarray(maps["iedf_ion_counts"], dtype=float)
    edges = np.asarray(maps["iedf_edges_ev"], dtype=float)
    total = float(counts.sum())
    if counts.size == 0 or total <= 0.0:
        return {"total_macro_ions": total}
    centres = 0.5 * (edges[:-1] + edges[1:])
    pdf = counts / total
    cdf = np.cumsum(pdf)
    return {
        "total_macro_ions": total, "mean_energy_ev": float(np.sum(pdf * centres)), "median_energy_ev": float(centres[int(np.searchsorted(cdf, 0.5))]),
        "peak_energy_ev": float(centres[int(np.argmax(counts))]),
        "low_energy_fraction": iedf_low_energy_fraction(counts, edges, anode_v),
        "fraction_below_50pct_anode": float(counts[centres < 0.5 * anode_v].sum() / total),
        "fraction_above_90pct_anode": float(counts[centres >= 0.9 * anode_v].sum() / total),
        "low_energy_bound_ev": protocol_module.IEDF_LOW_ENERGY_FRACTION_OF_ANODE * anode_v, "bins": int(counts.size), "iedf_max_ev": float(edges[-1]),
    }


def _series_window_rates(results: Path, keys: tuple[str, ...], window_steps: int) -> dict[str, Any] | None:
    """Trailing-window rates of cumulative ledger keys from series.jsonl (delta cumulative / delta time between the last record and the record at the window start)."""

    path = results / "series.jsonl"
    if not path.is_file():
        return None
    records = [r for r in runner._read_jsonl(path) if "ledger" in r]
    if len(records) < 2:
        return None
    last = records[-1]
    start_step = int(last["step"]) - window_steps
    first = min(records, key=lambda r: abs(int(r["step"]) - start_step))
    if first is last:
        first = records[0]
    dt = float(last["time_s"]) - float(first["time_s"])
    if dt <= 0.0:
        return None
    out = {"window_steps": int(last["step"]) - int(first["step"]), "window_s": dt}
    for key in keys:
        a = (first["ledger"].get("cumulative") or {}).get(key)
        b = (last["ledger"].get("cumulative") or {}).get(key)
        out[f"{key}_rate"] = None if a is None or b is None else (float(b) - float(a)) / dt
    return out


def run_quantities(results: Path, grid=None, *, anode_v: float = 300.0, space_charge_limit_yield: float | None = None) -> dict[str, Any]:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    peak = _peak_from_maps(results / "maps.npz")
    triad = summary.get("grid_heating_triad") or {}
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
    currents = summary["window_currents_a"]
    cumulative = ((summary.get("final_series") or {}).get("ledger") or {}).get("cumulative") or {}
    out: dict[str, Any] = {
        "discharge_current_a": currents["discharge_a"],
        "exit_ion_beam_a": currents["exit_ion_beam_a"],
        "ionization_rate_per_s": summary["neutral_inventory"]["trailing_20pct_mean_ionization_rate_per_s"],
        "gross_utilisation": summary["neutral_inventory"]["propellant_utilisation_trailing"],
        "neutral_density_per_m3": summary["neutral_inventory"]["trailing_20pct_mean_density_per_m3"],
        "peak_n_e_window_per_m3": peak["peak_n_e_window_per_m3"], "t_e_peak_window_ev": peak["t_e_peak_window_ev"], "peak_node": peak["node"],
        "anode_ion_a": currents.get("anode_ion_a"),
        "wall_electron_a": currents.get("wall_electron_a"), "wall_ion_a": currents.get("wall_ion_a"),
        "stop_reason": summary["stop_reason"], "ion_transit_times": summary["ion_transit_times"], "steps_completed": summary["steps_completed"],
        "plateau": summary.get("plateau"),
        "windowed_residual_over_electrode_work": triad.get("windowed_energy_residual_over_electrode_work"),
        "windowed_residual_window_complete": triad.get("windowed_energy_residual_window_complete"),
        "cumulative_residual_over_electrode_work": triad.get("energy_residual_over_electrode_work"),
        "cells_per_debye_window_last": debye.get("cells_per_debye_window_last"), "cells_per_debye_window_trailing_mean": debye.get("trailing_20pct_mean_cells_per_debye_window"),
        "peak_debye_soft_ok": debye.get("soft_ok"), "maps_kind": summary.get("maps_kind"), "sessions": len(summary.get("sessions") or []),
        "git_head": summary.get("git_head"), "protocol_sha256": summary.get("protocol_sha256"), "config_sha256": (summary.get("provenance") or {}).get("config_sha256"),
        "iedf_low_energy_fraction": None, "wall_electron_power_w": None, "wall_ion_mean_energy_ev": None,
    }
    with np.load(results / "maps.npz") as archive:
        maps = {k: np.asarray(archive[k]) for k in archive.files}
    out["iedf"] = iedf_report(maps, anode_v)
    if out["iedf"] is not None:
        out["iedf_low_energy_fraction"] = out["iedf"].get("low_energy_fraction")
    if grid is not None:
        power, ion_energy = wall_power_and_ion_energy(maps, grid)
        out["wall_electron_power_w"] = power
        out["wall_ion_mean_energy_ev"] = ion_energy
        if all(k in maps for k in ("phi_v", "t_e_ev", "wall_electron_flux_per_m2_s", "wall_ion_flux_per_m2_s")):
            out["per_cusp"] = per_cusp_report(maps, grid, space_charge_limit_yield=space_charge_limit_yield)
    # v2.2.0 SEE readings (only when the wall emitted)
    if "see_emission_a" in currents:
        see: dict[str, Any] = {"window_emission_current_a": currents.get("see_emission_a"), "window_effective_yield": currents.get("see_effective_yield"),
                               "cumulative": {k: cumulative.get(k) for k in SEE_CUMULATIVE_KEYS if k in cumulative}}
        series_path = results / "series.npz"
        if series_path.is_file():
            with np.load(series_path) as s:
                n = s["step"].size
                tail = slice(max(n - max(n // 5, 1), 0), None)
                see["trailing_20pct_means"] = {k: float(np.nanmean(s[k][tail])) for k in SEE_SERIES_KEYS if k in s.files}
        if out.get("per_cusp") is not None:
            see["cusps_space_charge_limited"] = int(sum(1 for c in out["per_cusp"] if c.get("space_charge_limited")))
            see["cusp_effective_yields"] = [c.get("see_effective_yield") for c in out["per_cusp"]]
        out["see"] = see
    # v2.3.0 collision-set readings (only when the ion MCC ran)
    if "cex_rate_per_s" in currents:
        s_rate = float(out["ionization_rate_per_s"]) if out["ionization_rate_per_s"] else None
        collision: dict[str, Any] = {k: currents.get(k) for k in XE_CURRENT_KEYS if k in currents}
        collision["cex_over_ionization"] = (float(currents["cex_rate_per_s"]) / s_rate) if s_rate else None
        collision["cumulative"] = {k: cumulative.get(k) for k in XE_CUMULATIVE_KEYS if k in cumulative}
        exc = [cumulative.get(f"excitations_level_{i}") for i in (1, 2, 3, 4)]
        total_exc = sum(float(x) for x in exc if x is not None)
        collision["excitation_level_shares"] = [float(x) / total_exc if x is not None and total_exc > 0 else None for x in exc]
        window_steps = int(summary.get("averaging_window_steps") or 0) or 400_000
        rates = _series_window_rates(results, ("pz_fast_neutral_exit", "pz_fast_neutral_wall", "ke_fast_neutral_exit_j", "pz_exit_ions", "ion_neutral_loss_j"), window_steps)
        if rates is not None:
            collision["trailing_window_rates"] = {**rates, "source": "series.jsonl cumulative differences", "fast_neutral_exit_momentum_rate_n": rates.get("pz_fast_neutral_exit_rate"),
                                                  "exit_ion_momentum_rate_n": rates.get("pz_exit_ions_rate"), "fast_neutral_exit_power_w": rates.get("ke_fast_neutral_exit_j_rate")}
        time_s = float((summary.get("final_series") or {}).get("time_s") or summary.get("simulated_time_s") or 0.0)
        collision["run_average_rates"] = {"source": "final cumulative / simulated time (witness; includes the seed transient)",
                                          "fast_neutral_exit_momentum_rate_n": (float(cumulative["pz_fast_neutral_exit"]) / time_s) if time_s > 0 and "pz_fast_neutral_exit" in cumulative else None,
                                          "exit_ion_momentum_rate_n": (float(cumulative["pz_exit_ions"]) / time_s) if time_s > 0 and "pz_exit_ions" in cumulative else None}
        out["collision_set"] = collision
    return out


def reference_quantities_from_files(results: Path = REFERENCE_RESULTS, grid=None) -> dict[str, Any] | None:
    """Recompute the pinned reference numbers from the ss-v4 results directory (fail-closed consistency check)."""

    if not (results / "summary.json").is_file() or not (results / "maps.npz").is_file():
        return None
    q = run_quantities(results, grid)
    return {k: q.get(k) for k in QUANTITY_KEYS}


def _shift_rows(run: dict[str, Any], reference: dict[str, Any], hypotheses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key in QUANTITY_KEYS:
        ref = reference.get(key)
        value = run.get(key)
        sign = hypotheses.get(key, {}).get("sign")
        if ref is None or value is None:
            rows[key] = {"reference": ref, "value": value, "shift": None, "kind": None, "band": None, "hypothesis_sign": sign, "status": "unavailable"}
            continue
        ref = float(ref)
        value = float(value)
        if key in ABSOLUTE_BAND:
            shift, kind, band = value - ref, "absolute", ABSOLUTE_BAND[key]
        else:
            shift, kind, band = ((value - ref) / abs(ref) if ref != 0.0 else float("inf")), "relative", PARTICLE_BAND.get(key)
        if band is None or sign is None:
            status = "reported"
        elif abs(shift) <= band:
            status = "confirming" if sign == "0" else "inside_band"
        elif sign == "0":
            status = "contradicting"
        elif (shift > 0) == (sign == "+"):
            status = "confirming"
        else:
            status = "contradicting"
        rows[key] = {"reference": ref, "value": value, "shift": shift, "kind": kind, "band": band, "hypothesis_sign": sign, "status": status}
    return rows


def _consistency(pinned: dict[str, Any], results: Path, grid) -> dict[str, Any] | None:
    recomputed = reference_quantities_from_files(results, grid)
    if recomputed is None:
        return None
    out = {}
    for key, value in recomputed.items():
        if value is None or pinned.get(key) is None:
            continue
        out[key] = {"pinned": float(pinned[key]), "recomputed": float(value), "agree": abs(float(value) - float(pinned[key])) <= 1e-9 * max(abs(float(pinned[key])), 1e-300)}
    return out


def hypothesis_verdict(a_plateau: bool, shifts: dict[str, Any], key_quantities: tuple[str, ...]) -> str:
    """confirmed / not_confirmed / inconclusive by the predeclared rule (acceptance d_verdict.per_case_hypothesis_verdict)."""

    if not a_plateau:
        return "inconclusive"
    if any(row["status"] == "contradicting" for row in shifts.values()):
        return "not_confirmed"
    if all(shifts[k]["status"] == "confirming" for k in key_quantities):
        return "confirmed"
    return "inconclusive"


def assess_case(case: str, *, results: Path | None = None, protocol: dict[str, Any] | None = None, output: Path | None = None, reference_check: bool = True,
                log: Callable[[str], None] = _log) -> dict[str, Any]:
    """Per-case plateau status (plateau_clean / plateau_heating / no_plateau), hypothesis verdict and the shift table against the ss-v4 reference."""

    results = case_results(case) if results is None else results
    protocol = load_case_protocol(case) if protocol is None else protocol
    if not (results / "summary.json").is_file():
        raise PIC2DValidationError(f"{results} has no summary.json to assess")
    acceptance = protocol["stopping_rule"]["acceptance"]
    reference = protocol["reference_run"]["quantities"]
    hypotheses = HYPOTHESES_BY_CASE[case]
    config = runner.build_config(protocol, backend="cpu")
    grid = config.grid
    anode_v = float(protocol["operating_point"]["anode_potential_v"])
    limit = config.see.space_charge_limit_yield if config.see is not None else None
    run = run_quantities(results, grid, anode_v=anode_v, space_charge_limit_yield=limit)
    a_plateau = run["stop_reason"] == "plateau_reached_after_min_transit_times"
    windowed = run["windowed_residual_over_electrode_work"]
    b_ok = windowed is not None and bool(run["windowed_residual_window_complete"]) and windowed < 0.02
    plateau_status = "plateau_clean" if (a_plateau and b_ok) else "plateau_heating" if a_plateau else "no_plateau"
    shifts = _shift_rows(run, reference, hypotheses)
    verdict = hypothesis_verdict(a_plateau, shifts, KEY_QUANTITIES[case])
    consistency = None
    reference_cusps = None
    if reference_check:
        consistency = _consistency(reference, REFERENCE_RESULTS, grid)
        if consistency is not None and not all(entry["agree"] for entry in consistency.values()):
            raise PIC2DValidationError("reference_run.quantities disagree with the ss-v4 artifacts on disk: " + json.dumps({k: v for k, v in consistency.items() if not v["agree"]}))
        if (REFERENCE_RESULTS / "maps.npz").is_file():
            reference_cusps = run_quantities(REFERENCE_RESULTS, grid, anode_v=anode_v).get("per_cusp")
    cusp_rows = None
    if run.get("per_cusp") is not None and reference_cusps is not None:
        cusp_rows = []
        for mine, ref in zip(run["per_cusp"], reference_cusps, strict=True):
            row = {"z_c_m": mine["z_c_m"],
                   "electron_wall_current_a": {"value": mine["electron_wall_current_a"], "reference": ref["electron_wall_current_a"],
                                               "relative_shift": (mine["electron_wall_current_a"] - ref["electron_wall_current_a"]) / abs(ref["electron_wall_current_a"]) if ref["electron_wall_current_a"] else None},
                   "ion_wall_current_a": {"value": mine["ion_wall_current_a"], "reference": ref["ion_wall_current_a"]},
                   "sheath_drop_v": {"value": mine["sheath_drop_v"], "reference": ref["sheath_drop_v"], "difference_v": mine["sheath_drop_v"] - ref["sheath_drop_v"],
                                     "relative_shift": (mine["sheath_drop_v"] - ref["sheath_drop_v"]) / abs(ref["sheath_drop_v"]) if ref["sheath_drop_v"] else None,
                                     "hypothesis_sign": hypotheses.get("cusp_sheath_drop_v", {}).get("sign")},
                   "near_wall_drop_v": {"value": mine["near_wall_drop_v"], "reference": ref["near_wall_drop_v"]},
                   "near_wall_t_e_ev": {"value": mine["near_wall_t_e_ev"], "reference": ref["near_wall_t_e_ev"]},
                   "wall_ion_mean_energy_ev": {"value": mine.get("wall_ion_mean_energy_ev"), "reference": ref.get("wall_ion_mean_energy_ev")}}
            if "see_effective_yield" in mine:
                row["see"] = {k: mine.get(k) for k in ("see_effective_yield", "see_current_a", "see_mean_emitted_energy_ev", "space_charge_limited", "space_charge_limit_rule")}
            cusp_rows.append(row)
    record = {
        "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "case": case, "effects": protocol["campaign"]["effects"],
        "results_dir": results.relative_to(HERE).as_posix() if results.is_relative_to(HERE) else str(results), "git_head_now": runner.git_head(), "run": run,
        "reference": reference, "reference_case": REFERENCE_CASE, "reference_corrected_ledger": protocol["reference_run"]["corrected_ledger"], "reference_consistency": consistency,
        "a_plateau": {"passed": a_plateau, "stop_reason": run["stop_reason"], "ion_transit_times": run["ion_transit_times"], "plateau": run["plateau"], "rule": acceptance["a_plateau"]},
        "b_residual_power": {"passed": b_ok, "windowed_residual_over_electrode_work": windowed, "window_complete": run["windowed_residual_window_complete"], "bound": 0.02,
                             "one_sided": True, "ledger": "v2.0.6 W-corrected (native)", "cumulative_witness": run["cumulative_residual_over_electrode_work"],
                             "reference_reads": protocol["reference_run"]["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"], "rule": acceptance["b_residual_power"]},
        "c_shifts_vs_reference": shifts, "hypotheses": hypotheses, "key_quantities": list(KEY_QUANTITIES[case]), "per_cusp_vs_reference": cusp_rows,
        "plateau_status": plateau_status, "plateau_status_rule": acceptance["d_verdict"]["plateau_status"][plateau_status],
        "hypothesis_verdict": verdict, "hypothesis_verdict_rule": acceptance["d_verdict"]["per_case_hypothesis_verdict"][verdict],
        "peak_debye_window": {"cells_per_debye_last": run["cells_per_debye_window_last"], "trailing_mean": run["cells_per_debye_window_trailing_mean"], "soft_ok": run["peak_debye_soft_ok"]},
        "claim_boundary": protocol["claim_boundary"],
    }
    artifacts.write_canonical_json(output or (results / "assessment.json"), record)
    log(f"[assess] {case}: {plateau_status} (a {a_plateau}, b {b_ok} [{windowed}]) -> {verdict}; shifts vs {REFERENCE_CASE}: "
        + ", ".join(f"{k} {_fmt_shift(v)} {v['status']}" for k, v in shifts.items() if v["status"] != "unavailable"))
    return record


def _fmt_shift(row: dict[str, Any]) -> str:
    if row["shift"] is None:
        return "n/a"
    return f"{row['shift']:+.3f}" if row["kind"] == "absolute" else f"{row['shift'] * 100:+.1f}%"


def additivity(assessments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The combined-vs-sum-of-parts statement (acceptance d_verdict.combined_vs_sum_of_parts) from the three case assessments."""

    reached = {c: a for c, a in assessments.items() if a is not None and a["a_plateau"]["passed"]}
    if not all(c in reached for c in (*SINGLE_CASES, COMBINED_CASE)):
        return {"statement": "not_evaluable", "reason": f"requires all three cases at (a); reached: {sorted(reached)}", "rows": None}
    rows: dict[str, Any] = {}
    for key in QUANTITY_KEYS:
        band = ABSOLUTE_BAND.get(key, PARTICLE_BAND.get(key))
        parts = [assessments[c]["c_shifts_vs_reference"][key]["shift"] for c in SINGLE_CASES]
        combined = assessments[COMBINED_CASE]["c_shifts_vs_reference"][key]["shift"]
        if band is None or combined is None or any(p is None for p in parts) or not all(np.isfinite([combined, *parts])):
            rows[key] = {"combined": combined, "sum_of_parts": None if any(p is None for p in parts) else sum(parts), "interaction": None, "band": band, "classification": "reported"}
            continue
        total = float(sum(parts))
        interaction = float(combined) - total
        if abs(interaction) <= band:
            cls = "additive"
        elif interaction * total > 0:
            cls = "super_additive"
        else:
            cls = "sub_additive"
        rows[key] = {"combined": combined, "parts": dict(zip(SINGLE_CASES, parts, strict=True)), "sum_of_parts": total, "interaction": interaction, "band": band, "classification": cls}
    judged = [r["classification"] for r in rows.values() if r["classification"] != "reported"]
    statement = "additive" if all(c == "additive" for c in judged) else "interacting"
    return {"statement": statement, "rows": rows, "non_additive_quantities": [k for k, r in rows.items() if r["classification"] in ("super_additive", "sub_additive")]}


def assess_campaign(*, results_root: Path = RESULTS, cases_override: dict[str, Path] | None = None, output: Path | None = None, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """The three per-case verdicts and the additivity statement; unreached / unassessed cases are listed."""

    campaign = load_campaign()
    reference = campaign["reference_run"]["quantities"]
    points: dict[str, dict[str, Any]] = {REFERENCE_CASE: {"reached": True, "verdict": "reference (recorded ss-v4 plateau; (b) FAIL at +2.46 % corrected)",
                                                          "quantities": {k: reference.get(k) for k in QUANTITY_KEYS}}}
    assessments: dict[str, dict[str, Any] | None] = {}
    for case in CASES:
        results = (cases_override or {}).get(case, case_results(case, results_root))
        assessment_path = results / "assessment.json"
        entry: dict[str, Any] = {"effects": list(CASES[case]["effects"]), "reached": False, "plateau_status": None, "verdict": None, "quantities": None, "results_dir": str(results)}
        assessments[case] = None
        if assessment_path.is_file():
            assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
            assessments[case] = assessment
            entry.update({"plateau_status": assessment["plateau_status"], "verdict": assessment["hypothesis_verdict"], "reached": assessment["a_plateau"]["passed"],
                          "b_passed": assessment["b_residual_power"]["passed"], "quantities": {k: assessment["run"].get(k) for k in QUANTITY_KEYS},
                          "shifts": assessment["c_shifts_vs_reference"], "per_cusp": assessment.get("per_cusp_vs_reference"),
                          "see": assessment["run"].get("see"), "collision_set": assessment["run"].get("collision_set")})
        elif (results / "summary.json").is_file():
            entry["verdict"] = "not assessed (summary present; run `assess --case`)"
        points[case] = entry
    reached = [c for c in CASES if points[c]["reached"]]
    add = additivity(assessments)
    record = {
        "schema_version": CAMPAIGN_ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": EXPERIMENT_ID, "git_head_now": runner.git_head(), "points": points,
        "cases_reached": reached, "cases_unreached": [c for c in CASES if c not in reached],
        "verdicts": {c: points[c]["verdict"] for c in CASES}, "plateau_status": {c: points[c]["plateau_status"] for c in CASES},
        "additivity": add, "additivity_rule": campaign["acceptance"][COMBINED_CASE]["d_verdict"]["combined_vs_sum_of_parts"],
        "launch_priority": list(LAUNCH_PRIORITY),
        "note": "evaluates the assessed cases only; provisional until every case has a terminal state and final once all three are in",
    }
    output = output or (results_root / "campaign-assessment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.write_canonical_json(output, record)
    log(f"[assess --campaign] verdicts {record['verdicts']}; reached {reached}; additivity {add['statement']}")
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
    ass.add_argument("--campaign", action="store_true")
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
        if args.campaign or args.case is None:
            assess_campaign()
        else:
            assess_case(args.case, results=None if args.results is None else Path(args.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
