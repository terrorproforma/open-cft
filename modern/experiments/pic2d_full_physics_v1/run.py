"""Full physics v1: the preregistered R4 / R5 / R1-R5-combined campaign on the 33 um reference plateau.

Six cases (``protocols/<case>.json``) = the ss-v4 protocol with the v2.0.6 gates, K = 5, the model v2.1.1 drift-member arming latch, the
v2.0 ignition gate and the case's physics blocks (Coulomb; the spatial Knudsen gas with metastables at F = 1 and F = 10; every R2-R5
effect together at alpha = 0, 1/16, 0.345); the reference point of every shift table is the RECORDED ss-v4 plateau
(``pic2d_cft_steady_state_v4/results``, 0d228ad2), which fails its own acceptance (b) at +2.46 % on the corrected ledger.

Stages (from ``modern/`` with ``PYTHONPATH=src:.``)::

    python -m experiments.pic2d_full_physics_v1.run compose [--budget-from-preflight]            # (re)write protocols/*.json + protocol.json
    python -m experiments.pic2d_full_physics_v1.run preflight --case coulomb [--gpu-timing]        # -> preflight-<case>.json
    python -m experiments.pic2d_full_physics_v1.run shakedown --case coulomb                        # 100k steps -> finalize -> assess -> shakedown-<case>.json
    python -m experiments.pic2d_full_physics_v1.run launch --case coulomb --expect-commit SHA [--require-mps] [--resume]
    python -m experiments.pic2d_full_physics_v1.run status
    python -m experiments.pic2d_full_physics_v1.run finalize --case coulomb [...]                  # externally stopped run only
    python -m experiments.pic2d_full_physics_v1.run assess --case coulomb                          # per-case verdict -> results/<case>/assessment.json
    python -m experiments.pic2d_full_physics_v1.run assess --campaign                              # sustain table, alpha trend, additivity, F qualification

The stepping is the shared runner ``experiments.pic2d_cft_steady_state_v1.run``; the stages follow the v4 / alpha-series / physics-effects
preregistration discipline (clean worktree, expected commit, sealed protocol == recomposition, O_EXCL execution lock, preflight + shakedown records).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d import artifacts
from cft_revival.pic2d.coulomb import column_frequency_profile, coulomb_log_ee
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import PIC2DValidationError
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
from experiments.pic2d_full_physics_v1 import protocol as protocol_module
from experiments.pic2d_full_physics_v1.protocol import (
    ABSOLUTE_BAND,
    AT_RESULTS,
    CASES,
    CUSP_PLANES_M,
    EXPERIMENT_ID,
    F_PAIR,
    FULL_PHYSICS_CASES,
    HYPOTHESES_BY_CASE,
    KEY_QUANTITIES,
    LAUNCH_PRIORITY,
    MONOTONE_QUANTITIES,
    PARTICLE_BAND,
    PE_RESULTS,
    PLATEAU_SCALARS,
    QUANTITY_KEYS,
    REFERENCE_CASE,
    REPORTED_ONLY_SPATIAL,
    STEPS_TO_3_TRANSITS,
    compose_campaign,
    compose_case_protocol,
    ionization_centroid_from_maps,
    load_campaign,
    load_case_protocol,
    protocol_sha256,
    wall_power_and_ion_energy,
    write_sealed_protocols,
)
from experiments.pic2d_physics_effects_v1 import run as pe

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
RESULTS = HERE / "results"
LOCK_NAME = "execution-lock.json"
LOCK_SCHEMA = "cft-revival.pic2d-steady-state-execution-lock/1.0.0"
ASSESSMENT_SCHEMA = "cft-revival.pic2d-full-physics-v1.assessment/1.0.0"
CAMPAIGN_ASSESSMENT_SCHEMA = "cft-revival.pic2d-full-physics-v1.campaign-assessment/1.0.0"
PREFLIGHT_SCHEMA = "cft-revival.pic2d-full-physics-v1.preflight/1.0.0"
SHAKEDOWN_SCHEMA = "cft-revival.pic2d-full-physics-v1.shakedown/1.0.0"
REFERENCE_RESULTS = protocol_module.V4_RESULTS
SHAKEDOWN_OVERRIDES = pe.SHAKEDOWN_OVERRIDES
COULOMB_SERIES_KEYS = ("coulomb_nu_e_spitzer_peak_per_s", "coulomb_nu_e_spitzer_peak_over_nu_en", "coulomb_nu_ee_mean_per_s", "coulomb_nu_ei_mean_per_s",
                       "coulomb_nu_en_elastic_mean_per_s", "coulomb_nu_ee_over_nu_en", "coulomb_mean_s_ee", "coulomb_mean_s_ei", "coulomb_mean_coulomb_log_ee",
                       "coulomb_mean_coulomb_log_ei", "coulomb_interval_ee_pairs", "coulomb_interval_ei_pairs")
COULOMB_CUMULATIVE_KEYS = ("coulomb_ee_pairs", "coulomb_ei_pairs", "coulomb_ii_pairs", "coulomb_cycles", "pz_coulomb", "ke_coulomb_j")
NEUTRAL_CUMULATIVE_KEYS = ("neutral_substeps", "neutral_fed", "neutral_ionized", "neutral_effused", "neutral_recycled", "neutral_fast_in", "neutral_excited_to_pool",
                           "meta_ionized", "meta_superelastic", "meta_wall_deexcited", "meta_effused", "stepwise_ionizations", "superelastic", "ionizations", "cex")
ANOMALOUS_CUMULATIVE_KEYS = ("anomalous",)
SUSTAIN_PROBE_TIMES_S = (0.1e-6, 0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6)
EXTINCTION_FRACTION = 0.25
DENSE_SEED_DENSITY = 3.0e17          # the second (denser) synthetic plateau load timed for the spatial cases (~7.7 M particles)


def preflight_path(case: str) -> Path:
    return HERE / f"preflight-{case}.json"


def shakedown_path(case: str) -> Path:
    return HERE / f"shakedown-{case}.json"


def case_results(case: str, results: Path = RESULTS) -> Path:
    return results / case


def _log(text: str) -> None:
    print(text, flush=True)


compute_apps = pe.compute_apps


def _effect_identity(config) -> dict[str, Any]:
    return {"see": None if config.see is None else config.see.to_dict(),
            "collision_set": None if config.mcc is None or config.mcc.collision_set is None else config.mcc.collision_set.to_dict(),
            "coulomb": None if config.coulomb is None else config.coulomb.to_dict(),
            "neutrals_spatial": None if config.neutrals_spatial is None else config.neutrals_spatial.to_dict(),
            "mcc_ceiling_per_m3": None if config.mcc is None else config.mcc.neutral_density_per_m3,
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
    key = f"modern/experiments/pic2d_full_physics_v1/protocols/{case}.json"
    on_disk = protocol_sha256(sealed)
    if campaign["sealed_protocols"].get(key) != on_disk:
        raise PIC2DValidationError(f"campaign protocol.json lists {campaign['sealed_protocols'].get(key)} for {case}, on disk {on_disk}")
    return sealed


# -- preflight --------------------------------------------------------------------------------------------------------------

def _effect_events(cumulative: dict[str, Any]) -> dict[str, Any]:
    keys = ("see_impacts", "see_electrons", "cex", "mex", "excitations_level_1", "coulomb_ee_pairs", "coulomb_ei_pairs", "anomalous", "neutral_substeps",
            "neutral_ionized", "stepwise_ionizations", "meta_ionized")
    return {k: cumulative.get(k) for k in keys if k in cumulative}


def _neutral_state(sim: Simulation) -> dict[str, Any] | None:
    if not getattr(sim, "spatial_neutrals_on", False):
        return None
    state = sim.state
    particles = getattr(state, "neutral_particles", None)
    out: dict[str, Any] = {}
    if particles is not None:
        for name in ("count", "capacity"):
            value = getattr(particles, name, None)
            if value is not None:
                out[name] = int(value)
        arrays = [getattr(particles, k, None) for k in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s", "weight", "state")]
        out["host_bytes"] = int(sum(int(a.nbytes) for a in arrays if isinstance(a, np.ndarray)))
    return out or None


def preflight(case: str, *, backend: str = "warp-cuda", timing_steps: int = 2000, loaded_seed_density: float = 1.75e17, dense_seed_density: float | None = None,
              gpu_timing: bool = True, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """Real inputs on the launch box: field, mesh, factorisation, memory (incl. the neutral particles), ms/step at the seed and plateau loads; the budget derivation.
    Spatial cases are timed at a second, denser synthetic load as well (their peak density is expected to rise). Non-evidentiary."""

    protocol = load_case_protocol(case)
    meta = CASES[case]
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
        "arming_and_ignition": {"drift_members_arming": {k: v for k, v in protocol["stopping_rule"]["grid_heating_triad"]["drift_members_arming"].items() if k != "note"},
                                "ignition_gate": {k: v for k, v in protocol["stopping_rule"]["ignition_gate"].items() if k != "note"}},
    }
    t0 = time.perf_counter()
    field_map, cross_sections = runner.load_inputs(config, None, None, protocol=protocol)
    record["field"] = {"sha256": field_map.sha256, "source_sha256": getattr(field_map, "source_sha256", None), "max_b_t": field_map.max_b_t, "seconds": time.perf_counter() - t0}
    record["cross_sections_sha256"] = cross_sections.payload_sha256 if cross_sections is not None else None
    record["cross_sections_processes"] = [getattr(p, "identifier", str(p)) for p in getattr(cross_sections, "processes", [])] if cross_sections is not None else None
    masks = build_mesh_masks(grid)
    record["mesh"] = masks.to_dict()
    if config.neutrals_spatial is not None:
        ceiling = float(config.mcc.neutral_density_per_m3)
        record["neutral_ceiling_admissibility"] = {
            "mcc_ceiling_per_m3": ceiling, "knudsen_anode_density_per_m3": protocol_module.KNUDSEN_ANODE_DENSITY_PER_M3,
            "ceiling_over_knudsen_anode": ceiling / protocol_module.KNUDSEN_ANODE_DENSITY_PER_M3,
            "passes": ceiling > protocol_module.KNUDSEN_ANODE_DENSITY_PER_M3,
            "time_acceleration": config.neutrals_spatial.time_acceleration, "macro_weight": config.neutrals_spatial.macro_weight,
        }
        if ceiling <= protocol_module.KNUDSEN_ANODE_DENSITY_PER_M3:
            raise PIC2DValidationError(f"the MCC ceiling {ceiling:.3g} must exceed the Knudsen anode density {protocol_module.KNUDSEN_ANODE_DENSITY_PER_M3:.3g} (fail-closed)")
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
    record["neutral_particles_at_seed"] = _neutral_state(sim)
    log(f"[preflight] {case} ({', '.join(protocol['campaign']['effects'])}): grid {grid.cell_shape}; field {field_map.sha256[:12]} max |B| {field_map.max_b_t:.3f} T; "
        f"factorisation {record['factorisation_seconds']:.1f} s; seed {seed_state.electrons.count} e-; neutrals {record['neutral_particles_at_seed']}; MPS clients before: {len(others)}; "
        f"GPU before: {gpu_before}")
    timing_seed = _time_steps(sim, timing_steps, warmup=200)
    after = device_memory()
    timing_seed.update({"electrons_after": sim.state.electrons.count, "ions_after": sim.state.ions.count, "step_graph": sim.step_graph_state(),
                        "effect_events": _effect_events(sim.state.cumulative), "neutral_particles_after": _neutral_state(sim)})
    record["timing_seed_load"] = timing_seed
    record["last_series_record"] = sim.series[-1].to_dict() if sim.series else None
    log(f"[preflight] seed load: {timing_seed['ms_per_step']:.3f} ms/step over {timing_steps} steps ({sim.state.electrons.count} e-; events {timing_seed['effect_events']})")
    del sim
    loads = [("timing_plateau_load", loaded_seed_density)]
    if dense_seed_density is None and meta["neutrals"]:
        dense_seed_density = DENSE_SEED_DENSITY
    if dense_seed_density is not None:
        loads.append(("timing_dense_plateau_load", dense_seed_density))
    memory_after: dict[str, Any] = {}
    basis_ms = None
    basis_label = None
    for label, density in loads:
        loaded = copy.deepcopy(protocol)
        loaded["operating_point"]["seed_plasma_density_per_m3"] = density
        loaded_config = runner.build_config(loaded, backend=backend)
        sim2 = Simulation(loaded_config, field_map, cross_sections=cross_sections, backend=backend, step_graph=runner.step_graph_flag(protocol))
        loaded_state = sim2.state
        timing = _time_steps(sim2, timing_steps, warmup=200)
        memory_after[label] = device_memory()
        timing.update({"seed_density_per_m3": density, "electrons": loaded_state.electrons.count, "ions": loaded_state.ions.count,
                       "electrons_after": sim2.state.electrons.count, "ions_after": sim2.state.ions.count, "step_graph": sim2.step_graph_state(),
                       "effect_events": _effect_events(sim2.state.cumulative), "neutral_particles_after": _neutral_state(sim2)})
        record[label] = timing
        log(f"[preflight] {label}: {timing['ms_per_step']:.3f} ms/step over {timing_steps} steps ({loaded_state.electrons.count} e- + {loaded_state.ions.count} i)")
        if basis_ms is None or timing["ms_per_step"] > basis_ms:
            basis_ms, basis_label = timing["ms_per_step"], label
        del sim2
    plateau = record["timing_plateau_load"]
    record["memory"] = {
        "device_before": before, "device_after_seed_run": after, "device_after_loaded_runs": memory_after,
        "device_used_by_plateau_load_bytes": None if before is None or memory_after.get("timing_plateau_load") is None
        else before["free_bytes"] - memory_after["timing_plateau_load"]["free_bytes"],
        "host_peak_working_set_bytes": peak_working_set_bytes(),
        "neutral_memory_note": ("the spatial cases' device pool includes the neutral particle arrays (~4 M macro-neutrals x 7 arrays) and the 13 per-cell neutral fields; compare "
                                "device_used_by_plateau_load_bytes with the coulomb case's to read the neutral share (declared ~0.5 GB)"),
    }
    ms = float(basis_ms)
    hours_3 = STEPS_TO_3_TRANSITS * ms / 3.6e6
    budget = float(np.ceil(1.5 * STEPS_TO_3_TRANSITS * ms / 1e3 / 600.0) * 600.0)      # 1.5 x, rounded up to 10 min
    record["projection"] = {"steps_to_3_transits": STEPS_TO_3_TRANSITS, "hours_to_3_transits_at_plateau_load": STEPS_TO_3_TRANSITS * plateau["ms_per_step"] / 3.6e6,
                            "hours_to_3_transits_at_budget_basis": hours_3, "budget_basis_load": basis_label,
                            "hours_to_3_transits_at_seed_load": STEPS_TO_3_TRANSITS * timing_seed["ms_per_step"] / 3.6e6,
                            "ms_per_step_per_million_particles": (plateau["ms_per_step"] - timing_seed["ms_per_step"]) / max((plateau["electrons"] + plateau["ions"]
                                                                                                                              - seed_state.electrons.count - seed_state.ions.count) / 1e6, 1e-9)}
    record["budget_derivation"] = {"wall_budget_seconds": budget, "basis_ms_per_step": ms, "basis_load": basis_label, "factor": 1.5,
                                   "note": f"{budget / 3600:.1f} h = 1.5 x {hours_3:.2f} h (launch-box {basis_label} preflight {ms:.2f} ms/step with {len(others)} other MPS "
                                           f"clients x {STEPS_TO_3_TRANSITS} steps to 3 transits), rounded up to 10 min; preflight-{case}.json"}
    ref = protocol["reference_run"]["quantities"]
    record["expected_at_v4_peak"] = {"peak_n_e_per_m3": ref["peak_n_e_window_per_m3"], "t_e_peak_ev": ref["t_e_peak_window_ev"],
                                     "cells_per_debye": ref["cells_per_debye_at_peak_window"],
                                     "hypothesis": {k: v["sign"] for k, v in HYPOTHESES_BY_CASE[case].items() if k in ("peak_n_e_window_per_m3", "t_e_peak_window_ev")}}
    artifacts.write_canonical_json(preflight_path(case), record)
    log(f"[preflight] projection {hours_3:.2f} h to 3 transits at the {basis_label}; budget {budget / 3600:.1f} h; written {preflight_path(case)}")
    return record


# -- shakedown --------------------------------------------------------------------------------------------------------------

shakedown_protocol = pe.shakedown_protocol


def shakedown(case: str, *, results: Path | None = None, backend: str = "warp-cuda", max_steps: int = SHAKEDOWN_OVERRIDES["max_steps"], reuse_run: bool = False,
              log: Callable[[str], None] = _log) -> dict[str, Any]:
    """100 000-step real-input run of the case (cadences shrunk) through finalize -> assess (case + campaign) -> refinalize; ``reuse_run`` skips the stepping when the
    results directory already holds a completed run of the byte-identical shakedown protocol (the assessment / record stages re-run on the existing artifacts)."""

    protocol = load_case_protocol(case)
    results = HERE / "results-shakedown" / case if results is None else results
    p = shakedown_protocol(protocol)
    shake_protocol_path = results / "protocol-shakedown.json"
    clients_before = compute_apps()
    if reuse_run:
        if not (results / "summary.json").is_file() or not shake_protocol_path.is_file():
            raise PIC2DValidationError(f"--reuse-run needs a completed run in {results}")
        if artifacts.read_canonical_json(shake_protocol_path) != p:
            raise PIC2DValidationError("--reuse-run refused: the existing run's shakedown protocol differs from the current one")
        summary_path = results / "summary.json"
        summary = artifacts.read_canonical_json(summary_path)
        run_seconds = float(sum(float(s.get("wall_seconds", 0.0) or 0.0) for s in (summary.get("sessions") or [])) or summary.get("wall_seconds_total") or 0.0)
        log(f"[shakedown] {case}: reusing the completed run in {results} (git head of the run {summary.get('git_head')})")
    else:
        if results.exists():
            shutil.rmtree(results)
        results.mkdir(parents=True)
        artifacts.write_canonical_json(shake_protocol_path, p)
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
    meta = CASES[case]
    record = {
        "schema_version": SHAKEDOWN_SCHEMA, "utc": utc_now(), "git_head": runner.git_head(), "run_git_head": summary.get("git_head"), "run_reused": bool(reuse_run),
        "non_evidentiary": True, "case": case, "effects": protocol["campaign"]["effects"],
        "host": socket.gethostname(), "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), "concurrent_mps_clients": len(others),
        "concurrent_mps_client_pids": [a["pid"] for a in others], "overrides": {**SHAKEDOWN_OVERRIDES, "max_steps": max_steps}, "results_dir": results.relative_to(HERE).as_posix(),
        "run_seconds": run_seconds, "refinalize_seconds": refinalize_seconds, "steps_completed": summary["steps_completed"], "stop_reason": summary["stop_reason"],
        "stability_gate_message": summary.get("stability_gate_message"),
        "ms_per_step": summary["ms_per_step_this_session"], "final_counts": summary["final_counts"], "config_sha256": (summary.get("provenance") or {}).get("config_sha256"),
        "frames": summary["artifacts"]["frames"]["count"] if summary["artifacts"].get("frames") else 0,
        "effect_cumulative": {k: cumulative.get(k) for k in pe.SEE_CUMULATIVE_KEYS + pe.XE_CUMULATIVE_KEYS + COULOMB_CUMULATIVE_KEYS + NEUTRAL_CUMULATIVE_KEYS + ANOMALOUS_CUMULATIVE_KEYS
                             if k in cumulative},
        "see_readings": run.get("see"), "collision_readings": run.get("collision_set"), "coulomb_readings": run.get("coulomb"), "neutral_readings": run.get("neutrals"),
        "anomalous_readings": run.get("anomalous"), "sustain_readings": run.get("sustain"), "iedf_exit_plane": run.get("iedf"), "per_cusp": run.get("per_cusp"),
        "ionization_centroid": run.get("ionization_centroid_detail"),
        "peak_debye_window": {"records": len(windows), "enforced_records": len(enforced), "last": windows[-1] if windows else None,
                              "max_cells_per_debye_enforced": max((w["cells_per_debye"] for w in enforced), default=None),
                              "floor_kind": "accumulated_particle_steps" if windows and windows[-1].get("min_accumulated_macro_particle_steps_at_peak") else "mean_occupancy"},
        "windowed_residual": {"records_with_complete_window": len(complete),
                              "last": None if not complete else {k: complete[-1][k] for k in complete[-1] if k.startswith("windowed") or k == "energy_residual_over_electrode_work"}},
        "drift_members_arming_last": None if not triads else triads[-1].get("drift_members_arming"),
        "ignition_last": summary.get("ignition"),
        "plateau_keys": sorted(summary["plateau"]) if summary.get("plateau") else None,
        "summary_keys_present": {key: (summary.get(key) is not None) for key in ("plateau", "grid_heating_triad", "peak_node_debye", "neutral_inventory", "ledger", "window_currents_a", "ignition")},
        "assessment": {k: assessment[k] for k in ("plateau_status", "stop_class", "hypothesis_verdict", "a_plateau", "b_residual_power", "reference_consistency", "sustain")},
        "campaign_assessment": {"cases_reached": campaign["cases_reached"], "additivity": campaign["additivity"]["statement"], "f_qualification": campaign["f_qualification"]["statement"],
                                "sustain_table": campaign["sustain"]["table"]},
        "artifacts": {k: summary["artifacts"].get(k) for k in ("maps_npz_sha256", "series_npz_sha256")},
        "gate_not_inert_check": {"peak_window_enforced_at_least_once": bool(enforced), "peak_window_resolved_nodes_last": windows[-1].get("resolved_nodes") if windows else None,
                                 "residual_window_completed_at_least_once": bool(complete),
                                 "arming_latch_present": bool(triads and triads[-1].get("drift_members_arming") is not None),
                                 "ignition_gate_evaluated": summary.get("ignition") is not None,
                                 "see_events_nonzero": (cumulative.get("see_electrons", 0) or 0) > 0 if meta["see"] else None,
                                 "cex_events_nonzero": (cumulative.get("cex", 0) or 0) > 0 if meta["collision_set"] else None,
                                 "level_split_nonzero": all((cumulative.get(f"excitations_level_{i}", 0) or 0) > 0 for i in (1, 2, 3, 4)) if meta["collision_set"] else None,
                                 "coulomb_pairs_nonzero": (cumulative.get("coulomb_ee_pairs", 0) or 0) > 0 and (cumulative.get("coulomb_ei_pairs", 0) or 0) > 0 if meta["coulomb"] else None,
                                 "neutral_substeps_nonzero": (cumulative.get("neutral_substeps", 0) or 0) > 0 if meta["neutrals"] else None,
                                 "stepwise_ionizations_nonzero": (cumulative.get("stepwise_ionizations", 0) or 0) > 0 if meta["neutrals"] else None,
                                 "anomalous_events_nonzero": (cumulative.get("anomalous", 0) or 0) > 0 if meta["alpha"] > 0 else None,
                                 "neutral_ceiling_violation_fraction_max": (run.get("neutrals") or {}).get("ceiling_violation_fraction_max") if meta["neutrals"] else None},
    }
    artifacts.write_canonical_json(shakedown_path(case), record)
    log(f"[shakedown] {case}: {summary['steps_completed']} steps, {summary['stop_reason']}, {summary['ms_per_step_this_session']:.2f} ms/step, {record['frames']} frames, "
        f"events {record['effect_cumulative']}, peak window enforced {len(enforced)}/{len(windows)} (max {record['peak_debye_window']['max_cells_per_debye_enforced']}), "
        f"residual windows complete {len(complete)}; plateau {assessment['plateau_status']}, verdict {assessment['hypothesis_verdict']}; sustain {assessment['sustain']}; "
        f"written {shakedown_path(case)}")
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

def _trailing_mean(values: np.ndarray, fraction: float = 0.2) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    tail = values[-max(int(values.size * fraction), 1):]
    tail = tail[np.isfinite(tail)]
    return float(np.mean(tail)) if tail.size else None


def _load_series(results: Path) -> dict[str, np.ndarray] | None:
    path = results / "series.npz"
    if not path.is_file():
        return None
    with np.load(path) as s:
        return {k: np.asarray(s[k]) for k in s.files}


def coulomb_readings(series: dict[str, np.ndarray] | None, maps: dict[str, np.ndarray], grid, coulomb_log_floor: float = 2.0) -> dict[str, Any] | None:
    """Trailing-20 % means of the series' Coulomb block; nu_ee / nu_ei (pair-mean) and the NRL Spitzer nu_e at the window's peak cell and in the cusp columns."""

    if series is None or "coulomb_nu_ee_mean_per_s" not in series:
        return None
    out: dict[str, Any] = {"trailing_20pct_means": {k.removeprefix("coulomb_"): _trailing_mean(series[k]) for k in COULOMB_SERIES_KEYS if k in series}}
    if "coulomb_nu_ee_per_s" not in maps:
        out["maps"] = None
        return out
    nu_ee, nu_ei, seconds = maps["coulomb_nu_ee_per_s"], maps["coulomb_nu_ei_per_s"], maps["coulomb_electron_seconds"]
    n_e = maps["n_e_per_m3"]
    cells = 0.25 * (n_e[:-1, :-1] + n_e[1:, :-1] + n_e[:-1, 1:] + n_e[1:, 1:])
    weight = seconds[:-1, :-1]
    cells_masked = np.where(weight > 0.0, cells, -1.0)
    i, j = np.unravel_index(int(np.argmax(cells_masked)), cells_masked.shape)
    maps_out: dict[str, Any] = {
        "peak_cell": {"cell": [int(i), int(j)], "r_m": float((i + 0.5) * grid.dr_m), "z_m": float(grid.geometry.z_min_m + (j + 0.5) * grid.dz_m),
                      "n_e_window_per_m3": float(cells_masked[i, j]), "nu_ee_pair_mean_per_s": float(nu_ee[i, j]), "nu_ei_pair_mean_per_s": float(nu_ei[i, j]),
                      "electron_seconds": float(weight[i, j])},
        "cusp_columns_nu_ee_pair_mean_per_s": column_frequency_profile(nu_ee, seconds, grid, CUSP_PLANES_M),
        "cusp_columns_nu_ei_pair_mean_per_s": column_frequency_profile(nu_ei, seconds, grid, CUSP_PLANES_M),
        "definition_note": ("coulomb_nu_*_per_s are the operator's pair-mean deflection rates <s> / dt_c (a 1/g^3-weighted mean: heavy-tailed, several times the thermal rate); "
                            "nu_e_spitzer_* is the NRL electron collision rate 2.91e-6 n lnL T^-3/2 from the window n_e / T_e maps (the audit's gap-(d) definition)"),
    }
    resolved = weight > 0.0
    if resolved.any():
        maps_out["electron_weighted_mean_nu_ee_pair_mean_per_s"] = float(np.sum(nu_ee[:-1, :-1][resolved] * weight[resolved]) / weight[resolved].sum())
    if "t_e_ev" in maps:
        t_nodes = maps["t_e_ev"]
        t_cells = 0.25 * (t_nodes[:-1, :-1] + t_nodes[1:, :-1] + t_nodes[:-1, 1:] + t_nodes[1:, 1:])
        valid = (cells > 0.0) & (t_cells > 0.0)
        lnl = np.where(valid, coulomb_log_ee(np.where(valid, cells, 1.0), np.where(valid, t_cells, 1.0), coulomb_log_floor), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            spitzer = np.where(valid, 2.91e-6 * cells * 1e-6 * lnl * np.where(valid, t_cells, 1.0) ** -1.5, 0.0)
        spitzer_nodes = np.zeros(nu_ee.shape)
        spitzer_nodes[:-1, :-1] = spitzer
        maps_out["peak_cell"].update({"t_e_window_ev": float(t_cells[i, j]), "nu_e_spitzer_per_s": float(spitzer[i, j]), "coulomb_log_ee": float(lnl[i, j])})
        maps_out["cusp_columns_nu_e_spitzer_per_s"] = column_frequency_profile(spitzer_nodes, seconds, grid, CUSP_PLANES_M)
        if resolved.any():
            maps_out["electron_weighted_mean_nu_e_spitzer_per_s"] = float(np.sum(spitzer[resolved] * weight[resolved]) / weight[resolved].sum())
    out["maps"] = maps_out
    return out


def _axial_profile(cell_map: np.ndarray, radial_cells: int | None) -> list[float]:
    values = np.asarray(cell_map, dtype=np.float64)
    inner = values if radial_cells is None else values[:radial_cells]
    return [float(v) for v in np.mean(inner, axis=0)]


def neutral_readings(summary: dict[str, Any], series: dict[str, np.ndarray] | None, maps: dict[str, np.ndarray], grid) -> dict[str, Any] | None:
    """The spatial gas: channel-mean / anode / exit densities, the axis profile, depletion vs the initial profile, metastable fraction / stepwise share, ledger closure."""

    inventory = summary.get("neutral_inventory") or {}
    if inventory.get("model") != "neutrals_spatial_v1" and "final_axis_density_anode_per_m3" not in inventory:
        return None
    out: dict[str, Any] = {
        "model": inventory.get("model"), "time_acceleration": inventory.get("time_acceleration"),
        "channel_mean_density_trailing_per_m3": inventory.get("trailing_20pct_mean_density_per_m3"),
        "final_channel_mean_density_per_m3": inventory.get("final_channel_mean_density_per_m3"),
        "final_axis_density_anode_per_m3": inventory.get("final_axis_density_anode_per_m3"), "final_axis_density_exit_per_m3": inventory.get("final_axis_density_exit_per_m3"),
        "gross_utilisation_trailing": inventory.get("gross_utilisation_trailing"), "net_utilisation_trailing": inventory.get("net_utilisation_trailing"),
        "effusion_rate_trailing_per_s": inventory.get("trailing_20pct_mean_effusion_rate_per_s"), "recycled_rate_trailing_per_s": inventory.get("trailing_20pct_mean_recycled_rate_per_s"),
        "neutral_exit_thrust_trailing_n": inventory.get("trailing_20pct_mean_neutral_exit_thrust_n"), "neutral_time_s_total": inventory.get("neutral_time_s_total"),
        "ledger": {k: inventory.get(k) for k in ("cumulative_ledger_atoms_neutral_time", "cumulative_ledger_atoms_real_time_plasma_terms", "max_interval_ledger_residual_atoms",
                                                 "max_interval_meta_ledger_residual_atoms", "max_sink_consistency_atoms", "final_debt_ground_atoms")},
        "metastables": inventory.get("metastables"),
    }
    if series is not None:
        if "neutral_density_per_m3" in series and series["neutral_density_per_m3"].size:
            n0 = float(series["neutral_density_per_m3"][0])
            n_tail = _trailing_mean(series["neutral_density_per_m3"])
            out["initial_channel_mean_density_per_m3"] = n0
            out["depletion_fraction"] = None if n_tail is None or n0 <= 0.0 else (n0 - n_tail) / n0
        if "neutral_ceiling_violation_fraction" in series:
            v = series["neutral_ceiling_violation_fraction"]
            out["ceiling_violation_fraction_max"] = float(np.nanmax(v)) if v.size else None
            out["ceiling_violation_fraction_trailing"] = _trailing_mean(v)
        if "neutral_density_max_per_m3" in series:
            out["density_max_trailing_per_m3"] = _trailing_mean(series["neutral_density_max_per_m3"])
        if "neutral_axis_density_anode_per_m3" in series and "neutral_axis_density_exit_per_m3" in series:
            a, e = _trailing_mean(series["neutral_axis_density_anode_per_m3"]), _trailing_mean(series["neutral_axis_density_exit_per_m3"])
            out["axis_anode_over_exit_trailing"] = None if not a or not e else a / e
    if "neutral_density_per_m3" in maps:
        density = np.asarray(maps["neutral_density_per_m3"], dtype=np.float64)
        out["axis_density_profile_per_m3"] = _axial_profile(density, 1)
        out["inner_third_density_profile_per_m3"] = _axial_profile(density, max(density.shape[0] // 3, 1))
        out["profile_z_m"] = [float(grid.geometry.z_min_m + (k + 0.5) * grid.dz_m) for k in range(density.shape[1])]
        out["anode_over_exit_axis_window"] = float(density[0, 0] / density[0, -1]) if density[0, -1] > 0 else None
        out["cusp_plane_density"] = []
        for z_c in CUSP_PLANES_M:
            j = max(0, min(density.shape[1] - 1, int((z_c - grid.geometry.z_min_m) / grid.dz_m)))
            out["cusp_plane_density"].append({"z_c_m": float(z_c), "axis_per_m3": float(density[0, j]), "column_mean_per_m3": float(np.mean(density[:, j][density[:, j] > 0])) if np.any(density[:, j] > 0) else None})
        if "metastable_density_per_m3" in maps:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(density > 0, np.asarray(maps["metastable_density_per_m3"], dtype=np.float64) / np.maximum(density, 1e-300), 0.0)
            out["axis_metastable_fraction_profile"] = _axial_profile(ratio, 1)
            out["metastable_fraction_max"] = float(ratio.max())
            for entry in out["cusp_plane_density"]:
                j = max(0, min(density.shape[1] - 1, int((entry["z_c_m"] - grid.geometry.z_min_m) / grid.dz_m)))
                entry["metastable_fraction_axis"] = float(ratio[0, j])
    return out


def sustain_readings(summary: dict[str, Any], series: dict[str, np.ndarray] | None) -> dict[str, Any]:
    """The ignition gate's record plus N_e / I_d / S at fixed probe times and their ratios to the 0.05-0.2 us reference; the late-extinction diagnostic."""

    out: dict[str, Any] = {"ignition": summary.get("ignition"), "stop_reason": summary.get("stop_reason")}
    if series is None or "electrons" not in series or series["electrons"].size < 2:
        return out
    t = np.asarray(series["time_s"], dtype=np.float64)
    n_e = np.asarray(series["electrons"], dtype=np.float64)
    i_d = np.asarray(series["current_discharge_a"], dtype=np.float64) if "current_discharge_a" in series else None
    s_rate = np.asarray(series["current_ionization_rate_per_s"], dtype=np.float64) if "current_ionization_rate_per_s" in series else None
    ref = (t >= 0.05e-6) & (t < 0.2e-6)
    n_ref = float(n_e[ref].mean()) if ref.any() else None
    s_ref = float(s_rate[ref].mean()) if ref.any() and s_rate is not None else None
    probes = []
    for tp in SUSTAIN_PROBE_TIMES_S:
        if float(t[-1]) < tp:
            break
        mask = (t >= tp - 0.05e-6) & (t <= tp)
        if not mask.any():
            continue
        entry = {"time_s": tp, "electrons": float(n_e[mask].mean()), "electron_ratio": None if not n_ref else float(n_e[mask].mean()) / n_ref}
        if i_d is not None:
            entry["discharge_a"] = float(i_d[mask].mean())
        if s_rate is not None:
            entry["ionization_rate_per_s"] = float(s_rate[mask].mean())
            entry["s_ratio"] = None if not s_ref else float(s_rate[mask].mean()) / s_ref
        probes.append(entry)
    out.update({"reference_window_s": [0.05e-6, 0.2e-6], "electrons_reference": n_ref, "s_reference_per_s": s_ref, "probes": probes, "time_s_last": float(t[-1]),
                "electrons_last": float(n_e[-1]), "electrons_max": float(n_e.max())})
    # direction over the last quarter of the record (the shakedowns end at 0.14 us: a rising / falling reading, not the gate)
    q = max(n_e.size // 4, 2)
    if n_e.size >= 2 * q:
        first, last = float(n_e[:q].mean()), float(n_e[-q:].mean())
        out["electron_direction_last_quarter"] = {"first_quarter_mean": first, "last_quarter_mean": last, "ratio": last / first if first > 0 else None,
                                                  "reading": "rising" if last > 1.05 * first else "falling" if last < 0.95 * first else "flat"}
    n_tail = _trailing_mean(n_e)
    late = {"trailing_20pct_electrons_over_max": None if n_tail is None or n_e.max() <= 0 else n_tail / float(n_e.max())}
    if i_d is not None and i_d.size >= 20:
        blocks = np.array_split(i_d, 20)
        block_means = np.array([float(np.nanmean(b)) for b in blocks if b.size])
        i_tail = _trailing_mean(i_d)
        late["trailing_20pct_discharge_over_running_max"] = None if i_tail is None or block_means.max() <= 0 else i_tail / float(block_means.max())
    late["late_extinction"] = bool(late["trailing_20pct_electrons_over_max"] is not None and late["trailing_20pct_electrons_over_max"] < EXTINCTION_FRACTION
                                   and late.get("trailing_20pct_discharge_over_running_max") is not None and late["trailing_20pct_discharge_over_running_max"] < EXTINCTION_FRACTION)
    out["late_extinction_diagnostic"] = late
    return out


def classify_stop(summary: dict[str, Any]) -> str:
    reason = str(summary.get("stop_reason") or "")
    message = str(summary.get("stability_gate_message") or "").lower()
    if reason == "plateau_reached_after_min_transit_times":
        return "plateau"
    if reason == "no_ignition":
        return "no_ignition"
    if reason == "wall_clock_budget_reached":
        return "budget"
    if reason == "grid_heating_triad_gate_stopped_run":
        return "residual_power" if "windowed energy residual" in message or "energy residual" in message else "triad_drift"
    if reason == "runtime_stability_gate_stopped_run":
        return "peak_debye_gate" if ("debye" in message or "lambda_d" in message) else "runtime_stability_gate"
    return reason or "other"


def run_quantities(results: Path, grid=None, *, anode_v: float = 300.0, space_charge_limit_yield: float | None = None, coulomb_log_floor: float = 2.0) -> dict[str, Any]:
    """Every quantity of the shift table plus the effect readings, from summary.json / maps.npz / series.npz."""

    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    peak = _peak_from_maps(results / "maps.npz")
    triad = summary.get("grid_heating_triad") or {}
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
    currents = summary["window_currents_a"]
    cumulative = ((summary.get("final_series") or {}).get("ledger") or {}).get("cumulative") or {}
    inventory = summary.get("neutral_inventory") or {}
    out: dict[str, Any] = {
        "discharge_current_a": currents["discharge_a"],
        "exit_ion_beam_a": currents["exit_ion_beam_a"],
        "ionization_rate_per_s": inventory.get("trailing_20pct_mean_ionization_rate_per_s"),
        "gross_utilisation": inventory.get("propellant_utilisation_trailing"),
        "neutral_density_per_m3": inventory.get("trailing_20pct_mean_density_per_m3"),
        "peak_n_e_window_per_m3": peak["peak_n_e_window_per_m3"], "t_e_peak_window_ev": peak["t_e_peak_window_ev"], "peak_node": peak["node"],
        "anode_ion_a": currents.get("anode_ion_a"),
        "wall_electron_a": currents.get("wall_electron_a"), "wall_ion_a": currents.get("wall_ion_a"),
        "stop_reason": summary["stop_reason"], "stop_class": classify_stop(summary), "stability_gate_message": summary.get("stability_gate_message"),
        "ion_transit_times": summary["ion_transit_times"], "steps_completed": summary["steps_completed"], "simulated_time_s": summary.get("simulated_time_s"),
        "plateau": summary.get("plateau"),
        "windowed_residual_over_electrode_work": triad.get("windowed_energy_residual_over_electrode_work"),
        "windowed_residual_window_complete": triad.get("windowed_energy_residual_window_complete"),
        "cumulative_residual_over_electrode_work": triad.get("energy_residual_over_electrode_work"),
        "drift_members_arming": triad.get("drift_members_arming"),
        "cells_per_debye_window_last": debye.get("cells_per_debye_window_last"), "cells_per_debye_window_trailing_mean": debye.get("trailing_20pct_mean_cells_per_debye_window"),
        "peak_debye_soft_ok": debye.get("soft_ok"), "maps_kind": summary.get("maps_kind"), "sessions": len(summary.get("sessions") or []),
        "git_head": summary.get("git_head"), "protocol_sha256": summary.get("protocol_sha256"), "config_sha256": (summary.get("provenance") or {}).get("config_sha256"),
        "iedf_low_energy_fraction": None, "wall_electron_power_w": None, "wall_ion_mean_energy_ev": None,
        "ionization_centroid_z_m": None, "neutral_density_anode_over_exit": None, "neutral_depletion_fraction": None, "metastable_fraction_of_ground": None,
        "stepwise_fraction_of_ionization": None, "nu_e_spitzer_peak_over_nu_en": None,
    }
    with np.load(results / "maps.npz") as archive:
        maps = {k: np.asarray(archive[k]) for k in archive.files}
    series = _load_series(results)
    out["iedf"] = pe.iedf_report(maps, anode_v)
    if out["iedf"] is not None:
        out["iedf_low_energy_fraction"] = out["iedf"].get("low_energy_fraction")
    if grid is not None:
        power, ion_energy = wall_power_and_ion_energy(maps, grid)
        out["wall_electron_power_w"] = power
        out["wall_ion_mean_energy_ev"] = ion_energy
        if all(k in maps for k in ("phi_v", "t_e_ev", "wall_electron_flux_per_m2_s", "wall_ion_flux_per_m2_s")):
            out["per_cusp"] = pe.per_cusp_report(maps, grid, space_charge_limit_yield=space_charge_limit_yield)
        centroid = ionization_centroid_from_maps(maps, grid)
        out["ionization_centroid_detail"] = centroid
        out["ionization_centroid_z_m"] = None if centroid is None else centroid.get("centroid_z_m")
    # v2.2.0 SEE readings (only when the wall emitted)
    if "see_emission_a" in currents:
        see: dict[str, Any] = {"window_emission_current_a": currents.get("see_emission_a"), "window_effective_yield": currents.get("see_effective_yield"),
                               "cumulative": {k: cumulative.get(k) for k in pe.SEE_CUMULATIVE_KEYS if k in cumulative}}
        if series is not None:
            see["trailing_20pct_means"] = {k: _trailing_mean(series[k]) for k in pe.SEE_SERIES_KEYS if k in series}
        if out.get("per_cusp") is not None:
            see["cusps_space_charge_limited"] = int(sum(1 for c in out["per_cusp"] if c.get("space_charge_limited")))
            see["cusp_effective_yields"] = [c.get("see_effective_yield") for c in out["per_cusp"]]
        out["see"] = see
    # v2.3.0 collision-set readings (only when the ion MCC ran)
    if "cex_rate_per_s" in currents:
        s_rate = float(out["ionization_rate_per_s"]) if out["ionization_rate_per_s"] else None
        collision: dict[str, Any] = {k: currents.get(k) for k in pe.XE_CURRENT_KEYS if k in currents}
        collision["cex_over_ionization"] = (float(currents["cex_rate_per_s"]) / s_rate) if s_rate else None
        collision["cumulative"] = {k: cumulative.get(k) for k in pe.XE_CUMULATIVE_KEYS if k in cumulative}
        exc = [cumulative.get(f"excitations_level_{i}") for i in (1, 2, 3, 4)]
        total_exc = sum(float(x) for x in exc if x is not None)
        collision["excitation_level_shares"] = [float(x) / total_exc if x is not None and total_exc > 0 else None for x in exc]
        window_steps = int(summary.get("averaging_window_steps") or 0) or 400_000
        rates = pe._series_window_rates(results, ("pz_fast_neutral_exit", "pz_fast_neutral_wall", "ke_fast_neutral_exit_j", "pz_exit_ions", "ion_neutral_loss_j"), window_steps)
        if rates is not None:
            collision["trailing_window_rates"] = {**rates, "source": "series.jsonl cumulative differences", "fast_neutral_exit_momentum_rate_n": rates.get("pz_fast_neutral_exit_rate"),
                                                  "exit_ion_momentum_rate_n": rates.get("pz_exit_ions_rate"), "fast_neutral_exit_power_w": rates.get("ke_fast_neutral_exit_j_rate")}
        out["collision_set"] = collision
    # v2.4.0 Coulomb readings
    coulomb = coulomb_readings(series, maps, grid, coulomb_log_floor) if grid is not None else None
    if coulomb is not None:
        coulomb["cumulative"] = {k: cumulative.get(k) for k in COULOMB_CUMULATIVE_KEYS if k in cumulative}
        out["coulomb"] = coulomb
        out["nu_e_spitzer_peak_over_nu_en"] = coulomb["trailing_20pct_means"].get("nu_e_spitzer_peak_over_nu_en")
    # v2.5.0 spatial-gas readings
    neutrals = neutral_readings(summary, series, maps, grid) if grid is not None else None
    if neutrals is not None:
        neutrals["cumulative"] = {k: cumulative.get(k) for k in NEUTRAL_CUMULATIVE_KEYS if k in cumulative}
        out["neutrals"] = neutrals
        out["neutral_density_anode_over_exit"] = neutrals.get("axis_anode_over_exit_trailing") or neutrals.get("anode_over_exit_axis_window")
        out["neutral_depletion_fraction"] = neutrals.get("depletion_fraction")
        meta = neutrals.get("metastables") or {}
        out["metastable_fraction_of_ground"] = meta.get("trailing_20pct_mean_fraction_of_ground")
        out["stepwise_fraction_of_ionization"] = meta.get("trailing_20pct_mean_stepwise_fraction_of_ionization")
    # v2.1.0 anomalous readings
    if "anomalous" in cumulative:
        time_s = float(out.get("simulated_time_s") or 0.0)
        n_last = float(series["electrons"][-1]) if series is not None and "electrons" in series and series["electrons"].size else None
        out["anomalous"] = {"cumulative_events": cumulative.get("anomalous"), "mean_rate_per_s": (float(cumulative["anomalous"]) / time_s) if time_s > 0 else None,
                            "mean_rate_per_electron_per_s": (float(cumulative["anomalous"]) / time_s / n_last) if time_s > 0 and n_last else None}
    out["sustain"] = sustain_readings(summary, series)
    return out


def reference_quantities_from_files(results: Path = REFERENCE_RESULTS, grid=None) -> dict[str, Any] | None:
    """Recompute the pinned reference numbers from the ss-v4 results directory (fail-closed consistency check)."""

    if not (results / "summary.json").is_file() or not (results / "maps.npz").is_file():
        return None
    q = run_quantities(results, grid)
    return {k: q.get(k) for k in QUANTITY_KEYS}


def shift_rows(run: dict[str, Any], reference: dict[str, Any], hypotheses: dict[str, dict[str, Any]], *, reported_only: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
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
        elif ref == 0.0:
            shift, kind, band = value, "absolute_from_zero_reference", None
        else:
            shift, kind, band = (value - ref) / abs(ref), "relative", PARTICLE_BAND.get(key)
        if key in reported_only or band is None or sign is None or sign not in ("+", "-", "0"):
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


def hypothesis_verdict(case: str, a_plateau: bool, extinguished: bool, shifts: dict[str, Any], key_quantities: tuple[str, ...], *, sustained: bool | None) -> str:
    """confirmed / not_confirmed / inconclusive by the predeclared rule (acceptance d_verdict.per_case_hypothesis_verdict)."""

    if case in FULL_PHYSICS_CASES and "sustains" in key_quantities:
        if extinguished:
            return "not_confirmed"
        if not a_plateau or sustained is not True:
            return "inconclusive"
        if any(row["status"] == "contradicting" for row in shifts.values()):
            return "not_confirmed"
        return "confirmed"
    if not a_plateau:
        return "inconclusive"
    if any(row["status"] == "contradicting" for row in shifts.values()):
        return "not_confirmed"
    if all(shifts[k]["status"] == "confirming" for k in key_quantities):
        return "confirmed"
    return "inconclusive"


def _ignition_passed(sustain: dict[str, Any]) -> bool | None:
    ignition = sustain.get("ignition")
    if ignition is None:
        return None
    checks = [c for c in ignition.get("checks", []) if c.get("evaluated")]
    if ignition.get("failed"):
        return False
    if len(checks) < len(ignition.get("checks", [])):
        return None     # not every declared check reached
    return all(c.get("passed") for c in checks)


def _companion_assessment(path: Path) -> dict[str, Any] | None:
    if not (path / "assessment.json").is_file():
        return None
    return json.loads((path / "assessment.json").read_text(encoding="utf-8"))


def assess_case(case: str, *, results: Path | None = None, protocol: dict[str, Any] | None = None, output: Path | None = None, reference_check: bool = True,
                alpha0_results: Path | None = None, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """Per-case plateau status (plateau_clean / plateau_heating / no_plateau / extinguished), hypothesis verdict, the shift table against ss-v4 (and, for the alpha cases, against
    full-physics-alpha0 when its record exists), the per-cusp / effect readings and the sustain reading."""

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
    floor = config.coulomb.coulomb_log_floor if config.coulomb is not None else 2.0
    run = run_quantities(results, grid, anode_v=anode_v, space_charge_limit_yield=limit, coulomb_log_floor=floor)
    a_plateau = run["stop_reason"] == "plateau_reached_after_min_transit_times"
    windowed = run["windowed_residual_over_electrode_work"]
    b_ok = windowed is not None and bool(run["windowed_residual_window_complete"]) and windowed < 0.02
    late = run["sustain"].get("late_extinction_diagnostic") or {}
    extinguished = run["stop_class"] == "no_ignition" or (not a_plateau and bool(late.get("late_extinction")))
    plateau_status = ("plateau_clean" if (a_plateau and b_ok) else "plateau_heating" if a_plateau else "extinguished" if extinguished else "no_plateau")
    reported_only = tuple(acceptance["c_shifts"].get("reported_only") or ())
    shifts = shift_rows(run, reference, hypotheses, reported_only=reported_only)
    ignition_passed = _ignition_passed(run["sustain"])
    sustained: bool | None = None
    if case in FULL_PHYSICS_CASES:
        sustained = False if extinguished else (True if (a_plateau and ignition_passed is not False) else None)
    # the alpha cases: sign rows against full-physics-alpha0 when its record exists (the predeclared judged_against)
    shifts_vs_alpha0 = None
    alpha0_source = None
    if case in FULL_PHYSICS_CASES and case != "full-physics-alpha0":
        alpha0_dir = alpha0_results if alpha0_results is not None else case_results("full-physics-alpha0", results.parent)
        alpha0 = _companion_assessment(alpha0_dir)
        if alpha0 is not None and alpha0.get("a_plateau", {}).get("passed"):
            alpha0_source = str(alpha0_dir)
            shifts_vs_alpha0 = shift_rows(run, {k: alpha0["run"].get(k) for k in QUANTITY_KEYS}, hypotheses, reported_only=reported_only)
    judged = shifts_vs_alpha0 if shifts_vs_alpha0 is not None else ({k: dict(v, status="reported") for k, v in shifts.items()} if case in FULL_PHYSICS_CASES and case != "full-physics-alpha0" else shifts)
    verdict = hypothesis_verdict(case, a_plateau, extinguished, judged, KEY_QUANTITIES[case], sustained=sustained)
    consistency = None
    reference_cusps = None
    if reference_check:
        ref_grid = runner.build_config(protocol_module.load_v4_protocol(), backend="cpu").grid      # the reference's own grid (== the case grid in production)
        consistency = _consistency(reference, REFERENCE_RESULTS, ref_grid)
        if consistency is not None and not all(entry["agree"] for entry in consistency.values()):
            raise PIC2DValidationError("reference_run.quantities disagree with the ss-v4 artifacts on disk: " + json.dumps({k: v for k, v in consistency.items() if not v["agree"]}))
        if (REFERENCE_RESULTS / "maps.npz").is_file():
            reference_cusps = run_quantities(REFERENCE_RESULTS, ref_grid, anode_v=anode_v).get("per_cusp")
    cusp_rows = None
    if run.get("per_cusp") is not None and reference_cusps is not None:
        cusp_rows = []
        cusp_gas = {c["z_c_m"]: c for c in ((run.get("neutrals") or {}).get("cusp_plane_density") or [])}
        coulomb_cols = ((run.get("coulomb") or {}).get("maps") or {})
        for mine, ref in zip(run["per_cusp"], reference_cusps, strict=True):
            row = {"z_c_m": mine["z_c_m"],
                   "electron_wall_current_a": {"value": mine["electron_wall_current_a"], "reference": ref["electron_wall_current_a"],
                                               "relative_shift": (mine["electron_wall_current_a"] - ref["electron_wall_current_a"]) / abs(ref["electron_wall_current_a"]) if ref["electron_wall_current_a"] else None,
                                               "hypothesis_sign": hypotheses.get("cusp_electron_wall_current_a", {}).get("sign")},
                   "ion_wall_current_a": {"value": mine["ion_wall_current_a"], "reference": ref["ion_wall_current_a"]},
                   "sheath_drop_v": {"value": mine["sheath_drop_v"], "reference": ref["sheath_drop_v"], "difference_v": mine["sheath_drop_v"] - ref["sheath_drop_v"],
                                     "relative_shift": (mine["sheath_drop_v"] - ref["sheath_drop_v"]) / abs(ref["sheath_drop_v"]) if ref["sheath_drop_v"] else None,
                                     "hypothesis_sign": hypotheses.get("cusp_sheath_drop_v", {}).get("sign")},
                   "near_wall_drop_v": {"value": mine["near_wall_drop_v"], "reference": ref["near_wall_drop_v"]},
                   "near_wall_t_e_ev": {"value": mine["near_wall_t_e_ev"], "reference": ref["near_wall_t_e_ev"]},
                   "wall_ion_mean_energy_ev": {"value": mine.get("wall_ion_mean_energy_ev"), "reference": ref.get("wall_ion_mean_energy_ev")}}
            if "see_effective_yield" in mine:
                row["see"] = {k2: mine.get(k2) for k2 in ("see_effective_yield", "see_current_a", "see_mean_emitted_energy_ev", "space_charge_limited", "space_charge_limit_rule")}
            if mine["z_c_m"] in cusp_gas:
                row["neutral_gas"] = cusp_gas[mine["z_c_m"]]
            if coulomb_cols:
                # column_frequency_profile keys its planes "6.028mm" / "12.000mm" / "17.972mm"
                plane_key = f"{mine['z_c_m'] * 1e3:.3f}mm"
                row["coulomb"] = {"nu_ee_pair_mean_per_s": (coulomb_cols.get("cusp_columns_nu_ee_pair_mean_per_s") or {}).get(plane_key),
                                  "nu_e_spitzer_per_s": (coulomb_cols.get("cusp_columns_nu_e_spitzer_per_s") or {}).get(plane_key)}
            cusp_rows.append(row)
    record = {
        "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "case": case, "effects": protocol["campaign"]["effects"],
        "group": protocol["campaign"]["group"], "alpha": protocol["campaign"]["alpha"], "time_acceleration": protocol["campaign"]["time_acceleration"],
        "results_dir": results.relative_to(HERE).as_posix() if results.is_relative_to(HERE) else str(results), "git_head_now": runner.git_head(), "run": run,
        "reference": reference, "reference_case": REFERENCE_CASE, "reference_corrected_ledger": protocol["reference_run"]["corrected_ledger"], "reference_consistency": consistency,
        "a_plateau": {"passed": a_plateau, "stop_reason": run["stop_reason"], "stop_class": run["stop_class"], "ion_transit_times": run["ion_transit_times"], "plateau": run["plateau"],
                      "drift_members_arming": run["drift_members_arming"], "rule": acceptance["a_plateau"]},
        "b_residual_power": {"passed": b_ok, "windowed_residual_over_electrode_work": windowed, "window_complete": run["windowed_residual_window_complete"], "bound": 0.02,
                             "one_sided": True, "ledger": "v2.0.6 W-corrected (native)", "cumulative_witness": run["cumulative_residual_over_electrode_work"],
                             "reference_reads": protocol["reference_run"]["corrected_ledger"]["windowed_residual_over_electrode_work_corrected"], "rule": acceptance["b_residual_power"]},
        "c_shifts_vs_reference": shifts, "c_shifts_vs_full_physics_alpha0": shifts_vs_alpha0, "alpha0_record": alpha0_source, "hypotheses": hypotheses,
        "key_quantities": list(KEY_QUANTITIES[case]), "reported_only": list(reported_only), "per_cusp_vs_reference": cusp_rows,
        "plateau_status": plateau_status, "plateau_status_rule": acceptance["d_verdict"]["plateau_status"][plateau_status], "stop_class": run["stop_class"],
        "extinguished": extinguished, "sustain": {"applies": case in FULL_PHYSICS_CASES, "ignition_gate_passed": ignition_passed, "sustained": sustained,
                                                  "reading": "sustains" if sustained else "extinguished" if extinguished else "undecided", "diagnostic": run["sustain"]},
        "hypothesis_verdict": verdict, "hypothesis_verdict_rule": acceptance["d_verdict"]["per_case_hypothesis_verdict"][verdict],
        "peak_debye_window": {"cells_per_debye_last": run["cells_per_debye_window_last"], "trailing_mean": run["cells_per_debye_window_trailing_mean"], "soft_ok": run["peak_debye_soft_ok"]},
        "claim_boundary": protocol["claim_boundary"],
    }
    artifacts.write_canonical_json(output or (results / "assessment.json"), record)
    log(f"[assess] {case}: {plateau_status} ({run['stop_class']}; a {a_plateau}, b {b_ok} [{windowed}]) -> {verdict}; sustain {record['sustain']['reading']}; shifts vs {REFERENCE_CASE}: "
        + ", ".join(f"{k} {pe._fmt_shift(v)} {v['status']}" for k, v in shifts.items() if v["status"] != "unavailable"))
    return record


# -- campaign ------------------------------------------------------------------------------------------------------------------

def _load_assessment(results: Path) -> dict[str, Any] | None:
    path = results / "assessment.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def f_qualification(assessments: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """F = 10 qualified / disqualified / not_evaluable by the plateau scalars of the F pair (acceptance f_qualification)."""

    a, b = assessments.get(F_PAIR[0]), assessments.get(F_PAIR[1])
    if a is None or b is None or not (a["a_plateau"]["passed"] and b["a_plateau"]["passed"]):
        reached = [c for c in F_PAIR if assessments.get(c) is not None and assessments[c]["a_plateau"]["passed"]]
        return {"statement": "not_evaluable", "reason": f"both members must reach (a); reached: {reached}", "rows": None}
    rows: dict[str, Any] = {}
    for key in PLATEAU_SCALARS:
        v1, v10 = a["run"].get(key), b["run"].get(key)
        band = PARTICLE_BAND.get(key)
        if v1 is None or v10 is None or float(v1) == 0.0:
            rows[key] = {"f1": v1, "f10": v10, "relative_difference": None, "band": band, "inside_band": None}
            continue
        diff = (float(v10) - float(v1)) / abs(float(v1))
        rows[key] = {"f1": float(v1), "f10": float(v10), "relative_difference": diff, "band": band, "inside_band": bool(abs(diff) <= band)}
    expected_to_differ = {key: {"f1": a["run"].get(key), "f10": b["run"].get(key)} for key in ("metastable_fraction_of_ground", "stepwise_fraction_of_ionization", "neutral_depletion_fraction")}
    judged = [r for r in rows.values() if r["inside_band"] is not None]
    statement = "F_qualified" if judged and all(r["inside_band"] for r in judged) else "F_disqualified"
    return {"statement": statement, "rows": rows, "outside_band": [k for k, r in rows.items() if r["inside_band"] is False], "expected_to_differ_reported_not_judged": expected_to_differ,
            "consequence": ("F = 10 may be used in later runs (a neutral steady state needs F ~ 100-300; a further qualification at that F is owed)" if statement == "F_qualified"
                            else "F is DISQUALIFIED: the acceleration moves the plasma plateau; only F = 1 runs may be quoted")}


def sustain_table(assessments: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """The full-physics alphas beside the dilute-gas alpha-series outcomes and the ext-val bohm-0.4 record (operating-point comparison)."""

    table: dict[str, Any] = {}
    for case in FULL_PHYSICS_CASES:
        a = assessments.get(case)
        alpha = CASES[case]["alpha"]
        if a is None:
            table[case] = {"alpha": alpha, "gas": "Knudsen profile (channel mean ~2.5e20)", "reading": "pending"}
            continue
        table[case] = {"alpha": alpha, "gas": "Knudsen profile (channel mean ~2.5e20)", "reading": a["sustain"]["reading"], "plateau_status": a["plateau_status"], "stop_class": a["stop_class"],
                       "ignition_gate_passed": a["sustain"]["ignition_gate_passed"], "ion_transit_times": a["a_plateau"]["ion_transit_times"],
                       "discharge_current_a": a["run"].get("discharge_current_a"), "ionization_rate_per_s": a["run"].get("ionization_rate_per_s")}
    dilute: dict[str, Any] = {}
    for at_case, alpha in (("alpha-1over64", 1 / 64), ("alpha-1over16", 1 / 16), ("alpha-0.345", 0.345)):
        rec = _companion_assessment(AT_RESULTS / at_case)
        if rec is None:
            dilute[at_case] = {"alpha": alpha, "gas": "0-D inventory (n_g 5.5e19 -> ~3.2e19)", "reading": "pending (record not in this tree)"}
            continue
        stop = rec.get("a_plateau", {}).get("stop_reason")
        run = rec.get("run") or {}
        dilute[at_case] = {"alpha": alpha, "gas": "0-D inventory (n_g 5.5e19 -> ~3.2e19)", "plateau_status": rec.get("plateau_status"), "stop_reason": stop,
                           "reading": ("sustains" if rec.get("a_plateau", {}).get("passed") else "extinguished" if (stop == "no_ignition" or at_case == "alpha-1over16") else "undecided"),
                           "discharge_current_a": run.get("discharge_current_a"), "note": ("launch 1 record 0916a4f8: EXTINCTION under the closure (N_e e-fold 0.88 us, I_d 3.1 -> 0.06 mA)"
                                                                                           if at_case == "alpha-1over16" else None)}
    statements = {}
    for case in FULL_PHYSICS_CASES:
        entry = table[case]
        reading = entry.get("reading")
        statements[case] = (f"the Knudsen gas sustains the Bohm-leaky full-physics discharge at alpha = {entry['alpha']:.4g}: "
                            + ("YES" if reading == "sustains" else "NO (extinguished)" if reading == "extinguished" else "UNDECIDED" if reading == "undecided" else "pending"))
    return {"table": table, "dilute_gas_alpha_series": dilute, "statements": statements,
            "ext_val_bohm_0_4": "pic2d_external_validation_v0 launch 2 (alpha 0.345 at the static 2e20 Brandt gas): read its record for the third operating point",
            "rule": "sustains = ignition gate passed at 1.0 and 2.0 us AND (a) reached; extinguished = plateau_status extinguished; undecided otherwise"}


def alpha_trend(assessments: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Monotonicity of the full-physics points in alpha (acceptance e_sustain.alpha_trend)."""

    points = [(CASES[c]["alpha"], c, assessments.get(c)) for c in FULL_PHYSICS_CASES]
    reached = [(alpha, c, a) for alpha, c, a in points if a is not None and a["a_plateau"]["passed"]]
    reached.sort(key=lambda t: t[0])
    rows: dict[str, Any] = {}
    signs = {k: v["sign"] for k, v in HYPOTHESES_BY_CASE["full-physics-alpha1over16"].items() if k in MONOTONE_QUANTITIES}
    for key in MONOTONE_QUANTITIES:
        values = [(alpha, a["run"].get(key)) for alpha, _, a in reached]
        rows[key] = {"sign": signs.get(key), "values_by_alpha": values, "monotone": None}
        if len(values) >= 2 and all(v is not None for _, v in values):
            seq = [float(v) for _, v in values]
            band = PARTICLE_BAND.get(key, 0.0)
            ok = True
            for lo, hi in zip(seq[:-1], seq[1:], strict=True):
                step = (hi - lo) / abs(lo) if lo else 0.0
                if signs.get(key) == "+" and step < -band:
                    ok = False
                if signs.get(key) == "-" and step > band:
                    ok = False
            rows[key]["monotone"] = ok
    contradicting = []
    for _, c, a in reached:
        rows_vs = a.get("c_shifts_vs_full_physics_alpha0") or {}
        contradicting += [f"{c}:{k}" for k in MONOTONE_QUANTITIES if (rows_vs.get(k) or {}).get("status") == "contradicting"]
    if len(reached) < 3:
        statement = "inconclusive"
    elif rows["discharge_current_a"]["monotone"] and rows["peak_n_e_window_per_m3"]["monotone"] and not contradicting:
        statement = "trend_confirmed"
    else:
        statement = "trend_not_confirmed"
    return {"statement": statement, "points_reached": [c for _, c, _ in reached], "rows": rows, "contradicting": contradicting,
            "rule": "trend_confirmed needs all three full-physics points at (a) AND I_d and peak n_e monotone in the declared direction AND no monotone quantity contradicting vs alpha = 0"}


def additivity(assessments: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """full-physics-alpha0 against see-bn+xe-set-v2 (physics-effects record) + coulomb + [neutrals-spatial - xe-set-v2] (acceptance g_additivity)."""

    full = assessments.get("full-physics-alpha0")
    coul = assessments.get("coulomb")
    r5 = assessments.get("neutrals-spatial")
    pe_both = _companion_assessment(PE_RESULTS / "see-bn+xe-set-v2")
    pe_xe = _companion_assessment(PE_RESULTS / "xe-set-v2")
    parts_ok = {"full-physics-alpha0": full is not None and full["a_plateau"]["passed"], "coulomb": coul is not None and coul["a_plateau"]["passed"],
                "neutrals-spatial": r5 is not None and r5["a_plateau"]["passed"],
                "pe:see-bn+xe-set-v2": pe_both is not None and pe_both.get("a_plateau", {}).get("passed", False),
                "pe:xe-set-v2": pe_xe is not None and pe_xe.get("a_plateau", {}).get("passed", False)}
    out: dict[str, Any] = {"parts_reached": parts_ok}
    # R5 as the operating-point change: reported whenever the R5 case and the coulomb case exist
    dominance = None
    if r5 is not None and coul is not None:
        dominance = {}
        for key in PLATEAU_SCALARS:
            s_r5 = (r5["c_shifts_vs_reference"].get(key) or {}).get("shift")
            s_c = (coul["c_shifts_vs_reference"].get(key) or {}).get("shift")
            s_pe = ((pe_both or {}).get("c_shifts_vs_reference", {}).get(key) or {}).get("shift") if pe_both else None
            others = None if s_c is None or s_pe is None else s_c + s_pe
            dominance[key] = {"shift_r5_incl_set_v2": s_r5, "shift_coulomb": s_c, "shift_see_bn_xe_set_v2": s_pe,
                              "operating_point_dominates": None if s_r5 is None or others is None else bool(abs(s_r5) > abs(others))}
    out["r5_operating_point_dominance"] = dominance
    if not all(parts_ok.values()):
        out.update({"statement": "not_evaluable", "reason": "every part must reach (a) and both physics-effects records must exist: " + json.dumps(parts_ok), "rows": None})
        return out
    rows: dict[str, Any] = {}
    for key in QUANTITY_KEYS:
        band = ABSOLUTE_BAND.get(key, PARTICLE_BAND.get(key))
        f = full["c_shifts_vs_reference"][key]["shift"]
        parts = [pe_both["c_shifts_vs_reference"].get(key, {}).get("shift"), coul["c_shifts_vs_reference"][key]["shift"], r5["c_shifts_vs_reference"][key]["shift"],
                 pe_xe["c_shifts_vs_reference"].get(key, {}).get("shift")]
        if band is None or f is None or any(p is None for p in parts) or not all(np.isfinite([f, *parts])) or key in REPORTED_ONLY_SPATIAL:
            rows[key] = {"combined": f, "sum_of_parts": None, "interaction": None, "band": band, "classification": "reported"}
            continue
        total = float(parts[0] + parts[1] + (parts[2] - parts[3]))
        interaction = float(f) - total
        cls = "additive" if abs(interaction) <= band else ("super_additive" if interaction * total > 0 else "sub_additive")
        rows[key] = {"combined": f, "parts": {"see-bn+xe-set-v2": parts[0], "coulomb": parts[1], "R5 = neutrals-spatial - xe-set-v2": parts[2] - parts[3]}, "sum_of_parts": total,
                     "interaction": interaction, "band": band, "classification": cls}
    judged = [r["classification"] for r in rows.values() if r["classification"] != "reported"]
    out.update({"statement": "additive" if all(c == "additive" for c in judged) else "interacting", "rows": rows,
                "non_additive_quantities": [k for k, r in rows.items() if r["classification"] in ("super_additive", "sub_additive")]})
    return out


def assess_campaign(*, results_root: Path = RESULTS, cases_override: dict[str, Path] | None = None, output: Path | None = None, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """The six per-case verdicts, the sustain table, the alpha trend, the additivity statement and the F qualification; unreached / unassessed cases are listed."""

    campaign = load_campaign()
    reference = campaign["reference_run"]["quantities"]
    points: dict[str, dict[str, Any]] = {REFERENCE_CASE: {"reached": True, "verdict": "reference (recorded ss-v4 plateau; (b) FAIL at +2.46 % corrected)",
                                                          "quantities": {k: reference.get(k) for k in QUANTITY_KEYS}}}
    assessments: dict[str, dict[str, Any] | None] = {}
    for case in CASES:
        results = (cases_override or {}).get(case, case_results(case, results_root))
        assessment = _load_assessment(results)
        entry: dict[str, Any] = {"effects": list(CASES[case]["effects"]), "group": CASES[case]["group"], "alpha": CASES[case]["alpha"], "reached": False, "plateau_status": None,
                                 "verdict": None, "quantities": None, "results_dir": str(results)}
        assessments[case] = assessment
        if assessment is not None:
            entry.update({"plateau_status": assessment["plateau_status"], "stop_class": assessment["stop_class"], "verdict": assessment["hypothesis_verdict"],
                          "reached": assessment["a_plateau"]["passed"], "b_passed": assessment["b_residual_power"]["passed"],
                          "quantities": {k: assessment["run"].get(k) for k in QUANTITY_KEYS}, "shifts": assessment["c_shifts_vs_reference"],
                          "shifts_vs_alpha0": assessment.get("c_shifts_vs_full_physics_alpha0"), "per_cusp": assessment.get("per_cusp_vs_reference"), "sustain": assessment.get("sustain"),
                          "see": assessment["run"].get("see"), "collision_set": assessment["run"].get("collision_set"), "coulomb": assessment["run"].get("coulomb"),
                          "neutrals": {k: v for k, v in (assessment["run"].get("neutrals") or {}).items() if not k.endswith("profile") and not k.endswith("profile_per_m3") and k != "profile_z_m"} or None})
        elif (results / "summary.json").is_file():
            entry["verdict"] = "not assessed (summary present; run `assess --case`)"
        points[case] = entry
    reached = [c for c in CASES if points[c]["reached"]]
    record = {
        "schema_version": CAMPAIGN_ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": EXPERIMENT_ID, "git_head_now": runner.git_head(), "points": points,
        "cases_reached": reached, "cases_unreached": [c for c in CASES if c not in reached],
        "verdicts": {c: points[c]["verdict"] for c in CASES}, "plateau_status": {c: points[c]["plateau_status"] for c in CASES},
        "sustain": sustain_table(assessments), "alpha_trend": alpha_trend(assessments), "additivity": additivity(assessments), "f_qualification": f_qualification(assessments),
        "rules": {k: campaign["acceptance"][LAUNCH_PRIORITY[0]][k] for k in ("e_sustain", "f_qualification", "g_additivity")},
        "launch_priority": list(LAUNCH_PRIORITY),
        "note": "evaluates the assessed cases only; provisional until every case has a terminal state and final once all six (and the physics-effects parts) are in",
    }
    output = output or (results_root / "campaign-assessment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.write_canonical_json(output, record)
    log(f"[assess --campaign] verdicts {record['verdicts']}; reached {reached}; sustain {record['sustain']['statements']}; alpha trend {record['alpha_trend']['statement']}; "
        f"additivity {record['additivity']['statement']}; F {record['f_qualification']['statement']}")
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
    pre.add_argument("--dense-seed-density", type=float, default=None, help="second synthetic load (default: 3e17 for the spatial cases, none otherwise)")
    pre.add_argument("--gpu-timing", action="store_true", help="time the step on the launch GPU (default: inputs + mesh only)")
    shake = sub.add_parser("shakedown")
    shake.add_argument("--case", required=True, choices=sorted(CASES))
    shake.add_argument("--backend", default="warp-cuda")
    shake.add_argument("--max-steps", type=int, default=SHAKEDOWN_OVERRIDES["max_steps"])
    shake.add_argument("--reuse-run", action="store_true", help="skip the stepping: assess / record an existing completed run of the identical shakedown protocol")
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
        preflight(args.case, backend=args.backend, timing_steps=args.timing_steps, loaded_seed_density=args.loaded_seed_density, dense_seed_density=args.dense_seed_density,
                  gpu_timing=args.gpu_timing)
    elif args.command == "shakedown":
        shakedown(args.case, backend=args.backend, max_steps=args.max_steps, reuse_run=args.reuse_run)
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
