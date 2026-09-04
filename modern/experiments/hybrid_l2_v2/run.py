"""Hybrid L2 v2 on the reference material-aware field, judged against the accepted PIC v2 base plateau.

Stages (from ``modern/``, ``$env:PYTHONPATH="$PWD\\src;$PWD"``; CPU only)::

    python -m experiments.hybrid_l2_v2.run preflight                 # real field on every case grid, partition check, closures, ms/step
    python -m experiments.hybrid_l2_v2.run shakedown                 # synthetic field + real field short runs through finalize + assess
    python -m experiments.hybrid_l2_v2.run launch --case base [--expect-commit SHA]
    python -m experiments.hybrid_l2_v2.run launch --case <name>      # spatial / temporal / weight / seed / closure levels
    python -m experiments.hybrid_l2_v2.run status
    python -m experiments.hybrid_l2_v2.run assess                    # GATE-L2 metrics over every finished case -> results/assessment.json

The model is ``cft_revival.hybrid.l2`` (see ``modern/docs/hybrid-l2-v2.md``).  Preregistration discipline as the PIC's
steady-state v4: preflight and shakedown records committed before the launch, a launch that refuses a dirty worktree /
an unexpected commit / an existing lock, and an assessment against the PIC reference table frozen in ``protocol.json``
(re-derived from the PIC artifacts at assessment time and required to agree).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.hybrid import gates
from cft_revival.hybrid.cells import CellPartition, load_reference_partition, synthetic_partition
from cft_revival.hybrid.checkpoint_v2 import (
    hybrid_code_identity,
    load_checkpoint_v2,
    save_checkpoint_v2,
)
from cft_revival.hybrid.l2 import (
    MODEL_VERSION,
    HybridL2Config,
    HybridL2Simulation,
    PlateauRule,
)
from cft_revival.hybrid.models import HybridError, HybridValidationError
from cft_revival.hybrid.pb_solver import PBConfig
from cft_revival.orbit_mc.artifacts import canonical_bytes, content_hash
from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import (
    MagneticFieldMap,
    build_p2_psi_field,
    linear_psi_field_map,
    sample_field_map,
)
from cft_revival.pic2d.mcc import XenonCrossSections
from cft_revival.pic2d.models import BoundaryPotentials, ChannelGeometry, Grid2D
from cft_revival.pic2d.neutrals import NeutralInventoryConfig
from experiments.hybrid_l2_v2 import closure
from experiments.pic2d_design_mini_sweep_v1.closure import extract_targets

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
PROTOCOL_PATH = HERE / "protocol.json"
RESULTS = HERE / "results"
PREFLIGHT_PATH = HERE / "preflight.json"
SHAKEDOWN_PATH = HERE / "shakedown.json"
CATALOGUE_RESULTS = MODERN / "experiments" / "cusp_topology_search_v3_1" / "results"
PIC_V2 = MODERN / "experiments" / "pic2d_cft_steady_state_v2"
PIC_V4 = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results"
LOCK_NAME = "execution-lock.json"
ASSESSMENT_SCHEMA = "cft-revival.hybrid-l2-v2.assessment/1.0.0"
SUMMARY_SCHEMA = "cft-revival.hybrid-l2-v2.summary/1.0.0"

Log = Callable[[str], None]


def _print(text: str) -> None:
    print(text, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def git_head() -> str | None:
    try:
        return git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# -- inputs ------------------------------------------------------------------------------------------------------------------

def case_grid(protocol: dict[str, Any], case: dict[str, Any]) -> Grid2D:
    g = protocol["geometry"]
    return Grid2D(ChannelGeometry(g["bore_radius_m"], g["z_min_m"], g["z_max_m"], g["cone_start_z_m"], g["exit_radius_m"]),
                  int(case["radial_cells"]), int(case["axial_cells"]))


def resolve_case(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    base = dict(protocol["cases"]["base"])
    if name == "base":
        base["name"] = "base"
        return base
    if name not in protocol["cases"]:
        raise HybridValidationError(f"unknown case {name!r}; known: {sorted(protocol['cases'])}")
    merged = {**base, **{k: v for k, v in protocol["cases"][name].items() if k != "note"}}
    merged["name"] = name
    merged["note"] = protocol["cases"][name].get("note")
    return merged


def results_dir_for(name: str) -> Path:
    return RESULTS if name == "base" else HERE / f"results-{name}"


def real_partition(grid: Grid2D, protocol: dict[str, Any]) -> CellPartition:
    cells = protocol["cells"]
    return load_reference_partition(CATALOGUE_RESULTS, set_id=cells["set_id"], design_id=cells["design_id"], grid=grid,
                                    declared_cusp_planes_m=cells["declared_pic_cusp_planes_m"], plane_tolerance_m=cells["plane_tolerance_m"])


def real_field(grid: Grid2D) -> MagneticFieldMap:
    psi, evidence = build_p2_psi_field(REPOSITORY_ROOT, role="primary")
    return sample_field_map(psi, grid, evidence)


def synthetic_inputs(grid: Grid2D) -> tuple[MagneticFieldMap, CellPartition, XenonCrossSections]:
    """A divergence-free analytic field with three artificial cusp planes and the synthetic cross sections (SHAKEDOWN only)."""

    field = linear_psi_field_map(grid, 30.0)
    partition = synthetic_partition(grid.geometry.z_min_m, grid.geometry.domain_z_max_m, [0.006, 0.012, 0.018])
    return field, partition, XenonCrossSections.synthetic_for_tests()


def build_config(protocol: dict[str, Any], case: dict[str, Any]) -> HybridL2Config:
    op = protocol["operating_point"]
    num = protocol["numerics"]
    clo = protocol["closures"]
    scale_g = float(case.get("conductance_scale", 1.0))
    scale_w = float(case.get("leak_width_scale", 1.0))
    inventory = NeutralInventoryConfig(float(op["neutral_inventory"]["feed_atoms_per_s"]), float(op["neutral_inventory"]["relaxation_time_s"]))
    plateau = protocol["stopping_rule"]["plateau"]
    return HybridL2Config(
        grid=case_grid(protocol, case), potentials=BoundaryPotentials(op["anode_potential_v"], op["exit_plane_potential_v"]),
        dt_s=float(case["dt_s"]), macro_weight=float(case["macro_weight"]), seed=int(case["seed"]),
        injection_current_a=float(op["electron_injection_current_a"]), injection_temperature_ev=float(op["electron_injection_temperature_ev"]),
        seed_density_per_m3=float(op["seed_plasma_density_per_m3"]), seed_electron_temperature_ev=float(op["seed_electron_temperature_ev"]),
        neutral_ceiling_per_m3=float(op["neutral_density_per_m3"]), neutral_temperature_k=float(op["neutral_temperature_k"]),
        neutral_inventory=inventory,
        cusp_conductance_s=tuple(scale_g * float(g) for g in clo["cusp_conductance_s"]),
        leak_half_width_m=tuple(scale_w * float(w) for w in clo["leak_half_width_m"]),
        access_floor=float(clo["access_floor"]), pressure_term=bool(clo["pressure_term"]),
        pb=PBConfig(**{k: v for k, v in num["poisson_boltzmann"].items() if not k.endswith("_note")}),
        series_interval_steps=int(case["series_interval_steps"]), averaging_window_steps=int(case["averaging_window_steps"]),
        checkpoint_every_steps=int(case["checkpoint_every_steps"]), residual_window_steps=int(case["residual_window_steps"]),
        plateau=PlateauRule(float(plateau["min_transit_times"]), float(plateau["ion_transit_time_s"]), float(plateau["threshold"]),
                            float(plateau["window_fraction"])),
        max_steps=int(case["max_steps"]),
    )


# -- running ------------------------------------------------------------------------------------------------------------------

def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # a torn final line from an abrupt stop; every earlier line must parse
            raise
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
    os.replace(tmp, path)


def neutral_ledger_closure(sim: HybridL2Simulation) -> dict[str, float]:
    ledger = sim.state.neutral.ledger
    v = sim.neutrals.volume_m3
    n0 = sim.neutrals.initial_density
    n1 = sim.state.neutral.density_per_m3
    closure_atoms = ledger["fed"] + ledger.get("recycled", 0.0) - ledger["ionized"] - ledger["effused"] - ledger["artificial"] - v * (n1 - n0)
    return {"closure_atoms": closure_atoms, "closure_relative_to_inventory": closure_atoms / (v * n0), "ledger": dict(ledger)}


def charge_identity_relative(series: dict[str, np.ndarray]) -> float:
    """Largest |plasma + wall + induced| over the recorded solves relative to the represented electron charge."""

    identity = np.abs(series["total_charge_identity_c"])
    scale = 1.602176634e-19 * np.maximum(series["electrons"], 1.0)
    return float(np.max(identity / scale))


def window_means(series: dict[str, np.ndarray], start_step: int, end_step: int) -> dict[str, float]:
    mask = (series["step"] > start_step) & (series["step"] <= end_step)
    if not mask.any():
        mask = np.ones_like(series["step"], dtype=bool)
    keys = ("current_anode_electron_a", "current_anode_ion_a", "current_discharge_a", "current_exit_electron_a", "current_exit_ion_beam_a",
            "current_injected_electron_a", "current_wall_electron_a", "current_wall_ion_a", "current_ionization_rate_per_s",
            "neutral_density_per_m3", "electrons", "ions")
    return {key.replace("current_", ""): float(np.mean(series[key][mask])) for key in keys}


def finalize(sim: HybridL2Simulation, results: Path, *, protocol: dict[str, Any], case: dict[str, Any], stop_reason: str,
             wall_seconds: float, field_kind: str, log: Log = _print, sessions: list[dict[str, Any]] | None = None) -> Path:
    """maps.npz / series.npz / summary.json / l2-targets.json / checkpoint-final from the live simulation."""

    series = sim.series_arrays()
    if not series:
        raise HybridValidationError("nothing to finalize: no series record")
    maps_and_kind = sim.maps()
    if maps_and_kind is None:
        raise HybridValidationError("nothing to finalize: no completed averaging window")
    maps, maps_kind = maps_and_kind
    window = sim.completed_window if maps_kind == "window_average" else sim.window
    start_step, end_step = window.start_step, window.start_step + window.steps
    maps_sha = artifacts.write_npz(results / "maps.npz", maps)
    series_sha = artifacts.write_npz(results / "series.npz", series)
    save_checkpoint_v2(results, "checkpoint-final", sim)
    means = window_means(series, start_step, end_step)
    window_currents = {k: means[k] for k in ("anode_electron_a", "anode_ion_a", "discharge_a", "exit_electron_a", "exit_ion_beam_a",
                                                 "injected_electron_a", "wall_electron_a", "wall_ion_a", "ionization_rate_per_s")}
    targets = extract_targets(maps, closure._Mapping(sim.config.grid), list(sim.partition.cusp_z_m), closure.partition_cells(sim.partition),
                              window_currents=window_currents)
    artifacts.write_canonical_json(results / "l2-targets.json", _nan_free(targets))
    plateau = sim.plateau()
    residual = sim.windowed_residual_over_electrode_work()
    n_e = maps["n_e_per_m3"]
    peak = int(np.nanargmax(n_e))
    pi_, pj = np.unravel_index(peak, n_e.shape)
    steps_done = int(sim.state.step)
    summary = {
        "schema_version": SUMMARY_SCHEMA, "experiment_id": protocol["experiment_id"], "model_version": MODEL_VERSION, "case": case,
        "field_kind": field_kind, "status": protocol["status"], "claim_boundary": protocol["claim_boundary"],
        "git_head": git_head(), "protocol_sha256": file_sha256(PROTOCOL_PATH), "config_sha256": content_hash(sim.config.to_dict()),
        "stop_reason": stop_reason, "steps_completed": steps_done, "simulated_time_s": float(sim.state.time_s),
        "ion_transit_times": float(sim.state.time_s) / sim.config.plateau.ion_transit_time_s,
        "wall_seconds_total": wall_seconds, "ms_per_step": 1e3 * wall_seconds / max(steps_done, 1), "sessions": sessions or [],
        "final_counts": {"ions": sim.state.ions.count, "electrons": float(sim.state.electron_count.sum())},
        "plateau": plateau, "windowed_energy_residual": residual, "maps_kind": maps_kind,
        "averaging_window_step_range": [int(start_step), int(end_step)], "averaging_window_steps": int(window.steps),
        "window_currents_a": window_currents,
        "window_means": means,
        "window_maps_summary": {"peak_n_e_per_m3": float(n_e[pi_, pj]), "peak_node": [int(pi_), int(pj)], "t_e_peak_ev": float(maps["t_e_ev"][pi_, pj]),
                                "phi_max_v": float(np.max(maps["phi_v"])), "phi_min_v": float(np.min(maps["phi_v"][sim.masks.plasma_node]))},
        "neutral_inventory": {**sim.neutrals.to_dict(), "final_density_per_m3": sim.state.neutral.density_per_m3,
                              "gross_utilisation_window": means["ionization_rate_per_s"] / sim.neutrals.config.feed_atoms_per_s,
                              "ledger_closure": neutral_ledger_closure(sim)},
        "cumulative": dict(sim.state.cumulative),
        "energy_ledger": {"cumulative_residual_over_electrode_work": (sim.state.cumulative["energy_residual_j"] / sim.state.cumulative["electrode_work_j"])
                          if sim.state.cumulative["electrode_work_j"] != 0.0 else None, **residual},
        "charge_identity_max_relative": charge_identity_relative(series),
        "cells": {f"cell{k}": {"potential_v": float(series[f"cell{k}_potential_v"][-1]), "temperature_ev": float(series[f"cell{k}_temperature_ev"][-1]),
                              "electron_count": float(series[f"cell{k}_electron_count"][-1])} for k in range(sim.partition.cell_count)},
        "provenance": sim.to_provenance(), "runtime": artifacts.runtime_identity() | {"hybrid_code_sha256": hybrid_code_identity()},
        "artifacts": {"maps_npz_sha256": maps_sha, "series_npz_sha256": series_sha, "checkpoint": "checkpoint-final.json", "series_jsonl": "series.jsonl",
                      "status_jsonl": "status.jsonl", "l2_targets": "l2-targets.json"},
    }
    path = results / "summary.json"
    artifacts.write_canonical_json(path, _nan_free(summary))
    log(f"[finalize] {results.name}: {stop_reason} after {steps_done} steps ({sim.state.time_s * 1e6:.2f} us, {summary['ion_transit_times']:.2f} transits), "
        f"I_d {1e3 * window_currents['discharge_a']:.3f} mA, S {window_currents['ionization_rate_per_s']:.3e} /s, residual window "
        f"{residual.get('ratio')}, {summary['ms_per_step']:.1f} ms/step")
    return path


def _nan_free(value: Any) -> Any:
    """Canonical JSON refuses NaN/inf: replace them by None (recorded as null)."""

    if isinstance(value, dict):
        return {k: _nan_free(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_free(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _nan_free(value.tolist())
    return value


def run_case(protocol: dict[str, Any], case: dict[str, Any], results: Path, *, field_kind: str = "real", max_steps: int | None = None,
             log: Log = _print, resume: bool = False) -> Path:
    """Run one case to its stopping rule and finalize it.  ``field_kind`` 'synthetic' is the shakedown path."""

    results.mkdir(parents=True, exist_ok=True)
    if max_steps is not None:
        case = {**case, "max_steps": int(max_steps)}
    config = build_config(protocol, case)
    grid = config.grid
    t0 = time.perf_counter()
    if field_kind == "real":
        field = real_field(grid)
        partition = real_partition(grid, protocol)
        cross_sections = XenonCrossSections.from_file()
    else:
        field, partition, cross_sections = synthetic_inputs(grid)
    sim = HybridL2Simulation(config, field, cross_sections, partition)
    setup_seconds = time.perf_counter() - t0
    sessions = []
    series_path = results / "series.jsonl"
    status_path = results / "status.jsonl"
    checkpoint_json = results / "checkpoint-latest.json"
    stop_file = results / "STOP"
    if resume:
        if not checkpoint_json.is_file():
            raise HybridValidationError("--resume needs checkpoint-latest.json")
        report = load_checkpoint_v2(checkpoint_json, sim)
        # a run stopped between two checkpoints has series records past the checkpoint step; drop them so the
        # resumed run re-produces (not duplicates) that stretch.  The truncated file is the canonical series.
        kept = [r for r in _read_jsonl(series_path) if int(r["step"]) <= int(report["step"])]
        dropped = len(_read_jsonl(series_path)) - len(kept)
        _write_jsonl(series_path, kept)
        sim.series = kept
        sessions = json.loads((results / "sessions.json").read_text(encoding="utf-8")) if (results / "sessions.json").is_file() else []
        _append_jsonl(status_path, {"utc": utc_now(), "event": "resume", "step": int(report["step"]), "series_records_dropped": dropped})
        log(f"[run] resumed {results.name} at step {report['step']} (field replay {report['field']['mode']}; {dropped} series records past the checkpoint dropped)")
    else:
        if series_path.exists() or checkpoint_json.exists():
            raise HybridValidationError(f"{results} already holds a run; use --resume or a fresh directory")
    sessions.append({"pid": os.getpid(), "started_utc": utc_now(), "resumed_from_step": sim.state.step, "host": socket.gethostname(),
                     "git_head": git_head(), "blas_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}})
    (results / "sessions.json").write_text(json.dumps(sessions, indent=1), encoding="utf-8", newline="\n")
    log(f"[run] {results.name}: grid {grid.cell_shape} dt {config.dt_s:.2e} W {config.macro_weight:g} seed {config.seed}; field {field.source_sha256[:12]} "
        f"({field_kind}), populated {int(sim.populated_node.sum())}/{int(sim.masks.plasma_node.sum())} nodes, setup {setup_seconds:.1f} s")
    start_step = sim.state.step
    t_run = time.perf_counter()
    stop_reason = "max_steps_reached"
    last_status = time.perf_counter()
    try:
        while sim.state.step < config.max_steps:
            record = sim.step()
            if record:
                _append_jsonl(series_path, record)
                if sim.state.step % config.checkpoint_every_steps == 0:
                    save_checkpoint_v2(results, "checkpoint-latest", sim)
                if stop_file.exists():
                    # the runner's stop mechanism: a STOP file in the results directory ends the run at the next series
                    # record with a checkpoint (no finalize); `launch --resume` continues it later.  The file is consumed.
                    save_checkpoint_v2(results, "checkpoint-latest", sim)
                    stop_file.unlink()
                    _append_jsonl(status_path, {"utc": utc_now(), "event": "stop_requested", "step": sim.state.step, "time_s": sim.state.time_s,
                                                "wall_seconds_session": time.perf_counter() - t_run})
                    log(f"[run] {results.name} STOP requested: checkpoint-latest written at step {sim.state.step}; resume with --resume")
                    return results / "checkpoint-latest.json"
                plateau = sim.plateau()
                now = time.perf_counter()
                if now - last_status > 30.0 or (plateau is not None and plateau["reached"]):
                    last_status = now
                    _append_jsonl(status_path, {"utc": utc_now(), "step": sim.state.step, "time_s": sim.state.time_s, "ions": sim.state.ions.count,
                                                "electrons": float(sim.state.electron_count.sum()), "discharge_a": record["current_discharge_a"],
                                                "ionization_rate_per_s": record["current_ionization_rate_per_s"],
                                                "neutral_density_per_m3": record["neutral_density_per_m3"], "plateau": plateau,
                                                "ms_per_step": 1e3 * (now - t_run) / max(sim.state.step - start_step, 1),
                                                "cells": {f"cell{k}": [record[f"cell{k}_potential_v"], record[f"cell{k}_temperature_ev"]]
                                                          for k in range(sim.partition.cell_count)}})
                    log(f"[run] {results.name} step {sim.state.step} t {sim.state.time_s * 1e6:.3f} us I_d {1e3 * record['current_discharge_a']:.3f} mA "
                        f"S {record['current_ionization_rate_per_s']:.3e} n_g {record['neutral_density_per_m3']:.3e} ions {sim.state.ions.count} "
                        f"T {[round(record[f'cell{k}_temperature_ev'], 1) for k in range(sim.partition.cell_count)]} "
                        f"phi {[round(record[f'cell{k}_potential_v']) for k in range(sim.partition.cell_count)]} "
                        f"{1e3 * (now - t_run) / max(sim.state.step - start_step, 1):.0f} ms/step")
                if plateau is not None and plateau["reached"]:
                    stop_reason = "plateau_reached_after_min_transit_times"
                    break
    except HybridError as error:
        stop_reason = f"fail_closed:{type(error).__name__}"
        _append_jsonl(status_path, {"utc": utc_now(), "event": "fail_closed", "error": str(error), "step": sim.state.step})
        log(f"[run] {results.name} FAIL-CLOSED at step {sim.state.step}: {error}")
        if not sim.series:
            raise
    wall = time.perf_counter() - t_run
    return finalize(sim, results, protocol=protocol, case=case, stop_reason=stop_reason, wall_seconds=wall, field_kind=field_kind, log=log, sessions=sessions)


# -- preflight / shakedown -------------------------------------------------------------------------------------------------------

def preflight(protocol: dict[str, Any], *, timing_steps: int = 40, output: Path = PREFLIGHT_PATH, log: Log = _print) -> dict[str, Any]:
    """Real inputs on every case grid: cusp-plane check, populated fraction, closures re-derived, factorisation and ms/step. Non-evidentiary."""

    record: dict[str, Any] = {"schema_version": "cft-revival.hybrid-l2-v2.preflight/1.0.0", "utc": utc_now(), "git_head": git_head(),
                              "protocol_sha256": file_sha256(PROTOCOL_PATH), "non_evidentiary": True, "host": socket.gethostname(),
                              "python": sys.version.split()[0], "grids": {}}
    base = resolve_case(protocol, "base")
    partition = real_partition(case_grid(protocol, base), protocol)
    reference = closure.build_pic_reference(partition, PIC_V2)
    record["partition"] = partition.to_dict()
    record["closures_rederived"] = reference["closures"]
    declared = protocol["closures"]
    agree = (np.allclose(reference["closures"]["cusp_conductance_s"], declared["cusp_conductance_s"], rtol=1e-9)
             and np.allclose(reference["closures"]["leak_half_width_m"], declared["leak_half_width_m"], rtol=1e-9))
    record["closures_agree_with_protocol"] = bool(agree)
    if not agree:
        raise HybridValidationError("protocol closures disagree with the PIC artifacts on disk")
    quantities = reference["quantities"]
    pinned = protocol["pic_reference"]["quantities"]
    mismatch = [k for k in pinned if abs(quantities[k]["reference"] - pinned[k]["reference"]) > 1e-9 * max(abs(pinned[k]["reference"]), 1e-300)]
    record["pic_reference_agrees_with_protocol"] = not mismatch
    if mismatch:
        raise HybridValidationError(f"protocol pic_reference disagrees with the PIC artifacts: {mismatch}")
    seen_grids: set[tuple[int, int]] = set()
    for name in protocol["cases"]:
        case = resolve_case(protocol, name)
        key = (int(case["radial_cells"]), int(case["axial_cells"]))
        if key in seen_grids:
            continue
        seen_grids.add(key)
        config = build_config(protocol, case)
        t0 = time.perf_counter()
        field = real_field(config.grid)
        t_field = time.perf_counter() - t0
        t0 = time.perf_counter()
        sim = HybridL2Simulation(config, field, XenonCrossSections.from_file(), real_partition(config.grid, protocol))
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(timing_steps):
            sim.step()
        t_steps = time.perf_counter() - t0
        record["grids"][f"{key[0]}x{key[1]}"] = {
            "dr_m": config.grid.dr_m, "dz_m": config.grid.dz_m, "field_seconds": t_field, "field_source_sha256": field.source_sha256,
            "field_sha256": field.sha256, "max_b_t": field.max_b_t, "setup_seconds": t_setup, "mesh": sim.masks.to_dict(),
            "populated_nodes": int(sim.populated_node.sum()), "plasma_nodes": int(sim.masks.plasma_node.sum()),
            "population_threshold_wb": [float(t) for t in sim.population_threshold_wb],
            "timing_seed_load": {"steps": timing_steps, "ms_per_step": 1e3 * t_steps / timing_steps, "ions": sim.state.ions.count,
                                 "newton_iterations_mean": float(np.mean([r["newton_iterations_mean"] for r in sim.series])) if sim.series else None},
            "cases_on_grid": [n for n in protocol["cases"] if (int(resolve_case(protocol, n)["radial_cells"]), int(resolve_case(protocol, n)["axial_cells"])) == key],
        }
        log(f"[preflight] grid {key}: field {t_field:.1f} s, setup {t_setup:.1f} s, {1e3 * t_steps / timing_steps:.0f} ms/step at seed load, "
            f"populated {int(sim.populated_node.sum())}/{int(sim.masks.plasma_node.sum())}")
    base_ms = record["grids"][f"{base['radial_cells']}x{base['axial_cells']}"]["timing_seed_load"]["ms_per_step"]
    record["projection"] = {
        "base_steps_to_3_transits": 3.0 * protocol["stopping_rule"]["plateau"]["ion_transit_time_s"] / float(base["dt_s"]),
        "base_minutes_to_3_transits_at_seed_load": 3.0 * protocol["stopping_rule"]["plateau"]["ion_transit_time_s"] / float(base["dt_s"]) * base_ms / 6e4,
        "note": "the step cost grows with the ion count (~+50 % from seed to plateau in the development check); the PIC base plateau cost 10,141 s on an RTX 5090",
    }
    artifacts.write_canonical_json(output, _nan_free(record))
    log(f"[preflight] written {output}")
    return record


def shakedown(protocol: dict[str, Any], *, output: Path = SHAKEDOWN_PATH, log: Log = _print, real_steps: int = 300, synthetic_steps: int = 400) -> dict[str, Any]:
    """Synthetic-field full path (run -> finalize -> assess) on a coarse grid, then the real field for a short run. Non-evidentiary."""

    import shutil

    record: dict[str, Any] = {"schema_version": "cft-revival.hybrid-l2-v2.shakedown/1.0.0", "utc": utc_now(), "git_head": git_head(), "non_evidentiary": True}
    synthetic_case = {**resolve_case(protocol, "spatial-coarse"), "name": "shakedown-synthetic", "series_interval_steps": 10,
                      "averaging_window_steps": 100, "checkpoint_every_steps": 100, "residual_window_steps": 100}
    results = HERE / "results-shakedown-synthetic"
    if results.exists():
        shutil.rmtree(results)
    t0 = time.perf_counter()
    summary_path = run_case(protocol, synthetic_case, results, field_kind="synthetic", max_steps=synthetic_steps, log=log)
    summary = artifacts.read_canonical_json(summary_path)
    record["synthetic"] = {"results_dir": results.name, "seconds": time.perf_counter() - t0, "stop_reason": summary["stop_reason"],
                           "steps": summary["steps_completed"], "ms_per_step": summary["ms_per_step"],
                           "charge_identity_max_relative": summary["charge_identity_max_relative"],
                           "neutral_ledger_closure_relative": summary["neutral_inventory"]["ledger_closure"]["closure_relative_to_inventory"],
                           "windowed_energy_residual": summary["windowed_energy_residual"]}
    # resume path: reload the latest checkpoint into a fresh simulation and step once
    config = build_config(protocol, {**synthetic_case, "max_steps": synthetic_steps})
    field, partition, xs = synthetic_inputs(config.grid)
    sim = HybridL2Simulation(config, field, xs, partition)
    report = load_checkpoint_v2(results / "checkpoint-final.json", sim)
    sim.step()
    record["synthetic"]["resume_check"] = {"loaded_step": report["step"], "field_mode": report["field"]["mode"], "stepped_to": sim.state.step}
    assessment = assess(protocol, cases=[("shakedown-synthetic", results)], output=results / "assessment.json", log=log, require_reference_consistency=True)
    record["synthetic"]["assessment_verdict"] = assessment["gate_l2"]["verdict"]
    # real field, short
    real_case = {**resolve_case(protocol, "base"), "name": "shakedown-real", "series_interval_steps": 10, "averaging_window_steps": 100,
                 "checkpoint_every_steps": 100, "residual_window_steps": 100}
    results_real = HERE / "results-shakedown-real"
    if results_real.exists():
        shutil.rmtree(results_real)
    t0 = time.perf_counter()
    summary_path = run_case(protocol, real_case, results_real, field_kind="real", max_steps=real_steps, log=log)
    summary = artifacts.read_canonical_json(summary_path)
    record["real"] = {"results_dir": results_real.name, "seconds": time.perf_counter() - t0, "stop_reason": summary["stop_reason"],
                      "steps": summary["steps_completed"], "ms_per_step": summary["ms_per_step"], "window_currents_a": summary["window_currents_a"],
                      "cells": summary["cells"], "charge_identity_max_relative": summary["charge_identity_max_relative"],
                      "windowed_energy_residual": summary["windowed_energy_residual"],
                      "field_source_sha256": summary["provenance"]["field"]["field_source_sha256"]}
    assessment = assess(protocol, cases=[("shakedown-real", results_real)], output=results_real / "assessment.json", log=log)
    record["real"]["assessment_verdict"] = assessment["gate_l2"]["verdict"]
    artifacts.write_canonical_json(output, _nan_free(record))
    log(f"[shakedown] synthetic {record['synthetic']['steps']} steps ({record['synthetic']['stop_reason']}), real {record['real']['steps']} steps "
        f"({record['real']['stop_reason']}, {record['real']['ms_per_step']:.0f} ms/step); written {output}")
    return record


# -- launch -----------------------------------------------------------------------------------------------------------------------

def worktree_status() -> list[str]:
    return [line for line in git("status", "--porcelain", "--untracked-files=normal").splitlines() if line.strip()]


def acquire_lock(results: Path, payload: dict[str, Any]) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    path = results / LOCK_NAME
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise HybridValidationError(f"execution lock already exists at {path}; refusing to launch") from None
    with os.fdopen(fd, "wb") as handle:
        handle.write(canonical_bytes(payload) + b"\n")
    return path


def launch(protocol: dict[str, Any], *, case_name: str, expect_commit: str | None, allow_dirty: bool, resume: bool, log: Log = _print) -> Path:
    head = git_head()
    if expect_commit is not None and (head is None or not head.startswith(expect_commit)):
        raise HybridValidationError(f"HEAD {head} is not the preregistration commit {expect_commit}")
    dirty = worktree_status()
    if dirty and not allow_dirty:
        raise HybridValidationError(f"worktree is not clean ({len(dirty)} entries, e.g. {dirty[0]!r})")
    if git("rev-parse", f"HEAD:{PROTOCOL_PATH.relative_to(REPOSITORY_ROOT).as_posix()}") != git("hash-object", "--", str(PROTOCOL_PATH)):
        raise HybridValidationError("protocol.json on disk differs from the committed blob at HEAD")
    if not PREFLIGHT_PATH.is_file() or not SHAKEDOWN_PATH.is_file():
        raise HybridValidationError("preflight.json and shakedown.json must exist before a launch")
    case = resolve_case(protocol, case_name)
    results = results_dir_for(case_name)
    payload = {"schema_version": "cft-revival.hybrid-l2-v2.execution-lock/1.0.0", "experiment_id": protocol["experiment_id"], "case": case_name,
               "commit": head, "protocol_sha256": file_sha256(PROTOCOL_PATH), "config_sha256": content_hash(build_config(protocol, case).to_dict()),
               "host": socket.gethostname(), "pid": os.getpid(), "acquired_at_utc": utc_now(), "clean_worktree_attested": not dirty}
    if not resume:
        acquire_lock(results, payload)
        log(f"[launch] lock acquired for {case_name} at {head[:12] if head else '?'}")
    return run_case(protocol, case, results, field_kind="real", log=log, resume=resume)


# -- assessment ---------------------------------------------------------------------------------------------------------------------

def l2_quantities(results: Path) -> dict[str, Any]:
    summary = artifacts.read_canonical_json(results / "summary.json")
    targets = json.loads((results / "l2-targets.json").read_text(encoding="utf-8"))
    wc = summary["window_currents_a"]
    run = {"targets": targets, "window_currents_a": wc, "neutral_density_per_m3": summary["window_means"]["neutral_density_per_m3"],
           "ionization_rate_per_s": wc["ionization_rate_per_s"], "gross_utilisation": summary["neutral_inventory"]["gross_utilisation_window"],
           "peak_n_e_per_m3": summary["window_maps_summary"]["peak_n_e_per_m3"]}
    quantities = closure.scalar_quantities(run)
    return {"summary": summary, "quantities": quantities}


def assess(protocol: dict[str, Any], *, cases: list[tuple[str, Path]] | None = None, output: Path | None = None, log: Log = _print,
           require_reference_consistency: bool = True, pic_v4_results: str | Path | None = None) -> dict[str, Any]:
    """GATE-L2 metrics over the finished cases; the base case (or the single given case) carries the comparison."""

    if cases is None:
        cases = [(name, results_dir_for(name)) for name in protocol["cases"] if (results_dir_for(name) / "summary.json").is_file()]
    if not cases:
        raise HybridValidationError("no finished case to assess")
    pinned = protocol["pic_reference"]["quantities"]
    consistency = None
    if require_reference_consistency:
        base = resolve_case(protocol, "base")
        partition = real_partition(case_grid(protocol, base), protocol)
        rederived = closure.build_pic_reference(partition, PIC_V2)["quantities"]
        consistency = {k: {"pinned": pinned[k]["reference"], "recomputed": rederived[k]["reference"],
                           "agree": abs(rederived[k]["reference"] - pinned[k]["reference"]) <= 1e-9 * max(abs(pinned[k]["reference"]), 1e-300)} for k in pinned}
        if not all(v["agree"] for v in consistency.values()):
            raise HybridValidationError("protocol.pic_reference disagrees with the PIC artifacts on disk")
    per_case: dict[str, Any] = {}
    for name, results in cases:
        try:
            per_case[name] = l2_quantities(results)
            per_case[name]["finished"] = True
        except (OSError, KeyError, ValueError) as error:
            per_case[name] = {"finished": False, "error": f"{type(error).__name__}: {error}"}
    headline_name = "base" if "base" in per_case else cases[0][0]
    headline = per_case[headline_name]
    if not headline.get("finished"):
        raise HybridValidationError(f"headline case {headline_name} did not finish: {headline.get('error')}")
    summary = headline["summary"]
    bounds = protocol["gates"]["interface_conservation"]
    conservation = gates.interface_conservation(
        charge_identity_max_relative=summary["charge_identity_max_relative"], charge_identity_bound=bounds["charge_identity_relative_max"],
        neutral_ledger_closure_relative=summary["neutral_inventory"]["ledger_closure"]["closure_relative_to_inventory"],
        neutral_ledger_bound=bounds["neutral_ledger_relative_max"],
        windowed_energy_residual_ratio=summary["windowed_energy_residual"].get("ratio"), energy_residual_bound=bounds["energy_residual_over_electrode_work_max"],
        plateau_reached=bool(summary["stop_reason"] == "plateau_reached_after_min_transit_times"),
    )
    comparisons = []
    for key, entry in pinned.items():
        value = headline["quantities"].get(key)
        comparisons.append(gates.compare(key, value, entry["reference"], entry["tolerance"]))
    comparison = gates.code_comparison(comparisons)
    level_keys = protocol["gates"]["level_quantities"]

    def level_rows(names: list[str]) -> list[dict[str, Any]]:
        rows = []
        for n in names:
            entry = per_case.get(n)
            finished = bool(entry and entry.get("finished") and entry["summary"]["stop_reason"] == "plateau_reached_after_min_transit_times")
            rows.append({"label": n, "finished": finished, "quantities": entry["quantities"] if entry and entry.get("finished") else {},
                         "stop_reason": entry["summary"]["stop_reason"] if entry and entry.get("finished") else entry.get("error") if entry else "not run"})
        return rows

    spatial = gates.levels_gate(level_rows(protocol["gates"]["spatial_levels"]), minimum=3, quantity_keys=level_keys)
    temporal = gates.levels_gate(level_rows(protocol["gates"]["temporal_levels"]), minimum=3, quantity_keys=level_keys)
    statistical = gates.levels_gate(level_rows(protocol["gates"]["statistical_levels"]), minimum=1, quantity_keys=level_keys)
    input_levels = gates.levels_gate(level_rows(protocol["gates"]["input_levels"]), minimum=1, quantity_keys=level_keys)

    def spread_statement(block: dict[str, Any]) -> dict[str, Any]:
        values = {k: v["max_relative_spread"] for k, v in block["spread"].items()}
        finite = [v for v in values.values() if v is not None]
        return {"value": max(finite) if finite else None, "per_quantity": values, "levels": block["labels"]}

    uncertainty = gates.uncertainty_components(
        input_component={**spread_statement(input_levels), "statement": "closure sensitivity: cusp conductances and leak widths scaled 0.7 / 1.3"},
        numerical={"value": max([v for b in (spatial, temporal, statistical) for v in spread_statement(b)["per_quantity"].values() if v is not None], default=None),
                   "spatial": spread_statement(spatial), "temporal": spread_statement(temporal), "statistical": spread_statement(statistical)},
        emulator={"value": 0.0, "statement": "no emulator or surrogate is used anywhere in L2 v2 (the component is identically zero)"},
        model_discrepancy={"value": max([abs(c.relative_difference) for c in comparisons if c.relative_difference is not None], default=None),
                           "statement": "L2 minus PIC over the compared quantities (the code comparison itself)",
                           "outside": comparison["outside"]},
    )
    failed = sum(1 for n in protocol["cases"] if n in per_case and (not per_case[n].get("finished")
                                                                    or per_case[n]["summary"]["stop_reason"] != "plateau_reached_after_min_transit_times"))
    gate = gates.evaluate_l2_gates(conservation=conservation, spatial=spatial, temporal=temporal, comparison=comparison, uncertainty=uncertainty, failed_cases=failed)
    pic_v4 = None
    v4_dir = PIC_V4 if pic_v4_results is None else Path(pic_v4_results)
    if (v4_dir / "summary.json").is_file() and (v4_dir / "maps.npz").is_file():
        # the 33 um refinement of the PIC base plateau, read-only and INFORMATIONAL (its own preregistered assessment is not ours):
        # the same per-cell extraction on the L2 partition gives a second reference column and the PIC's own grid sensitivity
        base_case = resolve_case(protocol, "base")
        v4_run = closure.pic_run_targets(v4_dir, real_partition(case_grid(protocol, base_case), protocol))
        v4_q = closure.scalar_quantities(v4_run)
        v4_summary = json.loads((v4_dir / "summary.json").read_text(encoding="utf-8"))
        pic_v4 = {
            "results_dir": str(v4_dir), "stop_reason": v4_summary.get("stop_reason"), "steps": v4_summary.get("steps_completed"),
            "simulated_time_s": v4_summary.get("simulated_time_s"), "wall_seconds_total": v4_summary.get("wall_seconds_total"),
            "maps_sha256": v4_run["maps_sha256"], "summary_sha256": v4_run["summary_sha256"],
            "quantities": v4_q,
            "l2_relative_to_v4": {k: ((headline["quantities"][k] - v4_q[k]) / abs(v4_q[k]) if v4_q.get(k) not in (None, 0.0) and np.isfinite(v4_q[k])
                                      and headline["quantities"].get(k) is not None else None) for k in pinned},
            "v4_relative_to_base": {k: ((v4_q[k] - pinned[k]["reference"]) / abs(pinned[k]["reference"]) if pinned[k]["reference"] != 0.0 and np.isfinite(v4_q[k]) else None)
                                    for k in pinned},
            "note": "pic2d_cft_steady_state_v4 (33 um / 1.4 ps refinement of the base plateau) finished before this assessment; informational only - "
                    "its acceptance is judged by its own preregistered protocol, not here",
        }
    cost = {"l2_wall_seconds": summary["wall_seconds_total"], "l2_ms_per_step": summary["ms_per_step"], "l2_steps": summary["steps_completed"],
            "pic_base_wall_seconds": protocol["pic_reference"]["base_cost"]["wall_seconds_total"], "pic_base_steps": protocol["pic_reference"]["base_cost"]["steps"],
            "pic_base_device": protocol["pic_reference"]["base_cost"]["device"],
            "wall_clock_ratio_pic_over_l2": protocol["pic_reference"]["base_cost"]["wall_seconds_total"] / summary["wall_seconds_total"],
            "note": "L2 on one CPU process vs the PIC on an RTX 5090; a same-hardware ratio needs the PIC's CPU step cost (not measured here)"}
    record = {
        "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "git_head_now": git_head(),
        "headline_case": headline_name, "cases": {n: {k: v for k, v in c.items() if k != "summary"} | ({"stop_reason": c["summary"]["stop_reason"],
                                                                                                  "steps": c["summary"]["steps_completed"],
                                                                                                  "wall_seconds": c["summary"]["wall_seconds_total"]} if c.get("finished") else {})
                                                   for n, c in per_case.items()},
        "reference_consistency": consistency, "interface_conservation": conservation, "code_comparison": comparison,
        "spatial_levels": spatial, "temporal_levels": temporal, "statistical_levels": statistical, "input_levels": input_levels,
        "uncertainty": uncertainty, "gate_l2": gate, "pic_v4": pic_v4, "cost": cost, "claim_boundary": protocol["claim_boundary"],
        "prohibited_until_accepted": protocol["prohibited_until_accepted"],
    }
    target = output or (RESULTS / "assessment.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    artifacts.write_canonical_json(target, _nan_free(record))
    log(f"[assess] verdict {gate['verdict']}: conservation {conservation['passed']}, comparison {comparison['passed']} "
        f"({comparison['compared']} compared, outside {comparison['outside']}), spatial {spatial['levels_completed']}, temporal {temporal['levels_completed']}, "
        f"failed cases {failed}; written {target}")
    return record


def status(protocol: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in protocol["cases"]:
        results = results_dir_for(name)
        lines = _read_jsonl(results / "status.jsonl")
        out[name] = {"finished": (results / "summary.json").is_file(), "last_status": lines[-1] if lines else None}
    return out


# -- CLI --------------------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--timing-steps", type=int, default=40)
    sub.add_parser("shakedown")
    la = sub.add_parser("launch")
    la.add_argument("--case", default="base")
    la.add_argument("--expect-commit", default=None)
    la.add_argument("--allow-dirty", action="store_true")
    la.add_argument("--resume", action="store_true")
    sub.add_parser("status")
    ass = sub.add_parser("assess")
    ass.add_argument("--no-reference-check", action="store_true")
    ass.add_argument("--pic-v4-results", default=None, help="read-only path of a finished pic2d_cft_steady_state_v4 results directory (informational)")
    args = parser.parse_args(argv)
    protocol = load_protocol()
    if args.command == "preflight":
        preflight(protocol, timing_steps=args.timing_steps)
    elif args.command == "shakedown":
        shakedown(protocol)
    elif args.command == "launch":
        launch(protocol, case_name=args.case, expect_commit=args.expect_commit, allow_dirty=args.allow_dirty, resume=args.resume)
    elif args.command == "status":
        print(json.dumps(status(protocol), indent=1, default=str))
    else:
        assess(protocol, require_reference_consistency=not args.no_reference_check, pic_v4_results=args.pic_v4_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
