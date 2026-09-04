"""PIC design mini-sweep v1 - runner stages.

From ``modern/`` (``$env:PYTHONPATH="$PWD\\src;$PWD"`` / ``PYTHONPATH=src:.``)::

    python -m experiments.pic2d_design_mini_sweep_v1.run fields [--design ID]                    # CPU: padded P2 solves -> fields/<id>/binding.json (sequential)
    python -m experiments.pic2d_design_mini_sweep_v1.run preflight --domain channel --grid 33um   # whole-set preflight -> preflight-channel-33um.json
    python -m experiments.pic2d_design_mini_sweep_v1.run protocol --design ID --grid 33um         # print the composed per-design run protocol
    python -m experiments.pic2d_design_mini_sweep_v1.run compose --grid 33um                      # seal protocols/<design>-channel-33um[-seed-replicate].json + protocol.json
    python -m experiments.pic2d_design_mini_sweep_v1.run cost                                     # cost table + schedules
    python -m experiments.pic2d_design_mini_sweep_v1.run mps-replay --design ID --grid 33um       # GPU: same-seed determinism under CUDA MPS -> mps-replay.json
    python -m experiments.pic2d_design_mini_sweep_v1.run shakedown --design ID --grid 33um        # GPU: shrunk-cadence run -> finalize -> assess -> targets -> shakedown-channel-33um.json
    python -m experiments.pic2d_design_mini_sweep_v1.run launch --design ID --grid 33um --expect-commit SHA [--require-mps] [--resume]   # PREREGISTERED execution
    python -m experiments.pic2d_design_mini_sweep_v1.run run --design ID --grid 33um --allow-launch ...   # labelled development / shakedown run (never evidence)
    python -m experiments.pic2d_design_mini_sweep_v1.run status|finalize|assess|targets --design ID --grid 33um

``launch`` and ``run`` step with the shared steady-state runner (``experiments.pic2d_cft_steady_state_v1.run``: checkpoints,
resume, plateau rule, gates, frames) under the per-design protocol of ``protocol.build_protocol`` with the design's hash-bound
node field passed in directly; results go to ``results/<design_id>-<option>[-<case>]/``.  ``assess`` applies the predeclared
per-design acceptance (a plateau, b residual power, verdict; the steady-state v4 convergence verdict cited when it exists);
``targets`` extracts the closure targets (``closure.extract_targets``) from a finished run's ``maps.npz`` + ``summary.json``.

Preregistration discipline of ``launch`` (the v4 discipline, per design): HEAD == --expect-commit, clean worktree, the
experiment ``protocol.json`` and the sealed ``protocols/<design>-<option>.json`` blobs equal HEAD's, the recomposed protocol
equals the sealed file byte for byte, ``preflight-<option>.json`` + ``shakedown-<option>.json`` + ``mps-replay.json`` present,
``--require-mps`` -> the CUDA MPS pipe directory must exist, O_EXCL ``execution-lock.json`` in the results directory
(``--resume`` continues only under the same commit + protocol).  ``run`` stays a labelled development entry: it refuses to
start without ``--allow-launch`` and never writes a lock.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

from . import cost as cost_module
from . import designs as design_module
from . import fields as field_module
from . import preflight as preflight_module
from . import protocol as protocol_module
from .closure import extract_targets, load_maps
from .protocol import CASES, GRID_VARIANTS, build_protocol, compose_run_protocol, composed_protocol_path, option_tag, protocol_bytes

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
RESULTS_ROOT = HERE / "results"
LOCK_NAME = "execution-lock.json"
LOCK_SCHEMA = "cft.pic2d.design-mini-sweep.execution-lock/1.0.0"
ASSESSMENT_SCHEMA = "cft.pic2d.design-mini-sweep.assessment/1.0.0"
SHAKEDOWN_SCHEMA = "cft.pic2d.design-mini-sweep.shakedown/1.0.0"
MPS_REPLAY_SCHEMA = "cft.pic2d.design-mini-sweep.mps-replay/1.0.0"
MPS_REPLAY_PATH = HERE / "mps-replay.json"
STEADY_STATE_V4_ASSESSMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results" / "assessment.json"

# shakedown / replay: the real protocol with only the cadences shrunk (every gate, the grid, dt, W, field and seed are the real ones) - the v4 values
SHRUNK_CADENCES = {
    "series_interval_steps": 200, "device_sync_steps": 200, "checkpoint_every_steps": 4000, "averaging_window_steps": 40000,
    "frame_cadence_steps": 2000, "peak_debye_window_steps": 40000, "peak_debye_window_snapshot_steps": 4000, "residual_window_steps": 40000,
}
SHAKEDOWN_MAX_STEPS = 100000
MPS_REPLAY_STEPS = 6000


def results_dir(design_id: str, domain: str, grid: str = "50um", case: str = "base") -> Path:
    suffix = "" if case == "base" else f"-{case}"
    return RESULTS_ROOT / f"{design_id}-{option_tag(domain, grid)}{suffix}"


def shakedown_path(domain: str, grid: str) -> Path:
    return HERE / f"shakedown-{option_tag(domain, grid)}.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=1, sort_keys=True, default=str))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=_plain).encode("utf-8") + b"\n")


def _plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _runner():
    from experiments.pic2d_cft_steady_state_v1 import run as runner

    return runner


def git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


# -- composition ----------------------------------------------------------------------------------------------------------------


def command_fields(args: argparse.Namespace) -> int:
    ids = (args.design,) if args.design else design_module.design_ids()
    for design_id in ids:      # sequential: one host factorisation / solve at a time (BLAS oversubscription lesson)
        binding = field_module.produce_field(design_id)
        print(f"[fields] {design_id}: {'ok' if binding['gates']['all_passed'] else 'GATES FAILED'} -> {field_module.binding_path(design_id)}")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    path, report = preflight_module.write_preflight(args.domain, grid=args.grid, design_ids=(args.design,) if args.design else None)
    print(f"[preflight] {option_tag(args.domain, args.grid)}: all_passed={report['all_passed']} over {report['design_count']} designs -> {path}")
    return 0 if report["all_passed"] else 1


def command_protocol(args: argparse.Namespace) -> int:
    if args.with_field:
        protocol, _, _ = compose_run_protocol(args.design, args.domain, args.grid, args.case)
    else:
        protocol, _ = build_protocol(args.design, args.domain, grid=args.grid, case=args.case)
    _print_json(protocol)
    return 0


def _preflight_summary(domain: str, grid: str) -> dict[str, Any] | None:
    path = preflight_module.preflight_path(domain, grid)
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {"domain": report["domain"], "grid": report.get("grid", "50um"), "option": report.get("option", option_tag(domain, grid)), "all_passed": report["all_passed"],
            "design_count": report["design_count"], "generated_utc": report["generated_utc"], "platform": report.get("platform"),
            "designs": {r["design_id"]: {"passed": r["passed"], "gates": {k: v["passed"] for k, v in r["gates"].items()},
                                         "dt_s": r["gates"].get("field_map", {}).get("dt_s"), "macro_weight": r["gates"].get("protocol", {}).get("macro_weight"),
                                         "cells": r["gates"].get("protocol", {}).get("cells"), "wall_budget_hours": r["gates"].get("protocol", {}).get("wall_budget_hours"),
                                         "field_source_sha256": r["gates"].get("field_map", {}).get("field_source_sha256")} for r in report["designs"]}}


def command_compose(args: argparse.Namespace) -> int:
    """Seal the per-design run protocols of the option and rewrite the experiment protocol.json around them."""

    sealed = protocol_module.compose_all(args.domain, args.grid)
    summary = _preflight_summary(args.domain, args.grid)
    out = protocol_module.write_experiment_protocol(preflight_summary=summary, sealed=sealed, preregistered=(args.domain, args.grid) == protocol_module.PREREGISTERED_OPTION)
    print(f"[compose] {len(sealed)} sealed run protocols under {protocol_module.PROTOCOLS_DIR}; experiment protocol -> {out} (sha256 {_sha256(out)[:12]})")
    return 0


def command_draft_protocol(args: argparse.Namespace) -> int:
    out = protocol_module.write_experiment_protocol(preflight_summary=_preflight_summary(args.domain, args.grid), preregistered=(args.domain, args.grid) == protocol_module.PREREGISTERED_OPTION)
    print(f"[experiment-protocol] wrote {out}")
    return 0


def command_cost(args: argparse.Namespace) -> int:
    built = [design_module.build_design(d.design_id) for d in design_module.SWEEP_DESIGNS]
    table = cost_module.cost_table(built)
    for domain, rows in table.items():
        print(f"== {domain} ==")
        for row in rows:
            print(f"  {row['design_id']:<28} nodes {row['nodes'][0]}x{row['nodes'][1]:<5} W {row['macro_weight']:<9.6g} N {row['particles_projected_m']['total_m']:.2f} M  "
                  f"{row['ms_per_step']:.2f} ms/step ({row['platform']}; 5090 model {row['ms_per_step_rtx5090_model']:.2f}, H100 MPS-4 {row['ms_per_step_h100_mps4_per_process']:.2f})  "
                  f"transit {row['transit_s']*1e6:.2f} us  {row['steps_to_transits']/1e6:.2f} M steps  {row['wall_hours']:.1f} h  {row['device_gb_projected']:.1f} GB  fact {row['factorisation_s']/60:.1f} min")
    for option in table:
        schedule = cost_module.serial_schedule(table, option=option)
        print(f"serial schedule {option}: {schedule['total_hours']:.1f} h = {schedule['total_days_at_24h']:.2f} days for {len(schedule['items'])} runs")
    print("anchors:", json.dumps(cost_module.anchor_residuals()))
    return 0


# -- shrunk-cadence protocol (shakedown / replay) ------------------------------------------------------------------------------


def shrunk_protocol(protocol: dict[str, Any], label: str, overrides: dict[str, Any] = SHRUNK_CADENCES) -> dict[str, Any]:
    """The real protocol with every cadence shrunk (NON-EVIDENTIARY): grid, dt, W, field, seed and gate thresholds untouched."""

    p = copy.deepcopy(protocol)
    num = p["numerics"]
    num["series_interval_steps"] = overrides["series_interval_steps"]
    num["device_sync_steps"] = overrides["device_sync_steps"]
    num["checkpoint_every_steps"] = overrides["checkpoint_every_steps"]
    num["averaging_window_steps"] = overrides["averaging_window_steps"]
    if num.get("frame_recorder") is not None:
        num["frame_recorder"] = {"cadence_steps": overrides["frame_cadence_steps"], "precision": "float32"}
    gate = num.get("peak_debye_gate") or {}
    if gate.get("window_steps") is not None:
        gate["window_steps"] = overrides["peak_debye_window_steps"]
        gate["window_snapshot_steps"] = overrides["peak_debye_window_snapshot_steps"]
    triad = p["stopping_rule"].get("grid_heating_triad") or {}
    if triad.get("residual_window_steps") is not None:
        triad["residual_window_steps"] = overrides["residual_window_steps"]
    p["status"] = f"{label}_non_evidentiary_shrunk_cadences"
    p["experiment_id"] = protocol["experiment_id"] + f"-{label}"
    return p


# -- development / shakedown run ----------------------------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    if not args.allow_launch:
        print("[run] REFUSED: `run` is the labelled development entry (never evidence); pass --allow-launch for a shakedown / replay, or use `launch` for the preregistered execution", file=sys.stderr)
        return 2
    path = preflight_module.preflight_path(args.domain, args.grid)
    if not path.is_file():
        print(f"[run] REFUSED: no whole-set preflight for {option_tag(args.domain, args.grid)} ({path}); run `preflight --domain {args.domain} --grid {args.grid}` first", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["all_passed"]:
        print(f"[run] REFUSED: the whole-set preflight for {option_tag(args.domain, args.grid)} did not pass every design", file=sys.stderr)
        return 2
    protocol, _, field_map = compose_run_protocol(args.design, args.domain, args.grid, args.case)
    if args.shrunk_cadences:
        protocol = shrunk_protocol(protocol, args.label or "development")
    results = Path(args.results_dir) if args.results_dir else results_dir(args.design, args.domain, args.grid, args.case)
    results.mkdir(parents=True, exist_ok=True)
    protocol_path = results / "protocol.json"
    protocol_path.write_bytes(protocol_bytes(protocol))
    _runner().run_steady_state(protocol, results, backend=args.backend, field_map=field_map, max_steps=args.max_steps, wall_budget_seconds=args.wall_budget_seconds,
                               require_same_code=not args.ignore_code_identity, protocol_path=protocol_path)
    return 0


# -- MPS determinism replay ---------------------------------------------------------------------------------------------------


def _replay_child_command(design_id: str, domain: str, grid: str, results: Path, steps: int) -> list[str]:
    return [sys.executable, "-u", "-m", "experiments.pic2d_design_mini_sweep_v1.run", "run", "--design", design_id, "--domain", domain, "--grid", grid,
            "--allow-launch", "--shrunk-cadences", "--label", "mps-replay", "--max-steps", str(steps), "--results-dir", str(results)]


# Float-atomic DIAGNOSTIC accumulators (v2.0.2 window sums: sum w v, sum w v^2, sample counts; the energy-ledger interval sums; the
# peak-node statistics derived from them).  Device float atomics are order-dependent, so these differ at round-off between ANY two
# runs on the same GPU (solo or concurrent).  The PHYSICS state (particles, fixed-point charge deposition, potential, densities,
# ionisation, wall fluxes, currents, counts, neutral inventory) is bitwise.  A replay PASSES when every physics array / record is
# bitwise and every diagnostic difference is within DIAGNOSTIC_RTOL - and MPS is "neutral" when the concurrent pairs show the same
# pattern as the solo pair.
DIAGNOSTIC_MAP_KEYS = {"t_e_ev", "sample_count_e", "t_e_perp_ev", "t_e_par_ev", "mean_energy_e_ev", "sample_count_i",
                       "wall_electron_mean_energy_ev", "wall_ion_mean_energy_ev"}     # energy sums per wall cell (float atomics); the FLUXES (counts) are physics
DIAGNOSTIC_CHECKPOINT_KEYS = {"cumulative", "cumulative_extra"}
DIAGNOSTIC_SERIES_TOP_KEYS = {"ledger", "peak_node"}                  # series.jsonl record blocks built from the float-atomic sums
DIAGNOSTIC_SERIES_NPZ_PREFIXES = ("peak_node_", "interval_", "ledger_", "cumulative_")   # their flattened series.npz columns
DIAGNOSTIC_RTOL = 1.0e-6


def _is_diagnostic(kind: str, key: str) -> bool:
    if kind == "maps.npz":
        return key in DIAGNOSTIC_MAP_KEYS
    if kind == "checkpoint-final.npz":
        return key in DIAGNOSTIC_CHECKPOINT_KEYS
    if kind == "series.npz":
        return key.startswith(DIAGNOSTIC_SERIES_NPZ_PREFIXES)
    return key.split("/")[0] in DIAGNOSTIC_SERIES_TOP_KEYS                # "series": record path "ledger/cumulative/field_work_j"


def _walk_diff(x: Any, y: Any, path: str, out: dict[str, float]) -> None:
    if isinstance(x, dict) and isinstance(y, dict):
        for key in set(x) | set(y):
            _walk_diff(x.get(key), y.get(key), f"{path}/{key}" if path else str(key), out)
    elif x != y:
        rel = _max_rel(x, y) if isinstance(x, (int, float, list)) and isinstance(y, (int, float, list)) and not isinstance(x, bool) else None
        out[path] = max(out.get(path, 0.0), rel if rel is not None else float("inf"))


def _max_rel(a, b) -> float | None:
    import numpy as np

    x, y = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if x.shape != y.shape:
        return None
    scale = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-300)
    with np.errstate(invalid="ignore"):
        rel = np.abs(x - y) / scale
    rel = rel[np.isfinite(rel)]
    return float(rel.max()) if rel.size else 0.0


def _compare_runs(a: Path, b: Path) -> dict[str, Any]:
    """Compare two runs: physics arrays / records bitwise, float-atomic diagnostics within DIAGNOSTIC_RTOL (see the note above)."""

    import numpy as np

    out: dict[str, Any] = {}
    sa, sb = (a / "series.jsonl").read_bytes().splitlines(), (b / "series.jsonl").read_bytes().splitlines()
    out["series_records"] = {"a": len(sa), "b": len(sb), "identical_lines": sum(1 for x, y in zip(sa, sb) if x == y), "bitwise_equal": sa == sb}
    physics_ok, diagnostics_ok = True, True
    # series.jsonl: every record, every (nested) key; max relative difference per key path
    series_diff: dict[str, float] = {}
    if len(sa) == len(sb):
        for x, y in zip(sa, sb):
            _walk_diff(json.loads(x), json.loads(y), "", series_diff)
    else:
        physics_ok = False
    for key, rel in series_diff.items():
        if _is_diagnostic("series", key):
            diagnostics_ok = diagnostics_ok and rel <= DIAGNOSTIC_RTOL
        else:
            physics_ok = False
    out["series_records"]["differing_keys_max_rel"] = dict(sorted(series_diff.items(), key=lambda kv: -kv[1])[:40])
    out["series_records"]["physics_keys_bitwise"] = all(_is_diagnostic("series", k) for k in series_diff)
    out["series_records"]["physics_differing_keys"] = [k for k in series_diff if not _is_diagnostic("series", k)][:20]
    for name in ("maps.npz", "checkpoint-final.npz", "series.npz"):
        pa, pb = a / name, b / name
        if not pa.is_file() or not pb.is_file():
            out[name] = {"present": False}
            physics_ok = False
            continue
        with np.load(pa) as za, np.load(pb) as zb:
            keys = sorted(set(za.files) | set(zb.files))
            per_key = {k: (k in za.files and k in zb.files and np.array_equal(za[k], zb[k], equal_nan=True)) for k in keys}
            rel_diff = {k: _max_rel(za[k], zb[k]) for k, ok in per_key.items() if not ok and k in za.files and k in zb.files}
        differing = [k for k, ok in per_key.items() if not ok]
        physics_differing = [k for k in differing if not _is_diagnostic(name, k)]
        diagnostic_differing = {k: rel_diff.get(k) for k in differing if _is_diagnostic(name, k)}
        physics_ok = physics_ok and not physics_differing
        diagnostics_ok = diagnostics_ok and all(r is not None and r <= DIAGNOSTIC_RTOL for r in diagnostic_differing.values())
        out[name] = {"present": True, "keys": len(keys), "bitwise_equal_keys": sum(per_key.values()), "bitwise_equal": all(per_key.values()),
                     "physics_keys_bitwise": not physics_differing, "physics_differing_keys": physics_differing[:20],
                     "diagnostic_differing_keys_max_rel": diagnostic_differing, "file_sha256_equal": _sha256(pa) == _sha256(pb)}
    summary_a, summary_b = json.loads((a / "summary.json").read_text(encoding="utf-8")), json.loads((b / "summary.json").read_text(encoding="utf-8"))
    out["final_counts_equal"] = summary_a.get("final_counts") == summary_b.get("final_counts")
    out["steps_completed"] = {"a": summary_a.get("steps_completed"), "b": summary_b.get("steps_completed")}
    out["ms_per_step"] = {"a": summary_a.get("ms_per_step_this_session"), "b": summary_b.get("ms_per_step_this_session")}
    out["window_currents_equal"] = summary_a.get("window_currents_a") == summary_b.get("window_currents_a")
    physics_ok = physics_ok and out["final_counts_equal"] and out["steps_completed"]["a"] == out["steps_completed"]["b"] and out["window_currents_equal"]
    out["all_bitwise"] = bool(out["series_records"]["bitwise_equal"] and all(v.get("bitwise_equal", True) for k, v in out.items() if isinstance(v, dict) and k.endswith(".npz"))
                              and out["final_counts_equal"] and out["steps_completed"]["a"] == out["steps_completed"]["b"])
    out["physics_bitwise"] = bool(physics_ok)
    out["diagnostics_within_rtol"] = bool(diagnostics_ok)
    out["diagnostic_rtol"] = DIAGNOSTIC_RTOL
    out["passed"] = bool(physics_ok and diagnostics_ok)
    return out


def command_mps_replay(args: argparse.Namespace) -> int:
    """Two same-seed processes concurrently under CUDA MPS (+ one solo afterwards): every physics record must replay bitwise."""

    root = Path(args.results_root) if args.results_root else HERE / "results-mps-replay"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"src{os.pathsep}.")
    mps_pipe = env.get("CUDA_MPS_PIPE_DIRECTORY")
    record: dict[str, Any] = {"schema_version": MPS_REPLAY_SCHEMA, "utc": utc_now(), "git_head": _runner().git_head(), "non_evidentiary": True,
                              "design_id": args.design, "option": option_tag(args.domain, args.grid), "steps": args.steps, "concurrent_processes": args.processes,
                              "host": socket.gethostname(), "cuda_mps_pipe_directory": mps_pipe, "mps_pipe_present": bool(mps_pipe and Path(mps_pipe).exists()),
                              "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "shrunk_cadences": SHRUNK_CADENCES,
                              "gpu": _gpu_inventory()}
    names = [chr(ord("a") + i) for i in range(args.processes)]
    procs = []
    t0 = time.perf_counter()
    for name in names:
        results = root / f"concurrent-{name}"
        log = open(root / f"concurrent-{name}.log", "wb")
        procs.append((name, results, subprocess.Popen(_replay_child_command(args.design, args.domain, args.grid, results, args.steps), cwd=str(MODERN), env=env,
                                                     stdout=log, stderr=subprocess.STDOUT), log))
    exit_codes = {}
    for name, _, proc, log in procs:
        exit_codes[name] = proc.wait()
        log.close()
    record["concurrent"] = {"exit_codes": exit_codes, "wall_seconds": time.perf_counter() - t0}
    if any(code != 0 for code in exit_codes.values()):
        record["verdict"] = "FAILED (a replay process did not exit cleanly)"
        _write_json(Path(args.output) if args.output else MPS_REPLAY_PATH, record)
        print(f"[mps-replay] {record['verdict']}: {exit_codes}", file=sys.stderr)
        return 1
    pairs = {}
    for (name_a, res_a, _, _), (name_b, res_b, _, _) in zip(procs, procs[1:]):
        pairs[f"{name_a}-vs-{name_b}"] = _compare_runs(res_a, res_b)
    record["concurrent_pairs"] = pairs
    solo: dict[str, Any] | None = None
    if not args.skip_solo:
        solo = {"runs": {}, "exit_codes": {}}
        solo_dirs = []
        for index in range(max(1, args.solo_runs)):
            results = root / f"solo-{index + 1}"
            t1 = time.perf_counter()
            with open(root / f"solo-{index + 1}.log", "wb") as log:
                code = subprocess.run(_replay_child_command(args.design, args.domain, args.grid, results, args.steps), cwd=str(MODERN), env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            solo["exit_codes"][f"solo-{index + 1}"] = code
            solo["runs"][f"solo-{index + 1}"] = {"wall_seconds": time.perf_counter() - t1}
            if code == 0:
                solo_dirs.append(results)
        if solo_dirs:
            solo["solo-1_vs_concurrent_a"] = _compare_runs(solo_dirs[0], procs[0][1])
        if len(solo_dirs) >= 2:
            solo["solo-1_vs_solo-2"] = _compare_runs(solo_dirs[0], solo_dirs[1])
        record["solo"] = solo
    comparisons = list(pairs.values()) + [v for k, v in (solo or {}).items() if k.endswith(("_vs_concurrent_a", "_vs_solo-2"))]
    physics_bitwise = all(c["physics_bitwise"] for c in comparisons) and bool(comparisons) and (solo is None or all(code == 0 for code in solo["exit_codes"].values()))
    diagnostics_ok = all(c["diagnostics_within_rtol"] for c in comparisons)
    all_bitwise = all(c["all_bitwise"] for c in comparisons)
    # MPS is neutral when the concurrent pairs behave like the MPS-free solo pair: physics bitwise in both, and either both fully
    # bitwise or both differing only in the float-atomic diagnostics (the same status with and without concurrency)
    solo_pair = (solo or {}).get("solo-1_vs_solo-2")
    mps_neutral = None if solo_pair is None else bool(physics_bitwise and solo_pair["physics_bitwise"] and
                                                       solo_pair["all_bitwise"] == all(p["all_bitwise"] for p in pairs.values()))
    passed = physics_bitwise and diagnostics_ok
    record.update({
        "all_bitwise": all_bitwise, "physics_bitwise": physics_bitwise, "diagnostics_within_rtol": diagnostics_ok, "diagnostic_rtol": DIAGNOSTIC_RTOL,
        "diagnostic_keys_note": "float-atomic DIAGNOSTIC accumulators (window velocity moments -> T_e maps / peak-node statistics; energy-momentum ledger interval sums) are "
                                "order-dependent device atomics and differ at round-off between ANY two runs on one GPU, solo or concurrent; the physics state (particles, "
                                "fixed-point charge deposition, potential, densities, ionisation, wall fluxes, currents, counts, neutral inventory) must be bitwise",
        "mps_neutral": mps_neutral, "passed": passed,
    })
    if all_bitwise:
        record["verdict"] = "BITWISE: every concurrent process under MPS and the solo processes produced identical records - MPS does not change a process's own kernel order"
    elif passed:
        record["verdict"] = ("PHYSICS BITWISE under MPS: particle state, deposited fields / densities / fluxes, currents, counts and the neutral inventory replay bitwise between "
                             "the concurrent MPS processes and the solo processes; only the float-atomic diagnostic accumulators differ, at round-off (<= "
                             f"{max((max(c['series_records'].get('differing_keys_max_rel', {}).values(), default=0.0) for c in comparisons), default=0.0):.2e} relative), "
                             f"{'with the SAME key set solo-vs-solo (MPS-neutral)' if mps_neutral else 'solo-vs-solo comparison not available'} - MPS does not change a process's own kernel order")
    else:
        record["verdict"] = "FAILED: a physics record differs between replays (or a diagnostic exceeds the tolerance) - see the pair records"
    _write_json(Path(args.output) if args.output else MPS_REPLAY_PATH, record)
    print(f"[mps-replay] {record['verdict']} (steps {args.steps}, concurrent ms/step {[p['ms_per_step'] for p in pairs.values()]}, "
          f"solo ms/step {[(k, v.get('ms_per_step')) for k, v in (solo or {}).items() if k.endswith('_vs_concurrent_a')]})")
    return 0 if passed else 1


def _gpu_inventory() -> list[str] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
    except Exception:  # noqa: BLE001
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else None


# -- shakedown --------------------------------------------------------------------------------------------------------------------


def command_shakedown(args: argparse.Namespace) -> int:
    """One design, real field, shrunk cadences: run -> (runner finalization) -> re-finalize path -> assess -> targets; NON-EVIDENTIARY."""

    runner = _runner()
    protocol, mapping, field_map = compose_run_protocol(args.design, args.domain, args.grid, "base")
    p = shrunk_protocol(protocol, "shakedown")
    results = HERE / f"results-shakedown-{option_tag(args.domain, args.grid)}"
    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True)
    protocol_path = results / "protocol-shakedown.json"
    protocol_path.write_bytes(protocol_bytes(p))
    t0 = time.perf_counter()
    summary_path = runner.run_steady_state(p, results, backend=args.backend, field_map=field_map, max_steps=args.max_steps, protocol_path=protocol_path)
    run_seconds = time.perf_counter() - t0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # assessment and closure extraction on the runner's own window-average artifacts (the production path)
    assessment = assess_run(p, results, mapping=mapping)
    targets = extract_run_targets(args.design, args.domain, results, mapping=mapping)
    assessment_sha, targets_sha = _sha256(results / "assessment.json"), _sha256(results / "closure-targets.json")
    # LAST: the externally-stopped path (a killed process): re-finalize from the checkpoint - this DOWNGRADES the maps to
    # instantaneous checkpoint maps by design, so it runs after assess / targets on a scratch directory only
    t1 = time.perf_counter()
    runner.finalize(p, results, backend=args.backend, field_map=field_map, stop_reason="shakedown_refinalize", protocol_path=protocol_path, allow_refinalize=True)
    refinalize_seconds = time.perf_counter() - t1
    refinalized = json.loads(summary_path.read_text(encoding="utf-8"))
    status_lines = runner._read_jsonl(results / "status.jsonl")
    samples = [s for s in status_lines if "event" not in s]
    windows = [s["peak_node"]["window"] for s in samples if (s.get("peak_node") or {}).get("window") is not None]
    enforced = [w for w in windows if w.get("gate_enforced")]
    triads = [s["grid_heating_triad"] for s in samples if s.get("grid_heating_triad") is not None]
    complete = [t for t in triads if t.get("windowed_energy_residual_window_complete")]
    record = {
        "schema_version": SHAKEDOWN_SCHEMA, "utc": utc_now(), "git_head": runner.git_head(), "non_evidentiary": True, "design_id": args.design,
        "option": option_tag(args.domain, args.grid), "host": socket.gethostname(), "gpu": _gpu_inventory(), "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "overrides": {**SHRUNK_CADENCES, "max_steps": args.max_steps}, "results_dir": results.name, "run_seconds": run_seconds,
        "steps_completed": summary["steps_completed"], "stop_reason": summary["stop_reason"], "ms_per_step": summary["ms_per_step_this_session"],
        "final_counts": summary["final_counts"], "frames": summary["artifacts"]["frames"]["count"] if summary["artifacts"].get("frames") else 0,
        "field": {"sha256": field_map.sha256, "source_sha256": field_map.source_sha256, "max_b_t": field_map.max_b_t, "kind": field_map.provenance.get("kind")},
        "protocol": {"case_id": p["case"]["id"], "cells": [p["case"]["radial_cells"], p["case"]["axial_cells"]], "macro_weight": p["case"]["macro_weight"], "dt_s": p["numerics"]["dt_s"],
                     "wall_budget_seconds": protocol["stopping_rule"]["wall_budget_seconds"], "model_version": p.get("model_version")},
        "peak_debye_window": {"records": len(windows), "enforced_records": len(enforced), "last": windows[-1] if windows else None,
                              "max_cells_per_debye_enforced": max((w["cells_per_debye"] for w in enforced), default=None)},
        "windowed_residual": {"records_with_complete_window": len(complete),
                              "last": None if not complete else {k: complete[-1][k] for k in complete[-1] if k.startswith("windowed") or k == "energy_residual_over_electrode_work"}},
        "plateau_keys": sorted(summary["plateau"]) if summary.get("plateau") else None,
        "summary_keys_present": {key: (summary.get(key) is not None) for key in ("plateau", "grid_heating_triad", "peak_node_debye", "neutral_inventory", "ledger", "window_currents_a")},
        "refinalize": {"seconds": refinalize_seconds, "stop_reason_after": refinalized.get("stop_reason"), "maps_kind_after": refinalized.get("maps_kind"),
                       "maps_downgraded_to_instantaneous_as_designed": refinalized.get("maps_kind") == "instantaneous_checkpoint",
                       "ran_after_assess_and_targets": True, "assessment_sha256_before": assessment_sha, "closure_targets_sha256_before": targets_sha},
        "assessment": {k: assessment[k] for k in ("verdict", "a_plateau", "b_residual_power", "steady_state_v4_verdict")},
        "targets": {"cusps": [{"z_c_m": c["z_c_m"], "electron_wall_current_a": c["electron_wall_current_a"], "ion_wall_current_a": c["ion_wall_current_a"],
                               "sheath_drop_v": c["sheath_drop_v"], "leak_width_fwhm_m": c["leak_width_fwhm_m"]} for c in targets["cusps"]],
                    "cells": [{"cell_id": c["cell_id"], "ionisation_share": c["ionisation_share"], "ion_wall_loss_fraction": c["ion_wall_loss_fraction"]} for c in targets["cells"]],
                    "kornfeld_chain": targets["kornfeld_chain"], "anode_edge_electron_wall_current_a": targets["anode_edge_electron_wall_current_a"],
                    "diffuse_non_cusp_electron_wall_current_a": targets["diffuse_non_cusp_electron_wall_current_a"]},
        "artifacts": {k: summary["artifacts"][k] for k in ("maps_npz_sha256", "series_npz_sha256")},
        "gate_not_inert_check": {"peak_window_enforced_at_least_once": bool(enforced), "peak_window_resolved_nodes_last": windows[-1].get("resolved_nodes") if windows else None,
                                 "residual_window_completed_at_least_once": bool(complete)},
    }
    out = Path(args.output) if args.output else shakedown_path(args.domain, args.grid)
    _write_json(out, record)
    print(f"[shakedown] {args.design}: {summary['steps_completed']} steps, {summary['stop_reason']}, {summary['ms_per_step_this_session']:.2f} ms/step, {record['frames']} frames, "
          f"peak window enforced in {len(enforced)}/{len(windows)} records (max {record['peak_debye_window']['max_cells_per_debye_enforced']}), residual window complete in "
          f"{len(complete)} records; refinalize ok; assessment {assessment['verdict']}; targets extracted; written {out}")
    return 0


# -- preregistered launch -----------------------------------------------------------------------------------------------------------


def worktree_status(cwd: Path = REPOSITORY_ROOT) -> list[str]:
    return [line for line in git("status", "--porcelain", "--untracked-files=normal", cwd=cwd).splitlines() if line.strip()]


def acquire_lock(results: Path, payload: dict[str, Any]) -> Path:
    """O_EXCL canonical lock in the results directory; refuses to overwrite (same-attempt / different-attempt classified)."""

    from cft_revival.orbit_mc.artifacts import canonical_bytes
    from cft_revival.pic2d.models import PIC2DValidationError

    results.mkdir(parents=True, exist_ok=True)
    path = results / LOCK_NAME
    data = canonical_bytes(payload) + b"\n"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        same = all(existing.get(k) == payload.get(k) for k in ("experiment_id", "design_id", "commit", "protocol_sha256"))
        raise PIC2DValidationError(f"execution lock already exists at {path} ({'same-attempt' if same else 'different-attempt'}: commit {existing.get('commit', '?')[:12]}, "
                                   f"acquired {existing.get('acquired_at_utc')}); refusing to launch")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _blob_matches_head(path: Path) -> bool:
    relative = path.relative_to(REPOSITORY_ROOT).as_posix()
    return git("rev-parse", f"HEAD:{relative}") == git("hash-object", "--", str(path))


def launch(design_id: str, domain: str, grid: str, *, case: str = "base", backend: str = "warp-cuda", expect_commit: str | None = None, resume: bool = False,
           allow_dirty: bool = False, require_mps: bool = False, wall_budget_seconds: float | None = None, log=lambda text: print(text, flush=True)) -> Path:
    """Preregistered execution of one design: clean worktree, expected commit, sealed protocol, exclusive lock, then the shared runner (blocking)."""

    from cft_revival.pic2d import artifacts
    from cft_revival.pic2d.models import PIC2DValidationError

    runner = _runner()
    head = git("rev-parse", "HEAD")
    if expect_commit is not None and not head.startswith(expect_commit):
        raise PIC2DValidationError(f"HEAD {head[:12]} is not the preregistration commit {expect_commit}")
    dirty = worktree_status()
    if dirty and not allow_dirty:
        raise PIC2DValidationError(f"worktree is not clean ({len(dirty)} entries, e.g. {dirty[0]!r}); the preregistered launch requires a clean checkout")
    if (domain, grid) != protocol_module.PREREGISTERED_OPTION:
        raise PIC2DValidationError(f"only the preregistered option {option_tag(*protocol_module.PREREGISTERED_OPTION)} may be launched; {option_tag(domain, grid)} is a draft option (use `run --allow-launch`)")
    sealed_path = composed_protocol_path(design_id, domain, grid, case)
    experiment_protocol = protocol_module.DRAFT_PROTOCOL_PATH
    for path in (sealed_path, experiment_protocol):
        if not path.is_file():
            raise PIC2DValidationError(f"{path} is missing: the option is not sealed (run `compose` and commit)")
        if not _blob_matches_head(path):
            raise PIC2DValidationError(f"{path.name} on disk differs from the committed blob at HEAD")
    required = [preflight_module.preflight_path(domain, grid), shakedown_path(domain, grid), MPS_REPLAY_PATH]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise PIC2DValidationError(f"preregistration records missing: {missing} (preflight / shakedown / MPS replay must exist and be committed)")
    preflight_report = json.loads(required[0].read_text(encoding="utf-8"))
    if not preflight_report["all_passed"]:
        raise PIC2DValidationError("the whole-set preflight of the option did not pass every design")
    replay = json.loads(MPS_REPLAY_PATH.read_text(encoding="utf-8"))
    if not replay.get("passed"):
        raise PIC2DValidationError("mps-replay.json does not record a passed replay (physics bitwise under MPS); refusing a shared-GPU launch")
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if require_mps and not (mps_pipe and Path(mps_pipe).exists()):
        raise PIC2DValidationError(f"--require-mps: CUDA_MPS_PIPE_DIRECTORY {mps_pipe!r} is not set or does not exist in this environment")
    # the sealed protocol must be what this checkout composes on THIS platform (field-derived dt policy included)
    protocol, mapping, field_map = compose_run_protocol(design_id, domain, grid, case)
    recomposed = protocol_bytes(protocol)
    sealed_bytes = sealed_path.read_bytes()
    if recomposed != sealed_bytes:
        raise PIC2DValidationError(f"the recomposed protocol of {design_id} differs from the sealed {sealed_path.name} (code, template, binding or platform drift); refusing to launch")
    protocol_sha = hashlib.sha256(sealed_bytes).hexdigest()
    results = results_dir(design_id, domain, grid, case)
    payload = {
        "schema_version": LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "design_id": design_id, "option": option_tag(domain, grid), "case": case, "commit": head,
        "protocol_sha256": protocol_sha, "sealed_protocol": sealed_path.relative_to(REPOSITORY_ROOT).as_posix(), "experiment_protocol_sha256": _sha256(experiment_protocol),
        "config_sha256": artifacts.config_identity(runner.build_config(protocol, backend=backend)), "field_source_sha256": field_map.source_sha256, "field_map_sha256": field_map.sha256,
        "backend": backend, "command": " ".join(sys.argv), "host": socket.gethostname(), "pid": os.getpid(), "acquired_at_utc": utc_now(),
        "clean_worktree_attested": not dirty, "worktree": str(REPOSITORY_ROOT), "immutable": True,
        "gpu": _gpu_inventory(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_mps_pipe_directory": mps_pipe, "mps_required": require_mps,
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
        log(f"[launch] {design_id}: resuming under the existing lock (commit {head[:12]}, acquired {existing.get('acquired_at_utc')})")
    else:
        if runner.find_checkpoint(results) is not None:
            raise PIC2DValidationError(f"{results} already holds a checkpoint; use --resume for a new session under the same lock")
        acquire_lock(results, payload)
        log(f"[launch] {design_id}: execution lock acquired: commit {head[:12]}, protocol {protocol_sha[:12]}, clean worktree {not dirty}, MPS pipe {mps_pipe}")
    protocol_path = results / "protocol.json"
    if not protocol_path.is_file():
        protocol_path.write_bytes(sealed_bytes)
    elif protocol_path.read_bytes() != sealed_bytes:
        raise PIC2DValidationError("results/protocol.json differs from the sealed protocol")
    return runner.run_steady_state(protocol, results, backend=backend, field_map=field_map, protocol_path=protocol_path, wall_budget_seconds=wall_budget_seconds, log=log)


def command_launch(args: argparse.Namespace) -> int:
    launch(args.design, args.domain, args.grid, case=args.case, backend=args.backend, expect_commit=args.expect_commit, resume=args.resume, allow_dirty=args.allow_dirty,
           require_mps=args.require_mps, wall_budget_seconds=args.wall_budget_seconds)
    return 0


# -- finalize / status / assess / targets -----------------------------------------------------------------------------------------------


def _run_protocol(design_id: str, domain: str, grid: str, case: str, results: Path) -> dict[str, Any]:
    """The protocol a results directory ran under (its own protocol.json), else the composed one."""

    path = results / "protocol.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    protocol, _ = build_protocol(design_id, domain, grid=grid, case=case)
    return protocol


def command_finalize(args: argparse.Namespace) -> int:
    protocol, _, field_map = compose_run_protocol(args.design, args.domain, args.grid, args.case)
    results = Path(args.results_dir) if args.results_dir else results_dir(args.design, args.domain, args.grid, args.case)
    on_disk = _run_protocol(args.design, args.domain, args.grid, args.case, results)
    _runner().finalize(on_disk if on_disk else protocol, results, backend=args.backend, field_map=field_map, stop_reason=args.stop_reason, protocol_path=results / "protocol.json",
                       allow_refinalize=args.allow_refinalize, recover_runner_stop=args.recover_runner_stop)
    return 0


def command_status(args: argparse.Namespace) -> int:
    results = Path(args.results_dir) if args.results_dir else results_dir(args.design, args.domain, args.grid, args.case)
    _print_json(_runner().status(results, _run_protocol(args.design, args.domain, args.grid, args.case, results)))
    return 0


def steady_state_v4_verdict() -> dict[str, Any]:
    """The reference's 50 -> 33 um convergence verdict (steady-state v4 assessment.json) if it exists in this checkout, else 'pending'."""

    if STEADY_STATE_V4_ASSESSMENT.is_file():
        try:
            record = json.loads(STEADY_STATE_V4_ASSESSMENT.read_text(encoding="utf-8"))
            return {"status": "available", "verdict": record.get("verdict"), "path": STEADY_STATE_V4_ASSESSMENT.relative_to(REPOSITORY_ROOT).as_posix(),
                    "utc": record.get("utc"), "c_convergence_all_within": (record.get("c_convergence") or {}).get("all_within")}
        except (OSError, json.JSONDecodeError):
            pass
    return {"status": "pending", "verdict": None, "note": "pic2d_cft_steady_state_v4 (392129e5) has not recorded its assessment in this checkout; the caveat applies in full"}


def assess_run(protocol: dict[str, Any], results: Path, *, mapping=None, log=lambda text: print(text, flush=True)) -> dict[str, Any]:
    """Predeclared per-design acceptance (protocol.stopping_rule.acceptance): (a) plateau, (b) windowed residual power, verdict, v4 caveat."""

    import numpy as np

    from cft_revival.pic2d import artifacts
    from cft_revival.pic2d.models import PIC2DValidationError

    if not (results / "summary.json").is_file():
        raise PIC2DValidationError(f"{results} has no summary.json to assess")
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    acceptance = protocol["stopping_rule"].get("acceptance") or {}
    maps = artifacts.read_npz(results / "maps.npz")
    n = np.asarray(maps["n_e_per_m3"])
    t = np.asarray(maps["t_e_ev"])
    i, j = np.unravel_index(int(np.nanargmax(n)), n.shape)
    triad = summary.get("grid_heating_triad") or {}
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
    inventory = summary.get("neutral_inventory") or {}
    currents = summary.get("window_currents_a") or {}
    run = {
        "stop_reason": summary["stop_reason"], "ion_transit_times": summary.get("ion_transit_times"), "steps_completed": summary["steps_completed"], "plateau": summary.get("plateau"),
        "discharge_current_a": currents.get("discharge_a"), "exit_ion_beam_a": currents.get("exit_ion_beam_a"),
        "ionization_rate_per_s": inventory.get("trailing_20pct_mean_ionization_rate_per_s"), "gross_utilisation": inventory.get("propellant_utilisation_trailing"),
        "net_utilisation": inventory.get("net_utilisation_trailing"), "neutral_density_per_m3": inventory.get("trailing_20pct_mean_density_per_m3"),
        "peak_n_e_window_per_m3": float(n[i, j]), "t_e_peak_window_ev": float(t[i, j]), "peak_node": [int(i), int(j)],
        "windowed_residual_over_electrode_work": triad.get("windowed_energy_residual_over_electrode_work"), "windowed_residual_window_complete": triad.get("windowed_energy_residual_window_complete"),
        "cumulative_residual_over_electrode_work": triad.get("energy_residual_over_electrode_work"),
        "cells_per_debye_window_last": debye.get("cells_per_debye_window_last"), "cells_per_debye_window_trailing_mean": debye.get("trailing_20pct_mean_cells_per_debye_window"),
        "peak_debye_soft_ok": debye.get("soft_ok"), "sessions": len(summary.get("sessions") or []), "git_head": summary.get("git_head"), "protocol_sha256": summary.get("protocol_sha256"),
        "config_sha256": (summary.get("provenance") or {}).get("config_sha256"), "runtime": (summary.get("provenance") or {}).get("runtime"),
    }
    a_plateau = run["stop_reason"] == "plateau_reached_after_min_transit_times"
    windowed = run["windowed_residual_over_electrode_work"]
    b_ok = windowed is not None and bool(run["windowed_residual_window_complete"]) and windowed < 0.02
    if a_plateau and b_ok:
        verdict = "closure_quotable"
    elif a_plateau:
        verdict = "plateau_with_heating"
    else:
        verdict = "no_plateau"
    v4 = steady_state_v4_verdict()
    record = {
        "schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "design_id": protocol.get("design_id"), "results_dir": results.name,
        "git_head_now": _runner().git_head(), "run": run,
        "a_plateau": {"passed": a_plateau, "stop_reason": run["stop_reason"], "ion_transit_times": run["ion_transit_times"], "plateau": run["plateau"], "rule": acceptance.get("a_plateau")},
        "b_residual_power": {"passed": b_ok, "windowed_residual_over_electrode_work": windowed, "window_complete": run["windowed_residual_window_complete"], "bound": 0.02, "one_sided": True,
                             "cumulative_witness": run["cumulative_residual_over_electrode_work"], "rule": acceptance.get("b_residual_power")},
        "verdict": verdict, "verdict_rule": (acceptance.get("d_verdicts") or {}).get(verdict),
        "closure_targets_quotable": verdict == "closure_quotable",
        "peak_debye_window": {"cells_per_debye_last": run["cells_per_debye_window_last"], "trailing_mean": run["cells_per_debye_window_trailing_mean"], "soft_ok": run["peak_debye_soft_ok"]},
        "steady_state_v4_verdict": v4,
        "convergence_caveat": acceptance.get("f_convergence_caveat", protocol_module.CONVERGENCE_CAVEAT),
        "convergence_statement": {
            "converged": "the 33 um values of this design carry the v4 tolerances as their grid band (10 % I_d / S / n_g / utilisation / I_beam, 20 % peak n_e / T_e,peak)",
            "resolution_limited": "the 33 um values of this design are the resolved numbers but carry NO grid band of their own (the 33 um grid is not itself certified)",
            "refinement_heating": "the reference grid is not certified: quotability rests on this design's own residual-power and peak-Debye readings; design comparisons are 'at 33 um, uncertified'",
            "no_plateau": "the reference grid is not certified: quotability rests on this design's own residual-power and peak-Debye readings; design comparisons are 'at 33 um, uncertified'",
            None: "PENDING: the steady-state v4 verdict is not yet recorded; this assessment must be re-read once it is (the caveat applies in full)",
        }[v4.get("verdict")],
        "design_specific": acceptance.get("g_design_specific"),
        "claim_boundary": protocol.get("claim_boundary"),
    }
    artifacts.write_canonical_json(results / "assessment.json", record)
    log(f"[assess] {results.name}: verdict {verdict} (a {a_plateau}, b {b_ok} [{windowed}]); v4 verdict {v4.get('verdict') or 'pending'}")
    return record


def command_assess(args: argparse.Namespace) -> int:
    results = Path(args.results_dir) if args.results_dir else results_dir(args.design, args.domain, args.grid, args.case)
    protocol = _run_protocol(args.design, args.domain, args.grid, args.case, results)
    assess_run(protocol, results)
    return 0


def extract_run_targets(design_id: str, domain: str, results: Path, *, mapping=None, grid: str = "33um") -> dict[str, Any]:
    maps_path, summary_path = results / "maps.npz", results / "summary.json"
    if not maps_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"no finished run under {results}")
    if mapping is None:
        built = design_module.build_design(design_id)
        target_cell_m, _ = GRID_VARIANTS[grid]
        mapping = design_module.pic_geometry(built, domain) if target_cell_m is None else design_module.pic_geometry(built, domain, target_cell_m=target_cell_m)
    binding = field_module.load_binding(design_id)
    topology = binding.get("topology_under_iron")
    if topology is not None:
        cusps = [c["z_c_m"] for c in topology["wall_cusps"]]
        cells = topology["cells"]
    else:
        entry = design_module.catalogue_entry(design_id)
        cusps = [c["z_c_m"] for c in entry["wall_cusps"]]
        cells = entry["cells"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    targets = extract_targets(load_maps(maps_path), mapping, cusps, cells, window_currents=summary.get("window_currents_a"))
    targets["design_id"] = design_id
    targets["cusp_source"] = "binding.topology_under_iron (material-aware level-0 P2)" if topology is not None else "catalogue entry (P2 level-1 v3.1)"
    targets["stop_reason"] = summary.get("stop_reason")
    out = results / "closure-targets.json"
    out.write_bytes(json.dumps(targets, indent=1, sort_keys=True, allow_nan=True, default=_plain).encode("utf-8") + b"\n")
    return targets


def command_targets(args: argparse.Namespace) -> int:
    results = Path(args.results_dir) if args.results_dir else results_dir(args.design, args.domain, args.grid, args.case)
    try:
        extract_run_targets(args.design, args.domain, results, grid=args.grid)
    except FileNotFoundError as error:
        print(f"[targets] {error}", file=sys.stderr)
        return 1
    print(f"[targets] wrote {results / 'closure-targets.json'}")
    return 0


# -- CLI ----------------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, *, design: bool | None = None, domain: bool = False, grid: bool = False, case: bool = False, results: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        if design is not None:
            p.add_argument("--design", required=design, default=None)
        if domain:
            p.add_argument("--domain", default="channel", choices=design_module.DOMAIN_OPTIONS)
        if grid:
            p.add_argument("--grid", default="50um", choices=tuple(GRID_VARIANTS), help="50um = draft template grid; 33um = the preregistered channel-33um option (v4 grid / dt / W parity)")
        if case:
            p.add_argument("--case", default="base", choices=CASES)
        if results:
            p.add_argument("--results-dir", default=None, help="override the results directory (development / replay only)")
        p.set_defaults(fn=fn)
        return p

    add("fields", command_fields, design=False)
    add("preflight", command_preflight, design=False, domain=True, grid=True)
    protocol_parser = add("protocol", command_protocol, design=True, domain=True, grid=True, case=True)
    protocol_parser.add_argument("--with-field", action="store_true", help="compose on the design's node field (dt policy, cathode placement) - what compose/launch use")
    add("compose", command_compose, domain=True, grid=True)
    add("draft-protocol", command_draft_protocol, domain=True, grid=True)
    add("experiment-protocol", command_draft_protocol, domain=True, grid=True)
    add("cost", command_cost)
    run_parser = add("run", command_run, design=True, domain=True, grid=True, case=True, results=True)
    run_parser.add_argument("--backend", default="warp-cuda")
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--wall-budget-seconds", type=float, default=None)
    run_parser.add_argument("--ignore-code-identity", action="store_true")
    run_parser.add_argument("--shrunk-cadences", action="store_true", help="shakedown / replay cadences (non-evidentiary)")
    run_parser.add_argument("--label", default=None)
    run_parser.add_argument("--allow-launch", action="store_true", help="required; development guard (labelled non-evidentiary runs only)")
    replay = add("mps-replay", command_mps_replay, design=True, domain=True, grid=True)
    replay.add_argument("--steps", type=int, default=MPS_REPLAY_STEPS)
    replay.add_argument("--processes", type=int, default=2)
    replay.add_argument("--skip-solo", action="store_true")
    replay.add_argument("--solo-runs", type=int, default=2, help="solo runs after the concurrent ones (2 -> a solo-vs-solo pair shows the MPS-free difference pattern)")
    replay.add_argument("--results-root", default=None)
    replay.add_argument("--output", default=None)
    shake = add("shakedown", command_shakedown, design=True, domain=True, grid=True)
    shake.add_argument("--backend", default="warp-cuda")
    shake.add_argument("--max-steps", type=int, default=SHAKEDOWN_MAX_STEPS)
    shake.add_argument("--output", default=None)
    la = add("launch", command_launch, design=True, domain=True, grid=True, case=True)
    la.add_argument("--backend", default="warp-cuda")
    la.add_argument("--expect-commit", default=None)
    la.add_argument("--resume", action="store_true")
    la.add_argument("--allow-dirty", action="store_true", help="development only; never for the preregistered execution")
    la.add_argument("--require-mps", action="store_true", help="refuse unless CUDA_MPS_PIPE_DIRECTORY is set and exists (the four-slot H100 configuration)")
    la.add_argument("--wall-budget-seconds", type=float, default=None)
    fin = add("finalize", command_finalize, design=True, domain=True, grid=True, case=True, results=True)
    fin.add_argument("--backend", default="warp-cuda")
    fin.add_argument("--stop-reason", default="finalized_from_checkpoint")
    fin.add_argument("--allow-refinalize", action="store_true")
    fin.add_argument("--recover-runner-stop", action="store_true")
    add("status", command_status, design=True, domain=True, grid=True, case=True, results=True)
    add("assess", command_assess, design=True, domain=True, grid=True, case=True, results=True)
    add("targets", command_targets, design=True, domain=True, grid=True, case=True, results=True)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
