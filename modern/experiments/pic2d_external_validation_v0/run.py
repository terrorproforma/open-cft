"""External validation v0 - runner stages (code-to-code vs Brandt 2016; preregistered option ``channel-20um`` for the Lambda H100).

From ``modern/`` (``$env:PYTHONPATH="$PWD\\src;$PWD"`` / ``PYTHONPATH=src:.``)::

    python -m experiments.pic2d_external_validation_v0.run reference                    # print the reference record (setup table, reported quantities, DOI)
    python -m experiments.pic2d_external_validation_v0.run fields [--no-sensitivity]    # CPU: P2 solve of the reconstruction (+ no-ring sensitivity) -> fields/<id>/binding.json
    python -m experiments.pic2d_external_validation_v0.run regate                       # recompute the field gates from the bound checkpoint (no solve)
    python -m experiments.pic2d_external_validation_v0.run protocol [--variant V] [--grid G] [--with-field]   # print a composed run protocol
    python -m experiments.pic2d_external_validation_v0.run comparison                   # write comparison-spec.json (validated)
    python -m experiments.pic2d_external_validation_v0.run compose                      # seal protocols/*.json (primary, sensitivity, 15 um) + protocol.json
    python -m experiments.pic2d_external_validation_v0.run preflight [--variant V --grid G] [--gpu-timing]   # whole-set preflight (+ the option's launch-box GPU timing) -> preflight-<option>.json
    python -m experiments.pic2d_external_validation_v0.run cost                         # cost table (20 / 33 / 15 um, plume box)
    python -m experiments.pic2d_external_validation_v0.run shakedown                    # GPU: shrunk-cadence run -> assess -> compare -> re-finalize -> shakedown-channel-20um.json
    python -m experiments.pic2d_external_validation_v0.run launch --expect-commit SHA [--require-mps] [--resume]   # PREREGISTERED execution (one)
    python -m experiments.pic2d_external_validation_v0.run run --allow-launch ...       # labelled development run through the shared runner (never evidence)
    python -m experiments.pic2d_external_validation_v0.run status|finalize|assess|compare [--results-dir DIR]

``launch`` and ``run`` step with the shared steady-state runner (``experiments.pic2d_cft_steady_state_v1.run``) under the composed protocol with the
reconstructed node field passed in directly; results go to ``results/<option>/``.  ``compare`` evaluates every channel-comparable row of the
comparison spec with the run's trailing-window quantities (S) and writes ``comparison.json`` next to the run's ``summary.json``.

Preregistration discipline of ``launch`` (the mini-sweep's, per option): HEAD == --expect-commit, clean worktree, the experiment ``protocol.json``
and the sealed ``protocols/<option>.json`` blobs equal HEAD's, the recomposed protocol equals the sealed file byte for byte, the whole-set
preflight (with a passed launch-box GPU timing) and the shakedown record present and passed, ``--require-mps`` -> the CUDA MPS pipe directory
must exist, O_EXCL ``execution-lock.json`` in the results directory (``--resume`` continues only under the same commit + protocol).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import comparison as comparison_module
from . import fields as field_module
from . import preflight as preflight_module
from . import protocol as protocol_module
from . import reference
from .protocol import (
    GRIDS,
    VARIANTS,
    build_protocol,
    compose_run_protocol,
    option_tag,
    protocol_bytes,
)

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
RESULTS_ROOT = HERE / "results"
ASSESSMENT_SCHEMA = "cft.pic2d.external-validation-v0.assessment/1.0.0"
COMPARISON_RESULT_SCHEMA = "cft.pic2d.external-validation-v0.comparison-result/1.0.0"
SHAKEDOWN_SCHEMA = "cft.pic2d.external-validation-v0.shakedown/1.0.0"
LOCK_NAME = "execution-lock.json"
LOCK_SCHEMA = "cft.pic2d.external-validation-v0.execution-lock/1.0.0"
SHRUNK_CADENCES = {"series_interval_steps": 200, "device_sync_steps": 200, "checkpoint_every_steps": 4000, "averaging_window_steps": 40000, "frame_cadence_steps": 2000,
                   "peak_debye_window_steps": 40000, "peak_debye_window_snapshot_steps": 4000, "residual_window_steps": 40000}
SHAKEDOWN_MAX_STEPS = 100000


def results_dir(variant: str, grid: str) -> Path:
    return RESULTS_ROOT / option_tag(variant, grid)


def shakedown_path(variant: str = protocol_module.PRIMARY_VARIANT, grid: str = protocol_module.PRIMARY_GRID) -> Path:
    return HERE / f"shakedown-{option_tag(variant, grid)}.json"


def git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def _gpu_inventory() -> list[str] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
    except Exception:  # noqa: BLE001
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=_plain).encode("utf-8") + b"\n")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=1, sort_keys=True, default=str))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runner():
    from experiments.pic2d_cft_steady_state_v1 import run as runner

    return runner


# -- composition ----------------------------------------------------------------------------------------------------------------


def command_reference(args: argparse.Namespace) -> int:
    _print_json(reference.reference_document())
    return 0


def command_fields(args: argparse.Namespace) -> int:
    binding = field_module.produce_field(with_sensitivity=not args.no_sensitivity)
    print(f"[fields] {'ok' if binding['gates']['all_passed'] else 'GATES FAILED'} -> {field_module.binding_path()}")
    return 0 if binding["gates"]["all_passed"] else 1


def command_regate(args: argparse.Namespace) -> int:
    _print_json(field_module.regate_field())
    return 0


def command_protocol(args: argparse.Namespace) -> int:
    if args.with_field:
        protocol, _, _ = compose_run_protocol(args.variant, args.grid)
    else:
        protocol, _ = build_protocol(args.variant, args.grid)
    _print_json(protocol)
    return 0


def command_comparison(args: argparse.Namespace) -> int:
    path = protocol_module.write_comparison_spec()
    print(f"[comparison] wrote {path} (sha256 {_sha256(path)[:12]})")
    return 0


def _preflight_summary() -> dict[str, Any] | None:
    path = preflight_module.preflight_path()
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {"all_passed": report["all_passed"], "launch_set_passed": report["launch_set_passed"], "generated_utc": report["generated_utc"], "platform": report["platform"],
            "options": {r["option"]: {"passed": r["passed"], "gates": {k: v["passed"] for k, v in r["gates"].items()},
                                      "dt_s": r["gates"].get("field_map", {}).get("dt_s"), "cells_per_debye_at_published": r["gates"].get("field_map", {}).get("cells_per_debye_at_published"),
                                      "macro_weight": r["gates"].get("protocol", {}).get("macro_weight"), "cells": r["gates"].get("protocol", {}).get("cells"),
                                      "wall_budget_hours": r["gates"].get("protocol", {}).get("wall_budget_hours"), "field_source_sha256": r["gates"].get("field_map", {}).get("field_source_sha256")}
                        for r in report["options"]}}


def _binding_summary() -> dict[str, Any] | None:
    try:
        binding = field_module.load_binding()
    except Exception:  # noqa: BLE001 - the summary is optional
        return None
    gates = binding["gates"]
    return {"all_passed": gates["all_passed"], "source_strength_scale": binding["source_strength_scale"], "implied_remanence_t": gates["G1_scale"]["implied_remanence_t"],
            "interior_nulls_m": gates["G2_interior_nulls"]["interior_nulls_m"], "exit_null_m": gates["G3_exit_null"]["nearest_null_m"], "b_at_17mm_t": gates["G4_exit_point"]["b_t"],
            "axis_max_t": gates["G5_axis_maximum"]["axis_max_t"], "axis_max_z_m": gates["G5_axis_maximum"]["axis_max_z_m"],
            "wall_cusp_b_t": [r["wall_b_max_within_0p5mm_t"] for r in gates["D6_wall_cusp_field"]["cusps"]],
            "low_field_contour_radius_m": [r["low_field_contour_radius_m"] for r in gates["D6_wall_cusp_field"]["cusps"]],
            "checkpoint_file_sha256": binding["map"]["checkpoint_file_sha256"], "p2_dofs": binding["solve"]["p2_dofs"], "solve_wall_seconds": binding["solve"]["solve_wall_seconds"],
            "no_ring_bracket": (binding.get("sensitivity_no_rings") or {}).get("bracket"), "genealogy_entries": len(gates.get("genealogy", []))}


def _shakedown_summary() -> dict[str, Any] | None:
    path = shakedown_path()
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    keys = ("utc", "git_head", "host", "option", "steps_completed", "stop_reason", "ms_per_step", "frames", "concurrent_mps_clients", "passed", "assessment", "comparison", "refinalize",
            "gate_not_inert_check", "non_evidentiary")
    return {k: record.get(k) for k in keys}


def command_compose(args: argparse.Namespace) -> int:
    protocol_module.write_comparison_spec()
    sealed = protocol_module.compose_all()
    out = protocol_module.write_experiment_protocol(preflight_summary=_preflight_summary(), sealed=sealed, field_binding_summary=_binding_summary(), shakedown_summary=_shakedown_summary())
    print(f"[compose] {len(sealed)} run protocols sealed under {protocol_module.PROTOCOLS_DIR}; experiment protocol -> {out} (sha256 {_sha256(out)[:12]})")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    timing = None
    variant, grid = getattr(args, "variant", protocol_module.PRIMARY_VARIANT), getattr(args, "grid", protocol_module.PRIMARY_GRID)
    if args.gpu_timing:
        timing = preflight_module.gpu_timing(variant, grid, backend=args.backend, timing_steps=args.timing_steps)
    # amendment 1: the record carrying an option's GPU timing is that option's file (the base keeps preflight-channel-20um.json, sealed at launch 1)
    path, report = preflight_module.write_preflight(gpu_timing_record=timing, path=preflight_module.preflight_path(variant, grid))
    print(f"[preflight] all_passed={report['all_passed']} (launch set {report['launch_set_passed']}) over {report['option_count']} options"
          + (f"; launch-box timing {'PASS' if timing['passed'] else 'FAIL'}: seed {timing['timing_seed_load']['ms_per_step']:.2f} / plateau load {timing['timing_plateau_load']['ms_per_step']:.2f} ms/step "
             f"with {timing['concurrent_mps_clients']} other MPS client(s)" if timing else "")
          + f" -> {path}")
    return 0 if report["all_passed"] and (timing is None or timing["passed"]) else 1


def command_cost(args: argparse.Namespace) -> int:
    document = protocol_module.experiment_protocol_document(sealed={})
    for name, row in document["cost_table"].items():
        print(f"{name:<22} cells {row['cells'][0]}x{row['cells'][1]:<5} dt {row['dt_s']*1e12:.2f} ps W {row['macro_weight']:<9.6g} N {row['particles_projected_m']:.1f} M  "
              f"MPS-4 {row['ms_per_step_h100_mps4_per_process']:.1f} ms/step (solo {row['ms_per_step_h100_solo_equivalent']:.1f}; 5090 model {row['ms_per_step_rtx5090_model']:.1f})  "
              f"{row['steps_to_3_transits']/1e6:.2f} M steps  {row['hours_to_3_transits_mps4']:.1f} h MPS-4 / {row['hours_to_3_transits_h100_solo_equivalent']:.1f} h solo  {row['device_gb_projected']:.1f} GB  "
              f"reference 76 us: {row['hours_to_reference_time_mps4']:.0f} h")
    for row in document["grid_argument"]["rows"]:
        print(f"grid {row['grid']:<5} cells/lambda_D {row['cells_per_debye_at_published']:.2f} (hard pi {row['admissible_hard_pi']}, soft 2.5 {row['soft_2p5_met']}); "
              f"hard density {row['hard_gate_density_at_10ev_per_m3']:.2e}, omega_pe dt gate density {row['omega_pe_dt_gate_density_per_m3']:.2e}, courant {row['electron_courant_400ev']:.2f}")
    return 0


# -- development run / launch guard ------------------------------------------------------------------------------------------------


def shrunk_protocol(protocol: dict[str, Any], label: str) -> dict[str, Any]:
    p = copy.deepcopy(protocol)
    num = p["numerics"]
    num["series_interval_steps"] = SHRUNK_CADENCES["series_interval_steps"]
    num["device_sync_steps"] = SHRUNK_CADENCES["device_sync_steps"]
    num["checkpoint_every_steps"] = SHRUNK_CADENCES["checkpoint_every_steps"]
    num["averaging_window_steps"] = SHRUNK_CADENCES["averaging_window_steps"]
    if num.get("frame_recorder") is not None:
        num["frame_recorder"] = {"cadence_steps": SHRUNK_CADENCES["frame_cadence_steps"], "precision": "float32"}
    gate = num.get("peak_debye_gate") or {}
    if gate.get("window_steps") is not None:
        gate["window_steps"] = SHRUNK_CADENCES["peak_debye_window_steps"]
        gate["window_snapshot_steps"] = SHRUNK_CADENCES["peak_debye_window_snapshot_steps"]
    triad = p["stopping_rule"].get("grid_heating_triad") or {}
    if triad.get("residual_window_steps") is not None:
        triad["residual_window_steps"] = SHRUNK_CADENCES["residual_window_steps"]
    p["status"] = f"{label}_non_evidentiary_shrunk_cadences"
    p["experiment_id"] = protocol["experiment_id"] + f"-{label}"
    return p


def command_run(args: argparse.Namespace) -> int:
    if not args.allow_launch:
        print("[run] REFUSED: `run` is the labelled development entry (never evidence); pass --allow-launch for a shakedown", file=sys.stderr)
        return 2
    path = preflight_module.preflight_path()
    if not path.is_file():
        print(f"[run] REFUSED: no whole-set preflight ({path}); run `preflight` first", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["all_passed"]:
        print("[run] REFUSED: the whole-set preflight did not pass every option", file=sys.stderr)
        return 2
    protocol, _, field_map = compose_run_protocol(args.variant, args.grid)
    if args.shrunk_cadences:
        protocol = shrunk_protocol(protocol, args.label or "development")
    results = Path(args.results_dir) if args.results_dir else results_dir(args.variant, args.grid)
    results.mkdir(parents=True, exist_ok=True)
    protocol_path = results / "protocol.json"
    protocol_path.write_bytes(protocol_bytes(protocol))
    _runner().run_steady_state(protocol, results, backend=args.backend, field_map=field_map, max_steps=args.max_steps, wall_budget_seconds=args.wall_budget_seconds,
                               require_same_code=not args.ignore_code_identity, protocol_path=protocol_path)
    return 0


# -- shakedown ---------------------------------------------------------------------------------------------------------------------------


def command_shakedown(args: argparse.Namespace) -> int:
    """The primary option on its real field at shrunk cadences: run -> assess -> compare -> re-finalize path; NON-EVIDENTIARY record."""

    runner = _runner()
    protocol, _mapping, field_map = compose_run_protocol(args.variant, args.grid)
    p = shrunk_protocol(protocol, "shakedown")
    results = HERE / f"results-shakedown-{option_tag(args.variant, args.grid)}"
    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True)
    protocol_path = results / "protocol-shakedown.json"
    protocol_path.write_bytes(protocol_bytes(p))
    clients_before = preflight_module.compute_apps()
    t0 = time.perf_counter()
    summary_path = runner.run_steady_state(p, results, backend=args.backend, field_map=field_map, max_steps=args.max_steps, protocol_path=protocol_path)
    run_seconds = time.perf_counter() - t0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assessment = assess_run(p, results)
    comparison_record = compare_run(results, p)
    assessment_sha, comparison_sha = _sha256(results / "assessment.json"), _sha256(results / "comparison.json")
    # LAST: the externally-stopped path (re-finalize from the checkpoint) - it DOWNGRADES the maps to instantaneous ones by design, so it runs after assess / compare
    t1 = time.perf_counter()
    runner.finalize(p, results, backend=args.backend, field_map=field_map, stop_reason="shakedown_refinalize", protocol_path=protocol_path, allow_refinalize=True)
    refinalize_seconds = time.perf_counter() - t1
    refinalized = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = [s for s in runner._read_jsonl(results / "status.jsonl") if "event" not in s]
    windows = [s["peak_node"]["window"] for s in samples if (s.get("peak_node") or {}).get("window") is not None]
    enforced = [w for w in windows if w.get("gate_enforced")]
    triads = [s["grid_heating_triad"] for s in samples if s.get("grid_heating_triad") is not None]
    complete = [t for t in triads if t.get("windowed_energy_residual_window_complete")]
    compared = [r for r in comparison_record["rows"] if r.get("compared")]
    own = os.getpid()
    others = [a for a in clients_before if a["pid"] != own and a["used_memory_mib"] > 200.0]
    record = {
        "schema_version": SHAKEDOWN_SCHEMA, "utc": utc_now(), "git_head": runner.git_head(), "non_evidentiary": True, "option": option_tag(args.variant, args.grid), "host": socket.gethostname(),
        "gpu": _gpu_inventory(), "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), "concurrent_mps_clients": len(others), "concurrent_mps_client_pids": [a["pid"] for a in others],
        "overrides": {**SHRUNK_CADENCES, "max_steps": args.max_steps}, "results_dir": results.name, "run_seconds": run_seconds,
        "steps_completed": summary["steps_completed"], "stop_reason": summary["stop_reason"], "ms_per_step": summary["ms_per_step_this_session"], "final_counts": summary["final_counts"],
        "frames": summary["artifacts"]["frames"]["count"] if summary["artifacts"].get("frames") else 0,
        "field": {"sha256": field_map.sha256, "source_sha256": field_map.source_sha256, "max_b_t": field_map.max_b_t, "kind": field_map.provenance.get("kind")},
        "protocol": {"case_id": p["case"]["id"], "cells": [p["case"]["radial_cells"], p["case"]["axial_cells"]], "macro_weight": p["case"]["macro_weight"], "dt_s": p["numerics"]["dt_s"],
                     "wall_budget_seconds": protocol["stopping_rule"]["wall_budget_seconds"], "model_version": p.get("model_version"), "static_neutrals": p["operating_point"].get("neutral_inventory") is None},
        "peak_debye_window": {"records": len(windows), "enforced_records": len(enforced), "last": windows[-1] if windows else None,
                              "max_cells_per_debye_enforced": max((w["cells_per_debye"] for w in enforced), default=None)},
        "windowed_residual": {"records_with_complete_window": len(complete),
                              "last": None if not complete else {k: complete[-1][k] for k in complete[-1] if k.startswith("windowed") or k == "energy_residual_over_electrode_work"}},
        "plateau_keys": sorted(summary["plateau"]) if summary.get("plateau") else None,
        "summary_keys_present": {key: (summary.get(key) is not None) for key in ("plateau", "grid_heating_triad", "peak_node_debye", "neutral_inventory", "ledger", "window_currents_a")},
        "assessment": {k: assessment[k] for k in ("verdict", "a_plateau", "b_residual_power")},
        "comparison": {"rows_compared": len(compared), "rows_not_compared": [{"quantity_id": r["quantity_id"], "reason": r.get("reason")} for r in comparison_record["rows"] if not r.get("compared")],
                       "verdicts": sorted({r["verdict"] for r in compared}), "s_values": comparison_record["s_values"], "quotable": comparison_record["quotable"],
                       "note": "non-evidentiary numbers of a 0.07 us transient: the stage is exercised, the values mean nothing"},
        "refinalize": {"seconds": refinalize_seconds, "stop_reason_after": refinalized.get("stop_reason"), "maps_kind_after": refinalized.get("maps_kind"),
                       "maps_downgraded_to_instantaneous_as_designed": refinalized.get("maps_kind") == "instantaneous_checkpoint", "ran_after_assess_and_compare": True,
                       "assessment_sha256_before": assessment_sha, "comparison_sha256_before": comparison_sha},
        "artifacts": {k: summary["artifacts"][k] for k in ("maps_npz_sha256", "series_npz_sha256")},
        "omega_pe_dt_gate": {"statistic": "v2.0.4 resolved-node single-step peak (nodes holding >= the peak-Debye floor of macro-electrons); raw single-node peak recorded alongside",
                             "max_resolved": max((s.get("peak_omega_pe_dt") or 0.0) for s in samples) if samples else None,
                             "max_raw": max((s.get("peak_omega_pe_dt_raw") or 0.0) for s in samples) if samples else None,
                             "limit": p["numerics"]["stability_limits"]["max_omega_pe_dt"]},
        "gate_not_inert_check": {
            "peak_window_computed_at_least_once": bool(windows) and any(bool(w.get("window_complete")) for w in windows),
            "peak_window_enforced_at_least_once": bool(enforced), "peak_window_resolved_nodes_last": windows[-1].get("resolved_nodes") if windows else None,
            "peak_window_enforcement_note": ("enforcement needs a node whose window-mean occupancy reaches the 32-macro-electron floor; at W 8.2e4 on 20 um nodes that happens once the "
                                            "local density passes ~1.4e18 m^-3 at mid radius (229 macro-electrons per mid-radius cell at the published 1e19), not in a 0.07 us "
                                            "shakedown from the 5e16 seed - the production run must show resolved_nodes > 0 (v2.0.2 lesson)"),
            "expected_live_at_published_density": float(p["case"]["macro_weight_policy"]["macro_electrons_per_cell_at_mid_radius_at_published_density"]) >= 32.0,
            "residual_window_completed_at_least_once": bool(complete),
        },
    }
    record["passed"] = bool(summary["stop_reason"] == "target_steps_reached" and summary["steps_completed"] == args.max_steps and record["gate_not_inert_check"]["peak_window_computed_at_least_once"]
                            and record["gate_not_inert_check"]["expected_live_at_published_density"] and complete and len(compared) >= 8
                            and record["refinalize"]["maps_downgraded_to_instantaneous_as_designed"])
    out = Path(args.output) if args.output else shakedown_path(args.variant, args.grid)
    _write_json(out, record)
    print(f"[shakedown] {record['option']}: {summary['steps_completed']} steps, {summary['stop_reason']}, {summary['ms_per_step_this_session']:.2f} ms/step with {len(others)} other MPS client(s), "
          f"{record['frames']} frames, peak window computed in {len(windows)} records / enforced in {len(enforced)} (resolved nodes last {record['gate_not_inert_check']['peak_window_resolved_nodes_last']}), "
          f"omega_pe dt resolved max {record['omega_pe_dt_gate']['max_resolved']} / raw max {record['omega_pe_dt_gate']['max_raw']}, residual window complete in {len(complete)} records; "
          f"assessment {assessment['verdict']}; {len(compared)} comparison rows formed; refinalize ok; {'PASS' if record['passed'] else 'FAIL'}; written {out}")
    return 0 if record["passed"] else 1


# -- preregistered launch -------------------------------------------------------------------------------------------------------------------


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
        same = all(existing.get(k) == payload.get(k) for k in ("experiment_id", "option", "commit", "protocol_sha256"))
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


def launch(variant: str, grid: str, *, expect_commit: str, backend: str = "warp-cuda", resume: bool = False, allow_dirty: bool = False, require_mps: bool = False,
           wall_budget_seconds: float | None = None, log=lambda text: print(text, flush=True)) -> Path:
    """Preregistered execution of the launch-set option: clean worktree, expected commit, sealed protocol + records, exclusive lock, then the shared runner (blocking)."""

    from cft_revival.pic2d import artifacts
    from cft_revival.pic2d.models import PIC2DValidationError

    runner = _runner()
    if not expect_commit or len(expect_commit) < 7:
        raise PIC2DValidationError("--expect-commit <preregistration sha> (>= 7 hex digits) is required for the preregistered launch")
    head = git("rev-parse", "HEAD")
    if not head.startswith(expect_commit):
        raise PIC2DValidationError(f"HEAD {head[:12]} is not the preregistration commit {expect_commit}")
    dirty = worktree_status()
    if dirty and not allow_dirty:
        raise PIC2DValidationError(f"worktree is not clean ({len(dirty)} entries, e.g. {dirty[0]!r}); the preregistered launch requires a clean checkout")
    if (variant, grid) not in protocol_module.LAUNCH_SET:
        raise PIC2DValidationError(f"only the launch set {[option_tag(*o) for o in protocol_module.LAUNCH_SET]} may be launched; {option_tag(variant, grid)} is sealed but not launched "
                                   "(use `run --allow-launch` for a labelled development run)")
    sealed_path = protocol_module.composed_protocol_path(variant, grid)
    experiment_protocol = protocol_module.EXPERIMENT_PROTOCOL_PATH
    for path in (sealed_path, experiment_protocol):
        if not path.is_file():
            raise PIC2DValidationError(f"{path} is missing: the option is not sealed (run `compose` and commit)")
        if not _blob_matches_head(path):
            raise PIC2DValidationError(f"{path.name} on disk differs from the committed blob at HEAD")
    experiment_document = json.loads(experiment_protocol.read_text(encoding="utf-8"))
    if not str(experiment_document.get("status", "")).startswith("preregistered"):
        raise PIC2DValidationError(f"protocol.json status {experiment_document.get('status')!r} is not a preregistration")
    required = [preflight_module.preflight_path(variant, grid), shakedown_path(variant, grid)]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise PIC2DValidationError(f"preregistration records missing: {missing} (the launch-box preflight and the shakedown must exist and be committed)")
    for path in required:
        if not _blob_matches_head(path):
            raise PIC2DValidationError(f"{path.name} on disk differs from the committed blob at HEAD")
    preflight_report = json.loads(required[0].read_text(encoding="utf-8"))
    if not (preflight_report["all_passed"] and preflight_report["launch_set_passed"]):
        raise PIC2DValidationError("the whole-set preflight did not pass every option")
    if not preflight_module.timing_passed(preflight_report):
        raise PIC2DValidationError("the preflight carries no passed launch-box GPU timing (preflight --gpu-timing on the launch box, >= 2000 steps, budget covering the measured 3-transit wall)")
    shakedown_record = json.loads(required[1].read_text(encoding="utf-8"))
    if not shakedown_record.get("passed"):
        raise PIC2DValidationError(f"{required[1].name} does not record a passed shakedown (run -> assess -> compare -> re-finalize)")
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if require_mps and not (mps_pipe and Path(mps_pipe).exists()):
        raise PIC2DValidationError(f"--require-mps: CUDA_MPS_PIPE_DIRECTORY {mps_pipe!r} is not set or does not exist in this environment")
    # the sealed protocol must be what this checkout composes on THIS platform (field-derived dt policy included)
    protocol, _mapping, field_map = compose_run_protocol(variant, grid)
    recomposed = protocol_bytes(protocol)
    sealed_bytes = sealed_path.read_bytes()
    if recomposed != sealed_bytes:
        raise PIC2DValidationError(f"the recomposed protocol differs from the sealed {sealed_path.name} (code, template, binding or platform drift); refusing to launch")
    protocol_sha = hashlib.sha256(sealed_bytes).hexdigest()
    results = results_dir(variant, grid)
    payload = {
        "schema_version": LOCK_SCHEMA, "experiment_id": protocol["experiment_id"], "option": option_tag(variant, grid), "variant": variant, "grid": grid, "commit": head,
        "protocol_sha256": protocol_sha, "sealed_protocol": sealed_path.relative_to(REPOSITORY_ROOT).as_posix(), "experiment_protocol_sha256": _sha256(experiment_protocol),
        "preflight_sha256": _sha256(required[0]), "shakedown_sha256": _sha256(required[1]),
        "config_sha256": artifacts.config_identity(runner.build_config(protocol, backend=backend)), "field_source_sha256": field_map.source_sha256, "field_map_sha256": field_map.sha256,
        "backend": backend, "command": " ".join(sys.argv), "host": socket.gethostname(), "pid": os.getpid(), "acquired_at_utc": utc_now(),
        "clean_worktree_attested": not dirty, "worktree": str(REPOSITORY_ROOT), "immutable": True,
        "gpu": _gpu_inventory(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_mps_pipe_directory": mps_pipe, "mps_required": require_mps,
        "concurrent_compute_apps_at_launch": preflight_module.compute_apps(),
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
        log(f"[launch] {option_tag(variant, grid)}: resuming under the existing lock (commit {head[:12]}, acquired {existing.get('acquired_at_utc')})")
    else:
        if runner.find_checkpoint(results) is not None:
            raise PIC2DValidationError(f"{results} already holds a checkpoint; use --resume for a new session under the same lock")
        acquire_lock(results, payload)
        log(f"[launch] {option_tag(variant, grid)}: execution lock acquired: commit {head[:12]}, protocol {protocol_sha[:12]}, clean worktree {not dirty}, MPS pipe {mps_pipe}, "
            f"{len(payload['concurrent_compute_apps_at_launch'])} compute app(s) on the GPU at launch")
    protocol_path = results / "protocol.json"
    if not protocol_path.is_file():
        protocol_path.write_bytes(sealed_bytes)
    elif protocol_path.read_bytes() != sealed_bytes:
        raise PIC2DValidationError("results/protocol.json differs from the sealed protocol")
    return runner.run_steady_state(protocol, results, backend=backend, field_map=field_map, protocol_path=protocol_path, wall_budget_seconds=wall_budget_seconds, log=log)


def command_launch(args: argparse.Namespace) -> int:
    from cft_revival.pic2d.models import PIC2DValidationError

    try:
        launch(args.variant, args.grid, expect_commit=args.expect_commit, backend=args.backend, resume=args.resume, allow_dirty=args.allow_dirty, require_mps=args.require_mps,
               wall_budget_seconds=args.wall_budget_seconds)
    except PIC2DValidationError as error:
        print(f"[launch] REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


# -- finalize / status ----------------------------------------------------------------------------------------------------------------------


def command_finalize(args: argparse.Namespace) -> int:
    protocol, _, field_map = compose_run_protocol(args.variant, args.grid)
    results = Path(args.results_dir) if args.results_dir else results_dir(args.variant, args.grid)
    on_disk = _run_protocol(results, args.variant, args.grid)
    _runner().finalize(on_disk if on_disk else protocol, results, backend=args.backend, field_map=field_map, stop_reason=args.stop_reason, protocol_path=results / "protocol.json",
                       allow_refinalize=args.allow_refinalize, recover_runner_stop=args.recover_runner_stop)
    return 0


def command_status(args: argparse.Namespace) -> int:
    results = Path(args.results_dir) if args.results_dir else results_dir(args.variant, args.grid)
    _print_json(_runner().status(results, _run_protocol(results, args.variant, args.grid)))
    return 0


# -- assessment and comparison ----------------------------------------------------------------------------------------------------------


def _run_protocol(results: Path, variant: str, grid: str) -> dict[str, Any]:
    path = results / "protocol.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    protocol, _ = build_protocol(variant, grid)
    return protocol


def assess_run(protocol: dict[str, Any], results: Path, *, log=lambda text: print(text, flush=True)) -> dict[str, Any]:
    """(a) plateau, (b) windowed residual power, the resolution flag, verdict (stopping_rule.acceptance)."""

    from cft_revival.pic2d import artifacts
    from cft_revival.pic2d.models import PIC2DValidationError

    if not (results / "summary.json").is_file():
        raise PIC2DValidationError(f"{results} has no summary.json to assess")
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    acceptance = protocol["stopping_rule"].get("acceptance") or {}
    triad = summary.get("grid_heating_triad") or {}
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
    plateau = summary.get("plateau") or {}
    a_plateau = summary["stop_reason"] == "plateau_reached_after_min_transit_times"
    windowed = triad.get("windowed_energy_residual_over_electrode_work")
    b_ok = windowed is not None and bool(triad.get("windowed_energy_residual_window_complete")) and windowed < 0.02
    drifts_ok = bool(plateau.get("drifts_within_threshold")) and (summary.get("ion_transit_times") or 0.0) >= 3.0
    soft_ok = debye.get("soft_ok")
    if a_plateau and b_ok:
        verdict = "comparison_quotable"
    elif (not a_plateau) and drifts_ok and soft_ok is False and b_ok and bool(plateau.get("triad_soft_ok", True)):
        verdict = "comparison_resolution_flagged"
    elif a_plateau:
        verdict = "plateau_with_heating"
    else:
        verdict = "no_plateau"
    record = {"schema_version": ASSESSMENT_SCHEMA, "utc": utc_now(), "experiment_id": protocol["experiment_id"], "option": protocol.get("option"), "results_dir": results.name,
              "git_head_now": _runner().git_head(),
              "run": {"stop_reason": summary["stop_reason"], "ion_transit_times": summary.get("ion_transit_times"), "steps_completed": summary["steps_completed"], "plateau": plateau,
                      "windowed_residual_over_electrode_work": windowed, "cells_per_debye_window_last": debye.get("cells_per_debye_window_last"), "peak_debye_soft_ok": soft_ok},
              "a_plateau": {"passed": a_plateau, "rule": acceptance.get("a_plateau")}, "b_residual_power": {"passed": b_ok, "value": windowed, "bound": 0.02, "rule": acceptance.get("b_residual_power")},
              "verdict": verdict, "verdict_rule": (acceptance.get("d_verdicts") or {}).get(verdict), "claim_ceiling": comparison_module.CLAIM_CEILING}
    artifacts.write_canonical_json(results / "assessment.json", record)
    log(f"[assess] {results.name}: verdict {verdict} (a {a_plateau}, b {b_ok} [{windowed}], soft {soft_ok})")
    return record


def extract_s(results: Path, protocol: dict[str, Any], cusp_planes_m: list[float]) -> dict[str, Any]:
    """Our S per channel-comparable row from summary.json + the trailing-window maps.npz (estimands of the comparison spec)."""

    from cft_revival.pic2d import artifacts

    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    maps = artifacts.read_npz(results / "maps.npz")
    currents = summary.get("window_currents_a") or {}
    i_a = currents.get("discharge_a")
    i_beam = currents.get("exit_ion_beam_a")
    q_in = float(reference.SETUP["mass_flow"]["value"]["atoms_per_s"])
    e = reference.ELEMENTARY_CHARGE_C
    from cft_revival.pic2d.models import ChannelGeometry, Grid2D

    n_e = np.asarray(maps["n_e_per_m3"])
    n_i = np.asarray(maps["n_i_per_m3"]) if "n_i_per_m3" in maps else n_e
    phi = np.asarray(maps["phi_v"]) if "phi_v" in maps else None
    g = protocol["geometry"]
    grid = Grid2D(ChannelGeometry(g["bore_radius_m"], g["z_min_m"], g["z_max_m"], g["cone_start_z_m"], g["exit_radius_m"]), int(protocol["case"]["radial_cells"]), int(protocol["case"]["axial_cells"]))
    grid_r, grid_z = np.asarray(grid.r_m), np.asarray(grid.z_m)
    anode_v = float(protocol["operating_point"]["anode_potential_v"])
    out: dict[str, Any] = {"anode_electron_current_a": i_a, "net_ionisation_fraction": None if i_a is None else i_a / (e * q_in), "ion_beam_current_a": i_beam,
                           "beam_fraction_of_feed": None if i_beam is None else i_beam / (e * q_in)}
    if phi is not None and phi.shape == (len(grid_r), len(grid_z)):
        r_w = float(protocol["geometry"]["bore_radius_m"])
        interior = grid_r <= r_w - 0.5e-3
        planes = [0.0, *sorted(cusp_planes_m), float(protocol["geometry"]["z_max_m"])]
        cell_potentials = []
        for z0, z1 in itertools.pairwise(planes):
            sel = (grid_z >= z0) & (grid_z < z1)
            weights = n_e[np.ix_(interior, sel)]
            cell_potentials.append(float(np.sum(weights * phi[np.ix_(interior, sel)]) / max(np.sum(weights), 1e-300)))
        out["cell_potentials_v"] = cell_potentials
        out["plasma_potential_near_anode_above_anode_v"] = cell_potentials[0] - anode_v
        if len(cell_potentials) >= 3:
            out["potential_drop_first_cusp_v"] = cell_potentials[0] - cell_potentials[1]
            out["potential_drop_second_cusp_v"] = cell_potentials[1] - cell_potentials[2]
    occupancy = np.asarray(maps["sample_count_e"]) if "sample_count_e" in maps else None
    resolved = n_i > 0 if occupancy is None else occupancy >= 32
    out["ion_density_typical_per_m3"] = float(np.max(np.where(resolved, n_i, 0.0)))
    out["ion_density_channel_mean_per_m3"] = float(np.mean(n_i[n_i > 0])) if np.any(n_i > 0) else None
    if "wall_ion_mean_energy_ev" in maps:
        energy = np.asarray(maps["wall_ion_mean_energy_ev"])
        out["wall_ion_energy_max_ev"] = float(np.nanmax(energy))
    if "wall_ion_flux_per_m2_s" in maps:
        flux = np.asarray(maps["wall_ion_flux_per_m2_s"])
        out["wall_ion_current_density_max_a_per_m2"] = float(np.nanmax(flux)) * e
    return out


def compare_run(results: Path, protocol: dict[str, Any], *, log=lambda text: print(text, flush=True)) -> dict[str, Any]:
    from cft_revival.pic2d import artifacts

    spec = comparison_module.comparison_document()
    binding = field_module.load_binding()
    cusps = binding["gates"]["G2_interior_nulls"]["interior_nulls_m"]
    s_values = extract_s(results, protocol, cusps)
    assessment = json.loads((results / "assessment.json").read_text(encoding="utf-8")) if (results / "assessment.json").is_file() else None
    rows = []
    for quantity in spec["quantities"]:
        qid = quantity["quantity_id"]
        if "channel" not in quantity["comparable_under"]:
            rows.append({"quantity_id": qid, "compared": False, "reason": "not comparable under the channel-only protocol (needs the plume box)"})
            continue
        s = s_values.get(qid)
        if s is None or not math.isfinite(float(s)) or (quantity.get("log_scale") and float(s) <= 0.0):
            rows.append({"quantity_id": qid, "compared": False, "reason": "estimand not available in this run's artifacts"})
            continue
        metric = comparison_module.validation_metric(quantity, float(s))
        rows.append({**metric, "compared": True, "resolution_flag": None if assessment is None else assessment["verdict"] == "comparison_resolution_flagged"})
    record = {"schema_version": COMPARISON_RESULT_SCHEMA, "utc": utc_now(), "results_dir": results.name, "assessment_verdict": None if assessment is None else assessment["verdict"],
              "quotable": assessment is not None and assessment["verdict"] == "comparison_quotable", "claim_ceiling": comparison_module.CLAIM_CEILING, "s_values": s_values, "rows": rows,
              "closure_differences": spec["closure_differences"], "reference_doi": reference.DOI}
    artifacts.write_canonical_json(results / "comparison.json", record)
    compared = [r for r in rows if r.get("compared")]
    log(f"[compare] {results.name}: {len(compared)} rows compared; verdicts {sorted({r['verdict'] for r in compared})}; quotable {record['quotable']}")
    return record


def command_assess(args: argparse.Namespace) -> int:
    results = Path(args.results_dir) if args.results_dir else results_dir(args.variant, args.grid)
    assess_run(_run_protocol(results, args.variant, args.grid), results)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    results = Path(args.results_dir) if args.results_dir else results_dir(args.variant, args.grid)
    compare_run(results, _run_protocol(results, args.variant, args.grid))
    return 0


# -- CLI ---------------------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, *, option: bool = False, results: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        if option:
            p.add_argument("--variant", default=protocol_module.PRIMARY_VARIANT, choices=tuple(VARIANTS))
            p.add_argument("--grid", default=protocol_module.PRIMARY_GRID, choices=tuple(GRIDS))
        if results:
            p.add_argument("--results-dir", default=None)
        p.set_defaults(fn=fn)
        return p

    add("reference", command_reference)
    f = add("fields", command_fields)
    f.add_argument("--no-sensitivity", action="store_true")
    add("regate", command_regate)
    pr = add("protocol", command_protocol, option=True)
    pr.add_argument("--with-field", action="store_true")
    add("comparison", command_comparison)
    add("compose", command_compose)
    pf = add("preflight", command_preflight, option=True)      # amendment 1: --variant/--grid select the option whose GPU timing (and record file) is written
    pf.add_argument("--gpu-timing", action="store_true", help="launch box: time >= 2000 production steps of the option at the seed and plateau loads (records the MPS contention)")
    pf.add_argument("--timing-steps", type=int, default=2000)
    pf.add_argument("--backend", default="warp-cuda")
    add("cost", command_cost)
    r = add("run", command_run, option=True, results=True)
    r.add_argument("--backend", default="warp-cuda")
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--wall-budget-seconds", type=float, default=None)
    r.add_argument("--ignore-code-identity", action="store_true")
    r.add_argument("--shrunk-cadences", action="store_true")
    r.add_argument("--label", default=None)
    r.add_argument("--allow-launch", action="store_true")
    sh = add("shakedown", command_shakedown, option=True)
    sh.add_argument("--backend", default="warp-cuda")
    sh.add_argument("--max-steps", type=int, default=SHAKEDOWN_MAX_STEPS)
    sh.add_argument("--output", default=None)
    la = add("launch", command_launch, option=True)
    la.add_argument("--backend", default="warp-cuda")
    la.add_argument("--expect-commit", required=True, help="the preregistration commit (HEAD must be it)")
    la.add_argument("--resume", action="store_true")
    la.add_argument("--allow-dirty", action="store_true", help="development only; never for the preregistered execution")
    la.add_argument("--require-mps", action="store_true", help="refuse unless CUDA_MPS_PIPE_DIRECTORY is set and exists (the four-slot H100 configuration)")
    la.add_argument("--wall-budget-seconds", type=float, default=None)
    fin = add("finalize", command_finalize, option=True, results=True)
    fin.add_argument("--backend", default="warp-cuda")
    fin.add_argument("--stop-reason", default="finalized_from_checkpoint")
    fin.add_argument("--allow-refinalize", action="store_true")
    fin.add_argument("--recover-runner-stop", action="store_true")
    add("status", command_status, option=True, results=True)
    add("assess", command_assess, option=True, results=True)
    add("compare", command_compare, option=True, results=True)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
