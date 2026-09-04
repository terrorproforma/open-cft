"""External validation v0 (DRAFT) - runner stages.

From ``modern/`` (``$env:PYTHONPATH="$PWD\\src;$PWD"`` / ``PYTHONPATH=src:.``)::

    python -m experiments.pic2d_external_validation_v0.run reference                    # print the reference record (setup table, reported quantities, DOI)
    python -m experiments.pic2d_external_validation_v0.run fields [--no-sensitivity]    # CPU: P2 solve of the reconstruction (+ no-ring sensitivity) -> fields/<id>/binding.json
    python -m experiments.pic2d_external_validation_v0.run regate                       # recompute the field gates from the bound checkpoint (no solve)
    python -m experiments.pic2d_external_validation_v0.run protocol [--variant V] [--grid G] [--with-field]   # print a composed run protocol
    python -m experiments.pic2d_external_validation_v0.run comparison                   # write comparison-spec.json (validated)
    python -m experiments.pic2d_external_validation_v0.run compose                      # write protocols/*.json (draft, both variants) + protocol.json
    python -m experiments.pic2d_external_validation_v0.run preflight                    # whole-set preflight -> preflight-channel-20um.json
    python -m experiments.pic2d_external_validation_v0.run cost                         # cost table (20 / 33 / 15 um, plume box)
    python -m experiments.pic2d_external_validation_v0.run run --allow-launch ...       # labelled development / shakedown run through the shared runner (never evidence)
    python -m experiments.pic2d_external_validation_v0.run launch ...                   # REFUSES: nothing here is preregistered
    python -m experiments.pic2d_external_validation_v0.run assess|compare --results-dir DIR   # after a run: acceptance verdict; the comparison rows with E, u_val, statement

``run`` steps with the shared steady-state runner (``experiments.pic2d_cft_steady_state_v1.run``) under the composed protocol with the reconstructed
node field passed in directly; results go to ``results/<option>/``.  ``compare`` evaluates every channel-comparable row of the comparison spec with the
run's trailing-window quantities (S) and writes ``comparison.json`` next to the run's ``summary.json``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
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
ASSESSMENT_SCHEMA = "cft.pic2d.external-validation-v0.assessment/0.1.0-draft"
COMPARISON_RESULT_SCHEMA = "cft.pic2d.external-validation-v0.comparison-result/0.1.0-draft"
SHRUNK_CADENCES = {"series_interval_steps": 200, "device_sync_steps": 200, "checkpoint_every_steps": 4000, "averaging_window_steps": 40000, "frame_cadence_steps": 2000,
                   "peak_debye_window_steps": 40000, "peak_debye_window_snapshot_steps": 4000, "residual_window_steps": 40000}


def results_dir(variant: str, grid: str) -> Path:
    return RESULTS_ROOT / option_tag(variant, grid)


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


def command_compose(args: argparse.Namespace) -> int:
    protocol_module.write_comparison_spec()
    sealed = protocol_module.compose_all()
    out = protocol_module.write_experiment_protocol(preflight_summary=_preflight_summary(), sealed=sealed, field_binding_summary=_binding_summary())
    print(f"[compose] {len(sealed)} draft run protocols under {protocol_module.PROTOCOLS_DIR}; experiment protocol -> {out} (sha256 {_sha256(out)[:12]})")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    path, report = preflight_module.write_preflight()
    print(f"[preflight] all_passed={report['all_passed']} (launch set {report['launch_set_passed']}) over {report['option_count']} options -> {path}")
    return 0 if report["all_passed"] else 1


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


def command_launch(args: argparse.Namespace) -> int:
    print("[launch] REFUSED: external validation v0 is a DRAFT - nothing is preregistered. The coordinator preregisters (protocol.json + protocols/ + preflight + shakedown records "
          "committed) and launches from that commit; until then only `run --allow-launch` (labelled development) exists.", file=sys.stderr)
    return 2


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
    add("preflight", command_preflight)
    add("cost", command_cost)
    r = add("run", command_run, option=True, results=True)
    r.add_argument("--backend", default="warp-cuda")
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--wall-budget-seconds", type=float, default=None)
    r.add_argument("--ignore-code-identity", action="store_true")
    r.add_argument("--shrunk-cadences", action="store_true")
    r.add_argument("--label", default=None)
    r.add_argument("--allow-launch", action="store_true")
    add("launch", command_launch, option=True)
    add("assess", command_assess, option=True, results=True)
    add("compare", command_compare, option=True, results=True)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
