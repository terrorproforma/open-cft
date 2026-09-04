"""Generate the standalone PIC-2D CFT steady-state v4 dashboard: the preregistered 33.3 um grid-refinement check
of the 50 um base plateau and its convergence verdict.

Headline: ``modern/experiments/pic2d_cft_steady_state_v4/results`` (the one preregistered execution at commit
``392129e5``: 90 x 720 cells, dt 1.4 ps, W 2.667e4, v1.3 closure, v2.0.3 gates) assessed by its predeclared
``assessment.json`` against the accepted 50 um base plateau ``pic2d_cft_steady_state_v2/results`` (commit
``24ab82f4``) with the 50 um convergence pair (``results-seed-b``, ``results-w-0.7``) as the particle-resolution
band.  Every embedded input is hash-verified against its ``.sha256.json`` sidecar and against the hashes the run
recorded (protocol, configuration, artifacts); the pinned reference quantities are re-derived from the v2
artifacts; the windowed residual power is recomputed from the series and must reproduce the runner's value.
No timestamps or runtime measurements are added, so identical inputs give identical bytes on the anchor
platform (``<name>.anchor-platform.json``); the page is self-contained (no network access) and states the
claim boundary and the verdict on the first screen.

Energy-ledger correction (model v2.0.6, post hoc): every embedded case carries its ``ledger-corrected.json`` sidecar
(bound to the case's series hash; the corrected windowed residual is recomputed here from the series with
``cft_revival.pic2d.ledger_recompute`` and must reproduce the sidecar's end value), and the refined run's post-hoc
re-read ``assessment-corrected-ledger.json`` (bound to the sidecar, the recorded assessment, the summary and the
protocol) is embedded beside the recorded assessment.  The page shows BOTH readings, clearly labelled: the recorded
verdict stands as recorded; on the corrected ledger acceptance (b) fails (+2.46 % > +2 %).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from hashlib import sha256
from math import isfinite, pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np

MODERN = Path(__file__).resolve().parents[1]
SRC = MODERN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MODERN) not in sys.path:
    sys.path.insert(0, str(MODERN))

from cft_revival.pic2d.artifacts import (
    platform_fingerprint,
    read_canonical_json,
    read_npz,
)
from cft_revival.pic2d.ledger_recompute import SIDECAR_NAME as LEDGER_SIDECAR_NAME
from cft_revival.pic2d.ledger_recompute import corrected_residual, windowed_ratios

EXPERIMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v4"
RESULTS = EXPERIMENT / "results"
PROTOCOL = EXPERIMENT / "protocol.json"
REFERENCE_EXPERIMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v2"
REFERENCE_PROTOCOL = REFERENCE_EXPERIMENT / "protocol.json"
DEFAULT_OUTPUT = Path(__file__).with_name("pic2d-cft-steady-state-v4.html")
SCHEMA = "cft-pic2d-cft-steady-state-v4-visualization/0.2.0"
STATUS = "preregistered_resolution_convergence_study_not_validated"
VERDICTS = ("converged", "resolution_limited", "refinement_heating", "no_plateau")
REREAD_NAME = "assessment-corrected-ledger.json"
REREAD_SCHEMA = "cft-revival.pic2d-cft-steady-state-v4.assessment-corrected-ledger/1.0.0"
LEDGER_SIDECAR_SCHEMA = "cft.pic2d.ledger-corrected/1.0.0"
ACCEPTANCE_B_BOUND = 0.02
STOP_REASONS = {"plateau_reached_after_min_transit_times", "wall_clock_budget_reached"}
# the 50 um cases embedded beside the refined run: (results directory name, label, role)
REFERENCE_CASES = (("results", "50 µm base (v2, reference)", "reference"), ("results-seed-b", "50 µm seed-b (band)", "band"),
                   ("results-w-0.7", "50 µm W×0.7 (band)", "band"))
SERIES_KEYS = (
    "time_s", "step", "electrons", "ions", "current_discharge_a", "current_exit_ion_beam_a", "current_ionization_rate_per_s",
    "current_wall_ion_a", "current_wall_electron_a", "neutral_density_per_m3", "neutral_fixed_point_per_m3", "neutral_ionization_rate_per_s",
    "peak_omega_pe_dt", "phi_max_v", "phi_min_v", "phi_mean_v", "kinetic_electron_j", "kinetic_ion_j", "field_energy_j", "total_energy_j",
)
V4_SERIES_KEYS = ("peak_node_window_cells_per_debye", "peak_node_cells_per_debye", "peak_node_window_n_e_peak_per_m3", "peak_node_window_t_e_peak_ev",
                  "peak_node_n_e_peak_per_m3", "peak_node_t_e_peak_ev", "peak_node_window_resolved_nodes")
MAX_SERIES_POINTS = 3200
RESIDUAL_WINDOW_STEPS = 400_000
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31
# the quantities of acceptance (c) in display order: (key, label, unit, scale for display)
QUANTITIES = (
    ("discharge_current_a", "I_d (anode e⁻ − anode Xe⁺, window)", "mA", 1e3),
    ("exit_ion_beam_a", "I_beam,i (exit plane, window)", "mA", 1e3),
    ("ionization_rate_per_s", "S (trailing-20 % mean)", "s⁻¹", 1.0),
    ("gross_utilisation", "utilisation S / Q_in (trailing)", "", 1.0),
    ("neutral_density_per_m3", "n_g (trailing-20 % mean)", "m⁻³", 1.0),
    ("peak_n_e_window_per_m3", "peak n_e (window maps, densest node)", "m⁻³", 1.0),
    ("t_e_peak_window_ev", "T_e at the peak node (window)", "eV", 1.0),
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _verify_sidecar(path: Path) -> str:
    sidecar = json.loads(path.with_name(path.name + ".sha256.json").read_text(encoding="utf-8"))
    digest = _file_sha256(path)
    if sidecar["byte_sha256"] != digest:
        raise ValueError(f"{path.name}: sidecar SHA-256 mismatch")
    return digest


def _round(values: Any, digits: int = 6) -> list[Any]:
    out: list[Any] = []
    for value in np.asarray(values, dtype=np.float64).ravel().tolist():
        out.append(None if not isfinite(value) else float(f"{value:.{digits}g}"))
    return out


def _decimate(values: np.ndarray, stride: int) -> np.ndarray:
    values = np.asarray(values)
    if stride <= 1:
        return values
    out = values[::stride]
    if (values.shape[0] - 1) % stride:
        out = np.concatenate([out, values[-1:]])
    return out


def _peak_from_maps(maps: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = np.asarray(maps["n_e_per_m3"])
    t = np.asarray(maps["t_e_ev"])
    i, j = np.unravel_index(int(np.nanargmax(n)), n.shape)
    return {"peak_n_e_window_per_m3": float(n[i, j]), "t_e_peak_window_ev": float(t[i, j]), "node": [int(i), int(j)]}


def _quantities(summary: Mapping[str, Any], peak: Mapping[str, Any]) -> dict[str, float]:
    """The acceptance-(c) quantities exactly as ``pic2d_cft_steady_state_v4.run.run_quantities`` reads them."""

    return {
        "discharge_current_a": float(summary["window_currents_a"]["discharge_a"]),
        "exit_ion_beam_a": float(summary["window_currents_a"]["exit_ion_beam_a"]),
        "ionization_rate_per_s": float(summary["neutral_inventory"]["trailing_20pct_mean_ionization_rate_per_s"]),
        "gross_utilisation": float(summary["neutral_inventory"]["propellant_utilisation_trailing"]),
        "neutral_density_per_m3": float(summary["neutral_inventory"]["trailing_20pct_mean_density_per_m3"]),
        "peak_n_e_window_per_m3": float(peak["peak_n_e_window_per_m3"]),
        "t_e_peak_window_ev": float(peak["t_e_peak_window_ev"]),
    }


def windowed_residual(series: Mapping[str, np.ndarray], window_steps: int = RESIDUAL_WINDOW_STEPS) -> np.ndarray:
    """Trailing-window ledger residual / electrode work per record (the v2.0.3 gate statistic), NaN while incomplete.

    The series records are exact interval integrals, so the trailing sum over ``window_steps / interval`` records is
    the runner's window (verified against ``summary.grid_heating_triad`` for the v4 run to 1e-12).
    """

    step = np.asarray(series["step"], dtype=np.float64)
    interval = float(step[1] - step[0]) if step.size > 1 else float(step[0])
    n = round(window_steps / interval)
    residual = np.asarray(series["interval_residual_j"], dtype=np.float64)
    work = np.asarray(series["interval_electrode_work_j"], dtype=np.float64)
    out = np.full(residual.shape, np.nan)
    if residual.size < n:
        return out
    cr = np.concatenate([[0.0], np.cumsum(residual)])
    cw = np.concatenate([[0.0], np.cumsum(work)])
    num = cr[n:] - cr[:-n]
    den = cw[n:] - cw[:-n]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den != 0.0, num / den, np.nan)
    out[n - 1:] = ratio
    return out


def _lambda_d(n: float, t_ev: float) -> float:
    return sqrt(EPS0 * t_ev * E_CHARGE / (n * E_CHARGE**2))


def corrected_windowed_residual(series: Mapping[str, np.ndarray], window_steps: int = RESIDUAL_WINDOW_STEPS) -> np.ndarray:
    """The v2.0.6 CORRECTED trailing-window residual / electrode work per record (NaN while the window is incomplete).

    Corrected residual per record = H = field work + dU - electrode work (``cft_revival.pic2d.ledger_recompute``); the
    trailing window is the runner's (records with step > last - window, complete once a record precedes the window).
    """

    corrected, _h, _resume_first = corrected_residual(series)
    ratios = windowed_ratios(np.asarray(series["step"], dtype=np.float64), corrected, np.asarray(series["interval_electrode_work_j"], dtype=np.float64), window_steps)
    return np.where(ratios["complete"], ratios["ratio"], np.nan)


def ledger_sidecar_digest(results: Path, *, series_sha: str, recorded_last: float | None, corrected_last: float | None) -> dict[str, Any]:
    """Hash-verified digest of ``ledger-corrected.json`` bound to the case's series and checked against the recomputation."""

    path = results / LEDGER_SIDECAR_NAME
    if not path.is_file():
        raise ValueError(f"{results.name}: {LEDGER_SIDECAR_NAME} is missing - run `python -m cft_revival.pic2d.ledger_recompute {results}`")
    sidecar_sha = _verify_sidecar(path)
    sidecar = read_canonical_json(path)
    if sidecar.get("schema") != LEDGER_SIDECAR_SCHEMA:
        raise ValueError(f"{results.name}: {LEDGER_SIDECAR_NAME} has schema {sidecar.get('schema')!r}")
    if sidecar["inputs"]["series"]["sha256"] != series_sha:
        raise ValueError(f"{results.name}: {LEDGER_SIDECAR_NAME} describes another series ({sidecar['inputs']['series']['sha256'][:12]} vs {series_sha[:12]})")
    end = sidecar["end_state_window"]
    for name, mine, theirs in (("recorded", recorded_last, end["recorded_ratio"]), ("corrected", corrected_last, end["corrected_ratio"])):
        if mine is None or theirs is None or abs(mine - theirs) > 1e-9 * max(abs(theirs), 1e-12):
            raise ValueError(f"{results.name}: the {name} windowed residual recomputed here ({mine}) differs from the sidecar's ({theirs})")
    if end.get("recorded_ratio_matches_summary") is False:      # None: the run's summary has no gate reading (v1.3 runs)
        raise ValueError(f"{results.name}: the sidecar's recorded reading does not match the run's summary")
    gate = sidecar["threshold_crossings"]["0.05"]
    bound = sidecar["threshold_crossings"]["0.02"]
    return {
        "sidecar_sha256": sidecar_sha, "generated_by": sidecar.get("generated_by"), "macro_weight": sidecar["parameters"]["macro_weight"],
        "window_steps": int(sidecar["parameters"]["window_steps"]), "records": int(sidecar["records"]), "last_time_s": float(sidecar["last_time_s"]),
        "recorded_windowed": end["recorded_ratio"], "corrected_windowed": end["corrected_ratio"], "omitted_windowed": end["omitted_ratio"],
        "recorded_cumulative": sidecar["cumulative"]["recorded_over_electrode"], "corrected_cumulative": sidecar["cumulative"]["corrected_over_electrode"],
        "electrode_work_in_window_j": end["electrode_work_j"], "corrected_residual_in_window_j": end["corrected_residual_j"],
        "max_corrected_over_complete_windows": sidecar["max_over_complete_windows"]["corrected"],
        "corrected_first_checkpoint_at_or_above_0p02_time_s": None if bound["corrected_first_crossing_at_checkpoint"] is None else bound["corrected_first_crossing_at_checkpoint"]["time_s"],
        "recorded_gate_0p05_first_checkpoint_time_s": None if gate["recorded_first_crossing_at_checkpoint"] is None else gate["recorded_first_crossing_at_checkpoint"]["time_s"],
        "corrected_gate_0p05_first_checkpoint_time_s": None if gate["corrected_first_crossing_at_checkpoint"] is None else gate["corrected_first_crossing_at_checkpoint"]["time_s"],
        "acceptance_b_recorded_passes": None if end["recorded_ratio"] is None else bool(end["recorded_ratio"] < ACCEPTANCE_B_BOUND),
        "acceptance_b_corrected_passes": None if end["corrected_ratio"] is None else bool(end["corrected_ratio"] < ACCEPTANCE_B_BOUND),
        "cross_check_relative_difference": (sidecar.get("cross_check_vs_final_counts") or {}).get("relative_difference"),
    }


def build_case(results: Path, protocol_path: Path, *, label: str, role: str) -> dict[str, Any]:
    """Hash-verified digest of one steady-state results directory (summary, series, maps; protocol bound)."""

    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_sha = _verify_sidecar(summary_path)
    protocol_sha = _file_sha256(protocol_path)
    if summary["protocol_sha256"] != protocol_sha:
        raise ValueError(f"{results.name}: protocol drift - the run recorded {summary['protocol_sha256'][:12]}, the file hashes {protocol_sha[:12]}")
    maps_sha = _verify_sidecar(results / "maps.npz")
    series_sha = _verify_sidecar(results / "series.npz")
    if summary["artifacts"]["maps_npz_sha256"] != maps_sha or summary["artifacts"]["series_npz_sha256"] != series_sha:
        raise ValueError(f"{results.name}: artifact hashes differ from the ones the run recorded")
    maps = read_npz(results / "maps.npz")
    series = read_npz(results / "series.npz")
    peak = _peak_from_maps(maps)
    quantities = _quantities(summary, peak)
    grid = summary["provenance"]["config"]["grid"]
    dz = float(grid["dz_m"])
    lam = _lambda_d(peak["peak_n_e_window_per_m3"], peak["t_e_peak_window_ev"])
    stride = max(1, -(-len(series["time_s"]) // MAX_SERIES_POINTS))
    keys = list(SERIES_KEYS) + [k for k in V4_SERIES_KEYS if k in series]
    decimated = {key: _round(_decimate(series[key], stride), 6) for key in keys if key in series}
    residual_series = windowed_residual(series)
    decimated["windowed_residual_over_electrode_work"] = _round(_decimate(residual_series, stride), 5)
    residual_last = float(residual_series[-1]) if residual_series.size and isfinite(float(residual_series[-1])) else None
    corrected_series = corrected_windowed_residual(series)
    decimated["windowed_residual_corrected_over_electrode_work"] = _round(_decimate(corrected_series, stride), 5)
    corrected_last = float(corrected_series[-1]) if corrected_series.size and isfinite(float(corrected_series[-1])) else None
    ledger_corrected = ledger_sidecar_digest(results, series_sha=series_sha, recorded_last=residual_last, corrected_last=corrected_last)
    triad = summary.get("grid_heating_triad") or {}
    n_e = np.asarray(maps["n_e_per_m3"], dtype=np.float64)
    t_e = np.asarray(maps["t_e_ev"], dtype=np.float64)
    finite = np.where(np.isfinite(n_e), n_e, -np.inf)
    axial_peak_r = np.argmax(finite, axis=0)
    axial_peak_n = np.take_along_axis(finite, axial_peak_r[None, :], axis=0)[0]
    axial_peak_t = np.take_along_axis(np.where(np.isfinite(t_e), t_e, np.nan), axial_peak_r[None, :], axis=0)[0]
    node_r = np.arange(n_e.shape[0]) * float(grid["dr_m"])
    node_z = np.arange(n_e.shape[1]) * dz
    i_peak, j_peak = peak["node"]
    case = {
        "id": summary["case"]["id"], "label": label, "role": role, "results_dir": results.name, "experiment_id": summary["experiment_id"],
        "model_version": summary["model_version"], "git_head": summary.get("git_head"), "protocol_sha256": summary["protocol_sha256"],
        "config_sha256": summary["provenance"]["config_sha256"], "summary_sha256": summary_sha, "maps_npz_sha256": maps_sha,
        "series_npz_sha256": series_sha, "backend": summary["backend"], "stop_reason": summary["stop_reason"], "steps_completed": int(summary["steps_completed"]),
        "simulated_time_s": float(summary["simulated_time_s"]), "ion_transit_times": float(summary["ion_transit_times"]),
        "wall_seconds_total": float(summary["wall_seconds_total"]), "ms_per_step_last_session": summary.get("ms_per_step_this_session"),
        "sessions": len(summary.get("sessions") or []), "frames": (summary["artifacts"].get("frames") or {}).get("count"),
        "grid": {"radial_cells": int(grid["radial_cells"]), "axial_cells": int(grid["axial_cells"]), "dr_m": float(grid["dr_m"]), "dz_m": dz},
        "dt_s": float(summary["provenance"]["config"]["dt_s"]), "macro_weight": float(summary["provenance"]["config"]["macro_weight"]),
        "seed": int(summary["case"]["seed"]), "final_counts": summary["final_counts"], "plateau": summary.get("plateau"),
        "averaging_window_step_range": summary.get("averaging_window_step_range"), "averaging_window_steps": summary.get("averaging_window_steps"),
        "window_currents_a": summary["window_currents_a"], "window_maps_summary": summary["window_maps_summary"],
        "neutral_inventory": {k: summary["neutral_inventory"].get(k) for k in (
            "trailing_20pct_mean_density_per_m3", "trailing_20pct_analytic_fixed_point_per_m3", "trailing_20pct_mean_ionization_rate_per_s",
            "propellant_utilisation_trailing", "feed_atoms_per_s", "zero_ionization_density_per_m3", "cumulative_ledger_closure_relative_to_inventory")},
        "ledger": {k: summary["ledger"].get(k) for k in ("cumulative_residual_over_electrode_work", "cumulative_electrode_work_j", "cumulative_residual_j")},
        "grid_heating_triad": {k: triad.get(k) for k in (
            "windowed_energy_residual_over_electrode_work", "windowed_energy_residual_window_complete", "energy_residual_over_electrode_work",
            "ionisation_rate_drift", "t_e_dense_drift", "omega_pe_dt_drift", "soft_ok", "hard_failures")} if triad else None,
        "peak_node_debye": summary.get("peak_node_debye"),
        "quantities": quantities,
        "peak": {"node": peak["node"], "r_m": float(node_r[i_peak]), "z_m": float(node_z[j_peak]), "lambda_d_m": lam, "cells_per_debye": dz / lam,
                 "omega_pe_dt": sqrt(peak["peak_n_e_window_per_m3"] * E_CHARGE**2 / (EPS0 * M_E)) * float(summary["provenance"]["config"]["dt_s"])},
        "windowed_residual_recomputed": residual_last,
        "windowed_residual_corrected_recomputed": corrected_last,
        "ledger_corrected": ledger_corrected,
        "series_stride": stride, "series": decimated,
        "profiles": {"z_m": _round(node_z, 6), "axial_peak_n_e_per_m3": _round(np.where(np.isfinite(axial_peak_n), axial_peak_n, np.nan), 5),
                     "axial_peak_t_e_ev": _round(axial_peak_t, 5), "r_m": _round(node_r, 6),
                     "radial_n_e_at_peak_z_per_m3": _round(n_e[:, j_peak], 5), "radial_t_e_at_peak_z_ev": _round(t_e[:, j_peak], 5)},
    }
    recomputed = case["windowed_residual_recomputed"]
    recorded = (triad or {}).get("windowed_energy_residual_over_electrode_work")
    if recorded is not None and recomputed is not None and abs(recomputed - recorded) > 1e-6 * max(abs(recorded), 1e-12):
        raise ValueError(f"{results.name}: recomputed windowed residual {recomputed} differs from the runner's {recorded}")
    return case


def build_comparison(refined: Mapping[str, Any], reference: Mapping[str, Any], bands: list[Mapping[str, Any]], assessment: Mapping[str, Any]) -> dict[str, Any]:
    tolerances = {k: v["tolerance"] for k, v in assessment["c_convergence"]["quantities"].items()}
    rows = []
    for key, label, unit, scale in QUANTITIES:
        ref = reference["quantities"][key]
        value = refined["quantities"][key]
        entry = assessment["c_convergence"]["quantities"][key]
        if abs(entry["reference"] - ref) > 1e-12 * abs(ref) or abs(entry["value"] - value) > 1e-12 * abs(value):
            raise ValueError(f"{key}: assessment.json does not describe the embedded artifacts")
        rows.append({
            "key": key, "quantity": label, "unit": unit, "display_scale": scale, "reference": ref, "refined": value,
            "relative_difference": entry["relative_difference"], "tolerance": tolerances[key], "within": bool(entry["within"]),
            "bands": [{"label": b["label"], "value": b["quantities"][key], "relative_difference": (b["quantities"][key] - ref) / abs(ref)} for b in bands],
        })
    debye = {
        "reference_cells_per_debye_at_peak": reference["peak"]["cells_per_debye"], "refined_cells_per_debye_at_peak_maps": refined["peak"]["cells_per_debye"],
        "refined_window_gate_last": assessment["peak_debye_window"]["cells_per_debye_last"], "refined_window_gate_trailing_mean": assessment["peak_debye_window"]["trailing_mean"],
        "soft": 2.5, "hard": pi, "soft_ok": bool(assessment["peak_debye_window"]["soft_ok"]),
        "bands": [{"label": b["label"], "cells_per_debye_at_peak": b["peak"]["cells_per_debye"]} for b in bands],
        "cic_threshold_note": "Birdsall-Langdon CIC finite-grid-instability threshold pi; plume attempt 8 heated from ~3.2, the 50 um base sits at 3.17",
    }
    residuals = {
        "refined_windowed": refined["grid_heating_triad"]["windowed_energy_residual_over_electrode_work"], "refined_cumulative": refined["ledger"]["cumulative_residual_over_electrode_work"],
        "reference_windowed_recomputed": reference["windowed_residual_recomputed"], "reference_cumulative": reference["ledger"]["cumulative_residual_over_electrode_work"],
        "bands": [{"label": b["label"], "windowed_recomputed": b["windowed_residual_recomputed"], "cumulative": b["ledger"]["cumulative_residual_over_electrode_work"],
                   "windowed_corrected": b["ledger_corrected"]["corrected_windowed"], "cumulative_corrected": b["ledger_corrected"]["corrected_cumulative"]} for b in bands],
        "refined_windowed_corrected": refined["ledger_corrected"]["corrected_windowed"], "refined_cumulative_corrected": refined["ledger_corrected"]["corrected_cumulative"],
        "reference_windowed_corrected": reference["ledger_corrected"]["corrected_windowed"], "reference_cumulative_corrected": reference["ledger_corrected"]["corrected_cumulative"],
        "acceptance_bound": assessment["b_residual_power"]["bound"], "gate": 0.05, "one_sided": True,
        "statistic_note": "recorded = the pre-v2.0.6 ledger (H - L_inel, biased negative by the inelastic power); corrected = H (model v2.0.6 sidecar); same window, same bound",
    }
    return {"rows": rows, "all_within": bool(assessment["c_convergence"]["all_within"]), "debye": debye, "residuals": residuals,
            "failed": [r["key"] for r in rows if not r["within"]]}


def build_payload(results: Path = RESULTS, protocol_path: Path = PROTOCOL, reference_experiment: Path = REFERENCE_EXPERIMENT) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assessment_path = results / "assessment.json"
    assessment = read_canonical_json(assessment_path)
    assessment_sha = _verify_sidecar(assessment_path)
    run_state = read_canonical_json(results / "run_state.json")
    _verify_sidecar(results / "run_state.json")
    _verify_sidecar(results / "checkpoint-final.json")
    lock = json.loads((results / "execution-lock.json").read_text(encoding="utf-8"))
    refined = build_case(results, protocol_path, label="33.3 µm v4 (refined, this run)", role="refined")
    if assessment["run"]["protocol_sha256"] != refined["protocol_sha256"] or assessment["run"]["config_sha256"] != refined["config_sha256"]:
        raise ValueError("assessment.json does not describe the embedded run")
    if lock["commit"] != refined["git_head"] or lock["protocol_sha256"] != refined["protocol_sha256"] or lock["config_sha256"] != refined["config_sha256"]:
        raise ValueError("execution lock does not bind the embedded run")
    reference_protocol = reference_experiment / "protocol.json"
    cases = [refined]
    for name, label, role in REFERENCE_CASES:
        case_dir = reference_experiment / name
        if not (case_dir / "summary.json").is_file():
            raise ValueError(f"reference case {name} is not materialised")
        cases.append(build_case(case_dir, reference_protocol, label=label, role=role))
    reference = cases[1]
    pinned = protocol["reference_run"]["quantities"]
    for key, _label, _unit, _scale in QUANTITIES:
        if abs(float(pinned[key]) - reference["quantities"][key]) > 1e-9 * max(abs(float(pinned[key])), 1e-300):
            raise ValueError(f"protocol.reference_run.quantities.{key} disagrees with the v2 base artifacts")
    if assessment["verdict"] not in VERDICTS:
        raise ValueError("unknown verdict")
    comparison = build_comparison(refined, reference, cases[2:], assessment)
    expected_verdict = ("converged" if assessment["a_plateau"]["passed"] and assessment["b_residual_power"]["passed"] and comparison["all_within"]
                        else "resolution_limited" if assessment["a_plateau"]["passed"] and assessment["b_residual_power"]["passed"]
                        else "refinement_heating" if assessment["a_plateau"]["passed"] else "no_plateau")
    if expected_verdict != assessment["verdict"]:
        raise ValueError("assessment verdict is not the one its own (a)-(c) outcomes imply")
    acceptance = protocol["stopping_rule"]["acceptance"]
    budget = protocol["budget_v1_3"]
    corrected_ledger = build_corrected_ledger(results, assessment_sha=assessment_sha, protocol_sha=_file_sha256(protocol_path), cases=cases, assessment=assessment,
                                              all_within=comparison["all_within"])
    reread = corrected_ledger["reread"]
    statement = (
        f"Preregistered grid-refinement check (33.3 µm / 1.4 ps / W 2.667e4) of the development PIC-MCC 50 µm plateau under the same v1.3 closure and "
        f"operating point; one execution at commit {lock['commit'][:8]}; recorded verdict {assessment['verdict'].replace('_', ' ')} at the predeclared tolerances "
        f"(10 % I_d / S / utilisation / n_g / I_beam, 20 % peak n_e / T_e,peak). Energy-ledger correction (model v2.0.6, post hoc): the recorded ledger "
        f"residuals lacked the macro weight on the inelastic sink; on the corrected ledger the 33 µm plateau heats at "
        f"{100.0 * reread['b_residual_power']['corrected']['windowed_residual_over_electrode_work']:+.2f} % of the electrode work (acceptance (b) < +2 %: "
        f"recorded {'PASS' if reread['b_residual_power']['recorded']['passed'] else 'FAIL'} → corrected {'PASS' if reread['b_residual_power']['corrected']['passed'] else 'FAIL'}; "
        f"predeclared (d) tree on the corrected ledger → {reread['verdict_on_corrected_ledger'].replace('_', ' ')}) and the 50 µm base at "
        f"{100.0 * cases[1]['ledger_corrected']['corrected_windowed']:+.1f} %; the recorded verdict stands as recorded, the 33 µm plateau is not a clean "
        f"(energy-conserving) reference and neither grid may be called converged before the 25 µm ladder point reports. Single seed, one refined grid and one "
        f"weight (grid and particle-weight effects entangled); not validated against experiment; not a thruster performance prediction; the neutral transient "
        f"is artificial and only the fixed point is physical."
    )
    return {
        "schema": SCHEMA, "experiment_id": protocol["experiment_id"], "status": STATUS, "verdict": assessment["verdict"],
        "model_version": protocol["model_version"], "claim_boundary": protocol["claim_boundary"], "claim_statement": statement,
        "simplifications": list(protocol["simplifications"]),
        "protocol": {
            "file_sha256": _file_sha256(protocol_path), "reference_protocol_sha256": _file_sha256(reference_protocol),
            "preregistration_commit": lock["commit"], "reference_commit": protocol["reference_run"]["commit"], "experiment_id": protocol["experiment_id"],
            "acceptance": {k: v for k, v in acceptance.items() if k != "c_convergence_tolerances"},
            "tolerances": {k: v for k, v in acceptance["c_convergence_tolerances"].items() if k != "note"}, "tolerances_note": acceptance["c_convergence_tolerances"]["note"],
            "plateau_rule": protocol["stopping_rule"]["plateau"], "wall_budget_seconds": protocol["stopping_rule"]["wall_budget_seconds"],
            "ion_transit_time_s": budget["ion_transit_time_s"], "ion_transit_note": budget["ion_transit_note"], "min_transit_times": protocol["stopping_rule"]["min_transit_times"],
            "convergence_pair": protocol["reference_run"]["convergence_pair"], "preregistration": protocol["preregistration"],
        },
        "assessment": {
            "sha256": assessment_sha, "utc": assessment["utc"], "verdict": assessment["verdict"], "git_head_now": assessment["git_head_now"],
            "a_plateau": assessment["a_plateau"], "b_residual_power": assessment["b_residual_power"], "c_convergence": assessment["c_convergence"],
            "d_reclassification": assessment["d_reclassification"], "peak_debye_window": assessment["peak_debye_window"],
            "reference_consistency": assessment["reference_consistency"],
        },
        "execution": {"lock": lock, "run_state": run_state},
        "comparison": comparison, "cases": cases, "corrected_ledger": corrected_ledger,
    }


def build_corrected_ledger(results: Path, *, assessment_sha: str, protocol_sha: str, cases: list[Mapping[str, Any]], assessment: Mapping[str, Any],
                           all_within: bool) -> dict[str, Any]:
    """The post-hoc re-read (``assessment-corrected-ledger.json``) bound to the embedded artifacts, plus every case's sidecar reading."""

    path = results / REREAD_NAME
    if not path.is_file():
        raise ValueError(f"{REREAD_NAME} is missing - run `python -m experiments.pic2d_cft_steady_state_v4.assess_corrected_ledger`")
    reread_sha = _verify_sidecar(path)
    reread = read_canonical_json(path)
    refined = cases[0]
    if reread.get("schema_version") != REREAD_SCHEMA:
        raise ValueError(f"{REREAD_NAME}: unexpected schema {reread.get('schema_version')!r}")
    inputs = reread["inputs"]
    if inputs["assessment"]["sha256"] != assessment_sha:
        raise ValueError(f"{REREAD_NAME} does not bind the embedded assessment.json")
    if inputs["ledger_corrected"]["sha256"] != refined["ledger_corrected"]["sidecar_sha256"]:
        raise ValueError(f"{REREAD_NAME} does not bind the embedded {LEDGER_SIDECAR_NAME}")
    if inputs["summary"]["sha256"] != refined["summary_sha256"] or inputs["protocol"]["sha256"] != protocol_sha:
        raise ValueError(f"{REREAD_NAME} does not bind the embedded summary / protocol")
    if reread["verdict_recorded"] != assessment["verdict"]:
        raise ValueError(f"{REREAD_NAME} records another recorded verdict than assessment.json")
    b = reread["b_residual_power"]
    if abs(b["corrected"]["windowed_residual_over_electrode_work"] - refined["ledger_corrected"]["corrected_windowed"]) > 1e-12 * abs(refined["ledger_corrected"]["corrected_windowed"]):
        raise ValueError(f"{REREAD_NAME}: corrected (b) value is not the sidecar's")
    if not all(bool(v) for v in inputs["binding_checks"].values()):
        raise ValueError(f"{REREAD_NAME}: a recorded binding check failed")
    return {
        "reread": {
            "file": REREAD_NAME, "sha256": reread_sha, "utc": reread["utc"], "generated_by": reread["generated_by"], "kind": reread["kind"],
            "model_version_note": reread["model_version_note"], "verdict_recorded": reread["verdict_recorded"],
            "verdict_on_corrected_ledger": reread["verdict_on_corrected_ledger"], "verdict_statement": reread["verdict_statement"],
            "b_residual_power": b, "d_reclassification": reread["d_reclassification"], "disallowed_wording": reread["disallowed_wording"],
            "binding_checks": inputs["binding_checks"], "all_within_c": bool(all_within),
        },
        "cases": [{"id": c["id"], "label": c["label"], "role": c["role"], "results_dir": c["results_dir"], **c["ledger_corrected"]} for c in cases],
        "spec": "spec/pic2d/pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6 / gate_recalibration_v2_0_6",
        "thresholds": {"acceptance_b": ACCEPTANCE_B_BOUND, "hard_gate": 0.05, "kept": "5 % hard gate and 2 % acceptance bound are KEPT (not loosened) on the corrected statistic"},
    }


def validate_payload(payload: Mapping[str, Any]) -> None:
    required = {"schema", "experiment_id", "status", "verdict", "model_version", "claim_boundary", "claim_statement", "simplifications", "protocol",
                "assessment", "execution", "comparison", "cases", "corrected_ledger"}
    if set(payload) != required:
        raise ValueError("payload keys do not match the closed schema")
    if payload["schema"] != SCHEMA or payload["status"] != STATUS:
        raise ValueError("unsupported payload schema or status")
    if payload["verdict"] not in VERDICTS or payload["assessment"]["verdict"] != payload["verdict"]:
        raise ValueError("verdict must be one of the four declared outcomes and match the assessment")
    statement = payload["claim_statement"].lower()
    if not payload["simplifications"] or "not validated" not in statement or "preregistered" not in statement or payload["verdict"].replace("_", " ") not in statement:
        raise ValueError("claim boundary must be explicit and carry the verdict")
    if "corrected ledger" not in statement or "energy-ledger correction" not in statement:
        raise ValueError("claim boundary must disclose the energy-ledger correction and the corrected reading")
    for key in ("file_sha256", "reference_protocol_sha256"):
        if not isinstance(payload["protocol"][key], str) or len(payload["protocol"][key]) != 64:
            raise ValueError(f"protocol {key} must be a SHA-256")
    if not payload["cases"] or payload["cases"][0]["role"] != "refined" or payload["cases"][1]["role"] != "reference":
        raise ValueError("payload must contain the refined case first and the reference second")
    if payload["cases"][0]["protocol_sha256"] != payload["protocol"]["file_sha256"]:
        raise ValueError("refined case protocol hash differs from the protocol file")
    for case in payload["cases"]:
        for key in ("summary_sha256", "maps_npz_sha256", "series_npz_sha256", "protocol_sha256", "config_sha256"):
            digest = case[key]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{case['id']}: {key} must be a SHA-256")
        if case["role"] != "refined" and case["protocol_sha256"] != payload["protocol"]["reference_protocol_sha256"]:
            raise ValueError(f"{case['id']}: reference case protocol hash differs from the v2 protocol file")
        if case["stop_reason"] not in STOP_REASONS:
            raise ValueError(f"{case['id']}: unknown stop reason")
        n = len(case["series"]["time_s"])
        for key, values in case["series"].items():
            if len(values) != n:
                raise ValueError(f"{case['id']}: series {key} length differs from time_s")
        for key in QUANTITIES:
            if not isfinite(case["quantities"][key[0]]):
                raise ValueError(f"{case['id']}: quantity {key[0]} is not finite")
        if len(case["profiles"]["z_m"]) != len(case["profiles"]["axial_peak_n_e_per_m3"]) or len(case["profiles"]["r_m"]) != len(case["profiles"]["radial_n_e_at_peak_z_per_m3"]):
            raise ValueError(f"{case['id']}: profile lengths differ")
    comparison = payload["comparison"]
    if [row["key"] for row in comparison["rows"]] != [q[0] for q in QUANTITIES]:
        raise ValueError("comparison rows must cover the acceptance quantities in order")
    for row in comparison["rows"]:
        if row["within"] != (abs(row["relative_difference"]) <= row["tolerance"]):
            raise ValueError(f"{row['key']}: 'within' does not follow from the difference and the tolerance")
    if comparison["all_within"] != all(row["within"] for row in comparison["rows"]) or set(comparison["failed"]) != {r["key"] for r in comparison["rows"] if not r["within"]}:
        raise ValueError("comparison verdict fields are inconsistent")
    a = payload["assessment"]
    verdict = ("converged" if a["a_plateau"]["passed"] and a["b_residual_power"]["passed"] and comparison["all_within"]
               else "resolution_limited" if a["a_plateau"]["passed"] and a["b_residual_power"]["passed"] else "refinement_heating" if a["a_plateau"]["passed"] else "no_plateau")
    if verdict != payload["verdict"]:
        raise ValueError("verdict does not follow from the recorded (a)-(c) outcomes")
    lock = payload["execution"]["lock"]
    if lock["commit"] != payload["cases"][0]["git_head"] or lock["protocol_sha256"] != payload["cases"][0]["protocol_sha256"]:
        raise ValueError("execution lock does not bind the refined case")
    if not payload["execution"]["run_state"]["finished"]:
        raise ValueError("the refined run must be finished")
    _validate_corrected_ledger(payload, a, comparison)


def _validate_corrected_ledger(payload: Mapping[str, Any], assessment: Mapping[str, Any], comparison: Mapping[str, Any]) -> None:
    """Both readings must be present, hash-bound and mutually consistent; the corrected (b) follows from the corrected value and the bound."""

    block = payload["corrected_ledger"]
    if set(block) != {"reread", "cases", "spec", "thresholds"}:
        raise ValueError("corrected_ledger keys do not match the closed schema")
    reread = block["reread"]
    if reread["verdict_recorded"] != payload["verdict"] or reread["verdict_on_corrected_ledger"] not in VERDICTS:
        raise ValueError("corrected-ledger re-read must name the recorded verdict and a declared outcome")
    if not isinstance(reread["sha256"], str) or len(reread["sha256"]) != 64 or not all(bool(v) for v in reread["binding_checks"].values()):
        raise ValueError("corrected-ledger re-read must be hash-bound with every binding check passed")
    b = reread["b_residual_power"]
    bound = block["thresholds"]["acceptance_b"]
    if bound != ACCEPTANCE_B_BOUND or b["bound"] != ACCEPTANCE_B_BOUND or block["thresholds"]["hard_gate"] != 0.05:
        raise ValueError("the acceptance bound and the hard gate may not be changed")
    corrected_value = b["corrected"]["windowed_residual_over_electrode_work"]
    if not isfinite(corrected_value) or b["corrected"]["passed"] != (bool(b["corrected"]["window_complete"]) and corrected_value < bound) or b["passed"] != b["corrected"]["passed"]:
        raise ValueError("corrected (b) does not follow from the corrected value and the bound")
    if b["recorded"]["passed"] != assessment["b_residual_power"]["passed"] or abs(b["recorded"]["windowed_residual_over_electrode_work"] - assessment["b_residual_power"]["windowed_residual_over_electrode_work"]) > 1e-12:
        raise ValueError("recorded (b) in the re-read must be the recorded assessment's")
    expected = ("converged" if assessment["a_plateau"]["passed"] and b["corrected"]["passed"] and comparison["all_within"]
                else "resolution_limited" if assessment["a_plateau"]["passed"] and b["corrected"]["passed"] else "refinement_heating" if assessment["a_plateau"]["passed"] else "no_plateau")
    if reread["verdict_on_corrected_ledger"] != expected:
        raise ValueError("verdict on the corrected ledger does not follow from (a), the corrected (b) and (c)")
    words = reread["verdict_statement"]
    if "corrected ledger" not in words or ("FAILED" in words) == bool(b["corrected"]["passed"]):
        raise ValueError("verdict statement must name the corrected-ledger outcome of (b)")
    if [c["id"] for c in block["cases"]] != [c["id"] for c in payload["cases"]]:
        raise ValueError("corrected-ledger case rows must cover the embedded cases in order")
    for row, case in zip(block["cases"], payload["cases"], strict=True):
        if row["sidecar_sha256"] != case["ledger_corrected"]["sidecar_sha256"] or len(row["sidecar_sha256"]) != 64:
            raise ValueError(f"{case['id']}: corrected-ledger row is not the case's sidecar")
        for key in ("recorded_windowed", "corrected_windowed", "recorded_cumulative", "corrected_cumulative"):
            if row[key] is None or not isfinite(row[key]) or row[key] != case["ledger_corrected"][key]:
                raise ValueError(f"{case['id']}: corrected-ledger {key} is not finite or not the case's")
        if row["acceptance_b_corrected_passes"] != (row["corrected_windowed"] < bound) or row["acceptance_b_recorded_passes"] != (row["recorded_windowed"] < bound):
            raise ValueError(f"{case['id']}: acceptance (b) flags do not follow from the values")
        if case["windowed_residual_corrected_recomputed"] is None or abs(case["windowed_residual_corrected_recomputed"] - row["corrected_windowed"]) > 1e-9 * max(abs(row["corrected_windowed"]), 1e-12):
            raise ValueError(f"{case['id']}: recomputed corrected residual differs from the sidecar")
        if "windowed_residual_corrected_over_electrode_work" not in case["series"]:
            raise ValueError(f"{case['id']}: corrected residual series missing")
    refined_row = block["cases"][0]
    if abs(refined_row["corrected_windowed"] - corrected_value) > 1e-12 * abs(corrected_value):
        raise ValueError("the re-read's corrected (b) value is not the refined case's sidecar value")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIC-2D CFT steady state v4: 33 µm refinement of the 50 µm plateau (preregistered)</title>
<style>
:root{color-scheme:dark;--bg:#07100f;--panel:#0f1c1a;--panel2:#14262380;--text:#eef7f4;--muted:#9bb8b0;--line:#2b4540;--accent:#5ad6c0;--warn:#ffcf67;--red:#ff6b6b;--blue:#58a8ff;--shadow:#0008;--window:#5ad6c022;--window2:#58a8ff22}
[data-theme=light]{color-scheme:light;--bg:#edf5f2;--panel:#fff;--panel2:#f2f8f6;--text:#10231f;--muted:#4f6a63;--line:#bfd3cc;--accent:#087f6e;--warn:#7a5700;--red:#b83232;--blue:#176db5;--shadow:#3452;--window:#087f6e22;--window2:#176db522}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#153b34 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
button,select{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:.55rem;padding:.48rem .7rem}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
header,main,footer{width:min(1500px,calc(100% - 2rem));margin:auto}header{padding:2rem 0 1rem}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(1.9rem,4.5vw,3.6rem);line-height:.98;margin:.2rem 0 .8rem;max-width:1000px}h2{margin:.1rem 0 .8rem;font-size:1.1rem}h3{font-size:.95rem;margin:.8rem 0 .3rem}p{margin:.35rem 0}
.claim{border:1px solid #8b681c;background:#513d1438;color:var(--warn);padding:.8rem 1rem;border-radius:.65rem;font-weight:650}.claim ul{margin:.4rem 0 0 1.1rem;font-weight:500;color:var(--text)}
.reread{border:1px solid var(--red);background:#5a1e1e38;padding:.8rem 1rem;border-radius:.65rem;margin:.8rem 0}.reread b{color:var(--red)}
.verdict{display:flex;gap:.8rem;flex-wrap:wrap;align-items:center;margin:.8rem 0}.verdict .pill{font-size:1.3rem;font-weight:800;padding:.4rem 1rem;border-radius:999px;border:2px solid}.verdict .pill small{display:block;font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.85}.pill.limited{color:var(--warn);border-color:var(--warn)}.pill.converged{color:var(--accent);border-color:var(--accent)}.pill.heating,.pill.none{color:var(--red);border-color:var(--red)}
.chips{display:flex;gap:.4rem;flex-wrap:wrap}.chip{border:1px solid var(--line);border-radius:999px;padding:.15rem .6rem;color:var(--muted);font-size:.85rem}.chip b{color:var(--text)}
.controls{display:flex;gap:.6rem;flex-wrap:wrap;align-items:end;margin:1rem 0}.control{display:grid;gap:.25rem}.control label{color:var(--muted);font-size:.8rem}
.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:.9rem;padding:1rem;box-shadow:0 12px 30px var(--shadow);min-width:0;margin:1rem 0}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.plots .panel{margin:0}.plot{width:100%;height:260px;display:block}.wide{grid-column:1/-1}
.kv{display:grid;grid-template-columns:1fr auto;gap:.22rem .6rem}.kv span{min-width:0;overflow-wrap:anywhere}.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){font-variant-numeric:tabular-nums;text-align:right}h1,h2,h3,p,li,td{overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-weight:600}td,th{padding:.2rem .45rem;border-bottom:1px solid var(--line);vertical-align:top}.ok{color:var(--accent)}.marginal{color:var(--warn)}.bad{color:var(--red)}.num{text-align:right}
.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;color:var(--muted)}code{font-size:.78em;overflow-wrap:anywhere}.small{font-size:.82rem;color:var(--muted)}footer{padding:1rem 0 2.5rem;color:var(--muted)}
.legend{display:flex;gap:.8rem;flex-wrap:wrap;font-size:.85rem}.legend label{display:flex;gap:.3rem;align-items:center;cursor:pointer}.sw{display:inline-block;width:1.1rem;height:.35rem;border-radius:2px}
@media(max-width:900px){.plots{grid-template-columns:1fr}}@media(max-width:520px){header,main,footer{width:min(100% - 1rem,1500px)}.panel{padding:.7rem}}
</style>
</head>
<body>
<header><div class="eyebrow">PIC-MCC · axisymmetric (r,z) · preregistered grid-refinement check · model v1.3 closure · v2.0.3 gates · v2.0.6 ledger correction (post hoc)</div>
<h1>Steady-state v4: is the 50 µm channel plateau resolution-converged? The 33.3 µm refinement answers</h1>
<div class="verdict" id="verdict"></div>
<div id="reread" class="reread" role="note"></div>
<div id="claim" class="claim" role="note"></div>
<div class="controls"><div class="control"><label for="tscale">Time-series x axis</label><select id="tscale"><option value="us">time (µs)</option><option value="transits">ion transits (2.4 µs)</option></select></div><button id="theme" type="button" aria-pressed="false">Light theme</button></div>
<p class="small">Shaded bands on the time series: the trailing-20 % plateau windows of the refined run (teal) and of the 50 µm base (blue); dotted vertical: the 3-transit floor. Dashed horizontals: the declared gates / tolerances. Toggle cases with the legend checkboxes.</p></header>
<main>
<section class="panel"><h2>Predeclared acceptance (a)–(d) — recorded (<code>results/assessment.json</code>) and on the corrected ledger (<code>results/assessment-corrected-ledger.json</code>)</h2><div id="acceptance"></div></section>
<section class="panel"><h2>Energy-ledger correction (model v2.0.6, post hoc): recorded vs corrected residual power per run</h2><div id="ledger"></div></section>
<section class="panel"><h2>Convergence comparison: 50 µm base vs 33.3 µm refinement, with the 50 µm particle-resolution band</h2><div id="comparison"></div></section>
<section class="panel"><h2>Plateau time series</h2><div class="legend" id="legend" aria-label="Case toggles"></div>
<div class="plots" style="margin-top:.8rem">
<div class="panel"><h2>Discharge current I_d</h2><canvas class="plot" id="p_id" role="img" aria-label="Discharge current versus time for the refined run, the base and the band cases"></canvas></div>
<div class="panel"><h2>Exit ion beam current I_beam</h2><canvas class="plot" id="p_ib" role="img" aria-label="Exit ion beam current versus time"></canvas></div>
<div class="panel"><h2>Ionisation rate S</h2><canvas class="plot" id="p_s" role="img" aria-label="Ionisation rate versus time"></canvas></div>
<div class="panel"><h2>Neutral density n_g</h2><canvas class="plot" id="p_ng" role="img" aria-label="Neutral density versus time"></canvas></div>
<div class="panel"><h2>Macro-electron count N_e</h2><canvas class="plot" id="p_ne" role="img" aria-label="Macro-electron count versus time"></canvas></div>
<div class="panel"><h2>Windowed ledger residual / electrode work — recorded (solid, pre-v2.0.6) and corrected (dashed, v2.0.6), recomputed for every case</h2><canvas class="plot" id="p_res" role="img" aria-label="Trailing-window energy residual over electrode work versus time, recorded and corrected ledger, with the 5 percent gate and 2 percent acceptance bound"></canvas></div>
<div class="panel"><h2>Peak Δ/λ_D (refined run: window gate statistic and single-step witness)</h2><canvas class="plot" id="p_deb" role="img" aria-label="Cells per Debye length at the peak node versus time with the soft and hard gates"></canvas></div>
<div class="panel"><h2>Peak ω_pe Δt</h2><canvas class="plot" id="p_wpe" role="img" aria-label="Peak plasma frequency times time step versus time"></canvas></div>
<div class="panel"><h2>Axial profile of max_r n_e(z) (window maps)</h2><canvas class="plot" id="p_axn" role="img" aria-label="Radial maximum of the window-averaged electron density versus axial position for both grids"></canvas></div>
<div class="panel"><h2>Axial profile of T_e at the densest radius (window maps)</h2><canvas class="plot" id="p_axt" role="img" aria-label="Electron temperature at the densest node of each axial position for both grids"></canvas></div>
<div class="panel"><h2>Radial n_e through the peak plane</h2><canvas class="plot" id="p_rn" role="img" aria-label="Radial electron density profile at the axial position of the peak for both grids"></canvas></div>
<div class="panel"><h2>Radial T_e through the peak plane</h2><canvas class="plot" id="p_rt" role="img" aria-label="Radial electron temperature profile at the axial position of the peak for both grids"></canvas></div>
</div></section>
<section class="panel"><h2>Run records (hash-verified)</h2><div id="records"></div></section>
<section class="panel"><h2>Simplifications, protocol and identity</h2><div id="identity"></div></section>
</main><footer>Self-contained offline dashboard generated by <code>modern/visualization/generate_pic2d_cft_steady_state_v4.py</code>. Preregistered resolution-convergence study of a development model: not validated, not a performance prediction. Recorded and corrected-ledger (model v2.0.6, post hoc) readings are shown side by side; the recorded files are unchanged.</footer>
<script id="pic2d-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("pic2d-data").textContent);
const $=id=>document.getElementById(id);let raf=0,xMode="us";
const COLORS=["#5ad6c0","#58a8ff","#ffcf67","#c58bff"];const visible=DATA.cases.map(()=>true);
const fmt=(v,n=4)=>v==null||!isFinite(v)?"–":Number(v).toLocaleString(undefined,{maximumSignificantDigits:n});
const sci=(v,n=3)=>v==null||!isFinite(v)?"–":Number(v).toExponential(n-1);
const pct=(v,n=3,sign=false)=>v==null||!isFinite(v)?"–":(sign&&v>0?"+":"")+fmt(v*100,n)+" %";
const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const A=DATA.assessment,C=DATA.comparison,R=DATA.cases[0],B0=DATA.cases[1],T=DATA.protocol.ion_transit_time_s,CL=DATA.corrected_ledger,RR=CL.reread,RB=RR.b_residual_power;
const verdictClassOf=v=>({converged:"converged",resolution_limited:"limited",refinement_heating:"heating",no_plateau:"none"}[v]);
const verdictClass=verdictClassOf(DATA.verdict),correctedClass=verdictClassOf(RR.verdict_on_corrected_ledger);
$("verdict").innerHTML=`<span class="pill ${verdictClass}"><small>recorded verdict (assessment.json)</small>${DATA.verdict.replaceAll("_"," ")}</span><span class="pill ${correctedClass}"><small>corrected ledger, v2.0.6 post hoc (b) ${RB.corrected.passed?"pass":"FAIL"} ${pct(RB.corrected.windowed_residual_over_electrode_work,3,true)}</small>${RR.verdict_on_corrected_ledger.replaceAll("_"," ")}</span><div class="chips"><span class="chip">50 µm base → <b>${DATA.verdict==="resolution_limited"?"RESOLUTION-LIMITED":DATA.verdict==="converged"?"resolution-converged":"not classified"}</b> (as recorded; base heats at <b>${pct(B0.ledger_corrected.corrected_windowed,3,true)}</b> corrected)</span><span class="chip">(a) plateau <b>${A.a_plateau.passed?"pass":"fail"}</b> · ${fmt(A.a_plateau.ion_transit_times,4)} transits</span><span class="chip">(b) windowed residual recorded <b>${pct(A.b_residual_power.windowed_residual_over_electrode_work,3,true)}</b> → corrected <b>${pct(RB.corrected.windowed_residual_over_electrode_work,3,true)}</b> &lt; +2 %: ${A.b_residual_power.passed?"PASS":"FAIL"} → <b class="${RB.corrected.passed?"ok":"bad"}">${RB.corrected.passed?"PASS":"FAIL"}</b></span><span class="chip">(c) within tolerance <b>${C.rows.filter(r=>r.within).length}/${C.rows.length}</b>${C.failed.length?" · exceeded: "+C.failed.join(", "):""}</span><span class="chip">Δ/λ_D at the peak <b>${fmt(C.debye.refined_window_gate_last,3)}</b> (soft 2.5 ${C.debye.soft_ok?"held":"exceeded"}; base ${fmt(C.debye.reference_cells_per_debye_at_peak,3)})</span><span class="chip">prereg <b>${DATA.protocol.preregistration_commit.slice(0,8)}</b> · one execution</span></div>`;
$("reread").innerHTML=`<strong>Corrected-ledger re-read (post hoc, <code>${RR.file}</code>):</strong> ${RR.verdict_statement}<br><span class="small">${RR.model_version_note}. The recorded verdict stands as the recorded outcome; the predeclared (d) tree applied with the corrected (b) gives <b>${RR.verdict_on_corrected_ledger.replaceAll("_"," ")}</b>. Bounds kept: ${CL.thresholds.kept}. Disallowed wording: ${RR.disallowed_wording.join("; ")}.</span>`;
$("claim").innerHTML=`<strong>Claim boundary:</strong> ${DATA.claim_statement}<ul>${DATA.simplifications.map(s=>`<li>${s}</li>`).join("")}</ul>`;
const okSpan=(ok,txt)=>`<span class="${ok?"ok":"bad"}">${txt}</span>`;
function renderAcceptance(){const a=A.a_plateau,b=A.b_residual_power,pl=a.plateau||{};const drift=v=>v==null?"–":`<span class="${Math.abs(v)<.04?"ok":Math.abs(v)<.05?"marginal":"bad"}">${pct(v,3,true)}</span>`;
const rows=[["(a) plateau",`${okSpan(a.passed,a.passed?"PASS":"FAIL")} — stop <code>${a.stop_reason}</code> at ${fmt(a.ion_transit_times,4)} transits (floor ${DATA.protocol.min_transit_times}; transit = ${sci(T,2)} s: ${DATA.protocol.ion_transit_note}); trailing-20 % drifts I_d ${drift(pl.discharge_current_drift)}, N_e ${drift(pl.electron_count_drift)}, n_g ${drift(pl.neutral_density_drift)} (threshold ${pct(pl.threshold,2)}); triad soft ${okSpan(pl.triad_soft_ok,pl.triad_soft_ok?"ok":"exceeded")}; peak-Debye soft margin ${okSpan(pl.peak_debye_soft_ok,pl.peak_debye_soft_ok?"held":"exceeded")}<br><span class="small">${a.rule}</span>`,`${okSpan(a.passed,a.passed?"PASS":"FAIL")} — unchanged (not a ledger quantity)`],
["(b) residual power",`${okSpan(b.passed,b.passed?"PASS":"FAIL")} — trailing-400 000-step ledger residual / electrode work <b>${pct(b.windowed_residual_over_electrode_work,3,true)}</b> (bound &lt; ${pct(b.bound,2,true)}, one-sided; window complete ${b.window_complete}); cumulative witness ${pct(b.cumulative_witness,3,true)}<br><span class="small">${b.rule}</span><br><span class="small">recorded statistic: ${RB.recorded.statistic}</span>`,`${okSpan(RB.corrected.passed,RB.corrected.passed?"PASS":"FAIL")} — <b>${pct(RB.corrected.windowed_residual_over_electrode_work,3,true)}</b> (same window, same bound &lt; ${pct(RB.bound,2,true)}); cumulative ${pct(RB.corrected.cumulative_witness,3,true)}; omitted inelastic power in the window ${pct(RB.corrected.omitted_inelastic_over_electrode_work_in_window,3,true)}; maximum over complete windows ${pct(RB.corrected.max_over_complete_windows.ratio,3,true)} at ${fmt(RB.corrected.max_over_complete_windows.time_s*1e6,3)} µs; first checkpoint ≥ 2 %: ${RB.corrected.first_checkpoint_at_or_above_bound?fmt(RB.corrected.first_checkpoint_at_or_above_bound.time_s*1e6,3)+" µs":"never"}; 5 % hard gate ${RB.corrected.hard_gate_0p05_would_have_fired?"<b class=\"bad\">would have fired</b>":"never fires"}; ${fmt(RB.corrected.numerical_heating_power_w_in_window*1e3,3)} mW of numerical heating on ${fmt(RB.corrected.electrode_power_w_in_window,3)} W of electrode power<br><span class="small">${RB.corrected.statistic}</span><br><b>${RB.status_change}</b>`],
["(c) convergence",`${okSpan(C.all_within,C.all_within?"ALL WITHIN":"EXCEEDED: "+C.failed.join(", "))} — see the table below<br><span class="small">${A.c_convergence.rule}</span>`,`unchanged (not a ledger quantity); the 50 µm reference itself reads <b>${pct(B0.ledger_corrected.corrected_windowed,3,true)}</b> corrected (recorded ${pct(B0.ledger_corrected.recorded_windowed,3,true)}): it was heating; the 5 % gate would have stopped it at ${B0.ledger_corrected.corrected_gate_0p05_first_checkpoint_time_s!=null?fmt(B0.ledger_corrected.corrected_gate_0p05_first_checkpoint_time_s*1e6,3)+" µs":"never"}`],
["(d) re-classification",`<b>${DATA.verdict.replaceAll("_"," ")}</b> — ${A.d_reclassification}`,`<b>${RR.verdict_on_corrected_ledger.replaceAll("_"," ")}</b> (predeclared tree with the corrected (b)) — ${RR.d_reclassification.corrected_text}<br><span class="small">${RR.d_reclassification.what_stands}</span>`],
["reference consistency",`${Object.values(A.reference_consistency).every(v=>v.agree)?'<span class="ok">7/7</span>':'<span class="bad">disagreement</span>'} — the pinned <code>protocol.reference_run.quantities</code> were re-derived from the v2 base artifacts on disk by the assess stage and again by this generator`,`binding checks ${Object.values(RR.binding_checks).filter(Boolean).length}/${Object.keys(RR.binding_checks).length} — the re-read binds the sidecar, the recorded assessment, the summary and the protocol by byte hash`]];
$("acceptance").innerHTML=`<table aria-label="Predeclared acceptance, recorded and on the corrected ledger"><thead><tr><th></th><th>recorded (<code>assessment.json</code>, ${A.utc})</th><th>corrected ledger (<code>${RR.file}</code>, ${RR.utc})</th></tr></thead><tbody>${rows.map(([k,v,w])=>`<tr><th style="white-space:nowrap">${k}</th><td>${v}</td><td>${w}</td></tr>`).join("")}</tbody></table><p class="small">Assessed ${A.utc} (<code>assessment.json</code> SHA-256 <code>${A.sha256.slice(0,16)}…</code>) with the frozen protocol of the preregistration commit <code>${DATA.protocol.preregistration_commit.slice(0,12)}</code>; re-read ${RR.utc} (<code>${RR.file}</code> SHA-256 <code>${RR.sha256.slice(0,16)}…</code>, <code>${RR.generated_by}</code>). Outcome values: ${Object.entries(DATA.protocol.acceptance.d_reclassification).map(([k,v])=>`<b>${k}</b>: ${v}`).join(" · ")}</p>`}
function renderLedger(){const rows=CL.cases.map(c=>`<tr><td>${c.label}<br><code>${c.results_dir}</code></td><td class="num">${fmt(c.last_time_s*1e6,4)} µs</td><td class="num">${pct(c.recorded_windowed,3,true)}</td><td class="num"><b class="${c.corrected_windowed<CL.thresholds.acceptance_b?"ok":"bad"}">${pct(c.corrected_windowed,3,true)}</b></td><td class="num">${pct(c.omitted_windowed,3,true)}</td><td class="num">${pct(c.recorded_cumulative,3,true)} → ${pct(c.corrected_cumulative,3,true)}</td><td class="num">${pct(c.max_corrected_over_complete_windows.ratio,3,true)} @ ${fmt(c.max_corrected_over_complete_windows.time_s*1e6,3)} µs</td><td class="num">${c.recorded_gate_0p05_first_checkpoint_time_s==null?"never":fmt(c.recorded_gate_0p05_first_checkpoint_time_s*1e6,3)+" µs"} → ${c.corrected_gate_0p05_first_checkpoint_time_s==null?"never":"<b class=\"bad\">"+fmt(c.corrected_gate_0p05_first_checkpoint_time_s*1e6,3)+" µs</b>"}</td><td>${c.acceptance_b_recorded_passes?"pass":"FAIL"} → <b class="${c.acceptance_b_corrected_passes?"ok":"bad"}">${c.acceptance_b_corrected_passes?"pass":"FAIL"}</b></td><td><code>${c.sidecar_sha256.slice(0,12)}…</code></td></tr>`).join("");
$("ledger").innerHTML=`<table aria-label="Energy-ledger correction per run"><thead><tr><th>run</th><th class="num">end</th><th class="num">recorded windowed</th><th class="num">corrected windowed</th><th class="num">omitted inelastic</th><th class="num">cumulative recorded → corrected</th><th class="num">max corrected (complete windows)</th><th class="num">5 % gate fires recorded → corrected</th><th>(b) &lt; +2 % recorded → corrected</th><th>sidecar</th></tr></thead><tbody>${rows}</tbody></table><p class="small">${C.residuals.statistic_note}. Up to model v2.0.5 the ledger's <code>inelastic_loss_j</code> counted macro-events × threshold energy without the macro weight W, so every recorded interval residual was H − L_inel, biased negative by the inelastic power; the sidecars (<code>ledger-corrected.json</code>, <code>python -m cft_revival.pic2d.ledger_recompute</code>) rebuild H = field work + ΔU − electrode work from the recorded series and this generator recomputes the corrected windowed statistic from the same series (it refuses a sidecar it cannot reproduce). Recorded files are unchanged. The three accepted 50 µm plateaus were heating numerically at +7…+13 % of the electrode power (the v2.0.3 5 % gate would have stopped them before their plateau declarations); the 33 µm plateau heats at ${pct(R.ledger_corrected.corrected_windowed,3,true)}, above the 2 % acceptance bound and below the 5 % hard gate. Spec: <code>${CL.spec}</code>.</p>`}
function renderComparison(){const bands=DATA.cases.slice(2);const head=`<tr><th>quantity</th><th class="num">50 µm base</th>${bands.map(b=>`<th class="num">${b.label}</th>`).join("")}<th class="num">33.3 µm v4</th><th class="num">Δ vs base</th><th class="num">tolerance</th><th>within</th></tr>`;
const val=(v,r)=>r.unit==="mA"?fmt(v*r.display_scale,4)+" mA":r.unit==="eV"?fmt(v,4)+" eV":r.unit===""?fmt(v,4):sci(v,4)+" "+r.unit;
const body=C.rows.map(r=>`<tr><td>${r.quantity}</td><td class="num">${val(r.reference,r)}</td>${r.bands.map(b=>`<td class="num">${val(b.value,r)}<br><span class="small">${pct(b.relative_difference,2,true)}</span></td>`).join("")}<td class="num"><b>${val(r.refined,r)}</b></td><td class="num"><b class="${r.within?"ok":"bad"}">${pct(r.relative_difference,3,true)}</b></td><td class="num">±${pct(r.tolerance,2)}</td><td>${r.within?'<span class="ok">yes</span>':'<span class="bad">NO</span>'}</td></tr>`).join("");
const D=C.debye,Rs=C.residuals;const extra=`<tr><td>Δ/λ_D at the peak (window maps; v4: gate statistic)</td><td class="num">${fmt(D.reference_cells_per_debye_at_peak,3)}</td>${D.bands.map(b=>`<td class="num">${fmt(b.cells_per_debye_at_peak,3)}</td>`).join("")}<td class="num"><b>${fmt(D.refined_window_gate_last,3)}</b> (trailing mean ${fmt(D.refined_window_gate_trailing_mean,3)}; maps ${fmt(D.refined_cells_per_debye_at_peak_maps,3)})</td><td class="num">–</td><td class="num">soft 2.5 · hard π</td><td>${D.soft_ok?'<span class="ok">soft held</span>':'<span class="bad">soft exceeded</span>'}</td></tr>
<tr><td>windowed residual / electrode work, RECORDED ledger (trailing 400 000 steps; v2 runs recomputed from their series; pre-v2.0.6 statistic)</td><td class="num">${pct(Rs.reference_windowed_recomputed,3,true)}</td>${Rs.bands.map(b=>`<td class="num">${pct(b.windowed_recomputed,3,true)}</td>`).join("")}<td class="num"><b>${pct(Rs.refined_windowed,3,true)}</b></td><td class="num">–</td><td class="num">&lt; +2 % (gate +5 %)</td><td>${Rs.refined_windowed<Rs.acceptance_bound?'<span class="ok">yes (recorded)</span>':'<span class="bad">NO (recorded)</span>'}</td></tr>
<tr><td>windowed residual / electrode work, CORRECTED ledger (model v2.0.6 sidecars; same window)</td><td class="num"><b class="${Rs.reference_windowed_corrected<Rs.acceptance_bound?"ok":"bad"}">${pct(Rs.reference_windowed_corrected,3,true)}</b></td>${Rs.bands.map(b=>`<td class="num"><b class="${b.windowed_corrected<Rs.acceptance_bound?"ok":"bad"}">${pct(b.windowed_corrected,3,true)}</b></td>`).join("")}<td class="num"><b class="${Rs.refined_windowed_corrected<Rs.acceptance_bound?"ok":"bad"}">${pct(Rs.refined_windowed_corrected,3,true)}</b></td><td class="num">–</td><td class="num">&lt; +2 % (gate +5 %)</td><td>${Rs.refined_windowed_corrected<Rs.acceptance_bound?'<span class="ok">yes (corrected)</span>':'<span class="bad">NO (corrected)</span>'}</td></tr>
<tr><td>cumulative residual / electrode work, recorded → corrected</td><td class="num">${pct(Rs.reference_cumulative,3,true)} → ${pct(Rs.reference_cumulative_corrected,3,true)}</td>${Rs.bands.map(b=>`<td class="num">${pct(b.cumulative,3,true)} → ${pct(b.cumulative_corrected,3,true)}</td>`).join("")}<td class="num">${pct(Rs.refined_cumulative,3,true)} → ${pct(Rs.refined_cumulative_corrected,3,true)}</td><td class="num">–</td><td class="num">witness</td><td>–</td></tr>
<tr><td>peak node (r, z)</td><td class="num">${fmt(B0.peak.r_m*1e3,3)} mm, ${fmt(B0.peak.z_m*1e3,4)} mm</td>${bands.map(b=>`<td class="num">${fmt(b.peak.r_m*1e3,3)}, ${fmt(b.peak.z_m*1e3,4)} mm</td>`).join("")}<td class="num">${fmt(R.peak.r_m*1e3,3)} mm, ${fmt(R.peak.z_m*1e3,4)} mm</td><td class="num">–</td><td class="num">–</td><td>–</td></tr>
<tr><td>plateau: transits · steps · stop</td><td class="num">${fmt(B0.ion_transit_times,3)} · ${B0.steps_completed} · ${B0.stop_reason.replaceAll("_"," ")}</td>${bands.map(b=>`<td class="num">${fmt(b.ion_transit_times,3)} · ${b.steps_completed} · ${b.stop_reason.replaceAll("_"," ")}</td>`).join("")}<td class="num">${fmt(R.ion_transit_times,3)} · ${R.steps_completed} · ${R.stop_reason.replaceAll("_"," ")}</td><td class="num">–</td><td class="num">–</td><td>–</td></tr>
<tr><td>grid · Δt · W · particles at the end (e⁻ + Xe⁺)</td><td class="num">${B0.grid.radial_cells}×${B0.grid.axial_cells} · ${fmt(B0.dt_s*1e12,3)} ps · ${sci(B0.macro_weight,3)} · ${B0.final_counts.electrons}+${B0.final_counts.ions}</td>${bands.map(b=>`<td class="num">${b.grid.radial_cells}×${b.grid.axial_cells} · ${fmt(b.dt_s*1e12,3)} ps · ${sci(b.macro_weight,3)} · ${b.final_counts.electrons}+${b.final_counts.ions}</td>`).join("")}<td class="num">${R.grid.radial_cells}×${R.grid.axial_cells} · ${fmt(R.dt_s*1e12,3)} ps · ${sci(R.macro_weight,3)} · ${R.final_counts.electrons}+${R.final_counts.ions}</td><td class="num">–</td><td class="num">–</td><td>–</td></tr>`;
const cp=DATA.protocol.convergence_pair;$("comparison").innerHTML=`<table aria-label="Convergence comparison"><thead>${head}</thead><tbody>${body}${extra}</tbody></table><p class="small">${DATA.protocol.tolerances_note}</p><p class="small"><strong>Band use:</strong> ${cp.use}. The seed-b case stopped on its wall budget at ${fmt(DATA.cases[2].ion_transit_times,3)} transits (no plateau declaration possible below 3), its values are its own trailing window. Δ/λ_D for the 50 µm cases is Δz / λ_D(peak n_e, T_e,peak) from their window maps (the v2 runs had no runtime Debye gate); for v4 the window-mode gate statistic recorded at the stop is shown with the maps value beside it. The windowed residual of the 50 µm runs is recomputed from their 200-step interval ledgers with the same 400 000-step trailing window (for v4 the recomputation reproduces the runner's value to 1e-12 and the generator refuses otherwise).</p>`}
function renderRecords(){$("records").innerHTML=`<table aria-label="Run records"><thead><tr><th>case</th><th>results</th><th>stop · transits</th><th>steps · time</th><th>wall · ms/step</th><th>sessions · frames</th><th>summary SHA-256</th><th>maps · series SHA-256</th><th>config · git HEAD</th></tr></thead><tbody>${DATA.cases.map(c=>`<tr><td>${c.label}<br><code>${c.id}</code></td><td><code>${c.results_dir}</code></td><td>${c.stop_reason.replaceAll("_"," ")} · ${fmt(c.ion_transit_times,4)}</td><td>${c.steps_completed} · ${fmt(c.simulated_time_s*1e6,4)} µs</td><td>${fmt(c.wall_seconds_total/3600,3)} h · ${fmt(c.ms_per_step_last_session,3)}</td><td>${c.sessions} · ${c.frames==null?"–":c.frames}</td><td><code>${c.summary_sha256.slice(0,16)}…</code></td><td><code>${c.maps_npz_sha256.slice(0,12)}…</code> · <code>${c.series_npz_sha256.slice(0,12)}…</code></td><td><code>${c.config_sha256.slice(0,12)}…</code> · <code>${(c.git_head||"").slice(0,12)}</code></td></tr>`).join("")}</tbody></table><p class="small">Execution lock (O_EXCL, acquired ${DATA.execution.lock.acquired_at_utc}): commit <code>${DATA.execution.lock.commit}</code>, protocol <code>${DATA.execution.lock.protocol_sha256.slice(0,16)}…</code>, configuration <code>${DATA.execution.lock.config_sha256.slice(0,16)}…</code>, host <code>${DATA.execution.lock.host}</code>, PID ${DATA.execution.lock.pid}, clean worktree attested ${DATA.execution.lock.clean_worktree_attested}. Run state: finished ${DATA.execution.run_state.finished}, ${DATA.execution.run_state.sessions.length} session(s), ${fmt(DATA.execution.run_state.wall_seconds_total,6)} s, ${DATA.execution.run_state.frames_written} frames written.</p>`}
function renderIdentity(){$("identity").innerHTML=`<p><span class="badge">status</span> ${DATA.status.replaceAll("_"," ")}</p><p><span class="badge">model</span> ${DATA.model_version}</p><p><span class="badge">plateau rule</span> ${DATA.protocol.plateau_rule}</p><p><span class="badge">wall budget</span> ${DATA.protocol.wall_budget_seconds} s cumulative (used ${fmt(R.wall_seconds_total,6)} s)</p><p><span class="badge">preregistration</span> ${Object.entries(DATA.protocol.preregistration).map(([k,v])=>`<b>${k}</b>: ${v}`).join(" · ")}</p><p><span class="badge">v4 protocol SHA-256</span> <code>${DATA.protocol.file_sha256}</code> (frozen at ${DATA.protocol.preregistration_commit.slice(0,12)})</p><p><span class="badge">v2 protocol SHA-256</span> <code>${DATA.protocol.reference_protocol_sha256}</code> (reference commit ${DATA.protocol.reference_commit})</p><p><span class="badge">assessment SHA-256</span> <code>${A.sha256}</code></p><p><span class="badge">claim boundary (protocol)</span> ${DATA.claim_boundary}</p>`}
function setup(canvas){const r=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);return {c,w:r.width,h:r.height}}
function tick(v,lo,hi){const m=Math.max(Math.abs(lo),Math.abs(hi));return m>=1e5||(m>0&&m<1e-2)?(v===0?"0":Number(v).toExponential(2)):fmt(v,3)}
function axes(c,b,w,h,xlabel,ylabel,xmin,xmax,ymin,ymax,ylog=false){c.strokeStyle=themeColor("--line");c.fillStyle=themeColor("--muted");c.lineWidth=1;c.font="12px system-ui";c.strokeRect(b.l,b.t,b.r-b.l,b.b-b.t);c.textAlign="center";for(let i=0;i<=4;i++){const x=b.l+(b.r-b.l)*i/4;c.fillText(tick(xmin+(xmax-xmin)*i/4,xmin,xmax),x,b.b+18)}c.fillText(xlabel,(b.l+b.r)/2,h-6);c.save();c.translate(13,(b.t+b.b)/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="right";for(let i=0;i<=4;i++){const v=ymax-(ymax-ymin)*i/4;c.fillText(ylog?"1e"+fmt(v,3):tick(v,ymin,ymax),b.l-6,b.t+(b.b-b.t)*i/4+4)}c.textAlign="left"}
function quantile(values,q){const s=[...values].sort((a,b)=>a-b);if(!s.length)return NaN;const k=(s.length-1)*q,i=Math.floor(k);return s[i]+(s[Math.min(i+1,s.length-1)]-s[i])*(k-i)}
function drawPlot(id,series,xLabel,yLabel,log=false,marks={}){const s=setup($(id)),c=s.c,b={l:64,t:16,r:s.w-16,b:s.h-40},pts=series.filter(q=>q&&q.x.length);c.clearRect(0,0,s.w,s.h);c.fillStyle=themeColor("--panel");c.fillRect(0,0,s.w,s.h);if(!pts.length){c.fillStyle=themeColor("--muted");c.fillText("no case selected",b.l+8,b.t+16);return}const all=pts.flatMap(q=>q.y.filter(v=>v!=null&&isFinite(v)&&(!log||v>0))),xmin=marks.xmin!=null?marks.xmin:Math.min(...pts.flatMap(q=>q.x)),xmax=marks.xmax!=null?marks.xmax:Math.max(...pts.flatMap(q=>q.x));let ymin=Math.min(...all),ymax=Math.max(...all);if(marks.robust){ymin=quantile(all,.01);ymax=quantile(all,.99)}if(marks.ymin!=null)ymin=marks.ymin;if(marks.ymax!=null)ymax=marks.ymax;if(log){ymin=Math.log10(Math.max(ymin,1e-300));ymax=Math.log10(Math.max(ymax,1e-299))}else{const pad=(ymax-ymin||1)*.08;ymin-=pad;ymax+=pad}const X=x=>b.l+(x-xmin)/(xmax-xmin||1)*(b.r-b.l),Y=v=>b.b-((log?Math.log10(v):v)-ymin)/(ymax-ymin||1)*(b.b-b.t);
if(marks.bands){marks.bands.forEach(bd=>{const x0=Math.max(b.l,X(bd.x[0])),x1=Math.min(b.r,X(bd.x[1]));c.fillStyle=bd.color;c.fillRect(x0,b.t,x1-x0,b.b-b.t)})}
if(marks.vlines){c.save();c.setLineDash([4,4]);c.strokeStyle=themeColor("--muted");c.lineWidth=1;marks.vlines.forEach(v=>{if(v==null||v<xmin||v>xmax)return;c.beginPath();c.moveTo(X(v),b.t);c.lineTo(X(v),b.b);c.stroke()});c.restore()}
if(marks.hlines){c.save();c.setLineDash([2,4]);c.lineWidth=1;c.font="11px system-ui";marks.hlines.forEach(h=>{const yy=log?Math.log10(h.y):h.y;if(!(yy>=ymin&&yy<=ymax))return;c.strokeStyle=h.color||themeColor("--muted");const py=Y(h.y);c.beginPath();c.moveTo(b.l,py);c.lineTo(b.r,py);c.stroke();c.fillStyle=h.color||themeColor("--muted");c.fillText(h.name,b.r-8-c.measureText(h.name).width,py-3)});c.restore()}
axes(c,b,s.w,s.h,xLabel,yLabel,xmin,xmax,ymin,ymax,log);c.save();c.beginPath();c.rect(b.l,b.t,b.r-b.l,b.b-b.t);c.clip();pts.forEach(q=>{c.strokeStyle=q.color;c.lineWidth=q.width||1.4;if(q.dash)c.setLineDash(q.dash);else c.setLineDash([]);c.beginPath();let started=false;q.x.forEach((x,i)=>{const v=q.y[i];if(v==null||!isFinite(v)||(log&&v<=0)){started=false;return}const px=X(x),py=Y(v);started?c.lineTo(px,py):c.moveTo(px,py);started=true});c.stroke()});c.setLineDash([]);c.restore();c.font="12px system-ui";pts.forEach((q,k)=>{c.fillStyle=q.color;c.fillText(q.name,b.l+8,b.t+14+k*15)});if(marks.robust){c.fillStyle=themeColor("--muted");c.font="10px system-ui";const note="y-range: 1–99 % quantiles (seed transient clipped)";c.fillText(note,b.r-6-c.measureText(note).width,b.b-4)}}
const tx=c=>c.series.time_s.map(v=>xMode==="us"?v*1e6:v/T);
function lines(key,scale=1,name=null){return DATA.cases.map((c,i)=>visible[i]&&c.series[key]?{x:tx(c),y:c.series[key].map(v=>v==null?null:v*scale),name:name||c.label,color:COLORS[i],width:i===0?1.8:1.1}:null)}
function winBand(c,color){const t1=c.simulated_time_s,f=c.plateau&&c.plateau.window_fraction!=null?c.plateau.window_fraction:.2,t0=t1-f*t1;return {x:xMode==="us"?[t0*1e6,t1*1e6]:[t0/T,t1/T],color}}
function drawSeries(){const xl=xMode==="us"?"t (µs)":"ion transits",tm={bands:[winBand(R,themeColor("--window")),winBand(B0,themeColor("--window2"))],vlines:[xMode==="us"?3*T*1e6:3]};
drawPlot("p_id",lines("current_discharge_a",1e3),xl,"I_d (mA)",false,{...tm,robust:true});
drawPlot("p_ib",lines("current_exit_ion_beam_a",1e3),xl,"I_beam,i (mA)",false,{...tm,robust:true});
drawPlot("p_s",lines("current_ionization_rate_per_s"),xl,"S (s⁻¹)",false,{...tm,robust:true});
drawPlot("p_ng",lines("neutral_density_per_m3"),xl,"n_g (m⁻³)",false,{...tm,hlines:[{y:R.neutral_inventory.zero_ionization_density_per_m3,name:"n_g0 = Q_in/c",color:"#ffcf67"}]});
drawPlot("p_ne",lines("electrons"),xl,"macro-electrons",false,tm);
drawPlot("p_res",[...lines("windowed_residual_over_electrode_work",100).map(q=>q&&{...q,name:q.name+" (recorded)"}),...lines("windowed_residual_corrected_over_electrode_work",100).map(q=>q&&{...q,name:q.name+" (corrected, v2.0.6)",dash:[6,4]})],xl,"residual / electrode work (%)",false,{...tm,hlines:[{y:5,name:"v2.0.3 gate +5 %",color:"#ff6b6b"},{y:2,name:"acceptance (b) +2 %",color:"#ffcf67"},{y:0,name:"0",color:themeColor("--muted")}],ymin:-20,ymax:16});
drawPlot("p_deb",[visible[0]&&R.series.peak_node_window_cells_per_debye?{x:tx(R),y:R.series.peak_node_window_cells_per_debye,name:"window gate statistic (400 000-step interval average)",color:COLORS[0],width:1.8}:null,visible[0]&&R.series.peak_node_cells_per_debye?{x:tx(R),y:R.series.peak_node_cells_per_debye,name:"single-step witness",color:"#9bb8b0",width:.8}:null],xl,"Δ/λ_D at the peak",false,{...tm,hlines:[{y:Math.PI,name:"hard π",color:"#ff6b6b"},{y:2.5,name:"soft 2.5",color:"#ffcf67"},{y:C.debye.reference_cells_per_debye_at_peak,name:"50 µm base at its peak (3.17)",color:COLORS[1]}],ymin:0,ymax:3.5});
drawPlot("p_wpe",lines("peak_omega_pe_dt"),xl,"peak ω_pe Δt",false,{...tm,hlines:[{y:.2,name:"gate 0.2",color:"#ff6b6b"}],ymin:0,ymax:.22});
const prof=(key,xkey,scale=1)=>DATA.cases.map((c,i)=>visible[i]?{x:c.profiles[xkey].map(v=>v*1e3),y:c.profiles[key].map(v=>v==null?null:v*scale),name:c.label,color:COLORS[i],width:i===0?1.8:1.1}:null);
const cz=[6.028,12,17.972];
drawPlot("p_axn",prof("axial_peak_n_e_per_m3","z_m"),"z (mm)","max_r n_e (m⁻³)",false,{vlines:cz});
drawPlot("p_axt",prof("axial_peak_t_e_ev","z_m"),"z (mm)","T_e at the densest node (eV)",false,{vlines:cz,robust:true});
drawPlot("p_rn",prof("radial_n_e_at_peak_z_per_m3","r_m"),"r (mm)","n_e at z_peak (m⁻³)");
drawPlot("p_rt",prof("radial_t_e_at_peak_z_ev","r_m"),"r (mm)","T_e at z_peak (eV)",false,{robust:true})}
function renderLegend(){$("legend").innerHTML=DATA.cases.map((c,i)=>`<label><input type="checkbox" data-i="${i}" ${visible[i]?"checked":""}> <span class="sw" style="background:${COLORS[i]}"></span>${c.label} — ${c.grid.radial_cells}×${c.grid.axial_cells}, W ${sci(c.macro_weight,3)}, seed ${c.seed}</label>`).join("");$("legend").querySelectorAll("input").forEach(el=>el.onchange=()=>{visible[Number(el.dataset.i)]=el.checked;schedule()})}
function drawAll(){drawSeries()}
function schedule(){cancelAnimationFrame(raf);raf=requestAnimationFrame(drawAll)}
renderAcceptance();renderLedger();renderComparison();renderRecords();renderIdentity();renderLegend();
$("tscale").onchange=e=>{xMode=e.target.value;schedule()};
$("theme").onclick=()=>{const light=document.documentElement.dataset.theme!=="light";document.documentElement.dataset.theme=light?"light":"dark";$("theme").textContent=light?"Dark theme":"Light theme";$("theme").setAttribute("aria-pressed",light);schedule()};
new ResizeObserver(schedule).observe(document.querySelector("main"));window.addEventListener("pageshow",schedule);drawAll();
</script></body></html>
"""


def fill_template(template: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("__DATA__", encoded)


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    return fill_template(HTML_TEMPLATE, payload)


def anchor_platform_path(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """The platform-fingerprint sidecar written next to the HTML (``<name>.anchor-platform.json``)."""

    return output_path.with_name(output_path.stem + ".anchor-platform.json")


def write_anchor_platform(output_path: Path, html: str) -> Path:
    """Record WHERE the checked-in bytes were generated (byte-exact replay only on this fingerprint; see 79c2a3f8)."""

    record = {
        "schema": "cft-pic2d-dashboard-anchor-platform/1.0.0",
        "html_file": output_path.name,
        "html_sha256": sha256(html.encode("utf-8")).hexdigest(),
        "platform": platform_fingerprint(),
        "policy": "byte-exact replay of the checked-in HTML is asserted only under the same platform fingerprint_sha256; elsewhere the embedded "
                  "payload must agree structurally with numeric leaves within one unit in their last recorded significant digit (rel 1e-9 floor)",
    }
    path = anchor_platform_path(output_path)
    path.write_bytes(json.dumps(record, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    return path


def generate(output_path: Path = DEFAULT_OUTPUT, results: Path = RESULTS, protocol_path: Path = PROTOCOL) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(build_payload(results, protocol_path))
    output_path.write_text(html, encoding="utf-8", newline="\n")
    write_anchor_platform(output_path, html)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate(args.output, args.results, args.protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
