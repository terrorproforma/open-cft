"""Generate the standalone PIC-2D design mini-sweep v1 dashboard: the four preregistered 33 um channel-only designs across the
cusp-strength ratio rho, side by side, with the design-vs-rho trend table the sweep was run for.

Headline: ``modern/experiments/pic2d_design_mini_sweep_v1/results/<design>-channel-33um/`` (the preregistered executions at commit
``291a9227``; 056 launch 2 under amendment 1 ``ee35bc84``): each terminal plateau record is assessed by its predeclared
``assessment.json`` ((a) plateau, (b) windowed residual power on the CORRECTED ledger, model v2.0.6) and its closure targets are read
from ``closure-targets.json`` (RECORDED DATA ONLY: the 0-D plasma-network consumer they were designed for was dropped by the user on
2026-09-04, so the dashboard shows them as per-design measurements).  Every embedded input is hash-verified against its
``.sha256.json`` sidecar and against the hashes the run recorded (protocol, configuration, artifacts); the corrected windowed residual
is recomputed from the series and must reproduce the sidecar; the ss-v4 reference-grid verdict is read from its recorded assessment
AND its corrected-ledger re-read (both readings shown, the recorded one standing).

Interim rows: a design whose canonical results directory has no terminal record yet is shown from its most recent archived
gate-stopped record (``...-launch1-triad-gate-stop/``) flagged ``gate_stopped_interim`` (no assessment, no trend contribution to the
verdict); the sweep verdict stays PROVISIONAL until every primary design has a terminal record.  Re-running the generator once the
canonical directory holds a finished run picks it up.  No timestamps or runtime measurements are added, so identical inputs give
identical bytes on the anchor platform (``<name>.anchor-platform.json``); the page is self-contained (no network access).
"""

from __future__ import annotations

import argparse
import itertools
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

from cft_revival.pic2d.artifacts import platform_fingerprint, read_canonical_json, read_npz
from cft_revival.pic2d.ledger_recompute import SIDECAR_NAME as LEDGER_SIDECAR_NAME
from cft_revival.pic2d.ledger_recompute import corrected_residual, windowed_ratios

EXPERIMENT = MODERN / "experiments" / "pic2d_design_mini_sweep_v1"
RESULTS = EXPERIMENT / "results"
CAMPAIGN_PROTOCOL = EXPERIMENT / "protocol.json"
FIELDS = EXPERIMENT / "fields"
V4_RESULTS = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results"
DEFAULT_OUTPUT = Path(__file__).with_name("pic2d-design-mini-sweep-v1.html")
SCHEMA = "cft-pic2d-design-mini-sweep-v1-visualization/0.1.0"
OPTION = "channel-33um"
# the four primary designs in launch order: (design id, short label, role)
DESIGNS = (
    ("l1a-gs-v2-047-e3196a8aa5", "047", "low-rho"),
    ("divergent-exit-stack", "ref", "reference"),
    ("l1a-gs-v3-009-d0c686b4aa", "009", "mid-rho"),
    ("l1a-gs-v3-056-effcbc8686", "056", "hemp-like"),
)
INTERIM_SUFFIX = "-launch1-triad-gate-stop"
STATUSES = ("plateau", "gate_stopped_interim")
DESIGN_VERDICTS = ("closure_quotable", "plateau_with_heating", "no_plateau")
STOP_REASONS = {"plateau_reached_after_min_transit_times", "wall_clock_budget_reached", "grid_heating_triad_gate_stopped_run"}
LEDGER_SIDECAR_SCHEMA = "cft.pic2d.ledger-corrected/1.0.0"
ACCEPTANCE_B_BOUND = 0.02
RESIDUAL_WINDOW_STEPS = 400_000
MAX_SERIES_POINTS = 2400
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31
# the reference's seed-b / W x 0.7 spread per quantity (the recorded 50 um particle band; acceptance (e) of the sweep: a design effect
# counts only above it) - the same numbers the alpha-series protocol pins as PARTICLE_BAND
REFERENCE_SPREAD = {"discharge_current_a": 0.057, "exit_ion_beam_a": 0.057, "ionization_rate_per_s": 0.046, "gross_utilisation": 0.046,
                    "neutral_density_per_m3": 0.040, "peak_n_e_window_per_m3": 0.119, "t_e_peak_window_ev": 0.093}
QUANTITIES = (
    ("discharge_current_a", "I_d (anode e⁻ − anode Xe⁺, window)", "mA", 1e3),
    ("exit_ion_beam_a", "I_beam,i (exit plane, window)", "mA", 1e3),
    ("ionization_rate_per_s", "S (trailing-20 % mean)", "s⁻¹", 1.0),
    ("gross_utilisation", "utilisation S / Q_in (trailing)", "", 1.0),
    ("neutral_density_per_m3", "n_g (trailing-20 % mean)", "m⁻³", 1.0),
    ("peak_n_e_window_per_m3", "peak n_e (window maps, densest node)", "m⁻³", 1.0),
    ("t_e_peak_window_ev", "T_e at the peak node (window)", "eV", 1.0),
    ("wall_electron_a", "wall electron current (window)", "mA", 1e3),
    ("wall_ion_a", "wall ion current (window)", "mA", 1e3),
    ("ion_wall_loss_fraction", "ion wall loss / ionisation", "", 1.0),
    ("ionisation_centroid_m", "ionisation axial centroid / channel length", "", 1.0),
)
SERIES_KEYS = ("time_s", "step", "electrons", "ions", "current_discharge_a", "current_exit_ion_beam_a", "current_ionization_rate_per_s",
               "current_wall_ion_a", "current_wall_electron_a", "neutral_density_per_m3", "peak_omega_pe_dt", "peak_node_window_cells_per_debye",
               "peak_node_t_e_dense_ev")


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


def _sig(value: float | None, digits: int = 6) -> float | None:
    return None if value is None or not isfinite(float(value)) else float(f"{float(value):.{digits}g}")


def _decimate(values: np.ndarray, stride: int) -> np.ndarray:
    values = np.asarray(values)
    if stride <= 1:
        return values
    out = values[::stride]
    if (values.shape[0] - 1) % stride:
        out = np.concatenate([out, values[-1:]])
    return out


def _lambda_d(n: float, t_ev: float) -> float:
    return sqrt(EPS0 * t_ev * E_CHARGE / (n * E_CHARGE**2))


def windowed_residual(series: Mapping[str, np.ndarray], window_steps: int = RESIDUAL_WINDOW_STEPS) -> np.ndarray:
    """Trailing-window RECORDED ledger residual / electrode work per record (NaN while incomplete) - the runner's statistic."""

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


def corrected_windowed_residual(series: Mapping[str, np.ndarray], window_steps: int = RESIDUAL_WINDOW_STEPS) -> np.ndarray:
    """The v2.0.6 CORRECTED trailing-window residual / electrode work per record (H = field work + dU - electrode work)."""

    corrected, _h, _first = corrected_residual(series)
    ratios = windowed_ratios(np.asarray(series["step"], dtype=np.float64), corrected, np.asarray(series["interval_electrode_work_j"], dtype=np.float64), window_steps)
    return np.where(ratios["complete"], ratios["ratio"], np.nan)


def ledger_sidecar_digest(results: Path, *, series_sha: str, recorded_last: float | None, corrected_last: float | None) -> dict[str, Any]:
    path = results / LEDGER_SIDECAR_NAME
    if not path.is_file():
        raise ValueError(f"{results.name}: {LEDGER_SIDECAR_NAME} is missing - run `python -m cft_revival.pic2d.ledger_recompute {results}`")
    sidecar_sha = _verify_sidecar(path)
    sidecar = read_canonical_json(path)
    if sidecar.get("schema") != LEDGER_SIDECAR_SCHEMA:
        raise ValueError(f"{results.name}: {LEDGER_SIDECAR_NAME} has schema {sidecar.get('schema')!r}")
    if sidecar["inputs"]["series"]["sha256"] != series_sha:
        raise ValueError(f"{results.name}: {LEDGER_SIDECAR_NAME} describes another series")
    end = sidecar["end_state_window"]
    for name, mine, theirs in (("recorded", recorded_last, end["recorded_ratio"]), ("corrected", corrected_last, end["corrected_ratio"])):
        if mine is None or theirs is None or abs(mine - theirs) > 1e-9 * max(abs(theirs), 1e-12):
            raise ValueError(f"{results.name}: the {name} windowed residual recomputed here ({mine}) differs from the sidecar's ({theirs})")
    if end.get("recorded_ratio_matches_summary") is False:
        raise ValueError(f"{results.name}: the sidecar's recorded reading does not match the run's summary")
    gate = sidecar["threshold_crossings"]["0.05"]
    bound = sidecar["threshold_crossings"]["0.02"]
    return {
        "sidecar_sha256": sidecar_sha, "macro_weight": sidecar["parameters"]["macro_weight"], "window_steps": int(sidecar["parameters"]["window_steps"]),
        "recorded_windowed": end["recorded_ratio"], "corrected_windowed": end["corrected_ratio"], "omitted_windowed": end["omitted_ratio"],
        "recorded_cumulative": sidecar["cumulative"]["recorded_over_electrode"], "corrected_cumulative": sidecar["cumulative"]["corrected_over_electrode"],
        "max_corrected_over_complete_windows": sidecar["max_over_complete_windows"]["corrected"],
        "corrected_first_checkpoint_at_or_above_0p02_time_s": None if bound["corrected_first_crossing_at_checkpoint"] is None else bound["corrected_first_crossing_at_checkpoint"]["time_s"],
        "corrected_gate_0p05_first_checkpoint_time_s": None if gate["corrected_first_crossing_at_checkpoint"] is None else gate["corrected_first_crossing_at_checkpoint"]["time_s"],
        "acceptance_b_recorded_passes": bool(end["recorded_ratio"] < ACCEPTANCE_B_BOUND), "acceptance_b_corrected_passes": bool(end["corrected_ratio"] < ACCEPTANCE_B_BOUND),
        "already_w_scaled": bool((sidecar.get("cross_check_vs_final_counts") or {}).get("already_w_scaled")),
    }


def design_topology(design_id: str) -> dict[str, Any]:
    """rho (the conservative interior Koch cusp-strength ratio) and the wall-cusp planes, exactly as the interim sweep panel reads
    them: the material-aware binding under iron when it carries the topology, else the sealed catalogue entry."""

    from visualization.interim_sweep_panel import design_cusps_and_rho, design_r_w_over_l

    found = design_cusps_and_rho(design_id, FIELDS)
    if found["rho"] is None:
        raise ValueError(f"{design_id}: no rho in the field binding or the catalogue")
    return {"rho": float(found["rho"]), "rho_source": found["source"], "cusp_z_m": found["cusp_z_m"], "r_w_over_l": design_r_w_over_l(design_id)}


def ionisation_axial_statistics(maps: Mapping[str, np.ndarray], grid: Mapping[str, Any], channel_length_m: float) -> dict[str, Any]:
    """Axial distribution of the window-averaged ionisation rate (volume-weighted over the axisymmetric nodes): centroid, quartiles, shares."""

    rate = np.asarray(maps["ionization_rate_per_m3_s"], dtype=np.float64)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    dr, dz = float(grid["dr_m"]), float(grid["dz_m"])
    r = np.arange(rate.shape[0]) * dr
    weight_r = np.where(r > 0.0, 2.0 * pi * r * dr, pi * (dr / 2.0) ** 2)      # node volumes per unit z (axis node: the r <= dr/2 disc)
    per_z = (rate * weight_r[:, None]).sum(axis=0) * dz                         # ionisations per second per axial node
    z = np.arange(rate.shape[1]) * dz
    channel = z <= channel_length_m + 0.5 * dz
    total = float(per_z[channel].sum())
    if total <= 0.0:
        raise ValueError("ionisation map carries no events inside the channel")
    cdf = np.cumsum(per_z[channel]) / total
    zc = z[channel]
    quartiles = [float(np.interp(q, cdf, zc)) for q in (0.25, 0.5, 0.75)]
    edges = np.linspace(0.0, channel_length_m, 5)
    shares = [float(per_z[channel][(zc >= a) & ((zc < b) if b < channel_length_m else (zc <= b))].sum() / total) for a, b in itertools.pairwise(edges)]
    return {"centroid_m": float((per_z[channel] * zc).sum() / total), "quartiles_m": quartiles, "quarter_length_shares": shares,
            "fraction_upstream_of_mid_channel": float(cdf[np.searchsorted(zc, 0.5 * channel_length_m)]), "total_per_s": total,
            "profile_z_m": _round(zc, 6), "profile_per_s_per_m": _round(per_z[channel] / dz, 5)}


def _peak_from_maps(maps: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = np.asarray(maps["n_e_per_m3"])
    t = np.asarray(maps["t_e_ev"])
    i, j = np.unravel_index(int(np.nanargmax(n)), n.shape)
    return {"peak_n_e_window_per_m3": float(n[i, j]), "t_e_peak_window_ev": float(t[i, j]), "node": [int(i), int(j)]}


def closure_targets_digest(path: Path) -> dict[str, Any]:
    targets = json.loads(path.read_text(encoding="utf-8"))
    chain = targets["kornfeld_chain"]["cusps_exit_to_anode"]
    return {
        "file_sha256": _file_sha256(path), "cusp_source": targets["cusp_source"],
        "cusps": [{"z_c_m": c["z_c_m"], "electron_wall_current_a": c["electron_wall_current_a"], "ion_wall_current_a": c["ion_wall_current_a"], "sheath_drop_v": c["sheath_drop_v"],
                   "near_wall_drop_v": c["near_wall_drop_v"], "near_wall_electron_temperature_ev": c["near_wall_electron_temperature_ev"], "electron_wall_mean_energy_ev": c["electron_wall_mean_energy_ev"],
                   "leak_width_fwhm_m": c["leak_width_fwhm_m"]} for c in targets["cusps"]],
        "kornfeld_chain_exit_to_anode": [{"z_c_m": k["z_c_m"], "p_transit": k["p_transit"], "electron_wall_current_a": k["electron_wall_current_a"], "je_arriving_a": k["je_arriving_a"]} for k in chain],
        "entering_electron_current_a": targets["kornfeld_chain"]["entering_electron_current_a"], "electron_current_reaching_anode_cell_a": targets["kornfeld_chain"]["electron_current_reaching_anode_cell_a"],
        "cells_anode_to_exit": [{"cell_id": c["cell_id"], "kind": c["kind"], "z_start_m": c["z_start_m"], "z_end_m": c["z_end_m"], "ionisation_share": c["ionisation_share"],
                                 "ion_wall_loss_fraction": c["ion_wall_loss_fraction"], "density_weighted_potential_v": c["density_weighted_potential_v"],
                                 "density_weighted_electron_temperature_ev": c["density_weighted_electron_temperature_ev"], "peak_electron_density_per_m3": c["peak_electron_density_per_m3"]} for c in targets["cells"]],
        "potential_steps_v": targets["potential_steps_v"], "phi_max_v": targets["phi_max_v"], "phi_min_v": targets["phi_min_v"],
        "total_wall_electron_current_a": targets["total_wall_electron_current_a"], "total_wall_ion_current_a": targets["total_wall_ion_current_a"],
        "diffuse_non_cusp_electron_wall_current_a": targets["diffuse_non_cusp_electron_wall_current_a"], "anode_edge_electron_wall_current_a": targets["anode_edge_electron_wall_current_a"],
        "total_ionisation_rate_per_s": targets["total_ionisation_rate_per_s"],
        "note": "RECORDED DATA ONLY: extracted by run.py targets for the plasma-network v2 calibration route the user dropped on 2026-09-04 (no 0-D development); shown as per-design measurements",
    }


def assessment_digest(results: Path, run_summary: Mapping[str, Any]) -> dict[str, Any]:
    path = results / "assessment.json"
    digest = _verify_sidecar(path)
    a = read_canonical_json(path)
    if a["run"]["protocol_sha256"] != run_summary["protocol_sha256"] or a["run"]["config_sha256"] != run_summary["provenance"]["config_sha256"]:
        raise ValueError(f"{results.name}: assessment.json does not describe the embedded run")
    if a["verdict"] not in DESIGN_VERDICTS:
        raise ValueError(f"{results.name}: unknown design verdict {a['verdict']!r}")
    b = a["b_residual_power"]
    return {"sha256": digest, "utc": a["utc"], "verdict": a["verdict"], "verdict_rule": a["verdict_rule"], "a_plateau": a["a_plateau"]["passed"], "a_transits": a["a_plateau"]["ion_transit_times"],
            "b_passed": b["passed"], "b_basis": b["basis"], "b_recorded": b["recorded"]["windowed_residual_over_electrode_work"], "b_recorded_passed": b["recorded"]["passed"],
            "b_corrected": None if b["corrected"] is None else b["corrected"]["windowed_residual_over_electrode_work"], "b_corrected_passed": None if b["corrected"] is None else b["corrected"]["passed"],
            "closure_targets_quotable": a["closure_targets_quotable"], "convergence_statement": a["convergence_statement"], "convergence_statement_corrected_ledger": a["convergence_statement_corrected_ledger"],
            "steady_state_v4_verdict": a["steady_state_v4_verdict"]["verdict"], "steady_state_v4_verdict_on_corrected_ledger": (a["steady_state_v4_verdict"].get("corrected_ledger") or {}).get("verdict_on_corrected_ledger"),
            "design_specific": a.get("design_specific")}


def gate_stop_digest(results: Path) -> dict[str, Any]:
    path = results / "triad-stop-diagnosis.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    return {"file_sha256": _file_sha256(path), "verdict": record["verdict"], "member_drifts_at_stop": record.get("member_drifts_at_stop"),
            "gate_reading_at_291a9227": record.get("gate_reading_at_291a9227"), "heating_signature": record.get("heating_signature"),
            "note": "launch 1 stopped by the pre-v2.0.4 RAW single-node omega_pe dt drift member (shot-noise artefact, README section 8.5); launch 2 (amendment 1) runs in the canonical directory"}


def build_design(design_id: str, label: str, role: str, results_root: Path = RESULTS) -> dict[str, Any]:
    canonical = results_root / f"{design_id}-{OPTION}"
    interim = results_root / f"{design_id}-{OPTION}{INTERIM_SUFFIX}"
    if (canonical / "summary.json").is_file() and (canonical / "assessment.json").is_file():
        results, status = canonical, "plateau"
    elif (interim / "summary.json").is_file():
        results, status = interim, "gate_stopped_interim"
    else:
        raise ValueError(f"{design_id}: no record under {canonical.name} (terminal) or {interim.name} (interim)")
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_sha = _verify_sidecar(summary_path)
    protocol_path = results / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = _file_sha256(protocol_path)
    if summary["protocol_sha256"] != protocol_sha:
        raise ValueError(f"{results.name}: protocol drift - the run recorded {summary['protocol_sha256'][:12]}, the file hashes {protocol_sha[:12]}")
    maps_sha = _verify_sidecar(results / "maps.npz")
    series_sha = _verify_sidecar(results / "series.npz")
    if summary["artifacts"]["maps_npz_sha256"] != maps_sha or summary["artifacts"]["series_npz_sha256"] != series_sha:
        raise ValueError(f"{results.name}: artifact hashes differ from the ones the run recorded")
    _verify_sidecar(results / "run_state.json")
    _verify_sidecar(results / "checkpoint-final.json")
    run_state = read_canonical_json(results / "run_state.json")
    lock = json.loads((results / "execution-lock.json").read_text(encoding="utf-8"))
    if lock["commit"] != summary["git_head"] or lock["protocol_sha256"] != summary["protocol_sha256"] or lock["config_sha256"] != summary["provenance"]["config_sha256"]:
        raise ValueError(f"{results.name}: execution lock does not bind the run")
    if protocol["design_id"] != design_id or protocol["option"] != OPTION:
        raise ValueError(f"{results.name}: protocol names {protocol['design_id']} / {protocol['option']}")
    maps = read_npz(results / "maps.npz")
    series = read_npz(results / "series.npz")
    grid = summary["provenance"]["config"]["grid"]
    geometry = protocol["geometry"]
    channel_length_m = float(geometry["z_max_m"]) - float(geometry["z_min_m"])
    peak = _peak_from_maps(maps)
    currents = summary["window_currents_a"]
    inventory = summary["neutral_inventory"]
    ionisation = ionisation_axial_statistics(maps, grid, channel_length_m)
    quantities = {
        "discharge_current_a": float(currents["discharge_a"]), "exit_ion_beam_a": float(currents["exit_ion_beam_a"]),
        "ionization_rate_per_s": float(inventory["trailing_20pct_mean_ionization_rate_per_s"]), "gross_utilisation": float(inventory["propellant_utilisation_trailing"]),
        "neutral_density_per_m3": float(inventory["trailing_20pct_mean_density_per_m3"]), "peak_n_e_window_per_m3": peak["peak_n_e_window_per_m3"], "t_e_peak_window_ev": peak["t_e_peak_window_ev"],
        "wall_electron_a": float(currents["wall_electron_a"]), "wall_ion_a": float(currents["wall_ion_a"]),
        "ion_wall_loss_fraction": float(currents["wall_ion_a"]) / (E_CHARGE * float(inventory["trailing_20pct_mean_ionization_rate_per_s"])),
        "ionisation_centroid_m": ionisation["centroid_m"] / channel_length_m,
    }
    stride = max(1, -(-len(series["time_s"]) // MAX_SERIES_POINTS))
    decimated = {key: _round(_decimate(series[key], stride), 6) for key in SERIES_KEYS if key in series}
    residual_series = windowed_residual(series)
    decimated["windowed_residual_over_electrode_work"] = _round(_decimate(residual_series, stride), 5)
    residual_last = float(residual_series[-1]) if residual_series.size and isfinite(float(residual_series[-1])) else None
    corrected_series = corrected_windowed_residual(series)
    decimated["windowed_residual_corrected_over_electrode_work"] = _round(_decimate(corrected_series, stride), 5)
    corrected_last = float(corrected_series[-1]) if corrected_series.size and isfinite(float(corrected_series[-1])) else None
    triad = summary.get("grid_heating_triad") or {}
    recorded = triad.get("windowed_energy_residual_over_electrode_work")
    if recorded is not None and residual_last is not None and abs(residual_last - recorded) > 1e-6 * max(abs(recorded), 1e-12):
        raise ValueError(f"{results.name}: recomputed windowed residual {residual_last} differs from the runner's {recorded}")
    ledger = ledger_sidecar_digest(results, series_sha=series_sha, recorded_last=residual_last, corrected_last=corrected_last)
    n_e = np.asarray(maps["n_e_per_m3"], dtype=np.float64)
    finite = np.where(np.isfinite(n_e), n_e, -np.inf)
    axial_peak = finite.max(axis=0)
    node_z = np.arange(n_e.shape[1]) * float(grid["dz_m"])
    topology = design_topology(design_id)
    debye = (summary.get("peak_node_debye") or {}).get("window") or {}
    dz = float(grid["dz_m"])
    case = {
        "id": design_id, "label": label, "role": role, "status": status, "results_dir": results.name, "experiment_id": summary["experiment_id"], "rho": topology["rho"],
        "rho_source": topology["rho_source"], "cusp_z_m": topology["cusp_z_m"], "r_w_over_l": topology["r_w_over_l"], "channel_length_m": channel_length_m,
        "bore_radius_m": float(geometry["bore_radius_m"]), "git_head": summary["git_head"], "protocol_sha256": protocol_sha, "config_sha256": summary["provenance"]["config_sha256"],
        "summary_sha256": summary_sha, "maps_npz_sha256": maps_sha, "series_npz_sha256": series_sha, "lock": {k: lock[k] for k in ("commit", "protocol_sha256", "config_sha256", "acquired_at_utc", "pid", "host")},
        "stop_reason": summary["stop_reason"], "steps_completed": int(summary["steps_completed"]), "simulated_time_s": float(summary["simulated_time_s"]),
        "ion_transit_times": float(summary["ion_transit_times"]), "wall_seconds_total": float(summary["wall_seconds_total"]), "ms_per_step_last_session": summary.get("ms_per_step_this_session"),
        "sessions": len(summary.get("sessions") or []), "frames": (summary["artifacts"].get("frames") or {}).get("count"), "finished": bool(run_state["finished"]),
        "grid": {"radial_cells": int(grid["radial_cells"]), "axial_cells": int(grid["axial_cells"]), "dr_m": float(grid["dr_m"]), "dz_m": dz},
        "dt_s": float(summary["provenance"]["config"]["dt_s"]), "macro_weight": float(summary["provenance"]["config"]["macro_weight"]), "seed": int(summary["case"]["seed"]),
        "final_counts": summary["final_counts"], "plateau": summary.get("plateau"), "window_currents_a": currents,
        "neutral_inventory": {k: inventory.get(k) for k in ("trailing_20pct_mean_density_per_m3", "trailing_20pct_analytic_fixed_point_per_m3", "trailing_20pct_mean_ionization_rate_per_s",
                                                            "propellant_utilisation_trailing", "net_utilisation_trailing", "feed_atoms_per_s", "zero_ionization_density_per_m3")},
        "grid_heating_triad": {k: triad.get(k) for k in ("windowed_energy_residual_over_electrode_work", "windowed_energy_residual_window_complete", "energy_residual_over_electrode_work",
                                                        "ionisation_rate_drift", "t_e_dense_drift", "omega_pe_dt_drift", "soft_ok", "hard_failures")},
        "peak_debye_window": {"cells_per_debye_window_last": debye.get("cells_per_debye_window_last"), "trailing_mean": debye.get("trailing_20pct_mean_cells_per_debye_window"),
                              "max": debye.get("max_cells_per_debye_window"), "soft_ok": debye.get("soft_ok")},
        "quantities": quantities,
        "peak": {"node": peak["node"], "r_m": peak["node"][0] * float(grid["dr_m"]), "z_m": peak["node"][1] * dz, "cells_per_debye_maps": dz / _lambda_d(peak["peak_n_e_window_per_m3"], peak["t_e_peak_window_ev"])},
        "ionisation": {k: ionisation[k] for k in ("centroid_m", "quartiles_m", "quarter_length_shares", "fraction_upstream_of_mid_channel", "total_per_s")},
        "windowed_residual_recomputed": residual_last, "windowed_residual_corrected_recomputed": corrected_last, "ledger_corrected": ledger,
        "assessment": assessment_digest(results, summary) if status == "plateau" else None,
        "gate_stop": gate_stop_digest(results) if status == "gate_stopped_interim" else None,
        "closure_targets": closure_targets_digest(results / "closure-targets.json") if (results / "closure-targets.json").is_file() else None,
        "series_stride": stride, "series": decimated,
        "profiles": {"z_m": _round(node_z, 6), "z_over_channel": _round(node_z / channel_length_m, 6), "axial_peak_n_e_per_m3": _round(np.where(np.isfinite(axial_peak), axial_peak, np.nan), 5),
                     "ionisation_z_m": ionisation["profile_z_m"], "ionisation_per_s_per_m": ionisation["profile_per_s_per_m"]},
    }
    return case


def reference_grid_verdict() -> dict[str, Any]:
    """The ss-v4 (50 -> 33 um) verdict as recorded and on the corrected ledger, hash-verified (the sweep's convergence caveat (f))."""

    assessment_path = V4_RESULTS / "assessment.json"
    reread_path = V4_RESULTS / "assessment-corrected-ledger.json"
    if not assessment_path.is_file() or not reread_path.is_file():
        raise ValueError("the steady-state v4 assessment and its corrected-ledger re-read must be present")
    a_sha = _verify_sidecar(assessment_path)
    r_sha = _verify_sidecar(reread_path)
    a = read_canonical_json(assessment_path)
    r = read_canonical_json(reread_path)
    if r["inputs"]["assessment"]["sha256"] != a_sha:
        raise ValueError("the v4 corrected-ledger re-read does not bind the v4 assessment on disk")
    return {"assessment_sha256": a_sha, "reread_sha256": r_sha, "verdict_recorded": a["verdict"], "verdict_on_corrected_ledger": r["verdict_on_corrected_ledger"],
            "b_recorded": r["b_residual_power"]["recorded"]["windowed_residual_over_electrode_work"], "b_corrected": r["b_residual_power"]["corrected"]["windowed_residual_over_electrode_work"],
            "verdict_statement": r["verdict_statement"], "commit": a["run"]["git_head"]}


def build_trend(designs: list[Mapping[str, Any]]) -> dict[str, Any]:
    reference = next(d for d in designs if d["role"] == "reference")
    rows = []
    for key, label, unit, scale in QUANTITIES:
        ref = reference["quantities"][key]
        spread = REFERENCE_SPREAD.get(key)
        values = []
        for d in designs:
            value = d["quantities"][key]
            rel = (value - ref) / abs(ref) if ref else None
            values.append({"id": d["id"], "label": d["label"], "rho": d["rho"], "status": d["status"], "value": value, "relative_to_reference": rel,
                           "above_reference_spread": None if spread is None or rel is None or d["role"] == "reference" else bool(abs(rel) > spread)})
        rows.append({"key": key, "quantity": label, "unit": unit, "display_scale": scale, "reference_spread": spread, "values": values})
    terminal = [d for d in designs if d["status"] == "plateau"]
    return {"rows": rows, "ordered_by_rho": [d["id"] for d in designs], "terminal_designs": [d["id"] for d in terminal], "interim_designs": [d["id"] for d in designs if d["status"] != "plateau"],
            "spread_note": "reference spread = the recorded 50 um seed-b / W x 0.7 pair (acceptance (e): a design effect counts only above it); the interim (gate-stopped) design carries no trend contribution"}


def build_payload(results: Path = RESULTS, campaign_protocol: Path = CAMPAIGN_PROTOCOL) -> dict[str, Any]:
    campaign = json.loads(campaign_protocol.read_text(encoding="utf-8"))
    designs = sorted((build_design(design_id, label, role, results) for design_id, label, role in DESIGNS), key=lambda d: d["rho"])
    prereg_commits = {d["lock"]["commit"] for d in designs if d["status"] == "plateau"}
    if len(prereg_commits) != 1:
        raise ValueError(f"the terminal designs must share one preregistration commit; found {sorted(prereg_commits)}")
    prereg_commit = prereg_commits.pop()
    run_protocol = json.loads((results / designs[[d["role"] for d in designs].index("reference")]["results_dir"] / "protocol.json").read_text(encoding="utf-8"))
    acceptance = run_protocol["stopping_rule"]["acceptance"]
    v4 = reference_grid_verdict()
    trend = build_trend(designs)
    terminal, interim = trend["terminal_designs"], trend["interim_designs"]
    reference = next(d for d in designs if d["role"] == "reference")
    verdicts = {d["id"]: (d["assessment"] or {}).get("verdict") for d in designs}
    sweep = {
        "terminal_records": len(terminal), "primary_designs": len(designs), "provisional": bool(interim), "interim": interim,
        "per_design_verdicts": verdicts,
        "statement": (f"{len(terminal)} of {len(designs)} primary designs have terminal plateau records; " + (f"{', '.join(interim)} shown from an archived gate-stopped record (launch 2 running) - the sweep-wide reading is PROVISIONAL" if interim else "every primary design has a terminal record")
                      + f". Per-design verdicts: {', '.join(f'{k} {v}' for k, v in verdicts.items() if v)}. The reference design heats at {100.0 * reference['ledger_corrected']['corrected_windowed']:+.2f} % on the corrected ledger "
                      f"(acceptance (b) FAIL, as the ss-v4 record), so the sweep's own reference is not a clean plateau and every 33 um value is 'at 33 um, uncertified'."),
    }
    statement = (
        f"Preregistered design mini-sweep (four channel-only designs across the cusp-strength ratio rho, 33.3 um / 1.4 ps / W parity, the v1.3 closure and the v2.0.3 gates; commit {prereg_commit[:8]}, "
        f"amendment 1 for design 056's gate reading) of a development PIC-MCC model; one seed per design. Verdicts per design as recorded in their assessment.json ((a) plateau, (b) windowed residual power "
        f"on the corrected ledger, model v2.0.6 post hoc): {', '.join(f'{k} {v}' for k, v in verdicts.items() if v)}. Reference-grid caveat (f): the ss-v4 33 um refinement reads "
        f"{v4['verdict_recorded'].replace('_', ' ')} as recorded and {v4['verdict_on_corrected_ledger'].replace('_', ' ')} on the corrected ledger ((b) {100.0 * v4['b_corrected']:+.2f} %), so every value "
        f"here is 'at 33 um, uncertified': the resolved numbers with no grid band of their own; the design-vs-rho trend is a property of this closure and this operating point. "
        f"{'PROVISIONAL: ' + ', '.join(interim) + ' is shown from its archived gate-stopped launch-1 record while launch 2 runs; it carries no trend contribution. ' if interim else ''}"
        f"Closure targets are recorded data only (the 0-D consumer was dropped). Not validated against experiment; not a thruster performance prediction; the neutral transient is artificial and only the fixed point is physical."
    )
    return {
        "schema": SCHEMA, "experiment_id": campaign["experiment_id"], "status": campaign["status"], "option": OPTION, "model_version": run_protocol["model_version"],
        "claim_boundary": run_protocol["claim_boundary"], "claim_statement": statement, "simplifications": list(run_protocol["simplifications"]),
        "protocol": {"campaign_file_sha256": _file_sha256(campaign_protocol), "preregistration_commit": prereg_commit, "acceptance": acceptance, "plateau_rule": run_protocol["stopping_rule"]["plateau"],
                     "min_transit_times": run_protocol["stopping_rule"]["min_transit_times"], "replication_policy": campaign["replication_policy"],
                     "amendments": [{"case": a.get("case"), "changes": a.get("changes")} for a in campaign.get("amendments") or []],
                     "closure_targets_declared": [{"name": t["name"], "per": t["per"], "role": t["role"]} for t in campaign["closure_targets"]]},
        "reference_grid_verdict": v4, "sweep": sweep, "trend": trend, "designs": designs,
    }


def validate_payload(payload: Mapping[str, Any]) -> None:
    required = {"schema", "experiment_id", "status", "option", "model_version", "claim_boundary", "claim_statement", "simplifications", "protocol", "reference_grid_verdict", "sweep", "trend", "designs"}
    if set(payload) != required:
        raise ValueError("payload keys do not match the closed schema")
    if payload["schema"] != SCHEMA or payload["option"] != OPTION:
        raise ValueError("unsupported payload schema or option")
    statement = payload["claim_statement"].lower()
    if not payload["simplifications"] or "not validated" not in statement or "preregistered" not in statement or "uncertified" not in statement:
        raise ValueError("claim boundary must be explicit, name the preregistration and the 33 um caveat")
    if "recorded data only" not in statement:
        raise ValueError("claim boundary must state that closure targets are recorded data only")
    designs = payload["designs"]
    if [d["id"] for d in designs] != [d for d in payload["trend"]["ordered_by_rho"]] or len(designs) != len(DESIGNS):
        raise ValueError("designs must be the four primary designs ordered by rho")
    if any(a["rho"] > b["rho"] for a, b in itertools.pairwise(designs)):
        raise ValueError("designs are not ordered by rho")
    if sum(d["role"] == "reference" for d in designs) != 1:
        raise ValueError("exactly one reference design")
    for d in designs:
        if d["status"] not in STATUSES or d["stop_reason"] not in STOP_REASONS:
            raise ValueError(f"{d['id']}: unknown status / stop reason")
        for key in ("summary_sha256", "maps_npz_sha256", "series_npz_sha256", "protocol_sha256", "config_sha256"):
            if not isinstance(d[key], str) or len(d[key]) != 64:
                raise ValueError(f"{d['id']}: {key} must be a SHA-256")
        if d["lock"]["commit"] != d["git_head"] or d["lock"]["protocol_sha256"] != d["protocol_sha256"] or d["lock"]["config_sha256"] != d["config_sha256"]:
            raise ValueError(f"{d['id']}: execution lock does not bind the run")
        if not d["finished"]:
            raise ValueError(f"{d['id']}: run not finished")
        n = len(d["series"]["time_s"])
        if any(len(v) != n for v in d["series"].values()) or "windowed_residual_corrected_over_electrode_work" not in d["series"]:
            raise ValueError(f"{d['id']}: series lengths differ or the corrected residual series is missing")
        for key, *_ in QUANTITIES:
            if not isfinite(d["quantities"][key]):
                raise ValueError(f"{d['id']}: quantity {key} is not finite")
        lc = d["ledger_corrected"]
        if len(lc["sidecar_sha256"]) != 64 or lc["acceptance_b_corrected_passes"] != (lc["corrected_windowed"] < ACCEPTANCE_B_BOUND) or lc["acceptance_b_recorded_passes"] != (lc["recorded_windowed"] < ACCEPTANCE_B_BOUND):
            raise ValueError(f"{d['id']}: corrected-ledger flags do not follow from the values")
        if d["windowed_residual_corrected_recomputed"] is None or abs(d["windowed_residual_corrected_recomputed"] - lc["corrected_windowed"]) > 1e-9 * max(abs(lc["corrected_windowed"]), 1e-12):
            raise ValueError(f"{d['id']}: recomputed corrected residual differs from the sidecar")
        if d["status"] == "plateau":
            a = d["assessment"]
            if a is None or d["gate_stop"] is not None or d["stop_reason"] != "plateau_reached_after_min_transit_times" or d["ion_transit_times"] < payload["protocol"]["min_transit_times"]:
                raise ValueError(f"{d['id']}: a plateau design needs an assessment, a plateau stop and >= the transit floor")
            expected_b = a["b_corrected"] if a["b_corrected"] is not None else a["b_recorded"]
            b_ok = expected_b < ACCEPTANCE_B_BOUND
            expected = "closure_quotable" if a["a_plateau"] and b_ok else "plateau_with_heating" if a["a_plateau"] else "no_plateau"
            if a["verdict"] != expected or a["b_passed"] != b_ok or a["closure_targets_quotable"] != (a["verdict"] == "closure_quotable"):
                raise ValueError(f"{d['id']}: assessment verdict does not follow from (a) and the decisive (b)")
            if a["b_corrected"] is not None and abs(a["b_corrected"] - lc["corrected_windowed"]) > 1e-9 * max(abs(lc["corrected_windowed"]), 1e-12):
                raise ValueError(f"{d['id']}: the assessment's corrected (b) is not the sidecar's")
            if d["closure_targets"] is None:
                raise ValueError(f"{d['id']}: a plateau design carries its closure targets")
        else:
            if d["assessment"] is not None or d["gate_stop"] is None or d["stop_reason"] == "plateau_reached_after_min_transit_times":
                raise ValueError(f"{d['id']}: an interim design has a gate-stop digest and no assessment")
    sweep = payload["sweep"]
    interim = [d["id"] for d in designs if d["status"] != "plateau"]
    if sweep["interim"] != interim or sweep["provisional"] != bool(interim) or sweep["terminal_records"] != len(designs) - len(interim) or sweep["primary_designs"] != len(designs):
        raise ValueError("sweep block does not describe the designs")
    if bool(interim) != ("provisional" in statement):
        raise ValueError("the claim statement must say PROVISIONAL exactly when an interim design is shown")
    if sweep["per_design_verdicts"] != {d["id"]: (d["assessment"] or {}).get("verdict") for d in designs}:
        raise ValueError("sweep verdicts are not the designs' assessment verdicts")
    trend = payload["trend"]
    if [r["key"] for r in trend["rows"]] != [q[0] for q in QUANTITIES]:
        raise ValueError("trend rows must cover the quantities in order")
    reference = next(d for d in designs if d["role"] == "reference")
    for row in trend["rows"]:
        if [v["id"] for v in row["values"]] != [d["id"] for d in designs]:
            raise ValueError(f"{row['key']}: trend values must cover the designs in rho order")
        for v, d in zip(row["values"], designs, strict=True):
            if v["value"] != d["quantities"][row["key"]] or v["status"] != d["status"]:
                raise ValueError(f"{row['key']}: trend value is not the design's")
            ref = reference["quantities"][row["key"]]
            if v["relative_to_reference"] is not None and abs(v["relative_to_reference"] - (v["value"] - ref) / abs(ref)) > 1e-12:
                raise ValueError(f"{row['key']}: relative shift is not vs the reference")
            if row["reference_spread"] is not None and d["role"] != "reference" and v["above_reference_spread"] != (abs(v["relative_to_reference"]) > row["reference_spread"]):
                raise ValueError(f"{row['key']}: spread flag does not follow")
    v4 = payload["reference_grid_verdict"]
    if len(v4["assessment_sha256"]) != 64 or len(v4["reread_sha256"]) != 64 or v4["verdict_on_corrected_ledger"] not in ("converged", "resolution_limited", "refinement_heating", "no_plateau"):
        raise ValueError("reference-grid verdict must be hash-bound and a declared outcome")
    if v4["verdict_recorded"].replace("_", " ") not in statement or v4["verdict_on_corrected_ledger"].replace("_", " ") not in statement:
        raise ValueError("claim statement must carry both reference-grid readings")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIC-2D design mini-sweep v1: four designs across rho at 33 µm (preregistered)</title>
<style>
:root{color-scheme:dark;--bg:#07100f;--panel:#0f1c1a;--panel2:#14262380;--text:#eef7f4;--muted:#9bb8b0;--line:#2b4540;--accent:#5ad6c0;--warn:#ffcf67;--red:#ff6b6b;--blue:#58a8ff;--shadow:#0008;--window:#5ad6c022}
[data-theme=light]{color-scheme:light;--bg:#edf5f2;--panel:#fff;--panel2:#f2f8f6;--text:#10231f;--muted:#4f6a63;--line:#bfd3cc;--accent:#087f6e;--warn:#7a5700;--red:#b83232;--blue:#176db5;--shadow:#3452;--window:#087f6e22}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#153b34 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
button,select{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:.55rem;padding:.48rem .7rem}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
header,main,footer{width:min(1500px,calc(100% - 2rem));margin:auto}header{padding:2rem 0 1rem}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(1.9rem,4.5vw,3.4rem);line-height:.98;margin:.2rem 0 .8rem;max-width:1000px}h2{margin:.1rem 0 .8rem;font-size:1.1rem}h3{font-size:.95rem;margin:.8rem 0 .3rem}p{margin:.35rem 0}
.claim{border:1px solid #8b681c;background:#513d1438;color:var(--warn);padding:.8rem 1rem;border-radius:.65rem;font-weight:650}.claim ul{margin:.4rem 0 0 1.1rem;font-weight:500;color:var(--text)}
.provisional{border:1px solid var(--blue);background:#1e3a5a38;padding:.8rem 1rem;border-radius:.65rem;margin:.8rem 0}.provisional b{color:var(--blue)}
.verdict{display:flex;gap:.8rem;flex-wrap:wrap;align-items:center;margin:.8rem 0}.verdict .pill{font-size:1.15rem;font-weight:800;padding:.4rem 1rem;border-radius:999px;border:2px solid}.verdict .pill small{display:block;font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.85}.pill.quotable{color:var(--accent);border-color:var(--accent)}.pill.heating{color:var(--warn);border-color:var(--warn)}.pill.none,.pill.interim{color:var(--red);border-color:var(--red)}.pill.interim{color:var(--blue);border-color:var(--blue)}
.chips{display:flex;gap:.4rem;flex-wrap:wrap}.chip{border:1px solid var(--line);border-radius:999px;padding:.15rem .6rem;color:var(--muted);font-size:.85rem}.chip b{color:var(--text)}
.controls{display:flex;gap:.6rem;flex-wrap:wrap;align-items:end;margin:1rem 0}.control{display:grid;gap:.25rem}.control label{color:var(--muted);font-size:.8rem}
.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:.9rem;padding:1rem;box-shadow:0 12px 30px var(--shadow);min-width:0;margin:1rem 0}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.plots .panel{margin:0}.plot{width:100%;height:260px;display:block}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-weight:600}td,th{padding:.2rem .45rem;border-bottom:1px solid var(--line);vertical-align:top}.ok{color:var(--accent)}.marginal{color:var(--warn)}.bad{color:var(--red)}.num{text-align:right}.dim{color:var(--muted)}
.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;color:var(--muted)}code{font-size:.78em;overflow-wrap:anywhere}.small{font-size:.82rem;color:var(--muted)}footer{padding:1rem 0 2.5rem;color:var(--muted)}h1,h2,h3,p,li,td{overflow-wrap:anywhere}
.legend{display:flex;gap:.8rem;flex-wrap:wrap;font-size:.85rem}.legend label{display:flex;gap:.3rem;align-items:center;cursor:pointer}.sw{display:inline-block;width:1.1rem;height:.35rem;border-radius:2px}
@media(max-width:900px){.plots{grid-template-columns:1fr}}@media(max-width:520px){header,main,footer{width:min(100% - 1rem,1500px)}.panel{padding:.7rem}}
</style>
</head>
<body>
<header><div class="eyebrow">PIC-MCC · axisymmetric (r,z) · preregistered design mini-sweep · 33.3 µm channel-only · v1.3 closure · v2.0.3 gates · v2.0.6 ledger correction (post hoc)</div>
<h1>Design mini-sweep v1: what changes with the cusp-strength ratio ρ at 33 µm?</h1>
<div class="verdict" id="verdict"></div>
<div id="provisional" class="provisional" role="note"></div>
<div id="claim" class="claim" role="note"></div>
<div class="controls"><div class="control"><label for="tscale">Time-series x axis</label><select id="tscale"><option value="us">time (µs)</option><option value="transits">design ion transits</option></select></div><button id="theme" type="button" aria-pressed="false">Light theme</button></div>
<p class="small">Designs ordered by ρ. Shaded band on the time series: the reference's trailing-20 % plateau window; dotted vertical: the 3-transit floor (in transit units). Dashed horizontals: the declared gates / bounds. Toggle designs with the legend checkboxes.</p></header>
<main>
<section class="panel"><h2>Design-vs-ρ trend table (trailing-window plateau values; interim design flagged)</h2><div id="trend"></div></section>
<section class="panel"><h2>Predeclared acceptance per design — (a) plateau, (b) windowed residual power on the CORRECTED ledger (decisive) with the recorded reading beside it, verdict</h2><div id="acceptance"></div></section>
<section class="panel"><h2>Energy-ledger correction (model v2.0.6, post hoc): recorded vs corrected residual power per run</h2><div id="ledger"></div></section>
<section class="panel"><h2>Closure targets per design — RECORDED DATA ONLY (Kornfeld per-cusp transit loss, cusp wall currents, sheath drops, cell potentials, where the ionisation sits)</h2><div id="targets"></div></section>
<section class="panel"><h2>Time series and profiles</h2><div class="legend" id="legend" aria-label="Design toggles"></div>
<div class="plots" style="margin-top:.8rem">
<div class="panel"><h2>Discharge current I_d</h2><canvas class="plot" id="p_id" role="img" aria-label="Discharge current versus time for the designs"></canvas></div>
<div class="panel"><h2>Exit ion beam current I_beam</h2><canvas class="plot" id="p_ib" role="img" aria-label="Exit ion beam current versus time"></canvas></div>
<div class="panel"><h2>Ionisation rate S</h2><canvas class="plot" id="p_s" role="img" aria-label="Ionisation rate versus time"></canvas></div>
<div class="panel"><h2>Neutral density n_g</h2><canvas class="plot" id="p_ng" role="img" aria-label="Neutral density versus time"></canvas></div>
<div class="panel"><h2>Macro-electron count N_e</h2><canvas class="plot" id="p_ne" role="img" aria-label="Macro-electron count versus time"></canvas></div>
<div class="panel"><h2>Windowed ledger residual / electrode work — recorded (solid) and corrected (dashed)</h2><canvas class="plot" id="p_res" role="img" aria-label="Trailing-window energy residual over electrode work versus time, recorded and corrected"></canvas></div>
<div class="panel"><h2>Peak Δ/λ_D (window gate statistic)</h2><canvas class="plot" id="p_deb" role="img" aria-label="Cells per Debye length at the peak node versus time with the soft and hard gates"></canvas></div>
<div class="panel"><h2>Peak ω_pe Δt (the statistic each run's code gated on)</h2><canvas class="plot" id="p_wpe" role="img" aria-label="Peak plasma frequency times time step versus time"></canvas></div>
<div class="panel"><h2>Axial profile of max_r n_e(z) vs z / L</h2><canvas class="plot" id="p_axn" role="img" aria-label="Radial maximum of the window-averaged electron density versus normalised axial position"></canvas></div>
<div class="panel"><h2>Axial ionisation profile (r-integrated, per m of z) vs z / L</h2><canvas class="plot" id="p_ion" role="img" aria-label="Radially integrated ionisation rate versus normalised axial position"></canvas></div>
</div></section>
<section class="panel"><h2>Run records (hash-verified)</h2><div id="records"></div></section>
<section class="panel"><h2>Simplifications, protocol and identity</h2><div id="identity"></div></section>
</main><footer>Self-contained offline dashboard generated by <code>modern/visualization/generate_pic2d_design_mini_sweep_v1.py</code>. Preregistered design sweep of a development model: not validated, not a performance prediction. Recorded and corrected-ledger (model v2.0.6, post hoc) readings are shown side by side; the recorded files are unchanged.</footer>
<script id="pic2d-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("pic2d-data").textContent);
const $=id=>document.getElementById(id);let raf=0,xMode="us";
const COLORS=["#5ad6c0","#58a8ff","#ffcf67","#c58bff"];const visible=DATA.designs.map(()=>true);
const fmt=(v,n=4)=>v==null||!isFinite(v)?"–":Number(v).toLocaleString(undefined,{maximumSignificantDigits:n});
const sci=(v,n=3)=>v==null||!isFinite(v)?"–":Number(v).toExponential(n-1);
const pct=(v,n=3,sign=false)=>v==null||!isFinite(v)?"–":(sign&&v>0?"+":"")+fmt(v*100,n)+" %";
const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const D=DATA.designs,REF=D.find(d=>d.role==="reference"),SW=DATA.sweep,V4=DATA.reference_grid_verdict,TR=DATA.trend;
const pillClass=d=>d.status!=="plateau"?"interim":({closure_quotable:"quotable",plateau_with_heating:"heating",no_plateau:"none"}[d.assessment.verdict]);
const pillText=d=>d.status!=="plateau"?"gate-stopped interim (launch 2 running)":d.assessment.verdict.replaceAll("_"," ");
$("verdict").innerHTML=D.map(d=>`<span class="pill ${pillClass(d)}"><small>${d.label} · ρ ${fmt(d.rho,3)} · ${d.role}</small>${pillText(d)}</span>`).join("")+`<div class="chips"><span class="chip">terminal records <b>${SW.terminal_records}/${SW.primary_designs}</b>${SW.provisional?" · <b>PROVISIONAL</b>":""}</span><span class="chip">reference grid (ss-v4 33 µm): recorded <b>${V4.verdict_recorded.replaceAll("_"," ")}</b> · corrected ledger <b>${V4.verdict_on_corrected_ledger.replaceAll("_"," ")}</b> ((b) ${pct(V4.b_corrected,3,true)})</span><span class="chip">prereg <b>${DATA.protocol.preregistration_commit.slice(0,8)}</b> · one execution per design</span></div>`;
$("provisional").innerHTML=`<strong>Sweep reading:</strong> ${SW.statement}`;
$("claim").innerHTML=`<strong>Claim boundary:</strong> ${DATA.claim_statement}<ul>${DATA.simplifications.map(s=>`<li>${s}</li>`).join("")}</ul>`;
const okSpan=(ok,txt)=>`<span class="${ok?"ok":"bad"}">${txt}</span>`;
function val(v,r){return r.unit==="mA"?fmt(v*r.display_scale,4)+" mA":r.unit==="eV"?fmt(v,4)+" eV":r.unit===""?fmt(v,4):sci(v,4)+" "+r.unit}
function renderTrend(){const head=`<tr><th>quantity</th>${D.map(d=>`<th class="num">${d.label} · ρ ${fmt(d.rho,3)}${d.status!=="plateau"?'<br><span class="bad">interim</span>':""}</th>`).join("")}<th class="num">reference spread</th></tr>`;
const body=TR.rows.map(r=>`<tr><td>${r.quantity}</td>${r.values.map(v=>`<td class="num ${v.status!=="plateau"?"dim":""}"><b>${val(v.value,r)}</b>${v.relative_to_reference!=null&&v.label!==REF.label?`<br><span class="small ${v.above_reference_spread?"ok":""}">${pct(v.relative_to_reference,3,true)} vs ref${v.above_reference_spread===false?" (inside spread)":""}</span>`:'<br><span class="small">reference</span>'}</td>`).join("")}<td class="num">${r.reference_spread==null?"–":"±"+pct(r.reference_spread,2)}</td></tr>`).join("");
const extra=`<tr><td>plateau: transits · steps · stop</td>${D.map(d=>`<td class="num ${d.status!=="plateau"?"dim":""}">${fmt(d.ion_transit_times,4)} · ${d.steps_completed} · ${d.stop_reason.replaceAll("_"," ")}</td>`).join("")}<td class="num">–</td></tr>
<tr><td>corrected residual / electrode work (b &lt; +2 %)</td>${D.map(d=>`<td class="num"><b class="${d.ledger_corrected.corrected_windowed<0.02?"ok":"bad"}">${pct(d.ledger_corrected.corrected_windowed,3,true)}</b><br><span class="small">recorded ${pct(d.ledger_corrected.recorded_windowed,3,true)}</span></td>`).join("")}<td class="num">gate +5 %</td></tr>
<tr><td>Δ/λ_D at the peak (window gate; maps)</td>${D.map(d=>`<td class="num">${fmt(d.peak_debye_window.cells_per_debye_window_last,3)} (${fmt(d.peak.cells_per_debye_maps,3)})</td>`).join("")}<td class="num">soft 2.5 · hard π</td></tr>
<tr><td>grid · Δt · W · particles at the end</td>${D.map(d=>`<td class="num">${d.grid.radial_cells}×${d.grid.axial_cells} · ${fmt(d.dt_s*1e12,3)} ps · ${sci(d.macro_weight,3)} · ${d.final_counts.electrons}+${d.final_counts.ions}</td>`).join("")}<td class="num">–</td></tr>
<tr><td>verdict (assessment.json)</td>${D.map(d=>`<td class="num"><b>${pillText(d)}</b></td>`).join("")}<td class="num">–</td></tr>`;
$("trend").innerHTML=`<table aria-label="Design versus rho trend"><thead>${head}</thead><tbody>${body}${extra}</tbody></table><p class="small">${TR.spread_note}. Relative shifts are vs the reference design (ρ ${fmt(REF.rho,3)}); the interim column is a gate-stopped, non-plateau state at ${fmt(D.find(d=>d.status!=="plateau")?.ion_transit_times,3)} transits and is shown for orientation only. Ionisation centroid = volume-weighted axial centroid of the window-averaged ionisation-rate map over the channel, divided by the channel length.</p>`}
function renderAcceptance(){$("acceptance").innerHTML=`<table aria-label="Acceptance per design"><thead><tr><th>design</th><th>(a) plateau</th><th>(b) residual power: corrected (decisive) · recorded</th><th>verdict</th><th>reference-grid caveat (f)</th></tr></thead><tbody>${D.map(d=>{if(d.status!=="plateau"){return `<tr><td>${d.label} · <code>${d.results_dir}</code></td><td><span class="bad">not reached</span> — ${d.stop_reason.replaceAll("_"," ")} at ${fmt(d.ion_transit_times,3)} transits</td><td>${pct(d.ledger_corrected.corrected_windowed,3,true)} corrected · ${pct(d.ledger_corrected.recorded_windowed,3,true)} recorded (not an acceptance: no plateau)</td><td><b>interim</b> — ${d.gate_stop.verdict.replaceAll("_"," ")}: ${d.gate_stop.note}</td><td class="small">applies once launch 2 records</td></tr>`}const a=d.assessment,pl=d.plateau||{};const drift=v=>v==null?"–":`<span class="${Math.abs(v)<.04?"ok":Math.abs(v)<.05?"marginal":"bad"}">${pct(v,3,true)}</span>`;return `<tr><td>${d.label} · <code>${d.results_dir}</code></td><td>${okSpan(a.a_plateau,a.a_plateau?"PASS":"FAIL")} — ${fmt(a.a_transits,4)} transits; drifts I_d ${drift(pl.discharge_current_drift)}, N_e ${drift(pl.electron_count_drift)}, n_g ${drift(pl.neutral_density_drift)}; triad soft ${okSpan(pl.triad_soft_ok,pl.triad_soft_ok?"ok":"exceeded")}; Debye soft ${okSpan(pl.peak_debye_soft_ok,pl.peak_debye_soft_ok?"held":"exceeded")}</td><td>${okSpan(a.b_passed,a.b_passed?"PASS":"FAIL")} — <b>${pct(a.b_corrected,3,true)}</b> corrected · ${pct(a.b_recorded,3,true)} recorded (${a.b_recorded_passed?"pass":"FAIL"})<br><span class="small">${a.b_basis}</span></td><td><b>${a.verdict.replaceAll("_"," ")}</b><br><span class="small">${a.verdict_rule}</span></td><td class="small">${a.convergence_statement_corrected_ledger||a.convergence_statement}</td></tr>`}).join("")}</tbody></table><p class="small">Reference grid: the ss-v4 33 µm refinement of the 50 µm base reads <b>${V4.verdict_recorded.replaceAll("_"," ")}</b> as recorded and <b>${V4.verdict_on_corrected_ledger.replaceAll("_"," ")}</b> on the corrected ledger ((b) ${pct(V4.b_recorded,3,true)} → ${pct(V4.b_corrected,3,true)}): ${V4.verdict_statement}</p>`}
function renderLedger(){$("ledger").innerHTML=`<table aria-label="Energy-ledger correction per run"><thead><tr><th>run</th><th class="num">end</th><th class="num">recorded windowed</th><th class="num">corrected windowed</th><th class="num">omitted inelastic</th><th class="num">cumulative recorded → corrected</th><th class="num">max corrected (complete windows)</th><th class="num">first ≥ 2 % · 5 % gate</th><th>(b) recorded → corrected</th><th>sidecar</th></tr></thead><tbody>${D.map(d=>{const c=d.ledger_corrected;return `<tr><td>${d.label} <code>${d.results_dir}</code></td><td class="num">${fmt(d.simulated_time_s*1e6,4)} µs</td><td class="num">${pct(c.recorded_windowed,3,true)}</td><td class="num"><b class="${c.corrected_windowed<0.02?"ok":"bad"}">${pct(c.corrected_windowed,3,true)}</b></td><td class="num">${pct(c.omitted_windowed,3,true)}</td><td class="num">${pct(c.recorded_cumulative,3,true)} → ${pct(c.corrected_cumulative,3,true)}</td><td class="num">${pct(c.max_corrected_over_complete_windows.ratio,3,true)} @ ${fmt(c.max_corrected_over_complete_windows.time_s*1e6,3)} µs</td><td class="num">${c.corrected_first_checkpoint_at_or_above_0p02_time_s==null?"never":fmt(c.corrected_first_checkpoint_at_or_above_0p02_time_s*1e6,3)+" µs"} · ${c.corrected_gate_0p05_first_checkpoint_time_s==null?"never":"<b class=\"bad\">"+fmt(c.corrected_gate_0p05_first_checkpoint_time_s*1e6,3)+" µs</b>"}</td><td>${c.acceptance_b_recorded_passes?"pass":"FAIL"} → <b class="${c.acceptance_b_corrected_passes?"ok":"bad"}">${c.acceptance_b_corrected_passes?"pass":"FAIL"}</b></td><td><code>${c.sidecar_sha256.slice(0,12)}…</code></td></tr>`}).join("")}</tbody></table><p class="small">recorded = the pre-v2.0.6 ledger (H − L_inel, biased negative by the inelastic power; every sweep run executed pre-v2.0.6 code in its locked worktree); corrected = H = field work + ΔU − electrode work rebuilt from the recorded series by <code>cft_revival.pic2d.ledger_recompute</code> (the sidecars) and recomputed here (the generator refuses a sidecar it cannot reproduce). Same window (400 000 steps), same bounds: acceptance (b) &lt; +2 %, hard gate +5 %.</p>`}
function renderTargets(){const rows=D.map(d=>{const t=d.closure_targets,io=d.ionisation;if(!t){return `<tr><td>${d.label} · ρ ${fmt(d.rho,3)} <span class="bad">interim</span></td><td colspan="6" class="small">no closure targets: gate-stopped record (non-plateau); ionisation centroid ${fmt(io.centroid_m/d.channel_length_m,3)} L, quarter shares ${io.quarter_length_shares.map(s=>fmt(s,3)).join(" / ")} (orientation only)</td></tr>`}const chain=t.kornfeld_chain_exit_to_anode.map(k=>`${fmt(k.z_c_m*1e3,4)} mm: p ${fmt(k.p_transit,3)} (L ${fmt(k.electron_wall_current_a*1e3,3)} mA of ${fmt(k.je_arriving_a*1e3,3)} arriving)`).join("<br>");const cusps=t.cusps.map(c=>`${fmt(c.z_c_m*1e3,4)} mm: e⁻ ${fmt(c.electron_wall_current_a*1e3,3)} / Xe⁺ ${fmt(c.ion_wall_current_a*1e3,3)} mA · drop ${fmt(c.sheath_drop_v,4)} V · T_e,wall ${fmt(c.near_wall_electron_temperature_ev,3)} eV`).join("<br>");const cells=t.cells_anode_to_exit.map(c=>`${c.cell_id} (${c.kind.replace("_"," ")}, ${fmt(c.z_start_m*1e3,3)}–${fmt(c.z_end_m*1e3,4)} mm): S share ${fmt(c.ionisation_share,3)} · ion wall loss ${fmt(c.ion_wall_loss_fraction,3)} · φ ${fmt(c.density_weighted_potential_v,4)} V · T_e ${fmt(c.density_weighted_electron_temperature_ev,3)} eV`).join("<br>");return `<tr><td>${d.label} · ρ ${fmt(d.rho,3)}<br><span class="small">${t.cusp_source}</span></td><td class="small">${chain}<br>entering ${fmt(t.entering_electron_current_a*1e3,3)} mA → anode cell ${fmt(t.electron_current_reaching_anode_cell_a*1e3,3)} mA</td><td class="small">${cusps}<br>diffuse non-cusp ${fmt(t.diffuse_non_cusp_electron_wall_current_a*1e3,3)} of ${fmt(t.total_wall_electron_current_a*1e3,3)} mA${t.anode_edge_electron_wall_current_a?"; anode-edge band "+fmt(t.anode_edge_electron_wall_current_a*1e3,3)+" mA":""}</td><td class="small">${cells}</td><td class="num">${t.potential_steps_v.map(v=>fmt(v,4)).join(" / ")} V<br><span class="small">φ_max ${fmt(t.phi_max_v,4)} V</span></td><td class="num">${fmt(io.centroid_m*1e3,4)} mm = ${fmt(io.centroid_m/d.channel_length_m,3)} L<br><span class="small">quartiles ${io.quartiles_m.map(q=>fmt(q*1e3,3)).join(" / ")} mm · upstream of mid ${pct(io.fraction_upstream_of_mid_channel,3)}</span></td><td class="num">${io.quarter_length_shares.map(s=>fmt(s,3)).join(" / ")}</td></tr>`}).join("");
$("targets").innerHTML=`<table aria-label="Closure targets per design"><thead><tr><th>design</th><th>Kornfeld chain (exit → anode)</th><th>cusp wall currents · sheath drops</th><th>cells (anode → exit)</th><th class="num">potential steps</th><th class="num">ionisation centroid</th><th class="num">S shares by quarter length (anode → exit)</th></tr></thead><tbody>${rows}</tbody></table><p class="small">${(D.find(d=>d.closure_targets)||{closure_targets:{note:""}}).closure_targets.note}. Declared targets: ${DATA.protocol.closure_targets_declared.map(t=>`${t.name} (${t.per}, ${t.role})`).join("; ")}.</p>`}
function renderRecords(){$("records").innerHTML=`<table aria-label="Run records"><thead><tr><th>design</th><th>results</th><th>status</th><th>stop · transits</th><th>steps · time</th><th>wall · ms/step</th><th>sessions · frames</th><th>summary SHA-256</th><th>maps · series SHA-256</th><th>lock: commit · protocol · config</th></tr></thead><tbody>${D.map(d=>`<tr><td>${d.label} <code>${d.id}</code></td><td><code>${d.results_dir}</code></td><td>${d.status.replaceAll("_"," ")}</td><td>${d.stop_reason.replaceAll("_"," ")} · ${fmt(d.ion_transit_times,4)}</td><td>${d.steps_completed} · ${fmt(d.simulated_time_s*1e6,4)} µs</td><td>${fmt(d.wall_seconds_total/3600,3)} h · ${fmt(d.ms_per_step_last_session,3)}</td><td>${d.sessions} · ${d.frames==null?"–":d.frames}</td><td><code>${d.summary_sha256.slice(0,16)}…</code></td><td><code>${d.maps_npz_sha256.slice(0,12)}…</code> · <code>${d.series_npz_sha256.slice(0,12)}…</code></td><td><code>${d.lock.commit.slice(0,12)}</code> · <code>${d.lock.protocol_sha256.slice(0,12)}…</code> · <code>${d.lock.config_sha256.slice(0,12)}…</code><br><span class="small">PID ${d.lock.pid} · ${d.lock.acquired_at_utc}</span></td></tr>`).join("")}</tbody></table>`}
function renderIdentity(){const P=DATA.protocol;$("identity").innerHTML=`<p><span class="badge">status</span> ${DATA.status.replaceAll("_"," ")}</p><p><span class="badge">model</span> ${DATA.model_version}</p><p><span class="badge">plateau rule</span> ${JSON.stringify(P.plateau_rule)}</p><p><span class="badge">acceptance</span> ${Object.entries(P.acceptance).filter(([k])=>k!=="d_verdicts").map(([k,v])=>`<b>${k}</b>: ${typeof v==="string"?v:JSON.stringify(v)}`).join(" · ")}</p><p><span class="badge">verdict values</span> ${Object.entries(P.acceptance.d_verdicts||{}).map(([k,v])=>`<b>${k}</b>: ${v}`).join(" · ")}</p><p><span class="badge">replication policy</span> ${Object.entries(P.replication_policy).map(([k,v])=>`<b>${k}</b>: ${v}`).join(" · ")}</p><p><span class="badge">amendments</span> ${P.amendments.map(a=>`<b>${a.case}</b>: ${(a.changes||[]).join(" ")}`).join(" · ")||"none"}</p><p><span class="badge">campaign protocol SHA-256</span> <code>${P.campaign_file_sha256}</code> · preregistration commit <code>${P.preregistration_commit}</code></p><p><span class="badge">reference grid</span> ss-v4 assessment <code>${V4.assessment_sha256.slice(0,16)}…</code> · corrected-ledger re-read <code>${V4.reread_sha256.slice(0,16)}…</code> (commit ${V4.commit.slice(0,12)})</p><p><span class="badge">claim boundary (protocol)</span> ${Object.entries(DATA.claim_boundary).map(([k,v])=>`<b>${k}</b>: ${v}`).join(" · ")}</p>`}
function setup(canvas){const r=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);return {c,w:r.width,h:r.height}}
function tick(v,lo,hi){const m=Math.max(Math.abs(lo),Math.abs(hi));return m>=1e5||(m>0&&m<1e-2)?(v===0?"0":Number(v).toExponential(2)):fmt(v,3)}
function axes(c,b,w,h,xlabel,ylabel,xmin,xmax,ymin,ymax){c.strokeStyle=themeColor("--line");c.fillStyle=themeColor("--muted");c.lineWidth=1;c.font="12px system-ui";c.strokeRect(b.l,b.t,b.r-b.l,b.b-b.t);c.textAlign="center";for(let i=0;i<=4;i++){const x=b.l+(b.r-b.l)*i/4;c.fillText(tick(xmin+(xmax-xmin)*i/4,xmin,xmax),x,b.b+18)}c.fillText(xlabel,(b.l+b.r)/2,h-6);c.save();c.translate(13,(b.t+b.b)/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="right";for(let i=0;i<=4;i++){const v=ymax-(ymax-ymin)*i/4;c.fillText(tick(v,ymin,ymax),b.l-6,b.t+(b.b-b.t)*i/4+4)}c.textAlign="left"}
function quantile(values,q){const s=[...values].sort((a,b)=>a-b);if(!s.length)return NaN;const k=(s.length-1)*q,i=Math.floor(k);return s[i]+(s[Math.min(i+1,s.length-1)]-s[i])*(k-i)}
function drawPlot(id,series,xLabel,yLabel,marks={}){const s=setup($(id)),c=s.c,b={l:64,t:16,r:s.w-16,b:s.h-40},pts=series.filter(q=>q&&q.x.length);c.clearRect(0,0,s.w,s.h);c.fillStyle=themeColor("--panel");c.fillRect(0,0,s.w,s.h);if(!pts.length){c.fillStyle=themeColor("--muted");c.fillText("no design selected",b.l+8,b.t+16);return}const all=pts.flatMap(q=>q.y.filter(v=>v!=null&&isFinite(v))),xmin=marks.xmin!=null?marks.xmin:Math.min(...pts.flatMap(q=>q.x)),xmax=marks.xmax!=null?marks.xmax:Math.max(...pts.flatMap(q=>q.x));let ymin=Math.min(...all),ymax=Math.max(...all);if(marks.robust){ymin=quantile(all,.01);ymax=quantile(all,.99)}if(marks.ymin!=null)ymin=marks.ymin;if(marks.ymax!=null)ymax=marks.ymax;const pad=(ymax-ymin||1)*.08;ymin-=pad;ymax+=pad;const X=x=>b.l+(x-xmin)/(xmax-xmin||1)*(b.r-b.l),Y=v=>b.b-(v-ymin)/(ymax-ymin||1)*(b.b-b.t);
if(marks.bands){marks.bands.forEach(bd=>{const x0=Math.max(b.l,X(bd.x[0])),x1=Math.min(b.r,X(bd.x[1]));c.fillStyle=bd.color;c.fillRect(x0,b.t,x1-x0,b.b-b.t)})}
if(marks.vlines){c.save();c.setLineDash([4,4]);c.strokeStyle=themeColor("--muted");c.lineWidth=1;marks.vlines.forEach(v=>{if(v==null||v<xmin||v>xmax)return;c.beginPath();c.moveTo(X(v),b.t);c.lineTo(X(v),b.b);c.stroke()});c.restore()}
if(marks.hlines){c.save();c.setLineDash([2,4]);c.lineWidth=1;c.font="11px system-ui";marks.hlines.forEach(h=>{if(!(h.y>=ymin&&h.y<=ymax))return;c.strokeStyle=h.color||themeColor("--muted");const py=Y(h.y);c.beginPath();c.moveTo(b.l,py);c.lineTo(b.r,py);c.stroke();c.fillStyle=h.color||themeColor("--muted");c.fillText(h.name,b.r-8-c.measureText(h.name).width,py-3)});c.restore()}
axes(c,b,s.w,s.h,xLabel,yLabel,xmin,xmax,ymin,ymax);c.save();c.beginPath();c.rect(b.l,b.t,b.r-b.l,b.b-b.t);c.clip();pts.forEach(q=>{c.strokeStyle=q.color;c.lineWidth=q.width||1.4;if(q.dash)c.setLineDash(q.dash);else c.setLineDash([]);c.beginPath();let started=false;q.x.forEach((x,i)=>{const v=q.y[i];if(v==null||!isFinite(v)){started=false;return}const px=X(x),py=Y(v);started?c.lineTo(px,py):c.moveTo(px,py);started=true});c.stroke()});c.setLineDash([]);c.restore();c.font="12px system-ui";pts.forEach((q,k)=>{c.fillStyle=q.color;c.fillText(q.name,b.l+8,b.t+14+k*15)});if(marks.robust){c.fillStyle=themeColor("--muted");c.font="10px system-ui";const note="y-range: 1–99 % quantiles (seed transient clipped)";c.fillText(note,b.r-6-c.measureText(note).width,b.b-4)}}
const transitOf=d=>d.simulated_time_s/d.ion_transit_times;
const tx=d=>d.series.time_s.map(v=>xMode==="us"?v*1e6:v/transitOf(d));
function lines(key,scale=1,extra=""){return D.map((d,i)=>visible[i]&&d.series[key]?{x:tx(d),y:d.series[key].map(v=>v==null?null:v*scale),name:d.label+" ρ "+fmt(d.rho,3)+extra,color:COLORS[i],width:d.role==="reference"?1.8:1.1}:null)}
function winBand(d,color){const t1=d.simulated_time_s,f=d.plateau&&d.plateau.window_fraction!=null?d.plateau.window_fraction:.2,t0=t1-f*t1;return {x:xMode==="us"?[t0*1e6,t1*1e6]:[t0/transitOf(d),t1/transitOf(d)],color}}
function drawSeries(){const xl=xMode==="us"?"t (µs)":"design ion transits",tm={bands:[winBand(REF,themeColor("--window"))],vlines:xMode==="us"?[]:[3]};
drawPlot("p_id",lines("current_discharge_a",1e3),xl,"I_d (mA)",{...tm,robust:true});
drawPlot("p_ib",lines("current_exit_ion_beam_a",1e3),xl,"I_beam,i (mA)",{...tm,robust:true});
drawPlot("p_s",lines("current_ionization_rate_per_s"),xl,"S (s⁻¹)",{...tm,robust:true});
drawPlot("p_ng",lines("neutral_density_per_m3"),xl,"n_g (m⁻³)",{...tm,hlines:[{y:REF.neutral_inventory.zero_ionization_density_per_m3,name:"n_g0 = Q_in/c (reference feed)",color:"#ffcf67"}]});
drawPlot("p_ne",lines("electrons"),xl,"macro-electrons",tm);
drawPlot("p_res",[...lines("windowed_residual_over_electrode_work",100," (recorded)"),...lines("windowed_residual_corrected_over_electrode_work",100," (corrected)").map(q=>q&&{...q,dash:[6,4]})],xl,"residual / electrode work (%)",{...tm,hlines:[{y:5,name:"v2.0.3 gate +5 %",color:"#ff6b6b"},{y:2,name:"acceptance (b) +2 %",color:"#ffcf67"},{y:0,name:"0",color:themeColor("--muted")}],ymin:-20,ymax:8});
drawPlot("p_deb",lines("peak_node_window_cells_per_debye"),xl,"Δ/λ_D at the peak (window)",{...tm,hlines:[{y:Math.PI,name:"hard π",color:"#ff6b6b"},{y:2.5,name:"soft 2.5",color:"#ffcf67"}],ymin:0,ymax:3.5});
drawPlot("p_wpe",lines("peak_omega_pe_dt"),xl,"peak ω_pe Δt",{...tm,hlines:[{y:.2,name:"gate 0.2",color:"#ff6b6b"}],ymin:0,ymax:.3});
drawPlot("p_axn",D.map((d,i)=>visible[i]?{x:d.profiles.z_over_channel,y:d.profiles.axial_peak_n_e_per_m3,name:d.label+" ρ "+fmt(d.rho,3),color:COLORS[i],width:d.role==="reference"?1.8:1.1}:null),"z / L","max_r n_e (m⁻³)",{xmin:0,xmax:1.05});
drawPlot("p_ion",D.map((d,i)=>visible[i]?{x:d.profiles.ionisation_z_m.map(z=>z/d.channel_length_m),y:d.profiles.ionisation_per_s_per_m,name:d.label+" ρ "+fmt(d.rho,3),color:COLORS[i],width:d.role==="reference"?1.8:1.1}:null),"z / L","ionisations s⁻¹ m⁻¹",{xmin:0,xmax:1.0})}
function renderLegend(){$("legend").innerHTML=D.map((d,i)=>`<label><input type="checkbox" data-i="${i}" ${visible[i]?"checked":""}> <span class="sw" style="background:${COLORS[i]}"></span>${d.label} · ρ ${fmt(d.rho,3)} · ${d.grid.radial_cells}×${d.grid.axial_cells} · W ${sci(d.macro_weight,3)} · cusps ${d.cusp_z_m.map(z=>fmt(z*1e3,3)).join("/")} mm${d.status!=="plateau"?' <span class="bad">(interim)</span>':""}</label>`).join("");$("legend").querySelectorAll("input").forEach(el=>el.onchange=()=>{visible[Number(el.dataset.i)]=el.checked;schedule()})}
function drawAll(){drawSeries()}
function schedule(){cancelAnimationFrame(raf);raf=requestAnimationFrame(drawAll)}
renderTrend();renderAcceptance();renderLedger();renderTargets();renderRecords();renderIdentity();renderLegend();
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
    return output_path.with_name(output_path.stem + ".anchor-platform.json")


def write_anchor_platform(output_path: Path, html: str) -> Path:
    record = {
        "schema": "cft-pic2d-dashboard-anchor-platform/1.0.0", "html_file": output_path.name, "html_sha256": sha256(html.encode("utf-8")).hexdigest(),
        "platform": platform_fingerprint(),
        "policy": "byte-exact replay of the checked-in HTML is asserted only under the same platform fingerprint_sha256; elsewhere the embedded "
                  "payload must agree structurally with numeric leaves within one unit in their last recorded significant digit (rel 1e-9 floor)",
    }
    path = anchor_platform_path(output_path)
    path.write_bytes(json.dumps(record, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    return path


def generate(output_path: Path = DEFAULT_OUTPUT, results: Path = RESULTS, campaign_protocol: Path = CAMPAIGN_PROTOCOL) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(build_payload(results, campaign_protocol))
    output_path.write_text(html, encoding="utf-8", newline="\n")
    write_anchor_platform(output_path, html)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--campaign-protocol", type=Path, default=CAMPAIGN_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate(args.output, args.results, args.campaign_protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
