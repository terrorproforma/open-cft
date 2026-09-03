"""Generate the standalone PIC-2D CFT steady-state dashboard.

Headline: the model v1.3 development steady state of
``modern/experiments/pic2d_cft_steady_state_v2/results`` (quasi-steady 0-D neutral
inventory, plateau declared after >= 3 ion transit times).  Every embedded input is
hash-verified against its ``.sha256.json`` sidecar and the run's recorded protocol
hash (fail-closed: a protocol file that drifted from the run is rejected).  Finished
convergence variants (``results-<variant>/`` with a ``summary.json``) are embedded as
additional cases; unfinished ones are listed as pending.  The history panels keep the
steady-state predecessors (v1 no-ignition reference, v1.3 attempt 1), the snapshot v2
growth cases and the snapshot v1 fail-closed cases.  No timestamps or runtime
measurements are added, so identical inputs give identical bytes; the page is
self-contained (no network access) and states its claim boundary on every view.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite, pi, sqrt
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

MODERN = Path(__file__).resolve().parents[1]
SRC = MODERN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MODERN) not in sys.path:
    sys.path.insert(0, str(MODERN))

from cft_revival.pic2d.artifacts import read_canonical_json, read_npz  # noqa: E402
from cft_revival.pic2d.mesh import build_mesh_masks  # noqa: E402
from cft_revival.pic2d.models import ChannelGeometry, Grid2D  # noqa: E402


def _load_snapshot_generator():
    import importlib.util

    path = Path(__file__).with_name("generate_pic2d_cft_snapshot.py")
    spec = importlib.util.spec_from_file_location("pic2d_snapshot_generator", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(path.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snapshot_dashboard = _load_snapshot_generator()

EXPERIMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v2"
RESULTS = EXPERIMENT / "results"
PROTOCOL = EXPERIMENT / "protocol.json"
VARIANTS = EXPERIMENT / "variants.json"
REFERENCE_V1 = MODERN / "experiments" / "pic2d_cft_steady_state_v1"
SNAPSHOT_V2 = MODERN / "experiments" / "pic2d_cft_snapshot_v2"
DEFAULT_OUTPUT = Path(__file__).with_name("pic2d-cft-steady-state.html")
SCHEMA = "cft-pic2d-cft-steady-state-visualization/0.1.0"
STOP_REASONS = {
    "plateau_reached_after_min_transit_times", "wall_clock_budget_reached", "runtime_stability_gate_stopped_run",
    "finalized_no_ignition_reference_after_3_transit_times", "stopped_no_ignition_attempt1_after_1us",
    "target_steps_reached", "plateau_reached_after_min_steps",
}
MAP_KEYS = ("n_e_per_m3", "n_i_per_m3", "phi_v", "t_e_ev", "ionization_rate_per_m3_s")
WALL_KEYS = ("wall_electron_flux_per_m2_s", "wall_ion_flux_per_m2_s", "wall_electron_mean_energy_ev", "wall_ion_mean_energy_ev")
EXIT_KEYS = ("exit_ion_current_density_a_per_m2", "exit_electron_current_density_a_per_m2")
SERIES_KEYS = (
    "time_s", "step", "electrons", "ions", "phi_min_v", "phi_mean_v", "phi_max_v", "kinetic_electron_j", "kinetic_ion_j",
    "field_energy_j", "total_energy_j", "interval_residual_j", "interval_electrode_work_j", "peak_omega_pe_dt",
    "current_discharge_a", "current_exit_ion_beam_a", "current_exit_electron_a", "current_wall_electron_a",
    "current_wall_ion_a", "current_injected_electron_a", "current_anode_electron_a",
    "neutral_density_per_m3", "neutral_fixed_point_per_m3", "neutral_ionization_rate_per_s", "neutral_effusion_rate_per_s",
    "neutral_artificial_rate_per_s", "neutral_ledger_fed", "neutral_ledger_ionized", "neutral_ledger_effused",
    "neutral_ledger_artificial", "neutral_interval_ledger_residual_atoms",
)
MAX_SERIES_POINTS = 3200
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31

_round = snapshot_dashboard._round
_matrix = snapshot_dashboard._matrix
_file_sha256 = snapshot_dashboard._file_sha256
_verify_sidecar = snapshot_dashboard._verify_sidecar


def _decimate(values: np.ndarray, stride: int) -> np.ndarray:
    values = np.asarray(values)
    if stride <= 1:
        return values
    out = values[::stride]
    if (values.shape[0] - 1) % stride:
        out = np.concatenate([out, values[-1:]])
    return out


def _grid(summary: Mapping[str, Any]) -> tuple[Grid2D, dict[str, Any]]:
    grid = summary["provenance"]["config"]["grid"]
    geometry = grid["geometry"]
    plume = {key: geometry[key] for key in ("plume_radius_m", "plume_length_m", "body_dielectric_radius_m") if geometry.get(key)}   # v2.0
    return Grid2D(
        ChannelGeometry(geometry["bore_radius_m"], geometry["z_min_m"], geometry["z_max_m"], geometry["cone_start_z_m"], geometry["exit_radius_m"], **plume),
        int(grid["radial_cells"]), int(grid["axial_cells"]),
    ), grid


def cusp_positions(maps: Mapping[str, np.ndarray], grid: Grid2D, grid_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Axial cusp planes from the embedded density/field context.

    The maps do not carry B, so the cusp planes are located from the wall ion-flux
    structure only when the P2 field map cannot be rebuilt; when the field module is
    importable the sign changes of B_z on the axis are used (exact for the P2 design).
    """

    try:
        from cft_revival.pic2d.fields import build_p2_psi_field, sample_field_map

        psi, evaluated = build_p2_psi_field(MODERN.parent, role="primary")
        field = sample_field_map(psi, grid, evaluated)
        b_z = np.asarray(field.b_z_t)
        z = float(grid_dict["geometry"]["z_min_m"]) + np.arange(b_z.shape[1]) * float(grid_dict["dz_m"])
        flips = np.where(np.diff(np.sign(b_z[0])) != 0)[0]
        cusps = [float(f"{0.5 * (z[i] + z[i + 1]):.6g}") for i in flips]
        b_r_wall = np.abs(np.asarray(field.b_r_t)[-1])
        # magnet mid-planes: local minima of |B_r| on the bore wall between cusps
        mids: list[float] = []
        bounds = [z[0], *[0.5 * (z[i] + z[i + 1]) for i in flips], z[-1]]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            sel = np.where((z >= lo) & (z <= hi))[0]
            if sel.size:
                mids.append(float(f"{z[sel[int(np.argmin(b_r_wall[sel]))]]:.6g}"))
        return {"source": "B_z sign change on the axis of the sampled P2 field map", "cusp_z_m": cusps, "magnet_midplane_z_m": mids,
                "field_map_sha256": field.sha256}
    except Exception as exc:  # pragma: no cover - only without the P2 checkpoint
        return {"source": f"field map unavailable ({type(exc).__name__}); cusps not marked", "cusp_z_m": [], "magnet_midplane_z_m": []}


def build_case(case_dir: Path, protocol_path: Path, *, label: str, role: str, protocol_sha_expected: str | None = None,
               raw_out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hash-verified digest of one finished steady-state case directory (maps + series embedded).

    ``raw_out`` (if given) receives the full-resolution ``series``/``maps`` arrays and the
    plasma mask for the between-case comparison; they are not embedded.
    """

    summary_path = case_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"{case_dir.name}: summary.json is missing")
    summary_sha = _verify_sidecar(summary_path)
    summary = read_canonical_json(summary_path)
    if summary.get("protocol_sha256") != _file_sha256(protocol_path):
        raise ValueError(f"{case_dir.name}: protocol.json differs from the hash recorded by the run (protocol drift)")
    if protocol_sha_expected is not None and summary["protocol_sha256"] != protocol_sha_expected:
        raise ValueError(f"{case_dir.name}: protocol hash differs from the headline case")
    maps_sha = summary["artifacts"]["maps_npz_sha256"]
    series_sha = summary["artifacts"]["series_npz_sha256"]
    maps = read_npz(case_dir / "maps.npz", expected_sha256=maps_sha)
    series = read_npz(case_dir / "series.npz", expected_sha256=series_sha)
    run_state = read_canonical_json(case_dir / "run_state.json") if (case_dir / "run_state.json").is_file() else None
    grid, grid_dict = _grid(summary)
    nr, nz = grid.radial_cells, grid.axial_cells
    r = np.arange(nr + 1) * float(grid_dict["dr_m"])
    z = float(grid_dict["geometry"]["z_min_m"]) + np.arange(nz + 1) * float(grid_dict["dz_m"])
    stride = 1 if nz <= 256 else 2
    masks = build_mesh_masks(grid)
    plasma = masks.plasma_node
    if raw_out is not None:
        raw_out.update({"series": {k: np.asarray(v) for k, v in series.items()}, "maps": {k: np.asarray(v) for k, v in maps.items()},
                        "plasma": plasma, "summary": summary, "dz_m": float(grid_dict["dz_m"]), "masks": masks, "grid": grid})
    embedded_maps = {key: _matrix(np.where(plasma, maps[key], np.nan)[:, ::stride]) for key in MAP_KEYS}
    sampling = snapshot_dashboard.sampling_block(maps, masks, float(summary["provenance"]["config"]["macro_weight"]),
                                                 float(summary["provenance"]["config"]["dt_s"]), stride)
    # peak-density location (node) and the axial profile of the radial maximum
    n_e = np.where(plasma, maps["n_e_per_m3"], 0.0)
    ri, zi = np.unravel_index(int(np.argmax(n_e)), n_e.shape)
    axial_peak = np.max(n_e, axis=0)
    n_samples = int(series["time_s"].shape[0])
    series_stride = max(1, -(-n_samples // MAX_SERIES_POINTS))
    embedded_series = {key: _round(_decimate(series[key], series_stride)) for key in SERIES_KEYS if key in series}
    window = summary.get("averaging_window_step_range")
    w_summary = summary["window_maps_summary"]
    t_e_ref = w_summary.get("t_e_density_weighted_mean_ev") or 8.0
    n_peak = w_summary["n_e_peak_per_m3"]
    lambda_d_peak = sqrt(EPS0 * t_e_ref / (max(n_peak, 1.0) * E_CHARGE))
    resolvability = {
        "n_e_peak_per_m3": n_peak,
        "t_e_reference_ev": t_e_ref,
        "lambda_d_at_peak_m": lambda_d_peak,
        "dz_over_lambda_d_at_peak": float(grid_dict["dz_m"]) / lambda_d_peak,
        "dr_over_lambda_d_at_peak": float(grid_dict["dr_m"]) / lambda_d_peak,
        "omega_pe_dt_at_peak": sqrt(n_peak * E_CHARGE**2 / (EPS0 * M_E)) * float(summary["provenance"]["config"]["dt_s"]),
        "max_observed_omega_pe_dt": (summary.get("budget_check") or {}).get("max_observed_omega_pe_dt"),
        "n_e_peak_over_n_max": (summary.get("budget_check") or {}).get("n_e_peak_over_n_max"),
    }
    neutral = summary.get("neutral_inventory")
    return {
        "id": summary["case"]["id"],
        "label": label,
        "role": role,
        "results_dir": case_dir.name,
        "case": summary["case"],
        "summary_sha256": summary_sha,
        "maps_npz_sha256": maps_sha,
        "series_npz_sha256": series_sha,
        "protocol_sha256": summary["protocol_sha256"],
        "git_head": summary.get("git_head"),
        "backend": summary["backend"],
        "maps_kind": summary.get("maps_kind"),
        "steps_completed": summary["steps_completed"],
        "simulated_time_s": summary["simulated_time_s"],
        "ion_transit_times": summary.get("ion_transit_times"),
        "stop_reason": summary["stop_reason"],
        "stability_gate_message": summary.get("stability_gate_message"),
        "plateau": summary.get("plateau"),
        "ledger": summary.get("ledger"),
        "neutral_inventory": neutral,
        "window_currents_a": summary.get("window_currents_a"),
        "window_maps_summary": w_summary,
        "budget_check": summary.get("budget_check"),
        "averaging_window_step_range": window,
        "averaging_window_steps": summary.get("averaging_window_steps"),
        "wall_seconds_total": float(f"{summary['wall_seconds_total']:.5g}"),
        "ms_per_step_last_session": float(f"{summary['ms_per_step_this_session']:.4g}") if summary.get("ms_per_step_this_session") else None,
        "sessions": [{k: v for k, v in s.items() if k != "pid"} for s in (run_state or {}).get("sessions", [])] if run_state else summary.get("sessions"),
        "final_counts": summary["final_counts"],
        "peak_counts": summary.get("peak_counts"),
        "stability_gate": summary["provenance"]["stability_gate"],
        "mesh": summary["provenance"]["mesh"],
        "config": {
            "dt_s": summary["provenance"]["config"]["dt_s"],
            "macro_weight": summary["provenance"]["config"]["macro_weight"],
            "seed": summary["provenance"]["config"].get("seed"),
            "grid": {"radial_cells": nr, "axial_cells": nz, "dr_m": grid_dict["dr_m"], "dz_m": grid_dict["dz_m"]},
            "potentials": summary["provenance"]["config"]["potentials"],
            "injection": summary["provenance"]["config"]["injection"],
            "seed_plasma": summary["provenance"]["config"]["seed_plasma"],
            "mcc": summary["provenance"]["config"]["mcc"],
            "neutral_inventory": summary["provenance"]["config"].get("neutral_inventory"),
        },
        "field": summary["provenance"]["field"],
        "cross_sections": summary["provenance"].get("cross_sections"),
        "resolvability_at_peak": resolvability,
        "peak_density_node": {"r_m": float(f"{r[ri]:.6g}"), "z_m": float(f"{z[zi]:.6g}"), "n_e_per_m3": float(f"{n_e[ri, zi]:.6g}")},
        "axial_peak_n_e_per_m3": _round(axial_peak[::stride]),
        "cusps": cusp_positions(maps, grid, grid_dict),
        "grid_r_m": _round(r),
        "grid_z_m": _round(z[::stride]),
        "maps": embedded_maps,
        "sampling": sampling,
        "wall_z_m": _round(z[:-1] + 0.5 * float(grid_dict["dz_m"])),
        "wall": {key: _round(maps[key]) for key in WALL_KEYS},
        "exit_r_m": _round(0.5 * (r[:-1] + r[1:])),
        "exit": {key: _round(maps[key]) for key in EXIT_KEYS},
        "series_stride": series_stride,
        "series_samples": n_samples,
        "series": embedded_series,
        "final_series": summary["final_series"],
    }


# -- between-case comparison (statistical consistency) ---------------------------

BLOCK_SECONDS = 3.0e-8   # batch-means block (100 series intervals of 0.3 ns): longer than the ~ns fluctuation correlation time


def _window_stats(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float) -> tuple[float, float, int]:
    """Window mean, its batch-means standard error (accounts for autocorrelated fluctuations), sample count."""

    sel = (t > t_start) & (t <= t_end) & np.isfinite(y)
    values = np.asarray(y[sel], dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(values.mean())
    times = t[sel]
    blocks = np.floor((times - times[0]) / BLOCK_SECONDS).astype(int)
    means = np.array([values[blocks == b].mean() for b in np.unique(blocks)])
    se = float(means.std(ddof=1) / sqrt(means.size)) if means.size > 1 else float("nan")
    return mean, se, int(values.size)


def _series_quantities(raw: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, str, str]]:
    """Name -> (values, unit, shot-noise kind) derived from a full-resolution series."""

    s = raw["series"]
    w = float(raw["summary"]["provenance"]["config"]["macro_weight"])
    n_e = np.maximum(s["electrons"].astype(np.float64), 1.0)
    t_e = (2.0 / 3.0) * s["kinetic_electron_j"] / (n_e * w * E_CHARGE)
    return {
        "I_d": (s["current_discharge_a"], "A", "current"),
        "I_beam,i": (s["current_exit_ion_beam_a"], "A", "current"),
        "I_wall,i": (s["current_wall_ion_a"], "A", "current"),
        "I_wall,e": (s["current_wall_electron_a"], "A", "current"),
        "I_exit,e (returned)": (s["current_exit_electron_a"], "A", "current"),
        "S": (s["neutral_ionization_rate_per_s"], "1/s", "rate"),
        "n_g": (s["neutral_density_per_m3"], "1/m^3", "none"),
        "N_e (macro)": (s["electrons"].astype(np.float64), "", "count"),
        "N_i (macro)": (s["ions"].astype(np.float64), "", "count"),
        "<T_e> (2/3 K/N)": (t_e, "eV", "count"),
        "phi_max": (s["phi_max_v"], "V", "none"),
        "phi_mean": (s["phi_mean_v"], "V", "none"),
        "phi_min": (s["phi_min_v"], "V", "none"),
        "peak omega_pe dt": (s["peak_omega_pe_dt"], "", "none"),
    }


def _shot_noise_relative(kind: str, mean: float, window_seconds: float, w: float, samples: int) -> float | None:
    """Pure counting-noise relative sigma of the window mean (no correlations): 1/sqrt(macro events)."""

    if not isfinite(mean) or mean == 0:
        return None
    if kind == "current":
        events = abs(mean) * window_seconds / (E_CHARGE * w)
    elif kind == "rate":
        events = abs(mean) * window_seconds / w
    elif kind == "count":
        events = abs(mean)  # one snapshot of N macro-particles (per-sample sigma; the window mean is not more precise if N is conserved)
    else:
        return None
    return 1.0 / sqrt(events) if events > 0 else None


def compare_series(base: Mapping[str, Any], other: Mapping[str, Any], t_start: float, t_end: float) -> list[dict[str, Any]]:
    """Window means of the two runs over [t_start, t_end] with batch-means errors and z-scores."""

    rows = []
    qb, qo = _series_quantities(base), _series_quantities(other)
    tb, to = base["series"]["time_s"], other["series"]["time_s"]
    w = float(base["summary"]["provenance"]["config"]["macro_weight"])
    for name, (yb, unit, kind) in qb.items():
        mb, sb, nb = _window_stats(tb, yb, t_start, t_end)
        mo, so, no = _window_stats(to, qo[name][0], t_start, t_end)
        diff = mo - mb
        denom = sqrt(sb**2 + so**2) if isfinite(sb) and isfinite(so) and (sb > 0 or so > 0) else float("nan")
        rows.append({
            "quantity": name, "unit": unit, "base": mb, "other": mo, "abs_diff": diff,
            "rel_diff": diff / abs(mb) if mb else None,
            "se_base": sb, "se_other": so, "samples_base": nb, "samples_other": no,
            "z": diff / denom if isfinite(denom) else None,
            "shot_noise_rel": _shot_noise_relative(kind, mb, t_end - t_start, w, nb),
        })
    return [{k: (float(f"{v:.6g}") if isinstance(v, float) and isfinite(v) else (None if isinstance(v, float) else v)) for k, v in r.items()} for r in rows]


def compare_maps(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Plateau-window map comparison (windows may differ in time; the caller labels that)."""

    pb, po = base["plasma"], other["plasma"]
    mb, mo = base["maps"], other["maps"]
    out: dict[str, Any] = {}

    def rel(a: float, b: float) -> float | None:
        return float(f"{(b - a) / abs(a):.4g}") if a else None

    for key, label in (("n_e_per_m3", "n_e"), ("t_e_ev", "T_e"), ("phi_v", "phi"), ("ionization_rate_per_m3_s", "ionisation rate")):
        a = np.where(pb, mb[key], np.nan)
        b = np.where(po, mo[key], np.nan)
        wa = np.where(pb, mb["n_e_per_m3"], 0.0)
        wb = np.where(po, mo["n_e_per_m3"], 0.0)
        mean_a = float(np.nanmean(a)) if key != "t_e_ev" else float(np.nansum(a * wa) / wa.sum())
        mean_b = float(np.nanmean(b)) if key != "t_e_ev" else float(np.nansum(b * wb) / wb.sum())
        peak_a, peak_b = float(np.nanmax(a)), float(np.nanmax(b))
        both = np.isfinite(a) & np.isfinite(b)
        l2 = float(np.sqrt(np.nansum((b[both] - a[both]) ** 2) / np.nansum(a[both] ** 2))) if key != "phi_v" else float(np.sqrt(np.nanmean((b[both] - a[both]) ** 2)))
        out[label] = {"mean_base": mean_a, "mean_other": mean_b, "mean_rel_diff": rel(mean_a, mean_b), "peak_base": peak_a, "peak_other": peak_b,
                      "peak_rel_diff": rel(peak_a, peak_b), ("relative_l2_diff" if key != "phi_v" else "rms_diff_v"): float(f"{l2:.4g}")}
    dz = base["dz_m"]
    for key, label in (("wall_ion_flux_per_m2_s", "wall ion flux"), ("wall_electron_flux_per_m2_s", "wall electron flux")):
        a, b = mb[key], mo[key]
        za, zb = (int(np.argmax(a)) + 0.5) * dz, (int(np.argmax(b)) + 0.5) * dz
        out[label] = {"peak_base": float(a.max()), "peak_other": float(b.max()), "peak_rel_diff": rel(float(a.max()), float(b.max())),
                      "peak_z_base_m": float(f"{za:.5g}"), "peak_z_other_m": float(f"{zb:.5g}"),
                      "relative_l2_diff": float(f"{np.sqrt(((b - a) ** 2).sum() / (a ** 2).sum()):.4g}"),
                      "total_rel_diff": rel(float(a.sum()), float(b.sum()))}
    for key, label in (("exit_ion_current_density_a_per_m2", "exit ion j_z"), ("exit_electron_current_density_a_per_m2", "exit electron j_z")):
        a, b = mb[key], mo[key]
        out[label] = {"axis_base": float(a[0]), "axis_other": float(b[0]), "axis_rel_diff": rel(float(a[0]), float(b[0])),
                      "relative_l2_diff": float(f"{np.sqrt(((b - a) ** 2).sum() / (a ** 2).sum()):.4g}")}
    return out


def build_comparison(base: Mapping[str, Any], other: Mapping[str, Any], base_case: Mapping[str, Any], other_case: Mapping[str, Any]) -> dict[str, Any]:
    """Statistical comparison of a variant against the headline run.

    Window A: the variant's trailing-20 % window evaluated in BOTH runs (same simulated
    time; both runs are at the same stage).  Window B: the base run's plateau window vs
    the variant's trailing window when they do not overlap (time-offset; the variant may
    still be drifting).  Maps are compared as written (their windows are listed).
    """

    t_end_other = float(other_case["simulated_time_s"])
    frac = (other_case["plateau"] or {}).get("window_fraction", 0.2)
    a_start, a_end = (1.0 - frac) * t_end_other, t_end_other
    base_window = base_case["averaging_window_step_range"]
    dt = float(base_case["config"]["dt_s"])
    b_start, b_end = base_window[0] * dt, base_window[1] * dt
    windows = [{
        "label": "A: common window (variant's trailing 20 %, same simulated time in both runs)",
        "t_start_s": a_start, "t_end_s": a_end, "rows": compare_series(base, other, a_start, a_end),
    }]
    if b_start >= a_end:
        base_rows = {r["quantity"]: r for r in compare_series(base, base, b_start, b_end)}
        other_rows = {r["quantity"]: r for r in compare_series(other, other, a_start, a_end)}
        rows = []
        for name, rb in base_rows.items():
            ro = other_rows[name]
            mb, mo, sb, so = rb["base"], ro["base"], rb["se_base"], ro["se_base"]
            diff = None if mb is None or mo is None else mo - mb
            denom = sqrt(sb**2 + so**2) if sb is not None and so is not None and (sb > 0 or so > 0) else None
            rows.append({"quantity": name, "unit": rb["unit"], "base": mb, "other": mo, "abs_diff": diff,
                         "rel_diff": None if diff is None or not mb else float(f"{diff / abs(mb):.6g}"),
                         "se_base": sb, "se_other": so, "samples_base": rb["samples_base"], "samples_other": ro["samples_base"],
                         "z": None if diff is None or denom is None else float(f"{diff / denom:.6g}"), "shot_noise_rel": rb["shot_noise_rel"]})
        windows.append({
            "label": "B: base plateau window vs variant trailing window (time-offset: the variant had not reached the base window)",
            "t_start_s": b_start, "t_end_s": b_end, "other_t_start_s": a_start, "other_t_end_s": a_end, "rows": rows,
        })
    other_window = other_case["averaging_window_step_range"]
    return {
        "base_id": base_case["id"], "other_id": other_case["id"], "other_label": other_case["label"],
        "other_stop_reason": other_case["stop_reason"], "other_transit_times": other_case["ion_transit_times"],
        "windows": windows,
        "maps": {
            "base_window_s": [b_start, b_end],
            "other_window_s": [other_window[0] * dt, other_window[1] * dt] if other_window else None,
            "note": "window-average maps as written by each run; the windows differ in simulated time when the variant stopped early",
            "rows": compare_maps(base, other),
        },
        "block_seconds": BLOCK_SECONDS,
        "method": (
            "Window means; standard errors by batch means over 30 ns blocks (captures autocorrelated plasma fluctuations, not only "
            "counting noise); z = difference / sqrt(SE_a^2 + SE_b^2); |z| < 2 is consistent at the 95 % level. shot_noise_rel is the "
            "pure counting-noise sigma 1/sqrt(macro events in the window) for reference: the real run-to-run spread is larger because "
            "the fluctuations are correlated and both runs are still drifting."
        ),
    }


def _history_row(case_dir: Path, protocol_path: Path, label: str, note: str) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    summary_sha = _verify_sidecar(summary_path)
    summary = read_canonical_json(summary_path)
    # history rows may legitimately predate the current protocol file (attempt 1 ran under the
    # attempt-1 protocol); the hash recorded by the run is embedded and the mismatch is flagged
    protocol_matches = summary.get("protocol_sha256") == _file_sha256(protocol_path)
    neutral = summary.get("neutral_inventory") or {}
    final = summary["final_series"]
    grid = summary["provenance"]["config"]["grid"]
    return {
        "label": label,
        "id": summary["case"]["id"],
        "experiment_id": summary["experiment_id"],
        "model_version": summary.get("model_version"),
        "grid": f"{int(grid['radial_cells'])}x{int(grid['axial_cells'])}",
        "macro_weight": summary["provenance"]["config"]["macro_weight"],
        "seed_plasma_density_per_m3": summary["provenance"]["config"]["seed_plasma"].get("density_per_m3"),
        "neutral_density_initial_per_m3": summary["provenance"]["config"]["mcc"].get("neutral_density_per_m3"),
        "neutral_density_final_per_m3": neutral.get("final_density_per_m3"),
        "steps_completed": summary["steps_completed"],
        "simulated_time_s": summary["simulated_time_s"],
        "ion_transit_times": summary.get("ion_transit_times"),
        "stop_reason": summary["stop_reason"],
        "final_electrons": summary["final_counts"]["electrons"],
        "final_ions": summary["final_counts"]["ions"],
        "final_discharge_a": final["currents_a"]["discharge_a"],
        "final_exit_ion_beam_a": final["currents_a"].get("exit_ion_beam_a"),
        "final_ionization_rate_per_s": (final.get("neutral") or {}).get("ionization_rate_per_s", final["currents_a"].get("ionization_rate_per_s")),
        "n_e_peak_per_m3": summary["window_maps_summary"].get("n_e_peak_per_m3"),
        "plateau": summary.get("plateau"),
        "summary_sha256": summary_sha,
        "protocol_sha256_at_run": summary.get("protocol_sha256"),
        "protocol_matches_current_file": protocol_matches,
        "note": note,
    }


def build_snapshot_v2_digest(results: Path = SNAPSHOT_V2 / "results") -> dict[str, Any] | None:
    manifest_path = results / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_canonical_json(manifest_path)
    manifest_sha = _verify_sidecar(manifest_path)
    rows = []
    for case, entry in manifest["cases"].items():
        summary_path = results / case / "summary.json"
        if _verify_sidecar(summary_path) != entry["summary_sha256"]:
            raise ValueError(f"{case}: snapshot v2 summary SHA-256 differs from manifest")
        summary = read_canonical_json(summary_path)
        grid = summary["provenance"]["config"]["grid"]
        plateau = summary.get("plateau") or {}
        rows.append({
            "id": case,
            "grid": f"{int(grid['radial_cells'])}x{int(grid['axial_cells'])}",
            "macro_weight": summary["provenance"]["config"]["macro_weight"],
            "steps_completed": summary["steps_completed"],
            "simulated_time_s": summary["simulated_time_s"],
            "ion_transit_times": summary.get("ion_transit_times"),
            "stop_reason": summary["stop_reason"],
            "n_e_peak_per_m3": summary["window_maps_summary"].get("n_e_peak_per_m3"),
            "window_discharge_a": (summary.get("window_currents_a") or {}).get("discharge_a"),
            "electron_count_drift": plateau.get("electron_count_drift"),
            "summary_sha256": entry["summary_sha256"],
        })
    return {
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": manifest_sha,
        "model_version": manifest.get("model_version"),
        "cases": rows,
        "lesson": (
            "Snapshot v2 (model v1.1, static n_g = 1e20 m^-3) ignited from the same 3 mA injection and 5e16 m^-3 seed "
            "but had no saturation channel: the electron count kept growing (~ +40 % over the trailing 20 %) and the "
            "window peak density passed 3.7-5.9 x n_max within one ion transit time. Model v1.3 replaced the static "
            "background by the quasi-steady inventory so that ionisation depletes the neutrals; the plateau above is "
            "the fixed point of that depletion balance."
        ),
    }


def build_payload(results: Path = RESULTS, protocol_path: Path = PROTOCOL, variants_path: Path | None = VARIANTS,
                  reference_v1: Path | None = REFERENCE_V1, snapshot_v2: Path | None = SNAPSHOT_V2 / "results",
                  snapshot_v1_results: Path | None = snapshot_dashboard.HISTORY_RESULTS,
                  snapshot_v1_protocol: Path | None = snapshot_dashboard.HISTORY_PROTOCOL) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = _file_sha256(protocol_path)
    raw_headline: dict[str, Any] = {}
    headline = build_case(results, protocol_path, label="v1.3 plateau (attempt 2)", role="headline", raw_out=raw_headline)
    cases = [headline]
    comparisons: list[dict[str, Any]] = []
    variants: dict[str, Any] = {}
    variant_status: list[dict[str, Any]] = []
    if variants_path is not None and variants_path.is_file():
        variants_doc = json.loads(variants_path.read_text(encoding="utf-8"))
        variants = variants_doc.get("variants", {})
        for name, spec in variants.items():
            case_dir = results.parent / f"results-{name}"
            run_state_path = case_dir / "run_state.json"
            finished = run_state_path.is_file() and bool(json.loads(run_state_path.read_text(encoding="utf-8")).get("finished")) and (case_dir / "summary.json").is_file()
            entry = {"name": name, "results_dir": case_dir.name, "note": spec.get("note"), "overrides": {k: v for k, v in spec.items() if k not in ("note",)},
                     "state": "finished" if finished else ("running_or_pending" if case_dir.is_dir() else "not_started")}
            if finished:
                raw_variant: dict[str, Any] = {}
                case = build_case(case_dir, protocol_path, label=f"variant {name}", role="variant", protocol_sha_expected=protocol_sha, raw_out=raw_variant)
                cases.append(case)
                comparisons.append(build_comparison(raw_headline, raw_variant, headline, case))
                entry["reached_plateau"] = bool((case["plateau"] or {}).get("reached"))
                entry["transit_times"] = case["ion_transit_times"]
            variant_status.append(entry)
    # convergence table across finished cases (window-averaged quantities)
    convergence: dict[str, Any] = {}
    quantities = {
        "discharge_a": lambda c: (c["window_currents_a"] or {}).get("discharge_a"),
        "exit_ion_beam_a": lambda c: (c["window_currents_a"] or {}).get("exit_ion_beam_a"),
        "ionization_rate_per_s": lambda c: (c["window_currents_a"] or {}).get("ionization_rate_per_s"),
        "neutral_density_per_m3": lambda c: (c["neutral_inventory"] or {}).get("trailing_20pct_mean_density_per_m3"),
        "n_e_mean_per_m3": lambda c: c["window_maps_summary"].get("n_e_mean_per_m3"),
        "n_e_peak_per_m3": lambda c: c["window_maps_summary"].get("n_e_peak_per_m3"),
        "t_e_density_weighted_mean_ev": lambda c: c["window_maps_summary"].get("t_e_density_weighted_mean_ev"),
        "phi_max_v": lambda c: c["window_maps_summary"].get("phi_max_v"),
    }
    for name, getter in quantities.items():
        values = {c["id"]: getter(c) for c in cases}
        finite = [v for v in values.values() if v is not None and isfinite(v)]
        spread = (max(finite) - min(finite)) / abs(np.mean(finite)) if len(finite) >= 2 and np.mean(finite) != 0 else None
        convergence[name] = {"values": values, "relative_spread": None if spread is None else float(f"{spread:.4g}")}
    history_rows: list[dict[str, Any]] = []
    if reference_v1 is not None and (reference_v1 / "results" / "summary.json").is_file():
        history_rows.append(_history_row(reference_v1 / "results", reference_v1 / "protocol.json", "v1.2 reference (no ignition)",
                                         "static-inventory model v1.2 at n_g = 1.5e19 m^-3 with a 1e16 m^-3 seed: the beam mirrored back, the seed decayed to a beam-transit floor; finalized after 3.4 transit times as the no-ignition reference"))
    for attempt in protocol.get("attempts", []):
        attempt_dir = results.parent / attempt["results_dir"]
        if attempt_dir == results or not (attempt_dir / "summary.json").is_file():
            continue
        history_rows.append(_history_row(attempt_dir, protocol_path, f"v1.3 attempt {attempt['attempt']} (no ignition)", attempt.get("diagnosis") or attempt.get("outcome")))
    snapshot_v2_digest = build_snapshot_v2_digest(snapshot_v2) if snapshot_v2 is not None else None
    snapshot_v1_history = None
    if snapshot_v1_results is not None and snapshot_v1_protocol is not None and (snapshot_v1_results / "manifest.json").is_file():
        snapshot_v1_history = snapshot_dashboard.build_history(snapshot_v1_results, snapshot_v1_protocol)
    neutral = headline["neutral_inventory"] or {}
    op = protocol["operating_point"]
    feed = neutral.get("feed_atoms_per_s", op["neutral_inventory"]["feed_atoms_per_s"])
    payload = {
        "schema": SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "model_version": protocol["model_version"],
        "status": protocol["status"],
        "claim_boundary": protocol["claim_boundary"],
        "claim_statement": (
            "Development steady state (model v1.3); the headline is a single seed and the convergence pair has completed: "
            "seed-b (statistical spread <= 1 % on the window currents over the common window) and W x 0.7 (particle-resolution "
            "sensitivity: I_d +4.7 %, I_beam +2.1 %, S -4.5 %, wall currents -7.5 %, peak n_e -12 % over the common window), so the "
            "plateau quantities carry a ~5 % resolution band (peak density ~10 %); "
            "not preregistered; not validated against any experiment; not a thruster performance prediction. The plateau "
            "criterion (< 5 % drift of I_d, N_e and n_g over the trailing 20 %, after >= 3 ion transit times) held with the "
            "electron-count drift at 4.98 % - marginal, reported as such. The window peak density is 4.1 x the a-priori "
            "resolvability ceiling n_max (dz = 3.0 lambda_D at the peak node, omega_pe dt up to 0.118 against the 0.2 gate): the "
            "peak region between the 12.0 and 17.95 mm cusps is under-resolved and the mean density (0.93 x the projected "
            "0-D equilibrium) is the quantity inside the budget. The neutral transient is artificial; only its fixed point is physical. "
            "Numerics verified by the tests in modern/tests/pic2d; physics simplified as listed."
        ),
        "simplifications": protocol["simplifications"],
        "protocol": {
            "file_sha256": protocol_sha,
            "variants_file_sha256": _file_sha256(variants_path) if variants_path is not None and variants_path.is_file() else None,
            "operating_point": op,
            "numerics": protocol["numerics"],
            "geometry": protocol["geometry"],
            "stopping_rule": protocol["stopping_rule"],
            "attempts": protocol.get("attempts", []),
            "model_spec": protocol.get("model_spec"),
        },
        "budget": protocol.get("budget_v1_3"),
        "operating_point_summary": {
            "feed_atoms_per_s": feed,
            "mass_flow_mg_per_s": neutral.get("mass_flow_mg_per_s"),
            "effusion_coefficient_m3_per_s": neutral.get("effusion_coefficient_m3_per_s"),
            "zero_ionization_density_per_m3": neutral.get("zero_ionization_density_per_m3"),
            "relaxation_time_s": neutral.get("relaxation_time_s"),
            "physical_time_constant_s": neutral.get("physical_time_constant_s"),
            "propellant_utilisation_trailing": neutral.get("propellant_utilisation_trailing"),
            "beam_fraction_of_discharge": (headline["window_currents_a"]["exit_ion_beam_a"] / headline["window_currents_a"]["discharge_a"]) if headline["window_currents_a"].get("discharge_a") else None,
            "ionisations_per_injected_electron": headline["window_currents_a"]["ionization_rate_per_s"] / (headline["window_currents_a"]["injected_electron_a"] / E_CHARGE) if headline["window_currents_a"].get("injected_electron_a") else None,
            "ion_transit_time_s": (protocol.get("budget_v1_3") or {}).get("ion_transit_time_s"),
            "exit_area_m2": neutral.get("exit_area_m2", pi * protocol["geometry"].get("exit_radius_m", 3e-3) ** 2 if isinstance(protocol.get("geometry"), dict) else None),
        },
        "convergence": convergence,
        "comparisons": comparisons,
        "variants": variant_status,
        "cases": cases,
        "history": {
            "steady_state": history_rows,
            "snapshot_v2": snapshot_v2_digest,
            "snapshot_v1": snapshot_v1_history,
            "lesson": (
                "Ignition needed a seed plasma dense enough to build the potential structure that traps the 300 V beam "
                "electrons in the cusped field: with a 1e16 m^-3 seed (v1.2 reference at 1.5e19, v1.3 attempt 1 at 5e19) "
                "91-96 % of the injected electrons mirrored back to the exit plane and the seed cooled and decayed; with the "
                "5e16 m^-3 seed at 5.5e19 the beam was absorbed from the first 100 ns, S grew 2.9e15 -> 3.9e16 s^-1 and the "
                "discharge settled at 3.4 mA with 46 % propellant utilisation and 67 % of the discharge current leaving as "
                "ion beam. The neutral density fell to 54 % of its zero-ionisation value and sat on the analytic fixed point "
                "(Q_in - S)/c to 0.03 %."
            ),
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema", "experiment_id", "model_version", "status", "claim_boundary", "claim_statement", "simplifications", "protocol",
        "budget", "operating_point_summary", "convergence", "comparisons", "variants", "cases", "history",
    }
    if set(payload) != required:
        raise ValueError("payload keys do not match the closed schema")
    if payload["schema"] != SCHEMA:
        raise ValueError("unsupported payload schema")
    if payload["status"] != "development_screening_not_preregistered":
        raise ValueError("payload must carry the development/screening status")
    statement = payload["claim_statement"].lower()
    if not payload["simplifications"] or "not preregistered" not in statement or "not validated" not in statement:
        raise ValueError("claim boundary must be explicit")
    if not payload["cases"] or payload["cases"][0]["role"] != "headline":
        raise ValueError("payload must contain the headline case first")
    if not isinstance(payload["protocol"]["file_sha256"], str) or len(payload["protocol"]["file_sha256"]) != 64:
        raise ValueError("protocol file_sha256 must be a SHA-256")
    for case in payload["cases"]:
        for key in ("summary_sha256", "maps_npz_sha256", "series_npz_sha256", "protocol_sha256"):
            digest = case[key]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{case['id']}: {key} must be a SHA-256")
        if case["protocol_sha256"] != payload["protocol"]["file_sha256"]:
            raise ValueError(f"{case['id']}: case protocol hash differs from the protocol file")
        if case["stop_reason"] not in STOP_REASONS:
            raise ValueError(f"{case['id']}: unknown stop reason")
        if case["plateau"] is None or case["ledger"] is None or case["neutral_inventory"] is None:
            raise ValueError(f"{case['id']}: plateau, ledger and neutral inventory are required")
        nr, nz = len(case["grid_r_m"]), len(case["grid_z_m"])
        for key in MAP_KEYS:
            matrix = case["maps"][key]
            if len(matrix) != nr or any(len(row) != nz for row in matrix):
                raise ValueError(f"{case['id']}: map {key} shape does not match the grid")
        snapshot_dashboard.validate_sampling(case)
        n = len(case["series"]["time_s"])
        for key, values in case["series"].items():
            if len(values) != n:
                raise ValueError(f"{case['id']}: series {key} length differs from time_s")
    for row in payload["history"]["steady_state"]:
        if row["stop_reason"] not in STOP_REASONS or len(row["summary_sha256"]) != 64:
            raise ValueError(f"history {row['id']}: unknown stop reason or bad digest")
    for entry in payload["variants"]:
        if entry["state"] not in ("finished", "running_or_pending", "not_started"):
            raise ValueError(f"variant {entry['name']}: unknown state")
    case_ids = {case["id"] for case in payload["cases"]}
    finished = [case["id"] for case in payload["cases"] if case["role"] == "variant"]
    if [c["other_id"] for c in payload["comparisons"]] != finished:
        raise ValueError("every finished variant needs exactly one comparison against the headline")
    for comparison in payload["comparisons"]:
        if comparison["base_id"] != payload["cases"][0]["id"] or comparison["other_id"] not in case_ids:
            raise ValueError("comparison ids must reference embedded cases")
        if not comparison["windows"] or not comparison["windows"][0]["rows"]:
            raise ValueError("comparison must carry the common-window rows")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIC-2D CFT steady state (development)</title>
<style>
:root{color-scheme:dark;--bg:#07100f;--panel:#0f1c1a;--panel2:#14262380;--text:#eef7f4;--muted:#9bb8b0;--line:#2b4540;--accent:#5ad6c0;--warn:#ffcf67;--red:#ff6b6b;--blue:#58a8ff;--shadow:#0008;--window:#5ad6c022}
[data-theme=light]{color-scheme:light;--bg:#edf5f2;--panel:#fff;--panel2:#f2f8f6;--text:#10231f;--muted:#4f6a63;--line:#bfd3cc;--accent:#087f6e;--warn:#7a5700;--red:#b83232;--blue:#176db5;--shadow:#3452;--window:#087f6e22}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#153b34 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
button,select{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:.55rem;padding:.48rem .7rem}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
header,main,footer{width:min(1500px,calc(100% - 2rem));margin:auto}header{padding:2rem 0 1rem}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(1.9rem,4.5vw,3.8rem);line-height:.98;margin:.2rem 0 .8rem;max-width:960px}h2{margin:.1rem 0 .8rem;font-size:1.1rem}p{margin:.35rem 0}
.claim{border:1px solid #8b681c;background:#513d1438;color:var(--warn);padding:.8rem 1rem;border-radius:.65rem;font-weight:650}.claim ul{margin:.4rem 0 0 1.1rem;font-weight:500;color:var(--text)}
.controls{display:flex;gap:.6rem;flex-wrap:wrap;align-items:end;margin:1rem 0}.control{display:grid;gap:.25rem}.control label{color:var(--muted);font-size:.8rem}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,.8fr);gap:1rem;margin:1rem 0}.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:.9rem;padding:1rem;box-shadow:0 12px 30px var(--shadow);min-width:0}
.canvas-wrap{position:relative;min-height:300px}.canvas-wrap canvas{width:100%;height:clamp(300px,34vw,460px);display:block}.tip{position:absolute;pointer-events:none;background:#07100fee;color:#fff;border:1px solid #7f9a93;border-radius:.35rem;padding:.35rem .5rem;display:none;white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.65rem}.metric-card{border:1px solid var(--line);border-radius:.7rem;padding:.75rem;background:var(--panel);min-width:0}.metric-card.active{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}.metric-card h3{font-size:.95rem;margin:0 0 .55rem}
.kv{display:grid;grid-template-columns:1fr auto;gap:.22rem .6rem}.kv span{min-width:0;overflow-wrap:anywhere}.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){font-variant-numeric:tabular-nums;text-align:right}h1,h2,h3,p,li{overflow-wrap:anywhere}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.plot{width:100%;height:260px;display:block}.wide{grid-column:1/-1}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;color:var(--muted)}code{font-size:.78em;overflow-wrap:anywhere}.small{font-size:.82rem;color:var(--muted)}footer{padding:1rem 0 2.5rem;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-weight:600}td,th{padding:.15rem .4rem;border-bottom:1px solid var(--line)}.ok{color:var(--accent)}.marginal{color:var(--warn)}.bad{color:var(--red)}
@media(max-width:900px){.grid,.plots{grid-template-columns:1fr}.canvas-wrap canvas{height:360px}}@media(max-width:520px){header,main,footer{width:min(100% - 1rem,1500px)}.canvas-wrap canvas{height:300px}.panel{padding:.7rem}}
</style>
</head>
<body>
<header><div class="eyebrow">PIC-MCC · axisymmetric (r,z) · development steady state · model v1.3</div><h1>Divergent-exit CFT channel: first kinetic plateau with a self-consistent neutral inventory</h1>
<div id="claim" class="claim" role="note"></div>
<div class="controls">
<div class="control"><label for="case">Case</label><select id="case"></select></div>
<div class="control"><label for="map">Map (time-averaged over the plateau window)</label><select id="map"><option value="n_e_per_m3">Electron density n_e (m⁻³)</option><option value="n_i_per_m3">Ion density n_i (m⁻³)</option><option value="phi_v">Potential φ (V)</option><option value="t_e_ev">Electron temperature T_e (eV)</option><option value="ionization_rate_per_m3_s">Ionisation rate (m⁻³ s⁻¹)</option></select></div>
<div class="control"><label for="scale">Colour scale</label><select id="scale"><option value="linear">linear</option><option value="log">log10</option></select></div>
__MAP_CONTROLS__
<button id="theme" type="button" aria-pressed="false">Light theme</button>
</div><p class="small">Keyboard: 1–4 select cases; arrow keys move the map cursor; Home resets the cursor. Dashed verticals on the map and wall plots are the cusp planes (B_z = 0 on axis); the shaded band on the time series is the trailing-20 % plateau window, the dotted line the 3-transit floor.</p></header>
<main>
<section class="metrics" id="metrics" aria-label="Case metrics"></section>
<section class="panel" style="margin:1rem 0"><h2>Plateau verification and window-averaged final state</h2><div id="verification"></div></section>
<section class="grid">
<div class="panel"><h2 id="mapTitle">Plateau-window map</h2><div class="canvas-wrap"><canvas id="field" tabindex="0" role="img" aria-label="Interactive (r,z) heatmap of the selected plateau-window quantity"></canvas><div id="tip" class="tip" role="status" aria-live="polite"></div></div><p class="small" id="mapCaption"></p><p class="small">Canvas raster of the node grid (radial-major). White: dielectric/outside the plasma cell mask; grey: sampled by fewer macro-particles than the threshold (the "speckle" of a log map is the counting noise of those cells, not structure). Straight bore wall at r = 2 mm is exact; the cone is a one-cell stair-step. Anode at z = 0 (fixed potential), exit plane at z = 24 mm (0 V reference). Dashed lines: cusp planes.</p></div>
<aside class="panel"><h2 id="detailTitle">Case details</h2><div id="details"></div></aside>
</section>
<section class="plots">
<div class="panel"><h2>Macro-particle counts to the plateau</h2><canvas class="plot" id="counts" role="img" aria-label="Electron and ion macro-particle counts versus time with the plateau window marked"></canvas></div>
<div class="panel"><h2>Currents</h2><canvas class="plot" id="currents" role="img" aria-label="Discharge, exit ion beam and wall currents versus time"></canvas></div>
<div class="panel"><h2>Neutral density vs analytic fixed point</h2><canvas class="plot" id="neutral" role="img" aria-label="Neutral density and the analytic fixed point (Q_in - S)/c versus time"></canvas><p class="small">n_g relaxes toward n_g* = (Q_in − S)/c with the artificial τ_g = 30 ns (the physical V/c is 221 µs): the transient is a numerical device, only the fixed point is physical. The dashed line is the zero-ionisation ceiling n_g0 = Q_in/c.</p></div>
<div class="panel"><h2>Atom rates and utilisation</h2><canvas class="plot" id="rates" role="img" aria-label="Ionisation, effusion and artificial atom rates versus time"></canvas></div>
<div class="panel"><h2>Potential range</h2><canvas class="plot" id="phi" role="img" aria-label="Minimum, mean and maximum potential versus time"></canvas></div>
<div class="panel"><h2>Energy ledger</h2><canvas class="plot" id="energy" role="img" aria-label="Kinetic, field and total energy with the electrode work and the interval ledger residual"></canvas><p class="small">Residual = Δ(K+U) − (injected − absorbed − inelastic + born-ion kinetic energy + electrode work) per interval, where electrode work = Σ V_k (ΔQ_induced,k − q_absorbed,k) is the energy the 300 V supply delivers. What remains is the momentum-conserving scheme's numerical non-conservation (grid heating / self-force with finite particles per cell); it is reported, not hidden.</p></div>
<div class="panel"><h2>Wall impact flux along the dielectric</h2><canvas class="plot" id="wall" role="img" aria-label="Electron and ion wall flux versus axial position with cusp planes"></canvas></div>
<div class="panel"><h2>Axial ion current density at the exit plane</h2><canvas class="plot" id="exit" role="img" aria-label="Exit-plane ion current density versus radius"></canvas></div>
<div class="panel"><h2>Axial profile of the radial-maximum density vs cusps</h2><canvas class="plot" id="axial" role="img" aria-label="Radial maximum of the electron density versus axial position with cusp planes"></canvas></div>
<div class="panel"><h2>Stability metric</h2><canvas class="plot" id="wpe" role="img" aria-label="Peak plasma-frequency times timestep versus time"></canvas></div>
<div class="panel wide"><h2>Convergence pair and statistical variance</h2><div id="convergence"></div></div>
</section>
<section class="panel" style="margin:1rem 0"><h2>Neutral ledger and operating-point budget</h2><div id="budget"></div></section>
<section class="panel" style="margin:1rem 0"><h2>Development history: how the plateau was reached</h2><div id="history"></div></section>
<section class="panel" style="margin:1rem 0"><h2>Simplifications (model v1.3) and identity</h2><div id="identity"></div></section>
</main><footer>Self-contained offline dashboard generated by <code>modern/visualization/generate_pic2d_cft_steady_state.py</code>. Development evidence only: single seed until the convergence pair completes, not preregistered, not validated.</footer>
<script id="pic2d-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("pic2d-data").textContent);
const $=id=>document.getElementById(id);let selected=0,mapKey="n_e_per_m3",scaleMode="linear",cursor=null,raf=0;
const caseSelect=$("case");DATA.cases.forEach((c,i)=>{const o=document.createElement("option");o.value=i;o.textContent=c.label;caseSelect.append(o)});
const fmt=(v,n=4)=>v==null||!isFinite(v)?"–":Number(v).toLocaleString(undefined,{maximumSignificantDigits:n});
const sci=(v,n=3)=>v==null||!isFinite(v)?"–":Number(v).toExponential(n-1);
const pct=(v,n=3)=>v==null||!isFinite(v)?"–":fmt(v*100,n)+" %";
$("claim").innerHTML=`<strong>Claim boundary:</strong> ${DATA.claim_statement}<ul>${DATA.simplifications.map(s=>`<li>${s}</li>`).join("")}</ul>`;
const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function setup(canvas){const r=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);return {c,w:r.width,h:r.height}}
function color(t,signed){t=Math.max(0,Math.min(1,t));if(signed){if(t<.5){const q=t*2;return `rgb(${Math.round(35+220*q)},${Math.round(92+163*q)},255)`}const q=(t-.5)*2;return `rgb(255,${Math.round(255-210*q)},${Math.round(255-215*q)})`}return `rgb(${Math.round(12+240*t)},${Math.round(28+190*Math.sqrt(t))},${Math.round(90+100*(1-t))})`}
__MAP_VIEW_JS__
function windowOf(c){const t0=c.series.time_s[0],t1=c.simulated_time_s,f=c.plateau&&c.plateau.window_fraction!=null?c.plateau.window_fraction:.2;return {start:(t1-f*(t1-t0))*1e6,end:t1*1e6,transit3:DATA.operating_point_summary.ion_transit_time_s?3*DATA.operating_point_summary.ion_transit_time_s*1e6:null}}
function drift(v){return v==null?"–":`<span class="${Math.abs(v)<.04?"ok":Math.abs(v)<.05?"marginal":"bad"}">${fmt(v*100,3)} %</span>`}
function renderMetrics(){const root=$("metrics");root.textContent="";DATA.cases.forEach((c,i)=>{const w=c.window_maps_summary,card=document.createElement("article");card.className="metric-card"+(i===selected?" active":"");card.tabIndex=0;card.setAttribute("role","button");card.setAttribute("aria-pressed",i===selected);const pl=c.plateau||{},lg=c.ledger||{},wc=c.window_currents_a||{},ni=c.neutral_inventory||{};card.innerHTML=`<h3>${c.label}</h3><div class="kv"><span>role</span><span>${c.role} · seed ${c.case.seed} · W ${sci(c.config.macro_weight,2)}</span><span>steps / time</span><span>${c.steps_completed} · ${fmt(c.simulated_time_s*1e6,3)} µs (${fmt(c.ion_transit_times,3)} τ_i)</span><span>stop</span><span>${c.stop_reason.replaceAll("_"," ")}</span><span>plateau drifts I_d / N_e / n_g</span><span>${drift(pl.discharge_current_drift)} / ${drift(pl.electron_count_drift)} / ${drift(pl.neutral_density_drift)}</span><span>I_d · I_beam,i (window)</span><span>${fmt(wc.discharge_a*1e3,3)} · ${fmt(wc.exit_ion_beam_a*1e3,3)} mA</span><span>S · utilisation S/Q_in</span><span>${sci(wc.ionization_rate_per_s,3)} s⁻¹ · ${pct(ni.propellant_utilisation_trailing,3)}</span><span>n_g (window) / fixed point</span><span>${sci(ni.trailing_20pct_mean_density_per_m3,4)} / ${sci(ni.trailing_20pct_analytic_fixed_point_per_m3,4)}</span><span>peak / mean n_e</span><span>${sci(w.n_e_peak_per_m3)} / ${sci(w.n_e_mean_per_m3)} m⁻³</span><span>⟨T_e⟩_n · φ range</span><span>${fmt(w.t_e_density_weighted_mean_ev,3)} eV · ${fmt(w.phi_min_v,3)}…${fmt(w.phi_max_v,3)} V</span><span>ledger residual / electrode work</span><span>${pct(lg.cumulative_residual_over_electrode_work,3)}</span><span>wall / throughput</span><span>${fmt(c.wall_seconds_total/3600,3)} h · ${fmt(c.ms_per_step_last_session,3)} ms/step</span></div>`;card.onclick=()=>select(i);card.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select(i)}};root.append(card)})}
function renderVerification(){const c=DATA.cases[selected],pl=c.plateau||{},wc=c.window_currents_a||{},ni=c.neutral_inventory||{},w=c.window_maps_summary,lg=c.ledger||{},rs=c.resolvability_at_peak,ops=DATA.operating_point_summary,pk=c.peak_density_node,cusps=c.cusps.cusp_z_m;
const rows=[["ion transit times elapsed (floor 3)",`${fmt(pl.transit_times_elapsed,3)} ${pl.transit_times_elapsed>=3?'<span class="ok">≥ 3</span>':'<span class="bad">&lt; 3</span>'}`],["trailing-20 % drift I_d (threshold 5 %)",drift(pl.discharge_current_drift)],["trailing-20 % drift N_e",drift(pl.electron_count_drift)],["trailing-20 % drift n_g",drift(pl.neutral_density_drift)],["plateau declared",pl.reached?'<span class="ok">yes</span>':'<span class="bad">no</span>'],["averaging window (steps · duration)",c.averaging_window_step_range?`${c.averaging_window_step_range[0]}–${c.averaging_window_step_range[1]} (${c.averaging_window_steps} steps · ${duration(c.sampling.window_s)})`:"–"],["I_d (anode e⁻ − anode Xe⁺)",`${fmt(wc.discharge_a*1e3,4)} mA`],["I_beam,i (exit plane)",`${fmt(wc.exit_ion_beam_a*1e3,4)} mA = ${pct(ops.beam_fraction_of_discharge,3)} of I_d`],["wall Xe⁺ / wall e⁻ / exit e⁻ / injected",`${fmt(wc.wall_ion_a*1e3,3)} / ${fmt(wc.wall_electron_a*1e3,3)} / ${fmt(wc.exit_electron_a*1e3,3)} / ${fmt(wc.injected_electron_a*1e3,3)} mA`],["S (ionisation rate)",`${sci(wc.ionization_rate_per_s,4)} s⁻¹ = ${fmt(wc.ionization_rate_per_s*1.602176634e-19*1e3,3)} mA equivalent; ${fmt(ops.ionisations_per_injected_electron,3)} ionisations per injected electron`],["utilisation S / Q_in",pct(ni.propellant_utilisation_trailing,3)],["n_g window mean / analytic fixed point (Q_in − S)/c",`${sci(ni.trailing_20pct_mean_density_per_m3,4)} / ${sci(ni.trailing_20pct_analytic_fixed_point_per_m3,4)} m⁻³ (distance ${pct((ni.trailing_20pct_mean_density_per_m3-ni.trailing_20pct_analytic_fixed_point_per_m3)/ni.trailing_20pct_mean_density_per_m3,2)}; n_g/n_g0 = ${fmt(ni.trailing_20pct_mean_density_per_m3/ni.zero_ionization_density_per_m3,3)})`],["peak / mean n_e (window maps)",`${sci(w.n_e_peak_per_m3,4)} / ${sci(w.n_e_mean_per_m3,4)} m⁻³ (peak = ${fmt(rs.n_e_peak_over_n_max,3)} × n_max; mean = ${fmt(c.budget_check?c.budget_check.n_e_mean_over_projected_n_eq:null,3)} × projected 0-D n_eq)`],["⟨T_e⟩ (density-weighted) · T_e max",`${fmt(w.t_e_density_weighted_mean_ev,3)} eV · ${fmt(w.t_e_max_ev,3)} eV`],["φ range (window map)",`${fmt(w.phi_min_v,4)} … ${fmt(w.phi_max_v,4)} V (anode ${c.config.potentials.anode_v} V)`],["energy-ledger residual (cumulative, with electrode work)",`${sci(lg.cumulative_residual_j,3)} J = ${pct(lg.cumulative_residual_over_electrode_work,3)} of the electrode work ${sci(lg.cumulative_electrode_work_j,3)} J; interval RMS ${sci(lg.interval_residual_rms_j,3)} J`],["neutral-ledger closure",`${sci(ni.cumulative_ledger_closure_atoms,3)} atoms = ${sci(ni.cumulative_ledger_closure_relative_to_inventory,2)} of the inventory; max interval residual ${sci(ni.max_interval_ledger_residual_atoms,2)} atoms`],["peak n_e node",`z = ${fmt(pk.z_m*1e3,4)} mm, r = ${fmt(pk.r_m*1e3,3)} mm; cusp planes at z = ${cusps.map(v=>fmt(v*1e3,4)).join(", ")} mm (magnet mid-planes ${c.cusps.magnet_midplane_z_m.map(v=>fmt(v*1e3,3)).join(", ")} mm)`],["resolvability at the peak node",`<span class="${rs.dz_over_lambda_d_at_peak>2?"bad":"ok"}">Δz/λ_D = ${fmt(rs.dz_over_lambda_d_at_peak,3)}, Δr/λ_D = ${fmt(rs.dr_over_lambda_d_at_peak,3)}</span> (λ_D = ${fmt(rs.lambda_d_at_peak_m*1e6,3)} µm at ⟨T_e⟩); ω_pe Δt at the peak ${fmt(rs.omega_pe_dt_at_peak,3)}, max observed ${fmt(rs.max_observed_omega_pe_dt,3)} (gate 0.2)`],["particles (final e⁻ / Xe⁺; peak)",`${c.final_counts.electrons} / ${c.final_counts.ions}; peak ${c.peak_counts?c.peak_counts.electrons+" / "+c.peak_counts.ions:"–"}`],["wall time · throughput",`${fmt(c.wall_seconds_total,5)} s (${fmt(c.wall_seconds_total/3600,3)} h) · ${fmt(c.ms_per_step_last_session,3)} ms/step (last session) · ${c.sessions?c.sessions.length:"–"} session(s)`]];
$("verification").innerHTML=`<table aria-label="Plateau verification and final state"><tbody>${rows.map(([k,v])=>`<tr><th>${k}</th><td>${v}</td></tr>`).join("")}</tbody></table><p class="small">Drift = linear-fit slope × window / |mean| over the trailing 20 % of the simulated time, computed from the full-resolution series (the embedded series is decimated ×${c.series_stride} for display). Green &lt; 4 %, amber 4–5 % (passed but marginal), red ≥ 5 %.</p>`}
function renderDetails(){const c=DATA.cases[selected],g=c.stability_gate,op=DATA.protocol.operating_point,ops=DATA.operating_point_summary;let html=`<div class="kv"><span>backend</span><span>${c.backend}</span><span>Δr × Δz</span><span>${fmt(c.config.grid.dr_m*1e6,3)} × ${fmt(c.config.grid.dz_m*1e6,3)} µm</span><span>grid</span><span>${c.config.grid.radial_cells}×${c.config.grid.axial_cells}</span><span>Δt · ion subcycle</span><span>${sci(c.config.dt_s,3)} s · k = ${DATA.protocol.numerics.ion_subcycle}</span><span>macro weight · seed</span><span>${sci(c.config.macro_weight,2)} · ${c.case.seed}</span><span>anode / exit</span><span>${c.config.potentials.anode_v} / ${c.config.potentials.exit_v} V</span><span>e⁻ injection</span><span>${op.electron_injection_current_a*1e3} mA @ ${op.electron_injection_temperature_ev} eV</span><span>seed plasma</span><span>${sci(op.seed_plasma_density_per_m3,2)} m⁻³ @ ${op.seed_electron_temperature_ev} eV</span><span>n_g0 = Q_in / c</span><span>${sci(ops.zero_ionization_density_per_m3,2)} m⁻³</span><span>Q_in (feed)</span><span>${sci(ops.feed_atoms_per_s,3)} s⁻¹ = ${fmt(ops.mass_flow_mg_per_s,4)} mg/s</span><span>c = v̄ A_exit / 4</span><span>${sci(ops.effusion_coefficient_m3_per_s,4)} m³/s</span><span>τ_g (artificial) · V/c</span><span>${sci(ops.relaxation_time_s,2)} s · ${sci(ops.physical_time_constant_s,3)} s</span><span>ion transit time</span><span>${sci(ops.ion_transit_time_s,2)} s</span><span>GPU wall</span><span>${fmt(c.wall_seconds_total,5)} s</span><span>maps kind</span><span>${c.maps_kind||"–"}</span></div>`;
html+=`<h2 style="margin-top:1rem">Stability gate (configured reference)</h2><div class="kv"><span>ω_pe Δt</span><span>${fmt(g.omega_pe_dt,3)}</span><span>Ω_ce Δt</span><span>${fmt(g.omega_ce_dt,3)}</span><span>cell / λ_D</span><span>${fmt(g.cell_debye_ratio,3)}</span><span>Courant</span><span>${fmt(g.particle_courant,3)}</span><span>P_coll</span><span>${sci(g.max_collision_probability,2)}</span><span>max |B| on nodes</span><span>${fmt(g.max_b_t*1e3,4)} mT</span></div>`;
if(c.stability_gate_message)html+=`<p class="small"><strong>Fail-closed stop:</strong> ${c.stability_gate_message}</p>`;
$("detailTitle").textContent=c.label;$("details").innerHTML=html;$("mapTitle").textContent=`${$("map").selectedOptions[0].textContent} — ${c.label}`;
const cv=DATA.convergence,rows=Object.entries(cv).map(([k,v])=>`<tr><td>${k}</td>${DATA.cases.map(cc=>`<td>${sci(v.values[cc.id])}</td>`).join("")}<td>${v.relative_spread==null?"–":pct(v.relative_spread,3)}</td></tr>`).join("");
const vs=DATA.variants.map(v=>`<tr><td>${v.name}</td><td>${v.state.replaceAll("_"," ")}${v.state==="finished"?` · ${fmt(v.transit_times,3)} τ_i · plateau ${v.reached_plateau?"declared":"not declared"}`:""}</td><td><code>${v.results_dir}</code></td><td>${Object.entries(v.overrides).map(([k,x])=>`${k} = ${typeof x==="number"?sci(x,3):x}`).join(", ")}</td></tr>`).join("");
const zc=z=>z==null?"–":`<span class="${Math.abs(z)<2?"ok":Math.abs(z)<3?"marginal":"bad"}">${fmt(z,3)}</span>`;
const cmpHtml=DATA.comparisons.map(cp=>`<h3 style="margin:.8rem 0 .3rem">${cp.other_label} vs headline — ${cp.other_stop_reason.replaceAll("_"," ")} at ${fmt(cp.other_transit_times,3)} τ_i${cp.other_transit_times<3?" (no plateau declaration possible: &lt; 3 transits)":""}</h3>${cp.windows.map(w=>`<p class="small"><strong>${w.label}</strong> — ${fmt(w.t_start_s*1e6,4)}–${fmt(w.t_end_s*1e6,4)} µs${w.other_t_start_s!=null?` (variant ${fmt(w.other_t_start_s*1e6,4)}–${fmt(w.other_t_end_s*1e6,4)} µs)`:""}</p><table aria-label="${w.label}"><thead><tr><th>quantity</th><th>headline</th><th>variant</th><th>Δ</th><th>Δ rel.</th><th>SE (batch means) head. / var.</th><th>z</th><th>shot-noise σ_rel (ref.)</th></tr></thead><tbody>${w.rows.map(r=>`<tr><td>${r.quantity}${r.unit?" ("+r.unit+")":""}</td><td>${sci(r.base,4)}</td><td>${sci(r.other,4)}</td><td>${sci(r.abs_diff,3)}</td><td>${pct(r.rel_diff,3)}</td><td>${sci(r.se_base,2)} / ${sci(r.se_other,2)}</td><td>${zc(r.z)}</td><td>${r.shot_noise_rel==null?"–":pct(r.shot_noise_rel,2)}</td></tr>`).join("")}</tbody></table>`).join("")}<p class="small"><strong>Maps</strong> (headline window ${fmt(cp.maps.base_window_s[0]*1e6,4)}–${fmt(cp.maps.base_window_s[1]*1e6,4)} µs vs variant window ${cp.maps.other_window_s?fmt(cp.maps.other_window_s[0]*1e6,4)+"–"+fmt(cp.maps.other_window_s[1]*1e6,4):"–"} µs): ${cp.maps.note}.</p><table aria-label="Map comparison"><thead><tr><th>field</th><th>mean head. / var. (Δ rel.)</th><th>peak head. / var. (Δ rel.)</th><th>shape difference</th></tr></thead><tbody>${Object.entries(cp.maps.rows).map(([k,m])=>`<tr><td>${k}</td><td>${m.mean_base!=null?`${sci(m.mean_base,3)} / ${sci(m.mean_other,3)} (${pct(m.mean_rel_diff,3)})`:m.axis_base!=null?`axis ${sci(m.axis_base,3)} / ${sci(m.axis_other,3)} (${pct(m.axis_rel_diff,3)})`:m.total_rel_diff!=null?`total Δ ${pct(m.total_rel_diff,3)}`:"–"}</td><td>${m.peak_base!=null?`${sci(m.peak_base,3)} / ${sci(m.peak_other,3)} (${pct(m.peak_rel_diff,3)})${m.peak_z_base_m!=null?` at z ${fmt(m.peak_z_base_m*1e3,4)} / ${fmt(m.peak_z_other_m*1e3,4)} mm`:""}`:"–"}</td><td>${m.relative_l2_diff!=null?"rel. L2 "+pct(m.relative_l2_diff,3):m.rms_diff_v!=null?"RMS "+fmt(m.rms_diff_v,3)+" V":"–"}</td></tr>`).join("")}</tbody></table><p class="small">${cp.method}</p>`).join("");
$("convergence").innerHTML=`<p class="small">Window-averaged quantities across the finished cases at the same operating point (each over its own final window). Relative spread = (max−min)/|mean|. The plateau is a <em>single-seed</em> development result until the convergence pair — seed-b (statistical variance) and the reduced-weight case (particle-resolution sensitivity) — finishes; finished variants are embedded here with a same-time-window statistical comparison.</p><table aria-label="Convergence between cases"><thead><tr><th>quantity</th>${DATA.cases.map(cc=>`<th>${cc.label}</th>`).join("")}<th>spread</th></tr></thead><tbody>${rows}</tbody></table>${cmpHtml}<h3 style="margin:.8rem 0 .3rem">Convergence-pair status</h3><table aria-label="Variant status"><thead><tr><th>variant</th><th>state</th><th>results</th><th>overrides</th></tr></thead><tbody>${vs||'<tr><td colspan="4">no variants declared</td></tr>'}</tbody></table>${DATA.variants.map(v=>v.note?`<p class="small"><strong>${v.name}:</strong> ${v.note}</p>`:"").join("")}`;
const B=DATA.budget,ni=c.neutral_inventory||{},L=ni.cumulative_ledger_atoms||{};$("budget").innerHTML=`<div class="kv" style="grid-template-columns:repeat(auto-fit,minmax(230px,1fr))"><span>atoms fed</span><span>${sci(L.fed,4)}</span><span>atoms ionised</span><span>${sci(L.ionized,4)}</span><span>atoms effused (physical)</span><span>${sci(L.effused,4)}</span><span>atoms removed by the artificial relaxation</span><span>${sci(L.artificial,4)}</span><span>closure (fed − ionised − effused − artificial − ΔN)</span><span>${sci(ni.cumulative_ledger_closure_atoms,3)} atoms</span><span>trailing artificial rate (should → 0 at the fixed point)</span><span>${sci(ni.trailing_20pct_mean_artificial_rate_per_s,3)} s⁻¹ (${pct(ni.trailing_20pct_mean_artificial_rate_per_s/ops.feed_atoms_per_s,2)} of Q_in; = n_g − n_g* of ${sci(ni.trailing_20pct_mean_artificial_rate_per_s*ops.relaxation_time_s/c.mesh.plasma_volume_m3,2)} m⁻³)</span></div><p class="small">The artificial ledger equals the inventory drop from n_g0 to the fixed point (5.5e19 → 2.97e19 m⁻³ × V): physically that depletion would take ~V/c = 221 µs of effusion; τ_g = 30 ns does it in ~100 ns so the plasma sees a quasi-steady n_g. Only the fixed point (Q_in = S + c n_g) is physical, and the window mean sits on it to ${pct((ni.trailing_20pct_mean_density_per_m3-ni.trailing_20pct_analytic_fixed_point_per_m3)/ni.trailing_20pct_mean_density_per_m3,2)}.</p>${B?`<h3 style="margin:.8rem 0 .3rem">v1.3 a-priori budget vs outcome</h3><table aria-label="Budget versus outcome"><thead><tr><th>quantity</th><th>a priori</th><th>outcome (window)</th></tr></thead><tbody><tr><td>n_max (design ceiling, 2 λ_D per cell at 8 eV)</td><td>${sci(B.n_max_per_m3,2)} m⁻³</td><td>peak ${sci(c.window_maps_summary.n_e_peak_per_m3,3)} (${fmt(c.resolvability_at_peak.n_e_peak_over_n_max,3)} ×), mean ${sci(c.window_maps_summary.n_e_mean_per_m3,3)}</td></tr><tr><td>projected 0-D n_eq</td><td>${sci(B.n_eq_projected_per_m3,2)} (range ${sci(B.n_eq_projected_range_per_m3[0],2)}–${sci(B.n_eq_projected_range_per_m3[1],2)})</td><td>${sci(c.window_maps_summary.n_e_mean_per_m3,3)} m⁻³</td></tr><tr><td>neutral fixed point</td><td>${sci(B.neutral_fixed_point_per_m3,2)} (range ${sci(B.neutral_fixed_point_range_per_m3[0],2)}–${sci(B.neutral_fixed_point_range_per_m3[1],2)})</td><td>${sci(ni.trailing_20pct_mean_density_per_m3,4)} m⁻³</td></tr><tr><td>particles at n_eq</td><td>${B.particles_at_projected_n_eq}</td><td>${c.final_counts.electrons} + ${c.final_counts.ions}</td></tr><tr><td>ω_pe Δt at n_max</td><td>${fmt(B.omega_pe_dt,3)}</td><td>max observed ${fmt(c.resolvability_at_peak.max_observed_omega_pe_dt,3)}</td></tr><tr><td>cell / λ_D,min</td><td>${fmt(B.cell_over_lambda_d_min,3)}</td><td>${fmt(c.resolvability_at_peak.dz_over_lambda_d_at_peak,3)} (Δz at the peak node)</td></tr><tr><td>ion transit time</td><td>${sci(B.ion_transit_time_s,2)} s (${B.ion_transit_note})</td><td>${fmt(c.ion_transit_times,3)} transits elapsed</td></tr></tbody></table>`:""}<p class="small">Stopping rule: ${DATA.protocol.stopping_rule.plateau}</p>`;
const H=DATA.history;let hh=`<p class="small">${H.lesson}</p>`;if(H.steady_state.length){hh+=`<h3 style="margin:.6rem 0 .3rem">Steady-state predecessors (hash-verified summaries)</h3><table aria-label="Steady-state predecessors"><thead><tr><th>run</th><th>model</th><th>n_g0 → n_g,final</th><th>seed</th><th>W</th><th>t (µs) · τ_i</th><th>final e⁻ / Xe⁺</th><th>final I_d · I_beam (mA)</th><th>final S (s⁻¹)</th><th>stop</th></tr></thead><tbody>${H.steady_state.map(h=>`<tr><td>${h.label}<br><code>${h.id}</code></td><td>${h.model_version||"–"}</td><td>${sci(h.neutral_density_initial_per_m3,2)} → ${h.neutral_density_final_per_m3==null?"static":sci(h.neutral_density_final_per_m3,3)}</td><td>${sci(h.seed_plasma_density_per_m3,1)}</td><td>${sci(h.macro_weight,2)}</td><td>${fmt(h.simulated_time_s*1e6,3)} · ${fmt(h.ion_transit_times,3)}</td><td>${h.final_electrons} / ${h.final_ions}</td><td>${fmt(h.final_discharge_a*1e3,3)} · ${fmt(h.final_exit_ion_beam_a*1e3,3)}</td><td>${sci(h.final_ionization_rate_per_s,2)}</td><td>${h.stop_reason.replaceAll("_"," ")}</td></tr>`).join("")}</tbody></table>${H.steady_state.map(h=>`<p class="small"><strong>${h.label}:</strong> ${h.note} (summary <code>${h.summary_sha256.slice(0,12)}</code>; protocol at run <code>${(h.protocol_sha256_at_run||"").slice(0,12)}</code>${h.protocol_matches_current_file?"":" — predates the current protocol file, differences documented in its attempts block"})</p>`).join("")}`}
if(H.snapshot_v2){hh+=`<h3 style="margin:.8rem 0 .3rem">Snapshot v2 (model v1.1, static neutrals): growth without saturation</h3><p class="small">${H.snapshot_v2.lesson} Hash-verified from <code>${H.snapshot_v2.experiment_id}</code> (manifest <code>${H.snapshot_v2.manifest_sha256}</code>).</p><table aria-label="Snapshot v2 cases"><thead><tr><th>case</th><th>grid</th><th>W</th><th>steps</th><th>t (µs) · τ_i</th><th>window peak n_e</th><th>window I_d (mA)</th><th>N_e drift</th><th>stop</th></tr></thead><tbody>${H.snapshot_v2.cases.map(h=>`<tr><td>${h.id}</td><td>${h.grid}</td><td>${sci(h.macro_weight,2)}</td><td>${h.steps_completed}</td><td>${fmt(h.simulated_time_s*1e6,3)} · ${fmt(h.ion_transit_times,3)}</td><td>${sci(h.n_e_peak_per_m3,3)}</td><td>${fmt(h.window_discharge_a*1e3,3)}</td><td>${pct(h.electron_count_drift,3)}</td><td>${h.stop_reason.replaceAll("_"," ")}</td></tr>`).join("")}</tbody></table>`}
const S1=H.snapshot_v1;if(S1){hh+=`<h3 style="margin:.8rem 0 .3rem">Snapshot v1 (fail-closed)</h3><p class="small">${S1.lesson}</p><table aria-label="Snapshot v1 fail-closed cases"><thead><tr><th>case</th><th>grid</th><th>weight</th><th>steps</th><th>t (ns)</th><th>window peak n_e</th><th>final I_d (mA)</th><th>stop</th></tr></thead><tbody>${S1.cases.map(h=>`<tr><td>${h.id}</td><td>${h.grid}</td><td>${sci(h.macro_weight,2)}</td><td>${h.steps_completed}</td><td>${fmt(h.simulated_time_s*1e9,3)}</td><td>${sci(h.n_e_peak_per_m3,3)}</td><td>${fmt(h.final_discharge_a*1e3,3)}</td><td>${h.stop_reason.replaceAll("_"," ")}</td></tr>`).join("")}</tbody></table>`}
$("history").innerHTML=hh;
$("identity").innerHTML=`<p><span class="badge">status</span> ${DATA.status.replaceAll("_"," ")}</p><p><span class="badge">model</span> ${DATA.model_version} (${DATA.protocol.model_spec||"–"})</p><p><span class="badge">protocol SHA-256</span> <code>${DATA.protocol.file_sha256}</code> (frozen; every embedded case recorded this hash)</p><p><span class="badge">variants SHA-256</span> <code>${DATA.protocol.variants_file_sha256||"–"}</code></p><p><span class="badge">case summary SHA-256</span> <code>${c.summary_sha256}</code></p><p><span class="badge">maps npz SHA-256</span> <code>${c.maps_npz_sha256}</code></p><p><span class="badge">series npz SHA-256</span> <code>${c.series_npz_sha256}</code></p><p><span class="badge">git HEAD at run</span> <code>${c.git_head||"–"}</code></p><p><span class="badge">P2 field map SHA-256</span> <code>${c.field.field_map_sha256}</code> (design ${c.field.provenance.design_id}, checkpoint <code>${c.field.provenance.checkpoint_file_sha256}</code>)</p><p><span class="badge">cross sections</span> ${c.cross_sections?c.cross_sections.provenance_status+" · payload <code>"+c.cross_sections.payload_sha256+"</code>":"–"}</p><p><span class="badge">cusp planes</span> ${c.cusps.source}</p>`}
function bounds(w,h){return {l:58,t:18,r:w-78,b:h-46}}
function mapPoint(z,r,c,b){const zs=c.grid_z_m,rs=c.grid_r_m;return [b.l+(z-zs[0])/(zs.at(-1)-zs[0])*(b.r-b.l),b.b-(r-rs[0])/(rs.at(-1)-rs[0])*(b.b-b.t)]}
function drawField(){const c=DATA.cases[selected],s=setup($("field")),ctx=s.c,b=bounds(s.w,s.h),view=viewMatrix(c,mapKey),range=viewRange(view,mapKey);
ctx.clearRect(0,0,s.w,s.h);ctx.fillStyle=themeColor("--panel");ctx.fillRect(0,0,s.w,s.h);paintView(ctx,b,view,range);
ctx.save();ctx.setLineDash([5,4]);ctx.strokeStyle="#ffffffcc";ctx.lineWidth=1;c.cusps.cusp_z_m.forEach(zc=>{const p=mapPoint(zc,c.grid_r_m[0],c,b);ctx.beginPath();ctx.moveTo(p[0],b.t);ctx.lineTo(p[0],b.b);ctx.stroke()});ctx.restore();
axes(ctx,b,s.w,s.h,"z (m)","r (m)",c.grid_z_m[0],c.grid_z_m.at(-1),c.grid_r_m[0],c.grid_r_m.at(-1));drawColorbar(ctx,s,b,range);mapCaption(c,view,mapKey);
if(cursor){const p=mapPoint(cursor.z,cursor.r,c,b);ctx.strokeStyle="#fff";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p[0]-8,p[1]);ctx.lineTo(p[0]+8,p[1]);ctx.moveTo(p[0],p[1]-8);ctx.lineTo(p[0],p[1]+8);ctx.stroke()}}
function tick(v,lo,hi){const m=Math.max(Math.abs(lo),Math.abs(hi));return m>=1e5||(m>0&&m<1e-2)?(v===0?"0":Number(v).toExponential(2)):fmt(v,3)}
function axes(c,b,w,h,xlabel,ylabel,xmin,xmax,ymin,ymax,ylog=false){c.strokeStyle=themeColor("--line");c.fillStyle=themeColor("--muted");c.lineWidth=1;c.font="12px system-ui";c.strokeRect(b.l,b.t,b.r-b.l,b.b-b.t);c.textAlign="center";for(let i=0;i<=4;i++){const x=b.l+(b.r-b.l)*i/4;c.fillText(tick(xmin+(xmax-xmin)*i/4,xmin,xmax),x,b.b+18)}c.fillText(xlabel,(b.l+b.r)/2,h-6);c.save();c.translate(13,(b.t+b.b)/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="right";for(let i=0;i<=4;i++){const v=ymax-(ymax-ymin)*i/4;c.fillText(ylog?"1e"+fmt(v,3):tick(v,ymin,ymax),b.l-6,b.t+(b.b-b.t)*i/4+4)}c.textAlign="left"}
function quantile(values,q){const s=[...values].sort((a,b)=>a-b);if(!s.length)return NaN;const k=(s.length-1)*q,i=Math.floor(k);return s[i]+(s[Math.min(i+1,s.length-1)]-s[i])*(k-i)}
function updateCursor(clientX,clientY){const canvas=$("field"),rect=canvas.getBoundingClientRect(),b=bounds(rect.width,rect.height),c=DATA.cases[selected],x=Math.max(b.l,Math.min(b.r,clientX-rect.left)),y=Math.max(b.t,Math.min(b.b,clientY-rect.top));cursor=viewCursor(c,viewMatrix(c,mapKey),b,x,y);showTip(clientX-rect.left,clientY-rect.top);schedule(false)}
function showTip(x,y){const c=DATA.cases[selected],t=$("tip");if(!cursor){t.style.display="none";return}t.textContent=cellText(c,viewMatrix(c,mapKey),mapKey);t.style.display="block";t.style.left=Math.min(x+12,t.parentElement.clientWidth-t.offsetWidth-5)+"px";t.style.top=Math.max(4,y-36)+"px"}
function drawPlot(id,series,xLabel,yLabel,log=false,marks={}){const s=setup($(id)),c=s.c,b={l:64,t:16,r:s.w-16,b:s.h-40},pts=series.filter(q=>q.x.length);if(!pts.length){c.clearRect(0,0,s.w,s.h);return}const all=pts.flatMap(q=>q.y.filter(v=>v!=null&&isFinite(v)&&(!log||v>0))),xmin=Math.min(...pts.flatMap(q=>q.x)),xmax=Math.max(...pts.flatMap(q=>q.x));let ymin=Math.min(...all),ymax=Math.max(...all);if(marks.robust){ymin=quantile(all,.002);ymax=quantile(all,.998)}if(log){ymin=Math.log10(Math.max(ymin,1e-300));ymax=Math.log10(Math.max(ymax,1e-299))}else{const pad=(ymax-ymin||1)*.08;ymin-=pad;ymax+=pad}c.clearRect(0,0,s.w,s.h);c.fillStyle=themeColor("--panel");c.fillRect(0,0,s.w,s.h);const X=x=>b.l+(x-xmin)/(xmax-xmin||1)*(b.r-b.l);
if(marks.band){const x0=Math.max(b.l,X(marks.band[0])),x1=Math.min(b.r,X(marks.band[1]));c.fillStyle=themeColor("--window");c.fillRect(x0,b.t,x1-x0,b.b-b.t)}
if(marks.vlines){c.save();c.setLineDash([4,4]);c.strokeStyle=themeColor("--muted");c.lineWidth=1;marks.vlines.forEach(v=>{if(v==null||v<xmin||v>xmax)return;c.beginPath();c.moveTo(X(v),b.t);c.lineTo(X(v),b.b);c.stroke()});c.restore()}
if(marks.hlines){c.save();c.setLineDash([2,4]);c.lineWidth=1;marks.hlines.forEach(h=>{const yy=log?Math.log10(h.y):h.y;if(!(yy>=ymin&&yy<=ymax))return;c.strokeStyle=h.color||themeColor("--muted");const py=b.b-(yy-ymin)/(ymax-ymin||1)*(b.b-b.t);c.beginPath();c.moveTo(b.l,py);c.lineTo(b.r,py);c.stroke();c.fillStyle=h.color||themeColor("--muted");c.fillText(h.name,b.r-8-c.measureText(h.name).width,py-3)});c.restore()}
axes(c,b,s.w,s.h,xLabel,yLabel,xmin,xmax,ymin,ymax,log);c.save();c.beginPath();c.rect(b.l,b.t,b.r-b.l,b.b-b.t);c.clip();pts.forEach(q=>{c.strokeStyle=q.color;c.lineWidth=q.width||1.6;c.beginPath();let started=false;q.x.forEach((x,i)=>{const v=q.y[i];if(v==null||!isFinite(v)||(log&&v<=0)){started=false;return}const yy=log?Math.log10(v):v,px=X(x),py=b.b-(yy-ymin)/(ymax-ymin||1)*(b.b-b.t);started?c.lineTo(px,py):c.moveTo(px,py);started=true});c.stroke()});c.restore();pts.forEach((q,k)=>{c.fillStyle=q.color;c.fillText(q.name,b.l+8,b.t+14+k*15)});if(marks.robust){c.fillStyle=themeColor("--muted");c.font="10px system-ui";const note="y-range: 0.2–99.8 % quantiles (seed transient clipped)";c.fillText(note,b.r-6-c.measureText(note).width,b.b-4);c.font="12px system-ui"}}
function drawSeries(){const c=DATA.cases[selected],S=c.series,t=S.time_s.map(v=>v*1e6),cur=k=>S["current_"+k]||[],W=windowOf(c),tm={band:[W.start,W.end],vlines:[W.transit3]},cz=c.cusps.cusp_z_m.map(v=>v*1e3),ops=DATA.operating_point_summary;
drawPlot("counts",[{x:t,y:S.electrons,name:"electrons",color:"#5ad6c0"},{x:t,y:S.ions,name:"Xe⁺",color:"#ff6b6b"}],"t (µs)","macro-particles",false,tm);
drawPlot("currents",[{x:t,y:cur("discharge_a").map(v=>v*1e3),name:"discharge (anode)",color:"#5ad6c0"},{x:t,y:cur("exit_ion_beam_a").map(v=>v*1e3),name:"exit ion beam",color:"#ff6b6b"},{x:t,y:cur("wall_electron_a").map(v=>v*1e3),name:"wall e⁻",color:"#58a8ff"},{x:t,y:cur("wall_ion_a").map(v=>v*1e3),name:"wall Xe⁺",color:"#ffcf67"},{x:t,y:cur("exit_electron_a").map(v=>v*1e3),name:"exit e⁻ (returned)",color:"#c58bff"},{x:t,y:cur("injected_electron_a").map(v=>v*1e3),name:"injected e⁻",color:"#9bb8b0"}],"t (µs)","current (mA)",false,{...tm,robust:true});
drawPlot("neutral",[{x:t,y:S.neutral_fixed_point_per_m3,name:"analytic fixed point (Q_in − S)/c from the interval S (noisy)",color:"#ff6b6b",width:1},{x:t,y:S.neutral_density_per_m3,name:"n_g(t) (τ_g-relaxed, lags 30 ns)",color:"#5ad6c0",width:1.4}],"t (µs)","n_g (m⁻³)",false,{...tm,hlines:[{y:ops.zero_ionization_density_per_m3,name:"n_g0 = Q_in/c",color:"#ffcf67"}]});
drawPlot("rates",[{x:t,y:S.neutral_ionization_rate_per_s,name:"S ionisation",color:"#ff6b6b"},{x:t,y:S.neutral_effusion_rate_per_s,name:"c·n_g effusion",color:"#58a8ff"},{x:t,y:(S.neutral_artificial_rate_per_s||[]).map(Math.abs),name:"|artificial relaxation|",color:"#c58bff"}],"t (µs)","atoms/s",true,{...tm,hlines:[{y:ops.feed_atoms_per_s,name:"Q_in feed",color:"#ffcf67"}]});
drawPlot("phi",[{x:t,y:S.phi_max_v,name:"max φ",color:"#ff6b6b"},{x:t,y:S.phi_mean_v,name:"mean φ (plasma nodes)",color:"#5ad6c0"},{x:t,y:S.phi_min_v,name:"min φ",color:"#58a8ff"}],"t (µs)","φ (V)",false,tm);
drawPlot("energy",[{x:t,y:S.total_energy_j,name:"K+U total",color:"#eef7f4"},{x:t,y:S.kinetic_electron_j,name:"K electrons",color:"#5ad6c0"},{x:t,y:S.kinetic_ion_j,name:"K ions",color:"#ff6b6b"},{x:t,y:S.field_energy_j,name:"U field",color:"#58a8ff"},{x:t,y:(S.interval_electrode_work_j||[]).map(Math.abs),name:"|electrode work| per interval",color:"#c58bff"},{x:t,y:S.interval_residual_j.map(Math.abs),name:"|interval residual|",color:"#ffcf67"}],"t (µs)","energy (J)",true,tm);
const wz=c.wall_z_m.map(v=>v*1e3);drawPlot("wall",[{x:wz,y:c.wall.wall_electron_flux_per_m2_s,name:"electron flux (m⁻² s⁻¹)",color:"#5ad6c0"},{x:wz,y:c.wall.wall_ion_flux_per_m2_s,name:"ion flux (m⁻² s⁻¹)",color:"#ff6b6b"}],"z (mm)","flux",true,{vlines:cz});
drawPlot("exit",[{x:c.exit_r_m.map(v=>v*1e3),y:c.exit.exit_ion_current_density_a_per_m2,name:"ion j_z (A m⁻²)",color:"#ff6b6b"},{x:c.exit_r_m.map(v=>v*1e3),y:c.exit.exit_electron_current_density_a_per_m2,name:"electron j_z (A m⁻²)",color:"#5ad6c0"}],"r (mm)","A/m²");
drawPlot("axial",[{x:c.grid_z_m.map(v=>v*1e3),y:c.axial_peak_n_e_per_m3,name:"max_r n_e(z) (m⁻³)",color:"#5ad6c0",width:2}],"z (mm)","n_e (m⁻³)",false,{vlines:cz,hlines:[{y:DATA.budget?DATA.budget.n_max_per_m3:null,name:"n_max budget",color:"#ffcf67"}]});
drawPlot("wpe",[{x:t,y:S.peak_omega_pe_dt,name:"peak ω_pe Δt",color:"#ffcf67"}],"t (µs)","ω_pe Δt",false,{...tm,hlines:[{y:.2,name:"gate 0.2",color:"#ff6b6b"},{y:c.stability_gate.omega_pe_dt,name:"design n_max",color:"#58a8ff"}]})}
function drawAll(){renderMetrics();renderVerification();renderDetails();drawField();drawSeries()}
function schedule(full=true){cancelAnimationFrame(raf);raf=requestAnimationFrame(full?drawAll:drawField)}
function select(i){selected=i;caseSelect.value=i;cursor=null;showTip();schedule()}
caseSelect.onchange=()=>select(Number(caseSelect.value));$("map").onchange=e=>{mapKey=e.target.value;schedule()};$("scale").onchange=e=>{scaleMode=e.target.value;schedule(false)};wireMapControls();
$("theme").onclick=()=>{const light=document.documentElement.dataset.theme!=="light";document.documentElement.dataset.theme=light?"light":"dark";$("theme").textContent=light?"Dark theme":"Light theme";$("theme").setAttribute("aria-pressed",light);schedule()};
$("field").addEventListener("pointermove",e=>updateCursor(e.clientX,e.clientY));$("field").addEventListener("pointerleave",()=>{cursor=null;showTip();schedule(false)});
$("field").addEventListener("keydown",e=>{const c=DATA.cases[selected],view=viewMatrix(c,mapKey),zs=view.z,rs=view.r;if(e.key==="Home"){cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]}}else{if(!cursor)cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]};if(e.key==="ArrowLeft")cursor.zi=Math.max(0,cursor.zi-1);else if(e.key==="ArrowRight")cursor.zi=Math.min(zs.length-1,cursor.zi+1);else if(e.key==="ArrowDown")cursor.ri=Math.max(0,cursor.ri-1);else if(e.key==="ArrowUp")cursor.ri=Math.min(rs.length-1,cursor.ri+1);else return;cursor.z=zs[cursor.zi];cursor.r=rs[cursor.ri]}e.preventDefault();showTip(70,30);schedule(false)});
window.addEventListener("keydown",e=>{if(["INPUT","SELECT","BUTTON"].includes(e.target.tagName))return;const k=Number(e.key);if(k>=1&&k<=DATA.cases.length)select(k-1)});new ResizeObserver(schedule).observe(document.querySelector("main"));window.addEventListener("pageshow",schedule);drawAll();
</script></body></html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    return snapshot_dashboard.fill_template(HTML_TEMPLATE, payload)


def generate(output_path: Path = DEFAULT_OUTPUT, results: Path = RESULTS, protocol_path: Path = PROTOCOL) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(build_payload(results, protocol_path)), encoding="utf-8", newline="\n")
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
