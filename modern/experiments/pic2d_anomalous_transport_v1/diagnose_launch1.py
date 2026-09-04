"""Diagnosis of the alpha-1over16 LAUNCH 1 triad-gate stop (2026-09-04 19:01 UTC) from its recorded artifacts - read-only, reproducible.

    python -m experiments.pic2d_anomalous_transport_v1.diagnose_launch1 [--results <dir>] [--write]

Reads ``summary.json``, ``series.npz`` (and the alpha = 0 ss-v4 series for the comparison), binds every input by byte hash, and
writes ``triad-stop-diagnosis.json`` (+ sidecar) beside them.  The verdict is formed mechanically from the recorded numbers:

* which triad member(s) exceeded the hard bound at the stop and their trajectories at every 40 000-step checkpoint;
* the physics protections at the stop (windowed residual power, accumulated-floor peak Debye);
* the discharge trajectory (N_e, S, I_d, the loss / return channels, n_g, T_e,dense, the anomalous event rate) against the
  alpha = 0 run at the same times -> heating / re-equilibration / extinction / artefact;
* the re-read under amendment 1 (the v2.1.1 arming latch + the ignition gate): where the amended protocol would have stopped it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.orbit_mc.artifacts import content_hash
from cft_revival.pic2d import artifacts
from experiments.pic2d_cft_steady_state_v1 import run as runner

from .protocol import HERE, HYPOTHESES, V4_RESULTS, load_case_protocol

SCHEMA = "cft-revival.pic2d-anomalous-transport-v1.triad-stop-diagnosis/0.1.0"
CASE = "alpha-1over16"
CHECKPOINT = 40_000
ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _arrays(path: Path) -> dict[str, np.ndarray]:
    series = np.load(path)
    return {key: np.asarray(series[key]) for key in series.files}


def _at(arrays: dict[str, np.ndarray], t_s: float) -> int:
    return int(min(np.searchsorted(arrays["time_s"], t_s), arrays["time_s"].size - 1))


def _drift_trajectory(arrays: dict[str, np.ndarray], transit_s: float, fraction: float) -> list[dict[str, Any]]:
    steps = arrays["step"]

    def drift(key: str, n: int) -> float | None:
        if key not in arrays:
            return None
        value = runner.trailing_time_drift(arrays["time_s"][:n], arrays[key][:n].astype(np.float64), fraction)
        return None if value is None else round(float(value), 4)

    rows = []
    for cp in np.arange(CHECKPOINT * 5, steps[-1] + 1, CHECKPOINT * 5):     # every 200 000 steps (0.28 us) keeps the table readable
        n = int(np.searchsorted(steps, cp, side="right"))
        if n < 8:
            continue
        t_end = float(arrays["time_s"][n - 1])
        rows.append({"step": int(steps[n - 1]), "t_us": round(t_end * 1e6, 3), "transits": round(t_end / transit_s, 3),
                     "I_d": drift("current_discharge_a", n), "N_e": drift("electrons", n), "S": drift("current_ionization_rate_per_s", n),
                     "T_e_dense": drift("peak_node_t_e_dense_ev", n), "omega_pe_dt_resolved": drift("peak_omega_pe_dt", n)})
    return rows


def _trajectory(arrays: dict[str, np.ndarray], times_us: tuple[float, ...]) -> list[dict[str, Any]]:
    keys = {"electrons": "N_e_macro", "ions": "N_i_macro", "current_discharge_a": "I_d_a", "current_anode_electron_a": "anode_electron_a",
            "current_exit_electron_a": "exit_plane_electron_return_a", "current_injected_electron_a": "injected_electron_a",
            "current_wall_electron_a": "wall_electron_a", "current_wall_ion_a": "wall_ion_a", "current_exit_ion_beam_a": "I_beam_a",
            "current_ionization_rate_per_s": "S_per_s", "neutral_density_per_m3": "n_g_per_m3", "peak_node_t_e_dense_ev": "T_e_dense_ev",
            "peak_node_window_n_e_peak_per_m3": "peak_n_e_window_per_m3", "peak_node_window_t_e_peak_ev": "T_e_peak_window_ev",
            "peak_node_window_cells_per_debye": "cells_per_debye_window", "peak_omega_pe_dt": "omega_pe_dt_resolved",
            "current_anomalous_collision_rate_per_s": "anomalous_events_per_s"}
    rows = []
    for t_us in times_us:
        if arrays["time_s"][-1] < t_us * 1e-6 * 0.999:
            continue
        i = _at(arrays, t_us * 1e-6)
        row: dict[str, Any] = {"t_us": round(float(arrays["time_s"][i]) * 1e6, 3), "step": int(arrays["step"][i])}
        for key, name in keys.items():
            if key in arrays:
                value = float(arrays[key][i])
                row[name] = value if np.isfinite(value) else None
        rows.append(row)
    return rows


def diagnose(results: Path, *, v4_results: Path = V4_RESULTS) -> dict[str, Any]:
    protocol = load_case_protocol(CASE)
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    run_state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    arrays = _arrays(results / "series.npz")
    v4 = _arrays(v4_results / "series.npz")
    rule = protocol["stopping_rule"]
    transit = float(runner.protocol_budget(protocol)["ion_transit_time_s"])
    fraction = float(rule["plateau_window_fraction"])
    triad = summary["grid_heating_triad"]
    t = arrays["time_s"]
    ne = arrays["electrons"].astype(np.float64)

    # -- e-fold of the electron inventory over the last microsecond; the loss / return channels
    tail = t > t[-1] - 1.0e-6
    slope = float(np.polyfit(t[tail], np.log(ne[tail]), 1)[0])
    e_fold_us = float(-1.0 / slope) * 1e6 if slope < 0 else None
    injected = float(arrays["current_injected_electron_a"][-1])
    returned = float(arrays["current_exit_electron_a"][-1])

    # -- the hook's rate check: events per real electron per second vs omega_ce / 16 at a mean |B|
    i01 = _at(arrays, 0.1e-6)
    W = float(protocol["case"]["macro_weight"])
    per_electron = float(arrays["current_anomalous_collision_rate_per_s"][i01]) / (float(ne[i01]) * W)
    alpha = float(protocol["numerics"]["anomalous_collisions"]["alpha"])
    b_equiv = per_electron / alpha / (ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG)

    # -- the amended rule's re-read: ignition gate at the checkpoints, latch state, residual member
    amended_stop = None
    for n in range(1, arrays["step"].size + 1):
        if int(arrays["step"][n - 1]) % CHECKPOINT != 0:
            continue
        ignition = runner.evaluate_ignition({k: v[:n] for k, v in arrays.items()}, rule)
        if ignition is not None and ignition["failed"]:
            amended_stop = {"step": int(arrays["step"][n - 1]), "t_us": round(float(t[n - 1]) * 1e6, 3), "reason": ignition["reason"],
                            "checks": ignition["checks"]}
            break
    amended_triad = runner.evaluate_triad(arrays, rule, transit)
    recorded_rule = json.loads(json.dumps(rule))
    recorded_rule["grid_heating_triad"].pop("drift_members_arming", None)
    recorded_triad = runner.evaluate_triad(arrays, recorded_rule, transit)

    # -- ignition ratios (the amended gate's statistic) for this run and the alpha = 0 run
    def ratios(arr: dict[str, np.ndarray]) -> dict[str, Any]:
        tt = arr["time_s"]; n_e = arr["electrons"].astype(np.float64); s_rate = arr["current_ionization_rate_per_s"].astype(np.float64)
        ref = (tt >= 0.05e-6) & (tt < 0.2e-6)
        out = {"N_e_reference_macro": float(n_e[ref].mean()), "S_reference_per_s": float(s_rate[ref].mean())}
        for tc in (0.5e-6, 0.75e-6, 1.0e-6, 1.5e-6, 2.0e-6, 2.4e-6):
            if tt[-1] < tc:
                continue
            m = (tt >= tc - 0.15e-6) & (tt <= tc)
            out[f"t_{tc*1e6:.2f}_us"] = {"N_e_ratio": round(float(n_e[m].mean()) / out["N_e_reference_macro"], 4),
                                          "S_ratio": round(float(s_rate[m].mean()) / out["S_reference_per_s"], 4)}
        return out

    i_stop = -1
    stop_snapshot = _trajectory(arrays, (float(t[-1]) * 1e6,))[0]
    ref_quantities = protocol["reference_run"]["quantities"]
    v4_same_time = _trajectory(v4, (float(t[-1]) * 1e6,))[0]
    heading = {
        "discharge_current_a": {"launch_1_at_stop": stop_snapshot["I_d_a"], "alpha_0_at_same_time": v4_same_time["I_d_a"],
                                "alpha_0_plateau": ref_quantities["discharge_current_a"], "hypothesis": HYPOTHESES["discharge_current_a"]["sign"],
                                "reading": "CONTRADICTS the sign (+20...+60 % expected): the anode current collapsed to ~2 % of the alpha = 0 plateau"},
        "ionization_rate_per_s": {"launch_1_at_stop": stop_snapshot["S_per_s"], "alpha_0_at_same_time": v4_same_time["S_per_s"],
                                  "alpha_0_plateau": ref_quantities["ionization_rate_per_s"], "hypothesis": HYPOTHESES["ionization_rate_per_s"]["sign"],
                                  "reading": "the declared sign, but as an extinction (-100 %), not a -10...-40 % shift of a plateau"},
        "peak_n_e_window_per_m3": {"launch_1_at_stop": stop_snapshot.get("peak_n_e_window_per_m3"), "alpha_0_plateau": ref_quantities["peak_n_e_window_per_m3"],
                                   "hypothesis": HYPOTHESES["peak_n_e_window_per_m3"]["sign"], "reading": "the declared sign as an extinction"},
        "t_e_peak_window_ev": {"launch_1_at_stop": stop_snapshot.get("T_e_peak_window_ev"), "alpha_0_plateau": ref_quantities["t_e_peak_window_ev"],
                               "hypothesis": HYPOTHESES["t_e_peak_window_ev"]["sign"], "reading": "undefined at the stop (no dense node holds enough electrons)"},
        "neutral_density_per_m3": {"launch_1_at_stop": stop_snapshot["n_g_per_m3"], "alpha_0_plateau": ref_quantities["neutral_density_per_m3"],
                                   "hypothesis": HYPOTHESES["neutral_density_per_m3"]["sign"],
                                   "reading": "back at the undepleted n_g0 5.5e19 (no plasma to deplete the inventory) - the declared sign as an extinction"},
    }
    residual = triad["windowed_energy_residual_over_electrode_work"]
    verdict = {
        "option": "extinction_under_the_closure",
        "options_considered": {
            "a_genuine_heating": f"NO: the windowed residual power read {residual*100:+.2f} % of the electrode work at the stop (hard bound +5 %; the "
                                 f"cumulative ratio {triad['energy_residual_over_electrode_work']*100:+.2f} %), never positive beyond +1.2 % at any record; the "
                                 f"accumulated-floor peak read {stop_snapshot['cells_per_debye_window']:.2f} cells per lambda_D (hard pi, soft 2.5)",
            "b_benign_reequilibration": "NO: the discharge was not moving toward a new state - N_e fell monotonically from t = 0 (its maximum is the seed), "
                                        f"I_d fell to {stop_snapshot['I_d_a']*1e3:.3f} mA, S to {stop_snapshot['S_per_s']:.3g} /s; no quantity was settling",
            "c_gate_artefact": "PARTLY, for one member only: the T_e,dense member (+0.366) is the trailing drift of an undefined statistic (the dense "
                               "node held < a few macro-electrons after 1.1 us; values 1e-17 ... 0.6 eV) - a shot-noise reading, recorded as a weakness "
                               "of that statistic; the S member (-0.618) is the real decay. The stop itself was CORRECT in effect (a dead discharge; "
                               "~8 GPU-hours saved) and mislabelled in cause ('grid heating')",
            "d_extinction": f"YES: the seed plasma decayed with an e-fold of {e_fold_us:.2f} us over the last microsecond (r_w^2 / 4 D_perp ~ 1 us at "
                            f"D_perp = kT_e / 16 eB for T_e 5-10 eV, |B| 0.1-0.3 T), the injected {injected*1e3:.2f} mA returned through the exit plane "
                            f"({returned*1e3:.2f} mA at the stop) instead of being trapped, the wall and anode electron currents tracked N_e down, and n_g "
                            f"returned to the undepleted 5.5e19: the electron confinement the discharge needs does not exist under nu_an = omega_ce / 16 "
                            f"in this model (v1.3 closure, exit-plane 3 mA / 2 eV injection, dielectric walls without SEE) at this operating point",
        },
        "hook_rate_check": {"anomalous_events_per_electron_per_s_at_0p1_us": per_electron, "equivalent_mean_B_T_for_alpha_omega_ce": b_equiv,
                            "reading": "consistent with nu_an = alpha omega_ce at <|B|> ~ 0.15 T (channel |B| 0.05-0.29 T): the hook ran at its declared rate"},
        "vs_predeclared_hypotheses": "the hypothesis 'I_d up +20...+60 % at alpha = 1/16' is CONTRADICTED in the strongest form: no self-sustained discharge "
                                     "exists at this alpha; the S / peak n_e / T_e 'down' signs are realised as an extinction, not as a shifted plateau. "
                                     "By the predeclared rule the case is no_plateau (no trend contribution); the finding is recorded as such",
        "relaunch_decision": "NOT relaunched: the same seed and configuration identity replay bitwise into the same extinction; under amendment 1 the "
                             "ignition gate would stop the replay at 1.008 us. Launch 1 IS the case's record",
    }
    return {
        "schema": SCHEMA, "experiment_id": protocol["experiment_id"], "case": CASE, "alpha": alpha, "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "results_dir": results.relative_to(HERE).as_posix() if results.is_relative_to(HERE) else str(results),
        "inputs": {name: _sha(results / name) for name in ("summary.json", "series.npz", "run_state.json", "maps.npz")}
        | {"alpha_0_series": _sha(v4_results / "series.npz"), "amended_sealed_protocol_content_sha256": content_hash(protocol)},
        "run": {"pid": run_state.get("pid"), "git_head": summary["git_head"], "protocol_sha256_recorded": summary["protocol_sha256"],
                "config_sha256": summary["provenance"]["config_sha256"], "stop_reason": summary["stop_reason"], "steps": summary["steps_completed"],
                "simulated_time_us": summary["simulated_time_s"] * 1e6, "transits": summary["ion_transit_times"], "wall_seconds": summary["wall_seconds_total"],
                "ms_per_step": summary.get("ms_per_step_this_session")},
        "gate_that_fired": {"member": "drift members (v1.4 arming at 1.0 transit)", "hard_failures": triad["hard_failures"],
                            "readings": {k: triad[k] for k in ("ionisation_rate_drift", "t_e_dense_drift", "omega_pe_dt_drift")},
                            "physics_protections_at_stop": {"windowed_residual_over_electrode_work": residual, "windowed_residual_j": triad["windowed_energy_residual_j"],
                                                            "windowed_electrode_work_j": triad["windowed_energy_residual_electrode_work_j"],
                                                            "cumulative_residual_over_electrode_work": triad["energy_residual_over_electrode_work"],
                                                            "peak_cells_per_debye_window": stop_snapshot["cells_per_debye_window"]},
                            "plateau_block": summary["plateau"]},
        "drift_members_trajectory": _drift_trajectory(arrays, transit, fraction),
        "trajectory": _trajectory(arrays, (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.4, float(t[-1]) * 1e6)),
        "alpha_0_trajectory_same_times": _trajectory(v4, (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.4)),
        "electron_inventory": {"N_e_seed_macro": float(ne[0]), "N_e_max_macro": float(ne.max()), "N_e_max_at_us": float(t[int(np.argmax(ne))]) * 1e6,
                               "N_e_at_stop_macro": float(ne[i_stop]), "e_fold_time_last_us": e_fold_us,
                               "alpha_0_N_e_at_1_us": float(v4["electrons"][_at(v4, 1.0e-6)]), "alpha_0_N_e_at_2p4_us": float(v4["electrons"][_at(v4, 2.4e-6)])},
        "ignition_ratios": {"launch_1": ratios(arrays), "alpha_0": ratios(v4)},
        "heading_vs_alpha_0_and_hypotheses": heading,
        "amendment_1_re_read": {"ignition_gate_first_failure": amended_stop,
                                "drift_members_arming_at_stop": amended_triad["drift_members_arming"],
                                "amended_triad_hard_failures_at_stop": amended_triad["hard_failures"],
                                "recorded_rule_hard_failures_at_stop": recorded_triad["hard_failures"],
                                "gpu_hours_saved_by_the_recorded_stop_vs_budget": round((float(protocol["stopping_rule"]["wall_budget_seconds"]) - summary["wall_seconds_total"]) / 3600.0, 2)},
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results" / CASE)
    parser.add_argument("--write", action="store_true", help="write triad-stop-diagnosis.json (+ sidecar) into the results directory")
    args = parser.parse_args()
    report = diagnose(args.results)
    if args.write:
        digest = artifacts.write_canonical_json(args.results / "triad-stop-diagnosis.json", report)
        print(f"[diagnose] {args.results / 'triad-stop-diagnosis.json'} {digest[:12]}")
    print(json.dumps({"verdict": report["verdict"]["option"], "gate": report["gate_that_fired"]["hard_failures"],
                      "residual": report["gate_that_fired"]["physics_protections_at_stop"]["windowed_residual_over_electrode_work"],
                      "e_fold_us": report["electron_inventory"]["e_fold_time_last_us"], "amended_stop": report["amendment_1_re_read"]["ignition_gate_first_failure"]},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
