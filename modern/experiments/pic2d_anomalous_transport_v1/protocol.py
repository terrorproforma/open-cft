"""Composition of the alpha-series run protocols on the steady-state v4 (33 um) template.

Each case is the ss-v4 protocol (``pic2d_cft_steady_state_v4/protocol.json``: reference design, 90 x 720 cells at 33.33 um,
dt 1.4 ps, W 26 666.7, the v1.3 closure, seed 20260903, frames ON) with exactly these changes:

* ``numerics.anomalous_collisions`` = the v2.1.0 perpendicular-rotation Bohm model at the case's alpha (``nu_an = alpha
  omega_ce``, ``D_perp = (kT_e/eB) alpha / (1 + alpha^2)``; ``cft_revival.pic2d.sensitivity``);
* the v2.0.6 gates: the peak-Debye gate's accumulated-particle-step floor (64 000 macro-electron-steps at the gated node) is
  declared; the corrected energy ledger is code (v2.0.6, no protocol key) so acceptance (b) reads the CORRECTED residual natively;
* ``numerics.performance.moment_sample_interval`` K = 5 (v2.0.5; physics bitwise, enters ``config_sha256`` by policy);
* the acceptance block: (a) the v4 plateau rule, (b) corrected windowed residual < +2 %, the alpha = 0 reference (ss-v4) with its
  (b) FAIL at +2.46 % stated, the predeclared hypotheses (signs from the physics audit section 4.c) and the series verdict rule;
* the wall budget from the launch-box measured rate x 1.5 (filled from the preflight before the preregistration commit);
* AMENDMENT 1 (2026-09-05, after launch 1 of alpha-1over16 extinguished and was stopped by the drift members at 1.00 transit): the model
  v2.1.1 ``drift_members_arming`` latch on the triad's drift members and the v2.0 ``ignition_gate`` calibrated on the accepted 33 um runs
  (``DRIFT_MEMBERS_ARMING`` / ``IGNITION_GATE`` / ``AMENDMENTS``; stopping-rule keys outside ``config_sha256``).

Everything else (geometry, operating point, dt, grid, W, seed, cadences, plateau rule, the v2.0.3 gate thresholds) is byte-for-byte
the v4 protocol; ``test_pic2d_anomalous_transport_v1.py`` pins that.
"""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d.sensitivity import ANOMALOUS_MODEL_ROTATION, BOHM_ALPHA_SERIES

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
V4_DIR = MODERN / "experiments" / "pic2d_cft_steady_state_v4"
V4_PROTOCOL_PATH = V4_DIR / "protocol.json"
V4_RESULTS = V4_DIR / "results"
PROTOCOLS_DIR = HERE / "protocols"
CAMPAIGN_PROTOCOL_PATH = HERE / "protocol.json"

SCHEMA_VERSION = "cft-revival.pic2d-anomalous-transport-v1.protocol/0.1.0"
CASE_SCHEMA_VERSION = "cft-revival.pic2d-anomalous-transport-v1.case-protocol/0.1.0"
EXPERIMENT_ID = "pic2d-anomalous-transport-v1"
MODEL_VERSION = ("pic2d v1.3 closure (quasi-steady 0-D neutral inventory, NO wall-ion recycling) + model v2.1.0 anomalous cross-field "
                 "transport (Bohm-type perpendicular-velocity rotation, nu_an = alpha omega_ce, Brandt et al. 2016 event model) with the "
                 "v2.0.6 gates (window-mode peak-Debye gate with the accumulated-particle-step floor, windowed residual-power gate on the "
                 "W-corrected ledger) and the v2.0.5 K = 5 moment sampling")
MODEL_SPEC = ("modern/spec/pic2d/pic2d-model-v1.3.json (physics) + modern/spec/pic2d/pic2d-model-v2.0.json (gates_v2_0: v2.0.3 gates, "
              "v2.0.6 ledger correction + Debye floor, v2.0.5 performance block, anomalous_transport_v2_1_0)")

# the P2 divergent-exit-stack cusp planes (cusp topology search v3.1, cec47f12; the ss-v4 dashboard's overlay) for the per-cusp report
CUSP_PLANES_M: tuple[float, ...] = (6.028e-3, 12.000e-3, 17.972e-3)
CUSP_HALF_WIDTH_M = 1.0e-3

# alpha-series cases (audit section 4.c / roadmap R1): a quarter of the classical Bohm value, the classical Bohm value, Brandt's coefficient
CASES: dict[str, dict[str, Any]] = {
    "alpha-1over64": {"alpha": BOHM_ALPHA_SERIES[0], "label": "1/64",
                      "role": "a quarter of the classical Bohm frequency: the weak-transport end of the bracket (Smirnov et al. 2004 inferred omega_c/16 for the cylindrical HT)"},
    "alpha-1over16": {"alpha": BOHM_ALPHA_SERIES[1], "label": "1/16",
                      "role": "the classical Bohm value nu_an = omega_ce / 16 (D_perp = kT_e / 16 eB x 1 / (1 + 1/256)); the audit's central expectation is stated at this point"},
    "alpha-0.345": {"alpha": BOHM_ALPHA_SERIES[2], "label": "0.345",
                    "role": "Brandt et al. 2016's D_perp = 0.4 kT_e / eB: nu = 0.4 omega_ce gives the exact Green-Kubo factor 0.4 / 1.16 = 0.345; declared here as "
                            "alpha = 0.345 so that the SMALL-alpha reading of the hook (D ~ alpha kT/eB) names the reference's coefficient - the exact factor of this "
                            "case is 0.345 / 1.119 = 0.308 (both readings are recorded; the strongest-transport end of the series)"},
}
LAUNCH_PRIORITY: tuple[str, ...] = ("alpha-1over16", "alpha-1over64", "alpha-0.345")
REFERENCE_CASE = "alpha-0"
REFERENCE_CORRECTED_RESIDUAL = 0.02458578453535502      # ss-v4 ledger-corrected.json end_state_window.corrected_ratio (+2.46 %)
STEPS_TO_3_TRANSITS = 5_142_858

# -- amendment 1 (2026-09-05, after launch 1 of alpha-1over16 stopped at exactly 1.00 transit on the drift members) ------------------
# model v2.1.1 arming of the triad's drift members relative to the run's own discharge (spec triad_drift_arming_v2_1_1) + the v2.0
# ignition gate calibrated on the accepted 33 um runs; both are stopping-rule keys OUTSIDE config_sha256 (identities unchanged)
DRIFT_MEMBERS_ARMING: dict[str, Any] = {
    "min_transit_times": 2.0,
    "settle_quantity": "discharge_current",
    "settle_drift_max": 0.05,
    "settle_check_cadence_steps": 40000,
    "note": ("model v2.1.1 (amendment 1): the triad's DRIFT members (trailing-20 % drifts of S, T_e,dense, resolved omega_pe dt; hard 0.25) are enforced "
             "only once >= 2.0 ion transits have elapsed AND the trailing-20 % I_d drift has read < 0.05 (the plateau threshold) at a 40 000-step "
             "checkpoint at or after 2.0 transits - a 'settled once' latch, a pure function of the series. Under the v1.4 rule (enforced_after_transit_times "
             "1.0, calibrated on alpha = 0 plateaus: ss-v4 I_d drift +0.116 / S +0.10 / T_e,dense +0.02 at 1.0 transit) a discharge re-equilibrating to "
             "a different state under the closure could be stopped for moving. The physics protections are unchanged and independent of the arming: the "
             "one-sided windowed residual-POWER gate (>= 5 % of the electrode work, from the first complete 400 000-step window) and the window-mode "
             "peak-Debye hard gate (pi cells per lambda_D on the accumulated-floor peak). Calibration on the accepted runs: ss-v4's latch closes at 2.66 "
             "transits (checkpoint 4 560 000, drift +0.049); 047 / 009 / 056-L2 read |I_d drift| < 0.05 from 2.0 transits; no accepted run ever tripped a "
             "drift member; plume attempt 8 and the ext-val launch 1 are stopped by the residual-power member at their first complete window whatever the "
             "arming. Launch 1 of alpha-1over16 (record results/alpha-1over16, protocol b59b4402...) stopped at 1.0033 transits on S drift -0.618 and "
             "T_e,dense drift +0.366 with the residual at +1.15 % and the peak at 0.48 cells per lambda_D: an EXTINCTION (see ignition_gate), not heating"),
}
IGNITION_GATE: dict[str, Any] = {
    "reference_window_s": [0.05e-6, 0.2e-6],
    "check_window_s": 0.15e-6,
    "checks": [
        {"time_s": 1.0e-6, "min_s_ratio": 0.3, "min_electron_ratio": 0.6},
        {"time_s": 2.0e-6, "min_s_ratio": 0.4, "min_electron_ratio": 0.6},
    ],
    "note": ("amendment 1: fail-closed stop_reason no_ignition (the v2.0 runner gate), evaluated at every checkpoint from the series: trailing 0.15 us means "
             "of S and N_e over their 0.05-0.2 us reference means (after the seed dump). Required with the drift-member arming latch: a discharge that never "
             "settles is an extinction the latch can never arm on, and would otherwise run to its wall budget. Calibration (N_e ratio / S ratio at 1.0 and "
             "2.0 us): accepted 33 um plateaus ss-v4 1.40 / 1.03 and 2.01 / 1.58; sweep 047 1.31 / 0.96 and 1.68 / 1.08; 009 1.32 / 1.17 and 1.76 / 1.33; "
             "056-L2 1.44 / 0.99 and 1.67 / 1.11 (margins >= 2.2x on N_e, >= 2.4x on S over the bounds); the extinguished alpha-1over16 launch 1: 0.45 / 0.37 "
             "at 1.0 us (fails on N_e) and 0.14 / 0.02 at 2.0 us - it would have stopped at the first checkpoint past 1.0 us (720 000 steps, ~1 h) instead "
             "of 2.408 us. A case re-equilibrated at the hypotheses' extreme (S, N_e -40 % of alpha = 0) reads ~0.84 / 0.6 - inside. A non-ignition under "
             "the frozen seed and injection is a recorded outcome of the closure at that alpha (stopping_rule.ignition_check), never a reason to adjust"),
}
PRE_AMENDMENT_SEALED_SHA256: dict[str, str] = {
    "alpha-1over64": "33acb08a0767c4d74ea1685c35a5b2141930eafc4dc5664cf5bcec5166876c3f",
    "alpha-1over16": "b59b4402ac36e33891374a24f9147c22452acc6fc7bae5658182d44bfed83eaa",
    "alpha-0.345": "a9519acb5a97d0e43dfc639a6f6b0c7cb241d269a809adf30778c609525b2d74",
}
AMENDMENTS: list[dict[str, Any]] = [
    {
        "id": 1,
        "date_utc": "2026-09-04T20:00Z (2026-09-05 06:00 AEST)",
        "preregistration_commit": "057841cf",
        "trigger": ("LAUNCH 1 of alpha-1over16 (PID 46438, 17:08:39-19:01:09 UTC 2026-09-04, lock 057841cf, protocol b59b4402...) stopped "
                    "grid_heating_triad_gate_stopped_run at step 1 720 000 = 2.408 us = 1.0033 transits, the first checkpoint after the drift members armed: "
                    "ionisation_rate_drift -0.618 and t_e_dense_drift +0.366 exceeded the hard 0.25. Diagnosis (results/alpha-1over16/triad-stop-diagnosis.json): "
                    "NOT heating (windowed residual +1.15 % of the electrode work, cumulative -0.17 %, accumulated-floor peak 0.48 cells per lambda_D), NOT a "
                    "re-equilibration toward the hypothesised state (I_d 3.1 -> 0.06 mA) and NOT a gate artefact in the S member: the discharge EXTINGUISHED "
                    "under the closure - N_e fell monotonically from the seed (6.06e5 -> 4.62e5 at 0.1 us -> 1.89e5 at 1.0 us -> 3.74e4 at 2.41 us; e-fold "
                    "0.88 us over the last microsecond, matching r_w^2 / 4 D_perp ~ 1 us at D_perp = kT_e / 16 eB), S 2.1e16 -> 6.2e15 -> 0 /s, the injected "
                    "3.0 mA returned through the exit plane (0.32 -> 2.2 -> 2.9 mA) while the anode electron current fell 3.1 -> 0.93 -> 0.06 mA, n_g back at "
                    "the undepleted 5.49e19; the anomalous event rate 2.0e19 /s at 0.1 us = omega_ce / 16 per electron at <|B|> 0.15 T (the hook at its "
                    "declared rate). The T_e,dense member's +0.366 IS a shot-noise reading of an undefined statistic (the dense node held < a few "
                    "macro-electrons; values 1e-17 ... 0.6 eV) - recorded as a gate-statistic weakness; the S member's -0.618 is the real decay"),
        "changes": [
            ("stopping_rule.grid_heating_triad.drift_members_arming added (model v2.1.1 latch: min_transit_times 2.0, settle_quantity discharge_current, "
             "settle_drift_max 0.05, settle_check_cadence_steps 40000); enforced_after_transit_times 1.0 kept in the block as the recorded, superseded rule"),
            ("stopping_rule.ignition_gate added (v2.0 runner gate; reference 0.05-0.2 us, checks at 1.0 us N_e >= 0.6 / S >= 0.3 and 2.0 us N_e >= 0.6 / "
             "S >= 0.4 of the reference means; calibrated on ss-v4 / 047 / 009 / 056-L2 and the extinguished launch 1)"),
            "stopping_rule.fail_closed and grid_heating_triad.note text updated to name both; campaign.changes lists them",
            ("all three sealed protocols re-sealed identically (the same two stopping-rule keys); config_sha256 identities UNCHANGED "
             "(28ca0391 / 90cf53f1 / 8ea88273: stopping rules are outside the physics identity)"),
        ],
        "not_changed": ("geometry, operating point, grid, dt, W, seed, closure model and alphas, cadences, the plateau rule, the residual-power and peak-Debye "
                        "gates and their thresholds, acceptance (a)-(d), the hypotheses, the reference block, the budgets (37 200 s from the preflights)"),
        "pre_amendment_sealed_sha256": PRE_AMENDMENT_SEALED_SHA256,
        "launch_1_disposition": ("alpha-1over16 launch 1 is RECORDED as the case's result (results/alpha-1over16/, executed under the pre-amendment seal "
                                 "b59b4402...; stop_reason grid_heating_triad_gate_stopped_run; assess --case -> no_plateau): NOT relaunched - the same "
                                 "seed and configuration identity replay bitwise into the same extinction, and under this amendment the ignition gate "
                                 "would stop the replay at 1.0 us. The series' 1/16 point is 'extinguished under the closure' (a no_plateau outcome that "
                                 "CONTRADICTS the hypothesis I_d up: no discharge exists at alpha = 1/16 under the v1.3 closure at this operating point "
                                 "with this cathode model)"),
        "remaining_launches": ("alpha-1over64 then alpha-0.345, each with --expect-commit at this amendment's commit, one MPS slot each via the box "
                               "slot-waiter (r1-queue); alpha-0.345 (D_perp 5x the 1/16 value) is expected to extinguish faster - the ignition gate bounds "
                               "that cost to ~1 h and records the point; alpha-1over64 (D_perp 4x smaller) is the case the latch protects"),
        "series_verdict_consequence": ("with 1/16 at no_plateau, trend_confirmed needs both 1/64 and 0.345 to reach (a) (3 of 4 points incl. alpha = 0); "
                                       "if 0.345 also extinguishes the predeclared rule returns inconclusive, and the recorded finding is stated as such: the "
                                       "Bohm closure at alpha >= 1/16 extinguishes this discharge; the sign hypotheses are not testable at those alphas"),
    },
]

# predeclared expectations (audit section 4.c and roadmap R1; the SIGN is the hypothesis, the magnitude is what the series measures)
HYPOTHESES: dict[str, dict[str, Any]] = {
    "discharge_current_a": {"sign": "+", "expected_at_1over16": "+20 to +60 %", "reason": "the closure opens a cross-field leak path to the anode through the "
                            "cusp mirrors: more electron current reaches the anode per injected electron"},
    "ionization_rate_per_s": {"sign": "-", "expected_at_1over16": "-10 to -40 %", "reason": "electrons leak before ionising: the confinement time and the "
                              "electron inventory fall"},
    "gross_utilisation": {"sign": "-", "expected_at_1over16": "-10 to -40 %", "reason": "utilisation = S / feed"},
    "neutral_density_per_m3": {"sign": "+", "expected_at_1over16": "+5 to +25 %", "reason": "less ionisation depletes the inventory less (n_g fixed point rises)"},
    "peak_n_e_window_per_m3": {"sign": "-", "expected_at_1over16": "-15 to -40 %", "reason": "the density that builds up before the cusp mirrors is bounded by the leak"},
    "t_e_peak_window_ev": {"sign": "-", "expected_at_1over16": "-5 to -25 %", "reason": "the mirror-trapped hot tail leaks; less finite-grid heating at lower n_e"},
    "exit_ion_beam_a": {"sign": "-", "expected_at_1over16": "-10 to -40 %", "reason": "follows S (weak: the beam fraction may rise as the anode ion loss changes)"},
    "cusp_electron_wall_current_a": {"sign": "+", "expected_at_1over16": "up", "reason": "per-cusp wall electron loss rises as the closure feeds electrons into the loss cone region"},
    "cusp_sheath_drop_v": {"sign": "-", "expected_at_1over16": "down", "reason": "cusp sheath drops fall as the hot tail is no longer mirror-trapped"},
}
MONOTONE_QUANTITIES: tuple[str, ...] = ("discharge_current_a", "ionization_rate_per_s", "gross_utilisation", "peak_n_e_window_per_m3", "t_e_peak_window_ev")
# the ss-v4 particle band (seed-b / W x 0.7 of the 50 um pair, 2x = the v4 tolerances): a shift inside the band is not a trend
PARTICLE_BAND: dict[str, float] = {"discharge_current_a": 0.057, "exit_ion_beam_a": 0.057, "ionization_rate_per_s": 0.046, "gross_utilisation": 0.046,
                                   "neutral_density_per_m3": 0.040, "peak_n_e_window_per_m3": 0.119, "t_e_peak_window_ev": 0.093}


def load_v4_protocol() -> dict[str, Any]:
    return json.loads(V4_PROTOCOL_PATH.read_text(encoding="utf-8"))


def v4_reference_block() -> dict[str, Any]:
    """The alpha = 0 point of the series: the recorded ss-v4 plateau (0d228ad2) with its CORRECTED ledger status."""

    v5 = json.loads((MODERN / "experiments" / "pic2d_cft_steady_state_v5" / "protocol.json").read_text(encoding="utf-8"))["reference_run"]
    quantities = dict(v5["quantities"])
    return {
        "case": REFERENCE_CASE, "alpha": 0.0, "experiment": "modern/experiments/pic2d_cft_steady_state_v4", "results_dir": "modern/experiments/pic2d_cft_steady_state_v4/results",
        "commit": v5["commit"], "run_git_head": v5["run_git_head"], "protocol_sha256_prefix": v5["protocol_sha256_prefix"], "config_sha256_prefix": v5["config_sha256_prefix"],
        "grid": v5["grid"], "plateau": v5["plateau"], "quantities": quantities,
        "corrected_ledger": {
            "sidecar": "modern/experiments/pic2d_cft_steady_state_v4/results/ledger-corrected.json (v2.0.6, 02013df0)",
            "windowed_residual_over_electrode_work_corrected": REFERENCE_CORRECTED_RESIDUAL,
            "acceptance_b_below_0p02": False,
            "statement": "the alpha = 0 reference FAILS its own acceptance (b) on the corrected ledger: +2.46 % of the electrode work in the trailing 400 000-step "
                         "window at the stop (recorded -7.67 % before the v2.0.6 W fix; still rising at the plateau). It is the series' alpha = 0 point AS RECORDED, "
                         "not a clean reference: every difference reported against it carries this caveat, and a case that passes (b) while the reference does not is "
                         "itself evidence for the audit's expectation that the leak path bounds n_e (less finite-grid heating)",
        },
        "particle_band": PARTICLE_BAND,
        "particle_band_note": "the 50 um convergence pair's relative bands (seed-b / W x 0.7; the v4 c-tolerances are 2x these): a shift smaller than the band is "
                              "reported as 'inside the particle band' and does not count toward the trend verdict",
    }


def compose_case_protocol(case_id: str, *, wall_budget_seconds: float | None = None, budget_note: str | None = None) -> dict[str, Any]:
    """The ss-v4 protocol with the alpha-series changes for ``case_id`` (deterministic; sealed under ``protocols/``)."""

    if case_id not in CASES:
        raise KeyError(f"unknown case {case_id!r}; cases: {sorted(CASES)}")
    case = CASES[case_id]
    alpha = float(case["alpha"])
    p = copy.deepcopy(load_v4_protocol())
    p["schema_version"] = CASE_SCHEMA_VERSION
    p["experiment_id"] = f"{EXPERIMENT_ID}-{case_id}"
    p["campaign"] = {"experiment_id": EXPERIMENT_ID, "case": case_id, "alpha": alpha, "alpha_label": case["label"], "role": case["role"],
                     "series": {k: v["alpha"] for k, v in CASES.items()} | {REFERENCE_CASE: 0.0}, "launch_priority": list(LAUNCH_PRIORITY),
                     "template": "modern/experiments/pic2d_cft_steady_state_v4/protocol.json (the alpha = 0 run; every key not named in campaign.changes is byte-for-byte its value)",
                     "changes": ["numerics.anomalous_collisions (model bohm_perpendicular_rotation, alpha)", "numerics.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak (v2.0.6 floor)",
                                 "numerics.performance.moment_sample_interval = 5 (v2.0.5)", "stopping_rule.wall_budget_seconds (launch-box measured rate x 1.5)",
                                 "stopping_rule.acceptance (b corrected ledger, c -> alpha-series trend, hypotheses)", "case.id", "status / classification / model_version / model_spec / claim_boundary / simplifications text",
                                 "reference_run -> the ss-v4 plateau as the alpha = 0 point with its corrected-ledger status",
                                 "AMENDMENT 1: stopping_rule.grid_heating_triad.drift_members_arming (model v2.1.1 latch) + stopping_rule.ignition_gate (v2.0 gate, calibrated on the accepted 33 um runs)"]}
    p["status"] = "preregistered_anomalous_transport_alpha_series_not_validated"
    p["classification"] = ("axisymmetric_electrostatic_pic_mcc_bohm_alpha_series_on_the_33um_reference_plateau_v1_3_closure_v2_1_0_transport_v2_0_6_gates_not_validated")
    p["model_version"] = MODEL_VERSION
    p["model_spec"] = MODEL_SPEC
    num = p["numerics"]
    num["anomalous_collisions"] = {
        "model": ANOMALOUS_MODEL_ROTATION, "alpha": alpha,
        "alpha_note": (f"model v2.1.0 Bohm-type anomalous transport: every electron has its velocity rotated about the local B by a uniform random angle "
                       f"(v_parallel and |v| unchanged, gyro-centre shifted - Brandt et al. 2016's event model) with probability 1 - exp(-alpha omega_ce dt) per step, "
                       f"omega_ce = e|B|/m_e at the particle; nu_an = {alpha:.6g} omega_ce = {alpha * 1.7588e11 * 0.05:.3g} s^-1 at 0.05 T, {alpha * 1.7588e11 * 0.2914:.3g} s^-1 "
                       f"at the channel max |B| 0.2914 T (nu_an dt = {alpha * 1.7588e11 * 0.2914 * 1.4e-12:.4f}); cross-field diffusion D_perp = (kT_e/eB) alpha/(1+alpha^2) = "
                       f"{alpha / (1 + alpha**2):.4f} kT_e/eB (Green-Kubo, verified by tests/pic2d/test_pic2d_v210_anomalous_transport.py to 5 %). Elastic: no energy term, "
                       f"count tallied in cumulative.anomalous, axial momentum in pz_collisions; a separate exact-Poisson process outside the MCC null-collision budget "
                       f"(equivalent to O(nu_an dt x nu_mcc dt) ~ 1e-7). The alpha = 0 run is the ss-v4 plateau: without the block the configuration identity is "
                       f"byte-for-byte f10772b25b03 (test-pinned)"),
    }
    gate = num["peak_debye_gate"]
    gate["min_accumulated_macro_particle_steps_at_peak"] = 64000
    gate["min_accumulated_macro_particle_steps_at_peak_note"] = (
        "v2.0.6 (spec gates_v2_0.peak_debye_gate_accumulated_floor_v2_0_6): the gated node is the densest node whose ACCUMULATED macro-electron-steps over the "
        "400 000-step window reach 64 000 (= 32 crossings x ~2000 steps), so axis columns that hold < 32 macro-electrons per step but are visited by many are "
        "gate-able (the v2.0.3 mean-occupancy floor of 32 made them invisible); on the v4 maps the gated statistic is unchanged (2.154 at the same node, "
        "resolved nodes 19 650 -> 42 130); the mean-occupancy floor stays recorded alongside")
    num["performance"] = {"moment_sample_interval": 5,
                          "moment_sample_interval_note": "v2.0.5: electron window moments (T_e maps, peak-Debye T_e, sample counts) sampled every 5th accumulated step; "
                                                         "physics bitwise, gated Delta/lambda_D moves by 1.7e-5 median vs K = 1 (8aca6c3a); enters config_sha256 by the "
                                                         "v2.0.5 identity policy (K != 1 is a declared configuration); the alpha = 0 reference ran K = 1"}
    num["frame_recorder_note"] = ("v2.0 frame recorder ON (28 ns frames): the alpha-series' ionisation / density / potential structure at the cusp planes against the "
                                  "v4 frames (0d228ad2), and a video per case; frames are diagnostics, not gates")
    p["case"]["id"] = f"alpha-{case['label'].replace('/', 'over')}-bohm-rotation-33um-dt1.4ps-w2.667e4-ng0-5.5e19-seed5e16-inventory-v1.3-closure"
    p["case"]["seed_note"] = "seed 20260903 = the ss-v4 seed: the hook draws from its own stream (stream 3 on the CPU reference, seed-table slot 2 on Warp), so the seed / injection / MCC draws of the alpha = 0 run are reproduced step for step until the first anomalous event"
    p["budget_v1_3"]["cost_model"] = {
        "source": "the mini-sweep reference run (same 90 x 720 grid, W 26 666.7, ss-v4 template) on the H100 under CUDA MPS with 3-4 clients: 6.19 ms/step mean over 5.24 M "
                  "steps (schedule.py status, 2026-09-04 16:18 UTC) -> 8.8 h to 3 transits; the bohm_kernel adds one per-electron pass (audit: +1-2 %)",
        "steps_to_3_transits": STEPS_TO_3_TRANSITS,
        "a_priori_hours_to_3_transits": "8.9-9.0 h in an MPS slot (4.05 ms/step solo on the RTX 5090 = 5.8 h)",
        "gpu_memory_estimate_gb": "~4 (the v4 run's device pool; 4.5 M macro-particles at the plateau)",
        "budget_basis": "wall_budget_seconds = 1.5 x the launch-box preflight rate at the plateau load x steps_to_3_transits, recorded in preflight-<case>.json before the preregistration commit",
    }
    stop = p["stopping_rule"]
    if wall_budget_seconds is not None:
        stop["wall_budget_seconds"] = float(wall_budget_seconds)
        stop["wall_budget_note"] = budget_note or "1.5 x the launch-box measured plateau-load rate x 3-transit steps (preflight-<case>.json)"
    else:
        stop["wall_budget_seconds"] = 50400.0
        stop["wall_budget_note"] = ("A-PRIORI 14.0 h = 1.5 x 9.3 h (6.5 ms/step in an MPS slot x 5.14 M steps); REPLACED by the launch-box measured rate x 1.5 before the "
                                    "preregistration commit (compose --budget-from-preflight)")
    stop["fail_closed"] = stop["fail_closed"].replace("v2.0.3 window-mode peak-node Debye gate", "v2.0.6 window-mode peak-node Debye gate (accumulated-particle-step floor)") \
        .replace("v2.0.3 windowed residual-power bound", "v2.0.3 windowed residual-power bound on the v2.0.6 W-corrected ledger")
    stop["fail_closed"] += ("; AMENDMENT 1 (model v2.1.1): the triad's drift members are armed by the settled-once latch (grid_heating_triad.drift_members_arming: "
                            ">= 2 transits AND the I_d drift has read < 5 % at a checkpoint) instead of at 1.0 transit, and the v2.0 ignition gate "
                            "(stopping_rule.ignition_gate) stops an extinguished discharge at 1.0 / 2.0 us (stop_reason no_ignition)")
    stop["grid_heating_triad"]["note"] += ("; v2.0.6: the ledger's inelastic_loss_j carries W, so the windowed statistic IS the corrected one (the recorded ss-v4 "
                                           "series read -7.7 % where the corrected value was +2.46 %)"
                                           "; AMENDMENT 1 (model v2.1.1): enforced_after_transit_times 1.0 is SUPERSEDED by drift_members_arming (kept as the "
                                           "recorded rule launch 1 of alpha-1over16 ran under); the residual-power member is unchanged")
    stop["grid_heating_triad"]["drift_members_arming"] = copy.deepcopy(DRIFT_MEMBERS_ARMING)
    stop["ignition_gate"] = copy.deepcopy(IGNITION_GATE)
    stop["acceptance"] = {
        "declared": "predeclared before the launch; evaluated by `run.py assess --case <case>` (per case) and `run.py assess --series` (the trend) against reference_run "
                    "(the alpha = 0 ss-v4 plateau); verdicts recorded in results/<case>/assessment.json and results/series-assessment.json (results-only commits)",
        "a_plateau": "stop_reason == plateau_reached_after_min_transit_times under the v4 rule (>= 3 transits = 5 142 858 steps, trailing-20 % drifts of I_d, N_e, n_g "
                     "< 5 %, triad soft bounds, window-mode peak-Debye soft margin 2.5)",
        "b_residual_power": "summary.grid_heating_triad.windowed_energy_residual_over_electrode_work (trailing 400 000-step ratio at the stop, v2.0.6 W-corrected "
                            "ledger) < +0.02, one-sided; the alpha = 0 reference reads +0.0246 on the corrected ledger (FAIL) - a case that passes (b) is a "
                            "cleaner plateau than the reference",
        "c_series_trend": {
            "quantities": list(MONOTONE_QUANTITIES) + ["neutral_density_per_m3", "exit_ion_beam_a"],
            "rule": "for every quantity in hypotheses with a declared sign, the relative shift (case - reference) / |reference| is reported with the particle band; a "
                    "shift counts as CONFIRMING when it has the declared sign AND exceeds the band, CONTRADICTING when it has the opposite sign AND exceeds the band, "
                    "INSIDE THE BAND otherwise. The series is MONOTONE in a quantity when the values at alpha = 0, 1/64, 1/16, 0.345 are ordered in the declared "
                    "direction (ties inside the band allowed)",
            "hypotheses": HYPOTHESES,
            "particle_band": PARTICLE_BAND,
            "per_cusp": {"planes_m": list(CUSP_PLANES_M), "half_width_m": CUSP_HALF_WIDTH_M,
                         "report": "electron and ion wall current within +-1 mm of each cusp plane (maps.npz wall fluxes x wall area), the axis-to-wall potential "
                                   "drop and the near-wall T_e at the plane; reported per case beside the v4 values, signs judged by the hypotheses rows "
                                   "cusp_electron_wall_current_a / cusp_sheath_drop_v (they do not enter the monotone set: the v4 particle band for them is not measured)"},
        },
        "d_verdict": {
            "per_case": {
                "plateau_clean": "(a) AND (b): a quotable plateau of the closure at this alpha",
                "plateau_heating": "(a) but NOT (b): the plateau heats above 2 % (like the alpha = 0 reference); the shifts are reported with the heating caveat",
                "no_plateau": "NOT (a): budget / gate stop; trailing-window quantities reported, no trend contribution from this case",
            },
            "series": {
                "trend_confirmed": "at least 3 of the 4 series points reached (a) (the reference counts) AND I_d and peak n_e are monotone in the declared direction "
                                   "AND no monotone quantity CONTRADICTS at any reached point",
                "trend_not_confirmed": "at least 3 points reached (a) AND (I_d or peak n_e is not monotone in the declared direction, OR some monotone quantity "
                                       "CONTRADICTS at a reached point) - a finding about the closure's sign, recorded as such",
                "inconclusive": "fewer than 3 points reached (a), or every shift of I_d and peak n_e is inside the particle band at every reached point",
            },
            "note": "the verdict is about the SIGN and monotonicity of the closure's effect on the recorded plateau; the magnitude at each alpha is the measurement. "
                    "No alpha is 'chosen' by this campaign: alpha stays a declared closure parameter (audit section 6) until a companion r-theta / z-theta campaign "
                    "supplies the mobility",
        },
    }
    p["simplifications"] = [
        s.replace("electrostatic only, azimuthally symmetric: no azimuthal instabilities/anomalous transport",
                  "electrostatic only, azimuthally symmetric: no azimuthal instabilities; the anomalous cross-field transport they would drive is IMPOSED as a declared "
                  "Bohm-type closure (alpha), not computed")
        for s in p["simplifications"] if not s.startswith("development/screening run") and not s.startswith("single seed and a single refined grid") and not s.startswith("preregistered resolution-convergence")
    ] + [
        "the Bohm closure is a constant alpha everywhere (the probability follows the local |B| only); a per-cell effective mobility from companion instability-plane runs is the declared follow-up (audit section 6 ii)",
        "single seed per alpha: the shifts are judged against the recorded 50 um particle band, not a per-alpha replicate",
        "the alpha = 0 reference (ss-v4) heats at +2.46 % of the electrode work on the corrected ledger: differences against it carry that caveat",
        "preregistered closure-sensitivity study of a development model: no experimental validation, not a performance prediction",
    ]
    p["claim_boundary"] = ("preregistered alpha-series (Bohm-type anomalous cross-field transport, perpendicular-rotation model, alpha in {1/64, 1/16, 0.345} beside the "
                           "recorded alpha = 0 plateau) on the reference design at 33.3 um / 1.4 ps / W 2.667e4 under the v1.3 closure and the v4 operating point; the outcome "
                           "is the sign / monotonicity of the closure's effect on I_d, S, utilisation, n_g, I_beam, peak n_e, T_e,peak and the per-cusp wall losses, with the "
                           "magnitudes recorded; every discharge quantity of the 2D axisymmetric model is conditional on the declared alpha (physics audit section 6); not "
                           "validated against experiment; not a thruster performance prediction; the neutral transient is artificial and only the fixed point is physical")
    p["reference_run"] = v4_reference_block()
    p["preregistration"] = {
        "protocol": "protocols/<case>.json is frozen at the preregistration commit (its sha256 is listed in the campaign protocol.json); summary.json records protocol_sha256 "
                    "and git_head; run.py launch refuses a dirty worktree, a HEAD that is not the recorded preregistration commit, a sealed protocol that differs from its "
                    "recomposition, or an existing execution lock",
        "preflight": "preflight-<case>.json on the launch box (real P2 field on the 90 x 720 grid, mesh, factorisation, memory, ms/step at the seed load and at a synthetic "
                     "~4.5 M-particle load, GPU load before) - non-evidentiary; the budget is derived from it",
        "shakedown": "shakedown-<case>.json: a 100 000-step real-input run of one case (results-shakedown/, cadences shrunk, every gate live) through run -> finalize -> "
                     "assess (case and series) on the launch box - non-evidentiary, not committed beyond the record",
        "one_execution": "one detached launch per case from the scheduler's worktree at the preregistration commit, one MPS slot each, in the declared priority order as "
                         "slots free; a wall-budget stop may be resumed (new session, same identity, disclosed); no parameter is changed after the freeze",
    }
    return p


def protocol_sha256(protocol: dict[str, Any]) -> str:
    """sha256 of the sealed file bytes (``canonical_bytes + newline`` is exactly what ``write_sealed_protocols`` writes)."""

    return sha256(canonical_bytes(protocol) + b"\n").hexdigest()


def compose_campaign(case_protocols: dict[str, dict[str, Any]], *, budgets: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The campaign-level protocol: design, sealed case hashes, launch order, acceptance (mirrors the per-case block), amendments."""

    first = next(iter(case_protocols.values()))
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "preregistered_anomalous_transport_alpha_series_not_validated",
        "title": "Bohm-type anomalous cross-field transport: predeclared alpha-series on the 33 um reference plateau (roadmap R1)",
        "model_version": MODEL_VERSION,
        "model_spec": MODEL_SPEC,
        "design": {
            "template": "modern/experiments/pic2d_cft_steady_state_v4/protocol.json (reference design divergent-exit-stack, 90 x 720 at 33.33 um, dt 1.4 ps, W 26 666.7, "
                        "v1.3 closure, seed 20260903, frames ON)",
            "closure": {"model": ANOMALOUS_MODEL_ROTATION, "series": {k: v["alpha"] for k, v in CASES.items()} | {REFERENCE_CASE: 0.0},
                        "reference_point": "alpha = 0 = the recorded ss-v4 plateau (0d228ad2), NOT re-run",
                        "d_perp_over_kt_e_by_eb": {k: v["alpha"] / (1.0 + v["alpha"] ** 2) for k, v in CASES.items()} | {REFERENCE_CASE: 0.0}},
            "cases": {k: {"alpha": v["alpha"], "label": v["label"], "role": v["role"]} for k, v in CASES.items()},
            "launch_priority": list(LAUNCH_PRIORITY),
            "launch_priority_note": "1/16 first (the audit's central expectation), then 1/64 (the weak end), then 0.345 (Brandt's coefficient; the strongest change, "
                                    "likeliest to reach a clean plateau); each takes one H100 MPS slot as the scheduler frees one",
            "expected_changes": HYPOTHESES,
            "cusp_planes_m": list(CUSP_PLANES_M),
        },
        "acceptance": first["stopping_rule"]["acceptance"],
        "reference_run": first["reference_run"],
        "sealed_protocols": {f"modern/experiments/pic2d_anomalous_transport_v1/protocols/{case}.json": protocol_sha256(p) for case, p in case_protocols.items()},
        "budgets": budgets or {case: {"wall_budget_seconds": p["stopping_rule"]["wall_budget_seconds"], "note": p["stopping_rule"]["wall_budget_note"]} for case, p in case_protocols.items()},
        "identity_policy": {
            "alpha_0": "the hook absent -> config_sha256 f10772b25b03... = the ss-v4 record (test-pinned): the series' alpha = 0 point IS the recorded run",
            "alpha_gt_0": "numerics.anomalous_collisions {model, alpha} enters config_sha256 (a different closure is a different configuration identity)",
            "k_5": "moment_sample_interval 5 enters config_sha256 (v2.0.5 policy); physics bitwise vs K = 1, diagnostics differ at <= 1.6e-3 relative (8aca6c3a)",
            "debye_floor": "min_accumulated_macro_particle_steps_at_peak 64000 enters config_sha256 only because it is declared (v2.0.6 policy)",
            "ledger": "the v2.0.6 W correction is code (bug fix, identity unchanged); acceptance (b) reads the corrected statistic natively",
        },
        "preregistration": first["preregistration"],
        "amendments": copy.deepcopy(AMENDMENTS),
    }


def write_sealed_protocols(case_protocols: dict[str, dict[str, Any]], campaign: dict[str, Any]) -> list[Path]:
    PROTOCOLS_DIR.mkdir(exist_ok=True)
    written = []
    for case, p in case_protocols.items():
        path = PROTOCOLS_DIR / f"{case}.json"
        path.write_bytes(canonical_bytes(p) + b"\n")
        written.append(path)
    CAMPAIGN_PROTOCOL_PATH.write_bytes(canonical_bytes(campaign) + b"\n")
    written.append(CAMPAIGN_PROTOCOL_PATH)
    return written


def load_case_protocol(case_id: str) -> dict[str, Any]:
    path = PROTOCOLS_DIR / f"{case_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"sealed protocol {path} missing; run `run.py compose` first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_campaign() -> dict[str, Any]:
    return json.loads(CAMPAIGN_PROTOCOL_PATH.read_text(encoding="utf-8"))


__all__ = [
    "AMENDMENTS", "CASES", "CUSP_HALF_WIDTH_M", "CUSP_PLANES_M", "DRIFT_MEMBERS_ARMING", "EXPERIMENT_ID", "HYPOTHESES", "IGNITION_GATE", "LAUNCH_PRIORITY",
    "MONOTONE_QUANTITIES", "PARTICLE_BAND", "PRE_AMENDMENT_SEALED_SHA256", "REFERENCE_CASE", "REFERENCE_CORRECTED_RESIDUAL", "STEPS_TO_3_TRANSITS",
    "compose_campaign", "compose_case_protocol", "load_campaign", "load_case_protocol", "load_v4_protocol", "protocol_sha256", "v4_reference_block",
    "write_sealed_protocols",
]
