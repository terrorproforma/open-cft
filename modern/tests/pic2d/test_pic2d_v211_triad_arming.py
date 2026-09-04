"""Model v2.1.1 (2026-09-05): arming of the grid-heating triad's DRIFT members relative to the run's own discharge.

After the alpha-series launch 1 (alpha = 1/16) stopped ``grid_heating_triad_gate_stopped_run`` at exactly 1.00 transit - the
instant the drift members armed under the v1.4 rule ``enforced_after_transit_times = 1.0`` - with the residual-power member at
+1.15 % and the peak at 0.48 cells per lambda_D (nothing heating), the drift members may be armed by a "settled once" latch:
enforced only after ``min_transit_times`` AND after the trailing-20 % I_d drift has read inside the plateau bound at a checkpoint.

Regressions pinned here:

* contract: an absent block is the v1.4 rule; the block's keys are validated; the arming state travels with the triad record;
* a synthetic re-equilibration with a 40 % I_d / S drift at 1.0 transit trips the v1.4 rule and does NOT trip the latch rule; once
  the discharge has settled (latch closed after 2 transits) a later S runaway DOES trip - the members are not switched off;
* the residual-power member is independent of the arming (fires from the first complete window while unlatched);
* the accepted alpha = 0 plateau (ss-v4 series) never trips under the latch rule; its I_d latch closes at 2.65 transits;
* plume attempt 8 (the finite-grid heating runaway) is still stopped by the residual member at the same record as before;
* the alpha-series ignition gate separates the accepted 33 um runs (N_e ratio >= 1.31 at 1 us) from the extinguished 1/16 launch
  (0.45) with the declared margins;
* runner integration: a tiny CPU protocol with the block records the arming state in status / summary.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d.fields import linear_psi_field_map
from cft_revival.pic2d.models import PIC2DValidationError
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v1.gpu_sampler import GpuUtilisationSampler

MODERN = Path(__file__).resolve().parents[2]
TRANSIT = 2.4e-6
DT = 1.4e-12
CADENCE = 40_000

LEGACY_RULE = {
    "plateau_threshold": 0.05, "plateau_window_fraction": 0.2, "min_transit_times": 3,
    "grid_heating_triad": {"energy_residual_over_electrode_work_max": 0.10, "soft_drift_max": 0.05, "hard_drift_max": 0.25,
                           "enforced_after_transit_times": 1.0, "residual_window_steps": 400_000,
                           "windowed_energy_residual_over_electrode_work_max": 0.05},
}
ARMING_BLOCK = {"min_transit_times": 2.0, "settle_quantity": "discharge_current", "settle_drift_max": 0.05, "settle_check_cadence_steps": CADENCE}
LATCH_RULE = copy.deepcopy(LEGACY_RULE)
LATCH_RULE["grid_heating_triad"]["drift_members_arming"] = dict(ARMING_BLOCK)

# the alpha-series amendment-1 ignition gate (calibrated in spec triad_drift_arming_v2_1_1.calibration_on_the_accepted_runs)
ALPHA_IGNITION_GATE = {
    "reference_window_s": [0.05e-6, 0.2e-6], "check_window_s": 0.15e-6,
    "checks": [{"time_s": 1.0e-6, "min_s_ratio": 0.3, "min_electron_ratio": 0.6}, {"time_s": 2.0e-6, "min_s_ratio": 0.4, "min_electron_ratio": 0.6}],
}

V4_SERIES = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results" / "series.npz"
ATTEMPT8_SERIES = MODERN / "experiments" / "pic2d_cft_plume_v1" / "results-attempt8-grid-heating-triad-stop" / "series.npz"
ALPHA_1OVER16_SERIES = MODERN / "experiments" / "pic2d_anomalous_transport_v1" / "results" / "alpha-1over16" / "series.npz"


@pytest.fixture(autouse=True)
def _no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(runner, "GpuUtilisationSampler", functools.partial(GpuUtilisationSampler, query=lambda timeout_s: None))


def _truncate(arrays: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {key: value[:n] for key, value in arrays.items()}


def _load(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    series = np.load(path)
    return {key: np.asarray(series[key], dtype=np.float64) for key in keys if key in series.files}


TRIAD_KEYS = ("step", "time_s", "interval_residual_j", "interval_electrode_work_j", "current_ionization_rate_per_s", "peak_omega_pe_dt",
              "peak_node_t_e_dense_ev", "current_discharge_a", "electrons")


def _reequilibration(t_end_s: float, *, sigmoid_gain: float = 2.0, sigmoid_width_s: float = 0.3e-6, tail_gain: float = 1.5, tail_tau_s: float = 2.0e-6,
                     runaway_from_s: float | None = None, residual_ratio: float = -0.005, interval_steps: int = 2000) -> dict[str, np.ndarray]:
    """A discharge moving from the alpha = 0 state to a new one: I_d, N_e and S follow 1 + g1 sigmoid((t - 1 transit) / w) + g2 (1 -
    exp(-(t - 1 transit) / tau2)) - a fast transition around 1 transit and a slow tail - so that the trailing-20 % drift reads 0.41 at
    the first checkpoint past 1.0 transit (> the hard 0.25), 0.18 at 1.5, 0.074 at 2.0 and falls below 0.05 between 2.4 and 2.6
    transits (tuned offline); optional S runaway (linear +80 % per transit) from ``runaway_from_s``; quiet (cooling) residual;
    constant T_e,dense and omega_pe dt."""

    steps = np.arange(interval_steps, round(t_end_s / DT) + 1, interval_steps, dtype=float)
    t = steps * DT
    shape = 1.0 + sigmoid_gain / (1.0 + np.exp(-(t - TRANSIT) / sigmoid_width_s)) + tail_gain * (1.0 - np.exp(-np.maximum(t - TRANSIT, 0.0) / tail_tau_s))
    s_rate = 3.6e16 * shape
    if runaway_from_s is not None:
        s_rate = s_rate * np.where(t > runaway_from_s, 1.0 + 0.8 * (t - runaway_from_s) / TRANSIT, 1.0)
    electrode = np.full(t.size, 1.2 * interval_steps * DT)
    return {
        "step": steps, "time_s": t, "current_discharge_a": 3.8e-3 * shape, "electrons": 2.0e6 * shape,
        "current_ionization_rate_per_s": s_rate, "peak_omega_pe_dt": np.full(t.size, 0.09), "peak_node_t_e_dense_ev": np.full(t.size, 6.5),
        "interval_residual_j": residual_ratio * electrode, "interval_electrode_work_j": electrode,
    }


def _at_transits(arrays: dict[str, np.ndarray], transits: float) -> dict[str, np.ndarray]:
    """The prefix ending at the first checkpoint-cadence record at or after ``transits`` (the checkpoint the runner evaluates
    once that many transits have elapsed; the 1/16 launch stopped at the first checkpoint past 1.0 transit, 1.0033)."""

    mask = (arrays["time_s"] >= transits * TRANSIT - 1e-15) & (np.mod(arrays["step"], CADENCE) == 0)
    return _truncate(arrays, int(np.flatnonzero(mask)[0]) + 1)


# -- contract ------------------------------------------------------------------------------------------------------------

def test_absent_block_is_the_v14_rule_and_the_block_is_validated():
    arrays = _reequilibration(1.05 * TRANSIT)
    assert runner.drift_members_arming(arrays, LEGACY_RULE, TRANSIT) is None
    legacy = runner.evaluate_triad(arrays, LEGACY_RULE, TRANSIT)
    assert "drift_members_arming" not in legacy and legacy["enforced"] is True and legacy["thresholds"]["enforced_after_transit_times"] == 1.0
    assert "enforced_after_transit_times_superseded_by_arming_latch" not in legacy["thresholds"]
    latch = runner.evaluate_triad(arrays, LATCH_RULE, TRANSIT)
    arming = latch["drift_members_arming"]
    assert set(arming) >= {"latched", "armed", "transit_times_elapsed", "min_transit_times", "settle_quantity", "settle_drift_max",
                           "check_cadence_steps", "current_settle_drift", "latched_at_step", "latched_at_transit_times", "drift_at_latch"}
    assert arming["min_transit_times"] == 2.0 and arming["settle_drift_max"] == 0.05 and arming["check_cadence_steps"] == CADENCE
    assert latch["thresholds"]["drift_members_arming"] == {"min_transit_times": 2.0, "settle_quantity": "discharge_current", "settle_drift_max": 0.05,
                                                           "check_cadence_steps": CADENCE}
    assert latch["thresholds"]["enforced_after_transit_times_superseded_by_arming_latch"] is True
    assert latch["thresholds"]["enforced_after_transit_times"] == 1.0          # recorded, not used
    # defaults: settle_drift_max falls back to the plateau threshold, min_transit_times to 2.0, the quantity to I_d
    minimal = copy.deepcopy(LEGACY_RULE)
    minimal["grid_heating_triad"]["drift_members_arming"] = {"settle_check_cadence_steps": CADENCE}
    state = runner.drift_members_arming(arrays, minimal, TRANSIT)
    assert state["min_transit_times"] == 2.0 and state["settle_drift_max"] == 0.05 and state["settle_quantity"] == "discharge_current"
    # fail closed on an unknown settle quantity or a non-positive cadence
    bad = copy.deepcopy(LATCH_RULE)
    bad["grid_heating_triad"]["drift_members_arming"]["settle_quantity"] = "peak_density"
    with pytest.raises(PIC2DValidationError, match="settle_quantity"):
        runner.drift_members_arming(arrays, bad, TRANSIT)
    bad["grid_heating_triad"]["drift_members_arming"] |= {"settle_quantity": "electron_count", "settle_check_cadence_steps": 0}
    with pytest.raises(PIC2DValidationError, match="cadence"):
        runner.drift_members_arming(arrays, bad, TRANSIT)
    bad["grid_heating_triad"]["drift_members_arming"]["settle_check_cadence_steps"] = CADENCE
    assert runner.drift_members_arming(arrays, bad, TRANSIT)["settle_quantity"] == "electron_count"


# -- the re-equilibration that motivated the change ---------------------------------------------------------------------------

def test_a_40_percent_reequilibration_drift_at_one_transit_trips_the_v14_rule_and_not_the_latch_rule():
    arrays = _reequilibration(3.2 * TRANSIT)
    at_one = _at_transits(arrays, 1.0)
    legacy = runner.evaluate_triad(at_one, LEGACY_RULE, TRANSIT)
    assert legacy["enforced"] is True and legacy["hard_failures"] and "ionisation_rate_drift" in legacy["hard_failures"][0]
    assert 0.3 <= legacy["ionisation_rate_drift"] <= 0.5                     # the re-equilibration drift: ~0.4 at 1 transit
    latch = runner.evaluate_triad(at_one, LATCH_RULE, TRANSIT)
    assert latch["enforced"] is False and latch["hard_failures"] == [] and latch["drift_members_arming"]["armed"] is False
    assert latch["ionisation_rate_drift"] == legacy["ionisation_rate_drift"]  # the same reading, recorded not enforced
    assert latch["soft_ok"] is False                                          # still blocks the plateau verdict
    assert 0.3 <= latch["drift_members_arming"]["current_settle_drift"] <= 0.5
    # nothing enforced while the discharge is still moving (1.5 and 2.0 transits), whatever the drifts read
    for transits in (1.5, 2.0):
        state = runner.evaluate_triad(_at_transits(arrays, transits), LATCH_RULE, TRANSIT)
        assert state["hard_failures"] == [] and state["drift_members_arming"]["latched"] is False
    # the latch closes at the first checkpoint >= 2 transits whose trailing-20 % I_d drift reads inside 0.05, and never before 2 transits
    final = runner.evaluate_triad(arrays, LATCH_RULE, TRANSIT)
    arming = final["drift_members_arming"]
    assert arming["latched"] and arming["armed"] and arming["latched_at_transit_times"] >= 2.0 and abs(arming["drift_at_latch"]) < 0.05
    assert arming["latched_at_step"] % CADENCE == 0 and 2.0 <= arming["latched_at_transit_times"] <= 2.6
    before = runner.evaluate_triad(_at_transits(arrays, arming["latched_at_transit_times"] - CADENCE * DT / TRANSIT), LATCH_RULE, TRANSIT)
    assert before["drift_members_arming"]["latched"] is False
    # the same latch is recomputed identically from the full series (pure function of the records; resume / offline safe)
    assert runner.drift_members_arming(arrays, LATCH_RULE, TRANSIT)["latched_at_step"] == arming["latched_at_step"]
    assert final["hard_failures"] == [] and final["enforced"] is True          # settled: armed, drifts inside the hard bound


def test_once_settled_a_later_runaway_still_trips_the_drift_members():
    quiet = _reequilibration(3.2 * TRANSIT)
    latched_at = runner.drift_members_arming(quiet, LATCH_RULE, TRANSIT)["latched_at_transit_times"]
    runaway = _reequilibration(3.2 * TRANSIT, runaway_from_s=(latched_at + 0.05) * TRANSIT)
    # identical series up to the latch -> the latch closes at the same checkpoint
    assert runner.drift_members_arming(runaway, LATCH_RULE, TRANSIT)["latched_at_transit_times"] == latched_at
    final = runner.evaluate_triad(runaway, LATCH_RULE, TRANSIT)
    assert final["enforced"] is True and final["hard_failures"] and "ionisation_rate_drift" in final["hard_failures"][0]
    assert final["ionisation_rate_drift"] >= 0.25


def test_the_residual_power_member_is_independent_of_the_arming():
    # heating from 0.5 transit (per-interval ratio ramping to +20 %) on a discharge that never settles: the residual member fires
    # at its first complete window under BOTH rules while the drift members stay unarmed under the latch
    arrays = _reequilibration(1.6 * TRANSIT)
    t = arrays["time_s"]
    ratio = np.where(t < 0.5 * TRANSIT, -0.005, -0.005 + 0.2 * (t - 0.5 * TRANSIT) / TRANSIT)
    arrays["interval_residual_j"] = ratio * arrays["interval_electrode_work_j"]
    first = None
    for transits in np.arange(0.2, 1.61, 0.05):
        state = runner.evaluate_triad(_at_transits(arrays, transits), LATCH_RULE, TRANSIT)
        if state["hard_failures"]:
            first = (transits, state)
            break
    assert first is not None
    transits, state = first
    assert transits < 1.0 and all("windowed energy residual" in failure for failure in state["hard_failures"])
    assert state["drift_members_arming"]["armed"] is False and state["windowed_energy_residual_window_complete"] is True
    legacy = runner.evaluate_triad(_at_transits(arrays, transits), LEGACY_RULE, TRANSIT)
    assert any("windowed energy residual" in failure for failure in legacy["hard_failures"])


# -- calibration on the recorded runs ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not V4_SERIES.is_file(), reason="ss-v4 series not checked out")
def test_the_accepted_alpha_0_plateau_never_trips_under_the_latch_and_its_latch_closes_at_2p65_transits():
    arrays = _load(V4_SERIES, TRIAD_KEYS)
    assert arrays["step"][-1] == 5_200_000
    steps = arrays["step"]
    for n in range(1, steps.size + 1):
        if int(steps[n - 1]) % 200_000 != 0:
            continue
        legacy = runner.evaluate_triad(_truncate(arrays, n), LEGACY_RULE, TRANSIT)
        latch = runner.evaluate_triad(_truncate(arrays, n), LATCH_RULE, TRANSIT)
        assert legacy["hard_failures"] == [] and latch["hard_failures"] == [], (arrays["time_s"][n - 1], legacy["hard_failures"], latch["hard_failures"])
        assert latch["drift_members_arming"]["armed"] <= legacy["enforced"]  # never enforced earlier than under the v1.4 rule
    final = runner.evaluate_triad(arrays, LATCH_RULE, TRANSIT)
    arming = final["drift_members_arming"]
    # spec triad_drift_arming_v2_1_1: the I_d latch closes at the 40 000-step checkpoint 4 560 000 (2.66 transits) with drift +0.049
    assert arming["latched"] and arming["latched_at_step"] == 4_560_000 and arming["latched_at_transit_times"] == pytest.approx(2.66, abs=1e-3)
    assert 0.045 < arming["drift_at_latch"] < 0.05 and arming["armed"] is True and final["enforced"] is True
    at_one = runner.evaluate_triad(_at_transits(arrays, 1.0), LATCH_RULE, TRANSIT)
    assert at_one["drift_members_arming"]["current_settle_drift"] == pytest.approx(0.116, abs=0.01)   # the alpha = 0 I_d drift at 1 transit
    assert at_one["drift_members_arming"]["armed"] is False


@pytest.mark.skipif(not ATTEMPT8_SERIES.is_file(), reason="plume attempt-8 series not checked out")
def test_attempt_8_heating_is_still_stopped_by_the_residual_member_at_the_same_record():
    arrays = _load(ATTEMPT8_SERIES, TRIAD_KEYS)
    transit = 3.1e-6                                    # 4.98 us = 1.606 transits (summary.ion_transit_times)
    rule = copy.deepcopy(LATCH_RULE)
    rule["grid_heating_triad"]["drift_members_arming"]["settle_check_cadence_steps"] = 40_000

    def first_trip(rule_: dict) -> tuple[float, list[str], dict]:
        steps = arrays["step"]
        for n in range(1, steps.size + 1):
            if int(steps[n - 1]) % 40_000 != 0:
                continue
            triad = runner.evaluate_triad(_truncate(arrays, n), rule_, transit)
            if triad["hard_failures"]:
                return float(arrays["time_s"][n - 1]), triad["hard_failures"], triad
        raise AssertionError("no trip")

    t_latch, why_latch, triad_latch = first_trip(rule)
    t_legacy, why_legacy, _ = first_trip(LEGACY_RULE)
    assert t_latch == t_legacy and all("windowed energy residual" in w for w in why_latch) and any("windowed" in w for w in why_legacy)
    assert 3.0e-6 <= t_latch <= 3.4e-6                  # the recorded (biased) statistic crosses 5 % at ~3.24 us; corrected: 0.66 us
    assert triad_latch["drift_members_arming"]["armed"] is False   # the runaway never settled - the physics member did the work


@pytest.mark.skipif(not V4_SERIES.is_file(), reason="ss-v4 series not checked out")
def test_alpha_series_ignition_gate_passes_the_accepted_plateau_and_fails_the_extinguished_1over16_launch():
    rule = {"ignition_gate": ALPHA_IGNITION_GATE}
    v4 = _load(V4_SERIES, ("step", "time_s", "electrons", "current_ionization_rate_per_s"))
    result = runner.evaluate_ignition(v4, rule)
    assert result["failed"] is False and [c["evaluated"] for c in result["checks"]] == [True, True]
    assert result["checks"][0]["electron_ratio"] == pytest.approx(1.40, abs=0.02) and result["checks"][0]["s_ratio"] == pytest.approx(1.03, abs=0.03)
    assert result["checks"][1]["electron_ratio"] == pytest.approx(2.01, abs=0.03) and result["checks"][1]["s_ratio"] == pytest.approx(1.58, abs=0.05)
    # the accepted run clears every bound by >= 2.2x (N_e) / 2.4x (S) - spec triad_drift_arming_v2_1_1 (047 is the tightest: 1.31 / 0.96)
    for check in result["checks"]:
        assert check["electron_ratio"] >= 2.2 * check["min_electron_ratio"] and check["s_ratio"] >= 2.4 * check["min_s_ratio"]
    if ALPHA_1OVER16_SERIES.is_file():
        dead = _load(ALPHA_1OVER16_SERIES, ("step", "time_s", "electrons", "current_ionization_rate_per_s"))
        # the gate is evaluated at checkpoints: at the first checkpoint past 1.0 us the run fails on N_e (0.45 < 0.6) and S (0.37 > 0.3 passes)
        n = int(np.flatnonzero((dead["time_s"] >= 1.0e-6) & (np.mod(dead["step"], 40_000) == 0))[0]) + 1
        stop = runner.evaluate_ignition(_truncate(dead, n), rule)
        assert stop["failed"] is True and "no ignition" in stop["reason"] and stop["checks"][0]["evaluated"]
        assert stop["checks"][0]["electron_ratio"] == pytest.approx(0.45, abs=0.03) and stop["checks"][0]["electron_ratio"] < 0.6
        assert stop["checks"][0]["s_ratio"] == pytest.approx(0.37, abs=0.05)
        assert dead["time_s"][n - 1] <= 1.06e-6              # stopped ~1.4 us (60 % of the run) earlier than the recorded 2.408 us
        # the drift members alone would never have latched on the extinction (the I_d drift is -0.47 ... -0.70 from 0.5 transit)
        full = runner.evaluate_triad(_load(ALPHA_1OVER16_SERIES, TRIAD_KEYS), LATCH_RULE, TRANSIT)
        assert full["drift_members_arming"]["latched"] is False and full["drift_members_arming"]["current_settle_drift"] < -0.4


# -- runner integration -------------------------------------------------------------------------------------------------------

def _tiny_protocol() -> dict:
    """The v2.0.3 tiny CPU protocol (12 x 96 cells, 5 ps, 40-step checkpoints, 1 ns transit) with the v2.1.1 arming block."""

    protocol = copy.deepcopy(runner.load_protocol())
    protocol["case"].update({"radial_cells": 12, "axial_cells": 96, "macro_weight": 2.0e6, "seed": 3})
    protocol["operating_point"].update({"neutral_density_per_m3": 1.0e21, "electron_injection_current_a": 0.05, "seed_plasma_density_per_m3": 3.0e16})
    protocol["numerics"].update({
        "dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 40, "averaging_window_steps": 80, "ion_subcycle": 1,
        "peak_debye_gate": {"max_cells_per_debye": 50.0, "min_macro_particles_at_peak": 4, "dense_fraction": 0.5, "window_steps": 80,
                            "window_snapshot_steps": 40, "soft_cells_per_debye": 40.0, "max_cells_per_debye_note": "test"},
    })
    protocol["numerics"]["stability_limits"]["max_cell_debye_ratio"] = 4.0
    protocol["numerics"]["stability_reference"] = {"density_per_m3": 1.0e16, "electron_temperature_ev": 5.0, "max_electron_energy_ev": 400.0}
    protocol["budget_v1_2"]["ion_transit_time_s"] = 1.0e-9
    protocol["budget_v1_2"]["n_max_per_m3"] = 4.0e17
    protocol["budget_v1_2"]["n_eq_projected_per_m3"] = 1.0e17
    protocol["stopping_rule"]["grid_heating_triad"] = {
        "energy_residual_over_electrode_work_max": 0.10, "soft_drift_max": 0.05, "hard_drift_max": 0.25, "enforced_after_transit_times": 1.0,
        "residual_window_steps": 80, "windowed_energy_residual_over_electrode_work_max": 0.05,
        "drift_members_arming": {"min_transit_times": 2.0, "settle_quantity": "discharge_current", "settle_drift_max": 0.05, "settle_check_cadence_steps": 40},
    }
    return protocol


def test_runner_records_the_arming_state_in_status_and_summary(tmp_path: Path):
    from cft_revival.pic2d.mcc import XenonCrossSections
    protocol = _tiny_protocol()
    config = runner.build_config(protocol, backend="cpu")
    field = linear_psi_field_map(config.grid, 2.0)
    results = tmp_path / "results"
    summary_path = runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=XenonCrossSections.from_file(),
                                           max_steps=160, log=lambda _: None)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    triad = summary["grid_heating_triad"]
    arming = triad["drift_members_arming"]
    assert arming["check_cadence_steps"] == 40 and arming["min_transit_times"] == 2.0 and arming["settle_quantity"] == "discharge_current"
    assert triad["thresholds"]["enforced_after_transit_times_superseded_by_arming_latch"] is True
    # 160 steps of 5 ps = 0.8 ns at a 1 ns transit -> unarmed, so no drift member can have stopped the run; the tiny system's
    # residual member (80-step window, +7.9 %) does fire - independent of the arming, as designed
    assert arming["armed"] is False and arming["transit_times_elapsed"] == pytest.approx(0.8, abs=1e-6)
    assert all("drift" not in f for f in triad["hard_failures"])
    assert summary["stop_reason"] in ("target_steps_reached", "grid_heating_triad_gate_stopped_run")
    if summary["stop_reason"] == "grid_heating_triad_gate_stopped_run":
        assert all("windowed energy residual" in f for f in triad["hard_failures"])
    status = [json.loads(line) for line in (results / "status.jsonl").read_text(encoding="utf-8").splitlines()]
    with_triad = [s for s in status if s.get("grid_heating_triad") is not None]
    assert with_triad and all("drift_members_arming" in s["grid_heating_triad"] for s in with_triad)
    assert all(s["grid_heating_triad"]["drift_members_arming"]["armed"] is False for s in with_triad)
