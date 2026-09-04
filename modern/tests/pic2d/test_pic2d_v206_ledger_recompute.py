"""``cft_revival.pic2d.ledger_recompute`` (model v2.0.6): post-hoc correction of recorded energy-ledger series.

* Synthetic series with a known per-record inelastic loss: the corrected residual equals ``H`` exactly, the windowed
  statistic is the runner's ``windowed_energy_residual`` at every record, resume-first records stay at zero, the
  cross-check against the final counts is exact, and the count-based correction (``series.jsonl``) agrees with ``H``.
* A v2.0.6 (already W-scaled) record is recognised: corrected == recorded.
* The committed sidecars reproduce from the tracked series (ss-v4: recorded -7.67 % -> corrected +2.46 %; the recorded
  reading equals the summary's gate value).
* The sidecar never touches the recorded files; the CLI writes ``ledger-corrected.json`` + its hash sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts, ledger_recompute
from experiments.pic2d_cft_steady_state_v1 import run as runner

MODERN = Path(__file__).resolve().parents[2]
W = 26_666.7
E_LOSS_PER_EVENT_J = 12.13 * 1.602176634e-19


def _synthetic_series(n: int = 120, *, interval_steps: int = 2_000, resume_at: int | None = None, seed: int = 5) -> dict[str, np.ndarray]:
    """A recorded (pre-v2.0.6) series: residual_recorded = H - (W - 1) L_unscaled with known H and L per record."""

    rng = np.random.default_rng(seed)
    steps = np.arange(n, dtype=np.float64) * interval_steps
    time_s = steps * 1.4e-12
    electrode = np.concatenate([[0.0], rng.uniform(1.0e-10, 1.4e-10, n - 1)])
    h = np.concatenate([[0.0], rng.normal(2.0e-12, 5.0e-13, n - 1)])                 # the true numerical energy creation
    events = np.concatenate([[0.0], rng.integers(1_500, 2_500, n - 1).astype(np.float64)])
    loss_unscaled = events * E_LOSS_PER_EVENT_J
    recorded = h - (W - 1.0) * loss_unscaled
    energy = np.cumsum(rng.normal(0.0, 1.0e-12, n)) + 5.0e-9                        # field energy walk
    work = np.zeros(n)
    work[1:] = h[1:] - np.diff(energy) + electrode[1:]                                # so that H = work + dU - electrode
    if resume_at is not None:
        recorded[resume_at] = 0.0
        electrode[resume_at] = 0.0
        h[resume_at] = 0.0                                                            # the tool must zero it too
    return {
        "step": steps, "time_s": time_s, "interval_residual_j": recorded, "interval_electrode_work_j": electrode,
        "interval_field_work_j": work, "field_energy_j": energy, "current_ionization_rate_per_s": events * W / (interval_steps * 1.4e-12),
        "_h": h, "_loss_unscaled": loss_unscaled, "_events": events,
    }


def _write_results(tmp_path: Path, arrays: dict[str, np.ndarray], *, window_steps: int = 40_000, checkpoint: int = 4_000,
                   already_scaled: bool = False) -> Path:
    results = tmp_path / "results"
    results.mkdir(parents=True)
    public = {k: v for k, v in arrays.items() if not k.startswith("_")}
    artifacts.write_npz(results / "series.npz", public)
    cumulative = {"excitations": 0.0, "ionizations": float(arrays["_events"].sum()), "inelastic_loss_j": float(arrays["_loss_unscaled"].sum())}
    if already_scaled:
        cumulative["inelastic_loss_j"] *= W
        cumulative[ledger_recompute.PER_WEIGHT_KEY] = float(arrays["_loss_unscaled"].sum())
    summary = {
        "experiment_id": "synthetic", "stop_reason": "test", "case": {"macro_weight": W},
        "final_series": {"ledger": {"cumulative": cumulative}},
        "grid_heating_triad": {"thresholds": {"residual_window_steps": window_steps},
                               "windowed_energy_residual_over_electrode_work": None},
    }
    artifacts.write_canonical_json(results / "summary.json", summary)
    artifacts.write_canonical_json(results / "protocol.json", {"numerics": {"checkpoint_every_steps": checkpoint},
                                                                 "stopping_rule": {"acceptance": {"b_residual_power": "< 0.02"}}})
    return results


def test_corrected_residual_is_h_and_the_windowed_statistic_is_the_runners():
    arrays = _synthetic_series()
    corrected, h, resume_first = ledger_recompute.corrected_residual(arrays)
    assert np.allclose(corrected, arrays["_h"], rtol=0.0, atol=1e-24) and not resume_first.any()
    assert np.array_equal(corrected, h) and corrected[0] == 0.0
    # every record's trailing window equals the runner's evaluation of the truncated series
    block = {"residual_window_steps": 40_000, "windowed_energy_residual_over_electrode_work_max": 0.05}
    windowed = ledger_recompute.windowed_ratios(arrays["step"], corrected, arrays["interval_electrode_work_j"], 40_000)
    for n in (2, 15, 21, 60, 119, 120):
        truncated = {k: v[:n] for k, v in arrays.items() if not k.startswith("_")} | {"interval_residual_j": corrected[:n]}
        reference = runner.windowed_energy_residual(truncated, block)
        assert reference["window_complete"] == bool(windowed["complete"][n - 1])
        assert reference["residual_j"] == pytest.approx(windowed["residual_j"][n - 1], rel=1e-12, abs=1e-30)
        assert reference["electrode_work_j"] == pytest.approx(windowed["electrode_j"][n - 1], rel=1e-12, abs=1e-30)
        if reference["ratio"] is not None:
            assert reference["ratio"] == pytest.approx(windowed["ratio"][n - 1], rel=1e-12)
    assert not windowed["complete"][:20].any() and windowed["complete"][20:].all()     # 40 000 / 2 000 = 20 records back


def test_recompute_writes_a_sidecar_with_exact_cross_check_and_leaves_the_recorded_files_alone(tmp_path: Path):
    arrays = _synthetic_series()
    results = _write_results(tmp_path, arrays)
    before = {p.name: p.read_bytes() for p in results.iterdir()}
    record = ledger_recompute.recompute(results, label="synthetic")
    target = ledger_recompute.write_sidecar(results, record)
    assert target.name == ledger_recompute.SIDECAR_NAME and (results / (ledger_recompute.SIDECAR_NAME + ".sha256.json")).is_file()
    for name, data in before.items():
        assert (results / name).read_bytes() == data, name                       # nothing recorded was modified
    loaded = artifacts.read_canonical_json(target)
    assert loaded["schema"] == ledger_recompute.SCHEMA and loaded["parameters"]["macro_weight"] == W
    assert loaded["parameters"]["window_steps"] == 40_000 and loaded["parameters"]["window_steps_source"].startswith("summary.json")
    assert loaded["parameters"]["checkpoint_steps"] == 4_000 and loaded["records"] == 120 and loaded["resume_first_records"] == 0
    # the omitted energy per record is (W - 1) L_unscaled: cumulative and end-state windows agree with the construction
    omitted = (W - 1.0) * arrays["_loss_unscaled"]
    assert loaded["cumulative"]["omitted_inelastic_j"] == pytest.approx(float(omitted[1:].sum()), rel=1e-9)
    assert loaded["cumulative"]["corrected_residual_j"] == pytest.approx(float(arrays["_h"][1:].sum()), rel=1e-9)
    in_window = arrays["step"] > arrays["step"][-1] - 40_000
    end = loaded["end_state_window"]
    assert end["window_complete"] and end["corrected_residual_j"] == pytest.approx(float(arrays["_h"][in_window].sum()), rel=1e-9)
    assert end["recorded_ratio"] < 0.0 < end["corrected_ratio"] and end["omitted_ratio"] == pytest.approx(end["corrected_ratio"] - end["recorded_ratio"])
    check = loaded["cross_check_vs_final_counts"]
    assert check["available"] and not check["already_w_scaled"] and check["verdict"].startswith("exact")
    assert abs(check["relative_difference"]) < 1e-9                              # synthetic: no classical/relativistic mismatch
    # threshold crossings: the recorded statistic (~ -100 % of the loss) never fires, the corrected one is ~+1.7 % (< 5 %)
    gate = loaded["threshold_crossings"]["0.05"]
    assert gate["recorded_first_crossing_at_checkpoint"] is None and gate["corrected_first_crossing_at_checkpoint"] is None
    assert loaded["acceptance_b_residual_power_below_0p02"]["declared_in_protocol"] is True
    trajectory = loaded["trajectory_at_checkpoints"]
    assert all(t["step"] % 4_000 == 0 for t in trajectory[:-1]) and trajectory[-1]["step"] == int(arrays["step"][-1])
    assert "| synthetic |" in "\n".join(ledger_recompute.table_rows([loaded]))


def test_resume_first_records_stay_zero_and_the_cross_check_says_approximate(tmp_path: Path):
    arrays = _synthetic_series(resume_at=50)
    corrected, _, resume_first = ledger_recompute.corrected_residual(arrays)
    assert resume_first.sum() == 1 and resume_first[50] and corrected[50] == 0.0
    results = _write_results(tmp_path, arrays)
    record = ledger_recompute.recompute(results)
    assert record["resume_first_records"] == 1
    assert record["cross_check_vs_final_counts"]["verdict"].startswith("approximate: 1 resume-first")
    assert record["cross_check_vs_final_counts"]["resume_first_records_excluded"] == 1


def test_already_w_scaled_records_are_recognised_and_left_unchanged(tmp_path: Path):
    arrays = _synthetic_series()
    arrays["interval_residual_j"] = arrays["_h"].copy()                            # a v2.0.6 run records H directly
    results = _write_results(tmp_path, arrays, already_scaled=True)
    record = ledger_recompute.recompute(results)
    check = record["cross_check_vs_final_counts"]
    assert check["already_w_scaled"] and check["expected_omitted_j_from_counts"] == 0.0
    assert abs(check["omitted_j_from_series"]) <= 1e-24
    assert record["end_state_window"]["corrected_ratio"] == pytest.approx(record["end_state_window"]["recorded_ratio"], rel=1e-12)


def test_series_jsonl_with_per_record_counts_gives_the_count_based_correction(tmp_path: Path):
    arrays = _synthetic_series(n=40)
    results = tmp_path / "withdrawn"
    results.mkdir()
    cumulative_loss = np.cumsum(arrays["_loss_unscaled"])
    cumulative_events = np.cumsum(arrays["_events"])
    with (results / "series.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for k in range(40):
            handle.write(json.dumps({
                "step": int(arrays["step"][k]), "time_s": float(arrays["time_s"][k]), "field_energy_j": float(arrays["field_energy_j"][k]),
                "ledger": {"interval_residual_j": float(arrays["interval_residual_j"][k]), "interval_electrode_work_j": float(arrays["interval_electrode_work_j"][k]),
                           "interval_field_work_j": float(arrays["interval_field_work_j"][k]),
                           "cumulative": {"excitations": 0.0, "ionizations": float(cumulative_events[k]), "inelastic_loss_j": float(cumulative_loss[k])}},
                "currents_a": {"ionization_rate_per_s": float(arrays["current_ionization_rate_per_s"][k])},
            }) + "\n")
    with pytest.raises(ledger_recompute.LedgerRecomputeError, match="macro_weight"):
        ledger_recompute.recompute(results)                                        # no summary / protocol: W must be given
    record = ledger_recompute.recompute(results, macro_weight=W, window_steps=40_000)
    assert record["inputs"]["series"]["kind"] == "jsonl" and record["inputs"]["series"]["per_record_counts"] is True
    assert record["parameters"]["window_steps_source"] == "command line" and record["parameters"]["checkpoint_steps_source"] == "default"
    counts = record["count_based_correction"]
    assert counts is not None and counts["max_abs_difference_over_max_abs_h"] < 1e-9   # synthetic: exact agreement
    assert counts["sum_by_counts_j"] == pytest.approx(counts["sum_by_h_j"], rel=1e-9)
    assert record["cross_check_vs_final_counts"]["available"] is False               # no summary.json


def test_cli_writes_sidecars_and_prints_the_table(tmp_path: Path, capsys):
    results = _write_results(tmp_path, _synthetic_series())
    assert ledger_recompute.main([str(results), "--label", "cli-run"]) == 0
    out = capsys.readouterr().out
    assert "| cli-run |" in out and "cross-check exact" in out
    assert (results / ledger_recompute.SIDECAR_NAME).is_file()
    # --dry-run computes without writing
    other = _write_results(tmp_path / "dry", _synthetic_series(seed=9))
    assert ledger_recompute.main([str(other), "--dry-run", "--json"]) == 0
    assert not (other / ledger_recompute.SIDECAR_NAME).exists()
    assert json.loads(capsys.readouterr().out)[0]["schema"] == ledger_recompute.SCHEMA


V4_RESULTS = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "results"
COMMITTED = (
    (V4_RESULTS, -0.0766734225349, 0.0245857845353, "ss-v4 33 um: (b) < +2 % recorded pass -> corrected FAIL"),
    (MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results", 0.00372683642438, 0.130055444476, "v2 base 50 um: +13 % heating"),
    (MODERN / "experiments" / "pic2d_cft_plume_v1" / "results-attempt8-grid-heating-triad-stop", 0.416930662920, 0.672586335617, "attempt 8"),
    (MODERN / "experiments" / "pic2d_external_validation_v0" / "results" / "channel-20um-launch1-triad-gate-stop", 0.0743416347812, 0.616734619536, "ext-val L1"),
)


@pytest.mark.parametrize(("results", "recorded", "corrected", "label"), COMMITTED, ids=[c[3].split(":")[0] for c in COMMITTED])
def test_committed_sidecars_reproduce_from_the_tracked_series(results: Path, recorded: float, corrected: float, label: str):
    if not (results / "series.npz").is_file() or not (results / ledger_recompute.SIDECAR_NAME).is_file():
        pytest.skip("recorded series / sidecar not checked out")
    record = ledger_recompute.recompute(results)
    sidecar = artifacts.read_canonical_json(results / ledger_recompute.SIDECAR_NAME)
    end = record["end_state_window"]
    assert end["recorded_ratio"] == pytest.approx(recorded, rel=1e-9) and end["corrected_ratio"] == pytest.approx(corrected, rel=1e-9), label
    assert sidecar["end_state_window"]["corrected_ratio"] == pytest.approx(end["corrected_ratio"], rel=1e-12)
    assert sidecar["cumulative"]["corrected_residual_j"] == pytest.approx(record["cumulative"]["corrected_residual_j"], rel=1e-12)
    assert sidecar["inputs"]["series"]["sha256"] == record["inputs"]["series"]["sha256"]                # bound to the same bytes
    if end["recorded_ratio_in_summary"] is not None:
        assert end["recorded_ratio_matches_summary"] is True                        # the tool reproduces the recorded gate reading
    check = record["cross_check_vs_final_counts"]
    assert check["available"] and abs(check["relative_difference"]) < 5e-3


@pytest.mark.skipif(not (V4_RESULTS / "series.npz").is_file(), reason="ss-v4 series not checked out")
def test_ss_v4_acceptance_b_changes_status_and_the_50um_base_would_have_fired_the_gate():
    v4 = ledger_recompute.recompute(V4_RESULTS)
    assert v4["acceptance_b_residual_power_below_0p02"] == {"declared_in_protocol": True, "recorded_passes": True, "corrected_passes": False}
    assert v4["v2_0_3_hard_gate_0p05"] == {"recorded_would_have_fired": False, "corrected_would_have_fired": False}
    assert v4["max_over_complete_windows"]["corrected"]["ratio"] < 0.03
    base = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results"
    if (base / "series.npz").is_file():
        record = ledger_recompute.recompute(base)
        fired = record["threshold_crossings"]["0.05"]["corrected_first_crossing_at_checkpoint"]
        assert record["v2_0_3_hard_gate_0p05"]["corrected_would_have_fired"] and 2.6e-6 <= fired["time_s"] <= 2.8e-6
        assert record["threshold_crossings"]["0.05"]["recorded_first_crossing_at_checkpoint"] is None
