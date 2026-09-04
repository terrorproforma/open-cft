"""``experiments.pic2d_cft_steady_state_v4.assess_corrected_ledger``: the post-hoc re-read of the preregistered acceptance on the
corrected energy ledger (model v2.0.6).

* Synthetic records: (b) is re-evaluated on the sidecar's corrected statistic with the predeclared bound (not loosened); the
  recorded verdict is carried unchanged beside the (d) mapping on the corrected ledger; the verdict statement names the failure
  (or the pass) explicitly; the record is bound to the byte hashes of the sidecar, the assessment, the summary and the protocol.
* Binding refusals: a sidecar that describes another series, a tampered assessment, a sidecar whose recorded reading is not the
  assessment's (b), an already W-scaled sidecar.
* The committed ``results/assessment-corrected-ledger.json`` reproduces from the tracked files (ss-v4: (b) PASS -> FAIL at
  +2.46 %; verdict on the corrected ledger ``refinement_heating``; recorded ``resolution_limited`` stands).
"""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.models import PIC2DValidationError
from experiments.pic2d_cft_steady_state_v4 import assess_corrected_ledger as reread_module
from experiments.pic2d_cft_steady_state_v4 import run as v4

MODERN = Path(__file__).resolve().parents[2]
EXPERIMENT = MODERN / "experiments" / "pic2d_cft_steady_state_v4"
RESULTS = EXPERIMENT / "results"


def _fake_experiment(tmp_path: Path, *, recorded: float = -0.0767, corrected: float = 0.0246, recorded_verdict: str = "resolution_limited",
                     all_within: bool = False, a_plateau: bool = True, series_sha: str = "a" * 64, already_scaled: bool = False) -> Path:
    """A results directory holding exactly what the re-read binds: protocol, summary, assessment and the ledger sidecar."""

    experiment = tmp_path / f"experiment-{len(list(tmp_path.iterdir()))}"
    results = experiment / "results"
    results.mkdir(parents=True)
    shutil.copy(EXPERIMENT / "protocol.json", experiment / "protocol.json")
    protocol_sha = sha256((experiment / "protocol.json").read_bytes()).hexdigest()
    protocol = v4.load_protocol(experiment / "protocol.json")
    summary = {"experiment_id": protocol["experiment_id"], "stop_reason": "plateau_reached_after_min_transit_times", "steps_completed": 5_200_000,
               "artifacts": {"series_npz_sha256": series_sha, "maps_npz_sha256": "b" * 64}, "provenance": {"config": {"dt_s": 1.4e-12}}}
    summary_sha = artifacts.write_canonical_json(results / "summary.json", summary)
    b_recorded_ok = recorded < 0.02
    assessment = {
        "schema_version": v4.ASSESSMENT_SCHEMA, "utc": "2026-09-04T08:21:44+00:00", "experiment_id": protocol["experiment_id"], "results_dir": "results",
        "run": {"protocol_sha256": protocol_sha, "steps_completed": 5_200_000, "config_sha256": "c" * 64},
        "a_plateau": {"passed": a_plateau, "stop_reason": summary["stop_reason"], "ion_transit_times": 3.033, "plateau": {"reached": a_plateau}, "rule": "r"},
        "b_residual_power": {"passed": b_recorded_ok, "windowed_residual_over_electrode_work": recorded, "window_complete": True, "bound": 0.02,
                             "one_sided": True, "cumulative_witness": -0.0914, "rule": "r"},
        "c_convergence": {"all_within": all_within, "quantities": {"discharge_current_a": {"within": all_within}}, "rule": "r"},
        "d_reclassification": protocol["stopping_rule"]["acceptance"]["d_reclassification"][recorded_verdict],
        "verdict": recorded_verdict, "peak_debye_window": {"cells_per_debye_last": 2.15, "trailing_mean": 2.10, "soft_ok": True},
    }
    artifacts.write_canonical_json(results / "assessment.json", assessment)
    crossing = None if corrected < 0.02 else {"step": 3_440_000, "time_s": 4.816e-6, "ratio": 0.02002}
    sidecar = {
        "schema": "cft.pic2d.ledger-corrected/1.0.0", "experiment_id": protocol["experiment_id"], "generated_by": "python -m cft_revival.pic2d.ledger_recompute",
        "inputs": {"series": {"file": "series.npz", "sha256": series_sha}, "summary": {"file": "summary.json", "sha256": summary_sha}},
        "parameters": {"macro_weight": 26666.7, "window_steps": 400_000}, "last_step": 5_200_000, "last_time_s": 7.28e-6,
        "end_state_window": {"recorded_ratio": recorded, "corrected_ratio": corrected, "window_complete": True, "recorded_ratio_matches_summary": True,
                             "omitted_ratio": corrected - recorded, "corrected_residual_j": corrected * 6.3866e-7, "electrode_work_j": 6.3866e-7, "window_steps": 400_000},
        "cumulative": {"recorded_over_electrode": -0.0914, "corrected_over_electrode": 0.0181},
        "max_over_complete_windows": {"corrected": {"ratio": corrected, "step": 5_194_800, "time_s": 7.2727e-6}},
        "threshold_crossings": {"0.02": {"corrected_first_crossing_at_checkpoint": crossing}, "0.05": {"corrected_first_crossing_at_checkpoint": None}},
        "acceptance_b_residual_power_below_0p02": {"declared_in_protocol": True, "recorded_passes": b_recorded_ok, "corrected_passes": corrected < 0.02},
        "cross_check_vs_final_counts": {"available": True, "already_w_scaled": already_scaled},
    }
    artifacts.write_canonical_json(results / reread_module.SIDECAR_NAME, sidecar)
    return results


def test_synthetic_re_read_flips_b_carries_both_readings_and_binds_its_inputs(tmp_path: Path) -> None:
    results = _fake_experiment(tmp_path)
    protocol = v4.load_protocol(results.parent / "protocol.json")
    before = {p.name: p.read_bytes() for p in results.iterdir()}
    record = reread_module.reread(protocol, results, reference_results=None, write=False, log=lambda _: None)
    assert record["schema_version"] == reread_module.SCHEMA and record["kind"].startswith("post_hoc")
    assert record["verdict_recorded"] == "resolution_limited" and record["verdict_on_corrected_ledger"] == "refinement_heating"
    b = record["b_residual_power"]
    assert b["recorded"]["passed"] is True and b["corrected"]["passed"] is False and b["passed"] is False and b["bound"] == 0.02
    assert b["status_change"] == "PASS (recorded) -> FAIL (corrected)"
    assert b["corrected"]["windowed_residual_over_electrode_work"] == 0.0246 and b["recorded"]["windowed_residual_over_electrode_work"] == -0.0767
    assert b["corrected"]["first_checkpoint_at_or_above_bound"]["time_s"] == pytest.approx(4.816e-6) and b["corrected"]["hard_gate_0p05_would_have_fired"] is False
    assert b["corrected"]["electrode_power_w_in_window"] == pytest.approx(6.3866e-7 / (400_000 * 1.4e-12))
    assert record["a_plateau"]["unchanged"] is True and record["c_convergence"]["unchanged"] is True and record["c_convergence"]["reference_corrected_ledger"] is None
    statement = record["verdict_statement"]
    assert statement.startswith("plateau reached; convergence vs 50 µm as recorded (resolution_limited for 50 µm); residual precondition (b) FAILED on the corrected ledger")
    assert "heating at +2.5 % of electrode work and is NOT a clean reference; 25 µm (v5) pending" in statement
    d = record["d_reclassification"]
    assert d["recorded_verdict"] == "resolution_limited" and d["verdict_on_corrected_ledger"] == "refinement_heating" and "not a convergence test" in d["corrected_text"]
    assert "converged (for 33 um or 50 um) before the 25 um ladder point reports" in record["disallowed_wording"]
    inputs = record["inputs"]
    assert inputs["ledger_corrected"]["sha256"] == sha256((results / reread_module.SIDECAR_NAME).read_bytes()).hexdigest()
    assert inputs["assessment"]["sha256"] == sha256((results / "assessment.json").read_bytes()).hexdigest()
    assert inputs["summary"]["sha256"] == sha256((results / "summary.json").read_bytes()).hexdigest()
    assert inputs["protocol"]["sha256"] == sha256((results.parent / "protocol.json").read_bytes()).hexdigest()
    assert all(inputs["binding_checks"].values())
    # the record is written canonically with its own hash sidecar; nothing recorded was touched
    written = reread_module.reread(protocol, results, reference_results=None, log=lambda _: None)
    assert (results / reread_module.OUTPUT_NAME).is_file() and (results / (reread_module.OUTPUT_NAME + ".sha256.json")).is_file()
    for name, data in before.items():
        assert (results / name).read_bytes() == data, name
    loaded = artifacts.read_canonical_json(results / reread_module.OUTPUT_NAME)
    assert loaded["verdict_statement"] == written["verdict_statement"] == statement


def test_synthetic_pass_on_the_corrected_ledger_keeps_the_recorded_verdict(tmp_path: Path) -> None:
    results = _fake_experiment(tmp_path, corrected=0.009)
    protocol = v4.load_protocol(results.parent / "protocol.json")
    record = reread_module.reread(protocol, results, reference_results=None, write=False, log=lambda _: None)
    assert record["b_residual_power"]["passed"] is True and record["b_residual_power"]["status_change"] == "PASS (recorded) -> PASS (corrected)"
    assert record["verdict_on_corrected_ledger"] == record["verdict_recorded"] == "resolution_limited"
    assert "holds on the corrected ledger (+0.90 % of electrode work < +2 %)" in record["verdict_statement"]
    assert not (results / reread_module.OUTPUT_NAME).exists()
    # a converged recording whose corrected (b) fails maps to refinement_heating as well; a no-plateau recording stays no_plateau
    converged = _fake_experiment(tmp_path, recorded_verdict="converged", all_within=True, corrected=0.03)
    record = reread_module.reread(v4.load_protocol(converged.parent / "protocol.json"), converged, reference_results=None, write=False, log=lambda _: None)
    assert record["verdict_recorded"] == "converged" and record["verdict_on_corrected_ledger"] == "refinement_heating"
    none = _fake_experiment(tmp_path, recorded_verdict="no_plateau", a_plateau=False, corrected=0.03)
    record = reread_module.reread(v4.load_protocol(none.parent / "protocol.json"), none, reference_results=None, write=False, log=lambda _: None)
    assert record["verdict_on_corrected_ledger"] == "no_plateau"


def test_binding_refusals(tmp_path: Path) -> None:
    quiet = lambda _: None
    # the sidecar describes another series
    results = _fake_experiment(tmp_path)
    protocol = v4.load_protocol(results.parent / "protocol.json")
    sidecar = json.loads((results / reread_module.SIDECAR_NAME).read_text(encoding="utf-8"))
    sidecar["inputs"]["series"]["sha256"] = "f" * 64
    artifacts.write_canonical_json(results / reread_module.SIDECAR_NAME, sidecar)
    with pytest.raises(PIC2DValidationError, match="sidecar_series_sha256_equals_summary_artifact"):
        reread_module.reread(protocol, results, reference_results=None, write=False, log=quiet)
    # a tampered assessment is refused by its hash sidecar
    results = _fake_experiment(tmp_path)
    path = results / "assessment.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PIC2DValidationError, match="SHA-256"):
        reread_module.reread(protocol, results, reference_results=None, write=False, log=quiet)
    # the sidecar's recorded reading must be the assessment's (b) value
    results = _fake_experiment(tmp_path)
    sidecar = json.loads((results / reread_module.SIDECAR_NAME).read_text(encoding="utf-8"))
    sidecar["end_state_window"]["recorded_ratio"] = -0.05
    artifacts.write_canonical_json(results / reread_module.SIDECAR_NAME, sidecar)
    with pytest.raises(PIC2DValidationError, match="sidecar_recorded_ratio_equals_assessment_b"):
        reread_module.reread(protocol, results, reference_results=None, write=False, log=quiet)
    # an already W-scaled record needs no re-read
    results = _fake_experiment(tmp_path, already_scaled=True)
    with pytest.raises(PIC2DValidationError, match="already_w_scaled"):
        reread_module.reread(protocol, results, reference_results=None, write=False, log=quiet)
    # a protocol on disk that is not the one the assessment binds
    results = _fake_experiment(tmp_path)
    (results.parent / "protocol.json").write_bytes((results.parent / "protocol.json").read_bytes() + b"\n")
    with pytest.raises(PIC2DValidationError, match="protocol.json on disk"):
        reread_module.reread(protocol, results, reference_results=None, write=False, log=quiet)
    # a missing sidecar
    results = _fake_experiment(tmp_path)
    (results / reread_module.SIDECAR_NAME).unlink()
    with pytest.raises(PIC2DValidationError, match="is missing"):
        reread_module.reread(protocol, results, reference_results=None, write=False, log=quiet)


def test_verdict_mapping_is_the_predeclared_tree() -> None:
    f = reread_module.verdict_from_outcomes
    assert f(True, True, True) == "converged" and f(True, True, False) == "resolution_limited"
    assert f(True, False, True) == "refinement_heating" and f(True, False, False) == "refinement_heating"
    assert f(False, True, True) == "no_plateau" and f(False, False, False) == "no_plateau"


COMMITTED = RESULTS / reread_module.OUTPUT_NAME


@pytest.mark.skipif(not COMMITTED.is_file() or not (RESULTS / reread_module.SIDECAR_NAME).is_file(), reason="ss-v4 corrected-ledger record not checked out")
def test_committed_re_read_reproduces_from_the_tracked_files() -> None:
    committed = artifacts.read_canonical_json(COMMITTED)
    record = reread_module.reread(v4.load_protocol(), RESULTS, write=False, log=lambda _: None)
    volatile = {"utc", "git_head_now"}
    assert {k: v for k, v in record.items() if k not in volatile} == {k: v for k, v in committed.items() if k not in volatile}
    for name, key in (("assessment.json", "assessment"), (reread_module.SIDECAR_NAME, "ledger_corrected"), ("summary.json", "summary")):
        assert committed["inputs"][key]["sha256"] == sha256((RESULTS / name).read_bytes()).hexdigest(), name
    assert committed["inputs"]["protocol"]["sha256"] == sha256((EXPERIMENT / "protocol.json").read_bytes()).hexdigest()
    assert committed["verdict_recorded"] == "resolution_limited" and committed["verdict_on_corrected_ledger"] == "refinement_heating"
    b = committed["b_residual_power"]
    assert b["recorded"]["windowed_residual_over_electrode_work"] == pytest.approx(-0.07667, abs=1e-4) and b["recorded"]["passed"] is True
    assert b["corrected"]["windowed_residual_over_electrode_work"] == pytest.approx(0.02459, abs=1e-4) and b["corrected"]["passed"] is False
    assert b["corrected"]["hard_gate_0p05_would_have_fired"] is False and b["corrected"]["first_checkpoint_at_or_above_bound"]["time_s"] == pytest.approx(4.816e-6)
    assert b["corrected"]["numerical_heating_power_w_in_window"] == pytest.approx(0.028, abs=2e-3) and b["corrected"]["electrode_power_w_in_window"] == pytest.approx(1.14, abs=0.01)
    assert committed["c_convergence"]["reference_corrected_ledger"]["corrected_windowed"] == pytest.approx(0.130, abs=2e-3)
    assert committed["c_convergence"]["reference_corrected_ledger"]["hard_gate_0p05_corrected_first_crossing_time_s"] == pytest.approx(2.70e-6, abs=2e-8)
    assert committed["verdict_statement"] == ("plateau reached; convergence vs 50 µm as recorded (resolution_limited for 50 µm); residual precondition (b) "
                                              "FAILED on the corrected ledger → the 33 µm plateau is itself heating at +2.5 % of electrode work and is NOT a "
                                              "clean reference; 25 µm (v5) pending")


def _reject_constant(value: str) -> Any:
    raise AssertionError(value)


@pytest.mark.skipif(not COMMITTED.is_file(), reason="ss-v4 corrected-ledger record not checked out")
def test_committed_re_read_is_canonical_with_a_hash_sidecar() -> None:
    data = COMMITTED.read_bytes()
    sidecar = json.loads(COMMITTED.with_name(COMMITTED.name + ".sha256.json").read_text(encoding="utf-8"))
    assert sidecar["byte_sha256"] == sha256(data).hexdigest()
    assert b"\r" not in data and json.loads(data.decode("utf-8"), parse_constant=_reject_constant)["schema_version"] == reread_module.SCHEMA
