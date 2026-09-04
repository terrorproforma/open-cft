"""Post-hoc re-read of the preregistered acceptance (a)-(d) of steady-state v4 on the CORRECTED energy ledger (model v2.0.6).

The recorded ``results/assessment.json`` (``run.py assess``, 08:21 UTC 2026-09-04) evaluated acceptance (b) on the run's
recorded windowed residual (-7.67 %).  Like every pre-v2.0.6 ledger that statistic lacked the macro weight W on the
inelastic sink (``spec/pic2d/pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6``); the sidecar
``results/ledger-corrected.json`` (``python -m cft_revival.pic2d.ledger_recompute``, commit 02013df0) carries the corrected
statistic (+2.46 % at the stop).  This module re-evaluates the SAME predeclared rules (``protocol.stopping_rule.acceptance``)
with (b) read from the sidecar and writes ``results/assessment-corrected-ledger.json`` (+ ``.sha256.json``), bound to the
byte hashes of the sidecar, the recorded assessment, ``summary.json`` and ``protocol.json``.  It carries BOTH readings:
the recorded verdict stands as recorded; the corrected re-read is a post-hoc statement about the same run.  Nothing
recorded is modified.

    python -m experiments.pic2d_cft_steady_state_v4.assess_corrected_ledger [--results DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.ledger_recompute import SIDECAR_NAME
from cft_revival.pic2d.models import PIC2DValidationError
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v4 import run as v4

HERE = Path(__file__).resolve().parent
OUTPUT_NAME = "assessment-corrected-ledger.json"
SCHEMA = "cft-revival.pic2d-cft-steady-state-v4.assessment-corrected-ledger/1.0.0"
GENERATED_BY = "python -m experiments.pic2d_cft_steady_state_v4.assess_corrected_ledger"
ACCEPTANCE_B_BOUND = 0.02
HARD_GATE = 0.05
RELATIVE_TOLERANCE = 1e-9


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _verified(path: Path) -> tuple[dict[str, Any], str]:
    """A canonical JSON record with its byte hash; the ``.sha256.json`` sidecar must exist and agree."""

    if not path.is_file():
        raise PIC2DValidationError(f"{path} is missing")
    sidecar = path.with_name(path.name + ".sha256.json")
    if not sidecar.is_file():
        raise PIC2DValidationError(f"{path.name}: hash sidecar {sidecar.name} is missing")
    digest = _file_sha256(path)
    recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("byte_sha256")
    if recorded != digest:
        raise PIC2DValidationError(f"{path.name}: sidecar SHA-256 mismatch")
    return json.loads(path.read_text(encoding="utf-8")), digest


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= RELATIVE_TOLERANCE * max(abs(float(a)), abs(float(b)), 1e-300)


def verdict_from_outcomes(a_plateau: bool, b_ok: bool, all_within: bool) -> str:
    """The predeclared (d) mapping of ``run.py assess``: (a) AND (b) AND (c) -> converged; (a) AND (b) -> resolution_limited;
    (a) -> refinement_heating; else no_plateau."""

    if a_plateau and b_ok and all_within:
        return "converged"
    if a_plateau and b_ok:
        return "resolution_limited"
    if a_plateau:
        return "refinement_heating"
    return "no_plateau"


def bind_sidecar_to_run(sidecar: Mapping[str, Any], summary: Mapping[str, Any], summary_sha: str, assessment: Mapping[str, Any],
                        experiment_id: str) -> dict[str, Any]:
    """Prove that ``ledger-corrected.json`` describes the run the recorded assessment assessed (fail closed)."""

    series_sha = summary["artifacts"]["series_npz_sha256"]
    checks = {
        "sidecar_series_sha256_equals_summary_artifact": sidecar["inputs"]["series"]["sha256"] == series_sha,
        "sidecar_summary_sha256_equals_summary_file": (sidecar["inputs"].get("summary") or {}).get("sha256") == summary_sha,
        "sidecar_experiment_id_equals_protocol": sidecar.get("experiment_id") == experiment_id,
        "sidecar_recorded_ratio_equals_summary_gate_reading": bool(sidecar["end_state_window"].get("recorded_ratio_matches_summary")),
        "sidecar_recorded_ratio_equals_assessment_b": _close(sidecar["end_state_window"]["recorded_ratio"],
                                                             assessment["b_residual_power"]["windowed_residual_over_electrode_work"]),
        "sidecar_last_step_equals_assessment_steps": int(sidecar["last_step"]) == int(assessment["run"]["steps_completed"]),
        "sidecar_is_not_already_w_scaled": not bool((sidecar.get("cross_check_vs_final_counts") or {}).get("already_w_scaled")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise PIC2DValidationError(f"{SIDECAR_NAME} does not describe the assessed run: {failed}")
    return checks


def reference_corrected_reading(reference_results: Path = v4.REFERENCE_RESULTS) -> dict[str, Any] | None:
    """The 50 um base's own corrected reading (its sidecar, bound to its summary) - the (c) reference was heating too."""

    sidecar_path = reference_results / SIDECAR_NAME
    summary_path = reference_results / "summary.json"
    if not sidecar_path.is_file() or not summary_path.is_file():
        return None
    sidecar, sidecar_sha = _verified(sidecar_path)
    summary, summary_sha = _verified(summary_path)
    if sidecar["inputs"]["series"]["sha256"] != summary["artifacts"]["series_npz_sha256"]:
        raise PIC2DValidationError("the 50 um base sidecar does not describe the base summary's series")
    end = sidecar["end_state_window"]
    gate = sidecar["threshold_crossings"]["0.05"]["corrected_first_crossing_at_checkpoint"]
    return {
        "results_dir": reference_results.name, "sidecar_sha256": sidecar_sha, "summary_sha256": summary_sha,
        "recorded_windowed": end["recorded_ratio"], "corrected_windowed": end["corrected_ratio"],
        "recorded_cumulative": sidecar["cumulative"]["recorded_over_electrode"], "corrected_cumulative": sidecar["cumulative"]["corrected_over_electrode"],
        "hard_gate_0p05_corrected_first_crossing_time_s": None if gate is None else gate["time_s"],
        "reading": (f"the 50 um base plateau reads {100.0 * end['corrected_ratio']:+.1f} % of the electrode work on the corrected ledger "
                    f"(recorded {100.0 * end['recorded_ratio']:+.1f} %): it was heating numerically; the v2.0.3 5 % gate would have stopped it at "
                    + ("never" if gate is None else f"{1e6 * gate['time_s']:.2f} us")),
    }


def reread(protocol: Mapping[str, Any], results: Path = v4.RESULTS, *, output: Path | None = None, write: bool = True,
           reference_results: Path | None = v4.REFERENCE_RESULTS, log: Callable[[str], None] = lambda text: print(text, flush=True)) -> dict[str, Any]:
    """Re-evaluate the predeclared acceptance with (b) on the corrected statistic; write the hash-bound post-hoc record."""

    protocol_path = results.parent / "protocol.json"
    protocol_sha = _file_sha256(protocol_path) if protocol_path.is_file() else None
    assessment, assessment_sha = _verified(results / "assessment.json")
    sidecar, sidecar_sha = _verified(results / SIDECAR_NAME)
    summary, summary_sha = _verified(results / "summary.json")
    experiment_id = protocol["experiment_id"]
    if assessment.get("experiment_id") != experiment_id or summary.get("experiment_id") != experiment_id:
        raise PIC2DValidationError("assessment.json / summary.json do not belong to this experiment")
    if protocol_sha is not None and assessment["run"].get("protocol_sha256") != protocol_sha:
        raise PIC2DValidationError("protocol.json on disk is not the protocol the recorded assessment binds")
    if sidecar.get("schema") != "cft.pic2d.ledger-corrected/1.0.0":
        raise PIC2DValidationError(f"{SIDECAR_NAME}: unexpected schema {sidecar.get('schema')!r}")
    binding = bind_sidecar_to_run(sidecar, summary, summary_sha, assessment, experiment_id)

    acceptance = protocol["stopping_rule"]["acceptance"]
    recorded_verdict = assessment["verdict"]
    a_plateau = bool(assessment["a_plateau"]["passed"])
    all_within = bool(assessment["c_convergence"]["all_within"])
    b_recorded = assessment["b_residual_power"]
    if verdict_from_outcomes(a_plateau, bool(b_recorded["passed"]), all_within) != recorded_verdict:
        raise PIC2DValidationError("the recorded assessment's verdict does not follow from its own (a)-(c) outcomes")
    end = sidecar["end_state_window"]
    corrected_windowed = end["corrected_ratio"]
    window_complete = bool(end["window_complete"])
    b_corrected_ok = corrected_windowed is not None and window_complete and float(corrected_windowed) < ACCEPTANCE_B_BOUND
    if bool(sidecar["acceptance_b_residual_power_below_0p02"]["corrected_passes"]) != b_corrected_ok:
        raise PIC2DValidationError("the sidecar's own (b) reading disagrees with the re-evaluation")
    crossing_2 = sidecar["threshold_crossings"]["0.02"]["corrected_first_crossing_at_checkpoint"]
    crossing_5 = sidecar["threshold_crossings"]["0.05"]["corrected_first_crossing_at_checkpoint"]
    verdict_corrected = verdict_from_outcomes(a_plateau, b_corrected_ok, all_within)
    reference = reference_corrected_reading(reference_results) if reference_results is not None else None
    if corrected_windowed is None:
        raise PIC2DValidationError(f"{SIDECAR_NAME}: no corrected end-state ratio (zero electrode work in the trailing window)")
    pct = 100.0 * float(corrected_windowed)
    window_seconds = float(end["window_steps"]) * float(summary["provenance"]["config"]["dt_s"])
    if b_corrected_ok:
        statement = (f"plateau reached; convergence vs 50 µm as recorded ({recorded_verdict} for 50 µm); residual precondition (b) holds on the "
                     f"corrected ledger ({pct:+.2f} % of electrode work < +2 %); 25 µm (v5) pending")
    else:
        statement = (f"plateau reached; convergence vs 50 µm as recorded ({recorded_verdict} for 50 µm); residual precondition (b) FAILED on the "
                     f"corrected ledger → the 33 µm plateau is itself heating at {pct:+.1f} % of electrode work and is NOT a clean reference; "
                     f"25 µm (v5) pending")
    record: dict[str, Any] = {
        "schema_version": SCHEMA, "utc": v4.utc_now(), "experiment_id": experiment_id, "results_dir": results.name,
        "kind": "post_hoc_re_read_on_the_corrected_energy_ledger",
        "generated_by": GENERATED_BY, "git_head_now": runner.git_head(),
        "model_version_note": ("pic2d model v2.0.6 energy-ledger correction (inelastic_loss_j lacked the macro weight W up to v2.0.5) applied post hoc "
                               "through the ledger-corrected.json sidecar; the recorded series, summary, maps and assessment.json are unchanged"),
        "inputs": {
            "assessment": {"file": "assessment.json", "sha256": assessment_sha, "utc": assessment.get("utc"), "verdict": recorded_verdict},
            "ledger_corrected": {"file": SIDECAR_NAME, "sha256": sidecar_sha, "schema": sidecar["schema"], "generated_by": sidecar.get("generated_by"),
                                 "series_sha256": sidecar["inputs"]["series"]["sha256"], "summary_sha256": sidecar["inputs"]["summary"]["sha256"],
                                 "macro_weight": sidecar["parameters"]["macro_weight"], "window_steps": sidecar["parameters"]["window_steps"]},
            "summary": {"file": "summary.json", "sha256": summary_sha},
            "protocol": {"file": "../protocol.json", "sha256": protocol_sha},
            "binding_checks": binding,
        },
        "recorded": {"verdict": recorded_verdict, "a_plateau": a_plateau, "b_residual_power_passed": bool(b_recorded["passed"]),
                     "b_windowed_residual_over_electrode_work": b_recorded["windowed_residual_over_electrode_work"],
                     "b_cumulative_witness": b_recorded["cumulative_witness"], "c_all_within": all_within,
                     "d_reclassification": assessment["d_reclassification"]},
        "a_plateau": {**assessment["a_plateau"], "unchanged": True,
                      "note": "not a ledger quantity: the plateau rule reads I_d, N_e and n_g drifts, the triad soft bounds and the peak-Debye soft margin"},
        "b_residual_power": {
            "rule": acceptance["b_residual_power"], "bound": ACCEPTANCE_B_BOUND, "one_sided": True,
            "recorded": {"windowed_residual_over_electrode_work": b_recorded["windowed_residual_over_electrode_work"],
                         "cumulative_witness": b_recorded["cumulative_witness"], "window_complete": b_recorded["window_complete"],
                         "passed": bool(b_recorded["passed"]), "statistic": "summary.grid_heating_triad.windowed_energy_residual_over_electrode_work (pre-v2.0.6: H - L_inel)"},
            "corrected": {"windowed_residual_over_electrode_work": corrected_windowed, "cumulative_witness": sidecar["cumulative"]["corrected_over_electrode"],
                          "window_complete": window_complete, "passed": b_corrected_ok,
                          "omitted_inelastic_over_electrode_work_in_window": end["omitted_ratio"],
                          "max_over_complete_windows": sidecar["max_over_complete_windows"]["corrected"],
                          "first_checkpoint_at_or_above_bound": crossing_2,
                          "hard_gate_0p05_would_have_fired": crossing_5 is not None, "hard_gate_0p05_first_crossing": crossing_5,
                          "numerical_heating_power_w_in_window": float(end["corrected_residual_j"]) / window_seconds,
                          "electrode_power_w_in_window": float(end["electrode_work_j"]) / window_seconds,
                          "statistic": "ledger-corrected.json end_state_window.corrected_ratio = H / electrode work over the trailing window, H = field work + dU - electrode work"},
            "passed": b_corrected_ok,
            "basis": "corrected statistic (the recorded statistic is kept beside it; the bound is NOT loosened)",
            "status_change": f"{'PASS' if b_recorded['passed'] else 'FAIL'} (recorded) -> {'PASS' if b_corrected_ok else 'FAIL'} (corrected)",
        },
        "c_convergence": {**assessment["c_convergence"], "unchanged": True,
                          "note": "not a ledger quantity: the (c) tolerances compare plateau quantities; the reference's own corrected reading is reported below",
                          "reference_corrected_ledger": reference},
        "d_reclassification": {
            "recorded_verdict": recorded_verdict, "recorded_text": assessment["d_reclassification"],
            "verdict_on_corrected_ledger": verdict_corrected, "corrected_text": acceptance["d_reclassification"][verdict_corrected],
            "mapping": "the predeclared (d) outcome tree of run.py assess applied with (b) on the corrected statistic; (a) and (c) as recorded",
            "what_stands": ("the recorded verdict stands as the recorded outcome of the preregistered execution; the 50 um base's classification as "
                            "resolution-limited is not rescued by the re-read (the base reads " + (f"{100.0 * reference['corrected_windowed']:+.1f} %" if reference else "n/a")
                            + " on the corrected ledger, more than the 33 um run); what changes is that the 33 um plateau carries its own numerical heating "
                            "power above the predeclared 2 % bound and may not be called a clean (energy-conserving) reference"),
        },
        "verdict_recorded": recorded_verdict,
        "verdict_on_corrected_ledger": verdict_corrected,
        "verdict_statement": statement,
        "disallowed_wording": ["converged (for 33 um or 50 um) before the 25 um ladder point reports", "energy-conserving (for the 33 um plateau)",
                               "any residual value recorded before v2.0.6 without the corrected value next to it"],
        "peak_debye_window": assessment["peak_debye_window"],
        "claim_boundary": protocol["claim_boundary"],
    }
    if write:
        artifacts.write_canonical_json(output or (results / OUTPUT_NAME), record)
    log(f"[assess-corrected-ledger] {results.name}: recorded {recorded_verdict} ((b) {100.0 * float(b_recorded['windowed_residual_over_electrode_work']):+.2f} % "
        f"{'pass' if b_recorded['passed'] else 'FAIL'}) -> corrected (b) {pct:+.2f} % {'pass' if b_corrected_ok else 'FAIL'}; (d) on the corrected ledger "
        f"{verdict_corrected}" + ("" if write else " (dry run, nothing written)"))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=None, help="results directory (default: the experiment's results/)")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = parser.parse_args(argv)
    protocol = v4.load_protocol()
    record = reread(protocol, v4.RESULTS if args.results is None else args.results, write=not args.dry_run)
    if args.dry_run:
        print(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True))     # console-safe on every code page
    return 0


if __name__ == "__main__":
    sys.exit(main())
