"""Post-hoc energy-ledger correction of recorded PIC-2D series (model v2.0.6, 2026-09-05).

Up to model v2.0.5 both backends added the MCC tally's per-MACRO-event threshold energy
``(n_exc E_exc + n_ion E_ion) e`` to ``cumulative["inelastic_loss_j"]`` WITHOUT the macro weight ``W``, while
every other ledger term carries ``W``.  The recorded interval residual was therefore

    residual_recorded = H - (W - 1) L_unscaled ~= H - L_inel,      H = field work + dU_field - electrode work,

where ``H`` is the true numerical energy creation of the field-particle coupling (the particle-side identity
``dKE = field work + injected - absorbed + born - W (n_exc E_exc + n_ion E_ion) e`` closes to round-off, see
``tests/pic2d/test_pic2d_v206_ledger.py``) and ``L_inel`` the W-scaled inelastic sink.  Every recorded residual in
the project is biased NEGATIVE by the inelastic power.

This module rebuilds the corrected residual from the RECORDED series without touching it: ``H`` needs only the
three series arrays ``interval_field_work_j``, ``field_energy_j`` and ``interval_electrode_work_j`` that every
``series.npz`` carries (no event counts required), so the correction is exact wherever the identity holds.  Where
the per-record event counts exist (``series.jsonl`` with ``ledger.cumulative``) the count-based correction
``recorded + (W - 1) dL_unscaled`` is computed as well and its agreement with ``H`` reported; the run's final
cumulative counts (``summary.json``) give a one-number cross-check of the omitted energy.  Records that start a
resumed session carry a zero recorded residual and zero electrode work by construction (no previous energy sample);
the corrected series keeps them at zero so the windowed statistic reads exactly what the runner's gate would have.

Output: a SIDECAR ``ledger-corrected.json`` (canonical JSON + ``.sha256.json``) next to the recorded files - the
recorded ``series.npz`` / ``summary.json`` are never modified - with the corrected windowed and cumulative
residual-power ratios, the end-state comparison, the first threshold crossings under both statistics (the v2.0.3
gate's 5 % stop and the 2 % plateau precondition / acceptance (b)) and the per-checkpoint trajectory.

    python -m cft_revival.pic2d.ledger_recompute <results-dir> [<results-dir> ...] [--dry-run] [--window-steps N]
                                                  [--checkpoint-steps N] [--macro-weight W] [--label TEXT]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_canonical_json

SCHEMA = "cft.pic2d.ledger-corrected/1.0.0"
SIDECAR_NAME = "ledger-corrected.json"
DEFAULT_WINDOW_STEPS = 400_000        # the v2.0.3 residual-power window
DEFAULT_CHECKPOINT_STEPS = 40_000     # the runner's gate-evaluation cadence
THRESHOLDS = (0.02, 0.05, 0.10)       # acceptance (b) / plateau precondition, the v2.0.3 hard stop, the v1.4 cumulative bound
PER_WEIGHT_KEY = "inelastic_loss_per_weight_j"   # present in the cumulative ledger from v2.0.6 on (already W-scaled records)

SERIES_KEYS = ("step", "time_s", "interval_residual_j", "interval_electrode_work_j", "interval_field_work_j", "field_energy_j")
COUNT_KEYS = ("excitations", "ionizations", "inelastic_loss_j")


class LedgerRecomputeError(RuntimeError):
    """A results directory that cannot be corrected (missing series, missing macro weight, malformed record)."""


# -- inputs ------------------------------------------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_series(results: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The series arrays from ``series.npz`` (preferred) or ``series.jsonl`` (untracked runs, withdrawn launches).

    Returns the arrays and a provenance block.  From ``series.jsonl`` the per-record cumulative counts are extracted
    too (``count_<key>``) when the records carry ``ledger.cumulative``.
    """

    npz = results / "series.npz"
    if npz.is_file():
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
        missing = [key for key in SERIES_KEYS if key not in arrays]
        if missing:
            raise LedgerRecomputeError(f"{npz}: series lacks {missing}")
        return arrays, {"file": npz.name, "sha256": _file_sha256(npz), "records": int(arrays["step"].size), "kind": "npz"}
    jsonl = results / "series.jsonl"
    if not jsonl.is_file():
        raise LedgerRecomputeError(f"{results}: neither series.npz nor series.jsonl found")
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise LedgerRecomputeError(f"{jsonl}: no records")
    columns: dict[str, list[float]] = {key: [] for key in SERIES_KEYS}
    columns["current_ionization_rate_per_s"] = []
    with_counts = all(isinstance(record.get("ledger", {}).get("cumulative"), dict) for record in records)
    if with_counts:
        for key in COUNT_KEYS + (PER_WEIGHT_KEY,):
            columns[f"count_{key}"] = []
    for record in records:
        ledger = record["ledger"]
        columns["step"].append(float(record["step"]))
        columns["time_s"].append(float(record["time_s"]))
        columns["field_energy_j"].append(float(record["field_energy_j"]))
        for key in ("interval_residual_j", "interval_electrode_work_j", "interval_field_work_j"):
            columns[key].append(float(ledger[key]))
        columns["current_ionization_rate_per_s"].append(float(record.get("currents_a", {}).get("ionization_rate_per_s", float("nan"))))
        if with_counts:
            cumulative = ledger["cumulative"]
            for key in COUNT_KEYS:
                columns[f"count_{key}"].append(float(cumulative.get(key, 0.0)))
            columns[f"count_{PER_WEIGHT_KEY}"].append(float(cumulative.get(PER_WEIGHT_KEY, float("nan"))))
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in columns.items()}
    return arrays, {"file": jsonl.name, "sha256": _file_sha256(jsonl), "records": len(records), "kind": "jsonl", "per_record_counts": with_counts}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dig(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def resolve_parameters(summary: dict[str, Any] | None, protocol: dict[str, Any] | None, *, macro_weight: float | None,
                       window_steps: int | None, checkpoint_steps: int | None) -> dict[str, Any]:
    """Macro weight, residual window and checkpoint cadence with their provenance (CLI > summary > protocol > default)."""

    out: dict[str, Any] = {}
    for name, value, candidates, default in (
        ("macro_weight", macro_weight,
         ((summary, ("case", "macro_weight")), (summary, ("provenance", "config", "macro_weight")), (protocol, ("case", "macro_weight"))), None),
        ("window_steps", window_steps,
         ((summary, ("grid_heating_triad", "thresholds", "residual_window_steps")),
          (protocol, ("stopping_rule", "grid_heating_triad", "residual_window_steps"))), DEFAULT_WINDOW_STEPS),
        ("checkpoint_steps", checkpoint_steps,
         ((protocol, ("numerics", "checkpoint_every_steps")), (summary, ("provenance", "numerics", "checkpoint_every_steps"))), DEFAULT_CHECKPOINT_STEPS),
    ):
        source = "command line"
        if value is None:
            for mapping, keys in candidates:
                found = _dig(mapping, *keys)
                if found is not None:
                    value, source = found, ("summary.json" if mapping is summary else "protocol.json") + "/" + "/".join(keys)
                    break
        if value is None:
            value, source = default, "default"
        if value is None:
            raise LedgerRecomputeError(f"{name} is not recorded (summary.json / protocol.json) - pass --{name.replace('_', '-')}")
        out[name] = (float(value) if name == "macro_weight" else int(value))
        out[f"{name}_source"] = source
    if out["macro_weight"] <= 0.0 or out["window_steps"] < 1 or out["checkpoint_steps"] < 1:
        raise LedgerRecomputeError("macro weight, window and checkpoint cadence must be positive")
    return out


# -- the correction ------------------------------------------------------------------------------------------------------

def corrected_residual(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``H_k = field work_k + (U_k - U_{k-1}) - electrode work_k`` per record; zero at record 0 and at resume-first
    records (recorded residual and electrode work both exactly zero).  Returns (corrected, H, resume_first_mask)."""

    residual = np.asarray(arrays["interval_residual_j"], dtype=np.float64)
    electrode = np.asarray(arrays["interval_electrode_work_j"], dtype=np.float64)
    work = np.asarray(arrays["interval_field_work_j"], dtype=np.float64)
    energy = np.asarray(arrays["field_energy_j"], dtype=np.float64)
    n = residual.size
    if n == 0:
        return residual.copy(), residual.copy(), np.zeros(0, dtype=bool)
    h = np.zeros(n)
    h[1:] = work[1:] + np.diff(energy) - electrode[1:]
    resume_first = np.zeros(n, dtype=bool)
    resume_first[1:] = (residual[1:] == 0.0) & (electrode[1:] == 0.0)
    corrected = np.where(resume_first, 0.0, h)
    corrected[0] = 0.0
    return corrected, h, resume_first


def windowed_ratios(steps: np.ndarray, residual: np.ndarray, electrode: np.ndarray, window_steps: int) -> dict[str, np.ndarray]:
    """The runner's ``windowed_energy_residual`` at EVERY record: records with ``step > step_k - window`` are inside;
    the window is complete when a record exists before it.  Vectorised with cumulative sums."""

    steps = np.asarray(steps, dtype=np.float64)
    cum_res = np.concatenate([[0.0], np.cumsum(residual)])
    cum_el = np.concatenate([[0.0], np.cumsum(electrode)])
    lo = np.searchsorted(steps, steps - float(window_steps), side="right")     # first index inside the window
    idx = np.arange(steps.size)
    res = cum_res[idx + 1] - cum_res[lo]
    el = cum_el[idx + 1] - cum_el[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(el > 0.0, res / np.where(el > 0.0, el, 1.0), np.nan)
    complete = lo > 0
    return {"residual_j": res, "electrode_j": el, "ratio": ratio, "complete": complete, "window_steps": steps - np.where(lo > 0, steps[np.maximum(lo - 1, 0)], steps[0])}


def _finite(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _first_crossing(steps: np.ndarray, time_s: np.ndarray, ratio: np.ndarray, complete: np.ndarray, threshold: float,
                    mask: np.ndarray | None = None) -> dict[str, Any] | None:
    fire = complete & np.isfinite(ratio) & (ratio >= threshold)
    if mask is not None:
        fire &= mask
    hits = np.flatnonzero(fire)
    if hits.size == 0:
        return None
    k = int(hits[0])
    return {"step": int(steps[k]), "time_s": float(time_s[k]), "ratio": float(ratio[k])}


def recompute(results: Path, *, macro_weight: float | None = None, window_steps: int | None = None,
              checkpoint_steps: int | None = None, label: str | None = None) -> dict[str, Any]:
    """Build the corrected-ledger record for one results directory (nothing is written here)."""

    results = Path(results)
    arrays, series_info = load_series(results)
    summary = _read_json(results / "summary.json")
    protocol = _read_json(results / "protocol.json")
    if protocol is None:
        protocol = _read_json(results.parent / "protocol.json")
        protocol_info = {"file": "../protocol.json"} if protocol is not None else None
    else:
        protocol_info = {"file": "protocol.json", "sha256": _file_sha256(results / "protocol.json")}
    parameters = resolve_parameters(summary, protocol, macro_weight=macro_weight, window_steps=window_steps, checkpoint_steps=checkpoint_steps)
    w = parameters["macro_weight"]
    window = parameters["window_steps"]
    cadence = parameters["checkpoint_steps"]

    steps = arrays["step"]
    time_s = arrays["time_s"]
    order = np.argsort(steps, kind="stable")
    if not np.array_equal(order, np.arange(steps.size)):
        raise LedgerRecomputeError(f"{results}: series steps are not monotonic")
    recorded = np.asarray(arrays["interval_residual_j"], dtype=np.float64)
    electrode = np.asarray(arrays["interval_electrode_work_j"], dtype=np.float64)
    corrected, h, resume_first = corrected_residual(arrays)
    omitted = corrected - recorded                                   # (W - 1) L_unscaled per record where the identity holds

    # -- already W-scaled records (v2.0.6+): the recorded residual IS H; the correction must then be ~0
    final_cumulative = _dig(summary, "final_series", "ledger", "cumulative")
    already_scaled = bool(final_cumulative is not None and PER_WEIGHT_KEY in final_cumulative) or bool(
        f"count_{PER_WEIGHT_KEY}" in arrays and np.isfinite(arrays[f"count_{PER_WEIGHT_KEY}"]).any())

    rec_w = windowed_ratios(steps, recorded, electrode, window)
    cor_w = windowed_ratios(steps, corrected, electrode, window)
    total_electrode = float(electrode[1:].sum())
    cum_rec = float(recorded[1:].sum())
    cum_cor = float(corrected[1:].sum())
    cum_rec_ratio = cum_rec / total_electrode if abs(total_electrode) > 0.0 else None
    cum_cor_ratio = cum_cor / total_electrode if abs(total_electrode) > 0.0 else None
    checkpoint_mask = (steps % cadence == 0)
    complete = cor_w["complete"]

    def _max_complete(ratio: np.ndarray) -> dict[str, Any] | None:
        ok = complete & np.isfinite(ratio)
        if not ok.any():
            return None
        k = int(np.flatnonzero(ok)[np.argmax(ratio[ok])])
        return {"ratio": float(ratio[k]), "step": int(steps[k]), "time_s": float(time_s[k])}

    thresholds: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        thresholds[key] = {
            "recorded_first_crossing_any_record": _first_crossing(steps, time_s, rec_w["ratio"], rec_w["complete"], threshold),
            "corrected_first_crossing_any_record": _first_crossing(steps, time_s, cor_w["ratio"], cor_w["complete"], threshold),
            "recorded_first_crossing_at_checkpoint": _first_crossing(steps, time_s, rec_w["ratio"], rec_w["complete"], threshold, checkpoint_mask),
            "corrected_first_crossing_at_checkpoint": _first_crossing(steps, time_s, cor_w["ratio"], cor_w["complete"], threshold, checkpoint_mask),
        }

    # -- cross-check against the run's final cumulative event counts (summary.json)
    cross_check: dict[str, Any] = {"available": False}
    if final_cumulative is not None and "inelastic_loss_j" in final_cumulative:
        recorded_loss = float(final_cumulative["inelastic_loss_j"])
        first_loss = 0.0
        if "count_inelastic_loss_j" in arrays:
            first_loss = float(arrays["count_inelastic_loss_j"][0])
        # the series' pairwise sums start at record 1: the loss accumulated before record 0 is not in any interval
        expected_unscaled = recorded_loss - first_loss
        expected_omitted = 0.0 if already_scaled else (w - 1.0) * expected_unscaled
        omitted_sum = float(omitted[~resume_first].sum())
        excluded = int(resume_first.sum())
        denominator = max(abs(expected_omitted), abs(omitted_sum), 1e-300)
        relative = (omitted_sum - expected_omitted) / denominator
        # the MCC removes the thresholds in the CLASSICAL electron energy while the ledger is relativistic: the count-based
        # omitted energy and sum(H - recorded) differ by O(v^2/c^2) of the colliding electrons times the loss - 7e-5..9e-5 on
        # the ignited discharges (collisions at 10-30 eV), 1.2e-3 on the no-ignition plume attempt 3 (cathode electrons at
        # hundreds of eV); anything beyond 5e-3 is not this effect
        if already_scaled:
            verdict = "already W-scaled record (v2.0.6+): corrected == recorded expected"
        elif excluded == 0:
            verdict = ("exact (no resumes): sum(H - recorded) equals (W - 1) x the recorded unscaled loss up to the classical-vs-"
                       "relativistic threshold bookkeeping (O(v^2/c^2) of the colliding electrons, 1e-4..1e-3 of the loss)"
                       + ("" if abs(relative) <= 5e-3 else " - MISMATCH beyond 5e-3: inspect"))
        else:
            verdict = (f"approximate: {excluded} resume-first record(s) carry no residual, their intervals' inelastic loss "
                       f"is in the cumulative count but not in the series sum")
        cross_check = {
            "available": True, "already_w_scaled": already_scaled,
            "summary_cumulative_inelastic_loss_j": recorded_loss,
            "summary_cumulative_excitations": _finite(final_cumulative.get("excitations")),
            "summary_cumulative_ionizations": _finite(final_cumulative.get("ionizations")),
            "expected_omitted_j_from_counts": expected_omitted, "omitted_j_from_series": omitted_sum,
            "relative_difference": float(relative), "resume_first_records_excluded": excluded, "verdict": verdict,
        }

    # -- exact per-record count-based correction where the counts exist (series.jsonl)
    count_based: dict[str, Any] | None = None
    if "count_inelastic_loss_j" in arrays and not already_scaled:
        d_loss = np.zeros_like(recorded)
        d_loss[1:] = np.diff(arrays["count_inelastic_loss_j"])
        by_counts = recorded + (w - 1.0) * d_loss
        by_counts = np.where(resume_first, 0.0, by_counts)
        by_counts[0] = 0.0
        gap = np.abs(by_counts - corrected)
        count_based = {
            "max_abs_difference_to_h_j": float(gap.max()) if gap.size else 0.0,
            "max_abs_difference_over_max_abs_h": float(gap.max() / max(np.abs(h).max(), 1e-300)) if gap.size else 0.0,
            "sum_by_counts_j": float(by_counts[1:].sum()), "sum_by_h_j": cum_cor,
        }

    recorded_summary_reading = _finite(_dig(summary, "grid_heating_triad", "windowed_energy_residual_over_electrode_work"))
    end_rec = _finite(rec_w["ratio"][-1])
    end_cor = _finite(cor_w["ratio"][-1])
    acceptance_b = _dig(protocol, "stopping_rule", "acceptance", "b_residual_power")
    trajectory = [
        {
            "step": int(steps[k]), "time_s": float(time_s[k]), "window_complete": bool(complete[k]),
            "recorded_windowed": _finite(rec_w["ratio"][k]), "corrected_windowed": _finite(cor_w["ratio"][k]),
            "window_electrode_work_j": float(cor_w["electrode_j"][k]),
        }
        for k in np.flatnonzero(checkpoint_mask | (np.arange(steps.size) == steps.size - 1))
    ]
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "model_version": "pic2d v2.0.6 energy-ledger correction, applied post hoc; recorded values unchanged",
        "generated_by": "python -m cft_revival.pic2d.ledger_recompute",
        "results_dir": results.name,
        "label": label or results.name,
        "experiment_id": _dig(summary, "experiment_id"),
        "stop_reason": _dig(summary, "stop_reason"),
        "inputs": {"series": series_info, "summary": ({"file": "summary.json", "sha256": _file_sha256(results / "summary.json")}
                                                       if summary is not None else None), "protocol": protocol_info},
        "parameters": parameters,
        "method": {
            "corrected_residual": "H_k = interval_field_work_j_k + (field_energy_j_k - field_energy_j_{k-1}) - interval_electrode_work_j_k; "
                                  "0 at record 0 and at resume-first records (recorded residual and electrode work both exactly 0)",
            "identity": "dKE = field work + injected - absorbed + born - W (n_exc E_exc + n_ion E_ion) e closes to round-off, so the "
                        "residual with the W-scaled sink equals H; the recorded (pre-v2.0.6) residual was H - (W - 1) L_unscaled",
            "windowed_statistic": "runner windowed_energy_residual: sum over records with step > last - window_steps of the residual "
                                  "over the electrode work; complete when a record exists before the window; evaluated here at every "
                                  "record, the runner evaluates it at checkpoints",
        },
        "records": int(steps.size), "resume_first_records": int(resume_first.sum()),
        "first_step": int(steps[0]), "last_step": int(steps[-1]), "last_time_s": float(time_s[-1]),
        "cumulative": {
            "electrode_work_j": total_electrode, "recorded_residual_j": cum_rec, "corrected_residual_j": cum_cor,
            "omitted_inelastic_j": cum_cor - cum_rec,
            "recorded_over_electrode": _finite(cum_rec_ratio), "corrected_over_electrode": _finite(cum_cor_ratio),
            "omitted_over_electrode": _finite((cum_cor - cum_rec) / total_electrode) if abs(total_electrode) > 0.0 else None,
        },
        "end_state_window": {
            "step": int(steps[-1]), "time_s": float(time_s[-1]), "window_steps": int(cor_w["window_steps"][-1]),
            "window_complete": bool(complete[-1]), "electrode_work_j": float(cor_w["electrode_j"][-1]),
            "recorded_residual_j": float(rec_w["residual_j"][-1]), "corrected_residual_j": float(cor_w["residual_j"][-1]),
            "recorded_ratio": end_rec, "corrected_ratio": end_cor,
            "omitted_ratio": (end_cor - end_rec) if end_rec is not None and end_cor is not None else None,
            "recorded_ratio_in_summary": recorded_summary_reading,
            "recorded_ratio_matches_summary": (None if recorded_summary_reading is None or end_rec is None
                                               else bool(abs(recorded_summary_reading - end_rec) <= 1e-9 * max(1.0, abs(end_rec)))),
        },
        "max_over_complete_windows": {"recorded": _max_complete(rec_w["ratio"]), "corrected": _max_complete(cor_w["ratio"])},
        "threshold_crossings": thresholds,
        "acceptance_b_residual_power_below_0p02": {
            "declared_in_protocol": acceptance_b is not None,
            "recorded_passes": None if end_rec is None else bool(end_rec < 0.02),
            "corrected_passes": None if end_cor is None else bool(end_cor < 0.02),
        },
        "v2_0_3_hard_gate_0p05": {
            "recorded_would_have_fired": thresholds["0.05"]["recorded_first_crossing_at_checkpoint"] is not None,
            "corrected_would_have_fired": thresholds["0.05"]["corrected_first_crossing_at_checkpoint"] is not None,
        },
        "cross_check_vs_final_counts": cross_check,
        "count_based_correction": count_based,
        "trajectory_at_checkpoints": trajectory,
    }
    return record


def write_sidecar(results: Path, record: Mapping[str, Any]) -> Path:
    target = Path(results) / SIDECAR_NAME
    write_canonical_json(target, record)
    return target


# -- reporting -----------------------------------------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.1f} %"


def _when(entry: Mapping[str, Any] | None) -> str:
    return "never" if entry is None else f"{1e6 * entry['time_s']:.2f} us"


def table_rows(records: list[Mapping[str, Any]]) -> list[str]:
    """Markdown table: recorded vs corrected windowed residual at the end state, cumulative, the 5 % gate, acceptance (b)."""

    header = ("| run | end step / time | recorded windowed | corrected windowed | recorded cumulative | corrected cumulative "
              "| 5 % gate fires (recorded -> corrected) | (b) < +2 % recorded -> corrected |")
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for record in records:
        end = record["end_state_window"]
        cum = record["cumulative"]
        gate = record["threshold_crossings"]["0.05"]
        accept = record["acceptance_b_residual_power_below_0p02"]

        def passes(value: bool | None) -> str:
            return "n/a" if value is None else ("pass" if value else "FAIL")

        lines.append(
            f"| {record['label']} | {record['last_step']:,} / {1e6 * record['last_time_s']:.3f} us | {_pct(end['recorded_ratio'])} | "
            f"{_pct(end['corrected_ratio'])} | {_pct(cum['recorded_over_electrode'])} | {_pct(cum['corrected_over_electrode'])} | "
            f"{_when(gate['recorded_first_crossing_at_checkpoint'])} -> {_when(gate['corrected_first_crossing_at_checkpoint'])} | "
            f"{passes(accept['recorded_passes'])} -> {passes(accept['corrected_passes'])} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="results directories (series.npz or series.jsonl inside)")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write no sidecar")
    parser.add_argument("--window-steps", type=int, default=None, help="residual window in steps (default: the run's declared window, else 400000)")
    parser.add_argument("--checkpoint-steps", type=int, default=None, help="gate-evaluation cadence in steps (default: the protocol's checkpoint cadence, else 40000)")
    parser.add_argument("--macro-weight", type=float, default=None, help="macro weight W when summary.json / protocol.json do not record it")
    parser.add_argument("--label", action="append", default=None, help="label per results directory (repeat, in order)")
    parser.add_argument("--json", action="store_true", help="print the full record(s) as JSON instead of the table")
    args = parser.parse_args(argv)
    labels = args.label or []
    records = []
    for index, results in enumerate(args.results):
        record = recompute(results, macro_weight=args.macro_weight, window_steps=args.window_steps, checkpoint_steps=args.checkpoint_steps,
                           label=labels[index] if index < len(labels) else None)
        if not args.dry_run:
            write_sidecar(results, record)
        records.append(record)
    if args.json:
        print(json.dumps(records, indent=1, sort_keys=True))
    else:
        print("\n".join(table_rows(records)))
        for record in records:
            check = record["cross_check_vs_final_counts"]
            if check.get("available"):
                print(f"{record['label']}: cross-check {check['verdict']}; relative difference {check['relative_difference']:+.3e}")
    return 0


__all__ = [
    "DEFAULT_CHECKPOINT_STEPS",
    "DEFAULT_WINDOW_STEPS",
    "SCHEMA",
    "SIDECAR_NAME",
    "THRESHOLDS",
    "LedgerRecomputeError",
    "corrected_residual",
    "load_series",
    "main",
    "recompute",
    "resolve_parameters",
    "table_rows",
    "windowed_ratios",
    "write_sidecar",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
