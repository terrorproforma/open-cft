"""INTERIM development visualisation of the PIC design mini-sweep while its runs execute.

Nothing produced here is a result.  The panels show the LATEST recorded frame of runs that have not
reached a plateau; every output carries the label "INTERIM Â· development Â· t = â€¦ Âµs Â· not a plateau".
Never writes into a run's results directory.

Three steps, each re-runnable while the runs continue (frames are complete files written by an
atomic replace; a frame that fails to load is skipped, a gap ends the staged sequence):

  manifest  read the scheduler's ``jobs/<id>/state.json`` (checkout, results dir, transit time),
            the run's ``results/protocol.json`` (design id, grid, W, dt, geometry) and the design's
            field binding (``topology_under_iron.wall_cusps`` under iron; the v3.1 catalogue for the
            P2 reference) -> ``runs.json`` with the design's OWN cusp planes, rho and r_w/L
  stage     mirror the loadable, contiguous frames of a running job (symlinks; copies where symlinks
            are unavailable) next to a SYNTHESISED ``summary.json`` (grid / W / dt from the protocol)
            so that the v0.2 renderer's readers (``grid_from_summary``, ``run_constants``,
            ``build_player_payload``) work on a run that has not written its final artifacts
  render    per design: the five v0.2 panels as videos with an INTERIM banner + the HTML player;
            across designs: the 4 x 3 comparison PNG (ion density log, potential, windowed
            ionisation rate with the v0.2 window / mask / percentile treatment) at each run's latest
            frame with colour scales shared per column (declared in the figure and in the JSON
            report), and the 4-axis time-series strip (I_d, S, N_e, peak Delta/lambda_D) from
            ``status.jsonl``; plus a status table (markdown + JSON)

Usage (from ``modern/`` with ``PYTHONPATH=src:.``)::

    python visualization/interim_sweep_panel.py manifest --jobs-dir /work/jobs \
        --job sweep-047 --job sweep-reference --job sweep-009 --job sweep-056 --out /work/interim/runs.json
    python visualization/interim_sweep_panel.py status --manifest /work/interim/runs.json
    python visualization/interim_sweep_panel.py render --manifest /work/interim/runs.json --out-dir /work/interim
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
for _entry in (str(MODERN / "src"), str(MODERN), str(HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import render_pic2d_video as video

from cft_revival.pic2d.frames import (
    FRAME_DIRNAME,
    FRAME_PATTERN,
    MAP_KEYS,
    FrameSet,
    load_frames,
)
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import Grid2D

SCHEMA = "cft-pic2d-interim-sweep-panel/0.1.0"
INTERIM_STATUS = "development_screening_not_preregistered"   # the only status validate_player_payload accepts
INTERIM_TAG = "INTERIM \u00b7 development"
NOT_PLATEAU = "not a plateau"
INTERIM_CLAIM = (
    "INTERIM DEVELOPMENT VISUALISATION of a run that is still executing: the frames so far of a preregistered "
    "design mini-sweep run, read while it runs. Not a plateau, not an assessment, not a result; the run's own "
    "status is set by its assessment stage, never by this rendering. summary.json here is SYNTHESISED from the "
    "run's protocol.json (grid, macro weight, time step) because the run has not written its final artifacts."
)
REFERENCE_DESIGN_ID = "divergent-exit-stack"
DEFAULT_JOBS = ("sweep-047", "sweep-reference", "sweep-009", "sweep-056")
FIELDS_DIR = MODERN / "experiments" / "pic2d_design_mini_sweep_v1" / "fields"
BOUNDARY_TOLERANCE_M = 2.5e-4              # v3.1 boundary-ambiguity tolerance: cusps this close to the ends are boundary cusps
PANEL_MAPS = ("n_i_per_m3", "phi_v", video.IZ_KEY)
FIGURE_WIDTH = 2400
RADIAL_EXAGGERATION = 2.0
NI_DECADES = 3.0
NI_TOP_PERCENTILE = 99.9
GATE_HARD = math.pi
GATE_SOFT = 2.5
DESIGN_COLOURS = ((88, 168, 255), (232, 236, 239), (90, 214, 192), (255, 107, 107), (200, 160, 255), (255, 207, 103))
BG = (17, 20, 23)
PANEL_BG = (24, 28, 32)
TEXT = (232, 236, 239)
MUTED = (155, 184, 176)
CUSP_RGB = (255, 207, 103)
BOUNDARY_CUSP_RGB = (255, 140, 60)
BANNER_BG = (74, 44, 12)
BANNER_FG = (255, 221, 130)


def interim_title(t_us: float) -> str:
    return f"{INTERIM_TAG} \u00b7 t = {t_us:.3f} \u00b5s \u00b7 {NOT_PLATEAU}"


def short_label(design_id: str) -> str:
    if design_id == REFERENCE_DESIGN_ID:
        return "ref"
    match = re.search(r"-(\d{3})-", design_id)
    return match.group(1) if match else design_id[:8]


# -- manifest -----------------------------------------------------------------------------------------------


@dataclass
class RunSpec:
    """One run of the sweep as the panel sees it (all paths absolute; cusps are the design's own planes)."""

    job_id: str
    design_id: str
    label: str
    results: str
    rho: float | None = None
    rho_source: str = ""
    r_w_over_l: float | None = None
    cusp_z_m: list[float] = field(default_factory=list)
    boundary_cusp_z_m: list[float] = field(default_factory=list)
    cusp_source: str = ""
    transit_time_s: float | None = None
    target_transits: float = 3.0
    pid: int | None = None
    note: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RunSpec:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


def design_cusps_and_rho(design_id: str, fields_dir: Path = FIELDS_DIR) -> dict[str, Any]:
    """The design's own cusp planes and Koch rho: field binding under iron, else the sealed catalogue entry."""

    binding_path = fields_dir / design_id / "binding.json"
    if binding_path.is_file():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        topology = binding.get("topology_under_iron")
        if topology:
            rho = topology.get("min_rho_conservative_interior", topology.get("min_rho_conservative"))
            return {"cusp_z_m": [float(c["z_c_m"]) for c in topology["wall_cusps"]], "rho": None if rho is None else float(rho),
                    "source": f"{binding_path.name}: topology_under_iron (material-aware level-0 P2)"}
    from experiments.pic2d_design_mini_sweep_v1 import (
        designs,  # sealed catalogue lookups (the reference has no iron solve)
    )

    entry = designs.catalogue_entry(design_id)
    rows = designs.rho_conservative_from_entry(entry)
    return {"cusp_z_m": [float(c["z_c_m"]) for c in entry["wall_cusps"]], "rho": float(min(r["rho_conservative"] for r in rows)),
            "source": "catalogue entry (cusp topology v3.1 / sweep v3)"}


def design_r_w_over_l(design_id: str) -> float | None:
    try:
        from experiments.pic2d_design_mini_sweep_v1 import designs

        return float(designs.design_summary(design_id)["wall_radius_over_pitch"])
    except Exception:  # noqa: BLE001 - r_w/L is an annotation; the panel must not depend on the geometry rebuild
        return None


def classify_boundary_cusps(cusps: Sequence[float], z_min_m: float, z_max_m: float, tolerance_m: float = BOUNDARY_TOLERANCE_M) -> list[float]:
    return [float(z) for z in cusps if z < z_min_m + tolerance_m or z > z_max_m - tolerance_m]


def build_manifest(jobs_dir: Path, job_ids: Sequence[str], *, fields_dir: Path = FIELDS_DIR, log=print) -> dict[str, Any]:
    runs = []
    for job_id in job_ids:
        state = json.loads((jobs_dir / job_id / "state.json").read_text(encoding="utf-8"))
        results = Path(state["checkout"]) / state["results"]
        protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
        design_id = str(protocol["design_id"])
        geometry = protocol["geometry"]
        found = design_cusps_and_rho(design_id, fields_dir)
        spec = RunSpec(
            job_id=job_id, design_id=design_id, label=short_label(design_id), results=str(results), rho=found["rho"],
            rho_source=found["source"], r_w_over_l=design_r_w_over_l(design_id), cusp_z_m=found["cusp_z_m"],
            boundary_cusp_z_m=classify_boundary_cusps(found["cusp_z_m"], float(geometry["z_min_m"]), float(geometry["z_max_m"])),
            cusp_source=found["source"], transit_time_s=state.get("transit_time_s"), target_transits=float(state.get("target_transits") or 3.0),
            pid=state.get("pid"), note=str(protocol.get("design", {}).get("note", "")),
        )
        log(f"[manifest] {job_id}: {design_id} rho {spec.rho} r_w/L {spec.r_w_over_l} cusps {[round(z * 1e3, 3) for z in spec.cusp_z_m]} mm")
        runs.append(asdict(spec))
    runs.sort(key=lambda r: (r["rho"] if r["rho"] is not None else math.inf, r["design_id"]))
    return {"schema": SCHEMA, "kind": "interim-sweep-manifest", "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "jobs_dir": str(jobs_dir),
            "runs": runs}


def load_manifest(path: Path) -> list[RunSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [RunSpec.from_mapping(r) for r in raw["runs"]]
    runs.sort(key=lambda r: (r.rho if r.rho is not None else math.inf, r.design_id))
    return runs


# -- staging (mirror + synthesised summary) ---------------------------------------------------------------------


def synthesise_summary(protocol: Mapping[str, Any], *, frames_count: int, run_state: Mapping[str, Any] | None = None,
                       protocol_sha256: str | None = None) -> dict[str, Any]:
    """A summary.json stand-in carrying exactly what the v0.2 renderer reads (grid, W, dt, status, claim boundary)."""

    case, numerics, geo = protocol["case"], protocol["numerics"], protocol["geometry"]
    geometry = {key: geo.get(key) for key in ("bore_radius_m", "z_min_m", "z_max_m", "cone_start_z_m", "exit_radius_m",
                                                "plume_radius_m", "plume_length_m", "body_dielectric_radius_m")}
    config = {"grid": {"geometry": geometry, "radial_cells": int(case["radial_cells"]), "axial_cells": int(case["axial_cells"])},
              "macro_weight": float(case["macro_weight"]), "dt_s": float(numerics["dt_s"])}
    return {
        "schema": SCHEMA, "kind": "synthesised-interim-summary", "synthesised": True,
        "synthesised_from": "results/protocol.json (case, numerics, geometry) and run_state.json; the run has not written summary.json",
        "experiment_id": protocol.get("experiment_id"), "design_id": protocol.get("design_id"), "model_version": protocol.get("model_version"),
        "run_protocol_status": protocol.get("status"), "status": INTERIM_STATUS, "claim_boundary": INTERIM_CLAIM,
        "protocol_sha256": protocol_sha256, "provenance": {"config": config}, "run_state": dict(run_state) if run_state else None,
        "artifacts": {"frames": {"count": int(frames_count), "sha256": None, "note": "frames of a running job, mirrored; no manifest hash"}},
    }


def loadable_frames(results: Path) -> tuple[list[tuple[Path, int, int]], list[dict[str, str]], list[str]]:
    """(contiguous loadable frames, skipped files with the error, files dropped after a gap)."""

    folder = Path(results) / FRAME_DIRNAME
    files = sorted(p for p in folder.iterdir() if FRAME_PATTERN.match(p.name)) if folder.is_dir() else []
    good: list[tuple[Path, int, int]] = []
    skipped: list[dict[str, str]] = []
    for path in files:
        try:
            with np.load(path) as data:
                start, end = int(data["start_step"][0]), int(data["end_step"][0])
                for key in MAP_KEYS:
                    _ = data[key].shape
                _ = str(data["scalars_json"][0])
        except Exception as error:  # noqa: BLE001 - a frame mid-write (or truncated) is skipped, never fatal
            skipped.append({"file": path.name, "error": f"{type(error).__name__}: {error}"})
            continue
        good.append((path, start, end))
    kept: list[tuple[Path, int, int]] = []
    dropped: list[str] = []
    for item in good:
        if kept and kept[-1][2] != item[1]:
            dropped.append(item[0].name)
            continue
        if dropped:
            dropped.append(item[0].name)
            continue
        kept.append(item)
    return kept, skipped, dropped


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.symlink(source.resolve(), target)
        return "symlink"
    except (OSError, NotImplementedError):
        shutil.copy2(source, target)
        return "copy"


def stage_mirror(results: Path, mirror: Path, *, log=print) -> dict[str, Any]:
    """Mirror a running job's frames + a synthesised summary.json under ``mirror`` (the results dir is only read)."""

    results, mirror = Path(results), Path(mirror)
    frames_dir = mirror / FRAME_DIRNAME
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.iterdir():
        old.unlink()
    kept, skipped, dropped = loadable_frames(results)
    modes = {_link_or_copy(path, frames_dir / path.name) for path, _, _ in kept}
    protocol_path = results / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    run_state_path = results / "run_state.json"
    run_state = None
    if run_state_path.is_file():
        try:
            run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_state = None
    summary = synthesise_summary(protocol, frames_count=len(kept), run_state=run_state, protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest())
    (mirror / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    report = {"results": str(results), "mirror": str(mirror), "frames_staged": len(kept), "link_mode": sorted(modes),
              "frames_skipped": skipped, "frames_dropped_after_gap": dropped,
              "latest_end_step": kept[-1][2] if kept else None, "staged_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    log(f"[stage] {results.name}: {len(kept)} frames staged ({', '.join(sorted(modes)) or 'none'}), {len(skipped)} skipped, {len(dropped)} dropped")
    return report


# -- status series ---------------------------------------------------------------------------------------------

STATUS_SCALARS = ("time_s", "step", "discharge_a", "ionization_rate_per_s", "electrons", "ions", "n_e_peak_node_per_m3", "n_g_per_m3",
                  "ms_per_step", "t_e_mean_ev", "wall_seconds_total")


def _get(record: Mapping[str, Any], *path: str) -> float:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return math.nan
        value = value[key]
    try:
        return math.nan if value is None else float(value)
    except (TypeError, ValueError):
        return math.nan


def load_status_series(results: Path) -> dict[str, Any]:
    """status.jsonl -> arrays (bad / partial lines skipped) plus the last complete record."""

    path = Path(results) / "status.jsonl"
    records: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_bytes().split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, Mapping) and record.get("step") is not None:
                records.append(dict(record))
    series: dict[str, Any] = {key: np.array([_get(r, key) for r in records], dtype=np.float64) for key in STATUS_SCALARS}
    series["peak_cells_per_debye"] = np.array([_get(r, "peak_node", "cells_per_debye") for r in records])
    series["window_cells_per_debye"] = np.array([_get(r, "peak_node", "window", "cells_per_debye") for r in records])
    series["peak_node_n_e_per_m3"] = np.array([_get(r, "peak_node", "n_e_peak_per_m3") for r in records])
    series["windowed_residual"] = np.array([_get(r, "grid_heating_triad", "windowed_energy_residual_over_electrode_work") for r in records])
    series["records"] = len(records)
    series["last"] = records[-1] if records else None
    return series


def status_row(spec: RunSpec, *, frames_staged: int | None = None, series: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One row of the status table from the last status record, run_state and the frame count."""

    results = Path(spec.results)
    series = series if series is not None else load_status_series(results)
    last = series.get("last") or {}
    run_state: dict[str, Any] = {}
    if (results / "run_state.json").is_file():
        try:
            run_state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_state = {}
    ms = series["ms_per_step"][-20:] if series.get("records") else np.array([])
    ms = ms[np.isfinite(ms)]
    time_s = _get(last, "time_s")
    triad = last.get("grid_heating_triad") or {}
    return {
        "job_id": spec.job_id, "design_id": spec.design_id, "label": spec.label, "rho": spec.rho, "r_w_over_l": spec.r_w_over_l,
        "step": last.get("step"), "time_us": None if math.isnan(time_s) else time_s * 1e6,
        "transits": None if math.isnan(time_s) or not spec.transit_time_s else time_s / float(spec.transit_time_s),
        "target_transits": spec.target_transits, "ms_per_step_median20": float(np.median(ms)) if ms.size else None,
        "discharge_ma": _get(last, "discharge_a") * 1e3, "ionization_rate_per_s": _get(last, "ionization_rate_per_s"),
        "electrons": _get(last, "electrons"), "n_g_per_m3": _get(last, "n_g_per_m3"), "n_e_peak_node_per_m3": _get(last, "n_e_peak_node_per_m3"),
        "peak_node_n_e_per_m3": _get(last, "peak_node", "n_e_peak_per_m3"), "peak_cells_per_debye": _get(last, "peak_node", "cells_per_debye"),
        "window_cells_per_debye": _get(last, "peak_node", "window", "cells_per_debye"),
        "window_complete": _get(last, "peak_node", "window", "window_complete") == 1.0,
        "windowed_residual_over_electrode_work": _get(last, "grid_heating_triad", "windowed_energy_residual_over_electrode_work"),
        "windowed_residual_window_complete": bool(triad.get("windowed_energy_residual_window_complete", False)),
        "triad_soft_ok": triad.get("soft_ok"), "triad_enforced": triad.get("enforced"), "triad_hard_failures": list(triad.get("hard_failures") or []),
        "plateau_reached": (last.get("plateau") or {}).get("reached"), "t_e_mean_ev": _get(last, "t_e_mean_ev"),
        "frames_written": run_state.get("frames_written"), "frames_staged": frames_staged, "finished": run_state.get("finished"),
        "stop_reason": run_state.get("stop_reason"), "wall_seconds_total": _get(last, "wall_seconds_total"), "status_records": series.get("records", 0),
    }


def _fmt(value: Any, spec: str = ".3g", none: str = "-") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return none
    return format(value, spec)


def format_status_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = ["design", "rho", "r_w/L", "t_sim (us)", "transits", "ms/step", "I_d (mA)", "S (1e16/s)", "N_e (M)", "n_g (1e19)",
              "peak n_e (m^-3)", "Delta/lambda_D step|win", "residual_w", "triad soft_ok (enforced)", "frames"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        s = r.get("ionization_rate_per_s")
        lines.append("| " + " | ".join([
            f"{r['label']} {r['design_id']}", _fmt(r.get("rho"), ".2f"), _fmt(r.get("r_w_over_l"), ".2f"), _fmt(r.get("time_us"), ".3f"),
            f"{_fmt(r.get('transits'), '.3f')}/{r.get('target_transits', 3):g}", _fmt(r.get("ms_per_step_median20"), ".2f"),
            _fmt(r.get("discharge_ma"), ".2f"), _fmt(None if s is None or not math.isfinite(s) else s / 1e16, ".2f"),
            _fmt(None if r.get("electrons") is None or not math.isfinite(r["electrons"]) else r["electrons"] / 1e6, ".2f"),
            _fmt(None if r.get("n_g_per_m3") is None or not math.isfinite(r["n_g_per_m3"]) else r["n_g_per_m3"] / 1e19, ".2f"),
            _fmt(r.get("n_e_peak_node_per_m3"), ".2e"),
            f"{_fmt(r.get('peak_cells_per_debye'), '.2f')} | {_fmt(r.get('window_cells_per_debye'), '.2f')}",
            f"{_fmt(r.get('windowed_residual_over_electrode_work'), '+.1%')}{'' if r.get('windowed_residual_window_complete') else ' (partial)'}",
            f"{r.get('triad_soft_ok')} ({'enforced' if r.get('triad_enforced') else 'unenforced'})"
            + (f" HARD {r['triad_hard_failures']}" if r.get("triad_hard_failures") else ""),
            f"{r.get('frames_staged') if r.get('frames_staged') is not None else '-'}/{r.get('frames_written') or '-'}",
        ]) + " |")
    return "\n".join(lines) + "\n"


# -- run data -------------------------------------------------------------------------------------------------


@dataclass
class RunData:
    spec: RunSpec
    mirror: Path
    stage: dict[str, Any]
    frames: FrameSet
    summary: dict[str, Any]
    grid: Grid2D
    ionisation: dict[str, Any]
    series: dict[str, Any]

    @property
    def plasma(self) -> np.ndarray:
        return video._plasma(self.grid)

    @property
    def latest(self) -> int:
        return self.frames.count - 1

    @property
    def t_us(self) -> float:
        return float(self.frames.time_s[self.latest] * 1e6)


def prepare_run(spec: RunSpec, mirror: Path, *, iz_window: int | None = None, min_events: float = video.MIN_SAMPLES_DEFAULT,
                min_samples: int = video.MIN_SAMPLES_DEFAULT, log=print) -> RunData:
    stage = stage_mirror(Path(spec.results), mirror, log=log)
    if stage["frames_staged"] == 0:
        raise FileNotFoundError(f"{spec.design_id}: no loadable frames under {spec.results}")
    frames = load_frames(mirror)
    summary = video.load_summary(mirror)
    grid = video.grid_from_summary(summary)
    if grid.node_shape != tuple(frames.maps["n_i_per_m3"].shape[1:]):
        raise ValueError(f"{spec.design_id}: protocol grid {grid.node_shape} does not match the frame maps {frames.maps['n_i_per_m3'].shape[1:]}")
    # the renderer caches mesh masks by id(grid); a grid freed earlier in the same process can hand its id to this one,
    # so the entry is written explicitly (the RunData keeps the grid alive for as long as the masks are used)
    masks = build_mesh_masks(grid)
    video._MASKS_CACHE[id(grid)] = masks
    macro_weight, dt_s = video.run_constants(summary)
    ionisation = video.prepare_ionisation(frames, masks, macro_weight, dt_s, window=iz_window, min_events=min_events, min_samples=min_samples)
    return RunData(spec, mirror, stage, frames, summary, grid, ionisation, load_status_series(Path(spec.results)))


# -- shared colour scales -------------------------------------------------------------------------------------


def _log_scale(values: np.ndarray, lo: float, hi: float, basis: str, percentiles: Sequence[float] | None = None) -> dict[str, Any]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= 0.0:
        lo, hi = 1.0, 10.0
    if not lo < hi:
        lo, hi = (hi / 10.0, hi * 10.0) if hi > 0 else (0.1, 1.0)
    lo = max(lo, hi * 1e-12)
    out = {"kind": "log", "lo": float(lo), "hi": float(hi), "decades": float(math.log10(hi / lo)), "basis": basis, "samples": int(values.size)}
    if percentiles is not None:
        out["percentiles"] = [float(p) for p in percentiles]
    return out


def panel_scales(runs: Sequence[RunData], *, min_samples: int = video.MIN_SAMPLES_DEFAULT, ni_decades: float = NI_DECADES,
                 ni_top_percentile: float = NI_TOP_PERCENTILE, iz_percentiles: Sequence[float] = video.IZ_PERCENTILES) -> dict[str, dict[str, Any]]:
    """One colour scale per column, pooled over the LATEST frame of every run (declared in the figure and the report)."""

    ni, phi, iz = [], [], []
    for run in runs:
        i, plasma = run.latest, run.plasma
        resolved = plasma & (run.frames.maps["sample_count_e"][i] >= min_samples)
        values = run.frames.maps["n_i_per_m3"][i].astype(np.float64)
        ni.append(values[resolved & np.isfinite(values) & (values > 0.0)])
        potential = run.frames.maps["phi_v"][i].astype(np.float64)
        phi.append(potential[plasma & np.isfinite(potential)])
        rate = run.ionisation["rate"][i]
        iz.append(rate[run.ionisation["resolved"][i] & np.isfinite(rate) & (rate > 0.0)])
    ni_v, phi_v, iz_v = (np.concatenate(v) if v else np.array([]) for v in (ni, phi, iz))
    hi = float(np.percentile(ni_v, ni_top_percentile)) if ni_v.size else 1.0
    scales = {
        "n_i_per_m3": _log_scale(ni_v, hi / 10.0**ni_decades, hi,
                                 f"hi = {ni_top_percentile:g}th percentile of the electron-resolved (>= {min_samples} samples) plasma nodes pooled over the "
                                 f"latest frame of every design; lo = hi / 10^{ni_decades:g}"),
        "phi_v": {"kind": "signed", "lo": float(np.min(phi_v)) if phi_v.size else 0.0, "hi": float(np.max(phi_v)) if phi_v.size else 1.0,
                  "basis": "min-max of the plasma nodes pooled over the latest frame of every design (diverging palette, as the v0.2 videos)",
                  "samples": int(phi_v.size)},
    }
    if not scales["phi_v"]["lo"] < scales["phi_v"]["hi"]:
        scales["phi_v"]["lo"], scales["phi_v"]["hi"] = scales["phi_v"]["lo"] - 1.0, scales["phi_v"]["hi"] + 1.0
    if iz_v.size:
        lo, hi = (float(v) for v in np.percentile(iz_v, list(iz_percentiles)))
    else:
        lo, hi = 1.0, 10.0
    scales[video.IZ_KEY] = _log_scale(iz_v, lo, hi, f"{iz_percentiles[0]:g}th-{iz_percentiles[1]:g}th percentile of the resolved windowed nodes "
                                      f"(>= {runs[0].ionisation['min_events']:g} events) pooled over the latest frame of every design", iz_percentiles)
    return scales


# -- drawing helpers -------------------------------------------------------------------------------------------


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:   # pragma: no cover - Pillow < 10.1
        return ImageFont.load_default()


def _dashed_vertical(draw, x: int, y0: int, y1: int, colour: tuple[int, int, int], dash: int = 6, gap: int = 4, width: int = 1) -> None:
    y = y0
    while y < y1:
        draw.line([(x, y), (x, min(y + dash, y1))], fill=colour, width=width)
        y += dash + gap


def _dashed_horizontal(draw, x0: int, x1: int, y: int, colour: tuple[int, int, int], dash: int = 8, gap: int = 5, width: int = 1) -> None:
    x = x0
    while x < x1:
        draw.line([(x, y), (min(x + dash, x1), y)], fill=colour, width=width)
        x += dash + gap


def _shade(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(round(BG[k] + (colour[k] - BG[k]) * factor) for k in range(3))   # type: ignore[return-value]


def add_banner(image: np.ndarray, text: str, *, height: int = 30) -> np.ndarray:
    """A banner strip above a composed video frame (the INTERIM label every frame must carry)."""

    from PIL import Image, ImageDraw

    h, w = image.shape[:2]
    canvas = Image.new("RGB", (w, h + height), BANNER_BG)
    canvas.paste(Image.fromarray(np.asarray(image, dtype=np.uint8), "RGB"), (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), text, fill=BANNER_FG, font=_font(16))
    return np.asarray(canvas)


def _map_image(run: RunData, key: str, scale: Mapping[str, Any], *, min_samples: int, size: tuple[int, int]):
    from PIL import Image

    i, plasma = run.latest, run.plasma
    if key == video.IZ_KEY:
        idx = video.index_frame(run.ionisation["rate"][i], run.ionisation["events"][i], plasma, scale, run.ionisation["min_events"])
    else:
        idx = video.index_frame(run.frames.maps[key][i], run.frames.maps["sample_count_e"][i], plasma, scale, min_samples)
    return Image.fromarray(video.to_rgb(idx, scale["kind"]), "RGB").resize(size, Image.NEAREST)


def _panel_lines(run: RunData, key: str) -> list[str]:
    spec, i = run.spec, run.latest
    rho = f"{spec.rho:.2f}" if spec.rho is not None else "-"
    rwl = f"{spec.r_w_over_l:.2f}" if spec.r_w_over_l is not None else "-"
    i_d = run.frames.scalars["discharge_a"][i] * 1e3
    s = run.frames.scalars["ionization_rate_per_s"][i]
    n_e = run.frames.scalars["electrons"][i]
    steps = f"steps {int(run.frames.start_step[i])}-{int(run.frames.end_step[i])}"
    lines = [f"{spec.label} \u00b7 {spec.design_id} \u00b7 rho {rho} \u00b7 r_w/L {rwl}",
             (f"t = {run.t_us:.3f} \u00b5s (frame {i + 1}/{run.frames.count}, {steps}) \u00b7 "
              f"I_d {i_d:.2f} mA \u00b7 S {s:.2e} /s \u00b7 N_e {n_e / 1e6:.2f} M \u00b7 {NOT_PLATEAU}")]
    if key == video.IZ_KEY:
        iz = run.ionisation
        share = iz["share_resolved"][i]
        share_txt = f"{share * 100:.0f} %" if np.isfinite(share) else "-"
        lines.append(f"window {int(iz['frames_in_window'][i])}/{iz['window']} frames = {iz['window_s'][i] * 1e9:.0f} ns (causal) \u00b7 "
                     f"resolved {int(iz['resolved_nodes'][i])}/{iz['plasma_nodes']} nodes (>= {iz['min_events']:g} events) carry {share_txt} of "
                     f"S_window {iz['s_window_per_s'][i]:.2e} /s")
    elif key == "n_i_per_m3":
        peak = run.series["peak_cells_per_debye"][-1] if run.series.get("records") else math.nan
        lines.append(f"latest peak Delta/lambda_D {peak:.2f} (single step; hard gate {GATE_HARD:.2f}, soft {GATE_SOFT:g}) \u00b7 "
                     f"grey: < {video.MIN_SAMPLES_DEFAULT} electron samples in the frame")
    else:
        lines.append(f"phi_max {float(np.nanmax(run.frames.maps['phi_v'][i][run.plasma])):.1f} V \u00b7 anode-side at z = 0, exit plane at the right")
    return lines


def render_panel(runs: Sequence[RunData], out_png: Path, *, scales: Mapping[str, Mapping[str, Any]] | None = None,
                 min_samples: int = video.MIN_SAMPLES_DEFAULT, width: int = FIGURE_WIDTH, radial_exaggeration: float = RADIAL_EXAGGERATION,
                 title: str | None = None, stamp: str | None = None) -> dict[str, Any]:
    """The 4 x 3 comparison figure at each run's latest frame; returns the declared scales and layout."""

    from PIL import Image, ImageDraw

    runs = sorted(runs, key=lambda r: (r.spec.rho if r.spec.rho is not None else math.inf, r.spec.design_id))
    scales = dict(scales) if scales is not None else panel_scales(runs, min_samples=min_samples)
    margin, gap, header_h, title_h, row_gap, footer_h = 16, 12, 118, 56, 18, 150
    col_w = (width - 2 * margin - 2 * gap) // 3
    max_nz = max(r.grid.node_shape[1] for r in runs)
    ax_factor = min(1.0, col_w / max_nz)
    rad_factor = ax_factor * radial_exaggeration
    sizes = [(max(2, round(r.grid.node_shape[1] * ax_factor)), max(2, round(r.grid.node_shape[0] * rad_factor))) for r in runs]
    height = header_h + sum(title_h + h + row_gap for _, h in sizes) + footer_h
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    big, med, small = _font(24), _font(15), _font(13)
    t_lo, t_hi = min(r.t_us for r in runs), max(r.t_us for r in runs)
    head = title or (f"PIC design mini-sweep v1 \u00b7 {INTERIM_TAG} \u00b7 t = {t_lo:.3f}-{t_hi:.3f} \u00b5s \u00b7 {NOT_PLATEAU}")
    draw.text((margin, 10), head, fill=BANNER_FG, font=big)
    transits = [r.t_us * 1e-6 / r.spec.transit_time_s for r in runs if r.spec.transit_time_s]
    progress = (f"Runs are ~{min(transits):.2f}-{max(transits):.2f} ion transits into a 3-transit budget: " if transits else "")
    sub = [
        (f"Latest recorded frame of each run{(' at ' + stamp) if stamp else ''}; rows ordered by Koch rho (material-aware P2 field); "
         "columns: ion density n_i (log), potential phi, windowed ionisation rate (renderer v0.2: causal window, >= event mask, "
         "fixed percentile scale)."),
        (f"Colour scales shared per column (declared below). Dashed verticals = the design's OWN cusp planes (orange = boundary cusp "
         f"within {BOUNDARY_TOLERANCE_M * 1e3:.2f} mm of an end); grey = unresolved; dark = thruster body. Axial scale identical across "
         f"rows ({ax_factor:.2f} px per 33 um node); radial axis exaggerated {radial_exaggeration:g}x."),
        f"{progress}development pictures of the ignition phase, not plateau results.",
    ]
    for k, line in enumerate(sub):
        draw.text((margin, 48 + 20 * k), line, fill=MUTED, font=small)
    y = header_h
    layout = []
    for run, (w, h) in zip(runs, sizes):
        for c, key in enumerate(PANEL_MAPS):
            x = margin + c * (col_w + gap)
            scale = scales[key]
            for k, line in enumerate(_panel_lines(run, key)):
                draw.text((x, y + 2 + 17 * k), line, fill=TEXT if k == 0 else MUTED, font=med if k == 0 else small)
            top = y + title_h
            draw.rectangle([(x - 1, top - 1), (x + w, top + h)], outline=(60, 70, 78))
            canvas.paste(_map_image(run, key, scale, min_samples=min_samples, size=(w, h)), (x, top))
            for (x0, y0), (x1, y1) in video.mask_outline(run.plasma):
                draw.line([(x + x0 * ax_factor, top + y0 * rad_factor), (x + x1 * ax_factor, top + y1 * rad_factor)], fill=MUTED, width=1)
            z0, z1 = run.grid.geometry.z_min_m, run.grid.geometry.domain_z_max_m
            nz = run.grid.node_shape[1]
            for z in run.spec.cusp_z_m:
                px = x + round((z - z0) / (z1 - z0) * (nz - 1) * ax_factor)
                boundary = z in run.spec.boundary_cusp_z_m
                _dashed_vertical(draw, px, top, top + h, BOUNDARY_CUSP_RGB if boundary else CUSP_RGB, width=1)
                draw.text((px + 3, top + h - 15), f"{z * 1e3:.2f}", fill=BOUNDARY_CUSP_RGB if boundary else CUSP_RGB, font=small)
            extent = f"z {z0 * 1e3:.0f}-{z1 * 1e3:.1f} mm, r <= {run.grid.geometry.max_radius_m * 1e3:.2f} mm"
            draw.text((x + col_w - 6 - int(draw.textlength(extent, font=small)), y + 3), extent, fill=MUTED, font=small)
            layout.append({"design_id": run.spec.design_id, "map": key, "x": x, "y": top, "w": w, "h": h, "frame": run.latest, "t_us": run.t_us})
        y += title_h + h + row_gap
    # footer: one colour bar per column with its declaration
    foot = height - footer_h + 10
    labels = {"n_i_per_m3": "ion density n_i (m^-3)", "phi_v": "potential phi (V)", video.IZ_KEY: "windowed ionisation rate (m^-3 s^-1)"}
    for c, key in enumerate(PANEL_MAPS):
        x = margin + c * (col_w + gap)
        scale = scales[key]
        bar = video.palette(scale["kind"])[:254]
        bar_img = Image.fromarray(np.repeat(bar[None, :, :], 14, axis=0), "RGB").resize((360, 14), Image.NEAREST)
        canvas.paste(bar_img, (x, foot + 20))
        lo_txt = f"{scale['lo']:.2e}" if scale["kind"] == "log" else f"{scale['lo']:.1f}"
        hi_txt = f"{scale['hi']:.2e}" if scale["kind"] == "log" else f"{scale['hi']:.1f}"
        draw.text((x, foot), labels[key], fill=TEXT, font=med)
        draw.text((x, foot + 36), lo_txt, fill=TEXT, font=small)
        draw.text((x + 360 - int(draw.textlength(hi_txt, font=small)), foot + 36), hi_txt, fill=TEXT, font=small)
        draw.rectangle([(x + 372, foot + 20), (x + 386, foot + 34)], fill=video.MASK_RGB)
        draw.text((x + 390, foot + 20), "unresolved", fill=MUTED, font=small)
        draw.rectangle([(x + 470, foot + 20), (x + 484, foot + 34)], fill=video.BODY_RGB)
        draw.text((x + 488, foot + 20), "body", fill=MUTED, font=small)
        kind = f"log10, {scale['decades']:.1f} decades" if scale["kind"] == "log" else "linear, diverging palette"
        for k, line in enumerate(_wrap(f"{kind}; {scale['basis']}", 118)):
            draw.text((x, foot + 56 + 15 * k), line, fill=MUTED, font=small)
    ionisation_note = (f"Ionisation window per run (declared on the panel): K = smallest window whose median resolved event-bearing node holds >= "
                       f"{runs[0].ionisation['target_median_events']:g} events; mask >= {runs[0].ionisation['min_events']:g} windowed events; no spatial smoothing. "
                       f"{INTERIM_TAG} \u00b7 {NOT_PLATEAU}.")
    draw.text((margin, height - 22), ionisation_note, fill=BANNER_FG, font=small)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, format="PNG", optimize=True)
    return {"path": str(out_png), "width": width, "height": height, "scales": scales, "layout": layout, "radial_exaggeration": radial_exaggeration,
            "axial_px_per_node": ax_factor, "rows": [r.spec.design_id for r in runs], "columns": list(PANEL_MAPS),
            "sha256": hashlib.sha256(out_png.read_bytes()).hexdigest()}


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# -- time-series strip -----------------------------------------------------------------------------------------

TS_AXES = (("discharge_a", 1e3, "I_d (mA)"), ("ionization_rate_per_s", 1e-16, "S (1e16 /s)"), ("electrons", 1e-6, "N_e (M macro)"),
           ("peak_cells_per_debye", 1.0, "peak Delta/lambda_D (thin: single step; thick: window mean)"))


TS_PERCENTILES = (1.0, 99.0)              # axis range of the strip: robust to the seed transient (values beyond it are clipped)


def _robust_range(pooled: np.ndarray, key: str, percentiles: tuple[float, float] = TS_PERCENTILES) -> tuple[float, float]:
    """Axis range = robust percentiles of the pooled records; zero-based for the counting series; the Debye axis shows the gates."""

    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(finite, list(percentiles)))
    if key == "peak_cells_per_debye":
        lo, hi = 0.0, max(hi, GATE_HARD * 1.15)
    elif key in ("discharge_a", "ionization_rate_per_s", "electrons"):
        lo = min(lo, 0.0)
    if not lo < hi:
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def _value_mapper(lo: float, hi: float, y0: int, y1: int):
    span = (hi - lo) or 1.0

    def y_of(value: float) -> float:
        return y1 - (value - lo) / span * (y1 - y0)

    return y_of


def render_timeseries(runs: Sequence[RunData], out_png: Path, *, width: int = FIGURE_WIDTH, height: int = 1000, title: str | None = None,
                      stamp: str | None = None) -> dict[str, Any]:
    """Four stacked axes (I_d, S, N_e, peak Delta/lambda_D) with every design on each, from status.jsonl."""

    from PIL import Image, ImageDraw

    runs = sorted(runs, key=lambda r: (r.spec.rho if r.spec.rho is not None else math.inf, r.spec.design_id))
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    big, med, small = _font(22), _font(14), _font(12)
    left, right, top, bottom, gap = 120, 320, 70, 50, 26
    t_hi = max((float(np.nanmax(r.series["time_s"])) if r.series.get("records") else 0.0) for r in runs) * 1e6 or 1.0
    head = title or f"PIC design mini-sweep v1 \u00b7 {INTERIM_TAG} \u00b7 series to t = {t_hi:.3f} \u00b5s \u00b7 {NOT_PLATEAU}"
    draw.text((16, 10), head, fill=BANNER_FG, font=big)
    draw.text((16, 42), f"status.jsonl records (every series interval){(' read at ' + stamp) if stamp else ''}; x = simulated time (us); "
              f"each axis spans the {TS_PERCENTILES[0]:g}-{TS_PERCENTILES[1]:g}th percentile of the pooled records (the seed transient is clipped "
              "at the axis edge); Debye axis: thin = single-step peak node, thick = window mean, gate lines hard pi (red), soft 2.5 (orange). "
              "Not a plateau.", fill=MUTED, font=small)
    axis_h = (height - top - bottom - gap * (len(TS_AXES) - 1)) // len(TS_AXES)

    def x_of(t_us: float) -> float:
        return left + t_us / t_hi * (width - left - right)

    report: dict[str, Any] = {"axes": [], "t_max_us": t_hi}
    for a, (key, factor, label) in enumerate(TS_AXES):
        y0 = top + a * (axis_h + gap)
        y1 = y0 + axis_h
        draw.rectangle([(left, y0), (width - right, y1)], fill=PANEL_BG, outline=(60, 70, 78))
        values = []
        for r in runs:
            if r.series.get("records"):
                v = r.series[key] * factor
                values.append(v[np.isfinite(v)])
                if key == "peak_cells_per_debye":
                    w = r.series["window_cells_per_debye"]
                    values.append(w[np.isfinite(w)])
        lo, hi = _robust_range(np.concatenate(values) if values else np.array([0.0, 1.0]), key)
        pad = 0.05 * (hi - lo)
        y_of = _value_mapper(lo - (0.0 if lo == 0.0 else pad), hi + pad, y0, y1)
        for tick in np.linspace(lo, hi, 5):
            yy = round(y_of(tick))
            draw.line([(left, yy), (width - right, yy)], fill=(40, 48, 54), width=1)
            txt = f"{tick:.3g}"
            draw.text((left - 8 - int(draw.textlength(txt, font=small)), yy - 7), txt, fill=MUTED, font=small)
        draw.text((left + 8, y0 + 4), label, fill=TEXT, font=med)
        if key == "peak_cells_per_debye":
            for value, colour, name in ((GATE_HARD, (255, 90, 90), "hard gate pi"), (GATE_SOFT, BOUNDARY_CUSP_RGB, "soft 2.5")):
                yy = round(y_of(value))
                _dashed_horizontal(draw, left, width - right, yy, colour)
                draw.text((width - right - 90, yy - 15), name, fill=colour, font=small)
        for r, colour in zip(runs, DESIGN_COLOURS):
            if not r.series.get("records"):
                continue
            t = r.series["time_s"] * 1e6
            if key == "peak_cells_per_debye":
                thin = np.clip(r.series["peak_cells_per_debye"], lo, hi)
                pts = [(x_of(t[k]), y_of(thin[k])) for k in range(t.size) if np.isfinite(thin[k])]
                if len(pts) > 1:
                    draw.line(pts, fill=_shade(colour, 0.45), width=1)
                v = np.clip(r.series["window_cells_per_debye"], lo, hi)
                width_px = 3
            else:
                v = np.clip(r.series[key] * factor, lo, hi)
                width_px = 2
            pts = [(x_of(t[k]), y_of(v[k])) for k in range(t.size) if np.isfinite(v[k])]
            if len(pts) > 1:
                draw.line(pts, fill=colour, width=width_px)
        report["axes"].append({"key": key, "label": label, "lo": lo, "hi": hi})
    for k in range(6):
        t = t_hi * k / 5
        xx = round(x_of(t))
        draw.line([(xx, height - bottom), (xx, height - bottom + 5)], fill=MUTED, width=1)
        draw.text((xx - 14, height - bottom + 8), f"{t:.3f}", fill=MUTED, font=small)
    draw.text((width // 2 - 60, height - 22), "simulated time t (us)", fill=MUTED, font=small)
    legend_x, legend_y = width - right + 16, top
    for r, colour in zip(runs, DESIGN_COLOURS):
        n = r.series.get("records", 0)
        last_t = float(r.series["time_s"][-1] * 1e6) if n else float("nan")
        rho = f"{r.spec.rho:.2f}" if r.spec.rho is not None else "-"
        draw.rectangle([(legend_x, legend_y + 4), (legend_x + 18, legend_y + 14)], fill=colour)
        draw.text((legend_x + 26, legend_y), f"{r.spec.label} rho {rho} \u00b7 {r.spec.design_id}", fill=TEXT, font=small)
        draw.text((legend_x + 26, legend_y + 16), f"{n} records to t = {last_t:.3f} us; frames {r.frames.count}", fill=MUTED, font=small)
        legend_y += 42
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, format="PNG", optimize=True)
    report.update({"path": str(out_png), "width": width, "height": height, "rows": [r.spec.design_id for r in runs],
                   "sha256": hashlib.sha256(out_png.read_bytes()).hexdigest()})
    return report


# -- per-design videos (v0.2 panels + INTERIM banner) + HTML player ------------------------------------------------


def render_videos(run: RunData, out_dir: Path, *, maps: Sequence[str] = video.DEFAULT_MAPS, fps: int = 8, upscale: int = 3, factor: int = 2,
                  min_samples: int = video.MIN_SAMPLES_DEFAULT, backend: str | None = None, html: bool = True, log=print) -> dict[str, Any]:
    """render_run's per-map loop on the mirror, every composed frame wearing the INTERIM banner; the player title too."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, grid, plasma, spec = run.frames, run.grid, run.plasma, run.spec
    name = f"interim-{spec.label}-{spec.design_id}"
    rho = f"{spec.rho:.2f}" if spec.rho is not None else "-"
    report: dict[str, Any] = {"design_id": spec.design_id, "label": spec.label, "frames": frames.count, "videos": {}, "html": None, "backend": None,
                              "banner_example": f"{interim_title(run.t_us)} \u00b7 {spec.label} {spec.design_id} \u00b7 rho {rho}"}
    for key in maps:
        images = []
        if key == video.IZ_KEY:
            scale = run.ionisation["scale"]
            for i in range(frames.count):
                title_suffix, legend = video.ionisation_legend(run.ionisation, i)
                idx = video.index_frame(run.ionisation["rate"][i], run.ionisation["events"][i], plasma, scale, run.ionisation["min_events"])
                images.append(video.compose_video_frame(idx, scale, key, i, frames, grid, upscale=upscale, min_samples=min_samples,
                                                        cusp_z_m=spec.cusp_z_m, title_suffix=title_suffix, legend=legend))
        else:
            scale = video.colour_scale(frames, key, plasma, min_samples)
            for i in range(frames.count):
                idx = video.index_frame(frames.maps[key][i], frames.maps["sample_count_e"][i], plasma, scale, min_samples)
                images.append(video.compose_video_frame(idx, scale, key, i, frames, grid, upscale=upscale, min_samples=min_samples, cusp_z_m=spec.cusp_z_m))
        images = [add_banner(im, f"{interim_title(float(frames.time_s[i] * 1e6))} \u00b7 {spec.label} {spec.design_id} \u00b7 rho {rho} \u00b7 "
                             f"frame {i + 1}/{frames.count}") for i, im in enumerate(images)]
        path, used = video.write_video(images, out_dir / f"pic2d-{name}-{key}", fps, backend)
        report["videos"][key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "scale": scale}
        report["backend"] = used
        log(f"[video] {spec.label} {key}: {path.name} ({path.stat().st_size / 1e6:.2f} MB, {used})")
    if html:
        payload = video.build_player_payload(run.mirror, frames, run.summary, grid, maps=maps, min_samples=min_samples, factor=factor,
                                             cusp_z_m=spec.cusp_z_m, ionisation=run.ionisation if video.IZ_KEY in maps else None)
        text = video.render_player_html(payload, f"{interim_title(run.t_us)} \u00b7 {spec.label} {spec.design_id} \u00b7 rho {rho}")
        path = out_dir / f"pic2d-{name}-timeseries.html"
        path.write_text(text, encoding="utf-8", newline="\n")
        report["html"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return report


# -- driver ------------------------------------------------------------------------------------------------------


def render_all(manifest: Path, out_dir: Path, *, stage_dir: Path | None = None, videos: bool = True, panel: bool = True,
               maps: Sequence[str] = video.DEFAULT_MAPS, fps: int = 8, upscale: int = 3, factor: int = 2, backend: str | None = None,
               iz_window: int | None = None, min_samples: int = video.MIN_SAMPLES_DEFAULT, log=print) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    specs = load_manifest(Path(manifest))
    runs: list[RunData] = []
    problems: list[dict[str, str]] = []
    for spec in specs:
        mirror = (Path(stage_dir) if stage_dir else out_dir) / spec.design_id / "mirror"
        try:
            runs.append(prepare_run(spec, mirror, iz_window=iz_window, min_samples=min_samples, log=log))
        except Exception as error:  # noqa: BLE001 - one run without frames must not block the others
            problems.append({"design_id": spec.design_id, "error": f"{type(error).__name__}: {error}"})
            log(f"[render] {spec.design_id}: skipped - {type(error).__name__}: {error}")
    if not runs:
        raise FileNotFoundError("no run has loadable frames yet")
    report: dict[str, Any] = {"schema": SCHEMA, "kind": "interim-sweep-render", "label": f"{INTERIM_TAG} \u00b7 {NOT_PLATEAU}", "rendered_utc": stamp,
                              "manifest": str(manifest), "out_dir": str(out_dir), "problems": problems, "runs": [], "videos": {}}
    rows = []
    for run in runs:
        row = status_row(run.spec, frames_staged=run.stage["frames_staged"], series=run.series)
        rows.append(row)
        report["runs"].append({"spec": asdict(run.spec), "stage": run.stage, "status": row, "latest_frame": run.latest, "t_us": run.t_us,
                               "ionisation": {"window_frames": run.ionisation["window"], "window_s": run.ionisation["nominal_window_s"],
                                              "auto": run.ionisation["auto"], "min_events": run.ionisation["min_events"],
                                              "resolved_nodes_latest": int(run.ionisation["resolved_nodes"][run.latest]),
                                              "plasma_nodes": run.ionisation["plasma_nodes"],
                                              "share_resolved_latest": (None if not np.isfinite(run.ionisation["share_resolved"][run.latest])
                                                                        else float(run.ionisation["share_resolved"][run.latest])),
                                              "scale_run": run.ionisation["scale"]}})
    table = format_status_table(rows)
    (out_dir / "interim-sweep-status.md").write_text(f"# {INTERIM_TAG} \u00b7 status at {stamp} \u00b7 {NOT_PLATEAU}\n\n{table}", encoding="utf-8", newline="\n")
    report["status_table_markdown"] = table
    if panel:
        report["panel"] = render_panel(runs, out_dir / "interim-sweep-panel.png", min_samples=min_samples, stamp=stamp)
        report["timeseries"] = render_timeseries(runs, out_dir / "interim-sweep-timeseries.png", stamp=stamp)
        log(f"[panel] {report['panel']['path']} ({report['panel']['width']}x{report['panel']['height']}); {report['timeseries']['path']}")
    if videos:
        for run in runs:
            report["videos"][run.spec.design_id] = render_videos(run, out_dir / run.spec.design_id / "video", maps=maps, fps=fps, upscale=upscale,
                                                                 factor=factor, min_samples=min_samples, backend=backend, log=log)
    (out_dir / "interim-sweep-report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=_plain) + "\n", encoding="utf-8", newline="\n")
    return report


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def cmd_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(Path(args.jobs_dir), args.job or list(DEFAULT_JOBS), fields_dir=Path(args.fields_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"[manifest] {len(manifest['runs'])} runs -> {out}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rows = []
    for spec in load_manifest(Path(args.manifest)):
        kept, _, _ = loadable_frames(Path(spec.results))
        rows.append(status_row(spec, frames_staged=len(kept)))
    if args.json:
        print(json.dumps(rows, indent=1, default=_plain))
    else:
        print(f"{INTERIM_TAG} \u00b7 {dt.datetime.now(dt.timezone.utc).isoformat()} \u00b7 {NOT_PLATEAU}\n")
        print(format_status_table(rows))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    report = render_all(Path(args.manifest), Path(args.out_dir), stage_dir=Path(args.stage_dir) if args.stage_dir else None, videos=not args.no_videos,
                        panel=not args.no_panel, maps=args.maps, fps=args.fps, upscale=args.upscale, factor=args.downsample, backend=args.backend,
                        iz_window=args.iz_window, min_samples=args.min_samples)
    print(report["status_table_markdown"])
    if report.get("panel"):
        print(f"panel: {report['panel']['path']}\ntimeseries: {report['timeseries']['path']}")
    for design_id, block in report["videos"].items():
        for key, item in block["videos"].items():
            print(f"video {design_id} {key}: {item['path']}")
        if block.get("html"):
            print(f"html  {design_id}: {block['html']['path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest", help="runs.json from the scheduler job states + design bindings")
    p.add_argument("--jobs-dir", required=True)
    p.add_argument("--job", action="append", default=None, help="job id (repeatable; default: the four primary sweep jobs)")
    p.add_argument("--fields-dir", default=str(FIELDS_DIR))
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_manifest)
    p = sub.add_parser("status", help="status table of the runs in the manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("render", help="stage mirrors, comparison PNGs, per-design videos + HTML players")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stage-dir", default=None, help="where the mirrors go (default: <out-dir>/<design>/mirror)")
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--no-panel", action="store_true")
    p.add_argument("--maps", nargs="*", default=list(video.DEFAULT_MAPS), choices=list(video.MAP_LABELS))
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--backend", choices=["imageio_ffmpeg", "ffmpeg", "pillow_gif"], default=None)
    p.add_argument("--iz-window", type=int, default=None)
    p.add_argument("--min-samples", type=int, default=video.MIN_SAMPLES_DEFAULT)
    p.set_defaults(func=cmd_render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
