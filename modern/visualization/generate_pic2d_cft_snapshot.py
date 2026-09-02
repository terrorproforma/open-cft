"""Generate the standalone PIC-2D CFT snapshot dashboard.

Embeds the hash-verified case summaries, time series and time-averaged maps of
``modern/experiments/pic2d_cft_snapshot_v1/results``.  No timestamps or runtime
measurements are added, so identical inputs give identical bytes.  The page is
self-contained (no network access) and states its development/screening claim
boundary on every view.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

MODERN = Path(__file__).resolve().parents[1]
SRC = MODERN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cft_revival.pic2d.artifacts import read_canonical_json, read_npz  # noqa: E402

RESULTS = MODERN / "experiments" / "pic2d_cft_snapshot_v1" / "results"
PROTOCOL = MODERN / "experiments" / "pic2d_cft_snapshot_v1" / "protocol.json"
DEFAULT_OUTPUT = Path(__file__).with_name("pic2d-cft-snapshot.html")
SCHEMA = "cft-pic2d-cft-snapshot-visualization/0.1.0"
MAP_KEYS = ("n_e_per_m3", "n_i_per_m3", "phi_v", "t_e_ev", "ionization_rate_per_m3_s")
WALL_KEYS = ("wall_electron_flux_per_m2_s", "wall_ion_flux_per_m2_s", "wall_electron_mean_energy_ev", "wall_ion_mean_energy_ev")
EXIT_KEYS = ("exit_ion_current_density_a_per_m2", "exit_electron_current_density_a_per_m2")
SERIES_KEYS = (
    "time_s", "electrons", "ions", "phi_min_v", "phi_mean_v", "phi_max_v", "kinetic_electron_j", "kinetic_ion_j",
    "field_energy_j", "total_energy_j", "interval_residual_j", "interval_sources_j", "peak_omega_pe_dt",
    "current_discharge_a", "current_exit_ion_beam_a", "current_exit_electron_a", "current_wall_electron_a",
    "current_wall_ion_a", "current_injected_electron_a", "current_anode_electron_a", "current_ionization_rate_per_s",
)


def _round(values: np.ndarray, digits: int = 6) -> list[Any]:
    out: list[Any] = []
    for value in np.asarray(values, dtype=np.float64).ravel().tolist():
        if not isfinite(value):
            out.append(None)
        else:
            out.append(float(f"{value:.{digits}g}"))
    return out


def _matrix(values: np.ndarray, digits: int = 5) -> list[list[Any]]:
    return [_round(row, digits) for row in np.asarray(values, dtype=np.float64)]


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _verify_sidecar(path: Path) -> str:
    sidecar = json.loads(path.with_name(path.name + ".sha256.json").read_text(encoding="utf-8"))
    digest = _file_sha256(path)
    if sidecar["byte_sha256"] != digest:
        raise ValueError(f"{path.name}: sidecar SHA-256 mismatch")
    return digest


def build_payload(results: Path = RESULTS, protocol_path: Path = PROTOCOL) -> dict[str, Any]:
    manifest_path = results / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("snapshot manifest is missing; run `run.py summarize` first")
    manifest = read_canonical_json(manifest_path)
    manifest_sha = _verify_sidecar(manifest_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _file_sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("protocol file differs from the hash recorded in the manifest")
    cases: list[dict[str, Any]] = []
    for case, entry in manifest["cases"].items():
        case_dir = results / case
        summary_path = case_dir / "summary.json"
        if _verify_sidecar(summary_path) != entry["summary_sha256"]:
            raise ValueError(f"{case}: summary SHA-256 differs from manifest")
        summary = read_canonical_json(summary_path)
        maps = read_npz(case_dir / "maps.npz", expected_sha256=entry["maps_npz_sha256"])
        series = read_npz(case_dir / "series.npz", expected_sha256=entry["series_npz_sha256"])
        grid = summary["provenance"]["config"]["grid"]
        nr, nz = int(grid["radial_cells"]), int(grid["axial_cells"])
        r = np.arange(nr + 1) * float(grid["dr_m"])
        z = float(grid["geometry"]["z_min_m"]) + np.arange(nz + 1) * float(grid["dz_m"])
        stride = 1 if nz <= 256 else 2
        plasma = np.isfinite(maps["phi_v"]) & (maps["n_e_per_m3"] >= 0)
        # nodes outside the plasma carry zeros in the maps; mask them as null for display
        from cft_revival.pic2d.mesh import build_mesh_masks
        from cft_revival.pic2d.models import ChannelGeometry, Grid2D

        geometry = grid["geometry"]
        masks = build_mesh_masks(
            Grid2D(ChannelGeometry(geometry["bore_radius_m"], geometry["z_min_m"], geometry["z_max_m"], geometry["cone_start_z_m"], geometry["exit_radius_m"]), nr, nz)
        )
        plasma = masks.plasma_node
        embedded_maps: dict[str, Any] = {}
        for key in MAP_KEYS:
            values = np.where(plasma, maps[key], np.nan)[:, ::stride]
            embedded_maps[key] = _matrix(values)
        wall_z = (z[:-1] + 0.5 * float(grid["dz_m"]))
        cases.append(
            {
                "id": case,
                "label": case,
                "spec": summary["case_spec"],
                "summary_sha256": entry["summary_sha256"],
                "maps_npz_sha256": entry["maps_npz_sha256"],
                "series_npz_sha256": entry["series_npz_sha256"],
                "steps_completed": summary["steps_completed"],
                "target_steps": summary["target_steps"],
                "simulated_time_s": summary["simulated_time_s"],
                "stop_reason": summary["stop_reason"],
                "stability_gate_message": summary.get("stability_gate_message"),
                "averaging_window_step_range": summary.get("averaging_window_step_range"),
                "averaging_window_steps": summary["averaging_window_steps"],
                "wall_seconds_run": float(f"{summary['wall_seconds_run']:.4g}"),
                "steps_per_second": float(f"{summary['steps_per_second']:.4g}"),
                "gpu_utilisation_percent_samples": summary.get("gpu_utilisation_percent_samples", []),
                "backend": summary["backend"],
                "final_counts": summary["final_counts"],
                "window_maps_summary": summary["window_maps_summary"],
                "stability_gate": summary["provenance"]["stability_gate"],
                "mesh": summary["provenance"]["mesh"],
                "config": {
                    "dt_s": summary["provenance"]["config"]["dt_s"],
                    "macro_weight": summary["provenance"]["config"]["macro_weight"],
                    "grid": {"radial_cells": nr, "axial_cells": nz, "dr_m": grid["dr_m"], "dz_m": grid["dz_m"]},
                    "potentials": summary["provenance"]["config"]["potentials"],
                    "injection": summary["provenance"]["config"]["injection"],
                    "seed_plasma": summary["provenance"]["config"]["seed_plasma"],
                    "mcc": summary["provenance"]["config"]["mcc"],
                },
                "field": summary["provenance"]["field"],
                "cross_sections": summary["provenance"].get("cross_sections"),
                "grid_r_m": _round(r),
                "grid_z_m": _round(z[::stride]),
                "maps": embedded_maps,
                "wall_z_m": _round(wall_z),
                "wall": {key: _round(maps[key]) for key in WALL_KEYS},
                "exit_r_m": _round(0.5 * (r[:-1] + r[1:])),
                "exit": {key: _round(maps[key]) for key in EXIT_KEYS},
                "series": {key: _round(series[key]) for key in SERIES_KEYS if key in series},
                "final_series": summary["final_series"],
            }
        )
    payload = {
        "schema": SCHEMA,
        "experiment_id": manifest["experiment_id"],
        "status": manifest["status"],
        "claim_boundary": manifest["claim_boundary"],
        "claim_statement": (
            "Development/screening PIC-MCC snapshot. Not preregistered. Not validated against any experiment. "
            "Not a thruster performance prediction. Numerics verified by the tests in modern/tests/pic2d; physics "
            "simplified as listed."
        ),
        "simplifications": manifest["simplifications"],
        "manifest": {"file": manifest_path.name, "file_sha256": manifest_sha, "protocol_sha256": manifest["protocol_sha256"]},
        "protocol": {
            "operating_point": protocol["operating_point"],
            "numerics": protocol["numerics"],
            "geometry": protocol["geometry"],
            "stopping_rule": protocol["stopping_rule"],
        },
        "convergence": manifest["convergence"],
        "cases": cases,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    required = {"schema", "experiment_id", "status", "claim_boundary", "claim_statement", "simplifications", "manifest", "protocol", "convergence", "cases"}
    if set(payload) != required:
        raise ValueError("payload keys do not match the closed schema")
    if payload["schema"] != SCHEMA:
        raise ValueError("unsupported payload schema")
    if payload["status"] != "development_screening_not_preregistered":
        raise ValueError("payload must carry the development/screening status")
    if not payload["simplifications"] or "not preregistered" not in payload["claim_statement"].lower():
        raise ValueError("claim boundary must be explicit")
    if not payload["cases"]:
        raise ValueError("payload must contain at least one case")
    for case in payload["cases"]:
        for key in ("summary_sha256", "maps_npz_sha256", "series_npz_sha256"):
            digest = case[key]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{case['id']}: {key} must be a SHA-256")
        if case["stop_reason"] not in {"target_steps_reached", "wall_clock_budget_reached", "runtime_stability_gate_stopped_run"}:
            raise ValueError(f"{case['id']}: unknown stop reason")
        nr = len(case["grid_r_m"])
        nz = len(case["grid_z_m"])
        for key in MAP_KEYS:
            matrix = case["maps"][key]
            if len(matrix) != nr or any(len(row) != nz for row in matrix):
                raise ValueError(f"{case['id']}: map {key} shape does not match the grid")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIC-2D CFT snapshot (development)</title>
<style>
:root{color-scheme:dark;--bg:#07100f;--panel:#0f1c1a;--panel2:#14262380;--text:#eef7f4;--muted:#9bb8b0;--line:#2b4540;--accent:#5ad6c0;--warn:#ffcf67;--red:#ff6b6b;--blue:#58a8ff;--shadow:#0008}
[data-theme=light]{color-scheme:light;--bg:#edf5f2;--panel:#fff;--panel2:#f2f8f6;--text:#10231f;--muted:#4f6a63;--line:#bfd3cc;--accent:#087f6e;--warn:#7a5700;--red:#b83232;--blue:#176db5;--shadow:#3452}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#153b34 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
button,select{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:.55rem;padding:.48rem .7rem}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
header,main,footer{width:min(1500px,calc(100% - 2rem));margin:auto}header{padding:2rem 0 1rem}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(1.9rem,4.5vw,3.8rem);line-height:.98;margin:.2rem 0 .8rem;max-width:960px}h2{margin:.1rem 0 .8rem;font-size:1.1rem}p{margin:.35rem 0}
.claim{border:1px solid #8b681c;background:#513d1438;color:var(--warn);padding:.8rem 1rem;border-radius:.65rem;font-weight:650}.claim ul{margin:.4rem 0 0 1.1rem;font-weight:500;color:var(--text)}
.controls{display:flex;gap:.6rem;flex-wrap:wrap;align-items:end;margin:1rem 0}.control{display:grid;gap:.25rem}.control label{color:var(--muted);font-size:.8rem}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,.8fr);gap:1rem;margin:1rem 0}.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:.9rem;padding:1rem;box-shadow:0 12px 30px var(--shadow);min-width:0}
.canvas-wrap{position:relative;min-height:300px}.canvas-wrap canvas{width:100%;height:clamp(300px,34vw,460px);display:block}.tip{position:absolute;pointer-events:none;background:#07100fee;color:#fff;border:1px solid #7f9a93;border-radius:.35rem;padding:.35rem .5rem;display:none;white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.65rem}.metric-card{border:1px solid var(--line);border-radius:.7rem;padding:.75rem;background:var(--panel);min-width:0}.metric-card.active{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}.metric-card h3{font-size:.95rem;margin:0 0 .55rem}
.kv{display:grid;grid-template-columns:1fr auto;gap:.22rem .6rem}.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){font-variant-numeric:tabular-nums;text-align:right}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.plot{width:100%;height:260px;display:block}.wide{grid-column:1/-1}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;color:var(--muted)}code{font-size:.78em;overflow-wrap:anywhere}.small{font-size:.82rem;color:var(--muted)}footer{padding:1rem 0 2.5rem;color:var(--muted)}
@media(max-width:900px){.grid,.plots{grid-template-columns:1fr}.canvas-wrap canvas{height:360px}}@media(max-width:520px){header,main,footer{width:min(100% - 1rem,1500px)}.canvas-wrap canvas{height:300px}.panel{padding:.7rem}}
</style>
</head>
<body>
<header><div class="eyebrow">PIC-MCC · axisymmetric (r,z) · development snapshot</div><h1>Divergent-exit CFT channel: first kinetic snapshot</h1>
<div id="claim" class="claim" role="note"></div>
<div class="controls">
<div class="control"><label for="case">Case</label><select id="case"></select></div>
<div class="control"><label for="map">Map (time-averaged over the final window)</label><select id="map"><option value="n_e_per_m3">Electron density n_e (m⁻³)</option><option value="n_i_per_m3">Ion density n_i (m⁻³)</option><option value="phi_v">Potential φ (V)</option><option value="t_e_ev">Electron temperature T_e (eV)</option><option value="ionization_rate_per_m3_s">Ionisation rate (m⁻³ s⁻¹)</option></select></div>
<div class="control"><label for="scale">Colour scale</label><select id="scale"><option value="linear">linear</option><option value="log">log10</option></select></div>
<button id="theme" type="button" aria-pressed="false">Light theme</button>
</div><p class="small">Keyboard: 1–4 select cases; arrow keys move the map cursor; Home resets the cursor.</p></header>
<main>
<section class="metrics" id="metrics" aria-label="Case metrics"></section>
<section class="grid">
<div class="panel"><h2 id="mapTitle">Time-averaged map</h2><div class="canvas-wrap"><canvas id="field" tabindex="0" role="img" aria-label="Interactive (r,z) heatmap of the selected time-averaged quantity"></canvas><div id="tip" class="tip" role="status" aria-live="polite"></div></div><p class="small">Canvas raster of the node grid (radial-major). White: dielectric/outside the plasma cell mask. Straight bore wall at r = 2 mm is exact; the cone is a one-cell stair-step. Anode at z = 0 (fixed potential), exit plane at z = 24 mm (0 V reference).</p></div>
<aside class="panel"><h2 id="detailTitle">Case details</h2><div id="details"></div></aside>
</section>
<section class="plots">
<div class="panel"><h2>Macro-particle counts</h2><canvas class="plot" id="counts" role="img" aria-label="Electron and ion macro-particle counts versus time"></canvas></div>
<div class="panel"><h2>Currents</h2><canvas class="plot" id="currents" role="img" aria-label="Discharge, exit ion beam and wall currents versus time"></canvas></div>
<div class="panel"><h2>Potential range</h2><canvas class="plot" id="phi" role="img" aria-label="Minimum, mean and maximum potential versus time"></canvas></div>
<div class="panel"><h2>Energy ledger</h2><canvas class="plot" id="energy" role="img" aria-label="Kinetic, field and total energy with the interval ledger residual"></canvas><p class="small">Residual = Δ(K+U) − (injected − absorbed − inelastic + born-ion) kinetic energy per interval. It includes untracked electrode/injection electrostatic work and the momentum-conserving scheme's non-conservation; it is reported, not hidden.</p></div>
<div class="panel"><h2>Wall impact flux along the dielectric</h2><canvas class="plot" id="wall" role="img" aria-label="Electron and ion wall flux versus axial position"></canvas></div>
<div class="panel"><h2>Axial ion current density at the exit plane</h2><canvas class="plot" id="exit" role="img" aria-label="Exit-plane ion current density versus radius"></canvas></div>
<div class="panel wide"><h2>Stability metrics and convergence between cases</h2><canvas class="plot" id="wpe" role="img" aria-label="Peak plasma-frequency times timestep versus time"></canvas><div id="convergence"></div></div>
</section>
<section class="panel" style="margin:1rem 0"><h2>Simplifications (v1) and identity</h2><div id="identity"></div></section>
</main><footer>Self-contained offline dashboard generated by <code>modern/visualization/generate_pic2d_cft_snapshot.py</code>. Development/screening evidence only.</footer>
<script id="pic2d-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("pic2d-data").textContent);
const $=id=>document.getElementById(id);let selected=0,mapKey="n_e_per_m3",scaleMode="linear",cursor=null,raf=0;
const caseSelect=$("case");DATA.cases.forEach((c,i)=>{const o=document.createElement("option");o.value=i;o.textContent=c.label;caseSelect.append(o)});
const fmt=(v,n=4)=>v==null||!isFinite(v)?"–":Number(v).toLocaleString(undefined,{maximumSignificantDigits:n});
const sci=(v,n=3)=>v==null||!isFinite(v)?"–":Number(v).toExponential(n-1);
$("claim").innerHTML=`<strong>Claim boundary:</strong> ${DATA.claim_statement}<ul>${DATA.simplifications.map(s=>`<li>${s}</li>`).join("")}</ul>`;
const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function setup(canvas){const r=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);return {c,w:r.width,h:r.height}}
function color(t,signed){t=Math.max(0,Math.min(1,t));if(signed){if(t<.5){const q=t*2;return `rgb(${Math.round(35+220*q)},${Math.round(92+163*q)},255)`}const q=(t-.5)*2;return `rgb(255,${Math.round(255-210*q)},${Math.round(255-215*q)})`}return `rgb(${Math.round(12+240*t)},${Math.round(28+190*Math.sqrt(t))},${Math.round(90+100*(1-t))})`}
function renderMetrics(){const root=$("metrics");root.textContent="";DATA.cases.forEach((c,i)=>{const w=c.window_maps_summary,card=document.createElement("article");card.className="metric-card"+(i===selected?" active":"");card.tabIndex=0;card.setAttribute("role","button");card.setAttribute("aria-pressed",i===selected);card.innerHTML=`<h3>${c.label}</h3><div class="kv"><span>grid</span><span>${c.config.grid.radial_cells}×${c.config.grid.axial_cells}</span><span>macro weight</span><span>${sci(c.config.macro_weight,2)}</span><span>steps</span><span>${c.steps_completed}/${c.target_steps}</span><span>simulated</span><span>${fmt(c.simulated_time_s*1e9,3)} ns</span><span>stop</span><span>${c.stop_reason.replaceAll("_"," ")}</span><span>peak n_e</span><span>${sci(w.n_e_peak_per_m3)} m⁻³</span><span>φ range</span><span>${fmt(w.phi_min_v,3)}…${fmt(w.phi_max_v,3)} V</span><span>⟨T_e⟩_n</span><span>${fmt(w.t_e_density_weighted_mean_ev,3)} eV</span><span>I_d (final)</span><span>${fmt(c.final_series.currents_a.discharge_a*1e3,3)} mA</span><span>I_beam,i (window)</span><span>${fmt(w.exit_ion_current_a*1e3,3)} mA</span></div>`;card.onclick=()=>select(i);card.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select(i)}};root.append(card)})}
function renderDetails(){const c=DATA.cases[selected],g=c.stability_gate,op=DATA.protocol.operating_point;let html=`<div class="kv"><span>backend</span><span>${c.backend}</span><span>Δr × Δz</span><span>${fmt(c.config.grid.dr_m*1e6,3)} × ${fmt(c.config.grid.dz_m*1e6,3)} µm</span><span>Δt</span><span>${sci(c.config.dt_s,3)} s</span><span>wall time</span><span>${fmt(c.wall_seconds_run,4)} s</span><span>throughput</span><span>${fmt(c.steps_per_second,3)} steps/s</span><span>GPU util. samples</span><span>${c.gpu_utilisation_percent_samples.length?fmt(c.gpu_utilisation_percent_samples.reduce((a,b)=>a+b,0)/c.gpu_utilisation_percent_samples.length,3)+" %":"–"}</span><span>final e⁻ / Xe⁺ macro</span><span>${c.final_counts.electrons} / ${c.final_counts.ions}</span><span>window steps</span><span>${c.averaging_window_steps}${c.averaging_window_step_range?` (${c.averaging_window_step_range[0]}–${c.averaging_window_step_range[1]})`:""}</span><span>anode / exit</span><span>${c.config.potentials.anode_v} / ${c.config.potentials.exit_v} V</span><span>n_g (Xe)</span><span>${sci(op.neutral_density_per_m3,2)} m⁻³</span><span>e⁻ injection</span><span>${op.electron_injection_current_a} A @ ${op.electron_injection_temperature_ev} eV</span><span>seed plasma</span><span>${sci(op.seed_plasma_density_per_m3,2)} m⁻³ @ ${op.seed_electron_temperature_ev} eV</span></div>`;
html+=`<h2 style="margin-top:1rem">Stability gate (configured reference)</h2><div class="kv"><span>ω_pe Δt</span><span>${fmt(g.omega_pe_dt,3)}</span><span>Ω_ce Δt</span><span>${fmt(g.omega_ce_dt,3)}</span><span>cell / λ_D</span><span>${fmt(g.cell_debye_ratio,3)}</span><span>Courant</span><span>${fmt(g.particle_courant,3)}</span><span>P_coll</span><span>${sci(g.max_collision_probability,2)}</span><span>max |B| on nodes</span><span>${fmt(g.max_b_t*1e3,4)} mT</span></div>`;
if(c.stability_gate_message)html+=`<p class="small"><strong>Fail-closed stop:</strong> ${c.stability_gate_message}</p>`;
$("detailTitle").textContent=c.label;$("details").innerHTML=html;$("mapTitle").textContent=`${$("map").selectedOptions[0].textContent} — ${c.label}`;
const cv=DATA.convergence,rows=Object.entries(cv).map(([k,v])=>`<tr><td>${k}</td>${DATA.cases.map(cc=>`<td>${sci(v.values[cc.id])}</td>`).join("")}<td>${v.relative_spread==null?"–":fmt(v.relative_spread*100,3)+" %"}</td></tr>`).join("");
$("convergence").innerHTML=`<p class="small">Window-averaged summaries across resolution / particle-weight pairs. Relative spread = (max−min)/|mean|; this is a convergence <em>statement</em> for screening, not a verified convergence order.</p><table style="width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums" aria-label="Convergence between cases"><thead><tr><th style="text-align:left">quantity</th>${DATA.cases.map(cc=>`<th>${cc.label}</th>`).join("")}<th>spread</th></tr></thead><tbody>${rows}</tbody></table>`;
$("identity").innerHTML=`<p><span class="badge">status</span> ${DATA.status.replaceAll("_"," ")}</p><p><span class="badge">manifest SHA-256</span> <code>${DATA.manifest.file_sha256}</code></p><p><span class="badge">protocol SHA-256</span> <code>${DATA.manifest.protocol_sha256}</code></p><p><span class="badge">case summary SHA-256</span> <code>${c.summary_sha256}</code></p><p><span class="badge">maps npz SHA-256</span> <code>${c.maps_npz_sha256}</code></p><p><span class="badge">series npz SHA-256</span> <code>${c.series_npz_sha256}</code></p><p><span class="badge">P2 field map SHA-256</span> <code>${c.field.field_map_sha256}</code> (design ${c.field.provenance.design_id}, checkpoint <code>${c.field.provenance.checkpoint_file_sha256}</code>)</p><p><span class="badge">cross sections</span> ${c.cross_sections?c.cross_sections.provenance_status+" · payload <code>"+c.cross_sections.payload_sha256+"</code>":"–"}</p>`}
function bounds(w,h){return {l:58,t:18,r:w-78,b:h-46}}
function mapPoint(z,r,c,b){const zs=c.grid_z_m,rs=c.grid_r_m;return [b.l+(z-zs[0])/(zs.at(-1)-zs[0])*(b.r-b.l),b.b-(r-rs[0])/(rs.at(-1)-rs[0])*(b.b-b.t)]}
function drawField(){const c=DATA.cases[selected],s=setup($("field")),ctx=s.c,b=bounds(s.w,s.h),m=c.maps[mapKey],flat=m.flat().filter(v=>v!=null&&isFinite(v)),signed=mapKey==="phi_v"&&Math.min(...flat)<0;let lo,hi;const log=scaleMode==="log"&&!signed;if(log){const pos=flat.filter(v=>v>0);lo=Math.log10(Math.max(Math.min(...pos),Math.max(...pos)*1e-4));hi=Math.log10(Math.max(...pos))}else if(signed){const ma=Math.max(...flat.map(Math.abs));lo=-ma;hi=ma}else{lo=Math.min(...flat);hi=Math.max(...flat)}
ctx.clearRect(0,0,s.w,s.h);ctx.fillStyle=themeColor("--panel");ctx.fillRect(0,0,s.w,s.h);const off=document.createElement("canvas");off.width=c.grid_z_m.length;off.height=c.grid_r_m.length;const oc=off.getContext("2d"),img=oc.createImageData(off.width,off.height);
for(let i=0;i<off.height;i++)for(let j=0;j<off.width;j++){const v=m[off.height-1-i][j],k=(i*off.width+j)*4;if(v==null||!isFinite(v)||(log&&v<=0)){img.data[k]=img.data[k+1]=img.data[k+2]=245;img.data[k+3]=255;continue}const t=log?(Math.log10(v)-lo)/(hi-lo||1):(v-lo)/(hi-lo||1),rgb=color(t,signed).match(/\d+/g).map(Number);img.data[k]=rgb[0];img.data[k+1]=rgb[1];img.data[k+2]=rgb[2];img.data[k+3]=255}
oc.putImageData(img,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(off,b.l,b.t,b.r-b.l,b.b-b.t);
axes(ctx,b,s.w,s.h,"z (m)","r (m)",c.grid_z_m[0],c.grid_z_m.at(-1),c.grid_r_m[0],c.grid_r_m.at(-1));const x=s.w-52;ctx.font="11px system-ui";for(let k=0;k<80;k++){ctx.fillStyle=color(1-k/79,signed);ctx.fillRect(x,b.t+k*(b.b-b.t)/80,15,(b.b-b.t)/80+1)}ctx.fillStyle=themeColor("--text");ctx.fillText(log?"1e"+fmt(hi,3):sci(hi,3),x-14,b.t-5);ctx.fillText(log?"1e"+fmt(lo,3):sci(lo,3),x-14,b.b+14);
if(cursor){const p=mapPoint(cursor.z,cursor.r,c,b);ctx.strokeStyle="#fff";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p[0]-8,p[1]);ctx.lineTo(p[0]+8,p[1]);ctx.moveTo(p[0],p[1]-8);ctx.lineTo(p[0],p[1]+8);ctx.stroke()}}
function axes(c,b,w,h,xlabel,ylabel,xmin,xmax,ymin,ymax){c.strokeStyle=themeColor("--line");c.fillStyle=themeColor("--muted");c.lineWidth=1;c.font="12px system-ui";c.strokeRect(b.l,b.t,b.r-b.l,b.b-b.t);c.textAlign="center";for(let i=0;i<=4;i++){const x=b.l+(b.r-b.l)*i/4;c.fillText(fmt(xmin+(xmax-xmin)*i/4,3),x,b.b+18)}c.fillText(xlabel,(b.l+b.r)/2,h-6);c.save();c.translate(13,(b.t+b.b)/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="right";for(let i=0;i<=4;i++)c.fillText(fmt(ymax-(ymax-ymin)*i/4,3),b.l-6,b.t+(b.b-b.t)*i/4+4);c.textAlign="left"}
function nearest(values,v){let k=0;for(let i=1;i<values.length;i++)if(Math.abs(values[i]-v)<Math.abs(values[k]-v))k=i;return k}
function updateCursor(clientX,clientY){const canvas=$("field"),rect=canvas.getBoundingClientRect(),b=bounds(rect.width,rect.height),c=DATA.cases[selected],x=Math.max(b.l,Math.min(b.r,clientX-rect.left)),y=Math.max(b.t,Math.min(b.b,clientY-rect.top)),zs=c.grid_z_m,rs=c.grid_r_m,zi=nearest(zs,zs[0]+(x-b.l)/(b.r-b.l)*(zs.at(-1)-zs[0])),ri=nearest(rs,rs[0]+(b.b-y)/(b.b-b.t)*(rs.at(-1)-rs[0]));cursor={zi,ri,z:zs[zi],r:rs[ri]};showTip(clientX-rect.left,clientY-rect.top);schedule(false)}
function showTip(x,y){const c=DATA.cases[selected],t=$("tip");if(!cursor){t.style.display="none";return}const v=c.maps[mapKey][cursor.ri][cursor.zi];t.textContent=`z ${fmt(cursor.z*1e3,4)} mm · r ${fmt(cursor.r*1e3,4)} mm · ${v==null?"outside plasma":sci(v,4)} · n_e ${sci(c.maps.n_e_per_m3[cursor.ri][cursor.zi],3)} · φ ${fmt(c.maps.phi_v[cursor.ri][cursor.zi],4)} V · T_e ${fmt(c.maps.t_e_ev[cursor.ri][cursor.zi],3)} eV`;t.style.display="block";t.style.left=Math.min(x+12,t.parentElement.clientWidth-t.offsetWidth-5)+"px";t.style.top=Math.max(4,y-36)+"px"}
function drawPlot(id,series,xLabel,yLabel,log=false){const s=setup($(id)),c=s.c,b={l:64,t:16,r:s.w-16,b:s.h-40},pts=series.filter(q=>q.x.length);if(!pts.length){c.clearRect(0,0,s.w,s.h);return}const all=pts.flatMap(q=>q.y.filter(v=>v!=null&&isFinite(v)&&(!log||v>0))),xmin=Math.min(...pts.flatMap(q=>q.x)),xmax=Math.max(...pts.flatMap(q=>q.x));let ymin=Math.min(...all),ymax=Math.max(...all);if(log){ymin=Math.log10(Math.max(ymin,1e-300));ymax=Math.log10(Math.max(ymax,1e-299))}else{const pad=(ymax-ymin||1)*.08;ymin-=pad;ymax+=pad}c.clearRect(0,0,s.w,s.h);c.fillStyle=themeColor("--panel");c.fillRect(0,0,s.w,s.h);axes(c,b,s.w,s.h,xLabel,yLabel,xmin,xmax,ymin,ymax);pts.forEach((q,k)=>{c.strokeStyle=q.color;c.lineWidth=q.width||1.6;c.beginPath();let started=false;q.x.forEach((x,i)=>{const v=q.y[i];if(v==null||!isFinite(v)||(log&&v<=0)){started=false;return}const yy=log?Math.log10(v):v,px=b.l+(x-xmin)/(xmax-xmin||1)*(b.r-b.l),py=b.b-(yy-ymin)/(ymax-ymin||1)*(b.b-b.t);started?c.lineTo(px,py):c.moveTo(px,py);started=true});c.stroke();c.fillStyle=q.color;c.fillText(q.name,b.l+8,b.t+14+k*15)})}
function drawSeries(){const c=DATA.cases[selected],S=c.series,t=S.time_s.map(v=>v*1e9),cur=k=>S["current_"+k]||[];drawPlot("counts",[{x:t,y:S.electrons,name:"electrons",color:"#5ad6c0"},{x:t,y:S.ions,name:"Xe⁺",color:"#ff6b6b"}],"t (ns)","macro-particles");
drawPlot("currents",[{x:t,y:cur("discharge_a").map(v=>v*1e3),name:"discharge (anode)",color:"#5ad6c0"},{x:t,y:cur("exit_ion_beam_a").map(v=>v*1e3),name:"exit ion beam",color:"#ff6b6b"},{x:t,y:cur("wall_electron_a").map(v=>v*1e3),name:"wall e⁻",color:"#58a8ff"},{x:t,y:cur("wall_ion_a").map(v=>v*1e3),name:"wall Xe⁺",color:"#ffcf67"},{x:t,y:cur("exit_electron_a").map(v=>v*1e3),name:"exit e⁻",color:"#c58bff"}],"t (ns)","current (mA)");
drawPlot("phi",[{x:t,y:S.phi_max_v,name:"max φ",color:"#ff6b6b"},{x:t,y:S.phi_mean_v,name:"mean φ (plasma nodes)",color:"#5ad6c0"},{x:t,y:S.phi_min_v,name:"min φ",color:"#58a8ff"}],"t (ns)","φ (V)");
drawPlot("energy",[{x:t,y:S.total_energy_j,name:"K+U total",color:"#eef7f4"},{x:t,y:S.kinetic_electron_j,name:"K electrons",color:"#5ad6c0"},{x:t,y:S.kinetic_ion_j,name:"K ions",color:"#ff6b6b"},{x:t,y:S.field_energy_j,name:"U field",color:"#58a8ff"},{x:t,y:S.interval_residual_j.map(Math.abs),name:"|interval residual|",color:"#ffcf67"}],"t (ns)","energy (J)",true);
const wz=c.wall_z_m.map(v=>v*1e3);drawPlot("wall",[{x:wz,y:c.wall.wall_electron_flux_per_m2_s,name:"electron flux (m⁻² s⁻¹)",color:"#5ad6c0"},{x:wz,y:c.wall.wall_ion_flux_per_m2_s,name:"ion flux (m⁻² s⁻¹)",color:"#ff6b6b"}],"z (mm)","flux",true);
drawPlot("exit",[{x:c.exit_r_m.map(v=>v*1e3),y:c.exit.exit_ion_current_density_a_per_m2,name:"ion j_z (A m⁻²)",color:"#ff6b6b"},{x:c.exit_r_m.map(v=>v*1e3),y:c.exit.exit_electron_current_density_a_per_m2,name:"electron j_z (A m⁻²)",color:"#5ad6c0"}],"r (mm)","A/m²");
drawPlot("wpe",[{x:t,y:S.peak_omega_pe_dt,name:"peak ω_pe Δt (gate 0.2)",color:"#ffcf67"}],"t (ns)","ω_pe Δt")}
function drawAll(){renderMetrics();renderDetails();drawField();drawSeries()}
function schedule(full=true){cancelAnimationFrame(raf);raf=requestAnimationFrame(full?drawAll:drawField)}
function select(i){selected=i;caseSelect.value=i;cursor=null;showTip();schedule()}
caseSelect.onchange=()=>select(Number(caseSelect.value));$("map").onchange=e=>{mapKey=e.target.value;schedule()};$("scale").onchange=e=>{scaleMode=e.target.value;schedule(false)};
$("theme").onclick=()=>{const light=document.documentElement.dataset.theme!=="light";document.documentElement.dataset.theme=light?"light":"dark";$("theme").textContent=light?"Dark theme":"Light theme";$("theme").setAttribute("aria-pressed",light);schedule()};
$("field").addEventListener("pointermove",e=>updateCursor(e.clientX,e.clientY));$("field").addEventListener("pointerleave",()=>{cursor=null;showTip();schedule(false)});
$("field").addEventListener("keydown",e=>{const c=DATA.cases[selected],zs=c.grid_z_m,rs=c.grid_r_m;if(e.key==="Home"){cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]}}else{if(!cursor)cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]};if(e.key==="ArrowLeft")cursor.zi=Math.max(0,cursor.zi-1);else if(e.key==="ArrowRight")cursor.zi=Math.min(zs.length-1,cursor.zi+1);else if(e.key==="ArrowDown")cursor.ri=Math.max(0,cursor.ri-1);else if(e.key==="ArrowUp")cursor.ri=Math.min(rs.length-1,cursor.ri+1);else return;cursor.z=zs[cursor.zi];cursor.r=rs[cursor.ri]}e.preventDefault();showTip(70,30);schedule(false)});
window.addEventListener("keydown",e=>{if(["INPUT","SELECT","BUTTON"].includes(e.target.tagName))return;const k=Number(e.key);if(k>=1&&k<=DATA.cases.length)select(k-1)});new ResizeObserver(schedule).observe(document.querySelector("main"));window.addEventListener("pageshow",schedule);drawAll();
</script></body></html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", encoded)


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
