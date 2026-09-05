"""Generate the standalone Hybrid L2 v2 dashboard (``hybrid-l2-v2.html``).

Headline: the preregistered comparison of the per-cell hybrid (``experiments/hybrid_l2_v2``) against the
accepted PIC steady-state v2 base plateau.  Every embedded input is hash-verified against its
``.sha256.json`` sidecar (assessment, every case summary, the L2 and PIC maps); the comparison statuses
are recomputed from value / reference / tolerance and must agree with the assessment; the verdict must
be the one the gate functions produce from the embedded metrics.  No timestamps or paths of the
generating machine are embedded (identical inputs give identical bytes); the page is self-contained
(no network access, SVG/canvas only) and states its claim boundary on every view.

The dashboard is built from the TRACKED record only: the base case carries ``series.npz``; the four
other finished cases are recorded by their ``summary.json`` alone (PARKED record, 2026-09-04), so their
series are not embedded (``cases[<name>]["series"] is None``) and the series panel draws the base case.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

MODERN = Path(__file__).resolve().parents[1]
SRC = MODERN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MODERN) not in sys.path:
    sys.path.insert(0, str(MODERN))

from cft_revival.hybrid import gates  # noqa: E402

EXPERIMENT = MODERN / "experiments" / "hybrid_l2_v2"
RESULTS = EXPERIMENT / "results"
PIC_V2 = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results"
DEFAULT_OUTPUT = Path(__file__).with_name("hybrid-l2-v2.html")
SCHEMA = "cft-hybrid-l2-v2-visualization/0.1.0"
MAX_HTML_BYTES = 2_500_000
MAP_STRIDE_R = 2
MAP_STRIDE_Z = 4
MAX_SERIES_POINTS = 600
CLASSIFICATION = "per_cell_hybrid_kinetic_ions_boltzmann_electron_cells_self_consistent_electrostatics_development_not_validated"
VERDICTS = ("accepted", "rejected_on_comparison", "not_evaluable")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _verified_json(path: Path) -> dict[str, Any]:
    """Load a canonical JSON artifact only if its sidecar byte hash matches."""

    raw = path.read_bytes()
    sidecar = json.loads(path.with_name(path.name + ".sha256.json").read_text(encoding="utf-8"))
    if sidecar["byte_sha256"] != sha256(raw).hexdigest():
        raise ValueError(f"{path.name}: sidecar SHA-256 mismatch")
    return json.loads(raw.decode("utf-8"))


def _verified_npz(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    sidecar = json.loads(path.with_name(path.name + ".sha256.json").read_text(encoding="utf-8"))
    if sidecar["byte_sha256"] != sha256(raw).hexdigest():
        raise ValueError(f"{path.name}: sidecar SHA-256 mismatch")
    import io

    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            return None
        return float(f"{f:.{digits}g}")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return _round(value.tolist(), digits)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _decimate(values: np.ndarray, count: int = MAX_SERIES_POINTS) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] <= count:
        return values
    index = np.unique(np.linspace(0, values.shape[0] - 1, count).astype(int))
    return values[index]


def _map_block(maps: Mapping[str, np.ndarray], grid: Mapping[str, Any]) -> dict[str, Any]:
    phi = np.asarray(maps["phi_v"])[::MAP_STRIDE_R, ::MAP_STRIDE_Z]
    n_e = np.asarray(maps["n_e_per_m3"])[::MAP_STRIDE_R, ::MAP_STRIDE_Z]
    n_i = np.asarray(maps["n_i_per_m3"])[::MAP_STRIDE_R, ::MAP_STRIDE_Z]
    return {
        "dr_m": float(grid["dr_m"]) * MAP_STRIDE_R, "dz_m": float(grid["dz_m"]) * MAP_STRIDE_Z,
        "shape": list(phi.shape),
        "phi_v": _round(phi, 5), "log10_n_e": _round(np.where(n_e > 0, np.log10(np.maximum(n_e, 1.0)), 0.0), 4),
        "log10_n_i": _round(np.where(n_i > 0, np.log10(np.maximum(n_i, 1.0)), 0.0), 4),
        "wall_ion_flux_per_m2_s": _round(np.asarray(maps["wall_ion_flux_per_m2_s"]), 4),
        "wall_electron_flux_per_m2_s": _round(np.asarray(maps["wall_electron_flux_per_m2_s"]), 4),
        "axis_phi_v": _round(np.asarray(maps["phi_v"])[0, :], 5),
    }


def build_payload(experiment: Path = EXPERIMENT, pic_v2: Path = PIC_V2) -> dict[str, Any]:
    results = experiment / "results"
    assessment = _verified_json(results / "assessment.json")
    protocol = json.loads((experiment / "protocol.json").read_text(encoding="utf-8"))
    cases: dict[str, Any] = {}
    for name in protocol["cases"]:
        directory = results if name == "base" else experiment / f"results-{name}"
        if not (directory / "summary.json").is_file():
            cases[name] = {"finished": False}
            continue
        summary = _verified_json(directory / "summary.json")
        k_cells = len(protocol["cells"]["partition"]["cells"])
        case: dict[str, Any] = {
            "finished": True, "summary_sha256": _file_sha256(directory / "summary.json"), "stop_reason": summary["stop_reason"],
            "steps": summary["steps_completed"], "simulated_time_s": summary["simulated_time_s"], "ion_transit_times": summary["ion_transit_times"],
            "wall_seconds": summary["wall_seconds_total"], "ms_per_step": summary["ms_per_step"], "case": summary["case"],
            "window_currents_a": summary["window_currents_a"], "cells": summary["cells"], "plateau": summary["plateau"],
            "windowed_energy_residual": summary["windowed_energy_residual"], "charge_identity_max_relative": summary["charge_identity_max_relative"],
            "neutral_ledger_closure_relative": summary["neutral_inventory"]["ledger_closure"]["closure_relative_to_inventory"],
            "final_counts": summary["final_counts"], "peak_n_e_per_m3": summary["window_maps_summary"]["peak_n_e_per_m3"],
            "series": None,
        }
        if name == "base":
            # only the base case's series.npz is part of the tracked record (see the module docstring)
            series = _verified_npz(directory / "series.npz")
            case["series"] = _round({
                "time_us": _decimate(series["time_s"]) * 1e6, "discharge_ma": _decimate(series["current_discharge_a"]) * 1e3,
                "beam_ma": _decimate(series["current_exit_ion_beam_a"]) * 1e3, "wall_ion_ma": _decimate(series["current_wall_ion_a"]) * 1e3,
                "ionization_rate_per_s": _decimate(series["current_ionization_rate_per_s"]), "neutral_density_per_m3": _decimate(series["neutral_density_per_m3"]),
                "electrons": _decimate(series["electrons"]), "ions": _decimate(series["ions"]),
                "cell_potential_v": [_decimate(series[f"cell{k}_potential_v"]) for k in range(k_cells)],
                "cell_temperature_ev": [_decimate(series[f"cell{k}_temperature_ev"]) for k in range(k_cells)],
                "residual_ratio": _decimate(np.where(series["interval_electrode_work_j"] != 0.0, series["interval_residual_j"] / np.where(series["interval_electrode_work_j"] != 0.0, series["interval_electrode_work_j"], 1.0), 0.0)),
            }, 5)
        cases[name] = case
    base_summary = _verified_json(results / "summary.json")
    l2_maps = _map_block(_verified_npz(results / "maps.npz"), base_summary["provenance"]["config"]["grid"])
    pic_summary = _verified_json(pic_v2 / "summary.json")
    pic_maps = _map_block(_verified_npz(pic_v2 / "maps.npz"), pic_summary["provenance"]["config"]["grid"])
    comparison = assessment["code_comparison"]
    payload = {
        "schema": SCHEMA, "classification": CLASSIFICATION, "experiment_id": protocol["experiment_id"], "model_version": protocol["model_version"],
        "identity": {"assessment_sha256": _file_sha256(results / "assessment.json"), "protocol_sha256": _file_sha256(experiment / "protocol.json"),
                     "pic_base_maps_sha256": _file_sha256(pic_v2 / "maps.npz"), "l2_base_maps_sha256": _file_sha256(results / "maps.npz"),
                     "prereg_commit_hint": base_summary.get("git_head")},
        "verdict": assessment["gate_l2"]["verdict"], "metrics": assessment["gate_l2"]["metrics"],
        "interface_conservation": assessment["interface_conservation"], "comparison": comparison,
        "spatial_levels": assessment["spatial_levels"], "temporal_levels": assessment["temporal_levels"],
        "statistical_levels": assessment["statistical_levels"], "input_levels": assessment["input_levels"],
        "uncertainty": assessment["uncertainty"], "cost": assessment["cost"], "pic_v4": assessment.get("pic_v4"),
        "closures": {k: protocol["closures"][k] for k in ("cusp_conductance_s", "leak_half_width_m", "access_floor", "pressure_term")},
        "partition": protocol["cells"]["partition"], "cases": cases, "maps": {"l2": l2_maps, "pic": pic_maps},
        "pic_reference": {k: {kk: v[kk] for kk in ("reference", "band", "tolerance", "status")} for k, v in protocol["pic_reference"]["quantities"].items()},
        "claim_boundary": protocol["claim_boundary"], "prohibited_until_accepted": protocol["prohibited_until_accepted"],
        "paper_admission": "not admitted - no paper claim references this experiment",
    }
    payload = _round(payload, 7)
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("classification") != CLASSIFICATION:
        raise ValueError("dashboard payload schema / classification differ")
    if payload["verdict"] not in VERDICTS:
        raise ValueError("unknown verdict")
    if payload["paper_admission"] != "not admitted - no paper claim references this experiment":
        raise ValueError("paper admission wording changed")
    comparison = payload["comparison"]
    recomputed = [gates.compare(c["name"], c["value"], c["reference"], c["tolerance"]) for c in comparison["comparisons"]]
    for stored, fresh in zip(comparison["comparisons"], recomputed, strict=True):
        if stored["status"] != fresh.status:
            raise ValueError(f"comparison status of {stored['name']} does not reproduce")
    block = gates.code_comparison(recomputed)
    if block["passed"] != comparison["passed"] or sorted(block["outside"]) != sorted(comparison["outside"]):
        raise ValueError("code comparison verdict does not reproduce")
    metrics = payload["metrics"]
    structural = metrics["interface_conservation_passed"] and metrics["spatial_levels"] >= 3 and metrics["temporal_levels"] >= 3 and metrics["numerical_uncertainty_reported"]
    expected = "accepted" if structural and metrics["code_comparison_passed"] else "rejected_on_comparison" if structural else "not_evaluable"
    if payload["verdict"] != expected:
        raise ValueError("verdict does not follow from the metrics")
    if sorted(metrics["uncertainty_components"]) != ["emulator", "input", "model_discrepancy", "numerical"]:
        raise ValueError("uncertainty components incomplete")
    for name, case in payload["cases"].items():
        if case.get("finished") and case["stop_reason"] == "plateau_reached_after_min_transit_times" and not case["plateau"]["reached"]:
            raise ValueError(f"{name}: plateau stop without a reached plateau")


def _script_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid L2 v2 - per-cell hybrid vs the PIC base plateau</title>
<style>
:root{--bg:#0f1318;--panel:#161c24;--ink:#e6edf3;--muted:#9aa7b4;--ok:#2ea043;--bad:#d1494b;--warn:#d29922;--line:#2b3440}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line)} h1{font-size:20px;margin:0 0 6px} h2{font-size:16px;margin:0 0 8px}
.chips span{display:inline-block;padding:2px 8px;border-radius:10px;margin-right:6px;background:#22303c;color:var(--ink);font-size:12px}
.chips .ok{background:var(--ok)} .chips .bad{background:var(--bad)} .chips .warn{background:var(--warn);color:#111}
main{padding:12px 20px;display:grid;grid-template-columns:1fr 1fr;gap:14px} section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
section.wide{grid-column:1/-1} table{border-collapse:collapse;width:100%;font-size:12px} th,td{padding:3px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} tr.within td{color:#9be49f} tr.outside td{color:#ff9b9d} tr.nc td{color:var(--muted)}
canvas{width:100%;height:auto;background:#0b0f14;border-radius:6px} .muted{color:var(--muted)} .boundary{font-size:12px;color:var(--muted);border-top:1px dashed var(--line);padding-top:8px;margin-top:8px}
#jserrors{color:var(--bad);white-space:pre-wrap;font-size:12px} .small{font-size:11px}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Hybrid L2 v2 - per-cell hybrid on the reference material-aware field vs the accepted PIC base plateau</h1>
<div class="chips" id="chips"></div><div id="headline" class="muted"></div><div id="jserrors"></div></header>
<main>
<section class="wide"><h2>GATE-L2 metric constraints (paper/evidence/result-gates.json)</h2><table id="gates"></table></section>
<section class="wide"><h2>Code comparison: L2 base case vs PIC v2 base plateau (tolerance = clip(2 x PIC particle band, 5 %, 12 %))</h2><table id="comparison"></table></section>
<section><h2>Cell potentials and temperatures (L2 vs PIC)</h2><canvas id="cells" width="720" height="360"></canvas></section>
<section><h2>Axis potential (L2 vs PIC) and wall ion flux</h2><canvas id="axis" width="720" height="360"></canvas></section>
<section><h2>L2 base: potential and log10 n_e</h2><canvas id="mapl2" width="720" height="420"></canvas></section>
<section><h2>PIC base plateau: potential and log10 n_e</h2><canvas id="mappic" width="720" height="420"></canvas></section>
<section class="wide"><h2>Series of the base case (I_d, S, n_g, cell T_e, energy residual) - the only case whose series.npz is part of the tracked record</h2><canvas id="series" width="1440" height="420"></canvas></section>
<section class="wide"><h2>Refinement families, input sensitivity and cost</h2><table id="levels"></table><div id="cost" class="muted small"></div></section>
<section class="wide"><h2>Claim boundary</h2><div id="boundary" class="boundary"></div></section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
const sink=document.getElementById('jserrors');window.addEventListener('error',e=>{sink.textContent+=(e.message||e)+'\n'});
const D=JSON.parse(document.getElementById('payload').textContent);
const fmt=(v,d=3)=>v===null||v===undefined?'-':(Math.abs(v)>=1e4||Math.abs(v)<1e-3&&v!==0?Number(v).toExponential(d-1):Number(v).toPrecision(d));
const pct=v=>v===null||v===undefined?'-':(100*v).toFixed(1)+' %';
const chips=document.getElementById('chips');const m=D.metrics;
function chip(t,c){const s=document.createElement('span');s.textContent=t;if(c)s.className=c;chips.appendChild(s)}
chip('verdict: '+D.verdict,D.verdict==='accepted'?'ok':D.verdict==='rejected_on_comparison'?'warn':'bad');
chip('conservation '+(m.interface_conservation_passed?'pass':'FAIL'),m.interface_conservation_passed?'ok':'bad');
chip('comparison '+(m.code_comparison_passed?'pass':'FAIL')+' ('+D.comparison.compared+' compared, '+D.comparison.outside.length+' outside)',m.code_comparison_passed?'ok':'bad');
chip('spatial levels '+m.spatial_levels+'/3',m.spatial_levels>=3?'ok':'bad');chip('temporal levels '+m.temporal_levels+'/3',m.temporal_levels>=3?'ok':'bad');
chip('failed cases '+m.failed_cases_count,m.failed_cases_count===0?'ok':'warn');chip('development model - not validated','warn');chip(D.paper_admission);
const base=D.cases.base;const wc=base.window_currents_a;
document.getElementById('headline').textContent='L2 base: I_d '+fmt(wc.discharge_a*1e3)+' mA, S '+fmt(wc.ionization_rate_per_s)+' /s, beam '+fmt(wc.exit_ion_beam_a*1e3)+' mA, '+base.steps+' steps ('+fmt(base.simulated_time_s*1e6)+' us, '+fmt(base.ion_transit_times,2)+' transits) in '+fmt(base.wall_seconds/60,3)+' min on one CPU process; PIC base plateau: I_d 3.444 mA, S 3.93e16 /s, beam 2.29 mA in '+fmt(D.cost.pic_base_wall_seconds/3600,3)+' h on an RTX 5090. Wall-clock ratio PIC/L2 = '+fmt(D.cost.wall_clock_ratio_pic_over_l2,3)+'.';
const g=document.getElementById('gates');const ic=D.interface_conservation.checks;
const rows=[['interface_conservation_passed',m.interface_conservation_passed,'charge identity '+fmt(ic.charge_identity.value,2)+' <= '+ic.charge_identity.bound+'; atoms '+fmt(ic.neutral_ledger.value,2)+'; energy window '+pct(ic.energy_residual_window.value)+' (|.| <= '+pct(ic.energy_residual_window.bound)+'); plateau '+ic.plateau.value],
['spatial_levels >= 3',m.spatial_levels>=3,m.spatial_levels+' finished: '+D.spatial_levels.labels.join(', ')],['temporal_levels >= 3',m.temporal_levels>=3,m.temporal_levels+' finished: '+D.temporal_levels.labels.join(', ')],
['code_comparison_passed',m.code_comparison_passed,D.comparison.compared+' compared; outside: '+(D.comparison.outside.join(', ')||'none')+'; not compared: '+D.comparison.not_compared.length],
['numerical_uncertainty_reported',m.numerical_uncertainty_reported,'spatial / temporal / statistical spreads of the level quantities'],['failed_cases_count',true,String(m.failed_cases_count)],
['uncertainty_components',true,m.uncertainty_components.join(', ')]];
g.innerHTML='<tr><th>constraint</th><th>status</th><th>reading</th></tr>'+rows.map(r=>'<tr class="'+(r[1]?'within':'outside')+'"><td>'+r[0]+'</td><td>'+(r[1]?'pass':'FAIL')+'</td><td style="text-align:left;white-space:normal">'+r[2]+'</td></tr>').join('');
const c=document.getElementById('comparison');const v4=D.pic_v4;
c.innerHTML='<tr><th>quantity</th><th>L2 base</th><th>PIC base</th><th>PIC band</th><th>tolerance</th><th>L2-PIC</th><th>status</th>'+(v4?'<th>PIC v4 (33 um, informational)</th><th>v4-base</th>':'')+'</tr>'+D.comparison.comparisons.map(r=>{const ref=D.pic_reference[r.name]||{};const cls=r.status==='within'?'within':r.status==='outside'?'outside':'nc';
return '<tr class="'+cls+'"><td>'+r.name+'</td><td>'+fmt(r.value,4)+'</td><td>'+fmt(r.reference,4)+'</td><td>'+pct(ref.band)+'</td><td>'+pct(r.tolerance)+'</td><td>'+pct(r.relative_difference)+'</td><td>'+r.status+'</td>'+(v4?'<td>'+fmt(v4.quantities[r.name],4)+'</td><td>'+pct(v4.v4_relative_to_base[r.name])+'</td>':'')+'</tr>'}).join('');
function bars(id){const cv=document.getElementById(id),x=cv.getContext('2d');x.fillStyle='#0b0f14';x.fillRect(0,0,cv.width,cv.height);const K=D.partition.cells.length;
const l2phi=[],picphi=[],l2t=[],pict=[];for(let k=0;k<K;k++){l2phi.push(base.cells['cell'+k].potential_v);l2t.push(base.cells['cell'+k].temperature_ev);picphi.push(D.pic_reference['cell'+k+'_potential_v'].reference);pict.push(D.pic_reference['cell'+k+'_temperature_ev'].reference)}
const panel=(x0,w,l2,pic,title,unit)=>{const mx=Math.max(...l2,...pic)*1.15;x.fillStyle='#e6edf3';x.font='13px sans-serif';x.fillText(title,x0+8,18);const bw=w/(K*3);for(let k=0;k<K;k++){const h1=l2[k]/mx*280,h2=pic[k]/mx*280;x.fillStyle='#4ea1ff';x.fillRect(x0+bw*(3*k+0.5),320-h1,bw*0.9,h1);x.fillStyle='#ffb347';x.fillRect(x0+bw*(3*k+1.5),320-h2,bw*0.9,h2);x.fillStyle='#9aa7b4';x.font='11px sans-serif';x.fillText('cell '+(k+1),x0+bw*(3*k+0.6),338);x.fillText(l2[k].toFixed(unit==='V'?0:1),x0+bw*(3*k+0.5),316-h1);x.fillText(pic[k].toFixed(unit==='V'?0:1),x0+bw*(3*k+1.5),316-h2)}};
panel(0,340,l2phi,picphi,'density-weighted cell potential [V]  blue L2 / orange PIC','V');panel(370,340,l2t,pict,'cell electron temperature [eV]  blue L2 / orange PIC','eV')}
bars('cells');
function axis(){const cv=document.getElementById('axis'),x=cv.getContext('2d');x.fillStyle='#0b0f14';x.fillRect(0,0,cv.width,cv.height);const L=D.maps.l2,P=D.maps.pic;
const draw=(arr,dz,col,y0,h,vmin,vmax)=>{x.strokeStyle=col;x.lineWidth=1.5;x.beginPath();arr.forEach((v,i)=>{const px=40+i*dz/0.024*660,py=y0+h-(v-vmin)/(vmax-vmin)*h;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()};
x.fillStyle='#e6edf3';x.font='12px sans-serif';x.fillText('axis potential [V] (blue L2, orange PIC); cusp planes dashed',48,16);
const pmax=Math.max(...L.axis_phi_v,...P.axis_phi_v);draw(L.axis_phi_v,L.dz_m/MAPZ(L),'#4ea1ff',24,140,-20,pmax*1.05);draw(P.axis_phi_v,P.dz_m/MAPZ(P),'#ffb347',24,140,-20,pmax*1.05);
x.setLineDash([4,4]);x.strokeStyle='#ffffff88';D.partition.cusp_z_m.forEach(z=>{const px=40+z/0.024*660;x.beginPath();x.moveTo(px,24);x.lineTo(px,340);x.stroke()});x.setLineDash([]);
x.fillStyle='#e6edf3';x.fillText('wall ion flux [log10 m^-2 s^-1] per axial cell (blue L2, orange PIC)',48,190);
const lw=L.wall_ion_flux_per_m2_s.map(v=>v>0?Math.log10(v):17),pw=P.wall_ion_flux_per_m2_s.map(v=>v>0?Math.log10(v):17);const dzL=0.024/lw.length,dzP=0.024/pw.length;
draw(lw,dzL,'#4ea1ff',200,140,17,21.5);draw(pw,dzP,'#ffb347',200,140,17,21.5);x.fillStyle='#9aa7b4';x.fillText('0 mm',40,356);x.fillText('24 mm',680,356)}
function MAPZ(M){return 1}
axis();
function heat(id,M,title){const cv=document.getElementById(id),x=cv.getContext('2d');x.fillStyle='#0b0f14';x.fillRect(0,0,cv.width,cv.height);const [nr,nz]=M.shape;
const cell=(arr,y0,h,vmin,vmax,label)=>{const W=680/nz,H=h/nr;for(let i=0;i<nr;i++)for(let j=0;j<nz;j++){const v=arr[i][j];if(v===null||v===0&&label!=='phi')continue;const t=Math.max(0,Math.min(1,(v-vmin)/(vmax-vmin)));x.fillStyle='hsl('+(240-240*t)+',80%,'+(25+35*t)+'%)';x.fillRect(40+j*W,y0+h-(i+1)*H,W+0.5,H+0.5)}x.fillStyle='#e6edf3';x.font='12px sans-serif';x.fillText(label,48,y0-4)};
const pm=Math.max(...M.phi_v.flat());cell(M.phi_v,20,170,-20,pm,'phi [V] '+title+' (0..'+pm.toFixed(0)+' V)');const ne=M.log10_n_e;cell(ne,220,170,15,18.5,'log10 n_e [m^-3] '+title+' (15..18.5)');
x.setLineDash([3,3]);x.strokeStyle='#ffffffaa';D.partition.cusp_z_m.forEach(z=>{const px=40+z/0.024*680;x.beginPath();x.moveTo(px,20);x.lineTo(px,390);x.stroke()});x.setLineDash([])}
heat('mapl2',D.maps.l2,'L2');heat('mappic',D.maps.pic,'PIC');
function series(){const cv=document.getElementById('series'),x=cv.getContext('2d');x.fillStyle='#0b0f14';x.fillRect(0,0,cv.width,cv.height);const names=Object.keys(D.cases).filter(n=>D.cases[n].finished&&D.cases[n].series);
const cols=['#4ea1ff','#ffb347','#9be49f','#ff9b9d','#c9a0ff','#7fd8ff','#ffd27f','#b8f2b0','#ffb3c1','#d0b3ff','#a0e0ff'];
const panel=(x0,w,key,title,ymin,ymax,pick)=>{x.fillStyle='#e6edf3';x.font='12px sans-serif';x.fillText(title,x0+6,14);const tmax=Math.max(...names.map(n=>D.cases[n].series.time_us.slice(-1)[0]));
names.forEach((n,ci)=>{const s=D.cases[n].series;const ys=pick?pick(s):s[key];x.strokeStyle=cols[ci%cols.length];x.lineWidth=n==='base'?2:1;x.beginPath();ys.forEach((v,i)=>{const px=x0+s.time_us[i]/tmax*w,py=400-(v-ymin)/(ymax-ymin)*370;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()});
x.fillStyle='#9aa7b4';x.fillText(ymin.toPrecision(2),x0+2,400);x.fillText(ymax.toPrecision(2),x0+2,30)};
const all=k=>names.flatMap(n=>D.cases[n].series[k]);const mx=k=>Math.max(...all(k));
panel(0,270,'discharge_ma','I_d [mA]',0,mx('discharge_ma')*1.05);panel(290,270,'ionization_rate_per_s','S [1/s]',0,mx('ionization_rate_per_s')*1.05);panel(580,270,'neutral_density_per_m3','n_g [m^-3]',0,mx('neutral_density_per_m3')*1.05);
panel(870,270,'cell_temperature_ev','T_e cell 3 [eV]',0,Math.max(...names.map(n=>Math.max(...D.cases[n].series.cell_temperature_ev[2])))*1.05,s=>s.cell_temperature_ev[2]);panel(1160,270,'residual_ratio','energy residual / electrode work',-0.5,0.5);
names.forEach((n,ci)=>{x.fillStyle=cols[ci%cols.length];x.fillText(n,10+ci*125,416)})}
series();
const lv=document.getElementById('levels');const fam=[['spatial',D.spatial_levels],['temporal',D.temporal_levels],['statistical',D.statistical_levels],['input (closure x 0.7 / 1.3)',D.input_levels]];
lv.innerHTML='<tr><th>family</th><th>finished levels</th><th>quantity</th><th>values (level order)</th><th>max relative spread</th></tr>'+fam.flatMap(([f,b])=>Object.entries(b.spread).map(([q,s])=>'<tr><td>'+f+'</td><td>'+b.labels.join(', ')+'</td><td>'+q+'</td><td>'+s.values.map(v=>fmt(v,4)).join(', ')+'</td><td>'+pct(s.max_relative_spread)+'</td></tr>')).join('');
document.getElementById('cost').textContent='Cost: L2 base '+fmt(D.cost.l2_wall_seconds,4)+' s ('+fmt(D.cost.l2_ms_per_step,3)+' ms/step, '+D.cost.l2_steps+' steps, one CPU process, numpy) vs PIC base '+fmt(D.cost.pic_base_wall_seconds,5)+' s ('+D.cost.pic_base_steps+' steps on '+D.cost.pic_base_device+'). '+D.cost.note+(v4?' PIC v4 (33 um) finished: '+v4.stop_reason+' after '+v4.steps+' steps / '+fmt(v4.wall_seconds_total,4)+' s; '+v4.note:'');
document.getElementById('boundary').textContent=D.claim_boundary+' Prohibited until GATE-L2 is accepted: '+D.prohibited_until_accepted.join('; ')+'. Closures from the PIC base plateau: cusp conductances '+D.closures.cusp_conductance_s.map(v=>fmt(v,3)).join(' / ')+' S, leak half-widths '+D.closures.leak_half_width_m.map(v=>(v*1e3).toFixed(3)).join(' / ')+' mm. '+D.paper_admission+'.';
})();
</script></body></html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    return HTML_TEMPLATE.replace("__PAYLOAD__", _script_json(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_payload()
    html = render_html(payload)
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        raise ValueError("dashboard exceeds its size budget")
    args.output.write_bytes(html.encode("utf-8"))
    print(f"written {args.output} ({len(html.encode('utf-8'))} bytes, verdict {payload['verdict']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
