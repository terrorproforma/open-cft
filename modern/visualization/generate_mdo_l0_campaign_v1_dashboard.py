"""Generate the standalone MDO L0 campaign v1 results dashboard.

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/mdo_l0_campaign_v1`` (or from that experiment's committed
``protocol.json`` for verbatim declarations).  The generator verifies every
manifest entry byte-for-byte before rendering and refuses to render on any
mismatch.  It emits no wall-clock timestamps or machine paths of its own, so
identical inputs produce identical bytes.  The page is offline (no external
resources) and draws every chart with inline SVG built by inline JavaScript.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

MODERN = Path(__file__).resolve().parents[1]
SRC = MODERN / "src"
for entry in (str(SRC), str(MODERN)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

HERE = Path(__file__).resolve().parent
EXPERIMENT = MODERN / "experiments" / "mdo_l0_campaign_v1"
DEFAULT_RESULTS = EXPERIMENT / "results"
DEFAULT_OUTPUT = HERE / "mdo-l0-campaign-v1.html"

SCHEMA = "cft-revival.mdo-l0-campaign-v1-dashboard/1.0.0"
MAX_HTML_BYTES = 2_500_000

# Committed identity of the recorded campaign (exp/mdo-l0-campaign-v1).
EXPECTED_MANIFEST_SHA256: str | None = (
    "2a326f3cf2e286a0ba8f9c91871d78b935a11e3e2c56bd9164acdc6166485381"
)
RESULTS_COMMIT_SHA: str | None = "c553124b7393890d8ee9c6fc022e536c8a1fd35e"
PREREGISTRATION_COMMIT_SHA = "4898d0fd3decddc5f308072e724d1936660c00e9"

STRATEGIES = ("qlognehvi", "nsga3", "lhs")
STRATEGY_LABELS = {
    "qlognehvi": "BoTorch qLogNEHVI",
    "nsga3": "pymoo NSGA-III",
    "lhs": "Latin hypercube (random)",
}
OBJECTIVES = (
    "axial_thrust_n",
    "specific_impulse_s",
    "thruster_electrical_to_beam_efficiency",
    "anode_input_power_w",
)
OBJECTIVE_LABELS = {
    "axial_thrust_n": "axial thrust [N]",
    "specific_impulse_s": "specific impulse [s]",
    "thruster_electrical_to_beam_efficiency": "thruster-electrical-to-beam efficiency [1]",
    "anode_input_power_w": "anode input power [W]",
}
DESIGN_VARIABLES = ("discharge_voltage_v", "anode_current_a", "propellant_mass_flow_kg_per_s")


# --------------------------------------------------------------------------- #
# Strict loading
# --------------------------------------------------------------------------- #
def _load_json_bytes(raw: bytes, label: str) -> Any:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {label}")

    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=closed_object, parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _sig(value: float, digits: int = 6) -> float:
    return float(f"{value:.{digits}g}")


class Bundle:
    """The verified results bundle."""

    def __init__(self, root: Path, *, expected_manifest_sha256: str | None) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing manifest: {manifest_path}")
        manifest_bytes = manifest_path.read_bytes()
        self.manifest_sha256 = _digest(manifest_bytes)
        if expected_manifest_sha256 is not None and self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("manifest.json does not match the pinned campaign identity")
        self.manifest = _load_json_bytes(manifest_bytes, "manifest.json")
        if self.manifest.get("experiment_id") not in {
            "mdo-l0-campaign-v1",
            "mdo-l0-campaign-v1-shakedown",
        }:
            raise ValueError("bundle is not the mdo-l0-campaign-v1 experiment")
        if expected_manifest_sha256 is not None and self.manifest["experiment_id"] != "mdo-l0-campaign-v1":
            raise ValueError("a pinned dashboard must render the evidentiary bundle")
        self.state = str(self.manifest["state"])
        self.files: dict[str, bytes] = {}
        for entry in self.manifest["artifacts"]:
            if entry["type"] != "file":
                continue
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe manifest path {entry['path']}")
            path = root.joinpath(*relative.parts)
            data = path.read_bytes()
            if _digest(data) != entry["byte_sha256"] or len(data) != entry["bytes"]:
                raise ValueError(f"byte mismatch for {entry['path']}")
            self.files[str(relative)] = data
        self.verified_count = len(self.files)

    def json(self, relative: str) -> Any:
        if relative not in self.files:
            raise ValueError(f"bundle has no artifact {relative}")
        return _load_json_bytes(self.files[relative], relative)


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _front_rows(designs: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = []
    for design in designs:
        objectives = design[key]
        if objectives is None:
            continue
        rows.append(
            {
                "design_id": design["design_id"][:12],
                "values": [_sig(v, 8) for v in design["values"]],
                "objectives": [_sig(objectives[name], 8) for name in OBJECTIVES],
                "robust_margin_a": _sig(design["constraints"]["robust_beam_current_margin_a"], 6),
                "nominal_margin_a": _sig(design["constraints"]["nominal_beam_current_margin_a"], 6),
            }
        )
    return rows


def build_payload(bundle: Bundle) -> dict[str, Any]:
    protocol = bundle.json("artifacts/protocol.json")
    plan = bundle.json("artifacts/campaign-plan.json")
    terminal = _load_json_bytes(bundle.files["terminal.json"], "terminal.json")
    metrics = bundle.json("artifacts/metrics.json")
    gates = bundle.json("artifacts/gates.json")
    pooled = bundle.json("artifacts/pooled-fronts.json")
    per_strategy = bundle.json("artifacts/per-strategy-fronts.json")
    curves = bundle.json("artifacts/hypervolume-curves.json")
    sensitivity = bundle.json("artifacts/sensitivity.json")
    campaign_result = bundle.json("artifacts/campaign-result.json")
    contract = bundle.json("artifacts/code-contract.json")
    dense_summary = bundle.json("artifacts/dense-reference-summary.json")
    probes = bundle.json("artifacts/device-probes.json")
    pareto_sets = bundle.json("artifacts/pareto-sets.json")
    lock = (
        _load_json_bytes(bundle.files["execution-lock.json"], "execution-lock.json")
        if "execution-lock.json" in bundle.files
        else None
    )

    if terminal["state"] != bundle.state:
        raise ValueError("terminal state disagrees with the manifest")
    if campaign_result["all_binding_gates_passed"] != gates["all_binding_passed"]:
        raise ValueError("campaign-result and gates disagree on the binding gates")
    seeds = [int(seed) for seed in plan["seeds"]]
    strategies = list(plan["strategies"])
    if tuple(strategies) != STRATEGIES:
        raise ValueError("unexpected strategy order")

    runs = {}
    for strategy in strategies:
        for seed in seeds:
            key = f"{strategy}:{seed}"
            summary = metrics["runs"][key]
            table = metrics["hypervolume_table"][key]
            if summary["final_hypervolume"] != table["final_hypervolume"]:
                raise ValueError(f"metrics disagree for {key}")
            curve = curves[key]
            if curve[-1]["hypervolume"] != summary["final_hypervolume"]:
                raise ValueError(f"curve tail disagrees with the summary for {key}")
            artifact = bundle.json(f"artifacts/runs/{strategy}-{seed}.json")
            if artifact["summary"]["final_hypervolume"] != summary["final_hypervolume"]:
                raise ValueError(f"run artifact disagrees with metrics for {key}")
            iteration_log = artifact["optimizer"].get("iteration_log", [])
            runs[key] = {
                "strategy": strategy,
                "seed": seed,
                "evaluations": summary["evaluations"],
                "feasible": summary["feasible_evaluations"],
                "infeasible": summary["infeasible_evaluations"],
                "final_hypervolume": summary["final_hypervolume"],
                "attained_fraction": table["attained_fraction_of_dense_reference"],
                "pareto_set_size": summary["pareto_set_size"],
                "wall_clock_seconds": _sig(summary["wall_clock_seconds"], 5),
                "timing": {k: _sig(v, 5) for k, v in metrics["timing"][key].items()},
                "curve": [[c["evaluations"], _sig(c["hypervolume"], 8)] for c in curve],
                "bo_iterations": [
                    {
                        "iteration": entry["iteration"],
                        "training_points": entry["training_points"],
                        "fit_seconds": _sig(entry["fit_seconds"], 4),
                        "acquisition_seconds": _sig(entry["acquisition_seconds"], 4),
                        "hypervolume": _sig(entry["hypervolume"], 8),
                    }
                    for entry in iteration_log
                ],
                "pareto": [
                    {
                        "index": d["index"],
                        "design_id": d["design_id"][:12],
                        "values": [_sig(v, 8) for v in d["values"]],
                        "robust": [_sig(d["robust_objectives"][name], 8) for name in OBJECTIVES],
                        "nominal": (
                            None
                            if d["nominal_objectives"] is None
                            else [_sig(d["nominal_objectives"][name], 8) for name in OBJECTIVES]
                        ),
                    }
                    for d in pareto_sets[key]["designs"]
                ],
                "records": [
                    {
                        "index": r["index"],
                        "status": r["status"],
                        "values": [_sig(v, 8) for v in r["design"]["values"]],
                        "robust": (
                            None
                            if r["robust_objectives"] is None
                            else [_sig(r["robust_objectives"][name], 8) for name in OBJECTIVES]
                        ),
                        "margin": _sig(r["constraints"]["robust_beam_current_margin_a"], 6),
                    }
                    for r in artifact["records"]
                ],
            }

    reported = gates["reported_not_binding"]
    payload = {
        "schema": SCHEMA,
        "identity": {
            "experiment_id": bundle.manifest["experiment_id"],
            "terminal_state": bundle.state,
            "manifest_sha256": bundle.manifest_sha256,
            "terminal_byte_sha256": bundle.manifest["terminal_byte_sha256"],
            "lock_byte_sha256": bundle.manifest["lock_byte_sha256"],
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_files": bundle.verified_count,
            "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
            "results_commit": RESULTS_COMMIT_SHA,
            "lock_commit": None if lock is None else lock["commit"],
            "lock_acquired_at_utc": None if lock is None else lock["acquired_at_utc"]["value"],
            "source_sha256": contract["source_sha256"],
            "package_versions": contract["observed_package_versions"],
            "python": contract["python"],
            "device_probes": probes,
        },
        "protocol": {
            "classification": protocol["classification"],
            "claim_boundary": protocol["claim_boundary"],
            "design_variables": protocol["design_variables"],
            "excluded_legacy_variables": protocol["excluded_legacy_variables"],
            "uncertain_inputs": protocol["uncertain_inputs"]["inputs"],
            "cusp_prior_calibration": protocol["uncertain_inputs"]["cusp_prior_calibration"],
            "sample": protocol["uncertain_inputs"]["sample"],
            "closure": protocol["closures"]["CL-1"],
            "fixed_closures": protocol["closures"]["fixed"],
            "objectives": protocol["objectives"],
            "reference_point": protocol["reference_point"],
            "constraints": protocol["constraints"],
            "robust_formulation": protocol["robust_formulation"],
            "optimizers": protocol["optimizers"],
            "budget": protocol["budget"],
            "gates": protocol["gates"],
            "prior_model_disclosure": protocol["prior_model_disclosure"],
            "wall_loss_v4": protocol["authority"]["wall_loss_v4"],
        },
        "plan": plan,
        "runs": runs,
        "seeds": seeds,
        "strategies": strategies,
        "gates": {
            "binding": {name: item["passed"] for name, item in gates["binding"].items()},
            "all_binding_passed": gates["all_binding_passed"],
            "bo_beats_random": reported["bo_beats_random"],
            "bo_beats_nsga3": reported["bo_beats_nsga3"],
            "design_set_invariance": {
                "passed": reported["design_set_invariance"]["passed"],
                "definition": reported["design_set_invariance"]["definition"],
                "per_prior": reported["design_set_invariance"]["per_prior"],
            },
            "robust_vs_nominal": reported["robust_vs_nominal"],
        },
        "seed_variance": metrics["seed_variance"],
        "dense_reference": {
            **metrics["dense_reference"],
            "evaluation_seconds": _sig(dense_summary["evaluation_seconds"], 5),
            "feasible": dense_summary["feasible"],
            "infeasible": dense_summary["infeasible"],
            "separability": dense_summary["separability"],
        },
        "pooled": {
            "unique_designs": pooled["unique_designs"],
            "robust": {
                "front_size": pooled["robust"]["front_size"],
                "hypervolume": pooled["robust"]["hypervolume"],
                "candidates": pooled["robust"]["candidates"],
                "designs": _front_rows(pooled["robust"]["designs"], "robust_objectives"),
                "nominal_of_robust_front": _front_rows(pooled["robust"]["designs"], "nominal_objectives"),
            },
            "nominal": {
                "front_size": pooled["nominal"]["front_size"],
                "hypervolume": pooled["nominal"]["hypervolume"],
                "candidates": pooled["nominal"]["candidates"],
                "robust_feasible_members": pooled["nominal"]["robust_feasible_members"],
                "designs": _front_rows(pooled["nominal"]["designs"], "nominal_objectives"),
            },
            "shared_designs": len(pooled["shared_design_ids"]),
            "jaccard": pooled["jaccard_robust_nominal"],
        },
        "per_strategy": {
            strategy: {
                "robust_front_size": per_strategy[strategy]["robust"]["front_size"],
                "robust_hypervolume": per_strategy[strategy]["robust"]["hypervolume"],
                "nominal_front_size": per_strategy[strategy]["nominal"]["front_size"],
                "robust_designs": _front_rows(per_strategy[strategy]["robust"]["designs"], "robust_objectives"),
            }
            for strategy in strategies
        },
        "sensitivity": {
            "priors": [
                {k: (_sig(v, 6) if isinstance(v, float) else v) for k, v in item.items() if k != "front_design_ids"}
                for item in sensitivity["priors"]
            ],
            "scenarios": [
                {
                    "id": item["id"],
                    "cusp_probabilities": item["cusp_probabilities"],
                    "survival": _sig(item["survival"], 6),
                    "pareto_designs_evaluated": item["pareto_designs_evaluated"],
                    "pareto_designs_infeasible": item["pareto_designs_infeasible"],
                    "objective_ranges": item["objective_ranges"],
                    "hypervolume": _sig(item["hypervolume"], 6),
                }
                for item in sensitivity["scenarios"]
            ],
        },
        "campaign_result": campaign_result,
    }
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA:
        raise ValueError("payload schema mismatch")
    for key in payload["runs"]:
        run = payload["runs"][key]
        curve = [point[1] for point in run["curve"]]
        if any(b < a for a, b in zip(curve, curve[1:], strict=False)):
            raise ValueError(f"hypervolume curve not monotone for {key}")
        if run["evaluations"] != payload["plan"]["evaluations_per_run"]:
            raise ValueError(f"budget mismatch for {key}")
    text = json.dumps(payload, allow_nan=False)
    if "NaN" in text or "Infinity" in text:
        raise ValueError("payload contains nonfinite numbers")


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MDO L0 campaign v1 — robust multi-objective optimisation of the corrected L0 model</title>
<style>
:root{--bg:#0f1419;--panel:#171d25;--ink:#e7ecf2;--muted:#9aa7b5;--line:#2a3441;--bo:#4fc3f7;--nsga:#ffb74d;--lhs:#a5d6a7;--warn:#ff7043;--ok:#66bb6a;--nom:#ce93d8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;overflow-wrap:anywhere}
p,li,td,code,.mono{overflow-wrap:anywhere;word-break:break-word}
header{padding:20px 24px;border-bottom:1px solid var(--line)}
h1{font-size:20px;margin:0 0 6px}
h2{font-size:16px;margin:0 0 10px;color:var(--ink)}
h3{font-size:14px;margin:12px 0 6px;color:var(--muted)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;padding:16px 24px 40px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
section.wide{grid-column:1/-1}
.claim{border-color:var(--warn)}
.claim p{margin:6px 0}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;margin-right:6px}
.ok{background:#1b3a25;color:var(--ok)}.fail{background:#4a1f16;color:var(--warn)}.info{background:#1f2a3a;color:var(--bo)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:600}
svg{width:100%;height:auto;display:block}
.legend span{display:inline-block;margin-right:12px;font-size:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
code,.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--muted)}
.muted{color:var(--muted)}
ul{margin:6px 0;padding-left:18px}
.scroll{overflow-x:auto}
footer{padding:12px 24px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
@media (max-width:600px){main{padding:10px}header{padding:14px}}
</style>
</head>
<body>
<header>
  <h1>MDO L0 campaign v1 — robust multi-objective optimisation of the corrected L0 model</h1>
  <div id="headline" class="muted"></div>
</header>
<main>
  <section class="claim wide" id="claim"></section>
  <section class="wide" id="hv"></section>
  <section id="hvtable"></section>
  <section id="gates"></section>
  <section class="wide" id="fronts"></section>
  <section class="wide" id="parallel"></section>
  <section class="wide" id="sensitivity"></section>
  <section id="timing"></section>
  <section id="protocol"></section>
  <section id="provenance"></section>
</main>
<footer id="footer"></footer>
<div id="jserrors" hidden></div>
<script>
window.addEventListener("error", function(event){
  var sink = document.getElementById("jserrors");
  if (sink) { sink.textContent += (event.message || "error") + "\n"; }
});
</script>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
"use strict";
const P = JSON.parse(document.getElementById("payload").textContent);
const COLORS = {qlognehvi:"var(--bo)", nsga3:"var(--nsga)", lhs:"var(--lhs)"};
const RAW = {qlognehvi:"#4fc3f7", nsga3:"#ffb74d", lhs:"#a5d6a7"};
const LABELS = {qlognehvi:"BoTorch qLogNEHVI", nsga3:"pymoo NSGA-III", lhs:"Latin hypercube (random)"};
const OBJ = ["axial_thrust_n","specific_impulse_s","thruster_electrical_to_beam_efficiency","anode_input_power_w"];
const OBJL = ["thrust [N]","Isp [s]","efficiency [1]","anode power [W]"];
const VARL = ["Ua [V]","Ia [A]","mdot [kg/s]"];
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d) => (v === null || v === undefined) ? "—" : (typeof v === "number" ? (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(d ?? 3) : v.toPrecision(d ?? 5)) : String(v));
const badge = (ok, t) => `<span class="badge ${ok ? "ok" : "fail"}">${esc(t)}</span>`;
const el = id => document.getElementById(id);

// ---- headline ---------------------------------------------------------------
el("headline").innerHTML = `terminal state <b>${esc(P.identity.terminal_state)}</b> · ${P.plan.run_ids.length} runs · ${P.campaign_result.total_evaluations} L0 design evaluations (${P.campaign_result.infeasible_evaluations} infeasible) · manifest <code>${P.identity.manifest_sha256.slice(0,12)}</code> · preregistration <code>${(P.identity.preregistration_commit||"").slice(0,12)}</code>`;

// ---- claim boundary -----------------------------------------------------------
{
  const cb = P.protocol.claim_boundary;
  el("claim").innerHTML = `<h2>Claim boundary</h2>
  <p><b>${esc(cb.statement)}</b></p>
  <p><b>Why the cusp probabilities are uncertain inputs.</b> ${esc(cb.why_cusp_probabilities_are_uncertain_inputs)}</p>
  <p><b>Why geometry is excluded.</b> ${esc(cb.why_geometry_variables_are_excluded)}</p>
  <p><b>Closure.</b> ${esc(cb.closure_disclosure)} <span class="mono">${esc(P.protocol.closure.statement)}</span></p>
  <p class="muted"><b>Forbidden readings:</b> ${cb.forbidden_readings.map(esc).join("; ")}.</p>
  <p class="muted"><b>Prior model disclosure:</b> ${esc(P.protocol.prior_model_disclosure.corrected_four_cell_solver_probe)}</p>`;
}

// ---- SVG helpers ------------------------------------------------------------
function svgOpen(w, h){ return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" role="img">`; }
function axis(x0, y0, x1, y1){ return `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="#4a5666" stroke-width="1"/>`; }
function ticks(scale, n, isX, x0, y0, x1, y1, fmtf){
  let s = "";
  for (let i = 0; i <= n; i++){
    const t = scale.min + (scale.max - scale.min) * i / n;
    if (isX){ const x = x0 + (x1 - x0) * i / n; s += `<line x1="${x}" y1="${y0}" x2="${x}" y2="${y0+4}" stroke="#4a5666"/><text x="${x}" y="${y0+16}" fill="#9aa7b5" font-size="10" text-anchor="middle">${fmtf(t)}</text>`; }
    else { const y = y0 - (y0 - y1) * i / n; s += `<line x1="${x0-4}" y1="${y}" x2="${x0}" y2="${y}" stroke="#4a5666"/><text x="${x0-6}" y="${y+3}" fill="#9aa7b5" font-size="10" text-anchor="end">${fmtf(t)}</text>`; }
  }
  return s;
}
const lin = (v, s, a, b) => a + (b - a) * (v - s.min) / ((s.max - s.min) || 1);
function extent(vals){ let mn = Infinity, mx = -Infinity; for (const v of vals){ if (v < mn) mn = v; if (v > mx) mx = v; } if (!isFinite(mn)) { mn = 0; mx = 1; } if (mn === mx){ mx = mn + 1; } return {min: mn, max: mx}; }

// ---- hypervolume curves -----------------------------------------------------
{
  const W = 900, H = 340, x0 = 60, y0 = 300, x1 = 880, y1 = 20;
  const allHv = []; for (const k in P.runs) for (const p of P.runs[k].curve) allHv.push(p[1]);
  const ys = {min: 0, max: Math.max(extent(allHv).max, P.dense_reference.robust_hypervolume) * 1.05};
  const xs = {min: 0, max: P.plan.evaluations_per_run};
  let s = svgOpen(W, H) + axis(x0, y0, x1, y0) + axis(x0, y0, x0, y1);
  s += ticks(xs, 6, true, x0, y0, x1, y1, t => t.toFixed(0)) + ticks(ys, 5, false, x0, y0, x1, y1, t => t.toExponential(1));
  const yref = lin(P.dense_reference.robust_hypervolume, ys, y0, y1);
  s += `<line x1="${x0}" y1="${yref}" x2="${x1}" y2="${yref}" stroke="#9aa7b5" stroke-dasharray="4 4"/><text x="${x1}" y="${yref-4}" fill="#9aa7b5" font-size="10" text-anchor="end">dense reference (${P.dense_reference.count} designs) robust HV = ${P.dense_reference.robust_hypervolume.toExponential(3)}</text>`;
  const xinit = lin(P.plan.initial_design, xs, x0, x1);
  s += `<line x1="${xinit}" y1="${y0}" x2="${xinit}" y2="${y1}" stroke="#2a3441"/><text x="${xinit+3}" y="${y1+10}" fill="#9aa7b5" font-size="10">shared initial design (${P.plan.initial_design})</text>`;
  for (const k in P.runs){
    const r = P.runs[k];
    const pts = r.curve.map(p => `${lin(p[0], xs, x0, x1).toFixed(1)},${lin(p[1], ys, y0, y1).toFixed(1)}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="${RAW[r.strategy]}" stroke-width="1.6" opacity="0.9"/>`;
  }
  s += `<text x="${(x0+x1)/2}" y="${H-4}" fill="#9aa7b5" font-size="11" text-anchor="middle">L0 design evaluations (each = 64 QMC + 1 nominal L0 points)</text>`;
  s += `<text transform="translate(14,${(y0+y1)/2}) rotate(-90)" fill="#9aa7b5" font-size="11" text-anchor="middle">robust hypervolume (dimensionless, all-maximise frame)</text></svg>`;
  el("hv").innerHTML = `<h2>Hypervolume versus evaluations (robust CVaR objectives, feasible nondominated set)</h2>
  <div class="legend">${P.strategies.map(st => `<span><i style="background:${RAW[st]}"></i>${esc(LABELS[st])} (seeds ${P.seeds.join(", ")})</span>`).join("")}</div>${s}
  <p class="muted">Every curve is the exact hypervolume of the cumulative feasible nondominated set after each evaluation; the binding gate <code>hypervolume_monotone</code> ${badge(P.gates.binding.hypervolume_monotone, P.gates.binding.hypervolume_monotone ? "passed" : "FAILED")}.</p>`;
}

// ---- hypervolume table + seed variance ---------------------------------------
{
  let rows = "";
  for (const st of P.strategies) for (const seed of P.seeds){
    const r = P.runs[`${st}:${seed}`];
    rows += `<tr><td style="color:${RAW[st]}">${esc(LABELS[st])}</td><td>${seed}</td><td>${r.final_hypervolume.toExponential(4)}</td><td>${fmt(r.attained_fraction, 3)}</td><td>${r.pareto_set_size}</td><td>${r.infeasible}</td><td>${fmt(r.wall_clock_seconds, 4)}</td></tr>`;
  }
  let vrows = "";
  for (const st of P.strategies){ const v = P.seed_variance[st]; vrows += `<tr><td style="color:${RAW[st]}">${esc(LABELS[st])}</td><td>${v.mean.toExponential(4)}</td><td>${v.minimum.toExponential(4)}</td><td>${v.maximum.toExponential(4)}</td><td>${v.sample_std === null ? "—" : v.sample_std.toExponential(3)}</td></tr>`; }
  const br = P.gates.bo_beats_random, bn = P.gates.bo_beats_nsga3;
  el("hvtable").innerHTML = `<h2>Final hypervolume per optimiser × seed</h2><div class="scroll"><table><thead><tr><th>strategy</th><th>seed</th><th>final HV</th><th>fraction of dense ref.</th><th>Pareto set</th><th>infeasible</th><th>wall [s]</th></tr></thead><tbody>${rows}</tbody></table></div>
  <h3>Seed-repeat variance of the final hypervolume</h3><div class="scroll"><table><thead><tr><th>strategy</th><th>mean</th><th>min</th><th>max</th><th>sample std</th></tr></thead><tbody>${vrows}</tbody></table></div>
  <h3>Predeclared comparisons (reported, not binding)</h3>
  <p>${badge(br.passed, `BO beats random: ${br.wins}/${br.seeds} seeds (needs ≥ ${br.required_wins})`)} ${badge(bn.passed, `BO beats NSGA-III: ${bn.wins}/${bn.seeds} seeds`)}</p>
  <div class="scroll"><table><thead><tr><th>seed</th><th>qLogNEHVI</th><th>LHS</th><th>NSGA-III</th></tr></thead><tbody>${P.seeds.map(sd => `<tr><td>${sd}</td><td>${P.runs["qlognehvi:"+sd].final_hypervolume.toExponential(4)}</td><td>${P.runs["lhs:"+sd].final_hypervolume.toExponential(4)}</td><td>${P.runs["nsga3:"+sd].final_hypervolume.toExponential(4)}</td></tr>`).join("")}</tbody></table></div>`;
}

// ---- gates ------------------------------------------------------------------
{
  const g = P.gates.binding;
  const names = Object.keys(g);
  el("gates").innerHTML = `<h2>Gates</h2><p>${badge(P.gates.all_binding_passed, P.gates.all_binding_passed ? "all binding gates passed" : "binding gate failed")} terminal state <code>${esc(P.identity.terminal_state)}</code></p>
  <div class="scroll"><table><thead><tr><th>binding gate</th><th>result</th><th>declaration</th></tr></thead><tbody>${names.map(n => `<tr><td>${esc(n)}</td><td>${badge(g[n], g[n] ? "pass" : "FAIL")}</td><td style="white-space:normal;text-align:left" class="muted">${esc(P.protocol.gates.binding[n] || "")}</td></tr>`).join("")}</tbody></table></div>
  <p class="muted">Terminal rule: ${esc(P.protocol.gates.terminal_rule)}</p>`;
}

// ---- fronts: robust vs nominal (two projections) -----------------------------
function scatter(title, rows, ix, iy, extra){
  const W = 440, H = 320, x0 = 64, y0 = 280, x1 = 425, y1 = 16;
  const all = rows.flatMap(r => r.points);
  const xs = extent(all.map(p => p[ix])), ys = extent(all.map(p => p[iy]));
  xs.min = 0; ys.min = ys.min > 0 ? 0 : ys.min; xs.max *= 1.05; ys.max *= 1.05;
  let s = svgOpen(W, H) + axis(x0, y0, x1, y0) + axis(x0, y0, x0, y1);
  s += ticks(xs, 5, true, x0, y0, x1, y1, t => fmt(t, 3)) + ticks(ys, 5, false, x0, y0, x1, y1, t => fmt(t, 3));
  for (const r of rows){
    for (const p of r.points){
      s += `<circle cx="${lin(p[ix], xs, x0, x1).toFixed(1)}" cy="${lin(p[iy], ys, y0, y1).toFixed(1)}" r="${r.r || 3}" fill="${r.fill ? r.color : "none"}" stroke="${r.color}" stroke-width="1.2" opacity="${r.opacity || 0.9}"/>`;
    }
  }
  s += `<text x="${(x0+x1)/2}" y="${H-2}" fill="#9aa7b5" font-size="11" text-anchor="middle">${esc(OBJL[ix])}</text>`;
  s += `<text transform="translate(12,${(y0+y1)/2}) rotate(-90)" fill="#9aa7b5" font-size="11" text-anchor="middle">${esc(OBJL[iy])}</text></svg>`;
  return `<div><h3>${esc(title)}</h3>${s}${extra || ""}</div>`;
}
{
  const robust = P.pooled.robust.designs.map(d => d.objectives);
  const nominal = P.pooled.nominal.designs.map(d => d.objectives);
  const robustNominal = P.pooled.robust.nominal_of_robust_front.map(d => d.objectives);
  const rows = [
    {points: nominal, color: "#ce93d8", fill: false, r: 3},
    {points: robustNominal, color: "#ffffff", fill: false, r: 2, opacity: 0.6},
    {points: robust, color: "#4fc3f7", fill: true, r: 3},
  ];
  const rvn = P.gates.robust_vs_nominal;
  el("fronts").innerHTML = `<h2>Pooled Pareto fronts: robust (CVaR) versus nominal (prior midpoints)</h2>
  <div class="legend"><span><i style="background:#4fc3f7"></i>robust front, robust objectives (${P.pooled.robust.front_size} designs, HV ${P.pooled.robust.hypervolume.toExponential(3)})</span><span><i style="background:#ce93d8"></i>nominal front, nominal objectives (${P.pooled.nominal.front_size} designs, HV ${P.pooled.nominal.hypervolume.toExponential(3)})</span><span><i style="background:#fff"></i>robust-front designs re-evaluated at the nominal theta</span></div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px">
  ${scatter("thrust versus anode power", rows, 3, 0)}
  ${scatter("efficiency versus Isp", rows, 1, 2)}
  ${scatter("thrust versus Isp", rows, 1, 0)}
  </div>
  <p>Pooled over all ${P.pooled.unique_designs} unique evaluated designs (${P.pooled.robust.candidates} robust-feasible, ${P.pooled.nominal.candidates} nominally feasible). Shared designs between the two fronts: ${rvn.shared_designs} (Jaccard ${fmt(rvn.jaccard, 3)}); ${rvn.nominal_front_members_robust_feasible} of the ${rvn.nominal_front_size} nominal-front designs are robust-feasible. ${esc(P.protocol.robust_formulation.predeclared_expectation)}</p>`;
}

// ---- parallel coordinates of the pooled robust Pareto designs ----------------
{
  const designs = P.pooled.robust.designs;
  const W = 900, H = 320, top = 30, bottom = 280, left = 60, right = 860;
  const axes = [
    {label: VARL[0], get: d => d.values[0]}, {label: VARL[1], get: d => d.values[1]}, {label: VARL[2], get: d => d.values[2]},
    {label: "margin [A]", get: d => d.robust_margin_a},
    {label: OBJL[0], get: d => d.objectives[0]}, {label: OBJL[1], get: d => d.objectives[1]}, {label: OBJL[2], get: d => d.objectives[2]}, {label: OBJL[3], get: d => d.objectives[3]},
  ];
  const scales = axes.map(a => extent(designs.map(a.get)));
  const xOf = i => left + (right - left) * i / (axes.length - 1);
  let s = svgOpen(W, H);
  axes.forEach((a, i) => {
    const x = xOf(i);
    s += `<line x1="${x}" y1="${top}" x2="${x}" y2="${bottom}" stroke="#4a5666"/><text x="${x}" y="${top-12}" fill="#9aa7b5" font-size="10" text-anchor="middle">${esc(a.label)}</text>`;
    s += `<text x="${x}" y="${bottom+14}" fill="#9aa7b5" font-size="9" text-anchor="middle">${fmt(scales[i].min, 3)}</text><text x="${x}" y="${top-2}" fill="#9aa7b5" font-size="9" text-anchor="middle">${fmt(scales[i].max, 3)}</text>`;
  });
  designs.forEach(d => {
    const pts = axes.map((a, i) => `${xOf(i).toFixed(1)},${lin(a.get(d), scales[i], bottom, top).toFixed(1)}`).join(" ");
    const hue = 200 + 120 * (d.objectives[3] - scales[7].min) / ((scales[7].max - scales[7].min) || 1);
    s += `<polyline points="${pts}" fill="none" stroke="hsl(${hue.toFixed(0)},80%,65%)" stroke-width="1" opacity="0.75"/>`;
  });
  s += `</svg>`;
  el("parallel").innerHTML = `<h2>Parallel coordinates of the ${designs.length} pooled robust-Pareto designs</h2>${s}<p class="muted">Colour encodes anode power (blue low → magenta high). Design variables, the robust beam-current margin and the four robust objectives are shown on independent min–max axes.</p>`;
}

// ---- sensitivity to the cusp probabilities -----------------------------------
{
  const pri = P.sensitivity.priors, sc = P.sensitivity.scenarios;
  const inv = P.gates.design_set_invariance;
  let prow = pri.map(p => `<tr><td>U[0, ${p.cusp_upper}]</td><td>${fmt(p.survival_min,3)} – ${fmt(p.survival_max,3)}</td><td>${fmt(p.survival_mean,3)}</td><td>${p.feasible}</td><td>${p.front_size}</td><td>${p.hypervolume.toExponential(3)}</td><td>${p.common_feasible_designs}</td><td>${badge(p.identical_on_common_feasible_set_up_to_ties, p.identical_on_common_feasible_set_up_to_ties ? "identical" : "differs")} ${p.identical_on_common_feasible_set ? "" : `<span class="muted">(exact: ${p.common_front_symmetric_difference} tie-difference)</span>`}</td><td>${fmt(p.jaccard_with_campaign_front,3)}</td></tr>`).join("");
  let srow = sc.map(x => `<tr><td>${esc(x.id)}</td><td class="mono">${x.cusp_probabilities.join(", ")}</td><td>${x.survival.toExponential(3)}</td><td>${x.pareto_designs_evaluated} / ${x.pareto_designs_infeasible}</td>${OBJ.map(o => `<td>${x.objective_ranges[o] ? `${fmt(x.objective_ranges[o].minimum,3)} – ${fmt(x.objective_ranges[o].maximum,3)}` : "—"}</td>`).join("")}<td>${x.hypervolume.toExponential(3)}</td></tr>`).join("");
  el("sensitivity").innerHTML = `<h2>Sensitivity of the front to the uncertain cusp probabilities</h2>
  <p>${badge(inv.passed, inv.passed ? "design-set invariance holds on the common feasible set" : "design-set invariance violated")} <span class="muted">${esc(inv.definition)}</span></p>
  <h3>Alternative priors (all recorded designs re-evaluated; campaign prior is U[0, 0.45])</h3>
  <div class="scroll"><table><thead><tr><th>cusp prior</th><th>survival S range</th><th>mean S</th><th>feasible designs</th><th>front size</th><th>HV</th><th>common feasible</th><th>front on common set</th><th>Jaccard vs campaign</th></tr></thead><tbody>${prow}</tbody></table></div>
  <h3>Scenarios (pooled robust-Pareto designs evaluated at fixed cusp probabilities, other closures nominal)</h3>
  <div class="scroll"><table><thead><tr><th>scenario</th><th>p₁..p₄</th><th>S</th><th>evaluated / infeasible</th>${OBJL.map(l => `<th>${esc(l)}</th>`).join("")}<th>HV</th></tr></thead><tbody>${srow}</tbody></table></div>
  <p class="muted">${esc(P.protocol.cusp_prior_calibration)}</p>`;
}

// ---- timing -------------------------------------------------------------------
{
  let rows = "";
  for (const st of P.strategies) for (const seed of P.seeds){ const r = P.runs[`${st}:${seed}`]; rows += `<tr><td style="color:${RAW[st]}">${esc(LABELS[st])}</td><td>${seed}</td><td>${fmt(r.timing.wall_clock_seconds,4)}</td><td>${fmt(r.timing.evaluation_seconds,3)}</td><td>${r.bo_iterations.length ? fmt(r.timing.bo_fit_seconds,3) : "—"}</td><td>${r.bo_iterations.length ? fmt(r.timing.bo_acquisition_seconds,4) : "—"}</td></tr>`; }
  const bo = P.seeds.map(sd => P.runs["qlognehvi:"+sd].bo_iterations);
  const W = 420, H = 200, x0 = 50, y0 = 170, x1 = 410, y1 = 12;
  const allAcq = bo.flatMap(l => l.map(e => e.acquisition_seconds));
  const ys = {min: 0, max: extent(allAcq).max * 1.1}, xs = {min: 1, max: Math.max(...bo.map(l => l.length))};
  let s = svgOpen(W, H) + axis(x0, y0, x1, y0) + axis(x0, y0, x0, y1) + ticks(xs, Math.min(10, xs.max-1), true, x0, y0, x1, y1, t => t.toFixed(0)) + ticks(ys, 4, false, x0, y0, x1, y1, t => t.toFixed(0));
  bo.forEach((l, i) => { s += `<polyline points="${l.map(e => `${lin(e.iteration, xs, x0, x1).toFixed(1)},${lin(e.acquisition_seconds, ys, y0, y1).toFixed(1)}`).join(" ")}" fill="none" stroke="#4fc3f7" stroke-width="1.4" opacity="${0.5 + 0.25*i}"/>`; });
  s += `<text x="${(x0+x1)/2}" y="${H-2}" fill="#9aa7b5" font-size="10" text-anchor="middle">BO iteration (batch of ${P.plan.qlognehvi_batch_size})</text><text transform="translate(12,${(y0+y1)/2}) rotate(-90)" fill="#9aa7b5" font-size="10" text-anchor="middle">acquisition seconds</text></svg>`;
  el("timing").innerHTML = `<h2>Timing</h2><div class="scroll"><table><thead><tr><th>strategy</th><th>seed</th><th>wall [s]</th><th>L0 eval [s]</th><th>GP fit [s]</th><th>acquisition [s]</th></tr></thead><tbody>${rows}</tbody></table></div>
  <h3>qLogNEHVI acquisition seconds per iteration (three seeds)</h3>${s}
  <p class="muted">Dense reference: ${P.dense_reference.count} designs in ${fmt(P.dense_reference.evaluation_seconds,4)} s (${P.dense_reference.feasible} feasible). Device: ${esc(P.protocol.optimizers.qlognehvi.device)} — ${esc(P.protocol.optimizers.qlognehvi.device_note)}</p>`;
}

// ---- protocol summary -----------------------------------------------------------
{
  const pr = P.protocol;
  el("protocol").innerHTML = `<h2>Protocol (frozen at preregistration)</h2>
  <h3>Design variables</h3><ul>${pr.design_variables.map(v => `<li><code>${esc(v.name)}</code> ∈ [${v.lower}, ${v.upper}] ${esc(v.units)} — ${esc(v.legacy)}</li>`).join("")}</ul>
  <p class="muted">Excluded legacy radii: ${pr.excluded_legacy_variables.map(v => esc(v.name)).join(", ")}.</p>
  <h3>Uncertain inputs (independent uniform priors; frozen ${pr.sample.count}-row QMC sample <code>${esc(pr.sample.sha256.slice(0,12))}</code>)</h3><ul>${pr.uncertain_inputs.map(u => `<li><code>${esc(u.name)}</code> ∈ [${u.lower}, ${u.upper}] — ${esc(u.meaning)}</li>`).join("")}</ul>
  <p class="muted">v4 anchor: pooled wall hit ${pr.wall_loss_v4.pooled_wall_hit.successes}/${pr.wall_loss_v4.pooled_wall_hit.trials}; per cell ${Object.entries(pr.wall_loss_v4.per_cell_wall_hit).map(([k,v]) => `${esc(k)} ${v.successes}/${v.trials}`).join(", ")}; reflections ${pr.wall_loss_v4.reflections}.</p>
  <h3>Objectives</h3><ul>${pr.objectives.map(o => `<li><code>${esc(o.name)}</code> ${esc(o.direction)} [${esc(o.units)}]: <span class="muted">${esc(o.definition)}</span></li>`).join("")}</ul>
  <h3>Constraint</h3><ul>${pr.constraints.map(c => `<li><code>${esc(c.name)}</code> ${esc(c.sense)} ${c.threshold} ${esc(c.units)} (${esc(c.role)}): <span class="muted">${esc(c.definition)}</span></li>`).join("")}</ul>
  <h3>Robust formulation</h3><p class="muted">${esc(pr.robust_formulation.definition)}</p>
  <h3>Optimisers and budget</h3><ul>
  <li><b>qLogNEHVI</b>: ${esc(pr.optimizers.qlognehvi.model)}; ${esc(pr.optimizers.qlognehvi.acquisition)}; ${esc(pr.optimizers.qlognehvi.candidate_optimizer)}; MC samples ${pr.optimizers.qlognehvi.mc_samples}; batch ${pr.budget.qlognehvi_batch_size} × ${pr.budget.qlognehvi_iterations} iterations after ${pr.budget.initial_design} initial points.</li>
  <li><b>NSGA-III</b>: ${esc(pr.optimizers.nsga3.reference_directions)}; population ${pr.budget.nsga3_population_size} × ${pr.budget.nsga3_generations} generations; ${esc(pr.optimizers.nsga3.infeasible_placeholder)}</li>
  <li><b>LHS</b>: ${esc(pr.optimizers.lhs.design)}</li>
  <li>${pr.budget.evaluations_per_run} evaluations per run, seeds ${pr.budget.seeds.join(", ")}, ${pr.budget.total_evaluations} in total; fairness: ${pr.budget.fairness.map(esc).join("; ")}.</li></ul>`;
}

// ---- provenance -------------------------------------------------------------------
{
  const id = P.identity;
  el("provenance").innerHTML = `<h2>Provenance</h2><table><tbody>
  <tr><td>experiment</td><td class="mono">${esc(id.experiment_id)}</td></tr>
  <tr><td>terminal state</td><td class="mono">${esc(id.terminal_state)}</td></tr>
  <tr><td>manifest SHA-256</td><td class="mono">${esc(id.manifest_sha256)}</td></tr>
  <tr><td>terminal bytes SHA-256</td><td class="mono">${esc(id.terminal_byte_sha256)}</td></tr>
  <tr><td>lock bytes SHA-256</td><td class="mono">${esc(id.lock_byte_sha256)}</td></tr>
  <tr><td>artifacts verified</td><td class="mono">${id.verified_files} files (manifest count ${id.artifact_count})</td></tr>
  <tr><td>preregistration commit</td><td class="mono">${esc(id.preregistration_commit || "—")}</td></tr>
  <tr><td>results commit</td><td class="mono">${esc(id.results_commit || "recorded after this dashboard was generated")}</td></tr>
  <tr><td>lock commit</td><td class="mono">${esc(id.lock_commit || "—")}</td></tr>
  <tr><td>lock acquired (UTC)</td><td class="mono">${esc(id.lock_acquired_at_utc || "—")}</td></tr>
  <tr><td>source hash (optimization, active_learning, surrogates, physics, spec, experiment)</td><td class="mono">${esc(id.source_sha256)}</td></tr>
  <tr><td>packages</td><td class="mono">${Object.entries(id.package_versions).map(([k,v]) => `${esc(k)} ${esc(v)}`).join(", ")}</td></tr>
  <tr><td>CUDA probe (recorded, not used)</td><td class="mono">${esc(id.device_probes.cuda.available ? `${id.device_probes.cuda.device_name}, torch ${id.device_probes.cuda.torch_version}, CUDA ${id.device_probes.cuda.cuda_version}` : "unavailable")}</td></tr>
  </tbody></table>`;
  el("footer").textContent = `${P.schema} · generated from the immutable results bundle of modern/experiments/mdo_l0_campaign_v1 (manifest ${id.manifest_sha256.slice(0,16)}) · offline, no external resources · every number on this page is read from the bundle or the frozen protocol.`;
}
})();
</script>
</body>
</html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    text = text.replace("</", "<\\/")
    html = TEMPLATE.replace("__PAYLOAD__", text)
    if "__PAYLOAD__" in html:
        raise ValueError("template substitution failed")
    return html


def generate(
    results: Path = DEFAULT_RESULTS,
    output: Path = DEFAULT_OUTPUT,
    *,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    bundle = Bundle(results, expected_manifest_sha256=expected_manifest_sha256)
    payload = build_payload(bundle)
    validate_payload(payload)
    html = render_html(payload)
    data = html.encode("utf-8")
    if len(data) > MAX_HTML_BYTES:
        raise ValueError(f"dashboard exceeds {MAX_HTML_BYTES} bytes ({len(data)})")
    output.write_bytes(data)
    return {
        "output": str(output),
        "bytes": len(data),
        "manifest_sha256": bundle.manifest_sha256,
        "verified_files": bundle.verified_count,
        "terminal_state": bundle.state,
        "html_sha256": _digest(data),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help="skip the pinned manifest identity (development against a shakedown bundle)",
    )
    arguments = parser.parse_args(argv)
    report = generate(
        arguments.results,
        arguments.output,
        expected_manifest_sha256=None if arguments.no_pin else EXPECTED_MANIFEST_SHA256,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
