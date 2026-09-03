"""Generate the standalone MDO L0 campaign v2 results dashboard.

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/mdo_l0_campaign_v2`` (or from that experiment's sealed protocol for
verbatim declarations); the v1-versus-v2 panel reads the v1 bundle of
``modern/experiments/mdo_l0_campaign_v1`` the same way.  Both bundles are verified
byte-for-byte before rendering and the generator refuses to render on any mismatch.  It
emits no wall-clock timestamps or machine paths of its own, so identical inputs produce
identical bytes.  The page is offline (no external resources) and draws every chart with
inline SVG built by inline JavaScript.
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
EXPERIMENT = MODERN / "experiments" / "mdo_l0_campaign_v2"
DEFAULT_RESULTS = EXPERIMENT / "results"
DEFAULT_OUTPUT = HERE / "mdo-l0-campaign-v2.html"
V1_RESULTS = MODERN / "experiments" / "mdo_l0_campaign_v1" / "results"

SCHEMA = "cft-revival.mdo-l0-campaign-v2-dashboard/1.0.0"
MAX_HTML_BYTES = 2_500_000

# Committed identity of the recorded campaigns.
EXPECTED_MANIFEST_SHA256: str | None = (
    "ca3b58ce21eedb8ef094a3d73894b508fe8c438183fa02620d05f759541f7b1f"
)
RESULTS_COMMIT_SHA: str | None = "a003f766c330d4e5648844ba49cdf1c3a3ce3bc1"
PREREGISTRATION_COMMIT_SHA = "99914dc2fdbe88d18ab11ca86acad634129b4e08"
V1_EXPECTED_MANIFEST_SHA256 = "2a326f3cf2e286a0ba8f9c91871d78b935a11e3e2c56bd9164acdc6166485381"
V1_RESULTS_COMMIT_SHA = "c553124b7393890d8ee9c6fc022e536c8a1fd35e"

STRATEGIES = ("qlognehvi", "nsga3", "lhs")
OBJECTIVES = (
    "axial_thrust_n",
    "specific_impulse_s",
    "thruster_electrical_to_beam_efficiency",
    "anode_input_power_w",
)


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
        return json.loads(raw.decode("utf-8"), object_pairs_hook=closed_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _sig(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    return float(f"{float(value):.{digits}g}")


class Bundle:
    """A verified results bundle."""

    def __init__(self, root: Path, *, experiment_ids: set[str], expected_manifest_sha256: str | None, evidentiary_id: str) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing manifest: {manifest_path}")
        manifest_bytes = manifest_path.read_bytes()
        self.manifest_sha256 = _digest(manifest_bytes)
        if expected_manifest_sha256 is not None and self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("manifest.json does not match the pinned campaign identity")
        self.manifest = _load_json_bytes(manifest_bytes, "manifest.json")
        if self.manifest.get("experiment_id") not in experiment_ids:
            raise ValueError(f"bundle is not one of {sorted(experiment_ids)}")
        if expected_manifest_sha256 is not None and self.manifest["experiment_id"] != evidentiary_id:
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


def load_v2_bundle(results: Path, expected_manifest_sha256: str | None) -> Bundle:
    return Bundle(
        results,
        experiment_ids={"mdo-l0-campaign-v2", "mdo-l0-campaign-v2-shakedown"},
        expected_manifest_sha256=expected_manifest_sha256,
        evidentiary_id="mdo-l0-campaign-v2",
    )


def load_v1_bundle(results: Path = V1_RESULTS) -> Bundle:
    return Bundle(
        results,
        experiment_ids={"mdo-l0-campaign-v1"},
        expected_manifest_sha256=V1_EXPECTED_MANIFEST_SHA256,
        evidentiary_id="mdo-l0-campaign-v1",
    )


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _front_rows(designs: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = []
    for design in designs:
        if "design" in design:  # a full evaluation record (dense-reference fronts)
            design = {**design, **design["design"]}
        objectives = design[key]
        if objectives is None:
            continue
        rows.append(
            {
                "design_id": design["design_id"][:12],
                "catalogue_index": design["catalogue_index"],
                "case_id": design["case_id"],
                "values": [_sig(v, 8) for v in design["values"]],
                "objectives": [_sig(objectives[name], 8) for name in OBJECTIVES],
                "robust_margin_a": _sig(design["constraints"]["robust_beam_current_margin_a"], 6),
                "nominal_margin_a": _sig(design["constraints"]["nominal_beam_current_margin_a"], 6),
                "survival": {k: _sig(v, 6) for k, v in design["survival_statistics"].items()},
            }
        )
    return rows


def _v1_summary(v1: Bundle) -> dict[str, Any]:
    metrics = v1.json("artifacts/metrics.json")
    gates = v1.json("artifacts/gates.json")
    protocol = v1.json("artifacts/protocol.json")
    plan = v1.json("artifacts/campaign-plan.json")
    result = v1.json("artifacts/campaign-result.json")
    reported = gates["reported_not_binding"]
    return {
        "manifest_sha256": v1.manifest_sha256,
        "results_commit": V1_RESULTS_COMMIT_SHA,
        "terminal_state": v1.state,
        "classification": protocol["classification"],
        "design_space": "operating point only (Ua, Ia, mdot); geometry excluded",
        "cusp_probabilities": "independent uniform [0, 0.45] per cell (wide prior calibrated to the v4 pooled survival)",
        "evaluations_per_run": plan["evaluations_per_run"],
        "initial_design": plan["initial_design"],
        "seeds": plan["seeds"],
        "total_evaluations": result["total_evaluations"],
        "hypervolume_table": {
            key: {
                "final_hypervolume": row["final_hypervolume"],
                "attained_fraction": row["attained_fraction_of_dense_reference"],
                "pareto_set_size": row["pareto_set_size"],
            }
            for key, row in metrics["hypervolume_table"].items()
        },
        "seed_variance": metrics["seed_variance"],
        "dense_reference": metrics["dense_reference"],
        "bo_beats_random": {"wins": reported["bo_beats_random"]["wins"], "seeds": reported["bo_beats_random"]["seeds"]},
        "bo_beats_nsga3": {"wins": reported["bo_beats_nsga3"]["wins"], "seeds": reported["bo_beats_nsga3"]["seeds"]},
        "robust_vs_nominal": {
            k: reported["robust_vs_nominal"][k]
            for k in ("robust_front_size", "nominal_front_size", "shared_designs", "jaccard", "robust_hypervolume", "nominal_hypervolume")
        },
        "binding_gates": len(gates["binding"]),
    }


def build_payload(bundle: Bundle, v1: Bundle | None) -> dict[str, Any]:
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
    dense = bundle.json("artifacts/dense-reference.json")
    catalogue = bundle.json("artifacts/catalogue.json")
    binding = bundle.json("artifacts/catalogue-binding.json")
    probes = bundle.json("artifacts/device-probes.json")
    pareto_sets = bundle.json("artifacts/pareto-sets.json")
    import_scope = bundle.json("artifacts/import-scope.json")
    lock = _load_json_bytes(bundle.files["execution-lock.json"], "execution-lock.json") if "execution-lock.json" in bundle.files else None

    if terminal["state"] != bundle.state:
        raise ValueError("terminal state disagrees with the manifest")
    if campaign_result["all_binding_gates_passed"] != gates["all_binding_passed"]:
        raise ValueError("campaign-result and gates disagree on the binding gates")
    if dense_summary["robust_hypervolume"] != metrics["dense_reference"]["robust_hypervolume"]:
        raise ValueError("dense reference summary disagrees with metrics")
    if catalogue["catalogue_sha256"] != protocol["catalogue_binding_identity"]["catalogue_sha256"]:
        raise ValueError("sealed catalogue identity disagrees with the protocol")
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
            info = artifact["optimizer"]
            runs[key] = {
                "strategy": strategy,
                "seed": seed,
                "evaluations": summary["evaluations"],
                "unique_designs": summary["unique_designs"],
                "distinct_catalogue_designs": summary["distinct_catalogue_designs"],
                "feasible": summary["feasible_evaluations"],
                "infeasible": summary["infeasible_evaluations"],
                "final_hypervolume": summary["final_hypervolume"],
                "attained_fraction": table["attained_fraction_of_dense_reference"],
                "pareto_set_size": summary["pareto_set_size"],
                "pareto_catalogue_indices": summary["pareto_catalogue_indices"],
                "wall_clock_seconds": _sig(summary["wall_clock_seconds"], 5),
                "timing": {k: _sig(v, 5) for k, v in metrics["timing"][key].items()},
                "curve": [[c["evaluations"], _sig(c["hypervolume"], 8)] for c in curve],
                "labels": {
                    "acquisition": info.get("acquisition"),
                    "model": info.get("model"),
                    "declared_generations": info.get("declared_generations"),
                    "pymoo_n_gen": info.get("pymoo_n_gen"),
                    "eliminate_duplicates": info.get("eliminate_duplicates"),
                    "design": info.get("design"),
                    "torch_threads": info.get("torch_threads"),
                },
                "bo_iterations": [
                    {
                        "iteration": entry["iteration"],
                        "training_points": entry["training_points"],
                        "fit_seconds": _sig(entry["fit_seconds"], 4),
                        "candidate_stage_seconds": _sig(entry["candidate_stage_seconds"], 4),
                        "refinement_seconds": _sig(entry["refinement_seconds"], 4),
                        "refinements_accepted": sum(1 for r in entry["refinement"] if r["accepted"]),
                        "hypervolume": _sig(entry["hypervolume"], 8),
                    }
                    for entry in info.get("iteration_log", [])
                ],
                "records": [
                    {
                        "index": r["index"],
                        "status": r["status"],
                        "catalogue_index": r["design"]["catalogue_index"],
                        "values": [_sig(v, 8) for v in r["design"]["values"]],
                        "robust": None if r["robust_objectives"] is None else [_sig(r["robust_objectives"][name], 8) for name in OBJECTIVES],
                        "margin": _sig(r["constraints"]["robust_beam_current_margin_a"], 6),
                    }
                    for r in artifact["records"]
                ],
            }

    reported = gates["reported_not_binding"]
    dense_per_design = [
        {
            "catalogue_index": row["catalogue_index"],
            "case_id": row["case_id"],
            "feasible": row["feasible"],
            "robust_front_size": row["robust_front_size"],
            "robust_hypervolume": _sig(row["robust_hypervolume"], 6),
            "nominal_hypervolume": _sig(row["nominal_hypervolume"], 6),
            "nominal_survival_cl1": _sig(row["nominal_survival_cl1"], 6),
        }
        for row in dense["per_design"]
    ]
    catalogue_rows = []
    for design in catalogue["designs"]:
        geometry = design["geometry"]
        catalogue_rows.append(
            {
                "catalogue_index": design["catalogue_index"],
                "case_id": design["case_id"],
                "geometry_sha256": design["geometry_sha256"][:12],
                "cells": [_sig(cell["probability"], 6) for cell in design["cells"]],
                "cells_wilson": [[_sig(cell["wilson_95"][0], 4), _sig(cell["wilson_95"][1], 4)] for cell in design["cells"]],
                "pooled": _sig(design["pooled"]["probability"], 6),
                "pooled_wilson": [_sig(design["pooled"]["wilson_95"][0], 4), _sig(design["pooled"]["wilson_95"][1], 4)],
                "reflected": design["pooled"]["reflected"],
                "nominal_survival_cl1": _sig(design["nominal_survival_cl1"], 6),
                "nominal_survival_cl2": _sig(design["nominal_survival_cl2"], 6),
                "geometry": {
                    "chamber_length_m": geometry["chamber_length_m"],
                    "wall_radius_m": geometry["wall_radius_m"],
                    "stage_count": geometry["stage_count"],
                    "stage_pitch_m": geometry["stage_pitch_m"],
                    "has_divergent_exit": geometry["has_divergent_exit"],
                    "exit_length_m": geometry["exit_length_m"],
                    "magnet_outer_radius_m": geometry["magnet_outer_radius_m"],
                    "first_polarity": geometry["first_polarity"],
                },
            }
        )

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
            "source_files": len(contract["source_files"]),
            "import_scope_matches": import_scope["matches"],
            "package_versions": contract["observed_package_versions"],
            "python": contract["python"],
            "device_probes": probes,
            "catalogue_sha256": catalogue["catalogue_sha256"],
            "catalogue_binding": {
                "dataset_file_sha256": binding["dataset_file_sha256"],
                "screening_result_commit": binding["screening_result_commit"],
                "passed": binding["passed"],
            },
        },
        "protocol": {
            "classification": protocol["classification"],
            "claim_boundary": protocol["claim_boundary"],
            "design_space": protocol["design_space"],
            "uncertain_inputs": {
                "per_design_inputs": protocol["uncertain_inputs"]["per_design_inputs"],
                "pooled_input": protocol["uncertain_inputs"]["pooled_input"],
                "shared_inputs": protocol["uncertain_inputs"]["shared_inputs"],
                "sample": protocol["uncertain_inputs"]["sample"],
                "nominal": protocol["uncertain_inputs"]["nominal"],
                "sensitivity_widths": protocol["uncertain_inputs"]["sensitivity_widths"],
            },
            "closures": protocol["closures"],
            "objectives": protocol["objectives"],
            "reference_point": protocol["reference_point"],
            "constraints": protocol["constraints"],
            "robust_formulation": protocol["robust_formulation"],
            "optimizers": protocol["optimizers"],
            "budget": protocol["budget"],
            "dense_reference": protocol["dense_reference"],
            "gates": protocol["gates"],
            "v1_audit_disclosures_closed": protocol["v1_audit_disclosures_closed"],
            "authority": protocol["authority"],
        },
        "plan": plan,
        "runs": runs,
        "seeds": seeds,
        "strategies": strategies,
        "gates": {
            "semantics": gates["semantics"],
            "binding": {name: item["passed"] for name, item in gates["binding"].items()},
            "binding_detail": {
                "hypervolume_monotone_largest_relative_decrease": gates["binding"]["hypervolume_monotone"]["largest_relative_decrease"],
                "import_scope_imported_count": gates["binding"]["code_hash_scope_matches_imports"]["imported_count"],
                "nsga3_duplicates": gates["binding"]["nsga3_duplicates_eliminated"]["runs"],
                "replayed": gates["binding"]["replay_bit_exact"]["replayed"],
            },
            "all_binding_passed": gates["all_binding_passed"],
            "bo_beats_random": reported["bo_beats_random"],
            "bo_beats_nsga3": reported["bo_beats_nsga3"],
            "robust_vs_nominal": reported["robust_vs_nominal"],
            "closure_cl1_vs_cl2": reported["closure_cl1_vs_cl2"],
            "uncertainty_width_sensitivity": reported["uncertainty_width_sensitivity"],
            "per_design_separability": reported["per_design_separability"],
        },
        "seed_variance": metrics["seed_variance"],
        "dense_reference": {
            **metrics["dense_reference"],
            "evaluation_seconds": _sig(dense_summary["evaluation_seconds"], 5),
            "workers": dense_summary["workers"],
            "feasible": dense_summary["feasible"],
            "infeasible": dense_summary["infeasible"],
            "columns_sha256": dense_summary["columns_sha256"],
            "per_design": dense_per_design,
            "robust_front": _front_rows(dense["fronts"]["robust"]["records"], "robust_objectives"),
        },
        "catalogue": catalogue_rows,
        "pooled": {
            "unique_designs": pooled["unique_designs"],
            "distinct_catalogue_designs": pooled["distinct_catalogue_designs"],
            "robust": {
                "front_size": pooled["robust"]["front_size"],
                "hypervolume": pooled["robust"]["hypervolume"],
                "candidates": pooled["robust"]["candidates"],
                "catalogue_indices": pooled["robust"]["catalogue_indices"],
                "designs": _front_rows(pooled["robust"]["designs"], "robust_objectives"),
                "nominal_of_robust_front": _front_rows(pooled["robust"]["designs"], "nominal_objectives"),
                "catalogue_membership": [
                    {
                        "catalogue_index": item["catalogue_index"],
                        "case_id": item["case_id"],
                        "front_members": item["front_members"],
                        "pooled": _sig(item["pooled"]["probability"], 6),
                        "pooled_wilson": [_sig(item["pooled"]["wilson_95"][0], 4), _sig(item["pooled"]["wilson_95"][1], 4)],
                        "cells": [_sig(cell["probability"], 6) for cell in item["cells"]],
                        "nominal_survival_cl1": _sig(item["nominal_survival_cl1"], 6),
                        "geometry": item["geometry"],
                    }
                    for item in pooled["robust"]["catalogue_membership"]
                ],
            },
            "nominal": {
                "front_size": pooled["nominal"]["front_size"],
                "hypervolume": pooled["nominal"]["hypervolume"],
                "candidates": pooled["nominal"]["candidates"],
                "catalogue_indices": pooled["nominal"]["catalogue_indices"],
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
                "robust_catalogue_indices": per_strategy[strategy]["robust"]["catalogue_indices"],
                "distinct_catalogue_designs": per_strategy[strategy]["distinct_catalogue_designs"],
            }
            for strategy in strategies
        },
        "sensitivity": {
            "closure_cl2": {k: (_sig(v, 6) if isinstance(v, float) else v) for k, v in sensitivity["closure_cl2"].items() if k not in ("front_design_ids", "front_members")},
            "cl2_front": [
                {
                    "catalogue_index": member["catalogue_index"],
                    "case_id": member["case_id"],
                    "values": [_sig(v, 8) for v in member["values"]],
                    "objectives": [_sig(member["robust_objectives"][name], 8) for name in OBJECTIVES],
                }
                for member in sensitivity["closure_cl2"]["front_members"]
            ],
            "widths": [
                {k: (_sig(v, 6) if isinstance(v, float) else v) for k, v in item.items() if k not in ("front_design_ids", "front_members")}
                for item in sensitivity["widths"]
            ],
        },
        "campaign_result": {
            k: campaign_result[k]
            for k in (
                "classification",
                "claim_boundary",
                "closure",
                "closure_identification_disclosure",
                "sensitivity_closure",
                "gate_semantics",
                "runs",
                "total_evaluations",
                "infeasible_evaluations",
                "bo_beats_random",
                "bo_beats_random_wins",
                "bo_beats_nsga3",
                "bo_beats_nsga3_wins",
                "all_binding_gates_passed",
                "assessment_seconds",
            )
        },
        "v1": None if v1 is None else _v1_summary(v1),
    }
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA:
        raise ValueError("payload schema mismatch")
    for key in payload["runs"]:
        run = payload["runs"][key]
        curve = [point[1] for point in run["curve"]]
        for a, b in zip(curve, curve[1:], strict=False):
            if b < a and (a - b) / a > 1e-12:
                raise ValueError(f"hypervolume curve not monotone for {key}")
        if run["evaluations"] != payload["plan"]["evaluations_per_run"]:
            raise ValueError(f"budget mismatch for {key}")
    if len(payload["catalogue"]) != 96:
        raise ValueError("catalogue must have 96 designs")
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
<title>MDO L0 campaign v2 — the corrected L0 model over the screened design catalogue</title>
<style>
:root{--bg:#0f1419;--panel:#171d25;--ink:#e7ecf2;--muted:#9aa7b5;--line:#2a3441;--bo:#4fc3f7;--nsga:#ffb74d;--lhs:#a5d6a7;--warn:#ff7043;--ok:#66bb6a;--nom:#ce93d8;--v1:#b0bec5}
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
  <h1>MDO L0 campaign v2 — the corrected L0 model over the screened design catalogue</h1>
  <div id="headline" class="muted"></div>
</header>
<main>
  <section class="claim wide" id="claim"></section>
  <section class="wide" id="v1v2"></section>
  <section class="wide" id="hv"></section>
  <section id="hvtable"></section>
  <section id="gates"></section>
  <section class="wide" id="catalogue"></section>
  <section class="wide" id="fronts"></section>
  <section class="wide" id="closures"></section>
  <section class="wide" id="widths"></section>
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
const RAW = {qlognehvi:"#4fc3f7", nsga3:"#ffb74d", lhs:"#a5d6a7"};
const LABELS = {qlognehvi:"BoTorch qLogNEHVI (mixed GP)", nsga3:"pymoo NSGA-III (mixed variables)", lhs:"Latin hypercube over catalogue x operating point"};
const OBJ = ["axial_thrust_n","specific_impulse_s","thruster_electrical_to_beam_efficiency","anode_input_power_w"];
const OBJL = ["thrust [N]","Isp [s]","efficiency [1]","anode power [W]"];
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d) => (v === null || v === undefined) ? "—" : (typeof v === "number" ? (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(d ?? 3) : v.toPrecision(d ?? 5)) : String(v));
const badge = (ok, t) => `<span class="badge ${ok ? "ok" : "fail"}">${esc(t)}</span>`;
const el = id => document.getElementById(id);
const CAT = {}; for (const c of P.catalogue) CAT[c.catalogue_index] = c;
const geo = c => `L ${(c.geometry.chamber_length_m*1e3).toFixed(1)} mm, r_w ${(c.geometry.wall_radius_m*1e3).toFixed(2)} mm, ${c.geometry.stage_count} stages, pitch ${(c.geometry.stage_pitch_m*1e3).toFixed(1)} mm${c.geometry.has_divergent_exit ? ", divergent exit" : ""}`;

// ---- headline ---------------------------------------------------------------
el("headline").innerHTML = `terminal state <b>${esc(P.identity.terminal_state)}</b> · ${P.plan.run_ids.length} runs · ${P.campaign_result.total_evaluations} design evaluations (${P.campaign_result.infeasible_evaluations} infeasible) over ${P.pooled.distinct_catalogue_designs} of 96 catalogue designs · manifest <code>${P.identity.manifest_sha256.slice(0,12)}</code> · preregistration <code>${(P.identity.preregistration_commit||"").slice(0,12)}</code>`;

// ---- claim boundary -----------------------------------------------------------
{
  const cb = P.protocol.claim_boundary;
  el("claim").innerHTML = `<h2>Claim boundary — <code>${esc(P.protocol.classification)}</code></h2>
  <p><b>${esc(cb.statement)}</b></p>
  <p><b>Closure identification.</b> ${esc(cb.closure_identification)}</p>
  <p><b>Why no surrogate.</b> ${esc(cb.why_no_surrogate)}</p>
  <p><b>Why the catalogue is the design space.</b> ${esc(cb.why_the_catalogue_is_the_design_space)}</p>
  <p><b>Gate semantics.</b> ${esc(P.gates.semantics)}</p>
  <p class="muted"><b>Forbidden readings:</b> ${cb.forbidden_readings.map(esc).join("; ")}.</p>`;
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

// ---- v1 vs v2 -------------------------------------------------------------------
{
  const v1 = P.v1;
  if (!v1) { el("v1v2").innerHTML = `<h2>v1 versus v2</h2><p class="muted">v1 bundle not available at render time.</p>`; }
  else {
    const row = (label, a, b) => `<tr><td>${esc(label)}</td><td style="white-space:normal;text-align:left">${a}</td><td style="white-space:normal;text-align:left">${b}</td></tr>`;
    const hvcell = (t, st) => { const v = []; for (const k in t) if (k.startsWith(st + ":")) v.push(t[k].final_hypervolume.toExponential(3)); return v.join(" / "); };
    const attcell = (t, st) => { const v = []; for (const k in t) if (k.startsWith(st + ":")) v.push(fmt(t[k].attained_fraction, 3)); return v.join(" / "); };
    const T2 = {}; for (const k in P.runs) T2[k] = {final_hypervolume: P.runs[k].final_hypervolume, attained_fraction: P.runs[k].attained_fraction};
    el("v1v2").innerHTML = `<h2>v1 versus v2 (both read from their hash-bound bundles)</h2>
    <div class="scroll"><table><thead><tr><th></th><th>v1 — <code>${esc(v1.manifest_sha256.slice(0,12))}</code> (${esc(v1.terminal_state)})</th><th>v2 — <code>${esc(P.identity.manifest_sha256.slice(0,12))}</code> (${esc(P.identity.terminal_state)})</th></tr></thead><tbody>
    ${row("design space", esc(v1.design_space), `96 screened designs (categorical) x operating point (Ua, Ia, mdot)`)}
    ${row("cusp / wall-loss probabilities", esc(v1.cusp_probabilities), `per design: Jeffreys Beta posterior of the accepted-2N screening counts (4 cells x 128 launches); no surrogate`)}
    ${row("budget", `${v1.evaluations_per_run} per run (${v1.initial_design} initial), seeds ${v1.seeds.join(", ")}, ${v1.total_evaluations} total`, `${P.plan.evaluations_per_run} per run (${P.plan.initial_design} initial), seeds ${P.seeds.join(", ")}, ${P.campaign_result.total_evaluations} total`)}
    ${row("dense reference robust HV", `${v1.dense_reference.robust_hypervolume.toExponential(4)} (${v1.dense_reference.count} points)`, `${P.dense_reference.robust_hypervolume.toExponential(4)} (${P.dense_reference.count} points = 96 x ${P.dense_reference.points_per_design}; front on designs ${P.dense_reference.robust_front_catalogue_indices.join(", ")})`)}
    ${row("qLogNEHVI final HV (seeds)", hvcell(v1.hypervolume_table, "qlognehvi"), hvcell(T2, "qlognehvi"))}
    ${row("NSGA-III final HV (seeds)", hvcell(v1.hypervolume_table, "nsga3"), hvcell(T2, "nsga3"))}
    ${row("LHS final HV (seeds)", hvcell(v1.hypervolume_table, "lhs"), hvcell(T2, "lhs"))}
    ${row("qLogNEHVI attained fraction of dense ref.", attcell(v1.hypervolume_table, "qlognehvi"), attcell(T2, "qlognehvi"))}
    ${row("BO beats LHS / NSGA-III (seed counts)", `${v1.bo_beats_random.wins}/${v1.bo_beats_random.seeds} · ${v1.bo_beats_nsga3.wins}/${v1.bo_beats_nsga3.seeds}`, `${P.gates.bo_beats_random.wins}/${P.gates.bo_beats_random.seeds} · ${P.gates.bo_beats_nsga3.wins}/${P.gates.bo_beats_nsga3.seeds}`)}
    ${row("robust / nominal front (shared, Jaccard)", `${v1.robust_vs_nominal.robust_front_size} / ${v1.robust_vs_nominal.nominal_front_size} (${v1.robust_vs_nominal.shared_designs}, ${fmt(v1.robust_vs_nominal.jaccard,3)})`, `${P.gates.robust_vs_nominal.robust_front_size} / ${P.gates.robust_vs_nominal.nominal_front_size} (${P.gates.robust_vs_nominal.shared_designs}, ${fmt(P.gates.robust_vs_nominal.jaccard,3)})`)}
    ${row("binding gates", `${v1.binding_gates} (integrity; audit e9f9af16 ACCEPTED WITH DISCLOSURES F9 F10 F22 F26 F27 F28)`, `${Object.keys(P.gates.binding).length} (integrity; the six v1 disclosures closed: ${Object.entries(P.protocol.v1_audit_disclosures_closed).map(([k,v]) => `<b>${esc(k)}</b> ${esc(v)}`).join("; ")})`)}
    </tbody></table></div>
    <p class="muted">Hypervolumes share v1's reference point, comparison scales and normalisation, so they are directly comparable; v2's are smaller because the screened per-cell P(wall) (pooled 0.375–0.869) give a CL-1 survival far below v1's wide-prior mean 0.36, and 73 of 96 designs have at least one saturated cell (128/128) and contribute ~nothing.</p>`;
  }
}

// ---- hypervolume curves -----------------------------------------------------
{
  const W = 900, H = 340, x0 = 70, y0 = 300, x1 = 880, y1 = 20;
  const allHv = []; for (const k in P.runs) for (const p of P.runs[k].curve) allHv.push(p[1]);
  const ys = {min: 0, max: Math.max(extent(allHv).max, P.dense_reference.robust_hypervolume) * 1.05};
  const xs = {min: 0, max: P.plan.evaluations_per_run};
  let s = svgOpen(W, H) + axis(x0, y0, x1, y0) + axis(x0, y0, x0, y1);
  s += ticks(xs, 8, true, x0, y0, x1, y1, t => t.toFixed(0)) + ticks(ys, 5, false, x0, y0, x1, y1, t => t.toExponential(1));
  const yref = lin(P.dense_reference.robust_hypervolume, ys, y0, y1);
  s += `<line x1="${x0}" y1="${yref}" x2="${x1}" y2="${yref}" stroke="#9aa7b5" stroke-dasharray="4 4"/><text x="${x1}" y="${yref-4}" fill="#9aa7b5" font-size="10" text-anchor="end">dense reference (96 x ${P.dense_reference.points_per_design}) robust HV = ${P.dense_reference.robust_hypervolume.toExponential(3)}</text>`;
  const xinit = lin(P.plan.initial_design, xs, x0, x1);
  s += `<line x1="${xinit}" y1="${y0}" x2="${xinit}" y2="${y1}" stroke="#2a3441"/><text x="${xinit+3}" y="${y1+10}" fill="#9aa7b5" font-size="10">shared initial design (${P.plan.initial_design})</text>`;
  for (const k in P.runs){
    const r = P.runs[k];
    const pts = r.curve.map(p => `${lin(p[0], xs, x0, x1).toFixed(1)},${lin(p[1], ys, y0, y1).toFixed(1)}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="${RAW[r.strategy]}" stroke-width="1.6" opacity="0.9"/>`;
  }
  s += `<text x="${(x0+x1)/2}" y="${H-4}" fill="#9aa7b5" font-size="11" text-anchor="middle">design evaluations (each = 64 posterior draws + 1 nominal L0 point of the chosen catalogue design)</text>`;
  s += `<text transform="translate(14,${(y0+y1)/2}) rotate(-90)" fill="#9aa7b5" font-size="11" text-anchor="middle">robust hypervolume (v1 frame)</text></svg>`;
  el("hv").innerHTML = `<h2>Hypervolume versus evaluations (robust CVaR objectives, feasible nondominated set)</h2>
  <div class="legend">${P.strategies.map(st => `<span><i style="background:${RAW[st]}"></i>${esc(LABELS[st])} (seeds ${P.seeds.join(", ")})</span>`).join("")}</div>${s}
  <p class="muted">Binding gate <code>hypervolume_monotone</code> ${badge(P.gates.binding.hypervolume_monotone, P.gates.binding.hypervolume_monotone ? "passed" : "FAILED")} (largest relative roundoff decrease ${fmt(P.gates.binding_detail.hypervolume_monotone_largest_relative_decrease, 2)}, tolerance 1e-12).</p>`;
}

// ---- hypervolume table + seed variance ---------------------------------------
{
  let rows = "";
  for (const st of P.strategies) for (const seed of P.seeds){
    const r = P.runs[`${st}:${seed}`];
    rows += `<tr><td style="color:${RAW[st]}">${esc(LABELS[st])}</td><td>${seed}</td><td>${r.final_hypervolume.toExponential(4)}</td><td>${fmt(r.attained_fraction, 3)}</td><td>${r.pareto_set_size}</td><td>${r.pareto_catalogue_indices.join(", ")}</td><td>${r.distinct_catalogue_designs}</td><td>${r.infeasible}</td><td>${fmt(r.wall_clock_seconds, 4)}</td></tr>`;
  }
  let vrows = "";
  for (const st of P.strategies){ const v = P.seed_variance[st]; vrows += `<tr><td style="color:${RAW[st]}">${esc(LABELS[st])}</td><td>${v.mean.toExponential(4)}</td><td>${v.minimum.toExponential(4)}</td><td>${v.maximum.toExponential(4)}</td><td>${v.sample_std === null ? "—" : v.sample_std.toExponential(3)}</td></tr>`; }
  const br = P.gates.bo_beats_random, bn = P.gates.bo_beats_nsga3;
  el("hvtable").innerHTML = `<h2>Final hypervolume per optimiser × seed</h2><div class="scroll"><table><thead><tr><th>strategy</th><th>seed</th><th>final HV</th><th>fraction of dense ref.</th><th>Pareto set</th><th>Pareto catalogue designs</th><th>distinct designs visited</th><th>infeasible</th><th>wall [s]</th></tr></thead><tbody>${rows}</tbody></table></div>
  <h3>Seed-repeat variance of the final hypervolume</h3><div class="scroll"><table><thead><tr><th>strategy</th><th>mean</th><th>min</th><th>max</th><th>sample std</th></tr></thead><tbody>${vrows}</tbody></table></div>
  <h3>Predeclared comparisons (reported, not binding; counts of seeds, not significance)</h3>
  <p>${badge(br.passed, `BO beats LHS: ${br.wins}/${br.seeds} seeds (rule ≥ ${br.required_wins})`)} ${badge(bn.passed, `BO beats NSGA-III: ${bn.wins}/${bn.seeds} seeds`)}</p>
  <p class="muted">${esc(br.statement)}</p>`;
}

// ---- gates ------------------------------------------------------------------
{
  const g = P.gates.binding;
  const names = Object.keys(g);
  el("gates").innerHTML = `<h2>Gates (recording-integrity gates; acceptance ≠ efficacy)</h2><p>${badge(P.gates.all_binding_passed, P.gates.all_binding_passed ? "all binding gates passed" : "binding gate failed")} terminal state <code>${esc(P.identity.terminal_state)}</code></p>
  <div class="scroll"><table><thead><tr><th>binding gate</th><th>result</th><th>declaration</th></tr></thead><tbody>${names.map(n => `<tr><td>${esc(n)}</td><td>${badge(g[n], g[n] ? "pass" : "FAIL")}</td><td style="white-space:normal;text-align:left" class="muted">${esc(P.protocol.gates.binding[n] || "")}</td></tr>`).join("")}</tbody></table></div>
  <p class="muted">Replayed ${P.gates.binding_detail.replayed} records bit-exactly; import scope ${P.gates.binding_detail.import_scope_imported_count} files = hash scope; NSGA-III duplicates ${Object.entries(P.gates.binding_detail.nsga3_duplicates).map(([k,v]) => `${esc(k)} ${v.duplicates}`).join(", ")}. Terminal rule: ${esc(P.protocol.gates.terminal_rule)}</p>`;
}

// ---- catalogue: which designs win under CL-1 and their P(wall) / geometry ----------
{
  const members = P.pooled.robust.catalogue_membership;
  const denseHV = {}; for (const d of P.dense_reference.per_design) denseHV[d.catalogue_index] = d;
  const W = 900, H = 320, x0 = 70, y0 = 280, x1 = 880, y1 = 16;
  const xs = {min: 0.3, max: 0.9};
  const hvs = P.dense_reference.per_design.map(d => d.robust_hypervolume).filter(v => v > 0);
  const logs = hvs.map(v => Math.log10(v));
  const ys = {min: Math.floor(Math.min(...logs)), max: Math.ceil(Math.max(...logs))};
  let s = svgOpen(W, H) + axis(x0, y0, x1, y0) + axis(x0, y0, x0, y1);
  s += ticks(xs, 6, true, x0, y0, x1, y1, t => t.toFixed(2)) + ticks(ys, ys.max - ys.min, false, x0, y0, x1, y1, t => "1e" + t.toFixed(0));
  const onFront = new Set(members.map(m => m.catalogue_index));
  for (const d of P.dense_reference.per_design){
    const c = CAT[d.catalogue_index];
    if (!(d.robust_hypervolume > 0)) continue;
    const x = lin(c.pooled, xs, x0, x1), y = lin(Math.log10(d.robust_hypervolume), ys, y0, y1);
    const hit = onFront.has(d.catalogue_index);
    const sat = c.cells.some(p => p >= 1);
    s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${hit ? 5 : 3}" fill="${hit ? "#4fc3f7" : (sat ? "none" : "#9aa7b5")}" stroke="${hit ? "#4fc3f7" : "#9aa7b5"}" stroke-width="1.2" opacity="0.9"><title>design ${d.catalogue_index} ${esc(c.case_id)}: pooled P(wall) ${c.pooled}, cells ${c.cells.join("/")}, dense robust HV ${d.robust_hypervolume.toExponential(2)}, ${esc(geo(c))}</title></circle>`;
    if (hit) s += `<text x="${(x+7).toFixed(1)}" y="${(y+3).toFixed(1)}" fill="#4fc3f7" font-size="10">${d.catalogue_index}</text>`;
  }
  s += `<text x="${(x0+x1)/2}" y="${H-2}" fill="#9aa7b5" font-size="11" text-anchor="middle">screening pooled P(wall) of the design (accepted-2N, 512 launches)</text>`;
  s += `<text transform="translate(14,${(y0+y1)/2}) rotate(-90)" fill="#9aa7b5" font-size="11" text-anchor="middle">per-design dense robust hypervolume under CL-1 (log10)</text></svg>`;
  const rows = members.map(m => `<tr><td>${m.catalogue_index}</td><td class="mono">${esc(m.case_id)}</td><td>${m.front_members}</td><td>${m.pooled} [${m.pooled_wilson[0]}, ${m.pooled_wilson[1]}]</td><td class="mono">${m.cells.join(" / ")}</td><td>${fmt(m.nominal_survival_cl1,3)}</td><td>${denseHV[m.catalogue_index] ? denseHV[m.catalogue_index].robust_hypervolume.toExponential(3) : "—"}</td><td style="white-space:normal;text-align:left">${esc(geo(CAT[m.catalogue_index]))}</td></tr>`).join("");
  el("catalogue").innerHTML = `<h2>Catalogue designs on the pooled robust front — the geometry → performance link under CL-1</h2>
  <p>${members.length} of 96 screened designs carry the ${P.pooled.robust.front_size}-point pooled robust front (${P.pooled.unique_designs} unique evaluated designs over ${P.pooled.distinct_catalogue_designs} catalogue members). Every P(wall) below is the screening estimate (SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS), not a plasma quantity; a design "wins" under the declared closure only.</p>
  <div class="scroll"><table><thead><tr><th>design</th><th>case id</th><th>front members</th><th>pooled P(wall) [Wilson 95]</th><th>cells 1–4 P(wall)</th><th>nominal S (CL-1)</th><th>dense robust HV</th><th>sealed geometry</th></tr></thead><tbody>${rows}</tbody></table></div>
  <div class="legend"><span><i style="background:#4fc3f7"></i>on the pooled robust front</span><span><i style="background:#9aa7b5"></i>no saturated cell</span><span><i style="border:1px solid #9aa7b5;background:none"></i>at least one cell 128/128 (open)</span></div>${s}
  <p class="muted">Dense reference: every catalogue design × the fixed ${P.dense_reference.points_per_design}-point operating grid (${P.dense_reference.count} evaluations, ${P.dense_reference.feasible} feasible, ${fmt(P.dense_reference.evaluation_seconds,4)} s on ${P.dense_reference.workers} workers); its robust front (${P.dense_reference.robust_front_size} points) lies on designs ${P.dense_reference.robust_front_catalogue_indices.join(", ")}. Hover a point for the design's counts and geometry.</p>`;
}

// ---- fronts: robust vs nominal (two projections) -----------------------------
function scatter(title, rows, ix, iy){
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
  return `<div><h3>${esc(title)}</h3>${s}</div>`;
}
{
  const robust = P.pooled.robust.designs.map(d => d.objectives);
  const nominal = P.pooled.nominal.designs.map(d => d.objectives);
  const robustNominal = P.pooled.robust.nominal_of_robust_front.map(d => d.objectives);
  const dense = P.dense_reference.robust_front.map(d => d.objectives);
  const rows = [
    {points: dense, color: "#4a5666", fill: true, r: 2, opacity: 0.8},
    {points: nominal, color: "#ce93d8", fill: false, r: 3},
    {points: robustNominal, color: "#ffffff", fill: false, r: 2, opacity: 0.6},
    {points: robust, color: "#4fc3f7", fill: true, r: 3},
  ];
  const rvn = P.gates.robust_vs_nominal;
  el("fronts").innerHTML = `<h2>Pooled Pareto fronts: robust (CVaR) versus nominal (posterior means), with the dense-reference robust front</h2>
  <div class="legend"><span><i style="background:#4fc3f7"></i>robust front (${P.pooled.robust.front_size} designs, HV ${P.pooled.robust.hypervolume.toExponential(3)}, catalogue ${P.pooled.robust.catalogue_indices.join(", ")})</span><span><i style="background:#ce93d8"></i>nominal front (${P.pooled.nominal.front_size} designs, HV ${P.pooled.nominal.hypervolume.toExponential(3)}, catalogue ${P.pooled.nominal.catalogue_indices.join(", ")})</span><span><i style="background:#fff"></i>robust-front designs at their nominal theta</span><span><i style="background:#4a5666"></i>dense-reference robust front (${P.dense_reference.robust_front_size})</span></div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px">
  ${scatter("thrust versus anode power", rows, 3, 0)}
  ${scatter("efficiency versus Isp", rows, 1, 2)}
  ${scatter("thrust versus Isp", rows, 1, 0)}
  </div>
  <p>Shared designs between the robust and nominal fronts: ${rvn.shared_designs} (Jaccard ${fmt(rvn.jaccard, 3)}); ${rvn.nominal_front_members_robust_feasible} of the ${rvn.nominal_front_size} nominal-front designs are robust-feasible. ${esc(P.protocol.robust_formulation.predeclared_expectation)} Per-design separability ${badge(P.gates.per_design_separability.passed, P.gates.per_design_separability.passed ? "holds" : "violated")} over ${P.gates.per_design_separability.designs} designs.</p>`;
}

// ---- CL-1 vs CL-2 -----------------------------------------------------------------
{
  const c2 = P.sensitivity.closure_cl2;
  const cl1set = new Set(c2.campaign_front_catalogue_indices), cl2set = new Set(c2.front_catalogue_indices);
  const both = [...cl1set].filter(k => cl2set.has(k)), only1 = [...cl1set].filter(k => !cl2set.has(k)), only2 = [...cl2set].filter(k => !cl1set.has(k));
  const list = ks => ks.length ? ks.map(k => `<span title="${esc(geo(CAT[k]))}">${k} <span class="muted">(pooled ${CAT[k].pooled}, cells ${CAT[k].cells.join("/")})</span></span>`).join("; ") : "—";
  el("closures").innerHTML = `<h2>Closure sensitivity: CL-1 (per-cell product) versus CL-2 (pooled survival)</h2>
  <p class="muted">${esc(P.protocol.closures["CL-2"].statement)} — every recorded design re-evaluated under CL-2 with the campaign posterior width; reported, not gated.</p>
  <div class="scroll"><table><thead><tr><th></th><th>CL-1 (campaign)</th><th>CL-2 (sensitivity)</th></tr></thead><tbody>
  <tr><td>feasible / infeasible pooled designs</td><td>${P.pooled.robust.candidates} / ${P.pooled.unique_designs - P.pooled.robust.candidates}</td><td>${c2.feasible} / ${c2.infeasible}</td></tr>
  <tr><td>robust front size</td><td>${P.pooled.robust.front_size}</td><td>${c2.front_size}</td></tr>
  <tr><td>robust hypervolume</td><td>${c2.campaign_hypervolume.toExponential(4)}</td><td>${c2.hypervolume.toExponential(4)}</td></tr>
  <tr><td>catalogue designs on the front</td><td style="white-space:normal;text-align:left">${c2.campaign_front_catalogue_indices.join(", ")}</td><td style="white-space:normal;text-align:left">${c2.front_catalogue_indices.join(", ")}</td></tr>
  <tr><td>shared front designs / Jaccard</td><td colspan="2" style="text-align:left">${c2.shared_with_campaign_front} shared (Jaccard ${fmt(c2.jaccard_with_campaign_front,3)}); on the common feasible set the fronts are ${c2.identical_on_common_feasible_set_up_to_ties ? "identical up to ties" : `different (${c2.common_front_symmetric_difference} designs)`}</td></tr>
  </tbody></table></div>
  <p><b>Catalogue designs on both fronts:</b> ${list(both)}<br><b>CL-1 only:</b> ${list(only1)}<br><b>CL-2 only:</b> ${list(only2)}</p>
  <p class="muted">Under CL-2 a design with one saturated cell keeps the survival 1 − P(wall, pooled) of its other cells, so many more designs are competitive (and more operating points are infeasible through the larger beam current); under CL-1 one saturated cell removes the design. The overlap is the measure of how much the recorded front depends on the declared closure.</p>`;
}

// ---- uncertainty width -----------------------------------------------------------
{
  const rows = P.sensitivity.widths.map(w => `<tr><td>${esc(String(w.width_scale))}${w.is_campaign_posterior ? " (campaign)" : ""}</td><td style="white-space:normal;text-align:left" class="muted">${esc(w.meaning)}</td><td>${fmt(w.survival_min,3)} – ${fmt(w.survival_max,3)}</td><td>${w.feasible} / ${w.infeasible}</td><td>${w.front_size}</td><td>${w.hypervolume.toExponential(3)}</td><td>${w.shared_with_campaign_front}</td><td>${fmt(w.jaccard_with_campaign_front,3)}</td><td>${badge(w.identical_on_common_feasible_set_up_to_ties, w.identical_on_common_feasible_set_up_to_ties ? "identical" : `differs (${w.common_front_symmetric_difference})`)}</td><td style="white-space:normal;text-align:left">${w.front_catalogue_indices.join(", ")}</td></tr>`).join("");
  el("widths").innerHTML = `<h2>Sensitivity of the robust front to the binomial uncertainty width of the screening counts</h2>
  <p class="muted">${esc(P.protocol.uncertain_inputs.sensitivity_widths.rule)}</p>
  <div class="scroll"><table><thead><tr><th>width scale w</th><th>meaning</th><th>CL-1 survival range over the sample</th><th>feasible / infeasible</th><th>front size</th><th>HV</th><th>shared with campaign front</th><th>Jaccard</th><th>common-set front</th><th>catalogue designs</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

// ---- timing -------------------------------------------------------------------
{
  let rows = "";
  for (const st of P.strategies) for (const seed of P.seeds){ const r = P.runs[`${st}:${seed}`]; rows += `<tr><td style="color:${RAW[st]}">${esc(LABELS[st])}</td><td>${seed}</td><td>${fmt(r.timing.wall_clock_seconds,4)}</td><td>${fmt(r.timing.evaluation_seconds,3)}</td><td>${r.bo_iterations.length ? fmt(r.timing.bo_fit_seconds,3) : "—"}</td><td>${r.bo_iterations.length ? fmt(r.timing.bo_acquisition_seconds,4) : "—"}</td></tr>`; }
  const bo = P.seeds.map(sd => P.runs["qlognehvi:"+sd].bo_iterations);
  const W = 420, H = 200, x0 = 50, y0 = 170, x1 = 410, y1 = 12;
  const allAcq = bo.flatMap(l => l.map(e => e.candidate_stage_seconds + e.refinement_seconds));
  const ys = {min: 0, max: extent(allAcq).max * 1.1}, xs = {min: 1, max: Math.max(...bo.map(l => l.length))};
  let s = svgOpen(W, H) + axis(x0, y0, x1, y0) + axis(x0, y0, x0, y1) + ticks(xs, Math.min(8, xs.max-1), true, x0, y0, x1, y1, t => t.toFixed(0)) + ticks(ys, 4, false, x0, y0, x1, y1, t => t.toFixed(0));
  bo.forEach((l, i) => { s += `<polyline points="${l.map(e => `${lin(e.iteration, xs, x0, x1).toFixed(1)},${lin(e.candidate_stage_seconds + e.refinement_seconds, ys, y0, y1).toFixed(1)}`).join(" ")}" fill="none" stroke="#4fc3f7" stroke-width="1.4" opacity="${0.5 + 0.25*i}"/>`; });
  s += `<text x="${(x0+x1)/2}" y="${H-2}" fill="#9aa7b5" font-size="10" text-anchor="middle">BO iteration (batch of ${P.plan.qlognehvi_batch_size})</text><text transform="translate(12,${(y0+y1)/2}) rotate(-90)" fill="#9aa7b5" font-size="10" text-anchor="middle">candidate + refinement seconds</text></svg>`;
  const accepted = bo.map(l => l.reduce((a, e) => a + e.refinements_accepted, 0));
  el("timing").innerHTML = `<h2>Timing</h2><div class="scroll"><table><thead><tr><th>strategy</th><th>seed</th><th>wall [s]</th><th>L0 eval [s]</th><th>GP fit [s]</th><th>acquisition [s]</th></tr></thead><tbody>${rows}</tbody></table></div>
  <h3>qLogNEHVI acquisition seconds per iteration (three seeds)</h3>${s}
  <p class="muted">Continuous refinements accepted per seed: ${accepted.join(" / ")} of ${P.plan.qlognehvi_iterations * P.plan.qlognehvi_batch_size}. Assessment ${fmt(P.campaign_result.assessment_seconds,4)} s. Device: ${esc(P.protocol.optimizers.qlognehvi.device)}, torch threads ${P.runs["qlognehvi:"+P.seeds[0]].labels.torch_threads}.</p>`;
}

// ---- protocol summary -----------------------------------------------------------
{
  const pr = P.protocol;
  const q = pr.optimizers.qlognehvi, n = pr.optimizers.nsga3;
  el("protocol").innerHTML = `<h2>Protocol (frozen at preregistration)</h2>
  <h3>Design space</h3><ul><li><code>${esc(pr.design_space.catalogue.name)}</code>: ${esc(pr.design_space.catalogue.statement)}</li>${pr.design_space.operating_point.map(v => `<li><code>${esc(v.name)}</code> ∈ [${v.lower}, ${v.upper}] ${esc(v.units)}</li>`).join("")}</ul>
  <h3>Uncertain inputs (frozen 64-row QMC sample per design; unit rows <code>${esc(pr.uncertain_inputs.sample.unit_rows_sha256.slice(0,12))}</code>, catalogue sample <code>${esc(pr.uncertain_inputs.sample.catalogue_sample_sha256.slice(0,12))}</code>)</h3><ul>${pr.uncertain_inputs.per_design_inputs.map(u => `<li><code>${esc(u.name)}</code>: ${esc(u.parameters)}</li>`).join("")}${pr.uncertain_inputs.shared_inputs.map(u => `<li><code>${esc(u.name)}</code> ∈ [${u.lower}, ${u.upper}] — ${esc(u.meaning)}</li>`).join("")}</ul>
  <p class="muted">Nominal: ${esc(pr.uncertain_inputs.nominal)}</p>
  <h3>Closures</h3><ul><li><b>CL-1</b> ${esc(pr.closures["CL-1"].statement)} <span class="muted">${esc(pr.closures["CL-1"].status)}</span></li><li><b>CL-2</b> ${esc(pr.closures["CL-2"].statement)}</li></ul>
  <h3>Objectives</h3><ul>${pr.objectives.map(o => `<li><code>${esc(o.name)}</code> ${esc(o.direction)} [${esc(o.units)}], scale ${o.comparison_scale}</li>`).join("")}</ul>
  <h3>Constraint</h3><ul>${pr.constraints.map(c => `<li><code>${esc(c.name)}</code> ${esc(c.sense)} ${c.threshold} ${esc(c.units)} (${esc(c.role)})</li>`).join("")}</ul>
  <h3>Robust formulation</h3><p class="muted">${esc(pr.robust_formulation.definition)}</p>
  <h3>Optimisers and budget</h3><ul>
  <li><b>qLogNEHVI</b>: ${esc(q.model)} — ${esc(q.candidate_stage)} — ${esc(q.refinement_stage)} — MC ${q.mc_samples}, batch ${pr.budget.qlognehvi_batch_size} × ${pr.budget.qlognehvi_iterations} after ${pr.budget.initial_design} initial points. <span class="muted">${esc(q.why_this_optimiser)}</span></li>
  <li><b>NSGA-III</b>: ${esc(n.variables)}; ${esc(n.reference_directions)}; population ${n.population_size} × ${n.generations} generations; duplicate elimination: ${esc(n.eliminate_duplicates_implementation)}</li>
  <li><b>LHS</b>: ${esc(pr.optimizers.lhs.design)}</li>
  <li>${pr.budget.evaluations_per_run} evaluations per run, seeds ${pr.budget.seeds.join(", ")}, ${pr.budget.total_evaluations} in total. <span class="muted">${esc(pr.budget.sizing)}</span></li></ul>
  <h3>Dense reference</h3><p class="muted">${esc(pr.dense_reference.method)} ${esc(pr.dense_reference.cost_note)} ${esc(pr.dense_reference.parallelism_note)}</p>`;
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
  <tr><td>source hash (${id.source_files} import-bound files; scope = imports ${id.import_scope_matches ? "✓" : "✗"})</td><td class="mono">${esc(id.source_sha256)}</td></tr>
  <tr><td>catalogue (96 designs) SHA-256</td><td class="mono">${esc(id.catalogue_sha256)}</td></tr>
  <tr><td>screening dataset SHA-256 / result commit</td><td class="mono">${esc(id.catalogue_binding.dataset_file_sha256)} / ${esc(id.catalogue_binding.screening_result_commit.slice(0,12))} (${id.catalogue_binding.passed ? "bound" : "UNBOUND"})</td></tr>
  <tr><td>packages</td><td class="mono">${Object.entries(id.package_versions).map(([k,v]) => `${esc(k)} ${esc(v)}`).join(", ")}</td></tr>
  <tr><td>CUDA probe (recorded, not used)</td><td class="mono">${esc(id.device_probes.cuda.available ? `${id.device_probes.cuda.device_name}, torch ${id.device_probes.cuda.torch_version}, CUDA ${id.device_probes.cuda.cuda_version}` : "unavailable")}</td></tr>
  <tr><td>v1 bundle</td><td class="mono">${P.v1 ? `${esc(P.v1.manifest_sha256)} (${esc(P.v1.results_commit)})` : "—"}</td></tr>
  </tbody></table>`;
  el("footer").textContent = `${P.schema} · generated from the immutable results bundles of modern/experiments/mdo_l0_campaign_v2 (manifest ${id.manifest_sha256.slice(0,16)}) and mdo_l0_campaign_v1 · offline, no external resources · every number on this page is read from a bundle or a sealed protocol.`;
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
    v1_results: Path | None = V1_RESULTS,
) -> dict[str, Any]:
    bundle = load_v2_bundle(results, expected_manifest_sha256)
    v1 = load_v1_bundle(v1_results) if v1_results is not None and (v1_results / "manifest.json").is_file() else None
    payload = build_payload(bundle, v1)
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
        "v1_manifest_sha256": None if v1 is None else v1.manifest_sha256,
        "terminal_state": bundle.state,
        "html_sha256": _digest(data),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-pin", action="store_true", help="skip the pinned manifest identity (development against a shakedown bundle)")
    arguments = parser.parse_args(argv)
    report = generate(arguments.results, arguments.output, expected_manifest_sha256=None if arguments.no_pin else EXPECTED_MANIFEST_SHA256)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
