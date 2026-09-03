"""Generate the offline L1b HEMP confirmation dashboard (material-aware P2 check of 15 L1a designs).

Every number shown is read from the immutable, hash-bound results bundles of
``modern/experiments/l1b_hemp_confirmation_v1_1`` (the accepted campaign) and
``modern/experiments/l1b_hemp_confirmation_v1`` (the recorded development rejection it
supersedes). The generator byte-verifies every manifest entry of both bundles, re-derives the
verdict and the headline counts from the per-design rows and refuses to render on any
inconsistency. It emits no wall-clock timestamps or machine paths, so identical inputs produce
identical bytes.

Labels carried everywhere: ``P2_MATERIAL_AWARE_FIELD_CONFIRMATION_NOT_HARDWARE_VALID`` and
``SCREENING_P2_MATERIAL_FIELD_SEPARATRIX_CUSP_TOPOLOGY``. Nothing is a plasma,
mirror-probability, wall-loss or performance claim; the verdict is a statement about the L1a
cusp topology and rho classification under a linear-iron P2 field. Paper admission is not in
scope.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
EXPERIMENT = MODERN / "experiments" / "l1b_hemp_confirmation_v1_1"
RESULTS = EXPERIMENT / "results"
V1_EXPERIMENT = MODERN / "experiments" / "l1b_hemp_confirmation_v1"
V1_RESULTS = V1_EXPERIMENT / "results"
TEMPLATE_PATH = HERE / "l1b-hemp-confirmation-v1.template.html"
DEFAULT_OUTPUT = HERE / "l1b-hemp-confirmation-v1.html"

SCHEMA = "cft-revival.l1b-hemp-confirmation-v1-dashboard/1.0.0"
CLASSIFICATION = "P2_MATERIAL_AWARE_FIELD_CONFIRMATION_NOT_HARDWARE_VALID"
TOPOLOGY_LABEL = "SCREENING_P2_MATERIAL_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
VERDICTS = ("CONFIRMED", "PARTIALLY_CONFIRMED", "DISCONFIRMED")
MAX_HTML_BYTES = 4_000_000
MAX_OVERLAYS = 6


# --------------------------------------------------------------------------- #
# Strict loading and bundle verification
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


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = _load_json_bytes(path.read_bytes(), label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def verify_bundle(results: Path, *, expected_state: str = "accepted_result") -> dict[str, Any]:
    manifest = _load_object(results / "manifest.json", "manifest.json")
    if manifest.get("state") != expected_state:
        raise ValueError(f"bundle state is {manifest.get('state')!r}, not {expected_state}")
    verified = 0
    for entry in manifest["artifacts"]:
        if entry.get("type") != "file":
            continue
        raw = (results / entry["path"]).read_bytes()
        if sha256(raw).hexdigest() != entry["byte_sha256"] or len(raw) != entry["bytes"]:
            raise ValueError(f"SHA-256 / size mismatch for {entry['path']}")
        verified += 1
    terminal_raw = (results / "terminal.json").read_bytes()
    lock_raw = (results / "execution-lock.json").read_bytes()
    if sha256(terminal_raw).hexdigest() != manifest["terminal_byte_sha256"]:
        raise ValueError("terminal.json does not match the manifest")
    if sha256(lock_raw).hexdigest() != manifest["lock_byte_sha256"]:
        raise ValueError("execution-lock.json does not match the manifest")
    lock = _load_json_bytes(lock_raw, "execution-lock.json")
    terminal = _load_json_bytes(terminal_raw, "terminal.json")
    if terminal["state"] != expected_state:
        raise ValueError(f"terminal state is not {expected_state}")
    return {
        "manifest_file_sha256": sha256((results / "manifest.json").read_bytes()).hexdigest(),
        "terminal_file_sha256": manifest["terminal_byte_sha256"],
        "lock_file_sha256": manifest["lock_byte_sha256"],
        "experiment_id": manifest["experiment_id"],
        "state": manifest["state"],
        "preregistration_commit_sha": lock["commit"],
        "execution_command": lock["command"],
        "device": lock["device"],
        "artifact_count": manifest["artifact_count"],
        "verified_file_count": verified,
    }


def _artifact(results: Path, relative: str) -> dict[str, Any]:
    return _load_object(results / "artifacts" / relative, relative)


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _design_row(row: Mapping[str, Any]) -> dict[str, Any]:
    comparison = row["comparison"]
    p2 = row["p2"]
    return {
        "id": row["design_id"],
        "ordinal": row["ordinal"],
        "representative": row["representative"],
        "stages": row["derived"]["stage_count"],
        "pitch_m": row["derived"]["represented_stage_pitch_m"],
        "rw_m": row["geometry"]["wall_radius_m"],
        "x_w": row["derived"]["x_w"],
        "rw_over_L": row["derived"]["wall_radius_over_pitch"],
        "scale": comparison["source_strength_scale"],
        "l1a_nulls": comparison["l1a_axis_null_count"],
        "p2_nulls": comparison["p2_axis_null_count"],
        "l1a_cusps": comparison["l1a_wall_cusp_count"],
        "p2_cusps": comparison["p2_wall_cusp_count"],
        "l1a_cells": comparison["l1a_cell_count"],
        "p2_cells": comparison["p2_cell_count"],
        "strict": comparison["count_agreement_strict"],
        "boundary_tolerant": comparison["count_agreement_boundary_tolerant"],
        "bijection": comparison["cusp_match"]["bijection"],
        "matched": comparison["matched_cusp_count"],
        "max_shift_m": comparison["max_cusp_shift_m"],
        "max_shift_tol": comparison["max_cusp_shift_over_tolerance"],
        "tolerance_m": comparison["cusp_position_tolerance_m"],
        "channel_null_shift_m": comparison["channel_axis_nulls"]["max_sorted_shift_m"],
        "channel_null_bijection": comparison["channel_axis_null_match"]["bijection"],
        "lean_l1a_m": comparison["separatrix_lean_m"]["l1a_max"],
        "lean_p2_m": comparison["separatrix_lean_m"]["p2_max"],
        "outside_null_shifts_m": comparison["outside_channel_axis_nulls"]["shifts_m"],
        "wall_ratio": comparison["peak_wall_b_ratio_p2_over_l1a"],
        "wall_ratio_unscaled": comparison["peak_wall_b_ratio_unscaled"],
        "axis_ratio": comparison["axis_peak_b_ratio_p2_over_l1a"],
        "cusp_wall_ratios": [pair["wall_b_ratio_p2_over_l1a"] for pair in comparison["matched_cusps"]],
        "l1a_rho": [item["rho_conservative"] for item in row["l1a"]["rho"]],
        "p2_rho": [item["rho_conservative"] for item in row["p2_rho"]],
        "l1a_min_rho": comparison["l1a_min_rho_conservative"],
        "p2_min_rho": comparison["p2_min_rho_conservative"],
        "l1a_hemp": comparison["l1a_hemp_like_all_cusps"],
        "p2_hemp": comparison["p2_hemp_like_all_cusps"],
        "l1a_b3_b1": row["l1a"]["wall_harmonics"].get("b3_over_b1"),
        "p2_b3_b1": row["p2_wall_harmonics"].get("b3_over_b1"),
        "dofs": [level["p2_dofs"] for level in p2["levels"]],
        "iterations": [level["iterations"] for level in p2["levels"]],
        "residual": max(level["relative_true_residual_l2"] for level in p2["levels"]),
        "converged": p2["all_levels_converged"],
        "disc_shift_m": row["p2_discretisation"]["max_wall_intersection_shift_m"],
        "disc_null_shift_m": row["p2_discretisation"]["max_axis_null_shift_m"],
        "sampling_stable": row["sampling_stability"]["stable"],
        "sampling_shift_m": row["sampling_stability"]["max_wall_intersection_shift_m"],
        "seconds": p2["total_seconds"],
        "rss_bytes": p2["peak_rss_bytes"],
        "gates": all(row["gate_checks"].values()),
        "record_path": row["record_path"],
    }


def _scatter(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        comparison = row["comparison"]
        for pair in comparison["matched_cusps"]:
            points.append(
                {
                    "id": row["design_id"],
                    "stages": row["derived"]["stage_count"],
                    "x_w": row["derived"]["x_w"],
                    "l1a_z_m": pair["l1a_z_c_m"],
                    "p2_z_m": pair["p2_z_c_m"],
                    "shift_m": pair["shift_m"],
                    "shift_tol": pair["shift_over_tolerance"],
                    "wall_ratio": pair["wall_b_ratio_p2_over_l1a"],
                    "l1a_rho": pair["l1a_rho_conservative"],
                    "p2_rho": pair["p2_rho_conservative"],
                    "rho_ratio": pair["rho_conservative_ratio_p2_over_l1a"],
                    "l1a_angle": pair["l1a_angle_to_wall_normal_deg"],
                    "p2_angle": pair["p2_angle_to_wall_normal_deg"],
                    "ambiguous": pair["l1a_boundary_ambiguous"] or pair["p2_boundary_ambiguous"],
                    "z_over_L": pair["l1a_z_c_m"] / row["geometry"]["chamber_length_m"],
                }
            )
        for unmatched in comparison["unmatched_cusps"]:
            points.append({"id": row["design_id"], "stages": row["derived"]["stage_count"], "x_w": row["derived"]["x_w"], "unmatched_side": unmatched["side"], "z_m": unmatched["z_c_m"], "near_end": unmatched["near_straight_section_end"]})
    return points


def _overlay(results: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    record = _load_object(results / row["record_path"], row["record_path"])
    accepted = record["accepted"]
    descriptors = record["descriptors"]["accepted"]
    reference = record["l1a_reference"]
    if [c["z_c_m"] for c in accepted["topology"]["wall_cusps"]] != [c["z_c_m"] for c in row["p2_wall_cusps"]]:
        raise ValueError(f"{row['key']}: record cusps differ from the dataset row")
    if descriptors["profiles"] is None or reference["profiles"] is None:
        raise ValueError(f"{row['key']}: missing stored profiles")
    traces = []
    for trace in accepted["separatrix_traces"]:
        if trace["path_rz_m"] is None:
            raise ValueError(f"{row['key']}: trace without a sampled path")
        traces.append({"null_id": trace["null_id"], "reaches_wall": trace["reaches_wall"], "path": [[round(p[0], 10), round(p[1], 10)] for p in trace["path_rz_m"]]})
    geometry = record["geometry"]
    return {
        "id": row["design_id"],
        "stages": row["derived"]["stage_count"],
        "rw_m": geometry["wall_radius_m"],
        "L_m": geometry["chamber_length_m"],
        "pitch_m": geometry["stage_pitch_m"],
        "straight_m": [geometry["straight_z_min_m"], geometry["straight_z_max_m"]],
        "stage_centres_m": geometry["stage_centres_m"],
        "magnet_t_m": row["derived"]["magnet_axial_thickness_m"],
        "rm_in_m": row["derived"]["magnet_inner_radius_m"],
        "x_w": row["derived"]["x_w"],
        "verdict_row": {"strict": row["comparison"]["count_agreement_strict"], "bijection": row["comparison"]["cusp_match"]["bijection"], "max_shift_m": row["comparison"]["max_cusp_shift_m"], "tolerance_m": row["comparison"]["cusp_position_tolerance_m"], "p2_hemp": row["comparison"]["p2_hemp_like_all_cusps"]},
        "p2_profiles": {key: [round(v, 9) for v in descriptors["profiles"][key]] for key in ("z_m", "wall_abs_b_t", "wall_b_r_t", "axis_b_z_t")},
        "l1a_profiles": {key: [round(v, 9) for v in reference["profiles"][key]] for key in ("z_m", "wall_abs_b_t", "wall_b_r_t", "axis_b_z_t")},
        "p2_nulls": [{"z_m": n["z_m"], "zone": n["zone"]} for n in accepted["axis_nulls"]["nulls"]],
        "l1a_nulls": [{"z_m": n["z_m"], "zone": n["zone"]} for n in reference["axis_nulls"]],
        "p2_cusps": [{"id": c["cusp_id"], "z_m": c["z_c_m"], "b_t": c["wall_b_t"], "angle_deg": c["angle_to_wall_normal_deg"], "rho": r["rho_conservative"], "hemp": r["hemp_like_conservative"]} for c, r in zip(accepted["topology"]["wall_cusps"], descriptors["cusps"], strict=True)],
        "l1a_cusps": [{"id": c["cusp_id"], "z_m": c["z_c_m"], "b_t": c["wall_b_t"], "angle_deg": c["angle_to_wall_normal_deg"], "rho": r["rho_conservative"], "hemp": r["hemp_like_conservative"]} for c, r in zip(reference["wall_cusps"], reference["rho"], strict=True)],
        "p2_cells": [{"id": c["cell_id"], "kind": c["kind"], "z0": c["z_start_m"], "z1": c["z_end_m"]} for c in accepted["topology"]["cells"]],
        "traces": traces,
        "p2_levels": [{"level": level["level"], "dofs": level["p2_dofs"], "iterations": level["iterations"], "residual": level["relative_true_residual_l2"]} for level in row["p2"]["levels"]],
        "regions": [{"id": region["region_id"], "mu_r": region["relative_permeability"], "br_t": region["remanence_z_t"]} for region in row["p2"]["regions"] if region["relative_permeability"] != 1.0 or region["remanence_z_t"] != 0.0],
    }


def _select_overlays(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Representatives first (protocol order), then the largest cusp shift, then the largest wall-|B| ratio."""

    picked = [row for row in rows if row["representative"]]
    remaining = [row for row in rows if not row["representative"]]
    by_shift = sorted(remaining, key=lambda row: -(row["comparison"]["max_cusp_shift_m"] or 0.0))
    by_ratio = sorted(remaining, key=lambda row: -(row["comparison"]["peak_wall_b_ratio_p2_over_l1a"] or 0.0))
    for candidate in (*by_shift[:1], *by_ratio[:1]):
        if candidate not in picked and len(picked) < MAX_OVERLAYS:
            picked.append(candidate)
    return picked[:MAX_OVERLAYS]


def _distribution(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0}
    return {"count": len(clean), "min": min(clean), "median": statistics.median(clean), "max": max(clean)}


def _sliver_rows(results: Path, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Per design and level: minimum angle and elements below the qualification's 10 deg (from the records)."""

    output = {}
    for row in rows:
        record = _load_object(results / row["record_path"], row["record_path"])
        levels = record["evidence"]["p2"]["levels"]
        output[row["design_id"]] = [
            {"level": level["level"], "min_angle_deg": level["mesh_quality"]["minimum_angle_deg"], "below_10deg": level["mesh_quality"]["sliver"]["elements_below_threshold"], "elements": level["mesh_quality"]["sliver"]["element_count"], "regions": level["mesh_quality"]["sliver"]["regions_below_threshold"]}
            for level in levels
        ]
    return output


def predecessor_record(v1_results: Path = V1_RESULTS) -> dict[str, Any]:
    """The recorded v1 development rejection, byte-verified."""

    identity = verify_bundle(v1_results, expected_state="development_rejection")
    failures = _artifact(v1_results, "design-failures.json")["failed"]
    protocol = _artifact(v1_results, "protocol.json")
    terminal = _load_object(v1_results / "terminal.json", "terminal.json")
    records = sorted(path.stem for path in (v1_results / "artifacts" / "designs" / "hemp_like_v3").glob("*.json") if not path.name.endswith(".sha256.json"))
    if terminal["payload"]["failed_design_count"] != len(failures) or terminal["payload"]["resolved_design_count"] != len(records):
        raise ValueError("v1 terminal payload does not match its failures / records")
    return {
        **identity,
        "reject_below_angle_deg": protocol["p2"]["mesh"]["reject_below_angle_deg"],
        "failed_designs": [{"design_id": item["key"].split(":")[1], "stage": item["stage"], "reason": item["reason"]} for item in failures],
        "resolved_design_count": len(records),
        "stage_wall_s": terminal["payload"]["stage_wall_s"],
        "statement": "v1 executed once at its preregistration commit and ended in development_rejection: two level-0 meshes fell below the qualification's 10 deg angle gate before any solve (geometric slivers of the body-fitted mesher). No assessment, gates or verdict exist for v1; v1.1 re-preregistered with a 5 deg gate (disclosed) and a whole-set mesh preflight.",
    }


def build_payload(results: Path = RESULTS, experiment: Path = EXPERIMENT, v1_results: Path = V1_RESULTS) -> dict[str, Any]:
    identity = verify_bundle(results)
    predecessor = predecessor_record(v1_results)
    dataset = _artifact(results, "confirmation-dataset.json")
    campaign = _artifact(results, "campaign-result.json")
    gates = _artifact(results, "gates.json")
    protocol = _artifact(results, "protocol.json")
    if dataset["classification"] != CLASSIFICATION or campaign["classification"] != CLASSIFICATION:
        raise ValueError("bundle classification is not the confirmation label")
    if dataset["topology_label"] != TOPOLOGY_LABEL:
        raise ValueError("bundle topology label differs")
    if campaign["verdict"] not in VERDICTS or campaign["status"] != f"accepted_l1b_confirmation_{campaign['verdict'].lower()}" or campaign["evidentiary"] is not True:
        raise ValueError(f"campaign status/verdict is {campaign['status']!r} / {campaign['verdict']!r}")
    if not gates["passed"] or not all(gates["campaign"].values()):
        raise ValueError("gates.json does not record an all-true binding gate set")
    if campaign["headline"] != dataset["headline"] or gates["confirmation"]["verdict"] != campaign["verdict"]:
        raise ValueError("campaign headline / verdict differs from the dataset and gates")
    rows = dataset["designs"]
    designs = [_design_row(row) for row in rows]
    if len(designs) != dataset["design_count"] or dataset["design_count"] != campaign["design_count"] or len(campaign["agreement_table"]) != len(designs):
        raise ValueError("design count differs between dataset, campaign result and agreement table")
    confirmation = gates["confirmation"]
    b = confirmation["cusp_count_unchanged"]
    c = confirmation["cusp_position_shift"]
    d = confirmation["hemp_like_preserved"]
    if sum(item["boundary_tolerant"] for item in designs) != b["agreeing_designs_boundary_tolerant"] or sum(item["strict"] for item in designs) != b["agreeing_designs_strict"]:
        raise ValueError("cusp-count agreement counts do not reproduce from the rows")
    shifts = [point["shift_tol"] for point in _scatter(rows) if "shift_tol" in point]
    if len(shifts) != c["matched_cusp_count"] or (shifts and abs(max(shifts) - c["max_shift_over_tolerance"]) > 1e-12):
        raise ValueError("cusp shift statistics do not reproduce from the matched pairs")
    if all(item["bijection"] for item in designs) != c["all_designs_bijective"]:
        raise ValueError("bijection flag does not reproduce")
    if sum(item["p2_hemp"] and item["l1a_hemp"] for item in designs) != d["preserved_count"]:
        raise ValueError("HEMP-like preserved count does not reproduce")
    b_passed = b["fraction_boundary_tolerant"] >= b["pass_threshold"]
    c_passed = bool(c["all_designs_bijective"] and c["max_shift_over_tolerance"] is not None and c["max_shift_over_tolerance"] <= c["pass_threshold"])
    verdict = "CONFIRMED" if (b_passed and c_passed) else ("PARTIALLY_CONFIRMED" if (b_passed or c_passed) else "DISCONFIRMED")
    if b_passed != b["passed"] or c_passed != c["passed"] or verdict != campaign["verdict"]:
        raise ValueError("the verdict does not reproduce from the predeclared thresholds")
    if not all(item["converged"] and item["gates"] and item["sampling_stable"] for item in designs):
        raise ValueError("a design row is not converged / gated / stable in an accepted bundle")
    if protocol["predecessor"]["preregistration_commit"] != predecessor["preregistration_commit_sha"]:
        raise ValueError("the v1.1 protocol's predecessor commit differs from the recorded v1 bundle")
    slivers = _sliver_rows(results, rows)
    gate = float(protocol["p2"]["mesh"]["reject_below_angle_deg"])
    if any(level["min_angle_deg"] < gate for levels in slivers.values() for level in levels):
        raise ValueError("a recorded level falls below the declared angle gate")
    protocol_text = (experiment / "protocol.json").read_bytes().replace(b"\r\n", b"\n")
    payload = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "topology_label": TOPOLOGY_LABEL,
        "classification_statement": dataset["classification_statement"],
        "claim_boundary": dataset["claim_boundary"],
        "identity": {
            **identity,
            "protocol_semantic_sha256": dataset["protocol_semantic_sha256"],
            "experiment_code_sha256": dataset["experiment_code_sha256"],
            "dependency_source_sha256": dataset["dependency_source_sha256"],
            "field_pipeline_source_sha256": dataset["field_pipeline_source_sha256"],
            "protocol_file_sha256_lf": sha256(protocol_text).hexdigest(),
            "generator_sha256": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "template_sha256": sha256(TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "sealed_sources": dataset["sealed_sources"],
        },
        "purpose": protocol["purpose"],
        "predecessor": {**predecessor, "protocol_block": protocol["predecessor"]},
        "angle_gate": {"reject_below_angle_deg": gate, "disclosure": protocol["p2"]["mesh"]["angle_gate_disclosure"], "per_design_levels": slivers, "designs_with_elements_below_10deg": sorted(design_id for design_id, levels in slivers.items() if any(level["below_10deg"] > 0 for level in levels))},
        "verdict": campaign["verdict"],
        "verdict_rule": confirmation["verdict_rule"],
        "confirmation": confirmation,
        "gate_definitions": protocol["gates"],
        "p2": {key: protocol["p2"][key] for key in ("solver", "materials", "mesh", "adaptivity", "resources", "sampling")},
        "comparison_rule": protocol["comparison"],
        "definition_import": protocol["definition_v3_import"]["source"],
        "headline": dataset["headline"],
        "estimands": dataset["estimands"],
        "agreement_table": campaign["agreement_table"],
        "gates": {"campaign": gates["campaign"], "definitions": gates["definitions"]["binding_integrity"], "reported_not_binding": gates["definitions"]["reported_not_binding"], "replays": gates["replays"], "peak_rss_bytes": gates["peak_rss_bytes"], "ram_budget": gates["ram_budget"]},
        "execution": campaign["execution_mode"],
        "paper_admission": campaign["paper_admission"],
        "designs": designs,
        "scatter": _scatter(rows),
        "overlays": [_overlay(results, row) for row in _select_overlays(rows)],
        "distributions": {
            "shift_m": _distribution([point["shift_m"] for point in _scatter(rows) if "shift_m" in point]),
            "wall_ratio": _distribution([point["wall_ratio"] for point in _scatter(rows) if "wall_ratio" in point]),
            "rho_ratio": _distribution([point["rho_ratio"] for point in _scatter(rows) if "rho_ratio" in point]),
            "p2_seconds": _distribution([item["seconds"] for item in designs]),
            "level1_dofs": _distribution([item["dofs"][-1] for item in designs]),
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA or payload["classification"] != CLASSIFICATION:
        raise ValueError("payload schema/classification is invalid")
    if payload["claim_boundary"]["forbid_mirror_probability_publication"] is not True or payload["claim_boundary"]["forbid_plasma_performance_publication"] is not True:
        raise ValueError("claim boundary must forbid plasma and mirror-probability publication")
    if payload["verdict"] not in VERDICTS or payload["confirmation"]["verdict"] != payload["verdict"]:
        raise ValueError("verdict is invalid or inconsistent")
    if "NOT in scope" not in payload["paper_admission"]:
        raise ValueError("paper admission must be recorded as out of scope")
    if payload["predecessor"]["state"] != "development_rejection" or len(payload["predecessor"]["failed_designs"]) != 2:
        raise ValueError("the recorded v1 rejection must be carried with its two failures")
    ids = [item["id"] for item in payload["designs"]]
    if len(set(ids)) != len(ids) or len(ids) != payload["headline"]["design_count"]:
        raise ValueError("design rows are not unique or do not match the count")
    for item in payload["designs"]:
        if len(item["p2_rho"]) != item["p2_cusps"] or len(item["l1a_rho"]) != item["l1a_cusps"]:
            raise ValueError(f"{item['id']}: cusp/rho counts are inconsistent")
        if item["p2_hemp"] != (bool(item["p2_rho"]) and all(r >= 1.5 for r in item["p2_rho"])):
            raise ValueError(f"{item['id']}: P2 HEMP-like flag does not reproduce from rho")
        if not item["l1a_hemp"]:
            raise ValueError(f"{item['id']}: every design of the set is HEMP-like under L1a by construction")
        if item["strict"] != (item["p2_cusps"] == item["l1a_cusps"]):
            raise ValueError(f"{item['id']}: strict count agreement does not reproduce")
    matched = sum(1 for point in payload["scatter"] if "shift_m" in point)
    if matched != sum(item["matched"] for item in payload["designs"]):
        raise ValueError("scatter points do not match the matched cusp count")
    if not payload["overlays"] or len(payload["overlays"]) > MAX_OVERLAYS:
        raise ValueError("overlays must be present and bounded")
    text = json.dumps(payload)
    if "http://" in text or "https://" in text:
        raise ValueError("payload must not reference network resources")


def render_html(payload: Mapping[str, Any], template_path: Path = TEMPLATE_PATH) -> str:
    template = template_path.read_text(encoding="utf-8")
    if template.count("__PAYLOAD_JSON__") != 1:
        raise ValueError("template must contain exactly one payload slot")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = template.replace("__PAYLOAD_JSON__", encoded)
    if "__PAYLOAD_JSON__" in html:
        raise ValueError("payload slot was not replaced")
    data = html.encode("utf-8")
    if len(data) > MAX_HTML_BYTES:
        raise ValueError(f"dashboard exceeds {MAX_HTML_BYTES} bytes ({len(data)})")
    return html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--v1-results", type=Path, default=V1_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    payload = build_payload(arguments.results, arguments.experiment, arguments.v1_results)
    html = render_html(payload)
    arguments.output.write_bytes(html.encode("utf-8"))
    print(json.dumps({"output": str(arguments.output), "bytes": len(html.encode("utf-8")), "designs": payload["headline"]["design_count"], "verdict": payload["verdict"], "overlays": len(payload["overlays"])}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
