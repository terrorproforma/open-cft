"""Generate the offline L1a geometry sweep v3 dashboard (HEMP-like wall-radius-to-pitch regime).

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/l1a_geometry_sweep_v3``. The generator byte-verifies every manifest
entry, re-derives the headline counts from the per-design rows and refuses to render on any
inconsistency. It emits no wall-clock timestamps or machine paths, so identical inputs
produce identical bytes.

Labels carried everywhere: ``L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID`` (the sweep) and
``SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY`` (the cusp catalogue). Nothing is a plasma,
mirror-probability, wall-loss or performance claim; rho is a field ratio of a linear-vacuum
screening field, and the L1b/P2 confirmation for r_w / L > 0.5 is queued, not run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
EXPERIMENT = MODERN / "experiments" / "l1a_geometry_sweep_v3"
RESULTS = EXPERIMENT / "results"
TEMPLATE_PATH = HERE / "l1a-geometry-sweep-v3.template.html"
DEFAULT_OUTPUT = HERE / "l1a-geometry-sweep-v3.html"

SCHEMA = "cft-revival.l1a-geometry-sweep-v3-dashboard/1.0.0"
CLASSIFICATION = "L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID"
TOPOLOGY_LABEL = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
MAX_HTML_BYTES = 4_000_000
MAX_CUSP_MAPS = 6
CURVE_POINTS = 120


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
# Bessel functions for the prediction curve (same series as the experiment)
# --------------------------------------------------------------------------- #
def _bessel_i(order: int, x: float) -> float:
    half = 0.5 * x
    term = half**order / math.factorial(order)
    total = term
    for m in range(1, 400):
        term = term * half * half / (m * (m + order))
        total += term
        if abs(term) <= 1e-17 * abs(total):
            break
    return total


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _design_row(row: Mapping[str, Any]) -> dict[str, Any]:
    rho = [item["rho_conservative"] for item in row["rho"]]
    return {
        "set": row["set_id"],
        "id": row["design_id"],
        "ordinal": row["ordinal"],
        "in_v2_box": row["inside_sweep_v2_box"],
        "stages": row["derived"]["stage_count"],
        "pitch_m": row["derived"]["represented_stage_pitch_m"],
        "rw_m": row["geometry"]["wall_radius_m"],
        "rm_in_m": row["derived"]["magnet_inner_radius_m"],
        "x_w": row["x_w"],
        "x_m": row["x_m_inner"],
        "rw_over_L": row["wall_radius_over_pitch"],
        "i1": row["ppm_prediction"]["i1_x_w"],
        "i1_i0": row["ppm_prediction"]["i1_over_i0_x_w"],
        "predicted": row["predicted_hemp_like_i1"],
        "nulls": row["axis_null_count"],
        "cusps": row["wall_cusp_count"],
        "z_c_m": [c["z_c_m"] for c in row["wall_cusps"]],
        "rho": rho,
        "rho_down": [item["rho_downstream"] for item in row["rho"]],
        "rho_up": [item["rho_upstream"] for item in row["rho"]],
        "rho_wall": [item["rho_wall"] for item in row["rho"]],
        "cusp_wall_max": [item["cusp_is_wall_maximum"] for item in row["rho"]],
        "min_rho": row["min_rho_conservative"],
        "hemp": row["hemp_like_all_cusps"],
        "five_four": row["five_stage_four_cusp_hemp_like"],
        "b3_b1": row["wall_harmonics"]["b3_over_b1"],
        "b5_b1": row["wall_harmonics"]["b5_over_b1"],
        "angle_max": max((c["angle_to_wall_normal_deg"] for c in row["wall_cusps"]), default=None),
        "stable": row["stability"]["stable"],
        "shift_m": row["stability"]["max_wall_intersection_shift_m"],
        "rho_sens": row["resolution_sensitivity"].get("max_relative_rho_difference"),
        "v2_gates": row["v2_gates"]["passed"],
        "boundary": row["qois"]["boundary_to_peak_ratio"],
        "confidence": row["qois"]["topology_confidence"],
        "gates": all(row["gate_checks"].values()),
        "held_out": row["held_out"]["passed"] if row["held_out"]["applies"] else None,
        "record_path": row["record_path"],
        "exit_m": row["derived"]["represented_exit_length_m"],
    }


def _cusp_map(results: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    record = _load_object(results / row["record_path"], row["record_path"])
    accepted = record["accepted"]
    descriptors = record["descriptors"]["accepted"]
    if [c["z_c_m"] for c in accepted["topology"]["wall_cusps"]] != [c["z_c_m"] for c in row["wall_cusps"]]:
        raise ValueError(f"{row['key']}: record cusps differ from the dataset row")
    profiles = descriptors["profiles"]
    if profiles is None:
        raise ValueError(f"{row['key']}: no stored profiles")
    traces = []
    for trace in accepted["separatrix_traces"]:
        if trace["path_rz_m"] is None:
            raise ValueError(f"{row['key']}: representative trace without a sampled path")
        traces.append({"null_id": trace["null_id"], "reaches_wall": trace["reaches_wall"], "path": [[round(p[0], 10), round(p[1], 10)] for p in trace["path_rz_m"]]})
    geometry = record["geometry"]
    return {
        "set": row["set_id"],
        "id": row["design_id"],
        "stages": row["derived"]["stage_count"],
        "rw_m": geometry["wall_radius_m"],
        "L_m": geometry["chamber_length_m"],
        "pitch_m": geometry["stage_pitch_m"],
        "straight_m": [geometry["straight_z_min_m"], geometry["straight_z_max_m"]],
        "stage_centres_m": geometry["stage_centres_m"],
        "rm_in_m": row["derived"]["magnet_inner_radius_m"],
        "rm_out_m": row["derived"]["magnet_outer_radius_m"],
        "magnet_t_m": row["derived"]["magnet_axial_thickness_m"],
        "x_w": row["x_w"],
        "i1": row["ppm_prediction"]["i1_x_w"],
        "hemp": row["hemp_like_all_cusps"],
        "profiles": {key: [round(v, 9) for v in profiles[key]] for key in ("z_m", "wall_abs_b_t", "wall_b_r_t", "axis_b_z_t")},
        "nulls": [{"z_m": n["z_m"], "zone": n["zone"]} for n in accepted["axis_nulls"]["nulls"]],
        "cusps": [
            {"id": c["cusp_id"], "z_m": c["z_c_m"], "b_t": c["wall_b_t"], "angle_deg": c["angle_to_wall_normal_deg"], "rho": r["rho_conservative"], "rho_down": r["rho_downstream"], "rho_up": r["rho_upstream"], "rho_wall": r["rho_wall"], "up_peak_t": r["upstream_axis_peak_t"], "down_peak_t": r["downstream_axis_peak_t"], "hemp": r["hemp_like_conservative"]}
            for c, r in zip(accepted["topology"]["wall_cusps"], descriptors["cusps"], strict=True)
        ],
        "cells": [{"id": c["cell_id"], "kind": c["kind"], "z0": c["z_start_m"], "z1": c["z_end_m"]} for c in accepted["topology"]["cells"]],
        "traces": traces,
        "harmonics": {key: descriptors["wall_harmonics"].get(key) for key in ("b3_over_b1", "b5_over_b1", "fit_rms_over_max")},
        "stability": {"stable": record["stability"]["stable"], "max_shift_m": record["stability"]["max_wall_intersection_shift_m"]},
        "rho_sens": record["descriptors"]["resolution_sensitivity"]["max_relative_rho_difference"],
    }


def _histogram(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: int(pair[0])))


def _distribution(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0}
    return {"count": len(clean), "min": min(clean), "median": statistics.median(clean), "max": max(clean)}


def _select_cusp_maps(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """HEMP-like representatives: five-stage four-cusp first, then by descending min rho; plus one v2-region design."""

    sobol = [row for row in rows if row["set_id"] == "sobol_v3"]
    hemp = [row for row in sobol if row["hemp_like_all_cusps"]]
    ordered = sorted(hemp, key=lambda row: (not row["five_stage_four_cusp_hemp_like"], -(row["min_rho_conservative"] or 0.0), row["design_id"]))
    picked = ordered[: MAX_CUSP_MAPS - 1]
    v2_region = sorted((row for row in rows if row["inside_sweep_v2_box"] and row["set_id"] == "sobol_v3"), key=lambda row: row["design_id"])
    if not v2_region:
        v2_region = sorted((row for row in rows if row["set_id"] == "sweep_v2" and row["representative"]), key=lambda row: row["design_id"])
    if v2_region:
        picked.append(v2_region[0])
    return picked


def build_payload(results: Path = RESULTS, experiment: Path = EXPERIMENT) -> dict[str, Any]:
    identity = verify_bundle(results)
    dataset = _artifact(results, "sweep-dataset.json")
    campaign = _artifact(results, "campaign-result.json")
    gates = _artifact(results, "gates.json")
    catalogue = _artifact(results, "cusp-cell-catalogue-v3.json")
    protocol = _artifact(results, "protocol.json")
    if dataset["classification"] != CLASSIFICATION or campaign["classification"] != CLASSIFICATION:
        raise ValueError("bundle classification is not the sweep label")
    if dataset["topology_label"] != TOPOLOGY_LABEL:
        raise ValueError("bundle topology label differs")
    if campaign["status"] != "accepted_l1a_sweep_v3" or campaign["evidentiary"] is not True:
        raise ValueError(f"campaign status is {campaign['status']!r}")
    if not gates["passed"] or not all(gates["campaign"].values()):
        raise ValueError("gates.json does not record an all-true binding gate set")
    if campaign["headline"] != dataset["headline"]:
        raise ValueError("campaign headline differs from the dataset headline")
    rows = dataset["designs"]
    designs = [_design_row(row) for row in rows]
    if len(designs) != dataset["design_count"] or dataset["design_count"] != campaign["design_count"] or catalogue["design_count"] != dataset["design_count"]:
        raise ValueError("design count differs between dataset, campaign result and catalogue")
    headline = dataset["headline"]
    sobol = [item for item in designs if item["set"] == "sobol_v3"]
    v2_region = [item for item in designs if item["set"] == "sweep_v2" or item["in_v2_box"]]
    if sum(item["hemp"] for item in sobol) != headline["sobol_hemp_like_count"]:
        raise ValueError("HEMP-like count does not reproduce from the rows")
    if sum(item["five_four"] for item in sobol) != headline["sobol_five_stage_four_cusp_hemp_like_count"]:
        raise ValueError("five-stage four-cusp HEMP-like count does not reproduce")
    if sum(item["predicted"] for item in sobol) != headline["sobol_predicted_hemp_like_i1_count"]:
        raise ValueError("predicted HEMP-like count does not reproduce")
    if sum(item["hemp"] for item in v2_region) != headline["sweep_v2_region_hemp_like_count"]:
        raise ValueError("v2-region HEMP-like count does not reproduce")
    if _histogram([item["cusps"] for item in sobol]) != headline["sobol_wall_cusp_count_histogram"]:
        raise ValueError("Sobol wall-cusp histogram does not reproduce")
    if sum(item["stable"] for item in designs) != headline["stable_design_count"]:
        raise ValueError("stable count does not reproduce")
    if catalogue["hemp_like_design_count"] != sum(entry["hemp_like_all_cusps"] for entry in catalogue["entries"]):
        raise ValueError("catalogue HEMP-like count does not reproduce")
    scatter = []
    for item in designs:
        for index, (rho, rho_down, rho_wall) in enumerate(zip(item["rho"], item["rho_down"], item["rho_wall"], strict=True)):
            scatter.append({"set": item["set"], "id": item["id"], "stages": item["stages"], "x_w": item["x_w"], "i1": item["i1"], "rho": rho, "rho_down": rho_down, "rho_wall": rho_wall, "in_v2_box": item["in_v2_box"], "cusp": index + 1, "of": item["cusps"], "end_cusp": index == 0 or index == item["cusps"] - 1})
    x_lo = min(item["x_w"] for item in designs)
    x_hi = max(item["x_w"] for item in designs)
    curve = []
    for k in range(CURVE_POINTS + 1):
        x = x_lo + (x_hi - x_lo) * k / CURVE_POINTS
        i0 = _bessel_i(0, x)
        i1 = _bessel_i(1, x)
        curve.append({"x": x, "i1": i1, "i1_i0": i1 / i0})
    v2_box = protocol["sampling"]["sweep_v2_box"]
    v2_x_range = [math.pi * v2_box["chamber_outer_radius_m"][0] / v2_box["stage_pitch_m"][1], math.pi * v2_box["chamber_outer_radius_m"][1] / v2_box["stage_pitch_m"][0]]
    by_stage = {}
    for stages in (3, 4, 5):
        items = [item for item in sobol if item["stages"] == stages]
        by_stage[str(stages)] = {
            "count": len(items),
            "hemp": sum(item["hemp"] for item in items),
            "predicted": sum(item["predicted"] for item in items),
            "cusp_histogram": _histogram([item["cusps"] for item in items]),
            "n_minus_1": sum(item["cusps"] == stages - 1 for item in items),
            "four_cusps": sum(item["cusps"] == 4 for item in items),
            "x_w": _distribution([item["x_w"] for item in items]),
        }
    test = dataset["estimands"]["sobol_v3"]["hypothesis_test"]
    hypothesis = protocol["descriptors_v3"]["hypothesis"]
    h1 = {"slope_in_range": 0.80 <= test["slope_through_origin"] <= 1.00, "band_fraction_ok": test["fraction_within_band"] >= 0.80}
    h2 = {"accuracy_ok": (test["prediction_accuracy"] or 0.0) >= 0.85, "no_hemp_like_in_v2_box": headline["sweep_v2_region_hemp_like_count"] == 0}
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
        "hypothesis": hypothesis,
        "hypothesis_outcome": {"h1": h1, "h2": h2, "test": test, "note": "reported tests, not gates: the thresholds were preregistered in protocol.json#descriptors_v3.hypothesis and the numbers are what the campaign measured"},
        "koch_rho": protocol["descriptors_v3"]["koch_rho"],
        "ppm_prediction": protocol["descriptors_v3"]["ppm_prediction"],
        "hemp_like_rule": protocol["descriptors_v3"]["hemp_like_rule"],
        "sampling": {"algorithm": protocol["sampling"]["algorithm"], "design_count": protocol["sampling"]["design_count"], "seed": protocol["sampling"]["seed"], "variables": protocol["sampling"]["variables"], "sweep_v2_box": v2_box, "regime_coverage": protocol["sampling"]["regime_coverage"], "bounds_provenance": protocol["sampling"]["bounds_provenance"]},
        "manufacturability": protocol["geometry"]["manufacturability_limits_assumed"],
        "length_policy": protocol["geometry"]["length_binary64_policy"],
        "headline": headline,
        "estimands": {set_id: dataset["estimands"][set_id] for set_id in ("sobol_v3", "sweep_v2", "pooled_all", "sweep_v2_region_pooled")},
        "held_out": dataset["held_out"],
        "gates": {"campaign": gates["campaign"], "definitions": gates["definitions"]["binding_integrity"], "reported_not_binding": gates["definitions"]["reported_not_binding"], "sweep_v2_gate_breakdown": gates["sweep_v2_gate_breakdown"], "not_applicable": gates["sweep_v2_gate_not_applicable"], "replays": gates["replays"]},
        "execution": campaign["execution_mode"],
        "l1b_p2_queue": campaign["l1b_p2_confirmation_queue"],
        "by_stage": by_stage,
        "scatter": scatter,
        "curve": curve,
        "v2_x_range": v2_x_range,
        "x_star": test["x_star_prediction"],
        "designs": designs,
        "cusp_maps": [_cusp_map(results, row) for row in _select_cusp_maps(rows)],
        "catalogue": {"schema_version": catalogue["schema_version"], "design_count": catalogue["design_count"], "stable_design_count": catalogue["stable_design_count"], "hemp_like_design_count": catalogue["hemp_like_design_count"], "consumer_contract": catalogue["consumer_contract"], "path": "modern/experiments/l1a_geometry_sweep_v3/results/artifacts/cusp-cell-catalogue-v3.json"},
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA or payload["classification"] != CLASSIFICATION:
        raise ValueError("payload schema/classification is invalid")
    if payload["claim_boundary"]["forbid_mirror_probability_publication"] is not True or payload["claim_boundary"]["forbid_plasma_performance_publication"] is not True:
        raise ValueError("claim boundary must forbid plasma and mirror-probability publication")
    if payload["l1b_p2_queue"]["status"] != "queued_not_run":
        raise ValueError("the L1b/P2 confirmation must be recorded as queued, not run")
    ids = [(item["set"], item["id"]) for item in payload["designs"]]
    if len(set(ids)) != len(ids) or len(ids) != payload["headline"]["design_count"]:
        raise ValueError("design rows are not unique or do not match the count")
    for item in payload["designs"]:
        if len(item["z_c_m"]) != item["cusps"] or len(item["rho"]) != item["cusps"]:
            raise ValueError(f"{item['id']}: cusp/rho counts are inconsistent")
        if item["hemp"] != (bool(item["rho"]) and all(r >= 1.5 for r in item["rho"])):
            raise ValueError(f"{item['id']}: HEMP-like flag does not reproduce from rho")
    if len(payload["scatter"]) != sum(item["cusps"] for item in payload["designs"]):
        raise ValueError("scatter points do not match the cusp count")
    if not payload["cusp_maps"] or len(payload["cusp_maps"]) > MAX_CUSP_MAPS:
        raise ValueError("cusp maps must be present and bounded")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    payload = build_payload(arguments.results, arguments.experiment)
    html = render_html(payload)
    arguments.output.write_bytes(html.encode("utf-8"))
    headline = payload["headline"]
    print(json.dumps({"output": str(arguments.output), "bytes": len(html.encode("utf-8")), "designs": headline["design_count"], "hemp_like": headline["sobol_hemp_like_count"], "cusp_maps": len(payload["cusp_maps"])}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
