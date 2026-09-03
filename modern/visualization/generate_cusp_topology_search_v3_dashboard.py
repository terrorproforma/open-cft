"""Generate the offline cusp topology search v3 dashboard (v3.1 accepted bundle, v3 lineage).

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/cusp_topology_search_v3_1`` (the accepted campaign) or, for the
lineage panel, from the recorded ``assessment_rejection`` bundle of
``modern/experiments/cusp_topology_search_v3`` and the sealed v1/v2 datasets it compares
against. The generator byte-verifies every manifest entry, re-derives the headline from
the per-design rows and refuses to render on any inconsistency. It emits no wall-clock
timestamps or machine paths, so identical inputs produce identical bytes.

Labels carried everywhere: ``SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY`` (L1a sets) and
``P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY`` (the single P2 row). Nothing is a plasma,
mirror-probability, wall-loss or performance claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
EXPERIMENT = MODERN / "experiments" / "cusp_topology_search_v3_1"
RESULTS = EXPERIMENT / "results"
LINEAGE_EXPERIMENT = MODERN / "experiments" / "cusp_topology_search_v3"
LINEAGE_RESULTS = LINEAGE_EXPERIMENT / "results"
V1_DATASET = MODERN / "experiments" / "cft_topology_characterization_v1" / "results" / "dataset.json"
V2_DATASET = MODERN / "experiments" / "four_cell_topology_search_v2" / "results" / "dataset.json"
TEMPLATE_PATH = HERE / "cusp-topology-search-v3.template.html"
DEFAULT_OUTPUT = HERE / "cusp-topology-search-v3.html"

SCHEMA = "cft-revival.cusp-topology-search-v3-dashboard/1.0.0"
CLASSIFICATION = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
P2_CLASSIFICATION = "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
MAX_HTML_BYTES = 4_000_000
SET_ORDER = ("sweep_v2", "four_cell_v2", "characterization_v1", "p2_divergent_exit")
SET_LABELS = {
    "sweep_v2": "L1a geometry sweep v2 (96 accepted designs)",
    "four_cell_v2": "four-cell topology search v2 candidates (128)",
    "characterization_v1": "topology characterization v1 cases (56)",
    "p2_divergent_exit": "P2 divergent-exit-stack (1, P2-qualified, iron present)",
}


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


def verify_bundle(results: Path, *, expected_state: str) -> dict[str, Any]:
    """Byte-verify every manifest entry; return identity facts."""

    manifest = _load_object(results / "manifest.json", "manifest.json")
    if manifest.get("state") != expected_state:
        raise ValueError(f"bundle state is {manifest.get('state')!r}, not {expected_state}")
    hashes: dict[str, str] = {}
    verified = 0
    for entry in manifest["artifacts"]:
        if entry.get("type") != "file":
            continue
        path = results / entry["path"]
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest != entry["byte_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {entry['path']}")
        if len(raw) != entry["bytes"]:
            raise ValueError(f"size mismatch for {entry['path']}")
        hashes[entry["path"]] = digest
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
        "artifact_count": manifest["artifact_count"],
        "verified_file_count": verified,
        "artifact_hashes": hashes,
    }


def _artifact(results: Path, relative: str) -> dict[str, Any]:
    return _load_object(results / "artifacts" / relative, relative)


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _round(value: float | None, digits: int = 9) -> float | None:
    return None if value is None else round(float(value), digits)


def _design_row(row: Mapping[str, Any]) -> dict[str, Any]:
    geometry = row["geometry"]
    return {
        "set": row["set_id"],
        "id": row["design_id"],
        "label": row["label"],
        "rep": row["representative"],
        "stages": len(geometry["stage_centres_m"]),
        "pitch_m": geometry["stage_pitch_m"],
        "rw_m": geometry["wall_radius_m"],
        "L_m": geometry["chamber_length_m"],
        "straight_m": [geometry["straight_z_min_m"], geometry["straight_z_max_m"]],
        "axis_nulls_m": [null["z_m"] for null in row["axis_nulls"]],
        "axis_null_zones": [null["zone"] for null in row["axis_nulls"]],
        "channel_nulls": row["channel_axis_null_count"],
        "cusps": row["wall_cusp_count"],
        "cells": row["cell_count"],
        "z_c_m": [cusp["z_c_m"] for cusp in row["wall_cusps"]],
        "z_c_over_L": [cusp["z_c_over_length"] for cusp in row["wall_cusps"]],
        "cusp_b_t": [cusp["wall_b_t"] for cusp in row["wall_cusps"]],
        "cusp_angle_deg": [cusp["angle_to_wall_normal_deg"] for cusp in row["wall_cusps"]],
        "cusp_gap_distance_m": [cusp["distance_to_nearest_stage_gap_m"] for cusp in row["wall_cusps"]],
        "cusp_ambiguous": [cusp["boundary_ambiguous"] for cusp in row["wall_cusps"]],
        "outside": [{"z_m": item["z_c_m"], "zone": item["zone"]} for item in row["outside_intersections"]],
        "cell_kinds": [cell["kind"] for cell in row["cells"]],
        "cell_bounds_m": [[cell["z_start_m"], cell["z_end_m"]] for cell in row["cells"]],
        "wall_mirror": [cell["wall_mirror_ratio"] for cell in row["cells"]],
        "axis_mirror": [cell["axis_mirror_ratio"] for cell in row["cells"]],
        "wall_b_min_t": [cell["wall_b_min_t"] for cell in row["cells"]],
        "axis_bz_peak_t": [cell["axis_bz_peak_t"] for cell in row["cells"]],
        "four_cusps": row["four_wall_cusps"],
        "four_cells": row["four_cells"],
        "stable": row["stability"]["stable"],
        "max_shift_m": row["stability"]["max_wall_intersection_shift_m"],
        "held_out": row["held_out"]["passed"] if row["held_out"]["applies"] else None,
        "gates": all(row["gate_checks"].values()),
        "grid": row["grid"],
        "record_path": row["record_path"],
    }


def _representative_plot(results: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    record = _load_object(results / row["record_path"], row["record_path"])
    accepted = record["accepted"]
    if [cusp["z_c_m"] for cusp in accepted["topology"]["wall_cusps"]] != [cusp["z_c_m"] for cusp in row["wall_cusps"]]:
        raise ValueError(f"{row['key']}: record cusps differ from the dataset row")
    traces = []
    for trace in accepted["separatrix_traces"]:
        if trace["path_rz_m"] is None:
            raise ValueError(f"{row['key']}: representative trace without a sampled path")
        traces.append(
            {
                "null_id": trace["null_id"],
                "termination": trace["termination"],
                "path": [[round(point[0], 10), round(point[1], 10)] for point in trace["path_rz_m"]],
                "z_c_m": trace.get("z_c_m"),
                "reaches_wall": trace["reaches_wall"],
            }
        )
    geometry = record["geometry"]
    grid = accepted["grid"]
    # Plot window: the axis search window (channel extended by one pitch), clipped to the grid.
    z_range = [
        max(grid["z_min_m"], geometry["straight_z_min_m"] - geometry["stage_pitch_m"]),
        min(grid["z_max_m"], geometry["chamber_length_m"] + geometry["stage_pitch_m"]),
    ]
    return {
        "set": row["set_id"],
        "id": row["design_id"],
        "label": row["label"],
        "rw_m": geometry["wall_radius_m"],
        "L_m": geometry["chamber_length_m"],
        "straight_m": [geometry["straight_z_min_m"], geometry["straight_z_max_m"]],
        "stage_centres_m": geometry["stage_centres_m"],
        "pitch_m": geometry["stage_pitch_m"],
        "z_range_m": z_range,
        "grid_z_range_m": [grid["z_min_m"], grid["z_max_m"]],
        "nulls": [{"z_m": null["z_m"], "zone": null["zone"], "id": null["null_id"]} for null in accepted["axis_nulls"]["nulls"]],
        "cusps": [{"z_m": cusp["z_c_m"], "id": cusp["cusp_id"], "b_t": cusp["wall_b_t"], "angle_deg": cusp["angle_to_wall_normal_deg"], "ambiguous": cusp["boundary_ambiguous"]} for cusp in accepted["topology"]["wall_cusps"]],
        "outside": [{"z_m": item["z_c_m"], "zone": item["zone"]} for item in accepted["topology"]["outside_intersections"]],
        "cells": [{"id": cell["cell_id"], "kind": cell["kind"], "z0": cell["z_start_m"], "z1": cell["z_end_m"], "wall_mirror": cell["wall_mirror_ratio"], "axis_mirror": cell["axis_mirror_ratio"], "wall_b_min_t": cell["wall_b_min_t"], "axis_bz_peak_t": cell["axis_bz_peak_t"]} for cell in accepted["topology"]["cells"]],
        "traces": traces,
        "stability": {"stable": record["stability"]["stable"], "max_shift_m": record["stability"]["max_wall_intersection_shift_m"]},
        "grid_cells_across_bore": grid["radial_cells_across_bore"],
        "refined_grid": row["refined_grid"],
        "p2_consistency": record.get("p2_consistency"),
        "sweep_reference": {key: record["reference"].get(key) for key in ("sweep_axis_null_positions_m", "sweep_axis_bz_peak_positions_m")} if row["set_id"] == "sweep_v2" else None,
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
    return {"count": len(clean), "min": min(clean), "median": statistics.median(clean), "max": max(clean), "values": sorted(clean)}


def _lineage(v1_dataset_path: Path, v2_dataset_path: Path, lineage_results: Path) -> dict[str, Any]:
    v1 = _load_object(v1_dataset_path, "characterization v1 dataset")
    v2 = _load_object(v2_dataset_path, "four-cell v2 dataset")
    lineage_identity = verify_bundle(lineage_results, expected_state="assessment_rejection")
    lineage_gates = _artifact(lineage_results, "gates.json")
    lineage_campaign = _artifact(lineage_results, "campaign-result.json")
    failing = {name: designs for name, designs in lineage_gates["failing_designs"].items() if designs}
    return {
        "frozen_definition_results": {
            "characterization_v1": {
                "dataset_file_sha256": sha256(v1_dataset_path.read_bytes()).hexdigest(),
                "cases": v1["summary"]["evaluated_count"],
                "stable_eligible_cusp_count": v1["summary"]["stable_eligible_cusp_count"],
                "stable_eligible_cell_count": v1["summary"]["stable_eligible_cell_count"],
                "clustered_root_total": sum(case["maps"]["primary"]["clustered_root_count"] for case in v1["cases"]),
                "definition": "wall-side X-type vector null with cell-bounding constant-psi separatrix; axis nulls found (3-9 per case) but excluded as descriptors",
            },
            "four_cell_v2": {
                "dataset_file_sha256": sha256(v2_dataset_path.read_bytes()).hexdigest(),
                "candidates": v2["summary"]["evaluated_count"],
                "stable_count": v2["summary"]["stable_count"],
                "failure_counts": v2["summary"]["failure_counts"],
                "definition": "exactly four geometry-registered interior cusps (wall-side vector nulls at stage midplane slots) stable across three maps",
            },
        },
        "v3_recorded_rejection": {
            **{key: lineage_identity[key] for key in ("manifest_file_sha256", "experiment_id", "state", "preregistration_commit_sha", "artifact_count")},
            "status": lineage_campaign["status"],
            "campaign_gates": lineage_gates["campaign"],
            "failing_gates": failing,
            "failing_design_count": sum(len(designs) for designs in failing.values()),
            "stable_design_count": lineage_campaign["headline"]["stable_design_count"],
            "wall_cusp_count_histogram": lineage_campaign["headline"]["wall_cusp_count_histogram"],
            "root_cause": (
                "held-out reference kept sealed v1 axis clusters with centroid r_m == 0.0; 26 of 206 clusters carry a bilinear Newton member "
                "at r <= 1.6e-8 m and were dropped (22 inside the channel, in exactly the 14 failing designs); see cusp_topology_search_v3/POSTHOC_AUDIT.md"
            ),
        },
    }


def build_payload(
    results: Path = RESULTS,
    experiment: Path = EXPERIMENT,
    *,
    lineage_results: Path = LINEAGE_RESULTS,
    v1_dataset_path: Path = V1_DATASET,
    v2_dataset_path: Path = V2_DATASET,
) -> dict[str, Any]:
    identity = verify_bundle(results, expected_state="accepted_result")
    dataset = _artifact(results, "topology-dataset.json")
    campaign = _artifact(results, "campaign-result.json")
    gates = _artifact(results, "gates.json")
    catalogue = _artifact(results, "cusp-cell-catalogue.json")
    protocol = _artifact(results, "protocol.json")
    if dataset["classification"] != CLASSIFICATION or campaign["classification"] != CLASSIFICATION:
        raise ValueError("bundle classification is not the screening label")
    if dataset["p2_row_classification"] != P2_CLASSIFICATION:
        raise ValueError("bundle P2 label differs")
    if campaign["status"] != "accepted_topology_screening" or campaign["evidentiary"] is not True:
        raise ValueError(f"campaign status is {campaign['status']!r}")
    if not gates["passed"] or not all(gates["campaign"].values()):
        raise ValueError("gates.json does not record an all-true binding gate set")
    designs = [_design_row(row) for row in dataset["designs"]]
    if len(designs) != dataset["design_count"] or dataset["design_count"] != campaign["design_count"] or catalogue["design_count"] != dataset["design_count"]:
        raise ValueError("design count differs between dataset, campaign result and catalogue")
    if campaign["headline"] != dataset["headline"]:
        raise ValueError("campaign headline differs from the dataset headline")
    # Re-derive the headline from the rows.
    histogram = _histogram([item["cusps"] for item in designs])
    if histogram != dataset["headline"]["wall_cusp_count_histogram"]:
        raise ValueError("wall-cusp histogram does not reproduce from the rows")
    if sum(item["stable"] for item in designs) != dataset["headline"]["stable_design_count"]:
        raise ValueError("stable count does not reproduce from the rows")
    for set_id in SET_ORDER:
        rows = [item for item in designs if item["set"] == set_id]
        if not rows:
            raise ValueError(f"set {set_id} is empty")
        if sum(item["four_cusps"] for item in rows) / len(rows) != dataset["headline"]["four_wall_cusp_fraction_by_set"][set_id]:
            raise ValueError(f"{set_id}: four-wall-cusp fraction does not reproduce")
        if _histogram([item["cusps"] for item in rows]) != dataset["headline"]["wall_cusp_count_histogram_by_set"][set_id]:
            raise ValueError(f"{set_id}: per-set histogram does not reproduce")
    representatives = [_representative_plot(results, row) for row in dataset["designs"] if row["representative"]]
    by_set: dict[str, Any] = {}
    for set_id in SET_ORDER:
        rows = [item for item in designs if item["set"] == set_id]
        interior = [(m_w, m_a) for item in rows for m_w, m_a, kind in zip(item["wall_mirror"], item["axis_mirror"], item["cell_kinds"]) if kind == "interior"]
        by_set[set_id] = {
            "label": SET_LABELS[set_id],
            "count": len(rows),
            "stable": sum(item["stable"] for item in rows),
            "cusp_histogram": _histogram([item["cusps"] for item in rows]),
            "cell_histogram": _histogram([item["cells"] for item in rows]),
            "channel_null_histogram": _histogram([item["channel_nulls"] for item in rows]),
            "four_cusp_fraction": sum(item["four_cusps"] for item in rows) / len(rows),
            "four_cell_fraction": sum(item["four_cells"] for item in rows) / len(rows),
            "with_cusps": sum(item["cusps"] > 0 for item in rows),
            "z_c_over_L": _distribution([value for item in rows for value in item["z_c_over_L"]]),
            "gap_distance_over_pitch": _distribution([value / item["pitch_m"] for item in rows for value in item["cusp_gap_distance_m"]]),
            "angle_deg": _distribution([value for item in rows for value in item["cusp_angle_deg"]]),
            "interior_wall_mirror": _distribution([pair[0] for pair in interior]),
            "interior_axis_mirror": _distribution([pair[1] for pair in interior]),
            "max_shift_m": _distribution([item["max_shift_m"] for item in rows if item["max_shift_m"] is not None]),
            "held_out_passed": sum(1 for item in rows if item["held_out"] is True),
            "held_out_applies": sum(1 for item in rows if item["held_out"] is not None),
            "ambiguous_cusps": sum(sum(item["cusp_ambiguous"]) for item in rows),
            "outside_zones": _histogram_str([o["zone"] for item in rows for o in item["outside"]]),
        }
    protocol_text = (experiment / "protocol.json").read_bytes().replace(b"\r\n", b"\n")
    payload = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "p2_classification": P2_CLASSIFICATION,
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
        },
        "definition": {
            "literature_basis": protocol["definition_v3"]["literature_basis"],
            "axis_null": protocol["definition_v3"]["axis_null"]["statement"],
            "separatrix": protocol["definition_v3"]["separatrix"]["statement"],
            "cusp": protocol["definition_v3"]["wall_cusp_and_cell"]["cusp"],
            "cell": protocol["definition_v3"]["wall_cusp_and_cell"]["cell"],
            "mirror_descriptors": protocol["definition_v3"]["wall_cusp_and_cell"]["mirror_descriptors"],
            "stability": protocol["definition_v3"]["stability"]["statement"],
            "stability_tolerance_m": protocol["definition_v3"]["stability_tolerance_m"],
            "held_out_tolerance_m": protocol["definition_v3"]["held_out_tolerance_m"],
        },
        "design_sets": {set_id: protocol["design_sets"][set_id]["why"] for set_id in SET_ORDER},
        "headline": dataset["headline"],
        "held_out": dataset["held_out"],
        "p2_consistency": dataset["p2_consistency"],
        "gates": {"campaign": gates["campaign"], "definitions": gates["definitions"]["binding_integrity"], "reported_not_binding": gates["definitions"]["reported_not_binding"], "replays": gates["replays"]},
        "execution": campaign["execution_mode"],
        "by_set": by_set,
        "designs": designs,
        "representatives": representatives,
        "catalogue": {"schema_version": catalogue["schema_version"], "design_count": catalogue["design_count"], "stable_design_count": catalogue["stable_design_count"], "consumer_contract": catalogue["consumer_contract"]},
        "lineage": _lineage(v1_dataset_path, v2_dataset_path, lineage_results),
        "relation_to_v3": protocol["relation_to_v3"],
        "relation_to_prior_nulls": protocol["relation_to_prior_nulls"],
        "prior_campaign_disclosure": protocol["prior_campaign_disclosure"],
    }
    validate_payload(payload)
    return payload


def _histogram_str(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA or payload["classification"] != CLASSIFICATION:
        raise ValueError("payload schema/classification is invalid")
    if payload["claim_boundary"]["forbid_mirror_probability_publication"] is not True:
        raise ValueError("claim boundary must forbid mirror-probability publication")
    ids = [(item["set"], item["id"]) for item in payload["designs"]]
    if len(set(ids)) != len(ids) or len(ids) != payload["headline"]["design_count"]:
        raise ValueError("design rows are not unique or do not match the count")
    for item in payload["designs"]:
        if item["label"] != (P2_CLASSIFICATION if item["set"] == "p2_divergent_exit" else CLASSIFICATION):
            raise ValueError(f"{item['id']}: unexpected label")
        if len(item["z_c_m"]) != item["cusps"] or item["cells"] != (item["cusps"] + 1 if item["cusps"] else 1):
            raise ValueError(f"{item['id']}: cusp/cell counts are inconsistent")
        if item["z_c_m"] != sorted(item["z_c_m"]):
            raise ValueError(f"{item['id']}: cusps are not sorted")
    if len(payload["representatives"]) != sum(item["rep"] for item in payload["designs"]):
        raise ValueError("representative plots do not match the representative rows")
    text = json.dumps(payload)
    if "http://" in text or "https://" in text:
        # The literature locators are the only permitted URLs; they live in definition.literature_basis.
        stripped = dict(payload)
        stripped["definition"] = {key: value for key, value in payload["definition"].items() if key != "literature_basis"}
        if "http://" in json.dumps(stripped) or "https://" in json.dumps(stripped):
            raise ValueError("payload must not reference network resources outside the literature locators")


def render_html(payload: Mapping[str, Any], template_path: Path = TEMPLATE_PATH) -> str:
    template = template_path.read_text(encoding="utf-8")
    if template.count("__PAYLOAD_JSON__") != 1:
        raise ValueError("template must contain exactly one payload slot")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace("</", "<\\/")
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
    print(json.dumps({"output": str(arguments.output), "bytes": len(html.encode("utf-8")), "designs": payload["headline"]["design_count"], "histogram": payload["headline"]["wall_cusp_count_histogram"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
