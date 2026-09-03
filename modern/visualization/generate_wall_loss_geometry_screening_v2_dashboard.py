"""Generate the offline wall-loss-vs-geometry screening dashboard (v2: catalogue cells).

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/orbit_wall_loss_geometry_screening_v2`` (or from its committed protocol
for verbatim strings). The generator verifies every file of the bundle against
``results/manifest.json`` (byte SHA-256 and sizes), cross-checks every cell's pooled counts
against the sealed per-case summaries, recomputes the headline, and refuses to render on any
inconsistency. It emits no wall-clock timestamps or machine paths, so identical inputs produce
identical bytes.

Classification carried everywhere: ``SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`` (sweep
rows) and ``P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN`` (the single P2 row).
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
EXPERIMENT = MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v2"
RESULTS = EXPERIMENT / "results"
TEMPLATE_PATH = HERE / "wall-loss-geometry-screening-v2.template.html"
DEFAULT_OUTPUT = HERE / "wall-loss-geometry-screening-v2.html"

SCHEMA = "cft-revival.wall-loss-geometry-screening-v2-dashboard/1.0.0"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
LABEL_P2 = "P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN"
MAX_HTML_BYTES = 2_500_000
DESIGN_VALUE_NAMES = (
    "stage_count_selector", "stage_pitch_m", "magnet_axial_fraction", "chamber_outer_radius_m",
    "dielectric_thickness_m", "radial_clearance_m", "magnet_radial_thickness_m", "source_strength_scale",
    "exit_length_fraction", "exit_expansion_descriptor", "first_polarity_selector",
)


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


def verify_bundle(results: Path) -> dict[str, Any]:
    """Byte-verify every manifest entry; return identity facts."""

    manifest = _load_object(results / "manifest.json", "manifest.json")
    if manifest.get("state") != "accepted_result":
        raise ValueError(f"bundle state is {manifest.get('state')!r}, not accepted_result")
    verified = 0
    for entry in manifest["artifacts"]:
        if entry.get("type") != "file":
            continue
        raw = (results / entry["path"]).read_bytes()
        if sha256(raw).hexdigest() != entry["byte_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {entry['path']}")
        if len(raw) != entry["bytes"]:
            raise ValueError(f"size mismatch for {entry['path']}")
        verified += 1
    terminal_raw = (results / "terminal.json").read_bytes()
    lock_raw = (results / "execution-lock.json").read_bytes()
    if sha256(terminal_raw).hexdigest() != manifest["terminal_byte_sha256"]:
        raise ValueError("terminal.json does not match the manifest")
    if sha256(lock_raw).hexdigest() != manifest["lock_byte_sha256"]:
        raise ValueError("execution-lock.json does not match the manifest")
    lock = _load_json_bytes(lock_raw, "execution-lock.json")
    terminal = _load_json_bytes(terminal_raw, "terminal.json")
    if terminal["state"] != "accepted_result":
        raise ValueError("terminal state is not accepted_result")
    return {
        "manifest_file_sha256": sha256((results / "manifest.json").read_bytes()).hexdigest(),
        "terminal_file_sha256": manifest["terminal_byte_sha256"],
        "lock_file_sha256": manifest["lock_byte_sha256"],
        "experiment_id": manifest["experiment_id"],
        "preregistration_commit_sha": lock["commit"],
        "execution_command": lock["command"],
        "artifact_count": manifest["artifact_count"],
        "verified_file_count": verified,
    }


def _artifact(results: Path, relative: str) -> dict[str, Any]:
    return _load_object(results / "artifacts" / relative, relative)


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _estimate(estimate: Mapping[str, Any]) -> dict[str, Any]:
    return {"p": estimate["probability"], "lo": estimate["lower"], "hi": estimate["upper"], "k": estimate["successes"], "n": estimate["trials"]}


def _cell_row(design: Mapping[str, Any], cell: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    # Cross-check the pooled counts against the sealed N-step case summaries of this cell.
    trials = wall = reflected = escaped = 0
    for key, case in design["cases"].items():
        if case["cell_id"] != cell["cell_id"] or case["timestep"] != "N":
            continue
        summary = summaries[key]["summary"]
        if summary["termination_counts"] != case["termination_counts"] or summary["trial_count"] != case["trial_count"]:
            raise ValueError(f"{key}: dataset case totals differ from the sealed summary")
        trials += summary["trial_count"]
        wall += summary["termination_counts"]["wall_hit"]
        reflected += summary["termination_counts"]["reflected"]
        escaped += summary["termination_counts"]["domain_escape"]
    final = cell["final"]
    if (trials, wall, reflected, escaped) != (final["trials"], final["wall_hit"], final["reflected"], final["domain_escape"]):
        raise ValueError(f"{design['design_key']} {cell['cell_id']}: pooled cell counts differ from the sealed summaries")
    control = cell["control"] or {}
    return {
        "design_key": design["design_key"],
        "short": design["design_key"].split("-")[3] if design["set_id"] == "sweep_v2" else "P2",
        "set_id": design["set_id"],
        "label": design["label"],
        "cell_id": cell["cell_id"],
        "index": cell["index"],
        "kind": cell["kind"],
        "position": cell["position_class"],
        "z_start_m": cell["z_start_m"],
        "z_end_m": cell["z_end_m"],
        "length_m": cell["length_m"],
        "launch_z_m": cell["launch_z_m"],
        "chamber_length_m": design["geometry"]["chamber_length_m"],
        "wall_area_m2": cell["wall_area_m2"],
        "length_over_pitch": cell["length_over_pitch"],
        "wall_mirror_ratio": cell["wall_mirror_ratio"],
        "axis_mirror_ratio": cell["axis_mirror_ratio"],
        "short_cell": cell["short_cell"],
        "injector_flag": cell["launch_plane_inside_injector_zone"],
        "boundary_ambiguous": cell["boundary_ambiguous"],
        "n1": cell["stage1"]["trials"],
        "k1": cell["stage1"]["wall_hit"],
        "width1": cell["stage1"]["wilson_width"],
        "topped": cell["topped_up"],
        "n": final["trials"],
        "wall": _estimate(final["p_wall"]),
        "refl": _estimate(final["p_reflected"]),
        "esc": _estimate(final["p_escape"]),
        "timeout": _estimate(final["p_timeout"]),
        "width": final["wilson_width"],
        "floor": final["binomial_floor"],
        "jfloor": final["jeffreys_floor"],
        "ready": final["surrogate_ready"],
        "control": {"n": control.get("n_control", 0), "wall_N": control.get("wall_N", 0), "wall_2N": control.get("wall_2N", 0), "discordant": control.get("discordant", 0), "delta": control.get("delta_p_wall")},
    }


def _design_row(design: Mapping[str, Any]) -> dict[str, Any]:
    v1 = design["v1_comparison"]
    return {
        "design_key": design["design_key"],
        "short": design["design_key"].split("-")[3] if design["set_id"] == "sweep_v2" else "P2",
        "set_id": design["set_id"],
        "label": design["label"],
        "representative": design["representative"],
        "sweep_index": design["sweep_index"],
        "design": None if design["design_values"] is None else {name: design["design_values"][name] for name in DESIGN_VALUE_NAMES},
        "geometry": {k: design["geometry"][k] for k in ("wall_radius_m", "chamber_length_m", "injector_length_m", "exit_start_m", "stage_count", "stage_pitch_m", "has_divergent_exit")},
        "field": {k: design["field"][k] for k in ("bore_max_b_t", "interpolation_b_relative_rms", "cross_resolution_b_relative_rms")},
        "cusps_z_m": design["catalogue"]["wall_cusps_z_m"],
        "cell_count": design["launch_design"]["cell_count"],
        "cells": [{"cell_id": c["cell_id"], "kind": c["kind"], "z0": c["z_start_m"], "z1": c["z_end_m"], "zl": c["launch_z_m"], "n": c["final"]["trials"], "wall": _estimate(c["final"]["p_wall"]), "refl": c["final"]["p_reflected"]["probability"], "esc": c["final"]["p_escape"]["probability"], "topped": c["topped_up"], "ready": c["final"]["surrogate_ready"]} for c in design["cells"]],
        "pooled": {w: {"p": design["pooled"][w]["probability"], "lo": design["pooled"][w]["lower"], "hi": design["pooled"][w]["upper"]} for w in ("wall_area", "launches")},
        "control": {k: design["control"][k] for k in ("n_control", "wall_N", "wall_2N", "delta_p_wall", "discordant", "quantum", "passed")},
        "timestep_passed": design["convergence_flags"]["timestep_passed"],
        "sealed": design["sealed"],
        "launches": {k: design["launch_design"][k] for k in ("stage1_launches", "stage2_launches", "control_launches", "final_launches")},
        "topped_cells": design["allocation"]["topped_up_cell_count"],
        "v1": None if v1 is None else {"p": v1["v1_probability"], "lo": v1["v1_interval"][0], "hi": v1["v1_interval"][1], "cells_z_m": v1["v1_cells_z_m"], "per_cell": v1["v1_per_cell_p_wall"], "diff_area": v1["comparison"]["wall_area"]["difference_v2_minus_v1"], "diff_launch": v1["comparison"]["launches"]["difference_v2_minus_v1"], "overlap_launch": v1["comparison"]["launches"]["intervals_overlap"]},
        "reflections": design["diagnostics"]["reflections_final_n"],
        "escapes": design["diagnostics"]["domain_escape_subclasses_final_n"],
        "mu_max": design["diagnostics"]["magnetic_moment_variation"]["max"],
        "gates": {"structural_passed": design["gates"]["structural_passed"], "timeout_free": design["gates"]["timeout_free"], "failed_checks": sorted(name for name, ok in design["gates"]["checks"].items() if not ok), "allocation_replay": design["allocation_replay"]["passed"]},
    }


def _axis_profile(results: Path, design_key: str) -> dict[str, Any]:
    """Axis B_z and wall |B| of the accepted bore grid.

    Sweep fields carry the solver's B components; the P2 adapter serialises psi only, so B is
    derived from psi by central differences there (B_z = (1/r) dpsi/dr on the axis limit,
    B_r = -(1/r) dpsi/dz) and the source is stated in the payload.
    """

    field = _artifact(results, f"fields/{design_key}.json")
    z = field["z_m"]
    r = field["r_m"]
    wall_index = len(r) - 1
    if "b_z_t" in field and "b_r_t" in field:
        bz_axis = field["b_z_t"][0]
        br_wall = field["b_r_t"][wall_index]
        bz_wall = field["b_z_t"][wall_index]
        source = "solver B components"
    else:
        psi = field["psi_wb"]
        dr = r[1] - r[0]
        # axis: B_z = lim (1/r) dpsi/dr = 2 psi(dr) / dr^2 for psi ~ r^2 near the axis
        bz_axis = [2.0 * psi[1][j] / (dr * dr) for j in range(len(z))]
        rw = r[wall_index]
        bz_wall = [(psi[wall_index][j] - psi[wall_index - 1][j]) / (dr * rw) for j in range(len(z))]
        br_wall = []
        for j in range(len(z)):
            j0, j1 = max(0, j - 1), min(len(z) - 1, j + 1)
            br_wall.append(-(psi[wall_index][j1] - psi[wall_index][j0]) / ((z[j1] - z[j0]) * rw))
        source = "central differences of the serialised psi (P2 adapter stores psi only)"
    return {"design_key": design_key, "z_m": z, "axis_bz_t": bz_axis, "wall_r_m": r[wall_index], "wall_b_t": [(br * br + bz * bz) ** 0.5 for br, bz in zip(br_wall, bz_wall)], "source": source}


def build_payload(results: Path = RESULTS, experiment: Path = EXPERIMENT, *, allow_non_evidentiary: bool = False) -> dict[str, Any]:
    identity = verify_bundle(results)
    dataset = _artifact(results, "geometry-wall-loss-dataset-v2.json")
    campaign = _artifact(results, "campaign-result.json")
    gates = _artifact(results, "gates.json")
    consumer = _artifact(results, "coupling-consumer-record.json")
    exclusions = _artifact(results, "design-exclusions.json")
    allocation = _artifact(results, "allocation-decisions.json")
    comparison = _artifact(results, "v1-comparison.json")
    protocol = _artifact(results, "protocol.json")
    plan = _artifact(results, "campaign-plan.json")
    block_count = int(plan["stage2_points_per_stratum"]) // int(plan["stage1_points_per_stratum"])
    if dataset["classification"] != CLASSIFICATION or campaign["classification"] != CLASSIFICATION:
        raise ValueError("bundle classification is not the screening label")
    allowed = {"accepted_screening_dataset"} | ({"shakedown_passed"} if allow_non_evidentiary else set())
    if campaign["status"] not in allowed:
        raise ValueError(f"campaign status is {campaign['status']!r}")
    if campaign["evidentiary"] is not (campaign["status"] == "accepted_screening_dataset"):
        raise ValueError("campaign evidentiary flag disagrees with its status")
    summaries: dict[str, Mapping[str, Any]] = {}
    for design in dataset["designs"]:
        for key in design["cases"]:
            summaries[key] = _artifact(results, f"summaries/{key}.json")
    cells = [_cell_row(design, cell, summaries) for design in dataset["designs"] for cell in design["cells"]]
    designs = [_design_row(design) for design in dataset["designs"]]
    if len(designs) != dataset["design_count"] or dataset["design_count"] != campaign["design_count"] or len(cells) != dataset["cell_count"]:
        raise ValueError("design/cell counts differ between dataset and campaign result")
    headline = dataset["headline"]
    sweep_cells = [cell for cell in cells if cell["set_id"] == "sweep_v2"]
    recomputed = {
        "cell_count": len(cells),
        "cells_topped_up": sum(cell["topped"] for cell in cells),
        "cells_saturated_after_stage1": sum(not cell["topped"] for cell in cells),
        "cells_surrogate_ready": sum(cell["ready"] for cell in cells),
        "sweep_cells_surrogate_ready": sum(cell["ready"] for cell in sweep_cells),
        "jeffreys_floor_median": median(cell["jfloor"] for cell in cells),
        "sealed_design_count": sum(design["sealed"] for design in designs),
        "control_flag_true_design_count": sum(design["timestep_passed"] for design in designs),
        "total_reflections_final_n": sum(design["reflections"] for design in designs),
        "stage2_launches": sum(design["launches"]["stage2_launches"] for design in designs),
    }
    for key, value in recomputed.items():
        if headline[key] != value:
            raise ValueError(f"headline {key} does not reproduce from the designs")
    pooled_n = sum(design["control"]["n_control"] for design in designs)
    if headline["control"]["n_control"] != pooled_n:
        raise ValueError("pooled control size does not reproduce")
    if allocation["summary"]["topped_up_cells"] != recomputed["cells_topped_up"] or allocation["summary"]["replay_all_passed"] is not True:
        raise ValueError("allocation decisions do not reproduce")
    representatives = [design["design_key"] for design in designs if design["representative"]]
    profiles = [_axis_profile(results, key) for key in representatives]
    protocol_text = (experiment / "protocol.json").read_bytes().replace(b"\r\n", b"\n")
    payload = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "label_p2": LABEL_P2,
        "classification_statement": dataset["classification_statement"],
        "claim_boundary": dataset["claim_boundary"],
        "identity": {
            **identity,
            "protocol_semantic_sha256": dataset["protocol_semantic_sha256"],
            "orbit_mc_source_sha256": dataset["orbit_mc_source_sha256"],
            "field_pipeline_source_sha256": dataset["field_pipeline_source_sha256"],
            "catalogue_file_sha256": dataset["catalogue_file_sha256"],
            "protocol_file_sha256_lf": sha256(protocol_text).hexdigest(),
            "generator_sha256": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "template_sha256": sha256(TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        },
        "field_source": dataset["field_source"],
        "catalogue": dataset["cusp_cell_catalogue"],
        "launch_design": dataset["launch_design"],
        "allocation_rule": dataset["allocation_rule"],
        "control_rule": dataset["control_rule"],
        "estimators": dataset["estimators"],
        "orbit_geometry_rule": {key: dataset["orbit_geometry_rule"][key] for key in ("wall_z_min_meaning", "wall_z_max", "domain_z_max", "divergent_exit_policy", "max_path_channel_lengths", "max_time_rule", "timestep_policies")},
        "known_defect": protocol["orbit_mc_contract"]["known_defect_v1_7"]["statement"],
        "headline": headline,
        "control_gate": dataset["control_gate"],
        "design_count": dataset["design_count"],
        "cell_count": dataset["cell_count"],
        "excluded_designs": exclusions["excluded"],
        "designs": designs,
        "cells": cells,
        "axis_profiles": profiles,
        "v1_comparison": {key: comparison[key] for key in ("design_count", "statement", "spearman_rank_correlation", "mean_difference_v2_minus_v1", "mean_absolute_difference", "interval_overlap_fraction")},
        "gates": {
            "passed": gates["passed"],
            "structural_all_passed": gates["structural_all_passed"],
            "allocation_replay_all_passed": gates["allocation_replay_all_passed"],
            "control_gate": gates["control_gate"],
            "manufactured": gates["manufactured"],
            "sealed_case_count": gates["sealed_case_count"],
            "case_count": gates["case_count"],
            "exact_authority_replay_count": gates["exact_authority_replay_count"],
            "validator_failures": gates["validator_failures"],
            "validators": campaign["validators"],
        },
        "execution": {
            "orbit_count": campaign["orbit_count"],
            "case_count": campaign["case_count"],
            "worker_pool_size": campaign["execution_mode"]["worker_pool_size"],
            "cases_wall_s": campaign["execution_mode"]["cases_wall_s"],
            "assessment_wall_s": campaign["execution_mode"]["assessment_wall_s"],
        },
        "consumer": {
            "consumer_id": consumer["consumer_id"],
            "v4_reference": consumer["v4_reference"]["reference_row"],
            "absence_statement": consumer["statement"],
            "per_cell_statement": consumer["per_cell_statement"],
            "cases_consumed": sum(item["consumption_status"] == "consumed_verified_handoff" for item in consumer["screening_cases_consumed"]),
            "cases_unsealed": sum(item["consumption_status"] != "consumed_verified_handoff" for item in consumer["screening_cases_consumed"]),
            "catalogue_consumed": consumer["catalogue_consumed"],
        },
        "plan_kind": campaign["plan_kind"],
        "block_count": block_count,
        "evidentiary": campaign["evidentiary"],
        "campaign_status": campaign["status"],
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA or payload["classification"] != CLASSIFICATION:
        raise ValueError("payload schema/classification is invalid")
    if payload["claim_boundary"]["not_p2_qualified"] is not True or payload["claim_boundary"]["p2_row_is_not_v4_replication"] is not True:
        raise ValueError("claim boundary must state not P2-qualified and not a v4 replication")
    keys = [item["design_key"] for item in payload["designs"]]
    if len(set(keys)) != len(keys) or len(keys) != payload["design_count"]:
        raise ValueError("design rows are not unique or do not match the count")
    if len(payload["cells"]) != payload["cell_count"]:
        raise ValueError("cell rows do not match the count")
    for cell in payload["cells"]:
        for key in ("wall", "refl", "esc", "timeout"):
            estimate = cell[key]
            if not 0.0 <= estimate["lo"] <= estimate["p"] <= estimate["hi"] <= 1.0:
                raise ValueError(f"{cell['design_key']} {cell['cell_id']} {key} interval is malformed")
        if cell["wall"]["k"] + cell["refl"]["k"] + cell["esc"]["k"] + cell["timeout"]["k"] != cell["n"]:
            raise ValueError(f"{cell['design_key']} {cell['cell_id']}: estimands do not partition the trials")
        blocks = int(payload["block_count"])
        if cell["n"] not in (cell["n1"], blocks * cell["n1"]) or cell["topped"] != (cell["n"] == blocks * cell["n1"]):
            raise ValueError(f"{cell['design_key']} {cell['cell_id']}: final n is not one or {blocks} blocks")
        if cell["position"] not in ("anode_side", "interior", "exit_side", "unbounded"):
            raise ValueError("unknown position class")
    for item in payload["designs"]:
        if item["set_id"] == "sweep_v2" and (item["label"] != CLASSIFICATION or item["v1"] is None):
            raise ValueError(f"{item['design_key']}: sweep row without label or v1 comparison")
        if item["set_id"] == "p2_divergent_exit" and (item["label"] != LABEL_P2 or item["v1"] is not None):
            raise ValueError("P2 row must carry its own label and no v1 comparison")
    text = json.dumps(payload)
    if "http://" in text or "https://" in text:
        raise ValueError("payload must not reference network resources")


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
    parser.add_argument("--allow-non-evidentiary", action="store_true", help="development only: render a NON-EVIDENTIARY shakedown bundle (status shakedown_passed)")
    arguments = parser.parse_args(argv)
    payload = build_payload(arguments.results, arguments.experiment, allow_non_evidentiary=arguments.allow_non_evidentiary)
    html = render_html(payload)
    arguments.output.write_bytes(html.encode("utf-8"))
    print(json.dumps({"output": str(arguments.output), "bytes": len(html.encode("utf-8")), "designs": payload["design_count"], "cells": payload["cell_count"], "headline": {k: payload["headline"][k] for k in ("cells_topped_up", "cells_saturated_after_stage1", "fraction_cells_surrogate_ready", "control")}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
