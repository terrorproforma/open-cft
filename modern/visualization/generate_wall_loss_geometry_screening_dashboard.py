"""Generate the offline wall-loss-vs-geometry screening dashboard (v1).

Every number shown is read from the immutable, hash-bound results bundle of
``modern/experiments/orbit_wall_loss_geometry_screening_v1`` (or from its
committed protocol for verbatim strings). The generator verifies every file of
the bundle against ``results/manifest.json`` (byte SHA-256 and sizes), cross
checks the dataset against the per-case summaries, and refuses to render on any
inconsistency. It emits no wall-clock timestamps or machine paths, so identical
inputs produce identical bytes.

Classification carried everywhere: ``SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS``
(L1a screening fields, not P2-qualified; never accepted physical-orbit evidence).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
EXPERIMENT = MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1"
RESULTS = EXPERIMENT / "results"
TEMPLATE_PATH = HERE / "wall-loss-geometry-screening-v1.template.html"
DEFAULT_OUTPUT = HERE / "wall-loss-geometry-screening-v1.html"

SCHEMA = "cft-revival.wall-loss-geometry-screening-v1-dashboard/1.0.0"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
MAX_HTML_BYTES = 2_500_000
DESIGN_VALUE_NAMES = (
    "stage_count_selector", "stage_pitch_m", "magnet_axial_fraction", "chamber_outer_radius_m",
    "dielectric_thickness_m", "radial_clearance_m", "magnet_radial_thickness_m", "source_strength_scale",
    "exit_length_fraction", "exit_expansion_descriptor", "first_polarity_selector",
)
PARAMETER_AXES = (
    ("chamber_outer_radius_m", "wall radius r_w (m)", "design"),
    ("stage_pitch_m", "stage pitch (m)", "design"),
    ("chamber_length_m", "chamber length L (m)", "geometry"),
    ("source_strength_scale", "source strength scale", "design"),
    ("exit_length_fraction", "exit length fraction", "design"),
    ("magnet_axial_fraction", "magnet axial fraction", "design"),
    ("radial_clearance_m", "radial clearance (m)", "design"),
    ("bore_max_b_t", "bore |B| max (T)", "field"),
    ("centreline_mid_abs_bz_t", "axis |Bz| at mid-chamber (T)", "field"),
    ("minimum_mirror_ratio", "minimum mirror ratio", "field"),
    ("stage_gradient_rms_t_per_m", "stage gradient rms (T/m)", "field"),
    ("stage_count", "stage count", "geometry"),
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
        "artifact_hashes": hashes,
    }


def _artifact(results: Path, relative: str) -> dict[str, Any]:
    return _load_object(results / "artifacts" / relative, relative)


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _probability_row(estimate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "p": estimate["probability"],
        "lo": estimate["lower"],
        "hi": estimate["upper"],
        "k": estimate["successes"],
        "n": estimate["trials"],
    }


def _design_row(row: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reported = row["reported"]
    cases = row["cases"]
    fine_key = f"{row['case_id']}--accepted-2N"
    coarse_key = f"{row['case_id']}--accepted-N"
    fine = summaries[fine_key]
    coarse = summaries[coarse_key]
    # Cross-check the dataset row against the sealed per-case summaries.
    if fine["summary"]["wall_hit"] != reported["wall_hit"]:
        raise ValueError(f"{row['case_id']}: dataset wall_hit differs from the 2N summary")
    if coarse["summary"]["wall_hit"]["probability"] != row["convergence"]["probabilities"]["accepted-N"]:
        raise ValueError(f"{row['case_id']}: dataset N probability differs from the N summary")
    if fine["summary"]["termination_counts"] != cases["accepted-2N"]["termination_counts"]:
        raise ValueError(f"{row['case_id']}: termination counts differ from the 2N summary")
    cells_2n = row["per_cell"]["accepted-2N"]
    cells_n = row["per_cell"]["accepted-N"]
    cell_ids = sorted(cells_2n)
    strata = [
        {
            "cell": item["cell_id"],
            "E": item["kinetic_energy_ev"],
            "pitch": item["pitch_angle_deg"],
            "dir": item["parallel_direction"],
            "n": item["trials"],
            "wall": item["wall_hit"],
            "refl": item["reflected"],
            "esc": item["domain_escape"],
            "timeout": item["timeout"],
        }
        for item in row["per_stratum"]["accepted-2N"]
    ]
    if sum(item["wall"] for item in strata) != reported["wall_hit"]["successes"]:
        raise ValueError(f"{row['case_id']}: strata wall counts do not sum to the reported successes")
    geometry = row["geometry"]
    field = row["field"]
    qois = field["sweep_qois"]
    refined = None
    if "refined-N" in cases:
        refined = {
            "p": cases["refined-N"]["wall_hit"]["probability"],
            "lo": cases["refined-N"]["wall_hit"]["lower"],
            "hi": cases["refined-N"]["wall_hit"]["upper"],
            "change_vs_accepted_N": row["convergence"]["field_resolution_sensitivity"]["change"],
        }
    return {
        "case_id": row["case_id"],
        "short": row["case_id"].split("-")[3],
        "index": row["sweep_index"],
        "batch": row["batch"],
        "representative": row["representative"],
        "design": {name: row["design_values"][name] for name in DESIGN_VALUE_NAMES},
        "geometry": {
            "wall_radius_m": geometry["wall_radius_m"],
            "chamber_length_m": geometry["chamber_length_m"],
            "injector_length_m": geometry["injector_length_m"],
            "exit_start_m": geometry["exit_start_m"],
            "exit_length_m": geometry["exit_length_m"],
            "exit_outer_radius_m": geometry["exit_outer_radius_m"],
            "stage_count": geometry["stage_count"],
            "stage_pitch_m": geometry["stage_pitch_m"],
            "has_divergent_exit": geometry["has_divergent_exit"],
        },
        "field": {
            "bore_max_b_t": field["bore_max_b_t"],
            "centreline_mid_abs_bz_t": qois["centreline_mid_abs_bz_t"],
            "centreline_abs_bz_peak_t": qois["centreline_abs_bz_peak_t"],
            "minimum_mirror_ratio": qois["minimum_mirror_ratio"],
            "stage_gradient_rms_t_per_m": qois["stage_gradient_rms_t_per_m"],
            "field_energy_j": qois["field_energy_j"],
            "axis_cusp_count": len(qois["axis_cusp_positions_m"]),
            "axis_null_positions_m": qois["axis_null_positions_m"],
            "axis_cusp_positions_m": qois["axis_cusp_positions_m"],
            "interpolation_b_relative_rms": field["interpolation_b_relative_rms"],
            "cross_resolution_b_relative_rms": field["cross_resolution_b_relative_rms"],
        },
        "cells_z_m": [cell["axial_center_m"] for cell in row["launch_design"]["cells"]],
        "p": {
            "wall_2N": _probability_row(reported["wall_hit"]),
            "escape_2N": _probability_row(reported["domain_escape"]),
            "reflected_2N": _probability_row(reported["reflected"]),
            "timeout_2N": _probability_row(reported["timeout"]),
            "wall_N": _probability_row(coarse["summary"]["wall_hit"]),
        },
        "convergence": {
            "change": row["convergence"]["successive_change"],
            "overlap": row["convergence"]["adjacent_wilson_overlap"],
            "converged": row["convergence"]["converged"],
            "sealed": row["convergence"]["sealed"],
        },
        "refined_N": refined,
        "per_cell_2N": [cells_2n[cell]["wall_hit"]["probability"] for cell in cell_ids],
        "per_cell_N": [cells_n[cell]["wall_hit"]["probability"] for cell in cell_ids],
        "per_cell_reflected_2N": [cells_2n[cell]["reflected"]["probability"] for cell in cell_ids],
        "cell_ids": cell_ids,
        "escapes_2N": {
            key: reported["domain_escape_subclasses"].get(key, 0)
            for key in ("upstream_anode_plane", "exit_plane", "divergent_section_radial", "unclassified")
        },
        "reflections": {
            "N": row["diagnostics"]["reflection_counts"]["accepted-N"],
            "2N": row["diagnostics"]["reflection_counts"]["accepted-2N"],
            "all_cases": sum(row["diagnostics"]["reflection_counts"].values()),
        },
        "timeouts_2N": sum(cases["accepted-2N"]["timeout_counts"].values()),
        "mu": {
            "median": row["diagnostics"]["magnetic_moment_variation"]["median"],
            "max": row["diagnostics"]["magnetic_moment_variation"]["max"],
            "above_0p5": row["diagnostics"]["magnetic_moment_variation"]["count_above_0p5"],
        },
        "tolerance_close_share": row["diagnostics"]["tolerance_close_share"],
        "steps_median_2N": row["diagnostics"]["steps"]["median"],
        "per_orbit_ms_2N": cases["accepted-2N"]["per_orbit_ms"],
        "gates": {
            "structural_passed": row["gates"]["structural_passed"],
            "converged": row["gates"]["converged"],
            "sealed": row["gates"]["sealed"],
            "timeout_free": row["gates"]["timeout_free"],
            "passed": row["gates"]["passed"],
            "failed_checks": sorted(name for name, ok in row["gates"]["checks"].items() if not ok),
        },
        "strata_2N": strata,
    }


def _axis_profile(results: Path, case_id: str) -> dict[str, Any]:
    field = _artifact(results, f"fields/{case_id}.json")
    z = field["z_m"]
    bz_axis = field["b_z_t"][0]
    wall_index = len(field["r_m"]) - 1
    br_wall = field["b_r_t"][wall_index]
    bz_wall = field["b_z_t"][wall_index]
    return {
        "case_id": case_id,
        "z_m": z,
        "axis_bz_t": bz_axis,
        "wall_r_m": field["r_m"][wall_index],
        "wall_b_t": [(br * br + bz * bz) ** 0.5 for br, bz in zip(br_wall, bz_wall)],
    }


def build_payload(
    results: Path = RESULTS, experiment: Path = EXPERIMENT, *, allow_non_evidentiary: bool = False
) -> dict[str, Any]:
    identity = verify_bundle(results)
    dataset = _artifact(results, "geometry-wall-loss-dataset.json")
    campaign = _artifact(results, "campaign-result.json")
    gates = _artifact(results, "gates.json")
    consumer = _artifact(results, "coupling-consumer-record.json")
    exclusions = _artifact(results, "design-exclusions.json")
    protocol = _artifact(results, "protocol.json")
    if dataset["classification"] != CLASSIFICATION or campaign["classification"] != CLASSIFICATION:
        raise ValueError("bundle classification is not the screening label")
    allowed = {"accepted_screening_dataset"} | ({"shakedown_passed"} if allow_non_evidentiary else set())
    if campaign["status"] not in allowed:
        raise ValueError(f"campaign status is {campaign['status']!r}")
    if campaign["evidentiary"] is not (campaign["status"] == "accepted_screening_dataset"):
        raise ValueError("campaign evidentiary flag disagrees with its status")
    summaries: dict[str, Mapping[str, Any]] = {}
    for row in dataset["designs"]:
        for key in row["cases"]:
            case_key = f"{row['case_id']}--{key}"
            summaries[case_key] = _artifact(results, f"summaries/{case_key}.json")
    designs = [_design_row(row, summaries) for row in dataset["designs"]]
    if len(designs) != dataset["design_count"] or dataset["design_count"] != campaign["design_count"]:
        raise ValueError("design count differs between dataset and campaign result")
    wall = [item["p"]["wall_2N"]["p"] for item in designs]
    headline = dataset["headline"]
    recomputed = {
        "wall_hit_probability_min": min(wall),
        "wall_hit_probability_max": max(wall),
        "wall_hit_probability_median": median(wall),
        "converged_design_count": sum(item["convergence"]["converged"] for item in designs),
        "sealed_design_count": sum(item["convergence"]["sealed"] for item in designs),
        "timeout_free_design_count": sum(item["gates"]["timeout_free"] for item in designs),
        "total_reflections": sum(item["reflections"]["all_cases"] for item in designs),
    }
    for key, value in recomputed.items():
        if headline[key] != value:
            raise ValueError(f"headline {key} does not reproduce from the designs")
    representatives = [item["case_id"] for item in designs if item["representative"]]
    profiles = [_axis_profile(results, case_id) for case_id in representatives]
    protocol_text = (experiment / "protocol.json").read_bytes().replace(b"\r\n", b"\n")
    payload = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "classification_statement": dataset["classification_statement"],
        "claim_boundary": dataset["claim_boundary"],
        "identity": {
            **identity,
            "protocol_semantic_sha256": dataset["protocol_semantic_sha256"],
            "orbit_mc_source_sha256": dataset["orbit_mc_source_sha256"],
            "field_pipeline_source_sha256": dataset["field_pipeline_source_sha256"],
            "protocol_file_sha256_lf": sha256(protocol_text).hexdigest(),
            "generator_sha256": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "template_sha256": sha256(TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        },
        "field_source": dataset["field_source"],
        "launch_rule": dataset["launch_rule"],
        "orbit_geometry_rule": {
            key: dataset["orbit_geometry_rule"][key]
            for key in ("wall_z_min_meaning", "wall_z_max", "domain_z_max", "divergent_exit_policy", "max_path_channel_lengths", "max_time_rule", "timestep_policies")
        },
        "headline": {
            **headline,
            "total_reflections_2N": sum(item["reflections"]["2N"] for item in designs),
            "designs_with_reflections_2N": sum(item["reflections"]["2N"] > 0 for item in designs),
        },
        "reflection_note": "v4 (one P2-qualified design) recorded zero reflections; here total_reflections counts every case (N, 2N and refined-N are separate orbit sets of the same launches), total_reflections_2N counts the reported 2N case only.",
        "design_count": dataset["design_count"],
        "excluded_designs": exclusions["excluded"],
        "batches": {"primary": sum(item["batch"] == "primary" for item in designs), "extension": sum(item["batch"] == "extension" for item in designs)},
        "parameter_axes": [{"key": key, "label": label, "group": group} for key, label, group in PARAMETER_AXES],
        "designs": designs,
        "axis_profiles": profiles,
        "gates": {
            "passed": gates["passed"],
            "structural_all_passed": gates["structural_all_passed"],
            "manufactured": gates["manufactured"],
            "converged_design_count": gates["converged_design_count"],
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
            "v4_design_in_screening_set": consumer["v4_reference"]["design_in_screening_set"],
            "absence_statement": consumer["statement"],
            "screening_consumed": sum(item["consumption_status"] == "consumed_verified_handoff" for item in consumer["screening_designs_consumed"]),
            "screening_unsealed": sum(item["consumption_status"] != "consumed_verified_handoff" for item in consumer["screening_designs_consumed"]),
        },
        "prior_v4": protocol["prior_campaign_disclosure"]["v4"]["headline"],
        "plan_kind": campaign["plan_kind"],
        "evidentiary": campaign["evidentiary"],
        "campaign_status": campaign["status"],
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA or payload["classification"] != CLASSIFICATION:
        raise ValueError("payload schema/classification is invalid")
    if payload["claim_boundary"]["not_p2_qualified"] is not True:
        raise ValueError("claim boundary must state not P2-qualified")
    ids = [item["case_id"] for item in payload["designs"]]
    if len(set(ids)) != len(ids) or len(ids) != payload["design_count"]:
        raise ValueError("design rows are not unique or do not match the count")
    for item in payload["designs"]:
        for key in ("wall_2N", "escape_2N", "reflected_2N", "timeout_2N", "wall_N"):
            estimate = item["p"][key]
            if not 0.0 <= estimate["lo"] <= estimate["p"] <= estimate["hi"] <= 1.0:
                raise ValueError(f"{item['case_id']} {key} interval is malformed")
        if len(item["per_cell_2N"]) != 4 or len(item["strata_2N"]) != 32:
            raise ValueError(f"{item['case_id']} per-cell/strata shape is wrong")
        total = item["p"]["wall_2N"]["k"] + item["p"]["escape_2N"]["k"] + item["p"]["reflected_2N"]["k"] + item["p"]["timeout_2N"]["k"]
        if total != item["p"]["wall_2N"]["n"]:
            raise ValueError(f"{item['case_id']} estimands do not partition the trials")
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
    parser.add_argument(
        "--allow-non-evidentiary",
        action="store_true",
        help="development only: render a NON-EVIDENTIARY shakedown bundle (status shakedown_passed)",
    )
    arguments = parser.parse_args(argv)
    payload = build_payload(arguments.results, arguments.experiment, allow_non_evidentiary=arguments.allow_non_evidentiary)
    html = render_html(payload)
    arguments.output.write_bytes(html.encode("utf-8"))
    print(json.dumps({"output": str(arguments.output), "bytes": len(html.encode('utf-8')), "designs": payload["design_count"], "headline": payload["headline"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
