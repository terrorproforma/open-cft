"""Read-only post-hoc audit of the recorded v3 held_out_correspondence gate failure.

The v3 execution (preregistration 69159934, result 8cbcdbe6) ended ``assessment_rejection``
because the binding gate ``held_out_correspondence`` failed for 14 of the 56
characterization-v1 designs. This script re-derives, from the sealed v3 bundle and the sealed
v1 dataset only, (a) which sealed v1 axis roots the v3 reference extraction dropped and why,
and (b) the correspondence under the intended filter. It never writes into ``results/``.

Root cause: ``fields._resolve_characterization`` kept a sealed v1 root as an axis root only
when its clustered centroid ``r_m == 0.0``. v1 clusters an ``axis_sign_change`` detection
with any ``bilinear_vector_root`` detection within 0.75 mesh cells; when the Newton root of
the bilinear cell next to the axis converged at r ~ 3e-8 m the centroid became ~1.6e-8 m and
the cluster was excluded. The intended filter is "the cluster contains an axis-detected
member" (``axis_sign_change`` or ``axis_grid``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime.canonical import strict_json_file

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
RESULTS = EXPERIMENT / "results"
V1_DATASET = MODERN / "experiments" / "cft_topology_characterization_v1" / "results" / "dataset.json"
AXIS_METHODS = ("axis_sign_change", "axis_grid")
CHANNEL_ZONES = ("plasma_channel", "channel_axial_margin")
RECORDED_TERMINAL_STATE = "assessment_rejection"
RECORDED_FAILING_GATE = "held_out_correspondence"
RECORDED_FAILING_DESIGN_COUNT = 14


def _match_sorted(reference: Sequence[float], observed: Sequence[float], tolerance: float) -> dict[str, Any]:
    reference = sorted(reference)
    remaining = sorted(observed)
    pairs = []
    unmatched_reference = []
    for value in reference:
        if remaining:
            nearest = min(remaining, key=lambda item: abs(item - value))
            if abs(nearest - value) <= tolerance:
                pairs.append(abs(nearest - value))
                remaining.remove(nearest)
                continue
        unmatched_reference.append(value)
    return {
        "matched_count": len(pairs),
        "unmatched_reference_z_m": unmatched_reference,
        "unmatched_observed_z_m": remaining,
        "max_difference_m": max(pairs, default=None),
        "bijection": not unmatched_reference and not remaining,
    }


def sealed_axis_clusters(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sealed v1 primary-map clusters that contain an axis-detected member (intended filter)."""

    return [
        root
        for root in case["maps"]["primary"]["roots"]
        if not root["finite_box_boundary"] and any(member["method"] in AXIS_METHODS for member in root["members"])
    ]


def audit(results: Path = RESULTS, v1_dataset_path: Path = V1_DATASET) -> dict[str, Any]:
    manifest = strict_json_file(results / "manifest.json")
    terminal = strict_json_file(results / "terminal.json")
    gates = strict_json_file(results / "artifacts" / "gates.json")
    protocol = strict_json_file(results / "artifacts" / "protocol.json")
    dataset = strict_json_file(results / "artifacts" / "topology-dataset.json")
    tolerance = float(protocol["definition_v3"]["held_out_tolerance_m"])
    v1_dataset = strict_json_file(v1_dataset_path)
    recorded_failures = list(gates["failing_designs"][RECORDED_FAILING_GATE])
    rows = []
    dropped_total = 0
    nonzero_centroid = 0
    max_centroid_r = 0.0
    corrected_pass = 0
    for case in v1_dataset["cases"]:
        record = strict_json_file(results / "artifacts" / "designs" / "characterization_v1" / f"{case['case_id']}.json")
        clusters = sealed_axis_clusters(case)
        exact_zero = [root for root in clusters if root["r_m"] == 0.0]
        dropped = [root for root in clusters if root["r_m"] != 0.0]
        dropped_total += len(dropped)
        nonzero_centroid += len(dropped)
        max_centroid_r = max([max_centroid_r] + [abs(root["r_m"]) for root in dropped])
        reference = [root["z_m"] for root in clusters if root["geometry_association"]["zone"] in CHANNEL_ZONES]
        observed = [null["z_m"] for null in record["accepted"]["axis_nulls"]["nulls"] if null["zone"] == "channel"]
        corrected = _match_sorted(reference, observed, tolerance)
        classifications = all(
            root["local_topology"]["classification"] == "X" for root in clusters if root["geometry_association"]["zone"] in CHANNEL_ZONES
        )
        corrected_passed = bool(corrected["bijection"] and classifications)
        corrected_pass += corrected_passed
        key = f"characterization_v1:{case['case_id']}"
        dropped_in_channel = [root for root in dropped if root["geometry_association"]["zone"] in CHANNEL_ZONES]
        rows.append(
            {
                "case_id": case["case_id"],
                "stage_count": case["stage_count"],
                "recorded_gate_passed": record["held_out"]["passed"],
                "recorded_in_failing_list": key in recorded_failures,
                "sealed_axis_clusters": len(clusters),
                "clusters_with_exact_zero_centroid": len(exact_zero),
                "clusters_dropped_by_recorded_filter": [
                    {"root_id": root["root_id"], "z_m": root["z_m"], "centroid_r_m": root["r_m"], "zone": root["geometry_association"]["zone"], "methods": root["methods"], "member_count": root["member_count"]}
                    for root in dropped
                ],
                "clusters_dropped_in_channel": len(dropped_in_channel),
                "recorded_unmatched_observed_z_m": record["held_out"].get("unmatched_observed_z_m", []),
                "corrected_filter": corrected,
                "corrected_filter_passed": corrected_passed,
            }
        )
    return {
        "schema_version": "cft-revival.cusp-topology-search-v3.posthoc-held-out-audit/1.0.0",
        "read_only": True,
        "bundle": {
            "manifest_state": manifest["state"],
            "terminal_state": terminal["state"],
            "recorded_failing_gate": RECORDED_FAILING_GATE,
            "recorded_failing_designs": recorded_failures,
            "other_campaign_gates_all_true": all(value for name, value in gates["campaign"].items() if name != RECORDED_FAILING_GATE),
            "design_count": dataset["design_count"],
            "stable_design_count": dataset["headline"]["stable_design_count"],
        },
        "root_cause": (
            "fields._resolve_characterization selected sealed v1 axis roots with `r_m == 0.0`; v1 clusters an axis_sign_change "
            "detection with a bilinear Newton root of the neighbouring cell when both lie within 0.75 mesh cells, and the centroid "
            "of such a cluster is not exactly zero, so the cluster was dropped from the reference and the corresponding v3 null "
            "counted as unmatched"
        ),
        "held_out_tolerance_m": tolerance,
        "sealed_axis_clusters_total": sum(row["sealed_axis_clusters"] for row in rows),
        "clusters_dropped_by_recorded_filter_total": dropped_total,
        "max_dropped_centroid_r_m": max_centroid_r,
        "corrected_filter_pass_count": corrected_pass,
        "corrected_filter_max_difference_m": max((row["corrected_filter"]["max_difference_m"] or 0.0) for row in rows),
        "recorded_failures_explained_by_dropped_clusters": bool(
            all(
                (row["clusters_dropped_in_channel"] > 0) == (not row["recorded_gate_passed"])
                and row["recorded_in_failing_list"] == (not row["recorded_gate_passed"])
                and len(row["recorded_unmatched_observed_z_m"]) == row["clusters_dropped_in_channel"]
                for row in rows
            )
        ),
        "designs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the audit JSON here (must be outside results/)")
    arguments = parser.parse_args(argv)
    report = audit()
    if arguments.json is not None:
        if RESULTS in arguments.json.resolve().parents or arguments.json.resolve() == RESULTS:
            raise SystemExit("refusing to write inside the immutable results directory")
        arguments.json.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))
    summary = {key: value for key, value in report.items() if key != "designs"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
