"""Promote completed guarded checkpoints after post-processing interruption."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from cft_revival.fem_reference import (
    artifact_from_bound_chain,
    checkpoint_metadata_summary,
    evaluate_phase_matched_domain_expansion,
    load_checkpoint_bundle,
    preflight_third_level,
    refresh_bound_artifact_authority,
    replay_artifact,
    viewer_contract,
    write_checkpoint_bundle,
    write_json,
)
from cft_revival.geometry import reference_variants

from run_reference_campaign import (
    DORFLER_THETA,
    LEVELS,
    MAXIMUM_ADJACENT_SIZE_GROWTH,
    MAXIMUM_P2_DOFS,
    MINIMUM_THIRD_LEVEL_FREE_RAM_BYTES,
    _finalize_checkpoint_chain,
    _observed_orders,
    _read_l1b_qois,
    _relative,
)


def _refresh_chain(output, paths):
    previous = "0" * 64
    provisional = []
    records = []
    for level, path in enumerate(paths):
        preflight_third_level(1)
        checkpoint, _verified = load_checkpoint_bundle(path)
        checkpoint["bound_artifact"] = refresh_bound_artifact_authority(
            checkpoint["bound_artifact"]
        )
        checkpoint["previous_checkpoint_file_sha256"] = previous
        checkpoint["chain_authority"] = {
            "status": "provisional_not_authoritative_until_finalized"
        }
        file_hash = write_checkpoint_bundle(path, checkpoint)
        summary = checkpoint_metadata_summary(path)
        provisional.append(
            {
                "level": checkpoint["level"],
                "file": str(path.relative_to(output)),
                "file_sha256": file_hash,
                "payload_sha256": summary["payload_sha256"],
                "mesh_sha256": checkpoint["mesh_sha256"],
                "parent_mesh_sha256": checkpoint["parent_mesh_sha256"],
                "previous_checkpoint_file_sha256": previous,
                **(
                    {
                        "padding_factor": checkpoint["domain_study"][
                            "padding_factor"
                        ]
                    }
                    if "domain_study" in checkpoint
                    else {}
                ),
            }
        )
        previous = file_hash
        records.append(checkpoint)
    return provisional, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    arguments = parser.parse_args()
    geometry = next(
        (
            item
            for item in reference_variants()
            if item.config_id.removesuffix("-v1") == arguments.design
        ),
        None,
    )
    if geometry is None:
        raise RuntimeError(f"unknown FEM design {arguments.design!r}")
    preflight = preflight_third_level(1)
    started = perf_counter()
    modern = Path(__file__).resolve().parents[2]
    output = (
        Path(__file__).resolve().parent
        / "artifacts"
        / "third-level"
        / arguments.design
    )
    checkpoint_root = output / "checkpoints"
    adaptive_paths = [
        checkpoint_root / f"{arguments.design}.level-{level}.json"
        for level in range(LEVELS)
    ]
    domain_paths = [
        checkpoint_root / f"{arguments.design}.domain-padding-{padding:.1f}.json"
        for padding in (0.5, 1.0, 1.5)
    ]
    if not all(path.is_file() for path in adaptive_paths + domain_paths):
        raise RuntimeError("completed adaptive and domain checkpoints are required")

    adaptive_provisional, adaptive_records = _refresh_chain(
        output, adaptive_paths
    )
    adaptive_leaf = adaptive_records[-1]["bound_artifact"]
    adaptive_anchors = _finalize_checkpoint_chain(
        output,
        adaptive_provisional,
        adaptive_leaf,
        require_third_level_ram=True,
    )
    domain_provisional, domain_records = _refresh_chain(output, domain_paths)
    domain_leaf = domain_records[-1]["bound_artifact"]
    domain_anchors = _finalize_checkpoint_chain(
        output,
        domain_provisional,
        adaptive_leaf,
        final_artifact=domain_leaf,
        chain_kind="domain",
        require_third_level_ram=True,
    )

    runs = [record["run"] for record in adaptive_records]
    qoi_keys = sorted(
        key for key in runs[-1]["qois_bz_t"] if key.endswith("-bore-average")
    )
    changes = [
        {
            key: _relative(left["qois_bz_t"][key], right["qois_bz_t"][key])
            for key in qoi_keys
        }
        for left, right in zip(runs, runs[1:])
    ]
    orders = _observed_orders(runs, qoi_keys)
    domain_inputs = []
    domain_runs = []
    for record in domain_records:
        run = record["run"]
        bound = record["bound_artifact"]
        domain_runs.append(run)
        domain_inputs.append(
            {
                "padding_factor": run["padding_factor"],
                "qois_bz_t": {
                    key: run["qois_bz_t"][key] for key in qoi_keys
                },
                "qoi_h_m": {
                    key: run["resolution"]["qoi_h_m"][key] for key in qoi_keys
                },
                "local_h_m": run["bound_local_h_m"],
                "domain": bound["problem"]["domain"],
            }
        )
    domain_evaluation = evaluate_phase_matched_domain_expansion(
        tuple(domain_inputs)
    )
    two_successive = len(changes) >= 2 and all(
        value < 0.01 for change in changes[-2:] for value in change.values()
    )
    stable_positive = all(
        value is not None and value > 0.0 for value in orders.values()
    )
    growth_gate = all(
        run["adjacent_area_size_growth"] <= 1.3 + 1.0e-12 for run in runs
    )
    convergence = {
        "adaptive_nested_levels": LEVELS,
        "completed_adaptive_levels": len(runs),
        "maximum_p2_dofs": MAXIMUM_P2_DOFS,
        "successive_volume_qoi_relative_changes": changes,
        "observed_orders_from_actual_qoi_h": orders,
        "two_successive_less_than_one_percent": two_successive,
        "stable_positive_order": stable_positive,
        "adjacent_size_growth_gate": growth_gate,
        "less_than_one_percent_reached": bool(
            two_successive
            and stable_positive
            and growth_gate
            and domain_evaluation["passed"]
        ),
        "phase_matched_domain_expansion_gate": bool(
            domain_evaluation["passed"]
        ),
        "domain_expansion": domain_evaluation,
        "acceptance_qois": qoi_keys,
        "diagnostic_only_qois": sorted(
            set(runs[-1]["qois_bz_t"]) - set(qoi_keys)
        ),
        "cell_interface_maxima_policy": "screening_only_not_used_for_acceptance",
        "energy_identity_policy": "diagnostic_only_not_an_acceptance_gate",
    }
    name = arguments.design
    l1b_path = (
        Path(__file__).resolve().parents[1]
        / "material_fields"
        / "artifacts"
        / f"{name}.material-field.json"
    )
    l1b_qois = _read_l1b_qois(l1b_path)
    l1b_comparison = {}
    for l1b_key, l1b_value in l1b_qois.items():
        fem_key = (
            l1b_key.replace("-axis", "-axis-patch")
            if l1b_key.endswith("-axis")
            else l1b_key
        )
        fem_value = runs[-1]["qois_bz_t"][fem_key]
        l1b_comparison[l1b_key] = {
            "fem_qoi_key": fem_key,
            "fem_reference_bz_t": fem_value,
            "l1b_structured_grid_bz_t": l1b_value,
            "relative_difference": _relative(fem_value, l1b_value),
            "identical_qoi_semantics": True,
            "fem_evaluation": (
                "weighted_quadratic_axis_patch_recovery"
                if l1b_key.endswith("-axis")
                else "piecewise_P2_axisymmetric_volume_integral"
            ),
            "l1b_evaluation": (
                "structured_grid_axis_interpolation"
                if l1b_key.endswith("-axis")
                else "structured_grid_axisymmetric_volume_quadrature"
            ),
        }
    comparisons = {
        "l1b_artifact": str(l1b_path.relative_to(modern)),
        "l1b_artifact_sha256": sha256(l1b_path.read_bytes()).hexdigest(),
        "l1b_fixed_and_volume_qois": l1b_comparison,
    }
    preflight_third_level(1)
    artifact = artifact_from_bound_chain(
        adaptive_leaf,
        level_evidence=adaptive_anchors,
        domain_studies=domain_anchors,
        evidence_base_path=str(output.relative_to(modern)),
        convergence=convergence,
        comparisons=comparisons,
    )
    artifact_path = output / f"{name}.fem-reference.json"
    viewer_path = output / f"{name}.fem-reference.viewer.json"
    artifact_hash = write_json(artifact_path, artifact)
    preflight_third_level(1)
    viewer_hash = write_json(viewer_path, viewer_contract(artifact))
    preflight_third_level(1)
    replay = replay_artifact(artifact)
    manifest_payload = {
        "schema_version": "cft_revival.fem_reference.campaign/1.1.0",
        "classification": (
            "independent_numerical_reference_not_hardware_validation"
        ),
        "artifact_authority": (
            "schema_v1.3_recomputed_acceptance_with_bound_checkpoint_chain"
        ),
        "adaptive_levels": LEVELS,
        "maximum_p2_dofs": MAXIMUM_P2_DOFS,
        "resource_policy_revision": {
            "accuracy_gates_relaxed": False,
            "minimum_third_level_free_ram_bytes": (
                MINIMUM_THIRD_LEVEL_FREE_RAM_BYTES
            ),
            "one_design_at_a_time": True,
            "explicit_third_level_opt_in": True,
            "preflight": preflight,
        },
        "dorfler_theta": DORFLER_THETA,
        "maximum_adjacent_size_growth": MAXIMUM_ADJACENT_SIZE_GROWTH,
        "domain_expansion_evidence": {
            "required": True,
            "phase_matched_fixed_local_h": True,
            "padding_factors": [0.5, 1.0, 1.5],
            "maximum_qoi_relative_change": 0.01,
            "status": "completed",
        },
        "designs": [
            {
                "config_id": geometry.config_id,
                "artifact": artifact_path.name,
                "viewer": viewer_path.name,
                "artifact_file_sha256": artifact_hash,
                "viewer_file_sha256": viewer_hash,
                "artifact_payload_sha256": artifact["integrity"][
                    "payload_sha256"
                ],
                "runs": runs,
                "checkpoints": adaptive_anchors,
                "domain_checkpoints": domain_anchors,
                "domain_runs": domain_runs,
                "convergence": convergence,
                "qualification_status": (
                    "NUMERICAL_P2_QUALIFIED"
                    if convergence["less_than_one_percent_reached"]
                    else "SCREENING_ONLY"
                ),
                "l1b_comparison": l1b_comparison,
                "classification": artifact["classification"],
            }
        ],
        "wall_seconds": perf_counter() - started,
        "less_than_one_percent_all_designs": convergence[
            "less_than_one_percent_reached"
        ],
        "diagnostic_policy": {
            "timing_and_memory": "DIAGNOSTIC_ONLY",
            "hardware_validation": False,
        },
        "limitations": [
            "Independent numerical reference, not hardware validation.",
            "Qualification is numerical P2 evidence only.",
        ],
    }
    manifest = {
        **manifest_payload,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "payload_sha256": sha256(
                json.dumps(
                    manifest_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        },
    }
    manifest_hash = write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "artifact_file_sha256": artifact_hash,
                "viewer_file_sha256": viewer_hash,
                "manifest_file_sha256": manifest_hash,
                "replay": replay,
                "convergence": convergence,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
