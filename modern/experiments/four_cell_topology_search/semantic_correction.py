"""Metadata-only correction for the preserved v1 numerical search outputs."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment import (
    CLASSIFICATION,
    MANIFEST_VERSION,
    PROTOCOL_STATUS,
    SCHEMA_VERSION,
    STATE_DIMENSION,
    _report,
    _write_bytes,
    load_sealed_json,
    stable_hash,
    validate_bundle,
    write_sealed_json,
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _correct_plasma_outcome(
    original: dict[str, Any],
    *,
    case_id: str,
    audit_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    outcome = deepcopy(original)
    operating_point_id = outcome["operating_point"]["operating_point_id"]
    prior_valid = bool(outcome.pop("valid_state"))
    prior_performance = outcome.pop("screening_performance")
    corrected_attempts = []
    for attempt in outcome["attempts"]:
        corrected = deepcopy(attempt)
        state = corrected.pop("state")
        prior_state_publication = corrected.pop("valid_state_published")
        conservation = corrected.pop("conservation")
        root = bool(prior_state_publication and corrected["diagnostics"]["converged"])
        powers = None if conservation is None else conservation.get("powers")
        closures = None if conservation is None else conservation.get("closures")
        corrected.update(
            {
                "residual_root_found": root,
                "outcome_classification": (
                    "non_identifiable_screening_equation_residual_root"
                    if root
                    else "no_residual_root"
                ),
                "conservation_diagnostics": (
                    None if closures is None else {"closures": closures}
                ),
            }
        )
        corrected_attempts.append(corrected)
        if audit_records is not None and (
            state is not None or powers is not None or prior_state_publication
        ):
            audit_records.append(
                {
                    "case_id": case_id,
                    "operating_point_id": operating_point_id,
                    "start_index": corrected["start_index"],
                    "claim_status": (
                        "raw_unpublished_development_data_invalid_for_"
                        "physical_or_performance_claims"
                    ),
                    "prior_valid_state_published": prior_state_publication,
                    "raw_state": state,
                    "raw_power_diagnostics": powers,
                    "numeric_values_modified": False,
                }
            )
    outcome["attempts"] = corrected_attempts
    selected = corrected_attempts[outcome["selected_start_index"]]
    rank = selected["diagnostics"]["jacobian_rank"]
    root = bool(prior_valid and selected["residual_root_found"])
    outcome.update(
        {
            "residual_root_found": root,
            "outcome_classification": (
                "non_identifiable_screening_equation_residual_root"
                if root
                else "no_residual_root"
            ),
            "identifiable_state": False,
            "identifiability": {
                "status": "non_identifiable",
                "jacobian_rank": rank,
                "state_dimension": STATE_DIMENSION,
                "full_column_rank": rank == STATE_DIMENSION,
                "publication_allowed": False,
                "reason": (
                    "v1 residual roots are screening-equation diagnostics only; "
                    "the observed root is rank 22/25 and the protocol is invalid "
                    "for identifiable-state claims"
                ),
            },
        }
    )
    if audit_records is not None and prior_performance is not None:
        audit_records.append(
            {
                "case_id": case_id,
                "operating_point_id": operating_point_id,
                "start_index": outcome["selected_start_index"],
                "claim_status": (
                    "withdrawn_prior_semantic_object_preserved_for_audit_only"
                ),
                "prior_screening_performance_object": prior_performance,
                "numeric_values_modified": False,
            }
        )
    return outcome


def _correct_case(
    original: dict[str, Any],
    *,
    audit_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    case = deepcopy(original)
    case["plasma"] = [
        _correct_plasma_outcome(
            outcome,
            case_id=case["case_id"],
            audit_records=audit_records,
        )
        for outcome in case.get("plasma", [])
    ]
    return case


def correct_existing_bundle(output: Path) -> dict[str, Any]:
    """Rewrite only semantic metadata around unchanged v1 numerical records."""

    output = output.resolve()
    dataset_path = output / "dataset.json"
    manifest_path = output / "manifest.json"
    report_path = output / "report.md"
    old_dataset_file_hash = _file_hash(dataset_path)
    old_manifest_file_hash = _file_hash(manifest_path)
    old_report_file_hash = _file_hash(report_path)
    old_dataset = load_sealed_json(dataset_path)
    old_manifest = load_sealed_json(manifest_path)
    if old_dataset["schema_version"] == SCHEMA_VERSION:
        return validate_bundle(output)

    audit_records: list[dict[str, Any]] = []
    corrected_cases = [
        _correct_case(case, audit_records=audit_records)
        for case in old_dataset["cases"]
    ]
    corrected_ranking = [
        _correct_case(case, audit_records=None)
        for case in old_dataset["ranking"]
    ]
    compatible = [
        case
        for case in corrected_cases
        if case.get("topology", {}).get("compatible")
    ]
    root_candidates = [
        case
        for case in compatible
        if any(outcome["residual_root_found"] for outcome in case["plasma"])
    ]
    root_count = sum(
        outcome["residual_root_found"]
        for case in compatible
        for outcome in case["plasma"]
    )
    prior_summary = {
        key: old_dataset["summary"][key]
        for key in (
            "plasma_converged_candidate_count",
            "plasma_converged_state_count",
            "performance_publication_count",
        )
    }
    summary = deepcopy(old_dataset["summary"])
    summary.pop("plasma_converged_candidate_count")
    summary.pop("plasma_converged_state_count")
    summary.update(
        {
            "plasma_residual_root_candidate_count": len(root_candidates),
            "plasma_residual_root_count": root_count,
            "identifiable_state_count": 0,
            "performance_publication_count": 0,
        }
    )
    corrected = deepcopy(old_dataset)
    corrected.update(
        {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "protocol_status": PROTOCOL_STATUS,
            "model_chain": [
                "accepted geometry v1.1",
                "L1a current-equivalent Warp field",
                "coupling v2 deprecated same-z mirror proxy",
                "four-cell global plasma screening equations",
            ],
            "cases": corrected_cases,
            "ranking": corrected_ranking,
            "summary": summary,
        }
    )
    corrected["declared_gates"] = corrected.pop("predeclared_gates")
    corrected["plasma_policy"].update(
        {
            "publication": "none",
            "outcome_classification": (
                "rank-deficient residual roots are non-identifiable "
                "screening-equation diagnostics only"
            ),
        }
    )
    corrected["limitations"] = [
        *corrected["limitations"],
        "Version 1 was not preregistered.",
        "Coupling v2 used a deprecated same-z mirror proxy and roundoff null lows.",
        "Residual roots are rank-deficient and are not identifiable plasma states.",
    ]
    correction = {
        "kind": "semantic_publication_metadata_correction",
        "numerical_simulations_rerun": False,
        "numerical_values_modified": False,
        "selection_or_ranking_modified": False,
        "representative_artifacts_modified": False,
        "prior_dataset_file_sha256": old_dataset_file_hash,
        "prior_dataset_payload_sha256": old_dataset["integrity"]["payload_sha256"],
        "prior_manifest_file_sha256": old_manifest_file_hash,
        "prior_manifest_payload_sha256": old_manifest["integrity"]["payload_sha256"],
        "prior_report_file_sha256": old_report_file_hash,
        "selection_identity_sha256": stable_hash(
            [
                {"case_id": case["case_id"], "rank": case["rank"]}
                for case in corrected_ranking
            ]
        ),
        "supersession_required": PROTOCOL_STATUS["supersession_required"],
    }
    corrected["semantic_correction"] = correction
    corrected["audit_raw_numerical_data"] = {
        "claim_status": (
            "audit_only_not_a_physical_mirror_identifiable_state_or_performance_publication"
        ),
        "numeric_values_modified": False,
        "prior_semantic_labels": prior_summary,
        "records": audit_records,
    }
    corrected.pop("integrity", None)
    dataset, dataset_hash = write_sealed_json(dataset_path, corrected)
    report_hash = _write_bytes(report_path, _report(dataset).encode("utf-8"))

    files = deepcopy(old_manifest["deterministic_files"])
    for entry in files:
        if entry["kind"] == "dataset":
            entry["file_sha256"] = dataset_hash
            entry["payload_sha256"] = dataset["integrity"]["payload_sha256"]
        elif entry["kind"] == "report":
            entry["file_sha256"] = report_hash
    new_manifest = {
        "schema_version": MANIFEST_VERSION,
        "classification": CLASSIFICATION,
        "protocol_status": PROTOCOL_STATUS,
        "dataset_payload_sha256": dataset["integrity"]["payload_sha256"],
        "deterministic_files": files,
        "representatives": old_manifest["representatives"],
        "semantic_correction": correction,
    }
    manifest, _ = write_sealed_json(manifest_path, new_manifest)
    validated = validate_bundle(output)
    return {"dataset": validated["dataset"], "manifest": manifest}


def main() -> None:
    correct_existing_bundle(Path(__file__).resolve().parent / "results")


if __name__ == "__main__":
    main()
