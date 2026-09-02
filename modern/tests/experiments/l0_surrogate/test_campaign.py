"""Integrity and reproducibility tests for the L0 emulation campaign."""

from __future__ import annotations

import json
from pathlib import Path

from cft_revival.surrogates import ExactGP
from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate.campaign import (
    DEFAULT_ARTIFACT_DIR,
    OUTPUT_NAMES,
    load_predeclaration,
    run_campaign,
)

EXPERIMENT = Path(__file__).parents[3] / "experiments" / "l0_surrogate"


def _artifact(name: str) -> dict[str, object]:
    value = json.loads((DEFAULT_ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _without_hash(value: dict[str, object], field: str) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != field}


def test_quality_gates_were_hash_pinned_before_execution() -> None:
    declaration = load_predeclaration(EXPERIMENT / "predeclared_campaign.json")
    gates = declaration["quality_gates"]
    assert gates == {
        "assessment_scope": "combined fixed interpolation/boundary/OOD assessment set",
        "range_normalized_rmse_maximum": 0.05,
        "interval_coverage_target": 0.9,
        "interval_coverage_absolute_tolerance": 0.05,
        "minimum_coverage_sample_count": 30,
        "worst_case_range_normalized_absolute_error_maximum": 0.15,
        "stratum_metrics_required": True,
        "ood_detection_report_required": True,
        "thresholds_tunable_after_run": False,
    }
    assert declaration["predeclaration_hash"] == (
        "456431bb8eda55d9f6e854ca7a1a5209545dbcb65b50f68b0223a6394fb1a620"
    )


def test_dataset_partition_has_no_group_or_index_leakage() -> None:
    manifest = _artifact("dataset_manifest.json")
    assert manifest["manifest_hash"] == canonical_hash(
        _without_hash(manifest, "manifest_hash")
    )
    partition = manifest["partition"]
    assessment_groups = partition["assessment_groups"]
    assessment_indices = partition["assessment_indices"]
    calibration_groups = set(partition["calibration_groups"])
    calibration_indices = set(partition["calibration_indices"])
    eligible_indices = set(partition["eligible_indices"])

    heldout_groups: set[str] = set()
    heldout_indices: set[int] = set()
    for stratum in ("interpolation", "boundary", "ood"):
        groups = set(assessment_groups[stratum])
        indices = set(assessment_indices[stratum])
        assert len(indices) >= 48
        assert heldout_groups.isdisjoint(groups)
        assert heldout_indices.isdisjoint(indices)
        heldout_groups.update(groups)
        heldout_indices.update(indices)

    assert heldout_groups.isdisjoint(calibration_groups)
    assert heldout_indices.isdisjoint(calibration_indices)
    assert heldout_indices.isdisjoint(eligible_indices)
    assert calibration_indices.isdisjoint(eligible_indices)
    assert partition["partition_hash"] == canonical_hash(
        _without_hash(partition, "partition_hash")
    )


def test_models_reload_with_exact_hashes_and_bounded_rows() -> None:
    campaign = _artifact("campaign.json")
    model_hashes = campaign["final_model_hashes"]
    for name in OUTPUT_NAMES:
        model = ExactGP.load(DEFAULT_ARTIFACT_DIR / f"{name}.model.json")
        assert model.model_hash == model_hashes[name]
        assert model.diagnostics.training_rows == campaign["final_training_rows"]
        assert (
            model.diagnostics.training_rows
            <= campaign["maximum_exact_gp_training_rows"]
        )
        assert model.diagnostics.length_scale_mode == "ard"


def test_selected_indices_are_unique_and_never_held_out() -> None:
    campaign = _artifact("campaign.json")
    manifest = _artifact("dataset_manifest.json")
    selected = campaign["selected_indices"]
    eligible = set(manifest["partition"]["eligible_indices"])
    assert campaign["selected_indices_unique"] is True
    assert len(selected) == len(set(selected))
    assert set(selected) <= eligible
    flattened_rounds = [
        index
        for round_record in campaign["acquisition_rounds"]
        for index in round_record["selected_indices"]
    ]
    assert selected[32:] == flattened_rounds


def test_saved_status_is_an_honest_conjunction_of_predeclared_gates() -> None:
    campaign = _artifact("campaign.json")
    benchmark = _artifact("benchmark.json")
    combined = benchmark["active"]["combined"]
    recomputed = all(
        output["range_normalized_rmse"] <= 0.05
        and abs(output["interval_coverage"] - 0.9) <= 0.05
        and output["sample_count"] >= 30
        and output["worst_case_range_normalized_absolute_error"] <= 0.15
        for output in combined.values()
    )
    assert benchmark["model_quality_passed"] is recomputed
    assert campaign["all_predeclared_gates_passed"] is recomputed
    assert campaign["stop_reason"] == (
        "all-predeclared-gates-passed"
        if recomputed
        else "maximum-budget-exhausted-with-gates-failed"
    )
    assert benchmark["benchmark_hash"] == canonical_hash(
        _without_hash(benchmark, "benchmark_hash")
    )
    assert campaign["campaign_hash"] == canonical_hash(
        _without_hash(campaign, "campaign_hash")
    )


def test_full_campaign_replays_bit_for_bit() -> None:
    saved_campaign = _artifact("campaign.json")
    saved_benchmark = _artifact("benchmark.json")
    replay = run_campaign(write_artifacts=False)
    assert replay["campaign"] == saved_campaign
    assert replay["benchmark"] == saved_benchmark
    assert replay["dataset_manifest"] == _artifact("dataset_manifest.json")

