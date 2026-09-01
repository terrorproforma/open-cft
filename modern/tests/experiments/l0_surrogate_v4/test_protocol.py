"""Scientific inheritance and pre-execution artifact tests for v4."""

from __future__ import annotations

import json

from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v4 import protocol as v4


def test_v4_predeclaration_is_hash_pinned_without_scientific_change() -> None:
    declaration = v4.load_predeclaration()
    assert declaration["predeclaration_hash"] == (
        "419889126694c4a24c0ae7b4e8c201ddfb80f0b13f97392276d88b1b442d6800"
    )
    assert declaration["scientific_protocol"]["scientific_change"] is False
    assert declaration["scientific_protocol"]["active_and_fixed_rows"] == 96
    assert declaration["scientific_protocol"]["quality_gates"] == {
        "range_normalized_rmse_maximum": 0.05,
        "worst_case_range_normalized_absolute_error_maximum": 0.15,
        "coverage_interval": [0.85, 0.95],
        "scopes": ["interpolation", "boundary", "ood", "overall"],
        "acceptance": "every output/scope/replicate passes",
    }


def test_partitions_inherit_v3_without_scientific_delta() -> None:
    partitions = json.loads(v4.PARTITIONS.read_text(encoding="utf-8"))
    assert partitions == v4.build_partitions()
    assert partitions["partitions_hash"] == canonical_hash(
        {key: value for key, value in partitions.items() if key != "partitions_hash"}
    )
    assert partitions["inherited_v3_partitions_hash"] == v4.V3_PARTITIONS_HASH
    assert partitions["scientific_partition_delta"] == "none"
    assert len(partitions["replicates"]) == 3


def test_empty_result_state_before_execution() -> None:
    placeholder = json.loads(
        (v4.ROOT / "results-placeholder.json").read_text(encoding="utf-8")
    )
    assert placeholder["status"] == "not-executed"
    assert placeholder["results"] == []
    assert not v4.RESULTS.exists()


def test_recorded_preflight_is_hash_valid_and_assessment_blind() -> None:
    record = json.loads(v4.PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    assert record["preflight_hash"] == canonical_hash(
        {key: value for key, value in record.items() if key != "preflight_hash"}
    )
    assert record["passed"] is True
    assert record["real_assessment_labels_accessed"] is False
    assert record["synthetic_pipeline"]["replicate_count"] == 3
