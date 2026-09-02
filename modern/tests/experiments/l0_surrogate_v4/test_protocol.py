"""Scientific inheritance and lifecycle-aware artifact tests for v4.

Before the single authorised execution these tests assert the pre-execution
state (no ``results/``). Once ``results/run-manifest.json`` exists they assert
instead that the recorded bundle is bound to the frozen preregistration.
"""

from __future__ import annotations

import json
from pathlib import Path

from cft_revival.surrogates import ExactGP
from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v2 import protocol as science
from experiments.l0_surrogate_v4 import protocol as v4

RUN_MANIFEST = v4.RESULTS / "run-manifest.json"
EXECUTED = RUN_MANIFEST.is_file()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_result_state_and_lifecycle() -> None:
    placeholder = json.loads(
        (v4.ROOT / "results-placeholder.json").read_text(encoding="utf-8")
    )
    assert placeholder["status"] == "not-executed"
    assert placeholder["results"] == []
    if not EXECUTED:
        assert not v4.RESULTS.exists()
        return
    # Executed: the immutable bundle must be bound to the frozen preregistration.
    declaration = v4.load_predeclaration()
    partitions = _load(v4.PARTITIONS)
    preflight = _load(v4.PREFLIGHT_RECORD)
    manifest = _load(RUN_MANIFEST)
    lock = _load(v4.RESULTS / "execution-lock.json")
    declared = manifest.pop("run_manifest_hash")
    assert canonical_hash(manifest) == declared
    for record in (manifest, lock):
        assert record["predeclaration_hash"] == declaration["predeclaration_hash"]
        assert record["partitions_hash"] == partitions["partitions_hash"]
        assert record["preflight_hash"] == preflight["preflight_hash"]
    assert manifest["commit_binding"] == lock["commit_binding"]
    binding = lock["commit_binding"]
    assert lock["single_execution"] is True
    assert binding["intervening_protocol_changes"] == 0
    assert binding["remote_ref"] == "origin/feat/sota-foundation"
    assert set(binding["protocol_paths"]) >= {
        "modern/experiments/l0_surrogate_v4/protocol.py",
        "modern/experiments/l0_surrogate_v4/predeclaration.json",
        "modern/experiments/l0_surrogate_v4/partitions.json",
        "modern/experiments/l0_surrogate_v4/preflight-record.json",
    }
    assert manifest["scientific_identity_valid"] is True
    assert manifest["v3_provenance_failure_hash"] == v4.V3_PROVENANCE_FAILURE_HASH
    assert manifest["status"] in {"accepted", "failed-predeclared-gates"}
    assert manifest["all_active_replicates_passed"] == all(
        item["active_passed"] for item in manifest["replicates"]
    )
    assert [item["replicate_id"] for item in manifest["replicates"]] == [
        item["replicate_id"] for item in partitions["replicates"]
    ]
    for item in manifest["replicates"]:
        assert canonical_hash(
            {key: value for key, value in item.items() if key != "replicate_result_hash"}
        ) == item["replicate_result_hash"]
        frozen = _load(v4.RESULTS / item["replicate_id"] / "frozen-before-assessment.json")
        assert frozen["frozen_hash"] == item["frozen_hash"]
        assert canonical_hash(
            {key: value for key, value in frozen.items() if key != "frozen_hash"}
        ) == item["frozen_hash"]
        for campaign in ("active", "fixed-baseline"):
            assessment = _load(v4.RESULTS / item["replicate_id"] / f"{campaign}.assessment.json")
            assert assessment["frozen_hash"] == item["frozen_hash"]
            for output in science.OUTPUT_NAMES:
                model_path = (
                    v4.RESULTS / item["replicate_id"] / campaign / "models" / f"{output}.model.json"
                )
                assert ExactGP.load(model_path).model_hash == frozen["model_hashes"][campaign][output]


def test_recorded_preflight_is_hash_valid_and_assessment_blind() -> None:
    record = json.loads(v4.PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    assert record["preflight_hash"] == canonical_hash(
        {key: value for key, value in record.items() if key != "preflight_hash"}
    )
    assert record["passed"] is True
    assert record["real_assessment_labels_accessed"] is False
    assert record["synthetic_pipeline"]["replicate_count"] == 3
