"""Pre-execution tests for v3 atomic serialization and protocol inheritance."""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from cft_revival.surrogates import ExactGP
from cft_revival.surrogates.identity import canonical_hash
from experiments.l0_surrogate_v2 import protocol as v2
from experiments.l0_surrogate_v3 import protocol as v3
from experiments.l0_surrogate_v3.serialization import (
    ArtifactWriteError,
    AtomicArtifactStore,
)


def test_missing_and_deep_parents_are_created_atomically(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path / "absent-root")
    target = store.write_json("one/two/three/artifact.json", {"value": 7})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 7}
    assert store.temporary_files() == ()


def test_exact_gp_model_creates_parent_and_reloads_hash(tmp_path: Path) -> None:
    model = v3._synthetic_model()
    store = AtomicArtifactStore(tmp_path)
    target = store.write_model("replicate/active/deep/models/model.json", model)
    assert ExactGP.load(target).model_hash == model.model_hash
    assert store.temporary_files() == ()


def test_permission_failure_preserves_existing_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    store = AtomicArtifactStore(tmp_path)
    target = store.write_json("readonly/artifact.json", {"before": True})

    def permission_denied(source: object, destination: object) -> object:
        raise PermissionError("simulated read-only destination")

    with pytest.raises(ArtifactWriteError) as captured:
        store.write_json(
            "readonly/artifact.json",
            {"after": True},
            replace=permission_denied,
        )
    assert isinstance(captured.value.__cause__, PermissionError)
    assert json.loads(target.read_text(encoding="utf-8")) == {"before": True}
    assert store.temporary_files() == ()


def test_atomic_replace_failure_does_not_publish_partial_file(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path)

    def interrupted(source: object, destination: object) -> object:
        raise OSError("simulated interrupted replace")

    with pytest.raises(ArtifactWriteError):
        store.write_bytes("deep/new.bin", b"partial", replace=interrupted)
    assert not store.path("deep/new.bin").exists()
    assert store.temporary_files() == ()


@pytest.mark.parametrize("path", ("../escape.json", "/absolute.json", "a/../../b.json"))
def test_store_rejects_paths_outside_root(tmp_path: Path, path: str) -> None:
    with pytest.raises(ArtifactWriteError):
        AtomicArtifactStore(tmp_path).write_json(path, {})


def test_three_replicate_synthetic_pipeline_writes_complete_layout(
    tmp_path: Path,
) -> None:
    result = v3._synthetic_pipeline(tmp_path)
    assert result["replicate_count"] == 3
    assert len(result["model_hashes"]) == 3
    assert len(result["frozen_hashes"]) == 3
    for replicate in range(1, 4):
        root = tmp_path / f"synthetic-replicate-{replicate}"
        assert (root / "active.selection.json").is_file()
        assert (root / "active/deep/models/synthetic.model.json").is_file()
        assert (root / "active.calibration.json").is_file()
        assert (root / "frozen-before-assessment.json").is_file()
        assert (root / "active.assessment.json").is_file()
    assert AtomicArtifactStore(tmp_path).temporary_files() == ()


def test_preflight_never_loads_real_v3_rows_or_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_real_rows(declaration: object) -> object:
        raise AssertionError("real L0 rows must not be loaded by preflight")

    monkeypatch.setattr(v2, "load_l0_rows", forbidden_real_rows)
    result = v3.preflight(record=False)
    assert result["passed"] is True
    assert result["real_v3_assessment_labels_accessed"] is False


def test_tiny_three_replicate_execute_uses_real_output_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = copy.deepcopy(v2.load_predeclaration())
    declaration["campaign"]["initial_rows"] = 8
    declaration["campaign"]["acquisition_batch_rows"] = 4
    declaration["campaign"]["final_rows"] = 16
    inputs = tuple(
        tuple(((index + 1) * (dimension * 8 + 5) % 97) / 97.0 for dimension in range(5))
        for index in range(84)
    )
    outputs = tuple(
        (
            0.01 + 0.02 * row[0] + 0.004 * row[2],
            800.0 + 1200.0 * row[0] ** 0.5 + 60.0 * row[4],
        )
        for row in inputs
    )
    calibration_indices = {
        "interpolation": list(range(48, 54)),
        "boundary": list(range(54, 60)),
        "ood": list(range(60, 66)),
    }
    assessment_indices = {
        "interpolation": list(range(66, 72)),
        "boundary": list(range(72, 78)),
        "ood": list(range(78, 84)),
    }
    replicates = []
    for seed in declaration["partition"]["replicate_seeds"]:
        replicate = {
            "replicate_id": f"tiny-{seed}",
            "replicate_partition_hash": canonical_hash({"seed": seed}),
            "candidate_indices": list(range(48)),
            "calibration": {
                name: {"groups": [f"cal-{name}"], "indices": indices}
                for name, indices in calibration_indices.items()
            },
            "assessment": {
                name: {"groups": [f"assess-{name}"], "indices": indices}
                for name, indices in assessment_indices.items()
            },
        }
        replicates.append(replicate)
    partitions = {"partitions_hash": "tiny-partitions", "replicates": replicates}
    partitions_path = tmp_path / "partitions.json"
    partitions_path.write_text(json.dumps(partitions), encoding="utf-8")
    preflight = {"passed": True}
    preflight["preflight_hash"] = canonical_hash(preflight)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    monkeypatch.setattr(v3, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(v3, "PARTITIONS", partitions_path)
    monkeypatch.setattr(v3, "PREFLIGHT_RECORD", preflight_path)
    monkeypatch.setattr(v3, "inherited_protocol", lambda: declaration)
    monkeypatch.setattr(v2, "load_l0_rows", lambda ignored: (inputs, outputs))

    manifest = v3.execute("a" * 40)
    assert len(manifest["replicates"]) == 3
    assert (v3.RESULTS / "run-manifest.json").is_file()
    assert not (v3.RESULTS / "failure-manifest.json").exists()
    for replicate in replicates:
        root = v3.RESULTS / replicate["replicate_id"]
        for campaign in ("active", "fixed-baseline"):
            for output in v2.OUTPUT_NAMES:
                model_path = root / campaign / "models" / f"{output}.model.json"
                assert ExactGP.load(model_path).model_hash
            assert (root / f"{campaign}.assessment.json").is_file()
    assert AtomicArtifactStore(v3.RESULTS).temporary_files() == ()


def test_v3_inherits_v2_science_and_binds_failure() -> None:
    declaration = v3.load_predeclaration()
    inherited = v3.inherited_protocol()
    assert inherited == v2.load_predeclaration()
    assert declaration["versioned_delta"]["scientific_change"] is False
    assert declaration["provenance"]["v2_predeclaration_hash"] == (
        inherited["predeclaration_hash"]
    )
    assert declaration["provenance"]["v2_failure_manifest_hash"] == v3.V2_FAILURE_HASH
    assert inherited["campaign"]["final_rows"] == 96
    assert inherited["campaign"]["assessment_based_stopping"] is False
    assert len(inherited["partition"]["replicate_seeds"]) == 3


def test_v3_partitions_are_input_only_and_hash_valid() -> None:
    partitions = json.loads(v3.PARTITIONS.read_text(encoding="utf-8"))
    assert partitions == v3.build_partitions()
    assert partitions["partitions_hash"] == canonical_hash(
        {key: value for key, value in partitions.items() if key != "partitions_hash"}
    )
    assert partitions["inherited_v2_partitions_hash"] == v3.V2_PARTITIONS_HASH
    serialized_replicates = json.dumps(partitions["replicates"]).lower()
    assert "label" not in serialized_replicates
    assert "output" not in serialized_replicates


def test_real_result_path_and_placeholder_are_empty_before_freeze() -> None:
    placeholder = json.loads(
        (v3.ROOT / "results-placeholder.json").read_text(encoding="utf-8")
    )
    assert placeholder["status"] == "not-executed"
    assert placeholder["results"] == []
    assert not v3.RESULTS.exists()


def test_recorded_preflight_is_hash_valid_and_assessment_blind() -> None:
    record = json.loads(v3.PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    assert record["preflight_hash"] == canonical_hash(
        {key: value for key, value in record.items() if key != "preflight_hash"}
    )
    assert record["passed"] is True
    assert record["synthetic_pipeline"]["replicate_count"] == 3
    assert record["real_v3_assessment_labels_accessed"] is False
