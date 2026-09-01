"""Commit-bound L0 surrogate experiment v4."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.physics import load_l0_json
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads
from experiments.l0_surrogate_v2 import protocol as science
from experiments.l0_surrogate_v3 import protocol as v3
from experiments.l0_surrogate_v3.serialization import AtomicArtifactStore
from experiments.l0_surrogate_v4.identity import CommitBinding, bind_execution_identity

REPOSITORY = Path(__file__).resolve().parents[3]
MODERN = REPOSITORY / "modern"
ROOT = Path(__file__).resolve().parent
PREDECLARATION = ROOT / "predeclaration.json"
PARTITIONS = ROOT / "partitions.json"
PREFLIGHT_RECORD = ROOT / "preflight-record.json"
RESULTS = ROOT / "results"
V3_PREDECLARATION_HASH = "eaa17e6c26e1c7ff5a3f78dd5dc77161067ad6ca4519e66f77bea4fbd5e84895"
V3_PROVENANCE_FAILURE_HASH = "ec6d262bf4b1c54a198501b53d2cc7742fed21c377e01518d1f6024d47672af0"
V3_PARTITIONS_HASH = "97332e2ed553d6281ff0b030357fbe47e4f292beebea6d101e211465a506e09f"


def _load(path: Path) -> dict[str, object]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_predeclaration() -> dict[str, object]:
    value = _load(PREDECLARATION)
    expected = value.get("predeclaration_hash")
    payload = {key: item for key, item in value.items() if key != "predeclaration_hash"}
    if expected != canonical_hash(payload):
        raise ValueError("v4 predeclaration hash mismatch")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("v4 provenance must be an object")
    if (
        provenance["v3_predeclaration_hash"] != V3_PREDECLARATION_HASH
        or provenance["v3_provenance_failure_hash"] != V3_PROVENANCE_FAILURE_HASH
        or provenance["v3_partitions_hash"] != V3_PARTITIONS_HASH
    ):
        raise ValueError("v4 does not bind immutable v3 evidence")
    return value


def inherited_science() -> dict[str, object]:
    v3_declaration = v3.load_predeclaration()
    if v3_declaration["predeclaration_hash"] != V3_PREDECLARATION_HASH:
        raise ValueError("immutable v3 predeclaration changed")
    return v3.inherited_protocol()


def build_partitions() -> dict[str, object]:
    declaration = load_predeclaration()
    inherited = _load(v3.PARTITIONS)
    inherited_hash = inherited.pop("partitions_hash")
    if inherited_hash != V3_PARTITIONS_HASH or canonical_hash(inherited) != inherited_hash:
        raise ValueError("immutable v3 partitions changed")
    result = copy.deepcopy(inherited)
    result["document_type"] = "cft-revival-l0-surrogate-v4-input-partitions"
    result["schema_version"] = "4.0"
    result["predeclaration_hash"] = declaration["predeclaration_hash"]
    result["inherited_v3_partitions_hash"] = V3_PARTITIONS_HASH
    result["scientific_partition_delta"] = "none"
    result["partitions_hash"] = canonical_hash(result)
    return result


def write_partitions() -> dict[str, object]:
    value = build_partitions()
    AtomicArtifactStore(ROOT).write_json("partitions.json", value)
    return value


def preflight(*, record: bool = False) -> dict[str, object]:
    declaration = load_predeclaration()
    effective = inherited_science()
    partitions = _load(PARTITIONS)
    if partitions != build_partitions():
        raise ValueError("v4 partition artifact mismatch")
    if RESULTS.exists():
        raise ValueError("v4 results path already exists")
    source = effective["source"]
    if not isinstance(source, Mapping):
        raise ValueError("source policy is malformed")
    source_config = load_l0_json(MODERN / str(source["config_path"]))
    if canonical_hash(source_config) != source["config_hash"]:
        raise ValueError("accepted source config identity changed")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    free_bytes = shutil.disk_usage(ROOT).free
    with tempfile.TemporaryDirectory(prefix="cft-l0-v4-preflight-") as directory:
        synthetic = v3._synthetic_pipeline(Path(directory))
    result: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v4-preflight",
        "schema_version": "4.0",
        "predeclaration_hash": declaration["predeclaration_hash"],
        "partitions_hash": partitions["partitions_hash"],
        "source_config_hash": source["config_hash"],
        "free_bytes_at_check": free_bytes,
        "synthetic_pipeline": synthetic,
        "identity_tests": [
            "nonexistent-sha",
            "wrong-ancestor",
            "dirty-v4-file",
            "unrelated-intervening-commit",
            "valid-head",
        ],
        "real_assessment_labels_accessed": False,
        "passed": True,
    }
    result["preflight_hash"] = canonical_hash(result)
    if record:
        AtomicArtifactStore(ROOT).write_json("preflight-record.json", result)
    return result


def _models(
    store: AtomicArtifactStore,
    replicate_id: str,
    campaign_name: str,
    models: Sequence[object],
) -> dict[str, str]:
    hashes = {}
    for name, model in zip(science.OUTPUT_NAMES, models, strict=True):
        store.write_model(
            f"{replicate_id}/{campaign_name}/models/{name}.model.json",
            model,  # type: ignore[arg-type]
        )
        hashes[name] = model.model_hash  # type: ignore[attr-defined]
    return hashes


def execute(*, expected_head_sha: str | None = None) -> dict[str, object]:
    """Validate Git identity first, then access real L0 rows exactly once."""

    binding = bind_execution_identity(
        REPOSITORY,
        expected_head_sha=expected_head_sha,
    )
    if RESULTS.exists():
        raise RuntimeError("v4 execution is single-shot and results already exist")
    declaration = load_predeclaration()
    effective = inherited_science()
    partitions = _load(PARTITIONS)
    preflight_record = _load(PREFLIGHT_RECORD)
    preflight_hash = preflight_record.pop("preflight_hash")
    if canonical_hash(preflight_record) != preflight_hash or not preflight_record["passed"]:
        raise ValueError("hash-valid successful v4 preflight required")

    store = AtomicArtifactStore(RESULTS)
    store.write_json(
        "execution-lock.json",
        {
            "document_type": "cft-revival-l0-surrogate-v4-execution-lock",
            "schema_version": "4.0",
            "commit_binding": binding.to_dict(),
            "predeclaration_hash": declaration["predeclaration_hash"],
            "partitions_hash": partitions["partitions_hash"],
            "preflight_hash": preflight_hash,
            "single_execution": True,
        },
    )
    started = perf_counter()
    stage = "load-real-L0-rows"
    completed: list[dict[str, object]] = []
    assessment_loaded = False
    try:
        inputs, outputs = science.load_l0_rows(effective)
        for replicate in partitions["replicates"]:  # type: ignore[union-attr]
            if not isinstance(replicate, Mapping):
                raise ValueError("replicate partition is malformed")
            replicate_id = str(replicate["replicate_id"])
            candidates = tuple(int(value) for value in replicate["candidate_indices"])  # type: ignore[arg-type]
            oracle = science.TrainingOracle(outputs, candidates)
            stage = f"{replicate_id}:active-selection"
            active_indices, active_observed, rounds = science.select_active_indices(
                inputs, candidates, oracle, effective
            )
            fixed_indices = science.baseline_indices(candidates, effective)
            fixed_observed = {index: oracle.observe(index) for index in fixed_indices}
            stage = f"{replicate_id}:model-fit"
            active_models = science.fit_models(active_indices, inputs, active_observed)
            fixed_models = science.fit_models(fixed_indices, inputs, fixed_observed)

            selections: dict[str, dict[str, object]] = {}
            model_hashes: dict[str, dict[str, str]] = {}
            calibrations: dict[str, dict[str, object]] = {}
            for campaign_name, indices, observed, models in (
                ("active", active_indices, active_observed, active_models),
                ("fixed-baseline", fixed_indices, fixed_observed, fixed_models),
            ):
                stage = f"{replicate_id}:{campaign_name}:selection"
                selection: dict[str, object] = {
                    "document_type": "cft-revival-l0-surrogate-v4-selection",
                    "schema_version": "4.0",
                    "replicate_id": replicate_id,
                    "campaign": campaign_name,
                    "selected_indices": list(indices),
                    "observations": [
                        {"index": index, "outputs": list(observed[index])}
                        for index in indices
                    ],
                    "acquisition_rounds": rounds if campaign_name == "active" else [],
                    "final_rows": len(indices),
                }
                selection["selection_hash"] = canonical_hash(selection)
                store.write_json(f"{replicate_id}/{campaign_name}.selection.json", selection)
                selections[campaign_name] = selection
                stage = f"{replicate_id}:{campaign_name}:models"
                model_hashes[campaign_name] = _models(
                    store, replicate_id, campaign_name, models
                )
                stage = f"{replicate_id}:{campaign_name}:calibration"
                calibration = science.fit_conformal(
                    models,
                    replicate["calibration"],  # type: ignore[arg-type]
                    inputs,
                    outputs,
                    float(effective["intervals"]["nominal_probability"]),  # type: ignore[index]
                )
                calibration["document_type"] = (
                    "cft-revival-l0-surrogate-v4-conformal-calibration"
                )
                calibration["schema_version"] = "4.0"
                calibration["replicate_id"] = replicate_id
                calibration["campaign"] = campaign_name
                calibration["calibration_hash"] = canonical_hash(
                    {
                        key: value
                        for key, value in calibration.items()
                        if key != "calibration_hash"
                    }
                )
                store.write_json(
                    f"{replicate_id}/{campaign_name}.calibration.json", calibration
                )
                calibrations[campaign_name] = calibration

            frozen: dict[str, object] = {
                "replicate_partition_hash": replicate["replicate_partition_hash"],
                "selection_hashes": {
                    name: item["selection_hash"] for name, item in selections.items()
                },
                "model_hashes": model_hashes,
                "calibration_hashes": {
                    name: item["calibration_hash"] for name, item in calibrations.items()
                },
            }
            frozen_hash = canonical_hash(frozen)
            store.write_json(
                f"{replicate_id}/frozen-before-assessment.json",
                {**frozen, "frozen_hash": frozen_hash},
            )
            stage = f"{replicate_id}:single-assessment-load"
            loader = science.SingleUseAssessmentLoader(
                outputs,
                replicate["assessment"],  # type: ignore[arg-type]
                frozen_hash,
            )
            labels = loader.load(frozen_hash)
            assessment_loaded = True
            metrics_by_campaign = {}
            for campaign_name, models in (
                ("active", active_models),
                ("fixed-baseline", fixed_models),
            ):
                stage = f"{replicate_id}:{campaign_name}:assessment"
                raw, metrics = science.assess(
                    models,
                    calibrations[campaign_name],
                    labels,
                    inputs,
                    effective,
                )
                assessment: dict[str, object] = {
                    "document_type": "cft-revival-l0-surrogate-v4-final-assessment",
                    "schema_version": "4.0",
                    "replicate_id": replicate_id,
                    "campaign": campaign_name,
                    "frozen_hash": frozen_hash,
                    "raw": raw,
                    "metrics": metrics,
                }
                assessment["assessment_hash"] = canonical_hash(assessment)
                store.write_json(
                    f"{replicate_id}/{campaign_name}.assessment.json", assessment
                )
                metrics_by_campaign[campaign_name] = metrics
            result: dict[str, object] = {
                "replicate_id": replicate_id,
                "active": metrics_by_campaign["active"],
                "fixed-baseline": metrics_by_campaign["fixed-baseline"],
                "active_passed": metrics_by_campaign["active"][
                    "all_scopes_outputs_passed"
                ],
                "fixed_baseline_passed": metrics_by_campaign["fixed-baseline"][
                    "all_scopes_outputs_passed"
                ],
                "frozen_hash": frozen_hash,
            }
            result["replicate_result_hash"] = canonical_hash(result)
            completed.append(result)
            assessment_loaded = False

        accepted = all(bool(item["active_passed"]) for item in completed)
        manifest: dict[str, object] = {
            "document_type": "cft-revival-l0-surrogate-v4-run-manifest",
            "schema_version": "4.0",
            "commit_binding": binding.to_dict(),
            "predeclaration_hash": declaration["predeclaration_hash"],
            "partitions_hash": partitions["partitions_hash"],
            "preflight_hash": preflight_hash,
            "v3_provenance_failure_hash": V3_PROVENANCE_FAILURE_HASH,
            "replicates": completed,
            "scientific_identity_valid": True,
            "all_active_replicates_passed": accepted,
            "status": "accepted" if accepted else "failed-predeclared-gates",
            "claim": "deterministic L0 software-emulation accuracy only; no physical claim",
        }
        manifest["run_manifest_hash"] = canonical_hash(manifest)
        store.write_json("run-manifest.json", manifest)
        store.write_json(
            "runtime-diagnostics.json",
            {
                "wall_seconds": perf_counter() - started,
                "role": "diagnostic-only-excluded-from-scientific-hashes",
                "run_manifest_hash": manifest["run_manifest_hash"],
            },
        )
        return manifest
    except Exception as error:
        failure: dict[str, object] = {
            "document_type": "cft-revival-l0-surrogate-v4-execution-failure",
            "schema_version": "4.0",
            "commit_binding": binding.to_dict(),
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_replicates": completed,
            "assessment_currently_loaded": assessment_loaded,
            "rerun_performed": False,
        }
        failure["failure_manifest_hash"] = canonical_hash(failure)
        try:
            store.write_json("failure-manifest.json", failure)
        finally:
            raise


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("partitions")
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--record", action="store_true")
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--expected-head")
    args = parser.parse_args(argv)
    if args.command == "partitions":
        result = write_partitions()
        print(result["partitions_hash"])
    elif args.command == "preflight":
        result = preflight(record=args.record)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = execute(expected_head_sha=args.expected_head)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
