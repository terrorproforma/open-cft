"""L0 surrogate v3: immutable v2 science with hardened atomic output."""

from __future__ import annotations

import copy
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from cft_revival.physics import load_l0_json
from cft_revival.surrogates import ExactGP, SurrogateSchema
from cft_revival.surrogates.identity import canonical_hash, strict_json_loads
from experiments.l0_surrogate_v2 import protocol as v2
from experiments.l0_surrogate_v3.serialization import (
    ArtifactWriteError,
    AtomicArtifactStore,
)

MODERN = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
PREDECLARATION = ROOT / "predeclaration.json"
PARTITIONS = ROOT / "partitions.json"
PREFLIGHT_RECORD = ROOT / "preflight-record.json"
RESULTS = ROOT / "results"
V2_ROOT = MODERN / "experiments" / "l0_surrogate_v2"
V2_PREDECLARATION_HASH = "640ec66125d4de07944b012402a64e6cd7be012f1b6877b166b12c4271ff15cf"
V2_FAILURE_HASH = "cc0e608928e6c000c059bcc01f23a288edf9ff99d8736c3c212e70a0c2f1d63d"
V2_PARTITIONS_HASH = "a19d317e148ff0318c38095123a1dad4c8f850833f7df32a821f3f7ad0c91897"


def _load_object(path: Path) -> dict[str, object]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_predeclaration() -> dict[str, object]:
    value = _load_object(PREDECLARATION)
    expected = value.get("predeclaration_hash")
    payload = {key: item for key, item in value.items() if key != "predeclaration_hash"}
    if expected != canonical_hash(payload):
        raise ValueError("v3 predeclaration hash mismatch")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("v3 provenance must be an object")
    if (
        provenance["v2_predeclaration_hash"] != V2_PREDECLARATION_HASH
        or provenance["v2_failure_manifest_hash"] != V2_FAILURE_HASH
        or provenance["v2_partitions_hash"] != V2_PARTITIONS_HASH
    ):
        raise ValueError("v3 provenance does not bind the immutable v2 failure")
    return value


def inherited_protocol() -> dict[str, object]:
    declaration = v2.load_predeclaration()
    if declaration["predeclaration_hash"] != V2_PREDECLARATION_HASH:
        raise ValueError("immutable v2 predeclaration identity changed")
    failure = _load_object(V2_ROOT / "results" / "failure-manifest.json")
    failure_hash = failure.pop("failure_manifest_hash")
    if failure_hash != V2_FAILURE_HASH or canonical_hash(failure) != V2_FAILURE_HASH:
        raise ValueError("immutable v2 failure identity changed")
    return declaration


def build_partitions() -> dict[str, object]:
    v3 = load_predeclaration()
    inherited = _load_object(V2_ROOT / "partitions.json")
    inherited_hash = inherited.pop("partitions_hash")
    if inherited_hash != V2_PARTITIONS_HASH or canonical_hash(inherited) != V2_PARTITIONS_HASH:
        raise ValueError("immutable v2 partition identity changed")
    result = copy.deepcopy(inherited)
    result["document_type"] = "cft-revival-l0-surrogate-v3-input-partitions"
    result["schema_version"] = "3.0"
    result["predeclaration_hash"] = v3["predeclaration_hash"]
    result["inherited_v2_partitions_hash"] = V2_PARTITIONS_HASH
    result["scientific_partition_delta"] = "none"
    result["partitions_hash"] = canonical_hash(result)
    return result


def write_preregistration_partitions() -> dict[str, object]:
    result = build_partitions()
    AtomicArtifactStore(ROOT).write_json("partitions.json", result)
    return result


def _synthetic_model() -> ExactGP:
    inputs = tuple(
        tuple(((index + 1) * (dimension * 6 + 5) % 53) / 53.0 for dimension in range(5))
        for index in range(12)
    )
    outputs = tuple(
        0.01 + 0.02 * row[0] + 0.004 * row[2] + 0.001 * row[4]
        for row in inputs
    )
    return ExactGP.fit(
        inputs,
        outputs,
        schema=SurrogateSchema(
            tuple(f"x{index}" for index in range(5)),
            ("synthetic",),
        ),
        length_scale_mode="ard",
        nominal_probability=0.9,
    )


def _synthetic_pipeline(root: Path) -> dict[str, object]:
    store = AtomicArtifactStore(root)
    model = _synthetic_model()
    model_hashes = []
    frozen_hashes = []
    synthetic_outputs = ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0))
    assessment = {
        "interpolation": {"indices": [0]},
        "boundary": {"indices": [1]},
        "ood": {"indices": [2]},
    }
    for replicate in range(3):
        replicate_id = f"synthetic-replicate-{replicate + 1}"
        selection = {
            "replicate_id": replicate_id,
            "selected_indices": list(range(12)),
            "synthetic_only": True,
        }
        selection["selection_hash"] = canonical_hash(selection)
        store.write_json(f"{replicate_id}/active.selection.json", selection)
        model_path = store.write_model(
            f"{replicate_id}/active/deep/models/synthetic.model.json",
            model,
        )
        reloaded = ExactGP.load(model_path)
        model_hashes.append(reloaded.model_hash)
        calibration = {
            "replicate_id": replicate_id,
            "strata": ["interpolation", "boundary", "ood"],
            "synthetic_only": True,
        }
        calibration["calibration_hash"] = canonical_hash(calibration)
        store.write_json(f"{replicate_id}/active.calibration.json", calibration)
        frozen = {
            "selection_hash": selection["selection_hash"],
            "model_hash": model.model_hash,
            "calibration_hash": calibration["calibration_hash"],
        }
        frozen_hash = canonical_hash(frozen)
        frozen_hashes.append(frozen_hash)
        store.write_json(
            f"{replicate_id}/frozen-before-assessment.json",
            {**frozen, "frozen_hash": frozen_hash},
        )
        loader = v2.SingleUseAssessmentLoader(
            synthetic_outputs,
            assessment,
            frozen_hash,
        )
        labels = loader.load(frozen_hash)
        store.write_json(
            f"{replicate_id}/active.assessment.json",
            {
                "synthetic_only": True,
                "frozen_hash": frozen_hash,
                "strata": list(labels),
            },
        )
        try:
            loader.load(frozen_hash)
        except RuntimeError:
            pass
        else:
            raise AssertionError("synthetic assessment loader was not single-use")
    if store.temporary_files():
        raise AssertionError("synthetic pipeline left temporary files")
    return {
        "replicate_count": 3,
        "model_hashes": model_hashes,
        "frozen_hashes": frozen_hashes,
    }


def preflight(*, record: bool = False) -> dict[str, object]:
    v3 = load_predeclaration()
    effective = inherited_protocol()
    partitions = _load_object(PARTITIONS)
    expected_partitions = build_partitions()
    if partitions != expected_partitions:
        raise ValueError("v3 input-only partitions differ from inherited protocol")
    if RESULTS.exists():
        raise ValueError("real v3 results path already exists")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    source = effective["source"]
    if not isinstance(source, Mapping):
        raise ValueError("inherited source policy is malformed")
    source_config = load_l0_json(MODERN / str(source["config_path"]))
    if canonical_hash(source_config) != source["config_hash"]:
        raise ValueError("accepted source config identity changed")
    free_bytes = shutil.disk_usage(ROOT).free
    minimum_free = int(v3["preflight"]["minimum_free_bytes"])  # type: ignore[index]
    if free_bytes < minimum_free:
        raise RuntimeError("insufficient disk for v3 execution")

    with tempfile.TemporaryDirectory(prefix="cft-l0-v3-preflight-") as directory:
        temporary_root = Path(directory)
        store = AtomicArtifactStore(temporary_root)
        store.write_json("missing/deep/parents/check.json", {"ok": True})
        if json.loads(
            (temporary_root / "missing/deep/parents/check.json").read_text(encoding="utf-8")
        ) != {"ok": True}:
            raise AssertionError("deep-parent atomic JSON round trip failed")

        permission_target = "permission/read-only-target.json"

        def deny_replace(source: object, target: object) -> object:
            raise PermissionError("synthetic read-only/permission failure")

        try:
            store.write_json(permission_target, {"must_not_exist": True}, replace=deny_replace)
        except ArtifactWriteError as error:
            if not isinstance(error.__cause__, PermissionError):
                raise
        else:
            raise AssertionError("permission failure was not propagated")
        if store.path(permission_target).exists() or store.temporary_files():
            raise AssertionError("failed atomic write left target or temporary files")
        synthetic = _synthetic_pipeline(temporary_root / "pipeline")

    record_value: dict[str, object] = {
        "document_type": "cft-revival-l0-surrogate-v3-preflight",
        "schema_version": "3.0",
        "predeclaration_hash": v3["predeclaration_hash"],
        "partitions_hash": partitions["partitions_hash"],
        "v2_failure_manifest_hash": V2_FAILURE_HASH,
        "python": platform.python_version(),
        "minimum_free_bytes": minimum_free,
        "observed_free_bytes_at_check": free_bytes,
        "source_config_hash": source["config_hash"],
        "deep_parent_serialization_passed": True,
        "permission_failure_cleanup_passed": True,
        "atomic_model_reload_passed": True,
        "synthetic_pipeline": synthetic,
        "real_v3_assessment_labels_accessed": False,
        "passed": True,
    }
    record_value["preflight_hash"] = canonical_hash(record_value)
    if record:
        AtomicArtifactStore(ROOT).write_json("preflight-record.json", record_value)
    return record_value


def _write_models(
    store: AtomicArtifactStore,
    replicate_id: str,
    campaign_name: str,
    models: Sequence[ExactGP],
) -> dict[str, str]:
    hashes = {}
    for name, model in zip(v2.OUTPUT_NAMES, models, strict=True):
        store.write_model(
            f"{replicate_id}/{campaign_name}/models/{name}.model.json",
            model,
        )
        hashes[name] = model.model_hash
    return hashes


def execute(preregistration_commit_sha: str) -> dict[str, object]:
    if len(preregistration_commit_sha) != 40:
        raise ValueError("full preregistration commit SHA required")
    if RESULTS.exists():
        raise RuntimeError("v3 execution is single-shot and results already exist")
    v3 = load_predeclaration()
    declaration = inherited_protocol()
    partitions = _load_object(PARTITIONS)
    preflight_record = _load_object(PREFLIGHT_RECORD)
    preflight_hash = preflight_record.pop("preflight_hash")
    if canonical_hash(preflight_record) != preflight_hash or not preflight_record["passed"]:
        raise ValueError("successful hash-valid preflight record required")

    store = AtomicArtifactStore(RESULTS)
    store.write_json(
        "execution-lock.json",
        {
            "document_type": "cft-revival-l0-surrogate-v3-execution-lock",
            "schema_version": "3.0",
            "preregistration_commit_sha": preregistration_commit_sha,
            "predeclaration_hash": v3["predeclaration_hash"],
            "partitions_hash": partitions["partitions_hash"],
            "preflight_hash": preflight_hash,
            "single_execution": True,
        },
    )
    started = perf_counter()
    stage = "load-real-L0-rows"
    assessment_loaded = False
    completed_replicates: list[dict[str, object]] = []
    try:
        inputs, outputs = v2.load_l0_rows(declaration)
        for replicate in partitions["replicates"]:  # type: ignore[union-attr]
            if not isinstance(replicate, Mapping):
                raise ValueError("replicate partition is malformed")
            replicate_id = str(replicate["replicate_id"])
            candidate_indices = tuple(
                int(value) for value in replicate["candidate_indices"]  # type: ignore[arg-type]
            )
            oracle = v2.TrainingOracle(outputs, candidate_indices)
            stage = f"{replicate_id}:active-selection"
            active_indices, active_observed, rounds = v2.select_active_indices(
                inputs,
                candidate_indices,
                oracle,
                declaration,
            )
            fixed_indices = v2.baseline_indices(candidate_indices, declaration)
            fixed_observed = {index: oracle.observe(index) for index in fixed_indices}
            stage = f"{replicate_id}:final-model-fit"
            active_models = v2.fit_models(active_indices, inputs, active_observed)
            fixed_models = v2.fit_models(fixed_indices, inputs, fixed_observed)

            selections: dict[str, dict[str, object]] = {}
            model_hashes: dict[str, dict[str, str]] = {}
            calibrations: dict[str, dict[str, object]] = {}
            for campaign_name, indices, observed, models in (
                ("active", active_indices, active_observed, active_models),
                ("fixed-baseline", fixed_indices, fixed_observed, fixed_models),
            ):
                stage = f"{replicate_id}:{campaign_name}:selection-serialization"
                selection: dict[str, object] = {
                    "document_type": "cft-revival-l0-surrogate-v3-selection",
                    "schema_version": "3.0",
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

                stage = f"{replicate_id}:{campaign_name}:model-serialization"
                model_hashes[campaign_name] = _write_models(
                    store, replicate_id, campaign_name, models
                )
                stage = f"{replicate_id}:{campaign_name}:calibration"
                calibration = v2.fit_conformal(
                    models,
                    replicate["calibration"],  # type: ignore[arg-type]
                    inputs,
                    outputs,
                    float(declaration["intervals"]["nominal_probability"]),  # type: ignore[index]
                )
                calibration["document_type"] = (
                    "cft-revival-l0-surrogate-v3-conformal-calibration"
                )
                calibration["schema_version"] = "3.0"
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
                    f"{replicate_id}/{campaign_name}.calibration.json",
                    calibration,
                )
                calibrations[campaign_name] = calibration

            frozen: dict[str, object] = {
                "replicate_partition_hash": replicate["replicate_partition_hash"],
                "selection_hashes": {
                    name: value["selection_hash"] for name, value in selections.items()
                },
                "model_hashes": model_hashes,
                "calibration_hashes": {
                    name: value["calibration_hash"]
                    for name, value in calibrations.items()
                },
            }
            frozen_hash = canonical_hash(frozen)
            stage = f"{replicate_id}:freeze-before-assessment"
            store.write_json(
                f"{replicate_id}/frozen-before-assessment.json",
                {**frozen, "frozen_hash": frozen_hash},
            )
            loader = v2.SingleUseAssessmentLoader(
                outputs,
                replicate["assessment"],  # type: ignore[arg-type]
                frozen_hash,
            )
            stage = f"{replicate_id}:single-assessment-load"
            labels = loader.load(frozen_hash)
            assessment_loaded = True
            campaign_metrics = {}
            for campaign_name, models in (
                ("active", active_models),
                ("fixed-baseline", fixed_models),
            ):
                stage = f"{replicate_id}:{campaign_name}:assessment"
                raw, metrics = v2.assess(
                    models,
                    calibrations[campaign_name],
                    labels,
                    inputs,
                    declaration,
                )
                assessment: dict[str, object] = {
                    "document_type": "cft-revival-l0-surrogate-v3-final-assessment",
                    "schema_version": "3.0",
                    "replicate_id": replicate_id,
                    "campaign": campaign_name,
                    "frozen_hash": frozen_hash,
                    "raw": raw,
                    "metrics": metrics,
                }
                assessment["assessment_hash"] = canonical_hash(assessment)
                store.write_json(
                    f"{replicate_id}/{campaign_name}.assessment.json",
                    assessment,
                )
                campaign_metrics[campaign_name] = metrics
            result: dict[str, object] = {
                "replicate_id": replicate_id,
                "active": campaign_metrics["active"],
                "fixed-baseline": campaign_metrics["fixed-baseline"],
                "active_passed": campaign_metrics["active"][
                    "all_scopes_outputs_passed"
                ],
                "fixed_baseline_passed": campaign_metrics["fixed-baseline"][
                    "all_scopes_outputs_passed"
                ],
                "frozen_hash": frozen_hash,
            }
            result["replicate_result_hash"] = canonical_hash(result)
            completed_replicates.append(result)
            assessment_loaded = False

        accepted = all(bool(item["active_passed"]) for item in completed_replicates)
        manifest: dict[str, object] = {
            "document_type": "cft-revival-l0-surrogate-v3-run-manifest",
            "schema_version": "3.0",
            "preregistration_commit_sha": preregistration_commit_sha,
            "predeclaration_hash": v3["predeclaration_hash"],
            "partitions_hash": partitions["partitions_hash"],
            "preflight_hash": preflight_hash,
            "v2_failure_manifest_hash": V2_FAILURE_HASH,
            "replicates": completed_replicates,
            "all_active_replicates_passed": accepted,
            "status": "accepted" if accepted else "failed-predeclared-gates",
            "claim": "deterministic L0 software-emulation accuracy only; no physical claim",
        }
        manifest["run_manifest_hash"] = canonical_hash(manifest)
        stage = "run-manifest-serialization"
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
            "document_type": "cft-revival-l0-surrogate-v3-execution-failure",
            "schema_version": "3.0",
            "preregistration_commit_sha": preregistration_commit_sha,
            "predeclaration_hash": v3["predeclaration_hash"],
            "partitions_hash": partitions["partitions_hash"],
            "preflight_hash": preflight_hash,
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_replicates": completed_replicates,
            "assessment_currently_loaded": assessment_loaded,
            "rerun_performed": False,
            "status": "execution-failed",
        }
        failure["failure_manifest_hash"] = canonical_hash(failure)
        try:
            store.write_json("failure-manifest.json", failure)
            store.write_json(
                "runtime-diagnostics.json",
                {
                    "wall_seconds": perf_counter() - started,
                    "role": "diagnostic-only-excluded-from-scientific-hashes",
                    "failure_manifest_hash": failure["failure_manifest_hash"],
                },
            )
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
    execute_parser.add_argument("--preregistration-commit", required=True)
    args = parser.parse_args(argv)
    if args.command == "partitions":
        result = write_preregistration_partitions()
        print(result["partitions_hash"])
    elif args.command == "preflight":
        result = preflight(record=args.record)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = execute(args.preregistration_commit)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
