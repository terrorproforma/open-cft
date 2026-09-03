"""Dataset binding, label extraction, working-space transforms and the frozen role partition.

Every label read goes through :func:`labels_for_role`, so callers can record an
``ExperimentRuntime`` label-access record before the read; the module itself
never caches labels by role.  The dataset is the immutable screening artifact
``geometry-wall-loss-dataset.json`` of ``orbit_wall_loss_geometry_screening_v1``
(classification ``SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS``).
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime.canonical import semantic_sha256, strict_json_file

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent

ROLES = ("fit", "method-selection", "calibration", "assessment")
EXTRAPOLATION = "extrapolation"
ALL_ROLES = ROLES + (EXTRAPOLATION,)
SOURCE_CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"


class DatasetBindingError(RuntimeError):
    """The dataset on disk is not the preregistered screening artifact."""


# --------------------------------------------------------------------------
# Binding (hashes + Git blobs + manifest entry)
# --------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=REPOSITORY, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _blob_of(commit: str, relative: str) -> str | None:
    try:
        return _git("rev-parse", f"{commit}:{relative}")
    except subprocess.CalledProcessError:
        return None


def _is_ancestor(commit: str) -> bool | None:
    try:
        completed = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=REPOSITORY,
            capture_output=True,
        )
    except OSError:
        return None
    return completed.returncode == 0


def dataset_binding_report(spec: Mapping[str, Any], *, use_git: bool = True) -> dict[str, Any]:
    """Byte hashes, Git blob identities and the manifest entry of the dataset."""

    dataset_path = REPOSITORY / spec["dataset_path"]
    manifest_path = REPOSITORY / spec["manifest_path"]
    csv_path = REPOSITORY / spec["dataset_csv_path"]
    dataset_bytes = dataset_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    csv_bytes = csv_path.read_bytes()
    manifest = strict_json_file(manifest_path)
    entry = next(
        (item for item in manifest["artifacts"] if item.get("path") == "artifacts/geometry-wall-loss-dataset.json"),
        None,
    )
    checks = {
        "dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest() == spec["dataset_file_sha256"],
        "dataset_bytes": len(dataset_bytes) == spec["dataset_bytes"],
        "dataset_csv_file_sha256": hashlib.sha256(csv_bytes).hexdigest() == spec["dataset_csv_file_sha256"],
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest() == spec["manifest_file_sha256"],
        "manifest_state": manifest.get("state") == spec["manifest_state"],
        "manifest_entry_matches": (
            entry is not None
            and entry.get("byte_sha256") == spec["dataset_file_sha256"]
            and entry.get("bytes") == spec["dataset_bytes"]
        ),
    }
    git: dict[str, Any] = {"used": use_git}
    if use_git:
        dataset_blob = _blob_of(spec["screening_result_commit"], spec["dataset_path"])
        manifest_blob = _blob_of(spec["screening_result_commit"], spec["manifest_path"])
        working_dataset_blob = _git("hash-object", str(dataset_path))
        working_manifest_blob = _git("hash-object", str(manifest_path))
        git.update(
            {
                "dataset_blob_at_result_commit": dataset_blob,
                "manifest_blob_at_result_commit": manifest_blob,
                "dataset_blob_working_tree": working_dataset_blob,
                "manifest_blob_working_tree": working_manifest_blob,
                "result_commit_is_ancestor_of_head": _is_ancestor(spec["screening_result_commit"]),
                "merge_commit_is_ancestor_of_head": _is_ancestor(spec["screening_merge_commit"]),
            }
        )
        checks["dataset_git_blob"] = (
            dataset_blob == spec["dataset_git_blob"] == working_dataset_blob
        )
        checks["manifest_git_blob"] = (
            manifest_blob == spec["manifest_git_blob"] == working_manifest_blob
        )
        checks["result_commit_is_ancestor"] = git["result_commit_is_ancestor_of_head"] is True
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_bytes": len(dataset_bytes),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_artifact_count": manifest.get("artifact_count"),
        "manifest_state": manifest.get("state"),
        "git": git,
        "screening_result_commit": spec["screening_result_commit"],
        "screening_preregistration_commit": spec["screening_preregistration_commit"],
        "screening_merge_commit": spec["screening_merge_commit"],
    }


def require_dataset_binding(spec: Mapping[str, Any], *, use_git: bool = True) -> dict[str, Any]:
    report = dataset_binding_report(spec, use_git=use_git)
    if not report["passed"]:
        failed = sorted(name for name, ok in report["checks"].items() if not ok)
        raise DatasetBindingError("dataset binding failed: " + ", ".join(failed))
    return report


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignRow:
    case_id: str
    design_id: str
    sweep_index: int
    batch: str
    inputs: tuple[float, ...]
    chamber_length_m: float
    stage_count: int
    counts: Mapping[str, tuple[int, int]]  # output name -> (successes, trials)
    stored_probabilities: Mapping[str, float]
    converged: bool


def load_rows(spec: Mapping[str, Any], input_names: Sequence[str], output_spec: Mapping[str, Any]) -> tuple[DesignRow, ...]:
    """Load every design row; the stored probabilities must equal count ratios exactly."""

    dataset = strict_json_file(REPOSITORY / spec["dataset_path"])
    if dataset["classification"] != SOURCE_CLASSIFICATION:
        raise DatasetBindingError("dataset classification is not the screening label")
    if dataset["design_count"] != spec["design_count"] or len(dataset["designs"]) != spec["design_count"]:
        raise DatasetBindingError("dataset design count differs from the protocol")
    reported_case = spec["reported_case"]
    cells = tuple(output_spec["cells"])
    rows = []
    for record in dataset["designs"]:
        if record["classification"] != SOURCE_CLASSIFICATION:
            raise DatasetBindingError(f"{record['case_id']}: row classification is not the screening label")
        if spec["accepted_2n_convergence_required"] and not record["convergence"]["converged"]:
            raise DatasetBindingError(f"{record['case_id']}: accepted-2N not converged")
        case = record["cases"][reported_case]
        trials = int(case["trial_count"])
        counts: dict[str, tuple[int, int]] = {}
        stored: dict[str, float] = {}
        for index, cell in enumerate(cells, start=1):
            item = record["per_cell"][reported_case][cell]["wall_hit"]
            counts[f"p_wall_cell{index}"] = (int(item["successes"]), int(item["trials"]))
            stored[f"p_wall_cell{index}"] = float(item["probability"])
        counts["p_wall_pooled"] = (int(case["termination_counts"]["wall_hit"]), trials)
        stored["p_wall_pooled"] = float(case["wall_hit"]["probability"])
        counts["p_reflect_pooled"] = (int(case["termination_counts"]["reflected"]), trials)
        stored["p_reflect_pooled"] = float(case["reflected"]["probability"])
        for name, (successes, n) in counts.items():
            if n != int(output_spec["trials"][name]):
                raise DatasetBindingError(f"{record['case_id']}: {name} trials {n} differ from the protocol")
            if stored[name] != successes / n:
                raise DatasetBindingError(f"{record['case_id']}: stored {name} is not the count ratio")
        if sum(counts[f"p_wall_cell{i}"][0] for i in range(1, 5)) != counts["p_wall_pooled"][0]:
            raise DatasetBindingError(f"{record['case_id']}: cell wall counts do not sum to the pooled count")
        rows.append(
            DesignRow(
                case_id=record["case_id"],
                design_id=record["design_id"],
                sweep_index=int(record["sweep_index"]),
                batch=record["batch"],
                inputs=tuple(float(record["design_values"][name]) for name in input_names),
                chamber_length_m=float(record["geometry"]["chamber_length_m"]),
                stage_count=int(record["geometry"]["stage_count"]),
                counts=counts,
                stored_probabilities=stored,
                converged=bool(record["convergence"]["converged"]),
            )
        )
    rows.sort(key=lambda row: row.case_id)
    if len({row.case_id for row in rows}) != len(rows) or len({row.design_id for row in rows}) != len(rows):
        raise DatasetBindingError("case or design ids are not unique")
    return tuple(rows)


def degeneracy_report(rows: Sequence[DesignRow], input_names: Sequence[str], minimum_distinct: int = 8) -> dict[str, Any]:
    distinct = {
        name: len({row.inputs[index] for row in rows}) for index, name in enumerate(input_names)
    }
    tuples = {row.inputs for row in rows}
    passed = all(count >= minimum_distinct for count in distinct.values()) and len(tuples) == len(rows)
    return {
        "distinct_values_per_input": distinct,
        "distinct_input_tuples": len(tuples),
        "rows": len(rows),
        "minimum_distinct_required": minimum_distinct,
        "passed": passed,
    }


# --------------------------------------------------------------------------
# Working-space transforms (known binomial noise)
# --------------------------------------------------------------------------


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def to_working(successes: int, trials: int, transform: str) -> tuple[float, float]:
    """(working value, known observation variance) for one binomial label."""

    if transform == "logit":
        value = math.log((successes + 0.5) / (trials - successes + 0.5))
        variance = 1.0 / (successes + 0.5) + 1.0 / (trials - successes + 0.5)
        return value, variance
    if transform == "direct":
        smoothed = (successes + 1.0) / (trials + 2.0)
        return successes / trials, smoothed * (1.0 - smoothed) / trials
    raise ValueError(f"unknown transform {transform!r}")


def observation_noise_at(mean: float, trials: int, transform: str) -> float:
    """Prediction-time binomial variance of an n-launch replicate at the predicted mean."""

    n = float(trials)
    if transform == "logit":
        p = min(max(sigmoid(mean), 0.5 / (n + 1.0)), (n + 0.5) / (n + 1.0))
        return 1.0 / (n * p * (1.0 - p))
    if transform == "direct":
        p = min(max(mean, 1.0 / (n + 2.0)), (n + 1.0) / (n + 2.0))
        return p * (1.0 - p) / n
    raise ValueError(f"unknown transform {transform!r}")


def working_to_probability(value: float, transform: str) -> float:
    if transform == "logit":
        return sigmoid(value)
    if transform == "direct":
        return min(max(value, 0.0), 1.0)
    raise ValueError(f"unknown transform {transform!r}")


def binomial_floor(labels: Sequence[tuple[int, int]]) -> float:
    """sqrt(mean p(1-p)/n) over labels given as (successes, trials)."""

    if not labels:
        return 0.0
    return math.sqrt(
        math.fsum((s / n) * (1.0 - s / n) / n for s, n in labels) / len(labels)
    )


def floor_corrected(rmse: float, floor: float) -> float:
    return math.sqrt(max(rmse * rmse - floor * floor, 0.0))


# --------------------------------------------------------------------------
# Frozen partition
# --------------------------------------------------------------------------


def _stratum_rng(namespace: str, seed: int, stratum: str) -> Random:
    digest = hashlib.sha256(f"{namespace}:{seed}:{stratum}".encode("utf-8")).digest()
    return Random(int.from_bytes(digest[:8], "big"))


def build_partition(
    rows: Sequence[DesignRow],
    partition_spec: Mapping[str, Any],
    *,
    namespace: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Deterministic role assignment; fails closed on any count mismatch."""

    namespace = partition_spec["seed_namespace"] if namespace is None else namespace
    seed = int(partition_spec["seed"]) if seed is None else int(seed)
    cluster = partition_spec["extrapolation_cluster"]
    count = int(cluster["count"])
    if count != math.ceil(0.10 * len(rows)):
        raise ValueError("extrapolation cluster count is not ceil(0.10 * rows)")
    by_length = sorted(rows, key=lambda row: (-row.chamber_length_m, row.case_id))
    extrapolation = [row.case_id for row in by_length[:count]]
    threshold = by_length[count - 1].chamber_length_m
    if by_length[count].chamber_length_m >= threshold:
        raise ValueError("extrapolation cluster boundary is a tie")
    roles: dict[str, list[str]] = {role: [] for role in ALL_ROLES}
    roles[EXTRAPOLATION] = sorted(extrapolation)
    held = set(extrapolation)
    counts = partition_spec["counts"]
    for stratum in ("primary", "extension"):
        ids = sorted(row.case_id for row in rows if row.batch == stratum and row.case_id not in held)
        expected = counts[stratum]
        if len(ids) != int(expected["remaining_after_extrapolation"]):
            raise ValueError(
                f"stratum {stratum}: {len(ids)} designs remain, protocol declares "
                f"{expected['remaining_after_extrapolation']}"
            )
        if sum(int(expected[role]) for role in ROLES) != len(ids):
            raise ValueError(f"stratum {stratum}: role counts do not sum to the remaining designs")
        rng = _stratum_rng(namespace, seed, stratum)
        rng.shuffle(ids)
        cursor = 0
        for role in ROLES:
            take = int(expected[role])
            roles[role].extend(ids[cursor : cursor + take])
            cursor += take
    for role in ROLES:
        roles[role].sort()
        if len(roles[role]) != int(counts["totals"][role]):
            raise ValueError(f"role {role} has {len(roles[role])} designs, protocol declares {counts['totals'][role]}")
    if len(roles[EXTRAPOLATION]) != int(counts["totals"][EXTRAPOLATION]):
        raise ValueError("extrapolation count differs from the protocol totals")
    all_ids = [case_id for role in ALL_ROLES for case_id in roles[role]]
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != {row.case_id for row in rows}:
        raise ValueError("roles are not a disjoint cover of the designs")
    by_id = {row.case_id: row for row in rows}
    return {
        "schema_version": "cft-revival.wall-loss-geometry-surrogate-v1.partitions/1.0.0",
        "seed_namespace": namespace,
        "seed": seed,
        "roles": roles,
        "role_sha256": {
            role: hashlib.sha256("\n".join(roles[role]).encode("utf-8")).hexdigest() for role in ALL_ROLES
        },
        "counts": {role: len(roles[role]) for role in ALL_ROLES},
        "stratum_counts": {
            stratum: {role: sum(by_id[c].batch == stratum for c in roles[role]) for role in ALL_ROLES}
            for stratum in ("primary", "extension")
        },
        "extrapolation_cluster": {
            "rule": cluster["rule"],
            "chamber_length_threshold_m": threshold,
            "next_shorter_chamber_length_m": by_length[count].chamber_length_m,
            "stage_counts": sorted({by_id[c].stage_count for c in roles[EXTRAPOLATION]}),
            "nondominated_in_cluster": sum(by_id[c].batch == "primary" for c in roles[EXTRAPOLATION]),
            "case_ids": roles[EXTRAPOLATION],
        },
        "design_ids": {role: [by_id[c].design_id for c in roles[role]] for role in ALL_ROLES},
    }


def partition_sha256(partition: Mapping[str, Any]) -> str:
    return semantic_sha256(partition)


def partition_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Role-assignment differences between two partitions of the same designs."""

    same_role = 0
    total = 0
    per_role = {}
    for role in ALL_ROLES:
        a = set(left["roles"][role])
        b = set(right["roles"][role])
        per_role[role] = {"shared": len(a & b), "left": len(a), "right": len(b)}
        same_role += len(a & b)
        total += len(a)
    return {
        "designs_with_same_role": same_role,
        "designs": total,
        "identical": same_role == total,
        "per_role": per_role,
    }


def labels_for_role(
    rows: Sequence[DesignRow], partition: Mapping[str, Any], role: str
) -> tuple[DesignRow, ...]:
    """The rows of one role, in the frozen order. Callers record the label access first."""

    if role not in ALL_ROLES:
        raise ValueError(f"unknown role {role!r}")
    by_id = {row.case_id: row for row in rows}
    return tuple(by_id[case_id] for case_id in partition["roles"][role])
