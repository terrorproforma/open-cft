"""v2 candidates (GP on derived features, per-stage-count GP mixture), the tree baseline and the learning curve.

The GP machinery is v1's (``fit_botorch_stgp``: BoTorch SingleTaskGP, fixed binomial
noise, Matérn-5/2 ARD with the dimension-scaled prior), re-used verbatim on the
derived-feature training table.  New here: the per-stage-count mixture (one such GP
per realised stage count, all sharing the same prior family and the same fit-role
unit box, with the all-count GP as the declared fallback), a gradient-boosted-trees
baseline (scikit-learn, deterministic) and the fit-role learning curve.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from random import Random
from typing import Any, Mapping, Sequence

import numpy as np

from ..wall_loss_geometry_surrogate_v1 import models as v1m
from ..wall_loss_geometry_surrogate_v1.data import DesignRow
from ..wall_loss_geometry_surrogate_v1.models import (
    BASELINE_ORDER as V1_BASELINE_ORDER,
    CELL_OUTPUTS,
    RIDGE_PENALTIES,
    Baseline,
    FittedCandidate,
    Normaliser,
    TrainingTable,
    ard_length_scales,
    candidate_probabilities,
    fit_botorch_stgp,
    fit_global_mean,
    fit_knn,
    fit_ridge,
    permutation_importance,
    physical_matrix,
    rmse,
    torch_environment,
)
from .features import FEATURE_NAMES, STAGE_COUNTS

__all__ = [
    "BASELINE_ORDER",
    "CANDIDATE_ORDER",
    "CELL_OUTPUTS",
    "GBT_GRID",
    "MIXTURE_MINIMUM_PER_COUNT",
    "RIDGE_PENALTIES",
    "TRANSFORM_OF",
    "Baseline",
    "FittedCandidate",
    "MixtureCandidate",
    "Normaliser",
    "TrainingTable",
    "ard_length_scales",
    "candidate_probabilities",
    "fit_candidate",
    "fit_gbt",
    "fit_global_mean",
    "fit_knn",
    "fit_ridge",
    "fit_stage_mixture",
    "learning_curve",
    "native_latent",
    "permutation_importance",
    "physical_matrix",
    "rmse",
    "subset_table",
    "torch_environment",
]

CANDIDATE_ORDER = ("botorch-stgp-logit", "botorch-stgp-direct", "stage-mixture-stgp-logit")
TRANSFORM_OF = {
    "botorch-stgp-logit": "logit",
    "botorch-stgp-direct": "direct",
    "stage-mixture-stgp-logit": "logit",
}
BASELINE_ORDER = V1_BASELINE_ORDER + ("gbt",)
MIXTURE_MINIMUM_PER_COUNT = 8
STAGE_FEATURE = "stage_count"
GBT_GRID: tuple[dict[str, Any], ...] = tuple(
    {"max_depth": depth, "n_estimators": trees, "learning_rate": 0.05}
    for depth in (1, 2, 3)
    for trees in (100, 300)
)


def subset_table(table: TrainingTable, rows: Sequence[DesignRow]) -> TrainingTable:
    """A training table on a subset of rows that KEEPS the parent's unit box (shared normaliser)."""

    physical = np.asarray([row.inputs for row in rows], dtype=float)
    return TrainingTable(
        table.input_names,
        table.output_names,
        dict(table.trials),
        tuple(rows),
        table.normaliser,
        physical,
        table.normaliser.transform(physical),
    )


# --------------------------------------------------------------------------
# Per-stage-count mixture
# --------------------------------------------------------------------------


@dataclass
class MixtureCandidate(FittedCandidate):
    """One GP per stage count (>= minimum fit designs) plus the all-count GP as the declared fallback."""

    stage_column: int = 0
    stage_minimum: float = 0.0
    stage_span: float = 1.0
    parts: dict[str, FittedCandidate] = field(default_factory=dict, repr=False)

    def stage_keys(self, normalized: np.ndarray) -> list[str]:
        values = np.asarray(normalized, dtype=float)[:, self.stage_column] * self.stage_span + self.stage_minimum
        return [str(int(round(float(v)))) for v in values]

    def route(self, name: str, normalized: np.ndarray) -> list[str]:
        spec = self.output_spec(name)
        dispatch = spec["dispatch"]
        return [dispatch["models"].get(key, dispatch["default"]) for key in self.stage_keys(normalized)]

    def latent(self, normalized: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        normalized = np.asarray(normalized, dtype=float)
        spec = self.output_spec(name)
        routes = self.route(name, normalized)
        mean = np.empty(normalized.shape[0])
        variance = np.empty(normalized.shape[0])
        for model_id in sorted(set(routes)):
            index = np.asarray([i for i, route in enumerate(routes) if route == model_id], dtype=int)
            block_mean, block_variance = self.compiled[model_id].latent(normalized[index], int(spec["task"]))
            mean[index] = block_mean
            variance[index] = block_variance
        return mean, variance


def fit_stage_mixture(
    table: TrainingTable,
    transform: str,
    candidate_id: str,
    *,
    threads: int,
    seed: int,
    minimum_per_count: int = MIXTURE_MINIMUM_PER_COUNT,
) -> MixtureCandidate:
    started = time.perf_counter()
    column = table.input_names.index(STAGE_FEATURE)
    counts = {count: [row for row in table.rows if row.stage_count == count] for count in STAGE_COUNTS}
    fallback = fit_botorch_stgp(table, transform, f"{candidate_id}:all", threads=threads, seed=seed)
    fallback.compile()
    parts: dict[str, FittedCandidate] = {"all": fallback}
    blocks: dict[str, dict[str, Any]] = dict(fallback.blocks)
    served: dict[int, str] = {}
    for count in STAGE_COUNTS:
        rows = counts[count]
        if len(rows) < minimum_per_count:
            continue
        part = fit_botorch_stgp(subset_table(table, rows), transform, f"{candidate_id}:sc{count}", threads=threads, seed=seed)
        part.compile()
        parts[f"sc{count}"] = part
        blocks.update(part.blocks)
        served[count] = f"sc{count}"
    outputs = []
    for name in table.output_names:
        default_block = fallback.output_spec(name)["model"]
        outputs.append(
            {
                "name": name,
                "model": default_block,
                "task": 0,
                "transform": transform,
                "trials": int(table.trials[name]),
                "dispatch": {
                    "feature": STAGE_FEATURE,
                    "rule": f"route a design to the GP fitted on the fit-role designs with the same realised stage count when that count has >= {minimum_per_count} fit designs; otherwise the all-count GP",
                    "models": {str(count): parts[key].output_spec(name)["model"] for count, key in served.items()},
                    "default": default_block,
                },
            }
        )
    diagnostics: dict[str, Any] = {
        "library": fallback.diagnostics["library"] + " (one fit per stage count + all-count fallback)",
        "fit_seconds": time.perf_counter() - started,
        "torch_threads": fallback.diagnostics.get("torch_threads"),
        "fit_designs_per_stage_count": {str(count): len(rows) for count, rows in counts.items()},
        "minimum_per_count": minimum_per_count,
        "served_stage_counts": sorted(served),
        "fallback_stage_counts": sorted(count for count in STAGE_COUNTS if count not in served),
        "per_part": {key: part.diagnostics["per_output"] for key, part in parts.items()},
        "per_output": fallback.diagnostics["per_output"],
    }
    candidate = MixtureCandidate(
        candidate_id,
        transform,
        blocks,
        outputs,
        diagnostics,
        native={},
        stage_column=column,
        stage_minimum=float(table.normaliser.minimum[column]),
        stage_span=float(table.normaliser.span[column]),
        parts=parts,
    )
    candidate.compile()
    return candidate


def native_latent(candidate: FittedCandidate, table: TrainingTable, physical: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Library posterior (torch) at physical inputs, honouring the mixture's routing."""

    normalized = table.normaliser.transform(np.asarray(physical, dtype=float))
    if not isinstance(candidate, MixtureCandidate):
        return v1m.native_latent_botorch(candidate, normalized, name)
    routes = candidate.route(name, normalized)
    by_block = {part.output_spec(name)["model"]: part for part in candidate.parts.values()}
    mean = np.empty(normalized.shape[0])
    variance = np.empty(normalized.shape[0])
    for model_id in sorted(set(routes)):
        index = np.asarray([i for i, route in enumerate(routes) if route == model_id], dtype=int)
        part_mean, part_variance = v1m.native_latent_botorch(by_block[model_id], normalized[index], name)
        mean[index] = part_mean
        variance[index] = part_variance
    return mean, variance


def fit_candidate(candidate_id: str, table: TrainingTable, *, threads: int, seed: int) -> FittedCandidate:
    transform = TRANSFORM_OF[candidate_id]
    if candidate_id.startswith("botorch-stgp"):
        fitted = fit_botorch_stgp(table, transform, candidate_id, threads=threads, seed=seed)
        fitted.compile()
        return fitted
    if candidate_id == "stage-mixture-stgp-logit":
        return fit_stage_mixture(table, transform, candidate_id, threads=threads, seed=seed)
    raise ValueError(f"unknown candidate {candidate_id}")


# --------------------------------------------------------------------------
# Gradient-boosted-trees baseline (P units; scikit-learn, deterministic)
# --------------------------------------------------------------------------


def fit_gbt(table: TrainingTable, parameters: Mapping[str, Any], *, seed: int = 0) -> Baseline:
    from sklearn.ensemble import GradientBoostingRegressor  # noqa: WPS433 (declared optional dependency)

    models: dict[str, Any] = {}
    importances: dict[str, list[float]] = {}
    for name in table.output_names:
        model = GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(parameters["learning_rate"]),
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            subsample=1.0,
            random_state=seed,
        )
        model.fit(table.normalized, table.probabilities(name))
        models[name] = model
        importances[name] = [float(v) for v in model.feature_importances_]
    mean_importance = np.mean(np.asarray([importances[name] for name in table.output_names]), axis=0)

    def predict(x: np.ndarray, name: str) -> np.ndarray:
        return np.clip(models[name].predict(np.asarray(x, dtype=float)), 0.0, 1.0)

    return Baseline(
        "gbt",
        {
            **{key: parameters[key] for key in ("learning_rate", "n_estimators", "max_depth")},
            "random_state": seed,
            "subsample": 1.0,
            "loss": "squared_error",
            "feature_importances": importances,
            "feature_importance_mean_over_outputs": dict(zip(table.input_names, [float(v) for v in mean_importance], strict=True)),
            "feature_ranking_mean_over_outputs": [table.input_names[i] for i in np.argsort(-mean_importance, kind="stable")],
        },
        predict,
    )


# --------------------------------------------------------------------------
# Learning curve (fit-role subsets, scored on a declared evaluation role)
# --------------------------------------------------------------------------


def learning_curve(
    candidate_id: str,
    fit_rows: Sequence[DesignRow],
    evaluation_rows: Sequence[DesignRow],
    input_names: Sequence[str],
    output_names: Sequence[str],
    trials: Mapping[str, int],
    *,
    sizes: Sequence[int],
    seeds: Sequence[int],
    threads: int,
    torch_seed: int,
    namespace: str,
) -> dict[str, Any]:
    """Nested seeded subsets of the fit role; the candidate is refitted from scratch at each size."""

    pool_ids = sorted(row.case_id for row in fit_rows)
    by_id = {row.case_id: row for row in fit_rows}
    evaluation_physical = physical_matrix(evaluation_rows)
    truths = {name: np.asarray([row.counts[name][0] / row.counts[name][1] for row in evaluation_rows]) for name in output_names}
    runs = []
    for seed in seeds:
        rng = Random(int.from_bytes(hashlib.sha256(f"{namespace}:{candidate_id}:{seed}".encode("utf-8")).digest()[:8], "big"))
        order = pool_ids[:]
        rng.shuffle(order)
        curve = []
        for size in sizes:
            ids = sorted(order[: int(size)]) if int(size) < len(pool_ids) else pool_ids
            table = TrainingTable.build([by_id[c] for c in ids], input_names, output_names, trials)
            fitted = fit_candidate(candidate_id, table, threads=threads, seed=torch_seed)
            normalized = table.normaliser.transform(evaluation_physical)
            per_output = {name: rmse(candidate_probabilities(fitted, normalized, name) - truths[name]) for name in output_names}
            per_output["mean_over_outputs"] = float(np.mean([per_output[n] for n in output_names]))
            curve.append({"size": int(size), "rmse": per_output, "case_ids": ids, "stage_counts": {str(k): sum(by_id[c].stage_count == k for c in ids) for k in STAGE_COUNTS}})
        runs.append({"seed": int(seed), "curve": curve})
    summary = []
    for index, size in enumerate(sizes):
        pooled = [run["curve"][index]["rmse"]["p_wall_pooled"] for run in runs]
        mean_over = [run["curve"][index]["rmse"]["mean_over_outputs"] for run in runs]
        summary.append(
            {
                "size": int(size),
                "pooled_rmse_mean": float(np.mean(pooled)),
                "pooled_rmse_min": float(np.min(pooled)),
                "pooled_rmse_max": float(np.max(pooled)),
                "mean_over_outputs_rmse_mean": float(np.mean(mean_over)),
            }
        )
    return {
        "candidate": candidate_id,
        "sizes": [int(s) for s in sizes],
        "seeds": [int(s) for s in seeds],
        "pool_size": len(pool_ids),
        "runs": runs,
        "summary": summary,
        "extrapolation": power_law_extrapolation(summary, target=0.05),
        "gated": False,
    }


def power_law_extrapolation(summary: Sequence[Mapping[str, Any]], *, target: float) -> dict[str, Any]:
    """Least-squares log-log slope of pooled RMSE vs fit size; designs needed to reach ``target`` if the slope is negative."""

    sizes = np.asarray([item["size"] for item in summary], dtype=float)
    values = np.asarray([item["pooled_rmse_mean"] for item in summary], dtype=float)
    if np.any(values <= 0.0) or sizes.size < 2:
        return {"fitted": False, "reason": "non-positive RMSE or fewer than two sizes"}
    design = np.column_stack([np.ones(sizes.size), np.log(sizes)])
    coefficients, *_ = np.linalg.lstsq(design, np.log(values), rcond=None)
    intercept, slope = (float(v) for v in coefficients)
    result: dict[str, Any] = {"fitted": True, "model": "log RMSE = a + b log n", "a": intercept, "b": slope, "target_pooled_rmse": target}
    if slope < 0.0:
        needed = float(np.exp((np.log(target) - intercept) / slope))
        result["designs_needed_for_target"] = needed
        result["statement"] = f"at the fitted rate (b = {slope:.3f}) about {needed:.0f} fit designs would be needed to reach pooled RMSE {target}"
    else:
        result["designs_needed_for_target"] = None
        result["statement"] = "the pooled RMSE does not decrease with fit size over 20..50 designs; more designs of the same kind are not predicted to reach the gate"
    return result
