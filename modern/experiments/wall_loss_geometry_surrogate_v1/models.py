"""Candidate surrogates, baselines and their export to the predictor contract.

Every candidate is fitted on the FIT role only and exported to one or more
contract blocks (see :mod:`.predictor`).  All downstream scoring goes through
the compiled numpy blocks; the native posterior of each library is kept only
for the ``predictor_contract_replay`` gate.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from random import Random
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.surrogates import ExactGP, SurrogateSchema

from .data import DesignRow, to_working
from .predictor import MODEL_KIND, CompiledModel

CANDIDATE_ORDER = (
    "pkg-exactgp-logit",
    "pkg-exactgp-direct",
    "botorch-stgp-logit",
    "botorch-stgp-direct",
    "botorch-icm-logit",
)
TRANSFORM_OF = {
    "pkg-exactgp-logit": "logit",
    "pkg-exactgp-direct": "direct",
    "botorch-stgp-logit": "logit",
    "botorch-stgp-direct": "direct",
    "botorch-icm-logit": "logit",
}
BASELINE_ORDER = ("global-mean", "knn-3", "ridge")
RIDGE_PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CELL_OUTPUTS = ("p_wall_cell1", "p_wall_cell2", "p_wall_cell3", "p_wall_cell4")


# --------------------------------------------------------------------------
# Training table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Normaliser:
    minimum: tuple[float, ...]
    span: tuple[float, ...]

    @classmethod
    def fit(cls, physical: np.ndarray) -> "Normaliser":
        minimum = physical.min(axis=0)
        maximum = physical.max(axis=0)
        span = np.where(maximum > minimum, maximum - minimum, 1.0)
        return cls(tuple(float(v) for v in minimum), tuple(float(v) for v in span))

    def transform(self, physical: np.ndarray) -> np.ndarray:
        return (np.asarray(physical, dtype=float) - np.asarray(self.minimum)) / np.asarray(self.span)

    def to_dict(self) -> dict[str, list[float]]:
        return {"minimum": list(self.minimum), "span": list(self.span)}


@dataclass(frozen=True)
class TrainingTable:
    """Fit-role inputs (physical + normalised) and per-output binomial counts."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    trials: Mapping[str, int]
    rows: tuple[DesignRow, ...]
    normaliser: Normaliser
    physical: np.ndarray
    normalized: np.ndarray

    @classmethod
    def build(
        cls,
        rows: Sequence[DesignRow],
        input_names: Sequence[str],
        output_names: Sequence[str],
        trials: Mapping[str, int],
    ) -> "TrainingTable":
        physical = np.asarray([row.inputs for row in rows], dtype=float)
        normaliser = Normaliser.fit(physical)
        return cls(
            tuple(input_names),
            tuple(output_names),
            dict(trials),
            tuple(rows),
            normaliser,
            physical,
            normaliser.transform(physical),
        )

    def working(self, output: str, transform: str) -> tuple[np.ndarray, np.ndarray]:
        pairs = [to_working(*row.counts[output], transform) for row in self.rows]
        return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])

    def probabilities(self, output: str) -> np.ndarray:
        return np.asarray([row.counts[output][0] / row.counts[output][1] for row in self.rows], dtype=float)


def physical_matrix(rows: Sequence[DesignRow]) -> np.ndarray:
    return np.asarray([row.inputs for row in rows], dtype=float)


# --------------------------------------------------------------------------
# Fitted candidate = contract blocks + native replay hooks
# --------------------------------------------------------------------------


@dataclass
class FittedCandidate:
    candidate_id: str
    transform: str
    blocks: dict[str, dict[str, Any]]
    outputs: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    native: dict[str, Any] = field(default_factory=dict, repr=False)
    compiled: dict[str, CompiledModel] = field(default_factory=dict, repr=False)

    def compile(self) -> None:
        self.compiled = {model_id: CompiledModel(model_id, block) for model_id, block in self.blocks.items()}

    def output_spec(self, name: str) -> dict[str, Any]:
        return next(item for item in self.outputs if item["name"] == name)

    def latent(self, normalized: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        spec = self.output_spec(name)
        return self.compiled[spec["model"]].latent(np.asarray(normalized, dtype=float), int(spec["task"]))

    def hyperparameter_vector(self) -> np.ndarray:
        values: list[float] = []
        for model_id in sorted(self.blocks):
            block = self.blocks[model_id]
            values.extend(float(v) for v in block["lengthscales"])
            values.append(float(block["outputscale"]))
            values.extend(float(v) for row in block["task_covariance"] for v in row)
            values.extend(float(v) for v in block["mean_constants"])
            values.append(float(block["standardize"]["mean"]))
            values.append(float(block["standardize"]["scale"]))
        return np.asarray(values, dtype=float)


def _block(
    *,
    family: str,
    lengthscales: Sequence[float],
    outputscale: float,
    task_covariance: Sequence[Sequence[float]],
    mean_constants: Sequence[float],
    standardize_mean: float,
    standardize_scale: float,
    train_x: np.ndarray,
    train_task: Sequence[int],
    y_working: Sequence[float],
    noise_working: Sequence[float],
    jitter: float,
    outputs: Sequence[str],
) -> dict[str, Any]:
    return {
        "kind": MODEL_KIND,
        "family": family,
        "outputs": list(outputs),
        "lengthscales": [float(v) for v in lengthscales],
        "outputscale": float(outputscale),
        "task_covariance": [[float(v) for v in row] for row in task_covariance],
        "mean_constants": [float(v) for v in mean_constants],
        "standardize": {"mean": float(standardize_mean), "scale": float(standardize_scale)},
        "train": {
            "x": [[float(v) for v in row] for row in np.asarray(train_x, dtype=float)],
            "task": [int(v) for v in train_task],
            "y_working": [float(v) for v in y_working],
            "noise_working": [float(v) for v in noise_working],
            "jitter": float(jitter),
        },
    }


# ---- package ExactGP --------------------------------------------------------


def fit_pkg_exactgp(table: TrainingTable, transform: str, candidate_id: str) -> FittedCandidate:
    blocks: dict[str, dict[str, Any]] = {}
    outputs = []
    diagnostics: dict[str, Any] = {"per_output": {}}
    native: dict[str, Any] = {}
    started = time.perf_counter()
    for name in table.output_names:
        y, noise = table.working(name, transform)
        model = ExactGP.fit(
            table.physical.tolist(),
            y.tolist(),
            observation_variance=noise.tolist(),
            schema=SurrogateSchema(table.input_names, (name,)),
            length_scale_mode="ard",
            nominal_probability=0.9,
        )
        normaliser = model.input_normalizer.to_dict()
        if list(normaliser["minimum"]) != list(table.normaliser.minimum) or list(normaliser["span"]) != list(table.normaliser.span):
            raise RuntimeError("ExactGP input normaliser differs from the fit-role unit box")
        artifact = model.to_dict()
        fitted = artifact["fitted_parameters"]
        block_id = f"{candidate_id}:{name}"
        blocks[block_id] = _block(
            family=candidate_id,
            lengthscales=fitted["length_scales"],
            outputscale=fitted["signal_variance"],
            task_covariance=[[1.0]],
            mean_constants=[0.0],
            standardize_mean=model.output_normalizer.mean,
            standardize_scale=model.output_normalizer.scale,
            train_x=table.normalized,
            train_task=[0] * len(table.rows),
            y_working=y,
            noise_working=noise,
            jitter=fitted["jitter"],
            outputs=[name],
        )
        outputs.append({"name": name, "model": block_id, "task": 0, "transform": transform, "trials": int(table.trials[name])})
        diagnostics["per_output"][name] = {
            "length_scales": list(fitted["length_scales"]),
            "signal_variance": fitted["signal_variance"],
            "jitter": fitted["jitter"],
            "log_marginal_likelihood": fitted["log_marginal_likelihood"],
            "model_hash": artifact["model_hash"],
        }
        native[name] = model
    diagnostics["fit_seconds"] = time.perf_counter() - started
    diagnostics["library"] = "cft_revival.surrogates.ExactGP"
    return FittedCandidate(candidate_id, transform, blocks, outputs, diagnostics, native)


def native_latent_pkg(candidate: FittedCandidate, physical: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    model: ExactGP = candidate.native[name]
    predictions = model.predict(np.asarray(physical, dtype=float).tolist())
    return (
        np.asarray([p.mean for p in predictions], dtype=float),
        np.asarray([p.variance for p in predictions], dtype=float),
    )


# ---- BoTorch ---------------------------------------------------------------


def torch_api(threads: int, device: str = "cpu") -> dict[str, Any]:
    """Import the CPU float64 BoTorch stack lazily; fail closed on CUDA requests."""

    if device != "cpu":
        raise RuntimeError("this campaign runs on cpu only")
    import torch  # noqa: WPS433 (lazy optional dependency)
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import MultiTaskGP, SingleTaskGP
    from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.set_num_threads(int(threads))
    return {
        "torch": torch,
        "SingleTaskGP": SingleTaskGP,
        "MultiTaskGP": MultiTaskGP,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "covar": get_covar_module_with_dim_scaled_prior,
    }


def torch_environment(threads: int) -> dict[str, Any]:
    api = torch_api(threads)
    torch = api["torch"]
    probe = torch.linalg.cholesky(torch.tensor([[4.0, 2.0], [2.0, 3.0]], dtype=torch.float64))
    return {
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "float64_cholesky_probe": [float(v) for v in probe.flatten().tolist()],
        "torch_version": str(torch.__version__),
        "cuda_used": False,
    }


def _marginal_log_likelihood(model: Any, mll: Any) -> float:
    """Per-row exact marginal log likelihood at the fitted hyperparameters (train mode)."""

    import torch  # noqa: WPS433

    model.train()
    with torch.no_grad():
        value = float(mll(model(*model.train_inputs), model.train_targets).item())
    model.eval()
    return value


def _data_kernel_parameters(kernel: Any) -> tuple[list[float], float]:
    base = kernel.base_kernel if hasattr(kernel, "base_kernel") else kernel
    if getattr(base, "nu", None) != 2.5:
        raise RuntimeError("data kernel is not Matern-5/2")
    lengthscales = [float(v) for v in base.lengthscale.detach().flatten().tolist()]
    outputscale = float(kernel.outputscale.item()) if hasattr(kernel, "outputscale") else 1.0
    return lengthscales, outputscale


def fit_botorch_stgp(table: TrainingTable, transform: str, candidate_id: str, *, threads: int, seed: int) -> FittedCandidate:
    api = torch_api(threads)
    torch = api["torch"]
    blocks: dict[str, dict[str, Any]] = {}
    outputs = []
    diagnostics: dict[str, Any] = {"per_output": {}}
    native: dict[str, Any] = {}
    started = time.perf_counter()
    train_x = torch.tensor(table.normalized, dtype=torch.float64)
    for name in table.output_names:
        y, noise = table.working(name, transform)
        torch.manual_seed(seed)
        model = api["SingleTaskGP"](
            train_x,
            torch.tensor(y[:, None], dtype=torch.float64),
            train_Yvar=torch.tensor(noise[:, None], dtype=torch.float64),
            covar_module=api["covar"](ard_num_dims=train_x.shape[1], use_rbf_kernel=False),
        )
        mll = api["ExactMarginalLogLikelihood"](model.likelihood, model)
        api["fit_gpytorch_mll"](mll)
        mll_value = _marginal_log_likelihood(model, mll)
        model.eval()
        lengthscales, outputscale = _data_kernel_parameters(model.covar_module)
        constant = float(model.mean_module.constant.item())
        standardize = model.outcome_transform
        block_id = f"{candidate_id}:{name}"
        blocks[block_id] = _block(
            family=candidate_id,
            lengthscales=lengthscales,
            outputscale=outputscale,
            task_covariance=[[1.0]],
            mean_constants=[constant],
            standardize_mean=float(standardize.means.flatten()[0].item()),
            standardize_scale=float(standardize.stdvs.flatten()[0].item()),
            train_x=table.normalized,
            train_task=[0] * len(table.rows),
            y_working=y,
            noise_working=noise,
            jitter=0.0,
            outputs=[name],
        )
        outputs.append({"name": name, "model": block_id, "task": 0, "transform": transform, "trials": int(table.trials[name])})
        diagnostics["per_output"][name] = {
            "length_scales": lengthscales,
            "outputscale": outputscale,
            "mean_constant": constant,
            "standardize_mean": blocks[block_id]["standardize"]["mean"],
            "standardize_scale": blocks[block_id]["standardize"]["scale"],
            "marginal_log_likelihood_per_row": mll_value,
        }
        native[name] = model
    diagnostics["fit_seconds"] = time.perf_counter() - started
    diagnostics["library"] = "botorch.SingleTaskGP(train_Yvar) + get_covar_module_with_dim_scaled_prior(Matern-5/2)"
    diagnostics["torch_threads"] = torch.get_num_threads()
    return FittedCandidate(candidate_id, transform, blocks, outputs, diagnostics, native)


def fit_botorch_icm(
    table: TrainingTable,
    transform: str,
    candidate_id: str,
    *,
    threads: int,
    seed: int,
    rank: int = 2,
    single_task_for: Sequence[str] = (),
) -> FittedCandidate:
    """ICM over the four cell outputs; the pooled outputs come from single-task fits."""

    api = torch_api(threads)
    torch = api["torch"]
    started = time.perf_counter()
    dim = table.normalized.shape[1]
    n = len(table.rows)
    cells = [name for name in table.output_names if name in CELL_OUTPUTS]
    if len(cells) != 4:
        raise RuntimeError("the ICM candidate requires the four cell outputs")
    xs, ys, vs = [], [], []
    for task, name in enumerate(cells):
        y, noise = table.working(name, transform)
        xs.append(np.hstack([table.normalized, np.full((n, 1), float(task))]))
        ys.append(y)
        vs.append(noise)
    train_x = torch.tensor(np.vstack(xs), dtype=torch.float64)
    train_y = torch.tensor(np.concatenate(ys)[:, None], dtype=torch.float64)
    train_yvar = torch.tensor(np.concatenate(vs)[:, None], dtype=torch.float64)
    torch.manual_seed(seed)
    model = api["MultiTaskGP"](
        train_x,
        train_y,
        task_feature=-1,
        train_Yvar=train_yvar,
        rank=rank,
        covar_module=api["covar"](ard_num_dims=dim, use_rbf_kernel=False),
    )
    mll = api["ExactMarginalLogLikelihood"](model.likelihood, model)
    api["fit_gpytorch_mll"](mll)
    model.eval()
    kernels = list(model.covar_module.kernels)
    lengthscales, outputscale = _data_kernel_parameters(kernels[0])
    task_covariance = kernels[1].covar_matrix.to_dense().detach().numpy().tolist()
    means = [float(m.constant.item()) for m in model.mean_module.base_means]
    standardize = model.outcome_transform
    block_id = f"{candidate_id}:cells"
    blocks = {
        block_id: _block(
            family=candidate_id,
            lengthscales=lengthscales,
            outputscale=outputscale,
            task_covariance=task_covariance,
            mean_constants=means,
            standardize_mean=float(standardize.means.flatten()[0].item()),
            standardize_scale=float(standardize.stdvs.flatten()[0].item()),
            train_x=np.vstack([table.normalized] * 4),
            train_task=[task for task in range(4) for _ in range(n)],
            y_working=np.concatenate(ys),
            noise_working=np.concatenate(vs),
            jitter=0.0,
            outputs=cells,
        )
    }
    outputs = [
        {"name": name, "model": block_id, "task": task, "transform": transform, "trials": int(table.trials[name])}
        for task, name in enumerate(cells)
    ]
    native: dict[str, Any] = {"cells": model}
    diagnostics: dict[str, Any] = {
        "icm": {
            "length_scales": lengthscales,
            "outputscale": outputscale,
            "task_covariance": task_covariance,
            "rank": rank,
            "mean_constants": means,
            "standardize_mean": blocks[block_id]["standardize"]["mean"],
            "standardize_scale": blocks[block_id]["standardize"]["scale"],
        },
        "per_output": {},
    }
    # Pooled outputs: single-task fixed-noise fits (same transform).
    pooled_names = [name for name in table.output_names if name not in CELL_OUTPUTS]
    if pooled_names:
        pooled_table = TrainingTable(
            table.input_names, tuple(pooled_names), table.trials, table.rows, table.normaliser, table.physical, table.normalized
        )
        pooled = fit_botorch_stgp(pooled_table, transform, candidate_id, threads=threads, seed=seed)
        blocks.update(pooled.blocks)
        outputs.extend(pooled.outputs)
        native.update(pooled.native)
        diagnostics["per_output"].update(pooled.diagnostics["per_output"])
    diagnostics["fit_seconds"] = time.perf_counter() - started
    diagnostics["library"] = "botorch.MultiTaskGP(train_Yvar, rank=2) x Matern-5/2 ARD + SingleTaskGP for pooled outputs"
    diagnostics["torch_threads"] = torch.get_num_threads()
    return FittedCandidate(candidate_id, transform, blocks, outputs, diagnostics, native)


def native_latent_botorch(candidate: FittedCandidate, normalized: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    import torch  # noqa: WPS433

    spec = candidate.output_spec(name)
    x = torch.tensor(np.asarray(normalized, dtype=float), dtype=torch.float64)
    if name in candidate.native:
        model = candidate.native[name]
        with torch.no_grad():
            posterior = model.posterior(x)
        return posterior.mean.flatten().numpy().copy(), posterior.variance.flatten().numpy().copy()
    model = candidate.native["cells"]
    with torch.no_grad():
        posterior = model.posterior(x)
    task = int(spec["task"])
    return posterior.mean[:, task].numpy().copy(), posterior.variance[:, task].numpy().copy()


def native_latent(candidate: FittedCandidate, table: TrainingTable, physical: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if candidate.candidate_id.startswith("pkg-"):
        return native_latent_pkg(candidate, physical, name)
    return native_latent_botorch(candidate, table.normaliser.transform(physical), name)


def fit_candidate(candidate_id: str, table: TrainingTable, *, threads: int, seed: int) -> FittedCandidate:
    transform = TRANSFORM_OF[candidate_id]
    if candidate_id.startswith("pkg-exactgp"):
        fitted = fit_pkg_exactgp(table, transform, candidate_id)
    elif candidate_id.startswith("botorch-stgp"):
        fitted = fit_botorch_stgp(table, transform, candidate_id, threads=threads, seed=seed)
    elif candidate_id == "botorch-icm-logit":
        fitted = fit_botorch_icm(table, transform, candidate_id, threads=threads, seed=seed)
    else:
        raise ValueError(f"unknown candidate {candidate_id}")
    fitted.compile()
    return fitted


# --------------------------------------------------------------------------
# Baselines (P units)
# --------------------------------------------------------------------------


@dataclass
class Baseline:
    baseline_id: str
    parameters: dict[str, Any]
    _predict: Any = field(repr=False)

    def predict(self, normalized: np.ndarray, name: str) -> np.ndarray:
        return self._predict(np.asarray(normalized, dtype=float), name)


def fit_global_mean(table: TrainingTable) -> Baseline:
    means = {name: float(table.probabilities(name).mean()) for name in table.output_names}
    return Baseline("global-mean", {"means": means}, lambda x, name: np.full(x.shape[0], means[name]))


def fit_knn(table: TrainingTable, k: int = 3) -> Baseline:
    train = table.normalized
    ids = [row.case_id for row in table.rows]
    targets = {name: table.probabilities(name) for name in table.output_names}

    def predict(x: np.ndarray, name: str) -> np.ndarray:
        out = np.empty(x.shape[0])
        for index, point in enumerate(x):
            distances = np.sqrt(((train - point) ** 2).sum(axis=1))
            order = sorted(range(len(ids)), key=lambda i: (distances[i], ids[i]))[:k]
            out[index] = float(targets[name][order].mean())
        return out

    return Baseline("knn-3", {"k": k}, predict)


def fit_ridge(table: TrainingTable, penalty: float) -> Baseline:
    design = np.hstack([np.ones((table.normalized.shape[0], 1)), table.normalized])
    regulariser = penalty * np.eye(design.shape[1])
    regulariser[0, 0] = 0.0  # unpenalised intercept
    gram = design.T @ design + regulariser
    weights = {
        name: np.linalg.solve(gram, design.T @ table.probabilities(name)) for name in table.output_names
    }

    def predict(x: np.ndarray, name: str) -> np.ndarray:
        return np.hstack([np.ones((x.shape[0], 1)), x]) @ weights[name]

    return Baseline("ridge", {"penalty": penalty, "weights": {k: v.tolist() for k, v in weights.items()}}, predict)


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------


def rmse(errors: Sequence[float]) -> float:
    values = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def candidate_probabilities(candidate: FittedCandidate, normalized: np.ndarray, name: str) -> np.ndarray:
    from .data import working_to_probability

    mean, _ = candidate.latent(normalized, name)
    transform = candidate.output_spec(name)["transform"]
    return np.asarray([working_to_probability(float(m), transform) for m in mean], dtype=float)


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------


def permutation_importance(
    candidate: FittedCandidate,
    table: TrainingTable,
    output_names: Sequence[str],
    *,
    repeats: int,
    namespace: str,
) -> dict[str, Any]:
    """In-sample (fit-role) permutation importance in P units."""

    base = {name: table.probabilities(name) for name in output_names}
    baseline_rmse = {
        name: rmse(candidate_probabilities(candidate, table.normalized, name) - base[name]) for name in output_names
    }
    per_input: dict[str, dict[str, float]] = {}
    for column, input_name in enumerate(table.input_names):
        increases = {name: [] for name in output_names}
        for repeat in range(repeats):
            digest = hashlib.sha256(f"{namespace}:{input_name}:{repeat}".encode("utf-8")).digest()
            rng = Random(int.from_bytes(digest[:8], "big"))
            order = list(range(table.normalized.shape[0]))
            rng.shuffle(order)
            permuted = table.normalized.copy()
            permuted[:, column] = permuted[order, column]
            for name in output_names:
                value = rmse(candidate_probabilities(candidate, permuted, name) - base[name])
                increases[name].append(value - baseline_rmse[name])
        per_input[input_name] = {name: float(np.mean(increases[name])) for name in output_names}
        per_input[input_name]["mean_over_outputs"] = float(np.mean([per_input[input_name][n] for n in output_names]))
    ranking = sorted(table.input_names, key=lambda n: -per_input[n]["mean_over_outputs"])
    return {
        "role": "fit (in-sample)",
        "repeats": repeats,
        "baseline_rmse": baseline_rmse,
        "increase_by_input": per_input,
        "ranking": ranking,
    }


def ard_length_scales(candidate: FittedCandidate, input_names: Sequence[str]) -> dict[str, Any]:
    report = {}
    for model_id in sorted(candidate.blocks):
        block = candidate.blocks[model_id]
        scales = dict(zip(input_names, block["lengthscales"], strict=True))
        report[model_id] = {
            "outputs": block["outputs"],
            "length_scales_unit_box": scales,
            "most_sensitive_inputs": sorted(input_names, key=lambda n: scales[n])[:4],
        }
    return report


# --------------------------------------------------------------------------
# Active-learning add-on (reported, never gated)
# --------------------------------------------------------------------------


def active_learning_study(
    fit_rows: Sequence[DesignRow],
    evaluation_rows: Sequence[DesignRow],
    input_names: Sequence[str],
    output_names: Sequence[str],
    trials: Mapping[str, int],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Uncertainty sampling vs seeded random subsets, scored on the evaluation role."""

    pool_ids = sorted(row.case_id for row in fit_rows)
    by_id = {row.case_id: row for row in fit_rows}
    budgets = [int(b) for b in spec["budgets"]]
    initial = int(spec["initial_count"])
    seeds = [int(s) for s in spec["random_seeds"]]
    evaluation_physical = physical_matrix(evaluation_rows)
    truths = {
        name: np.asarray([row.counts[name][0] / row.counts[name][1] for row in evaluation_rows]) for name in output_names
    }

    def score(ids: Sequence[str]) -> dict[str, float]:
        table = TrainingTable.build([by_id[c] for c in ids], input_names, output_names, trials)
        fitted = fit_pkg_exactgp(table, "logit", spec["model"])
        fitted.compile()
        normalized = table.normaliser.transform(evaluation_physical)
        per_output = {
            name: rmse(candidate_probabilities(fitted, normalized, name) - truths[name]) for name in output_names
        }
        per_output["mean_over_outputs"] = float(np.mean([per_output[n] for n in output_names]))
        return per_output

    random_runs = []
    orders: dict[int, list[str]] = {}
    for seed in seeds:
        rng = Random(int.from_bytes(hashlib.sha256(f"{spec['model']}:al-random:{seed}".encode()).digest()[:8], "big"))
        order = pool_ids[:]
        rng.shuffle(order)
        orders[seed] = order
        random_runs.append(
            {"seed": seed, "curve": [{"budget": b, "rmse": score(order[:b]), "case_ids": sorted(order[:b])} for b in budgets]}
        )
    selected = orders[seeds[0]][:initial]
    curve = []
    remaining = [c for c in pool_ids if c not in selected]
    for budget in budgets:
        while len(selected) < budget:
            table = TrainingTable.build([by_id[c] for c in selected], input_names, output_names, trials)
            fitted = fit_pkg_exactgp(table, "logit", spec["model"])
            fitted.compile()
            candidates = table.normaliser.transform(physical_matrix([by_id[c] for c in remaining]))
            total_variance = np.zeros(len(remaining))
            for name in output_names:
                _, variance = fitted.latent(candidates, name)
                total_variance += variance
            best = sorted(range(len(remaining)), key=lambda i: (-total_variance[i], remaining[i]))[0]
            selected.append(remaining.pop(best))
        curve.append({"budget": budget, "rmse": score(selected), "case_ids": sorted(selected)})
    comparison = []
    for index, budget in enumerate(budgets):
        random_mean = float(np.mean([run["curve"][index]["rmse"]["mean_over_outputs"] for run in random_runs]))
        al_set = set(curve[index]["case_ids"])
        overlaps = [len(al_set & set(run["curve"][index]["case_ids"])) for run in random_runs]
        comparison.append(
            {
                "budget": budget,
                "active_rmse_mean_over_outputs": curve[index]["rmse"]["mean_over_outputs"],
                "random_rmse_mean_over_outputs_mean_of_seeds": random_mean,
                "active_better": curve[index]["rmse"]["mean_over_outputs"] < random_mean,
                "overlap_with_random_subsets": overlaps,
            }
        )
    return {
        "model": spec["model"],
        "pool_role": spec["pool_role"],
        "evaluation_role": spec["evaluation_role"],
        "pool_size": len(pool_ids),
        "initial_count": initial,
        "budgets": budgets,
        "acquisition": spec["acquisition"],
        "active": curve,
        "random": random_runs,
        "comparison": comparison,
        "active_better_at_every_budget": all(item["active_better"] for item in comparison),
        "gated": False,
    }
