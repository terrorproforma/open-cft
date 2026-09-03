"""Optimiser adapters sharing one recording evaluator.

Three strategies with an identical evaluation budget and identical initial
observations per seed:

* ``qlognehvi``  -- BoTorch constrained qLogNEHVI, independent exact GPs per
  output (four objectives + one constraint) on the declared torch device;
* ``nsga3``      -- pymoo NSGA-III with energy reference directions;
* ``lhs``        -- two-stage Latin-hypercube random baseline (stdlib RNG).

Failed (infeasible) evaluations never enter any front or hypervolume.  pymoo
requires finite objective placeholders for infeasible individuals; the
placeholders are the reference point (zero attained hypervolume) and are
never recorded as observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cft_revival.optimization.botorch_adapter import (
    ModelOutputLayout,
    build_qlognehvi,
    load_api,
    transform_model_output_values,
)

from . import model as m

STRATEGIES: tuple[str, ...] = ("qlognehvi", "nsga3", "lhs")


class BudgetExceededError(RuntimeError):
    """An optimiser requested more evaluations than the declared budget."""


# --------------------------------------------------------------------------
# Recording evaluator
# --------------------------------------------------------------------------


@dataclass
class RunLedger:
    """Ordered evaluation records for one (strategy, seed) run."""

    strategy: str
    seed: int
    budget: int
    sample: tuple[Mapping[str, float], ...]
    nominal: Mapping[str, float]
    tail_fraction: float
    records: list[dict[str, Any]] = field(default_factory=list)
    hypervolume_curve: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    checkpoint: Callable[["RunLedger"], None] | None = None

    def evaluate(
        self, values: Sequence[float], *, batch: int, provenance: str
    ) -> m.DesignEvaluation:
        if len(self.records) >= self.budget:
            raise BudgetExceededError(
                f"{self.strategy} seed {self.seed}: budget {self.budget} exhausted"
            )
        tick = time.perf_counter()
        evaluation = m.evaluate_design(
            values, self.sample, nominal=self.nominal, tail_fraction=self.tail_fraction
        )
        record = evaluation.to_record()
        record.update(
            {
                "index": len(self.records),
                "batch": int(batch),
                "provenance": provenance,
                "evaluation_seconds": time.perf_counter() - tick,
            }
        )
        self.records.append(record)
        self._append_hypervolume()
        return evaluation

    # -- objective-space bookkeeping ------------------------------------

    def feasible_points(self) -> list[tuple[int, tuple[float, ...]]]:
        return [
            (
                record["index"],
                m.normalized_objectives(
                    [record["robust_objectives"][name] for name in m.OBJECTIVE_NAMES]
                ),
            )
            for record in self.records
            if record["status"] == "success"
        ]

    def _append_hypervolume(self) -> None:
        points = [point for _index, point in self.feasible_points()]
        self.hypervolume_curve.append(
            {
                "evaluations": len(self.records),
                "feasible": len(points),
                "hypervolume": m.hypervolume(points),
                "elapsed_seconds": time.perf_counter() - self.started_at,
            }
        )
        if self.checkpoint is not None:
            self.checkpoint(self)

    def summary(self) -> dict[str, Any]:
        points = self.feasible_points()
        vectors = [point for _index, point in points]
        front = m.nondominated_indices(vectors)
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "budget": self.budget,
            "evaluations": len(self.records),
            "feasible_evaluations": len(points),
            "infeasible_evaluations": sum(
                1 for record in self.records if record["status"] != "success"
            ),
            "final_hypervolume": (
                self.hypervolume_curve[-1]["hypervolume"] if self.hypervolume_curve else 0.0
            ),
            "pareto_set_size": len(front),
            "pareto_record_indices": [points[index][0] for index in front],
            "wall_clock_seconds": time.perf_counter() - self.started_at,
        }


def lhs_rows(count: int, rng: Random, dimensions: int = 3) -> np.ndarray:
    """One Latin-hypercube design in the unit cube drawn from ``rng`` (stdlib only).

    Each dimension is an independent random permutation of ``count`` equal
    strata with a uniform jitter inside the stratum.  ``random.Random`` gives a
    version-stable Mersenne-Twister stream, so the design is reproducible
    without scipy or torch.
    """

    if count < 1 or dimensions < 1:
        raise ValueError("LHS requires positive count and dimensions")
    columns = []
    for _dimension in range(dimensions):
        strata = list(range(count))
        rng.shuffle(strata)
        columns.append([(stratum + rng.random()) / count for stratum in strata])
    return np.asarray(columns, dtype=float).T


def shared_initial_points(seed: int, count: int, dimensions: int = 3) -> np.ndarray:
    """LHS points in the unit cube shared by all strategies of a seed."""

    return lhs_rows(count, Random(seed), dimensions)


def lhs_points(seed: int, count: int, initial_count: int, dimensions: int = 3) -> np.ndarray:
    """Two-stage LHS baseline: the shared initial design, then one LHS of the rest.

    Both stages are drawn from the same seeded stream so the first
    ``initial_count`` rows equal ``shared_initial_points(seed, initial_count)``.
    """

    if not 0 < initial_count <= count:
        raise ValueError("initial_count must lie in (0, count]")
    rng = Random(seed)
    first = lhs_rows(initial_count, rng, dimensions)
    if count == initial_count:
        return first
    rest = lhs_rows(count - initial_count, rng, dimensions)
    return np.vstack((first, rest))


def denormalize(unit: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        variable.lower + float(coordinate) * (variable.upper - variable.lower)
        for coordinate, variable in zip(unit, m.DESIGN_VARIABLES, strict=True)
    )


def _clip_unit(unit: Sequence[float]) -> tuple[float, ...]:
    return tuple(min(1.0, max(0.0, float(coordinate))) for coordinate in unit)


# --------------------------------------------------------------------------
# Strategy: LHS random baseline
# --------------------------------------------------------------------------


def run_lhs(ledger: RunLedger, *, initial_count: int) -> None:
    points = lhs_points(ledger.seed, ledger.budget, initial_count)
    for index, unit in enumerate(points):
        batch = 0 if index < initial_count else 1 + (index - initial_count)
        ledger.evaluate(
            denormalize(unit),
            batch=batch,
            provenance=f"lhs:seed={ledger.seed}:index={index}",
        )


# --------------------------------------------------------------------------
# Strategy: pymoo NSGA-III
# --------------------------------------------------------------------------


def run_nsga3(
    ledger: RunLedger,
    *,
    initial_count: int,
    population_size: int,
    generations: int,
    reference_direction_seed: int,
) -> dict[str, Any]:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    if population_size != initial_count:
        raise ValueError("NSGA-III population must equal the shared initial design size")
    if population_size * generations != ledger.budget:
        raise ValueError("NSGA-III population * generations must equal the budget")
    placeholder = np.zeros(len(m.OBJECTIVES))  # reference point in the maximise frame
    generation = {"value": 0}

    class RobustL0Problem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=len(m.DESIGN_VARIABLES),
                n_obj=len(m.OBJECTIVES),
                n_ieq_constr=1,
                xl=np.zeros(len(m.DESIGN_VARIABLES)),
                xu=np.ones(len(m.DESIGN_VARIABLES)),
            )

        def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            objectives = []
            constraints = []
            for row_index, row in enumerate(x):
                evaluation = ledger.evaluate(
                    denormalize(_clip_unit(row)),
                    batch=generation["value"],
                    provenance=(
                        f"nsga3:seed={ledger.seed}:generation={generation['value']}"
                        f":member={row_index}"
                    ),
                )
                # pymoo minimises: negate the all-maximise normalized frame.
                if evaluation.status == "success":
                    objectives.append(
                        -np.asarray(m.normalized_objectives(evaluation.robust_objectives))
                    )
                else:
                    objectives.append(-placeholder)
                constraints.append([-evaluation.robust_margin_a / m.ROBUST_CONSTRAINT.violation_scale])
            generation["value"] += 1
            out["F"] = np.vstack(objectives)
            out["G"] = np.asarray(constraints)

    reference_directions = get_reference_directions(
        "energy", len(m.OBJECTIVES), population_size, seed=reference_direction_seed
    )
    initial = shared_initial_points(ledger.seed, initial_count)
    algorithm = NSGA3(
        ref_dirs=reference_directions,
        pop_size=population_size,
        sampling=initial,
        eliminate_duplicates=False,
    )
    result = minimize(
        RobustL0Problem(),
        algorithm,
        ("n_gen", generations),
        seed=ledger.seed,
        verbose=False,
    )
    return {
        "reference_direction_count": int(reference_directions.shape[0]),
        "generations_completed": int(result.algorithm.n_gen),
        "pymoo_reported_evaluations": int(result.algorithm.evaluator.n_eval),
    }


# --------------------------------------------------------------------------
# Strategy: BoTorch constrained qLogNEHVI
# --------------------------------------------------------------------------


def model_layout() -> ModelOutputLayout:
    """Objectives scaled by their comparison scale, then the robust constraint."""

    from cft_revival.optimization import ObjectiveSpec

    scaled = tuple(
        ObjectiveSpec(
            objective.name,
            objective.direction,
            objective.units,
            1.0,
            objective.absolute_tolerance,
            objective.relative_tolerance,
        )
        for objective in m.OBJECTIVES
    )
    return ModelOutputLayout(objectives=scaled, constraints=(m.ROBUST_CONSTRAINT,))


def _scaled_physical(vector: Sequence[float]) -> list[float]:
    return [
        float(value) / objective.comparison_scale
        for value, objective in zip(vector, m.OBJECTIVES, strict=True)
    ]


def torch_environment(device: str) -> dict[str, Any]:
    """Prove the declared device with a real float64 operation; fail closed."""

    api = load_api()
    torch = api["torch"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("declared CUDA device is unavailable")
    resolved = torch.device(device)
    probe = torch.linalg.cholesky(
        torch.tensor([[4.0, 2.0], [2.0, 3.0]], dtype=torch.float64, device=resolved)
    )
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    report = {
        "device": str(resolved),
        "device_name": (
            torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "cpu"
        ),
        "float64_cholesky_probe": [float(value) for value in probe.flatten().tolist()],
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }
    return report


def run_qlognehvi(
    ledger: RunLedger,
    *,
    initial_count: int,
    batch_size: int,
    device: str,
    num_restarts: int,
    raw_samples: int,
    mc_samples: int,
    fit_noise_floor: float,
    sequential: bool = True,
    maxiter: int = 200,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    api = load_api()
    torch = api["torch"]
    Standardize = getattr(__import__("botorch.models.transforms.outcome", fromlist=["Standardize"]), "Standardize")
    SobolQMCNormalSampler = getattr(
        __import__("botorch.sampling.normal", fromlist=["SobolQMCNormalSampler"]),
        "SobolQMCNormalSampler",
    )
    GreaterThan = getattr(__import__("gpytorch.constraints", fromlist=["GreaterThan"]), "GreaterThan")
    GaussianLikelihood = getattr(
        __import__("gpytorch.likelihoods", fromlist=["GaussianLikelihood"]), "GaussianLikelihood"
    )

    layout = model_layout()
    tensor_device = torch.device(device)
    dtype = torch.float64
    torch.manual_seed(ledger.seed)
    if tensor_device.type == "cuda":
        torch.cuda.manual_seed_all(ledger.seed)
    bounds = torch.tensor(
        [[0.0] * len(m.DESIGN_VARIABLES), [1.0] * len(m.DESIGN_VARIABLES)],
        dtype=dtype,
        device=tensor_device,
    )
    reference = torch.tensor(
        _scaled_physical([m.REFERENCE_POINT[name] for name in m.OBJECTIVE_NAMES]),
        dtype=dtype,
        device=tensor_device,
    )

    unit_rows: list[tuple[float, ...]] = []
    transformed_rows: list[tuple[float, ...] | None] = []
    constraint_rows: list[float] = []

    def record(unit: Sequence[float], evaluation: m.DesignEvaluation) -> None:
        unit_rows.append(tuple(float(value) for value in unit))
        margin = evaluation.robust_margin_a
        if evaluation.status == "success":
            values = transform_model_output_values(
                [*_scaled_physical(evaluation.robust_objectives), margin], layout
            )
            transformed_rows.append(values[: len(m.OBJECTIVES)])
            constraint_rows.append(values[len(m.OBJECTIVES)])
        else:
            transformed_rows.append(None)
            constraint_rows.append(
                transform_model_output_values(
                    [*([0.0] * len(m.OBJECTIVES)), margin], layout
                )[len(m.OBJECTIVES)]
            )

    initial = shared_initial_points(ledger.seed, initial_count)
    for index, unit in enumerate(initial):
        record(
            unit,
            ledger.evaluate(
                denormalize(unit),
                batch=0,
                provenance=f"qlognehvi:seed={ledger.seed}:initial:index={index}",
            ),
        )

    iteration_log: list[dict[str, Any]] = []
    iteration = 0
    while len(ledger.records) < ledger.budget:
        iteration += 1
        q = min(batch_size, ledger.budget - len(ledger.records))
        tick = time.perf_counter()
        train_x = torch.tensor(unit_rows, dtype=dtype, device=tensor_device)
        feasible_mask = [row is not None for row in transformed_rows]
        if sum(feasible_mask) < 2:
            raise RuntimeError("qLogNEHVI requires at least two feasible observations")
        feasible_x = train_x[torch.tensor(feasible_mask, device=tensor_device)]
        feasible_y = torch.tensor(
            [row for row in transformed_rows if row is not None],
            dtype=dtype,
            device=tensor_device,
        )
        constraint_y = torch.tensor(constraint_rows, dtype=dtype, device=tensor_device).unsqueeze(-1)
        models = []
        for column in range(feasible_y.shape[-1]):
            models.append(
                _fit_single_task(
                    api,
                    feasible_x,
                    feasible_y[..., column : column + 1],
                    Standardize,
                    GaussianLikelihood,
                    GreaterThan,
                    fit_noise_floor,
                )
            )
        models.append(
            _fit_single_task(
                api,
                train_x,
                constraint_y,
                Standardize,
                GaussianLikelihood,
                GreaterThan,
                fit_noise_floor,
            )
        )
        model = api["ModelListGP"](*models)
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        acquisition = build_qlognehvi(
            model,
            reference,
            train_x,
            layout,
            model_outputs_are_direction_transformed=True,
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([mc_samples]), seed=ledger.seed * 1000 + iteration
            ),
        )
        candidates, acquisition_value = api["optimize_acqf"](
            acq_function=acquisition,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": maxiter, "seed": ledger.seed * 1000 + iteration},
            sequential=sequential,
        )
        acquisition_seconds = time.perf_counter() - tick
        candidate_rows = candidates.detach().cpu().tolist()
        for member, unit in enumerate(candidate_rows):
            clipped = _clip_unit(unit)
            record(
                clipped,
                ledger.evaluate(
                    denormalize(clipped),
                    batch=iteration,
                    provenance=(
                        f"qlognehvi:seed={ledger.seed}:iteration={iteration}:member={member}"
                    ),
                ),
            )
        entry = {
            "iteration": iteration,
            "training_points": int(train_x.shape[0]),
            "feasible_training_points": int(feasible_x.shape[0]),
            "batch_size": q,
            "fit_seconds": fit_seconds,
            "acquisition_seconds": acquisition_seconds,
            "acquisition_value": float(acquisition_value.detach().max().cpu()),
            "evaluations": len(ledger.records),
            "hypervolume": ledger.hypervolume_curve[-1]["hypervolume"],
        }
        iteration_log.append(entry)
        if progress is not None:
            progress(entry)
    return {
        "iterations": iteration,
        "iteration_log": iteration_log,
        "device": str(tensor_device),
        "model": "independent SingleTaskGP per output (Matern-5/2 ARD, Standardize outcome transform, inferred noise with floor)",
        "acquisition": "qLogNoisyExpectedHypervolumeImprovement, prune_baseline, sequential greedy batch",
    }


def _fit_single_task(
    api: Mapping[str, Any],
    train_x: Any,
    train_y: Any,
    Standardize: Any,
    GaussianLikelihood: Any,
    GreaterThan: Any,
    noise_floor: float,
) -> Any:
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise_floor))
    model = api["SingleTaskGP"](
        train_x,
        train_y,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = api["ExactMarginalLogLikelihood"](model.likelihood, model)
    api["fit_gpytorch_mll"](mll)
    return model
