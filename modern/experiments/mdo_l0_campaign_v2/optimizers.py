"""Optimiser adapters over the mixed design space (catalogue index x operating point).

Three strategies with an identical evaluation budget and identical initial observations
per seed:

* ``qlognehvi``  -- BoTorch constrained qLogNEHVI over a ``ModelListGP`` of
  ``MixedSingleTaskGP`` models (categorical kernel on the catalogue index, Matern-5/2 ARD
  on the unit-cube operating point), candidate stage over the whole catalogue then a
  per-member continuous refinement; every descriptive label is built from the arguments
  actually passed (v1 audit F28);
* ``nsga3``      -- pymoo NSGA-III with mixed-variable sampling/mating and duplicate
  elimination (v1 audit F27);
* ``lhs``        -- two-stage Latin hypercube over the catalogue and the operating point
  (stdlib RNG).

Failed (infeasible) evaluations never enter any front or hypervolume.  pymoo requires
finite objective placeholders for infeasible individuals; the placeholders are the
reference point (zero attained hypervolume) and are never recorded as observations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from importlib import import_module
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
UNIT_DIMENSIONS = 1 + len(m.DESIGN_VARIABLES)  # catalogue coordinate + operating point
CATALOGUE_UNIT_NAME = "catalogue_unit"
UNIT_NAMES: tuple[str, ...] = (CATALOGUE_UNIT_NAME, *m.CONTINUOUS_NAMES)


class BudgetExceededError(RuntimeError):
    """An optimiser requested more evaluations than the declared budget."""


# --------------------------------------------------------------------------
# Unit cube <-> mixed design
# --------------------------------------------------------------------------


def catalogue_index_from_unit(coordinate: float) -> int:
    """Map a unit coordinate to a catalogue index (equal-width strata, upper end inclusive)."""

    value = float(coordinate)
    if not (0.0 <= value <= 1.0):
        raise ValueError("catalogue unit coordinate must lie in [0, 1]")
    return min(m.CATALOGUE_SIZE - 1, int(math.floor(value * m.CATALOGUE_SIZE)))


def denormalize(unit: Sequence[float]) -> tuple[float, ...]:
    if len(unit) != len(m.DESIGN_VARIABLES):
        raise ValueError("operating-point unit vector must have three coordinates")
    return tuple(
        variable.lower + float(coordinate) * (variable.upper - variable.lower)
        for coordinate, variable in zip(unit, m.DESIGN_VARIABLES, strict=True)
    )


def _clip_unit(unit: Sequence[float]) -> tuple[float, ...]:
    return tuple(min(1.0, max(0.0, float(coordinate))) for coordinate in unit)


def unit_to_design(unit: Sequence[float]) -> tuple[int, tuple[float, ...]]:
    """(catalogue index, physical operating point) of a 4-dimensional unit row."""

    if len(unit) != UNIT_DIMENSIONS:
        raise ValueError("unit row must have four coordinates")
    clipped = _clip_unit(unit)
    return catalogue_index_from_unit(clipped[0]), denormalize(clipped[1:])


# --------------------------------------------------------------------------
# Recording evaluator
# --------------------------------------------------------------------------


@dataclass
class RunLedger:
    """Ordered evaluation records for one (strategy, seed) run."""

    strategy: str
    seed: int
    budget: int
    context: m.EvaluationContext
    records: list[dict[str, Any]] = field(default_factory=list)
    hypervolume_curve: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    checkpoint: Callable[["RunLedger"], None] | None = None

    def evaluate(self, index: int, values: Sequence[float], *, batch: int, provenance: str) -> m.DesignEvaluation:
        if len(self.records) >= self.budget:
            raise BudgetExceededError(f"{self.strategy} seed {self.seed}: budget {self.budget} exhausted")
        tick = time.perf_counter()
        evaluation = m.evaluate_design(index, values, self.context)
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
            (record["index"], m.normalized_objectives([record["robust_objectives"][name] for name in m.OBJECTIVE_NAMES]))
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
            "unique_designs": len({record["design"]["design_id"] for record in self.records}),
            "distinct_catalogue_designs": len({record["design"]["catalogue_index"] for record in self.records}),
            "feasible_evaluations": len(points),
            "infeasible_evaluations": sum(1 for record in self.records if record["status"] != "success"),
            "final_hypervolume": self.hypervolume_curve[-1]["hypervolume"] if self.hypervolume_curve else 0.0,
            "pareto_set_size": len(front),
            "pareto_record_indices": [points[index][0] for index in front],
            "pareto_catalogue_indices": sorted(
                {self.records[points[index][0]]["design"]["catalogue_index"] for index in front}
            ),
            "wall_clock_seconds": time.perf_counter() - self.started_at,
        }


def lhs_rows(count: int, rng: Random, dimensions: int = UNIT_DIMENSIONS) -> np.ndarray:
    """One Latin-hypercube design in the unit cube drawn from ``rng`` (stdlib only; v1 construction)."""

    if count < 1 or dimensions < 1:
        raise ValueError("LHS requires positive count and dimensions")
    columns = []
    for _dimension in range(dimensions):
        strata = list(range(count))
        rng.shuffle(strata)
        columns.append([(stratum + rng.random()) / count for stratum in strata])
    return np.asarray(columns, dtype=float).T


def shared_initial_points(seed: int, count: int) -> np.ndarray:
    """Unit rows (catalogue coordinate + operating point) shared by all strategies of a seed."""

    return lhs_rows(count, Random(seed))


def lhs_points(seed: int, count: int, initial_count: int) -> np.ndarray:
    """Two-stage LHS baseline: the shared initial design, then one LHS of the rest."""

    if not 0 < initial_count <= count:
        raise ValueError("initial_count must lie in (0, count]")
    rng = Random(seed)
    first = lhs_rows(initial_count, rng)
    if count == initial_count:
        return first
    rest = lhs_rows(count - initial_count, rng)
    return np.vstack((first, rest))


# --------------------------------------------------------------------------
# Strategy: LHS random baseline over the catalogue x operating point
# --------------------------------------------------------------------------


def run_lhs(ledger: RunLedger, *, initial_count: int) -> dict[str, Any]:
    points = lhs_points(ledger.seed, ledger.budget, initial_count)
    for row_index, unit in enumerate(points):
        batch = 0 if row_index < initial_count else 1 + (row_index - initial_count)
        index, values = unit_to_design(unit)
        ledger.evaluate(index, values, batch=batch, provenance=f"lhs:seed={ledger.seed}:index={row_index}")
    return {
        "points": int(ledger.budget),
        "stages": [int(initial_count), int(ledger.budget - initial_count)],
        "design": (
            f"two-stage Latin hypercube in the 4-dimensional unit cube (stdlib random.Random({ledger.seed})): "
            f"stage 1 = the shared {initial_count}-point initial design, stage 2 = one {ledger.budget - initial_count}-point LHS "
            f"from the continuation of the same stream; the first coordinate maps to the catalogue index by "
            f"floor(u * {m.CATALOGUE_SIZE}), the other three to the operating point"
        ),
    }


# --------------------------------------------------------------------------
# Strategy: pymoo NSGA-III with mixed variables and duplicate elimination
# --------------------------------------------------------------------------

PYMOO_VARIABLE_NAMES: tuple[str, ...] = (m.CATALOGUE_VARIABLE, *m.CONTINUOUS_NAMES)


def pymoo_individual(unit: Sequence[float]) -> dict[str, Any]:
    """The mixed-variable dictionary pymoo evaluates for one unit row (catalogue index + unit operating point)."""

    clipped = _clip_unit(unit)
    individual: dict[str, Any] = {m.CATALOGUE_VARIABLE: catalogue_index_from_unit(clipped[0])}
    for name, coordinate in zip(m.CONTINUOUS_NAMES, clipped[1:], strict=True):
        individual[name] = float(coordinate)
    return individual


def run_nsga3(
    ledger: RunLedger,
    *,
    initial_count: int,
    population_size: int,
    generations: int,
    reference_direction_seed: int,
) -> dict[str, Any]:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableMating
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.core.variable import Choice, Real
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    if population_size != initial_count:
        raise ValueError("NSGA-III population must equal the shared initial design size")
    if population_size * generations != ledger.budget:
        raise ValueError("NSGA-III population * generations must equal the budget")
    placeholder = np.zeros(len(m.OBJECTIVES))  # reference point in the maximise frame
    generation = {"value": 0}

    class RobustCatalogueProblem(Problem):
        def __init__(self) -> None:
            variables: dict[str, Any] = {m.CATALOGUE_VARIABLE: Choice(options=list(range(m.CATALOGUE_SIZE)))}
            for name in m.CONTINUOUS_NAMES:
                variables[name] = Real(bounds=(0.0, 1.0))
            super().__init__(vars=variables, n_obj=len(m.OBJECTIVES), n_ieq_constr=1)

        def _evaluate(self, x: Any, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            objectives = []
            constraints = []
            for member, individual in enumerate(x):
                index = int(individual[m.CATALOGUE_VARIABLE])
                unit = _clip_unit([float(individual[name]) for name in m.CONTINUOUS_NAMES])
                evaluation = ledger.evaluate(
                    index,
                    denormalize(unit),
                    batch=generation["value"],
                    provenance=f"nsga3:seed={ledger.seed}:generation={generation['value']}:member={member}",
                )
                # pymoo minimises: negate the all-maximise normalized frame.
                if evaluation.status == "success":
                    objectives.append(-np.asarray(m.normalized_objectives(evaluation.robust_objectives)))
                else:
                    objectives.append(-placeholder)
                constraints.append([-evaluation.robust_margin_a / m.ROBUST_CONSTRAINT.violation_scale])
            generation["value"] += 1
            out["F"] = np.vstack(objectives)
            out["G"] = np.asarray(constraints)

    reference_directions = get_reference_directions(
        "energy", len(m.OBJECTIVES), population_size, seed=reference_direction_seed
    )
    initial = Population.new("X", [pymoo_individual(unit) for unit in shared_initial_points(ledger.seed, initial_count)])
    duplicate_elimination = MixedVariableDuplicateElimination()
    algorithm = NSGA3(
        ref_dirs=reference_directions,
        pop_size=population_size,
        sampling=initial,
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=duplicate_elimination,
    )
    result = minimize(RobustCatalogueProblem(), algorithm, ("n_gen", generations), seed=ledger.seed, verbose=False)
    return {
        "reference_direction_count": int(reference_directions.shape[0]),
        "reference_direction_seed": int(reference_direction_seed),
        "declared_generations": int(generations),
        "pymoo_n_gen": int(result.algorithm.n_gen),
        "pymoo_n_gen_note": "pymoo's post-incremented generation counter (declared_generations + 1); the budget is population_size * declared_generations",
        "pymoo_reported_evaluations": int(result.algorithm.evaluator.n_eval),
        "eliminate_duplicates": True,
        "eliminate_duplicates_implementation": {
            "population": type(duplicate_elimination).__name__,
            "mating": type(algorithm.mating).__name__ + "(eliminate_duplicates=MixedVariableDuplicateElimination)",
        },
        "variables": {
            m.CATALOGUE_VARIABLE: f"Choice(options=range({m.CATALOGUE_SIZE}))",
            **{name: "Real(bounds=(0, 1))" for name in m.CONTINUOUS_NAMES},
        },
        "unique_designs": len({record["design"]["design_id"] for record in ledger.records}),
    }


# --------------------------------------------------------------------------
# Strategy: BoTorch constrained qLogNEHVI over MixedSingleTaskGP models
# --------------------------------------------------------------------------


def model_layout() -> ModelOutputLayout:
    """Objectives scaled by their comparison scale, then the robust constraint."""

    from cft_revival.optimization import ObjectiveSpec

    scaled = tuple(
        ObjectiveSpec(
            objective.name, objective.direction, objective.units, 1.0, objective.absolute_tolerance, objective.relative_tolerance
        )
        for objective in m.OBJECTIVES
    )
    return ModelOutputLayout(objectives=scaled, constraints=(m.ROBUST_CONSTRAINT,))


def _scaled_physical(vector: Sequence[float]) -> list[float]:
    return [float(value) / objective.comparison_scale for value, objective in zip(vector, m.OBJECTIVES, strict=True)]


def torch_environment(device: str) -> dict[str, Any]:
    """Prove the declared device with a real float64 operation; fail closed."""

    api = load_api()
    torch = api["torch"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("declared CUDA device is unavailable")
    resolved = torch.device(device)
    probe = torch.linalg.cholesky(torch.tensor([[4.0, 2.0], [2.0, 3.0]], dtype=torch.float64, device=resolved))
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    return {
        "device": str(resolved),
        "device_name": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "cpu",
        "float64_cholesky_probe": [float(value) for value in probe.flatten().tolist()],
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "torch_threads": int(torch.get_num_threads()),
    }


def acquisition_label(
    *,
    q: int,
    mc_samples: int,
    candidates_per_design: int,
    refine_maxiter: int,
    refine_num_restarts: int,
    sequential_candidate_stage: bool,
) -> str:
    """Descriptive label built from the acquisition arguments actually used (v1 audit F28)."""

    total = m.CATALOGUE_SIZE * candidates_per_design
    return (
        f"qLogNoisyExpectedHypervolumeImprovement (prune_baseline, cache_pending, SobolQMCNormalSampler {mc_samples} samples); "
        f"candidate stage: optimize_acqf_discrete over all {m.CATALOGUE_SIZE} catalogue designs x {candidates_per_design} "
        f"LHS operating points ({total} candidates), q={q} {'sequential greedy' if sequential_candidate_stage else 'joint'} (unique); "
        f"refinement stage: per member optimize_acqf L-BFGS-B (num_restarts {refine_num_restarts}, fixed catalogue feature, "
        f"other members pending, maxiter {refine_maxiter}), refined point accepted iff its acquisition value is not lower"
    )


def model_label(model: Any, *, noise_floor: float) -> str:
    """Descriptive label read from the fitted model objects (kernel classes), not hard-coded."""

    first = model.models[0]
    covar = first.covar_module

    def kernel_names(kernel: Any) -> list[str]:
        names = [type(kernel).__name__]
        for child in kernel.children():
            names.extend(kernel_names(child))
        return names

    seen = sorted(set(kernel_names(covar)))
    return (
        f"ModelListGP of {len(model.models)} {type(first).__name__} (cat_dims=[0] catalogue index; kernels {', '.join(seen)}; "
        f"outcome transform {type(first.outcome_transform).__name__}; {type(first.likelihood).__name__} with noise floor {noise_floor}); "
        "objectives fitted on feasible observations only, constraint on all observations"
    )


def run_qlognehvi(
    ledger: RunLedger,
    *,
    initial_count: int,
    batch_size: int,
    device: str,
    torch_threads: int,
    mc_samples: int,
    candidates_per_design: int,
    refine_maxiter: int,
    fit_noise_floor: float,
    refine_num_restarts: int = 1,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    api = load_api()
    torch = api["torch"]
    torch.set_num_threads(int(torch_threads))
    MixedSingleTaskGP = getattr(import_module("botorch.models.gp_regression_mixed"), "MixedSingleTaskGP")
    Standardize = getattr(import_module("botorch.models.transforms.outcome"), "Standardize")
    SobolQMCNormalSampler = getattr(import_module("botorch.sampling.normal"), "SobolQMCNormalSampler")
    GreaterThan = getattr(import_module("gpytorch.constraints"), "GreaterThan")
    GaussianLikelihood = getattr(import_module("gpytorch.likelihoods"), "GaussianLikelihood")
    optimize_acqf_discrete = getattr(import_module("botorch.optim"), "optimize_acqf_discrete")
    dim_scaled = getattr(import_module("botorch.models.utils.gpytorch_modules"), "get_covar_module_with_dim_scaled_prior")

    def matern_factory(batch_shape: Any, ard_num_dims: int, active_dims: Any) -> Any:
        return dim_scaled(ard_num_dims=ard_num_dims, batch_shape=batch_shape, use_rbf_kernel=False, active_dims=active_dims)

    layout = model_layout()
    tensor_device = torch.device(device)
    dtype = torch.float64
    torch.manual_seed(ledger.seed)
    if tensor_device.type == "cuda":
        torch.cuda.manual_seed_all(ledger.seed)
    bounds = torch.tensor(
        [[0.0] * UNIT_DIMENSIONS, [float(m.CATALOGUE_SIZE - 1), 1.0, 1.0, 1.0]], dtype=dtype, device=tensor_device
    )
    reference = torch.tensor(
        _scaled_physical([m.REFERENCE_POINT[name] for name in m.OBJECTIVE_NAMES]), dtype=dtype, device=tensor_device
    )
    categories = torch.arange(m.CATALOGUE_SIZE, dtype=dtype, device=tensor_device)

    model_rows: list[tuple[float, ...]] = []  # [catalogue index, u_Ua, u_Ia, u_mdot]
    transformed_rows: list[tuple[float, ...] | None] = []
    constraint_rows: list[float] = []

    def record(index: int, unit: Sequence[float], evaluation: m.DesignEvaluation) -> None:
        model_rows.append((float(index), *[float(value) for value in unit]))
        margin = evaluation.robust_margin_a
        if evaluation.status == "success":
            values = transform_model_output_values([*_scaled_physical(evaluation.robust_objectives), margin], layout)
            transformed_rows.append(values[: len(m.OBJECTIVES)])
            constraint_rows.append(values[len(m.OBJECTIVES)])
        else:
            transformed_rows.append(None)
            constraint_rows.append(
                transform_model_output_values([*([0.0] * len(m.OBJECTIVES)), margin], layout)[len(m.OBJECTIVES)]
            )

    initial = shared_initial_points(ledger.seed, initial_count)
    for row_index, unit in enumerate(initial):
        index, values = unit_to_design(unit)
        record(
            index,
            _clip_unit(unit[1:]),
            ledger.evaluate(index, values, batch=0, provenance=f"qlognehvi:seed={ledger.seed}:initial:index={row_index}"),
        )

    def fit(train_x: Any, train_y: Any) -> Any:
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(fit_noise_floor))
        gp = MixedSingleTaskGP(
            train_x,
            train_y,
            cat_dims=[0],
            cont_kernel_factory=matern_factory,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        mll = api["ExactMarginalLogLikelihood"](gp.likelihood, gp)
        api["fit_gpytorch_mll"](mll)
        return gp

    iteration_log: list[dict[str, Any]] = []
    iteration = 0
    fitted_label: str | None = None
    while len(ledger.records) < ledger.budget:
        iteration += 1
        q = min(batch_size, ledger.budget - len(ledger.records))
        tick = time.perf_counter()
        train_x = torch.tensor(model_rows, dtype=dtype, device=tensor_device)
        feasible_mask = [row is not None for row in transformed_rows]
        if sum(feasible_mask) < 2:
            raise RuntimeError("qLogNEHVI requires at least two feasible observations")
        feasible_x = train_x[torch.tensor(feasible_mask, device=tensor_device)]
        feasible_y = torch.tensor([row for row in transformed_rows if row is not None], dtype=dtype, device=tensor_device)
        constraint_y = torch.tensor(constraint_rows, dtype=dtype, device=tensor_device).unsqueeze(-1)
        models = [fit(feasible_x, feasible_y[..., column : column + 1]) for column in range(feasible_y.shape[-1])]
        models.append(fit(train_x, constraint_y))
        model = api["ModelListGP"](*models)
        if fitted_label is None:
            fitted_label = model_label(model, noise_floor=fit_noise_floor)
        fit_seconds = time.perf_counter() - tick

        tick = time.perf_counter()
        acquisition = build_qlognehvi(
            model,
            reference,
            train_x,
            layout,
            model_outputs_are_direction_transformed=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=ledger.seed * 1000 + iteration),
        )
        # Candidate stage: every catalogue design x a fresh LHS of operating points.
        continuous = torch.tensor(
            lhs_rows(candidates_per_design, Random(ledger.seed * 1000 + iteration), len(m.DESIGN_VARIABLES)),
            dtype=dtype,
            device=tensor_device,
        )
        choices = torch.cat(
            [categories.repeat_interleave(candidates_per_design).unsqueeze(-1), continuous.repeat(m.CATALOGUE_SIZE, 1)],
            dim=-1,
        )
        discrete, discrete_values = optimize_acqf_discrete(
            acq_function=acquisition, q=q, choices=choices, max_batch_size=1024, unique=True
        )
        candidate_stage_seconds = time.perf_counter() - tick

        # Refinement stage: per member, continuous L-BFGS-B with the catalogue feature fixed
        # and every other member pending; accepted only if the acquisition does not decrease.
        tick = time.perf_counter()
        refined: list[Any] = []
        refinement_log = []
        for member in range(q):
            others = [*refined, *[discrete[j] for j in range(member + 1, q)]]
            acquisition.set_X_pending(torch.stack(others) if others else None)
            with torch.no_grad():
                start_value = float(acquisition(discrete[member : member + 1].unsqueeze(0)).item())
            candidate, refined_value = api["optimize_acqf"](
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                num_restarts=refine_num_restarts,
                raw_samples=1,
                fixed_features={0: float(discrete[member, 0])},
                batch_initial_conditions=discrete[member : member + 1].unsqueeze(0),
                options={"maxiter": refine_maxiter, "batch_limit": 1, "seed": ledger.seed * 1000 + iteration},
                sequential=False,
            )
            refined_value_f = float(refined_value.detach().max().item())
            accepted = refined_value_f >= start_value
            chosen = candidate.squeeze(0).detach() if accepted else discrete[member].detach()
            chosen = chosen.clone()
            chosen[0] = discrete[member, 0]
            refined.append(chosen)
            refinement_log.append(
                {
                    "member": member,
                    "catalogue_index": int(round(float(discrete[member, 0]))),
                    "start_value": start_value,
                    "refined_value": refined_value_f,
                    "accepted": accepted,
                }
            )
        acquisition.set_X_pending(None)
        refinement_seconds = time.perf_counter() - tick

        for member, row in enumerate(refined):
            unit = _clip_unit(row[1:].cpu().tolist())
            index = int(round(float(row[0])))
            record(
                index,
                unit,
                ledger.evaluate(
                    index,
                    denormalize(unit),
                    batch=iteration,
                    provenance=f"qlognehvi:seed={ledger.seed}:iteration={iteration}:member={member}",
                ),
            )
        entry = {
            "iteration": iteration,
            "training_points": int(train_x.shape[0]),
            "feasible_training_points": int(feasible_x.shape[0]),
            "batch_size": q,
            "fit_seconds": fit_seconds,
            "candidate_stage_seconds": candidate_stage_seconds,
            "refinement_seconds": refinement_seconds,
            "acquisition_seconds": candidate_stage_seconds + refinement_seconds,
            "candidate_stage_values": [float(v) for v in discrete_values.detach().cpu().tolist()],
            "refinement": refinement_log,
            "acquisition_value": float(discrete_values.detach().max().cpu()),
            "evaluations": len(ledger.records),
            "hypervolume": ledger.hypervolume_curve[-1]["hypervolume"],
        }
        iteration_log.append(entry)
        if progress is not None:
            progress(entry)
    arguments = {
        "q": int(batch_size),
        "mc_samples": int(mc_samples),
        "candidates_per_design": int(candidates_per_design),
        "refine_maxiter": int(refine_maxiter),
        "refine_num_restarts": int(refine_num_restarts),
        "sequential_candidate_stage": True,
        "fit_noise_floor": float(fit_noise_floor),
        "torch_threads": int(torch.get_num_threads()),
        "dtype": "float64",
        "device": str(tensor_device),
    }
    return {
        "iterations": iteration,
        "iteration_log": iteration_log,
        "device": str(tensor_device),
        "torch_threads": int(torch.get_num_threads()),
        "arguments": arguments,
        "model": fitted_label,
        "acquisition": acquisition_label(
            q=arguments["q"],
            mc_samples=arguments["mc_samples"],
            candidates_per_design=arguments["candidates_per_design"],
            refine_maxiter=arguments["refine_maxiter"],
            refine_num_restarts=arguments["refine_num_restarts"],
            sequential_candidate_stage=arguments["sequential_candidate_stage"],
        ),
    }
