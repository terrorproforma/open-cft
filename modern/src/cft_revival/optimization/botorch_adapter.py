"""Lazy BoTorch boundary for multi-fidelity constrained MOBO.

Core campaign code never imports this module's optional dependencies. The API
imports below were checked against the official BoTorch 0.18.1 documentation,
but cannot be runtime-verified in this environment because torch/BoTorch are
intentionally not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from math import isfinite
from typing import Any, Callable, Sequence

from .domain import (
    ConstraintSense,
    ContinuousConstraint,
    ObjectiveDirection,
    ObjectiveSpec,
)


class OptionalDependencyError(ImportError):
    """Raised with an actionable message when the SOTA adapter is unavailable."""


class UnsupportedTaskNoiseError(ValueError):
    """Known per-row task noise is unsupported by this MultiTaskGP contract."""


@dataclass(frozen=True)
class ModelOutputLayout:
    """Ordered model outputs: objectives first, constraints second."""

    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ContinuousConstraint, ...] = ()
    constraint_boundary_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if any(not isinstance(item, ObjectiveSpec) for item in self.objectives):
            raise ValueError("layout objectives must be ObjectiveSpec records")
        if any(
            not isinstance(item, ContinuousConstraint)
            for item in self.constraints
        ):
            raise ValueError(
                "layout constraints must be ContinuousConstraint records"
            )
        names = tuple(item.name for item in (*self.objectives, *self.constraints))
        if not self.objectives or len(names) != len(set(names)):
            raise ValueError("model output names must be unique with objectives present")
        if (
            not isfinite(self.constraint_boundary_epsilon)
            or self.constraint_boundary_epsilon <= 0.0
        ):
            raise ValueError("constraint boundary epsilon must be finite and positive")

    @property
    def objective_indices(self) -> tuple[int, ...]:
        return tuple(range(len(self.objectives)))

    @property
    def constraint_indices(self) -> tuple[int, ...]:
        start = len(self.objectives)
        return tuple(range(start, start + len(self.constraints)))

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in (*self.objectives, *self.constraints))


@dataclass(frozen=True)
class GPModelPlan:
    kind: str
    outputs: tuple[str, ...]
    observed_heteroskedastic_noise: bool
    source_task_feature: int | None
    discrepancy_strategy: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "outputs", tuple(str(item) for item in self.outputs))
        object.__setattr__(
            self, "discrepancy_strategy", str(self.discrepancy_strategy)
        )
        if not isinstance(self.observed_heteroskedastic_noise, bool):
            raise ValueError("GP noise plan flag must be boolean")
        if self.source_task_feature is not None and (
            isinstance(self.source_task_feature, bool)
            or not isinstance(self.source_task_feature, int)
        ):
            raise ValueError("source task feature must be an integer or None")
        if not self.kind or not self.outputs or any(not item for item in self.outputs):
            raise ValueError("GP model plan requires a kind and named outputs")
        if not self.discrepancy_strategy:
            raise ValueError("GP model plan requires an explicit discrepancy strategy")


@dataclass(frozen=True)
class AcquisitionPlan:
    primary: str = "qLogNEHVI"
    fallback: str = "qLogNParEGO"
    sequential_greedy: bool = True
    pending_aware: bool = True
    constraint_convention: str = "strictly-negative-is-feasible"
    qlognparego_batch_optimizer: str = "optimize_acqf_list"

    def __post_init__(self) -> None:
        for name in (
            "primary",
            "fallback",
            "constraint_convention",
            "qlognparego_batch_optimizer",
        ):
            object.__setattr__(self, name, str(getattr(self, name)))
        if not isinstance(self.sequential_greedy, bool) or not isinstance(
            self.pending_aware, bool
        ):
            raise ValueError("acquisition plan flags must be boolean")


def objective_direction_signs(
    objectives: Sequence[ObjectiveSpec],
) -> tuple[float, ...]:
    """Return signs mapping physical outputs to BoTorch's maximize convention."""
    if not objectives:
        raise ValueError("at least one objective specification is required")
    return tuple(
        1.0 if objective.direction is ObjectiveDirection.MAXIMIZE else -1.0
        for objective in objectives
    )


def transform_objective_values(
    values: Sequence[float],
    objectives: Sequence[ObjectiveSpec],
) -> tuple[float, ...]:
    """Transform physical objective values to an all-maximize representation."""
    signs = objective_direction_signs(objectives)
    if len(values) != len(signs):
        raise ValueError("objective values and specifications must have equal length")
    transformed = tuple(float(value) * sign for value, sign in zip(values, signs, strict=True))
    if not all(isfinite(value) for value in transformed):
        raise ValueError("objective values must be finite")
    return transformed


def botorch_constraint_value(
    physical_value: float,
    specification: ContinuousConstraint,
    *,
    boundary_epsilon: float = 1e-12,
) -> float:
    """Map a physical constraint to BoTorch's strict ``value < 0`` convention.

    The small dimensionless epsilon makes a physically feasible equality
    strictly negative without changing a meaningful normalized violation.
    """
    if not isfinite(boundary_epsilon) or boundary_epsilon <= 0.0:
        raise ValueError("boundary epsilon must be finite and positive")
    return specification.normalized_residual(physical_value) - boundary_epsilon


def transform_model_output_values(
    values: Sequence[float],
    layout: ModelOutputLayout,
) -> tuple[float, ...]:
    """Transform physical objectives and constraints without mixing their roles."""
    if len(values) != len(layout.output_names):
        raise ValueError("model values and output layout must have equal length")
    objective_count = len(layout.objectives)
    objectives = transform_objective_values(
        values[:objective_count],
        layout.objectives,
    )
    constraints = tuple(
        botorch_constraint_value(
            value,
            specification,
            boundary_epsilon=layout.constraint_boundary_epsilon,
        )
        for value, specification in zip(
            values[objective_count:],
            layout.constraints,
            strict=True,
        )
    )
    return (*objectives, *constraints)


def transform_model_output_variances(
    variances: Sequence[float],
    layout: ModelOutputLayout,
) -> tuple[float, ...]:
    """Preserve objective variance and scale constraint variance by ``s**2``."""
    if len(variances) != len(layout.output_names):
        raise ValueError("model variances and output layout must have equal length")
    normalized = tuple(float(value) for value in variances)
    if any(not isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError("model variances must be finite and non-negative")
    objective_count = len(layout.objectives)
    constraint_variances = tuple(
        variance / specification.violation_scale**2
        for variance, specification in zip(
            normalized[objective_count:],
            layout.constraints,
            strict=True,
        )
    )
    return (*normalized[:objective_count], *constraint_variances)


def _transform_tensor_model_outputs(
    values: Any,
    layout: ModelOutputLayout,
) -> Any:
    if values.shape[-1] != len(layout.output_names):
        raise ValueError("tensor output width does not match model output layout")
    columns = [
        values[..., index] * sign
        for index, sign in zip(
            layout.objective_indices,
            objective_direction_signs(layout.objectives),
            strict=True,
        )
    ]
    for index, specification in zip(
        layout.constraint_indices,
        layout.constraints,
        strict=True,
    ):
        if specification.sense is ConstraintSense.LESS_THAN_OR_EQUAL:
            residual = values[..., index] - specification.threshold
        else:
            residual = specification.threshold - values[..., index]
        columns.append(
            residual / specification.violation_scale
            - layout.constraint_boundary_epsilon
        )
    return import_module("torch").stack(columns, dim=-1)


def _transform_tensor_model_variances(
    variances: Any,
    layout: ModelOutputLayout,
) -> Any:
    if variances.shape[-1] != len(layout.output_names):
        raise ValueError("tensor variance width does not match model output layout")
    columns = [
        variances[..., index] for index in layout.objective_indices
    ]
    columns.extend(
        variances[..., index] / specification.violation_scale**2
        for index, specification in zip(
            layout.constraint_indices,
            layout.constraints,
            strict=True,
        )
    )
    return import_module("torch").stack(columns, dim=-1)


def _transform_tensor_objectives(
    values: Any,
    objectives: Sequence[ObjectiveSpec],
) -> Any:
    signs = values.new_tensor(objective_direction_signs(objectives))
    if values.shape[-1] != len(signs):
        raise ValueError("tensor width does not match objective specifications")
    return values * signs


def constraint_output_callables(
    layout: ModelOutputLayout,
) -> tuple[Callable[[Any], Any], ...]:
    """Select only declared constraint outputs; each is feasible when `< 0`."""
    return tuple(
        lambda samples, index=index: samples[..., index]
        for index in layout.constraint_indices
    )


def dependencies_available() -> bool:
    return all(
        find_spec(package) is not None
        for package in ("torch", "botorch", "gpytorch")
    )


def require_dependencies() -> None:
    missing = [
        package
        for package in ("torch", "botorch", "gpytorch")
        if find_spec(package) is None
    ]
    if missing:
        raise OptionalDependencyError(
            "BoTorch adapter requires optional packages "
            + ", ".join(missing)
            + "; install the project optimization extra in an isolated environment"
        )


def default_model_plan(outputs: Sequence[str]) -> GPModelPlan:
    return GPModelPlan(
        kind="independent exact GPs with fixed observed noise",
        outputs=tuple(outputs),
        observed_heteroskedastic_noise=True,
        source_task_feature=-1,
        discrepancy_strategy=(
            "source-task covariance plus per-output residual discrepancy; "
            "do not merge emulator posterior variance with physical discrepancy"
        ),
    )


def load_api() -> dict[str, Any]:
    """Resolve documented symbols lazily, failing cleanly at the boundary."""
    require_dependencies()
    try:
        return {
            "torch": import_module("torch"),
            "SingleTaskGP": getattr(import_module("botorch.models"), "SingleTaskGP"),
            "MultiTaskGP": getattr(import_module("botorch.models"), "MultiTaskGP"),
            "ModelListGP": getattr(import_module("botorch.models"), "ModelListGP"),
            "StratifiedStandardize": getattr(
                import_module("botorch.models.transforms.outcome"),
                "StratifiedStandardize",
            ),
            "IdentityMCMultiOutputObjective": getattr(
                import_module("botorch.acquisition.multi_objective.objective"),
                "IdentityMCMultiOutputObjective",
            ),
            "fit_gpytorch_mll": getattr(
                import_module("botorch.fit"), "fit_gpytorch_mll"
            ),
            "ExactMarginalLogLikelihood": getattr(
                import_module("gpytorch.mlls"), "ExactMarginalLogLikelihood"
            ),
            "qLogNEHVI": getattr(
                import_module("botorch.acquisition.multi_objective.logei"),
                "qLogNoisyExpectedHypervolumeImprovement",
            ),
            "qLogNParEGO": getattr(
                import_module("botorch.acquisition.multi_objective.parego"),
                "qLogNParEGO",
            ),
            "optimize_acqf": getattr(import_module("botorch.optim"), "optimize_acqf"),
            "optimize_acqf_list": getattr(
                import_module("botorch.optim"), "optimize_acqf_list"
            ),
        }
    except (ImportError, AttributeError) as exc:
        raise OptionalDependencyError(
            "installed torch/BoTorch/GPyTorch versions do not expose the documented "
            "BoTorch 0.18.1 adapter API"
        ) from exc


def build_exact_output_models(
    train_x: Any,
    train_y: Any,
    train_yvar: Any,
    layout: ModelOutputLayout,
) -> Any:
    """Build and fit independent exact GPs with observed heteroskedastic noise.

    ``train_yvar`` is measurement-noise variance, not model discrepancy.
    """
    api = load_api()
    if train_y.ndim != 2 or train_yvar.shape != train_y.shape:
        raise ValueError("train_y and train_yvar must be equal two-dimensional tensors")
    train_y = _transform_tensor_model_outputs(train_y, layout)
    train_yvar = _transform_tensor_model_variances(train_yvar, layout)
    models = []
    for output in range(train_y.shape[-1]):
        model = api["SingleTaskGP"](
            train_x,
            train_y[..., output : output + 1],
            train_Yvar=train_yvar[..., output : output + 1],
        )
        mll = api["ExactMarginalLogLikelihood"](model.likelihood, model)
        api["fit_gpytorch_mll"](mll)
        models.append(model)
    return api["ModelListGP"](*models)


def build_source_task_models(
    train_x_with_source: Any,
    train_y: Any,
    layout: ModelOutputLayout,
    *,
    train_yvar: Any | None = None,
    task_feature: int = -1,
) -> Any:
    """Fit task-stratified GPs with BoTorch's supported inferred-noise contract."""
    if train_yvar is not None:
        raise UnsupportedTaskNoiseError(
            "BoTorch 0.18.1 MultiTaskGP does not support differing known noise "
            "across tasks; use inferred task noise here or fit per-source "
            "SingleTaskGP models with build_exact_output_models"
        )
    api = load_api()
    if train_y.ndim != 2:
        raise ValueError("train_y must be a two-dimensional tensor")
    train_y = _transform_tensor_model_outputs(train_y, layout)
    task_values = train_x_with_source[..., task_feature].unique(sorted=True)
    models = []
    for output in range(train_y.shape[-1]):
        outcome_transform = api["StratifiedStandardize"](
            stratification_idx=task_feature,
            all_task_values=task_values,
            dtype=train_x_with_source.dtype,
        )
        model = api["MultiTaskGP"](
            train_x_with_source,
            train_y[..., output : output + 1],
            task_feature=task_feature,
            train_Yvar=None,
            outcome_transform=outcome_transform,
        )
        mll = api["ExactMarginalLogLikelihood"](model.likelihood, model)
        api["fit_gpytorch_mll"](mll)
        models.append(model)
    return api["ModelListGP"](*models)


def build_qlognehvi(
    model: Any,
    reference_point: Any,
    x_baseline: Any,
    layout: ModelOutputLayout,
    *,
    x_pending: Any | None = None,
    model_outputs_are_direction_transformed: bool = False,
) -> Any:
    """Construct qLogNEHVI with all-maximize outputs and strict-negative constraints."""
    if not model_outputs_are_direction_transformed:
        raise ValueError(
            "model outputs must be fitted through the mixed-direction transform"
        )
    api = load_api()
    transformed_reference = _transform_tensor_objectives(
        reference_point,
        layout.objectives,
    )
    objective = api["IdentityMCMultiOutputObjective"](
        outcomes=list(layout.objective_indices)
    )
    return api["qLogNEHVI"](
        model=model,
        ref_point=transformed_reference,
        X_baseline=x_baseline,
        X_pending=x_pending,
        objective=objective,
        constraints=list(constraint_output_callables(layout)) or None,
        prune_baseline=True,
        cache_pending=True,
    )


def build_qlognparego(
    model: Any,
    x_baseline: Any,
    layout: ModelOutputLayout,
    *,
    x_pending: Any | None = None,
    model_outputs_are_direction_transformed: bool = False,
) -> Any:
    """Construct one qLogNParEGO acquisition over transformed model outputs."""
    if not model_outputs_are_direction_transformed:
        raise ValueError(
            "model outputs must be fitted through the mixed-direction transform"
        )
    api = load_api()
    objective = api["IdentityMCMultiOutputObjective"](
        outcomes=list(layout.objective_indices)
    )
    return api["qLogNParEGO"](
        model=model,
        X_baseline=x_baseline,
        X_pending=x_pending,
        objective=objective,
        constraints=list(constraint_output_callables(layout)) or None,
    )


def optimize_qlognparego_batch(
    model: Any,
    x_baseline: Any,
    bounds: Any,
    layout: ModelOutputLayout,
    *,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    x_pending: Any | None = None,
    model_outputs_are_direction_transformed: bool = False,
) -> Any:
    """Implement documented sequential-greedy batching via ``optimize_acqf_list``."""
    if batch_size < 1 or num_restarts < 1 or raw_samples < 1:
        raise ValueError("batch and acquisition optimization counts must be positive")
    acquisitions = [
        build_qlognparego(
            model,
            x_baseline,
            layout,
            x_pending=x_pending,
            model_outputs_are_direction_transformed=model_outputs_are_direction_transformed,
        )
        for _ in range(batch_size)
    ]
    api = load_api()
    return api["optimize_acqf_list"](
        acq_function_list=acquisitions,
        bounds=bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
