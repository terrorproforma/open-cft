"""Dependency-free CPU reference surface for the plasma workstream."""

from __future__ import annotations

from typing import Sequence

from .models import (
    PlasmaMultiStartResult,
    PlasmaSolveResult,
    PlasmaState,
    ResidualEvaluation,
    XenonGlobalInputs,
)
from .residuals import evaluate_residual, evaluate_residual_batch
from .solver import (
    SolverOptions,
    solve_global_discharge,
    solve_global_discharge_multistart,
)


def evaluate_plasma_residual_cpu(
    state: PlasmaState, inputs: XenonGlobalInputs
) -> ResidualEvaluation:
    return evaluate_residual(state, inputs)


def evaluate_plasma_residual_batch_cpu(
    states: Sequence[PlasmaState], inputs: Sequence[XenonGlobalInputs]
) -> tuple[ResidualEvaluation, ...]:
    return evaluate_residual_batch(states, inputs)


def solve_global_discharge_cpu(
    inputs: XenonGlobalInputs,
    initial_state: PlasmaState | None = None,
    *,
    options: SolverOptions | None = None,
) -> PlasmaSolveResult:
    return solve_global_discharge(inputs, initial_state, options=options)


def solve_global_discharge_multistart_cpu(
    inputs: XenonGlobalInputs,
    initial_states: Sequence[PlasmaState] | None = None,
    *,
    start_count: int = 5,
    options: SolverOptions | None = None,
) -> PlasmaMultiStartResult:
    return solve_global_discharge_multistart(
        inputs,
        initial_states,
        start_count=start_count,
        options=options,
    )
