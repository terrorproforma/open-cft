"""Dependency-free deterministic bounded nonlinear least-squares solver."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from sys import float_info
from typing import Callable, Sequence

from .models import (
    PlasmaNumericsError,
    PlasmaMultiStartResult,
    PlasmaSolveResult,
    PlasmaState,
    PlasmaValidationError,
    SolverDiagnostics,
    StateBounds,
    XenonGlobalInputs,
)
from .residuals import (
    analytic_jacobian,
    default_state_bounds,
    evaluate_residual,
    is_feasible,
)

VectorFunction = Callable[[tuple[float, ...]], Sequence[float]]
MatrixFunction = Callable[[tuple[float, ...]], Sequence[Sequence[float]]]
FeasibilityFunction = Callable[[tuple[float, ...]], bool]
ProjectionFunction = Callable[[tuple[float, ...]], tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class SolverOptions:
    max_iterations: int = 250
    residual_tolerance: float = 1.0e-9
    gradient_tolerance: float = 1.0e-10
    step_tolerance: float = 1.0e-12
    initial_damping: float = 1.0e-3
    finite_difference_step: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise PlasmaValidationError("max_iterations must be positive")
        for name in (
            "residual_tolerance",
            "gradient_tolerance",
            "step_tolerance",
            "initial_damping",
            "finite_difference_step",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise PlasmaValidationError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class LeastSquaresResult:
    vector: tuple[float, ...] | None
    residual: tuple[float, ...] | None
    diagnostics: SolverDiagnostics


def _finite_vector(values: Sequence[float], expected: int | None = None) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if expected is not None and len(vector) != expected:
        raise PlasmaNumericsError("callback returned an unexpected vector length")
    if len(vector) == 0 or any(not isfinite(value) for value in vector):
        raise PlasmaNumericsError("callback returned a non-finite or empty vector")
    return vector


def _cost(residual: Sequence[float]) -> float:
    value = 0.5 * fsum(component * component for component in residual)
    if not isfinite(value):
        raise PlasmaNumericsError("least-squares cost is non-finite")
    return value


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return fsum(a * b for a, b in zip(left, right, strict=True))


def _pivoted_qr(
    matrix: Sequence[Sequence[float]],
    right: Sequence[float] | None = None,
) -> tuple[tuple[float, ...] | None, int, float]:
    """Solve by column-pivoted modified Gram-Schmidt and estimate rank."""

    row_count = len(matrix)
    column_count = len(matrix[0])
    columns = [
        [float(matrix[row][column]) for row in range(row_count)]
        for column in range(column_count)
    ]
    permutation = list(range(column_count))
    triangular = [[0.0] * column_count for _ in range(column_count)]
    transformed = [0.0] * column_count
    diagonal: list[float] = []
    scale = max(
        sqrt(_dot(column, column)) for column in columns
    )
    tolerance = max(row_count, column_count) * float_info.epsilon * max(scale, 1.0)
    rank = 0
    for pivot in range(column_count):
        selected = max(
            range(pivot, column_count),
            key=lambda index: _dot(columns[index], columns[index]),
        )
        if selected != pivot:
            columns[pivot], columns[selected] = columns[selected], columns[pivot]
            permutation[pivot], permutation[selected] = (
                permutation[selected],
                permutation[pivot],
            )
            for previous in range(pivot):
                triangular[previous][pivot], triangular[previous][selected] = (
                    triangular[previous][selected],
                    triangular[previous][pivot],
                )
        norm = sqrt(_dot(columns[pivot], columns[pivot]))
        if norm <= tolerance:
            break
        rank += 1
        diagonal.append(norm)
        basis = [value / norm for value in columns[pivot]]
        triangular[pivot][pivot] = norm
        if right is not None:
            transformed[pivot] = _dot(basis, right)
        for column in range(pivot + 1, column_count):
            coefficient = _dot(basis, columns[column])
            triangular[pivot][column] = coefficient
            columns[column] = [
                value - coefficient * direction
                for value, direction in zip(columns[column], basis, strict=True)
            ]
    condition = max(diagonal) / min(diagonal) if diagonal else float_info.max
    if right is None:
        return None, rank, condition
    pivoted_solution = [0.0] * column_count
    for row in range(rank - 1, -1, -1):
        remainder = fsum(
            triangular[row][column] * pivoted_solution[column]
            for column in range(row + 1, rank)
        )
        pivoted_solution[row] = (
            transformed[row] - remainder
        ) / triangular[row][row]
    solution = [0.0] * column_count
    for pivoted_index, original_index in enumerate(permutation):
        solution[original_index] = pivoted_solution[pivoted_index]
    if any(not isfinite(value) for value in solution):
        raise PlasmaNumericsError("QR solve produced a non-finite step")
    return tuple(solution), rank, condition


def _scaled_lm_step(
    jacobian: Sequence[Sequence[float]],
    residual: Sequence[float],
    variable_scale: Sequence[float],
    damping: float,
) -> tuple[tuple[float, ...], int, float, float]:
    column_count = len(variable_scale)
    scaled = [
        [
            jacobian[row][column] * variable_scale[column]
            for column in range(column_count)
        ]
        for row in range(len(jacobian))
    ]
    _, rank, condition = _pivoted_qr(scaled)
    augmented = [row.copy() for row in scaled]
    root_damping = sqrt(damping)
    for column in range(column_count):
        damping_row = [0.0] * column_count
        damping_row[column] = root_damping
        augmented.append(damping_row)
    right = tuple(-value for value in residual) + (0.0,) * column_count
    scaled_step, _, _ = _pivoted_qr(augmented, right)
    if scaled_step is None:
        raise PlasmaNumericsError("QR solver did not return a step")
    gradient = [
        fsum(jacobian[row][column] * residual[row] for row in range(len(residual)))
        for column in range(column_count)
    ]
    step = tuple(
        scaled_step[column] * variable_scale[column]
        for column in range(column_count)
    )
    return step, rank, condition, max(abs(value) for value in gradient)


def finite_difference_matrix(
    function: VectorFunction,
    vector: tuple[float, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    relative_step: float,
) -> tuple[tuple[float, ...], ...]:
    base = _finite_vector(function(vector))
    columns: list[tuple[float, ...]] = []
    for index, value in enumerate(vector):
        nominal = relative_step * max(1.0, abs(value))
        low_room = value - lower[index]
        high_room = upper[index] - value
        if low_room >= nominal and high_room >= nominal:
            left = list(vector)
            right = list(vector)
            left[index] -= nominal
            right[index] += nominal
            low_value = _finite_vector(function(tuple(left)), len(base))
            high_value = _finite_vector(function(tuple(right)), len(base))
            columns.append(
                tuple(
                    (high - low) / (2.0 * nominal)
                    for low, high in zip(low_value, high_value, strict=True)
                )
            )
        elif high_room > 0.0:
            step = min(nominal, high_room)
            right = list(vector)
            right[index] += step
            high_value = _finite_vector(function(tuple(right)), len(base))
            columns.append(
                tuple(
                    (high - low) / step
                    for low, high in zip(base, high_value, strict=True)
                )
            )
        elif low_room > 0.0:
            step = min(nominal, low_room)
            left = list(vector)
            left[index] -= step
            low_value = _finite_vector(function(tuple(left)), len(base))
            columns.append(
                tuple(
                    (high - low) / step
                    for low, high in zip(low_value, base, strict=True)
                )
            )
        else:
            raise PlasmaNumericsError(
                f"variable {index} is fixed but no analytic Jacobian was supplied"
            )
    return tuple(
        tuple(columns[column][row] for column in range(len(vector)))
        for row in range(len(base))
    )


def solve_bounded_least_squares(
    function: VectorFunction,
    initial: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    jacobian: MatrixFunction | None = None,
    feasible: FeasibilityFunction | None = None,
    project: ProjectionFunction | None = None,
    options: SolverOptions | None = None,
) -> LeastSquaresResult:
    """Solve a finite bounded least-squares problem with deterministic diagnostics."""

    selected = SolverOptions() if options is None else options
    vector = tuple(float(value) for value in initial)
    low = tuple(float(value) for value in lower)
    high = tuple(float(value) for value in upper)
    size = len(vector)
    if size == 0 or len(low) != size or len(high) != size:
        raise PlasmaValidationError("initial, lower, and upper must have one equal nonzero length")
    if any(not isfinite(value) for value in (*vector, *low, *high)):
        raise PlasmaValidationError("initial and bounds must be finite")
    if any(left > right for left, right in zip(low, high, strict=True)):
        raise PlasmaValidationError("lower bounds must not exceed upper bounds")

    def apply_projection(candidate: tuple[float, ...]) -> tuple[float, ...]:
        clipped = tuple(
            min(max(value, left), right)
            for value, left, right in zip(candidate, low, high, strict=True)
        )
        if project is None:
            return clipped
        try:
            projected = tuple(float(value) for value in project(clipped))
        except (TypeError, ValueError, OverflowError) as error:
            raise PlasmaValidationError(
                "project must return a finite numeric sequence"
            ) from error
        if len(projected) != size:
            raise PlasmaValidationError(
                "project must return a vector of unchanged length"
            )
        if any(not isfinite(value) for value in projected):
            raise PlasmaValidationError("project must return only finite values")
        if any(
            value < left or value > right
            for value, left, right in zip(projected, low, high, strict=True)
        ):
            raise PlasmaValidationError(
                "project returned a value outside the declared bounds"
            )
        return tuple(
            min(max(value, left), right)
            for value, left, right in zip(projected, low, high, strict=True)
        )

    vector = apply_projection(vector)
    function_evaluations = 0
    jacobian_evaluations = 0
    damping = selected.initial_damping
    maximum = float_info.max
    final_rank = 0
    final_condition = maximum

    def failed(
        reason: str,
        iterations: int,
        initial_cost: float = maximum,
        residual: tuple[float, ...] = (),
    ) -> LeastSquaresResult:
        diagnostics = SolverDiagnostics(
            converged=False,
            reason=reason,
            iterations=iterations,
            function_evaluations=function_evaluations,
            jacobian_evaluations=jacobian_evaluations,
            initial_cost=initial_cost,
            final_cost=initial_cost,
            residual_inf_norm=maximum,
            gradient_inf_norm=maximum,
            damping=min(damping, maximum),
            active_bound_count=sum(
                value == left or value == right
                for value, left, right in zip(vector, low, high, strict=True)
            ),
            feasible=False,
            finite=True,
            normalized_residuals=residual,
            jacobian_rank=final_rank,
            jacobian_condition_estimate=final_condition,
        )
        return LeastSquaresResult(None, None, diagnostics)

    if feasible is not None and not feasible(vector):
        return failed("infeasible_initial_state", 0)
    try:
        residual = _finite_vector(function(vector))
        function_evaluations += 1
        current_cost = _cost(residual)
    except (ArithmeticError, ValueError, OverflowError):
        function_evaluations += 1
        return failed("nonfinite_initial_evaluation", 0)
    initial_cost = current_cost
    gradient_norm = maximum

    for iteration in range(selected.max_iterations + 1):
        residual_norm = max(abs(value) for value in residual)
        if residual_norm <= selected.residual_tolerance:
            if final_rank == 0:
                if jacobian is None:
                    rank_matrix = finite_difference_matrix(
                        function,
                        vector,
                        low,
                        high,
                        selected.finite_difference_step,
                    )
                    function_evaluations += 1 + 2 * size
                else:
                    rank_matrix = tuple(
                        tuple(float(value) for value in row)
                        for row in jacobian(vector)
                    )
                jacobian_evaluations += 1
                variable_scale = tuple(
                    max(abs(value), 0.25 * (right - left), float_info.min)
                    for value, left, right in zip(
                        vector, low, high, strict=True
                    )
                )
                scaled_rank_matrix = tuple(
                    tuple(
                        row[column] * variable_scale[column]
                        for column in range(size)
                    )
                    for row in rank_matrix
                )
                _, final_rank, final_condition = _pivoted_qr(
                    scaled_rank_matrix
                )
            diagnostics = SolverDiagnostics(
                converged=True,
                reason="residual_tolerance",
                iterations=iteration,
                function_evaluations=function_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                initial_cost=initial_cost,
                final_cost=current_cost,
                residual_inf_norm=residual_norm,
                gradient_inf_norm=0.0 if iteration == 0 else gradient_norm,
                damping=damping,
                active_bound_count=sum(
                    value == left or value == right
                    for value, left, right in zip(vector, low, high, strict=True)
                ),
                feasible=feasible is None or feasible(vector),
                finite=True,
                normalized_residuals=residual,
                jacobian_rank=final_rank,
                jacobian_condition_estimate=final_condition,
            )
            return LeastSquaresResult(vector, residual, diagnostics)
        if iteration == selected.max_iterations:
            break
        try:
            if jacobian is None:
                matrix = finite_difference_matrix(
                    function, vector, low, high, selected.finite_difference_step
                )
                function_evaluations += 2 * size
            else:
                matrix = tuple(
                    tuple(float(value) for value in row) for row in jacobian(vector)
                )
            jacobian_evaluations += 1
            if (
                len(matrix) != len(residual)
                or any(len(row) != size for row in matrix)
                or any(not isfinite(value) for row in matrix for value in row)
            ):
                raise PlasmaNumericsError("Jacobian has invalid shape or values")
            variable_scale = tuple(
                max(
                    abs(value),
                    0.25 * (right - left),
                    float_info.min,
                )
                for value, left, right in zip(vector, low, high, strict=True)
            )
            step, final_rank, final_condition, gradient_norm = _scaled_lm_step(
                matrix,
                residual,
                variable_scale,
                damping,
            )
            if gradient_norm <= selected.gradient_tolerance:
                reason = "gradient_tolerance_without_balance"
                break
            trial = apply_projection(
                tuple(
                    value + delta
                    for value, delta in zip(vector, step, strict=True)
                )
            )
            step_norm = max(
                abs(after - before)
                for before, after in zip(vector, trial, strict=True)
            )
            scale = max(1.0, max(abs(value) for value in vector))
            if step_norm <= selected.step_tolerance * scale:
                reason = "step_tolerance_without_balance"
                break
            if feasible is not None and not feasible(trial):
                damping = min(damping * 10.0, 1.0e100)
                continue
            trial_residual = _finite_vector(function(trial), len(residual))
            function_evaluations += 1
            trial_cost = _cost(trial_residual)
            if trial_cost < current_cost:
                vector = trial
                residual = trial_residual
                current_cost = trial_cost
                damping = max(damping * 0.3, 1.0e-18)
            else:
                damping = min(damping * 10.0, 1.0e100)
        except (ArithmeticError, ValueError, OverflowError):
            reason = "nonfinite_or_singular_iteration"
            break
    else:
        reason = "iteration_limit"
    if "reason" not in locals():
        reason = "iteration_limit"
    diagnostics = SolverDiagnostics(
        converged=False,
        reason=reason,
        iterations=min(iteration, selected.max_iterations),
        function_evaluations=function_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        initial_cost=initial_cost,
        final_cost=current_cost,
        residual_inf_norm=max(abs(value) for value in residual),
        gradient_inf_norm=gradient_norm,
        damping=damping,
        active_bound_count=sum(
            value == left or value == right
            for value, left, right in zip(vector, low, high, strict=True)
        ),
        feasible=feasible is None or feasible(vector),
        finite=all(
            isfinite(value)
            for value in (initial_cost, current_cost, gradient_norm, damping, *residual)
        ),
        normalized_residuals=residual,
        jacobian_rank=final_rank,
        jacobian_condition_estimate=final_condition,
    )
    return LeastSquaresResult(None, residual, diagnostics)


def representative_initial_state(inputs: XenonGlobalInputs) -> PlasmaState:
    """Deterministic, source-scale initial state; it is not a fitted solution."""

    voltage = inputs.anode_voltage_v
    current = inputs.anode_current_a
    electron = (
        0.10 * current,
        0.11 * current,
        0.60 * current,
        0.85 * current,
        1.001 * current,
    )
    ion = tuple(current - value for value in electron)
    probability = inputs.cusp_arrival_probabilities
    return PlasmaState(
        plasma_potential_v=(0.02 * voltage, 0.90 * voltage, voltage, voltage),
        electron_temperature_ev=(
            0.01 * voltage,
            0.10 * voltage,
            0.05 * voltage,
            0.025 * voltage,
        ),
        ionization_source_current_a=(
            0.01 * current,
            0.50 * current,
            0.30 * current,
            0.15 * current,
        ),
        electron_current_a=electron,
        ion_current_a=ion,  # type: ignore[arg-type]
        cusp_ion_current_a=tuple(
            electron[index] * probability[index] for index in range(3)
        ),  # type: ignore[arg-type]
    )


def solve_global_discharge(
    inputs: XenonGlobalInputs,
    initial_state: PlasmaState | None = None,
    *,
    bounds: StateBounds | None = None,
    options: SolverOptions | None = None,
    use_analytic_jacobian: bool = True,
) -> PlasmaSolveResult:
    """Solve the normalized reduced Kornfeld balances on the CPU."""

    initial = representative_initial_state(inputs) if initial_state is None else initial_state
    selected_bounds = default_state_bounds(inputs) if bounds is None else bounds

    def function(vector: tuple[float, ...]) -> Sequence[float]:
        return evaluate_residual(PlasmaState.from_vector(vector), inputs).normalized

    def matrix(vector: tuple[float, ...]) -> Sequence[Sequence[float]]:
        return analytic_jacobian(PlasmaState.from_vector(vector), inputs)

    def feasible(vector: tuple[float, ...]) -> bool:
        try:
            return is_feasible(PlasmaState.from_vector(vector), inputs, selected_bounds)
        except (ArithmeticError, ValueError, OverflowError):
            return False

    def project(vector: tuple[float, ...]) -> tuple[float, ...]:
        values = list(vector)
        ordered_phi = sorted(values[0:4])
        ordered_phi[3] = max(ordered_phi[3], inputs.anode_voltage_v)
        values[0:4] = ordered_phi
        values[20] = max(values[20], -values[21])
        return tuple(
            min(max(value, low), high)
            for value, low, high in zip(
                values, selected_bounds.lower, selected_bounds.upper, strict=True
            )
        )

    solved = solve_bounded_least_squares(
        function,
        initial.to_vector(),
        selected_bounds.lower,
        selected_bounds.upper,
        jacobian=matrix if use_analytic_jacobian else None,
        feasible=feasible,
        project=project,
        options=options,
    )
    if not solved.diagnostics.converged or solved.vector is None:
        return PlasmaSolveResult(state=None, evaluation=None, diagnostics=solved.diagnostics)
    state = PlasmaState.from_vector(solved.vector)
    evaluation = evaluate_residual(state, inputs)
    if not is_feasible(state, inputs, selected_bounds):
        raise PlasmaNumericsError("solver attempted to publish an infeasible converged state")
    return PlasmaSolveResult(state=state, evaluation=evaluation, diagnostics=solved.diagnostics)


def deterministic_initial_states(
    inputs: XenonGlobalInputs,
    *,
    count: int = 5,
) -> tuple[PlasmaState, ...]:
    """Generate a fixed, reproducible spread around the documented seed."""

    if count <= 0 or count > 9:
        raise PlasmaValidationError("count must be in [1, 9]")
    base = representative_initial_state(inputs).to_vector()
    patterns = (
        (1.00, 1.00, 1.00),
        (0.80, 0.70, 0.75),
        (1.20, 1.30, 1.25),
        (0.60, 1.20, 0.85),
        (1.35, 0.80, 1.15),
        (0.90, 1.40, 0.65),
        (1.10, 0.60, 1.35),
        (0.70, 0.90, 1.30),
        (1.30, 1.10, 0.70),
    )
    states: list[PlasmaState] = []
    for potential_factor, temperature_factor, current_factor in patterns[:count]:
        values = list(base)
        values[0] *= potential_factor
        values[1] = inputs.anode_voltage_v - (
            inputs.anode_voltage_v - values[1]
        ) * potential_factor
        for index in range(4, 8):
            values[index] *= temperature_factor
        for index in (*range(8, 17), *range(22, 25)):
            values[index] *= current_factor
        states.append(PlasmaState.from_vector(values))
    return tuple(states)


def solve_global_discharge_multistart(
    inputs: XenonGlobalInputs,
    initial_states: Sequence[PlasmaState] | None = None,
    *,
    start_count: int = 5,
    bounds: StateBounds | None = None,
    options: SolverOptions | None = None,
    use_analytic_jacobian: bool = True,
) -> PlasmaMultiStartResult:
    """Run deterministic starts and retain strict success or the residual floor."""

    states = (
        deterministic_initial_states(inputs, count=start_count)
        if initial_states is None
        else tuple(initial_states)
    )
    if len(states) == 0:
        raise PlasmaValidationError("initial_states must not be empty")
    if any(not isinstance(state, PlasmaState) for state in states):
        raise PlasmaValidationError("every initial state must be PlasmaState")
    attempts = tuple(
        solve_global_discharge(
            inputs,
            state,
            bounds=bounds,
            options=options,
            use_analytic_jacobian=use_analytic_jacobian,
        )
        for state in states
    )
    converged = [
        (index, attempt)
        for index, attempt in enumerate(attempts)
        if attempt.diagnostics.converged
    ]
    candidates = converged if converged else list(enumerate(attempts))
    selected_index, best = min(
        candidates,
        key=lambda item: (
            item[1].diagnostics.residual_inf_norm,
            item[0],
        ),
    )
    residual_floor = min(
        attempt.diagnostics.residual_inf_norm for attempt in attempts
    )
    return PlasmaMultiStartResult(
        best=best,
        attempts=attempts,
        selected_start_index=selected_index,
        residual_floor=residual_floor,
    )
