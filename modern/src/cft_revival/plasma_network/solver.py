"""Deterministic solving with canonical post-backend publication verification."""

from __future__ import annotations

from math import fsum, isfinite, sqrt
from sys import float_info
from typing import Callable, Protocol, Sequence, runtime_checkable

from cft_revival.plasma import (
    LeastSquaresResult,
    project_nondecreasing,
    solve_bounded_least_squares,
)

from .ledger import generate_equation_ledger
from .models import (
    DynamicBounds,
    IdentifiabilityDiagnostics,
    NetworkInputs,
    NetworkMultiStartResult,
    NetworkNumericsError,
    NetworkSolveDiagnostics,
    NetworkSolveResult,
    NetworkSolverOptions,
    NetworkState,
    NetworkValidationError,
    PublicationPolicy,
)
from .residuals import analytic_jacobian, default_bounds, evaluate_residual, is_feasible
from .topology import validate_topology

VectorFunction = Callable[[tuple[float, ...]], Sequence[float]]
MatrixFunction = Callable[[tuple[float, ...]], Sequence[Sequence[float]]]
FeasibilityFunction = Callable[[tuple[float, ...]], bool]
ProjectionFunction = Callable[[tuple[float, ...]], tuple[float, ...]]


@runtime_checkable
class LeastSquaresBackend(Protocol):
    def solve(
        self,
        function: VectorFunction,
        initial: Sequence[float],
        bounds: DynamicBounds,
        jacobian: MatrixFunction,
        feasible: FeasibilityFunction,
        project: ProjectionFunction,
        options: NetworkSolverOptions,
    ) -> LeastSquaresResult:
        """Return a candidate; all status and residual claims are untrusted."""


class ScaledQrLmBackend:
    def solve(
        self,
        function: VectorFunction,
        initial: Sequence[float],
        bounds: DynamicBounds,
        jacobian: MatrixFunction,
        feasible: FeasibilityFunction,
        project: ProjectionFunction,
        options: NetworkSolverOptions,
    ) -> LeastSquaresResult:
        return solve_bounded_least_squares(
            function,
            initial,
            bounds.lower,
            bounds.upper,
            jacobian=jacobian,
            feasible=feasible,
            project=project,
            options=options.least_squares,
        )


def physical_variable_scales(inputs: NetworkInputs) -> tuple[float, ...]:
    """Fixed SI scales independent of state guesses and caller-provided bounds."""

    validate_topology(inputs.topology)
    n = inputs.dimensions.cell_count
    voltage = inputs.anode_voltage_v
    current = inputs.anode_current_a
    scales = (
        *(voltage for _ in range(n)),
        *(voltage for _ in range(n)),
        *(current for _ in range(n)),
        *(current for _ in range(n + 1)),
        *(current for _ in range(n + 1)),
        *(current for _ in range(n - 1)),
    )
    if (
        len(scales) != inputs.dimensions.state_size
        or any(not isfinite(value) or value <= 0.0 for value in scales)
    ):
        raise NetworkValidationError("physical variable scales are invalid")
    return scales


def _orthonormalize(
    vectors: Sequence[Sequence[float]], tolerance: float
) -> tuple[tuple[float, ...], ...]:
    basis: list[tuple[float, ...]] = []
    for vector in vectors:
        work = [float(value) for value in vector]
        for existing in basis:
            projection = fsum(
                value * direction
                for value, direction in zip(work, existing, strict=True)
            )
            work = [
                value - projection * direction
                for value, direction in zip(work, existing, strict=True)
            ]
        norm = sqrt(fsum(value * value for value in work))
        if not isfinite(norm) or norm <= tolerance:
            raise NetworkNumericsError("nullspace basis is numerically dependent")
        basis.append(tuple(value / norm for value in work))
    return tuple(basis)


def validate_nullspace_basis(
    matrix: Sequence[Sequence[float]],
    variable_scales: Sequence[float],
    basis: Sequence[Sequence[float]],
    *,
    expected_count: int,
    residual_tolerance: float,
    orthonormality_tolerance: float,
) -> tuple[bool, float, float]:
    """Validate dimensionless right-null vectors against the scaled canonical J."""

    rows = len(matrix)
    columns = len(variable_scales)
    if (
        expected_count < 0
        or rows == 0
        or columns == 0
        or any(len(row) != columns for row in matrix)
        or len(basis) != expected_count
        or any(len(vector) != columns for vector in basis)
    ):
        return False, float_info.max, float_info.max
    if any(
        not isfinite(value)
        for value in (
            *variable_scales,
            *(item for row in matrix for item in row),
            *(item for vector in basis for item in vector),
        )
    ):
        return False, float_info.max, float_info.max
    maximum_jv = 0.0
    maximum_orthonormality_error = 0.0
    for left_index, left in enumerate(basis):
        norm = fsum(value * value for value in left)
        maximum_orthonormality_error = max(
            maximum_orthonormality_error, abs(norm - 1.0)
        )
        for right_index in range(left_index):
            dot = fsum(
                a * b for a, b in zip(left, basis[right_index], strict=True)
            )
            maximum_orthonormality_error = max(
                maximum_orthonormality_error, abs(dot)
            )
        for row in matrix:
            product = fsum(
                row[column] * variable_scales[column] * left[column]
                for column in range(columns)
            )
            maximum_jv = max(maximum_jv, abs(product))
    valid = (
        maximum_jv <= residual_tolerance
        and maximum_orthonormality_error <= orthonormality_tolerance
    )
    return valid, maximum_jv, maximum_orthonormality_error


def _rank_and_nullspace(
    matrix: Sequence[Sequence[float]],
    variable_scales: Sequence[float],
    *,
    structural_rank: int,
    structural_nullity: int,
    represented: bool,
    rank_relative_tolerance: float,
    nullspace_residual_tolerance: float,
    orthonormality_tolerance: float,
) -> IdentifiabilityDiagnostics:
    rows = len(matrix)
    columns = len(variable_scales)
    if rows == 0 or columns == 0 or any(len(row) != columns for row in matrix):
        raise NetworkNumericsError("rank matrix has invalid dimensions")
    scaled = [
        [float(matrix[row][column]) * variable_scales[column] for column in range(columns)]
        for row in range(rows)
    ]
    if any(not isfinite(value) for row in scaled for value in row):
        raise NetworkNumericsError("rank matrix contains a non-finite value")
    columns_data = [
        [scaled[row][column] for row in range(rows)] for column in range(columns)
    ]
    permutation = list(range(columns))
    triangular = [[0.0] * columns for _ in range(columns)]
    initial_norm = max(
        sqrt(fsum(value * value for value in column))
        for column in columns_data
    )
    threshold = rank_relative_tolerance * max(initial_norm, float_info.min)
    pivot_magnitudes: list[float] = []
    rank = 0
    for pivot_index in range(columns):
        selected = max(
            range(pivot_index, columns),
            key=lambda index: fsum(value * value for value in columns_data[index]),
        )
        if selected != pivot_index:
            columns_data[pivot_index], columns_data[selected] = (
                columns_data[selected],
                columns_data[pivot_index],
            )
            permutation[pivot_index], permutation[selected] = (
                permutation[selected],
                permutation[pivot_index],
            )
            for previous in range(pivot_index):
                triangular[previous][pivot_index], triangular[previous][selected] = (
                    triangular[previous][selected],
                    triangular[previous][pivot_index],
                )
        norm = sqrt(
            fsum(value * value for value in columns_data[pivot_index])
        )
        if norm <= threshold:
            break
        rank += 1
        pivot_magnitudes.append(norm)
        triangular[pivot_index][pivot_index] = norm
        basis_column = [value / norm for value in columns_data[pivot_index]]
        for column in range(pivot_index + 1, columns):
            coefficient = fsum(
                direction * value
                for direction, value in zip(
                    basis_column, columns_data[column], strict=True
                )
            )
            columns_data[column] = [
                value - coefficient * direction
                for value, direction in zip(
                    columns_data[column], basis_column, strict=True
                )
            ]
            # A second MGS pass controls loss of orthogonality near the threshold.
            correction = fsum(
                direction * value
                for direction, value in zip(
                    basis_column, columns_data[column], strict=True
                )
            )
            columns_data[column] = [
                value - correction * direction
                for value, direction in zip(
                    columns_data[column], basis_column, strict=True
                )
            ]
            triangular[pivot_index][column] = coefficient + correction
    raw_basis: list[tuple[float, ...]] = []
    for free in range(rank, columns):
        pivoted_direction = [0.0] * columns
        pivoted_direction[free] = 1.0
        for row in range(rank - 1, -1, -1):
            remainder = fsum(
                triangular[row][column] * pivoted_direction[column]
                for column in range(row + 1, columns)
            )
            pivoted_direction[row] = -remainder / triangular[row][row]
        direction = [0.0] * columns
        for pivoted_index, original_index in enumerate(permutation):
            direction[original_index] = pivoted_direction[pivoted_index]
        raw_basis.append(tuple(direction))
    basis = _orthonormalize(raw_basis, rank_relative_tolerance) if raw_basis else ()
    basis_valid, maximum_jv, maximum_orthonormality_error = validate_nullspace_basis(
        matrix,
        variable_scales,
        basis,
        expected_count=columns - rank,
        residual_tolerance=nullspace_residual_tolerance,
        orthonormality_tolerance=orthonormality_tolerance,
    )
    condition = (
        max(pivot_magnitudes) / min(pivot_magnitudes)
        if pivot_magnitudes
        else float_info.max
    )
    return IdentifiabilityDiagnostics(
        numerical_rank=rank,
        state_size=columns,
        nullity=columns - rank,
        structural_rank=structural_rank,
        structural_nullity=structural_nullity,
        condition_estimate=condition,
        rank_relative_tolerance=rank_relative_tolerance,
        nullspace_residual_tolerance=nullspace_residual_tolerance,
        max_nullspace_residual=maximum_jv,
        max_orthonormality_error=maximum_orthonormality_error,
        variable_scales=tuple(float(value) for value in variable_scales),
        nullspace_basis=basis,
        basis_valid=basis_valid,
        expected_rank=rank == structural_rank,
        represented=represented and rank < columns and basis_valid,
    )


def rank_diagnostics(
    state: NetworkState,
    inputs: NetworkInputs,
    bounds: DynamicBounds | None = None,
    *,
    represented: bool = False,
    rank_relative_tolerance: float = 1.0e-11,
    nullspace_residual_tolerance: float = 1.0e-10,
    orthonormality_tolerance: float = 1.0e-10,
) -> IdentifiabilityDiagnostics:
    validate_topology(inputs.topology)
    if state.dimensions != inputs.dimensions:
        raise NetworkValidationError("state and topology dimensions differ")
    if bounds is not None and len(bounds.lower) != inputs.dimensions.state_size:
        raise NetworkValidationError("bounds and state dimensions differ")
    dimensions = inputs.dimensions
    return _rank_and_nullspace(
        analytic_jacobian(state, inputs),
        physical_variable_scales(inputs),
        structural_rank=dimensions.structural_rank,
        structural_nullity=dimensions.structural_nullity,
        represented=represented,
        rank_relative_tolerance=rank_relative_tolerance,
        nullspace_residual_tolerance=nullspace_residual_tolerance,
        orthonormality_tolerance=orthonormality_tolerance,
    )


def _projection(inputs: NetworkInputs, bounds: DynamicBounds) -> ProjectionFunction:
    n = inputs.dimensions.cell_count

    def project(vector: tuple[float, ...]) -> tuple[float, ...]:
        values = list(vector)
        # Isotonic (pool-adjacent-violators) projection; ``sorted`` permutes
        # variable identities and stalled the accepted four-cell solver at
        # 1000 V (see global-plasma-closure-analysis.md, 2026-09-03).
        ordered_phi = list(project_nondecreasing(values[0:n]))
        ordered_phi[-1] = max(ordered_phi[-1], inputs.anode_voltage_v)
        values[0:n] = ordered_phi
        values[5 * n] = max(values[5 * n], -values[5 * n + 1])
        return tuple(
            min(max(value, low), high)
            for value, low, high in zip(values, bounds.lower, bounds.upper, strict=True)
        )

    return project


def representative_initial_state(inputs: NetworkInputs) -> NetworkState:
    validate_topology(inputs.topology)
    n = inputs.dimensions.cell_count
    voltage = inputs.anode_voltage_v
    current = inputs.anode_current_a
    phi = tuple(
        (0.02 + 0.98 * index / max(n - 1, 1)) * voltage for index in range(n)
    )
    if n == 1:
        phi = (voltage,)
    temperature = tuple(0.05 * voltage for _ in range(n))
    source = tuple(0.25 * current / n for _ in range(n))
    electron = tuple((0.1 + 0.9 * index / n) * current for index in range(n + 1))
    ion = tuple(current - value for value in electron)
    probability = inputs.arrival_probabilities
    cusp = tuple(electron[index] * probability[index] for index in range(n - 1))
    return NetworkState(phi, temperature, source, electron, ion, cusp)


def deterministic_initial_states(
    inputs: NetworkInputs,
    *,
    count: int = 5,
    seed: NetworkState | None = None,
) -> tuple[NetworkState, ...]:
    validate_topology(inputs.topology)
    if count < 1 or count > 9:
        raise NetworkValidationError("count must be in [1, 9]")
    base_state = representative_initial_state(inputs) if seed is None else seed
    if base_state.dimensions != inputs.dimensions:
        raise NetworkValidationError("seed and topology dimensions differ")
    base = base_state.to_vector()
    n = inputs.dimensions.cell_count
    patterns = (
        (1.00, 1.00, 1.00),
        (0.85, 0.75, 0.80),
        (1.15, 1.25, 1.20),
        (0.70, 1.20, 0.90),
        (1.25, 0.85, 1.10),
        (0.90, 1.35, 0.70),
        (1.10, 0.65, 1.30),
        (0.75, 0.95, 1.25),
        (1.20, 1.10, 0.75),
    )
    bounds = default_bounds(inputs)
    project = _projection(inputs, bounds)
    states: list[NetworkState] = []
    for potential_factor, temperature_factor, current_factor in patterns[:count]:
        values = list(base)
        for index in range(n):
            values[index] *= potential_factor
            values[n + index] *= temperature_factor
        for index in range(2 * n, len(values)):
            values[index] *= current_factor
        states.append(NetworkState.from_vector(project(tuple(values)), n))
    return tuple(states)


def _conservation_inf_norm(
    normalized: Sequence[float], inputs: NetworkInputs
) -> float:
    conservation_families = {
        "electron_continuity",
        "ion_continuity",
        "interface_current",
        "cusp_loss",
        "cell_energy",
        "global_energy",
    }
    ledger = generate_equation_ledger(inputs.topology)
    values = tuple(
        abs(normalized[index])
        for index, row in enumerate(ledger)
        if row.family in conservation_families
    )
    return max(values, default=0.0)


def _failed_result(
    *,
    reason: str,
    start_index: int,
    backend_result: LeastSquaresResult | None,
) -> NetworkSolveResult:
    return NetworkSolveResult(
        state=None,
        evaluation=None,
        diagnostics=NetworkSolveDiagnostics(
            numerical_converged=False,
            published=False,
            reason=reason,
            residual_inf_norm=float_info.max,
            conservation_inf_norm=float_info.max,
            feasible=False,
            deterministic_start_index=start_index,
            equation_residuals=(),
            identifiability=None,
            backend=None if backend_result is None else backend_result.diagnostics,
        ),
    )


def solve_network(
    inputs: NetworkInputs,
    initial_state: NetworkState | None = None,
    *,
    bounds: DynamicBounds | None = None,
    options: NetworkSolverOptions | None = None,
    backend: LeastSquaresBackend | None = None,
    start_index: int = 0,
) -> NetworkSolveResult:
    """Treat the backend as an untrusted candidate generator, then verify canonically."""

    validate_topology(inputs.topology)
    selected_options = NetworkSolverOptions() if options is None else options
    selected_bounds = default_bounds(inputs) if bounds is None else bounds
    if len(selected_bounds.lower) != inputs.dimensions.state_size:
        raise NetworkValidationError("bounds and topology dimensions differ")
    initial = representative_initial_state(inputs) if initial_state is None else initial_state
    if initial.dimensions != inputs.dimensions:
        raise NetworkValidationError("initial state and topology dimensions differ")
    selected_backend = ScaledQrLmBackend() if backend is None else backend
    if not isinstance(selected_backend, LeastSquaresBackend):
        raise NetworkValidationError("backend must implement LeastSquaresBackend")
    n = inputs.dimensions.cell_count

    def function(vector: tuple[float, ...]) -> Sequence[float]:
        return evaluate_residual(NetworkState.from_vector(vector, n), inputs).normalized

    def matrix(vector: tuple[float, ...]) -> Sequence[Sequence[float]]:
        return analytic_jacobian(NetworkState.from_vector(vector, n), inputs)

    def feasible(vector: tuple[float, ...]) -> bool:
        try:
            return is_feasible(NetworkState.from_vector(vector, n), inputs, selected_bounds)
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            return False

    solved_object = selected_backend.solve(
        function,
        initial.to_vector(),
        selected_bounds,
        matrix,
        feasible,
        _projection(inputs, selected_bounds),
        selected_options,
    )
    if not isinstance(solved_object, LeastSquaresResult):
        return _failed_result(
            reason="backend_result_contract_invalid",
            start_index=start_index,
            backend_result=None,
        )
    solved = solved_object
    if solved.vector is None:
        return _failed_result(
            reason="backend_returned_no_candidate",
            start_index=start_index,
            backend_result=solved,
        )
    try:
        candidate = NetworkState.from_vector(solved.vector, n)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return _failed_result(
            reason="backend_candidate_invalid",
            start_index=start_index,
            backend_result=solved,
        )
    vector = candidate.to_vector()
    in_bounds = all(
        low <= value <= high
        for value, low, high in zip(
            vector, selected_bounds.lower, selected_bounds.upper, strict=True
        )
    )
    try:
        evaluation = evaluate_residual(candidate, inputs)
        feasible_state = in_bounds and is_feasible(candidate, inputs, selected_bounds)
        residual_norm = max(abs(value) for value in evaluation.normalized)
        conservation_norm = _conservation_inf_norm(evaluation.normalized, inputs)
        represented = (
            selected_options.publication_policy is PublicationPolicy.REPRESENT_NULLSPACE
        )
        identity = rank_diagnostics(
            candidate,
            inputs,
            represented=represented,
            rank_relative_tolerance=selected_options.rank_relative_tolerance,
            nullspace_residual_tolerance=selected_options.nullspace_residual_tolerance,
            orthonormality_tolerance=(
                selected_options.nullspace_orthonormality_tolerance
            ),
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return _failed_result(
            reason="canonical_candidate_evaluation_failed",
            start_index=start_index,
            backend_result=solved,
        )
    residual_ok = residual_norm <= selected_options.least_squares.residual_tolerance
    conservation_ok = conservation_norm <= selected_options.conservation_tolerance
    canonical_converged = feasible_state and residual_ok and conservation_ok
    expected_rank = identity.expected_rank
    if identity.numerical_rank == identity.state_size:
        rank_publishable = True
    elif selected_options.publication_policy is PublicationPolicy.REQUIRE_FULL_RANK:
        rank_publishable = identity.numerical_rank == identity.state_size
    else:
        rank_publishable = (
            identity.nullity == identity.structural_nullity
            and identity.basis_valid
            and identity.represented
        )
    published = canonical_converged and expected_rank and rank_publishable
    if not in_bounds:
        reason = "canonical_bounds_failed"
    elif not feasible_state:
        reason = "canonical_inequalities_failed"
    elif not residual_ok:
        reason = "canonical_residual_tolerance_failed"
    elif not conservation_ok:
        reason = "canonical_conservation_tolerance_failed"
    elif not expected_rank:
        reason = "unexpected_jacobian_rank"
    elif not rank_publishable:
        reason = "rank_deficient_publication_prohibited"
    else:
        reason = "canonical_publication_gates_passed"
    equation_ids = tuple(row.row_id for row in generate_equation_ledger(inputs.topology))
    diagnostics = NetworkSolveDiagnostics(
        numerical_converged=canonical_converged,
        published=published,
        reason=reason,
        residual_inf_norm=residual_norm,
        conservation_inf_norm=conservation_norm,
        feasible=feasible_state,
        deterministic_start_index=start_index,
        equation_residuals=tuple(
            zip(equation_ids, evaluation.normalized, strict=True)
        ),
        identifiability=identity,
        backend=solved.diagnostics,
    )
    return NetworkSolveResult(
        state=candidate if published else None,
        evaluation=evaluation if published else None,
        diagnostics=diagnostics,
    )


def solve_network_multistart(
    inputs: NetworkInputs,
    initial_states: Sequence[NetworkState] | None = None,
    *,
    start_count: int = 5,
    seed: NetworkState | None = None,
    bounds: DynamicBounds | None = None,
    options: NetworkSolverOptions | None = None,
    backend: LeastSquaresBackend | None = None,
) -> NetworkMultiStartResult:
    validate_topology(inputs.topology)
    states = (
        deterministic_initial_states(inputs, count=start_count, seed=seed)
        if initial_states is None
        else tuple(initial_states)
    )
    if len(states) == 0:
        raise NetworkValidationError("initial_states must not be empty")
    attempts = tuple(
        solve_network(
            inputs,
            state,
            bounds=bounds,
            options=options,
            backend=backend,
            start_index=index,
        )
        for index, state in enumerate(states)
    )
    published = [
        (index, attempt)
        for index, attempt in enumerate(attempts)
        if attempt.diagnostics.published
    ]
    candidates = published if published else list(enumerate(attempts))
    selected_index, best = min(
        candidates,
        key=lambda item: (item[1].diagnostics.residual_inf_norm, item[0]),
    )
    return NetworkMultiStartResult(
        best=best,
        attempts=attempts,
        selected_start_index=selected_index,
        residual_floor=min(
            attempt.diagnostics.residual_inf_norm for attempt in attempts
        ),
    )
