"""Deterministic bounded least-squares solve of the v2 system.

The Levenberg-Marquardt / pivoted-QR machinery of the v1 package
(``cft_revival.plasma.solve_bounded_least_squares``,
``project_nondecreasing``) is imported read-only.  Publication is
fail-closed exactly as in v1: every normalized row must meet tolerance and
the state must pass finite, box and nonlinear feasibility checks; otherwise
the residual floor and the reason are reported and no state is published.
"""

from __future__ import annotations

from sys import float_info
from typing import Sequence

from cft_revival.plasma import (
    PlasmaNumericsError,
    PlasmaState,
    PlasmaValidationError,
    SolverOptions,
    potential_parametrized_state,
    project_nondecreasing,
    representative_initial_state,
    solve_bounded_least_squares,
)

from .manifold import closure_probabilities, reduced_solve, sheath_drops
from .models import (
    CORE_STATE_SIZE,
    RESIDUAL_SIZE,
    STATE_SIZE,
    AnodeRow,
    CuspLossClosure,
    RankReport,
    SheathClosureInputs,
    SheathClosureState,
    SheathMultiStartResult,
    SheathSolveResult,
    SolverPolicy,
)
from .residuals import (
    analytic_jacobian,
    default_state_bounds,
    evaluate_residual,
    is_feasible,
    matrix_rank,
)


def representative_initial_state_v2(inputs: SheathClosureInputs) -> SheathClosureState:
    """Deterministic source-scale seed: v1 representative core + closure guesses."""

    probability = (
        inputs.declared_cusp_probabilities
        if inputs.cusp_loss_closure is CuspLossClosure.CL1_DECLARED
        else closure_probabilities(inputs, None)
        if inputs.cusp_loss_closure is CuspLossClosure.CL3_SHEATH_LIMITED
        else (0.05, 0.05, 0.05)
    )
    core = representative_initial_state(inputs.v1_inputs((*probability, inputs.anode_cusp_probability)))
    values = list(core.to_vector())
    # The anode sheath row needs a strictly positive anode ion current.
    values[21] = min(values[21], -1.0e-3 * inputs.anode_current_a)
    values[16] = inputs.anode_current_a - values[21]
    core = PlasmaState.from_vector(values)
    return SheathClosureState(core, sheath_drops(inputs, core.electron_temperature_ev), probability)


def deterministic_initial_states(
    inputs: SheathClosureInputs, *, count: int = 5
) -> tuple[SheathClosureState, ...]:
    """Fixed reproducible spread around the representative seed (v1 patterns)."""

    if count <= 0 or count > 9:
        raise PlasmaValidationError("count must be in [1, 9]")
    base = representative_initial_state_v2(inputs).to_vector()
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
    states: list[SheathClosureState] = []
    for potential_factor, temperature_factor, current_factor in patterns[:count]:
        values = list(base)
        values[0] *= potential_factor
        values[1] = inputs.anode_voltage_v - (inputs.anode_voltage_v - values[1]) * potential_factor
        for index in range(4, 8):
            values[index] *= temperature_factor
        for index in (*range(8, 17), *range(22, 25)):
            values[index] *= current_factor
        for index in range(25, 28):
            values[index] *= temperature_factor
        states.append(SheathClosureState.from_vector(values))
    return tuple(states)


def _projection(inputs: SheathClosureInputs, lower: tuple[float, ...], upper: tuple[float, ...]):
    def project(vector: tuple[float, ...]) -> tuple[float, ...]:
        values = list(vector)
        ordered = list(project_nondecreasing(values[0:4]))
        ordered[3] = max(ordered[3], inputs.anode_voltage_v)
        values[0:4] = ordered
        if inputs.potentials.anode_row is AnodeRow.SHEATH:
            values[21] = min(values[21], -1.0e-12 * inputs.anode_current_a)
        values[20] = max(values[20], -values[21])
        return tuple(
            min(max(value, low), high) for value, low, high in zip(values, lower, upper, strict=True)
        )

    return project


def solve_sheath_closure(
    inputs: SheathClosureInputs,
    initial_state: SheathClosureState | None = None,
    *,
    options: SolverOptions | None = None,
    policy: SolverPolicy | None = None,
    use_analytic_jacobian: bool = True,
) -> SheathSolveResult:
    """One deterministic bounded LM solve from one start (fail-closed)."""

    selected_policy = SolverPolicy() if policy is None else policy
    lower, upper = default_state_bounds(inputs)
    seeded = False
    if initial_state is None:
        if selected_policy.seed_from_manifold:
            reduced = reduced_solve(inputs)
            if reduced.state is not None:
                initial_state = reduced.state
                seeded = True
        if initial_state is None:
            initial_state = representative_initial_state_v2(inputs)
    enforce = selected_policy.enforce_cusp_energy_margin

    def function(vector: tuple[float, ...]) -> Sequence[float]:
        return evaluate_residual(SheathClosureState.from_vector(vector), inputs).normalized

    def matrix(vector: tuple[float, ...]) -> Sequence[Sequence[float]]:
        return analytic_jacobian(SheathClosureState.from_vector(vector), inputs)

    def feasible(vector: tuple[float, ...]) -> bool:
        try:
            return is_feasible(
                SheathClosureState.from_vector(vector),
                inputs,
                (lower, upper),
                enforce_cusp_energy_margin=enforce,
            )
        except (ArithmeticError, ValueError, OverflowError):
            return False

    solved = solve_bounded_least_squares(
        function,
        initial_state.to_vector(),
        lower,
        upper,
        jacobian=matrix if use_analytic_jacobian else None,
        feasible=feasible,
        project=_projection(inputs, lower, upper),
        options=options,
    )
    if not solved.diagnostics.converged or solved.vector is None:
        return SheathSolveResult(None, None, solved.diagnostics, seeded)
    state = SheathClosureState.from_vector(solved.vector)
    evaluation = evaluate_residual(state, inputs)
    if not is_feasible(state, inputs, (lower, upper), enforce_cusp_energy_margin=enforce):
        raise PlasmaNumericsError("solver attempted to publish an infeasible converged state")
    return SheathSolveResult(state, evaluation, solved.diagnostics, seeded)


def solve_sheath_closure_multistart(
    inputs: SheathClosureInputs,
    initial_states: Sequence[SheathClosureState] | None = None,
    *,
    start_count: int = 5,
    options: SolverOptions | None = None,
    policy: SolverPolicy | None = None,
    use_analytic_jacobian: bool = True,
) -> SheathMultiStartResult:
    """Manifold seed (if it exists) plus deterministic spread; keep the best strict solution."""

    selected_policy = SolverPolicy() if policy is None else policy
    if initial_states is None:
        starts: list[SheathClosureState | None] = []
        if selected_policy.seed_from_manifold:
            starts.append(None)  # resolved inside solve_sheath_closure
        starts.extend(deterministic_initial_states(inputs, count=start_count))
    else:
        starts = list(initial_states)
    if not starts:
        raise PlasmaValidationError("initial_states must not be empty")
    attempts = tuple(
        solve_sheath_closure(
            inputs,
            start,
            options=options,
            policy=selected_policy,
            use_analytic_jacobian=use_analytic_jacobian,
        )
        for start in starts
    )
    converged = [(index, attempt) for index, attempt in enumerate(attempts) if attempt.diagnostics.converged]
    candidates = converged if converged else list(enumerate(attempts))
    selected_index, best = min(
        candidates, key=lambda item: (item[1].diagnostics.residual_inf_norm, item[0])
    )
    floor = min(attempt.diagnostics.residual_inf_norm for attempt in attempts)
    return SheathMultiStartResult(best, attempts, selected_index, floor)


def manifold_projection(state: SheathClosureState, inputs: SheathClosureInputs) -> SheathClosureState:
    """Return the exact R00-R30 manifold point with the state's potentials and probabilities.

    Used before rank evaluation: the identity ``R27 == 0`` makes the R27
    gradient a combination of the other gradients only ON the manifold; a
    least-squares state a few 1e-12 off it would otherwise read one rank too
    high.
    """

    core = potential_parametrized_state(
        inputs.v1_inputs((*state.cusp_probability, inputs.anode_cusp_probability)),
        state.core.plasma_potential_v,
    )
    return SheathClosureState(core, sheath_drops(inputs, core.electron_temperature_ev), state.cusp_probability)


def rank_report(state: SheathClosureState, inputs: SheathClosureInputs) -> RankReport:
    """Rank of the scaled Jacobian, block by block, at the manifold projection of a state.

    * ``rank_corrected_core``: rows R00-R27 over the 25 core columns (v1 layout
      with the corrected R27); 21 = the rank the closure analysis predicts
      once R27 is an identity (nullity 4: all potentials free).
    * ``rank_with_sheath_and_anode``: rows R00-R31 and R35-R37 over all 31
      columns; the sheath rows identify their own new unknowns and the anode
      row identifies one potential relation, so the nullity is 3.
    * ``rank_full``: all 38 rows; full column rank 31 when the three declared
      potential relations R32-R34 are present.
    """

    state = manifold_projection(state, inputs)
    jacobian = analytic_jacobian(state, inputs)
    lower, upper = default_state_bounds(inputs)
    vector = state.to_vector()
    scale = tuple(
        max(abs(value), 0.25 * (high - low), float_info.min)
        for value, low, high in zip(vector, lower, upper, strict=True)
    )
    scaled = [[jacobian[row][column] * scale[column] for column in range(STATE_SIZE)] for row in range(RESIDUAL_SIZE)]
    rank_full, condition = matrix_rank(scaled)
    core_block = [row[:CORE_STATE_SIZE] for row in scaled[:28]]
    rank_core, _ = matrix_rank(core_block)
    without_closure = [scaled[row] for row in range(RESIDUAL_SIZE) if row not in (32, 33, 34)]
    rank_partial, _ = matrix_rank(without_closure)
    closure = inputs.potentials
    declared: list[str] = [
        f"phi_3 - phi_2 = {closure.interior_step_3_v:g} V (R32)",
        f"phi_4 - phi_3 = {closure.interior_step_4_v:g} V (R33)",
    ]
    if closure.anode_row is AnodeRow.DECLARED_FALL:
        declared.append(f"phi_4 - Ua = {closure.anode_fall_v:g} V (R31 declared)")
    if closure.fourth_row.value == "anode_fall_declared":
        declared.append(f"phi_4 - Ua = {closure.anode_fall_v:g} V (R34)")
    else:
        declared.append(f"phi_1 - phi_0 = {closure.cathode_coupling_v:g} V (R34)")
    return RankReport(
        rows=RESIDUAL_SIZE,
        unknowns=STATE_SIZE,
        rank_full=rank_full,
        rank_corrected_core=rank_core,
        rank_with_sheath_and_anode=rank_partial,
        nullity_before_potential_closure=STATE_SIZE - rank_partial,
        solved_potential=closure.solved_potential,
        declared_relations=tuple(declared),
        condition_estimate=condition,
    )
