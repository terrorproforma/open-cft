from math import isfinite

import pytest

from cft_revival.plasma import (
    PlasmaValidationError,
    SolverOptions,
    XenonGlobalInputs,
    project_nondecreasing,
    solve_bounded_least_squares,
    solve_global_discharge,
    solve_global_discharge_multistart,
)


@pytest.mark.parametrize("initial", [(-9.0, 9.0), (0.0, 0.0), (9.0, -9.0)])
def test_manufactured_linear_balance_converges_from_multiple_guesses(initial) -> None:
    # Manufactured particle and energy closures with deliberately different raw scales.
    def residual(vector):
        particle, energy = vector
        return ((particle - 2.0), (1.0e9 * energy - 3.0e9) / 1.0e9)

    def jacobian(_vector):
        return ((1.0, 0.0), (0.0, 1.0))

    result = solve_bounded_least_squares(
        residual,
        initial,
        (-10.0, -10.0),
        (10.0, 10.0),
        jacobian=jacobian,
        options=SolverOptions(
            residual_tolerance=1.0e-11,
            gradient_tolerance=1.0e-14,
        ),
    )
    assert result.diagnostics.converged
    assert result.vector == pytest.approx((2.0, 3.0), abs=1.0e-10)
    assert result.diagnostics.reason == "residual_tolerance"
    assert result.diagnostics.finite
    assert result.diagnostics.jacobian_rank == 2
    assert isfinite(result.diagnostics.jacobian_condition_estimate)
    assert len(result.diagnostics.normalized_residuals) == 2


def test_manufactured_nonlinear_balance_uses_true_bounds() -> None:
    result = solve_bounded_least_squares(
        lambda vector: (vector[0] * vector[0] - 4.0,),
        (0.25,),
        (0.0,),
        (5.0,),
        jacobian=lambda vector: ((2.0 * vector[0],),),
        options=SolverOptions(
            residual_tolerance=1.0e-10,
            gradient_tolerance=1.0e-15,
        ),
    )
    assert result.diagnostics.converged
    assert result.vector == pytest.approx((2.0,), abs=1.0e-10)
    assert 0.0 <= result.vector[0] <= 5.0


def test_finite_difference_solver_matches_analytic_solver() -> None:
    kwargs = dict(
        function=lambda vector: (vector[0] - 0.125, 2.0 * vector[1] + 1.0),
        initial=(0.9, 0.9),
        lower=(-1.0, -1.0),
        upper=(1.0, 1.0),
        options=SolverOptions(residual_tolerance=1.0e-10),
    )
    finite_difference = solve_bounded_least_squares(**kwargs)
    analytic = solve_bounded_least_squares(
        **kwargs, jacobian=lambda _vector: ((1.0, 0.0), (0.0, 2.0))
    )
    assert finite_difference.diagnostics.converged
    assert analytic.diagnostics.converged
    assert finite_difference.vector == pytest.approx(analytic.vector, abs=2.0e-9)


def test_nonconvergence_is_fail_closed_and_deterministic() -> None:
    arguments = dict(
        function=lambda _vector: (1.0,),
        initial=(0.5,),
        lower=(0.0,),
        upper=(1.0,),
        jacobian=lambda _vector: ((0.0,),),
        options=SolverOptions(max_iterations=4),
    )
    first = solve_bounded_least_squares(**arguments)
    second = solve_bounded_least_squares(**arguments)
    assert first == second
    assert first.vector is None
    assert not first.diagnostics.converged
    assert first.diagnostics.reason == "gradient_tolerance_without_balance"
    assert first.diagnostics.finite


def test_nonfinite_callback_never_publishes_a_state() -> None:
    result = solve_bounded_least_squares(
        lambda _vector: (float("inf"),),
        (0.5,),
        (0.0,),
        (1.0,),
    )
    assert result.vector is None
    assert result.residual is None
    assert result.diagnostics.reason == "nonfinite_initial_evaluation"
    assert result.diagnostics.finite
    assert all(
        isfinite(value)
        for value in (
            result.diagnostics.initial_cost,
            result.diagnostics.final_cost,
            result.diagnostics.residual_inf_norm,
        )
    )


@pytest.mark.parametrize(
    "projection,match",
    [
        (lambda _vector: (), "unchanged length"),
        (lambda _vector: (float("nan"),), "finite"),
        (lambda _vector: (2.0,), "outside"),
    ],
)
def test_public_projection_contract_rejects_invalid_results(
    projection, match
) -> None:
    with pytest.raises(PlasmaValidationError, match=match):
        solve_bounded_least_squares(
            lambda vector: (vector[0] - 0.5,),
            (0.5,),
            (0.0,),
            (1.0,),
            project=projection,
        )


def test_projection_receives_clipped_state_and_result_is_revalidated() -> None:
    observed = []

    def projection(vector):
        observed.append(vector)
        return vector

    result = solve_bounded_least_squares(
        lambda vector: (vector[0] - 1.0,),
        (2.0,),
        (0.0,),
        (1.0,),
        project=projection,
        jacobian=lambda _vector: ((1.0,),),
    )
    assert observed[0] == (1.0,)
    assert result.diagnostics.converged
    assert result.vector == (1.0,)


def test_published_rounded_case_is_external_evidence_not_false_convergence(
    dm92_published_state, dm92_inputs
) -> None:
    result = solve_global_discharge(
        dm92_inputs,
        dm92_published_state,
        options=SolverOptions(max_iterations=2, residual_tolerance=1.0e-8),
    )
    assert result.state is None
    assert result.evaluation is None
    assert not result.diagnostics.converged
    assert result.diagnostics.residual_inf_norm > 1.0e-8
    assert result.diagnostics.feasible
    assert len(result.diagnostics.normalized_residuals) == 28
    assert result.diagnostics.jacobian_rank == 22


def test_self_consistent_global_case_has_strict_closure_and_rank_status(
    self_consistent_case,
) -> None:
    inputs, state = self_consistent_case
    result = solve_global_discharge(
        inputs,
        state,
        options=SolverOptions(residual_tolerance=1.0e-12),
    )
    assert result.state == state
    assert result.evaluation is not None
    assert result.diagnostics.converged
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.residual_inf_norm < 1.0e-15
    assert result.diagnostics.jacobian_rank == 22
    assert result.evaluation.powers.closure_w == pytest.approx(0.0, abs=1.0e-12)


def test_deterministic_multistart_selects_strict_solution_and_retains_attempts(
    self_consistent_case,
) -> None:
    inputs, state = self_consistent_case
    result = solve_global_discharge_multistart(
        inputs,
        [state, state],
        options=SolverOptions(residual_tolerance=1.0e-12),
    )
    repeated = solve_global_discharge_multistart(
        inputs,
        [state, state],
        options=SolverOptions(residual_tolerance=1.0e-12),
    )
    assert result == repeated
    assert result.selected_start_index == 0
    assert len(result.attempts) == 2
    assert result.best.state == state
    assert result.residual_floor < 1.0e-15


@pytest.mark.parametrize(
    "values,expected",
    [
        ((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0, 4.0)),
        ((4.0, 3.0, 2.0, 1.0), (2.5, 2.5, 2.5, 2.5)),
        ((1.0, 3.0, 2.0, 4.0), (1.0, 2.5, 2.5, 4.0)),
        ((0.0, 10.0, 9.0, 8.0), (0.0, 9.0, 9.0, 9.0)),
        ((5.0,), (5.0,)),
    ],
)
def test_nondecreasing_projection_pools_adjacent_violators(values, expected) -> None:
    projected = project_nondecreasing(values)
    assert projected == pytest.approx(expected)
    assert all(a <= b for a, b in zip(projected, projected[1:], strict=False))
    assert project_nondecreasing(projected) == pytest.approx(projected)
    # Euclidean projection: never farther from the input than the sorted permutation.
    sorted_distance = sum((a - b) ** 2 for a, b in zip(values, sorted(values), strict=True))
    projected_distance = sum((a - b) ** 2 for a, b in zip(values, projected, strict=True))
    assert projected_distance <= sorted_distance + 1.0e-12


def test_nondecreasing_projection_rejects_nonfinite_values() -> None:
    with pytest.raises(PlasmaValidationError, match="finite"):
        project_nondecreasing((1.0, float("nan")))


@pytest.mark.parametrize("voltage", [150.0, 1000.0])
@pytest.mark.parametrize("current", [0.1, 1.0, 3.0])
def test_zero_cusp_grid_closes_including_the_1000_v_cases(voltage, current) -> None:
    # Before the isotonic projection the sort-based projection stalled the
    # 1000 V / {0.1, 1, 3} A zero-cusp solves (13/16 of the 2026-09-03 probe grid).
    inputs = XenonGlobalInputs(voltage, current, (0.0, 0.0, 0.0, 0.0))
    options = SolverOptions(residual_tolerance=1.0e-8)
    result = solve_global_discharge_multistart(inputs, start_count=3, options=options)
    assert result.best.state is not None
    assert result.best.diagnostics.converged
    assert result.residual_floor <= 1.0e-8
    assert result.best.diagnostics.jacobian_rank == 22
    # Every published zero-cusp state sits on the phi_4 = Ua boundary: the
    # global row reduces to 2*(j_e3+I4)*(phi_4-Ua) there.
    assert result.best.state.plasma_potential_v[3] == pytest.approx(voltage, abs=1.0e-6 * voltage)
    repeated = solve_global_discharge_multistart(inputs, start_count=3, options=options)
    assert repeated == result
