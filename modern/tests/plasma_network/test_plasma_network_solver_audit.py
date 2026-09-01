from dataclasses import replace
from math import isfinite

import pytest

from cft_revival.plasma import SolverOptions
from cft_revival.plasma.solver import LeastSquaresResult
from cft_revival.plasma_network import (
    DynamicBounds,
    NetworkInputs,
    NetworkSolverOptions,
    NetworkState,
    NetworkValidationError,
    PublicationPolicy,
    analytic_jacobian,
    evaluate_residual,
    make_chain_topology,
    physical_variable_scales,
    manufactured_zero_cusp_case,
    provenance_hash,
    rank_diagnostics,
    solve_network,
    solve_network_multistart,
    validate_nullspace_basis,
)


class FixedBackend:
    def __init__(self, result: LeastSquaresResult) -> None:
        self.result = result

    def solve(self, *_args, **_kwargs) -> LeastSquaresResult:
        return self.result


@pytest.mark.parametrize("cell_count", range(1, 7))
def test_rank_and_nullspace_dimensions_are_reported(cell_count: int) -> None:
    case = manufactured_zero_cusp_case(cell_count)
    diagnostics = rank_diagnostics(case.state, case.inputs, represented=True)
    assert diagnostics.numerical_rank == 5 * cell_count + 2
    assert diagnostics.state_size == 6 * cell_count + 1
    assert diagnostics.nullity == cell_count - 1
    assert diagnostics.structural_rank == 5 * cell_count + 2
    assert diagnostics.structural_nullity == cell_count - 1
    assert len(diagnostics.nullspace_basis) == cell_count - 1
    assert all(
        len(direction) == diagnostics.state_size
        for direction in diagnostics.nullspace_basis
    )
    jacobian = analytic_jacobian(case.state, case.inputs)
    scales = physical_variable_scales(case.inputs)
    assert all(
        max(
            abs(
                sum(
                    row[index] * scales[index] * direction[index]
                    for index in range(len(direction))
                )
            )
            for row in jacobian
        )
        < 1.0e-12
        for direction in diagnostics.nullspace_basis
    )
    assert diagnostics.basis_valid
    assert diagnostics.expected_rank
    assert diagnostics.max_nullspace_residual < 1.0e-12
    assert diagnostics.max_orthonormality_error < 1.0e-12
    assert isfinite(diagnostics.condition_estimate)


@pytest.mark.parametrize("cell_count", (2, 3, 4, 5, 6))
def test_rank_deficiency_prohibits_publication_by_default(cell_count: int) -> None:
    case = manufactured_zero_cusp_case(cell_count)
    result = solve_network(
        case.inputs,
        case.state,
        options=NetworkSolverOptions(
            least_squares=SolverOptions(residual_tolerance=1.0e-12)
        ),
    )
    assert result.diagnostics.numerical_converged
    assert not result.diagnostics.published
    assert result.state is None
    assert result.evaluation is None
    assert result.diagnostics.reason == "rank_deficient_publication_prohibited"
    assert not result.diagnostics.identifiability.represented


@pytest.mark.parametrize("cell_count", (1, 2, 3))
def test_manufactured_n1_to_n3_strictly_converge_and_publish_with_nullspace(
    cell_count: int,
) -> None:
    case = manufactured_zero_cusp_case(cell_count)
    result = solve_network(
        case.inputs,
        case.state,
        options=NetworkSolverOptions(
            least_squares=SolverOptions(residual_tolerance=1.0e-12),
            publication_policy=PublicationPolicy.REPRESENT_NULLSPACE,
        ),
    )
    assert result.diagnostics.numerical_converged
    assert result.diagnostics.published
    assert result.state == case.state
    assert result.evaluation is not None
    assert result.diagnostics.residual_inf_norm < 3.0e-15
    assert abs(result.evaluation.powers.closure_w) < 1.0e-8
    assert result.diagnostics.identifiability.nullity == cell_count - 1
    if cell_count > 1:
        assert result.diagnostics.identifiability.represented


def test_full_rank_n1_publishes_under_default_rank_policy() -> None:
    case = manufactured_zero_cusp_case(1)
    result = solve_network(
        case.inputs,
        case.state,
        options=NetworkSolverOptions(
            least_squares=SolverOptions(residual_tolerance=1.0e-12)
        ),
    )
    assert result.diagnostics.published
    assert not result.diagnostics.identifiability.rank_deficient


def test_deterministic_multistart_identity_and_attempt_ledger() -> None:
    case = manufactured_zero_cusp_case(3)
    options = NetworkSolverOptions(
        least_squares=SolverOptions(residual_tolerance=1.0e-12),
        publication_policy=PublicationPolicy.REPRESENT_NULLSPACE,
    )
    first = solve_network_multistart(
        case.inputs, (case.state, case.state), options=options
    )
    second = solve_network_multistart(
        case.inputs, (case.state, case.state), options=options
    )
    assert first == second
    assert first.selected_start_index == 0
    assert first.best.state == case.state
    assert first.residual_floor < 3.0e-15
    assert len(first.attempts) == 2
    assert all(
        len(attempt.diagnostics.equation_residuals) == 21
        for attempt in first.attempts
    )


def test_nonconverged_network_is_fail_closed() -> None:
    case = manufactured_zero_cusp_case(2)
    perturbed = replace(
        case.state,
        ionization_source_current_a=tuple(
            value * 0.5 for value in case.state.ionization_source_current_a
        ),
    )
    result = solve_network(
        case.inputs,
        perturbed,
        options=NetworkSolverOptions(
            least_squares=SolverOptions(max_iterations=1, residual_tolerance=1.0e-15),
            publication_policy=PublicationPolicy.REPRESENT_NULLSPACE,
        ),
    )
    assert not result.diagnostics.published
    assert result.state is None
    assert result.evaluation is None


def test_backend_reported_zero_cannot_hide_actual_point_426_residual() -> None:
    case = manufactured_zero_cusp_case(1)
    honest = solve_network(case.inputs, case.state)
    assert honest.diagnostics.backend is not None
    values = list(case.state.to_vector())
    values[4] += 0.426 * case.inputs.anode_current_a
    lie = LeastSquaresResult(
        vector=tuple(values),
        residual=(0.0,) * 7,
        diagnostics=replace(
            honest.diagnostics.backend,
            converged=True,
            reason="malicious_zero",
            residual_inf_norm=0.0,
            normalized_residuals=(0.0,) * 7,
        ),
    )
    result = solve_network(
        case.inputs,
        case.state,
        backend=FixedBackend(lie),
    )
    assert result.state is None
    assert result.evaluation is None
    assert not result.diagnostics.numerical_converged
    assert result.diagnostics.reason == "canonical_residual_tolerance_failed"
    assert result.diagnostics.residual_inf_norm == pytest.approx(0.426)
    assert result.diagnostics.equation_residuals[4][0] == "R04"
    assert result.diagnostics.equation_residuals[4][1] == pytest.approx(0.426)


def test_canonical_conservation_tolerance_is_an_independent_gate() -> None:
    case = manufactured_zero_cusp_case(1)
    honest = solve_network(case.inputs, case.state)
    assert honest.diagnostics.backend is not None
    values = list(case.state.to_vector())
    values[4] += 0.426 * case.inputs.anode_current_a
    candidate = LeastSquaresResult(
        tuple(values), (0.0,) * 7, honest.diagnostics.backend
    )
    result = solve_network(
        case.inputs,
        case.state,
        backend=FixedBackend(candidate),
        options=NetworkSolverOptions(
            least_squares=SolverOptions(residual_tolerance=0.5),
            conservation_tolerance=1.0e-3,
        ),
    )
    assert result.diagnostics.residual_inf_norm == pytest.approx(0.426)
    assert result.diagnostics.conservation_inf_norm == pytest.approx(0.426)
    assert result.diagnostics.reason == "canonical_conservation_tolerance_failed"
    assert result.state is None


def test_backend_status_and_residual_are_ignored_for_exact_candidate() -> None:
    case = manufactured_zero_cusp_case(1)
    honest = solve_network(case.inputs, case.state)
    assert honest.diagnostics.backend is not None
    lie = LeastSquaresResult(
        vector=case.state.to_vector(),
        residual=(99.0,) * 7,
        diagnostics=replace(
            honest.diagnostics.backend,
            converged=False,
            reason="malicious_failure",
            residual_inf_norm=99.0,
            normalized_residuals=(99.0,) * 7,
        ),
    )
    result = solve_network(case.inputs, case.state, backend=FixedBackend(lie))
    assert result.diagnostics.published
    assert result.state == case.state
    assert result.diagnostics.residual_inf_norm < 3.0e-15


def test_backend_candidate_bounds_and_inequalities_are_rechecked() -> None:
    case1 = manufactured_zero_cusp_case(1)
    honest1 = solve_network(case1.inputs, case1.state)
    assert honest1.diagnostics.backend is not None
    outside = list(case1.state.to_vector())
    outside[0] = 2.0 * case1.inputs.anode_voltage_v
    outside_result = solve_network(
        case1.inputs,
        case1.state,
        backend=FixedBackend(
            LeastSquaresResult(
                tuple(outside), (0.0,) * 7, honest1.diagnostics.backend
            )
        ),
    )
    assert outside_result.diagnostics.reason == "canonical_bounds_failed"
    assert outside_result.state is None

    case3 = manufactured_zero_cusp_case(3)
    honest3 = solve_network(
        case3.inputs,
        case3.state,
        options=NetworkSolverOptions(
            publication_policy=PublicationPolicy.REPRESENT_NULLSPACE
        ),
    )
    assert honest3.diagnostics.backend is not None
    unordered = list(case3.state.to_vector())
    unordered[1] = 1.001 * case3.inputs.anode_voltage_v
    unordered_result = solve_network(
        case3.inputs,
        case3.state,
        backend=FixedBackend(
            LeastSquaresResult(
                tuple(unordered), (0.0,) * 21, honest3.diagnostics.backend
            )
        ),
        options=NetworkSolverOptions(
            publication_policy=PublicationPolicy.REPRESENT_NULLSPACE
        ),
    )
    assert unordered_result.diagnostics.reason == "canonical_inequalities_failed"
    assert unordered_result.state is None


@pytest.mark.parametrize(
    "vector",
    [
        (),
        (float("nan"),) * 7,
        (float("inf"),) * 7,
    ],
)
def test_malformed_backend_candidate_fails_closed(vector: tuple[float, ...]) -> None:
    case = manufactured_zero_cusp_case(1)
    honest = solve_network(case.inputs, case.state)
    assert honest.diagnostics.backend is not None
    malicious = LeastSquaresResult(vector, (0.0,) * 7, honest.diagnostics.backend)
    result = solve_network(case.inputs, case.state, backend=FixedBackend(malicious))
    assert result.state is None
    assert result.evaluation is None
    assert result.diagnostics.reason == "backend_candidate_invalid"


def test_rank_is_independent_of_loose_bounds_and_invalid_basis_is_rejected() -> None:
    case = manufactured_zero_cusp_case(3)
    size = case.inputs.dimensions.state_size
    loose = DynamicBounds((-1.0e200,) * size, (1.0e200,) * size)
    baseline = rank_diagnostics(case.state, case.inputs, represented=True)
    extreme = rank_diagnostics(case.state, case.inputs, loose, represented=True)
    assert extreme.numerical_rank == baseline.numerical_rank
    assert extreme.variable_scales == baseline.variable_scales
    assert extreme.max_nullspace_residual == baseline.max_nullspace_residual

    jacobian = analytic_jacobian(case.state, case.inputs)
    scales = physical_variable_scales(case.inputs)
    bogus = tuple(
        value * 1.0e100 for value in baseline.nullspace_basis[0]
    )
    valid, maximum_jv, orthonormality = validate_nullspace_basis(
        jacobian,
        scales,
        (bogus, baseline.nullspace_basis[1]),
        expected_count=2,
        residual_tolerance=1.0e-10,
        orthonormality_tolerance=1.0e-10,
    )
    assert not valid
    assert maximum_jv > 1.0e-10 or orthonormality > 1.0e-10
    duplicate_valid, _, duplicate_orthonormality = validate_nullspace_basis(
        jacobian,
        scales,
        (baseline.nullspace_basis[0], baseline.nullspace_basis[0]),
        expected_count=2,
        residual_tolerance=1.0e-10,
        orthonormality_tolerance=1.0e-10,
    )
    assert not duplicate_valid
    assert duplicate_orthonormality > 0.9


def test_configured_rank_tolerance_is_recorded_and_extra_deficiency_rejected() -> None:
    case = manufactured_zero_cusp_case(2)
    options = NetworkSolverOptions(
        least_squares=SolverOptions(residual_tolerance=1.0e-12),
        publication_policy=PublicationPolicy.REPRESENT_NULLSPACE,
        rank_relative_tolerance=0.1,
    )
    result = solve_network(case.inputs, case.state, options=options)
    assert result.state is None
    assert result.diagnostics.identifiability is not None
    assert result.diagnostics.identifiability.rank_relative_tolerance == 0.1
    assert not result.diagnostics.identifiability.expected_rank
    assert result.diagnostics.reason == "unexpected_jacobian_rank"


@pytest.mark.parametrize(
    "voltage,current",
    [
        (0.0, 1.0),
        (1000.0, 0.0),
        (1.0e-310, 1.0e310),
        (1.0e310, 1.0),
        (float("nan"), 1.0),
        (1000.0, float("inf")),
    ],
)
def test_zero_nonfinite_and_extreme_scales_are_rejected(
    voltage: float, current: float
) -> None:
    topology = make_chain_topology(1, (), provenance_seed="scale-rejection")
    with pytest.raises(NetworkValidationError):
        NetworkInputs(
            topology,
            voltage,
            current,
            0.0,
            0.0,
            provenance_hash("scale-rejection:anode-loss"),
        )


def test_nonfinite_state_and_probability_are_rejected() -> None:
    case = manufactured_zero_cusp_case(1)
    with pytest.raises(NetworkValidationError, match="finite"):
        replace(case.state, plasma_potential_v=(float("nan"),))
    with pytest.raises(NetworkValidationError, match=r"\[0, 1\)"):
        replace(case.inputs, anode_arrival_probability=1.0)
    with pytest.raises(NetworkValidationError, match="non-negative"):
        replace(case.inputs, anode_arrival_standard_uncertainty=-0.1)
    with pytest.raises(NetworkValidationError, match="SHA-256"):
        replace(case.inputs, anode_arrival_provenance_sha256="implicit")


def test_valid_large_scale_remains_finite() -> None:
    topology = make_chain_topology(1, (), provenance_seed="large-valid")
    inputs = NetworkInputs(
        topology,
        1.0e100,
        1.0e-50,
        0.0,
        0.0,
        provenance_hash("large-valid:anode-loss"),
    )
    state = NetworkState(
        plasma_potential_v=(1.0e100,),
        electron_temperature_ev=(1.0,),
        ionization_source_current_a=(0.0,),
        electron_current_a=(1.0e-50, 1.0e-50),
        ion_current_a=(0.0, 0.0),
        cusp_ion_current_a=(),
    )
    evaluation = evaluate_residual(state, inputs)
    assert all(isfinite(value) for value in evaluation.normalized)
