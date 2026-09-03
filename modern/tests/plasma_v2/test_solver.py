"""Rank, solve modes, determinism and fail-closed publication of the v2 solver."""

from __future__ import annotations

import pytest

from cft_revival.plasma import PlasmaValidationError
from cft_revival.plasma_v2 import (
    MASS_FLUX_RATIO,
    AnodeRow,
    CuspLossClosure,
    CuspSheathSpec,
    FourthPotentialRow,
    PotentialClosure,
    SheathClosureInputs,
    SheathRegime,
    SolverPolicy,
    evaluate_residual,
    is_feasible,
    manifold_state,
    rank_report,
    reduced_solve,
    solve_sheath_closure,
    solve_sheath_closure_multistart,
)

KORNFELD_P = (0.060, 0.119, 0.160, 0.254)


def test_rank_blocks_at_the_kornfeld_manifold_point(kornfeld_mode_c, kornfeld_mode_a) -> None:
    state, _ = manifold_state(kornfeld_mode_c, (14.1, 1000.0, 1000.0, 1000.0))
    for inputs, solved in ((kornfeld_mode_c, "none"), (kornfeld_mode_a, "phi_1")):
        report = rank_report(state, inputs)
        assert (report.rows, report.unknowns) == (38, 31)
        assert report.rank_corrected_core == 21  # nullity 4: all potentials free (closure analysis)
        assert report.rank_with_sheath_and_anode == 28  # sheath rows self-identify, anode row +1
        assert report.nullity_before_potential_closure == 3
        assert report.rank_full == 31  # full column rank with the three declared relations
        assert report.solved_potential == solved
        assert len(report.declared_relations) == (4 if solved == "none" else 3)


def test_mode_a_solve_reproduces_kornfeld_phi_1_and_temperatures(kornfeld_mode_a) -> None:
    result = solve_sheath_closure_multistart(kornfeld_mode_a)
    best = result.best
    assert best.diagnostics.converged and best.state is not None and best.evaluation is not None
    assert best.diagnostics.residual_inf_norm <= 1.0e-9
    assert best.diagnostics.jacobian_rank == 31
    assert best.seeded_from_manifold
    core = best.state.core
    assert abs(core.plasma_potential_v[0] - 14.1) < 0.05  # Kornfeld: 14.1 V (solved here, not declared)
    for ours, theirs in zip(core.electron_temperature_ev, (8.9, 100.1, 43.1, 23.5), strict=True):
        assert abs(ours - theirs) < 0.06
    # Anode fall 0 V declared => the anode ion fraction is exactly 1/K0 through R31.
    fraction = -core.ion_current_a[4] / core.electron_current_a[4]
    assert fraction == pytest.approx(1.0 / MASS_FLUX_RATIO, rel=1.0e-8)
    assert max(abs(value) for value in best.evaluation.normalized) <= 1.0e-9
    assert is_feasible(best.state, kornfeld_mode_a)


def test_mode_c_direct_evaluation_is_exact(kornfeld_mode_c) -> None:
    result = solve_sheath_closure(kornfeld_mode_c)
    assert result.diagnostics.converged and result.state is not None
    assert result.diagnostics.residual_inf_norm <= 1.0e-14
    assert result.state.core.plasma_potential_v == (14.1, 1000.0, 1000.0, 1000.0)


def _mode_b_inputs(scl_cusps, coupling_v: float) -> SheathClosureInputs:
    return SheathClosureInputs(
        1000.0,
        1.0,
        scl_cusps,
        KORNFELD_P[3],
        cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
        declared_cusp_probabilities=KORNFELD_P[:3],
        potentials=PotentialClosure(
            anode_row=AnodeRow.SHEATH,
            fourth_row=FourthPotentialRow.CATHODE_COUPLING_DECLARED,
            cathode_coupling_v=coupling_v,
        ),
    )


@pytest.mark.parametrize("coupling_v, expected_fall_v", [(14.0, 13.2), (14.05, 3.9)])
def test_mode_b_solves_the_anode_fall_from_a_declared_cathode_coupling(
    scl_cusps, coupling_v: float, expected_fall_v: float
) -> None:
    """Below the mode-A coupling (14.074 V) the cascade under-delivers at zero fall and
    the anode row is closed by a positive fall that raises the exit drop."""

    inputs = _mode_b_inputs(scl_cusps, coupling_v)
    reduced = reduced_solve(inputs)
    assert reduced.state is not None and reduced.root_variable == "phi_4 - Ua"
    result = solve_sheath_closure_multistart(inputs)
    assert result.best.diagnostics.converged and result.best.state is not None
    state = result.best.state
    fall = state.core.plasma_potential_v[3] - 1000.0
    assert fall == pytest.approx(expected_fall_v, abs=0.1)
    assert 0.0 < fall < state.core.electron_temperature_ev[3]
    assert state.core.plasma_potential_v[0] == pytest.approx(coupling_v)


def test_mode_b_fails_closed_when_the_anode_fall_exceeds_t4(scl_cusps) -> None:
    """Above the mode-A coupling the only root of the anode row sits at a ~55 V fall.

    The exit drop feeds back on the cascade (d j_e4 / d phi_2 ~ I_2 / dE_2), so
    the small root disappears and the remaining one has (Ua - phi_4 + T4) < 0:
    the corrected anode electron term is negative and the state is
    inadmissible.  R31 plus the monoenergetic anode energy bound the admissible
    anode ion fraction to <= e/K0 = 1.39 %.  No state is published.
    """

    inputs = _mode_b_inputs(scl_cusps, 14.1)
    reduced = reduced_solve(inputs)
    assert reduced.state is not None
    fall = reduced.state.core.plasma_potential_v[3] - 1000.0
    assert fall > reduced.state.core.electron_temperature_ev[3]
    assert not is_feasible(reduced.state, inputs)
    result = solve_sheath_closure_multistart(inputs)
    assert not result.best.diagnostics.converged and result.best.state is None


def test_solver_is_deterministic(kornfeld_mode_a) -> None:
    first = solve_sheath_closure_multistart(kornfeld_mode_a)
    second = solve_sheath_closure_multistart(kornfeld_mode_a)
    assert first.best.state is not None and second.best.state is not None
    assert first.best.state.to_vector() == second.best.state.to_vector()
    assert first.residual_floor == second.residual_floor
    assert [a.diagnostics.reason for a in first.attempts] == [a.diagnostics.reason for a in second.attempts]


def test_lm_without_manifold_seed_still_closes(kornfeld_mode_a) -> None:
    policy = SolverPolicy(seed_from_manifold=False)
    result = solve_sheath_closure_multistart(kornfeld_mode_a, policy=policy)
    assert result.best.diagnostics.converged and result.best.state is not None
    assert not result.best.seeded_from_manifold
    assert abs(result.best.state.core.plasma_potential_v[0] - 14.1) < 0.05


def test_no_emission_sheath_fails_closed_on_the_cusp_energy_margin(no_emission_cusps) -> None:
    cusps = tuple(CuspSheathSpec(access_fraction=0.6445) for _ in range(3))
    inputs = SheathClosureInputs(300.0, 3.44e-3, cusps, 0.1)  # type: ignore[arg-type]
    reduced = reduced_solve(inputs)
    assert reduced.state is not None and reduced.reason == "root"
    evaluation = evaluate_residual(reduced.state, inputs)
    assert max(abs(value) for value in evaluation.normalized) <= 1.0e-9
    assert all(margin < 0.0 for margin in evaluation.cusp_energy_margins_ev)
    assert not is_feasible(reduced.state, inputs)
    assert is_feasible(reduced.state, inputs, enforce_cusp_energy_margin=False)
    result = solve_sheath_closure_multistart(inputs)
    assert not result.best.diagnostics.converged
    assert result.best.state is None and result.best.evaluation is None
    assert result.best.diagnostics.reason == "infeasible_initial_state"
    # Relaxing the margin by policy publishes the same manifold root.
    relaxed = solve_sheath_closure_multistart(inputs, policy=SolverPolicy(enforce_cusp_energy_margin=False))
    assert relaxed.best.diagnostics.converged and relaxed.best.state is not None
    assert relaxed.best.state.core.plasma_potential_v[0] == pytest.approx(reduced.state.core.plasma_potential_v[0], abs=1.0e-6)


def test_cascade_that_cannot_carry_the_anode_current_reports_a_floor() -> None:
    cusps = tuple(CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED, access_fraction=1.0) for _ in range(3))
    inputs = SheathClosureInputs(150.0, 3.0, cusps, 0.1)  # type: ignore[arg-type]
    reduced = reduced_solve(inputs)
    assert reduced.state is None and reduced.reason == "no_bracket"
    result = solve_sheath_closure_multistart(inputs)
    assert not result.best.diagnostics.converged
    assert result.best.state is None
    assert 0.0 < result.residual_floor < 1.0


def test_invalid_closure_combinations_are_rejected(scl_cusps) -> None:
    with pytest.raises(PlasmaValidationError):
        PotentialClosure(anode_row=AnodeRow.DECLARED_FALL, fourth_row=FourthPotentialRow.ANODE_FALL_DECLARED)
    with pytest.raises(PlasmaValidationError):
        PotentialClosure(fourth_row=FourthPotentialRow.CATHODE_COUPLING_DECLARED)
    with pytest.raises(PlasmaValidationError):
        PotentialClosure(anode_fall_v=-1.0)
    with pytest.raises(PlasmaValidationError):
        SheathClosureInputs(300.0, 1.0, scl_cusps, 1.0)
    with pytest.raises(PlasmaValidationError):
        SheathClosureInputs(300.0, 1.0, scl_cusps[:2], 0.1)  # type: ignore[arg-type]
