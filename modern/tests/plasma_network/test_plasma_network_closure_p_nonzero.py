"""The N=4 network inherits the accepted ledger's global-row inconsistency.

See ``tests/plasma/test_closure_p_nonzero.py`` and
``docs/workstreams/global-plasma-closure-analysis.md``: with any positive
interior cusp probability the global power row cannot vanish inside the
admissible region, while anode-only loss closes on the ``phi_N = Ua`` boundary.
"""

from __future__ import annotations

import pytest

from cft_revival.plasma import SolverOptions
from cft_revival.plasma_network import (
    NetworkInputs,
    NetworkSolverOptions,
    PublicationPolicy,
    make_chain_topology,
    provenance_hash,
    solve_network_multistart,
)

OPTIONS = NetworkSolverOptions(
    least_squares=SolverOptions(residual_tolerance=1.0e-8),
    publication_policy=PublicationPolicy.REPRESENT_NULLSPACE,
)


def _inputs(interior: tuple[float, float, float], anode: float) -> NetworkInputs:
    topology = make_chain_topology(
        4, interior, provenance_seed=f"closure-analysis:{interior}:{anode}"
    )
    return NetworkInputs(
        topology=topology,
        anode_voltage_v=300.0,
        anode_current_a=1.0,
        anode_arrival_probability=anode,
        anode_arrival_standard_uncertainty=0.0,
        anode_arrival_provenance_sha256=provenance_hash("closure-analysis:anode"),
    )


def test_interior_cusp_loss_fails_closed_in_the_network_formulation() -> None:
    result = solve_network_multistart(
        _inputs((0.060, 0.119, 0.160), 0.254), start_count=3, options=OPTIONS
    )
    assert result.best.state is None
    assert result.best.evaluation is None
    assert result.residual_floor > 1.0e-4
    assert all(not attempt.diagnostics.numerical_converged for attempt in result.attempts)
    assert all(not attempt.diagnostics.published for attempt in result.attempts)
    assert all(
        attempt.diagnostics.reason == "backend_returned_no_candidate"
        for attempt in result.attempts
    )
    # The backend ledger retains every normalized row; the global row dominates.
    for attempt in result.attempts:
        backend = attempt.diagnostics.backend
        assert backend is not None
        assert not backend.converged
        assert backend.reason in {"iteration_limit", "step_tolerance_without_balance"}
        rows = backend.normalized_residuals
        assert len(rows) == 28
        assert backend.residual_inf_norm > 1.0e-4
    selected = result.attempts[result.selected_start_index].diagnostics.backend
    best_rows = selected.normalized_residuals
    assert max(range(28), key=lambda index: abs(best_rows[index])) == 27


def test_anode_only_cusp_loss_closes_on_the_anode_potential_boundary() -> None:
    result = solve_network_multistart(_inputs((0.0, 0.0, 0.0), 0.3), start_count=3, options=OPTIONS)
    assert result.best.state is not None
    assert result.best.diagnostics.published
    assert result.residual_floor <= 1.0e-8
    # Closure sits on the phi_N = Ua face (the global row is 2*(j_e3*(1-p4)+I4)*(phi_N-Ua) there).
    assert result.best.state.plasma_potential_v[-1] == pytest.approx(300.0, abs=1.0e-6)
    assert result.best.diagnostics.identifiability is not None
    assert result.best.diagnostics.identifiability.nullity == 3
