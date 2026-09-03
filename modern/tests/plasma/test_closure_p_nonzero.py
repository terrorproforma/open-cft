"""Pin the 2026-09-03 closure finding: no admissible root for interior cusp loss.

These tests record the analysed behaviour of the corrected four-cell ledger so
that it cannot silently regress in either direction:

* rows R00-R26 are consistent and are parametrized by the four potentials;
* on that manifold the global row R27 has the closed form
  ``2*(j_e3*(1-p4)+I4)*(phi_4-Ua) + EI*(p1*j_e0+p2*j_e1+p3*j_e2)``;
* therefore the solver closes for ``p1=p2=p3=0`` (any ``p4``) and cannot close
  for any positive interior probability, which is a model inconsistency and not
  a solver defect (the correction is ``PROPOSED_NOT_ACCEPTED`` in the ledger).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from cft_revival.plasma import (
    PlasmaValidationError,
    SolverOptions,
    XenonGlobalInputs,
    constraint_margins,
    evaluate_plasma_residual_cpu,
    global_row_closed_form,
    is_feasible,
    potential_parametrized_state,
    solve_global_discharge_multistart,
)

SPEC_ROOT = Path(__file__).resolve().parents[2] / "spec" / "plasma"
DM92_PROBABILITIES = (0.060, 0.119, 0.160, 0.254)
GLOBAL_ROW = 27


def _random_case(rng: random.Random) -> tuple[XenonGlobalInputs, tuple[float, ...]]:
    voltage = rng.choice((150.0, 300.0, 500.0, 1000.0))
    current = rng.choice((0.1, 0.5, 1.0, 3.0))
    probability = tuple(rng.uniform(0.0, 0.7) for _ in range(4))
    interior = sorted(rng.uniform(0.01 * voltage, voltage) for _ in range(3))
    potentials = (*interior, rng.uniform(voltage, 1.5 * voltage))
    return XenonGlobalInputs(voltage, current, probability), potentials


def test_potentials_parametrize_the_27_row_manifold_exactly() -> None:
    rng = random.Random(20260903)
    for _ in range(200):
        inputs, potentials = _random_case(rng)
        state = potential_parametrized_state(inputs, potentials)
        normalized = evaluate_plasma_residual_cpu(state, inputs).normalized
        assert max(abs(value) for value in normalized[:GLOBAL_ROW]) < 1.0e-11


def test_global_row_matches_its_closed_form_on_the_manifold() -> None:
    rng = random.Random(1)
    for _ in range(200):
        inputs, potentials = _random_case(rng)
        state = potential_parametrized_state(inputs, potentials)
        raw_global = evaluate_plasma_residual_cpu(state, inputs).raw[GLOBAL_ROW]
        predicted = global_row_closed_form(state, inputs)
        assert raw_global == pytest.approx(predicted, rel=1.0e-9, abs=1.0e-9)


def test_closed_form_reproduces_the_documented_dm92_global_misfit() -> None:
    # The ledger records 1.4866e-3 at R27 for the rounded published DM9.2 state;
    # on the exact manifold with the published potentials the closed form gives
    # EI*(p1*j_e0+p2*j_e1+p3*j_e2)/(Ua*Ia), the same order and sign.
    inputs = XenonGlobalInputs(1000.0, 1.0, DM92_PROBABILITIES)
    state = potential_parametrized_state(inputs, (14.1, 1000.0, 1000.0, 1000.0))
    normalized = evaluate_plasma_residual_cpu(state, inputs).normalized
    assert max(abs(value) for value in normalized[:GLOBAL_ROW]) < 1.0e-12
    assert 1.3e-3 < normalized[GLOBAL_ROW] < 1.6e-3
    assert normalized[GLOBAL_ROW] == pytest.approx(
        global_row_closed_form(state, inputs) / 1000.0, rel=1.0e-9
    )


@pytest.mark.parametrize("probabilities", [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.3)])
def test_global_row_vanishes_at_the_anode_potential_without_interior_loss(
    probabilities,
) -> None:
    inputs = XenonGlobalInputs(300.0, 1.0, probabilities)
    # phi_1 = 21 V makes the electron cascade carry the imposed 1 A (j_e4 >= Ia).
    state = potential_parametrized_state(inputs, (21.0, 120.0, 260.0, 300.0))
    normalized = evaluate_plasma_residual_cpu(state, inputs).normalized
    assert max(abs(value) for value in normalized) < 1.0e-12
    assert is_feasible(state, inputs)
    assert state.electron_current_a[4] >= inputs.anode_current_a


def test_manifold_global_misfit_is_linear_in_interior_probability() -> None:
    floors = []
    for epsilon in (1.0e-3, 1.0e-2, 1.0e-1):
        inputs = XenonGlobalInputs(300.0, 1.0, (epsilon, epsilon, epsilon, 0.0))
        state = potential_parametrized_state(inputs, (21.0, 120.0, 260.0, 300.0))
        floors.append(evaluate_plasma_residual_cpu(state, inputs).normalized[GLOBAL_ROW])
    assert all(value > 0.0 for value in floors)
    assert 8.0 < floors[1] / floors[0] < 12.0
    assert 6.0 < floors[2] / floors[1] < 12.0


def test_interior_cusp_probability_has_no_admissible_root() -> None:
    inputs = XenonGlobalInputs(300.0, 1.0, DM92_PROBABILITIES)
    result = solve_global_discharge_multistart(
        inputs, start_count=3, options=SolverOptions(residual_tolerance=1.0e-8)
    )
    assert result.best.state is None
    assert result.best.evaluation is None
    assert result.residual_floor > 1.0e-4
    for attempt in result.attempts:
        diagnostics = attempt.diagnostics
        assert not diagnostics.converged
        assert diagnostics.reason in {
            "iteration_limit",
            "step_tolerance_without_balance",
        }
        assert diagnostics.jacobian_rank == 22
        assert len(diagnostics.normalized_residuals) == 28
    rows = result.attempts[result.selected_start_index].diagnostics.normalized_residuals
    assert max(range(28), key=lambda index: abs(rows[index])) == GLOBAL_ROW


def test_relaxed_anode_constraint_root_is_a_compensating_error_root() -> None:
    # Dropping phi_4 >= Ua admits an exact root a few volts below the anode:
    # 2*(j_e3*(1-p4)+I4)*(phi_4-Ua) = -EI*sum(p_k*j_e,k-1).  It balances two
    # bookkeeping errors against each other and is rejected by the admissible
    # region, so it must never be published.
    inputs = XenonGlobalInputs(300.0, 1.0, DM92_PROBABILITIES)
    interior = (4.23, 270.0, 285.0)

    def global_row(phi_4: float) -> float:
        state = potential_parametrized_state(inputs, (*interior, phi_4))
        return evaluate_plasma_residual_cpu(state, inputs).raw[GLOBAL_ROW]

    low, high = interior[2], 300.0
    assert global_row(low) < 0.0 < global_row(high)
    for _ in range(200):
        middle = 0.5 * (low + high)
        if global_row(middle) > 0.0:
            high = middle
        else:
            low = middle
    root = 0.5 * (low + high)
    state = potential_parametrized_state(inputs, (*interior, root))
    evaluation = evaluate_plasma_residual_cpu(state, inputs)
    assert max(abs(value) for value in evaluation.normalized) < 1.0e-10
    assert 0.1 < 300.0 - root < 5.0
    assert not is_feasible(state, inputs)
    assert constraint_margins(state, inputs)[4] < 0.0


def test_potential_parametrization_rejects_invalid_potentials() -> None:
    inputs = XenonGlobalInputs(300.0, 1.0, DM92_PROBABILITIES)
    with pytest.raises(PlasmaValidationError, match="four values"):
        potential_parametrized_state(inputs, (1.0, 2.0, 3.0))
    with pytest.raises(PlasmaValidationError, match="cathode"):
        potential_parametrized_state(inputs, (-1.0, 2.0, 3.0, 300.0))
    with pytest.raises(PlasmaValidationError, match="non-negative"):
        potential_parametrized_state(inputs, (50.0, 2.0, 3.0, 300.0))


def test_equation_ledger_flags_the_proposed_correction_as_not_accepted() -> None:
    ledger = json.loads((SPEC_ROOT / "equation-ledger.json").read_text(encoding="utf-8"))
    analysis = ledger["global_row_consistency"]
    assert analysis["status"] == "PROPOSED_NOT_ACCEPTED"
    assert analysis["analysis_date"] == "2026-09-03"
    assert "2*(j_e3*(1-p4)+I4)*(phi_4-Ua)" in analysis["closed_form_on_manifold"]
    assert "EI*(p1*j_e0+p2*j_e1+p3*j_e2)" in analysis["closed_form_on_manifold"]
    proposals = {item["id"]: item for item in analysis["proposed_corrections"]}
    assert set(proposals) == {"Pcusp", "Panode_e"}
    assert "+EI" not in proposals["Pcusp"]["proposed_expression"].replace(" ", "")
    assert "Ua-phi_4+T4" in proposals["Panode_e"]["proposed_expression"].replace(" ", "")
    for item in proposals.values():
        assert item["status"] == "PROPOSED_NOT_ACCEPTED"
    # The executable rows and power expressions are unchanged by the proposal.
    assert len(ledger["residual_rows"]) == 28
    assert len(ledger["power_expressions"]) == 7
    cusp = next(item for item in ledger["power_expressions"] if item["id"] == "Pcusp")
    assert "+EI" in cusp["expression"].replace(" ", "")
    assert (SPEC_ROOT.parents[1] / analysis["document"]).is_file()
