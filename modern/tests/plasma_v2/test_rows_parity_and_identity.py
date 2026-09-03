"""Rows R00-R26 equal the read-only v1 rows; the corrected R27 is an identity."""

from __future__ import annotations

import random

import pytest

from cft_revival.plasma import (
    PlasmaError,
    XenonGlobalInputs,
    evaluate_plasma_residual_cpu,
    global_row_closed_form,
    potential_parametrized_state,
)
from cft_revival.plasma_v2 import (
    RESIDUAL_SIZE,
    STATE_SIZE,
    AnodeRow,
    CuspLossClosure,
    CuspSheathSpec,
    FourthPotentialRow,
    PotentialClosure,
    SheathClosureInputs,
    SheathClosureState,
    SheathRegime,
    analytic_jacobian,
    evaluate_residual,
    finite_difference_jacobian,
    raw_residual,
    sheath_drops,
)


def _random_manifold_case(rng: random.Random):
    """A v1 manifold state and the matching v2 inputs/state (mode C, CL-1)."""

    while True:
        voltage = rng.uniform(150.0, 1200.0)
        current = rng.uniform(0.05, 3.0)
        probability = tuple(rng.uniform(0.0, 0.5) for _ in range(4))
        phi_1 = rng.uniform(0.02, 0.08) * voltage
        phi_4 = voltage * rng.uniform(1.0, 1.05)
        phi_3 = phi_4 - rng.uniform(0.0, 0.05) * voltage
        phi_2 = phi_3 - rng.uniform(0.0, 0.05) * voltage
        v1 = XenonGlobalInputs(voltage, current, probability)  # type: ignore[arg-type]
        try:
            core = potential_parametrized_state(v1, (phi_1, phi_2, phi_3, phi_4))
        except PlasmaError:
            continue
        regime = rng.choice((SheathRegime.FLOATING_NO_EMISSION, SheathRegime.SPACE_CHARGE_LIMITED))
        inputs = SheathClosureInputs(
            voltage,
            current,
            tuple(CuspSheathSpec(regime=regime) for _ in range(3)),  # type: ignore[arg-type]
            probability[3],
            cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
            declared_cusp_probabilities=probability[:3],
            potentials=PotentialClosure(
                interior_step_3_v=phi_3 - phi_2,
                interior_step_4_v=phi_4 - phi_3,
                anode_row=AnodeRow.DECLARED_FALL,
                fourth_row=FourthPotentialRow.CATHODE_COUPLING_DECLARED,
                anode_fall_v=phi_4 - voltage,
                cathode_coupling_v=phi_1,
            ),
        )
        state = SheathClosureState(core, sheath_drops(inputs, core.electron_temperature_ev), probability[:3])
        return v1, core, inputs, state


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rows_r00_r26_match_v1_to_round_off(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(40):
        v1, core, inputs, state = _random_manifold_case(rng)
        ours = raw_residual(state.to_vector(), inputs)
        theirs = evaluate_plasma_residual_cpu(core, v1).raw
        scale = v1.anode_voltage_v * v1.anode_current_a
        assert max(abs(float(a) - b) for a, b in zip(ours[:27], theirs[:27], strict=True)) <= 1.0e-12 * scale


def test_corrected_r27_is_an_identity_on_the_manifold_while_v1_r27_is_not() -> None:
    rng = random.Random(20260903)
    for _ in range(60):
        v1, core, inputs, state = _random_manifold_case(rng)
        ours = raw_residual(state.to_vector(), inputs)
        theirs = evaluate_plasma_residual_cpu(core, v1).raw
        scale = v1.anode_voltage_v * v1.anode_current_a
        assert abs(float(ours[27])) <= 1.0e-12 * scale
        # v1's R27 equals its documented closed form and is positive for p > 0.
        assert abs(theirs[27] - global_row_closed_form(core, v1)) <= 1.0e-9 * scale
        assert theirs[27] > 0.0


def test_the_two_conventions_differ_by_exactly_the_two_corrections() -> None:
    """v1 R27 - v2 R27 = EI * sum(p_k je_{k-1}) + 2 (Je_3 + I_4)(phi_4 - Ua) at ANY state."""

    rng = random.Random(7)
    for _ in range(30):
        v1, core, inputs, state = _random_manifold_case(rng)
        # Perturb the state off the manifold so that the relation is tested as an algebraic identity.
        vector = list(state.to_vector())
        for index in range(4, 25):
            vector[index] *= 1.0 + rng.uniform(-0.2, 0.2)
        vector[4:8] = [abs(value) + 1.0e-3 for value in vector[4:8]]
        perturbed = SheathClosureState.from_vector(vector)
        v2_raw = raw_residual(perturbed.to_vector(), inputs)
        v1_raw = evaluate_plasma_residual_cpu(perturbed.core, v1).raw
        electron = perturbed.core.electron_current_a
        p = v1.cusp_arrival_probabilities
        recombination = inputs.xenon_ionization_energy_ev * sum(electron[k] * p[k] for k in range(3))
        transported = electron[3] * (1.0 - p[3]) + perturbed.core.ionization_source_current_a[3]
        anode = 2.0 * transported * (perturbed.core.plasma_potential_v[3] - v1.anode_voltage_v)
        expected = recombination + anode
        scale = v1.anode_voltage_v * v1.anode_current_a
        assert abs((v1_raw[27] - float(v2_raw[27])) - expected) <= 1.0e-9 * scale


def test_analytic_jacobian_matches_finite_differences() -> None:
    rng = random.Random(11)
    for _ in range(3):
        _, _, inputs, state = _random_manifold_case(rng)
        analytic = analytic_jacobian(state, inputs)
        numeric = finite_difference_jacobian(state, inputs)
        assert len(analytic) == RESIDUAL_SIZE and all(len(row) == STATE_SIZE for row in analytic)
        scale = max(1.0, max(abs(value) for row in analytic for value in row))
        worst = max(
            abs(a - b) for ra, rb in zip(analytic, numeric, strict=True) for a, b in zip(ra, rb, strict=True)
        )
        assert worst <= 1.0e-6 * scale


def test_evaluation_shapes_and_normalization() -> None:
    rng = random.Random(3)
    _, _, inputs, state = _random_manifold_case(rng)
    evaluation = evaluate_residual(state, inputs)
    assert len(evaluation.raw) == RESIDUAL_SIZE
    assert len(evaluation.normalized) == RESIDUAL_SIZE
    assert len(evaluation.margins) == len(evaluation.margin_names)
    assert max(abs(value) for value in evaluation.normalized) <= 1.0e-12
