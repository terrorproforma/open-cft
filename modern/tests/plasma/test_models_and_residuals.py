from dataclasses import replace
from math import isfinite

import pytest

from cft_revival.plasma import (
    AnodeIonEnergySign,
    PlasmaState,
    PlasmaValidationError,
    XenonGlobalInputs,
    analytic_jacobian,
    constraint_margins,
    default_state_bounds,
    evaluate_plasma_residual_batch_cpu,
    evaluate_plasma_residual_cpu,
    finite_difference_jacobian,
    is_feasible,
)


def test_inputs_enforce_units_domain_and_energy_simplex() -> None:
    with pytest.raises(PlasmaValidationError):
        XenonGlobalInputs(0.0, 1.0, (0.1, 0.1, 0.1, 0.1))
    with pytest.raises(PlasmaValidationError):
        XenonGlobalInputs(1000.0, 1.0, (0.1, 0.1, 0.1, 1.0))
    with pytest.raises(PlasmaValidationError, match="sum to one"):
        XenonGlobalInputs(
            1000.0,
            1.0,
            (0.1, 0.1, 0.1, 0.1),
            excitation_fraction=0.2,
            ionization_fraction=0.2,
            thermalization_fraction=0.2,
        )


def test_state_round_trip_and_bounds(dm92_published_state, dm92_inputs) -> None:
    assert PlasmaState.from_vector(dm92_published_state.to_vector()) == dm92_published_state
    bounds = default_state_bounds(dm92_inputs)
    assert len(bounds.lower) == len(bounds.upper) == 25
    assert all(isfinite(value) for value in (*bounds.lower, *bounds.upper))
    assert is_feasible(dm92_published_state, dm92_inputs, bounds)
    assert all(value >= 0.0 for value in constraint_margins(dm92_published_state, dm92_inputs))


def test_published_table_confirms_corrected_cusp_current_sign(
    dm92_published_state, dm92_inputs
) -> None:
    residual = evaluate_plasma_residual_cpu(dm92_published_state, dm92_inputs)
    # ji[k] = ji[k+1] + I[k] - jic[k] closes with the rounded 2007 table.
    assert residual.raw[8] == pytest.approx(0.0, abs=2.0e-17)
    assert residual.raw[9] == pytest.approx(0.0, abs=2.0e-17)
    assert residual.raw[10] == pytest.approx(0.0, abs=2.0e-17)
    # The archived MATLAB's plus-cusp RHS would miss by 2*jic3 = 0.204 A.
    legacy_broken_third_cell = (
        dm92_published_state.ion_current_a[2]
        - dm92_published_state.ion_current_a[3]
        - dm92_published_state.ionization_source_current_a[2]
        - dm92_published_state.cusp_ion_current_a[2]
    )
    assert legacy_broken_third_cell == pytest.approx(-0.204)


def test_published_terminal_ion_current_uses_signed_ji4(
    dm92_published_state, dm92_inputs
) -> None:
    residual = evaluate_plasma_residual_cpu(dm92_published_state, dm92_inputs)
    assert 0.155 + 0.002 == pytest.approx(0.157)
    assert residual.raw[11] == pytest.approx(0.0, abs=2.0e-17)


def test_fourth_thermal_row_uses_transmitted_current_not_je4(
    dm92_published_state, dm92_inputs
) -> None:
    evaluation = evaluate_plasma_residual_cpu(dm92_published_state, dm92_inputs)
    temperature = dm92_published_state.electron_temperature_ev[3]
    electron = dm92_published_state.electron_current_a
    source = dm92_published_state.ionization_source_current_a[3]
    probability = dm92_inputs.cusp_arrival_probabilities[3]
    gain = (
        dm92_published_state.plasma_potential_v[3]
        - dm92_published_state.plasma_potential_v[2]
        + dm92_published_state.electron_temperature_ev[2]
    )
    expected = (
        temperature * (electron[3] * (1.0 - probability) + source)
        - dm92_inputs.thermalization_fraction
        * electron[3]
        * (1.0 - probability)
        * gain
    )
    legacy_je4 = (
        temperature * electron[4]
        - dm92_inputs.thermalization_fraction
        * electron[3]
        * (1.0 - probability)
        * gain
    )
    assert evaluation.raw[14] == pytest.approx(expected)
    assert abs(evaluation.raw[14]) < abs(legacy_je4) / 100.0


def test_residuals_are_normalized_by_declared_current_and_power_scales(
    dm92_published_state, dm92_inputs
) -> None:
    evaluation = evaluate_plasma_residual_cpu(dm92_published_state, dm92_inputs)
    assert len(evaluation.raw) == len(evaluation.normalized) == 28
    for row in range(12):
        assert evaluation.normalized[row] == evaluation.raw[row] / dm92_inputs.anode_current_a
    for row in range(12, 15):
        assert evaluation.normalized[row] == evaluation.raw[row] / 1000.0
    assert evaluation.powers.closure_w == pytest.approx(
        evaluation.powers.beam_power_w
        + evaluation.powers.ionization_loss_w
        + evaluation.powers.excitation_loss_w
        + evaluation.powers.cusp_loss_w
        + evaluation.powers.anode_net_power_w
        - evaluation.powers.input_power_w
    )
    assert evaluation.closures.global_energy_residual_w == evaluation.powers.closure_w


def test_analytic_chain_rule_jacobian_matches_finite_difference(
    dm92_published_state, dm92_inputs
) -> None:
    analytic = analytic_jacobian(dm92_published_state, dm92_inputs)
    finite_difference = finite_difference_jacobian(
        dm92_published_state, dm92_inputs, relative_step=2.0e-6
    )
    assert len(analytic) == len(finite_difference) == 28
    assert all(len(row) == 25 for row in analytic)
    maximum_error = max(
        abs(analytic[row][column] - finite_difference[row][column])
        for row in range(28)
        for column in range(25)
    )
    assert maximum_error < 2.0e-8


def test_batch_api_is_ordered_fixed_shape_and_rejects_ragged_contracts(
    dm92_published_state, dm92_inputs
) -> None:
    batch = evaluate_plasma_residual_batch_cpu(
        [dm92_published_state, dm92_published_state],
        [dm92_inputs, dm92_inputs],
    )
    assert len(batch) == 2
    assert batch[0] == batch[1]
    with pytest.raises(PlasmaValidationError):
        evaluate_plasma_residual_batch_cpu([], [])
    with pytest.raises(PlasmaValidationError):
        evaluate_plasma_residual_batch_cpu([dm92_published_state], [dm92_inputs, dm92_inputs])


def test_unresolved_anode_sign_is_an_explicit_hypothesis(
    dm92_published_state, dm92_inputs
) -> None:
    moved = replace(
        dm92_published_state,
        plasma_potential_v=(14.1, 950.0, 980.0, 1050.0),
    )
    source = evaluate_plasma_residual_cpu(
        moved, dm92_inputs
    ).powers.anode_ion_energy_exchange_w
    alternative_inputs = replace(
        dm92_inputs,
        anode_ion_energy_sign=AnodeIonEnergySign.OCR_PLUS_SIGN_ALTERNATIVE,
    )
    alternative_powers = evaluate_plasma_residual_cpu(
        moved, alternative_inputs
    ).powers
    expected_difference = (
        2.0
        * dm92_published_state.ion_current_a[4]
        * (moved.plasma_potential_v[3] - dm92_inputs.anode_voltage_v)
    )
    assert alternative_powers.anode_ion_energy_exchange_w - source == pytest.approx(
        expected_difference
    )
    assert source > 0.0
    assert alternative_powers.anode_ion_energy_exchange_w < 0.0
    assert alternative_powers.anode_electron_loss_w >= 0.0
