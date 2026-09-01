from dataclasses import replace

import pytest

from cft_revival.plasma import AnodeIonEnergySign, PlasmaState, XenonGlobalInputs
from cft_revival.plasma_network import (
    NetworkInputs,
    NetworkState,
    NetworkValidationError,
    analytic_jacobian,
    analytic_jacobian_batch,
    constraint_margins,
    evaluate_residual,
    evaluate_residual_batch,
    finite_difference_jacobian,
    from_accepted_four_cell,
    manufactured_zero_cusp_case,
    prove_four_cell_compatibility,
)


def accepted_case(
    sign: AnodeIonEnergySign = AnodeIonEnergySign.SOURCE_MINUS_SIGN,
) -> tuple[XenonGlobalInputs, PlasmaState]:
    return (
        XenonGlobalInputs(
            1000.0,
            1.0,
            (0.060, 0.119, 0.160, 0.254),
            anode_ion_energy_sign=sign,
        ),
        PlasmaState(
            plasma_potential_v=(14.1, 1000.0, 1000.0, 1000.0),
            electron_temperature_ev=(8.9, 100.1, 43.1, 23.5),
            ionization_source_current_a=(0.008, 0.543, 0.310, 0.157),
            electron_current_a=(0.106, 0.107, 0.637, 0.845, 1.002),
            ion_current_a=(0.894, 0.893, 0.363, 0.155, -0.002),
            cusp_ion_current_a=(0.007, 0.013, 0.102),
        ),
    )


@pytest.mark.parametrize("cell_count", range(1, 7))
def test_manufactured_networks_close_all_particle_and_energy_rows(
    cell_count: int,
) -> None:
    case = manufactured_zero_cusp_case(cell_count)
    evaluation = evaluate_residual(case.state, case.inputs)
    assert len(evaluation.raw) == 7 * cell_count
    assert len(case.state.to_vector()) == 6 * cell_count + 1
    assert max(abs(value) for value in evaluation.normalized) < 3.0e-15
    assert max(abs(value) for value in evaluation.closures.interface_current_a) < 1.0e-12
    assert max(abs(value) for value in evaluation.closures.ion_continuity_a) < 1.0e-12
    assert max(abs(value) for value in evaluation.closures.cell_energy_w) < 1.0e-9
    assert abs(evaluation.closures.global_energy_w) < 1.0e-8
    assert all(margin >= 0.0 for margin in constraint_margins(case.state, case.inputs))


@pytest.mark.parametrize(
    "sign",
    [
        AnodeIonEnergySign.SOURCE_MINUS_SIGN,
        AnodeIonEnergySign.OCR_PLUS_SIGN_ALTERNATIVE,
    ],
)
def test_n4_compatibility_is_bit_exact_row_by_row_for_both_branches(
    sign: AnodeIonEnergySign,
) -> None:
    inputs, state = accepted_case(sign)
    report = prove_four_cell_compatibility(inputs, state)
    assert report.compatible
    assert len(report.rows) == 28
    assert [row.row_id for row in report.rows] == [f"R{i:02d}" for i in range(28)]
    assert all(row.accepted_raw == row.network_raw for row in report.rows)
    assert all(
        row.accepted_normalized == row.network_normalized for row in report.rows
    )


def test_current_and_energy_signs_are_oriented_and_named() -> None:
    inputs, state = accepted_case()
    network_inputs, network_state = from_accepted_four_cell(inputs, state)
    evaluation = evaluate_residual(network_state, network_inputs)
    assert evaluation.raw[8:11] == pytest.approx((0.0, 0.0, 0.0), abs=2.0e-17)
    assert evaluation.raw[11] == pytest.approx(0.0, abs=2.0e-17)
    assert evaluation.powers.cusp_loss_w > 0.0
    assert evaluation.powers.anode_electron_loss_w >= 0.0

    moved = replace(
        network_state,
        plasma_potential_v=(14.1, 950.0, 980.0, 1050.0),
    )
    source = evaluate_residual(moved, network_inputs).powers
    alternative = evaluate_residual(
        moved,
        replace(
            network_inputs,
            anode_ion_energy_sign=AnodeIonEnergySign.OCR_PLUS_SIGN_ALTERNATIVE,
        ),
    ).powers
    assert source.anode_ion_energy_exchange_w > 0.0
    assert alternative.anode_ion_energy_exchange_w < 0.0


@pytest.mark.parametrize("cell_count", range(1, 7))
def test_analytic_jacobian_matches_bound_aware_finite_difference(
    cell_count: int,
) -> None:
    case = manufactured_zero_cusp_case(cell_count)
    analytic = analytic_jacobian(case.state, case.inputs)
    finite = finite_difference_jacobian(
        case.state, case.inputs, relative_step=2.0e-6
    )
    assert len(analytic) == len(finite) == 7 * cell_count
    assert all(len(row) == 6 * cell_count + 1 for row in analytic)
    error = max(
        abs(analytic[row][column] - finite[row][column])
        for row in range(7 * cell_count)
        for column in range(6 * cell_count + 1)
    )
    assert error < 5.0e-7


def test_batch_interface_is_homogeneous_ordered_and_rejects_ragged_topology() -> None:
    case2 = manufactured_zero_cusp_case(2)
    case3 = manufactured_zero_cusp_case(3)
    batch = evaluate_residual_batch(
        (case2.state, case2.state), (case2.inputs, case2.inputs)
    )
    assert batch[0] == batch[1]
    jacobians = analytic_jacobian_batch(
        (case2.state, case2.state), (case2.inputs, case2.inputs)
    )
    assert jacobians[0] == jacobians[1]
    assert len(jacobians[0]) == 14
    with pytest.raises(NetworkValidationError, match="homogeneous"):
        evaluate_residual_batch(
            (case2.state, case3.state), (case2.inputs, case3.inputs)
        )
    with pytest.raises(NetworkValidationError, match="homogeneous"):
        analytic_jacobian_batch(
            (case2.state, case3.state), (case2.inputs, case3.inputs)
        )
    with pytest.raises(NetworkValidationError, match="non-empty"):
        evaluate_residual_batch((), ())


def test_terminal_n1_case_has_no_interior_cusp_current_or_cusp_power() -> None:
    case = manufactured_zero_cusp_case(1)
    evaluation = evaluate_residual(case.state, case.inputs)
    assert case.state.cusp_ion_current_a == ()
    assert evaluation.closures.cusp_current_a == ()
    assert evaluation.powers.cusp_loss_w == 0.0


def test_state_topology_mismatch_is_rejected() -> None:
    case2 = manufactured_zero_cusp_case(2)
    case3 = manufactured_zero_cusp_case(3)
    with pytest.raises(NetworkValidationError, match="dimensions"):
        evaluate_residual(case2.state, case3.inputs)
