"""Current and energy conservation identities on published v2 states."""

from __future__ import annotations

from math import fsum

import pytest

from cft_revival.plasma_v2 import (
    CuspLossClosure,
    CuspSheathSpec,
    SheathClosureInputs,
    SheathRegime,
    power_balance,
    solve_sheath_closure_multistart,
)


@pytest.fixture(params=["kornfeld_cl1", "cl3_scl_300v"])
def solved_case(request, kornfeld_mode_a):
    if request.param == "kornfeld_cl1":
        inputs = kornfeld_mode_a
    else:
        cusps = tuple(CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED, access_fraction=0.6445) for _ in range(3))
        inputs = SheathClosureInputs(300.0, 3.44e-3, cusps, 0.1, cusp_loss_closure=CuspLossClosure.CL3_SHEATH_LIMITED)  # type: ignore[arg-type]
    result = solve_sheath_closure_multistart(inputs)
    assert result.best.state is not None
    return inputs, result.best.state


def test_interface_currents_sum_to_the_anode_current(solved_case) -> None:
    inputs, state = solved_case
    for je, ji in zip(state.core.electron_current_a, state.core.ion_current_a, strict=True):
        assert je + ji == pytest.approx(inputs.anode_current_a, rel=1.0e-9)


def test_floating_dielectric_cusps_carry_zero_net_current(solved_case) -> None:
    inputs, state = solved_case
    for k in range(3):
        lost_electrons = state.core.electron_current_a[k] * state.cusp_probability[k]
        assert state.core.cusp_ion_current_a[k] == pytest.approx(lost_electrons, rel=1.0e-9, abs=1.0e-15)


def test_ion_continuity_cell_by_cell(solved_case) -> None:
    inputs, state = solved_case
    ji = state.core.ion_current_a
    source = state.core.ionization_source_current_a
    jic = state.core.cusp_ion_current_a
    scale = inputs.anode_current_a
    for k in range(3):
        assert abs(ji[k] - ji[k + 1] - source[k] + jic[k]) <= 1.0e-9 * scale
    assert abs(ji[3] - source[3] - ji[4]) <= 1.0e-9 * scale


def test_global_power_identity_closes_to_round_off(solved_case) -> None:
    inputs, state = solved_case
    powers = power_balance(state, inputs)
    total = fsum(
        (
            powers.beam_power_w,
            powers.ionization_loss_w,
            powers.excitation_loss_w,
            powers.cusp_loss_w,
            powers.anode_electron_loss_w,
            powers.anode_ion_loss_w,
        )
    )
    assert total == pytest.approx(powers.input_power_w, rel=1.0e-9)
    assert abs(powers.closure_w) <= 1.0e-9 * powers.input_power_w
    assert powers.ionization_loss_w == pytest.approx(inputs.xenon_ionization_energy_ev * fsum(state.core.ionization_source_current_a))


def test_cusp_energy_split_sums_to_the_booked_total(solved_case) -> None:
    inputs, state = solved_case
    powers = power_balance(state, inputs)
    for split in powers.cusps:
        assert split.electron_wall_w + split.ion_wall_w == pytest.approx(split.total_w, rel=1.0e-12, abs=1.0e-18)
        assert split.total_w == pytest.approx(split.lost_electron_current_a * split.entering_energy_ev)
        assert split.ion_wall_w == pytest.approx(split.lost_electron_current_a * split.sheath_drop_v)
        assert split.electron_wall_energy_margin_ev >= 0.0
    assert powers.cusp_electron_wall_w + powers.cusp_ion_wall_w == pytest.approx(powers.cusp_loss_w)


def test_cusp_wall_potentials_are_below_their_cell_potentials(solved_case) -> None:
    inputs, state = solved_case
    for k in range(3):
        assert state.cusp_wall_potential_v[k] == pytest.approx(state.core.plasma_potential_v[k] - state.sheath_drop_v[k])
        assert state.sheath_drop_v[k] >= 0.0
