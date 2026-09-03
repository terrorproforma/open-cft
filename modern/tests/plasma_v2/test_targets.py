"""Reproduction targets: Kornfeld 2007 Table 3.1 and Puca 2024 Table 1 (theirs vs ours)."""

from __future__ import annotations

import pytest

from cft_revival.plasma import PlasmaError, XenonGlobalInputs, evaluate_plasma_residual_cpu, global_row_closed_form
from cft_revival.plasma_v2 import AnodeRow, SheathRegime, evaluate_residual, solve_sheath_closure_multistart
from cft_revival.plasma_v2.targets import (
    KORNFELD_DM10,
    KORNFELD_DM92,
    PUCA_DM10,
    PUCA_DM92,
    REPRODUCTION_TARGETS,
    v1_power_components,
    v2_power_components,
)


def test_transcriptions_are_internally_consistent() -> None:
    assert len(REPRODUCTION_TARGETS) == 4
    assert sum(KORNFELD_DM92.published_powers_w.values()) == pytest.approx(1005.93, abs=0.01)
    assert sum(KORNFELD_DM10.published_powers_w.values()) == pytest.approx(1003.93, abs=0.01)
    assert KORNFELD_DM92.sheath_drops_v == pytest.approx((6.0, 40.0, 35.0))
    je0, derived = PUCA_DM92.cathode_emission_a()
    assert derived and je0 == pytest.approx((0.2528 + 0.0644) / (1.0 - 0.49))
    assert not KORNFELD_DM92.cathode_emission_a()[1]
    assert PUCA_DM92.implied_anode_fall_v() is None  # j_i4 = 0 printed
    assert KORNFELD_DM92.implied_anode_fall_v() < 0.0  # 0.2 % anode ions: below the sheath row's minimum
    assert KORNFELD_DM10.implied_anode_fall_v() > 50.0  # 11 % anode ions: far above


def test_kornfeld_dm92_published_state_under_the_v2_rows() -> None:
    state = KORNFELD_DM92.v2_state()
    inputs = KORNFELD_DM92.v2_inputs(regime=SheathRegime.SPACE_CHARGE_LIMITED)
    evaluation = evaluate_residual(state, inputs)
    # Rows R00-R26: rounding of the printed table only.
    assert max(abs(value) for value in evaluation.normalized[:27]) < 1.0e-3
    # Corrected R27 is near zero; v1's R27 carries the documented 1.49 W misfit.
    assert abs(evaluation.raw[27]) < 0.1
    v1 = evaluate_plasma_residual_cpu(
        KORNFELD_DM92.core_state(), XenonGlobalInputs(1000.0, 1.0, KORNFELD_DM92.cusp_probabilities)
    )
    assert 1.4 < v1.raw[27] < 1.6
    # Their printed cusp potentials do not satisfy either sheath relation.
    assert max(abs(value) for value in evaluation.normalized[28:31]) > 0.01
    # The printed cusp loss (22.9 W) equals the v2 convention (no +EI), not the v1 one.
    v2_powers = v2_power_components(KORNFELD_DM92, SheathRegime.SPACE_CHARGE_LIMITED)
    v1_powers = v1_power_components(KORNFELD_DM92)
    assert v2_powers["cusp"] == pytest.approx(22.9, abs=0.1)
    assert v1_powers["cusp"] == pytest.approx(22.9 + 12.1 * (0.007 + 0.013 + 0.102), abs=0.1)
    assert v2_powers["anode"] == pytest.approx(27.7, abs=0.1)
    assert v2_powers["ionization"] == pytest.approx(12.3, abs=0.1)


@pytest.mark.parametrize(
    "target, phi_1_tolerance, temperature_tolerance",
    [(KORNFELD_DM92, 0.05, 0.1), (KORNFELD_DM10, 1.0, 0.3)],
)
def test_kornfeld_targets_reproduced_with_phi_1_solved(
    target, phi_1_tolerance: float, temperature_tolerance: float
) -> None:
    inputs = target.v2_inputs(regime=SheathRegime.SPACE_CHARGE_LIMITED, anode_row=AnodeRow.SHEATH)
    result = solve_sheath_closure_multistart(inputs)
    assert result.best.diagnostics.converged and result.best.state is not None
    core = result.best.state.core
    assert abs(core.plasma_potential_v[0] - target.plasma_potential_v[0]) < phi_1_tolerance
    for ours, theirs in zip(core.electron_temperature_ev[1:], target.electron_temperature_ev[1:], strict=True):
        assert abs(ours - theirs) < temperature_tolerance
    assert abs(core.electron_temperature_ev[0] - target.electron_temperature_ev[0]) < 0.5


def test_kornfeld_dm92_mode_c_matches_the_printed_currents_to_rounding() -> None:
    inputs = KORNFELD_DM92.v2_inputs(regime=SheathRegime.SPACE_CHARGE_LIMITED)
    result = solve_sheath_closure_multistart(inputs)
    assert result.best.diagnostics.converged and result.best.state is not None
    core = result.best.state.core
    for ours, theirs in zip(core.ionization_source_current_a, KORNFELD_DM92.ionization_source_current_a, strict=True):
        assert abs(ours - theirs) < 0.005
    for ours, theirs in zip(core.electron_current_a, KORNFELD_DM92.electron_current_a, strict=True):
        assert abs(ours - theirs) < 0.01
    for ours, theirs in zip(core.cusp_ion_current_a, KORNFELD_DM92.cusp_ion_current_a, strict=True):
        assert abs(ours - theirs) < 0.002


def test_no_emission_regime_rejects_the_kornfeld_state_by_the_energy_margin() -> None:
    inputs = KORNFELD_DM92.v2_inputs(regime=SheathRegime.FLOATING_NO_EMISSION, anode_row=AnodeRow.SHEATH)
    result = solve_sheath_closure_multistart(inputs)
    assert not result.best.diagnostics.converged
    assert result.best.diagnostics.reason == "infeasible_initial_state"


@pytest.mark.parametrize("target", [PUCA_DM92, PUCA_DM10])
def test_puca_states_are_far_from_a_root_of_the_kornfeld_rows(target) -> None:
    state = target.v2_state()
    inputs = target.v2_inputs(regime=SheathRegime.SPACE_CHARGE_LIMITED)
    evaluation = evaluate_residual(state, inputs)
    # The cathode row and interface-0 current are violated at the 0.4-0.5 level once
    # j_e0 is derived from their own R01 (their variant made the cathode current an input).
    assert max(abs(value) for value in evaluation.normalized[:27]) > 0.4
    assert evaluation.raw[27] > 100.0  # even the corrected R27 is >100 W off for their state
    assert target.ionization_source_current_a[0] < 0.0
    with pytest.raises(PlasmaError):
        # j_i4 = 0 leaves the anode sheath row undefined for their state.
        evaluate_residual(state, target.v2_inputs(regime=SheathRegime.SPACE_CHARGE_LIMITED, anode_row=AnodeRow.SHEATH))


def test_v1_package_inconsistency_still_holds_untouched() -> None:
    """The v1 closed form still shows the R27 inconsistency for the DM9.2 p (v1 is read-only)."""

    v1_inputs = XenonGlobalInputs(1000.0, 1.0, KORNFELD_DM92.cusp_probabilities)
    closed_form = global_row_closed_form(KORNFELD_DM92.core_state(), v1_inputs)
    assert closed_form > 1.0  # W; EI * sum(p_k je_{k-1}) alone is 1.36 W
