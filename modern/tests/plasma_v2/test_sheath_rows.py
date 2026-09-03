"""Sheath rows R28-R30, cusp-loss closures and the cusp energy consistency criterion."""

from __future__ import annotations

from math import exp, log

import pytest

from cft_revival.plasma import PlasmaValidationError
from cft_revival.plasma_v2 import (
    CRITICAL_EMISSION_YIELD,
    FLOATING_SHEATH_COEFFICIENT,
    MASS_FLUX_RATIO,
    SPACE_CHARGE_LIMITED_COEFFICIENT,
    AnodeRow,
    CuspLossClosure,
    CuspSheathSpec,
    FourthPotentialRow,
    PotentialClosure,
    SheathClosureInputs,
    SheathRegime,
    closure_probabilities,
    evaluate_residual,
    manifold_state,
)
from cft_revival.plasma_v2.constants import ELECTRON_MASS_KG, XENON_MASS_KG


def test_xenon_sheath_constants() -> None:
    assert MASS_FLUX_RATIO == pytest.approx((XENON_MASS_KG / (2.0 * 3.141592653589793 * ELECTRON_MASS_KG)) ** 0.5)
    assert 195.0 < MASS_FLUX_RATIO < 195.4
    assert FLOATING_SHEATH_COEFFICIENT == pytest.approx(log(MASS_FLUX_RATIO))
    assert 5.27 < FLOATING_SHEATH_COEFFICIENT < 5.28
    assert SPACE_CHARGE_LIMITED_COEFFICIENT == 1.02
    assert 0.98 < CRITICAL_EMISSION_YIELD < 0.985


def test_sheath_coefficient_by_regime() -> None:
    assert CuspSheathSpec().sheath_coefficient() == pytest.approx(FLOATING_SHEATH_COEFFICIENT)
    assert CuspSheathSpec(area_ratio=4.0).sheath_coefficient() == pytest.approx(FLOATING_SHEATH_COEFFICIENT + log(4.0))
    assert CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED).sheath_coefficient() == 1.02
    mild = CuspSheathSpec(regime=SheathRegime.FLOATING_WITH_EMISSION, emission_yield=0.5)
    assert mild.sheath_coefficient() == pytest.approx(log(0.5 * MASS_FLUX_RATIO))
    strong = CuspSheathSpec(regime=SheathRegime.FLOATING_WITH_EMISSION, emission_yield=0.99)
    assert strong.sheath_coefficient() == 1.02  # Hobbs-Wesson space-charge limit
    assert strong.emission_is_space_charge_limited


def test_invalid_cusp_specs_are_rejected() -> None:
    with pytest.raises(PlasmaValidationError):
        CuspSheathSpec(emission_yield=0.2)  # emission under the no-emission regime
    with pytest.raises(PlasmaValidationError):
        CuspSheathSpec(regime=SheathRegime.FLOATING_WITH_EMISSION, emission_yield=1.0)
    with pytest.raises(PlasmaValidationError):
        CuspSheathSpec(access_fraction=1.5)
    with pytest.raises(PlasmaValidationError):
        CuspSheathSpec(area_ratio=0.0)


def test_cl3_probability_is_access_times_boltzmann_factor(scl_cusps) -> None:
    cusps = tuple(CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED, access_fraction=a) for a in (0.375, 0.6445, 0.869))
    inputs = SheathClosureInputs(300.0, 1.0, cusps, 0.1)  # type: ignore[arg-type]
    probability = closure_probabilities(inputs, None)
    for value, access in zip(probability, (0.375, 0.6445, 0.869), strict=True):
        assert value == pytest.approx(access * exp(-1.02))
    no_emission = SheathClosureInputs(300.0, 1.0, tuple(CuspSheathSpec(access_fraction=0.6445) for _ in range(3)), 0.1)  # type: ignore[arg-type]
    assert closure_probabilities(no_emission, None)[0] == pytest.approx(0.6445 / MASS_FLUX_RATIO)


def test_sheath_rows_close_exactly_on_the_manifold(kornfeld_mode_c) -> None:
    state, _ = manifold_state(kornfeld_mode_c, (14.1, 1000.0, 1000.0, 1000.0))
    evaluation = evaluate_residual(state, kornfeld_mode_c)
    assert max(abs(value) for value in evaluation.normalized[28:31]) <= 1.0e-15
    for k in range(3):
        assert state.sheath_drop_v[k] == pytest.approx(1.02 * state.core.electron_temperature_ev[k])
        assert state.cusp_wall_potential_v[k] == pytest.approx(state.core.plasma_potential_v[k] - state.sheath_drop_v[k])


@pytest.mark.parametrize("regime", [SheathRegime.FLOATING_NO_EMISSION, SheathRegime.SPACE_CHARGE_LIMITED])
@pytest.mark.parametrize("voltage", [300.0, 1000.0])
def test_cusp_energy_margin_closed_form_criterion(regime: SheathRegime, voltage: float) -> None:
    """margin_k >= 0  <=>  c_s,k <= (1 + I_k/Je_k) / CT, exactly, on the manifold.

    With T_k (Je_k + I_k) = CT Je_k dE_k the wall energy of the lost electrons
    is dE_k - c_s T_k = dE_k [1 - c_s CT Je_k/(Je_k + I_k)], so the sign is
    independent of the step size.  For xenon without emission (c_s = 5.27)
    this requires I_k/Je_k >= CT ln K0 - 1 = 2.59, i.e. dE_k >= 447 V; the
    space-charge-limited sheath (1.02 < 1/CT = 1.47) always satisfies it.
    """

    cusps = tuple(CuspSheathSpec(regime=regime) for _ in range(3))
    phi_1 = 0.02 * voltage
    inputs = SheathClosureInputs(
        voltage,
        1.0,
        cusps,  # type: ignore[arg-type]
        0.254,
        cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
        declared_cusp_probabilities=(0.06, 0.119, 0.16),
        potentials=PotentialClosure(
            anode_row=AnodeRow.DECLARED_FALL,
            fourth_row=FourthPotentialRow.CATHODE_COUPLING_DECLARED,
            cathode_coupling_v=phi_1,
        ),
    )
    state, _ = manifold_state(inputs, (phi_1, voltage, voltage, voltage))
    evaluation = evaluate_residual(state, inputs)
    coefficient = cusps[0].sheath_coefficient()
    core = state.core
    p = (*state.cusp_probability, 0.254)
    threshold_gain_v = (inputs.xenon_ionization_energy_ev / inputs.ionization_fraction) * (
        inputs.thermalization_fraction * coefficient - 1.0
    )
    for k in range(3):
        transmitted = core.electron_current_a[k] * (1.0 - p[k])
        ratio = core.ionization_source_current_a[k] / transmitted
        predicted_sign = (1.0 + ratio) / inputs.thermalization_fraction - coefficient
        margin = evaluation.cusp_energy_margins_ev[k]
        assert (margin >= 0.0) == (predicted_sign >= 0.0)
        gain = evaluation.powers.cusps[k].entering_energy_ev
        assert (margin >= 0.0) == (gain >= threshold_gain_v - 1.0e-9)
    if regime is SheathRegime.SPACE_CHARGE_LIMITED:
        assert all(value >= 0.0 for value in evaluation.cusp_energy_margins_ev)
        assert threshold_gain_v < 0.0
    else:
        assert 440.0 < threshold_gain_v < 455.0
        # Only the exit cusp (dE_2 ~ Ua) can pass, and only at 1 kV.
        assert evaluation.cusp_energy_margins_ev[0] < 0.0
        assert evaluation.cusp_energy_margins_ev[2] < 0.0
        assert (evaluation.cusp_energy_margins_ev[1] >= 0.0) == (voltage >= 500.0)


def test_cl4_requires_density_field_and_radius() -> None:
    cusps = tuple(CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED) for _ in range(3))
    with pytest.raises(PlasmaValidationError):
        SheathClosureInputs(300.0, 1.0, cusps, 0.1, cusp_loss_closure=CuspLossClosure.CL4_HYBRID_AREA)  # type: ignore[arg-type]
    complete = tuple(
        CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED, electron_density_per_m3=1.0e15, wall_field_t=0.1)
        for _ in range(3)
    )
    with pytest.raises(PlasmaValidationError):
        SheathClosureInputs(300.0, 1.0, complete, 0.1, cusp_loss_closure=CuspLossClosure.CL4_HYBRID_AREA)  # type: ignore[arg-type]
    declared = PotentialClosure(
        anode_row=AnodeRow.DECLARED_FALL,
        fourth_row=FourthPotentialRow.CATHODE_COUPLING_DECLARED,
        cathode_coupling_v=10.0,
    )
    inputs = SheathClosureInputs(
        300.0, 1.0, complete, 0.1, cusp_loss_closure=CuspLossClosure.CL4_HYBRID_AREA, wall_radius_m=0.002,  # type: ignore[arg-type]
        potentials=declared,
    )
    state, iterations = manifold_state(inputs, (10.0, 300.0, 300.0, 300.0))
    assert iterations > 0
    assert all(0.0 < value < 0.9 for value in state.cusp_probability)
    # The CL-4 rows are satisfied by the fixed point and the loss grows with the prefactor.
    evaluation = evaluate_residual(state, inputs)
    assert max(abs(value) for value in evaluation.normalized[35:38]) <= 1.0e-10
    doubled = SheathClosureInputs(
        300.0, 1.0, complete, 0.1, cusp_loss_closure=CuspLossClosure.CL4_HYBRID_AREA, wall_radius_m=0.002,  # type: ignore[arg-type]
        leak_width_prefactor=2.0, potentials=declared,
    )
    state_2, _ = manifold_state(doubled, (10.0, 300.0, 300.0, 300.0))
    assert all(b > a for a, b in zip(state.cusp_probability, state_2.cusp_probability, strict=True))
    # With a 100x larger declared density the thermal leak current exceeds the
    # cascade current and the probabilities saturate at the bound (fail-closed later).
    dense = tuple(
        CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED, electron_density_per_m3=1.0e17, wall_field_t=0.1)
        for _ in range(3)
    )
    saturated, _ = manifold_state(
        SheathClosureInputs(
            300.0, 1.0, dense, 0.1, cusp_loss_closure=CuspLossClosure.CL4_HYBRID_AREA, wall_radius_m=0.002,  # type: ignore[arg-type]
            potentials=declared,
        ),
        (10.0, 300.0, 300.0, 300.0),
    )
    assert max(saturated.cusp_probability) > 0.99
