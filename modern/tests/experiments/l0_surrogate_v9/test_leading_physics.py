"""Dimensional and identity tests for the corrected exact L0 leading mean."""

from __future__ import annotations

from math import isclose, sqrt

from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    STANDARD_GRAVITY_M_PER_S2,
    XENON_ATOM_MASS_KG,
    evaluate_batch,
)
from experiments.l0_surrogate_v7.design import operating_points
from experiments.l0_surrogate_v9 import models, protocol


def test_charge_mixture_identity_and_dimensions() -> None:
    row = (0.5, 0.4, 0.6, 0.8, 0.7)
    thrust, isp = models.analytic_outputs(row)
    voltage = 150.0 + 350.0 * row[0]
    flow = 2e-7 + 1.8e-6 * row[1]
    ionized = 0.65 + 0.33 * row[2]
    share = 0.15 * row[3]
    axial = 0.75 + 0.23 * row[4]
    velocity = sqrt(2 * ELEMENTARY_CHARGE_C * voltage / XENON_ATOM_MASS_KG)
    mixture = 1 + (sqrt(2) - 1) * share
    assert isclose(thrust, flow * ionized * mixture * velocity * axial)
    assert isclose(isp, ionized * mixture * velocity * axial / STANDARD_GRAVITY_M_PER_S2)
    assert isclose(thrust / flow, isp * STANDARD_GRAVITY_M_PER_S2)


def test_random_and_extreme_identity_against_accepted_reference() -> None:
    declaration = protocol.load_declaration()
    fixture = protocol._identity_fixture(declaration)
    assert fixture["points"] == 36
    assert fixture["maximum_relative_error"] < 2e-14


def test_explicit_power_boundaries_match_reference() -> None:
    rows = (
        (0.0,) * 8,
        (1.0,) * 8,
        (0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6),
    )
    declaration = protocol.load_declaration()
    results = evaluate_batch(operating_points(rows, protocol._ranges(declaration)))
    for row, result in zip(rows, results, strict=True):
        values = models.analytic_quantities(row)
        budget = result.power_budget
        assert isclose(values["beam_current_a"], budget.beam_current_a, rel_tol=2e-14)
        assert isclose(values["anode_input_power_w"], budget.anode_input_power_w, rel_tol=2e-14)
        assert isclose(values["thruster_electrical_input_power_w"], budget.thruster_electrical_input_power_w, rel_tol=2e-14)
        assert isclose(values["ppu_input_power_w"], budget.ppu_input_power_w, rel_tol=2e-14)
