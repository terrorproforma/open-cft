import json
from pathlib import Path
from sys import float_info

import pytest

from cft_revival.plasma import (
    PlasmaValidationError,
    XenonGlobalInputs,
    default_state_bounds,
    evaluate_plasma_residual_cpu,
)

SPEC_ROOT = Path(__file__).resolve().parents[2] / "spec" / "plasma"


def test_equation_ledger_records_corrections_hypotheses_and_limits() -> None:
    ledger = json.loads((SPEC_ROOT / "equation-ledger.json").read_text(encoding="utf-8"))
    rows = ledger["residual_rows"]
    assert len(rows) == 28
    assert [row["row"] for row in rows] == list(range(28))
    assert all(
        {"expression", "unit", "source", "sign", "confidence", "branch"} <= row.keys()
        for row in rows
    )
    assert ledger["source_label"] == "MDO (original)"
    assert "editorial" in ledger["source_label_policy"].lower()
    assert rows[11]["expression"] == "j_i3-I4-j_i4"
    assert "j_e3*(1-p4)+I4" in rows[14]["expression"]
    assert len(ledger["power_expressions"]) == 7
    exchange = next(
        item for item in ledger["power_expressions"]
        if item["id"] == "Panode_i_exchange"
    )
    assert "signed exchange" in exchange["sign"]
    assert any("neutral density or mass-flow" in item for item in ledger["limitations"])


def test_2020_values_are_external_comparison_only() -> None:
    fixture_text = (SPEC_ROOT / "external-comparison-2020.json").read_text(
        encoding="utf-8"
    )
    evidence = json.loads(fixture_text)
    assert evidence["role"] == "external_cross_model_comparison_only"
    assert "solver acceptance tolerance" in evidence["prohibited_uses"]
    cases = {case["id"]: case for case in evidence["cases"]}
    original = cases["YEO2020-S1-MDO-ORIGINAL"]
    assert original["source_model_label"] == "MDO (original)"
    expected_editorial = (
        "postprocessed/" + "corrected " + "low-fidelity interpretation"
    )
    assert original["editorial_interpretation"] == expected_editorial
    assert original["thrust_n"] == 0.1027
    assert cases["YEO2020-S1-PIC"]["specific_impulse_s"] == 1333.0
    assert cases["YEO2020-S1-PIC-INFORMED"]["reported_efficiency_fraction"] == 0.146
    stale_identifier = "CORRECTED" + "-GLOBAL"
    assert stale_identifier not in fixture_text


@pytest.mark.parametrize(
    "voltage,current",
    [(1.0e-150, 1.0e-150), (1.0e150, 1.0e-150), (1.0e-150, 1.0e150)],
)
def test_extreme_representable_inputs_have_finite_bounds(voltage, current) -> None:
    inputs = XenonGlobalInputs(voltage, current, (0.0, 0.1, 0.2, 0.3))
    bounds = default_state_bounds(inputs)
    assert all(abs(value) <= float_info.max for value in (*bounds.lower, *bounds.upper))


def test_nonrepresentable_derived_bounds_fail_closed() -> None:
    with pytest.raises(PlasmaValidationError, match="non-representable"):
        XenonGlobalInputs(float_info.max, 1.0, (0.0, 0.1, 0.2, 0.3))


def test_input_power_overflow_and_underflow_are_typed_failures() -> None:
    with pytest.raises(PlasmaValidationError, match="non-representable"):
        XenonGlobalInputs(1.0e200, 1.0e200, (0.0, 0.1, 0.2, 0.3))
    with pytest.raises(PlasmaValidationError, match="normal positive"):
        XenonGlobalInputs(1.0e-200, 1.0e-200, (0.0, 0.1, 0.2, 0.3))


def test_tiny_voltage_domain_is_rejected_even_when_input_power_is_normal() -> None:
    with pytest.raises(PlasmaValidationError, match="too small"):
        XenonGlobalInputs(1.0e-250, 1.0e100, (0.0, 0.1, 0.2, 0.3))


@pytest.mark.parametrize(
    ("voltage", "current"),
    (
        (1.0e-310, 1.0e100),
        (1.0e100, 1.0e-310),
    ),
)
def test_subnormal_voltage_or_current_scale_is_rejected_before_derivation(
    voltage, current
) -> None:
    with pytest.raises(PlasmaValidationError, match="residual scales.*normal"):
        XenonGlobalInputs(voltage, current, (0.0, 0.1, 0.2, 0.3))


def test_minimum_normal_current_and_normal_product_are_preserved() -> None:
    inputs = XenonGlobalInputs(
        1.0,
        float_info.min,
        (0.0, 0.1, 0.2, 0.3),
    )
    assert inputs.anode_current_a == float_info.min
    assert inputs.anode_voltage_v * inputs.anode_current_a == float_info.min
    bounds = default_state_bounds(inputs)
    assert bounds.upper[8] == 2.0 * float_info.min


def test_minimum_normal_voltage_is_rejected_for_derived_cathode_scale() -> None:
    with pytest.raises(
        PlasmaValidationError,
        match=r"anode_voltage_v is too small.*voltage\^\(3/2\) scale",
    ):
        XenonGlobalInputs(
            float_info.min,
            1.0,
            (0.0, 0.1, 0.2, 0.3),
        )


def test_published_state_has_only_finite_public_values(
    dm92_published_state, dm92_inputs
) -> None:
    evaluation = evaluate_plasma_residual_cpu(dm92_published_state, dm92_inputs)
    assert all(abs(value) <= float_info.max for value in evaluation.raw)
    assert all(abs(value) <= float_info.max for value in evaluation.normalized)
