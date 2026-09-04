"""The six GATE-L2 metric constraints of ``paper/evidence/result-gates.json`` made concrete for L2 v2.

``result-gates.json`` requires, for the coupled hybrid model, ``interface_conservation_passed``,
``spatial_levels >= 3``, ``temporal_levels >= 3``, ``code_comparison_passed``,
``numerical_uncertainty_reported``, ``failed_cases_count >= 0`` and the four
``uncertainty_components`` (input, numerical, emulator, model_discrepancy).  This module turns
them into pure functions over the L2 result records so that every gate can be exercised - and
made to fail - by a synthetic test.  The tolerances are inputs (predeclared in the experiment
protocol), never defaults hidden here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

NOT_COMPARED = "not_compared"


@dataclass(frozen=True, slots=True)
class Comparison:
    name: str
    value: float | None
    reference: float | None
    tolerance: float | None          # relative; None -> not compared (the PIC gives no band)
    relative_difference: float | None
    status: str                      # "within" | "outside" | "not_compared"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "reference": self.reference, "tolerance": self.tolerance,
                "relative_difference": self.relative_difference, "status": self.status}


def compare(name: str, value: float | None, reference: float | None, tolerance: float | None) -> Comparison:
    """Relative comparison; ``tolerance None`` or an unusable reference gives ``not_compared``."""

    if tolerance is None or reference is None or value is None or not isfinite(reference) or not isfinite(value) or reference == 0.0:
        return Comparison(name, value, reference, tolerance, None, NOT_COMPARED)
    rel = (value - reference) / abs(reference)
    return Comparison(name, value, reference, tolerance, rel, "within" if abs(rel) <= tolerance else "outside")


def interface_conservation(
    *,
    charge_identity_max_relative: float,
    charge_identity_bound: float,
    neutral_ledger_closure_relative: float,
    neutral_ledger_bound: float,
    windowed_energy_residual_ratio: float | None,
    energy_residual_bound: float,
    plateau_reached: bool,
) -> dict[str, Any]:
    """Charge (plasma + wall + induced = 0), atoms (inventory ledger) and energy (windowed residual) closures."""

    checks = {
        "charge_identity": {"value": charge_identity_max_relative, "bound": charge_identity_bound,
                            "passed": isfinite(charge_identity_max_relative) and charge_identity_max_relative <= charge_identity_bound},
        "neutral_ledger": {"value": neutral_ledger_closure_relative, "bound": neutral_ledger_bound,
                           "passed": isfinite(neutral_ledger_closure_relative) and abs(neutral_ledger_closure_relative) <= neutral_ledger_bound},
        "energy_residual_window": {"value": windowed_energy_residual_ratio, "bound": energy_residual_bound,
                                   "passed": windowed_energy_residual_ratio is not None and isfinite(windowed_energy_residual_ratio)
                                   and abs(windowed_energy_residual_ratio) <= energy_residual_bound},
        "plateau": {"value": plateau_reached, "bound": True, "passed": bool(plateau_reached)},
    }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def levels_gate(levels: Sequence[Mapping[str, Any]], *, minimum: int, quantity_keys: Sequence[str]) -> dict[str, Any]:
    """Count the finished levels of a refinement family and report the spread of every quantity across them."""

    finished = [level for level in levels if level.get("finished")]
    spread: dict[str, Any] = {}
    for key in quantity_keys:
        values = [float(level["quantities"][key]) for level in finished if level.get("quantities", {}).get(key) is not None]
        if len(values) >= 2 and all(isfinite(v) for v in values):
            centre = values[-1]
            spread[key] = {"values": values, "max_relative_spread": max(abs(v - centre) for v in values) / abs(centre) if centre != 0.0 else None}
        else:
            spread[key] = {"values": values, "max_relative_spread": None}
    return {"levels_completed": len(finished), "minimum": minimum, "passed": len(finished) >= minimum, "spread": spread,
            "labels": [level.get("label") for level in finished]}


def code_comparison(comparisons: Sequence[Comparison]) -> dict[str, Any]:
    compared = [c for c in comparisons if c.status != NOT_COMPARED]
    outside = [c.name for c in compared if c.status == "outside"]
    return {"passed": bool(compared) and not outside, "compared": len(compared), "outside": outside,
            "not_compared": [c.name for c in comparisons if c.status == NOT_COMPARED], "comparisons": [c.to_dict() for c in comparisons]}


def uncertainty_components(*, input_component: Mapping[str, Any], numerical: Mapping[str, Any], emulator: Mapping[str, Any],
                           model_discrepancy: Mapping[str, Any]) -> dict[str, Any]:
    """The four required components; ``reported`` is True only when every component carries a value or an explicit statement."""

    components = {"input": dict(input_component), "numerical": dict(numerical), "emulator": dict(emulator), "model_discrepancy": dict(model_discrepancy)}
    reported = all(("value" in c or "statement" in c) for c in components.values())
    return {"components": components, "names": sorted(components), "reported": reported}


def evaluate_l2_gates(
    *,
    conservation: Mapping[str, Any],
    spatial: Mapping[str, Any],
    temporal: Mapping[str, Any],
    comparison: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    failed_cases: int,
) -> dict[str, Any]:
    """Assemble the GATE-L2 metric block and the verdict.

    ``accepted`` requires every constraint; ``rejected_on_comparison`` when only the code comparison fails (a valid,
    informative outcome); ``not_evaluable`` when the refinement families or the conservation gate fail (the comparison
    is then not the binding statement).
    """

    metrics = {
        "interface_conservation_passed": bool(conservation["passed"]),
        "spatial_levels": int(spatial["levels_completed"]),
        "temporal_levels": int(temporal["levels_completed"]),
        "code_comparison_passed": bool(comparison["passed"]),
        "numerical_uncertainty_reported": bool(uncertainty["reported"]) and bool(spatial["passed"]) and bool(temporal["passed"]),
        "failed_cases_count": int(failed_cases),
        "uncertainty_components": list(uncertainty["names"]),
    }
    structural = metrics["interface_conservation_passed"] and spatial["passed"] and temporal["passed"] and metrics["numerical_uncertainty_reported"]
    if structural and metrics["code_comparison_passed"]:
        verdict = "accepted"
    elif structural:
        verdict = "rejected_on_comparison"
    else:
        verdict = "not_evaluable"
    return {"metrics": metrics, "verdict": verdict}


__all__ = ["NOT_COMPARED", "Comparison", "code_comparison", "compare", "evaluate_l2_gates", "interface_conservation", "levels_gate",
           "uncertainty_components"]
