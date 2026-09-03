"""Verification runner for the v2 sheath closure (development evidence only).

``python -m cft_revival.plasma_v2.verification --out docs/workstreams/plasma-v2-verification.json``
(run from ``modern/``) recomputes:

(a) structure: v1-row parity, the R27 identity on the corrected manifold,
    Jacobian check and block ranks;
(b) reproduction targets: Kornfeld 2007 Table 3.1 (DM9.2, DM10) and Puca 2024
    Table 1 (DM9.2, DM10) evaluated under the v2 rows, then re-solved with
    the published potential structure (mode C) and with phi_1 solved through
    the anode sheath (mode A);
(c) closure grid: Ua x Ia x access fraction x sheath regime under CL-3, and
    CL-1 with the Kornfeld DM9.2 probabilities;
(d) leak-width prefactor sensitivity (CL-4 and the CL-3 area-ratio variant);
(e) model-to-model context against the pic2d steady-state v2 plateau;
(f) the byte identity of the read-only v1 package against the paper manifest.

Everything is deterministic (fixed seeds, fixed grids).  The output is a
development record, not a preregistered result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from math import exp, isfinite, log
from pathlib import Path
from typing import Any

from cft_revival.plasma import (
    PlasmaError,
    XenonGlobalInputs,
    evaluate_plasma_residual_cpu,
    potential_parametrized_state,
)

from . import constants
from .manifold import reduced_solve
from .models import (
    ROW_IDS,
    AnodeRow,
    CuspLossClosure,
    CuspSheathSpec,
    FourthPotentialRow,
    PotentialClosure,
    SheathClosureInputs,
    SheathClosureState,
    SheathRegime,
    SolverPolicy,
)
from .residuals import (
    analytic_jacobian,
    default_state_bounds,
    evaluate_residual,
    finite_difference_jacobian,
    is_feasible,
    raw_residual,
)
from .solver import rank_report, solve_sheath_closure_multistart
from .targets import (
    REPRODUCTION_TARGETS,
    PublishedFourCellState,
    v1_power_components,
    v2_power_components,
)

V1_PACKAGE_FILES = ("__init__.py", "models.py", "reference.py", "residuals.py", "solver.py")
V1_PACKAGE_DIR = Path("src/cft_revival/plasma")
PAPER_MANIFEST = Path("../paper/evidence/manifests/four-cell-closure.json")

GRID_VOLTAGES_V = (150.0, 300.0, 500.0, 1000.0)
GRID_CURRENTS_A = (0.1, 0.5, 1.0, 3.0)
GRID_ACCESS_FRACTIONS = (0.26, 0.375, 0.622, 0.869, 1.0)
GRID_ANODE_CUSP_PROBABILITY = 0.1
KORNFELD_DM92_P = (0.060, 0.119, 0.160, 0.254)
REGIMES = (SheathRegime.FLOATING_NO_EMISSION, SheathRegime.SPACE_CHARGE_LIMITED)

PIC_VOLTAGE_V = 300.0
PIC_DISCHARGE_A = 0.0034443993821890494  # window mean discharge current of the plateau
PIC_ACCESS_FRACTION = 0.64453125  # v4 primary-2N pooled wall-hit fraction on the same P2 field
PIC_MAPS = Path("experiments/pic2d_cft_steady_state_v2/results/maps.npz")
PIC_SUMMARY = Path("experiments/pic2d_cft_steady_state_v2/results/summary.json")
PIC_CUSP_WALL_FIELD_FALLBACK_T = 0.107


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _state_dict(state: SheathClosureState) -> dict[str, Any]:
    core = state.core
    return {
        "phi_v": list(core.plasma_potential_v),
        "T_ev": list(core.electron_temperature_ev),
        "I_a": list(core.ionization_source_current_a),
        "je_a": list(core.electron_current_a),
        "ji_a": list(core.ion_current_a),
        "jic_a": list(core.cusp_ion_current_a),
        "sheath_drop_v": list(state.sheath_drop_v),
        "cusp_wall_potential_v": list(state.cusp_wall_potential_v),
        "p": list(state.cusp_probability),
    }


def _solve_record(inputs: SheathClosureInputs, *, policy: SolverPolicy | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    reduced = reduced_solve(inputs)
    reduced_record: dict[str, Any] = {
        "reason": reduced.reason,
        "root_variable": reduced.root_variable,
        "root_value": reduced.root_value,
        "anode_row_residual_v": reduced.anode_row_residual_v,
        "bracket_found": reduced.bracket_found,
        "probability_iterations": reduced.probability_iterations,
    }
    classification = "no_manifold_root"
    if reduced.state is not None:
        evaluation = evaluate_residual(reduced.state, inputs)
        lower, upper = default_state_bounds(inputs)
        vector = reduced.state.to_vector()
        in_bounds = all(low <= value <= high for value, low, high in zip(vector, lower, upper, strict=True))
        feasible_all = is_feasible(reduced.state, inputs)
        feasible_no_energy = is_feasible(reduced.state, inputs, enforce_cusp_energy_margin=False)
        reduced_record["max_normalized_residual"] = max(abs(value) for value in evaluation.normalized)
        reduced_record["in_bounds"] = in_bounds
        reduced_record["je4_minus_Ia_a"] = reduced.state.core.electron_current_a[4] - inputs.anode_current_a
        reduced_record["feasible_all_margins"] = feasible_all
        reduced_record["feasible_without_cusp_energy_margin"] = feasible_no_energy
        reduced_record["cusp_energy_margins_ev"] = list(evaluation.cusp_energy_margins_ev)
        reduced_record["violated_margins"] = [
            name for name, value in zip(evaluation.margin_names, evaluation.margins, strict=True) if value < 0.0
        ]
        reduced_record["state"] = _state_dict(reduced.state)
        if feasible_all:
            classification = "manifold_root_admissible"
        elif feasible_no_energy:
            classification = "manifold_root_blocked_by_cusp_energy_margin"
        else:
            classification = "manifold_root_violates_bounds_or_current_margins"
    result = solve_sheath_closure_multistart(inputs, policy=policy)
    best = result.best
    if best.diagnostics.converged:
        classification = "closed"
    record: dict[str, Any] = {
        "reduced": reduced_record,
        "classification": classification,
        "converged": best.diagnostics.converged,
        "reason": best.diagnostics.reason,
        "residual_floor": result.residual_floor,
        "residual_inf_norm": best.diagnostics.residual_inf_norm,
        "iterations": best.diagnostics.iterations,
        "jacobian_rank": best.diagnostics.jacobian_rank,
        "seeded_from_manifold": best.seeded_from_manifold,
        "attempt_outcomes": [attempt.diagnostics.reason for attempt in result.attempts],
        "selected_start_index": result.selected_start_index,
        "wall_seconds": time.perf_counter() - started,
    }
    if best.state is not None and best.evaluation is not None:
        rank = rank_report(best.state, inputs)
        powers = best.evaluation.powers
        record["state"] = _state_dict(best.state)
        record["rank"] = asdict(rank)
        record["powers_w"] = {
            "beam": powers.beam_power_w,
            "ionization": powers.ionization_loss_w,
            "excitation": powers.excitation_loss_w,
            "cusp_total": powers.cusp_loss_w,
            "cusp_electron_wall": powers.cusp_electron_wall_w,
            "cusp_ion_wall": powers.cusp_ion_wall_w,
            "anode_electron": powers.anode_electron_loss_w,
            "anode_ion": powers.anode_ion_loss_w,
            "input": powers.input_power_w,
            "closure": powers.closure_w,
        }
        record["cusp_energy_margins_ev"] = list(best.evaluation.cusp_energy_margins_ev)
        record["anode_ion_fraction"] = -best.state.core.ion_current_a[4] / best.state.core.electron_current_a[4]
        record["cusp_lost_electron_current_a"] = [split.lost_electron_current_a for split in powers.cusps]
    return record


def structure_section(seed: int = 20260903, samples: int = 200) -> dict[str, Any]:
    rng = random.Random(seed)
    parity_max = 0.0
    identity_max = 0.0
    v1_r27_min = float("inf")
    jacobian_max = 0.0
    evaluated = 0
    for _ in range(samples):
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
        cusps = tuple(CuspSheathSpec(regime=rng.choice(REGIMES)) for _ in range(3))
        inputs = SheathClosureInputs(
            voltage,
            current,
            cusps,  # type: ignore[arg-type]
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
        coefficients = inputs.sheath_coefficients()
        state = SheathClosureState(
            core,
            tuple(coefficients[k] * core.electron_temperature_ev[k] for k in range(3)),  # type: ignore[arg-type]
            probability[:3],
        )
        v2_raw = raw_residual(state.to_vector(), inputs)
        v1_raw = evaluate_plasma_residual_cpu(core, v1).raw
        parity_max = max(parity_max, max(abs(float(a) - b) for a, b in zip(v2_raw[:27], v1_raw[:27], strict=True)))
        identity_max = max(identity_max, abs(float(v2_raw[27])) / (voltage * current))
        v1_r27_min = min(v1_r27_min, v1_raw[27] / (voltage * current))
        evaluated += 1
        if evaluated <= 5:
            analytic = analytic_jacobian(state, inputs)
            numeric = finite_difference_jacobian(state, inputs)
            scale = max(1.0, max(abs(value) for row in analytic for value in row))
            jacobian_max = max(
                jacobian_max,
                max(abs(a - b) for ra, rb in zip(analytic, numeric, strict=True) for a, b in zip(ra, rb, strict=True))
                / scale,
            )
    from .targets import KORNFELD_DM92

    mode_c = KORNFELD_DM92.v2_inputs(regime=SheathRegime.SPACE_CHARGE_LIMITED)
    core = potential_parametrized_state(
        mode_c.v1_inputs(KORNFELD_DM92.cusp_probabilities), KORNFELD_DM92.plasma_potential_v
    )
    coefficients = mode_c.sheath_coefficients()
    manifold = SheathClosureState(
        core,
        tuple(coefficients[k] * core.electron_temperature_ev[k] for k in range(3)),  # type: ignore[arg-type]
        KORNFELD_DM92.cusp_probabilities[:3],
    )
    ranks = {
        "mode_c_declared_potentials": asdict(rank_report(manifold, mode_c)),
        "mode_a_anode_sheath": asdict(
            rank_report(manifold, KORNFELD_DM92.v2_inputs(anode_row=AnodeRow.SHEATH))
        ),
    }
    return {
        "sampled_states": evaluated,
        "v1_row_parity_max_abs_difference": parity_max,
        "corrected_r27_identity_max_relative": identity_max,
        "v1_r27_min_relative_on_same_states": v1_r27_min,
        "jacobian_relative_max_difference_first_5": jacobian_max,
        "ranks": ranks,
        "constants": {
            "mass_flux_ratio_K0": constants.MASS_FLUX_RATIO,
            "floating_sheath_coefficient_ln_K0": constants.FLOATING_SHEATH_COEFFICIENT,
            "space_charge_limited_coefficient": constants.SPACE_CHARGE_LIMITED_COEFFICIENT,
            "critical_emission_yield_xenon": constants.CRITICAL_EMISSION_YIELD,
            "boltzmann_factor_no_emission": exp(-constants.FLOATING_SHEATH_COEFFICIENT),
            "boltzmann_factor_space_charge_limited": exp(-constants.SPACE_CHARGE_LIMITED_COEFFICIENT),
        },
    }


def _published_evaluation(target: PublishedFourCellState, regime: SheathRegime) -> dict[str, Any]:
    state = target.v2_state()
    inputs = target.v2_inputs(regime=regime, anode_row=AnodeRow.DECLARED_FALL)
    evaluation = evaluate_residual(state, inputs)
    worst = sorted(
        ((abs(value), ROW_IDS[index], value) for index, value in enumerate(evaluation.normalized)),
        reverse=True,
    )[:6]
    v1_state = target.core_state()
    v1_inputs = XenonGlobalInputs(target.anode_voltage_v, target.anode_current_a, target.cusp_probabilities)
    v1_eval = evaluate_plasma_residual_cpu(v1_state, v1_inputs)
    coefficients = inputs.sheath_coefficients()
    je0, derived = target.cathode_emission_a()
    return {
        "regime": regime.value,
        "je0_a": je0,
        "je0_derived_from_R01": derived,
        "max_normalized_residual_all_rows": max(abs(value) for value in evaluation.normalized),
        "max_normalized_residual_R00_R26": max(abs(value) for value in evaluation.normalized[:27]),
        "worst_rows": [{"row": row, "normalized": value} for _, row, value in worst],
        "corrected_R27_w": evaluation.raw[27],
        "v1_R27_w": v1_eval.raw[27],
        "v1_max_normalized_residual_R00_R26": max(abs(value) for value in v1_eval.normalized[:27]),
        "printed_sheath_drops_v": list(target.sheath_drops_v),
        "printed_sheath_drops_over_T": [
            target.sheath_drops_v[k] / target.electron_temperature_ev[k] for k in range(3)
        ],
        "sheath_row_drops_v": [coefficients[k] * target.electron_temperature_ev[k] for k in range(3)],
        "sheath_row_residuals_normalized": list(evaluation.normalized[28:31]),
        "cusp_energy_margins_ev": list(evaluation.cusp_energy_margins_ev),
        "violated_margins": [
            name for name, value in zip(evaluation.margin_names, evaluation.margins, strict=True) if value < 0.0
        ],
        "implied_anode_fall_v": target.implied_anode_fall_v(),
        "powers_w": {
            "published": target.published_powers_w,
            "v1_convention": v1_power_components(target),
            "v2_convention": v2_power_components(target, regime),
        },
    }


def _compare(target: PublishedFourCellState, state: SheathClosureState) -> dict[str, Any]:
    je0, _ = target.cathode_emission_a()
    theirs = {
        "phi_v": list(target.plasma_potential_v),
        "T_ev": list(target.electron_temperature_ev),
        "I_a": list(target.ionization_source_current_a),
        "je_a": [je0, *target.electron_current_a[1:]],
        "ji_a": list(target.ion_current_a),
        "jic_a": list(target.cusp_ion_current_a),
        "sheath_drop_v": list(target.sheath_drops_v),
    }
    ours = _state_dict(state)
    comparison: dict[str, Any] = {}
    for key, values in theirs.items():
        comparison[key] = {
            "published": values,
            "v2": ours[key],
            "difference": [b - a for a, b in zip(values, ours[key], strict=True)],
        }
    return comparison


def reproduction_section() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for target in REPRODUCTION_TARGETS:
        record: dict[str, Any] = {
            "identifier": target.identifier,
            "source": target.source,
            "notes": list(target.notes),
            "operating_point": {"Ua_v": target.anode_voltage_v, "Ia_a": target.anode_current_a},
            "published_p": list(target.cusp_probabilities),
            "published_state_under_v2_rows": {},
            "mode_c_published_potentials_declared": {},
            "mode_a_phi_1_solved": {},
        }
        for regime in REGIMES:
            record["published_state_under_v2_rows"][regime.value] = _published_evaluation(target, regime)
            mode_c = target.v2_inputs(regime=regime, anode_row=AnodeRow.DECLARED_FALL)
            solve_c = _solve_record(mode_c)
            if "state" in solve_c:
                solve_c["comparison"] = _compare(target, SheathClosureState.from_vector(
                    _vector_from_dict(solve_c["state"])
                ))
            record["mode_c_published_potentials_declared"][regime.value] = solve_c
            try:
                mode_a = target.v2_inputs(regime=regime, anode_row=AnodeRow.SHEATH)
            except PlasmaError as error:
                record["mode_a_phi_1_solved"][regime.value] = {"error": str(error)}
                continue
            solve_a = _solve_record(mode_a)
            if "state" in solve_a:
                solve_a["comparison"] = _compare(target, SheathClosureState.from_vector(
                    _vector_from_dict(solve_a["state"])
                ))
            record["mode_a_phi_1_solved"][regime.value] = solve_a
        records.append(record)
    return {"targets": records}


def _vector_from_dict(state: dict[str, Any]) -> tuple[float, ...]:
    return (
        *state["phi_v"],
        *state["T_ev"],
        *state["I_a"],
        *state["je_a"],
        *state["ji_a"],
        *state["jic_a"],
        *state["sheath_drop_v"],
        *state["p"],
    )


def closure_grid_section() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for regime in REGIMES:
        for access in GRID_ACCESS_FRACTIONS:
            cusps = tuple(CuspSheathSpec(regime=regime, access_fraction=access) for _ in range(3))
            for voltage in GRID_VOLTAGES_V:
                for current in GRID_CURRENTS_A:
                    inputs = SheathClosureInputs(
                        voltage,
                        current,
                        cusps,  # type: ignore[arg-type]
                        GRID_ANODE_CUSP_PROBABILITY,
                        cusp_loss_closure=CuspLossClosure.CL3_SHEATH_LIMITED,
                    )
                    record = _solve_record(inputs)
                    record.update(
                        {
                            "closure": "CL-3-sheath-limited",
                            "regime": regime.value,
                            "access_fraction": access,
                            "Ua_v": voltage,
                            "Ia_a": current,
                            "p_effective": [access * exp(-cusps[0].sheath_coefficient())] * 3,
                        }
                    )
                    cases.append(record)
        cusps = tuple(CuspSheathSpec(regime=regime) for _ in range(3))
        for voltage in GRID_VOLTAGES_V:
            for current in GRID_CURRENTS_A:
                inputs = SheathClosureInputs(
                    voltage,
                    current,
                    cusps,  # type: ignore[arg-type]
                    KORNFELD_DM92_P[3],
                    cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
                    declared_cusp_probabilities=KORNFELD_DM92_P[:3],
                )
                record = _solve_record(inputs)
                record.update(
                    {
                        "closure": "CL-1-declared (Kornfeld DM9.2 p)",
                        "regime": regime.value,
                        "access_fraction": None,
                        "Ua_v": voltage,
                        "Ia_a": current,
                        "p_effective": list(KORNFELD_DM92_P[:3]),
                    }
                )
                cases.append(record)

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        closed = [case for case in subset if case["converged"]]
        floors = [case["residual_floor"] for case in subset if not case["converged"] and isfinite(case["residual_floor"]) and case["residual_floor"] < 1e300]
        reduced_root = [case for case in subset if case["reduced"].get("state") is not None]
        margin_blocked = [
            case
            for case in reduced_root
            if not case["reduced"]["feasible_all_margins"] and case["reduced"]["feasible_without_cusp_energy_margin"]
        ]
        classifications: dict[str, int] = {}
        for case in subset:
            classifications[case["classification"]] = classifications.get(case["classification"], 0) + 1
        return {
            "cases": len(subset),
            "closed": len(closed),
            "closure_rate": len(closed) / len(subset) if subset else None,
            "manifold_root_found": len(reduced_root),
            "manifold_root_blocked_only_by_cusp_energy_margin": len(margin_blocked),
            "classifications": dict(sorted(classifications.items())),
            "residual_floor_min": min(floors) if floors else None,
            "residual_floor_max": max(floors) if floors else None,
            "reasons": sorted({case["reason"] for case in subset}),
        }

    summary: dict[str, Any] = {"all": summarize(cases)}
    for regime in REGIMES:
        for closure in ("CL-3-sheath-limited", "CL-1-declared (Kornfeld DM9.2 p)"):
            subset = [case for case in cases if case["regime"] == regime.value and case["closure"] == closure]
            summary[f"{closure} | {regime.value}"] = summarize(subset)
            if closure.startswith("CL-3"):
                for access in GRID_ACCESS_FRACTIONS:
                    sub = [case for case in subset if case["access_fraction"] == access]
                    summary[f"{closure} | {regime.value} | A={access}"] = summarize(sub)
    return {
        "grid": {
            "Ua_v": list(GRID_VOLTAGES_V),
            "Ia_a": list(GRID_CURRENTS_A),
            "access_fractions": list(GRID_ACCESS_FRACTIONS),
            "anode_cusp_probability": GRID_ANODE_CUSP_PROBABILITY,
            "potential_closure": "CL-3-potentials flat interior (0 V steps), anode fall 0 V declared, phi_1 solved through the anode sheath (R31)",
        },
        "summary": summary,
        "cases": cases,
    }


def _pic_wall_fields() -> tuple[tuple[float, float, float], dict[str, Any]]:
    try:
        from cft_revival.pic2d.fields import build_p2_psi_field
        from .pic_context import BORE_RADIUS_M, CUSP_PLANES_Z_M

        field, evidence = build_p2_psi_field(Path("..").resolve())
        values = []
        for z in CUSP_PLANES_Z_M:
            br, bz = field.field_cylindrical(BORE_RADIUS_M, z)
            values.append((br * br + bz * bz) ** 0.5)
        return tuple(values), {  # type: ignore[return-value]
            "source": "P2 bicubic field of the pic2d run sampled at (r = bore radius, z = cusp plane)",
            "checkpoint_path": evidence["checkpoint_path"],
            "checkpoint_file_sha256": evidence["checkpoint_file_sha256"],
        }
    except Exception as error:  # noqa: BLE001 - fallback is declared and recorded
        return (PIC_CUSP_WALL_FIELD_FALLBACK_T,) * 3, {
            "source": "declared fallback (field sampling unavailable)",
            "error": repr(error),
        }


def prefactor_section(pic_densities: tuple[float, float, float] | None) -> dict[str, Any]:
    fields, field_provenance = _pic_wall_fields()
    densities = pic_densities if pic_densities is not None else (2.13e17,) * 3
    records: list[dict[str, Any]] = []
    for voltage, current, label in ((PIC_VOLTAGE_V, PIC_DISCHARGE_A, "pic2d point"), (1000.0, 1.0, "Kornfeld point")):
        for prefactor in (1.0, 2.0, 3.0, 4.0):
            cusps = tuple(
                CuspSheathSpec(
                    regime=SheathRegime.SPACE_CHARGE_LIMITED,
                    electron_density_per_m3=densities[k],
                    wall_field_t=fields[k],
                )
                for k in range(3)
            )
            inputs = SheathClosureInputs(
                voltage,
                current,
                cusps,  # type: ignore[arg-type]
                GRID_ANODE_CUSP_PROBABILITY,
                cusp_loss_closure=CuspLossClosure.CL4_HYBRID_AREA,
                leak_width_prefactor=prefactor,
                wall_radius_m=0.002,
            )
            record = _solve_record(inputs)
            record.update({"closure": "CL-4-hybrid-area", "operating_point": label, "Ua_v": voltage, "Ia_a": current, "prefactor": prefactor})
            records.append(record)
        for ratio in (1.0, 2.0, 3.0, 4.0):
            cusps = tuple(
                CuspSheathSpec(regime=SheathRegime.FLOATING_NO_EMISSION, area_ratio=ratio, access_fraction=PIC_ACCESS_FRACTION)
                for _ in range(3)
            )
            inputs = SheathClosureInputs(
                voltage,
                current,
                cusps,  # type: ignore[arg-type]
                GRID_ANODE_CUSP_PROBABILITY,
                cusp_loss_closure=CuspLossClosure.CL3_SHEATH_LIMITED,
            )
            record = _solve_record(inputs)
            record.update(
                {
                    "closure": "CL-3-sheath-limited, area ratio rho = prefactor (no emission)",
                    "operating_point": label,
                    "Ua_v": voltage,
                    "Ia_a": current,
                    "prefactor": ratio,
                    "sheath_coefficient": cusps[0].sheath_coefficient(),
                    "p_effective": PIC_ACCESS_FRACTION * exp(-cusps[0].sheath_coefficient()),
                }
            )
            records.append(record)
    return {
        "declared_inputs": {
            "electron_density_per_m3": list(densities),
            "density_source": "pic2d steady-state v2 window maps, density-weighted segment means (cells 2-4) if available, else the run's mean 2.13e17",
            "wall_field_t": list(fields),
            "wall_field_provenance": field_provenance,
            "wall_radius_m": 0.002,
            "leak_width_prefactors": [1.0, 2.0, 3.0, 4.0],
        },
        "cases": records,
    }


def pic_context_section() -> dict[str, Any]:
    try:
        from .pic_context import load_pic_plateau_context
    except ImportError as error:
        return {"error": repr(error)}
    if not PIC_MAPS.is_file():
        return {"error": f"missing {PIC_MAPS.as_posix()}"}
    context = load_pic_plateau_context(PIC_MAPS)
    summary = json.loads(PIC_SUMMARY.read_text(encoding="utf-8")) if PIC_SUMMARY.is_file() else {}
    window = summary.get("window_currents_a", {})
    anode_fraction_pic = (
        window["anode_ion_a"] / window["anode_electron_a"] if "anode_ion_a" in window else None
    )
    segment_t = [segment.density_weighted_electron_temperature_ev for segment in context.segments]
    implied_fall = (
        segment_t[3] * log(constants.MASS_FLUX_RATIO * anode_fraction_pic) if anode_fraction_pic else None
    )
    pic_steps = context.potential_steps_v
    variants: dict[str, Any] = {}
    cusps_scl = tuple(CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED, access_fraction=PIC_ACCESS_FRACTION) for _ in range(3))
    variants["CL-3 SCL, flat interior, anode fall 0 V"] = _solve_record(
        SheathClosureInputs(PIC_VOLTAGE_V, PIC_DISCHARGE_A, cusps_scl, GRID_ANODE_CUSP_PROBABILITY)  # type: ignore[arg-type]
    )
    pic_fall = max(context.segments[3].density_weighted_potential_v - PIC_VOLTAGE_V, 0.0)
    variants["CL-3 SCL, PIC-declared steps and anode fall"] = _solve_record(
        SheathClosureInputs(
            PIC_VOLTAGE_V,
            PIC_DISCHARGE_A,
            cusps_scl,  # type: ignore[arg-type]
            GRID_ANODE_CUSP_PROBABILITY,
            potentials=PotentialClosure(
                interior_step_3_v=max(pic_steps[1], 0.0),
                interior_step_4_v=max(pic_steps[2], 0.0),
                anode_fall_v=pic_fall,
            ),
        )
    )
    cusps_g0 = tuple(CuspSheathSpec(access_fraction=PIC_ACCESS_FRACTION) for _ in range(3))
    variants["CL-3 no emission, flat interior"] = _solve_record(
        SheathClosureInputs(PIC_VOLTAGE_V, PIC_DISCHARGE_A, cusps_g0, GRID_ANODE_CUSP_PROBABILITY)  # type: ignore[arg-type]
    )
    cusps_k = tuple(CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED) for _ in range(3))
    variants["CL-1 Kornfeld DM9.2 p, SCL, flat interior"] = _solve_record(
        SheathClosureInputs(
            PIC_VOLTAGE_V,
            PIC_DISCHARGE_A,
            cusps_k,  # type: ignore[arg-type]
            KORNFELD_DM92_P[3],
            cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
            declared_cusp_probabilities=KORNFELD_DM92_P[:3],
        )
    )
    return {
        "claim_boundary": "model-to-model context only: the PIC run is a development, single-seed, under-resolved plateau and the v2 model is a development closure; neither validates the other",
        "pic": context.to_dict(),
        "pic_window_currents_a": window,
        "pic_anode_ion_fraction": anode_fraction_pic,
        "pic_implied_anode_fall_from_R31_v": implied_fall,
        "pic_segment_density_weighted_T_ev": segment_t,
        "declared_mapping": {
            "model cell 1 (plume)": "PIC cone region [17.95, 24] mm",
            "model cusp 2 (exit drop)": "PIC cusp at 17.95 mm",
            "model cell 2": "PIC [12, 17.95] mm",
            "model cusp 3": "PIC cusp at 12.0 mm",
            "model cell 3": "PIC [6, 12] mm",
            "model anode-cusp electrons (p_4)": "PIC cusp at 6.0 mm (dielectric in the PIC; metal anode in the model - mismatch)",
            "model cell 4": "PIC [0, 6] mm",
            "model cusp 1": "no magnetic counterpart; PIC dielectric cone wall reported beside it",
        },
        "model_inputs": {
            "Ua_v": PIC_VOLTAGE_V,
            "Ia_a": PIC_DISCHARGE_A,
            "access_fraction": PIC_ACCESS_FRACTION,
            "access_fraction_source": "cft_orbit_wall_loss_v4 primary-2N pooled wall-hit fraction (collisionless geometric access on the same P2 field; screening label)",
            "anode_cusp_probability": GRID_ANODE_CUSP_PROBABILITY,
        },
        "model_variants": variants,
    }


def v1_binding_section() -> dict[str, Any]:
    manifest = json.loads(PAPER_MANIFEST.read_text(encoding="utf-8")) if PAPER_MANIFEST.is_file() else None
    bound = {}
    if manifest is not None:
        bound = {entry["path"]: entry["sha256_lf"] for entry in manifest["executed_package"]["files"]}
    files = []
    for name in V1_PACKAGE_FILES:
        path = V1_PACKAGE_DIR / name
        digest = _sha256_lf(path)
        key = f"modern/src/cft_revival/plasma/{name}"
        files.append({"path": key, "sha256_lf": digest, "matches_manifest": bound.get(key) == digest})
    return {"manifest": PAPER_MANIFEST.as_posix() if manifest is not None else None, "files": files, "all_match": all(f["matches_manifest"] for f in files)}


def run(out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    structure = structure_section()
    reproduction = reproduction_section()
    pic = pic_context_section()
    densities = None
    if "pic" in pic:
        segments = pic["pic"]["segments"]
        densities = tuple(segments[k]["mean_electron_density_per_m3"] for k in (1, 2, 3))
    prefactor = prefactor_section(densities)
    grid = closure_grid_section()
    payload = {
        "schema_version": "1.0",
        "document_type": "plasma-v2-verification-record",
        "status": "DEVELOPMENT - not accepted evidence; no thruster claim",
        "generator": "python -m cft_revival.plasma_v2.verification",
        "structure": structure,
        "reproduction_targets": reproduction,
        "closure_grid": grid,
        "prefactor_sensitivity": prefactor,
        "pic_context": pic,
        "v1_package_binding": v1_binding_section(),
        "wall_seconds": time.perf_counter() - started,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/workstreams/plasma-v2-verification.json"))
    args = parser.parse_args(argv)
    payload = run(args.out)
    summary = payload["closure_grid"]["summary"]["all"]
    sys.stdout.write(
        f"wrote {args.out} in {payload['wall_seconds']:.1f} s; grid closure {summary['closed']}/{summary['cases']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
