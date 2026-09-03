"""Reduced (manifold) solve of the v2 system.

Rows R00-R26 are solved explicitly by the read-only v1 parametrization
``cft_revival.plasma.potential_parametrized_state`` (given the four
potentials and the cusp probabilities).  The sheath rows R28-R30 then give
``Delta phi_s,k = c_s,k T_k`` explicitly, R32-R34 fix three potential
relations by declaration, and the remaining scalar (``phi_1`` or the anode
fall) is the root of the anode row R31.  The reduced solve is deterministic
(bracket scan + bisection), seeds the full least-squares solve, and is a
cross-check of it; it publishes nothing by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, pi, sqrt
from typing import Callable

from cft_revival.plasma import (
    PlasmaError,
    PlasmaState,
    PlasmaValidationError,
    potential_parametrized_state,
)

from .constants import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, MASS_FLUX_RATIO, XENON_MASS_KG
from .models import (
    AnodeRow,
    CuspLossClosure,
    FourthPotentialRow,
    SheathClosureInputs,
    SheathClosureState,
)


@dataclass(frozen=True, slots=True)
class ReducedSolve:
    """Outcome of the reduced solve (diagnostic; never a published state)."""

    state: SheathClosureState | None
    root_variable: str
    root_value: float | None
    anode_row_residual_v: float | None
    bracket_found: bool
    bisection_iterations: int
    probability_iterations: int
    reason: str


def sheath_drops(inputs: SheathClosureInputs, temperature: tuple[float, ...]) -> tuple[float, float, float]:
    coefficients = inputs.sheath_coefficients()
    return tuple(coefficients[k] * temperature[k] for k in range(3))  # type: ignore[return-value]


def _hybrid_leak_width(temperature_ev: float, wall_field_t: float) -> float:
    r_e = sqrt(2.0 * ELECTRON_MASS_KG * ELEMENTARY_CHARGE_C * temperature_ev) / (
        ELEMENTARY_CHARGE_C * wall_field_t
    )
    r_i = sqrt(XENON_MASS_KG * ELEMENTARY_CHARGE_C * temperature_ev) / (
        ELEMENTARY_CHARGE_C * wall_field_t
    )
    return sqrt(r_e * r_i)


def closure_probabilities(
    inputs: SheathClosureInputs, core: PlasmaState | None
) -> tuple[float, float, float]:
    """Evaluate p_1..p_3 under the declared closure for a given core state.

    CL-1 and CL-3 do not depend on the state (the Boltzmann factor of a
    floating sheath is ``exp(-c_s,k)``); CL-4 needs ``T_k`` and ``j_e,k-1``.
    """

    closure = inputs.cusp_loss_closure
    coefficients = inputs.sheath_coefficients()
    if closure is CuspLossClosure.CL1_DECLARED:
        return inputs.declared_cusp_probabilities
    if closure is CuspLossClosure.CL3_SHEATH_LIMITED:
        return tuple(  # type: ignore[return-value]
            inputs.cusps[k].access_fraction * exp(-coefficients[k]) for k in range(3)
        )
    if core is None:
        raise PlasmaValidationError("CL-4 needs a core state to evaluate the leak current")
    radius = inputs.wall_radius_m
    assert radius is not None
    values: list[float] = []
    for k in range(3):
        spec = inputs.cusps[k]
        density = spec.electron_density_per_m3
        field = spec.wall_field_t
        assert density is not None and field is not None
        temperature = core.electron_temperature_ev[k]
        entering = core.electron_current_a[k]
        if temperature <= 0.0 or entering <= 0.0:
            raise PlasmaValidationError("CL-4 needs positive T_k and j_e,k-1")
        mean_speed = sqrt(8.0 * ELEMENTARY_CHARGE_C * temperature / (pi * ELECTRON_MASS_KG))
        area = inputs.leak_width_prefactor * _hybrid_leak_width(temperature, field) * 2.0 * pi * radius
        current = 0.25 * ELEMENTARY_CHARGE_C * density * mean_speed * area * exp(-coefficients[k])
        values.append(min(current / entering, 1.0 - 1.0e-9))
    return tuple(values)  # type: ignore[return-value]


def manifold_state(
    inputs: SheathClosureInputs,
    potentials: tuple[float, float, float, float],
    *,
    max_probability_iterations: int = 200,
    probability_tolerance: float = 1.0e-14,
) -> tuple[SheathClosureState, int]:
    """Unique state satisfying R00-R30 and R35-R37 for the given potentials.

    Returns the state and the number of CL-4 fixed-point iterations (0 for
    CL-1/CL-3).  Raises ``PlasmaValidationError`` when the cascade has no
    admissible point (negative gain, non-positive transported current).
    """

    if inputs.cusp_loss_closure is not CuspLossClosure.CL4_HYBRID_AREA:
        probability = closure_probabilities(inputs, None)
        core = potential_parametrized_state(
            inputs.v1_inputs((*probability, inputs.anode_cusp_probability)), potentials
        )
        return SheathClosureState(core, sheath_drops(inputs, core.electron_temperature_ev), probability), 0
    probability = (1.0e-3, 1.0e-3, 1.0e-3)
    iterations = 0
    for iterations in range(1, max_probability_iterations + 1):
        core = potential_parametrized_state(
            inputs.v1_inputs((*probability, inputs.anode_cusp_probability)), potentials
        )
        proposal = closure_probabilities(inputs, core)
        updated = tuple(0.5 * (old + new) for old, new in zip(probability, proposal, strict=True))
        change = max(abs(new - old) for new, old in zip(updated, probability, strict=True))
        probability = updated  # type: ignore[assignment]
        if change <= probability_tolerance:
            break
    else:
        raise PlasmaValidationError("CL-4 probability fixed point did not converge")
    core = potential_parametrized_state(
        inputs.v1_inputs((*probability, inputs.anode_cusp_probability)), potentials
    )
    return SheathClosureState(core, sheath_drops(inputs, core.electron_temperature_ev), probability), iterations


def anode_row_residual_v(state: SheathClosureState, inputs: SheathClosureInputs) -> float:
    """Row R31 in volts for a manifold state."""

    core = state.core
    fall = core.plasma_potential_v[3] - inputs.anode_voltage_v
    if inputs.potentials.anode_row is AnodeRow.DECLARED_FALL:
        return fall - inputs.potentials.anode_fall_v
    anode_ion = -core.ion_current_a[4]
    electron = core.electron_current_a[4]
    if anode_ion <= 0.0 or electron <= 0.0:
        raise PlasmaValidationError("anode sheath row needs a positive anode ion current")
    return fall - core.electron_temperature_ev[3] * log(MASS_FLUX_RATIO * anode_ion / electron)


def _potentials_from(
    inputs: SheathClosureInputs, *, phi_1: float, anode_fall: float
) -> tuple[float, float, float, float]:
    closure = inputs.potentials
    phi_4 = inputs.anode_voltage_v + anode_fall
    phi_3 = phi_4 - closure.interior_step_4_v
    phi_2 = phi_3 - closure.interior_step_3_v
    return (phi_1, phi_2, phi_3, phi_4)


def _sign(value: float | None, undefined_sign: float) -> float:
    if value is None or not isfinite(value):
        return undefined_sign
    if value == 0.0:
        return 0.0
    return 1.0 if value > 0.0 else -1.0


def _bisect(
    function: Callable[[float], float | None],
    lower: float,
    upper: float,
    *,
    scan_points: int,
    tolerance: float,
    max_iterations: int = 400,
    undefined_sign: float = 1.0,
) -> tuple[float | None, float | None, bool, int]:
    """Deterministic bracket scan followed by bisection on a partially defined function.

    ``function`` returns ``None`` where the manifold has no admissible point.
    For the anode row that region is ``j_e4 <= Ia`` (no ion current to the
    anode), where the row tends to ``+inf``; an undefined value therefore
    carries the sign ``undefined_sign`` so that the edge of the admissible
    region can bracket a root together with a negative value just inside it.
    """

    if not (upper > lower):
        return None, None, False, 0
    grid = [lower + (upper - lower) * index / (scan_points - 1) for index in range(scan_points)]
    previous: tuple[float, float | None] | None = None
    bracket: tuple[float, float | None, float, float | None] | None = None
    for x in grid:
        value = function(x)
        if value is not None and isfinite(value) and value == 0.0:
            return x, 0.0, True, 0
        if previous is not None:
            previous_x, previous_value = previous
            both_undefined = value is None and previous_value is None
            if not both_undefined and _sign(value, undefined_sign) != _sign(previous_value, undefined_sign):
                bracket = (previous_x, previous_value, x, value)
                break
        previous = (x, value)
    if bracket is None:
        return None, None, False, 0
    a, fa, b, fb = bracket
    sign_a = _sign(fa, undefined_sign)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        middle = 0.5 * (a + b)
        value = function(middle)
        sign_middle = _sign(value, undefined_sign)
        if sign_middle == 0.0:
            a, fa, b, fb = middle, value, middle, value
            break
        if sign_middle == sign_a:
            a, fa = middle, value
        else:
            b, fb = middle, value
        if abs(b - a) <= tolerance * max(1.0, abs(a), abs(b)):
            break
    if fa is None and fb is not None:
        root = b
    elif fb is None and fa is not None:
        root = a
    else:
        root = 0.5 * (a + b)
    final = function(root)
    return root, final, True, iterations


def reduced_solve(
    inputs: SheathClosureInputs,
    *,
    scan_points: int = 257,
    tolerance: float = 1.0e-13,
) -> ReducedSolve:
    """Solve the v2 system on the R00-R26 manifold.

    * anode row SHEATH + anode fall declared: root in ``phi_1``;
    * anode row SHEATH + cathode coupling declared: root in the anode fall;
    * anode row DECLARED_FALL (+ cathode coupling declared): direct evaluation.
    """

    closure = inputs.potentials
    cathode = inputs.cathode_potential_v
    iterations_p = 0

    def build(phi_1: float, anode_fall: float) -> SheathClosureState | None:
        nonlocal iterations_p
        potentials = _potentials_from(inputs, phi_1=phi_1, anode_fall=anode_fall)
        if potentials[0] < cathode or potentials[0] > potentials[1]:
            return None
        try:
            state, iterations = manifold_state(inputs, potentials)
        except PlasmaError:
            return None
        iterations_p = max(iterations_p, iterations)
        return state

    if closure.anode_row is AnodeRow.DECLARED_FALL:
        assert closure.cathode_coupling_v is not None
        state = build(cathode + closure.cathode_coupling_v, closure.anode_fall_v)
        if state is None:
            return ReducedSolve(None, "none", None, None, False, 0, iterations_p, "cascade_infeasible")
        return ReducedSolve(
            state, "none", None, anode_row_residual_v(state, inputs), True, 0, iterations_p, "declared"
        )

    if closure.fourth_row is FourthPotentialRow.ANODE_FALL_DECLARED:
        anode_fall = closure.anode_fall_v
        phi_2 = _potentials_from(inputs, phi_1=cathode, anode_fall=anode_fall)[1]

        def residual_phi_1(phi_1: float) -> float | None:
            state = build(phi_1, anode_fall)
            if state is None:
                return None
            try:
                return anode_row_residual_v(state, inputs)
            except PlasmaError:
                return None

        lower = cathode + 1.0e-9 * inputs.anode_voltage_v
        root, value, found, iterations = _bisect(
            residual_phi_1, lower, phi_2, scan_points=scan_points, tolerance=tolerance
        )
        if root is None or not found:
            return ReducedSolve(None, "phi_1", None, None, False, iterations, iterations_p, "no_bracket")
        state = build(root, anode_fall)
        if state is None:
            return ReducedSolve(None, "phi_1", root, value, True, iterations, iterations_p, "cascade_infeasible")
        return ReducedSolve(state, "phi_1", root, value, True, iterations, iterations_p, "root")

    assert closure.cathode_coupling_v is not None
    phi_1 = cathode + closure.cathode_coupling_v

    def residual_fall(anode_fall: float) -> float | None:
        state = build(phi_1, anode_fall)
        if state is None:
            return None
        try:
            return anode_row_residual_v(state, inputs)
        except PlasmaError:
            return None

    root, value, found, iterations = _bisect(
        residual_fall, 0.0, 0.5 * inputs.anode_voltage_v, scan_points=scan_points, tolerance=tolerance
    )
    if root is None or not found:
        return ReducedSolve(None, "phi_4 - Ua", None, None, False, iterations, iterations_p, "no_bracket")
    state = build(phi_1, root)
    if state is None:
        return ReducedSolve(None, "phi_4 - Ua", root, value, True, iterations, iterations_p, "cascade_infeasible")
    return ReducedSolve(state, "phi_4 - Ua", root, value, True, iterations, iterations_p, "root")
