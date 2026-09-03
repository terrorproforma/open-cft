"""Evaluation chain of the MDO L0 campaign v1.

design x = (Ua, Ia, mdot)  --+
                             +--> declared closure CL-1 --> corrected L0 model
uncertain theta = (p1..p4,   |     (cusp survival S(p))    (cft_revival.physics)
                   eta, zeta, gamma)

Every function in this module is deterministic pure Python and is replayed
bit-exactly by the assessment stage.  Nothing here is a thruster-performance
claim: the chain evaluates the L0 conservation model under declared input
uncertainty and nothing else (see ``protocol.json#claim_boundary``).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime import canonical_bytes
from cft_revival.optimization import (
    ConstraintSense,
    ContinuousConstraint,
    Design,
    ObjectiveDirection,
    ObjectiveSpec,
    Variable,
)
from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    XENON_ATOM_MASS_KG,
    BeamDivergenceFactors,
    ChargeStateFractions,
    IdealPerformanceResult,
    MassUtilization,
    PhysicsValidationError,
    PowerBoundaryInputs,
    PropellantMassFlow,
    XenonOperatingPoint,
    evaluate_performance,
)

# --------------------------------------------------------------------------
# Declared spaces (mirrored verbatim in protocol.json; the protocol test
# asserts both agree so the frozen document, not this module, is the authority)
# --------------------------------------------------------------------------

DESIGN_VARIABLES: tuple[Variable, ...] = (
    Variable("discharge_voltage_v", 150.0, 500.0, "V"),
    Variable("anode_current_a", 0.1, 2.5, "A"),
    Variable("propellant_mass_flow_kg_per_s", 2.0e-7, 2.0e-6, "kg/s"),
)

UNCERTAIN_INPUTS: tuple[tuple[str, float, float, str], ...] = (
    ("cusp_probability_cell_1", 0.0, 0.45, "1"),
    ("cusp_probability_cell_2", 0.0, 0.45, "1"),
    ("cusp_probability_cell_3", 0.0, 0.45, "1"),
    ("cusp_probability_cell_4", 0.0, 0.45, "1"),
    ("ionized_number_fraction", 0.65, 0.98, "1"),
    ("xe_double_plus_fraction_of_ions", 0.0, 0.15, "1"),
    ("axial_momentum_fraction_of_ion_momentum", 0.75, 0.98, "1"),
)
UNCERTAIN_NAMES: tuple[str, ...] = tuple(item[0] for item in UNCERTAIN_INPUTS)
CUSP_NAMES: tuple[str, ...] = UNCERTAIN_NAMES[:4]

FIXED_CLOSURES: dict[str, float] = {
    "cathode_input_power_w": 15.0,
    "ppu_efficiency_fraction": 0.9,
}

OBJECTIVES: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("axial_thrust_n", ObjectiveDirection.MAXIMIZE, "N", 0.06, 1e-15, 1e-12),
    ObjectiveSpec("specific_impulse_s", ObjectiveDirection.MAXIMIZE, "s", 3000.0, 1e-9, 1e-12),
    ObjectiveSpec(
        "thruster_electrical_to_beam_efficiency",
        ObjectiveDirection.MAXIMIZE,
        "1",
        1.0,
        1e-15,
        1e-12,
    ),
    ObjectiveSpec("anode_input_power_w", ObjectiveDirection.MINIMIZE, "W", 1300.0, 1e-9, 1e-12),
)
OBJECTIVE_NAMES: tuple[str, ...] = tuple(item.name for item in OBJECTIVES)

# Hypervolume reference point in physical units (worst acceptable corner).
REFERENCE_POINT: dict[str, float] = {
    "axial_thrust_n": 0.0,
    "specific_impulse_s": 0.0,
    "thruster_electrical_to_beam_efficiency": 0.0,
    "anode_input_power_w": 1300.0,
}

ROBUST_CONSTRAINT = ContinuousConstraint(
    "robust_beam_current_margin_a",
    ConstraintSense.GREATER_THAN_OR_EQUAL,
    0.0,
    "A",
    0.1,
)
NOMINAL_CONSTRAINT = ContinuousConstraint(
    "nominal_beam_current_margin_a",
    ConstraintSense.GREATER_THAN_OR_EQUAL,
    0.0,
    "A",
    0.1,
)

QMC_BASES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17)
QMC_SEED = 20260903
QMC_SAMPLE_SIZE = 64
CVAR_TAIL_FRACTION = 0.25
INFEASIBLE_CODE = "beam_current_exceeds_anode_current"
CLOSURE_ID = "CL-1-multiplicative-cusp-survival"


class ModelError(ValueError):
    """The evaluation chain received an input outside its declared spaces."""


# --------------------------------------------------------------------------
# Quasi-Monte-Carlo sample of the uncertain inputs (frozen and hashed)
# --------------------------------------------------------------------------


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def unit_qmc_rows(count: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Deterministic prime-base radical-inverse rows with a digital offset.

    Same construction as ``cft_revival.physics.workflows`` (offset
    ``17 + seed * 104729``); dependency-free so the sample never depends on a
    random-module or library version.
    """

    if count < 1 or seed < 0:
        raise ModelError("QMC sample requires count >= 1 and seed >= 0")
    start = 17 + seed * 104_729
    return tuple(
        tuple(_radical_inverse(start + row, base) for base in QMC_BASES)
        for row in range(1, count + 1)
    )


def uncertain_bounds(
    cusp_upper: float | None = None,
) -> tuple[tuple[str, float, float, str], ...]:
    """Declared bounds, optionally with a different cusp-probability upper bound.

    The alternative cusp upper bounds are used ONLY by the post-hoc sensitivity
    analysis; the campaign prior is the declared ``UNCERTAIN_INPUTS``.
    """

    if cusp_upper is None:
        return UNCERTAIN_INPUTS
    if not (0.0 <= cusp_upper < 1.0):
        raise ModelError("cusp probability upper bound must lie in [0, 1)")
    return tuple(
        (name, lower, cusp_upper if name in CUSP_NAMES else upper, units)
        for name, lower, upper, units in UNCERTAIN_INPUTS
    )


def uncertain_sample(
    *,
    count: int = QMC_SAMPLE_SIZE,
    seed: int = QMC_SEED,
    cusp_upper: float | None = None,
) -> tuple[dict[str, float], ...]:
    """Frozen QMC sample of the uncertain inputs in physical units."""

    bounds = uncertain_bounds(cusp_upper)
    rows = unit_qmc_rows(count, seed)
    sample = []
    for row in rows:
        theta = {}
        for (name, lower, upper, _units), coordinate in zip(bounds, row, strict=True):
            theta[name] = lower + coordinate * (upper - lower) if upper > lower else lower
        sample.append(theta)
    return tuple(sample)


def nominal_theta(cusp_upper: float | None = None) -> dict[str, float]:
    """Midpoint of every declared prior (the nominal evaluation point)."""

    return {
        name: 0.5 * (lower + upper)
        for name, lower, upper, _units in uncertain_bounds(cusp_upper)
    }


def sample_sha256(sample: Sequence[Mapping[str, float]]) -> str:
    return hashlib.sha256(canonical_bytes([dict(theta) for theta in sample])).hexdigest()


# --------------------------------------------------------------------------
# Closure CL-1 and the L0 operating point
# --------------------------------------------------------------------------


def cusp_survival(theta: Mapping[str, float]) -> float:
    """CL-1: fraction of produced ions reaching the exhaust, ``prod_k (1 - p_k)``.

    Declared closure, not a derived result.  The corrected cusp-current relation
    ``jic[k] = p[k] * je[k-1]`` (dielectric neutrality: equal electron and ion
    currents recombine at the wall cusp) is applied as a multiplicative cascade
    over the four cells; the wall current is a closed loop outside the anode
    circuit so the anode current is unaffected.
    """

    survival = 1.0
    for name in CUSP_NAMES:
        probability = float(theta[name])
        if not (0.0 <= probability < 1.0):
            raise ModelError(f"{name} must lie in [0, 1)")
        survival *= 1.0 - probability
    return survival


def charge_state_fractions(theta: Mapping[str, float]) -> ChargeStateFractions:
    ionized = float(theta["ionized_number_fraction"]) * cusp_survival(theta)
    double_share = float(theta["xe_double_plus_fraction_of_ions"])
    neutral = 1.0 - ionized
    double_plus = ionized * double_share
    plus = 1.0 - neutral - double_plus
    return ChargeStateFractions(neutral, plus, double_plus)


def beam_current_a(mass_flow_kg_per_s: float, theta: Mapping[str, float]) -> float:
    """Exhaust beam current implied by CL-1 and the L0 charge-state accounting."""

    fractions = charge_state_fractions(theta)
    return (
        float(mass_flow_kg_per_s)
        * (ELEMENTARY_CHARGE_C / XENON_ATOM_MASS_KG)
        * fractions.charge_weighted_ion_fraction
    )


def operating_point(
    values: Sequence[float], theta: Mapping[str, float]
) -> XenonOperatingPoint:
    """Build the validated L0 operating point for one (design, theta) pair."""

    voltage, anode_current, mass_flow = (float(item) for item in values)
    fractions = charge_state_fractions(theta)
    beam_current = mass_flow * (ELEMENTARY_CHARGE_C / XENON_ATOM_MASS_KG) * (
        fractions.charge_weighted_ion_fraction
    )
    beam_fraction = beam_current / anode_current
    anode_power = voltage * anode_current
    cathode_power = FIXED_CLOSURES["cathode_input_power_w"]
    thruster_power = math.fsum((anode_power, cathode_power))
    ppu_power = thruster_power / FIXED_CLOSURES["ppu_efficiency_fraction"]
    return XenonOperatingPoint(
        discharge_voltage_v=voltage,
        propellant_mass_flow=PropellantMassFlow(mass_flow),
        charge_state_fractions=fractions,
        mass_utilization=MassUtilization.from_charge_states(fractions),
        beam_divergence_factors=BeamDivergenceFactors(
            beam_fraction, float(theta["axial_momentum_fraction_of_ion_momentum"])
        ),
        power_boundaries=PowerBoundaryInputs(cathode_power, ppu_power),
    )


def objective_vector(result: IdealPerformanceResult) -> tuple[float, float, float, float]:
    efficiency = result.power_budget.thruster_electrical_to_beam_efficiency
    if efficiency is None:
        raise ModelError("L0 did not publish a thruster-electrical-to-beam efficiency")
    return (
        float(result.axial_thrust_n),
        float(result.specific_impulse_s),
        float(efficiency),
        float(result.power_budget.anode_input_power_w),
    )


def evaluate_l0(values: Sequence[float], theta: Mapping[str, float]) -> tuple[float, ...]:
    """One L0 evaluation; raises PhysicsValidationError outside the L0 domain."""

    return objective_vector(evaluate_performance(operating_point(values, theta)))


# --------------------------------------------------------------------------
# Robust aggregation
# --------------------------------------------------------------------------


def tail_count(sample_size: int, tail_fraction: float = CVAR_TAIL_FRACTION) -> int:
    if sample_size < 1 or not (0.0 < tail_fraction <= 1.0):
        raise ModelError("invalid CVaR tail specification")
    return max(1, math.ceil(sample_size * tail_fraction))


def cvar(values: Sequence[float], direction: ObjectiveDirection, tail: int) -> float:
    """Mean of the ``tail`` worst values (lower tail for maximise, upper for minimise)."""

    ordered = sorted(float(item) for item in values)
    if tail < 1 or tail > len(ordered):
        raise ModelError("CVaR tail must lie in [1, sample size]")
    worst = ordered[:tail] if direction is ObjectiveDirection.MAXIMIZE else ordered[-tail:]
    return math.fsum(worst) / tail


@dataclass(frozen=True)
class DesignEvaluation:
    """Complete, replayable record of one design evaluation."""

    values: tuple[float, float, float]
    design_id: str
    robust_margin_a: float
    nominal_margin_a: float
    status: str
    failure_code: str | None
    robust_objectives: tuple[float, float, float, float] | None
    robust_statistics: dict[str, dict[str, float]] | None
    nominal_objectives: tuple[float, float, float, float] | None
    sample_result_sha256: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "design": {
                "values": list(self.values),
                "variables": [variable.name for variable in DESIGN_VARIABLES],
                "design_id": self.design_id,
            },
            "constraints": {
                ROBUST_CONSTRAINT.name: self.robust_margin_a,
                NOMINAL_CONSTRAINT.name: self.nominal_margin_a,
            },
            "status": self.status,
            "failure_code": self.failure_code,
            "robust_objectives": (
                None
                if self.robust_objectives is None
                else dict(zip(OBJECTIVE_NAMES, self.robust_objectives, strict=True))
            ),
            "robust_statistics": self.robust_statistics,
            "nominal_objectives": (
                None
                if self.nominal_objectives is None
                else dict(zip(OBJECTIVE_NAMES, self.nominal_objectives, strict=True))
            ),
            "sample_result_sha256": self.sample_result_sha256,
        }


def make_design(values: Sequence[float], provenance: str = "") -> Design:
    return Design(tuple(float(item) for item in values), DESIGN_VARIABLES, provenance=provenance)


def evaluate_design(
    values: Sequence[float],
    sample: Sequence[Mapping[str, float]],
    *,
    nominal: Mapping[str, float] | None = None,
    tail_fraction: float = CVAR_TAIL_FRACTION,
) -> DesignEvaluation:
    """Evaluate one design under the frozen sample; fail closed on domain violations.

    * The robust constraint is the anode-current margin against the LARGEST
      beam current over the sample (worst case); a negative margin means the
      L0 domain (beam current <= anode current) is violated for some theta and
      the design has no robust objectives.
    * Robust objectives are the CVaR of each objective over the sample.
    * Nominal objectives use the prior midpoints when the nominal margin is
      non-negative (they feed the nominal front, never the optimiser).
    """

    design = make_design(values)
    voltage, anode_current, mass_flow = design.values
    nominal_point = dict(nominal) if nominal is not None else nominal_theta()
    beam_currents = [beam_current_a(mass_flow, theta) for theta in sample]
    robust_margin = anode_current - max(beam_currents)
    nominal_margin = anode_current - beam_current_a(mass_flow, nominal_point)

    nominal_objectives = None
    if nominal_margin >= 0.0:
        nominal_objectives = evaluate_l0(design.values, nominal_point)

    if robust_margin < 0.0:
        return DesignEvaluation(
            values=design.values,  # type: ignore[arg-type]
            design_id=design.design_id,
            robust_margin_a=robust_margin,
            nominal_margin_a=nominal_margin,
            status="infeasible",
            failure_code=INFEASIBLE_CODE,
            robust_objectives=None,
            robust_statistics=None,
            nominal_objectives=nominal_objectives,  # type: ignore[arg-type]
            sample_result_sha256=None,
        )

    try:
        rows = [evaluate_l0(design.values, theta) for theta in sample]
    except PhysicsValidationError as error:  # pragma: no cover - guarded by the margin
        raise ModelError(f"L0 rejected a design with non-negative margin: {error}") from error
    tail = tail_count(len(rows), tail_fraction)
    robust = []
    statistics: dict[str, dict[str, float]] = {}
    for index, objective in enumerate(OBJECTIVES):
        column = [row[index] for row in rows]
        value = cvar(column, objective.direction, tail)
        robust.append(value)
        statistics[objective.name] = {
            "cvar": value,
            "mean": math.fsum(column) / len(column),
            "minimum": min(column),
            "maximum": max(column),
        }
    return DesignEvaluation(
        values=design.values,  # type: ignore[arg-type]
        design_id=design.design_id,
        robust_margin_a=robust_margin,
        nominal_margin_a=nominal_margin,
        status="success",
        failure_code=None,
        robust_objectives=tuple(robust),  # type: ignore[arg-type]
        robust_statistics=statistics,
        nominal_objectives=nominal_objectives,  # type: ignore[arg-type]
        sample_result_sha256=hashlib.sha256(
            canonical_bytes([list(row) for row in rows])
        ).hexdigest(),
    )


# --------------------------------------------------------------------------
# Objective-space helpers shared by the optimisers and the assessment
# --------------------------------------------------------------------------


def normalized_objectives(vector: Sequence[float]) -> tuple[float, ...]:
    """Map physical objectives to the all-maximise unit hypervolume frame.

    ``(value - reference) / scale`` for maximise objectives and
    ``(reference - value) / scale`` for minimise objectives, so the reference
    point maps to the origin and hypervolume is dimensionless.
    """

    if len(vector) != len(OBJECTIVES):
        raise ModelError("objective vector length mismatch")
    out = []
    for value, objective in zip(vector, OBJECTIVES, strict=True):
        reference = REFERENCE_POINT[objective.name]
        if objective.direction is ObjectiveDirection.MAXIMIZE:
            out.append((float(value) - reference) / objective.comparison_scale)
        else:
            out.append((reference - float(value)) / objective.comparison_scale)
    return tuple(out)


def dominates_maximize(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(l >= r for l, r in zip(left, right, strict=True)) and any(
        l > r for l, r in zip(left, right, strict=True)
    )


def nondominated_indices(
    points: Sequence[Sequence[float]], *, relative_tolerance: float = 0.0
) -> tuple[int, ...]:
    """Indices of the nondominated points (all-maximise frame), stable order.

    Exact duplicates keep their first occurrence only.  Vectorised with numpy so
    the dense reference (thousands of points) is affordable; with the default
    zero tolerance the result equals the pairwise definition
    ``dominates_maximize`` (asserted by the tests).  A positive
    ``relative_tolerance`` makes dominance roundoff-aware: ``other`` dominates
    ``candidate`` when ``other >= candidate - tol`` in every component and
    ``other > candidate + tol`` in at least one, with ``tol`` the tolerance
    times the larger magnitude of the two components (the analysis uses this
    for the cusp-prior invariance check, never for a recorded front).
    """

    array = np.asarray([[float(c) for c in point] for point in points], dtype=float)
    if array.size == 0:
        return ()
    if array.ndim != 2:
        raise ModelError("points must be a two-dimensional sequence")
    if relative_tolerance < 0.0:
        raise ModelError("relative tolerance must be non-negative")
    count = array.shape[0]
    keep = np.ones(count, dtype=bool)
    _unique, first = np.unique(array, axis=0, return_index=True)
    duplicates = np.ones(count, dtype=bool)
    duplicates[first] = False
    keep &= ~duplicates
    for index in range(count):
        if not keep[index]:
            continue
        candidate = array[index]
        if relative_tolerance:
            tolerance = relative_tolerance * np.maximum(np.abs(array), np.abs(candidate))
        else:
            tolerance = 0.0
        ge = (array >= candidate - tolerance).all(axis=1)
        gt = (array > candidate + tolerance).any(axis=1)
        if bool((ge & gt).any()):
            keep[index] = False
    return tuple(int(index) for index in np.flatnonzero(keep))


def hypervolume(points: Sequence[Sequence[float]]) -> float:
    """Exact hypervolume of ``points`` (all-maximise frame) against the origin.

    Points with any non-positive coordinate contribute nothing.  Recursive
    slicing (HSO) over the last coordinate down to a vectorised two-dimensional
    sweep; exact, deterministic and dependency-free beyond numpy so the
    assessment replays it.  The tests cross-check it against pymoo's exact
    indicator on random sets.
    """

    clean = [
        tuple(float(c) for c in point)
        for point in points
        if all(float(c) > 0.0 for c in point)
    ]
    if not clean:
        return 0.0
    front = np.asarray([clean[index] for index in nondominated_indices(clean)], dtype=float)
    return float(_hv_recursive(front))


def _hv_2d(points: np.ndarray) -> float:
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    ordered = points[order]
    running = np.maximum.accumulate(ordered[:, 1])
    previous = np.concatenate(([0.0], running[:-1]))
    return float(np.sum(ordered[:, 0] * np.clip(running - previous, 0.0, None)))


def _hv_recursive(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    dims = points.shape[1]
    if dims == 1:
        return float(points[:, 0].max())
    if dims == 2:
        return _hv_2d(points)
    order = np.argsort(points[:, -1], kind="stable")
    ordered = points[order]
    levels = np.unique(ordered[:, -1])
    total = 0.0
    previous = 0.0
    for level in levels:
        remaining = ordered[ordered[:, -1] >= level]
        total += _hv_recursive(remaining[:, :-1]) * (float(level) - previous)
        previous = float(level)
    return total
