"""Evaluation chain of the MDO L0 campaign v2 (screened design catalogue x operating point).

design x = (k, Ua, Ia, mdot)          --+
  k = catalogue index (96 screened designs)  +--> declared closure CL-1 --> corrected L0 model
uncertain theta_k = (p_1..p_4 | design k,   |   S(p) = prod(1 - p_k)       (cft_revival.physics)
                     eta, zeta, gamma)      |   (sensitivity closure CL-2: S = 1 - p_pooled)

The per-cell wall-loss probabilities of design k are the accepted-2N screening counts of
``orbit_wall_loss_geometry_screening_v1`` through their Jeffreys Beta posterior
(:mod:`.catalogue`); no surrogate is involved.  Every function here is deterministic pure
Python/numpy and is replayed bit-exactly by the assessment stage.  Nothing here is a
thruster-performance claim (see ``protocol.json#claim_boundary``): CL-1 identifies the
collisionless test-particle wall-hit probability of a launch cell with the closure's
per-cusp survival factor, which the v1 scenario analysis showed is NOT the Kornfeld
per-cusp probability of a sustained discharge.

The L0 chain, CVaR aggregation, normalisation, dominance and hypervolume are the v1
functions (``experiments/mdo_l0_campaign_v1/model.py``, frozen at 4898d0fd) copied here so
this campaign's hash scope is self-contained; the only changes are the design identity
(catalogue index + operating point) and the closure switch.
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

from . import catalogue as cat

# --------------------------------------------------------------------------
# Declared spaces (mirrored verbatim in protocol.json; the protocol test asserts
# both agree so the frozen document, not this module, is the authority)
# --------------------------------------------------------------------------

CATALOGUE_VARIABLE = "catalogue_index"
CATALOGUE_SIZE = cat.CATALOGUE_SIZE

DESIGN_VARIABLES: tuple[Variable, ...] = (
    Variable("discharge_voltage_v", 150.0, 500.0, "V"),
    Variable("anode_current_a", 0.1, 2.5, "A"),
    Variable("propellant_mass_flow_kg_per_s", 2.0e-7, 2.0e-6, "kg/s"),
)
CONTINUOUS_NAMES: tuple[str, ...] = tuple(v.name for v in DESIGN_VARIABLES)

SHARED_UNCERTAIN_INPUTS = cat.SHARED_UNCERTAIN_INPUTS
CUSP_NAMES = cat.CUSP_NAMES
POOLED_NAME = cat.POOLED_NAME
UNCERTAIN_NAMES = cat.UNCERTAIN_NAMES

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

# Hypervolume reference point in physical units (worst acceptable corner) -- v1's, so
# hypervolumes are comparable between the campaigns.
REFERENCE_POINT: dict[str, float] = {
    "axial_thrust_n": 0.0,
    "specific_impulse_s": 0.0,
    "thruster_electrical_to_beam_efficiency": 0.0,
    "anode_input_power_w": 1300.0,
}

ROBUST_CONSTRAINT = ContinuousConstraint(
    "robust_beam_current_margin_a", ConstraintSense.GREATER_THAN_OR_EQUAL, 0.0, "A", 0.1
)
NOMINAL_CONSTRAINT = ContinuousConstraint(
    "nominal_beam_current_margin_a", ConstraintSense.GREATER_THAN_OR_EQUAL, 0.0, "A", 0.1
)

QMC_SAMPLE_SIZE = cat.QMC_SAMPLE_SIZE
CVAR_TAIL_FRACTION = 0.25
INFEASIBLE_CODE = "beam_current_exceeds_anode_current"

CLOSURE_CL1 = "CL-1-multiplicative-cusp-survival-per-cell-test-particle-wall-loss"
CLOSURE_CL2 = "CL-2-pooled-test-particle-wall-loss-survival"
CLOSURES: tuple[str, ...] = (CLOSURE_CL1, CLOSURE_CL2)
CLOSURE_ID = CLOSURE_CL1


class ModelError(ValueError):
    """The evaluation chain received an input outside its declared spaces."""


# --------------------------------------------------------------------------
# Closures and the L0 operating point
# --------------------------------------------------------------------------


def survival(theta: Mapping[str, float], closure: str = CLOSURE_CL1) -> float:
    """Fraction of produced ions reaching the exhaust under the declared closure.

    CL-1: ``prod_k (1 - p_k)`` with ``p_k`` = P(wall | launch cell k, design) -- the corrected
    cusp-current relation ``jic[k] = p[k] * je[k-1]`` (dielectric neutrality) applied as a
    multiplicative cascade over the four launch cells, exactly as in v1 but with the
    design's own screened probabilities.  CL-2 (sensitivity): ``1 - p_pooled`` with the
    design's pooled P(wall) over all 512 launches.  Both are DECLARED closures; neither is
    derived or validated; the identification of a test-particle wall-hit probability with
    a discharge cusp-loss probability is the declared assumption of this campaign.
    """

    if closure == CLOSURE_CL1:
        value = 1.0
        for name in CUSP_NAMES:
            probability = float(theta[name])
            if not (0.0 <= probability < 1.0):
                raise ModelError(f"{name} must lie in [0, 1)")
            value *= 1.0 - probability
        return value
    if closure == CLOSURE_CL2:
        probability = float(theta[POOLED_NAME])
        if not (0.0 <= probability < 1.0):
            raise ModelError(f"{POOLED_NAME} must lie in [0, 1)")
        return 1.0 - probability
    raise ModelError(f"unknown closure {closure!r}")


def charge_state_fractions(theta: Mapping[str, float], closure: str = CLOSURE_CL1) -> ChargeStateFractions:
    ionized = float(theta["ionized_number_fraction"]) * survival(theta, closure)
    double_share = float(theta["xe_double_plus_fraction_of_ions"])
    neutral = 1.0 - ionized
    double_plus = ionized * double_share
    plus = 1.0 - neutral - double_plus
    return ChargeStateFractions(neutral, plus, double_plus)


def beam_current_a(mass_flow_kg_per_s: float, theta: Mapping[str, float], closure: str = CLOSURE_CL1) -> float:
    """Exhaust beam current implied by the closure and the L0 charge-state accounting."""

    fractions = charge_state_fractions(theta, closure)
    return float(mass_flow_kg_per_s) * (ELEMENTARY_CHARGE_C / XENON_ATOM_MASS_KG) * fractions.charge_weighted_ion_fraction


def operating_point(values: Sequence[float], theta: Mapping[str, float], closure: str = CLOSURE_CL1) -> XenonOperatingPoint:
    """Build the validated L0 operating point for one (operating point, theta) pair."""

    voltage, anode_current, mass_flow = (float(item) for item in values)
    fractions = charge_state_fractions(theta, closure)
    beam_current = mass_flow * (ELEMENTARY_CHARGE_C / XENON_ATOM_MASS_KG) * fractions.charge_weighted_ion_fraction
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


def evaluate_l0(values: Sequence[float], theta: Mapping[str, float], closure: str = CLOSURE_CL1) -> tuple[float, ...]:
    """One L0 evaluation; raises PhysicsValidationError outside the L0 domain."""

    return objective_vector(evaluate_performance(operating_point(values, theta, closure)))


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


# --------------------------------------------------------------------------
# Design identity and the evaluation context
# --------------------------------------------------------------------------


def check_values(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != len(DESIGN_VARIABLES):
        raise ModelError("operating point must have three coordinates")
    out = []
    for value, variable in zip(values, DESIGN_VARIABLES, strict=True):
        v = float(value)
        if not math.isfinite(v) or not (variable.lower <= v <= variable.upper):
            raise ModelError(f"{variable.name}={v} outside [{variable.lower}, {variable.upper}]")
        out.append(v)
    return out[0], out[1], out[2]


def check_index(index: int) -> int:
    if isinstance(index, bool) or int(index) != index or not (0 <= int(index) < CATALOGUE_SIZE):
        raise ModelError(f"catalogue index must be an integer in [0, {CATALOGUE_SIZE})")
    return int(index)


def design_id(index: int, case_id: str, values: Sequence[float]) -> str:
    return hashlib.sha256(
        canonical_bytes({"catalogue_index": int(index), "case_id": str(case_id), "values": [float(v) for v in values]})
    ).hexdigest()


@dataclass(frozen=True)
class EvaluationContext:
    """The catalogue with its frozen per-design sample and nominal points, under one closure."""

    designs: tuple[cat.CatalogueDesign, ...]
    sample: tuple[tuple[Mapping[str, float], ...], ...]
    nominal: tuple[Mapping[str, float], ...]
    closure: str = CLOSURE_CL1
    tail_fraction: float = CVAR_TAIL_FRACTION

    def __post_init__(self) -> None:
        if self.closure not in CLOSURES:
            raise ModelError(f"unknown closure {self.closure!r}")
        if not (len(self.designs) == len(self.sample) == len(self.nominal) == CATALOGUE_SIZE):
            raise ModelError("context must cover the whole catalogue")

    @property
    def sample_sha256(self) -> str:
        return cat.catalogue_sample_sha256(self.sample)


def build_context(
    designs: Sequence[cat.CatalogueDesign] | None = None,
    *,
    closure: str = CLOSURE_CL1,
    width_scale: float | None = 1.0,
    tail_fraction: float = CVAR_TAIL_FRACTION,
) -> EvaluationContext:
    """Load the catalogue (if not given) and freeze its sample under the given posterior width."""

    catalogue = tuple(designs) if designs is not None else cat.load_catalogue()
    return EvaluationContext(
        designs=catalogue,
        sample=cat.catalogue_sample(catalogue, width_scale=width_scale),
        nominal=tuple(cat.design_nominal_theta(d) for d in catalogue),
        closure=closure,
        tail_fraction=tail_fraction,
    )


@dataclass(frozen=True)
class DesignEvaluation:
    """Complete, replayable record of one design evaluation."""

    catalogue_index: int
    case_id: str
    screening_design_id: str
    values: tuple[float, float, float]
    design_id: str
    closure: str
    robust_margin_a: float
    nominal_margin_a: float
    status: str
    failure_code: str | None
    robust_objectives: tuple[float, float, float, float] | None
    robust_statistics: dict[str, dict[str, float]] | None
    nominal_objectives: tuple[float, float, float, float] | None
    survival_statistics: dict[str, float]
    sample_result_sha256: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "design": {
                "catalogue_index": self.catalogue_index,
                "case_id": self.case_id,
                "screening_design_id": self.screening_design_id,
                "values": list(self.values),
                "variables": list(CONTINUOUS_NAMES),
                "design_id": self.design_id,
            },
            "closure": self.closure,
            "constraints": {
                ROBUST_CONSTRAINT.name: self.robust_margin_a,
                NOMINAL_CONSTRAINT.name: self.nominal_margin_a,
            },
            "status": self.status,
            "failure_code": self.failure_code,
            "robust_objectives": (
                None if self.robust_objectives is None else dict(zip(OBJECTIVE_NAMES, self.robust_objectives, strict=True))
            ),
            "robust_statistics": self.robust_statistics,
            "nominal_objectives": (
                None if self.nominal_objectives is None else dict(zip(OBJECTIVE_NAMES, self.nominal_objectives, strict=True))
            ),
            "survival_statistics": self.survival_statistics,
            "sample_result_sha256": self.sample_result_sha256,
        }


def evaluate_design(index: int, values: Sequence[float], context: EvaluationContext) -> DesignEvaluation:
    """Evaluate one (design, operating point) under the frozen per-design sample; fail closed.

    * The robust constraint is the anode-current margin against the LARGEST beam current over
      the design's 64 draws (worst sampled case); a negative margin means the L0 domain
      (beam current <= anode current) is violated for some theta and the design has no
      robust objectives.
    * Robust objectives are the CVaR (mean of the 16 worst of 64) of each objective.
    * Nominal objectives use the posterior means / prior midpoints when the nominal margin is
      non-negative (they feed the nominal front, never the optimiser).
    """

    k = check_index(index)
    voltage, anode_current, mass_flow = check_values(values)
    design = context.designs[k]
    sample = context.sample[k]
    nominal_point = context.nominal[k]
    closure = context.closure
    survivals = [survival(theta, closure) for theta in sample]
    beam_currents = [beam_current_a(mass_flow, theta, closure) for theta in sample]
    robust_margin = anode_current - max(beam_currents)
    nominal_margin = anode_current - beam_current_a(mass_flow, nominal_point, closure)
    survival_statistics = {
        "minimum": min(survivals),
        "maximum": max(survivals),
        "mean": math.fsum(survivals) / len(survivals),
        "nominal": survival(nominal_point, closure),
    }
    identity = design_id(k, design.case_id, (voltage, anode_current, mass_flow))

    nominal_objectives = None
    if nominal_margin >= 0.0:
        nominal_objectives = evaluate_l0((voltage, anode_current, mass_flow), nominal_point, closure)

    if robust_margin < 0.0:
        return DesignEvaluation(
            catalogue_index=k,
            case_id=design.case_id,
            screening_design_id=design.design_id,
            values=(voltage, anode_current, mass_flow),
            design_id=identity,
            closure=closure,
            robust_margin_a=robust_margin,
            nominal_margin_a=nominal_margin,
            status="infeasible",
            failure_code=INFEASIBLE_CODE,
            robust_objectives=None,
            robust_statistics=None,
            nominal_objectives=nominal_objectives,  # type: ignore[arg-type]
            survival_statistics=survival_statistics,
            sample_result_sha256=None,
        )

    try:
        rows = [evaluate_l0((voltage, anode_current, mass_flow), theta, closure) for theta in sample]
    except PhysicsValidationError as error:  # pragma: no cover - guarded by the margin
        raise ModelError(f"L0 rejected a design with non-negative margin: {error}") from error
    tail = tail_count(len(rows), context.tail_fraction)
    robust = []
    statistics: dict[str, dict[str, float]] = {}
    for column_index, objective in enumerate(OBJECTIVES):
        column = [row[column_index] for row in rows]
        value = cvar(column, objective.direction, tail)
        robust.append(value)
        statistics[objective.name] = {
            "cvar": value,
            "mean": math.fsum(column) / len(column),
            "minimum": min(column),
            "maximum": max(column),
        }
    return DesignEvaluation(
        catalogue_index=k,
        case_id=design.case_id,
        screening_design_id=design.design_id,
        values=(voltage, anode_current, mass_flow),
        design_id=identity,
        closure=closure,
        robust_margin_a=robust_margin,
        nominal_margin_a=nominal_margin,
        status="success",
        failure_code=None,
        robust_objectives=tuple(robust),  # type: ignore[arg-type]
        robust_statistics=statistics,
        nominal_objectives=nominal_objectives,  # type: ignore[arg-type]
        survival_statistics=survival_statistics,
        sample_result_sha256=hashlib.sha256(canonical_bytes([list(row) for row in rows])).hexdigest(),
    )


# --------------------------------------------------------------------------
# Objective-space helpers shared by the optimisers and the assessment (v1 verbatim)
# --------------------------------------------------------------------------


def normalized_objectives(vector: Sequence[float]) -> tuple[float, ...]:
    """Map physical objectives to the all-maximise unit hypervolume frame.

    ``(value - reference) / scale`` for maximise objectives and ``(reference - value) / scale``
    for minimise objectives, so the reference point maps to the origin and hypervolume is
    dimensionless (identical to v1, hence comparable).
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


def nondominated_indices(points: Sequence[Sequence[float]], *, relative_tolerance: float = 0.0) -> tuple[int, ...]:
    """Indices of the nondominated points (all-maximise frame), stable order; first duplicate kept."""

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


def nondominated_indices_blockwise(
    points: Sequence[Sequence[float]], blocks: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    """Nondominated indices of a large set given a partition into blocks.

    The global front is a subset of the union of the per-block fronts, so filtering each
    block and then the union is exact; used for the dense reference (96 designs x 1024
    operating points), where the pairwise filter over all points would be quadratic.
    """

    survivors: list[int] = []
    for block in blocks:
        block_points = [points[i] for i in block]
        survivors.extend(block[j] for j in nondominated_indices(block_points))
    survivors.sort()
    final = nondominated_indices([points[i] for i in survivors])
    return tuple(survivors[j] for j in final)


def hypervolume(points: Sequence[Sequence[float]]) -> float:
    """Exact hypervolume of ``points`` (all-maximise frame) against the origin (v1 algorithm)."""

    clean = [tuple(float(c) for c in point) for point in points if all(float(c) > 0.0 for c in point)]
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
