"""Deterministic launch construction, batched reduction, and binomial evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from hashlib import sha256
import json
from math import isfinite, pi, sqrt
from numbers import Real
from typing import Iterable, Sequence

import numpy as np

from .integrator import integrate_orbit
from .models import AxisymmetricField, ElectronLaunch, OrbitConfig, OrbitResult, OrbitValidationError, Termination


@dataclass(frozen=True, slots=True)
class ProbabilityEstimate:
    successes: int
    trials: int
    probability: float
    lower: float
    upper: float
    method: str = "wilson-95"


@dataclass(frozen=True, slots=True)
class EnsembleSummary:
    ensemble_id: str
    trial_count: int
    wall_hit: ProbabilityEstimate
    reflected: ProbabilityEstimate
    escaped: ProbabilityEstimate
    incomplete: ProbabilityEstimate
    termination_counts: tuple[tuple[str, int], ...]
    result_identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["termination_counts"] = dict(self.termination_counts)
        return value


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> ProbabilityEstimate:
    if (
        isinstance(successes, bool) or isinstance(trials, bool)
        or not isinstance(successes, int) or not isinstance(trials, int)
        or trials < 1 or not 0 <= successes <= trials
    ):
        raise OrbitValidationError("binomial counts are invalid")
    if isinstance(z, bool) or not isinstance(z, Real):
        raise OrbitValidationError("Wilson z must be a finite positive scalar")
    try:
        z = float(z)
    except (TypeError, ValueError, OverflowError) as error:
        raise OrbitValidationError("Wilson z must be a real scalar") from error
    if not isfinite(z) or z <= 0.0:
        raise OrbitValidationError("Wilson z must be a finite positive scalar")
    p = successes / trials
    z2 = z*z
    denominator = 1.0 + z2/trials
    centre = (p + z2/(2.0*trials)) / denominator
    half = z * sqrt(p*(1.0-p)/trials + z2/(4.0*trials*trials)) / denominator
    method = (
        "wilson-95"
        if z == 1.959963984540054
        else f"wilson-z={z:.17g}"
    )
    return ProbabilityEstimate(
        successes,
        trials,
        p,
        max(0.0, centre-half),
        min(1.0, centre+half),
        method,
    )


def deterministic_gyrophases(count: int) -> tuple[float, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise OrbitValidationError("gyrophase count must be a positive integer")
    return tuple(2.0*pi*index/count for index in range(count))


def build_launch_ensemble(
    *,
    ensemble_id: str,
    energies_ev: Sequence[float],
    pitch_angles_rad: Sequence[float],
    positions: Sequence[tuple[str, tuple[float, float, float]]],
    directions: Sequence[int] = (-1, 1),
    gyrophases_rad: Sequence[float] | None = None,
    gyrophase_count: int = 8,
) -> tuple[ElectronLaunch, ...]:
    """Cartesian product with stable hash-derived seed and identity."""

    if not ensemble_id:
        raise OrbitValidationError("ensemble_id must be non-empty")
    phases = tuple(gyrophases_rad) if gyrophases_rad is not None else deterministic_gyrophases(gyrophase_count)
    if not energies_ev or not pitch_angles_rad or not positions or not directions or not phases:
        raise OrbitValidationError("every launch dimension must be non-empty")
    launches: list[ElectronLaunch] = []
    for energy_index, energy in enumerate(energies_ev):
        for pitch_index, pitch in enumerate(pitch_angles_rad):
            for position_index, (surface_id, position) in enumerate(positions):
                for direction in directions:
                    for phase_index, phase in enumerate(phases):
                        identity = (
                            f"{ensemble_id}:E{energy_index}:P{pitch_index}:X{position_index}:"
                            f"D{direction:+d}:G{phase_index}"
                        )
                        seed = int.from_bytes(sha256(identity.encode()).digest()[:8], "big")
                        launches.append(
                            ElectronLaunch(identity, seed, energy, pitch, position, direction, phase, surface_id)
                        )
    return tuple(launches)


def _result_identity(results: Sequence[OrbitResult]) -> str:
    rows = []
    for result in sorted(results, key=lambda item: item.launch_id):
        row = asdict(result)
        row["termination"] = result.termination.value
        rows.append(row)
    return result_records_identity(rows)


def result_records_identity(results: Sequence[dict[str, object]]) -> str:
    ordered = copy.deepcopy(
        sorted(results, key=lambda item: str(item.get("launch_id", "")))
    )
    for result in ordered:
        witness = result.get("event_witness")
        if isinstance(witness, dict):
            for name in (
                "field_identity_sha256",
                "config_identity_sha256",
                "policy_identity_sha256",
            ):
                if name in witness:
                    witness[name] = "0" * 64
    return sha256(
        json.dumps(
            ordered,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def reduce_results(ensemble_id: str, results: Sequence[OrbitResult]) -> EnsembleSummary:
    if not results:
        raise OrbitValidationError("cannot reduce an empty ensemble")
    ordered = sorted(results, key=lambda item: item.launch_id)
    if len({item.launch_id for item in ordered}) != len(ordered):
        raise OrbitValidationError("launch result IDs must be unique")
    counts = {termination.value: 0 for termination in Termination}
    for result in ordered:
        counts[result.termination.value] += 1
    n = len(ordered)
    wall = counts[Termination.WALL_HIT.value]
    reflected = counts[Termination.REFLECTED.value]
    escaped = counts[Termination.DOMAIN_ESCAPE.value]
    incomplete = n - wall - reflected - escaped
    return EnsembleSummary(
        ensemble_id, n, wilson_interval(wall, n), wilson_interval(reflected, n),
        wilson_interval(escaped, n), wilson_interval(incomplete, n),
        tuple(sorted(counts.items())), _result_identity(ordered),
    )


def run_ensemble(
    ensemble_id: str,
    launches: Iterable[ElectronLaunch],
    field: AxisymmetricField,
    config: OrbitConfig,
    *,
    batch_size: int = 64,
) -> tuple[tuple[OrbitResult, ...], EnsembleSummary]:
    """Stable batches produce identical reduction independent of batch size."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise OrbitValidationError("batch_size must be a positive integer")
    ordered = sorted(tuple(launches), key=lambda item: item.launch_id)
    results: list[OrbitResult] = []
    for start in range(0, len(ordered), batch_size):
        results.extend(integrate_orbit(launch, field, config) for launch in ordered[start:start+batch_size])
    return tuple(results), reduce_results(ensemble_id, results)


def asymptotic_loss_cone_comparator(
    *,
    b_min_t: float,
    b_max_t: float,
    maximum_rho_over_scale: float,
    maximum_mu_relative_variation: float,
    complete_gyrocycles: int,
    rho_gate: float = 0.05,
    mu_gate: float = 0.1,
    minimum_complete_gyrocycles: int = 5,
) -> dict[str, object]:
    """Return an explicitly non-authoritative comparator only after adiabatic gates."""

    scalar_values = {
        "b_min_t": b_min_t,
        "b_max_t": b_max_t,
        "maximum_rho_over_scale": maximum_rho_over_scale,
        "maximum_mu_relative_variation": maximum_mu_relative_variation,
        "rho_gate": rho_gate,
        "mu_gate": mu_gate,
    }
    for name, value in scalar_values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise OrbitValidationError(f"{name} must be a finite nonnegative scalar")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise OrbitValidationError(f"{name} must be a real scalar") from error
        if not isfinite(converted) or converted < 0.0:
            raise OrbitValidationError(f"{name} must be a finite nonnegative scalar")
    if rho_gate <= 0.0 or mu_gate <= 0.0:
        raise OrbitValidationError("adiabatic scalar gates must be positive")
    if b_max_t < b_min_t:
        raise OrbitValidationError("magnetic comparator range is inverted")
    if (
        isinstance(complete_gyrocycles, bool)
        or not isinstance(complete_gyrocycles, int)
        or complete_gyrocycles < 0
        or isinstance(minimum_complete_gyrocycles, bool)
        or not isinstance(minimum_complete_gyrocycles, int)
        or minimum_complete_gyrocycles < 1
    ):
        raise OrbitValidationError("gyrocycle counts must be valid integers")
    passed = (
        0.0 < b_min_t <= b_max_t
        and maximum_rho_over_scale <= rho_gate
        and maximum_mu_relative_variation <= mu_gate
        and complete_gyrocycles >= minimum_complete_gyrocycles
    )
    return {
        "authority": "asymptotic_comparator_only_not_wall_loss_evidence",
        "adiabatic_gates_passed": passed,
        "loss_cone_probability": (
            1.0 - sqrt(max(0.0, 1.0 - b_min_t/b_max_t)) if passed else None
        ),
        "mirror_ratio": b_max_t/b_min_t if b_min_t > 0.0 else None,
        "gates": {
            "maximum_rho_over_scale": rho_gate,
            "maximum_mu_relative_variation": mu_gate,
            "minimum_complete_gyrocycles": minimum_complete_gyrocycles,
        },
    }


def probability_convergence(summaries: Sequence[EnsembleSummary]) -> dict[str, object]:
    if len(summaries) < 2:
        raise OrbitValidationError("probability convergence requires at least two maps/steps")
    values = [summary.wall_hit.probability for summary in summaries]
    return {
        "wall_hit_probabilities": values,
        "successive_absolute_changes": [abs(b-a) for a, b in zip(values, values[1:])],
        "confidence_intervals_overlap": [
            max(a.wall_hit.lower, b.wall_hit.lower) <= min(a.wall_hit.upper, b.wall_hit.upper)
            for a, b in zip(summaries, summaries[1:])
        ],
    }
