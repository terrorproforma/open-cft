"""Checked configuration and JSON artifacts for the verified L0 physics model."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import json
from math import fsum, isfinite
from numbers import Real
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Mapping, Sequence

from .models import (
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
)
from .reference import evaluate_batch, evaluate_performance

L0_MODEL_FIDELITY = "L0-conservation-reduced-performance"
L0_MODEL_CLAIM = (
    "Conservation-based reduced-performance calculation with externally supplied "
    "charge-state, beam-current, divergence, cathode, and PPU factors; not a "
    "calibrated or predictive plasma model."
)
MAX_L0_SWEEP_BATCH_SIZE = 1_000_000

_POINT_TOP_LEVEL_FIELDS = {
    "document_type",
    "schema_version",
    "model_fidelity",
    "hypothetical_inputs",
    "description",
    "inputs",
}
_POINT_INPUT_FIELDS = {
    "discharge_voltage_v",
    "propellant_mass_flow_kg_per_s",
    "charge_state_number_fractions",
    "beam_current_fraction_of_anode_current",
    "axial_momentum_fraction_of_ion_momentum",
    "cathode_input_power_w",
    "ppu_input_power_w",
    "xenon_atom_mass_kg",
}
_CHARGE_FRACTION_FIELDS = {
    "xe_neutral",
    "xe_plus",
    "xe_double_plus",
}
_SWEEP_RANGE_FIELDS = {
    "discharge_voltage_v",
    "propellant_mass_flow_kg_per_s",
    "ionized_number_fraction",
    "xe_double_plus_fraction_of_ions",
    "beam_current_fraction_of_anode_current",
    "axial_momentum_fraction_of_ion_momentum",
    "cathode_input_power_w",
    "ppu_efficiency_fraction",
}
_SWEEP_TOP_LEVEL_FIELDS = {
    "document_type",
    "schema_version",
    "model_fidelity",
    "hypothetical_inputs",
    "description",
    "batch_size",
    "seed",
    "ranges",
}


class PhysicsConfigurationError(PhysicsValidationError):
    """A checked L0 JSON configuration is malformed or semantically invalid."""


def _allow_keys(
    raw: Mapping[str, Any],
    *,
    path: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise PhysicsConfigurationError(
            f"{path} contains unknown field(s): {names}"
        )
    missing = required - set(raw)
    if missing:
        names = ", ".join(sorted(missing))
        raise PhysicsConfigurationError(
            f"{path} is missing required field(s): {names}"
        )


def _reject_json_constant(value: str) -> None:
    raise PhysicsConfigurationError(f"L0 configuration contains {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhysicsConfigurationError(
                f"L0 configuration contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _validate_json_finite(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise PhysicsConfigurationError(
                f"L0 configuration contains non-finite number at {path}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_finite(item, f"{path}.{key}")
        return
    raise PhysicsConfigurationError(
        f"L0 configuration contains unsupported value at {path}"
    )


def load_l0_json(path: Path) -> Mapping[str, Any]:
    """Load strict JSON without duplicate fields or non-finite extensions."""

    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhysicsConfigurationError(
            f"cannot read L0 configuration {path}: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise PhysicsConfigurationError("L0 configuration root must be an object")
    _validate_json_finite(decoded)
    return decoded


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicsConfigurationError(f"{name} must be a JSON object")
    return value


def _text(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise PhysicsConfigurationError(f"{name} must be non-empty text")
    return value


def _number(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PhysicsConfigurationError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise PhysicsConfigurationError(f"{name} must be finite")
    return converted


def _integer(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PhysicsConfigurationError(f"{name} must be an integer")
    return value


def _optional_number(
    mapping: Mapping[str, Any], name: str, default: float
) -> float:
    if name not in mapping:
        return default
    return _number(mapping, name)


def _require_header(
    raw: Mapping[str, Any],
    document_type: str,
    *,
    allowed: set[str],
    required: set[str],
) -> None:
    _allow_keys(raw, path="$", allowed=allowed, required=required)
    if _text(raw, "document_type") != document_type:
        raise PhysicsConfigurationError(
            f"document_type must be {document_type!r}"
        )
    if _text(raw, "schema_version") != "1.0":
        raise PhysicsConfigurationError("schema_version must be '1.0'")
    if _text(raw, "model_fidelity") != L0_MODEL_FIDELITY:
        raise PhysicsConfigurationError(
            f"model_fidelity must be {L0_MODEL_FIDELITY!r}"
        )
    if raw.get("hypothetical_inputs") is not True:
        raise PhysicsConfigurationError(
            "hypothetical_inputs must be true; L0 examples are not fitted data"
        )
    if "description" in raw:
        _text(raw, "description")


def operating_point_from_config(raw: Mapping[str, Any]) -> XenonOperatingPoint:
    """Build one validated operating point from an SI-explicit JSON mapping."""

    _require_header(
        raw,
        "cft-revival-l0-operating-point",
        allowed=_POINT_TOP_LEVEL_FIELDS,
        required=_POINT_TOP_LEVEL_FIELDS - {"description"},
    )
    inputs = _mapping("inputs", raw.get("inputs"))
    _allow_keys(
        inputs,
        path="$.inputs",
        allowed=_POINT_INPUT_FIELDS,
        required=_POINT_INPUT_FIELDS - {"xenon_atom_mass_kg"},
    )
    fractions_raw = _mapping(
        "inputs.charge_state_number_fractions",
        inputs.get("charge_state_number_fractions"),
    )
    _allow_keys(
        fractions_raw,
        path="$.inputs.charge_state_number_fractions",
        allowed=_CHARGE_FRACTION_FIELDS,
        required=_CHARGE_FRACTION_FIELDS,
    )
    fractions = ChargeStateFractions(
        _number(fractions_raw, "xe_neutral"),
        _number(fractions_raw, "xe_plus"),
        _number(fractions_raw, "xe_double_plus"),
    )
    return XenonOperatingPoint(
        discharge_voltage_v=_number(inputs, "discharge_voltage_v"),
        propellant_mass_flow=PropellantMassFlow(
            _number(inputs, "propellant_mass_flow_kg_per_s")
        ),
        charge_state_fractions=fractions,
        mass_utilization=MassUtilization.from_charge_states(fractions),
        beam_divergence_factors=BeamDivergenceFactors(
            _number(inputs, "beam_current_fraction_of_anode_current"),
            _number(inputs, "axial_momentum_fraction_of_ion_momentum"),
        ),
        power_boundaries=PowerBoundaryInputs(
            _number(inputs, "cathode_input_power_w"),
            _number(inputs, "ppu_input_power_w"),
        ),
        xenon_atom_mass_kg=_optional_number(
            inputs, "xenon_atom_mass_kg", XENON_ATOM_MASS_KG
        ),
    )


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def operating_point_to_dict(point: XenonOperatingPoint) -> dict[str, Any]:
    fractions = point.charge_state_fractions
    factors = point.beam_divergence_factors
    boundaries = point.power_boundaries
    return {
        "discharge_voltage_v": point.discharge_voltage_v,
        "propellant_mass_flow_kg_per_s": point.propellant_mass_flow.kg_per_s,
        "charge_state_number_fractions": {
            "xe_neutral": fractions.xe_neutral,
            "xe_plus": fractions.xe_plus,
            "xe_double_plus": fractions.xe_double_plus,
        },
        "mass_utilization_fraction_of_inlet_mass": (
            point.mass_utilization.fraction_of_inlet_mass
        ),
        "beam_current_fraction_of_anode_current": (
            factors.beam_current_fraction_of_anode_current
        ),
        "axial_momentum_fraction_of_ion_momentum": (
            factors.axial_momentum_fraction_of_ion_momentum
        ),
        "cathode_input_power_w": boundaries.cathode_input_power_w,
        "ppu_input_power_w": boundaries.ppu_input_power_w,
        "xenon_atom_mass_kg": point.xenon_atom_mass_kg,
    }


def result_to_dict(result: IdealPerformanceResult) -> dict[str, Any]:
    return _json_value(asdict(result))  # type: ignore[return-value]


def evaluate_operating_point_artifact(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    point = operating_point_from_config(raw)
    result = evaluate_performance(point)
    return {
        "document_type": "cft-revival-l0-result",
        "schema_version": "1.0",
        "model_fidelity": L0_MODEL_FIDELITY,
        "model_claim": L0_MODEL_CLAIM,
        "backend": {"implementation": "python-reference", "device": "cpu"},
        "hypothetical_inputs": True,
        "input": operating_point_to_dict(point),
        "result": result_to_dict(result),
    }


_SWEEP_BASES = (2, 3, 5, 7, 11, 13, 17, 19)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def _coordinates(count: int, seed: int) -> tuple[tuple[float, ...], ...]:
    # A deterministic digital offset avoids a random-module/version dependency.
    start = 17 + seed * 104_729
    return tuple(
        tuple(
            _radical_inverse(start + row, base)
            for base in _SWEEP_BASES
        )
        for row in range(1, count + 1)
    )


def _range(
    ranges: Mapping[str, Any], name: str, *, lower_bound: float | None = None
) -> tuple[float, float]:
    raw = ranges.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise PhysicsConfigurationError(f"ranges.{name} must be [minimum, maximum]")
    values = {"minimum": raw[0], "maximum": raw[1]}
    minimum = _number(values, "minimum")
    maximum = _number(values, "maximum")
    if minimum >= maximum:
        raise PhysicsConfigurationError(f"ranges.{name} must increase")
    if lower_bound is not None and minimum < lower_bound:
        raise PhysicsConfigurationError(
            f"ranges.{name} minimum must be at least {lower_bound}"
        )
    return minimum, maximum


def _interpolate(bounds: tuple[float, float], coordinate: float) -> float:
    return bounds[0] + coordinate * (bounds[1] - bounds[0])


def sweep_points_from_config(
    raw: Mapping[str, Any],
) -> tuple[tuple[XenonOperatingPoint, ...], dict[str, Any]]:
    """Generate a deterministic, domain-checked L0 operating-point batch."""

    _require_header(
        raw,
        "cft-revival-l0-sweep",
        allowed=_SWEEP_TOP_LEVEL_FIELDS,
        required=_SWEEP_TOP_LEVEL_FIELDS - {"description"},
    )
    count = _integer(raw, "batch_size")
    seed = _integer(raw, "seed")
    if not 1 <= count <= MAX_L0_SWEEP_BATCH_SIZE:
        raise PhysicsConfigurationError(
            f"batch_size must lie in [1, {MAX_L0_SWEEP_BATCH_SIZE}]"
        )
    if seed < 0:
        raise PhysicsConfigurationError("seed must be non-negative")
    ranges = _mapping("ranges", raw.get("ranges"))
    _allow_keys(
        ranges,
        path="$.ranges",
        allowed=_SWEEP_RANGE_FIELDS,
        required=_SWEEP_RANGE_FIELDS,
    )
    bounds = (
        _range(ranges, "discharge_voltage_v", lower_bound=0.0),
        _range(ranges, "propellant_mass_flow_kg_per_s", lower_bound=0.0),
        _range(ranges, "ionized_number_fraction", lower_bound=0.0),
        _range(ranges, "xe_double_plus_fraction_of_ions", lower_bound=0.0),
        _range(
            ranges,
            "beam_current_fraction_of_anode_current",
            lower_bound=0.0,
        ),
        _range(
            ranges,
            "axial_momentum_fraction_of_ion_momentum",
            lower_bound=0.0,
        ),
        _range(ranges, "cathode_input_power_w", lower_bound=0.0),
        _range(ranges, "ppu_efficiency_fraction", lower_bound=0.0),
    )
    unit_interval_names = (
        "ionized_number_fraction",
        "xe_double_plus_fraction_of_ions",
        "beam_current_fraction_of_anode_current",
        "axial_momentum_fraction_of_ion_momentum",
        "ppu_efficiency_fraction",
    )
    for name in unit_interval_names:
        minimum, maximum = _range(ranges, name, lower_bound=0.0)
        if maximum > 1.0 or (name != "xe_double_plus_fraction_of_ions" and minimum == 0.0):
            raise PhysicsConfigurationError(
                f"ranges.{name} must lie in (0, 1]"
                if name != "xe_double_plus_fraction_of_ions"
                else f"ranges.{name} must lie in [0, 1]"
            )
    if bounds[0][0] <= 0.0:
        raise PhysicsConfigurationError(
            "ranges.discharge_voltage_v minimum must be greater than zero"
        )
    if bounds[1][0] <= 0.0:
        raise PhysicsConfigurationError(
            "ranges.propellant_mass_flow_kg_per_s minimum must be greater than zero"
        )

    points: list[XenonOperatingPoint] = []
    for coordinates in _coordinates(count, seed):
        (
            voltage,
            mass_flow,
            ionized,
            double_share,
            beam_fraction,
            axial_fraction,
            cathode_power,
            ppu_efficiency,
        ) = (
            _interpolate(bound, coordinate)
            for bound, coordinate in zip(bounds, coordinates, strict=True)
        )
        neutral = 1.0 - ionized
        double_plus = ionized * double_share
        plus = 1.0 - neutral - double_plus
        fractions = ChargeStateFractions(neutral, plus, double_plus)
        beam_current = (
            mass_flow
            * (ELEMENTARY_CHARGE_C / XENON_ATOM_MASS_KG)
            * fractions.charge_weighted_ion_fraction
        )
        anode_power = voltage * beam_current / beam_fraction
        thruster_power = fsum((anode_power, cathode_power))
        ppu_power = thruster_power / ppu_efficiency
        points.append(
            XenonOperatingPoint(
                discharge_voltage_v=voltage,
                propellant_mass_flow=PropellantMassFlow(mass_flow),
                charge_state_fractions=fractions,
                mass_utilization=MassUtilization.from_charge_states(fractions),
                beam_divergence_factors=BeamDivergenceFactors(
                    beam_fraction, axial_fraction
                ),
                power_boundaries=PowerBoundaryInputs(
                    cathode_power, ppu_power
                ),
            )
        )
    return tuple(points), {
        "method": "deterministic-prime-base-radical-inverse",
        "seed": seed,
        "batch_size": count,
        "dimensions": list(ranges.keys()),
    }


_PARITY_FIELDS = (
    "total_xenon_particle_rate_per_s",
    "neutral_particle_rate_per_s",
    "xe_plus_particle_rate_per_s",
    "xe_double_plus_particle_rate_per_s",
    "xe_plus_speed_m_per_s",
    "xe_double_plus_speed_m_per_s",
    "undiverged_ion_thrust_n",
    "axial_thrust_n",
    "specific_impulse_s",
    "power_budget.beam_current_a",
    "power_budget.anode_current_a",
    "power_budget.beam_kinetic_power_w",
    "power_budget.anode_input_power_w",
    "power_budget.cathode_input_power_w",
    "power_budget.thruster_electrical_input_power_w",
    "power_budget.requested_ppu_input_power_w",
    "power_budget.ppu_input_power_w",
    "power_budget.ppu_boundary_adjustment_w",
    "power_budget.ppu_conversion_loss_w",
    "power_budget.anode_to_beam_efficiency",
    "power_budget.thruster_electrical_to_beam_efficiency",
    "power_budget.ppu_input_to_beam_efficiency",
    "diagnostics.particle_rate_residual_particles_per_s",
    "diagnostics.mass_flow_residual_kg_per_s",
    "diagnostics.beam_current_residual_a",
    "diagnostics.beam_power_residual_w",
)


def _field(result: IdealPerformanceResult, path: str) -> float | None:
    value: object = result
    for part in path.split("."):
        value = getattr(value, part)
    if value is None:
        return None
    return float(value)


def _parity(
    actual: Sequence[IdealPerformanceResult],
    reference: Sequence[IdealPerformanceResult],
) -> dict[str, Any]:
    maximum_absolute: dict[str, float] = {}
    maximum_relative: dict[str, float] = {}
    mismatch_count = 0
    for path in _PARITY_FIELDS:
        absolute = 0.0
        relative = 0.0
        for observed, expected in zip(actual, reference, strict=True):
            observed_value = _field(observed, path)
            expected_value = _field(expected, path)
            if observed_value is None or expected_value is None:
                if observed_value != expected_value:
                    mismatch_count += 1
                continue
            difference = abs(observed_value - expected_value)
            scale = max(abs(expected_value), 1.0e-300)
            absolute = max(absolute, difference)
            relative = max(relative, difference / scale)
            if path == "diagnostics.particle_rate_residual_particles_per_s":
                allowed = 4.0e-16 * max(
                    1.0, observed.total_xenon_particle_rate_per_s
                )
            elif path == "diagnostics.mass_flow_residual_kg_per_s":
                allowed = 3.0e-21
            elif path == "diagnostics.beam_current_residual_a":
                allowed = 2.0e-13
            elif path == "diagnostics.beam_power_residual_w":
                allowed = 5.0e-14 * max(
                    1.0, observed.power_budget.beam_kinetic_power_w
                )
            else:
                allowed = 2.0e-14 * max(abs(expected_value), 1.0e-300)
            if difference > allowed:
                mismatch_count += 1
        maximum_absolute[path] = absolute
        maximum_relative[path] = relative
    return {
        "reference_backend": "python-reference",
        "compared_count": len(reference),
        "fields_compared": list(_PARITY_FIELDS),
        "maximum_absolute_error_by_field": maximum_absolute,
        "maximum_relative_error_by_field": maximum_relative,
        "mismatch_count": mismatch_count,
        "within_documented_binary64_tolerance": mismatch_count == 0,
        "tolerance_policy": (
            "2e-14 relative for published values; conservation diagnostics use "
            "the physics Warp regression gates because reduction order may differ."
        ),
    }


def _result_summary(
    points: Sequence[XenonOperatingPoint],
    results: Sequence[IdealPerformanceResult],
) -> dict[str, Any]:
    range_paths = (
        "axial_thrust_n",
        "specific_impulse_s",
        "power_budget.beam_current_a",
        "power_budget.anode_input_power_w",
        "power_budget.beam_kinetic_power_w",
        "power_budget.ppu_input_to_beam_efficiency",
    )
    output_ranges = {
        path: {
            "minimum": min(
                value
                for result in results
                if (value := _field(result, path)) is not None
            ),
            "maximum": max(
                value
                for result in results
                if (value := _field(result, path)) is not None
            ),
        }
        for path in range_paths
    }
    particle_relative = []
    mass_relative = []
    current_relative = []
    power_relative = []
    for point, result in zip(points, results, strict=True):
        diagnostics = result.diagnostics
        particle_relative.append(
            abs(diagnostics.particle_rate_residual_particles_per_s)
            / result.total_xenon_particle_rate_per_s
        )
        mass_relative.append(
            abs(diagnostics.mass_flow_residual_kg_per_s)
            / point.propellant_mass_flow.kg_per_s
        )
        current_relative.append(
            abs(diagnostics.beam_current_residual_a)
            / max(result.power_budget.beam_current_a, 1.0e-300)
        )
        power_relative.append(
            abs(diagnostics.beam_power_residual_w)
            / max(result.power_budget.beam_kinetic_power_w, 1.0e-300)
        )
    return {
        "output_ranges": output_ranges,
        "maximum_absolute_conservation_residuals": {
            "particle_rate_residual_particles_per_s": max(
                abs(result.diagnostics.particle_rate_residual_particles_per_s)
                for result in results
            ),
            "mass_flow_residual_kg_per_s": max(
                abs(result.diagnostics.mass_flow_residual_kg_per_s)
                for result in results
            ),
            "beam_current_residual_a": max(
                abs(result.diagnostics.beam_current_residual_a)
                for result in results
            ),
            "beam_power_residual_w": max(
                abs(result.diagnostics.beam_power_residual_w)
                for result in results
            ),
        },
        "maximum_relative_conservation_residuals": {
            "particle_rate": max(particle_relative),
            "mass_flow": max(mass_relative),
            "beam_current": max(current_relative),
            "beam_power": max(power_relative),
        },
    }


def evaluate_sweep_artifact(
    raw: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    if not isinstance(device, str) or re.fullmatch(
        r"(?:python|cpu|cuda|cuda:(?:0|[1-9][0-9]*))", device
    ) is None:
        raise PhysicsConfigurationError(
            "device must be 'python', 'cpu', 'cuda', or 'cuda:N'"
        )
    points, sampling = sweep_points_from_config(raw)
    reference_started = perf_counter()
    reference = evaluate_batch(points)
    reference_seconds = perf_counter() - reference_started

    evaluation_started = perf_counter()
    if device == "python":
        results = reference
        resolved_device = "cpu"
        implementation = "python-reference"
        evaluation_seconds = reference_seconds
    else:
        from .warp_backend import evaluate_performance_warp

        warp_result = evaluate_performance_warp(points, device=device)
        results = warp_result.results
        resolved_device = warp_result.device
        implementation = "nvidia-warp"
        evaluation_seconds = perf_counter() - evaluation_started

    return {
        "document_type": "cft-revival-l0-sweep-result",
        "schema_version": "1.0",
        "model_fidelity": L0_MODEL_FIDELITY,
        "model_claim": L0_MODEL_CLAIM,
        "hypothetical_inputs": True,
        "backend": {
            "implementation": implementation,
            "device": resolved_device,
        },
        "sampling": sampling,
        "runtime": {
            "evaluation_elapsed_seconds": evaluation_seconds,
            "evaluation_throughput_points_per_second": (
                len(points) / evaluation_seconds
                if evaluation_seconds > 0.0
                else None
            ),
            "cpu_reference_elapsed_seconds": reference_seconds,
            "timing_controlled": False,
            "timing_note": (
                "End-to-end diagnostic timing includes Python preprocessing, "
                "allocation, transfers, synchronization, and result construction; "
                "no CPU/GPU speedup may be inferred."
            ),
        },
        "summary": _result_summary(points, results),
        "cpu_reference_parity": _parity(results, reference),
        "points": [
            {
                "index": index,
                "input": operating_point_to_dict(point),
                "result": result_to_dict(result),
            }
            for index, (point, result) in enumerate(
                zip(points, results, strict=True)
            )
        ],
    }
