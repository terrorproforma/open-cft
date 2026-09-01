"""Strict validation and dependency-free initial designs for campaign spec v1.4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from .campaign import CampaignConfig
from .domain import (
    ContinuousConstraint,
    Fidelity,
    InformationSource,
    ObjectiveSpec,
    Variable,
)
from .sampling import initial_designs


class CampaignSpecError(ValueError):
    """A campaign specification is malformed or internally inconsistent."""


def _reject_constant(value: str) -> None:
    raise CampaignSpecError(f"campaign specification contains {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignSpecError(
                f"campaign specification contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _validate_finite(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise CampaignSpecError(f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
        return
    raise CampaignSpecError(f"unsupported JSON value at {path}")


def load_json_strict(path: Path) -> Mapping[str, Any]:
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignSpecError(f"cannot read campaign spec {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise CampaignSpecError("campaign specification root must be an object")
    _validate_finite(decoded)
    return decoded


def _mapping(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise CampaignSpecError(f"{name} must be an object")
    return value


def _sequence(raw: Mapping[str, Any], name: str) -> list[Any]:
    value = raw.get(name)
    if not isinstance(value, list):
        raise CampaignSpecError(f"{name} must be an array")
    return value


def _integer(raw: Mapping[str, Any], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignSpecError(f"{name} must be an integer")
    return value


def _real(raw: Mapping[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CampaignSpecError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise CampaignSpecError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ValidatedCampaignSpec:
    campaign_spec_id: str
    schema_version: str
    variables: tuple[Variable, ...]
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ContinuousConstraint, ...]
    sources: tuple[InformationSource, ...]
    campaign_config: CampaignConfig


def validate_campaign_spec(raw: Mapping[str, Any]) -> ValidatedCampaignSpec:
    """Validate the executable subset and cross-field invariants of spec v1.4."""

    if raw.get("document_type") != "cft-revival-optimization-campaign":
        raise CampaignSpecError("unexpected campaign document_type")
    if raw.get("schema_version") != "1.4":
        raise CampaignSpecError("only campaign schema_version '1.4' is supported")
    campaign_spec_id = raw.get("campaign_spec_id")
    if not isinstance(campaign_spec_id, str) or not campaign_spec_id:
        raise CampaignSpecError("campaign_spec_id must be non-empty text")

    decision_space = _mapping(raw, "decision_space")
    variable_records = _sequence(decision_space, "variables")
    try:
        variables = tuple(Variable(**item) for item in variable_records)
    except (TypeError, ValueError) as error:
        raise CampaignSpecError(f"invalid decision variable: {error}") from error
    if _integer(decision_space, "dimensions") != len(variables):
        raise CampaignSpecError("decision_space.dimensions does not match variables")

    try:
        objectives = tuple(ObjectiveSpec(**item) for item in _sequence(raw, "objectives"))
    except (TypeError, ValueError) as error:
        raise CampaignSpecError(f"invalid objective: {error}") from error
    if len({objective.name for objective in objectives}) != len(objectives):
        raise CampaignSpecError("objective names must be unique")

    constraint_records = _sequence(decision_space, "coupled_constraints")
    try:
        constraints = tuple(
            ContinuousConstraint(
                name=item["name"],
                sense=item["sense"],
                threshold=item["threshold"],
                units=item["units"],
                violation_scale=item["violation_scale"],
            )
            for item in constraint_records
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignSpecError(f"invalid continuous constraint: {error}") from error

    source_records = _sequence(raw, "information_sources")
    sources: list[InformationSource] = []
    budgets: list[tuple[Fidelity, int]] = []
    for item in source_records:
        if not isinstance(item, Mapping):
            raise CampaignSpecError("information source entries must be objects")
        try:
            fidelity = Fidelity(item["fidelity"])
            budget = _integer(item, "initial_budget")
            equivalent_cost = _real(item, "equivalent_f3_cost")
            name = str(item["name"])
            sources.append(
                InformationSource(
                    fidelity=fidelity,
                    name=name,
                    description=f"Campaign source {name}",
                    equivalent_cost=equivalent_cost,
                    default_uncertainty=0.0,
                )
            )
            budgets.append((fidelity, budget))
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignSpecError(f"invalid information source: {error}") from error
    if {source.fidelity for source in sources} != set(Fidelity):
        raise CampaignSpecError("information_sources must define F0, F1, F2, and F3 once")

    highest = _mapping(raw, "highest_fidelity_attempt_policy")
    stopping = _mapping(raw, "stopping_gates")
    gates = _sequence(stopping, "all_must_hold_unless_cost_ceiling_reached")
    gate_values = {
        item.get("gate"): item.get("value")
        for item in gates
        if isinstance(item, Mapping)
    }
    try:
        config = CampaignConfig(
            campaign_spec_id=campaign_spec_id,
            fidelity_budgets=tuple(budgets),
            mandatory_high_fidelity_validations=_integer(
                highest, "successful_validation_target"
            ),
            minimum_high_fidelity_fraction=float(
                gate_values.get("minimum_f3_success_fraction", -1.0)
            ),
            highest_fidelity_attempt_limit=_integer(highest, "total_attempt_limit"),
            reserved_high_fidelity_retries=_integer(
                highest, "reserved_retry_attempts"
            ),
            maximum_retries_per_promotion=_integer(
                _mapping(_mapping(raw, "iteration_policy"), "promotion"),
                "maximum_retries_per_promotion",
            ),
            maximum_equivalent_f3_cost=_real(
                stopping, "hard_equivalent_f3_cost_ceiling"
            ),
        )
    except (TypeError, ValueError) as error:
        raise CampaignSpecError(f"invalid campaign policy: {error}") from error
    if _integer(highest, "initial_attempt_budget") != dict(budgets)[Fidelity.F3]:
        raise CampaignSpecError(
            "highest-fidelity initial_attempt_budget must match the F3 source budget"
        )
    return ValidatedCampaignSpec(
        campaign_spec_id=campaign_spec_id,
        schema_version="1.4",
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        sources=tuple(sources),
        campaign_config=config,
    )


def campaign_spec_artifact(
    raw: Mapping[str, Any],
    *,
    initial_design_count: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    if initial_design_count < 0:
        raise CampaignSpecError("initial_design_count must be non-negative")
    if seed < 0:
        raise CampaignSpecError("seed must be non-negative")
    spec = validate_campaign_spec(raw)
    designs = initial_designs(
        spec.variables,
        initial_design_count,
        seed=seed,
        include_boundary_challenges=True,
    )
    return {
        "document_type": "cft-revival-optimization-initial-design",
        "schema_version": "1.0",
        "campaign_spec_id": spec.campaign_spec_id,
        "validated_campaign_schema_version": spec.schema_version,
        "generator": {
            "method": "boundary-challenges-then-shifted-halton",
            "seed": seed,
            "count": initial_design_count,
            "requires_botorch": False,
        },
        "summary": {
            "dimensions": len(spec.variables),
            "objectives": len(spec.objectives),
            "constraints": len(spec.constraints),
            "information_sources": [source.fidelity.value for source in spec.sources],
            "fidelity_budgets": {
                fidelity.value: budget
                for fidelity, budget in spec.campaign_config.fidelity_budgets
            },
        },
        "designs": [
            {
                "design_id": design.design_id,
                "provenance": design.provenance,
                "values": {
                    variable.name: {
                        "value": value,
                        "units": variable.units,
                    }
                    for variable, value in zip(
                        design.variables, design.values, strict=True
                    )
                },
            }
            for design in designs
        ],
    }


def campaign_validation_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a concise validation receipt without generating any designs."""

    spec = validate_campaign_spec(raw)
    return {
        "document_type": "cft-revival-optimization-campaign-validation",
        "schema_version": "1.0",
        "campaign_spec_id": spec.campaign_spec_id,
        "validated_campaign_schema_version": spec.schema_version,
        "valid": True,
        "requires_botorch": False,
        "summary": {
            "dimensions": len(spec.variables),
            "objectives": len(spec.objectives),
            "constraints": len(spec.constraints),
            "information_sources": [source.fidelity.value for source in spec.sources],
            "fidelity_budgets": {
                fidelity.value: budget
                for fidelity, budget in spec.campaign_config.fidelity_budgets
            },
            "highest_fidelity_attempt_limit": (
                spec.campaign_config.highest_fidelity_attempt_limit
            ),
            "reserved_high_fidelity_retries": (
                spec.campaign_config.reserved_high_fidelity_retries
            ),
            "maximum_equivalent_f3_cost": (
                spec.campaign_config.maximum_equivalent_f3_cost
            ),
        },
    }
