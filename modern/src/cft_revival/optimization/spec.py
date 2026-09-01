"""Strict validation and dependency-free initial designs for campaign spec v1.4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import fsum, isclose, isfinite
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


def _record(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignSpecError(f"{path} must be an object")
    return value


def _allow_keys(
    raw: Mapping[str, Any],
    *,
    path: str,
    allowed: set[str],
    required: set[str] | None = None,
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise CampaignSpecError(f"{path} contains unknown field(s): {names}")
    missing = (required or allowed) - set(raw)
    if missing:
        names = ", ".join(sorted(missing))
        raise CampaignSpecError(f"{path} is missing required field(s): {names}")


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


def _text(raw: Mapping[str, Any], name: str, *, path: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise CampaignSpecError(f"{path}.{name} must be non-empty text")
    return value


def _boolean(raw: Mapping[str, Any], name: str, *, path: str) -> bool:
    value = raw.get(name)
    if not isinstance(value, bool):
        raise CampaignSpecError(f"{path}.{name} must be boolean")
    return value


def _string_array(raw: Mapping[str, Any], name: str, *, path: str) -> tuple[str, ...]:
    values = _sequence(raw, name)
    if any(not isinstance(value, str) or not value for value in values):
        raise CampaignSpecError(f"{path}.{name} must contain non-empty strings")
    return tuple(values)


@dataclass(frozen=True)
class ValidatedCampaignSpec:
    campaign_spec_id: str
    schema_version: str
    variables: tuple[Variable, ...]
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ContinuousConstraint, ...]
    sources: tuple[InformationSource, ...]
    campaign_config: CampaignConfig


def _validate_closed_schema(raw: Mapping[str, Any]) -> None:
    """Reject undeclared fields before constructing any domain records."""

    top = {
        "document_type",
        "schema_version",
        "campaign",
        "campaign_spec_id",
        "identity_policy",
        "decision_space",
        "objectives",
        "information_sources",
        "iteration_policy",
        "highest_fidelity_attempt_policy",
        "stopping_gates",
        "benchmark",
    }
    _allow_keys(raw, path="$", allowed=top, required=top)

    identity = _mapping(raw, "identity_policy")
    identity_keys = {
        "design_id_fields",
        "evaluation_key_fields",
        "event_log",
        "decoded_number_policy",
        "decoded_object_policy",
    }
    _allow_keys(
        identity,
        path="$.identity_policy",
        allowed=identity_keys,
        required=identity_keys,
    )
    _string_array(
        identity, "design_id_fields", path="$.identity_policy"
    )
    _string_array(
        identity, "evaluation_key_fields", path="$.identity_policy"
    )
    for name in (
        "event_log",
        "decoded_number_policy",
        "decoded_object_policy",
    ):
        _text(identity, name, path="$.identity_policy")

    decision = _mapping(raw, "decision_space")
    decision_keys = {"dimensions", "variables", "coupled_constraints"}
    _allow_keys(
        decision,
        path="$.decision_space",
        allowed=decision_keys,
        required=decision_keys,
    )
    variable_keys = {"name", "lower", "upper", "units"}
    for index, value in enumerate(_sequence(decision, "variables")):
        item = _record(value, path=f"$.decision_space.variables[{index}]")
        _allow_keys(
            item,
            path=f"$.decision_space.variables[{index}]",
            allowed=variable_keys,
            required=variable_keys,
        )
    constraint_keys = {
        "name",
        "type",
        "sense",
        "threshold",
        "units",
        "violation_scale",
        "applies_pairwise_to_variables_3_through_7",
    }
    for index, value in enumerate(_sequence(decision, "coupled_constraints")):
        item = _record(
            value, path=f"$.decision_space.coupled_constraints[{index}]"
        )
        _allow_keys(
            item,
            path=f"$.decision_space.coupled_constraints[{index}]",
            allowed=constraint_keys,
            required=constraint_keys,
        )

    objective_keys = {
        "name",
        "direction",
        "units",
        "comparison_scale",
        "absolute_tolerance",
        "relative_tolerance",
    }
    for index, value in enumerate(_sequence(raw, "objectives")):
        item = _record(value, path=f"$.objectives[{index}]")
        _allow_keys(
            item,
            path=f"$.objectives[{index}]",
            allowed=objective_keys,
            required=objective_keys,
        )

    source_keys = {
        "fidelity",
        "name",
        "initial_budget",
        "equivalent_f3_cost",
    }
    for index, value in enumerate(_sequence(raw, "information_sources")):
        item = _record(value, path=f"$.information_sources[{index}]")
        _allow_keys(
            item,
            path=f"$.information_sources[{index}]",
            allowed=source_keys,
            required=source_keys,
        )

    iteration = _mapping(raw, "iteration_policy")
    iteration_keys = {
        "cheap_medium_evaluations",
        "high_fidelity_evaluations",
        "asynchronous_pending_aware",
        "botorch_output_transform",
        "model_output_layout",
        "botorch_constraint_convention",
        "source_task_model",
        "acquisition_mix",
        "qLogNParEGO_batch_path",
        "promotion",
    }
    _allow_keys(
        iteration,
        path="$.iteration_policy",
        allowed=iteration_keys,
        required=iteration_keys,
    )
    evaluation_keys = {"minimum", "maximum", "fidelities"}
    for name in ("cheap_medium_evaluations", "high_fidelity_evaluations"):
        item = _mapping(iteration, name)
        _allow_keys(
            item,
            path=f"$.iteration_policy.{name}",
            allowed=evaluation_keys,
            required=evaluation_keys,
        )
    layout = _mapping(iteration, "model_output_layout")
    layout_keys = {
        "ordering",
        "acquisition_objective_selector",
        "constraint_selector",
        "constraints_can_be_objectives",
        "constraint_variance_transform",
        "objective_variance_transform",
    }
    _allow_keys(
        layout,
        path="$.iteration_policy.model_output_layout",
        allowed=layout_keys,
        required=layout_keys,
    )
    source_task = _mapping(iteration, "source_task_model")
    source_task_keys = {
        "outcome_transform",
        "known_differing_task_noise",
        "supported_noise_contract",
    }
    _allow_keys(
        source_task,
        path="$.iteration_policy.source_task_model",
        allowed=source_task_keys,
        required=source_task_keys,
    )
    acquisition_keys = {"strategy", "fraction"}
    for index, value in enumerate(_sequence(iteration, "acquisition_mix")):
        item = _record(
            value, path=f"$.iteration_policy.acquisition_mix[{index}]"
        )
        _allow_keys(
            item,
            path=f"$.iteration_policy.acquisition_mix[{index}]",
            allowed=acquisition_keys,
            required=acquisition_keys,
        )
    promotion = _mapping(iteration, "promotion")
    promotion_keys = {
        "requires_nondominated",
        "requires_robust_feasibility_sigma",
        "minimum_per_constraint_feasibility_probability",
        "probability_policy",
        "mandatory_highest_available_reevaluation",
        "maximum_retries_per_promotion",
        "concurrent_attempts_per_promotion_source",
        "retry_requires_terminal_failure_key",
        "retry_requires_solver_failure_retryable",
        "lineage_identity",
        "pareto_comparable_context_identity",
        "cross_context_pareto_comparison",
        "source_seed_or_observation_id_defines_lineage",
    }
    _allow_keys(
        promotion,
        path="$.iteration_policy.promotion",
        allowed=promotion_keys,
        required=promotion_keys,
    )

    highest = _mapping(raw, "highest_fidelity_attempt_policy")
    highest_keys = {
        "initial_attempt_budget",
        "successful_validation_target",
        "reserved_retry_attempts",
        "total_attempt_limit",
        "failed_attempts_consume_cost",
        "failed_f3_charged_cost_must_be_finite_positive",
        "all_retry_charged_cost_must_be_finite_positive",
        "pre_execution_rejection",
        "retry_slots_are_not_available_to_initial_attempts",
    }
    _allow_keys(
        highest,
        path="$.highest_fidelity_attempt_policy",
        allowed=highest_keys,
        required=highest_keys,
    )

    stopping = _mapping(raw, "stopping_gates")
    stopping_keys = {
        "all_must_hold_unless_cost_ceiling_reached",
        "terminal_failure",
        "hard_equivalent_f3_cost_ceiling",
    }
    _allow_keys(
        stopping,
        path="$.stopping_gates",
        allowed=stopping_keys,
        required=stopping_keys,
    )
    gate_allowed = {
        "gate",
        "value",
        "maximum",
        "window_iterations",
        "numerator",
        "denominator",
        "evidence_source",
    }
    for index, value in enumerate(
        _sequence(stopping, "all_must_hold_unless_cost_ceiling_reached")
    ):
        item = _record(value, path=f"$.stopping_gates.gates[{index}]")
        _allow_keys(
            item,
            path=f"$.stopping_gates.gates[{index}]",
            allowed=gate_allowed,
            required={"gate"},
        )

    benchmark = _mapping(raw, "benchmark")
    benchmark_keys = {
        "results",
        "comparison_metric",
        "strategies",
        "fairness_requirements",
    }
    _allow_keys(
        benchmark,
        path="$.benchmark",
        allowed=benchmark_keys,
        required=benchmark_keys,
    )
    metric = _mapping(benchmark, "comparison_metric")
    metric_keys = {
        "primary",
        "cost_axis",
        "verification_fidelity",
        "report_uncertainty_across_repeated_campaign_seeds",
    }
    _allow_keys(
        metric,
        path="$.benchmark.comparison_metric",
        allowed=metric_keys,
        required=metric_keys,
    )
    strategy_keys = {
        "name",
        "role",
        "implementation_requirement",
        "same_cost_and_F3_verification",
    }
    for index, value in enumerate(_sequence(benchmark, "strategies")):
        item = _record(value, path=f"$.benchmark.strategies[{index}]")
        _allow_keys(
            item,
            path=f"$.benchmark.strategies[{index}]",
            allowed=strategy_keys,
            required={"name", "role"},
        )


def validate_campaign_spec(raw: Mapping[str, Any]) -> ValidatedCampaignSpec:
    """Validate the closed schema and cross-field invariants of spec v1.4."""

    _validate_closed_schema(raw)
    if raw.get("document_type") != "cft-revival-optimization-campaign":
        raise CampaignSpecError("unexpected campaign document_type")
    if raw.get("schema_version") != "1.4":
        raise CampaignSpecError("only campaign schema_version '1.4' is supported")
    campaign_name = _text(raw, "campaign", path="$")
    campaign_spec_id = raw.get("campaign_spec_id")
    if not isinstance(campaign_spec_id, str) or not campaign_spec_id:
        raise CampaignSpecError("campaign_spec_id must be non-empty text")
    if campaign_spec_id != f"{campaign_name}@1.4":
        raise CampaignSpecError(
            "campaign_spec_id must equal '<campaign>@<schema_version>'"
        )

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
    for index, item_value in enumerate(constraint_records):
        item = _record(
            item_value,
            path=f"$.decision_space.coupled_constraints[{index}]",
        )
        if item["type"] != "continuous":
            raise CampaignSpecError(
                f"coupled_constraints[{index}].type must be 'continuous'"
            )
        if not _boolean(
            item,
            "applies_pairwise_to_variables_3_through_7",
            path=f"$.decision_space.coupled_constraints[{index}]",
        ):
            raise CampaignSpecError(
                "successive radial clearance must apply pairwise to variables 3-7"
            )

    expected_objectives = (
        ("thrust_n", "maximize"),
        ("total_efficiency", "maximize"),
        ("specific_impulse_s", "maximize"),
        ("anode_power_w", "minimize"),
    )
    actual_objectives = tuple(
        (objective.name, objective.direction.value) for objective in objectives
    )
    if actual_objectives != expected_objectives:
        raise CampaignSpecError(
            "objectives must be ordered as +thrust, +efficiency, +Isp, -power"
        )

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
            name = _text(item, "name", path="$.information_sources[]")
            if budget <= 0:
                raise CampaignSpecError("initial_budget must be positive")
            if equivalent_cost <= 0.0:
                raise CampaignSpecError("equivalent_f3_cost must be positive")
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

    iteration = _mapping(raw, "iteration_policy")
    if not _boolean(
        iteration,
        "asynchronous_pending_aware",
        path="$.iteration_policy",
    ):
        raise CampaignSpecError("asynchronous_pending_aware must be true")
    transform = iteration["botorch_output_transform"]
    if (
        not isinstance(transform, list)
        or any(isinstance(value, bool) or not isinstance(value, Real) for value in transform)
        or [float(value) for value in transform] != [1.0, 1.0, 1.0, -1.0]
    ):
        raise CampaignSpecError(
            "botorch_output_transform must be [1.0, 1.0, 1.0, -1.0] "
            "for +thrust, +efficiency, +Isp, -power"
        )
    expected_fidelity_sets = {
        "cheap_medium_evaluations": ("F0", "F1", "F2"),
        "high_fidelity_evaluations": ("F3",),
    }
    for name, expected_fidelities in expected_fidelity_sets.items():
        policy = _mapping(iteration, name)
        minimum = _integer(policy, "minimum")
        maximum = _integer(policy, "maximum")
        if minimum < 0 or maximum < minimum:
            raise CampaignSpecError(
                f"iteration_policy.{name} requires 0 <= minimum <= maximum"
            )
        fidelities = _string_array(
            policy,
            "fidelities",
            path=f"$.iteration_policy.{name}",
        )
        if fidelities != expected_fidelities:
            raise CampaignSpecError(
                f"iteration_policy.{name}.fidelities must be "
                f"{list(expected_fidelities)!r}"
            )
    layout = _mapping(iteration, "model_output_layout")
    if _boolean(
        layout,
        "constraints_can_be_objectives",
        path="$.iteration_policy.model_output_layout",
    ):
        raise CampaignSpecError("constraints_can_be_objectives must be false")
    _string_array(
        layout,
        "ordering",
        path="$.iteration_policy.model_output_layout",
    )
    for name in (
        "acquisition_objective_selector",
        "constraint_selector",
        "constraint_variance_transform",
        "objective_variance_transform",
    ):
        _text(layout, name, path="$.iteration_policy.model_output_layout")
    _text(
        iteration,
        "botorch_constraint_convention",
        path="$.iteration_policy",
    )
    _text(iteration, "qLogNParEGO_batch_path", path="$.iteration_policy")
    source_task = _mapping(iteration, "source_task_model")
    for name in (
        "outcome_transform",
        "known_differing_task_noise",
        "supported_noise_contract",
    ):
        _text(
            source_task,
            name,
            path="$.iteration_policy.source_task_model",
        )

    acquisitions = _sequence(iteration, "acquisition_mix")
    fractions: list[float] = []
    acquisition_names: set[str] = set()
    for index, item_value in enumerate(acquisitions):
        item = _record(
            item_value,
            path=f"$.iteration_policy.acquisition_mix[{index}]",
        )
        name = _text(
            item,
            "strategy",
            path=f"$.iteration_policy.acquisition_mix[{index}]",
        )
        fraction = _real(item, "fraction")
        if not 0.0 <= fraction <= 1.0:
            raise CampaignSpecError(
                f"acquisition_mix[{index}].fraction must lie in [0, 1]"
            )
        if name in acquisition_names:
            raise CampaignSpecError("acquisition strategy names must be unique")
        acquisition_names.add(name)
        fractions.append(fraction)
    if not acquisitions or not isclose(
        fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise CampaignSpecError("acquisition fractions must sum to one")

    promotion = _mapping(iteration, "promotion")
    for name in (
        "requires_nondominated",
        "mandatory_highest_available_reevaluation",
        "retry_requires_terminal_failure_key",
        "retry_requires_solver_failure_retryable",
    ):
        if not _boolean(promotion, name, path="$.iteration_policy.promotion"):
            raise CampaignSpecError(f"promotion.{name} must be true")
    for name in (
        "cross_context_pareto_comparison",
        "source_seed_or_observation_id_defines_lineage",
    ):
        if _boolean(promotion, name, path="$.iteration_policy.promotion"):
            raise CampaignSpecError(f"promotion.{name} must be false")
    robust_sigma = _real(promotion, "requires_robust_feasibility_sigma")
    probability = _real(
        promotion, "minimum_per_constraint_feasibility_probability"
    )
    retry_limit = _integer(promotion, "maximum_retries_per_promotion")
    concurrency = _integer(
        promotion, "concurrent_attempts_per_promotion_source"
    )
    if robust_sigma < 0.0:
        raise CampaignSpecError("promotion robust-feasibility sigma cannot be negative")
    if not 0.0 <= probability <= 1.0:
        raise CampaignSpecError(
            "promotion feasibility probability must lie in [0, 1]"
        )
    if retry_limit < 0 or concurrency != 1:
        raise CampaignSpecError(
            "promotion retries must be non-negative and concurrency must equal one"
        )
    _string_array(promotion, "lineage_identity", path="$.iteration_policy.promotion")
    _string_array(
        promotion,
        "pareto_comparable_context_identity",
        path="$.iteration_policy.promotion",
    )

    highest = _mapping(raw, "highest_fidelity_attempt_policy")
    for name in (
        "initial_attempt_budget",
        "successful_validation_target",
        "reserved_retry_attempts",
        "total_attempt_limit",
    ):
        value = _integer(highest, name)
        if value < 0:
            raise CampaignSpecError(
                f"highest_fidelity_attempt_policy.{name} cannot be negative"
            )
    _text(
        highest,
        "pre_execution_rejection",
        path="$.highest_fidelity_attempt_policy",
    )
    stopping = _mapping(raw, "stopping_gates")
    gates = _sequence(stopping, "all_must_hold_unless_cost_ceiling_reached")
    gate_records = [
        _record(item, path=f"$.stopping_gates.gates[{index}]")
        for index, item in enumerate(gates)
    ]
    gate_names = [
        _text(item, "gate", path="$.stopping_gates.gates[]")
        for item in gate_records
    ]
    expected_gate_names = {
        "mandatory_f3_success_count",
        "minimum_f3_success_fraction",
        "verified_hypervolume_relative_improvement",
        "pending_jobs",
        "surrogate_calibration_checked",
        "promoted_candidates_pass_guardrails",
        "acquisition_converged",
        "iteration_acquisition_policy_satisfied",
    }
    if set(gate_names) != expected_gate_names or len(gate_names) != len(
        expected_gate_names
    ):
        raise CampaignSpecError("stopping gates must contain each v1.4 gate exactly once")
    gate_by_name = dict(zip(gate_names, gate_records, strict=True))
    gate_fields = {
        "mandatory_f3_success_count": {"gate", "value"},
        "minimum_f3_success_fraction": {
            "gate",
            "value",
            "numerator",
            "denominator",
        },
        "verified_hypervolume_relative_improvement": {
            "gate",
            "maximum",
            "window_iterations",
        },
        "pending_jobs": {"gate", "value"},
        "surrogate_calibration_checked": {"gate", "value"},
        "promoted_candidates_pass_guardrails": {"gate", "value"},
        "acquisition_converged": {"gate", "value"},
        "iteration_acquisition_policy_satisfied": {
            "gate",
            "value",
            "evidence_source",
        },
    }
    for name, fields in gate_fields.items():
        _allow_keys(
            gate_by_name[name],
            path=f"$.stopping_gates.{name}",
            allowed=fields,
            required=fields,
        )
    for name in ("numerator", "denominator"):
        _text(
            gate_by_name["minimum_f3_success_fraction"],
            name,
            path="$.stopping_gates.minimum_f3_success_fraction",
        )
    _text(
        gate_by_name["iteration_acquisition_policy_satisfied"],
        "evidence_source",
        path="$.stopping_gates.iteration_acquisition_policy_satisfied",
    )
    mandatory_count = _integer(
        gate_by_name["mandatory_f3_success_count"], "value"
    )
    minimum_fraction = _real(
        gate_by_name["minimum_f3_success_fraction"], "value"
    )
    pending_jobs = _integer(gate_by_name["pending_jobs"], "value")
    hypervolume_gate = gate_by_name[
        "verified_hypervolume_relative_improvement"
    ]
    hypervolume_maximum = _real(hypervolume_gate, "maximum")
    hypervolume_window = _integer(hypervolume_gate, "window_iterations")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise CampaignSpecError("minimum_f3_success_fraction must lie in [0, 1]")
    if pending_jobs != 0:
        raise CampaignSpecError("pending_jobs stopping value must equal zero")
    if not 0.0 <= hypervolume_maximum <= 1.0 or hypervolume_window < 1:
        raise CampaignSpecError(
            "verified hypervolume gate requires maximum in [0, 1] and positive window"
        )
    for name in (
        "surrogate_calibration_checked",
        "promoted_candidates_pass_guardrails",
        "acquisition_converged",
        "iteration_acquisition_policy_satisfied",
    ):
        if not _boolean(gate_by_name[name], "value", path="$.stopping_gates"):
            raise CampaignSpecError(f"stopping gate {name} must require true")

    try:
        config = CampaignConfig(
            campaign_spec_id=campaign_spec_id,
            fidelity_budgets=tuple(budgets),
            mandatory_high_fidelity_validations=_integer(
                highest, "successful_validation_target"
            ),
            minimum_high_fidelity_fraction=minimum_fraction,
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
    if mandatory_count != _integer(highest, "successful_validation_target"):
        raise CampaignSpecError(
            "mandatory F3 stopping count must match successful_validation_target"
        )
    for name in (
        "failed_attempts_consume_cost",
        "failed_f3_charged_cost_must_be_finite_positive",
        "all_retry_charged_cost_must_be_finite_positive",
        "retry_slots_are_not_available_to_initial_attempts",
    ):
        if not _boolean(highest, name, path="$.highest_fidelity_attempt_policy"):
            raise CampaignSpecError(
                f"highest_fidelity_attempt_policy.{name} must be true"
            )
    if retry_limit == 0 and _integer(highest, "reserved_retry_attempts") > 0:
        raise CampaignSpecError(
            "reserved retry attempts require maximum_retries_per_promotion > 0"
        )

    benchmark = _mapping(raw, "benchmark")
    if benchmark["results"] is not None:
        raise CampaignSpecError(
            "benchmark.results must remain null until verified benchmark evidence exists"
        )
    metric = _mapping(benchmark, "comparison_metric")
    for name in ("primary", "cost_axis", "verification_fidelity"):
        _text(metric, name, path="$.benchmark.comparison_metric")
    if not _boolean(
        metric,
        "report_uncertainty_across_repeated_campaign_seeds",
        path="$.benchmark.comparison_metric",
    ):
        raise CampaignSpecError(
            "benchmark comparison must report uncertainty across repeated seeds"
        )
    _string_array(
        benchmark,
        "fairness_requirements",
        path="$.benchmark",
    )
    strategies = _sequence(benchmark, "strategies")
    strategy_names: set[str] = set()
    for index, item_value in enumerate(strategies):
        item = _record(item_value, path=f"$.benchmark.strategies[{index}]")
        name = _text(item, "name", path=f"$.benchmark.strategies[{index}]")
        _text(item, "role", path=f"$.benchmark.strategies[{index}]")
        if name in strategy_names:
            raise CampaignSpecError("benchmark strategy names must be unique")
        strategy_names.add(name)
        if "implementation_requirement" in item:
            _text(
                item,
                "implementation_requirement",
                path=f"$.benchmark.strategies[{index}]",
            )
        if "same_cost_and_F3_verification" in item:
            if not _boolean(
                item,
                "same_cost_and_F3_verification",
                path=f"$.benchmark.strategies[{index}]",
            ):
                raise CampaignSpecError(
                    "benchmark same-cost F3 verification flags must be true"
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
