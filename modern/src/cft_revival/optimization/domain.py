"""Typed, immutable records for multi-fidelity optimization campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from typing import Any, Mapping, Sequence


class DomainError(ValueError):
    """Raised when an optimization record violates a domain invariant."""


class ObjectiveDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintSense(str, Enum):
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN_OR_EQUAL = ">="


class Fidelity(str, Enum):
    F0 = "F0"
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"

    @property
    def ordinal(self) -> int:
        return ("F0", "F1", "F2", "F3").index(self.value)


class EvaluationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


def _canonical_float(value: float) -> str:
    if not isfinite(value):
        raise DomainError("canonical identifiers require finite values")
    if value == 0.0:
        value = 0.0
    return format(value, ".17g")


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DomainError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise DomainError(f"{name} must be finite")
    return result


def _string_pairs(name: str, values: object) -> tuple[tuple[str, str], ...]:
    try:
        source = (
            values.items()
            if isinstance(values, Mapping)
            else values
        )
        result = tuple((str(key), str(value)) for key, value in source)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise DomainError(f"{name} must contain key/value pairs") from exc
    if any(not key or not value for key, value in result):
        raise DomainError(f"{name} cannot contain empty keys or values")
    if len({key for key, _ in result}) != len(result):
        raise DomainError(f"{name} keys must be unique")
    return tuple(sorted(result))


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Variable:
    name: str
    lower: float
    upper: float
    units: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "lower", _finite_float("variable lower bound", self.lower))
        object.__setattr__(self, "upper", _finite_float("variable upper bound", self.upper))
        if not self.name or not self.units:
            raise DomainError("variables require names and units")
        if self.lower >= self.upper:
            raise DomainError(f"invalid bounds for {self.name}")


@dataclass(frozen=True)
class Design:
    """A bounded design whose ID is stable across processes and platforms."""

    values: tuple[float, ...]
    variables: tuple[Variable, ...]
    provenance: str = ""

    def __post_init__(self) -> None:
        try:
            values = tuple(
                _finite_float(f"design value {index}", value)
                for index, value in enumerate(self.values)
            )
            variables = tuple(self.variables)
        except TypeError as exc:
            raise DomainError("design values and variables must be iterable") from exc
        if any(not isinstance(variable, Variable) for variable in variables):
            raise DomainError("design variables must be Variable records")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "provenance", str(self.provenance))
        if not self.values or len(self.values) != len(self.variables):
            raise DomainError("design values and variables must have equal non-zero length")
        if len({variable.name for variable in self.variables}) != len(self.variables):
            raise DomainError("design variable names must be unique")
        for value, variable in zip(self.values, self.variables, strict=True):
            if not isfinite(value) or not variable.lower <= value <= variable.upper:
                raise DomainError(f"{variable.name}={value!r} outside its finite bounds")

    @property
    def design_id(self) -> str:
        return stable_hash(
            {
                "variables": [
                    {
                        "name": variable.name,
                        "lower": _canonical_float(variable.lower),
                        "upper": _canonical_float(variable.upper),
                        "units": variable.units,
                    }
                    for variable in self.variables
                ],
                "values": [_canonical_float(value) for value in self.values],
            }
        )

    def normalized(self) -> tuple[float, ...]:
        return tuple(
            (value - variable.lower) / (variable.upper - variable.lower)
            for value, variable in zip(self.values, self.variables, strict=True)
        )


@dataclass(frozen=True)
class InformationSource:
    fidelity: Fidelity
    name: str
    description: str
    equivalent_cost: float
    default_uncertainty: float

    def __post_init__(self) -> None:
        try:
            fidelity = Fidelity(self.fidelity)
        except ValueError as exc:
            raise DomainError(f"unknown fidelity {self.fidelity!r}") from exc
        object.__setattr__(self, "fidelity", fidelity)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(
            self,
            "equivalent_cost",
            _finite_float("information-source cost", self.equivalent_cost),
        )
        object.__setattr__(
            self,
            "default_uncertainty",
            _finite_float("information-source uncertainty", self.default_uncertainty),
        )
        if not self.name or not self.description:
            raise DomainError("information sources require a name and description")
        if self.equivalent_cost <= 0.0 or not isfinite(self.equivalent_cost):
            raise DomainError("information-source cost must be finite and positive")
        if self.default_uncertainty < 0.0 or not isfinite(self.default_uncertainty):
            raise DomainError("information-source uncertainty must be finite and non-negative")


DEFAULT_SOURCES: tuple[InformationSource, ...] = (
    InformationSource(
        Fidelity.F0,
        "corrected-analytical",
        "Corrected analytical model",
        1 / 256,
        0.20,
    ),
    InformationSource(
        Fidelity.F1,
        "fields-reduced",
        "Field solve plus reduced plasma model",
        1 / 96,
        0.10,
    ),
    InformationSource(Fidelity.F2, "hybrid", "Hybrid high-resolution model", 1 / 32, 0.05),
    InformationSource(Fidelity.F3, "pic-experiment", "PIC simulation or experiment", 1.0, 0.02),
)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: ObjectiveDirection
    units: str
    comparison_scale: float = 1.0
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        try:
            direction = ObjectiveDirection(self.direction)
        except ValueError as exc:
            raise DomainError(f"unknown objective direction {self.direction!r}") from exc
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "units", str(self.units))
        for field_name in (
            "comparison_scale",
            "absolute_tolerance",
            "relative_tolerance",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(field_name, getattr(self, field_name)),
            )
        if not self.name or not self.units:
            raise DomainError("objectives require names and units")
        if (
            self.comparison_scale <= 0.0
            or self.absolute_tolerance < 0.0
            or self.relative_tolerance < 0.0
        ):
            raise DomainError("objective scale must be positive and tolerances non-negative")


Objective = ObjectiveSpec


@dataclass(frozen=True)
class ObjectiveValue:
    name: str
    value: float
    units: str
    standard_error: float = 0.0
    replicates: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "value", _finite_float("objective value", self.value))
        object.__setattr__(
            self,
            "standard_error",
            _finite_float("objective standard error", self.standard_error),
        )
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, int):
            raise DomainError("objective replicate count must be an integer")
        if not self.name or not self.units:
            raise DomainError("objective values require names and units")
        if not isfinite(self.value) or not isfinite(self.standard_error):
            raise DomainError("objective value and uncertainty must be finite")
        if self.standard_error < 0.0 or self.replicates < 1:
            raise DomainError("objective uncertainty must be non-negative and replicates positive")


@dataclass(frozen=True)
class ContinuousConstraint:
    """A numeric constraint; feasibility is never represented by a category."""

    name: str
    sense: ConstraintSense
    threshold: float
    units: str
    violation_scale: float = 1.0

    def __post_init__(self) -> None:
        try:
            sense = ConstraintSense(self.sense)
        except ValueError as exc:
            raise DomainError(f"unknown constraint sense {self.sense!r}") from exc
        object.__setattr__(self, "sense", sense)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(
            self, "threshold", _finite_float("constraint threshold", self.threshold)
        )
        object.__setattr__(
            self,
            "violation_scale",
            _finite_float("constraint violation scale", self.violation_scale),
        )
        if not self.name or not self.units:
            raise DomainError("constraints require names, units, and finite thresholds")
        if self.violation_scale <= 0.0:
            raise DomainError("constraint violation scale must be positive")

    def residual(self, value: float) -> float:
        """Return a residual where <= 0 is feasible for either sense."""
        if not isfinite(value):
            raise DomainError("constraint values must be finite")
        if self.sense is ConstraintSense.LESS_THAN_OR_EQUAL:
            return value - self.threshold
        return self.threshold - value

    def normalized_residual(self, value: float) -> float:
        return self.residual(value) / self.violation_scale


@dataclass(frozen=True)
class ConstraintValue:
    name: str
    value: float
    units: str
    standard_error: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "value", _finite_float("constraint value", self.value))
        object.__setattr__(
            self,
            "standard_error",
            _finite_float("constraint standard error", self.standard_error),
        )
        if not self.name or not self.units:
            raise DomainError("constraint values require names and units")
        if not isfinite(self.value) or not isfinite(self.standard_error):
            raise DomainError("constraint value and uncertainty must be finite")
        if self.standard_error < 0.0:
            raise DomainError("constraint uncertainty cannot be negative")


@dataclass(frozen=True)
class SolverFailure:
    """Explicit failure metadata, kept separate from objective/constraint values."""

    code: str
    message: str
    retryable: bool
    stage: str
    diagnostics: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "stage", str(self.stage))
        if not isinstance(self.retryable, bool):
            raise DomainError("solver failure retryable flag must be boolean")
        object.__setattr__(
            self, "diagnostics", _string_pairs("failure diagnostics", self.diagnostics)
        )
        if not self.code or not self.message or not self.stage:
            raise DomainError("solver failures require code, message, and stage")


@dataclass(frozen=True)
class Provenance:
    model_version: str
    code_revision: str
    input_hash: str
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_version", str(self.model_version))
        object.__setattr__(self, "code_revision", str(self.code_revision))
        object.__setattr__(self, "input_hash", str(self.input_hash))
        object.__setattr__(
            self, "environment", _string_pairs("provenance environment", self.environment)
        )
        if not self.model_version or not self.code_revision or not self.input_hash:
            raise DomainError("provenance requires model version, revision, and input hash")

    @property
    def provenance_id(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class EvaluationRequest:
    design: Design
    fidelity: Fidelity
    seed: int
    requested_replicates: int = 1
    objective_specs: tuple[ObjectiveSpec, ...] = ()
    constraint_specs: tuple[ContinuousConstraint, ...] = ()
    result_context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.design, Design):
            raise DomainError("evaluation request design must be a Design")
        try:
            fidelity = Fidelity(self.fidelity)
        except ValueError as exc:
            raise DomainError(f"unknown fidelity {self.fidelity!r}") from exc
        object.__setattr__(self, "fidelity", fidelity)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DomainError("seed must be an integer")
        if (
            isinstance(self.requested_replicates, bool)
            or not isinstance(self.requested_replicates, int)
        ):
            raise DomainError("requested replicate count must be an integer")
        try:
            objective_specs = tuple(self.objective_specs)
            constraint_specs = tuple(self.constraint_specs)
        except TypeError as exc:
            raise DomainError("request schemas must be iterable") from exc
        if not objective_specs or any(
            not isinstance(item, ObjectiveSpec) for item in objective_specs
        ):
            raise DomainError("evaluation requests require typed objective specifications")
        if any(
            not isinstance(item, ContinuousConstraint) for item in constraint_specs
        ):
            raise DomainError("constraint specifications must be typed")
        if len({item.name for item in objective_specs}) != len(objective_specs):
            raise DomainError("objective specification names must be unique")
        if len({item.name for item in constraint_specs}) != len(constraint_specs):
            raise DomainError("constraint specification names must be unique")
        object.__setattr__(self, "objective_specs", objective_specs)
        object.__setattr__(self, "constraint_specs", constraint_specs)
        object.__setattr__(
            self, "result_context", _string_pairs("result context", self.result_context)
        )
        if self.seed < 0 or self.requested_replicates < 1:
            raise DomainError("seed must be non-negative and replicate count positive")

    @property
    def evaluation_key(self) -> str:
        return stable_hash(
            {
                "design_id": self.design.design_id,
                "fidelity": self.fidelity.value,
                "seed": self.seed,
                "requested_replicates": self.requested_replicates,
                "objective_specs": [
                    {
                        **asdict(spec),
                        "direction": spec.direction.value,
                    }
                    for spec in self.objective_specs
                ],
                "constraint_specs": [
                    {
                        **asdict(spec),
                        "sense": spec.sense.value,
                    }
                    for spec in self.constraint_specs
                ],
                "result_context": self.result_context,
            }
        )


@dataclass(frozen=True)
class Observation:
    request: EvaluationRequest
    status: EvaluationStatus
    objectives: tuple[ObjectiveValue, ...]
    constraints: tuple[ConstraintValue, ...]
    provenance: Provenance
    wall_time_seconds: float
    charged_cost: float
    failure: SolverFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, EvaluationRequest):
            raise DomainError("observation request must be an EvaluationRequest")
        try:
            status = EvaluationStatus(self.status)
        except ValueError as exc:
            raise DomainError(f"unknown evaluation status {self.status!r}") from exc
        object.__setattr__(self, "status", status)
        try:
            objectives = tuple(self.objectives)
            constraints = tuple(self.constraints)
        except TypeError as exc:
            raise DomainError("observation outcomes must be iterable") from exc
        if any(not isinstance(item, ObjectiveValue) for item in objectives):
            raise DomainError("observation objectives must be ObjectiveValue records")
        if any(not isinstance(item, ConstraintValue) for item in constraints):
            raise DomainError("observation constraints must be ConstraintValue records")
        if not isinstance(self.provenance, Provenance):
            raise DomainError("observation provenance must be a Provenance record")
        if self.failure is not None and not isinstance(self.failure, SolverFailure):
            raise DomainError("observation failure must be a SolverFailure record")
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(
            self,
            "wall_time_seconds",
            _finite_float("wall time", self.wall_time_seconds),
        )
        object.__setattr__(
            self, "charged_cost", _finite_float("charged cost", self.charged_cost)
        )
        if (
            self.wall_time_seconds < 0.0
            or self.charged_cost < 0.0
        ):
            raise DomainError("time and charged cost must be finite and non-negative")
        if self.status is EvaluationStatus.SUCCESS:
            if self.failure is not None or not self.objectives:
                raise DomainError("successful observations require objectives and no failure")
        elif self.failure is None or self.objectives or self.constraints:
            raise DomainError("failed observations require failure metadata and no fake outcomes")
        if len({item.name for item in self.objectives}) != len(self.objectives):
            raise DomainError("objective names must be unique")
        if len({item.name for item in self.constraints}) != len(self.constraints):
            raise DomainError("constraint names must be unique")
        if self.status is EvaluationStatus.SUCCESS:
            objective_by_name = {item.name: item for item in self.objectives}
            constraint_by_name = {item.name: item for item in self.constraints}
            if set(objective_by_name) != {
                item.name for item in self.request.objective_specs
            }:
                raise DomainError("observation objective schema is partial or unexpected")
            if set(constraint_by_name) != {
                item.name for item in self.request.constraint_specs
            }:
                raise DomainError("observation constraint schema is partial or unexpected")
            for spec in self.request.objective_specs:
                measured = objective_by_name[spec.name]
                if measured.units != spec.units:
                    raise DomainError(f"objective units do not match schema: {spec.name}")
                if measured.replicates != self.request.requested_replicates:
                    raise DomainError(
                        f"reported replicates do not match request: {spec.name}"
                    )
            for spec in self.request.constraint_specs:
                if constraint_by_name[spec.name].units != spec.units:
                    raise DomainError(f"constraint units do not match schema: {spec.name}")

    @property
    def observation_id(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": request_to_dict(self.request),
            "status": self.status.value,
            "objectives": [asdict(item) for item in self.objectives],
            "constraints": [asdict(item) for item in self.constraints],
            "provenance": asdict(self.provenance),
            "wall_time_seconds": self.wall_time_seconds,
            "charged_cost": self.charged_cost,
            "failure": asdict(self.failure) if self.failure else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Observation":
        failure_raw = raw.get("failure")
        provenance_raw = dict(raw["provenance"])
        provenance_raw["environment"] = tuple(
            tuple(item) for item in provenance_raw.get("environment", ())
        )
        failure = None
        if failure_raw:
            failure_values = dict(failure_raw)
            failure_values["diagnostics"] = tuple(
                tuple(item) for item in failure_values.get("diagnostics", ())
            )
            failure = SolverFailure(**failure_values)
        return cls(
            request=request_from_dict(raw["request"]),
            status=str(raw["status"]),  # type: ignore[arg-type]
            objectives=tuple(ObjectiveValue(**item) for item in raw["objectives"]),
            constraints=tuple(ConstraintValue(**item) for item in raw["constraints"]),
            provenance=Provenance(**provenance_raw),
            wall_time_seconds=float(raw["wall_time_seconds"]),
            charged_cost=float(raw["charged_cost"]),
            failure=failure,
        )


def request_to_dict(request: EvaluationRequest) -> dict[str, Any]:
    return {
        "design": {
            "values": list(request.design.values),
            "variables": [asdict(variable) for variable in request.design.variables],
            "provenance": request.design.provenance,
        },
        "fidelity": request.fidelity.value,
        "seed": request.seed,
        "requested_replicates": request.requested_replicates,
        "objective_specs": [
            {**asdict(spec), "direction": spec.direction.value}
            for spec in request.objective_specs
        ],
        "constraint_specs": [
            {**asdict(spec), "sense": spec.sense.value}
            for spec in request.constraint_specs
        ],
        "result_context": [list(item) for item in request.result_context],
    }


def request_from_dict(raw: Mapping[str, Any]) -> EvaluationRequest:
    design_raw = raw["design"]
    return EvaluationRequest(
        design=Design(
            tuple(float(value) for value in design_raw["values"]),
            tuple(Variable(**item) for item in design_raw["variables"]),
            str(design_raw.get("provenance", "")),
        ),
        fidelity=str(raw["fidelity"]),  # type: ignore[arg-type]
        seed=int(raw["seed"]),
        requested_replicates=int(raw.get("requested_replicates", 1)),
        objective_specs=tuple(
            ObjectiveSpec(
                name=str(item["name"]),
                direction=str(item["direction"]),  # type: ignore[arg-type]
                units=str(item["units"]),
                comparison_scale=float(item.get("comparison_scale", 1.0)),
                absolute_tolerance=float(item.get("absolute_tolerance", 1e-12)),
                relative_tolerance=float(item.get("relative_tolerance", 1e-12)),
            )
            for item in raw["objective_specs"]
        ),
        constraint_specs=tuple(
            ContinuousConstraint(
                name=str(item["name"]),
                sense=str(item["sense"]),  # type: ignore[arg-type]
                threshold=float(item["threshold"]),
                units=str(item["units"]),
                violation_scale=float(item.get("violation_scale", 1.0)),
            )
            for item in raw.get("constraint_specs", ())
        ),
        result_context=tuple(
            tuple(str(value) for value in item)
            for item in raw.get("result_context", ())
        ),
    )


def source_for(
    fidelity: Fidelity,
    sources: Sequence[InformationSource] = DEFAULT_SOURCES,
) -> InformationSource:
    for source in sources:
        if source.fidelity is fidelity:
            return source
    raise DomainError(f"no source configured for {fidelity.value}")
