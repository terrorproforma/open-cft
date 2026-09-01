"""Typed, versioned contracts for verification, validation, and evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum
import ipaddress
from math import isfinite, sqrt
import re
from typing import Any, Mapping
from urllib.parse import urlparse

CONTRACT_VERSION = "2.0.0"


class ValidationError(ValueError):
    """A validation-workstream contract was violated."""


class UnitError(ValidationError):
    """A quantity used an unknown or dimensionally incompatible unit."""


class EvidenceKind(str, Enum):
    ANALYTICAL = "analytical_verification"
    MANUFACTURED = "manufactured_solution_verification"
    CROSS_CODE = "cross_code_comparison"
    STOCHASTIC_PIC = "stochastic_pic_replicate"
    EXPERIMENT = "experiment"
    PUBLISHED_EXTERNAL = "published_external_model_output"


class SourceAuthority(str, Enum):
    ANALYTICAL_REFERENCE = "analytical_reference"
    MANUFACTURED_REFERENCE = "manufactured_reference"
    INDEPENDENT_CODE = "independent_code"
    SIMULATION = "simulation"
    EXPERIMENT = "experiment"
    PUBLISHED_MODEL_OUTPUT = "published_model_output"


class EvidencePartition(str, Enum):
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    REFERENCE_ONLY = "reference_only"


class CredibilityLevel(IntEnum):
    TRACEABLE = 1
    CHECKED = 2
    CORROBORATED = 3
    VALIDATION_SUPPORT = 4


class ClaimLevel(IntEnum):
    DIAGNOSTIC_ONLY = 1
    VERIFIED_IMPLEMENTATION = 2
    CROSS_MODEL_AGREEMENT = 3
    VALIDATION_EVIDENCE = 4
    PREDICTIVE_VALIDITY = 5


@dataclass(frozen=True, slots=True)
class _UnitDefinition:
    dimension: str
    to_si: float
    is_si: bool


_UNITS: dict[str, _UnitDefinition] = {
    "1": _UnitDefinition("dimensionless", 1.0, True),
    "fraction": _UnitDefinition("dimensionless", 1.0, False),
    "%": _UnitDefinition("dimensionless", 0.01, False),
    "m": _UnitDefinition("length", 1.0, True),
    "mm": _UnitDefinition("length", 1.0e-3, False),
    "N": _UnitDefinition("force", 1.0, True),
    "mN": _UnitDefinition("force", 1.0e-3, False),
    "s": _UnitDefinition("time", 1.0, True),
    "ms": _UnitDefinition("time", 1.0e-3, False),
    "W": _UnitDefinition("power", 1.0, True),
    "kW": _UnitDefinition("power", 1.0e3, False),
    "A": _UnitDefinition("current", 1.0, True),
    "V": _UnitDefinition("voltage", 1.0, True),
    "kg/s": _UnitDefinition("mass_flow", 1.0, True),
}


_KIND_AUTHORITY: dict[EvidenceKind, SourceAuthority] = {
    EvidenceKind.ANALYTICAL: SourceAuthority.ANALYTICAL_REFERENCE,
    EvidenceKind.MANUFACTURED: SourceAuthority.MANUFACTURED_REFERENCE,
    EvidenceKind.CROSS_CODE: SourceAuthority.INDEPENDENT_CODE,
    EvidenceKind.STOCHASTIC_PIC: SourceAuthority.SIMULATION,
    EvidenceKind.EXPERIMENT: SourceAuthority.EXPERIMENT,
    EvidenceKind.PUBLISHED_EXTERNAL: SourceAuthority.PUBLISHED_MODEL_OUTPUT,
}

_AUTHORITY_CLAIM_CEILING: dict[SourceAuthority, ClaimLevel] = {
    SourceAuthority.ANALYTICAL_REFERENCE: ClaimLevel.VERIFIED_IMPLEMENTATION,
    SourceAuthority.MANUFACTURED_REFERENCE: ClaimLevel.VERIFIED_IMPLEMENTATION,
    SourceAuthority.INDEPENDENT_CODE: ClaimLevel.CROSS_MODEL_AGREEMENT,
    SourceAuthority.SIMULATION: ClaimLevel.CROSS_MODEL_AGREEMENT,
    SourceAuthority.EXPERIMENT: ClaimLevel.PREDICTIVE_VALIDITY,
    SourceAuthority.PUBLISHED_MODEL_OUTPUT: ClaimLevel.CROSS_MODEL_AGREEMENT,
}

_AUTHORITY_CREDIBILITY_CEILING: dict[SourceAuthority, CredibilityLevel] = {
    SourceAuthority.ANALYTICAL_REFERENCE: CredibilityLevel.CHECKED,
    SourceAuthority.MANUFACTURED_REFERENCE: CredibilityLevel.CHECKED,
    SourceAuthority.INDEPENDENT_CODE: CredibilityLevel.CORROBORATED,
    SourceAuthority.SIMULATION: CredibilityLevel.CORROBORATED,
    SourceAuthority.EXPERIMENT: CredibilityLevel.VALIDATION_SUPPORT,
    SourceAuthority.PUBLISHED_MODEL_OUTPUT: CredibilityLevel.TRACEABLE,
}


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a number, not boolean")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be numeric") from exc
    if not isfinite(converted):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return converted


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _enum(name: str, value: Any, enum_type: type[Enum]) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} is not a valid {enum_type.__name__}") from exc


def validate_locator(name: str, value: str, *, require_web: bool = False) -> str:
    """Require a resolvable web URI or explicit artifact URI."""

    locator = _nonempty(name, value)
    if any(character.isspace() or ord(character) < 32 for character in locator):
        raise ValidationError(f"{name} must not contain whitespace or control characters")
    parsed = urlparse(locator)
    if parsed.scheme in {"http", "https"} and parsed.netloc and parsed.hostname:
        if parsed.username is not None or parsed.password is not None:
            raise ValidationError(f"{name} must not contain user information")
        try:
            parsed.port
        except ValueError as exc:
            raise ValidationError(f"{name} has an invalid port") from exc
        host = parsed.hostname
        try:
            ipaddress.ip_address(host)
        except ValueError:
            labels = host.rstrip(".").split(".")
            if any(
                not label
                or len(label) > 63
                or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                for label in labels
            ):
                raise ValidationError(f"{name} has an invalid absolute host")
        return locator
    if (
        not require_web
        and parsed.scheme == "artifact"
        and (parsed.netloc or parsed.path.strip("/"))
    ):
        return locator
    raise ValidationError(
        f"{name} must be an http(s) URI or artifact:// locator, got {value!r}"
    )


def validate_doi(value: str) -> str:
    doi = _nonempty("doi", value).removeprefix("https://doi.org/").strip()
    if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        raise ValidationError(
            "doi must match 10.<4-9 digits>/<non-whitespace suffix>"
        )
    if any(ord(character) < 33 or character.isspace() for character in doi):
        raise ValidationError("doi must not contain whitespace or control characters")
    return doi


@dataclass(frozen=True, slots=True)
class Quantity:
    """A finite scalar with an explicit supported unit."""

    value: float
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite("quantity value", self.value))
        if not isinstance(self.unit, str) or self.unit not in _UNITS:
            raise UnitError(f"unsupported unit {self.unit!r}")

    @property
    def dimension(self) -> str:
        return _UNITS[self.unit].dimension

    @property
    def is_si(self) -> bool:
        return _UNITS[self.unit].is_si

    @property
    def si_value(self) -> float:
        return self.value * _UNITS[self.unit].to_si

    def to(self, unit: str) -> Quantity:
        if unit not in _UNITS:
            raise UnitError(f"unsupported unit {unit!r}")
        target = _UNITS[unit]
        if target.dimension != self.dimension:
            raise UnitError(
                f"cannot convert {self.dimension} quantity from {self.unit!r} to {unit!r}"
            )
        return Quantity(self.si_value / target.to_si, unit)


@dataclass(frozen=True, slots=True)
class NamedQuantity:
    name: str
    quantity: Quantity

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("quantity name", self.name))
        if not isinstance(self.quantity, Quantity):
            raise ValidationError("named quantity value must be a Quantity")


@dataclass(frozen=True, slots=True)
class UncertaintyComponent:
    name: str
    standard_uncertainty: Quantity
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("uncertainty component name", self.name))
        object.__setattr__(self, "method", _nonempty("uncertainty method", self.method))
        if not isinstance(self.standard_uncertainty, Quantity):
            raise ValidationError("standard_uncertainty must be a Quantity")
        if self.standard_uncertainty.value < 0.0:
            raise ValidationError("standard uncertainty must be non-negative")


@dataclass(frozen=True, slots=True)
class ExperimentUncertainty:
    components: tuple[UncertaintyComponent, ...]
    coverage_factor: float = 2.0

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components or not all(
            isinstance(item, UncertaintyComponent) for item in components
        ):
            raise ValidationError("experiment uncertainty requires typed components")
        if len({item.standard_uncertainty.dimension for item in components}) != 1:
            raise UnitError("uncertainty components must have a common dimension")
        names = [item.name for item in components]
        if len(names) != len(set(names)):
            raise ValidationError("uncertainty component names must be unique")
        coverage = _finite("coverage_factor", self.coverage_factor)
        if coverage <= 0.0:
            raise ValidationError("coverage_factor must be greater than zero")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "coverage_factor", coverage)

    @property
    def standard_uncertainty_si(self) -> float:
        return sqrt(sum(item.standard_uncertainty.si_value**2 for item in self.components))

    @property
    def expanded_uncertainty_si(self) -> float:
        return self.coverage_factor * self.standard_uncertainty_si

    def interval(self, measured: Quantity) -> tuple[float, float]:
        if not isinstance(measured, Quantity):
            raise ValidationError("measured value must be a Quantity")
        if measured.dimension != self.components[0].standard_uncertainty.dimension:
            raise UnitError("measurement and uncertainty budget dimensions differ")
        width = self.expanded_uncertainty_si
        return measured.si_value - width, measured.si_value + width


@dataclass(frozen=True, slots=True)
class MetricObservation:
    name: str
    quantity: Quantity
    uncertainty: ExperimentUncertainty | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("metric name", self.name))
        if not isinstance(self.quantity, Quantity):
            raise ValidationError("metric quantity must be a Quantity")
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, ExperimentUncertainty):
                raise ValidationError("metric uncertainty must be ExperimentUncertainty")
            self.uncertainty.interval(self.quantity)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Source-native wording is preserved separately from editorial interpretation."""

    source_title: str
    source_native_label: str
    editorial_interpretation: str
    source_locator: str
    doi: str | None = None
    is_experimental_truth: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "source_title",
            "source_native_label",
            "editorial_interpretation",
        ):
            object.__setattr__(
                self, field_name, _nonempty(field_name, getattr(self, field_name))
            )
        object.__setattr__(
            self,
            "source_locator",
            validate_locator("source_locator", self.source_locator),
        )
        if self.doi is not None:
            if not isinstance(self.doi, str):
                raise ValidationError("doi must be a string when supplied")
            object.__setattr__(self, "doi", validate_doi(self.doi))
        if not isinstance(self.is_experimental_truth, bool):
            raise ValidationError("is_experimental_truth must be boolean")


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    facility: str
    hardware_article_id: str
    campaign_id: str
    acquired_at_utc: str
    instrument_ids: tuple[str, ...]
    raw_data_locator: str

    def __post_init__(self) -> None:
        for field_name in ("facility", "hardware_article_id", "campaign_id"):
            object.__setattr__(
                self, field_name, _nonempty(field_name, getattr(self, field_name))
            )
        timestamp = _nonempty("acquired_at_utc", self.acquired_at_utc)
        if not timestamp.endswith("Z"):
            raise ValidationError("acquired_at_utc must be an ISO-8601 UTC timestamp")
        try:
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError as exc:
            raise ValidationError("acquired_at_utc must be an ISO-8601 UTC timestamp") from exc
        instruments = tuple(self.instrument_ids)
        if not instruments or any(
            not isinstance(item, str) or not item.strip() for item in instruments
        ):
            raise ValidationError("instrument_ids must contain non-empty strings")
        if len(instruments) != len(set(instruments)):
            raise ValidationError("instrument_ids must be unique")
        object.__setattr__(self, "instrument_ids", instruments)
        object.__setattr__(
            self,
            "raw_data_locator",
            validate_locator("raw_data_locator", self.raw_data_locator),
        )


@dataclass(frozen=True, slots=True)
class IndependenceIdentity:
    """Immutable design/run/physical identifiers used instead of caller groups."""

    design_identity: str
    run_lineage_id: str
    hardware_article_id: str | None = None
    test_campaign_id: str | None = None
    specimen_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("design_identity", "run_lineage_id"):
            object.__setattr__(
                self, field_name, _nonempty(field_name, getattr(self, field_name))
            )
        for field_name in (
            "hardware_article_id",
            "test_campaign_id",
            "specimen_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _nonempty(field_name, value))

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(
            f"{name}:{value}"
            for name, value in (
                ("design", self.design_identity),
                ("run_lineage", self.run_lineage_id),
                ("hardware_article", self.hardware_article_id),
                ("test_campaign", self.test_campaign_id),
                ("specimen", self.specimen_id),
            )
            if value is not None
        )

    def overlaps(self, other: IndependenceIdentity) -> tuple[str, ...]:
        if not isinstance(other, IndependenceIdentity):
            raise ValidationError("independence comparison requires typed identities")
        return tuple(sorted(set(self.signature) & set(other.signature)))

    @property
    def independent_sample_key(self) -> tuple[str, ...]:
        """Prefer immutable specimen/article identity over caller labels or run IDs."""

        physical = tuple(
            value
            for value in (self.hardware_article_id, self.specimen_id)
            if value is not None
        )
        if physical:
            return physical
        return self.design_identity, self.run_lineage_id


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Immutable evidence with authority-limited claim semantics."""

    record_id: str
    kind: EvidenceKind
    source_authority: SourceAuthority
    partition: EvidencePartition
    observations: tuple[MetricObservation, ...]
    provenance: Provenance
    model_context_id: str
    result_context_id: str
    group_id: str
    independence_identity: IndependenceIdentity
    model_revision: str
    code_revision: str
    operating_parameters: tuple[NamedQuantity, ...] = ()
    credibility: CredibilityLevel = CredibilityLevel.TRACEABLE
    maximum_claim: ClaimLevel = ClaimLevel.DIAGNOSTIC_ONLY
    seed: int | None = None
    seed_policy_id: str | None = None
    mesh_id: str | None = None
    experiment_metadata: ExperimentMetadata | None = None
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "record_id",
            "model_context_id",
            "result_context_id",
            "group_id",
            "model_revision",
            "code_revision",
        ):
            object.__setattr__(
                self, field_name, _nonempty(field_name, getattr(self, field_name))
            )
        if self.schema_version != CONTRACT_VERSION:
            raise ValidationError(
                f"unsupported evidence schema {self.schema_version!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        kind = _enum("kind", self.kind, EvidenceKind)
        authority = _enum("source_authority", self.source_authority, SourceAuthority)
        partition = _enum("partition", self.partition, EvidencePartition)
        credibility = _enum("credibility", self.credibility, CredibilityLevel)
        claim = _enum("maximum_claim", self.maximum_claim, ClaimLevel)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_authority", authority)
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "credibility", credibility)
        object.__setattr__(self, "maximum_claim", claim)
        if authority is not _KIND_AUTHORITY[kind]:
            raise ValidationError(
                f"{kind.value} requires source authority {_KIND_AUTHORITY[kind].value}"
            )
        if claim > _AUTHORITY_CLAIM_CEILING[authority]:
            raise ValidationError(
                f"{authority.value} cannot support claim {claim.name.lower()}"
            )
        if credibility > _AUTHORITY_CREDIBILITY_CEILING[authority]:
            raise ValidationError(
                f"{authority.value} cannot assert credibility {credibility.name.lower()}"
            )
        observations = tuple(self.observations)
        if not observations or not all(
            isinstance(item, MetricObservation) for item in observations
        ):
            raise ValidationError("evidence requires typed metric observations")
        names = [item.name for item in observations]
        if len(names) != len(set(names)):
            raise ValidationError("metric observation names must be unique")
        object.__setattr__(self, "observations", observations)
        if not isinstance(self.provenance, Provenance):
            raise ValidationError("provenance must be a Provenance record")
        if not isinstance(self.independence_identity, IndependenceIdentity):
            raise ValidationError(
                "evidence requires an immutable IndependenceIdentity"
            )
        parameters = tuple(self.operating_parameters)
        if not all(isinstance(item, NamedQuantity) for item in parameters):
            raise ValidationError("operating_parameters must contain NamedQuantity records")
        parameter_names = [item.name for item in parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValidationError("operating parameter names must be unique")
        object.__setattr__(self, "operating_parameters", parameters)
        for optional_name in ("seed_policy_id", "mesh_id"):
            value = getattr(self, optional_name)
            if value is not None:
                object.__setattr__(self, optional_name, _nonempty(optional_name, value))
        if kind is EvidenceKind.STOCHASTIC_PIC:
            if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
                raise ValidationError("stochastic PIC evidence requires a non-negative seed")
            if self.seed_policy_id is None:
                raise ValidationError("stochastic PIC evidence requires seed_policy_id")
        elif self.seed is not None or self.seed_policy_id is not None:
            raise ValidationError("seed fields are reserved for stochastic PIC evidence")
        if kind is EvidenceKind.EXPERIMENT:
            if not self.provenance.is_experimental_truth:
                raise ValidationError(
                    "experiment evidence must explicitly identify measured truth"
                )
            if not isinstance(self.experiment_metadata, ExperimentMetadata):
                raise ValidationError("experiment evidence requires complete experiment metadata")
            if (
                self.independence_identity.hardware_article_id
                != self.experiment_metadata.hardware_article_id
                or self.independence_identity.test_campaign_id
                != self.experiment_metadata.campaign_id
                or self.independence_identity.specimen_id is None
            ):
                raise ValidationError(
                    "experiment independence identity must match hardware/campaign "
                    "metadata and include specimen_id"
                )
            if any(item.uncertainty is None for item in observations):
                raise ValidationError("every experiment observation requires uncertainty")
            if claim >= ClaimLevel.VALIDATION_EVIDENCE:
                if partition is not EvidencePartition.VALIDATION:
                    raise ValidationError(
                        "experiment validation claims require the validation partition"
                    )
                if credibility < CredibilityLevel.VALIDATION_SUPPORT:
                    raise ValidationError(
                        "experiment validation claims require validation-support credibility"
                    )
        else:
            if self.provenance.is_experimental_truth:
                raise ValidationError(
                    f"{authority.value} evidence cannot be labelled experimental truth"
                )
            if self.experiment_metadata is not None:
                raise ValidationError("experiment metadata is reserved for experiment evidence")
        if (
            kind is EvidenceKind.PUBLISHED_EXTERNAL
            and partition is not EvidencePartition.REFERENCE_ONLY
        ):
            raise ValidationError("published external evidence must be reference-only")
        if (
            partition is EvidencePartition.CALIBRATION
            and claim >= ClaimLevel.VALIDATION_EVIDENCE
        ):
            raise ValidationError("calibration evidence cannot support a validation claim")

    @property
    def authority_claim_ceiling(self) -> ClaimLevel:
        return _AUTHORITY_CLAIM_CEILING[self.source_authority]


@dataclass(frozen=True, slots=True)
class ReplicateEnsemble:
    """Homogeneous stochastic PIC replicates from one predeclared seed policy."""

    records: tuple[EvidenceRecord, ...]
    minimum_replicates: int = 3

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if (
            not isinstance(self.minimum_replicates, int)
            or isinstance(self.minimum_replicates, bool)
            or self.minimum_replicates < 3
        ):
            raise ValidationError("minimum_replicates must be at least three")
        if len(records) < self.minimum_replicates:
            raise ValidationError("insufficient stochastic PIC replicates")
        if not all(isinstance(item, EvidenceRecord) for item in records):
            raise ValidationError("replicate ensemble requires typed evidence records")
        if any(item.kind is not EvidenceKind.STOCHASTIC_PIC for item in records):
            raise ValidationError("replicate ensemble accepts only stochastic PIC evidence")
        identities = {
            (
                item.source_authority,
                item.model_context_id,
                item.result_context_id,
                item.group_id,
                item.independence_identity.signature,
                item.partition,
                item.model_revision,
                item.code_revision,
                item.seed_policy_id,
                item.mesh_id,
                tuple(
                    (parameter.name, parameter.quantity.value, parameter.quantity.unit)
                    for parameter in item.operating_parameters
                ),
            )
            for item in records
        }
        if len(identities) != 1:
            raise ValidationError(
                "PIC replicates must share authority, model/code/result context, "
                "group, seed policy, mesh, and operating parameters"
            )
        seeds = [item.seed for item in records]
        if len(seeds) != len(set(seeds)):
            raise ValidationError("PIC replicate seeds must be unique")
        metric_schemas = {
            tuple((item.name, item.quantity.unit) for item in record.observations)
            for record in records
        }
        if len(metric_schemas) != 1:
            raise ValidationError("PIC replicates must share exact quantity names and units")
        object.__setattr__(self, "records", records)


@dataclass(frozen=True, slots=True)
class ApplicabilityRange:
    parameter: str
    minimum: Quantity
    maximum: Quantity

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter", _nonempty("parameter", self.parameter))
        if not isinstance(self.minimum, Quantity) or not isinstance(self.maximum, Quantity):
            raise ValidationError("applicability endpoints must be Quantity records")
        if self.minimum.unit != self.maximum.unit or not self.minimum.is_si:
            raise UnitError("applicability endpoints must use the same explicit SI unit")
        if self.minimum.si_value >= self.maximum.si_value:
            raise ValidationError("applicability range minimum must be below maximum")


@dataclass(frozen=True, slots=True)
class QuantityRequirement:
    name: str
    si_unit: str
    minimum_evidence: int
    minimum_independent_groups: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("quantity requirement name", self.name))
        if self.si_unit not in _UNITS or not _UNITS[self.si_unit].is_si:
            raise UnitError("quantity requirements must name a supported SI unit")
        for field_name in ("minimum_evidence", "minimum_independent_groups"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValidationError(f"{field_name} must be a positive integer")
        if self.minimum_independent_groups > self.minimum_evidence:
            raise ValidationError("independent-group minimum cannot exceed evidence minimum")


@dataclass(frozen=True, slots=True)
class ContextOfUseLedger:
    """NASA-STD-7009B-style context ledger; never a certification record."""

    ledger_id: str
    decision: str
    consequence_of_wrong_decision: str
    model_role: str
    quantity_requirements: tuple[QuantityRequirement, ...]
    applicability_ranges: tuple[ApplicabilityRange, ...]
    required_model_context_id: str
    required_result_context_id: str
    required_model_revision: str
    required_code_revision: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    validation_evidence_ids: tuple[str, ...]
    certification_claimed: bool = False
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "ledger_id",
            "decision",
            "consequence_of_wrong_decision",
            "model_role",
            "required_model_context_id",
            "required_result_context_id",
            "required_model_revision",
            "required_code_revision",
        ):
            object.__setattr__(
                self, field_name, _nonempty(field_name, getattr(self, field_name))
            )
        if self.schema_version != CONTRACT_VERSION:
            raise ValidationError("unsupported context-of-use schema version")
        if not isinstance(self.certification_claimed, bool):
            raise ValidationError("certification_claimed must be boolean")
        if self.certification_claimed:
            raise ValidationError(
                "this ledger is NASA-STD-7009B-style and must not claim certification"
            )
        requirements = tuple(self.quantity_requirements)
        if not requirements or not all(
            isinstance(item, QuantityRequirement) for item in requirements
        ):
            raise ValidationError("quantity_requirements must contain typed requirements")
        names = [item.name for item in requirements]
        if len(names) != len(set(names)):
            raise ValidationError("quantity requirement names must be unique")
        object.__setattr__(self, "quantity_requirements", requirements)
        ranges = tuple(self.applicability_ranges)
        if not ranges or not all(isinstance(item, ApplicabilityRange) for item in ranges):
            raise ValidationError("applicability_ranges must contain typed ranges")
        parameters = [item.parameter for item in ranges]
        if len(parameters) != len(set(parameters)):
            raise ValidationError("applicability range parameters must be unique")
        object.__setattr__(self, "applicability_ranges", ranges)
        for field_name in ("assumptions", "limitations", "validation_evidence_ids"):
            values = tuple(getattr(self, field_name))
            if not values or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValidationError(f"{field_name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise ValidationError(f"{field_name} must not contain duplicates")
            object.__setattr__(self, field_name, values)

    def contains(self, parameters: Mapping[str, Quantity]) -> bool:
        expected = {item.parameter for item in self.applicability_ranges}
        if set(parameters) != expected:
            raise ValidationError(
                "applicability check must supply exactly the declared parameters"
            )
        for item in self.applicability_ranges:
            value = parameters[item.parameter]
            if not isinstance(value, Quantity):
                raise ValidationError("applicability values must be Quantity records")
            if value.unit != item.minimum.unit:
                raise UnitError(
                    f"applicability parameter {item.parameter} must use {item.minimum.unit}"
                )
            if not item.minimum.si_value <= value.si_value <= item.maximum.si_value:
                return False
        return True
