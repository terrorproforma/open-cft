"""Canonical evidence hashing, strict loading, and conflict detection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields, is_dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import (
    CONTRACT_VERSION,
    ClaimLevel,
    CredibilityLevel,
    EvidenceKind,
    EvidencePartition,
    EvidenceRecord,
    IndependenceIdentity,
    MetricObservation,
    Provenance,
    Quantity,
    SourceAuthority,
    ValidationError,
    validate_doi,
    validate_locator,
)

INTEGRITY_NOTICE = (
    "SHA-256 detects accidental or post-publication modification only; it is not "
    "authentication. An attacker who can replace the payload can recompute its hash."
)


class EvidenceSerializationError(ValidationError):
    """Evidence JSON is malformed or fails its declared integrity identity."""


class DuplicateEvidenceError(ValidationError):
    """The same evidence identity or semantic payload was supplied twice."""


class ConflictingEvidenceError(ValidationError):
    """An evidence identity was reused for different content."""


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceSerializationError("canonical mappings require string keys")
            if key in result:
                raise EvidenceSerializationError(f"duplicate canonical key {key!r}")
            result[key] = _plain(item)
        return result
    if isinstance(value, float) and not isfinite(value):
        raise EvidenceSerializationError("non-finite numbers are forbidden in evidence")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise EvidenceSerializationError(f"unsupported evidence value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evidence_hash(record: EvidenceRecord) -> str:
    if not isinstance(record, EvidenceRecord):
        raise EvidenceSerializationError("evidence_hash requires an EvidenceRecord")
    return content_sha256(record)


def _evidence_payload_hash(record: EvidenceRecord) -> str:
    payload = _plain(record)
    if not isinstance(payload, dict):
        raise EvidenceSerializationError("evidence payload must serialize as an object")
    return content_sha256({key: value for key, value in payload.items() if key != "record_id"})


def _reject_constant(token: str) -> None:
    raise EvidenceSerializationError(f"non-finite JSON token {token!r} is forbidden")


def _parse_int(token: str) -> int:
    try:
        value = int(token)
    except (ValueError, OverflowError) as exc:
        raise EvidenceSerializationError(
            "decoded integer overflows supported finite range"
        ) from exc
    try:
        converted = float(value)
    except OverflowError as exc:
        raise EvidenceSerializationError(
            "decoded integer overflows finite binary64 range"
        ) from exc
    if not isfinite(converted):
        raise EvidenceSerializationError(
            "decoded integer overflows finite binary64 range"
        )
    return value


def _parse_float(token: str) -> float:
    try:
        value = float(token)
    except (ValueError, OverflowError) as exc:
        raise EvidenceSerializationError(
            "decoded float overflows supported finite range"
        ) from exc
    if not isfinite(value):
        raise EvidenceSerializationError("decoded float overflows finite range")
    return value


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceSerializationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: str) -> Any:
    if not isinstance(payload, str):
        raise EvidenceSerializationError("JSON payload must be text")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
            parse_int=_parse_int,
            parse_float=_parse_float,
        )
    except EvidenceSerializationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceSerializationError("invalid evidence JSON") from exc
    _check_finite_tree(value)
    return value


def _check_finite_tree(value: Any) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            converted = float(value)
        except (ValueError, OverflowError) as exc:
            raise EvidenceSerializationError(
                "decoded number overflows finite binary64 range"
            ) from exc
        if not isfinite(converted):
            raise EvidenceSerializationError("non-finite decoded number is forbidden")
    if isinstance(value, list):
        for item in value:
            _check_finite_tree(item)
    elif isinstance(value, dict):
        for item in value.values():
            _check_finite_tree(item)


def _reject_null_tree(value: Any, path: str = "$") -> None:
    if value is None:
        raise EvidenceSerializationError(f"null is forbidden at {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_null_tree(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_null_tree(item, f"{path}.{key}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceSerializationError(f"{context} must be an object")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise EvidenceSerializationError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceSerializationError(f"{context} must be a non-empty string")
    return value.strip()


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceSerializationError(f"{context} must be a number, not boolean")
    try:
        converted = float(value)
    except (ValueError, OverflowError) as exc:
        raise EvidenceSerializationError(f"{context} overflows finite range") from exc
    if not isfinite(converted):
        raise EvidenceSerializationError(f"{context} must be finite")
    return converted


def _integer(value: Any, context: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceSerializationError(f"{context} must be an integer, not boolean")
    if not minimum <= value <= maximum:
        raise EvidenceSerializationError(
            f"{context} must be in [{minimum}, {maximum}]"
        )
    return value


class EvidenceRegistry:
    """Content-addressed registry with record and semantic-payload identities."""

    def __init__(self, records: Iterable[EvidenceRecord] = ()) -> None:
        self._by_id: dict[str, EvidenceRecord] = {}
        self._payload_hash_to_id: dict[str, str] = {}
        for record in records:
            self.add(record)

    def add(self, record: EvidenceRecord) -> str:
        if not isinstance(record, EvidenceRecord):
            raise ValidationError("registry accepts only EvidenceRecord values")
        digest = evidence_hash(record)
        payload_digest = _evidence_payload_hash(record)
        existing = self._by_id.get(record.record_id)
        if existing is not None:
            if evidence_hash(existing) == digest:
                raise DuplicateEvidenceError(f"duplicate evidence id {record.record_id!r}")
            raise ConflictingEvidenceError(
                f"evidence id {record.record_id!r} has conflicting content"
            )
        duplicate_id = self._payload_hash_to_id.get(payload_digest)
        if duplicate_id is not None:
            raise DuplicateEvidenceError(
                f"evidence payload duplicates {duplicate_id!r} under {record.record_id!r}"
            )
        self._by_id[record.record_id] = record
        self._payload_hash_to_id[payload_digest] = record.record_id
        return digest

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))

    @property
    def identity_sha256(self) -> str:
        return content_sha256(
            {record.record_id: evidence_hash(record) for record in self.records}
        )

    def get(self, record_id: str) -> EvidenceRecord:
        try:
            return self._by_id[record_id]
        except KeyError as exc:
            raise ValidationError(f"unknown evidence id {record_id!r}") from exc


def load_published_evidence(path: str | Path) -> tuple[EvidenceRecord, ...]:
    """Verify bundle integrity first, then enforce the complete closed schema."""

    payload = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    bundle = _mapping(payload, "evidence bundle")
    declared_hash = bundle.get("bundle_sha256")
    if not isinstance(declared_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", declared_hash
    ):
        raise EvidenceSerializationError("bundle_sha256 must be 64 lowercase hex digits")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if declared_hash != content_sha256(unsigned):
        raise EvidenceSerializationError("evidence bundle hash mismatch")

    _reject_null_tree(bundle)
    _exact_keys(
        bundle,
        {
            "schema_version",
            "bundle_sha256",
            "integrity_notice",
            "policy",
            "source",
            "records",
        },
        "bundle",
    )
    if bundle["schema_version"] != CONTRACT_VERSION:
        raise EvidenceSerializationError("unsupported bundle schema version")
    if bundle["integrity_notice"] != INTEGRITY_NOTICE:
        raise EvidenceSerializationError("bundle integrity notice is missing or altered")
    _string(bundle["policy"], "policy")
    source = _parse_source(bundle["source"])
    raw_records = bundle["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise EvidenceSerializationError("bundle records must be a non-empty array")
    parsed = tuple(_parse_published_record(item, source) for item in raw_records)
    EvidenceRegistry(parsed)
    return parsed


def _parse_source(value: Any) -> Mapping[str, Any]:
    source = _mapping(value, "source")
    _exact_keys(source, {"title", "doi", "source_url", "publication_year"}, "source")
    title = _string(source["title"], "source.title")
    doi = _string(source["doi"], "source.doi")
    try:
        doi = validate_doi(doi)
    except ValidationError as exc:
        raise EvidenceSerializationError(str(exc)) from exc
    source_url = _string(source["source_url"], "source.source_url")
    try:
        validate_locator("source.source_url", source_url, require_web=True)
    except ValidationError as exc:
        raise EvidenceSerializationError(str(exc)) from exc
    year = _integer(source["publication_year"], "source.publication_year", 1900, 2100)
    return {
        "title": title,
        "doi": doi,
        "source_url": source_url,
        "publication_year": year,
    }


def _parse_published_record(
    value: Any, source: Mapping[str, Any]
) -> EvidenceRecord:
    record = _mapping(value, "published record")
    _exact_keys(
        record,
        {
            "record_id",
            "source_native_label",
            "editorial_interpretation",
            "role",
            "kind",
            "source_authority",
            "partition",
            "model_context_id",
            "result_context_id",
            "group_id",
            "independence_identity",
            "model_revision",
            "code_revision",
            "observations",
        },
        "published record",
    )
    expected_enums = {
        "role": "external_cross_model_only_not_experimental_truth",
        "kind": EvidenceKind.PUBLISHED_EXTERNAL.value,
        "source_authority": SourceAuthority.PUBLISHED_MODEL_OUTPUT.value,
        "partition": EvidencePartition.REFERENCE_ONLY.value,
    }
    for key, expected in expected_enums.items():
        if record[key] != expected:
            raise EvidenceSerializationError(
                f"published record {key} must equal {expected!r}"
            )
    observations = record["observations"]
    if not isinstance(observations, list) or not observations:
        raise EvidenceSerializationError("published observations must be a non-empty array")
    parsed_observations = tuple(_parse_published_observation(item) for item in observations)
    independence = _parse_independence_identity(record["independence_identity"])
    try:
        return EvidenceRecord(
            record_id=_string(record["record_id"], "record_id"),
            kind=EvidenceKind.PUBLISHED_EXTERNAL,
            source_authority=SourceAuthority.PUBLISHED_MODEL_OUTPUT,
            partition=EvidencePartition.REFERENCE_ONLY,
            observations=parsed_observations,
            provenance=Provenance(
                source_title=source["title"],
                source_native_label=_string(
                    record["source_native_label"], "source_native_label"
                ),
                editorial_interpretation=_string(
                    record["editorial_interpretation"], "editorial_interpretation"
                ),
                source_locator=source["source_url"],
                doi=source["doi"],
                is_experimental_truth=False,
            ),
            model_context_id=_string(record["model_context_id"], "model_context_id"),
            result_context_id=_string(
                record["result_context_id"], "result_context_id"
            ),
            group_id=_string(record["group_id"], "group_id"),
            independence_identity=independence,
            model_revision=_string(record["model_revision"], "model_revision"),
            code_revision=_string(record["code_revision"], "code_revision"),
            credibility=CredibilityLevel.TRACEABLE,
            maximum_claim=ClaimLevel.CROSS_MODEL_AGREEMENT,
        )
    except ValidationError as exc:
        raise EvidenceSerializationError(str(exc)) from exc


def _parse_published_observation(value: Any) -> MetricObservation:
    observation = _mapping(value, "published observation")
    _exact_keys(observation, {"name", "value", "unit"}, "published observation")
    name = _string(observation["name"], "observation.name")
    number = _number(observation["value"], f"{name}.value")
    unit = _string(observation["unit"], f"{name}.unit")
    try:
        quantity = Quantity(number, unit)
    except ValidationError as exc:
        raise EvidenceSerializationError(str(exc)) from exc
    if name == "thrust":
        if quantity.dimension != "force" or quantity.si_value <= 0.0:
            raise EvidenceSerializationError("thrust must be positive force")
    elif name == "reported_model_efficiency":
        if unit not in {"%", "fraction"}:
            raise EvidenceSerializationError("efficiency must use % or fraction")
        if not 0.0 <= quantity.si_value <= 1.0:
            raise EvidenceSerializationError("efficiency must be in [0, 1]")
    elif name == "specific_impulse":
        if quantity.dimension != "time" or quantity.si_value <= 0.0:
            raise EvidenceSerializationError("specific_impulse must be positive time")
    else:
        raise EvidenceSerializationError(f"unknown published metric {name!r}")
    return MetricObservation(name, quantity)


def _parse_independence_identity(value: Any) -> IndependenceIdentity:
    identity = _mapping(value, "independence_identity")
    _exact_keys(
        identity,
        {
            "design_identity",
            "run_lineage_id",
            "hardware_article_id",
            "test_campaign_id",
            "specimen_id",
        },
        "independence_identity",
    )
    optional: dict[str, str | None] = {}
    for name in ("hardware_article_id", "test_campaign_id", "specimen_id"):
        raw = identity[name]
        if raw == "not_applicable":
            optional[name] = None
        else:
            optional[name] = _string(raw, f"independence_identity.{name}")
    try:
        return IndependenceIdentity(
            design_identity=_string(
                identity["design_identity"], "independence_identity.design_identity"
            ),
            run_lineage_id=_string(
                identity["run_lineage_id"], "independence_identity.run_lineage_id"
            ),
            **optional,
        )
    except ValidationError as exc:
        raise EvidenceSerializationError(str(exc)) from exc
