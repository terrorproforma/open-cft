import json
from dataclasses import replace
from pathlib import Path

import pytest

from cft_revival.validation import (
    ClaimLevel,
    ConflictingEvidenceError,
    DuplicateEvidenceError,
    EvidenceKind,
    EvidencePartition,
    EvidenceRecord,
    EvidenceRegistry,
    EvidenceSerializationError,
    INTEGRITY_NOTICE,
    IndependenceIdentity,
    MetricObservation,
    Provenance,
    Quantity,
    SourceAuthority,
    content_sha256,
    evidence_hash,
    load_published_evidence,
    strict_json_loads,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "data" / "validation" / "yeo-2020-s1-external-evidence-v2.json"


def _write_rehashed(tmp_path: Path, payload: dict, name: str) -> Path:
    payload["bundle_sha256"] = content_sha256(
        {key: value for key, value in payload.items() if key != "bundle_sha256"}
    )
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(record_id: str, value: float = 1.0) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        kind=EvidenceKind.CROSS_CODE,
        source_authority=SourceAuthority.INDEPENDENT_CODE,
        partition=EvidencePartition.VALIDATION,
        observations=(MetricObservation("thrust", Quantity(value, "N")),),
        provenance=Provenance(
            "Code B",
            "Code B native",
            "independent-code fixture",
            "artifact://validation/code-b",
        ),
        model_context_id="equations-v1",
        result_context_id="outputs-v1",
        group_id="case-1",
        independence_identity=IndependenceIdentity("design-1", "run-1"),
        model_revision="model-v1",
        code_revision="code-b-v1",
        maximum_claim=ClaimLevel.CROSS_MODEL_AGREEMENT,
    )


def test_published_labels_values_and_citation_are_source_faithful() -> None:
    records = load_published_evidence(PUBLISHED)
    assert [item.provenance.source_native_label for item in records] == [
        "MDO (original)",
        "PIC",
        "MDO (modified)",
    ]
    assert [item.record_id for item in records] == [
        "YEO2020-S1-MDO-ORIGINAL",
        "YEO2020-S1-PIC",
        "YEO2020-S1-MDO-MODIFIED",
    ]
    by_label = {item.provenance.source_native_label: item for item in records}
    expected = {
        "MDO (original)": (102.7, 36.5, 2131.0),
        "PIC": (62.8, 15.2, 1333.0),
        "MDO (modified)": (61.7, 14.6, 1280.0),
    }
    for label, values in expected.items():
        record = by_label[label]
        assert record.provenance.doi == "10.2514/1.A34584"
        assert record.provenance.source_locator == "https://doi.org/10.2514/1.A34584"
        assert record.kind is EvidenceKind.PUBLISHED_EXTERNAL
        assert record.source_authority is SourceAuthority.PUBLISHED_MODEL_OUTPUT
        assert not record.provenance.is_experimental_truth
        assert "editorial" in record.provenance.editorial_interpretation.lower() or (
            label == "PIC" and "benchmark" in record.provenance.editorial_interpretation
        )
        assert tuple(item.quantity.value for item in record.observations) == values


def test_hash_detects_unrehashed_change_but_is_not_authentication(tmp_path: Path) -> None:
    original = PUBLISHED.read_text(encoding="utf-8")
    tampered = tmp_path / "tampered.json"
    tampered.write_text(original.replace("102.7", "102.8"), encoding="utf-8")
    with pytest.raises(EvidenceSerializationError, match="hash mismatch"):
        load_published_evidence(tampered)
    assert INTEGRITY_NOTICE in original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["records"][0].update({"extra": 1}), "keys do not match"),
        (
            lambda data: data["records"][0]["observations"][0].update({"value": True}),
            "not boolean",
        ),
        (
            lambda data: data["records"][0].update({"editorial_interpretation": None}),
            "null is forbidden",
        ),
        (
            lambda data: data["records"][0].update({"role": "experimental_truth"}),
            "role must equal",
        ),
        (
            lambda data: data["source"].update({"source_url": "relative/path"}),
            "must be an http",
        ),
        (
            lambda data: data["source"].update(
                {"source_url": "https://user@example.com/source"}
            ),
            "must not contain user",
        ),
        (
            lambda data: data["source"].update({"doi": "10.123/x"}),
            "must match",
        ),
        (
            lambda data: data["source"].update({"publication_year": True}),
            "integer, not boolean",
        ),
        (
            lambda data: data["records"][0]["observations"][1].update({"value": 101.0}),
            "efficiency must be",
        ),
    ],
)
def test_recomputed_malicious_hash_does_not_bypass_schema(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    mutation(payload)
    path = _write_rehashed(tmp_path, payload, "rehashed.json")
    with pytest.raises(EvidenceSerializationError, match=message):
        load_published_evidence(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"x": 1, "x": 2}',
        '{"outer": {"x": 1, "x": 2}}',
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": 1e999}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(payload: str) -> None:
    with pytest.raises(EvidenceSerializationError):
        strict_json_loads(payload)


def test_strict_json_wraps_huge_integer_overflow() -> None:
    payload = '{"value": ' + ("9" * 10000) + "}"
    with pytest.raises(EvidenceSerializationError, match="overflows"):
        strict_json_loads(payload)


def test_registry_detects_duplicate_conflict_and_unit_sensitive_hashes() -> None:
    first = _record("one")
    registry = EvidenceRegistry((first,))
    with pytest.raises(DuplicateEvidenceError):
        registry.add(first)
    with pytest.raises(ConflictingEvidenceError):
        registry.add(
            replace(
                first,
                observations=(MetricObservation("thrust", Quantity(2.0, "N")),),
            )
        )
    with pytest.raises(DuplicateEvidenceError):
        registry.add(replace(first, record_id="two"))
    assert evidence_hash(first) != evidence_hash(
        replace(
            first,
            observations=(MetricObservation("thrust", Quantity(1000.0, "mN")),),
        )
    )


def test_validation_specs_are_strict_json_and_version_aligned() -> None:
    for name in (
        "evidence-contract-v2.schema.json",
        "integration-contract-v2.json",
    ):
        payload = strict_json_loads(
            (ROOT / "spec" / "validation" / name).read_text(encoding="utf-8")
        )
        assert isinstance(payload, dict)
        if name.startswith("integration"):
            assert payload["schema_version"] == "2.0.0"
        else:
            assert payload["properties"]["schema_version"]["const"] == "2.0.0"
