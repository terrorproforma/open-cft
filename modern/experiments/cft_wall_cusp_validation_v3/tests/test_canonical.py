from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import pytest

from experiments.cft_wall_cusp_validation_v3.canonical import (
    CanonicalTypeError,
    CanonicalValueError,
    TaggedSchema,
    canonical_bytes,
    diagnose_bytes,
    load_canonical,
    semantic_hash,
    write_canonical,
)


class DuplicateKeyMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        return 1

    def __iter__(self) -> Iterator[str]:
        yield "same"
        yield "same"

    def __len__(self) -> int:
        return 2

    def items(self):  # type: ignore[override]
        return (("same", 1), ("same", 2))


@pytest.mark.parametrize(
    "value",
    (
        Path("forbidden"),
        date(2026, 9, 2),
        time(1, 2, 3),
        b"forbidden",
        {1, 2},
        Decimal("1.25"),
        Fraction(1, 3),
        UUID("00000000-0000-0000-0000-000000000001"),
        1 + 2j,
        (item for item in (1, 2)),
    ),
)
def test_unsupported_values_are_rejected_with_object_paths(value: object) -> None:
    with pytest.raises(CanonicalTypeError, match=r"\$\.outer\[1\]\.bad"):
        canonical_bytes({"outer": [0, {"bad": value}]})


def test_naive_datetime_nonfinite_and_collision_are_path_diagnostic() -> None:
    with pytest.raises(CanonicalValueError, match=r"\$\.clock"):
        canonical_bytes({"clock": datetime.now()})
    with pytest.raises(CanonicalValueError, match=r"\$\.metric"):
        canonical_bytes({"metric": float("nan")})
    with pytest.raises(CanonicalValueError, match=r"\$\.same"):
        canonical_bytes(DuplicateKeyMapping())


def test_aware_datetime_is_deterministically_normalized_to_utc() -> None:
    local = datetime(
        2026,
        9,
        2,
        10,
        15,
        1,
        123,
        tzinfo=timezone(timedelta(hours=10)),
    )
    utc = datetime(2026, 9, 2, 0, 15, 1, 123, tzinfo=timezone.utc)
    assert canonical_bytes({"clock": local}) == canonical_bytes({"clock": utc})
    assert b"2026-09-02T00:15:01.000123Z" in canonical_bytes({"clock": local})


def test_explicit_tagged_schema_is_the_only_unsupported_escape_hatch() -> None:
    encoded = canonical_bytes({"data": TaggedSchema("bytes-hex", b"\x00\xff")})
    assert encoded == b'{"data":{"$type":"bytes-hex","value":"00ff"}}'


def test_exact_persisted_bytes_are_the_semantically_hashed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "evidence.canonical.json"
    stored = write_canonical(
        path,
        {"z": 2, "a": {"clock": datetime(2026, 1, 1, tzinfo=timezone.utc)}},
        exclusive=True,
    )
    assert path.read_bytes() == canonical_bytes(stored)
    assert load_canonical(path) == stored
    body = dict(stored)
    body.pop("semantic_integrity")
    assert stored["semantic_integrity"]["payload_sha256"] == semantic_hash(body)


def test_payload_is_canonicalized_before_exclusive_create(tmp_path: Path) -> None:
    target = tmp_path / "must-not-exist.json"
    with pytest.raises(CanonicalTypeError):
        write_canonical(
            target,
            {"bad": {"deep": Path("forbidden")}},
            exclusive=True,
        )
    assert not target.exists()


def test_empty_and_truncated_lock_bytes_remain_diagnosable() -> None:
    empty = diagnose_bytes(b"", source="empty-lock")
    truncated = diagnose_bytes(b'{"schema_version":', source="truncated-lock")
    assert not empty["canonical"] and empty["byte_count"] == 0
    assert not truncated["canonical"] and "invalid canonical JSON" in truncated["error"]


def test_numpy_and_device_objects_are_explicitly_rejected() -> None:
    numpy = pytest.importorskip("numpy")
    with pytest.raises(CanonicalTypeError, match="numerical/device"):
        canonical_bytes({"value": numpy.float64(1.0)})
    Device = type("Device", (), {"__module__": "warp.context"})
    with pytest.raises(CanonicalTypeError, match="numerical/device"):
        canonical_bytes({"device": Device()})
