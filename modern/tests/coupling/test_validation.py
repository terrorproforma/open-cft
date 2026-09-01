from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from cft_revival.coupling import (
    CouplingValidationError,
    FieldProvenance,
    MapValidationPolicy,
    hash_axisymmetric_map,
    validate_axisymmetric_map,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


@dataclass
class GenericMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


def field_map() -> GenericMap:
    r = (0.0, 0.05, 0.1)
    z = tuple(index * 0.01 for index in range(7))
    zeros = (0.0,) * len(z)
    return GenericMap(r, z, (zeros, zeros, zeros), (z, z, z))


def provenance(**changes: object) -> FieldProvenance:
    values = {
        "field_model_id": "analytic-test",
        "field_model_hash": HASH_A,
        "source_hash": HASH_B,
        "generated_at_utc": NOW,
        **changes,
    }
    return FieldProvenance(**values)


def test_generic_map_is_copied_validated_and_hashed_deterministically() -> None:
    first = validate_axisymmetric_map(field_map(), provenance(), reference_time_utc=NOW)
    second = validate_axisymmetric_map(field_map(), provenance(), reference_time_utc=NOW)
    assert first.field_map_hash == second.field_map_hash
    assert first.field_map_hash == hash_axisymmetric_map(
        first.r_m, first.z_m, first.b_r_t, first.b_z_t
    )
    changed = field_map()
    changed.b_z_t = (changed.b_z_t[0], changed.b_z_t[1], tuple(reversed(changed.z_m)))
    assert hash_axisymmetric_map(
        changed.r_m, changed.z_m, changed.b_r_t, changed.b_z_t
    ) != first.field_map_hash


@pytest.mark.parametrize(
    ("attribute", "replacement", "message"),
    [
        ("r_m", (0.0, 0.1, 0.05), "strictly increasing"),
        ("z_m", (0.0, 0.01, 0.02, 0.03, 0.04, 0.03, 0.06), "strictly increasing"),
        ("z_m", (0.0, 0.01), "undersampled"),
        (
            "b_z_t",
            ((0.0,) * 7, (0.0,) * 6, (0.0,) * 7),
            "row lengths",
        ),
        (
            "b_z_t",
            ((0.0,) * 7, (0.0,) * 7, (0.0,) * 6 + (float("nan"),)),
            "finite",
        ),
    ],
)
def test_malformed_inverted_nonfinite_and_undersampled_maps_fail_closed(
    attribute: str, replacement: object, message: str
) -> None:
    field = field_map()
    setattr(field, attribute, replacement)
    with pytest.raises(CouplingValidationError, match=message):
        validate_axisymmetric_map(field, provenance(), reference_time_utc=NOW)


def test_axis_regularity_is_checked() -> None:
    field = field_map()
    field.b_r_t = ((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),) + field.b_r_t[1:]
    with pytest.raises(CouplingValidationError, match="axis regularity"):
        validate_axisymmetric_map(field, provenance(), reference_time_utc=NOW)


def test_explicit_nonconvergence_and_nonfinite_direct_hashing_are_rejected() -> None:
    field = field_map()
    field.diagnostics = type("Diagnostics", (), {"converged": False})()
    with pytest.raises(CouplingValidationError, match="convergence"):
        validate_axisymmetric_map(field, provenance(), reference_time_utc=NOW)
    with pytest.raises(CouplingValidationError, match="finite"):
        hash_axisymmetric_map(
            (0.0, 1.0),
            (0.0, 1.0),
            ((0.0, 0.0), (0.0, float("nan"))),
            ((0.0, 0.0), (0.0, 0.0)),
        )


def test_si_hash_and_freshness_provenance_are_required() -> None:
    with pytest.raises(CouplingValidationError, match="metres and tesla"):
        validate_axisymmetric_map(
            field_map(),
            provenance(coordinate_unit="mm"),
            reference_time_utc=NOW,
        )
    with pytest.raises(CouplingValidationError, match="SHA-256"):
        validate_axisymmetric_map(
            field_map(),
            provenance(field_model_hash="not-a-hash"),
            reference_time_utc=NOW,
        )
    stale = provenance(generated_at_utc=NOW - timedelta(seconds=11))
    with pytest.raises(CouplingValidationError, match="stale"):
        validate_axisymmetric_map(
            field_map(),
            stale,
            MapValidationPolicy(maximum_age_s=10),
            reference_time_utc=NOW,
        )
    with pytest.raises(CouplingValidationError, match="timezone-aware"):
        validate_axisymmetric_map(
            field_map(),
            provenance(generated_at_utc=datetime(2026, 9, 1)),
            reference_time_utc=NOW,
        )
