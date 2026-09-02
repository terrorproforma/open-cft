from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from dataclasses import asdict
from math import copysign, nextafter

import pytest

from cft_revival.fields import (
    ARTIFACT_SCHEMA_VERSION,
    LEGACY_ARTIFACT_SCHEMA_VERSION,
    LEGACY_MANIFEST_SCHEMA_VERSION,
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    FieldArtifactValidationError,
    SolverConfig,
    canonical_field_artifact_bytes,
    canonical_payload_sha256,
    contains_negative_zero,
    field_artifact,
    field_artifact_canonical_bytes,
    design_manifest,
    normalize_field_artifact_value,
    reload_field_artifact_bytes,
    solve_problem_cpu,
    validate_field_artifact,
    validate_design_manifest,
    write_field_artifact,
    write_design_manifest,
)
from cft_revival.fields.serialization import parse_field_json_bytes
from cft_revival.fields.verification import manufactured_values
from cft_revival.fields.numerics import solve_current_density_cpu


def _problem(name: str = "serialization") -> AxisymmetricProblem:
    domain = AxisymmetricDomain(0.1, -0.1, 0.1, 16, 32)
    return AxisymmetricProblem(
        name,
        domain,
        (AzimuthalCurrentBand("coil", 0.03, 0.05, -0.02, 0.02, 1_000),),
    )


def _artifact():
    problem = _problem()
    config = SolverConfig()
    return field_artifact(problem, config, solve_problem_cpu(problem, config))


def test_recursive_normalizer_canonicalizes_only_signed_zero() -> None:
    minimum_subnormal = nextafter(0.0, 1.0)
    value = {
        "diagnostics": {
            "history": [-0.0, {"negative": -0.0, "positive": 0.0}]
        },
        "finite_boundaries": [
            minimum_subnormal,
            -minimum_subnormal,
            sys.float_info.min,
            sys.float_info.max,
        ],
        "flags": [True, False],
    }
    normalized = normalize_field_artifact_value(value)
    assert not contains_negative_zero(normalized)
    assert copysign(1.0, normalized["diagnostics"]["history"][0]) > 0.0
    assert normalized["finite_boundaries"] == value["finite_boundaries"]
    assert normalized["flags"] == [True, False]
    assert canonical_payload_sha256({"x": -0.0}) == canonical_payload_sha256(
        {"x": 0.0}
    )
    data = canonical_field_artifact_bytes(normalized, representation="file")
    loaded = parse_field_json_bytes(
        data, source="finite-boundaries", require_canonical_file_bytes=True
    )
    assert loaded["finite_boundaries"] == value["finite_boundaries"]


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        -float("nan"),
        struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0],
        math.inf,
        -math.inf,
    ],
)
def test_programmatic_nonfinite_variants_are_rejected(value: float) -> None:
    with pytest.raises(FieldArtifactValidationError, match="finite"):
        canonical_field_artifact_bytes({"value": value}, representation="file")


@pytest.mark.parametrize("spelling", ["-0", "-0.0", "-0e0"])
def test_noncanonical_signed_zero_json_spellings_are_rejected(spelling: str) -> None:
    artifact = field_artifact(
        AxisymmetricProblem("zero", AxisymmetricDomain(0.1, -0.1, 0.1, 8, 16)),
        SolverConfig(),
        solve_problem_cpu(
            AxisymmetricProblem(
                "zero", AxisymmetricDomain(0.1, -0.1, 0.1, 8, 16)
            )
        ),
    )
    data = field_artifact_canonical_bytes(artifact)
    assert not contains_negative_zero(artifact)
    tampered = data.replace(b": 0.0", f": {spelling}".encode("ascii"), 1)
    assert tampered != data
    with pytest.raises(FieldArtifactValidationError, match="not canonical"):
        reload_field_artifact_bytes(tampered)


@pytest.mark.parametrize(
    "token",
    ["NaN", "-NaN", "Infinity", "-Infinity", "1e309", "-1e309"],
)
def test_nonfinite_json_tokens_and_exponent_overflow_are_rejected(token: str) -> None:
    data = (
        '{"schema_version":"'
        + ARTIFACT_SCHEMA_VERSION
        + '","value":'
        + token
        + "}\n"
    ).encode("ascii")
    with pytest.raises(FieldArtifactValidationError):
        reload_field_artifact_bytes(data)


def test_bool_in_numeric_field_is_rejected_before_persistence(tmp_path) -> None:
    artifact = _artifact()
    artifact["summary"]["b_magnitude_max_t"] = True
    payload = {key: value for key, value in artifact.items() if key != "integrity"}
    artifact["integrity"]["payload_sha256"] = canonical_payload_sha256(payload)
    with pytest.raises(FieldArtifactValidationError, match="must be a number"):
        write_field_artifact(tmp_path / "bad.json", artifact)
    assert not (tmp_path / "bad.json").exists()


def test_solve_artifact_canonical_bytes_reload_validate_roundtrip(tmp_path) -> None:
    artifact = _artifact()
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert not contains_negative_zero(artifact)
    data = field_artifact_canonical_bytes(artifact)
    reloaded = reload_field_artifact_bytes(data)
    validate_field_artifact(reloaded)
    assert reloaded == artifact
    path = tmp_path / "field.json"
    digest = write_field_artifact(path, artifact)
    assert path.read_bytes() == data
    assert hashlib.sha256(data).hexdigest() == digest


def test_manufactured_field_full_canonical_roundtrip_preserves_finite_extremes() -> None:
    domain = AxisymmetricDomain(0.12, -0.15, 0.15, 16, 32)
    _, source, _, _ = manufactured_values(domain)
    field = solve_current_density_cpu(
        domain,
        source,
        permeability_h_per_m=1.2566370614359173e-6,
    )
    payload = {"kind": "manufactured-field-map", "field": asdict(field)}
    data = canonical_field_artifact_bytes(payload, representation="file")
    loaded = parse_field_json_bytes(
        data,
        source="manufactured-field-map",
        require_canonical_file_bytes=True,
    )
    assert loaded == normalize_field_artifact_value(payload)
    assert not contains_negative_zero(loaded)


def test_legacy_v11_signed_zero_is_read_only_and_hashes_old_bytes(tmp_path) -> None:
    legacy = _artifact()
    legacy["schema_version"] = LEGACY_ARTIFACT_SCHEMA_VERSION
    legacy["field_map"]["b_r_t"][0][0] = -0.0
    legacy["integrity"]["canonicalization"] = "json-sort-keys-compact-utf8-v1"
    payload = {key: value for key, value in legacy.items() if key != "integrity"}
    old_payload_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    legacy["integrity"]["payload_sha256"] = hashlib.sha256(
        old_payload_bytes
    ).hexdigest()
    old_file_bytes = (
        json.dumps(
            legacy,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    loaded = reload_field_artifact_bytes(old_file_bytes)
    assert contains_negative_zero(loaded)
    with pytest.raises(FieldArtifactValidationError, match="disabled"):
        reload_field_artifact_bytes(old_file_bytes, allow_legacy_v1_1=False)
    with pytest.raises(FieldArtifactValidationError, match="read-only"):
        write_field_artifact(tmp_path / "legacy.json", loaded)

    current_manifest = design_manifest(
        [
            {
                "name": legacy["input"]["name"],
                "artifact": "legacy.json",
                "artifact_file_sha256": hashlib.sha256(old_file_bytes).hexdigest(),
                "artifact_payload_sha256": legacy["integrity"]["payload_sha256"],
                "backend": legacy["diagnostics"]["backend"],
                "iterations": legacy["diagnostics"]["iterations"],
                "relative_residual_l2": legacy["diagnostics"][
                    "relative_residual_l2"
                ],
                "b_magnitude_min_t": legacy["summary"]["b_magnitude_min_t"],
                "b_magnitude_max_t": legacy["summary"]["b_magnitude_max_t"],
                "topology": legacy["summary"]["topology"],
            }
        ]
    )
    current_manifest["schema_version"] = LEGACY_MANIFEST_SCHEMA_VERSION
    current_manifest["integrity"][
        "canonicalization"
    ] = "json-sort-keys-compact-utf8-v1"
    manifest_payload = {
        key: value for key, value in current_manifest.items() if key != "integrity"
    }
    current_manifest["integrity"]["payload_sha256"] = hashlib.sha256(
        json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    validate_design_manifest(current_manifest)
    with pytest.raises(FieldArtifactValidationError, match="read-only"):
        write_design_manifest(tmp_path / "legacy-manifest.json", current_manifest)

