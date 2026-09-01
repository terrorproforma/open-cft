from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from cft_revival.coupling import (
    MapValidationPolicy,
    ProfileRole,
    TopologyStatus,
    TopologyPolicy,
    UncertaintyModel,
    build_screening_proxy,
    source_map_binding_hash,
)
from cft_revival.coupling.records import (
    COUPLING_SCHEMA_VERSION,
    coupling_record_dict,
    global_solver_inputs,
)
from tests.coupling.evidence_helpers import (
    AnalyticMap,
    accepted_evidence,
    claims_for,
    two_cusp_map,
)

build_coupling_record = build_screening_proxy
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*deprecated screening_proxy.*:DeprecationWarning"
)


def test_verified_record_uses_topology_midplanes_and_complete_identity() -> None:
    record = build_coupling_record(
        accepted_evidence(),
        wall_radius_m=0.75,
        uncertainty_model=UncertaintyModel(absolute_independent_sigma_t=1.0e-3),
    )
    assert record.schema_version == COUPLING_SCHEMA_VERSION
    assert record.topology_status is TopologyStatus.RESOLVED
    assert [segment.representative_cusp_z_m for segment in record.segments] == [
        -1.0,
        1.0,
    ]
    assert [
        (segment.z_start_m, segment.z_end_m) for segment in record.segments
    ] == [(-2.0, 0.0), (0.0, 2.0)]
    assert len(record.record_hash) == 64
    assert record.artifact_schema_version == "cft-axisymmetric-field-map/1.1.0"
    assert record.model_level == "L1a"
    assert record.backend_id == "python-reference"
    assert record.diagnostics.converged


def test_solver_projection_retains_hashes_diagnostics_and_probability_interval() -> None:
    record = build_coupling_record(
        accepted_evidence(),
        wall_radius_m=0.75,
        uncertainty_model=UncertaintyModel(
            relative_independent_sigma=0.01,
            common_mode_sigma_t=0.002,
            residual_correlation=0.25,
        ),
    )
    rows = global_solver_inputs(record)
    assert len(rows) == 2
    row = rows[0]
    for key in (
        "record_hash",
        "field_map_hash",
        "artifact_hash",
        "source_hash",
        "source_map_binding_hash",
        "field_model_hash",
        "code_hash",
        "config_hash",
        "adapter_code_hash",
        "coupling_model_hash",
    ):
        assert len(str(row[key])) == 64
    assert row["artifact_schema_version"] == "cft-axisymmetric-field-map/1.1.0"
    assert row["model_level"] == "L1a"
    assert row["residual_norm"] <= row["residual_tolerance"]
    assert row["loss_cone_probability_lower"] <= row["loss_cone_probability_upper"]
    assert row["input_covariance_t2"] is not None
    payload = coupling_record_dict(record)
    assert json.loads(json.dumps(payload, allow_nan=False))["record_hash"] == (
        record.record_hash
    )


def test_record_and_model_hashes_change_with_radius_role_and_uncertainty() -> None:
    base = build_coupling_record(accepted_evidence(), wall_radius_m=0.75)
    other_wall = build_coupling_record(accepted_evidence(), wall_radius_m=0.8)
    uncertain = build_coupling_record(
        accepted_evidence(),
        wall_radius_m=0.75,
        uncertainty_model=UncertaintyModel(relative_independent_sigma=0.01),
    )
    inner_evidence = accepted_evidence(
        two_cusp_map(inner_radius_m=0.1),
        policy=MapValidationPolicy(require_axis=False),
    )
    inner = build_coupling_record(inner_evidence, wall_radius_m=0.75)
    assert len(
        {
            base.record_hash,
            other_wall.record_hash,
            uncertain.record_hash,
            inner.record_hash,
        }
    ) == 4
    assert len(
        {
            base.coupling_model_hash,
            other_wall.coupling_model_hash,
            uncertain.coupling_model_hash,
            inner.coupling_model_hash,
        }
    ) == 4
    assert inner.inner_profile_role is ProfileRole.INNER_RADIAL_PROFILE
    assert inner.inner_profile.name == "inner_radial_profile"
    assert inner.inner_profile.sampled_r_m == pytest.approx(0.1)


def test_record_hash_covers_artifact_source_backend_diagnostics_and_time() -> None:
    field = two_cusp_map()
    baseline_evidence = accepted_evidence(field)
    baseline = build_coupling_record(baseline_evidence, wall_radius_m=0.75)
    other_artifact = build_coupling_record(
        accepted_evidence(field, artifact_bytes=b'{"accepted":"other"}'),
        wall_radius_m=0.75,
    )
    claims = claims_for(field)
    source_hash = "5" * 64
    source_claims = replace(
        claims,
        source_hash=source_hash,
        source_map_binding_hash=source_map_binding_hash(
            claims.map_content_hash, source_hash, claims.artifact_hash
        ),
    )
    changed_source = build_coupling_record(
        accepted_evidence(field, claims=source_claims), wall_radius_m=0.75
    )
    changed_backend = build_coupling_record(
        accepted_evidence(field, claims=replace(claims, backend_id="cpu-v2")),
        wall_radius_m=0.75,
    )
    changed_diagnostics = build_coupling_record(
        accepted_evidence(
            field,
            claims=replace(
                claims,
                diagnostics=replace(claims.diagnostics, residual_norm=2.0e-12),
            ),
        ),
        wall_radius_m=0.75,
    )
    changed_time = build_coupling_record(
        accepted_evidence(
            field,
            claims=replace(
                claims, generated_at_utc=claims.generated_at_utc - timedelta(seconds=1)
            ),
        ),
        wall_radius_m=0.75,
    )
    changed_freshness = build_coupling_record(
        accepted_evidence(field, policy=MapValidationPolicy(maximum_age_s=60.0)),
        wall_radius_m=0.75,
        reference_time_utc=claims.generated_at_utc,
    )
    assert len(
        {
            baseline.record_hash,
            other_artifact.record_hash,
            changed_source.record_hash,
            changed_backend.record_hash,
            changed_diagnostics.record_hash,
            changed_time.record_hash,
            changed_freshness.record_hash,
        }
    ) == 7


def test_resampling_changes_map_and_record_identity_not_field_model_identity() -> None:
    coarse = build_coupling_record(
        accepted_evidence(two_cusp_map(17)), wall_radius_m=0.75
    )
    fine = build_coupling_record(
        accepted_evidence(two_cusp_map(33)), wall_radius_m=0.75
    )
    assert coarse.field_map_hash != fine.field_map_hash
    assert coarse.record_hash != fine.record_hash
    assert coarse.field_model_hash == fine.field_model_hash


@pytest.mark.parametrize(
    ("axis_bz", "expected_status"),
    [
        ((1.0,) * 7, TopologyStatus.DEGENERATE),
        ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), TopologyStatus.NO_TOPOLOGY),
    ],
)
def test_degenerate_and_monotonic_profiles_do_not_create_segments(
    axis_bz: tuple[float, ...], expected_status: TopologyStatus
) -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    field = AnalyticMap(
        (0.0, 0.5, 1.0),
        z,
        ((0.0,) * 7, (1.0,) * 7, (2.0,) * 7),
        (axis_bz, axis_bz, axis_bz),
    )
    record = build_coupling_record(
        accepted_evidence(field), wall_radius_m=0.75
    )
    assert record.topology_status is expected_status
    assert record.segments == ()
    assert global_solver_inputs(record) == ()


def test_uncertainty_gated_ripple_returns_ambiguous_record() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    ripple = (1.0, 0.99, 1.0, 0.99, 1.0, 0.99, 1.0)
    field = AnalyticMap(
        (0.0, 0.5, 1.0),
        z,
        ((0.0,) * 7, (1.0,) * 7, (2.0,) * 7),
        (ripple, ripple, ripple),
    )
    record = build_coupling_record(
        accepted_evidence(field),
        wall_radius_m=0.75,
        uncertainty_model=UncertaintyModel(absolute_independent_sigma_t=0.02),
    )
    assert record.topology_status is TopologyStatus.AMBIGUOUS
    assert record.segments == ()
    assert record.alternative_candidates


def test_boundary_minimum_requires_explicit_opt_in() -> None:
    z = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    descending = (7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    field = AnalyticMap(
        (0.0, 0.5, 1.0),
        z,
        ((0.0,) * 7, (1.0,) * 7, (2.0,) * 7),
        (descending, descending, descending),
    )
    evidence = accepted_evidence(field)
    default = build_coupling_record(evidence, wall_radius_m=0.75)
    allowed = build_coupling_record(
        evidence,
        wall_radius_m=0.75,
        topology_policy=TopologyPolicy(allow_boundary_minima_as_cusps=True),
    )
    assert default.topology_status is TopologyStatus.NO_TOPOLOGY
    assert default.segments == ()
    assert allowed.topology_status is TopologyStatus.RESOLVED
    assert len(allowed.segments) == 1
