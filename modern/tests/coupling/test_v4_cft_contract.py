from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import timedelta
from math import erf, exp, pi, sin, sqrt

import pytest

from cft_revival.coupling.models import (
    AdapterVersionContract,
    CouplingValidationError,
    EvidenceVerificationError,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    UncertaintyModel,
)
from cft_revival.coupling.v3_evidence import (
    hash_psi_map,
    v3_evidence_binding_hash,
    verify_v3_field_artifact,
)
from cft_revival.coupling.v3_models import V3ArtifactClaims
from cft_revival.coupling.v4_evidence import verify_v4_map_set
from cft_revival.coupling.v4_models import (
    AxialDominancePolicy,
    CFT_V4_DEVELOPMENT_MANIFEST,
    CFTCellRegistration,
    CFTGeometry,
    CFTStabilityPolicy,
    ElectronOrbitSample,
    FieldLineSeed,
    FieldLineTracePolicy,
    HeldOutCaseRegistration,
    HeldOutCaseOutcome,
    HeldOutValidationClaims,
    HeldOutValidationPolicy,
    HeldOutValidationRegistration,
    OrbitVerificationClaims,
    V4Criterion,
    V4Status,
    WallCuspPolicy,
    ValidationSetManifest,
    validation_set_manifest_hash,
)
from cft_revival.coupling.v4_records import (
    accept_cft_projection,
    build_cft_coupling_record,
    cft_coupling_record_dict,
    cft_preregistration_hash,
    cft_solver_inputs,
)
from cft_revival.coupling.v4_validation import verify_held_out_validation
from tests.coupling.evidence_helpers import NOW


@dataclass(frozen=True)
class PsiMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    psi_wb: tuple[tuple[float, ...], ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


def hemp_map(
    radial_points: int,
    axial_points: int,
    *,
    radius_m: float = 1.3,
    axial_half_width_m: float = 2.5,
    cusp_shift_m: float = 0.0,
    ripple_amplitude_t: float = 0.0,
    extra_wall_peak_t: float = 0.0,
) -> PsiMap:
    radii = tuple(radius_m * index / (radial_points - 1) for index in range(radial_points))
    axial = tuple(
        -axial_half_width_m
        + 2.0 * axial_half_width_m * index / (axial_points - 1)
        for index in range(axial_points)
    )
    amplitude, width, b0 = 1.5, 0.25, 0.5

    def sample(radius: float, z: float) -> tuple[float, float, float]:
        left, right = -1.0 + cusp_shift_m, 1.0 + cusp_shift_m
        h = amplitude * (
            exp(-((z - right) / width) ** 2)
            - exp(-((z - left) / width) ** 2)
        )
        h += ripple_amplitude_t * sin(60.0 * z)
        h += extra_wall_peak_t * exp(-(z / 0.08) ** 2)
        integral = amplitude * width * sqrt(pi) * 0.5 * (
            erf((z - right) / width) - erf((z - left) / width)
        )
        psi = radius * radius * (0.5 * b0 - integral)
        return psi, radius * h, b0 - 2.0 * integral

    rows = tuple(tuple(sample(radius, z) for z in axial) for radius in radii)
    return PsiMap(
        radii,
        axial,
        tuple(tuple(value[0] for value in row) for row in rows),
        tuple(tuple(value[1] for value in row) for row in rows),
        tuple(tuple(value[2] for value in row) for row in rows),
    )


class Adapter:
    adapter_id = "tests.v4.psi-adapter"
    adapter_code_hash = "a" * 64
    version_contract = AdapterVersionContract(
        "cft-v4-psi-direct",
        "1.0.0",
        "cft-axisymmetric-field-map/1.1.0",
        "cft-axisymmetric-field-map/1.1.0",
        "L1a",
    )

    def __init__(self, claims: V3ArtifactClaims) -> None:
        self.claims = claims

    def verify_v3_artifact(self, artifact_bytes: bytes) -> V3ArtifactClaims:
        return self.claims


def accepted(
    field: PsiMap,
    role: str,
    *,
    policy: MapValidationPolicy = MapValidationPolicy(),
    **claim_changes,
):
    artifact = f'{{"accepted":"hemp-v4","role":"{role}"}}'.encode()
    artifact_hash = hashlib.sha256(artifact).hexdigest()
    map_hash = hash_psi_map(field)
    mesh_hash = hashlib.sha256(f"mesh-{role}".encode()).hexdigest()
    domain_hash = hashlib.sha256(f"domain-{role}".encode()).hexdigest()
    claims = V3ArtifactClaims(
        field, "cft-axisymmetric-field-map/1.1.0", "L1a",
        artifact_hash, map_hash, "1" * 64, "2" * 64, "3" * 64,
        mesh_hash, domain_hash,
        v3_evidence_binding_hash(
            map_hash, "1" * 64, "2" * 64, "3" * 64,
            mesh_hash, domain_hash, artifact_hash,
        ),
        "manufactured-python", "1.0", "source-backed-hemp-wall-cusp",
        "4" * 64, "5" * 64, "6" * 64, NOW,
        SolverDiagnosticsEvidence(True, 1e-12, 1e-10, 1e-12, 1e-10, 20),
    )
    claims = replace(claims, **claim_changes)
    return verify_v3_field_artifact(
        artifact,
        Adapter(claims),
        policy,
        reference_time_utc=NOW,
    )


def map_set(
    *,
    enlarged_shift: float = 0.0,
    policy: MapValidationPolicy = MapValidationPolicy(),
    **claim_changes,
):
    return verify_v4_map_set(
        accepted(
            hemp_map(21, 41),
            "primary",
            policy=policy,
            **claim_changes,
        ),
        accepted(
            hemp_map(41, 81),
            "refined",
            policy=policy,
            **claim_changes,
        ),
        accepted(
            hemp_map(
                31, 61, radius_m=1.5, axial_half_width_m=3.0,
                cusp_shift_m=enlarged_shift,
            ),
            "enlarged",
            policy=policy,
            **claim_changes,
        ),
        reference_time_utc=NOW,
    )


class OrbitAdapter:
    adapter_id = "tests.converged-full-orbit"
    adapter_version = "2.1.0"
    adapter_code_hash = "b" * 64
    orbit_model_id = "manufactured-lorentz-orbit"
    orbit_model_version = "1.2.0"
    orbit_code_hash = "c" * 64
    orbit_config_hash = "d" * 64
    convergence_id = "mu-relative-window"
    convergence_version = "1.0.0"
    convergence_config_hash = "e" * 64

    def __init__(self, variation: float = 0.001) -> None:
        self.variation = variation

    def verify_orbit(self, path_points_rz_m, path_hash, sample):
        return OrbitVerificationClaims(
            path_hash=path_hash,
            sample_id=sample.sample_id,
            converged=True,
            maximum_mu_relative_variation=self.variation,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_code_hash=self.adapter_code_hash,
            orbit_model_id=self.orbit_model_id,
            orbit_model_version=self.orbit_model_version,
            orbit_code_hash=self.orbit_code_hash,
            orbit_config_hash=self.orbit_config_hash,
            convergence_id=self.convergence_id,
            convergence_version=self.convergence_version,
            convergence_config_hash=self.convergence_config_hash,
        )


class HeldOutAdapter:
    adapter_id = "tests.held-out-family-adapter"
    adapter_code_hash = "d" * 64

    def __init__(self, claims: HeldOutValidationClaims) -> None:
        self.claims = claims

    def verify_validation_artifact(
        self, artifact_bytes: bytes
    ) -> HeldOutValidationClaims:
        return self.claims


GEOMETRY = CFTGeometry(1.0, -2.0, 2.0, 0.6, "manufactured-cylinder")
HELD_OUT_CASE_ID = "held-out-hemp-case-001"
HELD_OUT_FAMILY_ID = "held-out-hemp-family-b"
HELD_OUT_MANIFEST = ValidationSetManifest(
    manifest_id="held-out-hemp-validation-v1",
    case_ids=(HELD_OUT_CASE_ID,),
    geometry_family_ids=(HELD_OUT_FAMILY_ID,),
    manifest_hash=validation_set_manifest_hash(
        "held-out-hemp-validation-v1",
        (HELD_OUT_CASE_ID,),
        (HELD_OUT_FAMILY_ID,),
    ),
)
VALIDATION_REGISTRATION = HeldOutValidationRegistration(
    development_manifest=CFT_V4_DEVELOPMENT_MANIFEST,
    held_out_manifest=HELD_OUT_MANIFEST,
    evaluated_case_id=HELD_OUT_CASE_ID,
    evaluated_geometry_family_id=HELD_OUT_FAMILY_ID,
    required_case_count=1,
    required_outcomes=(
        HeldOutCaseRegistration(
            HELD_OUT_CASE_ID,
            HELD_OUT_FAMILY_ID,
        ),
    ),
    validation_adapter_id=HeldOutAdapter.adapter_id,
    validation_adapter_code_hash=HeldOutAdapter.adapter_code_hash,
    validation_code_hash="8" * 64,
    validation_config_hash="9" * 64,
)
REGISTRATIONS = (
    CFTCellRegistration(
        "inter-cusp-1",
        (
            FieldLineSeed(
                "seed-q50", 0.9, 0.0,
                (ElectronOrbitSample("100eV-45deg", 100.0, pi / 4.0),),
            ),
        ),
    ),
)


def build(**changes):
    arguments = {
        "evidence": map_set(),
        "geometry": GEOMETRY,
        "registrations": REGISTRATIONS,
        "validation_registration": VALIDATION_REGISTRATION,
        "orbit_adapter": OrbitAdapter(),
        "cusp_policy": WallCuspPolicy(
            minimum_prominence_t=0.02,
            endpoint_plane_tolerance_m=0.3,
        ),
        "trace_policy": FieldLineTracePolicy(
            step_m=0.005,
            maximum_psi_drift_wb=0.01,
        ),
        "stability_policy": CFTStabilityPolicy(
            maximum_cusp_shift_m=0.08,
            maximum_cusp_strength_relative_change=0.1,
            maximum_endpoint_shift_m=0.08,
            maximum_cell_bound_shift_m=0.08,
            maximum_axial_metric_change=0.05,
        ),
        "reference_time_utc": NOW,
    }
    arguments.update(changes)
    return build_cft_coupling_record(**arguments)


def canonically_rehash(record):
    unsigned = replace(record, record_hash="")
    encoded = json.dumps(
        cft_coupling_record_dict(unsigned),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return replace(
        record,
        record_hash=hashlib.sha256(
            b"cft-coupling-record-v4\0" + encoded
        ).hexdigest(),
    )


def held_out_evidence(
    development_record,
    *,
    artifact: bytes = b'{"held_out":"new-hemp-family","version":1}',
    **claim_changes,
):
    preregistration_hash = cft_preregistration_hash(
        geometry=development_record.geometry,
        registrations=development_record.registrations,
        validation_registration=development_record.validation_registration,
        three_map_hashes=(
            development_record.stability.primary.identity.full_map_hash,
            development_record.stability.refined.identity.full_map_hash,
            development_record.stability.enlarged.identity.full_map_hash,
        ),
        three_map_evidence_fingerprints=(
            development_record.evidence_fingerprints
        ),
        orbit_identity=development_record.orbit_identity,
        cusp_policy=development_record.cusp_policy,
        trace_policy=development_record.trace_policy,
        axial_policy=development_record.axial_policy,
        stability_policy=development_record.stability_policy,
        uncertainty_model=development_record.uncertainty_model,
        criterion=development_record.criterion,
    )
    claims = HeldOutValidationClaims(
        criterion_id=development_record.criterion.criterion_id,
        criterion_version=development_record.criterion.criterion_version,
        development_manifest=(
            development_record.validation_registration.development_manifest
        ),
        held_out_manifest=(
            development_record.validation_registration.held_out_manifest
        ),
        evaluated_case_id=(
            development_record.validation_registration.evaluated_case_id
        ),
        evaluated_geometry_family_id=(
            development_record.validation_registration.evaluated_geometry_family_id
        ),
        outcomes=(
            HeldOutCaseOutcome(
                case_id=(
                    development_record.validation_registration.evaluated_case_id
                ),
                geometry_family_id=(
                    development_record.validation_registration.evaluated_geometry_family_id
                ),
                three_map_hashes=(
                    development_record.stability.primary.identity.full_map_hash,
                    development_record.stability.refined.identity.full_map_hash,
                    development_record.stability.enlarged.identity.full_map_hash,
                ),
                three_map_evidence_fingerprints=(
                    development_record.evidence_fingerprints
                ),
                passed=True,
            ),
        ),
        preregistration_hash=preregistration_hash,
        validation_artifact_hash=hashlib.sha256(artifact).hexdigest(),
        validation_code_hash="8" * 64,
        validation_config_hash="9" * 64,
        generated_at_utc=NOW,
        diagnostics=SolverDiagnosticsEvidence(
            True, 1e-12, 1e-10, 1e-12, 1e-10, 20
        ),
    )
    claims = replace(claims, **claim_changes)
    return verify_held_out_validation(
        artifact,
        HeldOutAdapter(claims),
        reference_time_utc=NOW,
        policy=development_record.validation_registration.policy,
    )


def test_wall_cusps_define_inter_cusp_cell_without_o_point_requirement() -> None:
    assert len(CFT_V4_DEVELOPMENT_MANIFEST.case_ids) == 56
    assert CFT_V4_DEVELOPMENT_MANIFEST.manifest_hash == (
        validation_set_manifest_hash(
            CFT_V4_DEVELOPMENT_MANIFEST.manifest_id,
            CFT_V4_DEVELOPMENT_MANIFEST.case_ids,
            CFT_V4_DEVELOPMENT_MANIFEST.geometry_family_ids,
        )
    )
    record = build()
    assert record.schema_version.endswith("/4.1.0")
    assert record.status is V4Status.RESOLVED
    assert len(record.stability.primary.cusps) == 2
    assert all(
        cusp.stable and cusp.bundle_endpoint_count >= 1
        for cusp in record.stability.primary.cusps
    )
    assert all(
        cusp.radial_fraction >= record.cusp_policy.minimum_wall_radial_fraction
        for cusp in record.stability.primary.cusps
    )
    cell = record.stability.primary.cells[0]
    assert cell.z_start_m < 0.0 < cell.z_end_m
    assert cell.axial_metrics.passed
    assert not cell.closed_islands
    outcome = cell.seed_outcomes[0]
    assert outcome.status is V4Status.RESOLVED
    assert {
        outcome.negative_path.termination,
        outcome.positive_path.termination,
    } == {"channel_wall"}
    assert all(
        path.wall_endpoint_rz_m[0] == pytest.approx(GEOMETRY.channel_wall_radius_m)
        for path in (outcome.negative_path, outcome.positive_path)
    )


def test_orbit_failure_atomically_suppresses_cell_and_probability() -> None:
    record = build(orbit_adapter=OrbitAdapter(variation=0.5))
    assert record.status is V4Status.AMBIGUOUS
    paths = record.stability.primary.cells[0].seed_outcomes[0]
    assert paths.negative_path.status is V4Status.NONADIABATIC
    assert paths.negative_path.mirror_probability is None


def test_map_matching_rejects_shifted_wall_cusp_family() -> None:
    record = build(evidence=map_set(enlarged_shift=0.1))
    assert record.status is V4Status.AMBIGUOUS
    assert not record.stability.passed


def test_plasma_wall_must_be_interior_to_computational_domain() -> None:
    with pytest.raises(CouplingValidationError, match="extend beyond"):
        build(geometry=replace(GEOMETRY, channel_wall_radius_m=1.5))


def test_negative_policy_cannot_hide_by_matching_axial_coordinate() -> None:
    with pytest.raises(CouplingValidationError, match="non-negative"):
        build(
            cusp_policy=WallCuspPolicy(
                minimum_prominence_t=GEOMETRY.plasma_z_min_m
            )
        )


def test_coverage_factor_widens_path_bounds_and_changes_identity() -> None:
    narrow = build(
        uncertainty_model=UncertaintyModel(
            absolute_independent_sigma_t=1e-4, coverage_factor=1.0
        )
    )
    wide = build(
        uncertainty_model=UncertaintyModel(
            absolute_independent_sigma_t=1e-4, coverage_factor=10.0
        )
    )
    narrow_path = narrow.stability.primary.cells[0].seed_outcomes[0].positive_path
    wide_path = wide.stability.primary.cells[0].seed_outcomes[0].positive_path
    assert wide_path.interpolation_error_t > narrow_path.interpolation_error_t
    assert wide.record_hash != narrow.record_hash


def test_solver_projection_waits_for_new_held_out_geometry_family() -> None:
    development = build()
    assert development.criterion.development_evidence_role == "development_non_validation"
    assert cft_solver_inputs(development, reference_time_utc=NOW) == ()
    evidence = held_out_evidence(development)
    validated = build(
        held_out_validation_evidence=evidence,
    )
    accepted_projection = accept_cft_projection(
        validated,
        map_set(),
        held_out_validation_evidence=evidence,
        orbit_adapter=OrbitAdapter(),
        reference_time_utc=NOW,
    )
    rows = cft_solver_inputs(
        accepted_projection,
        reference_time_utc=NOW,
    )
    assert len(rows) == 2
    assert {row["direction"] for row in rows} == {-1, 1}
    assert all(row["mirror_probability"] is not None for row in rows)
    assert all(len(row["path_hash"]) == 64 for row in rows)
    assert all(row["path_status"] == "resolved" for row in rows)
    assert all(row["path_termination"] == "channel_wall" for row in rows)
    assert all(
        row["primary_evidence_fingerprint"]
        == validated.evidence_fingerprints[0]
        for row in rows
    )
    assert all(row["record_hash"] == validated.record_hash for row in rows)
    assert all(
        row["held_out_validation_config_hash"] == "9" * 64
        for row in rows
    )
    assert all(
        row["held_out_preregistration_hash"]
        == validated.held_out_validation.preregistration_hash
        for row in rows
    )
    assert validated.held_out_validation is not None
    assert not (
        set(validated.held_out_validation.development_manifest.case_ids)
        & set(validated.held_out_validation.held_out_manifest.case_ids)
    )
    assert validated.criterion.held_out_validation_status == (
        "validated_new_geometry_family"
    )


def test_validation_status_cannot_be_self_declared_or_preregistration_swapped() -> None:
    with pytest.raises(CouplingValidationError, match="evidence-derived"):
        build(
            criterion=replace(
                V4Criterion(),
                held_out_validation_status="validated_new_geometry_family",
            )
        )
    development = build()
    mismatched = held_out_evidence(
        development,
        preregistration_hash="9" * 64,
    )
    with pytest.raises(EvidenceVerificationError, match="preregistration"):
        build(held_out_validation_evidence=mismatched)
    unregistered_config = held_out_evidence(
        development,
        validation_config_hash="a" * 64,
    )
    with pytest.raises(EvidenceVerificationError, match="preregistration"):
        build(held_out_validation_evidence=unregistered_config)


def test_held_out_evidence_recomputes_disjointness_and_complete_outcomes() -> None:
    development = build()
    overlapping = replace(
        HELD_OUT_MANIFEST,
        case_ids=(
            CFT_V4_DEVELOPMENT_MANIFEST.case_ids[0],
        ),
        manifest_hash=validation_set_manifest_hash(
            HELD_OUT_MANIFEST.manifest_id,
            (CFT_V4_DEVELOPMENT_MANIFEST.case_ids[0],),
            HELD_OUT_MANIFEST.geometry_family_ids,
        ),
    )
    with pytest.raises(EvidenceVerificationError, match="disjoint"):
        held_out_evidence(development, held_out_manifest=overlapping)
    with pytest.raises(EvidenceVerificationError, match="manifest hash"):
        held_out_evidence(
            development,
            held_out_manifest=replace(
                HELD_OUT_MANIFEST,
                manifest_hash="0" * 64,
            ),
        )
    with pytest.raises(EvidenceVerificationError, match="complete manifest"):
        held_out_evidence(development, outcomes=())
    failed = HeldOutCaseOutcome(
        HELD_OUT_CASE_ID,
        HELD_OUT_FAMILY_ID,
        (
            development.stability.primary.identity.full_map_hash,
            development.stability.refined.identity.full_map_hash,
            development.stability.enlarged.identity.full_map_hash,
        ),
        development.evidence_fingerprints,
        False,
    )
    with pytest.raises(EvidenceVerificationError, match="must pass"):
        held_out_evidence(development, outcomes=(failed,))


def test_stale_held_out_evidence_is_reverified_at_build_time() -> None:
    registration = replace(
        VALIDATION_REGISTRATION,
        policy=HeldOutValidationPolicy(
            maximum_age_s=3600.0,
            maximum_future_skew_s=1.0,
        ),
    )
    development = build(validation_registration=registration)
    evidence = held_out_evidence(development)
    with pytest.raises(EvidenceVerificationError, match="stale"):
        build(
            validation_registration=registration,
            held_out_validation_evidence=evidence,
            reference_time_utc=NOW + timedelta(hours=2),
        )


def test_held_out_future_skew_is_preregistered_and_enforced() -> None:
    development = build()
    with pytest.raises(EvidenceVerificationError, match="future"):
        held_out_evidence(
            development,
            generated_at_utc=NOW + timedelta(seconds=2),
        )


def test_record_projection_reverifies_hash_and_all_atomic_gates() -> None:
    development = build()
    evidence = held_out_evidence(development)
    validated = build(held_out_validation_evidence=evidence)
    accepted_projection = accept_cft_projection(
        validated,
        map_set(),
        held_out_validation_evidence=evidence,
        orbit_adapter=OrbitAdapter(),
        reference_time_utc=NOW,
    )
    assert cft_solver_inputs(accepted_projection, reference_time_utc=NOW)
    with pytest.raises(EvidenceVerificationError, match="reproducible"):
        accept_cft_projection(
            replace(validated, record_hash="0" * 64),
            map_set(),
            held_out_validation_evidence=evidence,
            orbit_adapter=OrbitAdapter(),
            reference_time_utc=NOW,
        )
    with pytest.raises(EvidenceVerificationError, match="reproducible"):
        accept_cft_projection(
            replace(validated, stability=replace(validated.stability, passed=False)),
            map_set(),
            held_out_validation_evidence=evidence,
            orbit_adapter=OrbitAdapter(),
            reference_time_utc=NOW,
        )


def test_canonical_rehash_cannot_authorize_invalid_path_or_missing_probability() -> None:
    development = build()
    evidence = held_out_evidence(development)
    validated = build(held_out_validation_evidence=evidence)
    primary = validated.stability.primary
    cell = primary.cells[0]
    outcome = cell.seed_outcomes[0]
    invalid_path = replace(
        outcome.positive_path,
        status=V4Status.INVALID,
        reason="caller changed path",
        mirror_probability=None,
    )
    forged_outcome = replace(
        outcome,
        positive_path=invalid_path,
        status=V4Status.INVALID,
        reason="caller changed path",
    )
    forged_cell = replace(
        cell,
        seed_outcomes=(forged_outcome,),
        status=V4Status.INVALID,
        reason="caller changed path",
    )
    forged_primary = replace(primary, cells=(forged_cell,))
    forged = canonically_rehash(
        replace(
            validated,
            stability=replace(
                validated.stability,
                primary=forged_primary,
            ),
        )
    )
    assert forged.record_hash != validated.record_hash
    with pytest.raises(EvidenceVerificationError, match="reproducible"):
        accept_cft_projection(
            forged,
            map_set(),
            held_out_validation_evidence=evidence,
            orbit_adapter=OrbitAdapter(),
            reference_time_utc=NOW,
        )


def test_canonical_rehash_cannot_authorize_nonconverged_map_diagnostics() -> None:
    development = build()
    evidence = held_out_evidence(development)
    validated = build(held_out_validation_evidence=evidence)
    primary = validated.stability.primary
    forged_identity = replace(
        primary.identity,
        diagnostics=replace(
            primary.identity.diagnostics,
            converged=False,
        ),
    )
    forged = canonically_rehash(
        replace(
            validated,
            stability=replace(
                validated.stability,
                primary=replace(primary, identity=forged_identity),
            ),
        )
    )
    with pytest.raises(EvidenceVerificationError, match="reproducible"):
        accept_cft_projection(
            forged,
            map_set(),
            held_out_validation_evidence=evidence,
            orbit_adapter=OrbitAdapter(),
            reference_time_utc=NOW,
        )


@pytest.mark.parametrize(
    "identity_changes",
    (
        {"code_hash": "7" * 64},
        {"generated_at_utc": NOW - timedelta(seconds=1)},
    ),
)
def test_canonical_rehash_cannot_authorize_provenance_or_timestamp_changes(
    identity_changes,
) -> None:
    development = build()
    evidence = held_out_evidence(development)
    validated = build(held_out_validation_evidence=evidence)
    primary = validated.stability.primary
    forged = canonically_rehash(
        replace(
            validated,
            stability=replace(
                validated.stability,
                primary=replace(
                    primary,
                    identity=replace(primary.identity, **identity_changes),
                ),
            ),
        )
    )
    with pytest.raises(EvidenceVerificationError, match="reproducible"):
        accept_cft_projection(
            forged,
            map_set(),
            held_out_validation_evidence=evidence,
            orbit_adapter=OrbitAdapter(),
            reference_time_utc=NOW,
        )


def test_projection_rechecks_stale_maps_with_fresh_held_out_wrapper() -> None:
    short_map_policy = MapValidationPolicy(maximum_age_s=3600.0)
    source_maps = map_set(policy=short_map_policy)
    development = build(evidence=source_maps)
    evidence = held_out_evidence(development)
    validated = build(
        evidence=source_maps,
        held_out_validation_evidence=evidence,
    )
    accepted_projection = accept_cft_projection(
        validated,
        source_maps,
        held_out_validation_evidence=evidence,
        orbit_adapter=OrbitAdapter(),
        reference_time_utc=NOW,
    )
    assert cft_solver_inputs(
        accepted_projection,
        reference_time_utc=NOW,
    )
    assert cft_solver_inputs(
        accepted_projection,
        reference_time_utc=NOW + timedelta(hours=2),
    ) == ()


def test_projection_rechecks_held_out_freshness_at_each_clock() -> None:
    registration = replace(
        VALIDATION_REGISTRATION,
        policy=HeldOutValidationPolicy(
            maximum_age_s=3600.0,
            maximum_future_skew_s=1.0,
        ),
    )
    source_maps = map_set(
        policy=MapValidationPolicy(maximum_age_s=None),
    )
    development = build(
        evidence=source_maps,
        validation_registration=registration,
    )
    evidence = held_out_evidence(development)
    validated = build(
        evidence=source_maps,
        validation_registration=registration,
        held_out_validation_evidence=evidence,
    )
    accepted_projection = accept_cft_projection(
        validated,
        source_maps,
        held_out_validation_evidence=evidence,
        orbit_adapter=OrbitAdapter(),
        reference_time_utc=NOW,
    )
    assert cft_solver_inputs(
        accepted_projection,
        reference_time_utc=NOW + timedelta(hours=2),
    ) == ()


@pytest.mark.parametrize(
    "claim_changes",
    (
        {"field_model_hash": "7" * 64},
        {"code_hash": "7" * 64},
        {"config_hash": "7" * 64},
    ),
)
def test_same_field_values_with_substituted_provenance_are_not_members(
    claim_changes,
) -> None:
    development = build()
    evidence = held_out_evidence(development)
    substituted_maps = map_set(**claim_changes)
    substituted = build(evidence=substituted_maps)
    assert (
        substituted.stability.primary.identity.full_map_hash
        == development.stability.primary.identity.full_map_hash
    )
    assert substituted.evidence_fingerprints != development.evidence_fingerprints
    with pytest.raises(EvidenceVerificationError, match="preregistration"):
        build(
            evidence=substituted_maps,
            held_out_validation_evidence=evidence,
        )


def test_projection_clock_is_explicit_and_deterministic() -> None:
    development = build()
    evidence = held_out_evidence(development)
    validated = build(held_out_validation_evidence=evidence)
    accepted_projection = accept_cft_projection(
        validated,
        map_set(),
        held_out_validation_evidence=evidence,
        orbit_adapter=OrbitAdapter(),
        reference_time_utc=NOW,
    )
    first = cft_solver_inputs(accepted_projection, reference_time_utc=NOW)
    second = cft_solver_inputs(accepted_projection, reference_time_utc=NOW)
    assert first == second
    assert cft_solver_inputs(
        accepted_projection,
        reference_time_utc=NOW.replace(tzinfo=None),
    ) == ()


def test_v2_proxy_and_v3_record_cannot_masquerade_as_v4() -> None:
    from cft_revival.coupling import (
        build_screening_proxy,
        closed_contour_solver_inputs,
    )
    from tests.coupling.evidence_helpers import accepted_evidence
    from tests.coupling.test_v3_flux_contract import evidence_and_study
    from cft_revival.coupling.v3_models import (
        CellRegistration,
        ElectronAdiabaticInputs,
    )
    from cft_revival.coupling.v3_records import build_coupling_record

    with pytest.warns(DeprecationWarning):
        proxy = build_screening_proxy(
            accepted_evidence(), wall_radius_m=0.75, reference_time_utc=NOW
        )
    assert cft_solver_inputs(proxy, reference_time_utc=NOW) == ()  # type: ignore[arg-type]
    _, evidence, study = evidence_and_study()
    v3 = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.8,)),),
        electron_inputs=ElectronAdiabaticInputs(100.0),
        reference_time_utc=NOW,
    )
    assert closed_contour_solver_inputs(v3)
    assert cft_solver_inputs(v3, reference_time_utc=NOW) == ()  # type: ignore[arg-type]


def test_three_map_roles_reject_weakened_refinement_and_shrunken_domain() -> None:
    primary = accepted(hemp_map(21, 41), "primary")
    with pytest.raises(EvidenceVerificationError, match="higher resolution"):
        verify_v4_map_set(
            primary,
            accepted(hemp_map(41, 31), "refined"),
            accepted(
                hemp_map(31, 61, radius_m=1.5, axial_half_width_m=3.0),
                "enlarged",
            ),
            reference_time_utc=NOW,
        )
    with pytest.raises(EvidenceVerificationError, match="contain and extend"):
        verify_v4_map_set(
            primary,
            accepted(hemp_map(41, 81), "refined"),
            accepted(
                hemp_map(31, 61, radius_m=1.5, axial_half_width_m=2.0),
                "enlarged",
            ),
            reference_time_utc=NOW,
        )


def test_three_map_evidence_rejects_mixed_implementation_provenance() -> None:
    with pytest.raises(EvidenceVerificationError, match="code_hash"):
        verify_v4_map_set(
            accepted(hemp_map(21, 41), "primary"),
            accepted(
                hemp_map(41, 81),
                "refined",
                code_hash="f" * 64,
            ),
            accepted(
                hemp_map(31, 61, radius_m=1.5, axial_half_width_m=3.0),
                "enlarged",
            ),
            reference_time_utc=NOW,
        )


def test_path_extrema_and_orbits_are_bound_to_each_exact_map_trajectory() -> None:
    record = build()
    paths = [
        path
        for assessment in (
            record.stability.primary,
            record.stability.refined,
            record.stability.enlarged,
        )
        for outcome in assessment.cells[0].seed_outcomes
        for path in (outcome.negative_path, outcome.positive_path)
    ]
    assert len({path.path_hash for path in paths}) == len(paths)
    assert all(path.b_low_t <= path.b_high_t for path in paths)
    assert all(path.b_low_location_rz_m != path.b_high_location_rz_m for path in paths)
    assert all(
        assessment.path_hash == path.path_hash
        for path in paths
        for assessment in path.orbit_assessments
    )


def test_uncertainty_can_remove_positive_field_bound_atomically() -> None:
    record = build(
        uncertainty_model=UncertaintyModel(
            absolute_independent_sigma_t=10.0,
            coverage_factor=2.0,
        )
    )
    assert record.status is V4Status.AMBIGUOUS
    paths = record.stability.primary.cells[0].seed_outcomes[0]
    assert paths.negative_path.status is V4Status.UNCERTAINTY_DOMINATED
    assert paths.negative_path.mirror_probability is None
    assert cft_solver_inputs(record, reference_time_utc=NOW) == ()


def test_orbit_implementation_swap_invalidates_preregistration_projection() -> None:
    class SwappedOrbitAdapter(OrbitAdapter):
        orbit_config_hash = "f" * 64

    development = build()
    evidence = held_out_evidence(development)
    with pytest.raises(EvidenceVerificationError, match="preregistration"):
        build(
            orbit_adapter=SwappedOrbitAdapter(),
            held_out_validation_evidence=evidence,
        )
    swapped = build(orbit_adapter=SwappedOrbitAdapter())
    assert swapped.orbit_identity != development.orbit_identity
    assert swapped.record_hash != development.record_hash


def test_orbit_claim_identity_must_echo_preregistered_versions_and_hashes() -> None:
    class MismatchedClaimsAdapter(OrbitAdapter):
        def verify_orbit(self, path_points_rz_m, path_hash, sample):
            claims = super().verify_orbit(
                path_points_rz_m,
                path_hash,
                sample,
            )
            return replace(claims, orbit_model_version="unexpected-9.0")

    with pytest.raises(EvidenceVerificationError, match="preregistered"):
        build(orbit_adapter=MismatchedClaimsAdapter())


def test_evaluated_three_map_hashes_must_be_manifest_members() -> None:
    development = build()
    mismatched = HeldOutCaseOutcome(
        case_id=HELD_OUT_CASE_ID,
        geometry_family_id=HELD_OUT_FAMILY_ID,
        three_map_hashes=("f" * 64, "e" * 64, "d" * 64),
        three_map_evidence_fingerprints=development.evidence_fingerprints,
        passed=True,
    )
    evidence = held_out_evidence(development, outcomes=(mismatched,))
    with pytest.raises(EvidenceVerificationError, match="manifest member"):
        build(held_out_validation_evidence=evidence)


def test_development_manifest_ids_cannot_be_held_out() -> None:
    overlapping = ValidationSetManifest(
        manifest_id="invalid-held-out-development-reuse",
        case_ids=(CFT_V4_DEVELOPMENT_MANIFEST.case_ids[0],),
        geometry_family_ids=("new-family",),
        manifest_hash=validation_set_manifest_hash(
            "invalid-held-out-development-reuse",
            (CFT_V4_DEVELOPMENT_MANIFEST.case_ids[0],),
            ("new-family",),
        ),
    )
    registration = replace(
        VALIDATION_REGISTRATION,
        held_out_manifest=overlapping,
        evaluated_case_id=overlapping.case_ids[0],
        evaluated_geometry_family_id=overlapping.geometry_family_ids[0],
        required_case_count=1,
        required_outcomes=(
            HeldOutCaseRegistration(
                overlapping.case_ids[0],
                overlapping.geometry_family_ids[0],
            ),
        ),
    )
    with pytest.raises(CouplingValidationError, match="disjoint"):
        build(validation_registration=registration)


def test_physical_prominence_is_stable_across_81_161_321_grids() -> None:
    evidence = verify_v4_map_set(
        accepted(hemp_map(31, 81), "primary"),
        accepted(hemp_map(61, 161), "refined"),
        accepted(
            hemp_map(
                51,
                321,
                radius_m=1.5,
                axial_half_width_m=3.0,
            ),
            "enlarged",
        ),
        reference_time_utc=NOW,
    )
    record = build(
        evidence=evidence,
        cusp_policy=WallCuspPolicy(
            minimum_prominence_t=0.2,
            prominence_support_half_width_m=0.35,
            minimum_cusp_separation_m=0.2,
            endpoint_plane_tolerance_m=0.3,
        ),
    )
    assert record.stability.cusp_counts == (2, 2, 2)
    assert all(
        cusp.prominence_t >= 0.2
        for assessment in (
            record.stability.primary,
            record.stability.refined,
            record.stability.enlarged,
        )
        for cusp in assessment.cusps
    )


def test_topographic_prominence_rejects_high_frequency_wall_ripples() -> None:
    evidence = verify_v4_map_set(
        accepted(hemp_map(31, 81, ripple_amplitude_t=0.05), "primary"),
        accepted(hemp_map(61, 161, ripple_amplitude_t=0.05), "refined"),
        accepted(
            hemp_map(
                51,
                321,
                radius_m=1.5,
                axial_half_width_m=3.0,
                ripple_amplitude_t=0.05,
            ),
            "enlarged",
        ),
        reference_time_utc=NOW,
    )
    record = build(
        evidence=evidence,
        cusp_policy=WallCuspPolicy(
            minimum_prominence_t=0.15,
            prominence_support_half_width_m=0.3,
            minimum_cusp_separation_m=0.15,
            endpoint_plane_tolerance_m=0.3,
        ),
    )
    assert record.stability.cusp_counts == (2, 2, 2)


def test_event_aware_trace_never_samples_beyond_nearby_wall_domain() -> None:
    radius = 1.00002
    evidence = verify_v4_map_set(
        accepted(hemp_map(31, 81, radius_m=radius), "primary"),
        accepted(hemp_map(61, 161, radius_m=radius), "refined"),
        accepted(
            hemp_map(
                51,
                201,
                radius_m=1.1,
                axial_half_width_m=3.0,
            ),
            "enlarged",
        ),
        reference_time_utc=NOW,
    )
    trace_policy = FieldLineTracePolicy(
        step_m=0.05,
        wall_tolerance_m=1e-5,
        maximum_psi_drift_wb=0.02,
    )
    record = build(evidence=evidence, trace_policy=trace_policy)
    paths = (
        record.stability.primary.cells[0].seed_outcomes[0].negative_path,
        record.stability.primary.cells[0].seed_outcomes[0].positive_path,
    )
    assert all(path.termination == "channel_wall" for path in paths)
    assert all(
        path.wall_endpoint_error_m is not None
        and path.wall_endpoint_error_m <= trace_policy.wall_tolerance_m
        for path in paths
    )
    assert all(
        max(point[0] for point in path.points_rz_m)
        <= GEOMETRY.channel_wall_radius_m
        for path in paths
    )


def test_cross_map_cusp_count_change_is_typed_not_raised() -> None:
    evidence = verify_v4_map_set(
        accepted(hemp_map(31, 81), "primary"),
        accepted(
            hemp_map(61, 161, extra_wall_peak_t=3.0),
            "refined",
        ),
        accepted(
            hemp_map(
                51,
                201,
                radius_m=1.5,
                axial_half_width_m=3.0,
            ),
            "enlarged",
        ),
        reference_time_utc=NOW,
    )
    record = build(evidence=evidence)
    assert record.status is V4Status.AMBIGUOUS
    assert record.stability.cusp_counts == (2, 3, 2)
    assert record.stability.refined.detected_cusp_count == 3
    assert not record.stability.refined.cells
    assert record.stability.cusp_assignment == ((0, 0, 0), (1, 2, 1))
    assert "requires 2" in record.stability.refined.reason
    assert "primary=2, refined=3, enlarged=2" in record.stability.reason


def test_v4_dictionary_carries_complete_held_out_identity() -> None:
    development = build()
    validated = build(
        held_out_validation_evidence=held_out_evidence(development)
    )
    payload = cft_coupling_record_dict(validated)
    assert payload["schema_version"].endswith("/4.1.0")
    assert payload["held_out_validation"]["development_manifest"]["manifest_id"] == (
        "assessed-56-case-characterization"
    )
    assert payload["orbit_identity"]["adapter_version"] == "2.1.0"
    assert payload["orbit_identity"]["orbit_config_hash"] == "d" * 64
    changed_registration = replace(
        VALIDATION_REGISTRATION,
        validation_config_hash="a" * 64,
    )
    changed_development = build(
        validation_registration=changed_registration
    )
    changed = build(
        validation_registration=changed_registration,
        held_out_validation_evidence=held_out_evidence(
            changed_development,
            validation_config_hash="a" * 64,
        ),
    )
    assert changed.record_hash != validated.record_hash
