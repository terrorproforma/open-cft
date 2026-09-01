from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from math import exp, isfinite, sqrt

import pytest

from cft_revival.coupling import (
    AdapterVersionContract,
    CellRegistration,
    FluxContour,
    ElectronAdiabaticInputs,
    EvidenceVerificationError,
    FluxSurfacePolicy,
    MapValidationPolicy,
    SolverDiagnosticsEvidence,
    StabilityCase,
    SurfaceStatus,
    TopologyResolutionError,
    TopologyStabilityStudy,
    UncertaintyModel,
    V3ArtifactClaims,
    build_closed_contour_record as build_coupling_record,
    certify_contour_field,
    closed_contour_solver_inputs as global_solver_inputs,
    hash_psi_map,
    magnetic_null_geometry,
    require_same_flux_surface,
    trace_flux_contours,
    validate_simple_contour,
    v3_evidence_binding_hash,
    verify_topology_stability,
    verify_v3_field_artifact,
    verify_v3_topology_stability,
)
from cft_revival.coupling.v3_models import ValidatedPsiMap
from tests.coupling.evidence_helpers import NOW


@dataclass(frozen=True)
class PsiMap:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]
    psi_wb: tuple[tuple[float, ...], ...]
    b_r_t: tuple[tuple[float, ...], ...]
    b_z_t: tuple[tuple[float, ...], ...]


def island_map(
    radial_points: int = 41,
    axial_points: int = 41,
    *,
    radius_m: float = 3.0,
    axial_half_width_m: float = 2.0,
) -> PsiMap:
    phi = (1.0 + sqrt(5.0)) * 0.5
    radii = tuple(
        sorted(
            {radius_m * index / (radial_points - 1) for index in range(radial_points)}
            | {phi}
        )
    )
    axial = tuple(
        -axial_half_width_m
        + 2.0 * axial_half_width_m * index / (axial_points - 1)
        for index in range(axial_points)
    )

    def values(radius: float, z: float) -> tuple[float, float, float]:
        envelope = exp(-((radius - 1.0) ** 2 + z * z))
        psi = radius * radius * envelope
        br = 2.0 * z * radius * envelope
        bz = 2.0 * envelope * (1.0 - radius * (radius - 1.0))
        return psi, br, bz

    rows = tuple(
        tuple(values(radius, z) for z in axial)
        for radius in radii
    )
    return PsiMap(
        radii,
        axial,
        tuple(tuple(value[0] for value in row) for row in rows),
        tuple(tuple(value[1] for value in row) for row in rows),
        tuple(tuple(value[2] for value in row) for row in rows),
    )


class V3TestAdapter:
    adapter_id = "tests.v3.psi-adapter"
    adapter_code_hash = "a" * 64
    version_contract = AdapterVersionContract(
        "cft-v3-psi-direct",
        "1.0.0",
        "cft-axisymmetric-field-map/1.1.0",
        "cft-axisymmetric-field-map/1.1.0",
        "L1a",
    )

    def __init__(self, claims: V3ArtifactClaims) -> None:
        self.claims = claims

    def verify_v3_artifact(self, artifact_bytes: bytes) -> V3ArtifactClaims:
        return self.claims


def accepted_v3(field: PsiMap, role: str):
    artifact = ('{"accepted":"psi-island-v3","role":"' + role + '"}').encode()
    artifact_hash = hashlib.sha256(artifact).hexdigest()
    full_hash = hash_psi_map(field)
    hashes = {
        "source": "1" * 64,
        "geometry": "2" * 64,
        "material": "3" * 64,
        "mesh": hashlib.sha256(("mesh-" + role).encode()).hexdigest(),
        "domain": hashlib.sha256(("domain-" + role).encode()).hexdigest(),
    }
    claims = V3ArtifactClaims(
        field,
        "cft-axisymmetric-field-map/1.1.0",
        "L1a",
        artifact_hash,
        full_hash,
        hashes["source"],
        hashes["geometry"],
        hashes["material"],
        hashes["mesh"],
        hashes["domain"],
        v3_evidence_binding_hash(
            full_hash,
            hashes["source"],
            hashes["geometry"],
            hashes["material"],
            hashes["mesh"],
            hashes["domain"],
            artifact_hash,
        ),
        "manufactured-python",
        "1.0",
        "axisymmetric-flux-island",
        "6" * 64,
        "7" * 64,
        "8" * 64,
        NOW,
        SolverDiagnosticsEvidence(True, 1e-12, 1e-10, 1e-12, 1e-10, 20),
    )
    return verify_v3_field_artifact(
        artifact, V3TestAdapter(claims), reference_time_utc=NOW
    )


def evidence_and_study():
    field = island_map()
    evidence = accepted_v3(field, "full")
    downsampled = accepted_v3(island_map(21, 21), "downsampled")
    enlarged = accepted_v3(
        island_map(51, 51, radius_m=3.5, axial_half_width_m=2.5),
        "enlarged",
    )
    study = verify_v3_topology_stability(
        evidence,
        downsampled,
        enlarged,
        maximum_cusp_shift_m=0.05,
        reference_time_utc=NOW,
    )
    return field, evidence, study


def test_marching_squares_closes_manufactured_constant_psi_island() -> None:
    field, evidence, _ = evidence_and_study()
    # Access only through a reverified build-compatible map.
    from cft_revival.coupling import reverify_v3_evidence

    validated = reverify_v3_evidence(evidence, reference_time_utc=NOW).field_map
    target = 0.8 * max(value for row in field.psi_wb for value in row)
    contours = trace_flux_contours(validated, target)
    assert contours
    assert any(contour.closed and not contour.touches_boundary for contour in contours)
    assert max(contour.maximum_psi_residual_wb for contour in contours) < 1e-10


def test_same_z_different_psi_is_rejected() -> None:
    _, evidence, _ = evidence_and_study()
    from cft_revival.coupling import reverify_v3_evidence

    field = reverify_v3_evidence(evidence, reference_time_utc=NOW).field_map
    with pytest.raises(TopologyResolutionError, match="different psi"):
        require_same_flux_surface(field, ((0.0, 0.0), (1.0, 0.0)))


def test_endpoint_nulls_are_diagnostics_not_interior_cells() -> None:
    _, evidence, _ = evidence_and_study()
    from cft_revival.coupling import reverify_v3_evidence

    field = reverify_v3_evidence(evidence, reference_time_utc=NOW).field_map
    br = [list(row) for row in field.b_r_t]
    bz = [list(row) for row in field.b_z_t]
    br[0][0] = bz[0][0] = 0.0
    br[0][-1] = bz[0][-1] = 0.0
    modified = replace(
        field,
        b_r_t=tuple(tuple(row) for row in br),
        b_z_t=tuple(tuple(row) for row in bz),
    )
    interior, boundary = magnetic_null_geometry(modified)
    assert len(interior) == 1
    assert {item.boundary for item in boundary} >= {"z_min", "z_max"}


def test_stability_count_change_is_ambiguous_and_rejected() -> None:
    _, evidence, study = evidence_and_study()
    from cft_revival.coupling import (
        reverify_v3_evidence,
        reverify_v3_topology_stability,
    )

    field = reverify_v3_evidence(evidence, reference_time_utc=NOW).field_map
    summary = reverify_v3_topology_stability(
        study, full_map_hash=field.full_map_hash, reference_time_utc=NOW
    )
    bad = replace(
        summary,
        downsampled=replace(
            summary.downsampled, cell_count=2, interior_cusp_z_m=(-0.2, 0.2)
        ),
    )
    with pytest.raises(TopologyResolutionError, match="cell count"):
        verify_topology_stability(bad, field=field, observed_cusp_z_m=(0.0,))


def test_v3_record_publishes_only_same_surface_bounded_distribution() -> None:
    _, evidence, study = evidence_and_study()
    record = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.75, 0.85)),),
        electron_inputs=ElectronAdiabaticInputs(100.0),
        reference_time_utc=NOW,
    )
    assert record.schema_version.endswith("/3.0.0")
    assert record.cells[0].surfaces
    assert all(
        surface.contour.psi_wb == surface.psi_wb
        for surface in record.cells[0].surfaces
    )
    valid = [
        surface
        for surface in record.cells[0].surfaces
        if surface.probability.status is SurfaceStatus.VALID
    ]
    assert valid
    assert global_solver_inputs(record)
    assert all(
        surface.probability.probability_lower
        <= surface.probability.nominal_probability
        <= surface.probability.probability_upper
        for surface in valid
    )


def test_missing_adiabatic_inputs_and_dominant_uncertainty_publish_no_nominal() -> None:
    _, evidence, study = evidence_and_study()
    missing = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.8,)),),
        electron_inputs=None,
        reference_time_utc=NOW,
    )
    assert global_solver_inputs(missing) == ()
    assert all(
        surface.probability.nominal_probability is None
        for surface in missing.cells[0].surfaces
    )
    dominated = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.8,)),),
        electron_inputs=ElectronAdiabaticInputs(100.0),
        uncertainty_model=UncertaintyModel(relative_independent_sigma=10.0),
        reference_time_utc=NOW,
    )
    assert all(
        surface.probability.nominal_probability is None
        for surface in dominated.cells[0].surfaces
    )


def test_v3_evidence_binds_geometry_material_and_full_psi_map() -> None:
    field, evidence, _ = evidence_and_study()
    assert hash_psi_map(field)
    from cft_revival.coupling import reverify_v3_evidence

    snapshot = reverify_v3_evidence(evidence, reference_time_utc=NOW)
    forged_claims = replace(snapshot.claims, material_hash="f" * 64)
    object.__setattr__(snapshot, "claims", forged_claims)
    with pytest.raises(EvidenceVerificationError, match="invariant"):
        reverify_v3_evidence(evidence, reference_time_utc=NOW)


def test_x_point_is_interior_and_separatrix_is_not_a_closed_mirror() -> None:
    r = tuple(index / 10 for index in range(21))
    z = tuple(-1.0 + index / 10 for index in range(21))
    psi = tuple(
        tuple(radius * radius * ((radius - 1.0) ** 2 - axial * axial) for axial in z)
        for radius in r
    )
    br = tuple(
        tuple(2.0 * axial * radius for axial in z)
        for radius in r
    )
    bz = tuple(
        tuple(
            2.0 * ((radius - 1.0) ** 2 - axial * axial)
            + 2.0 * radius * (radius - 1.0)
            for axial in z
        )
        for radius in r
    )
    field = ValidatedPsiMap(r, z, psi, br, bz, "c" * 64)
    interior, _ = magnetic_null_geometry(field, absolute_tolerance_t=1e-14)
    assert (1.0, 0.0) in interior
    separatrix = trace_flux_contours(field, 0.0)
    assert not separatrix or any(
        not contour.closed
        or contour.touches_boundary
        or not contour.simple
        for contour in separatrix
    )


def test_analytic_dipole_points_are_bound_by_flux_not_equal_z() -> None:
    r = tuple(0.2 + 0.05 * index for index in range(37))
    z = tuple(0.5 + 0.05 * index for index in range(41))

    def dipole(radius: float, axial: float) -> tuple[float, float, float]:
        distance2 = radius * radius + axial * axial
        root5 = distance2 ** 2.5
        psi = radius * radius / (distance2 ** 1.5)
        return psi, 3.0 * radius * axial / root5, (
            2.0 * axial * axial - radius * radius
        ) / root5

    rows = tuple(tuple(dipole(radius, axial) for axial in z) for radius in r)
    field = ValidatedPsiMap(
        r,
        z,
        tuple(tuple(item[0] for item in row) for row in rows),
        tuple(tuple(item[1] for item in row) for row in rows),
        tuple(tuple(item[2] for item in row) for row in rows),
        "d" * 64,
    )
    # Analytic dipole ψ labels differ at equal z and agree only along field lines.
    with pytest.raises(TopologyResolutionError):
        require_same_flux_surface(field, ((0.5, 1.0), (1.0, 1.0)))
    level = dipole(0.8, 1.0)[0]
    contours = trace_flux_contours(field, level)
    assert contours
    assert all(contour.maximum_psi_residual_wb < 1e-10 for contour in contours)


def test_high_energy_or_strict_ordering_marks_surface_nonadiabatic() -> None:
    _, evidence, study = evidence_and_study()
    record = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.8,)),),
        electron_inputs=ElectronAdiabaticInputs(
            1.0e4, maximum_gyroradius_to_scale_length=1.0e-12
        ),
        reference_time_utc=NOW,
    )
    assert global_solver_inputs(record) == ()
    assert all(
        surface.probability.status is SurfaceStatus.NONADIABATIC
        for surface in record.cells[0].surfaces
    )


def test_extreme_finite_flux_interpolation_stays_finite() -> None:
    field = ValidatedPsiMap(
        (0.0, 1.0, 2.0),
        (-1.0, 0.0, 1.0),
        (
            (0.0, 0.0, 0.0),
            (-1.0e308, 0.0, 1.0e308),
            (-1.0e308, 0.0, 1.0e308),
        ),
        ((0.0, 0.0, 0.0),) * 3,
        ((1.0, 1.0, 1.0),) * 3,
        "e" * 64,
    )
    contours = trace_flux_contours(
        field,
        5.0e307,
        FluxSurfacePolicy(
            psi_absolute_tolerance_wb=0.0,
            psi_relative_tolerance=1.0e-15,
        ),
    )
    assert contours
    assert all(
        all(isfinite(value) for point in contour.points_rz_m for value in point)
        for contour in contours
    )


def test_v2_same_z_api_is_explicitly_deprecated_and_not_solver_accepted() -> None:
    from cft_revival.coupling import build_screening_proxy
    from tests.coupling.evidence_helpers import accepted_evidence

    with pytest.warns(DeprecationWarning, match="screening_proxy"):
        proxy = build_screening_proxy(
            accepted_evidence(), wall_radius_m=0.75, reference_time_utc=NOW
        )
    assert proxy.schema_version.endswith("/2.0.0")
    assert global_solver_inputs(proxy) == ()  # type: ignore[arg-type]


def test_certified_segment_interior_null_suppresses_probability() -> None:
    field = ValidatedPsiMap(
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
        ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)),
        ((0.0, 0.0, 0.0),) * 3,
        "f" * 64,
    )
    contour = FluxContour(
        1.0,
        ((0.25, 0.5), (0.75, 0.5), (0.75, 1.5), (0.25, 1.5), (0.25, 0.5)),
        True,
        False,
        0.0,
        0.0,
        True,
        "simple edge graph",
        4,
        4,
    )
    certificate = certify_contour_field(
        field,
        contour,
        null_floor_t=1e-15,
        absolute_tolerance_t=1e-14,
        relative_tolerance=0.01,
        maximum_depth=20,
    )
    assert not certificate.regular
    assert certificate.certified_b_low_lower_t == 0.0
    from cft_revival.coupling.v3_records import _surface_mirror

    mirror = _surface_mirror(
        field,
        contour,
        cell_id="crossing",
        quantile=0.5,
        component=0,
        uncertainty=UncertaintyModel(),
        surface_policy=FluxSurfacePolicy(),
        electron_inputs=ElectronAdiabaticInputs(100.0),
    )
    assert mirror.probability.status is SurfaceStatus.EXACT_NULL
    assert mirror.probability.nominal_probability is None


def test_every_registered_quantile_is_atomic_and_hash_visible() -> None:
    _, evidence, study = evidence_and_study()
    record = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.01, 0.8)),),
        electron_inputs=ElectronAdiabaticInputs(100.0),
        reference_time_utc=NOW,
    )
    assert len(record.cells[0].quantile_outcomes) == 2
    assert record.cell_registrations[0].flux_quantiles == (0.01, 0.8)
    assert any(
        outcome.status is not SurfaceStatus.VALID
        for outcome in record.cells[0].quantile_outcomes
    )
    assert record.topology_status.value == "ambiguous"
    assert global_solver_inputs(record) == ()


@pytest.mark.parametrize(
    ("values", "expected_pairs"),
    (
        ((2.0, -1.0, 2.0, -1.0), (((0, 1)), ((2, 3)))),
        ((-1.0, 2.0, -1.0, 2.0), (((0, 3)), ((1, 2)))),
        ((-2.0, 1.0, -2.0, 1.0), (((0, 1)), ((2, 3)))),
    ),
)
def test_asymptotic_decider_rotations_and_signs(
    values: tuple[float, float, float, float],
    expected_pairs: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    from cft_revival.coupling.surfaces import _cell_segments

    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    segments = _cell_segments(corners, values, 0.0, 0.0, 1e-12, "reject")
    edge_for_point = {
        (round(point[0], 12), round(point[1], 12)): edge
        for edge, point in enumerate(
            (
                (abs(values[0]) / (abs(values[0]) + abs(values[1])), 0.0),
                (1.0, abs(values[1]) / (abs(values[1]) + abs(values[2]))),
                (1.0 - abs(values[2]) / (abs(values[2]) + abs(values[3])), 1.0),
                (0.0, 1.0 - abs(values[3]) / (abs(values[3]) + abs(values[0]))),
            )
        )
    }
    actual = {
        tuple(
            sorted(
                edge_for_point[(round(point[0], 12), round(point[1], 12))]
                for point in segment
            )
        )
        for segment in segments
    }
    assert actual == {tuple(sorted(pair)) for pair in expected_pairs}


def test_exact_saddle_requires_declared_pairing() -> None:
    from cft_revival.coupling.surfaces import _cell_segments

    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    values = (1.0, -1.0, 1.0, -1.0)
    with pytest.raises(TopologyResolutionError, match="tie policy"):
        _cell_segments(corners, values, 0.0, 0.0, 1e-12, "reject")
    first = _cell_segments(corners, values, 0.0, 0.0, 1e-12, "pair_01_23")
    second = _cell_segments(corners, values, 0.0, 0.0, 1e-12, "pair_03_12")
    assert first != second


def test_retraced_grid_aligned_loop_is_not_simple() -> None:
    points = (
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
        (0.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 1.0),
        (2.0, 1.0), (2.0, 0.0), (0.0, 0.0), (0.0, 1.0),
        (0.0, 0.0),
    )
    simple, reason, unique, edges = validate_simple_contour(
        points, tolerance_m=1e-12
    )
    assert not simple
    assert unique == 8
    assert edges == 12
    assert "repeated" in reason or "duplicate" in reason


def test_coverage_factor_widens_bounds_and_changes_record_identity() -> None:
    _, evidence, study = evidence_and_study()
    records = tuple(
        build_coupling_record(
            evidence,
            stability_evidence=study,
            cell_registrations=(CellRegistration("cell-1", (0.8,)),),
            electron_inputs=ElectronAdiabaticInputs(100.0),
            uncertainty_model=UncertaintyModel(
                relative_independent_sigma=0.005,
                coverage_factor=factor,
            ),
            reference_time_utc=NOW,
        )
        for factor in (1.0, 10.0)
    )
    surfaces = tuple(record.cells[0].surfaces[0] for record in records)
    widths = tuple(
        surface.probability.probability_upper
        - surface.probability.probability_lower
        for surface in surfaces
    )
    assert widths[1] > widths[0]
    assert records[0].record_hash != records[1].record_hash
    transition = tuple(
        build_coupling_record(
            evidence,
            stability_evidence=study,
            cell_registrations=(CellRegistration("cell-1", (0.8,)),),
            electron_inputs=ElectronAdiabaticInputs(100.0),
            uncertainty_model=UncertaintyModel(
                relative_independent_sigma=0.1,
                coverage_factor=factor,
            ),
            reference_time_utc=NOW,
        )
        for factor in (1.0, 10.0)
    )
    assert transition[0].cells[0].status is SurfaceStatus.VALID
    assert transition[1].cells[0].status is SurfaceStatus.UNCERTAINTY_DOMINATED
    assert global_solver_inputs(transition[1]) == ()


def test_extreme_opposite_axis_root_is_midpoint_and_subnormal_safe() -> None:
    def axis_field(left: float, right: float) -> ValidatedPsiMap:
        return ValidatedPsiMap(
            (0.0, 1.0, 2.0),
            (-2.0, -1.0, 1.0, 2.0),
            ((0.0, 0.0, 0.0, 0.0),) * 3,
            ((0.0, 0.0, 0.0, 0.0),) * 3,
            (
                (left, left, right, right),
                (1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0),
            ),
            "a" * 64,
        )

    huge, _ = magnetic_null_geometry(
        axis_field(-1e308, 1e308),
        relative_tolerance=0.0,
        absolute_tolerance_t=0.0,
    )
    tiny, _ = magnetic_null_geometry(
        axis_field(-5e-324, 5e-324),
        relative_tolerance=0.0,
        absolute_tolerance_t=0.0,
    )
    assert any(radius == 0.0 and axial == pytest.approx(0.0) for radius, axial in huge)
    assert any(radius == 0.0 and axial == pytest.approx(0.0) for radius, axial in tiny)


def test_record_identity_carries_complete_three_map_evidence() -> None:
    _, evidence, study = evidence_and_study()
    record = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.8,)),),
        electron_inputs=ElectronAdiabaticInputs(100.0),
        reference_time_utc=NOW,
    )
    assert record.identity.field_model_id == "axisymmetric-flux-island"
    assert record.identity.validation_policy.maximum_age_s == 86400.0
    cases = (
        record.stability_study.full_resolution,
        record.stability_study.downsampled,
        record.stability_study.enlarged_domain,
    )
    assert len({case.artifact_hash for case in cases}) == 3
    assert all(case.field_model_id and case.evidence_binding_hash for case in cases)


def test_extreme_finite_mirror_uncertainty_is_typed_not_published() -> None:
    field = ValidatedPsiMap(
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
        ((0.0, 0.0, 0.0),) * 3,
        ((1e308, 1e308, 1e308),) * 3,
        "b" * 64,
    )
    contour = FluxContour(
        1.0,
        ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)),
        True,
        False,
        0.0,
        0.0,
        True,
        "simple edge graph",
        4,
        4,
    )
    from cft_revival.coupling.v3_records import _surface_mirror

    mirror = _surface_mirror(
        field,
        contour,
        cell_id="extreme",
        quantile=0.5,
        component=0,
        uncertainty=UncertaintyModel(
            relative_independent_sigma=10.0,
            coverage_factor=10.0,
        ),
        surface_policy=FluxSurfacePolicy(),
        electron_inputs=ElectronAdiabaticInputs(100.0),
    )
    assert mirror.probability.status is SurfaceStatus.NUMERICALLY_INVALID
    assert mirror.probability.nominal_probability is None


def test_nonrelativistic_model_rejects_relativistic_energy() -> None:
    _, evidence, study = evidence_and_study()
    record = build_coupling_record(
        evidence,
        stability_evidence=study,
        cell_registrations=(CellRegistration("cell-1", (0.8,)),),
        electron_inputs=ElectronAdiabaticInputs(1.0e6),
        reference_time_utc=NOW,
    )
    assert all(
        surface.probability.status is SurfaceStatus.PHYSICALLY_INVALID
        for surface in record.cells[0].surfaces
    )
    assert global_solver_inputs(record) == ()
