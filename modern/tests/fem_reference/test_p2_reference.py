from __future__ import annotations

import copy
import hashlib
import json
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np

import pytest

import cft_revival.fem_reference.artifacts as artifact_module
from cft_revival.fem_reference import (
    FEMValidationError,
    ResourceBlockedError,
    Domain,
    FEMProblem,
    FEMResult,
    Region,
    SolverDiagnostics,
    adjacent_size_growth,
    bore_volume_average,
    bore_wall_line_average,
    build_body_fitted_mesh,
    component_dorfler_mark,
    current_process_rss_bytes,
    checkpoint_metadata_summary,
    dorfler_mark,
    edge_flux_jump_term,
    estimate_indicators,
    estimate_peak_allocation_bytes,
    evaluate_phase_matched_domain_expansion,
    artifact_from_result,
    graded_mesh_geometry,
    mesh_geometry,
    mesh_quality,
    load_checkpoint_bundle,
    patch_recovered_axis_bz,
    preflight_third_level,
    preflight_level_allocation,
    prolong_p2_solution,
    qois,
    refine_mesh,
    replay_artifact,
    solve,
    validate_artifact,
    viewer_contract,
    write_checkpoint_bundle,
)
from cft_revival.fem_reference.verification import (
    dipole_robin_case,
    piecewise_interface_case,
    pm_representation_equivalence,
    smooth_manufactured_convergence,
    uniform_medium_l1a_crosscheck,
)
from cft_revival.fem_reference.assembly import assemble
from cft_revival.geometry import divergent_exit_stack, reference_variants
from cft_revival.material_fields.acceptance import _bore_average as l1b_bore_average


def test_smooth_axis_regular_manufactured_solution_has_p2_order() -> None:
    report = smooth_manufactured_convergence()
    assert min(report["observed_orders"]) > 2.8
    assert min(report["observed_energy_orders"]) > 1.9
    assert report["relative_errors"][-1] < 6.0e-5
    assert report["integrated_energy_relative_errors"][-1] < 2.0e-3
    assert all(
        fine < coarse
        for coarse, fine in zip(
            report["estimator_norms"], report["estimator_norms"][1:]
        )
    )
    assert max(report["estimator_effectivities"]) / min(
        report["estimator_effectivities"]
    ) < 3.0
    assert max(report["qoi_localization_ratios"]) < 1.0


@pytest.mark.parametrize("oblique", [False, True])
def test_piecewise_mu_interface_is_conforming_and_flux_consistent(oblique: bool) -> None:
    report = piecewise_interface_case(oblique=oblique)
    assert report["relative_solution_error"] < 2.0e-10
    assert report["relative_true_residual"] < 3.0e-11


def test_corrected_dipole_robin_matches_analytic_solution() -> None:
    report = dipole_robin_case(12)
    assert report["relative_solution_error"] < 3.0e-3
    assert report["relative_true_residual"] < 3.0e-11


def test_pm_recoil_equivalent_current_and_polarity() -> None:
    report = pm_representation_equivalence(radial_divisions=6, axial_divisions=12)
    assert report["recoil_equivalent_relative_max_difference"] < 5.0e-10
    assert report["polarity_reversal_relative_max_error"] < 5.0e-10
    assert report["recoil_energy_action_relative"] < 1.0e-12


def test_uniform_medium_agrees_with_l1a_at_fixed_axis_points() -> None:
    report = uniform_medium_l1a_crosscheck()
    assert report["maximum_relative_difference"] < 3.0e-3


def _analytic_quadratic_result() -> FEMResult:
    domain = Domain(0.0, 1.0, -1.0, 1.0)
    problem = FEMProblem(
        "analytic-qoi",
        domain,
        (Region("vacuum", "vacuum", 1.0),),
        lambda _r, _z: "vacuum",
        outer_boundary="dirichlet",
        dirichlet_a_phi=lambda _r, _z: 0.0,
    )
    mesh = build_body_fitted_mesh(
        domain, (), problem.region_at, radial_divisions=5, axial_divisions=8
    )
    coefficient = 0.75
    radial = mesh.p2_nodes_rz_m[:, 0]
    a_phi = radial + coefficient * radial**2
    diagnostics = SolverDiagnostics(
        True, 0, 0.0, 0.0, 0.0, (0.0,), 0.0, 0.0, 0.0, 0.0, 0.0, 0
    )
    return FEMResult(problem, mesh, a_phi, diagnostics, "0" * 64)


def test_bore_qoi_is_axisymmetric_volume_average_not_wall_line() -> None:
    result = _analytic_quadratic_result()
    radius = 0.8
    volume = bore_volume_average(result, radius, -0.5, 0.5)
    wall = bore_wall_line_average(result, radius, -0.5, 0.5)
    assert volume == pytest.approx(2.0 + 2.0 * 0.75 * radius, rel=2.0e-14)
    assert wall == pytest.approx(2.0 + 3.0 * 0.75 * radius, rel=2.0e-14)
    assert volume != pytest.approx(wall)
    assert patch_recovered_axis_bz(result, 0.0) == pytest.approx(2.0, rel=2.0e-13)


def test_bore_volume_average_matches_l1b_bilinear_quadrature() -> None:
    result = _analytic_quadratic_result()
    domain = {
        "dr_m": 0.1,
        "dz_m": 0.25,
        "radial_intervals": 10,
        "axial_intervals": 8,
        "z_min_m": -1.0,
    }
    coefficient = 0.75
    values = [
        [2.0 + 3.0 * coefficient * (0.1 * radial) for _ in range(9)]
        for radial in range(11)
    ]
    l1b_value = l1b_bore_average(domain, values, 0.8, -0.5, 0.5)
    fem_value = bore_volume_average(result, 0.8, -0.5, 0.5)
    assert fem_value == pytest.approx(l1b_value, rel=2.0e-14)


def test_dorfler_marking_and_refinement_are_nested_and_deterministic() -> None:
    result = _analytic_quadratic_result()
    report = estimate_indicators(result, (("qoi", 0.8, -0.5, 0.5),))
    marked = component_dorfler_mark(report, theta=0.5)
    for component in (
        report.residual_squared,
        report.flux_jump_squared,
        report.qoi_proxy_squared,
    ):
        assert component[marked].sum() >= 0.5 * component.sum()
    first = refine_mesh(
        result.mesh,
        result.problem.domain,
        marked,
        maximum_adjacent_size_growth=1.3,
    )
    second = refine_mesh(
        result.mesh,
        result.problem.domain,
        marked,
        maximum_adjacent_size_growth=1.3,
    )
    assert first.sha256 == second.sha256
    assert first.parent_mesh_sha256 == result.mesh.sha256
    assert first.refinement_level == result.mesh.refinement_level + 1
    assert np.array_equal(
        first.vertices_rz_m[: len(result.mesh.vertices_rz_m)],
        result.mesh.vertices_rz_m,
    )
    assert mesh_quality(first)["minimum_angle_deg"] >= 10.0
    assert adjacent_size_growth(first) <= 1.3 + 1.0e-12


def test_zero_dorfler_components_do_not_create_synthetic_marks() -> None:
    assert dorfler_mark(np.zeros(4), theta=0.5).size == 0


def test_constant_flux_jump_has_edge_length_squared_scaling() -> None:
    jump = 3.25
    first = edge_flux_jump_term(0.2, np.full(3, jump))
    second = edge_flux_jump_term(0.4, np.full(3, jump))
    assert first == pytest.approx(0.2**2 * jump**2)
    assert second == pytest.approx(0.4**2 * jump**2)
    assert second / first == pytest.approx(4.0)


def test_two_pass_csr_is_deterministic_symmetric_and_sparse() -> None:
    result = _analytic_quadratic_result()
    first = assemble(result.problem, result.mesh)
    second = assemble(result.problem, result.mesh)
    assert np.array_equal(first.matrix.indptr, second.matrix.indptr)
    assert np.array_equal(first.matrix.indices, second.matrix.indices)
    assert np.array_equal(first.matrix.data, second.matrix.data)
    dense = np.zeros(first.matrix.shape)
    for row in range(first.matrix.shape[0]):
        start, stop = first.matrix.indptr[row : row + 2]
        dense[row, first.matrix.indices[start:stop]] = first.matrix.data[start:stop]
    assert np.allclose(dense, dense.T, rtol=0.0, atol=2.0e-14)


def test_prolonged_initial_guess_preserves_solution_and_reduces_iterations() -> None:
    domain = Domain(0.0, 1.0, -1.0, 1.0)
    problem = FEMProblem(
        "warm-start",
        domain,
        (Region("vacuum", "vacuum", 1.0),),
        lambda _r, _z: "vacuum",
        free_current_phi=lambda r_m, z_m: r_m * (1.0 + z_m),
        outer_boundary="dirichlet",
        dirichlet_a_phi=lambda _r, _z: 0.0,
    )
    coarse_mesh = build_body_fitted_mesh(
        domain, (), problem.region_at, radial_divisions=4, axial_divisions=8
    )
    coarse = solve(problem, coarse_mesh, relative_tolerance=1.0e-10)
    fine_mesh = refine_mesh(coarse_mesh, domain)
    initial = prolong_p2_solution(coarse, fine_mesh)
    cold = solve(problem, fine_mesh, relative_tolerance=1.0e-10)
    warm = solve(
        problem,
        fine_mesh,
        relative_tolerance=1.0e-10,
        initial_a_phi_dofs_t_m=initial,
    )
    assert np.allclose(
        warm.a_phi_dofs_t_m, cold.a_phi_dofs_t_m, rtol=2.0e-9, atol=2.0e-12
    )
    cold_qois = qois(cold, (("window", 0.5, -0.5, 0.5),))
    warm_qois = qois(warm, (("window", 0.5, -0.5, 0.5),))
    for key in cold_qois:
        assert warm_qois[key] == pytest.approx(
            cold_qois[key], rel=2.0e-9, abs=2.0e-12
        )
    assert warm.diagnostics.iterations <= cold.diagnostics.iterations


def test_parent_topology_prolongation_is_near_linear_over_four_levels(
    monkeypatch,
) -> None:
    coarse = _analytic_quadratic_result()
    monkeypatch.setattr(
        "cft_revival.fem_reference.solver.locate_element",
        lambda *_args, **_kwargs: pytest.fail("global element scan was used"),
    )
    normalized_times = []
    current = coarse
    for _ in range(4):
        fine_mesh = refine_mesh(current.mesh, current.problem.domain)
        timings = []
        prolonged = None
        for _repeat in range(3):
            started = perf_counter()
            prolonged = prolong_p2_solution(current, fine_mesh)
            timings.append(perf_counter() - started)
        assert prolonged is not None
        radial = fine_mesh.p2_nodes_rz_m[:, 0]
        assert np.allclose(
            prolonged,
            radial + 0.75 * radial**2,
            rtol=2.0e-13,
            atol=2.0e-15,
        )
        work = 6 * len(fine_mesh.triangles)
        normalized_times.append(min(timings) / work)
        current = FEMResult(
            current.problem,
            fine_mesh,
            prolonged,
            current.diagnostics,
            "0" * 64,
        )
    assert max(normalized_times) / min(normalized_times) < 5.0


def test_third_level_resource_preflight_is_fail_closed() -> None:
    with pytest.raises(FEMValidationError, match="exactly one design"):
        preflight_third_level(2, available_bytes=16 * 1024**3)
    with pytest.raises(ResourceBlockedError, match="NOT_EVALUATED") as blocked:
        preflight_third_level(1, available_bytes=8 * 1024**3 - 1)
    assert blocked.value.status == "NOT_EVALUATED"
    assert "8 GiB" in str(blocked.value)
    report = preflight_third_level(1, available_bytes=8 * 1024**3)
    assert report["passed"]
    assert report["maximum_p2_dofs"] >= 1_500_000


def test_calibrated_memory_model_and_per_assembly_recheck(monkeypatch) -> None:
    estimate = estimate_peak_allocation_bytes(
        p2_dofs=4_959, triangles=2_400, robin_edges=120
    )
    assert estimate["required_free_ram_bytes"] > 2_914_493
    assert current_process_rss_bytes() > 0
    required = int(estimate["required_free_ram_bytes"])
    with pytest.raises(FEMValidationError, match="allocation preflight"):
        preflight_level_allocation(
            p2_dofs=4_959,
            triangles=2_400,
            robin_edges=120,
            third_level=False,
            available_bytes=required - 1,
        )
    assert preflight_level_allocation(
        p2_dofs=4_959,
        triangles=2_400,
        robin_edges=120,
        third_level=False,
        available_bytes=required,
    )["passed"]
    result = _analytic_quadratic_result()
    monkeypatch.setattr(
        "cft_revival.fem_reference.resource_policy.available_ram_bytes",
        lambda: 0,
    )
    with pytest.raises(ResourceBlockedError, match="NOT_EVALUATED"):
        solve(result.problem, result.mesh, required_available_ram_bytes=1)


def test_every_allocation_preflight_independently_enforces_dof_cap() -> None:
    assert preflight_level_allocation(
        p2_dofs=1_500_000,
        triangles=750_000,
        third_level=True,
        available_bytes=1 << 60,
    )["passed"]
    for p2_dofs in (1_500_001, 3_000_000):
        with pytest.raises(ResourceBlockedError, match="NOT_EVALUATED"):
            preflight_level_allocation(
                p2_dofs=p2_dofs,
                triangles=750_000,
                third_level=False,
                available_bytes=1 << 60,
            )
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(FEMValidationError):
            preflight_level_allocation(
                p2_dofs=invalid,
                triangles=1,
                third_level=False,
                available_bytes=1 << 60,
            )
    with pytest.raises(ResourceBlockedError, match="topology"):
        preflight_level_allocation(
            p2_dofs=1,
            triangles=1_500_001,
            third_level=False,
            available_bytes=1 << 60,
        )
    with pytest.raises(FEMValidationError):
        build_body_fitted_mesh(
            Domain(0.0, 1.0, -1.0, 1.0),
            (),
            lambda _r, _z: "vacuum",
            radial_divisions=True,
            axial_divisions=2,
        )
    with pytest.raises(ResourceBlockedError, match="NOT_EVALUATED"):
        build_body_fitted_mesh(
            Domain(0.0, 1.0, -1.0, 1.0),
            (),
            lambda _r, _z: "vacuum",
            radial_divisions=3_000_000,
            axial_divisions=2,
        )


def test_domain_expansion_requires_phase_matched_local_h() -> None:
    studies = tuple(
        {
            "padding_factor": padding,
            "qois_bz_t": {"stage-1-bore-average": value},
            "qoi_h_m": {"stage-1-bore-average": 1.0e-4},
            "local_h_m": {"source": 8.0e-5, "stage-1-bore-average": 1.0e-4},
            "domain": {
                "r_min_m": 0.0,
                "r_max_m": 1.0 + padding,
                "z_min_m": -1.0 - padding,
                "z_max_m": 1.0 + padding,
            },
        }
        for padding, value in ((0.5, 1.0), (1.0, 1.005), (1.5, 1.007))
    )
    assert evaluate_phase_matched_domain_expansion(studies)["passed"]
    mismatched = copy.deepcopy(studies)
    mismatched[1]["qoi_h_m"]["stage-1-bore-average"] = 1.1e-4
    with pytest.raises(FEMValidationError, match="phase matched"):
        evaluate_phase_matched_domain_expansion(mismatched)
    nonfinite = copy.deepcopy(studies)
    nonfinite[1]["qoi_h_m"]["stage-1-bore-average"] = float("nan")
    with pytest.raises(FEMValidationError, match="invalid"):
        evaluate_phase_matched_domain_expansion(nonfinite)
    bad_extent = copy.deepcopy(studies)
    bad_extent[2]["domain"]["r_max_m"] = float("nan")
    with pytest.raises(FEMValidationError, match="extent"):
        evaluate_phase_matched_domain_expansion(bad_extent)


@pytest.mark.parametrize(
    "keywords",
    [
        {"relative_tolerance": 0.0},
        {"relative_tolerance": 1.0},
        {"relative_tolerance": float("nan")},
        {"absolute_tolerance": -1.0},
        {"absolute_tolerance": float("inf")},
        {"max_iterations": 0},
        {"max_iterations": True},
        {"max_iterations": 2.5},
    ],
)
def test_solver_controls_fail_closed_with_typed_errors(keywords) -> None:
    result = _analytic_quadratic_result()
    with pytest.raises(FEMValidationError):
        solve(result.problem, result.mesh, **keywords)


def test_divergent_geometry_mesh_is_body_fitted_tagged_and_deterministic() -> None:
    geometry = divergent_exit_stack()
    problem, first = mesh_geometry(
        geometry, radial_divisions=8, axial_divisions=16, padding_factor=0.5
    )
    _, second = mesh_geometry(
        geometry, radial_divisions=8, axial_divisions=16, padding_factor=0.5
    )
    assert first.sha256 == second.sha256
    assert set(first.triangle_region_ids) == {
        "ambient-background",
        *(region.region_id for region in geometry.regions),
    }
    quality = mesh_quality(first)
    assert quality["minimum_area_m2"] > 0.0
    assert quality["minimum_angle_deg"] > 0.1
    assert problem.geometry_sha256 == geometry.canonical_sha256


def test_initial_design_meshes_use_deterministic_quality_gated_grading() -> None:
    for geometry in reference_variants():
        _, first = graded_mesh_geometry(
            geometry, bore_elements=8, feature_elements=4, padding_factor=0.5
        )
        _, second = graded_mesh_geometry(
            geometry, bore_elements=8, feature_elements=4, padding_factor=0.5
        )
        assert first.sha256 == second.sha256
        assert mesh_quality(first)["minimum_angle_deg"] >= 20.0
        assert adjacent_size_growth(first) <= 1.3 + 1.0e-12


@pytest.fixture(scope="module")
def authoritative_artifact() -> dict[str, object]:
    geometry = divergent_exit_stack()
    problem, mesh = mesh_geometry(
        geometry, radial_divisions=6, axial_divisions=12, padding_factor=0.5
    )
    result = solve(problem, mesh, relative_tolerance=1.0e-8)
    stage = geometry.stages[0]
    windows = (
        (
            "stage-1",
            geometry.chamber.outer_radius_m,
            stage.z_min_m,
            stage.z_max_m,
        ),
    )
    return artifact_from_result(result, qoi_windows=windows)


def _reseal(artifact: dict[str, object]) -> None:
    payload = {key: value for key, value in artifact.items() if key != "integrity"}
    artifact["integrity"]["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _synthetic_authoritative_chain(tmp_path):
    base = _analytic_quadratic_result()
    problem = FEMProblem(
        "synthetic-authority-chain",
        base.problem.domain,
        base.problem.regions,
        lambda _r, _z: "vacuum",
        outer_boundary="dipole_robin",
    )
    windows = (("qoi", 0.8, -0.5, 0.5),)
    results = []
    bound_artifacts = []
    mesh = base.mesh
    for _level in range(3):
        result = solve(problem, mesh)
        values = qois(result, windows)
        results.append(result)
        bound_artifacts.append(
            artifact_from_result(result, qoi_values=values, qoi_windows=windows)
        )
        mesh = refine_mesh(mesh, problem.domain)
    root = bound_artifacts[-1]["acceptance_evidence"]["checkpoint_authority"]
    final_run = results[-1].run_sha256
    final_mesh = results[-1].mesh.sha256
    anchors = []
    previous_file = "0" * 64
    for level, (result, bound) in enumerate(zip(results, bound_artifacts)):
        points = result.mesh.vertices_rz_m
        areas = []
        centroids = []
        for triangle in result.mesh.triangles:
            triangle_points = points[triangle]
            first = triangle_points[1] - triangle_points[0]
            second = triangle_points[2] - triangle_points[0]
            areas.append(
                abs(float(first[0] * second[1] - first[1] * second[0]))
            )
            centroids.append(np.mean(triangle_points, axis=0))
        centroids = np.asarray(centroids)
        selected = (
            (centroids[:, 0] <= 0.8)
            & (centroids[:, 1] >= -0.5)
            & (centroids[:, 1] <= 0.5)
        )
        h = float(np.sqrt(np.mean(np.asarray(areas)[selected])))
        checkpoint = {
            "schema_version": "cft_revival.fem_reference.checkpoint/1.2.0",
            "classification": (
                "independent_numerical_reference_not_hardware_validation"
            ),
            "config_id": result.problem.problem_id,
            "level": level,
            "run_sha256": result.run_sha256,
            "mesh_sha256": result.mesh.sha256,
            "parent_mesh_sha256": result.mesh.parent_mesh_sha256,
            "previous_checkpoint_file_sha256": previous_file,
            "run": {
                "qois_bz_t": bound["qois_bz_t"],
                "resolution": {"qoi_h_m": {"qoi-bore-average": h}},
            },
            "bound_artifact": bound,
            "chain_authority": {
                "authority_root_sha256": root["authority_root_sha256"],
                "artifact_schema": root["artifact_schema"],
                "classification": root["classification"],
                "design_id": root["design_id"],
                "geometry_sha256": root["geometry_sha256"],
                "magnetics_sha256": root["magnetics_sha256"],
                "config_id": root["config_id"],
                "implementation_sha256": root["implementation_sha256"],
                "acceptance_code_sha256": root["acceptance_code_sha256"],
                "base_problem_sha256": root["problem_sha256"],
                "chain_kind": "adaptive",
                "final_checkpoint_run_sha256": final_run,
                "final_checkpoint_mesh_sha256": final_mesh,
            },
            "integrity": {},
        }
        path = tmp_path / f"level-{level}.json"
        file_hash = write_checkpoint_bundle(path, checkpoint)
        metadata = checkpoint_metadata_summary(path)
        robin_edges = sum(
            len(result.mesh.boundary_edges[name])
            for name in ("outer_radial", "z_min", "z_max")
        )
        anchors.append(
            {
                "level": level,
                "file": path.name,
                "file_sha256": file_hash,
                "payload_sha256": metadata["payload_sha256"],
                "mesh_sha256": result.mesh.sha256,
                "parent_mesh_sha256": result.mesh.parent_mesh_sha256,
                "previous_checkpoint_file_sha256": previous_file,
                "p2_dofs": len(result.mesh.p2_nodes_rz_m),
                "triangles": len(result.mesh.triangles),
                "robin_edges": robin_edges,
                "chain_final_run_sha256": final_run,
                "chain_final_mesh_sha256": final_mesh,
                "run_sha256": result.run_sha256,
                "problem_sha256": bound["acceptance_evidence"][
                    "checkpoint_authority"
                ]["problem_sha256"],
            }
        )
        previous_file = file_hash
    keys = ["qoi-bore-average"]
    changes = [{keys[0]: 0.0}, {keys[0]: 0.0}]
    convergence = {
        "acceptance_qois": keys,
        "successive_volume_qoi_relative_changes": changes,
        "observed_orders_from_actual_qoi_h": {keys[0]: None},
        "two_successive_less_than_one_percent": True,
        "stable_positive_order": False,
        "adjacent_size_growth_gate": True,
        "phase_matched_domain_expansion_gate": False,
        "less_than_one_percent_reached": False,
    }
    artifact = artifact_from_result(
        results[-1],
        qoi_values=bound_artifacts[-1]["qois_bz_t"],
        qoi_windows=windows,
        level_evidence=anchors,
        evidence_base_path=str(tmp_path),
        convergence=convergence,
    )
    return artifact, anchors


def test_complete_schema_1_3_chain_replays_and_rejects_foreign_and_low_ram(
    tmp_path,
) -> None:
    artifact, anchors = _synthetic_authoritative_chain(tmp_path)
    validate_artifact(artifact)
    assert replay_artifact(artifact)["acceptance_authority"] == "recomputed"

    understated = copy.deepcopy(artifact)
    understated["acceptance_evidence"]["level_evidence"][0]["p2_dofs"] -= 1
    understated["acceptance_evidence"]["checkpoint_authority"][
        "ordered_level_chain_sha256"
    ] = hashlib.sha256(
        json.dumps(
            understated["acceptance_evidence"]["level_evidence"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    _reseal(understated)
    with pytest.raises(FEMValidationError, match="verified headers"):
        validate_artifact(understated)

    first_path = tmp_path / anchors[0]["file"]
    domain_checkpoint, domain_verified = load_checkpoint_bundle(first_path)
    foreign_bound = domain_checkpoint["bound_artifact"]
    foreign_bound["anchors"]["geometry_sha256"] = "f" * 64
    controls = foreign_bound["acceptance_evidence"]["solver_controls"]
    solution = np.asarray(
        foreign_bound["solution"]["a_phi_dofs_t_m"], dtype=np.float64
    )
    run_payload = {
        "problem_id": foreign_bound["problem"]["problem_id"],
        "mesh_sha256": foreign_bound["anchors"]["mesh_sha256"],
        "geometry_sha256": "f" * 64,
        "magnetics_sha256": foreign_bound["anchors"]["magnetics_sha256"],
        "implementation_sha256": foreign_bound["acceptance_evidence"][
            "implementation_sha256"
        ],
        "relative_tolerance": controls["relative_tolerance"],
        "absolute_tolerance": controls["absolute_tolerance"],
        "max_iterations": controls["max_iterations"],
        "required_available_ram_bytes": controls["required_available_ram_bytes"],
        "initial_solution_sha256": foreign_bound["acceptance_evidence"][
            "initial_solution_sha256"
        ],
        "solution_sha256": hashlib.sha256(solution.tobytes()).hexdigest(),
    }
    foreign_run = hashlib.sha256(
        json.dumps(
            run_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    foreign_bound["anchors"]["run_sha256"] = foreign_run
    bound_payload = {
        key: value for key, value in foreign_bound.items() if key != "integrity"
    }
    foreign_bound["acceptance_evidence"]["checkpoint_authority"] = (
        artifact_module._checkpoint_authority(bound_payload)
    )
    _reseal(foreign_bound)
    top_authority = artifact["acceptance_evidence"]["checkpoint_authority"]
    domain_checkpoint["run_sha256"] = foreign_run
    domain_checkpoint["domain_study"] = {"padding_factor": 0.5}
    domain_checkpoint["chain_authority"] = {
        "authority_root_sha256": top_authority["authority_root_sha256"],
        "artifact_schema": top_authority["artifact_schema"],
        "classification": top_authority["classification"],
        "design_id": top_authority["design_id"],
        "geometry_sha256": top_authority["geometry_sha256"],
        "magnetics_sha256": top_authority["magnetics_sha256"],
        "config_id": top_authority["config_id"],
        "implementation_sha256": top_authority["implementation_sha256"],
        "acceptance_code_sha256": top_authority["acceptance_code_sha256"],
        "base_problem_sha256": top_authority["problem_sha256"],
        "chain_kind": "domain",
        "final_checkpoint_run_sha256": foreign_run,
        "final_checkpoint_mesh_sha256": foreign_bound["anchors"]["mesh_sha256"],
    }
    domain_path = tmp_path / "foreign-domain.json"
    domain_file_hash = write_checkpoint_bundle(domain_path, domain_checkpoint)
    domain_summary = checkpoint_metadata_summary(domain_path)
    domain_anchor = {
        **anchors[0],
        "file": domain_path.name,
        "file_sha256": domain_file_hash,
        "payload_sha256": domain_summary["payload_sha256"],
        "run_sha256": foreign_run,
        "padding_factor": 0.5,
        "chain_final_run_sha256": foreign_run,
            "chain_final_mesh_sha256": foreign_bound["anchors"]["mesh_sha256"],
    }
    with pytest.raises(FEMValidationError, match="geometry identity"):
        artifact_module._load_bound_checkpoint(
            domain_anchor,
            tmp_path,
            "0" * 64,
            "0" * 64,
            top_authority,
            "domain",
        )

    foreign = copy.deepcopy(artifact)
    checkpoint = json.loads(first_path.read_bytes())
    checkpoint["chain_authority"]["design_id"] = "unrelated-design"
    checkpoint_payload = {
        key: value for key, value in checkpoint.items() if key != "integrity"
    }
    checkpoint["integrity"]["payload_sha256"] = hashlib.sha256(
        json.dumps(
            checkpoint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    encoded = json.dumps(
        checkpoint, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    first_path.write_bytes(encoded)
    foreign_anchor = foreign["acceptance_evidence"]["level_evidence"][0]
    foreign_anchor["file_sha256"] = hashlib.sha256(encoded).hexdigest()
    foreign_anchor["payload_sha256"] = checkpoint["integrity"]["payload_sha256"]
    foreign["acceptance_evidence"]["checkpoint_authority"][
        "ordered_level_chain_sha256"
    ] = hashlib.sha256(
        json.dumps(
            foreign["acceptance_evidence"]["level_evidence"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    _reseal(foreign)
    with pytest.raises(FEMValidationError, match="unrelated authority"):
        validate_artifact(foreign)


def test_legacy_preliminary_finalization_is_guarded_before_json_parse(
    tmp_path, monkeypatch
) -> None:
    legacy = tmp_path / "legacy-array-heavy.json"
    with legacy.open("wb") as target:
        target.write(b"{")
        target.seek(8 * 1024**2 + 1)
        target.write(b"}")
    monkeypatch.setattr(
        "cft_revival.fem_reference.resource_policy.available_ram_bytes",
        lambda: 0,
    )
    with pytest.raises(ResourceBlockedError, match="legacy_checkpoint_migration"):
        load_checkpoint_bundle(legacy)
    campaign_source = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "fem_reference"
        / "run_reference_campaign.py"
    ).read_text(encoding="utf-8")
    finalize_source = campaign_source.split(
        "def _finalize_checkpoint_chain", 1
    )[1].split("def _read_l1b_qois", 1)[0]
    assert "read_bytes(" not in finalize_source
    assert "load_checkpoint_bundle(" in finalize_source
    assert "checkpoint_metadata_summary(" in finalize_source


def test_artifact_and_viewer_are_replayable_and_tamper_evident(
    authoritative_artifact,
) -> None:
    artifact = authoritative_artifact
    assert replay_artifact(artifact)["passed"]
    assert replay_artifact(artifact)["acceptance_authority"] == "recomputed"
    viewer = viewer_contract(artifact)
    assert viewer["artifact_payload_sha256"] == artifact["integrity"]["payload_sha256"]
    tampered = copy.deepcopy(artifact)
    tampered["solution"]["a_phi_dofs_t_m"][3] += 1.0
    with pytest.raises(FEMValidationError, match="integrity"):
        validate_artifact(tampered)


def test_rehashed_topology_tampering_still_fails_validation(
    authoritative_artifact,
) -> None:
    tampered = copy.deepcopy(authoritative_artifact)
    tampered["mesh"]["p2_nodes_rz_m"][-1][0] += 1.0e-4
    tampered["anchors"]["mesh_sha256"] = "a" * 64
    _reseal(tampered)
    with pytest.raises(FEMValidationError, match="midpoint"):
        validate_artifact(tampered)


def test_rehashed_qoi_and_comparison_claims_fail_authoritative_replay(
    authoritative_artifact,
) -> None:
    tampered = copy.deepcopy(authoritative_artifact)
    qoi_key = next(iter(tampered["qois_bz_t"]))
    tampered["qois_bz_t"][qoi_key] += 0.1
    _reseal(tampered)
    with pytest.raises(FEMValidationError, match="QoI"):
        validate_artifact(tampered)


def test_resealed_dtype_shape_endian_descriptor_fails_authoritative_replay(
    authoritative_artifact,
) -> None:
    artifact = copy.deepcopy(authoritative_artifact)
    artifact["acceptance_evidence"]["array_contract"][
        "solution.a_phi_dofs_t_m"
    ]["dtype"] = ">f8"
    _reseal(artifact)
    with pytest.raises(FEMValidationError, match="dtype/shape/endian"):
        validate_artifact(artifact)


def test_resealed_fabricated_prior_levels_and_order_fail_authoritative_replay(
    authoritative_artifact,
) -> None:
    artifact = copy.deepcopy(authoritative_artifact)
    key = next(
        name for name in artifact["qois_bz_t"] if name.endswith("-bore-average")
    )
    value = artifact["qois_bz_t"][key]
    values = (1.006 * value, 1.002 * value, value)
    levels = [
        {
            "qois_bz_t": {key: level_value},
            "resolution": {"qoi_h_m": {key: h}},
            "adjacent_area_size_growth": 1.0,
        }
        for level_value, h in zip(values, (4.0, 2.0, 1.0))
    ]
    changes = [
        {key: abs(right - left) / max(abs(left), abs(right))}
        for left, right in zip(values, values[1:])
    ]
    order = log(
        abs(values[0] - values[1]) / abs(values[1] - values[2])
    ) / log(2.0)
    artifact["acceptance_evidence"]["level_evidence"] = levels
    artifact["convergence"] = {
        "acceptance_qois": [key],
        "successive_volume_qoi_relative_changes": changes,
        "observed_orders_from_actual_qoi_h": {key: order},
        "two_successive_less_than_one_percent": True,
        "stable_positive_order": True,
        "adjacent_size_growth_gate": True,
        "phase_matched_domain_expansion_gate": False,
        "less_than_one_percent_reached": False,
    }
    _reseal(artifact)
    with pytest.raises(FEMValidationError, match="complete checkpoint anchor"):
        validate_artifact(artifact)
