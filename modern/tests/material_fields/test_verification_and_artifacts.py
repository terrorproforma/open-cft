from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from math import ceil, cos, hypot, log, pi, sqrt

import pytest

from cft_revival.fields import AxisymmetricDomain, MU0_H_PER_M, SolverConfig
from cft_revival.geometry import (
    PermanentMagnetAuthority,
    PermanentMagnetRepresentationPlan,
    compact_high_gradient_stack,
)
from cft_revival.material_fields import (
    MaterialFieldValidationError,
    MaterialSolveConfig,
    RasterizedMaterialProblem,
    adapt_geometry,
    apply_material_operator,
    design_domain,
    device_available,
    material_field_artifact,
    max_result_difference,
    replay_raw_run,
    solve_material_problem_cpu,
    solve_material_problem_warp,
    validate_artifact,
    validate_viewer_contract,
    viewer_contract,
)
from cft_revival.material_fields.acceptance import _bore_average

SCREENING_CONFIG = MaterialSolveConfig(allow_underresolved_screening=True)


def test_composite_bore_quadrature_is_exact_for_bilinear_field() -> None:
    domain = {
        "radius_m": 1.0, "z_min_m": -0.4, "z_max_m": 0.6,
        "radial_intervals": 7, "axial_intervals": 9,
        "dr_m": 1.0 / 7.0, "dz_m": 1.0 / 9.0,
    }
    rows = tuple(
        tuple(
            1.0 + 2.0 * (i / 7.0) + 3.0 * (-0.4 + j / 9.0)
            + 4.0 * (i / 7.0) * (-0.4 + j / 9.0)
            for j in range(10)
        )
        for i in range(8)
    )
    radius, z0, z1 = 0.63, -0.23, 0.31
    z_mean = 0.5 * (z0 + z1)
    radial_mean = 2.0 * radius / 3.0
    expected = 1.0 + 2.0 * radial_mean + 3.0 * z_mean + 4.0 * radial_mean * z_mean
    assert _bore_average(domain, rows, radius, z0, z1) == pytest.approx(
        expected, rel=2.0e-15
    )


def test_composite_bore_quadrature_handles_shifted_steep_cell_field() -> None:
    domain = {
        "radius_m": 1.0, "z_min_m": -0.37, "z_max_m": 0.63,
        "radial_intervals": 8, "axial_intervals": 11,
        "dr_m": 0.125, "dz_m": 1.0 / 11.0,
    }
    rows = tuple(
        tuple(
            (1.0 + 20.0 * (i / 8.0) ** 4)
            * (1.0 + 15.0 * abs(-0.37 + j / 11.0 - 0.08) ** 3)
            for j in range(12)
        )
        for i in range(9)
    )
    radius, z0, z1 = 0.57, -0.23, 0.31
    observed = _bore_average(domain, rows, radius, z0, z1)
    # Independent dense midpoint integration of the represented bilinear field.
    samples = 1400
    numerator = denominator = 0.0
    for ir in range(samples):
        radial = radius * (ir + 0.5) / samples
        for iz in range(samples // 2):
            axial = z0 + (z1 - z0) * (iz + 0.5) / (samples // 2)
            i = min(7, int(radial / domain["dr_m"]))
            j = min(10, int((axial - domain["z_min_m"]) / domain["dz_m"]))
            tr = radial / domain["dr_m"] - i
            tz = (axial - domain["z_min_m"]) / domain["dz_m"] - j
            value = (
                (1 - tr) * (1 - tz) * rows[i][j]
                + tr * (1 - tz) * rows[i + 1][j]
                + (1 - tr) * tz * rows[i][j + 1]
                + tr * tz * rows[i + 1][j + 1]
            )
            numerator += radial * value
            denominator += radial
    assert observed == pytest.approx(numerator / denominator, rel=2.0e-6)


def _piecewise_problem(domain: AxisymmetricDomain) -> RasterizedMaterialProblem:
    nr, nz = domain.shape
    count = nr * nz
    nu = []
    ids = []
    for i in range(nr):
        for j in range(nz):
            value = 1.0 if j <= domain.axial_intervals // 2 else 5.0
            nu.append(value)
            ids.append("mu-left" if value == 1.0 else "mu-right")
    return RasterizedMaterialProblem(
        "piecewise-manufactured",
        domain,
        "2" * 64,
        "3" * 64,
        "equivalent_bound_current",
        tuple(ids),
        (None,) * count,
        (None,) * count,
        tuple(nu),
        (0.0,) * count,
        (0.0,) * count,
        (0.0,) * count,
        (),
        (0.0, 0.0),
    )


def _flatten(rows):
    return tuple(value for row in rows for value in row)


def test_manufactured_piecewise_mu_solution_and_true_residual() -> None:
    domain = AxisymmetricDomain(1.0, -1.0, 1.0, 20, 40)
    problem = _piecewise_problem(domain)
    exact = tuple(
        (r * r - r**4) * cos(0.5 * pi * z) ** 2
        for r in (i * domain.dr_m for i in range(domain.shape[0]))
        for z in (
            domain.z_min_m + j * domain.dz_m for j in range(domain.shape[1])
        )
    )
    rhs = tuple(apply_material_operator(problem, exact))
    manufactured = replace(problem, free_current_phi_a_per_m2=rhs)
    result = solve_material_problem_cpu(
        manufactured,
        MaterialSolveConfig(SolverConfig(relative_tolerance=2.0e-11)),
    )
    error = sqrt(sum((a - b) ** 2 for a, b in zip(_flatten(result.field.psi_wb), exact)))
    scale = sqrt(sum(value**2 for value in exact))
    assert error / scale < 3.0e-10
    assert result.diagnostics.relative_true_residual_l2 < 2.0e-11
    assert result.diagnostics.energy_balance_relative < 3.0e-11


def test_uniform_manufactured_mesh_convergence_is_second_order() -> None:
    errors = []
    for intervals in (12, 24, 48):
        domain = AxisymmetricDomain(1.0, -1.0, 1.0, intervals, 2 * intervals)
        count = domain.shape[0] * domain.shape[1]
        problem = _piecewise_problem(domain)
        problem = replace(
            problem,
            reluctivity_per_m_h=(2.0,) * count,
            radial_face_reluctivity_per_m_h=(),
            axial_face_reluctivity_per_m_h=(),
        )
        exact = []
        source = []
        for i in range(domain.shape[0]):
            r = i * domain.dr_m
            radial = r * r - r**4
            for j in range(domain.shape[1]):
                z = domain.z_min_m + j * domain.dz_m
                q = cos(0.5 * pi * z) ** 2
                exact.append(radial * q)
                source.append(
                    2.0
                    * (
                        8.0 * r * q
                        + (radial / r if r else 0.0) * 0.5 * pi**2 * cos(pi * z)
                    )
                )
        solved = solve_material_problem_cpu(
            replace(problem, free_current_phi_a_per_m2=tuple(source)),
            MaterialSolveConfig(SolverConfig(relative_tolerance=1.0e-11)),
        )
        actual = _flatten(solved.field.psi_wb)
        errors.append(
            sqrt(sum((a - b) ** 2 for a, b in zip(actual, exact)))
            / sqrt(sum(value**2 for value in exact))
        )
    orders = [log(coarse / fine, 2.0) for coarse, fine in zip(errors, errors[1:])]
    assert min(orders) > 1.8


@pytest.mark.skipif(not device_available("cuda"), reason="CUDA parity evidence unavailable")
def test_design_artifact_is_hash_anchored_complete_and_tamper_evident() -> None:
    geometry = compact_high_gradient_stack()
    domain = design_domain(geometry, radial_intervals=32, axial_intervals=64, padding_factor=3.0)
    problem = adapt_geometry(geometry, domain)
    result = solve_material_problem_cpu(problem, SCREENING_CONFIG)
    def expanded(factor):
        provisional = design_domain(
            geometry, radial_intervals=4, axial_intervals=4, padding_factor=factor
        )
        nr = ceil(provisional.radius_m / domain.dr_m)
        lower = ceil((domain.z_min_m - provisional.z_min_m) / domain.dz_m)
        upper = ceil((provisional.z_max_m - domain.z_max_m) / domain.dz_m)
        return AxisymmetricDomain(
            nr * domain.dr_m,
            domain.z_min_m - lower * domain.dz_m,
            domain.z_max_m + upper * domain.dz_m,
            nr,
            domain.axial_intervals + lower + upper,
        )

    expansions = tuple(
        solve_material_problem_cpu(
            adapt_geometry(
                geometry,
                expanded(factor),
            ),
            SCREENING_CONFIG,
        )
        for factor in (4.5, 6.75)
    )
    fine = solve_material_problem_cpu(
        adapt_geometry(
            geometry,
            AxisymmetricDomain(domain.radius_m, domain.z_min_m, domain.z_max_m, 40, 80),
        ),
        SCREENING_CONFIG,
    )
    third = solve_material_problem_cpu(
        adapt_geometry(
            geometry,
            AxisymmetricDomain(
                domain.radius_m, domain.z_min_m, domain.z_max_m, 44, 88
            ),
        ),
        SCREENING_CONFIG,
    )
    alignment = solve_material_problem_cpu(
        adapt_geometry(
            geometry,
            AxisymmetricDomain(
                domain.radius_m,
                domain.z_min_m + 0.25 * domain.dz_m,
                domain.z_max_m + 0.25 * domain.dz_m,
                33,
                65,
            ),
        ),
        SCREENING_CONFIG,
    )
    equivalent_geometry = replace(
        geometry,
        permanent_magnet_plan=PermanentMagnetRepresentationPlan(
            f"{geometry.config_id}-{PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT.value}-v1",
            PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT,
        ),
    )
    equivalent = solve_material_problem_cpu(
        adapt_geometry(equivalent_geometry, domain), SCREENING_CONFIG
    )
    equivalent_fine = solve_material_problem_cpu(
        adapt_geometry(equivalent_geometry, fine.problem.domain), SCREENING_CONFIG
    )
    parity_cuda = solve_material_problem_warp(
        problem, device="cuda", config=SCREENING_CONFIG
    )
    artifact = material_field_artifact(
        result,
        domain_expansions=expansions,
        mesh_fine=fine,
        mesh_third=third,
        alignment_sweeps=(alignment,),
        equivalent_base=equivalent,
        equivalent_fine=equivalent_fine,
        parity_cpu=result,
        parity_cuda=parity_cuda,
        downsample_stride=3,
    )
    validate_artifact(artifact, require_accepted=False)
    wrong_role_order = copy.deepcopy(artifact)
    wrong_role_order["acceptance"]["raw_runs"][0:2] = reversed(
        wrong_role_order["acceptance"]["raw_runs"][0:2]
    )
    with pytest.raises(
        MaterialFieldValidationError, match="role order/cardinality/multiplicity"
    ):
        validate_artifact(wrong_role_order, require_accepted=False)
    inconsistent_residual = copy.deepcopy(
        artifact["acceptance"]["raw_runs"][0]["raw"]
    )
    inconsistent_residual["diagnostics"]["relative_true_residual_l2"] *= 2.0
    assert not replay_raw_run(
        inconsistent_residual,
        backend=artifact["acceptance"]["raw_runs"][0]["backend"],
    ).passed
    assert artifact["anchors"]["geometry_sha256"] == geometry.canonical_sha256
    assert artifact["classification"].startswith("hypothetical")
    assert set(artifact["full_field_map"]) >= {
        "psi_wb",
        "b_r_t",
        "b_z_t",
        "b_magnitude_t",
        "material_id",
        "free_current_phi_a_per_m2",
        "pm_bound_current_phi_a_per_m2",
        "remanence_z_t",
    }
    viewer = viewer_contract(artifact)
    validate_viewer_contract(viewer, artifact=artifact)
    assert viewer["artifact_payload_sha256"] == artifact["integrity"]["payload_sha256"]
    altered = {**artifact, "classification": "validated"}
    with pytest.raises(MaterialFieldValidationError, match="unsupported"):
        validate_artifact(altered)
    bad_algorithm = {
        **artifact,
        "integrity": {**artifact["integrity"], "algorithm": "md5"},
    }
    with pytest.raises(MaterialFieldValidationError, match="integrity"):
        validate_artifact(bad_algorithm, require_accepted=False)
    nested_missing = {
        **artifact,
        "summary": {
            key: value for key, value in artifact["summary"].items() if key != "axis_bz_peak"
        },
    }
    with pytest.raises(MaterialFieldValidationError, match="summary"):
        validate_artifact(nested_missing, require_accepted=False)
    promoted = copy.deepcopy(artifact)
    promoted["acceptance"]["status"] = "ACCEPTED_PUBLICATION_EVIDENCE"
    for gate in promoted["acceptance"]["gates"]:
        gate["status"] = "PASS"
        gate["measured_value"] = (
            gate["threshold"]
            if gate["gate_id"] in {
                "required_domain_expansions",
                "minimum_base_padding",
                "domain_expansion_factor",
                "minimum_effective_feature_cells",
            }
            else 0.0
        )
    promoted["acceptance"]["warning_codes"] = []
    promoted["summary"]["warning_codes"] = []
    payload = {key: value for key, value in promoted.items() if key != "integrity"}
    promoted["integrity"]["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(MaterialFieldValidationError, match="derived|recomputed"):
        validate_artifact(promoted, require_accepted=False)
    replay_tamper = copy.deepcopy(artifact)
    run = replay_tamper["acceptance"]["raw_runs"][1]
    encoded = run["raw"]["solution"]["data_base64"]
    run["raw"]["solution"]["data_base64"] = (
        ("A" if encoded[0] != "A" else "B") + encoded[1:]
    )
    run_anchors = {
        "study_id": run["study_id"],
        "role": run["role"],
        "config_sha256": run["config_sha256"],
            "solver_config_identity_sha256": run["solver_config_identity_sha256"],
        "geometry_sha256": run["geometry_sha256"],
        "material_sha256": run["material_sha256"],
        "design_geometry_sha256": run["design_geometry_sha256"],
        "material_registry_sha256": run["material_registry_sha256"],
        "implementation_sha256": run["implementation_sha256"],
        "evidence_implementation_sha256": run["evidence_implementation_sha256"],
        "backend": run["backend"],
        "grid_sha256": run["grid_sha256"],
        "domain_sha256": run["domain_sha256"],
        "problem_sha256": run["problem_sha256"],
        "raw": run["raw"],
    }
    run["run_sha256"] = hashlib.sha256(
        json.dumps(
            run_anchors, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    next(
        study
        for study in replay_tamper["acceptance"]["studies"]
        if study["study_id"] == run["study_id"]
    )["run_sha256"] = run["run_sha256"]
    payload = {
        key: value for key, value in replay_tamper.items() if key != "integrity"
    }
    replay_tamper["integrity"]["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    with pytest.raises(MaterialFieldValidationError, match="solution|replay"):
        validate_artifact(replay_tamper, require_accepted=False)
    with pytest.raises(MaterialFieldValidationError, match="referenced artifact"):
        validate_viewer_contract(viewer)


@pytest.mark.skipif(not device_available("cpu"), reason="optional Warp unavailable")
def test_cpu_warp_cpu_parity() -> None:
    geometry = compact_high_gradient_stack()
    problem = adapt_geometry(
        geometry,
        design_domain(geometry, radial_intervals=16, axial_intervals=32, padding_factor=0.5),
    )
    cpu = solve_material_problem_cpu(problem, SCREENING_CONFIG)
    warp = solve_material_problem_warp(
        problem, device="cpu", config=SCREENING_CONFIG
    )
    assert warp.diagnostics.host_synchronization_count <= (
        warp.diagnostics.iterations
        // warp.diagnostics.convergence_check_interval
        + 8
    )
    assert warp.diagnostics.host_synchronization_count * 10 < max(
        1, 2 * warp.diagnostics.iterations
    )
    differences = max_result_difference(cpu, warp)
    assert differences["psi_scale_relative"] < 2.0e-8
    assert differences["br_scale_relative"] < 2.0e-8
    assert differences["bz_scale_relative"] < 2.0e-8
    assert all(
        hypot(br, bz) < float("inf")
        for br_row, bz_row in zip(warp.field.b_r_t, warp.field.b_z_t)
        for br, bz in zip(br_row, bz_row)
    )


def test_extreme_reluctivity_remains_finite_or_fails_closed() -> None:
    domain = AxisymmetricDomain(1.0, -1.0, 1.0, 8, 8)
    problem = _piecewise_problem(domain)
    extreme = replace(
        problem,
        reluctivity_per_m_h=tuple(
            1.0 / MU0_H_PER_M if index % 2 else 1.0e308
            for index in range(len(problem.reluctivity_per_m_h))
        ),
    )
    applied = apply_material_operator(
        extreme, (1.0,) * len(extreme.reluctivity_per_m_h)
    )
    assert all(value == value and abs(value) < float("inf") for value in applied)
