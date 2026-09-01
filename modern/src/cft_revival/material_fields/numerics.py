"""Conservative matrix-free L1b finite-volume solver for ``psi=r*A_phi``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from math import fsum, isfinite, pi, sqrt
from pathlib import Path
from typing import Sequence

from cft_revival.fields import FieldMap
from cft_revival.fields.numerics import finalize_field_map

from .models import (
    MaterialFieldConvergenceError,
    MaterialFieldResult,
    MaterialFieldValidationError,
    MaterialSolveConfig,
    MaterialSolverDiagnostics,
    RasterizedMaterialProblem,
)


def _config_sha256(config: MaterialSolveConfig, backend: str) -> str:
    return hashlib.sha256(_config_json(config, backend).encode("utf-8")).hexdigest()


def _config_json(config: MaterialSolveConfig, backend: str) -> str:
    return json.dumps(
        {"backend": backend, "config": asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _implementation_sha256(*filenames: str) -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for filename in sorted(filenames):
        digest.update(filename.encode("utf-8"))
        digest.update((root / filename).read_bytes())
    return digest.hexdigest()


def _harmonic(left: float, right: float) -> float:
    smaller, larger = (left, right) if left <= right else (right, left)
    value = smaller / (0.5 + 0.5 * (smaller / larger))
    if not isfinite(value) or value <= 0.0:
        raise MaterialFieldValidationError("harmonic face reluctivity is invalid")
    return value


def apply_material_operator(
    problem: RasterizedMaterialProblem, vector: Sequence[float]
) -> list[float]:
    """Apply ``-div((nu/r) grad psi)`` using harmonic face reluctivity."""

    domain = problem.domain
    nr, nz = domain.shape
    if len(vector) != nr * nz:
        raise MaterialFieldValidationError("operator vector shape mismatch")
    output = [0.0] * len(vector)
    dr2 = domain.dr_m**2
    dz2 = domain.dz_m**2
    radial_faces = problem.radial_face_reluctivity_per_m_h
    axial_faces = problem.axial_face_reluctivity_per_m_h
    for i in range(1, nr - 1):
        radius = i * domain.dr_m
        for j in range(1, nz - 1):
            k = i * nz + j
            ar_minus = radial_faces[(i - 1) * nz + j] / (
                radius - 0.5 * domain.dr_m
            )
            ar_plus = radial_faces[i * nz + j] / (
                radius + 0.5 * domain.dr_m
            )
            az_minus = axial_faces[i * (nz - 1) + j - 1] / radius
            az_plus = axial_faces[i * (nz - 1) + j] / radius
            output[k] = (
                (ar_minus + ar_plus) * vector[k] / dr2
                - ar_minus * vector[k - nz] / dr2
                - ar_plus * vector[k + nz] / dr2
                + (az_minus + az_plus) * vector[k] / dz2
                - az_minus * vector[k - 1] / dz2
                - az_plus * vector[k + 1] / dz2
            )
            if problem.outer_boundary_kind == "dipole_robin_psi":
                if i == nr - 2:
                    output[k] -= (
                        ar_plus * problem.robin_radial_q[j] * vector[k] / dr2
                    )
                if j == 1:
                    output[k] -= (
                        az_minus * problem.robin_z_min_q[i] * vector[k] / dz2
                    )
                if j == nz - 2:
                    output[k] -= (
                        az_plus * problem.robin_z_max_q[i] * vector[k] / dz2
                    )
    if any(not isfinite(value) for value in output):
        raise MaterialFieldValidationError("material operator result is unrepresentable")
    return output


def minimum_operator_eigenvalue(
    problem: RasterizedMaterialProblem, *, maximum_unknowns: int = 1024
) -> float:
    """Return the exact small-grid interior eigenvalue used for definiteness audits."""
    import numpy as np

    nr, nz = problem.domain.shape
    active = tuple(
        i * nz + j
        for i in range(1, nr - 1)
        for j in range(1, nz - 1)
    )
    if len(active) > maximum_unknowns:
        raise MaterialFieldValidationError(
            "exact eigen audit is restricted to small verification grids"
        )
    matrix = np.empty((len(active), len(active)), dtype=np.float64)
    for column, index in enumerate(active):
        basis = [0.0] * (nr * nz)
        basis[index] = 1.0
        applied = apply_material_operator(problem, basis)
        matrix[:, column] = [applied[row] for row in active]
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > 1.0e-10 * max(1.0, float(np.max(np.abs(matrix)))):
        raise MaterialFieldValidationError("interior operator eigen audit is not symmetric")
    return float(np.linalg.eigvalsh(matrix)[0])


def material_operator_diagonal(problem: RasterizedMaterialProblem) -> tuple[float, ...]:
    domain = problem.domain
    nr, nz = domain.shape
    diagonal = [1.0] * (nr * nz)
    radial_faces = problem.radial_face_reluctivity_per_m_h
    axial_faces = problem.axial_face_reluctivity_per_m_h
    for i in range(1, nr - 1):
        radius = i * domain.dr_m
        for j in range(1, nz - 1):
            k = i * nz + j
            diagonal[k] = (
                (
                    radial_faces[(i - 1) * nz + j] / (radius - 0.5 * domain.dr_m)
                    + radial_faces[i * nz + j] / (radius + 0.5 * domain.dr_m)
                )
                / domain.dr_m**2
                + (
                    axial_faces[i * (nz - 1) + j - 1] / radius
                    + axial_faces[i * (nz - 1) + j] / radius
                )
                / domain.dz_m**2
            )
            if problem.outer_boundary_kind == "dipole_robin_psi":
                if i == nr - 2:
                    diagonal[k] -= (
                        radial_faces[i * nz + j]
                        / (radius + 0.5 * domain.dr_m)
                        * problem.robin_radial_q[j]
                        / domain.dr_m**2
                    )
                if j == 1:
                    diagonal[k] -= (
                        axial_faces[i * (nz - 1)]
                        / radius
                        * problem.robin_z_min_q[i]
                        / domain.dz_m**2
                    )
                if j == nz - 2:
                    diagonal[k] -= (
                        axial_faces[i * (nz - 1) + j]
                        / radius
                        * problem.robin_z_max_q[i]
                        / domain.dz_m**2
                    )
    return tuple(diagonal)


def assemble_rhs(problem: RasterizedMaterialProblem) -> tuple[float, ...]:
    """Assemble ``J_phi-div(nu*Br_z, -nu*Br_r)`` without PM double counting."""

    domain = problem.domain
    nr, nz = domain.shape
    gr_faces = problem.remanence_g_r_face_a_per_m
    gz_faces = problem.remanence_g_z_face_a_per_m
    rhs = [0.0] * (nr * nz)
    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            k = i * nz + j
            div_g = (
                (
                    gr_faces[i * nz + j]
                    - gr_faces[(i - 1) * nz + j]
                )
                / domain.dr_m
                + (
                    gz_faces[i * (nz - 1) + j]
                    - gz_faces[i * (nz - 1) + j - 1]
                )
                / domain.dz_m
            )
            rhs[k] = (
                problem.free_current_phi_a_per_m2[k]
                + problem.pm_bound_current_phi_a_per_m2[k]
                - div_g
            )
    if any(not isfinite(value) for value in rhs):
        raise MaterialFieldValidationError("assembled source is nonfinite")
    return tuple(rhs)


def _dot(problem: RasterizedMaterialProblem, left: Sequence[float], right: Sequence[float]) -> float:
    nr, nz = problem.domain.shape
    return fsum(
        left[i * nz + j] * right[i * nz + j]
        for i in range(1, nr - 1)
        for j in range(1, nz - 1)
    )


def _true_residual(
    problem: RasterizedMaterialProblem, rhs: Sequence[float], solution: Sequence[float]
) -> list[float]:
    applied = apply_material_operator(problem, solution)
    return [rhs[index] - applied[index] for index in range(len(rhs))]


def _rows(values: Sequence, shape: tuple[int, int]):
    nr, nz = shape
    return tuple(tuple(values[i * nz + j] for j in range(nz)) for i in range(nr))


def _apply_outer_boundary_values(
    problem: RasterizedMaterialProblem, solution: Sequence[float]
) -> list[float]:
    values = list(solution)
    if problem.outer_boundary_kind != "dipole_robin_psi":
        return values
    nr, nz = problem.domain.shape
    for j in range(1, nz - 1):
        values[(nr - 1) * nz + j] = (
            problem.robin_radial_q[j] * values[(nr - 2) * nz + j]
        )
    for i in range(1, nr - 1):
        values[i * nz] = problem.robin_z_min_q[i] * values[i * nz + 1]
        values[i * nz + nz - 1] = (
            problem.robin_z_max_q[i] * values[i * nz + nz - 2]
        )
    for j, axial_q in (
        (0, problem.robin_z_min_q[nr - 1]),
        (nz - 1, problem.robin_z_max_q[nr - 1]),
    ):
        radial_prediction = problem.robin_radial_q[j] * values[(nr - 2) * nz + j]
        axial_prediction = axial_q * values[(nr - 1) * nz + (1 if j == 0 else nz - 2)]
        values[(nr - 1) * nz + j] = 0.5 * (
            radial_prediction + axial_prediction
        )
    return values


def solve_material_problem_cpu(
    problem: RasterizedMaterialProblem,
    config: MaterialSolveConfig = MaterialSolveConfig(),
    *,
    raise_on_nonconvergence: bool = True,
) -> MaterialFieldResult:
    """Solve with binary64 Jacobi-PCG and mandatory true-residual acceptance."""

    if problem.feature_effective_cells:
        observed = min(
            min(radial, axial)
            for _, radial, axial in problem.feature_effective_cells
        )
        if (
            observed < config.minimum_effective_feature_cells * (1.0 - 1.0e-12)
            and not config.allow_underresolved_screening
        ):
            raise MaterialFieldValidationError(
                f"minimum feature resolution {observed:.6g} cells is below "
                f"{config.minimum_effective_feature_cells:.6g}"
            )
    rhs = assemble_rhs(problem)
    count = len(rhs)
    x = [0.0] * count
    residual = list(rhs)
    initial = sqrt(max(0.0, _dot(problem, residual, residual)))
    history = [initial]
    linear = config.linear
    threshold = max(linear.absolute_tolerance, linear.relative_tolerance * initial)
    diagonal = material_operator_diagonal(problem)
    iterations = 0
    restart_count = 0
    converged = initial == 0.0
    if not converged:
        preconditioned = [residual[k] / diagonal[k] for k in range(count)]
        direction = preconditioned.copy()
        rho = _dot(problem, residual, preconditioned)
        for iteration in range(1, linear.max_iterations + 1):
            applied = apply_material_operator(problem, direction)
            denominator = _dot(problem, direction, applied)
            if not isfinite(denominator) or denominator <= 0.0:
                raise MaterialFieldConvergenceError("PCG lost positive definiteness")
            alpha = rho / denominator
            for k in range(count):
                x[k] += alpha * direction[k]
                residual[k] -= alpha * applied[k]
            iterations = iteration
            recurrence = sqrt(max(0.0, _dot(problem, residual, residual)))
            if iteration % linear.residual_history_stride == 0:
                history.append(recurrence)
            if not isfinite(recurrence) or any(not isfinite(value) for value in x):
                raise MaterialFieldConvergenceError("PCG state became nonfinite")
            if recurrence <= threshold:
                residual = _true_residual(problem, rhs, x)
                true_norm = sqrt(max(0.0, _dot(problem, residual, residual)))
                history.append(true_norm)
                if true_norm <= threshold:
                    converged = True
                    break
                if restart_count >= linear.max_true_residual_restarts:
                    break
                restart_count += 1
                preconditioned = [residual[k] / diagonal[k] for k in range(count)]
                direction = preconditioned.copy()
                rho = _dot(problem, residual, preconditioned)
                continue
            preconditioned = [residual[k] / diagonal[k] for k in range(count)]
            new_rho = _dot(problem, residual, preconditioned)
            beta = new_rho / rho
            for k in range(count):
                direction[k] = preconditioned[k] + beta * direction[k]
            rho = new_rho

    residual = _true_residual(problem, rhs, x)
    final = sqrt(max(0.0, _dot(problem, residual, residual)))
    converged = converged and final <= threshold
    if history[-1] != final:
        history.append(final)
    if not converged and raise_on_nonconvergence:
        raise MaterialFieldConvergenceError(
            f"material PCG did not converge: true residual {final:.6e} > {threshold:.6e}"
        )
    field_solution = _apply_outer_boundary_values(problem, x)
    base = finalize_field_map(
        problem.domain,
        field_solution,
        converged=converged,
        iterations=iterations,
        initial_residual=initial,
        final_residual=final,
        residual_history=history,
        backend="material_fields:python",
        true_residual_restarts=restart_count,
        stagnation_detected=not converged,
    )
    field = FieldMap(
        r_m=base.r_m,
        z_m=base.z_m,
        psi_wb=base.psi_wb,
        b_r_t=base.b_r_t,
        b_z_t=base.b_z_t,
        diagnostics=base.diagnostics,
        level="L1b",
    )
    applied = apply_material_operator(problem, x)
    magnetic = pi * problem.domain.dr_m * problem.domain.dz_m * _dot(problem, x, applied)
    source = pi * problem.domain.dr_m * problem.domain.dz_m * _dot(problem, x, rhs)
    energy_error = abs(magnetic - source) / max(abs(magnetic), abs(source), 1.0e-300)
    diagnostics = MaterialSolverDiagnostics(
        converged,
        iterations,
        initial,
        final,
        0.0 if initial == 0.0 else final / initial,
        tuple(history),
        restart_count,
        "material_fields:python",
        magnetic,
        source,
        energy_error,
        _config_sha256(config, "material_fields:python"),
        _implementation_sha256("adapters.py", "models.py", "numerics.py"),
        _config_json(config, "material_fields:python"),
        0,
        1,
    )
    shape = problem.domain.shape
    return MaterialFieldResult(
        field,
        diagnostics,
        _rows(problem.material_ids, shape),
        _rows(problem.free_current_phi_a_per_m2, shape),
        _rows(problem.remanence_r_t, shape),
        _rows(problem.remanence_z_t, shape),
        problem,
    )
