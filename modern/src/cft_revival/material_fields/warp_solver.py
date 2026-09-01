"""Optional Warp operator and solver parity path for L1b."""

from __future__ import annotations

from math import fsum, isfinite, pi, sqrt

from cft_revival.fields import FieldDeviceError, FieldMap
from cft_revival.fields.numerics import finalize_field_map

from .models import (
    MaterialFieldConvergenceError,
    MaterialFieldResult,
    MaterialFieldValidationError,
    MaterialSolveConfig,
    MaterialSolverDiagnostics,
    RasterizedMaterialProblem,
)
from .numerics import (
    _config_sha256,
    _config_json,
    _dot,
    _implementation_sha256,
    _apply_outer_boundary_values,
    _rows,
    assemble_rhs,
    material_operator_diagonal,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None  # type: ignore[assignment]


if wp is not None:

    @wp.kernel
    def _material_operator_kernel(
        x: wp.array(dtype=wp.float64),
        radial_faces: wp.array(dtype=wp.float64),
        axial_faces: wp.array(dtype=wp.float64),
        robin_radial: wp.array(dtype=wp.float64),
        robin_z_min: wp.array(dtype=wp.float64),
        robin_z_max: wp.array(dtype=wp.float64),
        use_robin: int,
        out: wp.array(dtype=wp.float64),
        nr: int,
        nz: int,
        dr: wp.float64,
        dz: wp.float64,
    ):
        k = wp.tid()
        i = k // nz
        j = k - i * nz
        if i > 0 and i < nr - 1 and j > 0 and j < nz - 1:
            r = wp.float64(i) * dr
            hm = radial_faces[(i - 1) * nz + j]
            hp = radial_faces[i * nz + j]
            zm = axial_faces[i * (nz - 1) + j - 1]
            zp = axial_faces[i * (nz - 1) + j]
            am = hm / (r - wp.float64(0.5) * dr)
            ap = hp / (r + wp.float64(0.5) * dr)
            bm = zm / r
            bp = zp / r
            out[k] = (
                ((am + ap) * x[k] - am * x[k - nz] - ap * x[k + nz]) / (dr * dr)
                + ((bm + bp) * x[k] - bm * x[k - 1] - bp * x[k + 1]) / (dz * dz)
            )
            if use_robin == 1:
                if i == nr - 2:
                    out[k] = out[k] - ap * robin_radial[j] * x[k] / (dr * dr)
                if j == 1:
                    out[k] = out[k] - bm * robin_z_min[i] * x[k] / (dz * dz)
                if j == nz - 2:
                    out[k] = out[k] - bp * robin_z_max[i] * x[k] / (dz * dz)
        else:
            out[k] = wp.float64(0.0)

    @wp.kernel
    def _dot_kernel(
        left: wp.array(dtype=wp.float64),
        right: wp.array(dtype=wp.float64),
        result: wp.array(dtype=wp.float64),
        nr: int,
        nz: int,
    ):
        k = wp.tid()
        i = k // nz
        j = k - i * nz
        if i > 0 and i < nr - 1 and j > 0 and j < nz - 1:
            wp.atomic_add(result, 0, left[k] * right[k])

    @wp.kernel
    def _pcg_update_kernel(
        x: wp.array(dtype=wp.float64),
        residual: wp.array(dtype=wp.float64),
        direction: wp.array(dtype=wp.float64),
        applied: wp.array(dtype=wp.float64),
        rho: wp.array(dtype=wp.float64),
        denominator: wp.array(dtype=wp.float64),
    ):
        k = wp.tid()
        alpha = rho[0] / denominator[0]
        x[k] = x[k] + alpha * direction[k]
        residual[k] = residual[k] - alpha * applied[k]

    @wp.kernel
    def _jacobi_kernel(
        residual: wp.array(dtype=wp.float64),
        diagonal: wp.array(dtype=wp.float64),
        preconditioned: wp.array(dtype=wp.float64),
    ):
        k = wp.tid()
        preconditioned[k] = residual[k] / diagonal[k]

    @wp.kernel
    def _direction_kernel(
        direction: wp.array(dtype=wp.float64),
        preconditioned: wp.array(dtype=wp.float64),
        rho_new: wp.array(dtype=wp.float64),
        rho_old: wp.array(dtype=wp.float64),
    ):
        k = wp.tid()
        beta = rho_new[0] / rho_old[0]
        direction[k] = preconditioned[k] + beta * direction[k]

    @wp.kernel
    def _copy_scalar_kernel(
        source: wp.array(dtype=wp.float64),
        target: wp.array(dtype=wp.float64),
    ):
        target[0] = source[0]

    @wp.kernel
    def _residual_kernel(
        rhs: wp.array(dtype=wp.float64),
        applied: wp.array(dtype=wp.float64),
        residual: wp.array(dtype=wp.float64),
    ):
        k = wp.tid()
        residual[k] = rhs[k] - applied[k]


def device_available(device: str) -> bool:
    if wp is None:
        return False
    try:
        wp.init()
        wp.get_device("cuda:0" if device == "cuda" else device)
    except (RuntimeError, ValueError):
        return False
    return True


def solve_material_problem_warp(
    problem: RasterizedMaterialProblem,
    *,
    device: str = "cpu",
    config: MaterialSolveConfig = MaterialSolveConfig(),
    raise_on_nonconvergence: bool = True,
) -> MaterialFieldResult:
    """Run host-controlled PCG with a float64 Warp matrix-free operator."""

    if wp is None:
        raise FieldDeviceError("NVIDIA Warp is unavailable")
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
    wp.init()
    requested = "cuda:0" if device == "cuda" else device
    try:
        resolved = wp.get_device(requested)
    except (RuntimeError, ValueError) as error:
        raise FieldDeviceError(f"Warp device {requested!r} is unavailable") from error
    count = len(problem.reluctivity_per_m_h)
    nr, nz = problem.domain.shape
    radial_faces_device = wp.array(
        problem.radial_face_reluctivity_per_m_h, dtype=wp.float64, device=resolved
    )
    axial_faces_device = wp.array(
        problem.axial_face_reluctivity_per_m_h, dtype=wp.float64, device=resolved
    )
    robin_radial_device = wp.array(problem.robin_radial_q, dtype=wp.float64, device=resolved)
    robin_z_min_device = wp.array(problem.robin_z_min_q, dtype=wp.float64, device=resolved)
    robin_z_max_device = wp.array(problem.robin_z_max_q, dtype=wp.float64, device=resolved)

    use_robin = 1 if problem.outer_boundary_kind == "dipole_robin_psi" else 0

    def apply_device(values, output):
        wp.launch(
            _material_operator_kernel,
            dim=count,
            inputs=[
                values,
                radial_faces_device,
                axial_faces_device,
                robin_radial_device,
                robin_z_min_device,
                robin_z_max_device,
                use_robin,
                output,
                nr,
                nz,
                problem.domain.dr_m,
                problem.domain.dz_m,
            ],
            device=resolved,
        )

    rhs = assemble_rhs(problem)
    rhs_device = wp.array(rhs, dtype=wp.float64, device=resolved)
    x_device = wp.zeros(count, dtype=wp.float64, device=resolved)
    residual_device = wp.array(rhs, dtype=wp.float64, device=resolved)
    diagonal_device = wp.array(
        material_operator_diagonal(problem), dtype=wp.float64, device=resolved
    )
    preconditioned = wp.empty_like(residual_device)
    direction = wp.empty_like(residual_device)
    applied_device = wp.empty_like(residual_device)
    reduction = wp.zeros(1, dtype=wp.float64, device=resolved)
    rho_device = wp.zeros(1, dtype=wp.float64, device=resolved)
    rho_new_device = wp.zeros(1, dtype=wp.float64, device=resolved)
    denominator_device = wp.zeros(1, dtype=wp.float64, device=resolved)
    host_synchronizations = 0

    def launch(kernel, inputs):
        wp.launch(kernel, dim=count, inputs=inputs, device=resolved)

    def dot_into(left, right, target) -> None:
        target.zero_()
        launch(_dot_kernel, [left, right, target, nr, nz])

    def scalar(target) -> float:
        nonlocal host_synchronizations
        host_synchronizations += 1
        return float(target.numpy()[0])

    dot_into(residual_device, residual_device, reduction)
    initial = sqrt(max(0.0, scalar(reduction)))
    history = [initial]
    linear = config.linear
    threshold = max(linear.absolute_tolerance, linear.relative_tolerance * initial)
    converged = initial == 0.0
    iterations = 0
    restart_count = 0
    if not converged:
        launch(_jacobi_kernel, [residual_device, diagonal_device, preconditioned])
        wp.copy(direction, preconditioned)
        dot_into(residual_device, preconditioned, rho_device)
        check_interval = max(25, linear.residual_history_stride)
        for iteration in range(1, linear.max_iterations + 1):
            apply_device(direction, applied_device)
            dot_into(direction, applied_device, denominator_device)
            launch(
                _pcg_update_kernel,
                [
                    x_device,
                    residual_device,
                    direction,
                    applied_device,
                    rho_device,
                    denominator_device,
                ],
            )
            iterations = iteration
            launch(_jacobi_kernel, [residual_device, diagonal_device, preconditioned])
            dot_into(residual_device, preconditioned, rho_new_device)
            launch(
                _direction_kernel,
                [direction, preconditioned, rho_new_device, rho_device],
            )
            wp.launch(
                _copy_scalar_kernel,
                dim=1,
                inputs=[rho_new_device, rho_device],
                device=resolved,
            )
            if iteration % check_interval != 0:
                continue
            dot_into(residual_device, residual_device, reduction)
            norm = sqrt(max(0.0, scalar(reduction)))
            history.append(norm)
            if not isfinite(norm):
                raise MaterialFieldConvergenceError("Warp PCG residual became nonfinite")
            if norm <= threshold:
                apply_device(x_device, applied_device)
                launch(_residual_kernel, [rhs_device, applied_device, residual_device])
                dot_into(residual_device, residual_device, reduction)
                norm = sqrt(max(0.0, scalar(reduction)))
                history.append(norm)
                if norm <= threshold:
                    converged = True
                    break
                if restart_count >= linear.max_true_residual_restarts:
                    break
                restart_count += 1
                launch(_jacobi_kernel, [residual_device, diagonal_device, preconditioned])
                wp.copy(direction, preconditioned)
                dot_into(residual_device, preconditioned, rho_device)
                continue
    apply_device(x_device, applied_device)
    launch(_residual_kernel, [rhs_device, applied_device, residual_device])
    dot_into(residual_device, residual_device, reduction)
    final = sqrt(max(0.0, scalar(reduction)))
    x = [float(value) for value in x_device.numpy()]
    host_synchronizations += 1
    applied = [float(value) for value in applied_device.numpy()]
    host_synchronizations += 1
    converged = converged and final <= threshold
    if history[-1] != final:
        history.append(final)
    if not converged and raise_on_nonconvergence:
        raise MaterialFieldConvergenceError(
            f"Warp material PCG true residual {final:.6e} > {threshold:.6e}"
        )
    backend = f"material_fields:warp:{resolved}"
    field_solution = _apply_outer_boundary_values(problem, x)
    base = finalize_field_map(
        problem.domain,
        field_solution,
        converged=converged,
        iterations=iterations,
        initial_residual=initial,
        final_residual=final,
        residual_history=history,
        backend=backend,
        true_residual_restarts=restart_count,
        stagnation_detected=not converged,
    )
    field = FieldMap(
        base.r_m, base.z_m, base.psi_wb, base.b_r_t, base.b_z_t, base.diagnostics, "L1b"
    )
    magnetic = pi * problem.domain.dr_m * problem.domain.dz_m * _dot(problem, x, applied)
    source = pi * problem.domain.dr_m * problem.domain.dz_m * _dot(problem, x, rhs)
    diagnostics = MaterialSolverDiagnostics(
        converged,
        iterations,
        initial,
        final,
        0.0 if initial == 0.0 else final / initial,
        tuple(history),
        restart_count,
        backend,
        magnetic,
        source,
        abs(magnetic - source) / max(abs(magnetic), abs(source), 1.0e-300),
        _config_sha256(config, backend),
        _implementation_sha256("adapters.py", "models.py", "numerics.py", "warp_solver.py"),
        _config_json(config, backend),
        host_synchronizations,
        max(25, linear.residual_history_stride),
    )
    return MaterialFieldResult(
        field,
        diagnostics,
        _rows(problem.material_ids, problem.domain.shape),
        _rows(problem.free_current_phi_a_per_m2, problem.domain.shape),
        _rows(problem.remanence_r_t, problem.domain.shape),
        _rows(problem.remanence_z_t, problem.domain.shape),
        problem,
    )
