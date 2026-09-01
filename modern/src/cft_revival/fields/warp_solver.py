"""Real Warp kernels for the L1a matrix-free axisymmetric PCG solve."""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Sequence

from .models import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    FieldConvergenceError,
    FieldDeviceError,
    FieldValidationError,
    SolverConfig,
)
from .numerics import (
    _dot_interior,
    _recompute_true_residual,
    _validate_flat_source,
    current_density_grid,
    finalize_field_map,
    operator_diagonal,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without the optional extra.
    wp = None  # type: ignore[assignment]


if wp is not None:

    @wp.kernel
    def _apply_operator_kernel(
        x: wp.array(dtype=wp.float64),
        output: wp.array(dtype=wp.float64),
        nr_nodes: int,
        nz_nodes: int,
        dr: wp.float64,
        dz: wp.float64,
    ):
        index = wp.tid()
        i = index // nz_nodes
        j = index - i * nz_nodes
        if i > 0 and i < nr_nodes - 1 and j > 0 and j < nz_nodes - 1:
            radius = wp.float64(i) * dr
            c_minus = wp.float64(1.0) / (radius - wp.float64(0.5) * dr)
            c_plus = wp.float64(1.0) / (radius + wp.float64(0.5) * dr)
            radial_scale = wp.float64(1.0) / (dr * dr)
            z_coefficient = wp.float64(1.0) / (radius * dz * dz)
            output[index] = (
                (c_minus + c_plus) * radial_scale * x[index]
                - c_minus * radial_scale * x[index - nz_nodes]
                - c_plus * radial_scale * x[index + nz_nodes]
                + z_coefficient
                * (
                    wp.float64(2.0) * x[index]
                    - x[index - 1]
                    - x[index + 1]
                )
            )
        else:
            output[index] = wp.float64(0.0)

    @wp.kernel
    def _dot_kernel(
        left: wp.array(dtype=wp.float64),
        right: wp.array(dtype=wp.float64),
        result: wp.array(dtype=wp.float64),
        nr_nodes: int,
        nz_nodes: int,
    ):
        index = wp.tid()
        i = index // nz_nodes
        j = index - i * nz_nodes
        if i > 0 and i < nr_nodes - 1 and j > 0 and j < nz_nodes - 1:
            wp.atomic_add(result, 0, left[index] * right[index])

    @wp.kernel
    def _pcg_x_r_update(
        x: wp.array(dtype=wp.float64),
        residual: wp.array(dtype=wp.float64),
        direction: wp.array(dtype=wp.float64),
        operator_direction: wp.array(dtype=wp.float64),
        alpha: wp.float64,
    ):
        index = wp.tid()
        x[index] = x[index] + alpha * direction[index]
        residual[index] = residual[index] - alpha * operator_direction[index]

    @wp.kernel
    def _jacobi_kernel(
        residual: wp.array(dtype=wp.float64),
        diagonal: wp.array(dtype=wp.float64),
        preconditioned: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        preconditioned[index] = residual[index] / diagonal[index]

    @wp.kernel
    def _direction_update_kernel(
        direction: wp.array(dtype=wp.float64),
        preconditioned: wp.array(dtype=wp.float64),
        beta: wp.float64,
    ):
        index = wp.tid()
        direction[index] = preconditioned[index] + beta * direction[index]


def _resolve_device(device: str):
    if wp is None:
        raise FieldDeviceError("NVIDIA Warp is unavailable")
    wp.init()
    requested = device.strip().lower() if isinstance(device, str) else ""
    if requested == "cuda":
        requested = "cuda:0"
    if requested != "cpu" and not requested.startswith("cuda:"):
        raise FieldDeviceError("device must be 'cpu', 'cuda', or 'cuda:N'")
    try:
        return wp.get_device(requested)
    except (RuntimeError, ValueError) as error:
        raise FieldDeviceError(f"Warp device {requested!r} is unavailable") from error


def device_available(device: str) -> bool:
    try:
        _resolve_device(device)
    except FieldDeviceError:
        return False
    return True


def solve_current_density_warp(
    domain: AxisymmetricDomain,
    current_density_a_per_m2: Sequence[float],
    *,
    permeability_h_per_m: float,
    device: str,
    config: SolverConfig = SolverConfig(),
    raise_on_nonconvergence: bool = True,
):
    """Solve with float64 Warp operator/vector/reduction kernels."""

    resolved = _resolve_device(device)
    if wp is None:  # Static narrowing after _resolve_device.
        raise FieldDeviceError("NVIDIA Warp is unavailable")
    source = _validate_flat_source(domain, current_density_a_per_m2)
    if not isfinite(permeability_h_per_m) or permeability_h_per_m <= 0.0:
        raise FieldValidationError("permeability_h_per_m must be finite and positive")
    nr_nodes, nz_nodes = domain.shape
    count = nr_nodes * nz_nodes
    b_host = [
        0.0
        if i in (0, nr_nodes - 1) or j in (0, nz_nodes - 1)
        else permeability_h_per_m * source[i * nz_nodes + j]
        for i in range(nr_nodes)
        for j in range(nz_nodes)
    ]
    if any(not isfinite(value) for value in b_host):
        raise FieldValidationError("permeability times source is nonfinite")

    b = wp.array(b_host, dtype=wp.float64, device=resolved)
    x = wp.zeros(count, dtype=wp.float64, device=resolved)
    residual = wp.array(b_host, dtype=wp.float64, device=resolved)
    diagonal = wp.array(operator_diagonal(domain), dtype=wp.float64, device=resolved)
    preconditioned = wp.empty_like(residual)
    direction = wp.empty_like(residual)
    operator_direction = wp.empty_like(residual)
    reduction = wp.zeros(1, dtype=wp.float64, device=resolved)

    def launch(kernel, inputs):
        wp.launch(kernel, dim=count, inputs=inputs, device=resolved)

    def dot(left, right) -> float:
        reduction.zero_()
        launch(_dot_kernel, [left, right, reduction, nr_nodes, nz_nodes])
        return float(reduction.numpy()[0])

    initial = sqrt(max(0.0, dot(residual, residual)))
    history = [initial]
    threshold = max(config.absolute_tolerance, config.relative_tolerance * initial)
    if initial == 0.0:
        return finalize_field_map(
            domain,
            x.numpy(),
            converged=True,
            iterations=0,
            initial_residual=0.0,
            final_residual=0.0,
            residual_history=history,
            backend=f"warp:{resolved}",
        )

    launch(_jacobi_kernel, [residual, diagonal, preconditioned])
    wp.copy(direction, preconditioned)
    rho = dot(residual, preconditioned)
    converged = False
    restart_count = 0
    stagnation_detected = False
    final = initial
    iterations = 0
    for iteration in range(1, config.max_iterations + 1):
        launch(
            _apply_operator_kernel,
            [
                direction,
                operator_direction,
                nr_nodes,
                nz_nodes,
                domain.dr_m,
                domain.dz_m,
            ],
        )
        denominator = dot(direction, operator_direction)
        if not isfinite(denominator) or denominator <= 0.0:
            raise FieldConvergenceError("Warp PCG lost positive definiteness")
        alpha = rho / denominator
        launch(
            _pcg_x_r_update,
            [x, residual, direction, operator_direction, alpha],
        )
        final = sqrt(max(0.0, dot(residual, residual)))
        iterations = iteration
        if iteration % config.residual_history_stride == 0:
            history.append(final)
        if not isfinite(final):
            raise FieldConvergenceError("Warp PCG residual became nonfinite")
        if final <= threshold:
            x_probe = [float(value) for value in x.numpy()]
            true_residual = _recompute_true_residual(domain, b_host, x_probe)
            true_norm = sqrt(_dot_interior(domain, true_residual, true_residual))
            if not isfinite(true_norm):
                raise FieldConvergenceError("Warp true PCG residual became nonfinite")
            if history[-1] != true_norm:
                history.append(true_norm)
            if true_norm <= threshold:
                final = true_norm
                converged = True
                break
            if restart_count >= config.max_true_residual_restarts:
                final = true_norm
                stagnation_detected = True
                break
            restart_count += 1
            residual = wp.array(true_residual, dtype=wp.float64, device=resolved)
            launch(_jacobi_kernel, [residual, diagonal, preconditioned])
            wp.copy(direction, preconditioned)
            rho = dot(residual, preconditioned)
            final = true_norm
            continue
        launch(_jacobi_kernel, [residual, diagonal, preconditioned])
        new_rho = dot(residual, preconditioned)
        launch(_direction_update_kernel, [direction, preconditioned, new_rho / rho])
        rho = new_rho

    wp.synchronize_device(resolved)
    x_host = [float(value) for value in x.numpy()]
    true_residual = _recompute_true_residual(domain, b_host, x_host)
    final = sqrt(_dot_interior(domain, true_residual, true_residual))
    converged = converged and final <= threshold
    if history[-1] != final:
        history.append(final)
    result = finalize_field_map(
        domain,
        x_host,
        converged=converged,
        iterations=iterations,
        initial_residual=initial,
        final_residual=final,
        residual_history=history,
        backend=f"warp:{resolved}",
        true_residual_restarts=restart_count,
        stagnation_detected=stagnation_detected,
    )
    if not converged and raise_on_nonconvergence:
        raise FieldConvergenceError(
            f"Warp PCG did not converge in {iterations} iterations: "
            f"true residual {final:.6e}, required <= {threshold:.6e}"
        )
    return result


def solve_problem_warp(
    problem: AxisymmetricProblem,
    *,
    device: str,
    config: SolverConfig = SolverConfig(),
    raise_on_nonconvergence: bool = True,
):
    return solve_current_density_warp(
        problem.domain,
        current_density_grid(problem),
        permeability_h_per_m=problem.permeability_h_per_m,
        device=device,
        config=config,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
