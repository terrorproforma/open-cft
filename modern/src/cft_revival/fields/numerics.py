"""Matrix-free conservative finite differences for axisymmetric flux ``psi``."""

from __future__ import annotations

from math import fsum, isfinite, sqrt
from time import perf_counter
from typing import Sequence

from .models import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    FieldConvergenceError,
    FieldMap,
    FieldValidationError,
    SolverConfig,
    SolverDiagnostics,
)


def coordinates(domain: AxisymmetricDomain) -> tuple[tuple[float, ...], tuple[float, ...]]:
    r = tuple(i * domain.dr_m for i in range(domain.radial_intervals + 1))
    z = tuple(
        domain.z_min_m + j * domain.dz_m
        for j in range(domain.axial_intervals + 1)
    )
    return r, z


def current_density_grid(problem: AxisymmetricProblem) -> tuple[float, ...]:
    """Average ``J_phi`` over nodal dual cells while conserving ampere-turns."""

    domain = problem.domain
    r, z = coordinates(domain)
    values = [0.0] * (len(r) * len(z))
    nz = len(z)
    dual_area = domain.dr_m * domain.dz_m
    for source in problem.sources:
        weights = _source_overlap_weights(problem, source)
        represented_area = fsum(weight for _, weight in weights)
        requested_area = (source.r_outer_m - source.r_inner_m) * (
            source.z_max_m - source.z_min_m
        )
        if represented_area <= 0.0 or abs(represented_area - requested_area) > (
            1.0e-12 * requested_area
        ):
            raise FieldValidationError(
                f"source {source.name!r} dual-cell overlap does not preserve its "
                "geometric area; refine the grid or move the band inside support"
            )
        signed_ampere_turns = source.polarity * source.ampere_turns_a
        for index, overlap_area in weights:
            values[index] += (
                signed_ampere_turns * overlap_area / (requested_area * dual_area)
            )
    if any(not isfinite(value) for value in values):
        raise FieldValidationError("sampled current density is nonfinite")
    return tuple(values)


def _source_overlap_weights(problem, source) -> list[tuple[int, float]]:
    domain = problem.domain
    r, z = coordinates(domain)
    nz = len(z)
    weights: list[tuple[int, float]] = []
    for i in range(1, len(r) - 1):
        radial_overlap = max(
            0.0,
            min(r[i] + 0.5 * domain.dr_m, source.r_outer_m)
            - max(r[i] - 0.5 * domain.dr_m, source.r_inner_m),
        )
        if radial_overlap == 0.0:
            continue
        for j in range(1, len(z) - 1):
            axial_overlap = max(
                0.0,
                min(z[j] + 0.5 * domain.dz_m, source.z_max_m)
                - max(z[j] - 0.5 * domain.dz_m, source.z_min_m),
            )
            if axial_overlap > 0.0:
                weights.append((i * nz + j, radial_overlap * axial_overlap))
    return weights


def source_discretization_diagnostics(
    problem: AxisymmetricProblem,
) -> tuple[dict[str, float | int | str], ...]:
    """Quantify each band's conservative dual-cell geometry transfer."""

    domain = problem.domain
    r, z = coordinates(domain)
    nz = len(z)
    diagnostics: list[dict[str, float | int | str]] = []
    for source in problem.sources:
        weights = _source_overlap_weights(problem, source)
        represented_area = fsum(weight for _, weight in weights)
        requested_area = (source.r_outer_m - source.r_inner_m) * (
            source.z_max_m - source.z_min_m
        )
        requested_r = 0.5 * (source.r_inner_m + source.r_outer_m)
        requested_z = 0.5 * (source.z_min_m + source.z_max_m)
        represented_r = fsum(
            r[index // nz] * weight for index, weight in weights
        ) / represented_area
        represented_z = fsum(
            z[index % nz] * weight for index, weight in weights
        ) / represented_area
        requested_current = source.polarity * source.ampere_turns_a
        represented_current = fsum(
            requested_current
            * overlap_area
            / (requested_area * domain.dr_m * domain.dz_m)
            * domain.dr_m
            * domain.dz_m
            for _, overlap_area in weights
        )
        radial_nodes = {index // nz for index, _ in weights}
        axial_nodes = {index % nz for index, _ in weights}
        diagnostics.append(
            {
                "name": source.name,
                "requested_area_m2": requested_area,
                "represented_overlap_area_m2": represented_area,
                "area_error_m2": represented_area - requested_area,
                "requested_centroid_r_m": requested_r,
                "represented_centroid_r_m": represented_r,
                "centroid_r_error_m": represented_r - requested_r,
                "requested_centroid_z_m": requested_z,
                "represented_centroid_z_m": represented_z,
                "centroid_z_error_m": represented_z - requested_z,
                "requested_signed_ampere_turns_a": requested_current,
                "represented_signed_ampere_turns_a": represented_current,
                "ampere_turn_error_a": represented_current - requested_current,
                "radial_nodes_touched": len(radial_nodes),
                "axial_nodes_touched": len(axial_nodes),
                "dual_cells_touched": len(weights),
            }
        )
    return tuple(diagnostics)


def _validate_flat_source(
    domain: AxisymmetricDomain, values: Sequence[float]
) -> tuple[float, ...]:
    expected = domain.shape[0] * domain.shape[1]
    if isinstance(values, (str, bytes)):
        raise FieldValidationError("current-density grid must be a flat numeric sequence")
    try:
        actual_length = len(values)
    except TypeError as error:
        raise FieldValidationError(
            "current-density grid must be a flat numeric sequence"
        ) from error
    if actual_length != expected:
        raise FieldValidationError(
            f"current-density grid has {actual_length} entries; expected {expected}"
        )
    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise FieldValidationError(
            "current-density grid must be a flat numeric sequence"
        ) from error
    if any(not isfinite(value) for value in converted):
        raise FieldValidationError("current-density grid must contain only finite values")
    return converted


def operator_diagonal(domain: AxisymmetricDomain) -> tuple[float, ...]:
    nr_nodes, nz_nodes = domain.shape
    dr2 = domain.dr_m * domain.dr_m
    dz2 = domain.dz_m * domain.dz_m
    diagonal = [1.0] * (nr_nodes * nz_nodes)
    for i in range(1, nr_nodes - 1):
        radius = i * domain.dr_m
        c_minus = 1.0 / (radius - 0.5 * domain.dr_m)
        c_plus = 1.0 / (radius + 0.5 * domain.dr_m)
        value = (c_minus + c_plus) / dr2 + 2.0 / (radius * dz2)
        for j in range(1, nz_nodes - 1):
            diagonal[i * nz_nodes + j] = value
    return tuple(diagonal)


def apply_operator(domain: AxisymmetricDomain, vector: Sequence[float]) -> list[float]:
    """Apply ``-div((1/r) grad(psi))`` with zero-Dirichlet box boundaries."""

    nr_nodes, nz_nodes = domain.shape
    if len(vector) != nr_nodes * nz_nodes:
        raise FieldValidationError("operator input shape does not match the domain")
    dr2 = domain.dr_m * domain.dr_m
    dz2 = domain.dz_m * domain.dz_m
    output = [0.0] * len(vector)
    for i in range(1, nr_nodes - 1):
        radius = i * domain.dr_m
        c_minus = 1.0 / (radius - 0.5 * domain.dr_m)
        c_plus = 1.0 / (radius + 0.5 * domain.dr_m)
        z_coefficient = 1.0 / (radius * dz2)
        radial_scale = 1.0 / dr2
        for j in range(1, nz_nodes - 1):
            index = i * nz_nodes + j
            output[index] = (
                (c_minus + c_plus) * radial_scale * vector[index]
                - c_minus * radial_scale * vector[index - nz_nodes]
                - c_plus * radial_scale * vector[index + nz_nodes]
                + z_coefficient
                * (2.0 * vector[index] - vector[index - 1] - vector[index + 1])
            )
    return output


def _dot_interior(
    domain: AxisymmetricDomain, left: Sequence[float], right: Sequence[float]
) -> float:
    nr_nodes, nz_nodes = domain.shape
    return fsum(
        left[i * nz_nodes + j] * right[i * nz_nodes + j]
        for i in range(1, nr_nodes - 1)
        for j in range(1, nz_nodes - 1)
    )


def _recover_fields(
    domain: AxisymmetricDomain, psi: Sequence[float]
) -> tuple[list[float], list[float]]:
    nr_nodes, nz_nodes = domain.shape
    dr = domain.dr_m
    dz = domain.dz_m
    b_r = [0.0] * len(psi)
    b_z = [0.0] * len(psi)

    for j in range(nz_nodes):
        # psi = a(z) r^2 + O(r^4), so Bz(0,z) = 2a(z).
        b_z[j] = (16.0 * psi[nz_nodes + j] - psi[2 * nz_nodes + j]) / (
            6.0 * dr * dr
        )
    for i in range(1, nr_nodes):
        radius = i * dr
        for j in range(nz_nodes):
            index = i * nz_nodes + j
            if i == nr_nodes - 1:
                psi_r = (
                    3.0 * psi[index]
                    - 4.0 * psi[index - nz_nodes]
                    + psi[index - 2 * nz_nodes]
                ) / (2.0 * dr)
            else:
                psi_r = (psi[index + nz_nodes] - psi[index - nz_nodes]) / (2.0 * dr)
            b_z[index] = psi_r / radius

            if j == 0:
                psi_z = (-3.0 * psi[index] + 4.0 * psi[index + 1] - psi[index + 2]) / (
                    2.0 * dz
                )
            elif j == nz_nodes - 1:
                psi_z = (3.0 * psi[index] - 4.0 * psi[index - 1] + psi[index - 2]) / (
                    2.0 * dz
                )
            else:
                psi_z = (psi[index + 1] - psi[index - 1]) / (2.0 * dz)
            b_r[index] = -psi_z / radius
    return b_r, b_z


def max_flux_reconstruction_identity(
    domain: AxisymmetricDomain, b_r: Sequence[float], b_z: Sequence[float]
) -> float:
    """Check the mixed-derivative identity from reconstructing both fields from psi.

    This detects inconsistent field reconstruction.  It is not an independent
    Maxwell-equation validation because both components share the same flux.
    """

    nr_nodes, nz_nodes = domain.shape
    maximum = 0.0
    for i in range(1, nr_nodes - 1):
        radius = i * domain.dr_m
        for j in range(1, nz_nodes - 1):
            index = i * nz_nodes + j
            radial = (
                (radius + domain.dr_m) * b_r[index + nz_nodes]
                - (radius - domain.dr_m) * b_r[index - nz_nodes]
            ) / (2.0 * domain.dr_m * radius)
            axial = (b_z[index + 1] - b_z[index - 1]) / (2.0 * domain.dz_m)
            maximum = max(maximum, abs(radial + axial))
    return maximum


def _rows(values: Sequence[float], shape: tuple[int, int]) -> tuple[tuple[float, ...], ...]:
    nr_nodes, nz_nodes = shape
    return tuple(
        tuple(float(values[i * nz_nodes + j]) for j in range(nz_nodes))
        for i in range(nr_nodes)
    )


def finalize_field_map(
    domain: AxisymmetricDomain,
    psi: Sequence[float],
    *,
    converged: bool,
    iterations: int,
    initial_residual: float,
    final_residual: float,
    residual_history: Sequence[float],
    backend: str,
    true_residual_restarts: int = 0,
    stagnation_detected: bool = False,
) -> FieldMap:
    if any(not isfinite(float(value)) for value in psi):
        raise FieldConvergenceError("solver produced a nonfinite flux map")
    b_r, b_z = _recover_fields(domain, psi)
    if any(not isfinite(value) for value in (*b_r, *b_z)):
        raise FieldConvergenceError("field recovery produced nonfinite values")
    relative = 0.0 if initial_residual == 0.0 else final_residual / initial_residual
    r, z = coordinates(domain)
    diagnostics = SolverDiagnostics(
        converged=converged,
        iterations=iterations,
        initial_residual_l2=initial_residual,
        final_residual_l2=final_residual,
        relative_residual_l2=relative,
        residual_history_l2=tuple(float(value) for value in residual_history),
        max_flux_reconstruction_identity_t_per_m=max_flux_reconstruction_identity(
            domain, b_r, b_z
        ),
        true_residual_restarts=true_residual_restarts,
        stagnation_detected=stagnation_detected,
        backend=backend,
    )
    return FieldMap(
        r_m=r,
        z_m=z,
        psi_wb=_rows(psi, domain.shape),
        b_r_t=_rows(b_r, domain.shape),
        b_z_t=_rows(b_z, domain.shape),
        diagnostics=diagnostics,
    )


def solve_current_density_cpu(
    domain: AxisymmetricDomain,
    current_density_a_per_m2: Sequence[float],
    *,
    permeability_h_per_m: float,
    config: SolverConfig = SolverConfig(),
    raise_on_nonconvergence: bool = True,
) -> FieldMap:
    """Solve one source grid with dependency-free binary64 Jacobi-PCG."""

    source = _validate_flat_source(domain, current_density_a_per_m2)
    if not isfinite(permeability_h_per_m) or permeability_h_per_m <= 0.0:
        raise FieldValidationError("permeability_h_per_m must be finite and positive")
    nr_nodes, nz_nodes = domain.shape
    count = nr_nodes * nz_nodes
    boundary = lambda index: (
        index // nz_nodes in (0, nr_nodes - 1)
        or index % nz_nodes in (0, nz_nodes - 1)
    )
    b = [
        0.0 if boundary(index) else permeability_h_per_m * source[index]
        for index in range(count)
    ]
    if any(not isfinite(value) for value in b):
        raise FieldValidationError("permeability times source is nonfinite")
    x = [0.0] * count
    r = b.copy()
    initial = sqrt(_dot_interior(domain, r, r))
    history = [initial]
    threshold = max(config.absolute_tolerance, config.relative_tolerance * initial)
    if initial == 0.0:
        return finalize_field_map(
            domain,
            x,
            converged=True,
            iterations=0,
            initial_residual=0.0,
            final_residual=0.0,
            residual_history=history,
            backend="python",
        )

    diagonal = operator_diagonal(domain)
    z = [r[index] / diagonal[index] for index in range(count)]
    p = z.copy()
    rho = _dot_interior(domain, r, z)
    converged = False
    restart_count = 0
    stagnation_detected = False
    final = initial
    iterations = 0
    for iteration in range(1, config.max_iterations + 1):
        ap = apply_operator(domain, p)
        denominator = _dot_interior(domain, p, ap)
        if not isfinite(denominator) or denominator <= 0.0:
            raise FieldConvergenceError("PCG lost positive definiteness")
        alpha = rho / denominator
        for index in range(count):
            x[index] += alpha * p[index]
            r[index] -= alpha * ap[index]
        final = sqrt(_dot_interior(domain, r, r))
        iterations = iteration
        if iteration % config.residual_history_stride == 0:
            history.append(final)
        if not isfinite(final):
            raise FieldConvergenceError("PCG residual became nonfinite")
        if final <= threshold:
            true_residual = _recompute_true_residual(domain, b, x)
            true_norm = sqrt(_dot_interior(domain, true_residual, true_residual))
            if not isfinite(true_norm):
                raise FieldConvergenceError("true PCG residual became nonfinite")
            if history[-1] != true_norm:
                history.append(true_norm)
            if true_norm <= threshold:
                final = true_norm
                r = true_residual
                converged = True
                break
            if restart_count >= config.max_true_residual_restarts:
                final = true_norm
                r = true_residual
                stagnation_detected = True
                break
            restart_count += 1
            r = true_residual
            z = [r[index] / diagonal[index] for index in range(count)]
            p = z.copy()
            rho = _dot_interior(domain, r, z)
            final = true_norm
            continue
        z = [r[index] / diagonal[index] for index in range(count)]
        new_rho = _dot_interior(domain, r, z)
        beta = new_rho / rho
        for index in range(count):
            p[index] = z[index] + beta * p[index]
        rho = new_rho

    # Recompute the true algebraic residual; recurrence status alone is rejected.
    true_residual = _recompute_true_residual(domain, b, x)
    final = sqrt(_dot_interior(domain, true_residual, true_residual))
    converged = converged and final <= threshold
    if not history or history[-1] != final:
        history.append(final)
    result = finalize_field_map(
        domain,
        x,
        converged=converged,
        iterations=iterations,
        initial_residual=initial,
        final_residual=final,
        residual_history=history,
        backend="python",
        true_residual_restarts=restart_count,
        stagnation_detected=stagnation_detected,
    )
    if not converged and raise_on_nonconvergence:
        raise FieldConvergenceError(
            f"PCG did not converge in {iterations} iterations: "
            f"true residual {final:.6e}, required <= {threshold:.6e}"
        )
    return result


def _recompute_true_residual(
    domain: AxisymmetricDomain, rhs: Sequence[float], solution: Sequence[float]
) -> list[float]:
    applied = apply_operator(domain, solution)
    return [rhs[index] - applied[index] for index in range(len(rhs))]


def solve_problem_cpu(
    problem: AxisymmetricProblem,
    config: SolverConfig = SolverConfig(),
    *,
    raise_on_nonconvergence: bool = True,
) -> FieldMap:
    return solve_current_density_cpu(
        problem.domain,
        current_density_grid(problem),
        permeability_h_per_m=problem.permeability_h_per_m,
        config=config,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )


def diagnostic_runtime_seconds(function, /, *args, **kwargs) -> tuple[object, float]:
    """One-shot wall time for diagnostics only; this is not a benchmark."""

    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started
