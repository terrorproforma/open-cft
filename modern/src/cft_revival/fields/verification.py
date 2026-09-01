"""Manufactured-solution and backend-parity verification for L1a."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt

from .models import AxisymmetricDomain, FieldMap, MU0_H_PER_M, SolverConfig
from .numerics import (
    apply_operator,
    coordinates,
    solve_current_density_cpu,
)
from .warp_solver import solve_current_density_warp


@dataclass(frozen=True, slots=True)
class ManufacturedErrors:
    intervals: tuple[int, int]
    psi_relative_l2: float
    field_relative_l2: float
    axis_bz_max_abs_t: float
    source_operator_relative_l2: float
    max_flux_reconstruction_identity_t_per_m: float
    relative_solver_residual: float


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    cases: tuple[ManufacturedErrors, ...]
    psi_orders: tuple[float, ...]
    field_orders: tuple[float, ...]


def manufactured_values(
    domain: AxisymmetricDomain, psi_scale_wb: float = 1.0e-5
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return analytic ``psi, J_phi, Br, Bz`` on the domain nodes.

    ``psi = C (x²-x⁴) sin(k(z-z0))`` where ``x=r/R`` and
    ``k=pi/(z1-z0)``.  It is smooth, regular on-axis, and zero on every box
    boundary.
    """

    r, z = coordinates(domain)
    radius = domain.radius_m
    length = domain.z_max_m - domain.z_min_m
    k = pi / length
    psi: list[float] = []
    source: list[float] = []
    b_r: list[float] = []
    b_z: list[float] = []
    for radial in r:
        x = radial / radius
        radial_shape = x * x - x**4
        for axial in z:
            phase = k * (axial - domain.z_min_m)
            from math import cos, sin

            sine = sin(phase)
            cosine = cos(phase)
            psi.append(psi_scale_wb * radial_shape * sine)
            # -div(r^-1 grad psi) = mu0 J_phi.
            source.append(
                (
                    psi_scale_wb
                    * (
                        8.0 * radial / radius**4
                        + k * k * (radial / radius**2 - radial**3 / radius**4)
                    )
                    * sine
                )
                / MU0_H_PER_M
            )
            b_r.append(
                -psi_scale_wb
                * k
                * (radial / radius**2 - radial**3 / radius**4)
                * cosine
            )
            b_z.append(
                psi_scale_wb
                * (2.0 / radius**2 - 4.0 * radial**2 / radius**4)
                * sine
            )
    return tuple(psi), tuple(source), tuple(b_r), tuple(b_z)


def _relative_l2(actual, expected) -> float:
    numerator = sqrt(sum((float(a) - float(e)) ** 2 for a, e in zip(actual, expected)))
    denominator = sqrt(sum(float(e) ** 2 for e in expected))
    return numerator / denominator


def _flatten(rows) -> tuple[float, ...]:
    return tuple(value for row in rows for value in row)


def manufactured_error(field: FieldMap, domain: AxisymmetricDomain) -> ManufacturedErrors:
    exact_psi, source, exact_br, exact_bz = manufactured_values(domain)
    actual_psi = _flatten(field.psi_wb)
    actual_br = _flatten(field.b_r_t)
    actual_bz = _flatten(field.b_z_t)
    field_error = sqrt(
        sum((a - e) ** 2 for a, e in zip(actual_br, exact_br))
        + sum((a - e) ** 2 for a, e in zip(actual_bz, exact_bz))
    ) / sqrt(sum(value * value for value in (*exact_br, *exact_bz)))
    discrete_exact = apply_operator(domain, exact_psi)
    rhs = tuple(MU0_H_PER_M * value for value in source)
    nr_nodes, nz_nodes = domain.shape
    residual_indices = (
        i * nz_nodes + j
        for i in range(1, nr_nodes - 1)
        for j in range(1, nz_nodes - 1)
    )
    pairs = tuple((discrete_exact[index], rhs[index]) for index in residual_indices)
    operator_error = sqrt(sum((a - b) ** 2 for a, b in pairs)) / sqrt(
        sum(b * b for _, b in pairs)
    )
    axis_indices = range(nz_nodes)
    axis_error = max(abs(actual_bz[j] - exact_bz[j]) for j in axis_indices)
    return ManufacturedErrors(
        intervals=(domain.radial_intervals, domain.axial_intervals),
        psi_relative_l2=_relative_l2(actual_psi, exact_psi),
        field_relative_l2=field_error,
        axis_bz_max_abs_t=axis_error,
        source_operator_relative_l2=operator_error,
        max_flux_reconstruction_identity_t_per_m=(
            field.diagnostics.max_flux_reconstruction_identity_t_per_m
        ),
        relative_solver_residual=field.diagnostics.relative_residual_l2,
    )


def run_manufactured_convergence(
    resolutions: tuple[int, ...] = (16, 32, 64),
    *,
    backend: str = "python",
    config: SolverConfig = SolverConfig(relative_tolerance=1.0e-11),
) -> ConvergenceReport:
    cases = []
    for resolution in resolutions:
        domain = AxisymmetricDomain(
            radius_m=0.12,
            z_min_m=-0.15,
            z_max_m=0.15,
            radial_intervals=resolution,
            axial_intervals=2 * resolution,
        )
        _, source, _, _ = manufactured_values(domain)
        if backend == "python":
            field = solve_current_density_cpu(
                domain,
                source,
                permeability_h_per_m=MU0_H_PER_M,
                config=config,
            )
        elif backend.startswith("warp:"):
            field = solve_current_density_warp(
                domain,
                source,
                permeability_h_per_m=MU0_H_PER_M,
                device=backend.removeprefix("warp:"),
                config=config,
            )
        else:
            raise ValueError("backend must be 'python' or 'warp:<device>'")
        cases.append(manufactured_error(field, domain))
    psi_orders = tuple(
        log(coarse.psi_relative_l2 / fine.psi_relative_l2, 2.0)
        for coarse, fine in zip(cases, cases[1:])
    )
    field_orders = tuple(
        log(coarse.field_relative_l2 / fine.field_relative_l2, 2.0)
        for coarse, fine in zip(cases, cases[1:])
    )
    return ConvergenceReport(tuple(cases), psi_orders, field_orders)


def max_field_difference(left: FieldMap, right: FieldMap) -> dict[str, float]:
    if left.r_m != right.r_m or left.z_m != right.z_m:
        raise ValueError("field-map grids differ")

    def metric(left_rows, right_rows) -> tuple[float, float]:
        pairs = tuple(
            (a, b)
            for left_row, right_row in zip(left_rows, right_rows)
            for a, b in zip(left_row, right_row)
        )
        absolute = max(abs(a - b) for a, b in pairs)
        scale = max(max(abs(a), abs(b)) for a, b in pairs)
        return absolute, absolute / max(scale, 1.0e-300)

    psi_abs, psi_rel = metric(left.psi_wb, right.psi_wb)
    br_abs, br_rel = metric(left.b_r_t, right.b_r_t)
    bz_abs, bz_rel = metric(left.b_z_t, right.b_z_t)
    return {
        "psi_max_abs_wb": psi_abs,
        "psi_scale_relative": psi_rel,
        "br_max_abs_t": br_abs,
        "br_scale_relative": br_rel,
        "bz_max_abs_t": bz_abs,
        "bz_scale_relative": bz_rel,
    }
