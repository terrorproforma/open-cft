"""Focused analytical and cross-backend verification utilities for L1b."""

from __future__ import annotations

from math import sqrt

from .models import MaterialFieldResult


def max_result_difference(
    left: MaterialFieldResult, right: MaterialFieldResult
) -> dict[str, float]:
    if left.field.r_m != right.field.r_m or left.field.z_m != right.field.z_m:
        raise ValueError("result grids differ")

    def difference(left_rows, right_rows) -> tuple[float, float]:
        pairs = tuple(
            (a, b)
            for lrow, rrow in zip(left_rows, right_rows)
            for a, b in zip(lrow, rrow)
        )
        absolute = max(abs(a - b) for a, b in pairs)
        scale = max(max(abs(a), abs(b)) for a, b in pairs)
        return absolute, absolute / max(scale, 1.0e-300)

    psi = difference(left.field.psi_wb, right.field.psi_wb)
    br = difference(left.field.b_r_t, right.field.b_r_t)
    bz = difference(left.field.b_z_t, right.field.b_z_t)
    return {
        "psi_max_abs_wb": psi[0],
        "psi_scale_relative": psi[1],
        "br_max_abs_t": br[0],
        "br_scale_relative": br[1],
        "bz_max_abs_t": bz[0],
        "bz_scale_relative": bz[1],
    }


def relative_field_l2(left: MaterialFieldResult, right: MaterialFieldResult) -> float:
    """Compare equal grids with a joint Br/Bz relative L2 norm."""

    if left.field.r_m != right.field.r_m or left.field.z_m != right.field.z_m:
        raise ValueError("result grids differ")
    numerator = 0.0
    denominator = 0.0
    for left_rows, right_rows in (
        (left.field.b_r_t, right.field.b_r_t),
        (left.field.b_z_t, right.field.b_z_t),
    ):
        for lrow, rrow in zip(left_rows, right_rows):
            for actual, reference in zip(lrow, rrow):
                numerator += (actual - reference) ** 2
                denominator += reference**2
    return sqrt(numerator / max(denominator, 1.0e-300))


def interface_jump_residuals(
    result: MaterialFieldResult, radial_index: int, axial_index: int
) -> dict[str, float]:
    """Estimate Bn and Ht jumps across one radial material face."""

    field = result.field
    problem = result.problem
    nz = problem.domain.shape[1]
    left = radial_index * nz + axial_index
    right = (radial_index + 1) * nz + axial_index
    b_normal_jump = field.b_r_t[radial_index + 1][axial_index] - field.b_r_t[
        radial_index
    ][axial_index]
    h_left = problem.reluctivity_per_m_h[left] * (
        field.b_z_t[radial_index][axial_index] - problem.remanence_z_t[left]
    )
    h_right = problem.reluctivity_per_m_h[right] * (
        field.b_z_t[radial_index + 1][axial_index] - problem.remanence_z_t[right]
    )
    return {"normal_b_jump_t": b_normal_jump, "tangential_h_jump_a_per_m": h_right - h_left}
