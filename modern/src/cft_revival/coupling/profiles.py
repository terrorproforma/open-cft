"""Role-preserving profile extraction with stable interpolation."""

from __future__ import annotations

from bisect import bisect_right
from math import hypot, isfinite

from .models import (
    CouplingValidationError,
    FieldProfile,
    ProfileRole,
    UncertaintyModel,
    ValidatedAxisymmetricMap,
)


def stable_lerp(left: float, right: float, fraction: float) -> float:
    """Convex interpolation that avoids overflowing ``right-left``."""

    if not 0.0 <= fraction <= 1.0:
        raise CouplingValidationError("interpolation fraction must be in [0, 1]")
    difference = right - left
    result = (
        left + fraction * difference
        if isfinite(difference)
        else (1.0 - fraction) * left + fraction * right
    )
    if not isfinite(result):
        raise CouplingValidationError("finite interpolation inputs produced overflow")
    return result


def validate_uncertainty_model(model: UncertaintyModel) -> UncertaintyModel:
    values = (
        model.absolute_independent_sigma_t,
        model.relative_independent_sigma,
        model.common_mode_sigma_t,
        model.coverage_factor,
        model.residual_correlation,
    )
    if any(not isfinite(float(value)) for value in values):
        raise CouplingValidationError("uncertainty model values must be finite")
    if (
        model.absolute_independent_sigma_t < 0.0
        or model.relative_independent_sigma < 0.0
        or model.common_mode_sigma_t < 0.0
        or model.coverage_factor <= 0.0
    ):
        raise CouplingValidationError(
            "uncertainty sigmas must be non-negative and coverage positive"
        )
    if not -1.0 <= model.residual_correlation <= 1.0:
        raise CouplingValidationError("residual_correlation must be in [-1, 1]")
    return model


def _independent_uncertainty(
    br_t: tuple[float, ...],
    bz_t: tuple[float, ...],
    model: UncertaintyModel,
) -> tuple[float, ...]:
    result = tuple(
        model.absolute_independent_sigma_t
        + model.relative_independent_sigma * hypot(br, bz)
        for br, bz in zip(br_t, bz_t, strict=True)
    )
    if any(not isfinite(value) for value in result):
        raise CouplingValidationError("field uncertainty calculation overflowed")
    return result


def extract_profiles(
    field: ValidatedAxisymmetricMap,
    wall_radius_m: float,
    *,
    uncertainty_model: UncertaintyModel | None = None,
    absolute_uncertainty_t: float = 0.0,
    relative_uncertainty: float = 0.0,
) -> tuple[FieldProfile, FieldProfile]:
    """Extract the innermost and linearly interpolated wall profiles."""

    if uncertainty_model is not None and (
        absolute_uncertainty_t != 0.0 or relative_uncertainty != 0.0
    ):
        raise CouplingValidationError(
            "legacy uncertainty arguments cannot accompany uncertainty_model"
        )
    model = validate_uncertainty_model(
        uncertainty_model
        if uncertainty_model is not None
        else UncertaintyModel(
            absolute_independent_sigma_t=float(absolute_uncertainty_t),
            relative_independent_sigma=float(relative_uncertainty),
        )
    )
    wall = float(wall_radius_m)
    if not isfinite(wall) or wall <= field.r_m[0] or wall > field.r_m[-1]:
        raise CouplingValidationError(
            "wall_radius_m must lie above the inner profile and within the map"
        )
    inner_br = field.b_r_t[0]
    inner_bz = field.b_z_t[0]
    inner_role = (
        ProfileRole.CENTRELINE
        if field.r_m[0] == 0.0
        else ProfileRole.INNER_RADIAL_PROFILE
    )
    inner = FieldProfile(
        name=inner_role.value,
        role=inner_role,
        sampled_r_m=field.r_m[0],
        z_m=field.z_m,
        b_r_t=inner_br,
        b_z_t=inner_bz,
        independent_sigma_b_t=_independent_uncertainty(inner_br, inner_bz, model),
        common_mode_sigma_t=model.common_mode_sigma_t,
    )
    upper = bisect_right(field.r_m, wall)
    if upper == len(field.r_m) or field.r_m[upper - 1] == wall:
        lower = upper - 1
        wall_br = field.b_r_t[lower]
        wall_bz = field.b_z_t[lower]
    else:
        lower = upper - 1
        fraction = (wall - field.r_m[lower]) / (
            field.r_m[upper] - field.r_m[lower]
        )
        wall_br = tuple(
            stable_lerp(left, right, fraction)
            for left, right in zip(
                field.b_r_t[lower], field.b_r_t[upper], strict=True
            )
        )
        wall_bz = tuple(
            stable_lerp(left, right, fraction)
            for left, right in zip(
                field.b_z_t[lower], field.b_z_t[upper], strict=True
            )
        )
    wall_profile = FieldProfile(
        name=ProfileRole.WALL.value,
        role=ProfileRole.WALL,
        sampled_r_m=wall,
        z_m=field.z_m,
        b_r_t=wall_br,
        b_z_t=wall_bz,
        independent_sigma_b_t=_independent_uncertainty(wall_br, wall_bz, model),
        common_mode_sigma_t=model.common_mode_sigma_t,
    )
    return inner, wall_profile


def interpolate_profile(
    profile: FieldProfile, z_m: float
) -> tuple[float, float, float, float, float]:
    """Return ``Br, Bz, |B|, independent_sigma, common_sigma`` at ``z``."""

    z = float(z_m)
    if not isfinite(z) or z < profile.z_m[0] or z > profile.z_m[-1]:
        raise CouplingValidationError("requested z_m lies outside the profile")
    upper = bisect_right(profile.z_m, z)
    if upper == 0 or upper == len(profile.z_m) or profile.z_m[upper - 1] == z:
        index = 0 if upper == 0 else upper - 1
        br = profile.b_r_t[index]
        bz = profile.b_z_t[index]
        independent = profile.independent_sigma_b_t[index]
    else:
        lower = upper - 1
        fraction = (z - profile.z_m[lower]) / (
            profile.z_m[upper] - profile.z_m[lower]
        )
        br = stable_lerp(profile.b_r_t[lower], profile.b_r_t[upper], fraction)
        bz = stable_lerp(profile.b_z_t[lower], profile.b_z_t[upper], fraction)
        independent = stable_lerp(
            profile.independent_sigma_b_t[lower],
            profile.independent_sigma_b_t[upper],
            fraction,
        )
    magnitude = hypot(br, bz)
    if not isfinite(magnitude):
        raise CouplingValidationError("profile magnitude overflowed")
    return br, bz, magnitude, independent, profile.common_mode_sigma_t
