"""HEMP design-ratio descriptors of one characterized L1a field (sweep v3).

Definitions (frozen in protocol.json#descriptors_v3; every number is a field descriptor,
never a probability):

* **Koch design ratio rho.** Koch, Harmann, Kornfeld, IEPC-2007-110 (Table 1 and the
  sentence "High reflection of electrons in the cusp-mirror situation is achieved for a
  high ratio of the (predominantly radial) magnetic field strength at the inner discharge
  tube radius at the cusp middle plane to the (predominantly axial) magnetic field strength
  downstream the cusp"; values 1.53/1.70/3.07 for DM9-1 and 4.07/4.2/10.57 for DM10). The
  radius at which the downstream axial field is taken is not stated in the paper
  (twt-ppm-physics-for-hemp.md, section 8). This module therefore reports, per wall cusp
  ``c`` with axis null ``z_k`` and wall intersection ``z_c``:

  - ``rho_downstream`` = |B|(r_w, z_c) / max |B_z(0, z)| over the axis interval from ``z_k``
    to the next axis null downstream (Koch's wording, axis reading);
  - ``rho_upstream`` = the same with the upstream interval;
  - ``rho_conservative`` = |B|(r_w, z_c) / max(both axis peaks) - the binding classifier for
    "HEMP-like" (rho >= 1.5 at every wall cusp inside the straight dielectric);
  - ``rho_wall`` = |B|(r_w, z_c) / max |B|(r_w, z) over the two adjacent wall cells (the
    "cusp is the wall maximum" reading; PPM fundamental-only prediction I_1(x_w)/I_0(x_w)).

* **PPM prediction.** For an infinite single-harmonic stack of pitch L and wall radius
  r_w (twt-ppm-physics-for-hemp.md, section 2.2), |B|(r_w, z_c) / b_1 = I_1(x_w) with
  x_w = pi r_w / L, so rho_axis ~ I_1(x_w) and rho_wall ~ I_1(x_w) / I_0(x_w). The
  hypothesis tested by the campaign is that the realised rho follows I_1(x_w); the
  threshold I_1(x*) = 1.5 is at x* = 1.93732 (r_w / L = 0.61667), computed here by
  bisection on the power series.

* **Wall harmonic content.** B_r(r_w, z) is sampled along the straight wall between the
  first and last stage centres and fitted (linear least squares, free phase per harmonic)
  to sin/cos pairs at k = 1, 3, 5 times kappa = pi / L with the GEOMETRIC pitch. The
  amplitudes are those measured at the wall - never an extrapolation of axis-fitted
  harmonics (the review documents a 1.7-2.1x error for that shortcut).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.orbit_mc import PsiBicubicField

from experiments.cusp_topology_search_v3_1 import topology as topology_module
from experiments.cusp_topology_search_v3_1.topology import ChannelGeometry, TopologyPolicy, TracingGrid

HEMP_LIKE_RHO = 1.5
WALL_HARMONICS = (1, 3, 5)


# --------------------------------------------------------------------------
# Modified Bessel functions (power series; exact enough for x <= 30)
# --------------------------------------------------------------------------


def bessel_i(order: int, x: float) -> float:
    half = 0.5 * float(x)
    term = half**order / math.factorial(order)
    total = term
    for m in range(1, 400):
        term = term * half * half / (m * (m + order))
        total += term
        if abs(term) <= 1e-17 * abs(total):
            break
    return total


def bessel_i0(x: float) -> float:
    return bessel_i(0, x)


def bessel_i1(x: float) -> float:
    return bessel_i(1, x)


def i1_root(target: float, *, low: float = 1.0e-6, high: float = 12.0) -> float:
    """x such that I_1(x) == target (I_1 is increasing on x > 0)."""

    if not bessel_i1(low) < target < bessel_i1(high):
        raise ValueError("target outside the bracket")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if bessel_i1(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


X_STAR_HEMP_LIKE = i1_root(HEMP_LIKE_RHO)  # 1.93732...
RW_OVER_L_STAR_HEMP_LIKE = X_STAR_HEMP_LIKE / math.pi  # 0.61667...


def ppm_prediction(x_w: float) -> dict[str, float]:
    i0 = bessel_i0(x_w)
    i1 = bessel_i1(x_w)
    return {
        "x_w": x_w,
        "wall_radius_over_pitch": x_w / math.pi,
        "i1_x_w": i1,
        "i0_x_w": i0,
        "i1_over_i0_x_w": i1 / i0,
        "predicted_hemp_like": i1 >= HEMP_LIKE_RHO,
    }


# --------------------------------------------------------------------------
# Per-cusp rho
# --------------------------------------------------------------------------


def _wall_max(field: PsiBicubicField, wall_radius_m: float, z_low: float, z_high: float, count: int) -> tuple[float, float]:
    best_value, best_z = -math.inf, z_low
    for index in range(count + 1):
        z = z_low + (z_high - z_low) * index / count
        magnitude = math.hypot(*field.field_cylindrical(wall_radius_m, z))
        if magnitude > best_value:
            best_value, best_z = magnitude, z
    return best_value, best_z


def cusp_rho_table(
    field: PsiBicubicField,
    grid: TracingGrid,
    geometry: ChannelGeometry,
    characterization: Mapping[str, Any],
    policy: TopologyPolicy,
) -> list[dict[str, Any]]:
    """Koch-ratio readings for every wall cusp inside the straight dielectric."""

    nulls = sorted(float(null["z_m"]) for null in characterization["axis_nulls"]["nulls"])
    window_low, window_high = characterization["axis_nulls"]["window_m"]
    cusps = characterization["topology"]["wall_cusps"]
    z_lo, z_hi = geometry.straight_z_min_m, geometry.straight_z_max_m
    z_grid_lo, z_grid_hi = float(grid.z_m[0]), float(grid.z_m[-1])
    wall_boundaries = [z_lo] + [float(cusp["z_c_m"]) for cusp in cusps] + [z_hi]
    rows: list[dict[str, Any]] = []
    for index, cusp in enumerate(cusps):
        z_k = float(cusp["axis_null_z_m"])
        b_c = float(cusp["wall_b_t"])
        previous = [z for z in nulls if z < z_k - 1.0e-12]
        following = [z for z in nulls if z > z_k + 1.0e-12]
        axis_up_low = max(previous) if previous else max(window_low, z_grid_lo)
        axis_down_high = min(following) if following else min(window_high, z_grid_hi)
        axis_up_low = max(axis_up_low, z_grid_lo)
        axis_down_high = min(axis_down_high, z_grid_hi)
        peak_up, peak_up_z = topology_module._axis_peak(field, axis_up_low, z_k, policy.axis_samples_per_interval)
        peak_down, peak_down_z = topology_module._axis_peak(field, z_k, axis_down_high, policy.axis_samples_per_interval)
        wall_up_max, wall_up_z = _wall_max(field, geometry.wall_radius_m, wall_boundaries[index], wall_boundaries[index + 1], policy.wall_samples_per_cell)
        wall_down_max, wall_down_z = _wall_max(field, geometry.wall_radius_m, wall_boundaries[index + 1], wall_boundaries[index + 2], policy.wall_samples_per_cell)
        strongest_axis = max(peak_up, peak_down)
        weakest_axis = min(peak_up, peak_down)
        rows.append(
            {
                "cusp_id": cusp["cusp_id"],
                "null_id": cusp["null_id"],
                "z_c_m": float(cusp["z_c_m"]),
                "axis_null_z_m": z_k,
                "wall_b_t": b_c,
                "wall_b_r_t": float(cusp["wall_b_r_t"]),
                "wall_b_z_t": float(cusp["wall_b_z_t"]),
                "angle_to_wall_normal_deg": float(cusp["angle_to_wall_normal_deg"]),
                "boundary_ambiguous": bool(cusp["boundary_ambiguous"]),
                "upstream_axis_interval_m": [axis_up_low, z_k],
                "downstream_axis_interval_m": [z_k, axis_down_high],
                "upstream_axis_peak_t": peak_up,
                "upstream_axis_peak_z_m": peak_up_z,
                "downstream_axis_peak_t": peak_down,
                "downstream_axis_peak_z_m": peak_down_z,
                "upstream_wall_interval_m": [wall_boundaries[index], wall_boundaries[index + 1]],
                "downstream_wall_interval_m": [wall_boundaries[index + 1], wall_boundaries[index + 2]],
                "upstream_wall_max_b_t": wall_up_max,
                "upstream_wall_max_z_m": wall_up_z,
                "downstream_wall_max_b_t": wall_down_max,
                "downstream_wall_max_z_m": wall_down_z,
                "rho_downstream": (b_c / peak_down) if peak_down > 0.0 else None,
                "rho_upstream": (b_c / peak_up) if peak_up > 0.0 else None,
                "rho_conservative": (b_c / strongest_axis) if strongest_axis > 0.0 else None,
                "rho_permissive": (b_c / weakest_axis) if weakest_axis > 0.0 else None,
                "rho_wall": (b_c / max(wall_up_max, wall_down_max)) if max(wall_up_max, wall_down_max) > 0.0 else None,
                "hemp_like_conservative": bool(strongest_axis > 0.0 and b_c / strongest_axis >= HEMP_LIKE_RHO),
                "cusp_is_wall_maximum": bool(b_c >= max(wall_up_max, wall_down_max) * (1.0 - 1.0e-9)),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Wall harmonic content
# --------------------------------------------------------------------------


def wall_harmonic_fit(field: PsiBicubicField, geometry: ChannelGeometry, *, samples: int = 400, harmonics: Sequence[int] = WALL_HARMONICS) -> dict[str, Any]:
    """Least-squares sin/cos fit of the wall B_r between the first and last stage centres."""

    centres = geometry.stage_centres_m
    z_low, z_high = float(centres[0]), float(centres[-1])
    z_low = max(z_low, geometry.straight_z_min_m)
    z_high = min(z_high, geometry.straight_z_max_m)
    if not z_high > z_low:
        return {"applies": False, "reason": "no stage-centre span inside the straight dielectric"}
    kappa = math.pi / geometry.stage_pitch_m
    z = np.linspace(z_low, z_high, samples + 1)
    br = np.array([field.field_cylindrical(geometry.wall_radius_m, float(value))[0] for value in z])
    bz = np.array([field.field_cylindrical(geometry.wall_radius_m, float(value))[1] for value in z])
    z0 = float(centres[0])
    columns = []
    for k in harmonics:
        columns.append(np.sin(k * kappa * (z - z0)))
        columns.append(np.cos(k * kappa * (z - z0)))
    matrix = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(matrix, br, rcond=None)
    fitted = matrix @ coefficients
    amplitudes = {}
    phases = {}
    for index, k in enumerate(harmonics):
        s, c = float(coefficients[2 * index]), float(coefficients[2 * index + 1])
        amplitudes[str(k)] = math.hypot(s, c)
        phases[str(k)] = math.atan2(c, s)
    fundamental = amplitudes[str(harmonics[0])]
    max_abs = float(np.max(np.abs(br))) if br.size else 0.0
    return {
        "applies": True,
        "window_m": [z_low, z_high],
        "samples": int(samples + 1),
        "pitch_m_used": geometry.stage_pitch_m,
        "harmonics": [int(k) for k in harmonics],
        "wall_b_r_amplitude_t": amplitudes,
        "wall_b_r_phase_rad": phases,
        "b3_over_b1": (amplitudes["3"] / fundamental) if fundamental > 0.0 and "3" in amplitudes else None,
        "b5_over_b1": (amplitudes["5"] / fundamental) if fundamental > 0.0 and "5" in amplitudes else None,
        "fit_rms_over_max": (float(np.sqrt(np.mean((fitted - br) ** 2))) / max_abs) if max_abs > 0.0 else None,
        "wall_b_r_max_abs_t": max_abs,
        "wall_b_z_max_abs_t": float(np.max(np.abs(bz))) if bz.size else 0.0,
        "wall_b_r_over_b_z_max": (max_abs / float(np.max(np.abs(bz)))) if bz.size and float(np.max(np.abs(bz))) > 0.0 else None,
    }


# --------------------------------------------------------------------------
# Design-level summary
# --------------------------------------------------------------------------


def _range(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "min": None, "max": None}
    return {"count": len(clean), "min": min(clean), "max": max(clean)}


def field_profiles(field: PsiBicubicField, grid: TracingGrid, geometry: ChannelGeometry, *, samples: int = 240) -> dict[str, Any]:
    """Wall |B|, wall B_r and axis B_z on the plot window (channel extended by one pitch)."""

    z_low = max(float(grid.z_m[0]), geometry.straight_z_min_m - geometry.stage_pitch_m)
    z_high = min(float(grid.z_m[-1]), geometry.chamber_length_m + geometry.stage_pitch_m)
    z = [z_low + (z_high - z_low) * index / samples for index in range(samples + 1)]
    wall = [field.field_cylindrical(geometry.wall_radius_m, value) for value in z]
    axis = [field.field_cylindrical(0.0, value)[1] for value in z]
    return {
        "z_m": z,
        "wall_abs_b_t": [math.hypot(br, bz) for br, bz in wall],
        "wall_b_r_t": [br for br, _ in wall],
        "wall_b_z_t": [bz for _, bz in wall],
        "axis_b_z_t": axis,
    }


def design_descriptors(
    grid: TracingGrid,
    geometry: ChannelGeometry,
    characterization: Mapping[str, Any],
    policy: TopologyPolicy,
    *,
    source_identity_sha256: str,
    minimum_certificate_tightness_ratio: float,
    stage_count: int,
    with_profiles: bool = False,
) -> dict[str, Any]:
    """rho table, PPM prediction, wall harmonics and HEMP-like flags for one map."""

    field = topology_module.bicubic_field(grid, source_identity_sha256=source_identity_sha256, minimum_certificate_tightness_ratio=minimum_certificate_tightness_ratio)
    x_w = math.pi * geometry.wall_radius_m / geometry.stage_pitch_m
    prediction = ppm_prediction(x_w)
    cusps = cusp_rho_table(field, grid, geometry, characterization, policy)
    conservative = [row["rho_conservative"] for row in cusps]
    interior_ratios = [row["rho_conservative"] / prediction["i1_x_w"] for row in cusps if row["rho_conservative"] is not None]
    wall_ratios = [row["rho_wall"] / prediction["i1_over_i0_x_w"] for row in cusps if row["rho_wall"] is not None]
    harmonics = wall_harmonic_fit(field, geometry)
    hemp_like = bool(cusps) and all(row["hemp_like_conservative"] for row in cusps)
    return {
        "x_w": x_w,
        "wall_radius_over_pitch": geometry.wall_radius_m / geometry.stage_pitch_m,
        "x_m_inner": None,
        "ppm_prediction": prediction,
        "hemp_like_threshold": {"rho": HEMP_LIKE_RHO, "x_star": X_STAR_HEMP_LIKE, "wall_radius_over_pitch_star": RW_OVER_L_STAR_HEMP_LIKE},
        "stage_count": int(stage_count),
        "expected_interior_cusps_n_minus_1": int(stage_count) - 1,
        "wall_cusp_count": len(cusps),
        "cusps": cusps,
        "rho_conservative_range": _range(conservative),
        "rho_downstream_range": _range([row["rho_downstream"] for row in cusps]),
        "rho_wall_range": _range([row["rho_wall"] for row in cusps]),
        "rho_over_i1_range": _range(interior_ratios),
        "rho_wall_over_i1_i0_range": _range(wall_ratios),
        "min_rho_conservative": min(conservative) if conservative and all(v is not None for v in conservative) else None,
        "hemp_like_all_cusps": hemp_like,
        "hemp_like_cusp_count": sum(row["hemp_like_conservative"] for row in cusps),
        "cusp_is_wall_maximum_count": sum(row["cusp_is_wall_maximum"] for row in cusps),
        "predicted_hemp_like_i1": bool(prediction["predicted_hemp_like"]),
        "prediction_agrees_with_realised": bool(prediction["predicted_hemp_like"]) == hemp_like if cusps else None,
        "four_wall_cusps": len(cusps) == 4,
        "five_stage_four_cusp_hemp_like": bool(stage_count == 5 and len(cusps) == 4 and hemp_like),
        "wall_harmonics": harmonics,
        "separatrix_angle_deg_range": _range([row["angle_to_wall_normal_deg"] for row in cusps]),
        "profiles": field_profiles(field, grid, geometry) if with_profiles else None,
    }


def resolution_sensitivity(accepted: Mapping[str, Any], refined: Mapping[str, Any]) -> dict[str, Any]:
    """Per-cusp |rho_acc - rho_ref| / rho_ref when the cusp counts agree (reported)."""

    a = accepted["cusps"]
    r = refined["cusps"]
    if len(a) != len(r):
        return {"comparable": False, "accepted_cusp_count": len(a), "refined_cusp_count": len(r), "max_relative_rho_difference": None, "rows": []}
    rows = []
    for left, right in zip(a, r, strict=True):
        if left["rho_conservative"] is None or right["rho_conservative"] is None or right["rho_conservative"] == 0.0:
            rows.append({"cusp_id": left["cusp_id"], "relative_rho_difference": None})
            continue
        rows.append(
            {
                "cusp_id": left["cusp_id"],
                "accepted_rho_conservative": left["rho_conservative"],
                "refined_rho_conservative": right["rho_conservative"],
                "relative_rho_difference": abs(left["rho_conservative"] - right["rho_conservative"]) / abs(right["rho_conservative"]),
                "z_c_shift_m": abs(left["z_c_m"] - right["z_c_m"]),
            }
        )
    differences = [row["relative_rho_difference"] for row in rows if row.get("relative_rho_difference") is not None]
    return {
        "comparable": True,
        "accepted_cusp_count": len(a),
        "refined_cusp_count": len(r),
        "max_relative_rho_difference": max(differences) if differences else None,
        "hemp_like_flag_agrees": accepted["hemp_like_all_cusps"] == refined["hemp_like_all_cusps"],
        "rows": rows,
    }
