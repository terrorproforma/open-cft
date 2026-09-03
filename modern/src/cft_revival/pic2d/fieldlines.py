"""Event-aware magnetic field-line tracing on the PIC node field (v2.0 cathode placement).

A field line is integrated along the unit vector of the bilinearly interpolated node field
(the same interpolation the particle gather uses), in both directions from a start point,
with a fixed step of a fraction of a cell.  Tracing stops - and the stop is classified - when
the line leaves the plasma region:

  ``channel``   the line crossed the exit plane into the channel bore (r < r_exit at z = L)
  ``anode``     it reached the anode plane z = z_min inside the channel
  ``wall``      it hit the channel dielectric wall / the divergent cone (internal boundary)
  ``body``      it hit the thruster front face z = L, r > r_exit (dielectric or conductor)
  ``far_field`` it left through the plume's outer boundary r = R_plume or z = z_max
  ``axis``      it reached the axis (a pure-B_z line on axis; treated as a stop)
  ``null``      |B| below the floor (a magnetic null: the line is undefined there)
  ``length``    the length budget ran out (a closed/trapped line)

Physics: in the electrostatic model electrons stream freely along B and cross it only by
collisions and the E x B / grad-B drifts, so a cathode whose flux tube terminates on the
body face or the far field cannot feed the channel (review blocker 4d).  The channel-
connected flux tube of the plume is found by tracing OUTWARD from the exit aperture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .fields import MagneticFieldMap
from .mesh import MeshMasks
from .models import PIC2DValidationError

TERMINATIONS = ("channel", "anode", "wall", "body", "far_field", "axis", "null", "length")


@dataclass(frozen=True, slots=True)
class FieldLine:
    """One traced half-line (from the start point in one direction)."""

    direction: int                # +1 along B, -1 against B
    points: np.ndarray            # (N, 2) of (r, z) in metres, the start point first
    termination: str
    length_m: float
    b_min_t: float
    b_max_t: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction, "termination": self.termination, "length_m": self.length_m,
            "b_min_t": self.b_min_t, "b_max_t": self.b_max_t, "end_r_m": float(self.points[-1, 0]), "end_z_m": float(self.points[-1, 1]),
            "points": int(self.points.shape[0]),
        }


def interpolate_b(field: MagneticFieldMap, r_m: float, z_m: float) -> tuple[float, float]:
    """Bilinear (B_r, B_z) at (r, z) on the node grid (clamped to the grid box at its edges)."""

    grid = field.grid
    nr, nz = grid.node_shape
    fr = min(max((r_m - grid.r_m[0]) / grid.dr_m, 0.0), nr - 1.0)
    fz = min(max((z_m - grid.z_m[0]) / grid.dz_m, 0.0), nz - 1.0)
    i = min(int(math.floor(fr)), nr - 2)
    j = min(int(math.floor(fz)), nz - 2)
    a, b = fr - i, fz - j
    w = ((1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b)
    idx = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
    br = sum(wk * field.b_r_t[p] for wk, p in zip(w, idx))
    bz = sum(wk * field.b_z_t[p] for wk, p in zip(w, idx))
    return float(br), float(bz)


def classify_point(masks: MeshMasks, r_m: float, z_m: float) -> str | None:
    """None while inside the plasma region, else the boundary class the point crossed into."""

    grid = masks.grid
    geometry = grid.geometry
    eps = 1e-12
    if r_m < 0.0:
        return "axis"
    if z_m <= geometry.z_min_m + eps:
        return "anode"
    if z_m >= geometry.domain_z_max_m - eps or r_m >= geometry.max_radius_m - eps:
        return "far_field"
    j = min(int((z_m - geometry.z_min_m) / grid.dz_m), grid.axial_cells - 1)
    i = min(int(r_m / grid.dr_m), grid.radial_cells - 1)
    if masks.plasma_cell[i, j]:
        return None
    if z_m < geometry.z_max_m:
        return "wall"
    return "body"


def trace_field_line(field: MagneticFieldMap, masks: MeshMasks, r0_m: float, z0_m: float, *, direction: int = 1,
                     step_fraction: float = 0.25, max_length_m: float | None = None, b_floor_t: float = 1e-6) -> FieldLine:
    """Integrate along +/-B from (r0, z0) with midpoint steps of ``step_fraction`` cells until an event."""

    grid = field.grid
    geometry = grid.geometry
    if classify_point(masks, r0_m, z0_m) is not None:
        raise PIC2DValidationError("field-line start point must lie in the plasma region")
    h = step_fraction * min(grid.dr_m, grid.dz_m)
    budget = max_length_m if max_length_m is not None else 4.0 * (geometry.domain_z_max_m - geometry.z_min_m + geometry.max_radius_m)
    z_exit = geometry.z_max_m
    r, z = float(r0_m), float(z0_m)
    pts = [(r, z)]
    length = 0.0
    b_min, b_max = math.inf, 0.0
    termination = "length"
    while length < budget:
        br, bz = interpolate_b(field, r, z)
        mag = math.hypot(br, bz)
        b_min, b_max = min(b_min, mag), max(b_max, mag)
        if mag < b_floor_t:
            termination = "null"
            break
        # midpoint (RK2) step along the unit direction
        ur, uz = direction * br / mag, direction * bz / mag
        rm, zm = r + 0.5 * h * ur, z + 0.5 * h * uz
        brm, bzm = interpolate_b(field, rm, zm)
        magm = math.hypot(brm, bzm)
        if magm < b_floor_t:
            termination = "null"
            break
        r_new, z_new = r + h * direction * brm / magm, z + h * direction * bzm / magm
        event = classify_point(masks, r_new, z_new)
        length += h
        if event is not None:
            # crossing the exit plane inward inside the aperture is "entered the channel"
            if z >= z_exit > z_new and r_new < geometry.exit_radius_m:
                termination = "channel"
            elif event == "wall" and z >= z_exit:
                termination = "body"   # came from the plume side onto the front face (r > r_exit)
            else:
                termination = event
            pts.append((r_new, z_new))
            break
        if z >= z_exit > z_new and r_new < geometry.exit_radius_m:
            pts.append((r_new, z_new))
            termination = "channel"
            break
        r, z = r_new, z_new
        pts.append((r, z))
    return FieldLine(direction, np.array(pts, dtype=np.float64), termination, length, b_min if math.isfinite(b_min) else 0.0, b_max)


def trace_both(field: MagneticFieldMap, masks: MeshMasks, r0_m: float, z0_m: float, **kwargs: Any) -> tuple[FieldLine, FieldLine]:
    return trace_field_line(field, masks, r0_m, z0_m, direction=1, **kwargs), trace_field_line(field, masks, r0_m, z0_m, direction=-1, **kwargs)


def connects_to_channel(field: MagneticFieldMap, masks: MeshMasks, r0_m: float, z0_m: float, **kwargs: Any) -> tuple[bool, tuple[str, str]]:
    """True if either half-line from (r0, z0) enters the channel through the exit aperture."""

    fwd, bwd = trace_both(field, masks, r0_m, z0_m, **kwargs)
    return (fwd.termination == "channel" or bwd.termination == "channel"), (fwd.termination, bwd.termination)


def channel_connected_flux_tube(field: MagneticFieldMap, masks: MeshMasks, *, n_lines: int = 24, z_probe_m: Sequence[float] | None = None,
                                **kwargs: Any) -> dict[str, Any]:
    """Trace OUTWARD from the exit aperture: where the channel's flux tube runs in the plume.

    Returns the traced lines (as point arrays), their plume terminations, and for each probe
    plane z the radial band [r_min, r_max] of the tube (None where no line crosses the plane)."""

    grid = field.grid
    geometry = grid.geometry
    if not geometry.has_plume:
        raise PIC2DValidationError("channel_connected_flux_tube needs a plume geometry")
    z0 = geometry.z_max_m + 0.5 * grid.dz_m
    radii = (np.arange(n_lines) + 0.5) / n_lines * geometry.exit_radius_m
    lines: list[FieldLine] = []
    for r0 in radii:
        fwd, bwd = trace_both(field, masks, float(r0), z0, **kwargs)
        # the plume-side half is the one NOT entering the channel
        plume_half = bwd if fwd.termination == "channel" else fwd
        lines.append(plume_half)
    probes = list(z_probe_m) if z_probe_m is not None else list(geometry.z_max_m + grid.dz_m * np.arange(2, int(round(geometry.plume_length_m / grid.dz_m)), 8))
    bands: dict[str, list[float] | None] = {}
    for zp in probes:
        crossings: list[float] = []
        for line in lines:
            pts = line.points
            if pts.shape[0] < 2:
                continue
            z = pts[:, 1]
            hit = np.nonzero((z[:-1] - zp) * (z[1:] - zp) <= 0.0)[0]
            for k in hit:
                dz = z[k + 1] - z[k]
                t = 0.0 if dz == 0 else (zp - z[k]) / dz
                crossings.append(float(pts[k, 0] + t * (pts[k + 1, 0] - pts[k, 0])))
        bands[f"{zp:.6f}"] = [min(crossings), max(crossings)] if crossings else None
    terminations = {name: sum(1 for line in lines if line.termination == name) for name in TERMINATIONS}
    return {
        "start_radii_m": radii.tolist(), "start_z_m": z0, "lines": [line.to_dict() for line in lines], "line_points": [line.points for line in lines],
        "terminations": {k: v for k, v in terminations.items() if v}, "bands_by_probe_z_m": bands,
    }


def annulus_connectivity(field: MagneticFieldMap, masks: MeshMasks, r_inner_m: float, r_outer_m: float, z_start_m: float, z_end_m: float, *,
                         n_r: int = 6, n_z: int = 4, **kwargs: Any) -> dict[str, Any]:
    """Fraction of a cathode annulus (uniform sample) whose field lines enter the channel; per-sample terminations."""

    samples: list[dict[str, Any]] = []
    connected = 0
    for r0 in r_inner_m + (np.arange(n_r) + 0.5) / n_r * (r_outer_m - r_inner_m):
        for z0 in z_start_m + (np.arange(n_z) + 0.5) / n_z * (z_end_m - z_start_m):
            ok, (fwd, bwd) = connects_to_channel(field, masks, float(r0), float(z0), **kwargs)
            connected += int(ok)
            samples.append({"r_m": float(r0), "z_m": float(z0), "connected": ok, "forward": fwd, "backward": bwd})
    total = len(samples)
    ends = {}
    for s in samples:
        for key in ("forward", "backward"):
            ends[s[key]] = ends.get(s[key], 0) + 1
    return {"samples": samples, "connected_fraction": connected / total if total else 0.0, "terminations": ends, "n": total}


__all__ = [
    "FieldLine", "TERMINATIONS", "annulus_connectivity", "channel_connected_flux_tube", "classify_point", "connects_to_channel",
    "interpolate_b", "trace_both", "trace_field_line",
]
