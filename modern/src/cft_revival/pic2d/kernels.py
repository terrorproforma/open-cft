"""CPU reference particle kernels (vectorised numpy).

These define the numerical contract that the Warp backend must reproduce:

* bilinear (area) weighting in the ``(r, z)`` plane with node charges
  ``Q_n = sum_p q W S_n(r_p, z_p)``, optionally quantised to a fixed-point
  integer grid so that summation order cannot change the result;
* relativistic-momentum Boris rotation in the particle's meridional frame
  (``x`` radial, ``y`` azimuthal) with the exact operation order of
  ``cft_revival.orbit_mc.integrator.relativistic_boris_push``;
* Cartesian position advance followed by rotation back to the meridional
  frame (the standard axisymmetric PIC treatment of ``r = 0``);
* boundary classification against the plasma-cell mask and renormalised
  bilinear surface-charge deposition on the plasma-side wall nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import MeshMasks, cell_index
from .models import LIGHT_SPEED_M_PER_S, Grid2D, PIC2DValidationError, ParticleArrays, Species2D

FIXED_POINT_BITS = 40
FIXED_POINT_SCALE = float(2**FIXED_POINT_BITS)

BOUNDARY_INSIDE = 0
BOUNDARY_ANODE = 1
BOUNDARY_EXIT = 2
BOUNDARY_WALL = 3
BOUNDARY_INVALID = 4


def bilinear_weights(grid: Grid2D, r_m: np.ndarray, z_m: np.ndarray):
    i, j, s, t = cell_index(grid, r_m, z_m)
    w00 = (1.0 - s) * (1.0 - t)
    w10 = s * (1.0 - t)
    w01 = (1.0 - s) * t
    w11 = s * t
    return i, j, (w00, w10, w01, w11)


def deposit_node_charge(
    masks: MeshMasks,
    species: Species2D,
    particles: ParticleArrays,
    *,
    fixed_point: bool,
) -> np.ndarray:
    """Return node charges in coulombs for one species.

    ``fixed_point=True`` rounds each bilinear weight to ``2**-40`` and sums the
    integers exactly (the GPU uses the same integers with atomics), so the
    result is independent of particle order.  The quantisation error is at
    most ``2**-40 * |q| W`` per particle contribution.
    """

    grid = masks.grid
    i, j, weights = bilinear_weights(grid, particles.r_m, particles.z_m)
    per_particle = species.charge_c * species.macro_weight
    shape = grid.node_shape
    if fixed_point:
        accumulator = np.zeros(shape, dtype=np.int64)
        for di, dj, weight in ((0, 0, weights[0]), (1, 0, weights[1]), (0, 1, weights[2]), (1, 1, weights[3])):
            counts = np.rint(weight * FIXED_POINT_SCALE).astype(np.int64)
            np.add.at(accumulator, (i + di, j + dj), counts)
        return accumulator.astype(np.float64) * (per_particle / FIXED_POINT_SCALE)
    charge = np.zeros(shape, dtype=np.float64)
    for di, dj, weight in ((0, 0, weights[0]), (1, 0, weights[1]), (0, 1, weights[2]), (1, 1, weights[3])):
        np.add.at(charge, (i + di, j + dj), weight * per_particle)
    return charge


def deposit_node_moment(masks: MeshMasks, particles: ParticleArrays, values: np.ndarray) -> np.ndarray:
    """Float64 bilinear deposition of an arbitrary per-particle scalar (diagnostics)."""

    i, j, weights = bilinear_weights(masks.grid, particles.r_m, particles.z_m)
    out = np.zeros(masks.grid.node_shape, dtype=np.float64)
    for di, dj, weight in ((0, 0, weights[0]), (1, 0, weights[1]), (0, 1, weights[2]), (1, 1, weights[3])):
        np.add.at(out, (i + di, j + dj), weight * values)
    return out


def gather_nodes(grid: Grid2D, node_values: np.ndarray, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    i, j, (w00, w10, w01, w11) = bilinear_weights(grid, r_m, z_m)
    return (
        w00 * node_values[i, j]
        + w10 * node_values[i + 1, j]
        + w01 * node_values[i, j + 1]
        + w11 * node_values[i + 1, j + 1]
    )


def boris_push(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    ex: np.ndarray, ez: np.ndarray, bx: np.ndarray, bz: np.ndarray,
    charge_c: float, mass_kg: float, dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relativistic-momentum Boris push with ``E = (ex, 0, ez)``, ``B = (bx, 0, bz)``.

    Operation order matches ``orbit_mc.integrator.relativistic_boris_push``
    so single-particle results agree to roundoff.
    """

    c2 = LIGHT_SPEED_M_PER_S**2
    speed2 = vx * vx + vy * vy + vz * vz
    if np.any(speed2 >= c2):
        raise PIC2DValidationError("Boris input velocity is superluminal")
    gamma = 1.0 / np.sqrt(1.0 - speed2 / c2)
    ux = gamma * vx
    uy = gamma * vy
    uz = gamma * vz
    half_kick = charge_c * dt_s / (2.0 * mass_kg)
    ux_m = ux + half_kick * ex
    uy_m = uy  # E has no azimuthal component
    uz_m = uz + half_kick * ez
    gamma_m = np.sqrt(1.0 + (ux_m * ux_m + uy_m * uy_m + uz_m * uz_m) / c2)
    tx = charge_c * dt_s * bx / (2.0 * mass_kg * gamma_m)
    tz = charge_c * dt_s * bz / (2.0 * mass_kg * gamma_m)
    t2 = tx * tx + tz * tz
    sx = 2.0 * tx / (1.0 + t2)
    sz = 2.0 * tz / (1.0 + t2)
    # u' = u- + u- x t  with t = (tx, 0, tz)
    upx = ux_m + (uy_m * tz - uz_m * 0.0)
    upy = uy_m + (uz_m * tx - ux_m * tz)
    upz = uz_m + (ux_m * 0.0 - uy_m * tx)
    # u+ = u- + u' x s  with s = (sx, 0, sz)
    ux_p = ux_m + (upy * sz - upz * 0.0)
    uy_p = uy_m + (upz * sx - upx * sz)
    uz_p = uz_m + (upx * 0.0 - upy * sx)
    ux_n = ux_p + half_kick * ex
    uy_n = uy_p
    uz_n = uz_p + half_kick * ez
    gamma_n = np.sqrt(1.0 + (ux_n * ux_n + uy_n * uy_n + uz_n * uz_n) / c2)
    return ux_n / gamma_n, uy_n / gamma_n, uz_n / gamma_n


def advance_positions(
    r_m: np.ndarray, z_m: np.ndarray, vr: np.ndarray, vt: np.ndarray, vz: np.ndarray, dt_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance in the meridional Cartesian frame and rotate velocity back.

    Returns ``(r_new, z_new, vr_new, vt_new, cos_alpha, sin_alpha)`` where
    ``alpha`` is the azimuthal rotation of the particle's frame this step.
    """

    x = r_m + vr * dt_s
    y = vt * dt_s
    r_new = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_a = np.where(r_new > 0.0, x / r_new, 1.0)
        sin_a = np.where(r_new > 0.0, y / r_new, 0.0)
    vr_new = vr * cos_a + vt * sin_a
    vt_new = -vr * sin_a + vt * cos_a
    return r_new, z_m + vz * dt_s, vr_new, vt_new, cos_a, sin_a


def classify_boundary(masks: MeshMasks, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    """Return a boundary code per particle for positions *after* a push."""

    grid = masks.grid
    geometry = grid.geometry
    codes = np.full(r_m.shape, BOUNDARY_INSIDE, dtype=np.int8)
    anode = z_m < geometry.z_min_m
    exit_plane = z_m >= geometry.z_max_m
    codes[anode] = BOUNDARY_ANODE
    codes[exit_plane] = BOUNDARY_EXIT
    remaining = ~(anode | exit_plane)
    fr = r_m / grid.dr_m
    i = np.floor(fr).astype(np.int64)
    j = np.clip(np.floor((z_m - geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, grid.axial_cells - 1)
    beyond_box = i >= grid.radial_cells
    i_clipped = np.clip(i, 0, grid.radial_cells - 1)
    in_plasma = masks.plasma_cell[i_clipped, j] & ~beyond_box
    wall = remaining & ~in_plasma
    codes[wall] = BOUNDARY_WALL
    # A wall impact must land in a cell that still touches the plasma region
    # (at least one plasma node), otherwise the particle jumped more than one
    # cell: a Courant violation that fails closed.
    if np.any(wall):
        idx = np.flatnonzero(wall)
        ii = i[idx]
        jj = j[idx]
        touches = np.zeros(idx.shape, dtype=bool)
        # one cell beyond the outer box edge is still a wall hit on the outer
        # radial grid line (straight bores whose wall is the box edge)
        inside_box = ii <= grid.radial_cells
        ic = np.clip(ii, 0, grid.radial_cells - 1)
        for di in (0, 1):
            for dj in (0, 1):
                touches |= masks.plasma_node[ic + di, jj + dj]
        invalid = ~(touches & inside_box)
        codes[idx[invalid]] = BOUNDARY_INVALID
    return codes


def wall_surface_deposit(
    masks: MeshMasks, r_m: np.ndarray, z_m: np.ndarray, charge_c: np.ndarray, *, fixed_point: bool, quantum_c: float
) -> np.ndarray:
    """Deposit absorbed charge onto plasma-side wall nodes with renormalised bilinear weights.

    ``quantum_c`` is the fixed-point unit (``|q| W / 2**40``); ``charge_c`` must
    be an integer multiple of it in magnitude for the fixed-point path.
    """

    grid = masks.grid
    surface = np.zeros(grid.node_shape, dtype=np.float64)
    if r_m.size == 0:
        return surface
    fr = np.clip(r_m / grid.dr_m, 0.0, grid.radial_cells - 1e-12)
    fz = np.clip((z_m - grid.geometry.z_min_m) / grid.dz_m, 0.0, grid.axial_cells - 1e-12)
    i = np.floor(fr).astype(np.int64)
    j = np.floor(fz).astype(np.int64)
    s = fr - i
    t = fz - j
    raw = [(0, 0, (1 - s) * (1 - t)), (1, 0, s * (1 - t)), (0, 1, (1 - s) * t), (1, 1, s * t)]
    plasma = masks.plasma_node
    total = np.zeros_like(s)
    masked = []
    for di, dj, w in raw:
        keep = plasma[i + di, j + dj]
        w_kept = np.where(keep, w, 0.0)
        masked.append((di, dj, w_kept))
        total += w_kept
    if np.any(total <= 0.0):
        raise PIC2DValidationError("wall impact cell has no plasma node (particle jumped more than one cell)")
    if fixed_point:
        accumulator = np.zeros(grid.node_shape, dtype=np.int64)
        signs = np.sign(charge_c).astype(np.int64)
        for di, dj, w in masked:
            counts = np.rint(w / total * FIXED_POINT_SCALE).astype(np.int64) * signs
            np.add.at(accumulator, (i + di, j + dj), counts)
        return accumulator.astype(np.float64) * (quantum_c / FIXED_POINT_SCALE)
    for di, dj, w in masked:
        np.add.at(surface, (i + di, j + dj), w / total * charge_c)
    return surface


@dataclass(frozen=True, slots=True)
class BoundaryTally:
    anode: int
    exit: int
    wall: int
    invalid: int
    kinetic_energy_anode_j: float
    kinetic_energy_exit_j: float
    kinetic_energy_wall_j: float


def kinetic_energy_j(species: Species2D, particles: ParticleArrays) -> float:
    """Represented relativistic kinetic energy ``sum W (gamma-1) m c^2`` in joules."""

    c2 = LIGHT_SPEED_M_PER_S**2
    speed2 = particles.speed_squared()
    gamma_minus_one = speed2 / c2 / (1.0 + np.sqrt(1.0 - speed2 / c2))  # stable (gamma - 1)
    return float(np.sum(gamma_minus_one)) * species.mass_kg * c2 * species.macro_weight


__all__ = [
    "BOUNDARY_ANODE",
    "BOUNDARY_EXIT",
    "BOUNDARY_INSIDE",
    "BOUNDARY_INVALID",
    "BOUNDARY_WALL",
    "BoundaryTally",
    "FIXED_POINT_BITS",
    "FIXED_POINT_SCALE",
    "advance_positions",
    "bilinear_weights",
    "boris_push",
    "classify_boundary",
    "deposit_node_charge",
    "deposit_node_moment",
    "gather_nodes",
    "kinetic_energy_j",
    "wall_surface_deposit",
]
