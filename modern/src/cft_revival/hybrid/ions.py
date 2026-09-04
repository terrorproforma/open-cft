"""Kinetic Xe+ macroparticles of the L2 v2 hybrid on the PIC's (r,z) mesh.

The ion push reuses the PIC's CPU reference kernels verbatim (``cft_revival.pic2d.kernels``:
bilinear gather, relativistic-momentum Boris rotation in the meridional frame, Cartesian
advance with rotation back, boundary classification against the plasma-cell mask and the
renormalised bilinear wall-charge deposit), so the ion numerics of L2 are those of the PIC's
ions - the Warp parity of those kernels is established in ``tests/pic2d``.  What differs is the
time step (nanoseconds: the electron time scales are not resolved) and the field (the
per-cell Poisson-Boltzmann potential).

Ion births sample the fluid ionisation source ``S(node) = n_e(node) n_g k_iz(T_k) V_node`` of
the node they are born at and start with a Maxwellian velocity at the neutral temperature,
as the PIC's MCC ions do.  A carry keeps the expected number of births exact in the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from ..pic2d.kernels import (
    BOUNDARY_ANODE,
    BOUNDARY_EXIT,
    BOUNDARY_INSIDE,
    BOUNDARY_INVALID,
    BOUNDARY_WALL,
    advance_positions,
    boris_push,
    classify_boundary,
    deposit_node_charge,
    gather_nodes,
    kinetic_energy_j,
    wall_surface_deposit,
)
from ..pic2d.mcc import maxwellian_velocity
from ..pic2d.mesh import MeshMasks, cell_index
from ..pic2d.models import ParticleArrays, PIC2DValidationError, Species2D
from .cells import CellPartition
from .models import HybridValidationError


@dataclass(frozen=True, slots=True)
class IonPushTally:
    """Boundary exchange of one ion push (counts are macroparticles; energies represented joules)."""

    anode: int
    exit: int
    wall: int
    ke_anode_j: float
    ke_exit_j: float
    ke_wall_j: float
    field_work_j: float
    wall_hits_per_axial_cell: np.ndarray        # (nz,) macroparticle counts
    wall_energy_per_axial_cell_j: np.ndarray    # (nz,) represented joules
    exit_hits_per_radial_cell: np.ndarray       # (nr,) macroparticle counts
    wall_hits_per_cell: np.ndarray              # (K,) by partition cell of the impact
    anode_hits: int
    surface_deposit_c: np.ndarray               # node array (C) deposited on the wall this push
    wall_impact_z_m: np.ndarray                 # positions of the wall impacts (for diagnostics)


class IonPopulation:
    """Structure-of-arrays Xe+ population with the PIC kernels as its numerical contract."""

    def __init__(self, species: Species2D, particles: ParticleArrays | None = None) -> None:
        if species.charge_c <= 0.0:
            raise HybridValidationError("the kinetic species must be a positive ion")
        self.species = species
        self.particles = ParticleArrays.empty() if particles is None else particles

    @property
    def count(self) -> int:
        return self.particles.count

    def kinetic_energy_j(self) -> float:
        return kinetic_energy_j(self.species, self.particles)

    def deposit_charge_c(self, masks: MeshMasks) -> np.ndarray:
        """Bilinear node charge (C) of the population (``x^n``); multiply by ``charge_to_source`` for Gauss."""

        return deposit_node_charge(masks, self.species, self.particles, fixed_point=False)

    def push(
        self,
        masks: MeshMasks,
        *,
        e_r_nodes: np.ndarray,
        e_z_nodes: np.ndarray,
        b_r_nodes: np.ndarray,
        b_z_nodes: np.ndarray,
        dt_s: float,
        partition: CellPartition,
    ) -> IonPushTally:
        """Leapfrog ``v^(n-1/2) -> v^(n+1/2)``, ``x^n -> x^(n+1)`` with boundary absorption.

        ``partition`` maps wall-impact axial positions to electron cells for the per-cell tally.
        """

        grid = masks.grid
        nr, nz = grid.cell_shape
        p = self.particles
        cell_hits = np.zeros(partition.cell_count, dtype=np.float64)
        if p.count == 0:
            zeros_z = np.zeros(nz, dtype=np.float64)
            return IonPushTally(0, 0, 0, 0.0, 0.0, 0.0, 0.0, zeros_z.copy(), zeros_z.copy(), np.zeros(nr), cell_hits, 0,
                                np.zeros(grid.node_shape), np.zeros(0))
        e_r = gather_nodes(grid, e_r_nodes, p.r_m, p.z_m)
        e_z = gather_nodes(grid, e_z_nodes, p.r_m, p.z_m)
        b_r = gather_nodes(grid, b_r_nodes, p.r_m, p.z_m)
        b_z = gather_nodes(grid, b_z_nodes, p.r_m, p.z_m)
        ke_before = kinetic_energy_j(self.species, p)
        vr, vt, vz = boris_push(p.vr_m_per_s, p.vt_m_per_s, p.vz_m_per_s, e_r, e_z, b_r, b_z,
                                self.species.charge_c, self.species.mass_kg, dt_s)
        r_new, z_new, vr_new, vt_new, _, _ = advance_positions(p.r_m, p.z_m, vr, vt, vz, dt_s)
        moved = ParticleArrays(r_new, z_new, vr_new, vt_new, vz)
        field_work = kinetic_energy_j(self.species, moved) - ke_before
        codes = classify_boundary(masks, moved.r_m, moved.z_m)
        if np.any(codes == BOUNDARY_INVALID):
            raise PIC2DValidationError("an ion jumped more than one cell (Courant violation); reduce dt")
        inside = codes == BOUNDARY_INSIDE
        anode = codes == BOUNDARY_ANODE
        exit_plane = codes == BOUNDARY_EXIT
        wall = codes == BOUNDARY_WALL
        w = self.species.macro_weight
        energy_each = 0.5 * self.species.mass_kg * moved.speed_squared() * w
        wall_hits_z = np.zeros(nz, dtype=np.float64)
        wall_energy_z = np.zeros(nz, dtype=np.float64)
        exit_hits_r = np.zeros(nr, dtype=np.float64)
        surface = np.zeros(grid.node_shape, dtype=np.float64)
        impact_z = moved.z_m[wall]
        if np.any(wall):
            _, j, _, _ = cell_index(grid, moved.r_m[wall], moved.z_m[wall])
            np.add.at(wall_hits_z, j, 1.0)
            np.add.at(wall_energy_z, j, energy_each[wall])
            charge = np.full(int(wall.sum()), self.species.charge_c * w, dtype=np.float64)
            surface = wall_surface_deposit(masks, moved.r_m[wall], moved.z_m[wall], charge, fixed_point=False, quantum_c=abs(self.species.charge_c * w))
            cells = partition.cell_of_z(np.clip(impact_z, partition.z_min_m, partition.z_max_m))
            np.add.at(cell_hits, cells, 1.0)
        if np.any(exit_plane):
            i, _, _, _ = cell_index(grid, np.clip(moved.r_m[exit_plane], 0.0, grid.geometry.max_radius_m), moved.z_m[exit_plane])
            np.add.at(exit_hits_r, i, 1.0)
        self.particles = moved.select(inside)
        return IonPushTally(
            int(anode.sum()), int(exit_plane.sum()), int(wall.sum()),
            float(energy_each[anode].sum()), float(energy_each[exit_plane].sum()), float(energy_each[wall].sum()),
            float(field_work), wall_hits_z, wall_energy_z, exit_hits_r, cell_hits, int(anode.sum()), surface, impact_z,
        )

    def add(self, newborn: ParticleArrays) -> None:
        if newborn.count:
            self.particles = self.particles.append(newborn)


def sample_births(
    masks: MeshMasks,
    node_weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    mass_kg: float,
    temperature_k: float,
) -> ParticleArrays:
    """``count`` ions at positions drawn from the node source weights, Maxwellian at ``temperature_k``.

    Positions are uniform in the node's control volume (2 pi r measure radially), then clipped into the
    plasma cell mask (a node on the wall owns half a control volume outside the plasma).
    """

    if count <= 0:
        return ParticleArrays.empty()
    grid = masks.grid
    weights = np.asarray(node_weights, dtype=np.float64).ravel()
    if weights.shape[0] != grid.node_shape[0] * grid.node_shape[1] or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise HybridValidationError("node source weights must be a finite non-negative node array")
    total = float(weights.sum())
    if total <= 0.0:
        raise HybridValidationError("cannot sample births from an all-zero source")
    flat = rng.choice(weights.size, size=count, p=weights / total)
    i, j = np.unravel_index(flat, grid.node_shape)
    dr, dz = grid.dr_m, grid.dz_m
    r_lo = np.maximum(i * dr - 0.5 * dr, 0.0)
    r_hi = np.minimum(i * dr + 0.5 * dr, grid.geometry.max_radius_m)
    u = rng.random((6, count))
    r = np.sqrt(u[0] * (r_hi**2 - r_lo**2) + r_lo**2)
    z_lo = np.maximum(grid.geometry.z_min_m + j * dz - 0.5 * dz, grid.geometry.z_min_m)
    z_hi = np.minimum(grid.geometry.z_min_m + j * dz + 0.5 * dz, grid.geometry.domain_z_max_m)
    z = z_lo + u[1] * (z_hi - z_lo)
    # keep births strictly inside the plasma cells: clamp the radius under the plasma-cell mask of the axial column
    z = np.clip(z, grid.geometry.z_min_m + 1e-9 * dz, grid.geometry.domain_z_max_m - 1e-9 * dz)
    column = np.clip(np.floor((z - grid.geometry.z_min_m) / dz).astype(np.int64), 0, grid.axial_cells - 1)
    top_radius = (masks.top_plasma_cell[column] + 1) * dr
    r = np.minimum(r, top_radius * (1.0 - 1e-9))
    codes = classify_boundary(masks, r, z)
    if np.any(codes != BOUNDARY_INSIDE):
        raise HybridValidationError("a sampled ion birth lies outside the plasma region")
    vx, vy, vz = maxwellian_velocity(mass_kg, temperature_k, u[2:6])
    return ParticleArrays(r, z, vx, vy, vz)


def births_this_step(expected: float, carry: float) -> tuple[int, float]:
    """Deterministic carry rounding: returns (integer births, new carry) with exact mean."""

    if not isfinite(expected) or expected < 0.0:
        raise HybridValidationError("expected births must be finite and non-negative")
    total = expected + carry
    n = int(np.floor(total))
    return n, float(total - n)


def uniform_seed_ions(
    masks: MeshMasks, density_per_m3: float, macro_weight: float, rng: np.random.Generator, *, mass_kg: float, temperature_k: float,
    accept_node: np.ndarray | None = None, accept_volume_m3: float | None = None,
) -> ParticleArrays:
    """Uniform seed population over the plasma cells (rejection sampling in the bounding box).

    With ``accept_node`` (a node mask) only positions whose nearest node is in the mask are kept, and the
    represented volume is ``accept_volume_m3`` (the electron-populated volume) instead of the plasma volume.
    """

    grid = masks.grid
    volume = masks.plasma_volume_m3 if accept_volume_m3 is None else float(accept_volume_m3)
    expected = density_per_m3 * volume / macro_weight
    count = round(expected)
    if count <= 0:
        return ParticleArrays.empty()
    r_max = grid.geometry.max_radius_m
    accepted_r: list[np.ndarray] = []
    accepted_z: list[np.ndarray] = []
    remaining = count
    guard = 0
    while remaining > 0 and guard < 1000:
        guard += 1
        batch = int(remaining * 1.5) + 16
        u = rng.random((2, batch))
        r = r_max * np.sqrt(u[0])
        z = grid.geometry.z_min_m + u[1] * grid.geometry.length_m
        inside = classify_boundary(masks, r, z) == BOUNDARY_INSIDE
        if accept_node is not None:
            i = np.clip(np.rint(r / grid.dr_m).astype(np.int64), 0, grid.radial_cells)
            j = np.clip(np.rint((z - grid.geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, grid.axial_cells)
            inside &= np.asarray(accept_node, dtype=bool)[i, j]
        accepted_r.append(r[inside][:remaining])
        accepted_z.append(z[inside][:remaining])
        remaining -= int(min(inside.sum(), remaining))
    r = np.concatenate(accepted_r)
    z = np.concatenate(accepted_z)
    if temperature_k > 0.0:
        vx, vy, vz = maxwellian_velocity(mass_kg, temperature_k, rng.random((4, r.size)))
    else:
        vx = vy = vz = np.zeros(r.size)
    return ParticleArrays(r, z, vx, vy, vz)


def population_summary(population: IonPopulation) -> dict[str, Any]:
    return {"count": population.count, "macro_weight": population.species.macro_weight, "kinetic_energy_j": population.kinetic_energy_j()}


__all__ = [
    "BOUNDARY_ANODE",
    "BOUNDARY_EXIT",
    "BOUNDARY_WALL",
    "IonPopulation",
    "IonPushTally",
    "births_this_step",
    "population_summary",
    "sample_births",
    "uniform_seed_ions",
]
