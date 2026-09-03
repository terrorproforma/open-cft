"""Cell/node classification and finite-volume metrics for the (r,z) mesh.

The plasma region is the union of *plasma cells*: rectangular cells whose
outer radius lies inside the channel wall at the cell's lower-z edge (the wall
radius is non-decreasing in z, so this is the cell's smallest wall radius).
In the straight bore the wall coincides with a radial grid line and is exact;
in the divergent cone the wall is a stair-step approximation whose error is
one cell.  Everything else (volumes, conductances, wall nodes) is derived from
this single cell mask so particles, fields and diagnostics share one geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np

from .models import EPSILON_0_F_PER_M, Grid2D, PIC2DValidationError


@dataclass(frozen=True, slots=True)
class MeshMasks:
    """Immutable derived geometry for one ``Grid2D``.

    Shapes: node arrays are ``(nr+1, nz+1)``; cell arrays ``(nr, nz)``;
    ``cond_r`` is ``(nr, nz+1)`` (edge from node ``(i,j)`` to ``(i+1,j)``);
    ``cond_z`` is ``(nr+1, nz)`` (edge from node ``(i,j)`` to ``(i,j+1)``).
    Conductances are ``epsilon_0 * area / length`` in farads, so the discrete
    Gauss law reads ``sum_e C_e (phi_n - phi_m) = Q_n`` with ``Q_n`` in coulombs.
    """

    grid: Grid2D
    plasma_cell: np.ndarray
    plasma_node: np.ndarray
    dirichlet_node: np.ndarray
    anode_node: np.ndarray
    exit_node: np.ndarray
    unknown_node: np.ndarray
    wall_node: np.ndarray
    axis_node: np.ndarray
    cond_r: np.ndarray
    cond_z: np.ndarray
    diagonal: np.ndarray
    shape_volume_m3: np.ndarray
    geometric_volume_m3: np.ndarray
    charge_to_source: np.ndarray
    top_plasma_cell: np.ndarray
    plasma_volume_m3: float
    # v2.0 plume block (all-False / zero volume for a channel-only geometry):
    # ``far_field_node``: outer plume boundary (r = R_plume for z >= z_exit, and z = z_max), Dirichlet at
    # the reference potential; ``body_face_node``: front face of the thruster (z = z_exit, r > r_exit) -
    # dielectric wall nodes up to ``body_dielectric_radius_m`` and grounded conductor nodes
    # (``body_conductor_node``, Dirichlet at the reference potential) beyond; ``plume_cell``: cells
    # downstream of the exit plane.
    far_field_node: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    body_face_node: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    body_conductor_node: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    plume_cell: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    channel_volume_m3: float = 0.0

    @property
    def unknown_count(self) -> int:
        return int(np.count_nonzero(self.unknown_node))

    @property
    def has_plume(self) -> bool:
        return bool(self.grid.geometry.has_plume)

    def to_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "plasma_cells": int(np.count_nonzero(self.plasma_cell)),
            "plasma_nodes": int(np.count_nonzero(self.plasma_node)),
            "dirichlet_nodes": int(np.count_nonzero(self.dirichlet_node)),
            "unknown_nodes": self.unknown_count,
            "wall_nodes": int(np.count_nonzero(self.wall_node)),
            "plasma_volume_m3": self.plasma_volume_m3,
            "wall_representation": "exact radial grid line in the straight bore; one-cell stair-step in the cone",
        }
        if self.has_plume:
            record["plume"] = {
                "channel_cells": int(np.count_nonzero(self.plasma_cell & ~self.plume_cell)),
                "plume_cells": int(np.count_nonzero(self.plume_cell)),
                "channel_volume_m3": self.channel_volume_m3,
                "plume_volume_m3": self.plasma_volume_m3 - self.channel_volume_m3,
                "far_field_nodes": int(np.count_nonzero(self.far_field_node)),
                "body_face_nodes": int(np.count_nonzero(self.body_face_node)),
                "body_dielectric_nodes": int(np.count_nonzero(self.body_face_node & ~self.body_conductor_node)),
                "body_conductor_nodes": int(np.count_nonzero(self.body_conductor_node)),
                "boundary_representation": (
                    "channel walls, exit lip and front face are internal boundaries of one bounding box (exit plane, exit lip "
                    "and dielectric/conductor split on grid lines); far field Dirichlet at the reference potential"
                ),
            }
        return record


def build_mesh_masks(grid: Grid2D) -> MeshMasks:
    nr, nz = grid.cell_shape
    r = grid.r_m
    z = grid.z_m
    dr = grid.dr_m
    dz = grid.dz_m
    wall_at_cell_low_z = grid.geometry.wall_radius_m(z[:-1])  # (nz,)
    outer_radius = r[1:]  # (nr,)
    tolerance = 1.0e-9 * dr
    plasma_cell = outer_radius[:, None] <= wall_at_cell_low_z[None, :] + tolerance
    if not plasma_cell.any():
        raise PIC2DValidationError("no plasma cell: grid is coarser than the channel")
    if not plasma_cell[0, :].all():
        raise PIC2DValidationError("the axis cell column must be plasma everywhere")

    plasma_node = np.zeros((nr + 1, nz + 1), dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            plasma_node[di:nr + di, dj:nz + dj] |= plasma_cell

    anode_node = np.zeros_like(plasma_node)
    anode_node[:, 0] = plasma_node[:, 0]
    exit_node = np.zeros_like(plasma_node)
    exit_node[:, nz] = plasma_node[:, nz]
    far_field_node = np.zeros_like(plasma_node)
    body_face_node = np.zeros_like(plasma_node)
    body_conductor_node = np.zeros_like(plasma_node)
    plume_cell = np.zeros_like(plasma_cell)
    geometry = grid.geometry
    if geometry.has_plume:
        # v2.0: the far field is the whole outer plume boundary; the front face nodes are the
        # exit-plane row outside the exit lip (dielectric ring, then grounded conductor).
        j_exit = int(round(geometry.channel_length_m / dz))
        i_exit = int(round(geometry.exit_radius_m / dr))
        i_body = int(round(float(geometry.body_dielectric_radius_m) / dr))
        plume_cell[:, j_exit:] = plasma_cell[:, j_exit:]
        far_field_node[:, nz] = plasma_node[:, nz]
        far_field_node[nr, j_exit:] = plasma_node[nr, j_exit:]
        body_face_node[i_exit + 1:, j_exit] = plasma_node[i_exit + 1:, j_exit] & ~far_field_node[i_exit + 1:, j_exit]
        body_conductor_node[i_body + 1:, j_exit] = body_face_node[i_body + 1:, j_exit]
        exit_node = far_field_node | body_conductor_node
    dirichlet_node = anode_node | exit_node
    unknown_node = plasma_node & ~dirichlet_node
    axis_node = np.zeros_like(plasma_node)
    axis_node[0, :] = plasma_node[0, :]

    # A node is a wall node when it is an unknown plasma node adjacent to at
    # least one existing non-plasma cell.
    outside_cell = ~plasma_cell
    touches_outside = np.zeros_like(plasma_node)
    for di in (0, 1):
        for dj in (0, 1):
            touches_outside[di:nr + di, dj:nz + dj] |= outside_cell
    wall_node = unknown_node & touches_outside

    r_mid = 0.5 * (r[:-1] + r[1:])  # (nr,)
    # Radial edge conductance per plasma cell: face at r_{i+1/2}, half the cell height.
    radial_edge = EPSILON_0_F_PER_M * 2.0 * pi * r_mid * (0.5 * dz) / dr  # (nr,)
    cond_r = np.zeros((nr, nz + 1), dtype=np.float64)
    contribution = plasma_cell * radial_edge[:, None]
    cond_r[:, :-1] += contribution
    cond_r[:, 1:] += contribution
    # Axial edge conductances: inner node piece [r_i, r_{i+1/2}], outer piece [r_{i+1/2}, r_{i+1}].
    inner_axial = EPSILON_0_F_PER_M * pi * (r_mid**2 - r[:-1] ** 2) / dz  # (nr,)
    outer_axial = EPSILON_0_F_PER_M * pi * (r[1:] ** 2 - r_mid**2) / dz  # (nr,)
    cond_z = np.zeros((nr + 1, nz), dtype=np.float64)
    cond_z[:-1, :] += plasma_cell * inner_axial[:, None]
    cond_z[1:, :] += plasma_cell * outer_axial[:, None]

    diagonal = np.zeros((nr + 1, nz + 1), dtype=np.float64)
    diagonal[:-1, :] += cond_r
    diagonal[1:, :] += cond_r
    diagonal[:, :-1] += cond_z
    diagonal[:, 1:] += cond_z

    # Shape-function (bilinear hat) node volumes: integral of S_n over the plasma
    # region with the 2*pi*r measure.  Uniform density deposits to exactly
    # rho = n q with these volumes.
    inner_shape = 2.0 * pi * dr * (2.0 * r[:-1] + r[1:]) / 6.0 * (0.5 * dz)  # (nr,)
    outer_shape = 2.0 * pi * dr * (r[:-1] + 2.0 * r[1:]) / 6.0 * (0.5 * dz)
    shape_volume = np.zeros((nr + 1, nz + 1), dtype=np.float64)
    for dj in (0, 1):
        shape_volume[:-1, dj:nz + dj] += plasma_cell * inner_shape[:, None]
        shape_volume[1:, dj:nz + dj] += plasma_cell * outer_shape[:, None]
    # Geometric control volumes (for reference/diagnostics only).
    inner_geometric = pi * (r_mid**2 - r[:-1] ** 2) * (0.5 * dz)
    outer_geometric = pi * (r[1:] ** 2 - r_mid**2) * (0.5 * dz)
    geometric_volume = np.zeros_like(shape_volume)
    for dj in (0, 1):
        geometric_volume[:-1, dj:nz + dj] += plasma_cell * inner_geometric[:, None]
        geometric_volume[1:, dj:nz + dj] += plasma_cell * outer_geometric[:, None]

    # Volume-deposited node charge Q_n estimates rho_n = Q_n / V_shape; the
    # finite-volume Gauss law needs rho_n * V_geometric.  The ratio is exactly 1
    # on interior nodes and 3/4 on the axis (Verboncoeur 2001, JCP 174:421).
    # Surface charge on wall nodes enters the Gauss law directly (ratio 1).
    charge_to_source = np.zeros_like(shape_volume)
    charge_to_source[plasma_node] = geometric_volume[plasma_node] / shape_volume[plasma_node]

    top_plasma_cell = np.full(nz, -1, dtype=np.int64)
    for j in range(nz):
        column = np.flatnonzero(plasma_cell[:, j])
        top_plasma_cell[j] = int(column.max())
        if not plasma_cell[: top_plasma_cell[j] + 1, j].all():
            raise PIC2DValidationError("plasma cells must be contiguous from the axis")

    cell_volume = pi * (r[1:] ** 2 - r[:-1] ** 2) * dz
    plasma_volume = float(np.sum(plasma_cell * cell_volume[:, None]))
    if not np.isclose(plasma_volume, float(shape_volume.sum()), rtol=1e-12, atol=0.0):
        raise PIC2DValidationError("shape volumes do not partition the plasma volume")
    if not np.isclose(plasma_volume, float(geometric_volume.sum()), rtol=1e-12, atol=0.0):
        raise PIC2DValidationError("geometric volumes do not partition the plasma volume")

    channel_volume = float(np.sum((plasma_cell & ~plume_cell) * cell_volume[:, None])) if geometry.has_plume else plasma_volume
    return MeshMasks(
        grid, plasma_cell, plasma_node, dirichlet_node, anode_node, exit_node,
        unknown_node, wall_node, axis_node, cond_r, cond_z, diagonal,
        shape_volume, geometric_volume, charge_to_source, top_plasma_cell, plasma_volume,
        far_field_node=far_field_node, body_face_node=body_face_node, body_conductor_node=body_conductor_node,
        plume_cell=plume_cell, channel_volume_m3=channel_volume,
    )


def cell_index(grid: Grid2D, r_m: np.ndarray, z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (i, j, s, t): containing cell indices and unit-cell fractions.

    Positions on the outer boundary are clamped into the last cell so that a
    particle exactly on ``r = r_max`` or ``z = z_max`` still has a cell.
    """

    fr = r_m / grid.dr_m
    fz = (z_m - grid.geometry.z_min_m) / grid.dz_m
    i = np.clip(np.floor(fr).astype(np.int64), 0, grid.radial_cells - 1)
    j = np.clip(np.floor(fz).astype(np.int64), 0, grid.axial_cells - 1)
    return i, j, fr - i, fz - j


__all__ = ["MeshMasks", "build_mesh_masks", "cell_index"]
