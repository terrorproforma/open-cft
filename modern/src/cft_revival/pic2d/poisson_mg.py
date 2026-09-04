"""Geometric multigrid for the masked cylindrical finite-volume Poisson operator.

This module holds the *host* side of the ``device-mg`` field solve: the level
hierarchy (shared by the numpy reference solver below and the Warp kernels in
``warp_poisson_mg.py``) and a numpy V-cycle solver that is the CPU counterpart of
``WarpPoissonMG`` (same hierarchy, same sweeps, same fixed cycle count).

Design
------
* **Unknowns and operator.**  Level 0 is the node mesh of ``mesh.py``: the unknown
  nodes are the plasma nodes that are not Dirichlet (anode, exit plane, far field,
  grounded body conductor), and the operator is the conductance graph Laplacian
  ``A_uu`` restricted to the unknowns (edge conductances ``epsilon_0 A / l`` with the
  ``2 pi r`` weighting of the axisymmetric finite volumes, homogeneous Neumann into
  the dielectric solids, the Dirichlet couplings moved to the right-hand side by
  ``poisson.apply_operator`` exactly as the block-Thomas paths do).  The contract is
  the one every field solve in this package publishes against: the *true* residual
  ``|Q - A phi|`` recomputed with the mesh conductances (not with the multigrid's own
  stencil arrays) must satisfy ``<= max(absolute_tolerance, relative_tolerance |rhs|)``.
* **Coarsening.**  Vertex-centred coarsening by two in ``r`` and ``z``: the coarse
  nodes are the even fine nodes plus the last node of an axis with an odd number of
  cells (so 90 x 720 coarsens to 46 x 361, not to a grid that misses the wall).  The
  coarse unknown set is the image of the fine unknown set, so the channel bore, the
  stair-stepped cone, the exit lip, the dielectric body face and the electrodes are
  represented on every level by the *operator*, not by a re-classified mask.
* **Transfer operators.**  Operator-dependent (Alcouffe / Dendy "black-box")
  interpolation built from the level's own stencil: a fine node between two coarse
  nodes is interpolated with the collapsed-stencil weights (conductance-proportional
  on level 0), a fine cell-centre node from the four corners through the fine
  equation.  A coupling to a Dirichlet node is absent from ``A_uu`` while its
  conductance stays in the diagonal, so the weights decay towards electrodes; a
  solid neighbour has no conductance, so the weights reflect across the dielectric
  walls.  Restriction is the transpose.
* **Coarse operators.**  Galerkin: ``A_c = P^T A P`` (a symmetric 9-point stencil on
  every coarse level; the symmetry of the assembled product is checked to round-off,
  which is the self-test of the construction).  The recursion stops when the active
  count drops below ``mg_coarsest_max_unknowns`` and the coarsest operator is
  inverted densely once on the host (a few hundred unknowns: kilobytes to a few MB).
* **Cycle.**  V(nu1, nu2) with damped Jacobi sweeps (fully parallel, deterministic,
  one kernel per sweep on the device), a fused residual-and-restrict step, and a
  *fixed* number of cycles so the whole solve is a fixed kernel sequence that the
  step graph can capture.  The solve is warm-started from the previous potential.
  Convergence is not iterated on: it is *verified* against the contract (every step
  on the device by an in-graph running maximum of the contract ratio, plus the
  independent mesh-conductance residual read at each host sync) and the run stops
  fail-closed when a fixed-count solve misses it.

The stencil layout used by both solvers: node ``(i, j)`` at flat index ``i * nj + j``;
``coef[n, k]`` couples node ``n`` to node ``n + offset(k)`` with
``k = 3 (di + 1) + (dj + 1)`` (``k = 4`` is the diagonal); entries to inactive or
non-existent nodes are exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .mesh import MeshMasks
from .models import BoundaryPotentials, PIC2DConvergenceError, PIC2DValidationError, PoissonConfig2D
from .poisson import PoissonDiagnostics2D, PoissonResult2D, apply_operator, boundary_potential_array

# k -> (di, dj); k = 3 (di + 1) + (dj + 1)
STENCIL_OFFSETS: tuple[tuple[int, int], ...] = tuple((di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1))
DIAGONAL_SLOT = 4


@dataclass(frozen=True, slots=True)
class MGLevel:
    """One level of the hierarchy (arrays are flat over the ``ni x nj`` node grid)."""

    ni: int
    nj: int
    fine_i: np.ndarray            # (ni,) row index of each node on the next-finer level (level 0: arange)
    fine_j: np.ndarray            # (nj,)
    active: np.ndarray            # (ni*nj,) bool: unknown nodes of this level
    coef: np.ndarray              # (ni*nj, 9) float64 signed stencil; coef[:, 4] is the diagonal
    inv_diag: np.ndarray          # (ni*nj,) 1 / diagonal on active nodes, 0 elsewhere
    nbr: np.ndarray               # (ni*nj, 9) int64 flat neighbour index per stencil slot, -1 if none
    p_idx: np.ndarray | None      # (ni*nj, 4) int64 coarse parents (flat, next-coarser level) or -1
    p_w: np.ndarray | None        # (ni*nj, 4) interpolation weights
    r_idx: np.ndarray | None      # (nc, 9) int64 fine children of each coarse node or -1
    r_w: np.ndarray | None        # (nc, 9) restriction weights (= the children's weights towards this parent)

    @property
    def node_count(self) -> int:
        return self.ni * self.nj

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active))


@dataclass(frozen=True, slots=True)
class MGHierarchy:
    levels: tuple[MGLevel, ...]
    coarsest_active_index: np.ndarray     # (na,) flat indices of the coarsest active nodes
    coarsest_inverse: np.ndarray          # (na, na) dense inverse of the coarsest active operator

    @property
    def depth(self) -> int:
        return len(self.levels)

    def to_dict(self) -> dict[str, object]:
        return {
            "levels": [
                {"nodes": [level.ni, level.nj], "unknowns": level.active_count} for level in self.levels
            ],
            "coarsest_unknowns": int(self.coarsest_active_index.size),
            "coarsest_inverse_bytes": int(self.coarsest_inverse.nbytes),
            "transfer": "operator-dependent (collapsed-stencil) interpolation, transpose restriction",
            "coarse_operator": "Galerkin P^T A P (9-point)",
        }


def coarse_axis(node_count: int) -> np.ndarray:
    """Fine indices of the coarse nodes along one axis with ``node_count`` nodes."""

    last = node_count - 1
    if last < 1:
        raise PIC2DValidationError("an axis needs at least two nodes to coarsen")
    coarse = np.arange(0, last + 1, 2, dtype=np.int64)
    if last % 2 == 1:
        coarse = np.append(coarse, last)
    return coarse


def neighbour_table(ni: int, nj: int) -> np.ndarray:
    """(ni*nj, 9) flat neighbour indices per stencil slot (-1 outside the grid)."""

    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    table = np.full((ni * nj, 9), -1, dtype=np.int64)
    for k, (di, dj) in enumerate(STENCIL_OFFSETS):
        i2 = ii + di
        j2 = jj + dj
        valid = (i2 >= 0) & (i2 < ni) & (j2 >= 0) & (j2 < nj)
        flat = np.where(valid, i2 * nj + j2, -1)
        table[:, k] = flat.ravel()
    return table


def finest_stencil(masks: MeshMasks) -> tuple[np.ndarray, np.ndarray]:
    """``(active, coef)`` of ``A_uu`` on the node mesh in the 9-slot layout (5-point content)."""

    nr, nz = masks.grid.cell_shape
    ni, nj = nr + 1, nz + 1
    unknown = masks.unknown_node
    coef = np.zeros((ni, nj, 9), dtype=np.float64)
    coef[..., DIAGONAL_SLOT] = np.where(unknown, masks.diagonal, 0.0)
    both_z = unknown[:, :-1] & unknown[:, 1:]
    east = np.where(both_z, -masks.cond_z, 0.0)             # (ni, nz): (i,j) <-> (i,j+1)
    coef[:, :-1, 5] = east
    coef[:, 1:, 3] = east
    both_r = unknown[:-1, :] & unknown[1:, :]
    north = np.where(both_r, -masks.cond_r, 0.0)            # (nr, nj): (i,j) <-> (i+1,j)
    coef[:-1, :, 7] = north
    coef[1:, :, 1] = north
    return unknown.ravel().copy(), coef.reshape(ni * nj, 9)


def _index_maps(node_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = coarse_axis(node_count)
    is_coarse = np.zeros(node_count, dtype=bool)
    is_coarse[coarse] = True
    coarse_index = np.full(node_count, -1, dtype=np.int64)
    coarse_index[coarse] = np.arange(coarse.size)
    return coarse, is_coarse, coarse_index


def build_prolongation(
    ni: int, nj: int, active: np.ndarray, coef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Operator-dependent interpolation from the coarse level implied by ``coarse_axis``.

    Returns ``(fine_i, fine_j, active_coarse, p_idx, p_w)`` where ``fine_i``/``fine_j``
    are the fine indices of the coarse rows/columns, ``active_coarse`` the coarse unknown
    mask (flat), and ``p_idx``/``p_w`` the up-to-four coarse parents of every fine node.
    """

    ci, row_is_coarse, row_ci = _index_maps(ni)
    cj, col_is_coarse, col_cj = _index_maps(nj)
    ncj = cj.size
    act = active.reshape(ni, nj)
    c = coef.reshape(ni, nj, 9)
    active_coarse = act[np.ix_(ci, cj)].ravel().copy()

    def coarse_flat(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        ii = row_ci[i]
        jj = col_cj[j]
        return np.where((ii >= 0) & (jj >= 0), ii * ncj + jj, -1)

    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    type_cc = row_is_coarse[:, None] & col_is_coarse[None, :]          # coincident with a coarse node
    type_r = ~row_is_coarse[:, None] & col_is_coarse[None, :]           # between two coarse rows
    type_c = row_is_coarse[:, None] & ~col_is_coarse[None, :]           # between two coarse columns
    type_x = ~row_is_coarse[:, None] & ~col_is_coarse[None, :]          # cell centre

    def shifted_active(di: int, dj: int) -> np.ndarray:
        """``act`` read at ``(i + di, j + dj)`` (False outside the grid)."""

        out = np.zeros_like(act)
        src_i = slice(max(0, di), ni + min(0, di))
        dst_i = slice(max(0, -di), ni + min(0, -di))
        src_j = slice(max(0, dj), nj + min(0, dj))
        dst_j = slice(max(0, -dj), nj + min(0, -dj))
        out[dst_i, dst_j] = act[src_i, src_j]
        return out

    def collapsed(sel: np.ndarray, den: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray,
                  lo_active: np.ndarray, hi_active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Collapsed-stencil weights of the two parents of a line node.

        The couplings of the three stencil entries towards each parent are lumped into
        that parent (Dendy).  At a concave corner of the stair-stepped wall a parent can be
        a solid node while the diagonal entries towards its side are non-zero (9-point
        coarse operators); lumping them into a node that does not exist would lose that
        mass, so it goes to the other parent - constants stay exactly preserved on every
        pure-Neumann row, which is what the coarse-grid correction of smooth errors needs.
        """

        with np.errstate(divide="ignore", invalid="ignore"):
            ok = sel & act & (den > 0.0)
            f_lo = np.where(ok, s_lo / den, 0.0)
            f_hi = np.where(ok, s_hi / den, 0.0)
        w_lo = np.where(lo_active, f_lo + np.where(hi_active, 0.0, f_hi), 0.0)
        w_hi = np.where(hi_active, f_hi + np.where(lo_active, 0.0, f_lo), 0.0)
        return w_lo, w_hi

    # between rows (i odd): parents (i-1, j) and (i+1, j); collapse the stencil along j
    w_r_lo, w_r_hi = collapsed(
        type_r, c[..., 4] + c[..., 3] + c[..., 5],
        -(c[..., 0] + c[..., 1] + c[..., 2]), -(c[..., 6] + c[..., 7] + c[..., 8]),
        shifted_active(-1, 0), shifted_active(1, 0),
    )
    # between columns (j odd): parents (i, j-1) and (i, j+1); collapse along i
    w_c_lo, w_c_hi = collapsed(
        type_c, c[..., 4] + c[..., 1] + c[..., 7],
        -(c[..., 0] + c[..., 3] + c[..., 6]), -(c[..., 2] + c[..., 5] + c[..., 8]),
        shifted_active(0, -1), shifted_active(0, 1),
    )

    p_idx = np.full((ni, nj, 4), -1, dtype=np.int64)
    p_w = np.zeros((ni, nj, 4), dtype=np.float64)

    # coincident nodes: themselves
    sel = type_cc & act
    p_idx[sel, 0] = coarse_flat(ii[sel], jj[sel])
    p_w[sel, 0] = 1.0

    sel = type_r & act
    p_idx[sel, 0] = coarse_flat(ii[sel] - 1, jj[sel])
    p_w[sel, 0] = w_r_lo[sel]
    p_idx[sel, 1] = coarse_flat(ii[sel] + 1, jj[sel])
    p_w[sel, 1] = w_r_hi[sel]

    sel = type_c & act
    p_idx[sel, 0] = coarse_flat(ii[sel], jj[sel] - 1)
    p_w[sel, 0] = w_c_lo[sel]
    p_idx[sel, 1] = coarse_flat(ii[sel], jj[sel] + 1)
    p_w[sel, 1] = w_c_hi[sel]

    # cell centres (i odd, j odd): the fine equation with the edge neighbours expressed through the corners
    sel = type_x & act
    if sel.any():
        si, sj = ii[sel], jj[sel]
        diag = c[si, sj, 4]
        # edge neighbours: (i-1, j) is a between-columns node with parents (i-1, j-1), (i-1, j+1)
        #                  (i+1, j) likewise with (i+1, j-1), (i+1, j+1)
        #                  (i, j-1) is a between-rows node with parents (i-1, j-1), (i+1, j-1)
        #                  (i, j+1) likewise with (i-1, j+1), (i+1, j+1)
        w00 = c[si, sj, 0] + c[si, sj, 1] * w_c_lo[si - 1, sj] + c[si, sj, 3] * w_r_lo[si, sj - 1]
        w02 = c[si, sj, 2] + c[si, sj, 1] * w_c_hi[si - 1, sj] + c[si, sj, 5] * w_r_lo[si, sj + 1]
        w20 = c[si, sj, 6] + c[si, sj, 7] * w_c_lo[si + 1, sj] + c[si, sj, 3] * w_r_hi[si, sj - 1]
        w22 = c[si, sj, 8] + c[si, sj, 7] * w_c_hi[si + 1, sj] + c[si, sj, 5] * w_r_hi[si, sj + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(diag > 0.0, -1.0 / diag, 0.0)
        for slot, (di, dj, w) in enumerate(((-1, -1, w00), (-1, 1, w02), (1, -1, w20), (1, 1, w22))):
            p_idx[si, sj, slot] = coarse_flat(si + di, sj + dj)
            p_w[si, sj, slot] = w * scale

    p_idx = p_idx.reshape(ni * nj, 4)
    p_w = p_w.reshape(ni * nj, 4)
    # drop parents that are not coarse unknowns (their coefficient is already zero for solids; for
    # Dirichlet parents the correction is zero) and non-finite weights (defensive: never expected)
    parent_active = np.zeros(p_idx.shape, dtype=bool)
    valid = p_idx >= 0
    parent_active[valid] = active_coarse[p_idx[valid]]
    keep = valid & parent_active & np.isfinite(p_w) & (p_w != 0.0)
    p_idx = np.where(keep, p_idx, -1)
    p_w = np.where(keep, p_w, 0.0)
    return ci, cj, active_coarse, p_idx, p_w


def build_restriction(p_idx: np.ndarray, p_w: np.ndarray, coarse_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Transpose table: the (up to nine) fine children of every coarse node with their weights."""

    fine = np.repeat(np.arange(p_idx.shape[0], dtype=np.int64), p_idx.shape[1])
    parent = p_idx.ravel()
    weight = p_w.ravel()
    keep = parent >= 0
    fine, parent, weight = fine[keep], parent[keep], weight[keep]
    order = np.argsort(parent, kind="stable")
    fine, parent, weight = fine[order], parent[order], weight[order]
    starts = np.searchsorted(parent, parent, side="left")
    slot = np.arange(parent.size) - starts
    if slot.size and int(slot.max()) >= 9:
        raise PIC2DValidationError("a coarse node collected more than nine fine children")
    r_idx = np.full((coarse_count, 9), -1, dtype=np.int64)
    r_w = np.zeros((coarse_count, 9), dtype=np.float64)
    r_idx[parent, slot] = fine
    r_w[parent, slot] = weight
    return r_idx, r_w


def galerkin_coarse_operator(
    ni: int, nj: int, coef: np.ndarray, p_idx: np.ndarray, p_w: np.ndarray,
    nci: int, ncj: int, active_coarse: np.ndarray,
) -> np.ndarray:
    """``P^T A P`` assembled into the 9-slot layout of the coarse grid; symmetrised and checked."""

    nbr = neighbour_table(ni, nj)
    nc = nci * ncj
    accum = np.zeros(nc * 9, dtype=np.float64)
    for k in range(9):
        g = nbr[:, k]
        val = coef[:, k]
        pairs = (g >= 0) & (val != 0.0)
        f_sel = np.flatnonzero(pairs)
        if f_sel.size == 0:
            continue
        g_sel = g[f_sel]
        a_fg = val[f_sel]
        for a in range(4):
            ca = p_idx[f_sel, a]
            wa = p_w[f_sel, a]
            if not np.any(ca >= 0):
                continue
            for b in range(4):
                cb = p_idx[g_sel, b]
                wb = p_w[g_sel, b]
                w = wa * a_fg * wb
                valid = (ca >= 0) & (cb >= 0) & (w != 0.0)
                if not valid.any():
                    continue
                ca_v, cb_v, w_v = ca[valid], cb[valid], w[valid]
                d_i = cb_v // ncj - ca_v // ncj
                d_j = cb_v % ncj - ca_v % ncj
                if np.any(np.abs(d_i) > 1) or np.any(np.abs(d_j) > 1):
                    raise PIC2DValidationError("Galerkin coarse operator left the 9-point stencil")
                kc = 3 * (d_i + 1) + (d_j + 1)
                accum += np.bincount(ca_v * 9 + kc, weights=w_v, minlength=nc * 9)
    coarse = accum.reshape(nci, ncj, 9)
    # symmetry check + symmetrisation: coef[c, k] must equal coef[c + off_k, 8 - k]
    scale = float(np.abs(coarse).max()) if coarse.size else 0.0
    for k, (di, dj) in enumerate(STENCIL_OFFSETS):
        if k >= DIAGONAL_SLOT:
            break
        src_i = slice(max(0, -di), nci - max(0, di))
        src_j = slice(max(0, -dj), ncj - max(0, dj))
        dst_i = slice(max(0, di), nci - max(0, -di))
        dst_j = slice(max(0, dj), ncj - max(0, -dj))
        a = coarse[src_i, src_j, k]
        b = coarse[dst_i, dst_j, 8 - k]
        if scale > 0.0 and float(np.abs(a - b).max()) > 1e-11 * scale:
            raise PIC2DValidationError("Galerkin coarse operator is not symmetric to round-off")
        mean = 0.5 * (a + b)
        coarse[src_i, src_j, k] = mean
        coarse[dst_i, dst_j, 8 - k] = mean
    coarse = coarse.reshape(nc, 9)
    coarse[~active_coarse, :] = 0.0
    nbr_c = neighbour_table(nci, ncj)
    for k in range(9):
        if k == DIAGONAL_SLOT:
            continue
        target = nbr_c[:, k]
        inactive_target = (target < 0) | ~active_coarse[np.maximum(target, 0)]
        coarse[inactive_target, k] = 0.0
    diag = coarse[:, DIAGONAL_SLOT]
    if not np.all(diag[active_coarse] > 0.0) or not np.isfinite(coarse).all():
        raise PIC2DValidationError("Galerkin coarse operator has a non-positive or non-finite diagonal")
    return coarse


def build_hierarchy(masks: MeshMasks, *, coarsest_max_unknowns: int = 1024, max_levels: int = 12) -> MGHierarchy:
    """Build the whole hierarchy for ``masks`` (host, numpy; a few hundred ms to seconds)."""

    if coarsest_max_unknowns < 1:
        raise PIC2DValidationError("coarsest_max_unknowns must be positive")
    nr, nz = masks.grid.cell_shape
    ni, nj = nr + 1, nz + 1
    active, coef = finest_stencil(masks)
    if not active.any():
        raise PIC2DValidationError("the Poisson operator has no unknown node")
    fine_i = np.arange(ni, dtype=np.int64)
    fine_j = np.arange(nj, dtype=np.int64)
    levels: list[MGLevel] = []
    while True:
        inv_diag = np.zeros(ni * nj, dtype=np.float64)
        diag = coef[:, DIAGONAL_SLOT]
        if not np.all(diag[active] > 0.0):
            raise PIC2DValidationError("multigrid level has a non-positive diagonal on an unknown")
        inv_diag[active] = 1.0 / diag[active]
        nbr = neighbour_table(ni, nj)
        n_active = int(np.count_nonzero(active))
        can_coarsen = (ni > 3 or nj > 3) and len(levels) + 1 < max_levels and n_active > coarsest_max_unknowns
        if not can_coarsen:
            levels.append(MGLevel(ni, nj, fine_i, fine_j, active, coef, inv_diag, nbr, None, None, None, None))
            break
        ci, cj, active_c, p_idx, p_w = build_prolongation(ni, nj, active, coef)
        nci, ncj = ci.size, cj.size
        r_idx, r_w = build_restriction(p_idx, p_w, nci * ncj)
        coef_c = galerkin_coarse_operator(ni, nj, coef, p_idx, p_w, nci, ncj, active_c)
        levels.append(MGLevel(ni, nj, fine_i, fine_j, active, coef, inv_diag, nbr, p_idx, p_w, r_idx, r_w))
        ni, nj, fine_i, fine_j, active, coef = nci, ncj, ci, cj, active_c, coef_c
    coarsest = levels[-1]
    active_index = np.flatnonzero(coarsest.active)
    na = active_index.size
    position = np.full(coarsest.node_count, -1, dtype=np.int64)
    position[active_index] = np.arange(na)
    matrix = np.zeros((na, na), dtype=np.float64)
    for k in range(9):
        target = coarsest.nbr[active_index, k]
        value = coarsest.coef[active_index, k]
        ok = (target >= 0) & (value != 0.0)
        rows = np.arange(na)[ok]
        cols = position[target[ok]]
        if np.any(cols < 0):
            raise PIC2DValidationError("coarsest operator couples an unknown to an inactive node")
        matrix[rows, cols] += value[ok]
    inverse = np.linalg.inv(matrix)
    if not np.isfinite(inverse).all():
        raise PIC2DConvergenceError("coarsest multigrid operator is singular")
    return MGHierarchy(tuple(levels), active_index, inverse)


# ----------------------------------------------------------------------------- numpy operations

def level_apply(level: MGLevel, x: np.ndarray) -> np.ndarray:
    """``A x`` on this level (zero on inactive nodes; ``x`` is zero on inactive nodes)."""

    x_ext = np.append(x, 0.0)            # index -1 reads the appended zero
    out = np.zeros_like(x)
    for k in range(9):
        out += level.coef[:, k] * x_ext[level.nbr[:, k]]
    out[~level.active] = 0.0
    return out


def jacobi_sweep(level: MGLevel, x: np.ndarray, b: np.ndarray, omega: float) -> np.ndarray:
    """One damped Jacobi sweep ``x + omega D^-1 (b - A x)`` (inactive nodes stay zero)."""

    return x + omega * level.inv_diag * (b - level_apply(level, x))


def restrict_residual(level: MGLevel, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``P^T (b - A x)`` onto the next-coarser level."""

    assert level.r_idx is not None and level.r_w is not None
    r_ext = np.append(b - level_apply(level, x), 0.0)
    out = np.zeros(level.r_idx.shape[0], dtype=np.float64)
    for k in range(9):
        out += level.r_w[:, k] * r_ext[level.r_idx[:, k]]
    return out


def prolong_add(level: MGLevel, x: np.ndarray, x_coarse: np.ndarray) -> np.ndarray:
    """``x + P x_coarse`` on this level."""

    assert level.p_idx is not None and level.p_w is not None
    xc_ext = np.append(x_coarse, 0.0)
    out = x.copy()
    for a in range(4):
        out += level.p_w[:, a] * xc_ext[level.p_idx[:, a]]
    return out


class MultigridPoisson2D:
    """Host (numpy) fixed-cycle V-cycle solver; the reference for ``WarpPoissonMG``.

    ``solve`` mirrors ``BlockTridiagonalSolver.solve`` (same inputs, same contract, same
    ``PoissonResult2D``) and accepts the previous potential as a warm start.
    """

    def __init__(self, masks: MeshMasks, config: PoissonConfig2D = PoissonConfig2D(method="device-mg")) -> None:
        self.masks = masks
        self.config = config
        self.hierarchy = build_hierarchy(masks, coarsest_max_unknowns=config.mg_coarsest_max_unknowns)
        self.cycles = int(config.mg_cycles)
        self.pre = int(config.mg_pre_sweeps)
        self.post = int(config.mg_post_sweeps)
        self.omega = float(config.mg_omega)
        self.unknown_count = int(np.count_nonzero(masks.unknown_node))
        self.host_memory_bytes = int(self.hierarchy.coarsest_inverse.nbytes)

    # -- cycle ---------------------------------------------------------------------------------
    def vcycle(self, x: np.ndarray, b: np.ndarray, depth: int = 0) -> np.ndarray:
        levels = self.hierarchy.levels
        level = levels[depth]
        if depth == len(levels) - 1:
            index = self.hierarchy.coarsest_active_index
            out = np.zeros_like(b)
            out[index] = self.hierarchy.coarsest_inverse @ b[index]
            return out
        for _ in range(self.pre):
            x = jacobi_sweep(level, x, b, self.omega)
        b_c = restrict_residual(level, x, b)
        x_c = self.vcycle(np.zeros_like(b_c), b_c, depth + 1)
        x = prolong_add(level, x, x_c)
        for _ in range(self.post):
            x = jacobi_sweep(level, x, b, self.omega)
        return x

    def right_hand_side(self, node_charge_c: np.ndarray, potentials: BoundaryPotentials) -> np.ndarray:
        masks = self.masks
        if node_charge_c.shape != masks.grid.node_shape or not np.isfinite(node_charge_c).all():
            raise PIC2DValidationError("node charge array has the wrong shape or is nonfinite")
        boundary = boundary_potential_array(masks, potentials)
        rhs = node_charge_c - apply_operator(masks, boundary)
        rhs[~masks.unknown_node] = 0.0
        return rhs

    def run_cycles(
        self, node_charge_c: np.ndarray, potentials: BoundaryPotentials, *,
        initial_phi_v: np.ndarray | None = None, cycles: int | None = None,
    ) -> tuple[np.ndarray, list[float], float]:
        """Run ``cycles`` V-cycles; return ``(phi, per-cycle true residual norms, |rhs|)``.

        The residual history (index 0 = before the first cycle) is recomputed with the
        mesh conductances after every cycle - the measurement behind the fixed cycle count.
        """

        masks = self.masks
        rhs = self.right_hand_side(node_charge_c, potentials)
        unknown = masks.unknown_node
        boundary = boundary_potential_array(masks, potentials)
        x = np.zeros(rhs.size, dtype=np.float64)
        if initial_phi_v is not None:
            if initial_phi_v.shape != rhs.shape or not np.isfinite(initial_phi_v).all():
                raise PIC2DValidationError("initial potential has the wrong shape or is nonfinite")
            x = np.where(unknown, initial_phi_v, 0.0).ravel()
        b = rhs.ravel()

        def true_residual(x_flat: np.ndarray) -> float:
            phi = boundary.copy()
            phi[unknown] = x_flat.reshape(rhs.shape)[unknown]
            residual = node_charge_c - apply_operator(masks, phi)
            residual[~unknown] = 0.0
            return float(np.linalg.norm(residual))

        history = [true_residual(x)]
        for _ in range(self.cycles if cycles is None else int(cycles)):
            x = self.vcycle(x, b)
            history.append(true_residual(x))
        phi = boundary.copy()
        phi[unknown] = x.reshape(rhs.shape)[unknown]
        return phi, history, float(np.linalg.norm(rhs))

    def solve(
        self, node_charge_c: np.ndarray, potentials: BoundaryPotentials, *, initial_phi_v: np.ndarray | None = None,
    ) -> PoissonResult2D:
        phi, history, rhs_norm = self.run_cycles(node_charge_c, potentials, initial_phi_v=initial_phi_v)
        true_residual = history[-1]
        tolerance = max(self.config.absolute_tolerance, self.config.relative_tolerance * rhs_norm)
        converged = isfinite(true_residual) and true_residual <= tolerance and np.isfinite(phi).all()
        diagnostics = PoissonDiagnostics2D(
            bool(converged), self.cycles, history[0], true_residual, true_residual, tolerance, rhs_norm
        )
        if not converged:
            raise PIC2DConvergenceError(
                f"fixed-cycle multigrid solve failed its residual contract: {diagnostics.to_dict()}"
            )
        return PoissonResult2D(phi, diagnostics)


__all__ = [
    "DIAGONAL_SLOT",
    "MGHierarchy",
    "MGLevel",
    "MultigridPoisson2D",
    "STENCIL_OFFSETS",
    "build_hierarchy",
    "build_prolongation",
    "build_restriction",
    "coarse_axis",
    "finest_stencil",
    "galerkin_coarse_operator",
    "jacobi_sweep",
    "level_apply",
    "neighbour_table",
    "prolong_add",
    "restrict_residual",
]
