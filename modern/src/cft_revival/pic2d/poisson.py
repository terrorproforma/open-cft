"""Cylindrical finite-volume Poisson solver on the masked (r,z) node mesh.

Operator: for every plasma cell the four edges carry conductances
``C = epsilon_0 * face_area / edge_length`` (see ``mesh.py``).  The discrete
Gauss law at a node is ``sum_e C_e (phi_n - phi_m) = Q_n`` where ``Q_n`` is the
node charge in coulombs (volume-deposited plus accumulated wall surface
charge).  Dividing by the node control volume recovers the standard
second-order cylindrical stencil including the regular axis form
``4 (phi_1 - phi_0) / dr^2``.  Faces towards non-plasma cells carry no flux,
i.e. the dielectric backing is treated as a perfect insulator with zero field
(homogeneous Neumann plus the deposited surface charge).

The matrix restricted to unknown nodes is symmetric positive definite, so a
Jacobi-preconditioned conjugate-gradient iteration converges; the published
result requires an independently recomputed true residual within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from .mesh import MeshMasks
from .models import BoundaryPotentials, PIC2DConvergenceError, PIC2DValidationError, PoissonConfig2D


def apply_operator(masks: MeshMasks, phi: np.ndarray) -> np.ndarray:
    """Return ``A phi`` on every node (zero on nodes outside the plasma)."""

    out = np.zeros_like(phi)
    radial = masks.cond_r * (phi[:-1, :] - phi[1:, :])
    out[:-1, :] += radial
    out[1:, :] -= radial
    axial = masks.cond_z * (phi[:, :-1] - phi[:, 1:])
    out[:, :-1] += axial
    out[:, 1:] -= axial
    return out


def field_energy_j(masks: MeshMasks, phi: np.ndarray) -> float:
    """Exact discrete electrostatic energy ``0.5 * phi^T A phi`` in joules."""

    radial = masks.cond_r * (phi[:-1, :] - phi[1:, :]) ** 2
    axial = masks.cond_z * (phi[:, :-1] - phi[:, 1:]) ** 2
    return 0.5 * float(radial.sum() + axial.sum())


def boundary_potential_array(masks: MeshMasks, potentials: BoundaryPotentials) -> np.ndarray:
    phi = np.zeros(masks.grid.node_shape, dtype=np.float64)
    phi[masks.anode_node] = potentials.anode_v
    phi[masks.exit_node] = potentials.exit_v
    return phi


def induced_electrode_charge_c(masks: MeshMasks, phi: np.ndarray) -> tuple[float, float]:
    """Charge induced on the anode and exit electrodes by the discrete field.

    ``(A phi)`` evaluated on a Dirichlet node is the net conductance flux into
    that node, i.e. the charge the electrode must carry.  Together with the
    plasma node charges this satisfies the discrete Gauss law exactly.
    """

    flux = apply_operator(masks, phi)
    return float(flux[masks.anode_node].sum()), float(flux[masks.exit_node].sum())


@dataclass(frozen=True, slots=True)
class PoissonDiagnostics2D:
    converged: bool
    iterations: int
    initial_residual_l2: float
    final_residual_l2: float
    true_residual_l2: float
    tolerance: float
    rhs_l2: float

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_residual_l2": self.initial_residual_l2,
            "final_residual_l2": self.final_residual_l2,
            "true_residual_l2": self.true_residual_l2,
            "tolerance": self.tolerance,
            "rhs_l2": self.rhs_l2,
        }


@dataclass(frozen=True, slots=True)
class PoissonResult2D:
    phi_v: np.ndarray
    diagnostics: PoissonDiagnostics2D


class Poisson2D:
    """CPU reference solver (vectorised numpy PCG) for the masked FV operator."""

    def __init__(self, masks: MeshMasks, config: PoissonConfig2D = PoissonConfig2D()) -> None:
        self.masks = masks
        self.config = config
        self.unknown = masks.unknown_node
        diagonal = masks.diagonal[self.unknown]
        if not np.all(diagonal > 0.0) or not np.isfinite(diagonal).all():
            raise PIC2DValidationError("Poisson diagonal must be finite and positive on unknowns")
        self.inverse_diagonal = np.zeros_like(masks.diagonal)
        self.inverse_diagonal[self.unknown] = 1.0 / diagonal
        self.direct: BlockTridiagonalSolver | None = None
        if config.method in ("direct", "device-direct"):
            self.direct = BlockTridiagonalSolver(masks, relative_tolerance=config.relative_tolerance)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        # x is a full node array that is zero off the unknown set.
        out = apply_operator(self.masks, x)
        out[~self.unknown] = 0.0
        return out

    def right_hand_side(self, node_charge_c: np.ndarray, potentials: BoundaryPotentials) -> np.ndarray:
        if node_charge_c.shape != self.masks.grid.node_shape:
            raise PIC2DValidationError("node charge array has the wrong shape")
        if not np.isfinite(node_charge_c).all():
            raise PIC2DValidationError("node charge must be finite")
        boundary = boundary_potential_array(self.masks, potentials)
        rhs = node_charge_c - apply_operator(self.masks, boundary)
        rhs[~self.unknown] = 0.0
        return rhs

    def solve(
        self,
        node_charge_c: np.ndarray,
        potentials: BoundaryPotentials,
        *,
        initial_phi_v: np.ndarray | None = None,
    ) -> PoissonResult2D:
        if self.direct is not None:
            return self.direct.solve(node_charge_c, potentials)
        rhs = self.right_hand_side(node_charge_c, potentials)
        rhs_norm = float(np.linalg.norm(rhs))
        tolerance = max(self.config.absolute_tolerance, self.config.relative_tolerance * rhs_norm)
        if not isfinite(tolerance):
            raise PIC2DValidationError("Poisson tolerance is not finite")
        x = np.zeros_like(rhs)
        if initial_phi_v is not None:
            if initial_phi_v.shape != rhs.shape or not np.isfinite(initial_phi_v).all():
                raise PIC2DValidationError("initial potential has the wrong shape or is nonfinite")
            x[self.unknown] = initial_phi_v[self.unknown]
        residual = rhs - self._matvec(x)
        initial_norm = float(np.linalg.norm(residual))
        iterations = 0
        final_norm = initial_norm
        true_residual = initial_norm
        restarts = 0
        # The recurrence residual can drift from the true residual near the
        # binary64 floor; a restart recomputes it and continues from the same
        # iterate.  Publication always uses the recomputed true residual.
        while true_residual > tolerance and iterations < self.config.max_iterations and restarts <= 3:
            z = self.inverse_diagonal * residual
            p = z.copy()
            rho = float(np.vdot(residual, z))
            while iterations < self.config.max_iterations:
                iterations += 1
                q = self._matvec(p)
                denominator = float(np.vdot(p, q))
                if not isfinite(denominator) or denominator <= 0.0:
                    raise PIC2DConvergenceError("conjugate gradient lost positive definiteness")
                alpha = rho / denominator
                x += alpha * p
                residual -= alpha * q
                final_norm = float(np.linalg.norm(residual))
                if not isfinite(final_norm):
                    raise PIC2DConvergenceError("conjugate gradient residual became nonfinite")
                if final_norm <= tolerance:
                    break
                z = self.inverse_diagonal * residual
                rho_new = float(np.vdot(residual, z))
                p = z + (rho_new / rho) * p
                rho = rho_new
            residual = rhs - self._matvec(x)
            true_residual = float(np.linalg.norm(residual))
            restarts += 1
        converged = isfinite(true_residual) and true_residual <= tolerance
        diagnostics = PoissonDiagnostics2D(
            bool(converged), iterations, initial_norm, final_norm, true_residual, tolerance, rhs_norm
        )
        if not converged:
            raise PIC2DConvergenceError(
                f"Poisson solve did not meet its residual contract: {diagnostics.to_dict()}"
            )
        phi = boundary_potential_array(self.masks, potentials)
        phi[self.unknown] = x[self.unknown]
        if not np.isfinite(phi).all():
            raise PIC2DConvergenceError("Poisson potential is nonfinite")
        return PoissonResult2D(phi, diagnostics)


class BlockTridiagonalSolver:
    """Exact block-Thomas (block Cholesky) direct solver for the masked operator.

    Unknowns are grouped by axial column ``j``; each column block ``A_j`` is the
    radial tridiagonal operator and the coupling ``B_j`` to column ``j+1`` is
    diagonal (one ``cond_z`` edge per shared radial index).  The Schur
    complements ``S_j = A_j - B_{j-1}^T S_{j-1}^{-1} B_{j-1}`` are inverted once;
    each solve then costs two small dense matvecs per column, independent of
    the right-hand side.  The result is verified against a recomputed true
    residual like the iterative path.  Memory is ``2 * sum_j m_j^2`` doubles.
    """

    def __init__(self, masks: MeshMasks, *, relative_tolerance: float = 1.0e-10) -> None:
        self.masks = masks
        self.relative_tolerance = float(relative_tolerance)
        grid = masks.grid
        nr, nz = grid.cell_shape
        unknown = masks.unknown_node
        columns = [j for j in range(nz + 1) if unknown[:, j].any()]
        if columns != list(range(columns[0], columns[-1] + 1)):
            raise PIC2DValidationError("unknown columns must be contiguous")
        self.columns = columns
        self.rows: list[np.ndarray] = [np.flatnonzero(unknown[:, j]) for j in columns]
        self.s_inv: list[np.ndarray] = []
        self.g: list[np.ndarray | None] = []
        self.couplings: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
        previous_g = None
        previous_coupling = None
        for index, j in enumerate(columns):
            rows = self.rows[index]
            m = rows.size
            a = np.zeros((m, m), dtype=np.float64)
            a[np.arange(m), np.arange(m)] = masks.diagonal[rows, j]
            position = {int(i): k for k, i in enumerate(rows)}
            for k, i in enumerate(rows):
                if i + 1 in position:
                    c = masks.cond_r[i, j]
                    a[k, position[i + 1]] -= c
                    a[position[i + 1], k] -= c
            if index > 0 and previous_coupling is not None and previous_g is not None:
                # S_j = A_j - B^T G  where G = S_{j-1}^{-1} B (dense m_{j-1} x m_j)
                pos_prev, pos_cur, values = previous_coupling
                bt_g = np.zeros((m, m), dtype=np.float64)
                bt_g[pos_cur, :] = values[:, None] * previous_g[pos_prev, :]
                a = a - bt_g
            s_inv = np.linalg.inv(a)
            self.s_inv.append(s_inv)
            if index < len(columns) - 1:
                next_rows = self.rows[index + 1]
                next_position = {int(i): k for k, i in enumerate(next_rows)}
                shared = [(k, next_position[int(i)]) for k, i in enumerate(rows) if int(i) in next_position]
                pos_cur = np.array([k for k, _ in shared], dtype=np.int64)
                pos_next = np.array([k for _, k in shared], dtype=np.int64)
                values = -masks.cond_z[rows[pos_cur], j]
                coupling = (pos_cur, pos_next, values)
                g = np.zeros((m, next_rows.size), dtype=np.float64)
                g[:, pos_next] = s_inv[:, pos_cur] * values[None, :]
                self.couplings.append(coupling)
                self.g.append(g)
                previous_coupling = coupling
                previous_g = g
            else:
                self.couplings.append(None)
                self.g.append(None)
        self.unknown_count = int(sum(rows.size for rows in self.rows))

    def solve(
        self,
        node_charge_c: np.ndarray,
        potentials: BoundaryPotentials,
    ) -> PoissonResult2D:
        masks = self.masks
        if node_charge_c.shape != masks.grid.node_shape or not np.isfinite(node_charge_c).all():
            raise PIC2DValidationError("node charge array has the wrong shape or is nonfinite")
        boundary = boundary_potential_array(masks, potentials)
        rhs = node_charge_c - apply_operator(masks, boundary)
        rhs[~masks.unknown_node] = 0.0
        n = len(self.columns)
        u: list[np.ndarray] = [np.empty(0)] * n
        y_prev: np.ndarray | None = None
        for index, j in enumerate(self.columns):
            y = rhs[self.rows[index], j].copy()
            if index > 0:
                pos_prev, pos_cur, values = self.couplings[index - 1]  # type: ignore[misc]
                y[pos_cur] -= values * u[index - 1][pos_prev]
            u[index] = self.s_inv[index] @ y
        phi = boundary.copy()
        x_next: np.ndarray | None = None
        for index in range(n - 1, -1, -1):
            x = u[index]
            if x_next is not None:
                x = x - self.g[index] @ x_next  # type: ignore[operator]
            phi[self.rows[index], self.columns[index]] = x
            x_next = x
        # Full-equation residual on unknowns: Q_n - (A phi)_n with the Dirichlet
        # values already inside phi (equivalent to rhs - A_uu x).
        residual = node_charge_c - apply_operator(masks, phi)
        residual[~masks.unknown_node] = 0.0
        true_residual = float(np.linalg.norm(residual))
        rhs_norm = float(np.linalg.norm(rhs))
        tolerance = self.relative_tolerance * rhs_norm
        converged = isfinite(true_residual) and true_residual <= tolerance and np.isfinite(phi).all()
        diagnostics = PoissonDiagnostics2D(bool(converged), 1, rhs_norm, true_residual, true_residual, tolerance, rhs_norm)
        if not converged:
            raise PIC2DConvergenceError(f"direct Poisson solve failed its residual contract: {diagnostics.to_dict()}")
        return PoissonResult2D(phi, diagnostics)


def dense_reference_solve(
    masks: MeshMasks, node_charge_c: np.ndarray, potentials: BoundaryPotentials
) -> np.ndarray:
    """Assemble the unknown-block matrix explicitly and solve it directly.

    Intended for small verification grids only; it is the independent oracle
    for the iterative solver.
    """

    unknown = masks.unknown_node
    count = int(np.count_nonzero(unknown))
    if count > 6000:
        raise PIC2DValidationError("dense reference solve is limited to <= 6000 unknowns")
    index = -np.ones(masks.grid.node_shape, dtype=np.int64)
    index[unknown] = np.arange(count)
    matrix = np.zeros((count, count), dtype=np.float64)
    nr, nz = masks.grid.cell_shape
    for i in range(nr):
        for j in range(nz + 1):
            c = masks.cond_r[i, j]
            if c == 0.0:
                continue
            a, b = index[i, j], index[i + 1, j]
            if a >= 0:
                matrix[a, a] += c
            if b >= 0:
                matrix[b, b] += c
            if a >= 0 and b >= 0:
                matrix[a, b] -= c
                matrix[b, a] -= c
    for i in range(nr + 1):
        for j in range(nz):
            c = masks.cond_z[i, j]
            if c == 0.0:
                continue
            a, b = index[i, j], index[i, j + 1]
            if a >= 0:
                matrix[a, a] += c
            if b >= 0:
                matrix[b, b] += c
            if a >= 0 and b >= 0:
                matrix[a, b] -= c
                matrix[b, a] -= c
    boundary = boundary_potential_array(masks, potentials)
    rhs = (node_charge_c - apply_operator(masks, boundary))[unknown]
    solution = np.linalg.solve(matrix, rhs)
    phi = boundary.copy()
    phi[unknown] = solution
    return phi


def electric_field_nodes(masks: MeshMasks, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nodal ``E = -grad(phi)`` on plasma nodes; zero elsewhere.

    Central differences where both neighbours are plasma nodes, second-order
    one-sided differences at electrodes and walls (first order when only one
    interior neighbour exists), and ``E_r = 0`` on the axis by symmetry.
    """

    grid = masks.grid
    nr, nz = grid.cell_shape
    plasma = masks.plasma_node
    dr, dz = grid.dr_m, grid.dz_m
    e_r = np.zeros_like(phi)
    e_z = np.zeros_like(phi)

    def shifted(mask: np.ndarray, di: int, dj: int) -> np.ndarray:
        out = np.zeros_like(mask)
        src_i = slice(max(0, di), nr + 1 + min(0, di))
        dst_i = slice(max(0, -di), nr + 1 + min(0, -di))
        src_j = slice(max(0, dj), nz + 1 + min(0, dj))
        dst_j = slice(max(0, -dj), nz + 1 + min(0, -dj))
        out[dst_i, dst_j] = mask[src_i, src_j]
        return out

    def shifted_values(values: np.ndarray, di: int, dj: int) -> np.ndarray:
        out = np.zeros_like(values)
        src_i = slice(max(0, di), nr + 1 + min(0, di))
        dst_i = slice(max(0, -di), nr + 1 + min(0, -di))
        src_j = slice(max(0, dj), nz + 1 + min(0, dj))
        dst_j = slice(max(0, -dj), nz + 1 + min(0, -dj))
        out[dst_i, dst_j] = values[src_i, src_j]
        return out

    for axis, spacing, target in ((0, dr, e_r), (1, dz, e_z)):
        step = (1, 0) if axis == 0 else (0, 1)
        plus1 = shifted(plasma, *step)
        minus1 = shifted(plasma, -step[0], -step[1])
        plus2 = shifted(plasma, 2 * step[0], 2 * step[1])
        minus2 = shifted(plasma, -2 * step[0], -2 * step[1])
        phi_p1 = shifted_values(phi, *step)
        phi_m1 = shifted_values(phi, -step[0], -step[1])
        phi_p2 = shifted_values(phi, 2 * step[0], 2 * step[1])
        phi_m2 = shifted_values(phi, -2 * step[0], -2 * step[1])
        central = plasma & plus1 & minus1
        forward2 = plasma & plus1 & ~minus1 & plus2
        forward1 = plasma & plus1 & ~minus1 & ~plus2
        backward2 = plasma & minus1 & ~plus1 & minus2
        backward1 = plasma & minus1 & ~plus1 & ~minus2
        target[central] = -(phi_p1[central] - phi_m1[central]) / (2.0 * spacing)
        target[forward2] = -(-3.0 * phi[forward2] + 4.0 * phi_p1[forward2] - phi_p2[forward2]) / (2.0 * spacing)
        target[forward1] = -(phi_p1[forward1] - phi[forward1]) / spacing
        target[backward2] = -(3.0 * phi[backward2] - 4.0 * phi_m1[backward2] + phi_m2[backward2]) / (2.0 * spacing)
        target[backward1] = -(phi[backward1] - phi_m1[backward1]) / spacing
    e_r[masks.axis_node] = 0.0
    return e_r, e_z


__all__ = [
    "BlockTridiagonalSolver",
    "Poisson2D",
    "PoissonDiagnostics2D",
    "PoissonResult2D",
    "apply_operator",
    "boundary_potential_array",
    "dense_reference_solve",
    "electric_field_nodes",
    "field_energy_j",
    "induced_electrode_charge_c",
]
