"""Per-cell Poisson-Boltzmann field step of the L2 v2 hybrid on the PIC's masked (r,z) mesh.

Unknowns: the potential on the PIC mesh's unknown nodes (``cft_revival.pic2d.mesh.MeshMasks``,
the same finite-volume Gauss law and conductances as ``pic2d.poisson``) and one Boltzmann
reference ``ln C_k`` per electron cell.  Electrons in cell ``k`` are a Boltzmann fluid about
their own reference,

    n_e(node) = exp(ln C_k + phi(node) / T_k)            (phi in V, T_k in eV),

constrained to the cell's electron count ``sum_{node in k} n_e V_node = N_k``; the kinetic
ions enter as a deposited node charge; the dielectric wall carries a surface charge that is
advanced IMPLICITLY over the step by the Boltzmann electron flux to the wall
(``dsigma = -e dt (1/4) n_e(wall) v_bar_k A_eff``; the ion contribution is deposited by the
ion push and enters as part of ``surface_charge_c``).  The coupled nonlinear system is solved
by a damped Newton iteration; the Jacobian is the PIC operator plus a positive diagonal, so
the exact block-Thomas factorisation of ``pic2d.poisson`` applies with a shifted diagonal,
and the ``K`` cell constraints are eliminated by a bordered (Schur-complement) solve.

Every publication is fail-closed: the recomputed Gauss residual, the cell constraints, the
total-charge identity (plasma charge + induced electrode charge = 0) and finiteness are
verified on the final iterate; a Newton iteration that does not converge raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import Any

import numpy as np

from ..pic2d.mesh import MeshMasks
from ..pic2d.models import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, EV_J, BoundaryPotentials
from ..pic2d.poisson import apply_operator, boundary_potential_array, induced_electrode_charge_c
from .models import HybridError, HybridValidationError


class HybridConvergenceError(HybridError, RuntimeError):
    """The field step did not meet its residual contract (fail closed)."""


def electron_mean_speed_m_per_s(temperature_ev: np.ndarray | float) -> np.ndarray:
    """``v_bar = sqrt(8 e T / (pi m_e))`` for T in eV."""

    t = np.asarray(temperature_ev, dtype=np.float64)
    return np.sqrt(8.0 * EV_J * t / (pi * ELECTRON_MASS_KG))


class ShiftedBlockSolver:
    """Exact block-Thomas factorisation of ``A_uu + diag(shift)`` on the unknown nodes.

    The algorithm is ``pic2d.poisson.BlockTridiagonalSolver`` with the per-node shift added to
    each column block's diagonal; ``solve`` maps a node array (zero off the unknowns) to the
    solution node array (zero off the unknowns) - no Dirichlet handling, the caller works with
    increments.
    """

    def __init__(self, masks: MeshMasks, shift: np.ndarray) -> None:
        grid = masks.grid
        nz = grid.cell_shape[1]
        unknown = masks.unknown_node
        if shift.shape != grid.node_shape or not np.isfinite(shift).all() or np.any(shift[unknown] < 0.0):
            raise HybridValidationError("Jacobian shift must be a finite non-negative node array")
        columns = [j for j in range(nz + 1) if unknown[:, j].any()]
        if columns != list(range(columns[0], columns[-1] + 1)):
            raise HybridValidationError("unknown columns must be contiguous")
        self.masks = masks
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
            a[np.arange(m), np.arange(m)] = masks.diagonal[rows, j] + shift[rows, j]
            position = {int(i): k for k, i in enumerate(rows)}
            for k, i in enumerate(rows):
                if i + 1 in position:
                    c = masks.cond_r[i, j]
                    a[k, position[i + 1]] -= c
                    a[position[i + 1], k] -= c
            if index > 0 and previous_coupling is not None and previous_g is not None:
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

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        n = len(self.columns)
        u: list[np.ndarray] = [np.empty(0)] * n
        for index, j in enumerate(self.columns):
            y = rhs[self.rows[index], j].copy()
            if index > 0:
                pos_prev, pos_cur, values = self.couplings[index - 1]  # type: ignore[misc]
                y[pos_cur] -= values * u[index - 1][pos_prev]
            u[index] = self.s_inv[index] @ y
        x = np.zeros_like(rhs)
        x_next: np.ndarray | None = None
        for index in range(n - 1, -1, -1):
            value = u[index]
            if x_next is not None:
                value = value - self.g[index] @ x_next  # type: ignore[operator]
            x[self.rows[index], self.columns[index]] = value
            x_next = value
        return x


@dataclass(frozen=True, slots=True)
class FieldStepResult:
    """Published field step: potentials, electron fluid density, wall exchange and diagnostics."""

    phi_v: np.ndarray
    electron_density_per_m3: np.ndarray
    log_boltzmann_reference: np.ndarray            # (K,)
    surface_charge_c: np.ndarray                   # after the implicit electron wall deposit
    wall_electron_flux_per_s: np.ndarray           # node array, electrons/s to the wall (0 off wall nodes)
    anode_induced_charge_c: float
    exit_induced_charge_c: float
    newton_iterations: int
    gauss_residual_c: float
    gauss_source_norm_c: float
    constraint_residual_max: float
    total_charge_identity_c: float
    factorisations: int
    remaining_count: np.ndarray                    # (K,) electrons left in each cell after the implicit losses
    anode_flux_per_s: np.ndarray                   # (K,) Boltzmann electron flux to the anode per cell
    exit_flux_per_s: np.ndarray                    # (K,) ... to the exit plane per cell

    def to_dict(self) -> dict[str, Any]:
        return {
            "newton_iterations": self.newton_iterations,
            "gauss_residual_c": self.gauss_residual_c,
            "gauss_source_norm_c": self.gauss_source_norm_c,
            "gauss_relative_residual": self.gauss_residual_c / max(self.gauss_source_norm_c, np.finfo(float).tiny),
            "constraint_residual_max": self.constraint_residual_max,
            "total_charge_identity_c": self.total_charge_identity_c,
            "factorisations": self.factorisations,
        }


@dataclass(frozen=True, slots=True)
class PBConfig:
    relative_tolerance: float = 1.0e-8
    constraint_tolerance: float = 1.0e-8
    max_iterations: int = 100
    max_step_over_temperature: float = 6.0
    max_reference_step: float = 4.0
    max_backtracks: int = 6
    refactorise_every_iteration: bool = False
    refactorise_ratio: float = 0.2
    publish_relative_residual: float = 1.0e-7

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_tolerance": self.relative_tolerance, "constraint_tolerance": self.constraint_tolerance,
            "max_iterations": self.max_iterations, "max_step_over_temperature": self.max_step_over_temperature,
            "max_reference_step": self.max_reference_step, "max_backtracks": self.max_backtracks,
            "refactorise_every_iteration": self.refactorise_every_iteration, "refactorise_ratio": self.refactorise_ratio,
            "publish_relative_residual": self.publish_relative_residual,
            "method": "damped Newton on phi + ln C_k; block-Thomas of (A + D) with a bordered Schur solve for the cell constraints",
        }


class PoissonBoltzmannSolver:
    """Per-cell Poisson-Boltzmann Newton solver on a ``MeshMasks`` with a fixed cell partition."""

    def __init__(
        self,
        masks: MeshMasks,
        node_cell: np.ndarray,
        *,
        wall_effective_area_m2: np.ndarray,
        electrode_effective_area_m2: np.ndarray | None = None,
        populated_node: np.ndarray | None = None,
        config: PBConfig | None = None,
    ) -> None:
        config = PBConfig() if config is None else config
        grid = masks.grid
        if node_cell.shape != grid.node_shape or node_cell.dtype.kind not in "iu":
            raise HybridValidationError("node_cell must be an integer node array")
        if wall_effective_area_m2.shape != grid.node_shape or not np.isfinite(wall_effective_area_m2).all():
            raise HybridValidationError("wall_effective_area_m2 must be a finite node array")
        if np.any(wall_effective_area_m2 < 0.0) or np.any(wall_effective_area_m2[~masks.unknown_node] != 0.0):
            raise HybridValidationError("wall areas must be non-negative and live on unknown (plasma-side wall) nodes only")
        electrode = np.zeros(grid.node_shape) if electrode_effective_area_m2 is None else np.asarray(electrode_effective_area_m2, dtype=np.float64)
        if electrode.shape != grid.node_shape or not np.isfinite(electrode).all() or np.any(electrode < 0.0) \
                or np.any(electrode[~masks.dirichlet_node] != 0.0):
            raise HybridValidationError("electrode areas must be non-negative and live on Dirichlet nodes only")
        self.masks = masks
        self.config = config
        self.cell_count = int(node_cell[masks.plasma_node].max()) + 1
        if np.any(node_cell[masks.plasma_node] < 0):
            raise HybridValidationError("every plasma node must belong to a cell")
        self.cell_masks_all_plasma = [(np.where(masks.plasma_node, node_cell, -1) == k) & masks.plasma_node for k in range(self.cell_count)]
        self.node_cell = np.where(masks.plasma_node, node_cell, -1)
        # ``plasma`` here is the ELECTRON-populated plasma: nodes whose flux tube is depleted (footprint on the
        # dielectric outside every cusp leak window) carry no Boltzmann electrons and are plain Poisson nodes
        populated = masks.plasma_node if populated_node is None else np.asarray(populated_node, dtype=bool)
        if populated.shape != grid.node_shape or np.any(populated & ~masks.plasma_node):
            raise HybridValidationError("populated_node must be a node mask inside the plasma nodes")
        self.plasma = populated
        self.all_plasma = masks.plasma_node
        self.unknown = masks.unknown_node
        self.volume = np.where(self.plasma, masks.geometric_volume_m3, 0.0)
        self.wall_area = np.where(self.plasma, wall_effective_area_m2, 0.0)
        self.electrode_area = np.where(self.plasma, electrode, 0.0)
        self.cell_masks = [(self.node_cell == k) & self.plasma for k in range(self.cell_count)]
        self.cell_volume_m3 = np.array([float(self.volume[m].sum()) for m in self.cell_masks])
        if np.any(self.cell_volume_m3 <= 0.0):
            raise HybridValidationError("every cell must contain electron-populated plasma volume")

    # -- helpers ----------------------------------------------------------------------------------------------------

    def _node_temperature(self, temperature_ev: np.ndarray) -> np.ndarray:
        t = np.ones(self.masks.grid.node_shape, dtype=np.float64)
        for k, mask in enumerate(self.cell_masks):
            t[mask] = temperature_ev[k]
        return t

    def _density(self, phi: np.ndarray, log_c: np.ndarray, t_node: np.ndarray) -> np.ndarray:
        exponent = np.zeros(self.masks.grid.node_shape, dtype=np.float64)
        for k, mask in enumerate(self.cell_masks):
            exponent[mask] = log_c[k] + phi[mask] / t_node[mask]
        with np.errstate(over="raise"):
            density = np.where(self.plasma, np.exp(np.where(self.plasma, exponent, 0.0)), 0.0)
        return density

    def initial_reference(self, phi: np.ndarray, temperature_ev: np.ndarray, count: np.ndarray,
                          weight: np.ndarray | None = None) -> np.ndarray:
        """``ln C_k`` that satisfies each cell constraint exactly for the given potential (weight = V by default)."""

        t_node = self._node_temperature(temperature_ev)
        weight = self.volume if weight is None else weight
        log_c = np.empty(self.cell_count, dtype=np.float64)
        for k, mask in enumerate(self.cell_masks):
            exponent = phi[mask] / t_node[mask]
            shift = float(exponent.max())
            partition = float(np.sum(weight[mask] * np.exp(exponent - shift)))
            log_c[k] = float(np.log(count[k])) - shift - float(np.log(partition))
        return log_c

    # -- the step ---------------------------------------------------------------------------------------------------

    def solve(
        self,
        *,
        ion_source_c: np.ndarray,
        surface_charge_c: np.ndarray,
        temperature_ev: np.ndarray,
        count: np.ndarray,
        potentials: BoundaryPotentials,
        dt_s: float,
        initial_phi_v: np.ndarray | None = None,
        initial_log_reference: np.ndarray | None = None,
        trace: list[dict[str, float]] | None = None,
    ) -> FieldStepResult:
        masks = self.masks
        shape = masks.grid.node_shape
        temperature = np.asarray(temperature_ev, dtype=np.float64)
        n_k = np.asarray(count, dtype=np.float64)
        if temperature.shape != (self.cell_count,) or n_k.shape != (self.cell_count,):
            raise HybridValidationError("temperature and count must have one entry per cell")
        if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
            raise HybridValidationError("cell temperatures must be finite and positive")
        if not np.isfinite(n_k).all() or np.any(n_k <= 0.0):
            raise HybridValidationError("cell electron counts must be finite and positive")
        for name, array in (("ion_source_c", ion_source_c), ("surface_charge_c", surface_charge_c)):
            if array.shape != shape or not np.isfinite(array).all():
                raise HybridValidationError(f"{name} must be a finite node array")
        if not isfinite(dt_s) or dt_s < 0.0:
            raise HybridValidationError("dt_s must be finite and non-negative")
        t_node = self._node_temperature(temperature)
        v_bar_node = electron_mean_speed_m_per_s(t_node)
        wall_coefficient = dt_s * 0.25 * v_bar_node * self.wall_area          # m^3 per unit density -> electrons lost to the wall
        electrode_coefficient = dt_s * 0.25 * v_bar_node * self.electrode_area  # ... to the electrodes (Dirichlet nodes)
        beta = ELEMENTARY_CHARGE_C * (self.volume + wall_coefficient)         # C per unit density in the Gauss law
        beta[~self.unknown] = 0.0
        # implicit electron losses: the cell constraint counts the electrons still in the cell PLUS those lost over
        # the step to the wall and the electrodes at the new state, against the target count ``n_k``
        weight = self.volume + wall_coefficient + electrode_coefficient
        fixed_source = ion_source_c + surface_charge_c
        fixed_source = np.where(self.unknown, fixed_source, 0.0)

        boundary = boundary_potential_array(masks, potentials)
        phi = boundary.copy()
        if initial_phi_v is not None:
            if initial_phi_v.shape != shape or not np.isfinite(initial_phi_v).all():
                raise HybridValidationError("initial potential has the wrong shape or is nonfinite")
            phi[self.unknown] = initial_phi_v[self.unknown]
        log_c = self.initial_reference(phi, temperature, n_k, weight) if initial_log_reference is None else np.asarray(initial_log_reference, dtype=np.float64).copy()
        if log_c.shape != (self.cell_count,) or not np.isfinite(log_c).all():
            raise HybridValidationError("initial Boltzmann references are malformed")

        cfg = self.config
        factorisations = 0
        solver: ShiftedBlockSolver | None = None
        y_columns: list[np.ndarray] = []
        previous_norm = np.inf
        iterations = 0
        converged = False
        source_norm = float(np.linalg.norm(fixed_source[self.unknown]))
        for iterations in range(1, cfg.max_iterations + 1):
            density = self._density(phi, log_c, t_node)
            electron_term = beta * density
            residual = apply_operator(masks, phi) - fixed_source + electron_term
            residual[~self.unknown] = 0.0
            constraint = np.array([float(np.sum(density[m] * weight[m])) for m in self.cell_masks]) - n_k
            residual_norm = float(np.linalg.norm(residual))
            scale = max(source_norm, float(np.linalg.norm(electron_term[self.unknown])), np.finfo(float).tiny)
            if trace is not None:
                trace.append({"iteration": iterations, "residual": residual_norm, "scale": scale,
                              "constraint": float(np.max(np.abs(constraint) / n_k))})
            if residual_norm <= cfg.relative_tolerance * scale and np.all(np.abs(constraint) <= cfg.constraint_tolerance * n_k):
                converged = True
                break
            refactorise = solver is None or cfg.refactorise_every_iteration or residual_norm > cfg.refactorise_ratio * previous_norm
            if refactorise:
                shift = np.where(self.unknown, electron_term / t_node, 0.0)
                solver = ShiftedBlockSolver(masks, shift)
                factorisations += 1
                y_columns = []
                for mask in self.cell_masks:
                    column = np.where(mask & self.unknown, electron_term, 0.0)
                    y_columns.append(solver.solve(column))
            assert solver is not None
            previous_norm = residual_norm
            w = solver.solve(residual)
            g_rows = [np.where(m & self.unknown, density * weight / t_node, 0.0) for m in self.cell_masks]
            schur = np.empty((self.cell_count, self.cell_count), dtype=np.float64)
            rhs_c = np.empty(self.cell_count, dtype=np.float64)
            for k in range(self.cell_count):
                for l in range(self.cell_count):
                    schur[k, l] = (n_k[k] + constraint[k] if k == l else 0.0) - float(np.sum(g_rows[k] * y_columns[l]))
                rhs_c[k] = -constraint[k] + float(np.sum(g_rows[k] * w))
            try:
                delta_c = np.linalg.solve(schur, rhs_c)
            except np.linalg.LinAlgError as error:
                raise HybridConvergenceError("cell-constraint Schur complement is singular") from error
            delta_u = -w
            for k in range(self.cell_count):
                delta_u = delta_u - y_columns[k] * delta_c[k]
            delta_u[~self.unknown] = 0.0
            if not np.isfinite(delta_u).all() or not np.isfinite(delta_c).all():
                raise HybridConvergenceError("Newton increment is nonfinite")
            step_ratio = float(np.max(np.abs(delta_u[self.unknown]) / t_node[self.unknown])) if self.unknown.any() else 0.0
            damping = min(1.0, cfg.max_step_over_temperature / max(step_ratio, np.finfo(float).tiny),
                          cfg.max_reference_step / max(float(np.max(np.abs(delta_c))), np.finfo(float).tiny))
            # backtracking on the Gauss residual (the exponential can overshoot): halve the step until it decreases
            for _ in range(cfg.max_backtracks + 1):
                trial_phi = phi + damping * delta_u
                trial_c = log_c + damping * delta_c
                try:
                    trial_density = self._density(trial_phi, trial_c, t_node)
                except FloatingPointError:
                    damping *= 0.5
                    continue
                trial_residual = apply_operator(masks, trial_phi) - fixed_source + beta * trial_density
                trial_residual[~self.unknown] = 0.0
                if float(np.linalg.norm(trial_residual)) <= residual_norm or damping < 1e-3:
                    break
                damping *= 0.5
            if trace is not None:
                trace[-1]["damping"] = damping
                trace[-1]["step_over_t"] = step_ratio
            phi = trial_phi
            log_c = trial_c
        if not converged:
            raise HybridConvergenceError(
                f"Poisson-Boltzmann Newton did not converge in {cfg.max_iterations} iterations "
                f"(residual {residual_norm:.3e} vs {cfg.relative_tolerance * scale:.3e}; constraint {np.max(np.abs(constraint) / n_k):.3e})"
            )
        # -- fail-closed publication checks on the final iterate -----------------------------------------------------
        density = self._density(phi, log_c, t_node)
        wall_flux = np.where(self.wall_area > 0.0, 0.25 * v_bar_node * self.wall_area * density, 0.0)
        surface_new = surface_charge_c - ELEMENTARY_CHARGE_C * dt_s * wall_flux
        electron_charge = -ELEMENTARY_CHARGE_C * density * self.volume
        total_source = np.where(self.unknown, ion_source_c + surface_new + electron_charge, 0.0)
        gauss = apply_operator(masks, phi) - total_source
        gauss[~self.unknown] = 0.0
        gauss_residual = float(np.linalg.norm(gauss))
        gauss_scale = float(np.linalg.norm(np.where(self.unknown, np.abs(ion_source_c) + np.abs(surface_new) + np.abs(electron_charge), 0.0)))
        constraint = np.array([float(np.sum(density[m] * weight[m])) for m in self.cell_masks]) - n_k
        remaining = np.array([float(np.sum(density[m] * self.volume[m])) for m in self.cell_masks])
        electrode_flux_node = 0.25 * v_bar_node * self.electrode_area * density
        anode_flux = np.array([float(electrode_flux_node[m & masks.anode_node].sum()) for m in self.cell_masks])
        exit_flux = np.array([float(electrode_flux_node[m & masks.exit_node].sum()) for m in self.cell_masks])
        anode_q, exit_q = induced_electrode_charge_c(masks, phi)
        identity = float(total_source.sum()) + anode_q + exit_q
        if not np.isfinite(phi).all() or not np.isfinite(density).all() or not np.isfinite(surface_new).all():
            raise HybridConvergenceError("field step produced nonfinite values")
        if gauss_residual > cfg.publish_relative_residual * max(gauss_scale, np.finfo(float).tiny):
            raise HybridConvergenceError(f"Gauss-law residual {gauss_residual:.3e} C exceeds {cfg.publish_relative_residual:g} x {gauss_scale:.3e} C")
        if np.any(np.abs(constraint) > cfg.publish_relative_residual * n_k):
            raise HybridConvergenceError("cell electron-count constraint violated on the published iterate")
        identity_scale = float(np.abs(total_source).sum()) + abs(anode_q) + abs(exit_q)
        if abs(identity) > cfg.publish_relative_residual * max(identity_scale, np.finfo(float).tiny):
            raise HybridConvergenceError(f"total charge identity violated: plasma + induced = {identity:.3e} C of {identity_scale:.3e} C")
        if np.any(wall_flux < 0.0) or np.any(remaining <= 0.0):
            raise HybridConvergenceError("negative electron wall flux or an emptied cell")
        return FieldStepResult(
            phi, density, log_c, surface_new, wall_flux, anode_q, exit_q, iterations, gauss_residual, gauss_scale,
            float(np.max(np.abs(constraint) / n_k)), identity, factorisations, remaining, anode_flux, exit_flux,
        )


def wall_effective_areas(
    masks: MeshMasks, b_r_t: np.ndarray, b_z_t: np.ndarray, *, access_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geometric and magnetic-access-weighted wall areas per wall node.

    Every plasma cell face towards a non-plasma cell (or the outer box edge) is a dielectric wall
    face: radial faces (normal r) of area ``2 pi r_{i+1} dz`` split half/half between the two
    axial nodes, axial faces (stair-step risers, normal z) of area ``pi (r_{i+1}^2 - r_i^2)`` split
    between the two radial nodes in proportion to their geometric control volumes.  Electrode
    faces (anode and exit planes) are not walls.  The magnetic access factor of a face is
    ``max(|B . n| / |B|, access_floor)`` at the node: electrons stream along field lines, so a
    wall the field runs parallel to receives only the cross-field floor.  Returns
    ``(area_r, area_z, effective_area, access_factor_r)``.
    """

    grid = masks.grid
    nr, nz = grid.cell_shape
    r = grid.r_m
    dz = grid.dz_m
    plasma_cell = masks.plasma_cell
    area_r = np.zeros(grid.node_shape, dtype=np.float64)
    area_z = np.zeros(grid.node_shape, dtype=np.float64)
    r_mid = 0.5 * (r[:-1] + r[1:])
    inner = pi * (r_mid**2 - r[:-1] ** 2)
    outer = pi * (r[1:] ** 2 - r_mid**2)
    for i in range(nr):
        for j in range(nz):
            if not plasma_cell[i, j]:
                continue
            outside_outer = (i + 1 >= nr) or not plasma_cell[i + 1, j]
            if outside_outer:
                face = 2.0 * pi * r[i + 1] * dz
                area_r[i + 1, j] += 0.5 * face
                area_r[i + 1, j + 1] += 0.5 * face
            if j - 1 >= 0 and not plasma_cell[i, j - 1]:
                area_z[i, j] += inner[i]
                area_z[i + 1, j] += outer[i]
            if j + 1 < nz and not plasma_cell[i, j + 1]:
                area_z[i, j + 1] += inner[i]
                area_z[i + 1, j + 1] += outer[i]
    # wall nodes: plasma-side nodes carrying a dielectric face (this includes a straight bore on the outer box edge,
    # which the PIC mask does not flag as wall_node because no non-plasma cell exists beyond it)
    wall = masks.unknown_node & ((area_r + area_z) > 0.0)
    area_r = np.where(wall, area_r, 0.0)
    area_z = np.where(wall, area_z, 0.0)
    magnitude = np.hypot(b_r_t, b_z_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        access_r = np.where(magnitude > 0.0, np.abs(b_r_t) / magnitude, 1.0)
        access_z = np.where(magnitude > 0.0, np.abs(b_z_t) / magnitude, 1.0)
    access_r = np.maximum(access_r, access_floor)
    access_z = np.maximum(access_z, access_floor)
    effective = area_r * access_r + area_z * access_z
    return area_r, area_z, np.where(wall, effective, 0.0), np.where(wall, access_r, 0.0)


def electrode_face_areas(masks: MeshMasks) -> tuple[np.ndarray, np.ndarray]:
    """Plasma-side face areas of the anode-plane and exit-plane nodes (control-volume split)."""

    grid = masks.grid
    nr, nz = grid.cell_shape
    r = grid.r_m
    r_mid = 0.5 * (r[:-1] + r[1:])
    inner = pi * (r_mid**2 - r[:-1] ** 2)
    outer = pi * (r[1:] ** 2 - r_mid**2)
    anode = np.zeros(grid.node_shape, dtype=np.float64)
    exit_plane = np.zeros(grid.node_shape, dtype=np.float64)
    for i in range(nr):
        if masks.plasma_cell[i, 0]:
            anode[i, 0] += inner[i]
            anode[i + 1, 0] += outer[i]
        if masks.plasma_cell[i, nz - 1]:
            exit_plane[i, nz] += inner[i]
            exit_plane[i + 1, nz] += outer[i]
    return np.where(masks.anode_node, anode, 0.0), np.where(masks.exit_node, exit_plane, 0.0)


def thermal_speed_check(temperature_ev: float) -> float:
    if not isfinite(temperature_ev) or temperature_ev <= 0.0:
        raise HybridValidationError("temperature must be positive")
    return sqrt(8.0 * EV_J * temperature_ev / (pi * ELECTRON_MASS_KG))


__all__ = [
    "FieldStepResult",
    "HybridConvergenceError",
    "PBConfig",
    "PoissonBoltzmannSolver",
    "ShiftedBlockSolver",
    "electrode_face_areas",
    "electron_mean_speed_m_per_s",
    "thermal_speed_check",
    "wall_effective_areas",
]
