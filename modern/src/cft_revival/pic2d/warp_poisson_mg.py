"""Warp-native geometric multigrid Poisson solve (``poisson.method = "device-mg"``).

Device counterpart of ``poisson_mg.MultigridPoisson2D``: the same hierarchy (built once
on the host by ``poisson_mg.build_hierarchy``), the same fixed V(nu1, nu2) cycles of
damped Jacobi, the same operator-dependent transfers and dense coarsest solve, as a
fixed sequence of per-node kernels with no host synchronisation, so the whole solve is
captured inside the step graph exactly like ``WarpBlockThomas.solve_sequence``.

Interface: a drop-in for ``WarpBlockThomas`` (``bind`` / ``solve_sequence`` / ``solve`` /
``queue_residual_check`` / ``verify`` / ``bound_inputs`` / ``host_memory_bytes``), so the
backend selects it with one branch and everything downstream (graph capture, the
residual check at every host sync, checkpoint binding) is unchanged.

Contract enforcement.  A fixed cycle count cannot iterate to convergence, so every solve
recomputes the *true* residual with the mesh conductances (``matvec_kernel`` of the
backend, independent of the multigrid's own stencil arrays) inside the captured
sequence and keeps a running maximum of the contract ratio
``|r|^2 / max(abs_tol^2, rel_tol^2 |rhs|^2)`` over the sync interval; ``verify`` (called
by the backend at every host sync) reads the last residual and that maximum and raises
``PIC2DConvergenceError`` if any step of the interval missed the contract - the run
stops fail-closed, the same behaviour as the direct solve.  There is deliberately no
per-step fall-back to the block-Thomas solve: it would need a host synchronisation per
step (the block-Thomas graph is not even resident when the multigrid is selected), and
a missed contract is a configuration error (too few cycles for the source) to fix by
re-running with a larger ``mg_cycles`` from the last checkpoint, not a condition to
paper over silently.

Stencil storage per level (float64, flat over the ``ni x nj`` node grid; symmetric
9-point operator, four "forward" couplings per node + the diagonal):
``a_e[n]`` couples ``(i, j) - (i, j+1)``, ``a_n[n]`` couples ``(i, j) - (i+1, j)``,
``a_ne[n]`` couples ``(i, j) - (i+1, j+1)``, ``a_nw[n]`` couples ``(i, j) - (i+1, j-1)``;
the backward couplings are read from the neighbour's forward entries.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Any

import numpy as np

from .mesh import MeshMasks
from .models import PIC2DConvergenceError, PIC2DDeviceError
from .poisson import apply_operator, boundary_potential_array
from .poisson_mg import DIAGONAL_SLOT, MGHierarchy, MGLevel, build_hierarchy
from .warp_backend import (
    REDUCTION_GROUP,
    REDUCTION_THREADS,
    apply_dirichlet_kernel,
    dot_stride_kernel,
    final_sum_kernel,
    matvec_kernel,
    reduce_stage_kernel,
    residual_kernel,
    source_kernel,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dependency
    wp = None  # type: ignore[assignment]

# scalar slots (shared layout with the block-Thomas solver: 5 = |r|^2, 6 = |rhs|^2)
SLOT_RR = 5
SLOT_RHS2 = 6
SLOT_WORST = 7      # running maximum of the contract ratio over the sync interval

if wp is not None:
    F64 = wp.float64

    @wp.func
    def mg_apply_at(
        n: int, i: int, j: int, ni: int, nj: int,
        x: wp.array(dtype=F64), diag: wp.array(dtype=F64),
        a_e: wp.array(dtype=F64), a_n: wp.array(dtype=F64), a_ne: wp.array(dtype=F64), a_nw: wp.array(dtype=F64),
    ) -> F64:
        """``(A x)[n]`` for the symmetric compact 9-point stencil (couplings to missing nodes are zero)."""

        acc = diag[n] * x[n]
        if j + 1 < nj:
            acc += a_e[n] * x[n + 1]
        if j > 0:
            acc += a_e[n - 1] * x[n - 1]
        if i + 1 < ni:
            acc += a_n[n] * x[n + nj]
            if j + 1 < nj:
                acc += a_ne[n] * x[n + nj + 1]
            if j > 0:
                acc += a_nw[n] * x[n + nj - 1]
        if i > 0:
            acc += a_n[n - nj] * x[n - nj]
            if j > 0:
                acc += a_ne[n - nj - 1] * x[n - nj - 1]
            if j + 1 < nj:
                acc += a_nw[n - nj + 1] * x[n - nj + 1]
        return acc

    @wp.kernel
    def mg_init_kernel(phi: wp.array(dtype=F64), active: wp.array(dtype=wp.int32), x: wp.array(dtype=F64)):
        # warm start: the previous potential on the unknowns, zero elsewhere
        n = wp.tid()
        if active[n] != 0:
            x[n] = phi[n]
        else:
            x[n] = F64(0.0)

    @wp.kernel
    def mg_jacobi_kernel(
        x_in: wp.array(dtype=F64), b: wp.array(dtype=F64), active: wp.array(dtype=wp.int32), ni: int, nj: int,
        diag: wp.array(dtype=F64), inv_diag: wp.array(dtype=F64),
        a_e: wp.array(dtype=F64), a_n: wp.array(dtype=F64), a_ne: wp.array(dtype=F64), a_nw: wp.array(dtype=F64),
        omega: F64, x_out: wp.array(dtype=F64),
    ):
        n = wp.tid()
        if active[n] == 0:
            x_out[n] = F64(0.0)
            return
        i = n / nj
        j = n - i * nj
        ax = mg_apply_at(n, i, j, ni, nj, x_in, diag, a_e, a_n, a_ne, a_nw)
        x_out[n] = x_in[n] + omega * inv_diag[n] * (b[n] - ax)

    @wp.kernel
    def mg_restrict_kernel(
        x: wp.array(dtype=F64), b: wp.array(dtype=F64), active: wp.array(dtype=wp.int32), ni: int, nj: int,
        diag: wp.array(dtype=F64),
        a_e: wp.array(dtype=F64), a_n: wp.array(dtype=F64), a_ne: wp.array(dtype=F64), a_nw: wp.array(dtype=F64),
        r_idx: wp.array(dtype=wp.int32), r_w: wp.array(dtype=F64),
        b_coarse: wp.array(dtype=F64), x_coarse: wp.array(dtype=F64),
    ):
        # b_c = P^T (b - A x) gathered over the (up to nine) fine children; x_c = 0 (zero coarse initial guess)
        c = wp.tid()
        acc = F64(0.0)
        for k in range(9):
            f = r_idx[c * 9 + k]
            if f >= 0 and active[f] != 0:
                i = f / nj
                j = f - i * nj
                acc += r_w[c * 9 + k] * (b[f] - mg_apply_at(f, i, j, ni, nj, x, diag, a_e, a_n, a_ne, a_nw))
        b_coarse[c] = acc
        x_coarse[c] = F64(0.0)

    @wp.kernel
    def mg_prolong_kernel(
        x_coarse: wp.array(dtype=F64), p_idx: wp.array(dtype=wp.int32), p_w: wp.array(dtype=F64), x: wp.array(dtype=F64),
    ):
        # x += P x_c (inactive fine nodes have no parents and stay zero)
        n = wp.tid()
        acc = F64(0.0)
        for a in range(4):
            c = p_idx[n * 4 + a]
            if c >= 0:
                acc += p_w[n * 4 + a] * x_coarse[c]
        x[n] = x[n] + acc

    @wp.kernel
    def mg_dense_kernel(
        g: wp.array(dtype=F64), gather: wp.array(dtype=wp.int32), na: int, b: wp.array(dtype=F64), x: wp.array(dtype=F64),
    ):
        # coarsest level: x[active] = G b[active] with the dense inverse G (fixed summation order)
        a = wp.tid()
        acc = F64(0.0)
        base = a * na
        for k in range(na):
            acc += g[base + k] * b[gather[k]]
        x[gather[a]] = acc

    @wp.kernel
    def mg_copy_kernel(src: wp.array(dtype=F64), dst: wp.array(dtype=F64)):
        n = wp.tid()
        dst[n] = src[n]

    @wp.kernel
    def mg_track_kernel(scalars: wp.array(dtype=F64), abs2: F64, rel2: F64):
        # running maximum over the sync interval of |r|^2 / max(abs_tol^2, rel_tol^2 |rhs|^2)
        bound = wp.max(abs2, rel2 * scalars[SLOT_RHS2])
        rr = scalars[SLOT_RR]
        ratio = F64(0.0)
        if bound > F64(0.0):
            ratio = rr / bound
        elif rr > F64(0.0):
            ratio = F64(1.0e300)
        if ratio > scalars[SLOT_WORST]:
            scalars[SLOT_WORST] = ratio

    @wp.kernel
    def mg_reset_kernel(scalars: wp.array(dtype=F64)):
        scalars[SLOT_WORST] = F64(0.0)


class _DeviceLevel:
    """Device arrays of one hierarchy level."""

    __slots__ = ("ni", "nj", "count", "active", "diag", "inv_diag", "a_e", "a_n", "a_ne", "a_nw",
                 "x", "t", "b", "p_idx", "p_w", "r_idx", "r_w", "coarse_count")

    def __init__(self, level: MGLevel, device, x: Any = None, b: Any = None) -> None:
        f64 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), dtype=wp.float64, device=device)  # noqa: E731
        i32 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.int32).ravel(), dtype=wp.int32, device=device)  # noqa: E731
        self.ni, self.nj = level.ni, level.nj
        self.count = level.node_count
        coef = level.coef
        self.active = i32(level.active)
        self.diag = f64(coef[:, DIAGONAL_SLOT])
        self.inv_diag = f64(level.inv_diag)
        self.a_e = f64(coef[:, 5])
        self.a_n = f64(coef[:, 7])
        self.a_ne = f64(coef[:, 8])
        self.a_nw = f64(coef[:, 6])
        self.x = x if x is not None else wp.zeros(self.count, dtype=wp.float64, device=device)
        self.b = b if b is not None else wp.zeros(self.count, dtype=wp.float64, device=device)
        self.t = wp.zeros(self.count, dtype=wp.float64, device=device)
        if level.p_idx is not None:
            self.p_idx = i32(level.p_idx)
            self.p_w = f64(level.p_w)
            self.r_idx = i32(level.r_idx)
            self.r_w = f64(level.r_w)
            self.coarse_count = int(level.r_idx.shape[0])
        else:
            self.p_idx = self.p_w = self.r_idx = self.r_w = None
            self.coarse_count = 0

    @property
    def nbytes(self) -> int:
        total = 0
        for name in self.__slots__:
            value = getattr(self, name, None)
            if value is not None and hasattr(value, "capacity"):
                total += int(value.capacity)
        return total


class WarpPoissonMG:
    """Fixed-cycle geometric multigrid on the device; interface of ``WarpBlockThomas``."""

    def __init__(self, masks: MeshMasks, potentials, config, device, *, use_graph: bool = True) -> None:
        if wp is None:
            raise PIC2DDeviceError("NVIDIA Warp is unavailable")
        self.masks = masks
        self.config = config
        self.device = device
        grid = masks.grid
        nr, nz = grid.cell_shape
        self.nr, self.nz = nr, nz
        self.node_count = int(np.prod(grid.node_shape))
        self.cycles = int(config.mg_cycles)
        self.pre = int(config.mg_pre_sweeps)
        self.post = int(config.mg_post_sweeps)
        self.omega = float(config.mg_omega)
        self.hierarchy: MGHierarchy = build_hierarchy(masks, coarsest_max_unknowns=config.mg_coarsest_max_unknowns)
        dev = device
        f64 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), dtype=wp.float64, device=dev)  # noqa: E731
        i32 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.int32).ravel(), dtype=wp.int32, device=dev)  # noqa: E731
        zeros = lambda: wp.zeros(self.node_count, dtype=wp.float64, device=dev)  # noqa: E731
        # level 0 = the node mesh: the solve's unknown vector and right-hand side
        self.rhs, self.x, self.r, self.z, self.p, self.ax = (zeros() for _ in range(6))
        unknown = masks.unknown_node
        self.unknown = i32(unknown)
        self.cond_r = f64(masks.cond_r)
        self.cond_z = f64(masks.cond_z)
        inverse = np.zeros(grid.node_shape)
        inverse[unknown] = 1.0 / masks.diagonal[unknown]
        self.inv_diag = f64(inverse)
        boundary = boundary_potential_array(masks, potentials)
        offset = apply_operator(masks, boundary)
        offset[~unknown] = 0.0
        self.offset = f64(offset)
        self.boundary = f64(boundary)
        self.ratio = f64(masks.charge_to_source)
        self.levels: list[_DeviceLevel] = []
        for depth, level in enumerate(self.hierarchy.levels):
            if depth == 0:
                self.levels.append(_DeviceLevel(level, dev, x=self.x, b=self.rhs))
            else:
                self.levels.append(_DeviceLevel(level, dev))
        self.coarsest_inverse = f64(self.hierarchy.coarsest_inverse)
        self.coarsest_gather = i32(self.hierarchy.coarsest_active_index)
        self.coarsest_count = int(self.hierarchy.coarsest_active_index.size)
        # residual reductions (same deterministic pattern as the block-Thomas check)
        self.threads = int(min(REDUCTION_THREADS, max(64, self.node_count)))
        self.groups = (self.threads + REDUCTION_GROUP - 1) // REDUCTION_GROUP
        self.partial_a = wp.zeros(self.threads, dtype=wp.float64, device=dev)
        self.partial_b = wp.zeros(self.threads, dtype=wp.float64, device=dev)
        self.stage_a = wp.zeros(self.groups, dtype=wp.float64, device=dev)
        self.stage_b = wp.zeros(self.groups, dtype=wp.float64, device=dev)
        # scalars: [.., .., .., .., .., rr, rhs2, worst contract ratio]
        self.scalars = wp.zeros(8, dtype=wp.float64, device=dev)
        self.abs2 = float(config.absolute_tolerance) ** 2
        self.rel2 = float(config.relative_tolerance) ** 2
        self.use_graph = bool(use_graph) and device.is_cuda
        self.graph = None
        self.bound_inputs: tuple | None = None
        self.host_memory_bytes = int(self.hierarchy.coarsest_inverse.nbytes)
        self.last_worst_ratio = 0.0
        self.device_memory_bytes = int(
            sum(level.nbytes for level in self.levels)
            + self.coarsest_inverse.capacity + self.coarsest_gather.capacity
            + sum(a.capacity for a in (self.r, self.z, self.p, self.ax, self.unknown, self.cond_r, self.cond_z,
                                       self.inv_diag, self.offset, self.boundary, self.ratio))
        )
        self.launches_per_solve = self._count_launches()

    # -- bookkeeping ----------------------------------------------------------------------------
    def _count_launches(self) -> int:
        depth = len(self.levels)
        per_cycle = (depth - 1) * (self.pre + self.post + 2 + (self.pre % 2) + (self.post % 2)) + 1
        # source, |rhs|^2 (3), warm start, cycles, residual check (matvec, residual, |r|^2 (3), track), Dirichlet
        return 1 + 3 + 1 + self.cycles * per_cycle + 6 + 1

    def describe(self) -> dict[str, object]:
        record = self.hierarchy.to_dict()
        record.update({
            "method": "device-mg",
            "cycles": self.cycles,
            "pre_sweeps": self.pre,
            "post_sweeps": self.post,
            "omega": self.omega,
            "smoother": "damped Jacobi",
            "launches_per_solve": self.launches_per_solve,
            "device_memory_bytes": self.device_memory_bytes,
            "host_memory_bytes": self.host_memory_bytes,
        })
        return record

    # -- device sequence ------------------------------------------------------------------------
    def _jacobi(self, level: _DeviceLevel, x_in, x_out) -> None:
        wp.launch(mg_jacobi_kernel, dim=level.count,
                  inputs=[x_in, level.b, level.active, level.ni, level.nj, level.diag, level.inv_diag,
                          level.a_e, level.a_n, level.a_ne, level.a_nw, self.omega, x_out],
                  device=self.device)

    def _smooth(self, level: _DeviceLevel, sweeps: int) -> None:
        current, other = level.x, level.t
        for _ in range(sweeps):
            self._jacobi(level, current, other)
            current, other = other, current
        if current is not level.x:
            wp.launch(mg_copy_kernel, dim=level.count, inputs=[current, level.x], device=self.device)

    def _vcycle(self, depth: int) -> None:
        level = self.levels[depth]
        if depth == len(self.levels) - 1:
            wp.launch(mg_dense_kernel, dim=self.coarsest_count,
                      inputs=[self.coarsest_inverse, self.coarsest_gather, self.coarsest_count, level.b, level.x],
                      device=self.device)
            return
        coarse = self.levels[depth + 1]
        self._smooth(level, self.pre)
        wp.launch(mg_restrict_kernel, dim=coarse.count,
                  inputs=[level.x, level.b, level.active, level.ni, level.nj, level.diag,
                          level.a_e, level.a_n, level.a_ne, level.a_nw, level.r_idx, level.r_w, coarse.b, coarse.x],
                  device=self.device)
        self._vcycle(depth + 1)
        wp.launch(mg_prolong_kernel, dim=level.count, inputs=[coarse.x, level.p_idx, level.p_w, level.x], device=self.device)
        self._smooth(level, self.post)

    def _norm2(self, a, partial, stage, slot: int) -> None:
        n = self.node_count
        dev = self.device
        wp.launch(dot_stride_kernel, dim=self.threads, inputs=[a, a, n, self.threads, partial], device=dev)
        wp.launch(reduce_stage_kernel, dim=self.groups, inputs=[partial, self.threads, REDUCTION_GROUP, stage], device=dev)
        wp.launch(final_sum_kernel, dim=1, inputs=[stage, self.groups, self.scalars, slot], device=dev)

    def _residual_check(self) -> None:
        """True residual of the current ``x`` with the mesh conductances -> scalars[5]; contract ratio -> running max."""

        n = self.node_count
        dev = self.device
        wp.launch(matvec_kernel, dim=n, inputs=[self.x, self.unknown, self.cond_r, self.cond_z, self.nr, self.nz, self.ax], device=dev)
        wp.launch(residual_kernel, dim=n, inputs=[self.rhs, self.ax, self.inv_diag, self.r, self.z, self.p], device=dev)
        self._norm2(self.r, self.partial_a, self.stage_a, SLOT_RR)
        wp.launch(mg_track_kernel, dim=1, inputs=[self.scalars, self.abs2, self.rel2], device=dev)

    def _solve_sequence(self, q_e, q_i, surface, phi_out) -> None:
        n = self.node_count
        dev = self.device
        wp.launch(source_kernel, dim=n, inputs=[q_e, q_i, self.ratio, surface, self.offset, self.unknown, self.rhs], device=dev)
        self._norm2(self.rhs, self.partial_b, self.stage_b, SLOT_RHS2)
        wp.launch(mg_init_kernel, dim=n, inputs=[phi_out, self.unknown, self.x], device=dev)
        for _ in range(self.cycles):
            self._vcycle(0)
        self._residual_check()
        wp.launch(apply_dirichlet_kernel, dim=n, inputs=[self.x, self.boundary, self.unknown, phi_out], device=dev)

    # -- WarpBlockThomas interface ----------------------------------------------------------------
    def bind(self, q_e, q_i, surface, phi_out) -> None:
        """Fix the input/output arrays; load the kernels; capture the whole solve as one graph."""

        self.bound_inputs = (q_e, q_i, surface, phi_out)
        self.graph = None
        self._solve_sequence(q_e, q_i, surface, phi_out)   # loads the module before any capture
        wp.synchronize_device(self.device)
        wp.launch(mg_reset_kernel, dim=1, inputs=[self.scalars], device=self.device)
        if self.use_graph:
            with wp.ScopedCapture(device=self.device) as capture:
                self._solve_sequence(q_e, q_i, surface, phi_out)
            self.graph = capture.graph

    def solve_sequence(self, q_e, q_i, surface, phi_out) -> None:
        """The raw launch sequence of ``solve`` (for capture inside an enclosing step graph)."""

        self._solve_sequence(q_e, q_i, surface, phi_out)

    def solve(self, q_e, q_i, surface, phi_out) -> tuple[int, float, float]:
        if self.graph is not None and self.bound_inputs is not None and all(a is b for a, b in zip(self.bound_inputs, (q_e, q_i, surface, phi_out))):
            wp.capture_launch(self.graph)
        else:
            self._solve_sequence(q_e, q_i, surface, phi_out)
        return self.cycles, float("nan"), float("nan")

    def queue_residual_check(self) -> None:
        """No-op: every solve already queues its true residual and the interval maximum (read by ``verify``)."""

    def verify(self) -> tuple[float, float]:
        """Read the last true residual and the interval's worst contract ratio; raise if any step missed the contract."""

        scalars = self.scalars.numpy()
        true_residual = sqrt(max(float(scalars[SLOT_RR]), 0.0))
        rhs_norm = sqrt(max(float(scalars[SLOT_RHS2]), 0.0))
        worst = float(scalars[SLOT_WORST])
        tolerance = max(self.config.absolute_tolerance, self.config.relative_tolerance * rhs_norm)
        self.last_worst_ratio = worst
        if not isfinite(true_residual) or true_residual > tolerance or not isfinite(worst) or worst > 1.0:
            raise PIC2DConvergenceError(
                f"fixed-cycle multigrid solve failed its residual contract: last true residual {true_residual:.3e} > "
                f"{tolerance:.3e} or worst |r|^2/bound^2 over the interval {worst:.3e} > 1 "
                f"({self.cycles} V({self.pre},{self.post}) cycles); increase mg_cycles and resume from the last checkpoint"
            )
        wp.launch(mg_reset_kernel, dim=1, inputs=[self.scalars], device=self.device)
        return true_residual, tolerance


__all__ = ["SLOT_RHS2", "SLOT_RR", "SLOT_WORST", "WarpPoissonMG"]
