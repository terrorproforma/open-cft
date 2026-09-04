"""NVIDIA Warp (CPU or CUDA) backend reproducing the numpy reference cycle.

Contract with ``simulation.CPUBackend``:

* charge deposition uses the identical ``2**-40`` fixed-point integer
  accumulation, so node charges are bit-identical and order independent;
* gather, Boris push and position advance use the same formulas and operation
  order (differences are limited to FMA contraction roundoff);
* the Poisson solve is the same Jacobi-PCG on the same operator with
  deterministic two-stage reductions; the published potential must meet the
  same recomputed true-residual contract;
* MCC and injection use Warp's counter-based ``rand_init`` streams seeded from
  ``(seed, step, stream)``, so a GPU run is reproducible and resumable; its
  random numbers differ from the CPU stream, so parity is distributional.

Ledger sums (kinetic energies, field work) use float64 atomics and may differ
from a rerun at roundoff level; the dynamical state does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
import time
from typing import Any

import numpy as np

from . import warp_ion_mcc
from .fields import MagneticFieldMap
from .ion_mcc import IonNullCollisionMCC
from .kernels import FIXED_POINT_SCALE
from .mcc import MAX_EXCITATION_LEVELS, NullCollisionMCC, XenonCrossSections
from .mesh import MeshMasks
from .models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    EV_J,
    PIC2DConvergenceError,
    PIC2DDeviceError,
    PIC2DStabilityError,
    PIC2DValidationError,
    ParticleArrays,
    electron_species,
    xenon_ion_species,
)
from .poisson import Poisson2D, apply_operator, boundary_potential_array
from .see import see_birth_bound
from .simulation import (
    CATHODE_RATE_KEY,
    IEDF_BINS,
    INELASTIC_LOSS_PER_WEIGHT_KEY,
    PEAK_WINDOW_SUM_KEYS,
    SEE_KEYS,
    THETA_BINS,
    DiagnosticAccumulator,
    PIC2DConfig,
    SimulationState,
    StepTally,
    build_ion_mcc,
    iedf_max_ev,
    neutral_shape_cells,
    omega_pe_gate_min_macro_particles,
    peak_node_debye,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dependency
    wp = None  # type: ignore[assignment]

REDUCTION_THREADS = 4096
REDUCTION_GROUP = 64
# Threads per block for particle kernels that reduce per-particle scalars with
# block-level tile sums instead of one same-address atomic per particle.
PARTICLE_BLOCK = 256
# Lanes per output row in the block-Thomas sweeps (one block per unknown).  Warp
# compiles a module for one block width, so this equals PARTICLE_BLOCK.
THOMAS_LANES = PARTICLE_BLOCK
# Per-step RNG streams read from the device seed table (v1.4): 0 = MCC, 1 = injection,
# 2 = anomalous scattering, 3 = ion-neutral MCC (v2.3.0), 4 = SEE (v2.2.0); the seeds of streams 0-2 are unchanged, so
# every earlier run replays bitwise (stream_seed(seed, step, id) per table column).  Row = steps since the last host sync.
SEED_STREAMS = 5
SEED_STREAM_IDS = (1, 2, 3, 4, 5)       # stream_seed(seed, step, id) per table column

# Interval statistics slots (float64 device array, read once per host sync).
STATS_WORK = 0
STATS_MAX_SPEED2 = 1
STATS_INVALID = 2
STATS_PEAK_DENSITY = 3
STATS_E_COUNTS = 4          # anode, exit, wall (electrons)
STATS_E_ENERGY = 7
STATS_I_COUNTS = 10         # anode, exit, wall (ions)
STATS_I_ENERGY = 13
STATS_MCC = 16              # candidates, elastic, excitation, ionization, null
STATS_KE_INJECTED = 21
STATS_KE_BORN = 22
STATS_OVERFLOW = 23
STATS_INJECTED = 24         # v1.4: injected macro-electrons (device-side injection control)
STATS_ANOMALOUS = 25        # v1.4: Bohm-scattering hook tally
STATS_CARRY = 26            # v1.4: injection carry after the latest step (overwritten, not accumulated)
# v2.0 momentum ledger (kg m/s, macro weight applied) and plume tallies
STATS_PZ_IMPULSE = 27       # m W (v_z^+ - v_z^-) over pushes (E and B)
STATS_PZ_ELECTRIC = 28      # q E_z W dt over pushes
STATS_PZ_COLLISIONS = 29    # m_e W dv_z in MCC and Bohm scattering
STATS_PZ_BORN = 30          # momentum of ionisation products
STATS_PZ_INJECTED = 31      # momentum of emitted electrons
STATS_E_PZ = 32             # anode, exit, wall (electrons)
STATS_I_PZ = 35             # anode, exit, wall (ions)
STATS_BODY_FACE_E = 38      # front-face hits (electrons)
STATS_BODY_FACE_I = 39      # front-face hits (ions)
STATS_ION_PLUME = 40        # ionisation events downstream of the exit plane
STATS_PEAK_DENSITY_RESOLVED = 41   # v2.0.4: single-step peak electron density over nodes holding >= the gate's macro-particle floor (the omega_pe dt gate statistic)
# v2.3.0 (xe_collision_set_v2): ion-neutral MCC block (warp_ion_mcc.ION_STATS_* offsets) and per-level excitation counts
STATS_ION_MCC = 42
STATS_EXC_LEVELS = STATS_ION_MCC + warp_ion_mcc.ION_STATS_COUNT
# v2.2.0 SEE stage (warp_see.py): tallies of the wall's secondary emission per sync interval (after the v2.3.0 blocks)
STATS_SEE_IMPACTS = STATS_EXC_LEVELS + MAX_EXCITATION_LEVELS   # electron impacts on the emitting wall
STATS_SEE_EMITTED = STATS_SEE_IMPACTS + 1                        # emitted macro-electrons (electron-induced)
STATS_SEE_ION_EMITTED = STATS_SEE_IMPACTS + 2                    # emitted macro-electrons (ion-induced)
STATS_SEE_BACKSCATTERED = STATS_SEE_IMPACTS + 3                  # elastic + inelastic among the emitted
STATS_SEE_KE = STATS_SEE_IMPACTS + 4                             # W-scaled emitted kinetic energy
STATS_SEE_PZ = STATS_SEE_IMPACTS + 5                             # W-scaled emitted axial momentum
STATS_SEE_YIELD_SUM = STATS_SEE_IMPACTS + 6                      # sum of delta over the electron impacts
STATS_SEE_CLAMPED = STATS_SEE_IMPACTS + 7                        # impacts whose yield exceeded the per-impact cap
STATS_SIZE = STATS_SEE_CLAMPED + 1


def padded_dim(count: int, block: int) -> int:
    """Launch dimension rounded up so every tile block is fully populated."""

    return ((max(int(count), 1) + block - 1) // block) * block


def warp_available() -> bool:
    return wp is not None


def resolve_device(device: str):
    if wp is None:
        raise PIC2DDeviceError("NVIDIA Warp is unavailable")
    wp.init()
    requested = device.strip().lower()
    if requested == "cuda":
        requested = "cuda:0"
    if requested != "cpu" and not requested.startswith("cuda:"):
        raise PIC2DDeviceError("device must be 'cpu', 'cuda', or 'cuda:N'")
    try:
        return wp.get_device(requested)
    except (RuntimeError, ValueError) as error:
        raise PIC2DDeviceError(f"Warp device {requested!r} is unavailable") from error


def device_available(device: str) -> bool:
    try:
        resolve_device(device)
    except PIC2DDeviceError:
        return False
    return True


def stream_seed(seed: int, step: int, stream: int) -> int:
    digest = sha256(f"{seed}:{step}:{stream}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


if wp is not None:
    F64 = wp.float64

    # ------------------------------------------------------------------ deposition
    @wp.kernel
    def deposit_fixed_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32),
        slots: wp.array(dtype=wp.int32), slot: int,
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, scale: F64,
        accumulator: wp.array(dtype=wp.int64),
    ):
        p = wp.tid()
        if p >= slots[slot] or alive[p] == 0:
            return
        fr = r[p] / dr
        fz = (z[p] - z_min) / dz
        i = wp.clamp(int(wp.floor(fr)), 0, nr - 1)
        j = wp.clamp(int(wp.floor(fz)), 0, nz - 1)
        s = fr - F64(i)
        t = fz - F64(j)
        w00 = (F64(1.0) - s) * (F64(1.0) - t)
        w10 = s * (F64(1.0) - t)
        w01 = (F64(1.0) - s) * t
        w11 = s * t
        stride = nz + 1
        base = i * stride + j
        wp.atomic_add(accumulator, base, wp.int64(wp.rint(w00 * scale)))
        wp.atomic_add(accumulator, base + stride, wp.int64(wp.rint(w10 * scale)))
        wp.atomic_add(accumulator, base + 1, wp.int64(wp.rint(w01 * scale)))
        wp.atomic_add(accumulator, base + stride + 1, wp.int64(wp.rint(w11 * scale)))

    @wp.kernel
    def int_to_charge_kernel(accumulator: wp.array(dtype=wp.int64), factor: F64, out: wp.array(dtype=F64)):
        n = wp.tid()
        out[n] = F64(accumulator[n]) * factor

    @wp.kernel
    def deposit_moment_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32),
        slots: wp.array(dtype=wp.int32), slot: int,
        vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int,
        weight_sum: wp.array(dtype=F64), vr_sum: wp.array(dtype=F64), vt_sum: wp.array(dtype=F64),
        vz_sum: wp.array(dtype=F64), v2_sum: wp.array(dtype=F64),
    ):
        p = wp.tid()
        if p >= slots[slot] or alive[p] == 0:
            return
        fr = r[p] / dr
        fz = (z[p] - z_min) / dz
        i = wp.clamp(int(wp.floor(fr)), 0, nr - 1)
        j = wp.clamp(int(wp.floor(fz)), 0, nz - 1)
        s = fr - F64(i)
        t = fz - F64(j)
        stride = nz + 1
        base = i * stride + j
        v2 = vr[p] * vr[p] + vt[p] * vt[p] + vz[p] * vz[p]
        for k in range(4):
            di = k % 2
            dj = k // 2
            w = F64(0.0)
            if k == 0:
                w = (F64(1.0) - s) * (F64(1.0) - t)
            elif k == 1:
                w = s * (F64(1.0) - t)
            elif k == 2:
                w = (F64(1.0) - s) * t
            else:
                w = s * t
            n = base + di * stride + dj
            wp.atomic_add(weight_sum, n, w)
            wp.atomic_add(vr_sum, n, w * vr[p])
            wp.atomic_add(vt_sum, n, w * vt[p])
            wp.atomic_add(vz_sum, n, w * vz[p])
            wp.atomic_add(v2_sum, n, w * v2)

    # ------------------------------------------------------------------ field solve
    @wp.kernel
    def source_kernel(
        q_e: wp.array(dtype=F64), q_i: wp.array(dtype=F64), ratio: wp.array(dtype=F64),
        surface: wp.array(dtype=F64), offset: wp.array(dtype=F64), unknown: wp.array(dtype=wp.int32),
        rhs: wp.array(dtype=F64),
    ):
        n = wp.tid()
        if unknown[n] == 0:
            rhs[n] = F64(0.0)
        else:
            rhs[n] = (q_e[n] + q_i[n]) * ratio[n] + surface[n] - offset[n]

    @wp.kernel
    def host_source_kernel(
        q_e: wp.array(dtype=F64), q_i: wp.array(dtype=F64), ratio: wp.array(dtype=F64),
        surface: wp.array(dtype=F64), source: wp.array(dtype=F64),
    ):
        n = wp.tid()
        source[n] = (q_e[n] + q_i[n]) * ratio[n] + surface[n]

    @wp.kernel
    def matvec_kernel(
        x: wp.array(dtype=F64), unknown: wp.array(dtype=wp.int32),
        cond_r: wp.array(dtype=F64), cond_z: wp.array(dtype=F64), nr: int, nz: int,
        out: wp.array(dtype=F64),
    ):
        n = wp.tid()
        if unknown[n] == 0:
            out[n] = F64(0.0)
            return
        stride = nz + 1
        i = n / stride
        j = n - i * stride
        xi = x[n]
        acc = F64(0.0)
        if i < nr:
            acc += cond_r[i * stride + j] * (xi - x[n + stride])
        if i > 0:
            acc += cond_r[(i - 1) * stride + j] * (xi - x[n - stride])
        if j < nz:
            acc += cond_z[i * nz + j] * (xi - x[n + 1])
        if j > 0:
            acc += cond_z[i * nz + j - 1] * (xi - x[n - 1])
        out[n] = acc

    @wp.kernel
    def residual_kernel(rhs: wp.array(dtype=F64), ax: wp.array(dtype=F64), inv_diag: wp.array(dtype=F64),
                        r: wp.array(dtype=F64), z: wp.array(dtype=F64), p: wp.array(dtype=F64)):
        n = wp.tid()
        value = rhs[n] - ax[n]
        r[n] = value
        z[n] = inv_diag[n] * value
        p[n] = inv_diag[n] * value

    @wp.kernel
    def dot_stride_kernel(a: wp.array(dtype=F64), b: wp.array(dtype=F64), count: int, threads: int,
                          partial: wp.array(dtype=F64)):
        # Fixed-structure (deterministic) stage 1: thread k sums indices k, k+T, k+2T, ...
        k = wp.tid()
        acc = F64(0.0)
        n = k
        while n < count:
            acc += a[n] * b[n]
            n += threads
        partial[k] = acc

    @wp.kernel
    def dot2_stride_kernel(a: wp.array(dtype=F64), b: wp.array(dtype=F64), c: wp.array(dtype=F64),
                           count: int, threads: int, partial_ab: wp.array(dtype=F64), partial_ac: wp.array(dtype=F64)):
        k = wp.tid()
        acc1 = F64(0.0)
        acc2 = F64(0.0)
        n = k
        while n < count:
            acc1 += a[n] * b[n]
            acc2 += a[n] * c[n]
            n += threads
        partial_ab[k] = acc1
        partial_ac[k] = acc2

    @wp.kernel
    def reduce_stage_kernel(partial_in: wp.array(dtype=F64), count: int, group: int, partial_out: wp.array(dtype=F64)):
        k = wp.tid()
        start = k * group
        stop = wp.min(start + group, count)
        acc = F64(0.0)
        for n in range(start, stop):
            acc += partial_in[n]
        partial_out[k] = acc

    @wp.kernel
    def final_sum_kernel(partial: wp.array(dtype=F64), count: int, out: wp.array(dtype=F64), slot: int):
        acc = F64(0.0)
        for k in range(count):
            acc += partial[k]
        out[slot] = acc

    @wp.kernel
    def alpha_kernel(scalars: wp.array(dtype=F64)):
        # scalars: [rho, pq, alpha, beta, rho_new, rr]; a converged (zero) residual
        # makes every further iteration an exact no-op instead of 0/0.
        if scalars[1] > F64(0.0):
            scalars[2] = scalars[0] / scalars[1]
        else:
            scalars[2] = F64(0.0)

    @wp.kernel
    def update_kernel(x: wp.array(dtype=F64), r: wp.array(dtype=F64), z: wp.array(dtype=F64),
                      p: wp.array(dtype=F64), q: wp.array(dtype=F64), inv_diag: wp.array(dtype=F64),
                      scalars: wp.array(dtype=F64)):
        n = wp.tid()
        alpha = scalars[2]
        x[n] = x[n] + alpha * p[n]
        rn = r[n] - alpha * q[n]
        r[n] = rn
        z[n] = inv_diag[n] * rn

    @wp.kernel
    def beta_kernel(scalars: wp.array(dtype=F64)):
        if scalars[0] > F64(0.0):
            scalars[3] = scalars[4] / scalars[0]
        else:
            scalars[3] = F64(0.0)
        scalars[0] = scalars[4]

    @wp.kernel
    def direction_kernel(p: wp.array(dtype=F64), z: wp.array(dtype=F64), scalars: wp.array(dtype=F64)):
        n = wp.tid()
        p[n] = z[n] + scalars[3] * p[n]

    @wp.kernel
    def efield_kernel(
        phi: wp.array(dtype=F64), code_r: wp.array(dtype=wp.int32), code_z: wp.array(dtype=wp.int32),
        dr: F64, dz: F64, nz: int, e_r: wp.array(dtype=F64), e_z: wp.array(dtype=F64),
    ):
        # codes: 0 none, 1 central, 2 forward2, 3 forward1, 4 backward2, 5 backward1
        n = wp.tid()
        stride = nz + 1
        cr = code_r[n]
        value = F64(0.0)
        if cr == 1:
            value = -(phi[n + stride] - phi[n - stride]) / (F64(2.0) * dr)
        elif cr == 2:
            value = -(F64(-3.0) * phi[n] + F64(4.0) * phi[n + stride] - phi[n + 2 * stride]) / (F64(2.0) * dr)
        elif cr == 3:
            value = -(phi[n + stride] - phi[n]) / dr
        elif cr == 4:
            value = -(F64(3.0) * phi[n] - F64(4.0) * phi[n - stride] + phi[n - 2 * stride]) / (F64(2.0) * dr)
        elif cr == 5:
            value = -(phi[n] - phi[n - stride]) / dr
        e_r[n] = value
        cz = code_z[n]
        value = F64(0.0)
        if cz == 1:
            value = -(phi[n + 1] - phi[n - 1]) / (F64(2.0) * dz)
        elif cz == 2:
            value = -(F64(-3.0) * phi[n] + F64(4.0) * phi[n + 1] - phi[n + 2]) / (F64(2.0) * dz)
        elif cz == 3:
            value = -(phi[n + 1] - phi[n]) / dz
        elif cz == 4:
            value = -(F64(3.0) * phi[n] - F64(4.0) * phi[n - 1] + phi[n - 2]) / (F64(2.0) * dz)
        elif cz == 5:
            value = -(phi[n] - phi[n - 1]) / dz
        e_z[n] = value

    @wp.kernel
    def block_forward_kernel(
        g: wp.array(dtype=F64), c: wp.array(dtype=F64), b: wp.array(dtype=F64), y: wp.array(dtype=F64), i: int, m: int,
    ):
        # y_i = b_i - c_{i-1} * (G_{i-1} y_{i-1});  G stored row-major g[i, j, k].
        # One THOMAS_LANES-thread block per output j: lane q sums k = q, q+P, ...
        # (coalesced across lanes), then a deterministic tile reduction.
        t = wp.tid()
        lanes = wp.block_dim()
        j = t / lanes
        q = t % lanes
        acc = F64(0.0)
        if i > 0:
            prev = i - 1
            base_g = prev * m * m + j * m
            base_y = prev * m
            k = q
            while k < m:
                acc += g[base_g + k] * y[base_y + k]
                k += lanes
        value = wp.tile_extract(wp.tile_sum(wp.tile(acc)), 0)
        if q == 0:
            if i == 0:
                y[j] = b[j]
            else:
                y[i * m + j] = b[i * m + j] - c[(i - 1) * m + j] * value

    @wp.kernel
    def block_backward_kernel(
        g: wp.array(dtype=F64), c: wp.array(dtype=F64), y: wp.array(dtype=F64), x: wp.array(dtype=F64), i: int, m: int, last: int,
    ):
        # x_i = G_i (y_i - c_i * x_{i+1})
        t = wp.tid()
        lanes = wp.block_dim()
        j = t / lanes
        q = t % lanes
        base_g = i * m * m + j * m
        base = i * m
        acc = F64(0.0)
        k = q
        if i == last:
            while k < m:
                acc += g[base_g + k] * y[base + k]
                k += lanes
        else:
            nxt = (i + 1) * m
            while k < m:
                acc += g[base_g + k] * (y[base + k] - c[base + k] * x[nxt + k])
                k += lanes
        value = wp.tile_extract(wp.tile_sum(wp.tile(acc)), 0)
        if q == 0:
            x[base + j] = value

    @wp.kernel
    def apply_dirichlet_kernel(x: wp.array(dtype=F64), boundary: wp.array(dtype=F64), unknown: wp.array(dtype=wp.int32),
                               phi: wp.array(dtype=F64)):
        n = wp.tid()
        if unknown[n] == 0:
            phi[n] = boundary[n]
        else:
            phi[n] = x[n]

    @wp.kernel
    def axpy_kernel(y: wp.array(dtype=F64), a: F64, x: wp.array(dtype=F64)):
        n = wp.tid()
        y[n] = y[n] + a * x[n]

    @wp.kernel
    def abs_axpy_kernel(y: wp.array(dtype=F64), a: wp.array(dtype=F64), x: wp.array(dtype=F64)):
        n = wp.tid()
        y[n] = y[n] + a[n] * wp.abs(x[n])

    # ------------------------------------------------------------------ particles
    @wp.func
    def relativistic_boris(
        vx: F64, vy: F64, vz: F64, ex: F64, ez: F64, bx: F64, bz: F64, charge: F64, mass: F64, dt: F64
    ):
        c2 = F64(299792458.0) * F64(299792458.0)
        speed2 = vx * vx + vy * vy + vz * vz
        gamma = F64(1.0) / wp.sqrt(F64(1.0) - speed2 / c2)
        ux = gamma * vx
        uy = gamma * vy
        uz = gamma * vz
        half_kick = charge * dt / (F64(2.0) * mass)
        ux_m = ux + half_kick * ex
        uy_m = uy
        uz_m = uz + half_kick * ez
        gamma_m = wp.sqrt(F64(1.0) + (ux_m * ux_m + uy_m * uy_m + uz_m * uz_m) / c2)
        tx = charge * dt * bx / (F64(2.0) * mass * gamma_m)
        tz = charge * dt * bz / (F64(2.0) * mass * gamma_m)
        t2 = tx * tx + tz * tz
        sx = F64(2.0) * tx / (F64(1.0) + t2)
        sz = F64(2.0) * tz / (F64(1.0) + t2)
        upx = ux_m + (uy_m * tz - uz_m * F64(0.0))
        upy = uy_m + (uz_m * tx - ux_m * tz)
        upz = uz_m + (ux_m * F64(0.0) - uy_m * tx)
        ux_p = ux_m + (upy * sz - upz * F64(0.0))
        uy_p = uy_m + (upz * sx - upx * sz)
        uz_p = uz_m + (upx * F64(0.0) - upy * sx)
        ux_n = ux_p + half_kick * ex
        uy_n = uy_p
        uz_n = uz_p + half_kick * ez
        gamma_n = wp.sqrt(F64(1.0) + (ux_n * ux_n + uy_n * uy_n + uz_n * uz_n) / c2)
        return wp.vec3d(ux_n / gamma_n, uy_n / gamma_n, uz_n / gamma_n)

    @wp.func
    def kinetic_energy(vx: F64, vy: F64, vz: F64, mass_weight: F64) -> F64:
        c2 = F64(299792458.0) * F64(299792458.0)
        speed2 = vx * vx + vy * vy + vz * vz
        gm1 = speed2 / c2 / (F64(1.0) + wp.sqrt(F64(1.0) - speed2 / c2))
        return gm1 * mass_weight * c2

    @wp.kernel
    def push_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64),
        vz: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32),
        e_r: wp.array(dtype=F64), e_z: wp.array(dtype=F64), b_r: wp.array(dtype=F64), b_z: wp.array(dtype=F64),
        dr: F64, dz: F64, z_min: F64, z_max: F64, nr: int, nz: int,
        plasma_cell: wp.array(dtype=wp.int32), top_cell: wp.array(dtype=wp.int32), plasma_node: wp.array(dtype=wp.int32),
        slots: wp.array(dtype=wp.int32), slot: int,
        charge: F64, mass: F64, weight: F64, dt: F64, scale: F64, charge_sign: wp.int64,
        wall_accumulator: wp.array(dtype=wp.int64),
        stats: wp.array(dtype=F64), count_slot: int, energy_slot: int,
        accumulate: int, wall_columns: wp.array(dtype=F64), wall_column_energy: wp.array(dtype=F64),
        exit_bins: wp.array(dtype=F64),
        # v2.0 plume block: far-field side boundary, front face, momentum tallies and plume histograms
        has_plume: int, z_exit: F64, r_outer: F64, r_exit: F64, pz_slot: int, body_slot: int, is_ion: int,
        side_bins: wp.array(dtype=F64), theta_bins: wp.array(dtype=F64), iedf_bins: wp.array(dtype=F64), iedf_scale: F64,
        theta_scale: F64,
        # v2.2.0: per-slot wall-impact flag consumed by the SEE stage (warp_see.py) at the end of the step; the dead
        # slot keeps its pre-push arrays, from which the stage reconstructs the impact
        wall_hit: wp.array(dtype=wp.int32),
    ):
        # stats layout (see WarpBackend.STATS): counts at count_slot+code-1, energies at
        # energy_slot+code-1, momenta at pz_slot+code-1, work at 0, max speed^2 at 1, invalid at 2,
        # total impulse at STATS_PZ_IMPULSE, electric impulse at STATS_PZ_ELECTRIC.
        # No early returns: the per-particle work, speed^2 and impulses are reduced per
        # block with tile sums (launch dim padded to PARTICLE_BLOCK).
        p = wp.tid()
        active = 0
        if p < slots[slot]:
            if alive[p] != 0:
                active = 1
        work = F64(0.0)
        speed2 = F64(0.0)
        dpz = F64(0.0)
        epz = F64(0.0)
        if active != 0:
            rp = r[p]
            zp = z[p]
            fr = rp / dr
            fz = (zp - z_min) / dz
            i = wp.clamp(int(wp.floor(fr)), 0, nr - 1)
            j = wp.clamp(int(wp.floor(fz)), 0, nz - 1)
            s = fr - F64(i)
            t = fz - F64(j)
            w00 = (F64(1.0) - s) * (F64(1.0) - t)
            w10 = s * (F64(1.0) - t)
            w01 = (F64(1.0) - s) * t
            w11 = s * t
            stride = nz + 1
            n00 = i * stride + j
            n10 = n00 + stride
            n01 = n00 + 1
            n11 = n10 + 1
            ex = w00 * e_r[n00] + w10 * e_r[n10] + w01 * e_r[n01] + w11 * e_r[n11]
            ez = w00 * e_z[n00] + w10 * e_z[n10] + w01 * e_z[n01] + w11 * e_z[n11]
            bx = w00 * b_r[n00] + w10 * b_r[n10] + w01 * b_r[n01] + w11 * b_r[n11]
            bz = w00 * b_z[n00] + w10 * b_z[n10] + w01 * b_z[n01] + w11 * b_z[n11]
            vx0 = vr[p]
            vy0 = vt[p]
            vz0 = vz[p]
            mass_weight = mass * weight
            k_before = kinetic_energy(vx0, vy0, vz0, mass_weight)
            v = relativistic_boris(vx0, vy0, vz0, ex, ez, bx, bz, charge, mass, dt)
            x_new = rp + v[0] * dt
            y_new = v[1] * dt
            r_new = wp.sqrt(x_new * x_new + y_new * y_new)
            cos_a = F64(1.0)
            sin_a = F64(0.0)
            if r_new > F64(0.0):
                cos_a = x_new / r_new
                sin_a = y_new / r_new
            vr_new = v[0] * cos_a + v[1] * sin_a
            vt_new = -v[0] * sin_a + v[1] * cos_a
            vz_new = v[2]
            z_new = zp + vz_new * dt
            k_after = kinetic_energy(vr_new, vt_new, vz_new, mass_weight)
            work = k_after - k_before
            speed2 = vr_new * vr_new + vt_new * vt_new + vz_new * vz_new
            dpz = mass_weight * (vz_new - vz0)
            epz = charge * weight * dt * ez
            # boundary classification (identical to kernels.classify_boundary)
            code = 0
            if z_new < z_min:
                code = 1
            elif z_new >= z_max:
                code = 2
            elif has_plume != 0 and z_new >= z_exit and r_new >= r_outer:
                code = 2
            else:
                fi = int(wp.floor(r_new / dr))
                jj = wp.clamp(int(wp.floor((z_new - z_min) / dz)), 0, nz - 1)
                inside = 0
                if fi < nr:
                    if plasma_cell[fi * nz + jj] != 0:
                        inside = 1
                if inside == 0:
                    code = 3
                    if fi > nr:
                        code = 4
            if code == 0:
                r[p] = r_new
                z[p] = z_new
                vr[p] = vr_new
                vt[p] = vt_new
                vz[p] = vz_new
            else:
                alive[p] = 0
                if code == 4:
                    wp.atomic_add(stats, 2, F64(1.0))
                else:
                    wp.atomic_add(stats, count_slot + code - 1, F64(1.0))
                    wp.atomic_add(stats, energy_slot + code - 1, k_after)
                    wp.atomic_add(stats, pz_slot + code - 1, mass_weight * vz_new)
                    if code == 3 and has_plume != 0 and r_new >= r_exit:
                        wp.atomic_add(stats, body_slot, F64(1.0))
                    if code == 3:
                        # renormalised bilinear surface deposit on plasma nodes
                        gr = wp.clamp(r_new / dr, F64(0.0), F64(nr) - F64(1.0e-12))
                        gz = wp.clamp((z_new - z_min) / dz, F64(0.0), F64(nz) - F64(1.0e-12))
                        ii = int(wp.floor(gr))
                        jw = int(wp.floor(gz))
                        ss = gr - F64(ii)
                        tt = gz - F64(jw)
                        m00 = ii * stride + jw
                        m10 = m00 + stride
                        m01 = m00 + 1
                        m11 = m10 + 1
                        a00 = F64(0.0)
                        a10 = F64(0.0)
                        a01 = F64(0.0)
                        a11 = F64(0.0)
                        if plasma_node[m00] != 0:
                            a00 = (F64(1.0) - ss) * (F64(1.0) - tt)
                        if plasma_node[m10] != 0:
                            a10 = ss * (F64(1.0) - tt)
                        if plasma_node[m01] != 0:
                            a01 = (F64(1.0) - ss) * tt
                        if plasma_node[m11] != 0:
                            a11 = ss * tt
                        total = F64(0.0)
                        total += a00
                        total += a10
                        total += a01
                        total += a11
                        if total <= F64(0.0):
                            wp.atomic_add(stats, 2, F64(1.0))
                        else:
                            wall_hit[p] = 1
                            wp.atomic_add(wall_accumulator, m00, wp.int64(wp.rint(a00 / total * scale)) * charge_sign)
                            wp.atomic_add(wall_accumulator, m10, wp.int64(wp.rint(a10 / total * scale)) * charge_sign)
                            wp.atomic_add(wall_accumulator, m01, wp.int64(wp.rint(a01 / total * scale)) * charge_sign)
                            wp.atomic_add(wall_accumulator, m11, wp.int64(wp.rint(a11 / total * scale)) * charge_sign)
                            if accumulate != 0:
                                col = wp.clamp(int((z_new - z_min) / dz), 0, nz - 1)
                                wp.atomic_add(wall_columns, col, F64(1.0))
                                wp.atomic_add(wall_column_energy, col, k_after)
                    elif code == 2:
                        if accumulate != 0:
                            if z_new >= z_max:
                                bin_index = wp.clamp(int(r_new / dr), 0, nr - 1)
                                wp.atomic_add(exit_bins, bin_index, F64(1.0))
                            else:
                                side_index = wp.clamp(int((z_new - z_min) / dz), 0, nz - 1)
                                wp.atomic_add(side_bins, side_index, F64(1.0))
                            if is_ion != 0:
                                # ion current per solid angle about the aperture centre and the IEDF (as the CPU reference)
                                theta = wp.degrees(wp.atan2(r_new, wp.max(z_new - z_exit, F64(0.0))))
                                wp.atomic_add(theta_bins, wp.clamp(int(theta * theta_scale), 0, THETA_BINS - 1), F64(1.0))
                                energy_ev = k_after / (weight * F64(1.602176634e-19))
                                wp.atomic_add(iedf_bins, wp.clamp(int(energy_ev * iedf_scale), 0, IEDF_BINS - 1), F64(1.0))
        # block reductions (deterministic tree order) replace 2 same-address atomics per particle
        block_work = wp.tile_sum(wp.tile(work))
        wp.tile_atomic_add(stats, block_work, offset=0)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dpz)), offset=STATS_PZ_IMPULSE)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(epz)), offset=STATS_PZ_ELECTRIC)
        block_speed = wp.tile_extract(wp.tile_max(wp.tile(speed2)), 0)
        if p % wp.block_dim() == 0:
            wp.atomic_max(stats, 1, block_speed)

    @wp.kernel
    def wall_int_to_charge_kernel(accumulator: wp.array(dtype=wp.int64), factor: F64, surface: wp.array(dtype=F64)):
        n = wp.tid()
        surface[n] = surface[n] + F64(accumulator[n]) * factor
        accumulator[n] = wp.int64(0)

    # ------------------------------------------------------------------ MCC
    @wp.func
    def sigma_lookup(table: wp.array(dtype=F64), points: int, step_ev: F64, max_ev: F64, energy: F64, process: int) -> F64:
        e = wp.clamp(energy, F64(0.0), max_ev)
        position = e / step_ev
        index = wp.min(int(wp.floor(position)), points - 2)
        fraction = position - F64(index)
        lower = table[process * points + index]
        upper = table[process * points + index + 1]
        return lower + fraction * (upper - lower)

    @wp.func
    def isotropic(speed: F64, u1: F64, u2: F64):
        cos_chi = F64(1.0) - F64(2.0) * u1
        sin_chi = wp.sqrt(wp.max(F64(0.0), F64(1.0) - cos_chi * cos_chi))
        phi = F64(6.283185307179586) * u2
        return wp.vec3d(speed * sin_chi * wp.cos(phi), speed * sin_chi * wp.sin(phi), speed * cos_chi)

    @wp.func
    def rotate_about_field(vr: F64, vt: F64, vz: F64, br: F64, bz: F64, phi: F64):
        # v2.1.0 perpendicular-rotation Bohm model: Rodrigues rotation of (v_r, v_theta, v_z) about the unit field
        # b = (b_r, 0, b_z) / |B| by phi - v_parallel invariant, |v| preserved, gyro-phase reset (same operation
        # order as sensitivity.rotate_about_field); |B| = 0 leaves the velocity unchanged
        b_mag = wp.sqrt(br * br + bz * bz)
        if b_mag <= F64(0.0):
            return wp.vec3d(vr, vt, vz)
        nr = br / b_mag
        nz = bz / b_mag
        c = wp.cos(phi)
        s = wp.sin(phi)
        k = (nr * vr + nz * vz) * (F64(1.0) - c)
        return wp.vec3d(vr * c + (-nz * vt) * s + nr * k, vt * c + (nz * vr - nr * vz) * s, vz * c + (nr * vt) * s + nz * k)

    @wp.kernel
    def mcc_kernel(
        vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32),
        slots: wp.array(dtype=wp.int32), seed_table: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32),
        probability: F64, nu_max: F64, neutral_density_ctrl: wp.array(dtype=F64),
        table: wp.array(dtype=F64), points: int, step_ev: F64, max_ev: F64,
        # v2.3.0: excitation levels = table rows 1 .. n_exc with thresholds exc_thresholds[k]; ionisation is row n_exc + 1.
        # One level reproduces the v1.x-v2.0.6 arithmetic exactly; per-level counts go to exc_level_slot + k when n_exc > 1.
        exc_thresholds: wp.array(dtype=F64), n_exc: int, exc_level_slot: int, threshold_ion: F64, b_ev: F64, ion_thermal: F64,
        ionize: wp.array(dtype=wp.int32), sec_vr: wp.array(dtype=F64), sec_vt: wp.array(dtype=F64), sec_vz: wp.array(dtype=F64),
        ion_vr: wp.array(dtype=F64), ion_vt: wp.array(dtype=F64), ion_vz: wp.array(dtype=F64),
        stats: wp.array(dtype=F64), tally_slot: int, flag_bound: int,
        # v2.0: positions + cell-centred neutral density shape (two-zone field), plume-birth and momentum tallies
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), shape_cell: wp.array(dtype=F64), has_plume: int, z_exit: F64,
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, mass_weight: F64,
        # v2.0.5: born-ledger tallies (ion kinetic energy, axial momentum of the ionisation products) reduced
        # here with the other tallies instead of three strided sums + single-thread adds after the spawn
        ion_mass_weight: F64, ke_born_slot: int, pz_born_slot: int,
    ):
        # No early returns: the five tallies (candidates, elastic, excitation,
        # ionisation, null) are reduced per block with tile sums.
        # The per-step seed comes from the device seed table (row = steps since the
        # last host sync, column 0 = MCC stream) so the launch is graph-capturable (v1.4).
        p = wp.tid()
        if p < flag_bound:
            ionize[p] = 0
        candidate = F64(0.0)
        outcome = 4  # 0 elastic, 1 excitation, 2 ionisation, 3 null, 4 no candidate
        dpz = F64(0.0)
        plume_birth = F64(0.0)
        ke_born = F64(0.0)
        pz_born = F64(0.0)
        active = 0
        if p < slots[0]:
            if alive[p] != 0:
                active = 1
        if active != 0:
            seed = seed_table[SEED_STREAMS * counter[0]]
            state = wp.rand_init(seed, p)
            u0 = F64(wp.randf(state))
            if u0 < probability:
                candidate = F64(1.0)
                vx = vr[p]
                vy = vt[p]
                vzp = vz[p]
                speed2 = vx * vx + vy * vy + vzp * vzp
                speed = wp.sqrt(speed2)
                m_e = F64(9.1093837139e-31)
                ev = F64(1.602176634e-19)
                energy = F64(0.5) * m_e * speed2 / ev
                # the instantaneous n_g (n_g0 x scale) is device-resident so the CUDA-graph replay
                # sees every inventory update (a captured scalar would freeze it at capture time)
                neutral_density = neutral_density_ctrl[0]
                density = neutral_density
                if has_plume != 0:
                    ci = wp.clamp(int(wp.floor(r[p] / dr)), 0, nr - 1)
                    cj = wp.clamp(int(wp.floor((z[p] - z_min) / dz)), 0, nz - 1)
                    density = neutral_density * shape_cell[ci * nz + cj]
                nu0 = density * sigma_lookup(table, points, step_ev, max_ev, energy, 0) * speed
                # total excitation frequency over the levels (a single level: 0 + x == x, the legacy value bitwise)
                nu1 = F64(0.0)
                for k in range(n_exc):
                    nu_k = density * sigma_lookup(table, points, step_ev, max_ev, energy, 1 + k) * speed
                    if energy < exc_thresholds[k]:
                        nu_k = F64(0.0)
                    nu1 += nu_k
                nu2 = density * sigma_lookup(table, points, step_ev, max_ev, energy, 1 + n_exc) * speed
                if energy < threshold_ion:
                    nu2 = F64(0.0)
                selector = F64(wp.randf(state)) * nu_max
                c1 = nu0
                c2 = nu0 + nu1
                c3 = c2 + nu2
                u2 = F64(wp.randf(state))
                u3 = F64(wp.randf(state))
                if selector < c1:
                    v = isotropic(speed, u2, u3)
                    vr[p] = v[0]
                    vt[p] = v[1]
                    vz[p] = v[2]
                    outcome = 0
                elif selector < c2:
                    # which level: the selector's position inside [c1, c2) split by the per-level frequencies (the same
                    # partition the CPU reference applies with its cumulative sums); one level -> level 0 without a test
                    level = int(0)
                    if n_exc > 1:
                        acc = F64(c1)
                        chosen = int(-1)
                        for k in range(n_exc):
                            nu_k = density * sigma_lookup(table, points, step_ev, max_ev, energy, 1 + k) * speed
                            if energy < exc_thresholds[k]:
                                nu_k = F64(0.0)
                            acc += nu_k
                            if chosen < 0 and selector < acc:
                                chosen = k
                        if chosen < 0:
                            chosen = n_exc - 1     # round-off at the upper edge: the last level
                        level = chosen
                        wp.atomic_add(stats, exc_level_slot + level, F64(1.0))
                    remaining = wp.max(energy - exc_thresholds[level], F64(0.0))
                    v = isotropic(wp.sqrt(F64(2.0) * remaining * ev / m_e), u2, u3)
                    vr[p] = v[0]
                    vt[p] = v[1]
                    vz[p] = v[2]
                    outcome = 1
                elif selector < c3:
                    available = wp.max(energy - threshold_ion, F64(0.0))
                    u4 = F64(wp.randf(state))
                    secondary = b_ev * wp.tan(u4 * wp.atan(available / (F64(2.0) * b_ev)))
                    secondary = wp.clamp(secondary, F64(0.0), available)
                    primary = available - secondary
                    v = isotropic(wp.sqrt(F64(2.0) * primary * ev / m_e), u2, u3)
                    vr[p] = v[0]
                    vt[p] = v[1]
                    vz[p] = v[2]
                    u5 = F64(wp.randf(state))
                    u6 = F64(wp.randf(state))
                    sv = isotropic(wp.sqrt(F64(2.0) * secondary * ev / m_e), u5, u6)
                    sec_vr[p] = sv[0]
                    sec_vt[p] = sv[1]
                    sec_vz[p] = sv[2]
                    u7 = wp.max(F64(wp.randf(state)), F64(1.0e-30))
                    u8 = F64(wp.randf(state))
                    u9 = wp.max(F64(wp.randf(state)), F64(1.0e-30))
                    u10 = F64(wp.randf(state))
                    rad1 = wp.sqrt(F64(-2.0) * wp.log(u7))
                    rad2 = wp.sqrt(F64(-2.0) * wp.log(u9))
                    ivr = ion_thermal * rad1 * wp.cos(F64(6.283185307179586) * u8)
                    ivt = ion_thermal * rad1 * wp.sin(F64(6.283185307179586) * u8)
                    ivz = ion_thermal * rad2 * wp.cos(F64(6.283185307179586) * u10)
                    ion_vr[p] = ivr
                    ion_vt[p] = ivt
                    ion_vz[p] = ivz
                    ionize[p] = 1
                    outcome = 2
                    if has_plume != 0 and z[p] >= z_exit:
                        plume_birth = F64(1.0)
                    # v2.0.5: the same per-particle values the removed energy_sum / momentum_sum kernels
                    # formed from ion_v* / sec_vz after the spawn (summation order differs -> round-off)
                    ke_born = kinetic_energy(ivr, ivt, ivz, ion_mass_weight)
                    pz_born = ion_mass_weight * ivz + mass_weight * sv[2]
                else:
                    outcome = 3
                if outcome < 3:
                    dpz = mass_weight * (vz[p] - vzp)
        e_flag = F64(0.0)
        x_flag = F64(0.0)
        i_flag = F64(0.0)
        n_flag = F64(0.0)
        if outcome == 0:
            e_flag = F64(1.0)
        elif outcome == 1:
            x_flag = F64(1.0)
        elif outcome == 2:
            i_flag = F64(1.0)
        elif outcome == 3:
            n_flag = F64(1.0)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(candidate)), offset=tally_slot)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(e_flag)), offset=tally_slot + 1)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(x_flag)), offset=tally_slot + 2)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(i_flag)), offset=tally_slot + 3)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(n_flag)), offset=tally_slot + 4)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dpz)), offset=STATS_PZ_COLLISIONS)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(plume_birth)), offset=STATS_ION_PLUME)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(ke_born)), offset=ke_born_slot)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(pz_born)), offset=pz_born_slot)

    @wp.kernel
    def spawn_kernel(
        ionize: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32),
        e_capacity: int, i_capacity: int,
        r: wp.array(dtype=F64), z: wp.array(dtype=F64),
        sec_vr: wp.array(dtype=F64), sec_vt: wp.array(dtype=F64), sec_vz: wp.array(dtype=F64),
        ion_vr: wp.array(dtype=F64), ion_vt: wp.array(dtype=F64), ion_vz: wp.array(dtype=F64),
        e_r: wp.array(dtype=F64), e_z: wp.array(dtype=F64), e_vr: wp.array(dtype=F64), e_vt: wp.array(dtype=F64),
        e_vz: wp.array(dtype=F64), e_alive: wp.array(dtype=wp.int32),
        i_r: wp.array(dtype=F64), i_z: wp.array(dtype=F64), i_vr: wp.array(dtype=F64), i_vt: wp.array(dtype=F64),
        i_vz: wp.array(dtype=F64), i_alive: wp.array(dtype=wp.int32),
        stats: wp.array(dtype=F64), overflow_slot: int,
        # v2.0.5: the born ion joins the frozen ion charge here (exact int64 fixed point, order independent ->
        # bitwise the former separate born deposit) and, when accumulating, the ionisation-rate window map
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, scale: F64,
        ion_accumulator: wp.array(dtype=wp.int64), accumulate: int, ionization_map: wp.array(dtype=F64),
    ):
        # ``slots[0]``/``slots[1]`` are the electron/ion slot counts *before* this
        # step's births; ``spawn_commit_kernel`` advances them afterwards.
        p = wp.tid()
        if p >= slots[0] or ionize[p] == 0:
            return
        k = offsets[p]
        de = slots[0] + k
        di = slots[1] + k
        if de >= e_capacity or di >= i_capacity:
            # fail closed at the next host sync instead of writing out of bounds
            wp.atomic_add(stats, overflow_slot, F64(1.0))
            return
        rp = r[p]
        zp = z[p]
        e_r[de] = rp
        e_z[de] = zp
        e_vr[de] = sec_vr[p]
        e_vt[de] = sec_vt[p]
        e_vz[de] = sec_vz[p]
        e_alive[de] = 1
        i_r[di] = rp
        i_z[di] = zp
        i_vr[di] = ion_vr[p]
        i_vt[di] = ion_vt[p]
        i_vz[di] = ion_vz[p]
        i_alive[di] = 1
        # bilinear deposit of the born ion at the parent's position (identical arithmetic to deposit_fixed_kernel /
        # deposit_unit_kernel over the former born_r / born_z / born_flag arrays)
        fr = rp / dr
        fz = (zp - z_min) / dz
        i = wp.clamp(int(wp.floor(fr)), 0, nr - 1)
        j = wp.clamp(int(wp.floor(fz)), 0, nz - 1)
        s = fr - F64(i)
        t = fz - F64(j)
        w00 = (F64(1.0) - s) * (F64(1.0) - t)
        w10 = s * (F64(1.0) - t)
        w01 = (F64(1.0) - s) * t
        w11 = s * t
        stride = nz + 1
        base = i * stride + j
        wp.atomic_add(ion_accumulator, base, wp.int64(wp.rint(w00 * scale)))
        wp.atomic_add(ion_accumulator, base + stride, wp.int64(wp.rint(w10 * scale)))
        wp.atomic_add(ion_accumulator, base + 1, wp.int64(wp.rint(w01 * scale)))
        wp.atomic_add(ion_accumulator, base + stride + 1, wp.int64(wp.rint(w11 * scale)))
        if accumulate != 0:
            wp.atomic_add(ionization_map, base, w00)
            wp.atomic_add(ionization_map, base + stride, w10)
            wp.atomic_add(ionization_map, base + 1, w01)
            wp.atomic_add(ionization_map, base + stride + 1, w11)

    @wp.kernel
    def spawn_commit_kernel(
        ionize: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32),
        e_capacity: int, i_capacity: int,
    ):
        # total births = exclusive offset of the last candidate + its own flag
        n = slots[0]
        total = 0
        if n > 0:
            total = offsets[n - 1] + ionize[n - 1]
        slots[0] = wp.min(slots[0] + total, e_capacity)
        slots[1] = wp.min(slots[1] + total, i_capacity)

    @wp.kernel
    def add_injected_slots_kernel(slots: wp.array(dtype=wp.int32), ctrl: wp.array(dtype=F64)):
        slots[0] = slots[0] + int(ctrl[1])

    @wp.kernel
    def inject_control_kernel(ctrl: wp.array(dtype=F64), stats: wp.array(dtype=F64), count_slot: int):
        # ctrl[0] = fractional carry, ctrl[1] = macro-electrons to inject this step, ctrl[2] = rate per
        # step (device-resident so the continuity rule can change it between syncs without a
        # re-capture, v2.0).  Same arithmetic as the v1.0-v1.3 host bookkeeping (expected = rate + carry; floor).
        expected = ctrl[2] + ctrl[0]
        n = wp.floor(expected)
        ctrl[0] = expected - n
        ctrl[1] = n
        stats[count_slot] = stats[count_slot] + n

    @wp.kernel
    def tick_kernel(counter: wp.array(dtype=wp.int32)):
        counter[0] = counter[0] + 1

    @wp.kernel
    def carry_kernel(ctrl: wp.array(dtype=F64), stats: wp.array(dtype=F64), slot: int):
        stats[slot] = ctrl[0]      # latest injection carry, read back with the interval statistics

    @wp.kernel
    def inject_kernel(
        seed_table: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32), ctrl: wp.array(dtype=F64),
        slots: wp.array(dtype=wp.int32), capacity: int, r_max: F64, z_max: F64, dz: F64, thermal: F64,
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64),
        vz: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32), stats: wp.array(dtype=F64), slot: int, mass_weight: F64,
        overflow_slot: int,
        # v2.0: mode 0 = legacy exit-plane half-Maxwellian; mode 1 = cathode annulus (uniform in volume,
        # isotropic Maxwellian): r^2 = r_in2 + u (r_out2 - r_in2), z = z_start + u z_span
        mode: int, r_in2: F64, r_span2: F64, z_start: F64, z_span: F64, pz_slot: int,
    ):
        k = wp.tid()
        if F64(k) >= ctrl[1]:
            return
        base = slots[0]
        if base + k >= capacity:
            wp.atomic_add(stats, overflow_slot, F64(1.0))
            return
        seed = seed_table[SEED_STREAMS * counter[0] + 1]
        state = wp.rand_init(seed, k)
        u0 = F64(wp.randf(state))
        u1 = F64(wp.randf(state))
        u2 = wp.max(F64(wp.randf(state)), F64(1.0e-30))
        u3 = F64(wp.randf(state))
        u4 = F64(wp.randf(state))
        u5 = F64(wp.randf(state))
        u6 = wp.max(F64(wp.randf(state)), F64(1.0e-30))
        d = base + k
        rad1 = wp.sqrt(F64(-2.0) * wp.log(u2))
        vrk = thermal * rad1 * wp.cos(F64(6.283185307179586) * u3)
        vtk = thermal * rad1 * wp.sin(F64(6.283185307179586) * u3)
        if mode == 0:
            r[d] = r_max * wp.sqrt(u0) * (F64(1.0) - F64(1.0e-9))
            z[d] = z_max - F64(0.5) * dz * u1 - F64(1.0e-9) * dz
            vzk = -thermal * wp.sqrt(F64(-2.0) * wp.log(u6))
        else:
            r[d] = wp.sqrt(r_in2 + u0 * r_span2)
            z[d] = z_start + u1 * z_span
            vzk = thermal * wp.sqrt(F64(-2.0) * wp.log(wp.max(u4, F64(1.0e-30)))) * wp.cos(F64(6.283185307179586) * u5)
        vr[d] = vrk
        vt[d] = vtk
        vz[d] = vzk
        alive[d] = 1
        wp.atomic_add(stats, slot, kinetic_energy(vrk, vtk, vzk, mass_weight))
        wp.atomic_add(stats, pz_slot, mass_weight * vzk)

    @wp.kernel
    def bohm_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        alive: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32),
        seed_table: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32),
        b_r: wp.array(dtype=F64), b_z: wp.array(dtype=F64), dr: F64, dz: F64, z_min: F64, nr: int, nz: int,
        alpha_dt_e_over_m: F64, stats: wp.array(dtype=F64), tally_slot: int, mass_weight: F64, mode: int,
    ):
        # v1.4 sensitivity hook: speed-preserving velocity change with probability 1 - exp(-alpha omega_ce dt) at
        # the particle's |B| (same map as the CPU reference).  mode 0 = isotropic redirect (v1.4, also randomises
        # v_parallel); mode 1 = v2.1.0 rotation of the perpendicular velocity about B by a uniform random angle
        # (Brandt et al. 2016: v_parallel and |v| unchanged, gyro-centre shifted).
        p = wp.tid()
        hit = F64(0.0)
        dpz = F64(0.0)
        active = 0
        if p < slots[0]:
            if alive[p] != 0:
                active = 1
        if active != 0:
            fr = r[p] / dr
            fz = (z[p] - z_min) / dz
            i = wp.clamp(int(wp.floor(fr)), 0, nr - 1)
            j = wp.clamp(int(wp.floor(fz)), 0, nz - 1)
            s = fr - F64(i)
            t = fz - F64(j)
            stride = nz + 1
            n00 = i * stride + j
            w00 = (F64(1.0) - s) * (F64(1.0) - t)
            w10 = s * (F64(1.0) - t)
            w01 = (F64(1.0) - s) * t
            w11 = s * t
            bx = w00 * b_r[n00] + w10 * b_r[n00 + stride] + w01 * b_r[n00 + 1] + w11 * b_r[n00 + stride + 1]
            bz = w00 * b_z[n00] + w10 * b_z[n00 + stride] + w01 * b_z[n00 + 1] + w11 * b_z[n00 + stride + 1]
            probability = F64(1.0) - wp.exp(-alpha_dt_e_over_m * wp.sqrt(bx * bx + bz * bz))
            seed = seed_table[SEED_STREAMS * counter[0] + 2]
            state = wp.rand_init(seed, p)
            u0 = F64(wp.randf(state))
            if u0 < probability:
                u1 = F64(wp.randf(state))
                if mode == 0:
                    speed = wp.sqrt(vr[p] * vr[p] + vt[p] * vt[p] + vz[p] * vz[p])
                    u2 = F64(wp.randf(state))
                    v = isotropic(speed, u1, u2)
                else:
                    v = rotate_about_field(vr[p], vt[p], vz[p], bx, bz, F64(6.283185307179586) * u1)
                dpz = mass_weight * (v[2] - vz[p])
                vr[p] = v[0]
                vt[p] = v[1]
                vz[p] = v[2]
                hit = F64(1.0)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(hit)), offset=tally_slot)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dpz)), offset=STATS_PZ_COLLISIONS)

    @wp.kernel
    def compact_kernel(
        alive: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32), slot: int,
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        r2: wp.array(dtype=F64), z2: wp.array(dtype=F64), vr2: wp.array(dtype=F64), vt2: wp.array(dtype=F64), vz2: wp.array(dtype=F64),
        alive2: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        if p >= slots[slot] or alive[p] == 0:
            return
        d = offsets[p]
        r2[d] = r[p]
        z2[d] = z[p]
        vr2[d] = vr[p]
        vt2[d] = vt[p]
        vz2[d] = vz[p]
        alive2[d] = 1

    @wp.kernel
    def energy_sum_kernel(vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
                          alive: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32), slot: int, threads: int,
                          mass_weight: F64, partial: wp.array(dtype=F64)):
        k = wp.tid()
        count = slots[slot]
        acc = F64(0.0)
        p = k
        while p < count:
            if alive[p] != 0:
                acc += kinetic_energy(vr[p], vt[p], vz[p], mass_weight)
            p += threads
        partial[k] = acc

    @wp.kernel
    def momentum_sum_kernel(vz: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32), slot: int,
                            threads: int, mass_weight: F64, partial: wp.array(dtype=F64)):
        # v2.0: sum of m W v_z over the flagged particles (series momentum, born momentum)
        k = wp.tid()
        count = slots[slot]
        acc = F64(0.0)
        p = k
        while p < count:
            if alive[p] != 0:
                acc += mass_weight * vz[p]
            p += threads
        partial[k] = acc

    @wp.kernel
    def peak_density_kernel(q_e: wp.array(dtype=F64), inverse_volume: wp.array(dtype=F64), min_charge_c: F64, stats: wp.array(dtype=F64), slot_raw: int, slot_resolved: int):
        # raw = the peak over every plasma node (shot-noise extreme on small-volume nodes); resolved = the peak over nodes whose
        # deposit holds >= the gate's macro-particle floor (|q_e| >= min_charge_c) - the v2.0.4 omega_pe dt gate statistic
        n = wp.tid()
        q = wp.abs(q_e[n])
        density = q * inverse_volume[n]
        wp.atomic_max(stats, slot_raw, density)
        if q >= min_charge_c:
            wp.atomic_max(stats, slot_resolved, density)

    @wp.kernel
    def sum_kernel(values: wp.array(dtype=F64), count: int, threads: int, partial: wp.array(dtype=F64)):
        k = wp.tid()
        acc = F64(0.0)
        n = k
        while n < count:
            acc += values[n]
            n += threads
        partial[k] = acc


def efield_stencil_codes(masks: MeshMasks) -> tuple[np.ndarray, np.ndarray]:
    """Per-node finite-difference stencil selectors matching ``poisson.electric_field_nodes``."""

    grid = masks.grid
    nr, nz = grid.cell_shape
    plasma = masks.plasma_node

    def shifted(mask: np.ndarray, di: int, dj: int) -> np.ndarray:
        out = np.zeros_like(mask)
        src_i = slice(max(0, di), nr + 1 + min(0, di))
        dst_i = slice(max(0, -di), nr + 1 + min(0, -di))
        src_j = slice(max(0, dj), nz + 1 + min(0, dj))
        dst_j = slice(max(0, -dj), nz + 1 + min(0, -dj))
        out[dst_i, dst_j] = mask[src_i, src_j]
        return out

    codes = []
    for step in ((1, 0), (0, 1)):
        plus1 = shifted(plasma, *step)
        minus1 = shifted(plasma, -step[0], -step[1])
        plus2 = shifted(plasma, 2 * step[0], 2 * step[1])
        minus2 = shifted(plasma, -2 * step[0], -2 * step[1])
        code = np.zeros(grid.node_shape, dtype=np.int32)
        code[plasma & plus1 & minus1] = 1
        code[plasma & plus1 & ~minus1 & plus2] = 2
        code[plasma & plus1 & ~minus1 & ~plus2] = 3
        code[plasma & minus1 & ~plus1 & minus2] = 4
        code[plasma & minus1 & ~plus1 & ~minus2] = 5
        codes.append(code)
    codes[0][masks.axis_node] = 0
    return codes[0], codes[1]


class WarpPoisson:
    """Jacobi-PCG on the device with deterministic reductions and optional CUDA graphs.

    The solve runs an adaptive chunk of iterations (sized from the previous
    solve) and then recomputes the *true* residual on the device before a
    single host read; only unconverged solves pay for further chunks.
    """

    def __init__(self, masks: MeshMasks, potentials, config, device, *, use_graph: bool = True, block_iterations: int = 8) -> None:
        if wp is None:
            raise PIC2DDeviceError("NVIDIA Warp is unavailable")
        self.masks = masks
        self.config = config
        self.device = device
        self.node_count = int(np.prod(masks.grid.node_shape))
        nr, nz = masks.grid.cell_shape
        self.nr, self.nz = nr, nz
        f64 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), dtype=wp.float64, device=device)  # noqa: E731
        self.unknown = wp.array(masks.unknown_node.ravel().astype(np.int32), dtype=wp.int32, device=device)
        self.cond_r = f64(masks.cond_r)
        self.cond_z = f64(masks.cond_z)
        inverse = np.zeros(masks.grid.node_shape)
        inverse[masks.unknown_node] = 1.0 / masks.diagonal[masks.unknown_node]
        self.inv_diag = f64(inverse)
        boundary = boundary_potential_array(masks, potentials)
        offset = apply_operator(masks, boundary)
        offset[~masks.unknown_node] = 0.0
        self.offset = f64(offset)
        self.boundary = f64(boundary)
        self.ratio = f64(masks.charge_to_source)
        zeros = lambda: wp.zeros(self.node_count, dtype=wp.float64, device=device)  # noqa: E731
        self.rhs, self.x, self.r, self.z, self.p, self.q, self.ax = (zeros() for _ in range(7))
        self.threads = int(min(REDUCTION_THREADS, max(64, self.node_count)))
        self.groups = (self.threads + REDUCTION_GROUP - 1) // REDUCTION_GROUP
        self.partial_a = wp.zeros(self.threads, dtype=wp.float64, device=device)
        self.partial_b = wp.zeros(self.threads, dtype=wp.float64, device=device)
        self.stage_a = wp.zeros(self.groups, dtype=wp.float64, device=device)
        self.stage_b = wp.zeros(self.groups, dtype=wp.float64, device=device)
        # scalars: [rho, pq, alpha, beta, rho_new, rr, rhs2, unused]
        self.scalars = wp.zeros(8, dtype=wp.float64, device=device)
        self.block_iterations = int(block_iterations)
        self.use_graph = bool(use_graph) and device.is_cuda
        self.graph = None
        self.last_iterations = 4 * self.block_iterations
        self._warm()

    def _warm(self) -> None:
        self._iteration_block()
        wp.synchronize_device(self.device)
        if self.use_graph:
            with wp.ScopedCapture(device=self.device) as capture:
                self._iteration_block()
            self.graph = capture.graph
        for array in (self.x, self.r, self.z, self.p, self.q, self.ax, self.scalars):
            array.zero_()
        wp.synchronize_device(self.device)

    def _reduce(self, partial, stage, slot: int) -> None:
        wp.launch(reduce_stage_kernel, dim=self.groups, inputs=[partial, self.threads, REDUCTION_GROUP, stage], device=self.device)
        wp.launch(final_sum_kernel, dim=1, inputs=[stage, self.groups, self.scalars, slot], device=self.device)

    def _dot(self, a, b, slot: int) -> None:
        wp.launch(dot_stride_kernel, dim=self.threads, inputs=[a, b, self.node_count, self.threads, self.partial_a], device=self.device)
        self._reduce(self.partial_a, self.stage_a, slot)

    def _dot2(self, a, b, c, slot_ab: int, slot_ac: int) -> None:
        wp.launch(dot2_stride_kernel, dim=self.threads, inputs=[a, b, c, self.node_count, self.threads, self.partial_a, self.partial_b], device=self.device)
        self._reduce(self.partial_a, self.stage_a, slot_ab)
        self._reduce(self.partial_b, self.stage_b, slot_ac)

    def _iteration(self) -> None:
        n = self.node_count
        wp.launch(matvec_kernel, dim=n, inputs=[self.p, self.unknown, self.cond_r, self.cond_z, self.nr, self.nz, self.q], device=self.device)
        self._dot(self.p, self.q, 1)
        wp.launch(alpha_kernel, dim=1, inputs=[self.scalars], device=self.device)
        wp.launch(update_kernel, dim=n, inputs=[self.x, self.r, self.z, self.p, self.q, self.inv_diag, self.scalars], device=self.device)
        self._dot2(self.r, self.z, self.r, 4, 5)
        wp.launch(beta_kernel, dim=1, inputs=[self.scalars], device=self.device)
        wp.launch(direction_kernel, dim=n, inputs=[self.p, self.z, self.scalars], device=self.device)

    def _iteration_block(self) -> None:
        for _ in range(self.block_iterations):
            self._iteration()

    def _run_block(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._iteration_block()

    def _restart(self) -> None:
        """Recompute the true residual from ``x`` and restart the CG direction."""

        n = self.node_count
        wp.launch(matvec_kernel, dim=n, inputs=[self.x, self.unknown, self.cond_r, self.cond_z, self.nr, self.nz, self.ax], device=self.device)
        wp.launch(residual_kernel, dim=n, inputs=[self.rhs, self.ax, self.inv_diag, self.r, self.z, self.p], device=self.device)
        self._dot2(self.r, self.z, self.r, 0, 5)

    def solve(self, q_e, q_i, surface, phi_out) -> tuple[int, float, float]:
        """Solve into ``phi_out`` (device node array); ``self.x`` keeps the warm start.

        Returns ``(iterations, true_residual, tolerance)`` and raises on failure.
        """

        n = self.node_count
        wp.launch(source_kernel, dim=n, inputs=[q_e, q_i, self.ratio, surface, self.offset, self.unknown, self.rhs], device=self.device)
        self._dot(self.rhs, self.rhs, 6)
        self._restart()
        iterations = 0
        chunk_blocks = max(1, -(-self.last_iterations // self.block_iterations))
        checks = 0
        restarts = 0
        tolerance = 0.0
        true_residual = float("inf")
        while True:
            # Run a chunk of iterations (no restart: CG keeps its Krylov history),
            # then read the recurrence residual once.
            for _ in range(chunk_blocks):
                if iterations >= self.config.max_iterations:
                    break
                self._run_block()
                iterations += self.block_iterations
            scalars = self.scalars.numpy()
            checks += 1
            rhs_norm = sqrt(max(float(scalars[6]), 0.0))
            tolerance = max(self.config.absolute_tolerance, self.config.relative_tolerance * rhs_norm)
            recurrence = sqrt(max(float(scalars[5]), 0.0))
            if not isfinite(recurrence):
                raise PIC2DConvergenceError("GPU conjugate gradient residual became nonfinite")
            if recurrence <= tolerance or iterations >= self.config.max_iterations:
                # Recompute the true residual (also restarts the direction) and verify.
                self._restart()
                true_residual = sqrt(max(float(self.scalars.numpy()[5]), 0.0))
                restarts += 1
                if true_residual <= tolerance:
                    break
                if iterations >= self.config.max_iterations or restarts > 3:
                    raise PIC2DConvergenceError(
                        f"GPU Poisson solve did not meet its residual contract: true residual {true_residual:.3e} > "
                        f"{tolerance:.3e} after {iterations} iterations"
                    )
            chunk_blocks = max(1, chunk_blocks // 2)
        # Budget for the next warm-started solve (additive-increase/decrease
        # tuning): shrink when one chunk sufficed, grow when extra chunks were
        # needed, so the typical solve costs one residual read.
        if checks == 1:
            self.last_iterations = max(self.block_iterations, int(iterations * 0.9))
        else:
            self.last_iterations = max(self.block_iterations, int(iterations * 1.1))
        wp.launch(apply_dirichlet_kernel, dim=n, inputs=[self.x, self.boundary, self.unknown, phi_out], device=self.device)
        return iterations, true_residual, tolerance


class WarpBlockThomas:
    """Exact block-Thomas direct solve on the device, blocked along radial rows.

    Unknowns are grouped by radial index ``i`` (one block = the whole axial row,
    ``m = nz + 1`` nodes, identity rows for non-unknown nodes); the coupling to
    row ``i + 1`` is the diagonal ``-cond_r``.  The Schur complements
    ``S_i = D_i - C_{i-1} S_{i-1}^{-1} C_{i-1}`` are inverted once on the host;
    a solve is then ``2 nr + 1`` dense row-block matvecs (``m`` threads each,
    fixed summation order, no atomics) captured in a single CUDA graph, so the
    field solve costs no host synchronisation.  The true residual of the most
    recent solve is verified at every host sync against the same contract as
    the iterative and host paths (``relative_tolerance * |rhs|``).
    """

    def __init__(self, masks: MeshMasks, potentials, config, device, *, use_graph: bool = True) -> None:
        if wp is None:
            raise PIC2DDeviceError("NVIDIA Warp is unavailable")
        self.masks = masks
        self.config = config
        self.device = device
        grid = masks.grid
        nr, nz = grid.cell_shape
        self.nr, self.nz = nr, nz
        self.m = nz + 1
        self.node_count = int(np.prod(grid.node_shape))
        unknown = masks.unknown_node
        m = self.m
        g_blocks = np.zeros((nr + 1, m, m), dtype=np.float64)
        coupling = np.zeros((nr + 1, m), dtype=np.float64)
        previous_g: np.ndarray | None = None
        for i in range(nr + 1):
            d = np.zeros((m, m), dtype=np.float64)
            row_unknown = unknown[i]
            diag = np.where(row_unknown, masks.diagonal[i], 1.0)
            d[np.arange(m), np.arange(m)] = diag
            both = row_unknown[:-1] & row_unknown[1:]
            cz = np.where(both, -masks.cond_z[i], 0.0)
            idx = np.arange(m - 1)
            d[idx, idx + 1] = cz
            d[idx + 1, idx] = cz
            if i > 0 and previous_g is not None:
                c_prev = coupling[i - 1]
                d = d - (c_prev[:, None] * previous_g) * c_prev[None, :]
            g = np.linalg.inv(d)
            if not np.isfinite(g).all():
                raise PIC2DConvergenceError("block-Thomas Schur complement inversion produced nonfinite values")
            g_blocks[i] = g
            previous_g = g
            if i < nr:
                both_r = row_unknown & unknown[i + 1]
                coupling[i] = np.where(both_r, -masks.cond_r[i], 0.0)
        f64 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), dtype=wp.float64, device=device)  # noqa: E731
        self.g_blocks = f64(g_blocks)
        self.coupling = f64(coupling)
        self.unknown = wp.array(unknown.ravel().astype(np.int32), dtype=wp.int32, device=device)
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
        zeros = lambda: wp.zeros(self.node_count, dtype=wp.float64, device=device)  # noqa: E731
        self.rhs, self.x, self.y, self.r, self.z, self.p, self.ax = (zeros() for _ in range(7))
        self.threads = int(min(REDUCTION_THREADS, max(64, self.node_count)))
        self.groups = (self.threads + REDUCTION_GROUP - 1) // REDUCTION_GROUP
        self.partial_a = wp.zeros(self.threads, dtype=wp.float64, device=device)
        self.partial_b = wp.zeros(self.threads, dtype=wp.float64, device=device)
        self.stage_a = wp.zeros(self.groups, dtype=wp.float64, device=device)
        self.stage_b = wp.zeros(self.groups, dtype=wp.float64, device=device)
        # scalars: [unused, unused, unused, unused, unused, rr, rhs2, unused]
        self.scalars = wp.zeros(8, dtype=wp.float64, device=device)
        self.use_graph = bool(use_graph) and device.is_cuda
        self.graph = None
        self.bound_inputs: tuple | None = None
        self.host_memory_bytes = int(g_blocks.nbytes)

    def _sweeps(self) -> None:
        m = self.m
        dev = self.device
        # the CPU device runs one-thread blocks: one lane per output row
        dim = m * (THOMAS_LANES if dev.is_cuda else 1)
        for i in range(self.nr + 1):
            wp.launch(block_forward_kernel, dim=dim, inputs=[self.g_blocks, self.coupling, self.rhs, self.y, i, m],
                      device=dev, block_dim=THOMAS_LANES)
        for i in range(self.nr, -1, -1):
            wp.launch(block_backward_kernel, dim=dim, inputs=[self.g_blocks, self.coupling, self.y, self.x, i, m, self.nr],
                      device=dev, block_dim=THOMAS_LANES)

    def _solve_sequence(self, q_e, q_i, surface, phi_out) -> None:
        n = self.node_count
        dev = self.device
        wp.launch(source_kernel, dim=n, inputs=[q_e, q_i, self.ratio, surface, self.offset, self.unknown, self.rhs], device=dev)
        self._sweeps()
        wp.launch(apply_dirichlet_kernel, dim=n, inputs=[self.x, self.boundary, self.unknown, phi_out], device=dev)

    def bind(self, q_e, q_i, surface, phi_out) -> None:
        """Fix the input/output arrays and capture the whole solve as one graph."""

        self.bound_inputs = (q_e, q_i, surface, phi_out)
        self.graph = None
        if self.use_graph:
            self._solve_sequence(q_e, q_i, surface, phi_out)  # loads the module before capture
            wp.synchronize_device(self.device)
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
        return 1, float("nan"), float("nan")

    def queue_residual_check(self) -> None:
        """Launch the true-residual and rhs-norm reductions for the last solve (read via ``verify``)."""

        n = self.node_count
        dev = self.device
        wp.launch(matvec_kernel, dim=n, inputs=[self.x, self.unknown, self.cond_r, self.cond_z, self.nr, self.nz, self.ax], device=dev)
        wp.launch(residual_kernel, dim=n, inputs=[self.rhs, self.ax, self.inv_diag, self.r, self.z, self.p], device=dev)
        wp.launch(dot_stride_kernel, dim=self.threads, inputs=[self.r, self.r, n, self.threads, self.partial_a], device=dev)
        wp.launch(reduce_stage_kernel, dim=self.groups, inputs=[self.partial_a, self.threads, REDUCTION_GROUP, self.stage_a], device=dev)
        wp.launch(final_sum_kernel, dim=1, inputs=[self.stage_a, self.groups, self.scalars, 5], device=dev)
        wp.launch(dot_stride_kernel, dim=self.threads, inputs=[self.rhs, self.rhs, n, self.threads, self.partial_b], device=dev)
        wp.launch(reduce_stage_kernel, dim=self.groups, inputs=[self.partial_b, self.threads, REDUCTION_GROUP, self.stage_b], device=dev)
        wp.launch(final_sum_kernel, dim=1, inputs=[self.stage_b, self.groups, self.scalars, 6], device=dev)

    def verify(self) -> tuple[float, float]:
        """Read the queued residual check; raise if the direct solve broke its contract."""

        scalars = self.scalars.numpy()
        true_residual = sqrt(max(float(scalars[5]), 0.0))
        rhs_norm = sqrt(max(float(scalars[6]), 0.0))
        tolerance = max(self.config.absolute_tolerance, self.config.relative_tolerance * rhs_norm)
        if not isfinite(true_residual) or true_residual > tolerance:
            raise PIC2DConvergenceError(
                f"device block-Thomas solve failed its residual contract: true residual {true_residual:.3e} > {tolerance:.3e}"
            )
        return true_residual, tolerance


@dataclass
class DeviceSpecies:
    """Device particle arrays.

    ``capacity`` is the allocated length; ``bound`` is a host-side upper bound on
    the slots in use (the launch dimension); the exact slot count lives on the
    device in ``WarpBackend.slots`` and is only read back at a host sync.
    ``alive`` is the host ledger of live particles, exact at every sync.
    """

    capacity: int
    bound: int
    alive: int
    r: Any
    z: Any
    vr: Any
    vt: Any
    vz: Any
    alive_flags: Any


# Per-interval device statistics (float64; counts are exact integers in binary64).
def birth_bound(candidates: int, probability: float) -> int:
    """Upper bound on ionisations in one step: mean + 8 sigma of Binomial(n, P) + slack.

    Exceeding it is detected fail-closed at the next sync (device slot count
    above the host bound, or the overflow flag), so this only sets how much
    head-room the launch dimension and the allocations carry.
    """

    mean = candidates * probability
    return int(mean + 8.0 * sqrt(max(mean, 1.0)) + 64.0)


class WarpBackend:
    """Warp implementation of the PIC-MCC cycle; interface matches ``CPUBackend``.

    The time step is entirely on the device: particle slot counts, boundary
    tallies, ledger energies and the peak-density gate live in device arrays
    and are read back once every ``config.device_sync_steps`` steps (plus one
    scalar convergence read per Poisson solve when the device PCG is used).
    Compaction, ledger updates and the stability gate happen at those syncs;
    ``step`` returns a ``StepTally`` only on sync steps.
    """

    def __init__(
        self,
        config: PIC2DConfig,
        masks: MeshMasks,
        field: MagneticFieldMap,
        cross_sections: XenonCrossSections | None,
        *,
        device: str = "cuda:0",
        use_graph: bool = True,
        step_graph: bool = True,
    ) -> None:
        if wp is None:
            raise PIC2DDeviceError("NVIDIA Warp is unavailable")
        if not config.fixed_point_deposition:
            raise PIC2DValidationError("the Warp backend requires fixed-point deposition")
        self.config = config
        self.masks = masks
        self.field = field
        self.device = resolve_device(device)
        self.name = f"warp-{self.device}"
        self.electron = electron_species(config.macro_weight)
        self.ion = xenon_ion_species(config.macro_weight)
        self.mcc: NullCollisionMCC | None = None
        self.ion_mcc: IonNullCollisionMCC | None = None
        if config.mcc is not None:
            if cross_sections is None:
                raise PIC2DValidationError("MCC requires cross sections")
            self.mcc = NullCollisionMCC(cross_sections, config.mcc, self.ion)
            self.ion_mcc = build_ion_mcc(config, self.ion, masks)      # v2.3.0: None unless the collision set declares it
        self.sync_interval = int(config.sync_steps)
        grid = masks.grid
        self.nr, self.nz = grid.cell_shape
        self.node_count = int(np.prod(grid.node_shape))
        dev = self.device
        f64 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), dtype=wp.float64, device=dev)  # noqa: E731
        i32 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.int32).ravel(), dtype=wp.int32, device=dev)  # noqa: E731
        # Field solve: ``pcg`` runs the device Jacobi-PCG warm-started from the
        # previous potential (one scalar convergence read per solve); ``direct``
        # uses the exact host block-Thomas factorisation (two node-array copies
        # per step) and is identical to the CPU backend.
        self.gpu_poisson: WarpPoisson | None = None
        self.device_direct: WarpBlockThomas | None = None
        self.host_poisson: Poisson2D | None = None
        self.ratio = f64(masks.charge_to_source)
        self.use_graph = bool(use_graph)
        if config.poisson.method == "pcg":
            self.gpu_poisson = WarpPoisson(masks, config.potentials, config.poisson, dev, use_graph=use_graph)
        elif config.poisson.method == "device-direct":
            self.device_direct = WarpBlockThomas(masks, config.potentials, config.poisson, dev, use_graph=use_graph)
        elif config.poisson.method == "device-mg":
            # poisson_gmg_v1: fixed-cycle geometric multigrid with the WarpBlockThomas interface
            # (bind / solve_sequence / queue_residual_check / verify), so every downstream use of
            # ``device_direct`` (step graph, residual check at the host sync, checkpoint binding) is shared
            from .warp_poisson_mg import WarpPoissonMG

            self.device_direct = WarpPoissonMG(masks, config.potentials, config.poisson, dev, use_graph=use_graph)  # type: ignore[assignment]
        else:
            self.host_poisson = Poisson2D(masks, config.poisson)
            self.source_dev = wp.zeros(self.node_count, dtype=wp.float64, device=dev)
        code_r, code_z = efield_stencil_codes(masks)
        self.code_r = i32(code_r)
        self.code_z = i32(code_z)
        self.b_r = f64(field.b_r_t)
        self.b_z = f64(field.b_z_t)
        self.plasma_cell = i32(masks.plasma_cell)
        self.plasma_node = i32(masks.plasma_node)
        self.top_cell = i32(masks.top_plasma_cell)
        inverse_volume = np.zeros(grid.node_shape)
        inverse_volume[masks.plasma_node] = 1.0 / (ELEMENTARY_CHARGE_C * masks.shape_volume_m3[masks.plasma_node])
        self.inverse_volume = f64(inverse_volume)
        # v2.0.4: occupancy floor of the runtime omega_pe dt statistic as a deposited charge (a run constant: safe inside the CUDA graph)
        self.omega_pe_gate_min_charge_c = float(omega_pe_gate_min_macro_particles(config) * ELEMENTARY_CHARGE_C * config.macro_weight)
        zeros = lambda dtype=wp.float64: wp.zeros(self.node_count, dtype=dtype, device=dev)  # noqa: E731
        self.acc_e = zeros(wp.int64)
        self.acc_i = zeros(wp.int64)
        self.acc_wall = zeros(wp.int64)
        self.q_e, self.q_i, self.surface, self.phi, self.e_r, self.e_z = (zeros() for _ in range(6))
        self.stats = wp.zeros(STATS_SIZE, dtype=wp.float64, device=dev)
        self.slots = wp.zeros(2, dtype=wp.int32, device=dev)
        self.sample_out = wp.zeros(5, dtype=wp.float64, device=dev)
        # v1.4: device-side per-step state so that a whole step has fixed kernel arguments
        # (graph-capturable): seed table for the sync interval, steps-since-sync counter,
        # injection control [carry, count].
        self.seed_table = wp.zeros(SEED_STREAMS * self.sync_interval, dtype=wp.int32, device=dev)
        self.step_counter = wp.zeros(1, dtype=wp.int32, device=dev)
        self.inject_ctrl = wp.zeros(3, dtype=wp.float64, device=dev)     # [carry, count this step, rate per step]
        # emission: legacy exit-plane injection or the v2.0 cathode region; the rate is device-resident
        # (``inject_ctrl[2]``) and, under the continuity rule, updated by the driver at series records
        self.emission_rate_per_step = config.initial_emission_rate_per_step
        self.emission_enabled = config.emission_peak_current_a > 0.0
        peak_rate = config.emission_peak_current_a * config.dt_s / (ELEMENTARY_CHARGE_C * config.macro_weight)
        self.max_inject_per_step = int(np.floor(peak_rate)) + 1
        # v2.0 plume block: two-zone neutral shape per cell and the far-field/plume histograms
        self.has_plume = bool(masks.has_plume)
        self.neutral_shape = f64(neutral_shape_cells(masks))
        self.d_side_e = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_side_i = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_theta_i = wp.zeros(THETA_BINS, dtype=wp.float64, device=dev)
        self.d_iedf_i = wp.zeros(IEDF_BINS, dtype=wp.float64, device=dev)
        self.iedf_max_ev = iedf_max_ev(config)
        # CUDA-graph capture of the whole step (v1.4): one graph per step variant
        # (ion push?, ion redeposit?, accumulate?) and per particle-array allocation.
        self.step_graph = bool(step_graph) and self.device.is_cuda and config.poisson.method in ("device-direct", "device-mg")
        self.step_graph_active = False
        self.step_graphs: dict[tuple, Any] = {}
        self.graph_captures = 0
        # v2.0.5: inside the captured step the window-diagnostic branch (density accumulators + electron moment
        # deposition, which read the pre-push state and nothing the field solve writes) is forked onto a second
        # stream right after the charge deposits and joined before the push, so its float atomics overlap the
        # latency-bound block-Thomas chain instead of extending it (CUDA graph fork/join through events).
        self.side_stream = wp.Stream(self.device) if self.device.is_cuda else None
        self.fork_event = wp.Event(self.device) if self.device.is_cuda else None
        self.join_event = wp.Event(self.device) if self.device.is_cuda else None
        self.diagnostic_forks = 0
        # v2.0.5: the electron moment deposition (20 float64 atomics per electron) is sampled every
        # ``moment_sample_interval`` accumulated steps; ``moment_samples`` counts the samples of the window
        self.moment_sample_interval = int(config.moment_sample_interval)
        self.moment_samples = 0
        # diagnostics (device)
        self.d_n_e, self.d_n_i, self.d_phi, self.d_w, self.d_vr, self.d_vt, self.d_vz, self.d_v2, self.d_ion = (zeros() for _ in range(9))
        self.d_wall_e = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_wall_i = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_wall_e_energy = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_wall_i_energy = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_exit_e = wp.zeros(self.nr, dtype=wp.float64, device=dev)
        self.d_exit_i = wp.zeros(self.nr, dtype=wp.float64, device=dev)
        # v2.2.0 SEE stage (warp_see.py): device constants, per-column emitted count / energy window sums; the per-slot
        # flag / count / offset scratch is allocated with the particle scratch (_ensure_scratch)
        self.see_active = bool(config.see_active)
        self.see_ion_induced = self.see_active and config.see is not None and config.see.ion_induced_yield > 0.0
        self.d_see_e = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_see_energy = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.see_params = None
        if self.see_active:
            from .warp_see import see_params_array

            body = grid.geometry.body_dielectric_radius_m if masks.has_plume else None
            self.see_params = see_params_array(config.see, body, dev)  # type: ignore[arg-type]
        self.diag_steps = 0
        self.diagnostic_generation = 0     # v2.0.2: incremented by every reset_diagnostics (window bridging)
        self._far_flat = np.flatnonzero(masks.far_field_node.ravel())   # v2.0.2: far-field node rows of the window sums
        self.species: dict[str, DeviceSpecies] = {}
        self.scratch_capacity = 0
        self.state_meta: dict[str, Any] = {}
        if self.mcc is not None:
            self.table = f64(self.mcc.table.table_m2)
            self.table_points = self.mcc.table.point_count
            self.probability = self.mcc.collision_probability(config.dt_s)
            # device-resident instantaneous neutral density (graph-safe; see set_neutral_scale)
            self.neutral_density_ctrl = wp.array(np.array([self.mcc.neutral_density_per_m3]), dtype=wp.float64, device=dev)
            # v2.3.0: excitation-level thresholds (one for the legacy set) and the ion-neutral table
            self.exc_thresholds = f64(np.asarray(self.mcc.table.excitation_thresholds_ev))
            self.exc_count = self.mcc.table.excitation_count
            if self.ion_mcc is not None:
                self.ion_table = f64(self.ion_mcc.table.table_m2)
                self.ion_table_points = self.ion_mcc.table.point_count
                self.ion_probability = self.ion_mcc.collision_probability(config.dt_s * config.ion_subcycle)
                self.ion_march_limit = 4 * (self.nr + self.nz) + 8
        else:
            self.probability = 0.0
            self.neutral_density_ctrl = wp.zeros(1, dtype=wp.float64, device=dev)
        self.sync_count = 0
        self.steps_since_sync = 0
        self.injected_since_sync = 0
        self.ions_dirty = True
        self.last_iterations = 0
        self.last_tally: StepTally | None = None
        # Optional phase profiler: when ``profile`` is a dict, every phase boundary
        # synchronises the device and accumulates wall seconds per phase label.
        self.profile: dict[str, float] | None = None
        self._profile_clock = 0.0

    def _mark(self, label: str) -> None:
        if self.profile is None:
            return
        wp.synchronize_device(self.device)
        now = time.perf_counter()
        self.profile[label] = self.profile.get(label, 0.0) + (now - self._profile_clock)
        self._profile_clock = now

    # ------------------------------------------------------------------ helpers
    @property
    def step_index(self) -> int:
        return int(self.state_meta["step"])

    @property
    def time_s(self) -> float:
        return float(self.state_meta["time_s"])

    def _alloc_species(self, capacity: int) -> DeviceSpecies:
        dev = self.device
        arrays = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(5)]
        alive = wp.zeros(capacity, dtype=wp.int32, device=dev)
        return DeviceSpecies(capacity, 0, 0, *arrays, alive)

    def _ensure_scratch(self, capacity: int) -> None:
        if capacity <= self.scratch_capacity:
            return
        dev = self.device
        self.scratch_capacity = capacity
        self.step_graphs.clear()     # the scratch arrays are captured by the step graphs
        self.ionize = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.offsets = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.sec = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(3)]
        self.ionv = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(3)]
        self.tmp = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(5)]
        self.tmp_alive = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.partial_particles = wp.zeros(REDUCTION_THREADS, dtype=wp.float64, device=dev)
        # v2.2.0: wall-impact flags per species slot (written by push_kernel, consumed by the SEE stage; all zero between
        # steps), integer yield per slot and its exclusive scan (int32 x capacity each)
        self.see_hit_e = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.see_hit_i = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.see_count = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.see_offsets = wp.zeros(capacity, dtype=wp.int32, device=dev)

    def _grow(self, species: DeviceSpecies, minimum: int) -> DeviceSpecies:
        """Reallocate (at a sync) so that ``capacity >= minimum``; copies ``bound`` slots."""

        if minimum <= species.capacity:
            return species
        capacity = max(minimum, int(1.5 * species.capacity), 1024)
        self.step_graphs.clear()     # captured step graphs are bound to the old arrays
        new = self._alloc_species(capacity)
        if species.bound:
            for src, dst in zip((species.r, species.z, species.vr, species.vt, species.vz, species.alive_flags),
                                (new.r, new.z, new.vr, new.vt, new.vz, new.alive_flags)):
                wp.copy(dst, src, count=species.bound)
        new.bound = species.bound
        new.alive = species.alive
        self._ensure_scratch(capacity)
        return new

    def _upload(self, particles: ParticleArrays) -> DeviceSpecies:
        capacity = max(2 * particles.count, 1024)
        species = self._alloc_species(capacity)
        if particles.count:
            for values, target in zip(
                (particles.r_m, particles.z_m, particles.vr_m_per_s, particles.vt_m_per_s, particles.vz_m_per_s),
                (species.r, species.z, species.vr, species.vt, species.vz),
            ):
                wp.copy(target, wp.array(values, dtype=wp.float64, device=self.device), count=particles.count)
            wp.copy(species.alive_flags, wp.array(np.ones(particles.count, dtype=np.int32), dtype=wp.int32, device=self.device), count=particles.count)
        species.bound = particles.count
        species.alive = particles.count
        self._ensure_scratch(capacity)
        return species

    def _download(self, species: DeviceSpecies) -> ParticleArrays:
        """Host copy of a compacted species (call only right after a sync)."""

        n = species.alive
        if n == 0:
            return ParticleArrays.empty()
        arrays = [np.asarray(a.numpy()[:n], dtype=np.float64).copy() for a in (species.r, species.z, species.vr, species.vt, species.vz)]
        return ParticleArrays(*arrays)

    def _set_slots(self, electrons: int, ions: int) -> None:
        wp.copy(self.slots, wp.array(np.array([electrons, ions], dtype=np.int32), dtype=wp.int32, device=self.device))

    def _compact(self, species: DeviceSpecies, slot: int, used: int, expected_alive: int) -> None:
        """Compact live particles to the front; fail closed if the device count disagrees."""

        if used == 0:
            if expected_alive != 0:
                raise PIC2DValidationError("host ledger expects live particles but the device has none")
            species.bound = 0
            species.alive = 0
            return
        self._ensure_scratch(species.capacity)
        wp.utils.array_scan(species.alive_flags[:used], self.offsets[:used], inclusive=True)
        total = int(self.offsets.numpy()[used - 1])
        self.sync_count += 1
        if total != expected_alive:
            raise PIC2DValidationError(
                f"device alive count {total} disagrees with the host ledger {expected_alive}"
            )
        wp.utils.array_scan(species.alive_flags[:used], self.offsets[:used], inclusive=False)
        tmp = self.tmp
        self.tmp_alive.zero_()
        wp.launch(compact_kernel, dim=used, inputs=[species.alive_flags, self.offsets, self.slots, slot, species.r, species.z, species.vr,
                                                    species.vt, species.vz, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], self.tmp_alive],
                  device=self.device)
        species.alive_flags.zero_()
        if total:
            for src, dst in zip((tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]), (species.r, species.z, species.vr, species.vt, species.vz)):
                wp.copy(dst, src, count=total)
            wp.copy(species.alive_flags, self.tmp_alive, count=total)
        species.bound = total
        species.alive = total

    def _projected_bound(self, species: DeviceSpecies, is_electron: bool, steps: int) -> int:
        """Slots that ``steps`` more steps can use at most (births + injection)."""

        bound = species.bound
        injected_per_step = 0
        if is_electron and self.emission_enabled:
            injected_per_step = self.max_inject_per_step
        electrons = self.species["e"].bound if "e" in self.species else bound
        for _ in range(steps):
            births = birth_bound(electrons, self.probability) if self.mcc is not None else 0
            # v2.2.0: the wall's secondaries are electrons (fail-closed overflow at the sync if the reservation is exceeded)
            secondaries = see_birth_bound(electrons) if self.see_active else 0
            electrons += births + injected_per_step + secondaries
            bound += births + ((injected_per_step + secondaries) if is_electron else 0)
        return bound

    # ------------------------------------------------------------------ state exchange
    def set_neutral_scale(self, scale: float) -> None:
        """v1.3: the MCC kernel receives ``n_g0 * scale`` as its density at every launch.

        The density lives in a one-element device array read by the kernel, so the
        captured step graph (v1.4) sees every update.  Until 2026-09-04 it was a kernel
        scalar and therefore frozen at the value of the last graph capture (a particle
        array reallocation): plume attempt 4 captured during the inventory trough and
        kept ionising at n_g ~ 7e17 after n_g had refilled to 5.9e19 (S x100 low at an
        unchanged T_e).  The v1.3 records predate the graph and are unaffected.
        """

        if self.mcc is None:
            raise PIC2DValidationError("neutral scale requires MCC")
        self.mcc.set_neutral_scale(scale)
        if self.ion_mcc is not None:
            self.ion_mcc.set_neutral_scale(scale)      # the ion kernel reads the same device-resident density
        wp.copy(self.neutral_density_ctrl, wp.array(np.array([self.mcc.neutral_density_per_m3]), dtype=wp.float64, device=self.device))

    def set_emission_rate(self, rate_per_step: float) -> None:
        """v2.0: cathode emission rate (macro-electrons per step) for the coming steps; device-resident, graph-safe."""

        if not isfinite(rate_per_step) or rate_per_step < 0.0:
            raise PIC2DValidationError("emission rate must be finite and non-negative")
        peak = self.config.emission_peak_current_a * self.config.dt_s / (ELEMENTARY_CHARGE_C * self.config.macro_weight)
        if rate_per_step > peak * (1.0 + 1e-12):
            raise PIC2DValidationError("emission rate exceeds the configured peak current (device injection buffer)")
        self.emission_rate_per_step = float(rate_per_step)
        self.state_meta["cumulative"][CATHODE_RATE_KEY] = self.emission_rate_per_step
        self._push_inject_ctrl()

    def _push_inject_ctrl(self) -> None:
        wp.copy(self.inject_ctrl, wp.array(
            np.array([float(self.state_meta["injection_carry"]), 0.0, self.emission_rate_per_step]), dtype=wp.float64, device=self.device))

    def charge_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """v2.0: node charge maps from the last deposit (positions at the start of the last step)."""

        self.flush()
        shape = self.masks.grid.node_shape
        self.sync_count += 2
        return self.q_e.numpy().reshape(shape).copy(), self.q_i.numpy().reshape(shape).copy()

    def load_state(self, state: SimulationState) -> None:
        self.step_graphs.clear()
        self.emission_rate_per_step = float(state.cumulative.get(CATHODE_RATE_KEY, self.config.initial_emission_rate_per_step))
        self.species = {"e": self._upload(state.electrons), "i": self._upload(state.ions)}
        self._set_slots(state.electrons.count, state.ions.count)
        # copy into the existing node arrays so captured graphs stay bound to them
        wp.copy(self.surface, wp.array(state.surface_charge_c.ravel(), dtype=wp.float64, device=self.device))
        wp.copy(self.phi, wp.array(state.phi_v.ravel(), dtype=wp.float64, device=self.device))
        if self.device_direct is not None and self.device_direct.bound_inputs is None:
            self.device_direct.bind(self.q_e, self.q_i, self.surface, self.phi)
        if self.gpu_poisson is not None:
            x = state.phi_v.copy()
            x[~self.masks.unknown_node] = 0.0
            wp.copy(self.gpu_poisson.x, wp.array(x.ravel(), dtype=wp.float64, device=self.device))
        self.state_meta = {
            "step": int(state.step), "time_s": float(state.time_s),
            "injection_carry": float(state.injection_carry), "cumulative": dict(state.cumulative),
        }
        if self.see_active:
            for key in SEE_KEYS:      # v2.2.0: SEE ledger keys exist from the first record (as the CPU reference)
                self.state_meta["cumulative"].setdefault(key, 0.0)
        self.stats.zero_()
        self.steps_since_sync = 0
        self.injected_since_sync = 0
        self.ions_dirty = True
        self._begin_interval()
        self._reserve_capacity()

    def _begin_interval(self) -> None:
        """Upload the seeds of the next sync interval, reset the device step counter, push the injection carry."""

        base = int(self.state_meta["step"])
        seeds = np.array(
            [stream_seed(self.config.seed, base + k, stream) for k in range(self.sync_interval) for stream in SEED_STREAM_IDS],
            dtype=np.int32,
        )
        wp.copy(self.seed_table, wp.array(seeds, dtype=wp.int32, device=self.device))
        self.step_counter.zero_()
        self._push_inject_ctrl()

    def export_state(self) -> SimulationState:
        self.flush()
        shape = self.masks.grid.node_shape
        return SimulationState(
            self.state_meta["step"], self.state_meta["time_s"],
            self._download(self.species["e"]), self._download(self.species["i"]),
            self.surface.numpy().reshape(shape).copy(), self.phi.numpy().reshape(shape).copy(),
            self.state_meta["injection_carry"], dict(self.state_meta["cumulative"]),
        )

    def flush(self) -> StepTally | None:
        """Force a host sync (ledger, compaction, gate values) if steps are pending."""

        if self.steps_since_sync > 0:
            self._sync()
        return self.last_tally

    # ------------------------------------------------------------------ cycle
    def _deposit(self, species: DeviceSpecies, slot: int, accumulator, out, per_particle: float, *, redo: bool = True,
                 dim: int | None = None) -> None:
        grid = self.masks.grid
        if dim is None:
            dim = species.bound
        if redo:
            accumulator.zero_()
            if dim:
                wp.launch(deposit_fixed_kernel, dim=dim,
                          inputs=[species.r, species.z, species.alive_flags, self.slots, slot, grid.dr_m, grid.dz_m, grid.geometry.z_min_m,
                                  self.nr, self.nz, FIXED_POINT_SCALE, accumulator], device=self.device)
        wp.launch(int_to_charge_kernel, dim=self.node_count, inputs=[accumulator, per_particle / FIXED_POINT_SCALE, out], device=self.device)

    def step(self, accumulate: bool) -> StepTally | None:
        config = self.config
        meta = self.state_meta
        step_index = meta["step"]
        electrons = self.species["e"]
        ions = self.species["i"]
        ion_step = (step_index + 1) % config.ion_subcycle == 0
        redo_ions = self.ions_dirty
        n_bound = electrons.bound
        # v2.0.5: electron moments every K-th accumulated step (K = 1: every step, the v1.4-v2.0.4 behaviour);
        # the phase is anchored on the window's accumulated-step count, so a window reset restarts it
        moments = bool(accumulate) and self.diag_steps % self.moment_sample_interval == 0

        if self.step_graph:
            self._step_graph_launch(ion_step, redo_ions, accumulate, moments)
        else:
            self._launch_step(ion_step, redo_ions, accumulate, moments=moments, fixed_shape=False)
        if self.device_direct is not None and self.steps_since_sync + 1 >= self.sync_interval:
            self.device_direct.queue_residual_check()  # read and enforced in _sync (outside the captured step)

        # host bookkeeping: no device reads; the bounds are upper limits on the slots in use
        self.ions_dirty = ion_step   # ions are frozen between subcycle pushes; a push dirties the ion charge
        if self.mcc is not None and n_bound:
            births = birth_bound(n_bound, self.probability)
            electrons.bound = min(electrons.bound + births, electrons.capacity)
            ions.bound = min(ions.bound + births, ions.capacity)
        if self.emission_enabled:
            electrons.bound = min(electrons.bound + self.max_inject_per_step, electrons.capacity)
        if self.see_active:
            electrons.bound = min(electrons.bound + see_birth_bound(n_bound), electrons.capacity)
        meta["step"] = step_index + 1
        meta["time_s"] = meta["step"] * config.dt_s
        if accumulate:
            self.diag_steps += 1
            if moments:
                self.moment_samples += 1
        self.steps_since_sync += 1
        if self.steps_since_sync >= self.sync_interval:
            return self._sync()
        return None

    def _step_graph_launch(self, ion_step: bool, redo_ions: bool, accumulate: bool, moments: bool) -> None:
        """Replay (capturing on first use) the CUDA graph of this step variant for the current particle arrays."""

        electrons = self.species["e"]
        ions = self.species["i"]
        key = (ion_step, redo_ions, accumulate, moments, electrons.r.ptr, ions.r.ptr, electrons.capacity, ions.capacity)
        graph = self.step_graphs.get(key)
        if graph is None:
            if not self.step_graphs:
                import sys

                wp.load_module(module=sys.modules[__name__], device=self.device)   # no module loads inside a capture
                if self.ion_mcc is not None:
                    wp.load_module(module=warp_ion_mcc, device=self.device)
                if self.see_active:
                    from . import warp_see

                    wp.load_module(module=warp_see, device=self.device)
            profile, self.profile = self.profile, None
            try:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._launch_step(ion_step, redo_ions, accumulate, moments=moments, fixed_shape=True)
            finally:
                self.profile = profile
            graph = capture.graph
            self.step_graphs[key] = graph
            self.graph_captures += 1
            self.step_graph_active = True
        wp.capture_launch(graph)
        self._mark("graph-step")

    def _launch_step(self, ion_step: bool, redo_ions: bool, accumulate: bool, *, moments: bool | None = None, fixed_shape: bool) -> None:
        """Issue every device operation of one step.

        With ``fixed_shape`` the launch dimensions are the array capacities and nothing
        depends on host-side counts (all kernels guard on the device slot counts and
        flags), so the sequence can be captured into a CUDA graph and replayed; without
        it the dimensions are the host upper bounds (v1.0-v1.3 behaviour).  Both paths
        run the same kernels with the same device-side seeds and injection control.

        ``moments`` (v2.0.5) says whether this accumulated step deposits the electron moments
        (``None``: whenever ``accumulate``).  In the captured path the window-diagnostic branch
        runs on the side stream (fork after the charge deposits, join before the push).
        """

        config = self.config
        grid = self.masks.grid
        dev = self.device
        dt = config.dt_s
        electrons = self.species["e"]
        ions = self.species["i"]
        mcc = self.mcc
        e_dim = electrons.capacity if fixed_shape else electrons.bound
        i_dim = ions.capacity if fixed_shape else ions.bound
        if moments is None:
            moments = bool(accumulate)

        self._mark("other")
        self._deposit(electrons, 0, self.acc_e, self.q_e, self.electron.charge_c * config.macro_weight, dim=e_dim)
        # Ions are frozen between subcycle pushes; births are added incrementally
        # (exact integer accumulation), so a full redeposit is only needed after a push.
        self._deposit(ions, 1, self.acc_i, self.q_i, self.ion.charge_c * config.macro_weight, redo=redo_ions, dim=i_dim)
        self._mark("deposit")

        def window_branch() -> None:
            # density accumulators from this step's deposits and (every K-th accumulated step) the electron moments at
            # the pre-push positions / velocities: independent of the field solve, so it can overlap the Poisson chain
            wp.launch(abs_axpy_kernel, dim=self.node_count, inputs=[self.d_n_e, self.inverse_volume, self.q_e], device=dev)
            wp.launch(abs_axpy_kernel, dim=self.node_count, inputs=[self.d_n_i, self.inverse_volume, self.q_i], device=dev)
            if moments and e_dim:
                wp.launch(deposit_moment_kernel, dim=e_dim,
                          inputs=[electrons.r, electrons.z, electrons.alive_flags, self.slots, 0, electrons.vr, electrons.vt, electrons.vz,
                                  grid.dr_m, grid.dz_m, grid.geometry.z_min_m, self.nr, self.nz,
                                  self.d_w, self.d_vr, self.d_vt, self.d_vz, self.d_v2], device=dev)

        forked = False
        if accumulate:
            if fixed_shape and self.side_stream is not None:
                # fork: the side stream waits for everything issued so far on the capture stream (the deposits)
                main_stream = wp.get_stream(dev)
                self.side_stream.wait_stream(main_stream, self.fork_event)
                with wp.ScopedStream(self.side_stream, sync_enter=False, sync_exit=False):
                    window_branch()
                forked = True
                self.diagnostic_forks += 1
            else:
                window_branch()
                self._mark("window")
        if self.gpu_poisson is not None:
            if fixed_shape:
                raise PIC2DValidationError("the PCG field solve has a host convergence loop and cannot be graph-captured")
            iterations, _, _ = self.gpu_poisson.solve(self.q_e, self.q_i, self.surface, self.phi)
        elif self.device_direct is not None:
            if fixed_shape:
                self.device_direct.solve_sequence(self.q_e, self.q_i, self.surface, self.phi)   # raw launches inside the capture
            else:
                self.device_direct.solve(self.q_e, self.q_i, self.surface, self.phi)
            iterations = 0
        else:
            if fixed_shape:
                raise PIC2DValidationError("the host field solve cannot be graph-captured")
            wp.launch(host_source_kernel, dim=self.node_count, inputs=[self.q_e, self.q_i, self.ratio, self.surface, self.source_dev], device=dev)
            source = self.source_dev.numpy().reshape(grid.node_shape)
            result = self.host_poisson.solve(source, config.potentials)  # type: ignore[union-attr]
            iterations = result.diagnostics.iterations
            wp.copy(self.phi, wp.array(result.phi_v.ravel(), dtype=wp.float64, device=dev))
            self.sync_count += 1
        self.last_iterations = iterations
        self._mark("poisson")
        wp.launch(efield_kernel, dim=self.node_count, inputs=[self.phi, self.code_r, self.code_z, grid.dr_m, grid.dz_m, self.nz, self.e_r, self.e_z], device=dev)
        wp.launch(peak_density_kernel, dim=self.node_count, inputs=[self.q_e, self.inverse_volume, self.omega_pe_gate_min_charge_c, self.stats, STATS_PEAK_DENSITY, STATS_PEAK_DENSITY_RESOLVED],
                  device=dev)

        if accumulate:
            wp.launch(axpy_kernel, dim=self.node_count, inputs=[self.d_phi, 1.0, self.phi], device=dev)
        if forked:
            # join: the push overwrites the positions / velocities the moment deposition reads
            wp.get_stream(dev).wait_stream(self.side_stream, self.join_event)
        self._mark("field+diag")

        geometry = grid.geometry
        for slot, (species, particles, dim) in enumerate(((self.electron, electrons, e_dim), (self.ion, ions, i_dim))):
            is_electron = slot == 0
            if dim == 0 or (not is_electron and not ion_step):
                continue
            species_dt = dt if is_electron else dt * config.ion_subcycle
            wp.launch(
                push_kernel, dim=padded_dim(dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
                inputs=[particles.r, particles.z, particles.vr, particles.vt, particles.vz, particles.alive_flags,
                        self.e_r, self.e_z, self.b_r, self.b_z, grid.dr_m, grid.dz_m, geometry.z_min_m, geometry.domain_z_max_m,
                        self.nr, self.nz, self.plasma_cell, self.top_cell, self.plasma_node, self.slots, slot,
                        species.charge_c, species.mass_kg, species.macro_weight, species_dt, FIXED_POINT_SCALE,
                        wp.int64(1 if species.charge_c > 0 else -1), self.acc_wall,
                        self.stats, STATS_E_COUNTS if is_electron else STATS_I_COUNTS,
                        STATS_E_ENERGY if is_electron else STATS_I_ENERGY, 1 if accumulate else 0,
                        self.d_wall_e if is_electron else self.d_wall_i,
                        self.d_wall_e_energy if is_electron else self.d_wall_i_energy,
                        self.d_exit_e if is_electron else self.d_exit_i,
                        1 if self.has_plume else 0, geometry.z_max_m, geometry.max_radius_m, geometry.exit_radius_m,
                        STATS_E_PZ if is_electron else STATS_I_PZ, STATS_BODY_FACE_E if is_electron else STATS_BODY_FACE_I,
                        0 if is_electron else 1, self.d_side_e if is_electron else self.d_side_i, self.d_theta_i, self.d_iedf_i,
                        IEDF_BINS / self.iedf_max_ev, THETA_BINS / 90.0,
                        self.see_hit_e if is_electron else self.see_hit_i],
                device=dev,
            )
        wp.launch(wall_int_to_charge_kernel, dim=self.node_count,
                  inputs=[self.acc_wall, ELEMENTARY_CHARGE_C * config.macro_weight / FIXED_POINT_SCALE, self.surface], device=dev)
        self._mark("push")

        if self.ion_mcc is not None and ion_step and i_dim:
            # v2.3.0: Xe+ - Xe CEX / MEX on the pushed ions (velocities only: the frozen ion charge stays valid), before
            # this step's births join (as the CPU reference); RNG stream 3 of the seed table; n_g device-resident
            ion_mcc = self.ion_mcc
            wp.launch(
                warp_ion_mcc.ion_mcc_kernel, dim=padded_dim(i_dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
                inputs=[ions.r, ions.z, ions.vr, ions.vt, ions.vz, ions.alive_flags, self.slots, 1,
                        self.seed_table, SEED_STREAMS, 3, self.step_counter,
                        self.ion_probability, ion_mcc.nu_max, self.neutral_density_ctrl,
                        self.ion_table, self.ion_table_points, ion_mcc.table.energy_step_ev, ion_mcc.table.energy_max_ev,
                        self.ion.mass_kg, self.ion.mass_kg * config.macro_weight, ion_mcc.thermal_speed, ion_mcc.fast_speed_threshold,
                        self.neutral_shape, self.plasma_cell, 1 if self.has_plume else 0,
                        grid.dr_m, grid.dz_m, geometry.z_min_m, geometry.z_max_m, geometry.exit_radius_m, self.nr, self.nz, self.ion_march_limit,
                        self.stats, STATS_ION_MCC],
                device=dev,
            )
            self._mark("ion-mcc")

        if config.anomalous is not None and e_dim:
            # v1.4 hook: Bohm-type scattering after the push, before the MCC (as the CPU reference)
            wp.launch(
                bohm_kernel, dim=padded_dim(e_dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
                inputs=[electrons.r, electrons.z, electrons.vr, electrons.vt, electrons.vz, electrons.alive_flags, self.slots,
                        self.seed_table, self.step_counter, self.b_r, self.b_z, grid.dr_m, grid.dz_m, grid.geometry.z_min_m, self.nr, self.nz,
                        config.anomalous.alpha * dt * ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG, self.stats, STATS_ANOMALOUS,
                        ELECTRON_MASS_KG * config.macro_weight, 1 if config.anomalous.rotation else 0],
                device=dev,
            )

        if mcc is not None and e_dim:
            ion_thermal = sqrt(1.380649e-23 * config.mcc.neutral_temperature_k / self.ion.mass_kg)  # type: ignore[union-attr]
            wp.launch(
                mcc_kernel, dim=padded_dim(e_dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
                inputs=[electrons.vr, electrons.vt, electrons.vz, electrons.alive_flags, self.slots, self.seed_table, self.step_counter,
                        self.probability, mcc.nu_max, self.neutral_density_ctrl,  # n_g0 x scale, device-resident (graph-safe); ceiling fixed
                        self.table, self.table_points, mcc.table.energy_step_ev, mcc.table.energy_max_ev,
                        self.exc_thresholds, self.exc_count, STATS_EXC_LEVELS, mcc.table.thresholds_ev[-1], 8.7, ion_thermal,
                        self.ionize, self.sec[0], self.sec[1], self.sec[2], self.ionv[0], self.ionv[1], self.ionv[2],
                        self.stats, STATS_MCC, e_dim,
                        electrons.r, electrons.z, self.neutral_shape, 1 if self.has_plume else 0, geometry.z_max_m,
                        grid.dr_m, grid.dz_m, geometry.z_min_m, self.nr, self.nz, ELECTRON_MASS_KG * config.macro_weight,
                        # v2.0.5: born-ledger tallies (ke_born_ions_j, pz_born) tile-reduced inside the MCC kernel
                        self.ion.mass_kg * config.macro_weight, STATS_KE_BORN, STATS_PZ_BORN],
                device=dev,
            )
            self._mark("mcc")
            wp.utils.array_scan(self.ionize[:e_dim], self.offsets[:e_dim], inclusive=False)
            # v2.0.5: the spawn also deposits the born ion into the frozen ion charge (exact int64 add, so q_i is
            # bitwise the former separate deposit) and into the ionisation-rate window map; the three strided
            # born-ledger sums + single-thread adds and the two born flag passes of v1.x-v2.0.4 are gone
            wp.launch(
                spawn_kernel, dim=e_dim,
                inputs=[self.ionize, self.offsets, self.slots, electrons.capacity, ions.capacity, electrons.r, electrons.z,
                        self.sec[0], self.sec[1], self.sec[2], self.ionv[0], self.ionv[1], self.ionv[2],
                        electrons.r, electrons.z, electrons.vr, electrons.vt, electrons.vz, electrons.alive_flags,
                        ions.r, ions.z, ions.vr, ions.vt, ions.vz, ions.alive_flags,
                        self.stats, STATS_OVERFLOW,
                        grid.dr_m, grid.dz_m, geometry.z_min_m, self.nr, self.nz, FIXED_POINT_SCALE,
                        self.acc_i, 1 if accumulate else 0, self.d_ion],
                device=dev,
            )
            wp.launch(spawn_commit_kernel, dim=1, inputs=[self.ionize, self.offsets, self.slots, electrons.capacity, ions.capacity], device=dev)
        self._mark("spawn")

        if self.emission_enabled:
            # v1.4: injection count and carry live on the device (fixed launch shape); v2.0: so does the rate
            r_max = grid.r_m[self.masks.top_plasma_cell[self.nz - 1] + 1]
            thermal = sqrt(EV_J * config.emission_temperature_ev / ELECTRON_MASS_KG)
            cathode = config.cathode
            if cathode is None:
                mode, r_in2, r_span2, z_start, z_span = 0, 0.0, 0.0, 0.0, 0.0
            else:
                mode = 1
                r_in2 = cathode.r_inner_m**2
                r_span2 = cathode.r_outer_m**2 - cathode.r_inner_m**2
                z_start, z_span = cathode.z_start_m, cathode.z_end_m - cathode.z_start_m
            wp.launch(inject_control_kernel, dim=1, inputs=[self.inject_ctrl, self.stats, STATS_INJECTED], device=dev)
            wp.launch(inject_kernel, dim=self.max_inject_per_step,
                      inputs=[self.seed_table, self.step_counter, self.inject_ctrl, self.slots, electrons.capacity, float(r_max),
                              geometry.domain_z_max_m, grid.dz_m, thermal, electrons.r, electrons.z, electrons.vr, electrons.vt, electrons.vz,
                              electrons.alive_flags, self.stats, STATS_KE_INJECTED, ELECTRON_MASS_KG * config.macro_weight, STATS_OVERFLOW,
                              mode, r_in2, r_span2, z_start, z_span, STATS_PZ_INJECTED],
                      device=dev)
            wp.launch(add_injected_slots_kernel, dim=1, inputs=[self.slots, self.inject_ctrl], device=dev)
            wp.launch(carry_kernel, dim=1, inputs=[self.inject_ctrl, self.stats, STATS_CARRY], device=dev)
        if self.see_active:
            # v2.2.0: the wall's secondaries join last (as the CPU reference appends them after MCC and injection); the
            # stage reads the pre-push arrays of the flagged dead slots (untouched by every kernel above) and this
            # step's e_r / e_z, then the emitted +n e W units join the surface charge (the accumulator was zeroed after
            # the push conversion, so this second conversion adds exactly the emission)
            from .warp_see import launch_see_stage

            launch_see_stage(self, electrons, 0, e_dim, is_ion=False, accumulate=accumulate, species_dt=dt)
            if self.see_ion_induced and ion_step and i_dim:
                launch_see_stage(self, ions, 1, i_dim, is_ion=True, accumulate=accumulate, species_dt=dt * config.ion_subcycle)
            wp.launch(wall_int_to_charge_kernel, dim=self.node_count,
                      inputs=[self.acc_wall, ELEMENTARY_CHARGE_C * config.macro_weight / FIXED_POINT_SCALE, self.surface], device=dev)
            self._mark("see")
        wp.launch(tick_kernel, dim=1, inputs=[self.step_counter], device=dev)
        self._mark("inject")

    def _sync(self) -> StepTally:
        """Read the interval statistics once; update the ledger; compact; gate values."""

        config = self.config
        electrons = self.species["e"]
        ions = self.species["i"]
        stats = self.stats.numpy()
        slots = self.slots.numpy()
        self.sync_count += 2
        if self.device_direct is not None and self.steps_since_sync >= self.sync_interval:
            self.device_direct.verify()
            self.sync_count += 1
        if stats[STATS_OVERFLOW] != 0.0:
            raise PIC2DValidationError("device particle arrays overflowed their reserved capacity")
        if stats[STATS_INVALID] != 0.0:
            raise PIC2DStabilityError("a particle crossed more than one cell in a step (Courant violation)")
        if int(slots[0]) > electrons.bound or int(slots[1]) > ions.bound:
            raise PIC2DValidationError(
                f"device slot counts {int(slots[0])}/{int(slots[1])} exceed the host bounds {electrons.bound}/{ions.bound}"
            )
        cumulative = self.state_meta["cumulative"]
        if self.emission_enabled:
            # v1.4: the injection count and carry are device-side (fixed-shape step)
            self.injected_since_sync = int(stats[STATS_INJECTED])
            self.state_meta["injection_carry"] = float(stats[STATS_CARRY])
        if config.anomalous is not None:
            cumulative["anomalous"] = cumulative.get("anomalous", 0.0) + float(stats[STATS_ANOMALOUS])
        add = lambda key, value: cumulative.__setitem__(key, cumulative.get(key, 0.0) + float(value))  # noqa: E731
        for base, pz_base, label in ((STATS_E_COUNTS, STATS_E_PZ, "electrons"), (STATS_I_COUNTS, STATS_I_PZ, "ions")):
            cumulative[f"anode_{label}"] += float(stats[base])
            cumulative[f"exit_{label}"] += float(stats[base + 1])
            cumulative[f"wall_{label}"] += float(stats[base + 2])
            # v2.0 momentum ledger (same extra keys as the CPU reference)
            add(f"pz_anode_{label}", stats[pz_base])
            add(f"pz_exit_{label}", stats[pz_base + 1])
            add(f"pz_wall_{label}", stats[pz_base + 2])
        add("pz_impulse", stats[STATS_PZ_IMPULSE])
        add("pz_impulse_electric", stats[STATS_PZ_ELECTRIC])
        if self.mcc is not None or config.anomalous is not None:
            add("pz_collisions", stats[STATS_PZ_COLLISIONS])
        if self.mcc is not None:
            add("pz_born", stats[STATS_PZ_BORN])
        if self.emission_enabled:
            add("pz_injected", stats[STATS_PZ_INJECTED])
        if self.has_plume:
            add("body_face_electrons", stats[STATS_BODY_FACE_E])
            add("body_face_ions", stats[STATS_BODY_FACE_I])
            if self.mcc is not None:
                add("ionizations_plume", stats[STATS_ION_PLUME])
        for offset, key in ((0, "ke_absorbed_anode_j"), (1, "ke_absorbed_exit_j"), (2, "ke_absorbed_wall_j")):
            cumulative[key] += float(stats[STATS_E_ENERGY + offset] + stats[STATS_I_ENERGY + offset])
        cumulative["field_work_j"] += float(stats[STATS_WORK])
        cumulative["ke_injected_j"] += float(stats[STATS_KE_INJECTED])
        cumulative["ke_born_ions_j"] += float(stats[STATS_KE_BORN])
        cumulative["injected_electrons"] += float(self.injected_since_sync)
        n_ion = int(stats[STATS_MCC + 3])
        if self.mcc is not None:
            n_exc = float(stats[STATS_MCC + 2])
            cumulative["elastic"] += float(stats[STATS_MCC + 1])
            cumulative["excitations"] += n_exc
            cumulative["ionizations"] += float(n_ion)
            # v2.0.6: the MCC counts are macro events; the ledger is real energy -> times W (the unscaled sum is kept).
            # v2.3.0: sum over the excitation levels of count x threshold (one level: the v2.0.6 expression bitwise)
            thresholds = self.mcc.table.thresholds_ev
            if self.exc_count == 1:
                loss_ev = n_exc * thresholds[1]
            else:
                levels = [float(stats[STATS_EXC_LEVELS + k]) for k in range(self.exc_count)]
                if sum(levels) != n_exc:
                    raise PIC2DValidationError(f"per-level excitation counts {levels} do not sum to the excitation tally {n_exc}")
                loss_ev = 0.0
                for k, count in enumerate(levels):
                    loss_ev += count * thresholds[1 + k]
                    add(f"excitations_level_{k + 1}", count)
            per_weight = (loss_ev + n_ion * thresholds[-1]) * EV_J
            cumulative["inelastic_loss_j"] += per_weight * config.macro_weight
            add(INELASTIC_LOSS_PER_WEIGHT_KEY, per_weight)
        if self.ion_mcc is not None:
            # v2.3.0: ion-neutral tallies (counts exact; energy / momentum sums are float atomics)
            for offset, key in enumerate(warp_ion_mcc.ION_STATS_KEYS):
                add(key, stats[STATS_ION_MCC + offset])
        n_see = 0
        if self.see_active:
            # v2.2.0: SEE ledger (extra keys, as the CPU reference); the emitted electrons enter the particle-count identity
            n_see = int(stats[STATS_SEE_EMITTED] + stats[STATS_SEE_ION_EMITTED])
            add("see_impacts", stats[STATS_SEE_IMPACTS])
            add("see_electrons", stats[STATS_SEE_EMITTED])
            add("see_ion_induced_electrons", stats[STATS_SEE_ION_EMITTED])
            add("see_backscattered", stats[STATS_SEE_BACKSCATTERED])
            add("see_yield_sum", stats[STATS_SEE_YIELD_SUM])
            add("see_yield_clamped", stats[STATS_SEE_CLAMPED])
            add("ke_see_emitted_j", stats[STATS_SEE_KE])
            add("pz_see_emitted", stats[STATS_SEE_PZ])
        absorbed_e = int(stats[STATS_E_COUNTS] + stats[STATS_E_COUNTS + 1] + stats[STATS_E_COUNTS + 2])
        absorbed_i = int(stats[STATS_I_COUNTS] + stats[STATS_I_COUNTS + 1] + stats[STATS_I_COUNTS + 2])
        expected_e = electrons.alive + n_ion + self.injected_since_sync - absorbed_e + n_see
        expected_i = ions.alive + n_ion - absorbed_i
        self._compact(electrons, 0, int(slots[0]), expected_e)
        self._compact(ions, 1, int(slots[1]), expected_i)
        self._set_slots(electrons.alive, ions.alive)
        peak_density_raw = float(stats[STATS_PEAK_DENSITY])
        peak_density = float(stats[STATS_PEAK_DENSITY_RESOLVED])
        max_speed2 = float(stats[STATS_MAX_SPEED2])
        self.stats.zero_()
        self.steps_since_sync = 0
        self.injected_since_sync = 0
        self._reserve_capacity()
        self._begin_interval()
        omega_scale = sqrt(ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ELECTRON_MASS_KG)) * config.dt_s
        self.last_tally = StepTally(self.last_iterations, sqrt(peak_density) * omega_scale, sqrt(max_speed2), electrons.alive, ions.alive, sqrt(peak_density_raw) * omega_scale)
        self._mark("sync")
        return self.last_tally

    def _reserve_capacity(self) -> None:
        """Grow the device arrays so the next sync interval cannot overflow."""

        electrons = self.species["e"]
        ions = self.species["i"]
        need_e = self._projected_bound(electrons, True, self.sync_interval)
        need_i = self._projected_bound(ions, False, self.sync_interval)
        self.species["e"] = self._grow(electrons, need_e)
        self.species["i"] = self._grow(ions, need_i)
        self._ensure_scratch(max(self.species["e"].capacity, self.species["i"].capacity))

    # ------------------------------------------------------------------ diagnostics
    def series_sample(self) -> dict[str, Any]:
        """Kinetic energies, potential and surface charge for the time series (at a sync)."""

        self.flush()
        electrons = self.species["e"]
        ions = self.species["i"]
        config = self.config
        dev = self.device
        for slot, species, particles in ((0, self.electron, electrons), (1, self.ion, ions)):
            if particles.alive:
                wp.launch(energy_sum_kernel, dim=REDUCTION_THREADS,
                          inputs=[particles.vr, particles.vt, particles.vz, particles.alive_flags, self.slots, slot, REDUCTION_THREADS,
                                  species.mass_kg * config.macro_weight, self.partial_particles], device=dev)
            else:
                self.partial_particles.zero_()
            wp.launch(final_sum_kernel, dim=1, inputs=[self.partial_particles, REDUCTION_THREADS, self.sample_out, slot], device=dev)
        wp.launch(sum_kernel, dim=REDUCTION_THREADS, inputs=[self.surface, self.node_count, REDUCTION_THREADS, self.partial_particles], device=dev)
        wp.launch(final_sum_kernel, dim=1, inputs=[self.partial_particles, REDUCTION_THREADS, self.sample_out, 2], device=dev)
        # v2.0: represented axial momentum per species (momentum ledger)
        for slot, species, particles in ((0, self.electron, electrons), (1, self.ion, ions)):
            if particles.alive:
                wp.launch(momentum_sum_kernel, dim=REDUCTION_THREADS,
                          inputs=[particles.vz, particles.alive_flags, self.slots, slot, REDUCTION_THREADS,
                                  species.mass_kg * config.macro_weight, self.partial_particles], device=dev)
            else:
                self.partial_particles.zero_()
            wp.launch(final_sum_kernel, dim=1, inputs=[self.partial_particles, REDUCTION_THREADS, self.sample_out, 3 + slot], device=dev)
        out = self.sample_out.numpy()
        shape = self.masks.grid.node_shape
        phi = self.phi.numpy().reshape(shape).copy()
        surface = self.surface.numpy().reshape(shape).copy()
        self.sync_count += 3
        return {
            "step": self.state_meta["step"], "time_s": self.state_meta["time_s"],
            "electrons": electrons.alive, "ions": ions.alive,
            "kinetic_electron_j": float(out[0]), "kinetic_ion_j": float(out[1]),
            "momentum_z_electrons": float(out[3]), "momentum_z_ions": float(out[4]),
            "surface_charge_c": float(out[2]), "phi_v": phi, "surface_charge_map_c": surface,
            "cumulative": dict(self.state_meta["cumulative"]),
        }

    def peak_node_sample(self) -> dict[str, Any]:
        """v1.4: one-shot electron node moments at the current positions (peak-node Debye gate)."""

        self.flush()
        electrons = self.species["e"]
        grid = self.masks.grid
        if not hasattr(self, "gate_moments"):
            self.gate_moments = [wp.zeros(self.node_count, dtype=wp.float64, device=self.device) for _ in range(5)]
        for array in self.gate_moments:
            array.zero_()
        if electrons.alive:
            wp.launch(deposit_moment_kernel, dim=electrons.alive,
                      inputs=[electrons.r, electrons.z, electrons.alive_flags, self.slots, 0, electrons.vr, electrons.vt, electrons.vz,
                              grid.dr_m, grid.dz_m, grid.geometry.z_min_m, self.nr, self.nz, *self.gate_moments], device=self.device)
        shape = grid.node_shape
        moments = [array.numpy().reshape(shape) for array in self.gate_moments]
        self.sync_count += 1
        gate = self.config.peak_debye_gate
        return peak_node_debye(self.masks, self.config, *moments, dense_fraction=gate.dense_fraction if gate is not None else 0.5,
                               min_particles=gate.min_macro_particles_at_peak if gate is not None else 16)

    def diagnostic_arrays(self) -> dict[str, np.ndarray]:
        return self._host_accumulator().to_arrays(self.config.macro_weight, self.config.dt_s)

    def diagnostic_sums(self) -> dict[str, np.ndarray]:
        """v2.0 frame recorder: the raw device window sums on the host."""

        return self._host_accumulator().raw_sums()

    def far_field_window_sums(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """v2.0.2 plume-boundary gate: far-field rows of the device window sums ``sum_t n_e``, ``sum_t n_i``, the
        accumulated step count and the reset generation.  Read at the series-record sync (after ``flush``) - two node
        arrays to the host per record, the same accumulation ``diagnostic_sums`` / the frames use; no per-step sync."""

        self.flush()
        far = self._far_flat
        self.sync_count += 2
        return self.d_n_e.numpy()[far].copy(), self.d_n_i.numpy()[far].copy(), int(self.diag_steps), self.diagnostic_generation

    def peak_window_sums(self) -> tuple[dict[str, np.ndarray], int, int]:
        """v2.0.3 peak-Debye gate: the electron window sums (``sum_t n_e``, ``sum_t w``, ``sum_t w v_r/v_theta/v_z``,
        ``sum_t w v^2``) over the whole node map, the accumulated step count and the reset generation - the device
        accumulators behind ``maps.npz`` / the frames, read at the series-record sync (six node arrays per record,
        next to the five single-step gate moments); no per-step sync.  v2.0.5: the dict also carries the additive
        ``moment_samples`` count (the moment sums are over the sampled steps)."""

        self.flush()
        shape = self.masks.grid.node_shape
        arrays = (self.d_n_e, self.d_w, self.d_vr, self.d_vt, self.d_vz, self.d_v2)
        self.sync_count += len(arrays)
        sums = {key: array.numpy().reshape(shape).copy() for key, array in zip(PEAK_WINDOW_SUM_KEYS, arrays)}
        sums["moment_samples"] = np.array([self.moment_samples], dtype=np.int64)
        return sums, int(self.diag_steps), self.diagnostic_generation

    def surface_charge_map(self) -> np.ndarray:
        self.flush()
        return self.surface.numpy().reshape(self.masks.grid.node_shape).copy()

    def _host_accumulator(self) -> DiagnosticAccumulator:
        wp.synchronize_device(self.device)
        shape = self.masks.grid.node_shape
        acc = DiagnosticAccumulator(self.masks, self.iedf_max_ev, see=self.see_active)
        acc.steps = self.diag_steps
        acc.moment_samples = self.moment_samples
        if self.see_active:
            acc.wall_see_electrons = self.d_see_e.numpy()
            acc.wall_see_energy_j = self.d_see_energy.numpy()
        acc.side_electrons = self.d_side_e.numpy()
        acc.side_ions = self.d_side_i.numpy()
        acc.theta_ions = self.d_theta_i.numpy()
        acc.iedf_ions = self.d_iedf_i.numpy()
        acc.n_e = self.d_n_e.numpy().reshape(shape)
        acc.n_i = self.d_n_i.numpy().reshape(shape)
        acc.phi = self.d_phi.numpy().reshape(shape)
        acc.e_weight = self.d_w.numpy().reshape(shape)
        acc.e_vr = self.d_vr.numpy().reshape(shape)
        acc.e_vt = self.d_vt.numpy().reshape(shape)
        acc.e_vz = self.d_vz.numpy().reshape(shape)
        acc.e_v2 = self.d_v2.numpy().reshape(shape)
        acc.ionization = self.d_ion.numpy().reshape(shape)
        acc.wall_electrons = self.d_wall_e.numpy()
        acc.wall_ions = self.d_wall_i.numpy()
        acc.wall_electron_energy_j = self.d_wall_e_energy.numpy()
        acc.wall_ion_energy_j = self.d_wall_i_energy.numpy()
        acc.exit_electrons = self.d_exit_e.numpy()
        acc.exit_ions = self.d_exit_i.numpy()
        return acc

    def reset_diagnostics(self) -> None:
        for array in (self.d_n_e, self.d_n_i, self.d_phi, self.d_w, self.d_vr, self.d_vt, self.d_vz, self.d_v2, self.d_ion,
                      self.d_wall_e, self.d_wall_i, self.d_wall_e_energy, self.d_wall_i_energy, self.d_exit_e, self.d_exit_i,
                      self.d_side_e, self.d_side_i, self.d_theta_i, self.d_iedf_i, self.d_see_e, self.d_see_energy):
            array.zero_()
        self.diag_steps = 0
        self.moment_samples = 0
        self.diagnostic_generation += 1


__all__ = [
    "WarpBackend",
    "WarpPoisson",
    "birth_bound",
    "device_available",
    "efield_stencil_codes",
    "resolve_device",
    "stream_seed",
    "warp_available",
]
