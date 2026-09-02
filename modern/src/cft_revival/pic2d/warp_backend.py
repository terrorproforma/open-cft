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
from typing import Any

import numpy as np

from .fields import MagneticFieldMap
from .kernels import FIXED_POINT_SCALE
from .mcc import MCCConfig, NullCollisionMCC, XenonCrossSections
from .mesh import MeshMasks
from .models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    EV_J,
    LIGHT_SPEED_M_PER_S,
    PIC2DConvergenceError,
    PIC2DDeviceError,
    PIC2DStabilityError,
    PIC2DValidationError,
    ParticleArrays,
    electron_species,
    xenon_ion_species,
)
from .poisson import Poisson2D, apply_operator, boundary_potential_array
from .simulation import (
    DiagnosticAccumulator,
    PIC2DConfig,
    SimulationState,
    StepTally,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dependency
    wp = None  # type: ignore[assignment]

REDUCTION_THREADS = 4096
REDUCTION_GROUP = 64


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
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, scale: F64,
        accumulator: wp.array(dtype=wp.int64),
    ):
        p = wp.tid()
        if alive[p] == 0:
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
        vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int,
        weight_sum: wp.array(dtype=F64), vr_sum: wp.array(dtype=F64), vt_sum: wp.array(dtype=F64),
        vz_sum: wp.array(dtype=F64), v2_sum: wp.array(dtype=F64),
    ):
        p = wp.tid()
        if alive[p] == 0:
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

    @wp.kernel
    def deposit_unit_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), flags: wp.array(dtype=wp.int32),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, out: wp.array(dtype=F64),
    ):
        p = wp.tid()
        if flags[p] == 0:
            return
        fr = r[p] / dr
        fz = (z[p] - z_min) / dz
        i = wp.clamp(int(wp.floor(fr)), 0, nr - 1)
        j = wp.clamp(int(wp.floor(fz)), 0, nz - 1)
        s = fr - F64(i)
        t = fz - F64(j)
        stride = nz + 1
        base = i * stride + j
        wp.atomic_add(out, base, (F64(1.0) - s) * (F64(1.0) - t))
        wp.atomic_add(out, base + stride, s * (F64(1.0) - t))
        wp.atomic_add(out, base + 1, (F64(1.0) - s) * t)
        wp.atomic_add(out, base + stride + 1, s * t)

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
    def deferred_add_kernel(partial: wp.array(dtype=F64), count: int, out: wp.array(dtype=F64), slot: int):
        acc = F64(0.0)
        for k in range(count):
            acc += partial[k]
        out[slot] = out[slot] + acc

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
        charge: F64, mass: F64, weight: F64, dt: F64, scale: F64, charge_sign: wp.int64,
        wall_accumulator: wp.array(dtype=wp.int64),
        stats: wp.array(dtype=F64), count_slot: int, energy_slot: int,
        accumulate: int, wall_columns: wp.array(dtype=F64), wall_column_energy: wp.array(dtype=F64),
        exit_bins: wp.array(dtype=F64),
    ):
        # stats layout (see WarpBackend.STATS): counts at count_slot+code-1, energies at
        # energy_slot+code-1, work at 0, max speed^2 at 1, invalid at 2
        p = wp.tid()
        if alive[p] == 0:
            return
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
        wp.atomic_add(stats, 0, k_after - k_before)
        wp.atomic_max(stats, 1, vr_new * vr_new + vt_new * vt_new + vz_new * vz_new)
        # boundary classification (identical to kernels.classify_boundary)
        code = 0
        if z_new < z_min:
            code = 1
        elif z_new >= z_max:
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
            return
        alive[p] = 0
        if code == 4:
            wp.atomic_add(stats, 2, F64(1.0))
            return
        wp.atomic_add(stats, count_slot + code - 1, F64(1.0))
        wp.atomic_add(stats, energy_slot + code - 1, k_after)
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
                return
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
                bin_index = wp.clamp(int(r_new / dr), 0, nr - 1)
                wp.atomic_add(exit_bins, bin_index, F64(1.0))

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

    @wp.kernel
    def mcc_kernel(
        vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32),
        count: int, seed: int, probability: F64, nu_max: F64, neutral_density: F64,
        table: wp.array(dtype=F64), points: int, step_ev: F64, max_ev: F64,
        threshold_exc: F64, threshold_ion: F64, b_ev: F64, ion_thermal: F64,
        ionize: wp.array(dtype=wp.int32), sec_vr: wp.array(dtype=F64), sec_vt: wp.array(dtype=F64), sec_vz: wp.array(dtype=F64),
        ion_vr: wp.array(dtype=F64), ion_vt: wp.array(dtype=F64), ion_vz: wp.array(dtype=F64),
        stats: wp.array(dtype=F64), tally_slot: int,
    ):
        p = wp.tid()
        ionize[p] = 0
        if p >= count or alive[p] == 0:
            return
        state = wp.rand_init(seed, p)
        u0 = F64(wp.randf(state))
        if u0 >= probability:
            return
        wp.atomic_add(stats, tally_slot, F64(1.0))
        vx = vr[p]
        vy = vt[p]
        vzp = vz[p]
        speed2 = vx * vx + vy * vy + vzp * vzp
        speed = wp.sqrt(speed2)
        m_e = F64(9.1093837139e-31)
        ev = F64(1.602176634e-19)
        energy = F64(0.5) * m_e * speed2 / ev
        nu0 = neutral_density * sigma_lookup(table, points, step_ev, max_ev, energy, 0) * speed
        nu1 = neutral_density * sigma_lookup(table, points, step_ev, max_ev, energy, 1) * speed
        nu2 = neutral_density * sigma_lookup(table, points, step_ev, max_ev, energy, 2) * speed
        if energy < threshold_exc:
            nu1 = F64(0.0)
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
            wp.atomic_add(stats, tally_slot + 1, F64(1.0))
        elif selector < c2:
            remaining = wp.max(energy - threshold_exc, F64(0.0))
            v = isotropic(wp.sqrt(F64(2.0) * remaining * ev / m_e), u2, u3)
            vr[p] = v[0]
            vt[p] = v[1]
            vz[p] = v[2]
            wp.atomic_add(stats, tally_slot + 2, F64(1.0))
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
            ion_vr[p] = ion_thermal * rad1 * wp.cos(F64(6.283185307179586) * u8)
            ion_vt[p] = ion_thermal * rad1 * wp.sin(F64(6.283185307179586) * u8)
            ion_vz[p] = ion_thermal * rad2 * wp.cos(F64(6.283185307179586) * u10)
            ionize[p] = 1
            wp.atomic_add(stats, tally_slot + 3, F64(1.0))
        else:
            wp.atomic_add(stats, tally_slot + 4, F64(1.0))

    @wp.kernel
    def spawn_kernel(
        ionize: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), count: int,
        r: wp.array(dtype=F64), z: wp.array(dtype=F64),
        sec_vr: wp.array(dtype=F64), sec_vt: wp.array(dtype=F64), sec_vz: wp.array(dtype=F64),
        ion_vr: wp.array(dtype=F64), ion_vt: wp.array(dtype=F64), ion_vz: wp.array(dtype=F64),
        e_r: wp.array(dtype=F64), e_z: wp.array(dtype=F64), e_vr: wp.array(dtype=F64), e_vt: wp.array(dtype=F64),
        e_vz: wp.array(dtype=F64), e_alive: wp.array(dtype=wp.int32), e_base: int,
        i_r: wp.array(dtype=F64), i_z: wp.array(dtype=F64), i_vr: wp.array(dtype=F64), i_vt: wp.array(dtype=F64),
        i_vz: wp.array(dtype=F64), i_alive: wp.array(dtype=wp.int32), i_base: int,
        born_r: wp.array(dtype=F64), born_z: wp.array(dtype=F64), born_flag: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        if p >= count:
            return
        born_flag[p] = 0
        if ionize[p] == 0:
            return
        k = offsets[p]
        de = e_base + k
        e_r[de] = r[p]
        e_z[de] = z[p]
        e_vr[de] = sec_vr[p]
        e_vt[de] = sec_vt[p]
        e_vz[de] = sec_vz[p]
        e_alive[de] = 1
        di = i_base + k
        i_r[di] = r[p]
        i_z[di] = z[p]
        i_vr[di] = ion_vr[p]
        i_vt[di] = ion_vt[p]
        i_vz[di] = ion_vz[p]
        i_alive[di] = 1
        born_r[p] = r[p]
        born_z[p] = z[p]
        born_flag[p] = 1

    @wp.kernel
    def inject_kernel(
        seed: int, base: int, r_max: F64, z_max: F64, dz: F64, thermal: F64,
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64),
        vz: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32), stats: wp.array(dtype=F64), slot: int, mass_weight: F64,
    ):
        k = wp.tid()
        state = wp.rand_init(seed, k)
        u0 = F64(wp.randf(state))
        u1 = F64(wp.randf(state))
        u2 = wp.max(F64(wp.randf(state)), F64(1.0e-30))
        u3 = F64(wp.randf(state))
        u4 = F64(wp.randf(state))
        u5 = F64(wp.randf(state))
        u6 = wp.max(F64(wp.randf(state)), F64(1.0e-30))
        d = base + k
        r[d] = r_max * wp.sqrt(u0) * (F64(1.0) - F64(1.0e-9))
        z[d] = z_max - F64(0.5) * dz * u1 - F64(1.0e-9) * dz
        rad1 = wp.sqrt(F64(-2.0) * wp.log(u2))
        vrk = thermal * rad1 * wp.cos(F64(6.283185307179586) * u3)
        vtk = thermal * rad1 * wp.sin(F64(6.283185307179586) * u3)
        vzk = -thermal * wp.sqrt(F64(-2.0) * wp.log(u6))
        vr[d] = vrk
        vt[d] = vtk
        vz[d] = vzk
        alive[d] = 1
        wp.atomic_add(stats, slot, kinetic_energy(vrk, vtk, vzk, mass_weight))

    @wp.kernel
    def compact_kernel(
        alive: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), count: int,
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        r2: wp.array(dtype=F64), z2: wp.array(dtype=F64), vr2: wp.array(dtype=F64), vt2: wp.array(dtype=F64), vz2: wp.array(dtype=F64),
        alive2: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        if p >= count or alive[p] == 0:
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
                          alive: wp.array(dtype=wp.int32), count: int, threads: int, mass_weight: F64,
                          partial: wp.array(dtype=F64)):
        k = wp.tid()
        acc = F64(0.0)
        p = k
        while p < count:
            if alive[p] != 0:
                acc += kinetic_energy(vr[p], vt[p], vz[p], mass_weight)
            p += threads
        partial[k] = acc

    @wp.kernel
    def peak_density_kernel(q_e: wp.array(dtype=F64), inverse_volume: wp.array(dtype=F64), stats: wp.array(dtype=F64), slot: int):
        n = wp.tid()
        wp.atomic_max(stats, slot, wp.abs(q_e[n]) * inverse_volume[n])


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


@dataclass
class DeviceSpecies:
    capacity: int
    count: int
    alive_count: int
    r: Any
    z: Any
    vr: Any
    vt: Any
    vz: Any
    alive: Any


# Per-step device statistics (float64; counts are exact integers in binary64).
STATS_WORK = 0
STATS_MAX_SPEED2 = 1
STATS_INVALID = 2
STATS_PEAK_DENSITY = 3
STATS_E_COUNTS = 4          # anode, exit, wall (electrons)
STATS_E_ENERGY = 7
STATS_I_COUNTS = 10         # anode, exit, wall (ions)
STATS_I_ENERGY = 13
STATS_MCC = 16              # candidates, elastic, excitation, ionization, null
STATS_SIZE = 24


class WarpBackend:
    """Warp implementation of the PIC-MCC cycle; interface matches ``CPUBackend``.

    Host synchronisation per step is limited to the Poisson convergence read
    and one read of the ``STATS_SIZE`` statistics vector; the cumulative ledger
    is kept on the host from those statistics.
    """

    def __init__(
        self,
        config: PIC2DConfig,
        masks: MeshMasks,
        field: MagneticFieldMap,
        cross_sections: XenonCrossSections | None,
        *,
        device: str = "cuda:0",
        compaction_interval: int = 20,
        use_graph: bool = True,
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
        if config.mcc is not None:
            if cross_sections is None:
                raise PIC2DValidationError("MCC requires cross sections")
            self.mcc = NullCollisionMCC(cross_sections, config.mcc, self.ion)
        self.compaction_interval = int(compaction_interval)
        grid = masks.grid
        self.nr, self.nz = grid.cell_shape
        self.node_count = int(np.prod(grid.node_shape))
        dev = self.device
        f64 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float64).ravel(), dtype=wp.float64, device=dev)  # noqa: E731
        i32 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.int32).ravel(), dtype=wp.int32, device=dev)  # noqa: E731
        # Field solve: the exact host block-Thomas solver is the default (a few
        # milliseconds for <=1e5 unknowns, deterministic, identical to the CPU
        # backend); the device Jacobi-PCG remains available via method="pcg".
        self.gpu_poisson: WarpPoisson | None = None
        self.host_poisson: Poisson2D | None = None
        if config.poisson.method == "pcg":
            self.gpu_poisson = WarpPoisson(masks, config.potentials, config.poisson, dev, use_graph=use_graph)
        else:
            self.host_poisson = Poisson2D(masks, config.poisson)
            self.ratio = f64(masks.charge_to_source)
            self.source_dev = wp.zeros(self.node_count, dtype=wp.float64, device=dev)
            self.host_phi = np.zeros(grid.node_shape)
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
        zeros = lambda dtype=wp.float64: wp.zeros(self.node_count, dtype=dtype, device=dev)  # noqa: E731
        self.acc_e = zeros(wp.int64)
        self.acc_i = zeros(wp.int64)
        self.acc_wall = zeros(wp.int64)
        self.q_e, self.q_i, self.surface, self.phi, self.e_r, self.e_z = (zeros() for _ in range(6))
        self.stats = wp.zeros(STATS_SIZE, dtype=wp.float64, device=dev)
        # deferred ledger energies (injected, born ions) accumulated across steps
        self.deferred = wp.zeros(2, dtype=wp.float64, device=dev)
        # diagnostics (device)
        self.d_n_e, self.d_n_i, self.d_phi, self.d_w, self.d_vr, self.d_vt, self.d_vz, self.d_v2, self.d_ion = (zeros() for _ in range(9))
        self.d_wall_e = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_wall_i = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_wall_e_energy = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_wall_i_energy = wp.zeros(self.nz, dtype=wp.float64, device=dev)
        self.d_exit_e = wp.zeros(self.nr, dtype=wp.float64, device=dev)
        self.d_exit_i = wp.zeros(self.nr, dtype=wp.float64, device=dev)
        self.diag_steps = 0
        self.species: dict[str, DeviceSpecies] = {}
        self.scratch_capacity = 0
        self.state_meta: dict[str, Any] = {}
        if self.mcc is not None:
            self.table = f64(self.mcc.table.table_m2)
            self.table_points = self.mcc.table.point_count
        self.sync_count = 0

    # ------------------------------------------------------------------ helpers
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
        self.ionize = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.offsets = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.sec = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(3)]
        self.ionv = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(3)]
        self.born_r = wp.zeros(capacity, dtype=wp.float64, device=dev)
        self.born_z = wp.zeros(capacity, dtype=wp.float64, device=dev)
        self.born_flag = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.tmp = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(5)]
        self.tmp_alive = wp.zeros(capacity, dtype=wp.int32, device=dev)
        self.partial_particles = wp.zeros(REDUCTION_THREADS, dtype=wp.float64, device=dev)

    def _grow(self, species: DeviceSpecies, minimum: int) -> DeviceSpecies:
        if minimum <= species.capacity:
            return species
        capacity = max(minimum, 2 * species.capacity, 1024)
        new = self._alloc_species(capacity)
        if species.count:
            for src, dst in zip((species.r, species.z, species.vr, species.vt, species.vz, species.alive),
                                (new.r, new.z, new.vr, new.vt, new.vz, new.alive)):
                wp.copy(dst, src, count=species.count)
        new.count = species.count
        new.alive_count = species.alive_count
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
            wp.copy(species.alive, wp.array(np.ones(particles.count, dtype=np.int32), dtype=wp.int32, device=self.device), count=particles.count)
        species.count = particles.count
        species.alive_count = particles.count
        self._ensure_scratch(capacity)
        return species

    def _download(self, species: DeviceSpecies) -> ParticleArrays:
        self._compact(species)
        n = species.count
        if n == 0:
            return ParticleArrays.empty()
        arrays = [np.asarray(a.numpy()[:n], dtype=np.float64).copy() for a in (species.r, species.z, species.vr, species.vt, species.vz)]
        return ParticleArrays(*arrays)

    def _compact(self, species: DeviceSpecies) -> None:
        n = species.count
        if n == 0:
            return
        self._ensure_scratch(species.capacity)
        wp.utils.array_scan(species.alive[:n], self.offsets[:n], inclusive=True)
        total = int(self.offsets.numpy()[n - 1])
        self.sync_count += 1
        if total != species.alive_count:
            raise PIC2DValidationError(
                f"device alive count {total} disagrees with the host ledger {species.alive_count}"
            )
        # exclusive offsets = inclusive - alive
        wp.utils.array_scan(species.alive[:n], self.offsets[:n], inclusive=False)
        tmp = self.tmp
        self.tmp_alive.zero_()
        wp.launch(compact_kernel, dim=n, inputs=[species.alive, self.offsets, n, species.r, species.z, species.vr, species.vt, species.vz,
                                                 tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], self.tmp_alive], device=self.device)
        species.alive.zero_()
        if total:
            for src, dst in zip((tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]), (species.r, species.z, species.vr, species.vt, species.vz)):
                wp.copy(dst, src, count=total)
            wp.copy(species.alive, self.tmp_alive, count=total)
        species.count = total

    # ------------------------------------------------------------------ state exchange
    def load_state(self, state: SimulationState) -> None:
        self.species = {"e": self._upload(state.electrons), "i": self._upload(state.ions)}
        self.surface = wp.array(state.surface_charge_c.ravel(), dtype=wp.float64, device=self.device)
        self.phi = wp.array(state.phi_v.ravel(), dtype=wp.float64, device=self.device)
        if self.gpu_poisson is not None:
            x = state.phi_v.copy()
            x[~self.masks.unknown_node] = 0.0
            wp.copy(self.gpu_poisson.x, wp.array(x.ravel(), dtype=wp.float64, device=self.device))
        self.state_meta = {
            "step": int(state.step), "time_s": float(state.time_s),
            "injection_carry": float(state.injection_carry), "cumulative": dict(state.cumulative),
        }

    def export_state(self) -> SimulationState:
        wp.synchronize_device(self.device)
        self._flush_pending_energy()
        shape = self.masks.grid.node_shape
        return SimulationState(
            self.state_meta["step"], self.state_meta["time_s"],
            self._download(self.species["e"]), self._download(self.species["i"]),
            self.surface.numpy().reshape(shape).copy(), self.phi.numpy().reshape(shape).copy(),
            self.state_meta["injection_carry"], dict(self.state_meta["cumulative"]),
        )

    # ------------------------------------------------------------------ cycle
    def _deposit(self, species: DeviceSpecies, accumulator, out, per_particle: float) -> None:
        grid = self.masks.grid
        accumulator.zero_()
        if species.count:
            wp.launch(deposit_fixed_kernel, dim=species.count,
                      inputs=[species.r, species.z, species.alive, grid.dr_m, grid.dz_m, grid.geometry.z_min_m,
                              self.nr, self.nz, FIXED_POINT_SCALE, accumulator], device=self.device)
        wp.launch(int_to_charge_kernel, dim=self.node_count, inputs=[accumulator, per_particle / FIXED_POINT_SCALE, out], device=self.device)

    def step(self, accumulate: bool) -> StepTally:
        config = self.config
        grid = self.masks.grid
        dev = self.device
        dt = config.dt_s
        meta = self.state_meta
        step_index = meta["step"]
        electrons = self.species["e"]
        ions = self.species["i"]
        mcc = self.mcc

        self._deposit(electrons, self.acc_e, self.q_e, self.electron.charge_c * config.macro_weight)
        self._deposit(ions, self.acc_i, self.q_i, self.ion.charge_c * config.macro_weight)
        if self.gpu_poisson is not None:
            iterations, _, _ = self.gpu_poisson.solve(self.q_e, self.q_i, self.surface, self.phi)
        else:
            wp.launch(host_source_kernel, dim=self.node_count, inputs=[self.q_e, self.q_i, self.ratio, self.surface, self.source_dev], device=dev)
            source = self.source_dev.numpy().reshape(grid.node_shape)
            result = self.host_poisson.solve(source, config.potentials)  # type: ignore[union-attr]
            iterations = result.diagnostics.iterations
            wp.copy(self.phi, wp.array(result.phi_v.ravel(), dtype=wp.float64, device=dev))
        self.sync_count += 1
        wp.launch(efield_kernel, dim=self.node_count, inputs=[self.phi, self.code_r, self.code_z, grid.dr_m, grid.dz_m, self.nz, self.e_r, self.e_z], device=dev)
        self.stats.zero_()
        wp.launch(peak_density_kernel, dim=self.node_count, inputs=[self.q_e, self.inverse_volume, self.stats, STATS_PEAK_DENSITY], device=dev)

        if accumulate:
            wp.launch(axpy_kernel, dim=self.node_count, inputs=[self.d_phi, 1.0, self.phi], device=dev)
            wp.launch(abs_axpy_kernel, dim=self.node_count, inputs=[self.d_n_e, self.inverse_volume, self.q_e], device=dev)
            wp.launch(abs_axpy_kernel, dim=self.node_count, inputs=[self.d_n_i, self.inverse_volume, self.q_i], device=dev)
            if electrons.count:
                wp.launch(deposit_moment_kernel, dim=electrons.count,
                          inputs=[electrons.r, electrons.z, electrons.alive, electrons.vr, electrons.vt, electrons.vz,
                                  grid.dr_m, grid.dz_m, grid.geometry.z_min_m, self.nr, self.nz,
                                  self.d_w, self.d_vr, self.d_vt, self.d_vz, self.d_v2], device=dev)

        for index, (species, particles) in enumerate(((self.electron, electrons), (self.ion, ions))):
            if particles.count == 0:
                continue
            is_electron = index == 0
            wp.launch(
                push_kernel, dim=particles.count,
                inputs=[particles.r, particles.z, particles.vr, particles.vt, particles.vz, particles.alive,
                        self.e_r, self.e_z, self.b_r, self.b_z, grid.dr_m, grid.dz_m, grid.geometry.z_min_m, grid.geometry.z_max_m,
                        self.nr, self.nz, self.plasma_cell, self.top_cell, self.plasma_node,
                        species.charge_c, species.mass_kg, species.macro_weight, dt, FIXED_POINT_SCALE,
                        wp.int64(1 if species.charge_c > 0 else -1), self.acc_wall,
                        self.stats, STATS_E_COUNTS if is_electron else STATS_I_COUNTS,
                        STATS_E_ENERGY if is_electron else STATS_I_ENERGY, 1 if accumulate else 0,
                        self.d_wall_e if is_electron else self.d_wall_i,
                        self.d_wall_e_energy if is_electron else self.d_wall_i_energy,
                        self.d_exit_e if is_electron else self.d_exit_i],
                device=dev,
            )
        wp.launch(wall_int_to_charge_kernel, dim=self.node_count,
                  inputs=[self.acc_wall, ELEMENTARY_CHARGE_C * config.macro_weight / FIXED_POINT_SCALE, self.surface], device=dev)

        n_electrons = electrons.count
        if mcc is not None and n_electrons:
            electrons = self._grow(electrons, 2 * n_electrons + 1024)
            ions = self._grow(ions, ions.count + n_electrons + 1024)
            self._ensure_scratch(max(electrons.capacity, ions.capacity))
            ion_thermal = sqrt(1.380649e-23 * config.mcc.neutral_temperature_k / self.ion.mass_kg)  # type: ignore[union-attr]
            wp.launch(
                mcc_kernel, dim=n_electrons,
                inputs=[electrons.vr, electrons.vt, electrons.vz, electrons.alive, n_electrons, stream_seed(config.seed, step_index, 1),
                        mcc.collision_probability(dt), mcc.nu_max, config.mcc.neutral_density_per_m3,  # type: ignore[union-attr]
                        self.table, self.table_points, mcc.table.energy_step_ev, mcc.table.energy_max_ev,
                        mcc.table.thresholds_ev[1], mcc.table.thresholds_ev[2], 8.7, ion_thermal,
                        self.ionize, self.sec[0], self.sec[1], self.sec[2], self.ionv[0], self.ionv[1], self.ionv[2],
                        self.stats, STATS_MCC],
                device=dev,
            )

        # ---- single host read of the per-step statistics
        stats = self.stats.numpy()
        self.sync_count += 1
        if stats[STATS_INVALID] != 0.0:
            raise PIC2DStabilityError("a particle crossed more than one cell in a step (Courant violation)")
        cumulative = meta["cumulative"]
        for base, label in ((STATS_E_COUNTS, "electrons"), (STATS_I_COUNTS, "ions")):
            cumulative[f"anode_{label}"] += float(stats[base])
            cumulative[f"exit_{label}"] += float(stats[base + 1])
            cumulative[f"wall_{label}"] += float(stats[base + 2])
        electrons.alive_count -= int(stats[STATS_E_COUNTS] + stats[STATS_E_COUNTS + 1] + stats[STATS_E_COUNTS + 2])
        ions.alive_count -= int(stats[STATS_I_COUNTS] + stats[STATS_I_COUNTS + 1] + stats[STATS_I_COUNTS + 2])
        for offset, key in ((0, "ke_absorbed_anode_j"), (1, "ke_absorbed_exit_j"), (2, "ke_absorbed_wall_j")):
            cumulative[key] += float(stats[STATS_E_ENERGY + offset] + stats[STATS_I_ENERGY + offset])
        cumulative["field_work_j"] += float(stats[STATS_WORK])
        max_speed2 = float(stats[STATS_MAX_SPEED2])
        peak_density = float(stats[STATS_PEAK_DENSITY])

        if mcc is not None and n_electrons:
            n_ion = int(stats[STATS_MCC + 3])
            cumulative["elastic"] += float(stats[STATS_MCC + 1])
            cumulative["excitations"] += float(stats[STATS_MCC + 2])
            cumulative["ionizations"] += float(n_ion)
            cumulative["inelastic_loss_j"] += (float(stats[STATS_MCC + 2]) * mcc.table.thresholds_ev[1] + n_ion * mcc.table.thresholds_ev[2]) * EV_J
            if n_ion:
                n = n_electrons
                wp.utils.array_scan(self.ionize[:n], self.offsets[:n], inclusive=False)
                wp.launch(
                    spawn_kernel, dim=n,
                    inputs=[self.ionize, self.offsets, n, electrons.r, electrons.z, self.sec[0], self.sec[1], self.sec[2],
                            self.ionv[0], self.ionv[1], self.ionv[2],
                            electrons.r, electrons.z, electrons.vr, electrons.vt, electrons.vz, electrons.alive, electrons.count,
                            ions.r, ions.z, ions.vr, ions.vt, ions.vz, ions.alive, ions.count,
                            self.born_r, self.born_z, self.born_flag],
                    device=dev,
                )
                wp.launch(energy_sum_kernel, dim=REDUCTION_THREADS,
                          inputs=[self.ionv[0], self.ionv[1], self.ionv[2], self.born_flag, n, REDUCTION_THREADS,
                                  self.ion.mass_kg * config.macro_weight, self.partial_particles], device=dev)
                wp.launch(deferred_add_kernel, dim=1, inputs=[self.partial_particles, REDUCTION_THREADS, self.deferred, 1], device=dev)
                if accumulate:
                    wp.launch(deposit_unit_kernel, dim=n, inputs=[self.born_r, self.born_z, self.born_flag, grid.dr_m, grid.dz_m,
                                                                  grid.geometry.z_min_m, self.nr, self.nz, self.d_ion], device=dev)
                electrons.count += n_ion
                electrons.alive_count += n_ion
                ions.count += n_ion
                ions.alive_count += n_ion

        injected = 0
        if config.injection is not None and config.injection.electron_current_a > 0.0:
            expected = config.injection.electron_current_a * dt / (ELEMENTARY_CHARGE_C * config.macro_weight) + meta["injection_carry"]
            injected = int(np.floor(expected))
            meta["injection_carry"] = expected - injected
            if injected:
                electrons = self._grow(electrons, electrons.count + injected)
                r_max = grid.r_m[self.masks.top_plasma_cell[self.nz - 1] + 1]
                thermal = sqrt(EV_J * config.injection.electron_temperature_ev / ELECTRON_MASS_KG)
                wp.launch(inject_kernel, dim=injected,
                          inputs=[stream_seed(config.seed, step_index, 2), electrons.count, float(r_max), grid.geometry.z_max_m, grid.dz_m, thermal,
                                  electrons.r, electrons.z, electrons.vr, electrons.vt, electrons.vz, electrons.alive,
                                  self.deferred, 0, ELECTRON_MASS_KG * config.macro_weight], device=dev)
                electrons.count += injected
                electrons.alive_count += injected
                cumulative["injected_electrons"] += float(injected)
        self._pending_energy = (injected > 0) or (mcc is not None and n_electrons > 0)

        if (step_index + 1) % self.compaction_interval == 0:
            # The compaction read also flushes the deferred energy slots.
            self._flush_pending_energy()
            self._compact(electrons)
            self._compact(ions)
        self.species["e"] = electrons
        self.species["i"] = ions
        meta["step"] = step_index + 1
        meta["time_s"] = meta["step"] * dt
        if accumulate:
            self.diag_steps += 1
        omega_pe = sqrt(peak_density * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ELECTRON_MASS_KG))
        return StepTally(iterations, omega_pe * dt, sqrt(max_speed2), electrons.alive_count, ions.alive_count)

    def _flush_pending_energy(self) -> None:
        """Read the injected/born energy slots written after the per-step stats read."""

        if not getattr(self, "_pending_energy", False):
            return
        deferred = self.deferred.numpy()
        self.deferred.zero_()
        self.sync_count += 1
        cumulative = self.state_meta["cumulative"]
        cumulative["ke_injected_j"] += float(deferred[0])
        cumulative["ke_born_ions_j"] += float(deferred[1])
        self._pending_energy = False

    # ------------------------------------------------------------------ diagnostics
    def diagnostic_arrays(self) -> dict[str, np.ndarray]:
        wp.synchronize_device(self.device)
        shape = self.masks.grid.node_shape
        acc = DiagnosticAccumulator(self.masks)
        acc.steps = self.diag_steps
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
        return acc.to_arrays(self.config.macro_weight, self.config.dt_s)

    def reset_diagnostics(self) -> None:
        for array in (self.d_n_e, self.d_n_i, self.d_phi, self.d_w, self.d_vr, self.d_vt, self.d_vz, self.d_v2, self.d_ion,
                      self.d_wall_e, self.d_wall_i, self.d_wall_e_energy, self.d_wall_i_energy, self.d_exit_e, self.d_exit_i):
            array.zero_()
        self.diag_steps = 0


__all__ = [
    "WarpBackend",
    "WarpPoisson",
    "device_available",
    "efield_stencil_codes",
    "resolve_device",
    "stream_seed",
    "warp_available",
]
