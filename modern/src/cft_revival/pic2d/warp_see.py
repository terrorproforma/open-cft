# ruff: noqa: SIM102  (nested single-condition ifs are kept deliberately inside the Warp kernels: one comparison per branch)
"""Warp kernels of the v2.2.0 SEE stage (``see_dielectric_v1``), shared by the Warp CPU and CUDA backends.

Design (physics contract = ``see.py``, the numpy reference):

* ``push_kernel`` (warp_backend.py) flags every wall impact (boundary code 3) in a per-slot ``wall_hit`` array
  and, as before, leaves the dead slot's arrays at the PRE-push state.  The SEE stage runs at the END of the
  step (after MCC / spawn / injection, which never touch dead slots or flagged ones) and reconstructs the
  impact from that state: it re-gathers the fields the push used (``e_r``/``e_z`` are not modified before the
  next step) and repeats the identical Boris push + advance, so the impact position, velocity and energy are the
  ones the push kernel absorbed (same ``wp.func`` bodies; any FMA-contraction difference is round-off of a
  dead particle's diagnostics, never of a live one).
* ``see_sample_kernel``: per flagged slot, total yield ``delta(E, theta)`` (Vaughan + optional Sydorenko bump,
  or the constant model), integer yield ``n = floor + Bernoulli`` from the SEE random stream (seed-table stream 3
  = the 4th per-step stream; ``wp.rand_init(seed, slot)``), the wall-charge deposit of the ``n`` emitted electrons
  (``+n`` fixed-point units at the impact's renormalised bilinear stencil, the same arithmetic as the absorbed
  deposit), the per-column emitted count and the impact / yield tallies.  Writes ``count[slot] = n`` and clears
  the flag (the flags are all zero after every step: the invariant the fixed-shape graph relies on).
* exclusive scan of ``count`` -> ``offsets``; ``see_spawn_kernel`` re-derives the impact and the same random
  stream (the count draw is consumed first, so the emission draws follow it exactly as in the sample kernel's
  stream), then writes the ``n`` electrons at ``slots[0] + offsets[slot] + k`` with the component split and the
  velocity samplers of ``see.py``, tallying the emitted energy / momentum / backscattered count and the
  per-column emitted energy; ``see_commit_kernel`` advances ``slots[0]``.  Capacity overflow fails closed at the
  next host sync (``STATS_OVERFLOW``), like the MCC spawn.
* ion-induced emission (``ion_induced_yield > 0``) is the same three launches over the ION arrays on ion push
  steps, spawning electrons; its yield is constant and every emitted electron is a true secondary.

All constants live in a small device array (``see_params``), so the kernels have fixed arguments and the whole
stage is CUDA-graph capturable; the stage adds 3 (+3) launches and one scan per step, all guarded on the device
slot counts and flags, and touches only the flagged slots.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

from .kernels import FIXED_POINT_SCALE
from .models import ELECTRON_MASS_KG, EV_J
from .see import FACE_NUDGE, NORMAL_MINUS_R, NORMAL_MINUS_Z, NORMAL_PLUS_Z, SEEConfig

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dependency
    wp = None  # type: ignore[assignment]

# see_params layout (float64 device array)
P_MODEL = 0             # 0 = vaughan_components, 1 = constant
P_CONSTANT = 1
P_DELTA_MAX = 2
P_ENERGY_MAX = 3
P_THRESHOLD = 4
P_K_RISE = 5
P_K_FALL = 6
P_SMOOTHNESS = 7
P_ELASTIC = 8
P_INELASTIC = 9
P_BUMP_PEAK = 10
P_BUMP_PEAK_EV = 11
P_BUMP_THRESHOLD_EV = 12
P_BUMP_DECAY_EV = 13
P_THERMAL = 14          # sqrt(T_see / m_e) in m/s
P_ION_YIELD = 15
P_MAX_EMITTED = 16
P_BODY_RADIUS = 17      # front-face conductor starts here (plume geometries); huge otherwise
P_CONSTANT_THRESHOLD = 18
P_SIZE = 19

SEE_STREAM = 4          # 5th per-step seed-table column (0 MCC, 1 injection, 2 anomalous, 3 ion-neutral MCC, 4 SEE)


def see_params_array(config: SEEConfig, body_dielectric_radius_m: float | None, device: Any):
    """Device constants of the SEE stage for one configuration."""

    material = config.resolved_material()
    values = np.zeros(P_SIZE, dtype=np.float64)
    values[P_MODEL] = 0.0 if config.yield_model == "vaughan_components" else 1.0
    values[P_CONSTANT] = config.constant_yield
    values[P_DELTA_MAX] = material.delta_max
    values[P_ENERGY_MAX] = material.energy_max_ev
    values[P_THRESHOLD] = material.energy_threshold_ev
    values[P_K_RISE] = material.k_rise
    values[P_K_FALL] = material.k_fall
    values[P_SMOOTHNESS] = material.smoothness
    values[P_ELASTIC] = material.elastic_fraction
    values[P_INELASTIC] = material.inelastic_fraction
    values[P_BUMP_PEAK] = material.low_energy_elastic_peak
    values[P_BUMP_PEAK_EV] = material.low_energy_elastic_peak_ev
    values[P_BUMP_THRESHOLD_EV] = material.low_energy_elastic_threshold_ev
    values[P_BUMP_DECAY_EV] = material.low_energy_elastic_decay_ev
    values[P_THERMAL] = sqrt(EV_J * config.emission_temperature_ev / ELECTRON_MASS_KG)
    values[P_ION_YIELD] = config.ion_induced_yield
    values[P_MAX_EMITTED] = float(config.max_emitted_per_impact)
    values[P_BODY_RADIUS] = 1.0e30 if body_dielectric_radius_m is None else float(body_dielectric_radius_m)
    values[P_CONSTANT_THRESHOLD] = config.constant_yield_threshold_ev
    return wp.array(values, dtype=wp.float64, device=device)


if wp is not None:
    from .warp_backend import (
        F64,
        PARTICLE_BLOCK,
        SEED_STREAMS,
        STATS_OVERFLOW,
        STATS_SEE_BACKSCATTERED,
        STATS_SEE_CLAMPED,
        STATS_SEE_EMITTED,
        STATS_SEE_IMPACTS,
        STATS_SEE_ION_EMITTED,
        STATS_SEE_KE,
        STATS_SEE_PZ,
        STATS_SEE_YIELD_SUM,
        kinetic_energy,
        padded_dim,
        relativistic_boris,
    )

    vec6d = wp.types.vector(length=6, dtype=wp.float64)

    @wp.func
    def see_impact_state(
        p: int, r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64),
        vz: wp.array(dtype=F64), e_r: wp.array(dtype=F64), e_z: wp.array(dtype=F64), b_r: wp.array(dtype=F64),
        b_z: wp.array(dtype=F64), dr: F64, dz: F64, z_min: F64, nr: int, nz: int, charge: F64, mass: F64, dt: F64,
    ) -> vec6d:
        """Repeat the push of slot ``p`` from its retained pre-push state: (r_new, z_new, vr_new, vt_new, vz_new, 0)."""

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
        v = relativistic_boris(vr[p], vt[p], vz[p], ex, ez, bx, bz, charge, mass, dt)
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
        return vec6d(r_new, z_new, vr_new, vt_new, vz_new, F64(0.0))

    @wp.func
    def see_wall_crossing(
        r0: F64, z0: F64, r1: F64, z1: F64, dr: F64, dz: F64, z_min: F64, nr: int, nz: int, plasma_cell: wp.array(dtype=wp.int32),
    ) -> wp.vec3d:
        """Emission point and inward-normal code (see.wall_crossing): (r_e, z_e, code)."""

        fr0 = r0 / dr
        fz0 = (z0 - z_min) / dz
        fr1 = r1 / dr
        fz1 = (z1 - z_min) / dz
        i0 = wp.clamp(int(wp.floor(fr0)), 0, nr - 1)
        j0 = wp.clamp(int(wp.floor(fz0)), 0, nz - 1)
        inf = F64(1.0e300)
        t_r = inf
        if fr1 > F64(i0 + 1):
            t_r = (F64(i0 + 1) - fr0) / (fr1 - fr0)
        t_z = inf
        code_z = NORMAL_PLUS_Z
        j_behind = j0 - 1
        if fz1 >= F64(j0 + 1):
            t_z = (F64(j0 + 1) - fz0) / (fz1 - fz0)
            code_z = NORMAL_MINUS_Z
            j_behind = j0 + 1
        elif fz1 < F64(j0):
            t_z = (F64(j0) - fz0) / (fz1 - fz0)
        radial_behind = 0
        if i0 + 1 < nr:
            if plasma_cell[(i0 + 1) * nz + j0] != 0:
                radial_behind = 1
        axial_behind = 0
        if j_behind >= 0:
            if j_behind < nz:
                if plasma_cell[i0 * nz + j_behind] != 0:
                    axial_behind = 1
        has_r = 0
        if t_r < inf:
            has_r = 1
        has_z = 0
        if t_z < inf:
            has_z = 1
        radial_first = 0
        if t_r <= t_z:
            radial_first = 1
        radial_wall = 0
        if has_r != 0:
            if radial_first != 0:
                if radial_behind == 0 or has_z == 0:
                    radial_wall = 1
            elif has_z != 0:
                if axial_behind != 0:
                    radial_wall = 1
        axial_wall = 0
        if radial_wall == 0:
            if has_z != 0:
                axial_wall = 1
        t = F64(0.0)
        if radial_wall != 0:
            t = t_r
        elif axial_wall != 0:
            t = t_z
        t = wp.clamp(t, F64(0.0), F64(1.0))
        r_e = r0 + t * (r1 - r0)
        z_e = z0 + t * (z1 - z0)
        code = NORMAL_MINUS_R
        if axial_wall != 0:
            code = code_z
        r_face = (F64(i0 + 1) - F64(FACE_NUDGE)) * dr
        z_lo = z_min + (F64(j0) + F64(FACE_NUDGE)) * dz
        z_hi = z_min + (F64(j0 + 1) - F64(FACE_NUDGE)) * dz
        r_lo = F64(i0) * dr
        if axial_wall != 0:
            r_e = wp.clamp(r_e, r_lo, r_face)
            if code == NORMAL_MINUS_Z:
                z_e = z_hi
            else:
                z_e = z_lo
        else:
            r_e = r_face
            z_e = wp.clamp(z_e, z_lo, z_hi)
        return wp.vec3d(r_e, z_e, F64(code))

    @wp.func
    def see_yields(energy_ev: F64, theta: F64, params: wp.array(dtype=F64)) -> wp.vec3d:
        """(total, elastic, inelastic) yields at normal-incidence energy ``energy_ev`` and angle ``theta``."""

        if params[P_MODEL] > F64(0.5):
            total = F64(0.0)
            if energy_ev > params[P_CONSTANT_THRESHOLD]:
                total = params[P_CONSTANT]
            return wp.vec3d(total, total * params[P_ELASTIC], total * params[P_INELASTIC])
        pi = F64(3.141592653589793)
        smooth = params[P_SMOOTHNESS]
        e_max = params[P_ENERGY_MAX] * (F64(1.0) + smooth * theta * theta / (F64(2.0) * pi))
        d_max = params[P_DELTA_MAX] * (F64(1.0) + smooth * theta * theta / pi)
        e0 = params[P_THRESHOLD]
        base = F64(0.0)
        if energy_ev > e0:
            v = (energy_ev - e0) / (e_max - e0)
            if v <= F64(3.6):
                k = params[P_K_FALL]
                if v < F64(1.0):
                    k = params[P_K_RISE]
                vv = wp.max(v, F64(0.0))
                base = d_max * wp.pow(vv * wp.exp(F64(1.0) - vv), k)
            else:
                base = d_max * F64(1.125) / wp.pow(v, F64(0.35))
        bump = F64(0.0)
        peak = params[P_BUMP_PEAK]
        if peak > F64(0.0):
            e_th = params[P_BUMP_THRESHOLD_EV]
            e_pk = params[P_BUMP_PEAK_EV]
            if energy_ev > e_th:
                if energy_ev < e_pk:
                    v1 = (energy_ev - e_th) / (e_pk - e_th)
                    bump = peak * v1 * wp.exp(F64(1.0) - v1)
                else:
                    v2 = (energy_ev - e_pk) / params[P_BUMP_DECAY_EV]
                    bump = peak * (F64(1.0) + v2) * wp.exp(-v2)
        return wp.vec3d(base + bump, params[P_ELASTIC] * base + bump, params[P_INELASTIC] * base)

    @wp.func
    def see_orient(v_n: F64, v_a: F64, v_b: F64, code: int) -> wp.vec3d:
        """(normal, tangential a, tangential b) -> (v_r, v_theta, v_z) for the inward-normal code."""

        if code == NORMAL_MINUS_R:
            return wp.vec3d(-v_n, v_a, v_b)
        if code == NORMAL_MINUS_Z:
            return wp.vec3d(v_a, v_b, -v_n)
        return wp.vec3d(v_a, v_b, v_n)

    @wp.kernel
    def see_sample_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        hit: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32), slot: int,
        e_r: wp.array(dtype=F64), e_z: wp.array(dtype=F64), b_r: wp.array(dtype=F64), b_z: wp.array(dtype=F64),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, plasma_cell: wp.array(dtype=wp.int32), plasma_node: wp.array(dtype=wp.int32),
        charge: F64, mass: F64, weight: F64, dt: F64, is_ion: int, params: wp.array(dtype=F64),
        seed_table: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32),
        scale: F64, wall_accumulator: wp.array(dtype=wp.int64), stats: wp.array(dtype=F64),
        accumulate: int, see_columns: wp.array(dtype=F64), flag_bound: int,
    ):
        p = wp.tid()
        n = 0
        impact = F64(0.0)
        yield_sum = F64(0.0)
        clamped = F64(0.0)
        active = 0
        if p < flag_bound:
            if p < slots[slot]:
                if hit[p] != 0:
                    active = 1
        if active != 0:
            hit[p] = 0
            state6 = see_impact_state(p, r, z, vr, vt, vz, e_r, e_z, b_r, b_z, dr, dz, z_min, nr, nz, charge, mass, dt)
            r_new = state6[0]
            z_new = state6[1]
            if r_new < params[P_BODY_RADIUS]:      # the grounded front-face conductor of a plume box does not emit
                delta = F64(0.0)
                if is_ion != 0:
                    delta = params[P_ION_YIELD]
                else:
                    crossing = see_wall_crossing(r[p], z[p], r_new, z_new, dr, dz, z_min, nr, nz, plasma_cell)
                    code = int(crossing[2])
                    speed2 = state6[2] * state6[2] + state6[3] * state6[3] + state6[4] * state6[4]
                    speed = wp.sqrt(speed2)
                    v_normal = state6[4]
                    if code == NORMAL_MINUS_R:
                        v_normal = state6[2]
                    cos_inc = F64(1.0)
                    if speed > F64(0.0):
                        cos_inc = wp.clamp(wp.abs(v_normal) / speed, F64(0.0), F64(1.0))
                    theta = wp.acos(cos_inc)
                    energy_ev = kinetic_energy(state6[2], state6[3], state6[4], mass * weight) / (weight * F64(1.602176634e-19))
                    yields = see_yields(energy_ev, theta, params)
                    delta = yields[0]
                    impact = F64(1.0)
                    yield_sum = delta
                    if delta > params[P_MAX_EMITTED]:
                        clamped = F64(1.0)
                seed = seed_table[SEED_STREAMS * counter[0] + SEE_STREAM]
                rng = wp.rand_init(seed, p)
                u0 = F64(wp.randf(rng))
                base = wp.floor(delta)
                n = int(base)
                if u0 < delta - base:
                    n = n + 1
                n = wp.min(n, int(params[P_MAX_EMITTED]))
                if n > 0:
                    # the n emitted electrons leave +n e W on the wall at the impact's renormalised stencil (push_kernel's arithmetic)
                    stride = nz + 1
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
                    if total > F64(0.0):
                        units = wp.int64(n)
                        wp.atomic_add(wall_accumulator, m00, wp.int64(wp.rint(a00 / total * scale)) * units)
                        wp.atomic_add(wall_accumulator, m10, wp.int64(wp.rint(a10 / total * scale)) * units)
                        wp.atomic_add(wall_accumulator, m01, wp.int64(wp.rint(a01 / total * scale)) * units)
                        wp.atomic_add(wall_accumulator, m11, wp.int64(wp.rint(a11 / total * scale)) * units)
                    if accumulate != 0:
                        col = wp.clamp(int((z_new - z_min) / dz), 0, nz - 1)
                        wp.atomic_add(see_columns, col, F64(n))
        if p < flag_bound:
            count[p] = n
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(impact)), offset=STATS_SEE_IMPACTS)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(yield_sum)), offset=STATS_SEE_YIELD_SUM)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(clamped)), offset=STATS_SEE_CLAMPED)

    @wp.kernel
    def see_spawn_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        count: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32), e_capacity: int,
        e_r: wp.array(dtype=F64), e_z: wp.array(dtype=F64), b_r: wp.array(dtype=F64), b_z: wp.array(dtype=F64),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, plasma_cell: wp.array(dtype=wp.int32),
        charge: F64, mass: F64, weight: F64, dt: F64, is_ion: int, params: wp.array(dtype=F64),
        seed_table: wp.array(dtype=wp.int32), counter: wp.array(dtype=wp.int32),
        out_r: wp.array(dtype=F64), out_z: wp.array(dtype=F64), out_vr: wp.array(dtype=F64), out_vt: wp.array(dtype=F64),
        out_vz: wp.array(dtype=F64), out_alive: wp.array(dtype=wp.int32),
        stats: wp.array(dtype=F64), emitted_slot: int, electron_mass_weight: F64,
        accumulate: int, see_column_energy: wp.array(dtype=F64), flag_bound: int,
    ):
        p = wp.tid()
        n = 0
        if p < flag_bound:
            n = count[p]
        emitted = F64(0.0)
        back = F64(0.0)
        ke = F64(0.0)
        pz = F64(0.0)
        if n > 0:
            state6 = see_impact_state(p, r, z, vr, vt, vz, e_r, e_z, b_r, b_z, dr, dz, z_min, nr, nz, charge, mass, dt)
            r_new = state6[0]
            z_new = state6[1]
            crossing = see_wall_crossing(r[p], z[p], r_new, z_new, dr, dz, z_min, nr, nz, plasma_cell)
            r_e = crossing[0]
            z_e = crossing[1]
            code = int(crossing[2])
            p_elastic = F64(0.0)
            p_inelastic = F64(0.0)
            speed = F64(0.0)
            if is_ion == 0:
                speed2 = state6[2] * state6[2] + state6[3] * state6[3] + state6[4] * state6[4]
                speed = wp.sqrt(speed2)
                v_normal = state6[4]
                if code == NORMAL_MINUS_R:
                    v_normal = state6[2]
                cos_inc = F64(1.0)
                if speed > F64(0.0):
                    cos_inc = wp.clamp(wp.abs(v_normal) / speed, F64(0.0), F64(1.0))
                theta = wp.acos(cos_inc)
                energy_ev = kinetic_energy(state6[2], state6[3], state6[4], mass * weight) / (weight * F64(1.602176634e-19))
                yields = see_yields(energy_ev, theta, params)
                if yields[0] > F64(0.0):
                    p_elastic = yields[1] / yields[0]
                    p_inelastic = yields[2] / yields[0]
            seed = seed_table[SEED_STREAMS * counter[0] + SEE_STREAM]
            rng = wp.rand_init(seed, p)
            wp.randf(rng)     # consume the count draw of the sample kernel (same stream position)
            base = slots[0] + offsets[p]
            thermal = params[P_THERMAL]
            col = wp.clamp(int((z_new - z_min) / dz), 0, nz - 1)
            for k in range(n):
                d = base + k
                u_c = F64(wp.randf(rng))
                u1 = F64(wp.randf(rng))
                u2 = F64(wp.randf(rng))
                u3 = wp.max(F64(wp.randf(rng)), F64(1.0e-30))
                u4 = wp.max(F64(wp.randf(rng)), F64(1.0e-30))
                u5 = F64(wp.randf(rng))
                v = wp.vec3d(F64(0.0), F64(0.0), F64(0.0))
                is_back = 0
                if u_c < p_elastic + p_inelastic:
                    is_back = 1
                    speed_k = speed
                    if u_c >= p_elastic:
                        speed_k = wp.sqrt(u3) * speed      # inelastic: energy uniform in (0, E)
                    cos_t = wp.sqrt(u1)
                    sin_t = wp.sqrt(wp.max(F64(1.0) - u1, F64(0.0)))
                    phi = F64(6.283185307179586) * u2
                    v = see_orient(speed_k * cos_t, speed_k * sin_t * wp.cos(phi), speed_k * sin_t * wp.sin(phi), code)
                else:
                    v_n = thermal * wp.sqrt(F64(-2.0) * wp.log(u3))
                    rad = wp.sqrt(F64(-2.0) * wp.log(u4))
                    v = see_orient(v_n, thermal * rad * wp.cos(F64(6.283185307179586) * u5), thermal * rad * wp.sin(F64(6.283185307179586) * u5), code)
                if d >= e_capacity:
                    wp.atomic_add(stats, STATS_OVERFLOW, F64(1.0))
                else:
                    out_r[d] = r_e
                    out_z[d] = z_e
                    out_vr[d] = v[0]
                    out_vt[d] = v[1]
                    out_vz[d] = v[2]
                    out_alive[d] = 1
                    k_e = kinetic_energy(v[0], v[1], v[2], electron_mass_weight)
                    emitted += F64(1.0)
                    back += F64(is_back)
                    ke += k_e
                    pz += electron_mass_weight * v[2]
                    if accumulate != 0:
                        wp.atomic_add(see_column_energy, col, k_e)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(emitted)), offset=emitted_slot)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(back)), offset=STATS_SEE_BACKSCATTERED)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(ke)), offset=STATS_SEE_KE)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(pz)), offset=STATS_SEE_PZ)

    @wp.kernel
    def see_commit_kernel(count: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32),
                          flag_bound: int, e_capacity: int):
        total = 0
        if flag_bound > 0:
            total = offsets[flag_bound - 1] + count[flag_bound - 1]
        slots[0] = wp.min(slots[0] + total, e_capacity)

    def launch_see_stage(backend: Any, species: Any, slot: int, dim: int, *, is_ion: bool, accumulate: bool, species_dt: float) -> None:
        """Issue the three SEE launches (+ scan) of one species' wall impacts onto the backend's electron arrays."""

        config = backend.config
        grid = backend.masks.grid
        geometry = grid.geometry
        dev = backend.device
        electrons = backend.species["e"]
        hit = backend.see_hit_i if is_ion else backend.see_hit_e
        sp = backend.ion if is_ion else backend.electron
        emitted_slot = STATS_SEE_ION_EMITTED if is_ion else STATS_SEE_EMITTED
        if dim == 0:
            return
        wp.launch(
            see_sample_kernel, dim=padded_dim(dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
            inputs=[species.r, species.z, species.vr, species.vt, species.vz, hit, backend.see_count, backend.slots, slot,
                    backend.e_r, backend.e_z, backend.b_r, backend.b_z, grid.dr_m, grid.dz_m, geometry.z_min_m, backend.nr, backend.nz,
                    backend.plasma_cell, backend.plasma_node, sp.charge_c, sp.mass_kg, sp.macro_weight, species_dt, 1 if is_ion else 0,
                    backend.see_params, backend.seed_table, backend.step_counter, FIXED_POINT_SCALE, backend.acc_wall, backend.stats,
                    1 if accumulate else 0, backend.d_see_e, dim],
            device=dev,
        )
        wp.utils.array_scan(backend.see_count[:dim], backend.see_offsets[:dim], inclusive=False)
        wp.launch(
            see_spawn_kernel, dim=padded_dim(dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
            inputs=[species.r, species.z, species.vr, species.vt, species.vz, backend.see_count, backend.see_offsets, backend.slots,
                    electrons.capacity, backend.e_r, backend.e_z, backend.b_r, backend.b_z, grid.dr_m, grid.dz_m, geometry.z_min_m,
                    backend.nr, backend.nz, backend.plasma_cell, sp.charge_c, sp.mass_kg, sp.macro_weight, species_dt, 1 if is_ion else 0,
                    backend.see_params, backend.seed_table, backend.step_counter,
                    electrons.r, electrons.z, electrons.vr, electrons.vt, electrons.vz, electrons.alive_flags,
                    backend.stats, emitted_slot, ELECTRON_MASS_KG * config.macro_weight, 1 if accumulate else 0, backend.d_see_energy, dim],
            device=dev,
        )
        wp.launch(see_commit_kernel, dim=1, inputs=[backend.see_count, backend.see_offsets, backend.slots, dim, electrons.capacity], device=dev)


__all__ = [
    "P_SIZE",
    "SEE_STREAM",
    "launch_see_stage",
    "see_params_array",
]
