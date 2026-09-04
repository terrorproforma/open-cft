"""Warp kernels + device state of ``neutrals_spatial_v1`` / ``metastables_v1`` (model v2.5.0; device counterpart of
:mod:`cft_revival.pic2d.neutrals_spatial`).

Every launch has fixed arguments (array capacities, device-resident slot counts and carries), so the neutral sub-step is
captured inside the CUDA step graph as its own variant (the sub-step runs every ``substep_steps`` PIC steps).  The
arithmetic mirrors ``SpatialNeutrals.substep`` stage for stage (deplete -> spawn -> march -> deposit / publish); the random
streams are Warp's counter-based ``rand_init`` on the seed table's neutral column, so cross-backend parity is
distributional while the ATOM LEDGER identity holds exactly on both.

The per-cell density / metastable density / gas moments the electron and ion MCC kernels read live in device arrays
owned here and are rewritten at every sub-step (never a kernel scalar: the 2026-09-04 CUDA-graph lesson).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .neutrals_spatial import (
    NEUTRAL_SPATIAL_LEDGER_KEYS,
    SINK_FIXED_POINT,
    STATE_GROUND,
    STATE_METASTABLE,
    NeutralParticles,
    SpatialNeutrals,
    SpatialNeutralState,
)

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dependency
    wp = None  # type: ignore[assignment]

# neutral statistics slots (float64 device array, read at every host sync); the order of the first block follows
# NEUTRAL_SPATIAL_LEDGER_KEYS so the sync can add them by index
NSTATS_LEDGER = 0
NSTATS_OVERFLOW = len(NEUTRAL_SPATIAL_LEDGER_KEYS)
NSTATS_MARCH_UNRESOLVED = NSTATS_OVERFLOW + 1
NSTATS_SIZE = NSTATS_MARCH_UNRESOLVED + 1
_LEDGER_INDEX = {key: k for k, key in enumerate(NEUTRAL_SPATIAL_LEDGER_KEYS)}
SLOT_FED = _LEDGER_INDEX["neutral_fed"]
SLOT_RECYCLED = _LEDGER_INDEX["neutral_recycled"]
SLOT_FAST_IN = _LEDGER_INDEX["neutral_fast_in"]
SLOT_IONIZED = _LEDGER_INDEX["neutral_ionized"]
SLOT_CEX = _LEDGER_INDEX["neutral_cex_converted"]
SLOT_EXCITED = _LEDGER_INDEX["neutral_excited_to_pool"]
SLOT_EFFUSED = _LEDGER_INDEX["neutral_effused"]
SLOT_RETURNED = _LEDGER_INDEX["neutral_returned"]
SLOT_WALL_HITS = _LEDGER_INDEX["neutral_wall_hits"]
SLOT_ANODE_HITS = _LEDGER_INDEX["neutral_anode_hits"]
SLOT_REMOVED_G = _LEDGER_INDEX["neutral_removed_ground"]
SLOT_REMOVED_M = _LEDGER_INDEX["neutral_removed_meta"]
SLOT_PZ_EXIT = _LEDGER_INDEX["neutral_pz_exit"]
SLOT_KE_EXIT = _LEDGER_INDEX["neutral_ke_exit_j"]
SLOT_PZ_WALL = _LEDGER_INDEX["neutral_pz_wall"]
SLOT_CEILING = _LEDGER_INDEX["neutral_ceiling_violations"]
SLOT_SUBSTEPS = _LEDGER_INDEX["neutral_substeps"]
SLOT_META_PRODUCED = _LEDGER_INDEX["meta_produced"]
SLOT_META_IONIZED = _LEDGER_INDEX["meta_ionized"]
SLOT_META_SUPER = _LEDGER_INDEX["meta_superelastic"]
SLOT_META_WALL = _LEDGER_INDEX["meta_wall_deexcited"]
SLOT_META_RAD = _LEDGER_INDEX["meta_radiative"]
SLOT_META_EFFUSED = _LEDGER_INDEX["meta_effused"]
NEUTRAL_BLOCK = 256
# Deterministic (order-independent) integer accumulation: cell weight sums in units of macro_weight / 2**40, velocity moments in
# units of macro_weight x v_ref / 2**30 (v_ref = 1 km/s) and macro_weight x v_ref^2 / 2**30, the plasma's atom sinks as integer
# counts in units of the ion macro weight / 2**20 (so a branching fraction b_k enters as rint(b_k 2**20)).  Float atomics would make
# the published density depend on the thread order and break the graph-vs-direct (and run-to-run) bitwise replay.
WEIGHT_FIXED_POINT = float(2**40)
MOMENT_FIXED_POINT = float(2**30)
MOMENT_V_REF = 1.0e3

if wp is not None:
    F64 = wp.float64
    TWO_PI = 6.283185307179586

    @wp.func
    def cell_of(r: F64, z: F64, dr: F64, dz: F64, z_min: F64, nr: int, nz: int) -> int:
        i = wp.clamp(int(wp.floor(r / dr)), 0, nr - 1)
        j = wp.clamp(int(wp.floor((z - z_min) / dz)), 0, nz - 1)
        return i * nz + j

    @wp.func
    def gaussian(u1: F64, u2: F64) -> wp.vec2d:
        radius = wp.sqrt(F64(-2.0) * wp.log(wp.max(u1, F64(1.0e-300))))
        return wp.vec2d(radius * wp.cos(F64(TWO_PI) * u2), radius * wp.sin(F64(TWO_PI) * u2))

    # ------------------------------------------------------------------ stage 1: per-cell weight sums (int64 fixed point)
    @wp.kernel
    def neutral_cell_sum_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32),
        alive: wp.array(dtype=wp.int32), nslots: wp.array(dtype=wp.int32),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, weight_scale: F64,
        sum_g: wp.array(dtype=wp.int64), sum_m: wp.array(dtype=wp.int64),
    ):
        p = wp.tid()
        if p >= nslots[0] or alive[p] == 0:
            return
        c = cell_of(r[p], z[p], dr, dz, z_min, nr, nz)
        if state[p] == STATE_GROUND:
            wp.atomic_add(sum_g, c, wp.int64(wp.rint(w[p] * weight_scale)))
        else:
            wp.atomic_add(sum_m, c, wp.int64(wp.rint(w[p] * weight_scale)))

    # ------------------------------------------------------------------ stage 2: depletion factors, carries, spawn counts
    @wp.kernel
    def neutral_deplete_factor_kernel(
        sum_g_int: wp.array(dtype=wp.int64), sum_m_int: wp.array(dtype=wp.int64), weight_quantum: F64,
        sink_iz: wp.array(dtype=wp.int64), sink_cex: wp.array(dtype=wp.int64), sink_exc: wp.array(dtype=wp.int64),
        sink_miz: wp.array(dtype=wp.int64), sink_msuper: wp.array(dtype=wp.int64), recycle: wp.array(dtype=wp.int64), sink_quantum: F64,
        debt_g: wp.array(dtype=F64), debt_iz: wp.array(dtype=F64), debt_super: wp.array(dtype=F64),
        pending_feed: wp.array(dtype=F64), pending_recycle: wp.array(dtype=F64), pending_return: wp.array(dtype=F64),
        pending_meta: wp.array(dtype=F64), feed_share: wp.array(dtype=F64), feed_atoms: F64, f_acc: F64, w_n: F64, w_m: F64,
        meta_on: int, factor_g: wp.array(dtype=F64), factor_m: wp.array(dtype=F64),
        n_feed: wp.array(dtype=wp.int32), n_rec: wp.array(dtype=wp.int32), n_ret: wp.array(dtype=wp.int32),
        n_meta: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), nstats: wp.array(dtype=F64),
    ):
        c = wp.tid()
        sum_g = F64(sum_g_int[c]) * weight_quantum
        sum_m = F64(sum_m_int[c]) * weight_quantum
        s_iz = F64(sink_iz[c]) * sink_quantum
        s_cex = F64(sink_cex[c]) * sink_quantum
        s_exc = F64(sink_exc[c]) * sink_quantum
        s_miz = F64(sink_miz[c]) * sink_quantum
        s_msuper = F64(sink_msuper[c]) * sink_quantum
        s_rec = F64(recycle[c]) * sink_quantum
        # ground: demanded (x F) + debt against the cell weight
        demand_g = f_acc * (s_iz + s_cex + s_exc) + debt_g[c]
        removed_g = wp.min(demand_g, sum_g)
        debt_g[c] = demand_g - removed_g
        fg = F64(1.0)
        if sum_g > F64(0.0):
            fg = F64(1.0) - removed_g / sum_g
        factor_g[c] = fg
        # metastables: two demands (stepwise, superelastic) share the removal in proportion
        demand_iz = f_acc * s_miz + debt_iz[c]
        demand_super = f_acc * s_msuper + debt_super[c]
        demand_m = demand_iz + demand_super
        removed_m = wp.min(demand_m, sum_m)
        fm = F64(1.0)
        if sum_m > F64(0.0):
            fm = F64(1.0) - removed_m / sum_m
        factor_m[c] = fm
        share_super = F64(0.0)
        if demand_m > F64(0.0):
            share_super = demand_super / demand_m
        removed_super = removed_m * share_super
        removed_iz = removed_m - removed_super
        debt_iz[c] = demand_iz - removed_iz
        debt_super[c] = demand_super - removed_super
        # ledger (demanded atoms; carries stay in the true count)
        wp.atomic_add(nstats, SLOT_IONIZED, f_acc * s_iz)
        wp.atomic_add(nstats, SLOT_CEX, f_acc * s_cex)
        wp.atomic_add(nstats, SLOT_EXCITED, f_acc * s_exc)
        wp.atomic_add(nstats, SLOT_META_IONIZED, f_acc * s_miz)
        wp.atomic_add(nstats, SLOT_META_SUPER, f_acc * s_msuper)
        wp.atomic_add(nstats, SLOT_REMOVED_G, removed_g)
        wp.atomic_add(nstats, SLOT_REMOVED_M, removed_m)
        wp.atomic_add(nstats, SLOT_RECYCLED, f_acc * s_rec)
        wp.atomic_add(nstats, SLOT_RETURNED, removed_super)
        wp.atomic_add(nstats, SLOT_META_PRODUCED, f_acc * s_exc)
        # sources at the macro-weight granularity
        pf = pending_feed[c] + feed_atoms * feed_share[c]
        pr = pending_recycle[c] + f_acc * s_rec
        pt = pending_return[c] + removed_super
        pm = pending_meta[c] + f_acc * s_exc
        kf = int(wp.floor(pf / w_n))
        kr = int(wp.floor(pr / w_n))
        kt = int(wp.floor(pt / w_n))
        km = int(0)
        if meta_on != 0:
            km = int(wp.floor(pm / w_m))
        pending_feed[c] = pf - F64(kf) * w_n
        pending_recycle[c] = pr - F64(kr) * w_n
        pending_return[c] = pt - F64(kt) * w_n
        pending_meta[c] = pm - F64(km) * w_m
        n_feed[c] = kf
        n_rec[c] = kr
        n_ret[c] = kt
        n_meta[c] = km
        counts[c] = kf + kr + kt + km
        # the sinks are consumed
        sink_iz[c] = wp.int64(0)
        sink_cex[c] = wp.int64(0)
        sink_exc[c] = wp.int64(0)
        sink_miz[c] = wp.int64(0)
        sink_msuper[c] = wp.int64(0)
        recycle[c] = wp.int64(0)

    @wp.kernel
    def neutral_deplete_apply_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32),
        alive: wp.array(dtype=wp.int32), nslots: wp.array(dtype=wp.int32),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, factor_g: wp.array(dtype=F64), factor_m: wp.array(dtype=F64),
    ):
        p = wp.tid()
        if p >= nslots[0] or alive[p] == 0:
            return
        c = cell_of(r[p], z[p], dr, dz, z_min, nr, nz)
        if state[p] == STATE_GROUND:
            w[p] = w[p] * factor_g[c]
        else:
            w[p] = w[p] * factor_m[c]

    # ------------------------------------------------------------------ stage 3: spawn (feed, recycling, return, metastables)
    @wp.kernel
    def neutral_spawn_kernel(
        counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32),
        n_feed: wp.array(dtype=wp.int32), n_rec: wp.array(dtype=wp.int32), n_ret: wp.array(dtype=wp.int32), n_meta: wp.array(dtype=wp.int32),
        nslots: wp.array(dtype=wp.int32), capacity: int, seed_table: wp.array(dtype=wp.int32), seed_streams: int, stream: int,
        counter: wp.array(dtype=wp.int32),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, thermal_feed: F64, thermal_wall: F64, w_n: F64, w_m: F64,
        drift_r: wp.array(dtype=F64), drift_t: wp.array(dtype=F64), drift_z: wp.array(dtype=F64), thermal_cell: wp.array(dtype=F64),
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32), alive: wp.array(dtype=wp.int32), nstats: wp.array(dtype=F64),
    ):
        c = wp.tid()
        n = counts[c]
        if n == 0:
            return
        base = nslots[0] + offsets[c]
        i = c / nz
        j = c - i * nz
        r_lo = F64(i) * dr
        r_hi = r_lo + dr
        seed = seed_table[seed_streams * counter[0] + stream]
        rng = wp.rand_init(seed, c)
        kf = n_feed[c]
        kr = n_rec[c]
        kt = n_ret[c]
        k = int(0)
        while k < n:
            d = base + k
            if d >= capacity:
                wp.atomic_add(nstats, NSTATS_OVERFLOW, F64(1.0))
            else:
                u0 = F64(wp.randf(rng))
                u1 = F64(wp.randf(rng))
                u2 = F64(wp.randf(rng))
                u3 = F64(wp.randf(rng))
                u4 = F64(wp.randf(rng))
                u5 = F64(wp.randf(rng))
                rr = wp.sqrt(r_lo * r_lo + u0 * (r_hi * r_hi - r_lo * r_lo)) * (F64(1.0) - F64(1.0e-12))
                g12 = gaussian(u2, u3)
                g34 = gaussian(u4, u5)
                if k < kf:
                    # feed: at the anode face, cosine-law +z at the feed temperature
                    r[d] = rr
                    z[d] = z_min + F64(1.0e-6) * dz
                    vr[d] = thermal_feed * g12[0]
                    vt[d] = thermal_feed * g12[1]
                    vz[d] = thermal_feed * wp.sqrt(F64(-2.0) * wp.log(wp.max(u1, F64(1.0e-300))))
                    w[d] = w_n
                    state[d] = STATE_GROUND
                elif k < kf + kr:
                    # recycled wall ions: thermal atoms at T_w, uniform in the impact cell
                    r[d] = rr
                    z[d] = z_min + (F64(j) + u1) * dz
                    vr[d] = thermal_wall * g12[0]
                    vt[d] = thermal_wall * g12[1]
                    vz[d] = thermal_wall * g34[0]
                    w[d] = w_n
                    state[d] = STATE_GROUND
                elif k < kf + kr + kt:
                    # superelastic return: ground atoms at the feed temperature
                    r[d] = rr
                    z[d] = z_min + (F64(j) + u1) * dz
                    vr[d] = thermal_feed * g12[0]
                    vt[d] = thermal_feed * g12[1]
                    vz[d] = thermal_feed * g34[0]
                    w[d] = w_n
                    state[d] = STATE_GROUND
                else:
                    # metastables: the local gas velocity distribution (drift + thermal speed; feed temperature when empty)
                    th = thermal_cell[c]
                    if th <= F64(0.0):
                        th = thermal_feed
                    r[d] = rr
                    z[d] = z_min + (F64(j) + u1) * dz
                    vr[d] = drift_r[c] + th * g12[0]
                    vt[d] = drift_t[c] + th * g12[1]
                    vz[d] = drift_z[c] + th * g34[0]
                    w[d] = w_m
                    state[d] = STATE_METASTABLE
                alive[d] = 1
            k = k + 1

    @wp.kernel
    def neutral_spawn_commit_kernel(counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), n_cells: int,
                                    nslots: wp.array(dtype=wp.int32), capacity: int):
        total = offsets[n_cells - 1] + counts[n_cells - 1]
        nslots[0] = wp.min(nslots[0] + total, capacity)

    # fast neutrals from the ion MCC (per ion slot flag + stored velocity), appended after every ion MCC launch
    @wp.kernel
    def fast_neutral_spawn_kernel(
        fast_flag: wp.array(dtype=wp.int32), fast_offsets: wp.array(dtype=wp.int32), ion_slots: wp.array(dtype=wp.int32),
        ion_r: wp.array(dtype=F64), ion_z: wp.array(dtype=F64),
        fast_vr: wp.array(dtype=F64), fast_vt: wp.array(dtype=F64), fast_vz: wp.array(dtype=F64), weight: F64,
        nslots: wp.array(dtype=wp.int32), capacity: int,
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32), alive: wp.array(dtype=wp.int32), nstats: wp.array(dtype=F64),
    ):
        # ``fast_offsets`` is the INCLUSIVE scan of the flags: this slot's neutral index is its inclusive offset - 1 and the
        # total handed over is the last entry (independent of the flags, which this kernel clears)
        p = wp.tid()
        if p >= ion_slots[1] or fast_flag[p] == 0:
            return
        d = nslots[0] + fast_offsets[p] - 1
        fast_flag[p] = 0
        if d >= capacity:
            wp.atomic_add(nstats, NSTATS_OVERFLOW, F64(1.0))
            return
        r[d] = ion_r[p]
        z[d] = ion_z[p]
        vr[d] = fast_vr[p]
        vt[d] = fast_vt[p]
        vz[d] = fast_vz[p]
        w[d] = weight
        state[d] = STATE_GROUND
        alive[d] = 1
        wp.atomic_add(nstats, SLOT_FAST_IN, weight)

    @wp.kernel
    def fast_neutral_commit_kernel(fast_offsets: wp.array(dtype=wp.int32), ion_slots: wp.array(dtype=wp.int32),
                                   nslots: wp.array(dtype=wp.int32), capacity: int):
        # inclusive scan: the last entry is the number of flagged ion slots
        n = ion_slots[1]
        total = int(0)
        if n > 0:
            total = fast_offsets[n - 1]
        nslots[0] = wp.min(nslots[0] + total, capacity)

    # ------------------------------------------------------------------ stage 4: free flight with wall reflection
    @wp.kernel
    def neutral_march_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32), alive: wp.array(dtype=wp.int32), nslots: wp.array(dtype=wp.int32),
        seed_table: wp.array(dtype=wp.int32), seed_streams: int, stream: int, counter: wp.array(dtype=wp.int32), rng_offset: int,
        plasma_cell: wp.array(dtype=wp.int32), dr: F64, dz: F64, z_min: F64, z_exit: F64, z_domain_max: F64, r_exit: F64, r_max: F64,
        nr: int, nz: int, has_plume: int, substep_dt: F64, march_step: F64, mass: F64,
        accommodation: F64, thermal_wall: F64, meta_on: int, wall_deexcitation: F64, radiative_survival: F64, max_pieces: int,
        nstats: wp.array(dtype=F64),
    ):
        p = wp.tid()
        active = 0
        if p < nslots[0]:
            if alive[p] != 0:
                active = 1
        effused_g = F64(0.0)
        effused_m = F64(0.0)
        pz_exit = F64(0.0)
        ke_exit = F64(0.0)
        wall_hits = F64(0.0)
        anode_hits = F64(0.0)
        pz_wall = F64(0.0)
        meta_wall = F64(0.0)
        meta_rad = F64(0.0)
        unresolved = F64(0.0)
        if active != 0:
            seed = seed_table[seed_streams * counter[0] + stream]
            rng = wp.rand_init(seed, p + rng_offset)
            rp = r[p]
            zp = z[p]
            vrp = vr[p]
            vtp = vt[p]
            vzp = vz[p]
            wp_ = w[p]
            st = state[p]
            if meta_on != 0 and st == STATE_METASTABLE and radiative_survival < F64(1.0):
                if F64(wp.randf(rng)) > radiative_survival:
                    st = STATE_GROUND
                    meta_rad = wp_
            remaining = F64(substep_dt)
            pieces = int(0)
            done = int(0)
            while done == 0:
                speed = wp.sqrt(vrp * vrp + vtp * vtp + vzp * vzp)
                dt = remaining
                if speed > F64(0.0):
                    dt = wp.min(remaining, march_step / speed)
                x_new = rp + vrp * dt
                y_new = vtp * dt
                r_new = wp.sqrt(x_new * x_new + y_new * y_new)
                cos_a = F64(1.0)
                sin_a = F64(0.0)
                if r_new > F64(0.0):
                    cos_a = x_new / r_new
                    sin_a = y_new / r_new
                vr_new = vrp * cos_a + vtp * sin_a
                vt_new = -vrp * sin_a + vtp * cos_a
                z_new = zp + vzp * dt
                i0 = wp.clamp(int(wp.floor(rp / dr)), 0, nr - 1)
                j0 = wp.clamp(int(wp.floor((zp - z_min) / dz)), 0, nz - 1)
                exited = 0
                if z_new >= z_domain_max:
                    if has_plume != 0:
                        exited = 1
                    elif r_new < r_exit:
                        exited = 1
                if has_plume != 0 and z_new >= z_exit and r_new >= r_max:
                    exited = 1
                is_anode = 0
                if z_new < z_min:
                    is_anode = 1
                i1 = int(wp.floor(r_new / dr))
                j1 = wp.clamp(int(wp.floor((z_new - z_min) / dz)), 0, nz - 1)
                inside = 0
                if exited == 0 and is_anode == 0 and i1 < nr:
                    if plasma_cell[i1 * nz + j1] != 0:
                        inside = 1
                if exited != 0:
                    alive[p] = 0
                    if st == STATE_GROUND:
                        effused_g = wp_
                    else:
                        effused_m = wp_
                    pz_exit = wp_ * mass * vzp
                    ke_exit = F64(0.5) * mass * wp_ * (vr_new * vr_new + vt_new * vt_new + vzp * vzp)
                    done = 1
                elif inside != 0:
                    rp = r_new
                    zp = z_new
                    vrp = vr_new
                    vtp = vt_new
                    remaining = remaining - dt
                else:
                    # wall / anode reflection at the pre-move position (still inside the plasma)
                    normal_r = F64(0.0)
                    normal_z = F64(0.0)
                    if is_anode != 0:
                        normal_z = F64(1.0)
                        anode_hits += wp_
                    elif i1 != i0 or i1 >= nr:
                        if i1 > i0:
                            normal_r = F64(-1.0)
                        else:
                            normal_r = F64(1.0)
                    else:
                        if j1 < j0:
                            normal_z = F64(1.0)
                        else:
                            normal_z = F64(-1.0)
                    wall_hits += wp_
                    pz_wall += wp_ * mass * vzp
                    u0 = F64(wp.randf(rng))
                    u1 = F64(wp.randf(rng))
                    u2 = F64(wp.randf(rng))
                    u3 = F64(wp.randf(rng))
                    u4 = F64(wp.randf(rng))
                    if u0 < accommodation:
                        vn = thermal_wall * wp.sqrt(F64(-2.0) * wp.log(wp.max(u1, F64(1.0e-300))))
                        g = gaussian(u2, u3)
                        if normal_r != F64(0.0):
                            vrp = normal_r * vn
                            vzp = thermal_wall * g[0]
                        else:
                            vzp = normal_z * vn
                            vrp = thermal_wall * g[0]
                        vtp = thermal_wall * g[1]
                    else:
                        if normal_r != F64(0.0):
                            vrp = -vrp
                        if normal_z != F64(0.0):
                            vzp = -vzp
                    pz_wall -= wp_ * mass * vzp
                    if meta_on != 0 and st == STATE_METASTABLE:
                        if u4 < wall_deexcitation:
                            st = STATE_GROUND
                            meta_wall += wp_
                    remaining = remaining - F64(1.0e-3) * dt
                pieces = pieces + 1
                if done == 0:
                    if remaining <= F64(1.0e-15) * substep_dt:
                        done = 1
                    elif pieces >= max_pieces:
                        done = 1
                        unresolved = F64(1.0)
            if alive[p] != 0:
                r[p] = rp
                z[p] = zp
                vr[p] = vrp
                vt[p] = vtp
                vz[p] = vzp
                state[p] = st
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(effused_g)), offset=SLOT_EFFUSED)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(effused_m)), offset=SLOT_META_EFFUSED)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(pz_exit)), offset=SLOT_PZ_EXIT)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(ke_exit)), offset=SLOT_KE_EXIT)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(wall_hits)), offset=SLOT_WALL_HITS)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(anode_hits)), offset=SLOT_ANODE_HITS)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(pz_wall)), offset=SLOT_PZ_WALL)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(meta_wall)), offset=SLOT_META_WALL)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(meta_rad)), offset=SLOT_META_RAD)
        wp.tile_atomic_add(nstats, wp.tile_sum(wp.tile(unresolved)), offset=NSTATS_MARCH_UNRESOLVED)

    # ------------------------------------------------------------------ stage 5: deposit + publish
    @wp.kernel
    def neutral_deposit_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32), alive: wp.array(dtype=wp.int32), nslots: wp.array(dtype=wp.int32),
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int, weight_scale: F64, moment_scale: F64, moment2_scale: F64,
        sum_g: wp.array(dtype=wp.int64), sum_m: wp.array(dtype=wp.int64), mom_r: wp.array(dtype=wp.int64), mom_t: wp.array(dtype=wp.int64),
        mom_z: wp.array(dtype=wp.int64), mom_v2: wp.array(dtype=wp.int64),
    ):
        p = wp.tid()
        if p >= nslots[0] or alive[p] == 0:
            return
        c = cell_of(r[p], z[p], dr, dz, z_min, nr, nz)
        wpp = w[p]
        if state[p] == STATE_GROUND:
            wp.atomic_add(sum_g, c, wp.int64(wp.rint(wpp * weight_scale)))
            wp.atomic_add(mom_r, c, wp.int64(wp.rint(wpp * vr[p] * moment_scale)))
            wp.atomic_add(mom_t, c, wp.int64(wp.rint(wpp * vt[p] * moment_scale)))
            wp.atomic_add(mom_z, c, wp.int64(wp.rint(wpp * vz[p] * moment_scale)))
            wp.atomic_add(mom_v2, c, wp.int64(wp.rint(wpp * (vr[p] * vr[p] + vt[p] * vt[p] + vz[p] * vz[p]) * moment2_scale)))
        else:
            wp.atomic_add(sum_m, c, wp.int64(wp.rint(wpp * weight_scale)))

    @wp.kernel
    def neutral_publish_kernel(
        sum_g_int: wp.array(dtype=wp.int64), sum_m_int: wp.array(dtype=wp.int64), mom_r: wp.array(dtype=wp.int64), mom_t: wp.array(dtype=wp.int64),
        mom_z: wp.array(dtype=wp.int64), mom_v2: wp.array(dtype=wp.int64), weight_quantum: F64, moment_quantum: F64, moment2_quantum: F64,
        inv_volume: wp.array(dtype=F64), ceiling: F64,
        density: wp.array(dtype=F64), meta_density: wp.array(dtype=F64),
        drift_r: wp.array(dtype=F64), drift_t: wp.array(dtype=F64), drift_z: wp.array(dtype=F64), thermal: wp.array(dtype=F64),
        accumulate: int, d_density: wp.array(dtype=F64), d_meta: wp.array(dtype=F64), nstats: wp.array(dtype=F64),
    ):
        c = wp.tid()
        sum_g = F64(sum_g_int[c]) * weight_quantum
        sum_m = F64(sum_m_int[c]) * weight_quantum
        n = sum_g * inv_volume[c]
        if n > ceiling * (F64(1.0) + F64(1.0e-12)):
            wp.atomic_add(nstats, SLOT_CEILING, F64(1.0))
        density[c] = wp.min(n, ceiling)
        meta_density[c] = sum_m * inv_volume[c]
        ur = F64(0.0)
        ut = F64(0.0)
        uz = F64(0.0)
        th = F64(0.0)
        if sum_g > F64(0.0):
            ur = F64(mom_r[c]) * moment_quantum / sum_g
            ut = F64(mom_t[c]) * moment_quantum / sum_g
            uz = F64(mom_z[c]) * moment_quantum / sum_g
            th = wp.sqrt(wp.max(F64(mom_v2[c]) * moment2_quantum / sum_g - (ur * ur + ut * ut + uz * uz), F64(0.0)) / F64(3.0))
        drift_r[c] = ur
        drift_t[c] = ut
        drift_z[c] = uz
        thermal[c] = th
        if accumulate != 0:
            d_density[c] = d_density[c] + density[c]
            d_meta[c] = d_meta[c] + meta_density[c]
        if c == 0:
            nstats[SLOT_SUBSTEPS] = nstats[SLOT_SUBSTEPS] + F64(1.0)

    @wp.kernel
    def neutral_atoms_kernel(
        w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32), alive: wp.array(dtype=wp.int32), nslots: wp.array(dtype=wp.int32),
        out: wp.array(dtype=F64),
    ):
        p = wp.tid()
        g = F64(0.0)
        m = F64(0.0)
        cg = F64(0.0)
        cm = F64(0.0)
        if p < nslots[0]:
            if alive[p] != 0:
                if state[p] == STATE_GROUND:
                    g = w[p]
                    cg = F64(1.0)
                else:
                    m = w[p]
                    cm = F64(1.0)
        wp.tile_atomic_add(out, wp.tile_sum(wp.tile(g)), offset=0)
        wp.tile_atomic_add(out, wp.tile_sum(wp.tile(m)), offset=1)
        wp.tile_atomic_add(out, wp.tile_sum(wp.tile(cg)), offset=2)
        wp.tile_atomic_add(out, wp.tile_sum(wp.tile(cm)), offset=3)

    @wp.kernel
    def neutral_compact_kernel(
        alive: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32), nslots: wp.array(dtype=wp.int32),
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        w: wp.array(dtype=F64), state: wp.array(dtype=wp.int32),
        r2: wp.array(dtype=F64), z2: wp.array(dtype=F64), vr2: wp.array(dtype=F64), vt2: wp.array(dtype=F64), vz2: wp.array(dtype=F64),
        w2: wp.array(dtype=F64), state2: wp.array(dtype=wp.int32), alive2: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        if p >= nslots[0] or alive[p] == 0:
            return
        d = offsets[p]
        r2[d] = r[p]
        z2[d] = z[p]
        vr2[d] = vr[p]
        vt2[d] = vt[p]
        vz2[d] = vz[p]
        w2[d] = w[p]
        state2[d] = state[p]
        alive2[d] = 1


def padded(count: int, block: int = NEUTRAL_BLOCK) -> int:
    return ((max(int(count), 1) + block - 1) // block) * block


class WarpNeutralModel:
    """Device state and launch sequence of the spatial neutral model for ``WarpBackend``."""

    def __init__(self, spatial: SpatialNeutrals, device: Any, *, ion_capacity: int, seed_streams: int, stream_column: int,
                 stream_id: int) -> None:
        self.spatial = spatial
        self.config = spatial.config
        self.device = device
        self.nr, self.nz = spatial.nr, spatial.nz
        self.n_cells = spatial.n_cells
        self.seed_streams = seed_streams
        self.stream_column = stream_column
        self.stream_id = stream_id
        dev = device
        f64 = lambda values: wp.array(np.ascontiguousarray(np.asarray(values, dtype=np.float64)), dtype=wp.float64, device=dev)  # noqa: E731
        cells = lambda dtype=wp.float64: wp.zeros(self.n_cells, dtype=dtype, device=dev)  # noqa: E731
        # per-cell fields (the plasma's sinks and the deposit sums are int64 fixed point: order-independent, bitwise replay)
        self.sink_iz, self.sink_cex, self.sink_exc, self.sink_miz, self.sink_msuper, self.recycle = (cells(wp.int64) for _ in range(6))
        self.debt_g, self.debt_iz, self.debt_super = (cells() for _ in range(3))
        self.pending_feed, self.pending_recycle, self.pending_return, self.pending_meta = (cells() for _ in range(4))
        self.density, self.meta_density, self.drift_r, self.drift_t, self.drift_z, self.thermal = (cells() for _ in range(6))
        self.sum_g, self.sum_m, self.mom_r, self.mom_t, self.mom_z, self.mom_v2 = (cells(wp.int64) for _ in range(6))
        self.factor_g, self.factor_m = cells(), cells()
        self.counts, self.offsets, self.n_feed, self.n_rec, self.n_ret, self.n_meta = (cells(wp.int32) for _ in range(6))
        self.weight_scale = WEIGHT_FIXED_POINT / spatial.config.macro_weight
        self.moment_scale = MOMENT_FIXED_POINT / (spatial.config.macro_weight * MOMENT_V_REF)
        self.moment2_scale = MOMENT_FIXED_POINT / (spatial.config.macro_weight * MOMENT_V_REF**2)
        self.sink_quantum = spatial.ion_macro_weight / SINK_FIXED_POINT
        self.d_density, self.d_meta = cells(), cells()
        with np.errstate(divide="ignore"):
            inv_volume = np.where(spatial.cell_volume > 0.0, 1.0 / np.where(spatial.cell_volume > 0.0, spatial.cell_volume, 1.0), 0.0)
        self.inv_volume = f64(inv_volume)
        self.feed_share = f64(spatial.feed_share)
        branching = spatial.config.metastables.branching if spatial.config.metastables is not None else (0.0,)
        self.branching_units = wp.array(np.rint(np.asarray(branching) * SINK_FIXED_POINT).astype(np.int64), dtype=wp.int64, device=dev)
        self.nstats = wp.zeros(NSTATS_SIZE, dtype=wp.float64, device=dev)
        self.nslots = wp.zeros(1, dtype=wp.int32, device=dev)
        self.atoms_out = wp.zeros(4, dtype=wp.float64, device=dev)
        # particle arrays
        self.capacity = 0
        self.alive_count = 0
        self.arrays: list[Any] = []
        self.tmp: list[Any] = []
        self._allocate(1024)
        # fast-neutral staging per ion slot
        self.fast_capacity = 0
        self.fast_flag = self.fast_offsets = None
        self.fast_v: list[Any] = []
        self.ensure_fast_capacity(ion_capacity)
        self.neutral_samples = 0
        self.neutral_time_s = 0.0
        self.substeps = 0
        self.metastable_weight = spatial.metastable_weight
        self.march_limit = 64 * (self.nr + self.nz) + 64
        self.radiative_survival = 1.0
        if spatial.config.metastables is not None and spatial.config.metastables.radiative_decay_rate_per_s > 0.0:
            self.radiative_survival = float(np.exp(-spatial.config.metastables.radiative_decay_rate_per_s * spatial.substep_dt_s))

    # ------------------------------------------------------------------ allocation
    def _allocate(self, capacity: int) -> None:
        dev = self.device
        self.capacity = int(capacity)
        self.arrays = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(6)] + [wp.zeros(capacity, dtype=wp.int32, device=dev) for _ in range(2)]
        self.tmp = [wp.zeros(capacity, dtype=wp.float64, device=dev) for _ in range(6)] + [wp.zeros(capacity, dtype=wp.int32, device=dev) for _ in range(2)]
        self.scan_offsets = wp.zeros(capacity, dtype=wp.int32, device=dev)

    @property
    def r(self):
        return self.arrays[0]

    @property
    def z(self):
        return self.arrays[1]

    @property
    def vr(self):
        return self.arrays[2]

    @property
    def vt(self):
        return self.arrays[3]

    @property
    def vz(self):
        return self.arrays[4]

    @property
    def w(self):
        return self.arrays[5]

    @property
    def state(self):
        return self.arrays[6]

    @property
    def alive(self):
        return self.arrays[7]

    def ensure_fast_capacity(self, ion_capacity: int) -> bool:
        """Per-ion-slot staging of the CEX hand-off; returns True when reallocated (graphs must be recaptured)."""

        if ion_capacity <= self.fast_capacity:
            return False
        dev = self.device
        self.fast_capacity = int(ion_capacity)
        self.fast_flag = wp.zeros(ion_capacity, dtype=wp.int32, device=dev)
        self.fast_offsets = wp.zeros(ion_capacity, dtype=wp.int32, device=dev)
        self.fast_v = [wp.zeros(ion_capacity, dtype=wp.float64, device=dev) for _ in range(3)]
        return True

    def grow(self, minimum: int) -> bool:
        """Reallocate (at a sync) so that ``capacity >= minimum``; copies the live slots.  Returns True when reallocated."""

        if minimum <= self.capacity:
            return False
        capacity = max(int(minimum), int(1.5 * self.capacity), 1024)
        old = self.arrays
        used = int(self.nslots.numpy()[0])
        self._allocate(capacity)
        if used:
            for src, dst in zip(old, self.arrays, strict=True):
                wp.copy(dst, src, count=used)
        return True

    def reserve(self, sync_interval: int, ion_bound: int) -> bool:
        """Capacity for the next sync interval: live slots + every spawn the interval can produce (fail-closed overflow otherwise)."""

        used = int(self.nslots.numpy()[0])
        substeps = sync_interval // self.config.substep_steps
        feed_per_substep = int(np.ceil(self.config.feed_atoms_per_s * self.spatial.substep_dt_s / self.config.macro_weight)) + int(np.count_nonzero(self.spatial.feed_share))
        # recycling / return / metastable spawns are bounded by the plasma; reserve a generous margin and fail closed on overflow
        margin = substeps * (feed_per_substep + 2 * self.n_cells) + int(0.05 * ion_bound) + 1024
        return self.grow(used + margin)

    # ------------------------------------------------------------------ state exchange
    def upload(self, state: SpatialNeutralState) -> None:
        particles = state.particles
        n = particles.count
        self.grow(max(2 * n, 1024))
        for array in self.arrays:
            array.zero_()
        if n:
            for values, target in zip(particles.arrays(), self.arrays[:7], strict=True):
                dtype = wp.int32 if target.dtype == wp.int32 else wp.float64
                np_dtype = np.int32 if dtype == wp.int32 else np.float64
                wp.copy(target, wp.array(np.ascontiguousarray(values.astype(np_dtype)), dtype=dtype, device=self.device), count=n)
            wp.copy(self.alive, wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=self.device), count=n)
        wp.copy(self.nslots, wp.array(np.array([n], dtype=np.int32), dtype=wp.int32, device=self.device))
        self.alive_count = n
        for name, target in zip(SpatialNeutralState.CELL_ARRAY_KEYS, (
                self.debt_g, self.debt_iz, self.debt_super, self.pending_feed, self.pending_recycle, self.pending_return, self.pending_meta,
                self.density, self.meta_density, self.drift_r, self.drift_t, self.drift_z, self.thermal), strict=True):
            wp.copy(target, wp.array(np.ascontiguousarray(getattr(state, name)), dtype=wp.float64, device=self.device))
        for array in (self.sink_iz, self.sink_cex, self.sink_exc, self.sink_miz, self.sink_msuper, self.recycle):
            array.zero_()
        self.neutral_time_s = float(state.neutral_time_s)
        self.substeps = int(state.substeps)
        self.nstats.zero_()

    def download(self) -> SpatialNeutralState:
        """Host copy after a sync (compacted); the published fields and carries travel with it."""

        n = self.alive_count
        if n:
            values = [np.asarray(a.numpy()[:n]).copy() for a in self.arrays[:7]]
            particles = NeutralParticles(*values[:6], values[6].astype(np.int32))
        else:
            particles = NeutralParticles.empty()
        cell_arrays = [a.numpy().copy() for a in (self.debt_g, self.debt_iz, self.debt_super, self.pending_feed, self.pending_recycle,
                                                    self.pending_return, self.pending_meta, self.density, self.meta_density,
                                                    self.drift_r, self.drift_t, self.drift_z, self.thermal)]
        return SpatialNeutralState(particles, *cell_arrays, self.neutral_time_s, self.substeps)

    def atom_sums(self) -> tuple[float, float, int, int]:
        self.atoms_out.zero_()
        used = int(self.nslots.numpy()[0])
        if used:
            wp.launch(neutral_atoms_kernel, dim=padded(used), block_dim=NEUTRAL_BLOCK,
                      inputs=[self.w, self.state, self.alive, self.nslots, self.atoms_out], device=self.device)
        out = self.atoms_out.numpy()
        return float(out[0]), float(out[1]), int(out[2]), int(out[3])

    def compact(self) -> None:
        used = int(self.nslots.numpy()[0])
        if used == 0:
            self.alive_count = 0
            return
        wp.utils.array_scan(self.alive[:used], self.scan_offsets[:used], inclusive=True)
        total = int(self.scan_offsets.numpy()[used - 1])
        wp.utils.array_scan(self.alive[:used], self.scan_offsets[:used], inclusive=False)
        self.tmp[7].zero_()
        wp.launch(neutral_compact_kernel, dim=used, inputs=[self.alive, self.scan_offsets, self.nslots, *self.arrays[:7], *self.tmp[:7], self.tmp[7]],
                  device=self.device)
        self.alive.zero_()
        if total:
            for src, dst in zip(self.tmp, self.arrays, strict=True):
                wp.copy(dst, src, count=total)
        wp.copy(self.nslots, wp.array(np.array([total], dtype=np.int32), dtype=wp.int32, device=self.device))
        self.alive_count = total

    # ------------------------------------------------------------------ launches
    def launch_fast_hand_off(self, ions: Any, ion_slots: Any, i_dim: int, weight: float) -> None:
        """After an ion MCC launch: append the flagged CEX fast neutrals (weight = F x W) to the neutral arrays."""

        wp.utils.array_scan(self.fast_flag[:i_dim], self.fast_offsets[:i_dim], inclusive=True)
        wp.launch(fast_neutral_spawn_kernel, dim=i_dim,
                  inputs=[self.fast_flag, self.fast_offsets, ion_slots, ions.r, ions.z, *self.fast_v, weight, self.nslots, self.capacity,
                          *self.arrays[:7], self.alive, self.nstats], device=self.device)
        wp.launch(fast_neutral_commit_kernel, dim=1, inputs=[self.fast_offsets, ion_slots, self.nslots, self.capacity], device=self.device)

    def launch_substep(self, seed_table: Any, counter: Any, accumulate: bool, plasma_cell: Any) -> None:
        spatial = self.spatial
        grid = spatial.grid
        geometry = spatial.geometry
        dev = self.device
        cfg = self.config
        dim = padded(self.capacity)
        common = [grid.dr_m, grid.dz_m, geometry.z_min_m, self.nr, self.nz]
        # 1) per-cell weight sums of the current population
        self.sum_g.zero_()
        self.sum_m.zero_()
        wp.launch(neutral_cell_sum_kernel, dim=self.capacity,
                  inputs=[self.r, self.z, self.w, self.state, self.alive, self.nslots, *common, self.weight_scale, self.sum_g, self.sum_m], device=dev)
        # 2) depletion factors, debts, carries and spawn counts (consumes the sinks)
        meta_on = 1 if cfg.metastables is not None else 0
        wp.launch(neutral_deplete_factor_kernel, dim=self.n_cells,
                  inputs=[self.sum_g, self.sum_m, 1.0 / self.weight_scale, self.sink_iz, self.sink_cex, self.sink_exc, self.sink_miz, self.sink_msuper,
                          self.recycle, self.sink_quantum,
                          self.debt_g, self.debt_iz, self.debt_super, self.pending_feed, self.pending_recycle, self.pending_return, self.pending_meta,
                          self.feed_share, cfg.feed_atoms_per_s * spatial.substep_dt_s, cfg.time_acceleration, cfg.macro_weight,
                          self.metastable_weight, meta_on, self.factor_g, self.factor_m, self.n_feed, self.n_rec, self.n_ret, self.n_meta,
                          self.counts, self.nstats], device=dev)
        wp.launch(neutral_deplete_apply_kernel, dim=self.capacity,
                  inputs=[self.r, self.z, self.w, self.state, self.alive, self.nslots, *common, self.factor_g, self.factor_m], device=dev)
        wp.launch(_feed_ledger_kernel, dim=1, inputs=[self.nstats, cfg.feed_atoms_per_s * spatial.substep_dt_s], device=dev)
        # 3) spawn at the macro-weight granularity
        wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
        wp.launch(neutral_spawn_kernel, dim=self.n_cells,
                  inputs=[self.counts, self.offsets, self.n_feed, self.n_rec, self.n_ret, self.n_meta, self.nslots, self.capacity,
                          seed_table, self.seed_streams, self.stream_column, counter, *common, spatial.thermal_speed, spatial.wall_thermal_speed,
                          cfg.macro_weight, self.metastable_weight, self.drift_r, self.drift_t, self.drift_z, self.thermal,
                          *self.arrays[:7], self.alive, self.nstats], device=dev)
        wp.launch(neutral_spawn_commit_kernel, dim=1, inputs=[self.counts, self.offsets, self.n_cells, self.nslots, self.capacity], device=dev)
        # 4) free flight
        meta_cfg = cfg.metastables
        wp.launch(neutral_march_kernel, dim=dim, block_dim=NEUTRAL_BLOCK,
                  inputs=[*self.arrays[:7], self.alive, self.nslots, seed_table, self.seed_streams, self.stream_column, counter, self.n_cells,
                          plasma_cell, grid.dr_m, grid.dz_m, geometry.z_min_m, geometry.z_max_m, geometry.domain_z_max_m, geometry.exit_radius_m,
                          geometry.max_radius_m, self.nr, self.nz, 1 if geometry.has_plume else 0, spatial.substep_dt_s, spatial.march_step_m,
                          spatial.mass_kg, cfg.accommodation_coefficient, spatial.wall_thermal_speed, meta_on,
                          meta_cfg.wall_deexcitation_probability if meta_cfg is not None else 0.0, self.radiative_survival, self.march_limit,
                          self.nstats], device=dev)
        # 5) deposit + publish (the fields the MCC kernels read until the next sub-step)
        for array in (self.sum_g, self.sum_m, self.mom_r, self.mom_t, self.mom_z, self.mom_v2):
            array.zero_()
        wp.launch(neutral_deposit_kernel, dim=self.capacity,
                  inputs=[*self.arrays[:7], self.alive, self.nslots, *common, self.weight_scale, self.moment_scale, self.moment2_scale,
                          self.sum_g, self.sum_m, self.mom_r, self.mom_t, self.mom_z, self.mom_v2], device=dev)
        wp.launch(neutral_publish_kernel, dim=self.n_cells,
                  inputs=[self.sum_g, self.sum_m, self.mom_r, self.mom_t, self.mom_z, self.mom_v2, 1.0 / self.weight_scale, 1.0 / self.moment_scale,
                          1.0 / self.moment2_scale, self.inv_volume, spatial.ceiling,
                          self.density, self.meta_density, self.drift_r, self.drift_t, self.drift_z, self.thermal,
                          1 if accumulate else 0, self.d_density, self.d_meta, self.nstats], device=dev)

    def read_stats(self) -> dict[str, float]:
        """Host read of the neutral statistics (at the sync); zeroes them."""

        values = self.nstats.numpy()
        out = {key: float(values[k]) for k, key in enumerate(NEUTRAL_SPATIAL_LEDGER_KEYS)}
        out["overflow"] = float(values[NSTATS_OVERFLOW])
        out["march_unresolved"] = float(values[NSTATS_MARCH_UNRESOLVED])
        self.nstats.zero_()
        self.substeps += int(round(out["neutral_substeps"]))
        self.neutral_time_s += out["neutral_substeps"] * self.spatial.substep_dt_s
        return out

    def reset_diagnostics(self) -> None:
        self.d_density.zero_()
        self.d_meta.zero_()
        self.neutral_samples = 0


if wp is not None:

    @wp.kernel
    def _feed_ledger_kernel(nstats: wp.array(dtype=F64), feed_atoms: F64):
        nstats[SLOT_FED] = nstats[SLOT_FED] + feed_atoms


__all__ = ["NSTATS_SIZE", "WarpNeutralModel", "padded"]
