# ruff: noqa: SIM102  (nested single-condition ifs are kept deliberately inside the Warp kernels: one comparison per branch)
"""Warp kernels of the v2.4.0 Coulomb stage (``coulomb_v1``), shared by the Warp CPU and CUDA backends.

Design (physics contract = ``coulomb.py``, the numpy reference):

* PAIRING WITHOUT REORDERING.  The base RNG contract of this code keys every per-particle random stream (MCC,
  anomalous scattering, ion MCC, SEE) on the particle's SLOT index, so physically sorting the particle arrays
  would change every draw and break the bitwise replay of the recorded runs.  The stage therefore builds, per
  cycle, a cell-sorted PERMUTATION of the alive slots (``sorted[g]`` = slot at sorted position ``g``) and pairs
  through it; the particle arrays are never moved.  With Coulomb off nothing here runs and every configuration
  replays bitwise; with it on, the stage's own draws come from the dedicated seed-table stream (id 6, column 5).
* DETERMINISTIC COUNTING SORT.  ``cell_kernel`` writes the cell of every alive slot and its provisional position
  from an integer atomic (the counts are order-independent; the positions are not), an exclusive scan gives the
  segment starts, ``scatter_kernel`` fills a provisional segment list, and ``rank_kernel`` gives each slot its
  rank = number of same-cell slots with a smaller slot index - a deterministic order (slot order) whatever the
  atomic arrival order was - so the graph replays the direct launches bitwise and the same seed replays the
  same run.  Cost: one read of the segment per particle (occupancy k -> O(sum k^2), ~1-3 ms at 4.5 M particles
  on the H100 for both species; amortised over ``cycle_steps``).
* ``prepare_kernel`` (one thread per cell, sequential over the segment): the cell moments (count, temperature)
  for the Coulomb logarithm and the field density, the Fisher-Yates shuffle of the segment (random Takizuka-Abe
  partners; keyed on the cell index) and the electron-ion pairing shift of the cell.
* ``like_kernel`` (one thread per sorted position; the first member of each pair collides it): consecutive
  pairs of the shuffled segment, the Takizuka-Abe triplet for an odd count (one thread runs its three
  half-step sub-collisions sequentially, so no two threads touch the same particle).
* ``unlike_kernel`` (one thread per sorted ION position): the ion's electrons are the cell electrons
  ``l = l0 + m N_i`` with ``l0 = (i - shift) mod N_i``; the thread collides them sequentially against its ion
  (an electron belongs to exactly one ion, so no races), each at the field density ``n_i`` - every electron
  gets one collision, the ions ``N_e / N_i`` on average (both rates physical).
* Collision kinematics, deflection parameter, Nanbu angle and Coulomb logarithms are the ``wp.func`` ports of
  ``coulomb.py``; the pair conserves momentum and classical energy to round-off.  Tallies (pair counts, sum of
  ``s``, large-``s`` pairs, sum of ln Lambda, the relativistic pair energy change ``ke_coulomb_j`` and
  ``pz_coulomb``) are tile-reduced per block into the stage's own statistics array (float atomics: round-off
  order, diagnostics only - the particle state is deterministic); per-cell window sums feed the maps.

Every launch has a fixed shape (the array capacities / the cell count) and reads only device arrays, so the whole
stage is CUDA-graph capturable (``counts.zero_()`` + the scan are the same operations the MCC spawn uses).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .coulomb import COULOMB_KEYS, CoulombConfig, cell_maps_to_nodes, cell_volumes_m3
from .mesh import MeshMasks
from .models import Grid2D, ParticleArrays, PIC2DValidationError, Species2D

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dependency
    wp = None  # type: ignore[assignment]

COULOMB_STREAM = 5      # 6th per-step seed-table column (0 MCC, 1 injection, 2 anomalous, 3 ion-neutral MCC, 4 SEE, 5 Coulomb)

# statistics slots of the stage's own float64 array (read and zeroed at every host sync)
CS_EE_PAIRS, CS_EE_S, CS_EE_LARGE, CS_EE_LNL = 0, 1, 2, 3
CS_EI_PAIRS, CS_EI_S, CS_EI_LARGE, CS_EI_LNL = 4, 5, 6, 7
CS_II_PAIRS, CS_II_S, CS_II_LARGE, CS_II_LNL = 8, 9, 10, 11
CS_PZ = 12
CS_KE = 13
CS_E_CYCLES = 14
CS_I_CYCLES = 15
CS_CYCLES = 16
CS_SIZE = 17
CS_KEYS = {
    "coulomb_ee_pairs": CS_EE_PAIRS, "coulomb_ee_s_sum": CS_EE_S, "coulomb_ee_large_s": CS_EE_LARGE, "coulomb_ee_lnl_sum": CS_EE_LNL,
    "coulomb_ei_pairs": CS_EI_PAIRS, "coulomb_ei_s_sum": CS_EI_S, "coulomb_ei_large_s": CS_EI_LARGE, "coulomb_ei_lnl_sum": CS_EI_LNL,
    "coulomb_ii_pairs": CS_II_PAIRS, "coulomb_ii_s_sum": CS_II_S, "coulomb_ii_large_s": CS_II_LARGE, "coulomb_ii_lnl_sum": CS_II_LNL,
    "pz_coulomb": CS_PZ, "ke_coulomb_j": CS_KE, "coulomb_electron_cycles": CS_E_CYCLES, "coulomb_ion_cycles": CS_I_CYCLES,
    "coulomb_cycles": CS_CYCLES,
}
assert set(CS_KEYS) == set(COULOMB_KEYS)

# Coulomb-logarithm kinds of the device function
LNL_EE, LNL_EI, LNL_II = 0, 1, 2


if wp is not None:
    from .warp_backend import F64, PARTICLE_BLOCK, kinetic_energy, padded_dim

    vec4d = wp.types.vector(length=4, dtype=wp.float64)

    @wp.func
    def coulomb_log_device(kind: int, density: F64, temperature_ev: F64, fixed: F64, floor: F64) -> F64:
        """NRL Coulomb logarithms (n in cm^-3 inside, T in eV) with the floor; ``fixed > 0`` overrides (tests)."""

        if fixed > F64(0.0):
            return fixed
        n = wp.max(density * F64(1.0e-6), F64(1.0e-300))
        t = wp.max(temperature_ev, F64(1.0e-300))
        value = F64(0.0)
        if kind == 0:
            value = F64(23.5) - wp.log(wp.sqrt(n) * wp.pow(t, F64(-1.25))) - wp.sqrt(F64(1.0e-5) + (wp.log(t) - F64(2.0)) * (wp.log(t) - F64(2.0)) / F64(16.0))
        elif kind == 1:
            if t < F64(10.0):
                value = F64(23.0) - wp.log(wp.sqrt(n) * wp.pow(t, F64(-1.5)))
            else:
                value = F64(24.0) - wp.log(wp.sqrt(n) / t)
        else:
            value = F64(23.0) - wp.log(wp.sqrt(F64(2.0) * n / t) / t)
        return wp.max(value, floor)

    @wp.func
    def nanbu_cos_chi_device(s: F64, u: F64) -> F64:
        """``cos chi`` of Nanbu's cumulative angle (same branches as ``coulomb.nanbu_cos_chi``)."""

        uu = wp.clamp(u, F64(1.0e-30), F64(1.0))
        if s >= F64(6.0):
            return F64(2.0) * uu - F64(1.0)
        inv_a = F64(0.0)
        if s < F64(0.2):
            inv_a = F64(1.0) - wp.exp(-s)
            return wp.clamp(F64(1.0) + inv_a * wp.log(uu), F64(-1.0), F64(1.0))
        if s < F64(3.0):
            inv_a = F64(0.0056958) + s * (F64(0.9560202) + s * (F64(-0.508139) + s * (F64(0.47913906) + s * (F64(-0.12788975) + s * F64(0.02389567)))))
        else:
            inv_a = wp.exp(s) / F64(3.0)
        a = F64(1.0) / inv_a
        return wp.clamp(inv_a * wp.log(wp.exp(-a) + F64(2.0) * uu * wp.sinh(a)), F64(-1.0), F64(1.0))

    @wp.func
    def collide_pair(
        pa: int, vr_a: wp.array(dtype=F64), vt_a: wp.array(dtype=F64), vz_a: wp.array(dtype=F64),
        pb: int, vr_b: wp.array(dtype=F64), vt_b: wp.array(dtype=F64), vz_b: wp.array(dtype=F64),
        mass_a: F64, mass_b: F64, factor: F64, dt: F64, weight: F64, u1: F64, u2: F64,
    ) -> vec4d:
        """One binary collision (Takizuka-Abe kinematics, Nanbu angle).  ``factor`` = ``(lnL / 4 pi) (q_a q_b / eps0 m_ab)^2 n_field``
        so ``s = factor dt / g^3``.  Writes both velocities; returns (s, W dpz, W dKE_rel, 1) or zeros for a zero relative speed."""

        ax = vr_a[pa]
        ay = vt_a[pa]
        az = vz_a[pa]
        bx = vr_b[pb]
        by = vt_b[pb]
        bz = vz_b[pb]
        ux = ax - bx
        uy = ay - by
        uz = az - bz
        g2 = ux * ux + uy * uy + uz * uz
        if g2 <= F64(0.0):
            return vec4d(F64(0.0), F64(0.0), F64(0.0), F64(0.0))
        g = wp.sqrt(g2)
        s = factor * dt / (g2 * g)
        cos_chi = nanbu_cos_chi_device(s, u1)
        sin_chi = wp.sqrt(wp.max(F64(1.0) - cos_chi * cos_chi, F64(0.0)))
        one_minus = F64(1.0) - cos_chi
        phi = F64(6.283185307179586) * u2
        cos_phi = wp.cos(phi)
        sin_phi = wp.sin(phi)
        u_perp = wp.sqrt(ux * ux + uy * uy)
        dux = F64(0.0)
        duy = F64(0.0)
        duz = F64(0.0)
        if u_perp > F64(0.0):
            inv = F64(1.0) / u_perp
            dux = (ux * inv) * uz * sin_chi * cos_phi - (uy * inv) * g * sin_chi * sin_phi - ux * one_minus
            duy = (uy * inv) * uz * sin_chi * cos_phi + (ux * inv) * g * sin_chi * sin_phi - uy * one_minus
            duz = -u_perp * sin_chi * cos_phi - uz * one_minus
        else:
            dux = g * sin_chi * cos_phi
            duy = g * sin_chi * sin_phi
            duz = -uz * one_minus
        total = mass_a + mass_b
        fa = mass_b / total
        fb = mass_a / total
        ke_before = kinetic_energy(ax, ay, az, mass_a * weight) + kinetic_energy(bx, by, bz, mass_b * weight)
        ax2 = ax + fa * dux
        ay2 = ay + fa * duy
        az2 = az + fa * duz
        bx2 = bx - fb * dux
        by2 = by - fb * duy
        bz2 = bz - fb * duz
        vr_a[pa] = ax2
        vt_a[pa] = ay2
        vz_a[pa] = az2
        vr_b[pb] = bx2
        vt_b[pb] = by2
        vz_b[pb] = bz2
        ke_after = kinetic_energy(ax2, ay2, az2, mass_a * weight) + kinetic_energy(bx2, by2, bz2, mass_b * weight)
        dpz = weight * (mass_a * (az2 - az) + mass_b * (bz2 - bz))
        return vec4d(s, dpz, ke_after - ke_before, F64(1.0))

    # ------------------------------------------------------------------ cell sort (permutation only)
    @wp.kernel
    def coulomb_cell_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), alive: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32), slot: int,
        dr: F64, dz: F64, z_min: F64, nr: int, nz: int,
        cell: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        c = -1
        if p < slots[slot]:
            if alive[p] != 0:
                i = wp.clamp(int(wp.floor(r[p] / dr)), 0, nr - 1)
                j = wp.clamp(int(wp.floor((z[p] - z_min) / dz)), 0, nz - 1)
                c = i * nz + j
        cell[p] = c
        if c >= 0:
            pos[p] = wp.atomic_add(counts, c, 1)

    @wp.kernel
    def coulomb_scatter_kernel(cell: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                               tmp: wp.array(dtype=wp.int32)):
        p = wp.tid()
        c = cell[p]
        if c >= 0:
            tmp[starts[c] + pos[p]] = p

    @wp.kernel
    def coulomb_rank_kernel(cell: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32), tmp: wp.array(dtype=wp.int32),
                            sorted_slots: wp.array(dtype=wp.int32), cell_of_sorted: wp.array(dtype=wp.int32)):
        # rank = number of same-cell alive slots with a smaller slot index: the segment order is the slot order,
        # independent of the atomic arrival order of coulomb_cell_kernel (deterministic replay)
        p = wp.tid()
        c = cell[p]
        if c >= 0:
            base = starts[c]
            end = starts[c + 1]
            rank = int(0)      # noqa: UP018, RUF046  (Warp: a variable mutated in a dynamic loop is declared dynamic)
            for k in range(base, end):
                if tmp[k] < p:
                    rank += 1
            sorted_slots[base + rank] = p
            cell_of_sorted[base + rank] = c

    @wp.kernel
    def coulomb_prepare_kernel(
        starts: wp.array(dtype=wp.int32), sorted_slots: wp.array(dtype=wp.int32),
        vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64), mass: F64, weight: F64,
        cell_volume_r: wp.array(dtype=F64), nz: int, min_temperature_ev: F64,
        seed_table: wp.array(dtype=wp.int32), seed_streams: int, stream_column: int, counter: wp.array(dtype=wp.int32), key_offset: int,
        shuffle: int, cell_density: wp.array(dtype=F64), cell_temperature: wp.array(dtype=F64), shift_u: wp.array(dtype=F64),
        accumulate: int, dt_c: F64, window_seconds: wp.array(dtype=F64),
    ):
        # one thread per cell: moments over the segment in sorted order (deterministic), Fisher-Yates shuffle of the
        # segment (random Takizuka-Abe partners), the electron-ion pairing shift draw, the electron-seconds window sum
        c = wp.tid()
        base = starts[c]
        n = starts[c + 1] - base
        sr = F64(0.0)
        st = F64(0.0)
        sz = F64(0.0)
        s2 = F64(0.0)
        for k in range(base, base + n):
            p = sorted_slots[k]
            a = vr[p]
            b = vt[p]
            d = vz[p]
            sr += a
            st += b
            sz += d
            s2 += a * a + b * b + d * d
        temperature = min_temperature_ev
        if n > 0:
            inv = F64(1.0) / F64(n)
            variance = s2 * inv - (sr * sr + st * st + sz * sz) * inv * inv
            temperature = wp.max(mass * wp.max(variance, F64(0.0)) / (F64(3.0) * F64(1.602176634e-19)), min_temperature_ev)
        cell_temperature[c] = temperature
        i = c / nz
        cell_density[c] = F64(n) * weight / cell_volume_r[i]
        seed = seed_table[seed_streams * counter[0] + stream_column]
        rng = wp.rand_init(seed, key_offset + c)
        if shuffle != 0:
            if n >= 2:
                for m in range(n - 1):
                    k = n - 1 - m
                    j = int(F64(wp.randf(rng)) * F64(k + 1))
                    j = wp.min(j, k)
                    ia = base + k
                    ib = base + j
                    swap = sorted_slots[ia]
                    sorted_slots[ia] = sorted_slots[ib]
                    sorted_slots[ib] = swap
        shift_u[c] = F64(wp.randf(rng))
        if accumulate != 0:
            window_seconds[c] = window_seconds[c] + F64(n) * dt_c

    @wp.kernel
    def coulomb_cycle_kernel(starts_e: wp.array(dtype=wp.int32), starts_i: wp.array(dtype=wp.int32), n_cells: int, stats: wp.array(dtype=F64)):
        stats[CS_E_CYCLES] = stats[CS_E_CYCLES] + F64(starts_e[n_cells])
        stats[CS_I_CYCLES] = stats[CS_I_CYCLES] + F64(starts_i[n_cells])
        stats[CS_CYCLES] = stats[CS_CYCLES] + F64(1.0)

    # ------------------------------------------------------------------ collisions
    @wp.kernel
    def coulomb_like_kernel(
        starts: wp.array(dtype=wp.int32), sorted_slots: wp.array(dtype=wp.int32), cell_of_sorted: wp.array(dtype=wp.int32), n_cells: int,
        vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64), mass: F64, charge: F64, weight: F64,
        cell_density: wp.array(dtype=F64), cell_temperature: wp.array(dtype=F64), lnl_kind: int, lnl_fixed: F64, lnl_floor: F64, dt_c: F64,
        seed_table: wp.array(dtype=wp.int32), seed_streams: int, stream_column: int, counter: wp.array(dtype=wp.int32), key_offset: int,
        stats: wp.array(dtype=F64), slot_base: int, accumulate: int, window_s: wp.array(dtype=F64), window_pairs: wp.array(dtype=F64),
    ):
        # one thread per sorted position g; the first member of a pair performs it.  Even segment: (0,1), (2,3), ...;
        # odd segment: the triplet (0,1), (0,2), (1,2) with dt_c / 2 by the thread of member 0, then (3,4), (5,6), ...
        g = wp.tid()
        pairs = F64(0.0)
        s_sum = F64(0.0)
        large = F64(0.0)
        lnl_sum = F64(0.0)
        dpz = F64(0.0)
        dke = F64(0.0)
        total = starts[n_cells]
        if g < total:
            c = cell_of_sorted[g]
            base = starts[c]
            n = starts[c + 1] - base
            local = g - base
            mode = int(0)      # noqa: UP018, RUF046  (dynamic: 0 nothing, 1 full-step pair (g, g+1), 2 triplet)
            if n >= 2:
                if n % 2 == 0:
                    if local % 2 == 0:
                        mode = 1
                else:
                    if local == 0:
                        mode = 2
                    elif local >= 3:
                        if (local - 3) % 2 == 0:
                            mode = 1
            if mode != 0:
                density = cell_density[c]
                lnl = coulomb_log_device(lnl_kind, density, cell_temperature[c], lnl_fixed, lnl_floor)
                reduced = mass * F64(0.5)
                q2 = charge * charge / (F64(8.8541878128e-12) * reduced)
                factor = lnl / (F64(4.0) * F64(3.141592653589793)) * q2 * q2 * density
                seed = seed_table[seed_streams * counter[0] + stream_column]
                rng = wp.rand_init(seed, key_offset + g)
                if mode == 1:
                    u1 = F64(wp.randf(rng))
                    u2 = F64(wp.randf(rng))
                    out = collide_pair(sorted_slots[g], vr, vt, vz, sorted_slots[g + 1], vr, vt, vz, mass, mass, factor, dt_c, weight, u1, u2)
                    pairs += out[3]
                    s_sum += out[0]
                    if out[0] > F64(1.0):
                        large += F64(1.0)
                    lnl_sum += lnl * out[3]
                    dpz += out[1]
                    dke += out[2]
                    if accumulate != 0:
                        if out[3] > F64(0.0):
                            wp.atomic_add(window_s, c, out[0])
                            wp.atomic_add(window_pairs, c, F64(1.0))
                else:
                    half = F64(0.5) * dt_c
                    p0 = sorted_slots[g]
                    p1 = sorted_slots[g + 1]
                    p2 = sorted_slots[g + 2]
                    for sub in range(3):
                        pa = p0
                        pb = p1
                        if sub == 1:
                            pb = p2
                        elif sub == 2:
                            pa = p1
                            pb = p2
                        u1 = F64(wp.randf(rng))
                        u2 = F64(wp.randf(rng))
                        out = collide_pair(pa, vr, vt, vz, pb, vr, vt, vz, mass, mass, factor, half, weight, u1, u2)
                        pairs += out[3]
                        s_sum += out[0]
                        if out[0] > F64(1.0):
                            large += F64(1.0)
                        lnl_sum += lnl * out[3]
                        dpz += out[1]
                        dke += out[2]
                        if accumulate != 0:
                            if out[3] > F64(0.0):
                                wp.atomic_add(window_s, c, out[0])
                                wp.atomic_add(window_pairs, c, F64(1.0))
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(pairs)), offset=slot_base)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(s_sum)), offset=slot_base + 1)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(large)), offset=slot_base + 2)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(lnl_sum)), offset=slot_base + 3)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dpz)), offset=CS_PZ)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dke)), offset=CS_KE)

    @wp.kernel
    def coulomb_unlike_kernel(
        starts_e: wp.array(dtype=wp.int32), sorted_e: wp.array(dtype=wp.int32),
        starts_i: wp.array(dtype=wp.int32), sorted_i: wp.array(dtype=wp.int32), cell_of_sorted_i: wp.array(dtype=wp.int32), n_cells: int,
        e_vr: wp.array(dtype=F64), e_vt: wp.array(dtype=F64), e_vz: wp.array(dtype=F64),
        i_vr: wp.array(dtype=F64), i_vt: wp.array(dtype=F64), i_vz: wp.array(dtype=F64),
        mass_e: F64, mass_i: F64, charge_e: F64, charge_i: F64, weight: F64,
        cell_density_e: wp.array(dtype=F64), cell_temperature_e: wp.array(dtype=F64), cell_density_i: wp.array(dtype=F64),
        shift_u: wp.array(dtype=F64), lnl_fixed: F64, lnl_floor: F64, dt_c: F64,
        seed_table: wp.array(dtype=wp.int32), seed_streams: int, stream_column: int, counter: wp.array(dtype=wp.int32), key_offset: int,
        stats: wp.array(dtype=F64), accumulate: int, window_s: wp.array(dtype=F64), window_pairs: wp.array(dtype=F64),
    ):
        # one thread per sorted ION position: its electrons are l = l0 + m N_i (l0 = (i_local - shift) mod N_i), collided
        # sequentially against this ion at the field density n_i (every electron once; the ions N_e / N_i times on average)
        gi = wp.tid()
        pairs = F64(0.0)
        s_sum = F64(0.0)
        large = F64(0.0)
        lnl_sum = F64(0.0)
        dpz = F64(0.0)
        dke = F64(0.0)
        total_i = starts_i[n_cells]
        if gi < total_i:
            c = cell_of_sorted_i[gi]
            ibase = starts_i[c]
            n_i = starts_i[c + 1] - ibase
            ebase = starts_e[c]
            n_e = starts_e[c + 1] - ebase
            if n_e > 0:
                shift = wp.min(int(shift_u[c] * F64(n_i)), n_i - 1)
                l0 = ((gi - ibase - shift) % n_i + n_i) % n_i
                lnl = coulomb_log_device(LNL_EI, cell_density_e[c], cell_temperature_e[c], lnl_fixed, lnl_floor)
                reduced = mass_e * mass_i / (mass_e + mass_i)
                q2 = charge_e * charge_i / (F64(8.8541878128e-12) * reduced)
                factor = lnl / (F64(4.0) * F64(3.141592653589793)) * q2 * q2 * cell_density_i[c]
                seed = seed_table[seed_streams * counter[0] + stream_column]
                rng = wp.rand_init(seed, key_offset + gi)
                p_ion = sorted_i[gi]
                l = int(l0)
                while l < n_e:
                    pe = sorted_e[ebase + l]
                    u1 = F64(wp.randf(rng))
                    u2 = F64(wp.randf(rng))
                    out = collide_pair(pe, e_vr, e_vt, e_vz, p_ion, i_vr, i_vt, i_vz, mass_e, mass_i, factor, dt_c, weight, u1, u2)
                    pairs += out[3]
                    s_sum += out[0]
                    if out[0] > F64(1.0):
                        large += F64(1.0)
                    lnl_sum += lnl * out[3]
                    dpz += out[1]
                    dke += out[2]
                    if accumulate != 0:
                        if out[3] > F64(0.0):
                            wp.atomic_add(window_s, c, out[0])
                            wp.atomic_add(window_pairs, c, F64(1.0))
                    l += n_i
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(pairs)), offset=CS_EI_PAIRS)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(s_sum)), offset=CS_EI_S)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(large)), offset=CS_EI_LARGE)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(lnl_sum)), offset=CS_EI_LNL)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dpz)), offset=CS_PZ)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dke)), offset=CS_KE)


class WarpCoulombStage:
    """Device state and launch sequence of the Coulomb stage for one backend (or a standalone harness in tests).

    ``launch`` issues the fixed-shape kernel sequence for one cycle on the given particle arrays; ``sync_tallies``
    reads and zeroes the statistics (host sync); ``window_sums`` / ``reset_window`` expose the per-cell window sums
    in the node-shaped layout of ``DiagnosticAccumulator``.
    """

    def __init__(self, config: CoulombConfig, grid: Grid2D, masks: MeshMasks, macro_weight: float, device: Any, *,
                 electron: Species2D, ion: Species2D, seed_streams: int, stream_column: int = COULOMB_STREAM) -> None:
        if wp is None:
            raise PIC2DValidationError("NVIDIA Warp is unavailable")
        self.config = config
        self.grid = grid
        self.masks = masks
        self.device = device
        self.macro_weight = float(macro_weight)
        self.electron = electron
        self.ion = ion
        self.seed_streams = int(seed_streams)
        self.stream_column = int(stream_column)
        self.nr, self.nz = grid.cell_shape
        self.n_cells = self.nr * self.nz
        dev = device
        self.cell_volume_r = wp.array(cell_volumes_m3(grid), dtype=wp.float64, device=dev)
        self.counts_e = wp.zeros(self.n_cells + 1, dtype=wp.int32, device=dev)
        self.starts_e = wp.zeros(self.n_cells + 1, dtype=wp.int32, device=dev)
        self.counts_i = wp.zeros(self.n_cells + 1, dtype=wp.int32, device=dev)
        self.starts_i = wp.zeros(self.n_cells + 1, dtype=wp.int32, device=dev)
        self.density_e = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.temperature_e = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.density_i = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.temperature_i = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.shift_e = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.shift_i = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.stats = wp.zeros(CS_SIZE, dtype=wp.float64, device=dev)
        self.win_ee_s = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.win_ee_pairs = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.win_ei_s = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.win_ei_pairs = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.win_electron_seconds = wp.zeros(self.n_cells, dtype=wp.float64, device=dev)
        self.scratch_capacity = 0
        self.launches = 0

    # -- scratch --------------------------------------------------------------------------------------------------
    def ensure_scratch(self, capacity: int) -> None:
        if capacity <= self.scratch_capacity:
            return
        self.scratch_capacity = int(capacity)
        dev = self.device
        make = lambda: wp.zeros(self.scratch_capacity, dtype=wp.int32, device=dev)
        self.cell_e, self.pos_e, self.tmp_e, self.sorted_e, self.cell_of_sorted_e = (make() for _ in range(5))
        self.cell_i, self.pos_i, self.tmp_i, self.sorted_i, self.cell_of_sorted_i = (make() for _ in range(5))

    # -- launches -------------------------------------------------------------------------------------------------
    def _sort(self, species: Any, slot: int, dim: int, counts: Any, starts: Any, cell: Any, pos: Any, tmp: Any, sorted_slots: Any,
              cell_of_sorted: Any, slots: Any) -> None:
        grid = self.grid
        dev = self.device
        counts.zero_()
        wp.launch(coulomb_cell_kernel, dim=dim,
                  inputs=[species.r, species.z, species.alive_flags, slots, slot, grid.dr_m, grid.dz_m, grid.geometry.z_min_m, self.nr, self.nz,
                          cell, pos, counts], device=dev)
        wp.utils.array_scan(counts, starts, inclusive=False)
        wp.launch(coulomb_scatter_kernel, dim=dim, inputs=[cell, pos, starts, tmp], device=dev)
        wp.launch(coulomb_rank_kernel, dim=dim, inputs=[cell, starts, tmp, sorted_slots, cell_of_sorted], device=dev)

    def launch(self, electrons: Any, ions: Any, slots: Any, seed_table: Any, step_counter: Any, e_dim: int, i_dim: int, dt_c: float,
               accumulate: bool) -> None:
        """One Coulomb cycle on the backend's device species (``e_dim`` / ``i_dim`` = launch dimensions, capacities under a graph)."""

        config = self.config
        dev = self.device
        w = self.macro_weight
        n_cells = self.n_cells
        e_dim = max(int(e_dim), 1)
        i_dim = max(int(i_dim), 1)
        self.ensure_scratch(max(e_dim, i_dim))
        acc = 1 if accumulate else 0
        fixed = -1.0 if config.coulomb_log_fixed is None else float(config.coulomb_log_fixed)
        floor = float(config.coulomb_log_floor)
        need_ions = config.electron_ion or config.ion_ion
        self._sort(electrons, 0, e_dim, self.counts_e, self.starts_e, self.cell_e, self.pos_e, self.tmp_e, self.sorted_e, self.cell_of_sorted_e, slots)
        if need_ions:
            self._sort(ions, 1, i_dim, self.counts_i, self.starts_i, self.cell_i, self.pos_i, self.tmp_i, self.sorted_i, self.cell_of_sorted_i, slots)
        else:
            self.counts_i.zero_()
            self.starts_i.zero_()
        # per-cell moments + shuffles (electron keys [0, n_cells), ion keys [n_cells, 2 n_cells))
        wp.launch(coulomb_prepare_kernel, dim=n_cells,
                  inputs=[self.starts_e, self.sorted_e, electrons.vr, electrons.vt, electrons.vz, self.electron.mass_kg, w, self.cell_volume_r, self.nz,
                          config.min_temperature_ev, seed_table, self.seed_streams, self.stream_column, step_counter, 0,
                          1 if config.electron_electron else 0, self.density_e, self.temperature_e, self.shift_e, acc, float(dt_c), self.win_electron_seconds],
                  device=dev)
        if need_ions:
            wp.launch(coulomb_prepare_kernel, dim=n_cells,
                      inputs=[self.starts_i, self.sorted_i, ions.vr, ions.vt, ions.vz, self.ion.mass_kg, w, self.cell_volume_r, self.nz,
                              config.min_temperature_ev, seed_table, self.seed_streams, self.stream_column, step_counter, n_cells,
                              1 if config.ion_ion else 0, self.density_i, self.temperature_i, self.shift_i, 0, float(dt_c), self.win_electron_seconds],
                      device=dev)
        wp.launch(coulomb_cycle_kernel, dim=1, inputs=[self.starts_e, self.starts_i, n_cells, self.stats], device=dev)
        # pair keys: e-e [2 n_cells, 2 n_cells + cap_e); e-i [.. + cap_e, .. + 2 cap); i-i [.. + 2 cap, .. + 3 cap)
        cap = self.scratch_capacity
        if config.electron_electron:
            wp.launch(coulomb_like_kernel, dim=padded_dim(e_dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
                      inputs=[self.starts_e, self.sorted_e, self.cell_of_sorted_e, n_cells, electrons.vr, electrons.vt, electrons.vz,
                              self.electron.mass_kg, self.electron.charge_c, w, self.density_e, self.temperature_e, LNL_EE, fixed, floor, float(dt_c),
                              seed_table, self.seed_streams, self.stream_column, step_counter, 2 * n_cells,
                              self.stats, CS_EE_PAIRS, acc, self.win_ee_s, self.win_ee_pairs],
                      device=dev)
        if config.electron_ion:
            wp.launch(coulomb_unlike_kernel, dim=padded_dim(i_dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
                      inputs=[self.starts_e, self.sorted_e, self.starts_i, self.sorted_i, self.cell_of_sorted_i, n_cells,
                              electrons.vr, electrons.vt, electrons.vz, ions.vr, ions.vt, ions.vz,
                              self.electron.mass_kg, self.ion.mass_kg, self.electron.charge_c, self.ion.charge_c, w,
                              self.density_e, self.temperature_e, self.density_i, self.shift_e, fixed, floor, float(dt_c),
                              seed_table, self.seed_streams, self.stream_column, step_counter, 2 * n_cells + cap,
                              self.stats, acc, self.win_ei_s, self.win_ei_pairs],
                      device=dev)
        if config.ion_ion:
            wp.launch(coulomb_like_kernel, dim=padded_dim(i_dim, PARTICLE_BLOCK), block_dim=PARTICLE_BLOCK,
                      inputs=[self.starts_i, self.sorted_i, self.cell_of_sorted_i, n_cells, ions.vr, ions.vt, ions.vz,
                              self.ion.mass_kg, self.ion.charge_c, w, self.density_i, self.temperature_i, LNL_II, fixed, floor, float(dt_c),
                              seed_table, self.seed_streams, self.stream_column, step_counter, 2 * n_cells + 2 * cap,
                              self.stats, CS_II_PAIRS, 0, self.win_ee_s, self.win_ee_pairs],
                      device=dev)
        self.launches += 1

    # -- host reads -----------------------------------------------------------------------------------------------
    def sync_tallies(self) -> dict[str, float]:
        """Read and zero the interval statistics (at a host sync); keys = ``COULOMB_KEYS``."""

        values = self.stats.numpy().copy()      # (a CPU device array's .numpy() is a view: copy before zeroing)
        self.stats.zero_()
        return {key: float(values[slot]) for key, slot in CS_KEYS.items()}

    def window_sums(self) -> dict[str, np.ndarray]:
        shape = self.grid.node_shape
        cell_shape = (self.nr, self.nz)
        return {
            "coulomb_ee_s": cell_maps_to_nodes(self.win_ee_s.numpy().reshape(cell_shape), shape),
            "coulomb_ee_pairs": cell_maps_to_nodes(self.win_ee_pairs.numpy().reshape(cell_shape), shape),
            "coulomb_ei_s": cell_maps_to_nodes(self.win_ei_s.numpy().reshape(cell_shape), shape),
            "coulomb_ei_pairs": cell_maps_to_nodes(self.win_ei_pairs.numpy().reshape(cell_shape), shape),
            "coulomb_electron_seconds": cell_maps_to_nodes(self.win_electron_seconds.numpy().reshape(cell_shape), shape),
        }

    def reset_window(self) -> None:
        for array in (self.win_ee_s, self.win_ee_pairs, self.win_ei_s, self.win_ei_pairs, self.win_electron_seconds):
            array.zero_()

    # -- standalone harness (tests / parity) ---------------------------------------------------------------------
    def apply_host(self, electrons: ParticleArrays, ions: ParticleArrays, dt_c: float, *, seed: int, step: int,
                   accumulate: bool = False) -> tuple[ParticleArrays, ParticleArrays, dict[str, float]]:
        """Run one cycle on host particle arrays (uploaded to fresh device arrays); returns the new arrays and the tallies."""

        from .warp_backend import DeviceSpecies, stream_seed

        dev = self.device

        def upload(particles: ParticleArrays) -> DeviceSpecies:
            capacity = max(particles.count, 1)
            arrays = [wp.array(np.ascontiguousarray(getattr(particles, name), dtype=np.float64) if particles.count else np.zeros(1), dtype=wp.float64, device=dev)
                      for name in ("r_m", "z_m", "vr_m_per_s", "vt_m_per_s", "vz_m_per_s")]
            alive = wp.array(np.ones(capacity, dtype=np.int32) if particles.count else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=dev)
            return DeviceSpecies(capacity, particles.count, particles.count, *arrays, alive)

        e = upload(electrons)
        i = upload(ions)
        slots = wp.array(np.array([electrons.count, ions.count], dtype=np.int32), dtype=wp.int32, device=dev)
        seeds = np.zeros(self.seed_streams, dtype=np.int32)
        seeds[self.stream_column] = stream_seed(seed, step, self.stream_column + 1)
        seed_table = wp.array(seeds, dtype=wp.int32, device=dev)
        counter = wp.zeros(1, dtype=wp.int32, device=dev)
        self.launch(e, i, slots, seed_table, counter, e.capacity, i.capacity, dt_c, accumulate)
        wp.synchronize_device(dev)

        def download(species: DeviceSpecies, count: int) -> ParticleArrays:
            if count == 0:
                return ParticleArrays.empty()
            return ParticleArrays(*[np.asarray(a.numpy()[:count], dtype=np.float64).copy() for a in (species.r, species.z, species.vr, species.vt, species.vz)])

        return download(e, electrons.count), download(i, ions.count), self.sync_tallies()

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "cells": self.n_cells, "macro_weight": self.macro_weight, "ledger_keys": list(COULOMB_KEYS),
                "ion_mass_kg": self.ion.mass_kg, "electron_mass_kg": self.electron.mass_kg, "seed_stream_column": self.stream_column,
                "pairing": "cell-sorted slot permutation (counting sort + deterministic within-cell rank); particles never reordered"}


__all__ = ["COULOMB_STREAM", "CS_KEYS", "CS_SIZE", "WarpCoulombStage"]
