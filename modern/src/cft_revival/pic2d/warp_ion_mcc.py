"""Warp kernel for the Xe+ - Xe null-collision MCC (model v2.3.0 / R3b; device counterpart of ``ion_mcc``).

One thread per ion slot; graph-capturable (fixed arguments, device-resident neutral density, seeds from the
per-step seed table, stream index 3 = the fourth stream).  The arithmetic mirrors ``IonNullCollisionMCC.apply``
and ``fast_neutral_fate`` step for step (thermal-atom sampling, relative energy, process selection, CEX
velocity swap, isotropic centre-of-mass MEX, straight-line fast-neutral march through the cell mask); the
random streams differ from the CPU reference (counter-based ``rand_init``), so parity is distributional.

Tallies are tile-reduced per block into the backend's statistics array (slot layout ``ION_STATS_*``); the
float sums (energy loss, momenta) are float64 atomics across blocks and may differ from a rerun at round-off,
like every other ledger float of the Warp backend; the counts are exact.
"""

from __future__ import annotations

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dependency
    wp = None  # type: ignore[assignment]

# statistics slot layout relative to the block base handed to the kernel (16 slots)
ION_STATS_CANDIDATES = 0
ION_STATS_CEX = 1
ION_STATS_MEX = 2
ION_STATS_NULL = 3
ION_STATS_CEX_PLUME = 4
ION_STATS_CEILING = 5
ION_STATS_FAST_EXIT_CHANNEL = 6
ION_STATS_FAST_EXIT_PLUME = 7
ION_STATS_FAST_WALL = 8
ION_STATS_FAST_THERMAL = 9
ION_STATS_FAST_UNRESOLVED = 10
ION_STATS_ENERGY_LOSS = 11
ION_STATS_PZ_IONS = 12
ION_STATS_PZ_FAST_EXIT = 13
ION_STATS_PZ_FAST_WALL = 14
ION_STATS_KE_FAST_EXIT = 15
ION_STATS_COUNT = 16
# order of the cumulative-ledger keys matching the slots above (the backend's _sync uses it)
ION_STATS_KEYS = (
    "ion_mcc_candidates", "cex", "mex", "ion_mcc_null", "cex_plume", "ion_mcc_ceiling_violations",
    "fast_neutral_exit_channel", "fast_neutral_exit_plume", "fast_neutral_wall", "fast_neutral_thermal", "fast_neutral_unresolved",
    "ion_neutral_loss_j", "pz_ion_collisions", "pz_fast_neutral_exit", "pz_fast_neutral_wall", "ke_fast_neutral_exit_j",
)

if wp is not None:
    F64 = wp.float64

    @wp.func
    def ion_sigma_lookup(table: wp.array(dtype=F64), points: int, step_ev: F64, max_ev: F64, energy: F64, process: int) -> F64:
        e = wp.clamp(energy, F64(0.0), max_ev)
        position = e / step_ev
        index = wp.min(int(wp.floor(position)), points - 2)
        fraction = position - F64(index)
        lower = table[process * points + index]
        upper = table[process * points + index + 1]
        return lower + fraction * (upper - lower)

    @wp.func
    def fast_neutral_fate_march(
        plasma_cell: wp.array(dtype=wp.int32), r0: F64, z0: F64, vr: F64, vt: F64, vz: F64,
        dr: F64, dz: F64, z_min: F64, z_exit: F64, r_exit: F64, nr: int, nz: int, limit: int,
    ) -> int:
        # 0 exit (through the aperture), 1 wall / anode / box, 2 unresolved; same arithmetic as ion_mcc.fast_neutral_fate
        speed = wp.sqrt(vr * vr + vt * vt + vz * vz)
        if speed <= F64(0.0):
            return 1
        step_dt = F64(0.5) * wp.min(dr, dz) / speed
        result = int(2)
        k = int(1)
        while k <= limit:
            t = F64(k) * step_dt
            x = r0 + vr * t
            y = vt * t
            rr = wp.sqrt(x * x + y * y)
            zz = z0 + vz * t
            if zz >= z_exit:
                if rr < r_exit:
                    result = 0
                else:
                    result = 1
                k = limit + 1
            elif zz < z_min:
                result = 1
                k = limit + 1
            else:
                i = int(wp.floor(rr / dr))
                if i >= nr:
                    result = 1
                    k = limit + 1
                else:
                    j = wp.clamp(int(wp.floor((zz - z_min) / dz)), 0, nz - 1)
                    if plasma_cell[i * nz + j] == 0:
                        result = 1
                        k = limit + 1
                    else:
                        k = k + 1
        return result

    @wp.kernel
    def ion_mcc_kernel(
        r: wp.array(dtype=F64), z: wp.array(dtype=F64), vr: wp.array(dtype=F64), vt: wp.array(dtype=F64), vz: wp.array(dtype=F64),
        alive: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32), slot: int,
        seed_table: wp.array(dtype=wp.int32), seed_streams: int, stream: int, counter: wp.array(dtype=wp.int32),
        probability: F64, nu_max: F64, neutral_density_ctrl: wp.array(dtype=F64),
        table: wp.array(dtype=F64), points: int, step_ev: F64, max_ev: F64,
        mass: F64, mass_weight: F64, thermal: F64, fast_threshold: F64,
        shape_cell: wp.array(dtype=F64), plasma_cell: wp.array(dtype=wp.int32), has_plume: int,
        dr: F64, dz: F64, z_min: F64, z_exit: F64, r_exit: F64, nr: int, nz: int, march_limit: int,
        stats: wp.array(dtype=F64), base: int,
        # v2.5.0 (neutrals_spatial_v1): the published per-cell ground density and gas moments replace n_g x shape and the
        # Maxwellian at T_g; a CEX event flags the ion slot and stores the pre-event velocity for the neutral model's
        # hand-off (no fate march) and books the converted thermal atom as a ground-atom sink at the event cell
        spatial: int, density_cell: wp.array(dtype=F64), drift_r: wp.array(dtype=F64), drift_t: wp.array(dtype=F64),
        drift_z: wp.array(dtype=F64), thermal_cell: wp.array(dtype=F64),
        fast_flag: wp.array(dtype=wp.int32), fast_vr: wp.array(dtype=F64), fast_vt: wp.array(dtype=F64), fast_vz: wp.array(dtype=F64),
        sink_cex: wp.array(dtype=wp.int64), sink_unit: int, flag_bound: int,
    ):
        p = wp.tid()
        if spatial != 0 and p < flag_bound:
            fast_flag[p] = 0
        candidate = F64(0.0)
        cex = F64(0.0)
        mex = F64(0.0)
        null = F64(0.0)
        cex_plume = F64(0.0)
        ceiling = F64(0.0)
        fast_exit_channel = F64(0.0)
        fast_exit_plume = F64(0.0)
        fast_wall = F64(0.0)
        fast_thermal = F64(0.0)
        fast_unresolved = F64(0.0)
        energy_loss = F64(0.0)
        dpz = F64(0.0)
        pz_exit = F64(0.0)
        pz_wall = F64(0.0)
        ke_exit = F64(0.0)
        active = 0
        if p < slots[slot]:
            if alive[p] != 0:
                active = 1
        if active != 0:
            seed = seed_table[seed_streams * counter[0] + stream]
            state = wp.rand_init(seed, p)
            u0 = F64(wp.randf(state))
            if u0 < probability:
                candidate = F64(1.0)
                u1 = F64(wp.randf(state))
                # thermal atom (Box-Muller, four uniforms; same construction as the CPU maxwellian_velocity)
                u2 = wp.max(F64(wp.randf(state)), F64(1.0e-300))
                u3 = F64(wp.randf(state))
                u4 = wp.max(F64(wp.randf(state)), F64(1.0e-300))
                u5 = F64(wp.randf(state))
                rad1 = wp.sqrt(F64(-2.0) * wp.log(u2))
                rad2 = wp.sqrt(F64(-2.0) * wp.log(u4))
                cell = int(0)
                if has_plume != 0 or spatial != 0:
                    ci = wp.clamp(int(wp.floor(r[p] / dr)), 0, nr - 1)
                    cj = wp.clamp(int(wp.floor((z[p] - z_min) / dz)), 0, nz - 1)
                    cell = ci * nz + cj
                th = thermal
                ur = F64(0.0)
                ut = F64(0.0)
                uz = F64(0.0)
                if spatial != 0:
                    ur = drift_r[cell]
                    ut = drift_t[cell]
                    uz = drift_z[cell]
                    if thermal_cell[cell] > F64(0.0):
                        th = thermal_cell[cell]
                nvx = ur + th * rad1 * wp.cos(F64(6.283185307179586) * u3)
                nvy = ut + th * rad1 * wp.sin(F64(6.283185307179586) * u3)
                nvz = uz + th * rad2 * wp.cos(F64(6.283185307179586) * u5)
                ivx = vr[p]
                ivy = vt[p]
                ivz = vz[p]
                gx = ivx - nvx
                gy = ivy - nvy
                gz = ivz - nvz
                g2 = gx * gx + gy * gy + gz * gz
                g = wp.sqrt(g2)
                energy = F64(0.5) * mass * g2 / F64(1.602176634e-19)
                density = neutral_density_ctrl[0]
                if spatial != 0:
                    density = density_cell[cell]
                elif has_plume != 0:
                    density = density * shape_cell[cell]
                nu_cex = density * ion_sigma_lookup(table, points, step_ev, max_ev, energy, 0) * g
                nu_mex = density * ion_sigma_lookup(table, points, step_ev, max_ev, energy, 1) * g
                total = nu_cex + nu_mex
                if total > nu_max * (F64(1.0) + F64(1.0e-12)):
                    ceiling = F64(1.0)
                selector = u1 * nu_max
                new_vx = ivx
                new_vy = ivy
                new_vz = ivz
                changed = 0
                if selector < nu_cex:
                    cex = F64(1.0)
                    changed = 1
                    new_vx = nvx
                    new_vy = nvy
                    new_vz = nvz
                    # fast neutral = the ion's old velocity at the ion's position
                    fspeed = wp.sqrt(ivx * ivx + ivy * ivy + ivz * ivz)
                    in_plume = 0
                    if z[p] >= z_exit:
                        in_plume = 1
                        cex_plume = F64(1.0)
                    if spatial != 0:
                        # v2.5.0: hand the fast neutral to the neutral particle model; the converted atom is a ground sink
                        fast_flag[p] = 1
                        fast_vr[p] = ivx
                        fast_vt[p] = ivy
                        fast_vz[p] = ivz
                        wp.atomic_add(sink_cex, cell, wp.int64(sink_unit))
                    elif fspeed < fast_threshold:
                        fast_thermal = F64(1.0)
                    elif in_plume != 0:
                        fast_exit_plume = F64(1.0)
                        pz_exit = mass_weight * ivz
                        ke_exit = F64(0.5) * mass_weight * fspeed * fspeed
                    else:
                        fate = fast_neutral_fate_march(plasma_cell, r[p], z[p], ivx, ivy, ivz, dr, dz, z_min, z_exit, r_exit, nr, nz, march_limit)
                        if fate == 0:
                            fast_exit_channel = F64(1.0)
                            pz_exit = mass_weight * ivz
                            ke_exit = F64(0.5) * mass_weight * fspeed * fspeed
                        else:
                            fast_wall = F64(1.0)
                            pz_wall = mass_weight * ivz
                            if fate == 2:
                                fast_unresolved = F64(1.0)
                elif selector < total:
                    mex = F64(1.0)
                    changed = 1
                    u6 = F64(wp.randf(state))
                    u7 = F64(wp.randf(state))
                    cos_chi = F64(1.0) - F64(2.0) * u6
                    sin_chi = wp.sqrt(wp.max(F64(0.0), F64(1.0) - cos_chi * cos_chi))
                    phi = F64(6.283185307179586) * u7
                    gpx = g * sin_chi * wp.cos(phi)
                    gpy = g * sin_chi * wp.sin(phi)
                    gpz = g * cos_chi
                    new_vx = F64(0.5) * (ivx + nvx) + F64(0.5) * gpx
                    new_vy = F64(0.5) * (ivy + nvy) + F64(0.5) * gpy
                    new_vz = F64(0.5) * (ivz + nvz) + F64(0.5) * gpz
                else:
                    null = F64(1.0)
                if changed != 0:
                    ke_before = F64(0.5) * mass * (ivx * ivx + ivy * ivy + ivz * ivz)
                    ke_after = F64(0.5) * mass * (new_vx * new_vx + new_vy * new_vy + new_vz * new_vz)
                    energy_loss = (mass_weight / mass) * (ke_before - ke_after)
                    dpz = mass_weight * (new_vz - ivz)
                    vr[p] = new_vx
                    vt[p] = new_vy
                    vz[p] = new_vz
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(candidate)), offset=base + ION_STATS_CANDIDATES)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(cex)), offset=base + ION_STATS_CEX)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(mex)), offset=base + ION_STATS_MEX)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(null)), offset=base + ION_STATS_NULL)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(cex_plume)), offset=base + ION_STATS_CEX_PLUME)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(ceiling)), offset=base + ION_STATS_CEILING)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(fast_exit_channel)), offset=base + ION_STATS_FAST_EXIT_CHANNEL)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(fast_exit_plume)), offset=base + ION_STATS_FAST_EXIT_PLUME)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(fast_wall)), offset=base + ION_STATS_FAST_WALL)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(fast_thermal)), offset=base + ION_STATS_FAST_THERMAL)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(fast_unresolved)), offset=base + ION_STATS_FAST_UNRESOLVED)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(energy_loss)), offset=base + ION_STATS_ENERGY_LOSS)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(dpz)), offset=base + ION_STATS_PZ_IONS)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(pz_exit)), offset=base + ION_STATS_PZ_FAST_EXIT)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(pz_wall)), offset=base + ION_STATS_PZ_FAST_WALL)
        wp.tile_atomic_add(stats, wp.tile_sum(wp.tile(ke_exit)), offset=base + ION_STATS_KE_FAST_EXIT)


__all__ = ["ION_STATS_COUNT", "ION_STATS_KEYS"]
