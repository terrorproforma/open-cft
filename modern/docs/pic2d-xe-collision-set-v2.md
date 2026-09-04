# `xe_collision_set_v2` - the xenon collision set of PIC model v2.3.0 (R3)

**Status: code, data and tests complete; shakedown only; no preregistered run, no result.** Implements
R3a (four excitation levels) and R3b (Xe⁺–Xe charge exchange + momentum transfer with the fast-neutral
thrust tally) of `modern/docs/pic2d-physics-completeness-audit.md` §5 (gaps e1 and e4 of §4.e). Spec entry:
`modern/spec/pic2d/pic2d-model-v2.3.json` → `xe_collision_set_v2`. Tests:
`modern/tests/pic2d/test_pic2d_xe_collision_set_v2.py`. Out of scope here and not attempted: e2
(metastables / stepwise ionisation, R5b - a 0-D metastable pool and its own cross-section spec, not
trivial), e3 (Xe²⁺, R6 - a second ion species).

## 1. Process table

| process | kind | threshold / energy loss | data | payload sha256 (spec file) | source, version, retrieval |
|---|---|---|---|---|---|
| `elastic` | e-Xe elastic (momentum transfer, isotropic) | 0 | byte-identical to v1 | `9b39858a…5698228` (`xenon-cross-sections-v2.json`) | LXCat **Biagi-v7.1** (Magboltz 7.1), export of 21 May 2023 mirrored in `lanl/ThunderBoltz` `bdd3013d…`; retrieved 2026-09-02T15:35:58Z, upstream sha256 `7019dd0b…` re-verified 2026-09-04T16:18:21Z (extract byte-identical) |
| `excitation_8p315` | e-Xe excitation | 8.315 eV | LXCat block `Xe -> Xe*(8.315eV)` | same file | same database |
| `excitation_9p447` | e-Xe excitation | 9.447 eV | `Xe -> Xe*(9.447eV)` | same | same |
| `excitation_9p917` | e-Xe excitation | 9.917 eV | `Xe -> Xe*(9.917eV)` | same | same |
| `excitation_11p7` | e-Xe excitation | 11.7 eV | `Xe -> Xe*(11.7eV)` | same | same |
| `ionization` | e-Xe single ionisation | 12.13 eV | byte-identical to v1 | same | same |
| `cex` | Xe⁺+Xe resonant charge exchange | 0 (resonant) | σ = (87.3 − 13.6 log₁₀ E/eV) Å², held at 0.1 eV below 0.1 eV, used to 2000 eV | `6f259ba9…bac079cb` (`xenon-ion-neutral-cross-sections-v1.json`) | Miller, Pullins, Levandier, Chiu, Dressler, *J. Appl. Phys.* 91, 984 (2002), doi:10.1063/1.1426246, Eq. (4); measured 1–300 eV |
| `mex` | Xe⁺+Xe momentum transfer (isotropic in the CM) | 0 | Phelps isotropic component = 3.39e-19 E^(−1/2) m² (rows of the LXCat table) | same file | Phelps database (LXCat) from Piscitelli, Phelps, de Urquijo, Basurto, Pitchford, *Phys. Rev. E* 68, 046408 (2003), doi:10.1103/PhysRevE.68.046408; bytes from `BLAST-WarpX/warpx-data` `MCC_cross_sections/Xe/ion_scattering.dat` @ `c42f106f` (sha256 `5f196141…`), retrieved 2026-09-04T16:21Z |
| (cross-check) | Phelps backscatter component | – | 1.17× (10 eV) … 1.41× (300 eV) the Miller fit | recorded in the same file, not a process | `ion_back_scatter.dat` sha256 `1d47da85…` |

Energy argument of the ion tables: E = ½ M_Xe |v_ion − v_atom|² (the ion's energy for an atom at rest - the
laboratory frame of Miller's beam data and of Phelps' compilations; the CM energy of the symmetric pair is E/2;
WarpX evaluates the same tables at the CM energy, a documented factor-2 difference worth ≤ 10 % of σ).

Why Biagi-v7.1 and not v8.9 / Hayashi: the audit's four levels **are** the Biagi-v7.1 levels. The other e/Xe sets
of the same LXCat export were inspected and rejected as the production set (recorded in the spec provenance):
"Biagi" (Magboltz 8.9 lineage) resolves 33 levels whose sum is ~2× lower than v7.1 above 20 eV; Hayashi has 6
partial levels summing to ~1/3; Morgan one lumped total ~1.3× higher; BSR 47 levels to 80–400 eV. Keeping v7.1
keeps elastic, excitation and ionisation in one swarm-validated set (Bordage et al. 2013). λ_CEX at n_g = 3e19,
300 eV: 62 mm (audit: ~60 mm).

Attribution check (test + recorded in the file): Σ_k σ_k of the four levels equals the v1 lumped table to 0.24 %
above 10 eV (4.9e-23 m² absolute; the residual is the v1 grid's linear interpolation across the sharp 9.447 /
9.917 eV onsets, up to 2.7 % where σ < 2e-21 m²). The total excitation frequency is therefore the v1 one; what
changes is the energy removed per event: 8.32 eV → 9.4 eV (12 eV electrons), 9.8 (20 eV), 10.0–10.1 (50–100 eV).

## 2. Algorithms

* **Electron MCC** (`mcc.py`, `warp_backend.mcc_kernel`): the uniform table has rows elastic, levels 1..n,
  ionisation; the null-collision selector picks the process by its position in the cumulative frequencies, and
  inside the excitation band the level by the per-level frequencies (= σ_k/Σσ at the electron's energy). The
  electron loses the level's threshold and is redirected isotropically. Tallies: total + per-level counts; the
  inelastic ledger is `W (Σ_k n_k E_k + n_ion E_ion) e` (v2.0.6 W convention). One level reproduces the
  v1.x–v2.0.6 arithmetic exactly (`0.0 + x == x`, one loop iteration, the same threshold value; the GPU reads
  the threshold from a one-element device array).
* **Ion MCC** (`ion_mcc.py`, `warp_ion_mcc.ion_mcc_kernel`): once per ion sub-step (dt_i = `ion_subcycle` × dt)
  on the pushed ions before that step's births join. Candidate with P = 1 − exp(−ν_max dt_i); for a candidate a
  Maxwellian atom velocity at T_g (`MCCConfig.neutral_temperature_k`) is sampled (Box–Muller, four uniforms - the
  construction the ion-birth sampling already uses), E = ½ M |v_i − v_n|², ν_k = n_g σ_k(E) |v_i − v_n| with the
  instantaneous n_g (inventory scale × plume cone shape, device-resident so the CUDA graph sees every inventory
  update), process by the selector u ν_max. ν_max = n_g0 max_E Σσ_k(E) √(2E/M) over the table (0.05 eV to
  2000 eV; the ceiling sits at the 2000 eV end because σ_CEX falls only logarithmically); a candidate above the
  ceiling is counted (`ion_mcc_ceiling_violations`) and the run fails closed at the next series record.
  CEX: velocity swap (ion ← atom velocity; the atom leaves as a fast neutral with the ion's velocity).
  MEX: relative velocity redirected isotropically in the CM, both partners keep their CM speeds.
  RNG: CPU stream `[seed, step, 4]`; GPU stream 3 of the per-step seed table (streams 0–2 unchanged).

## 3. The fast-neutral contract (what changed in the ledger / inventory)

A CEX event turns an inventory atom into the (slow) ion and the ion into a **fast neutral that is not a
particle**. Its fate is decided at the event by a straight-line flight through the cell mask in 0.5 min(dr, dz)
steps - the same free-molecular assumption the 0-D inventory makes for thermal atoms - with identical arithmetic on
both backends (`ion_mcc.fast_neutral_fate`, `warp_ion_mcc.fast_neutral_fate_march`):

| case | tally | inventory | energy ledger | momentum ledger |
|---|---|---|---|---|
| speed < 4 √(kT_g/M) (550 m/s at 300 K) | `fast_neutral_thermal` | stays (a thermal atom) | in `ion_neutral_loss_j` | in `pz_ion_collisions` (gas) |
| born in the channel, reaches the exit plane inside the aperture without touching a wall | `fast_neutral_exit_channel` | **sink F**: `V dn_g/dt = Q_in + R − S − F − c n_g`, ledger key `fast_neutral_exit` | `ke_fast_neutral_exit_j` | `pz_fast_neutral_exit` → `thrust_total_n = flux + cold gas + fast neutral`, `fast_neutral_thrust_n` |
| wall, cone, anode, outer box, unresolved march (`fast_neutral_unresolved`) | `fast_neutral_wall` | none (thermalises, returns as a thermal atom) | in `ion_neutral_loss_j` | `pz_fast_neutral_wall` → `force_on_thruster_n` |
| born downstream of the exit plane (plume domains) | `fast_neutral_exit_plume`, `cex_plume` | none (the plume gas is the effusion cone, like `ionizations_plume`) | `ke_fast_neutral_exit_j` | `pz_fast_neutral_exit` |

No atom is counted twice: inventory → (ionised → ion particle | effused | CEX fast neutral out); ion particle →
(beam | wall/anode recycled → inventory); the slow CEX ion is an ordinary ion afterwards. The energy identity
gains the sink `ion_neutral_loss_j = W Σ ½M(|v_i|² − |v_i'|²)` over CEX and MEX (particle side:
dKE = field work + injected − absorbed + born − W Σ n_k E_k − ion_neutral_loss, closes to round-off on cpu /
warp-cpu / cuda); the plasma momentum ledger gains `pz_ion_collisions`; `gas_momentum_rate_n = −pz_ion_collisions
− pz_fast_neutral_exit − pz_fast_neutral_wall` is what stays in the 0-D gas (thermal atoms taken up, MEX
recoils, slow CEX neutrals; assumed thermalised with the walls). The neutral ledger carries the sixth key only
when the ion MCC is on, so v1.3/v1.4 states and checkpoints are byte-identical; `artifacts` accepts the three
layouts. Series records gain `currents_a.{cex,mex,cex_plume,fast_neutral_exit,fast_neutral_wall,
fast_neutral_thermal,ion_mcc_candidate}_rate_per_s`, `ledger.interval_ion_neutral_loss_j`,
`ledger.interval_ke_fast_neutral_exit_j`, `neutral.fast_neutral_exit_rate_per_s`, and the momentum record the
`ion_collision_momentum_rate_n`, `fast_neutral_{exit,wall}_momentum_rate_n`, `gas_momentum_rate_n`,
`fast_neutral_thrust_n`, `fast_neutral_exit_power_w` (the runner's `records_to_arrays` carries them; NaN / 0 on
older records). Declared simplifications: one atom temperature in the CEX sampling (the wall-recycled
population is not sampled at T_w); fast neutrals hitting the wall have their energy booked, not a sputter yield.

## 4. Identity policy

`MCCConfig.collision_set` (`cross_sections_xe.CollisionSetConfig`: name, electron file + payload sha256 + process
list with thresholds, ion-neutral file + payload sha256 + processes + table grid + speed-threshold factor +
kinematics text) enters `MCCConfig.to_dict` and therefore `config_sha256`; it is present only when declared. A
protocol selects the set with `operating_point.collision_set = {"name": "xe_collision_set_v2", "ion_neutral":
true | false | {...}}`; the hashes are never read from the protocol - the named set is loaded from the spec
files and its recomputed payload hashes enter the identity. `Simulation` refuses a cross-section object whose
payload hash or process list differs from the declaration, and refuses an undeclared multi-level set. Checkpoints
keep binding `cross_section_sha256` (electron payload) and `config_sha256` (now carrying the ion payload).

Legacy: `MCCConfig` without a collision set is the v1 lumped set with collisionless ions; `to_dict` is unchanged,
so every recorded `config_sha256` is unchanged. Verified 2026-09-05 by an old-vs-new replay of the same legacy
configuration (12 × 96 CFT grid, 300 V, 50 mA injection, n_g 1e21 with a recycling inventory, ion_subcycle 4,
300 steps) on the `0901138a` tree and on this tree: identical sha256 digests of (electrons, ions, φ, surface
charge), of the full cumulative ledger including the float sums, and of the neutral state, on the numpy backend
(`4e512742…`) and on the Warp CPU backend (`6b775d91…`); `SEED_STREAMS` 3 → 4 appends the ion stream without
changing streams 0–2. The CUDA graph-vs-direct bitwise test with the ion MCC on is part of the suite.

## 5. Predeclared expectations (from the audit; a result of the opposite sign is a finding)

Reference: the ss-v4 33 µm plateau `0d228ad2`. I_d ≈ unchanged (< 5 %); inelastic power +~15 %, T_e −3–5 %,
S −3–5 % (R3a); the exit IEDF gains a low-energy population of 15–30 % of the exit ions
(1 − exp(−L/λ_CEX) for 12–24 mm flight paths at λ ≈ 62 mm); ion beam thrust down by the exchanged fraction,
`fast_neutral_thrust_n` takes it, total ≈ unchanged; anode ion current up, divergence up (MEX); the n_g fixed
point moves with S and the new sink F. Not expected: any change of the elastic / ionisation rates at fixed T_e
(byte-identical tables) or of the total excitation frequency at fixed EEDF.

## 6. Shakedown

`modern/experiments/pic2d_xe_collision_set_v2_shakedown/` (the ss-v4 protocol with the collision set, 100 000
steps with the v4 shakedown cadences through finalize + assess on the Lambda H100 as an extra MPS client) records
the early CEX / MEX rates, the fast-neutral bookkeeping and the exit IEDF shape in `shakedown.json` - readings,
not results. The coordinator schedules the R3 comparison runs.
