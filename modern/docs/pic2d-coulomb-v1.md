# `coulomb_v1` - Coulomb collisions of PIC model v2.4.0 (R4)

**Status: code and tests complete; shakedown only; no preregistered run, no result.** Implements R4 (gap (d)
of section 4.d) of `modern/docs/pic2d-physics-completeness-audit.md`. Spec entry:
`modern/spec/pic2d/pic2d-model-v2.4.json` -> `coulomb_v1`. Code: `cft_revival.pic2d.coulomb` (numpy reference,
configuration, reference rates) and `cft_revival.pic2d.warp_coulomb` (device stage). Tests:
`modern/tests/pic2d/test_pic2d_v24_coulomb.py`.

## 1. What the operator does

Every `cycle_steps` = k steps (default 10, `Delta t_c = k dt`), after the Boris push and the wall absorption and
before the ion MCC / anomalous / electron MCC stages and that step's births:

1. the alive particles of each species are grouped by mesh cell (i, j);
2. **e-e**: the members of a cell are randomly permuted and paired consecutively (Takizuka & Abe 1977); an odd count
   uses the triplet rule - members 0, 1, 2 collide pairwise (0,1), (0,2), (1,2) with `Delta t_c / 2` each, so every
   particle gets one full time step of scattering; a cell with one electron does not collide;
3. **e-i**: every electron `l` of a cell collides once with ion `(l + shift) mod N_i` (random shift per cell) at the
   field density `n_i = N_i W / V_cell`; the ions collide `N_e / N_i` times on average - the Takizuka-Abe rule for
   unequal groups, which gives both species the physical rate (the electron's proportional to `n_i`, the ion's to
   `n_e`);
4. **i-i** (optional, default off): as e-e on the ion arrays with the i-i Coulomb logarithm;
5. each binary collision draws Nanbu's (1997) cumulative scattering angle for the pair's deflection parameter
   `s = (ln Lambda / 4 pi) (q_a q_b / (eps0 m_ab))^2 n_field Delta t / g^3` (`m_ab` the reduced mass, `g` the
   relative speed; `<1 - cos chi> = 1 - exp(-s)`, `<chi^2> = 2 s` for small `s` = the Takizuka-Abe variance
   `<tan^2(chi/2)> = s/2`) and a uniform azimuth, then rotates the relative velocity in the centre-of-mass frame:
   `v_a' = v_a + (m_b/M) Delta u`, `v_b' = v_b - (m_a/M) Delta u`, `|u + Delta u| = |u|`. With the shared macro
   weight the pair conserves momentum and classical kinetic energy to round-off.

The Coulomb logarithm is the NRL Formulary value from the cell's own `n = N W / V_cell` and
`T = m (<v^2> - |<v>|^2) / 3e` (temperature floor 0.01 eV, value floor 2.0; a fixed value is available and
declared). At the audit's plateau (1e18 m^-3, 7 eV) both electron logarithms are 12.1.

`(v_r, v_theta, v_z)` of a pair are treated as Cartesian components of one local frame (the MCC convention), so
`v_z` momentum is conserved exactly: `pz_coulomb` is zero by construction. The ledger's kinetic energy is
relativistic, so the classically elastic pairs change it by `O((v/c)^2)` of the redistributed energy (~1e-9 of
K_e per record); this remainder is tallied as `ke_coulomb_j` and booked so the particle-side identity closes to
round-off - no physical energy term is added.

## 2. Pairing on the GPU without breaking the RNG contract

Every per-particle random stream of the code (MCC, injection, anomalous scattering, ion MCC, SEE) is keyed on
the particle's slot index, so a physical cell sort would change every draw and break the bitwise replay of the
recorded runs. The stage therefore builds, per cycle, a cell-sorted **permutation** of the alive slots and pairs
through it; no particle is moved:

* `cell_kernel`: cell of each alive slot, provisional in-cell position from an integer atomic (order-dependent);
* exclusive scan of the cell counts -> segment starts; `scatter_kernel` -> provisional segment list;
* `rank_kernel`: rank = number of same-cell alive slots with a smaller slot index -> the segment order is the slot
  order whatever the atomic arrival order was (deterministic: graph = direct bitwise, same seed = same run);
* `prepare_kernel` (one thread per cell, sequential over the segment): moments in sorted order, Fisher-Yates
  shuffle of the segment, the e-i shift draw, the electron-seconds window sum;
* `like_kernel` (one thread per sorted position; the first member performs the pair; the thread of member 0 runs
  the odd-count triplet sequentially) and `unlike_kernel` (one thread per sorted **ion** position; it loops over
  its electrons `l = l0 + m N_i` - an electron belongs to exactly one ion, so no two threads write one particle).

All launches have fixed shapes (capacities / cell count) and read only device arrays, so the stage is captured
in the step graph (the graph key gains the Coulomb-cycle flag). Draws come from a new seed-table column (5,
stream id 6; CPU stream `[seed, step, 6]`); columns 0-4 are unchanged, so Coulomb off replays bitwise (the
v2.2.0 pin is re-asserted). The dominant cost is the rank kernel, `O(sum_cells k^2)` with `k` the cell occupancy;
the H100 numbers are in the shakedown README.

## 3. Time step

At 1.4 ps and the ext-val point (1e19 m^-3, 5 eV, ln Lambda ~ 10) `nu_ee dt ~ 4e-5`; k = 10 gives a peak-cell
`s ~ 4e-4` per cycle (the mean over pairs is larger, `s` being proportional to `1/g^3`) and `k dt = 14 ps` is
under half the 33 um cell crossing of a 5 eV electron (33 ps). Nanbu's angle is exact for any `s`, but the
once-per-cycle pairing with a fixed partner is coarse for `s >~ 1` (the development runs relaxed at 0.79-0.83
of the Trubnikov rate at mean `s` 0.3-0.8, 0.96-1.05 at mean `s` 0.04): the series records the mean `s` per pair
and the fraction of pairs beyond `s = 1` so a protocol can be checked; k = 1 and k = 2 relax at the same rate
within statistics (test).

## 4. Verification (numpy reference; the Warp stage on the cpu device; cuda on the box)

| check | result |
|---|---|
| pair kinematics | relative speed, momentum and classical energy to 1e-13 relative; the applied angle is the sampled one; uniform azimuth |
| Nanbu sampler | `<1 - cos chi> = 1 - exp(-s)` on every branch (1e-4 ... 8); `<chi^2> = 2 s` at small `s`; isotropic at large `s` |
| Trubnikov isotropization (bi-Maxwellian, 1e21 m^-3, lnL 10) | log-decay ratio measured / NRL path integral 0.963 (T_par 8 / T_perp 4 eV) and 1.048 (5.5 / 4.75 eV) at N = 200k, mean s 0.043; test tolerance 10 % (numpy, warp-cpu, cuda) |
| Spitzer / Lorentz drift decay (Maxwellian electrons on cold Xe+) | measured / bounded expectation 0.959 over 300 cycles (N 200k); initial slope = Braginskii `2.91e-6 n lnL T^-3/2` within 10 %; ion momentum gain = electron loss |
| two-temperature e-i exchange | the Landau integral realised by the formed pairs = Spitzer `nu_eps (T_i - T_e)` within 8 % for m_i / m_e = 10, 100, 1000 (cold ions); realised / expected 1.003 (m_i = 10 m_e, 200 cycles); without like-particle collisions the temperature decay falls to 0.74 of Spitzer (the electrons leave the Maxwellian) - physical, hence the comparison against the actual distribution |
| heavy-ion bound (Xe, ions at rest) | per-collision electron energy change < 4 x 4 m_e / M, speed kept to 1e-4; the electrons cool, the ions warm |
| conservation in a discharge (cpu / warp-cpu / cuda) | `pz_coulomb` < 1e-9 of the represented momentum per cycle; `ke_coulomb_j` < 1e-4 K_e; particle-side identity to 1e-6 |
| identity / off state | every parameter in `config_sha256`; off = the v2.2.0 pin (identity, ledger keys, window sums, 300-step tallies on cpu and warp-cpu) |
| graph / replay | CUDA graph vs direct bitwise with e-e + e-i + i-i on; same seed replays the same state |

## 5. Diagnostics

Series record block `coulomb`: `nu_ee_mean_per_s`, `nu_ei_mean_per_s`, `nu_ii_mean_per_s` (the operator's
pair-mean deflection rate `<s> / dt_c`, from `nu_ee = 2 sum s / (sum_cycles N_e Delta t_c)`), `mean_s_*`,
`fraction_large_s_*`, `mean_coulomb_log_*`, `interval_*_pairs`, `interval_cycles`, `interval_pz_coulomb_kg_m_s`,
`interval_ke_coulomb_j`, `nu_en_elastic_mean_per_s` (the MCC elastic tally per electron per second),
`nu_ee_over_nu_en`, and `nu_e_spitzer_peak_per_s` / `nu_e_spitzer_peak_over_nu_en` - the NRL electron collision rate
`2.91e-6 n lnL T^-3/2` at the record's peak node, the audit's gap-(d) definition. The two frequency definitions
differ by design: the pair-mean averages `nu_pair` proportional to `1/g^3` over the formed pairs, which the slowest
pairs dominate (the population mean diverges logarithmically and the sample mean grows with the sample size - the
box shakedown read 13x the Spitzer value at the peak cell); it is the operator's realised deflection statistic,
while the Spitzer form is the smooth function of n_e and T_e the audit's estimate uses. Maps / frames:
`coulomb_nu_ee_per_s`, `coulomb_nu_ei_per_s`, `coulomb_mean_s_ee` per cell in the node layout (cell (i, j) at
node index (i, j)); `coulomb.column_frequency_profile` gives the electron-weighted column mean at the cusp planes,
and the shakedown readings add the Spitzer-form frequency at the peak cell and per cusp column.

## 7. H100 record (2026-09-04, extra MPS client, contended)

Cost at a 4.5 M-particle plateau load on the 90 x 720 grid: Coulomb cycle 4.49 ms (cell sort of both species
1.14, prepare + pair kernels 3.34); inside the captured step 11.45 vs 6.62 ms -> 0.48 ms/step amortised over
k = 10 = +7.3 % of the contended step (the shakedown ran at 4.79 vs the off twin's 4.52 ms/step, +6 %). Shakedown
100k steps through finalize + assess: `pz_coulomb` 5e-29 kg m/s and `ke_coulomb_j` 3.9e-16 J over 8.5e9 pair
collisions; mean s per pair 6e-5 / 4.5e-5 with 1e-6 of the pairs beyond s = 1; Spitzer nu_e at the window's peak
cell 2.8e5 /s against nu_en 1.17e7 /s (0.024 in the 1e17 m^-3 seed transient; ~0.3 scaled to the plateau peak);
S / I_d / I_beam within 0.5 % of the Coulomb-off twin, window T_e,peak +3 %, window peak n_e -2 % - shot noise of a
70 ns average, not a reading of the plateau hypotheses. Details: `experiments/pic2d_coulomb_v1_shakedown/README.md`.

## 6. Out of scope here

Weighted-particle (Nanbu-Yonemura) pairing (every species shares one macro weight, so not needed); relativistic
collision kinematics (Perez 2012; irrelevant below 1 keV); Langevin / grid-based operators (Lemons 2009);
large-angle single events beyond the Rutherford cut (Higginson 2017); e-i collisions with the ions' own
sub-cycle timing (velocities only, at the electron cycle; the ions' Coulomb momentum change is `2 m_e / M` per
collision).
