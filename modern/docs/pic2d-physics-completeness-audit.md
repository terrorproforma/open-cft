# `cft_revival.pic2d` physics completeness audit (2026-09-05)

**Status: read-only audit (document only). No code, spec, protocol, result or paper file is changed by
this document.** Prepared 2026-09-05 against `origin/feat/sota-foundation` at `036bd679` (the ext-val v0
launch-1 record) in a detached worktree; rebased onto `8c70cff0` (model v2.0.6, which landed while this was
being written - see the note in §4.k) before commit. The subject is the production PIC-MCC step in
`modern/src/cft_revival/pic2d/` (`mcc.py`, `neutrals.py`, `sensitivity.py`, `simulation.py`,
`warp_backend.py`, `poisson.py`, `mesh.py`, `kernels.py`, `fields.py`, `fieldlines.py`), the model specs
`modern/spec/pic2d/pic2d-model-v1.1.json` ... `v2.1.json` and `xenon-cross-sections-v1.json`, and the
experiments that consume them (`pic2d_cft_steady_state_v4/v5`, `pic2d_design_mini_sweep_v1`,
`pic2d_external_validation_v0`).

Companion documents (this audit extends them and does not repeat their content):

* `modern/docs/literature/pic-mcc-blockers.md` (116 refs, `ccb22d5d`): blockers 1-6 - Debye resolution,
  cost, neutrals, missing physics (CEX, SEE, anomalous transport, exit/cathode boundary, cusp sheaths),
  plateau statistics, validation targets. Cited below as **B-nn**.
* `modern/docs/literature/pic-acceleration-methods.md` (147 refs, 2026-09-04): every physics-neutral and
  physics-altering speed-up, with claim risk and verification protocol. Cited below as **A-nn**.
* `modern/docs/pic2d-performance-audit.md` (`c2d3b88d`, H100 cost anatomy): the measured step budget the
  "cost per step" column below is expressed against (solo channel-33 step 3.31 ms; physics kernels
  45-55 % of it; §11-12 the GMG Poisson solver).

The question this audit answers, under the user directive "make the PIC as fast AND accurate as
possible, prioritising accurate physics - every physical interaction of consequence must be in there":
**which physical interactions of consequence for a HEMP / cusped-field thruster (dielectric channel,
SmCo PPM stack, xenon, 1-10 mA, 300-500 V) are absent from the code today, what each absence does to the
quantities we claim (per-cusp losses, cusp sheath drops, I_d, S, utilisation, plateau existence,
thrust), what the state-of-the-art HEMPT PIC codes include, what each costs to add, and in which order to
add them.**

## 0. Method and honesty rules

* **Code facts** come from reading the files at `036bd679`; every "ABSENT" below was re-checked with a
  text search of `modern/src/cft_revival/pic2d/` (the IDE search tool skips git-ignored worktrees, so a
  shell `rg --no-ignore` was used). File:line references are to that commit.
* **Literature**: every DOI in §8 was resolved on 2026-09-05 against the Crossref works record
  (`api.crossref.org/works/<DOI>`; title, first author and year read back and checked against the
  intended paper). Ten guessed DOIs returned 404 or resolved to a different paper and were replaced
  through a web search of the publisher landing page (Brandt 2015, Kalentev 2014, Schneider 2009, Duras
  2014, Kahnfeld 2018, Fabris 2015, Stephan and Märk 1984, Reza 2023, Villemant 2017/2019); one item
  (Hayashi's NIFS xenon compilation) has no DOI and is named in the text only. **151 references
  resolved; only resolved DOIs are cited.** 60 of them also appear in the two sibling reviews and are
  marked `[also B-nn/A-nn]` in §8; the other 91 are new to the repository.
* Full texts read: Brandt et al. 2016 (the closest analogue; its model description is quoted in §4).
  Everything else is abstract-level plus what the sibling reviews recorded. Where a number is an
  order-of-magnitude estimate made here (§9) it is labelled "estimate" and the formula is given.
* "Silent" means no resolved paper was found; it does not mean no such paper exists.
* Effort key (house style): **S** < 1 day, **M** 1-3 days, **L** > 3 days, for one developer in this
  Warp code including tests. "Cost/step" is the expected change of the solo H100 channel-33 step
  (3.31 ms) from the performance audit's cost model (0.97 ms per M electrons, 4.1 us per dependent
  launch).
* Reference plateau for "expected change": the accepted ss-v4 33 um channel-only plateau
  (`0d228ad2`): I_d 3.80 mA, I_beam 2.46 mA, S 3.60e16 s^-1, gross utilisation 0.42, n_g 3.19e19 m^-3,
  peak n_e 1.29e18 m^-3, T_e,peak 5.58 eV; energy residual +2.46 % of electrode work on the corrected
  (v2.0.6) ledger - its predeclared acceptance (b) "< +2 %" therefore reads FAIL while (a) plateau and (c)
  convergence stand (`02013df0`); the 50 um base plateau reads +13.0 % (it was heating). The discriminating external case is Brandt et al. 2016
  (20 um, 2e20 m^-3 static neutrals, 400 V, 1.8 mA source): I_a 4.3 mA, I_beam 2.5 mA, n_i ~1e19 m^-3,
  cusp drops ~10 / ~5 V, wall ion energy ~160 eV; our closure without anomalous transport produced an
  ionisation avalanche there (S = 2.5x feed, I_a 2.61 mA, cusp drops -1.3 / +41 V, wall ion energy
  381 eV, record `036bd679`).

## 1. Summary

* **The code is a clean, well-instrumented electrostatic PIC-MCC with the minimum xenon chemistry:**
  three electron-neutral processes (elastic, ONE lumped 8.32 eV excitation, single ionisation), a
  0-D quasi-steady neutral inventory with wall-ion recycling, a homogeneous-Neumann dielectric with
  accumulated surface charge, a volumetric flux-tube cathode (v2.0) or exit-plane injection (v1.3), a
  static P2 FEM magnetic field, a closed energy and axial-momentum ledger, and fail-closed
  Debye / omega_pe / residual-power gates. Everything else in the list the user asked about is ABSENT
  or a switched-off scaffold (§3).
* **Eleven physics gaps were graded (§4).** Four change the quantities we claim at the tens-of-percent
  level and are the "physics first" items: (1) anomalous cross-field transport - absent by construction
  in 2D axisymmetric, present in every HEMPT PIC we could read (Brandt et al. 2016 impose Bohm
  D_perp = 0.4 kT_e/eB); its absence is the most likely cause of the ext-val avalanche; (2) secondary
  electron emission from the dielectric at the cusps - flux-averaged Vaughan yields 0.3-0.5 (BN) /
  0.4-0.8 (Al2O3) at T_e 7-10 eV (estimate, §9), i.e. cusp sheath drops 10-45 % lower and wall electron
  power x1.5-2; (3) ion-neutral CEX + momentum exchange - lambda_CEX ~ 60 mm at n_g 3e19 against a
  24 mm channel, so 15-30 % of channel-born beam ions charge-exchange before the exit (estimate), which
  reshapes the IEDF and moves thrust from the ion to a fast-neutral ledger the code does not have;
  (4) Coulomb collisions - nu_ee/nu_en is 0.15-0.4 at our 1e18 m^-3 plateau and 1.4-3.4 at Brandt's
  1e19 m^-3 (estimate), so the e-e term is second-order for our plateau and first-order for the
  code-to-code comparison.
* **Three are medium**: the lumped excitation (the four Biagi levels are already in the bound LXCat
  extract; per-event loss is under-estimated by up to 3.4 eV), metastable / stepwise ionisation
  (estimate 0.2x the ground-state rate at n_e 1e18-1e19, uncertain by x3), and spatial neutrals (the 0-D
  inventory cannot place the ionisation "flames" or the anode-side density; the plateau it produces is
  a property of the closure).
* **Four are low or diagnostic-only**: Xe2+ (0.3-2 % of ionisation events at T_e 7-15 eV from the ground state; -1.5 % thrust
  per 5 % current fraction), dielectric permittivity + backing (a transient / edge effect; the DC
  floating condition is what the Neumann + surface-charge model already gives), self-magnetic field
  (beta <= 4e-3, induced B <= 3e-5 T against 0.05-0.7 T applied: negligible; keep a diagnostic), ion-induced
  SEE and sputtering (erosion claims only; the wall energy-angle maps already exist).
* **One numerics item was a physics prerequisite and has just landed**: the `inelastic_loss_j` macro-weight
  omission (`mcc.py:371`, `warp_backend.py:2191` at `036bd679`) biased every recorded residual negative by
  the inelastic power (7-14 % of electrode work at plateau). Model v2.0.6 (`4b53012d`, `8c70cff0`, 01:40
  AEST) fixes it in both backends, recomputes every recorded residual post hoc and adds the accumulated-step
  peak-Debye floor; R0 of the roadmap is therefore DONE and every physics on/off comparison below is to be
  read on the corrected ledger (ss-v4 +2.46 %, 50 um base +13.0 %).
* **Ranked roadmap (§5)**, "physics first, then speed": R0 ledger fix (landed) -> R1 anomalous-transport
  alpha-series + the sealed ext-val `bohm-0.4` run -> R2 SEE (BN and Al2O3, Hobbs-Wesson cap) ->
  R3 full e-Xe set (4 excitation levels, CEX + MEX, fast-neutral thrust tally) -> R4 Coulomb
  collisions -> R5 spatial neutrals + metastable pool -> R6 diagnostics (Xe2+, beta map, sputter yield,
  neutraliser gas). About 15 H100 runs (75-90 GPU-h with MPS-4) verify it; each run's expected
  direction and rough magnitude is stated so the results can be checked against expectation.
* **What the 2D axisymmetric model can never claim (§6)**: a self-consistent anomalous cross-field
  mobility (the azimuthal E x B electron drift instability has no azimuthal coordinate to live in),
  azimuthal spokes, or a divergence pattern that depends on the off-axis neutraliser. Every I_d, S,
  utilisation and cusp-loss number is conditional on a declared alpha closure until an r-theta / z-theta
  companion campaign or a (wedge) 3D run supplies the mobility; §6 costs those options.

## 2. What the code has now

| process / model | implementation (file:line at `036bd679`) | data source | validation status in the repo |
|---|---|---|---|
| e-Xe elastic | isotropic redirect, speed preserved (2 m_e/M loss neglected); null-collision MCC on a 0.05 eV uniform table to 2000 eV (`mcc.py:156-199, 318-344`; GPU `warp_backend.py:739-776`) | LXCat Biagi-v7.1 (Magboltz 7.1) momentum-transfer set, LXCat export 21 May 2023 mirrored in `lanl/ThunderBoltz`; 122 points 0-1000 eV; payload sha256 `4d37732c...` (`xenon-cross-sections-v1.json`) | `test_pic2d_mcc.py` (rates vs null-collision expectation, tamper rejection), warp parity in distribution |
| e-Xe excitation | ONE lumped level, 8.32 eV loss, isotropic (`mcc.py:345-349`) | sum of the four Biagi levels 8.315 / 9.447 / 9.917 / 11.7 eV (builder `spec/pic2d/build_xenon_cross_sections.py:297-304`); Szabo 2001 convention | as above; per-event loss under-estimated by <= 3.4 eV (spec notes) |
| e-Xe single ionisation | 12.13 eV; Vahedi-Surendra secondary energy, B = 8.7 eV; both electrons isotropic; ion born at the event with a 300 K Maxwellian (`mcc.py:355-368`; GPU `:784-819, 875-887`) | Biagi-v7.1 | `test_mcc_run_creates_ions_and_reports_rates`; ionisation-rate maps integrate to S (renderer tests) |
| ion-neutral CEX / MEX / elastic | ABSENT (declared OFF: `mcc.py:8-10`, `simulation.py:1139-1141`, spec v2.0 `neutrals_v2_0.ion_neutral`) | - | - |
| e-e / e-i Coulomb | ABSENT | - | - |
| Xe2+, metastables, stepwise ionisation, volume recombination | ABSENT (single species `xenon_ion_species`, `models.py:298-299`) | - | - |
| neutrals | 0-D quasi-steady inventory `V dn_g/dt = Q_in + R - S - c n_g (- artificial relaxation)`, exact per-interval integration, five atom ledgers; wall-ion recycling R with gamma = 1 at T_w (v1.4); plume: analytic capped-cosine effusion shape as an MCC density factor only (`neutrals.py`; `simulation.py:1131-1167`); density in a device array so the CUDA graph sees it (`warp_backend.py:750-753`) | Szabo 2001 / Brandt 2016 recycling convention (spec v1.4) | 17 tests in `test_pic2d_neutral_inventory.py`; graph-staleness regression test |
| dielectric wall (Poisson) | finite-volume Gauss law; no conductance into solid cells -> homogeneous Neumann (zero field inside the dielectric) + accumulated surface charge on the plasma-side wall nodes as a RHS source; wall nodes float (`poisson.py:1-11`, `mesh.py:161-203`, `simulation.py:869-871`) | - | Gauss law with volume + surface charge (`test_pic2d_mesh_poisson.py`, MG path) |
| dielectric wall (particles) | absorbed, charge deposited, KE and p_z tallied; ions recycled into the inventory (`kernels.py:160-242`, `simulation.py:1016-1030`) | - | boundary classification tests |
| SEE (electron-induced) | SCAFFOLD ONLY: Vaughan 1989 yield with provisional BN parameters (delta_max 2.9, E_max 350 eV, E_0 12.5 eV), Hobbs-Wesson limit as a check, virtual per-column yield diagnostic; `enabled=True` refused (`sensitivity.py:90-197`, `simulation.py:231-234`) | Dunaevsky 2003 / Tondu 2011 named, not digitised | `test_see_scaffold_vaughan_yield_and_fail_closed_enable` |
| ion-induced SEE, sputtering, erosion | ABSENT | - | - |
| anode / exit / far field | Dirichlet (anode V_a; exit or far plane V_exit = chamber reference); absorbing; I_d = e(anode_e - anode_i)/interval (`poisson.py:50-54`, `simulation.py:1856-1866`) | - | ledger tests |
| cathode | v1.3: exit-plane fixed current, half-Maxwellian into the channel (`simulation.py:1040-1105`); v2.0: volumetric annulus in the channel-connected flux tube, `fixed` or `continuity` (rate follows I_d, clamped) (`:71-135, 2133-2156`); field-line connectivity gate (`fieldlines.py`) | Szabo 2001 quasineutrality injection; Charoy 2019 cathode plane | plume tests, fieldline tests |
| anomalous transport | HOOK, default OFF: isotropic redirect with P = 1 - exp(-alpha omega_ce dt), energy-conserving, tallied (`sensitivity.py:43-87`; GPU `bohm_kernel` `warp_backend.py:996-1042`); bracket alpha in {1/64, 1/16}; ext-val `bohm-0.4` variant sealed, never run | Smirnov 2004 (nu_B ~ omega_c/16), Brandt 2016 (D_perp = 0.4 kT_e/eB) | `test_bohm_scattering_preserves_speed_and_matches_the_rate` |
| magnetic field | static P2 FEM map (hash-bound `A_phi` checkpoint -> bicubic psi -> node samples; plume extension v1/v2) (`fields.py`, `p2_field.py`) | L1b material-aware P2 | field tests, anchors, cross-platform identity |
| self-magnetic field | ABSENT (electrostatic, E_theta = 0) | - | - |
| pusher / weighting | relativistic Boris (`kernels.py:96-137`), bilinear CIC deposit + gather (momentum-conserving), cylindrical shape volumes with Verboncoeur 2001 axis correction, fixed-point deposition bitwise across CPU/GPU, ion subcycling k = 8 | Verboncoeur 2001 | 10 kernel tests; Boris vs orbit_mc to round-off |
| smoothing / filtering, merging / splitting | ABSENT | - | - |
| energy ledger | K_e + K_i + U_field vs injected - absorbed - inelastic + born + electrode work (`simulation.py:1836-1892`); at `036bd679` **`inelastic_loss_j` lacked W** (`mcc.py:371`, `warp_backend.py:2191`) - fixed in model v2.0.6 (`4b53012d`), post-hoc corrector `ledger_recompute` | - | v2.0.6 identity tests close to round-off on cpu / warp-cpu / cuda; every recorded run carries a `ledger-corrected.json` sidecar |
| momentum / thrust ledger | axial momentum ledger (impulse, collisions, born, injected, exit, wall, anode); thrust = far-field flux + cold-gas term; Maxwell-stress force on electrodes as an independent check; IEDF (256 bins), divergence (1 deg bins) (`simulation.py:1595-1667, 1981-2043`) | - | 23 plume tests |
| gates | a-priori Delta/lambda_D, omega_pe dt, omega_ce dt, Courant; runtime peak-Debye window gate (hard pi, soft 2.5, >= 32 macro-electrons), omega_pe dt on resolved nodes (v2.0.4), one-sided windowed residual-power gate >= 5 % (v2.0.3), plume-boundary pile-up gate (v2.0.2) | - | `test_pic2d_v203_gates.py`, `v204`, `v14_gates_hooks` |
| 3D / r-theta | ABSENT (particles carry v_theta, no theta position) | - | - |

## 3. Gap table against the state of the art (condensed)

"SOTA HEMPT" = what the closest published cusped-field PIC codes include. Brandt et al. 2016 (full text):
"electron-neutral elastic, ionisation, excitation, coulomb and charge exchange collisions"; neutrals from a
DSMC solution imported and held static, diffusely reflected at the walls at 500 K; Bohm
D_perp = 0.4 kT_e/eB imposed by rotating the perpendicular velocity of randomly selected electrons; a 1 mm
dielectric with its dielectric constant and grounded elements behind it; surface-charge accumulation on the
tube and on its top face; "a simple secondary electron emission model ... 50 % of the electrons are
re-emitted with 90 % of their incident energy"; electron source at the plume boundary; 1024 x 256 cells of
20 um, dt 3.17 ps, super-particle ratio 1:2618. The Greifswald code family (Tskhakaya et al. 2007 method
paper; Schneider et al. 2009; Matyash et al. 2010; Kalentev et al. 2014; Duras et al. 2017; Kahnfeld et
al. 2018, 2019; Matthias et al. 2019, 2020) adds, as recorded in B §0.2 / §3 / §4 / §6, neutral dynamics
that produce the ~100 kHz breathing mode, CEX post-processing to 1 m, SDTrimSP erosion post-processing and
non-equidistant plume grids. The LANDMARK community codes (Charoy et al. 2019; Villafana et al. 2021)
are 2D axial-azimuthal / radial-azimuthal E x B benchmarks with prescribed ionisation and no cusp; they
define the standard for the instability physics an axisymmetric code cannot carry.

| # | physics | in our code | in SOTA HEMPT PIC | consequence for OUR claims (direction; rough size) | effort | cost / step | rank |
|---|---|---|---|---|---|---|---|
| a | SEE from the dielectric (BN / Al2O3) | scaffold, emission refused | Brandt 2016: 50 % re-emitted at 90 % energy (crude); Matyash 2010: wall contact confined to the cusps | cusp sheath drops -10 to -45 % (delta 0.3-0.9); wall electron power x1.5-2; T_e,peak -10 to -25 %; I_d +10 to +30 % (near-wall conductivity); ion wall energy down | M-L (3-5 d) | +2-4 % (wall kernel + emission spawn) | 2 |
| b | dielectric permittivity + backing (floating capacitor vs Neumann) | Neumann + surface charge (zero field inside the solid) | Brandt 2016: epsilon_r in the solver, 1 mm dielectric, grounded elements behind | DC floating condition identical (Gamma_e = Gamma_i locally); differences are the charging transient (~0.1-1 ms with a backing, ~10-100 ns without) and tangential coupling near the exit lip / anode edge (~2 % of the sheath charge, estimate §9) | M (1-2 d) | ~0 (a few more conductances) | 9 |
| c | anomalous cross-field transport (azimuthal ECDI) | Bohm hook OFF by default; `bohm-0.4` sealed, never run | every HEMPT PIC imposes it (Brandt 2016 D = 0.4 kT_e/eB "derived from a 3D simulation"; Szabo 2001/2014 Bohm bracket); 2D-theta / 3D benchmarks compute it | the central limitation: without it the ext-val point avalanches; with alpha in {1/64, 1/16, 0.345} expect I_d up (+20 to +60 %, closure-set), peak n_e and S down (-30 to -60 % at Brandt's point), cusp sheath drops down, per-cusp wall loss up; plateau existence becomes robust (a leak path bounds n_e) | S-M (hook exists; perpendicular-rotation variant 1 d; alpha-series protocol 1 d) | +1-2 % (one more per-electron kernel) | 1 |
| d | Coulomb collisions (e-e, e-i) | ABSENT | Brandt 2016 include "coulomb" collisions; Tskhakaya 2007 method | nu_ee/nu_en = 0.15-0.4 at 1e18 m^-3 / 5-10 eV, 1.4-3.4 at 1e19 (estimate); tau_ee 0.06 us at 1e19 << transit: EEDF Maxwellianised, tail refilled -> S +5 to +20 % at our plateau, first-order at Brandt's point; T_e,peak -5 % | L (3-5 d: cell sort + Takizuka-Abe / Nanbu pairing, subcycled) | +15-30 % if every step; +2-3 % subcycled every 10-20 steps | 4 |
| e1 | multi-level excitation (4 Biagi levels) | lumped 8.32 eV | Biagi set has 4 levels | inelastic power +~15 %; T_e -3 to -5 %; S -3 to -5 % | S (0.5-1 d; data already bound) | ~0 | 3a |
| e2 | metastables + stepwise ionisation (+ radiation trapping) | ABSENT | not in Brandt 2016; CR models exist for HET optics (Karabadzhak 2006) | estimate: stepwise/ground = 0.18-0.23 at n_e 1e18-1e19 (uncertain x3); channel optically thick to resonance radiation (k_0 L >> 1), so the whole 6s manifold is effectively metastable -> S +10 to +25 %, utilisation up, n_g down | L (3-5 d for a 0-D metastable pool mirroring `neutrals.py`; data: Hyman 1979, Ton-That 1977, Erwin 2004, Jung 2005) | +1 % | 5b |
| e3 | double ionisation Xe2+ (from Xe and from Xe+) | ABSENT | not in Brandt 2016 | 0.3-2 % of ionisation events at T_e 7-15 eV from ground (estimate; Rejoub 2002, Syage 1992 ratios); HET measurements: of order 10 % of the ion current in Xe2+ at 300-1000 V (Hofer 2006); thrust -1.5 % per 5 % Xe2+ current, I_d +1-3 % | M (1-2 d second ion species) | +1 % | 6 |
| e4 | ion-neutral CEX + MEX (Xe+ + Xe) | ABSENT | Brandt 2016 include CEX; Duras 2017 post-process CEX to 1 m | lambda_CEX ~ 60 mm at 3e19 (sigma 5.4e-19 m^2 at 300 eV, Miller 2002) vs 24 mm channel -> 15-30 % of channel-born beam ions exchange before the exit (estimate); IEDF gains a low-energy population; thrust moves to fast neutrals the ledger cannot see; divergence up (MEX); I_beam count unchanged | M (1-2 d; analytic Miller fits hash-bound; ions already subcycled) | +2-3 % | 3b |
| e5 | anisotropic e-Xe scattering; 2 m_e/M elastic loss; volume recombination | isotropic on the momentum-transfer set; loss neglected; no recombination | standard | none of consequence: isotropic-on-sigma_m is the consistent choice (Vahedi 1995; Janssen 2016); elastic loss 4e-6 per collision; recombination ~1e-6 of ionisation at 1e19 | - | - | - |
| f | spatial neutrals (DSMC / free-molecular / fluid), wall accommodation | 0-D inventory + analytic plume cone | Brandt 2016: static DSMC field with diffuse 500 K reflection; Kahnfeld 2018: neutral dynamics for breathing; Katz 2011 free-molecular algorithm | where ionisation happens (anode-side density high, exit depleted): the cusp "flames" move upstream, exit-cusp S down; n_g x sqrt(T_g/T_w) shift at fixed flux (-13 to -18 % for 400-500 K walls); the 0-D plateau is a property of the closure; the physical approach time is V/c ~ 0.2 ms | L (5-8 d: view-factor / DSMC-lite solved to steady state between windows, Picard-iterated with S(r,z)) | ~0 on the GPU step (host, per window) | 5a |
| g | cathode / neutraliser | exit-plane fixed current (v1.3) or flux-tube volumetric emission with continuity (v2.0); no cathode plasma, keeper, coupling voltage or neutraliser gas | Brandt 2016: source at the plume boundary; Charoy 2019: cathode-plane current continuity; hollow-cathode physics only in dedicated codes (Mikellides 2005; Sary 2017) | I_d in plume runs is set by the continuity rule (the 6 mA vs 3.44 mA plume/channel gap is a closure property); the neutraliser's own gas (10-20 % of the propellant) is missing from the plume n_g and from plume CEX; emission T_e 1-2 eV physical vs the declared value | S (neutraliser gas as a second cosine source: 1 d); L and out of scope (cathode plasma) | ~0 | 7 |
| h | ion-induced SEE, sputtering | ABSENT | Greifswald: SDTrimSP post-processing (B §6); Brandt 2016 report the wall ion energy-angle map for sputtering | gamma_i ~ 0.01-0.1 for Xe+ on ceramics at 100-400 eV (Hagstrum 1954; Baragiola 1979): a few % of the cusp electron balance; sputter yield needed for erosion claims only (Garnier 1999; Yamamura 1996) | S-M (1-2 d diagnostic on the existing wall maps) | ~0 | 8 |
| i | self-magnetic field | ABSENT (electrostatic) | electrostatic everywhere | beta <= 4.5e-4 (0.3 T) ... 4e-3 (0.1 T) at 1e19 / 10 eV; Hall-loop field <= 3e-5 T at 50 mA vs >= 3e-2 T within 0.1 mm of the nulls: negligible | S (0.5 d beta / B_induced diagnostic from the window current moments) | ~0 | 11 |
| j | plume: CEX, neutralisation, far-field BC, thrust ledger | no plume CEX; continuity cathode; Dirichlet far plane (v2.1 48 mm); flux + force ledgers close | Brandt 2016: Dirichlet->Neumann changed plume ratios, domain "too small"; Duras 2017: 1 m ray-trace with CEX | plume CEX few % over 24 mm at n_g <= 1e19 but sets the low-energy wings and back-flow; without fast neutrals the ion-thrust ledger under-counts by the in-channel CEX fraction; comparison with 1 m angular data needs a collisionless ray-trace stage | M (with e4; ray-trace stage 2-3 d) | +1 % | 6 |
| k | physics-consequential numerics | momentum-conserving explicit; ledger W bug fixed in v2.0.6 (after `036bd679`); W parity rule; hard pi gate; no smoothing | Brandt 2016: ~2 lambda_D per cell, no heating diagnostic; LANDMARK: cells <= lambda_D | the ledger fix was the prerequisite (DONE); EC gather (A §2.2) would remove the pi gate but puts 1-3-cell cusp sheaths at risk; 8.6x parity weight heats before the CIC threshold (ext-val) | S (W fix, landed) | 0 | 0 |

Ranks: 0 prerequisite; 1-4 "physics first" (change claimed quantities by tens of percent); 5-8 second
wave; 9-11 diagnostics or declared negligible.

## 4. The eleven gaps in detail

Each subsection: consequence for our claims -> literature evidence (numbers where the resolved papers or
the §9 estimates give them) -> what SOTA HEMPT codes include -> implementation in this Warp code (effort,
cost per step) -> verification protocol (what repo evidence would admit it).

### 4.a Secondary electron emission from the dielectric wall

*Consequence.* In a HEMP the wall contact is confined to the cusps (Matyash et al. 2010; our own maps
show the wall electron flux peaking at the three P2 cusp planes 6.03 / 12.0 / 17.97 mm). The cusp
sheath is where the per-cusp loss, the sheath drop and the wall ion energy we report are set. With a
yield delta the classical floating-sheath drop falls from T_e ln[(1-delta) sqrt(M/2 pi m_e)] = 5.3 T_e
(delta = 0) to 4.6 T_e (0.5), 3.0 T_e (0.9) and 1.2 T_e at the Hobbs-Wesson limit delta* = 0.983 for
xenon (§9). For a Maxwellian electron flux at the wall, the Vaughan curve with the repo's provisional BN
parameters gives a flux-averaged yield 0.14 (T_e 5 eV), 0.28 (7 eV), 0.49 (10 eV), 1.0 (20 eV); the
Al2O3-like parameter set (delta_max 6.4, E_max 650 eV) gives 0.22 / 0.44 / 0.77 / 1.6 (estimate, §9).
At the cusps the impacting electrons are the mirror-fed hot tail, so local yields near or above 1 are
plausible: the sheath is then space-charge-limited (Hobbs and Wesson 1967) or inverted (Campanell et al.
2012a). Kinetic results for BN Hall-thruster channels: SEE lowers the sheath potential and cools the
bulk (~20 %) while roughly doubling the near-wall electron mobility (Tavant et al. 2018); the EEDF at the
wall is non-Maxwellian and the effective yield lower than the Maxwellian estimate (Sydorenko et al.
2006a,b; Kaganovich et al. 2007); the SEE sheath is unstable and relaxation oscillations appear
(Sydorenko et al. 2006a; Campanell et al. 2012b); near-wall conductivity carries part of the discharge
current (Barral et al. 2003; Raitses et al. 2011); the non-linear coupling of SEE with the drift
instability changes the anomalous conductivity (Héron and Adam 2013); partial thermalisation of the
secondaries matters (Ahedo and de Pablo 2007). Measured yields for HET-grade ceramics: Dunaevsky et al.
2003 (BN, quartz), Tondu et al. 2011 (BN-based and Al2O3 channel ceramics), Villemant et al. 2019 (BN,
10-1000 eV) with the energy balance of the emitted population in Villemant et al. 2017; the two-component
(true secondary + elastically / inelastically backscattered) emission model is Furman and Pivi 2002 and
the compact yield formulas Vaughan 1989, 1993. For our claims: per-cusp electron loss x1.5-2 (delta ~
0.3-0.5 returns that fraction of the flux but the lower sheath admits more), cusp sheath drops -10 to
-45 %, T_e,peak -10 to -25 %, I_d +10 to +30 % (more electrons reach the anode through the cusp sheaths;
near-wall conductivity), wall ion impact energy down by the same sheath fraction (erosion claims), peak
n_e -5 to -15 %; plateau still exists.

*SOTA HEMPT.* Brandt et al. 2016: "50 % of the electrons are re-emitted with 90 % of their incident
energy" - an energy-independent, elastic-reflection-dominated model with no true-secondary population.
No cusped-field PIC quantifies SEE sensitivity at the cusps (B §4.2(e)).

*Implementation (M-L, 3-5 d; +2-4 % per step).* (1) Digitise Tondu 2011 / Dunaevsky 2003 / Villemant
2019 yields for BN and Al2O3 into a hash-bound `see-yields-v1.json` (Vaughan fit per material, plus the
elastic-backscatter fraction at low energy from Villemant 2017/2019). (2) In `push_kernel`'s wall branch:
sample the number of secondaries from the local yield (Poisson or Bernoulli on delta), spawn them at the
wall node with a half-Maxwellian at 2 eV (true secondaries) or the incident energy x reflection fraction
(backscattered), deposit the net surface charge; tally emitted count, energy and momentum in the ledgers
(both are closed identities: an emitted electron is an injected-energy term). (3) Hobbs-Wesson: when the
per-column effective yield reaches delta*, cap the emitted flux at the space-charge-limited value and
record the columns (fail-closed alternative: stop). (4) Spec `pic2d-model-v2.2.json`: `see` block enters
`config_sha256`.

*Verification.* Unit: yield curve vs the digitised points; emitted flux = delta x incident flux on a slab
with a prescribed beam; energy ledger closes to round-off with emission on; Hobbs-Wesson cap reproduces
the 1D sheath-drop formula on a planar slab against the analytic T_e ln[...] with delta. Manufactured:
the `test_debye_sheath_forms_in_slab_limit` case with SEE on must show the lower drop. Replay: the
accepted 33 um ss-v4 plateau with SEE off (bitwise identical to `0d228ad2`) and on (BN) and on (Al2O3) as
a declared three-way comparison; acceptance = the on/off differences in cusp drop, wall electron power,
T_e,peak carry the sign stated above and the residual-power gate never fires. Ext-val: `bohm-0.4` + SEE
is the run whose wall ion energy should approach Brandt's 160 eV.

### 4.b Dielectric surface charging and the floating dielectric boundary condition

*What we have.* Wall nodes are unknowns of the Poisson system; the finite-volume operator has no
conductance into solid cells, so the field inside the dielectric is zero and the plasma-side normal field
is sigma/epsilon_0 (`poisson.py:1-11`, `mesh.py:161-203`). This is the thick-dielectric (or
epsilon_r -> 0) limit of a perfect insulator: the wall floats locally at the potential where the net
particle current vanishes, with no surface conduction and no coupling to a backing conductor.

*What SOTA does.* Brandt et al. 2016 solve Poisson through a 1 mm dielectric with its dielectric constant
and grounded elements behind it, and accumulate the surface charge in the boundary cell; they note the
dielectric "is not enforcing a constant potential along its surface" and that the main potential drop sits
"beyond the positive anchor of the dielectric" at the exit. Circuit-coupled dielectric boundaries for PIC
are standard since Verboncoeur et al. 1993 (1D) and Vahedi and DiPeso 1997 (2D).

*Consequence (estimate, §9).* In DC steady state with zero surface conduction the floating condition is
the same in both models, so the sheath drops and wall fluxes of a converged plateau do not depend on
epsilon_r. The differences are (i) the charging transient: with a backing at ~0 V behind 1 mm of Al2O3
the surface must hold sigma = epsilon_0 epsilon_r E ~ 3.5e-4 C/m^2 for a 400 V plasma, which at an ion
flux 1e22 m^-2 s^-1 takes ~0.2 ms to charge - longer than any run we make, so Brandt's wall potential
was itself a transient unless initialised; ours charges in tens of ns; (ii) tangential coupling through
the solid, ~epsilon_r epsilon_0 Delta V / L against the sheath charge epsilon_0 Delta V / lambda_D, i.e.
~2 % for a 40 V per-cusp step over 6 mm; (iii) the edges - the exit lip, the front face and the anode
edge, where the dielectric is thin next to a conductor and the "anchor" Brandt describes forms. For our
channel-only plateau claims this is low consequence; for the location of the exit potential drop and the
breathing-mode dynamics it is not negligible.

*Implementation (M, 1-2 d; ~0 per step).* Give solid cells a conductance epsilon_r epsilon_0 A/L, add a
backing Dirichlet class (anode potential behind the channel liner up to the magnet stack; ground behind
the front face), keep the surface-charge RHS on the interface nodes. The multigrid operator-dependent
interpolation already handles jumps in conductance (§11 of the performance audit).

*Verification.* Manufactured: a dielectric slab between two electrodes with a prescribed surface charge
against the analytic two-region solution (second order); Gauss law with epsilon_r; the existing plateau
replay must be unchanged in the DC limit (the wall fluxes) and differ only in the charging transient - a
clean discriminating test of the argument above.

### 4.c Anomalous cross-field electron transport

*Consequence.* This is the structural limitation of the code and the most likely cause of the ext-val
avalanche. In a 2D (r,z) electrostatic model the azimuthal electric field is identically zero, so the
E x B electron cyclotron drift instability (Ducrocq et al. 2006; Adam et al. 2004; Lafleur et al. 2016a,b,
2017; Boeuf and Garrigues 2018; Janhunen et al. 2018a,b; Taccogna et al. 2019; Charoy et al. 2019, 2021;
Petronio et al. 2023a,b), rotating spokes (Janes and Lowder 1966; Ellison et al. 2012) and the coupled
drift-driven modes (Hara and Tsikata 2020; Tsikata et al. 2009 observed them) cannot exist. What
remains is classical collisional transport (nu_en ~ 8e6 s^-1 at n_g 3e19) plus axisymmetric low-frequency
dynamics. Between the cusps the electrons are tied to nearly axial field lines and cross to the anode
only at the cusp planes, so the discharge current, the electron density that builds up before the cusp
mirrors, and hence the ionisation rate are all set by how much cross-field transport the closure allows.
With none, the ext-val point at Brandt's density (2e20 static neutrals) produced S = 2.5x the feed, an
inventory doubling time of 0.24 us and cusp drops of -1.3 / +41 V against Brandt's ~10 / ~5 V. A Bohm
term at alpha = 1/16 is nu_an = omega_ce/16 = 5.5e8 s^-1 at 0.05 T and 3.3e9 s^-1 at 0.3 T, i.e. 70-400
times the e-n rate (§9): the closure, not the collisions, sets the transport. Expected changes under a
predeclared alpha-series {0, 1/64, 1/16, 0.345}: I_d up (+20 to +60 % at 1/16; the magnitude is what the
series measures - the sign is fixed), peak n_e and S down (-30 to -60 % at Brandt's point, less at ours),
cusp sheath drops down (the mirror-trapped hot tail leaks), per-cusp wall loss up, T_e,peak down; the
plateau becomes robust because the leak path bounds n_e. Every I_d, S, utilisation and per-cusp number is
a property of the closure until §6's companion campaigns supply the mobility.

*Literature evidence.* Kaganovich et al. 2020 (§VII): reduced 2D models "always show stronger
instability" and "significantly overestimated cross field mobility as compared with ... three dimensions";
Villafana et al. 2023 and the 3D HET codes that followed (Chen et al. 2025; Xie et al. 2024; Zhong et al.
2026) find the EDI intrinsically three-dimensional; Zhong et al. 2026 add realistic (r,z) field components
and a neutral solver and show the transport depends on the field geometry. Data-driven closures fitted to
measurements exist (Jorns 2018) and their self-consistent implementation is known to be delicate (Marks and
Jorns 2023). Reduced-order "quasi-2D" PIC recovers the azimuthal instabilities at 2-15 % of the 2D cost
(Reza et al. 2023; Faraji et al. 2023). In a cylindrical Hall thruster the inferred effective frequency is
nu_B ~ omega_c/16 (Smirnov et al. 2004). Axisymmetric (r,z) codes either impose Bohm (Szabo et al. 2014;
Brandt et al. 2016) or report the mobility that axisymmetric oscillations produce (B §4.3: Cho 2015).

*SOTA HEMPT.* Brandt et al. 2016: D_perp = 0.4 kT_e/eB, "derived from a 3D simulation of a similar
thruster model", applied by rotating only the perpendicular velocity so the parallel speed is unchanged.
Our hook redirects isotropically (also randomising the parallel speed; the exact equivalent factor for
alpha = 0.4 is 0.345, `comparison.py:59-61`), so the sealed `bohm-0.4` variant is a bracket of Brandt's
model, not the model.

*Implementation.* (S) add a `bohm_rotation` variant to `sensitivity.py` / `bohm_kernel` that rotates the
perpendicular velocity by a random angle about B (energy- and parallel-speed-preserving), 1 d; (S) an
alpha-series protocol on the ss-v4 template, 1 d. Longer options are §6: (ii) r-theta / z-theta companion
runs to extract a per-cell effective mobility (M-L), (iii) a wedge 3D run (L, weeks of GPU time), (iv) a
data-driven closure trained on (ii)/(iii) snapshots (L).

*Verification.* Unit: the rotation variant preserves |v| and v_parallel to round-off and reproduces the
declared nu_an on a uniform-B slab (already done for the isotropic hook); the diffusion coefficient
measured from test-particle spreading on a uniform B equals kT_e/(eB) x alpha/(1 + alpha^2). Replay: the
alpha-series on the accepted 33 um plateau, each run compared with `0d228ad2`; acceptance = monotone
I_d(alpha) and n_e(alpha) with the stated signs. External: the sealed `bohm-0.4` channel-20um run is the
discriminating comparison against Brandt's 10 rows (V&V20 tolerances 20 % / +-5 V / 0.3 dex); with the
ledger fix and the particle-step Debye floor in place (R0) it should reach a plateau rather than
avalanche.

### 4.d Coulomb collisions

*Consequence (estimate, §9).* With the NRL rate nu_ee = 2.91e-6 n[cm^-3] ln Lambda T_e^-3/2, and the
e-n momentum-transfer rate from the bound Biagi table at n_g 3e19 (k_el ~ 2.7e-13 m^3/s -> nu_en
~ 8e6 s^-1), nu_ee/nu_en = 0.02-0.04 at 1e17 m^-3, 0.15-0.4 at 1e18 (our plateau, peak 1.3-1.6e18) and
1.4-3.4 at 1e19 (Brandt's point). tau_ee at 1e19 / 7 eV is 0.06 us against a 2.4 us ion transit, so at
the ext-val density the EEDF is Maxwellianised within the electron residence time; the inelastic-depleted
tail is refilled and the ionisation rate coefficient rises (k_iz is exponentially sensitive to the tail at
T_e 5-7 eV). e-i collisions add a comparable momentum-transfer rate (Z = 1) but transport across B from
them is classical and 2-3 orders below the Bohm term. Expected: S +5 to +20 % at our plateau (more at
1e19), T_e,peak -5 % (tail energy shared), I_d +few %; mandatory for a like-for-like comparison with
Brandt, who include Coulomb collisions.

*Literature.* Binary pairing per cell: Takizuka and Abe 1977; cumulative small-angle theory and weighted
particles: Nanbu 1997, Nanbu and Yonemura 1998, Bobylev and Nanbu 2000; grid-based Langevin alternatives:
Lemons et al. 2009; full-angle including single large-angle events: Higginson 2017; relativistic /
collisional-ionisation extensions: Pérez et al. 2012; energy bookkeeping in implicit schemes: A-4 (Angus
2022). Dense-plasma practice: Sentoku and Kemp 2008.

*SOTA HEMPT.* Brandt et al. 2016 list "coulomb" among the implemented collisions; the Greifswald method
paper (Tskhakaya et al. 2007) describes the binary model.

*Implementation (L, 3-5 d; +15-30 % per step if every step, +2-3 % subcycled).* Requires a per-cell
particle order: a counting sort every K steps (A §6.2 - the acceleration review already recommends periodic
cell sorting for locality), then Takizuka-Abe pairing within cells (odd counts handled by the standard
triple), Nanbu's cumulative angle for nu dt ~ 1e-4 per step, e-i pairs with the ion subcycle. Because nu_ee
dt = 2e-5 per 1.4 ps step, collide every 20-50 steps with the accumulated s = nu_ee K dt: cost falls to
+2-3 %. Energy and momentum are conserved pairwise to round-off (the ledger is untouched by construction;
add the pair count to `cumulative`).

*Verification.* Unit: relaxation of a bi-Maxwellian to isotropy at the Trubnikov rate; energy and
momentum of each pair conserved to round-off; the Nanbu cumulative angle reproduces the Spitzer
slowing-down rate on a test beam. Replay: the 33 um plateau with Coulomb on/off as a declared pair; the
ext-val `bohm-0.4` + Coulomb run is the like-for-like comparison with Brandt.

### 4.e The full xenon collision set and its data provenance

*What we bind.* `xenon-cross-sections-v1.json` is a verbatim extract of the LXCat Biagi-v7.1 e/Xe set
(Biagi 1999 Magboltz lineage; LXCat platform Pitchford et al. 2017; cross-set comparison Bordage et al.
2013) with the payload sha256 recomputed on load and the resampled table hashed into every checkpoint.
Elastic is the momentum-transfer cross section (correct for isotropic MCC: Vahedi and Surendra 1995;
Janssen et al. 2016 on angular models); Ramsauer minimum 2.75e-21 m^2 at 0.62 eV, sigma_iz peak
5.6e-20 m^2 at ~110 eV. The lumped excitation follows Szabo 2001; the spec records a factor-4 spread on
it across Hayashi / SIGLO / Puech-Mizzi (Puech and Mizzi 1991) / Morgan as the dominant data uncertainty.
Ionisation totals agree with Rapp and Englander-Golden 1965; partial (Xe+, Xe2+, Xe3+) cross sections are
Stephan and Märk 1984, Syage 1992, Rejoub et al. 2002; near-threshold excitation benchmarks Allan et al.
2006 and Zatsarinny and Bartschat 2010; level-resolved 6p excitation Fons and Lin 1998; secondary-electron
energy spectra behind the Vahedi-Surendra sampling: Opal et al. 1971; anisotropic-scattering formulas
Okhrimovskyy et al. 2002, Surendra et al. 1990; swarm consistency via BOLSIG+ (Hagelaar and Pitchford
2005); textbook rates Lieberman and Lichtenberg 2005.

*e1 - multi-level excitation (S, 0.5-1 d).* Split the lumped channel into the four Biagi levels already
in the bound extract (8.315, 9.447, 9.917, 11.7 eV). Per-event loss rises by up to 3.4 eV; inelastic
power +~15 %; T_e -3 to -5 %; S -3 to -5 %. Verification: total excitation frequency unchanged to
round-off; ledger closes; plateau replay on/off.

*e2 - metastables and stepwise ionisation (L, 3-5 d).* The 6s manifold (two metastable, two resonance
levels) is fed by ~30 % of the lumped excitation (estimate). At n_g 3e19 over a 3 mm bore the channel is
optically thick to the 147 / 130 nm resonance lines (k_0 L >> 1; Holstein 1947), so the resonance levels
are effectively metastable and the pool is the whole manifold. Ionisation from Xe* has a 3.8 eV threshold
and sigma ~ 1-3e-19 m^2 (Hyman 1979; Ton-That and Flannery 1977; Erwin and Kunc 2004), i.e. k_iz(Xe*) ~
11x k_iz(Xe) at T_e 7 eV (estimate); excitation out of the metastables is measured (Jung et al. 2005). A
0-D metastable balance (production - electron quenching/ionisation - wall loss at v_th/L ~ 1e5 s^-1)
gives n_m/n_g ~ 1.5-2e-2 and stepwise/ground ionisation ~0.18-0.23 at n_e 1e18-1e19 (estimate, §9;
uncertain by x3 through the branching and quenching rates; the HET collisional-radiative models of Chiu
et al. 2006 and Karabadzhak et al. 2006 are the reference for the level kinetics). Consequence: S +10 to
+25 %, utilisation up, n_g fixed point down; T_e slightly down. Implementation: a metastable inventory
mirroring `neutrals.py` (0-D first, spatial with f), a fourth MCC process on the metastable density, a
hash-bound `xenon-metastable-cross-sections-v1.json`. Verification: the metastable ledger closes; the
0-D CR balance reproduces the analytic fixed point; plateau replay on/off.

*e3 - Xe2+ (M, 1-2 d).* Threshold 33.3 eV from ground, 21.2 eV from Xe+. With sigma(Xe2+)/sigma(Xe+)
~ 0.1 at 100 eV (Stephan and Märk 1984; Rejoub et al. 2002) the Maxwellian ratio of rate coefficients is
0.001 (5 eV), 0.003 (7 eV), 0.009 (10 eV), 0.02 (15 eV) (estimate); Hall-thruster measurements at
300-1000 V put of order 10 % of the ion current in Xe2+ plus a few % in Xe3+ (Hofer and Gallimore 2006). Thrust per
ampere of Xe2+ is 1/sqrt(2) of Xe+, so a 5 % current fraction is -1.5 % thrust and +1-3 % I_d.
Implementation: a charge-state field on the ion arrays (or a third device species), ionisation of Xe+ as
an MCC process on the ion density (needs ion density at the electron position - the deposited n_i map).
Low priority for the plateau; medium for thrust and IEDF claims.

*e4 - Xe+ + Xe CEX and momentum exchange (M, 1-2 d; +2-3 % per step).* Miller et al. 2002 (with Pullins
et al. 2000) give sigma_CEX(E) = 87.3 - 13.6 log10(E/eV) A^2 (5.4e-19 m^2 at 300 eV, 7.4e-19 at 10 eV);
the MEX cross section is of the same order. lambda_CEX ~ 60 mm at n_g 3e19, 180 mm at 1e19, 1.8 m at
1e18 (estimate). Over the 12-24 mm an ion travels from its birth to the exit, 15-30 % of channel-born
beam ions exchange: the ion is replaced by a slow one born at that potential (a low-energy population in
the IEDF, more anode-side ion loss) and a fast neutral leaves carrying the momentum. I_beam by count is
unchanged; ion thrust is under-counted by the exchanged fraction; MEX raises divergence. In the plume
(n_g 1e18-1e19 near the exit) it is a few % over 24 mm but sets the low-energy wings, back-flow and the
1 m angular distribution (Duras et al. 2017 post-process it; Boyd 2001 and Boyd and Dressler 2002 for
plume CEX practice). Implementation: an ion MCC pass on the neutral density (0-D scale x plume shape) with
the two analytic fits hash-bound as a spec; the ions are already subcycled (k = 8); add a fast-neutral
axial-momentum tally to the thrust ledger (`MOMENTUM_KEYS`) since neutrals are not particles.
Verification: the CEX event fraction on a uniform-density slab equals 1 - exp(-L/lambda); IEDF acquires
the low-energy population; momentum ledger still closes with the fast-neutral term; plateau replay on/off
(I_d, S nearly unchanged; IEDF and thrust split change).

*e5 - declared negligible.* Isotropic scattering on the momentum-transfer set is the consistent choice
(anisotropy would require the elastic, not the momentum-transfer, cross section); the 2 m_e/M elastic
energy transfer is 4e-6 per collision (~30 s^-1 x E against inelastic losses of 1e5-1e6 s^-1 x threshold);
radiative / three-body recombination is ~1e-6 of the ionisation rate at n_e <= 1e19 (alpha_rec ~ 1e-19
m^3/s).

### 4.f Neutral dynamics

*Consequence.* The 0-D inventory gives one density for the whole channel, so the ionisation maps (the
"flames" anchored at the cusp planes, brightest just downstream of each cusp) show WHERE the electrons
are, not where the neutrals are. In free-molecular flow through a bore of L/D = 4 with diffuse walls the
density falls from the anode plenum to the exit by a factor set by the Clausing conductance and by the
ionisation sink; Brandt et al. 2016 imported a DSMC field with diffuse 500 K wall reflection and measured a
25 % depletion; Kahnfeld et al. 2018 needed neutral dynamics to obtain the ~100 kHz breathing mode.
Expected with a spatial neutral model: ionisation moves upstream (anode-side cusps brighten, exit-cusp S
falls), the exit-plane n_g that sets plume CEX and cold-gas thrust drops, the fixed point n_g x
sqrt(T_g/T_w) shifts -13 to -18 % at fixed flux for 400-500 K walls (already in v1.4 through the mixture
effusion coefficient; a partial accommodation coefficient is unmeasured for BN / Al2O3 - B §3(e)), and
the plateau becomes a converged n_g(r,z) map instead of the fixed point of a closure whose physical
approach time is V/c ~ 0.2 ms. S and utilisation change by +-10 % (estimate); the direction depends on
whether the anode-side or the exit-side cusp dominates the ionisation, which is exactly what the sweep's
rho-dependence ("cusp-anchored flames at low rho -> exit-cell-dominated body at high rho") would resolve.

*Literature.* DSMC: Bird 1994; the free-molecular view-factor algorithm with ionisation and walls written
for thruster PIC: Katz and Mikellides 2011 (B-54); particle neutrals with accommodation in HPHall (B §3);
the fluid / hybrid cusped-field models of the HIT group (Liu et al. 2014, 2015; Zhao et al. 2014 PIC);
breathing theory in B §3.

*SOTA HEMPT.* Static DSMC field (Brandt 2016); dynamic neutrals for breathing (Kahnfeld 2018).

*Implementation (L, 5-8 d; ~0 GPU cost).* A free-molecular view-factor solver (Katz-Mikellides) on the
(r,z) mesh: anode-plenum source, diffuse wall re-emission at T_w, exit aperture sink, volumetric
ionisation sink from the window `ionization_rate_per_m3_s` map (already accumulated) and the wall-ion
recycling source from the wall maps; solved to steady state on the host between windows and pushed to the
GPU as the MCC `density_shape` array (which exists, `warp_backend.py:1587`); Picard iteration with the
plasma over successive windows; the 0-D inventory kept as the integral check. Verification: the Clausing
conductance of the bore against the analytic value; the uniform-density limit reproduces the 0-D
inventory to round-off; atom ledger closes; plateau replay 0-D vs spatial as a declared pair.

### 4.g Cathode and neutraliser

*Consequence.* v1.3 injects a fixed electron current at the exit plane; v2.0 emits in the
channel-connected flux tube with a `continuity` rule that ties the emitted current to the previous
interval's I_d (clamped). Neither is a cathode: there is no cathode plasma, keeper, coupling voltage,
emission-temperature physics or neutraliser gas flow. The plume runs' ~6 mA against the channel-only
3.44 mA is a property of the closure (uncapped flux-tube emission) and must be declared as such; the
neutraliser's own xenon (10-20 % of the propellant in a HEMPT) is absent from the plume n_g, so plume
CEX and cold-gas thrust are under-estimated. Brandt et al. 2016 placed the source "far outside" at the
plume boundary and note the outer domain "is still too small"; Brandt et al. 2015 showed the radial
distribution of the source electrons changes the in-channel plasma. The community practice is
cathode-plane current continuity at a declared T_e (Charoy et al. 2019; Szabo 2001 "preserving
quasineutrality"). Hollow-cathode physics lives in dedicated codes (Goebel et al. 2005, 2007; Mikellides
et al. 2005, 2008; Sary et al. 2017; ion-acoustic turbulence and energetic ions in the cathode plume:
Jorns et al. 2014; Hara and Treece 2019; textbook Goebel and Katz 2008).

*Implementation.* (S, 1 d) neutraliser gas as a second cosine source in `plume_neutral_shape` with its own
flow; (S) emission T_e 1-2 eV as the declared physical value; (L, out of scope for the plateau) a
coupling-voltage / keeper model. Verification: plume n_g map integrates to both flows; the `continuity`
closure and a fixed-current closure on the same plume box as a declared pair (the I_d difference is the
cathode-closure sensitivity to report).

### 4.h Ion-induced secondary emission and sputtering

Ion-induced (Auger) electron emission coefficients are ~0.01-0.1 for slow noble-gas ions (Hagstrum 1954
on metals; Baragiola et al. 1979 for light ions) - a few % correction to the cusp electron balance, below
the SEE uncertainty; off for plasma dynamics. Sputtering (Yamamura and Tawara 1996 yield formula; Garnier
et al. 1999 low-energy xenon sputtering of SPT ceramics; DCFT erosion data Gildea et al. 2013; Greifswald
SDTrimSP post-processing, B §6) matters only for erosion / lifetime claims, and Brandt et al. 2016 report
the wall ion energy-angle map for that purpose. Our wall maps already carry the ion energy and angle per
column; a yield post-processor is S-M (1-2 d). Not a plateau item.

### 4.i Magnetic field from plasma currents

beta = n k T_e / (B^2 / 2 mu_0) = 4.5e-4 (1e19 m^-3, 10 eV, 0.3 T), 4e-3 (0.1 T), 1.1e-3 (1e18, 7 eV,
0.05 T). The azimuthal Hall / diamagnetic current loop induces B ~ mu_0 I / 2r = 6e-6 T for 10 mA and
3e-5 T for 50 mA at r = 1 mm, against 0.05-0.7 T applied and >= 3e-2 T within 0.1 mm of the axis nulls
(gradient ~0.3 T/mm); the electrostatic approximation stands (the SOTA codes are all electrostatic).
Keep a 0.5 d diagnostic: beta and B_induced maps from the window current moments, recorded per run so
the statement is evidence rather than assertion.

### 4.j Plume: CEX, neutralisation, far-field boundary, thrust ledger

Plume CEX is e4 (few % over 24 mm at n_g <= 1e19; sets the low-energy wings and back-flow);
neutralisation is the continuity cathode (4.g); the far-field Dirichlet plane at the chamber reference is
the community choice, and Brandt et al. 2016 report that switching it to Neumann changed the plume
ratios - our v2.1 lesson (the plane must sit past the 10 % axis-density point and outside the
acceleration region) is the same finding. Thrust: our ledger closes flux against force (an independent
Maxwell-stress check), which is stronger than the exit-plane momentum flux alone; what it lacks is the
fast-neutral term (e4) and a collisionless ray-trace stage to 1 m for comparison with measured angular
distributions (Duras et al. 2017; plume codes Wang et al. 2001; Cichocki et al. 2017; Domínguez-Vázquez
et al. 2018; Boyd 2001). Effort M (2-3 d with e4).

### 4.k Numerics that are physics-consequential

The acceleration review owns this topic (A §2, §7); here only what changes a physics claim:

* **Energy ledger W omission (prerequisite R0 - LANDED as model v2.0.6 while this audit was written).** At
  `036bd679` `inelastic_loss_j` = macro-event count x threshold without W in both backends, so every recorded
  residual was H - L_inel. Commit `4b53012d` (01:40 AEST) multiplies the tally by `macro_weight` in both
  backends (physics bitwise; the unscaled sum kept as `inelastic_loss_per_weight_j`), proves the
  particle-side identity dKE = field work + injected - absorbed + born - W sum n E to round-off on
  cpu / warp-cpu / cuda, ships `cft_revival.pic2d.ledger_recompute` which wrote `ledger-corrected.json`
  sidecars for 13 recorded runs (`3ec2af92` ... `b498d2b7`), keeps the gate thresholds at 5 % hard / 2 %
  acceptance after a firing-table recalibration, and adds the accumulated macro-electron-step floor to the
  window peak-Debye gate (`8c70cff0`). Corrected trailing-window end states: 50 um base +13.0 % (seed-b
  +11.1 %, W x 0.7 +7.2 %: the three accepted 50 um plateaus were heating at 7-13 % of electrode power),
  ss-v4 33 um +2.46 % (acceptance (b) PASS -> FAIL, (a) and (c) stand), ext-val +61.7 % at the stop
  (crossed 5 % at 0.24 transits). Nothing physical changed; every on/off comparison below is to be read on
  the corrected ledger, and the 2 % acceptance is now the binding constraint on the 33 um plateau itself -
  which is one more reason the 25 um ladder point and the alpha-series (R1) come before any other physics.
* **Momentum- vs energy-conserving gather.** The explicit Lewis gather (Lewis 1970; Langdon 1970;
  Brackbill 2016; A §2.2) removes the pi Debye gate but breaks momentum conservation and puts the
  1-3-cell cusp sheaths at risk; semi-implicit ECsim (Lapenta 2017; Markidis and Lapenta 2011; Chen et
  al. 2011) leaves T_e and the cusp sheaths unresolved by construction. Both are "speed after physics"
  and must be validated against our own explicit 33 / 25 um ladder - there is no energy-conserving PIC of
  a cusped-field device in the literature (A §10).
* **W parity.** 8.6x the parity weight (ext-val, 2.6-58 macro-electrons per cell) heated stochastically
  before the CIC threshold (Hockney 1971; Ueda 1994; Cormier-Michel et al. 2008 on too-few-particle
  effects; Turner et al. 2013 on ppc convergence): W parity is a resolution requirement, not a cost knob.
  Variable-weight / merging schemes (Assous et al. 2003; Vranic et al. 2015; Teunissen and Ebert 2014;
  Lapenta 2002) are the only route to fewer particles and are speed items.
* **Debye / omega_pe resolution.** lambda_D = 16.6 um (1e18, 5 eV), 6.2 um (1e19, 7 eV): the 25 um
  ladder point resolves our plateau at 1.5 cells/lambda_D but Brandt's density at 4.0 (past pi); no
  grid resolves 1e19 without Bohm transport lowering it - another reason (c) comes first.
* **Smoothing.** Rejected inside the same ladder (A §7.4: a 1-2-1 filter smears a 2-3-cell sheath by a
  cell); keep the code filter-free.
* **Axisymmetric weighting.** Bilinear CIC with the Verboncoeur 2001 shape-volume correction (Ruyten 1993
  is the density-conserving alternative) is correct as implemented; no change.

## 5. Ranked roadmap - physics first, then speed

Order chosen by (consequence for claimed quantities) x (probability the literature says it matters) /
(effort), with prerequisites first (R0 landed as v2.0.6 during the audit and is kept in the table for the
record). Runs are on the Lambda H100 under MPS-4 (4 slots, ~0.5x per-process
speed; a channel-33 run is ~5 h solo, ~10 h in a slot); every run is a declared comparison against the
accepted 33 um plateau `0d228ad2` (physics off) with the same seed, grid, W and gates, so "expected
change" can be checked against the result. Numbers are directions and rough magnitudes from §4 / §9;
the purpose of stating them is that a result of the opposite sign is a finding, not a tuning target.

| rank | item | effort | H100 runs | expected change of the reference plateau (direction; rough size) | what admits it |
|---|---|---|---|---|---|
| R0 | ledger W fix + gate recalibration + post-hoc residual sidecars (v2.0.6) - **DONE** (`4b53012d`, `8c70cff0`, 2026-09-05 01:40) | S | 0 (post hoc) | none physical; recorded residuals shifted + by the inelastic power (7-14 % of electrode work); the corrected ss-v4 reads +2.46 % (acceptance (b) FAIL), the 50 um base +13.0 % | particle-side identity closes to round-off on three backends (v2.0.6 tests); physics bitwise |
| R1 | anomalous transport: `bohm_rotation` variant + predeclared alpha-series {0, 1/64, 1/16, 0.345} on the ss-v4 template; then the sealed ext-val `bohm-0.4` channel-20um run | S-M | 3 channel-33 (15 GPU-h) + 1 channel-20um (12 h solo / 30 h slot) | I_d UP monotone in alpha (+20 to +60 % at 1/16); peak n_e DOWN (-15 to -40 % at ours, -30 to -60 % at Brandt's); S, utilisation DOWN; T_e,peak DOWN; cusp sheath drops DOWN; per-cusp wall electron loss UP; ext-val: plateau instead of avalanche, I_a toward 4.3 mA, n_i toward 1e19 | monotone series with the stated signs; ext-val rows inside V&V20 tolerance or a recorded miss; residual-power gate silent |
| R2 | SEE from the dielectric: digitised BN and Al2O3 Vaughan yields, backscatter fraction, 2 eV secondaries, Hobbs-Wesson cap; `pic2d-model-v2.2` | M-L | 2 channel-33 (BN, Al2O3) at the R1-chosen alpha | cusp sheath drops DOWN 10-45 %; wall electron power UP x1.5-2; T_e,peak DOWN 10-25 %; I_d UP 10-30 %; peak n_e DOWN 5-15 %; wall ion energy DOWN; Al2O3 > BN in every effect | slab sheath drop vs analytic with delta; ledger closes with emission; on/off/material triad with stated signs |
| R3a | four excitation levels | S | folded into R3b's run | T_e DOWN 3-5 %; S DOWN 3-5 %; inelastic power UP ~15 % | total excitation frequency unchanged; ledger |
| R3b | Xe+ + Xe CEX + MEX with fast-neutral thrust tally; `xenon-ion-neutral-cross-sections-v1.json` | M | 1 channel-33 + 1 plume-v2.1 | I_d, S ~unchanged (< 5 %); IEDF gains a low-energy population 15-30 % of exit ions; anode ion current UP; divergence UP; ion thrust DOWN by the exchanged fraction, total (ion + fast neutral) ~unchanged | CEX fraction = 1 - exp(-L/lambda) on a slab; momentum ledger closes with the neutral term |
| R4 | Coulomb collisions (Takizuka-Abe / Nanbu, cell sort, subcycled) | L | 1 channel-33; 1 ext-val (`bohm-0.4` + Coulomb + SEE = the like-for-like Brandt comparison) | S UP 5-20 % (more at 1e19); T_e,peak DOWN ~5 %; EEDF Maxwellian tail; I_d UP few % | bi-Maxwellian relaxation at the Trubnikov rate; pairwise conservation; on/off pair |
| R5a | spatial free-molecular neutrals (Katz-Mikellides view factors) with wall accommodation, Picard-iterated per window | L | 1 channel-33 (+ 1 plume) | ionisation moves UPSTREAM (anode-side cusp flames brighten, exit cusp dims); exit n_g DOWN; S, utilisation +-10 %; the 0-D fixed point replaced by a converged map | Clausing conductance vs analytic; uniform limit = 0-D to round-off; atom ledger |
| R5b | metastable pool + stepwise ionisation (0-D first) | L | 1 channel-33 | S UP 10-25 %; utilisation UP; n_g DOWN; T_e slightly DOWN | metastable ledger; CR fixed point vs analytic; on/off pair |
| R6 | diagnostics and small species: Xe2+ (M), neutraliser gas source (S), beta / B_induced map (S), sputter-yield post-processor (S-M), cathode-closure pair fixed vs continuity (0 code) | S-M | 1 plume-v2.1 (Xe2+ + neutraliser gas); 1 plume pair for the cathode closure | Xe2+: I_d +1-3 %, thrust -1.5 % per 5 % current fraction; neutraliser gas: plume CEX and cold-gas thrust UP; beta map: < 1e-2 everywhere except within 0.1 mm of the nulls; cathode pair: the I_d difference is the declared closure sensitivity | recorded maps; pair difference reported, not tuned |
| then speed | GMG Poisson (built, `poisson_gmg_v1`), launch fusion + cell sort (shared with R4), mixed precision, then an EC-gather trial against the explicit 33 / 25 um ladder | per A §8 | qualification replays | bitwise or +-5 % band per the acceleration review's protocol | A §8-9 |

Total: ~15 runs, 75-90 GPU-h (about 4-5 days on one H100 with MPS-4), roughly 20-30 developer days for
R0-R5. After R1-R2 the ext-val comparison can be re-run as `channel-20um-bohm-0.4-see` - that is the
single run most likely to move the project from "inconclusive" to a recorded code-to-code result.

## 6. What the 2D axisymmetric model can never claim

1. **A self-consistent anomalous cross-field mobility.** The E x B electron cyclotron drift instability,
   its ion-transit-time coupling (Charoy et al. 2021), the modified two-stream / lower-hybrid modes and
   rotating spokes all need an azimuthal coordinate. In (r,z) they are excluded by construction, so every
   I_d, S, utilisation, per-cusp loss and cusp sheath drop we report is **conditional on a declared
   closure** (alpha). 2D reduced models that do carry the instability "significantly overestimate" the
   mobility relative to 3D (Kaganovich et al. 2020), so even a companion 2D-theta run is a bracket.
2. **Azimuthal structure of any kind**: spokes, azimuthal non-uniformity of the cusp sheaths, the
   divergence asymmetry that an off-axis neutraliser produces.
3. **Electron heating by the instability** (the T_e the code produces is collisional + cusp-mirror
   heating only), and therefore the SEE regime at the cusps is itself closure-dependent.

What would close it, with cost for our channel (3 x 24 mm bore, three cusps):

* **(i) Calibrated Bohm / alpha closure with a predeclared alpha-series** (R1): preserves everything
  axisymmetric, costs nothing, and states honestly that alpha is a fit parameter. It is the SOTA HEMPT
  practice (Brandt 2016) and what our claims should be conditioned on now.
* **(ii) 2D companion runs in the instability planes.** At a cusp plane B is radial and E axial, so the
  drift is azimuthal: a z-theta slab at each cusp is the Charoy et al. 2019 (axial-azimuthal) geometry
  with our fields and n_e; between the cusps B is axial and E radial: an r-theta slab is the Villafana et
  al. 2021 (radial-azimuthal) geometry. Each companion is one LANDMARK-class run (comparable to one
  channel-33 run) and yields an effective mobility per (r,z) cell to feed the alpha closure as a map
  instead of a constant; reduced-order quasi-2D PIC (Reza et al. 2023; Faraji et al. 2023) can do it at
  2-15 % of that cost. Preserves the local physics of the instability; loses the coupling to the cusp
  sheath (r-theta) or to the mirror (z-theta) unless both are run. Effort L (2-3 weeks including the
  benchmark reproduction).
* **(iii) Full 3D.** At 33 um the circumference at r = 1.5 mm needs ~280 azimuthal cells: 90 x 720 x
  280 ~ 18 M cells, ~30 particles per cell -> ~550 M electrons, 250x today's 2.2 M. At the measured
  ~1e9 particle-steps per second per H100 a 5 us run (3.6 M steps) is ~2e15 particle-steps = ~23 days
  on one H100; a 30 deg periodic wedge is 8x cheaper (~3 days) and a self-similar scaled system
  (Matthias et al. 2020 limits) cheaper still but with the scaling caveats recorded in the ext-val
  README. The 3D HET codes that exist (Villafana et al. 2023; Chen et al. 2025; Zhong et al. 2026) are the
  precedent; no 3D PIC of a cusped-field thruster with a dielectric channel was found (silent). One or
  two heroic wedge runs are feasible on the H100 and would be the only way to close the claim.
* **(iv) Data-driven closure from (ii)/(iii) snapshots** (Jorns 2018; Marks and Jorns 2023 on the
  pitfalls): preserves whatever the training runs contained and nothing else; only meaningful after (ii)
  or (iii) exist.

The honest statement for the paper today: "The 2D axisymmetric model carries no azimuthal instability;
its cross-field transport is classical plus a declared Bohm-type closure alpha. All discharge quantities
are reported for the declared alpha-series; an r-theta / z-theta companion campaign (or a 3D wedge run)
is required before any of them can be called a prediction."

## 7. Where this audit is silent or uncertain

* Kahnfeld et al. 2019 (the canonical HEMPT review) is closed access; nothing above rests on its content
  beyond what the sibling review recorded.
* The Greifswald code's exact SEE, Coulomb and neutral models are known only from abstracts and from
  Brandt et al. 2016's description of their own run; the "SOTA HEMPT" column is therefore Brandt 2016
  plus recorded abstracts, not a code audit.
* All "estimate" numbers (§9) are order-of-magnitude with stated formulas; the metastable branching
  (~30 % of the lumped excitation into the 6s manifold) and the quenching rate are the least certain
  (x3).
* No resolved paper gives BN or Al2O3 yields for the exact liner grade of a micro-HEMPT, nor an
  accommodation coefficient for xenon on those ceramics at 400-500 K.
* No 3D or 2D-theta PIC of a cusped-field thruster isolating the anomalous contribution was found.

## 8. Bibliography (151 entries, alphabetical; DOI for every entry; all resolved 2026-09-05 through the Crossref works record)

Entries also present in the sibling reviews carry `[also B-nn/A-nn]` (60 entries); the other 91 are new
to the repository. The resolved Crossref title is used verbatim (hence the occasional upper-case or
mark-up remnant); IEPC papers without DOIs (Kornfeld et al. 2007; Koch et al. 2011) and Hayashi's NIFS
compilation are named in the text only and not counted.

1. Adam, J. C.; Héron, A.; Laval, G. "Study of stationary plasma thrusters using two-dimensional fully kinetic simulations." *Physics of Plasmas* 11, 295-305 (2004). doi:10.1063/1.1632904 [also B-1/A-1]
2. Ahedo, E.; De Pablo, V. "Combined effects of electron partial thermalization and secondary emission in Hall thruster discharges." *Physics of Plasmas* 14, 083501 (2007). doi:10.1063/1.2749237
3. Allan, M.; Zatsarinny, O.; Bartschat, K. "Near-threshold absolute angle-differential cross sections for electron-impact excitation of argon and xenon." *Physical Review A* 74, 030701 (2006). doi:10.1103/PhysRevA.74.030701
4. Assous, F.; Pougeard Dulimbert, T.; Segré, J. "A new method for coalescing particles in PIC codes." *Journal of Computational Physics* 187, 550-571 (2003). doi:10.1016/S0021-9991(03)00124-4 [also A-7]
5. Baragiola, R. A.; Alonso, E. V.; Florio, A. O. "Electron emission from clean metal surfaces induced by low-energy light ions." *Physical Review B* 19, 121-129 (1979). doi:10.1103/PhysRevB.19.121
6. Barral, S.; Makowski, K.; Peradzyński, Z.; Gascon, N.; Dudeck, M. "Wall material effects in stationary plasma thrusters. II. Near-wall and in-wall conductivity." *Physics of Plasmas* 10, 4137-4152 (2003). doi:10.1063/1.1611881
7. Biagi, S. "Monte Carlo simulation of electron drift and diffusion in counting gases under the influence of electric and magnetic fields." *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment* 421, 234-240 (1999). doi:10.1016/S0168-9002(98)01233-9
8. Bird, G. A. *Molecular Gas Dynamics And The Direct Simulation Of Gas Flows.* Oxford University Press (1994). doi:10.1093/oso/9780198561958.001.0001
9. Birdsall, C.; Langdon, A. *Plasma Physics via Computer Simulation.* CRC Press (2018 edition of the 1991 book). doi:10.1201/9781315275048 [also B-5/A-13]
10. Bobylev, A. V.; Nanbu, K. "Theory of collision algorithms for gases and plasmas based on the Boltzmann equation and the Landau-Fokker-Planck equation." *Physical Review E* 61, 4576-4586 (2000). doi:10.1103/PhysRevE.61.4576
11. Boeuf, J. P.; Garrigues, L. "Low frequency oscillations in a stationary plasma thruster." *Journal of Applied Physics* 84, 3541-3554 (1998). doi:10.1063/1.368529 [also B-7]
12. Boeuf, J. P. "Tutorial: Physics and modeling of Hall thrusters." *Journal of Applied Physics* 121, 011101 (2017). doi:10.1063/1.4972269 [also B-8/A-15]
13. Boeuf, J. P.; Garrigues, L. "E × B electron drift instability in Hall thrusters: Particle-in-cell simulations vs. theory." *Physics of Plasmas* 25, 061204 (2018). doi:10.1063/1.5017033 [also B-9/A-16]
14. Bordage, M. C.; Biagi, S. F.; Alves, L. L.; Bartschat, K.; Chowdhury, S.; Pitchford, L. C.; et al. "Comparisons of sets of electron–neutral scattering cross sections and swarm parameters in noble gases: III. Krypton and xenon." *Journal of Physics D: Applied Physics* 46, 334003 (2013). doi:10.1088/0022-3727/46/33/334003
15. Boyd, I. D. "Review of Hall Thruster Plume Modeling." *Journal of Spacecraft and Rockets* 38, 381-387 (2001). doi:10.2514/2.3695
16. Boyd, I. D.; Dressler, R. A. "Far field modeling of the plasma plume of a Hall thruster." *Journal of Applied Physics* 92, 1764-1774 (2002). doi:10.1063/1.1492014
17. Brackbill, J. "On energy and momentum conservation in particle-in-cell plasma simulation." *Journal of Computational Physics* 317, 405-427 (2016). doi:10.1016/j.jcp.2016.04.050 [also B-11/A-20]
18. Brandt, T.; Trottenberg, T.; Groll, R.; Jansen, F.; Hey, F. G.; Johann, U.; et al. "Simulations on the influence of the spatial distribution of source electrons on the plasma in a cusped-field thruster." *The European Physical Journal D* 69, 145 (2015). doi:10.1140/epjd/e2015-50571-4
19. Brandt, T.; Schneider, R.; Duras, J.; Kahnfeld, D.; Hey, F. G.; Kersten, H.; et al. "Particle-in-Cell Simulation of a Down-Scaled HEMP Thruster." *TRANSACTIONS OF THE JAPAN SOCIETY FOR AERONAUTICAL AND SPACE SCIENCES, AEROSPACE TECHNOLOGY JAPAN* 14, Pb_235-Pb_242 (2016). doi:10.2322/tastj.14.Pb_235 [also B-12/A-22]
20. Campanell, M. D.; Khrabrov, A. V.; Kaganovich, I. D. "Absence of Debye Sheaths due to Secondary Electron Emission." *Physical Review Letters* 108, 255001 (2012). doi:10.1103/PhysRevLett.108.255001 [also B-13]
21. Campanell, M. D.; Khrabrov, A. V.; Kaganovich, I. D. "General Cause of Sheath Instability Identified for Low Collisionality Plasmas in Devices with Secondary Electron Emission." *Physical Review Letters* 108, 235001 (2012). doi:10.1103/PhysRevLett.108.235001
22. Charoy, T.; Boeuf, J. P.; Bourdon, A.; Carlsson, J. A.; Chabert, P.; Cuenot, B.; et al. "2D axial-azimuthal particle-in-cell benchmark for low-temperature partially magnetized plasmas." *Plasma Sources Science and Technology* 28, 105010 (2019). doi:10.1088/1361-6595/ab46c5 [also B-19/A-30]
23. Charoy, T.; Lafleur, T.; Laguna, A. A.; Bourdon, A.; Chabert, P. "The interaction between ion transit-time and electron drift instabilities and their effect on anomalous electron transport in Hall thrusters." *Plasma Sources Science and Technology* 30, 065017 (2021). doi:10.1088/1361-6595/ac02b3
24. Chen, G.; Chacón, L.; Barnes, D. "An energy- and charge-conserving, implicit, electrostatic particle-in-cell algorithm." *Journal of Computational Physics* 230, 7018-7036 (2011). doi:10.1016/j.jcp.2011.05.031 [also B-20/A-31]
25. Chen, X.; Xie, L.; Zhong, K.; Luo, X.; Zhou, Z.; Wang, B.; et al. "Influence of plume region arrangement on Hall thruster azimuthal instability: 3D PIC simulations via a newly developed code PMSL-PIC-HET-3D." *Physics of Plasmas* 32, 052103 (2025). doi:10.1063/5.0253669
26. Chiu, Y. h.; Austin, B. L.; Williams, S.; Dressler, R. A.; Karabadzhak, G. F. "Passive optical diagnostic of Xe-propelled Hall thrusters. I. Emission cross sections." *Journal of Applied Physics* 99, 113304 (2006). doi:10.1063/1.2195018
27. Cichocki, F.; Domínguez-Vázquez, A.; Merino, M.; Ahedo, E. "Hybrid 3D model for the interaction of plasma thruster plumes with nearby objects." *Plasma Sources Science and Technology* 26, 125008 (2017). doi:10.1088/1361-6595/aa986e
28. Cormier-Michel, E.; Shadwick, B. A.; Geddes, C. G. R.; Esarey, E.; Schroeder, C. B.; Leemans, W. P. "Unphysical kinetic effects in particle-in-cell modeling of laser wakefield accelerators." *Physical Review E* 78, 016404 (2008). doi:10.1103/PhysRevE.78.016404
29. Croes, V.; Lafleur, T.; Bonaventura, Z.; Bourdon, A.; Chabert, P. "2D particle-in-cell simulations of the electron drift instability and associated anomalous electron transport in Hall-effect thrusters." *Plasma Sources Science and Technology* 26, 034001 (2017). doi:10.1088/1361-6595/aa550f [also B-25]
30. Domínguez-Vázquez, A.; Cichocki, F.; Merino, M.; Fajardo, P.; Ahedo, E. "Axisymmetric plasma plume characterization with 2D and 3D particle codes." *Plasma Sources Science and Technology* 27, 104009 (2018). doi:10.1088/1361-6595/aae702
31. Ducrocq, A.; Adam, J. C.; Héron, A.; Laval, G. "High-frequency electron drift instability in the cross-field configuration of Hall thrusters." *Physics of Plasmas* 13, 102111 (2006). doi:10.1063/1.2359718
32. Dunaevsky, A.; Raitses, Y.; Fisch, N. J. "Secondary electron emission from dielectric materials of a Hall thruster with segmented electrodes." *Physics of Plasmas* 10, 2574-2577 (2003). doi:10.1063/1.1568344 [also B-27]
33. Duras, J.; Matyash, K.; Tskhakaya, D.; Kalentev, O.; Schneider, R. "Self‐Force in 1D Electrostatic Particle‐in‐Cell Codes for NonEquidistant Grids." *Contributions to Plasma Physics* 54, 697-711 (2014). doi:10.1002/ctpp.201300060 [also B-28/A-48]
34. Duras, J.; Kahnfeld, D.; Bandelow, G.; Kemnitz, S.; Lüskow, K.; Matthias, P.; et al. "Ion angular distribution simulation of the Highly Efficient Multistage Plasma Thruster." *Journal of Plasma Physics* 83, 595830107 (2017). doi:10.1017/S0022377817000125 [also B-29/A-49]
35. Ellison, C. L.; Raitses, Y.; Fisch, N. J. "Cross-field electron transport induced by a rotating spoke in a cylindrical Hall thruster." *Physics of Plasmas* 19, 013503 (2012). doi:10.1063/1.3671920
36. Erwin, D. A.; Kunc, J. A. "Ionization of excited xenon atoms by electrons." *Physical Review A* 70, 022705 (2004). doi:10.1103/PhysRevA.70.022705
37. Fabris, A. L.; Young, C. V.; Manente, M.; Pavarin, D.; Cappelli, M. A. "Ion Velocimetry Measurements and Particle-In-Cell Simulation of a Cylindrical Cusped Plasma Accelerator." *IEEE Transactions on Plasma Science* 43, 54-63 (2015). doi:10.1109/TPS.2014.2321743 [also B-66]
38. Faraji, F.; Reza, M.; Knoll, A. "Verification of the generalized reduced-order particle-in-cell scheme in a radial–azimuthal E × B plasma configuration." *AIP Advances* 13, 025315 (2023). doi:10.1063/5.0136889
39. Fons, J. T.; Lin, C. C. "Measurement of the cross sections for electron-impact excitation into the5p56plevels of xenon." *Physical Review A* 58, 4603-4615 (1998). doi:10.1103/PhysRevA.58.4603
40. Furman, M.; Pivi, M. "Probabilistic model for the simulation of secondary electron emission." *Physical Review Special Topics - Accelerators and Beams* 5, 124404 (2002). doi:10.1103/PhysRevSTAB.5.124404 [also B-34]
41. Garnier, Y.; Viel, V.; Roussel, J. F.; Bernard, J. "Low-energy xenon ion sputtering of ceramics investigated for stationary plasma thrusters." *Journal of Vacuum Science & Technology A: Vacuum, Surfaces, and Films* 17, 3246-3254 (1999). doi:10.1116/1.582050
42. Gildea, S. R.; Matlock, T. S.; Martínez-Sánchez, M.; Hargus, W. A. "Erosion Measurements in a Low-Power Cusped-Field Plasma Thruster." *Journal of Propulsion and Power* 29, 906-918 (2013). doi:10.2514/1.B34607 [also B-40]
43. Goebel, D. M.; Jameson, K. K.; Watkins, R. M.; Katz, I.; Mikellides, I. G. "Hollow cathode theory and experiment. I. Plasma characterization using fast miniature scanning probes." *Journal of Applied Physics* 98, 113302 (2005). doi:10.1063/1.2135417
44. Goebel, D. M.; Jameson, K. K.; Katz, I.; Mikellides, I. G. "Potential fluctuations and energetic ion production in hollow cathode discharges." *Physics of Plasmas* 14, 103508 (2007). doi:10.1063/1.2784460
45. Goebel, D. M.; Katz, I. *Fundamentals of Electric Propulsion.* Wiley (2008). doi:10.1002/9780470436448
46. Hagelaar, G. J. M.; Pitchford, L. C. "Solving the Boltzmann equation to obtain electron transport coefficients and rate coefficients for fluid models." *Plasma Sources Science and Technology* 14, 722-733 (2005). doi:10.1088/0963-0252/14/4/011
47. Hagstrum, H. D. "Theory of Auger Ejection of Electrons from Metals by Ions." *Physical Review* 96, 336-365 (1954). doi:10.1103/PhysRev.96.336
48. Hara, K.; Treece, C. "Ion kinetics and nonlinear saturation of current-driven instabilities relevant to hollow cathode plasmas." *Plasma Sources Science and Technology* 28, 055013 (2019). doi:10.1088/1361-6595/ab18e4
49. Hara, K. "An overview of discharge plasma modeling for Hall effect thrusters." *Plasma Sources Science and Technology* 28, 044001 (2019). doi:10.1088/1361-6595/ab0f70 [also B-42/A-67]
50. Hara, K.; Tsikata, S. "Cross-field electron diffusion due to the coupling of drift-driven microinstabilities." *Physical Review E* 102, 023202 (2020). doi:10.1103/PhysRevE.102.023202
51. Higginson, D. P. "A full-angle Monte-Carlo scattering technique including cumulative and single-event Rutherford scattering in plasmas." *Journal of Computational Physics* 349, 589-603 (2017). doi:10.1016/j.jcp.2017.08.016
52. Hobbs, G. D.; Wesson, J. A. "Heat flow through a Langmuir sheath in the presence of electron emission." *Plasma Physics* 9, 85-87 (1967). doi:10.1088/0032-1028/9/1/410 [also B-44]
53. Hockney, R. "Measurements of collision and heating times in a two-dimensional thermal computer plasma." *Journal of Computational Physics* 8, 19-44 (1971). doi:10.1016/0021-9991(71)90032-5 [also B-45/A-70]
54. Hofer, R. R.; Gallimore, A. D. "High-Specific Impulse Hall Thrusters, Part 2: Efficiency Analysis." *Journal of Propulsion and Power* 22, 732-740 (2006). doi:10.2514/1.15954
55. Holstein, T. "Imprisonment of Resonance Radiation in Gases." *Physical Review* 72, 1212-1233 (1947). doi:10.1103/PhysRev.72.1212
56. Hyman, H. A. "Electron-impact ionization cross sections for excited states of the rare gases (Ne, Ar, Kr, Xe), cadmium, and mercury." *Physical Review A* 20, 855-859 (1979). doi:10.1103/PhysRevA.20.855
57. Héron, A.; Adam, J. C. "Anomalous conductivity in Hall thrusters: Effects of the non-linear coupling of the electron-cyclotron drift instability with secondary electron emission of the walls." *Physics of Plasmas* 20, 082313 (2013). doi:10.1063/1.4818796
58. Janes, G. S.; Lowder, R. S. "Anomalous Electron Diffusion and Ion Acceleration in a Low-Density Plasma." *The Physics of Fluids* 9, 1115-1123 (1966). doi:10.1063/1.1761810
59. Janhunen, S.; Smolyakov, A.; Chapurin, O.; Sydorenko, D.; Kaganovich, I.; Raitses, Y. "Nonlinear structures and anomalous transport in partially magnetized E×B plasmas." *Physics of Plasmas* 25, 011608 (2018). doi:10.1063/1.5001206
60. Janhunen, S.; Smolyakov, A.; Sydorenko, D.; Jimenez, M.; Kaganovich, I.; Raitses, Y. "Evolution of the electron cyclotron drift instability in two-dimensions." *Physics of Plasmas* 25, 082308 (2018). doi:10.1063/1.5033896
61. Janssen, J. F. J.; Pitchford, L. C.; Hagelaar, G. J. M.; van Dijk, J. "Evaluation of angular scattering models for electron-neutral collisions in Monte Carlo simulations." *Plasma Sources Science and Technology* 25, 055026 (2016). doi:10.1088/0963-0252/25/5/055026
62. Jorns, B. A.; Mikellides, I. G.; Goebel, D. M. "Ion acoustic turbulence in a 100-A LaB 6 hollow cathode." *Physical Review E* 90, 063106 (2014). doi:10.1103/PhysRevE.90.063106
63. Jorns, B. "Predictive, data-driven model for the anomalous electron collision frequency in a Hall effect thruster." *Plasma Sources Science and Technology* 27, 104007 (2018). doi:10.1088/1361-6595/aae472
64. Jung, R. O.; Boffard, J. B.; Anderson, L. W.; Lin, C. C. "Electron-impact excitation cross sections from the xenonJ=2metastable level." *Physical Review A* 72, 022723 (2005). doi:10.1103/PhysRevA.72.022723
65. Kaganovich, I. D.; Raitses, Y.; Sydorenko, D.; Smolyakov, A. "Kinetic effects in a Hall thruster discharge." *Physics of Plasmas* 14, 057104 (2007). doi:10.1063/1.2709865
66. Kaganovich, I. D.; Smolyakov, A.; Raitses, Y.; Ahedo, E.; Mikellides, I. G.; Jorns, B.; et al. "Physics of E × B discharges relevant to plasma propulsion and similar technologies." *Physics of Plasmas* 27, 120601 (2020). doi:10.1063/5.0010135 [also B-49]
67. Kahnfeld, D.; Heidemann, R.; Duras, J.; Matthias, P.; Bandelow, G.; Lüskow, K.; et al. "Breathing modes in HEMP thrusters." *Plasma Sources Science and Technology* 27, 124002 (2018). doi:10.1088/1361-6595/aaf29a [also B-51/A-76]
68. Kahnfeld, D.; Duras, J.; Matthias, P.; Kemnitz, S.; Arlinghaus, P.; Bandelow, G.; et al. "Numerical modeling of high efficiency multistage plasma thrusters for space applications." *Reviews of Modern Plasma Physics* 3, 11 (2019). doi:10.1007/s41614-019-0030-4 [also B-52/A-77]
69. Kalentev, O.; Matyash, K.; Duras, J.; Lüskow, K. F.; Schneider, R.; Koch, N.; et al. "Electrostatic Ion Thrusters ‐ Towards Predictive Modeling." *Contributions to Plasma Physics* 54, 235-248 (2014). doi:10.1002/ctpp.201300038 [also B-53]
70. Karabadzhak, G. F.; Chiu, Y. h.; Dressler, R. A. "Passive optical diagnostic of Xe propelled Hall thrusters. II. Collisional-radiative model." *Journal of Applied Physics* 99, 113305 (2006). doi:10.1063/1.2195019
71. Katz, I.; Mikellides, I. G. "Neutral gas free molecular flow algorithm including ionization and walls for use in plasma simulations." *Journal of Computational Physics* 230, 1454-1464 (2011). doi:10.1016/j.jcp.2010.11.013 [also B-54]
72. Lafleur, T.; Baalrud, S. D.; Chabert, P. "Theory for the anomalous electron transport in Hall effect thrusters. I. Insights from particle-in-cell simulations." *Physics of Plasmas* 23, 053502 (2016). doi:10.1063/1.4948495 [also B-58]
73. Lafleur, T.; Baalrud, S. D.; Chabert, P. "Theory for the anomalous electron transport in Hall effect thrusters. II. Kinetic model." *Physics of Plasmas* 23, 053503 (2016). doi:10.1063/1.4948496
74. Lafleur, T.; Baalrud, S. D.; Chabert, P. "Characteristics and transport effects of the electron drift instability in Hall-effect thrusters." *Plasma Sources Science and Technology* 26, 024008 (2017). doi:10.1088/1361-6595/aa56e2
75. Lafleur, T.; Chabert, P. "The role of instability-enhanced friction on ‘anomalous’ electron and ion transport in Hall-effect thrusters." *Plasma Sources Science and Technology* 27, 015003 (2018). doi:10.1088/1361-6595/aa9efe
76. Langdon, A. "Effects of the spatial grid in simulation plasmas." *Journal of Computational Physics* 6, 247-267 (1970). doi:10.1016/0021-9991(70)90024-0 [also B-61/A-80]
77. Lapenta, G. "Particle Rezoning for Multidimensional Kinetic Particle-In-Cell Simulations." *Journal of Computational Physics* 181, 317-337 (2002). doi:10.1006/jcph.2002.7126
78. Lapenta, G. "Exactly energy conserving semi-implicit particle in cell formulation." *Journal of Computational Physics* 334, 349-366 (2017). doi:10.1016/j.jcp.2017.01.002 [also B-62/A-84]
79. Lemons, D. S.; Winske, D.; Daughton, W.; Albright, B. "Small-angle Coulomb collision model for particle-in-cell simulations." *Journal of Computational Physics* 228, 1391-1403 (2009). doi:10.1016/j.jcp.2008.10.025
80. Lewis, H. "Energy-conserving numerical approximations for Vlasov plasmas." *Journal of Computational Physics* 6, 136-141 (1970). doi:10.1016/0021-9991(70)90012-4 [also A-86]
81. Lieberman, M. A.; Lichtenberg, A. J. *Principles of Plasma Discharges and Materials Processing.* Wiley (2005). doi:10.1002/0471724254
82. Liu, H.; Wu, H.; Zhao, Y.; Yu, D.; Ma, C.; Wang, D.; et al. "Study of the electric field formation in a multi-cusped magnetic field." *Physics of Plasmas* 21, 090706 (2014). doi:10.1063/1.4896250
83. Liu, H.; Wu, H.; Meng, Y.; Yang, S.; Zhang, J.; Yu, D. "Fluid Simulation of a Cusped Field Thruster." *Contributions to Plasma Physics* 55, 545-550 (2015). doi:10.1002/ctpp.201500011
84. MacDonald, N. A.; Young, C. V.; Cappelli, M. A.; Hargus, W. A. "Ion velocity and plasma potential measurements of a cylindrical cusped field thruster." *Journal of Applied Physics* 111, 093303 (2012). doi:10.1063/1.4707953 [also B-68]
85. Markidis, S.; Lapenta, G. "The energy conserving particle-in-cell method." *Journal of Computational Physics* 230, 7037-7052 (2011). doi:10.1016/j.jcp.2011.05.033 [also B-70/A-90]
86. Marks, T. A.; Jorns, B. A. "Challenges with the self-consistent implementation of closure models for anomalous electron transport in fluid simulations of Hall thrusters." *Plasma Sources Science and Technology* 32, 045016 (2023). doi:10.1088/1361-6595/accd18 [also B-71]
87. Matthias, P.; Kahnfeld, D.; Schneider, R.; Yeo, S. H.; Ogawa, H. "Particle‐in‐cell simulation of an optimized high‐efficiency multistage plasma thruster." *Contributions to Plasma Physics* 59, e201900028 (2019). doi:10.1002/ctpp.201900028 [also B-73]
88. Matthias, P.; Kahnfeld, D.; Kemnitz, S.; Duras, J.; Koch, N.; Schneider, R. "Similarity scaling‐application and limits for high‐efficiency‐multistage‐plasma‐thruster particle‐in‐cell modelling." *Contributions to Plasma Physics* 60, e201900199 (2020). doi:10.1002/ctpp.201900199 [also B-74/A-94]
89. Matyash, K.; Schneider, R.; Mutzke, A.; Kalentev, O.; Taccogna, F.; Koch, N.; et al. "Kinetic Simulations of SPT and HEMP Thrusters Including the Near-Field Plume Region." *IEEE Transactions on Plasma Science* 38, 2274-2280 (2010). doi:10.1109/TPS.2010.2056936 [also B-75]
90. Mikellides, I. G.; Katz, I.; Goebel, D. M.; Polk, J. E. "Hollow cathode theory and experiment. II. A two-dimensional theoretical model of the emitter region." *Journal of Applied Physics* 98, 113303 (2005). doi:10.1063/1.2135409
91. Mikellides, I. G.; Katz, I.; Goebel, D. M.; Jameson, K. K.; Polk, J. E. "Wear Mechanisms in Electron Sources for Ion Propulsion, II: Discharge Hollow Cathode." *Journal of Propulsion and Power* 24, 866-879 (2008). doi:10.2514/1.33462
92. Miller, J. S.; Pullins, S. H.; Levandier, D. J.; Chiu, Y. h.; Dressler, R. A. "Xenon charge exchange cross sections for electrostatic thruster models." *Journal of Applied Physics* 91, 984-991 (2002). doi:10.1063/1.1426246 [also B-77]
93. Nanbu, K. "Theory of cumulative small-angle collisions in plasmas." *Physical Review E* 55, 4642-4652 (1997). doi:10.1103/PhysRevE.55.4642
94. Nanbu, K.; Yonemura, S. "Weighted Particles in Coulomb Collision Simulations Based on the Theory of a Cumulative Scattering Angle." *Journal of Computational Physics* 145, 639-654 (1998). doi:10.1006/jcph.1998.6049
95. Okhrimovskyy, A.; Bogaerts, A.; Gijbels, R. "Electron anisotropic scattering in gases: A formula for Monte Carlo simulations." *Physical Review E* 65, 037402 (2002). doi:10.1103/PhysRevE.65.037402
96. Opal, C. B.; Peterson, W. K.; Beaty, E. C. "Measurements of Secondary-Electron Spectra Produced by Electron Impact Ionization of a Number of Simple Gases." *The Journal of Chemical Physics* 55, 4100-4106 (1971). doi:10.1063/1.1676707
97. Petronio, F.; Charoy, T.; Alvarez Laguna, A.; Bourdon, A.; Chabert, P. "Two-dimensional effects on electrostatic instabilities in Hall thrusters. I. Insights from particle-in-cell simulations and two-point power spectral density reconstruction techniques." *Physics of Plasmas* 30, 012103 (2023). doi:10.1063/5.0119253 [also A-106]
98. Petronio, F.; Charoy, T.; Alvarez Laguna, A.; Bourdon, A.; Chabert, P. "Two-dimensional effects on electrostatic instabilities in Hall thrusters. II. Comparison of particle-in-cell simulation results with linear theory dispersion relations." *Physics of Plasmas* 30, 012104 (2023). doi:10.1063/5.0119255
99. Pitchford, L. C.; Alves, L. L.; Bartschat, K.; Biagi, S. F.; Bordage, M.; Bray, I.; et al. "LXCat: an Open‐Access, Web‐Based Platform for Data Needed for Modeling Low Temperature Plasmas." *Plasma Processes and Polymers* 14, 1600098 (2017). doi:10.1002/ppap.201600098
100. Puech, V.; Mizzi, S. "Collision cross sections and transport parameters in neon and xenon." *Journal of Physics D: Applied Physics* 24, 1974-1985 (1991). doi:10.1088/0022-3727/24/11/011
101. Pullins, S.; Chiu, Y. H.; Levandier, D.; Dressler, R. "Ion dynamics in Hall effect and ion thrusters - Xe(+) + Xe symmetric charge transfer." *38th Aerospace Sciences Meeting and Exhibit* (2000). doi:10.2514/6.2000-603
102. Pérez, F.; Gremillet, L.; Decoster, A.; Drouin, M.; Lefebvre, E. "Improved modeling of relativistic collisions and collisional ionization in particle-in-cell codes." *Physics of Plasmas* 19, 083104 (2012). doi:10.1063/1.4742167
103. Qin, H.; Zhang, S.; Xiao, J.; Liu, J.; Sun, Y.; Tang, W. M. "Why is Boris algorithm so good?." *Physics of Plasmas* 20, 084503 (2013). doi:10.1063/1.4818428 [also A-111]
104. Quan, L.; Cao, Y.; Li, Y.; Liu, H.; Tian, B. "Influence of the axial oscillations on the electron cyclotron drift instability and electron transport in Hall thrusters." *Physics of Plasmas* 30, 043510 (2023). doi:10.1063/5.0134644
105. Raitses, Y.; Kaganovich, I. D.; Khrabrov, A.; Sydorenko, D.; Fisch, N. J.; Smolyakov, A. "Effect of Secondary Electron Emission on Electron Cross-Field Current in $E \times B$ Discharges." *IEEE Transactions on Plasma Science* 39, 995-1006 (2011). doi:10.1109/TPS.2011.2109403
106. Rapp, D.; Englander-Golden, P. "Total Cross Sections for Ionization and Attachment in Gases by Electron Impact. I. Positive Ionization." *The Journal of Chemical Physics* 43, 1464-1479 (1965). doi:10.1063/1.1696957
107. Rejoub, R.; Lindsay, B. G.; Stebbings, R. F. "Determination of the absolute partial and total cross sections for electron-impact ionization of the rare gases." *Physical Review A* 65, 042713 (2002). doi:10.1103/PhysRevA.65.042713
108. Reza, M.; Faraji, F.; Knoll, A. "Concept of the generalized reduced-order particle-in-cell scheme and verification in an axial-azimuthal Hall thruster configuration." *Journal of Physics D: Applied Physics* 56, 175201 (2023). doi:10.1088/1361-6463/acbb15 [also B-84/A-112]
109. Ruyten, W. M. "Density-Conserving Shape Factors for Particle Simulations in Cylindrical and Spherical Coordinates." *Journal of Computational Physics* 105, 224-232 (1993). doi:10.1006/jcph.1993.1070
110. Sary, G.; Garrigues, L.; Boeuf, J. P. "Hollow cathode modeling: I. A coupled plasma thermal two-dimensional model." *Plasma Sources Science and Technology* 26, 055007 (2017). doi:10.1088/1361-6595/aa6217
111. Schneider, R.; Matyash, K.; Kalentev, O.; Taccogna, F.; Koch, N.; Schirra, M. "Particle‐in‐Cell Simulations for Ion Thrusters." *Contributions to Plasma Physics* 49, 655-661 (2009). doi:10.1002/ctpp.200910070 [also B-88]
112. Sentoku, Y.; Kemp, A. "Numerical methods for particle simulations at extreme densities and temperatures: Weighted particles, relativistic collisions and reduced currents." *Journal of Computational Physics* 227, 6846-6861 (2008). doi:10.1016/j.jcp.2008.03.043
113. Smirnov, A.; Raitses, Y.; Fisch, N. J. "Electron cross-field transport in a low power cylindrical Hall thruster." *Physics of Plasmas* 11, 4922-4933 (2004). doi:10.1063/1.1791639 [also B-90]
114. Stephan, K.; Märk, T. D. "Absolute partial electron impact ionization cross sections of Xe from threshold up to 180 eV." *The Journal of Chemical Physics* 81, 3116-3117 (1984). doi:10.1063/1.448013
115. Surendra, M.; Graves, D. B.; Jellum, G. M. "Self-consistent model of a direct-current glow discharge: Treatment of fast electrons." *Physical Review A* 41, 1112-1125 (1990). doi:10.1103/PhysRevA.41.1112
116. Syage, J. A. "Electron-impact cross sections for multiple ionization of Kr and Xe." *Physical Review A* 46, 5666-5679 (1992). doi:10.1103/PhysRevA.46.5666
117. Sydorenko, D.; Smolyakov, A.; Kaganovich, I.; Raitses, Y. "Kinetic simulation of secondary electron emission effects in Hall thrusters." *Physics of Plasmas* 13, 014501 (2006). doi:10.1063/1.2158698 [also B-91]
118. Sydorenko, D.; Smolyakov, A.; Kaganovich, I.; Raitses, Y. "Modification of electron velocity distribution in bounded plasmas by secondary electron emission." *IEEE Transactions on Plasma Science* 34, 815-824 (2006). doi:10.1109/TPS.2006.875727
119. Szabo, J.; Warner, N.; Martinez-Sanchez, M.; Batishchev, O. "Full Particle-In-Cell Simulation Methodology for Axisymmetric Hall Effect Thrusters." *Journal of Propulsion and Power* 30, 197-208 (2014). doi:10.2514/1.B34774 [also B-93/A-124]
120. Taccogna, F.; Longo, S.; Capitelli, M. "Plasma sheaths in Hall discharge." *Physics of Plasmas* 12, 093506 (2005). doi:10.1063/1.2015257 [also B-95]
121. Taccogna, F.; Schneider, R.; Longo, S.; Capitelli, M. "Kinetic simulations of a plasma thruster." *Plasma Sources Science and Technology* 17, 024003 (2008). doi:10.1088/0963-0252/17/2/024003 [also B-96/A-127]
122. Taccogna, F.; Longo, S.; Capitelli, M.; Schneider, R. "Anomalous transport induced by sheath instability in Hall effect thrusters." *Applied Physics Letters* 94, 251502 (2009). doi:10.1063/1.3152270
123. Taccogna, F.; Minelli, P.; Asadi, Z.; Bogopolsky, G. "Numerical studies of the ExB electron drift instability in Hall thrusters." *Plasma Sources Science and Technology* 28, 064002 (2019). doi:10.1088/1361-6595/ab08af [also B-98]
124. Taccogna, F.; Garrigues, L. "Latest progress in Hall thrusters plasma modelling." *Reviews of Modern Plasma Physics* 3, 12 (2019). doi:10.1007/s41614-019-0033-1 [also B-99]
125. Takizuka, T.; Abe, H. "A binary collision model for plasma simulation with a particle code." *Journal of Computational Physics* 25, 205-219 (1977). doi:10.1016/0021-9991(77)90099-7
126. Tavant, A.; Croes, V.; Lucken, R.; Lafleur, T.; Bourdon, A.; Chabert, P. "The effects of secondary electron emission on plasma sheath characteristics and electron transport in an E × B discharge via kinetic simulations." *Plasma Sources Science and Technology* 27, 124001 (2018). doi:10.1088/1361-6595/aaeccd [also B-100]
127. Teunissen, J.; Ebert, U. "Controlling the weights of simulation particles: adaptive particle management using k-d trees." *Journal of Computational Physics* 259, 318-330 (2014). doi:10.1016/j.jcp.2013.12.005 [also A-131]
128. Ton-That, D.; Flannery, M. R. "Cross sections for ionization of metastable rare-gas atoms (Ne*, Ar*, Kr*, Xe*) and of metastableN2*, CO* molecules by electron impact." *Physical Review A* 15, 517-526 (1977). doi:10.1103/PhysRevA.15.517
129. Tondu, T.; Belhaj, M.; Inguimbert, V. "Electron-emission yield under electron impact of ceramics used as channel materials in Hall-effect thrusters." *Journal of Applied Physics* 110, 093301 (2011). doi:10.1063/1.3653820 [also B-101]
130. Tsikata, S.; Lemoine, N.; Pisarev, V.; Grésillon, D. M. "Dispersion relations of electron density fluctuations in a Hall thruster plasma, observed by collective light scattering." *Physics of Plasmas* 16, 033506 (2009). doi:10.1063/1.3093261
131. Tskhakaya, D.; Matyash, K.; Schneider, R.; Taccogna, F. "The Particle‐In‐Cell Method." *Contributions to Plasma Physics* 47, 563-594 (2007). doi:10.1002/ctpp.200710072 [also B-102]
132. Turner, M. M.; Derzsi, A.; Donkó, Z.; Eremin, D.; Kelly, S. J.; Lafleur, T.; et al. "Simulation benchmarks for low-pressure plasmas: Capacitive discharges." *Physics of Plasmas* 20, 013507 (2013). doi:10.1063/1.4775084 [also B-104/A-133]
133. Ueda, H.; Omura, Y.; Matsumoto, H.; Okuzawa, T. "A study of the numerical heating in electrostatic particle simulations." *Computer Physics Communications* 79, 249-259 (1994). doi:10.1016/0010-4655(94)90071-X [also B-106/A-135]
134. Vahedi, V.; Surendra, M. "A Monte Carlo collision model for the particle-in-cell method: applications to argon and oxygen discharges." *Computer Physics Communications* 87, 179-198 (1995). doi:10.1016/0010-4655(94)00171-W [also B-107/A-136]
135. Vahedi, V.; DiPeso, G. "Simultaneous Potential and Circuit Solution for Two-Dimensional Bounded Plasma Simulation Codes." *Journal of Computational Physics* 131, 149-163 (1997). doi:10.1006/jcph.1996.5591
136. Vaughan, J. "A new formula for secondary emission yield." *IEEE Transactions on Electron Devices* 36, 1963-1967 (1989). doi:10.1109/16.34278 [also B-108]
137. Vaughan, R. "Secondary emission formulas." *IEEE Transactions on Electron Devices* 40, 830 (1993). doi:10.1109/16.202798
138. Verboncoeur, J.; Alves, M.; Vahedi, V.; Birdsall, C. "Simultaneous Potential and Circuit Solution for 1D Bounded Plasma Particle Simulation Codes." *Journal of Computational Physics* 104, 321-328 (1993). doi:10.1006/jcph.1993.1034
139. Verboncoeur, J. "Symmetric Spline Weighting for Charge and Current Density in Particle Simulation." *Journal of Computational Physics* 174, 421-427 (2001). doi:10.1006/jcph.2001.6923
140. Verboncoeur, J. P. "Particle simulation of plasmas: review and advances." *Plasma Physics and Controlled Fusion* 47, A231-A260 (2005). doi:10.1088/0741-3335/47/5A/017
141. Villafana, W.; Petronio, F.; Denig, A. C.; Jimenez, M. J.; Eremin, D.; Garrigues, L.; et al. "2D radial-azimuthal particle-in-cell benchmark for E × B discharges." *Plasma Sources Science and Technology* 30, 075002 (2021). doi:10.1088/1361-6595/ac0a4a [also B-110/A-140]
142. Villafana, W.; Cuenot, B.; Vermorel, O. "3D particle-in-cell study of the electron drift instability in a Hall Thruster using unstructured grids." *Physics of Plasmas* 30, 033503 (2023). doi:10.1063/5.0133963 [also B-111]
143. Villemant, M.; Sarrailh, P.; Belhaj, M.; Garrigues, L.; Boniface, C. "Experimental investigation about energy balance of electron emission from materials under electron impacts at low energy: application to silver, graphite and SiO 2." *Journal of Physics D: Applied Physics* 50, 485204 (2017). doi:10.1088/1361-6463/aa91af
144. Villemant, M.; Belhaj, M.; Sarrailh, P.; Dadouch, S.; Garrigues, L.; Boniface, C. "Measurements of electron emission under electron impact on BN sample for incident electron energy between 10 eV and 1000 eV." *EPL (Europhysics Letters)* 127, 23001 (2019). doi:10.1209/0295-5075/127/23001
145. Vranic, M.; Grismayer, T.; Martins, J.; Fonseca, R.; Silva, L. "Particle merging algorithm for PIC codes." *Computer Physics Communications* 191, 65-73 (2015). doi:10.1016/j.cpc.2015.01.020 [also A-142]
146. Wang, J.; Brinza, D.; Young, M. "Three-Dimensional Particle Simulations of Ion Propulsion Plasma Environment for Deep Space 1." *Journal of Spacecraft and Rockets* 38, 433-440 (2001). doi:10.2514/2.3702
147. Xie, L.; Luo, X.; Zhou, Z.; Zhao, Y. "Effect of plasma initialization on 3D PIC simulation of Hall thruster azimuthal instability." *Physica Scripta* 99, 095602 (2024). doi:10.1088/1402-4896/ad69e5
148. Yamamura, Y.; Tawara, H. "Energy dependence of ion-induced sputtering yields from monatomic solids at normal incidence." *Atomic Data and Nuclear Data Tables* 62, 149-253 (1996). doi:10.1006/adnd.1996.0005
149. Zatsarinny, O.; Bartschat, K. "Benchmark calculations for near-threshold electron-impact excitation of krypton and xenon atoms." *Journal of Physics B: Atomic, Molecular and Optical Physics* 43, 074031 (2010). doi:10.1088/0953-4075/43/7/074031
150. Zhao, Y. J.; Liu, H.; Yu, D. R.; Hu, P.; Wu, H. "Particle-in-cell simulations for the effect of magnetic field strength on a cusped field thruster." *Journal of Physics D: Applied Physics* 47, 045201 (2014). doi:10.1088/0022-3727/47/4/045201 [also B-115]
151. Zhong, K.; Zeng, D.; Zhao, Y.; Yu, D. "Effects of RZ magnetic field components on electron drift instability in hall thrusters via 3D PIC simulations." *Physics Letters A* 590, 131809 (2026). doi:10.1016/j.physleta.2026.131809

### 8.1 Verification notes

* Resolution method: `api.crossref.org/works/<DOI>` on 2026-09-05; title, first author, year, container
  read back for every entry and compared with the intended paper. Mis-resolutions caught and replaced:
  Kalentev 2014 (guessed `ctpp.201310047` resolved to an unrelated paper; correct `ctpp.201300038`),
  Schneider 2009 (`ctpp.200910068` -> `ctpp.200910070`), Stephan and Märk 1984 (`1.448014` -> `1.448013`),
  Reza 2023 (`acbb16` -> `acbb15`), Brandt 2015 (`e2015-50651-0` -> `e2015-50571-4`), Duras 2014
  (`ctpp.201400024` -> `ctpp.201300060`), Villemant (the 2019 J. Phys. D guess does not exist: the energy
  balance paper is 2017 `aa91af`, the BN measurement is EPL 2019 `0295-5075/127/23001`). Dropped for want
  of a resolvable record: Hayashi 2003 (NIFS-DATA-79), Dawson 1966 ceramics SEE (guessed DOI wrong),
  Kornfeld 2007 and Koch 2011 (IEPC, no DOI).
* Crossref deposits: Goebel et al. 2005 is part I of the hollow-cathode pair (`1.2135417`), Mikellides
  et al. 2005 part II (`1.2135409`); Birdsall and Langdon resolves to the 2018 CRC edition of the 1991
  book; Bird 1994 resolves to the Oxford Scholarship Online record; Villafana 2021 and Kaganovich 2020
  titles contain HTML mark-up in the deposit (cleaned here); Yamamura and Tawara 1996 is deposited in
  upper case; Zhong et al. 2026 is an in-press 2026 record.
* Full text read: Brandt et al. 2016 (J-Stage PDF). Abstract-level: all others.

## 9. Appendix - order-of-magnitude estimates used above

All from `numpy` on the bound cross-section file and standard formulas (script kept outside the
repository); constants e, m_e, m_Xe = 2.1801714e-25 kg, epsilon_0, mu_0.

* **Rate coefficients from the bound Biagi table** (Maxwellian average): T_e 5 / 7 / 10 eV: k_el 2.67 /
  2.67 / 2.50e-13, k_ex 1.19 / 2.27 / 3.75e-14, k_iz 0.77 / 1.91 / 3.95e-14 m^3 s^-1.
* **nu_ee** (NRL: 2.91e-6 n[cm^-3] ln Lambda T_e^-3/2 s^-1) vs **nu_en = n_g k_el** at n_g 3e19:
  n_e 1e17: 0.02-0.04; 1e18: 0.15 (10 eV), 0.24 (7 eV), 0.38 (5 eV); 1e19: 1.4 / 2.2 / 3.4.
  tau_ee(1e19, 7 eV) = 0.06 us.
* **Bohm frequency** nu_an = alpha omega_ce: omega_ce = 8.8e9 (0.05 T), 1.8e10 (0.1 T), 5.3e10 s^-1
  (0.3 T); alpha 1/16 -> 5.5e8 ... 3.3e9 s^-1; alpha 1/64 -> 1.4e8 ... 8.2e8 s^-1.
* **beta** = n k T_e / (B^2/2 mu_0): 4.5e-4 (1e19, 10 eV, 0.3 T); 4.0e-3 (0.1 T); 1.1e-3 (1e18, 7 eV,
  0.05 T). Loop field mu_0 I / 2r: 6.3e-6 T (10 mA, 1 mm), 3.1e-5 T (50 mA, 1 mm), 1.0e-5 T (50 mA, 3 mm).
* **lambda_CEX** = 1/(n_g sigma), sigma = 5.4e-19 m^2 (Miller 2002 fit at 300 eV): 61 mm (3e19), 182 mm
  (1e19), 1.8 m (1e18). Exchange probability over L = 12-24 mm at 3e19: 18-33 %.
* **SEE** (Vaughan, E_0 12.5 eV): BN scaffold (delta_max 2.9, E_max 350): first crossover 32 eV, delta(20,
  50, 100 eV) = 0.59, 1.39, 2.06; Al2O3-like (6.4, 650): crossover 21 eV, delta = 0.92, 2.22, 3.41; a
  Dunaevsky-like BN (2.0, 400): crossover 59 eV. Flux-averaged over a Maxwellian wall flux, T_e 5 / 7 /
  10 / 20 eV: BN 0.14 / 0.28 / 0.49 / 1.01; Al2O3 0.22 / 0.44 / 0.77 / 1.61. Hobbs-Wesson critical yield
  1 - 8.3 sqrt(m_e/M) = 0.983 for xenon. Floating sheath drop T_e ln[(1-delta) sqrt(M/2 pi m_e)] =
  5.27 (0), 4.58 (0.5), 2.97 (0.9), 1.20 (0.983) T_e.
* **Stepwise ionisation**: k_iz(Xe*) with a 3.8 eV threshold and a 2e-19 m^2 plateau = 2.2e-13 m^3/s at
  7 eV = 11x k_iz(Xe); a 0-D balance n_m/n_g = n_e k_ex,m / (n_e (k_iz,m + k_q) + nu_wall) with k_ex,m =
  0.3 k_ex, k_q = 1e-13, nu_wall = v_th/L = 1.25e5 s^-1 gives n_m/n_g = 1.5e-2 (1e18), 2.1e-2 (1e19) and
  stepwise/ground = 0.18 / 0.23. Uncertain by x3.
* **Xe2+ from ground**: sigma_2/sigma_1 = 0.1 above 33.3 eV ramped over 60 eV: k_2/k_1 = 0.001 (5 eV),
  0.003 (7), 0.009 (10), 0.020 (15 eV).
* **Debye**: lambda_D = 16.6 um (1e18, 5 eV) -> 3.0 / 2.0 / 1.5 / 1.2 cells at 50 / 33 / 25 / 20 um;
  6.2 um (1e19, 7 eV) -> 8.0 / 5.4 / 4.0 / 3.2; 6.5 um (1.3e19, 10 eV) -> 7.7 / 5.1 / 3.8 / 3.1.
* **Dielectric charging** (4.b): sigma = epsilon_0 epsilon_r V/d = 8.85e-12 x 9.8 x 400 / 1e-3 =
  3.5e-5 C/m^2 per 100 V ... 3.5e-4 C/m^2 at 400 V behind 1 mm of Al2O3; at Gamma_i ~ 1e22 m^-2 s^-1
  (n 1e18, v_B 1e4 m/s) the charging time is sigma/(e Gamma_i) ~ 0.2 ms. Tangential coupling ratio
  (epsilon_r epsilon_0 Delta V / L) / (epsilon_0 Delta V / lambda_D) = epsilon_r lambda_D / L = 9.8 x 10 um /
  6 mm ~ 2 %.
