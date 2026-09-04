# Literature synthesis and revised roadmap

**Status: planning document (docs only). No code, spec, protocol, result or paper file is
changed by this document.** Prepared 2026-09-03 18:10 AEST against
`origin/feat/sota-foundation` at `b6bb6215` on branch `docs/literature-synthesis`.

It reconciles the three literature reviews written concurrently on 2026-09-03 and turns their
recommendations into one decision table and one ordered roadmap:

| review | file | commit | references |
| --- | --- | --- | --- |
| PIC-MCC blockers | `modern/docs/literature/pic-mcc-blockers.md` | `ccb22d5d` | 116 |
| Reduced-model, cusp-loss and topology blockers | `modern/docs/literature/reduced-models-cusp-topology-blockers.md` | `66879e00` | 72 |
| Surrogate, MDO and external-validation blockers | `modern/docs/literature/surrogate-mdo-validation-blockers.md` | `b6bb6215` | 157 |

345 verified references in total. Later reviews indexed here but not reconciled into the
decision tables below (each carries its own recommendation table): `twt-ppm-physics-for-hemp.md`
(`beb4772c`, 51 references, TWT/PPM focusing physics and a read-only field check) and
`pic-acceleration-methods.md` (2026-09-04, 147 references: how the `cft_revival.pic2d` PIC-MCC
could be made 5-10x faster - energy-conserving / semi-implicit / implicit schemes, similarity
scaling, time acceleration, Poisson solvers, kernel engineering, variance reduction - with the
speed-up, claim-risk and verification protocol per method in its section 8; it extends
`pic-mcc-blockers.md` blockers 1 and 2).

Every decision below cites the review row it answers; the
effort and risk columns are the reviews' own words, not re-estimates. Repository facts are
read from `modern/docs/ROADMAP_AUDIT.md`, `modern/docs/workstreams/pic2d-campaign-v1-proposal.md`,
`modern/docs/workstreams/pic2d-devlog.md` (phase 4), the recorded bundles named in the
stage-ladder tracker, and `paper/manuscript.tex` (Discussion and Limitations at `b6bb6215`).

## 0. Vocabulary, rules and what is already in flight

### 0.1 Decision vocabulary

- **ADOPT** - goes into the ordered roadmap with an owning experiment or package, a step number
  and gates; "adopt" is a planning decision, not evidence.
- **DEFER** - agreed in principle but placed after a named prerequisite; the trigger that
  un-defers it is stated.
- **REJECT** - not taken, with the reason; the review's argument is recorded so the rejection
  can be revisited.

Decisions are taken per *atomic* recommendation, because several summary-table rows bundle
three to five distinct actions. Section 4 gives the counts at both granularities.

### 0.2 Closure identifiers (one namespace, three reviews)

The reviews introduce closures with overlapping short names. This document uses the full
identifiers and asks every new protocol to do the same:

| identifier | meaning | introduced by |
| --- | --- | --- |
| `CL-1` | `S = prod_k (1 - p_k)` with per-cell test-particle wall-access fractions as `p_k` (MDO v1/v2 primary) | recorded (`mdo_l0_campaign_v2/protocol.json`) |
| `CL-2` | `S = 1 - p_pooled` (MDO v2 sensitivity) | recorded |
| `CL-3-sheath-limited` | `p_k,eff = A_k exp(-(phi_k - phi_ck)/T_k)`, `A_k` = geometric access fraction with its Wilson interval, sheath drop from the floating-dielectric row | reduced-models review section 2.4 |
| `CL-4-hybrid-area` | electron loss current to cusp k through a hybrid-gyroradius area `c sqrt(r_e r_i) 2 pi r_w` with `exp(-Delta phi_s/T_k)`, prefactor `c` swept over [1, 4] | reduced-models review section 2.4 |
| `CL-3-potentials` | potential closure for the four-cell balance: flat interior potential and one exit drop (Koch 2011 finding (ii); Brandt 2016) | reduced-models review section 1.5 |

The bare label "CL-3" is therefore ambiguous (a cusp-loss closure for the MDO chain versus a
potential closure for the discharge network). The plasma-network v2 stream should record its
potential closure as `CL-3-potentials` (or a distinct prefix such as `PC-1`) so that MDO v3's
`CL-3-sheath-limited` can cite it without collision.

### 0.3 Work already launched in response to the reviews (referenced, not re-planned)

| stream | branch / worktree | state at 18:10 AEST | what it implements |
| --- | --- | --- | --- |
| Cusp topology search v3 | `exp/cusp-topology-search-v3` (worktree `uni-project-topo-v3`, base `66879e00`) | running; zero commits | the literature definition (axis null -> separatrix -> wall intersection `z_c`; Gildea 2012, Lewerentz and Schneider 2023, Kornfeld 2007) over the 96 sweep-v2 designs plus the P2 divergent-exit field; output: a cusp / cell catalogue |
| PIC model v1.4 + steady-state v3 | `feat/pic-2d-axisymmetric` (worktree `uni-project-pic2d`, base `b6bb6215`) | running; zero commits; W x 0.7 case on the GPU since 17:04 AEST (12,600 s budget) | wall-ion recycling in the neutral inventory, fail-closed peak-node Debye gate + grid-heating triad, CUDA-graph capture of the whole step, Bohm-scattering and SEE sensitivity hooks; steady-state v3 after W x 0.7 ends |
| Four-cell power balance v2 (sheath closure) | `feat/plasma-network-v2-sheath` (worktree `uni-project-plasma-v2`, base `b6bb6215`) | being launched; zero commits | rows R28-R31 (per-cusp floating-dielectric sheath, anode sheath) and the declared potential closure `CL-3-potentials`; a **new module** beside `cft_revival.plasma`, because the five admitted `cft_revival/plasma/*.py` files are hash-bound by `paper/evidence/manifests/four-cell-closure.json` and any edit fails `check_paper.py` |

### 0.4 Cross-review reconciliation

Where two reviews recommend the same thing under different words, this document keeps one item:

- Closure calibration from PIC per-cusp fluxes: reduced-models row 2 ("calibrate against
  `pic2d` steady-state v2 per-cusp fluxes") and surrogate/MDO row 7 ("calibrate `p_k` from the
  PIC plateau at one point, re-check at a second") are the same study; it is one deferred item
  (D-4 in section 5.9).
- The long neutral-dynamic development case appears in PIC rows 2 and 5; counted once.
- Validation targets: the PIC review's section 6 table and the surrogate review's section 3.2
  table overlap (Brandt 2016, Keller 2015, Koch 2011, HIT, DCFT); merged into validation
  v0-v2 in the surrogate review's ASME V&V 20 form.
- Screening v2 is asked for by the surrogate review (row 1) and implied by the reduced-models
  review (row 3 "re-run the characterization on the same designs"; row 4 descriptors); it is
  one experiment whose launch cells come from the topology v3 catalogue.

## 1. Decision table - PIC-MCC review (`pic-mcc-blockers.md` section 7, 9 rows)

Effort key as in the review: S < 1 day, M 1-3 days, L > 3 days before GPU time.

| # | recommendation (review row) | decision | one-line reason | owner | step | effort (review) | risk (review) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1a | Peak-node Debye gate, fail-closed (`stability_limits.max_cell_debye_ratio` on the instantaneous window-peak node density) | ADOPT | the development run passed at 3 lambda_D because the gate looked at a reference density; the gate is in PIC v1.4 | `cft_revival.pic2d` (v1.4, launched) | 6a | S | low |
| P1b | 30 um grid (100 x 800) with the 20 um refinement case as the grid discriminator (10 % agreement) | ADOPT | 2 lambda_D per cell at the design peak is what Brandt 2016 ran; the refinement case decides the grid claim | `pic2d_campaign_v1` blocks A / C | 6d | S | the 20 um case costs 18-20 GPU-h |
| P1c | Grid-heating triad recorded and gated (ledger residual / electrode work; T_e and ionisation rate on W and grid variants; omega_pe dt drift) | ADOPT | all three signatures already appeared on the snapshot-v2 coarse pair; in v1.4 | `cft_revival.pic2d` (v1.4) | 6a | S | low |
| P1d | No artificial permittivity scaling in any campaign claim | ADOPT | a 2 mm bore whose cusp sheaths are the physics of interest cannot have lambda_D multiplied by sqrt(gamma) (Cho 2013; Taccogna and Minelli 2018) | `pic2d_campaign_v1/protocol.json` (policy field) | 6d | - | none |
| P1e | Energy-conserving deposition / gather pair as a switch in `kernels.py`, compared on the snapshot-v2 coarse case | DEFER | research item with no cusp-thruster precedent; Adams 2025 show EC schemes still heat for drifting beams; un-deferred after campaign v1 records | `cft_revival.pic2d` | after 6d | M | may heat for drifting beams |
| P2a | Capture the whole step in one CUDA graph; bitwise-test against the un-captured path | ADOPT | the ~1.2 ms WDDM launch floor is 60 % of the 2.0 ms step; no numerics change; in v1.4 | `cft_revival.pic2d` (v1.4) | 6a | S-M | graph capture must be bitwise-tested |
| P2b | Periodic particle sort by cell (Bowers 2001) | DEFER | only worth it if gather / deposit is memory-bound once the launch floor is gone; measure after P2a lands | `cft_revival.pic2d` | after 6a measurement | S-M | tallies must stay exact |
| P2c | Move the campaign to a TCC-mode or Linux host (~4x throughput) | DEFER | no such host is available; the only change that removes the floor; un-deferred by hardware access | external | - | external | none |
| P2d | Re-scope "converged" as a plateau under the quasi-steady neutral closure, and price one long (>= 50 us) development case at reduced W before freezing the campaign length | ADOPT | Brandt 2016: 76 us for steady state, ~10x more for breathing; a 5-transit plateau is a property of the neutral closure, not of the discharge | `pic2d_cft_development_50us` (new development run) | 6b | M | 10-20 GPU-h; no literature precedent for our budget |
| P3a | Wall-recycling term in the 0-D inventory now: `V dn_g/dt = Q_in + R_wall - S - c n_g` | ADOPT | 3.72 mA of Xe+ (59 % of ionisation) is removed from the atom inventory today; every kinetic-neutral thruster PIC recycles it; in v1.4 | `cft_revival.pic2d.neutrals` (v1.4) | 6a | S | fixed point moves: n_g* 2.97e19 -> ~4.5e19 before S responds |
| P3b | Replace the uniform inventory by the axial free-molecular column model implemented as Katz-Mikellides 2011 view factors on the (r, z) mask (anode feed, per-cell MCC sink, diffuse wall re-emission, exit effusion) | ADOPT | utilisation with a uniform n_g is self-inconsistent (depletion is local to the ionisation peak, Petronio 2024); model v2.0, after the 50 us case decides whether the closure oscillates | `cft_revival.pic2d` model v2.0 | 6c | M-L | a breathing closure may have no plateau at all |
| P3c | Artificial 30 ns relaxation kept only as a documented option, default off | ADOPT | the relaxation removes the transport delay that produces breathing (Lafleur 2021; Chapurin 2021); the 50 us case runs without it | `pic2d-model-v2.0.json` | 6b / 6c | S | the plateau criterion may need to become period-averaged |
| P3d | Quote net (recycled) utilisation, never gross | ADOPT | Brandt 2016's 24 % is net; our 46 % is gross (section 6.4 of this document) | dashboard generator + `protocol.json` named outputs | 6a | S | none |
| P4a | Xe+-Xe elastic and charge-exchange collisions with hash-bound Miller 2002 tables, reported as a sensitivity | ADOPT | in-channel effect is "few %" on the beam fraction (proposal 3.1) but becomes a blocker with a plume region; hash-bound like Biagi | `cft_revival.pic2d` model v2.0 | 6c | S-M | small in-channel effect expected |
| P4b | SEE: Vaughan yield for BN, Hobbs-Wesson space-charge cap, cusp-local yield diagnostics; SEE-off as the reported sensitivity | ADOPT | wall electron current equals wall ion current (3.73 vs 3.72 mA), so SEE changes the electron energy balance by O(1); hooks in v1.4, full model in v2.0 | `cft_revival.pic2d` (hooks v1.4; model v2.0) | 6a / 6c | M | space-charge-limited cusp sheaths |
| P4c | Declare "no anomalous transport" in the claim boundary; add a Bohm-scattering sensitivity block (nu_an = alpha omega_ce, alpha in {1/64, 1/16}); diagnose the axisymmetric low-frequency mobility (Cho 2015 method) | ADOPT | an axisymmetric ES code excludes the drift instability by construction; the block is a bracket, not a model; hooks in v1.4 | `cft_revival.pic2d` (hooks v1.4); `pic2d_campaign_v1` block E | 6a / 6d | S | the campaign cannot bound the physical transport |
| P4d | Plume extension (proposal 3.4a: 6 mm region, Neumann side, Dirichlet far plane) plus a current-continuity injection variant; freeze one, report the others | ADOPT | the Dirichlet 0 V plane 6 mm downstream fixes phi where the plume should be free (phi_max 337 V); also the prerequisite of validation v1 | `cft_revival.pic2d` model v2.0; `pic2d_campaign_v1` block D | 6c | M | Brandt 2016 found 20 x 5 mm still too small for plume ratios |
| P4e | Per-cusp leak width, local lambda_D, cells per lambda_D and sheath drop as campaign diagnostics; refinement case decides | ADOPT | no cusp-thruster PIC reports a cusp-sheath convergence study; cheap | `pic2d_campaign_v1` named outputs | 6d | S | none beyond P1 |
| P5a | Tightened plateau criterion (3 %, two consecutive checkpoints, >= 5 transits, peak density and omega_pe dt tracked) plus a spectral check of I_d and n_g (no line above the noise floor below 1 MHz) and the statement that the plateau is conditional on the neutral closure | ADOPT | the development run passed a 5 % drift rule while omega_pe dt was still rising | `pic2d_campaign_v1/protocol.json` | 6d | S | none |
| P5c | >= 3 seeds and the W variant; report mean +- sample SD, batch-means SE with block length, shot-noise floor | ADOPT | the seed pair showed a 3-7x excess over 1/sqrt(N); no published seed variance exists for any HEMP PIC | `pic2d_campaign_v1` blocks A / B | 6d | S | if the closure breathes the campaign becomes period-averaged and 2-3x longer |
| P5d | Benchmark-style convergence statement: same simulated window, ppc doubled, 5 % agreement on time-averaged axial E_z, n_i, T_e profiles as well as scalars | ADOPT | Charoy 2019 / Villafana 2021 practice; the W/2 block already exists | `pic2d_campaign_v1` block B | 6d | S | none |
| P6a | Report the literature's observables as named protocol outputs: anode electron current, net ionisation fraction, beam current and fraction, wall ion current and mean energy per cusp, potential step per cusp, plume divergence, axial max_r n_e against the cusp planes | ADOPT | these are the only quantities any published analogue reports; they feed validation v0 | `pic2d_campaign_v1/protocol.json`; dashboard generator | 6d | S-M | none |
| P6b | "Literature context" panel with Brandt 2016 and Keller 2015 numbers beside ours, labelled different closure / different geometry, never a gate | ADOPT | cheap; makes the closure difference visible | steady-state / campaign dashboard | 6a | S | none |
| P6c | No external-validation claim from the campaign; GATE-L3 stays closed | ADOPT | there is no experiment at 2 mm bore / 0.19 sccm / 3 mA | paper policy | - | - | none |

Row P5b (long neutral-dynamic case before freezing) is the same item as P2d and is not counted
twice.

## 2. Decision table - reduced-model, cusp-loss and topology review (`reduced-models-cusp-topology-blockers.md` section 5, 4 rows)

| # | recommendation (review row) | decision | one-line reason | owner | step | effort (review) | risk (review) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1a | Accept the two `PROPOSED_NOT_ACCEPTED` corrections (ionisation booked once; anode electron term with the electron's sign) **only together with** rows R28-R30 (per-cusp floating-dielectric sheath: potential drop, electron `2T + Delta phi`, ion `Delta phi + T/2`) and R31 (anode sheath or declared anode fall) | ADOPT | the corrections alone lower the rank to 21 and free four potentials; the literature closes potentials with sheath rows (Goebel 2007; Lieberman 2005); launched as network v2 | new module beside `cft_revival.plasma` (`feat/plasma-network-v2-sheath`, launched) | 5 | Medium | new rows need `n_e` |
| R1b | Declare `CL-3-potentials` (flat interior potential, one exit drop) until a cusp-conductivity row exists | ADOPT | Koch 2011 finding (ii) and Brandt 2016 both describe this structure; it must be labelled as declared, not derived | same module; ledger entry | 5 | Medium | a declared closure can be mistaken for a derived result unless labelled |
| R1c | Add a density / neutral balance, or take `n_e,k` per cell from `pic2d` | ADOPT (pic2d input) | the steady-state v2 plateau gives per-cell densities at 300 V; the neutral / mass-flow balance rows are a network v3 item | same module; input labelled development-tier from `pic2d_cft_steady_state_v2` (later v3) | 5 | Medium | `n_e,k` is a single-seed development number until steady-state v3 |
| R1d | Re-admit under `analytic-consistency` with Kornfeld Table 3.1 (DM9.2: phi_1 14.1 V, phi_2..4 ~ 1000 V, T2 100.1 eV, T3 43.1 eV, T4 23.5 eV, p = 0.06 / 0.119 / 0.160 / 0.254) and Puca 2024 Table 1 as reproduction targets, neither treated as truth | ADOPT | the only published states of this model; the existing gate kind admits a derivation verified numerically | `paper` (new gate; not this document) | 5 -> paper | Medium | the corrected + closed system may not reproduce either published state as a root (a recorded outcome) |
| R2a | Relabel the test-particle estimand as `collisionless_geometric_wall_access_fraction` in every consumer; coupling export field description at the next coupling revision | ADOPT | no published closure uses a collisionless wall-hit fraction as a cusp probability; it is an upper bound on sheath-free access | `cft_revival.coupling` (schema description, next revision); screening v2 protocol; paper at next admission | 2 (protocol wording), then coupling revision | Medium | none |
| R2b | Add `CL-3-sheath-limited` and `CL-4-hybrid-area` as declared closures for the MDO chain | ADOPT | the ranking is a property of the closure (CL-1 vs CL-2 Jaccard 0); two physically motivated closures bracket it | `mdo_l0_campaign_v3` model | 4 | Medium | sheath drop depends on SEE yield and `T_k`; leak-width prefactor uncertain by 1-4 |
| R2c | Calibrate / compare `A_k exp(-Delta phi_s/T_k)` against the `pic2d` per-cusp electron flux fraction before any MDO uses `CL-3`; compare qualitatively with Brandt 2016 | DEFER (deviation: MDO v3 uses `CL-3` / `CL-4` as declared scenarios, not calibrated closures) | the only kinetic result on our geometry is single-seed, development-tier and lacks wall recycling; steady-state v3 supplies the flux fractions; same item as S7 | closure calibration study (D-4, section 5.9) | after 6a (and a second operating point) | Medium | a calibrated closure may not hold self-consistently (Marks and Jorns 2023) |
| R2d | Keep the magnetic-moment variation as a published diagnostic beside every screening number | ADOPT | v4 and screening v1 already record it; shows where the loss-cone formula is inapplicable | `orbit_wall_loss_geometry_screening_v2` outputs | 2 | - | none |
| R3a | Topology search v3: cusp := axis null with converged Jacobian + its separatrix traced to the wall intersection `z_c,k` (with `B_z(r_w, z)` sign change and `|B_r(r_w, z)|` maximum checks); cell := region between consecutive separatrices; stability `|Delta z_c,k| <= 0.8 mm` across maps and unchanged axis-null count; per-cell mirror ratio as a descriptor | ADOPT | the standard PPM topology has no wall-side vector null; v2 / v1 searched for one by construction; launched | `exp/cusp-topology-search-v3` (launched; 96 designs + P2) | 1 | Low-medium | `z_c` may land outside the straight channel for many designs (a real finding) |
| R3b | Run the P2 field on the four representatives with and without a declared iron spacer to bound the shift in `z_c,k` and in the wall `|B_r|` maximum | DEFER | needs a material-aware field solve (L1b is `SCREENING_NOT_ACCEPTED`, 54/54 gates `NOT_EVALUATED`; ROADMAP_AUDIT section 4 item 8) | `cft_revival.fem_reference` / `material_fields` | after material-aware field qualification | Low-medium | none |
| R4a | MDO v3 evaluates `CL-1`, `CL-2`, `CL-3-sheath-limited`, `CL-4-hybrid-area` on the same catalogue, publishes the four fronts with pairwise Jaccard, and reports a design as robust only if nondominated under >= 3 closures | ADOPT (as a reported set beside the Ide-Schoebel sets of S6, never a gate) | v2 already shows the CL-1 / CL-2 disagreement; the >= 3-closure set is the "majority-robust" set between highly-robust (all) and flimsily-robust (any) | `mdo_l0_campaign_v3` | 4 | Medium | trend gates are hardware-specific |
| R4b | Literature trend gates, reported not binding: (a) weaker in-channel field -> higher thrust and anode efficiency (Hu 2016 AIP Adv.); (b) longer last stage -> higher, longer middle stage -> lower (Hu 2016 J. Phys. D); (c) interior optimum in channel length (Liu 2019, 2021); (d) exit separatrix more parallel to the exit plane -> narrower plume (Liu 2021; Li 2026); a closure contradicting all four is flagged | ADOPT | the only geometry -> performance evidence is experimental and hardware-specific; a flag, not a falsification | `mdo_l0_campaign_v3` assessment | 4 | Medium | contradiction is a flag, not a falsification |
| R4c | Exit-separatrix angle and per-cell mirror ratio as catalogue descriptors | ADOPT | the experimental literature varied exactly these; topology v3 produces them | topology v3 catalogue -> screening v2 dataset -> MDO v3 descriptors | 1 -> 2 -> 4 | Medium | none |
| R4d | Label every validation target as model-to-model or single-point (Kornfeld DM9.2, Keller 2015 points, Yeo 2020 S1 PIC row) | ADOPT | none is a measurement of our device | validation v0-v2 protocols; `data/validation/*` | 7 | - | none |

## 3. Decision table - surrogate, MDO and validation review (`surrogate-mdo-validation-blockers.md` section 5, 9 rows)

| # | recommendation (review row) | decision | one-line reason | owner | step | effort (review) | risk (review) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1a | Screening v2 option A: keep all 96 designs; add launches only to cells whose Wilson 95 % width exceeds 0.10 (non-saturated cells), bringing each to n = 512 | ADOPT (with the two-stage frozen rule of section 5.2, because the launch cells now come from the topology v3 catalogue) | 73 of 96 designs carry a 128/128 cell that gains nothing from replication; the v2 error sits in cells 1 and 4; lowers the cell floor from ~0.040 to ~0.020 where it matters | `orbit_wall_loss_geometry_screening_v2` (orbit_mc 1.7) | 2 | 1 d protocol + ~1.5-3 h CPU + 1 d record / audit | budget under contention (v1 ran 35 % over); the allocation rule must be frozen before the counts are seen |
| S1b | Screening v2 option B: 2048 launches (512 per cell) on 40 designs | REJECT | drops 56 catalogue designs that MDO v3 ranks and the trend gates need; the v3 catalogue re-partitions cells so v1's per-design "flat learning curve" does not transfer; option A buys the same floor reduction where the error is for ~0.9x v1's orbit count | - | - | (same row) | (same row) |
| S1c | Scrambled-Sobol launch set stratified by cell and pitch-angle sign (Caflisch 1998) | ADOPT | replaces the O(n^-1/2) binomial error with a discrepancy error for the smooth part; stratification by cell is already in place | `orbit_wall_loss_geometry_screening_v2` | 2 | (same row) | none |
| S1d | Run at the accepted N time step with a 10 % 2N control sample (MLMC logic; N -> 2N differences <= 0.0059 are below the target floor) | ADOPT | v1 ran every design at N and 2N; the control keeps the convergence evidence at a tenth of the cost | `orbit_wall_loss_geometry_screening_v2` | 2 | (same row) | none |
| S1e | Same launch set across designs only if the estimand is a between-design difference; independent sets if it is the per-design mean surface (Chen et al. 2012) | ADOPT (independent scrambled-Sobol sets per design) | the primary estimand is the per-design mean surface (surrogate labels, MDO posteriors); common random numbers degrade it | `orbit_wall_loss_geometry_screening_v2` protocol | 2 | (same row) | none |
| S2a | Surrogate v3 gates in probability units against the floor computed from the actual n: cell RMSE <= 1.5x the per-cell binomial floor (0.030 at n = 512); pooled RMSE <= 1.5x the pooled floor; 90 % interval coverage in [0.85, 0.97] under the binomial predictive | ADOPT | a 0.05 gate against a 0.035-0.040 floor tested the labels, not the model | `wall_loss_geometry_surrogate_v3/protocol.json` | 3 | 2-3 d (with S2b, S2c, S3) | a third rejection if screening v2 is not run first |
| S2b | Replace "2x best baseline" by a split-half reliability ceiling: a surrogate is useful if it recovers >= 70 % of the attainable R^2 | ADOPT | v2 made the 2x gate unmeetable by construction (ridge 0.0334 vs GP 0.0337); the ceiling is observation-level and model-free | `wall_loss_geometry_surrogate_v3` | 3 | (same row) | the ceiling itself may be low (then there is nothing to learn - an honest negative) |
| S2c | Primary model declared a priori (binomial-likelihood GP on the v2 derived features plus cusp / null distances in pitches, per cell; binomial GLM as the parametric candidate); nested CV with >= 26 designs in the selection role, or no selection at all | ADOPT | method selection on 10 designs was unstable (Cawley and Talbot 2010) | `wall_loss_geometry_surrogate_v3` | 3 | (same row) | none |
| S3 | Binomial-likelihood GP (variational or EP) per cell instead of delta-method logit noise | ADOPT | the coverage misses at truth = 1.0 are a likelihood misspecification; saturated cells are handled natively | `cft_revival.surrogates` (new model class + tests) before v3 | 3 (tooling first) | included above | calibration of variational posteriors needs its own check; GP tooling not yet in repo |
| S4 | MDO v3 design representation: LVGP or descriptor kernel with the per-cell screening P(wall) and the v2 top features as continuous descriptors; exhaustive categorical stage kept as a baseline | ADOPT | v2 seed 101 stalled on design 50 because the categorical kernel treats 96 designs as exchangeable | `cft_revival.optimization` (kernel) + `mdo_l0_campaign_v3` | 4 | 2 d | descriptors are noisy labels; propagate their Beta posteriors |
| S5a | >= 10 seeds per optimiser (15 if the budget allows) | ADOPT (>= 10; 15 conditional on wall time) | three seeds report counts, never significance (v1 audit F17) | `mdo_l0_campaign_v3` | 4 | +5-8x optimiser CPU (v2: 83 min for 3 seeds) | wall time under contention |
| S5b | Report HV and IGD+ and the additive epsilon indicator with bootstrap CIs over seeds; reference point by Ishibuchi et al.'s rule, recorded in the protocol; paired Wilcoxon only at n >= 10; effect sizes always | ADOPT | HV ranking depends on the reference point; no unary indicator ranks fronts in general | `cft_revival.optimization` (indicators) + `mdo_l0_campaign_v3` | 4 | (same row) | none |
| S6 | Report Ide-Schoebel sets across closures: highly robust (intersection), flimsily robust (union), minmax regret; never a pooled or averaged front | ADOPT | the CL-1 / CL-2 disagreement is the result; equal-weight pooling is a choice that must be argued (Tebaldi and Knutti 2007) | `mdo_l0_campaign_v3` assessment + dashboard | 4 | 1 d | none |
| S7 | Calibrate `p_k` from the PIC plateau wall fluxes at one point, re-check at a second (Marks-Jorns test) | DEFER (= R2c) | see R2c; the plateau lacks wall recycling and a second operating point does not exist | closure calibration study (D-4) | after 6a | 5-10 d + GPU | a calibrated closure may not hold self-consistently |
| S8a | Validation v0: code-to-code against a published Greifswald HEMP PIC case, tabulated LANDMARK-style | ADOPT (Brandt 2016 as the case, because it publishes the numbers) | the only analogue with published scalars near our scale; labelled model-to-model by closure | `validation_v0_code_to_code` under `cft_revival.validation` v2 contracts | 7a | 5-8 d | the Brandt field / magnet set must be reconstructable from the paper |
| S8b | Validation v1: main ion-energy peak relative to Ua and the angle of peak ion current density from the PIC plateau at 300 V against Koch 2011 (peak ~Ua - 15 V, 20 deg) and HIT RPA (Hu 2016), `E +- u_val` with `u_num` from a dz refinement, `u_input` from n_g and Q_in, declared `u_D` | ADOPT | the cheapest first E +- u_val exercise; needs the plume extension (P4d) and a refined-grid plateau | `validation_v1_rpa_peak_angle` | 7b | 3-5 d after a refined-grid plateau | published uncertainties missing (declare an assumed `u_D`) |
| S8c | Validation v2: thrust-voltage curve against Keller 2015 mu-HEMPT (50-360 uN; 50 uN at 600 V) or the low-power HEMP of Liu 2019, area metric, one operating point withheld | ADOPT | the right device class; the geometry must be modelled from published dimensions or author contact | `validation_v2_thrust_voltage` | 7c | 10-20 d + data access | device geometries may need author contact |
| S8d | Validation v3: geometry-trend comparison (sign / monotonicity of efficiency vs magnet / stage length and vs in-channel field strength) against HIT data | DEFER | needs a geometry -> performance chain with a physics-bearing closure; today the chain stops at the access fraction | after MDO v3 + D-4 | after 4 and D-4 | - | none |
| S9a | Register each `preregister ...` commit hash in an external registry (or mirror signed tags to a second host) at the freeze; record the receipt in `authorities.json` | ADOPT | Git history can be rewritten; a third-party timestamp is what preregistration means (Nosek 2018) | `cft_revival.experiment_runtime` + every new protocol from screening v2 on | 8 (applies from 2) | 1-2 d once | none material |
| S9b | Package each accepted bundle as an RO-Crate with the existing manifest as payload | ADOPT | small metadata cost; makes the bundles FAIR | `cft_revival.experiment_runtime` export stage | 8 | (same row) | none |
| S9c | `gate_genealogy` block in every protocol (predecessor gates and the evidence that motivated each change) | ADOPT | v1 -> v2 -> v3 surrogate gate changes are each justified, but a reader cannot tell justified from data-driven without the genealogy | protocol schema; first use in screening v2 and surrogate v3 | 8 (applies from 2) | (same row) | none |
| S9d | Shakedowns on synthetic or permanently excluded designs; every shakedown outcome logged; method-selection rules not editable after a shakedown | ADOPT | v1 surrogate's method instability is the visible symptom of shakedown peeking (Sagarin 2014; Albers and Lakens 2018) | every new campaign from screening v2 on | 8 (applies from 2) | (same row) | none |
| S9e | ADEMP fields (aims, data-generating mechanism, estimands, methods, performance measures) as mandatory protocol sections | ADOPT | most already exist under other names in `protocol.json`; naming them makes the structure auditable | protocol schema | 8 (applies from 2) | (same row) | none |

## 4. Counts

Atomic recommendations after de-duplication (S7 = R2c; P5b = P2d): **60**.

| decision | count | items |
| --- | --- | --- |
| ADOPT | 53 | P1a-d, P2a, P2d, P3a-d, P4a-e, P5a, P5c, P5d, P6a-c, R1a-d, R2a, R2b, R2d, R3a, R4a-d, S1a, S1c-e, S2a-c, S3, S4, S5a, S5b, S6, S8a-c, S9a-e |
| DEFER | 6 | P1e (EC deposition switch), P2b (cell sort), P2c (TCC / Linux host), R2c = S7 (closure calibration from PIC), R3b (iron sensitivity), S8d (validation v3 geometry trends) |
| REJECT | 1 | S1b (screening v2 option B: 40 designs x 2048 launches) |

By summary-table row (22 rows across the three reviews): every row has at least one adopted
component; six rows carry a deferred component (PIC rows 1 and 2; reduced-models rows 2 and 3;
surrogate rows 7 and 8); one row carries a rejected alternative (surrogate row 1).

## 5. Revised ordered roadmap

The order is a dependency order for the MDO chain (steps 1-4), with the two kinetic / reduced-model
streams (5, 6) running concurrently on their own branches and validation (7) and hygiene (8)
consuming their outputs. The single RTX 5090 (WDDM) serialises every GPU item; CPU items run
beside it. All estimates are planning figures derived from recorded rates (screening v1:
100,352 orbits in 95 min on 12 workers under load; MDO v2: 83 min for 3 BO seeds; steady-state
v2: 2.0 ms/step at ~1 M + 1 M macro-particles; campaign proposal section 5).

Dependency graph:

```
topology v3 (1) --cells, descriptors--> screening v2 (2) --labels--> surrogate v3 (3)
                                                   \--posteriors, descriptors--> MDO v3 (4) <--sheath drops / T_k (soft)-- plasma network v2 (5)
PIC v1.4 (6a) --> >= 50 us case (6b) --> breathing decision (6c) --> campaign v1 (6d) --> validation v0 (7a), v1 (7b), v2 (7c)
steady-state v3 (6a) --per-cell n_e, per-cusp fluxes--> plasma network v2 (5); closure calibration D-4 (deferred)
plume extension (6c, block D) --> validation v1 (7b)
hygiene (8) applies to every preregistration from (2) onward
```

### 5.1 Step 1 - Cusp topology search v3 (RUNNING)

- **Inputs.** The 96 accepted sweep-v2 field maps (`l1a_geometry_sweep_v2`, `f30cb42e`) and the P2
  divergent-exit field (`a1158bad`); the frozen v2 / v1 protocols as lineage (not edited); the
  definition of R3a.
- **Outputs.** A cusp / cell catalogue per design: axis nulls `(0, z_k)` with Jacobian, separatrix
  wall intersections `z_c,k`, cell boundaries, per-cell mirror ratio, exit-separatrix angle,
  distance of `z_c,k` from the stage boundaries; cross-map stability flags; the recorded count
  of designs with 2, 3, 4 channel-interior wall intersections plus the exit cusp.
- **Gates.** Preregistered one-shot through `experiment_runtime`; converged axis-null Jacobians;
  `|Delta z_c,k| <= 0.8 mm` across primary / downsampled / enlarged maps; unchanged axis-null
  count; `B_z(r_w, z)` sign change and `|B_r(r_w, z)|` maximum within tolerance of `z_c,k`;
  GPU replay of the field maps as in v2 / v1.
- **Estimate.** CPU <= 1 h (separatrix tracing on stored psi maps; v1 characterised 56 x 3 maps);
  no GPU.
- **What would stop it.** `z_c,k` outside the straight channel for most designs (a real finding
  that shrinks the catalogue); cross-map instability of `z_c,k` (a second null, now under the
  literature definition); disagreement between the L1a and P2 maps on the representatives.

### 5.2 Step 2 - Wall-loss geometry screening v2 (QUEUED, after 1)

- **Inputs.** The v3 cell catalogue (launch cells are separatrix-bounded cells, not protocol
  positions); orbit_mc 1.7.0 and its accepted N / 2N time steps; the v1 protocol as lineage.
- **Allocation rule (frozen before any launch).** Stage 1: n = 128 scrambled-Sobol launches per
  v3 cell for all 96 designs (about 49 k launches at N for four cells per design). Stage 2:
  every cell whose stage-1 Wilson 95 % width exceeds 0.10 is topped up to n = 512 (about 88 k
  launches at v1's 2.4 non-saturated cells per design). A 10 % 2N control sample runs on a
  frozen subset of cells. Independent launch sets per design (S1e). The rule is the
  preregistration; no allocation decision is taken after the counts are seen.
- **Outputs.** `geometry-wall-loss-dataset` v2 with per-cell counts at their actual n, Wilson
  intervals, the magnetic-moment diagnostic, the v3 descriptors (mirror ratio, separatrix
  angle) and the label `collisionless_geometric_wall_access_fraction` on every number; the
  coupling consumer record as in v1.
- **Gates.** Structural gates of v1 (authority replay, sealing, timeout-free, validator count);
  N -> 2N control agreement `|Delta P| <= 0.0059` (v1's maximum) on the control cells; the
  allocation rule applied exactly; `gate_genealogy` and ADEMP fields present (S9c, S9e);
  shakedown on excluded designs (S9d).
- **Estimate.** About 137 k launches at N plus about 9 k at 2N (about 155 k N-orbit-equivalents,
  1.5x v1) -> CPU 2.5-4 h on 12 workers under contention; no GPU. Record / audit 1 d.
- **What would stop it.** Fewer than two interior cells for most designs after step 1 (then the
  dataset is a two-cell dataset and the protocol says so); any orbit_mc change after the
  freeze (re-shakedown); CPU contention beyond the 35 % factor.

### 5.3 Step 3 - Wall-loss geometry surrogate v3 (QUEUED, after 2)

- **Inputs.** The v2 dataset at n = 512 in the non-saturated cells; the v2 derived features plus
  per-cell cusp / null distances in pitches (from the v3 catalogue); the frozen assessment /
  extrapolation partition inherited from v1 / v2 or re-frozen before the labels are read.
- **Tooling first.** A binomial-likelihood GP (variational, GPyTorch approximate GP in
  `.venv-sota`, or EP) and a binomial GLM added to `cft_revival.surrogates` with tests and a
  synthetic calibration check before the protocol is frozen.
- **Outputs.** Per-cell predictors with binomial predictive intervals; the split-half reliability
  ceiling per cell; the fraction of attainable R^2 recovered; the predictor contract published
  for audit; `usable_as_mdo_input` decided by the gates only.
- **Gates.** S2a (cell RMSE <= 1.5x floor at the actual n, 0.030 at n = 512; pooled <= 1.5x
  pooled floor; coverage in [0.85, 0.97] under the binomial predictive); S2b (>= 70 % of the
  reliability ceiling); nested CV or a single declared model; `no_tautology`, single-use
  labels, frozen partition unchanged.
- **Estimate.** CPU 1-3 h (per-cell variational GPs on <= 96 designs; nested CV multiplies);
  no GPU (BoTorch / GPyTorch fits are faster on CPU here while the GPU is busy).
- **What would stop it.** A low split-half reliability (labels not reproducible across halves ->
  nothing to learn, recorded as an honest negative); the GP tooling failing its calibration
  check; a third rejection on S2a (recorded, never re-run on the same labels).

### 5.4 Step 4 - MDO L0 campaign v3 (QUEUED, after 2; 3 optional)

- **Inputs.** The v2 catalogue with per-cell Beta posteriors at their actual n and the v3
  descriptors; the corrected L0 chain; closures `CL-1`, `CL-2`, `CL-3-sheath-limited`,
  `CL-4-hybrid-area`; sheath drops `phi_k - phi_ck` and `T_k` from plasma network v2 if it has
  landed, otherwise declared scenario parameters swept over a stated range (`Delta phi_s/T_k`
  in [3, 6] for a floating Xe sheath without SEE; `c` in [1, 4] for CL-4) and labelled as such;
  surrogate v3's predictor only if its gates passed (otherwise labels direct, as v2).
- **Outputs.** Four per-closure fronts with pairwise Jaccard; the Ide-Schoebel highly-robust,
  flimsily-robust and minmax-regret sets; the >= 3-closure majority-robust set; HV, IGD+ and
  additive epsilon with bootstrap CIs over >= 10 seeds and the Ishibuchi reference point
  recorded; the four literature trend flags (R4b) per closure; the closure-form uncertainty
  stated as the result.
- **Gates.** Integrity gates of v2 (12, including the import-bound hash scope) plus: seed count
  >= 10; indicators and reference point declared; no pooled front anywhere in the bundle; an
  efficacy statement only if the pre-declared efficacy gate ("median final HV over >= 10 seeds
  >= 0.9x the dense reference at the fixed budget") is in the protocol.
- **Estimate.** Dense references for four closures ~4 x 1 min on 12 workers; BO 10 seeds x
  ~30 min (v2 rate under contention) ~5 h; NSGA-III / LHS minutes -> CPU 5-8 h; no GPU.
- **What would stop it.** Wall time under contention (10 seeds x 3 optimisers); an empty
  highly-robust set (the honest result, not a stop); CL-3 / CL-4 inputs missing (then the
  scenario sweep is the declared fallback and the protocol says so).

### 5.5 Step 5 - Plasma network v2 with sheath closure (RUNNING)

- **Inputs.** The admitted ledger R00-R27 and its `PROPOSED_NOT_ACCEPTED` corrections; rows
  R28-R31 (R1a); `CL-3-potentials` (R1b); per-cell `n_e,k` from `pic2d_cft_steady_state_v2`
  labelled development-tier (R1c), replaced by steady-state v3 when it exists; Kornfeld Table
  3.1 and Puca 2024 Table 1 as reproduction targets (R1d).
- **Outputs.** A new module beside `cft_revival.plasma` (the admitted files stay byte-identical);
  a ledger revision with the new rows and the closure labelled declared; the DM9.2 and Puca
  states re-solved as roots (or the recorded residual if no root exists); the residual of the
  published p-vector under the new rows; a rank analysis showing the potentials identified.
- **Gates.** Structural rank = number of states; residual tolerance met (not a minimum-error
  stop); feasibility (`j_i4 >= 0`, `phi_4 >= Ua` handled by the anode row); the closed-form
  R27 check reproduced on the old manifold; `analytic-consistency` admission with the two
  reproduction targets as documented-not-truth rows.
- **Estimate.** CPU minutes to < 1 h (pure-Python solves at ~3 s each; a few hundred states);
  tests; paper re-admission separately.
- **What would stop it.** No root at either published state under the closed system (recorded
  as the outcome: the model does not reproduce the published states); `n_e,k` sensitivity so
  large that the declared-input label dominates the result (then the neutral / mass-flow
  balance rows move up from network v3).

### 5.6 Step 6 - PIC v1.4 -> steady-state v3 -> >= 50 us case -> breathing decision -> campaign v1 (RUNNING)

- **6a. Steady-state v3 (model v1.4).** Inputs: the v2 protocol as lineage; v1.4 with wall
  recycling (P3a), peak-node Debye gate and grid-heating triad (P1a, P1c), CUDA-graph step
  (P2a), Bohm / SEE hooks (P4b, P4c); the W x 0.7 pair case must finish first (GPU is single;
  budget ends by ~20:35 AEST). Outputs: a plateau (or its absence) at the recycled fixed point,
  net utilisation (P3d), per-cusp fluxes and per-cell `n_e` for step 5, the literature-context
  panel (P6b). Gates: the v2 plateau rule tightened per P5a; peak-node Debye and omega_pe dt
  fail-closed; CUDA-graph path bitwise-equal to the un-captured path on a short run. Estimate:
  GPU 3-5 h (>= 3 transits at the current grid; less if the graph capture removes most of the
  1.2 ms floor). Stops: the recycled fixed point (n_g* ~4.5e19 before S responds) avalanches or
  fails to ignite at the current seed (re-size the operating point first); the graph capture
  fails the bitwise test.
- **6b. >= 50 us development case.** Inputs: v1.4 at reduced W without the artificial relaxation
  (P3c). Outputs: I_d and n_g time series with a spectral check (P5a); the answer to "does the
  closure oscillate". Gates: none binding (development); the spectral check is the decision
  input. Estimate: GPU 10-20 h (review figure). Stops: a WDDM-bound step time that makes 50 us
  exceed 20 h (then the case is shortened to the first breathing period estimate, ~80 us at
  the predator-prey estimate f ~12 kHz, and reported as such).
- **6c. Breathing-mode decision and model v2.0.** If 6b breathes: the campaign observables become
  period-averaged means plus the frequency (Kahnfeld 2018 is then the comparison), the wall
  budget covers >= 3 periods, and the cost is 2-3x the proposal. If not: the plateau criterion
  of P5a is frozen with the closure statement. Either way model v2.0 adds P3b (view-factor
  neutrals), P4a (CEX), P4b (SEE, Hobbs-Wesson cap), P4d (plume extension + current-continuity
  variant) with development runs (proposal section 5: ~20 GPU-h). Stops: a breathing period so
  long that >= 3 periods exceeds the campaign budget (decision point, not a silent scope cut).
- **6d. Preregistered PIC campaign v1.** Inputs: `pic2d-model-v2.0.json`, the frozen protocol
  with blocks A (3 seeds), B (W/2), C (20 um grid), D (exit boundary), E (Bohm bracket), the
  named observables (P6a), the diagnostics (P4e), the claim boundary of the proposal section 2.6
  extended by P4c and P6c. Gates: proposal sections 2.3-2.5 (seed spread, W and grid
  sensitivity, ledger closures) plus P5a, P5d; one-shot through `experiment_runtime` with the
  hygiene of step 8. Estimate: GPU 110-125 h at dt = 1.5 ps (proposal section 5), 2-3x if
  period-averaged. Stops: a W-sensitivity above 10 % on any headline quantity in 6a (fail
  closed: re-size); the 14 h per-case wall budget at dt = 1 ps (decide dt = 1.5 ps and freeze).

### 5.7 Step 7 - External validation v0 -> v1 -> v2 (QUEUED)

- **7a. Validation v0 - code-to-code against Brandt et al. 2016**, labelled model-to-model by
  closure. Inputs: the published geometry (Z_thr 14 mm, R_thr 1.5 mm, 400 V, 0.27 sccm), a
  reconstructed magnet set and field, static neutral background ~2e20 m^-3, their closures
  (Bohm diffusion `D = 0.4 kT_e/eB`, 50 % / 90 % SEE re-emission, wall recycling) switched on
  in `pic2d` through the v1.4 / v2.0 hooks; both codes' stated numerical parameters tabulated
  LANDMARK-style. Outputs: anode electron current (their 4.3 mA), net ionisation (24 %), beam
  current (2.5 mA), the internal-cusp potential drops (~10 V, ~5 V), plume peak angle (50 deg)
  as `E = S - S_ref` with `u_num` from our refinement. Gates: preregistered comparison protocol
  through `cft_revival.validation` v2 contracts; the closure differences declared per row; opens
  no physics level. Estimate: CPU 5-8 d setup; GPU 40-100 h (their 2.4e7 steps at 1024 x 256
  cells are ~100 h at our step time; a shorter run to their quasi-steady state at our grid is
  the cheaper first pass). Stops: the magnet geometry or field not reconstructable from the
  paper (author contact); their 2e20 m^-3 neutral background pushing the peak density beyond
  our resolvable envelope.
- **7b. Validation v1 - ion-energy peak and peak-current angle** against Koch et al. 2011
  (HEMP-T 3050 at 1000 V: peak ~985 eV = Ua - 15 V; peak ion density at 20 deg; negligible
  above 50 deg) and HIT RPA (Hu et al. 2016), in ASME V&V 20 form. Inputs: the campaign v1
  plume-extended case (block D) and the refined-grid case (block C) for `u_num`; `u_input`
  from n_g and Q_in; `u_D` declared (+-5 % of Ua for the peak, +-5 deg for the angle) as an
  assumption. Outputs: `E +- u_val` for the two quantities; the statement that this tests
  same-class behaviour at 300 V on a different geometry, not the device. Gates: preregistered
  before the comparison; GATE-L1 candidate at most. Estimate: 3-5 d after the refined-grid
  plateau; no extra GPU if blocks C and D run in the campaign (otherwise 30-35 GPU-h). Stops:
  no plume region (P4d not landed); `dz/lambda_D = 3` at the peak (the v2 plateau is not
  sufficient for `u_num`).
- **7c. Validation v2 - thrust-voltage curve** against Keller et al. 2015 mu-HEMPT (50 uN at
  600 V, Isp 230 s; 180 and 360 uN at 610 and 860 s) or the low-power HEMP of Liu et al. 2019.
  Inputs: their geometry modelled from published dimensions (Keller: chamber 2-5 mm, SmCo OD
  10-40 mm, cusp length 1-10 mm - ranges, so a specific device needs author contact); the area
  metric across their operating points; one operating point withheld. Outputs: thrust vs Ua at
  fixed flow with the area metric. Estimate: 10-20 d plus data access; GPU 50-100 h (four to
  five operating points, each a full plateau run). Stops: no device-specific dimensions; each
  operating point needs its own resolvable-density budget.

### 5.8 Step 8 - Preregistration hygiene (applies from step 2 onward)

- External registry receipt for every `preregister ...` commit (or a signed tag mirrored to a
  second host), recorded in `authorities.json` (S9a); RO-Crate packaging of accepted bundles at
  the export stage (S9b); `gate_genealogy` block (S9c); shakedowns on synthetic or permanently
  excluded designs with every outcome logged and method-selection rules frozen (S9d); ADEMP
  fields (S9e). Gates: `prepare` refuses a protocol without the genealogy and ADEMP blocks; the
  export stage refuses without the registry receipt. Estimate: 1-2 d once (runtime + schema),
  then minutes per campaign. Stops: none material; the registry choice (a public registry or a
  second signed-tag host) is a user decision.

### 5.9 Deferred items and their triggers

| id | item | trigger |
| --- | --- | --- |
| D-1 | P1e energy-conserving deposition / gather switch | campaign v1 recorded; the snapshot-v2 coarse case is the test bed |
| D-2 | P2b periodic cell sort | a step-time profile after CUDA-graph capture shows gather / deposit memory-bound |
| D-3 | P2c TCC / Linux host | hardware access |
| D-4 | R2c = S7 closure calibration from PIC per-cusp fluxes with a second-operating-point re-check | steady-state v3 recorded (6a) and one more operating point available |
| D-5 | R3b iron-spacer sensitivity of `z_c,k` on the P2 representatives | material-aware field qualification (ROADMAP_AUDIT section 4 item 8) |
| D-6 | S8d validation v3 geometry trends vs HIT | MDO v3 with a calibrated closure (after D-4) |

## 6. Corrections to our own prior statements

These correct statements made in this project's own briefs, planning notes and interpretations.
None of them changes a recorded result; each changes how a recorded result may be described.

### 6.1 The "downstream AST 2024" paper is a FEEP paper

The DOI `10.1016/j.ast.2024.109516`, cited in our task briefs as a downstream cusped-field
optimisation paper ("Ma et al. 2024, Aerospace Science and Technology"), resolves to Yeo, Gadisa,
Ogawa and Bang, "Multi-objective design optimization and physics-based sensitivity analysis of
field emission electric propulsion for CubeSat platforms", *Aerospace Science and Technology*
154, 109516 (2024): a **field-emission electric propulsion** paper by the same group. All three
reviews resolved it independently (PIC review section 0; reduced-models review section 0 and
entry 73; surrogate review section 0.2). No 2024 AST cusped-field paper by "Ma" was found. The
real downstream lineage of the ISTS 2017 study is Fahey, Muffatti and Ogawa 2017 (*Aerospace*
4, 55) -> Yeo, Ogawa, Matthias, Kahnfeld and Schneider 2020 (*JSR* 57, 603) -> Yeo and Ogawa
2022 (*JPP* 38, 973), with Yeo and Ogawa 2023 (Monte-Carlo UQ) and, independently, Puca,
Panelli and Battista 2024 (*Aerotecnica Missili & Spazio* 103, 321). The wrong attribution
lived only in briefs and session notes; no committed document or manuscript sentence cites the
DOI as a cusped-field paper (checked with `rg` over `paper/`, `modern/docs/`, `.cursor/` at
`b6bb6215`). Rule going forward: resolve every DOI through Crossref before writing the sentence
that cites it.

### 6.2 There is no TU Berlin HEMP dataset

Briefs referred to a "TU Berlin low-power HEMP" dataset as a validation target. Two searches
found no TU Berlin HEMP publication. The German low-power / micro-HEMPT work is
Giessen / Airbus / ZARM / DLR: Keller et al. 2015 (*IEEE Trans. Plasma Sci.* 43, 45: 50-360 uN,
Isp 230-860 s, geometry systematically varied) and Hey et al. 2015 (direct thrust on a
double-pendulum balance). Validation v2 (section 5.7) targets those, and the Thales HEMP-T
data (Kornfeld 2007; Koch 2011) for the RPA comparison of validation v1.

### 6.3 The topology nulls were nulls of a non-standard definition

`four_cell_topology_search_v2` (0/128 stable) and `cft_topology_characterization_v1` (0 stable
eligible cusps / 0 cells over 56 designs) stand as recorded: preregistered nulls under their
frozen definitions, admitted as such. What changes is the interpretation we attached to them.
The frozen definition required an interior **vector null** of `psi` with X-type Jacobian,
geometry-registered at the stage midplanes near the wall. In the (r, z) half-plane of an
axisymmetric alternating ring stack the generic vector nulls lie **on the axis** (Gildea 2012,
sections 1.1 and 1.4: "the field strength only goes to zero where the ring cusp separatrices
intersect the thruster axis of symmetry"); the wall "ring cusp" is where the separatrix
`psi = psi(0, z_k)` meets the wall and `B_z` changes sign while `|B_r|` is maximal - it is not
a null. Koch 2011 and Lewerentz and Schneider 2023 place the cusps at the magnet ends, not the
stage midplanes. Our search therefore looked for an object the standard PPM topology does not
contain, and found the axis nulls (3-5 per design) that the literature calls the point cusps
but treated them as descriptors. The nulls are consistent with the literature topology; they
are not evidence against multi-cell confinement. Topology v3 (step 1) tests the literature
definition; the paper amendments of section 7 follow when it records.

### 6.4 Our quoted utilisation is gross, not net, and is inflated without wall recycling

The steady-state v2 plateau is reported as "utilisation 46 %" (S = 3.93e16 atoms/s against
Q_in = 8.55e16 atoms/s). That is gross ionisation over feed. In the same plateau 3.72 mA of
Xe+ (2.32e16 ions/s, 59 % of the 6.30 mA-equivalent ionisation) is absorbed by the dielectric
wall and removed from the atom inventory; every kinetic-neutral thruster PIC (Szabo 2001;
Brandt 2016) returns those ions to the gas as thermal neutrals and quotes **net** ionisation.
On the literature's definition the same plateau gives `(S - R_wall)/Q_in` ~ 18 % (or
`I_beam/(e Q_in)` = 2.29 mA / 13.7 mA ~ 17 %), not 46 %; Brandt 2016's 24 % (measured 25 %)
is a net figure. The number itself is provisional: with recycling in the inventory the fixed
point moves (n_g* from 2.97e19 to ~4.5e19 m^-3 before S responds), so no utilisation figure
from the current model should be compared with any published value until steady-state v3 runs
(P3a, P3d). Until then every quotation of 46 % must carry "gross, no wall recycling".

### 6.5 "Geometric access fraction" is the right name for the test-particle wall-hit quantity

The wall-loss v4 estimand (`electron_dielectric_wall_loss_probability` 0.6445) and the
screening v1 labels (P(wall) 0.375-0.869) are fractions of launched **collisionless** electrons
whose orbit intersects the dielectric before leaving the domain in a prescribed field with no
electric field, no sheath and no collisions. No published closure uses such a quantity as a cusp
loss probability; the literature's `p_k` is either a mirror-formula geometric fraction with mu
assumed conserved, a PIC-derived electron current, or a hybrid-gyroradius leak-width flux with
a Boltzmann sheath factor. A floating dielectric repels most electrons in the loss cone
(`exp(-Delta phi_s/T_e)` ~ 7e-3 at 5 T_e), so 0.64 is not a loss fraction of any real
population; it is an upper bound on sheath-free geometric access. The correct name is
**collisionless geometric wall-access fraction** (R2a). MDO v2's `closure_identification_disclosure`
already says the quantity is not the Kornfeld per-cusp probability; the relabel makes the
quantity's own name say so. Recorded bundles are not edited; the coupling schema description
changes at the next coupling revision, new protocols use the new name from screening v2 on,
and the manuscript adopts it at its next admission (section 7).

## 7. Paper amendments required (list only; the paper is not edited here)

Each amendment is tied to the recorded evidence that triggers it. Until that evidence exists the
current sentences stand as recorded; they are correct under their definitions and labels.

Trigger A - topology v3 records (step 1):

1. Discussion, paragraph "The assumed multi-cell wall-cusp topology is undemonstrated in the
   screened design space", after CLM-028: add that the frozen definition (a wall-side X-type
   null at the stage midplanes) is not the literature's cusp definition (axis null plus
   separatrix to the wall; Gildea 2012, Lewerentz and Schneider 2023, Kornfeld 2007), and
   state topology v3's recorded outcome under the literature definition with its own gate.
2. Same paragraph, "The accepted sweep does not contradict this: its axis cusps are sampled
   sign changes of the on-axis field, one per magnet stage, and the sweep's own QoI policy
   labels them descriptors rather than a critical-point proof": add that under the literature
   definition those axis nulls are the point cusps whose separatrices bound the cells, so the
   sweep's descriptors are the objects topology v3 characterises.
3. Same paragraph, "Whether a stable multi-cell wall-cusp topology exists under a material-aware
   field model, or under a different cusp and cell definition, remains a question that only an
   accepted field-resolved manifest (GATE-L1) can settle": split the two questions - the
   definition question is settled by topology v3 (a numerical-screening gate, not GATE-L1);
   the material question remains open.
4. Limitations, "The admitted topology-screening studies use linear-vacuum equivalent-current
   fields with a single mesh per design and frozen cusp and cell definitions; their nulls hold
   under those definitions and that field model only": add "and those definitions differ from
   the literature's; see the topology v3 admission".
5. CLM-044 (Discussion), "its cells were never shown to exist in the screened field space":
   qualify with "under the frozen wall-null definition"; if topology v3 finds separatrix-bounded
   cells, the sentence becomes "its cells, under the literature definition, exist in N of 96
   designs, and the balance still has no solution for the probabilities the legacy chain fed
   it".

Trigger B - relabel (R2a), at the next admission that touches Sections 7, 9 or 11:

6. Section 7 and CLM-012...CLM-017 prose: "wall-loss probability" / "wall-hit probability" ->
   "collisionless geometric wall-access fraction" wherever the estimand is named in prose (macro
   names and recorded artifact fields are unchanged); the Limitations sentence "its pooled
   fraction is a design average rather than a loss rate" gains "and a geometric access
   fraction, not a loss probability".
7. CLM-035 third reading, "the collisionless wall-hit probability of the wall-loss campaign and
   the per-cusp loss probability of the cusp cascade are different quantities": rename the first
   quantity; the statement itself is confirmed by the reduced-models review (no published
   closure uses it as `p_k`).
8. Abstract and Section 11 (`\WlgWallPMin` ... "per-design wall-hit probability"): rename in
   prose at the next regeneration of the section.

Trigger C - plasma network v2 records (step 5):

9. Limitations, "the correction stays unaccepted, the model has no potential closure": replace
   with the network v2 state (sheath rows R28-R31 and a **declared** `CL-3-potentials`), and
   state the reproduction outcome against Kornfeld Table 3.1 and Puca 2024 Table 1 as
   documented-not-truth rows in a new `analytic-consistency` admission; the admitted
   `cft_revival.plasma` blobs at `266d8a99` are not touched.
10. CLM-044, "the balance that turned those probabilities into performance has no solution for
    them": add that the closed system (v2) is the first version in which the potentials are
    identified, and whether it has a root at the published states is the recorded outcome of
    that admission.

Trigger D - PIC v1.4 steady-state v3 records (6a) and any first PIC admission:

11. Any sentence quoting the steady-state v2 plateau (none is in the manuscript today; the
    dashboard and devlogs carry it): utilisation must be quoted net of wall recycling with the
    closure named, and the plateau labelled conditional on the neutral closure (P3d, P5a).
12. Limitations, "No measured-thruster comparison, geometric field response, plasma chemistry,
    sheath or secondary-emission wall model, cathode model, thermal model, erosion model, or
    facility correction is active": when a PIC section is admitted, this sentence must be
    re-scoped to the L0 result and a PIC-specific limitations block added (no anomalous
    transport by construction; Bohm block is a bracket; plateau conditional on the neutral
    closure; `dz/lambda_D` at the peak; operating point far below a real CFT).
13. Section "Planned L3 result: PIC and experimental comparison": when validation v0 records,
    add that the first comparison is code-to-code against Brandt et al. 2016, labelled
    model-to-model by closure, opening no physics level; when v1 records, the RPA comparison
    against Koch et al. 2011 in ASME V&V 20 form with the declared `u_D`.

Trigger E - claim-matrix structure (surrogate review sections 3.5 and 4.3; not a summary-table
row, listed because it is a paper change):

14. Score every admitted gate on the NASA-STD-7009A factors (verification, validation, input
    pedigree, results uncertainty, results robustness) in the claim matrix; the present state is
    verification 3-4, validation 0, input pedigree 1, and the validation factor at 0 then
    appears in every table rather than in a footnote.

## 8. Honest gaps of this synthesis

- The three launched streams (section 0.3) had zero commits at 18:10 AEST; their state is taken
  from the launch briefs, not from artifacts. The tracker rows for them carry RUNNING and no
  rung moves until a commit exists.
- GPU / CPU figures are planning estimates from recorded rates under contention; every protocol
  must re-measure in its shakedown (surrogate review section 6).
- The two-stage allocation rule of step 2 is this document's proposal for reconciling "replicate
  non-saturated cells" with "cells from the v3 catalogue"; the screening v2 protocol may choose
  a different frozen rule, but it must be frozen before stage-1 counts are read.
- Validation v0's cost depends on whether Brandt et al. 2016 specify their magnet set well
  enough to rebuild the field; this was not checked here beyond the review's reading.
- Nothing in this document changes the recorded status of any experiment, gate or claim.
