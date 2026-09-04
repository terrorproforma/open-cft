# pic2d design mini-sweep v1 - DRAFT (not preregistered, no production launch)

**Status: DRAFT / preparation only.** Nothing here is evidence. The preregistration commit follows the
operating-point / grid decision that the plume attempt-8 verdict now forces: attempt 8 (PID 51256, model
v2.0.1) was stopped by its grid-heating triad gate at 4.98 us with NO plateau (S drift 0.253; record
`ac248e05`, 11:57 AEST 2026-09-04) - finite-grid heating from ~2.4 us once the peak-node Delta/lambda_D crossed
~3.2 (the CIC threshold pi), while the accepted channel-only plateau sat at 3.17 with the energy residual
closing to +0.4 %. The recorded resolution decision (Delta <= 32.4 um, dt <= 1.48 ps; v2.1 not launched;
recalibrate the peak-Debye gate to the CIC threshold + a windowed residual-power gate; then a channel-only
33 um / 1.4 ps refinement run or a plume run at a lower operating point) is what the sweep's grid and gates
must inherit. What IS done here: the design selection with the catalogue numbers, the four new material-aware
P2 fields (hash-bound, gated), the closure-target definition and its map onto plasma-network v2, the cost
table (50 um / 1.5 ps and the 33 um / 1.4 ps refinement variant) with a recommendation, a per-design protocol
composer that reuses the accepted steady-state / plume runner (grid / dt overridable), a whole-set preflight
over all five designs for the channel-only and the 24 mm plume options (both green), and tests.

Why this sweep: the roadmap's L0 closure needs a design-dependent cusp-loss law. The collisionless labels
saturated (screening v2: all 181 interior cells at P(wall) = 1), so surrogate v3 / MDO v3 on those labels
would contradict CLM-085; the next fidelity for the label is the PIC-MCC model, run on a few catalogue
designs that span Koch rho.

## 1. Design selection

Four primary designs with the SAME wall-cusp count as the reference (3 wall cusps -> 4 cells: a 1:1 map onto the
four-cell plasma network v2, model cusp 1 = no magnetic counterpart, as in `plasma_v2.pic_context`) spanning
rho_conservative from 0.38 to 2.4, plus the strongest four-cusp HEMP-like design as an optional fifth (it fills all
four Kornfeld probability slots p_1..p_4 but changes the cusp count as well as rho). Candidates were screened for
what the PIC mesher represents exactly (straight bore + linear exit cone on a 50 um grid) and for the L1b sliver
lesson (028's 0.254 mm taper, 048's 0.045 mm injector-magnet gap): every selected design has either no exit taper
or a long one (1.68 mm), and the whole-set level-0 mesh preflight passed at 22.7-32.4 deg (zero elements below the
qualification's 10 deg; the L1b gate is 5 deg).

| # | design | role | stages | cusps / cells | r_w mm | L mm | pitch mm | r_w/pitch | exit taper | rho_cons catalogue field | rho_cons under iron (this work, level 0) | rho_cons L1b v1.1 (level 1) | wall B at cusps, T (catalogue -> iron) | cusp z, mm (catalogue -> iron) | screening-v2 P(wall) per cell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `divergent-exit-stack` | reference (every pic2d run so far) | 4 | 3 / 4 | 2.000 | 24.0 (straight 1-18, cone to 3.0) | 6.00 | 0.33 | 6 mm | 0.596 / 0.610 / 0.618 (P2 level-1, material-aware) | - (existing field) | - | 0.1077 / 0.1063 / 0.1077 | 6.028 / 12.000 / 17.972 | 0.605 / 1.00 / 1.00 / 0.17 |
| 2 | `l1a-gs-v3-056-effcbc8686` | HEMP-like | 4 | 3 / 4 | 3.822 | 17.05 | 4.26 | 0.90 | none | 1.993 (L1a vacuum) | **2.357** | 2.372 | 0.215 / 0.209 / 0.215 -> 0.271 / 0.262 / 0.271 | 4.24 / 8.52 / 12.81 -> 4.31 / 8.53 / 12.74 | not screened |
| 3 | `l1a-gs-v2-047-e3196a8aa5` | low rho | 4 | 3 / 4 (+1 anode-edge boundary cusp under iron, see below) | 1.567 | 25.92 (straight to 24.24, 1.68 mm cone to 2.199) | 6.48 | 0.24 | 1.68 mm | 0.349 (L1a vacuum) | **0.377** (interior cusps) | - | 0.087 / 0.092 / 0.087 -> 0.081 / 0.076 / 0.081 | 6.31 / 12.96 / 19.61 -> 6.61 / 12.96 / 19.31 | 1.00 / 1.00 / 1.00 / 0.50 |
| 4 | `l1a-gs-v3-009-d0c686b4aa` | mid rho | 4 | 3 / 4 | 2.681 | 22.79 | 5.70 | 0.47 | none | 0.899 (L1a vacuum) | **0.924** | - | 0.061 / 0.064 / 0.061 -> 0.065 / 0.064 / 0.065 | 5.55 / 11.39 / 17.25 -> 5.74 / 11.40 / 17.05 | not screened (sobol_v3) |
| 5 (optional) | `l1a-gs-v3-106-ccec1c8b2f` | HEMP-like, four cusps | 5 | 4 / 5 | 4.102 | 20.18 | 4.04 | 1.02 | none | 2.560 (L1a vacuum) | **2.925** | 2.928 | 0.177 / 0.168 / 0.168 / 0.179 -> 0.232 / 0.231 / 0.231 / 0.232 | 3.95 / 8.27 / 11.92 / 16.25 -> 4.08 / 8.05 / 12.14 / 16.11 | not screened |

Sources: reference cusps/cells from the cusp topology search v3.1 catalogue (`p2_divergent_exit`, P2 level-1; rho
formed as wall |B| / max adjacent axis peak, the sweep-v3 descriptor definition); designs 2-5 from the sweep-v3
catalogue (`cusp-cell-catalogue-v3.json`, byte-checked against the sweep-v3 manifest); L1b numbers from
`l1b_hemp_confirmation_v1_1/results` (design records); P(wall) from the screening-v2 dataset (final, topped-up
cells). The "under iron" columns are this experiment's own level-0 solves (section 2) with the v3.1 definition
applied verbatim.

Why these and not others: the sweep-v2 minimum rho (0.197, design 088) has a cusp at z = 0.16 mm and a 3-stage /
4-cusp topology; the other low-rho sweep-v2 designs with 3 cusps (061, rho 0.38) carry a 5.9 mm taper; the mid-rho
sobol_v3 field between 0.85 and 1.5 has only two 3-cusp designs without a taper (009 at 0.90 and 033 at 1.04; 033
has a 0.34 mm steep taper - the 028 sliver pattern - and was not chosen); the HEMP-like 15 offer two taper-free
designs with strong wall fields (056 with 3 cusps, 106 with 4) whose L1b level-0 meshes are the cleanest of the
set (33.6 / 30.8 deg). Design 005 (5 stages, 4 cusps, rho 1.64-1.74, wall |B| 0.046 T) is the "just HEMP-like"
alternative if a point near the 1.5 threshold is preferred to 106.

Finding recorded during field production (design 047): under linear iron the anode-side axis null moves from
-1.40 mm (L1a) to -0.11 mm and its separatrix reaches the straight dielectric 0.073 mm from the anode plane (13 deg
to the wall normal). Under the v3.1 boundary-ambiguity tolerance (0.25 mm; the L1b v1.1 GATE (b) boundary-tolerant
count) it is a boundary classification, not a cell boundary: the three interior cusps match L1a within 0.30 mm
(tolerance 0.45 mm) and the interior rho is 0.377-0.385. In the PIC this 73 um region is the anode sheath. The
strict count differs (4 vs 3) and is disclosed in the binding; the substitute if the preregistration decides it
disqualifies 047 is `l1a-gs-v2-061-d8ebe65ef1` (rho 0.381, 3 cusps, 5.9 mm taper, wall |B| 0.12 T).

What four points can and cannot discriminate: they can test a monotone trend of the per-cusp transit-loss
probability and of the ion wall-loss fraction with rho over a 6x span (0.38 -> 0.61 -> 0.92 -> 2.36), and whether
the HEMP-like regime (rho >= 1.5, bracketed by 0.92 and 2.36) shows a qualitative change (cusp-confined electrons,
a potential staircase with steps at the cusps); with the optional fifth they can test whether a fourth cusp adds a
fourth loss channel of the same law. They cannot separate rho from its confounders along this one-dimensional cut
of an 11-variable design space (r_w 1.6-4.1 mm, L 17-26 mm, pitch 4.0-6.5 mm, wall |B| 0.065-0.27 T change with
it), cannot fit more than a two-parameter closure without over-fitting, cannot resolve non-monotonicity between
points, and cannot separate the iron effect from the geometry (the label is rho under iron; L1a rho is 0.93-0.96x
of it here). A design effect counts only if it exceeds the seed replicate spread (section 3).

## 2. Fields

What the PIC needs: a bilinear node field (B_r, B_z) on the PIC grid over the plasma nodes of the channel (v1.x) or
the L-shaped channel + plume box (v2.0 / v2.1), evaluated directly from a hash-bound quadratic A_phi FEM solution
whose mesh contains the whole box (`cft_revival.pic2d.fields`; the v2.1 extension binds the `domain-padding-1.5`
solve of the reference through `spec/pic2d/p2-field-plume-extension-v2.json`).

What exists: for the reference design everything (authority level-1 checkpoint for the channel, padding-1.5 for
any box up to 60 x 48 mm). For every other design NOTHING usable: the L1b v1.1 solutions were kept only as bore
samples (33 x ~230 nodes, r <= r_w, `results/artifacts/fields/*.json.gz`) of a padding-0.5 domain that ends 5-10 mm
behind the exit (e.g. 056: z <= 25.8 mm, 106: z <= 30.5 mm); they cannot serve a plume box and carry no element
data. The L1a fields are linear-vacuum equivalent-current maps (not material-aware).

New solves (CPU, sequential, BLAS single-threaded, `fields.produce_field`): fem_reference graded body-fitted level-0
mesh (bore r_w / 8, features / 4 - the qualification's domain-study setting), the L1b v1.1 materials (linear
soft-iron poles and return yoke mu_r 4000, SmCo-like recoil mu_r 1.05 + remanence, BN / Al / Cu at mu_r 1),
smallest ladder padding whose FEM box covers z <= L + 24 mm and r <= 12 mm with a 0.75 mm truncation margin,
Jacobi-PCG to relative true residual 2e-10, checkpoint bundle written with `fem_reference.write_checkpoint_bundle`
(the format `BoundP2Evaluator` reads), B post-scaled by the design's L1a `source_strength_scale` (the catalogue's
magnet strength; linear problem, exact - the L1b convention).

| design | padding | FEM box (r <= / z in) mm | P2 DOFs / triangles | min angle deg | PCG iterations / residual | solve s / total s / peak RSS MB | checkpoint (`fields/<id>/`) | file sha256 (12) | payload / mesh / run / sidecar (12) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 056 | 1.50 | 37.44 / [-26.83, 43.38] | 345,883 / 172,302 | 31.68 | 3076 / 1.98e-10 | 133 / 306 / 461 | `l1a-gs-v3-056-effcbc8686.domain-padding-1.50.level-0.json` (+ `.arrays.npz`) | `19adb8543001` | `54ea525bde24` / `033826a9831c` / `d38b2951e672` / `55b2b4b817b6` |
| 047 | 1.00 | 35.29 / [-26.92, 52.33] | 378,893 / 188,736 | 22.72 | 3628 / 1.99e-10 | 159 / 398 / 503 | `l1a-gs-v2-047-e3196a8aa5.domain-padding-1.00.level-0.json` | `dafa822adcc2` | `46d90de25d6d` / `fc8085ecbb8f` / `1230e90bfd81` / `1bd353a389dc` |
| 009 | 1.25 | 38.50 / [-29.61, 51.90] | 278,379 / 138,600 | 29.55 | 2881 / 1.98e-10 | 107 / 260 / 386 | `l1a-gs-v3-009-d0c686b4aa.domain-padding-1.25.level-0.json` | `651aa7662bd4` | `9fe44495770b` / `7a6b87904d7c` / `e00273822381` / `53ce83673083` |
| 106 | 1.25 | 37.48 / [-26.35, 46.04] | 314,995 / 156,884 | 32.37 | 3127 / 1.99e-10 | 116 / 291 / 431 | `l1a-gs-v3-106-ccec1c8b2f.domain-padding-1.25.level-0.json` | `99f9dc0518b6` | `cc8b69bf4678` / `edc8e88e3ef5` / `b363d68a2abe` / `0013cead3b8a` |

Full hashes and the supported PIC box per design are in `fields/<design_id>/binding.json` (schema
`cft.pic2d.design-mini-sweep.field-binding.v1`); the reference's `binding.json` names its existing spec files
(authority v1, plume extension v1 / v2) with their file hashes. The checkpoints are Git LFS objects
(`.gitattributes` in this directory); `fields.verify_binding` re-hashes them and `design_field_map` refuses any drift.

Gates per field (all passed, recomputable from the bound checkpoint without a solve: `fields.regate_field`):

* mesh angle >= 5 deg (whole set BEFORE any solve: 22.7 / 29.6 / 31.7 / 32.4 deg, zero elements below 10 deg);
* PCG converged, relative true residual <= 2e-10; FEM box covers the v2.1-sized plume box for the design;
* topology agreement (v3.1 definition on the scaled bore sample, 32 radial intervals, the sealed L1a axis window):
  boundary-tolerant cusp count equal to the catalogue (4/4 designs; strict count 3/4, the 047 anode-edge cusp), every
  cusp within max(r_w / 8, L1a dz) of its catalogue position - shifts 0.076 / 0.302 / 0.194 / 0.223 mm against
  tolerances 0.478 / 0.451 / 0.451 / 0.513 mm; all traces terminate cleanly;
* L1b v1.1 accepted-map agreement where it exists (056, 106): node-wise |dB| against the level-1 padding-0.5 bore map
  at equal scale, max 1.51 / 0.95 mT in the channel (rms 0.63 / 0.17 mT) at max |B| 0.32 / 0.31 T, gate 20 mT (the
  authority's component bound); rho under iron 2.357 / 2.925 vs L1b 2.372 / 2.928 (0.6 / 0.1 %).

Reference design (no new solve): the channel map is the authority bicubic (`build_p2_psi_field`), the 12 mm box the
v1 extension, the 24 mm box the v2 extension, exactly as the accepted runs; the channel cross-check (0.74 mT for
padding-1.5 vs level-1) runs at map build.

## 3. Closure targets (what each run measures and where it enters plasma-network v2)

Module: `cft_revival.plasma_v2` (`models.py`: `SheathClosureInputs`, `CuspSheathSpec`, `PotentialClosure`,
`SheathClosureState`; `pic_context.py`: the reference-design extraction this sweep generalises). Per design, from
the window-averaged `maps.npz` + `summary.json` of a run that satisfied the plateau rule (`closure.extract_targets`,
`run.py targets`):

| target | per | PIC observable | plasma-network v2 parameter | role |
| --- | --- | --- | --- | --- |
| cusp transit-loss probability p_k | cusp | L_k / je_arriving,k from the Kornfeld chain je_k = je_{k-1} - L_k + I_k seeded with the electron current entering the channel (injected - returned) | `declared_cusp_probabilities` (CL-1), `anode_cusp_probability` (anode-most cusp) | calibration |
| cusp electron / ion wall currents L_k, cusp ion current | cusp | e x sum over the cusp window (half-width min(1 mm, pitch/4)) of the wall flux x wall area | L_k in R23-R26 / R35-R37; `PlasmaState.cusp_ion_current_a` | calibration / reproduction |
| effective cusp-loss coefficient | cusp | L_k / (e N_e,cell) - the electron loss frequency the model implies | none explicit (enters only through p_k): the cross-field transport disclosure | disclosure |
| cusp leak width | cusp | FWHM of the wall electron flux about z_c | `leak_width_prefactor` (CL-4: width / hybrid gyroradius at B_w,k) | calibration |
| sheath drop and near-wall T_e | cusp | phi(axis) - phi(wall), phi(wall - 3 dr) - phi(wall); density-weighted T_e in the last 0.5 mm | `sheath_drop_v`, c_s,k = drop / T_k -> `CuspSheathSpec.area_ratio` via ln(K0 rho_k) and the regime | calibration |
| cusp wall field, cusp electron density | cusp | |B|(r_w, z_c) of the bound map; near-wall n_e | `CuspSheathSpec.wall_field_t`, `electron_density_per_m3` (CL-4) | input / calibration |
| ionisation share S_k / S | cell | volume integral of `ionization_rate_per_m3_s` per catalogue cell (the renderer v0.2 "flames" at the cusp planes) | `PlasmaState.ionization_source_current_a` (e S_k = I_k) | reproduction |
| ion wall-loss fraction | cell | e-weighted wall ion current of the cell / e S_k (the quantity the saturated screening label could not resolve) | closes ji_k in R06-R11 | calibration |
| cell potential and T_e | cell | n_e-weighted phi, T_e; steps between cells | `plasma_potential_v`; `PotentialClosure.interior_step_3_v / _4_v` (CL-3-potentials, the PIC staircase) | calibration |
| I_d, S, utilisation (gross / net), anode ion fraction | design | window currents, v1.4 neutral ledger | `anode_current_a`, sum I_k, beam ji_1 / e Q_in, anode row R31 | reproduction / calibration |
| I_beam, divergence half-angles, thrust | plume | far-field crossings, `plume_ion_current_per_sr_a`, momentum ledger | ji_1; none for divergence / thrust (disclosure) | reproduction / disclosure |

Kornfeld mapping (`closure.kornfeld_mapping`): cusps numbered from the exit; 3 wall cusps -> model cusps 2, 3 and
p_4 (cusp 1 has no magnetic counterpart, the `pic_context` convention); 4 wall cusps -> all four slots; cell 1 =
exit-side partial cell + cone (+ plume), cell N = anode-side partial cell. Non-cusp (diffuse) wall electron
current is reported separately - the four-cell model has no slot for it and its size is itself a finding.

Plateau / acceptance rule (the accepted steady-state runs, verbatim): relative drift < 5 % over the trailing 20 % of
the elapsed simulated time for I_d, N_e and n_g (linear fit, drift = slope x window / |mean|), evaluated at every
checkpoint, only after >= 3 ion transit times, grid-heating triad inside its soft bounds; fail-closed gates as the
template (omega_pe dt, peak-node Debye 4.5 cells / lambda_D, triad hard bounds after one transit, Courant, Poisson
residual, neutral ledger; plume: cathode connectivity at launch, ignition gate, v2.0.2 plume-boundary gate). Transit
time a priori: 2.4 us x L / 24 mm (the measured reference residence scaled with the channel length) + L_plume /
17 km/s; the measured N_i / L residence supersedes it. Every closure target is the trailing-window mean with its
block standard error; a target is quotable only from a run whose plateau rule held.

Replication policy (affordable): one base run per design (seed 20260903); one seed replicate (20260904) of the
HEMP-like design 056; the reference's existing seed-b / W x 0.7 pair (<= 1.1 % / <= 5.7 % on the plateau quantities)
is the reference's replication statement. Decision rule: a design effect is reported only if it exceeds both the 056
seed spread and the reference's seed / W spread. No W replicate per design in this budget (W = 6e4 throughout;
the composer scales W only above a projected 8 M particles, which no channel-only run reaches).

## 4. Cost and schedule (projection; `cost.py`, `run.py cost`)

Model: ms/step = fixed(grid) + 0.733 ms per M particles, fixed = 2 (nr+1) launches x 5 us + inverse-block reads
(nr+1) x 2 (nz+1)^2 x 8 B at 1.6 TB/s + node kernels; anchors: channel 61 x 481 at 2.0 M -> 1.98 ms (steady-state
v2 base, 5.12 M steps in 2.8 h) and 2.7 M -> 2.44 (W x 0.7); plume 241 x 721 at 4.45 M -> 7.08 (attempt 8). The
model reproduces the plume anchor and over-predicts the channel anchors by 13 % (kept, conservative). Particles:
reference plateau count scaled with the channel volume at equal W and mean density; plume options carry the v2.0
model's 1.75x channel count (attempt 8) plus the measured plume population. Reference rows reproduce the v2.1 spec
(8.25 vs 8.22 ms/step, 7.62 M steps, 17.7 vs 17.4 h).

| design | (a) channel: nodes / N / ms/step / transit / steps / hours / GB | (b) 12 mm box | (c) 24 mm box |
| --- | --- | --- | --- |
| reference | 61x481 / 2.0 M / 2.24 / 2.40 us / 4.8 M / **3.0 h** / 4.2 | 241x721 / 4.5 M / 7.1 / 3.11 us / 6.2 M / 12.4 h / 8.4 | 241x961 / 4.7 M / 8.3 / 3.81 us / 7.6 M / 17.7 h / 9.3 |
| 056 | 77x342 / 4.5 M / 4.2 / 1.71 us / 3.4 M / **4.0 h** / 7.5 | 222x582 / 10.2 M / 10.6 / 2.41 us / 4.8 M / 14.2 h / 15.4 | 222x822 / 10.6 M / 11.6 / 3.12 us / 6.2 M (7.2 M at dt 1.3 ps) / 20.2 h (23.3) / 16.4 |
| 047 | 45x519 / 1.2 M / 1.47 / 2.59 us / 5.2 M / **2.1 h** / 3.1 | 176x759 / 2.7 M / 4.9 / 3.30 us / 6.6 M / 9.0 h / 5.8 | 176x999 / 2.8 M / 5.7 / 4.00 us / 8.0 M / 12.9 h / 6.5 |
| 009 | 55x457 / 3.0 M / 2.87 / 2.28 us / 4.6 M / **3.6 h** / 5.5 | 190x697 / 6.7 M / 7.9 / 2.98 us / 6.0 M / 13.1 h / 11.0 | 190x937 / 6.9 M / 8.8 / 3.69 us / 7.4 M / 18.2 h / 11.9 |
| 106 (optional) | 83x405 / 6.2 M / 5.5 / 2.02 us / 4.0 M / **6.2 h** / 9.6 | 233x645 / 13.9 M / 13.6 / 2.72 us / 5.5 M / 20.7 h / 20.3 | 233x885 / 14.4 M / 14.9 / 3.43 us / 6.9 M / 28.5 h / 21.7 |
| serial, 4 primary | **12.7 h** | 48.7 h | 69.0 h |
| serial, all 5 | 18.9 h | 69.3 h | 97.5 h |

Grid-refinement variant of the attempt-8 verdict (channel-only, 33.3 um / 1.4 ps; `run.py protocol --grid 33um`):
reference 91x722 nodes, 2.91 ms/step, 5.14 M steps, **4.2 h**; 056 116x513, 4.84, 3.65 M, 4.9 h; 047 67x779, 1.99,
5.55 M, 3.1 h; 009 82x685, 3.44, 4.88 M, 4.7 h; 106 124x607, 6.29, 4.33 M, 7.6 h - **16.9 h serial for the four
primary, 24.5 h with the fifth** (particles per cell fall 2.25x at equal W; a W x 0.7 replicate is the W-sensitivity
check at this grid). The attempt-8 record projects the reference's refinement run itself at ~6-14 h.

Host factorisation per launch/resume: 0.2-0.4 min channel (0.9-1.9 min at 33 um), 2.4-5 min 12 mm, 6.8-11.8 min
24 mm (one at a time). Wall budgets in the composed protocols are 1.25x the projection (channel 50 um: 3.8 / 5.0 /
2.7 / 4.7 / 7.8 h).

Recommendation: run the closure sweep CHANNEL-ONLY (option a, model v1.4 physics = the v1.3 accepted plateau +
recycling / peak gate / triad, exit-plane injection) at the grid the attempt-8 follow-up validates - 50 um / 1.5 ps
(12.7 h serial for the four primary designs, 18.9 h with the fifth) if the recalibrated peak-Debye gate admits the
design's peak density, else the 33 um / 1.4 ps variant (16.9 / 24.5 h) - plus the 056 seed replicate (+4.0 / 4.9 h),
and ONE plume run (option c, v2.1, 24 mm box) for the reference design ONLY once an operating point / grid that
resolves the peak exists (attempt 8 did not; at 33 um the 24 mm box is 47.5 h by the attempt-8 record). Total ~21-29 h
of GPU for the channel sweep (`protocol.json` recommended_schedule / refined_grid_schedule). The channel-only
option is also where the accepted plateaus and the closed energy residual live. What is lost by not running the
plume box for the sweep designs: the design dependence of I_beam, divergence and thrust, the cathode-coupling
potential phi_1 per design, and the exit-plane potential structure under a self-consistent far field (the
channel-only exit plane is a 0 V Dirichlet boundary, which forced the acceleration drop inside the channel in
v1.x) - none of which the four-cell closure needs first; the per-design plume runs (48.7-69.0 h at 50 um) are the
natural follow-up once the channel closure is calibrated. Option (b) buys nothing over (c) for the closure and
carries the v2.0 far-field truncation (15 % at 36 mm).

Preflight findings the cost table must carry: for the 24 mm box, design 056's pole faces reach 0.821 T on the front
face -> omega_ce dt = 0.217 at 1.5 ps, above the 0.2 gate; the composer reduces dt to 1.3 ps (0.188; +15 % steps) for
that run only. Design 047's base cathode annulus is connected at 23/24 samples; the composer's shrink ladder selects
r_outer = 0.9 x min(r_w, r_exit) (24/24). Neither issue exists in the channel-only option.

## 5. Preflight (whole set, both options green)

`run.py preflight --domain channel` -> `preflight-channel.json`; `--domain plume-24mm` -> `preflight-plume-24mm.json`.
Per design and option: identity (rebuilt hashes = sealed authorities), field binding (hashes re-verified, production
gates), grid (bore / exit / front-face / plume radii and the cone start on grid lines, worst snap <= half a cell),
field map (builds inside the supported box; a-priori stability admitted: channel omega_ce dt 0.028-0.085 at
0.107-0.321 T, omega_pe dt 0.054 at 4e17 m^-3, 1.5 cells / lambda_D at the reference), mesh masks (plasma cells,
far-field and body-face nodes for the plume), protocol (`runner.build_config` accepts the composed protocol; feed
4.70e16-1.60e17 atoms/s for n_g0 = 5.5e19 at each exit area, the reference's 8.551e16 reproduced), cathode
connectivity (plume: 24/24 for every design after the placement rule) and the cost row. 5/5 designs pass both.

## 6. Protocol composition and operating-point policy (`protocol.py`)

Templates: channel -> `pic2d_cft_steady_state_v3/protocol.json` (model v1.4); 12 mm -> `pic2d_cft_plume_v1`
(v2.0.2); 24 mm -> `pic2d_cft_plume_v2_1` (v2.1). Only the design-dependent blocks change: geometry / grid
(dr = r_w / round(r_w / 50 um), dz = L / round(L / 50 um); exit radius, cone start, front-face dielectric radius =
magnet inner radius, plume radius = return-yoke outer radius snapped to grid lines and recorded - zero error for the
reference, <= 25 um otherwise), neutral feed Q_in = c n_g0 at the design's exit area (equal initial neutral density
5.5e19 m^-3 and equal null-collision headroom), cathode annulus r in [0.25 r_w, min(r_w, r_exit)] shrunk until
connected, z in [L + 0.3, L + 1.0] mm (the reference's 0.5-2.0 / 24.3-25.0 mm reproduced), plume-gate arming = one
transit, dt reduced only where omega_ce dt would exceed 0.95 x 0.2, wall budget 1.25x the projection, W = 6e4
(capped at a projected 8 M particles). Anode 300 V, seed plasma, series / checkpoint / window cadence, gates,
plateau rule and frame recorder are the template's verbatim. `run.py run` refuses to start without
`--allow-launch` and a green whole-set preflight of the option; results go to `results/<design>-<domain>/` (ignored).

## 7. Open decisions (before the preregistration)

1. Attempt 8's verdict is in (`ac248e05`): no plateau, finite-grid heating past Delta/lambda_D ~3.2. The sweep must
   inherit the follow-up's grid / gate decision: the recalibrated peak-Debye gate (CIC threshold) + windowed
   residual-power gate, and the 50 um vs 33 um / 1.4 ps choice, validated on the reference's channel-only
   refinement run BEFORE the sweep is preregistered (HEMP-like designs with 0.27 T cusp fields may reach higher
   peak densities than the reference; the runtime gates fail closed, but a design that trips them is a lost run).
2. Channel-only (model v1.4, exit-plane injection) vs plume for the sweep runs - recommended channel-only; if the
   verdict says the channel-only exit boundary distorts the cusp-loss estimands, the 24 mm option is ready (but
   needs the resolved-peak operating point first).
3. Design 047: keep with the disclosed anode-edge boundary cusp, or substitute 061.
4. Operating-point policy: equal n_g0 (chosen) vs equal mass flow vs equal mass-flow density; equal injection
   current (3 mA) vs current scaled with the exit area.
5. Whether the 056 seed replicate is the right single replicate (alternative: a W x 0.7 replicate of 106).
6. The estimand definitions (cusp window half-width min(1 mm, pitch/4); near-wall band 0.5 mm; the Kornfeld chain's
   "entering current" = injected minus returned) must be frozen; a non-evidentiary shakedown on ONE real run
   through `run -> finalize -> targets` before the prereg (the orbit v1-v3 / L1b lesson).
7. Field level: level-0 padded solves (as the reference's padding-1.5 map) - a level-1 refinement is not needed by
   the gates (1.5 mT vs the L1b level-1 map) but is the natural upgrade if the prereg wants P2-qualified fields.

## Commands (from `modern/`; CPU unless stated; never while another host factorisation runs)

    $env:PYTHONPATH="$PWD\src;$PWD"; $env:OMP_NUM_THREADS='1'
    python -m experiments.pic2d_design_mini_sweep_v1.run fields                      # 4 P2 solves, ~2-7 min each, RSS <= 0.5 GB
    python -m experiments.pic2d_design_mini_sweep_v1.run preflight --domain channel
    python -m experiments.pic2d_design_mini_sweep_v1.run preflight --domain plume-24mm
    python -m experiments.pic2d_design_mini_sweep_v1.run cost
    python -m experiments.pic2d_design_mini_sweep_v1.run protocol --design l1a-gs-v3-056-effcbc8686 --domain channel
    python -m experiments.pic2d_design_mini_sweep_v1.run draft-protocol
    # GPU, after the preregistration only (DRAFT guard): run --design ID --domain channel --allow-launch

Tests: `tests/pic2d/test_pic2d_design_mini_sweep.py` (13: design list and catalogue numbers, identity and PIC
mapping for every design and option, field bindings and their gates, node map hash / scale binding and fail-closed
behaviour, reference pipeline, protocol composition reproducing the reference values and scaling the feed, draft
protocol on disk = generator, cost anchors and the v2.1 spec row, Kornfeld mapping, closure extraction on a
synthetic plateau, whole-set channel preflight, launch guard).

## Launch log

* 2026-09-04 (DRAFT prepared, no run): design selection; four material-aware P2 fields produced and gated (CPU,
  22 min total, peak RSS 503 MB, GPU untouched: attempt 8 PID 51256 was alive at 11:25 when this work started and
  ended on its own grid-heating triad gate at 4.98 us, recorded by `ac248e05` at 11:57); whole-set preflight green
  for the channel and 24 mm options; protocol.json draft; tests.
