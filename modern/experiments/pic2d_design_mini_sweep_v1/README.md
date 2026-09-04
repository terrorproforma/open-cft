# pic2d design mini-sweep v1 - PREREGISTERED (channel-33um option on the Lambda H100, 2026-09-04)

**Status: the `channel-33um` option is PREREGISTERED** (section 8: the sealed per-design run protocols under
`protocols/`, the whole-set preflight, the MPS determinism replay and the one-design shakedown run on the launch box,
all bound by hash in `protocol.json`; the four primary designs are launched from the preregistration commit through
`tools/cloud/schedule.py` with four CUDA-MPS slots on one H100). No result exists yet; the closure targets of a design
are quotable only under its predeclared acceptance (section 8.4). Sections 1-6 are the design, field, closure-target,
cost and composition record of the DRAFT phase (2026-09-04 morning) and remain the description of what is being run;
section 7 lists the decisions that closed the draft's open points, with reasons.

**Draft-phase preamble (kept verbatim).** The preregistration commit follows the
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

## 7. Decisions that closed the draft's open points (2026-09-04, before the preregistration commit)

The draft listed seven open decisions. What was decided, and why (every item is also in `protocol.json`
`preregistration.decisions`):

1. **Grid / gates: channel-only 33.33 um / 1.4 ps with the PIC model v2.0.3 gates.** Attempt 8's verdict
   (`ac248e05`) retired 50 um for a dense peak; the refined grid is the steady-state v4 grid (`392129e5`, running
   locally on the 5090 as the reference's convergence test). Grid target changed from the draft's 3.33e-5 m to
   24 mm / 720 = 3.3333e-5 m so the reference design reproduces v4's 90 x 720 cells EXACTLY (the draft's target
   gave 90 x 721). Gates: window-mode peak-Debye gate hard pi / soft 2.5 on the 400 000-step interval-averaged peak
   (>= 32 macro-electrons), one-sided windowed residual-power gate >= 5 % of the electrode work, the v1.4 triad
   drift members, omega_pe dt / Courant / Poisson / inventory gates. Plateau rule: the accepted steady-state rule
   (>= 3 design transits, trailing-20 % drifts of I_d / N_e / n_g < 5 %) plus the v2.0.3 preconditions (triad soft,
   peak-Debye soft 2.5). Frames ON (20 000-step = 28 ns interval averages).
2. **Template / physics: the steady-state v4 protocol (model v1.3 closure, NO wall-ion recycling) - CHANGED from
   the draft's v1.4 (recycling) channel template.** Reasons: (i) the sweep's reference run must be comparable with
   the only 33 um reference run whose convergence verdict it has to cite (ss-v4: same physics, grid, dt, W, seed and
   operating point - the sweep's reference run is a numerical replication of v4 on a different GPU); (ii) v1.4 has
   no accepted plateau anywhere (steady-state v3 was never launched) and its recycled fixed point projects the
   reference peak at Delta/lambda_D ~2.45 on this grid - on the 2.5 soft precondition - so every denser design
   would risk "no plateau by soft margin"; the v1.3 peak projects 2.11; (iii) budgets and ignition expectations
   anchor on the accepted v1.3 plateaus. Wall-ion recycling is a declared follow-up closure (under v1.3 gross = net
   utilisation; recorded as such). Channel-only stays (the closure targets are channel quantities).
3. **Design 047 KEPT with the disclosed anode-edge boundary cusp.** Boundary classification under the v3.1 0.25 mm
   ambiguity tolerance; the three interior cusps match L1a within 0.30 mm of the 0.45 mm tolerance; the 73 um
   separatrix foot sits inside the anode sheath (2 cells at 33 um). Its electron loss is reported separately as
   `anode_edge_electron_wall_current_a` (closure.extract_targets, band 0.25 mm from the anode plane, all designs),
   never as an interior cusp. Substitute 061 (rho 0.381) NOT taken: its 5.9 mm exit taper would confound the rho
   ladder with a long cone the other three designs do not have.
4. **Operating point: equal n_g0 (5.5e19, feed Q_in = c n_g0 at the design's exit area) and equal 3 mA / 2 eV
   exit-plane injection** (the draft's choice, frozen; not equal mass flow, mass-flow density or current density).
5. **Replicate: the 056 seed replicate (seed 20260904)** is sealed as its own run protocol
   (`protocols/l1a-gs-v3-056-effcbc8686-channel-33um-seed-replicate.json`, `launch --case seed-replicate`); it is
   NOT launched in the first campaign (a fifth MPS slot slows every process: N = 8 gave 17.1 ms/step per process).
6. **Estimands frozen** as `closure.extract_targets` implements them: cusp window half-width min(1 mm, pitch / 4),
   near-wall band 0.5 mm, Kornfeld chain seeded with (injected - returned) exit electron current, anode-edge band
   0.25 mm; cusp planes and cells from the design's own material-aware topology (`binding.json`). The
   non-evidentiary shakedown of ONE real design through run -> assess -> targets -> re-finalize path ran on the
   launch box (section 8.3).
7. **Field level: level-0 padded material-aware solves** (binding gates: mesh >= 5 deg, residual <= 2e-10, coverage,
   topology within tolerance, |dB| vs L1b <= 1.5 mT); no level-1 refinement in this preregistration.

Also decided (not in the draft's list): **macro weight** with particles-per-cell PARITY to the 50 um runs,
W = 6e4 x dr dz / (50 um)^2 (26 666.7 for the reference = v4's value; 26 566.8 / 26 655.3 / 26 799.2 for 056 / 047 /
009), cap 12 M projected particles (80 GB H100; only the optional 106 hits it: W 30 877) - the draft kept W = 6e4,
i.e. 2.25x fewer particles per cell than the accepted runs; **budgets** 1.5x the projected 3-transit wall at the
H100 CUDA-MPS four-slot per-process rate (cost model scaled to the measured 8.71 ms/step N = 4 anchor for the v4
configuration), rounded up to 10 min, cumulative over resumes; **GPU model recorded** in every run protocol
(`execution.gpu`: NVIDIA H100 80GB HBM3, driver 580.105.08, CUDA MPS four slots, the determinism finding below).

## 8. Preregistration record (channel-33um, Lambda H100, 2026-09-04)

### 8.1 What is sealed

`protocol.json` (schema `cft.pic2d.design-mini-sweep.protocol/1.0.0`, status
`preregistered_design_mini_sweep_v1_channel_33um_h100_mps4_not_validated`) binds by sha256: the six sealed run
protocols under `protocols/` (four primary designs, the optional 106, the 056 seed replicate), the whole-set
preflight `preflight-channel-33um.json`, the shakedown `shakedown-channel-33um.json` and the MPS determinism replay
`mps-replay.json`. `run.py launch --design ID --grid 33um --expect-commit <prereg sha> --require-mps` refuses:
HEAD != the commit, a dirty worktree, `protocol.json` or the design's sealed protocol differing from HEAD's blob, a
recomposition (on the design's own node field, this platform) that differs from the sealed bytes, a missing or
failed record, a missing MPS pipe directory, an existing `execution-lock.json` (O_EXCL, per results directory).

### 8.2 Per-design protocol (the four launched designs; every value from `preflight-channel-33um.json` / `protocols/`)

| design | role / rho (iron) | cells (dr x dz um) | dt | W (parity) | N projected | ms/step MPS-4 (5090 model) | transit | steps to 3 transits | hours to 3 transits at MPS-4 | wall budget | GPU GB | max B / omega_ce dt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `divergent-exit-stack` | reference / 0.60 | 90 x 720 (33.33 x 33.33) = ss-v4 | 1.4 ps | 26 666.7 | 4.50 M | 8.71 (4.74) | 2.400 us | 5 142 857 | 12.5 h | 18.8 h (67 800 s) | 7.7 | 0.291 T / 0.072 |
| `l1a-gs-v3-056-effcbc8686` | HEMP-like / 2.36 | 115 x 512 (33.23 x 33.30) | 1.4 ps | 26 566.8 | 10.23 M | 16.57 (9.01) | 1.705 us | 3 653 784 | 16.8 h | 25.3 h (91 200 s) | 15.0 | 0.321 T / 0.079 |
| `l1a-gs-v2-047-e3196a8aa5` | low rho / 0.38 (+ anode-edge cusp disclosed) | 66 x 778 (33.31 x 33.32) | 1.4 ps | 26 655.3 | 2.68 M | 5.67 (3.09) | 2.592 us | 5 553 645 | 8.8 h | 13.3 h (48 000 s) | 5.3 | 0.254 T / 0.063 |
| `l1a-gs-v3-009-d0c686b4aa` | mid rho / 0.92 | 80 x 684 (33.51 x 33.32) | 1.4 ps | 26 799.2 | 6.67 M | 11.26 (6.13) | 2.279 us | 4 883 555 | 15.3 h | 23.0 h (82 800 s) | 10.5 | 0.107 T / 0.026 |

Sealed but not launched: `l1a-gs-v3-106-ccec1c8b2f` (123 x 606, W 30 877.1 at the 12 M cap, 19.4 ms/step, 4.33 M
steps, 23.3 h, budget 35.2 h) and the 056 seed replicate (as 056, seed 20260904). dt: 1.4 ps for every design - the
channel-only maps put omega_ce dt at 0.026-0.079; design 056's 1.3 ps requirement belongs to the 24 mm plume box's
0.821 T pole faces, not to this option. Total projected: 43.3 GB of the 80 GB; the campaign wall time is the longest
row plus its margin, not the sum (the four run concurrently; per-process speed improves as slots empty, so the
MPS-4 hours are upper bounds).

### 8.3 Records produced on the launch box (`ubuntu@68.209.75.2`, H100 80GB HBM3, driver 580.105.08, Python 3.12.14, numpy 2.5.2, warp-lang 1.14.0)

* **Whole-set preflight** (`preflight-channel-33um.json`, 05:27 UTC, 16 s, all 5 designs PASS): identity, field
  binding (file hashes re-verified; `field_source_sha256` per design recorded - the platform-independent binding),
  grid snaps <= half a cell, field map + a-priori stability at the composed dt (omega_pe dt 0.050 at 4e17,
  omega_ce dt <= 0.079, Courant 0.50), mesh masks, composed protocol accepted by `runner.build_config` (W, budget,
  v2.0.3 gate keys, frames, no recycling recorded), cathode connectivity skipped (channel-only: exit-plane
  injection, no cathode region - the 24/24 connectivity gate belongs to the plume options), cost row.
* **Shakedown** (`shakedown-channel-33um.json`, design 056 on its real material-aware field, 100 000 steps at the
  shrunk cadences 200 / 4000 / 40 000 / frames 2000, 283 s, 2.81 ms/step solo at 1.35 M e- + 1.43 M i): stop
  `target_steps_reached`, 25 checkpoints, 50 frames; the window-mode peak-Debye gate ENFORCED in 301/500 records
  (max 0.62 cells/lambda_D, 17 086 resolved nodes, mean occupancy 150 at the peak); the windowed residual-power
  window complete in 280 records (last -12.4 %, cooling side, as the accepted plateaus' seed windows);
  `assess` -> `no_plateau` (expected at 0.14 us; v4 verdict cited as pending); `targets` extracted every closure
  target (Kornfeld chain p 0.136 / 0.096 / 0.071 at the 12.74 / 8.53 / 4.31 mm cusps, ionisation shares
  0.07 / 0.22 / 0.29 / 0.42 anode -> exit, non-evidentiary numbers of a 0.14 us transient); THEN the externally-
  stopped path (`finalize --allow-refinalize`) ran on the scratch directory (2.8 s; maps downgraded to
  instantaneous as designed). The FIRST shakedown attempt died in the runner's finalization with
  `KeyError: 'n_max_per_m3'` - the composed budget block lacked the two keys `summary.budget_check` reads (the
  draft's composer had never reached a finalization); fixed in `8b21b868`, the shakedown re-run from scratch.
* **MPS determinism replay** (`mps-replay.json`, reference design, 6000 steps at the shrunk cadences: two
  concurrent processes under CUDA MPS + two solo processes, every pair compared): the PHYSICS state replays
  BITWISE - particle positions / velocities, fixed-point charge deposition, potential, densities, ionisation and
  wall FLUX maps, currents, counts, neutral inventory, `series.jsonl` physics blocks, `checkpoint-final` particle
  arrays, `final_counts`, `window_currents_a` - between the concurrent MPS processes and the solo processes alike.
  Only the float-atomic DIAGNOSTIC accumulators differ, at round-off (<= 2.2e-13 relative on the ledger interval
  residual, ~1e-16 elsewhere): the window velocity moments (T_e maps, `sample_count_e`, the peak-node Debye
  statistics), the energy / momentum ledger sums and the wall mean-energy maps. The solo-vs-solo pair shows the
  SAME pattern, so MPS is neutral: it does not change a process's own kernel order; the round-off lives in the
  device atomics of the v2.0.2 accumulators and exists without concurrency. Consequence recorded in
  `mps-replay.json`: same-seed replays are bitwise on every quantity a gate reads EXCEPT the windowed T_e / Debye
  and residual-power statistics, which agree to ~1e-13 - a gate decision could differ between two replays only in
  a case marginal at that level. The FIRST replay attempt was hit by the fallout of the interrupted first shakedown
  (below) and the second by the strict all-bitwise criterion (which the diagnostic atomics cannot meet); the
  criterion was made explicit (physics bitwise, diagnostics within 1e-6, solo pair for the MPS-free pattern) in
  `c51b6ea3` / `7717062b`, and the replay re-run.
* **Operational finding (Xid 31 under MPS):** killing the first shakedown mid-step (`tmux kill-session` + `pkill`,
  SIGTERM/SIGHUP; the runner has no signal handler) left an MMU fault (Xid 31, `FAULT_PDE`) attributed to the MPS
  server; 105 s later the NEXT client to connect faulted at the same address, the server tore that client down and
  reset its device context ("All clients belonging to error triggering process ... will be torn down"), and the
  sibling process that connected 0.5 s later got a sticky `unspecified launch failure` at Warp init. Processes
  started after the reset ran normally. Rule for the campaign: never kill a sweep process under MPS; the runs stop
  themselves (plateau / budget / gate). If a stop is unavoidable, expect the server to reset on the next
  connection and check `/tmp/nvidia-log/server.log` and `dmesg | grep Xid` before any relaunch. A SIGTERM handler
  that checkpoints and stops cleanly is the follow-up change to the shared runner.

### 8.4 Acceptance per design (verbatim in every sealed protocol, `stopping_rule.acceptance`)

(a) plateau under the rule of section 7.1; (b) windowed residual power at the stop < +2 % (one-sided); (c) closure
targets extracted from the trailing-window maps by `run.py targets`; verdicts `closure_quotable` ((a) AND (b)),
`plateau_with_heating` ((a) not (b)), `no_plateau`; (e) a design effect counts only above the reference's seed-b /
W x 0.7 spread and, once run, the 056 replicate's spread; (f) the **convergence caveat**: the 50 -> 33 um verdict of
the reference (ss-v4, `392129e5`, running locally; verdict expected 18:45-19:30 AEST 2026-09-04) is PENDING at this
preregistration and MUST be cited per design when the sweep is assessed - `converged` -> the 33 um values carry the
v4 tolerances as their grid band; `resolution_limited` -> the values are the resolved numbers with NO grid band of
their own; `refinement_heating` / `no_plateau` -> the reference grid is not certified and each design's quotability
rests on its own residual-power and peak-Debye readings ("at 33 um, uncertified"); `run.py assess` reads the v4
`assessment.json` when it exists and writes the applicable statement; (g) the 047 anode-edge disclosure.

### 8.5 Amendment 1 (2026-09-04, `protocol.json` `amendments[0]`): design 056 launch 2 under the v2.0.4 gate reading

Launch 1 of design 056 was stopped at 2.07 transits by the triad member `omega_pe_dt_drift` (+0.283 > 0.25), which at the
preregistration commit `291a9227` is the trailing-20 % drift of the RAW single-node single-step peak omega_pe dt. The
launch-log entry below and `results/l1a-gs-v3-056-effcbc8686-channel-33um-launch1-triad-gate-stop/triad-stop-diagnosis.json`
show the stop to be a shot-noise artefact of 1-3-macro-electron axis nodes (resolved-node member +0.0165, no heating
signature). Model v2.0.4 (`79e6a670`, landed on the branch at 21:54 AEST, after the sweep launch) reads the runtime
omega_pe dt gate - and therefore the triad member - on the peak over RESOLVED nodes (>= 32 macro-electrons, the peak-Debye
gate's floor) and records the raw peak alongside (`SeriesRecord.peak_omega_pe_dt_raw`, `v1_4_options.omega_pe_dt_gate`).

What the amendment does, and what it does not:

* **Relaunch**: design 056 launch 2 is a FRESH start (no cross-code resume) from a commit carrying v2.0.4, with the same
  seed 20260903, W 26 566.8, 115 x 512 cells, dt 1.4 ps, operating point, v1.3 closure, v2.0.3 gates and cadences, plateau
  rule, acceptance (a)-(g) and wall budget 91 200 s; one of the four H100 CUDA-MPS slots through `tools/cloud/schedule.py`
  (job `sweep-056-launch2`) once a sweep run frees one. Same-seed physics replays bitwise (`mps-replay.json`), so launch 2
  reproduces launch 1's physics up to step 2 520 000 and continues - a free regression check on the record.
* **Sealed protocol identity**: `protocols/l1a-gs-v3-056-effcbc8686-channel-33um.json` gains ONE top-level block,
  `omega_pe_dt_gate_reading` (`protocol.OMEGA_PE_DT_GATE_READING_V2_0_4`: statistic `resolved_node_single_step_peak`, floor
  32, limit 0.2, model v2.0.4, the launch-1 disclosure), emitted by the composer for `(design 056, case base)` only
  (`protocol.AMENDED_GATE_READING`). Re-sealed on the launch box (`compose --grid 33um` in a scratch worktree at `ccee5c60`):
  sha256 `35760e9b5bcd...` (launch 1) -> `8b876b31eb14...`; the five other sealed protocols are byte-identical to `291a9227`
  (reference `ec8baa2aa38d`, 047 `b23b66da579a`, 009 `eb54049c6d84`, 106 `2fe6577b6865`, 056 seed replicate `eef171af348e`).
  The block is documentary - `runner.build_config` reads named keys only - so the run's configuration identity is unchanged;
  the execution lock of launch 2 names the new protocol hash and the amendment commit.
* **Unchanged**: thresholds (omega_pe dt 0.2; triad soft 5 % / hard 25 %), grid, dt, W, seed, operating point, physics,
  the v2.0.3 window-mode peak-Debye and windowed residual-power gates, budget, GPU / MPS execution block. The reference,
  047 and 009 runs stay locked at `291a9227` with the RAW reading; their records disclose it (047: resolved +0.80 % vs raw
  +0.98 % at the plateau, declaration independent of the reading). The 056 seed replicate and design 106 are not amended:
  when launched (necessarily from a commit carrying v2.0.4) they need the same declaration first.
* **Why this is an amendment and not a new experiment**: the physics, numerics and acceptance rule are what the
  preregistration protects; the gate statistic is a numerical diagnostic whose raw form had already been shown (attempt 6
  plume-boundary gate, external-validation launch-box preflight) to read single macro-particles on the smallest nodes.

## Energy-ledger correction (model v2.0.6, post hoc; recorded values unchanged)

Up to model v2.0.5 the energy ledger's `inelastic_loss_j` lacked the macro weight W (found by the external-validation v0 launch-1 diagnosis, 036bd679), so every recorded interval residual was `H - L_inel` - biased NEGATIVE by the inelastic power - where `H = field work + dU - electrode work` is the true numerical energy creation. The sidecar(s) `ledger-corrected.json` (+ `.sha256.json`) were written by `python -m cft_revival.pic2d.ledger_recompute <results-dir>` from the recorded `series.npz` (corrected residual = H per record; `spec/pic2d/pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6`); **the recorded series, maps and summaries are unchanged.** Values below: trailing-400 000-step residual / electrode work at the last record, recorded -> corrected.

| design | sidecar | windowed recorded -> corrected | cumulative recorded -> corrected | acceptance (b) < +2 % |
|---|---|---|---|---|
| 047 (plateau, 3.003 transits) | `results/l1a-gs-v2-047-e3196a8aa5-channel-33um/ledger-corrected.json` | -7.1 % -> **+0.9 %** | -7.3 % -> +0.7 % | pass -> pass |
| 056 launch 1 (triad stop, shot noise) | `results/l1a-gs-v3-056-effcbc8686-channel-33um-launch1-triad-gate-stop/ledger-corrected.json` | -7.6 % -> **+0.6 %** | -8.9 % -> +0.5 % | pass -> pass |
| 009, reference, 056 launch 2 | results not committed at 036bd679 - run the tool when they land | | | |

Both corrected trajectories are flat (047 +0.0 % at 0.62 us -> +0.9 % at 7.78 us; 056 L1 +0.5 -> +0.6 %), well inside the 5 % gate and the 2 %
acceptance. The running / pending designs (009, reference, 056 launch 2) execute pre-v2.0.6 code in their locked worktrees: their recorded
`grid_heating_triad.windowed_energy_residual_over_electrode_work` stays biased by about -8 % and acceptance (b) must be evaluated on the
sidecar written by the tool. The whole-sweep `assess` should cite the sidecars.

## Commands (from `modern/`; CPU unless stated; on the box `PYTHONPATH=src:.` with the MPS variables exported)

    $env:PYTHONPATH="$PWD\src;$PWD"; $env:OMP_NUM_THREADS='1'
    python -m experiments.pic2d_design_mini_sweep_v1.run fields                                  # 4 P2 solves, ~2-7 min each, RSS <= 0.5 GB
    python -m experiments.pic2d_design_mini_sweep_v1.run preflight --domain channel --grid 33um  # whole-set preflight of the preregistered option
    python -m experiments.pic2d_design_mini_sweep_v1.run cost
    python -m experiments.pic2d_design_mini_sweep_v1.run protocol --design l1a-gs-v3-056-effcbc8686 --grid 33um [--with-field]
    python -m experiments.pic2d_design_mini_sweep_v1.run compose --grid 33um                     # seal protocols/ + protocol.json (run on the launch box)
    # GPU, launch box: mps-replay / shakedown (non-evidentiary records) and the preregistered launch
    python -m experiments.pic2d_design_mini_sweep_v1.run mps-replay --design divergent-exit-stack --grid 33um
    python -m experiments.pic2d_design_mini_sweep_v1.run shakedown --design l1a-gs-v3-056-effcbc8686 --grid 33um
    python -m experiments.pic2d_design_mini_sweep_v1.run launch --design ID --grid 33um --expect-commit <prereg sha> --require-mps   # via tools/cloud/schedule.py
    python -m experiments.pic2d_design_mini_sweep_v1.run status|assess|targets --design ID --grid 33um
    # labelled development runs only (never evidence): run --design ID --grid 33um --allow-launch [--shrunk-cadences --max-steps N]

Tests: `tests/pic2d/test_pic2d_design_mini_sweep.py` (20: design list and catalogue numbers, identity and PIC
mapping for every design and option, field bindings and their gates, node map hash / scale binding and fail-closed
behaviour, reference pipeline, 50 um protocol composition reproducing the reference values and scaling the feed,
the channel-33um option = the v4 configuration with the v2.0.3 gates and parity W for every design, dt policy,
experiment protocol on disk = generator, sealed protocols = recomposition (float-tolerant, any platform), cost
anchors incl. the H100 MPS-4 anchor, Kornfeld mapping, closure extraction incl. the anode-edge band, whole-set
preflight for both channel options, run / launch guards, shrunk-cadence protocol, replay comparison classifier).

## Launch log

* 2026-09-04 (DRAFT prepared, no run): design selection; four material-aware P2 fields produced and gated (CPU,
  22 min total, peak RSS 503 MB, GPU untouched: attempt 8 PID 51256 was alive at 11:25 when this work started and
  ended on its own grid-heating triad gate at 4.98 us, recorded by `ac248e05` at 11:57); whole-set preflight green
  for the channel and 24 mm options; protocol.json draft; tests.
* 2026-09-04 15:20-15:50 AEST (05:20-05:50 UTC), Lambda H100 `68.209.75.2`, code `412d240f` -> `7717062b`: the
  channel-33um option composed and its records produced on the box - preflight 5/5 PASS (05:27 UTC), shakedown of
  056 (100 000 steps, 283 s, 2.81 ms/step solo), MPS determinism replay (physics bitwise, diagnostics at
  round-off, MPS-neutral), six run protocols sealed; two composer defects found and fixed by the shakedown /
  replay before the freeze (budget keys the runner's finalization reads; the replay criterion). Xid-31 event from
  an interrupted shakedown recorded (section 8.3).
* **2026-09-04 15:52-15:56 AEST (05:52-05:56 UTC) - LAUNCH 1 of the four primary designs**, preregistration commit
  `291a9227` (launch config `9c426f90`: `tools/cloud/jobs.yaml` slots_per_gpu 4, MPS client variables), Lambda
  `gpu_1x_h100_sxm5` `ubuntu@68.209.75.2`, GPU 0 = NVIDIA H100 80GB HBM3 `GPU-a800b021-6364-473f-5177-cd6ae7ce0005`,
  driver 580.105.08, CUDA MPS daemon PID 14340 / server 14519, 6 BLAS threads per job. Launched by
  `tools/cloud/schedule.py launch --only sweep-reference sweep-056 sweep-047 sweep-009`: each job in its own detached
  worktree at `291a9227` under `/lambda/nfs/h100-files/cft/jobs/<id>/tree` (LFS smudged, ~80 s each), Warp `cuda:0`
  UUID cross-checked (`gpu_uuid_match true` in every `state.json`), `launch --expect-commit 291a9227 --require-mps`
  passed every check (clean worktree, blobs == HEAD, recomposition == sealed bytes, records present) and acquired the
  O_EXCL `execution-lock.json` (commit `291a9227`, MPS pipe `/tmp/nvidia-mps`, GPU recorded):

  | job | design | PID (wrapper) | started UTC | lock protocol / config | early ms/step (seed load, 4 active) | projected MPS-4 ms/step at plateau load | steps to 3 transits | hours to 3 transits (MPS-4 rate) | budget |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `sweep-reference` | `divergent-exit-stack` | 19764 (19709) | 05:52:03 | `ec8baa2aa38d` / `fc788b7eac22` | 4.46 | 8.71 | 5 142 857 | 12.5 h | 18.8 h |
  | `sweep-056` | `l1a-gs-v3-056-effcbc8686` | 19913 (19866) | 05:53:26 | `35760e9b5bcd` / `3d247f1ea3f6` | 5.58 | 16.6 | 3 653 784 | 16.8 h | 25.3 h |
  | `sweep-047` | `l1a-gs-v2-047-e3196a8aa5` | 20079 (20024) | 05:54:47 | `b23b66da579a` / `43709022227a` | 3.69 | 5.7 | 5 553 645 | 8.8 h | 13.3 h |
  | `sweep-009` | `l1a-gs-v3-009-d0c686b4aa` | 20189 (20176) | 05:56:09 | `eb54049c6d84` / `fba041575d99` | 4.46 | 11.3 | 4 883 555 | 15.3 h | 23.0 h |

  At 05:57 UTC: GPU 100 % busy, 5.0 GB used (1.07-1.42 GB per process at the seed load; projected 5-15 GB each at the
  plateau loads), no new Xid. Every run ignited from the seed as the reference does (I_d 1.3-2.2 mA, S 1-3e16 /s, n_g
  falling from 5.5e19 within the first 0.15 us). The per-process step will rise toward the projected MPS-4 rates as
  the particle counts grow to their plateau loads, and fall again as slots empty: expected 3-transit times from the
  MPS-4 projection are 047 ~01:00, reference ~04:30, 009 ~07:00, 056 ~08:40 AEST 2026-09-05 (upper bounds; the plateau
  rule may declare later, up to the budgets). Monitoring: `tools/cloud/schedule.py status` on the box; the runner's
  `gpu=100%` samples are whole-GPU readings under MPS. Do NOT kill a process (section 8.3, Xid 31). When a job stops:
  `run.py assess --design ID --grid 33um` and `run.py targets ...` from its worktree; results-only commit from the
  worktree on a `results/<id>` branch; cite the ss-v4 verdict (pending at launch) per design.
* **2026-09-04 12:49:44 UTC (22:49:44 AEST) - design 047 FINISHED on the plateau rule (launch 1, PID 20079, a valid
  record).** `results/l1a-gs-v2-047-e3196a8aa5-channel-33um/` (results-only commit; the steady-state v4 record contract:
  summary with the frames manifest, execution lock, run state, status / series / maps, final checkpoint metadata + npz
  sidecars, the sealed protocol copy; the 278 frames, the 56 MB checkpoint arrays, the field anchor, the 116 MB raw
  `series.jsonl` and the logs stay in the job worktree `jobs/sweep-047/tree` on the box). Stop
  `plateau_reached_after_min_transit_times` at step 5,560,000 = 7.784 us = 3.003 transits (2.592 us a priori), 24,888 s
  wall = 6.91 h, 4.48 ms/step mean under MPS-4 (exit 0, `finished: true`, one session, no resume). Lock: commit
  `291a9227`, protocol `b23b66da579a`, config `43709022227a`. Plateau block: I_d drift +0.60 %, N_e +2.20 %, n_g +0.11 %
  (threshold 5 %), triad soft ok (S -0.22 %, T_e,dense +0.21 %, omega_pe dt +0.98 %), peak-Debye soft ok. Trailing-window
  values (quotable only after `assess`): I_d 1.925 mA (anode e- 1.932, anode ions 0.007), I_beam 0.655 mA, exit electrons
  1.726 mA, wall e-/ions 1.641 / 1.635 mA, S 1.452e16 /s, n_g 3.76e19 (fixed point 3.55e19; feed 4.60e16 atoms/s),
  gross = net utilisation 0.316, N_e 0.687 M / N_i 0.700 M (the 2.68 M projection was 3.9x too high: the low-rho design
  sits at 0.51 of the reference's projected mean density), peak n_e (window, >= 32 macro-electrons) 8.83e17 at z 22.0 mm
  (the exit-side partial cell), T_e,peak 5.86 eV, T_e,dense 5.73 eV, Delta/lambda_D window 1.69 at the stop (max 2.37
  during ignition; soft 2.5 never exceeded), windowed energy residual -7.11 % (cooling side; cumulative -7.30 %; the +2 %
  acceptance bound (b) is met a priori), raw single-node omega_pe dt max 0.115 (limit 0.2). GPU sampler 83/83 samples,
  no Xid. **Raw-statistic disclosure (v2.0.4, `79e6a670`)**: the code this run executed (`291a9227`) forms the runtime
  omega_pe dt gate and the triad's `omega_pe_dt_drift` member from the RAW single-node single-step peak; reconstructed
  from the record (`peak_node.n_e_peak_per_m3` = the >= 32-macro-electron single-step peak, the v2.0.4 statistic) the
  member reads +0.80 % (window-averaged +0.20 %) against the raw +0.98 %, and the raw argmax sat on an axis node holding
  <= 4 macro-electrons in only 17 % of the trailing-window records - the plateau declaration does not depend on the
  reading. `assess` / `targets` are DEFERRED to the sweep-wide assessment after the reference and 009 finish, from a
  checkout that carries the steady-state v4 `assessment.json` (`0d228ad2`, verdict `resolution_limited`), so that the
  predeclared caveat (f) is cited from the file, not from a checkout where it reads "pending".
* **2026-09-04 10:52:09 UTC (20:52 AEST) - design 056 launch 1 STOPPED by the grid-heating triad gate at 2.07 transits
  (PID 19913): `omega_pe_dt_drift 0.283 exceeds 0.25` - DIAGNOSED as a SHOT-NOISE ARTEFACT of the pre-v2.0.4 RAW
  gate statistic, not heating.** Record `results/l1a-gs-v3-056-effcbc8686-channel-33um-launch1-triad-gate-stop/` (same
  contract as 047, plus `triad-stop-diagnosis.json`; archived under the `-launch1-triad-gate-stop` suffix so that
  launch 2 runs in the canonical directory; 126 frames, the 202 MB checkpoint and the 53 MB `series.jsonl` stay in
  `jobs/sweep-056/tree`). Stop `grid_heating_triad_gate_stopped_run` at step 2,520,000 = 3.528 us = 2.069 transits
  (1.705 us a priori), 17,914 s = 4.98 h, 7.11 ms/step mean (8.0 at the end), exit 0, `finished: true`; lock commit
  `291a9227`, protocol `35760e9b5bcd`, config `3d247f1ea3f6`. State at the stop: I_d 5.41 mA, I_beam 4.88 mA,
  S 4.30e16 /s, n_g 3.79e19 (fixed point 3.63e19), utilisation 0.311, N_e 2.51 M / N_i 2.50 M (the 10.2 M projection
  4x too high), peak n_e (window) 5.58e17 at z 10.4 mm, T_e,peak 4.81 eV, T_e,dense 5.26 eV, Delta/lambda_D window 1.54
  (max 1.55; soft 2.5 ok), windowed residual -7.58 % (cumulative -8.90 %), plateau drifts I_d +1.8 % / N_e +5.4 % /
  n_g -1.2 % (not yet a plateau: 2.07 < 3 transits and N_e still filling).
  **Diagnosis** (`triad-stop-diagnosis.json`, read-only over series.npz / summary.json / the series.jsonl `raw_peak`
  fields, the runner's own `trailing_time_drift`): the tripped member is `trailing_time_drift(peak_omega_pe_dt)`, and at
  `291a9227` `peak_omega_pe_dt` is the max over EVERY plasma node of the single-step |q_e| deposit (the raw statistic
  v2.0.4 now records only as a witness). Reconstructed under both readings at the stop: RAW +0.283 (hard 0.25) versus the
  v2.0.4 RESOLVED single-step statistic (`peak_node.n_e_peak_per_m3`, the densest node holding >= 32 macro-electrons -
  the same floor and the same single-step deposit the v2.0.4 gate reads) **+0.0165**, and the 400k-step window-averaged
  resolved peak **+0.034** - both inside the 5 % SOFT bound, let alone the hard one. Trailing-window values: raw
  omega_pe dt 0.100 mean (0.080-0.138; first tenth 0.092 -> last tenth 0.115), resolved 0.0625 (0.061-0.065, flat),
  window 0.058 (0.057-0.059). The raw argmax sat on an AXIS node (i = 0) holding <= 4 macro-electrons in 96.4 % of the
  2521 trailing records (p05 0.99, median ~1.5, p95 2.6 macro-electrons; 286 distinct nodes over z-index 0-400 - a
  different 1-3-particle axis node every record), whereas the resolved peak held 157-305 (mean 204) macro-electrons and
  was never on the axis (raw / resolved ratio 1.6 mean, up to 2.2). Trajectory at the checkpoints: the raw member rose
  0.050 (1.905 transits) -> 0.087 -> 0.132 -> 0.190 -> 0.249 -> 0.283 (2.069) over the last 0.16 transits while the
  resolved member FELL 0.019 -> 0.012 -> 0.008 -> 0.007 -> 0.010 -> 0.017 and the window member sat at 0.027-0.034; the
  resolved member exceeded the soft bound only at 1.02-1.45 transits (max 0.137, the still-densifying ignition phase:
  I_d drift +3 to +11 %, T_e,dense falling), never the hard one, and read <= 0.05 from 1.77 transits on. No heating
  signature: the energy-ledger residual per 0.4 us segment is -15.0, -14.3, -9.9, -8.6, -7.9, -7.5, -7.5, -7.5, -7.6 %
  of the electrode work (cooling side, flat - the accepted plateaus' pattern; attempt 8 went +2.4 -> +5.8 -> +54.8 %),
  T_e,dense 9.2 -> 6.3 -> 6.1 -> 6.0 -> 6.1 -> 5.6 -> 5.3 -> 5.2 -> 5.3 eV (falling, then flat: trailing drift +3.6 %)
  while I_d ROSE 2.75 -> 5.44 mA (trailing +1.8 %) - the opposite of the heating signature (T_e up while I_d falls);
  S +2.6 %, S/N_e -2.8 %, K_e/N_e +0.5 %, n_g -1.2 %, Delta/lambda_D window 1.53 rising with the density, far from pi.
  **Verdict: SHOT-NOISE ARTEFACT** (every heating-signature check negative; the resolved statistic 17x inside the hard
  bound while the raw one drifted on 1-3-particle axis nodes) - the same failure mode as the plume attempt-6
  plume-boundary gate and the external-validation preflight stop that produced v2.0.4. Under v2.0.4 the run would have
  continued; it needed ~1.13 M more steps to reach 3 transits (~2.5 h at 8 ms/step). Consequence: launch 2 of 056 under
  model v2.0.4 as a protocol AMENDMENT (next entry). What this record is: a gate-stopped run of the sweep (no plateau,
  not assessable, not a failure of the design); the sweep-wide `assess` runs after the reference and 009 finish.
* **2026-09-04 ~23:40 AEST - AMENDMENT 1 committed (section 8.5, `protocol.json` `amendments[0]`)**: design 056 launch 2
  under the v2.0.4 gate reading, fresh start, same identity otherwise; `protocols/l1a-gs-v3-056-effcbc8686-channel-33um.json`
  re-sealed on the box (`35760e9b5bcd` -> `8b876b31eb14`, the only sealed file that changed; `protocol.json` -> `a7229bf997e4`);
  tests/pic2d/test_pic2d_design_mini_sweep.py 20/20 on the PC. The launch (job `sweep-056-launch2`, `--expect-commit` = the
  amendment commit) waits for the reference or 009 to free a slot - next entry.
* **2026-09-05 00:00:38 AEST (2026-09-04 14:00:38 UTC) - LAUNCH 2 of design 056** (amendment 1, commit `ee35bc84`; jobs.yaml
  `8f68e865`): job `sweep-056-launch2` via `tools/cloud/schedule.py launch --only sweep-056-launch2` into a freed CUDA-MPS slot
  (the external-validation v0 run had stopped itself at 13:56 UTC on its windowed residual-power gate, so the box held three
  PIC clients - reference, 009, steady-state v5 - and the launch keeps it at four; `plan` had refused fail-closed at 13:31 UTC
  with four clients running). Detached worktree `jobs/sweep-056-launch2/tree` at `ee35bc84` (prereg check: ancestor of the
  box head, sealed protocol frozen), Warp `cuda:0` UUID cross-checked, PID 38282 (wrapper 38269), tmux `pic-sweep-056-launch2`,
  6 BLAS threads. Execution lock: commit `ee35bc847dd5`, protocol `8b876b31eb14`, experiment protocol `a7229bf997e4`, config
  identity `3d247f1ea3f6` = launch 1's (the amendment block is documentary, as declared), clean worktree, MPS pipe
  `/tmp/nvidia-mps`. Stepping at 5.9 ms/step at the seed load (three other clients). **Replay check at the first common record
  (step 4400)**: electrons 1 288 811, I_d 3.6332017077986196 mA and the RAW peak omega_pe dt 0.09895866033218684 are bitwise
  launch 1's values - the physics replays as `mps-replay.json` predicts; the series now carries both readings
  (`peak_omega_pe_dt` = resolved 0.0283, `peak_omega_pe_dt_raw` = 0.0990 at that record; the reconstruction from launch 1's
  `peak_node` moments gave 0.0276 there, 2.6 % apart - deposit vs moment sample - so the launch-1 diagnosis is a faithful
  proxy). Expected: 3 transits (5.115 us, 3.65 M steps) at ~6-8 ms/step -> ~06:00-08:00 AEST 2026-09-05; budget end 25.3 h
  after the start (~01:20 AEST 2026-09-06). Sweep status at the launch: 047 finished (plateau, 22:49 AEST), **009 FINISHED
  on the plateau rule at ~23:59 AEST (step 4 920 000 = 6.888 us = 3.02 transits, exit 0)** - its record commit is not part of
  this entry; the reference at 2.92 transits (~10 min to 3). Do NOT kill any process (section 8.3); the sweep-wide `assess`
  (citing v4's `resolution_limited`) runs after the reference and 056 launch 2 finish.
