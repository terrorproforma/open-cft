# pic2d external validation v0 - code-to-code vs Brandt et al. 2016 (PREREGISTERED option `channel-20um`, Lambda H100)

**Status: PREREGISTERED (section 9); two executions on the Lambda H100 — launch 1 `channel-20um` (section 10: heating stop, INCONCLUSIVE) and
launch 2 `channel-20um-bohm-0.4` under amendment 1 (section 14: drift-member ARTEFACT stop on a marginal sustained low-density discharge,
INCONCLUSIVE; no launch 3 now — it waits for the full-physics model).** This directory is the roadmap's
"External validation v0 - code-to-code vs Brandt 2016" step (`LITERATURE_SYNTHESIS.md` 5.7 / 7a; `paper/evidence/result-gates.json`
GATE-L3 stays closed): the reference case extracted from the paper and its companion thesis (section 1-2), the reconstruction of its
magnet stack on the parametric CFT geometry v1.1 (3), the material-aware P2 field with published-anchor gates (4), the ASME V&V 20
comparison spec (5), the run protocols composed on the steady-state v4 template at the published 20 um resolution (6), the whole-set
preflight (7), the declared inconclusiveness conditions (8) and the preregistration record produced on the launch box (9). Sections 1-8
are the DRAFT of 2026-09-04 17:44 AEST (`645c7de4..7fa9e6c6`) with the numbers unchanged; what changed between the draft and the
preregistration is listed in section 9.3. The claim ceiling of the whole exercise is **cross-model agreement** (the reference is a published
model output, `EvidenceKind.PUBLISHED_EXTERNAL`): it validates nothing against hardware and opens no physics level.

## 1. Reference: which paper, why, alternatives

**Chosen:** Brandt, T., Schneider, R., Duras, J., Kahnfeld, D., Hey, F. G., Kersten, H., Jansen, F., Braxmaier, C. (2016). *Particle-in-Cell
Simulation of a Down-Scaled HEMP Thruster.* Trans. JSASS Aerospace Tech. Japan 14 (ists30), Pb_235-Pb_242.
**doi:10.2322/tastj.14.Pb_235** (Crossref work record fetched 2026-09-04: title, authors, container, volume/issue/pages, year; open-access
full text on J-STAGE read in full).

**Companion (geometry source):** Brandt, T. (2017). *Computer modeling for improvement of a High Efficiency Multistage Plasma Thruster.*
Dr. rer. nat. thesis, Kiel, URN `urn:nbn:de:gbv:8-diss-224024` (https://nbn-resolving.org/urn:nbn:de:gbv:8-diss-224024; full text read
2026-09-04, chapters 5-8). The paper states the channel, operating point, numerics and scalar results but NOT the magnet stack; the thesis
states the stack (chapters 6-7) and contains a second, independently run version of the same case (domain 19.12 mm, scaling factor 8,
I_a 4.7 mA, peak angle 60 deg) - the spread against the paper (4.3 mA, 50 deg) is used as the reference's own reproducibility in u_D.

**Why this one:** the only published HEMP PIC-MCC case near our scale (14 x 1.5 mm channel, 400 V, mA-class; ours 24 x 2 mm, 300 V) with an
open-access full text that states operating point, grid, time step, super-particle ratio, boundary conditions and scalar results in the text;
same code family (Greifswald PIC-MCC) as every other HEMP PIC paper; the stack is reconstructable from the thesis without author contact; the
case is affordable at device scale (section 6).

| Alternative | DOI | Why not |
| --- | --- | --- |
| Matyash et al. 2010, IEEE TPS 38(9) 2274 (DM3a, 51 x 9 mm, ~1 kV) | 10.1109/TPS.2010.2056936 | 8-30x our cell count per transit at Debye-resolving cells; stack, operating point and scalar results not tabulated (closed access) |
| Matthias et al. 2019, CPP 59(9) e201900028 (optimised downscaled CFT) | 10.1002/ctpp.201900028 | design point / field / numerics not in the abstract; it is our own lineage's design - tests the MDO chain, not the kernel |
| Kahnfeld et al. 2018, PSST 27 124002 (DM3a breathing) | 10.1088/1361-6595/aaf29a | time-dependent target needing neutral dynamics; DM3a scale |
| Lewerentz et al. 2022, Front. Phys. 10 833159 (MS4) | 10.3389/fphy.2022.833159 | different device class; not examined beyond the abstract |
| Keller et al. 2015, IEEE TPS 43(1) 45 (the same micro-HEMPT family, experiment) | 10.1109/TPS.2014.2321095 | experimental = validation v2's target; geometry given as ranges; Brandt 2016/2017 pin ONE configuration |

## 2. Extracted setup (paper page / thesis chapter; `text` = stated, `figure` = digitisation needed, `inferred` = derived by the stated rule)

| Item | Value | Kind | Source |
| --- | --- | --- | --- |
| Channel radius / length | 1.5 mm / 14 mm from the anode surface | text | paper Pb_235-236; thesis ch. 7 |
| Dielectric tube | Al2O3, r 1.5-2.5 mm (1 mm), eps_r 9; surface charge on r = 1.5 mm and on the top end z = 14 mm, 1.5-2.5 mm | text | paper Pb_237, Pb_239; thesis ch. 7 |
| Grounded body | 0 <= z <= 14 mm, 2.5 <= r <= 5.12 mm ("the magnets and their distance rings") | text | paper Pb_237 |
| Anode | z = 0, r <= 1.5 mm, 400 V (thesis: physical anode to 1.25 mm, boundary at anode potential to 2.5 mm) | text | paper Pb_237; thesis ch. 7 |
| Domain | 20.48 x 5.12 mm; 0 V on 14 <= z <= 20.48 at r = 5.12 and on z = 20.48 (thesis run: 19.12 mm) | text | paper Pb_236-237 |
| Mass flow | 0.27 sccm Xe = 1.1e17 atoms/s (17.6 mA equivalent) | text | paper Pb_236, Pb_240 |
| Neutral background | static DSMC import, mean ~2e20 m^-3 (profile 6e20 -> 1e20 along the channel), 500 K, diffuse walls, depletion (25 %) neglected | text | paper Pb_236; thesis ch. 7 + Fig. 7.3 |
| Cathode / neutraliser | real: W filaments at r ~ 40 mm outside the domain; model: uniform 1 eV volume source on the outer rim (17.55 mA, 11.7 mA lost at max r, "a third" of 5.85 mA reach the thruster = 1.95 mA); continuity value I_a - I_beam = 1.8 mA; ignition aid source for 1.5e6 steps | text / inferred | paper Pb_237, Pb_240 |
| Magnet stack | 3 SmCo rings (grade not stated), 5 mm long, r 2.5-15 mm, alternating; 5 soft-iron distance rings ("Carbon steel forgings, annealed"), 0.5 mm, r 2.5-8 mm; stack 18 mm (3 x 5 + 5 x 0.5 = 17.5: 0.5 mm discrepancy); anode at the mid-plane of magnet 1; no yoke in the FEMM model | text (thesis) | thesis ch. 6-7; paper Pb_236 + Fig. 1-2 |
| Field anchors | 0.6 T at (r 0, z 11 mm) "e.g." for the channel maximum; ~0.2 T and lower near the cusps; 0.05 T at (0, 17 mm); thesis: maximum ~0.7 T, exit null ~16 mm, interior nulls at the distance rings | text | paper Pb_236; thesis ch. 7 |
| Grid / time step | 1024 x 256 cells, 20 um, dt 3.17 ps (original-system units of a factor-4 (paper) / factor-8 (thesis) self-similar scaled run) | text | paper Pb_237; thesis ch. 7 |
| Steps | 2.4e7 total; ignition aid off after 1.5e6; quasi-steady at 76.12 us; averages over 1e6 steps | text | paper Pb_237-238 |
| Macro-particles | 1:2618 (scaled system), >= 6 per axis cell at the maximum density | text | paper Pb_237 |
| Collisions | e-n elastic / ionisation / excitation (LLNL EEDL), Coulomb, CEX; thesis adds Xe2+ | text | paper Pb_237; thesis ch. 7 |
| Anomalous transport | D_perp = 0.4 k T_e / (e B), perpendicular-velocity rotation | text | paper Pb_237 |
| Walls | dielectric surface charge in the Poisson solve (eps_r 9); SEE 50 % re-emitted at 90 % energy | text | paper Pb_237, Pb_239 |
| T_e / n_max estimates | 10 eV; 1e19 m^-3 | text (thesis) | thesis ch. 7 |

Reported quantities (the objects of the comparison; D and u_D in section 5): anode electron current 4.3 mA (exp. 4.5; thesis run 4.7); net
ionisation 24 % (= I_a / e Q_in by Brandt's definition; exp. 25 %); beam current 2.5 mA (exp. 3.1); plasma potential near the anode ~5 V above
the anode; cusp potential drops ~10 V (first) and ~5 V (second); n_i "mostly about 1e19" (figure); wall ion energy up to 160 eV and current
density 640 A/m^2 at the exit-side internal cusp (figures); plume peak 50 deg (5 deg bins; exp. 60; thesis run 60); ~200 eV electrons near
the exit cusp (figure); sheath 5-10 lambda_D; ionisation peaks upstream of each cusp; flat interior potential with the main drop in a "bulge"
beyond the exit.

## 3. Geometry mapping onto the parametric CFT geometry v1.1 (`geometry.py`)

Frames: the geometry model needs every magnet inside `[0, L]`; Brandt's magnet 1 straddles the anode. The FEM geometry therefore starts
2.5 mm behind the anode (`z_FEM = z + 2.5 mm`; the 2.5 mm "injector zone" = the unmodelled inlet section the paper mentions) and the PIC node
map evaluates the bound solution at the FEM coordinate. Geometry `brandt2016-micro-hempt-v1`, schema 1.1.0, sha256 in `protocol.json
geometry_mapping`.

| Item | Reference | Represented | Approximation |
| --- | --- | --- | --- |
| Channel radius / length | 1.5 / 14 mm | 1.5 mm / FEM chamber 16.5 mm = 2.5 mm inlet zone + 14 mm channel | A1 (offset; exact) |
| Magnets | 3 x 5 mm, r 2.5-15 mm, alternating | 3 x 5 mm, r 2.5-15 mm, centres 0 / 5.5 / 11 mm (anode frame), first + | - |
| Distance rings | 5 x 0.5 mm, r 2.5-8 mm, annealed carbon steel (nonlinear) | 2 interior x 0.5 mm at 2.75 / 8.25 mm, r 2.5-15 mm, linear mu_r 4000 | A2 (no end rings), A3 (ring radius = magnet radius), A4 (linear iron) |
| Dielectric | 1.0 mm | 0.9 mm in the FEM (clearance rule; mu_r 1 -> no effect on B); PIC front face 2.5 mm | A5 |
| Return yoke / housing | none in the FEMM model | mandatory role filled by a mu_r 1 placeholder | A6 |
| Remanence | SmCo, grade not stated | 1.05 T contract, post-scaled to the 0.6 T axis anchor (s = 1.0814 -> 1.135 T) | A7 |
| Magnetisation model | FEMM linear-demagnetisation SmCo | uniform axial remanence, recoil mu_r 1.05 | A8 |
| PIC box (primary) | 20.48 x 5.12 mm incl. 6.48 mm plume | channel-only 14 x 1.5 mm, 75 x 700 cells at 20 um, Dirichlet 0 V exit | A9 |
| PIC box (plume option) | 1024 x 256 cells | 1024 x 256 cells at 20 um, body dielectric 2.5 mm, plume radius 5.12 mm (mapping only, costed) | - |

Every reference dimension is a multiple of 20 um, so every grid line is hit exactly (worst snap 0 cells). The no-ring sensitivity solve
(distance rings at mu_r 1) brackets A2-A4 (section 4).

## 4. Field: material-aware P2 of the reconstruction (`fields.py`, `fields/brandt2016-micro-hempt-v1/`)

Level-0 padded solve (padding 1.0: FEM box r <= 33 mm, z_FEM -17.5..33.5 mm, covers both PIC boxes with the 0.75 mm margin), graded mesh
bore r_w/8, features/4: **415,859 P2 DOFs, 207,258 triangles, min angle 31.70 deg (0 elements below 10 deg)**; Jacobi-PCG 3706 iterations to
1.99e-10, 171 s, RSS 118 MB (peak working set 136 MB); one CPU worker, BLAS 1 thread; 371 s total incl. the sensitivity solve (185,193 DOFs,
54 s). Checkpoint bundle (LFS) `...domain-padding-1.00.level-0.json` (10.9 MB) + `.arrays.npz` (11.8 MB); binding hashes file `ec520ec85051`,
payload `e0c1663e59e9`, mesh `13de1f5a10d7`, run `bc8a8879e4f4`, sidecar `5d7b6c7f6eea` (full values in `binding.json`).

Predeclared gates on the published anchors (anode frame) and their outcome:

| Gate | Rule (predeclared) | Result | Pass |
| --- | --- | --- | --- |
| G1 scale | s = 0.6 T / \|B\|_nominal(0, 11 mm) in [0.80, 1.20] (SmCo remanence 0.84-1.26 T) | nominal 0.555 T -> s = 1.0814 (remanence 1.135 T) | yes |
| G2 interior nulls | exactly two axis nulls in 0 < z < 14 mm, each within +-0.5 mm of a ring centre (2.75, 8.25) | 2.634 and 8.366 mm (shifts 0.116 / 0.116 mm) | yes |
| G3 exit null | one axis null in 14 < z <= 20.48 within +-1.5 mm of 16 mm | 15.85 mm | yes |
| G4 exit point | \|B\|(0, 17 mm) = 0.05 +- 0.025 T | 0.053 T | yes |
| G5 axis maxima (REVISED) | local axis maximum within +-0.5 mm of each interior magnet centre (5.5, 11.0) and channel maximum 0.7 +- 0.07 T | maxima 0.698 T at 5.55 mm and 0.601 T at 11.15 mm | yes |
| D6 wall cusp field (DESCRIPTOR) | reported, not gated | wall \|B\| 0.41 T on both cusp planes (0.49 T max within +-0.5 mm); \|B\| <= 0.2 T inside r < 0.9 mm (60 % of the bore); mirror ratio wall / adjacent axis peak 0.59 | n/a |
| G7 mesh / solver / coverage | angle >= 5 deg, residual <= 2e-10, both PIC boxes covered | 31.7 deg, 1.99e-10, covered | yes |

**Gate genealogy (recorded in `binding.json gates.genealogy`).** The first solve was run against G5 "channel axis maximum at 11 +- 0.5 mm"
and G6 "wall |B| at each cusp plane in [0.10, 0.35] T". Both FAILED on that solve (maximum 0.698 T at the magnet-2 centre; wall field
0.49 T). Both were misreadings of the anchors: the paper's "maximum ... (e.g. at Z = 11 mm ...) is about 0.6 T" names an example point of the
maximum level, and the thesis gives the maximum itself as "about 0.7 T" - which the calibrated field reproduces at the interior magnet, i.e. a
second anchor consistent with the first under one scale; "near the magnetic cusps the flux is about 0.2 T and lower" describes the
low-field region around the axis null through which the electrons cross (our 0.2 T contour sits at r = 0.9 mm), not the wall - the wall cusp
field is published nowhere for this device, so it became a descriptor. The rules were revised before any PIC composition and with nothing
preregistered; the original rules and outcomes stay in the binding.

**No-ring sensitivity bracket (A2-A4):** interior nulls move 0.12 mm (2.634 -> 2.516, 8.366 -> 8.483 mm), the exit null 15.85 -> 16.13 mm,
|B|(0, 17 mm) 0.053 -> 0.036 T, the wall cusp field 0.49 -> 0.40 T, the axis maximum 0.698 -> 0.688 T, the 0.2 T contour 0.9 -> 1.0 mm. The
reference's saturating 8 mm rings lie between the two columns; the null positions (the cusp planes the comparison uses) are robust to 0.12 mm,
the wall field to ~20 %.

## 5. Comparison spec (ASME V&V 20 form; `comparison.py`, `comparison-spec.json`)

E = S - D; u_val = sqrt(u_num^2 + u_input^2 + u_D^2); statements at k = 2: `agreement_within_u_val` (|E| <= 2 u_val), `agreement_within_tolerance`
(2 u_val < |E| <= tolerance), `discrepancy` (|E| > tolerance, attributed first to the declared closure differences, never to the PIC kernel, until
the sensitivity variants have run). u_num is PREDICTED from the accepted 50 um convergence pair (seed-b <= 1.1 %; W x 0.7: I_d 5.7 %, S 4.6 %,
peak n_e 11.9 %, T_e,peak 9.3 %; taken as one standard uncertainty per quantity class); the grid band is NOT in u_num (the run is at the
published grid, 2.7 cells / lambda_D at 1e19 / 10 eV - between the v2.0.3 soft 2.5 and hard pi levels; the 15 um sibling turns the caveat into a
number). u_input is NOT propagated in v0 (B scale +-8 %, uniform-vs-profile neutrals, effective source +-0.15 mA): every row is conditional and
says so. A row whose 2 u_val already exceeds its tolerance cannot discriminate at the tolerance level and is flagged
(`tolerance_below_expanded_u_val`; the potential steps are such rows: u_D 2.5 V + u_num 2 V -> 2 u_val = 6.4 V > 5 V).

| Quantity | D | u_D (components) | u_num predicted | Tolerance | Comparable under | Our estimand |
| --- | --- | --- | --- | --- | --- | --- |
| anode electron current | 4.3 mA | 0.21 mA (precision 0.05; paper-vs-thesis 0.2) | 5.7 % | 20 % | channel, plume | trailing-window discharge current; CONDITIONED on the 1.8 mA injection |
| net ionisation fraction | 0.24 | 0.014 | 5.7 % | 0.05 abs | channel, plume | I_a / (e 1.1e17); consistency row (same number as above) |
| ion beam current | 2.5 mA | 0.26 mA (precision; declared 10 % boundary sensitivity) | 6 % | 20 % | channel, plume | exit-plane ion current (channel-only over-counts vs a boundary count: known sign) |
| beam fraction of feed | 0.142 | 0.0145 | 6 % | 0.03 abs | channel, plume | I_beam / (e Q_in) |
| plasma potential near anode above anode | 5 V | 2.5 V | 2 V (declared) | 5 V | channel, plume | n_e-weighted phi - U_a in the anode cell |
| potential drop, first cusp | 10 V | 2.5 V | 2 V | 5 V | channel, plume | n_e-weighted cell potentials across the anode-side cusp plane |
| potential drop, second cusp | 5 V | 2.5 V | 2 V | 5 V | channel, plume | across the exit-side cusp plane (exit cell pulled by the 0 V plane: caveat) |
| ion density, typical | 1e19 m^-3 | 0.27 dex (digitisation 0.18; 1e19 vs 4e18 internal 0.2) | 0.049 dex | 0.3 dex | channel, plume | log10 of the resolved trailing-window peak n_i (+ channel mean reported) |
| wall ion energy max | 160 eV | 16 eV | 9.3 % | 20 % | channel, plume | max of the wall ion mean-energy map |
| wall ion current density max | 640 A/m^2 | 64 A/m^2 | 12 % | 20 % | channel, plume | e x max wall ion flux |
| plume peak angle | 50 deg | 5.6 deg (bin 2.5; paper-vs-thesis 5) | 5.7 % | 10 deg | plume only | far-field ion current per sr - NOT compared in v0 |
| electron energy near exit cusp | 200 eV | 50 eV | 9.3 % | 25 % | plume only | NOT compared in v0 (the exit cusp at ~16 mm is outside the channel box) |

Qualitative rows (flat interior potential, cusp dips, ionisation upstream of the cusps, wall-flux localisation, sheath 5-10 lambda_D, radial
fill) and the nine declared closure differences (Bohm transport, SEE, neutral profile, electron source, exit boundary, the reference's
self-similarity scaling - lambda_D / L larger by 2.0 / 2.83 in its plasma -, collision set, dielectric in the Poisson solve, time to steady
state) are in `comparison-spec.json`.

## 6. Protocol (composed on `pic2d_cft_steady_state_v4/protocol.json`; `protocol.py`, `protocols/`)

**Grid: the published 20 um (75 x 700 channel cells = the reference's channel block; 1024 x 256 = its whole box).** At the published density /
temperature (1e19 / 10 eV, lambda_D 7.43 um) it reads 2.69 cells / lambda_D: under the v2.0.3 HARD gate (pi) and over the SOFT plateau
precondition (2.5), so a plateau at exactly the published state is recorded as "resolution margin not met" but the code-to-code numbers exist at
the same grid as the reference. The ss-v4 option (33.3 um, 45 x 420) is **inadmissible a priori**: 4.48 cells / lambda_D at the published
density - our own hard gate stops the run once the interval-averaged peak reaches ~4.9e18 at 10 eV; no comparison could form. 15 um / 0.5 ps
meets the soft margin (2.02) and is the resolution follow-up (composed as a third protocol; 2.5x the cost). The reference's own 20 um is the
original-system value of a self-similar scaled run in which the cell was ~1 lambda_D (thesis: dr / lambda_D = 0.95): it resolved its Debye
length by scaling, we resolve ours by the gate; neither run is better resolved than the other in its own frame (a declared closure
difference).

| Block | Primary `channel-20um` | Variant `channel-20um-bohm-0.4` | Follow-up `channel-15um` |
| --- | --- | --- | --- |
| cells / nodes | 75 x 700 / 76 x 701 (dr = dz = 20 um) | same | 100 x 933 / 101 x 934 (15.0 x 15.005 um) |
| dt | 0.70 ps (omega_pe dt 0.125 at 1e19, gate 0.2 at 2.56e19; Courant 0.42 at 400 eV; omega_ce dt 0.095 at the map's 0.772 T) | same | 0.50 ps |
| resolvable envelope (n_max) | hard peak-Debye pi at 10 eV: 1.36e19 (scales with T_e); soft 2.5 at 0.86e19 | same | 2.42e19 / 1.53e19 |
| operating point | anode 400 V, exit plane 0 V, STATIC uniform Xe 2e20 at 500 K (inventory removed), injection 1.8 mA / 1 eV, seed 5e16 / 5 eV (template) | same | same |
| closure | v1.3 physics, no recycling, no SEE, no anomalous transport | + v1.4 Bohm hook alpha 0.4 (nu_an = 0.4 omega_ce ~ D_perp 0.345 k T_e / e B; isotropic, a different model of the reference's coefficient) | as primary |
| gates | v2.0.3 verbatim: window-mode peak-Debye hard pi / soft 2.5 (400k-step windows = 0.28 us), one-sided windowed residual power 5 %, triad, omega_pe dt, Courant, Poisson; a-priori cell-Debye limit set to pi (disclosed; template 2.0 was its own ratio 1.0) | same | same |
| frames | ON, 40,000 steps = 28 ns (the template's time cadence), 149 frames to 3 transits | same | 40,000 steps = 20 ns |
| W | 82,466.8 (parity 9,600 would give 103 M particles at the declared mean 5e18 over 9.9e-8 m^3 -> raised to the 12 M cap; 229 macro-electrons per mid-radius cell at 1e19) | same | 82,466.8 (parity 5,402) |
| transit / plateau | 1.4 us (2.4 us x 14 / 24 mm); >= 3 transits = 4.2 us = 6.0 M steps; trailing-20 % drifts of I_d and N_e < 5 % (n_g static), triad soft, peak-Debye soft | same | 8.4 M steps |
| ms/step | 18.3 per process at H100 MPS-4 (8.71 anchor scaled by nodes + particles; solo-equivalent 7.1; 5090 model 10.0) | same | 19.8 (solo 7.7) |
| hours to 3 transits | **30.6 h at MPS-4** (11.9 h solo-equivalent); the reference's 76 us would be 554 h (not budgeted) | same | 46.3 h (18.0 h solo) |
| wall budget | 1.5x -> **46.0 h** (165,600 s), cumulative over resumes | 46.0 h | 69.5 h |
| device memory | 17.4 GB projected (12 M particles + 0.3 GB inverse blocks) | same | 17.8 GB |
| acceptance | (a) plateau, (b) windowed residual < +2 %, (c) comparison rows; verdicts comparison_quotable / comparison_resolution_flagged / plateau_with_heating / no_plateau | same | same |

Draft protocol hashes (sha256, 12) at `7fa9e6c6`: `channel-20um` `41dbe84e0eee`, `channel-20um-bohm-0.4` `23c8392a773b`, `channel-15um`
`77afa105c8a0`. The SEALED hashes (recomposed on the launch box at the preregistration commit, `protocol.json preregistration.sealed_run_protocols`)
differ from the drafts only by the blocks section 9.3 names (status, schema, `execution`, `wall_budget_basis`, `model_version`, the case id
without `-draft`); grid, dt, W, operating point, gates, frames, seed, transit, budget (46.0 h) and acceptance are the draft's.

## 7. Whole-set preflight (`preflight.py`, `preflight-channel-20um.json`; CPU, 14 s; 3/3 options PASS)

Per option: reference record (DOI validates), geometry (builds under v1.1, hash = binding), field binding (5 hashes re-verified, G1-G5 / G7
passed, gates recomputed from the bound checkpoint to the recorded scale), grid (worst snap 0 cells), field map (76 x 701 = 53,276 plasma
nodes sampled at z_FEM = z + 2.5 mm times 1.0814; max |B| 0.772 T; a-priori stability at 1e19 / 10 eV / 400 eV: omega_pe dt 0.125,
omega_ce dt 0.095, Courant 0.42, 2.69 cells / lambda_D <= pi, soft 2.5 NOT met and recorded), mesh masks (52,500 plasma cells), protocol
(`runner.build_config` accepts it with `neutral_inventory None`, MCC density 2e20, injection 1.8 mA / 1 eV, anode 400 V; the bohm variant's
`AnomalousCollisionConfig(alpha 0.4)` accepted), comparison spec (valid; 10 channel-comparable rows, 2 plume-only), cost row. The preflight
also carries the grid argument table (20 / 33 / 15 um). Platform: Windows, numpy 2.5.2, CPU only - a preregistered launch re-runs it on the
launch box (derived floats are platform-specific; `field_source_sha256` is the platform-independent binding).

## 8. What would make v0 inconclusive (declared before the run; `comparison-spec.json inconclusive_conditions`)

1. No plateau within the 46 h budget -> no S; trailing-window values reported as transient.
2. Hard peak-Debye (pi) or omega_pe dt stop -> the discharge densified beyond the 20 um / 0.7 ps envelope (interval-averaged peak > 1.36e19 x
   T_e / 10 eV; > 2.6e19 for omega_pe dt): resolution-limited at the published grid; the 15 um sibling is the route. This is the most likely
   failure: the reference's typical density is 1e19 and our closure (no Bohm transport, no SEE) confines better.
3. Plateau with the peak between the soft 2.5 and hard pi levels -> E formed, every row flagged "resolution margin not met".
4. Windowed residual power >= 2 % -> grid heating; E reported, not quotable.
5. No ignition under the frozen seed / injection -> recorded outcome; no relaunch with adjusted inputs.
6. A field gate fails -> the reconstruction, not the plasma model, is under test; the preflight refuses.
7. Every current / potential row discrepant in the direction the closure table predicts (more confinement in ours) -> the exercise measured the
   closure difference, not the kernels; the bohm-0.4 variant is the discriminating run.
8. The reference's own two runs already span a row's tolerance (the potential steps: 2 u_val 6.4 V > 5 V; I_a: 4.3 vs 4.7 mA) -> the row cannot
   discriminate at the tolerance level and is reported as such.
9. The plume angle and the exit-cusp electron energy are not compared in v0 (channel-only); no "plume validation" statement is available.

## 9. Preregistration record (channel-20um, Lambda H100, 2026-09-04 22:00-22:30 AEST)

### 9.1 What is sealed and how the launch is guarded

`protocol.json` (schema `cft.pic2d.external-validation-v0.protocol/1.0.0`, status
`preregistered_external_validation_v0_channel_20um_h100_mps4_code_to_code_brandt2016_not_validated`) binds by sha256: the three sealed run
protocols under `protocols/` (`channel-20um` = the launch set; `channel-20um-bohm-0.4` and `channel-15um` sealed, not launched), the
launch-box whole-set preflight `preflight-channel-20um.json` (with the GPU timing block), the shakedown `shakedown-channel-20um.json`, the
comparison spec and the field binding. `run.py launch --expect-commit <prereg sha> --require-mps` refuses: HEAD != the commit, a dirty
worktree, an option outside the launch set, `protocol.json` / the sealed protocol / the two records differing from HEAD's blobs, a
`protocol.json` whose status is not `preregistered*`, a preflight that did not pass every option or whose launch-box timing did not pass
(budget must cover the measured 3-transit wall), a shakedown that did not pass, a missing MPS pipe directory, a recomposition on the launch
platform that differs from the sealed bytes, an existing `execution-lock.json` (O_EXCL, in `results/channel-20um/`). The scheduler job is
`ext-val-v0-channel-20um` in `tools/cloud/jobs.yaml` (detached worktree at the preregistration commit, `merge-base --is-ancestor` + byte-identical
protocol checks, Warp UUID cross-check, the MPS client variables).

### 9.2 Records produced on the launch box (`ubuntu@68.209.75.2`, H100 80GB HBM3 `GPU-a800b021-6364-473f-5177-cd6ae7ce0005`, driver 580.105.08, Python 3.12.14, numpy 2.5.2, warp-lang 1.14.0)

* **Whole-set preflight with GPU timing** (`preflight-channel-20um.json`, `run.py preflight --gpu-timing --timing-steps 2000`, 12:06-12:07 UTC
  at code `183e32a8`, all 3 options PASS): the CPU gates of section 7 re-run on the box (field binding re-verified through `field_source_sha256` `0562cb3ffd97...` -
  the platform-independent binding; the sampled map's content hash is Linux-specific and provenance only - mesh 52 500 plasma cells, dt 0.7 ps
  admitted at the map's 0.772 T (omega_ce dt 0.095), node map 76 x 701, comparison spec, cost row), then the GPU timing of the primary option
  on the real field: host factorisation 1.2 s (7.1 s in the first attempt under CPU contention), **seed load 5.12 ms/step** (60 000 seed
  electrons; 5.58 in the first attempt) and **plateau load 12.54 ms/step** (6.0 M electrons + 6.0 M ions = the 12 M-particle cap, a synthetic
  uniform 5e18 seed; 13.07 in the first attempt) over 2000 production steps after 200 warm-up steps, measured with **4 other CUDA-MPS clients
  active** (the three mini-sweep runs + another agent's GPU job; 5 in the first attempt). Projection: 6.0 M steps to 3 transits -> **20.9 h at
  the measured plateau-load rate** (budget / 3-transit wall 2.20); the cost model's MPS-4 projection is 30.6 h at 18.3 ms/step, so the
  measured box is FASTER than the model even under heavier-than-MPS-4 contention and the budget basis stays the (slower) cost model: 46.0 h.
  The per-process rate will fall further toward the solo rate (~7 ms/step by the model) as the sweep runs finish - these hours are upper bounds
  for the contention present at the measurement.
* **Shakedown** (`shakedown-channel-20um.json`, `run.py shakedown` at code `183e32a8`, 12:07-12:16 UTC: the primary option on its real field,
  100 000 steps at the shrunk cadences 200 / 4000 / 40 000 / frames 2000, the FULL production path run -> `assess` -> `compare` -> re-finalize,
  as the fifth MPS client (4 others: the three sweep runs + another agent's job) for 522 s at 5.20 ms/step): stop `target_steps_reached`,
  50 frames, 69 562 electrons + 78 642 ions at 0.07 us (the plasma is still the seed transient; the earlier run at `42e30aaa` replayed these
  counts exactly), windowed residual-power window complete in 280 records (last -16.6 %, cooling side, as every accepted seed window), the
  window-mode peak-Debye statistic computed in all 500 records but ENFORCED in none (0 resolved nodes: at W 82 467 on 20 um nodes no node
  reaches the 32-macro-electron window occupancy in a 0.07 us transient; at the published 1e19 a mid-radius cell holds 229 macro-electrons,
  so the gate becomes live once the local density passes ~1.4e18 m^-3 - the production run must show `resolved_nodes > 0`, the v2.0.2 lesson,
  recorded in `gate_not_inert_check`), `assess` -> `no_plateau` (expected at 0.07 us), `compare` formed all **10 channel-comparable rows**
  (non-evidentiary numbers of a transient: I_a 1.86 mA, potential drops 71 / 226 V, peak n_i 5.0e18, wall ion energy 228 eV - they mean
  nothing; the stage is exercised), then the externally-stopped path (`finalize --allow-refinalize`) ran on the scratch directory (1.6 s;
  maps downgraded to instantaneous as designed).
* **Defect found and fixed before the freeze - model v2.0.4 (`79e6a670`, tests `test_pic2d_v204_omega_pe_gate.py`).** The FIRST box
  preflight (11:39 UTC, code `05e8d68b`) died in the plateau-load timing: `observed peak omega_pe*dt = 0.212 exceeds 0.2` before the first
  timed step, and the first shakedown read a single-step peak density of 5.5e18 with 60 000 electrons spread over 53 000 nodes (mean 5e14).
  Cause: the runtime omega_pe dt gate read the peak over EVERY plasma node of the single-step electron deposit; on a 20 um axis node
  (V = pi (dr/2)^2 dz = 6.3e-15 m^3) ONE macro-electron of W 82 467 reads 1.3e19 m^-3 (omega_pe dt 0.14 at 0.7 ps) and two read 0.20 - a
  shot-noise extreme value decided by the smallest node (the plume-boundary lesson of 04:20 the same day), which would have ended the
  production run as a spurious "omega_pe dt stop" (inconclusiveness condition 2) long before any physical densification. v2.0.4: the gate
  statistic is the peak over the RESOLVED nodes (single-step deposit >= the peak-Debye gate's own floor, 32 macro-electrons), the raw
  single-node peak is recorded alongside (`peak_omega_pe_dt_raw` in every series record, `v1_4_options.omega_pe_dt_gate` in the provenance);
  CPU and Warp backends, warp-cpu parity tested; the physics is untouched (a run that never trips replays bitwise). The window-mode
  peak-Debye hard gate (pi = 1.36e19 at 10 eV on this grid, interval-averaged, 32-particle floor) binds first and stays the protective
  density gate; the floored omega_pe dt gate is the fast-transient backstop on resolved nodes. Consequence for the triad: its
  `omega_pe_dt_drift` member now reads the resolved statistic (the mini-sweep runs, locked at `291a9227`, keep the raw one).
* **Slot / contention disclosure.** Design 056 of the mini-sweep stopped on its grid-heating triad gate at 10:52 UTC (omega_pe dt drift
  0.283 > 0.25 at 2.07 transits, exit 0, `finished: true`; the sweep is NOT assessed here - that happens after all four), so one of the four
  MPS slots was free before this preregistration: the shakedown and the timing ran in that slot (a fourth client, never a fifth), and the
  launch enters it while the reference, 047 and 009 runs continue. Other agents' GPU jobs (the steady-state v5.1 shakedown, a profiling
  job) were active on the same GPU during the timing (5-6 clients in total) - the rates above are therefore pessimistic for the four-slot
  configuration. No Xid event was raised (`dmesg | grep Xid` clean).

### 9.3 Decisions vs the draft (`protocol.json preregistration.decisions_vs_draft`)

| item | draft (17:44 AEST) | preregistered | why |
| --- | --- | --- | --- |
| launch box / GPU model | H100 MPS-4 assumed for the cost row only | NVIDIA H100 80GB HBM3, one of four CUDA-MPS slots, recorded in every sealed protocol (`execution`) with the launch discipline and the scheduler job | the mini-sweep record (`291a9227`) put the GPU model, the slot and the shared-GPU caveats into the preregistration |
| slot | launch when 047 frees (~01:00 AEST) or solo after the sweep | the slot 056 freed at 10:52 UTC; three sweep runs active at launch | a slot became free earlier; still a four-client configuration |
| shakedown GPU use | a launch-box shakedown before the freeze | ran as the fourth MPS client for ~8 min, twice (before and after the v2.0.4 fix) | the sweep budgets are MPS-4 upper bounds; a fourth client costs them nothing beyond their declared configuration |
| wall budget | 46.0 h = 1.5 x the cost-model MPS-4 projection | 1.5 x max(cost model, MEASURED plateau-load rate), 10 min rounding, basis recorded (`stopping_rule.wall_budget_basis`); measured 12.5-13.1 ms/step < model 18.3 -> **46.0 h unchanged** | a preflight timing at production load beats a cost-table extrapolation; the measured rate may only raise the budget |
| gates | v2.0.3 verbatim | v2.0.3 thresholds unchanged; the omega_pe dt STATISTIC is the v2.0.4 resolved-node peak | the launch-box finding above |
| seed / frames / operating point / grid / dt / W / transit / acceptance / comparison spec | as drafted | unchanged (comparison-spec.json byte-identical: `543c81fcff01...`) | fixed before any run |
| launch stages | `launch` refused unconditionally | `preflight --gpu-timing`, `shakedown`, `launch --expect-commit --require-mps`, `status`, `finalize` | the v4 / mini-sweep discipline |

### 9.4 What the run will and will not settle

The two things most likely to make the comparison inconclusive, as section 8 declares them: (i) a **hard peak-Debye stop** - the interval-
averaged peak crossing pi cells / lambda_D (1.36e19 x T_e / 10 eV on 20 um) because our closure (no Bohm transport, no SEE) confines better than
the reference's; then the run is resolution-limited at the published grid and only the 15 um sibling can form E; (ii) **no plateau within the
46 h budget** (or a plateau sitting between the soft 2.5 and hard pi levels, which flags every row "resolution margin not met"). A third,
structural one: rows whose 2 u_val already exceeds the tolerance (the potential steps: 6.4 V > 5 V; I_a: the reference's own 4.3 vs 4.7 mA)
cannot discriminate whatever the run does, and every current / potential row discrepant in the closure-predicted direction measures the closure
difference, not the kernels (the bohm-0.4 variant is the discriminating follow-up).

## 10. Launch 1 outcome (`channel-20um`): STOPPED by the grid-heating triad gate at 0.52 transits - genuine finite-grid heating, INCONCLUSIVE

Record: `results/channel-20um-launch1-triad-gate-stop/` (v4 contract: summary with the frames manifest, execution lock, run state,
status/series/maps, final checkpoint metadata + sha256 sidecars, the sealed protocol copy; `assessment.json` / `comparison.json` from
`run.py assess` / `compare` executed in the job worktree at `3dc12cf6`; `triad-stop-diagnosis.json` with the analysis script embedded).
Bytes verified against the box's `sha256sum`. Frames (26), particle checkpoint, field anchor, `series.jsonl` and logs stay on the box
(`/lambda/nfs/h100-files/cft/jobs/ext-val-v0-channel-20um/`). The canonical `results/channel-20um/` is free for a future execution.

**Stop.** PID 31588, launched 12:26:44 UTC 2026-09-04, exited 13:56:18 UTC, `grid_heating_triad_gate_stopped_run` at step **1,040,000 =
0.728 us = 0.52 transits** (of 1.40 us), 26 frames, 5,366 s wall (1.49 h) at 5.8 ms/step as the fourth MPS client, exit 0, `finished: true`, no
finalization error, no Xid. State at the stop: I_d 2.74 mA (rising +11 % over the last 0.2 us), N_e 882 k / N_i 890 k macro-particles
(1.77 M of the 12 M cap), S 2.4-2.7e17 /s, T_e,dense 9.3 eV, window-mode peak 5.89e18 at 8.4 eV (2.26 cells / lambda_D, soft ok),
phi_max 466 V (+66 V above the anode), `assess` -> **`no_plateau`** (drifts I_d +8.6 % / N_e +42 % at 0.52 < 3 transits), `compare` ->
10 channel rows formed, **`quotable: false`**.

**Which member fired, and why it could fire at 0.52 transits.** The v2.0.3 **windowed residual-POWER member**: trailing-400,000-step
ledger residual / electrode work = **+0.0743 >= 0.05** (2.12e-8 J over 2.85e-7 J). That member is enforced from the FIRST complete
400,000-step window (0.28 us) by design, independent of the transit arming (`enforced_after_transit_times` 1.0) that gates the three
DRIFT members - which read S +0.58, resolved omega_pe dt +0.32, T_e,dense +0.19 at the stop (all past the 0.25 hard bound) but were
**not enforced** (`enforced: false`, 0.52 < 1.0 transit; the S member crossed 0.25 at 0.42 transits and would have stopped the run
there under a 0.5-transit arming). The arming rule worked as declared; the physics-protection member is the one that stopped the run.

**Verdict: (a) genuine finite-grid heating** (`triad-stop-diagnosis.json verdict`), not an ignition-transient artefact and not shot noise:

* **The gate fired LATE, not early - a ledger omission found during this diagnosis.** The energy ledger's `inelastic_loss_j` is the
  per-MACRO-event count times the threshold energy (`mcc.py` MCC tally, `warp_backend.py` flush) and lacks the macro-weight W, while
  every other ledger term carries W. Verified on this run: the particle-side identity dKE = field work + injected - absorbed + born -
  W (n_exc E_exc + n_ion E_ion) e closes to 4.5e-14 J per record (sum -9.4e-11 J on 1.6e-7 J); the recorded `interval_residual_j` equals
  H - L_inel to 4.5e-14 J per record, where **H = field work + dU_field - electrode work is the true energy created by the field-particle
  coupling** and L_inel the omitted W-scaled inelastic sink (2.63e-7 J cumulative here = 40 % of the electrode work). Recorded
  `inelastic_loss_j` / unscaled count = 1.0003 here and 1/W exactly (26 660, 26 649, 26 561, 59 986) on ss-v4, 047, 056-L1, attempts 7/8.
  Every recorded residual is biased NEGATIVE by the inelastic power. For this run H / electrode work per 28-ns frame: -0.7 % (0.03 us),
  +1.5 % (0.14), **+5.2 % (0.20)**, +11.8 % (0.31), +25.3 % (0.45), +50.6 % (0.59), +79.3 % (0.67), **+111.9 % (0.728 us)** - 1.22 W of
  numerical heating against 1.09 W of electrode power; cumulative +32 %. The trailing-400k gate reading corrected for the omission
  crossed 5 % at step 480,000 (0.34 us, 0.24 transits) while the recorded statistic still read -22 %; it crossed 5 % on the recorded
  statistic only at step 1,040,000.
* **Debye statistics recomputed on the 28-ns frames** (exact interval averages, 32-macro-electron mean-occupancy floor): resolved peak
  cells / lambda_D 1.19 (0.31 us) -> 1.67 (0.45) -> 2.08 (0.53) -> 2.44 (0.64) -> **2.85 (0.728 us)**; 0 resolved nodes above pi, **1,508
  above the soft 2.5** (8.7 % of the electron inventory), all in the near-axis column r 0.12-0.30 mm, z 3.2-7.3 mm (between the first
  and second nulls). The column's AXIS (r < 0.12 mm) is unresolved by construction - an axis node (V = 6.3e-15 m^3) holds 0.76
  macro-electrons at 1e19 and W 82,467; the first radial node reaching the floor at 1e19 is i = 6 - and there the frame-averaged n_i is
  1.07-1.34e19, n_e 0.8-1.0e19 at 9 eV: **2.9-3.3 cells / lambda_D, at or past pi**, invisible to the window-mode hard gate (whose
  0.28-us average of a 0.24-us-doubling density read 2.26). The final-checkpoint snapshot puts 9.5e18 in the core with a Maxwellian
  electron distribution (slope fit 9.0 eV = moment T_e 9.0 eV; 1.2 % above 50 eV): the statistic is not beam-inflated. The column's
  20-um cells hold 2.6 (axis) to 58 macro-electrons - at 8.6x the parity weight the stochastic (1/N_c) heating precedes the CIC aliasing
  threshold, which is why H turned positive at a resolved 1.2-1.7 cells / lambda_D.
* Corroboration: T_e,dense 7.5 -> 9.3 eV (+24 %) over the last 0.20 us; the electron kinetic energy rose 70 -> 118 nJ over the last
  0.17 us while the integrated H over those 0.17 us was 136 nJ - more than the gain (the numerical energy is spent on inelastic collisions -
  S doubled - and at the walls); I_d ROSE (2.46 -> 2.74 mA), so the plateau signature "T_e up while I_d down" does not apply to a discharge
  still igniting. (c) is excluded because the member is a 400,000-step ledger integral, not a single-node statistic; (b) is excluded because
  the accepted runs' ignition windows read -12 ... -25 % on the same (biased) statistic and never exceeded +0.4 % at any time.

**Where the discharge was heading vs the reference (transient values of the last 240,000-step window, non-evidentiary; `comparison.json`).**
I_a 2.61 mA vs 4.3 (E -1.69 mA, tolerance 0.86); net ionisation 0.148 vs 0.24; I_beam 0.81 vs 2.5 mA; anode-cell potential +16.4 V above
the anode vs ~5; cusp drops -1.3 / +41.0 V vs ~10 / ~5 (the mid-radius potential falls 30-40 V at the second null, 8.4 mm; the axis stays
at 410-429 V to z = 12 mm; the whole ~385 V acceleration sits in the last 1 mm before the 0 V exit plane, A9); resolved n_i 9.3e18 vs
~1e19 (the one row "within u_val"); wall ion energy 381 vs 160 eV; wall ion current density 439 vs 640 A/m^2. Trend: I_d +10 %, N_e +63 %,
S x1.8, T_e,dense +21 % over the last 0.17 us; ion production 43 mA-equivalent (S 2.7e17 /s = 2.5x the reference's 1.1e17 /s feed under the
static background) against 2.7 mA of ion losses -> the ion inventory doubles every 0.24 us with no saturation in sight. **Peak density vs
the 20 um hard-pi envelope** (1.36e19 x T_e / 10 eV = 1.25e19 at the frame's 9.1 eV): resolved frame peak 1.03e19 = 82 %; axis n_i
1.3-1.6e19 = 105-132 % (past it, unresolved); the gate's window statistic 5.9e18 = 52 % (lag). The hard-pi stop would have followed
within ~0.1 us; the avalanche would have carried the column past the 15 um envelope (2.42e19 at 10 eV) as well.

**Classification under section 8:** condition 4 (windowed residual power >= 2 % -> grid heating; not quotable), condition 1 (no plateau;
trailing-window values reported as transient), and condition 2 in substance (the discharge densified past the 20 um / W 82,467 envelope
in the unresolved axis column before the hard-pi gate could see it: resolution-limited at the published grid). Calibration disclosure:
the v2.0.3 residual-power gate was calibrated on the accepted PLATEAUS (recorded -12.7 % -> -0.2 %, i.e. on a statistic biased by the
omitted inelastic loss), was never exercised on an ignition transient at 2e20 / 400 V, and this protocol declares no ignition gate; with
the omission corrected it should have fired ~0.4 us earlier. Nothing about the drift members' arming needs to change for this run.

**Decision: no launch 2 at 20 um** (a genuine-heating stop is not a gate artefact; the README's own rule sends it to the resolution
follow-up). The sealed **15 um sibling** (`channel-15um`: 100 x 933, dt 0.5 ps, W 82,466.8 unchanged, 8.4 M steps to 3 transits, 19.8 ms/step
at MPS-4 -> 46.3 h, 18.0 h solo-equivalent, **budget 69.5 h**, 17.8 GB) is **not recommended as sealed**: at the same W it holds 0.42x the
macro-particles per cell (the axis cell would hold ~1 at 1e19), so the stochastic heating that started this run's H gets worse, and the
avalanche exceeds its 2.42e19 envelope too. What a conclusive v0 needs, in order: (i) model v2.0.6 - `inelastic_loss_j` x W in both backends,
tests, and a re-calibration of the residual-power gate on the accepted runs with the corrected statistic (end-state estimates: v2 base 50 um
~+13 %, seed-b ~+12.6 %, W x 0.7 ~+8.4 % - the 50 um plateau was heating at the 5 % level; ss-v4 33 um ~+1.9 %, 047 ~+2.6 %, 056-L1 ~+0.7 %;
attempt 8's last window ~+80 %); (ii) a peak-Debye floor in accumulated particle-steps (the v2.0.2 plume-gate design) so that the axis
column is gated; (iii) for THIS operating point either the parity weight (103 M particles - beyond the cap and ~100 ms/step) or a weaker
closure: the sealed `bohm-0.4` variant (the reference's own D_perp; the discriminating run of section 8.7) confines less and is the one
sealed option that can plausibly reach a resolvable plateau at 20 um - it needs (i) and (ii) first, then its own amendment, box preflight
and shakedown. Static 2e20 neutrals with a confining closure avalanche (retained lesson); the reference's static-DSMC steady state may not
be reachable without its anomalous transport.

## 11. Energy-ledger correction (model v2.0.6, post hoc; recorded values unchanged)

Up to model v2.0.5 the energy ledger's `inelastic_loss_j` lacked the macro weight W (found by the external-validation v0 launch-1 diagnosis, 036bd679), so every recorded interval residual was `H - L_inel` - biased NEGATIVE by the inelastic power - where `H = field work + dU - electrode work` is the true numerical energy creation. The sidecar(s) `ledger-corrected.json` (+ `.sha256.json`) were written by `python -m cft_revival.pic2d.ledger_recompute <results-dir>` from the recorded `series.npz` (corrected residual = H per record; `spec/pic2d/pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6`); **the recorded series, maps and summaries are unchanged.** Values below: trailing-400 000-step residual / electrode work at the last record, recorded -> corrected.

`results/channel-20um-launch1-triad-gate-stop/ledger-corrected.json`: **+7.4 % -> +61.7 %** at the stop (0.728 us; cumulative -7.8 % -> +31.9 %;
+627 mW of numerical heating on 1.02 W of electrode power in the last window); the corrected statistic crossed the 5 % gate at the 0.34 us
checkpoint (0.24 transits; +6.1 %) and 2 % at 0.31 us, against the recorded stop at 0.73 us; the cross-check against the final counts is exact
to 9.1e-5. The section-10 finding is implemented as model v2.0.6: (i) `inelastic_loss_j` x W in both backends with the identity tests and the
recalibration of the residual-power gate on the corrected statistic (thresholds kept at 5 %: the accepted 33 um plateaus peak at +2.5 %, the 50 um
plateaus were heating at 7-13 %), and (ii) the peak-Debye floor in accumulated macro-electron-steps (`min_accumulated_macro_particle_steps_at_peak`,
64 000): on this run's end-state map the near-axis nodes the occupancy floor excluded become resolved (8 843 -> 42 373 nodes; the gated node moves
from (8, 214) at 35.6 macro-electrons per step to (6, 215) at 27.6), the axis densest node (0.72 per step, 172 000 macro-electron-steps) is
resolved and reads 2.18 in the 240 000-step window average - the window lag on a 0.24-us-doubling density remains a disclosed limitation. Route
item (iii) (parity weight or the `bohm-0.4` closure) is unchanged.

### 11.1 Corrected-ledger re-read of the recorded acceptance (2026-09-05)

| item | recorded (`assessment.json` / `comparison.json`, section 8–10) | corrected ledger (`ledger-corrected.json`) |
| --- | --- | --- |
| stop | `grid_heating_triad_gate_stopped_run`: windowed residual power **+7.4 %** ≥ 5 % at step 1 040 000 = 0.728 µs (0.52 transits) | the same stop, read correctly: **+61.7 %** of the electrode work in the trailing window at the stop (+627 mW on 1.02 W); the corrected statistic crossed 5 % at the 0.34 µs checkpoint (0.24 transits) and 2 % at 0.31 µs — the gate saw the heating ~0.4 µs late because the recorded statistic was biased by the omitted inelastic power (−54 % of the electrode work in the last window) |
| (a) plateau | `no_plateau` (0.52 transits) | unchanged |
| (b) residual power < +2 % | FAIL (+7.4 %) | **FAIL (+61.7 %)** — status unchanged, magnitude 8× |
| comparison rows vs Brandt 2016 | 10 rows evaluated on a transient, **not quotable** | unchanged: not quotable; the run was an ionisation avalanche under our closure with true numerical heating from 0.2 µs |
| verdict | **INCONCLUSIVE** (section 8 conditions 4, 1, 2) | **stands** |

Nothing recorded is modified; the sidecar hash-binds the recorded `series.npz`. The section-10 diagnosis ("genuine finite-grid heating
seen late") is confirmed and quantified by the correction. Any future launch (parity weight, `bohm-0.4`) executes model ≥ v2.0.6 and
reads the corrected statistic natively.

## 12. Amendment 1 — `channel-20um-bohm-0.4` LAUNCH 2 of the campaign (model v2.1.0 closure, v2.0.6 gates; 2026-09-05)

**What and why** (`protocol.json` `amendments[0]`, mirrored in the sealed run protocol). Section 10's route to a conclusive v0 was
(i) the v2.0.6 ledger fix, (ii) the accumulated-particle-step Debye floor, then (iii) the sealed `bohm-0.4` variant — the one sealed
option that can plausibly reach a resolvable plateau at 20 µm because the reference's own D⊥ coefficient confines less. (i) and (ii)
landed (`4b53012d`, `8c70cff0`). The physics completeness audit (`0901138a` §4.c) then found that the sealed variant used the v1.4
**isotropic** redirect, which also randomises v∥ — a bracket of Brandt's model, not the model: Brandt et al. 2016 (p. Pb_237) rotate
"only the component of the velocity vector perpendicular to the local magnetic field direction … to ensure that the speed of the
electrons along the magnetic field lines does not change". Model v2.1.0 (`cft_revival.pic2d.sensitivity`,
`tests/pic2d/test_pic2d_v210_anomalous_transport.py`) implements that event (`bohm_perpendicular_rotation`) in both backends and
verifies D⊥ = (kT_e/eB) α/(1 + α²) by a diffusion test. For a code-to-code comparison the reference's own event model is the right
closure to seal, so the amendment switches the option to it before its first execution.

**Changes to the sealed `channel-20um-bohm-0.4` protocol** (`3ec0d405…`-era hash `b1a0d3b5…` → **`1aaa080d41cd…`**):
`numerics.anomalous_collisions = {model: bohm_perpendicular_rotation, alpha: 0.4}` (ν_an = 0.4 ω_ce — the D⊥ coefficient read as a
rate, the natural mapping of a selection probability that depends on |B| only; exact Green-Kubo D⊥ = 0.345 kT_e/eB disclosed as before;
ν_an Δt = 0.034 at 0.7 T); `numerics.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak = 64000` (v2.0.6: the near-axis
column launch 1 densified past π unresolved is gate-able); `numerics.performance.moment_sample_interval = 5` (v2.0.5); the W-corrected
ledger is code (acceptance (b) and the 5 % gate read the corrected statistic natively); `LAUNCH_SET = (bohm-0.4, 20um)` with the base's
one execution on record (`LAUNCH_HISTORY`; no launch 2 of the base at 20 µm, section 10's rule); `model_version` / `classification` /
`case.id` / `simplifications` text; config identity `a5ac7611af82…`. **Unchanged:** grid 75 × 700 at 20 µm, Δt 0.7 ps, **W 82 466.8**
(the 12 M-particle cap — parity would be 103 M particles, beyond the cap; section 10 names the parity weight as the alternative to a
weaker closure, not a lower W within the cap, so the 8.6× parity weight stays a declared limit of this run), seed 20260903, the operating
point, frames, the v2.0.3 thresholds (hard π / soft 2.5, 5 % residual power, triad drift bounds with their **1.0-transit arming** —
"nothing about the drift members' arming needs to change", section 10), the v2.0.4 ω_pe Δt statistic, the plateau rule, acceptance
(a)–(e), the comparison spec (byte-identical) and the inconclusiveness conditions. The base (`3ec0d405…`) and 15 µm (`7a6b7139…`)
sealed protocols are byte-identical to `3dc12cf6`.

**Launch-box preflight** (`preflight-channel-20um-bohm-0.4.json`, `preflight --variant bohm-0.4 --grid 20um --gpu-timing`,
16:56:52 UTC 2026-09-04, code `c1508c06`, the 4th CUDA-MPS client beside ss25-base, sweep-056-launch2 and ss33-fast): whole-set CPU
gates 3/3 options PASS; GPU timing PASS — factorisation 1.3 s, **2.71 ms/step at the 60 000-electron seed, 7.22 ms/step at the 12 M-particle
plateau load** (6.0 M e⁻ + 6.0 M i; 2.62 GB) → 12.0 h to 3 transits (6.0 M steps). Faster than the cost model (18.3 ms/step, 30.6 h) and
than launch 1's six-way-shared measurement (13.07), so by the budget rule the cost-model basis stands: **budget 165 600 s = 46.0 h**
(3.8× the measured 3-transit wall).

**Shakedown** (`shakedown-channel-20um-bohm-0.4.json`, `shakedown --variant bohm-0.4 --grid 20um`, 16:57–17:02 UTC, code `c1508c06`,
4th MPS client): 100 000 steps at shrunk cadences from the real field and seed, 2.81 ms/step, 283 s, 50 frames, `target_steps_reached` at
0.07 µs; the **accumulated-floor peak-Debye window enforced with 10 506 resolved nodes** (launch 1's shakedown could not enforce it — the
mean-occupancy floor resolved nothing at this load), windowed residual complete (−0.03 %, W-corrected); `assess` → `no_plateau` (b True);
`compare` → 10 channel rows formed, `quotable: false` (0.07 µs transient, numbers mean nothing); re-finalize path OK; **passed**. The
shakedown ran on the composition without the `LAUNCH_BOX_TIMING_BOHM` block (filled from this preflight before the amendment commit);
the sealed protocol differs from it only in `execution.launch_box_timing` / `budget_external_validation_v0.launch_box_timing` (outside
`config_sha256`; identity unchanged).

**Expected outcome (declared).** The inconclusiveness conditions of section 8 stay in force — the 20 µm / W 82 466.8 envelope (hard π at
1.36e19 × T_e/10 eV) may still be reached if the closure does not bound n_e enough. Discriminating outcomes: (i) a plateau under the
reference's closure with I_a toward 4.3 mA and n_i toward 1e19 (rows inside V&V20 tolerance, or recorded misses), which would put the
remaining differences on SEE / the neutral profile / W parity (audit R2, R5a); (ii) another heating or envelope stop, which would say the
reference's static-DSMC steady state is not reachable at this W with its transport closure alone.

**Launch log.** Launch 2 enters an MPS slot after the anomalous-transport α-series' first job (`at-alpha-1over16`, 17:08 UTC) as the
scheduler job `ext-val-v0-channel-20um-bohm-0.4` (jobs.yaml) — details appended here at the launch.

- **LAUNCH 2 — `channel-20um-bohm-0.4`: PID 49403, 19:07:11 UTC 2026-09-04 (05:07 AEST 5 Sep)**, `schedule.py launch --only
  ext-val-v0-channel-20um-bohm-0.4` by the box slot-waiter (`r1-queue`) into the slot `at-alpha-1over16` freed at 19:01 UTC; detached worktree
  `jobs/ext-val-v0-channel-20um-bohm-0.4/tree` at `a1065ce4`; 4th MPS client beside ss25-base (32709), sweep-056-launch2 (38282), ss33-fast
  (44430); 2.95 → 3.9 ms/step at the seed load; `results/channel-20um-bohm-0.4/`. Budget end 19:07 + 46.0 h ≈ 17:07 UTC 6 Sep.

## 13. Decision on the triad drift-member arming for this campaign (2026-09-05 06:30 AEST; model v2.1.1)

The α-series launch 1 (`pic2d_anomalous_transport_v1`, α = 1/16 on the 33 µm reference plateau) stopped at exactly 1.00 transit on the
triad's drift members — the run had EXTINGUISHED under the closure (its README §9) — and exposed that the drift members' 1.0-transit arming
(`enforced_after_transit_times`, this protocol's §3 / §12 "unchanged") was calibrated on α = 0 plateaus only. Model v2.1.1 (`e47ae78a`, spec
`triad_drift_arming_v2_1_1`) arms them by a settled-once latch (≥ 2 transits AND the I_d drift has read < 5 % at a checkpoint) and requires an
`ignition_gate` beside it; the α-series adopted both by its amendment 1 (`33be2a89`).

**This campaign is exposed to the same rule in principle**: the sealed `channel-20um-bohm-0.4` protocol (`1aaa080d41cd…`, §12) arms the
drift members at 1.0 transit = 1.40 µs = 2 000 000 steps and declares no ignition gate (§10: "nothing about the drift members' arming needs
to change for this run" was written for launch 1, which the residual-power member stopped at 0.52 transit). Launch 2's state at the decision
(0.43 µs, 0.31 transit): N_e 5.7e4 → 3.9e4 macro, sitting at 0.9× its 0.05–0.2 µs reference (neither launch 1's avalanche — N_e ratio 3.46
at 0.5 µs — nor yet a decay); I_d drift +0.01, S drift +0.11, residual window not yet complete.

**Decision (recorded, nothing changed on the running process or its sealed protocol):**

1. Launch 2 runs to its own outcome under the sealed 1.0-transit arming. It is never signalled (Xid-31 lesson) and its protocol is not amended
   while it runs.
2. If it is stopped by the drift members at ≈ 1.40 µs WITH a discharge that is re-equilibrating (I_d, N_e, S moving toward a state, residual
   < 5 %, peak below π — the α-series' "(b)" reading), a **launch 3 is preregistered under an ext-val amendment 2** that adopts the v2.1.1
   `drift_members_arming` block verbatim and an `ignition_gate` calibrated for THIS seed transient (20 µm / 2e20 / 400 V; the 33 µm bounds of the
   α-series do not transfer: launch 1's N_e ratio at 0.5 µs was 3.46 where the 33 µm accepted runs read 1.1) from launch 1's and launch 2's
   series; identities unchanged (stopping-rule keys), the comparison spec untouched.
3. If it is stopped by the drift members with a DECAYING discharge (N_e falling from its reference, the injected current returning — the
   1/16 extinction signature), or by the residual-power / peak-Debye members, or reaches its plateau or budget, the stop is the recorded outcome
   (§8 conditions apply) and no amendment follows.
4. `tools/cloud/jobs.yaml` keeps the job at `a1065ce4` (running); the α-series' `r1-queue` waiter continues behind it.

## 14. Launch 2 outcome (`channel-20um-bohm-0.4`): STOPPED by the T_e,dense drift member at 1.26 transits — a gate ARTEFACT on a marginal, sustained low-density discharge; INCONCLUSIVE; no launch 3 now (2026-09-05 10:45 AEST)

Record: `results/channel-20um-bohm-0.4-launch2-triad-gate-stop/` (v4 contract: summary with the frames manifest, execution lock, run state,
status / series / maps, final checkpoint metadata + sha256 sidecars, the sealed protocol copy; `assessment.json` / `comparison.json` from
`run.py assess` / `compare --results-dir`; `ledger-corrected.json` from `ledger_recompute` as the witness that the ledger is natively
W-corrected (`already_w_scaled: true`; recorded +0.250 %, H recomputed +0.252 %); `triad-stop-diagnosis.json` with the analysis script
embedded). Bytes verified against the box's `sha256sum`. Frames (63), the particle checkpoint (5.9 MB), the field anchor and `series.jsonl`
stay on the box (`/lambda/nfs/h100-files/cft/jobs/ext-val-v0-channel-20um-bohm-0.4/`). The canonical `results/channel-20um-bohm-0.4/` is
free for a future execution.

**Stop.** PID 49403, launched 19:07:11 UTC 2026-09-04, exited 21:18:54 UTC, `grid_heating_triad_gate_stopped_run` at step **2 520 000 =
1.764 µs = 1.26 transits** (of 1.40 µs), 63 frames, 7887 s wall (2.19 h) at 3.1 ms/step as the fourth MPS client, exit 0, `finished: true`, no
Xid. Member: **`t_e_dense_drift −0.328`** (hard 0.25); the S member read +0.001, the resolved ω_pe Δt member was **undefined** (`None`: no node
ever held ≥ 32 macro-electrons). The drift members had armed at 1.0 transit (step 2 000 000, the sealed rule of §12); this was the 14th
40 000-step checkpoint after arming — the T_e,dense member had random-walked between −0.21 and +0.22 at the previous 13.

**Classification (heating / re-equilibration / extinction / artefact, `triad-stop-diagnosis.json#verdict`):**

* **The STOP is (c) an ARTEFACT of an undefined statistic.** In the trailing 20 % of the run the densest node held **0.24–1.5 macro-electrons
  (median 0.62)**: the whole inventory is 5.55e4 macro-electrons over 52 500 plasma cells — about one per cell at W 82 467 — so the
  single-step "dense node" is a one-particle node whose density reads 4–8e18 (one macro-electron on a 20 µm node, the v2.0.4 lesson) and whose
  temperature is **0 to round-off in 73 % of the records** (median 4.6e-16 eV, p95 1.5 eV, max 33 eV). The member is the trailing-20 % drift of
  that noise. Its sibling, the ω_pe Δt member, carries the v2.0.4 occupancy floor and was correctly undefined; the T_e,dense member has no
  floor (code follow-up, not a physics finding). Under the model v2.1.1 latch (≥ 2 transits AND I_d settled) none of the drift members would
  have been armed at 1.26 transits.
* **The DISCHARGE is (b) a RE-EQUILIBRATION to a marginal, sustained LOW-DENSITY state — not (d) extinction.** N_e 5.75e4 (seed) → 4.6e4
  (0.05 µs) → **3.87e4 at 0.3 µs** (the minimum; 0.91 of the 0.05–0.2 µs reference at 0.5 µs) → 4.1e4 (0.5) → 5.0e4 (1.0) → 5.45e4 (1.4, arming)
  → **5.55e4 at the stop** (ratio 1.27, still rising at +1.5 % per trailing window); N_i 5.3e4 → 7.07e4; S/S_ref 1.50 ± 0.03 from 0.5 µs on;
  trailing drifts I_d +0.7 %, N_e +1.5 % (inside the plateau threshold at 1.26 < 3 transits, so no declaration). Neither the α-series'
  extinction signature (N_e monotone down from the seed, the injected current returning through the exit plane — here **0.19 of the 1.80 mA
  returns**) nor launch 1's avalanche (N_e ratio 3.46 at 0.5 µs, 13.4 at 0.7 µs).
* **Not (a) heating**: windowed residual power **+0.25 %** of the electrode work (natively W-corrected; maximum over complete windows +0.30 %;
  cumulative +0.14 %); accumulated-floor peak **0.53 cells/λ_D** (trailing mean 0.51; ~30 000 resolved nodes at 2.6–3.6 mean macro-electrons;
  the 4.48 in the window statistics is the seed-deposit record before the first complete window). Numerically the cleanest run of the
  campaign.

**Where the discharge sat (trailing 400 000-step window; `window_currents_a`).** Injected 1.80 mA → anode electron current **1.75 mA** (0.19 mA
returns through the exit plane): the injected beam reaches the anode almost entirely instead of being trapped. Ionisation S = 1.53e16 /s =
2.45 mA-equivalent → wall electrons **2.28 mA** / wall ions **2.28 mA**, ion beam **0.097 mA**, anode ions 0.035 mA; the balances close to the
+0.04 mA of slow inventory growth. Peak n_e (window, resolved) 2.35e17 at T_e 6.1 eV; resolved n_i 4.7e17, channel mean 7.7e16. The anomalous
hook ran at **ν_an = 0.4 ω_ce at ⟨|B|⟩ 0.52–0.60 T** over the inventory (4.2e10 events per electron per second; field maximum 0.77 T) — the
electrons sit in the strong-field region and the leak carries the beam to the anode; a weak, wall-loss-dominated discharge.

**vs Brandt 2016 (`comparison.json`, 10 channel rows formed, `quotable: false` — a 1.26-transit run without a plateau, §8 condition 1; the
rows say where the model sits, they are not a comparison result):** I_a **1.71 mA vs 4.3** (E −2.59 mA against the 0.86 mA tolerance; 0.40×);
net ionisation fraction 0.097 vs 0.24; I_beam 0.097 vs 2.5 mA (0.04×); resolved n_i 4.7e17 vs ~1e19 (−1.32 dex against 0.3; 0.05×);
near-anode potential +9.7 V above the anode vs ~5 (the one row inside u_val); cusp drops 17 / 93 V vs ~10 / ~5; wall ion energy 329 vs 160 eV;
wall ion current density 65 vs 640 A/m². Launch 1 (α = 0) sat at 2.6 mA / 9.3e18 and avalanching at its 0.52-transit heating stop; launch 2
(α = 0.4) sits at 1.7 mA / 5e17 and marginally sustained: the reference's transport coefficient turns the avalanche into a weak discharge, and
neither is the Brandt discharge.

**Verdict: INCONCLUSIVE** (§8 condition 1; conditions 2 and 4 do NOT apply — this run was numerically clean). **The model with Brandt's
transport coefficient (ν_an = 0.4 ω_ce, perpendicular rotation) does not sustain the Brandt discharge at their density either, at this grid /
W: it settles at ~40 % of the anode current and ~5 % of the peak density.** Remaining gaps, all outside this protocol: **SEE** from the
dielectric (absent — the audit's rank-2 effect; the R2 shakedown doubled I_d with SEE(BN) on the reference design), **spatial neutrals** (static
uniform 2e20 here vs the reference's DSMC / Knudsen profile with its anode-side maximum — R5), **W parity** (8.6× the parity weight → ~1
macro-electron per cell at this density, which is also what made the stop).

**Decision: NO launch 3 now; the ext-val launch 3 waits for the full-physics model.** §13 rule 2 (a drift-member stop with a re-equilibrating
discharge → launch 3 under an amendment 2 with the v2.1.1 latch + an ignition gate) is formally satisfied, but a launch 3 with this identity
replays bitwise to 1.26 transits and continues toward the same weak state (I_a 1.7 mA, n_i ~5e17, N_e growing +1.5 % per window): the plateau
it would declare (~10–12 h of GPU) quantifies a miss by 2.5× in I_a and 20× in density, not a validation, and the miss cannot be attributed to
the closure while SEE and the neutral profile are absent. The right launch 3 is a **Brandt-geometry case of the full-physics model**
(`pic2d_full_physics_v1`, `b45f6728`: SEE(BN) + Knudsen spatial neutrals + Coulomb + this closure, all in code) — composed as its own
preregistration, sealed at **particle parity** (at n_e ~ 1e17–1e19 a W of ~1e4 keeps the count inside the 12 M cap on the low-density branch and
removes the one-particle-per-cell statistics that ended this run; if the discharge densifies toward 1e19 the cap binds and the 33 µm-style
Debye margin must be re-derived) with the **v2.1.1 arming latch + an ignition gate calibrated on this seed transient** (N_e ratio 0.91 at 0.5 µs
is the sustained branch's reading; launch 1's 3.46 the avalanche's). Nothing of launch 2 is relaunched; its record is the campaign's launch-2
outcome. `tools/cloud/jobs.yaml` job `ext-val-v0-channel-20um-bohm-0.4` remains at `a1065ce4` and cannot relaunch (execution lock present) —
disable it when the jobs file is next edited.

    $env:PYTHONPATH="$PWD\src;$PWD"; $env:OMP_NUM_THREADS='1'
    python -m experiments.pic2d_external_validation_v0.run reference          # the reference record
    python -m experiments.pic2d_external_validation_v0.run fields             # P2 solve (+ no-ring sensitivity), ~6 min, RSS <= 150 MB
    python -m experiments.pic2d_external_validation_v0.run regate             # recompute the gates from the bound checkpoint
    python -m experiments.pic2d_external_validation_v0.run preflight          # whole-set preflight (CPU gates) -> preflight-channel-20um.json
    python -m experiments.pic2d_external_validation_v0.run compose            # comparison-spec.json + protocols/*.json + protocol.json (run on the launch box before the freeze)
    python -m experiments.pic2d_external_validation_v0.run cost
    python -m experiments.pic2d_external_validation_v0.run protocol --variant base --grid 20um --with-field
    # GPU, launch box: the preregistration records, then the one preregistered launch (via tools/cloud/schedule.py)
    python -m experiments.pic2d_external_validation_v0.run preflight --gpu-timing --timing-steps 2000   # + launch_box_timing block
    python -m experiments.pic2d_external_validation_v0.run shakedown                                    # run -> assess -> compare -> re-finalize -> shakedown-channel-20um.json
    python -m experiments.pic2d_external_validation_v0.run launch --expect-commit <prereg sha> --require-mps
    python -m experiments.pic2d_external_validation_v0.run status|assess|compare [--results-dir results/channel-20um]
    # labelled development runs only (never evidence)
    python -m experiments.pic2d_external_validation_v0.run run --allow-launch --shrunk-cadences --max-steps 100000 --label development

Tests: `tests/pic2d/test_pic2d_external_validation_v0.py` (21: reference DOI / record / u_D budgets; geometry under the v1.1 contract, frames,
approximations, PIC mapping at 20 / 33 um and the plume box; comparison-spec schema + metric incl. the log-scale and non-discriminating rows;
protocol composition on the v4 template with static neutrals, dt / W / budget policies incl. the measured-rate basis, the bohm variant, the grid
argument, the cost row; field binding gates + genealogy + bracket, node map offset / scale / refusals, regate; preflight + `timing_passed`; run /
launch guards incl. the launch-set and clean-worktree refusals; the shrunk-cadence protocol; sealed protocols = recomposition) and
`tests/pic2d/test_pic2d_v204_omega_pe_gate.py` (5).

## Launch log

* 2026-09-04 17:44 AEST - DRAFT prepared (`645c7de4..7fa9e6c6`): reference, geometry, field (CPU), comparison spec, protocols, CPU preflight;
  no GPU.
* 2026-09-04 21:30-22:30 AEST (11:30-12:30 UTC), Lambda H100 `68.209.75.2`, code `05e8d68b` -> `183e32a8`: the preregistration stages
  (`preflight --gpu-timing`, `shakedown`, `launch`, `status`, `finalize`) landed; the first box preflight exposed the omega_pe dt shot-noise
  defect -> model v2.0.4 (`79e6a670`); the records of section 9.2 produced (preflight + timing, shakedown, compose) on the box in the slot
  design 056 had freed, as the fourth MPS client (other agents' GPU jobs made it the fifth or sixth at times; disclosed).
* **2026-09-04 22:26:44 AEST (12:26:44 UTC) - LAUNCH 1 of `channel-20um`**, preregistration commit **`3dc12cf6d3a299c7c3702a1b2c349d69ffe1ddde`**
  (launch config `7697ce9f`: `tools/cloud/jobs.yaml` job `ext-val-v0-channel-20um`, slots_per_gpu 4, MPS client variables), Lambda
  `gpu_1x_h100_sxm5` `ubuntu@68.209.75.2`, GPU 0 = NVIDIA H100 80GB HBM3 `GPU-a800b021-6364-473f-5177-cd6ae7ce0005`, driver 580.105.08, CUDA MPS
  server 14519, 6 BLAS threads. Launched by `tools/cloud/schedule.py launch --only ext-val-v0-channel-20um` (`plan` clean: prereg commit an
  ancestor of HEAD `7697ce9f`, sealed protocol frozen) into the slot design 056 freed: detached worktree at `3dc12cf6` under
  `/lambda/nfs/h100-files/cft/jobs/ext-val-v0-channel-20um/tree` (LFS smudged), Warp `cuda:0` UUID == nvidia-smi (`gpu_uuid_match true`),
  `launch --expect-commit 3dc12cf6d3a299c7c3702a1b2c349d69ffe1ddde --require-mps` passed every check of section 9.1 and acquired the O_EXCL
  `execution-lock.json` (commit `3dc12cf6d3a2`, protocol `3ec0d405520f`, clean worktree, MPS pipe `/tmp/nvidia-mps`, 4 compute apps on the GPU
  at launch = the MPS server + the three sweep runs). **PID 31588** (wrapper tmux `pic-ext-val-v0-channel-20um`), stepping at **4.1-4.3 ms/step**
  at the seed load with the three sweep runs active (reference PID 19764 at 2.45 transits, 047 PID 20079 at 2.85, 009 PID 20189 at 2.52),
  970 MiB GPU memory at 0.07 us (projected ~17 GB at the 12 M-particle plateau load), whole GPU 100 %. Ignition from the 5e16 seed under the
  static 2e20 background and the 1.8 mA injection: N_e 48 000 -> 70 000 and S 2.8e16 -> 6.7e16 /s over 0.01 -> 0.07 us, I_d 1.0-2.1 mA,
  windowed residual -11 to -14 % (cooling side), resolved omega_pe dt 0.000 (no node at the 32-macro-electron floor yet; raw single-node peak
  0.7-1.1e19 = the shot-noise extreme the v2.0.4 statistic no longer gates on). No Xid. Expected wall: 6.0 M steps to 3 transits (4.2 us) at
  the seed-load rate would be ~7 h; at the measured 12 M-particle rate under 4-5 clients (12.5-13.1 ms/step) ~21-22 h; the step cost grows with
  the particle count toward the cap and falls as the sweep slots empty (047 within the hour, reference and 009 within ~2 h -> then solo, ~7
  ms/step by the model), so the 3-transit / plateau verdict is expected between ~06:00 and ~20:00 AEST 2026-09-05 (the plateau rule may
  declare later, up to the budget); budget end = 46.0 h cumulative = ~20:30 AEST 2026-09-06 if never resumed. Monitoring:
  `tools/cloud/schedule.py status` on the box (`gpu` samples are whole-GPU readings under MPS); do NOT kill the process (Xid-31 lesson).
  When it stops: from the job worktree `run.py assess` then `run.py compare` (results/channel-20um), a results-only record commit
  (`chore(pic2d)`), the comparison table into section 10 of this README and the roadmap canvas row `validation-v0-v2`.
* **2026-09-04 23:56:18 AEST (13:56:18 UTC) - launch 1 STOPPED by the grid-heating triad gate at 0.52 transits - genuine finite-grid
  heating (verdict (a)), INCONCLUSIVE, no launch 2 at 20 um.** `grid_heating_triad_gate_stopped_run` at step 1,040,000 = 0.728 us, 26
  frames, 5,366 s wall, 5.8 ms/step, exit 0, `finished: true`, no Xid (PID 31588, lock `3dc12cf6` / `3ec0d405520f`). Member: the v2.0.3
  windowed residual-POWER gate, +0.0743 >= 0.05 over the trailing 400,000 steps (enforced from the first complete window by design; the
  drift members S +0.58 / omega_pe dt +0.32 / T_e,dense +0.19 were NOT enforced at 0.52 < 1.0 transit). Diagnosis (section 10,
  `results/channel-20um-launch1-triad-gate-stop/triad-stop-diagnosis.json`): the ledger's `inelastic_loss_j` lacks the macro-weight, so the
  recorded residual is H - L_inel; the true non-conservation H = field work + dU - electrode work was +5 % of the electrode work at 0.20 us,
  +25 % at 0.45 us and +112 % (1.22 W) at the stop, while the gate statistic still read -22 % at 0.34 us - the gate fired ~0.4 us late.
  Frame-resolved peak 2.85 cells / lambda_D (1,508 nodes above 2.5, none above pi), the unresolved axis column (0.76 macro-electrons per
  axis node at 1e19) at 1.1-1.3e19 -> 2.9-3.3; the column's cells hold 2.6-58 macro-electrons at 8.6x the parity weight. Discharge at the
  stop: I_a 2.6-2.7 mA (ref 4.3), I_beam 0.8 mA (2.5), n_i 0.9-1.6e19 (~1e19), anode-cell potential +16 V (~5), S 2.7e17 /s = 43 mA-
  equivalent against 2.7 mA of ion losses (inventory doubling 0.24 us) - an avalanche under the static 2e20 background. `assess` ->
  `no_plateau`; `compare` -> 10 rows, `quotable: false`. Classification: section 8 conditions 4, 1 and (in substance) 2. Record commit:
  results-only (v4 contract + assessment / comparison / diagnosis), `.gitignore` negations, this section; the 15 um sibling (69.5 h) is not
  recommended as sealed (same W -> fewer particles per cell; the avalanche exceeds its envelope too); recommended route in section 10.
  `tools/cloud/jobs.yaml` job `ext-val-v0-channel-20um` remains enabled but cannot relaunch (execution lock present) - disable it when the
  jobs file is next edited. Slot freed at 13:56 UTC (the scheduler's four slots: ss25-base, sweep-056-launch2, sweep-reference, one free).
* **2026-09-05 05:07:11 AEST (2026-09-04 19:07:11 UTC) - LAUNCH 2 of `channel-20um-bohm-0.4`** (amendment 1, section 12; commit `a1065ce4`,
  sealed protocol `1aaa080d41cd`, config `a5ac7611af82`): details in section 12's launch log (PID 49403, r1-queue slot-waiter, fourth MPS client).
* **2026-09-05 07:18:54 AEST (2026-09-04 21:18:54 UTC) - launch 2 STOPPED by the grid-heating triad's `t_e_dense_drift` member (-0.328) at
  1.26 transits - a gate ARTEFACT on a marginal, sustained low-density discharge; INCONCLUSIVE; no launch 3 now (section 14).**
  `grid_heating_triad_gate_stopped_run` at step 2,520,000 = 1.764 us, 63 frames, 7,887 s wall, 3.1 ms/step, exit 0, `finished: true`, no Xid.
  The member read the trailing drift of an undefined statistic (the densest node held ~0.6 macro-electrons; T_e,dense 0 to round-off in 73 %
  of the trailing records; the sibling omega_pe dt member was correctly undefined under its v2.0.4 floor); the discharge itself had
  re-equilibrated to a weak sustained state (N_e 3.87e4 at 0.3 us -> 5.55e4 at the stop, S/S_ref 1.50 flat, I_d / N_e drifts +0.7 % / +1.5 %),
  numerically clean (residual +0.25 %, 0.53 cells / lambda_D). Where it sat: I_a 1.71 mA (ref 4.3), I_beam 0.10 (2.5), resolved n_i 4.7e17 (~1e19),
  injected 1.80 mA -> anode 1.75 mA, ionisation 2.45 mA-equivalent -> walls 2.28 / 2.28 mA. `assess` -> `no_plateau`; `compare` -> 10 rows,
  `quotable: false`. Record commit: results-only (v4 contract + assessment / comparison / ledger witness / diagnosis) under
  `results/channel-20um-bohm-0.4-launch2-triad-gate-stop/`, `.gitignore` negations, section 14. The slot freed at 21:18 UTC went to the
  alpha-series `at-alpha-1over64` (21:28 UTC; that run extinguished - its README section 10).
