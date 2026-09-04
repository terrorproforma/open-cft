# pic2d external validation v0 - code-to-code vs Brandt et al. 2016 (PREREGISTERED option `channel-20um`, Lambda H100)

**Status: PREREGISTERED (section 9); one execution on the Lambda H100 from the preregistration commit.** This directory is the roadmap's
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

## Commands (from `modern/`; CPU unless stated; on the box `PYTHONPATH=src:.` with the MPS variables exported)

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
