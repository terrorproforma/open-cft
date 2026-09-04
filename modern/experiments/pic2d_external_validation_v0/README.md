# pic2d external validation v0 - DRAFT (code-to-code vs Brandt et al. 2016; NOT preregistered, NOT launched)

**Status: DRAFT.** Everything in this directory is preparation for the roadmap's "External validation v0 - code-to-code vs Brandt 2016"
step (`LITERATURE_SYNTHESIS.md` 5.7 / 7a; `paper/evidence/result-gates.json` GATE-L3 stays closed): the reference case extracted from the
paper and its companion thesis, the reconstruction of its magnet stack on the parametric CFT geometry v1.1, the material-aware P2 field with
published-anchor gates, the run protocols composed on the steady-state v4 template at the published 20 um resolution, the ASME V&V 20
comparison spec, the whole-set preflight and the tests. No GPU was used. Nothing is hash-bound to a run; `run.py launch` refuses. The
coordinator preregisters (this record + a launch-box preflight + a labelled shakedown, committed) and launches from that commit. The claim
ceiling of the whole exercise is **cross-model agreement** (the reference is a published model output, `EvidenceKind.PUBLISHED_EXTERNAL`):
it validates nothing against hardware and opens no physics level.

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

Draft protocol hashes (sha256, 12): `channel-20um` `41dbe84e0eee`, `channel-20um-bohm-0.4` `23c8392a773b`, `channel-15um` `77afa105c8a0`
(`protocol.json preregistration.sealed_run_protocols_draft`; they are drafts, not sealed).

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

## Commands (from `modern/`; CPU unless stated)

    $env:PYTHONPATH="$PWD\src;$PWD"; $env:OMP_NUM_THREADS='1'
    python -m experiments.pic2d_external_validation_v0.run reference          # the reference record
    python -m experiments.pic2d_external_validation_v0.run fields             # P2 solve (+ no-ring sensitivity), ~6 min, RSS <= 150 MB
    python -m experiments.pic2d_external_validation_v0.run regate             # recompute the gates from the bound checkpoint
    python -m experiments.pic2d_external_validation_v0.run preflight          # whole-set preflight -> preflight-channel-20um.json
    python -m experiments.pic2d_external_validation_v0.run compose            # comparison-spec.json + protocols/*.json + protocol.json
    python -m experiments.pic2d_external_validation_v0.run cost
    python -m experiments.pic2d_external_validation_v0.run protocol --variant base --grid 20um --with-field
    # GPU (coordinator): labelled shakedown, then the preregistered launch after the freeze
    python -m experiments.pic2d_external_validation_v0.run run --allow-launch --shrunk-cadences --max-steps 100000 --label shakedown
    python -m experiments.pic2d_external_validation_v0.run assess|compare --results-dir results/channel-20um

Tests: `tests/pic2d/test_pic2d_external_validation_v0.py` (18: reference DOI / record / u_D budgets; geometry under the v1.1 contract, frames,
approximations, PIC mapping at 20 / 33 um and the plume box; comparison-spec schema + metric incl. the log-scale and non-discriminating rows;
protocol composition on the v4 template with static neutrals, dt / W / budget policies, the bohm variant, the grid argument, the cost row; field
binding gates + genealogy + bracket, node map offset / scale / refusals, regate; preflight; run / launch guards; draft protocols = recomposition).
