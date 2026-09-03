# PIC-MCC blockers of the cusped-field / HEMP thruster simulation: a cited literature review

**Status: literature review (document only). No code, spec, protocol or result was changed.**
Prepared 2026-09-03 against `feat/sota-foundation` at `96220ffc` for the `cft_revival.pic2d`
workstream (`modern/docs/workstreams/pic2d-devlog.md`, `pic2d-learning-scratchpad.md`,
`pic2d-campaign-v1-proposal.md`, `modern/spec/pic2d/pic2d-model-v1.3.json`,
`modern/experiments/pic2d_cft_steady_state_v2/{README.md,protocol.json}`).

## 0. Scope, method and honesty rules

- Every reference below was verified on 2026-09-03 against the Crossref work record
  (`api.crossref.org/works/<DOI>`), the OpenAlex record, or the publisher/repository landing
  page; title, authors, venue, year and DOI (or handle) are quoted as recorded there. Abstracts
  were read where the index exposed them; two full texts were read (Brandt et al. 2016, open
  access; Barnes and Chacon 2021, arXiv 1910.10833). Where a claim about a paper rests only on
  its abstract that is said. Nothing that could not be verified is cited; several candidate
  references (theses without a persistent handle, IEPC papers without DOIs, conference talks)
  were dropped for that reason and the bibliography is smaller than the field. The two theses
  that are cited (Szabo 2001; Gildea 2009) were verified on DSpace@MIT by handle.
- IEPC conference papers have no DOIs. Only one is cited, already carried by the repo
  (`modern/docs/REFERENCES.md`): Kornfeld, Koch and Harmann IEPC-2007-108. The ISTS 2017-b-32
  paper is likewise cited only as the repo carries it.
- "Silent" means: no verified paper found that addresses the point for HEMP/CFT PIC; it does
  not mean no such paper exists.
- The downstream paper the user cited, Yeo, Gadisa, Ogawa and Bang, *Aerospace Science and
  Technology* 154, 109516 (2024), doi:10.1016/j.ast.2024.109516, is about **field emission
  electric propulsion**, not cusped-field PIC. Its abstract (retrieved from the KAIST Pure
  record; the publisher page refused automated access) describes "multi-objective design
  optimization ... by incorporating electrostatic simulation coupled with an analytical
  performance model into evolutionary algorithms based on prediction from surrogate modeling",
  finds that emitter-tip length controls plume divergence, and that "sensitivity analysis has
  identified the mass flow rate and potential distributions as the most influential design
  factors". It is relevant here only as the same group's methodology (surrogate-assisted MDO
  plus a physics simulation used as a verification layer), which is the pattern of Yeo et al.
  2020, Yeo and Ogawa 2022 (surrogate-assisted MDO of the cusped-field thruster with an
  "improved power balance model" and magnetic simulation; anode current found most influential)
  and Matthias et al. 2019 for the cusped-field thruster. It is not a PIC or validation
  source and is not used as one below.
- The Greifswald group's own review of HEMP-T modelling (Kahnfeld et al. 2019, *Reviews of
  Modern Plasma Physics*) is closed access and no abstract is deposited; it is listed because
  it is the canonical entry point to that code family (Tskhakaya et al. 2007 describe the
  method; Schneider et al. 2009 and Matyash et al. 2010 the thruster application; Taccogna et
  al. 2008 collect the Bari/Greifswald Hall-thruster PIC-MCC models), but no claim below rests
  on its content.

### 0.1 Our PIC state in one paragraph (what the literature is being asked about)

2-D axisymmetric (r,z) explicit electrostatic PIC-MCC; Boris push with bilinear cylindrical
weighting (momentum-conserving); exact block-Thomas cylindrical Poisson solve on the device;
prescribed static B from the qualified P2 field; kinetic electrons and Xe+; null-collision MCC
in the Vahedi and Surendra 1995 form (Biagi v7.1: elastic, one lumped 8.32 eV excitation,
12.13 eV ionisation); quasi-steady 0-D
neutral inventory with an artificial 30 ns relaxation; absorbing dielectric wall with surface
charge; Dirichlet 0 V exit plane with 3 mA / 2 eV fixed-current electron injection; anode
300 V; ion subcycling k = 8; Warp CUDA on an RTX 5090 (Windows WDDM), ~2 ms/step at 1 M
macro-particles. Development plateau at 3.2 ion transits: I_d 3.44 mA, beam fraction 0.67,
utilisation 46 %, n_g 2.97e19 m^-3 on the analytic fixed point, mean n_e 2.13e17 m^-3,
**peak n_e 1.64e18 m^-3 = 4.1 x the a-priori ceiling with Dz = 3 lambda_D at the peak**;
seed-b agrees to <= 1 %; W x 0.7 running. Static neutrals gave avalanche (nu_iz tau_i > 1) or
no ignition.

### 0.2 The closest published analogues (read these first)

| analogue | geometry / operating point | what it shares with us | key numbers reported |
| --- | --- | --- | --- |
| Brandt et al. 2016 (Greifswald code, DLR/Airbus micro-HEMPT) | (r,z) explicit ES PIC-MCC; channel Z_thr = 14 mm, R_thr = 1.5 mm; 400 V anode; 0.27 sccm Xe (~1.1e17 atoms/s); grid 1024 x 256 cells over 20.48 x 5.12 mm (Dx = 20 um); static DSMC neutral background ~2e20 m^-3; 2.4e7 steps to 76 us; super-particle ratio 1:2618 (>= 6 per axis cell) | almost our channel (24 mm x 2 mm bore, 300 V, 0.19 sccm, 3 mA); same code family as Matyash/Kalentev/Kahnfeld; dielectric surface charge; simple SEE (50 % re-emitted at 90 % energy); Bohm-type anomalous diffusion D = 0.4 kT_e/eB imposed by velocity rotation; electron source at far boundaries at 1 eV; additional ignition source switched off after 1.5e6 steps | anode electron current 4.3 mA (measured 4.5 mA); net ionisation 24 % of feed (measured 25 %); beam current 2.5 mA (experiment 3.1 mA); potential flat near the anode, drops ~10 V and ~5 V at the two internal cusps, main drop at the exit cusp; sheath 5-10 lambda_D; ion energies to 160 eV at the exit-side internal cusp; plume peak 50 deg (exp. 60 deg); n_i ~1e19 m^-3 so lambda_D ~10 um = Dx/2; Dirichlet -> Neumann outer boundary changed the plume ratios; "thrust oscillations ... breathing modes ... an increase of the simulation time by about one order of magnitude is needed" |
| Liu et al. 2015 (HIT) | PIC-MCC of a CFT with channel radii 6, 4 and **2 mm** at fixed length and operating parameters | our bore radius | at 2 mm the cusps "cannot confine electrons very well", the electric field is "hard to establish", more electrons leak along the centreline to the anode, leak width grows; stronger B partly compensates |
| Zhao et al. 2013 (HIT) | PIC-MCC of a CFT at three B strengths, "convergent results" | same class of device | density peak moves toward the axis with B; main potential drop near the exit; potential steps at separatrices; wells/barriers near the wall |
| Matthias et al. 2019 (Greifswald + Yeo/Ogawa) | PIC of one design from the 0-D power-balance MDO (the lineage of this repository) | our MDO -> PIC chain | PIC used to check the MDO-predicted performance; 0-D power-balance diagnostics computed from PIC; PIC outputs fed back into the 0-D model |
| Yeo et al. 2020 (JSR, with Matthias/Kahnfeld/Schneider) | PIC of selected MDO design points of the downscaled CFT | our lineage | "prediction errors associated with uncertainties such as the beam current and divergence angle have been reduced to within 5 %" |
| Kahnfeld et al. 2018 (Greifswald) | fully kinetic PIC-MCC of the HEMP-T DM3a | same code family, same device class | breathing mode in the 100 kHz range "during stable operation"; compared with a 0-D predator-prey estimate and with a measured discharge current |
| Gildea 2009 (MIT DCFT, MS thesis) | full PIC of the diverging cusped-field thruster | cusp fields ~0.5 T | ps time steps required; the full-field run "has yet to be accomplished"; results at 1/5 B only |

---

## 1. Blocker 1: Debye resolution at the density peak (Dz = 3 lambda_D; grid heating)

### (a) Key references

- Birdsall and Langdon 1991 (book); Hockney and Eastwood 1988 (book): the explicit
  momentum-conserving scheme needs Dx <~ lambda_D and omega_pe Dt <~ 0.2; grid heating otherwise.
- Langdon 1970; Hockney 1971; Birdsall and Maron 1980; Ueda et al. 1994: the classic
  measurements of aliasing/self-heating and its saturation when lambda_D grows to the cell size.
- Adams, Werner and Cary 2025: first systematic map of grid-heating growth rates over Debye
  under-resolution and drift velocity for the momentum-conserving scheme; higher-order field
  solves do not remove it; energy-conserving (EC) linear/quadratic schemes also heat for some
  drift/under-resolution combinations; cold-beam stability limits derived.
- Barnes and Chacon 2021: EC-PIC is stable against aliasing for stationary plasmas and has a
  benign threshold for drifting finite-temperature plasmas, "usable in practice ... without the
  need to resolve Debye lengths spatially"; momentum-conserving PIC has no such threshold.
- Chen, Chacon and Barnes 2011; Chacon, Chen and Barnes 2013; Lapenta 2017 and 2023 (ECsim);
  Markidis and Lapenta 2011; Brackbill 2016: the implicit/semi-implicit energy-conserving
  family and its conservation trade-offs.
- Eremin 2022: energy- and charge-conserving implicit PIC for *collisional bounded* plasmas
  (the low-temperature-plasma variant closest to our problem class).
- Savard, Fubiani, Eremin and Dehnel 2025: when Dx > lambda_D the implicit EC scheme needs
  *more* particles per cell to reproduce converged solutions; non-uniform grids and collisions
  "exacerbate the errors"; in 1D it was slower than explicit MC PIC once accuracy is required.
- Ricketson and Hu 2025: an explicit, energy-conserving scheme (a third option).
- Taccogna, Longo, Capitelli and Schneider 2005; Lacina 1971; Matthias et al. 2020: geometric
  self-similarity scaling instead of permittivity/mass scaling, and its limits for HEMP-T PIC.
- Szabo 2001 (thesis) and Szabo et al. 2014: artificial permittivity plus artificial mass ratio
  with an explicit retrieval procedure; thrust within 5 %, current within 16 %.
- Cho, Komurasaki and Arakawa 2013; Cho et al. 2015: mass-ratio manipulation with a
  semi-implicit field solver and a mobility-recovery model, and an explicit refusal of
  permittivity or geometry scaling "to avoid unrecoverable change of physics".
- Duras et al. 2014; Colella and Norgaard 2010; Vay et al. 2004; Chacon and Chen 2016;
  Villafana, Cuenot and Vermorel 2023: non-uniform, refined, curvilinear and unstructured grids
  and the self-force they introduce.
- Ricketson and Cerfon 2017; Muralikrishnan et al. 2021; Garrigues et al. 2021 (I, II) and
  2024 (II): sparse-grid PIC, including a multi-cusp magnetic-field application.

### (b) What the field does

1. **Resolve it.** The recent E x B benchmarks (Charoy et al. 2019; Villafana et al. 2021) are
   specified with cells at or below the local Debye length and were run predominantly with
   explicit momentum-conserving codes; the HEMP work of the Greifswald group is explicit
   momentum-conserving on a uniform grid. Debye resolution is treated as a hard constraint that
   sets the grid, not a budget to be negotiated. Brandt et al. 2016 ran Dx = 20 um against lambda_D of
   order 10 um in the channel (i.e. about 2 lambda_D per cell), with the sheath quoted as
   5-10 lambda_D wide, and did not report grid heating diagnostics.
2. **Scale the physics so that it becomes resolvable.** Three families are in use: (i) artificial
   permittivity eps -> gamma eps_0 (Szabo 2001, 2014; Fubiani et al. 2017 for high-density ion
   sources), which multiplies lambda_D and the plasma period by sqrt(gamma); (ii) artificial
   ion mass (Szabo 2014; Cho 2013, 2015), which shortens the ion transit without touching the
   electron scales; (iii) geometric self-similarity (Taccogna et al. 2005; Lacina 1971) that
   shrinks the device while holding L/r_L and L/lambda_mfp fixed, used by Brandt et al. 2016
   (factor 4) and quantified for HEMP-T by Matthias et al. 2020.
3. **Change the scheme.** Energy-conserving (implicit or explicit) PIC removes the
   finite-grid instability for stationary plasmas (Barnes and Chacon 2021) and is the route the
   computational community recommends for Dx > lambda_D; sparse grids reduce the grid-based
   error and cost by 3-5x at 256^2-512^2 nodes (Garrigues et al. 2024, with a multi-cusp
   field configuration among the test cases).

### (c) Documented pitfalls

- Grid heating is not removed by a better Poisson solve (Adams et al. 2025) and its onset
  depends on drift as well as on Dx/lambda_D: our injected 300 V beam and the 126 eV cusp ions
  are drifting populations exactly where Adams et al. find EC schemes can still heat.
- EC implicit PIC is not a free lunch for a collisional bounded plasma: more particles per cell
  are needed when Dx > lambda_D, non-uniform grids and collisions make it worse, and in 1D it
  lost to explicit MC PIC on wall-clock at equal accuracy (Savard et al. 2025).
- Permittivity scaling enlarges every sheath by sqrt(gamma). In a 2 mm bore whose cusp sheaths
  and leak widths are the physics of interest (Hershkowitz, Leung and Romesser 1975; Leung,
  Hershkowitz and MacKenzie 1976), a factor of 4-10 in lambda_D changes the object being
  simulated; Cho et al. 2013 refuse it for that reason, and Taccogna and Minelli 2018 say of
  geometric scaling that it "irremediably changes ... the wall interaction and the axial
  component of the electric field".
- Geometric self-similarity fails when surface processes matter, because the
  surface-to-volume ratio does not scale (Brandt et al. 2016, citing their own use of it);
  Matthias et al. 2020 derive the limits for HEMP-T.
- Mass-ratio scaling changes the ratio of ion transit to breathing/ionisation time scales
  and requires a retrieval model for the electron mobility in weakly magnetised regions (Cho
  et al. 2013).
- Non-equidistant grids generate artificial self-forces at cell-size changes that destroy
  momentum conservation unless the field gather is modified (Duras et al. 2014, written for
  the Greifswald HEMP code); AMR refinement boundaries need explicit self-force control
  (Colella and Norgaard 2010; Vay et al. 2004).

### (d) What we should change (mapped to our spec/protocol)

1. Keep the explicit momentum-conserving scheme for the campaign and **make the grid follow the
   peak**: the proposal's 100 x 800 grid (Dr = Dz = 30 um) gives 2.0 lambda_D per cell at the
   design peak 2.0e18 m^-3 / 8 eV, which is what the Greifswald analogue ran; require the
   block-C refinement case (20 um) to agree within the declared 10 % before the grid claim is
   accepted (proposal section 2.3). Do not relax to 3 lambda_D anywhere.
2. Move `stability_limits.max_cell_debye_ratio` (protocol.json, currently 2.0 evaluated at the
   `stability_reference.density_per_m3 = 4e17`) onto the **instantaneous window-peak node
   density** and make it fail-closed, exactly as the omega_pe Dt gate already is (proposal
   section 3.5). The development run passed at 3 lambda_D because the gate looked at n_max.
3. Add the grid-heating triad as a recorded, gated diagnostic (proposal section 2.5 has the
   ledger part): (i) energy-ledger residual / electrode work, (ii) T_e and ionisation rate on the
   W and grid variants, (iii) omega_pe Dt drift. Adams et al. 2025 and Ueda et al. 1994 give the
   signatures; the snapshot-v2 coarse pair already showed all three.
4. Do **not** use artificial permittivity for the campaign claim. If a scaled development run
   is wanted for cost, run two gamma values and show the retrieval (Szabo et al. 2014) before
   trusting any cusp-sheath quantity; state that the sheath/bore ratio was changed.
5. Optional, after the campaign: implement the energy-conserving deposition/gather pair (the
   Lewis/Markidis form) as a switch in `kernels.py`, and compare against the momentum-conserving
   run on the snapshot-v2 coarse case, where grid heating is known to occur. Barnes and Chacon
   2021 and Adams et al. 2025 give the stability expectations to test against.

### (e) Where the literature is silent

- No verified HEMP/CFT PIC paper reports a grid-heating diagnostic (ledger residual or
  T_e vs Dx) at the cusps; Brandt et al. 2016 ran at ~2 lambda_D per cell without one.
- No energy-conserving (implicit or explicit) PIC has been applied to a cusped-field thruster.
- Savard et al. 2025 is 1D CCP; the ppc/cell-size penalty of implicit EC PIC in a magnetised
  2-D cusp geometry is unmeasured.
- Matthias et al. 2020 give limits for *geometric* similarity in HEMP-T; nothing verified
  quantifies permittivity-scaling error for a cusp thruster.

---

## 2. Blocker 2: cost (3 transits = 3 h; 110-125 GPU-h for the campaign)

### (a) Key references

- Brandt et al. 2016: 2.4e7 steps to a quasi-steady state at 76 us on 1024 x 256 cells with
  MPI; steady state "sufficient", breathing would need ~10x more.
- Duras et al. 2017: a parallelisation strategy for the Greifswald HEMP code, motivated by
  the impossibility of simulating 1 m plume domains.
- Kahnfeld et al. 2016: direct LU versus SOR Poisson solvers for the HEMP DM3a PIC (runtime
  comparison).
- Charoy et al. 2019; Villafana et al. 2021: seven independent codes, computing times and
  resources tabulated, convergence in macro-particles per cell demonstrated, 5 % agreement.
- Zhao and Zhao 2026 (review): compiled workloads and wall-clock costs of 3D PIC for the
  electron drift instability; strategies for time-to-solution.
- Decyk and Singh 2014; Juhasz et al. 2021 (PIC-MCC on GPU: up to 200x over CPU, 2.6 TFlop/s
  sustained); Bowers 2001 (counting sort for locality).
- Lapenta 2017, 2023 (ECsim); Chen et al. 2011; Eremin 2022; Savard et al. 2025 (implicit).
- Reza, Faraji and Knoll 2023; Faraji, Reza and Knoll 2022: reduced-order ("quasi-2D",
  "pseudo-2D") PIC at 2-15 % of full-2D cost with 2-4 % error on time-averaged fields.
- Garrigues et al. 2024: sparse-grid PIC speed-up ~3x (256^2) to ~5x (512^2), multi-cusp case.
- Cho et al. 2013: mass-ratio manipulation and a semi-implicit solver made a fully kinetic
  *lifetime* simulation feasible.
- Taccogna et al. 2005; Matthias et al. 2020: self-similarity as the HEMP community's cost tool.
- NVIDIA Technical Blog, "Getting Started with CUDA Graphs" (2019): per-launch overhead of
  several microseconds; one graph launch per time step amortises it.

### (b) What the field does

- **Accept long runs and average.** The Greifswald HEMP runs reach tens of microseconds
  (Brandt et al. 2016: 76 us) and average over 1e6 steps; the E x B benchmarks run to a
  time-averaged steady state and then vary ppc to demonstrate convergence (Charoy et al. 2019;
  Villafana et al. 2021). Nobody in the verified HEMP literature reports a converged discharge
  after 3 ion transits.
- **GPU PIC-MCC** is mature for 1D/2D (Juhasz et al. 2021: 10 M particles, 200x speed-up) and
  is the substrate of the newest 3D work (Zhao and Zhao 2026 mention WarpX-based studies).
- **Reduced-dimensional / reduced-order** schemes (Reza et al. 2023; Faraji et al. 2022) buy an
  order of magnitude for azimuthal-instability problems; they are not aimed at (r,z) cusp
  geometry.
- **Sparse grids** (Garrigues et al. 2024) and **implicit EC** (Eremin 2022) are the two
  "change the algorithm" routes; the first has a cusp application, the second has a documented
  ppc penalty when Dx > lambda_D (Savard et al. 2025).
- **Scaling** (mass ratio, self-similarity) remains the HEMP community's main lever (Matthias
  et al. 2020; Brandt et al. 2016).

### (c) Documented pitfalls

- Reduced-order and sparse-grid schemes trade grid-based error for cost; Reza et al. 2023
  report 15 -> 4 % (E) and 20 -> 2 % (n_i) as the approximation order rises, and Garrigues et al.
  2024 report the combination step matters (combine phi at regular nodes, not E).
- Implicit EC PIC can be *slower* at fixed accuracy (Savard et al. 2025).
- Launch-bound GPU loops: per-kernel submission overhead is of order microseconds on a datacentre
  GPU under Linux (NVIDIA 2019); our measured ~1.2 ms floor for ~40 launches per step on WDDM is
  ~30 us per launch, an order of magnitude worse, and is a driver-model property, not a kernel
  property (`pic2d-devlog.md`, phase 2).
- Short runs hide the slow variables: Brandt et al. 2016 explicitly state their 76 us was
  enough for steady state but an order of magnitude short for the breathing mode.

### (d) What we should change

1. **Capture the whole step in one CUDA graph** (the PCG path already used graphs; the
   device-direct path does ~40 launches). Expected gain: the WDDM launch floor (~1.2 ms of the
   2.0 ms step at 1 M particles; kernel time is 1.0-1.3 ns per particle-step, `pic2d-devlog.md`
   phase 2) shrinks toward the single-graph-launch cost, i.e. up to ~40 % off the step time at
   plateau particle counts if the floor is launch-bound as diagnosed. Cheaper than any algorithm
   change and it does not alter the numerics (bitwise-testable against the un-captured path).
2. Periodic **particle sort by cell** (Bowers 2001; Decyk and Singh 2014) every N steps to
   improve gather/deposit locality; test that tallies remain exact.
3. Move the campaign to a TCC-mode or Linux host if one is available; the proposal's own
   estimate is ~4x throughput (section 5). This is the only change that removes the floor.
4. Re-scope the budget around the literature's definition of "converged": the 5-transit,
   3 %-two-checkpoint criterion (proposal section 2.4) is a plateau under the quasi-steady
   neutral closure, not the tens-of-microseconds steady state of Brandt et al. 2016. State that
   in the claim boundary (section 2.6) and price one **long development case** (>= 50 us at
   reduced W) to test for slow modes before freezing the campaign length.
5. Do not adopt implicit EC or sparse grids for this campaign; both are research items with a
   documented accuracy/ppc trade-off and no cusp-thruster precedent in (r,z).

### (e) Where the literature is silent

- No GPU-hour or core-hour figures for a converged HEMP/CFT PIC in a journal (Brandt et al.
  2016 give steps and time simulated, not wall time; Kahnfeld et al. 2016 compare solver
  runtimes only).
- Nothing on Windows/WDDM launch behaviour for PIC; the CUDA-graph gain is documented for a
  V100 under Linux only (NVIDIA 2019).
- No published cost/accuracy comparison of explicit MC PIC versus implicit EC PIC in a
  magnetised 2-D cusp geometry.

---

## 3. Blocker 3: neutral model (uniform n_g(t) with an artificial 30 ns relaxation)

### (a) Key references

- Brandt et al. 2016: DSMC neutral field imported and held **static** ("the timescale of the
  simulation is too short for significant changes in the neutral gas"), diffuse wall reflection
  at 500 K matched the measured neutral thrust; measured depletion 25 %.
- Kahnfeld et al. 2018: with neutral dynamics in the fully kinetic HEMP model, breathing modes
  at ~100 kHz appear and are compared with a 0-D predator-prey estimate.
- Boeuf and Garrigues 1998; Barral and Ahedo 2009; Hara et al. 2014; Lafleur, Chabert and
  Bourdon 2021; Chapurin et al. 2021: breathing-mode theory; the instability lives in the
  neutral depletion/refill delay.
- Petronio et al. 2024: 1-D Euler neutrals coupled to 2-D PIC reproduce breathing; the usual
  0-D predator-prey gas-convection approximation "does not" hold; ionisation depends on the
  *spatial correlation* of n_e and n_g.
- Katz and Mikellides 2011; Mikellides and Katz 2012: free-molecular view-factor neutral
  algorithm including ionisation and walls (Hall2De).
- Parra et al. 2006 (HPHall hybrid): neutrals as particles with wall accommodation.
- Szabo 2001 and 2014: neutrals as kinetic particles, ions recombined at boundaries and
  "recycled into the flow"; breathing captured.
- Boeuf and Garrigues 2018; Charoy et al. 2019; Villafana et al. 2021: the alternative
  simplification, a **prescribed ionisation source profile** with no neutral dynamics, used
  deliberately for instability studies and benchmarks.
- Hara 2019 (review): the neutral/ion/electron model choices and their consequences.

### (b) What the field does

Three practices coexist: (1) frozen neutrals for short instability studies and benchmarks
(Boeuf and Garrigues 2018; Charoy et al. 2019), (2) kinetic or DSMC neutrals with wall
recycling for discharge-level simulations (Szabo 2001, 2014; Brandt et al. 2016; Kahnfeld et
al. 2018), (3) fluid or view-factor neutrals in hybrid codes (Boeuf and Garrigues 1998; Parra
et al. 2006; Katz and Mikellides 2011). Every code that reports a discharge-current time trace
and a breathing frequency resolves neutral *transport* with its real time scale.

### (c) Documented pitfalls

- A uniform n_g with 46 % utilisation is self-inconsistent: ionisation depends on where
  electrons and neutrals overlap (Petronio et al. 2024), and depletion is local to the
  ionisation peak between the last two cusps in our run.
- **Wall recycling is missing.** In our plateau 3.72 mA of Xe+ (59 % of the 6.30 mA ionisation)
  is absorbed by the dielectric wall and removed from the atom inventory. Every kinetic-neutral
  thruster PIC recycles wall-recombined ions as thermal neutrals (Szabo 2001; Brandt et al. 2016
  count ions "recycled after wall contact as neutrals" and quote *net* ionisation 24 %). Our
  utilisation and fixed point therefore correspond to a different closure than the literature's,
  and the comparison in section 6 must use net ionisation.
- The artificial relaxation removes the transport delay that produces breathing (Lafleur et
  al. 2021; Chapurin et al. 2021; Petronio et al. 2024): a plateau under this closure cannot
  tell whether the physical discharge would oscillate.
- Wall accommodation and neutral temperature change n_g by the factor sqrt(T_w/T_g) at fixed
  flux; Brandt et al. 2016 needed a diffuse-reflection assumption and 500 K to match neutral
  thrust.

### (d) What we should change

1. Add **wall recycling** to the 0-D balance now (`cft_revival.pic2d.neutrals`): V dn_g/dt =
   Q_in + R_wall - S - c n_g with R_wall the wall+anode ion absorption tally x W per interval,
   returned at the wall temperature; ledger it. This is a few lines and moves the fixed point:
   at the recorded plateau the net sink is S - R_wall = 6.30 - 3.77 mA-equivalent
   (3.93e16 - 2.35e16 atoms/s), so n_g* = (Q_in - S + R_wall)/c rises from 2.97e19 to
   ~4.5e19 m^-3 before S itself responds to the denser gas. It must be in
   `pic2d-model-v2.0.json` before any utilisation figure is quoted against Brandt et al. 2016.
2. Replace the uniform inventory by the proposal's 1-D axial free-molecular column model
   (section 3.3) implemented as Katz and Mikellides 2011 view factors on the (r,z) mask (anode
   feed, per-cell ionisation sink from the MCC tallies, diffuse wall re-emission, exit
   effusion), with the artificial relaxation kept **only** as a documented option and defaulting
   to off.
3. Run one development case with the physical neutral time scale (no relaxation) for >= 50 us
   at reduced W to observe whether the closure oscillates (section 5). Only then freeze the
   plateau criterion.
4. Record in `protocol.json` that "steady state" is conditional on the neutral closure and
   quote net (recycled) utilisation.

### (e) Where the literature is silent

- No verified precedent for a fast artificial relaxation of a 0-D neutral inventory toward
  its fixed point in a PIC discharge model; the closest are "frozen neutral" runs and 0-D
  predator-prey estimates used only as a check (Kahnfeld et al. 2018).
- No paper quantifies the error of the uniform-neutral assumption for a cusp thruster.
- Wall accommodation coefficients for BN/Al2O3 at HEMP wall temperatures are not reported in
  the verified set.

---

## 4. Blocker 4: missing physics

### 4.1 Ion-neutral elastic and charge-exchange collisions

- References: Miller et al. 2002 (Xe+-Xe CEX and momentum-transfer cross sections for
  thruster models); Brandt et al. 2016 (CEX and Coulomb collisions included in the HEMP channel
  model); Duras et al. 2017 (CEX must be post-processed in the plume to reproduce 1 m angular
  distributions; distributions are "quite sensitive" to boundary potentials, SEE and CEX);
  Szabo 2001, 2014.
- SOTA: include Xe+-Xe elastic and CEX in the MCC (Brandt et al. 2016 do); their main effect in
  the literature is on the *plume* ion energy/angular distribution (Duras et al. 2017).
- Pitfall: none documented inside a 24 mm channel at 3e19 m^-3, where the CEX mean free path
  (~5-7 cm at sigma ~5e-19 m^2) exceeds the channel; the proposal's own estimate is "few %" on
  the beam fraction.
- Recommendation: implement (proposal section 3.1) with the Miller et al. 2002 tables
  hash-bound like Biagi, but classify it as a *reported sensitivity*, not a campaign blocker;
  it becomes a blocker only when a plume region is added (section 4.4).
- Silence: no cusp-thruster PIC reports the in-channel beam-fraction change due to CEX.

### 4.2 Secondary electron emission from BN / Al2O3 and the cusp sheaths

- References: Vaughan 1989 and Furman and Pivi 2002 (yield models); Dunaevsky, Raitses and
  Fisch 2003 and Tondu, Belhaj and Inguimbert 2011 (measured yields of HET channel ceramics
  including BN-based and Al2O3); Hobbs and Wesson 1967 (space-charge-limited sheath at
  yield ~1); Campanell, Khrabrov and Kaganovich 2012 (inverse sheath when emission exceeds the
  limit); Sydorenko et al. 2006 (kinetic SEE effects, non-Maxwellian EEDF reduce the effective
  yield); Taccogna, Longo and Capitelli 2005 (sheath structure in a Hall discharge with SEE);
  Tavant et al. 2018 (SEE doubles near-wall mobility and cools the bulk by ~20 %; three
  regimes including an oscillatory one); Taccogna and Minelli 2018 (SEE insufficient to carry
  the neutralising electron current; acts to saturate the drift instability); Brandt et al.
  2016 (HEMP with a simple 50 %/90 % re-emission model); Matyash et al. 2010 (HEMP wall
  contact confined to the cusps).
- SOTA: energy-dependent yield (Vaughan or Furman-Pivi) with backscattered/rediffused/true
  secondary split, emitted half-Maxwellian at 1-2 eV, and a space-charge-limited cap at
  yield -> 1 (Hobbs and Wesson 1967) or explicit inverse-sheath handling (Campanell et al.
  2012).
- Pitfalls: (i) in a HEMP the wall flux is concentrated at the cusps (Matyash et al. 2010;
  our wall flux peaks at the 12.2 mm cusp), so SEE acts where the sheath is thinnest and the
  ion energy highest (126 eV mean impact in our run; 160 eV in Brandt et al. 2016), and the
  sheath can go space-charge-limited locally while remaining classical elsewhere; (ii) a
  Maxwellian-based yield overestimates emission when the EEDF is depleted at high energy
  (Sydorenko et al. 2006); (iii) SEE changes the electron energy balance by O(1) when the wall
  electron current equals the wall ion current, as in our plateau (3.73 vs 3.72 mA).
- Recommendation: implement the proposal's section 3.2 with a fixed material (BN) and the
  Vaughan form, fail closed at yield >= 1 unless the Hobbs-Wesson cap is implemented; report
  the cusp-local yield and the fraction of wall electron current re-emitted; run the campaign's
  SEE-off case as the reported sensitivity because the literature (Tavant et al. 2018) finds
  the spatially averaged mobility roughly unchanged while the wall-local sheath changes.
- Silence: no HEMP/CFT PIC quantifies SEE sensitivity at the cusps; the BN yield curves in
  Dunaevsky et al. 2003 / Tondu et al. 2011 are for HET-grade ceramics, not necessarily the
  grade of a micro-HEMPT liner.

### 4.3 Anomalous cross-field transport and whether (r,z) ES-PIC can capture it

- References: Adam, Heron and Laval 2004 (first 2-D kinetic demonstration that azimuthal
  turbulence supplies the anomalous conductivity); Cavalier et al. 2013 (E x B drift
  instability identified in experiment); Lafleur, Baalrud and Chabert 2016; Croes et al. 2017;
  Boeuf and Garrigues 2018; Lafleur et al. 2018; Taccogna et al. 2019; Charoy et al. 2019;
  Villafana et al. 2021, 2023; Zhao and Zhao 2026 (3-D review: "reduced dimensionality (1D/2D)
  PIC models tend to bias the mode structure and the inferred transport"); Carlsson et al.
  2018 and Powis et al. 2018 (spoke-driven anomalous transport in a Penning discharge, 2-D
  r-theta); Boeuf 2017 and Kaganovich et al. 2020 (reviews); Hara 2019; Marks and Jorns 2023
  (closure models diverge when implemented self-consistently).
- Axisymmetric (r,z) codes: Cho et al. 2015 observed an effective mobility two orders above
  classical in an r-z full PIC *without* an imposed model, attributed to a 20 kHz electron-flow
  oscillation coupled to the ionisation instability, i.e. low-frequency axisymmetric dynamics,
  not the mm-scale drift instability; Szabo 2001, 2014 and Brandt et al. 2016 impose Bohm-type
  scattering (Brandt: D = 0.4 kT_e/eB "derived from a 3D simulation of a similar thruster
  model"); Smirnov, Raitses and Fisch 2004 needed nu_B ~ omega_c/16 to explain the cylindrical
  Hall thruster's current.
- HEMP specifically: Matyash et al. 2010 and Kalentev et al. 2014 present the HEMP discharge
  as wall-contact-free except at the cusps; the Greifswald code nevertheless carries an
  anomalous-diffusion model (Brandt et al. 2016). The verified abstracts do not settle whether
  a HEMP needs anomalous transport to reproduce its discharge current.
- Pitfall: an axisymmetric electrostatic code excludes the electron drift instability by
  construction; any agreement of I_d with experiment in such a code is either fortuitous,
  produced by an imposed Bohm term, or produced by axisymmetric low-frequency dynamics (Cho et
  al. 2015). The exit-plane return of 1.84 of 3.00 mA in our plateau is a transport-limited
  quantity.
- Recommendation: (i) declare "no anomalous transport" in the claim boundary (proposal
  section 6 does); (ii) add one **Bohm-scattering sensitivity case** (equivalent collision
  frequency nu_an = alpha omega_ce with alpha in {1/64, 1/16}, the Szabo 2001 / Brandt 2016
  device) to the campaign matrix as block E, reported not binding; (iii) diagnose the
  low-frequency axisymmetric mobility in the plateau window (Cho et al. 2015 method: effective
  mobility from the electron flux and the fields) so the campaign reports what transport the
  model *does* contain.
- Silence: no (r,z)+theta or 3-D PIC of a cusped-field thruster that isolates the anomalous
  contribution; no verified paper states whether the HEMP discharge current is reproduced
  without an anomalous term.

### 4.4 Exit-plane / cathode boundary

- References: Szabo 2001 ("the cathode was [modelled] indirectly by injecting electrons at a
  rate which preserved quasineutrality"); Charoy et al. 2019 and Boeuf and Garrigues 2018
  (cathode-plane electron injection tied to the ion current leaving the domain, Dirichlet
  potentials); Brandt et al. 2016 (electron source on the far outer boundaries at 1 eV, 0 V
  Dirichlet; switching the outer boundary to Neumann changed the plume current ratios, "the
  outer simulation domain is still too small"); Duras et al. 2017 (angular distribution
  "quite sensitive to boundary conditions of the potential"; correction to 1 m); Matyash et
  al. 2010 (near-field plume included in the HEMP domain).
- Pitfalls: a Dirichlet 0 V plane 6 mm downstream of the last cusp fixes the potential where
  the plume should be free; our phi_max = 337 V hump above the 300 V anode and the 1.84 mA
  returned beam depend on it. Fixed-current injection is only the literature's quasi-neutral
  injection if the injected current happens to match the ion current leaving.
- Recommendation: proposal section 3.4 variant (a) (a 6 mm plume region, Neumann side,
  Dirichlet far plane) is the literature's practice (Brandt et al. 2016; Matyash et al. 2010);
  add the current-continuity injection rule (electron injection = ion current through the far
  plane + returned electron current) as variant (c) and freeze one; run the others as
  reported sensitivities. Note that Brandt et al. 2016 found even a 20 mm x 5 mm domain too
  small for plume ratios.
- Silence: no HEMP paper isolates the effect of the cathode model on the *in-channel*
  potential hump; the sensitivity statements are about the plume.

### 4.5 Sheath resolution at the cusps

- References: Hershkowitz, Leung and Romesser 1975 and Leung, Hershkowitz and MacKenzie 1976
  (cusp leak width and confinement); Matyash et al. 2010 (wall contact only in "very small
  areas of the magnetic field cusps"); Brandt et al. 2016 (sheath 5-10 lambda_D, Dx = 20 um
  with lambda_D ~ 10 um); Taccogna et al. 2005 sheaths; Tavant et al. 2018 (kinetic sheaths
  differ from analytic models).
- Recommendation: report, per cusp, the leak width, the local lambda_D, the local
  cells-per-lambda_D and the sheath potential drop as campaign diagnostics; the block-C
  refinement decides whether they are resolved. No local refinement (blocker 1 pitfalls).
- Silence: no cusp-thruster PIC reports a convergence study of the cusp sheath itself.

---

## 5. Blocker 5: plateau / steady-state criteria and statistical convergence

### (a) Key references

- Kahnfeld et al. 2018: HEMP-T DM3a "develop breathing mode oscillations with frequencies in
  the 100 kHz range during stable operation"; reproduced by fully kinetic PIC-MCC; compared
  with 0-D predator-prey and with a measured discharge current.
- Brandt et al. 2016: quasi-steady state at 76 us, averages over 1e6 steps; breathing needs
  ~10x longer.
- Gildea et al. 2010 (low-frequency oscillations in the DCFT); Gildea et al. 2013 (DCFT
  "high-current" mode "characterized by strongly oscillatory anode currents"); MacDonald,
  Cappelli and Hargus 2014 (time-synchronised LIF through the DCFT oscillation).
- Boeuf and Garrigues 1998; Barral and Ahedo 2009; Hara et al. 2014; Lafleur et al. 2021;
  Chapurin et al. 2021; Petronio et al. 2024: breathing mechanism and its dependence on neutral
  transport.
- Charoy et al. 2019; Villafana et al. 2021: steady state defined as a long time average after
  the transient; convergence demonstrated by varying macro-particles per cell; 5 % code-to-code
  agreement.
- Turner 2006 and 2016; Turner et al. 2013: MCC accelerates numerical thermalisation by up to
  three orders of magnitude; ppc convergence and verification against exact solutions.
- Cho et al. 2015: a 20 kHz global ionisation oscillation appears in an (r,z) full PIC.

### (b) What the field expects

The HEMP community does **not** expect a strict steady state: the DM3a breathes at ~100 kHz in
stable operation and the kinetic code reproduces it (Kahnfeld et al. 2018); the MIT DCFT has a
strongly oscillatory high-current mode (Gildea et al. 2013). "Steady state" in the PIC
literature means a time average over a window long compared with the slowest resolved mode,
followed by a ppc convergence check (Charoy et al. 2019; Villafana et al. 2021). A
predator-prey estimate for our channel, f ~ (1/2 pi) sqrt(v_i v_n)/L with v_i ~ 1.5e4 m/s,
v_n = 220 m/s, L = 24 mm, gives ~12 kHz, i.e. a period of ~80 us ~ 35 of our ion transits;
Kahnfeld et al. 2018 found ~100 kHz for the 51 mm DM3a, so the estimate is only an order of
magnitude, but either value is far longer than our 7.7 us run.

### (c) Documented pitfalls

- A drift criterion on a 20 % trailing window declared after 3 transits is a test of the
  neutral closure, not of the discharge (section 3). Our development run passed at +4.98 % on
  N_e while omega_pe Dt was still rising.
- MCC-driven numerical thermalisation (Turner 2006) means seed and ppc variance are not pure
  shot noise; our seed pair showed a 3-7x excess over 1/sqrt(N) for the smooth quantities.
- Time-offset comparisons between runs confound densification with variance (our window-B
  lesson; the benchmarks compare at the same simulated time).

### (d) What we should change

1. Keep the tightened criterion of proposal section 2.4 (3 %, two consecutive checkpoints,
   >= 5 transits, peak density and omega_pe Dt tracked) **and** add: (i) an explicit spectral
   check of I_d and n_g over the plateau window (no line above the noise floor below 1 MHz),
   (ii) the statement that the plateau is conditional on the neutral closure.
2. Before freezing the campaign, run the long development case of section 3(d)3 with physical
   neutral transport; if it breathes, the campaign observables become period-averaged means and
   the breathing frequency itself (Kahnfeld et al. 2018 is then the comparison), and the wall
   budget must cover >= 3 periods.
3. Keep >= 3 seeds and the W variant (proposal section 2.3); report mean +- sample standard
   deviation, batch-means SE with the block length, and the shot-noise floor, as in the seed-b
   comparison. State that the literature offers no seed-variance precedent to compare with.
4. Adopt the benchmark practice of Charoy et al. 2019 for the convergence statement: same
   simulated window, ppc doubled, 5 % agreement on the time-averaged axial profiles of E_z, n_i
   and T_e in addition to the scalar currents.

### (e) Where the literature is silent

- No published seed-to-seed variance for any HEMP/CFT PIC; convergence is shown by ppc only.
- No agreed plateau criterion; papers state the averaging window and let the reader judge.
- Whether a HEMP at our scale (2 mm bore, 0.19 sccm) breathes at all is not established;
  Keller et al. 2015 report stable operation down to 50 uN experimentally but no spectra in
  the verified abstract.

---

## 6. Blocker 6: validation targets in the published HEMP / CFT / DCFT / CHT literature

### (a) What exists and what it reports (verified abstracts and one full text)

| source | device / operating point | quantities reported that we could compare | caveat |
| --- | --- | --- | --- |
| Brandt et al. 2016 | DLR/Airbus micro-HEMPT, Z = 14 mm, R = 1.5 mm, 400 V, 0.27 sccm Xe; static neutrals ~2e20 m^-3; Bohm diffusion and SEE imposed | anode electron current 4.3 mA (exp. 4.5 mA); net ionisation 24 % (exp. 25 %); beam current 2.5 mA (exp. 3.1 mA) -> beam fraction ~0.58; potential drops ~10 V and ~5 V at the internal cusps, main drop at the exit cusp; sheath 5-10 lambda_D; wall ion energies to 160 eV at the exit-side internal cusp with 640 A/m^2; plume peak 50 deg (exp. 60 deg); n_i ~1e19 m^-3 | includes wall recycling, Bohm scattering and SEE that we lack; static neutrals; the closest operating point to ours in the literature |
| Keller et al. 2015 | the same downscaled HEMPT family, experiment | thrust 50-360 uN, Isp 230-860 s at up to 600 V; beam profile and ion acceleration versus geometry; anode material effect | experimental; no PIC |
| Matyash et al. 2010; Schneider et al. 2009; Kalentev et al. 2014 | HEMP-T DM3a (Thales), (r,z) PIC-MCC | wall contact limited to cusps; "no erosion inside the dielectric discharge channel" (SDTrimSP); ion energy flux to the wall small | operating points not in the abstracts |
| Kahnfeld et al. 2018 | DM3a | breathing frequency ~100 kHz vs measured discharge current; 0-D predator-prey comparison | needs neutral dynamics on our side |
| Duras et al. 2017 | HEMP-T | ion angular current and energy distributions at 1 m | needs a plume region and CEX post-processing |
| Matthias et al. 2019; Yeo et al. 2020 | the downscaled CFT of the Fahey-Muffatti-Ogawa 2017 MDO | PIC vs 0-D power balance performance; beam current and divergence uncertainties reduced to 5 % | design point details not in the abstracts; the direct descendant of this repository's model |
| Zhao et al. 2013; Liu et al. 2015; Hu et al. 2016 (HIT) | CFT PIC-MCC at three B strengths; channel radii 6/4/2 mm; experiment on B strength | density-peak position vs B; main potential drop near the exit; potential wells near the wall; 2 mm radius cusps confine poorly; experimentally weaker in-channel B gave higher thrust and anode efficiency | qualitative profiles; operating points not in the abstracts |
| Courtney et al. 2008; Matlock et al. 2009; Gildea 2009; Gildea et al. 2010, 2013; MacDonald et al. 2011, 2014 (MIT/Stanford DCFT) | DCFT, 165 W anode power (erosion test) | probe potentials and spectroscopy; PIC at 1/5 B; low-frequency oscillations; erosion 204 h in BN with maximum in a ring cusp, life 920-1220 h; LIF ion velocities incl. time-synchronised | different (diverging, 3-cusp) geometry; PIC never completed at full B |
| MacDonald et al. 2012; Lucca Fabris et al. 2015 (Stanford CCFT) | straight-channel cylindrical cusped-field thruster at 300 V and 150 V | LIF ion velocities and emissive-probe plasma potential in channel and plume; 3-D F3MPIC PIC vs LIF: steep potential drop along the top-cusp separatrix; 30 deg half-angle | the only cusped-field PIC-vs-LIF comparison verified here |
| Raitses and Fisch 2001; Smirnov, Raitses and Fisch 2004 (JAP and PoP) | Princeton 100 W cylindrical Hall thruster, 2.6 cm | plasma measurements in the channel; Bohm-level anomalous collision frequency nu_B ~ omega_c/16 needed to reproduce I_d | CHT, not multi-cusp |

### (b) SOTA for validation

The field compares time-averaged axial/radial profiles (n, phi, T_e), discharge current,
beam current and utilisation, wall flux/energy at the cusps, and plume angular distributions,
against probe/LIF data at a stated operating point (Brandt et al. 2016; Lucca Fabris et al.
2015; Cho et al. 2015; Szabo et al. 2014). Agreement of 5-20 % on scalar performance is the
reported norm (Szabo et al. 2014: thrust 5 %, current 16 %; Cho et al. 2015: < 20 %; Brandt et
al. 2016: current 5 %, utilisation 1 %, beam current 20 %).

### (c) Pitfalls

- All published analogues include at least one of {wall recycling, Bohm scattering, SEE,
  plume region} that our model lacks; a bare comparison of our 46 % utilisation to Brandt's
  24 % is a comparison of closures (section 3).
- Profiles are published as figures, not tables; only scalars can be compared without
  digitising.
- No experiment exists at our exact geometry (P2 divergent-exit channel); the ISTS 2017 study
  (`REFERENCES.md`) and Fahey, Muffatti and Ogawa 2017 are model outputs, not measurements.

### (d) What we should change

1. Make the campaign's reported observables the literature's: anode electron current, net
   ionisation fraction (after wall recycling), beam current and fraction, wall ion current and
   mean energy per cusp, potential step per cusp, plume divergence (needs section 4.4), and
   the axial max_r n_e profile against the cusp planes. Add these to the dashboard generator
   and to `protocol.json` as named outputs.
2. Add a "literature context" panel: Brandt et al. 2016 and Keller et al. 2015 numbers beside
   ours, labelled *different closure, different geometry*, never as a validation gate.
3. Do not claim external validation; the paper's L3 gate stays closed (`ROADMAP_AUDIT.md`
   section 2.8).

### (e) Where the literature is silent

- No tabulated density/potential profiles for any HEMP/CFT PIC.
- No published cusp wall flux with uncertainty; erosion data (Gildea et al. 2013) is the only
  quantitative wall-loss measurement and it is for the DCFT.
- No experiment at 2 mm bore / 0.19 sccm / 3 mA.

---

## 7. Summary table

| blocker | recommended change | effort | risk |
| --- | --- | --- | --- |
| 1 Debye resolution | peak-node Debye gate fail-closed (protocol `stability_limits`); 30 um grid + 20 um refinement case as the discriminator; grid-heating triad gated; no permittivity scaling in the claim; EC deposition as a post-campaign option | S (gate, diagnostics); M (EC option) | low for the gate; the 20 um case costs 18-20 GPU-h; EC option may heat for drifting beams (Adams et al. 2025) |
| 2 cost | CUDA-graph capture of the full step; periodic cell sort; TCC/Linux host if available; re-scope "converged" and price one long (>= 50 us) development case | S-M (graph, sort); external (host) | graph capture must be bitwise-tested; the long case is 10-20 GPU-h; no literature precedent for our budget |
| 3 neutrals | wall-recycling term in the inventory now; (r,z) view-factor neutrals (Katz-Mikellides) for v2.0; relaxation default off; net utilisation reported | S (recycling); M-L (view factors) | fixed point moves (n_g* up ~50 % before S responds); a breathing closure may have no plateau at all |
| 4a CEX / elastic Xe+-Xe | implement with hash-bound Miller 2002 tables; report as sensitivity | S-M | small in-channel effect expected; blocker only with a plume region |
| 4b SEE | Vaughan yield for BN, Hobbs-Wesson cap, cusp-local diagnostics; SEE-off as the sensitivity | M | space-charge-limited cusp sheaths; O(1) change in electron energy balance |
| 4c anomalous transport | declare absence; add Bohm-scattering sensitivity block (alpha = 1/64, 1/16); diagnose axisymmetric mobility | S | the campaign cannot bound the physical transport; Bohm block is a bracket, not a model |
| 4d exit boundary | plume extension (proposal 3.4a) plus current-continuity injection variant; freeze one, report the others | M | Brandt et al. 2016 found 20 x 5 mm still too small for plume ratios; adds cells and cost |
| 4e cusp sheaths | per-cusp leak width / lambda_D / sheath drop diagnostics; refinement case decides | S | none beyond blocker 1 |
| 5 plateau / statistics | tightened criterion + spectral check + closure statement; long neutral-dynamic case before freezing; 3 seeds + W variant; benchmark-style profile convergence | S (criterion); M (long case) | if the closure breathes, the campaign becomes period-averaged and 2-3x longer |
| 6 validation | report the literature's observables; context panel with Brandt/Keller numbers labelled by closure; no validation claim | S-M | none; the honest outcome is "no experiment at this point" |

Effort: S < 1 day, M 1-3 days, L > 3 days, before GPU time.

---

## 8. Bibliography (116 entries, alphabetical; DOI or handle for every entry; all verified 2026-09-03)

1. Adam, J. C.; Heron, A.; Laval, G. "Study of stationary plasma thrusters using two-dimensional fully kinetic simulations." *Physics of Plasmas* 11, 295-305 (2004). doi:10.1063/1.1632904
2. Adams, Luke C.; Werner, Gregory R.; Cary, John R. "Grid instability growth rates for explicit, electrostatic momentum- and energy-conserving particle-in-cell algorithms." *Physics of Plasmas* 32, 093905 (2025). doi:10.1063/5.0271598
3. Barnes, D. C.; Chacon, L. "Finite spatial-grid effects in energy-conserving particle-in-cell algorithms." *Computer Physics Communications* 258, 107560 (2021). doi:10.1016/j.cpc.2020.107560 (arXiv:1910.10833)
4. Barral, Serge; Ahedo, Eduardo. "Low-frequency model of breathing oscillations in Hall discharges." *Physical Review E* 79, 046401 (2009). doi:10.1103/physreve.79.046401
5. Birdsall, C. K.; Langdon, A. B. *Plasma Physics via Computer Simulation.* IOP Publishing (1991). doi:10.1887/0750301171
6. Birdsall, Charles K.; Maron, Neil. "Plasma self-heating and saturation due to numerical instabilities." *Journal of Computational Physics* 36, 1-19 (1980). doi:10.1016/0021-9991(80)90171-0
7. Boeuf, J. P.; Garrigues, L. "Low frequency oscillations in a stationary plasma thruster." *Journal of Applied Physics* 84, 3541-3554 (1998). doi:10.1063/1.368529
8. Boeuf, Jean-Pierre. "Tutorial: Physics and modeling of Hall thrusters." *Journal of Applied Physics* 121, 011101 (2017). doi:10.1063/1.4972269
9. Boeuf, J. P.; Garrigues, L. "E x B electron drift instability in Hall thrusters: Particle-in-cell simulations vs. theory." *Physics of Plasmas* 25, 061204 (2018). doi:10.1063/1.5017033
10. Bowers, K. J. "Accelerating a Particle-in-Cell Simulation Using a Hybrid Counting Sort." *Journal of Computational Physics* 173, 393-411 (2001). doi:10.1006/jcph.2001.6851
11. Brackbill, J. U. "On energy and momentum conservation in particle-in-cell plasma simulation." *Journal of Computational Physics* 317, 405-427 (2016). doi:10.1016/j.jcp.2016.04.050
12. Brandt, Tim; Schneider, Ralf; Duras, Julia; Kahnfeld, Daniel; Hey, Franz Georg; Kersten, Holger; Jansen, Frank; Braxmaier, Claus. "Particle-in-Cell Simulation of a Down-Scaled HEMP Thruster." *Transactions of the Japan Society for Aeronautical and Space Sciences, Aerospace Technology Japan* 14 (ists30), Pb_235-Pb_242 (2016). doi:10.2322/tastj.14.Pb_235
13. Campanell, M. D.; Khrabrov, A. V.; Kaganovich, I. D. "Absence of Debye Sheaths due to Secondary Electron Emission." *Physical Review Letters* 108, 255001 (2012). doi:10.1103/physrevlett.108.255001
14. Carlsson, Johan; Kaganovich, Igor; Powis, Andrew; Raitses, Yevgeny; Romadanov, Ivan; Smolyakov, Andrei. "Particle-in-cell simulations of anomalous transport in a Penning discharge." *Physics of Plasmas* 25, 061201 (2018). doi:10.1063/1.5017467
15. Cavalier, J.; Lemoine, N.; Bonhomme, G.; Tsikata, S.; Honore, C.; Gresillon, D. "Hall thruster plasma fluctuations identified as the E x B electron drift instability: Modeling and fitting on experimental data." *Physics of Plasmas* 20, 082107 (2013). doi:10.1063/1.4817743
16. Chacon, L.; Chen, G.; Barnes, D. C. "A charge- and energy-conserving implicit, electrostatic particle-in-cell algorithm on mapped computational meshes." *Journal of Computational Physics* 233, 1-9 (2013). doi:10.1016/j.jcp.2012.07.042
17. Chacon, L.; Chen, G. "A curvilinear, fully implicit, conservative electromagnetic PIC algorithm in multiple dimensions." *Journal of Computational Physics* 316, 578-597 (2016). doi:10.1016/j.jcp.2016.03.070
18. Chapurin, O.; Smolyakov, A. I.; Hagelaar, G.; Raitses, Y. "On the mechanism of ionization oscillations in Hall thrusters." *Journal of Applied Physics* 129, 233307 (2021). doi:10.1063/5.0049105
19. Charoy, T.; Boeuf, J. P.; Bourdon, A.; Carlsson, J. A.; Chabert, P.; Cuenot, B.; Eremin, D.; Garrigues, L.; Hara, K.; Kaganovich, I. D.; Powis, A. T.; Smolyakov, A.; Sydorenko, D.; Tavant, A.; Vermorel, O.; Villafana, W. "2D axial-azimuthal particle-in-cell benchmark for low-temperature partially magnetized plasmas." *Plasma Sources Science and Technology* 28, 105010 (2019). doi:10.1088/1361-6595/ab46c5
20. Chen, G.; Chacon, L.; Barnes, D. C. "An energy- and charge-conserving, implicit, electrostatic particle-in-cell algorithm." *Journal of Computational Physics* 230, 7018-7036 (2011). doi:10.1016/j.jcp.2011.05.031
21. Cho, Shinatora; Komurasaki, Kimiya; Arakawa, Yoshihiro. "Kinetic particle simulation of discharge and wall erosion of a Hall thruster." *Physics of Plasmas* 20, 063501 (2013). doi:10.1063/1.4810798
22. Cho, Shinatora; Watanabe, Hiroki; Kubota, Kenichi; Iihara, Shigeyasu; Fuchigami, Kenji; Uematsu, Kazuo; Funaki, Ikkoh. "Study of electron transport in a Hall thruster by axial-radial fully kinetic particle simulation." *Physics of Plasmas* 22, 103523 (2015). doi:10.1063/1.4935049
23. Colella, Phillip; Norgaard, Peter C. "Controlling self-force errors at refinement boundaries for AMR-PIC." *Journal of Computational Physics* 229, 947-957 (2010). doi:10.1016/j.jcp.2009.07.004
24. Courtney, Daniel; Lozano, Paulo; Martinez-Sanchez, Manuel. "Continued Investigation of Diverging Cusped Field Thruster." *44th AIAA/ASME/SAE/ASEE Joint Propulsion Conference and Exhibit* (2008). doi:10.2514/6.2008-4631
25. Croes, Vivien; Lafleur, Trevor; Bonaventura, Zdenek; Bourdon, Anne; Chabert, Pascal. "2D particle-in-cell simulations of the electron drift instability and associated anomalous electron transport in Hall-effect thrusters." *Plasma Sources Science and Technology* 26, 034001 (2017). doi:10.1088/1361-6595/aa550f
26. Decyk, Viktor K.; Singh, Tajendra V. "Particle-in-Cell algorithms for emerging computer architectures." *Computer Physics Communications* 185, 708-719 (2014). doi:10.1016/j.cpc.2013.10.013
27. Dunaevsky, A.; Raitses, Y.; Fisch, N. J. "Secondary electron emission from dielectric materials of a Hall thruster with segmented electrodes." *Physics of Plasmas* 10, 2574-2577 (2003). doi:10.1063/1.1568344
28. Duras, J.; Matyash, K.; Tskhakaya, D.; Kalentev, O.; Schneider, R. "Self-Force in 1D Electrostatic Particle-in-Cell Codes for Non-Equidistant Grids." *Contributions to Plasma Physics* 54, 697-711 (2014). doi:10.1002/ctpp.201300060
29. Duras, J.; Kahnfeld, D.; Bandelow, G.; Kemnitz, S.; Luskow, K.; Matthias, P.; Koch, N.; Schneider, R. "Ion angular distribution simulation of the Highly Efficient Multistage Plasma Thruster." *Journal of Plasma Physics* 83, 595830107 (2017). doi:10.1017/s0022377817000125
30. Eremin, D. "An energy- and charge-conserving electrostatic implicit particle-in-cell algorithm for simulations of collisional bounded plasmas." *Journal of Computational Physics* 452, 110934 (2022). doi:10.1016/j.jcp.2021.110934
31. Fahey, Thomas; Muffatti, Angus; Ogawa, Hideaki. "High Fidelity Multi-Objective Design Optimization of a Downscaled Cusped Field Thruster." *Aerospace* 4, 55 (2017). doi:10.3390/aerospace4040055
32. Faraji, F.; Reza, M.; Knoll, A. "Enhancing one-dimensional particle-in-cell simulations to self-consistently resolve instability-induced electron transport in Hall thrusters." *Journal of Applied Physics* 131, 193302 (2022). doi:10.1063/5.0090853
33. Fubiani, G.; Garrigues, L.; Hagelaar, G.; Kohen, N.; Boeuf, J. P. "Modeling of plasma transport and negative ion extraction in a magnetized radio-frequency plasma source." *New Journal of Physics* 19, 015002 (2017). doi:10.1088/1367-2630/19/1/015002
34. Furman, M.; Pivi, M. "Probabilistic model for the simulation of secondary electron emission." *Physical Review Special Topics - Accelerators and Beams* 5, 124404 (2002). doi:10.1103/physrevstab.5.124404
35. Garrigues, L.; Tezenas du Montcel, B.; Fubiani, G.; Bertomeu, F.; Deluzet, F.; Narski, J. "Application of sparse grid combination techniques to low temperature plasmas particle-in-cell simulations. I. Capacitively coupled radio frequency discharges." *Journal of Applied Physics* 129, 153303 (2021). doi:10.1063/5.0044363
36. Garrigues, L.; Tezenas du Montcel, B.; Fubiani, G.; Reman, B. C. G. "Application of sparse grid combination techniques to low temperature plasmas Particle-In-Cell simulations. II. Electron drift instability in a Hall thruster." *Journal of Applied Physics* 129, 153304 (2021). doi:10.1063/5.0044865
37. Garrigues, L.; Chung-To-Sang, M.; Fubiani, G.; Guillet, C.; Deluzet, F.; Narski, J. "Acceleration of particle-in-cell simulations using sparse grid algorithms. II. Application to partially magnetized low temperature plasmas." *Physics of Plasmas* 31, 073908 (2024). doi:10.1063/5.0211220
38. Gildea, Stephen R. *Fully kinetic modeling of a divergent cusped-field thruster.* S.M. thesis, Massachusetts Institute of Technology (2009). hdl:1721.1/54613
39. Gildea, Stephen; Matlock, Taylor; Lozano, Paulo; Martinez-Sanchez, Manuel. "Low Frequency Oscillations in the Diverging Cusped-Field Thruster." *46th AIAA/ASME/SAE/ASEE Joint Propulsion Conference and Exhibit* (2010). doi:10.2514/6.2010-7014
40. Gildea, Stephen R.; Matlock, Taylor S.; Martinez-Sanchez, Manuel; Hargus, William A. "Erosion Measurements in a Low-Power Cusped-Field Plasma Thruster." *Journal of Propulsion and Power* 29, 906-918 (2013). doi:10.2514/1.b34607
41. Hara, Kentaro; Sekerak, Michael J.; Boyd, Iain D.; Gallimore, Alec D. "Perturbation analysis of ionization oscillations in Hall effect thrusters." *Physics of Plasmas* 21, 122103 (2014). doi:10.1063/1.4903843
42. Hara, Kentaro. "An overview of discharge plasma modeling for Hall effect thrusters." *Plasma Sources Science and Technology* 28, 044001 (2019). doi:10.1088/1361-6595/ab0f70
43. Hershkowitz, Noah; Leung, K. N.; Romesser, Thomas. "Plasma Leakage Through a Low-beta Line Cusp." *Physical Review Letters* 35, 277-280 (1975). doi:10.1103/physrevlett.35.277
44. Hobbs, G. D.; Wesson, J. A. "Heat flow through a Langmuir sheath in the presence of electron emission." *Plasma Physics* 9, 85-87 (1967). doi:10.1088/0032-1028/9/1/410
45. Hockney, R. W. "Measurements of collision and heating times in a two-dimensional thermal computer plasma." *Journal of Computational Physics* 8, 19-44 (1971). doi:10.1016/0021-9991(71)90032-5
46. Hockney, R. W.; Eastwood, J. W. *Computer Simulation Using Particles.* Adam Hilger / CRC Press (1988). doi:10.1201/9781439822050
47. Hu, Peng; Liu, Hui; Gao, Yuanyuan; Yu, Daren. "Effects of magnetic field strength in the discharge channel on the performance of a multi-cusped field thruster." *AIP Advances* 6, 095003 (2016). doi:10.1063/1.4962548
48. Juhasz, Zoltan; Durian, Jan; Derzsi, Aranka; Matejcik, Stefan; Donko, Zoltan; Hartmann, Peter. "Efficient GPU implementation of the Particle-in-Cell/Monte-Carlo collisions method for 1D simulation of low-pressure capacitively coupled plasmas." *Computer Physics Communications* 263, 107913 (2021). doi:10.1016/j.cpc.2021.107913
49. Kaganovich, Igor D.; Smolyakov, Andrei; Raitses, Yevgeny; Ahedo, Eduardo; Mikellides, Ioannis G.; Jorns, Benjamin; Taccogna, Francesco; Gueroult, Renaud; Tsikata, Sedina; Bourdon, Anne; Boeuf, Jean-Pierre; Keidar, Michael; Powis, Andrew Tasman; Merino, Mario; Cappelli, Mark; Hara, Kentaro; Carlsson, Johan A.; Fisch, Nathaniel J.; Chabert, Pascal; Schweigert, Irina; Lafleur, Trevor; Matyash, Konstantin; Khrabrov, Alexander V.; Boswell, Rod W.; Fruchtman, Amnon. "Physics of E x B discharges relevant to plasma propulsion and similar technologies." *Physics of Plasmas* 27, 120601 (2020). doi:10.1063/5.0010135
50. Kahnfeld, D.; Schneider, R.; Matyash, K.; Kalentev, O.; Kemnitz, S.; Duras, J.; Luskow, K.; Bandelow, G. "Solution of Poisson's Equation in Electrostatic Particle-in-Cell Simulation." *Plasma Physics and Technology* 3, 66-71 (2016). doi:10.14311/ppt.2016.2.66 (title as deposited: "Solutioin of Poisson's Equation in Electrostatic Particle-on-cell Simulation")
51. Kahnfeld, D.; Heidemann, R.; Duras, J.; Matthias, P.; Bandelow, G.; Luskow, K.; Kemnitz, S.; Matyash, K.; Schneider, R. "Breathing modes in HEMP thrusters." *Plasma Sources Science and Technology* 27, 124002 (2018). doi:10.1088/1361-6595/aaf29a
52. Kahnfeld, Daniel; Duras, Julia; Matthias, Paul; Kemnitz, Stefan; Arlinghaus, Peter; Bandelow, Gunnar; Matyash, Konstantin; Koch, Norbert; Schneider, Ralf. "Numerical modeling of high efficiency multistage plasma thrusters for space applications." *Reviews of Modern Plasma Physics* 3, 11 (2019). doi:10.1007/s41614-019-0030-4
53. Kalentev, O.; Matyash, K.; Duras, J.; Luskow, K. F.; Schneider, R.; Koch, N.; Schirra, M. "Electrostatic Ion Thrusters - Towards Predictive Modeling." *Contributions to Plasma Physics* 54, 235-248 (2014). doi:10.1002/ctpp.201300038
54. Katz, Ira; Mikellides, Ioannis G. "Neutral gas free molecular flow algorithm including ionization and walls for use in plasma simulations." *Journal of Computational Physics* 230, 1454-1464 (2011). doi:10.1016/j.jcp.2010.11.013
55. Keller, Andreas; Kohler, Peter; Hey, Franz Georg; Berger, Marcel; Braxmaier, Claus; Feili, Davar; Weise, Dennis; Johann, Ulrich. "Parametric Study of HEMP-Thruster Downscaling to uN Thrust Levels." *IEEE Transactions on Plasma Science* 43, 45-53 (2015). doi:10.1109/tps.2014.2321095
56. Kornfeld, G.; Koch, N.; Harmann, H.-P. "Physics and Evolution of HEMP-Thrusters." IEPC-2007-108, 30th International Electric Propulsion Conference, Florence (2007). No DOI; cited as in `modern/docs/REFERENCES.md`.
57. Lacina, J. "Similarity rules in plasma physics." *Plasma Physics* 13, 303-312 (1971). doi:10.1088/0032-1028/13/4/003
58. Lafleur, T.; Baalrud, S. D.; Chabert, P. "Theory for the anomalous electron transport in Hall effect thrusters. I. Insights from particle-in-cell simulations." *Physics of Plasmas* 23, 053502 (2016). doi:10.1063/1.4948495
59. Lafleur, T.; Martorelli, R.; Chabert, P.; Bourdon, A. "Anomalous electron transport in Hall-effect thrusters: Comparison between quasi-linear kinetic theory and particle-in-cell simulations." *Physics of Plasmas* 25, 061202 (2018). doi:10.1063/1.5017626
60. Lafleur, T.; Chabert, P.; Bourdon, A. "The origin of the breathing mode in Hall thrusters and its stabilization." *Journal of Applied Physics* 130, 053305 (2021). doi:10.1063/5.0057095
61. Langdon, A. Bruce. "Effects of the spatial grid in simulation plasmas." *Journal of Computational Physics* 6, 247-267 (1970). doi:10.1016/0021-9991(70)90024-0
62. Lapenta, Giovanni. "Exactly energy conserving semi-implicit particle in cell formulation." *Journal of Computational Physics* 334, 349-366 (2017). doi:10.1016/j.jcp.2017.01.002
63. Lapenta, Giovanni. "Advances in the Implementation of the Exactly Energy Conserving Semi-Implicit (ECsim) Particle-in-Cell Method." *Physics* 5, 72-89 (2023). doi:10.3390/physics5010007
64. Leung, K. N.; Hershkowitz, Noah; MacKenzie, K. R. "Plasma confinement by localized cusps." *The Physics of Fluids* 19, 1045-1053 (1976). doi:10.1063/1.861575
65. Liu, Hui; Chen, Peng-Bo; Zhao, Yin-Jian; Yu, Da-Ren. "Particle-in-cell simulation for different magnetic mirror effects on the plasma distribution in a cusped field thruster." *Chinese Physics B* 24, 085202 (2015). doi:10.1088/1674-1056/24/8/085202
66. Lucca Fabris, Andrea; Young, Christopher V.; Manente, Marco; Pavarin, Daniele; Cappelli, Mark A. "Ion Velocimetry Measurements and Particle-In-Cell Simulation of a Cylindrical Cusped Plasma Accelerator." *IEEE Transactions on Plasma Science* 43, 54-63 (2015). doi:10.1109/tps.2014.2321743
67. MacDonald, N. A.; Cappelli, M. A.; Gildea, S. R.; Martinez-Sanchez, M.; Hargus, W. A. "Laser-induced fluorescence velocity measurements of a diverging cusped-field thruster." *Journal of Physics D: Applied Physics* 44, 295203 (2011). doi:10.1088/0022-3727/44/29/295203
68. MacDonald, N. A.; Young, C. V.; Cappelli, M. A.; Hargus, W. A. "Ion velocity and plasma potential measurements of a cylindrical cusped field thruster." *Journal of Applied Physics* 111, 093303 (2012). doi:10.1063/1.4707953
69. MacDonald, N. A.; Cappelli, M. A.; Hargus, W. A. "Time-synchronized continuous wave laser-induced fluorescence axial velocity measurements in a diverging cusped field thruster." *Journal of Physics D: Applied Physics* 47, 115204 (2014). doi:10.1088/0022-3727/47/11/115204
70. Markidis, Stefano; Lapenta, Giovanni. "The energy conserving particle-in-cell method." *Journal of Computational Physics* 230, 7037-7052 (2011). doi:10.1016/j.jcp.2011.05.033
71. Marks, Thomas A.; Jorns, Benjamin A. "Challenges with the self-consistent implementation of closure models for anomalous electron transport in fluid simulations of Hall thrusters." *Plasma Sources Science and Technology* 32, 045016 (2023). doi:10.1088/1361-6595/accd18
72. Matlock, Taylor; Daspit, Ryan; Batishchev, Oleg; Lozano, Paulo; Martinez-Sanchez, Manuel. "Spectroscopic and Electrostatic Investigation of the Diverging Cusped-Field Thruster." *45th AIAA/ASME/SAE/ASEE Joint Propulsion Conference and Exhibit* (2009). doi:10.2514/6.2009-4813
73. Matthias, Paul; Kahnfeld, Daniel; Schneider, Ralf; Yeo, Suk Hyun; Ogawa, Hideaki. "Particle-in-cell simulation of an optimized high-efficiency multistage plasma thruster." *Contributions to Plasma Physics* 59, e201900028 (2019). doi:10.1002/ctpp.201900028
74. Matthias, Paul; Kahnfeld, Daniel; Kemnitz, Stefan; Duras, Julia; Koch, Norbert; Schneider, Ralf. "Similarity scaling - application and limits for high-efficiency-multistage-plasma-thruster particle-in-cell modelling." *Contributions to Plasma Physics* 60, e201900199 (2020). doi:10.1002/ctpp.201900199
75. Matyash, Konstantin; Schneider, Ralf; Mutzke, Andreas; Kalentev, Oleksandr; Taccogna, Francesco; Koch, Norbert; Schirra, Martin. "Kinetic Simulations of SPT and HEMP Thrusters Including the Near-Field Plume Region." *IEEE Transactions on Plasma Science* 38, 2274-2280 (2010). doi:10.1109/tps.2010.2056936
76. Mikellides, Ioannis G.; Katz, Ira. "Numerical simulations of Hall-effect plasma accelerators on a magnetic-field-aligned mesh." *Physical Review E* 86, 046703 (2012). doi:10.1103/physreve.86.046703
77. Miller, J. Scott; Pullins, Steve H.; Levandier, Dale J.; Chiu, Yu-hui; Dressler, Rainer A. "Xenon charge exchange cross sections for electrostatic thruster models." *Journal of Applied Physics* 91, 984-991 (2002). doi:10.1063/1.1426246
78. Muralikrishnan, Sriramkrishnan; Cerfon, Antoine J.; Frey, Matthias; Ricketson, Lee F.; Adelmann, Andreas. "Sparse grid-based adaptive noise reduction strategy for particle-in-cell schemes." *Journal of Computational Physics: X* 11, 100094 (2021). doi:10.1016/j.jcpx.2021.100094
79. NVIDIA Technical Blog. "Getting Started with CUDA Graphs" (Alan Gray, 2019). https://developer.nvidia.com/blog/cuda-graphs/ (grey literature; accessed 2026-09-03).
80. Parra, F. I.; Ahedo, E.; Fife, J. M.; Martinez-Sanchez, M. "A two-dimensional hybrid model of the Hall thruster discharge." *Journal of Applied Physics* 100, 023304 (2006). doi:10.1063/1.2219165
81. Petronio, Federico; Alvarez Laguna, Alejandro; Bourdon, Anne; Chabert, Pascal. "Study of the breathing mode development in Hall thrusters using hybrid simulations." *Journal of Applied Physics* 135, 073301 (2024). doi:10.1063/5.0188859
82. Powis, Andrew T.; Carlsson, Johan A.; Kaganovich, Igor D.; Raitses, Yevgeny; Smolyakov, Andrei. "Scaling of spoke rotation frequency within a Penning discharge." *Physics of Plasmas* 25, 072110 (2018). doi:10.1063/1.5038733
83. Raitses, Y.; Fisch, N. J. "Parametric investigations of a nonconventional Hall thruster." *Physics of Plasmas* 8, 2579-2586 (2001). doi:10.1063/1.1355318
84. Reza, Maryam; Faraji, Farbod; Knoll, Aaron. "Concept of the generalized reduced-order particle-in-cell scheme and verification in an axial-azimuthal Hall thruster configuration." *Journal of Physics D: Applied Physics* 56, 175201 (2023). doi:10.1088/1361-6463/acbb15
85. Ricketson, L. F.; Cerfon, A. J. "Sparse grid techniques for particle-in-cell schemes." *Plasma Physics and Controlled Fusion* 59, 024002 (2017). doi:10.1088/1361-6587/59/2/024002
86. Ricketson, Lee F.; Hu, Jingwei. "An explicit, energy-conserving particle-in-cell scheme." *Journal of Computational Physics* 537, 114098 (2025). doi:10.1016/j.jcp.2025.114098
87. Savard, N.; Fubiani, G.; Eremin, D.; Dehnel, M. "Impact of particle number and cell size in fully implicit charge- and energy-conserving particle-in-cell schemes." *Physics of Plasmas* 32, 073903 (2025). doi:10.1063/5.0265414
88. Schneider, R.; Matyash, K.; Kalentev, O.; Taccogna, F.; Koch, N.; Schirra, M. "Particle-in-Cell Simulations for Ion Thrusters." *Contributions to Plasma Physics* 49, 655-661 (2009). doi:10.1002/ctpp.200910070
89. Smirnov, A.; Raitses, Y.; Fisch, N. J. "Plasma measurements in a 100 W cylindrical Hall thruster." *Journal of Applied Physics* 95, 2283-2292 (2004). doi:10.1063/1.1642734
90. Smirnov, A.; Raitses, Y.; Fisch, N. J. "Electron cross-field transport in a low power cylindrical Hall thruster." *Physics of Plasmas* 11, 4922-4933 (2004). doi:10.1063/1.1791639
91. Sydorenko, D.; Smolyakov, A.; Kaganovich, I.; Raitses, Y. "Kinetic simulation of secondary electron emission effects in Hall thrusters." *Physics of Plasmas* 13, 014501 (2006). doi:10.1063/1.2158698
92. Szabo, James Joseph, Jr. *Fully kinetic numerical modeling of a plasma thruster.* Ph.D. thesis, Massachusetts Institute of Technology (2001). hdl:1721.1/8889
93. Szabo, James; Warner, Noah; Martinez-Sanchez, Manuel; Batishchev, Oleg. "Full Particle-In-Cell Simulation Methodology for Axisymmetric Hall Effect Thrusters." *Journal of Propulsion and Power* 30, 197-208 (2014). doi:10.2514/1.b34774
94. Taccogna, Francesco; Longo, Savino; Capitelli, Mario; Schneider, Ralf. "Self-similarity in Hall plasma discharges: Applications to particle models." *Physics of Plasmas* 12, 053502 (2005). doi:10.1063/1.1877517
95. Taccogna, Francesco; Longo, Savino; Capitelli, Mario. "Plasma sheaths in Hall discharge." *Physics of Plasmas* 12, 093506 (2005). doi:10.1063/1.2015257
96. Taccogna, F.; Schneider, R.; Longo, S.; Capitelli, M. "Kinetic simulations of a plasma thruster." *Plasma Sources Science and Technology* 17, 024003 (2008). doi:10.1088/0963-0252/17/2/024003
97. Taccogna, Francesco; Minelli, Pierpaolo. "Three-dimensional particle-in-cell model of Hall thruster: The discharge channel." *Physics of Plasmas* 25, 061208 (2018). doi:10.1063/1.5023482
98. Taccogna, Francesco; Minelli, Pierpaolo; Asadi, Zahra; Bogopolsky, Guillaume. "Numerical studies of the ExB electron drift instability in Hall thrusters." *Plasma Sources Science and Technology* 28, 064002 (2019). doi:10.1088/1361-6595/ab08af
99. Taccogna, F.; Garrigues, L. "Latest progress in Hall thrusters plasma modelling." *Reviews of Modern Plasma Physics* 3, 12 (2019). doi:10.1007/s41614-019-0033-1
100. Tavant, Antoine; Croes, Vivien; Lucken, Romain; Lafleur, Trevor; Bourdon, Anne; Chabert, Pascal. "The effects of secondary electron emission on plasma sheath characteristics and electron transport in an E x B discharge via kinetic simulations." *Plasma Sources Science and Technology* 27, 124001 (2018). doi:10.1088/1361-6595/aaeccd
101. Tondu, T.; Belhaj, M.; Inguimbert, V. "Electron-emission yield under electron impact of ceramics used as channel materials in Hall-effect thrusters." *Journal of Applied Physics* 110, 093301 (2011). doi:10.1063/1.3653820
102. Tskhakaya, D.; Matyash, K.; Schneider, R.; Taccogna, F. "The Particle-In-Cell Method." *Contributions to Plasma Physics* 47, 563-594 (2007). doi:10.1002/ctpp.200710072
103. Turner, M. M. "Kinetic properties of particle-in-cell simulations compromised by Monte Carlo collisions." *Physics of Plasmas* 13, 033506 (2006). doi:10.1063/1.2169752
104. Turner, M. M.; Derzsi, A.; Donko, Z.; Eremin, D.; Kelly, S. J.; Lafleur, T.; Mussenbrock, T. "Simulation benchmarks for low-pressure plasmas: Capacitive discharges." *Physics of Plasmas* 20, 013507 (2013). doi:10.1063/1.4775084
105. Turner, M. M. "Verification of particle-in-cell simulations with Monte Carlo collisions." *Plasma Sources Science and Technology* 25, 054007 (2016). doi:10.1088/0963-0252/25/5/054007
106. Ueda, Hiroko; Omura, Yoshiharu; Matsumoto, Hiroshi; Okuzawa, Takashi. "A study of the numerical heating in electrostatic particle simulations." *Computer Physics Communications* 79, 249-259 (1994). doi:10.1016/0010-4655(94)90071-x
107. Vahedi, V.; Surendra, M. "A Monte Carlo collision model for the particle-in-cell method: applications to argon and oxygen discharges." *Computer Physics Communications* 87, 179-198 (1995). doi:10.1016/0010-4655(94)00171-w
108. Vaughan, J. R. M. "A new formula for secondary emission yield." *IEEE Transactions on Electron Devices* 36, 1963-1967 (1989). doi:10.1109/16.34278
109. Vay, J.-L.; Colella, P.; Kwan, J. W.; McCorquodale, P.; Serafini, D. B.; Friedman, A.; Grote, D. P.; Westenskow, G.; Adam, J.-C.; Heron, A.; Haber, I. "Application of adaptive mesh refinement to particle-in-cell simulations of plasmas and beams." *Physics of Plasmas* 11, 2928-2934 (2004). doi:10.1063/1.1689669
110. Villafana, W.; Petronio, F.; Denig, A. C.; Jimenez, M. J.; Eremin, D.; Garrigues, L.; Taccogna, F.; Alvarez-Laguna, A.; Boeuf, J. P.; Bourdon, A.; Chabert, P.; Charoy, T.; Cuenot, B.; Hara, K.; Pechereau, F.; Smolyakov, A.; Sydorenko, D.; Tavant, A.; Vermorel, O. "2D radial-azimuthal particle-in-cell benchmark for E x B discharges." *Plasma Sources Science and Technology* 30, 075002 (2021). doi:10.1088/1361-6595/ac0a4a
111. Villafana, W.; Cuenot, B.; Vermorel, O. "3D particle-in-cell study of the electron drift instability in a Hall Thruster using unstructured grids." *Physics of Plasmas* 30, 033503 (2023). doi:10.1063/5.0133963
112. Yeo, Suk H.; Ogawa, Hideaki; Matthias, Paul; Kahnfeld, Daniel; Schneider, Ralf. "Multiobjective Optimization and Particle-In-Cell Simulation of Cusped Field Thrusters for Microsatellite Platforms." *Journal of Spacecraft and Rockets* 57, 603-611 (2020). doi:10.2514/1.A34584
113. Yeo, Suk Hyun; Ogawa, Hideaki. "Multi-Objective Design Optimization of Cusped Field Thruster via Surrogate-Assisted Evolutionary Algorithms." *Journal of Propulsion and Power* 38, 973-988 (2022). doi:10.2514/1.b38854
114. Yeo, Suk Hyun; Gadisa, Dinaol; Ogawa, Hideaki; Bang, HyoChoong. "Multi-objective design optimization and physics-based sensitivity analysis of field emission electric propulsion for CubeSat platforms." *Aerospace Science and Technology* 154, 109516 (2024). doi:10.1016/j.ast.2024.109516
115. Zhao, Y. J.; Liu, H.; Yu, D. R.; Hu, P.; Wu, H. "Particle-in-cell simulations for the effect of magnetic field strength on a cusped field thruster." *Journal of Physics D: Applied Physics* 47, 045201 (2013). doi:10.1088/0022-3727/47/4/045201
116. Zhao, Zhongping; Zhao, Yinjian. "A review of 3D particle-in-cell simulations for electron drift instability in Hall thrusters." *Plasma Science and Technology* (2026, in press at verification time). doi:10.1088/2058-6272/ae69a9

Also carried by the repository and referred to above without re-verification here: Muffatti and
Ogawa, ISTS 2017-b-32 (`modern/docs/REFERENCES.md`).

### 8.1 Verification notes

- Crossref deposits the Kahnfeld et al. 2016 title with two typographical errors
  ("Solutioin", "Particle-on-cell"); the entry keeps the corrected reading and notes the
  deposited form.
- Barnes and Chacon 2021 appears in Crossref with issued date 2021 (volume 258) and in
  OpenAlex/Semantic Scholar as 2019/2020 (online first); the print citation is used.
- Szabo et al. 2014 is dated 2013 by OpenAlex (online first) and 2014 by Crossref (volume 30);
  the print citation is used.
- Zhao and Zhao 2026 had no volume/page at verification time.
- The count in the heading is the number of numbered entries; the ISTS paper is not counted.
