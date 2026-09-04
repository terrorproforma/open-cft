# Making the `cft_revival.pic2d` PIC-MCC 5-10x faster without invalidating its physics claims: a cited literature review

**Status: literature review (document only). No code, spec, protocol or result was changed.**
Prepared 2026-09-04 against `origin/feat/sota-foundation` at `a0235676` on branch
`docs/pic-acceleration-review`. Companion reviews: `pic-mcc-blockers.md` (the 116-reference PIC
review of 2026-09-03: Debye resolution, cost, neutrals, missing physics, plateau criteria,
validation targets - this document extends its blockers 1 and 2 and does not repeat them),
`reduced-models-cusp-topology-blockers.md`, `surrogate-mdo-validation-blockers.md`,
`twt-ppm-physics-for-hemp.md`, and the roadmap in `../LITERATURE_SYNTHESIS.md`.

## 0. Scope, method and honesty rules

**Question.** The code is an explicit, momentum-conserving, electrostatic, 2-D axisymmetric (r,z)
PIC-MCC with a direct block-inverse Poisson solve on an NVIDIA GPU. It is bound by the
finite-grid-heating constraint (measured onset at Delta/lambda_D ~ 3.2, hard gate pi on the
interval-averaged peak) to 33 um / 25 um grids and 1.4 / 1.0 ps steps, and by the ion transit
time to 5-7 us of simulated time, i.e. 5-20 h per run. Which published methods could make **this**
code 5-10x faster, what do they do to the quantities the repository claims (I_d, S and net
utilisation, the neutral fixed point n_g*, peak n_e / T_e, per-cusp wall fluxes and energies,
per-cusp sheath drops, I_beam / thrust, IEDF and divergence, the energy ledger), what would
they cost to implement in Warp, and what evidence would the repository need before a result
produced with them could be admitted?

**Method.** Six method families (sections 2-7), each with the key references, what the field
reports, what would change in our code, the expected speed-up computed against the measured
cost anatomy of section 1, the physical risk to each claim, the implementation cost in days,
and the verification protocol. Section 8 is the ranked recommendation table, section 9 the
applicability matrix (method x claim), section 10 the honest gaps, section 11 the bibliography.

**Verification policy.** Every DOI in section 11 was resolved on 2026-09-04 against the Crossref
works record (`api.crossref.org/works/<DOI>`: title, authors, container, year, volume, pages
quoted as deposited); one conference DOI (Parodi et al. 2022, EUCASS) is not in Crossref and was
resolved through `doi.org` to the publisher's full text. Abstracts were read where deposited
(Crossref, Semantic Scholar, publisher pages); full or partial texts were read for Marks and
Gorodetsky 2025, Powis and Kaganovich 2024, Sun et al. 2023, Taccogna et al. 2023 (section V C),
Petronio et al. 2023 (section on the scaled permittivity), Angus et al. 2024 (OSTI preprint),
Chen et al. 2012 and Yuan et al. 2020 (abstracts and cited passages). Where a statement about a
paper rests on its abstract only, that is said. Nothing that could not be resolved is cited;
four candidate references were dropped for that reason. Grey literature (NVIDIA CUDA Graphs
and MPS documentation, the Groenewald et al. SC24 poster) is named in the text where the
repository already relies on it but is not counted in the bibliography.

**Speed-up arithmetic.** Every speed-up below is derived from the cost model the repository
itself recorded (`spec/pic2d/pic2d-model-v2.1.json`, `cost_table_v2_1`: `ms/step = fixed(grid)
+ 0.733 ms per M particles`, `fixed = 2 (nr+1) sequential row-block launches x ~5 us +
inverse-block reads (nr+1) x 2 x (nz+1)^2 x 8 B at ~1.6 TB/s + node kernels`) and the H100
benchmark of 2026-09-04 (one H100 ~1x an RTX 5090 per process; the step is latency-bound
small kernels; MPS gives 1.54x aggregate over 4 slots, no single run faster). Published
speed-ups are quoted with the problem they were measured on; none of them was measured on a
cusped-field thruster.

### 0.1 The two runs the speed-ups are computed for

| run | grid | rows x nodes/row | particles | measured / modelled step | steps to 3 transits | wall |
| --- | --- | --- | --- | --- | --- | --- |
| channel-only, 33.3 um / 1.4 ps (`pic2d_cft_steady_state_v4`, accepted plateau) | 90 x 720 cells, 3 x 24 mm | 91 x 721 | ~2-4 M at the plateau (W 2.67e4, cell-area parity with the 50 um base) | 2.5 ms/step at preflight, 3.3 ms/step at 2.5 us, 3.5 ms/step averaged over the run (measured); model: 182 launches ~0.9 ms + 0.76 GB block reads ~0.5 ms + particles as the remainder (1.1-2.0 ms) | 5.2 M | 5.0 h recorded (the brief's "~6 h") |
| plume box 48 x 12 mm at 33.3 um / 1.4 ps (v2.1 domain at the v4 resolution; never launched) | 360 x 1440 cells | 361 x 1441 | ~10 M at W parity with v4 | model: 722 launches 3.6 ms + 12.0 GB block reads 7.5 ms + node kernels 0.3 ms + particles ~7.3 ms = ~18.7 ms/step | 8.2 M (transit 3.8 us) | ~42 h at constant N, ~48 h with the recorded particle growth (the brief's "~48 h"); 6 GB of inverse blocks on the device; host factorisation of 361 blocks of 1441^2 (hours on the Windows PC, tens of minutes on the Linux box) |

The two rows have different bottlenecks, and that decides the ranking in section 8: the channel
run is roughly half field solve (launch latency plus block reads) and half particles; the plume
run is ~60 % field solve, and the field-solve share grows as (nz+1)^2 because the dense
inverse blocks are the whole axial row.

---

## 1. Where the time goes, and what a 5-10x would have to remove

Per step, at the two operating points above (model of section 0.1):

| component | channel-33 | plume-33 (48 x 12 mm) | what sets it | which section can move it |
| --- | --- | --- | --- | --- |
| block-Thomas sweeps: 2 (nr+1) dependent kernel launches of m = nz+1 threads | 0.9 ms (~30 %) | 3.6 ms (19 %) | launch latency; one row block at a time | 5 (solver), 6 (launches) |
| inverse-block reads (nr+1) x 2 x m^2 x 8 B | 0.5 ms (~17 %) | 7.5 ms (40 %) | HBM bandwidth; dense m x m inverses | 5 |
| particle kernels (gather, Boris push, boundary classification, MCC, fixed-point deposit) | 1.1-2.0 ms (~50 %) | 7.3 ms (39 %) | 0.733 ms per M particles (fitted on the plume runs), latency-bound at this size | 6, 7, 3 (N), 2 (Delta t) |
| node kernels, reductions, gate moments (~300 launches) | ~0.05-0.3 ms | 0.3 ms | launch latency | 6 |

Three consequences that the literature cannot change:

1. The **number of steps** is set by the physics we claim: 3 ion transits (2.4 us channel
   residence, 3.8 us with the 24 mm plume) at a Delta t that resolves omega_pe (omega_pe Delta t
   <= 0.2 at the peak density) and the electron Courant condition. Only a scheme that tolerates
   omega_pe Delta t > 1 (section 2.3-2.4) or a shorter route to the plateau (section 4) changes
   it; nothing in sections 5-7 does.
2. The **number of particles** is set by the statistics band the repository has measured
   (seed-b <= 1.1 %; W x 0.7 vs base I_d +5.7 %, peak n_e -12 %; the 33 um refinement moved
   the same quantities in the W x 0.7 direction at ~2x its size, so grid and weight effects are
   entangled). Halving N widens a 5-12 % band to 7-17 % (shot noise ~ N^-1/2) and raises the
   noise-driven heating rate (Hockney 1971; Okuda and Birdsall 1970; Birdsall and Langdon 1991).
   "Fewer particles" is therefore only available with variance reduction (section 7), not as a
   free parameter.
3. The **per-particle cost** (0.73 ns per particle-step on the H100 and the 5090 alike) is
   already within a small factor of the bandwidth floor for a float64 five-component particle
   with an int64 fixed-point deposit (~150-200 B moved per particle-step: five coordinates read
   and written, eight field values gathered, four int64 atomics; at 1.6-3 TB/s that is
   0.05-0.12 ns, so the measured 0.73 ns is latency and atomic-contention bound, not
   bandwidth bound). Sorting, fusion and precision (section 6) work on this term.

A 5-10x therefore decomposes as: (a) remove most of the field-solve cost (2.5x on the plume,
1.5x on the channel, section 5); (b) halve the particle cost (section 6); and then, for the
remainder, (c) change what is being resolved - coarser grid and larger Delta t through an
energy-conserving or semi-implicit scheme (section 2), which is the only route to the full
factor and the only one that touches the physics.

---

## 2. Method family 1: relaxing the explicit constraints

### 2.1 What the constraint costs us

The momentum-conserving explicit scheme (bilinear deposition and the same bilinear gather of a
nodal E) is unstable to aliasing once Delta exceeds a few lambda_D (Langdon 1970; Birdsall and
Maron 1980; Ueda et al. 1994; Hockney and Eastwood 1988; Birdsall and Langdon 1991); Adams,
Werner and Cary 2025 give the first systematic growth-rate map over Delta/lambda_D and drift
velocity for the momentum-conserving scheme and show that a higher-order field solve does not
remove it. The repository measured exactly this: the energy-ledger residual ran away once the
peak Delta/lambda_D crossed ~3.2, the accepted 50 um plateau sat on the threshold at 3.17, and
the 33 um refinement changed I_d by +10 %, peak n_e by -21 % and T_e,peak by -25 %. The gate is
now Delta/lambda_D <= pi on the interval-averaged peak node. Every finer rung of the ladder
costs (33/25)^2 = 1.8x in cells, 1.4x in steps (Delta t 1.4 -> 1.0 ps) and, through the
(nz+1)^2 block reads, ~2.5x in field-solve bandwidth.

### 2.2 Explicit energy-conserving PIC (Lewis 1970)

**References.** Lewis 1970 (variational derivation; the scheme conserves energy exactly in the
zero-Delta t limit); Langdon 1973 (its dispersion analysis, the origin of the name in quotes);
Barnes and Chacon 2021 (EC-PIC is stable against aliasing for stationary plasmas and has a
"benign threshold" for drifting finite-temperature plasmas, "usable in practice ... without the
need to resolve Debye lengths spatially"; the momentum-conserving scheme has no such threshold);
Adams, Werner and Cary 2025 (EC linear and quadratic schemes still heat for some drift /
under-resolution combinations, growth rates tabulated); Ricketson and Hu 2025 (a new explicit,
exactly energy-conserving formulation); Brackbill 2016 (energy vs momentum conservation and
the self-force that EC schemes accept); Powis and Kaganovich 2024 (1-D RF-CCP: EC-PIC
"closely matches" the momentum-conserving result and "retains accuracy even for cell sizes up
to 8 times the electron Debye length"; beyond that it loses accuracy "due to poor resolution
of steep gradients within the radio frequency sheath"; a non-uniform grid recovers it with
cells up to 32 lambda_D in the bulk, "an up to 8 times reduction in the number of required
simulation cells"; EC-PIC "does not conserve momentum to machine precision"); Sun et al. 2023
(EC-PIC and direct-implicit PIC in EDIPIC-2D and LTP-PIC on a 2-D CCP: at equal Delta and
Delta t the cost is the same as momentum-conserving PIC; "adopting cell sizes dx/dx0 > 1 can
allow for order of magnitude reductions in compute time"; cathode current within ~7.7 % on
average across all DI and EC cases); Marks and Gorodetsky 2025 (WarpX's Hall-thruster runs use
the Lewis gather on a staggered grid because it "helps stabilize PIC simulations which
under-resolve the Debye length and reduces the degree of numerical heating"); Angus et al.
2023, 2024 (how the axis r = 0 and the cylindrical volumes must be treated for exact energy
and charge conservation in (r,z) - an axisymmetric implementation exists, for dense plasmas).

**What changes in our code.** The gather uses the *same* shape function as the deposit but
reads E from a staggered position: E_r on radial edges, E_z on axial edges, each the finite
difference of phi across one edge (in our finite-volume language, exactly the edge quantity that
already carries the conductance C_e in `mesh.py` and the field energy `0.5 phi^T A phi` in
`poisson.py`). The nodal central-difference `electric_field_nodes` is replaced by an edge
field; the push reads E_r from the two radial edges of the cell weighted linearly in z and
E_z from the two axial edges weighted linearly in r. The Poisson operator, the fixed-point
deposit, Boris, MCC and every diagnostic stay. The axis form and the cylindrical volume
weights need the care of Angus et al. 2024. Effort: 2-3 days of kernels and tests (CPU/Warp
parity on a tiny box, the manufactured Poisson solution unchanged, the energy ledger's
field-work term re-derived from edge quantities so that the ledger stays exact).

**Expected speed-up.** EC-PIC alone does not change Delta t (omega_pe Delta t and Courant
still bind). It permits a coarser grid. At 50 um instead of 33 um the channel has 0.44x the
cells and rows (61 x 481 vs 91 x 721): block reads fall 0.3x (0.23 GB), sweeps 0.67x, and
Delta t can return to 1.5 ps (1.07x fewer steps). With N held (same W, same noise) the
particle term is unchanged: channel step 2.5-3.3 -> ~2.0-2.6 ms, 5.2 -> 4.9 M steps ->
**~1.3x**. With N reduced in proportion to the cells (fixed particles per cell, the choice of
Powis and Kaganovich 2024 and Marks and Gorodetsky 2025) the particle term also falls 0.44x
-> **~2x**, but the statistics band widens by 1.5x (section 1, point 2). For the plume, 50 um
is the v2.1 cost table's own row: 8.2 ms/step, 17-20 h to 3 transits -> **~2.5x** versus 48 h
at 33 um, with no new particle-count risk because W is unchanged.

**Physical risk to our claims.** Three, all documented:

1. *Sheath resolution.* Our per-cusp sheath drops and wall fluxes are read at sheaths that
   Brandt et al. 2016 report as 5-10 lambda_D wide; at the v4 peak (lambda_D ~ 15 um) that is
   75-150 um, i.e. 2-3 cells at 50 um and 1-2 cells at 100 um. Powis and Kaganovich 2024 lose
   accuracy exactly when Delta approaches the sheath gradient length; Hedlof et al. 2026 state
   the same condition for the semi-implicit variant ("adequate results, provided that the
   distribution function and the spatiotemporal scales dictating the physics of the problem are
   resolved"). EC-PIC removes the *instability*, not the *truncation error* in the sheath. A
   coarser grid is admissible for I_d, S, n_g* and I_beam only if the per-cusp sheath quantities
   are shown converged on the ladder, or are reported as unresolved.
2. *Drifting populations.* The 300 V beam, the 126-184 eV cusp ions and the injected 3 mA
   electron stream are drifting populations; Adams et al. 2025 find EC schemes heat for some
   drift / under-resolution combinations and Barnes and Chacon 2021 call the EC threshold
   "benign" only for ambipolar plasmas at realistic mass ratio. Our windowed residual-power gate
   (>= 5 % of electrode work, one-sided) is the right instrument and must stay armed.
3. *Momentum non-conservation / self-force.* EC-PIC accepts a self-force (Brackbill 2016;
   Powis and Kaganovich 2024, Appendix D). On a uniform grid it is small; on a graded grid it is
   the Duras et al. 2014 self-force that the Greifswald group had to correct. A momentum ledger
   (wall + electrode + beam momentum vs field force) should be added beside the energy ledger
   before any thrust number is read from an EC run.

**Verification protocol (the evidence the repository would need).** (i) EC at 33.3 um / 1.4 ps
/ W 2.67e4 / seed 20260903 must reproduce the accepted v4 explicit plateau within the seed band
(<= ~2-3 % on I_d, S, n_g*, I_beam; <= 5 % on peak n_e, T_e,peak) - a same-window, same-seed,
scheme-only comparison; (ii) the EC ladder 33 -> 50 -> 66 um must show the 50 um point within
10 % of the 33 um point on the ladder quantities *and* per-cusp sheath drop and wall flux within
their stated tolerance; (iii) the windowed residual-power gate and a new momentum ledger green
throughout; (iv) the v5 25 um explicit point (running on the H100) is the independent
convergence anchor - EC must agree with it, not replace it.

### 2.3 Semi-implicit energy-conserving PIC (ECsim; the Barnes modified-Poisson scheme)

**References.** Lapenta 2017 (ECsim: "conserves energy exactly to round-off for any time step
or grid spacing", "unconditionally stable in time", "eliminates the constraint of the finite
grid instability", the particle mover costs the same as explicit PIC and "only the field solver
has an increased computational cost"); Lapenta 2023 (implementation advances); Markidis,
Lapenta and Rizwan-uddin 2010 (iPIC3D, the implicit-moment ancestor); Mason 1981 and Brackbill
and Forslund 1982 (implicit-moment method); Barnes 2021 (appendix: the modified-Poisson
semi-implicit scheme, per Marks and Gorodetsky 2025's citation); Hedlof et al. 2026 (its
verification in WarpX: "stable at time steps larger than twice the inverse plasma frequency and
cell sizes larger than the Debye length"); Marks and Gorodetsky 2025 (the only published
application of a semi-implicit EC scheme to a thruster: the Charoy et al. 2019 axial-azimuthal
benchmark on one H100: explicit baseline 38 h-1.8 days; semi-implicit with 2x coarser grid and
2x Delta t **11.5 h, a 3.5x speed-up**; 4x -> 4 h; 8x -> 2 h; time-averaged deviations from the
benchmark < 10 % at 2x and < 25 % at 4x and 8x; the implicitness constant C was set to 8 or 16
"as these values yielded the smallest deviation in the time-averaged electron temperature from
the baseline"; "in the 2x case, the GPU needs to do eight times less work ... we only observed
a speedup of 3.5 times ... the benchmark simulation is relatively small for this GPU, and some
amount of its computational cost is purely overhead"); Parodi, Lapenta and Magin 2022 (ECsim
for electric-propulsion plasmas, VKI `Pantera`, conference paper); Cohen, Langdon and Friedman
1982; Langdon, Cohen and Friedman 1983; Friedman 1990; Cohen et al. 1989 (the direct-implicit
lineage that Barnes' scheme "resembles" and whose numerical heating at large Delta t Sun et al.
2023 measure to scale with Delta t / Delta x); Vay et al. 2012 (direct implicit in Warp).

**What changes in our code.** In the Barnes form the Poisson operator acquires an implicit
electron susceptibility: the node conductances are multiplied by (1 + C Delta t^2 omega_pe^2)
evaluated from the *current* electron density. Two consequences for `WarpBlockThomas`: the
operator changes every step, so the once-inverted Schur complements are invalid, and the
solver must become iterative (section 5 is a prerequisite, not an option); and the deposit must
also produce n_e on the operator's edges. The Lewis gather (section 2.2) is required for the
energy property. Delta t is then bounded by the electron Courant condition (v_e Delta t <
Delta) and by the gyration if Boris is kept (omega_ce Delta t <= 0.2 at 0.5 T is 2.3 ps; at
the 0.2 T cusp-wall fields 5.7 ps), not by omega_pe. Effort: 10-15 days on top of sections 2.2
and 5 (variable-coefficient operator, C as a declared protocol parameter, its calibration
series, tests).

**Expected speed-up for our runs.** Taking Marks and Gorodetsky 2025 as the only measured
precedent and our cost model: 2x coarser (66 um) and 2x Delta t (2.8 ps) -> channel 5 h -> ~1.5 h
(**3-3.5x**; the arithmetic gives 6-8x but Marks and Gorodetsky 2025 measured 3.5x on an
8x work reduction because the small problem is overhead-bound, and our step is more
latency-bound than theirs), plume 48 h -> ~10-14 h (**3.5-5x**; the plume particle term at fixed
W does not shrink); 4x/4x -> **~8-10x** (channel ~35 min, plume ~5 h). At 66 um the peak
Delta/lambda_D is ~4.4; at 133 um ~9.

**Physical risk to our claims.** The 66 um and 133 um grids put 1-2 and 0.5-1 cells across the
cusp sheaths: the per-cusp sheath drop and wall flux are unresolved by construction, and T_e
depends on a tuned C (Marks and Gorodetsky 2025 chose C to minimise the T_e deviation - a
calibration against the explicit answer, which a preregistered claim cannot do silently).
Marks and Gorodetsky 2025 report the bulk fields and the instability spectrum preserved at 2x
and degraded at 4x-8x. For us the bulk quantities (I_d, S, n_g*, I_beam, thrust) are the
candidates for a semi-implicit campaign; per-cusp quantities, T_e,peak and the IEDF tail are
not. Kahnfeld et al. 2019 and Brandt et al. 2016 (our closest analogue) never used an implicit
scheme; there is no cusp-thruster precedent.

**Verification protocol.** As 2.2 (i)-(iv), plus: a C-series (C in {4, 8, 16}) at fixed grid,
reported with the T_e and I_d spread as the scheme uncertainty; the explicit 33 um plateau
stays the reference, never the semi-implicit run; every semi-implicit result labelled with
(Delta, Delta t, C) and the unresolved-sheath disclosure.

### 2.4 Fully implicit energy- and charge-conserving PIC

**References.** Chen, Chacon and Barnes 2011 (the electrostatic scheme: Crank-Nicolson,
Jacobian-free Newton-Krylov on the field unknowns with "kinetic enslavement" of the particles,
exact energy and charge conservation, sub-stepped orbits); Markidis and Lapenta 2011 (the
companion energy-conserving implicit formulation published back-to-back with it, which notes
that the exact scheme "requires the solution of a very large matrix whose rank is of the order
of the number of particles" and therefore matrix-free Newton-Krylov solvers); Chacon, Chen and
Barnes 2013 (mapped
non-uniform meshes); Chen and Chacon 2013 (an analytical particle mover; the error estimator in
the Crank-Nicolson mover "has significant impact on the overall performance"); Chen et al. 2014
(fluid preconditioning of the Newton-Krylov solve, "largely insensitive to the electron-ion
mass ratio"); Taitano et al. 2013 (the fully implicit moment method); Chen, Chacon and Barnes
2012 (mixed-precision hybrid CPU-GPU implementation: JFNK in double on the CPU, the implicit
particle mover in single precision on the GPU, "about 100 times faster than an equivalent
single-core CPU"; roofline-guided); Chen and Chacon 2015, Chacon and Chen 2016 (multi-D and
curvilinear electromagnetic extensions); Chen and Chacon 2023 (an **asymptotic-preserving
push for arbitrarily magnetised plasmas in uniform B**: "timesteps much larger than particle
gyroperiods", "preserves all particle drifts", "orders of magnitude wall-clock-time speedups vs.
the standard fully implicit electrostatic PIC algorithm"); Ricketson and Chacon 2020 (the
energy-conserving asymptotic-preserving orbit integrator for arbitrary fields, with the
"new numerical time-scale" that restricts Delta t and an adaptive strategy that steps over the
gyration "when physically justified"); Eremin 2022 (the collisional bounded-plasma variant,
ECCOPIC; the GPU version ran the Villafana et al. 2021 benchmark); Mattei et al. 2017 (fully
implicit PIC-MCC for an ICP ion source, "cell sizes much larger than the Debye length and time
steps in excess of the CFL condition whilst preserving the conservation of the total energy");
Angus et al. 2022 (implicit PIC with binary Coulomb collisions and the energy bookkeeping
between them), 2023, 2024 (**an axisymmetric (r,z) implicit energy- and charge-conserving
code**, with "a new particle pusher for axisymmetric systems ... compatible with exact energy
and charge conservation" and the r = 0 treatment "described in detail"); Savard et al. 2025
(the cost side: when Delta x > lambda_D the implicit EC scheme needs *more* particles per cell
to reproduce converged solutions, non-uniform grids and collisions "exacerbate the errors", and
in 1D it was slower than explicit PIC once accuracy is required); Taccogna et al. 2023 (section
V C: the FGI in EC schemes "only if the average electron drift exceeds their thermal velocity
(note, however, that such a study was never conducted for magnetized plasmas)"; large time steps
"when the resolution of electron cyclotron rotation ... is not needed" require the
asymptotic-preserving integrators).

**What it would buy us.** Delta t is freed from omega_pe and, with an asymptotic-preserving
push, from omega_ce. Our Delta t = 1.4 ps resolves the 0.5 T gyration (71 ps period) 50x; an AP
push at Delta t ~ 20-50 ps would cut the step count 15-35x. Each step then costs a nonlinear
solve: typically 3-10 Newton-Krylov iterations, each re-pushing all particles with sub-stepping
(Chen et al. 2011, 2014), so the per-step cost is 5-20x the explicit step. Net: **2-5x** on the
step count side, before the ppc penalty of Savard et al. 2025.

**Physical risk to our claims.** The largest of any method here. (i) The asymptotic-preserving
push is derived for uniform B (Chen and Chacon 2023) or with an adaptive Delta t that returns to
resolving the gyration where the field varies (Ricketson and Chacon 2020); our physics is the
cusp, where B goes through a null on the axis and electrons de-magnetise - precisely where the
scheme must revert to small steps, so the gain is lost where the loss happens. (ii) Taccogna et
al. 2023 note the EC finite-grid stability study "was never conducted for magnetized plasmas".
(iii) Savard et al. 2025: more particles per cell for the same accuracy when Delta > lambda_D,
worse with collisions - our MCC is the ionisation source. (iv) No thruster, cusp or (r,z)
low-temperature-plasma result exists; Angus et al. 2024 is dense-plasma Z-pinch physics.

**Implementation cost.** A new code (JFNK with particle enslavement, preconditioner, AP push,
axis treatment): weeks to months, not days; Warp has no nonlinear solver, so the field solve
would be an external Krylov library or a hand-written GMRES with our operator as the matvec.
**Not recommended for this project's horizon**; recorded so the decision is traceable.

### 2.5 Charge-conserving and symplectic / structure-preserving variants

Esirkepov 2001 and Villasenor and Buneman 1992 solve a problem we do not have (charge
conservation between Ampere's law and the particle current in electromagnetic PIC); an
electrostatic code that solves Poisson from the deposited charge every step is charge-conserving
by construction. Kraus et al. 2017 (GEMPIC) and Squire, Qin and Tang 2012 (variational PIC)
preserve the symplectic structure of Vlasov-Maxwell; Qin et al. 2013 show the Boris push is
volume-preserving, which is why it holds energy over long runs already. None of these addresses
Delta/lambda_D or omega_pe Delta t; they are not speed-up routes here.

### 2.6 Verdict for family 1

The Lewis energy-conserving gather (2.2) is the cheapest change in this document that removes
the hard pi constraint, has recent low-temperature-plasma evidence in 1-D and 2-D (Powis and
Kaganovich 2024; Sun et al. 2023) and a thruster user (WarpX, Marks and Gorodetsky 2025), and
can be validated against the accepted explicit 33 um plateau by a same-seed scheme swap. The
semi-implicit Barnes / ECsim step (2.3) is the only measured route to >= 5x on a thruster
benchmark, needs the iterative solver of section 5 first, and moves the per-cusp sheath
quantities out of the claim set unless the ladder proves otherwise. Fully implicit PIC (2.4) is
a different project.

---

## 3. Method family 2: similarity and scaling tricks

### 3.1 Vacuum-permittivity scaling (epsilon -> gamma epsilon_0)

**References.** Szabo et al. 2014 and Szabo 2001 (thesis, see `pic-mcc-blockers.md` entry 92):
artificial permittivity together with an artificial mass ratio and an explicit retrieval
procedure, thrust within 5 %, current within 16 % of measurement; Adam, Heron and Laval 2004
(the first fully kinetic axial-azimuthal Hall-thruster PIC, the ancestor of the family in
which permittivity scaling became routine); Coche and Garrigues 2014 (2-D axial-azimuthal Hall
thruster with gamma = 80, the reference subsequent LAPLACE / LPP axial-azimuthal work cites for
the technique); Boeuf 2017 (tutorial); Fubiani et al. 2017 and
Fubiani, Garrigues and Boeuf 2018 (negative-ion sources: a density-reduction scaling with the
Child-Langmuir law, "the extracted beam current may be scaled to any value of the plasma density
... the scaling factor must be derived numerically"); Garrigues, Fubiani and Boeuf 2017
(critical assessment: "mesh convergence is reached only if the grid spacing is on the order of
or smaller than the minimum Debye length in the simulation domain, and ... strong aberrations
in the extracted beam are observed if this constraint is not respected"); Garrigues and Fubiani
2023 (tutorial, the scaling laws and their limits); Petronio et al. 2023 (the effect of the
scaled permittivity on the axial-azimuthal instabilities: "the artificial permittivity has an
impact on the growth rate", wavelengths of some azimuthal modes lengthen; the ion transit-time
instability's dispersion does not depend on epsilon_0); Charoy et al. 2019 and Villafana et al.
2021 (the benchmarks deliberately use no scaling); Taccogna et al. 2023 (lists "artificial
vacuum permittivity, mass ratio, size scaling" among the "numerical tricks" that hardware
advances now allow codes to avoid); **Yuan et al. 2020** (the closest precedent: a ring-cusp
ion-thruster discharge chamber, three acceleration techniques compared in one code -
"increasing the permittivity thickens the sheath. When the sheath expands enough to extend to
the cusps, the distributions of the potentials and the plasma densities are affected,
influencing the current parameters"); Liu W. et al. 2023 (a miniature ring-cusp ion thruster
PIC/MCC that adopts gamma = 10^2 on the strength of Yuan et al. 2020, so that lambda_D is "in
the millimeter range rather than the micron range" - in a centimetre-scale chamber).

**Arithmetic.** lambda_D and omega_pe^-1 both scale with sqrt(gamma), so Delta and Delta t can
grow by sqrt(gamma): cells fall by gamma (2-D), steps by sqrt(gamma), cost by gamma^(3/2) at
fixed particles per cell - **8x at gamma = 4, 32x at gamma = 10**. The Poisson block reads fall
by gamma^(3/2) as well. It is a one-line change (`epsilon_0` in `mesh.py` conductances).

**What it preserves and what it distorts, for us.** Preserved: the quasi-neutral bulk potential,
the E x B and grad-B drifts (B and E in the bulk are unchanged), the ionisation rate at fixed
n_g and T_e, the ion acceleration through the same potential, I_d and thrust to the 5-16 % that
Szabo et al. 2014 retrieved. Distorted: every sheath grows by sqrt(gamma). In a 2 mm bore whose
cusp sheaths (75-150 um at the v4 peak) are 4-8 % of the radius, gamma = 4 doubles them and
gamma = 10 triples them; Yuan et al. 2020's failure mode ("when the sheath expands enough to
extend to the cusps") is our normal operating condition, because the cusp *is* where our wall
flux lives (Matyash et al. 2010 in `pic-mcc-blockers.md`). The per-cusp sheath drop, the
per-cusp wall ion energy (which is the sheath drop plus the pre-sheath), the leak width relative
to the sheath, and the peak n_e (set by the Debye-scale sheath at the wall) are changed
quantities, not retrievable ones. Plasma-frequency-scale physics (Petronio et al. 2023) is not
our claim in an axisymmetric code, but the omega_pe-scale sheath oscillations that set the
cusp electron loss are. Liu W. et al. 2023's gamma = 100 in a centimetre chamber is not a
precedent for a millimetre bore: their lambda_D at gamma = 100 is a millimetre against a 30 mm
chamber; ours would be 150 um against a 2 mm bore.

**Verdict.** A development and screening tool with a **declared gamma-series retrieval** (gamma
in {1, 2, 4} on the same seed and window, extrapolated to gamma = 1, and the extrapolation
error reported), never a campaign claim (this repeats `pic-mcc-blockers.md` section 1(d)4 and
`LITERATURE_SYNTHESIS.md` P1d). For the design mini-sweep's *ranking* question (four designs,
same closure) a gamma = 4 screen is defensible if the ranking is shown stable between gamma = 4
and gamma = 2; for any sheath, cusp-flux or peak-density number it is not.

### 3.2 Ion-mass scaling

**References.** Szabo et al. 2014 (artificial mass ratio with retrieval); Cho, Komurasaki and
Arakawa 2013 and Cho et al. 2015 (mass-ratio manipulation with a semi-implicit field solver and
a mobility-recovery model, an explicit refusal of permittivity or geometry scaling "to avoid
unrecoverable change of physics"); Taccogna et al. 2005 (two scalings to speed up a fully
kinetic Hall-thruster PIC, one of them the heavy-particle time scale); Yuan et al. 2020
("reducing the masses of heavy particles greatly influences ion properties, especially the
plasma density. Thus, it causes significant errors in the potential and current parameters.
Errors in the beam current can be significantly decreased by correcting the beam current using
an exponent relationship between the mass scaling factor and the plasma density").

**Arithmetic and risk.** Ion transit time falls as sqrt(m'/m), so the steps to 3 transits fall
by the same factor (m/100 -> 10x fewer steps). But n_i = j/(e v_i) falls by sqrt(m'/m) at fixed
current, the Bohm speed and every sheath change, the ionisation-to-transit ratio nu_iz tau_i
that decides avalanche vs ignition (the repository's phase-3 finding) changes, and the
quasi-steady neutral fixed point n_g* = (Q_in - S + R_wall)/c is a function of the ion residence
time. In our closure, mass scaling moves the operating point itself. Yuan et al. 2020's beam
correction is an empirical exponent fitted per device. **Not recommended**, even for screening,
because the quantities it distorts (n_e, utilisation, n_g*) are the mini-sweep's closure targets.

### 3.3 Geometric self-similarity

**References.** Lacina 1971 (similarity rules); Taccogna et al. 2005, 2007, 2008 (geometric
scaling for Hall-thruster PIC: shrink the device by zeta, raise B and n by zeta so that L/r_L and
L/lambda_mfp are held; "particular emphasis has been spent for the geometrical scaling");
Brandt et al. 2016 (factor 4 for the down-scaled HEMPT, with the caveat that it "fails when
surface processes matter" because the surface-to-volume ratio does not scale); Matthias et al.
2020 (the limits derived for HEMP-T modelling); Kahnfeld et al. 2019 (the Greifswald review,
closed access; the group's justification is in Matthias et al. 2020); Taccogna and Minelli 2018
(geometric scaling "irremediably changes ... the wall interaction and the axial component of
the electric field").

**Why it does not apply to us.** With n -> zeta n, lambda_D -> lambda_D / sqrt(zeta) while L ->
L / zeta, so lambda_D / L grows by sqrt(zeta): the *relative* sheath thickens exactly as under
permittivity scaling, and the wall-collision-to-volume-process ratio changes (Brandt et al.
2016). Our device is already the small end of the family (2 mm bore, 0.19 sccm); Brandt et al.
2016 scaled a 14 mm x 1.5 mm channel *down* by 4 to reach their grid, and the repository has
already recorded that their omega_pe Delta t = 0.2 in the scaled frame is 0.56 in ours. There
is nothing left to scale, and the objects we claim (cusp wall fluxes, sheath drops) are the
surface processes the scaling does not preserve. **Not applicable.**

### 3.4 Reduced geometry and reduced-order PIC

The code is already (r,z) axisymmetric, which excludes the azimuthal drift instability by
construction (`pic-mcc-blockers.md` section 4.3; Boeuf and Garrigues 2018 and Charoy et al.
2019 are the axial-azimuthal studies that resolve it). The reduced-order / pseudo-2-D schemes of
Reza, Faraji and Knoll 2023 and Faraji, Reza and Knoll 2022 (2-15 % of full-2-D cost, 2-4 %
error on time-averaged fields) decompose the *azimuthal-axial* or *radial-azimuthal* planes into
coupled 1-D regions; they are built for instability problems and have no (r,z) cusp-geometry
form. The channel-only box (3 x 24 mm) versus the plume box (12 x 48 mm) is the reduced geometry
we already use, at the price recorded in `pic-mcc-blockers.md` section 4.4 (the Dirichlet exit
plane). No further reduction is available without a hybrid or two-domain scheme.

---

## 4. Method family 3: time acceleration to the steady state

### 4.1 Sub-cycling and multi-rate stepping

Adam, Gourdin Serveniere and Langdon 1982 introduced electron sub-cycling for electromagnetic
PIC (the field is advanced on the slow scale, electrons on the fast); the mirror image, ion
sub-cycling, is the standard for electrostatic codes whose Delta t is set by omega_pe (Birdsall
and Langdon 1991, chapter 4). The code already pushes ions every k = 8 steps, so the ion push
is already ~1/8 of an electron push per step; the remaining gain is inside the ion share (a
few per cent of the step). Electron sub-cycling is meaningless in an electrostatic code (there is
no field time scale slower than omega_pe to sub-cycle against). Multi-rate stepping of *regions*
(a coarser Delta t in the plume than in the bore) is a two-domain scheme (the v2.1 spec's
`two_domain_approach`: "weeks, not days"). **Nothing left to take here.**

### 4.2 Neutral time-scale separation (already the largest acceleration we use)

The physical neutral transit (24 mm / 220 m s^-1 ~ 110 us) is 30-45 ion transits; Brandt et al.
2016 held neutrals static for 76 us of plasma time, and the repository's quasi-steady inventory
with wall recycling replaces the neutral transport by its fixed point. The breathing-mode
literature (Kahnfeld et al. 2018 for HEMP; Petronio et al. 2024 for the neutral-transport
mechanism) is the price: a plateau under the quasi-steady closure cannot tell whether the
physical discharge oscillates (`pic-mcc-blockers.md` section 3). This is the one place where
the code is already **~15-20x faster than the physical time scale demands** (a 5-7 us run
against the >= 100 us a neutral-transport steady state would need), and the claim boundary
already says so.

### 4.3 Restarting from converged coarse solutions (grid sequencing)

**References.** None in PIC. The multigrid literature's full-approximation-scheme / nested
iteration (Brandt 1977; Briggs, Henson and McCormick 2000) is the elliptic analogue; the
parareal parallel-in-time work on plasma turbulence (Samaddar, Newman and Sanchez 2010) is the
only plasma precedent for accelerating a time march and is not a PIC method. Marks and
Gorodetsky 2025 report the closest engineering practice: particle resampling "was thus only
triggered during the startup transient" and cut wall time by up to 25 %, but "thresholds lower
than the steady-state particle count adversely affected the accuracy", the electron
temperature at a boundary moving by 7 eV.

**What it could buy us.** A PIC plateau is (under a unique fixed point of the closure)
independent of the initial condition; a 33 um run initialised from the interpolated 50 um
plateau (phi, particle phase space re-sampled per cell from the coarse-run moments or copied
with the same W) skips part of the ~1-transit ignition transient. The plateau criterion still
needs >= 2 transits of trailing window after the restart, so the saving is <= 1 of 3 transits:
**<= 1.5x**, and only for ladder points above an existing plateau. It costs 1-2 days (checkpoint
re-interpolation, `restarted_from` provenance in `run_state`, the same-seed replay test against
a from-scratch run) and preserves every claim *if* the restarted run is shown to reach the same
plateau as a from-scratch run once (the ss-v4 record is the from-scratch reference). It is also
the only safe way to add a rung (25 um -> 20 um) to the ladder at bounded cost.

### 4.4 Particle merging / splitting to cap N

**References.** Lapenta and Brackbill 1994 (dynamic and selective control of the particle
number); Assous, Dulimbert and Segre 2003 (coalescence); Welch et al. 2007 (adaptive particle
management); Teunissen and Ebert 2014 (k-d-tree weight control, conserving the moments of the
distribution); Vranic et al. 2015 (merging that conserves charge, energy and momentum exactly
per merge); Pfeiffer et al. 2015 (two statistical split/merge methods); Luu, Tuckmantel and
Pukhov 2016 (Voronoi merging); Muraviev et al. 2021 (thinning vs merging strategies; the
leveling-thinning algorithm WarpX defaults to); Faghihi et al. 2020 (moment-preserving
constrained resampling); Marks and Gorodetsky 2025 (the thruster application above: -25 % wall
time, boundary T_e biased when the threshold sits below the steady-state ppc).

**For us.** N grows from 1 M to 4.6 M over the plume attempts and to ~10 M at 33 um in the
48 mm box, with 21 % of the ions in the plume. Capping N at ~5 M by merging in the plume
(where the far-field density is 1-3 % of the exit value and the shot-noise gate already needs
>= 64,000 particle-steps per node) removes up to ~40 % of the plume step's particle term:
**~1.2-1.4x on the plume, ~1x on the channel** (the channel's N is set by the bore, where we do
not want to merge). Risk: velocity-space diffusion (Turner 2006's numerical thermalisation
applies with a vengeance to merged particles), IEDF and EEDF tails, and the energy ledger
(Vranic et al. 2015's exact conservation per merge is the method to use; thinning is not).
Effort 3-4 days. Verification: merge-on vs merge-off on the channel plateau within the seed
band; ledger residual power unchanged; far-field IEDF moments within 5 %.

### 4.5 Fluid-kinetic hybrid electrons and steady-state seeking

The user has dropped the hybrid route (branch `feat/hybrid-l2-v2` parked: I_d 7.52 vs 3.44 mA,
no speed advantage at PIC/L2 wall-clock 1.66); the literature record is Parra et al. 2006 and
Petronio et al. 2024 for the hybrid form and Hara 2019 for the model-choice review. No published
PIC "steady-state seeking" algorithm (a Newton or relaxation step on the time-averaged moments)
exists in the verified set; Kahnfeld et al. 2018 and Brandt et al. 2016 reach their averages by
marching. **Silent.**

---

## 5. Method family 4: the Poisson solver

### 5.1 What we have and why it is bandwidth- and latency-bound

`WarpBlockThomas` groups unknowns by radial row (m = nz+1 per block), inverts the Schur
complements once on the host, and solves by 2 (nr+1) dependent dense row-block matvecs
captured in one CUDA graph. It is exact, deterministic, shared with the CPU path and was faster
than Jacobi-PCG at these sizes (the repository's lesson: Jacobi-PCG needed O(nz) iterations with
reductions). Its cost is (nr+1) x 2 x m^2 x 8 B of reads per solve - 0.76 GB on the channel,
12 GB on the 48 mm plume at 33 um - and 2 (nr+1) sequential launches. Kahnfeld et al. 2016
compared a direct LU decomposition with successive over-relaxation for the Greifswald HEMP-DM3a
code ("results and runtime of solvers were compared"; the abstract does not say which was
adopted); Duras et al. 2017 parallelised the same code for plume domains. Our solve is the
direct family: exact, memory-heavy, sequential in one direction.

### 5.2 Geometric multigrid on the masked cylindrical finite-volume grid

**References.** Brandt 1977 (multi-level adaptive solutions); Briggs, Henson and McCormick 2000
(tutorial); Johansen and Colella 1998 and Gibou et al. 2002 (Cartesian-grid embedded-boundary
Poisson on irregular domains, second-order symmetric discretisations - the cut-cell route if the
body were not grid-aligned; ours is, so the masked operator coarsens by Galerkin restriction
without cut cells); Zhang et al. 2019 (AMReX, whose MLMG is WarpX's electrostatic solver) and
Myers et al. 2021 (its GPU port); Marks and Gorodetsky 2025 (MLMG tolerance 1e-3 vs 1e-5 changed
the wall time materially; "the iterative algorithm used to solve the fields only needed to
perform three iterations" at the loose tolerance; the multi-GPU scaling of the solver was the
weak point, single-GPU was not).

**Cost estimate for us.** N_nodes = 6.6e4 (channel) / 5.2e5 (plume). A V-cycle touches ~10
arrays per level over a geometric series of levels (4/3 N nodes in 2-D): ~10 x 8 B x 4/3 N ~
7 MB (channel) / 55 MB (plume) per cycle; 8-12 cycles to the 1e-10 relative residual our
contract demands (fewer if the previous step's phi is the initial guess, which the direct solve
cannot exploit and MG can) -> ~70 MB / ~550 MB per solve against 0.76 / 12 GB today; **~0.3-0.6
ms per solve on the plume against ~11 ms**, ~0.2-0.3 ms on the channel against ~1.4 ms
(latency-bound, not bandwidth-bound, at these sizes). Launches: ~6 levels x (smooth, residual,
restrict, prolong, correct) x 2 x 10 cycles ~ 300-600 launches - more than the 182-722 of the
sweeps on the channel, similar on the plume; inside the existing CUDA graph each costs ~1-2 us
on Linux (NVIDIA CUDA Graphs documentation), so ~0.5-1 ms of latency, which is why the channel
gain is smaller than the plume gain. The axis needs the r-weighted operator (already in
`mesh.py`'s conductances) and a coarsening that keeps r = 0 a node; the Dirichlet electrodes
and the far plane coarsen as masked nodes.

**Expected speed-up.** Plume-33: 18.7 -> ~8-9 ms/step -> **~2.2x** (48 -> ~22 h), plus 6 GB of
device memory and the host factorisation (hours on Windows) removed, plus non-uniform and
variable-coefficient operators (sections 2.3, 7.4) become possible. Channel-33: 2.5-3.3 ->
1.8-2.4 ms -> **~1.3-1.4x**.

**Physical risk.** None, if the solve meets the same residual contract: the equations are
unchanged and the repository's `verify()` true-residual check applies unchanged. The bitwise
CPU-vs-GPU replay would become an allclose replay at the residual tolerance (the repository
already runs `numerical` mode for cross-platform anchors).

**Effort.** 4-6 days in Warp (restriction/prolongation kernels on the masked grid, red-black
Gauss-Seidel or damped Jacobi smoother, a coarsest-level dense solve reusing the existing
block-Thomas at low resolution, tests: manufactured solution order 2, same-seed replay against
block-Thomas to rtol 1e-10 on a short run, then the ss-v4 plateau reproduced).

### 5.3 PCG with an algebraic-multigrid preconditioner (AmgX, PyAMGX)

Stuben 2001 (AMG review); Bell, Dalton and Olson 2012 (fine-grained parallel AMG on GPUs);
Naumov et al. 2015 (AmgX: GPU AMG and preconditioned Krylov); Bell and Garland 2009 (sparse
matrix-vector products on throughput hardware). For a fixed operator the AMG setup is done
once; each solve is ~10-20 SpMV + smoother passes over a 5-point matrix (~5 nnz per row, 20-50
MB per pass on the plume) - the same order as geometric MG, with a mature library instead of
hand-written kernels, but a C++ dependency beside Warp (device-pointer interop through the CUDA
array interface), non-deterministic reductions unless configured, and no reuse of the masked
FV structure. Same speed-up as 5.2; lower implementation risk, higher integration risk.
Preferable only if the geometric MG stalls on the axis or the L-shaped mask.

### 5.4 Fourier-in-z plus tridiagonal-in-r; cyclic reduction; capacitance-matrix methods

Hockney 1965 and Swarztrauber 1977 (FFT + tridiagonal; FACR) need a separable operator on a
rectangle; our L-shaped domain with a Dirichlet body, a Dirichlet anode column, a dielectric
wall carrying surface charge and a Dirichlet far plane is not separable, so a direct FFT solve is
**not applicable**. Two classical repairs exist: (i) **capacitance-matrix / matrix-imbedding**
(Buzbee, Dorr, George and Golub 1971; Proskurowski and Widlund 1976): embed the irregular domain
in a rectangle, solve fast there, and correct through a dense "capacitance" system whose size is
the number of irregular-boundary nodes - for us the ~180 body-face nodes plus the ~720 channel
wall nodes plus the electrode columns, a ~1,000-2,000 dense system that is a single small
matvec per solve once factorised. That is applicable and would make the field solve a few FFTs
of length 1441 x 361 plus 361 tridiagonal solves plus one 2,000^2 matvec (~30 MB): sub-ms.
Effort is higher than MG (5-8 days) and the method is unfamiliar to reviewers; MG reaches the
same floor. (ii) **Block cyclic reduction** (Buzbee, Golub and Nielson 1970; on GPUs Zhang,
Cohen and Owens 2010 for the scalar tridiagonal case) replaces the 2 (nr+1) sequential
row-block steps by ~2 log2(nr) levels - it cuts the launch count 20-40x but still reads dense
row-block inverses, so it removes the latency term (0.9 / 3.6 ms) and not the bandwidth term
(0.5 / 7.5 ms). A useful intermediate if MG proves hard, worth 1.3x on the channel and 1.2x on
the plume alone.

### 5.5 Verdict for family 4

Geometric multigrid on the existing masked FV operator is the single change with **no physics
risk and the largest plume gain (2.2x)**; it is also the prerequisite for the semi-implicit
operator of 2.3 and for any graded grid. It should be built and validated first.

---

## 6. Method family 5: kernel-level engineering

### 6.1 Launch count and kernel fusion

The step issues ~480 launches inside one CUDA graph; 182-722 of them are the Thomas sweeps
(section 5 removes them), the rest are gather/push/boundary/MCC/deposit per species, node
kernels, gate moments and reductions. Graph capture (already done, the repository measured 7.8
-> 1.5 ms/step at 9 k particles on WDDM) removes the host submission cost, not the per-kernel
launch and drain latency inside the GPU (of order 1-2 us per dependent kernel on Linux, NVIDIA
CUDA Graphs documentation). Filipovic et al. 2015 quantify kernel fusion for memory-bound BLAS
chains; Juhasz et al. 2021 (1-D PIC-MCC on GPUs, roofline-guided, 2.6 TFlop/s sustained, up to
200x over a single CPU core at 10 M particles) and Mertmann et al. 2011 (fine-sorted 1-D
PIC-MCC on CUDA) both fuse gather, push, boundary and collision into one particle kernel; WarpX
"fuses [the push] with the field-gathering step above into a single kernel" (Marks and
Gorodetsky 2025). Our gather, Boris push, wall/electrode classification, MCC null-collision test
(Vahedi and Surendra 1995) and deposit are separable kernels today; fusing per species to one
kernel plus one deposit
would bring ~480 launches to ~40-60 after the solver change. **Expected: 1.2-1.4x on the
channel step (latency-bound), 1.1-1.2x on the plume.** Effort 3-5 days. Physical risk none; the
fused step must replay the un-fused step bitwise on the same seed (the fixed-point deposit
makes this testable - Warp's `wp.rint` fixed-point accumulation is order-independent), which is
the repository's existing regression instrument.

### 6.2 Particle sorting / binning for locality

Bowers 2001 (hybrid counting sort, the classic); Stantchev, Dorland and Gumerov 2008 (fast
particle-to-grid interpolation on the GPU by binning); Decyk and Singh 2011, 2014 (the
"adaptable" PIC algorithms for GPUs: particles kept sorted by tile so deposition runs in shared
memory without global atomics); Mertmann et al. 2011 (fine-sorting at the field-cell scale);
Bowers et al. 2008 (VPIC's cell-sorted layout); Marks and Gorodetsky 2025 (sorting every
100-500 steps, overhead ~80 % of one push; "more frequent particle sorting typically results in
better simulation performance, with a maximum speed-up of one hour" of a 38 h run, i.e. ~3 %).
Our deposit uses global int64 atomics on 4 nodes per particle; contention at the dense bore
nodes (1e5-1e6 particle-steps per node per window) is real, and gather reads of phi/E are
random-access without sorting. The published gains for *deposition-dominated* codes are
2-5x on the deposit kernel; for the whole step Marks and Gorodetsky 2025 found ~3 %. **Expected
1.1-1.3x on the particle share** with a periodic (every 100-500 steps) counting sort by cell;
effort 2-3 days (the sort must permute five float64 arrays and the alive flags and keep the
RNG-stream identity per slot, which breaks bitwise replay against the unsorted run - the test
becomes allclose on the moments). Physical risk none.

### 6.3 Structure-of-arrays and memory layout

`DeviceSpecies` already holds r, z, v_r, v_theta, v_z and the alive flags as separate device
arrays (structure of arrays); the node fields are separate arrays. Nothing to gain here except
cell-relative positions (a cell index plus a float32 offset, the PIConGPU layout - Burau et al.
2010; Bussmann et al. 2013), which is the precision enabler of 6.4, not a layout gain by itself.

### 6.4 Mixed precision

**References.** Chen, Chacon and Barnes 2012 (single-precision GPU particle mover under a
double-precision field solve, roofline-guided, "without apparent loss of robustness or accuracy
in a challenging long-timescale ion acoustic wave simulation"); Burau et al. 2010 and Bussmann
et al. 2013 (PIConGPU in single precision throughout); Derouillat et al. 2018 (Smilei, double);
Taccogna et al. 2023 ("GPUs offer numerous opportunities of utilizing additional computation
units performing computations with a reduced precision, which can be combined with more
accurate calculations in critical parts"); Williams, Waterman and Patterson 2009 (the roofline
model that decides whether precision matters: only for bandwidth- or FLOP-bound kernels).

**For us.** The particle kernels are latency- and atomic-bound (section 1, point 3), not FLOP
bound, so the consumer-GPU FP64 penalty (the RTX 5090's FP64 pipeline is a small fraction of
its FP32 rate; the H100's is one half) is not what we pay for today - the H100 benchmark's
"~1x a 5090" confirms it. Halving the bytes per particle (float32 positions and velocities,
int64 deposit and float64 solve kept) helps only the bandwidth part of the particle term:
**1.1-1.3x on the particle share** at most. The precision risk is real and specific: a float32
absolute position over a 48 mm domain has 4 nm resolution (fine), but the energy ledger that the
heating gate reads accumulates over 5e6 steps; Chen et al. 2012 show it can be held, and the
ledger and the residual-power gate would tell us. Cell-relative coordinates (6.3) remove the
position issue entirely. The CPU-vs-GPU bitwise parity the repository relies on becomes a
tolerance-based parity. Effort 2-3 days plus the parity-test rewrite. **Low priority** until
sections 5 and 6.1 have made the step bandwidth-bound.

### 6.5 MPS, concurrency and multi-GPU domain decomposition

MPS is already measured (1.54x aggregate at 4 slots, no single run faster). Multi-GPU domain
decomposition for **one** run is argued against by three measurements: (i) the H100 vs 5090
result shows the step is latency-bound at 45 k-520 k cells - splitting it across GPUs lowers
the per-GPU work below saturation and adds a halo exchange per kernel; (ii) Marks and
Gorodetsky 2025 found the electrostatic solver's strong scaling poor even at 1024 x 512 x 512
cells per GPU because of the multigrid's MPI traffic, while "the other parts of the code scaled
well"; (iii) the patch-based codes that scale (PSC, Germaschewski et al. 2016; OSIRIS, Fonseca et
al. 2013; PIConGPU, Bussmann et al. 2013; WarpX, Fedeli et al. 2022) do so for problems of
1e8-1e11 particles. Our whole plume run is one GPU's worth of work; the cloud's value is
parallel slots (seeds, W variants, designs), exactly as the H100 lesson recorded.

### 6.6 What the production codes do for electrostatic small-domain problems

PIConGPU (Burau et al. 2010; Bussmann et al. 2013): supercell-sorted particle frames,
shared-memory deposition without global atomics, float32. WarpX (Vay et al. 2018; Myers et al.
2021; Marks and Gorodetsky 2025): AMReX tiles, periodic sorting, fused gather-push, MLMG Poisson
with a loosened tolerance, the Lewis EC gather as an option, Barnes' semi-implicit Poisson as an
option, leveling-thinning resampling. Smilei (Derouillat et al. 2018): patch-based domain
decomposition with per-patch sorting and vectorised deposition. VSim / VORPAL (Nieter and Cary
2004): cut-cell boundaries for the field solve. The common thread for small electrostatic
domains is not the particle kernel - all of them are within a factor of two of each other per
particle-step - but (a) an iterative multigrid field solve, (b) sorting, (c) fused particle
kernels and (d) the willingness to coarsen the grid through an EC or semi-implicit scheme, which
is where Marks and Gorodetsky 2025's 3.5-10x came from.

---

## 7. Method family 6: variance reduction and fewer particles

### 7.1 Why fewer particles is not free

Hockney 1971 measured collision and heating times in a 2-D thermal computer plasma: the
noise-driven heating time scales with the number of particles per Debye cell; Okuda and
Birdsall 1970 give the finite-size-particle collision theory behind it; Birdsall and Langdon
1991 (chapter 12) and Dawson 1983 the textbook form; Turner 2006 shows Monte Carlo collisions
accelerate numerical thermalisation by up to three orders of magnitude (our MCC is the
ionisation source, so this applies); Turner et al. 2013 and Turner 2016 make particles-per-cell
convergence a verification requirement; Nevins et al. 2005 show discrete-particle noise
masquerading as physics in a well-known gyrokinetic dispute. Adams et al. 2025 add that the
finite-grid growth rates are seeded by the noise spectrum. Our own band (5-12 % between seed-b
and W x 0.7) and the entanglement of W and grid effects in the ss-v4 verdict are the local
evidence. Any reduction of N must be bought with a variance-reduction method, and the
statistics band must be re-measured, not assumed.

### 7.2 delta-f methods

Parker and Lee 1993; Aydemir 1994; Hu and Krommes 1994; Denton and Kotschenreuther 1995: the
particles carry the deviation from a known equilibrium f_0, and the noise scales with |delta f|
instead of |f|. It requires an f_0 that is close to f and analytically known. A bounded,
ionising, wall-loss discharge with sheaths, a 300 V beam and a 3 mA injected stream has no such
f_0; the method is used in gyrokinetics and beam physics, never (in the verified set) in a
low-temperature discharge. **Not applicable.**

### 7.3 Quiet starts

Sydora 1999 (low-noise initialisations for electromagnetic and relativistic PIC); Birdsall and
Langdon 1991 (chapter 16). A quiet start lowers the noise of the *initial* state; our seed
plasma is replaced within a fraction of an ion transit by the discharge's own particles, so the
gain is confined to the ignition transient. **Negligible.**

### 7.4 Smoothing and digital filtering

Birdsall and Langdon 1991 (Appendix C: the binomial 1-2-1 filter and its compensator; applying
the same filter to the gathered field keeps the momentum-conserving property); Vay et al. 2011
(binomial digital filters and their multi-pass variants as an instability-mitigation tool);
Langdon 1970 and Birdsall and Maron 1980 (the aliases at k ~ pi/Delta that the filter
suppresses are the ones that drive the finite-grid instability). A 1-2-1 filter on rho (and the
same on E) widens the effective particle shape, which lowers the heating rate at a given
Delta/lambda_D and therefore lets the gate move from pi toward ~4-5 (the quadratic-spline
column of Adams et al. 2025 is the closest quantitative guide, and it still heats for some
drifts). It costs one node kernel and 1-2 days. It also smears any 2-3-cell sheath by one more
cell and changes the discrete Gauss law used by the ledger (the filtered rho must be the one the
ledger integrates). **Expected <= 1.5-2x in cells at fixed W**, with the same sheath-resolution
caveat as section 2.2 and in competition with it (choose EC or filtering, not both, or the
ladder cannot attribute the change).

### 7.5 Importance sampling and variable weights

Welch et al. 2007; Lapenta and Brackbill 1994; Teunissen and Ebert 2014: spatially variable W
(low in the far plume where the shot-noise gate is starved, high in the dense bore) with
splitting at the exit plane and merging on return. Our N is dominated by the bore, where we
want the statistics we have; the far plume holds ~4 % of the ions beyond 33 mm. Splitting there
*raises* N a little and fixes the gate's resolved-node problem; it does not reduce cost.
**~1x on cost, a statistics gain in the plume.**

### 7.6 Sparse grids

Ricketson and Cerfon 2017 (the sparse-grid combination technique for PIC: the grid-based
statistical error falls because each component grid has far fewer cells for the same
particles); Deluzet et al. 2022 (convergence analysis); Muralikrishnan et al. 2021 (adaptive
noise reduction); Garrigues et al. 2021 I and II (CCP and the Hall-thruster drift instability);
Garrigues et al. 2024 (the "offset" scheme, **with multi-cusp magnetic-field configurations as
a test case**: it "reduces the error of the current collected at the walls to less than 5 %",
3-5x speed-up at 256^2-512^2); Taccogna et al. 2023 (section V C a). This is the one variance
method with a cusp precedent. It needs a hierarchy of anisotropic component grids over the
masked L-shaped domain with Dirichlet bodies, a combination step for phi (Garrigues et al. 2024:
combine phi at regular nodes, not E), and a Poisson solve on every component grid (cheap ones).
Effort is weeks (L), the axis and the mask are unproven, and the noise reduction competes with
the EC route for the same "fewer particles per cell" budget. **DEFER**, with Garrigues et al.
2024 as the trigger paper if a future rung needs 4x more particles.

### 7.7 The statistics band and what it means for N

At the accepted plateau the repository's band is 5-12 % on the ladder quantities between the
seed pair and the W x 0.7 pair; the ss-v4 acceptance tolerance is 10 %. A method that halves N
without variance reduction moves the band to 7-17 % and makes the 10 % criterion undecidable.
Every recommendation in section 8 therefore holds W (and N per unit volume) fixed unless the
method itself reduces the variance (sparse grids) or the claim set is reduced to the bulk
quantities whose band is at the low end (I_d, S: ~1-6 %).

---

## 8. Ranked recommendation table

Speed-ups are for the two runs of section 0.1 (channel-33: 5 h recorded, 6 h budgeted; plume
48 x 12 mm at 33 um: ~48 h) at fixed W unless stated; "preserved" means the equations solved are
unchanged and the result must match the current code to a stated tolerance; "re-validate"
means the equations change and the claim survives only if the named protocol passes; "at risk"
means the literature says the quantity is distorted. Effort is implementation plus tests before
GPU time (S < 1 d, M 1-3 d, L > 3 d, XL weeks).

| rank | method (section) | channel-33 | plume-33 | claims preserved | claims to re-validate | claims at risk | effort | verification protocol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Warp-native geometric multigrid replacing the block-inverse solve (5.2) | 1.3-1.4x (5 h -> 3.6-3.8 h) | 2.2x (48 -> ~22 h); -6 GB device, no host factorisation | all (same equations, same residual contract) | none | none | L (4-6 d) | manufactured solution order 2; same-seed short-run replay vs block-Thomas at rtol 1e-10; ss-v4 plateau reproduced within the seed band; `verify()` true residual every sync |
| 2 | kernel fusion + launch reduction (~480 -> ~50) + periodic cell sort (6.1, 6.2) | 1.2-1.4x | 1.1-1.3x | all | none | none | L (3-5 d) + M (2-3 d) | bitwise same-seed replay for the fusion (fixed-point deposit); allclose on moments for the sort; ms/step before/after recorded in the launch log |
| 3 | explicit energy-conserving (Lewis) gather + edge fields, then coarsen 33 -> 50 um (2.2) | 1.3x at fixed N; ~2x at fixed ppc | ~2.5x (48 -> 17-20 h, the v2.1 50 um row) | I_d, S, n_g*, I_beam, thrust (after protocol) | peak n_e, T_e,peak, per-cusp wall flux and energy, IEDF | per-cusp sheath drop at Delta >= 2-3 lambda_D (1-3 cells across the sheath); momentum (self-force) for thrust | L (2-3 d) + momentum ledger (1 d) | EC at 33 um / same seed reproduces the accepted explicit v4 plateau within the seed band; EC ladder 33 -> 50 -> 66 um with the 50 um point within 10 % on ladder quantities and per-cusp sheath drop / wall flux within tolerance; residual-power gate and momentum ledger green; the explicit 25 um (v5) point stays the convergence anchor |
| 4 | coarse-to-fine restart (grid sequencing) for ladder rungs (4.3) | <= 1.5x on new rungs (skips <= 1 of 3 transits) | <= 1.5x | all, if the plateau criterion still requires >= 2 trailing transits | none | none | M (1-2 d) | restarted 33 um run vs the from-scratch ss-v4 plateau within the seed band, once; `restarted_from` provenance |
| 5 | Barnes / ECsim semi-implicit Poisson with 2x coarser grid and 2x Delta t (2.3); needs 1 and 3 | 3-3.5x (5 h -> ~1.5 h); 4x/4x: ~8-10x | 3.5-5x (48 -> 10-14 h); 4x/4x: ~5 h | I_d, S, n_g*, I_beam (bulk; Marks and Gorodetsky 2025 < 10 % at 2x) | T_e (depends on C), thrust, IEDF | per-cusp sheath drop and wall flux (0.5-2 cells across the sheath); any omega_pe-scale sheath dynamics; T_e at 4x-8x (< 25 %) | XL (10-15 d after 1 and 3) | C-series {4, 8, 16} at fixed grid reported as scheme uncertainty; 2x and 4x points against the explicit 33 um plateau within 10 % on bulk quantities; every result labelled (Delta, Delta t, C) with the unresolved-sheath disclosure; explicit reference never replaced |
| 6 | particle merging (Vranic 2015) capping N in the plume (4.4) | ~1x | 1.2-1.4x | I_d, S, n_g* | I_beam, thrust, IEDF/EEDF tails, far-field T_e | none if exact-conservation merging; tails if thinning | L (3-4 d) | merge-on vs merge-off on the channel plateau within the seed band; ledger residual power unchanged; far-field IEDF moments within 5 % |
| 7 | mixed precision: float32 push/gather, int64 deposit, float64 solve (6.4) | 1.1-1.2x | 1.1-1.3x | all, if the ledger says so | energy ledger residual (5e6-step accumulation) | none documented (Chen et al. 2012) | M (2-3 d) + parity-test rewrite | residual-power gate unchanged within 1 % of electrode work; plateau within the seed band; CPU/GPU parity at a declared tolerance |
| 8 | binomial digital filtering of rho and E (7.4) | <= 1.5-2x in cells if the gate moves pi -> ~4-5 | same | I_d, S, n_g* | peak n_e, T_e, per-cusp quantities | sheath smeared by one cell; competes with 3 | M (1-2 d) | same ladder as 3; not combined with 3 in one campaign |
| 9 | permittivity scaling gamma = 4 (3.1) | 8x raw | 8x raw | design ranking (mini-sweep) if stable between gamma = 2 and 4 | I_d, S, thrust (Szabo 2014: 5-16 % after retrieval) | per-cusp sheath drop, wall ion energy, peak n_e, leak width vs sheath, n_g* (Yuan et al. 2020) | S + a 3-run gamma-series | gamma in {1, 2, 4} same seed and window, extrapolation to gamma = 1 within 10 % on I_d and S; sheath quantities reported as scaled, never as claims |
| 10 | block cyclic reduction of the existing solve (5.4 ii) | 1.3x | 1.2x | all | none | none | L (3-4 d) | as rank 1 |
| 11 | sparse-grid combination PIC (7.6) | 3-5x on N at fixed noise (Garrigues 2024, multi-cusp) | same | all in principle | all (grid-based error changes) | wall-current error < 5 % reported, cusp precedent | XL (weeks) | DEFER; trigger: a rung needing >= 4x N |
| 12 | fully implicit EC PIC with an asymptotic-preserving push (2.4) | 2-5x net after the nonlinear-solve cost | 2-5x | - | all | cusp de-magnetisation region forces small steps; ppc penalty (Savard 2025); no LTP / thruster precedent | XL (months) | not for this project's horizon |
| 13 | ion-mass scaling (3.2) | up to 10x in steps | same | none | - | n_e, utilisation, n_g*, sheaths, IEDF (Yuan et al. 2020: "significant errors in the potential and current") | S | not recommended |
| 14 | geometric self-similarity (3.3) | - | - | - | - | surface processes, sheath / L (Brandt 2016; Matthias 2020); we are already the small end | - | not applicable |
| 15 | multi-GPU domain decomposition for one run (6.5) | ~1x or worse | ~1x | all | none | none | XL | not recommended: latency-bound step; use MPS slots for ensembles |
| 16 | delta-f, quiet starts, importance sampling (7.2, 7.3, 7.5) | ~1x | ~1x (statistics gain in the far plume from splitting) | all | none | none | S-M | not a speed-up |
| 17 | hybrid electrons; reduced-order (r,theta)/(z,theta) PIC (4.5, 3.4) | - | - | - | - | - | - | dropped by the user / not (r,z) |

**Stacking.** Ranks 1 + 2 + 7 are physics-neutral and multiply to **~1.8-2.3x on the channel
(5 h -> 2.2-2.8 h) and ~3-3.5x on the plume (48 h -> 14-16 h)**. Adding rank 3 at 50 um gives
**~2.5-3x on the channel and ~7x on the plume (48 h -> ~7 h)** with the per-cusp quantities
under re-validation. Reaching **>= 5x on the channel** requires rank 5 (semi-implicit
coarsening), i.e. changing what is resolved, and carries the sheath disclosure. The requested
5-10x is therefore reachable on the plume without touching the physics of the bulk claims
(ranks 1-3), and on the channel only through rank 5 with a reduced claim set.

**Top three, with the risks stated once more.**

1. **Geometric multigrid Poisson (rank 1)**: 1.3x channel, 2.2x plume; no physics risk; it
   also unlocks ranks 3 and 5 (variable coefficients, graded grids) and removes the 6 GB block
   store and the host factorisation that currently limit the plume box. Risk: none to the
   claims; engineering risk that a masked axisymmetric MG needs an AMG fallback (5.3).
2. **Explicit energy-conserving gather with a same-seed scheme swap against the accepted 33 um
   plateau (rank 3)**: removes the hard pi gate that forces 33 / 25 um, ~2x channel at fixed
   ppc and ~2.5x plume at 50 um. Risk: the per-cusp sheath drop and wall flux become the
   unresolved quantities at 2-3 cells per sheath (Powis and Kaganovich 2024; Hedlof et al.
   2026), EC schemes can still heat drifting populations (Adams et al. 2025), and thrust needs
   a momentum ledger because EC-PIC gives up exact momentum conservation.
3. **Barnes / ECsim semi-implicit coarsening (rank 5)**, after 1 and 3: 3.5x at 2x/2x, up to
   ~10x at 4x/4x on the only measured thruster precedent (Marks and Gorodetsky 2025). Risk: the
   implicitness constant C is a tuning parameter calibrated against the explicit answer, T_e
   deviates by < 10 % (2x) to < 25 % (4x-8x), and the cusp sheaths are unresolved by
   construction; the bulk claims (I_d, S, n_g*, I_beam) are the candidates, the per-cusp claims
   are not.

---

## 9. Applicability matrix (method x claim)

P = preserved (same equations or shown by the literature not to move the quantity); R =
re-validate under the section-8 protocol; X = distorted by the method according to the cited
literature; - = not applicable. Claims are the repository's recorded observables.

| method | I_d | S, net utilisation | n_g* (fixed point) | peak n_e | T_e,peak | per-cusp wall flux / energy | per-cusp sheath drop | I_beam / thrust | IEDF / divergence | energy ledger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 multigrid Poisson | P | P | P | P | P | P | P | P | P | P |
| 2 fusion + sort | P | P | P | P | P | P | P | P | P | P |
| 3 explicit EC gather, 50 um | R | R | R | R | R | R | X (1-3 cells / sheath) | R (momentum ledger) | R | P (exact by construction) |
| 4 coarse-to-fine restart | P | P | P | P | P | P | P | P | P | P |
| 5 semi-implicit, 2x/2x | R | R | R | R | R (C-dependent) | X | X | R | R | P (exact by construction) |
| 6 merging in the plume | P | P | P | P | P | P | P | R | R (tails) | R |
| 7 mixed precision | P | P | P | P | P | P | P | P | P | R (accumulation) |
| 8 digital filtering | R | R | R | R | R | R | X (smeared) | R | R | R (filtered rho) |
| 9 permittivity gamma = 4 | R (5-16 %) | R | X | X | R | X | X (sqrt(gamma)) | R | R | P |
| 11 sparse grids | R | R | R | R | R | R (< 5 % wall current) | R | R | R | R |
| 12 fully implicit + AP push | R | R | R | R | R | X (cusp de-magnetisation) | X | R | R | P |
| 13 ion-mass scaling | X | X | X | X | R | X | X | X | X | P |

---

## 10. Where the literature is silent

- No energy-conserving (explicit, semi-implicit or fully implicit) PIC of a cusped-field or
  HEMP thruster exists; the thruster precedent (Marks and Gorodetsky 2025) is the axial-azimuthal
  Charoy et al. 2019 benchmark with a prescribed ionisation source and no walls in the plane.
- No study of the EC finite-grid stability threshold in a *magnetised* plasma (Taccogna et al.
  2023 say so explicitly); Adams et al. 2025 and Barnes and Chacon 2021 are unmagnetised.
- No published convergence of a cusp sheath drop or cusp wall flux with Delta/lambda_D under any
  scheme; Powis and Kaganovich 2024 is the only sheath-resolution study for EC-PIC (1-D, RF).
- No quantitative permittivity-scaling error for a millimetre-bore cusp device; Yuan et al.
  2020 is a centimetre ring-cusp chamber and reports the failure mode qualitatively.
- No PIC grid-sequencing / coarse-restart study; the parareal literature has no PIC entry.
- No GPU multigrid-vs-direct comparison for a masked axisymmetric electrostatic PIC domain;
  Kahnfeld et al. 2016 compare LU with SOR on the CPU.
- No published seed-to-seed statistics band for a HEMP/CFT PIC against which a
  variance-reduction claim could be measured (already noted in `pic-mcc-blockers.md` 5(e)).
- Sparse-grid PIC has a multi-cusp test case (Garrigues et al. 2024) but no axisymmetric,
  masked, Dirichlet-body application.

---

## 11. Bibliography (147 entries, alphabetical; DOI for every entry; all resolved 2026-09-04)

1. Adam, J. C.; Heron, A.; Laval, G. "Study of stationary plasma thrusters using two-dimensional fully kinetic simulations." *Physics of Plasmas* 11, 295-305 (2004). doi:10.1063/1.1632904
2. Adam, J. C.; Gourdin Serveniere, A.; Langdon, A. B. "Electron sub-cycling in particle simulation of plasma." *Journal of Computational Physics* 47, 229-244 (1982). doi:10.1016/0021-9991(82)90076-6
3. Adams, Luke C.; Werner, Gregory R.; Cary, John R. "Grid instability growth rates for explicit, electrostatic momentum- and energy-conserving particle-in-cell algorithms." *Physics of Plasmas* 32, 093905 (2025). doi:10.1063/5.0271598
4. Angus, Justin Ray; Link, Anthony; Friedman, Alex; Ghosh, Debojyoti; Johnson, Jamal D. "On numerical energy conservation for an implicit particle-in-cell method coupled with a binary Monte-Carlo algorithm for Coulomb collisions." *Journal of Computational Physics* 456, 111030 (2022). doi:10.1016/j.jcp.2022.111030
5. Angus, Justin Ray; Farmer, William; Friedman, Alex; Ghosh, Debojyoti; Grote, Dave; Larson, David; Link, Anthony. "An implicit particle code with exact energy and charge conservation for electromagnetic studies of dense plasmas." *Journal of Computational Physics* 491, 112383 (2023). doi:10.1016/j.jcp.2023.112383
6. Angus, Justin Ray; Farmer, William; Friedman, Alex; Geyko, Vasily; Ghosh, Debojyoti; Grote, Dave; Larson, David; Link, Anthony. "An implicit particle code with exact energy and charge conservation for studies of dense plasmas in axisymmetric geometries." *Journal of Computational Physics* 519, 113427 (2024). doi:10.1016/j.jcp.2024.113427
7. Assous, F.; Dulimbert, T.; Segre, J. "A new method for coalescing particles in PIC codes." *Journal of Computational Physics* 187, 550-571 (2003). doi:10.1016/S0021-9991(03)00124-4
8. Aydemir, A. Y. "A unified Monte Carlo interpretation of particle simulations and applications to non-neutral plasmas." *Physics of Plasmas* 1, 822-831 (1994). doi:10.1063/1.870740
9. Barnes, D. C.; Chacon, L. "Finite spatial-grid effects in energy-conserving particle-in-cell algorithms." *Computer Physics Communications* 258, 107560 (2021). doi:10.1016/j.cpc.2020.107560
10. Barnes, D. C. "Improved C1 shape functions for simplex meshes." *Journal of Computational Physics* 424, 109852 (2021). doi:10.1016/j.jcp.2020.109852 (the appendix carries the semi-implicit modified-Poisson scheme, per Marks and Gorodetsky 2025)
11. Bell, Nathan; Garland, Michael. "Implementing sparse matrix-vector multiplication on throughput-oriented processors." *Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis (SC '09)*, 1-11 (2009). doi:10.1145/1654059.1654078
12. Bell, Nathan; Dalton, Steven; Olson, Luke N. "Exposing Fine-Grained Parallelism in Algebraic Multigrid Methods." *SIAM Journal on Scientific Computing* 34, C123-C152 (2012). doi:10.1137/110838844
13. Birdsall, C. K.; Langdon, A. B. *Plasma Physics via Computer Simulation.* IOP Publishing (1991). doi:10.1887/0750301171
14. Birdsall, Charles K.; Maron, Neil. "Plasma self-heating and saturation due to numerical instabilities." *Journal of Computational Physics* 36, 1-19 (1980). doi:10.1016/0021-9991(80)90171-0
15. Boeuf, Jean-Pierre. "Tutorial: Physics and modeling of Hall thrusters." *Journal of Applied Physics* 121, 011101 (2017). doi:10.1063/1.4972269
16. Boeuf, J. P.; Garrigues, L. "E x B electron drift instability in Hall thrusters: Particle-in-cell simulations vs. theory." *Physics of Plasmas* 25, 061204 (2018). doi:10.1063/1.5017033
17. Bowers, K. J. "Accelerating a Particle-in-Cell Simulation Using a Hybrid Counting Sort." *Journal of Computational Physics* 173, 393-411 (2001). doi:10.1006/jcph.2001.6851
18. Bowers, K. J.; Albright, B. J.; Yin, L.; Bergen, B.; Kwan, T. J. T. "Ultrahigh performance three-dimensional electromagnetic relativistic kinetic plasma simulation." *Physics of Plasmas* 15, 055703 (2008). doi:10.1063/1.2840133
19. Brackbill, J. U.; Forslund, D. W. "An implicit method for electromagnetic plasma simulation in two dimensions." *Journal of Computational Physics* 46, 271-308 (1982). doi:10.1016/0021-9991(82)90016-X
20. Brackbill, J. U. "On energy and momentum conservation in particle-in-cell plasma simulation." *Journal of Computational Physics* 317, 405-427 (2016). doi:10.1016/j.jcp.2016.04.050
21. Brandt, Achi. "Multi-level adaptive solutions to boundary-value problems." *Mathematics of Computation* 31, 333-390 (1977). doi:10.1090/S0025-5718-1977-0431719-X
22. Brandt, Tim; Schneider, Ralf; Duras, Julia; Kahnfeld, Daniel; Hey, Franz Georg; Kersten, Holger; Jansen, Frank; Braxmaier, Claus. "Particle-in-Cell Simulation of a Down-Scaled HEMP Thruster." *Transactions of the Japan Society for Aeronautical and Space Sciences, Aerospace Technology Japan* 14 (ists30), Pb_235-Pb_242 (2016). doi:10.2322/tastj.14.Pb_235
23. Briggs, William L.; Henson, Van Emden; McCormick, Steve F. *A Multigrid Tutorial, Second Edition.* SIAM (2000). doi:10.1137/1.9780898719505
24. Burau, Heiko; Widera, Rene; Honig, Wolfgang; Juckeland, Guido; Debus, Alexander; Kluge, Thomas; Schramm, Ulrich; Cowan, Tomas E.; Sauerbrey, Roland; Bussmann, Michael. "PIConGPU: A Fully Relativistic Particle-in-Cell Code for a GPU Cluster." *IEEE Transactions on Plasma Science* 38, 2831-2839 (2010). doi:10.1109/TPS.2010.2064310
25. Bussmann, M.; Burau, H.; Cowan, T. E.; Debus, A.; Huebl, A.; Juckeland, G.; Kluge, T.; Nagel, W. E.; Pausch, R.; Schmitt, F.; Schramm, U.; Schuchart, J.; Widera, R. "Radiative signatures of the relativistic Kelvin-Helmholtz instability." *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC '13)*, 1-12 (2013). doi:10.1145/2503210.2504564
26. Buzbee, B. L.; Golub, G. H.; Nielson, C. W. "On Direct Methods for Solving Poisson's Equations." *SIAM Journal on Numerical Analysis* 7, 627-656 (1970). doi:10.1137/0707049
27. Buzbee, B. L.; Dorr, F. W.; George, J. A.; Golub, G. H. "The Direct Solution of the Discrete Poisson Equation on Irregular Regions." *SIAM Journal on Numerical Analysis* 8, 722-736 (1971). doi:10.1137/0708066
28. Chacon, L.; Chen, G.; Barnes, D. C. "A charge- and energy-conserving implicit, electrostatic particle-in-cell algorithm on mapped computational meshes." *Journal of Computational Physics* 233, 1-9 (2013). doi:10.1016/j.jcp.2012.07.042
29. Chacon, L.; Chen, G. "A curvilinear, fully implicit, conservative electromagnetic PIC algorithm in multiple dimensions." *Journal of Computational Physics* 316, 578-597 (2016). doi:10.1016/j.jcp.2016.03.070
30. Charoy, T.; Boeuf, J. P.; Bourdon, A.; Carlsson, J. A.; Chabert, P.; Cuenot, B.; Eremin, D.; Garrigues, L.; Hara, K.; Kaganovich, I. D.; Powis, A. T.; Smolyakov, A.; Sydorenko, D.; Tavant, A.; Vermorel, O.; Villafana, W. "2D axial-azimuthal particle-in-cell benchmark for low-temperature partially magnetized plasmas." *Plasma Sources Science and Technology* 28, 105010 (2019). doi:10.1088/1361-6595/ab46c5
31. Chen, G.; Chacon, L.; Barnes, D. C. "An energy- and charge-conserving, implicit, electrostatic particle-in-cell algorithm." *Journal of Computational Physics* 230, 7018-7036 (2011). doi:10.1016/j.jcp.2011.05.031
32. Chen, G.; Chacon, L.; Barnes, D. C. "An efficient mixed-precision, hybrid CPU-GPU implementation of a nonlinearly implicit one-dimensional particle-in-cell algorithm." *Journal of Computational Physics* 231, 5374-5388 (2012). doi:10.1016/j.jcp.2012.04.040
33. Chen, G.; Chacon, L. "An analytical particle mover for the charge- and energy-conserving, nonlinearly implicit, electrostatic particle-in-cell algorithm." *Journal of Computational Physics* 247, 79-87 (2013). doi:10.1016/j.jcp.2013.04.002
34. Chen, G.; Chacon, L.; Leibs, C. A.; Knoll, D. A.; Taitano, W. "Fluid preconditioning for Newton-Krylov-based, fully implicit, electrostatic particle-in-cell simulations." *Journal of Computational Physics* 258, 555-567 (2014). doi:10.1016/j.jcp.2013.10.052
35. Chen, G.; Chacon, L. "A multi-dimensional, energy- and charge-conserving, nonlinearly implicit, electromagnetic Vlasov-Darwin particle-in-cell algorithm." *Computer Physics Communications* 197, 73-87 (2015). doi:10.1016/j.cpc.2015.08.008
36. Chen, G.; Chacon, L. "An implicit, conservative and asymptotic-preserving electrostatic particle-in-cell algorithm for arbitrarily magnetized plasmas in uniform magnetic fields." *Journal of Computational Physics* 487, 112160 (2023). doi:10.1016/j.jcp.2023.112160
37. Cho, Shinatora; Komurasaki, Kimiya; Arakawa, Yoshihiro. "Kinetic particle simulation of discharge and wall erosion of a Hall thruster." *Physics of Plasmas* 20, 063501 (2013). doi:10.1063/1.4810798
38. Cho, Shinatora; Watanabe, Hiroki; Kubota, Kenichi; Iihara, Shigeyasu; Fuchigami, Kenji; Uematsu, Kazuo; Funaki, Ikkoh. "Study of electron transport in a Hall thruster by axial-radial fully kinetic particle simulation." *Physics of Plasmas* 22, 103523 (2015). doi:10.1063/1.4935049
39. Coche, P.; Garrigues, L. "A two-dimensional (azimuthal-axial) particle-in-cell model of a Hall thruster." *Physics of Plasmas* 21, 023503 (2014). doi:10.1063/1.4864625
40. Cohen, Bruce I.; Langdon, A. Bruce; Friedman, A. "Implicit time integration for plasma simulation." *Journal of Computational Physics* 46, 15-38 (1982). doi:10.1016/0021-9991(82)90002-X
41. Cohen, Bruce I.; Langdon, A. Bruce; Hewett, Dennis W.; Procassini, Richard J. "Performance and optimization of direct implicit particle simulation." *Journal of Computational Physics* 81, 151-168 (1989). doi:10.1016/0021-9991(89)90068-5
42. Dawson, John M. "Particle simulation of plasmas." *Reviews of Modern Physics* 55, 403-447 (1983). doi:10.1103/RevModPhys.55.403
43. Decyk, Viktor K.; Singh, Tajendra V. "Adaptable Particle-in-Cell algorithms for graphical processing units." *Computer Physics Communications* 182, 641-648 (2011). doi:10.1016/j.cpc.2010.11.009
44. Decyk, Viktor K.; Singh, Tajendra V. "Particle-in-Cell algorithms for emerging computer architectures." *Computer Physics Communications* 185, 708-719 (2014). doi:10.1016/j.cpc.2013.10.013
45. Deluzet, Fabrice; Fubiani, Gwenael; Garrigues, Laurent; Guillet, Clement; Narski, Jacek. "Sparse grid reconstructions for Particle-In-Cell methods." *ESAIM: Mathematical Modelling and Numerical Analysis* 56, 1809-1841 (2022). doi:10.1051/m2an/2022055
46. Denton, Richard E.; Kotschenreuther, M. "delta f Algorithm." *Journal of Computational Physics* 119, 283-294 (1995). doi:10.1006/jcph.1995.1136
47. Derouillat, J.; Beck, A.; Perez, F.; Vinci, T.; Chiaramello, M.; Grassi, A.; Fle, M.; Bouchard, G.; Plotnikov, I.; Aunai, N.; Dargent, J.; Riconda, C.; Grech, M. "Smilei: A collaborative, open-source, multi-purpose particle-in-cell code for plasma simulation." *Computer Physics Communications* 222, 351-373 (2018). doi:10.1016/j.cpc.2017.09.024
48. Duras, J.; Matyash, K.; Tskhakaya, D.; Kalentev, O.; Schneider, R. "Self-Force in 1D Electrostatic Particle-in-Cell Codes for Non-Equidistant Grids." *Contributions to Plasma Physics* 54, 697-711 (2014). doi:10.1002/ctpp.201300060
49. Duras, J.; Kahnfeld, D.; Bandelow, G.; Kemnitz, S.; Luskow, K.; Matthias, P.; Koch, N.; Schneider, R. "Ion angular distribution simulation of the Highly Efficient Multistage Plasma Thruster." *Journal of Plasma Physics* 83, 595830107 (2017). doi:10.1017/s0022377817000125
50. Eremin, D. "An energy- and charge-conserving electrostatic implicit particle-in-cell algorithm for simulations of collisional bounded plasmas." *Journal of Computational Physics* 452, 110934 (2022). doi:10.1016/j.jcp.2021.110934
51. Esirkepov, T. Zh. "Exact charge conservation scheme for Particle-in-Cell simulation with an arbitrary form-factor." *Computer Physics Communications* 135, 144-153 (2001). doi:10.1016/S0010-4655(00)00228-9
52. Faghihi, D.; Carey, V.; Michoski, C.; Hager, R.; Janhunen, S.; Chang, C. S.; Moser, R. D. "Moment preserving constrained resampling with applications to particle-in-cell methods." *Journal of Computational Physics* 409, 109317 (2020). doi:10.1016/j.jcp.2020.109317
53. Faraji, F.; Reza, M.; Knoll, A. "Enhancing one-dimensional particle-in-cell simulations to self-consistently resolve instability-induced electron transport in Hall thrusters." *Journal of Applied Physics* 131, 193302 (2022). doi:10.1063/5.0090853
54. Fedeli, Luca; Huebl, Axel; Boillod-Cerneux, France; Clark, Thomas; Gott, Kevin; Hillairet, Conrad; Jaure, Stephan; Leblanc, Adrien; Lehe, Remi; Myers, Andrew; Piechurski, Christelle; Sato, Mitsuhisa; Zaim, Neil; Zhang, Weiqun; Vay, Jean-Luc; Vincenti, Henri. "Pushing the Frontier in the Design of Laser-Based Electron Accelerators with Groundbreaking Mesh-Refined Particle-in-Cell Simulations on Exascale-Class Supercomputers." *SC22: International Conference for High Performance Computing, Networking, Storage and Analysis*, 1-12 (2022). doi:10.1109/SC41404.2022.00008
55. Filipovic, Jiri; Madzin, Matus; Fousek, Jan; Matyska, Ludek. "Optimizing CUDA code by kernel fusion: application on BLAS." *The Journal of Supercomputing* 71, 3934-3957 (2015). doi:10.1007/s11227-015-1483-z
56. Fonseca, R. A.; Vieira, J.; Fiuza, F.; Davidson, A.; Tsung, F. S.; Mori, W. B.; Silva, L. O. "Exploiting multi-scale parallelism for large scale numerical modelling of laser wakefield accelerators." *Plasma Physics and Controlled Fusion* 55, 124011 (2013). doi:10.1088/0741-3335/55/12/124011
57. Friedman, Alex. "A second-order implicit particle mover with adjustable damping." *Journal of Computational Physics* 90, 292-312 (1990). doi:10.1016/0021-9991(90)90168-Z
58. Fubiani, G.; Garrigues, L.; Hagelaar, G.; Kohen, N.; Boeuf, J. P. "Modeling of plasma transport and negative ion extraction in a magnetized radio-frequency plasma source." *New Journal of Physics* 19, 015002 (2017). doi:10.1088/1367-2630/19/1/015002
59. Fubiani, G.; Garrigues, L.; Boeuf, J. P. "Modeling of negative ion extraction from a magnetized plasma source: Derivation of scaling laws and description of the origins of aberrations in the ion beam." *Physics of Plasmas* 25, 023510 (2018). doi:10.1063/1.4999707
60. Garrigues, L.; Fubiani, G.; Boeuf, J. P. "Negative ion extraction via particle simulation for fusion: critical assessment of recent contributions." *Nuclear Fusion* 57, 014003 (2017; online 2016). doi:10.1088/0029-5515/57/1/014003
61. Garrigues, L.; Tezenas du Montcel, B.; Fubiani, G.; Bertomeu, F.; Deluzet, F.; Narski, J. "Application of sparse grid combination techniques to low temperature plasmas particle-in-cell simulations. I. Capacitively coupled radio frequency discharges." *Journal of Applied Physics* 129, 153303 (2021). doi:10.1063/5.0044363
62. Garrigues, L.; Tezenas du Montcel, B.; Fubiani, G.; Reman, B. C. G. "Application of sparse grid combination techniques to low temperature plasmas Particle-In-Cell simulations. II. Electron drift instability in a Hall thruster." *Journal of Applied Physics* 129, 153304 (2021). doi:10.1063/5.0044865
63. Garrigues, L.; Fubiani, G. "Tutorial: Modeling of the extraction and acceleration of negative ions from plasma sources using particle-based methods." *Journal of Applied Physics* 133, 041102 (2023). doi:10.1063/5.0128759
64. Garrigues, L.; Chung-To-Sang, M.; Fubiani, G.; Guillet, C.; Deluzet, F.; Narski, J. "Acceleration of particle-in-cell simulations using sparse grid algorithms. II. Application to partially magnetized low temperature plasmas." *Physics of Plasmas* 31, 073908 (2024). doi:10.1063/5.0211220
65. Germaschewski, Kai; Fox, William; Abbott, Stephen; Ahmadi, Narges; Maynard, Kristofor; Wang, Liang; Ruhl, Hartmut; Bhattacharjee, Amitava. "The Plasma Simulation Code: A modern particle-in-cell code with patch-based load balancing." *Journal of Computational Physics* 318, 305-326 (2016). doi:10.1016/j.jcp.2016.05.013
66. Gibou, Frederic; Fedkiw, Ronald P.; Cheng, Li-Tien; Kang, Myungjoo. "A Second-Order-Accurate Symmetric Discretization of the Poisson Equation on Irregular Domains." *Journal of Computational Physics* 176, 205-227 (2002). doi:10.1006/jcph.2001.6977
67. Hara, Kentaro. "An overview of discharge plasma modeling for Hall effect thrusters." *Plasma Sources Science and Technology* 28, 044001 (2019). doi:10.1088/1361-6595/ab0f70
68. Hedlof, Ryan M.; Barnes, Daniel C.; Groenewald, Roelof E.; Necas, Ales; Smith, Thomas M.; Lau, Calvin K.; Brandt, Steven; Zhang, Weiqun; Eckert, Zakari; Hooper, Russell. "Verification of an energy-conserving semi-implicit electrostatic particle-in-cell scheme for modeling high-density plasma at scale." *Physics of Plasmas* 33, 053902 (2026). doi:10.1063/5.0315721
69. Hockney, R. W. "A Fast Direct Solution of Poisson's Equation Using Fourier Analysis." *Journal of the ACM* 12, 95-113 (1965). doi:10.1145/321250.321259
70. Hockney, R. W. "Measurements of collision and heating times in a two-dimensional thermal computer plasma." *Journal of Computational Physics* 8, 19-44 (1971). doi:10.1016/0021-9991(71)90032-5
71. Hockney, R. W.; Eastwood, J. W. *Computer Simulation Using Particles.* Adam Hilger / CRC Press (1988). doi:10.1201/9781439822050
72. Hu, Genze; Krommes, John A. "Generalized weighting scheme for delta f particle-simulation method." *Physics of Plasmas* 1, 863-874 (1994). doi:10.1063/1.870745
73. Johansen, Hans; Colella, Phillip. "A Cartesian Grid Embedded Boundary Method for Poisson's Equation on Irregular Domains." *Journal of Computational Physics* 147, 60-85 (1998). doi:10.1006/jcph.1998.5965
74. Juhasz, Zoltan; Durian, Jan; Derzsi, Aranka; Matejcik, Stefan; Donko, Zoltan; Hartmann, Peter. "Efficient GPU implementation of the Particle-in-Cell/Monte-Carlo collisions method for 1D simulation of low-pressure capacitively coupled plasmas." *Computer Physics Communications* 263, 107913 (2021). doi:10.1016/j.cpc.2021.107913
75. Kahnfeld, D.; Schneider, R.; Matyash, K.; Kalentev, O.; Kemnitz, S.; Duras, J.; Luskow, K.; Bandelow, G. "Solution of Poisson's Equation in Electrostatic Particle-in-Cell Simulation." *Plasma Physics and Technology* 3, 66-71 (2016). doi:10.14311/ppt.2016.2.66 (title as deposited: "Solutioin of Poisson's Equation in Electrostatic Particle-on-cell Simulation")
76. Kahnfeld, D.; Heidemann, R.; Duras, J.; Matthias, P.; Bandelow, G.; Luskow, K.; Kemnitz, S.; Matyash, K.; Schneider, R. "Breathing modes in HEMP thrusters." *Plasma Sources Science and Technology* 27, 124002 (2018). doi:10.1088/1361-6595/aaf29a
77. Kahnfeld, Daniel; Duras, Julia; Matthias, Paul; Kemnitz, Stefan; Arlinghaus, Peter; Bandelow, Gunnar; Matyash, Konstantin; Koch, Norbert; Schneider, Ralf. "Numerical modeling of high efficiency multistage plasma thrusters for space applications." *Reviews of Modern Plasma Physics* 3, 11 (2019). doi:10.1007/s41614-019-0030-4
78. Kraus, Michael; Kormann, Katharina; Morrison, Philip J.; Sonnendrucker, Eric. "GEMPIC: geometric electromagnetic particle-in-cell methods." *Journal of Plasma Physics* 83, 905830401 (2017). doi:10.1017/S002237781700040X
79. Lacina, J. "Similarity rules in plasma physics." *Plasma Physics* 13, 303-312 (1971). doi:10.1088/0032-1028/13/4/003
80. Langdon, A. Bruce. "Effects of the spatial grid in simulation plasmas." *Journal of Computational Physics* 6, 247-267 (1970). doi:10.1016/0021-9991(70)90024-0
81. Langdon, A. Bruce. "'Energy-conserving' plasma simulation algorithms." *Journal of Computational Physics* 12, 247-268 (1973). doi:10.1016/s0021-9991(73)80014-2
82. Langdon, A. Bruce; Cohen, Bruce I.; Friedman, Alex. "Direct implicit large time-step particle simulation of plasmas." *Journal of Computational Physics* 51, 107-138 (1983). doi:10.1016/0021-9991(83)90083-9
83. Lapenta, Giovanni; Brackbill, J. U. "Dynamic and Selective Control of the Number of Particles in Kinetic Plasma Simulations." *Journal of Computational Physics* 115, 213-227 (1994). doi:10.1006/jcph.1994.1188
84. Lapenta, Giovanni. "Exactly energy conserving semi-implicit particle in cell formulation." *Journal of Computational Physics* 334, 349-366 (2017). doi:10.1016/j.jcp.2017.01.002
85. Lapenta, Giovanni. "Advances in the Implementation of the Exactly Energy Conserving Semi-Implicit (ECsim) Particle-in-Cell Method." *Physics* 5, 72-89 (2023). doi:10.3390/physics5010007
86. Lewis, H. Ralph. "Energy-conserving numerical approximations for Vlasov plasmas." *Journal of Computational Physics* 6, 136-141 (1970). doi:10.1016/0021-9991(70)90012-4
87. Liu, Wei; Wang, Weizong; Li, Yifei; Xue, Shuwen. "Revealing the plasma confinement behavior of an axial ring cusp hybrid discharge in a miniature ion thruster using PIC/MCC simulation." *Plasma Sources Science and Technology* 32, 085005 (2023). doi:10.1088/1361-6595/ace92d
88. Luu, Phuc T.; Tuckmantel, T.; Pukhov, A. "Voronoi particle merging algorithm for PIC codes." *Computer Physics Communications* 202, 165-174 (2016). doi:10.1016/j.cpc.2016.01.009
89. Markidis, Stefano; Lapenta, Giovanni; Rizwan-uddin. "Multi-scale simulations of plasma with iPIC3D." *Mathematics and Computers in Simulation* 80, 1509-1519 (2010). doi:10.1016/j.matcom.2009.08.038
90. Markidis, Stefano; Lapenta, Giovanni. "The energy conserving particle-in-cell method." *Journal of Computational Physics* 230, 7037-7052 (2011). doi:10.1016/j.jcp.2011.05.033
91. Marks, Thomas A.; Gorodetsky, Alex A. "GPU-accelerated kinetic Hall thruster simulations in WarpX." *Journal of Electric Propulsion* 4, 34 (2025). doi:10.1007/s44205-025-00133-1
92. Mason, Rodney J. "Implicit moment particle simulation of plasmas." *Journal of Computational Physics* 41, 233-244 (1981). doi:10.1016/0021-9991(81)90094-2
93. Mattei, S.; Nishida, K.; Onai, M.; Lettry, J.; Tran, M. Q.; Hatayama, A. "A fully-implicit Particle-In-Cell Monte Carlo Collision code for the simulation of inductively coupled plasmas." *Journal of Computational Physics* 350, 891-906 (2017). doi:10.1016/j.jcp.2017.09.015
94. Matthias, Paul; Kahnfeld, Daniel; Kemnitz, Stefan; Duras, Julia; Koch, Norbert; Schneider, Ralf. "Similarity scaling - application and limits for high-efficiency-multistage-plasma-thruster particle-in-cell modelling." *Contributions to Plasma Physics* 60, e201900199 (2020). doi:10.1002/ctpp.201900199
95. Mertmann, Philipp; Eremin, Denis; Mussenbrock, Thomas; Brinkmann, Ralf Peter; Awakowicz, Peter. "Fine-sorting one-dimensional particle-in-cell algorithm with Monte-Carlo collisions on a graphics processing unit." *Computer Physics Communications* 182, 2161-2167 (2011). doi:10.1016/j.cpc.2011.05.012
96. Muralikrishnan, Sriramkrishnan; Cerfon, Antoine J.; Frey, Matthias; Ricketson, Lee F.; Adelmann, Andreas. "Sparse grid-based adaptive noise reduction strategy for particle-in-cell schemes." *Journal of Computational Physics: X* 11, 100094 (2021). doi:10.1016/j.jcpx.2021.100094
97. Muraviev, A.; Bashinov, A.; Efimenko, E.; Volokitin, V.; Meyerov, I.; Gonoskov, A. "Strategies for particle resampling in PIC simulations." *Computer Physics Communications* 262, 107826 (2021). doi:10.1016/j.cpc.2021.107826
98. Myers, A.; Almgren, A.; Amorim, L. D.; Bell, J.; Fedeli, L.; Ge, L.; Gott, K.; Grote, D. P.; Hogan, M.; Huebl, A.; Jambunathan, R.; Lehe, R.; Ng, C.; Rowan, M.; Shapoval, O.; Thevenet, M.; Vay, J.-L.; Vincenti, H.; Yang, E.; Zaim, N.; Zhang, W.; Zhao, Y.; Zoni, E. "Porting WarpX to GPU-accelerated platforms." *Parallel Computing* 108, 102833 (2021). doi:10.1016/j.parco.2021.102833
99. Naumov, M.; Arsaev, M.; Castonguay, P.; Cohen, J.; Demouth, J.; Eaton, J.; Layton, S.; Markovskiy, N.; Reguly, I.; Sakharnykh, N.; Sellappan, V.; Strzodka, R. "AmgX: A Library for GPU Accelerated Algebraic Multigrid and Preconditioned Iterative Methods." *SIAM Journal on Scientific Computing* 37, S602-S626 (2015). doi:10.1137/140980260
100. Nevins, W. M.; Hammett, G. W.; Dimits, A. M.; Dorland, W.; Shumaker, D. E. "Discrete particle noise in particle-in-cell simulations of plasma microturbulence." *Physics of Plasmas* 12, 122305 (2005). doi:10.1063/1.2118729
101. Nieter, Chet; Cary, John R. "VORPAL: a versatile plasma simulation code." *Journal of Computational Physics* 196, 448-473 (2004). doi:10.1016/j.jcp.2003.11.004
102. Okuda, Hideo; Birdsall, C. K. "Collisions in a Plasma of Finite-Size Particles." *The Physics of Fluids* 13, 2123-2134 (1970). doi:10.1063/1.1693210
103. Parker, S. E.; Lee, W. W. "A fully nonlinear characteristic method for gyrokinetic simulation." *Physics of Fluids B: Plasma Physics* 5, 77-86 (1993). doi:10.1063/1.860870
104. Parodi, Pietro; Lapenta, Giovanni; Magin, Thierry. "Simulation of plasmas for electric propulsion using the energy-conserving semi-implicit PIC scheme." *9th European Conference for Aeronautics and Space Sciences (EUCASS)*, Lille (2022). doi:10.13009/EUCASS2022-7331 (conference paper; DOI resolved through doi.org to the EUCASS full text, not indexed by Crossref)
105. Parra, F. I.; Ahedo, E.; Fife, J. M.; Martinez-Sanchez, M. "A two-dimensional hybrid model of the Hall thruster discharge." *Journal of Applied Physics* 100, 023304 (2006). doi:10.1063/1.2219165
106. Petronio, Federico; Charoy, Thomas; Alvarez Laguna, Alejandro; Bourdon, Anne; Chabert, Pascal. "Two-dimensional effects on electrostatic instabilities in Hall thrusters. I. Insights from particle-in-cell simulations and two-point power spectral density reconstruction techniques." *Physics of Plasmas* 30, 012103 (2023). doi:10.1063/5.0119253
107. Petronio, Federico; Alvarez Laguna, Alejandro; Bourdon, Anne; Chabert, Pascal. "Study of the breathing mode development in Hall thrusters using hybrid simulations." *Journal of Applied Physics* 135, 073301 (2024). doi:10.1063/5.0188859
108. Pfeiffer, M.; Mirza, A.; Munz, C.-D.; Fasoulas, S. "Two statistical particle split and merge methods for Particle-in-Cell codes." *Computer Physics Communications* 191, 9-24 (2015). doi:10.1016/j.cpc.2015.01.010
109. Powis, A. T.; Kaganovich, I. D. "Accuracy of the explicit energy-conserving particle-in-cell method for under-resolved simulations of capacitively coupled plasma discharges." *Physics of Plasmas* 31, 023901 (2024). doi:10.1063/5.0174168
110. Proskurowski, Wlodzimierz; Widlund, Olof. "On the numerical solution of Helmholtz's equation by the capacitance matrix method." *Mathematics of Computation* 30, 433-468 (1976). doi:10.1090/s0025-5718-1976-0421102-4
111. Qin, Hong; Zhang, Shuangxi; Xiao, Jianyuan; Liu, Jian; Sun, Yajuan; Tang, William M. "Why is Boris algorithm so good?" *Physics of Plasmas* 20, 084503 (2013). doi:10.1063/1.4818428
112. Reza, Maryam; Faraji, Farbod; Knoll, Aaron. "Concept of the generalized reduced-order particle-in-cell scheme and verification in an axial-azimuthal Hall thruster configuration." *Journal of Physics D: Applied Physics* 56, 175201 (2023). doi:10.1088/1361-6463/acbb15
113. Ricketson, L. F.; Cerfon, A. J. "Sparse grid techniques for particle-in-cell schemes." *Plasma Physics and Controlled Fusion* 59, 024002 (2017; online 2016). doi:10.1088/1361-6587/59/2/024002
114. Ricketson, L. F.; Chacon, L. "An energy-conserving and asymptotic-preserving charged-particle orbit implicit time integrator for arbitrary electromagnetic fields." *Journal of Computational Physics* 418, 109639 (2020). doi:10.1016/j.jcp.2020.109639
115. Ricketson, Lee F.; Hu, Jingwei. "An explicit, energy-conserving particle-in-cell scheme." *Journal of Computational Physics* 537, 114098 (2025). doi:10.1016/j.jcp.2025.114098
116. Samaddar, D.; Newman, D. E.; Sanchez, R. "Parallelization in time of numerical simulations of fully-developed plasma turbulence using the parareal algorithm." *Journal of Computational Physics* 229, 6558-6573 (2010). doi:10.1016/j.jcp.2010.05.012
117. Savard, N.; Fubiani, G.; Eremin, D.; Dehnel, M. "Impact of particle number and cell size in fully implicit charge- and energy-conserving particle-in-cell schemes." *Physics of Plasmas* 32, 073903 (2025). doi:10.1063/5.0265414
118. Squire, J.; Qin, H.; Tang, W. M. "Geometric integration of the Vlasov-Maxwell system with a variational particle-in-cell scheme." *Physics of Plasmas* 19, 084501 (2012). doi:10.1063/1.4742985
119. Stantchev, George; Dorland, William; Gumerov, Nail. "Fast parallel Particle-To-Grid interpolation for plasma PIC simulations on the GPU." *Journal of Parallel and Distributed Computing* 68, 1339-1349 (2008). doi:10.1016/j.jpdc.2008.05.009
120. Stuben, K. "A review of algebraic multigrid." *Journal of Computational and Applied Mathematics* 128, 281-309 (2001). doi:10.1016/S0377-0427(00)00516-1
121. Sun, Haomin; Banerjee, Soham; Sharma, Sarveshwar; Powis, Andrew Tasman; Khrabrov, Alexander V.; Sydorenko, Dmytro; Chen, Jian; Kaganovich, Igor D. "Direct implicit and explicit energy-conserving particle-in-cell methods for modeling of capacitively coupled plasma devices." *Physics of Plasmas* 30, 103509 (2023). doi:10.1063/5.0160853
122. Swarztrauber, Paul N. "The Methods of Cyclic Reduction, Fourier Analysis and the FACR Algorithm for the Discrete Solution of Poisson's Equation on a Rectangle." *SIAM Review* 19, 490-501 (1977). doi:10.1137/1019071
123. Sydora, R. D. "Low-noise electromagnetic and relativistic particle-in-cell plasma simulation models." *Journal of Computational and Applied Mathematics* 109, 243-259 (1999). doi:10.1016/S0377-0427(99)00161-2
124. Szabo, James; Warner, Noah; Martinez-Sanchez, Manuel; Batishchev, Oleg. "Full Particle-In-Cell Simulation Methodology for Axisymmetric Hall Effect Thrusters." *Journal of Propulsion and Power* 30, 197-208 (2014). doi:10.2514/1.B34774
125. Taccogna, Francesco; Longo, Savino; Capitelli, Mario; Schneider, Ralf. "Self-similarity in Hall plasma discharges: Applications to particle models." *Physics of Plasmas* 12, 053502 (2005). doi:10.1063/1.1877517
126. Taccogna, F.; Longo, S.; Capitelli, M.; Schneider, R. "Particle-in-Cell Simulation of Stationary Plasma Thruster." *Contributions to Plasma Physics* 47, 635-656 (2007). doi:10.1002/ctpp.200710074
127. Taccogna, F.; Schneider, R.; Longo, S.; Capitelli, M. "Kinetic simulations of a plasma thruster." *Plasma Sources Science and Technology* 17, 024003 (2008). doi:10.1088/0963-0252/17/2/024003
128. Taccogna, Francesco; Minelli, Pierpaolo. "Three-dimensional particle-in-cell model of Hall thruster: The discharge channel." *Physics of Plasmas* 25, 061208 (2018). doi:10.1063/1.5023482
129. Taccogna, F.; Cichocki, F.; Eremin, D.; Fubiani, G.; Garrigues, L. "Plasma propulsion modeling with particle-based algorithms." *Journal of Applied Physics* 134, 150901 (2023). doi:10.1063/5.0153862 (preprint title "Plasma propulsion simulation using particles", arXiv:2304.05103)
130. Taitano, William T.; Knoll, Dana A.; Chacon, Luis; Chen, Guangye. "Development of a Consistent and Stable Fully Implicit Moment Method for Vlasov-Ampere Particle in Cell (PIC) System." *SIAM Journal on Scientific Computing* 35, S126-S149 (2013). doi:10.1137/120881385
131. Teunissen, Jannis; Ebert, Ute. "Controlling the weights of simulation particles: adaptive particle management using k-d trees." *Journal of Computational Physics* 259, 318-330 (2014). doi:10.1016/j.jcp.2013.12.005
132. Turner, M. M. "Kinetic properties of particle-in-cell simulations compromised by Monte Carlo collisions." *Physics of Plasmas* 13, 033506 (2006). doi:10.1063/1.2169752
133. Turner, M. M.; Derzsi, A.; Donko, Z.; Eremin, D.; Kelly, S. J.; Lafleur, T.; Mussenbrock, T. "Simulation benchmarks for low-pressure plasmas: Capacitive discharges." *Physics of Plasmas* 20, 013507 (2013). doi:10.1063/1.4775084
134. Turner, M. M. "Verification of particle-in-cell simulations with Monte Carlo collisions." *Plasma Sources Science and Technology* 25, 054007 (2016). doi:10.1088/0963-0252/25/5/054007
135. Ueda, Hiroko; Omura, Yoshiharu; Matsumoto, Hiroshi; Okuzawa, Takashi. "A study of the numerical heating in electrostatic particle simulations." *Computer Physics Communications* 79, 249-259 (1994). doi:10.1016/0010-4655(94)90071-X
136. Vahedi, V.; Surendra, M. "A Monte Carlo collision model for the particle-in-cell method: applications to argon and oxygen discharges." *Computer Physics Communications* 87, 179-198 (1995). doi:10.1016/0010-4655(94)00171-W
137. Vay, J.-L.; Geddes, C. G. R.; Cormier-Michel, E.; Grote, D. P. "Numerical methods for instability mitigation in the modeling of laser wakefield accelerators in a Lorentz-boosted frame." *Journal of Computational Physics* 230, 5908-5929 (2011). doi:10.1016/j.jcp.2011.04.003
138. Vay, J.-L.; Grote, D. P.; Cohen, R. H.; Friedman, A. "Novel methods in the Particle-In-Cell accelerator Code-Framework Warp." *Computational Science & Discovery* 5, 014019 (2012). doi:10.1088/1749-4699/5/1/014019
139. Vay, J.-L.; Almgren, A.; Bell, J.; Ge, L.; Grote, D. P.; Hogan, M.; Kononenko, O.; Lehe, R.; Myers, A.; Ng, C.; Park, J.; Ryne, R.; Shapoval, O.; Thevenet, M.; Zhang, W. "Warp-X: A new exascale computing platform for beam-plasma simulations." *Nuclear Instruments and Methods in Physics Research Section A* 909, 476-479 (2018). doi:10.1016/j.nima.2018.01.035
140. Villafana, W.; Petronio, F.; Denig, A. C.; Jimenez, M. J.; Eremin, D.; Garrigues, L.; Taccogna, F.; Alvarez-Laguna, A.; Boeuf, J. P.; Bourdon, A.; Chabert, P.; Charoy, T.; Cuenot, B.; Hara, K.; Pechereau, F.; Smolyakov, A.; Sydorenko, D.; Tavant, A.; Vermorel, O. "2D radial-azimuthal particle-in-cell benchmark for E x B discharges." *Plasma Sources Science and Technology* 30, 075002 (2021). doi:10.1088/1361-6595/ac0a4a
141. Villasenor, John; Buneman, Oscar. "Rigorous charge conservation for local electromagnetic field solvers." *Computer Physics Communications* 69, 306-316 (1992). doi:10.1016/0010-4655(92)90169-Y
142. Vranic, M.; Grismayer, T.; Martins, J. L.; Fonseca, R. A.; Silva, L. O. "Particle merging algorithm for PIC codes." *Computer Physics Communications* 191, 65-73 (2015). doi:10.1016/j.cpc.2015.01.020
143. Welch, D. R.; Genoni, T. C.; Clark, R. E.; Rose, D. V. "Adaptive particle management in a particle-in-cell code." *Journal of Computational Physics* 227, 143-155 (2007). doi:10.1016/j.jcp.2007.07.015
144. Williams, Samuel; Waterman, Andrew; Patterson, David. "Roofline: an insightful visual performance model for multicore architectures." *Communications of the ACM* 52, 65-76 (2009). doi:10.1145/1498765.1498785
145. Yuan, Tiannan; Ren, Junxue; Zhou, Jun; Zhang, Zhe; Wang, Yibai; Tang, Haibin. "The effects of numerical acceleration techniques on PIC-MCC simulations of ion thrusters." *AIP Advances* 10, 045115 (2020). doi:10.1063/1.5113561
146. Zhang, Yao; Cohen, Jonathan; Owens, John D. "Fast tridiagonal solvers on the GPU." *Proceedings of the 15th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '10)*, 127-136 (2010). doi:10.1145/1693453.1693472
147. Zhang, Weiqun; Almgren, Ann; Beckner, Vince; Bell, John; Blaschke, Johannes; Chan, Cy; Day, Marcus; Friesen, Brian; Gott, Kevin; Graves, Daniel; Katz, Max; Myers, Andrew; Nguyen, Tan; Nonaka, Andrew; Rosso, Michele; Williams, Samuel; Zingale, Michael. "AMReX: a framework for block-structured adaptive mesh refinement." *Journal of Open Source Software* 4, 1370 (2019). doi:10.21105/joss.01370

Grey literature referred to in the text and not counted: NVIDIA, "Getting Started with CUDA
Graphs" (developer blog, 2019) and the CUDA Multi-Process Service documentation (both already
relied on by the repository's H100 notes); Groenewald, R. et al., "New semi-implicit
electrostatic particle-in-cell method to extend scope of the exascale WarpX code", SC24 research
poster (2024), the WarpX implementation that Marks and Gorodetsky 2025 and Hedlof et al. 2026
build on; Szabo 2001 (MIT thesis, hdl:1721.1/8889, cited through `pic-mcc-blockers.md`).

### 11.1 Verification notes

- Garrigues, Fubiani and Boeuf 2017 and Ricketson and Cerfon 2017 carry a 2016 issue date in
  Crossref (online first) and a 2017 volume; the print citation is used.
- Fubiani, Garrigues and Boeuf 2018 (*Physics of Plasmas* 25, 023510) is the "scaling laws" paper;
  a first search suggested 2017, the Crossref record says 2018.
- Kahnfeld et al. 2016 is deposited with two typographical errors in the title; the corrected
  reading is used and the deposited form noted (as in `pic-mcc-blockers.md`).
- Barnes 2021 (*JCP* 424, 109852) is cited for its appendix only, on the authority of Marks and
  Gorodetsky 2025's reference 23; the appendix itself was not read.
- Marks and Gorodetsky 2025 reference 24 (Groenewald et al.) is an SC24 poster without a DOI;
  it is listed as grey literature and the "two orders of magnitude" Penning-discharge speed-up
  attributed to it is quoted from Marks and Gorodetsky 2025, not from the poster.
- Hedlof et al. 2026 has a 2026-05 issue date; it was verified on the Crossref record and the
  AIP abstract page.
- Parodi, Lapenta and Magin 2022 is not in Crossref; the DOI resolves through doi.org to the
  EUCASS proceedings PDF (title, authors and affiliations confirmed there).
- Liu W. et al. 2023 is cited only for the sentence that it adopts gamma = 10^2 on the strength
  of Yuan et al. 2020; the rest of the paper was not read.
- Dropped for want of a resolvable record: Barnes, Kamimura, Leboeuf and Tajima 1983 (implicit
  particle simulation of magnetized plasmas, *JCP* 52, 480); Liu et al. 2010 (*J. Phys. D* 43,
  165202, resolved but no abstract deposited and the full text not read, so no claim rests on
  it); a Taccogna 2008 *CPP* start-up-transient paper whose DOI resolved to a different title;
  Denavit and Walsh 1981 (quiet starts, no DOI).
- The count in the heading is the number of numbered entries (147).
