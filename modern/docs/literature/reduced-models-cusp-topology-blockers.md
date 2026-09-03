# Reduced-order model, cusp-loss and topology blockers: literature review

Status: literature review (no code, spec or experiment changed). Date: 2026-09-03.
Branch `docs/literature-models`. Companion to `docs/AUDIT.md`, `docs/REFERENCES.md`,
`docs/workstreams/global-plasma-closure-analysis.md`,
`spec/plasma/equation-ledger.json#global_row_consistency`, and the recorded
experiments `four_cell_topology_search_v2`, `cft_topology_characterization_v1`,
`cft_orbit_wall_loss_v4`, `orbit_wall_loss_geometry_screening_v1`,
`mdo_l0_campaign_v1` / `_v2`.

## 0. Scope, method and verification policy

Four blockers were reviewed against the published literature on HEMP / cusped-field
thrusters (Thales/DLR, Greifswald, MIT DCFT, Princeton CHT, HIT), on plasma
confinement in magnetic cusps, on sheath closures for reduced models, and on
magnetic-null topology. Every reference in Section 7 was verified on 2026-09-03 by
opening its Crossref DOI record, DSpace handle, arXiv page or the ERPS (IEPC)
PDF; the tag after each entry records the depth of verification:

- `[T]` full text retrieved and read for the statements attributed to it;
- `[R]` bibliographic record (title, authors, venue, year, DOI) verified; the
  statements attributed to it are limited to its title/abstract or to what other
  `[T]` sources say about it;
- `[C]` IEPC paper without a DOI, verified from the ERPS PDF header or from the
  reference lists of `[T]` sources.

References the task named but which could not be verified are listed in
Section 6 and are *not* cited. One supplied identifier resolves to a different
paper than described: DOI `10.1016/j.ast.2024.109516` is Yeo, Gadisa, Ogawa and
Bang, "Multi-objective design optimization and physics-based sensitivity
analysis of field emission electric propulsion for CubeSat platforms",
*Aerospace Science and Technology* 154 (2024) 109516 - a FEEP paper, not a
cusped-field paper [73]. A Crossref bibliographic search for a 2024 *AST*
cusped-field optimisation paper by Ma et al. returned nothing; the actual
downstream lineage of the original ISTS study is Fahey, Muffatti and Ogawa 2017
[6], Yeo et al. 2020 [7], Yeo and Ogawa 2022 [8] and, independently, Puca,
Panelli and Battista 2024 [4].

Notation follows the ledger: cells k = 1..4, cusp probabilities p_k, cell
potentials phi_k, cusp potentials phi_ck, temperatures T_k (eV), electron
currents j_ek, ion currents j_ik, source currents I_k, ionisation energy EI.

---

## 1. Blocker 1 - the four-cell power balance has no root for interior p > 0

### 1.1 Our finding (restated)

On the R00-R26 manifold the global row reduces exactly to
`R27 = 2 (j_e3 (1-p4) + I4)(phi_4 - Ua) + EI (p1 j_e0 + p2 j_e1 + p3 j_e2)`
(`global_row_closed_form`, 1.9e-13 relative), which is non-negative on the
admissible region and vanishes only for p1 = p2 = p3 = 0 with phi_4 = Ua. The
ledger attributes the two terms to (i) Kornfeld's assumption 8 (ionisation booked
as a frozen loss *and* again as cusp recombination) and (ii) the printed anode
electron term `(phi_4 - Ua + T4)` carrying an ion's sign. The proposed correction
(drop `+EI` from `Pcusp`; anode electron term `(Ua - phi_4 + T4)`) is recorded as
`PROPOSED_NOT_ACCEPTED` because it lowers the structural rank from 22 to 21 and
frees all four potentials.

### 1.2 What the sources actually print

**Kornfeld, Koch and Harmann, IEPC-2007-108 [1] `[T]`.** Section III.A lists twelve
assumptions. Assumption 2: "Upstream of each cell, electrons can hit at the cusps
the dielectric material of the channel wall which requests an identical ion
current for charge compensation. This condition determines selfconsistantly
adjusting cusp potentials phi_c1, phi_c2, phi_c3." Assumption 5: "At each cusp the
sum of electron and ion current should be 0 due to the dielectric boundary
condition." Assumption 8 (verbatim): "Ionization and excitation losses are
considered as frozen losses not contributing to the thermal load of the
thruster, except the recombination losses at boundaries." Assumption 12: the
electric power `Ua.Ia` "has to equal the ion beam power Pb, the frozen losses and
all thermal dissipations at anode and the respective magnetic cusps." The
global balance is written as `Ua.Ja = Pb + IL + EL + CL (electronic-, ionic- and
recombination) + AL (electronic- ionic and recombination)`, with `IL = IE (I1 + I2
+ I3 + I4)` and `CL = je0 p1 (phi_c1 - phi_0 + phi_1 - phi_c1 + IE) + ...`. The
28-unknown list includes "3 cusp potentials phi_c1 to phi_c3 at the ceramic
surface", but the cusp potentials cancel identically inside `CL` and inside the
cell rows, so no printed equation identifies them - the "zero net current"
condition (assumption 5) is used only to set `jic_k = p_k je_{k-1}`. Section III.C
states that MathCAD's root finder ("suchen") "did mostly not work for the 28
equation problem. Therefore the 'minfehl' command was used seeking for solution
with minimum failure. The power accuracy of the solution was in this case always
within 0.5%." Table 3.1 (DM9.2, 4-stage, 1 kV / 1 A) prints phi_1 = 14.1 V,
phi_2 = phi_3 = phi_4 = 1000 V, T2 = 100.1 eV, T3 = 43.1 eV, T4 = 23.5 eV, and
p = (0.06, 0.119, 0.160, 0.254). The paper also states that the cusp probabilities
are "only small fractions of ~5% to 10% of the electrons" from the mirror
formula (Section II.4, figure 2.2), and its point 6 on Table 3.1 attributes the
low cusp losses to "the small voltage difference between cusp potential and its
respective plasma potential in the order of 20 [V]" - a statement about a
quantity the printed system does not determine.

Two facts follow directly from the source and support the ledger's reading:
the published DM9.2 state sits on the `phi_4 = Ua` boundary that our closed form
predicts, and the published solution is a minimum-error solution, not a root.

**Puca, Panelli and Battista 2024 [4] `[T]`** is the only published independent
re-implementation of the Kornfeld system found. They keep the 28 rows, add
p_1..p_4 as unknowns with three of them tied by the electron-current definition
they attribute to Matthias et al. 2019 [10], `p_j = 1 - (I_ej - I_j)/I_ej+1` for
j = 1..3 (their eq. 14), convert the cathode current to an input to balance the
count, and solve a 33-equation system by minimising a weighted sum
of squared residuals (their eq. 15) with a genetic algorithm and with MATLAB
`fmincon`. Their Table 1 (DM9.2 reproduction) prints phi_2 = 979.15 V, phi_3 =
998.90 V, phi_4 = 1000 V (again pinned at Ua), I1 = -0.0644 A (a negative
ionisation source current), and p = (0.49, 0.63, 0.25, 6.1e-13) against
Kornfeld's (0.06, 0.119, 0.160, 0.254); the authors conclude that "the proposed
analytic pathway for the estimation of p_j in Eq. (12) cannot be adopted as a
trustworthy solution." Their Table 2 shows the GA solutions insensitive to
operating point and `fmincon` chosen for that reason. They do not identify the
double-booking or the anode sign, and they close the missing potential/field
information not inside the 28-equation system but by bolting on a Goebel-type
ring-cusp discharge model (their Table 3): primaries lost to the cusp through
`A_p = 2 r_p L_c`, plasma electrons through the hybrid area `A_a = sqrt(r_e r_i)
L_c` with a Boltzmann factor `exp(-q phi_s / kT_e)`, ions between cusps through
`(1/2) n_i q sqrt(kT_e/M) A_as f_c`, with the sheath potential phi_s taken from the
Hall-thruster chapter of Goebel and Katz [57] and the electron energy lost to the
wall `2 kT_e/q + phi_s` (their text after eq. 18). This is exactly the class of
closure our ledger says is missing.

**Matthias et al. 2019 [10] `[R]`** (abstract): "Several optimized thruster designs
have resulted from a MDO model based on a zero-dimensional (0D) power balance
model. However, the MDO solutions do not warrant self-consistency due to their
dependency on estimation from empirical modelling ... The 0D power balance model
is used to develop additional diagnostics for the PIC simulations ... Using input
parameters for the 0D power balance model from the PIC simulations allows further
improvement for the design optimization." This is the published precedent for
feeding kinetic (PIC) electron currents into the 0-D model rather than the
mirror formula.

**Yeo et al. 2020 [7] `[T]`** kept the 28 Kornfeld equations with mirror-formula
p_k from ANSYS Maxwell fields (their eqs. 7-8) and found, for the selected design
S1, `MDO (original)` I_b = 3.61 A, T = 102.7 mN, eta_t = 36.5 %, I_sp = 2131 s
against PIC 2.30 A, 62.8 mN, 15.2 %, 1333 s (their Table 7: 36-58 % differences),
then re-anchored the mass-utilisation and divergence sub-models on PIC output
(`MDO (modified)`). The power-balance system itself was not corrected.

**Reduced/global models outside HEMP.** Goebel, Wirz and Katz 2007 [54] `[R]` and
Goebel and Katz 2008 [57] `[R]` (ring-cusp ion thruster discharge model) book the
ionisation energy once, as `I_p U^+`, and the electron energy to the anode/wall as
`2 T_e + phi_s`; cusp electron loss is a thermal flux through a hybrid-gyroradius
area reduced by the sheath Boltzmann factor. Wirz and Goebel 2008 [55] `[R]` extend
this to field-topography-dependent performance with a 2-D hybrid model. Lieberman
and Lichtenberg 2005 [58] `[R]` (global discharge model): the absorbed power is
`e n u_B A_eff (E_c + E_e + E_i)` where the collisional energy per ion `E_c`
(ionisation + excitation + elastic) appears once and the wall energy `E_e + E_i`
is set by the sheath; the electron temperature is fixed by particle balance and
the density by power balance - i.e. the potentials are closed by Bohm-flux and
sheath relations, not left free. Ahedo, Martinez-Cerezo and Martinez-Sanchez 2001
[62] `[R]`, Ahedo 2002 [61] `[R]`, Keidar, Boyd and Beilis 2001 [64] `[R]` and
Barral et al. 2003 [63] `[R]` give the dielectric-wall sheath/presheath closures
(with and without secondary emission) used by 1-D Hall-thruster models; Hobbs and
Wesson 1967 [59] `[R]` is the classical floating-sheath-with-emission relation and
Riemann 1991 [60] `[R]` the Bohm criterion. In none of these does a wall
recombination term reappear beside a "frozen" ionisation loss, and in all of them
an electron arriving at an electrode/wall at a lower potential than the plasma
deposits `T + (phi_plasma - phi_wall)` (the sign our proposal restores).

**MIT DCFT and Princeton CHT.** No 0-D power-balance model of the Kornfeld type
was found for the DCFT or the CHT. Courtney, Lozano and Martinez-Sanchez 2008
[24] `[R]`, Courtney 2008 [25] `[R]`, Gildea et al. 2013 [28] `[R]`, Matlock 2012
[29] `[R]` and Gildea 2012 [27] `[T]` treat the DCFT experimentally and with
fluid/PIC models; Gildea 2012 (Sec. 1.2) reviews hybrid, fluid and PIC Hall-thruster
models and argues from Knudsen-number estimates that fluid closures are invalid in
the DCFT exit region. Raitses and Fisch 2001 [31] `[R]` and Smirnov, Raitses and
Fisch 2004 [32] `[R]` characterise the CHT with probes and efficiency budgets; no
multi-cell global model is published there. "Princeton CHT global models" and
"MIT DCFT global models" in the task brief therefore have no verified referent.

### 1.3 State of the art, as far as reduced HEMP models go

1. Nobody has published a root of the 28-equation Kornfeld system. The source
   [1] used a minimum-error solver ("power accuracy within 0.5 %"); the legacy
   MATLAB used `lsqnonlin` with flags accepted by status (`AUDIT.md` C7); Puca et
   al. [4] used GA/`fmincon` on a weighted sum of squares and obtained unphysical
   p_j and a negative source current. All three land on `phi_4 = Ua`.
2. The models that do close (ion-thruster and Hall-thruster reduced models
   [54,55,57,58,61,62]) do so by adding *sheath* rows (Bohm ion flux, Boltzmann
   electron flux, floating-wall zero-net-current, electron energy `2T_e + phi_s`)
   and a *density/ionisation* balance; the potentials are then identified.
3. Where potentials are not modelled, the field uses PIC to supply them
   (Matthias 2019 [10]; Yeo 2020 [7] for eta_m and divergence).

### 1.4 Pitfalls documented in the literature

- Rank deficiency with free potentials: the cusp potentials cancel in [1] and
  are "determined selfconsistantly" only in words; Puca [4] had to add unknowns
  and an input to balance the count. Structural identifiability is a property of
  the equation set, not of the solver (Bellman and Astrom 1970 [72]).
- Minimum-error "solutions" hide inconsistency: [1] and [4] both report weighted
  least-squares minima as states; our residual-floor ladder shows the floor is
  linear in p.
- Double counting energy sinks: [1] assumption 8 books recombination at
  boundaries as an extra thermal load while ionisation is also a frozen loss.
  Global models [58] book each joule once.
- Cusp probabilities from an electron-current identity ([10] eq. as used in [4]
  eq. 14) are *definitions* of a transmitted fraction, not independent physics;
  inserting them as extra equations does not add information about the wall.

### 1.5 Recommendation mapped to our ledger

- Accept the two `PROPOSED_NOT_ACCEPTED` corrections in
  `spec/plasma/equation-ledger.json#global_row_consistency` (they align with
  [54,57,58]: ionisation booked once; electron deposits `T + (phi_plasma -
  phi_electrode)`), **only together with** closure rows:
  - R28-R30 (one per dielectric cusp k = 1..3): floating-wall sheath with the
    zero-net-current condition already used for `jic_k`:
    `phi_k - phi_ck = T_k ln[ Gamma_e,th,k / Gamma_i,k ]` with `Gamma_i,k` the Bohm
    flux and, if SEE is declared, the Hobbs-Wesson [59] emission-corrected form;
    electron energy to the cusp `2 T_k + (phi_k - phi_ck)`, ion energy `(phi_k -
    phi_ck) + T_k/2` (presheath) [57,58,62]. This re-introduces phi_c1..phi_c3 as
    unknowns *with* their own rows, so the count stays balanced and the cusp
    energy terms stop depending on an unidentified potential.
  - R31 (anode): an electron-collecting anode sheath row, e.g. `j_e4 =
    (1/4) n_e4 v_e,th A_anode exp(-(phi_4 - Ua)/T4)` for `phi_4 >= Ua`, or the
    declared alternative `phi_4 - Ua = Delta_a` with `Delta_a` a documented anode
    fall [57]. Either identifies phi_4.
  - The remaining cell-to-cell potential differences (phi_2 - phi_1, phi_3 - phi_2)
    are *transport* quantities. The literature closes them by an electron
    conductivity across each cusp (Greifswald PIC uses an anomalous `D_perp = 0.4
    k T_e / eB`, Brandt et al. 2016 [18]) or by PIC input [10]. Until such a row
    exists, declare a closure `CL-3-potentials`: flat interior potential and one
    exit drop, which is what Koch et al. 2011 [2] state as finding (ii) ("The
    plasma potential inside the discharge channel is almost constant and close to
    the applied anode voltage, whilst the confined electron cloud at the thruster
    exit cusp leads to a sharp drop of almost the entire plasma potential") and
    what Brandt 2016 [18] simulate for a 400 V micro-HEMPT ("At the first
    magnetic cusp, the potential undergoes a drop of about 10 V"; main drop at the
    exit cusp).
- Because a density now enters (Bohm flux), add the neutral/ionisation balance
  the ledger lists as missing ("No neutral density or mass-flow balance") or
  declare `n_e` per cell as an input from `pic2d` (our steady-state v2 plateau
  gives per-cell densities at 300 V). Record which.
- Verification target for the corrected + closed system: reproduce Kornfeld
  Table 3.1 DM9.2 at 1 kV / 1 A (phi_1 = 14.1 V, phi_2..4 ~ 1000 V, T2 = 100.1
  eV, T3 = 43.1 eV, T4 = 23.5 eV) *as a root*, and report the residual of the
  published p-vector under the new rows; then the Puca DM9.2 column (Table 1 of
  [4]) as a second reference. Neither is truth; both are the only published
  states of this model.
- Update `docs/REFERENCES.md`: the Kornfeld model has one published independent
  re-solve [4] and one PIC-anchored correction path [7,10]; none produced a root.

---

## 2. Blocker 2 - loss-cone probability versus collisionless wall-hit

### 2.1 Our finding (restated)

The legacy chain (`FYP/cusp_prob.m:186-190`) derives p_k from the mirror
acceptance angle `theta_m = asin(sqrt(B_0/B_M))` and an isotropic angular
integral (Kornfeld [1] Sec. II.4; ISTS 2017-b-32 eqs. 14-17). The accepted v4
campaign (P2 divergent-exit field, 4608 orbits) recorded 0 reflections and a
wall-hit fraction 0.641-0.645; the 96-design L1a screening recorded reflections in
every design (22 % of orbits) and wall-hit 0.375-0.869. Feeding wall-hit into CL-1
gives survival ~7e-8 (no beam). The magnetic-moment variation diagnostic was
median 0.14, max 0.63 (v4).

### 2.2 What the literature says

**Cusp leak width and confinement.** Hershkowitz, Leung and Romesser 1975 [44]
`[R]`, Leung, Hershkowitz and MacKenzie 1976 [45] `[R]`, Pechacek et al. 1980 [47]
`[R]` and Bosch and Merlino 1986 [48] `[R]` are the experimental basis for the
statement that plasma leaks through a low-beta line/ring/spindle cusp over a
width of order the hybrid gyroradius `sqrt(r_e r_i)` rather than the electron
gyroradius; Haines 1977 [50] `[R]` reviews cusp containment theory. Hubble and
Foster 2008 [53] `[R]` measured the collection width in a 10 cm ring-cusp
discharge chamber. Puca et al. [4] `[T]` summarise the state of this question for
HEMP design: Hershkowitz et al. "were among the firsts to find these in the order
of 4 times the hybrid gyroradius. However, to know how the leak width scales is a
very questionable topic, and a unique answer does not exist"; they adopt a
prefactor of 1 for HEMP. Two effects modify the pure-geometry picture:
Hershkowitz, Smith and Kozima 1979 [46] `[R]` ("Electrostatic self-plugging of a
picket fence cusped magnetic field") - the ambipolar/sheath potential plugs the
cusp; and Knorr and Merlino 1984 [49] `[R]` ("The role of fast electrons for the
confinement of plasma by magnetic cusps") - the leak width depends on the fast
electron population, which [4] uses to argue for a smaller prefactor at HEMP
energies. The ion-thruster reduced models [54,55,56] use exactly this closure:
electron loss to a cusp = thermal flux x hybrid area x `exp(-phi_s/T_e)`.

**Non-adiabatic transit.** Dunnett 1969 [51] `[R]` ("Single-particle motion in
multiple-cusp magnetic fields") and Cohen, Rowlands and Foote 1978 [52] `[R]`
("Nonadiabaticity in mirror machines") establish that the first adiabatic
invariant is not conserved when the gyroradius is comparable to the field scale
length, so a loss-cone formula (which presumes mu conservation between the
low-field point and the cusp) is not applicable there. Our v4 diagnostic
(median mu variation 0.14, 60 % of orbits above 0.1) is a direct measurement of
this regime in the P2 field. The HEMP source itself uses the loss cone only for
"electrons ... within the acceptance angle ... (without considering further
collisions)" [1].

**How HEMP PIC papers describe electron loss at cusps.** Koch et al. 2011 [2]
`[T]`: "only few electrons are lost to the wall mostly at the cusps according
their loss cone angle" and "Only a few of them overcome the magnetic barrier due
to collisions or anomalous transport induced by electrostatic turbulence".
Brandt et al. 2016 [18] `[T]` (PIC of a down-scaled HEMPT at 400 V): "Only in the
cusp regions electrons are directed towards the dielectric channel wall and
produce pronounced maxima of the particle fluxes hitting the surface. In the
other regions radial transport has to overcome the magnetic confinement. This is
only possible by collisons or anomalous turbulent transport"; they apply an
anomalous diffusion `D_perp = 0.4 k_B T_e / eB` by rotating particle velocity
vectors; the dielectric potential is set self-consistently by surface charge
(Poisson with a dielectric constant, no explicit boundary condition), and the
first cusp shows "a drop of about 10 V". Lewerentz and Schneider 2023 [19] `[T]`
show DM3a PIC electron density with "density sinks" at the cusps "because the
plasma-wall interaction is concentrated at these locations" and use the field-line
mirror ratio `B_max/B_0` with the reflection condition `sin(theta) >= 1/sqrt(R_m)`
only as a *figure of merit*, not as a loss probability. Matyash et al. 2010 [13]
`[R]`, Kalentev et al. 2014 [14] `[R]`, Schneider et al. 2009 [15] `[R]`, Duras et
al. 2017 [16] `[R]`, Kahnfeld et al. 2018 [17] `[R]`, Kahnfeld et al. 2019 [12]
`[R]` and Lewerentz et al. 2022 [20] `[R]` are the PIC-MCC lineage; their full
texts were not retrievable here, so no figure numbers are cited from them (see
Section 6). HIT PIC: Zhao et al. 2014 [33] `[R]` (field strength), Liu et al.
2015 [35] `[R]` ("Particle-in-cell simulation for different magnetic mirror
effects on the plasma distribution in a cusped field thruster") and Liu et al.
2014 [34] `[R]` (electric-field formation in a multi-cusped field).

**Does any published model use collisionless test-particle wall-hit as a closure
input?** None was found. The closest analogues are: (a) the mirror-formula p_k
[1,6,7,8], which is a collisionless *geometric* access fraction with mu assumed
conserved; (b) PIC-derived electron currents inserted into the 0-D model
[10, 4]; (c) hybrid-gyroradius leak-width closures with sheath factors [54,55,4].
Our screening quantity - the fraction of launched collisionless electrons whose
orbit intersects the dielectric before leaving the domain - is a third,
different object: it is an upper bound on the sheath-free geometric access and
carries no sheath, collision or anomalous-transport physics.

### 2.3 Pitfalls

- Using a loss-cone formula where mu is not conserved [51,52]; the v4 mu
  diagnostic shows this is the regime for the P2 field.
- Ignoring the dielectric sheath barrier: the floating dielectric repels most of
  the electrons within the loss cone; the Boltzmann factor `exp(-Delta phi_s/T_e)`
  at `Delta phi_s ~ 5 T_e` (Xe, no SEE) is ~7e-3 [58,59]. A wall-hit fraction of
  0.64 is therefore not a loss fraction of any real population.
- Treating a collisionless single-pass quantity as a per-transit probability
  in a cascade (`prod (1-p_k)`): collisions/anomalous transport repopulate the
  loss cone [2,18], and reflections (present in every L1a design) mean many passes.
- Pooling equal-weight launch strata into one "probability": v4's per-cell
  bimodality shows the pooled 0.64 is a design average.
- Leak-width prefactor: experiments disagree by factors of a few [44,47,48,53];
  [4] chooses 1 by argument, not measurement.

### 2.4 Recommendation mapped to our experiments and spec

- Relabel the estimand of `cft_orbit_wall_loss_v4` and
  `orbit_wall_loss_geometry_screening_v1` in every consumer as
  `collisionless_geometric_wall_access_fraction` (upper bound on sheath-free
  access), never `cusp_probability`; the coupling export field
  `electron_dielectric_wall_loss_probability` should carry that label in its
  schema description at the next coupling revision.
- Define a declared closure `CL-3-sheath-limited` for the MDO chain:
  `p_k,eff = A_k * exp(-(phi_k - phi_ck)/T_k)`, where `A_k` is the geometric access
  fraction from the screening (per cell, with its Wilson interval) and
  `phi_k - phi_ck` comes from the floating-dielectric row of Section 1.5 (with an
  SEE yield declared or set to zero). Report CL-1, CL-2 and CL-3 fronts and their
  overlap, as v2 already does for CL-1/CL-2.
- Add a leak-width closure `CL-4-hybrid-area` as the ion-thruster alternative
  [54,55]: electron loss current to cusp k = `(1/4) n_e,k v_e,th,k * (c sqrt(r_e r_i)
  * 2 pi r_w) * exp(-Delta phi_s/T_k)` with the prefactor `c` swept over [1, 4]
  (the documented disagreement range [4,44]). This needs `n_e,k`; take it from
  `pic2d` steady-state v2 or the neutral balance of Section 1.5.
- Calibration/comparison targets that exist: our own `pic2d_cft_steady_state_v2`
  plateau at 300 V (wall ion flux peaking at the 12.2 mm cusp; electron wall
  current equal to ion wall current, 3.7 mA against I_d 3.44 mA per the pic2d
  devlog) is the only kinetic result on *our* geometry; externally, Brandt et al.
  2016 [18] (400 V micro-HEMPT: wall flux maxima only at cusps; 10 V first-cusp
  potential drop) is the closest published configuration in scale. Compare the
  per-cusp electron flux fraction of `pic2d` against `A_k exp(-Delta phi_s/T_k)`
  before any MDO uses CL-3.
- Keep mu variation as a diagnostic (as v4 does) and publish it beside every
  screening number so that readers can see where the loss-cone formula is
  inapplicable [51,52].

---

## 3. Blocker 3 - the frozen wall-cusp / cell definition finds nothing

### 3.1 Our finding (restated)

`four_cell_topology_search_v2/protocol.json#topology` and the characterization
v1 eligibility rules require an interior *vector null* of the accepted psi map
with converged X-type Jacobian and winding index, geometry-registered near the
wall (`geometry_cusp_slots` = stage midplanes), an offset constant-psi separatrix,
and cross-map stability. Result: 0/128 stable four-cell candidates; 0 stable
eligible cusps and 0 stable eligible cells over 56 characterised designs; all
channel-interior roots were X-type and none established a cell-bounding
separatrix; axis cusps (3-5 per design, sign changes of B_z on axis) exist in
every accepted sweep-v2 design.

### 3.2 How the literature defines cusps and cells

**DCFT (Gildea 2012 [27] `[T]`, Sec. 1.1):** "permanent ring-magnets of
alternating polarity down stream of a magnetic pole piece give rise to several
regions containing convergent magnetic field lines, referred to as cusps. In
addition to the three ring-cusps, point-cusps are established on the thruster
axis to either side of each ring-cusp ... Field strengths along the axis exceed
4 kG near the anode and 1 kG between C1 and C2, but decrease to zero where each
ring-cusp separatrix intersects the centerline. A separatrix is a special surface
which can be thought of as the divider between different magnetic cells. More
rigourously, a separatrix is defined as a surface on which the magnetic vector
potential is zero valued." And (Sec. 1.4): "The field strength only goes to zero
where the ring cusp separatrices intersect the thruster axis of symmetry."

**HEMP (Kornfeld 2007 [1] `[T]`):** "Three magnet rings define four plasma
cells, separated by three neutral- or cusp flux lines"; "The neutral cusp flux
lines can be interpreted as magnetic 1-hole grids." Koch 2011 [2] `[T]`: the
stack is "axially magnetized permanent magnet rings in opposite magnetization,
the so-called PPM system. Typically both discharge channel and magnetic system
are tapered including variations in magnet length"; the exit cusp is "formed by
the stray field of the last downstream magnet". Lewerentz and Schneider 2023 [19]
`[T]` characterise a stage by tracing field lines from `(r_0, z = 0)` at the stage
midplane until they hit the ideal wall `r = R`, define the separatrix by its
maximum axial coordinate `z_sep` on the axis, and quantify a configuration by
field-line length, the mirror ratio `B_max/B_0` between the start point and the
wall end-point of each line, and a weighted confinement-time proxy `tau = l
sqrt(R_m) r_0 e^{-r_0}`; they derive `L/R = sqrt(6)` for a homogeneous on-axis
field and note that "a rather complex optimization procedure, finite element
methods, power balance equations, and even kinetic corrections based on
Particle-in-Cell (PIC) calculations reached a similar value". HIT (Liu et al.
2021 [41] `[T]`, review): "the shape of the last magnetic separatrix influences
the plume divergence significantly"; "a higher downstream magnet ratio leads to
a separatrix more parallel to the exit plane".

**What this means for an axisymmetric stack.** In the (r, z) half-plane of an
axisymmetric field the only generic vector nulls of an alternating ring stack lie
on the axis (where `B_r = 0` by symmetry and `B_z` changes sign) - Gildea's
"point-cusps" - plus nulls inside or between magnet bodies. The "ring cusp" at
the wall is *not* a null: there `B_z` changes sign but `|B_r|` is maximal. The
literature's cell boundary is the separatrix `psi = psi(axis null)` (Gildea:
vector potential zero-valued surface), which leaves the axis null and meets the
wall at the ring cusp. Our frozen definition searched for an X-type null at the
wall-side stage midplanes, an object the standard PPM topology does not contain,
and treated the axis nulls (which it did find, 3-5 per design) as descriptors.
The null result is therefore consistent with the literature topology rather than
evidence against multi-cell confinement.

**Null finding and classification.** Parnell et al. 1996 [65] `[R]` (Jacobian
classification of nulls), Lau and Finn 1990 [70] `[R]`, Greene 1992 [69] `[R]`
(bisection root location on grids), Haynes and Parnell 2007 [66] `[R]` (trilinear
null finding) and 2010 [67] `[R]` (skeleton/separatrix construction), Murphy,
Parnell and Haynes 2015 [68] `[R]` (appearance/disappearance of nulls under
field changes) are the standard tools; they support what our protocol already
does for axis nulls (converged Jacobian, cross-resolution correspondence) and
warn that null counts depend on resolution near degenerate configurations.

**Magnet stacks and iron.** The HEMP source hardware is a PPM stack of axially
magnetised rings [1,2]; the earliest Thales chamber had "magnet rings and pole
pieces" (figure caption in [1]); Puca et al. [4] describe "ring-shaped
separators" of "pure iron, Mu-Metal" between magnets, citing [9]; the DCFT has a
magnetic pole piece upstream of the SmCo rings [27]; the down-scaled HEMPT of
Keller et al. 2015 [21] `[R]` varied the "strength of magnetic field (via the
outer diameter of SmCo magnet from 10 to 40 mm), cusp length (1 to 10 mm), and
number of cusps (up to three)" (DLR eLib record of the IEPC precursor). The
Greifswald design tool [19] computes the field with `magpylib` (Ortner and
Coliado Bandeira 2020 [71] `[R]`), an analytic linear-magnetisation model with no
iron, and states that analytic fields are preferred for PIC because they are
divergence-free. Yeo et al. [7] used ANSYS Maxwell; Puca [4], the legacy chain
(`FYP/FEMMrun.m`) and Brandt [18] used FEMM. Conclusion: for PPM-only stacks the
linear-vacuum equivalent-current model (our L1a) is standard practice; where
soft-iron spacers or pole pieces are present (DCFT; some HEMP variants) a
nonlinear material model changes the field magnitude and the wall B_r profile
but not the existence of the axis nulls. No paper was found that requires iron
to obtain wall cusps.

### 3.3 Pitfalls

- Defining a wall cusp as a vector null: in an axisymmetric PPM stack the nulls
  are on the axis [27]; a wall-side X-point search returns nothing by
  construction.
- Registering cusps to stage midplanes: Koch [2] and Lewerentz [19] place the
  cusps at the *ends* of magnets (between stages), and the exit cusp is formed by
  the stray field beyond the last magnet.
- Grid resolution near degenerate nulls changes counts [68,69]; keep the
  cross-resolution correspondence gate, but apply it to axis nulls and to the
  wall intersection `z_c` of the separatrix.
- Linear-vacuum fields for iron-backed stacks: magnitude and gradient errors,
  hence mirror-ratio and leak-width errors, but no topology change of the kind
  our protocol tests.

### 3.4 Recommendation mapped to our experiments

- Topology search v3 (new preregistration; do not edit v2/v1 artifacts):
  redefine
  - **cusp k** := the axis null `(0, z_k)` with converged Jacobian (as now) *and*
    its separatrix `psi = psi(0, z_k)` traced to the wall intersection
    `(r_w, z_c,k)`; consistency checks: `B_z(r_w, z)` changes sign at `z_c,k`
    and `|B_r(r_w, z)|` has a prominent maximum within a declared tolerance of
    `z_c,k`;
  - **cell k** := the region between consecutive separatrices, bounded by the
    wall between `z_c,k` and `z_c,k+1`; a "four-cell" design has three
    channel-interior wall intersections plus the exit cusp;
  - **stability** := `|Delta z_c,k| <= 0.8 mm` across primary/downsampled/enlarged
    maps (reuse `maximum_cross_map_cusp_shift_m`) and unchanged axis-null count;
  - **mirror descriptor** per cell := field-line mirror ratio `B(r_w, z_c)/B(r_0,
    z_mid)` at declared `r_0` (Lewerentz [19] eq. for `R_m`), reported as a
    descriptor only.
- Re-run the characterization on the same 56 designs (cheap: fields exist) to
  test whether the accepted sweep's 3-5 axis nulls yield 2-4 wall intersections
  inside the straight channel; record the descriptive counts and the distance of
  `z_c,k` from the stage boundaries (not midplanes).
- Keep L1a for the search; run the P2 (adaptive FEM) field on the four
  representatives with and without a declared iron spacer to bound the shift in
  `z_c,k` and in the wall `|B_r|` maximum before claiming material insensitivity.
- Update `paper` Discussion wording: the null is "under a wall-null definition
  that the PPM topology does not generically satisfy", citing [27] for the
  axis-null / separatrix definition.

---

## 4. Blocker 4 - the geometry -> loss -> performance link is closure-dependent

### 4.1 Our finding (restated)

MDO v2 over the 96-design screening catalogue: the CL-1 (`prod(1-p_k)`) robust
front lives on designs 49/50/94 (lowest pooled wall-access); the CL-2
(`1 - p_pooled`) front shares no design with CL-1 (Jaccard 0); 77 of 96 designs
have ~zero survival under CL-1. The ranking is a property of the closure.

### 4.2 Chains used in the literature and their stated limits

1. **0-D power balance + magnetostatic field + mirror-formula p_k.** Muffatti and
   Ogawa ISTS 2017-b-32 (see `REFERENCES.md`), Fahey, Muffatti and Ogawa 2017 [6]
   `[R]` ("a modified power distribution calculation and evolutionary algorithms
   assisted by surrogate modeling"; five design parameters incl. magnet radii;
   "anode current and the Outer Magnet Radius have the greatest effect"), Yeo
   and Ogawa 2022 [8] `[R]`, Yeo et al. 2020 [7] `[T]`. Stated limits in [7]: xenon
   at 500-2000 V ("this study was not able to take the effects of low ionization
   into account"); singly charged ions with a 0.9 correction factor; assumed 60
   deg divergence; combined acceleration+divergence efficiency 0.407 taken from
   Keller [21]; "The cusp arrival probabilities are sensitive to the accuracy of
   the simulated magnetic topologies"; MDO(original) over-predicted S1 beam
   current by 36 % and efficiency by 58 % relative to PIC (Table 7), corrected by
   fitting eta_m and `theta_eff = 7.7 I_a + 34.7` deg from PIC. Puca et al. 2024 [4]
   `[T]` (curve-fit sizing -> 33-equation power balance -> Goebel-type cusp model
   -> FEMM iteration on magnet thickness to match cusp B) report I_sp ~ 2000 s and
   23 % efficiency for a 4 mN design and state the model neglects neutral density,
   ionisation efficiency, doubly charged ions and beam structure.
2. **Field-descriptor figures of merit without a plasma model.** Lewerentz and
   Schneider 2023 [19] `[T]`: `L/R = sqrt(6)` for on-axis homogeneity; field-line
   length, mirror ratio, confinement-time proxy, active/passive volume; the
   authors call these "simple performance indicators" for "realistic starting
   points", not performance predictions.
3. **PIC parametric studies.** Matyash et al. 2010 [13], Kalentev et al. 2014 [14],
   Kahnfeld et al. 2019 [12], Matthias et al. 2020 [11] ("Similarity
   scaling-application and limits for ... particle-in-cell modelling": PIC "is not
   suited to explore a wide operational and design space"; self-similarity scaling
   is proposed and its limits derived), Brandt et al. 2016 [18], Lewerentz et al.
   2022 [20]; HIT: Zhao et al. 2014 [33], Liu et al. 2015 [35], Liu et al. 2014
   [34]. These give trends at one or a few operating points.
4. **Experimental parametric studies (the only geometry -> performance evidence).**
   Keller et al. 2015 [21] `[R]`: micro-HEMPT with chamber diameter 2-5 mm,
   SmCo OD 10-40 mm, cusp length 1-10 mm, up to three cusps; "The minimum
   achieved thrust was 50 uN at an anode voltage of 600 V, corresponding to a
   specific impulse of 230 s. Operation points with thrusts of 180 and 360 uN
   demonstrate a specific impulse of 610 and 860 s". Ma et al. 2015 [36] `[R]`
   (variable magnet length CFT; the validation case used by [7]). Hu et al. 2016
   J. Phys. D [37] `[R]`: "increasing L_u [ultimate-stage length] could prolong the
   axial motion range of the electrons and promote the ionization process ...
   Both of these aspects can help improve the thrust and the anode efficiency ...
   the longer L_m [middle stage] leads to a reduction of the thruster's
   performance for the enhanced ion loss to the wall". Hu et al. 2016 AIP Adv.
   [38] `[R]`: "increasing the magnetic field strength could restrain the radial
   cross-field electron current and decrease the radial width of main ionization
   region, which gives rise to the reduction of propellant utilization and
   thruster performance ... both the thrust and anode efficiency are higher for
   the weaker magnetic field in the discharge channel". Hu, Yu and Shen 2020 [39]
   `[R]` (magnet-stage optimisation of a 5 kW multi-cusped thruster). Liu et al.
   2019 [40] `[R]` (channel length). Liu et al. 2021 [41] `[T]` (HIT review):
   "a long channel leads to significant plume divergence and a reduction in
   overall efficiency, but a thruster with a too-short channel performs low
   propellant utilization due to ionization region reduction"; "there should be
   an optimal channel length". Hu et al. 2015 [43] `[R]` (plume-region field). Li,
   Lei and Huang 2026 [42] `[R]` (exit magnetic topology: trade-off between plume
   collimation and axial thrust). Yeo et al. 2021 [9] `[R]` reviews miniaturisation
   perspectives across these families.

### 4.3 Reading of our result against the literature

Yeo et al. [7] is the published analogue of our CL-1/CL-2 disagreement: the same
design under the original and the PIC-anchored closure moved from 102.7 mN /
36.5 % to 62.8 mN / 15.2 %. No published HEMP/CFT chain has demonstrated a
closure-independent geometry ranking; the experimental trend papers
[21,37,38,40,41] are the only anchors, and they are for specific hardware
(HIT 3-stage CFTs at 200-500 W; Giessen/Airbus micro-HEMPT) whose fields were
not published as maps.

### 4.4 Pitfalls

- Closure-dependent rankings presented as design rules (our MDO v2 and [7]
  both show the effect; [6] reports variable importances of the same kind that
  `REFERENCES.md` already flags as unvalidated).
- Surrogates trained on a closure inherit the closure; [7,8] gate surrogates on
  MSE against the 0-D model, not against PIC or experiment.
- Trend claims from one field ("longer channel loses less") that are not
  population trends: our screening (Spearman -0.05 for length) and [41]'s
  optimum-length statement both contradict a monotone rule.

### 4.5 Recommendation mapped to our experiments

- MDO v3: evaluate CL-1, CL-2, CL-3 (sheath-limited) and CL-4 (hybrid-area) on
  the same catalogue and publish the four fronts with pairwise Jaccard, as v2
  does for two; a design is reported as robust only if it is nondominated under
  at least three closures.
- Add literature trend gates (reported, not binding) to the MDO assessment:
  (a) weaker in-channel field -> higher thrust and anode efficiency at fixed
  operating point [38]; (b) longer last-stage magnet -> higher thrust and
  efficiency, longer middle stage -> lower [37]; (c) an interior optimum in channel
  length [40,41]; (d) exit separatrix more parallel to the exit plane -> narrower
  plume [41,42]. A closure whose front contradicts all four is flagged.
- Add the exit-separatrix angle and per-cell mirror ratio (Section 3.4) as
  catalogue descriptors, so the geometry -> loss map has the variables the
  experimental literature actually varied.
- External validation targets available without new hardware: Kornfeld DM9.2
  Table 3.1 [1] (model state, not measurement), Keller 2015 [21] micro-HEMPT
  operating points (50 uN / 230 s at 600 V; 180 uN / 610 s; 360 uN / 860 s),
  Yeo 2020 S1 PIC row [7] (already recorded in
  `data/validation/yeo-2020-s1-external-evidence-v2.json`). All are model-to-model
  or single-point comparisons and must be labelled as such.

---

## 5. Summary table

| Blocker | Change recommended | Effort | Risk |
| --- | --- | --- | --- |
| 1. Power balance has no root for interior p | Accept the two sign/booking corrections **with** new rows R28-R30 (floating-dielectric sheath per cusp: potential drop, electron `2T + Delta phi`, ion `Delta phi + T/2`) and R31 (anode sheath or declared anode fall); declare `CL-3-potentials` (flat interior, exit drop [2,18]) until a cusp-conductivity row exists; add a density/neutral balance or take `n_e,k` from `pic2d`; re-admit under `analytic-consistency` with Kornfeld Table 3.1 and Puca Table 1 as reproduction targets | Medium (ledger + solver rows + tests; paper re-admission) | Medium: new rows need `n_e`; a declared potential closure can be mistaken for a derived result unless labelled |
| 2. Loss-cone p vs collisionless wall-hit | Relabel test-particle outputs as geometric access fractions; add `CL-3-sheath-limited` and `CL-4-hybrid-area` closures; calibrate against `pic2d` steady-state v2 per-cusp fluxes and compare qualitatively with Brandt 2016 [18]; keep mu variation as a published diagnostic | Medium (closure code + one comparison study; no new campaigns needed for CL-3) | Low-medium: sheath drop depends on SEE yield and `T_k`; leak-width prefactor uncertain by 1-4 [4,44] |
| 3. Wall-cusp/cell definition finds nothing | Topology search v3: cusp := axis null + separatrix + wall intersection `z_c`; cell := region between separatrices; stability on `z_c`; mirror ratio as descriptor; test iron sensitivity on the four P2 representatives | Low-medium (fields exist; new protocol, tests, preregistration) | Low: definition follows [27,19,1]; risk is only that `z_c` lands outside the straight channel for many designs (a real finding) |
| 4. Geometry -> performance ranking is closure-dependent | MDO v3 with four closures and a >= 3-closure robustness rule; literature trend gates [37,38,40,41,42] reported; add separatrix-angle and mirror-ratio descriptors; label all validation targets as model-to-model | Medium (depends on 1-3) | Medium: trend gates are hardware-specific; contradiction is a flag, not a falsification |

---

## 6. Honest gaps

- Full texts **not** retrieved (Springer/IEEE fetches timed out or were refused):
  Kahnfeld et al. 2019 [12], Matyash et al. 2010 [13] (arXiv page also timed out),
  Kalentev et al. 2014 [14], Duras et al. 2017 [16], Keller et al. 2015 [21] (only
  the DLR eLib abstract and IEPC-precursor abstract were read). No figure or
  equation numbers are cited from them; the task's "compare against Kalentev 2014
  Fig N at 300 V" cannot be made concrete without the text.
- Not verifiable, therefore not cited: "Spalding" and "Berkowitz" cusp-containment
  papers (no DOI record found under the names given); "Koch" and "DLR/Thales HEMP
  models" as reduced models beyond [1,2]; "Princeton CHT global models" and "MIT
  DCFT global models" (no 0-D power-balance model of the Kornfeld type exists in
  the verified DCFT/CHT literature); "Ma et al. 2024 AST" (the supplied DOI is a
  FEEP paper [73]); Young/Cappelli DCF and MacDonald DCFT papers (not checked;
  not needed for the four blockers); "Kornfeld IEPC-2003" and "Koch IEPC-2007
  HEMP-T 3050 status" papers are cited inside [2] and [23] but their paper numbers
  were not independently confirmed, so they are omitted.
- No published closure was found that uses a collisionless test-particle
  wall-hit fraction; that statement is bounded to the corpus in Section 7.
- The leak-width prefactor (hybrid gyroradius x 1..4) is quoted from [4]'s
  summary and the titles/abstracts of [44,47,48,53]; the original numerical
  values in those papers were not re-read here.
- The recommendation to use `pic2d` per-cusp fluxes as a calibration target
  refers to a development-tier, single-seed result (see the pic2d devlog); it is
  not an accepted kinetic benchmark.
- Nothing in this review changes the recorded status of any experiment or claim;
  every recommendation is a proposal for a new preregistration or a ledger
  revision with its own admission.

---

## 7. Bibliography (verified 2026-09-03)

### HEMP / cusped-field sources and reduced models

1. Kornfeld, G., Koch, N., Harmann, H.-P., "Physics and Evolution of
   HEMP-Thrusters", IEPC-2007-108, 30th International Electric Propulsion
   Conference, Florence, 17-20 September 2007.
   https://electricrocket.org/IEPC/IEPC-2007-108.pdf `[T]`
2. Koch, N., Schirra, M., Weis, S., Lazurenko, A., van Reijen, B., Haderspeck,
   J., Genovese, A., Holtmann, P., Schneider, R., Matyash, K., Kalentyev, O.,
   "The HEMPT Concept - A Survey on Theoretical Considerations and Experimental
   Evidences", IEPC-2011-236, 32nd International Electric Propulsion Conference,
   Wiesbaden, 11-15 September 2011.
   https://electricrocket.org/IEPC/IEPC-2011-236.pdf `[T]`
3. Muffatti, A., Ogawa, H., "Multi-objective Design Optimisation of a Small
   Scale Cusped Field Thruster for Micro-satellite Platforms", ISTS 2017-b-32
   (see `docs/REFERENCES.md` for archive URL and digest). `[T]` (previously)
4. Puca, N., Panelli, M., Battista, F., "A Methodology for the Preliminary Design
   of a High-Efficiency Multistage Plasma Thruster", *Aerotecnica Missili &
   Spazio* 103(4), 321-338 (2024). DOI 10.1007/s42496-024-00203-x `[T]`
5. Kornfeld, G., Koch, N., Coustou, G., "First Test Results of the HEMP thruster
   concept", 28th IEPC, Toulouse, 2003 - cited in [23]; paper number not
   independently confirmed; **not cited above**.
6. Fahey, T., Muffatti, A., Ogawa, H., "High Fidelity Multi-Objective Design
   Optimization of a Downscaled Cusped Field Thruster", *Aerospace* 4(4), 55
   (2017). DOI 10.3390/aerospace4040055 `[R]` (Crossref abstract read)
7. Yeo, S. H., Ogawa, H., Matthias, P., Kahnfeld, D., Schneider, R.,
   "Multiobjective Optimization and Particle-In-Cell Simulation of Cusped Field
   Thrusters for Microsatellite Platforms", *Journal of Spacecraft and Rockets*
   57(3), 603-611 (2020). DOI 10.2514/1.A34584 `[T]`
8. Yeo, S. H., Ogawa, H., "Multi-Objective Design Optimization of Cusped Field
   Thruster via Surrogate-Assisted Evolutionary Algorithms", *Journal of
   Propulsion and Power* 38(6), 973-988 (2022). DOI 10.2514/1.B38854 `[R]`
9. Yeo, S. H., Ogawa, H., Kahnfeld, D., Schneider, R., "Miniaturization
   perspectives of electrostatic propulsion for small spacecraft platforms",
   *Progress in Aerospace Sciences* 126, 100742 (2021).
   DOI 10.1016/j.paerosci.2021.100742 `[R]`
10. Matthias, P., Kahnfeld, D., Schneider, R., Yeo, S. H., Ogawa, H.,
    "Particle-in-cell simulation of an optimized high-efficiency multistage plasma
    thruster", *Contributions to Plasma Physics* 59(9), e201900028 (2019).
    DOI 10.1002/ctpp.201900028 `[R]` (abstract read)
11. Matthias, P., Kahnfeld, D., Kemnitz, S., Duras, J., Koch, N., Schneider, R.,
    "Similarity scaling-application and limits for
    high-efficiency-multistage-plasma-thruster particle-in-cell modelling",
    *Contributions to Plasma Physics* 60, e201900199 (2020).
    DOI 10.1002/ctpp.201900199 `[R]` (abstract read)
12. Kahnfeld, D., Duras, J., Matthias, P., Kemnitz, S., Arlinghaus, P., Bandelow,
    G., Matyash, K., Koch, N., Schneider, R., "Numerical modeling of high
    efficiency multistage plasma thrusters for space applications", *Reviews of
    Modern Plasma Physics* 3, 11 (2019). DOI 10.1007/s41614-019-0030-4 `[R]`
13. Matyash, K., Schneider, R., Mutzke, A., Kalentev, O., Taccogna, F., Koch, N.,
    Schirra, M., "Kinetic Simulations of SPT and HEMP Thrusters Including the
    Near-Field Plume Region", *IEEE Transactions on Plasma Science* 38(9),
    2274-2280 (2010). DOI 10.1109/TPS.2010.2056936; arXiv:0912.0470 `[R]`
14. Kalentev, O., Matyash, K., Duras, J., Luskow, K. F., Schneider, R., Koch, N.,
    Schirra, M., "Electrostatic Ion Thrusters - Towards Predictive Modeling",
    *Contributions to Plasma Physics* 54(2), 235-248 (2014).
    DOI 10.1002/ctpp.201300038 `[R]` (abstract read)
15. Schneider, R., Matyash, K., Kalentev, O., Taccogna, F., Koch, N., Schirra, M.,
    "Particle-in-Cell Simulations for Ion Thrusters", *Contributions to Plasma
    Physics* 49(9), 655-661 (2009). DOI 10.1002/ctpp.200910070 `[R]`
16. Duras, J., Kahnfeld, D., Bandelow, G., Kemnitz, S., Luskow, K., Matthias, P.,
    Koch, N., Schneider, R., "Ion angular distribution simulation of the Highly
    Efficient Multistage Plasma Thruster", *Journal of Plasma Physics* 83(1),
    595830107 (2017). DOI 10.1017/S0022377817000125 `[R]` (abstract read)
17. Kahnfeld, D., Heidemann, R., Duras, J., Matthias, P., Bandelow, G., Luskow,
    K., Kemnitz, S., Matyash, K., Schneider, R., "Breathing modes in HEMP
    thrusters", *Plasma Sources Science and Technology* 27(12), 124002 (2018).
    DOI 10.1088/1361-6595/aaf29a `[R]` (abstract read)
18. Brandt, T., Schneider, R., Duras, J., Kahnfeld, D., Hey, F. G., Kersten, H.,
    Jansen, F., Braxmaier, C., "Particle-in-Cell Simulation of a Down-Scaled HEMP
    Thruster", *Transactions of the Japan Society for Aeronautical and Space
    Sciences, Aerospace Technology Japan* 14(ists30), Pb_235-Pb_242 (2016).
    DOI 10.2322/tastj.14.Pb_235 `[T]`
19. Lewerentz, L., Schneider, R., "Simplified Optimization of the Magnetic
    Configuration of HEMP-Thrusters", *Applied Sciences* 13(6), 3491 (2023).
    DOI 10.3390/app13063491 `[T]`
20. Lewerentz, L., Kahnfeld, D., Schulz, N., Heidemann, R., Schneider, R., "PIC
    Simulations of the MS4 Thruster", *Frontiers in Physics* 10, 833159 (2022).
    DOI 10.3389/fphy.2022.833159 `[R]`
21. Keller, A., Kohler, P., Hey, F. G., Berger, M., Braxmaier, C., Feili, D.,
    Weise, D., Johann, U., "Parametric Study of HEMP-Thruster Downscaling to uN
    Thrust Levels", *IEEE Transactions on Plasma Science* 43(1), 45-53 (2015).
    DOI 10.1109/TPS.2014.2321095 `[R]` (DLR eLib abstract read;
    https://elib.dlr.de/103534/)
22. Matyash, K., Kalentev, O., Schneider, R., Taccogna, F., Koch, N., Schirra,
    M., "Kinetic simulation of the stationary HEMP thruster including the
    near-field plume region", IEPC-2009-110, 31st IEPC, Ann Arbor, 2009. `[C]`
    (paper number confirmed from the reference lists of [23], [16] and [19])
23. Matyash, K., Schneider, R., Mutzke, A., Kalentev, O., Taccogna, F., Koch, N.,
    Schirra, M., "Comparison of SPT and HEMP thruster concepts from kinetic
    simulations", IEPC-2009-159, 31st IEPC, Ann Arbor, 20-24 September 2009.
    https://electricrocket.org/IEPC/IEPC-2009-159.pdf `[T]` (header and
    references)

### MIT DCFT and Princeton CHT

24. Courtney, D., Lozano, P., Martinez-Sanchez, M., "Continued Investigation of
    Diverging Cusped Field Thruster", 44th AIAA/ASME/SAE/ASEE Joint Propulsion
    Conference & Exhibit, AIAA 2008-4631 (2008). DOI 10.2514/6.2008-4631 `[R]`
25. Courtney, D. G., "Development and characterization of a diverging cusped
    field thruster and a lanthanum hexaboride hollow cathode", S.M. thesis, MIT
    Dept. of Aeronautics and Astronautics, 2008. http://hdl.handle.net/1721.1/45239 `[R]`
26. Gildea, S. R., "Fully kinetic modeling of a divergent cusped-field
    thruster", S.M. thesis, MIT, 2009. http://hdl.handle.net/1721.1/54613 `[R]`;
    Gildea, S., Batishchev, O., Martinez-Sanchez, M., "Fully Kinetic Modeling of
    Divergent Cusped Field Thrusters", AIAA 2009-4814 (2009).
    DOI 10.2514/6.2009-4814 `[R]`
27. Gildea, S. R., "Development of the plasma thruster particle-in-cell simulator
    to complement empirical studies of a low-power cusped-field thruster", Ph.D.
    thesis, MIT Dept. of Aeronautics and Astronautics, 2012.
    http://hdl.handle.net/1721.1/79338 `[T]`
28. Gildea, S. R., Matlock, T. S., Martinez-Sanchez, M., Hargus, W. A., "Erosion
    Measurements in a Low-Power Cusped-Field Plasma Thruster", *Journal of
    Propulsion and Power* 29(4), 906-918 (2013). DOI 10.2514/1.B34607 `[R]`
29. Matlock, T. S., "An exploration of prominent cusped-field thruster phenomena:
    the hollow conical plume and anode current bifurcation", Ph.D. thesis, MIT
    Dept. of Aeronautics and Astronautics, 2012. http://hdl.handle.net/1721.1/77097 `[R]`
30. Matlock, T., Daspit, R., Batishchev, O., Lozano, P., Martinez-Sanchez, M.,
    "Spectroscopic and Electrostatic Investigation of the Diverging Cusped-Field
    Thruster", AIAA 2009-4813 (2009). DOI 10.2514/6.2009-4813 `[R]`
31. Raitses, Y., Fisch, N. J., "Parametric investigations of a nonconventional
    Hall thruster", *Physics of Plasmas* 8(5), 2579-2586 (2001).
    DOI 10.1063/1.1355318 `[R]`
32. Smirnov, A., Raitses, Y., Fisch, N. J., "Plasma measurements in a 100 W
    cylindrical Hall thruster", *Journal of Applied Physics* 95(5), 2283-2292
    (2004). DOI 10.1063/1.1642734 `[R]`

### HIT multi-cusped field thrusters

33. Zhao, Y. J., Liu, H., Yu, D. R., Hu, P., Wu, H., "Particle-in-cell simulations
    for the effect of magnetic field strength on a cusped field thruster",
    *Journal of Physics D: Applied Physics* 47, 045201 (2014).
    DOI 10.1088/0022-3727/47/4/045201 `[R]`
34. Liu, H., Wu, H., Zhao, Y., Yu, D., Ma, C., Wang, D., Wei, H., "Study of the
    electric field formation in a multi-cusped magnetic field", *Physics of
    Plasmas* 21 (2014). DOI 10.1063/1.4896250 `[R]`
35. Liu, H., Chen, P.-B., Zhao, Y.-J., Yu, D.-R., "Particle-in-cell simulation for
    different magnetic mirror effects on the plasma distribution in a cusped field
    thruster", *Chinese Physics B* 24(8), 085202 (2015).
    DOI 10.1088/1674-1056/24/8/085202 `[R]`
36. Ma, C., Liu, H., Hu, Y., Yu, D., Chen, P., Sun, G., Zhao, Y., "Experimental
    study on a variable magnet length cusped field thruster", *Vacuum* 115,
    101-107 (2015). DOI 10.1016/j.vacuum.2015.02.007 `[R]`
37. Hu, P., Liu, H., Gao, Y., Mao, W., Yu, D., "An experimental study of the effect
    of magnet length on the performance of a multi-cusped field thruster",
    *Journal of Physics D: Applied Physics* 49(28), 285201 (2016).
    DOI 10.1088/0022-3727/49/28/285201 `[R]` (abstract read)
38. Hu, P., Liu, H., Gao, Y., Yu, D., "Effects of magnetic field strength in the
    discharge channel on the performance of a multi-cusped field thruster", *AIP
    Advances* 6 (2016). DOI 10.1063/1.4962548 `[R]` (abstract read)
39. Hu, P., Yu, D., Shen, Y., "Magnet stage optimization of 5 kW multi-cusped
    field thruster", *Plasma Science and Technology* 22(9), 094015 (2020).
    DOI 10.1088/2058-6272/aba680 `[R]`
40. Liu, H., Zeng, M., Yu, D., Huang, H., "Study of channel length effect on low
    power HEMP Thruster", *Vacuum* 163, 328-337 (2019).
    DOI 10.1016/j.vacuum.2019.02.035 `[R]`
41. Liu, H., Zeng, M., Niu, X., Huang, H., Yu, D., "Low Power Cusped Field
    Thruster Developed for the Space-Borne Gravitational Wave Detection Mission in
    China", *Applied Sciences* 11(14), 6549 (2021). DOI 10.3390/app11146549 `[T]`
42. Li, Y.-H., Lei, T.-T., Huang, T.-Y., "Trade-offs between plume collimation and
    axial thrust in a cusped field thruster with modified exit magnetic
    topology", *Aerospace Science and Technology* 178, 113168 (2026).
    DOI 10.1016/j.ast.2026.113168 `[R]`
43. Hu, P., Liu, H., Mao, W., Yu, D., Gao, Y., "The effects of magnetic field in
    plume region on the performance of multi-cusped field thruster", *Physics of
    Plasmas* 22 (2015). DOI 10.1063/1.4932077 `[R]`

### Cusp confinement, leak width, non-adiabaticity, ring-cusp reduced models

44. Hershkowitz, N., Leung, K. N., Romesser, T., "Plasma Leakage Through a
    Low-beta Line Cusp", *Physical Review Letters* 35, 277-280 (1975).
    DOI 10.1103/PhysRevLett.35.277 `[R]`
45. Leung, K. N., Hershkowitz, N., MacKenzie, K. R., "Plasma confinement by
    localized cusps", *The Physics of Fluids* 19, 1045-1053 (1976).
    DOI 10.1063/1.861575 `[R]`
46. Hershkowitz, N., Smith, J. R., Kozima, H., "Electrostatic self-plugging of a
    picket fence cusped magnetic field", *The Physics of Fluids* 22(1), 122-125
    (1979). DOI 10.1063/1.862450 `[R]`
47. Pechacek, R. E., Greig, J. R., Raleigh, M., Koopman, D. W., DeSilva, A. W.,
    "Measurement of the Plasma Width in a Ring Cusp", *Physical Review Letters*
    45, 256-259 (1980). DOI 10.1103/PhysRevLett.45.256 `[R]`
48. Bosch, R. A., Merlino, R. L., "Confinement properties of a low-beta discharge
    in a spindle cusp magnetic field", *The Physics of Fluids* 29, 1998-2006
    (1986). DOI 10.1063/1.865628; erratum DOI 10.1063/1.866025 `[R]`
49. Knorr, G., Merlino, R. L., "The role of fast electrons for the confinement of
    plasma by magnetic cusps", *Plasma Physics and Controlled Fusion* 26(2),
    433-442 (1984). DOI 10.1088/0741-3335/26/2/004 `[R]`
50. Haines, M. G., "Plasma containment in cusp-shaped magnetic fields", *Nuclear
    Fusion* 17, 811-858 (1977). DOI 10.1088/0029-5515/17/4/015 `[R]`
51. Dunnett, R. M., "Single-particle motion in multiple-cusp magnetic fields",
    *Nuclear Fusion* 9(1), 82-84 (1969). DOI 10.1088/0029-5515/9/1/012 `[R]`
52. Cohen, R. H., Rowlands, G., Foote, J. H., "Nonadiabaticity in mirror
    machines", *The Physics of Fluids* 21, 627-644 (1978). DOI 10.1063/1.862271
    `[R]`
53. Hubble, A., Foster, J., "Plasma Collection Width Measurements in a 10-cm Ring
    Cusp Discharge Chamber", 44th AIAA/ASME/SAE/ASEE Joint Propulsion Conference &
    Exhibit, AIAA 2008-4639 (2008). DOI 10.2514/6.2008-4639 `[R]`
54. Goebel, D. M., Wirz, R. E., Katz, I., "Analytical Ion Thruster Discharge
    Performance Model", *Journal of Propulsion and Power* 23(5), 1055-1067
    (2007). DOI 10.2514/1.26404 `[R]`
55. Wirz, R., Goebel, D., "Effects of magnetic field topography on ion thruster
    discharge performance", *Plasma Sources Science and Technology* 17(3), 035010
    (2008). DOI 10.1088/0963-0252/17/3/035010 `[R]`
56. Mao, H.-S., Wirz, R. E., Goebel, D. M., "Plasma Structure of Miniature
    Ring-Cusp Ion Thruster Discharges", *Journal of Propulsion and Power* 30(3),
    628-636 (2014). DOI 10.2514/1.B34759 `[R]`
57. Goebel, D. M., Katz, I., *Fundamentals of Electric Propulsion: Ion and Hall
    Thrusters*, Wiley (2008). DOI 10.1002/9780470436448 `[R]`
58. Lieberman, M. A., Lichtenberg, A. J., *Principles of Plasma Discharges and
    Materials Processing*, 2nd ed., Wiley (2005). DOI 10.1002/0471724254 `[R]`

### Sheath closures

59. Hobbs, G. D., Wesson, J. A., "Heat flow through a Langmuir sheath in the
    presence of electron emission", *Plasma Physics* 9, 85-87 (1967).
    DOI 10.1088/0032-1028/9/1/410 `[R]`
60. Riemann, K.-U., "The Bohm criterion and sheath formation", *Journal of Physics
    D: Applied Physics* 24, 493-518 (1991). DOI 10.1088/0022-3727/24/4/001 `[R]`
61. Ahedo, E., "Presheath/sheath model with secondary electron emission from two
    parallel walls", *Physics of Plasmas* 9, 4340-4347 (2002).
    DOI 10.1063/1.1503798 `[R]`
62. Ahedo, E., Martinez-Cerezo, P., Martinez-Sanchez, M., "One-dimensional model
    of the plasma flow in a Hall thruster", *Physics of Plasmas* 8(6), 3058-3068
    (2001). DOI 10.1063/1.1371519 `[R]`
63. Barral, S., Makowski, K., Peradzynski, Z., Gascon, N., Dudeck, M., "Wall
    material effects in stationary plasma thrusters. II. Near-wall and in-wall
    conductivity", *Physics of Plasmas* 10(10), 4137-4152 (2003).
    DOI 10.1063/1.1611881 `[R]`
64. Keidar, M., Boyd, I. D., Beilis, I. I., "Plasma flow and plasma-wall
    transition in Hall thruster channel", *Physics of Plasmas* 8(12), 5315-5322
    (2001). DOI 10.1063/1.1421370 `[R]`

### Magnetic-null topology and field computation

65. Parnell, C. E., Smith, J. M., Neukirch, T., Priest, E. R., "The structure of
    three-dimensional magnetic neutral points", *Physics of Plasmas* 3, 759-770
    (1996). DOI 10.1063/1.871810 `[R]`
66. Haynes, A. L., Parnell, C. E., "A trilinear method for finding null points in
    a three-dimensional vector space", *Physics of Plasmas* 14 (2007).
    DOI 10.1063/1.2756751 `[R]`
67. Haynes, A. L., Parnell, C. E., "A method for finding three-dimensional
    magnetic skeletons", *Physics of Plasmas* 17(9), 092903 (2010).
    DOI 10.1063/1.3467499 `[R]`
68. Murphy, N. A., Parnell, C. E., Haynes, A. L., "The appearance, motion, and
    disappearance of three-dimensional magnetic null points", *Physics of
    Plasmas* 22(10), 102117 (2015). DOI 10.1063/1.4934929 `[R]`
69. Greene, J. M., "Locating three-dimensional roots by a bisection method",
    *Journal of Computational Physics* 98, 194-198 (1992).
    DOI 10.1016/0021-9991(92)90137-N `[R]`
70. Lau, Y.-T., Finn, J. M., "Three-dimensional kinematic reconnection in the
    presence of field nulls and closed field lines", *The Astrophysical Journal*
    350, 672 (1990). DOI 10.1086/168419 `[R]`
71. Ortner, M., Coliado Bandeira, L. G., "Magpylib: A free Python package for
    magnetic field computation", *SoftwareX* 11, 100466 (2020).
    DOI 10.1016/j.softx.2020.100466 `[R]`

### Identifiability

72. Bellman, R., Astrom, K. J., "On structural identifiability", *Mathematical
    Biosciences* 7(3-4), 329-339 (1970). DOI 10.1016/0025-5564(70)90132-X `[R]`

### DOI supplied by the task that resolves to a different paper

73. Yeo, S. H., Gadisa, D., Ogawa, H., Bang, H., "Multi-objective design
    optimization and physics-based sensitivity analysis of field emission electric
    propulsion for CubeSat platforms", *Aerospace Science and Technology* 154,
    109516 (2024). DOI 10.1016/j.ast.2024.109516 `[R]` - FEEP, not cusped-field;
    listed only to document the check.

Count: 73 entries; 72 real, verified references cited or explicitly withheld
(entry 5 is withheld pending paper-number confirmation), plus one documented
misattribution (entry 73).
