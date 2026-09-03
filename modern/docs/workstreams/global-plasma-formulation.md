# Corrected Global Xenon Discharge Formulation

## Evidence boundary

The executable lineage starts with Kornfeld, Koch, and Harmann,
“Physics and Evolution of HEMP-Thrusters,” IEPC-2007-108 (2007). The paper
prints a stationary, four-cell, prescribed-voltage/prescribed-current power
model and a DM9.2 comparison table. Fahey, Muffatti, and Ogawa (2017) carried
that model into the CFT optimization lineage. Yeo et al. (2020) compared
`MDO (original)` with PIC-MCC. The fixture keeps that source-native label;
its postprocessing interpretation is separate editorial metadata. These
outputs are external cross-model evidence only.

This implementation does not claim a neutral-discharge simulation. Kornfeld
prescribes anode current because it includes neither neutral density nor gas
flow. It therefore predicts a conditional current/power distribution, not mass
utilization, ionization efficiency, thrust, or specific impulse.

## State and balances

Four cell potentials `phi[1:4]` [V], four electron temperatures `Te[1:4]`
[eV], four equal electron/ion source currents `I[1:4]` [A], five electron
currents `je[0:4]` [A], five ion currents `ji[0:4]` [A], and three cusp ion
currents `jic[1:3]` [A] form the 25-value reduced state.

Kornfeld lists three additional cusp potentials. In every printed local and
global cusp expression, their positive and negative terms cancel exactly.
Without a sheath-current relation they are unidentifiable. Solving for them
would create a three-dimensional null space, so the CPU solver eliminates
them and retains all 28 balance equations as an overdetermined 28-by-25
least-squares system.

For cell energy gain

`DeltaE[1] = phi[1]-phi0+Te0`

`DeltaE[k] = phi[k]-phi[k-1]+Te[k-1]`, for `k=2..4`.

The implemented equations are:

- cathode emission: `je0 = p0*(phi1-phi0)^(3/2)`;
- electron continuity: `je[k] = je[k-1]*(1-p[k]) + I[k]`;
- ionization source: `I[k] = je[k-1]*(1-p[k])*CI*DeltaE[k]/EI`;
- interface current: `je[k]+ji[k]=Ia`;
- dielectric cusp current: `jic[k]=p[k]*je[k-1]`;
- ion continuity: `ji[k-1]=ji[k]+I[k]-jic[k]`;
- signed terminal current: `ji3-I4-ji4=0`, hence `ji3=I4+ji4`,
  with `ji4<=0`;
- thermal transport:
  `Te[k]*(je[k-1]*(1-p[k])+I[k])`
  `=CT*je[k-1]*(1-p[k])*DeltaE[k]`, `k=2..4`;
- each cell: received electron power equals thermalized, ionization, and
  excitation losses;
- globally: `Ua*Ia = Pbeam+PI+PE+Pcusp+Panode`.

The complete indexed equations and source confidence are machine-readable in
`spec/plasma/equation-ledger.json`.

## Corrected signs

The archived `FYP/Power_B_EQs.m` uses
`ji[k-1]=ji[k]+I[k]+jic[k]`. A cusp current leaves the axial ion-current
control volume and must be subtracted. The published DM9.2 rounded table
independently closes the corrected relation in all three cells:

- `0.893 + 0.008 - 0.007 = 0.894 A`;
- `0.363 + 0.543 - 0.013 = 0.893 A`;
- `0.155 + 0.310 - 0.102 = 0.363 A`.

The archived third-cell equation misses by `2*0.102 = 0.204 A`.
At the terminal, signed `ji4=-0.002 A` gives
`0.155-0.157-(-0.002)=0`, or `0.155+0.002=0.157 A`.

Kornfeld's fourth thermal definition transports
`je3*(1-p4)+I4`, not `je4`. The direct `p4*je3` current reaches the anode
boundary, and no fourth electron recurrence equates `je4` to that transported
current. On the rounded DM9.2 state, this correction reduces row R14 from
`5.07211604 W` to `0.02831104 W`.

Excitation is also implemented as a loss. Kornfeld assumptions 7 and 11 say
received power is distributed to ionization, excitation, and thermalization,
with `CE+CI+CT=1`. The archived executable `+CE` term contradicts that prose
and its own commented expansion. The broken plus sign is not an alternative.

The raster/OCR sign on the terminal ion contribution to anode loss lacks an
independent text rendering. The API therefore exposes two named
`AnodeIonEnergySign` hypotheses. `SOURCE_MINUS_SIGN` is the default because
`-ji4*(phi4-Ua)` is nonnegative under the source inequalities. The alternative
is never selected silently. The term is reported as signed
`anode_ion_energy_exchange_w`: positive transfers power from plasma to anode;
negative transfers it from anode to plasma. Only the separately named
electron/recombination loss is constrained nonnegative, so a net sum cannot
hide an invalid named loss.

## Numerical contract

Current equations are divided by `Ia`; energy equations are divided by
`Ua*Ia`. Both `Ua` and `Ia` must independently be finite positive normal
binary64 values, and their product must also be normal and representable.
Subnormal scales are rejected during typed input construction, before bounds,
residuals, or solving. Normal input status is necessary but not sufficient:
`Ua=sys.float_info.min` is intentionally rejected because required
`Ua^(3/2)` underflows. `Ia=sys.float_info.min` is accepted only when `Ua`,
`Ua*Ia`, `Ua^(3/2)`, and all derived bounds remain valid. Bounds and physical
restrictions are constraints, not Boolean least-squares rows. The admissible
region includes finite box bounds,
nondecreasing cell potential, `phi4>=Ua`, positive temperature, `ji4<=0`,
`ji3>=abs(ji4)`, and nonnegative named power terms.

The dependency-free CPU solver uses variable-scaled, column-pivoted QR
Levenberg-Marquardt steps, projected bounds, deterministic damping, and
deterministic multi-start. Diagnostics retain every normalized row, numerical
rank, and a condition estimate for the independent scaled Jacobian columns.
A state is published only when the
normalized infinity norm meets tolerance and every value, bound, inequality,
and reported power is finite. The exact chain-rule Jacobian is generated by
forward-mode differentiation. A separate bound-aware central/one-sided finite
difference implementation checks it.

The corrected rounded DM9.2 state has maximum normalized residual
`1.48660935e-3` at global row R27. A 500-iteration source-seeded probe reduced
the observed floor to `4.89350681e-4` but did not close at strict tolerance;
the scaled Jacobian rank was 22 of 25. It is reported as a model/evidence
discrepancy, never as a solved state. A zero-cusp manufactured state derived
from the same current and energy equations closes below `1e-15`, proving the
strict publication path without treating rounded evidence as truth.

## Known global-row inconsistency for interior cusp loss (2026-09-03)

Rows R00-R26 are mutually consistent and are parametrized by the four cell
potentials (`potential_parametrized_state`). Substituting them into the global
power row leaves

`R27 = 2*(j_e3*(1-p4)+I4)*(phi_4-Ua) + EI*(p1*j_e0+p2*j_e1+p3*j_e2)`

(`global_row_closed_form`), which is non-negative on the admissible region and
vanishes only for `p1=p2=p3=0` with `phi_4=Ua`. Strict closure is therefore
possible for zero interior cusp probability (any `p4`) and impossible for any
positive interior probability; the residual floors reported by the solver for
`p != 0` are this inconsistency, not a convergence failure. The two terms are
inherited from the source: the recombination energy of cusp-lost ions is booked
in both `PI` and `Pcusp` (Kornfeld assumption 8), and the anode electron term
carries the sign of an ion. The derivation, evidence, and the minimal
correction (status `PROPOSED_NOT_ACCEPTED`, because it makes the global row an
identity and leaves all four potentials free) are in
`global-plasma-closure-analysis.md` and
`spec/plasma/equation-ledger.json#global_row_consistency`.

The ordering projection inside the solver is the isotonic
(pool-adjacent-violators) projection `project_nondecreasing`; the earlier
`sorted()` permuted potential identities and stalled zero-cusp solves at
1000 V.

## Remaining evidence blockers

- A machine-readable original MathCAD worksheet is needed to settle raster
  operators, especially the terminal anode-ion term.
- The cancelled cusp potentials require an independently sourced sheath
  current/voltage closure before they can become identifiable state variables.
- `CE`, `CI`, and `CT` were selected to reproduce observed thermal losses;
  independent xenon rate/cross-section closures and uncertainty are absent.
- Neutral density, injected mass flow, double ionization, charge exchange,
  non-Maxwellian electron kinetics, plume divergence, and facility effects
  are absent.
- The source reports minimum-error solutions with global power error up to
  0.5%; its rounded DM9.2 table is not an exact residual oracle.
- Original run archives and experimental measurements with uncertainty are
  required before predictive validation or optimizer coupling.
