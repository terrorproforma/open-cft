# Legacy MATLAB Technical Audit

## Scope and confidence

All 16 files under `FYP/` were inspected. No datasets, FEMM model files,
optimizer framework, test suite, or dependency manifest are present. Findings
labelled **confirmed** follow directly from MATLAB data flow or internal
contradictions. Findings labelled **suspicious** need the cited paper,
experimental data, or a runnable legacy environment before correction.
The primary ISTS publication is now identified and digested in
`docs/REFERENCES.md`; paper claims and code facts remain separate because the
publication-run source revision is not available.

## End-to-end flow

1. `CFTOpt.m:1-20` declares 8 decision variables, 4 objectives, and 1
   constraint through external `Range` objects.
2. `CFTOpt.m:28-33` calls `Performance_est`.
3. `Performance_est.m:20-39` sets global empirical constants and converts mass
   flow from sccm to kg/s and particles/s.
4. `Performance_est.m:66-80` applies radial ordering checks and one hard-coded
   rejected individual.
5. `Performance_est.m:83-89` runs FEMM, then the nonlinear plasma solve, and
   writes the 30 solution values.
6. `FEMMrun.m:13-226` builds and solves an axisymmetric magnetostatic model and
   exports centreline/wall field samples.
7. `HEMP_solver.m:28-64` derives cusp probabilities from those files and runs
   `lsqnonlin(@Power_B_EQs, ...)`.
8. `Performance_est.m:132-156` derives efficiency, thrust, specific impulse,
   and signed optimization objectives.

Decision-vector order is:

`[Ua V, Ia A, flow sccm, inner magnet radius mm, outer magnet radius mm,
inner shield radius mm, outer shield radius mm, outer enclosure radius mm]`.

The 30 plasma unknowns are documented in `Power_B_EQs.m:24-31` and
`boundaries.m:4-11`: 5 power/loss terms, 4 cell potentials, 4 electron
temperatures, 4 source currents, 5 electron currents, 5 ion currents, and 3
cusp ion currents.

## Confirmed correctness defects

### 1. Anode voltage/current do not reach the residual system

- `Performance_est.m:10` declares globals but omits `Ua` and `Ia`.
- `Performance_est.m:31-32` creates local `Ua` and `Ia`.
- `HEMP_solver.m:4,12-13` receives/assigns local variables with those names but
  declares only `p1..p4` global at line 10.
- `boundaries.m:14` and `Power_B_EQs.m:77` read **global** `Ua` and `Ia`.

MATLAB globals are distinct from same-named locals. Therefore the bound arrays
at `boundaries.m:20-30` are built from empty globals, while residual rows using
`Ua`/`Ia` can collapse to empty expressions. The initial guess still uses the
local values, hiding the state break. This invalidates legacy solver output
until fixed and regression-tested.

### 2. Boolean comparisons are incorrectly optimized as residual equations

`Power_B_EQs.m:140-168` appends 11 logical comparisons to the residual vector.
`lsqnonlin` minimizes residuals toward zero; in MATLAB a satisfied comparison is
`1` and a violated comparison is `0`. The solver is therefore rewarded for
violating these intended constraints. Bounds or a proper constrained
formulation are required.

The file claims 28 equations (`Power_B_EQs.m:13`) but actually supplies 33
continuous balance/definition residuals plus 11 logical residuals for 30
unknowns. Overdetermination is legal, but the undocumented formulation and
mixed logical residuals are not equivalent to the stated model.

### 3. A variable anode voltage is replaced by a hard-coded 1000 V

`Power_B_EQs.m:164` uses `x(9)-1000>=0`, despite `CFTOpt.m:9` allowing
`Ua=1..1000 V` and the adjacent comment saying this value is variable. This
also conflicts with `boundaries.m:20`, which intends a lower bound of `Ua`.

### 4. Energy-balance signs contradict their own equation description

`Power_B_EQs.m:111-117` says received power equals cusp, thermalization,
ionization, and excitation losses. The expanded commented equations at
lines 119-122 subtract the `CE` excitation term, but executable equations at
lines 123-126 add it. The implementation is internally inconsistent. The
physically correct sign must still be confirmed from Kornfeld et al.

### 5. FEMM parallelism is knowingly unsafe but enabled upstream

`params.m:7` requests 12 simultaneous evaluations. `FEMMrun.m:6-10` and
`:162-164` explicitly note that FEMM confuses multiple instances, while
`openfemm`/`closefemm` uses shared application automation. There is no process
lock, worker isolation, atomic output publish, or `onCleanup`. Parallel
optimizer runs can cross-control FEMM or leave it running after an exception.

### 6. Cusp extraction can use stale/undefined state

- `cusp_prob.m:7-8` relies on global `homefolder` and a wildcard for the run.
- If no files match, `B_dataname`, `Minima`, and `Maxima` are never defined.
- If multiple wall/centreline files match, lines 60-162 overwrite prior
  values; filesystem order determines the winner.
- “Maxima” and “Minima” at lines 101 and 151 are window **means**, not extrema.
- Empty windows produce empty means/NaNs with no diagnostic.
- `p=zeros(4)` at line 184 creates a 4x4 matrix although only four linear
  entries are used.
- Lines 180-181 replace cusp-4 high/low fields using an unexplained swap that
  can make `B_low > B_high`; line 187 then produces a complex angle.

The modern kernel rejects missing, non-finite, and inverted fields instead of
silently propagating a complex/NaN result.

### 7. Solver acceptance ignores residual quality

`HEMP_solver.m:64` discards residual norm and residual vector. In
`Performance_est.m:91-128`, exit flags 1-3 are accepted solely by status and
flag 4 is rejected. No equation-scale normalization or residual tolerance is
checked. With `TolFun=1e-50` (`HEMP_solver.m:61`), status is not credible
evidence that the physical balances close.

### 8. Non-finite objective values can be reported as success

`Performance_est.m:249-337` checks inequalities, reality, and negative power,
but never checks `isfinite`. MATLAB comparisons with NaN are false and
`isreal(NaN)` is true, so NaN thrust/efficiency/Isp can pass to
`output.f*` at lines 153-156.

### 9. `findLocalMin` is not robust

`findLocalMin.m:20-22` uses exact equality and `find`, so duplicate values can
return a vector that cannot be assigned to one row. It also shadows MATLAB's
`min` function and returns an unassigned variable when no local minimum exists.
No production caller exists in this repository.

## Suspicious or unverified physics/logic

- `Performance_est.m:28` sets cathode electron temperature to zero and labels
  it probably incorrect. This changes first-cell energy terms.
- `Performance_est.m:41` allows mass utilization up to 1.2, while the final
  total-efficiency check rejects only values above 1. Whether multi-charge
  states justify this needs documentation.
- `Performance_est.m:135` defines grid efficiency as `1-phi1/phi2`; verify
  which potentials represent beam/grid losses.
- `Power_B_EQs.m:93` uses `abs(x(27))`, creating a nondifferentiable residual,
  although bounds already intend `x(27)<0`.
- The 30 residuals have mixed physical scales (watts, volts/eV, amperes and
  logical values) with no normalization, so least squares weights high-scale
  equations more heavily.
- `FEMMrun.m:23-80` varies only radial geometry. Axial magnet locations,
  chamber length, contour endpoints, materials, magnet grades/orientations,
  mesh settings, and outer boundary are fixed.
- `cusp_prob.m:11-27` uses hard-coded axial windows not derived from the FEMM
  geometry. Its sign-based selector excludes exact interval endpoints.
- The special cusp-4 reassignment at `cusp_prob.m:174-181` is described as an
  “anode condition” but has no source or dimensional rationale.
- `FEMMrun.m:89-97` uses an asymptotic coefficient containing `0.2` while the
  model units are millimetres. Confirm the FEMM boundary formula and units.
- `plasmaCalc.m:11-13` assumes STP xenon density and an inlet area equal to
  0.1% of chamber area. It is exploratory, prints intermediate values, takes
  no inputs, returns no outputs, and is not connected to the solver.

## Objectives, constraints, and sign conventions

- `CFTOpt.m:3-5` advertises four objectives and one constraint.
- `Performance_est.m:153-156` negates thrust, total efficiency, and Isp so a
  minimizer maximizes them; anode power remains positive for minimization.
- Failure uses negative integer category codes (`-20` through `-250`) while
  success is `g=1`. These are labels, not continuous constraint violations.
  An optimizer expecting `g<=0` feasibility would interpret them backwards;
  repository-external framework behavior is required to determine impact.
- `Performance_est.m:180-237` requires five radii to be strictly ordered with
  more than 0.01 mm gap, inner magnet radius above 2.5 mm, and outer enclosure
  below 49.99 mm. `CFTOpt.m:14-19` itself only advertises `[2,50]`.
- `Performance_est.m:71-79` hard-codes rejection of generation 35,
  individual 39. This is run-history contamination, not a design constraint.
- `plotParetoOptwithColourDots.m:25-31` expects objectives in columns 4:7,
  constraint in 8, and decisions in 9:16. It plots signed negative objective
  values and reverses efficiency display at line 148.

## Reconciliation with ISTS 2017-b-32

The paper validates the high-level process chain and the form of the
mirror-angle, beam/grid efficiency, thrust, Isp, and `P=Ua*Ia` calculations. It
does **not** print the complete Kornfeld residual set and therefore does not
resolve the executable `CE` sign conflict or justify the logical residuals.

Cross-source differences must not be “fixed” by choosing one source:

- Paper: 3 objectives. Snapshot: 4 in `CFTOpt`/surrogate training, while the
  plotter uses 3 and treats power as color.
- Paper: 96 individuals over 100 generations. Snapshot:
  `params.m:11-12` uses 96 over 50, with generation-50 analysis filenames.
- Paper: 8 variables claimed, but only 5 are fully listed and reported.
  Snapshot: all 8 exist, including three shield/enclosure radii absent from
  publication tables.
- Paper: only use surrogates inside a 5% MSE threshold, then reports no model
  achieved 5%, but still presents surrogate-based Sobol indices. Snapshot:
  `SensitivityAnalysis_Surr_rev.m:16-22` performs no quality gate.
- Paper abstract: anode current is most influential for all objectives.
  Paper Tables 2-3/body/conclusion: mass flow is greatest for all three.
- Paper §2.5 limits singly charged mass utilization to 1. Snapshot:
  `Performance_est.m:42` allows 1.2.
- Paper Table 4 S1 values imply `990.6*3.30=3268.98 W`; §3.2 says 3466 W,
  while its percentage comparisons align with roughly 3269 W.

These conflicts make the paper valuable equation/source evidence, but not a
bit-for-bit oracle for this source snapshot or its sensitivity outputs.

## I/O and reproducibility

- Paths mix `/` and `\`, `pwd`, and global `homefolder`; outputs depend on the
  launch directory (`Performance_est.m:12`, `FEMMrun.m:192,214,221`).
- Run identity is encoded only as generation/individual filenames. There is no
  design hash, config snapshot, software version, material version, units
  metadata, or atomic completion marker.
- `csvwrite` writes a `.dat` file (`Performance_est.m:89`) with no headers.
- `try mkdir(...); catch; end` suppresses all filesystem errors.
- Random state is persisted for surrogate training (`buildSurrogates.m:41-49`)
  but no provenance ties a model to source data or code.

## Missing dependencies and absent inputs

Repository-external functions/classes include:

- optimizer: `Range`, `loadOptValues`, `assign_fitness`, `sort_nd_maxcv`;
- surrogate/SAEA: `Surrogate`, `set_range`, `add_points`, `train_rev`,
  `strsplit_rev`, `sensitivity_analysis`, global optimizer `state`;
- plotting helpers: `narrowColorBars`, `cblabel`, `appendLegend`;
- FEMM MATLAB interface: `openfemm`, all `mi_*` and `mo_*` functions.

MATLAB Optimization Toolbox is required for `optimoptions` and `lsqnonlin`.
Expected `CFTOpt-SAEA_rev-*.dat/.mat/.csv` files are absent. There is no
optimizer driver, FEMM installation manifest, cited-paper snapshot, baseline
solution, or automated regression oracle, so the complete legacy project
cannot currently be reproduced from this repository alone.

## Dead and experimental code

Large alternative-solver and post-hoc-boundary blocks in `HEMP_solver.m:65-139`
are commented out. `plasmaCalc.m`, `findLocalMin.m`, `vline.m`, and the
sensitivity/plot scripts are not called by the production path. `mse` is
accepted by `CFTOpt` but ignored (`CFTOpt.m:28`). `feasibles`, `xiscol`, and
several plot settings are computed but unused in
`plotParetoOptwithColourDots.m`.
