# Global Plasma Workstream Development Log

## 2026-09-03 — Four-cell closure analysis for p != 0 (branch fix/plasma-network-closure)

### Root cause

- Reproduced the MDO v1 probe exactly (80 cases, `start_count=3`,
  `residual_tolerance=1e-8`): 13/80 closed, all at `p=0`; floors
  `4.739e-4..1.960e-1`; 2.9 s per failing multistart solve; dominant row R27
  for moderate `p`, current rows R03/R06/R07/R11 for `p=0.64` below 1000 V.
- Rows R00-R26 are consistent and parametrized by the four potentials.
  Substituting them into R27 gives exactly
  `2*(j_e3*(1-p4)+I4)*(phi_4-Ua) + EI*(p1*j_e0+p2*j_e1+p3*j_e2)`
  (verified to `1.9e-13` relative over 400 random cases). Both terms are
  non-negative on the admissible region, so no root exists for any positive
  interior cusp probability; `p4`-only loss closes for every value tried
  (`1e-4..0.3`), and every zero-cusp closure sits at `phi_4 = Ua`.
- Origin: Kornfeld IEPC-2007-108 assumption 8 books ionization as a frozen
  loss and again as recombination in the cusp loss (`+EI` per cusp-lost
  electron, also in legacy `Power_B_EQs.m`), and the printed anode electron
  term `(phi_4-Ua+T4)` has the sign of an ion. The published DM9.2 table's
  R27 misfit (`1.49e-3`) and the earlier `4.89e-4` DM9.2 probe floor are the
  same inconsistency. The 2026-09-01 audit corrections cancel in the
  substitution and did not introduce it.
- Classification: (a) genuine model inconsistency for interior `p != 0`, with
  (d) the solution sub-region `p1=p2=p3=0`, `phi_4=Ua` (and an independent
  current-carrying infeasibility for large interior loss at low voltage), plus
  a secondary (b) solver defect below.

### Evidence

- Jacobian at every floor point: rank 22/25, independent-subspace condition
  9-200, `|J^T r|` comparable to the floor (constrained minimum on the
  `phi_4 >= Ua` face, not a stationary point).
- Continuation `p = eps*(1,1,1,1)` at 300 V/1 A: floor `1.28e-6, 1.28e-5,
  1.30e-4, 3.99e-4, 1.43e-3, 5.78e-3` for `eps = 1e-4..0.3` (linear; no lost
  branch). `p = (0,0,0,eps)`: closes to `1e-14` at every `eps`.
- Differential evolution over the 25-box (205k evaluations): best `2.06e-2`.
  200 random feasible starts through the production LM: 0/200 closed, floors
  `1.82e-3..8.25e-2`.
- Relaxing `phi_4 >= Ua` admits exact roots 0.6-12.6 V below `Ua`
  (compensating-error roots; rejected by `is_feasible`).
- Infimum of `|R27|` on the feasible exact manifold: `4.45e-3` (DM9.2 `p`,
  300 V/1 A), `3.19e-4` (1000 V/1 A); feasible manifold empty for
  `p=(0.64,0.64,0.64,0.64)` at 150-500 V.

### Solver defect fixed

- The `sorted()` ordering projection permuted potential identities and stalled
  3/16 zero-cusp probe cases at 1000 V (`step_tolerance_without_balance` with
  damping `1e10`, `phi_3 = phi_4` pinned). Replaced by the isotonic
  pool-adjacent-violators projection `project_nondecreasing` in
  `cft_revival.plasma.solver` and `cft_revival.plasma_network.solver`;
  zero-cusp grid now 16/16 (was 13/16). No effect on the `p != 0` outcome.

### Implemented

- `potential_parametrized_state`, `global_row_closed_form` (diagnostics),
  `project_nondecreasing` (exported).
- `spec/plasma/equation-ledger.json#global_row_consistency` with status
  `PROPOSED_NOT_ACCEPTED` (drop `+EI` from `Pcusp`; anode electron term
  `(Ua-phi_4+T4)`); rows and power expressions unchanged; new limitation line.
- `docs/workstreams/global-plasma-closure-analysis.md` (derivation,
  evidence, classification, proposal, MDO implication).
- Tests: `tests/plasma/test_closure_p_nonzero.py` (9),
  `tests/plasma_network/test_plasma_network_closure_p_nonzero.py` (2),
  `tests/plasma/test_solver.py` (+13: projection properties, zero-cusp grid
  incl. 1000 V, determinism).

### Validation

- `tests/plasma` 58 passed, `tests/plasma_network` 66 passed, `tests/physics`
  86 passed; `tests/experiments/{l1a_plasma_coupling,
  four_cell_topology_search, four_cell_topology_search_v2}` 8/9/10 passed.
- No `results/`, preregistration, or `FYP/` file touched.

### MDO implication

- The MDO v1 disclosure remains historically correct. The solver is now
  usable for anode-cusp-only loss (`p1=p2=p3=0`); a v2 campaign with interior
  cusp probabilities still needs the proposal above accepted together with a
  potential closure.

## 2026-09-01 — Corrected discharge foundation

### Evidence and derivation

- Reviewed the preserved MATLAB residual, bounds, solver, plasma scratch
  calculation, and performance post-processing without editing `FYP/`.
- Retrieved and re-read Kornfeld et al. IEPC-2007-108 equation text and DM9.2
  table; retained Fahey/Muffatti/Ogawa 2017 as CFT lineage and Yeo et al. 2020
  as external comparison only.
- Reduced the nominal 28 unknowns to 25 because the three printed cusp
  potentials cancel algebraically and have no sheath closure.
- Corrected axial ion continuity to subtract cusp outflow. The published
  rounded currents close all three corrected balances; the archived
  third-cell sign misses by 0.204 A.
- Kept excitation as a loss and exposed the independently unresolved
  anode-ion sign as two named hypotheses.

### Implemented

- Added immutable, unit-named discharge inputs, reduced state, bounds,
  residual, power, closure, result, and diagnostic records.
- Added 28 raw and normalized current/energy equations, true inequality
  margins, finite publication checks, a fixed-layout batch residual API, and
  a dependency-free CPU reference.
- Added projected damped Gauss-Newton solving, manufactured-system support,
  exact forward-mode Jacobians, and independent bound-aware finite
  differences.
- Added source/equation and 2020 external-evidence ledgers plus API,
  formulation, limitations, devlog, and learning documentation.

### Validation

- Focused `python -m pytest tests/plasma -q`: 23 passed.
- Default full collection exposed a concurrent shared-test basename collision
  between `tests/pic/test_warp_backend.py` and `tests/test_warp_backend.py`.
  No shared file was changed.
- Compatible full `python -m pytest --import-mode=importlib -q`: 471 passed,
  one unchanged optional-pybind11 test skipped.
- `python -m compileall -q src tests`: passed.
- Direct 100-column scan of plasma source/tests: passed.
- `git diff --exit-code -- FYP`: passed.
- `mypy` and `ruff` were not installed, so they were not run and were not
  installed.
- No dependency was installed and no Git mutation, commit, or push was
  performed.

### Deliberate exclusions

- No shared source, tests, configuration, package initializer, `FYP/`, Git
  state, dependency, C++/Warp implementation, commit, or push was changed.
- No predictive-accuracy, experimental-validation, or performance claim is
  made.

## 2026-09-01 — Global-plasma audit correction

### Corrected

- Changed terminal ion continuity to signed
  `j_i3-I4-j_i4=0`; the DM9.2 values close as
  `0.155+0.002=0.157 A`.
- Changed fourth-cell thermal transport from `T4*j_e4` to Kornfeld's
  `T4*(j_e3*(1-p4)+I4)`. The rounded DM9.2 row decreased from
  `5.07211604 W` to `0.02831104 W`.
- Replaced normal-equation LM steps with variable-scaled, column-pivoted QR
  LM steps and added deterministic multi-start.
- Added per-row residuals, numerical rank, and condition estimate to every
  solver status. Strict residual and feasibility gates remain mandatory.
- Hardened public projection callbacks: input is clipped first; projected
  dimensions, finiteness, and bounds are validated; accepted output is
  re-clipped before evaluation.
- Rejected nonrepresentable derived bounds, tiny `Ua^(3/2)` domains, and
  underflowed/subnormal/overflowed `Ua*Ia`; removed residual-scale fallbacks.
- Split anode boundary accounting into nonnegative electron loss, signed ion
  energy exchange, and net transfer.
- Expanded the machine ledger to all 28 rows and seven power expressions,
  using exact source label `MDO (original)` with editorial interpretation
  separate.

### Solver evidence

- A source-derived zero-cusp manufactured global state closes all normalized
  rows below `1e-15`; strict solve publication succeeds with Jacobian rank
  22 of 25.
- The corrected rounded DM9.2 state has maximum normalized residual
  `1.48660935e-3` at global row R27.
- A 500-iteration source-seeded DM9.2 probe reached
  `4.89350681e-4`, rank 22, and did not publish. This is an observed residual
  floor/model discrepancy, not a solved result.

### Validation

- Focused plasma suite: 32 passed.
- Analytic versus finite-difference 28-by-25 Jacobian maximum absolute
  difference: `5.4511062330675486e-11`.
- Strict manufactured two-start probe: converged at normalized infinity norm
  `9.027914968332446e-17`, rank 22, independent-subspace condition estimate
  `184.61494498692488`; deterministic start 0 selected.
- Corrected DM9.2 rows R08-R11 close within `1.4e-17 A`; thermal rows
  R12-R14 are `0.022194812`, `0.00112256`, and `0.02831104 W`.
- Full repository run reached 673 passes and one skip but had 14 failures and
  10 collection/setup errors in concurrently changing coupling/visualization
  artifacts and manifest contracts. No out-of-scope file was changed.
- Maximal compatible suite excluding only `tests/coupling` and
  `tests/visualization`: 598 passed, one unchanged optional-pybind11 skip.
- Native CTest: 1/1 passed.
- `python -m compileall -q src tests`: passed.
- `git diff --exit-code -- FYP`: passed.
- No packages were installed and no commit or push was performed.

## 2026-09-02 — Final normalization and fixture correction

- Required `Ua` and `Ia` independently to be finite positive normal binary64
  residual scales before any product, bound, residual, or solve construction.
- Added adversarial rejection for subnormal `Ua=1e-310` and
  `Ia=1e-310`, including the large-counterpart cases where `Ua*Ia` would
  otherwise remain normal.
- Preserved the exact minimum-normal current and minimum-normal input-power
  edge.
- Renamed the 2020 source case to `YEO2020-S1-MDO-ORIGINAL`, retained exact
  source label `MDO (original)`, and moved the postprocessing interpretation
  into a separate editorial field.
- Focused plasma: 35 passed.
- Fixture/normal-scale direct probe: passed; stale identifier scans across all
  owned plasma source, tests, specs, and docs returned no matches.
- Stable plasma plus shared-core suite: 68 passed, one unchanged optional
  pybind11 skip.
- A wider compatible run reached 616 passes and one skip with two unrelated
  concurrent failures in fields and validation. Excluding those unstable
  files exposed one further concurrent hybrid custom-species failure after
  582 passes. No out-of-scope files were changed.
- Full compileall passed; native CTest passed 1/1; `FYP` diff was clean.
- No package installation, commit, or push was performed.

## 2026-09-02 — Derived-scale acceptance reconciliation

- Made the edge contract explicit: normal `Ua` and `Ia` are necessary but all
  required derived scales and bounds must also be representable.
- Added a precise typed rejection for `Ua=sys.float_info.min`, whose required
  cathode `Ua^(3/2)` scale underflows, while preserving minimum-normal `Ia`
  when `Ua=1` keeps the input power and all derived values valid.
- Updated the authorized shared physics evidence fixture and its direct test
  to `YEO2020-S1-MDO-ORIGINAL`, source label `MDO (original)`, and separate
  editorial interpretation.
- Focused plasma plus affected physics evidence tests: 38 passed.
- Stable affected/core suite: 71 passed, one unchanged optional-pybind11
  skip.
- Direct edge probe rejected minimum-normal `Ua` with
  `anode_voltage_v is too small for a normal cathode voltage^(3/2) scale`;
  minimum-normal `Ia` with `Ua=1` remained accepted at minimum-normal power.
- Entire-modern stale identifier scan found no `CORRECTED-GLOBAL` ID.
  Remaining “corrected low-fidelity” text occurs only in explicit editorial
  fields and historical validation documentation that identifies it as
  editorial.
- Full compileall passed; CTest passed 1/1; `FYP` diff was clean.
- No installation, commit, or push was performed.
