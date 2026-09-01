# Global Plasma Workstream Development Log

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
