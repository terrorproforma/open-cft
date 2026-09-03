# Plasma Network Development Log

## 2026-09-03 — Isotonic projection; inherited global-row inconsistency

- `_projection` uses `cft_revival.plasma.project_nondecreasing`
  (pool-adjacent-violators) instead of `sorted()`, which permuted potential
  identities and stalled the accepted four-cell solver at 1000 V
  (`global-plasma-closure-analysis.md`, Section 7). Seeds from
  `deterministic_initial_states` are unchanged.
- Documented that the N=4 network inherits the global-row inconsistency for
  interior cusp loss (no admissible root for `p1..p3 > 0`; anode-only loss
  publishes under `REPRESENT_NULLSPACE` at `phi_N = Ua`). New tests
  `tests/plasma_network/test_plasma_network_closure_p_nonzero.py`.
- Validation: `tests/plasma_network` 66 passed.

## 2026-09-02 — Topology-general solver

### Implemented

- Added an ordered-chain topology contract for `N>=1`, geometry-classified
  interior cusps, explicit terminal boundaries, finite-boundary-null
  exclusion, uncertainty fields, and SHA-256 provenance.
- Added dynamic state/residual layouts (`6N+1`, `7N`), SI unit names,
  explicit electron/ion/cusp orientation, normalized balance rows, separate
  inequalities, named power terms, and a generated source/equation ledger.
- Added exact forward-mode Jacobians, bound-aware finite differences,
  homogeneous batch evaluation, scaled rank/nullspace diagnostics, and
  rank-gated publication.
- Reused the accepted deterministic scaled pivoted-QR LM implementation
  through a backend protocol and added deterministic topology-general
  multi-start retention.
- Added exact manufactured zero-cusp cases and an N=4 compatibility adapter.
- Added machine-readable equation-ledger schema and topology contract plus
  formulation, integration, learning, and development documentation.

### Verification evidence

- Focused plasma-network plus accepted plasma compatibility suite: 89 passed.
- Manufactured N=1...6 normalized residual maxima:
  `1.36e-16`, `1.26e-16`, `1.99e-16`, `2.03e-16`, `1.15e-16`, `1.45e-16`.
- Scaled numerical ranks N=1...6: `7`, `12`, `17`, `22`, `27`, `32`;
  nullities are `0`, `1`, `2`, `3`, `4`, `5`.
- Accepted N=4 corrected residual: all 28 raw and normalized rows are bit
  exact for equivalent state/input/branch.
- Full compatible collection reached 989 passes and one optional-extension
  skip, with 10 unrelated concurrent failures: seven in an incomplete
  four-cell-topology result bundle, two L0-v3 preflight path-state failures,
  and one material-fields fixture/schema mismatch. No failing path was edited.
- Maximal compatible suite excluding only those three failing files:
  975 passed and one unchanged optional-extension skip.
- Full `compileall` passed, all owned JSON parsed, and the `FYP` diff was clean.
- Ruff and mypy were not installed; neither was installed.

### Deliberate limits

- No accepted plasma, coupling, shared, experiment, material-field, FYP, Git,
  package configuration, dependency, commit, or install path was changed.
- No physical accuracy, validation, GPU speed, or predictive claim is made.

## 2026-09-02 — Publication and identity audit hardening

### Corrected

- Treat every nonlinear backend return as an untrusted candidate. Canonical
  shape/finiteness, bounds, inequalities, raw/normalized balances,
  conservation, analytic Jacobian, rank, and nullspace are recomputed before
  publication; backend status and residual claims are diagnostic only.
- Replaced bound-dependent rank scales with fixed SI scales: `Ua` for
  potentials/temperatures and `Ia` for all current blocks.
- Added recorded rank, nullspace residual, orthonormality, and conservation
  tolerances. Pivoted QR must recover generated structural rank `5N+2`;
  unexpected extra deficiency fails closed.
- Generated dimensionless null vectors are orthonormalized and independently
  checked for finite shape/count, independence, and canonical scaled `Jv`.
- Expanded immutable topology semantics to positions, confidence, full loss
  covariance, excluded-null reason/provenance, and geometry/material/source/
  artifact/model/code/schema hashes. Canonical JSON hashes every nested field.
- Added strict `PlasmaChainTopology.__post_init__` validation and repeated it
  at input, ledger, and solver boundaries.
- Renamed all owned tests to unique `test_plasma_network_*` basenames.

### Adversarial evidence

- A malicious backend reporting convergence and zero residual for a candidate
  with canonical residual exactly `0.426` is rejected.
- An exact candidate accompanied by backend failure and residual `99` is
  accepted after canonical verification, proving backend claims are ignored.
- Wrong dimensions, NaN/Inf, loose bounds, invalid/duplicate null vectors,
  non-orthonormal bases, extra numerical deficiency, bound violations,
  inequality violations, and topology replacement/bypass attempts fail closed.
- Focused plasma-network plus accepted N=4 suite: 100 passed before the final
  semantic-identity assertion expansion; exact parity remains bit-for-bit.
- Default-compatible suite: 969 passed, one unchanged optional-extension skip,
  and two unrelated L0-v4 state-dependent tests deselected.
- Importlib-compatible suite: 1024 passed, one unchanged optional-extension
  skip, and the same two L0-v4 state-dependent tests deselected.
