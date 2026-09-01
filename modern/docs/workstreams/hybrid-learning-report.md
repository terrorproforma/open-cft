# Hybrid Workstream Learning Report

## Retained lessons

- `[user]` Workstream isolation is a correctness requirement. Hybrid code,
  tests, specifications, and notes stay under the four permitted hybrid path
  families; repository-wide memory, shared coupling, fields, PIC, FYP, and Git
  state are not modified.
- `[self]` A macroparticle weight belongs in represented moments, not in
  charge-to-mass acceleration. This keeps trajectories independent of
  numerical sampling weight.
- `[self]` Cell-centred CIC needs a half-cell coordinate shift. Omitting
  `-0.5` changes where endpoint particles deposit even though the global sum
  can still look conservative.
- `[self]` Periodic deposition must preserve `sum S=1` at the endpoint and
  reject genuinely out-of-domain particles. Silent wrapping can conceal a
  missing boundary update.
- `[self]` Random seeds alone do not guarantee deterministic Monte Carlo
  behavior. Keying each draw by particle ID, step, stream, and draw makes
  traversal order irrelevant and gives checkpoint restart semantics.
- `[self]` Charge-exchange marker reset is not conservative by itself. The
  equal-and-opposite neutral-reservoir momentum and energy changes must be
  published alongside the changed ion.
- `[self]` An isothermal quasineutral electron state does not determine an
  electric field or anomalous mobility. Returning `None` is more accurate than
  introducing an unselected closure coefficient.
- `[tool]` The existing environment provides optional Warp float64 execution;
  no installation is needed. CPU and CUDA checks should remain tiny because
  concurrent GPU work makes performance measurements misleading.

## Preflight guardrails used

1. Check the active branch and existing untracked work before writing.
2. Create only paths matching `modern/src/cft_revival/hybrid/`,
   `modern/tests/hybrid/`, `modern/spec/hybrid/`, or
   `modern/docs/workstreams/hybrid-*`.
3. Keep the Python reference free of optional dependencies.
4. Label every synthetic physical input and avoid predictive language.
5. Test conservation by integrating deposited densities over cell volume.
6. Test random statistics and exact keyed repeatability separately.
7. Run FYP and ownership diffs before handoff.

## Open risks

- Constant synthetic cross sections verify machinery but carry no xenon
  predictive validity.
- The neutral reservoir is prescribed, so reported opposite sources are a
  coupling contract rather than a dynamically evolved neutral solution.
- The concrete electron fixture is quasineutral and isothermal; it does not
  close potential, energy, sheath, or anomalous transport.
- Warp atomic deposition can differ in final binary64 summation order; parity
  is therefore tolerance-based while integrated conservation remains the
  primary invariant.

## Session writeback

- Task summary: implemented and verified the isolated prescribed-field hybrid
  first slice, optional Warp kernels, reproducibility contracts, and roadmap.
- `[tool]` Default pytest import mode can collide when concurrent workstreams
  create equal test basenames without package isolation. The evidence-preserving
  workaround is `--import-mode=importlib`; do not edit another workstream to
  hide the collision.
- What worked: small analytic invariants caught trajectory, exchange, and
  normalization errors while keeping both Warp devices below benchmark scale.
- What was unavailable: Ruff and mypy were absent. No dependency was installed;
  compileall, direct line scans, focused tests, and the full behavioral suite
  supplied the available verification evidence.
- Next-session guardrail: recheck concurrent status and the default pytest
  collection path before interpreting a collection failure as a hybrid defect.

## 2026-09-01 audit correction

- `[user]` A Boris velocity update plus new-velocity drift is not a synchronous
  second-order state merely because the velocity rotation is second order.
  The stored time levels must be part of the type and checkpoint contract.
- `[self]` The existing algorithm is correct as leapfrog
  `x^n,v^(n-1/2) -> x^(n+1),v^(n+1/2)`. Correcting the semantics, adding
  explicit half-step initialization/synchronization, and rejecting synchronous
  states avoids replacing a sound kernel with a different integrator.
- `[self]` Electric work for Boris is the sum over both electric half-kicks.
  Half-level kinetic-energy diagnostics and synchronized endpoint work are
  different valid identities and must not be mixed.
- `[user]` Resonant Xe+ charge exchange can keep an ion marker as Xe+, but
  resetting Xe2+ while retaining Xe2+ silently represents the wrong reaction.
  Unsupported reaction products must fail before event sampling.
- `[self]` Counter-keyed events are permutation independent while floating
  reductions are not automatically so. Canonical particle-ID order plus
  `fsum` is required for byte-identical aggregate collision sources.
- `[user]` SHA-256 inside an editable envelope is an integrity check, not
  authenticity. A correctly rehashed mutation is expected to pass unless a
  trusted signature or external digest is introduced.
- `[self]` Complete restart identity includes species identifier, mass, charge,
  charge state, velocity staggering, unique particle ID, and RNG algorithm/
  seed—not only a display symbol.
- Next-session guardrail: adversarially test validly rehashed malformed
  payloads, because digest-mismatch tests alone never exercise schema parsing.
- Verification outcome: 41 focused tests and four explicit Warp CPU/CUDA
  checks passed; compatible default/importlib suites passed after excluding
  only the concurrently inconsistent axisymmetric visualization artifact test.

## 2026-09-02 hardening correction

- `[user]` Calling `float(value)` is coercion, not numeric type validation.
  Schema-number boundaries must first require real numeric primitives and
  reject bool, strings, bytes, or objects that merely implement `__float__`.
- `[user]` Supported xenon charge is an invariant, not restart customization.
  Derive `q=z e`; checkpoint charge is redundant validation data. Only custom
  species identifier and positive finite mass remain variable.
- `[self]` Python `bool` is an `int` subclass. Every identity/counter gate must
  use an exact built-in integer check before range tests, or `True` aliases
  counter value one.
- `[user]` Normal `json.loads` silently keeps the last duplicate member.
  Recursive `object_pairs_hook` rejection must happen before canonical hashing,
  because hashing the collapsed dictionary cannot reveal the duplicate.
- Next-session guardrail: test serialized syntax properties such as duplicate
  members with raw JSON text; constructing a Python dictionary cannot represent
  the defect.
- Verification outcome: both focused modes passed 70 tests; 22 checkpoint/Warp
  tests and four explicit CPU/CUDA checks passed; compatible default/importlib
  suites and full compileall also passed.
