# Learning Scratchpad

This is the running evidence/decision ledger for the rebuild.

## Established

- Repository contains only 16 MATLAB files; no datasets or external optimizer.
- Production path is optimizer → geometry checks → FEMM → cusp extraction →
  30-variable least squares → performance post-processing.
- `Ua`/`Ia` local/global split and logical least-squares constraints invalidate
  confidence in archived solver results.
- The cusp angular integral has a simple, defensible closed form suitable for
  a first C++ parity kernel.
- FEMM automation is explicitly known by the original author to conflict under
  multiple instances, despite optimizer parallelism being set to 12.
- ISTS 2017-b-32 confirms three publication objectives, 100 generations,
  mirror-angle physics, and performance post-processing, but the available
  code snapshot contains four objectives and 50 generations.
- The paper claims eight variables but reports only five, so the publication
  run cannot be reconstructed from Table 4.
- The paper reports that no surrogate met its 5% criterion and nevertheless
  presents surrogate-based Sobol results; these results cannot be a
  correctness oracle.
- Warp 1.14.0 can target both CPU and RTX 5090 `cuda:0` through its CUDA 12.9
  toolkit without a standalone `nvcc`.
- The initially translated form `0.5*(1-sqrt(1-r))` is numerically invalid for
  tiny mirror ratios: at `r=1e-18`, subtraction rounds the result to zero.
  Rationalization gives `0.5*r/(1+sqrt(1-r))`, preserving approximately
  `2.5e-19` while retaining exact `r=0` and `r=1` endpoints.
- IEEE `-0.0` passes the valid `B_low >= 0` contract but preserves its sign
  through multiplication/division. Every backend now branches only on exact
  zero and returns canonical `+0.0`; positive tiny and subnormal inputs remain
  on the rationalized path.

## Decisions

- Keep `FYP/` untouched.
- Name the new tree `modern/`.
- Use standard-library-only Python at runtime; optional pybind11/CMake native
  build.
- Preserve units in type names and serialized keys until a units package is
  justified.
- Read FEMM exports but do not automate FEMM in-process.
- Quarantine the plasma solve instead of encoding uncertain physics.
- Use Warp now only for verified batched kernels; require manufactured
  solutions and FEMM profile parity before calling a field solver a
  replacement.
- Apply one shared field-domain contract before Python, C++, or Warp dispatch,
  and test numerically difficult regimes rather than relying only on
  normal-scale random samples.

## Open questions

- Which Kornfeld et al. source equations define all 33 executable continuous
  residuals and their signs? ISTS 2017-b-32 defers the full set to that paper.
- Is the executable `+CE` sign deliberate or a transcription error?
- What physical definition justifies the cusp-4 high/low field swap?
- Are FEMM plot values already magnitudes, radial components, or another plot
  type for `mo_makeplot(1,...)` in the installed FEMM version?
- What convention does the absent optimizer use for `g` feasibility?
- Which historical cases have independently measured thrust/efficiency/Isp?
- Are multiply charged xenon ions intended in the 1.2 mass-utilization cap?

## Next evidence to acquire

- Kornfeld source paper and Keller thesis;
- exact publication-run source revision and all eight S1-S4 variables;
- external optimizer and surrogate library versions;
- FEMM version/material library;
- original EVS/ND/sensitivity files and solver outputs;
- machine-readable experimental baseline with uncertainty.

## 2026-09-01 public-release guardrails

- [user] Keep `FYP/` as unchanged historical evidence; publication preparation
  may add metadata and modernize licensing but must not clean up legacy
  whitespace or line endings.
- [tool] Candidate discovery must use Git's ignore-aware file list. This tree
  also contained unlicensed PDFs, presentation decks, AppleDouble `._*` files,
  and generated build/cache content that ordinary source globs did not reveal.
- [self] Use PowerShell-compatible exit checks in this Windows host; `&&` is
  not accepted by its parser.
- [self] Before a public push, inspect the exact staged list, scan candidates
  for credentials and user-home paths, and verify that generated scientific
  results and locally retrieved papers remain ignored.

## 2026-09-01 integrated physics and optimization guardrails

- [user] Preserve `FYP/` byte-for-byte while integrating shared package, CLI,
  documentation, and campaign surfaces.
- [self] L0's charge-state, beam-current, divergence, cathode, and PPU inputs
  are externally supplied. Conservation closure and CPU/CUDA parity justify
  implementation-consistency claims only, never measured predictive accuracy.
- [self] The campaign's historical `total_efficiency` objective is not
  interchangeable with L0's explicitly bounded anode-to-beam,
  thruster-electrical-to-beam, or PPU-input-to-beam efficiencies. Do not map
  them without a reviewed accounting boundary and uncertainty contract.
- [self] Charge-state current/energy uses charge number `z`; momentum uses
  common xenon mass and scales with `sqrt(z)`. Divergence changes axial
  momentum, not kinetic beam power.
- [self] Binary64 edge contracts are part of the model: exact-rational
  charge-fraction admission, canonical PPU snapping, `None` for represented
  `0/0` efficiencies, exponent-separated products, and finite-publication
  checks must remain shared across Python and Warp.
- [self] Optimization identities must include schemas and result context.
  Failed solves remain typed failures; retries require terminal retryable
  failures and paid F3 attempts; pre-execution rejection is a separate
  zero-cost, non-attempt event.
- [self] Pareto comparison is valid only inside one comparable context.
  Pending jobs block duplicates, fidelity budgets distinguish committed from
  charged cost, and F3 attempt/success/failure counts remain separate.
- [self] Shifted Halton must never be relabelled Sobol. Verified hypervolume
  requires a frozen F3 reference/normalization and cannot be inferred from
  surrogate outcomes.
- [tool] Warp 1.14.0 executes the full float64 L0 batch on RTX 5090 `cuda:0`,
  but varying shared GPU load and transfer-inclusive one-shot timings are not
  controlled benchmark evidence.
- [self] A campaign-spec validation-only request exposed that
  `initial_designs(..., count=0)` did not return an empty design. Keep the
  explicit zero-count regression.

Detailed derivations and review history remain in
`docs/workstreams/physics-learning-ledger.md` and
`docs/workstreams/optimization-learning-ledger.md`.
