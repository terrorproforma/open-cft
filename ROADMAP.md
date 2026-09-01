# Open CFT Roadmap

This roadmap keeps validation evidence ahead of fidelity or optimization
claims. `FYP/` remains an unchanged historical snapshot throughout.

## Foundation — complete

- Preserve and audit the legacy MATLAB source.
- Verify the cusp kernel across Python, C++17, Warp CPU, and Warp CUDA.
- Establish the SI-explicit Xe/Xe+/Xe2+ L0 reduced-performance model,
  conservation/PPU diagnostics, manufactured uniform-field fixtures, equation
  ledger, checked JSON workflows, and first RTX 5090 batch result.
- Establish immutable multi-fidelity campaign records, constrained Pareto
  semantics, replayable asynchronous scheduling, bounded retry/cost policy,
  guardrails, and deterministic shifted-Halton initial designs.
- Accept independently verified foundations for L1a linear-vacuum
  axisymmetric FDM fields, source-ledgered global plasma numerics, magnetics
  contracts, topology-aware coupling, prescribed-field hybrid kernels,
  reduced electrostatic PIC kernels, surrogate/active-learning tooling, and
  evidence/validation records.
- Publish browser-tested offline viewers for L0 results, geometry v1.1, and
  L1a axisymmetric results, plus an evidence-gated manuscript source whose
  L1--L3 result gates remain closed.

## Accepted versus screening

“Accepted” means the stated numerical or contract gate passed within that
workstream. It does not combine the workstreams into a predictive thruster
model. L1a is FDM, not FEM; the global plasma/hybrid/PIC slices are not
predictively validated L2/L3 CFT models; and the surrogate held-out quality
benchmark did not pass. Material-aware field results remain screening work and
are outside this integration.

## Evidence reconstruction — active

- Source and sign-check the complete plasma/current/energy balance; do not
  translate disputed Kornfeld residuals by inference.
- Define closure provenance and uncertainty for beam current, charge states,
  divergence, cathode, wall, thermalization, and facility effects.
- Acquire versioned experimental and independently simulated observables with
  units, uncertainty, configuration identity, and licensing.
- Freeze objective names and accounting boundaries before mapping any physics
  result to an optimization observation.

## Field and plasma solvers

- Extend the accepted manufactured-solution L1a FDM solver beyond
  constant-permeability equivalent-current finite-box problems.
- Reproduce linear-material FEMM profiles on identical meshes and domains
  before selecting MFEM/CUDA or Warp FEM.
- Add nonlinear material and open-boundary models only after linear parity.
- Preserve the accepted corrected global plasma residual/Jacobian foundation,
  while withholding predictive CFT claims until equation authority, closures,
  calibration, and independent validation are complete.

## Calibrated multi-fidelity campaign

- Define typed F0/F1/F2/F3 source subtypes and result-context identities.
- Measure source costs and calibrate model discrepancy and predictive coverage.
- Runtime-verify pinned BoTorch/GPyTorch and pymoo boundaries in an isolated
  environment; keep the dependency-free scheduler usable without them.
- Freeze objective normalization and an F3 reference point, then implement
  verified hypervolume.
- Compare acquisition strategies at common equivalent-F3 cost, common initial
  observations, and common highest-fidelity verification.

## Validation and release gates

- Validate against independent thrust, current, Faraday/RPA/LIF, field,
  thermal, and uncertainty data over a declared applicability domain.
- Benchmark CPU/GPU only with controlled load, warm-up, synchronization,
  repeated runs, dispersion, and kernel-only plus end-to-end timings.
- Require reproducible environments, signed/protected campaign artifacts,
  licensed evidence, clean secret/path scans, and an unchanged `FYP/` diff.
- Do not make hardware-design or measured-accuracy claims until independent
  domain review accepts the full evidence chain.
