# Orbit Monte Carlo devlog

## 2026-09-02 — full-orbit foundation

- Added an isolated `cft_revival.orbit_mc` package without modifying active
  coupling, experiment, hybrid, PIC, field, shared, FYP, or Git paths.
- Added immutable SI electron launches over energy, pitch, position/flux
  surface, both directions, deterministic gyrophase, seed and launch identity.
- Added C1 tensor Hermite ψ interpolation. Br and Bz are derivatives of the
  same interpolated ψ; the axis uses the regular second-radial derivative.
- Added a float64 CPU relativistic momentum-Boris reference with one
  conservative global-max-B timestep, analytic cylindrical first events,
  confirmed bounce detection, physical time/path limits, gamma guard, and
  complete failure taxonomy.
- Added accumulated local gyro phase, complete-cycle-only μ averages,
  instantaneous μ and energy diagnostics, wall endpoint, transit fraction,
  and timestep/backend provenance.
- Added deterministic batched reduction, Wilson 95% intervals, explicit wall,
  reflection, escape and incomplete probabilities, and an adiabatically gated
  non-authoritative loss-cone comparator.
- Added canonical hash-sealed result artifacts, atomic hash-chained
  checkpoints, strict top-level schemas, and a coupling-v4.2/plasma-network
  handoff protocol.
- Added optional Warp float64 CPU/CUDA full-orbit execution on the installed
  Warp 1.14 runtime: the relativistic push runs on the selected device while
  the identical audited interpolation/event loop remains on the host. No GPU
  throughput benchmark or uncontrolled campaign was run.

## Verification evidence

- Manufactured ψ interpolation: node ψ max error `0`; Br and Bz max errors
  `4.441e-16 T`; vector-B RMS relative error `2.790e-16`.
- Uniform-B three-cycle helix at 128 steps/cycle: position error
  `5.525e-6 m`, velocity error `1.943e4 m/s`, relative energy drift `0`, and
  accumulated-phase error `8.527e-14 rad`.
- N/2N/4N helix position errors were `2.208e-5`, `5.525e-6`, and
  `1.382e-6 m`; observed orders were `1.9984` and `1.9996`.
- Uniform-E relativistic momentum relative error was `5.732e-15`.
- Analytic divergence-free magnetic bottle: reflection-point relative error
  `1.855%`; instantaneous μ relative variation `0.667%`.
- First-order grad-B guiding-centre drift at `rho/L_B=0.00337`: correct
  electron drift sign and `0.884%` relative magnitude error.
- Linear cylindrical wall event: fraction and endpoint errors were exactly
  zero in binary64 for the manufactured case.
- Warp CPU and RTX 5090 CUDA float64 one-step parity both had zero observed
  velocity difference against the NumPy reference.

## Verification commands

```text
$env:PYTHONPATH='src'; python -m pytest tests/orbit_mc -q
python -m compileall -q src/cft_revival/orbit_mc tests/orbit_mc
python -c "import json,pathlib; [json.loads(p.read_text()) for p in pathlib.Path('spec/orbit_mc').glob('*.json')]"
git diff --check -- modern/src/cft_revival/orbit_mc modern/tests/orbit_mc modern/spec/orbit_mc modern/docs/workstreams/orbit-mc-*
git diff --exit-code -- FYP
```

The initial focused verification passed `21/21`. The handoff had not exercised
a public consumer and therefore provided no coupling/plasma-network
integration evidence.

## 2026-09-02 — audit hardening

- Replaced post-step event checks with a common candidate-fraction ordering
  for wall, domain, exact time, exact remaining path, and a 52-iteration
  \(v_\parallel=0\) reflection root. Deadline/path events now beat later wall
  intersections and retain no measurable manufactured overshoot.
- Replaced direct ψ-axis division with Hermite interpolation of
  \(g=(\psi-\psi_\mathrm{axis})/r^2\), even-axis extrapolation, and explicit
  homogeneous-plasma material-cell quarantine.
- Converted every traversable Hermite cell to power coefficients and used
  absolute polynomial/derivative coefficient sums as a conservative complete
  cell \(|B|\) certificate. Runtime max-B and relativistic rotation checks
  remain active. Inconsistent Br/Bz references and underdeclared reference
  maxima fail.
- Corrected accumulated phase to \(\int |q|B/(\gamma m)\,dt\). A gamma-two
  particle completes one reported cycle in one physical orbit.
- Added predictor/midpoint field sampling around the relativistic Boris push.
  Spatially varying-E position errors were `4.223e-12`, `1.053e-12`, and
  `2.601e-13 m`, with observed orders `2.0042/2.0171`.
- Added typed invalid-launch outcomes, exact reflection endpoint evidence,
  strict scalar/boolean rejection, finite positive custom Wilson-z validation,
  full result identity replay, exact Wilson replay, campaign/launch/result
  identity checks, closed hash-authoritative checkpoints, and externally
  anchored checkpoint reload.
- Relabeled the v4.2/plasma-network object as
  `export_only_pending_consumer_integration`; no exercised consumer is
  reported.

Updated manufactured errors: ψ node `3.469e-18 Wb`, Br/Bz maxima
`2.637e-16/3.331e-16 T`, B relative RMS `2.259e-16`; helix orders
`1.9984/1.9996`; mirror-point error `1.883%`; μ variation `0.692%`; grad-B
drift error `0.882%`; uniform-E momentum error `5.732e-15`; exact wall and
reflection-root residuals `0`.

Final verification passed `42/42` orbit tests. Adding unchanged coupling-schema
and plasma-network identity regression tests passed `57/57`; those adjacent
tests do not consume the export-only handoff. Warp CPU and RTX 5090 CUDA
float64 parity both reported zero observed velocity difference. Compileall,
four JSON specification parses, trailing-whitespace scans, and the FYP no-diff
guard passed. Ruff remained unavailable and was not installed. No commit,
physical CFT campaign, or wall-loss probability was produced.

## 2026-09-02 — evidence authority closure

- Bumped result and checkpoint contracts to `1.1.0`. Orbit records now retain
  configured time/path limits and event tolerance, enabling runtime replay of
  deadline, path, step/time, wall-endpoint, energy, phase/cycle, transit, μ,
  vector, type, range, and closed-enum semantics.
- Checkpoints now embed the complete immutable launch manifest and bind a
  campaign identity to its launch hash. Validation and loading require the
  externally trusted campaign ID and launch hash; missing, duplicate, forged,
  or substituted launches fail even after all internal hashes are recomputed.
- Added exact closed-contract checks for summary probability/count ranges,
  integrity algorithms, nonempty limitations, and field-evidence scalar types.
- Added deterministic certificate diagnostics for dense sampled maximum /
  certified bound. A preregistered ratio below `0.001` fails preflight as
  `NOT_EVALUATED`; the certified bound remains the safety authority.
- Added 25 coherently rehashed artifact/checkpoint tamper cases covering forged
  IDs and terminations, missing/duplicate launches, event/phase/state lies,
  seed/gyrophase/direction identity defects, transit/μ/range/type/unknown-key
  defects, and certificate evidence.
- Final focused verification passed `70/70`; compileall, all four JSON parses,
  FYP no-diff, owned-path whitespace, Warp CPU, and RTX 5090 CUDA smokes passed.

## 2026-09-02 — v1.2 software acceptance

- Added closed final-step witnesses for all termination classes. Runtime
  validation now replays wall/domain geometry, deadline/path candidate
  fractions, reflection brackets, earliest-event priority, endpoint/counters,
  step limits, gamma thresholds, and field/config/policy identities.
- Added optional deterministic full-result replay against bound launch,
  `OrbitConfig`, and field objects. A rehashed but plausible changed reason is
  accepted by structural semantics and rejected by deterministic replay.
- Made checkpoints partial-first: all launches remain immutable authority while
  results exactly cover complete frozen batches plus an optional prefix of one
  current batch. Pending IDs and counters are recomputed complements.
- Added normalized positive launch weights and exact order to the frozen batch
  manifest. Resume merge verifies ancestry, monotone completed batches, no
  dropped results, and byte-identical prior evidence.
- Bound the certificate-tightness floor and protocol identity externally.
  Rehashed floor weakening below `0.001` fails in artifacts and checkpoints.
- Bumped result/checkpoint schemas to `1.2.0` and expanded the coherent-rehash
  matrix for valid-enum event substitution, malformed batches, partial chains,
  resume/drop behavior, and policy-floor weakening.
- Final v1.2 verification passed `86/86`. Wall, domain, path, step, reflection,
  and extreme-gamma witness smokes passed. Warp CPU and RTX 5090 CUDA parity
  remained exact in the observed vectors; compileall, four spec parses,
  owned-path whitespace, and FYP no-diff guards passed.

## 2026-09-02 — v1.3 verified estimator boundary

- Added the closed `UNWEIGHTED_BINOMIAL` estimator policy. Batch-manifest
  weights must all equal \(1/N\) and sum to one; unequal `0.99/0.01`-style
  designs and unsupported weighted/stratified policies fail closed.
- Bound estimator policy and partition-independent estimator identity into
  artifact, checkpoint, batch, and campaign authority. Equal-weight input
  permutations retain one identity.
- Made deterministic replay mandatory for artifact publication.
  `write_artifact` now requires actual field/config objects plus external
  field/config/launch/policy authority and returns opaque verified evidence.
- Split loading into structural `load_artifact`, which returns a non-mapping
  `UnverifiedOrbitArtifact`, and `load_and_verify_artifact`, which replays and
  returns `VerifiedOrbitEvidence`. Only the latter can feed coupling handoff.
- Added coherent false extreme-relativity/reflection rejection before write
  and after structural load, structural-token coupling rejection, replay
  bypass tests, unequal-weight tests, and estimator permutation/identity tests.
- Bumped result/checkpoint/validation contracts to v1.3 and the export-only
  handoff to v1.2 with explicit replay and estimator status.
- Final focused verification passed `92/92`.

## 2026-09-02 — v1.4 batch authority closure

- Added mandatory external `expected_batch_manifest_sha256` to deterministic
  replay, artifact write, verified load, checkpoint construction/finalization,
  and verified coupling handoff.
- Every boundary compares that authority exactly before publishing bytes or
  creating a verified capability. The coupling export now carries the checked
  batch hash.
- Added a coherent one-batch-to-two-batch repartition attack. Structural
  rehashing remains valid, but replay and checkpoint finalization reject it
  against the original external hash. Canonically sorted equal-weight input
  permutations retain the exact frozen manifest/hash.
- Bumped result/checkpoint/validation contracts to v1.4 and handoff to v1.3.
- Final focused verification passed `93/93`.
