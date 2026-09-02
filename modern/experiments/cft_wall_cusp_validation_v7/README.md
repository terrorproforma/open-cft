# Coupling wall-cusp held-out validation v7

This is a one-shot held-out numerical/source-consistency validation of coupling
record v4.2 with canonical field artifacts v1.2. The eight new five-stage cases
use 5.8/7.2 mm pitch and 11.5/11.9 mm chamber radius. Their family, IDs, and
coordinates exclude all v1-v6 access. Every case retains six required wall
cusps and five inter-cusp cells.

V6 result `876c3a9d` is disclosed negative development evidence. It motivates
the prospective orbit algorithm correction but is not used to tune v7 samples
or thresholds; those are frozen from manufactured and pre-existing development
evidence before any v7 held-out access.

Patched runtime `b46e263` owns all roots, locking, cache management, atomic
artifact/sidecar writes, globally sorted inventory, counters, access records,
terminal state, and bundle validation. Result files use `* -text`. The launcher
accepts one clean detached execution:

```powershell
python -m experiments.cft_wall_cusp_validation_v7.run
```

Every map follows solve → canonical v1.2 bytes → atomic persistence → exact
byte reload → coupling verification. Orbit evidence derives Br/Bz from one
interpolated ψ potential, including a regular-axis limit and an exact
within-cell cylindrical-divergence identity. A midpoint-field Boris solve uses
nested N/2N/4N step counts with `dt=T/N` at one exact physical terminal time.
The fastest declared or path-encountered cyclotron frequency selects the step
scale.

Numerical trajectory convergence is independent of physical adiabaticity.
Nonconvergence is `ORBIT_UNVERIFIED`; a converged orbit that fails
gyro-averaged μ, rho/LB, or field-line-curvature ordering is `NONADIABATIC`.
Diagnostics preserve instantaneous and gyro-averaged μ, energy drift,
phase-aligned terminal/trajectory error, mirror detection, and ordering metrics
for all three maps even when a threshold fails. Uncertainty-not-evaluable and
invalid finite uncertainty bounds are separate outcomes.

Before held-out access, uniform, gradient, and mirror ψ fields verify the axis,
divergence, nested-time, invariant, ordering, and mirror paths. A manufactured
development case then exercises the production policy and adapter through all
three maps and a coupling v4.2 record. Static/runtime checks reject implicit
policy defaults, legacy reloads, migration metadata, or changed bytes.

Callback payloads are plain domain JSON and are canonicalized exactly once by
`RunContext.write_json`. Before held-out access, the actual callback writer
serializes resolved, ambiguous zero-cell/orbit, nonempty boundary-null, and
complete assessment-rejection summaries. A cusp-count rejection writes its
topology and outcome atomically and returns `Decision(False)`.

The runtime alone validates and publishes the terminal bundle; the launcher
does not perform redundant post-publication validation. `.gitattributes` is an
approved binary-line-ending placeholder on every validation path. Terminal
claim text is conditional on actual promotion. Promotion requires every
field/freshness/replay/six-cusp/wall/path/axial-core/orbit/uncertainty/projection
gate and does not claim hardware validity, experimental truth, or performance.
