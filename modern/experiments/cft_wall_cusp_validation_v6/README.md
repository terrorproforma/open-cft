# Coupling wall-cusp held-out validation v6

This experiment is the one-shot held-out numerical/source-consistency
validation of coupling record v4.2 using canonical field artifacts v1.2.

The eight-case five-stage geometry family is disjoint from development and
every case and coordinate accessed by failed attempts v1-v5, including
`wcval-v5-s05-p0-r0-neg`. Every case retains the frozen requirement of six wall
cusps and five inter-cusp cells. Prior held-out outcomes are not tuning inputs.

Patched runtime `b46e263` owns all roots, locking, cache management, atomic
artifact/sidecar writes, globally sorted inventory, counters, access records,
terminal state, and bundle validation. Result files use `* -text`. The launcher
accepts one clean detached execution:

```powershell
python -m experiments.cft_wall_cusp_validation_v6.run
```

Every map follows solve → canonical v1.2 bytes → atomic persistence → exact
byte reload → coupling verification. Orbit evidence uses a map-sampled
full-particle Boris trajectory and reports magnetic moment, kinetic energy,
pitch, mirror state, and timestep convergence. Polyline path-length convergence
is an independent gate.

Before held-out access, a manufactured development case exercises the actual
production policy and adapter through all three maps and a coupling v4.2
record. Static and runtime checks reject implicit `MapValidationPolicy()`
defaults, legacy reloads, schema normalization, migration metadata, or changed
bytes.

Callback payloads are plain domain JSON and are canonicalized exactly once by
`RunContext.write_json`. Before held-out access, the actual callback writer
serializes resolved, ambiguous zero-cell/orbit, nonempty boundary-null, and
complete assessment-rejection summaries. A cusp-count rejection writes its
topology and outcome atomically and returns `Decision(False)`.

Promotion requires every preregistered case and projection gate. It does not
claim hardware validity, experimental truth, or plasma performance.
