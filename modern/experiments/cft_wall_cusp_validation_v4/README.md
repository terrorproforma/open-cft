# Coupling wall-cusp held-out validation v4

This experiment is the one-shot held-out numerical/source-consistency
validation of coupling record v4.2 using canonical field artifacts v1.2.

The eight-case geometry family is disjoint from development and every case
accessed by failed attempts v1-v3. Those failures remain immutable. Thresholds
are frozen in `protocol.json` from manufactured and development-only evidence;
prior held-out outcomes are not tuning inputs.

All result roots, locking, cache management, atomic artifact/sidecar writes,
counters, access records, terminal state, and bundle validation are owned by
`cft_revival.experiment_runtime`. The launcher accepts one clean detached
execution:

```powershell
python -m experiments.cft_wall_cusp_validation_v4.run
```

Every map follows solve → canonical v1.2 bytes → atomic persistence → exact
byte reload → coupling verification. Orbit evidence uses a map-sampled
full-particle Boris trajectory and reports magnetic moment, kinetic energy,
pitch, mirror state, and timestep convergence. Polyline path-length convergence
is an independent gate.

Promotion requires every preregistered case and projection gate. It does not
claim hardware validity, experimental truth, or plasma performance.
