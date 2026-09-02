# L1a geometry design-space sweep

This experiment runs 96 deterministic shifted-Halton cases through the accepted
geometry v1.1 non-authoritative current-equivalent preview and accepted L1a
axisymmetric Warp solver.

From `modern/`:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.l1a_geometry_sweep.run
```

The fixed 80-by-144 domain, binary64 solver tolerances, variables, field-only
objectives, constraints, failure taxonomy, and QoI formulas are declared in
`experiment.py`. Failed cases never receive objective values or numerical
penalties and are excluded from ranking.

The accepted optimization sampler is reused. Ranking is experiment-local exact
constrained dominance because accepted campaign fidelity labels include plasma
or corrected analytical semantics and therefore do not accurately label this
L1a-only experiment.

`results/dataset.json`, `results/manifest.json`, representative geometries, and
representative full/downsampled L1a artifacts are hash sealed. Wall times are
isolated in `results/runtime-diagnostics.json`; concurrent GPU load makes them
uncontrolled diagnostics, not benchmark evidence.

No thrust, efficiency, plasma, thermal, or structural result is computed.
Nothing in this experiment is hardware-valid, material-aware, FEM, optimized
for construction, or build-qualified.
