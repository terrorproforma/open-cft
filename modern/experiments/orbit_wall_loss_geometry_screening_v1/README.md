# Orbit wall-loss geometry screening v1

**Classification: `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`.** Full-orbit
collisionless test-particle electron wall-loss probabilities (orbit_mc v1.7,
numpy CPU) integrated in the **accepted L1a linear-vacuum equivalent-current
fields** of the geometry sweep v2 (`modern/experiments/l1a_geometry_sweep_v2`),
for the 25 non-dominated designs plus the remaining 71 accepted designs (96 in
one execution). The fields are L1a screening maps, **not P2-qualified**, so no
number produced here is accepted physical-orbit evidence and none is a plasma
or performance claim. The dataset exists so that (a) the orbit_mc coupling
export format has an actual consumer, (b) the surrogate stack has physically
varying training labels, (c) the MDO's cusp/wall-loss input can become
design-dependent — each use must carry the screening label.

## Field provenance

Only the four sweep-v2 representatives have stored full-field maps. Every design
is therefore rebuilt with the sweep's own `build_case` and re-solved with the
accepted L1a CPU solver at the sweep protocol resolution (`designs.py`).
Identity is proven before any orbit: rebuilt geometry/source/config/case hashes
equal the sealed `raw-results.json` record, re-solved QoIs reproduce the recorded
QoIs within the sweep's preregistered replay tolerances, and the representatives
reproduce their stored maps node-wise (≤1e-15 Wb, ≤1e-9 T). A 2× refined re-solve
is a field-resolution diagnostic for every design (cross-resolution B rms ≤ 5 %)
and the representatives additionally run a `refined-N` orbit case. The solver
version and inputs are bound by `field_pipeline_source_sha256` (LF bytes of
`cft_revival/{fields,geometry,magnetics,optimization}`, `spec/fields`, and the
sweep's `experiment.py`/`protocol.py`/`protocol.json`).

## Launch design (per design)

Four axial cells at fractions 1/8, 3/8, 5/8, 7/8 of the channel-straight span
`[injector_length, exit_start]` × two radii (0.675, 0.800 of the wall radius) ×
energies {5, 25} eV × pitches {20°, 70°} × directions {−1, +1} × 8 gyrophases
(offset 11π/96) = 512 launches, 32 strata, equal weights, Wilson intervals;
timestep policies N (0.16 rad) and 2N (0.08 rad). Wall-hit authority is the
straight dielectric `0 ≤ z ≤ exit_start` (as v4); radial exit into a divergent
section is `domain_escape` (sub-class `divergent_section_radial`), `z = 0` is
`upstream_anode_plane`, `z = L` is `exit_plane`. `max_path = 2 L`,
`max_time = 2 · max_path / v(5 eV)`.

## Gates (per design)

Binding for the dataset: field adapter (interpolation ≤ 5 %, cross-resolution
≤ 5 %), `preflight_campaign`, zero numerical failures, energy drift ≤ 1e-10
(exactly 0.0 under v1.6+), final velocity = event velocity, wall endpoint
≤ 1e-8 m, earliest-event ordering, runtime rotation bound, exact deterministic
replay (write + verified reload), cross-process determinism sample, handoff
consumed. Per-design flag: `|P_wall(2N) − P_wall(N)| ≤ 0.02` with Wilson overlap
(`converged`); a non-converged design is reported through its summaries and
endpoint tables but its orbit artifacts are **not sealed** (the orbit_mc
convergence-evidence contract requires the flag), which is recorded. Timeouts
are reported, not gated. μ variation is a diagnostic, never a gate.

## Lifecycle

```
python -m experiments.orbit_wall_loss_geometry_screening_v1.run shakedown  # 3 designs x 64, temp root
python -m experiments.orbit_wall_loss_geometry_screening_v1.run prepare    # refuses without a valid shakedown.json
# commit "preregister orbit wall-loss geometry screening v1", push, then from a clean detached worktree
python -m experiments.orbit_wall_loss_geometry_screening_v1.run execute
python -m experiments.orbit_wall_loss_geometry_screening_v1.run validate
```

`prepare` binds the protocol hash, the orbit_mc source hash, the field-pipeline
source hash, the shakedown record and the extension decision
(`timing_projection.within_budget`); `execute` re-verifies all of them, requires
a clean detached pushed preregistration commit and takes a Git-common O_EXCL
lock. Cases run in a 12-worker spawn pool (≤ 12 of 24 CPUs; the GPU is not used).

## Outputs (`results/artifacts/`)

`geometry-wall-loss-dataset.json` and `.csv` (per design: identities, geometry,
field evidence, per-stratum/per-cell counts, wall-hit/escape/reflection/timeout
probabilities with Wilson intervals at N and 2N, convergence flag, gates,
diagnostics), `coupling-consumer-record.json` (formal consumer of the v4 export
format; the v4 divergent-exit design is absent from the screening set and is
carried as a labelled reference row), per-case summaries, gzipped endpoint
tables, orbit-artifact sidecars (full artifacts for the representatives),
handoffs, per-design bore fields and field evidence, `design-exclusions.json`,
`gates.json`, `campaign-result.json`.

Dashboard: `modern/visualization/wall-loss-geometry-screening-v1.html`.
