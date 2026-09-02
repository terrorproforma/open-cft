# CFT full-orbit wall-loss campaign v3

This directory preregisters nine explicit collisionless prescribed-field
test-particle wall-loss cases using the independently
`NUMERICAL_P2_QUALIFIED` divergent-exit P2 evidence. Historical-envelope and
compact-high-gradient designs are explicitly excluded because they remain
screening-only.

The experiment-local adapter verifies the committed P2 manifest, result,
checkpoint, sidecar, mesh, run and payload hashes. It samples quadratic
`A_phi` evidence into regular ψ grids over the homogeneous plasma subdomain,
quarantines non-plasma triangles, and measures withheld midpoint field error.

The physical wall estimand is restricted to the straight cylindrical
dielectric section (`r=2 mm`, `1<=z<=18 mm`). In the divergent section,
crossing the conservative `r=2 mm` plasma subdomain is reported as domain
escape, not wall impact. This avoids pretending the accepted cylindrical orbit
event contract implements a sloped wall.

From `modern/`, the controlled lifecycle is:

```powershell
$env:PYTHONPATH='src;.'
python -m experiments.cft_orbit_wall_loss_v3.run prepare
# commit and push the preregistration, then execute only from a clean detached worktree
python -m experiments.cft_orbit_wall_loss_v3.run execute
```

Each primary/refined/enlarged × N/2N/4N case has a case-equal
`ensemble_id`/`campaign_id`, 512 case-prefixed launch IDs, eight batches and
independent launch/batch/estimator/field/config/policy/case hashes. The physical
stratum design is common across cases, but its positions and gyrophases are
fresh relative to v2. The production preflight proves zero v2 identity, seed,
position and phase-space overlap, then validates partial-32, resumed-64 and
final-512 checkpoint chains for all nine cases before P2 or outcome access.

There is one immutable attempt. The shared `experiment_runtime` owns the result
bundle, access-before-label records, atomic artifacts, terminal state and
manifest. A retained Git-common lock prevents another execution. Coupling is
export-only and is emitted only after every preregistered numerical and exact
replay gate passes.

No result is PIC, self-consistent plasma, experimental validation, hardware
validation, total thruster performance, or mirror-formula publication.
