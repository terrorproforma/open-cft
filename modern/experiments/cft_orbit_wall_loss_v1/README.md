# CFT full-orbit wall-loss campaign v1

This directory preregisters the first collisionless prescribed-field
test-particle wall-loss campaign using the independently
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
python -m experiments.cft_orbit_wall_loss_v1.run prepare
# commit and push the preregistration, then execute only from a clean detached worktree
python -m experiments.cft_orbit_wall_loss_v1.run execute
```

There is one immutable attempt. The shared `experiment_runtime` owns the result
bundle, access-before-label records, atomic artifacts, terminal state and
manifest. A retained Git-common lock prevents another execution. Coupling is
export-only and is emitted only after every preregistered numerical and exact
replay gate passes.

No result is PIC, self-consistent plasma, experimental validation, hardware
validation, total thruster performance, or mirror-formula publication.
