# Preregistered L1b HEMP confirmation v1.1 - material-aware P2 check of the 15 HEMP-like designs

Re-preregistration of `l1b_hemp_confirmation_v1` after its recorded development rejection
(`b9449ee5` preregistration, `978c71be` result: 13/15 designs resolved, designs 028 and 048
failed the 10 deg level-0 mesh angle gate before any solve; see
`../l1b_hemp_confirmation_v1/POSTHOC_REJECTION.md`). v1.1 changes exactly two declarations:
the level-0 mesh minimum-angle gate is 5 deg (disclosed; the two designs' meshes have 5.3 / 9.3 deg
geometric slivers from body-fitted near-coincidences, recorded per level) and the shakedown
records a whole-set mesh preflight that `prepare` and `execute` verify. Every tolerance,
threshold and numerical parameter is identical to v1.

One-execution confirmation campaign: for each of the 15 HEMP-like designs of the accepted
L1a geometry sweep v3 (`l1a_geometry_sweep_v3/results`, Koch rho_conservative >= 1.5 at
every wall cusp) the material-aware field (adaptive P2 FEM of `cft_revival.fem_reference`:
soft-iron poles mu_r 4000 between the magnets, soft-iron return yoke, recoil-remanence SmCo-like
magnets) is solved on CPU and the cusp topology search v3.1 definition (axis nulls,
separatrices, wall cusps, cells, Koch rho) is applied verbatim. The cell count, the cusp
axial positions, the wall |B| and the HEMP-like classification are compared with the sealed
L1a (linear-vacuum equivalent-current FDM) record. `protocol.json` is the preregistration
authority (semantic-hash bound); `authorities.json`, `design-authorities.json` and
`shakedown.json` are frozen by `prepare`.

Per design (one at a time): sweep-v3 rebuild with identity proof against the sealed design
authorities and the sealed L1a record bytes; graded body-fitted P2 mesh (bore r_w / 8,
features / 3, padding 0.5); level-0 solve, Dorfler/red level-1 solve (relative true residual
2e-10, RAM guard against 0.4 x free RAM at start, DOF cap 600k); regular-grid sampling of
the bore (32 radial intervals, 1x and 2x; post-scaled by the L1a `source_strength_scale`);
definition-v3 characterization of the coarse (level 0), accepted (level 1) and refined (level 1
at 2x) maps on the sealed L1a axis window; sweep-v3 descriptors; comparison.

Gates: binding integrity (all designs resolved, identity, every solve converged = GATE (a),
nulls / traces / flux roots, sampling stability 0.25 mm, axis window reproduced, RAM policy,
determinism replay, hash bindings) decide whether the bundle is accepted evidence; the
predeclared confirmation gates GATE (b) (boundary-tolerant cusp-count agreement fraction >= 1.0)
and GATE (c) (every matched cusp shift <= max(r_w / 8, L1a dz) and every matching a bijection)
classify the outcome CONFIRMED / PARTIALLY_CONFIRMED / DISCONFIRMED; (d) HEMP-like preserved,
wall |B| and rho ratios are reported.

Outputs (`results/artifacts/`): `confirmation-dataset.json` + `.csv`, `gates.json`,
`campaign-result.json` (verdict + agreement table), per-design records and sampled P2 grids.

Lifecycle (from `modern/`, CPU only, one design at a time, GPU untouched):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
python -m experiments.l1b_hemp_confirmation_v1_1.run shakedown   # non-evidentiary + whole-set mesh preflight
python -m experiments.l1b_hemp_confirmation_v1_1.run prepare     # freezes the authorities
# commit "preregister L1b HEMP confirmation v1.1", push, then from a clean detached worktree:
python -m experiments.l1b_hemp_confirmation_v1_1.run execute
python -m experiments.l1b_hemp_confirmation_v1_1.run validate
```

Dashboard: `modern/visualization/l1b-hemp-confirmation-v1.html`
(`generate_l1b_hemp_confirmation_v1_dashboard.py`, reads this bundle and the v1 rejection
record). Paper admission is NOT in scope.
