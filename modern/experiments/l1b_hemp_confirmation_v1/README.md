# Preregistered L1b HEMP confirmation v1 - material-aware P2 check of the 15 HEMP-like designs

> **Recorded development rejection.** The one execution at `b9449ee5` resolved 13/15 designs and
> recorded two level-0 mesh-angle failures (028, 048) before any solve; results committed as is
> (`978c71be`), see `POSTHOC_REJECTION.md`. No verdict exists for v1. The campaign continues as
> `../l1b_hemp_confirmation_v1_1` (verdict CONFIRMED, `4db0a852`); the dashboard
> `modern/visualization/l1b-hemp-confirmation-v1.html` shows v1.1 and carries this record.

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

Why: the sweep-v3 claim boundary declared the soft-iron poles and yoke source-free vacuum and
queued this confirmation (`protocol.json#claim_boundary.l1b_p2_confirmation`,
`queued_not_run`); the TWT/PPM review (`beb4772c`, section 5.3) recommends a material-aware
check before any absolute-field, mirror-ratio or rho-based claim for r_w / L > 0.5. A
disagreement is a valid, reportable result.

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
python -m experiments.l1b_hemp_confirmation_v1.run shakedown   # non-evidentiary, before prepare
python -m experiments.l1b_hemp_confirmation_v1.run prepare     # freezes the authorities
# commit "preregister L1b HEMP confirmation v1", push, then from a clean detached worktree:
python -m experiments.l1b_hemp_confirmation_v1.run execute
python -m experiments.l1b_hemp_confirmation_v1.run validate
```

Dashboard: `modern/visualization/l1b-hemp-confirmation-v1.html`
(`generate_l1b_hemp_confirmation_v1_dashboard.py`). Paper admission is NOT in scope.
