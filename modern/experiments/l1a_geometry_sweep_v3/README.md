# Preregistered L1a geometry sweep v3 - the HEMP-like wall-radius-to-pitch regime

One-execution, 128-design scrambled-Sobol screen of PPM-stack geometries on a box that
contains the sweep-v2 box and extends the wall-radius-to-pitch ratio to
r_w / L = 1.24 (x_w = pi r_w / L up to 3.88), plus the 96 accepted sweep-v2 designs as a
held-out reproduction set. `protocol.json` is the preregistration authority
(semantic-hash bound); `authorities.json`, `design-authorities.json` and
`shakedown.json` are frozen by `prepare`.

Why: the TWT/PPM review (`modern/docs/literature/twt-ppm-physics-for-hemp.md`, commit
`beb4772c`) showed that every recorded field is a single-harmonic PPM field whose
wall-cusp |B| is I_1(x_w) = 0.45-0.61 of the axis peak, so Koch's HEMP design ratio
rho >= 1.5 (Koch et al. IEPC-2007-110, Table 1) was unreachable in the sweep-v2
catalogue (x_w <= 1.55). The PPM prediction puts the threshold at I_1(x*) = 1.5,
x* = 1.937318, r_w / L = 0.616668. This campaign tests that prediction (reported
hypotheses H1/H2 in `protocol.json#descriptors_v3.hypothesis`); nothing in it is a
plasma, mirror-probability or performance claim.

Per design: accepted CPU solve at the sweep-v2 domain/resolution and a 2x refined solve;
the sweep-v2 QoIs and six metric gates verbatim; the cusp topology search v3.1
definition imported unchanged (axis nulls, separatrices, wall cusps, cells, 2x
stability); per-cusp Koch rho (downstream / upstream / conservative / wall readings),
I_1(x_w) prediction, wall harmonic content, separatrix angle, profiles.

Outputs (`results/artifacts/`): `sweep-dataset.json` + `.csv`,
`cusp-cell-catalogue-v3.json` (schema 1.1.0, loader `catalogue.load_catalogue`),
`gates.json`, `campaign-result.json`, per-design records and psi grids.

Lifecycle (from `modern/`, CPU only, <= 6 workers):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
python -m experiments.l1a_geometry_sweep_v3.run shakedown   # non-evidentiary, before prepare
python -m experiments.l1a_geometry_sweep_v3.run prepare     # freezes the authorities
# commit "preregister L1a geometry sweep v3", push, then from a clean detached worktree:
python -m experiments.l1a_geometry_sweep_v3.run execute
python -m experiments.l1a_geometry_sweep_v3.run validate
```

Dashboard: `modern/visualization/l1a-geometry-sweep-v3.html`
(`generate_l1a_geometry_sweep_v3_dashboard.py`).
