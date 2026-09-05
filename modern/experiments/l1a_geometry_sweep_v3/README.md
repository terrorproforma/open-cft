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

## Post-hoc audit note: the sealed source contract is verified against the frozen commit

`experiment.verify_shakedown_record` is the PRE-execution gate: it requires the live worktree to
equal the code the shakedown proved (`*_sha256_current`), which is what `prepare` and the one
`execute` need; it is sealed under `experiment_code_sha256` and unchanged. After the terminal
bundle existed, `cft_revival.experiment_runtime` moved at `bb756418` (2026-09-03: pinned-descriptor
cap and `recovery.py` for the geometry-screening-v2 EMFILE), so the record's
`dependency_source_sha256` stopped equalling the LIVE tree although nothing about the evidence
changed - a live-tree assertion can only hold until the next commit to a shared package.

`frozen_contract.py` (post-execution; not in `EXPERIMENT_CODE_FILES`, nothing sealed is edited)
therefore asks the honest question: do the sealed digests describe the code at the commit the
immutable execution lock names (1923ef7601bcc07acafa28ce54db687f025922b6)? `verify_recorded_shakedown` recomputes
`experiment_code_sha256`, `dependency_source_sha256` and `field_pipeline_source_sha256` from the
Git blobs at that commit, using the file inventories the shakedown record and the bundle's
`artifacts/source-binding.json` carry, and requires equality with the sealed values (all three
recompute exactly). The live tree's digests are RECORDED beside them
(`live_tree.*_current`, `drift`, added / removed / changed files - today: `recovery.py` added;
`experiment_runtime/__init__.py`, `filesystem.py`, `lifecycle.py` changed) and never asserted equal;
`strict_live_tree=True` restores the pre-execution semantics. A tampered record, a missing blob
or a commit this repository cannot resolve fails closed. Shared plumbing:
`cft_revival.provenance` (`modern/src/cft_revival/provenance/`).
