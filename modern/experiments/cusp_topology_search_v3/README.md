# Cusp topology search v3 (literature cusp/cell definition)

Classification: `SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY` for the three L1a design
sets, `P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY` for the single P2 row. Nothing here is
a plasma, mirror-probability, wall-loss or performance claim.

## Why v3

The frozen v1 characterization and v2 search defined a wall cusp as a wall-side X-type
vector null with prominence/separation criteria and found 0 stable eligible cusps/cells over
56 designs and 0/128 four-cell candidates. The literature review
(`modern/docs/literature/reduced-models-cusp-topology-blockers.md`, Section 3) shows that
in a periodic-permanent-magnet stack the vector nulls sit **on the axis** and the wall cusp
of the HEMP/DCFT literature is where the **separatrix** emanating from an axis null meets
the dielectric wall (Gildea 2012; Kornfeld 2007; Koch 2011; Lewerentz & Schneider 2023,
DOI 10.3390/app13063491). v3 tests that definition. The paper's Section 8 nulls remain true
under their frozen definition; no v1/v2 artifact is edited.

## Definition v3 (frozen in `protocol.json#definition_v3`)

1. **Axis null**: sign change of `B_z(0, z)` on the axis, bisected on the axis-regular C1
   bicubic interpolant (`PsiBicubicField`) to a 1e-12 m bracket; classified with the
   characterization-v1 Jacobian/winding-index code (imported unchanged) and the analytic
   bicubic Jacobian `diag(-g_z, 2 g_z)` (every simple axis null is X-type; the separatrix
   leaves along `e_r`).
2. **Separatrix**: event-aware RK4 field-line trace (v4 scheme) from the radial
   eigen-direction of the null to the wall cylinder `r = r_w`; the intersection must agree
   with the root of `psi(r_w, z) - psi_axis` (the separatrix is the `g = 0` contour); the v4
   bilinear step is run as a reported comparison.
3. **Wall cusp / cell**: intersection inside the straight dielectric; cells between
   consecutive cusps plus anode-side and exit-side partial cells; mirror descriptors along
   the wall and against the on-axis `|B_z|` peak between the generating nulls.
4. **Stability**: 2x refined map; same axis-null count, same wall-reaching separatrices,
   `|dz| <= 0.25 mm`. Held-out: v3 must reproduce the sealed v1 axis roots and the sealed
   sweep axis nulls.

## Design sets (281 designs)

| set | count | provenance |
| --- | --- | --- |
| `sweep_v2` | 96 | accepted L1a geometry sweep v2, re-solved with the geometry screening's identity-proven CPU pipeline |
| `four_cell_v2` | 128 | sealed four-cell v2 candidates (source/material hashes and geometry hash under the recorded CRLF-era protocol hash equal the dataset) |
| `characterization_v1` | 56 | sealed characterization v1 cases (held-out null reference) |
| `p2_divergent_exit` | 1 | P2-qualified divergent-exit-stack FEM field through the orbit v4 adapter (iron present) |

## Lifecycle (from `modern/`)

```
python -m experiments.cusp_topology_search_v3.run shakedown   # real designs of every set, NON-EVIDENTIARY
python -m experiments.cusp_topology_search_v3.run prepare     # refuses without a passed shakedown
# commit "preregister cusp topology search v3", push, then from a clean detached worktree:
python -m experiments.cusp_topology_search_v3.run execute
python -m experiments.cusp_topology_search_v3.run validate
```

Outputs: `results/artifacts/topology-dataset.{json,csv}`, `cusp-cell-catalogue.json`
(consumer contract, loader `catalogue.load_catalogue`), `gates.json`,
`campaign-result.json`, per-design records under `artifacts/designs/` and gzipped tracing
grids under `artifacts/fields/`.

Tests: `modern/tests/experiments/cusp_topology_search_v3` (manufactured field with an
analytic curved separatrix, protocol, identity binding, catalogue contract, lifecycle-aware
bundle checks).

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
immutable execution lock names (691599340355818ff64d3834d45110768a751589)? `verify_recorded_shakedown` recomputes
`experiment_code_sha256`, `dependency_source_sha256` and `field_pipeline_source_sha256` from the
Git blobs at that commit, using the file inventories the shakedown record and the bundle's
`artifacts/source-binding.json` carry, and requires equality with the sealed values (all three
recompute exactly). The live tree's digests are RECORDED beside them
(`live_tree.*_current`, `drift`, added / removed / changed files - today: `recovery.py` added;
`experiment_runtime/__init__.py`, `filesystem.py`, `lifecycle.py` changed) and never asserted equal;
`strict_live_tree=True` restores the pre-execution semantics. A tampered record, a missing blob
or a commit this repository cannot resolve fails closed. Shared plumbing:
`cft_revival.provenance` (`modern/src/cft_revival/provenance/`).
