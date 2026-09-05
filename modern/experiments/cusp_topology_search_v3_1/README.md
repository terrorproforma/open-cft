# Cusp topology search v3.1 (corrected re-preregistration of v3)

Classification: `SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY` for the three L1a design
sets, `P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY` for the single P2 row. Nothing here is
a plasma, mirror-probability, wall-loss or performance claim.

## Relation to v3

`../cusp_topology_search_v3` (preregistration `69159934`, result `8cbcdbe6`) ended
`assessment_rejection`: every numerical gate true for all 281 designs, but the binding
held-out gate failed for 14 characterization-v1 designs because the reference extraction
kept only sealed v1 axis clusters with centroid `r_m == 0.0` and so dropped 26 genuine
clusters whose centroid carries a bilinear Newton member at `r <= 1.6e-8 m`
(`../cusp_topology_search_v3/POSTHOC_AUDIT.md`). v3.1 changes **only** that extraction
(`fields.v1_axis_reference`: a sealed axis root is a cluster containing an
`axis_sign_change`/`axis_grid` member) and adds the lineage disclosure; the definition,
tolerances, design sets, gates, field pipeline and dependency sources are unchanged. The
v3.1 shakedown adds `topology-s05-p0-r0-neg`, a v3 failing design, so the corrected path is
exercised on the defect's own case before the freeze.

For the definition, design sets and lifecycle see `../cusp_topology_search_v3/README.md`;
the commands are the same with `cusp_topology_search_v3_1`, the preregistration subject is
`preregister cusp topology search v3.1`, and the Git-common lock is
`cusp-topology-search-v3.1.execution.lock`.

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
immutable execution lock names (1600cfd3b102980eeba4b070930667d232a1105c)? `verify_recorded_shakedown` recomputes
`experiment_code_sha256`, `dependency_source_sha256` and `field_pipeline_source_sha256` from the
Git blobs at that commit, using the file inventories the shakedown record and the bundle's
`artifacts/source-binding.json` carry, and requires equality with the sealed values (all three
recompute exactly). The live tree's digests are RECORDED beside them
(`live_tree.*_current`, `drift`, added / removed / changed files - today: `recovery.py` added;
`experiment_runtime/__init__.py`, `filesystem.py`, `lifecycle.py` changed) and never asserted equal;
`strict_live_tree=True` restores the pre-execution semantics. A tampered record, a missing blob
or a commit this repository cannot resolve fails closed. Shared plumbing:
`cft_revival.provenance` (`modern/src/cft_revival/provenance/`).
