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
