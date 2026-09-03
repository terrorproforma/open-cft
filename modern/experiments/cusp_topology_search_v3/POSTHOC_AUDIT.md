# Post-hoc audit: cusp topology search v3 (recorded `assessment_rejection`)

Read-only audit of the single execution of preregistration `69159934`
(result commit `8cbcdbe6`, bundle `results/`, 1212 files). Nothing under
`results/` was edited; `audit_held_out.py` re-derives every number below from the
sealed bundle and the sealed characterization-v1 dataset and refuses to write
inside `results/`.

## Verdict

**Recorded rejection caused by an experiment-code defect in the held-out
reference extraction, not by the topology.** All 281 declared designs were
rebuilt with identity proofs and solved at both resolutions; the binding gates
`all_declared_designs_resolved`, `every_null_converged`,
`every_trace_terminates_cleanly`, `every_wall_trace_flux_consistent`,
`refinement_stability` (281/281 stable), `determinism_replay` and
`hash_bindings` are all true. The binding gate `held_out_correspondence` is
false for 14 of the 56 characterization-v1 designs (all 5-8 stage stacks),
so the campaign terminal state is `assessment_rejection` (`gates_failed`).
Under the protocol's `no_patch_or_rerun` rule the bundle stands as recorded.
The estimands it contains are **not accepted** results of this campaign; the
corrected campaign is `cusp_topology_search_v3_1`.

## Root cause

`fields._resolve_characterization` selected the sealed v1 primary-map axis
roots with `root["r_m"] == 0.0`. The v1 characterization clusters raw
detections within 0.75 mesh cells and reports the centroid; when the Newton
root of the bilinear cell next to the axis converged at `r ~ 3e-8 m` it was
clustered with the `axis_sign_change` detection and the centroid became
`r ~ 1e-9 .. 1.6e-8 m`. Those clusters (26 of the 206 sealed axis clusters;
22 inside the channel, in exactly the 14 failing designs) were dropped from the
reference, so the corresponding v3 nulls counted as "unmatched observed" and the
bijection failed. The shakedown designs (`s02-p0-r0-neg`, `s08-p0-r0-neg`) had
only single-member axis clusters with `r_m == 0.0` exactly, so the shakedown did
not reach the defect.

## What the sealed data show

| quantity | value |
| --- | --- |
| sealed v1 axis clusters (primary maps, non-boundary) | 206 |
| clusters with centroid `r != 0` (dropped by the recorded filter) | 26 (max `r` = 1.58e-8 m) |
| dropped clusters inside the channel | 22, in exactly the 14 recorded failing designs |
| recorded unmatched v3 nulls per failing design | equal to the dropped in-channel clusters of that design |
| correspondence under the intended filter ("cluster contains an `axis_sign_change`/`axis_grid` member") | 56/56 bijections, max matched difference 1.76e-5 m (tolerance 2.5e-4 m), all X |
| sweep-v2 held-out correspondence (unaffected) | 96/96, max difference 2.73e-5 m |

The "extra" v3 nulls are genuine: the sealed v1 dataset lists them
(e.g. `topology-s05-p0-r0-neg` roots `root-012` at 14.7856 mm and `root-017`
at 18.3789 mm, `methods: [axis_sign_change, bilinear_vector_root]`,
`member_count 2`) and the axis `B_z` of the sealed v3 tracing grid changes sign
across the neighbouring nodes.

## Disclosures

- The recorded estimands (cusp-count distribution 0:6 / 1:140 / 2:36 / 3:56 /
  4:25 / 5:6 / 6:6 / 7:6 over 281 designs; sweep-v2 four-wall-cusp fraction
  19/96; four-cell-v2 128/128 with exactly one wall cusp; P2 three cusps at
  6.028 / 12.000 / 17.972 mm) are reproduced by the corrected v3.1 campaign
  because the definition, tolerances, design sets and field pipeline are
  unchanged there; only the held-out reference filter differs. Cite v3.1, not
  this bundle, for any number.
- `catalogue.load_catalogue(results)` refuses this bundle by design
  (`state != accepted_result`); no consumer may ingest its catalogue.
- `tests/experiments/cusp_topology_search_v3/test_cts_results.py` binds the
  RECORDED outcome (terminal state, the exact failing gate and design list,
  every other gate true) and `test_cts_posthoc_audit.py` re-derives this audit.

## Reviewer reproduction

```
cd modern
python -m experiments.cusp_topology_search_v3.audit_held_out --json %TEMP%\cts-v3-audit.json
python -m pytest tests/experiments/cusp_topology_search_v3 -q
```
