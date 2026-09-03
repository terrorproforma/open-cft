# Cusp topology search v3 - devlog

## 2026-09-03 - build, shakedown, preregistration

- Read the literature review (Section 3) and the frozen v1/v2 definitions. Under the
  literature definition the cusp is the separatrix-wall intersection of an axis null; in the
  axis-regular flux variable `g = (psi - psi_axis)/r^2` the separatrix is the `g = 0`
  contour and every simple axis null is X-type with `J = diag(-g_z, 2 g_z)`.
- Verified before writing a line of protocol: the sweep's "axis cusp" QoI is the local
  maximum of `|B_z|` on the axis (stage centres), not a null; the sealed v2 geometry hash
  embeds the CRLF-era protocol byte hash through an evidence note (matches 128/128 once
  substituted; source/material hashes match directly); the v1 identities match 56/56; CPU
  re-solves reproduce the stored GPU maps to 1e-20 Wb / 5e-15 T (v2) as the screening had
  shown for the sweep.
- Shakedown 1 (8 real designs): caught two defects before the freeze - the v2 dataset's
  `artifact_sha256` is the artifact-bytes hash, not the payload hash; the P2 geometry's
  `exit_start_m` is `0.018000000000000002` against the v4 authority `0.018`. Fixed both.
- Shakedown 2: `accepted_result`, 8/8 designs, every gate true, bundle validated, 63 s;
  timing projection ~17 min wall for 281 designs at 12 workers with a 1.5x contention
  factor.
- Fields are published as gzipped canonical JSON (the v2 tracing grids are ~200 kB each).

## 2026-09-03 - execution (preregistration 69159934 -> result 8cbcdbe6) and post-hoc audit

- One detached execution, 12 CPU workers, 797 s design stage, 281/281 designs resolved,
  281/281 stable, every numerical gate true, replay bit-identical.
- Terminal state `assessment_rejection` (`gates_failed`): the binding gate
  `held_out_correspondence` failed for 14 of the 56 characterization-v1 designs (5-8
  stages). Root cause (POSTHOC_AUDIT.md, `audit_held_out.py`): the reference extraction
  kept only sealed v1 axis clusters with centroid `r_m == 0.0`; 26 clusters carry a bilinear
  Newton member at `r <= 1.6e-8 m` and were dropped, 22 of them inside the channel, in exactly
  the 14 failing designs. Under the intended filter the correspondence is 56/56 (max
  difference 17.6 um). Recorded as is, no patch, no rerun; the corrected campaign is
  `cusp_topology_search_v3_1`.
- Tests now bind the recorded outcome (`test_cts_results.py`) and re-derive the audit
  (`test_cts_posthoc_audit.py`); 45 tests green.
