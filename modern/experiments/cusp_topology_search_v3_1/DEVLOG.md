# Cusp topology search v3.1 - devlog

## 2026-09-03 - corrected re-preregistration of v3

- v3 executed once (`69159934` -> `8cbcdbe6`) and ended `assessment_rejection` on the
  held-out gate alone; root cause audited read-only (`../cusp_topology_search_v3/
  POSTHOC_AUDIT.md`): the reference filter `r_m == 0.0` dropped 26 sealed v1 axis clusters
  with a bilinear member (22 inside the channel, in exactly the 14 failing designs).
- v3.1 = byte-copy of the v3 package with `fields.v1_axis_reference` (member-method filter),
  the `prior_campaign_disclosure.v3` block, the held-out wording, and
  `topology-s05-p0-r0-neg` added to the shakedown. Definition, tolerances, design sets, gates
  and the field pipeline are unchanged.
