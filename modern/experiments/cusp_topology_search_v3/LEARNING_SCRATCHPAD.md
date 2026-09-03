# Cusp topology search v3 - learning scratchpad

- [self] Read the QoI extractor before reusing a name from a brief: the sweep's "axis cusp"
  is the axis `|B_z|` maximum, its `axis_null_positions_m` are the nulls.
- [self] In an axisymmetric field the separatrix of an axis null is the `g = 0` contour of
  `g = (psi - psi_axis)/r^2`; the wall cusp is the root of `psi(r_w, z) - psi_axis`. A 2-D
  vector-root search at the wall finds nothing by construction (the v1/v2 nulls).
- [self] Run the refined map on the SAME axis window as the accepted map (computed from the
  accepted mesh); otherwise a null near the window edge can enter or leave the count for a
  purely geometric reason and fail the stability gate spuriously.
- [self] Evaluate refinement stability on the wall intersections, not on their
  inside/outside classification: the P2 exit cusp sits 28 um inside the 18 mm straight end
  and would otherwise flip the cusp count between resolutions. Flag it `boundary_ambiguous`.
- [tool] `chamber.exit_start_m = 0.024 - 0.006` is `0.018000000000000002`; compare
  authorities with a tolerance, never `!=`.
- [tool] The four-cell v2 dataset's `maps.<role>.artifact_sha256` is the sha256 of the
  artifact BYTES (== the sidecar-verified file), while the artifact's own
  `integrity.payload_sha256` excludes the integrity block. Read the sealing code before
  asserting which hash a record carries.
- [tool] The v2 geometry hash embeds the protocol BYTE hash through an `EvidenceNote`; on an
  LF checkout it cannot be reproduced without substituting the recorded CRLF-era hash
  (documented in the v2 POSTHOC_AUDIT). Scope the substitution with a context manager.
- [self] The v2 candidate family (even stages at 16-42 % of the odd stages, first polarity
  +1) has ~one axis null in the channel: it is not an equal-strength PPM stack, so its
  zero four-cell count is doubly by construction (definition AND source policy).
- [self] (post-execution) A held-out reference must be extracted by the SOURCE's own
  semantics, never by a float identity on a derived quantity: v1 clusters an axis
  sign-change with a neighbouring bilinear Newton root and reports the centroid, so
  `r_m == 0.0` dropped 26 of 206 sealed axis clusters and cost the one-shot execution.
  Select by member method (`axis_sign_change`/`axis_grid`).
- [self] Pick shakedown designs that exercise the reference's edge cases (multi-member
  clusters), not just the smallest and largest of a family; both shakedown v1 cases had
  single-member axis clusters and the defect stayed invisible until the freeze.
