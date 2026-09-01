# CFT topology characterization learning scratchpad

Policy: experiment-local only. Accepted packages, prior experiments, shared
workstreams, material fields, and FYP remain untouched.

## 2026-09-02 preregistration

- [user] The four-cell v2 failure showed that raw detector points cannot be
  equated with physical cusps. Establish the stage-to-cusp-to-cell relation
  before another search.
- [audit] Coupling v3 deliberately reports axis, grid, and bilinear vector-null
  candidates, but its fixed 1 nm deduplication is not a mesh-scaled physical
  clustering policy. The characterization therefore clusters detections once
  in physical 2D before interpretation.
- [self] X and O points have different physical roles. A converged X/index -1
  point may bound cells; a converged O/index +1 point may identify a closed-cell
  center. Reporting them separately avoids calling every field null a cusp.
- [self] The axis r=0 is a coordinate singularity, not a material or finite-box
  wall. Local derivatives and winding are evaluated with the axisymmetric odd
  B_r/even B_z reflection.
- [self] Exact separatrices pass through a saddle and can trigger a marching-
  squares tie. The preregistered evidence uses fixed psi offsets on both sides
  and requires changed nearby closed-component counts.
- [self] Equal cardinalities do not justify forced root pairing. The Hungarian
  matrix includes real dummy alternatives on both sides, so a remote root can
  remain unmatched before stability decisions.
- [provenance] Dependency identity is a Git blob identity. JSON results use
  canonical semantic hashes and text uses normalized-LF hashes; platform
  worktree newline bytes are never scientific identities.
