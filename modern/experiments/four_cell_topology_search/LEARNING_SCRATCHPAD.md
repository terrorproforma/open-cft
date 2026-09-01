# Four-cell topology search learning scratchpad

Policy: experiment-local record. Shared workstreams and accepted packages remain
untouched.

## 2026-09-02 session

- [user] Search topology before performance and do not force a four-cell label.
- [self] In coupling v2, each accepted centreline null/minimum produces one
  direct solver segment; the corrected global model therefore requires exactly
  four accepted segments, not four geometry stages.
- [self] An exact zero low field has no finite high/low mirror ratio. Candidate
  count alone is insufficient; finite positive non-inverted mirror ordering is
  a mandatory gate.
- [self] Unequal alternating source strengths can remove or merge some sign
  changes while retaining four uncertainty-supported interior candidates.
- [test] The L1a preview emits an inner/outer opposite-current sheet pair for
  each magnet stage. Alternating strength must scale both sheets together;
  per-sheet scaling invalidates the accepted equivalent-current pairing.
- [self] Endpoint nulls can pass naive topology detection. Candidate sample
  indices and brackets must remain at least two field-grid cells from the
  finite Dirichlet boundary.
- [tool] Variable finite-box extents require binary64-stable endpoint
  construction so strict L1a artifact coordinate identities replay exactly.
- [self] Representative artifacts must preserve the exact accepted search
  bytes. A second GPU solve is not assumed bitwise identical under concurrent
  execution.
- [result] After correcting stage-paired source scaling, all 128 strict
  geometries and fields evaluated; only two maps passed the complete four-cell
  chain, and both converged at all three hypothetical operating points.
- [audit] Residual convergence does not establish an identifiable plasma
  state. Every selected v1 root has Jacobian rank 22 for 25 state variables.
- [audit] Coupling v2's same-z centreline-low/wall-high mirror proxy is
  deprecated, and the selected low fields are roundoff-scale nulls. The
  resulting nominal mirror ratios/probabilities cannot support physical mirror
  claims even when the numerical topology gates pass.
- [audit] The v1 protocol was not preregistered. Calling its six residual roots
  "performance publications" was a semantic error; the corrected counts are
  six residual roots, zero identifiable states, and zero performance
  publications.
- [self] Preserve withdrawn state/power values in a separate audit-only archive
  rather than deleting numerical evidence or leaving publication-shaped fields
  inside outcome objects.
