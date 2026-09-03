# Learning scratchpad — orbit wall-loss geometry screening v1

- [self] The shakedown rule paid for itself again: a synthetic preflight would
  never have produced `np.bool_` inside the field-evidence checks; the real
  re-solve did, and `canonical_bytes` rejected it. Every worker payload now goes
  through `_plain()` before it reaches the runtime.
- [self] `result_artifact` refuses to seal unless all three convergence flags
  are true. For a many-design screening this forces one worker task per DESIGN
  (all its cases integrated first, N→2N convergence assessed, then sealed);
  a non-converged design is reported through summaries/endpoints but carries no
  sealed artifact or handoff, and that is recorded rather than hidden.
- [self] Re-solving an accepted L1a case on CPU is a legitimate substitute for
  a missing stored map only because the sweep sealed geometry/source/config/case
  hashes AND QoIs: identity is proven three ways (hash equality, QoI replay under
  the sweep's own tolerances, node-wise agreement for the representatives).
- [self] Bind the experiment's own code (`experiment_code_sha256`) into the
  shakedown record and authorities, not only orbit_mc: otherwise a code change
  between shakedown and prepare is invisible to the gate.
- [tool] Bundle-size budget: full orbit artifacts (~260 KB gz per 512 orbits)
  for 196 cases would be ~50 MB; publish sidecars + compact endpoint tables for
  all cases and full artifacts only for the representatives.
- [tool] SVG `<path>` in an axis group is filled black unless `fill:none` is
  set; headless Chrome shows it as a large triangle.
- [tool] `Start-Process python -m http.server` + immediate Chrome gives
  ERR_CONNECTION_REFUSED; poll `netstat` for LISTENING first, and pick a fresh
  port (8765/8766 are held by other agents' stale servers).
- [tool] The Write tool emitted LF here, but every tracked text file was still
  byte-checked for CR before each commit (`git ls-files --eol`).
