# P2 FEM visualization devlog

## 2026-09-02

- Established an isolated visualization-only workstream. No solver, result,
  shared, material-field, FYP, or Git file is modified.
- Pinned the three third-level qualification manifests by exact file SHA-256.
- Added strict manifest canonical-payload replay and streamed file-hash checks
  for every linked result artifact, viewer, adaptive/domain checkpoint, and
  compressed array sidecar.
- Derived compact inline projections from verified evidence: actual nested mesh
  samples, field/material rasters, profiles, QoIs, orders, domain changes,
  residuals, quality, ancestry, memory, and timings.
- Kept acceptance semantics explicit: divergent is numerical P2 qualified;
  historical and compact remain screening-only due to non-positive observed
  order. Sub-percent changes alone are not presented as qualification.
- Added a standalone responsive HTML interface using Canvas for dense field,
  mesh, and profile rendering and SVG for convergence/domain plots.
- Added keyboard selection, reset, theme switching, accessible labels, offline
  restrictions, deterministic generation, JavaScript compilation, path/secret
  scans, and source/payload tamper tests.

## Verification record

- Generated `fem-reference-p2-qualification.html` successfully from all three
  verified evidence roots.
- Final SHA-256: `b3edba067d6e0aee990e742dbe125ab2a4c1d81681ea92c1ef7ef4477aa59d0c`.
- `python -m pytest modern/tests/fem_reference_visualization/test_dashboard.py -q`
  passes: `9 passed`.
- The suite includes embedded-JavaScript `node --check`, Python compilation,
  deterministic/current-output checks, offline/network scans, absolute-path
  and secret scans, evidence/status contracts, and payload tamper rejection.

## 2026-09-02 narrow-viewport correction

- Removed the fixed `320px` body minimum that could exceed the scrollbar-
  reduced containing block (`305px` observed at a nominal `320px` viewport).
- Made the shell use containing-block `width: 100%`, a `1500px` maximum,
  `min-width: 0`, border-box sizing, and percentage padding.
- Constrained grid/header/card/control descendants, selects, status, Canvas
  wrappers, canvases, SVG charts, and narrow hash columns to their containers.
- Added a structural regression covering 280–390 px containment rules and
  rejecting `100vw` or a restored fixed body minimum.
- Regenerated SHA-256:
  `e5961442bec2e5ea1fafcee4311088c52b7488f85a46659759f749ae9c1e3129`.
- Focused suite passes: `10 passed`, including JavaScript, offline, path,
  secret, deterministic-output, and Python compile checks.

## 2026-09-02 committed-evidence publication

- Rebased the visualization authority onto the actual committed P2 evidence at
  `a1158bad5eac3dd27ca6464a7649ce359524d8db`.
- Re-pinned all three committed manifest file hashes and now validates their
  explicit `qualification_status` and diagnostic-only timing/memory policy.
- The prior dirty-manifest HTML hash is obsolete. Regeneration from committed
  evidence produces SHA-256
  `b2c4b69b4d077de13c80a2f8313b036f86024567c498bdc3f29bbdf3132666a9`.
- Added local line-ending policy so the standalone HTML remains byte-stable
  when checked out on Windows.
