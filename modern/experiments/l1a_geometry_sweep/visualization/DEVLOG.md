# Visualization development log

## 2026-09-02 — deterministic offline L1a sweep dashboard

- Added `generate_dashboard.py`, which strictly validates the pinned sweep
  manifest/dataset and all manifest-listed deterministic representative files
  before producing a compact embedded visualization payload.
- Added `l1a-geometry-sweep.html` with linked canvas scatter/parallel views,
  inclusive nine-metric filters, nondominated/representative highlighting,
  representative field rasters, ψ contours, Bz profiles, geometry/source-band
  overlays, complete selected-case evidence, reset, and light/dark redraw.
- Added dedicated tests covering exact identities/counts/front, deterministic
  bytes, field semantics, JavaScript syntax, offline/path/secret safety,
  accessibility hooks, high-DPI canvas, and tamper rejection.
- Validation:
  - focused dashboard suite: `10 passed`;
  - compatible experiment/geometry/fields/optimization/visualization suite:
    `235 passed in 25.12s`;
  - visualization generator/test `compileall`: passed;
  - generation SHA-256 before final documentation-only additions:
    `0081461519d479a4d0d5059aeea4c8f45206cc65ab069478e0901f5a24bed948`.
- Browser runtime was not claimed: direct `file://` navigation was blocked and
  the browser harness failed to retain a localhost target tab. Manual browser
  checks remain listed in handoff.
- No dependencies were installed, no experiment/shared/source/result file was
  edited, and no commit was created.
