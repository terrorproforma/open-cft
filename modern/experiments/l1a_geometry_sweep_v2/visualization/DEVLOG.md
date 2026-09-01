# Visualization development log

## 2026-09-02 — committed preregistered sweep-v2 dashboard

- Added a v2-only deterministic generator and self-contained HTML dashboard.
- Strict generation checks bind:
  - preregistration commit `092f5fae692ee7d6711e0c7e1c94dac6a345f37c`;
  - direct-child results commit
    `f30cb42ec4a8633bf634a3d32ffa5b11f66be97a`;
  - commit authored times and intervening execution-lock timestamp;
  - protocol, lock, raw, summary, manifest and all deterministic representative
    file, sidecar, payload and Git-clean identities;
  - exact 96 evaluated/0 failed/25 nondominated counts;
  - seven independently recomputed gates and five roles coalesced to four
    unique representative cases.
- Added linked canvas scatter/parallel views, inclusive filters, selected-case
  inputs/hashes/parity, four representative field maps/ψ contours/profiles,
  geometry/source overlays, preregistration timeline, environment metadata,
  replay tolerances, reset/theme and keyboard/mouse behavior.
- Validation:
  - focused dashboard suite: `10 passed in 6.77s`;
  - compatible v2/geometry/fields/optimization/visualization suite:
    `240 passed in 24.64s`;
  - visualization generator/test `compileall`: passed;
  - generated HTML SHA-256:
    `2db0d0212191e49972d1915ca4029d9d42887f55c01254b1a63170f87a690cd3`.
- No dependency installation, Git mutation, commit, or edits outside the two
  authorized v2 visualization paths.

## 2026-09-02 — 320–390 px representative-control fix

- Constrained controls, labels, selects and flex/grid children with zero
  intrinsic minimum width and 100% maximum width.
- Stacked scatter and field controls below 590 px; representative values use
  visual ellipsis while retaining complete option text, titles and ARIA labels.
- Added a structural mobile-layout regression with a synthetic future role
  label substantially longer than current labels.
- Focused responsive dashboard suite: `11 passed in 5.86s`.
- Compatible v2/geometry/fields/optimization/visualization suite:
  `241 passed in 27.01s`; JavaScript/offline/path/secret and `compileall`
  checks passed.
- Regenerated HTML SHA-256:
  `3c3f5aeab98006b864c8a3a56fcb1ec0b65128ad107d778e0aeceb7fcf1cde86`.
