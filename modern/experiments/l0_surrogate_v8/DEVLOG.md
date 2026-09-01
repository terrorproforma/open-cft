# L0 surrogate v8 devlog

## 2026-09-02 — protocol construction

- Isolated V8 under new experiment and test paths.
- Reused V7's exact `Fraction`/integer group conformal implementation.
- Added raw and physics-informed ARD GP candidates at budgets 128/160/224.
- Added lower-only row validity and independent median/p90 width gates.
- Retained clean detached execution, actual-import blob closure and atomic lock.
- No prior assessment labels were loaded during protocol construction.
