# Development log

## 2026-09-02 — predeclaration and implementation

- Scoped all new source, tests, and generated outputs to the two authorized
  experiment paths.
- Declared 11 bounded variables spanning compact three-to-five-stage stacks,
  pitch, source radii and thickness, polarity, source strength, and divergent
  exit geometry.
- Declared fixed-domain L1a solver settings, field-only QoIs, objective
  directions, feasibility gates, exact constrained ranking, and a typed failure
  taxonomy.
- Added geometry/source/config/case SHA-256 anchors and strict JSON/sidecar
  bundle validation.
- Added representative geometry plus full/downsampled accepted L1a artifacts.
- Kept timing in a separate non-benchmark diagnostic artifact.

## Verification

- Real run: 96/96 cases evaluated on Warp 1.14.0 `cuda:0`, NVIDIA
  GeForce RTX 5090; zero typed failures, 96 feasible, 25 nondominated.
- All boundary, true-residual, source-transfer, topology-confidence,
  manufacturability, artifact, and six-case CPU/CUDA parity gates passed.
- Focused experiment/geometry/fields/optimization suite: 165 passed.
- `python -m compileall -q src experiments/l1a_geometry_sweep tests`: passed.
- `git diff --exit-code -- FYP`: passed.
- Full compatible suite with isolated imports: 888 passed, one optional native
  extension skip, and two failures confined to the concurrently changing,
  explicitly out-of-scope `material_fields` workstream.
- The ordinary full-suite invocation also encountered a pre-existing duplicate
  `test_campaign` module-name collection collision; isolated import mode
  completed collection.
- End-to-end experiment process wall time was about 48.8 seconds, but all
  timings are uncontrolled diagnostics under concurrent GPU load and provide
  no benchmark evidence.

## 2026-09-03 — results tracked for the committed dashboard

- Defect: the committed dashboard (`visualization/l1a-geometry-sweep.html`),
  its generator and `tests/experiments/l1a_geometry_sweep_visualization`
  render and verify `results/manifest.json`, `dataset.json` and the five
  representative artifacts, but `results/` was masked by the root
  `.gitignore` `Results/` rule (case-insensitive on this filesystem) and had
  never been committed; only the producing checkout held a copy, so a fresh
  clone collected the dashboard tests with an error.
- Resolution: the 38 result files (9.6 MB, 19 artifacts plus 19 SHA-256
  sidecars) were copied byte-for-byte from the producing checkout and
  force-added (`git add -f`), exactly as the v2 bundle (7.9 MB) and the
  four-cell topology and orbit wall-loss bundles were. Every artifact is
  byte-exact against its sidecar and against the digests pinned in the
  generator (`manifest.json` `eb73fc69…`, `dataset.json` `2fb0c19d…`);
  no artifact contains a CR byte. Ten sidecars had been written CRLF; Git's
  `eol=lf` pin stores them LF, which changes no attested digest (the sidecars
  name the artifacts' bytes, not their own).
- No artifact byte was edited; the copy in the producing checkout is
  untouched. Recorded in `docs/workstreams/test-health-devlog.md`.
