# Shared visualization development log

Scope: the standalone dashboards under `modern/visualization/`. Experiment-local
dashboards keep their own logs next to their generators.

## 2026-09-03 — plasma / magnetic topology results dashboard

- Added `generate_plasma_topology_dashboard.py`, its template
  `plasma-topology-results.template.html`, the generated
  `plasma-topology-results.html`, and
  `tests/visualization/test_plasma_topology_dashboard.py`.
- The dashboard embeds only committed, accepted or recorded artifacts and
  regenerates no physics:
  - CFT topology characterization v1 (56 cases, 7 representative primary maps,
    every clustered primary-map root with class/zone/exclusion);
  - four-cell topology search v2 (128 candidates, 2 representative downsampled
    maps, recorded interior wall-cusp positions on all three map roles);
  - four-cell topology search v1 (128 superseded screening cases and their
    coupling-v2 cusps, labelled deprecated);
  - preregistered L1a geometry sweep v2 (96 cases, 7 gates, 4 representative
    downsampled maps, recorded axis cusp/null positions and mirror ratios);
  - accepted axisymmetric v1.2 example artifacts (through the sibling
    generator's authoritative reload);
  - the NUMERICAL_P2_QUALIFIED divergent-exit FEM field (144×92 raster of the
    verified viewer, a 2 mm wall-line |B_r| profile, and the orbit v4 cell
    seeds);
  - coupling v4 wall-cusp held-out validation v1 and v2 failure records;
  - full-orbit wall-loss v4 accepted result read from
    `origin/exp/cft-orbit-wall-loss-v4` commit `6922a3cf…` via `git show`
    (9 campaigns × 32 strata with Wilson 95 % intervals), plus the v1–v3
    failure disclosure carried in its protocol.
- Verification before embedding: pinned manifest file hashes, canonical
  payload/semantic hashes, `.sha256` sidecars where they exist, manifest
  `byte_sha256` sidecars for the orbit result, `git diff --quiet <results
  commit>` for on-disk evidence, the P2 result/viewer stream hashes bound to
  the orbit v4 P2 input authority, and the `|B| = hypot(B_r, B_z)` identity for
  every embedded L1a field. Any mismatch refuses generation.
- Every embedded section, representative, campaign and ledger claim carries
  `sources` ids into a 72-entry source ledger (path, SHA-256 of bytes read,
  commit, identity method, byte-match flag). Two wall-cusp-v1 files match only
  their recorded semantic identity; the dashboard says so.
- Determinism: no wall-clock time; the footer "evidence snapshot" is the author
  time of the orbit v4 results commit (or `SOURCE_DATE_EPOCH`). Two renders are
  byte-identical; the checked HTML is regenerated from the committed evidence.
- Presentation: eight landmarked sections (overview and fidelity ladder, class
  distribution with parallel coordinates, representative |B|/ψ maps with roots,
  wall/axis ticks and mirror-ratio status, cusp-position strips and the P2
  field, orbit v4 probabilities, coupling v4 outcomes, validation ledger,
  provenance). Canvas rendering with device-pixel-ratio redraw, marching-
  squares ψ isolines, keyboard cursor, theme toggle, reset, no per-point DOM.
- Browser check (Chrome headless via Playwright, because the Cursor browser
  tool could not hold a tab in this session): no page errors, all 12 canvases
  painted, `scrollWidth == clientWidth` at 1440, 390 and 320 px, selects and
  keyboard cursor exercised. Fixed a redraw race where successive control
  changes cancelled each other's animation frame.
- Validation: `pytest tests/visualization/test_plasma_topology_dashboard.py`
  → 17 passed. Generated HTML is 3.7 MB (cap 15 MiB).
