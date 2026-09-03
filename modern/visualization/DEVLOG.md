# Shared visualization development log

Scope: the standalone dashboards under `modern/visualization/`. Experiment-local
dashboards keep their own logs next to their generators.

## 2026-09-03 10:20 AEST - MDO L0 campaign v1 dashboard

- New `generate_mdo_l0_campaign_v1_dashboard.py` -> `mdo-l0-campaign-v1.html`
  (340,268 bytes) from the recorded bundle of
  `experiments/mdo_l0_campaign_v1` (manifest `2a326f3c...`, result commit
  `c553124b`): verifies all 137 manifest files byte-for-byte, cross-checks
  metrics/curves/run artifacts/campaign-result against each other, embeds the
  payload as JSON and draws every chart with inline SVG (hypervolume curves,
  robust-vs-nominal fronts in three projections, parallel coordinates of the
  114 robust-Pareto designs, cusp-prior and scenario sensitivity tables,
  timing, protocol, provenance) under a claim-boundary panel.
- Deterministic (no wall clock, no paths), offline, error sink `#jserrors`;
  headless Edge: 0 JS errors at 1440 px and in a 390 px iframe host;
  screenshots `%TEMP%\mdo_scratch\mdo-dashboard-final-{desktop,top,narrow390}.png`.
- Tests `tests/visualization/test_mdo_l0_campaign_dashboard.py` (6): offline
  template + error sink, script-terminator escaping, tamper refusal, pinned
  manifest + byte-identical regeneration + committed freshness, payload equals
  bundle numbers, section presence.

## 2026-09-03 02:40 AEST — regenerate stale design gallery and first-results

- Break: `pytest tests/visualization` was 65 passed / 2 failed / 13 errors.
  `test_design_gallery.py::test_checked_artifact_is_byte_deterministic_and_current`
  and `::test_exact_sweep_config_sampling_and_dataset_identity` failed on
  `source.config_sha256`; every `test_first_results_visualization.py` test
  errored at fixture setup with `design gallery config SHA-256 does not match
  the sweep config` (`generate_first_results.py:100` refuses a stale gallery).
- Root cause: the gallery's `config_sha256` is the SHA-256 of the *exact bytes*
  of `config/l0-deterministic-sweep.json`. The committed blob has been LF
  (744 bytes, `2d727b1a…`) at `41bf9091` and at HEAD, but the gallery and both
  test pins were produced on 2026-09-01 from a `core.autocrlf=true` checkout
  whose working copy was CRLF (763 bytes, `a4703ac1…`). `fab0eccc` pinned
  `eol=lf` repo-wide, so the working copy is now LF and the recorded hash is
  no longer reproducible. Not an API change; not a physics change: the
  dataset identity `c0a36ed8…` (canonical JSON of all 8192 records), the five
  selected indices (6352/2752/1633/148/1192) and the median thrust threshold
  are byte-identical before and after regeneration.
- Regenerated (no hand edits): `design-gallery.json` via
  `python visualization/build_design_gallery.py` — 1 line changed
  (`config_sha256`). `first-results.html` via
  `python visualization/generate_first_results.py` — 5,994,361 bytes before
  and after; `old.replace(a4703ac1…, 2d727b1a…) == new` is True, i.e. the
  embedded gallery hash is the only difference. Both files LF, no BOM.
- Provenance pin changes (old → new, reason: CRLF-smudged bytes → committed
  LF bytes of the same config file):
  - `tests/visualization/test_design_gallery.py::EXPECTED_CONFIG_SHA256`
    `a4703ac1541539829f47f909d24d01d4996ed1da97a9d86e9e2323e54039fbbf` →
    `2d727b1af7d9be9f35f227cc318beae29af6cbd2fbead28842a4c17d67551b6b`
  - `tests/visualization/test_first_results_visualization.py::test_gallery_identities_and_indices_map_to_embedded_sweep`
    same old → new. `EXPECTED_DATASET_SHA256` and `EXPECTED_INDICES` untouched.
- Currency sweep of the other shared dashboards: `geometry-designs.html`,
  `axisymmetric-results.html` and `plasma-topology-results.html` regenerated to
  `%TEMP%` are byte-identical to the checked files (so the report that
  `geometry-designs.html` was stale was wrong; nothing to commit there).
- Validation: `pytest tests/visualization -q` → 80 passed (17 of them
  `test_plasma_topology_dashboard.py`); `tests/geometry` 41 passed;
  `tests/fields` 62 passed.
- Repo-wide per-directory sweep (`-x`, excluding `tests/pic`, `tests/orbit_mc`,
  `tests/experiment_runtime`): green — active_learning 96, coupling 143,
  fem_reference 37, fem_reference_visualization 10, hybrid 70, magnetics 52,
  optimization 76, physics 86, plasma 36, plasma_network 64, surrogates 38,
  validation 59, top-level `tests/test_*.py` 65 + 1 skipped,
  experiments/{cft_orbit_wall_loss_v4 37+1s, four_cell_topology_search 9,
  four_cell_topology_search_visualization 11, l0_surrogate 6, v2 5, v5 8,
  v6 7, v7 10, v8 10, v9 13, l1a_geometry_sweep 5+1s, l1a_plasma_coupling 8}.
  Red, NOT fixed here (outside `modern/visualization/`, see scratchpad):
  - `tests/experiments/l1a_geometry_sweep_visualization` (1 error):
    `experiments/l1a_geometry_sweep/results/manifest.json` does not exist in
    any checkout — `results/` is untracked and matched by `.gitignore:48`
    `Results/` (case-insensitive on Windows); only the main tree has a local
    copy. Needs a decision (track the results or skip when absent).
  - `tests/experiments/l1a_geometry_sweep_v2_visualization` (1 error,
    `protocol file SHA-256 mismatch`): the frozen preregistration sidecar
    `protocol.json.sha256` (commit `092f5fae`) and the immutable results
    (`manifest.json`, `raw-results.json`, `execution-lock.json`
    `protocol_file_sha256`) all record `64b2c58c…`, the CRLF-smudged hash;
    the committed blob has always been LF (`2a5ba9e4…`). The dashboard
    generator pins the recorded value and cross-checks it against the sidecar
    and every results file, so no pin edit can fix it without either touching
    frozen `results/` (forbidden) or changing evidence-verification semantics
    (needs the user's call). Same class as the orbit v4 post-hoc finding.
  - `tests/experiments/l1a_geometry_sweep_v2` (1 collection error) and
    `tests/experiments/l1a_field_surrogate_v1`, `_v2` (1 collection error
    each): `invalid SHA-256 sidecar for protocol.json` raised by
    `experiments/l1a_geometry_sweep_v2/protocol.py:73` — identical root cause.
  - `tests/material_fields` (4 failed in `test_spec_ledgers.py`: `raw run hash
    binding failed`, `src/cft_revival/material_fields/acceptance.py:1081`).
  - `tests/experiments/l0_surrogate_v3` (2 failed) and `l0_surrogate_v4`
    (2 failed): pre-execution tests assert `results/` does not exist, but the
    committed results now do (`real v3 results path already exists`,
    `v4 results path already exists`) — lifecycle-unaware tests.

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
## 2026-09-03 — CFT full-orbit wall-loss v4 results dashboard

- Added `generate_wall_loss_v4_dashboard.py`, its template
  `wall-loss-v4-results.template.html`, the generated
  `wall-loss-v4-results.html` (about 0.68 MB against a 1.2 MB cap) and
  `tests/visualization/test_wall_loss_v4_dashboard.py` (13 tests).
- Source: the accepted `experiments/cft_orbit_wall_loss_v4/results` bundle at
  results commit `6922a3cf` (preregistration `757e365f`). The generator
  verifies all 387 manifest-bound files, tolerates exactly the nine
  `artifacts/orbits/<case>.json.sha256` sidecars whose recorded hash is the
  CRLF form of the LF bytes, and fails on any other mismatch; every repeated
  quantity is cross-checked (terminal payload == campaign result == gates,
  summaries == orbit artifacts, strata == per-orbit outcomes, Wilson bounds
  recomputed, convergence chains recomputed from the campaign probabilities,
  |B| certificates reproduced through `PsiBicubicField`).
- Panels: headline KPIs and the 15 named binding gates; per-case Wilson table
  and dot-and-interval plot; timestep and cross-map convergence against the
  0.01 gate with overlap flags; 32-stratum heatmap with case/pooled selector
  and marginals; (r,z) channel geometry with all 4608 embedded endpoints
  histogrammed by class (wall, radial exit, injector plane, exit plane);
  reconstructed |B|/psi map for the three roles with certified and runtime
  bounds; diagnostics-not-gates (mu variation, tolerance-close share, step
  distributions, exact-zero energy error, lifecycle timing); verbatim claim
  boundary; v1/v2/v3 lineage with exact error strings, shakedown rule, v1.5/
  v1.6 fixes and the second latent v3 bug; provenance footer with commit SHAs,
  artifact hashes and LF-normalised generator/template hashes.
- Browser check: headless Chrome at 1440 px and inside a 390 px iframe host;
  zero runtime errors, no horizontal overflow, screenshots kept under %TEMP%.
