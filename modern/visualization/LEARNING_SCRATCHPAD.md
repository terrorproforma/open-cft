# Shared visualization learning scratchpad

Policy: committed with the shared `modern/visualization/` dashboards. Evidence
directories, accepted packages and experiment-local dashboards stay read-only.

## 2026-09-03 19:55 AEST - cusp topology search v3 dashboard

- [self] When the accepted campaign is a corrected re-run, the dashboard must load the
  rejected predecessor bundle too (byte-verified, expected state `assessment_rejection`)
  and show its failing gate and root cause; a generator that only refuses non-accepted
  bundles hides the lineage.
- [self] Separatrix plots over the full solver z-range squash the channel into a sliver;
  plot the axis search window (channel +/- one pitch) and clip the traces with an SVG
  clipPath instead of trimming the sealed paths.
- [tool] Edge `--dump-dom` under PowerShell redirection wrote 0 bytes while `--screenshot`
  worked; use the screenshot (and the rendered cards in it) as the JS-execution proof.

## 2026-09-03 10:20 AEST - MDO L0 campaign v1 dashboard

- [tool] `experiment_runtime` manifests key file digests as `byte_sha256` +
  `bytes` (not `sha256`); a verifier written against the wrong key fails on
  the first file. Read one manifest entry before writing the verifier.
- [self] Develop a dashboard generator against the shakedown bundle (same
  artifact layout, `experiment_id` suffixed `-shakedown`) with an explicit
  `--no-pin` flag, then pin the manifest and result commit once recorded.
- [tool] Headless Edge clamps window widths below ~500 px; use a 390 px
  iframe host page for the narrow check. `Get-Item f | Select-Object Length`
  prints nothing here; use `(Get-Item f).Length`.

## 2026-09-03 02:40 AEST — stale design gallery was an EOL artefact

- [self] Before touching any pin, classify it: hash the committed blob
  (`git show <rev>:<path>`) as-is and with `\n → \r\n`. If the recorded value
  equals the CRLF variant, the pin is a Windows `autocrlf` artefact and the
  data never changed; if neither matches, something real changed and the
  generator/physics must be diffed before any edit. Here the config blob was LF
  at every commit; only the 2026-09-01 working copy was CRLF.
- [self] Prove "provenance-only" by regenerating and reducing the diff to a
  substitution: `old.replace(OLD_HASH, NEW_HASH) == new` on the 6 MB HTML and a
  one-line `git diff -U0` on the JSON. Do not eyeball a 6 MB diff, and do not
  run `difflib.SequenceMatcher` on it (it did not finish in 3 minutes).
- [self] A generator that *refuses* on a stale upstream artifact
  (`generate_first_results.py` → "config SHA-256 does not match") turns one
  stale file into 13 fixture errors downstream. Regenerate upstream first
  (`build_design_gallery.py`), then the consumer, then rerun.
- [self] "Dashboard X is stale" reports need a byte check before belief:
  `geometry-designs.html`, `axisymmetric-results.html` and
  `plasma-topology-results.html` regenerate byte-identical; only the L0 pair
  was stale.
- [audit] Experiment-local dashboards that pin a *recorded* evidence hash and
  cross-check it against sidecars and `results/*.json` cannot be repaired by a
  pin edit when the recorded value is a CRLF hash (L1a geometry sweep v2:
  sidecar/manifest/raw/lock all say `64b2c58c…`, the LF blob is `2a5ba9e4…`).
  Repair options are (a) edit frozen preregistration/results artifacts —
  forbidden — or (b) make the verifier accept an EOL-normalised hash with an
  explicit disclosure, as the orbit v4 post-hoc audit did. That is a user
  decision, not a regeneration task; stop and report.
- [tool] `.gitignore:48` `Results/` plus `core.ignorecase=true` ignores every
  `results/` directory on Windows; tracked `results/` trees only exist where
  someone used `git add -f`. `experiments/l1a_geometry_sweep/results` was never
  added, so its dashboard test can only pass in the main tree.
- [tool] Same-basename test modules clash across `tests/<pkg>` directories and
  across `tests/experiments/<exp>` subdirectories; sweep each leaf directory in
  its own pytest invocation. Full sweep here took ~11 minutes, dominated by
  `l0_surrogate_v8`/`v9` (~160 s each), `coupling` and `fem_reference` (~65 s).

## 2026-09-03 — plasma / magnetic topology results dashboard

- [user] The user wants to *see* the topology results. The honest picture is
  mostly null: 56 characterization designs produced 1276 clustered vector nulls
  and 0 stable eligible cusps or cells; the preregistered four-cell v2 search
  produced 0 stable four-cell candidates in 128; the only "hits" are two v1
  screening-only cases under a deprecated mirror proxy. The dashboard has to
  show that distribution rather than manufacture a class taxonomy.
- [audit] Identity conventions differ per experiment: `.sha256` sidecars
  (`"{digest}  {name}\n"`), manifest `semantic_sha256` over canonical JSON of
  the *whole* file (dataset.json) versus `payload_sha256` excluding the
  integrity block, `byte-and-*` manifest entries, and orbit runtime
  `.sha256.json` sidecars. One verifier per convention; never assume one.
- [audit] `cft_wall_cusp_validation_v1/results` records `byte_sha256` values
  computed on CRLF worktree bytes; the committed blobs are LF and match only the
  recorded canonical/normalized semantic identities. Report byte identity and
  semantic identity separately instead of silently passing or failing.
- [self] Reading another branch's result via `git show <commit>:<path>` is
  deterministic and needs no checkout, but the generator must pin the full
  commit, verify the manifest byte hash, and confirm the result commit is the
  direct child of its preregistration.
- [self] "Generation time" and byte determinism conflict. Use the newest pinned
  evidence commit's author time (or `SOURCE_DATE_EPOCH`) and label it as an
  evidence snapshot, never wall-clock.
- [self] The P2 viewer raster from the FEM dashboard helper is z-major (rows are
  z bins, columns are r bins); transposing into the radial-major field layout
  is required before reusing the L1a raster/contour code.
- [tool] The Cursor browser tool created tabs that vanished immediately in this
  session, and `file://` is refused. A localhost `http.server` plus headless
  Chrome via Playwright gave real console/page-error capture, layout metrics and
  screenshots. Several agents share localhost ports; Windows lets multiple
  `http.server` processes bind the same port, so use `--directory` and an
  unusual port and verify the root listing before trusting a URL.
- [self] A `schedule(fn)` that cancels the previous animation frame drops
  updates when two controls change in one frame. Accumulate pending callbacks
  in a Set and flush them together.
- [self] Wall-normal |B_r| maxima computed here from the P2 wall line are a
  display diagnostic; the accepted cusp criterion lives in coupling v4 and was
  never promoted, so the dashboard labels those maxima as dashboard-derived.
## 2026-09-03 — CFT full-orbit wall-loss v4 results dashboard

- [audit] The v4 bundle is internally consistent except for the nine orbit
  `.json.sha256` text sidecars: the manifest records the CRLF byte hash and
  length (+1 byte) while the checkout is LF. Tolerate that exact set with the
  CRLF transform and nothing else; the orbit hash inside each sidecar matches
  the decompressed `.json.gz` exactly.
- [self] Cross-check before rendering: the summaries' `result_identity_sha256`
  identifies the coupling-export handoff case (refined-4N) where four cases
  share identical probabilities and intervals; matching on numbers alone is
  ambiguous.
- [self] Keep ordered lists (timestep policies, map roles) as explicit arrays
  in the payload; `sort_keys` JSON turns `{"N","2N","4N"}` into `2N, 4N, N`
  and the charts silently reorder.
- [self] Never set `innerHTML` of a `<table>` to markup that ends with
  `</table><p>`; the parser foster-parents the paragraph and the layout breaks.
  Render into a wrapper `<div>` instead.
- [tool] Headless Chrome (`--headless=new`) will not shrink below about 512 px
  or capture more than about 5400 px; use a 390 px iframe host page for
  narrow-layout checks and an offset iframe for page tails. `file://` is
  blocked for the IDE browser and the IDE browser tab was unavailable in this
  session, so a `python -m http.server --directory` on a free port served the
  file; several stale servers already shared port 8765 on this machine.
- [self] The KPI grid overflowed narrow viewports because unbreakable gate
  identifiers sat inside `minmax(170px,1fr)` tracks; use
  `minmax(min(170px,100%),1fr)`, `overflow-wrap:anywhere`, and move long
  identifier lists into their own flex-wrap row.
