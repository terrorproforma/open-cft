# Shared visualization learning scratchpad

Policy: committed with the shared `modern/visualization/` dashboards. Evidence
directories, accepted packages and experiment-local dashboards stay read-only.

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
