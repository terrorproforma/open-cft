# Roadmap status dashboard (offline build of the Cursor canvas)

`../roadmap-status.html` is the project's roadmap status dashboard - the eight-rung evidence
ladder over 44 rows with tabs Overview / Phases / Stage ladder / Experiments / Critical path /
Literature roadmap / Actions / Evidence / Details - as ONE self-contained HTML file. It opens from
`file://`, fetches nothing (no CDN, no fonts, no images), keeps the Cursor dark theme, and shows
exactly what the live canvas shows: the header chips (`N/44 externally validated · N in the paper ·
N/44 merged · N/44 on main`) and every table are derived at runtime from the same `ladderRows`
data the canvas carries.

## Source of record

The live canvas is `~/.cursor/projects/<workspace-slug>/canvases/open-cft-roadmap-status.canvas.tsx`
(edited by the agents at every milestone). `roadmap-status.canvas.tsx` in this directory is a
verbatim copy of it - the only difference is the line endings (the canvas is CRLF, the repo pins
`eol=lf`), which `build.mjs --sync` normalises. Do not edit the copy by hand: edit the canvas,
sync, rebuild.

## Files

| file | role |
| --- | --- |
| `roadmap-status.canvas.tsx` | verbatim copy of the canvas (imports only `cursor/canvas`; default-exports the component) |
| `cursor-canvas.tsx` | stand-in for the `cursor/canvas` runtime module: the layout / typography / surface / control primitives and `useHostTheme()` (fixed dark theme) in plain React, styled with the SDK's pinned dark tokens |
| `entry.tsx` | mounts the canvas component into `#root` with `react-dom/client` |
| `template.html` | page shell (dark body, font stack, `#root`, the inlined `<script>`) |
| `build.mjs` | esbuild bundle (React + ReactDOM + canvas, one IIFE, minified) inlined into the template -> `../roadmap-status.html` + `../roadmap-status.anchor-platform.json` |
| `verify.mjs` | headless-DOM check with jsdom: chips, all nine tabs, 44 ladder rows, first row expands, no external resources |
| `tsconfig.json` | `tsc --noEmit` of the copy against the shim (`npm run check`) |
| `package.json` / `package-lock.json` | pinned toolchain (see below); `node_modules/` is git-ignored |

Pinned toolchain (exact versions, `package-lock.json` is the lock): esbuild 0.28.2, react 19.2.8,
react-dom 19.2.8, jsdom 30.0.1, typescript 5.9.3, @types/react 19.2.18, @types/react-dom 19.2.7.
Node >= 24.15 (jsdom 30 requires it); the build itself needs only esbuild.

## Regenerate

```powershell
cd modern/visualization/roadmap-status
npm ci                              # installs the pinned toolchain into node_modules/ (ignored)
node build.mjs --sync               # copy the live canvas (CRLF -> LF) into roadmap-status.canvas.tsx, then build
#   or: node build.mjs --sync <path-to-open-cft-roadmap-status.canvas.tsx>
#   or: node build.mjs              # rebuild from the copy already in the repo
npm run check                       # tsc: the copy type-checks against the shim
node verify.mjs --expect-rows 44    # jsdom: chips, tabs, rows, no external resources
python -m pytest tests/visualization/test_roadmap_status_dashboard.py   # from modern/
```

`build.mjs` writes `../roadmap-status.html` (UTF-8, LF, no BOM) and the sidecar
`../roadmap-status.anchor-platform.json` (sha256 + byte count of the HTML, the canvas copy and the
shim, and the toolchain versions). The build is byte-deterministic: same inputs and pinned
toolchain give the same bytes on any platform (no timestamps are embedded; provenance is expressed
as content hashes in the head comment). The test rebuilds into a temporary directory and compares
byte-for-byte when `node_modules/esbuild` is present.

`--sync` derives the canvas path from the repository location (Cursor's workspace slug is the path
with the drive colon dropped and separators replaced by `-`); pass the path explicitly when the
checkout lives elsewhere.

## What is shimmed, what is not

The canvas uses `Callout, Card, CardBody, CardHeader, Code, CollapsibleSection, Divider, Grid, H1,
H2, H3, Link, Pill, Row, Stack, Stat, Table, Text, useEffect, useHostTheme, useState` and the
`CSSProperties` type. `cursor-canvas.tsx` provides all of them with the runtime's markup and dark
tokens (`canvasPaletteDark`, `categoryPaletteDark`, typography, spacing, radius), plus `Button`,
`Spacer` and `mergeStyle` for completeness. Differences from the IDE host, all deliberate:

- `useHostTheme()` always returns the dark theme - there is no editor to read a theme from.
- `useCanvasState` / `useCanvasAction` (persisted state sidecar, IDE actions) are not provided; the
  canvas does not use them. Charts, diff views, form controls, todo lists and the DAG layout are
  omitted for the same reason.
- Links open in a new tab (`target="_blank"`), as in the runtime; the only URLs in the page are the
  canvas's own github.com commit anchors plus the XML-namespace / error-decoder string constants
  React DOM carries. Nothing is fetched.

No Cursor-canvas-specific API had to be replaced in the canvas source itself: the copy is verbatim.

## The "on main" chip

`mergeTruth.mainMergedAt` (canvas data) holds origin/main's SHA once main has been fast-forwarded
onto feat/sota-foundation ("" while nothing is merged). `headlineCounts().onMain` counts the rows
whose merged state is `feature` exactly when `mergeTruth.mainHead === mergeTruth.mainMergedAt`;
the chip reads `${onMain}/${rows} on main` (green when > 0). The header meta chip and the
stage-ladder legend footnote follow the same field.
