# P2 FEM visualization learning scratchpad

## Decisions and observations

1. The accepted evidence is not one combined campaign file. Each design has an
   independently sealed third-level directory and manifest, so the dashboard
   must pin and verify all three roots.
2. Viewer files are tens of megabytes and final result artifacts are much
   larger. Embedding them verbatim would make the browser artifact impractical.
   Deterministic raster/profile projections preserve the inspected quantities
   while retaining exact source file and payload hashes.
3. Final viewers contain recovered fields and material-region ownership, while
   adaptive checkpoint sidecars contain actual topology for every level.
   Combining those two verified sources avoids inventing nested meshes.
4. Canonical payload verification can stream JSON serialization into SHA-256;
   it does not require constructing a second giant encoded byte string.
5. All three designs pass the two successive sub-percent changes and
   phase-matched domain gate. Those facts do not qualify historical or compact:
   their non-positive observed orders are the differentiating gate.
6. Dense Canvas rasterization is substantially smaller than one SVG node per
   finite element. SVG remains appropriate for the small convergence and
   domain series where vector labels and threshold lines matter.
7. A CSS minimum equal to the nominal viewport width is still wider than the
   document containing block when a classic vertical scrollbar consumes
   inline space. At a 320 px viewport, `body { min-width: 320px }` forced the
   shell beyond the observed 305 px `clientWidth`.
8. The robust narrow-layout contract is percentage sizing against the
   containing block, global border-box sizing, `min-width: 0` on grid/flex
   descendants, and explicit maximums on replaced elements such as Canvas and
   SVG. No viewport-width sizing is needed.
9. Manifest payloads can gain acceptance-audit metadata without changing their
   linked solver artifacts. Publication must still pin the committed manifest
   bytes and regenerate the dashboard identity; an HTML hash produced from an
   untracked predecessor manifest is not a valid committed-evidence target.

## Claim boundary

- Recovered vertex fields visualize accepted numerical artifacts but are not
  used to promote local interface maxima.
- Region identifiers show numerical material partitioning only.
- No displayed quantity is experimental, hardware, plasma, thrust, efficiency,
  or device-performance evidence.

## Remaining checks

- Automated regeneration, hash replay, deterministic-output, offline,
  path/secret, Python, and JavaScript checks pass.
- Browser handoff remains: open the generated file at 280, 320, 390 px and
  desktop widths; confirm `document.documentElement.scrollWidth ===
  document.documentElement.clientWidth`, then exercise keyboard, reset, theme,
  field, design, and level controls. This is a presentation check, not an
  evidence-replay gap.
