# Visualization learning scratchpad

Policy: `COMMITTED` with this visualization workstream.

## Durable guardrails

- Keep v1 and committed v2 experiment/source/result evidence read-only.
- Validate both checkout sidecar SHA-256 and Git-clean committed identity.
- Account for Git text filters: committed blobs use LF while the Windows
  checkout may use CRLF. Use `git diff --quiet <commit> -- <paths>` to verify
  committed equivalence and sidecar hashes to verify exact checkout bytes.
- Require preregistration commit → execution-lock time → direct-child results
  commit ordering.
- Recompute tolerance-aware dominance, five representative roles, four-case
  coalescence, and all seven gates from raw evidence.
- Never describe CUDA floating replay as bitwise reproducible.
- Keep sweep points on high-DPI canvas rather than per-point DOM.

## 2026-09-02 — preregistered v2 dashboard

- [user] Ownership is limited to new v2 visualization and matching test files.
- [tool] Direct Git-blob byte comparison initially disagreed with the CRLF
  checkout despite Git reporting the path clean; Git-filter-aware comparison
  fixed the false failure without weakening exact sidecar checks.
- [self] The dashboard separately labels bitwise identities, run-specific
  artifact hashes and tolerance-based CUDA numerical replay.
- [self] The four embedded field artifacts preserve all five role labels,
  including the coalesced energy/boundary roles on case 000.

## 2026-09-02 — mobile selector overflow

- [user] A 390 px viewport exposed native select intrinsic sizing: the
  representative label expanded the control to 465 px and document to 488 px.
- [self] Constrain the select, its label, every flex/grid child and containing
  control row with `min-width: 0` and `max-width: 100%`; stack controls on
  mobile rather than relying on flex wrapping alone.
- [self] Visual ellipsis must not shorten accessible content. Preserve complete
  labels in option text, `title` and `aria-label`, including future long roles.
- [self] Regression coverage should include 320 px-safe structural rules and a
  synthetic label much longer than current evidence labels.
