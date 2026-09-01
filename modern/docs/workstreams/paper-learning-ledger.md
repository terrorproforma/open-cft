# Paper workstream learning ledger

## 2026-09-01 — Evidence-gated scaffold

### Established

- [user] The paper must be built in new paths only under `paper/` and
  `modern/docs/workstreams/paper-*`; shared docs, code, `FYP/`, and Git history
  are outside this workstream.
- [user] Concurrent physics work is not evidence. Present claims are limited
  to committed revision `17a121921671ac6e0012e6def2a1c9e1afe48294`.
- [self] The strongest current physics result is numerical: one deterministic
  8,192-point hypothetical L0 sweep has full Python/CUDA parity over 26
  published numeric fields and reported conservation residuals.
- [self] L0 is an algebraic conservation-reduced operating-point model with
  externally supplied charge-state, beam-current, axial-momentum, cathode, and
  PPU quantities. It has no mesh, spatial coordinate, or geometric design
  input.
- [self] The one-shot CUDA and separate Python timings are uncontrolled and
  cannot support an acceleration or regression claim.
- [self] The optimization campaign has a detailed machine-readable policy, but
  its benchmark `results` field is null. Strategy names and budgets are methods,
  not outcomes.
- [tool] Quarto and Pandoc are unavailable. Python 3.12 and MiKTeX
  `latexmk`/`pdflatex`/`bibtex` are available, so the scaffold uses conservative
  LaTeX plus standard-library policy scripts and disables MiKTeX installation.
- [tool] MiKTeX's `latexmk` executable is present but unusable without Perl.
  The build wrapper detects that state and uses direct `pdflatex`/`bibtex`
  passes; no package or script engine is installed.
- [tool] This PowerShell host does not accept `&&`; use separate commands or
  PowerShell-compatible sequencing.

### Decisions

- Pin every verified manuscript claim to a claim ID, committed source path, and
  Git blob in `paper/evidence/claims.json`.
- Represent L1, L2, and L3 as visible `EvidenceGate` blocks with a separate
  machine-readable registry. A working-tree result cannot open a gate.
- Keep physics levels L0--L3 distinct from optimization information sources
  F0--F3; future manifests must state the mapping.
- Treat non-DOI conference papers explicitly as non-DOI records rather than
  inventing an identifier. For conference papers whose DOI resolves an arXiv
  record, state that scope in the bibliography note.
- Require all bibliography entries to be cited and every citation key to
  resolve.
- Let figure/table contracts name planned outputs without generating empty
  graphics or synthetic data.

### Guardrails

- Never characterize L0 as one-dimensional, geometrically predictive,
  physically calibrated, or experimentally validated.
- Never translate implementation parity, conservation closure, or a successful
  solver status into physical accuracy.
- Never compare the current diagnostic timings as a GPU speed benchmark.
- Never equate the campaign's historical `total_efficiency` objective with any
  explicitly bounded L0 efficiency.
- Never fill a planned result section with plausible numbers. Open it only
  after a committed manifest passes its required evidence list.
- Preserve failed/rejected cases and uncertainty components in future result
  manifests and displays.

### Open evidence

- A machine-readable manifest for the already reported L0 batch would eliminate
  manual table transcription and enable generated residual figures.
- L1 needs sourced field equations, geometry/material identity, manufactured
  solutions, and mesh/domain convergence.
- L2 needs closure provenance, coupling conservation, convergence, and
  calibrated discrepancy.
- L3 needs verified PIC inputs plus preregistered experimental data and
  measurement/facility uncertainty.
- Optimization publication needs executed surrogate diagnostics, a frozen
  F3-verified hypervolume definition, same-cost baselines, repeated seeds, and
  confidence intervals.

## 2026-09-01 — Strict evidence and deterministic-build correction

### Established

- [user] A manifest path is not evidence. Accepted L1--L3 manifests must have a
  recognized type/schema, resolvable ancestry, committed matching content,
  required file roles with Git blob and SHA-256 bindings, and gate-specific
  metrics.
- [self] Detached claim markers are bypassable because the surrounding prose
  can change while the ID remains. The corrected contract authorizes the exact
  macro body and location, or a specific generated artifact with a sidecar.
- [self] The committed accepted HTML at `41bf909` embeds 40 complete columns
  for 8,192 reconstructed reference-sweep records and a dataset SHA-256. It
  does not preserve raw CUDA device buffers, and the run revision remains
  unrecorded.
- [self] Python and Warp share canonical preprocessing. Their agreement is
  cross-backend implementation evidence, not fully independent replication.
- [tool] Git object content uses normalized line endings while a Windows
  checkout may use CRLF. Portable evidence binds the Git blob plus SHA-256 of
  the blob bytes; checkout-byte hashes are retained only where already embedded
  in a committed artifact.
- [tool] `SOURCE_DATE_EPOCH=1788270043`, UTC, omitted PDF dates/trailer ID,
  suppressed pdfTeX information, and clean direct TeX/BibTeX passes produced
  byte-identical PDFs.

### Guardrails

- An accepted manifest must live under `paper/evidence/manifests/`; a README,
  unsupported type/version, short fake revision, missing metric, or missing
  role fails closed.
- Claim-bearing numerical, experimental-accuracy, validation, parity, and
  accelerator-performance prose must be exact registered text. Reusing a valid
  ID with different text is a failure.
- Verified figures and tables must be regenerated from accepted evidence and
  match both output and provenance sidecar hashes. Manual edits are rejected.
- Timing data may be reproduced only with its uncontrolled-diagnostic caveat.
- Angus Muffatti is the current known original-project author. Coauthor order,
  contributions, affiliations, correspondence details, and manuscript approval
  remain human gates.
- `paper/build/` and Python caches stay local; evidence manifests, generated
  TeX sources, and sidecars must remain trackable for integration.
