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
## 2026-09-03 — Wall-loss v4 evidence

- [self] Generated TeX must be macro-only and the section must reference
  macros only; a regex test that strips comments, `\label`/`\ref` arguments,
  macro names and layout dimensions and then asserts "no digits" catches every
  hand-typed number, including layout constants that had to be exempted
  explicitly.
- [self] Long `\texttt` identifiers (`snake_case` classifications, hashes)
  overflow the line; format identifiers with `\allowbreak{}` after `\_` and
  `-`, and never put a box directly after a run-in `\paragraph` heading (the
  heading joins the box's paragraph and overflows by the heading width).
- [self] Provenance sidecars that hash a generator's working-tree bytes are
  CRLF-sensitive; after the repository-wide `eol=lf` pin the L0 sidecar was
  stale on every fresh checkout. Hash LF-normalised bytes in new generators.
- [tool] Headless Chrome (`--headless=new`) clamps the window to ~512 px wide
  and ~5400 px tall; test narrow layouts through a 390 px iframe host page and
  capture tails through an offset iframe.
- [user] Draft results stay outside `manuscript.tex` until a claim record and
  gate admit them; `manuscript_integration.status` in the evidence file
  records that boundary explicitly.

## 2026-09-03 — Admitting the wall-loss v4 campaign

- [self] A numerical campaign that is neither L0 nor an L1--L3 level needs its
  own gate kind, not a forced fit into `GATE-L1`: the L1 manifest schema
  demands an L0 mapping and would license topology and geometry-response
  claims that the campaign cannot support. `numerical-campaign` gates reuse
  the typed-manifest machinery verbatim and add `opens_level: null`.
- [self] `\input{sections/...}` is a bypass of the claim matrix unless the
  checker flattens it first: `extract_macros` and `find_unregistered_claims`
  only saw `manuscript.tex`, so a section could carry unregistered
  quantitative prose. Flatten before every prose check and let
  `_heading_at` resolve the section's own `\subsection` as the claim location.
- [self] Exact-text claims that contain macros work unchanged
  (`_normalize_tex` compares the TeX source), so a registered claim can be
  fully macro-bound and still be verified character for character.
- [self] Build `authorized_tex` by extracting the `\EvidenceClaim` bodies from
  the flattened manuscript and normalising them, never by retyping; the first
  attempt to hand-copy a body would have failed on a `{}` after a macro.
- [tool] `_parse_group` keeps `%` comments inside a macro body; a `%` used to
  suppress a newline in `\newcommand{\WallLossEvidenceRevision}` reached the
  comparison until the checker stripped comments first. A trailing newline
  inside a `\newcommand` body renders as a space (the existing
  `\EvidenceRevision` does this), so the checker also strips whitespace.
- [self] Claim IDs in TeX comments trip the detached-ID rule
  (`\bCLM-\d+\b` on masked text) because comments are not stripped there.
  Keep IDs out of comments rather than loosening the rule.
- [self] `Overfull \hbox` is fatal for `build.py`; a one-word rewording of the
  sentence that precedes a 40-hex `\texttt` hash moved a 2.9 pt overflow to
  zero. Test both the manuscript and the standalone driver, and give the
  standalone driver the same `microtype` as the manuscript so the two agree.
- [self] `require_committed=True` means the repository check can only pass
  after the manifest is committed; run the checker before the commit to see
  every other error, commit, rerun, and amend locally if anything remains
  (never force-push).

## 2026-09-03 — Admitting the sweep and the two topology nulls

- [self] A null result needs a gate kind whose `accepted` status cannot be
  read as "finding accepted". `numerical-screening` carries a
  `recorded_outcome` (`accepted-screening` / `preregistered-null` /
  `recorded-characterization`) that gate, manifest, evidence file and
  generator must all spell identically; the checker refuses any drift, and the
  claim text says "not shown stable" / "undemonstrated", never "does not
  exist".
- [self] Section headings are scanned by the literal-digit rule: "L1a" in a
  `\subsection` title failed the check. Rename the heading rather than exempt
  headings; the model level is rendered from a macro (`\SwpModelLevel`,
  `\FcnFieldModelLevel`, `\TchFieldModel`) inside the prose.
- [self] The unregistered-quantitative heuristic is case-insensitive:
  "v2-031, v2-063" (digits, comma, space, "v") and "+1, a nearby" both match
  `\d[\d,]* \s [WVASN]`. Render identifier lists as `\texttt{}` items and join
  clause lists with ";" instead of loosening the heuristic; a new formatter is
  cheaper than a weaker rule.
- [self] One generic `Bundle.verify` with the audited files as data is enough
  for both EOL defects; the checker then re-derives the CRLF digest on disk
  and requires both digests to appear verbatim in the experiment's own audit
  module, so the paper's tolerance can never drift from the experiment's.
- [self] The four-cell bundle's own summary records `gpu_replay_pass_count: 2`
  of 4; the field components reproduced but a residual diagnostic exceeded its
  limit on two candidates. Report it as recorded in the results claim rather
  than omitting it; the topology null does not depend on it, and the paper
  says so.
- [self] Hash-bound lineage (superseded proxy search, failed criterion
  validations) belongs inside a registered non-claim with its own manifest
  block (`lineage_files`, revisions of their own) and `lineage-` roles; it is
  quoted, never cited as evidence, and the checker validates the blobs anyway.
- [tool] `pdflatex -output-directory='$out'` with single quotes in PowerShell
  writes into a literal `$out` directory inside the repo; double-quote the
  variable and move any stray directory out before committing.
- [tool] A `$env:SOURCE_DATE_EPOCH` exported for a trial TeX build persists in
  the shell and changes the plasma-topology dashboard's footer time, making
  its byte-identity test fail; unset the build variables before running
  `modern/tests`.
- [tool] Long `\texttt{}` table cells without break points overflow `p{}`
  columns silently until the log is read; `>{\raggedright\arraybackslash}p{}`
  columns remove the underfull noise so the one real overfull stands out.

## 2026-09-03 — Admitting the MDO L0 campaign v1

- [self] Reusing a gate kind is a definitional test, not a convenience: the
  `numerical-campaign` definition ("one accepted, preregistered numerical
  campaign about a declared component model") fits the optimisation campaign
  because L0 + the declared closure CL-1 is the component model; the
  screening studies did not fit it because they screen a design space and two
  are nulls. Record the fit in a `kind_justification` field on the gate and in
  the gate-kind description rather than leaving it implicit.
- [self] "Optimiser evidence, not performance evidence" has to be carried by
  the artifacts, not only by prose: the manifest's policy metrics
  (`thruster_performance_claim_forbidden`, `design_recommendation_forbidden`,
  `optimiser_superiority_beyond_recorded_budget_forbidden`,
  `geometry_variables_excluded`, `closure_declared_not_derived`,
  `campaign_policy_benchmark_results_populated: false`) are fixed values the
  checker refuses to see changed, and the scope claim's `non_claims` must
  appear verbatim in the section.
- [self] A bundle whose frozen files are pretty-printed while the sealed copies
  are canonical JSON is not a byte match; compare the parsed payloads and say
  so in the manifest (`frozen_files_equal_sealed_copies`) instead of relaxing
  the blob binding of the frozen files at the preregistration commit.
- [self] An independent extraction of the same bundle (the results dashboard)
  is worth binding: pin its revision, require its embedded payload to equal
  the sealed artifacts before writing any macro, and bind its files by
  LF-normalised SHA-256 equal to the blob at that revision. It costs one
  fail-closed dependency and buys a second reader of every headline number.
- [self] Numbers that live only inside a protocol's free-text disclosure (the
  four-cell solver probe: 13/80, residual floors, seconds per solve) can still
  be macro-bound: parse them with a fixed regular expression as derived
  macros whose derivation names the pattern and the pointer, and fail closed
  if the text stops matching.
- [self] The literal-digit rule shapes the prose: "L0", "CL-1", "v4",
  "Xe$^{2+}$", "$p_1..p_4$", "16 of 64" and "U[0, 0.45]" are all digits.
  Render identifiers through `ident` macros (`\MdoFidelity`, `\MdoClosureId`),
  refer to the wall-loss campaign by `\ref`, count things with macros
  (`\MdoCellCount`, `\MdoTailCount{} of \MdoSampleCount`) and put the closure
  formula in the manuscript's section intro, where digits are allowed and the
  quantitative heuristic sees no unit after them.
- [self] The claim text must report the artifact, not the task brief: the
  Jeffreys scenario gives a maximum thrust of 2.70e-9 N, not "zero thrust";
  the lifecycle took 27.3 min, not "28 min". Keep the words ("that is no
  beam") and let the macro carry the number.
- [self] A statement that was true at the previous admission can become false
  at the next one: the Limitations sentence "no admitted hypervolume result or
  baseline comparison" had to be rewritten, and a test now asserts its
  absence. Re-read every boundary sentence of the manuscript when a new gate
  opens.
- [tool] `pdflatex` reports an overfull table as "lines N--N" of the *section*
  file at the macro-invocation line, not of the generated file that holds the
  tabular; look for the `(sections/...)` context in the log to find the file.
- [tool] Eight-column tables with `\times10^{}` cells overflow `\footnotesize`
  by ~230 pt on a 468 pt text width; `>{\raggedright\arraybackslash}p{}` for
  the two text columns, `\shortstack` two-line headers and `\scriptsize`
  bring them under width without shortening any value.

## 2026-09-03 — Admitting the four-cell power-balance closure analysis

- [self] A finding that was never executed under a protocol needs its own gate
  kind; forcing it into `numerical-campaign` or `numerical-screening` would
  have made the checker's bundle/preregistration rules meaningless. Define the
  kind by what “accepted” means (“the derivation and its numerical verification
  are admitted as recorded”) and by what it does not accept (the correction,
  any thruster statement, any physics level).
- [self] “Recompute the verification” has to be budgeted: the document's
  ladder (5 starts, 600 iterations) costs ~13 s per rung; one start reaches
  the same stall floor within 6 % in 2.7 s. Declare the reduced protocol, the
  tolerance (25 %) and the recorded precision (3 significant digits) in the
  generator, and cache the recomputation per process so 30 checker/test calls
  cost one run.
- [self] Least-squares stall floors are not bit-reproducible quantities (they
  move with start count, iteration budget and libm); record them rounded and
  compare to the document at a declared tolerance, and say in the section that
  documented and recomputed floors agree within tolerance, not bitwise.
- [self] A `sci`-formatted macro carries its own `$...$`; using it inside a
  caption's math gives “Missing $ inserted” two files away from the cause.
  Keep macros that render math outside `$...$`.
- [self] `\allowbreak` inside `\texttt` gives break points but no stretch, so
  long ledger expressions still overflow; `\hspace{0pt plus 1.5pt}` after
  each operator gives both.
- [self] Verify the brief's numbers against the files before writing claims:
  the legacy acceptance defect is “flags 1–3 accepted by status, flag 4
  rejected”, not “flag 4 accepted”; the `+IE` cusp terms are on line 136 of
  the blob while the document says 137 (the anode term). Bind both lines and
  make the checker require the documented line to lie in the span.
- [self] Digits in the displayed closed form are structural indices; the
  checker strips sub/superscripts and macros and refuses any remaining digit,
  so the coefficient and row index are macros (`\FccAnodeFallCoefficient`,
  `\FccGlobalRowIndex`) and the equation lives in the manuscript section where
  `\cite{Kornfeld2007}` (digits) is allowed.
- [tool] Per-file `git rev-parse`/`git show` cost ~170 ms each on this host
  (42 calls = 7 s); one `git ls-tree -r -z` per commit plus one
  `git cat-file --batch` binds 14 files in about a second.
- [tool] Importing the package under test from the checkout must be verified
  (`module.__file__` under `modern/src`); an installed `cft_revival` elsewhere
  on `sys.path` would silently recompute with the wrong code.

## 2026-09-03 - Admitting the wall-loss geometry screening

- [self] Verify the brief's qualitative trend before writing it as a claim:
  "longer channels / larger wall radius lose least" holds for the three
  least-loss designs but not for the population (Spearman rho -0.05 with
  length, -0.12 with radius over 96 designs; the strongest geometric
  associations are stage pitch +0.36 and stage count -0.31). Report the
  extremes and the rank correlations as macro-bound observations and say
  explicitly that none is a design rule.
- [self] Read the implementation, not only the protocol: the protocol says the
  refined re-solve and cross-resolution diagnostic apply to every design, but
  `designs.py` computes them only for `include_refined` representatives and
  the check passes vacuously otherwise (92 nulls in the dataset). Derive the
  count from the data (`WlgCrossResolutionDesigns` = 4) and state the scope in
  the section, Limitations and manifest.
- [self] A new outcome value is cheaper and more honest than a new gate kind
  when the object is still a design-space screening on L1a fields: extend
  `SCREENING_OUTCOMES`, the kind description and the notation, keep the kind's
  bundle/preregistration rules, and give the study its own manifest type and
  checker because its bundle shape (experiment_runtime sidecars, dashboard) is
  the MDO shape, not the sweep shape.
- [self] Re-implement the interval estimator operation for operation and
  require exact equality: the orbit_mc Wilson formula reproduces every sealed
  bound bit for bit (784 case estimates, 1,536 cell estimates); a tolerance
  would only have hidden a defect.
- [self] A statement that was true at an earlier admission ("no consumer model
  has ingested it", CLM-016) turns false when a later admission closes the gap;
  amend the registered claim and its section verbatim together, give the new
  `\ref` target to the earlier standalone driver, and assert the old wording's
  absence in a test.
- [self] Keep the section free of another study's macros (`\Wlf...`): bind the
  cross-study contrast through this bundle's own frozen protocol disclosure
  (`\WlgVFourHeadline`) inside the section and use the other study's macro only
  in the Discussion claim that is bound to both manifests.
- [tool] Compile the manuscript into `%TEMP%` with the same flags as `build.py`
  before the commit gate forces a round trip: `build.py` refuses to compile
  while the manifest is uncommitted, and the only overfull box (6.5 pt in a
  12-column `\scriptsize` table) was found this way.
- [tool] PowerShell mangles quotes and regex backslashes in `python -c`; put
  even ten-line probes into `%TEMP%` scripts (`explore.py`, `verify_numbers.py`,
  `build_manifest.py`, `claims_snippets.py`) and run them by path.

## 2026-09-03 - Admitting the MDO L0 campaign v2 (screened design catalogue)

- [self] Verify the brief's aggregate wording against the per-run records
  before writing it: "91 infeasible, all BO boundary probes" is 88 in the
  Bayesian runs and 3 in NSGA-III (0/2/1). The claim states the per-optimiser
  split as three macros instead of the aggregate sentence.
- [self] "Seed 101 never found design 49" needs the initial design checked:
  the 32-point shared initial design of seed 101 did not contain design 49
  while seeds 202 and 303's did, so the stall is an exploration effect of the
  categorical index, not an acquisition failure alone; record both facts as
  derived macros (`MdbBoAMissedInInitial`, `MdbBoBMissedInInitial`, ...).
- [self] When a second campaign shares a checker's shape, factor the shared
  checks into one function with a small spec (`_MdoFamily`: module, prefix,
  revision macro, metric maps, required macros) and keep the old entry point
  as a wrapper with the same signature and error strings; the v1 admission
  tests then pass unchanged and the v2 checker adds only its extra bindings.
- [self] A comparison table that reads another campaign's bundle must verify
  that bundle exactly as its own admission did (byte-for-byte against its
  manifest, pinned SHA-256 and results commit), bind those files in the
  manifest with their own roles, mark every such macro with its bundle and
  name it `...Prior...`, and prove the shared reference frame (reference
  point, scales, unit rows, CVaR, domain, constraint) as a fail-closed boolean
  macro before calling hypervolumes comparable.
- [self] The macro-only rule also bans "CL-1", "L0", "v1", "2N" and "F9" in
  the section: derive short names from the protocol's own keys
  (`MdbClosureShort` = the `protocol.closures` key whose id matches) and use
  tokens (`Nine`, `Ten`, `TwentyTwo`) for alphabetic macro names.
- [self] Guard against `sys.path` shadowing when writing probe scripts to
  `%TEMP%`: a scratch file named `inspect.py` shadowed the stdlib module and
  broke `dataclasses` on import of `check_paper`; name probes distinctively.
- [self] A text macro copied verbatim from a protocol ("energy, 4 objectives,
  32 points, seed 1") is scanned by the unregistered-quantitative heuristic in
  the generated file; bind numbers, not sentences.
- [self] Stale boundary sentences appeared in three places this time: CLM-035
  ("link ... is open"), CLM-052 ("future work") and the manuscript's MDO
  Discussion paragraph and Limitations ("no surrogate or optimisation
  consuming the dataset is admitted"); grep claims.json and manuscript.tex for
  "open", "future work" and "not admitted" before every admission and update
  the earlier admission's test that asserted the old wording.
- [tool] `git diff-tree --no-commit-id --name-only -r <commit>` is the
  cheapest proof of a results-only commit (v1 audit F9); bind the count and
  the outside-`results/` list as derived macros so the paper's statement is
  recomputed at every check.
## 2026-09-03 - Admitting the cusp topology search v3.1 and amending the topology Discussion

- [self] Verify a brief's topological statement against the sealed reference
  before writing it as a claim: "every vector null sits on the axis" was false
  for the sealed characterization dataset (20 of 200 in-channel X roots are
  off-axis bilinear roots at 0.16-0.54 of the wall radius). Bind the split as
  derived macros and state exactly what the data show ("none at the wall",
  "excluded by the frozen rule"), not the slogan.
- [self] "By construction" needs two bound facts, not a phrase: the definition
  (the literature wall cusp is a separatrix intersection, so a wall-side null
  search cannot find it) and the source policy (the sealed v2 strength ratios
  16-42 % leave one axis sign change, so 128/128 one cusp). Both are macros
  read from the frozen protocol and the sealed v2 dataset.
- [self] A recorded rejection is admitted as lineage by binding its whole
  bundle at its own revision, byte-verifying it, reproducing its audit from
  sealed data inside the generator, and giving the gate a `lineage` block with
  `cited_for_numbers: false` that the checker enforces; the histogram-equality
  boolean is the recomputed proof that the corrected campaign changed the
  recording layer only.
- [self] When a shared policy flag was defined against a definition the new
  study does not evaluate (`stable_multicell_wall_cusp_topology_demonstrated`),
  do not reuse it with a flipped meaning; add explicit flags for the new
  boundary (`confinement_cells_demonstrated`,
  `frozen_definition_nulls_remain_true`) and leave the old gates' flags as
  they were.
- [self] Recomputing a sealed derived quantity exactly needs the producer's
  convention: `distance_to_nearest_stage_gap_m` uses the half-pitch end gaps
  beyond the stack, not only the interior midpoints. Fail closed, then read the
  one failing design before changing the rule.
- [self] Version tokens (`v3.1`, `v2`, `L1a`, `RK4`) are digits under the
  macro-only rule; derive them from the experiment ids and set keys as macros
  or spell them out ("fourth-order").
- [tool] `\texttt{}` identifier lists (gate names, review paths) overflow the
  line even with `\allowbreak`; render long identifier lists as prose and keep
  a single identifier per `\texttt{}` at a sentence end.
- [tool] Validating N revision-bound files with per-file `git rev-parse` +
  `git cat-file` costs ~4 git calls per file; `git ls-tree -r -z <rev> -- paths`
  + one `git cat-file --batch` per revision binds 65 files in two calls.
- [tool] PowerShell `Measure-Command { python ... }` swallows the script's
  stdout; time with Python's `time.time()` inside the script instead.

## 2026-09-03 - Admitting the L1a geometry sweep v3 (HEMP-like regime)

- [self] Reuse an existing recorded outcome when the new study is the same kind
  of object (same rules, gates and field level, one more descriptor) and
  justify the reuse on the gate; add a new outcome only when the sealed status
  names a different kind of thing. Each admission still gets its own manifest
  type because the metric set differs.
- [self] Admit a preregistered hypothesis at its recorded outcome with explicit
  boolean macros (`hypothesis_h1_held: false`) that the checker requires to be
  false, and word the claim as recorded ("did not hold as preregistered";
  "upper envelope") with the checker enforcing the phrases; the generator
  refuses if either hypothesis turns out to hold.
- [self] Split end and interior cusps before pooling a per-cusp ratio and bind
  the split as derived macros; the finite-stack end effect (rho/I_1 0.80 vs
  0.87) is the recorded reason the envelope reading holds, and the generator
  asserts the end-cusp median is below the interior one.
- [self] Numbers computed with numpy in the experiment (pairwise sums) do not
  recompute bit-exactly with `math.fsum`; compare sums within a declared
  relative tolerance (1e-9) and keep counts, histograms and medians exact.
  Record the tolerance in the evidence file's `recomputation_rule`.
- [self] A macro-only section needs symbols with digits (`I_1`, `R^2`, `H1`,
  `b_3/b_1`, `L1a`, `v3.1`) as macros; emit them through a whitelisted
  `symbol` formatter so no arbitrary TeX can enter through the evidence file.
- [self] A recorded analysis outside any bundle (the review's check output) is
  bound as a definition-source file at its commit and read into derived
  macros; the checker verifies the blob, so its numbers are macro-bound without
  inventing a gate kind.
- [tool] booktabs `\midrule` treats a following `[` as its optional argument:
  never start a table row with a bracket; wrap intervals in `$...$`.
- [tool] Text macros copied from protocol prose can carry quantitative phrases
  ("128 points") that the unregistered-claim scan flags in the generated file;
  slice the clause or derive a shorter statement.
- [tool] `pdflatex` in the MiKTeX install accepts `--disable-installer`; the
  standalone-section test compiles twice and asserts no overfull box, so table
  column widths must be fixed before the suite passes.

## 2026-09-03 - Re-scoping the wall-loss reflection statements

- [self] "Zero reflections" is a statement about launch positions before it is
  a statement about a field: in a PPM mirror field the |B| maxima sit at the
  magnet centres, so launches there cannot mirror. Check the launch design
  against the stage centres (both are hash-bound: frozen protocol and topology
  record) before reading a reflection count as physics.
- [self] Re-scope by replacement, not by hedging: delete "unsupported" /
  "field-specific" and state what the field predicts at those launch positions,
  with the recorded launch-cell counts as macros; then make the checker reject
  the superseded wording so it cannot come back.
- [self] A recorded analysis (the review's committed check output) becomes
  citable by binding it as a definition-source file of an admitted manifest and
  deriving macros from it; the claim then says "recorded analysis of sealed
  orbits", never "new orbit result".
- [self] When an amended Discussion claim gains macros from a new manifest, add
  the manifest id and update every test that pins that claim's manifest set
  (three admission tests here); the pins are the point, so change them
  deliberately and say why in the test comment.
- [tool] `pdflatex` compile of the whole manuscript takes ~3.5 min per clean
  build here; run `verify_reproducible_build.py` in the background alongside
  the test suite and read both logs afterwards.

## 2026-09-04 - Admitting the wall-loss geometry screening v2 (catalogue cells)

- [self] A post-hoc event that is not an audit (the runtime's recovery published
  the manifest after the locked attempt died at publication) is bound as a
  *disclosure source* group: the disclosure note plus the code that performed
  the recovery and the cap that prevents a recurrence, at the commit that added
  them, with the disclosed hashes and counts verified against the committed
  bundle. `posthoc_audit` stays null and the checker refuses one; the
  results-bundle block carries `manifest_published_posthoc: true` and
  `published_inside_locked_attempt: false` so the boundary is machine-readable.
- [self] Verify a disclosure by regex against the bundle, not by trusting it:
  the disclosed `manifest_byte_sha256` must equal `sha256(manifest.json)`, the
  disclosed `terminal_byte_sha256` the manifest's binding, the file/artifact/
  transition counts the inventory, the results commit prefix the record commit,
  and the descriptor arithmetic (files > cap > pin cap) must describe the
  overflow. Also prove "nothing changed" with git: the record commit touches only
  `results/`, the disclosure commit only Markdown under the experiment.
- [self] Replay a paired control from the endpoint tables, orbit by orbit
  (launch key of the 2N case -> termination of its N-step partner), rather than
  trusting per-cell counts; the standard error must be recomputed with the
  experiment's own construction (per-design +1/-1/0 differences, `pstdev`).
- [self] When the brief's numbers do not match the artifacts, report the
  artifacts: the "6 straight-exit designs at 1.0" were 4 of 6, and the exit-side
  "direction structure 0.53-0.61 / 0.37-0.47" is really a one-sided split
  (one launch direction ~1, the other ~0, wall side = last-stage polarity in
  82/90). Derive the actual structure as macros and say so in the devlog.
- [self] Reuse the existing recorded outcome when the study is the same kind of
  object with a different launch design; justify on the gate and have the
  checker require the word "reused" in the justification. A new manifest type is
  still needed because the file roles and the metric set differ.
- [self] A macro-only section that must say "2N" needs the time-step names as
  macros (derived from the protocol's policy keys), and must refer to earlier
  campaigns by section reference, never by version token (v1/v4 carry digits).
- [tool] The first `check_paper.py` in a fresh worktree took 350 s on a cold
  file cache against 107 s warm; profile before optimising (the new generator
  itself is ~6 s: 17k files, 51 MB, 1105 gzipped endpoint tables).
- [tool] `unittest discover -s paper/tests -t .` fails ("start directory is not
  importable") because `paper/` has no `__init__.py`; use the documented
  `-s paper/tests` form.
- [tool] PowerShell double-quoted strings pass `\\` through unchanged, so an
  inline `python -c` regex with `\\\\newcommand` silently matches nothing;
  write the check as a temp `.py` under `%TEMP%` instead.
