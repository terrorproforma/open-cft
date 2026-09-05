# Agent Scratchpad

## File Policy

- Current policy: `COMMITTED`
- Rationale: User requires the learning-scratchpad loop; current task forbids edits to tracked files, so this new local note is not staged or committed.

## Retained Lessons

- [self] Do not extrapolate axis-fitted field harmonics to the wall: I_1(k kappa r)
  amplifies aliasing noise 50-80x for k = 5 at x_w ~ 1. Fit harmonics on the wall
  profile; the fundamental alone reproduces the wall B_r of a PPM stack to 10 %.
- [self] Before interpreting reflection statistics, locate the launch cells relative to
  the field extrema: v4 launched 0.5 mm from the magnet centres (|B| maxima, no mirror
  possible) and saw 0 reflections; the L1a screening launched some cells near the nulls
  and saw 32-88/128. Same physics, different launch positions.
- [tool] PowerShell `>` redirect of Python stdout produces UTF-16 with BOM; read it back
  as utf-16 or write the file from Python.
- [self] Before any one-shot preregistered campaign, run a labelled NON-EVIDENTIARY
  shakedown of the full production path on the REAL input data (real P2 field,
  real launch manifest, full validator). Synthetic preflights passed three
  times while the real field exposed a fresh defect each time (orbit v1/v2/v3).
- [self] "One immutable attempt, no rerun" is only cheap if the code has already
  seen real data. Otherwise each latent bug costs a whole prereg cycle.
- [self] Confirmed twice in one night: the v4 shakedown's first run caught a
  `zip(..., strict=True)` bug in the assessment stage that no synthetic
  preflight or unit test reached. Shakedowns must run the assessment and
  export stages too, not just integration.
- [self] Preregistration commits must be experiment-path isolated; commit
  tests separately (or inside the experiment dir) so `_bind_preregistration`
  accepts HEAD.
- [tool] Every text write that feeds a byte hash needs `newline="\n"` on
  Windows (`write_text(..., encoding=...)` alone emits CRLF). The v4 bundle
  recorded CRLF sidecar hashes that Git normalised to LF.
- [self] Pooled equal-weight probabilities over strata are design averages;
  report per-stratum structure (here bimodal by cell) before any pooled
  number.
- [self] PIC operating points must be sized from the MEASURED kinetic loss
  time (tau_i,eff 2.4 us here), not a Bohm bound; check nu_iz*tau < 1 or
  the run is an avalanche with no equilibrium.
- [self] Static-neutral PIC has no physical steady state in this channel:
  either avalanche (n_g >= 3.4e19) or no ignition. A neutral inventory
  model is required for a real plateau.
- [tool] Long GPU runs: launch detached with checkpoints + status.jsonl and
  return; never keep a subagent waiting on a multi-hour job.
- [user] Every progress bar must carry its basis and a "why" (building /
  failed gates / blocked by physics / needs data / accepted). Stale
  pre-audit bars next to audited phases read as dishonest.
- [user] For any blocker, search the literature for SOTA approaches and
  documented pitfalls before iterating further; cite verified DOIs only.
  Reviews live under `modern/docs/literature/`.
- [user] The canvas must answer "where are we" on the first screen in
  <10 s: one-line header + chips, stage strip, <= 8-row "right now" table,
  <= 6 key findings, numbered "next". Everything else collapsed into a
  Details tab; archive old milestones. Never append run-on status prose to
  the header again.
- [user] Prefer STAGE LADDERS over percentages for status: Specified ->
  Code written -> Tests pass -> Runs on real inputs -> Preregistered run
  recorded -> Numerically accepted -> In the paper -> Externally validated,
  plus a merged flag and stop-annotations (RUNNING / FAILED GATES / NULL /
  BLOCKED BY PHYSICS / NEEDS DATA). Percentages only inside the audit card.
- [tool] CUDA-graph capture freezes every kernel SCALAR argument at capture
  time; any per-step host input (neutral density, rates, counts) must live
  in a device array. The v1.4 graph baked in the MCC n_g and silently ran
  stale for two plume attempts. Test: graph vs direct with a CHANGING input.
- [self] Calibrate a fail-closed ignition/plateau gate on a run that
  succeeded as well as on the failures; a gate tuned on failures alone
  rejected the one run that had actually ignited.
- [self] In a collisionless electrostatic plume there is no cross-field
  path: an off-axis cathode must sit on the channel-connected flux tube or
  nothing couples. Trace field lines before placing sources.
- [self] A saturated label (all interior cells at 1.0) is a finding that
  ends a chain, not a dataset to fit: stop the surrogate/MDO iteration and
  move the closure source to the next fidelity (PIC).
- [tool] Windows CRT caps open descriptors at 8192; bundles with >8k files
  need the pin cap (now 4096 in experiment_runtime) or chunked inventories.
- [tool] orbit_mc's `wilson_interval(0, n).lower` is a positive round-off
  for 734 of the first 4000 n; validators requiring lower <= p reject
  zero-count cases at those n. Size cases accordingly or fix in v1.8.
- [self] Test-particle reflection statistics depend on WHERE you launch
  relative to the field maxima; a launch design must stratify position
  within each cell or the "reflection" estimand is a launch artefact (v4).
- [self] Check the design space against the device's own design criterion
  before optimising in it (Koch's rho was never reachable in the legacy
  parameterisation - a whole MDO study on the wrong region).
- [self] A null under a non-standard definition is a statement about the
  definition. Before freezing a definition in a preregistration, check it
  against the field's literature (topology v2 looked for wall-side nulls
  that PPM stacks never have; v3.1 with the textbook definition found N-1
  cusps per N stages).
- [self] In a multi-cell global model the sheath rows cannot fix the cell
  potentials (density cancels in the ambipolar balance); the interior
  potential structure must come from a kinetic model or be declared. The
  PIC shows a staircase, not a flat interior.
- [self] A design ranking produced under a declared closure is a property
  of the closure: report at least two closures and their Pareto overlap
  before any "design X is best" sentence (MDO v2: CL-1 vs CL-2 Jaccard 0).
- [self] When two surrogates fail for label noise, skip the surrogate and
  optimise over the measured catalogue directly; a discrete design set with
  measured outputs is a valid design space.
- [self] Before fitting a surrogate, check whether the input->target map
  has step discontinuities (discrete design selectors). A stationary GP on
  raw design parameters fails there; use derived physical features and/or
  per-category models, and always include a tree baseline.
- [self] A single-design result is field-specific until a sweep says
  otherwise: v4 (P2 divergent-exit) had zero reflections; 96 L1a designs
  all reflect. Generalise only from the sweep.
- [self] A residual floor that scales linearly with a parameter under
  continuation, with a well-conditioned Jacobian and |J^T r| ~ floor on a
  bound face, is a MODEL inconsistency, not a solver failure. Derive the
  closed-form reduced row before spending more solver effort.
- [self] The collisionless test-particle wall-hit probability (v4) and the
  Kornfeld per-cusp-transit loss probability are different quantities; the
  legacy chain conflated them. Never feed one into a model expecting the
  other without a declared closure.
- [tool] BoTorch GP fits on CUDA are 20-40x SLOWER than CPU while another
  process saturates the GPU; measure before assuming CUDA helps.
- [self] Low bars are rarely "testing remains": surrogates failed their
  gates for lack of physically-varying data; hybrid is blocked by the
  topology null; external validation needs a self-consistent prediction.
  Unblock by producing the missing DATA (geometry wall-loss dataset).
- [self] When a run fails, check whether an uncommitted fix already exists in the
  main tree (mtimes, `git status`) before writing a new one.
- [self] Committing from many worktrees leaves the main tree diverged and dirty
  with duplicates; reconcile (fetch/rebase/push) after every accepted stream.
- [tool] Reconciling a dirty tree: classify every path by blob hash against
  origin BEFORE stashing; use `git restore --source=stash@{0}^3` selectively
  instead of `git stash pop` (pop aborts wholesale on "already exists").
- [tool] Large fem_reference JSON are Git LFS pointers on origin; compare
  local sha256 against the pointer oid, not the pointer text.
- [tool] To compare a working-tree file with origin, use `git hash-object` vs
  `git rev-parse origin/branch:path`, or `git diff origin/branch -- path`.
  Piping `git show` through PowerShell `Set-Content`/`Out-File` corrupts
  line endings and produces fake whole-file diffs.
- [self] A numerical gate that most healthy orbits fail is a code defect
  until proven otherwise (chord-interpolated event velocity vs 1e-10 energy
  gate). Fix the observable; do not loosen the gate.
- [self] Warp CUDA is only worth it for batched kernels; a per-particle,
  host-driven event loop is 18x slower on GPU than numpy.
- [tool] System `core.autocrlf=true` silently rewrote 760 hash-bound files to
  CRLF on checkout. Root `.gitattributes` (`* text=auto eol=lf`) now pins LF
  (commit fab0eccc). Any worktree created BEFORE fab0eccc must be re-smudged
  (delete + `git checkout --` the `w/crlf` files) before computing code or
  artifact hashes. `git checkout -- path` alone does NOT re-smudge; the file
  must be removed first (stat cache).
- [tool] The Write tool emits CRLF on Windows; after writing a tracked text
  file, re-checkout it so the working copy is LF.
- [self] Tests that compare a fixed fixture timestamp to the wall clock are
  time bombs (tests/coupling failed 24 h after the fixture was written).
  Fixed in `4661a7be`. Guardrail: any coupling builder/verifier call in tests
  must pass `reference_time_utc=NOW` (or bind it once via `functools.partial`
  on the module alias); prove it by monkeypatching `datetime.now` to raise.
- [user] Keep the "Open Cft Roadmap Status" canvas
  (`.cursor/projects/.../canvases/open-cft-roadmap-status.canvas.tsx`) up to
  date at every milestone (each agent completion, each push, each campaign
  state change). Read the canvas skill before editing it.
- [user] Inspect actual environment, process, package, disk, and network state before retrying failed provisioning.
- [user] Keep ML provisioning isolated in `.venv-sota`; never install globally.
- [user] Do not claim GPU support unless a real CUDA tensor operation succeeds.
- [tool] Anything whose bytes are hashed must be written as bytes or with
  `newline="\n"`. `Path.write_text` without it emits CRLF on Windows; Git
  (`eol=lf`) stores LF, so recorded byte hashes become unreproducible from a
  checkout while every content hash stays valid. Readers using universal
  newlines hide this; only byte-exact manifests see it. Guard with a
  fail-closed AST lint, not a one-off grep (orbit_mc v1.7).
- [self] A preregistered experiment's tests that bind the LIVE worktree to
  the frozen code contract are right only until the one execution. After the
  terminal bundle exists, switch them to the RECORDED contract (bundle +
  preregistration-commit blobs) and assert the gate stays closed; otherwise
  the first legitimate package change after the campaign turns the
  experiment's own tests red for reasons unrelated to its evidence.
- [tool] `validate_bundle` binds `root_identity` (volume/file id/final path)
  to the producing directory; it can never pass on another checkout. The
  reviewer-reproducible check is the manifest inventory (byte hashes +
  sidecar pairs), which is what `audit_sidecar_eol.py` does.
- [tool] PowerShell: `git commit -F -` with a here-string does not read
  stdin; write the message to `$env:TEMP` and `-F <file>`. .NET
  `[IO.File]::ReadAllBytes` ignores the PowerShell `cd`; use Python for
  byte checks. `Out-File -Encoding utf8` adds a BOM.
- [tool] Same-basename test modules (`test_specs.py`, `test_audit_hardening.py`)
  clash when several `tests/<pkg>` dirs run in one pytest invocation; run
  the directories separately.
- [self] PIC-2D: Jacobi-PCG is the wrong default for masked cylindrical
  Poisson at these sizes; an exact block-Thomas column factorisation is faster,
  deterministic and shared by CPU and GPU backends. Never restart CG per chunk.
- [tool] Warp: RNG state is by-value inside `@wp.func`; strided reductions,
  one host read per step; `array_sum` has no int32.
- [tool] `.gitignore` negations for tracked result files must follow the
  generic `results/` and `*.npz` rules.
- [self] MDO v1 probe (2026-09-03 08:20): the corrected Kornfeld 4-cell solver
  (`cft_revival.plasma`) closes ONLY for p = (0,0,0,0); every nonzero cusp
  probability tried (Ua 150-1000 V, Ia 0.1-3 A) hit `iteration_limit` with
  residual floors 5e-4..0.19, at ~3 s per multistart solve (pure Python).
  It is unusable as an optimiser evaluation chain; cusp probabilities must
  enter the L0 model through an explicitly declared closure instead.
- [self] Root cause of the above (2026-09-03 11:30, `266d8a99`): on the
  R00-R26 manifold the global row is exactly
  `2*(je3(1-p4)+I4)*(phi4-Ua) + EI*(p1 je0 + p2 je1 + p3 je2)` -> no
  admissible root for any interior p > 0 (source double-books cusp
  recombination energy; anode electron term has an ion's sign). Solutions
  exist only for p1=p2=p3=0 (any p4) at phi4 = Ua. Correction is
  PROPOSED_NOT_ACCEPTED (it frees all four potentials). Before blaming a
  solver floor, substitute the consistent rows into the suspect row.
- [self] `sorted()` is not a projection onto the ordered cone; it permutes
  variable identities and stalled the LM (3/16 zero-cusp cases at 1000 V).
  Use pool-adjacent-violators (`project_nondecreasing`).
- [self] A finding never executed under a protocol needs its own paper gate
  kind (`analytic-consistency`: a derivation verified numerically, pinned by
  tests, recomputed by the checker); do not squeeze it into
  `numerical-campaign`/`numerical-screening`, whose bundle rules would then
  mean nothing. Define "accepted" as "derivation + verification admitted as
  recorded" and list what it does NOT accept.
- [self] Least-squares stall floors move with start count / iteration budget
  (5%-20%); when a paper checker recomputes them, declare the reduced
  protocol, a tolerance (25%) and a recording precision (3 sig. digits), and
  cache the recomputation per process (~35 s once, not per test).
- [self] Verify a brief's numbers against the files before writing claims:
  the legacy defect is "lsqnonlin flags 1-3 accepted by status, flag 4
  rejected", not "flag 4 accepted"; the `+IE` cusp terms are on line 136 of
  `Power_B_EQs.m` (the doc says 137 = the anode term).
- [tool] `sci`-formatted macros carry `$...$`; inside a caption's math they
  give "Missing $ inserted" two files away. `\allowbreak` in `\texttt` gives
  breaks but no stretch; use `\hspace{0pt plus 1.5pt}` after operators.
- [tool] Per-file `git rev-parse`/`show` is ~170 ms here; `git ls-tree -r -z`
  per commit + one `git cat-file --batch` binds 14 files in ~1 s.

- [tool] (owlgs v2, 2026-09-03 20:40) orbit_mc v1.7 `_validate_probability` (artifact
  sealing) and `coupling_v42_handoff` require `lower <= p <= upper` verbatim, but
  `wilson_interval(0, n).lower` is a POSITIVE round-off (~1e-17) for 734 of the first
  4000 n (3, 6, 7, 12, 14, 24, ..., 384, 768, 1536, ...) and `wilson_interval(n, n).upper`
  is 1 - ulp for 1238 of them (512, 640, 1152, ...). Any case whose zero-count category
  (timeouts are ALWAYS zero) lands on such an n raises at sealing and kills the one-shot
  run; v1 survived only because 512 is k=0-safe. Frozen package -> choose case sizes from
  the safe set (128, 64, 16 are safe for both ends); never let a case size be a free
  function of the design (3 cells x 128 = 384 would have died on 30/96 designs).
- [tool] orbit_mc v1.7 validates every launch id against
  `<campaign_id>:E[0-9]+:P[0-9]+:X[0-9]+:D[+-]1:G[0-9]+`; custom launch sets must encode
  their coordinates into that grammar (X = cell index, G = Sobol index here).
- [tool] canonical JSON integers are signed 64-bit; uint64 seeds (sha256[:8]) must be
  emitted as decimal strings (`CanonicalizationError: integers must fit signed 64-bit`).

## Session Entries

### 2026-09-03 23:15 AEST - orbit wall-loss geometry screening v2 (catalogue cells, scrambled Sobol, two-stage allocation)

#### Task Summary

- Worktree `uni-project-orbit-geo2`, branch `exp/orbit-wall-loss-geometry-screening-v2` from
  `9abbd537` (rebased onto `beb4772c` before the prereg push). Commits: `a7a884bf` code/tests,
  `cef1ee59` preregistration, `26029b72` result (results only), `bb756418` runtime fix +
  recovery + POSTHOC disclosure, `eef7ac82` dashboard, `066234d9` no-ff merge of upstream;
  `feat/sota-foundation` ff `1fb8561d..066234d9` (is-ancestor held via the merge; no force).
- One execution (86 min; 12 CPU workers): 97 designs (96 sweep + P2 row), 377 catalogue cells,
  1105 cases, 104,832 orbits; `accepted_screening_dataset`; 117 cells topped up (31 %), 260
  saturated; 294/377 cells surrogate-v3 ready; N->2N bias -0.9e-4 (2 discordant of 11,648);
  interior cells 181/181 at P(wall)=1; exit-side median 0.50; v1-vs-v2 pooled Spearman 0.15.
- Runtime published the manifest post hoc after EMFILE at publication (16,957 files); disclosed.

#### Mistakes And Fixes

- [self] Wrote the first launch ids in my own grammar; orbit_mc v1.7 enforces
  `E..:P..:X..:D..:G..`. Read the artifact validator's regexes before inventing ids.
- [self] Sized cases as design x stage (384/512/640 launches); orbit_mc's Wilson ordering check
  would have killed 30/96 designs at n = 384. Per-cell 128-launch blocks fixed it; the safe set
  is pinned by tests and by `require_safe_case_sizes`.
- [self] Did not budget the bundle FILE COUNT: 1105 cases x 8 files + 1889 access records x 2 =
  16,957 > the 8192-descriptor cap of the runtime's pinning step. The shakedown (578 files)
  could not show it. New guardrail: estimate files = cases x 8 + labels x 2 before prereg.
- [self] The planning top-up fraction (0.72 from v1's fixed cells) was 2.3x the realised 0.31;
  the timing projection was 1.7x too high. Catalogue cells saturate at p = 1 far more often.
- [tool] `Get-ChildItem`/`Get-Content` output vanished after `Start-Process`; `cmd /c dir`
  and Python probes worked (again).

#### What Worked

- Reading v1/v4/v3.1 code first and reusing by import (designs, consumer, ValidatorLedger,
  run_case_integration/export, catalogue loader) kept the new code to cells/sobol/experiment.
- Dependency-free scrambled Sobol (Joe-Kuo + LMS + digital shift) with the unscrambled reference
  points and (0,m,2)-net checks as tests; stage 2 = indices 16..63 of the same sequences.
- Four shakedowns before the freeze caught three real defects (uint64 seeds, id grammar, Wilson
  ordering at n = 6) that no unit test reached; the evidentiary run then passed every gate.
- Treating the post-terminal EMFILE as a runtime defect: fail-closed recovery through the
  runtime's own `_inventory` + `validate_bundle(manifest_override)`, tested against a simulated
  EMFILE, applied once, disclosed in POSTHOC_FINALIZATION.md, results commit kept results-only.

#### Guardrails For Next Session

- Interior catalogue cells are all saturated at P(wall) = 1 under the mid-plane / 0.65-0.825 r_w
  launch design; surrogate v3 must model the partial cells (anode-side median 0.98, exit-side
  0.50 with a direction/pitch structure) and treat interior cells as constants.
- Any paper admission of v2 must cite `POSTHOC_FINALIZATION.md`; never call the manifest
  "published inside the locked attempt".
- Keep the detached run worktree `uni-project-orbit-geo2-run` (validate_bundle root identity)
  and the Git-common lock `orbit-wall-loss-geometry-screening-v2.execution.lock`.
- Main tree `uni-project` is at `9abbd537`; needs `git pull --ff-only`.

### 2026-09-03 21:35 AEST - admit cusp topology search v3.1 to the paper (fifth screening gate)

#### Task Summary

- Worktree `uni-project-paper-topo31` (`paper/topology-v31-claim` from `9abbd537`): gate
  `GATE-CUSP-TOPOLOGY-V3-1` at NEW outcome `accepted-topology-screening`, manifest type
  `paper-separatrix-topology-screening-manifest` (52 source + 65 lineage + 3 reference
  files + definition source), generator with 430 `Ctv` macros and 4 tables that re-derives
  every estimand from the rows and reproduces the v3 audit from sealed data, claims
  CLM-061..068, CLM-028/044 amended (trigger A), Section 13, 32 tests; commit `726c8a69`
  -> rebased twice (disjoint concurrent pushes) -> `13d8ac6a`; `feat/sota-foundation` ff
  `44d0c63c..13d8ac6a`, no force. paper/tests 197 OK; PDF 49 pages `34e11c8e`, byte-identical.

#### Mistakes And Fixes

- [self] The brief's "every vector null sits on the axis: no wall-side X-type null exists"
  is contradicted by the sealed v1 dataset (200 in-channel X roots = 180 axis clusters +
  20 off-axis bilinear roots in 14 wide-bore cases at r/r_w 0.16-0.54, all excluded for
  `no_cell_bounding_separatrix`). Wrote the split as derived macros and "none at the wall";
  the "by construction" statement rests on the definition (a wall cusp is not a null) and
  the sealed v2 strength ratios (16-42 % -> one axis sign change), not on the slogan.
- [self] Recomputed `distance_to_nearest_stage_gap_m` with interior midpoints only; the
  sealed value includes the half-pitch end gaps (caught by the one N+1 design). Read the
  failing design, then fix the rule; never add a tolerance.
- [self] First checker issued ~4 git calls per lineage file (65 files, 14 s per call, 10 min
  of admission tests); `git ls-tree -r -z <rev> -- paths` + one `cat-file --batch` per
  revision and a per-revision `_resolves_to_commit` cache cut it to 6 s.
- [self] Overfull boxes from `\texttt{}` identifier lists (nine gate names) and a long
  review path: render identifier lists as prose, one `\texttt{}` per sentence end.
- [tool] `Measure-Command { python ... }` swallows the script's stdout; time inside Python.
- [tool] Origin moved twice between rebase and push (other agents); fetch + is-ancestor
  immediately before the ff push, in one shell command.

#### What Worked

- Same flow (generator -> lint with checker functions -> manifest from raw macro values ->
  claim bodies extracted from the section -> trial pdflatex in %TEMP% -> check_paper
  (only "not committed at HEAD") -> commit -> recheck/tests/build -> devlog amend -> rebase
  -> push -> ff).
- A rejected campaign admitted as lineage: whole bundle byte-verified, audit reproduced in
  the generator from sealed inputs, `lineage.cited_for_numbers: false` enforced by the
  checker, histogram-equality boolean as the recomputed proof that only the recording
  layer changed.
- Not reusing the frozen-definition policy flag with a flipped meaning; explicit new flags
  (`confinement_cells_demonstrated: false`, `frozen_definition_nulls_remain_true: true`).
- Version tokens (`v3.1`, `v2`, `L1a`) derived from experiment ids / set keys as macros so
  the section still types no digit.

#### Guardrails For Next Session

- Any consumer of the cusp-cell catalogue (screening v2 launch design, MDO closures) must
  carry the label and be admitted separately; the paper now says "no admitted consumer".
- Section 8's records are untouched; their `stable_multicell_wall_cusp_topology_demonstrated:
  false` stays. Do not edit CLM-018..027.
- The P2 row's kinetic-plane agreement is a reported consistency reference, not admitted
  PIC evidence; keep it phrased as development context until a PIC gate exists.
- Main tree `uni-project` is behind; needs `git pull --ff-only`. Canvas not updated here.

### 2026-09-03 20:35 AEST - literature review: TWT/PPM physics inherited by HEMP (docs/literature-twt-ppm)

#### Task Summary

- Worktree `uni-project-lit-twt`, branch `docs/literature-twt-ppm` from `fb5408bf`;
  `modern/docs/literature/twt-ppm-physics-for-hemp.md` (983 lines, 51 verified refs:
  6 [T] full text, 14 [R] record+abstract, 24 [C] record, 5 [B] books, 2 [P] patents) plus
  the read-only `scripts/ppm_axis_field_check.py` and its committed output JSON. Commit
  `beb4772c` (rebased once over the concurrent topology v3.1 landing `9abbd537`); pushed;
  `feat/sota-foundation` fast-forwarded `9abbd537..beb4772c` (is-ancestor held, no force).
- Headline physics: every recorded field (four L1a representatives and the iron P2) is a
  single-harmonic PPM field at the wall (wall b3/b1 0.2-0.5 %, FWHM 0.96-1.16 x 2L/3);
  wall cusp field / axis peak = 0.45-0.61 = I_1(pi r_w/L); the cusp is the |B| MINIMUM of
  its cell at every radius, so Koch's HEMP design ratio rho (1.5-3.1 DM9-1, 4-10.6 DM10,
  IEPC-2007-110 Table 1) is unreachable in the catalogue (max 1.03 over 96 designs); the
  recorded reflections are mirror reflections toward the magnet-centre maxima (median
  B_turn/B_launch 1.08-1.14 = 1/sin^2 70 deg), and v4's zero reflections follow from
  launching 0.5 mm from the magnet centres.

#### Mistakes And Fixes

- [self] Extended axis-fitted harmonics (b3, b5 at the 0.45 mm aliasing floor) to the wall
  with I_1(k kappa r): 1.7-2.1x over-prediction because I_1(5 x_w) amplifies 50-80x. Fit
  harmonic content on the WALL profile; use the fundamental alone for wall estimates.
- [self] Field-line bisection assumed psi increasing in r; for designs with negative b1
  it silently returned garbage (max-along-line < launch). Carry the sign of B_z.
- [self] Sampled the wall-field angle at the nearest grid node to z_c (0.22 mm off) and
  read 8-21 deg; linear interpolation at z_c gives 0.02-3.9 deg. Interpolate before
  computing a ratio whose numerator crosses zero.
- [self] Two DOIs typed from memory were wrong (Mendel 1954, Sterrett-Heffner 1958);
  the bibliographic Crossref query found the right ones. Never type a DOI.
- [tool] PowerShell `>` redirect of Python stdout writes UTF-16; read it back with
  `encoding='utf-16'` or write the report from inside Python.
- [tool] Slipped a heredoc into a Shell call again (parse error, harmless).
- [tool] Crossref rate-limits (429) after ~20 rapid queries; 1.2 s sleep and batching by
  DOI record fetch instead of bibliographic search avoided it.

#### What Worked

- Reading the sweep QoI extractor, the v3 catalogue and the screening endpoint tables
  BEFORE the theory: the v3 `axis_mirror_ratio` 0.47-0.69 and `wall_mirror_ratio` 1.00
  were already the PPM prediction I_1(x_w), I_1/I_0 < 1; the script then confirmed it.
- Deriving the Bessel extension and the harmonic attenuation factor a_k made the
  iron-vs-no-iron question answerable without any literature number: at r_w/L < 0.5 the
  wall cannot see pole-bore harmonics regardless of iron.
- Per-cell reflection counts against distance-to-magnet-centre (Table G2) turned the
  "v4 zero vs L1a 11-55 %" discrepancy into a one-line launch-position explanation.
- `pdftotext -layout` on IEPC-2007-108/-110/-236 gave verbatim heritage quotes and Koch's
  Table 1 rho values; Google Patents pages gave the patent metadata in one fetch each.
- Verifying that the doc's 99 table rows equal the script's stdout byte for byte, and
  that the v3 and v3.1 catalogues give identical rows, before committing.

#### Guardrails For Next Session

- Screening v2 must launch by cell position relative to the cusps AND stratify by
  direction toward/away from the nearer magnet centre; record |B| and field-direction
  rotation at the turning point to separate mirror reflections from v_par = 0 crossings.
- Any "cusp mirror" statement about our fields is false for this catalogue; the bottles
  are centred on the cusps and bounded by the magnet centres. Report rho per cusp in
  topology consumers and add r_w/L as a design variable before claiming HEMP-likeness.
- Cite the topology catalogue from v3.1 (`cec47f12`, accepted), never from v3.
- Main tree `uni-project` is behind; needs `git pull --ff-only`.

### 2026-09-03 20:05 AEST - cusp topology search v3 / v3.1 (literature cusp/cell definition)

#### Task Summary

- Worktree `uni-project-topo-v3`, branch `exp/cusp-topology-search-v3` from `66879e00`.
  v3: code/tests `bce595dc`, prereg `69159934`, result `8cbcdbe6` (RECORDED
  `assessment_rejection`: every numerical gate true for 281 designs, the held-out
  gate false for 14 v1 designs through a reference-extraction defect), audit
  `9fa6359a`. v3.1 (corrected re-preregistration, new dir
  `cusp_topology_search_v3_1`): code `ca811d11`, merge of upstream `988220f3`,
  prereg `1600cfd3`, result `cec47f12` (`accepted_result`, 281/281 stable, all gates
  true), dashboard `9abbd537`. `feat/sota-foundation` ff `fb5408bf..9abbd537`
  (is-ancestor held via a no-ff merge; no rebase after the first prereg push; no force).
- Headline (v3.1): wall cusps 0:6 / 1:140 / 2:36 / 3:56 / 4:25 / 5:6 / 6:6 / 7:6 over
  281 designs; sweep-v2 N stages -> N-1 cusps (25/40/18 of 26/45/25), four-wall-cusp
  fraction 19/96, cusps at the inter-magnet gaps (median 0.14 mm, max 0.26 mm);
  four-cell-v2: 128/128 exactly ONE wall cusp (weak even stages -> one axis null);
  P2: 6.028 / 12.000 / 17.972 mm (axis nulls within 31 um of the PIC planes).

#### Mistakes And Fixes

- [self] Extracted the sealed v1 axis roots with `r_m == 0.0`; v1 clusters an axis
  sign-change with a neighbouring bilinear Newton root and reports the centroid, so 26
  of 206 clusters (r <= 1.6e-8 m) were dropped and the one-shot v3 execution was
  REJECTED on its held-out gate. Extract a reference by the source's own semantics
  (member detection method), never by a float identity on a derived quantity.
- [self] The v3 shakedown's v1 cases (s02, s08-p0-r0-neg) had only single-member axis
  clusters, so the defect stayed invisible until the freeze. Put the reference's edge
  case (multi-member clusters, s05) into the shakedown; v3.1 did.
- [self] Shakedown 1 caught two more pre-freeze defects: the v2 dataset's
  `artifact_sha256` is the artifact-BYTES hash (not `integrity.payload_sha256`), and
  `chamber.exit_start_m = 0.024 - 0.006 = 0.018000000000000002` vs the v4 authority
  0.018 (compare with a tolerance).
- [tool] The shell's `PYTHONPATH` kept pointing at the detached run worktree after the
  execute step; the next command imported `cft_revival` from the wrong tree and the
  source binding refused. Reset `PYTHONPATH` whenever the working directory changes.
- [tool] Edge `--dump-dom` under PowerShell redirection wrote 0 bytes while
  `--screenshot` worked; the rendered cards in the screenshot are the JS proof.

#### What Worked

- Literature-driven redefinition before code: in the axis-regular flux variable
  `g = (psi - psi_axis)/r^2` the separatrix is the `g = 0` contour and the wall cusp is
  the root of `psi(r_w, z) - psi_axis`; every simple axis null is X-type
  (`J = diag(-g_z, 2 g_z)`). Trace and flux root agreed to < 1e-9 m on 588 cusps.
- Reusing the screening's identity-proven CPU re-solve (96/96), rebuilding the v2 (128)
  and v1 (56) designs against their sealed hashes, and loading the P2 field through the
  v4 adapter; all 281 stable under 2x refinement (max shift 33 um vs 0.25 mm).
- Evaluating refinement stability on the wall INTERSECTIONS, not their inside/outside
  classification (the P2 exit cusp sits 28 um inside the 18 mm straight end), and
  reusing the accepted map's axis window for the refined map.
- A recorded rejection handled by the book: immutable bundle, read-only audit script +
  tests binding the recorded outcome, corrected campaign as a NEW directory with the
  lineage disclosed and the v3 numbers never cited.

#### Guardrails For Next Session

- Cite v3.1 (`cusp_topology_search_v3_1`, `cec47f12`), never the v3 bundle, for any
  cusp/cell number; `catalogue.load_catalogue` refuses the v3 bundle by design.
- The paper's Section 8 nulls stay true under their frozen definition; the Discussion
  must now say the definition was non-standard and cite v3.1 (a paper admission with a
  `numerical-screening` gate is the next step; not done here).
- Mirror descriptors (wall ratio ~1.00-1.02 for interior cells; axis ratio 0.2-174) are
  field ratios, never probabilities; consumers must carry the label.
- Detached run worktrees `uni-project-topo-v3-run` and `uni-project-topo-v31-run` and the
  Git-common locks `cusp-topology-search-v3{,.1}.execution.lock` stay by design.
- Main tree `uni-project` is behind; needs `git pull --ff-only`.

### 2026-09-03 19:40 AEST - plasma v2: sheath-closed four-cell power balance (feat/plasma-network-v2-sheath)

#### Task Summary

- Worktree `uni-project-plasma-v2` from `b6bb6215`; new package
  `cft_revival.plasma_v2` (v1 imported read-only, its 5 files still match the
  paper manifest); rows R28-R37 (floating-dielectric sheath, anode sheath,
  declared potentials, cusp-loss closures); rank 21 -> 28 -> 31 documented;
  verification record (192-solve grid, Kornfeld/Puca reproduction, CL-4
  prefactor sweep, PIC v2 context); spec + formulation + devlog + learning;
  53 tests; commit `e75151ce` -> rebased twice over concurrent pushes
  (`8674cc5a` paper MDO v2 admission, `112bb250` pic2d v1.4; disjoint paths)
  to `fb5408bf`; feature branch pushed at pre-rebase `ea798971` (never force);
  `feat/sota-foundation` fast-forwarded `112bb250..fb5408bf`; `check_paper.py`
  green before commit and after each rebase.

#### Mistakes And Fixes

- [self] First bracket scan reset on undefined points and missed the anode-row
  root that sits in a 0.07 V sliver at the edge of the admissible cascade
  (`no_bracket` while the LM found it). Treat "undefined" as a signed limit
  (+inf here) so the edge itself brackets the root.
- [self] Rank read 22 instead of 21 at a converged LM state 1e-12 off the
  manifold; the identity row's gradient is dependent only ON the manifold.
  Evaluate structural rank at the exact manifold projection.
- [self] Assumed mode B (declared cathode coupling, solved anode fall) would
  close at Kornfeld's 14.1 V; the exit-drop feedback puts the only root at a
  57 V fall (> T4) -> inadmissible. Probe the admissible band before writing
  the test expectation (it is `V_c <= 14.074 V`).
- [self] CL-4 with PIC segment-mean densities saturates p_k (leak current
  10-100x the discharge current); the first test used 1e17 m^-3 and the
  default SHEATH anode row on a dead cascade -> exception. Use mode C
  potentials and a 1e15 density for the fixed-point test.
- [tool] `cursor-ide-browser` unusable from this subagent (tabs vanish);
  Springer table pages are a Cloudflare client challenge for urllib/WebFetch;
  headless Chrome `--dump-dom --virtual-time-budget=20000` returned the table.
- [tool] PowerShell inline `python -c` with `r'$env:TEMP'` produced a unicode
  escape error; use `os.environ['TEMP']` inside Python.

#### What Worked

- Deriving the cusp energy criterion `c_s <= (1 + I_k/Je_k)/CT` (step-size
  independent) BEFORE the grid: the grid then confirmed 0/96 (no emission) vs
  89/96 (SCL) instead of surprising me.
- Manifold seed (v1 `potential_parametrized_state` + 1-D bisection on the
  anode row) feeding the v1 LM: every admissible case closed to <= 1.3e-12 at
  rank 31 in 0.2-2 s.
- Reproduction as tables row by row: Kornfeld's printed cusp loss equals the
  NO-+EI convention (22.96 vs 22.9 W), his excitation matches neither and is
  identical across columns, Puca's GA columns violate R00/R15 at 0.5 once
  j_e0 is reconstructed from their own R01.
- `pdftotext -layout` on IEPC-2007-108 gave Table 3.1 verbatim (DM10 column
  too); Puca Table 1 via headless Chrome; both transcribed with notes.

#### Guardrails For Next Session

- Do not accept the R27 corrections on the strength of v2: the potentials are
  still DECLARED (flat interior), the PIC plateau shows a 94/55/125 V
  staircase, and the sheath temperature is the cell temperature while the PIC
  near-wall population is 6 eV under a 40 eV axis.
- `A_k` stays a collisionless access fraction; CL-3 turns it into
  `0.36 A_k` (SCL) or `0.0051 A_k` (no emission) - never call either a cusp
  probability.
- Any re-admission needs a new analytic-consistency manifest binding
  `plasma_v2`, the record and the formulation; the v1 gate must stay bound to
  the untouched v1 files.
- Main tree `uni-project` is behind; needs `git pull --ff-only`.


### 2026-09-03 18:35 AEST - literature synthesis + revised roadmap (docs/literature-synthesis) and canvas update

#### Task Summary

- Worktree `uni-project-lit-synth`, branch `docs/literature-synthesis` from `b6bb6215`; one
  file `modern/docs/LITERATURE_SYNTHESIS.md` (569 lines, ASCII, LF). Commit `11a10873`
  pushed on the branch; origin had moved to `a3793c27` (MDO v2 paper admission) so it was
  rebased to `8674cc5a` and `feat/sota-foundation` fast-forwarded `a3793c27..8674cc5a`
  (is-ancestor held after rebase, no force). Every summary-table row of the three reviews
  decided: 60 atomic items after de-duplication (S7 = R2c), 53 ADOPT / 6 DEFER / 1 REJECT,
  verified by a regex over the tables (61 rows, 61 unique ids).
- Canvas: new tab `Literature roadmap`; seven ladder rows (35 total); MDO v2 row -> In the
  paper; header / merge truth / actions / evidence at 18:10 AEST; stage counts recomputed
  in Node (stage 7: 13 (10), 5: 6 (4), 4: 8 (3), 3: 1, 1: 7 (1); RUNNING 4); TS check clean.

#### Mistakes And Fixes

- [self] Pushed the branch before re-fetching: origin had advanced by one commit in the
  ~10 min between worktree creation and push. Fetch immediately before the is-ancestor
  check, then rebase; the pushed branch stays at the pre-rebase SHA (never force).
- [tool] The canvas was edited by another agent while I was editing it (an action I was
  about to add already existed at 18:27). `StrReplace` failed safely on the moved anchor;
  re-read the exact lines (`Get-Content`[n]) immediately before every canvas edit and keep
  each edit to one anchor. Verify after each batch that earlier edits survived (`grep` for
  the new ids).
- [self] Counted decisions by hand first (53/7/1) and only then found the S7 = R2c
  duplicate; count with a script over the final tables, never by hand.

#### What Worked

- Reading the three reviews' summary tables first and splitting bundled rows into atomic
  items with ids (P1a.., R1a.., S1a..) made the counts checkable and the roadmap steps
  citable from the canvas (`synthesis §5.x`).
- Recomputing the stage summary in Node by evaluating the `ladderRows` literal with
  `ok/cav/no` stubs (`new Function`) instead of counting by eye; it also caught that MDO v2
  moved from stage 6 to 7 and that stage 6 is now empty.
- Representing a RUNNING stream with zero commits as caveat cells that cite the launch
  brief keeps the ladder honest and still shows the user-requested rung.

#### Guardrails For Next Session

- The bare label `CL-3` is overloaded (potential closure for the network vs sheath-limited
  cusp-loss closure for MDO); use `CL-3-potentials`, `CL-3-sheath-limited`,
  `CL-4-hybrid-area` in every new protocol.
- Every quotation of the PIC plateau's 46 % utilisation must say "gross, no wall
  recycling" (net ~17-18 %) until steady-state v3 records.
- The wrong "Ma 2024 AST" / "TU Berlin" attributions lived only in briefs; keep it that
  way - resolve DOIs through Crossref before citing.
- Main tree `uni-project` is behind origin; needs `git pull --ff-only`.

### 2026-09-03 18:05 AEST - admit MDO L0 campaign v2 (screened catalogue) to the paper

#### Task Summary

- Worktree `uni-project-paper-mdo2` (`paper/mdo-l0-v2-claim` from `0ea33a7e`):
  third `numerical-campaign` gate `GATE-MDO-L0-V2` with its own manifest type
  `paper-mdo-catalogue-campaign-manifest` (53 blob+sha files at `a003f766`:
  bundle, frozen files = `99914dc2` blobs, screening dataset/manifest =
  `ab7c2897` blobs, v1 bundle files, v1 `POSTHOC_AUDIT.md` = `e9f9af16` blob,
  dashboard at `0ea33a7e`), generator that verifies BOTH optimisation bundles
  byte for byte and recomputes every catalogue probability / Jeffreys mean /
  Wilson bound / survival, 611 `Mdb` macros, 4 tables, claims CLM-053..060,
  CLM-035 + CLM-052 amended (geometry link no longer "open"/"future work"),
  Section 12, shared `_check_mdo_family` checker, 26 tests; commit `a3793c27`
  (rebased over the concurrent literature/pic2d commits, no overlapping
  paths). paper/tests 165 OK; PDF 41 pages `86796210`, byte-identical
  twice. Pushed; origin/feat/sota-foundation fast-forwarded b6bb6215..a3793c27 (is-ancestor held after the rebase, no force).

#### Mistakes And Fixes

- [self] The brief's "91 infeasible, all BO boundary probes" is 88 in BO +
  3 in NSGA-III (0/2/1); the claim states the per-optimiser split. Verify
  aggregate wording against per-run records before writing it.
- [self] "Seed 101 never found design 49" needed the initial design checked:
  49 was absent from seed 101's 32-point shared initial design and present in
  202/303's; recorded as derived macros so the stall reads as an exploration
  effect of the categorical index.
- [tool] A `%TEMP%\inspect.py` probe shadowed the stdlib `inspect` module
  (sys.path[0] = script dir) and broke `dataclasses` inside `check_paper`;
  name scratch probes distinctively.
- [self] Macro names must be alphabetic: `MdbAuditF9Closure` failed; use
  word tokens (`Nine`, `Ten`, `TwentyTwo`). Likewise "CL-1"/"CL-2" in a
  macro-only section come from the protocol's closure keys as macros.
- [self] A text macro copied from the protocol ("energy, 4 objectives, 32
  points, seed 1") tripped the unregistered-quantitative heuristic in the
  generated file; bind numbers, not sentences.
- [self] The HV table overflowed 15 pt at `\scriptsize` (standalone compile
  caught it): narrow the Pareto-designs `p{}` column, `\tabcolsep` 2.5 pt.

#### What Worked

- Same flow as the earlier admissions (generator -> lint with the checker's
  functions -> manifest from raw macro values -> claim bodies extracted from
  the section -> trial pdflatex into %TEMP% -> check_paper -> commit ->
  recheck/tests/build -> devlog amend -> rebase -> push -> ff).
- Factoring the v1 checker into `_check_mdo_family(spec)` with the old
  signature kept as a wrapper: v1 tests unchanged, v2 adds only its extra
  bindings (prior bundle, dataset blob, audit disclosure list).
- Reading the v1 bundle through the same `Bundle` class and pinning its
  manifest SHA/results commit made the v1-vs-v2 table artifact-bound with
  `bundle: v1` marked on every macro; a fail-closed `MdbSameReferenceFrame`
  boolean proves comparability.
- `git diff-tree --no-commit-id --name-only -r` as a derived macro makes the
  "results-only commit" audit closure (F9) recomputed at every check.

#### Guardrails For Next Session

- Stale boundary sentences now live in three admissions' claims; before any
  admission grep claims.json AND manuscript.tex for "open", "future work",
  "not admitted", "no consumer" and update the earlier admission's test that
  asserted the old wording (geometry test asserted "future work").
- The v2 section says "wins under this closure only"; never promote the
  robust-front designs 49/50/94 to a recommendation. The CL-2 front shares
  no design with CL-1.
- Any change to `mdo_l0_campaign_v1/results`, `mdo_l0_campaign_v1/POSTHOC_AUDIT.md`
  or the screening dataset now fails the v2 checker (by design).
- Main tree `uni-project` is behind; needs `git pull --ff-only`.

### 2026-09-03 18:30 AEST - literature review: surrogate / MDO / validation blockers (docs/literature-mdo)

#### Task Summary

- Worktree `uni-project-lit-mdo`, branch `docs/literature-mdo` from `96220ffc`; one file
  `modern/docs/literature/surrogate-mdo-validation-blockers.md` (823 lines, 157 refs: 124
  Crossref-resolved DOIs, 18 arXiv, 15 conference/standard URLs). Commit `af98b3dd` pushed
  on the feature branch; rebased over the two concurrent literature commits
  (`66879e00`, `ccb22d5d`, disjoint filenames) to `b6bb6215`; `feat/sota-foundation`
  fast-forwarded `ccb22d5d..b6bb6215` (is-ancestor held after rebase, no force).

#### Mistakes And Fixes

- [self] The task brief's "Ma 2024 AST" DOI (10.1016/j.ast.2024.109516) resolves to Yeo,
  Gadisa, Ogawa, Bang 2024 on FEEP, not a cusped-field paper by Ma. Resolve every DOI
  BEFORE writing the sentence that cites it; four DOIs I "knew" (Marks-Jorns, Duras 2017,
  Ma 2015 Vacuum, Liu 2015 plume) were wrong or 404 and had to be searched.
- [tool] Crossref stdout on Windows cp1252 crashed on a modifier-letter apostrophe;
  `sys.stdout.reconfigure(encoding="utf-8")` first line of every check script.
- [tool] PowerShell redirect of Python UTF-8 output produced mojibake in the saved file;
  generate the final bibliography from the JSON records inside Python, never from a
  redirected text file.
- [tool] Slipped a `<<'EOF'` heredoc into a Shell call again (PowerShell parse error);
  patch scripts go to %TEMP% and run as files.

#### What Worked

- Batch verification: one Crossref script over DOI lists, one arXiv API call over IDs,
  HTTP GETs on electricrocket.org PDFs; the bibliography was then GENERATED from the
  verified records (title/authors/venue/year/volume/pages), so no typed citation drifted.
- A regex cross-check of every "Author (year)" in the prose against the bibliography
  caught two year mismatches (Rasmussen-Williams 2005 per MIT Press record; Chambers-
  Tzavella 2021 per Crossref) before commit.
- Mapping each recommendation to a number the repo already has (binomial floor 0.031-0.037
  at n = 128 -> 0.020 at 512; 73/96 designs saturated in one cell -> replicate only
  non-saturated cells; v1 orbit rate 100,352 in 95 min).

#### Guardrails For Next Session

- `modern/docs/literature/` now has three sibling reviews (PIC-MCC, reduced-model/topology,
  surrogate/MDO/validation) written concurrently; cross-reference them before adding a
  fourth and reconcile any duplicated recommendation wording.
- `origin/docs/literature-mdo` stays at pre-rebase `af98b3dd` by the never-force rule; the
  integrated commit is `b6bb6215` on `feat/sota-foundation`.
- Main tree `uni-project` is behind; needs `git pull --ff-only`.
- No TU Berlin HEMP dataset exists in the literature; do not cite one.


### 2026-09-03 18:10 AEST - PIC-MCC blockers literature review

#### Task Summary

- `modern/docs/literature/pic-mcc-blockers.md` on `docs/literature-pic` (`ccb22d5d` after a
  rebase over the concurrent `66879e00`), fast-forwarded into `feat/sota-foundation`; 116
  verified references; document only.

#### Mistakes And Fixes

- [self] Wrote the bibliography heading as "63 entries" for a 116-entry list and left five
  entries uncited; a scratch regex cross-check (citations vs numbered entries) caught both.
  Run such a check before committing any cited document.
- [self] The user's "downstream paper" DOI resolved to a FEEP paper, not a CFT PIC paper;
  verify a cited DOI's actual title before building on the premise and report the mismatch.
- [tool] PowerShell splits DOIs containing parentheses (`10.1016/0021-9991(69)90058-8`) into
  three arguments; quote every DOI argument.
- [tool] Python stdout is cp1252 in this shell; `sys.stdout.reconfigure(encoding="utf-8")`
  before printing Crossref/OpenAlex metadata (Unicode hyphens killed the first batch).

#### What Worked

- Verification chain that never needed a publisher page: Crossref
  `works?query.bibliographic=...&rows=3&select=DOI,title,author,container-title,issued,volume,page,article-number`
  for discovery, `works/<DOI>` for the record (Wiley/Springer deposit abstracts, IOP/AIP
  usually not), OpenAlex `works/doi:<DOI>` for abstracts (inverted index), Unpaywall for OA
  location, DSpace@MIT handles for theses. Elsevier (406), Wiley and Springer (Cloudflare /
  client challenge) block WebFetch; J-STAGE and arXiv PDFs download with urllib and
  `pdftotext` (MiKTeX) extracts them.
- Batching lookups in a %TEMP% Python script (one call, 10 queries) instead of one WebFetch
  per reference; appending every hit to a jsonl for the bibliography.
- Following the reference lists of the two closest papers (Matthias 2019/2020) surfaced the
  key HEMP items (Kahnfeld 2018 breathing modes, Brandt 2016 downscaled HEMPT, Lacina 1971)
  faster than keyword search.

#### Guardrails For Next Session

- Any utilisation figure quoted against Brandt et al. 2016 must be NET of wall recycling;
  our inventory currently removes wall-absorbed ions from the atom balance (59 % of S).
- Do not cite the Kahnfeld et al. 2019 Rev. Mod. Plasma Phys. review for any specific
  claim: closed access, no abstract deposited, not read.
- The concurrent `modern/docs/literature/reduced-models-cusp-topology-blockers.md` exists;
  cross-link rather than duplicate if a third review is written.

### 2026-09-03 18:40 AEST - literature review: reduced-model, cusp-loss and topology blockers

#### Task Summary

- Worktree `uni-project-lit-models`, branch `docs/literature-models` from
  `96220ffc`; one commit `66879e00` (`modern/docs/literature/
  reduced-models-cusp-topology-blockers.md`, 935 lines, LF, ASCII); pushed;
  `feat/sota-foundation` fast-forwarded `96220ffc..66879e00` (is-ancestor held,
  no rebase, no force). No code, spec or experiment file touched.
- 72 verified references (Crossref DOI / DSpace handle / ERPS PDF) with a
  per-entry verification tag ([T] full text read, [R] record only, [C] IEPC id).

#### Mistakes And Fixes

- [self] The brief's "downstream paper" DOI 10.1016/j.ast.2024.109516 is a FEEP
  optimisation paper (Yeo, Gadisa, Ogawa, Bang 2024), not a cusped-field paper.
  Resolve every supplied DOI through Crossref BEFORE building an argument on it;
  the real lineage is Fahey/Muffatti/Ogawa 2017, Yeo 2020/2022, Puca 2024.
- [self] First Crossref DOI guess for Kalentev 2014 (ctpp.201310047) was a
  different paper; a bibliographic query (`query.bibliographic`) found the right
  one (ctpp.201300038). Never cite a DOI from memory; query, then confirm title.
- [tool] PowerShell splits a DOI containing parentheses
  (`10.1016/0025-5564(70)90132-X`) into three arguments; quote it.
- [tool] Crossref JSON through WebFetch is dumped to a file; a 40-line Python
  helper printing title/authors/venue/year for a DOI list verified 26 DOIs in
  one call. Use it for any future citation work.

#### What Worked

- Grepping the ERPS PDF text of IEPC-2007-108 gave the verbatim assumptions
  (2, 5, 8, 12), the "minfehl" minimum-error solver statement and Table 3.1
  values that settle where the power-balance inconsistency comes from.
- Reading Puca et al. 2024 (the only published independent re-solve of the
  Kornfeld system) showed the same phi_4 = Ua pinning, a negative source
  current and unphysical p_j: independent corroboration of our closed form.
- Gildea's 2012 thesis text supplied the exact topology definition that
  explains the 0/128 and 0/56 nulls: nulls sit on the axis where the ring-cusp
  separatrix meets the centreline; the wall cusp is a separatrix intersection,
  not a vector null.

#### Guardrails For Next Session

- Do not accept the two R27 corrections without adding sheath rows (R28-R31 in
  the doc) and a declared potential closure; the literature closes potentials
  with Bohm/Boltzmann sheath relations plus a density balance, never by leaving
  them free.
- Any topology search v3 must define a cusp as axis null + separatrix + wall
  intersection z_c; a wall-side X-null search returns nothing by construction
  in a PPM stack.
- Relabel test-particle wall-hit as a geometric access fraction in every
  consumer; no published closure uses it as a cusp probability.
- Kahnfeld 2019, Matyash 2010 and Kalentev 2014 full texts were NOT retrieved
  (Springer/IEEE/arXiv timeouts); do not cite figure numbers from them until
  someone reads them.

### 2026-09-03 17:00 AEST - MDO L0 campaign v2 (catalogue x operating point, no surrogate)

#### Task Summary

- Worktree `uni-project-mdo-v2`, branch `exp/mdo-l0-campaign-v2` from `783a82c6`:
  code/tests `19c91a90`, preregistration `99914dc2`, result `a003f766`
  (`accepted_result`, 12/12 integrity gates, 1440 evaluations, ~83 min wall),
  dashboard/docs `0ea33a7e`; `feat/sota-foundation` fast-forwarded
  `783a82c6..0ea33a7e` (is-ancestor held, no rebase, no force). Detached run
  worktree `uni-project-mdo-v2-run` kept (validate_bundle root identity).
- qLogNEHVI (MixedSingleTaskGP, exhaustive categorical candidate stage +
  continuous refinement) beat LHS 3/3 and NSGA-III 3/3; seeds 202/303 reached
  1.13x the 96 x 1024 dense reference, seed 101 stuck on design 50 (0.49x).
  Robust front lives on catalogue designs 49, 50, 94 = the three lowest
  screening P(wall). CL-2 front shares 0 designs with CL-1.

#### Mistakes And Fixes

- [self] A strict `>=` hypervolume-monotone gate tripped on a -2.1e-16
  relative decrease in shakedown 1 (new nondominated point with negligible
  exclusive volume changed the slicing order). Gate now tolerates 1e-12
  relative and records the largest decrease; protocol wording updated before
  the freeze. Exact-arithmetic identities are not float identities.
- [self] Quantile-inversion test asserted the u-residual (1e-12) instead of
  x-accuracy; steep Beta(0.5, 511.5) CDFs fail that at one-ulp x accuracy.
  Bracket the quantile by +-8 ulps and require the CDF to straddle u.
- [self] Protocol sizing estimated ~50 min of BO from n<=128 probes; the
  candidate stage grew to 100+ s per batch as the baseline front grew, and
  the CPU was at 100 % from other agents: 27-30 min per BO seed, 83 min total.
  Budget NEHVI time from the LATE iterations, and add the contention factor.
- [tool] PowerShell inline `python -c` with nested quotes mangled twice
  (dashboard pin edit, devlog append); a heredoc slipped in once. Write the
  patch to `%TEMP%` and run it; use StrReplace for tracked-file edits.
- [tool] Headless Edge `--screenshot` silently wrote nothing until a fresh
  `--user-data-dir` was given (an Edge instance was already running);
  `--dump-dom` worked regardless.

#### What Worked

- Reading the v1 audit's "what a v2 must change" list first and turning every
  item into a protocol field + binding gate (F9 result-commit policy, F10
  import-bound explicit scope + `sys.modules` gate + fresh-interpreter test,
  F22 rule-computed probabilities, F26 gate semantics, F27 duplicate
  elimination gate, F28 labels rebuilt from arguments/fitted objects).
- Probing BoTorch mixed optimisers for 10 minutes before the protocol:
  `optimize_acqf_mixed_alternating` was 2.5x slower than exhaustive
  `optimize_acqf_discrete` + per-member `optimize_acqf(fixed_features)`.
- Per-design ProcessPoolExecutor for the dense reference: 98,304 evaluations
  in 54 s on 12 workers, bit-exact (pure function of (design, grid)).
- Per-design fronts then union (`nondominated_indices_blockwise`) made the
  98k-point dense front exact and cheap (0.7 s); the pairwise filter would
  have been O(n^2) = 1e10.
- Pure-Python regularised incomplete beta + safeguarded Newton quantile
  (agrees with scipy to 1e-13/1e-14) kept the frozen sample scipy-free.

#### Guardrails For Next Session

- Under CL-1 with per-design posteriors the width sensitivity is NOT a
  design-set invariance (each design's CVaR multiplier rescales
  differently): the w = 4 / point fronts differ on the common feasible set.
  Do not carry v1's invariance expectation into design-dependent priors.
- 77 of 96 catalogue designs have per-design dense HV < 1e-9 under CL-1 (73
  have a saturated 128/128 cell); any statement about "the catalogue" is a
  statement about designs 46, 49, 50, 73, 94.
- The Git-common lock `mdo-l0-campaign-v2.execution.lock` stays by design;
  the run worktree `uni-project-mdo-v2-run` is kept.
- Main tree `uni-project` is at `783a82c6`; needs `git pull --ff-only`.

### 2026-09-03 14:55 AEST - admit the wall-loss geometry screening v1 to the paper

#### Task Summary

- Worktree `uni-project-paper-geo` (`paper/geometry-screening-claim` from
  `22e2156b`): fourth `numerical-screening` gate
  `GATE-WALL-LOSS-GEOMETRY-SCREENING-V1` at a NEW recorded outcome
  `accepted-screening-dataset`, manifest type `paper-orbit-screening-manifest`
  (67 blob+sha files at `ab7c2897`, frozen files equal at `c86bfca3`,
  dashboard at `ab7c2897`), generator re-verifies all 2,835 bundle files and
  recomputes every Wilson interval exactly, 271 `Wlg` macros, 4 tables, claims
  CLM-045..052 (CLM-052 = Discussion: v4's mirror statement is field-specific;
  screening = geometry->wall-loss bridge at screening tier), CLM-016 amended
  (first coupling-export consumer), Section 11, 28 tests; `3003325d`
  fast-forwarded into `feat/sota-foundation` (`bfe123d4..3003325d`, after one
  rebase over the concurrent surrogate commits, no force). paper/tests 139 OK;
  PDF 33 pages `67a531f9`, byte-identical twice.

#### Mistakes And Fixes

- [self] The brief's "longer channels / larger wall radius lose least" is not
  a population trend (Spearman rho -0.05 / -0.12 over 96 designs; pitch +0.36,
  stage count -0.31 are the strongest). Reported extremes + rank correlations
  as observations, "none is a design rule" in the claim.
- [self] Protocol says the refined re-solve / cross-resolution runs for every
  design; `designs.py` runs it for the 4 representatives only (92 nulls, check
  passes vacuously). Derived the count from the data and disclosed it; the
  experiment README overstates it (not edited, reported).
- [self] Extremes table overflowed 6.5 pt at `\scriptsize` (found by the
  standalone compile, not by the manuscript build which refuses to run
  pre-commit); narrowed the two `p{}` columns.
- [tool] Write tool produced LF this session for every file; kept the
  post-write `\r` scan anyway.

#### What Worked

- Same flow as MDO/closure: generator -> section lint with the checker's own
  functions -> manifest from raw macro values -> claim bodies extracted from
  the section and inserted into claims.json by a scratch script (no manual
  paste) -> trial pdflatex into %TEMP% -> check_paper (only "not committed at
  HEAD") -> commit -> recheck/tests/build -> amend devlog -> rebase -> push -> ff.
- Re-implementing the orbit_mc Wilson formula and requiring EXACT equality
  held for 784 case + 1,536 cell estimates; a tolerance would only hide bugs.
- Keeping the section free of `\Wlf` macros: cross-study contrast bound via
  the bundle's own frozen protocol disclosure (`\WlgVFourHeadline`); the
  `\WlfPooledReflected` contrast lives only in the Discussion claim bound to
  both manifests.

#### Guardrails For Next Session

- A statement true at an earlier admission ("no consumer model has ingested
  it", CLM-016; "its coupling export has not been consumed", Limitations;
  "planned bridge ... future work") goes stale when a later gate opens; grep
  the manuscript AND claims.json for the previous boundary sentences.
- The geometry surrogate v1 (concurrent, `rejected_surrogate`) must NOT be
  admitted as a positive finding; if admitted at all it is a recorded null
  at its own outcome. The screening dataset stays "surrogate/MDO input under
  its label"; no consumer of it is admitted.
- Main tree `feat/sota-foundation` is behind; needs `git pull --ff-only`.

### 2026-09-03 14:45 AEST - wall-loss geometry surrogate v1 (first surrogate on physically varying data)

#### Task Summary

- Worktree `uni-project-wl-surrogate`, branch `exp/wall-loss-geometry-surrogate-v1`:
  GP surrogate of the geometry wall-loss screening dataset (96 designs, 11
  inputs, known binomial noise). Commits `aa9349a9` code/tests, `b602d147`
  preregistration, `b400d924` record, `bfe123d4` dashboard/docs;
  `feat/sota-foundation` fast-forwarded `c32dd780..bfe123d4`. Terminal state
  `assessment_rejection` / `rejected_surrogate`: pooled RMSE 0.0562 (gate 0.05),
  did NOT beat ridge (0.0546) or the global mean (0.0553); floor-corrected cell
  RMSE 0.129; coverage 0.80 (met at the boundary after a 3.34x variance
  inflation). Structural gates all passed. Honest negative, gates untuned.

#### Mistakes And Fixes

- [self] Shakedown run 1 ended `assessment_rejection` because the shakedown
  decision reused the science gates; fixed so the shakedown is decided by the
  structural gates only (science gates informational). Otherwise every honest
  negative preview looks like a broken pipeline.
- [self] `predictor.scope_flags` assumed the geometry-specific input names;
  a toy contract in the tests exposed the KeyError. Make consumer contracts
  degrade gracefully when optional inputs are absent.
- [tool] PowerShell has no heredoc (`python - <<'EOF'` is a parse error);
  write patch scripts to `%TEMP%` (already in the scratchpad; repeated anyway).

#### What Worked

- Reading `.venv-sota` BoTorch 0.18.1 before writing: `SingleTaskGP` defaults
  to RBF (pass `get_covar_module_with_dim_scaled_prior(use_rbf_kernel=False)`
  for Matern-5/2 ARD); `MultiTaskGP` accepts `train_Yvar`; both posteriors
  reproduce in numpy to 1e-14, so one GP contract block format covers the
  package ExactGP, STGP and ICM and the predictor gate is meaningful.
- Holding out the top-decile chamber lengths BEFORE splitting the rest into
  roles: one frozen model serves both scopes, no label is ever both fitted and
  assessed, and the extrapolation cluster is a deterministic function of data.
- Declaring the binomial floor (cells 0.035, pooled 0.020) and reporting raw
  + floor-corrected RMSE beside the user's gates instead of loosening them.
- Rebasing the unpushed prereg onto the moved base and checking the hash scope
  was untouched (`git diff --stat base..origin -- <scope>` empty) before pushing.

#### Guardrails For Next Session

- Any future wall-loss surrogate must change the INPUTS (realised geometry or
  cell position relative to the cusps) or the DATA (more designs / launches),
  not the model: the step discontinuities of the design -> geometry map
  (`stage_count_selector`, `exit_length_fraction`) carry the permutation
  importance and a stationary kernel cannot follow them.
- Method selection on 10 designs is unstable (shakedown chose
  `botorch-stgp-direct`, the evidentiary run `botorch-icm-logit`); a v2 needs
  a larger method-selection role or nested CV declared up front.
- Do not use `results/artifacts/predictor.json` as an MDO input: the recorded
  contract is rejected and is published for audit only.
- Main tree `uni-project` is behind; needs `git pull --ff-only`. The detached
  run worktree `uni-project-wl-surrogate-run` is kept (validate_bundle root
  identity); the Git-common lock `wall-loss-geometry-surrogate-v1.execution.lock`
  stays by design.

### 2026-09-03 13:55 AEST - orbit wall-loss geometry screening v1

#### Task Summary

- First wall-loss-vs-GEOMETRY dataset: orbit_mc 1.7 on 96 accepted L1a
  sweep-v2 fields (re-solved on CPU), 100 352 orbits, one preregistered
  execution, accepted as `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`.
  Branch `exp/orbit-wall-loss-geometry-screening-v1`, ff'd into
  `feat/sota-foundation` (`22e2156b`).

#### Mistakes And Fixes

- [self] Shakedown run 1 died on `CanonicalizationError: unsupported canonical
  type: bool` (numpy bools from `np.float64` comparisons in the field-evidence
  checks). Fix: `_plain()` every worker payload before it reaches the runtime.
  The synthetic tests never produced a numpy bool; the real re-solve did.
- [self] Wrote the test `wilson_interval(...).__dict__` for a slots dataclass;
  use `dataclasses.asdict`.
- [self] Projected 70 min from the shakedown's per-orbit cost; actual 95 min
  (median 2N per-orbit 242 ms, max 650 ms) under the concurrent MDO/PIC load.
  Projections from a 3-worker shakedown under-estimate 12-worker contention by
  ~35 %; budget with that factor.
- [tool] `.axis path` in inline SVG renders as a filled black polygon unless
  `fill:none`; visible only in the headless screenshot.
- [tool] `Start-Process python -m http.server` then Chrome immediately ->
  ERR_CONNECTION_REFUSED; poll `netstat` for LISTENING first; ports 8765/8766
  are held by other agents' stale servers.

#### What Worked

- [self] Re-solving accepted L1a cases on CPU reproduced the stored CUDA maps
  to 1e-21 Wb / 5e-16 T; identity proven three ways (geometry/source/config/
  case hashes vs the sealed raw record, QoI replay under the sweep's own
  tolerances, node-wise agreement for the representatives).
- [self] One worker task per DESIGN (all cases, then the N->2N check, then
  sealing) satisfies orbit_mc's requirement that all three
  `convergence_evidence` flags be true before `result_artifact` seals.
- [self] Binding `experiment_code_sha256` (own code) into shakedown + authorities
  in addition to orbit_mc/field-pipeline hashes.
- [self] Bundle-size policy: sidecar + compact gzipped endpoint table for every
  case, full orbit artifacts only for the representatives -> 23.8 MB for 196
  cases (v4: 17 MB for 9).
- [self] Clean `--no-ff` merge of the concurrent branch (no overlapping paths)
  kept every SHA incl. the preregistration commit bound in the lock, and let
  `feat/sota-foundation` fast-forward without force.

#### Guardrails For Next Session

- Consumers of `geometry-wall-loss-dataset.{json,csv}` must carry the screening
  label; reflections (22 % of orbits, every design) mean the mirror picture is
  NOT falsified on L1a fields the way it was on the v4 P2 field - do not
  generalise v4's "zero reflections" beyond its design.
- Any orbit_mc/fields/geometry change now drifts from this campaign's frozen
  hashes; `tests/experiments/orbit_wall_loss_geometry_screening_v1` is
  lifecycle-aware (recorded contract after execution) but re-check.

### 2026-09-03 13:10 AEST - MDO L0 campaign v1 post-hoc audit (audit/mdo-l0-v1-posthoc)

#### Task Summary

- Worktree `uni-project-mdo-audit` from `ba6875f6`; overlay = `POSTHOC_AUDIT.md`
  + read-only `audit_replay.py` + `tests/experiments/mdo_l0_campaign_v1/
  test_posthoc_audit.py` (19 tests); commit `6cb9a1af` -> rebased `e9f9af16`,
  `feat/sota-foundation` ff `22e2156b..e9f9af16`. Verdict ACCEPTED WITH DISCLOSURES: 137/137
  manifest files byte-exact (0 EOL-only, orbit_mc v1.7 fix confirmed), source
  hash reproduced from Git blobs at 4 commits, 864 records + dense 8192 +
  NSGA-III 3/3 + qLogNEHVI seed 101 bit-exact, independent chain <= 3e-13.
  Six disclosures (F9 spec/test files in the result commit, F10 hash scope
  inert/incomplete, F22 Jeffreys rounding, F26 integrity-only gates, F27
  NSGA-III duplicates, F28 wrong descriptive labels); no defect in evidence.

#### Mistakes And Fixes

- [self] Asserted `semantic_sha256 == byte sha` for every sidecar; the
  `write_blob` shakedown copy carries `semantic_sha256: null`. Read the sidecar
  contract before asserting a uniform rule (same lesson as v4's lock binding).
- [self] Pareto-index comparison listed every index of a duplicated design;
  the package keeps the FIRST index only (NSGA-III re-evaluates duplicates).
  Use first-occurrence semantics when re-deriving `pareto_record_indices`.
- [self] Recomputed pooled fronts in strategy-major order; the assessment
  iterates seed-major, so first-occurrence records and list order differ
  though the sets are equal. Reproduce the producer's iteration order before
  calling a canonical-bytes comparison a mismatch.
- [self] Task text "thrust <= 2.70e-9 N" is a 3-s.f. rounding of 2.7027e-9;
  literal `<=` failed again (v4 lesson repeated). Assert the exact value and
  the formatted string separately, always.
- [tool] `python ... | Tee-Object` swallowed every progress line until exit
  (~1 h blind). Use `cmd /c "... > out 2> err"` with `PYTHONUNBUFFERED=1` for
  long jobs whose progress matters.
- [tool] `.venv-sota` has no pytest (earlier "venv N passed" runs must have
  installed it ad hoc); installed pytest 9.1.1 into the venv only.

#### What Worked

- Independent re-implementation first (pure-Python L0/CL-1, CVaR, pairwise
  dominance, WFG hypervolume, stdlib LHS, radical inverse, Wilson) then the
  package replay: agreement to 1e-15 everywhere, so "bit-exact" and "within
  tolerance" could be reported separately and honestly.
- Recomputing the code-contract hash from `git cat-file blob` at four commits
  proved the chain without trusting the working tree; the import trace
  (`sys.modules` delta) exposed the inert/incomplete hash scope in one step.
- Access-record spacing vs recorded run wall clocks (gap >= wall) is a cheap,
  strong timeline consistency check; do it on every bundle.

#### Guardrails For Next Session

- Under CPU contention (12 foreign workers + PIC), a 520 s BoTorch replay took
  3708 s with iteration 1 alone 2707 s; results were still bit-exact with 24
  threads. Never change `torch` thread count to speed a bit-exact replay.
- Reported-not-binding "BO beats X >= 2/3 seeds" passes with p = 0.5 under the
  null; 3/3 is p = 0.125. Report counts, never significance, at 3 seeds.
- `audit_replay.py --bo` and the BO test are slow; `MDO_AUDIT_SKIP_BO=1` skips
  the test explicitly (it is otherwise run whenever torch imports).

### 2026-09-03 13:40 AEST - admit the four-cell closure analysis to the paper

#### Task Summary

- Worktree `uni-project-paper-closure` (`paper/four-cell-closure-claim` from
  `ba6875f6`): new gate kind `analytic-consistency`, gate
  `GATE-FOUR-CELL-CLOSURE-V1`, manifest type
  `paper-analytic-consistency-manifest` (14 blob+sha files at `266d8a99`,
  verified unchanged at `ba6875f6`; executed `cft_revival.plasma` must equal
  the blobs), generator that RECOMPUTES the verification (closed form 2.0e-13
  over 400 states; ladder floors within 6% of the doc; anode-only 6/6;
  DM9.2 misfit 1.47e-3; relaxed root 1.18 V; rank 22/25) and reads the 13/80
  probe from the frozen MDO protocol; 163 `Fcc` macros, 2 tables, claims
  CLM-036..044 (CLM-043/044 = labelled Discussion interpretations), Section
  10 with the closed form displayed (coefficient + row index macro-bound),
  28 tests; `d09ffee2` fast-forwarded into `feat/sota-foundation`
  (`ba6875f6..d09ffee2`, no rebase needed). paper/tests 111 OK; PDF 27 pages
  `6ac978b2`, byte-identical twice.

#### Mistakes And Fixes

- [self] First anode-only recomputation used 1 start with a 1e-12 bound; the
  solver stops at `residual_tolerance` 1e-8 (or gradient tolerance), so the
  achieved residual is anywhere below 1e-8. Bound = the declared tolerance;
  5 starts reproduce the doc's 1e-13..1e-16.
- [self] f-string `{}` inside TeX macro calls (`\Fcc{}~V`) is a syntax error;
  double the braces.
- [self] Wrote a nested-`$` caption and an overfull typewriter expression;
  both caught by the standalone compile before the manuscript build.

#### What Worked

- Same flow as v4/topology/MDO: generator -> lint the section with the
  checker's own functions -> build the manifest from the evidence raws ->
  extract `authorized_tex` from the flattened tex -> check_paper (expect only
  "not committed at HEAD") -> commit -> recheck -> tests -> two builds ->
  amend devlog -> push -> ff.
- Reading documented numbers from the bound blob with named fixed regexes
  (pointer = `regex:<name>[<group>]`) and requiring recomputed values to sit
  within declared tolerances made "verify every number from the files"
  mechanical and fail-closed.

#### Guardrails For Next Session

- Any change to `modern/src/cft_revival/plasma/*.py` now fails
  `check_paper.py` ("checkout differs from the blob admitted at 266d8a99");
  re-admit the analysis at the new revision (new manifest revision, rerun
  the generator) rather than loosening the binding.
- The paper's legacy-study reading is INTERPRETATION (CLM-043/044); never
  promote it to a claim without the legacy run's artifacts.
- Main tree `feat/sota-foundation` is behind; needs `git pull --ff-only`.

### 2026-09-03 12:05 AEST - admit MDO L0 campaign v1 to the paper

#### Task Summary

- Worktree `uni-project-paper-mdo` (`paper/mdo-l0-v1-claim` from `e642f38c`):
  second `numerical-campaign` gate `GATE-MDO-L0-V1` (`opens_level: null`,
  `kind_justification` on the gate), manifest type
  `paper-mdo-campaign-manifest` (36 blob+sha files at `c553124b`, frozen files
  at `4898d0fd`, dashboard at `e642f38c`, 68 raw-bound + 10 policy metrics),
  generator with 334 `Mdo` macros and 3 ArtifactClaim tables, claims
  CLM-029..035, Section 9 + Discussion + Limitations, `_check_mdo_campaign`,
  25 tests; `ba6875f6` fast-forwarded into `feat/sota-foundation`
  (`266d8a99..ba6875f6`, after one rebase over the concurrent plasma commit).

#### Mistakes And Fixes

- [self] Assumed every bundle artifact has a `.sha256.json` sidecar;
  `execution-lock.json` is bound by `manifest.lock_byte_sha256` instead.
  Read the manifest's own binding fields before asserting a sidecar rule.
- [self] Assumed the frozen `protocol.json` blob equals the sealed
  `artifacts/protocol.json`; the bundle seals canonical JSON, the experiment
  keeps the pretty-printed file. Compare parsed payloads, keep the blob
  binding at the prereg commit, and say so in the manifest.
- [self] A 3-significant-digit formatter via `:.3g` emits `1.63e+03` for 1627;
  compute the decimal count from `floor(log10)` instead and fall back to
  `\times10^{}` only outside `[1e-3, 1e5)`.
- [self] An eight-column table with `\times10^{}` cells overflowed by 229 pt
  at `\footnotesize`; ragged-right `p{}` label columns + `\shortstack`
  headers + `\scriptsize` fixed it without dropping a number. The log
  reports the overfull at the section's macro-invocation line, not inside
  the generated file.
- [tool] Concurrent agents push between "ancestor check" and "push": the ff
  push was refused once (non-force, as intended); fetch, rebase, rerun
  check/tests/build, then ff. Content-only PDF hash was unchanged.

#### What Worked

- Same flow as v4/topology: generator -> section (lint digits/unregistered
  prose with the checker's own functions) -> build manifest metrics from the
  evidence file's raw values -> extract `authorized_tex` from the flattened
  manuscript -> check_paper (expect only "not committed at HEAD") -> commit ->
  recheck -> build -> amend with devlog -> push -> ff.
- Binding the results dashboard as a second independent extraction of the
  bundle (payload must equal the sealed artifacts; files bound by
  LF-normalised sha == blob at its revision) cost little and adds a reader.
- Regex-parsed derived macros make free-text protocol disclosures (13/80
  probe) macro-bound and fail-closed.

#### Guardrails For Next Session

- The Discussion/section phrase "closed only for p = 0 in the recorded probe"
  is scoped to the frozen protocol's probe; the concurrent
  `global-plasma-closure-analysis.md` reproduced 13/80 exactly and proposes a
  correction (`PROPOSED_NOT_ACCEPTED`). If that correction is accepted, add a
  new admission; do not edit the recorded disclosure.
- Boundary sentences elsewhere in the manuscript can go stale when a gate
  opens (Limitations said "no admitted hypervolume result"); grep the
  manuscript for the previous "none exists" claims at every admission.
- Main tree `feat/sota-foundation` is behind; needs `git pull --ff-only`.

### 2026-09-03 11:35 AEST - four-cell closure analysis for p != 0 (fix/plasma-network-closure)

#### Task Summary

- Worktree `uni-project-plasma-closure`, branch `fix/plasma-network-closure`
  from `e642f38c`; one commit `266d8a99`, pushed; `feat/sota-foundation`
  fast-forwarded `e642f38c..266d8a99` (no force). Reproduced the MDO probe
  exactly (13/80, floors 4.739e-4..0.196, 2.9 s/solve), derived the closed
  form of R27 on the 27-row manifold, classified (a) model inconsistency with
  (d) sub-region p1=p2=p3=0 and a secondary (b) projection defect (fixed).
  Docs `global-plasma-closure-analysis.md`, ledger block
  `global_row_consistency` (PROPOSED_NOT_ACCEPTED), 24 new tests.

#### Mistakes And Fixes

- [self] First anode-only pin test used potentials whose electron cascade
  carried 0.13 A against Ia = 1 A: residual closed but `ji4 <= 0` failed.
  Feasibility of a manifold point is a separate check from residual closure;
  pick phi1 so that je4 >= Ia (21 V at 300 V / 1 A).
- [self] Network failure results carry empty `equation_residuals`; the row
  ledger of a failed network solve lives in `diagnostics.backend`.
- [tool] `Set-Content -Encoding utf8` put a BOM into the commit subject;
  strip with Python and `--amend` before pushing. Check `git log -1 --format=%s`.
- [tool] Write tool produced LF this session, but keep the post-write
  `\r` scan; do not rely on it.

#### What Worked

- Recovering the exact probe inputs from the MDO agent transcript.
- Substitution algebra first, numerics second: the closed form predicted the
  documented DM9.2 R27 misfit (1.47e-3 vs recorded 1.49e-3), the anode-only
  closure, the linear-in-eps floor, and the relaxed-constraint roots before
  any of them were run.
- `pdftotext` on the IEPC-2007-108 PDF (MiKTeX) gave the source prose that
  settled the origin (assumption 8) in one sentence.
- Trialling the isotonic projection in a scratch copy of the solve loop over
  the full 16-case zero-cusp grid before touching the package.

#### Guardrails For Next Session

- Never accept the proposed R27 corrections without adding a potential
  closure; the corrected ledger has rank 21 and would publish arbitrary
  potentials.
- The solver is usable for anode-cusp-only loss; any MDO v2 with interior
  cusp probabilities still needs the model decision, not more solver work.
- `experiments/cft_topology_characterization_v1/tests/...::
  test_accepted_dependencies_match_coupling_v3_commit` fails on the base
  commit already (`git diff --quiet f80a360f -- modern/spec ...`); not
  collected by `modern/tests`, not caused here.

### 2026-09-03 10:25 AEST - MDO L0 campaign v1 (first optimiser run on the new physics)

#### Task Summary

- Worktree `uni-project-mdo-v1`, branch `exp/mdo-l0-campaign-v1`: tests
  `fdc6b37d`, preregistration `4898d0fd`, result `c553124b`
  (`accepted_result`, 8/8 binding gates, 864 evaluations, 28 min), dashboard
  `e642f38c`; `feat/sota-foundation` fast-forwarded `8babb31e..e642f38c`.
  qLogNEHVI beat LHS 3/3 and NSGA-III 3/3 at 96 evaluations (HV 0.00386 vs
  0.0029/0.0032; 1.02x the 8192-point dense reference).

#### Mistakes And Fixes

- [self] Wanted the corrected Kornfeld solver as the cusp -> performance
  bridge; a 20-minute probe showed it never closes for p != 0 (13/80, all at
  p = 0) at 3 s/solve. Probe the chain before writing the protocol.
- [tool] First in-repo BoTorch call failed: `botorch.optim` has no
  `optimize_acqf_list` in 0.18.1 (`botorch.optim.optimize` does); replacing
  `acqf.sampler` after construction breaks cached base samples. Fixed in
  `botorch_adapter.py` (tests commit).
- [tool] GP fits/acquisition on the PIC-occupied `cuda:0` were 20-40x slower
  than cpu (fit 13.6 s vs 0.3 s; acq 250 s vs 25 s). Protocol declares cpu.
- [self] The system interpreter has numpy but no scipy; a scipy-Sobol
  baseline would have errored every campaign test there. Replaced by a
  stdlib-RNG two-stage LHS before the freeze.
- [self] `verify_shakedown_record` compared `kind:strategy:seed` run ids to
  `strategy:seed` keys (caught by the tampering tests, not the shakedown);
  `plan_record` tuples became `__cft_type__` tags that made the shakedown
  record un-rehashable in `prepare`. Each fix -> fresh shakedown (5 total).
- [self] Exact front-set invariance tripped on a one-ulp anode-power tie
  (L0 recomputes `Ua*I_beam/beam_fraction`); use roundoff-aware dominance
  and restrict invariance checks to the common feasible set.

#### What Worked

- Shakedown -> tests -> fix loop caught four defects before the freeze; the
  evidentiary run then passed first time. CPU float64 BO reproduced HV
  bit-identically across repeated shakedowns.
- Separability theorem stated as a predeclared expectation, then confirmed
  (ratio spread ~1e-15; identical common-set fronts for four priors).
- Rebased tests+prereg onto the moved base BEFORE pushing the prereg; the ff
  into `feat/sota-foundation` was then trivial.

#### Guardrails For Next Session

- A worst-case constraint over a finite QMC sample enforces the worst
  SAMPLED case (max S 0.704 here, not 1); the `no_wall_loss` scenario made
  110/114 robust-Pareto designs infeasible. Add the support vertex or say so.
- Under CL-1 the cusp probabilities rescale three objectives uniformly; the
  optimisation is only non-trivial through the feasible set. Geometry stays
  excluded until a geometry -> L0 map exists.
- `campaign-v1.json#benchmark.results` must stay null (validator requires F3
  evidence); use the instance index `spec/optimization/mdo-l0-campaign-v1.json`.
- Main tree is behind 4 (`git pull --ff-only` needed; not done from here).

### 2026-09-03 07:10 AEST - PIC-2D phase 2: v1.1 all-GPU step, snapshot v2

#### Task Summary

- Worktree `uni-project-pic2d` (`feat/pic-2d-axisymmetric`): phase-1 merge
  `62de2ca3`; v1 diagnosis; model v1.1 (device block-Thomas, tile-reduced
  kernels, ion subcycling k=8, electrode-work ledger; 40.7 -> 5.46 ms/step
  at 5.4 M); snapshot v2 at 3 mA / 1e20 m^-3 / 1.5 ps: no plateau, density
  ran 3.7-5.9x past the a-priori ceiling, coarse pair gate-stopped, not
  converged. Three commits, `feat/sota-foundation` fast-forwarded to
  `1cdaae80`.

#### Mistakes And Fixes

- [tool] `(Get-Content f) -replace ... | Set-Content -NoNewline` joined 450
  lines into one and destroyed an UNTRACKED runner; recovered by splitting on
  runs of >=4 spaces (indentation survived). Fix: commit WIP before any
  shell-side rewrite; never rewrite tracked text with PowerShell one-liners.
- [self] First v2 attempt (dt = 3 ps) tripped the omega_pe dt gate on
  axis-node shot noise (node peak 3x the window peak). Budget dt against the
  instantaneous node peak, not the physical peak.
- [self] The 0-D n_eq with unmagnetised Bohm loss is a loose LOWER bound in a
  cusped field (kinetic loss/source only 0.10-0.35 after one transit time);
  set the next operating point from the measured kinetic loss fraction.
- [tool] Warp: one block width per module (tile kernels at 64 vs default 256
  fail to compile together); `tile_broadcast`/`untile` break on the CPU
  device, `tile_extract(tile_sum(tile(x)), 0)` works everywhere.
- [tool] `Get-Item f | Select-Object Length` prints nothing for one file in
  this shell; use `(Get-Item f).Length`.

#### What Worked

- Diagnosing from artifacts (checkpoint peak node, series, ledger) before
  touching code; fail-closed gates ended the bad attempt in minutes.
- Per-block tile reductions + one atomic per block removed the same-address
  atomic serialisation; tallies stay exact integers vs numpy.
- Grid heating identified by three agreeing signals (coarse hotter 1.5-1.7x,
  ionisation 3.5x, ledger residual +41 % vs -13..-18 % fine).

#### Guardrails For Next Session

- Do not edit a snapshot's `protocol.json` after its runs (hash bound into
  every summary); record post-hoc interpretation in the dashboard/devlog.
- The <= 1.5 ms/step target at 1-2 M is launch-bound on WDDM (~1.2 ms floor,
  2.0 ms measured); only full-step CUDA-graph capture or a TCC/Linux host
  moves it.

### 2026-09-03 04:55 AEST - admit L1a sweep v2 + topology nulls to the paper

#### Task Summary

- Worktree `uni-project-paper-topo` (`paper/topology-and-sweep-claims`): new
  gate kind `numerical-screening` with `recorded_outcome`, three gates /
  typed manifests / evidence files / macro-only sections, claims
  CLM-018..028, screening checker, 25 tests, four-cell v2 posthoc EOL audit
  (`605be5ce`), paper admission `f171e9ec`; `feat/sota-foundation`
  fast-forwarded `7a30fc2e..f171e9ec`.

#### Mistakes And Fixes

- [self] "L1a" in a `\subsection` title tripped the literal-digit rule;
  renamed the heading and rendered the model level through a macro instead
  of exempting headings.
- [self] The unregistered-quantitative heuristic is case-insensitive:
  "v2-031, v2-063" and "+1, a nearby" matched `\d[\d,]* \s [WVASN]`. Fixed
  with `\texttt{}` id lists and ";"-joined clause lists, not a weaker rule.
- [tool] `-output-directory='$out'` in single quotes wrote a literal `$out`
  dir inside `paper/`; double-quote PowerShell variables and check
  `git status` before staging.
- [tool] `$env:SOURCE_DATE_EPOCH` left over from a trial TeX build changed
  the plasma-topology dashboard footer time and failed its byte-identity
  test; unset build env vars before running `modern/tests`.
- [self] Unbreakable `\texttt{}` cells overflow `p{}` columns; use
  `>{\raggedright\arraybackslash}p{}` and size the column to the longest
  identifier.

#### What Worked

- [self] One generic `Bundle.verify` with audited files as data covers both
  CRLF-era defects; the checker re-derives the CRLF digest on disk and
  requires both digests verbatim in the experiment's own audit module.
- [self] Build the manifests from the evidence file's raw macro values with a
  scratch script (`@Macro` references) so metric == raw holds by
  construction; type-equal comparison catches `5` vs `5.0`.
- [self] Report the bundle's own GPU-replay outcome (2 of 4) inside the
  results claim rather than dropping it; the null does not depend on it.
- [self] Same flow as v4: check_paper before commit (expect only "not
  committed at HEAD"), commit, recheck, amend locally with devlog, push, ff.

#### Guardrails For Next Session

- A null result gets a gate whose `accepted` status cannot read as "finding
  accepted": carry `recorded_outcome` and make gate, manifest, evidence and
  generator agree; phrase as "not shown stable", never "does not exist".
- Lineage (superseded/failed studies) lives inside a registered non-claim
  with `lineage_files` bound at their own revisions and `lineage-` roles.
- `four_cell_topology_search_v2/experiment.py::validate_results` still binds
  the live LF protocol digest and refuses the bundle on LF checkouts (its
  experiment-local test is not collected by `modern/tests`); disclosed in
  the audit, not fixed.

### 2026-09-03 03:15 AEST - PIC-2D axisymmetric PIC-MCC build

#### Task Summary

- Delivered `cft_revival.pic2d` + snapshot experiment + dashboard on
  `feat/pic-2d-axisymmetric` (dd5f2ff1); report handed to the parent agent.

#### Mistakes And Fixes

- [self] Geometric axis volumes bias the axis density by 4/3; shape-function
  volumes with the V_geom/V_shape source ratio fix it (uniform-density test).
- [self] Restarting CG every chunk stalled convergence (5e-3 for 550 its);
  check the recurrence residual between chunks, recompute truth at the end.
- [self] Jacobi-PCG on a 31×241 masked cylindrical grid needs ~470 iterations
  per step even warm-started; the exact block-Thomas column factorisation
  (1–20 ms per solve) replaced it as the default on both backends.
- [tool] Warp `@wp.func` receives `uint32` RNG state by value: a wrapper around
  `wp.randf(state)` returns the same number forever. Draw inline.
- [tool] Per-thread serial reductions (256 dependent loads) cost ~180 µs per
  dot product on the RTX 5090; strided 4096-thread stages cost ~8 µs.
- [tool] Every device→host read waits for the queued kernels (~0.5 ms);
  consolidate per-step statistics into one array and read once.
- [tool] Root `.gitignore` has `results/` and `*.npz`; negations must be
  placed AFTER those rules or they are re-ignored.
- [tool] `cursor-ide-browser` created tabs that immediately vanished from the
  subagent context and refuses file:// URLs; headless Edge
  (`msedge --headless=new --screenshot`) against a local http.server worked.
- [tool] PowerShell mangles quotes in `python -c` and has no heredocs; write
  patch scripts to %TEMP% and run them.

#### What Worked

- Scratch scripts in %TEMP% before formal tests; sharing the host field solve
  between CPU and Warp backends made φ bit-identical so parity tests isolate
  the particle kernels.
- Treating the runtime stability gate stop as a recorded outcome (not a crash)
  kept the snapshot artifacts valid and honest.

#### Guardrails For Next Session

- Any PIC operating point must be checked against the resolvable density
  envelope (ω_pe Δt, λ_D/Δx) before spending GPU time; the v1 model runs away
  in density at 300 V / 0.1 A / 5e20 m⁻³.
- Do not loosen the ω_pe Δt gate to get longer runs; change Δt/grid/point.

#### Follow-Ups / Risks

- Ledger residual ≈ 2 % per interval is untracked electrode work; add an
  electrode-work term before claiming energy closure.
- Preregistration needs: milder operating point or finer budget, ion–neutral
  collisions, SEE, exit-boundary sensitivity, cross-code comparison.

### 2026-09-03 03:20 AEST - admit wall-loss v4 to the paper claim matrix

#### Task Summary

- Worktree `uni-project-paper-v4` (`paper/wall-loss-v4-claim`): claim records
  CLM-012..017, typed campaign manifest, `numerical-campaign` gate
  `GATE-WALL-LOSS-V4`, checker extensions, manuscript integration, 14 new
  adversarial tests; `6f3e6dd5` fast-forwarded into `feat/sota-foundation`.

#### Mistakes And Fixes

- [self] Wrote claim IDs (`CLM-013, ...`) in the section's TeX header comment;
  the detached-ID rule scans masked text including comments and failed.
  Keep IDs out of comments; do not loosen the rule.
- [self] Read `\newcommand{\Macro}{...}` bodies with a lazy regex ending at
  `}\s*$`; a multi-line body with `\allowbreak{}` matched the first line.
  Use `extract_macros(text, "newcommand", 2)` (brace-balanced) and strip
  `%` comments before comparing.
- [self] Rewording "All current result statements" to "The L0 result
  statements" produced a 2.9 pt overfull box before a 40-hex `\texttt` hash;
  `build.py` treats any overfull as fatal. Reworded again; compile the
  manuscript early, before the commit gate forces a round trip.
- [self] The standalone section driver lacked `microtype` and showed an
  overfull the manuscript (with microtype) did not; mirror the manuscript's
  packages in any standalone driver.
- [tool] Both shared memory files moved under me again (concurrent agent);
  re-read the anchor immediately before writing.

#### What Worked

- [self] Extract `authorized_tex` from the flattened manuscript with the
  checker's own `extract_macros` + `_normalize_tex` and paste the JSON
  strings; zero body mismatches on first run.
- [self] Run `check_paper.py` before committing: with `require_committed`,
  the only expected pre-commit error is "manifest not committed at HEAD";
  everything else must already be green. Then commit, rerun, amend locally.
- [self] Flattening `\input{sections/...}` before every prose check closed a
  real bypass (section prose was invisible to `find_unregistered_claims`).
- [tool] MiKTeX ships `pdftoppm`/`pdftotext`/`pdfinfo`; `pdftotext -f N -l N`
  per page locates sections, `pdftoppm -r 70 -png` renders pages for review.

#### Guardrails For Next Session

- A campaign that is not a physics level gets a `numerical-campaign` gate
  with `opens_level: null`; never open GATE-L1 for it.
- Any new admitted section: macro-only numbers, `\EvidenceClaim` bodies
  registered verbatim, `allowed_locations` = the section's `\subsection`
  title, `non_claims` listed on the main claim record.
- After editing `generate_wall_loss_v4_evidence.py`, regenerate and commit
  all three outputs together; the checker regenerates on every run.
- "Never force" after a rebase: the pushed feature branch stays at the
  pre-rebase SHA; only `feat/sota-foundation` carries the rebased commit.

### 2026-09-03 02:50 AEST - stale L0 design gallery / first-results dashboards

#### Task Summary

- Worktree `uni-project-vizfix`, branch `fix/visualization-stale-gallery`.
  `tests/visualization` was 65 passed / 2 failed / 13 errors; root cause was
  the CRLF-smudged `config_sha256` pin (`a4703ac1` = CRLF bytes of
  `config/l0-deterministic-sweep.json`; committed blob always LF =
  `2d727b1a`). Regenerated `design-gallery.json` and `first-results.html`
  from their generators, updated the two provenance pins, 93/93 green after
  rebasing onto the concurrent v4-dashboard push; fast-forwarded
  `feat/sota-foundation` `5b85d2ad..8466c37a`.

#### What Worked

- [self] Classified the pin before editing: `sha256(blob)` vs
  `sha256(blob.replace(b"\n", b"\r\n"))` against every recorded value; a hit
  on the CRLF variant proves "EOL artefact, data unchanged" in one step.
- [self] Reduced the 6 MB HTML diff to `old.replace(OLD, NEW) == new` instead
  of a text diff; `difflib.SequenceMatcher` on the file never finished.
- [self] Regenerated the three "possibly stale" sibling dashboards to `%TEMP%`
  and byte-compared before believing the stale report: all identical.

#### Mistakes And Fixes

- [self] Ran the first `SequenceMatcher` diff in the foreground; it hung for
  3 min before I killed it and switched to the substitution check.
- [tool] Both memory files were edited by a concurrent agent between my read
  and my write; `StrReplace` failed safely. Re-read the anchor immediately
  before writing shared memory files.

#### Guardrails For Next Session

- Preregistered experiment dashboards that pin a *recorded* hash and
  cross-check it against sidecars + `results/*.json` (L1a geometry sweep v2:
  `64b2c58c` recorded, LF blob `2a5ba9e4`) cannot be fixed by a pin edit; the
  options are editing frozen artefacts (forbidden) or an EOL-normalised
  verifier with disclosure (user decision). Report, do not patch.
- `.gitignore:48 Results/` + `core.ignorecase=true` ignores every `results/`
  on Windows; `experiments/l1a_geometry_sweep/results` was never tracked, so
  its dashboard test only passes in the main tree.
- Sweep `tests/experiments/<exp>` leaf dirs one at a time too (same-basename
  clash); a full sweep is ~11 min (`l0_surrogate_v8/v9` ~160 s each).

#### Follow-Ups / Risks

- Red, not mine to fix: `tests/material_fields` (4, `raw run hash binding
  failed`), `tests/experiments/{l1a_geometry_sweep_v2, l1a_field_surrogate_v1,
  l1a_field_surrogate_v2}` (CRLF protocol sidecar), `l0_surrogate_v3/v4`
  (pre-execution tests assert `results/` absent), and the two L1a sweep
  dashboards above.
- This scratchpad is past the ~200-line consolidation threshold; roll over
  (archive-first) once the concurrent agents have handed off.

### 2026-09-03 02:40 AEST - wall-loss v4 dashboard + paper evidence

#### Task Summary

- Built `modern/visualization/wall-loss-v4-results.html` (generator +
  template + 13 tests) and the paper evidence chain
  (`paper/scripts/generate_wall_loss_v4_evidence.py`, evidence JSON, generated
  macros/tables, draft `paper/sections/wall-loss-v4.tex`, 8 tests) in worktree
  `uni-project-v4-dash` on `feat/wall-loss-v4-dashboard`; fast-forwarded into
  `feat/sota-foundation` (`5b85d2ad`) after rebasing over v1.7/audit/topology.

#### Mistakes And Fixes

- [self] Started `python -m http.server 8765` in the visualization folder, but
  five stale servers from other agents already shared port 8765 on this
  machine, so Chrome fetched another agent's directory (404s). Check
  `netstat -ano | Select-String ':PORT '` before serving; pick a fresh port and
  pass `--directory` explicitly.
- [self] Set a `<table>` element's innerHTML to `...</table><p>...`; the HTML
  parser foster-parented the paragraph and the field panel wrapped into a
  narrow column. Render mixed markup into a wrapper `<div>`.
- [self] Ordered categorical lists (N/2N/4N, primary/refined/enlarged) came
  out alphabetical because `sort_keys` JSON reorders dict keys; ship explicit
  order arrays in the payload.
- [self] `\paragraph{...}` immediately followed by a boxed minipage overflows
  by the heading width (run-in heading joins the box paragraph); put a sentence
  between them. Long `\texttt` identifiers need `\allowbreak{}` after `\_`.

#### What Worked

- [self] Verifying the whole results bundle first (387 files vs manifest) and
  cross-checking every repeated number (terminal==campaign==gates, summaries
  ==orbits, strata==per-orbit outcomes, Wilson bounds recomputed) made the
  dashboard and paper evidence provably artifact-bound; the tests re-derive
  everything independently of the generators.
- [tool] Headless Chrome via `--dump-dom --virtual-time-budget` with an
  injected error hook is a workable substitute when the IDE browser tab is
  unavailable and `file://` is blocked; a 390 px iframe host page gives a true
  narrow viewport (headless clamps windows to ~512 px) and an offset iframe
  captures page tails (height clamps at ~5400 px).
- [self] Recording `sha256(bytes.replace(CRLF, LF))` for generator/template
  hashes makes provenance footers checkout-neutral.

#### Guardrails For Next Session

- [self] Paper provenance sidecars hash working-tree bytes; after any eol
  change rerun `python paper/scripts/generate_tables.py` (the L0 sidecar was
  stale on every LF checkout and failed `check_paper.py`; fixed in `ea867bf1`).
- [self] Pre-existing: `tests/visualization/test_design_gallery.py` and
  `test_first_results_visualization.py` fail on origin (stale
  `design-gallery.json`), unrelated to the dashboards.
- [user] "Never force": after a rebase the pushed feature branch
  `origin/feat/wall-loss-v4-dashboard` stays at the pre-rebase commits
  (`fb4117e4`); the integrated history lives on `feat/sota-foundation`.

#### Follow-Ups / Risks

- The draft subsection is not `\input` into `manuscript.tex`; integration
  needs a claims.json record and a gate/manifest for the wall-loss campaign.
- The dashboard's `EXPECTED_MANIFEST_SHA256` pins the v4 bundle; the pending
  orbit_mc v1.7 audit must not rewrite `results/**` (it did not as of
  `258f69b2`).

### 2026-09-03 01:45 AEST - orbit_mc v1.7 LF sidecars + v4 posthoc audit

#### Task Summary

- Worktree `uni-project-orbit-v17` on `feat/orbit-mc-v1.7` from
  `origin/feat/sota-foundation` (`6922a3cf`). Fixed the sidecar EOL defect,
  bumped orbit_mc to 1.7.0, wrote the v4 post-hoc audit, pushed and
  fast-forwarded `feat/sota-foundation` to `258f69b2`.

#### What Worked

- [self] Proved the lint against the OLD blobs (`git show HEAD:...`) before
  trusting it: it reported exactly the two pre-fix lines
  (`orbit_mc/artifacts.py:1484`, `fields/artifacts.py:914`).
- [self] Proved "EOL is the only difference" two ways: per-file
  `sha256(lf.replace(\n, \r\n)) == recorded`, and a scratch copy with CRLF
  restored on the nine files makes the full `_inventory` equal the manifest
  and moves `validate_bundle`'s refusal to `root identity`.
- [self] Derived every audit number from the bundle (lock, transitions,
  gates, campaign-result, probability-convergence, decompressed orbits) and
  had the test re-derive them; the only external facts (duplicate execute,
  PID 484) are labelled operator-disclosed and corroborated by the
  Git-common lock file's O_EXCL creation time.

#### Mistakes And Fixes

- [self] The user's "changes <= 0.0039" is a rounding of 2/512 = 0.00390625;
  the first assertion used `<= 0.0039` literally and failed. Assert the exact
  rational and the rounded form separately.
- [self] Bumping `__version__` broke 7 frozen-v4 tests (live contract
  binding). Fixed by making them lifecycle-aware rather than editing any
  frozen file.

#### Guardrails For Next Session

- Any orbit_mc change now legitimately drifts from the v4 frozen contract;
  `tests/experiments/cft_orbit_wall_loss_v4` must stay green via the
  recorded-contract branch, never by editing protocol/authorities/shakedown.
- Do not touch `modern/experiments/cft_orbit_wall_loss_v4/results/**`
  (tree `447a5cf7`); the posthoc test asserts it unchanged.

### 2026-09-03 00:25 AEST - tests/coupling wall-clock time bomb

#### Task Summary

- 9 tests in `test_coupling_records.py` built records via
  `build_screening_proxy` without `reference_time_utc`, so `maximum_age_s`
  (86400 s) was evaluated against the wall clock relative to the fixed
  fixture `NOW`; they expired at 2026-09-02T12:00Z.

#### What Worked

- [self] Grepped `src/` for `datetime.now` first: only two freshness paths
  default to the wall clock, which bounded the audit to callers of those.
- [self] One-line fix at the alias (`partial(..., reference_time_utc=NOW)`)
  instead of editing ~20 call sites; explicit kwargs at call sites still win.
- [self] Proof of determinism: poisoned `datetime.now` in
  `coupling.validation` and `coupling.v4_records`, reran the 9 tests, green.
- [tool] `StrReplace` preserved LF on a tracked file (0 CRLF bytes after
  edit), unlike the Write tool.

#### Guardrails For Next Session

- New coupling tests: never call a builder/verifier without
  `reference_time_utc`; fixtures pin `generated_at_utc=NOW`.
- [tool] PowerShell reports git's stderr progress lines as
  `NativeCommandError` with exit code -1 even when push succeeded; confirm
  with `git ls-remote origin refs/heads/<branch>` instead of the exit code.

### 2026-09-02 23:30 AEST - Why the roadmap stalled (orbit wall-loss v1-v3)

#### Task Summary

- User asked why completion is slow; investigate and fix.

#### Findings

- Orbit wall-loss campaigns v1, v2, v3 each consumed a full preregistration
  cycle and each died on infrastructure/code, never physics:
  - v1 `prebundle_failure`: launch manifest differed from preregistered authority.
  - v2 `runtime_failure`: ordered launch/result/campaign identities inconsistent.
  - v3 `runtime_failure`: `physical event witness requires a positive step`.
- v3 root cause (integrator v1.4, commit 25dbeaaf): near the cylindrical wall
  `_first_cylinder_crossing` returns fraction 0.0 -> `step_dt = 0` -> zero
  displacement -> no event candidates -> loop spins to `max_steps` -> STEP_LIMIT
  witness with `step_dt_s == 0` rejected by `_validate_event_witness`.
- An uncommitted v1.5 fix (radial snap, tolerance-close candidates,
  zero-progress guard, `preflight_campaign`) sat untracked in the main tree
  with 102/102 tests passing but had never touched the real P2 field.
- Main tree `ahead 1, behind 13`, 84 dirty entries, 33 worktrees.
- Three dead 4-hour `pip install torch==2.9.1+cu128` retries from 2026-09-01
  were network failures; superseded by the working torch 2.13.0+cu130 venv.
- GPU 100% util / 7.7 GiB is `dwm.exe` + desktop apps, not our compute.

#### Actions

- Delegated: (A) validate v1.5 against v3's real primary-N field/launches in a
  fresh worktree `feat/orbit-mc-v1.5`, add regression test, push + fast-forward;
  (B) reconcile main tree onto origin with stash/rebase, no destructive ops.

#### Guardrails For Next Session

- Do not preregister orbit v4 until the real-field shakedown shows 512/512
  validator passes on all three maps.
- Keep orbit_mc changes on `feat/orbit-mc-v1.5` until merged; main tree copy
  is a duplicate to be deleted only when byte-identical to origin.

### 2026-09-02 05:00 AEST - Isolated ML runtime provisioning

#### Task Summary

- Diagnose and complete `.venv-sota` with current Python 3.12-compatible GPU ML packages.

#### Mistakes And Fixes

- [tool] Three repeated PyTorch commands left six orphaned launcher/child processes and only pip installed.
- Detection: process command lines, zero working sets, package inventory, and stale pip temporary directories.
- Fix: stop only matching stale installers, preserve the healthy venv, clean stale temporary entries, and use current pip with longer timeout/retry/resume settings.
- Preventive rule: after a tool timeout, inspect the process and package state before starting another installer.

#### Guardrails For Next Session

- Re-check this file before deleting or recreating the venv.
- Compare global package snapshots before handoff.

#### Follow-Ups / Risks

- [tool] The managed segmented-download wrapper was aborted at ~80%, but its
  child downloader and all part files survived. Checking process and byte
  growth before restarting avoided duplicate work; the download completed
  and passed its official SHA-256.
- [tool] Always repeat the exact Python-process audit immediately before
  handoff: the prior workflow launched six new downgrade attempts during the
  long download. Stopping all 12 launcher/child processes prevented a late
  replacement of the verified torch build.
- [tool] BoTorch's optional fused qLogEHVI extension needs `cl`; without it,
  the supported pure-Python fallback passes but is slower.
- Completed: CUDA float64, sm_120, GP posterior, constrained mixed-direction
  qLogNEHVI/qLogNParEGO optimization, deterministic pymoo NSGA-III/MOEA-D,
  package consistency, git ignore, and global-environment isolation all pass.

- [self] Book artificial relaxation terms as their own ledger: the balance then
  closes to round-off and "only the fixed point is physical" becomes a test
  (artificial ledger → 0 at the fixed point), not a disclaimer.
- [self] Ignition here is a seed-density threshold (plasma potential traps the
  beam), not a linear n_g avalanche: 5e19/1e16 seed returned 91–96 % of the beam,
  5.5e19/5e16 seed returned 60 % and ignited. Check the beam's returned fraction
  and ionisations per injected electron before turning the density knob.
- [tool] Any command that reloads a checkpoint must rebuild the config exactly as
  the run did (Poisson method is in the identity); test on a GPU-produced
  checkpoint. Emit optional config keys only when the feature is on so identities
  of runs in flight do not change.
- [tool] `!…/results/` un-ignores everything inside; re-ignore checkpoint arrays,
  series.jsonl, logs and pids and check `git status` before committing. A
  renamed `results-attempt1-…` directory is not covered by the generic rule.

### 2026-09-03 14:05 AEST - pic2d phase 4 (plateau, pair, dashboard)

- [self] Budget the resolvability on the PEAK density, not the mean: the 0-D mean
  projection landed (0.93 x) while the peak in the magnetic bottle between the last two
  cusps was 4.1 x n_max (3 lambda_D per cell). A reference-density stability gate cannot
  see this; gate on the instantaneous peak node.
- [self] A drift criterion on N_e is weak while the plasma is still slowly densifying
  (+4.98 % passed at the first eligible checkpoint with omega_pe dt still rising). Require
  two consecutive passing windows and track the peak density as well.
- [self] Physical reading of a plateau in one line: utilisation S/Q_in, beam fraction
  I_beam/I_d, wall-ion/beam ratio, and where the wall flux peaks (cusps). Those four
  numbers say more than the currents.
- [tool] Keep a finished run's protocol.json byte-frozen; put later variants in a sibling
  `variants.json` and let the dashboard fail closed when a case's recorded protocol hash
  differs from the file. Two "hash-bound" runs with one edited protocol would otherwise
  silently disagree.
- [tool] Review the headless screenshot before committing an HTML dashboard: it caught a
  seed-transient spike flattening the currents plot, log10 tick labels and a trace hidden
  under a coincident one - none of which the 9 tests could see.
- [tool] Windows PowerShell 5 `Get-Content` mangles UTF-8 markdown; append to UTF-8 docs
  from Python with LF. Here-strings/heredocs in the Shell tool fail under PowerShell:
  write a temp .py and run it.
- [tool] Node coordinates come from the config's `dr_m` (the pic2d radial grid spans the
  3 mm exit radius, not the 2 mm bore).
- [self] Fix the input representation before the model family: realised-geometry
  features cut EVERY model's pooled RMSE (GP, ridge, k-NN, trees) by 40 % on the same 16
  designs, and the GP's advantage over ridge stayed nil. A ">= 2x best baseline" gate then
  measures the model family, not usefulness - gate against the binomial floor instead.
- [self] A learning curve flat from 30 to 50 designs points at label precision (launches per
  design) as the next unit of evidence; report power-law extrapolations with that caveat and
  never as a headline.
- [tool] Dimension-dependent checks (in-sample ridge above the binomial floor) break when the
  feature count approaches the sample size; make "inputs cannot reconstruct the target" checks
  leave-one-out.
- [tool] When a dashboard shows another campaign's numbers, read them from THIS campaign's
  hash-bound comparison artifact and cross-check against the other bundle in the test.
- [tool] `Get-ChildItem` output was silently swallowed in the Shell tool after a headless
  browser run; use `cmd /c dir` before concluding the browser produced nothing. Chrome/Edge
  `--screenshot` with `--window-size=1400,9000` captures a full long page in one shot.
- [self] (topo v3, 2026-09-03 18:10) The sweep-v2 "axis cusp" QoI is the LOCAL MAXIMUM of
  |B_z| on the axis (stage centres), NOT a null; the sweep's `axis_null_positions_m` are the
  sign changes. Read the QoI extractor before reusing a name from a brief.
- [self] In an axisymmetric field the separatrix of an axis null is the g = 0 contour of the
  axis-regular flux variable g = (psi - psi_axis)/r^2 (orbit_mc `PsiBicubicField`); the wall
  cusp is therefore the root of psi(r_w, z) - psi_axis, and every simple axis null is X-type
  (J = diag(-g_z, 2 g_z)). No 2-D vector-root search is needed for PPM stacks.
- [tool] `four_cell_topology_search_v2` geometry_sha256 embeds the protocol BYTE hash through
  an EvidenceNote; on an LF checkout the rebuilt geometry hash differs from the sealed record
  (CRLF-era `ec2e9a73`) while source/material hashes match. Substitute the recorded hash
  before comparing, and say so.

### 2026-09-03 19:05 AEST - pic2d model v1.4

- [self] Utilisation is closure-dependent: with absorbing walls and no recycling, 60 % of the
  ions never return as atoms and S/Q_in overstates the atoms consumed (46 % gross ~ 36 % net).
  Quote net beside gross and name the closure; literature numbers (Brandt 2016: 24 % net) are
  model-to-model context, never validation.
- [self] Estimate a moved 0-D fixed point with one measured conductance k = S/n_g from the
  old plateau: n_g* = Q/(c + (1 - gamma f_w) k) reproduces the old plateau at gamma = 0 by
  construction (that is the algebra check), then gives 4.1e19 vs 4.49e19 for frozen S.
- [self] The axis is a trap for "peak" diagnostics: the r = 0 node volume (4.4e-14 m^3) makes
  one macro-particle read 1.4e18 m^-3 at W = 6e4. Any node-density argmax needs a particle
  floor, and the raw maximum should be kept in the record so the floor's effect is visible.
- [self] CUDA-graph capture is a refactoring discipline: list every per-step host input (RNG
  seed, injection count/carry, launch dims), move each into a device array, and make the
  captured and direct paths share one launch function; the bitwise test is then the proof.
  Warp 1.14 mempools make even `array_scan` capturable on WDDM.
- [self] A relative drift over a trailing 20 % window is bounded (0.22) for a linear ramp, so
  a "hard" drift gate above that only catches accelerating behaviour; test thresholds against
  synthetic ramps and exponentials before trusting what they can stop.
- [tool] `inf` is not canonical JSON: an undefined lambda_D leaked from a series record into
  summary.json and killed the runner suite. Use None for undefined physics, NaN only at the
  array layer; give optional record keys a read path for older artifacts.
- [tool] Contended-GPU timings are lower bounds on a speed-up ratio and useless as absolute
  numbers; report them as such and mark the clean measurement pending rather than extrapolate.
- [tool] PowerShell mangles a multi-line `git commit -F -` here-string into a pathspec; write
  the message to a temp file and `-F path`.
### 2026-09-03 21:40 AEST - L1a geometry sweep v3 (HEMP-like regime)

- [self] The single-harmonic PPM ratio I_1(x_w) is an UPPER ENVELOPE of the realised Koch
  rho, not its value: over 365 cusps rho/I_1 = 0.80 (end cusps, n = 256) and 0.87
  (interior cusps, n = 109) because the finite stack's end field raises the adjacent
  axis peaks. Preregister the analytic prediction as the hypothesis and expect the
  realised threshold to sit above it (here r_w/L 0.745 vs 0.617). Always split end vs
  interior cusps before pooling a per-cusp ratio.
- [self] Quantise every radial design length to 2^-40 m BEFORE applying geometry v1.1's
  ULP identities: 22/128 raw Sobol value sets broke `(r_w + d) - r_w == d` by rounding
  once the sum crossed 2^-8 m, and no inward-ULP walk on a derived radius can repair a
  pair of independent variables. Found by building the whole sample before the shakedown.
- [self] A 2-D Sobol projection is balanced only for t = 0 dimension pairs; test 1-D
  balance for every dimension and 2-D balance only for the first pairs.
- [tool] `experiment_runtime.validate_bundle` pins the root identity (volume/file-id/path);
  it passes only in the worktree that executed. After merging results into another
  worktree, verify bytes through the manifest (the results tests do), never `validate`.
- [tool] Headless Edge `--screenshot` captures from the top; to inspect a lower dashboard
  panel, render a %TEMP% copy with the upper sections hidden by CSS. `file://` URLs are
  refused by the Cursor browser tool.
- [self] Corner feasibility solves (six box corners) before fixing preregistered bounds
  cost 14 s and guaranteed the verbatim v2 gates could not sink the whole campaign; the
  bounds provenance must be written into the protocol so it is not mistaken for tuning.
- Scratchpad is ~1770 lines (threshold ~200): an archive-first rollover is due; not done
  here because other streams append concurrently tonight.
### 2026-09-03 21:55 AEST - pic2d model v2.0 (plume) + dashboards

- [self] A "plume box" is bounded by whatever field authority you sample: the P2 FEM
  domain ends at z = 36.25 mm, so L_plume = 0.5 L_channel was the honest size and the
  1-1.5 L_channel request became a declared deviation, not a silent shrink.
- [self] Interpolating a coarse psi grid across material interfaces is wrong in the
  plume where the yoke sits; sample the FEM solution at the PIC nodes directly and
  keep the interpolated path for the channel-only authority.
- [self] Two thrust estimates from one run (momentum flux vs Maxwell stress on the
  solid boundaries) close only if the ledger also carries the stored-momentum rate and
  the collision momentum handed to neutrals; a closure % without those terms is
  a discretisation artefact, not a conservation check.
- [self] The speckle of a log map is the counting noise of cells sampled by < ~20
  macro-particles; grey them (N^-1/2 error in the caption) and offer block means
  instead of tuning the colour scale. Record sample counts in the run artifacts so the
  mask does not have to be reconstructed.
- [self] Cold test-ion trajectories in the window-averaged field are a picture of the
  mean E field, not of tracked particles; label them as post-processing on the map.
- [tool] Chrome headless `file://` screenshots: hide the header/claim/metrics panels
  via an injected script to bring the map into a 1500x1000 frame; the in-IDE browser
  refuses `file://`.
- [tool] PowerShell has no heredoc: use `@'...'@` here-strings + `Set-Content`, and
  StrReplace for source edits; `rg` alternations containing `\"` break in PowerShell
  quoting - use the Grep tool.
- [tool] `np.load` on a `.npz` holds the file open lazily; on Windows wrap it in
  `with` before any `os.replace` on the same file.
### 2026-09-03 22:55 AEST - frame recorder / video renderer

- [self] Do not add a second accumulator for a second cadence: window sums are
  additive, so a frame is the difference of two cumulative snapshots. Difference the
  SUMS (not the finished maps) or T_e's drift correction is unrecoverable.
- [self] Make the cadence divide the checkpoint and the window, and write the frame
  before the checkpoint: resume then only has to delete frames past the checkpoint
  step - no partial-frame state in the checkpoint at all.
- [self] A launched run whose protocol is about to change is cheaper to kill during
  the 5 min factorisation than to keep: the hash would orphan its artifacts anyway.
- [tool] Pillow's fallback bitmap font has no glyphs for the en dash / micro sign: keep
  video overlays ASCII. `np.rint(126.5)` is 126 (banker's rounding) - assert it.
- [tool] `pip install imageio-ffmpeg` bundles its own ffmpeg (7.1) so it works
  without PATH; the system Gyan ffmpeg 8.1.2 accepts a rawvideo rgb24 pipe with
  `-pix_fmt yuv420p` if the frame is padded to even dimensions.

### 2026-09-03 23:55 AEST - paper: sweep v3 admission + reflection re-scoping

#### Task Summary

- Worktree `uni-project-paper-sweep3` on `paper/sweep-v3-and-twt-amendments`
  from `origin/feat/sota-foundation` (13d8ac6a). Admitted
  `l1a_geometry_sweep_v3` through `GATE-L1A-SWEEP-V3` (`accepted-screening`
  reused with justification; new manifest type
  `paper-l1a-regime-screening-manifest`), Section 14, CLM-069..076; then
  re-scoped the wall-loss zero-reflection statements (CLM-016/017/044/052,
  Sections 7/9/11, Limitations) per the TWT/PPM review bound as the
  definition source. Rebased onto 066234d9 (upstream moved), pushed,
  fast-forwarded `feat/sota-foundation` to ba852122.

#### What Worked

- [self] Reused the existing outcome value when the study is the same kind of
  object; a sixth outcome would have named nothing new. Justify on the gate.
- [self] Bound a recorded analysis that lives outside any results bundle (the
  review's committed check output JSON) as a `definition-source-*` file of the
  new manifest and derived macros from it; the checker verifies the blob, so
  the Discussion numbers are macro-bound without a new gate kind.
- [self] Made the hypothesis outcome a set of boolean macros the checker
  requires to be false and the generator refuses if true; wording enforced
  ("did not hold as preregistered", "upper envelope").
- [self] Regenerated `claims.json`/gate/manifest through an idempotent
  builder script kept in %TEMP% that extracts `authorized_tex` from the
  section/manuscript bodies, so claim text and section text cannot drift.
- [tool] Batched whole-suite runs (14 min) and the two-clean-build check
  (7 min) in background shells while editing; read `.log`/`.aux` for page
  numbers and overfull boxes instead of eyeballing.

#### Mistakes And Fixes

- [self] Assumed record fields were scalars (`hemp_like_threshold` is a dict;
  the field grid is the full 81x145 psi map, not the bore tracing grid;
  design authorities are keyed by `key`). Peek the artifact before writing an
  equality check; fix the check, never tolerate.
- [self] Started a table row with `[` after `\midrule`: booktabs swallowed it
  as an optional argument. Wrap intervals in `$...$`.
- [self] Duplicated `\label{sec:l1a-sweep-v3}` (section + subsection). The
  section file must carry a distinct label.
- [self] A protocol prose macro carried "the first 128 points" and tripped the
  unregistered-quantitative-claim scan on the generated TeX; slice the clause.
- [self] Made the WIP commit with an ad-hoc identity; amended with
  `--reset-author` so the author matches the repo config. Never pass `-c
  user.name` overrides in this repo.
- [tool] PowerShell redirected test output is UTF-16; decode by BOM before
  regex-parsing "Ran N tests".

#### Guardrails For Next Session

- When amending a Discussion claim with macros from a new manifest, add the
  manifest id AND update every admission test that pins that claim's manifest
  set (three here: geometry screening, cusp topology, four-cell closure).
- Every `check_paper.py` run now takes ~100 s (eight generators); the full
  paper suite ~14 min. Start them in the background early.
- The literature review's "0.17 pitch" class is a stated threshold, the
  realised max is 0.162; bind both (threshold constant + realised max).
- Scratchpad is ~1900 lines: archive-first rollover still due (not done here
  because other streams append concurrently tonight).

### 2026-09-04 00:20 AEST - plume attempt 3 no ignition -> field-line placement + ignition gate

- [self] When a volumetric electron source in a magnetised plume does not couple,
  trace the field lines FIRST: in an electrostatic collisionless model electrons
  never leave their flux tube. Trace the channel's tube outward from the aperture -
  it tells you where a source CAN sit. Here it closed on the front face within 1.5 mm
  of the exit (axis null at 25.45 mm); the "realistic" off-axis neutraliser annulus
  was on pole-face-to-far-field lines.
- [self] Classify tracer terminations from the LAST in-plasma point, not the first
  outside one: a plume line hitting the front face lands in a body cell with z < z_exit
  and looks like a channel-wall hit. And clamp the field interpolation to the grid box
  or the RK midpoint outside the last row reads as a magnetic null.
- [self] Calibrate a fail-closed ignition gate on the recorded series of the ignited
  AND failed runs before adopting a suggested threshold: "x3 in 0.75 us" would have
  killed the ignited v1.3 run (S ratio 1.07 at 0.75 us). Two-stage S + N_e ratios with
  ~25 % / ~15 % margins between the classes.
- [self] A protocol key consumed by the runner (require_channel_connected_fraction)
  must be filtered before `CathodeConfig(**...)`; the `_note`/`_justification` filter
  did not cover it.
- [tool] The Cursor shell swallowed the output of several `Get-ChildItem`/`rg` pipelines
  after a `cd` chain with `2>$null`; use the Glob tool or a fresh one-liner without
  the redirection.
- [git] The integration branch had moved (paper stream): cherry-pick the two commits
  onto feat/sota-foundation in the main checkout and push (ff), then merge
  origin/feat/sota-foundation back into the worktree branch - no force anywhere.
- Open: attempt 4 (PID 53756) - 20-min S check due ~00:32; gate verdicts at 0.75 us
  (~00:47) and 1.5 us (~01:23). phi_exit collapsed to ~3 V with the cathode on the
  exit tube: check the window-averaged phi against v1.x when they exist. If the gate
  trips: implement the declared keeper ignition sequence (exit-plane injection for N
  us ramped to zero, switch time recorded).

### 2026-09-04 01:35 AEST - stale MCC density in the CUDA-graph step (plume attempts 4-5)

- [self] When a rate coefficient S/(n_e n_g) jumps x100 while the T_e distribution is
  unchanged, the code is not seeing the density it reports. Compute the implied rate
  coefficient from the frames BEFORE theorising about the closure; I spent a fix on the
  neutral relaxation (kept as a guard) that the data did not support.
- [self] Every host scalar passed to a kernel inside a CUDA-graph capture is frozen at
  capture time. Audit the captured step for time-varying scalars: the emission rate was
  device-resident (done right), the MCC density was not. Make "graph-safe" a checklist
  item for any new per-interval control (density scale, alpha, anything the driver
  updates between records).
- [self] `step_graph` printed in the options line is `step_graph_active` BEFORE the first
  capture -> always false; do not read it as "graphs off". Check graph_captures > 0 or the
  backend's `step_graph` flag instead (fix the provenance next time it is touched).
- [self] A "fixed" run that reproduces the previous run's numbers step for step (attempt 5
  = attempt 4 until the crash) says the fix did not touch the mechanism - stop it early
  and look again rather than waiting for the gate.
- [git] Integration branch keeps moving (paper stream): cherry-pick each pic2d commit onto
  feat/sota-foundation in the main checkout, push (ff), merge origin back into the
  worktree branch. Three rounds tonight, no force.
- Open: attempt 6 (PID 53824) 3-min / 20-min checks; then the phi_exit ~ 0 V observation
  (cathode on the exit tube pins the exit-plane potential - compare with v1.x's Dirichlet
  exit plane once the window averages exist); the keeper ignition sequence remains the
  declared contingency if the gate trips again.

### 2026-09-04 02:30 AEST - paper: wall-loss geometry screening v2 admission

#### Task Summary

- Worktree `uni-project-paper-scr2` on `paper/screening-v2-claim` from `ba852122`.
  Admitted `orbit_wall_loss_geometry_screening_v2` through
  `GATE-WALL-LOSS-GEOMETRY-SCREENING-V2` (`accepted-screening-dataset` reused with
  justification; new manifest type `paper-orbit-cell-screening-manifest`), Section 15,
  CLM-077..085, 380 `\Wlh` macros, five tables, three disclosures verified against the
  bundle. check_paper green (107 s warm); 257 paper tests; two clean builds identical
  (65 pages, `ba7441c9...`).

#### What Worked

- [self] Treat a post-hoc event that is not an audit (runtime recovery published the
  manifest after the locked attempt died at publication) as a bound *disclosure source*
  group: the note + the recovery code + the cap + its tests at the commit that added
  them; verify the disclosure by regex against the bundle (manifest/terminal hashes,
  file/artifact/transition counts, results-commit prefix, files > cap > pin cap) and
  prove "nothing changed" with `git diff --name-only` (record commit = results/ only;
  disclosure commit = Markdown only). `posthoc_audit` stays null and the checker refuses
  one; the results-bundle block says `manifest_published_posthoc: true`.
- [self] Replay a paired control orbit by orbit from the gzipped endpoint tables (2N
  launch key -> N-step partner's termination); recompute the SE with the experiment's own
  construction (per-design +1/-1/0, `pstdev`). 1105 endpoint tables, 10 MB gz, ~6 s.
- [self] When the brief's numbers disagree with the artifacts, report the artifacts: "6
  straight-exit designs at 1.0" was 4 of 6 (one 0.992, one 0.316); the exit-side
  "direction structure 0.53-0.61 / 0.37-0.47" is really one-sided (one direction ~1, the
  other ~0; wall side = last-stage polarity in 82/90). Derived that as macros instead.
- [self] Registry files (gates/claims/contract/schemas) are hand-formatted JSON; insert
  new entries by TEXT at the right anchor (json.dumps(indent=2) blocks re-indented) so the
  diff is pure additions. My first canonical-json rewrite churned 3,600 lines; caught by
  `git show --stat` before push, restored from HEAD~1 and re-inserted (semantic equality
  asserted).

#### Mistakes And Fixes

- [self] Guessed record shapes four times (dataset `field_source` is a 5-key subset of the
  protocol's; `launch_design` has radii not per-cell counts; `allocation_replay` is the
  (checks, passed) subset; control cases have unequal/fewer strata). Peek before asserting.
- [self] Protocol prose macros ("49,152 N/2N pairs", "the same 96 fields", "8-point") and a
  table row "16,957 files, 9 transitions" tripped `find_unregistered_claims`; drop the
  prose macros, reword table rows so digits are not followed by files/fields/N/x.
- [self] Checker phrase "exhausted" vs claim "exhausts the collisionless": require the
  claim's own wording.
- [tool] PowerShell double-quoted `python -c` passes `\\\\` through, so `\\\\newcommand`
  in an inline regex matches nothing (all-empty output looked like success). Write temp
  `.py` under `%TEMP%` for any regex check.
- [tool] `unittest discover -s paper/tests -t .` fails (no `__init__.py` in paper/); use
  the documented `-s paper/tests` form.
- [tool] First `check_paper.py` in a fresh worktree: 350 s cold cache vs 107 s warm; the
  new generator itself is ~6 s. Do not optimise on the first timing.

#### Guardrails For Next Session

- A macro-only section that must say "2N" needs the time-step names as macros; refer to
  earlier campaigns by `\ref`, never by version token (v1/v4 carry digits).
- Any new admission must update the three pinned-wording tests (v1 screening, cusp
  topology, sweep v3) if it retires "planned"/"future work"/"no admitted consumer" text.
- Section 15's claim CLM-085 is the deferral of the surrogate/MDO iteration on the v2
  labels; a surrogate v3 or an MDO v3 on these labels contradicts it unless the closure
  changes (kinetic/sheath-limited).
- Scratchpad is ~2100 lines: archive-first rollover overdue; again deferred because the
  pic2d stream appends concurrently tonight.
