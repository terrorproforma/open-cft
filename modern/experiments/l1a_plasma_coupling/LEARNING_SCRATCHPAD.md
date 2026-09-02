# L1a plasma-coupling learning scratchpad

Policy: `COMMITTED` experiment-local record. Shared memory and accepted
workstream files remain untouched because this task owns only new experiment
paths.

## 2026-09-02 session

- [user] Use only accepted L1a artifact bytes/manifest and coupling v2 evidence;
  never create experiment-declared field maps.
- [user] Preserve every plasma residual, rank/conditioning, conservation and
  branch outcome, and never publish a state after tolerance failure.
- [tool] This Windows PowerShell version rejects `&&`; use separate statements.
- [self] A segment count of four is not sufficient topology compatibility.
  The triplet reaches four only by classifying finite Dirichlet map-edge zeros
  as apparent cusps; the undefined end mirror ratios must fail the global-model
  gate.
- [self] Compact topology fails earlier at coupling v2 because the accepted
  requested wall field does not satisfy the low/high magnetic-mirror ordering.
- [self] Keep legacy DM9.2 probabilities descriptive. They have no honest
  index alignment when the accepted field topology is incompatible.
- [self] The full suite's only behavioral failure was concurrent material-field
  artifact/schema drift; focused coupling/plasma/physics tests remained green.

## 2026-09-03 session (serialization v1.2 repair)

- [user] Load accepted artifacts through the current public v1.2 loader, not a
  hand-parsed copy of the old layout; never edit accepted packages, specs,
  examples or the other agent's orbit_mc working-tree files.
- [user] A frozen hash is either a provenance pin (update and record old->new
  with the reason) or a physics invariant (must still match); never paper over
  real numeric change.
- [tool] `git ls-files --eol` distinguishes index from worktree line endings;
  `core.autocrlf=true` at system scope silently converts hash-bound JSON and
  `.sha256` files on checkout. Writing `git cat-file blob HEAD:<path>` bytes
  back and `git update-index --refresh` restores exact bytes with an empty
  diff. `git -c core.autocrlf=false checkout-index -f` did not rewrite files
  git already considered up to date.
- [tool] PowerShell `python - @'...'@` here-strings do not feed stdin; write
  temp scripts under `C:\tmp` and run them by path. A fresh temp file may be
  invisible for a moment after the write tool returns; re-check before use.
- [self] "Regex did not match" on the first failing test was a symptom two
  layers above the cause: the sidecar gate failed before the schema pin was
  ever reached. Reproduce, then diff bytes, before touching the pin.
- [self] The v1.1->v1.2 migration is signed-zero normalization only: every
  finite value in all three artifacts is bit-identical; only axis-row
  `b_r_t` `-0.0` entries flipped. Coupling v2 `hash_axisymmetric_map` packs
  IEEE bits, so `field_map_hash` changes even though `-0.0 == 0.0` and all
  derived topology, mirror ratios and probabilities are identical. Compare the
  physics blocks and the identity blocks separately or the diff lies to you.
- [self] Compare against the pre-migration commit (`dbcab646~1`), not the
  commit that recorded the experiment: the migration was already an ancestor
  of `00cf29fc`, so that diff was empty.
- [self] Third cause hidden behind the first two: `cft_revival.coupling`
  re-pointed `build_coupling_record`/`coupling_record_dict`/
  `global_solver_inputs` at coupling v4 and demoted the v2 same-z path to the
  deprecated `build_screening_proxy`. Name the deprecation in the dataset and
  report instead of filtering the warning away.
- [self] The coupling v2 `MapValidationPolicy` default still names schema
  v1.1; an experiment that only accepts v1.2 must pin that field itself, and
  should reject any other explicit schema rather than silently rewrite it.
- [self] `tests/coupling/test_coupling_records.py` is a wall-clock time bomb
  (fixed `NOW` + default 86 400 s age gate, `reference_time_utc=None`); it
  started failing at 2026-09-02T12:00Z and is unrelated to this experiment.
