# Learning scratchpad

## File policy

- `COMMITTED`: this experiment-local scratchpad is part of the v4
  preregistration and is committed with it in phase 2.

## Retained lessons (carried from v1–v3)

- [self] Before any one-shot preregistered campaign, run a labelled
  NON-EVIDENTIARY shakedown of the full production path on the REAL input
  data. Synthetic preflights passed three times while the real field exposed a
  fresh code defect each time (v1 tuple-tag comparison, v2 launch-ID prefix,
  v3 zero-length step). v4 makes this a hard gate on `prepare`.
- [self] "One immutable attempt, no rerun" is only cheap if the code has
  already seen real data; each latent bug otherwise costs a full prereg cycle.
- [self] The accepted orbit event geometry is cylindrical; wall-hit authority is
  the straight dielectric only and divergent-section radial exit is escape.
- [self] Checkpoint launch IDs must be created with `ensemble_id ==
  campaign_id`; never rename after construction (seeds bind the full ID).
- [self] Never compare a raw tuple-bearing record to parsed canonical JSON, and
  never pass parsed tagged mappings back to `canonical_bytes`.
- [tool] PowerShell uses `;` for sequential commands; do not rely on `&&`.
- [tool] Orbit launch seeds are unsigned 64-bit; shared runtime JSON accepts
  signed 64-bit integers, so payload builders encode seeds as decimal strings.

## Session entry — 2026-09-03 (phase 1)

- Task: scaffold v4 from the v3 template and prove the shakedown phase on
  orbit_mc v1.5 without preregistering, committing or pushing.
- [self] The shakedown found a real defect on its first run: v3's
  `_convergence` used `zip(ordered, ordered[1:], strict=True)`, which always
  raises. v3 never reached it (it died earlier), so v3 would have failed its
  assessment even with a fixed integrator. The shakedown must exercise the
  code *after* integration (convergence, gates, export, publication), not just
  the integrator.
- [self] A sealed orbit artifact (`result-v1.5`) asserts convergence flags as
  `const true`; orbit_mc refuses to seal an unconverged campaign. For the
  shakedown the artifact stage therefore runs on evaluated shakedown-scale
  structural flags (preflight + rotation bound + zero failures across
  N/2N/4N; adapter + cross-map field convergence; backend parity) and the
  probability/energy gates stay informational. Both bases are recorded in
  `gates.artifact_convergence_flag_basis`; production keeps the binding
  campaign convergence gates.
- [self] Parallel cases are compatible with the shared runtime: record all
  nine label accesses in case order, then run a spawn pool. Determinism is
  proven, not assumed: the main process re-integrates a sample per case and
  `write_artifact` replays every orbit. Under nine concurrent workers the
  per-orbit cost rises ~30 % (132/258/488 ms at N/2N/4N vs ~100/200/376 ms
  sequential), so a full campaign projects to ~13 min wall.
- [tool] Frozen slotted dataclasses (`ElectronLaunch`, `OrbitResult`,
  `EnsembleSummary`) and `PsiBicubicField` pickle cleanly through
  `ProcessPoolExecutor(mp_context=spawn)`; keep worker functions module-level
  and guard `run.py` with `if __name__ == "__main__"`.
- [tool] `ExecutionAttestation.clean_worktree` must be exactly `True`; the
  shakedown records the real git dirtiness separately and discloses this.
- [user] Do not preregister v4 until the shakedown shows every validator
  passing for all nine cases on the real fields; energy-gate failures on v1.5
  are informational and v1.6 must land before phase 2.
- Outcome status: shakedown passed on v1.5 (9/9 cases, 289 validators, 0
  failures, `accepted_result`, bundle validated); `prepare` not run.

## Session entry — 2026-09-03 (phase 2, orbit_mc v1.6)

- Task: rebase the v4 worktree onto v1.6 + LF, adopt the v1.6 witness
  contract, re-run the shakedown, preregister, execute once, record.
- [tool] A worktree created before a repo-wide `eol=lf` pin keeps CRLF
  working copies; `git checkout` alone does not re-smudge. Delete the files
  and `git checkout -- <batch>`; verify with `git ls-files --eol`. Any
  source hash computed before that step is unreproducible from the blobs, so
  `orbit_mc_source_sha256()` now refuses CR bytes outright.
- [self] Under v1.6 the energy gate is not "approximately" met — it is exactly
  0.0 on every real-field orbit because the event velocity is a Boris state
  and the field is pure B. A gate that can be met exactly should be asserted
  exactly (`== 0.0` in the shakedown tests) so a future regression to
  chord interpolation cannot hide under a tolerance.
- [self] μ variation is physics here, not a numerical quality metric: the
  cusp field is non-adiabatic and 60 % of shakedown orbits exceed 0.1. It
  must live under a `diagnostics_not_gates` key so no reader (human or
  script) can misread it as a failed gate; only the energy gate, rotation
  bound, manufactured orders and CPU/CUDA parity establish fidelity.
- [tool] A "no μ gate" test must not match the substring `mu` — `maximum`
  contains it. Split on `_` and compare tokens.
- [user] Keep both artifact replays (write_artifact replay + verified reload);
  ~13 min wall is acceptable; never trade rigor for time.
- Outcome status: shakedown on v1.6 passed (9/9, 289/0, energy 0.0,
  576/576 velocity identity); tests green; proceeding to `prepare`.

## Phase-2 checklist (orbit_mc v1.6 rebase) — DONE

- [x] `orbit_mc_contract` binds `package_version`, result/checkpoint/
  validation-protocol `1.6.0`, handoff `1.3.0`.
- [x] Synthetic vector matrix carries the three v1.6 witness vectors and the
  zero failure-witness block.
- [x] `_result_gate_report` requires `final_velocity == event_velocity`
  exactly; energy gate binding; μ diagnostic only.
- [x] Shakedown re-run on v1.6 with LF hashes; energy gate passes exactly.
