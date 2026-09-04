# NOTE for the v4-fast (solver qualification) coordinator — acceptance (b) must be read on the corrected ledger

Written 2026-09-05 (AEST) by the corrected-ledger re-read agent. `pic2d_cft_steady_state_v4_fast` was **not on origin**
when this note was written (it exists only in a local worktree, preregistered at a local commit); its sealed protocol is
therefore **not modified here** and this note is left in the comparison target's directory instead. Fold its content into
the v4-fast README (a "corrected-ledger" note, not a protocol change) when that campaign lands on origin; if the protocol is
already sealed, do not touch it — the assess stage can read the sidecar without changing the frozen rules.

## Facts the v4-fast assessment must carry

1. **Model v2.0.6 (commit `4b53012d`, spec `spec/pic2d/pic2d-model-v2.0.json#gates_v2_0.energy_ledger_correction_v2_0_6`).**
   Up to v2.0.5 the ledger's `inelastic_loss_j` lacked the macro weight W, so every recorded
   `grid_heating_triad.windowed_energy_residual_over_electrode_work` read `H − L_inel`: biased NEGATIVE by the inelastic
   power (≈ −10 % of the electrode work on the 33 µm channel plateau). The v4-fast preregistration (local commit `cfa9c014`,
   parent `e1a24aec`) **predates v2.0.6**: its run executes pre-v2.0.6 code in its locked worktree, so its recorded (b)
   statistic will be biased the same way (a recorded reading near −7 … −8 % is the omitted inelastic power, not
   conservation).
2. **The comparison target failed (b) on the corrected ledger.** `pic2d_cft_steady_state_v4/results/ledger-corrected.json`
   (commit `02013df0`) and `results/assessment-corrected-ledger.json` (this re-read): v4's trailing-400 000-step residual /
   electrode work at the stop is **+2.46 %** corrected (recorded −7.67 %), so the predeclared v4 acceptance (b) "< +2 %"
   changes status **PASS → FAIL**; the predeclared (d) tree with the corrected (b) gives `refinement_heating`; the recorded
   verdict `resolution_limited` stands as recorded. Verdict wording: *plateau reached; convergence vs 50 µm as recorded
   (resolution_limited for 50 µm); residual precondition (b) FAILED on the corrected ledger → the 33 µm plateau is itself
   heating at +2.5 % of electrode work and is NOT a clean reference; 25 µm (v5) pending.*
3. **The 5 % hard gate and the 2 % acceptance bound are KEPT** (`gate_recalibration_v2_0_6`): (b) is not loosened to rescue
   v4 and must not be loosened for v4-fast.

## What to do at v4-fast assess time (no protocol change needed)

* Run `python -m cft_revival.pic2d.ledger_recompute <v4_fast results dir>` immediately after the run finalises (it writes
  the sidecar `ledger-corrected.json` + `.sha256.json` from the recorded `series.npz`; recorded files are untouched).
* Evaluate **(b) on the corrected statistic** (`end_state_window.corrected_ratio`) and record BOTH readings in
  `assessment.json` (recorded / corrected / `passed` on the corrected statistic / basis), as `pic2d_design_mini_sweep_v1/run.py
  assess_run` now does for the sweep runs (see `b_residual_power.recorded` / `.corrected` / `.basis` there) and as
  `pic2d_cft_steady_state_v4/assess_corrected_ledger.py` does for v4.
* `b_residual_delta_vs_v4` (reported, not judged, per the v4-fast protocol) must be computed **corrected-vs-corrected**
  (v4 corrected = +2.46 %); a recorded-vs-recorded delta is approximately unbiased only because both runs omit W the same
  way, but its absolute level is meaningless.
* Expect the fast replay to read ≈ +2.5 % corrected as well (a same-seed replay of a heating plateau heats the same way):
  that would be (b) **FAIL** for v4-fast under its own predeclared bound — the qualification verdict logic ("qualified"
  requires (b)) then cannot return `qualified` even if the replay tolerances (c) and the multigrid contract (d) hold. Report
  that honestly as the (e) outcome the tree gives and add the reading "(c) and (d) held; (b) fails on the corrected ledger
  exactly as the target does" — the *solver* qualification question ((c) within the seed-b band, (d) contract never missed)
  is answered by (c)+(d); the residual is a scheme property the fast run shares with v4, not a solver property. Do not
  redefine (e) after the fact; state what the predeclared tree gives and what the numbers show, both.
* Any quoted v4-fast plateau value inherits v4's disclosure: 33 µm heats at +2.5 % of the electrode work; not converged;
  not energy-conserving; 25 µm (v5) pending.

## Cross-references

* `modern/experiments/pic2d_cft_steady_state_v4/README.md` — "Corrected-ledger re-read of the preregistered acceptance".
* `modern/experiments/pic2d_cft_steady_state_v4/results/assessment-corrected-ledger.json` (+ `.sha256.json`).
* `modern/visualization/pic2d-cft-steady-state-v4.html` — recorded and corrected readings side by side.
* `modern/docs/pic2d-performance-audit.md` §13; `spec/pic2d/pic2d-model-v2.0.json` v2.0.6 entries.
