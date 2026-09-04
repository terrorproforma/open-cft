# NOTE for the v4-fast (solver qualification) coordinator — the comparison target failed (b) on the corrected ledger

Written 2026-09-05 (AEST) by the corrected-ledger re-read agent; **updated after the rebase**: while this re-read was
being prepared, `pic2d_cft_steady_state_v4_fast` landed on origin (preregistered at `b09f2b71`, jobs.yaml `6807f041`,
launch 1 logged at `02aef43d`, PID 44430 on the Lambda H100, 16:28:46 UTC 2026-09-04). Its sealed `protocol.json` is
**not modified** by this note or by any commit of this re-read; a short cross-reference paragraph was added to its README
only (a documentary note, not a protocol change).

## What the v4-fast preregistration already does right (verified on origin `b09f2b71`)

* It executes **model v2.0.6 code**, so its recorded `grid_heating_triad.windowed_energy_residual_over_electrode_work`
  is the corrected statistic natively (no post-hoc sidecar is needed to read it; `ledger_recompute` on its results must
  report "already W-scaled record: corrected == recorded" and is the cheap cross-check to run at assess time).
* Its acceptance **(b) is a replay criterion**: |corrected windowed residual − (+2.46 %)| ≤ 1 pp against
  `reference_run.quantities.windowed_residual_over_electrode_work_corrected_v2_0_6` = the v4 sidecar's end value
  (`ledger-corrected.json`, `02013df0`), and the project's plateau acceptance "< +2 %" is **reported, not judged**
  (`project_acceptance_b_below_0p02`), with the statement that v4 itself fails it at +2.46 %.

## What this re-read adds for the v4-fast assessment (documentary; no protocol change)

1. The comparison target now has a **committed post-hoc re-read**:
   `pic2d_cft_steady_state_v4/results/assessment-corrected-ledger.json` (+ `.sha256.json`, hash-bound to the sidecar, the
   recorded assessment, the summary and the protocol). Recorded verdict `resolution_limited` stands as recorded; on the
   corrected ledger v4's (b) **FAILS at +2.46 %** and the predeclared (d) tree gives `refinement_heating`. Verdict wording:
   *plateau reached; convergence vs 50 µm as recorded (resolution_limited for 50 µm); residual precondition (b) FAILED on
   the corrected ledger → the 33 µm plateau is itself heating at +2.5 % of electrode work and is NOT a clean reference;
   25 µm (v5) pending.* Cite this file (not only the sidecar) when the v4-fast assessment names its target.
2. **Every quoted v4-fast plateau value inherits v4's disclosure**: the 33 µm plateau heats at +2.5 % of the electrode
   work; not converged; not energy-conserving; 25 µm (v5) pending. A `qualified` verdict qualifies the *solver* (device-mg
   + K = 5 reproduce v4 within the seed-b band); it does not upgrade the 33 µm plateau to a clean reference.
3. **Bounds are kept**: the 5 % hard gate and the project's 2 % acceptance bound are not loosened
   (`gate_recalibration_v2_0_6`); the v4-fast ±1 pp replay band is a *different* predeclared criterion (a replay
   criterion), which is fine as long as `project_acceptance_b_below_0p02` is reported beside it as the protocol says.
4. At assess time, run `python -m cft_revival.pic2d.ledger_recompute <v4_fast results dir> --dry-run` and record its
   "already W-scaled" verdict in the assessment as the ledger cross-check; report the corrected residual trajectories of
   both runs at the checkpoints (the `heating` outcome text asks for exactly that).

## Cross-references

* `modern/experiments/pic2d_cft_steady_state_v4/README.md` — "Corrected-ledger re-read of the preregistered acceptance".
* `modern/experiments/pic2d_cft_steady_state_v4/results/assessment-corrected-ledger.json` (+ `.sha256.json`).
* `modern/experiments/pic2d_cft_steady_state_v4_fast/README.md` — acceptance (b) paragraph and the corrected-ledger note.
* `modern/visualization/pic2d-cft-steady-state-v4.html` — recorded and corrected readings side by side.
* `modern/docs/pic2d-performance-audit.md` §13; `spec/pic2d/pic2d-model-v2.0.json` v2.0.6 entries.
