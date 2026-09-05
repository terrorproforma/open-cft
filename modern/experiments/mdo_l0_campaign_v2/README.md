# MDO L0 campaign v2: the corrected L0 model over the screened design catalogue

Classification (verbatim, everywhere):
`l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance`

This campaign makes **no thruster-performance claim**. It optimises the corrected L0
conservation model (`cft_revival.physics`) over a **discrete catalogue** of the 96
accepted sweep-v2 designs screened by `orbit_wall_loss_geometry_screening_v1`
(classification `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`) times the continuous
operating point (Ua, Ia, mdot) of v1. Each design's four per-cell wall-hit
probabilities enter the L0 model **directly** from the screening counts (Jeffreys Beta
posterior of 128 launches per cell, frozen 64-row QMC sample per design); **no surrogate**
is used (both surrogates over the dataset were rejected on predeclared gates).

## Closure and claim boundary

* **CL-1 (campaign)** `S = prod_k (1 - p_k)` with `p_k := P(wall | launch cell k, design)`.
  This *identifies* a collisionless test-particle wall-hit probability with the closure's
  per-cusp survival factor. The v1 scenario analysis showed that quantity is **not** the
  Kornfeld per-cusp probability of a sustained discharge. A design that "wins" here wins
  **under this declared closure only**.
* **CL-2 (sensitivity)** `S = 1 - p_pooled` (512 launches). The pooled designs are
  re-evaluated under CL-2 and the two fronts are compared (reported, not gated).
* The corrected four-cell solver has no admissible root for interior cusp probabilities
  (`docs/workstreams/global-plasma-closure-analysis.md`, `266d8a99`); it is not used.

## What changed against v1 (and which v1 audit disclosures this closes)

| v1 audit | v2 |
| --- | --- |
| F9 non-results files in the result commit | `execution.result_commit_policy`: result commit = `results/` only; the spec pointer follows separately |
| F10 hash scope inert/incomplete | explicit import-bound file scope; binding gate `code_hash_scope_matches_imports`; import-trace test in a fresh interpreter |
| F22 rounded scenario probabilities | no rounded probability anywhere; posterior means computed from counts by rule |
| F26 integrity-only gates unlabelled | `gates.semantics` says so; every efficacy statement carries its seed count |
| F27 NSGA-III duplicates | `MixedVariableDuplicateElimination` + binding gate `nsga3_duplicates_eliminated` |
| F28 wrong descriptive labels | labels generated from the arguments / fitted objects + binding gate `labels_consistent`; `declared_generations` and `pymoo_n_gen` recorded separately |

## Layout

* `protocol.json` -- frozen protocol (authority for every declaration).
* `catalogue.py` -- dataset binding (bytes, Git blobs, manifest entry, ancestry), the 96
  design rows, the pure-Python incomplete beta / quantile, the frozen per-design sample.
* `model.py` -- evaluation chain (closures, L0 operating point, CVaR, dominance, hypervolume).
* `optimizers.py` -- LHS over the catalogue, pymoo mixed-variable NSGA-III, BoTorch
  qLogNEHVI over `MixedSingleTaskGP` models (exhaustive categorical candidate stage +
  continuous refinement).
* `experiment.py` -- plans, contract, parallel dense reference, gates, sensitivity, callbacks.
* `run.py` -- `shakedown` / `prepare` / `execute` / `validate` / `contract` / `imports`.
* `shakedown.json`, `authorities.json` -- frozen at preregistration.
* `results/` -- the single recorded execution (immutable).

## Lifecycle

```powershell
cd modern; $env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
$py = "..\.venv-sota\Scripts\python.exe"
& $py -m experiments.mdo_l0_campaign_v2.run shakedown     # non-evidentiary, temp result root
& $py -m experiments.mdo_l0_campaign_v2.run prepare       # freezes authorities.json
# commit "preregister MDO L0 campaign v2", push, then from a clean detached worktree:
& $py -m experiments.mdo_l0_campaign_v2.run execute
& $py -m experiments.mdo_l0_campaign_v2.run validate
```

Tests: `tests/experiments/mdo_l0_campaign_v2` (system Python runs the torch-free subset).
Dashboard: `visualization/mdo-l0-campaign-v2.html` (after the record).

## Post-hoc audit notes

### Import scope grew after the execution (disclosed here, never resealed)

The sealed `code_contract.source_hash_scope` (28 files; `protocol.json`, frozen by its semantic
hash in `authorities.json`) was exact when the campaign was preregistered and executed at
`99914dc2`: the recorded binding gate `code_hash_scope_matches_imports` in
`results/artifacts/gates.json` reads `imported_not_in_scope: []`, `in_scope_not_imported: []`.
At `bb756418` (2026-09-03, after this record) the shared runtime gained
`modern/src/cft_revival/experiment_runtime/recovery.py` — the fail-closed manifest recovery for the
geometry-screening-v2 EMFILE publication failure — and `experiment_runtime/__init__.py` re-exports
it, so a fresh-interpreter import trace of this campaign now lists that one file outside the sealed
scope. Nothing in the protocol, the authorities or the bundle was edited; the sealed hashes are
unchanged. The import-trace test (`test_import_trace_in_a_fresh_interpreter_equals_the_hash_scope`)
binds the growth to this note: every file it finds outside the scope must have **no blob at the
execution commit** (post-hoc growth, not a sealing omission) and must be named here. Any further
growth fails the test until it is disclosed in this section.
