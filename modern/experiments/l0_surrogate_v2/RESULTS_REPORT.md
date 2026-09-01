# L0 surrogate experiment v2 — immutable execution report

## Outcome

V2 is **not accepted**. Its single preregistered execution terminated before
calibration or final assessment, so no RMSE, worst-error, coverage, or gate
metrics exist for any replicate or baseline.

This is an execution failure, not a surrogate-quality gate failure. It must not
be represented as evidence for or against L0 emulation accuracy.

## Frozen sequence

1. Preregistration commit
   `9c93d004e62229a88cedfe8ad782b17acc432199` was pushed before execution.
2. The execution lock recorded that SHA, predeclaration hash
   `640ec66125d4de07944b012402a64e6cd7be012f1b6877b166b12c4271ff15cf`,
   and partition hash
   `a19d317e148ff0318c38095123a1dad4c8f850833f7df32a821f3f7ad0c91897`.
3. First-replicate active acquisition completed exactly 96 rows and froze
   selection hash
   `c11f6a3f6662019c0ecabfea32c5bcc7feab1d25bbfba0ef7c3a8f17ea366f5f`.
4. Serialization then raised `FileNotFoundError` because the preregistered
   `_save_models` function had not created
   `results/group-split-2026090201/active/`.
5. Assessment-loader invocation count remained zero. No calibration or
   assessment labels were loaded.

The execution lock, partial raw selection, failure manifest, and diagnostic
runtime are preserved unchanged. The protocol was not patched and the campaign
was not rerun.

## Requested metric matrix

Per-replicate active and baseline metrics are unavailable:

- `group-split-2026090201`: not evaluated;
- `group-split-2026090202`: not evaluated;
- `group-split-2026090203`: not evaluated.

All scientific gate statuses are `not-evaluated`; overall acceptance is false.
The calibration-label perturbation invariance test passed before execution, but
the full three-replicate selected-index comparison could not be produced.

## Integrity and next step

Failure manifest:
`cc0e608928e6c000c059bcc01f23a288edf9ff99d8736c3c212e70a0c2f1d63d`.
This hash establishes artifact integrity only.

The next scientifically valid action is a separately preregistered successor
experiment. It must create model directories and exercise the complete
selection → model save/reload → calibration save → frozen marker → one-shot
assessment writing path in a disposable test directory before its freeze
commit. V1 remains failed development evidence; v2 remains failed execution
evidence. Neither establishes physical accuracy.
