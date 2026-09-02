# L1a field-surrogate v4 immutable result

## Status

`failed-execution` — valid prospective software-execution failure.

- Preregistration: `99092987e827862fe325f905668a070dfb51ab37`
- Base: `7e1246b5b76830fe09afb10ffe076e953a3c2905`
- Exclusive lock: claimed for the preregistration and retained
- Run attempts: exactly one
- Patch/rerun after lock: none

## Failure

After the clean detached checkout passed its tests and the runner verified the
pushed preregistration, `execute()` claimed the exclusive lock. It then failed
at `run.py:979`:

```text
FileNotFoundError: [WinError 3] The system cannot find the path specified:
'...\modern\experiments\l1a_field_surrogate_v4\results\.working'
```

The preregistration did not create the `results/` parent directory, and
`cache.mkdir()` did not request `parents=True`. This statement is diagnosis
only; the sealed code was not repaired and the experiment was not rerun.

The failure occurred before the runner's result-bundle `try/finally` region.
Consequently, no standard provenance closure or phase record was written.
`prebundle-failure.json` records the exact observable zero-access state instead
of presenting a fabricated standard bundle.

## Access and cleanup evidence

All candidate, method, calibration and assessment counters are zero:

- low solver accesses/completions: 0
- fine solver accesses/completions: 0
- checkpoint reads: 0
- label reads: 0
- model fits: 0

No `.working` directory was created, so no ignored label cache was retained and
no manual deletion was needed.

## Preflight

Before preregistration, the deterministic production-code synthetic preflight
passed all 11 required path groups with 554 recorded path executions and source
hashes. Static unresolved-name coverage passed with no unresolved globals.
Geometry preprocessing produced 510 valid raw rows, 2 input-only rejections,
63 corrected rows and 240 hash-identical frozen rebuilds. The synthetic
preflight made zero real field-solver or real-label accesses.

## Scientific gates

No model was fitted or selected. Development, calibration, assessment,
topology, coverage and latency gates were not evaluated. No scientific
acceptance claim is made.
