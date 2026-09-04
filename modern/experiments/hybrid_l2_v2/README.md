# Hybrid L2 v2 - per-cell hybrid on the reference material-aware field vs the PIC base plateau

Model, cell concept, closures, gates and claim boundary: `modern/docs/hybrid-l2-v2.md`.
Protocol: `protocol.json` (frozen at the preregistration commit; `preflight` and `assess` re-derive the
closures and the PIC reference table from the PIC artifacts and refuse on any disagreement).

## Stages

From `modern/` with `$env:PYTHONPATH="$PWD\src;$PWD"` (CPU only; `$env:OPENBLAS_NUM_THREADS="4"` avoids
BLAS oversubscription when several cases run side by side):

```
python -m experiments.hybrid_l2_v2.run preflight              # real field on the three grids, partition check, closures, ms/step
python -m experiments.hybrid_l2_v2.run shakedown              # synthetic full path + real short run, through finalize + assess
python -m experiments.hybrid_l2_v2.run launch --case base --expect-commit <prereg sha>
python -m experiments.hybrid_l2_v2.run launch --case <spatial-coarse|spatial-fine|temporal-coarse|temporal-fine|weight-half|seed-b|closure-g-low|closure-g-high|closure-w-low|closure-w-high> --expect-commit <sha>
python -m experiments.hybrid_l2_v2.run status
python -m experiments.hybrid_l2_v2.run assess                 # GATE-L2 metrics over every finished case -> results/assessment.json
```

Each case writes `results/` (base) or `results-<case>/`: `series.jsonl` (untracked), `status.jsonl`,
`checkpoint-latest.*` (untracked), and at the end `maps.npz`, `series.npz`, `summary.json`,
`l2-targets.json` (the mini-sweep extraction applied to L2's own maps), `checkpoint-final.*`
(the `.npz` particle arrays untracked, the metadata and the field anchor tracked).

## Cases

| case | grid | dt | W | seed | closure scale | role |
|---|---|---|---|---|---|---|
| base | 60 x 480 (50 um) | 1 ns | 3e5 | 20260903 | 1 | headline comparison |
| spatial-coarse / spatial-fine | 30 x 240 / 90 x 720 | 1 ns | 3e5 | 20260903 | 1 | spatial levels |
| temporal-coarse / temporal-fine | 60 x 480 | 2 / 0.5 ns | 3e5 | 20260903 | 1 | temporal levels |
| weight-half / seed-b | 60 x 480 | 1 ns | 1.5e5 / 3e5 | 20260903 / 20260904 | 1 | statistical levels |
| closure-g-low/high, closure-w-low/high | 60 x 480 | 1 ns | 3e5 | 20260903 | G or w x 0.7 / 1.3 | input uncertainty |

## Launch log

(filled at launch; results-only commits add the case directories)
