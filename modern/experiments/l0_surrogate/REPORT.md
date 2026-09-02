# L0 surrogate-improvement experiment

## Scope

This experiment measures emulation of the hash-pinned, deterministic 8,192-point
L0 software sweep. L0 is numerical truth only for this experiment. These results
do **not** establish physical, experimental, L1, L1b, L2, or L3 accuracy.

The predeclared campaign hash is
`456431bb8eda55d9f6e854ca7a1a5209545dbcb65b50f68b0223a6394fb1a620`.
It fixed the split, 32-row initial design, 16-row acquisition batches, 96-row
maximum exact-GP budget, ARD Matérn-5/2 model, calibration policy, and all gates
before execution.

## Result

The active campaign exhausted its maximum budget at 96 training rows. It failed
the complete gate because calibrated 90% interval coverage remained below the
predeclared `[0.85, 0.95]` acceptance interval:

- thrust: 1.830% range-normalized RMSE, 0.624 coverage, 0.002927 N worst error;
- Isp: 2.109% range-normalized RMSE, 0.775 coverage, 137.895 s worst error.

Both outputs passed the `<=5%` RMSE gate and the `<=15%` range-normalized
worst-error gate. Coverage alone prevented acceptance. The assessment contained
178 points (64 interpolation, 62 boundary, 52 intentionally withheld OOD-corner
points), and calibration used a separate 61-point whole-group set.

The OOD-stratum RMSEs were 2.359% for thrust and 2.985% for Isp. The accepted
nearest-distance detector labelled 2/52 OOD-stratum points as OOD at the final
fit; this low detector sensitivity is reported, not reclassified after the run.

## Fixed-sequence comparison

At the same 96-row budget, the fixed Halton-like L0 sequence produced:

- thrust: 4.346% range-normalized RMSE, 0.652 coverage, 0.006397 N worst error;
- Isp: 6.034% range-normalized RMSE, 0.702 coverage, 443.604 s worst error.

Active selection therefore improved point accuracy and worst-case error, while
not solving calibration. Complete checkpoint curves and every selected index
are in `artifacts/campaign.json`.

## Reproduction and identities

From `modern/`:

```powershell
$env:PYTHONPATH="$PWD\src"
python experiments/l0_surrogate/campaign.py
python -m pytest -q tests/experiments/l0_surrogate
```

- source config: `1434eac84d954a4dfbf8e7a9621b16c5c645f1490d56ad23d8e1d979a2bbf2e4`
- dataset: `b25d249b8f4741086de6d45133682222386e197860d8ff5ac721c7d2e2264bfb`
- partition: `6865a3a9739749d4ace7a3071e22a24b9598c5322e13de64f9e86d5ee5b7a727`
- benchmark: `fbf77634d8078df1c88f1dcf145fb0daf106933d39af841602acd1317c1a6157`
- campaign: `d78723af909e1160031f3b5eaa8be01a4aedc64035b15543e1085e240f35dc47`
- thrust model: `33c50cb8b797561ef759f16c3cdba16236ec089ca4f12405127639b0069b884a`
- Isp model: `08a8933abc9990e227b6da50390042a166e574b43e8e1e819856ae95ab12e415`

The recorded 50.25 s campaign time and 0.70 s truth-generation time are
uncontrolled diagnostics and are excluded from deterministic hashes.

## Next step for L1/L1b

Use the same immutable campaign protocol with geometry/device-grouped splits,
a fixed-mesh hash, and POD field coefficients rather than scalar L0 outputs.
Acquire paired L1a/L1b cases using labelled discrepancy uncertainty, reserve
separate calibration and OOD geometry families, and require field-norm,
pointwise-extreme, conservation, and interval-coverage gates before promotion.
Do not use this L0 result as evidence that an L1/L1b field surrogate is accurate.
