# L1a field-surrogate v2 prospective result

- Preregistration: `373ffae8d8d08affc9d33401378a81c35dbdf964`
- Device: NVIDIA GeForce RTX 5090 (`cuda:0`)
- Raw input geometry: 254 valid, 2 rejected, 25 constructor-corrected
- Frozen execution: 112/112 coarse and 80/80 fine completed
- Numerical gates: all passed
- Model fits: 10 family/budget fits; 12 combined candidates evaluated
- Terminal status: **failed development selection gates**
- Calibration access: 0
- Assessment access: 0
- Patch/rerun: none after the lock-claimed execution

The best scalar development result was the budget-32 AR1 family, with 35.99%
worst scalar NRMSE and 78.70% worst range-normalized error, versus limits of 5%
and 15%. The best field result was budget-32 coarse-observed discrepancy POD,
with 41.93% worst relative L2 and 51.19% worst relative energy error, versus
5% limits; 11/16 topology comparisons matched.

Because no candidate passed development gates, model selection remained null.
Group conformal calibration, single-use assessment, OOD decision acceptance,
assessment coverage, representative predictions, and end-to-end latency were
not evaluated. Overall acceptance is false, while the result remains a valid
prospective development rejection.

This experiment emulates only the accepted L1a numerical discretization. It
provides no material, plasma, propulsion, thermal, structural, hardware, or
physical-accuracy evidence.
