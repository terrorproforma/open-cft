# L0 surrogate v6 results

## Status

V6 is **accepted** under every preregistered method-selection and final
assessment gate. This is prospective same-domain L0 software-emulation
validation of a protocol changed using V5 development evidence. It is not an
ahistorical independent confirmation and makes no physical-accuracy claim.

The first process launch was rejected at the pre-execution identity barrier
because the editable Python installation resolved accepted modules from the
main workspace. It acquired no lock, evaluated no physics labels and created
no results. The validated invocation explicitly bound `PYTHONPATH` to the
detached worktree; its actual module paths and Git blobs matched, then the
one-shot execution lock was acquired and retained.

## Frozen selection

All three candidate budgets passed the method-group point and stability gates:

- 96 rows: overall thrust NRMSE/worst `1.968% / 10.957%`; Isp `2.497% / 11.721%`
- 128 rows: overall thrust `1.762% / 10.333%`; Isp `2.012% / 9.299%`
- 160 rows: overall thrust `1.626% / 7.233%`; Isp `1.831% / 7.932%`

The preregistered smallest-passing rule therefore selected the 96-row accepted
ARD Matérn-5/2 GP.

Selected intervals and equal-group LOGO coverage/stability:

- Interpolation thrust: asymmetric signed absolute, `0.905 / 0.198`
- Interpolation Isp: symmetric raw-GP-SD, `0.900 / 0.141`
- Boundary thrust: symmetric input distance, `0.899 / 0.170`
- Boundary Isp: symmetric input distance, `0.901 / 0.173`
- OOD thrust: symmetric raw-GP-SD, `0.900 / 0.156`
- OOD Isp: symmetric raw-GP-SD, `0.902 / 0.148`

The selected model/method hash was frozen before final-calibration labels were
opened. Cluster-weighted grouped conformal parameters were then frozen before
the global assessment was loaded exactly once.

## Global assessment

Metrics are `range-normalized RMSE / worst normalized error / coverage`.

- Interpolation, 192 rows:
  - thrust `1.075% / 3.061% / 0.9063`
  - Isp `1.328% / 3.277% / 0.9219`
- Boundary, 198 rows:
  - thrust `2.134% / 6.226% / 0.9343`
  - Isp `2.160% / 5.064% / 0.9040`
- OOD, 196 rows:
  - thrust `2.922% / 9.788% / 0.8980`
  - Isp `3.680% / 7.811% / 0.8980`
- Overall, 586 rows:
  - thrust `2.185% / 9.788% / 0.9130`
  - Isp `2.586% / 7.811% / 0.9078`

Every NRMSE, worst-error and coverage result passed. Assessment-coordinate
intersection with all V3–V5 held-out sets was zero. Coarse spatial-domain
overlap remains intentional because this is same-domain prospective
validation.
