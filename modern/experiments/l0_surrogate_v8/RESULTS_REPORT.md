# L0 surrogate v8 results

## Outcome

V8 is a valid prospective **failed-development-selection-gates** result. The
single detached execution was bound to preregistration commit
`2e16e124e53c0d8644cf76648dff0cff28680eb0`. No candidate passed every
method-role gate, so the frozen protocol stopped before final-calibration labels
or assessment labels were accessed. There is consequently no final assessment
coverage, width, or OOD metric to report.

This is same-domain deterministic L0 software-emulation evidence only. The
assessment coordinates were fresh and exactly disjoint from prior V3–V7
assessment coordinates, while coarse spatial-domain overlap was intentional and
disclosed.

## Candidate selection

No model, feature family, or budget was selected. The strongest candidate was
the raw-input ARD Matérn-5/2 GP at 224 rows:

- interpolation NRMSE/worst: thrust 0.73%/1.97%, Isp 0.67%/2.14%;
- boundary NRMSE/worst: thrust 2.92%/15.41%, Isp 3.53%/14.60%;
- OOD NRMSE/worst: thrust 5.33%/24.82%, Isp 4.21%/14.76%;
- overall NRMSE/worst: thrust 3.53%/24.82%, Isp 3.20%/14.76%.

Thus thrust narrowly failed the 15% boundary worst-error gate and materially
failed both 5% OOD NRMSE and 15% OOD worst-error gates. The preregistered
physics-informed feature family was worse at every budget; at 224 rows its OOD
NRMSE/worst values were 7.83%/41.78% for thrust and 7.52%/31.40% for Isp.

## Method-role group validity and efficiency

For the raw 224-row candidate, every chosen interval family passed method-role
simultaneous-group coverage, row-lower, stability, median-width and p90-width
gates. Leave-one-group-out simultaneous coverage was 90.5% for interpolation
and boundary and 90.9% for OOD. Equal-group mean row coverage ranged from 97.0%
to 98.4%; values above 95% are diagnostic, not failures.

Selected-family normalized median/p90 full widths were:

- interpolation: thrust 3.30%/4.47%, Isp 2.91%/3.92%;
- boundary: thrust 12.06%/15.93%, Isp 13.99%/18.42%;
- OOD: thrust 20.46%/26.11%, Isp 13.82%/17.86%.

All are below the preregistered 25% median and 40% p90 engineering-scale limits.
These diagnostics did not rescue a mean model that failed point-accuracy gates.

## Integrity

- complete actual-import closure hash:
  `005e0895a4742548e56503689f78ff22d6eac1ca83eb14b0de1c7b0eba5a8351`;
- protocol tree hash:
  `16e0fd5821433fea4a354766cd7fdd4adbfb9ff6a225dc0d00c96f2be6246a28`;
- frozen method hash:
  `e3b0546bb68385aecf3fc1160c9979c384c47af32f4e4b3f83239e60a3758183`;
- run manifest hash:
  `f9d7691106cdf2b90d098494394397e466ae9d730545929ca30a3afae6ac4504`;
- exclusive `O_CREAT|O_EXCL` lock retained;
- reruns and post-result tuning: none.
