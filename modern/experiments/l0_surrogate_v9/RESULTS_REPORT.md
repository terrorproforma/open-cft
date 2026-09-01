# L0 surrogate v9 results

## Outcome

V9 is an **accepted** same-domain prospective L0 software-emulation result.
The smallest passing candidate was the exact-leading residual ARD Matérn-5/2 GP
at 64 training rows. Its inputs are the five output-relevant normalized
coordinates; its leading mean is the exact documented L0 momentum equation and
the GP learns the remaining residual. This is not evidence of physical or
higher-fidelity accuracy.

The assessment used 728 fresh rows in 129 independent groups: 244/46
interpolation rows/groups, 244/43 boundary, and 240/40 OOD. Exact coordinate
intersection with every V3–V8 assessment was zero. Coarse same-domain overlap
was intentional and disclosed.

## Assessment metrics

All gates passed for every output and scope:

- interpolation thrust NRMSE/worst `7.81e-17`/`2.11e-16`, group/row coverage
  100%/100%, normalized median/p90 width `8.07e-16`/`9.83e-16`;
- interpolation Isp NRMSE/worst `1.69e-16`/`4.72e-16`, group/row coverage
  100%/100%, median/p90 width `1.18e-15`/`1.18e-15`;
- boundary thrust NRMSE/worst `8.15e-17`/`3.51e-16`, group/row coverage
  100%/100%, median/p90 width `6.32e-16`/`7.02e-16`;
- boundary Isp NRMSE/worst `1.67e-16`/`7.08e-16`, group/row coverage 100%/100%,
  median/p90 width `1.42e-15`/`1.65e-15`;
- OOD thrust NRMSE/worst `1.19e-16`/`4.21e-16`, group/row coverage 100%/100%,
  median/p90 width `8.42e-16`/`9.83e-16`;
- OOD Isp NRMSE/worst `2.29e-16`/`7.08e-16`, group/row coverage 100%/100%,
  median/p90 width `1.89e-15`/`1.89e-15`;
- overall thrust NRMSE/worst `9.47e-17`/`4.21e-16`, group/row coverage 100%/100%,
  median/p90 width `7.72e-16`/`9.83e-16`;
- overall Isp NRMSE/worst `1.90e-16`/`7.08e-16`, group/row coverage 100%/100%,
  median/p90 width `1.42e-15`/`1.89e-15`.

Every equal-group row-coverage standard deviation was zero. Row coverage above
95% is diagnostic only under the preregistered simultaneous-group target.

## Selected interval families

- interpolation: asymmetric signed input-distance for thrust; asymmetric signed
  absolute for Isp;
- boundary: symmetric absolute for thrust; asymmetric signed raw-GP-SD for Isp;
- OOD: asymmetric signed absolute for both outputs.

## Integrity

- preregistration commit: `a4fc39dd46e1b93d775eab1456d0d8693ff4f4f9`;
- protocol tree hash:
  `f4d10b0a22f2491ffcd84bb3c29e2553f5c9212ef7c4c1d766ea10bc854a3467`;
- runtime closure hash:
  `02ca6b1e96bb44d5dd2546dbfeb5ad372704de80f445a27d34e6b5616c488a28`;
- frozen method hash:
  `f0098e9b262449cbb243334b43fc96250be82d1246804e96132c4006c862f566`;
- frozen calibration hash:
  `b3c7922cb0ff10b781fb0fe3cb726a0495947a63b01bc7acc46fbb64ad48e7d2`;
- run manifest hash:
  `c8170ed1264dfced07241feb93f9b6b9ae34946899c556bf8998b4f8a66e3944`;
- clean detached worktree and complete transitive Git-blob closure validated;
- exclusive `O_CREAT|O_EXCL` lock retained;
- assessment accessed exactly once after both freezes;
- reruns and post-assessment tuning: none.
