# L0 surrogate v7 results

## Outcome

V7 is a valid **failed prospective validation**. Exact group-count conformal
intervals achieved the preregistered simultaneous-group target in every scope,
but were rowwise conservative: every row-coverage result exceeded the 0.95
upper gate. OOD thrust also exceeded the 15% worst-error gate. No threshold,
model, interval or result was changed and the experiment was not rerun.

V7 preserves V6 as empirical evidence while correcting its pooled-row rank
defect. This remains same-domain deterministic L0 software-emulation evidence,
not a physical-accuracy claim.

## Exact method

The exchangeability unit is one independent spatial group. For symmetric
intervals, each group contributes its maximum absolute normalized residual and
the rank is `min(G, ceil((G+1)*9/10))`. For asymmetric intervals, each group
contributes separate lower and upper one-sided maxima, each ranked at
`min(G, ceil((G+1)*19/20))`; a union bound gives the 90% simultaneous target.
All ranks used `Fraction` and integer ceiling arithmetic.

Final ranks:

- Interpolation thrust: symmetric raw-GP-SD, `G=44`, rank `41/44`
- Interpolation Isp: asymmetric signed input-distance, lower/upper rank `43/44`
- Boundary thrust: asymmetric signed absolute, lower/upper rank `42/43`
- Boundary Isp: symmetric raw-GP-SD, rank `40/43`
- OOD thrust: asymmetric signed absolute, lower/upper rank `42/43`
- OOD Isp: asymmetric signed absolute, lower/upper rank `42/43`

## Development-role selection

- 96 rows failed because method-group Isp worst error was `15.248%`.
- 128 rows passed: overall thrust NRMSE/worst `1.799% / 7.903%`; Isp `1.873% / 6.128%`.
- 160 rows passed: overall thrust `1.685% / 7.499%`; Isp `1.555% / 5.690%`.

The preregistered smallest-passing rule selected the 128-row ARD Matérn-5/2
GP. Every selected interval family passed method-role simultaneous-group,
equal-group row-coverage, and stability gates before final calibration opened.

## Final assessment

Metrics are `NRMSE / worst error / row coverage / simultaneous-group coverage /
group-row-coverage SD`.

- Interpolation, 240 rows and 42 groups:
  - thrust `0.951% / 3.506% / 0.9583 / 0.9048 / 0.1733`
  - Isp `1.558% / 4.458% / 0.9625 / 0.8571 / 0.0852`
- Boundary, 240 rows and 46 groups:
  - thrust `1.597% / 6.213% / 0.9708 / 0.9130 / 0.1162`
  - Isp `1.570% / 4.593% / 0.9875 / 0.9348 / 0.0625`
- OOD, 242 rows and 44 groups:
  - thrust `2.901% / 17.302% / 0.9628 / 0.9091 / 0.1592`
  - Isp `2.711% / 9.740% / 0.9752 / 0.9091 / 0.0855`
- Overall, 722 rows and 132 groups:
  - thrust `1.992% / 17.302% / 0.9640 / 0.9091 / 0.1509`
  - Isp `2.023% / 9.740% / 0.9751 / 0.9015 / 0.0784`

All simultaneous-group and stability gates passed. All NRMSE gates passed.
Every row-coverage upper gate failed; OOD and overall thrust worst-error gates
also failed.

Assessment-coordinate intersection with all V3–V6 assessment sets was zero.
Method selection froze before final-calibration labels; cluster calibration
froze before the global assessment loaded exactly once.

## Lifecycle note

Post-result verification found a test-only branching error: the lifecycle test
mistook a failed completed assessment for an early-stop result because it
checked the wrong manifest key. The assertion was corrected to distinguish
presence of `assessment_metrics`. Scientific protocol and result artifacts
remain byte-for-byte unchanged, and no rerun occurred.
