# L0 surrogate v5 results

## Outcome

V5 executed exactly once from detached commit
`4fbe801320d0bd7d5d0871d2f6257c8c61a8856e`. Git protocol identity and the
transitive dependency tree were valid, the retained atomic execution lock was
acquired, and the new design had zero coordinate intersection with every v3/v4
calibration or assessment set.

The active campaign was rejected: none of three replicates passed every
predeclared assessment and group-held-out method-diagnostic gate. The fixed
baseline also failed all three replicates. Thresholds and models were not
changed and the experiment was not rerun.

Each metric below is `NRMSE / worst normalized error / coverage / scope pass`.
`T` is axial thrust and `I` is specific impulse.

## Assessment metrics

### Split 2026090501

- Active interpolation: T `0.01105 / 0.03782 / 0.90625 / pass`; I `0.01614 / 0.03933 / 0.93750 / pass`
- Active boundary: T `0.02069 / 0.07770 / 0.84694 / fail`; I `0.01830 / 0.05881 / 0.92857 / pass`
- Active OOD: T `0.03647 / 0.12790 / 0.89423 / pass`; I `0.03630 / 0.16496 / 0.84615 / fail`
- Active overall: T `0.02539 / 0.12790 / 0.88255 / pass`; I `0.02557 / 0.16496 / 0.90268 / fail`
- Baseline interpolation: T `0.01358 / 0.03294 / 0.88542 / pass`; I `0.01586 / 0.04449 / 0.87500 / pass`
- Baseline boundary: T `0.06078 / 0.29165 / 0.78571 / fail`; I `0.04248 / 0.14504 / 0.84694 / fail`
- Baseline OOD: T `0.10681 / 0.29930 / 0.88462 / fail`; I `0.10498 / 0.32565 / 0.90385 / fail`
- Baseline overall: T `0.07250 / 0.29930 / 0.85235 / fail`; I `0.06723 / 0.32565 / 0.87584 / fail`

### Split 2026090502

- Active interpolation: T `0.01244 / 0.03574 / 0.91667 / pass`; I `0.02294 / 0.07838 / 0.84375 / fail`
- Active boundary: T `0.02396 / 0.09206 / 0.85859 / pass`; I `0.02267 / 0.06766 / 0.88889 / pass`
- Active OOD: T `0.03694 / 0.11508 / 0.96078 / fail`; I `0.04484 / 0.09458 / 0.79412 / fail`
- Active overall: T `0.02664 / 0.11508 / 0.91246 / pass`; I `0.03212 / 0.09458 / 0.84175 / fail`
- Baseline interpolation: T `0.00938 / 0.05036 / 0.97917 / fail`; I `0.01828 / 0.04800 / 0.90625 / pass`
- Baseline boundary: T `0.04319 / 0.23106 / 0.93939 / fail`; I `0.03171 / 0.11787 / 0.90909 / pass`
- Baseline OOD: T `0.12304 / 0.43010 / 0.89216 / fail`; I `0.10342 / 0.29239 / 0.82353 / fail`
- Baseline overall: T `0.07648 / 0.43010 / 0.93603 / fail`; I `0.06416 / 0.29239 / 0.87879 / fail`

### Split 2026090503

- Active interpolation: T `0.01587 / 0.03968 / 0.72917 / fail`; I `0.01779 / 0.04975 / 0.90625 / pass`
- Active boundary: T `0.01767 / 0.05035 / 0.73469 / fail`; I `0.01885 / 0.05429 / 0.89796 / pass`
- Active OOD: T `0.04359 / 0.19653 / 0.89000 / fail`; I `0.03551 / 0.10959 / 0.84000 / fail`
- Active overall: T `0.02885 / 0.19653 / 0.78571 / fail`; I `0.02551 / 0.10959 / 0.88095 / pass`
- Baseline interpolation: T `0.01019 / 0.03162 / 0.87500 / pass`; I `0.01719 / 0.05181 / 0.85417 / pass`
- Baseline boundary: T `0.04654 / 0.14307 / 0.78571 / fail`; I `0.02558 / 0.10110 / 0.98980 / fail`
- Baseline OOD: T `0.12141 / 0.44708 / 0.81000 / fail`; I `0.10821 / 0.30243 / 0.91000 / fail`
- Baseline overall: T `0.07596 / 0.44708 / 0.82313 / fail`; I `0.06555 / 0.30243 / 0.91837 / fail`

## Selected interval methods

- Split 2026090501 active — interpolation: T `symmetric-raw-gp-sd`, I `asymmetric-signed-absolute`; boundary: T `symmetric-absolute`, I `asymmetric-signed-absolute`; OOD: T `symmetric-input-distance`, I `asymmetric-signed-raw-gp-sd`
- Split 2026090501 baseline — interpolation: T `symmetric-input-distance`, I `symmetric-input-distance`; boundary: T `symmetric-input-distance`, I `asymmetric-signed-raw-gp-sd`; OOD: T `symmetric-absolute`, I `asymmetric-signed-raw-gp-sd`
- Split 2026090502 active — interpolation: T `symmetric-input-distance`, I `symmetric-absolute`; boundary: T `symmetric-absolute`, I `asymmetric-signed-raw-gp-sd`; OOD: T `symmetric-raw-gp-sd`, I `asymmetric-signed-input-distance`
- Split 2026090502 baseline — interpolation: T `symmetric-raw-gp-sd`, I `asymmetric-signed-absolute`; boundary: T `symmetric-input-distance`, I `asymmetric-signed-raw-gp-sd`; OOD: T `symmetric-raw-gp-sd`, I `symmetric-input-distance`
- Split 2026090503 active — interpolation: T `symmetric-absolute`, I `symmetric-raw-gp-sd`; boundary: T `asymmetric-signed-input-distance`, I `asymmetric-signed-input-distance`; OOD: T `symmetric-raw-gp-sd`, I `symmetric-raw-gp-sd`
- Split 2026090503 baseline — interpolation: T `symmetric-absolute`, I `asymmetric-signed-absolute`; boundary: T `symmetric-absolute`, I `symmetric-raw-gp-sd`; OOD: T `symmetric-absolute`, I `asymmetric-signed-absolute`

## Interpretation

Active learning materially reduced OOD and overall point error relative to the
fixed baseline, but interval transfer remained unstable across independent
groups. Every active replicate also failed at least one new method-selection
stability gate. The evidence supports retaining the point-selection strategy
for future work, but not claiming calibrated uncertainty from this 96-row
protocol.
