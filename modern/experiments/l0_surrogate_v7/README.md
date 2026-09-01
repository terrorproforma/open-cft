# L0 surrogate v7

V7 preserves V6 as empirical evidence while correcting its strict
group-exchangeability defect. It is prospective same-domain L0
software-emulation validation, not a physical-accuracy claim.

The conformal target is simultaneous coverage of every row in a future
exchangeable spatial group. Symmetric scores are per-group maxima and use
`min(G, ceil((G+1)*9/10))`. Asymmetric intervals use separate per-group
one-sided maxima at exact 19/20 ranks; the union bound gives a 90%
simultaneous target. All ranks use `Fraction` and group count only.

One global method role selects the smallest passing 96/128/160-row mean model
and interval family. Final calibration remains inaccessible until that artifact
is frozen. The fresh global assessment remains inaccessible until cluster
calibration is frozen, and then loads once.
