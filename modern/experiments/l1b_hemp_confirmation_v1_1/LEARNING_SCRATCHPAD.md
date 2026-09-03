# Learning scratchpad - L1b HEMP confirmation v1.1

## Carried from v1 (established before execution)

- The material-aware solver already exists (`cft_revival.fem_reference`, NUMERICAL_P2_QUALIFIED
  on the divergent-exit stack); the confirmation reuses its screening configuration (two nested
  adaptive levels) and the v3.1 cusp definition unchanged. Only the source model changes (iron
  poles, yoke, recoil magnets instead of vacuum current sheets).
- [self] The default graded mesh (4 feature elements) is sized by the thin dielectric, not by
  the bore; 3 feature elements keep the largest level 1 under 470k DOFs.
- [self] Scale the linear P2 field by the design's L1a `source_strength_scale` before any |B|
  comparison, otherwise the "wall |B| ratio" measures the design variable, not the iron.
- [self] Axis nulls OUTSIDE the straight section move by 1-2 mm under the iron; the channel
  nulls by up to ~1 mm (036); the WALL cusps by <= 0.35 mm. Report the three populations
  separately and record the axis-null-to-cusp lean of both maps.
- [tool] Pin `OPENBLAS/MKL/OMP_NUM_THREADS=1` before numpy import for a bitwise determinism replay.

## Learned from the v1 rejection

- [self] A shakedown on a SAMPLE of designs proves the code path, not the design set. Two of the
  twelve un-shaken designs failed a cheap fail-closed gate (mesh angle) that costs 1-7 s per
  design to evaluate. Every fail-closed gate that can be evaluated without the expensive stage
  must be evaluated for EVERY declared design before the freeze (v1.1: whole-set mesh preflight
  recorded in shakedown.json and verified by prepare/execute).
- [self] Body-fitted structured meshers inherit geometric near-coincidences of the design
  (0.045 mm between two mandatory axial coordinates; a 0.25 mm exit taper) as slivers; the
  resulting minimum angle does not improve with resolution. A rejection gate copied from a
  campaign with regular geometry (10 deg) is not a property of a Sobol design set; declare the
  gate the set can meet and disclose the sliver statistics instead of silently excluding designs.
- [self] The single-execution / no-rerun rule worked as intended: the rejected bundle is
  recorded (`978c71be`), the diagnosis is post hoc on the same geometry (no evidence changed), and
  the fix is a new preregistration with the two casualties in its shakedown set.

## Claim guardrail

P2 with LINEAR soft-iron and recoil magnets, two nested levels: a numerical confirmation of the
L1a topology and rho classification. No saturation, no B-H curve, no plasma, mirror-probability,
thrust or efficiency claim; no paper admission in this campaign.
