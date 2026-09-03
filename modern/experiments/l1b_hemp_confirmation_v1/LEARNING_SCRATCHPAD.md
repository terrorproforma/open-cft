# Learning scratchpad - L1b HEMP confirmation v1

## Established before execution

- The material-aware solver already exists (`cft_revival.fem_reference`, NUMERICAL_P2_QUALIFIED
  on the divergent-exit stack); the confirmation reuses its screening configuration (two nested
  adaptive levels) and the v3.1 cusp definition unchanged. Nothing new is defined; only the
  source model changes (iron poles, yoke, recoil magnets instead of vacuum current sheets).
- [self] The default graded mesh (4 feature elements) is sized by the thin dielectric, not by
  the bore: the five-stage 25 mm designs reach 188k level-0 DOFs, i.e. ~750k at level 1 - outside
  the session's RAM budget. Survey the level-0 mesh of EVERY design before freezing the mesh
  parameters; 3 feature elements keep the largest level 1 under 470k.
- [self] The L1a and the P2 fields carry different magnet strengths by construction (the L1a
  bands are scaled by the Sobol variable `source_strength_scale`); scale the linear P2 field
  by the same factor before any |B| comparison, otherwise the "wall |B| ratio" measures the
  design variable, not the iron.
- [self] Axis nulls OUTSIDE the straight section (anode side / exit) move by 1-2 mm under the
  iron (un-cancelled end field); the channel nulls move by < 0.2 mm. Report the two populations
  separately; a pooled axis-null bijection within the cusp tolerance would fail for a reason
  that has nothing to do with the cusps.
- [tool] `fem_reference` PCG reductions run through BLAS; pin `OPENBLAS/MKL/OMP_NUM_THREADS=1`
  before numpy is imported so the determinism replay compares bitwise-reproducible solves.
- The cusp-position tolerance is tied to the discretisation (one level-0 bore element, never
  below the L1a axial step), not tuned on any outcome; the P2 level-0-to-level-1 cusp shift is
  recorded so the reader can see how much of the tolerance is P2 discretisation.

## Claim guardrail

P2 with LINEAR soft-iron and recoil magnets, two nested levels: a numerical confirmation of the
L1a topology and rho classification. No saturation, no B-H curve, no plasma, mirror-probability,
thrust or efficiency claim; no paper admission in this campaign.
