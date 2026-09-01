# Physics Workstream Learning Ledger

## Established

- Charge-state-weighted current and energy must use charge number `z`, while
  xenon momentum uses the common ion mass and therefore scales with `sqrt(z)`.
- Beam kinetic power equals `Vd*Ib` exactly under the same collisionless
  acceleration assumptions; retaining both calculations creates a useful
  conservation diagnostic.
- A divergence factor is an axial momentum correction, not a beam kinetic
  power correction.
- Mass utilization and charge utilization are different for mixed Xe+/Xe2+:
  `f1+f2` is ionized mass fraction, while `f1+2*f2` weights beam current.
- Boundary-specific efficiency names prevent an anode/model conversion number
  from being presented as complete system efficiency.
- Warp 1.14 can perform the complete reduced batch in float64 on both CPU and
  RTX 5090 CUDA without a native CUDA build toolchain.
- `A_phi=B0*r/2` is regular at the axis and yields exact uniform `Bz=B0`
  through the analytic `r -> 0` cylindrical-curl limit.
- Decimal-looking fraction tolerances are not numerical contracts. Comparing
  the exact rational values of binary64 inputs against a two-ULP-at-unity bound
  prevents correctly rounded `fsum` from hiding a mathematically larger error.
- A symmetric numeric epsilon is also insufficient at exponent boundaries,
  where the adjacent floats above and below a value have different spacing.
  Four scale-aware ULPs cover observed legitimate caller regrouping across all
  1024 reconstructed boundaries; the effective PPU load must then be snapped
  canonically so loss and efficiency use one representation.
- Computing `Isp=F/(mdot*g0)` loses information when representable velocity is
  multiplied by tiny mass flow first. The algebraic form
  `k_div*(f1*v1+f2*v2)/g0` preserves the representable result.
- Host preparation must prove every accepted point has finite derived states
  before a GPU launch; device results receive a second finite-publication gate.
- Absolute tolerances can silently accept complete loss of tiny observables.
  Tiny-state tests require nonzero values and relative/high-precision-oracle
  agreement.
- `sqrt(a/b)` can underflow `a/b` even when the square root is representable,
  and `(mdot/m)*e` can overflow or underflow intermediates. Separating square
  roots and binary exponents preserves the max-mass/max-flow oracle case.
- A useful algebraic limit does not make represented `0/0` a reported
  efficiency. Requested/effective power metadata and `None` for undefined
  ratios make the numerical boundary auditable.
- Manufactured solutions are also subject to the finite-publication contract;
  finite field strength and radius do not guarantee a finite vector potential.
- Ordering `B0*r/2` as `(0.5*B0)*r` avoids some overflow but destroys the
  minimum-subnormal field before a huge radius can restore a representable
  result. Combining `frexp` mantissas and exponents, then applying the half in
  the final exponent, handles both tiny/huge symmetries and correct subnormal
  rounding.

## Evidence policy

- Standard mechanics and SI definitions support the L0 equations.
- Yeo et al. 2020 S1 values are external cross-model fixtures. The
  PIC-informed result uses information from the PIC case and is not
  independent validation.
- The legacy 2017 model and its ambiguous “total efficiency” are not accepted
  as truth.
- Applicability warnings are part of the result because numerical consistency
  does not establish physical predictive accuracy.

## Open questions

- Which independently sourced equation set and sign convention should define
  a future global CFT model?
- What measured or independently simulated observables can constrain beam
  current fraction and divergence without circular calibration?
- Which xenon atomic mass convention and isotope composition should calibrated
  studies use, and what uncertainty is material?
- Which cathode, wall, anomalous transport, and collision models are valid in
  each operating regime?
- What field profile, mesh sequence, nonlinear material data, and boundary
  truncation establish the future FEM acceptance gate?
