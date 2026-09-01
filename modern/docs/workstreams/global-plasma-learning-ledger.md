# Global Plasma Learning Ledger

## 2026-09-01

- [user] This concurrent workstream owns only new plasma source, test, spec,
  and `global-plasma-*` documentation paths; preserve all shared and legacy
  files and Git state.
- [self] Equation prose alone was insufficient for the cusp-current sign.
  Substituting the primary paper's rounded DM9.2 table proved that cusp ion
  current is an outflow: `ji_upstream=ji_downstream+I_source-jic`.
- [self] The archived plus-cusp sign is not a small convention difference: the
  third cell misses its own published table by 0.204 A.
- [self] Kornfeld's three cusp potentials cancel from both local and global
  expressions. Keeping them as solver variables without a sheath closure
  creates an unidentifiable null space; eliminate them and document the
  reduction.
- [self] The archived executable excitation sign contradicts the source's
  energy-partition prose and `CE+CI+CT=1`. Do not preserve known broken signs
  as compatibility modes.
- [self] A genuinely unresolved raster sign belongs in a named hypothesis
  enum, with the physical nonnegative-loss constraint applied to every branch.
- [self] A least-squares status is not conservation evidence. Publish a plasma
  state only after normalized residual, finite-value, bound, and inequality
  checks.
- [self] The 2007 model prescribes current specifically because neutral flow
  and ionization cross sections are absent. Calling it a mass-flow discharge
  solver or using it to validate thrust would exceed its equations.
- [tool] This Windows PowerShell version does not accept `&&`; use separate
  statements and preserve command outcomes in the workstream devlog.
- [self] Yeo et al. 2020 values compare reduced and PIC models. Store them with
  machine-readable prohibited uses so they cannot drift into fit targets or
  solver tolerances.

## Open risks

- Original MathCAD equations are required to settle raster/OCR operators.
- Independent sheath, xenon collision/rate, neutral-flow, and uncertainty
  closures remain absent.
- Exact publication-run files and experimental data with uncertainty are not
  available.

## 2026-09-01 audit corrections

- [self] Signed terminal `j_i4` must remain signed in the equation:
  `j_i3-I4-j_i4=0`. Replacing it with an absolute-value rearrangement changes
  the balance; the table gives `0.155+0.002=0.157 A`.
- [self] The fourth thermal denominator is explicitly
  `j_e3*(1-p4)+I4`. `j_e4` includes a different terminal-current accounting
  path and produced a 5.072 W table mismatch.
- [self] Normal equations square the Jacobian condition number. Use
  variable-scaled pivoted QR for LM steps and report rank and independent
  subspace condition rather than implying a unique 25-variable root.
- [self] A public projection is an untrusted numerical callback. Clip its
  input, then reject changed dimension, nonfinite values, or output outside
  bounds before canonical re-clipping.
- [self] Residual scale fallbacks hide invalid operating domains. Validate
  `Ua`, `Ia`, `Ua*Ia`, voltage powers, and every derived bound once, then use
  the exact declared scales.
- [self] A signed boundary exchange is not a nonnegative loss. Report electron
  loss, ion exchange direction, and net power separately.
- [self] Corrected DM9.2 evidence remains rank 22 with a nonzero residual
  floor. Preserve all row residuals and never relabel minimum-error evidence
  as convergence.
- [self] A normal `Ua*Ia` product does not imply normal component scales:
  `Ua=1e100` can mask subnormal `Ia=1e-310`. Validate `Ua`, `Ia`, and their
  product independently before deriving bounds.
- [user] Preserve publication-native evidence names. Use `MDO (original)` and
  a stable source-native ID; keep any postprocessing description as editorial
  interpretation rather than rewriting the source model label.
- [self] IEEE normality of a primitive does not guarantee representability of
  a required nonlinear scale. Minimum-normal `Ua` passes the primitive gate
  but its `Ua^(3/2)` cathode scale underflows, so reject it with the derived
  reason. Evaluate minimum-normal `Ia` independently because `Ua=1` preserves
  a normal product and finite bounds.
