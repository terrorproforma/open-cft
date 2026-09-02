# Notation and glossary

## Physical symbols

- \(\dot m_{\mathrm{Xe}}\) — inlet xenon mass flow, kg s\(^{-1}\).
- \(m_{\mathrm{Xe}}\) — xenon particle mass, kg particle\(^{-1}\).
- \(\dot N_{\mathrm{Xe}}\) — inlet xenon particle rate, s\(^{-1}\).
- \(f_0,f_1,f_2\) — number fractions of Xe, Xe\(^+\), and Xe\(^{2+}\);
  the represented fractions sum to one.
- \(z\) — xenon ion charge number, restricted to 1 or 2 in L0.
- \(e\) — elementary charge, C.
- \(V_d\) — discharge voltage supplied to L0, V.
- \(v_z\) — ideal directed speed for represented charge state \(z\),
  m s\(^{-1}\).
- \(I_b\), \(I_a\) — beam and anode currents, A.
- \(k_b=I_b/I_a\) — supplied beam-current fraction.
- \(k_{\mathrm{div}}\) — supplied axial fraction of ion momentum.
- \(F_{\mathrm{ion}}\), \(F_{\mathrm{axial}}\) — undiverged ion momentum
  flux and axial thrust, N.
- \(g_0\) — standard acceleration of gravity, m s\(^{-2}\).
- \(I_{\mathrm{sp}}\) — specific impulse, s.
- \(P_{\mathrm{beam}}\), \(P_{\mathrm{anode}}\),
  \(P_{\mathrm{thruster}}\), \(P_{\mathrm{PPU}}\) — beam kinetic, anode,
  thruster-electrical, and power-processing-unit boundary powers, W.

## Fidelity and evidence terms

- **L0** — the current algebraic conservation-reduced operating-point
  baseline. It has no spatial coordinate, mesh, or geometric decision input.
- **L1** — reserved paper level for a field-resolved reduced model. It has no
  admitted result while `GATE-L1` is closed.
- **L2** — reserved paper level for a coupled hybrid plasma/field model. It has
  no admitted result while `GATE-L2` is closed.
- **L3** — reserved paper level for PIC and/or experimental comparison. It has
  no admitted result while `GATE-L3` is closed.
- **Numerical campaign** — an accepted, preregistered numerical result about a
  declared component model that is not a physics level of the ladder. It is
  admitted by a `numerical-campaign` gate whose typed manifest is committed
  and cross-checked against the sealed results bundle; it opens no L gate. The
  only current one is `GATE-WALL-LOSS-V4`, classified
  `collisionless_prescribed_field_test_particle_wall_loss_not_pic`.
- **Test-particle wall-hit probability** — the fraction of collisionless
  electron orbits, launched by a preregistered protocol in one prescribed
  magnetostatic field, that first intersect the dielectric wall rather than
  leave the domain. A pooled equal-weight value over launch strata is a design
  average of the protocol, not a plasma loss rate.
- **F0--F3** — optimization information-source labels from the campaign spec:
  corrected-analytical, fields/reduced, hybrid, and PIC/experiment.
- **Physics level versus information source** — L-labels describe the paper's
  model-evidence progression; F-labels identify records consumed by the
  optimization campaign. A manifest must map them explicitly.
- **Verification** — evidence that equations are implemented and solved
  consistently, including tests, manufactured solutions, residuals, and
  independent backend comparisons.
- **Validation** — comparison with physical observations including measurement
  and operating-condition uncertainty.
- **Calibration** — inference of uncertain parameters or discrepancy against a
  declared dataset. No calibration result is currently admitted.
- **Model discrepancy** — inadequacy arising from omitted or approximate
  physics; it is distinct from surrogate posterior uncertainty.
- **Evidence gate** — a structured closed manuscript block and registry entry
  that can open only through an accepted committed manifest.
- **Result manifest** — machine-readable identity for model, code, inputs,
  environment, outputs, checks, uncertainty, failures, artifacts, and hashes.
- **Verified hypervolume** — hypervolume computed under a frozen normalization
  and reference point using eligible highest-fidelity observations. It is not
  interchangeable with surrogate-predicted hypervolume.

## Prohibited equivalences

- L0 conservation closure is not physical predictive accuracy.
- Python/CUDA parity is not independent experimental validation.
- The historical `total_efficiency` objective is not interchangeable with
  L0's three explicitly bounded power-conversion efficiencies.
- A shifted-Halton design is not a Sobol sequence.
- A successful solver status is not adequate without residual and
  applicability checks.
- Working-tree output is not publication evidence until its manifest and
  artifacts are committed and accepted.
- A collisionless test-particle wall-hit fraction is not a plasma loss rate,
  a PIC result, a self-consistent result, or thruster performance.
- The launch cells of the wall-loss protocol are protocol positions in one
  qualified field; they are not demonstrated confinement cells, and the
  campaign does not open `GATE-L1`.
