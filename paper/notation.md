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
  current ones are `GATE-WALL-LOSS-V4`, classified
  `collisionless_prescribed_field_test_particle_wall_loss_not_pic`, and
  `GATE-MDO-L0-V1`, classified
  `l0_model_robust_multiobjective_optimisation_under_declared_input_uncertainty_not_thruster_performance`
  (optimiser evidence on the L0 model under the declared closure CL-1).
- **Closure CL-1** — the declared multiplicative cusp-survival closure of the
  optimisation campaign: \(S(\mathbf p)=\prod_{k=1}^{4}(1-p_k)\) scales the
  produced ionised fraction and leaves the anode current unchanged. It is a
  declared assumption, neither derived nor validated; every number of the
  campaign is conditional on it and on the declared priors.
- **Robust objective (CVaR)** — for each L0 objective the mean of the worst
  16 of 64 frozen Halton sample values of the uncertain inputs; the robust
  constraint is the worst sampled beam-current margin, which enforces the
  worst *sampled* case, not the worst case over the prior's support.
- **Numerical screening** — a preregistered, single-execution L1a field-only
  screening study (linear-vacuum equivalent-current fields; no
  permanent-magnet or nonlinear-iron material model) admitted by a
  `numerical-screening` gate at exactly its `recorded_outcome`:
  `accepted-screening` (`GATE-L1A-SWEEP-V2`), `preregistered-null`
  (`GATE-FOUR-CELL-V2`) or `recorded-characterization`
  (`GATE-TOPOLOGY-CHAR-V1`). Gate status `accepted` means admitted as
  recorded, never that a positive finding is accepted; it opens no L gate.
- **Axis cusp (sweep QoI)** — a sign change of the on-axis \(B_z\) between
  adjacent magnet stages in a sampled field map; a sampled-axis descriptor,
  not a continuous critical-point proof.
- **Interior cusp (four-cell search)** — a vector null of the accepted
  \(\psi\) map with finite-box boundary nulls excluded; a candidate is stable
  only if every map holds exactly four such cusps, geometry-registered and
  shifting by at most the frozen tolerance across maps.
- **Eligible cusp / eligible cell (characterization)** — an X-type (index
  \(-1\)) or O-type (index \(+1\)) clustered vector null strictly inside the
  plasma channel with the required constant-\(\psi\) connectivity; "stable"
  adds matching in all three maps with unchanged class and eligibility.
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
- A preregistered null (four-cell v2) or a recorded characterization null
  (characterization v1) is a null under its frozen cusp/cell definitions and
  linear-vacuum field model; it is not proof that no such design exists.
- Sweep axis cusps and per-cell mirror ratios are field-only screening
  quantities; they are not confinement cells, confinement predictions or
  performance.
- A robust or nominal Pareto front of the L0 optimisation campaign is an
  optimiser estimand under the declared closure CL-1 and priors; it is not
  thruster performance, a design recommendation, a calibration or a
  validation, and "qLogNEHVI beat the baselines" holds only for the recorded
  budget, seeds and model.
- The collisionless wall-hit probability of the wall-loss campaign is not the
  per-cusp loss probability of the Kornfeld cusp cascade; inserting the former
  as the latter gives no beam under CL-1.
