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
  (optimiser evidence on the L0 model under the declared closure CL-1), and
  `GATE-MDO-L0-V2`, classified
  `l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance`
  (optimiser evidence on the L0 model over the catalogue of 96 screened sweep
  designs under CL-1 with per-design screening posteriors).
- **Closure CL-1** — the declared multiplicative cusp-survival closure of the
  optimisation campaigns: \(S(\mathbf p)=\prod_{k=1}^{4}(1-p_k)\) scales the
  produced ionised fraction and leaves the anode current unchanged. It is a
  declared assumption, neither derived nor validated; every number of the
  campaigns is conditional on it and on the declared priors (v1: common uniform
  priors; v2: each design's Jeffreys Beta posteriors of its per-cell screening
  counts).
- **Closure identification (catalogue campaign)** — CL-1 in the catalogue
  campaign identifies the collisionless test-particle wall-hit probability of a
  launch cell on a linear-vacuum screening field with the per-cusp survival
  factor \(1-p_k\). The v1 scenario analysis showed this quantity is not the
  Kornfeld per-cusp probability of a sustained discharge, so the identification
  is declared, not derived; a design that wins under it wins under this closure
  only.
- **Closure CL-2** — the declared sensitivity closure of the catalogue
  campaign: \(S=1-p_{\mathrm{pooled}}\) with the pooled wall-hit probability
  over all 512 launches of a design. Evaluated on the recorded designs only;
  its front shares no design with the CL-1 front (Jaccard 0.0), which is why
  the paper calls the design ranking closure-dependent.
- **Catalogue** — the 96 accepted sweep-v2 designs of the wall-loss geometry
  screening, each carried with its sealed geometry and accepted-2N per-cell
  counts, entering the catalogue campaign as a categorical index; no geometry
  value is optimised and nothing is interpolated between designs (both
  surrogates fitted to the dataset were rejected).
- **Saturated cell** — a launch cell whose 128 launches all hit the wall in the
  screening; 73 of the 96 catalogue designs have one, and under CL-1 the
  multiplicative cascade then leaves a nominal survival of the order of the
  Jeffreys pseudo-count, which is why 77 of 96 designs have negligible own
  dense-reference hypervolume.
- **Robust objective (CVaR)** — for each L0 objective the mean of the worst
  16 of 64 frozen Halton sample values of the uncertain inputs; the robust
  constraint is the worst sampled beam-current margin, which enforces the
  worst *sampled* case, not the worst case over the prior's support.
- **Analytic consistency result** — a derivation about a declared equation set
  whose closed form is verified numerically to a stated tolerance, pinned by
  committed tests and recomputed by the checker at every run, admitted by an
  `analytic-consistency` gate. The current one is `GATE-FOUR-CELL-CLOSURE-V1`,
  classified
  `analytic_consistency_of_the_corrected_four_cell_power_balance_not_thruster_physics`:
  on the manifold of rows R00--R26 of the corrected four-cell ledger the global
  power row is \(R_{27}=2\,(J_{e,3}+I_4)(\varphi_4-U_a)+E_I(p_1 j_{e,0}+p_2
  j_{e,1}+p_3 j_{e,2})\) with \(J_{e,k}=j_{e,k}(1-p_k)\); both terms are
  non-negative on the admissible region, so no admissible root exists for any
  positive interior cusp probability. Gate status `accepted` means the
  derivation and its verification are admitted as recorded; it accepts no
  correction (the proposal is `PROPOSED_NOT_ACCEPTED`), says nothing about the
  physical thruster and opens no L gate.
- **Residual floor** — the least-squares stall value \(\max_i|r_i|\) of the
  production solver when no admissible root exists; for the four-cell ledger it
  is linear in the interior cusp probability with a Jacobian of constant rank
  22 of 25, a property of the equations rather than of the solver.
- **Numerical screening** — a preregistered, single-execution screening study
  on L1a linear-vacuum equivalent-current fields (no permanent-magnet or
  nonlinear-iron material model; not P2-qualified) admitted by a
  `numerical-screening` gate at exactly its `recorded_outcome`:
  `accepted-screening` (`GATE-L1A-SWEEP-V2` and `GATE-L1A-SWEEP-V3`),
  `preregistered-null` (`GATE-FOUR-CELL-V2`), `recorded-characterization`
  (`GATE-TOPOLOGY-CHAR-V1`), `accepted-screening-dataset`
  (`GATE-WALL-LOSS-GEOMETRY-SCREENING-V1`) or `accepted-topology-screening`
  (`GATE-CUSP-TOPOLOGY-V3-1`). Gate status `accepted` means admitted as
  recorded, never that a positive finding is accepted; it opens no L gate.
- **Koch design ratio \(\rho\)** — the HEMP design criterion of Koch, Harmann
  and Kornfeld (IEPC-2007-110): the wall field at the cusp plane over the
  adjacent axial field; in the sweep v3 (`GATE-L1A-SWEEP-V3`) the binding
  reading is the conservative one, \(|B|(r_w, z_c)\) over the larger of the two
  adjacent axis peaks, and a design is **HEMP-like** when every wall cusp has
  \(\rho \ge 1.5\). \(x_w = \pi r_w / L\) is the wall-radius-to-pitch parameter,
  \(I_1(x_w)\) the single-harmonic PPM prediction of \(\rho\) for an infinite
  stack and \(x^* = 1.937318\) the threshold at which \(I_1 = 1.5\). The
  campaign recorded \(I_1(x_w)\) as an upper envelope of the realised ratio
  (the finite stack's end field raises the axis peaks adjacent to the end
  cusps), not as its value; every reading is a field ratio of a linear-vacuum
  screening field, never a probability, and no HEMP-like design is a design
  recommendation (the material-aware confirmation is queued, not run).
- **Wall cusp (literature definition)** — the intersection of the separatrix of
  an axis null (the point on the axis where \(B_z\) changes sign; X-type by the
  analytic Jacobian) with the straight dielectric wall; the cusps sit at the
  inter-magnet gaps of a periodic-permanent-magnet stack. A **cell** is the wall
  interval between consecutive cusps (plus anode-side and exit-side partial
  cells). Both are geometric properties of a prescribed field map under the
  definition of `GATE-CUSP-TOPOLOGY-V3-1`, classified
  `SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY` (L1a sets) or
  `P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY` (the single P2 row); a cell is
  not a demonstrated confinement cell.
- **Wall / axis mirror ratio (topology screening)** — the weaker bounding
  cusp's wall field over the smallest wall field inside the cell, and that cusp
  field over the largest axial field on the axis of the cell; field ratios by
  construction, never probabilities or loss fractions.
- **Cusp-cell catalogue** — the per-design record of axis nulls, wall cusps,
  cells and mirror descriptors sealed by the topology screening; a consumer
  contract under its labels, whose first admitted consumer is the wall-loss
  geometry screening v2 (its launch cells).
- **Screening dataset (wall-loss geometry screening v2, catalogue cells)** —
  collisionless prescribed-field test-particle electron orbits (orbit_mc 1.7,
  numpy CPU) launched at the midpoints of the 377 separatrix-bounded catalogue
  cells of the 96 accepted sweep-v2 designs in their re-solved L1a screening
  fields (label `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`) plus one
  launch-design row on the P2-qualified field of the wall-loss campaign (label
  `P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN`, not a replication); per-cell
  wall-access, reflection, escape and timeout fractions with Wilson intervals
  and floors, a frozen two-stage allocation (128 launches per cell, top-up to
  512 when the stage-1 Wilson width exceeds 0.10), a paired N→2N control over
  one eighth of every cell's launches, declared design averages and a pooled
  comparison with the v1 screening. Its results manifest was published post hoc
  (disclosed).
- **Wall-access fraction (screening v2)** — the fraction of the launches of one
  catalogue cell (two radius bands, two energies, two pitch angles, both
  directions, scrambled-Sobol gyrophase) that first reach the straight
  dielectric wall; a collisionless geometric property of the field lines
  through the launch plane, never a loss probability, a per-cusp transit loss
  or a cusp probability for the plasma network without a declared closure.
- **Position class (screening v2)** — anode-side partial cell (anode plane to
  the first wall cusp), interior cell (between consecutive wall cusps) or
  exit-side partial cell (last wall cusp to the end of the straight
  dielectric), as the catalogue defines them.
- **Jeffreys floor (screening v2)** — \(\sqrt{\tilde p(1-\tilde p)/n}\) with
  \(\tilde p=(k+\tfrac12)/(n+1)\), the label-precision floor of a per-cell
  fraction; a cell is "surrogate-ready" when it is at most 0.02. Readiness is
  a precision statement about a screening label, not a fitted surrogate.
- **Direction split (screening v2)** — the exit-side value near one half of the
  divergent-exit designs decomposes into one parallel launch direction that
  reaches the wall in nearly every launch and one that reaches it in nearly
  none; the wall-reaching direction equals the last stage's polarity in 82 of
  90 designs. An observation of this launch design, never a design rule.
- **Screening dataset (wall-loss geometry screening)** — collisionless
  prescribed-field test-particle electron orbits (orbit_mc 1.7, numpy CPU)
  integrated in the re-solved L1a screening fields of all 96 accepted sweep-v2
  designs, classified `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`; per-design
  wall-hit, reflection, escape and timeout probabilities with Wilson intervals
  at two timesteps, a convergence flag, per-cell and per-stratum counts and a
  coupling-consumer record. It is surrogate and optimisation input only when
  carried with its label, never accepted physical-orbit evidence of the kind
  `GATE-WALL-LOSS-V4` admits on the P2-qualified field.
- **Launch cell (screening)** — one of four axial launch positions at fixed
  fractions (1/8, 3/8, 5/8, 7/8) of a design's straight span between the
  injector zone and the end of the straight dielectric; a protocol position
  scaled to each design, not a demonstrated confinement cell.
- **Reflection (test particle)** — an orbit that reverses its parallel velocity
  before reaching any boundary within the path and time budget; a termination
  class of the collisionless model, not a confinement statement.
- **Geometry association (screening)** — a Spearman rank correlation between
  the per-design wall-hit probability and a sealed geometry or field
  descriptor over the 96 designs; an observation of one launch design on
  linear-vacuum fields, never a design rule.
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
  linear-vacuum field model; it is not proof that no such design exists, and
  the literature-definition cusps of the topology screening do not contradict
  it (the frozen definition asked for a wall-side vector null, which the
  literature definition does not place at the wall).
- A separatrix-bounded cell of the topology screening is a geometric property
  of a prescribed field map; it is not a demonstrated plasma confinement cell,
  and its mirror ratios are not cusp-loss probabilities.
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
- The robust front of the catalogue campaign (designs 49, 50, 94 under CL-1) is
  a closure-dependent optimiser estimand at screening tier; it is not a design
  recommendation, not a demonstrated geometry-to-performance map, and the CL-2
  front shares no design with it.
